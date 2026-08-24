"""Unit tests for add_memory write triage (task 3127, PRD leaf beta).

``server/write_triage.py`` is the redirect-not-reject successor to
``server/near_duplicate_guard.py``: instead of returning a soft-block that
loses the submitted content, a restatement is attached as a SIGHTING child of
its canonical, and everything else is stored. Contract C1 is absolute —
triage never loses content, never blocks a write, and never edits a canonical.

Structure mirrors ``test_near_duplicate_guard.py``: pure selectors and
defensive config resolvers tested directly, with POST-RRF ``MemoryResult``
fixtures (cosine in ``metadata['store_score']``, ``relevance_score`` an
ordinal RRF value deliberately unrelated to it) so a regression that reads the
RRF ordinal instead of the cosine fails here rather than silently disabling
triage for every input.
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, Mock

import pytest

from fused_memory.models.enums import MEM0_PRIMARY, MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import write_triage
from fused_memory.server.write_triage import (
    _DEFAULT_CANDIDATE_K,
    _DEFAULT_WRITE_TRIAGE_ENABLED,
    _FAIL_OPEN_STORM_THRESHOLD,
    _FAIL_OPEN_STORM_WINDOW_SECONDS,
    CANONICAL_ID_KEY,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    TRIAGE_OUTCOMES,
    TriageFailOpenCounter,
    _stub_judge,
    decide_band,
    emit_triage_fail_open_storm_escalation,
    resolve_bands,
    resolve_candidate_k,
    resolve_write_triage_enabled,
    retrieve_candidates,
    triage_write,
)
from fused_memory.services.memory_service import RRF_K

# The real post-RRF relevance_score for a rank-1 hit, from production rather
# than restated as the literal 60 — see test_near_duplicate_guard.py.
_RRF_RANK1 = 1.0 / (RRF_K + 1)


def _result(
    id_: str,
    score: float | None,
    *,
    category: MemoryCategory | None = MemoryCategory.procedural_knowledge,
    source_store: SourceStore = SourceStore.mem0,
    content: str = 'some procedural content',
    relevance_score: float = _RRF_RANK1,
    store_rank: int = 1,
    omit_store_score: bool = False,
) -> MemoryResult:
    """Build a POST-RRF ``MemoryResult``: *score* is the COSINE, in metadata.

    ``relevance_score`` defaults to the ordinal RRF value a real rank-1 mem0
    hit carries, deliberately UNRELATED to *score* — so any test that passes
    only because the band router still reads ``relevance_score`` fails.
    """
    metadata: dict = {'store_rank': store_rank}
    if not omit_store_score:
        metadata['store_score'] = score
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=source_store,
        relevance_score=relevance_score,
        metadata=metadata,
    )


def _svc(**write_triage) -> types.SimpleNamespace:
    """A memory_service double whose config leaf is a REAL namespace.

    A plain ``Mock()`` is used deliberately in the negative cases below: an
    unspecced Mock auto-generates every attribute, so ``config.write_triage.
    enabled`` yields a truthy Mock rather than a bool. That is precisely the
    shape the resolvers must refuse.
    """
    return types.SimpleNamespace(
        config=types.SimpleNamespace(write_triage=types.SimpleNamespace(**write_triage)),
    )


# ---------------------------------------------------------------------------
# Ack contract constants (INV-1: one home for the wire names)
# ---------------------------------------------------------------------------

class TestAckContractConstants:
    """Leaf gamma and the boundary tests IMPORT these rather than restating.

    The ack is the only place a caller learns what triage did with its write,
    so its key names are a published contract. Pinning them here means a
    rename has exactly one place to fail, instead of drifting between the
    tool, the judge and the tests that assert on all three.
    """

    def test_the_outcome_set_is_the_four_published_values(self) -> None:
        assert frozenset({
            'stored', 'restated', 'amended', 'contested',
        }) == TRIAGE_OUTCOMES

    def test_every_outcome_constant_is_a_member_of_the_set(self) -> None:
        """The constants and the set cannot drift apart."""
        for constant in (OUTCOME_STORED, OUTCOME_RESTATED,
                         OUTCOME_AMENDED, OUTCOME_CONTESTED):
            assert constant in TRIAGE_OUTCOMES

    def test_the_ack_key_names(self) -> None:
        assert ROUTED_KEY == 'routed'
        assert CANONICAL_ID_KEY == 'canonical_id'


# ---------------------------------------------------------------------------
# Defensive config resolvers
# ---------------------------------------------------------------------------

class TestResolveWriteTriageEnabled:
    """The staged-rollout kill switch, read LIVE off the shared config.

    Defaults OFF on every malformed reading. This is the safe direction and
    the deliberate one: an unreadable config must leave today's behaviour in
    place, never silently enable a path whose judge is still a stub.
    """

    def test_the_module_default_is_off(self) -> None:
        assert _DEFAULT_WRITE_TRIAGE_ENABLED is False

    def test_a_literal_true_enables(self) -> None:
        assert resolve_write_triage_enabled(_svc(enabled=True)) is True

    def test_a_literal_false_disables(self) -> None:
        assert resolve_write_triage_enabled(_svc(enabled=False)) is False

    @pytest.mark.parametrize(
        ('label', 'service'),
        [
            ('no config attribute at all', types.SimpleNamespace()),
            ('config is None', types.SimpleNamespace(config=None)),
            ('no write_triage section', types.SimpleNamespace(config=types.SimpleNamespace())),
            ('write_triage is None',
             types.SimpleNamespace(config=types.SimpleNamespace(write_triage=None))),
            ('no enabled leaf', _svc()),
            ('enabled is None', _svc(enabled=None)),
        ],
    )
    def test_a_missing_hop_reads_as_off(self, label, service) -> None:
        assert resolve_write_triage_enabled(service) is False, label

    @pytest.mark.parametrize('value', [1, 0, 'true', 'yes', 1.0, [], object()])
    def test_a_non_bool_reads_as_off(self, value) -> None:
        """``isinstance(bool)`` only — a truthy 1 is not a kill switch.

        The int 1 is the one that matters: YAML's `enabled: 1` and a test
        double's `enabled = 1` both look enabled to a truthiness check, and
        enabling triage by accident is the failure this refuses.
        """
        assert resolve_write_triage_enabled(_svc(enabled=value)) is False

    def test_an_unspecced_mock_attribute_reads_as_off(self) -> None:
        """An unspecced Mock auto-generates a truthy attribute for anything.

        A test double wired without a real namespace would otherwise turn
        triage ON for every test in the suite that touches add_memory.
        """
        assert resolve_write_triage_enabled(Mock()) is False

    def test_the_flag_is_read_live_not_captured(self) -> None:
        """Green-tier hot-reload is only real if the read happens per call.

        `write_triage.enabled` is allowlisted in RELOADABLE_FIELDS, which
        mutates the shared config object in place. A resolver that captured
        the value at import or construction would leave the registration
        decorative and the kill switch restart-only.
        """
        service = _svc(enabled=False)
        assert resolve_write_triage_enabled(service) is False
        service.config.write_triage.enabled = True
        assert resolve_write_triage_enabled(service) is True


class TestResolveCandidateK:
    """Retrieval width. Falls back to the module default, never to zero."""

    def test_the_module_default_is_wider_than_the_retired_guards_five(self) -> None:
        """Measured same-category recall: 26.1% @5, 43.9% @10, 69.4% @20.

        k is a RANK property that caps what any band threshold can achieve —
        an unretrieved candidate cannot be scored at all — so narrowing this
        back toward the retired near-dup guard's hardcoded ``limit=5`` would
        silently discard three quarters of the duplicates triage exists to
        catch.
        """
        assert _DEFAULT_CANDIDATE_K > 5, (
            f'_DEFAULT_CANDIDATE_K must stay materially wider than the retired '
            f"guard's limit=5 (measured recall 26.1% @5 vs 69.4% @20); "
            f'got {_DEFAULT_CANDIDATE_K}'
        )

    def test_a_configured_int_is_used(self) -> None:
        assert resolve_candidate_k(_svc(candidate_k=37)) == 37

    @pytest.mark.parametrize(
        ('label', 'service'),
        [
            ('no config', types.SimpleNamespace()),
            ('config is None', types.SimpleNamespace(config=None)),
            ('write_triage is None',
             types.SimpleNamespace(config=types.SimpleNamespace(write_triage=None))),
            ('no leaf', _svc()),
            ('leaf is None', _svc(candidate_k=None)),
            ('unspecced mock', Mock()),
        ],
    )
    def test_a_missing_hop_falls_back_to_the_default(self, label, service) -> None:
        assert resolve_candidate_k(service) == _DEFAULT_CANDIDATE_K, label

    @pytest.mark.parametrize('value', ['20', 20.5, [], object(), True, False])
    def test_a_non_int_falls_back_to_the_default(self, value) -> None:
        """``bool`` is excluded despite being an ``int`` subclass.

        ``candidate_k=True`` would otherwise resolve to a retrieval width of
        1 — a single candidate, which is triage with almost no recall and no
        error anywhere to explain it.
        """
        assert resolve_candidate_k(_svc(candidate_k=value)) == _DEFAULT_CANDIDATE_K

    @pytest.mark.parametrize('value', [0, -1, -20])
    def test_a_non_positive_width_falls_back_to_the_default(self, value) -> None:
        """A zero width would read as "no comparable candidate" on every write.

        The schema bounds this ``ge=1``, so a 0 can only arrive from a
        hand-built config object or a partially-applied reload — and the
        resolver refuses it rather than letting triage become a silent no-op.
        """
        assert resolve_candidate_k(_svc(candidate_k=value)) == _DEFAULT_CANDIDATE_K


class TestResolveBands:
    """``(t_high, t_low)`` as floats-or-None, never a Mock.

    None is a FIRST-CLASS reading here, not an error: the landed schema uses
    it to mean UNCALIBRATED, and leaf alpha measured a corpus on which no
    deterministic band exists at all (the unrelated-pair max 0.8672 sits
    ABOVE the true-pair max 0.8532). Both must survive the resolver.
    """

    def test_configured_floats_are_returned_in_order(self) -> None:
        assert resolve_bands(_svc(t_high=0.88, t_low=0.52)) == (0.88, 0.52)

    def test_an_int_is_coerced_to_float(self) -> None:
        t_high, t_low = resolve_bands(_svc(t_high=1, t_low=0))
        assert (t_high, t_low) == (1.0, 0.0)
        assert isinstance(t_high, float) and isinstance(t_low, float)

    def test_an_uncalibrated_pair_reads_as_none(self) -> None:
        assert resolve_bands(_svc(t_high=None, t_low=None)) == (None, None)

    def test_an_empty_deterministic_band_is_preserved(self) -> None:
        """t_high=None with a real t_low is a MEASURED configuration.

        Leaf alpha found the distributions do not separate on this corpus, so
        `calibrate_write_triage.py` derives no t_high. The resolver must hand
        that through unchanged so the router can route everything at or above
        t_low to the judge, rather than treating it as a broken config.
        """
        assert resolve_bands(_svc(t_high=None, t_low=0.52)) == (None, 0.52)

    @pytest.mark.parametrize(
        ('label', 'service'),
        [
            ('no config', types.SimpleNamespace()),
            ('config is None', types.SimpleNamespace(config=None)),
            ('write_triage is None',
             types.SimpleNamespace(config=types.SimpleNamespace(write_triage=None))),
            ('no leaves', _svc()),
            ('unspecced mock', Mock()),
        ],
    )
    def test_a_missing_hop_reads_as_uncalibrated(self, label, service) -> None:
        assert resolve_bands(service) == (None, None), label

    @pytest.mark.parametrize('value', ['0.9', [], object(), True, False])
    def test_a_non_numeric_reads_as_uncalibrated(self, value) -> None:
        """``bool`` excluded despite being an ``int`` subclass.

        ``t_high=True`` would coerce to a cutoff of 1.0 and ``t_high=False``
        to 0.0 — the first silently empties the deterministic band and the
        second makes every candidate a restatement. Neither is a measurement.
        """
        assert resolve_bands(_svc(t_high=value, t_low=value)) == (None, None)

    def test_the_bands_are_read_live_not_captured(self) -> None:
        """Same green-tier reload requirement as the flag: a re-calibration
        must take effect on a running server without a restart."""
        service = _svc(t_high=0.88, t_low=0.52)
        assert resolve_bands(service) == (0.88, 0.52)
        service.config.write_triage.t_high = 0.91
        assert resolve_bands(service) == (0.91, 0.52)


# ---------------------------------------------------------------------------
# Pure band routing
# ---------------------------------------------------------------------------

# Deliberately NOT the shipped calibrated numbers: these are test inputs, and
# copying the live values in would make every assertion here re-break on the
# next recalibration for a reason that has nothing to do with the routing
# logic under test.
T_HIGH = 0.90
T_LOW = 0.60


class TestDecideBand:
    """Which band a candidate set falls in, decided on the MAX COSINE.

    Pure and synchronous: no I/O, and nothing raises on empty input.
    """

    def test_the_judge_sentinel_is_not_an_ack_outcome(self) -> None:
        """`judge` is INTERNAL routing, never a value a caller sees.

        The ack's `routed` field carries what triage DID with the write.
        "sent it to a judge" is not an answer to that — the judge's own
        verdict is. Leaking the sentinel into TRIAGE_OUTCOMES would make the
        closed set open to a value no caller can act on.
        """
        assert OUTCOME_JUDGE not in TRIAGE_OUTCOMES

    def test_an_empty_candidate_list_stores(self) -> None:
        d = decide_band([], t_high=T_HIGH, t_low=T_LOW)
        assert d.outcome == OUTCOME_STORED
        assert d.canonical_id is None

    def test_below_t_low_stores(self) -> None:
        d = decide_band([_result('m1', 0.10)], t_high=T_HIGH, t_low=T_LOW)
        assert d.outcome == OUTCOME_STORED
        assert d.canonical_id is None

    def test_at_or_above_t_high_is_deterministically_restated(self) -> None:
        d = decide_band([_result('m1', 0.97)], t_high=T_HIGH, t_low=T_LOW)
        assert d.outcome == OUTCOME_RESTATED
        assert d.canonical_id == 'm1'
        assert d.similarity == pytest.approx(0.97)

    def test_the_middle_band_routes_to_the_judge(self) -> None:
        d = decide_band([_result('m1', 0.75)], t_high=T_HIGH, t_low=T_LOW)
        assert d.outcome == OUTCOME_JUDGE
        assert d.canonical_id == 'm1'

    @pytest.mark.parametrize(
        ('score', 'expected'),
        [(T_HIGH, OUTCOME_RESTATED), (T_LOW, OUTCOME_JUDGE)],
        ids=['exactly-t_high', 'exactly-t_low'],
    )
    def test_both_boundaries_are_inclusive(self, score, expected) -> None:
        """`s >= t_high` and `t_low <= s`, exactly as the PRD spells them.

        An off-by-one here is invisible in aggregate and wrong on exactly the
        cases the calibration fitted the edges to — both bounds are order
        statistics of measured pairs, so the boundary value IS an observed
        duplicate.
        """
        assert decide_band(
            [_result('m1', score)], t_high=T_HIGH, t_low=T_LOW,
        ).outcome == expected

    def test_the_winner_is_the_max_cosine_not_the_first_result(self) -> None:
        d = decide_band(
            [_result('m1', 0.62), _result('m2', 0.95), _result('m3', 0.70)],
            t_high=T_HIGH, t_low=T_LOW,
        )
        assert d.canonical_id == 'm2'
        assert d.outcome == OUTCOME_RESTATED

    def test_the_winner_is_not_the_max_relevance_score(self) -> None:
        """The RRF rank-1 hit deliberately carries the LOWEST cosine here.

        Since task 3658 `relevance_score` is an ORDINAL RRF value — rank-1 is
        1/(RRF_K+1) ~ 0.0164 — so reading it as a similarity would never clear
        a cosine band and would silently disable triage for every input. A
        band router that sorted by it would also pick the wrong canonical
        even when it did fire.
        """
        results = [
            # rank 1 by RRF, worst by cosine.
            _result('rank1', 0.61, relevance_score=1.0 / (RRF_K + 1), store_rank=1),
            _result('rank2', 0.96, relevance_score=1.0 / (RRF_K + 2), store_rank=2),
        ]
        d = decide_band(results, t_high=T_HIGH, t_low=T_LOW)
        assert d.canonical_id == 'rank2', 'the max COSINE must win, not the top RRF rank'
        assert d.outcome == OUTCOME_RESTATED

    @pytest.mark.parametrize(
        ('label', 'result'),
        [
            ('store_score key absent', _result('m1', None, omit_store_score=True)),
            ('store_score is None', _result('m1', None)),
            ('store_score is a bool', _result('m1', True)),  # type: ignore[arg-type]
            ('store_score is a string', _result('m1', '0.99')),  # type: ignore[arg-type]
        ],
    )
    @pytest.mark.parametrize('t_low', [0.0, T_LOW])
    def test_an_uncomparable_candidate_never_qualifies_at_any_threshold(
        self, label, result, t_low,
    ) -> None:
        """A missing cosine means NOT COMPARABLE, not a similarity of 0.0.

        The t_low=0.0 leg is the one that matters: a router that coerced an
        absent score to 0.0 would have it clear a zero floor and route a
        Graphiti result (which carries no store_score at all) to the judge.
        """
        d = decide_band([result], t_high=T_HIGH, t_low=t_low)
        assert d.outcome == OUTCOME_STORED, label
        assert d.canonical_id is None

    def test_a_candidate_in_a_different_mem0_category_still_qualifies(self) -> None:
        """The cross-category blind spot this leaf exists to fix.

        The retired guard filtered candidates to the WRITE's own category, so
        a procedural_knowledge write could not be matched against an
        observations_and_summaries duplicate of itself. Reify esc-5547 and
        esc-5560 both had exactly that shape.
        """
        d = decide_band(
            [_result('m1', 0.97, category=MemoryCategory.observations_and_summaries)],
            t_high=T_HIGH, t_low=T_LOW,
        )
        assert d.outcome == OUTCOME_RESTATED
        assert d.canonical_id == 'm1'

    def test_a_topic_anchored_pin_never_wins_over_a_scored_candidate(self) -> None:
        """A pinned canonical is a SPENT slot, never a routing target.

        Topic anchoring (task 3111) promotes a cluster's canonical to the head
        of a search window without widening it, and deliberately stamps NO
        `metadata['store_score']` on the pin (`services/topic_anchor.py`) —
        a missing cosine means NOT COMPARABLE, so the pin can never qualify at
        any threshold. `retrieve_candidates` therefore passes
        `anchor_topics=False`, but this asserts the OTHER half of the
        contract: if that opt-out is ever reverted, a pin arriving at the head
        of the candidate set must still not change where the write routes.

        Otherwise the failure would be invisible here and only show up as
        quietly-worse recall in production — the eviction is a routing change,
        so it is asserted as one.
        """
        results = [
            # Pin shape: promoted to the head of the window, no store_score.
            _result('pinned-canonical', None, omit_store_score=True),
            _result('m1', 0.62),
            _result('m2', 0.97),
        ]
        d = decide_band(results, t_high=T_HIGH, t_low=T_LOW)
        assert d.canonical_id == 'm2', 'the max SCORED cosine must win, never the pin'
        assert d.outcome == OUTCOME_RESTATED
        assert d.similarity == pytest.approx(0.97)

    def test_the_decision_quotes_the_numbers_that_produced_it(self) -> None:
        """Inspectable, so the ack can say WHY without recomputing anything."""
        d = decide_band([_result('m1', 0.97)], t_high=T_HIGH, t_low=T_LOW)
        assert (d.similarity, d.t_high, d.t_low) == (pytest.approx(0.97), T_HIGH, T_LOW)

    def test_the_decision_is_frozen(self) -> None:
        """A downstream stage must not be able to rewrite the routing."""
        d = decide_band([_result('m1', 0.97)], t_high=T_HIGH, t_low=T_LOW)
        with pytest.raises((AttributeError, TypeError)):
            d.outcome = OUTCOME_STORED  # type: ignore[misc]

    # -- boundary-of-configuration cases ------------------------------------

    def test_an_empty_deterministic_band_routes_everything_to_the_judge(self) -> None:
        """`t_high is None` is a FIRST-CLASS configuration, not a broken one.

        Leaf alpha measured the esc-3181 cluster and found the distributions
        do not separate: the unrelated-pair MAX (0.8672) sits ABOVE the
        true-pair MAX (0.8532). `calibrate_write_triage.py` derives t_high as
        "the smallest measured duplicate score that strictly exceeds every
        measured negative", an objective only satisfiable when they DO
        separate — so on such a corpus there is honestly no deterministic
        band, and refusing to invent one is the correct outcome.

        A future reader must not "fix" this into an assertion that a
        deterministic band always exists: with t_high None, everything at or
        above t_low goes to the judge and NOTHING takes the autonomous
        restated path.
        """
        for score in (T_LOW, 0.75, 0.97, 1.0):
            d = decide_band([_result('m1', score)], t_high=None, t_low=T_LOW)
            assert d.outcome == OUTCOME_JUDGE, f'score {score} must reach the judge'
            assert d.canonical_id == 'm1'

    def test_an_empty_deterministic_band_still_stores_below_t_low(self) -> None:
        """The lower edge keeps working when the upper one is absent."""
        d = decide_band([_result('m1', 0.10)], t_high=None, t_low=T_LOW)
        assert d.outcome == OUTCOME_STORED

    @pytest.mark.parametrize('t_high', [None, T_HIGH])
    @pytest.mark.parametrize('score', [0.0, 0.5, 0.97, 1.0])
    def test_an_uncalibrated_t_low_stores_everything(self, t_high, score) -> None:
        """`t_low is None` is UNCALIBRATED — fail open to `stored`.

        Matches the landed schema's own reading of None, and it is the safe
        direction: with no measured lower edge there is no evidence that any
        candidate is a duplicate, so inventing a floor would attach real
        writes to unrelated canonicals. Holds whether or not t_high is set —
        an upper edge without a lower one does not license routing either.
        """
        d = decide_band([_result('m1', score)], t_high=t_high, t_low=None)
        assert d.outcome == OUTCOME_STORED
        assert d.canonical_id is None

    def test_decide_band_raises_nothing_on_any_band_configuration(self) -> None:
        """Pure and synchronous, like find_near_duplicate_memory before it.

        Every combination of present/absent band edges over an empty
        candidate list returns a decision rather than raising. The
        uncalibrated and empty-deterministic-band configurations are the ones
        a naive `if score >= t_high` would blow up on with a TypeError —
        inside a write path that C1 forbids from erroring.

        `decide_band` takes no service and no config: it is handed the two
        edges as arguments precisely so it cannot acquire an I/O dependency.
        """
        for t_high, t_low in ((None, None), (T_HIGH, T_LOW), (None, T_LOW)):
            assert decide_band([], t_high=t_high, t_low=t_low).outcome == OUTCOME_STORED


# ---------------------------------------------------------------------------
# Candidate retrieval (INV-5: one seam, no second implementation)
# ---------------------------------------------------------------------------

class TestRetrieveCandidates:
    """Exactly ONE ``MemoryService.search`` call, cross-category, mem0-only."""

    @staticmethod
    def _service(results=None, **write_triage) -> types.SimpleNamespace:
        service = _svc(**write_triage)
        service.search = AsyncMock(return_value=results if results is not None else [])
        return service

    @pytest.mark.asyncio
    async def test_issues_exactly_one_search_and_touches_nothing_else(self) -> None:
        """The INV-5 pin: this leaf builds no second retrieval.

        Task 3111 lands topic-anchored recall AT the `MemoryService.search`
        seam. Calling that seam once means the improvement arrives here for
        free; a topic-aware retrieval built inside `write_triage.py` would be
        a second implementation to keep in sync forever. A second call — to a
        store, an embedder, or search again with different arguments — is the
        regression this catches.
        """
        service = self._service()
        service.update_memory = AsyncMock()
        service.delete_memory = AsyncMock()
        service.add_memory = AsyncMock()

        await retrieve_candidates(service, 'some content', 'dark_factory', 20)

        assert service.search.await_count == 1
        # C1: triage never edits a canonical, on any path including this one.
        service.update_memory.assert_not_called()
        service.delete_memory.assert_not_called()
        service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_search_kwargs_are_the_published_shape(self) -> None:
        service = self._service()
        await retrieve_candidates(service, 'some content', 'dark_factory', 20)

        kwargs = service.search.await_args.kwargs
        assert kwargs['query'] == 'some content'
        assert kwargs['project_id'] == 'dark_factory'
        assert kwargs['stores'] == ['mem0']
        assert kwargs['limit'] == 20
        assert kwargs['anchor_topics'] is False

    @pytest.mark.asyncio
    async def test_topic_anchored_recall_is_opted_out_of(self) -> None:
        """`anchor_topics=False` is a CORRECTNESS pin, not a style preference.

        Topic anchoring (task 3111) is a PROMOTION, not an addition: the
        returned window stays exactly `limit` long, so every pinned canonical
        EVICTS the lowest-ranked genuine hit. `MemoryService.search`'s own
        docstring names this caller class and mandates the opt-out — a
        consumer that thresholds and post-filters its window must pass False
        or silently lose genuine candidates to displacement.

        The eviction is not merely a wash here, it is a pure loss: a pinned
        canonical deliberately carries NO `metadata['store_score']`
        (`services/topic_anchor.py`), and `decide_band` drops every candidate
        whose cosine is non-numeric — so a pin can never qualify at ANY
        threshold. Each pin is therefore a candidate slot spent on a record
        triage is structurally required to ignore, on exactly the consolidated
        topics where pins exist.

        Three landed sibling consumers already opt out for the same reason,
        all of them candidate-set/idempotency consumers exactly like this one:
        `reconciliation/mem0_dedup.py`, `reconciliation/stages/
        task_knowledge_sync.py`, and the retired near-duplicate write guard in
        `server/tools.py`, whose call-site comment explains this hazard
        verbatim.

        The default is True, so the omission is silent and degrades in the
        WRONG direction — nothing errors, the window is merely quietly worse.
        That is why it is pinned explicitly rather than left to the default.
        """
        service = self._service()
        await retrieve_candidates(service, 'c', 'p', 20)

        assert service.search.await_args.kwargs['anchor_topics'] is False

    @pytest.mark.asyncio
    async def test_the_categories_are_every_mem0_primary_category(self) -> None:
        """Asserted against the imported frozenset, never a hand-written triple.

        Cross-category retrieval IS the fix this leaf exists for — the retired
        guard filtered to the write's own category and could not see the
        cross-category duplicates in reify esc-5547/esc-5560. Binding the
        assertion to `MEM0_PRIMARY` means a future category addition carries
        this test with it instead of leaving a stale literal that still passes
        while the new category goes untriaged.
        """
        service = self._service()
        await retrieve_candidates(service, 'c', 'p', 20)

        assert service.search.await_args.kwargs['categories'] == sorted(
            c.value for c in MEM0_PRIMARY
        )

    @pytest.mark.asyncio
    async def test_the_resolved_width_reaches_limit(self) -> None:
        """A distinct sentinel, so a hardcoded default cannot pass by luck."""
        service = self._service()
        await retrieve_candidates(service, 'c', 'p', 37)
        assert service.search.await_args.kwargs['limit'] == 37

    @pytest.mark.asyncio
    async def test_the_shipped_default_width_is_wider_than_the_retired_guards_five(
        self,
    ) -> None:
        """The retired guard's `limit=5` must not be inherited by analogy.

        Measured same-category recall: 26.1% @5 vs 69.4% @20. Retrieval width
        caps what any band threshold can achieve, so narrowing back to 5 would
        discard three quarters of the duplicates triage exists to catch,
        silently.
        """
        service = self._service()
        await retrieve_candidates(
            service, 'c', 'p', resolve_candidate_k(self._service()),
        )
        assert service.search.await_args.kwargs['limit'] > 5

    @pytest.mark.asyncio
    async def test_the_results_are_returned_unchanged(self) -> None:
        """No filtering, no re-ranking: banding is `decide_band`'s job."""
        results = [_result('m1', 0.9), _result('m2', 0.5)]
        service = self._service(results=results)
        assert await retrieve_candidates(service, 'c', 'p', 20) == results

    @pytest.mark.asyncio
    async def test_a_raising_search_propagates(self) -> None:
        """Fail-open is the ORCHESTRATOR's job, not this helper's.

        Swallowing here would make the seam dishonest: a wiring bug (a renamed
        kwarg, a changed signature) would look like "no candidates found" and
        route every write to `stored` with nothing to distinguish it from a
        genuinely novel corpus. `triage_write` catches this and counts it as a
        fail-open, which is what makes the degradation visible.
        """
        service = self._service()
        service.search = AsyncMock(side_effect=RuntimeError('mem0 down'))
        with pytest.raises(RuntimeError):
            await retrieve_candidates(service, 'c', 'p', 20)


# ---------------------------------------------------------------------------
# Fail-open + storm counter (INV-4)
# ---------------------------------------------------------------------------

class _Clock:
    """An injected clock: the window is exercised by advancing it, never by
    sleeping."""

    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class TestTriageFailOpenCounter:
    """A burst of fail-opens means triage is DEGRADED, not that one write went wrong.

    One fail-open is routine (a transient mem0 blip). A burst inside the window
    means every write in that window silently fell back to pre-triage
    behaviour — exactly the silent degradation INV-4 exists to make visible.
    """

    def test_below_the_threshold_returns_none(self) -> None:
        counter = TriageFailOpenCounter(time_provider=_Clock())
        for _ in range(_FAIL_OPEN_STORM_THRESHOLD - 1):
            assert counter.record(project='dark_factory') is None

    def test_the_crossing_call_returns_a_json_serializable_summary(self) -> None:
        import json  # noqa: PLC0415

        counter = TriageFailOpenCounter(time_provider=_Clock())
        summary = None
        for _ in range(_FAIL_OPEN_STORM_THRESHOLD):
            summary = counter.record(project='dark_factory')

        assert summary is not None, 'crossing the threshold must fire'
        assert summary['count'] >= _FAIL_OPEN_STORM_THRESHOLD
        assert summary['threshold'] == _FAIL_OPEN_STORM_THRESHOLD
        assert summary['window_seconds'] == _FAIL_OPEN_STORM_WINDOW_SECONDS
        assert summary['hint']
        assert summary['projects'] == ['dark_factory']
        # It goes into an escalation detail, so it must survive a round-trip.
        json.loads(json.dumps(summary))

    def test_a_burst_is_attributed_to_every_project_in_the_window(self) -> None:
        """`projects`, not "whichever write happened to cross the threshold".

        An operator reading the escalation needs to know whether one project
        is broken or the whole server is. Naming only the crossing write's
        project would answer that wrong half the time.
        """
        counter = TriageFailOpenCounter(time_provider=_Clock())
        summary = None
        for i in range(_FAIL_OPEN_STORM_THRESHOLD):
            summary = counter.record(project=f'p{i % 2}')
        assert summary is not None
        assert summary['projects'] == ['p0', 'p1']

    def test_a_second_crossing_in_the_same_window_is_rate_limited(self) -> None:
        """A runaway must escalate once per window, not once per write."""
        counter = TriageFailOpenCounter(time_provider=_Clock())
        for _ in range(_FAIL_OPEN_STORM_THRESHOLD):
            counter.record(project='p')
        for _ in range(_FAIL_OPEN_STORM_THRESHOLD):
            assert counter.record(project='p') is None

    def test_the_window_ages_events_out(self) -> None:
        clock = _Clock()
        counter = TriageFailOpenCounter(time_provider=clock)
        for _ in range(_FAIL_OPEN_STORM_THRESHOLD - 1):
            counter.record(project='p')
        clock.advance(_FAIL_OPEN_STORM_WINDOW_SECONDS + 1)
        # The window is now empty, so one more event is nowhere near a burst.
        assert counter.record(project='p') is None

    def test_state_is_per_instance(self) -> None:
        """No module global: no bleed between servers, or between tests."""
        a = TriageFailOpenCounter(time_provider=_Clock())
        b = TriageFailOpenCounter(time_provider=_Clock())
        for _ in range(_FAIL_OPEN_STORM_THRESHOLD - 1):
            a.record(project='p')
        assert b.record(project='p') is None

    def test_an_unresolved_project_still_counts(self) -> None:
        """A write whose project could not be resolved is still a fail-open."""
        counter = TriageFailOpenCounter(time_provider=_Clock())
        summary = None
        for _ in range(_FAIL_OPEN_STORM_THRESHOLD):
            summary = counter.record(project=None)
        assert summary is not None
        assert summary['count'] >= _FAIL_OPEN_STORM_THRESHOLD
        assert summary['projects'] == []


class TestTriageWriteFailsOpen:
    """C1 is ABSOLUTE: never an error, never a blocked write, always a decision."""

    @staticmethod
    def _service(search=None, **write_triage) -> types.SimpleNamespace:
        service = _svc(**{'enabled': True, 't_high': T_HIGH, 't_low': T_LOW,
                          'candidate_k': 20, **write_triage})
        service.search = search or AsyncMock(return_value=[])
        return service

    @pytest.mark.asyncio
    async def test_a_raising_search_stores_and_counts_once(self) -> None:
        counter = TriageFailOpenCounter(time_provider=_Clock())
        service = self._service(search=AsyncMock(side_effect=RuntimeError('mem0 down')))

        decision = await triage_write(
            service, content='c', project_id='p', counter=counter,
        )

        assert decision.outcome == OUTCOME_STORED, 'the write is never blocked'
        assert counter.live_count() == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('exc', [TypeError, AttributeError, NameError])
    async def test_a_wiring_bug_class_also_fails_open_and_counts(self, exc) -> None:
        """The one place this deliberately diverges from the retired guard.

        `tools.py`'s near-dup call site RE-RAISES TypeError/AttributeError/
        NameError so a signature change surfaces loudly instead of being
        swallowed. C1 forbids that here — an errored write is a blocked write.
        The loudness is preserved differently: these are logged at ERROR naming
        the exception type AND counted, so a changed `MemoryService.search`
        signature surfaces as a storm escalation rather than as a stream of
        errored writes.
        """
        counter = TriageFailOpenCounter(time_provider=_Clock())
        service = self._service(search=AsyncMock(side_effect=exc('boom')))

        decision = await triage_write(
            service, content='c', project_id='p', counter=counter,
        )

        assert decision.outcome == OUTCOME_STORED
        assert counter.live_count() == 1

    @pytest.mark.asyncio
    async def test_a_raising_judge_stores_and_counts_once(self) -> None:
        counter = TriageFailOpenCounter(time_provider=_Clock())
        service = self._service(search=AsyncMock(return_value=[_result('m1', 0.75)]))

        async def _boom(**_kwargs):
            raise RuntimeError('judge down')

        decision = await triage_write(
            service, content='c', project_id='p', counter=counter, judge=_boom,
        )

        assert decision.outcome == OUTCOME_STORED
        assert counter.live_count() == 1

    @pytest.mark.asyncio
    async def test_the_deliberate_stub_judge_counts_zero(self) -> None:
        """A STUB IS NOT AN OUTAGE — the other side of the C1 boundary.

        Leaf gamma replaces `_stub_judge`; until then every middle-band write
        is answered `stored` by design. If that counted as a fail-open, the
        first `_FAIL_OPEN_STORM_THRESHOLD` middle-band writes after the flag
        flip would guarantee a storm escalation describing an outage that is
        not happening — which trains an operator to ignore the alarm that
        exists to catch a real one.
        """
        counter = TriageFailOpenCounter(time_provider=_Clock())
        service = self._service(search=AsyncMock(return_value=[_result('m1', 0.75)]))

        decision = await triage_write(
            service, content='c', project_id='p', counter=counter,
        )

        assert decision.outcome == OUTCOME_STORED
        assert counter.live_count() == 0, 'a deliberate stub is not a fail-open'

    @pytest.mark.asyncio
    async def test_a_clean_restated_counts_zero(self) -> None:
        counter = TriageFailOpenCounter(time_provider=_Clock())
        service = self._service(search=AsyncMock(return_value=[_result('m1', 0.97)]))

        decision = await triage_write(
            service, content='c', project_id='p', counter=counter,
        )

        assert decision.outcome == OUTCOME_RESTATED
        assert decision.canonical_id == 'm1'
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_a_clean_stored_counts_zero(self) -> None:
        counter = TriageFailOpenCounter(time_provider=_Clock())
        service = self._service(search=AsyncMock(return_value=[_result('m1', 0.10)]))

        decision = await triage_write(
            service, content='c', project_id='p', counter=counter,
        )

        assert decision.outcome == OUTCOME_STORED
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('label', 'broken'),
        [
            ('search raises', {'search': AsyncMock(side_effect=RuntimeError('x'))}),
            ('search raises TypeError', {'search': AsyncMock(side_effect=TypeError('x'))}),
            ('search returns a string', {'search': AsyncMock(return_value='not a list')}),
            ('search returns junk', {'search': AsyncMock(return_value=[object()])}),
            ('search is not async', {'search': Mock(return_value=[])}),
        ],
    )
    async def test_triage_write_never_raises_for_any_injected_failure(
        self, label, broken,
    ) -> None:
        """Always a decision — never an exception, never an error dict.

        The last three legs are the ones a narrower try/except would miss: a
        `search` returning the wrong SHAPE fails inside band routing rather
        than at the call, and a non-async `search` fails at the `await` itself.
        """
        counter = TriageFailOpenCounter(time_provider=_Clock())
        service = self._service(**broken)

        decision = await triage_write(
            service, content='c', project_id='p', counter=counter,
        )

        assert decision.outcome in TRIAGE_OUTCOMES, label
        assert decision.outcome == OUTCOME_STORED, label

    @pytest.mark.asyncio
    async def test_the_stub_judge_answers_stored(self) -> None:
        """Named as leaf gamma's replacement point, and it answers `stored`."""
        assert await _stub_judge(
            memory_service=None, content='c', project_id='p', decision=None,
        ) == OUTCOME_STORED


# ---------------------------------------------------------------------------
# The storm escalation (INV-4): the alarm that makes a fail-open burst visible
# ---------------------------------------------------------------------------

#: A storm summary in the exact shape `TriageFailOpenCounter.record` returns.
_STORM: dict = {
    'count': 12,
    'threshold': _FAIL_OPEN_STORM_THRESHOLD,
    'window_seconds': _FAIL_OPEN_STORM_WINDOW_SECONDS,
    'projects': ['/project-a', '/project-b'],
    'hint': 'add_memory write triage failed open repeatedly',
}


class TestEmitTriageFailOpenStormEscalation:
    """The best-effort queue write on a fail-open burst (INV-4).

    Purely ADDITIVE, exactly like `markup_tripwire.emit_markup_storm_escalation`
    (whose coverage in `test_markup_tripwire.py::TestEmitMarkupStormEscalation`
    this mirrors): by the time this runs the write's outcome is already decided
    and the write has already been stored, so every failure mode here must
    degrade to `None` rather than change it. Exercised against a real
    `EscalationQueue` in `tmp_path` — the escalation package is a fused-memory
    workspace dep.
    """

    def test_returns_none_for_a_none_project_root(self) -> None:
        """`add_memory` takes a project_id, not a project_root.

        An unknown project resolves to no root at all, so this must be a quiet
        no-op rather than a crash on `Path(None)` — a triage fail-open that
        also raised while trying to report itself would be the exact
        write-blocking failure C1 forbids.
        """
        assert emit_triage_fail_open_storm_escalation(None, _STORM) is None

    def test_returns_none_when_the_escalation_package_is_unavailable(
        self, tmp_path, monkeypatch,
    ) -> None:
        """The defensive-import no-op path (minimal envs without escalation)."""
        monkeypatch.setattr(write_triage, 'HAS_ESCALATION', False)
        assert emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM) is None
        assert not (tmp_path / 'data' / 'escalations').exists()

    def test_files_one_record_naming_the_leaf_the_numbers_and_the_kill_switch(
        self, tmp_path,
    ) -> None:
        """The record must be actionable without opening the code.

        An operator reading it has to be able to tell a real degradation (the
        judge or mem0 retrieval is down, so every write in the window was
        stored WITHOUT triage) apart from a misfiring tripwire, and must be
        pointed at the ONE lever that stops it deliberately rather than
        guessing at a nearby switch.

        The numbers are asserted as their EXACT labelled substrings: a bare
        `'12' in detail` would also be satisfied by a digit in the interpolated
        tmp_path (`/tmp/pytest-of-leo/pytest-124/...`), i.e. it would pass with
        the count dropped entirely.
        """
        esc_id = emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)
        if not write_triage.HAS_ESCALATION:
            assert esc_id is None
            return

        assert isinstance(esc_id, str)
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'expected exactly one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())

        assert payload['id'] == esc_id
        assert payload['category'] == 'write_triage_fail_open_storm'
        assert payload['level'] == 1
        assert payload['task_id'] == write_triage._ANCHOR_TASK_ID

        detail = payload['detail']
        routing_text = f'{payload["summary"]}\n{detail}\n{payload["suggested_action"]}'
        assert 'fail_opens_in_window=12' in detail, f'must state the count: {detail!r}'
        assert f'threshold={_FAIL_OPEN_STORM_THRESHOLD!r}' in detail, (
            f'must state the threshold that fired: {detail!r}'
        )
        assert f'window_seconds={_FAIL_OPEN_STORM_WINDOW_SECONDS!r}' in detail, (
            f'must state the window: {detail!r}'
        )
        assert "projects_in_window=['/project-a', '/project-b']" in detail, (
            f'must name every project the burst spanned: {detail!r}'
        )
        assert 'write_triage.enabled' in routing_text, (
            f'must name the one deliberate off switch: {payload!r}'
        )
        assert 'memory-write-path-convergence' in routing_text, (
            f'must route the burst at the owning PRD leaf: {payload!r}'
        )

    def test_the_detail_says_triage_is_degraded_not_that_writes_were_lost(
        self, tmp_path,
    ) -> None:
        """A burst is a DEGRADATION, not a data-loss incident.

        Every write in the window still landed — with today's pre-triage
        behaviour — so the record must say so plainly. An operator who reads a
        fail-open storm as "writes were dropped" reaches for the wrong
        remediation, and one who reads it as harmless never fixes retrieval:
        silent degradation is precisely what INV-4 exists to prevent.
        """
        if not write_triage.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)
        payload = json.loads(
            next((tmp_path / 'data' / 'escalations').glob('esc-*.json')).read_text()
        )
        text = f'{payload["summary"]}\n{payload["detail"]}'.lower()

        assert 'without triage' in text or 'pre-triage' in text, (
            f'must say the writes landed untriaged: {payload!r}'
        )
        assert 'no write was lost' in text or 'never blocked' in text, (
            f'must say no write was lost or blocked (contract C1): {payload!r}'
        )

    def test_the_escalation_id_is_greppable_via_the_stable_anchor(
        self, tmp_path,
    ) -> None:
        """The anchor is stable so the ids form one greppable series."""
        esc_id = emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)
        if not write_triage.HAS_ESCALATION:
            return
        assert esc_id is not None
        assert write_triage._ANCHOR_TASK_ID in esc_id, f'unexpected id shape: {esc_id!r}'

    def test_the_queue_is_opened_under_project_root_data_escalations(
        self, tmp_path,
    ) -> None:
        """The queue location is the project's own, never a global default."""
        if not write_triage.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)

        assert list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))

    def test_dedupes_against_an_already_open_escalation(self, tmp_path) -> None:
        """A sustained outage must not mint one escalation per window forever.

        The counter is already rate-limited per window, but retrieval down for
        hours would still file an escalation every hour; the anchor dedup
        collapses those into the one open record until it is resolved.
        """
        first = emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)
        second = emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)
        if not write_triage.HAS_ESCALATION:
            assert first is None and second is None
            return

        assert first is not None
        assert second == first, f'expected dedup; got first={first!r} second={second!r}'
        assert len(list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))) == 1

    def test_files_afresh_once_the_prior_escalation_is_resolved(self, tmp_path) -> None:
        """Dedup must not silence a NEW outage after the old one was cleared."""
        if not write_triage.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')
        from escalation.queue import EscalationQueue  # noqa: PLC0415

        first = emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)
        assert first is not None
        EscalationQueue(tmp_path / 'data' / 'escalations').resolve(first, 'mem0 back')

        second = emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)
        assert second is not None
        assert second != first

    def test_a_dedup_read_failure_still_files(self, tmp_path, monkeypatch) -> None:
        """If the dedup check itself fails, fall THROUGH and file.

        Best-effort dedup: losing duplicate-suppression is strictly better than
        losing the alarm for an outage that is actively happening.
        """
        if not write_triage.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        def _boom(self, task_id, status=None):
            raise OSError('cannot read queue')

        monkeypatch.setattr(write_triage.EscalationQueue, 'get_by_task', _boom)
        assert emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM) is not None

    def test_a_submit_failure_returns_none_rather_than_propagating(
        self, tmp_path, monkeypatch,
    ) -> None:
        """The function must NEVER raise.

        It is called from a write path whose outcome is already decided and
        whose write has already been stored; an exception escaping here would
        convert a successfully-degraded write into a failed one, which is the
        C1 violation the whole fail-open apparatus exists to prevent.
        """
        if not write_triage.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        def _boom(self, esc):
            raise OSError('disk on fire')

        monkeypatch.setattr(write_triage.EscalationQueue, 'submit', _boom)
        assert emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM) is None

    def test_a_queue_open_failure_returns_none(self, tmp_path, monkeypatch) -> None:
        """Even constructing the queue is wrapped — same never-raise contract."""
        if not write_triage.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        def _boom(*args, **kwargs):
            raise OSError('no such directory')

        monkeypatch.setattr(write_triage, 'EscalationQueue', _boom)
        assert emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM) is None

    def test_tolerates_a_storm_dict_missing_keys(self, tmp_path) -> None:
        """A degenerate storm shape still files a well-formed, routable record.

        "Never raises" is established by the call returning at all; what this
        pins is that the record stays USABLE — one whose category or anchor
        degraded along with its missing numbers could not be routed or deduped,
        which is exactly when an operator needs it most.
        """
        if not write_triage.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        esc_id = emit_triage_fail_open_storm_escalation(str(tmp_path), {})

        assert esc_id is not None
        payload = json.loads(
            next((tmp_path / 'data' / 'escalations').glob('esc-*.json')).read_text()
        )
        assert payload['category'] == 'write_triage_fail_open_storm'
        assert payload['task_id'] == write_triage._ANCHOR_TASK_ID
        assert payload['level'] == 1

    def test_the_anchor_is_this_leafs_own_and_shared_with_nobody(self) -> None:
        """A SQUATTED anchor is suppressed indefinitely, and reads as calm.

        Measured incident: the L1 escalation watcher files its own cluster
        records under the `markup-tripwire` anchor and SQUATS it — the tripwire
        filed nothing 2026-08-16..2026-08-19 while 41 rejections occurred, all
        17 records sitting at dedupe_count 0. A filer that dedupes against an
        anchor somebody else keeps open never files again, and the resulting
        silence is indistinguishable from health.

        That incident is why `emit_markup_storm_escalation` grew its
        `anchor_task_id` parameter (see its docstring), and it is why this leaf
        must never share an anchor with any other filer. Asserted against the
        siblings' constants IMPORTED FROM THEIR OWN HOMES, so a future rename
        that collides is caught here rather than in production silence.
        """
        from fused_memory.middleware.candidate_key_escalation import (  # noqa: PLC0415
            _ANCHOR_TASK_ID as CANDIDATE_KEY_ANCHOR,
        )
        from fused_memory.middleware.mem0_update_storm_escalator import (  # noqa: PLC0415
            _ANCHOR_TASK_ID as MEM0_UPDATE_ANCHOR,
        )
        from fused_memory.middleware.scope_violation_escalator import (  # noqa: PLC0415
            _ANCHOR_TASK_ID as SCOPE_VIOLATION_ANCHOR,
        )
        from fused_memory.server.markup_guard import (  # noqa: PLC0415
            _RESIDUE_ANCHOR_TASK_ID as GUARD_RESIDUE_ANCHOR,
        )
        from fused_memory.server.markup_guard import (
            _STORM_ANCHOR_TASK_ID as GUARD_STORM_ANCHOR,
        )
        from fused_memory.server.markup_tripwire import (  # noqa: PLC0415
            _ANCHOR_TASK_ID as TRIPWIRE_ANCHOR,
        )
        from fused_memory.server.markup_tripwire import (
            _RESIDUE_ANCHOR_TASK_ID as RESIDUE_ANCHOR,
        )

        ours = write_triage._ANCHOR_TASK_ID
        assert ours == 'write-triage-fail-open'
        for label, theirs in [
            ('markup_tripwire (the SQUATTED one)', TRIPWIRE_ANCHOR),
            ('markup_tripwire residue', RESIDUE_ANCHOR),
            ('markup_guard storm', GUARD_STORM_ANCHOR),
            ('markup_guard residue', GUARD_RESIDUE_ANCHOR),
            ('candidate_key_escalation', CANDIDATE_KEY_ANCHOR),
            ('mem0_update_storm_escalator', MEM0_UPDATE_ANCHOR),
            ('scope_violation_escalator', SCOPE_VIOLATION_ANCHOR),
        ]:
            assert ours != theirs, (
                f'write-triage must not share the {label} anchor {theirs!r}: a '
                'filer deduping against an anchor another party keeps open is '
                'suppressed indefinitely, and that silence reads as calm'
            )

    def test_the_filed_anchor_and_the_dedup_lookup_are_the_same(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Filing under one anchor while deduping against another is the bug.

        It would produce a record nobody dedupes against (unbounded duplicates)
        or a lookup nobody files under (permanent suppression). Pinned by
        capturing the anchor the dedup read is called with and comparing it to
        the `task_id` that actually landed.
        """
        if not write_triage.HAS_ESCALATION:
            pytest.skip('escalation package unavailable in this environment')

        seen: list = []
        real_get_by_task = write_triage.EscalationQueue.get_by_task

        def _spy(self, task_id, status=None):
            seen.append(task_id)
            return real_get_by_task(self, task_id, status=status)

        monkeypatch.setattr(write_triage.EscalationQueue, 'get_by_task', _spy)
        emit_triage_fail_open_storm_escalation(str(tmp_path), _STORM)

        payload = json.loads(
            next((tmp_path / 'data' / 'escalations').glob('esc-*.json')).read_text()
        )
        assert seen == [payload['task_id']], (
            f'deduped against {seen!r} but filed under {payload["task_id"]!r}'
        )
