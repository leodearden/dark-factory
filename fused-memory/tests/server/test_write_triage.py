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

import types
from unittest.mock import AsyncMock, Mock

import pytest

from fused_memory.models.enums import MEM0_PRIMARY, MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    _DEFAULT_CANDIDATE_K,
    _DEFAULT_WRITE_TRIAGE_ENABLED,
    CANONICAL_ID_KEY,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    TRIAGE_OUTCOMES,
    decide_band,
    resolve_bands,
    resolve_candidate_k,
    resolve_write_triage_enabled,
    retrieve_candidates,
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
