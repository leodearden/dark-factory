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
from unittest.mock import Mock

import pytest
from fused_memory.server.write_triage import (
    _DEFAULT_CANDIDATE_K,
    _DEFAULT_WRITE_TRIAGE_ENABLED,
    CANONICAL_ID_KEY,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    TRIAGE_OUTCOMES,
    resolve_bands,
    resolve_candidate_k,
    resolve_write_triage_enabled,
)

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
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
