"""Tests for standing_decision_constants — the INV-5 single-source module for
the entity-standing-decision batch (task 2894 α, PRD
plans/stage1-entity-standing-decision-prd.md).

These are contract/coherence assertions only: they lock the persisted-contract
string values (record kind, mem0 kind, TTL), the closed grounds enum, the
INV-5 grounds→token-family binding coherence, and the state/expiry_reason
vocabularies that β/γ/δ/ε/ζ consume. No introspection/docstring meta-tests.
"""

from __future__ import annotations

from fused_memory.reconciliation import standing_decision_constants as sdc


def test_grounds_enum_is_closed_frozenset_with_structural_size_conflation():
    """GROUNDS_ENUM is a frozenset (closed) whose sole seed member is
    GROUNDS_STRUCTURAL_SIZE_CONFLATION=='structural_size_conflation'."""
    assert isinstance(sdc.GROUNDS_ENUM, frozenset)
    assert sdc.GROUNDS_STRUCTURAL_SIZE_CONFLATION == 'structural_size_conflation'
    assert sdc.GROUNDS_STRUCTURAL_SIZE_CONFLATION in sdc.GROUNDS_ENUM
    assert frozenset({sdc.GROUNDS_STRUCTURAL_SIZE_CONFLATION}) == sdc.GROUNDS_ENUM


def test_grounds_token_families_covers_every_grounds_value_with_str_tokens():
    """INV-5 coherence: every grounds value has a bound token-family, and each
    bound family is a tuple/frozenset of str (α owns the binding STRUCTURE;
    seed token contents are γ-tactical)."""
    assert set(sdc.GROUNDS_TOKEN_FAMILIES.keys()) == set(sdc.GROUNDS_ENUM)
    for grounds, family in sdc.GROUNDS_TOKEN_FAMILIES.items():
        assert grounds in sdc.GROUNDS_ENUM
        assert isinstance(family, (tuple, frozenset)), (
            f'token family for {grounds!r} must be a tuple/frozenset, got {type(family)}'
        )
        assert all(isinstance(token, str) for token in family)


def test_persisted_contract_string_values():
    """Persisted-contract values (record kind, mem0 kind, TTL) are pinned —
    they cross a persistence/serialization boundary and must not drift."""
    assert sdc.RECORD_KIND_ENTITY_STANDING_DECISION == 'entity_standing_decision'
    assert sdc.MEM0_KIND_INVESTIGATION_OUTCOME == 'investigation_outcome'
    assert sdc.STANDING_DECISION_TTL_DAYS == 90


def test_state_vocabulary_and_members():
    """STANDING_DECISION_STATES is the closed {active, expired, revoked} set and
    each STATE_* constant is a member of it."""
    assert {'active', 'expired', 'revoked'} == sdc.STANDING_DECISION_STATES
    assert sdc.STATE_ACTIVE == 'active'
    assert sdc.STATE_EXPIRED == 'expired'
    assert sdc.STATE_REVOKED == 'revoked'
    for state in (sdc.STATE_ACTIVE, sdc.STATE_EXPIRED, sdc.STATE_REVOKED):
        assert state in sdc.STANDING_DECISION_STATES


def test_expiry_reason_vocabulary_and_members():
    """EXPIRY_REASONS is the closed {ttl, growth, merge, operator} set and each
    EXPIRY_REASON_* constant is a member of it (INV-2: structured expiry_reason
    stamped at the state transition)."""
    assert {'ttl', 'growth', 'merge', 'operator'} == sdc.EXPIRY_REASONS
    assert sdc.EXPIRY_REASON_TTL == 'ttl'
    assert sdc.EXPIRY_REASON_GROWTH == 'growth'
    assert sdc.EXPIRY_REASON_MERGE == 'merge'
    assert sdc.EXPIRY_REASON_OPERATOR == 'operator'
    for reason in (
        sdc.EXPIRY_REASON_TTL,
        sdc.EXPIRY_REASON_GROWTH,
        sdc.EXPIRY_REASON_MERGE,
        sdc.EXPIRY_REASON_OPERATOR,
    ):
        assert reason in sdc.EXPIRY_REASONS


def test_suppression_streak_record_kind_is_distinct_and_not_a_marker_kind():
    """The streak's ledger record kind is pinned and DISTINCT from the decision
    row's kind (a collision would make the streak upsert overwrite the decision
    row itself), and it is not a per-task marker kind, so gc()'s terminal-task
    DELETE arm never touches it and only the expires_at arm reaps it."""
    from fused_memory.reconciliation.recon_ledger import MARKER_KINDS

    assert (
        sdc.RECORD_KIND_ENTITY_SUPPRESSION_STREAK
        == 'entity_standing_decision_suppression_streak'
    )
    assert (
        sdc.RECORD_KIND_ENTITY_SUPPRESSION_STREAK
        != sdc.RECORD_KIND_ENTITY_STANDING_DECISION
    )
    assert sdc.RECORD_KIND_ENTITY_SUPPRESSION_STREAK not in MARKER_KINDS


def test_suppression_streak_threshold_is_three_cycles():
    """K is an int >= 2 (a one-cycle "streak" would measure a burst, which the
    per-cycle escape already owns, rather than persistence) and is decided at
    3, matching ζ's GROWTH_SWEEP_FAILURE_STREAK_THRESHOLD (PRD Open Question
    4)."""
    assert isinstance(sdc.SUPPRESSION_STREAK_THRESHOLD_CYCLES, int)
    assert not isinstance(sdc.SUPPRESSION_STREAK_THRESHOLD_CYCLES, bool)
    assert sdc.SUPPRESSION_STREAK_THRESHOLD_CYCLES >= 2
    assert sdc.SUPPRESSION_STREAK_THRESHOLD_CYCLES == 3


def test_streak_payload_key_is_distinct_from_done_suppressions_key():
    """The streak count's payload key is a non-empty str that does not collide
    with flag_dedup's consecutive-cycle done-suppression counter key."""
    from fused_memory.reconciliation.flag_dedup import _DONE_SUPPRESSIONS_PAYLOAD_KEY

    assert isinstance(sdc.STREAK_PAYLOAD_KEY, str)
    assert sdc.STREAK_PAYLOAD_KEY
    assert sdc.STREAK_PAYLOAD_KEY != _DONE_SUPPRESSIONS_PAYLOAD_KEY


def test_streak_window_payload_key_is_distinct_from_every_streak_row_key():
    """The per-cycle window's payload key is a non-empty str that collides with
    none of the other keys the ledger writes on a streak row, nor with
    flag_dedup's done-suppression counter key."""
    from fused_memory.reconciliation.flag_dedup import _DONE_SUPPRESSIONS_PAYLOAD_KEY

    assert isinstance(sdc.STREAK_WINDOW_PAYLOAD_KEY, str)
    assert sdc.STREAK_WINDOW_PAYLOAD_KEY
    other_keys = {
        sdc.STREAK_PAYLOAD_KEY,
        'last_run_id',
        'grounds',
        'updated_at',
        _DONE_SUPPRESSIONS_PAYLOAD_KEY,
    }
    assert sdc.STREAK_WINDOW_PAYLOAD_KEY not in other_keys


def test_streak_volume_threshold_is_the_per_cycle_n():
    """The PRD's single N governs both "more than N flags in one cycle" and
    "across a streak of cycles", so the streak's volume threshold is the
    per-cycle threshold, a non-bool int."""
    assert isinstance(sdc.SUPPRESSION_STREAK_VOLUME_THRESHOLD, int)
    assert not isinstance(sdc.SUPPRESSION_STREAK_VOLUME_THRESHOLD, bool)
    assert (
        sdc.SUPPRESSION_STREAK_VOLUME_THRESHOLD
        == sdc.SUPPRESSION_STORM_THRESHOLD_PER_CYCLE
    )


def test_a_decision_working_as_intended_never_trips_the_streak_escape():
    """A decision that works suppresses its re-derived complaint about once per
    cycle, indefinitely (PRD §Goal): Hook A drops the flag only after Stage 1
    has emitted it. The streak escape sums the last K cycles, so that steady
    state totals K·1. If K ever exceeded N, every healthy decision would page
    the storm escape once its streak reached K, which is the review-round-1
    defect this invariant keeps closed when either number is tuned."""
    steady_state_flags_per_cycle = 1
    assert (
        sdc.SUPPRESSION_STREAK_THRESHOLD_CYCLES * steady_state_flags_per_cycle
        <= sdc.SUPPRESSION_STREAK_VOLUME_THRESHOLD
    )
