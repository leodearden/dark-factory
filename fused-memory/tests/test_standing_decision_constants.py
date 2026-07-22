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
