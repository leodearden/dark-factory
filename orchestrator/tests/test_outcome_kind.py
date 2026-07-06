"""Tests for ``OutcomeKind`` — the ``_emit_merge_attempt`` outcome vocabulary.

Task 2165: ``OutcomeKind(StrEnum)`` is the enumerated contract for the
``data['outcome']`` values emitted by ``_emit_merge_attempt`` (merge_queue.py).
These tests pin: the exact 21-member vocabulary, str-compatibility (existing
``==``/``in``/json comparisons must keep working unchanged), the
``is_terminal`` classification, byte-identical payloads through the real
``EventStore`` emit chokepoint, and the non-terminal set as a frozen contract
(mirrored by the dashboard's ``_ACTIVE_ONLY`` allowlist — see
plans/dashboard-alignment-prd.md).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from orchestrator.event_store import EventStore
from orchestrator.merge_queue import _emit_merge_attempt
from orchestrator.merge_types import OutcomeKind

# The exact 21-member vocabulary emitted through _emit_merge_attempt (see its
# docstring at merge_queue.py:1363 and plans/dashboard-alignment-prd.md).
# 'blocked' is intentionally excluded: bare-infrastructure 'blocked' outcomes
# are documented as NOT emitted by _emit_merge_attempt.
_EXPECTED_VALUES = {
    'done', 'already_merged', 'unknown_branch', 'conflict', 'merge_failed',
    'verify_failed', 'advance_failed', 'dropped_plan_targets',
    'abandoned_verify_timeouts', 'train_incomplete', 'train_rebase_conflict',
    'train_partial_flip', 'cas_exhausted', 'main_health_red',
    'post_merge_equivalence_failed', 'post_merge_pyright_broken',
    'plan_files_not_touched', 'plan_files_narrowed', 'cas_retry',
    'gate_retry', 'post_merge_generation_chained',
}

# Attempt continues / merge is still live for these four — see design
# rationale in .task/plan.json (task 2165, design_decisions[1]).
_NON_TERMINAL_VALUES = {
    'cas_retry', 'gate_retry', 'post_merge_generation_chained', 'plan_files_narrowed',
}


class TestOutcomeKindVocabulary:
    """The enum's value set is exactly the 21-member _emit_merge_attempt vocabulary."""

    def test_value_set_is_exact(self) -> None:
        assert {m.value for m in OutcomeKind} == _EXPECTED_VALUES

    def test_member_names_match_values(self) -> None:
        """Lowercase name==value, matching EventType's convention (event_store.py:44)."""
        for member in OutcomeKind:
            assert member.name == member.value

    def test_blocked_is_not_a_member(self) -> None:
        assert 'blocked' not in {m.value for m in OutcomeKind}
        with pytest.raises(ValueError):
            OutcomeKind('blocked')


class TestOutcomeKindStrCompat:
    """Members are str instances (StrEnum) — existing ==/in/json comparisons keep working."""

    def test_str_of_member_equals_value(self) -> None:
        assert str(OutcomeKind.done) == 'done'

    def test_member_equals_raw_string(self) -> None:
        assert OutcomeKind.cas_retry == 'cas_retry'


class TestOutcomeKindIsTerminal:
    """is_terminal is False for the 4-member non-terminal set, True otherwise."""

    def test_non_terminal_members_are_not_terminal(self) -> None:
        for value in _NON_TERMINAL_VALUES:
            member = OutcomeKind(value)
            assert member.is_terminal is False, f'{member!r} should not be terminal'

    def test_terminal_members_are_terminal(self) -> None:
        terminal_values = _EXPECTED_VALUES - _NON_TERMINAL_VALUES
        assert len(terminal_values) == 17
        for value in terminal_values:
            member = OutcomeKind(value)
            assert member.is_terminal is True, f'{member!r} should be terminal'


class TestOutcomeKindPayloadIdentity:
    """_emit_merge_attempt(member) round-trips through EventStore byte-identically.

    Reuses the EventStore-construct + sqlite json_extract readback pattern from
    test_event_store.py's _query_all and test_merge_queue.py's
    "$.outcome" json_extract queries.
    """

    @pytest.mark.parametrize('value', sorted(_EXPECTED_VALUES))
    def test_payload_identity_through_real_chokepoint(
        self, tmp_path: Path, value: str,
    ) -> None:
        member = OutcomeKind(value)
        db_path = tmp_path / 'x.db'
        event_store = EventStore(db_path, 'run')

        _emit_merge_attempt(event_store, 'task-1', member)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') FROM events "
            "WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert rows == [(member.value,)], f'rows={rows}'
        assert isinstance(rows[0][0], str)


class TestOutcomeKindFrozenContract:
    """Pins the non-terminal set as a frozen contract.

    Changing this set REQUIRES updating
    dashboard/src/dashboard/data/merge_queue.py's ``_ACTIVE_ONLY`` allowlist
    in the SAME change — the dashboard mirrors this set with no orchestrator
    import, so nothing on the dashboard side will fail automatically if this
    set changes without a matching dashboard edit. This test is the drift
    tripwire.
    """

    def test_non_terminal_set_is_frozen(self) -> None:
        non_terminal = {m for m in OutcomeKind if not m.is_terminal}
        assert non_terminal == {OutcomeKind(v) for v in _NON_TERMINAL_VALUES}
