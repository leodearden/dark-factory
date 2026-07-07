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

import ast
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


# ---------------------------------------------------------------------------
# AST call-site boundary — every _emit_merge_attempt(...) call site in the
# three source modules passes an OutcomeKind member, never a bare string.
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).parent.parent / 'src' / 'orchestrator'
_CALL_SITE_FILES = ['merge_queue.py', 'merge_gates.py', 'workflow.py']


def _is_outcome_kind_attribute(node: ast.expr) -> bool:
    """``OutcomeKind.<member>`` — an attribute access off the ``OutcomeKind`` name."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'OutcomeKind'
    )


def _is_allowed_outcome_arg(node: ast.expr) -> bool:
    """The three shapes a converted outcome arg may take.

    (a) an ``OutcomeKind.<member>`` attribute access; (b) a conditional
    expression choosing between two such accesses (merge_queue.py's
    ``'conflict' if merge_result.conflicts else 'merge_failed'`` site); or
    (c) a bare ``Name`` — the ``emit_subtype`` local passthrough
    (merge_gates.py:618), which is guard-narrowed to ``OutcomeKind`` before
    the call and is typed, not literal, so it is exempt from the
    bare-string check below.
    """
    if _is_outcome_kind_attribute(node):
        return True
    if isinstance(node, ast.IfExp):
        return _is_outcome_kind_attribute(node.body) and _is_outcome_kind_attribute(node.orelse)
    return isinstance(node, ast.Name)


def _emit_merge_attempt_outcome_args(path: Path) -> list[tuple[int, ast.expr]]:
    """Return ``(lineno, outcome_arg)`` for every ``_emit_merge_attempt(...)`` call in *path*.

    The outcome arg is always positional ``args[2]`` (no call site anywhere
    in src passes ``outcome=`` as a keyword) — asserted here rather than
    silently mis-indexing if that convention is ever broken.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == '_emit_merge_attempt'
        ):
            assert not any(kw.arg == 'outcome' for kw in node.keywords), (
                f'{path.name}:{node.lineno} passes outcome= as a keyword; '
                'the AST scan assumes positional args[2]'
            )
            assert len(node.args) >= 3, (
                f'{path.name}:{node.lineno} has fewer than 3 positional args'
            )
            calls.append((node.lineno, node.args[2]))
    return calls


class TestOutcomeKindCallSiteBoundary:
    """Every _emit_merge_attempt call site passes an OutcomeKind member, never a bare string.

    This is the durable seam invariant — not "must literally be
    ``OutcomeKind.<member>``", which would false-fail two legitimate sites
    (merge_queue.py's conditional conflict/merge_failed pick, and
    merge_gates.py's ``emit_subtype`` local passthrough). pyright enforces
    member-typedness at the call; this test enforces that a bare string
    literal never reappears at any of these call sites. Parametrized per
    file so each reports independently.
    """

    @pytest.mark.parametrize('filename', _CALL_SITE_FILES)
    def test_no_bare_string_outcome_args(self, filename: str) -> None:
        path = _SRC_DIR / filename
        for lineno, arg in _emit_merge_attempt_outcome_args(path):
            location = f'{filename}:{lineno}'
            assert not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)), (
                f'{location} passes a bare string literal as the outcome arg; '
                'use an OutcomeKind member instead'
            )
            assert _is_allowed_outcome_arg(arg), (
                f'{location}: outcome arg is not an OutcomeKind attribute access, '
                f'a conditional over two such accesses, or a Name — got {ast.dump(arg)}'
            )
