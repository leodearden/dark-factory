"""Tests for orchestrator.merge_completion — merge_completion_eligible(event_store, task_id).

Covers the (b)+(c) run-scoped eligibility predicate for the merge-stage
completion mode (task 2633): a passing ``workflow_verify`` event (b, VERIFY
passed) AND a ``phase_enter`` event with ``phase=='merge'`` (c, REVIEW
passed), both keyed by task_id, read from the current run's event-store
history. (a) ``block_class == BlockClass.MERGE_VERIFY_RED`` is the CALLER's
precondition and is deliberately not modeled here — see
``orchestrator.merge_completion``'s module docstring.

step-1: failing tests, written before ``orchestrator.merge_completion``
exists (RED — ``ModuleNotFoundError``). step-2 adds the module and makes
these green.
"""

from __future__ import annotations

from typing import Any

from orchestrator.event_store import EventType


class _FakeEventStore:
    """Lightweight fake exposing ``fetch_events_by_type(event_type) -> list[dict]``,
    matching ``orchestrator.event_store.EventStore.fetch_events_by_type``'s
    return shape (each row a dict carrying at least 'task_id'/'phase'/'data')
    without touching sqlite.
    """

    def __init__(
        self,
        rows_by_type: dict[EventType, list[dict[str, Any]]] | None = None,
        *,
        raises: bool = False,
    ) -> None:
        self._rows_by_type = rows_by_type or {}
        self._raises = raises

    def fetch_events_by_type(self, event_type: EventType) -> list[dict[str, Any]]:
        if self._raises:
            raise RuntimeError('simulated event-store read failure')
        return self._rows_by_type.get(event_type, [])


def _verify_row(task_id: str, *, passed: bool) -> dict[str, Any]:
    """A canned workflow_verify row, matching fetch_events_by_type's shape."""
    return {'task_id': task_id, 'phase': None, 'data': {'passed': passed}}


def _phase_enter_row(task_id: str, *, phase: str) -> dict[str, Any]:
    """A canned phase_enter row, matching fetch_events_by_type's shape."""
    return {'task_id': task_id, 'phase': phase, 'data': {}}


class TestMergeCompletionEligible:
    """merge_completion_eligible is a strict fail-closed (b)+(c) boolean —
    see module docstring for what (b) and (c) mean and why (a) is excluded.
    """

    # -- (i) both conditions present -----------------------------------

    def test_true_when_both_conditions_present(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True)],
            EventType.phase_enter: [_phase_enter_row('42', phase='merge')],
        })
        assert merge_completion_eligible(store, '42') is True

    # -- (ii) condition (b) unmet ----------------------------------------

    def test_false_when_workflow_verify_row_missing_entirely(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore({
            EventType.phase_enter: [_phase_enter_row('42', phase='merge')],
        })
        assert merge_completion_eligible(store, '42') is False

    def test_false_when_workflow_verify_present_but_not_passed(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=False)],
            EventType.phase_enter: [_phase_enter_row('42', phase='merge')],
        })
        assert merge_completion_eligible(store, '42') is False

    def test_false_when_workflow_verify_passed_only_for_different_task(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('99', passed=True)],
            EventType.phase_enter: [_phase_enter_row('42', phase='merge')],
        })
        assert merge_completion_eligible(store, '42') is False

    # -- (iii) condition (c) unmet ----------------------------------------

    def test_false_when_phase_enter_merge_row_missing_entirely(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True)],
        })
        assert merge_completion_eligible(store, '42') is False

    def test_false_when_phase_enter_present_but_wrong_phase(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True)],
            EventType.phase_enter: [_phase_enter_row('42', phase='review')],
        })
        assert merge_completion_eligible(store, '42') is False

    def test_false_when_phase_enter_merge_only_for_different_task(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True)],
            EventType.phase_enter: [_phase_enter_row('99', phase='merge')],
        })
        assert merge_completion_eligible(store, '42') is False

    # -- (iv) fail-closed inputs ------------------------------------------

    def test_false_when_event_store_is_none(self):
        from orchestrator.merge_completion import merge_completion_eligible

        assert merge_completion_eligible(None, '42') is False

    def test_false_when_task_id_is_none(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True)],
            EventType.phase_enter: [_phase_enter_row('42', phase='merge')],
        })
        assert merge_completion_eligible(store, None) is False

    def test_false_when_fetch_events_by_type_raises(self):
        from orchestrator.merge_completion import merge_completion_eligible

        store = _FakeEventStore(raises=True)
        assert merge_completion_eligible(store, '42') is False
