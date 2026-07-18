"""Tests for orchestrator.verify_checkpoint — green_checkpoint_at_tip(...).

The durable verified-green checkpoint predicate (task 2752): returns True iff
a durable (cross-restart) ``workflow_verify`` row for *task_id* has a truthy
``data['passed']`` AND ``data['tip_sha'] == tip_sha`` — i.e. the branch was
verified green at the CURRENT branch tip in this or a prior orchestrator run.
Strict fail-closed: any missing/mismatched field, a None input, or a read
error returns False and NEVER raises.

step-3: failing tests, written before ``orchestrator.verify_checkpoint``
exists (RED — ``ModuleNotFoundError``). step-4 adds the module and makes
these green.
"""

from __future__ import annotations

from typing import Any

from orchestrator.event_store import EventType


class _FakeEventStore:
    """Lightweight fake exposing
    ``fetch_events_by_type_all_runs(event_type, *, task_id=None) -> list[dict]``,
    mirroring ``orchestrator.event_store.EventStore.fetch_events_by_type_all_runs``'s
    run-agnostic, optionally task-scoped return shape (each row a dict carrying
    at least 'task_id'/'data') without touching sqlite.
    """

    def __init__(
        self,
        rows_by_type: dict[EventType, list[dict[str, Any]]] | None = None,
        *,
        raises: bool = False,
    ) -> None:
        self._rows_by_type = rows_by_type or {}
        self._raises = raises

    def fetch_events_by_type_all_runs(
        self, event_type: EventType, *, task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._raises:
            raise RuntimeError('simulated event-store read failure')
        rows = self._rows_by_type.get(event_type, [])
        if task_id is not None:
            rows = [r for r in rows if r.get('task_id') == task_id]
        return rows


def _verify_row(task_id: str, *, passed: bool, tip_sha: str) -> dict[str, Any]:
    """A canned workflow_verify row, matching fetch_events_by_type_all_runs's shape."""
    return {
        'task_id': task_id,
        'data': {'passed': passed, 'tip_sha': tip_sha, 'base_sha': 'deadbeef'},
    }


class TestGreenCheckpointAtTip:
    """green_checkpoint_at_tip is a strict fail-closed durable-tip-match boolean."""

    # -- (i) the sole True case ------------------------------------------

    def test_true_when_passing_row_at_matching_tip(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True, tip_sha='tipA')],
        })
        assert green_checkpoint_at_tip(
            store, EventType.workflow_verify, '42', 'tipA'
        ) is True

    # -- (ii) tip / passed / task mismatches -----------------------------

    def test_false_when_tip_differs(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True, tip_sha='tipA')],
        })
        assert green_checkpoint_at_tip(
            store, EventType.workflow_verify, '42', 'tipDIFFERENT'
        ) is False

    def test_false_when_passed_is_falsy(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=False, tip_sha='tipA')],
        })
        assert green_checkpoint_at_tip(
            store, EventType.workflow_verify, '42', 'tipA'
        ) is False

    def test_false_when_only_matching_row_is_different_task(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('99', passed=True, tip_sha='tipA')],
        })
        assert green_checkpoint_at_tip(
            store, EventType.workflow_verify, '42', 'tipA'
        ) is False

    def test_false_when_no_rows(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        store = _FakeEventStore({})
        assert green_checkpoint_at_tip(
            store, EventType.workflow_verify, '42', 'tipA'
        ) is False

    # -- (iii) fail-closed inputs ----------------------------------------

    def test_false_when_event_store_is_none(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        assert green_checkpoint_at_tip(
            None, EventType.workflow_verify, '42', 'tipA'
        ) is False

    def test_false_when_task_id_is_none(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True, tip_sha='tipA')],
        })
        assert green_checkpoint_at_tip(
            store, EventType.workflow_verify, None, 'tipA'
        ) is False

    def test_false_when_tip_sha_is_none(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        store = _FakeEventStore({
            EventType.workflow_verify: [_verify_row('42', passed=True, tip_sha='tipA')],
        })
        assert green_checkpoint_at_tip(
            store, EventType.workflow_verify, '42', None
        ) is False

    def test_false_and_never_raises_when_fetch_raises(self):
        from orchestrator.verify_checkpoint import green_checkpoint_at_tip

        store = _FakeEventStore(raises=True)
        assert green_checkpoint_at_tip(
            store, EventType.workflow_verify, '42', 'tipA'
        ) is False
