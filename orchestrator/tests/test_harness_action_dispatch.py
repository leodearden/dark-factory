"""Tests for Harness action dispatch — β (Pair A, B, C, D, E, F).

Covers the refactored _on_escalation_resolved that generalises from a
resume-only cascade+orphan flip into a full action dispatch keyed on
Escalation.resolution_action, with legacy fallback mapping (D10).

Step-1 (Pair A) — legacy mapping + close_only no-op + dispatch skeleton:
  B8  — explicit close_only → no set_task_status
  B14 — legacy dismissed (resolution_action=None, status=dismissed) → close_only → no flip
  (Fix#1a regression) — legacy resolved → resume → set_task_status('pending')
  _resolve_escalation_action unit assertions (method must exist; fail until step-2)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Fixture — same pattern as test_cascade_unblock.py
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness with mocked internals for action-dispatch unit testing."""
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks (same as test_cascade_unblock.py)
    h.scheduler = MagicMock()
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()

    # _merge_worker stays None — unhalt branch skipped in all tests here
    return h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_esc(
    task_id: str = 'task-1',
    status: str = 'resolved',
    resolved_by: str = 'steward',
    level: int = 1,
    resolution_action: str | None = None,
) -> Escalation:
    return Escalation(
        id=f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category='infra_issue',
        summary='dispatch test',
        level=level,
        status=status,
        resolved_by=resolved_by,
        resolution_action=resolution_action,
    )


# ---------------------------------------------------------------------------
# Pair A — _resolve_escalation_action unit tests
# (RED until the helper is added in step-2)
# ---------------------------------------------------------------------------

class TestResolveEscalationAction:
    """Unit tests for the new _resolve_escalation_action() helper."""

    def test_explicit_resolution_action_returned(self, harness: Harness):
        """When resolution_action is set on the record, return it directly."""
        esc = _make_esc(resolution_action='close_only', status='dismissed')
        # Fails until _resolve_escalation_action exists (step-2)
        action = harness._resolve_escalation_action(esc)
        assert action == 'close_only'

    def test_explicit_resume_returned(self, harness: Harness):
        """Explicit resolution_action='resume' is returned as-is."""
        esc = _make_esc(resolution_action='resume', status='resolved')
        action = harness._resolve_escalation_action(esc)
        assert action == 'resume'

    def test_explicit_restart_returned(self, harness: Harness):
        """Explicit resolution_action='restart' is returned as-is."""
        esc = _make_esc(resolution_action='restart', status='resolved')
        action = harness._resolve_escalation_action(esc)
        assert action == 'restart'

    def test_legacy_dismissed_maps_to_close_only(self, harness: Harness):
        """Legacy: resolution_action=None, status='dismissed' → 'close_only' (D10)."""
        esc = _make_esc(resolution_action=None, status='dismissed', resolved_by='steward')
        action = harness._resolve_escalation_action(esc)
        assert action == 'close_only'

    def test_legacy_resolved_maps_to_resume(self, harness: Harness):
        """Legacy: resolution_action=None, status='resolved' → 'resume' (D10)."""
        esc = _make_esc(resolution_action=None, status='resolved', resolved_by='steward')
        action = harness._resolve_escalation_action(esc)
        assert action == 'resume'


# ---------------------------------------------------------------------------
# Pair A — dispatch behavioral tests
# (Some are already GREEN pre-step-2; all serve as regression net)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDispatchCloseOnly:
    """B8: close_only → no set_task_status call, no kill."""

    async def test_explicit_close_only_no_flip(self, harness: Harness):
        """Explicit resolution_action='close_only' → set_task_status NOT awaited."""
        esc = _make_esc(
            task_id='task-1',
            resolution_action='close_only',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_explicit_close_only_level0_no_flip(self, harness: Harness):
        """close_only at level 0 also produces no flip."""
        esc = _make_esc(
            task_id='task-2',
            resolution_action='close_only',
            status='dismissed',
            level=0,
        )
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
class TestDispatchLegacyPaths:
    """B14, D10: Legacy mapping paths through the dispatch."""

    async def test_legacy_dismiss_close_only_no_flip(self, harness: Harness):
        """Legacy dismissed (resolution_action=None, status='dismissed') → close_only → no flip."""
        esc = _make_esc(
            task_id='task-3',
            resolution_action=None,
            status='dismissed',
            resolved_by='steward',
            level=1,
        )
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_legacy_resolve_orphan_flip_pending(self, harness: Harness):
        """Legacy resolve (resolution_action=None, status='resolved') → resume → set_task_status('pending').

        Preserves Fix#1a: orphan L1 (no active workflow), not l2-cascade,
        task is blocked → flipped to pending.
        """
        task_id = 'task-4'
        esc = _make_esc(
            task_id=task_id,
            resolution_action=None,
            status='resolved',
            resolved_by='escalation-watcher-auto',
            level=1,
        )
        # No _escalation_events entry → orphaned (no active workflow)
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'pending',
        )

    async def test_legacy_resolve_active_workflow_no_flip(self, harness: Harness):
        """Legacy resolve with active workflow (task_id in _escalation_events) → no flip.

        The workflow owns its own re-pend (woken by event.set()).
        """
        task_id = 'task-5'
        esc = _make_esc(
            task_id=task_id,
            resolution_action=None,
            status='resolved',
            resolved_by='escalation-watcher-auto',
            level=1,
        )
        # Active workflow slot registered
        harness._escalation_events[task_id] = asyncio.Event()
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert harness._escalation_events[task_id].is_set(), (
            'Wake event must be set synchronously even when flip is suppressed'
        )

    async def test_legacy_resolve_level0_no_flip(self, harness: Harness):
        """Legacy resolve at level 0 → no flip (level gate preserved)."""
        task_id = 'task-6'
        esc = _make_esc(
            task_id=task_id,
            resolution_action=None,
            status='resolved',
            resolved_by='steward',
            level=0,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Pair B — resume level>=1 generalization (D7)
# Step-3: RED until the resume gate is changed from level==1 to level>=1
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestResumeLevelGate:
    """B7 — D7: resume gate must be level>=1, not level==1."""

    async def test_born_at_l2_resume_flips_blocked(self, harness: Harness):
        """(a) Born-at-L2 (level=2, resolution_action='resume', direct resolve,
        NOT in _escalation_events, status='blocked') → set_task_status('pending').

        Currently strands because the resume gate is level==1.  Fails until
        step-4 widens the gate to level>=1.
        """
        task_id = 'task-l2-born'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='resume',
            status='resolved',
            resolved_by='interactive',
            level=2,
        )
        # Not in _escalation_events → orphan (no active workflow)
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # RED until step-4: currently no flip because level==2 != 1
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'pending',
        )

    async def test_level0_resume_no_flip(self, harness: Harness):
        """(b) level=0 direct resume still does NOT flip — level floor preserved.

        After step-4 changes gate to level>=1, level==0 is still excluded.
        """
        task_id = 'task-l0'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='resume',
            status='resolved',
            resolved_by='steward',
            level=0,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_cascade_member_parent_resume_action_flips(self, harness: Harness):
        """(c) Cascade member (level=1, resolved_by='l2-cascade:<id>') whose
        parent L2 carries resolution_action='resume' → flips blocked→pending
        via _resolve_escalation_action parent lookup.
        """
        parent_id = 'esc-l2-7'
        task_id = 'task-member-7'
        # Wire a mock queue that returns a parent with explicit resolution_action
        parent_esc = _make_esc(
            task_id='task-cluster-7',
            resolution_action='resume',
            status='resolved',
            resolved_by='interactive',
            level=2,
        )
        parent_esc = Escalation(
            id=parent_id,
            task_id='task-cluster-7',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='infra_issue',
            summary='L2 parent with resume',
            level=2,
            status='resolved',
            resolved_by='interactive',
            resolution_action='resume',
        )
        mock_queue = MagicMock()
        mock_queue.get = MagicMock(return_value=parent_esc)
        harness._escalation_queue = mock_queue

        esc = Escalation(
            id=f'esc-{task_id}-1',
            task_id=task_id,
            agent_role='workflow',
            severity='blocking',
            category='infra_issue',
            summary='L1 cascade member',
            level=1,
            status='resolved',
            resolved_by=f'l2-cascade:{parent_id}',
            resolution_action=None,  # cascade doesn't copy resolution_action
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # Parent lookup returns 'resume' → should flip
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'pending',
        )
        # Verify queue.get was called with the parent id
        mock_queue.get.assert_called_once_with(parent_id)

    async def test_cascade_member_legacy_parent_no_action_still_flips(
        self, harness: Harness
    ):
        """Cascade member whose parent L2 has NO resolution_action → legacy map
        → status='resolved' → 'resume' → still flips (regression safety).
        """
        parent_id = 'esc-l2-legacy'
        task_id = 'task-member-legacy'
        parent_esc = Escalation(
            id=parent_id,
            task_id='task-cluster-legacy',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='infra_issue',
            summary='legacy L2 no action',
            level=2,
            status='resolved',
            resolved_by='interactive',
            resolution_action=None,  # legacy — no action set
        )
        mock_queue = MagicMock()
        mock_queue.get = MagicMock(return_value=parent_esc)
        harness._escalation_queue = mock_queue

        esc = Escalation(
            id=f'esc-{task_id}-1',
            task_id=task_id,
            agent_role='workflow',
            severity='blocking',
            category='infra_issue',
            summary='L1 cascade member',
            level=1,
            status='resolved',
            resolved_by=f'l2-cascade:{parent_id}',
            resolution_action=None,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # Legacy map: member status='resolved' → 'resume' → flip
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'pending',
        )


# ---------------------------------------------------------------------------
# Pair C — restart/park/abandon status writes + terminal recheck (no kill)
# Step-5: RED until _action_teardown_and_set_status is wired (step-6)
# ---------------------------------------------------------------------------

def _make_l1_for_queue(
    queue: EscalationQueue, esc_id: str, task_id: str,
) -> Escalation:
    """Submit a minimal L1 escalation to a real queue."""
    esc = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='steward',
        severity='blocking',
        category='design_concern',
        summary='L1 test escalation',
        level=1,
    )
    queue.submit(esc)
    return esc


def _make_l2_with_action_in_queue(
    queue: EscalationQueue,
    l2_id: str,
    member_ids: list[str],
    resolution_action: str | None,
) -> Escalation:
    """Submit a minimal L2 cluster escalation to a real queue."""
    esc = Escalation(
        id=l2_id,
        task_id='task-cluster-c',
        agent_role='escalation-watcher-auto',
        severity='blocking',
        category='design_concern',
        summary='L2 cluster with action',
        level=2,
        root_cause='shared root cause C',
        members=list(member_ids),
        resolution_action=resolution_action,
    )
    queue.submit(esc)
    return esc


@pytest.mark.asyncio
class TestDispatchTeardownNoKill:
    """Pair C (B4/B5): restart/park/abandon write status + terminal recheck.

    is_workflow_active is False in all tests here (no kill path exercised).
    Fails until _action_teardown_and_set_status is wired in step-6.
    """

    async def test_restart_sets_pending(self, harness: Harness):
        """resolution_action='restart' on orphan blocked task → set_task_status('pending')."""
        task_id = 'task-restart'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='restart',
            status='resolved',
            resolved_by='interactive',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'pending',
        )

    async def test_park_sets_deferred(self, harness: Harness):
        """resolution_action='park' → set_task_status('deferred')."""
        task_id = 'task-park'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='park',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'deferred',
        )

    async def test_abandon_sets_cancelled(self, harness: Harness):
        """resolution_action='abandon' → set_task_status('cancelled')."""
        task_id = 'task-abandon'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='abandon',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'cancelled',
        )

    async def test_terminal_task_not_overwritten(self, harness: Harness):
        """Terminal recheck: if task is already done/cancelled, no write."""
        for terminal_status in ('done', 'cancelled'):
            harness.scheduler.get_status = AsyncMock(return_value=terminal_status)
            harness.scheduler.set_task_status = AsyncMock()
            harness.is_workflow_active = MagicMock(return_value=False)

            for action in ('restart', 'park', 'abandon'):
                harness.scheduler.set_task_status.reset_mock()  # type: ignore[attr-defined]
                task_id = f'task-terminal-{terminal_status}-{action}'
                esc = _make_esc(
                    task_id=task_id,
                    resolution_action=action,
                    status='dismissed' if action in ('park', 'abandon') else 'resolved',
                    resolved_by='interactive',
                    level=1,
                )
                harness._on_escalation_resolved(esc)
                await asyncio.gather(*list(harness._background_tasks))

                harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_l2_park_cascade_sets_both_members_deferred(
        self, harness: Harness, tmp_path: Path
    ):
        """Integration: L2 cluster resolved with park + two blocked L1 members
        → both get set_task_status('deferred') via parent-action lookup.

        Requires harness._escalation_queue wired so the member callbacks can
        read the parent L2's resolution_action.
        """
        queue = EscalationQueue(tmp_path / 'esc_c')
        l1_a = _make_l1_for_queue(queue, 'esc-c-l1-a', 'task-park-a')
        l1_b = _make_l1_for_queue(queue, 'esc-c-l1-b', 'task-park-b')
        l2 = _make_l2_with_action_in_queue(
            queue, 'esc-c-l2-park', [l1_a.id, l1_b.id], resolution_action='park',
        )

        # Wire the queue so _resolve_escalation_action can look up the parent L2.
        harness._escalation_queue = queue
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)
        queue.set_resolve_callback(harness._on_escalation_resolved)

        queue.resolve(l2.id, 'park the cluster', dismiss=True)
        await asyncio.gather(*list(harness._background_tasks))

        awaits = harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
        # Both L1 member tasks should be set to 'deferred' (L2's task is also
        # written but we only assert the member tasks are present)
        task_ids_written = [a.args[0] for a in awaits]
        assert 'task-park-a' in task_ids_written, (
            f'task-park-a not written; writes: {task_ids_written}'
        )
        assert 'task-park-b' in task_ids_written, (
            f'task-park-b not written; writes: {task_ids_written}'
        )
        for a in awaits:
            assert a.args[1] == 'deferred', f'Expected deferred, got {a.args[1]}'

    async def test_l2_legacy_dismiss_no_touch(
        self, harness: Harness, tmp_path: Path
    ):
        """Integration: legacy L2 (no resolution_action) dismissed → close_only
        → members NOT touched (old behavior preserved)."""
        queue = EscalationQueue(tmp_path / 'esc_d')
        l1 = _make_l1_for_queue(queue, 'esc-d-l1', 'task-d-member')
        l2 = _make_l2_with_action_in_queue(
            queue, 'esc-d-l2-legacy', [l1.id], resolution_action=None,
        )

        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)
        queue.set_resolve_callback(harness._on_escalation_resolved)

        queue.resolve(l2.id, 'legacy dismiss', dismiss=True)
        await asyncio.gather(*list(harness._background_tasks))

        # Legacy L2 dismissed → close_only → no status write
        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Pair D — live-workflow kill sequence + status-precedes-kill (C3.1, D9)
# Step-7: RED until the kill sequence is wired in step-8
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTeardownKillSequence:
    """B6: kill live workflow after writing target status; status PRECEDES kill."""

    def _make_kill_harness(self, harness: Harness):
        """Set up harness for kill-sequence tests: live workflow initially active."""
        call_log: list[str] = []

        async def fake_set_status(tid, status):
            call_log.append(f'set_status:{tid}:{status}')

        def fake_cancel(tid):
            call_log.append(f'cancel:{tid}')
            return True

        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.scheduler.set_task_status = AsyncMock(side_effect=fake_set_status)
        harness.cancel_workflow = MagicMock(side_effect=fake_cancel)
        harness.hard_cancel_workflow = MagicMock(return_value=True)

        # is_workflow_active: True until cancel is called, then False (slot cleared)
        def fake_is_active(tid):
            return not any(e.startswith('cancel:') for e in call_log)

        harness.is_workflow_active = MagicMock(side_effect=fake_is_active)
        # Small poll budget so tests are fast
        harness.config.terminal_status_hard_cancel_polls = 3

        return call_log

    async def test_restart_status_precedes_kill(self, harness: Harness):
        """resolution_action='restart' on live workflow: set_task_status('pending')
        BEFORE cancel_workflow -- status-precedes-kill (C3.1).
        """
        call_log = self._make_kill_harness(harness)
        task_id = 'task-kill-restart'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='restart',
            status='resolved',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # cancel_workflow must be called
        harness.cancel_workflow.assert_called_once_with(task_id)  # type: ignore[attr-defined]

        # Ordering: set_status BEFORE cancel
        assert any('set_status' in e for e in call_log), 'set_task_status not recorded'
        assert any('cancel:' in e for e in call_log), 'cancel_workflow not recorded'
        first_set = next(i for i, e in enumerate(call_log) if 'set_status' in e)
        first_cancel = next(i for i, e in enumerate(call_log) if 'cancel:' in e)
        assert first_set < first_cancel, (
            f'set_status must precede cancel_workflow; call_log={call_log}'
        )

    async def test_park_kills_workflow(self, harness: Harness):
        """resolution_action='park' on live workflow: cancel_workflow called,
        task written to 'deferred'.
        """
        self._make_kill_harness(harness)
        task_id = 'task-kill-park'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='park',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.cancel_workflow.assert_called_once_with(task_id)  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'deferred',
        )

    async def test_abandon_kills_workflow(self, harness: Harness):
        """resolution_action='abandon' on live workflow: cancel_workflow called,
        task written to 'cancelled'.
        """
        self._make_kill_harness(harness)
        task_id = 'task-kill-abandon'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='abandon',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.cancel_workflow.assert_called_once_with(task_id)  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'cancelled',
        )

    async def test_hard_cancel_called_if_slot_persists(self, harness: Harness):
        """If the workflow slot does not clear within the poll budget,
        hard_cancel_workflow is called as the escalation.
        is_workflow_active always True -> polls exhaust -> hard_cancel called.
        """
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.scheduler.set_task_status = AsyncMock()
        harness.cancel_workflow = MagicMock(return_value=True)
        harness.hard_cancel_workflow = MagicMock(return_value=True)
        # Always active -- slot never clears
        harness.is_workflow_active = MagicMock(return_value=True)
        # Very small poll budget so test terminates fast
        harness.config.terminal_status_hard_cancel_polls = 2

        task_id = 'task-kill-hard'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='restart',
            status='resolved',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # hard_cancel must be called because the slot never cleared
        harness.hard_cancel_workflow.assert_called_once_with(task_id)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Pair F — _action_teardown_tasks set + scheduler hook wiring + stamp/clear
# Step-11: RED until step-12 wires the set, installs the hook, and stamps.
# ---------------------------------------------------------------------------

class TestActionTeardownTasksSet:
    """Pair F (step-11, task 1620): Harness._action_teardown_tasks set + scheduler hook.

    RED until step-12:
      - Adds ``self._action_teardown_tasks: set[str] = set()`` to Harness.__init__.
      - Installs ``self.scheduler._suppress_blocked_write = self._action_teardown_tasks.__contains__``
        right beside the existing _on_park_stop_trip / _on_external_dep_block installs.
    """

    def test_action_teardown_tasks_is_empty_set(self, harness: Harness):
        """(a.1) Harness exposes _action_teardown_tasks as an empty set after construction.

        Fails before step-12 with AttributeError: Harness has no _action_teardown_tasks.
        """
        assert isinstance(getattr(harness, '_action_teardown_tasks', None), set), (
            'Harness._action_teardown_tasks must be a set (fails before step-12)'
        )
        assert len(harness._action_teardown_tasks) == 0, (
            '_action_teardown_tasks must start empty'
        )

    def test_scheduler_suppress_hook_wired_to_teardown_set(
        self, tmp_path: Path, mock_orch_config
    ):
        """(a.2) scheduler._suppress_blocked_write is wired to _action_teardown_tasks.__contains__:
        returns True for stamped tids, False otherwise.

        Uses the real-Scheduler construction path (Scheduler NOT patched) so that the
        hook install ``scheduler._suppress_blocked_write = _action_teardown_tasks.__contains__``
        survives on the actual Scheduler instance — the harness fixture replaces the
        scheduler after construction, which would discard the hook.

        Before step-12, _action_teardown_tasks doesn't exist → AttributeError on .add().
        After step-12, the hook is the set's __contains__: returns True/False correctly.
        """
        with (
            patch('orchestrator.harness.McpLifecycle'),
            patch('orchestrator.harness.OverrideStore'),
            patch('orchestrator.harness.BriefingAssembler'),
        ):
            h = Harness(mock_orch_config)
        # h.scheduler is a real Scheduler here (not replaced by the fixture).
        hook = h.scheduler._suppress_blocked_write
        assert hook is not None, '_suppress_blocked_write must be wired (None before step-12)'
        h._action_teardown_tasks.add('task-X')  # AttributeError before step-12
        assert hook('task-X') is True, 'hook must return True for stamped task_id'
        assert hook('task-Y') is False, 'hook must return False for unstamped task_id'


@pytest.mark.asyncio
class TestTeardownStampClear:
    """Pair F (step-11): stamp/clear discipline in _action_teardown_and_set_status.

    RED until step-12 adds the stamp before the kill and the clear in the finally block.
    """

    async def test_stamp_present_during_cancel_window(self, harness: Harness):
        """(b) task_id is in _action_teardown_tasks when cancel_workflow is called
        (stamped BEFORE the kill) and cleared after the slot clears.

        Fails before step-12 because _action_teardown_tasks doesn't exist.
        """
        stamped_at_cancel: dict[str, bool] = {}
        call_log: list[str] = []

        async def fake_set_status(tid: str, status: str) -> None:
            call_log.append(f'set_status:{tid}:{status}')

        def fake_cancel(tid: str) -> None:
            call_log.append(f'cancel:{tid}')
            # Record whether the stamp was present at cancel time.
            stamped_at_cancel[tid] = tid in harness._action_teardown_tasks

        def fake_is_active(tid: str) -> bool:
            # Active until cancel is logged (simulates immediate slot clear).
            return not any(e == f'cancel:{tid}' for e in call_log)

        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.scheduler.set_task_status = AsyncMock(side_effect=fake_set_status)
        harness.cancel_workflow = MagicMock(side_effect=fake_cancel)
        harness.hard_cancel_workflow = MagicMock()
        harness.is_workflow_active = MagicMock(side_effect=fake_is_active)
        harness.config.terminal_status_hard_cancel_polls = 3

        task_id = 'task-stamp-check'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='restart',
            status='resolved',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.cancel_workflow.assert_called_once_with(task_id)  # type: ignore[attr-defined]
        # Stamp must have been present when cancel_workflow was called.
        assert stamped_at_cancel.get(task_id) is True, (
            f'task_id must be in _action_teardown_tasks when cancel_workflow is called; '
            f'stamped_at_cancel={stamped_at_cancel}'
        )
        # After slot clears, stamp must be discarded.
        assert task_id not in harness._action_teardown_tasks, (
            '_action_teardown_tasks must be cleared once the slot clears'
        )

    async def test_park_stamp_present_during_kill_and_cleared(self, harness: Harness):
        """park action stamps before cancel and clears after slot clears."""
        stamped_at_cancel: dict[str, bool] = {}
        call_log: list[str] = []

        def fake_cancel(tid: str) -> None:
            call_log.append(f'cancel:{tid}')
            stamped_at_cancel[tid] = tid in harness._action_teardown_tasks

        def fake_is_active(tid: str) -> bool:
            return not any(e == f'cancel:{tid}' for e in call_log)

        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.scheduler.set_task_status = AsyncMock()
        harness.cancel_workflow = MagicMock(side_effect=fake_cancel)
        harness.hard_cancel_workflow = MagicMock()
        harness.is_workflow_active = MagicMock(side_effect=fake_is_active)
        harness.config.terminal_status_hard_cancel_polls = 3

        task_id = 'task-park-stamp'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='park',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        assert stamped_at_cancel.get(task_id) is True, (
            f'task_id must be stamped when cancel_workflow is called for park; '
            f'stamped_at_cancel={stamped_at_cancel}'
        )
        assert task_id not in harness._action_teardown_tasks, (
            'stamp must be cleared after slot clears'
        )

    async def test_abandon_stamp_cleared_no_crash(self, harness: Harness):
        """(c) abandon stamps (harmless — abandon→cancelled is terminal, already protected
        by scheduler terminal-exit gate) and clears cleanly; no crash.

        Fails before step-12 because _action_teardown_tasks doesn't exist.
        """
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.scheduler.set_task_status = AsyncMock()
        harness.is_workflow_active = MagicMock(return_value=False)
        harness.cancel_workflow = MagicMock()

        task_id = 'task-abandon-stamp'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='abandon',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # No crash; stamp must be cleared after completion.
        assert task_id not in harness._action_teardown_tasks, (
            '_action_teardown_tasks must not contain task_id after abandon completes'
        )
