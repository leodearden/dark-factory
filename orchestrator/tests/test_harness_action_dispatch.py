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
import logging
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
