"""Tests for Fix #1b stranded-blocked sweep: category, startup replay, age gate, event-at-dispatch.

Acceptance criteria:
  B10 — startup sweep re-files a stranded blocked task with category='stranded_blocked',
        agent_role='harness-stranded-blocked-reaper', severity='blocking', level=1;
        resolving that L1 flips the task blocked→pending (Fix #1a orphan-unblock path).
  B15 — a blocked task with an OPEN (pending) L1 is NOT re-filed by the sweep.
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
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness with mocked internals for stranded-blocked sweep unit testing."""
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.get_task = AsyncMock(return_value=None)
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()
    # Provide dispatched/lock_table for mid_run path (not used in startup tests)
    h.scheduler._dispatched = set()
    h.scheduler.lock_table = MagicMock()
    h.scheduler.lock_table._held = set()

    # Wire git_ops mocks so blocked tasks fall through to Fix #1b
    h.git_ops.is_ancestor = AsyncMock(return_value=False)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)

    # Enable stranded-blocked escalation
    h.config.stranded_blocked_escalate_enabled = True

    return h


def _make_resolved_l1(queue: EscalationQueue, esc_id: str, task_id: str) -> Escalation:
    """Submit an L1, then resolve it (no callback set) to put it in archive."""
    esc = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='steward',
        severity='blocking',
        category='design_concern',
        summary='prior blocking issue (now resolved)',
        level=1,
    )
    queue.submit(esc)
    queue.resolve(esc_id, 'resolved out-of-band')
    return esc


def _make_pending_l1(queue: EscalationQueue, esc_id: str, task_id: str) -> Escalation:
    """Submit an L1 and leave it pending (open)."""
    esc = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='steward',
        severity='blocking',
        category='design_concern',
        summary='open blocking issue',
        level=1,
    )
    queue.submit(esc)
    return esc


# ---------------------------------------------------------------------------
# B10 — restart-window regression test + flip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestB10StartupSweepReFile:
    """B10: startup sweep re-files a stranded blocked task + Fix #1a flips it."""

    async def test_b10_new_l1_filed_with_stranded_blocked_category(
        self, harness: Harness, tmp_path: Path
    ):
        """[B10] Blocked task with RESOLVED L1 and no active workflow →
        _reconcile_stranded_in_progress() files a new L1 with
        category='stranded_blocked', agent_role='harness-stranded-blocked-reaper',
        severity='blocking', level=1.

        RED with current code (files 'task_failure' → assertion on
        'stranded_blocked' fails).
        """
        queue = EscalationQueue(tmp_path / 'esc_b10')
        task_id = 'task-b10'

        # Wire queue into harness
        harness._escalation_queue = queue

        # Pre-seed a RESOLVED L1 (no pending escalation)
        _make_resolved_l1(queue, 'esc-b10-prior', task_id)

        # Ensure no active workflow / recent cancel stamp
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()

        # Blocked task in status map
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'blocked'}, None)
        )

        await harness._reconcile_stranded_in_progress()

        filed = queue.get_by_task(task_id, status='pending')
        assert len(filed) == 1, (
            f'Expected exactly 1 pending L1 filed for {task_id}; got {len(filed)}'
        )
        new_l1 = filed[0]
        assert new_l1.category == 'stranded_blocked', (
            f'Expected category="stranded_blocked", got "{new_l1.category}" — '
            'Fix #1b must use stranded_blocked (not task_failure)'
        )
        assert new_l1.agent_role == 'harness-stranded-blocked-reaper'
        assert new_l1.severity == 'blocking'
        assert new_l1.level == 1

    async def test_b10_resolving_new_l1_flips_task_to_pending(
        self, harness: Harness, tmp_path: Path
    ):
        """[B10 flip] Resolving the stranded_blocked L1 (non-cascade) triggers
        Fix #1a orphan-unblock → set_task_status(task_id, 'pending').
        """
        queue = EscalationQueue(tmp_path / 'esc_b10_flip')
        task_id = 'task-b10-flip'

        harness._escalation_queue = queue
        _make_resolved_l1(queue, 'esc-b10-flip-prior', task_id)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'blocked'}, None)
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        await harness._reconcile_stranded_in_progress()

        filed = queue.get_by_task(task_id, status='pending')
        assert len(filed) == 1

        new_l1 = filed[0]

        # Wire harness for event-driven flip
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)

        # Resolve as the auto-watcher would (non-cascade)
        queue.resolve(new_l1.id, 'resolved by escalation-watcher-auto')
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_with(task_id, 'pending')  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# B15 — open L1 guard: blocked task with pending escalation is NOT re-filed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestB15OpenL1Guard:
    """B15: blocked task with an existing pending L1 is not re-filed."""

    async def test_b15_open_l1_suppresses_refiling(
        self, harness: Harness, tmp_path: Path
    ):
        """[B15] Blocked task with OPEN (pending) L1 → sweep leaves it alone
        (pending-escalation self-dedup gate fires).
        """
        queue = EscalationQueue(tmp_path / 'esc_b15')
        task_id = 'task-b15'

        harness._escalation_queue = queue
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()

        # Leave an OPEN (pending) L1 in the queue
        open_l1 = _make_pending_l1(queue, 'esc-b15-open', task_id)

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'blocked'}, None)
        )

        await harness._reconcile_stranded_in_progress()

        # Still exactly 1 pending escalation (the original open one); no second filed
        pending = queue.get_by_task(task_id, status='pending')
        assert len(pending) == 1, (
            f'Expected 1 pending L1 (the open one); got {len(pending)} — '
            'pending-escalation gate must suppress re-filing'
        )
        assert pending[0].id == open_l1.id, 'The surviving L1 must be the original open one'
