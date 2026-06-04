"""Tests for Fix #1b stranded-blocked sweep: category, startup replay, age gate, event-at-dispatch.

Acceptance criteria:
  B10 — startup sweep re-files a stranded blocked task with category='stranded_blocked',
        agent_role='harness-stranded-blocked-reaper', severity='blocking', level=1;
        resolving that L1 flips the task blocked→pending (Fix #1a orphan-unblock path).
  B15 — a blocked task with an OPEN (pending) L1 is NOT re-filed by the sweep.
"""

from __future__ import annotations

import asyncio
import time
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
    h.scheduler.update_task = AsyncMock(return_value=True)
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


# ---------------------------------------------------------------------------
# Age-gate — _workflow_cancel_recent unit tests + sweep integration
# ---------------------------------------------------------------------------

class TestWorkflowCancelRecentHelper:
    """Unit tests for _workflow_cancel_recent(tid) helper."""

    def test_absent_stamp_returns_false(self, harness: Harness):
        """No entry in _workflow_cancel_at → _workflow_cancel_recent returns False."""
        harness._workflow_cancel_at.clear()
        assert harness._workflow_cancel_recent('no-such-task') is False  # type: ignore[attr-defined]

    def test_fresh_stamp_returns_true(self, harness: Harness):
        """Stamp = now → within grace window → True."""
        harness._workflow_cancel_at['fresh-task'] = time.monotonic()
        assert harness._workflow_cancel_recent('fresh-task') is True  # type: ignore[attr-defined]

    def test_stale_stamp_returns_false(self, harness: Harness):
        """Stamp > _RECONCILE_CANCEL_GRACE_S old → outside grace window → False."""
        grace = harness._RECONCILE_CANCEL_GRACE_S
        harness._workflow_cancel_at['stale-task'] = time.monotonic() - grace - 1.0
        assert harness._workflow_cancel_recent('stale-task') is False  # type: ignore[attr-defined]


@pytest.mark.asyncio
class TestAgeGateSweepIntegration:
    """Sweep-level: stale stamp → re-filed; recent stamp → skipped."""

    async def test_stale_cancel_stamp_allows_refiling(
        self, harness: Harness, tmp_path: Path
    ):
        """Blocked task with a STALE _workflow_cancel_at stamp (> grace) IS re-filed.

        RED under current membership gate (tid not in _workflow_cancel_at → stale
        stamp is STILL in the map, so the gate fires and no L1 is filed).
        """
        queue = EscalationQueue(tmp_path / 'esc_stale')
        task_id = 'task-stale-stamp'

        harness._escalation_queue = queue
        harness._escalation_events.clear()

        # Stale stamp: older than _RECONCILE_CANCEL_GRACE_S
        grace = harness._RECONCILE_CANCEL_GRACE_S
        harness._workflow_cancel_at[task_id] = time.monotonic() - grace - 5.0

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'blocked'}, None)
        )

        await harness._reconcile_stranded_in_progress()

        filed = queue.get_by_task(task_id, status='pending')
        assert len(filed) == 1, (
            'Expected stranded_blocked L1 for stale-stamp task; got none — '
            'age-based gate must allow re-filing when stamp is stale'
        )

    async def test_recent_cancel_stamp_suppresses_refiling(
        self, harness: Harness, tmp_path: Path
    ):
        """Blocked task with a RECENT _workflow_cancel_at stamp is NOT re-filed.

        GREEN under both old membership gate and new age-based gate (recent stamp
        is in the map → both gates suppress re-filing).
        """
        queue = EscalationQueue(tmp_path / 'esc_recent')
        task_id = 'task-recent-stamp'

        harness._escalation_queue = queue
        harness._escalation_events.clear()

        # Recent stamp: just now
        harness._workflow_cancel_at[task_id] = time.monotonic()

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'blocked'}, None)
        )

        await harness._reconcile_stranded_in_progress()

        filed = queue.get_by_task(task_id, status='pending')
        assert len(filed) == 0, (
            f'Expected no L1 for recent-stamp task; got {len(filed)} — '
            'age-based gate must suppress re-filing within grace window'
        )


# ---------------------------------------------------------------------------
# Event-at-dispatch — _register_escalation_event helper + orphan-flip suppression
# ---------------------------------------------------------------------------

class TestRegisterEscalationEventHelper:
    """Unit tests for _register_escalation_event(task_id)."""

    def test_with_queue_returns_event_and_stores_it(
        self, harness: Harness, tmp_path: Path
    ):
        """With a truthy _escalation_queue → returns asyncio.Event, stores in _escalation_events."""
        queue = EscalationQueue(tmp_path / 'esc_reg')
        harness._escalation_queue = queue
        harness._escalation_events.clear()

        event = harness._register_escalation_event('T-reg')  # type: ignore[attr-defined]

        assert isinstance(event, asyncio.Event), (
            '_register_escalation_event must return an asyncio.Event when queue is set'
        )
        assert harness._escalation_events.get('T-reg') is event, (
            'Event must be stored in _escalation_events under the task_id'
        )

    def test_without_queue_returns_none_and_does_not_register(
        self, harness: Harness
    ):
        """With _escalation_queue=None → returns None, no entry in _escalation_events."""
        harness._escalation_queue = None
        harness._escalation_events.clear()

        result = harness._register_escalation_event('T-no-queue')  # type: ignore[attr-defined]

        assert result is None
        assert 'T-no-queue' not in harness._escalation_events


@pytest.mark.asyncio
class TestEventAtDispatchOrphanFlipSuppression:
    """After dispatch-time registration, _on_escalation_resolved must NOT orphan-flip."""

    async def test_dispatch_registered_event_suppresses_orphan_flip(
        self, harness: Harness, tmp_path: Path
    ):
        """[Event-at-dispatch] A task registered at dispatch (event in _escalation_events)
        is treated as having an active workflow — Fix #1a orphan-flip gate sees
        task_id IN _escalation_events and skips the orphan path.
        """
        queue = EscalationQueue(tmp_path / 'esc_dispatch')
        harness._escalation_queue = queue
        task_id = 'T-dispatch'

        # Simulate dispatch-time registration (as would happen in the dispatch path)
        harness._register_escalation_event(task_id)  # type: ignore[attr-defined]
        assert task_id in harness._escalation_events, 'Pre-condition: event registered'

        # Build a resolved non-cascade L1 as would come from the auto-watcher
        esc = Escalation(
            id=f'esc-{task_id}-1',
            task_id=task_id,
            agent_role='harness-stranded-blocked-reaper',
            severity='blocking',
            category='stranded_blocked',
            summary='stranded blocked re-file',
            level=1,
            status='resolved',
            resolved_by='escalation-watcher-auto',
        )

        harness._loop = asyncio.get_running_loop()
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # The registered event means Fix #1a sees task_id in _escalation_events
        # → orphan-flip path is skipped → set_task_status NOT called
        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Active-workflow gate — dispatch-time event suppresses stranded_blocked re-file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestActiveWorkflowGateSuppressesReFile:
    """Sweep-level: dispatch-time registered event fires the active-workflow gate.

    harness.py:1875 — `tid not in self._escalation_events` (no active workflow).
    This is the central race-fix: registering at dispatch time means the sweep
    correctly identifies in-flight tasks and skips re-filing even before
    _run_slot has had a chance to register the event itself.
    """

    async def test_dispatch_registered_event_suppresses_sweep_refiling(
        self, harness: Harness, tmp_path: Path
    ):
        """[Active-workflow gate] A blocked task whose event is in _escalation_events
        (registered at dispatch time) must NOT have a new L1 filed by the sweep.

        This validates the central race-fix end-to-end: the gate at harness.py:1875
        must fire and skip the stranded_blocked re-file when an active workflow is
        in flight, preventing a double-flip.
        """
        queue = EscalationQueue(tmp_path / 'esc_active_wf')
        task_id = 'task-active-workflow'

        harness._escalation_queue = queue
        harness._workflow_cancel_at.clear()

        # Simulate dispatch-time registration (as happens before create_task)
        harness._register_escalation_event(task_id)  # type: ignore[attr-defined]
        assert task_id in harness._escalation_events, (
            'Pre-condition: event must be in _escalation_events before sweep'
        )

        # Task appears as blocked in the status map (e.g. it was soft-cancelled
        # or flipped externally while the workflow slot is still starting up)
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'blocked'}, None)
        )

        await harness._reconcile_stranded_in_progress()

        filed = queue.get_by_task(task_id, status='pending')
        assert len(filed) == 0, (
            f'Expected no L1 filed for active-workflow task; got {len(filed)} — '
            'harness.py:1875 active-workflow gate must suppress re-filing when '
            'task_id is in _escalation_events'
        )
