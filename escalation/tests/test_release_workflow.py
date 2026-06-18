"""Tests for the ``release_workflow`` MCP tool — Step 6 of zombie-escalation
fix.

The tool is a thin wrapper over ``Harness.cancel_workflow`` /
``Harness.is_workflow_active``.  We exercise it by reaching into the
FastMCP server's tool registry and calling the wrapped function directly,
matching the pattern used by other MCP-tool unit tests in this repo.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server


class _FakeScheduler:
    """Minimal stand-in for ``Harness.scheduler`` (the park-to-blocked path)."""

    def __init__(self, status: str | None = None) -> None:
        # Default None → no park unless a test sets a status.
        self.get_status = AsyncMock(return_value=status)
        self.set_task_status = AsyncMock()


class _FakeHarness:
    """Minimal stand-in for ``orchestrator.harness.Harness``."""

    def __init__(self, status: str | None = None) -> None:
        self.events: dict[str, asyncio.Event] = {}
        self.scheduler = _FakeScheduler(status)

    def is_workflow_active(self, task_id: str) -> bool:
        return task_id in self.events

    def cancel_workflow(self, task_id: str) -> bool:
        ev = self.events.get(task_id)
        if ev is None:
            return False
        ev.set()
        return True


async def _call_release(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('release_workflow')
    return await tool.fn(**kwargs)


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.mark.asyncio
async def test_no_harness_reports_standalone(queue):
    server = create_server(queue, harness=None)
    result = await _call_release(server, task_id='1', timeout_secs=1)
    assert result['was_active'] is False
    assert result['released'] is False
    assert 'error' in result


@pytest.mark.asyncio
async def test_idle_task_returns_immediately(queue):
    harness = _FakeHarness()
    server = create_server(queue, harness=harness)
    result = await _call_release(server, task_id='nope', timeout_secs=1)
    assert result['was_active'] is False
    assert result['released'] is False
    assert result['slot_cleared'] is True


@pytest.mark.asyncio
async def test_active_task_clears_after_event_set(queue):
    harness = _FakeHarness()
    ev = asyncio.Event()
    harness.events['42'] = ev

    # Background task: when cancel_workflow sets the event, simulate the
    # workflow exiting by removing the slot from the harness registry.
    async def _watcher():
        await ev.wait()
        await asyncio.sleep(0.05)
        harness.events.pop('42', None)

    server = create_server(queue, harness=harness)
    asyncio.create_task(_watcher())
    result = await _call_release(server, task_id='42', timeout_secs=2)

    assert result['was_active'] is True
    assert result['released'] is True
    assert result['slot_cleared'] is True


@pytest.mark.asyncio
async def test_active_task_times_out_when_slot_does_not_clear(queue):
    """If the workflow doesn't exit within ``timeout_secs``, ``slot_cleared``
    is reported False — the caller should investigate.
    """
    harness = _FakeHarness()
    harness.events['42'] = asyncio.Event()  # never popped

    server = create_server(queue, harness=harness)
    result = await _call_release(server, task_id='42', timeout_secs=1)

    assert result['was_active'] is True
    assert result['released'] is True
    assert result['slot_cleared'] is False
    # Slot never cleared → must NOT park (parking would race a live workflow).
    harness.scheduler.set_task_status.assert_not_awaited()
    assert result['parked'] is None


@pytest.mark.asyncio
async def test_in_progress_task_parked_as_blocked(queue):
    """Slot clears while the task is still 'in-progress' → parked as 'blocked'.

    This is the /unblock shape: an escalated task whose agent was paused.
    Parking to the reaper-immune 'blocked' state protects the worktree from
    the stranded-in-progress sweep while the human finishes the merge.
    """
    harness = _FakeHarness(status='in-progress')
    ev = asyncio.Event()
    harness.events['42'] = ev

    async def _watcher():
        await ev.wait()
        await asyncio.sleep(0.05)
        harness.events.pop('42', None)

    server = create_server(queue, harness=harness)
    asyncio.create_task(_watcher())
    result = await _call_release(server, task_id='42', timeout_secs=2)

    assert result['slot_cleared'] is True
    assert result['parked'] == 'blocked'
    harness.scheduler.set_task_status.assert_awaited_once_with('42', 'blocked')


@pytest.mark.asyncio
async def test_terminal_task_not_parked(queue):
    """Slot clears but the task already went terminal ('done') → no park.

    The normal out-of-band-done path: nothing to protect, and we must not
    drag a done task back to 'blocked'.
    """
    harness = _FakeHarness(status='done')
    ev = asyncio.Event()
    harness.events['42'] = ev

    async def _watcher():
        await ev.wait()
        await asyncio.sleep(0.05)
        harness.events.pop('42', None)

    server = create_server(queue, harness=harness)
    asyncio.create_task(_watcher())
    result = await _call_release(server, task_id='42', timeout_secs=2)

    assert result['slot_cleared'] is True
    assert result['parked'] is None
    harness.scheduler.set_task_status.assert_not_awaited()


# ---------------------------------------------------------------------------
# Real-workflow-exit end-to-end (AC4): release_workflow drives a REAL
# Harness._run_slot exit via WorkflowOutcome.SOFT_CANCELLED.  The slot is
# cleared by the actual _run_slot finally block, not by a hand-flipped
# is_workflow_active flag.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_workflow_real_slot_exit_parks_blocked(
    queue: EscalationQueue,
) -> None:
    """AC4: release_workflow clears the slot via real _run_slot plumbing.

    A bypass-__init__ Harness (harness_for_run_slot pattern) drives _run_slot
    with a patched TaskWorkflow whose run() awaits the real harness-created
    cancel_event and returns WorkflowOutcome.SOFT_CANCELLED.

    release_workflow sets the cancel_event, the slot clears through the real
    _run_slot finally block, and the tool then parks the task as 'blocked'.

    Asserts:
    - was_active=True (slot was registered before release_workflow call)
    - slot_cleared=True (REAL _run_slot finally popped _workflow_cancel_events)
    - parked='blocked' (scheduler.get_status='in-progress' → parked to blocked)
    - scheduler.set_task_status called once with (tid, 'blocked')
    """
    from unittest.mock import MagicMock, patch  # noqa: PLC0415

    from orchestrator.harness import Harness  # type: ignore[reportMissingImports]
    from orchestrator.scheduler import TaskAssignment  # type: ignore[reportMissingImports]
    from orchestrator.workflow import WorkflowOutcome  # type: ignore[reportMissingImports]

    tid = '42'

    # ── Build bypass-__init__ Harness (mirrors harness_for_run_slot fixture) ──
    h: Harness = Harness.__new__(Harness)
    h._init_digest_state()

    # cancel / slot registries
    h._workflow_cancel_events = {}  # type: ignore[assignment]
    h._workflow_cancel_at = {}  # type: ignore[assignment]
    h._workflow_slot_tasks = {}  # type: ignore[assignment]
    h._terminal_cancel_counts = {}  # type: ignore[assignment]
    h._escalation_events = {}  # type: ignore[assignment]
    h._escalation_queue = None  # type: ignore[assignment]

    # run-slot supporting state
    h._recovered_plans = {}  # type: ignore[assignment]
    h._recovered_sessions = {}  # type: ignore[assignment]
    h._preserved_worktrees = set()  # type: ignore[assignment]
    h.event_store = None  # type: ignore[assignment]

    # TaskWorkflow constructor kwarg sources (all None; TaskWorkflow is patched)
    h.config = None  # type: ignore[assignment]
    h.git_ops = None  # type: ignore[assignment]
    h.briefing = None  # type: ignore[assignment]
    h.mcp = None  # type: ignore[assignment]
    h.usage_gate = None  # type: ignore[assignment]
    h._merge_queue = None  # type: ignore[assignment]
    h._merge_worker = None  # type: ignore[assignment]
    h._merge_inflight_registry = None  # type: ignore[assignment]
    h.cost_store = None  # type: ignore[assignment]

    # _collect_done_reports / _apply_retry_cap
    h._run_store = None  # type: ignore[assignment]
    h._run_id = None  # type: ignore[assignment]
    h.review_checkpoint = None  # type: ignore[assignment]

    # Scheduler: get_status='in-progress' → release_workflow parks as 'blocked'
    h.scheduler = MagicMock()
    h.scheduler.get_status = AsyncMock(return_value='in-progress')
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.release = MagicMock()
    h.scheduler.carries_substrate_probe = MagicMock(return_value=False)

    assignment = TaskAssignment(  # type: ignore[reportMissingImports]
        task_id=tid,
        task={'id': tid, 'title': 'real slot exit test'},
        modules=[],
    )
    sem = asyncio.Semaphore(0)

    # ── Patch TaskWorkflow to await the real cancel_event then return SOFT_CANCELLED ──
    cancel_event_holder: list[asyncio.Event] = []

    def _make_mock_workflow(*args, **kwargs):
        ce = kwargs.get('cancel_event')
        if ce is not None:
            cancel_event_holder.append(ce)
        mock_wf = MagicMock()
        mock_wf._steward = None

        async def _run_soft_cancelled():
            # Block until release_workflow triggers the cancel event.
            if cancel_event_holder:
                await cancel_event_holder[-1].wait()
            return WorkflowOutcome.SOFT_CANCELLED

        mock_wf.run = _run_soft_cancelled
        return mock_wf

    with patch('orchestrator.harness.TaskWorkflow', side_effect=_make_mock_workflow):
        # Wire harness into the escalation server so release_workflow can reach it.
        server = create_server(queue, harness=h)

        # Start the slot task — it blocks inside mock run() awaiting cancel_event.
        slot_task = asyncio.create_task(h._run_slot(assignment, sem))

        # Wait for _run_slot to register the cancel_event in _workflow_cancel_events.
        for _ in range(100):
            if tid in h._workflow_cancel_events:
                break
            await asyncio.sleep(0.01)
        assert tid in h._workflow_cancel_events, (
            '_run_slot did not register cancel_event within 1 s'
        )

        # Call the real release_workflow tool.
        result = await _call_release(server, task_id=tid, timeout_secs=5)

        # Wait for the slot task to finish (real _run_slot finally must have run).
        done, _ = await asyncio.wait({slot_task}, timeout=5.0)
        assert slot_task in done, 'slot_task did not complete within 5 s'

    # ── Assertions ──
    assert result['was_active'] is True, (
        f'Expected was_active=True, got {result!r}'
    )
    assert result['released'] is True, (
        f'Expected released=True, got {result!r}'
    )
    assert result['slot_cleared'] is True, (
        f'Expected slot_cleared=True (real _run_slot finally cleared the slot), '
        f'got {result!r}'
    )
    assert result['parked'] == 'blocked', (
        f'Expected parked="blocked" (scheduler.get_status="in-progress"), '
        f'got {result!r}'
    )
    # The real scheduler.set_task_status must have been called once with 'blocked'.
    h.scheduler.set_task_status.assert_awaited_once_with(tid, 'blocked')
    # No-requeue contract: SOFT_CANCELLED exits the slot with requeued=False.
    # A regression that re-treated SOFT_CANCELLED as requeue-triggering would call
    # scheduler.release(tid, requeued=True) — this assertion catches that.
    h.scheduler.release.assert_called_once_with(tid, requeued=False)
    # And the slot is truly cleared from the harness registry.
    assert not h.is_workflow_active(tid), (
        'Slot must be cleared from _workflow_cancel_events by _run_slot finally'
    )
