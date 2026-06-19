"""Tests for ``Harness._scan_for_terminal_active_tasks`` — Step 5 of
zombie-escalation fix.

The watcher is a simple poll (interval 30 s by default) that asks fused-
memory ``get_statuses`` for the set of active workflow ids and cancels any
whose status is in ``TERMINAL_STATUSES``.  We test the scan pass directly
so the test is deterministic; the loop wrapper (``asyncio.sleep`` +
exception swallow) is identical in shape to the orphan-L0 reaper loop and
covered there.

Task 1491 (ITEM 2) adds a consecutive-poll threshold: below threshold the
scan soft-cancels (existing behaviour); at/above threshold it hard-cancels
via ``hard_cancel_workflow`` and logs a WARNING once.  ``_terminal_cancel_counts``
tracks consecutive terminal polls per task and is cleared when the task
drops out of the active set or its status is no longer terminal.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import _init_harness_state_for_test

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness


@pytest.fixture
def harness() -> Harness:
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)  # task 1449: initialise digest counters
    h._workflow_cancel_events = {}
    # cancel_workflow stamps wall-clock for the reconcile sweep grace
    # window (R3 race guard) — match the real Harness attribute set.
    h._workflow_cancel_at = {}
    # task 1491: hard-cancel registry and consecutive-poll counts.
    h._workflow_slot_tasks = {}
    h._terminal_cancel_counts = {}
    # High threshold so existing tests never trigger hard-cancel path.
    h.config = OrchestratorConfig(terminal_status_hard_cancel_polls=100)
    h.scheduler = type('S', (), {})()  # type: ignore[assignment]
    return h


@pytest.fixture
def harness_low_threshold(harness: Harness) -> Harness:
    """Fixture with threshold=2 for hard-cancel escalation tests."""
    harness.config = OrchestratorConfig(terminal_status_hard_cancel_polls=2)
    return harness


@pytest.mark.asyncio
async def test_scan_no_active_tasks_is_noop(harness):
    cancelled = await harness._scan_for_terminal_active_tasks()
    assert cancelled == 0


@pytest.mark.asyncio
async def test_scan_cancels_done_task(harness):
    ev_done = asyncio.Event()
    ev_running = asyncio.Event()
    harness._workflow_cancel_events['1'] = ev_done
    harness._workflow_cancel_events['2'] = ev_running
    harness.scheduler.get_statuses = AsyncMock(
        return_value=({'1': 'done', '2': 'in-progress'}, None),
    )

    cancelled = await harness._scan_for_terminal_active_tasks()

    assert cancelled == 1
    assert ev_done.is_set() is True
    assert ev_running.is_set() is False


@pytest.mark.asyncio
async def test_scan_cancels_cancelled_task(harness):
    ev = asyncio.Event()
    harness._workflow_cancel_events['7'] = ev
    harness.scheduler.get_statuses = AsyncMock(
        return_value=({'7': 'cancelled'}, None),
    )

    cancelled = await harness._scan_for_terminal_active_tasks()

    assert cancelled == 1
    assert ev.is_set() is True


@pytest.mark.asyncio
async def test_scan_swallows_get_statuses_error(harness):
    ev = asyncio.Event()
    harness._workflow_cancel_events['1'] = ev
    harness.scheduler.get_statuses = AsyncMock(
        return_value=({}, RuntimeError('mcp down')),
    )

    cancelled = await harness._scan_for_terminal_active_tasks()

    assert cancelled == 0
    assert ev.is_set() is False


@pytest.mark.asyncio
async def test_scan_skips_blocked_and_pending(harness):
    """Only ``done``/``cancelled`` count as terminal — ``blocked`` and
    ``pending`` are not (a blocked task may still be retried)."""
    ev_blocked = asyncio.Event()
    ev_pending = asyncio.Event()
    harness._workflow_cancel_events['1'] = ev_blocked
    harness._workflow_cancel_events['2'] = ev_pending
    harness.scheduler.get_statuses = AsyncMock(
        return_value=({'1': 'blocked', '2': 'pending'}, None),
    )

    cancelled = await harness._scan_for_terminal_active_tasks()

    assert cancelled == 0
    assert ev_blocked.is_set() is False
    assert ev_pending.is_set() is False


# ---------------------------------------------------------------------------
# task 1491 ITEM 2: threshold / hard-cancel escalation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_terminal_poll_soft_cancels_and_increments_count(
    harness_low_threshold: Harness,
):
    """Poll 1 (count=1 < threshold=2): soft cancel (event set), slot task
    NOT hard-cancelled, ``_terminal_cancel_counts[tid] == 1``.

    RED: without count tracking the count stays 0 / missing → assertion fails.
    """
    ev = asyncio.Event()
    harness_low_threshold._workflow_cancel_events['5'] = ev
    slot_task = asyncio.ensure_future(asyncio.sleep(3600))
    harness_low_threshold._workflow_slot_tasks['5'] = slot_task
    harness_low_threshold.scheduler.get_statuses = AsyncMock(
        return_value=({'5': 'done'}, None),
    )

    cancelled = await harness_low_threshold._scan_for_terminal_active_tasks()

    # Soft-cancel: returns 1, event set.
    assert cancelled == 1
    assert ev.is_set() is True

    # Slot task must NOT be hard-cancelled yet (count=1 < threshold=2).
    assert not slot_task.cancelled()
    slot_task.cancel()  # cleanup

    # Count must be incremented to 1.
    assert harness_low_threshold._terminal_cancel_counts.get('5') == 1


@pytest.mark.asyncio
async def test_second_terminal_poll_hard_cancels_at_threshold(
    harness_low_threshold: Harness,
    caplog,
):
    """Poll 2 (count=2 >= threshold=2): hard cancel — slot task.cancel()
    requested — and WARNING logged exactly once at the threshold crossing.

    RED: without count tracking / hard-cancel branch, slot task is never
    cancelled and WARNING is never logged → assertions fail.
    """
    ev = asyncio.Event()
    harness_low_threshold._workflow_cancel_events['6'] = ev
    slot_task = asyncio.ensure_future(asyncio.sleep(3600))
    harness_low_threshold._workflow_slot_tasks['6'] = slot_task
    harness_low_threshold.scheduler.get_statuses = AsyncMock(
        return_value=({'6': 'cancelled'}, None),
    )

    with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
        # Poll 1: below threshold → soft cancel.
        await harness_low_threshold._scan_for_terminal_active_tasks()
        assert not slot_task.cancelled()

        # Poll 2: at threshold → hard cancel.
        cancelled = await harness_low_threshold._scan_for_terminal_active_tasks()

    # Hard-cancel: returns 1, slot task cancelled.
    assert cancelled == 1
    with pytest.raises(asyncio.CancelledError):
        await slot_task
    assert slot_task.cancelled()

    # WARNING must have been logged exactly once (at threshold crossing).
    hard_cancel_logs = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'hard' in r.message.lower()
    ]
    assert len(hard_cancel_logs) == 1, (
        f'Expected exactly 1 hard-cancel WARNING, got {len(hard_cancel_logs)}: '
        f'{[r.message for r in hard_cancel_logs]}'
    )


@pytest.mark.asyncio
async def test_scan_stops_reprocessing_after_task_drops_out(
    harness_low_threshold: Harness,
    caplog,
):
    """Once a task's entry is absent from ``_workflow_cancel_events``,
    subsequent scans return 0 and do not re-log the hard-cancel WARNING.

    This exercises the scan's skip-when-absent behaviour: it does *not*
    exercise the ``_run_slot`` finally-block cleanup that would remove the
    entry in production (that cleanup is a separate contract tested via
    ``_run_slot`` directly).
    """
    ev = asyncio.Event()
    harness_low_threshold._workflow_cancel_events['7'] = ev
    slot_task = asyncio.ensure_future(asyncio.sleep(3600))
    harness_low_threshold._workflow_slot_tasks['7'] = slot_task
    harness_low_threshold.scheduler.get_statuses = AsyncMock(
        return_value=({'7': 'done'}, None),
    )

    with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
        # Polls 1 + 2 reach threshold → hard cancel.
        await harness_low_threshold._scan_for_terminal_active_tasks()
        await harness_low_threshold._scan_for_terminal_active_tasks()

        # Simulate the slot task's finally block cleaning up the registry.
        harness_low_threshold._workflow_cancel_events.pop('7', None)
        harness_low_threshold._workflow_slot_tasks.pop('7', None)
        harness_low_threshold._terminal_cancel_counts.pop('7', None)

        warning_count_before = len([
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'hard' in r.message.lower()
        ])

        # Poll 3: task gone from active set → 0 cancelled.
        result = await harness_low_threshold._scan_for_terminal_active_tasks()

    assert result == 0
    warning_count_after = len([
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'hard' in r.message.lower()
    ])
    # No additional WARNING logged after the task dropped out.
    assert warning_count_after == warning_count_before


@pytest.mark.asyncio
async def test_terminal_cancel_count_cleared_on_non_terminal_status(
    harness_low_threshold: Harness,
):
    """``_terminal_cancel_counts[tid]`` is cleared when the status is no
    longer terminal (e.g. transitions from 'done' to 'blocked').

    RED: without the count-reset logic the count persists → assertion fails.
    """
    ev = asyncio.Event()
    harness_low_threshold._workflow_cancel_events['10'] = ev
    # Pre-seed a stale count (as if one terminal poll already ran).
    harness_low_threshold._terminal_cancel_counts['10'] = 1

    # Next scan sees non-terminal status → should clear the count.
    harness_low_threshold.scheduler.get_statuses = AsyncMock(
        return_value=({'10': 'blocked'}, None),
    )

    cancelled = await harness_low_threshold._scan_for_terminal_active_tasks()

    assert cancelled == 0
    assert '10' not in harness_low_threshold._terminal_cancel_counts, (
        f"Expected count cleared on non-terminal status, got "
        f"{harness_low_threshold._terminal_cancel_counts.get('10')!r}"
    )


# ---------------------------------------------------------------------------
# Step-5 RED: _scan_for_terminal_active_tasks WARNING on get_statuses error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_emits_warning_when_get_statuses_fails(harness, caplog):
    """get_statuses error → returns 0 AND WARNING emitted at 'orchestrator.harness'.

    Pre-fix: `if error is not None: return 0` is SILENT — no WARNING fires.
    After fix: WARNING at 'orchestrator.harness' naming the failure.
    """
    ev = asyncio.Event()
    harness._workflow_cancel_events['7'] = ev
    harness.scheduler.get_statuses = AsyncMock(
        return_value=({}, RuntimeError('mcp down')),
    )

    with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
        result = await harness._scan_for_terminal_active_tasks()

    assert result == 0, f'Expected 0, got {result!r}'
    assert any(
        r.levelno >= logging.WARNING
        for r in caplog.records
    ), f'Expected WARNING; got: {[r.message for r in caplog.records]!r}'
