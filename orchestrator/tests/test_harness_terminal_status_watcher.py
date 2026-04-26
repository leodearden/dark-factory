"""Tests for ``Harness._scan_for_terminal_active_tasks`` — Step 5 of
zombie-escalation fix.

The watcher is a simple poll (interval 30 s by default) that asks fused-
memory ``get_statuses`` for the set of active workflow ids and cancels any
whose status is in ``TERMINAL_STATUSES``.  We test the scan pass directly
so the test is deterministic; the loop wrapper (``asyncio.sleep`` +
exception swallow) is identical in shape to the orphan-L0 reaper loop and
covered there.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from orchestrator.harness import Harness


@pytest.fixture
def harness() -> Harness:
    h = Harness.__new__(Harness)
    h._workflow_cancel_events = {}
    h.scheduler = type('S', (), {})()
    return h


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
