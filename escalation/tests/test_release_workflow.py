"""Tests for the ``release_workflow`` MCP tool — Step 6 of zombie-escalation
fix.

The tool is a thin wrapper over ``Harness.cancel_workflow`` /
``Harness.is_workflow_active``.  We exercise it by reaching into the
FastMCP server's tool registry and calling the wrapped function directly,
matching the pattern used by other MCP-tool unit tests in this repo.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server


class _FakeHarness:
    """Minimal stand-in for ``orchestrator.harness.Harness``."""

    def __init__(self) -> None:
        self.events: dict[str, asyncio.Event] = {}

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
