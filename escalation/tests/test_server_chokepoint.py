"""Tests for terminal-task chokepoint in escalation MCP handlers.

Exercises the auto-resolve path: when the target task is already 'done' or
'cancelled', escalate_blocker / escalate_info should auto-resolve the
escalation immediately rather than leaving it pending.

Uses the same async FastMCP unit-test pattern as test_release_workflow.py:
    tool = await server.get_tool('escalate_blocker')
    result = await tool.fn(...)

and the same tmp_path isolation as test_server_dedupe.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON_KWARGS: dict[str, Any] = {
    'task_id': 'task-999',
    'agent_role': 'implementer',
    'category': 'scope_violation',
    'summary': 'target task already done',
}


async def _make_lookup(status: str | None):
    """Build a simple async stub that always returns *status*."""
    calls: list[str] = []

    async def _lookup(task_id: str) -> str | None:
        calls.append(task_id)
        return status

    _lookup.calls = calls  # type: ignore[attr-defined]
    return _lookup


async def _blocker(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_blocker')
    return await tool.fn(**kwargs)


async def _info(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_info')
    return await tool.fn(**kwargs)


def _queue_root_files(queue: EscalationQueue) -> list[Path]:
    """Return all esc-*.json files in the queue root (excludes archive)."""
    return sorted(queue.queue_dir.glob('esc-*.json'))


# ---------------------------------------------------------------------------
# Step-1 RED tests: terminal auto-resolve (done / cancelled)
# ---------------------------------------------------------------------------


class TestTerminalAutoResolve:
    """escalate_blocker / escalate_info auto-resolve when task is terminal."""

    @pytest.mark.asyncio
    async def test_blocker_done_task_returns_resolved(self, tmp_path: Path):
        """escalate_blocker against a 'done' task returns status='resolved'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result['status'] == 'resolved', f"Expected 'resolved', got: {result}"

    @pytest.mark.asyncio
    async def test_blocker_done_task_resolution_text(self, tmp_path: Path):
        """escalate_blocker response includes auto-resolve resolution text."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert 'resolution' in result, f"No 'resolution' key in: {result}"
        assert 'auto-resolved' in result['resolution'], f"Unexpected resolution: {result['resolution']}"
        assert 'done' in result['resolution'], f"Status not in resolution: {result['resolution']}"

    @pytest.mark.asyncio
    async def test_blocker_done_task_resolved_by(self, tmp_path: Path):
        """escalate_blocker response includes resolved_by='escalation-mcp-pre-submit-check'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result.get('resolved_by') == 'escalation-mcp-pre-submit-check', (
            f"Expected 'escalation-mcp-pre-submit-check', got: {result.get('resolved_by')}"
        )

    @pytest.mark.asyncio
    async def test_blocker_done_task_action_terminate(self, tmp_path: Path):
        """escalate_blocker response retains action='terminate_cleanly' on auto-resolve path."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        assert result.get('action') == 'terminate_cleanly', (
            f"Expected 'terminate_cleanly', got: {result.get('action')}"
        )

    @pytest.mark.asyncio
    async def test_blocker_done_task_on_disk_resolved(self, tmp_path: Path):
        """After auto-resolve, the escalation file on disk has status='resolved'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('done')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _blocker(server, **_COMMON_KWARGS)

        esc_id = result['id']
        esc = queue.get(esc_id)
        assert esc is not None, f"Escalation {esc_id} not found in queue"
        assert esc.status == 'resolved', f"Expected 'resolved', got: {esc.status}"

    @pytest.mark.asyncio
    async def test_info_cancelled_task_returns_resolved(self, tmp_path: Path):
        """escalate_info against a 'cancelled' task returns status='resolved'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert result['status'] == 'resolved', f"Expected 'resolved', got: {result}"

    @pytest.mark.asyncio
    async def test_info_cancelled_task_resolution_text(self, tmp_path: Path):
        """escalate_info response includes 'cancelled' in resolution text."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert 'resolution' in result
        assert 'auto-resolved' in result['resolution']
        assert 'cancelled' in result['resolution']

    @pytest.mark.asyncio
    async def test_info_cancelled_task_resolved_by(self, tmp_path: Path):
        """escalate_info response includes resolved_by='escalation-mcp-pre-submit-check'."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert result.get('resolved_by') == 'escalation-mcp-pre-submit-check'

    @pytest.mark.asyncio
    async def test_info_has_no_action_key(self, tmp_path: Path):
        """escalate_info response must NOT include 'action' key (blocker-only contract)."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        assert 'action' not in result, f"Unexpected 'action' key in info result: {result}"

    @pytest.mark.asyncio
    async def test_info_cancelled_task_on_disk_resolved(self, tmp_path: Path):
        """After auto-resolve via escalate_info, the on-disk record is resolved."""
        queue = EscalationQueue(tmp_path / 'esc')
        lookup = await _make_lookup('cancelled')
        server = create_server(queue, task_status_lookup=lookup)

        result = await _info(server, **_COMMON_KWARGS)

        esc_id = result['id']
        esc = queue.get(esc_id)
        assert esc is not None
        assert esc.status == 'resolved'
