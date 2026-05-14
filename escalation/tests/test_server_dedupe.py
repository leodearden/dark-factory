"""Tests for escalation deduplication via the MCP server tools.

Mirrors the FastMCP unit-test pattern from test_release_workflow.py:
    tool = await server.get_tool('escalate_blocker')
    result = await tool.fn(...)

All tests use tmp_path-backed EscalationQueue so they are fully isolated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from escalation.dedupe import DedupeConfig
from escalation.queue import EscalationQueue
from escalation.server import create_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(queue: EscalationQueue, dedupe_config: DedupeConfig | None = None):
    return create_server(queue, dedupe_config=dedupe_config)


async def _blocker(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_blocker')
    # escalate_blocker is a sync tool — tool.fn() returns dict directly
    return tool.fn(**kwargs)


async def _info(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_info')
    # escalate_info is a sync tool — tool.fn() returns dict directly
    return tool.fn(**kwargs)


def _queue_root_files(queue: EscalationQueue) -> list[Path]:
    """Return all esc-*.json files in the queue root (excludes archive)."""
    return sorted(queue.queue_dir.glob('esc-*.json'))


# ---------------------------------------------------------------------------
# TestEscalateBlockerDedupe
# ---------------------------------------------------------------------------


class TestEscalateBlockerDedupe:
    """escalate_blocker deduplication — two infra_issue calls with similar summaries."""

    @pytest.mark.asyncio
    async def test_first_call_creates_file(self, tmp_path: Path):
        """(a) First escalate_blocker creates exactly one esc-*.json."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        result = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )

        assert result['status'] == 'queued'
        assert result['action'] == 'terminate_cleanly'
        assert 'id' in result

        files = _queue_root_files(queue)
        assert len(files) == 1, f'Expected exactly 1 file, got: {files}'

    @pytest.mark.asyncio
    async def test_second_call_dedupes_to_parent(self, tmp_path: Path):
        """(b) Second call with similar summary dedupes: no new file, parent bumped."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        # Different tail / casing — same first 3 tokens after normalisation
        second = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id
        assert second['action'] == 'terminate_cleanly'

        # Still exactly one file in queue root
        files = _queue_root_files(queue)
        assert len(files) == 1, f'Expected 1 file after dedupe, got: {files}'

        # Parent on disk has dedupe_count == 1 and the child id in dedupe_children
        from escalation.models import Escalation
        parent = Escalation.from_json(files[0].read_text())
        assert parent.dedupe_count == 1
        assert len(parent.dedupe_children) == 1

    @pytest.mark.asyncio
    async def test_notify_callback_fires_once_for_parent_not_for_dedupe(self, tmp_path: Path):
        """(c) notify_callback fires for the parent submit but NOT for the deduped call."""
        queue = EscalationQueue(tmp_path / 'esc')
        server = _make_server(queue)

        fired_ids: list[str] = []
        queue.set_notify_callback(lambda esc: fired_ids.append(esc.id))

        first = await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        await _blocker(
            server,
            task_id='42',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        # Callback fires exactly once — only for the original submit
        assert fired_ids == [parent_id], (
            f'Expected notify callback to fire exactly once for parent; '
            f'got: {fired_ids}'
        )
