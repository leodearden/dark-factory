"""MCP-level tests for the get_queue_stats tool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server
from fused_memory.services.durable_queue import DurableWriteQueue

# ── helpers ────────────────────────────────────────────────────────────────


def _make_mock_service(get_stats_return=None):
    """Build a minimal mock MemoryService with a durable_queue."""
    svc = AsyncMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.get_stats = AsyncMock(
        return_value=get_stats_return
        or {'counts': {'completed': 8648, 'dead': 3}, 'oldest_pending_age_seconds': None}
    )
    return svc


_FAKE_STATS = {'counts': {'completed': 8648, 'dead': 3}, 'oldest_pending_age_seconds': None}


# ── step-1 tests ───────────────────────────────────────────────────────────


class TestGetQueueStatsProjectScoping:
    """get_queue_stats forwards project_id through to durable_queue.get_stats."""

    @pytest.mark.asyncio
    async def test_forwards_project_id_as_group_id(self):
        """project_id is forwarded as group_id, and the tool returns get_stats' result."""
        svc = _make_mock_service(get_stats_return=_FAKE_STATS)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_queue_stats',
            {'project_id': 'proj1'},
        )

        svc.durable_queue.get_stats.assert_called_once_with(group_id='proj1')
        assert result == _FAKE_STATS

    @pytest.mark.asyncio
    async def test_no_project_id_is_global(self):
        """Omitting project_id preserves the backward-compatible global path."""
        svc = _make_mock_service(get_stats_return=_FAKE_STATS)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_queue_stats',
            {},
        )

        svc.durable_queue.get_stats.assert_called_once_with(group_id=None)
        assert result == _FAKE_STATS


# ── step-3: real-queue acceptance test ──────────────────────────────────────


async def _poll_until_dead(
    q: DurableWriteQueue,
    *,
    group_id: str | None = None,
    expected_dead: int,
    timeout: float = 20.0,
) -> None:
    """Poll queue stats until at least *expected_dead* items have status='dead'.

    Mirrors the helper in test_durable_queue.py: avoids a fixed wall-clock
    sleep, which was a source of flakiness under xdist load.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        stats = await q.get_stats(group_id=group_id)
        if stats['counts'].get('dead', 0) >= expected_dead:
            return
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise AssertionError(
                f'Timed out waiting for {expected_dead} dead item(s) '
                f'(group_id={group_id!r}); last counts={stats["counts"]}'
            )
        await asyncio.sleep(0.05)


class TestGetQueueStatsConsistencyWithDeadLetters:
    """Reproduces the reported symptom: get_queue_stats' dead count must agree
    with get_dead_letters for the same project, against a real durable queue.
    """

    @pytest.mark.asyncio
    async def test_scoped_dead_counts_agree_with_get_dead_letters(self, tmp_path):
        execute = AsyncMock(side_effect=RuntimeError('fail'))
        q = DurableWriteQueue(
            data_dir=tmp_path / 'queue',
            execute_write=execute,
            workers_per_group=1,
            semaphore_limit=5,
            max_attempts=1,
            retry_base_seconds=0.01,
            write_timeout_seconds=2.0,
        )
        await q.initialize()
        try:
            for i in range(2):
                await q.enqueue(
                    group_id='proj-a', operation='add_episode',
                    payload={'content': f'a{i}', 'group_id': 'proj-a', 'name': f'a{i}'},
                )
            await q.enqueue(
                group_id='proj-b', operation='add_episode',
                payload={'content': 'b0', 'group_id': 'proj-b', 'name': 'b0'},
            )
            await _poll_until_dead(q, expected_dead=3)

            svc = AsyncMock()
            svc.durable_queue = q
            server = create_mcp_server(svc)

            stats_a = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj-a'},
            )
            stats_b = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj-b'},
            )
            stats_c = await server._tool_manager.call_tool(
                'get_queue_stats', {'project_id': 'proj-c'},
            )
            stats_global = await server._tool_manager.call_tool('get_queue_stats', {})

            assert stats_a['counts'].get('dead', 0) == 2
            assert stats_b['counts'].get('dead', 0) == 1
            assert stats_c['counts'].get('dead', 0) == 0
            assert stats_global['counts'].get('dead', 0) == 3

            dead_letters_a = await server._tool_manager.call_tool(
                'get_dead_letters', {'project_id': 'proj-a'},
            )
            dead_letters_b = await server._tool_manager.call_tool(
                'get_dead_letters', {'project_id': 'proj-b'},
            )
            dead_letters_c = await server._tool_manager.call_tool(
                'get_dead_letters', {'project_id': 'proj-c'},
            )

            assert stats_a['counts'].get('dead', 0) == dead_letters_a['counts']['durable_queue']
            assert stats_b['counts'].get('dead', 0) == dead_letters_b['counts']['durable_queue']
            assert stats_c['counts'].get('dead', 0) == dead_letters_c['counts']['durable_queue']
        finally:
            await q.close()
