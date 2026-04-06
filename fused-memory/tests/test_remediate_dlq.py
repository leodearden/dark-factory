"""Tests for the remediate_dlq maintenance script."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.maintenance.remediate_dlq import RemediationPlan, RemediationResult


def _make_dead_item(item_id: int, error: str = 'ResponseError: Query timed out') -> dict:
    return {
        'id': item_id,
        'group_id': 'dark_factory',
        'operation': 'add_memory_graphiti',
        'payload': {'content': f'item {item_id}'},
        'attempts': 5,
        'error': error,
        'created_at': time.time() - 3600,
    }


class TestRemediateDlqPlan:
    @pytest.mark.asyncio
    async def test_replays_then_purges_node_not_found(self):
        """execute() replays dead items, drains, then purges NodeNotFoundError items."""
        mock_queue = AsyncMock()

        # get_stats returns dead=5 initially, then dead=2 after replay drains
        stats_responses = [
            {'counts': {'dead': 5}, 'oldest_pending_age_seconds': None},  # baseline
            {'counts': {'dead': 5}, 'oldest_pending_age_seconds': None},  # during drain (pre-drain)
            {'counts': {'dead': 2}, 'oldest_pending_age_seconds': None},  # during drain (stabilised)
        ]
        mock_queue.get_stats = AsyncMock(side_effect=stats_responses)
        mock_queue.replay_dead = AsyncMock(return_value=5)
        mock_queue.get_dead_items = AsyncMock(return_value=[
            _make_dead_item(101, error='NodeNotFoundError: node abc not found'),
            _make_dead_item(102, error='NodeNotFoundError: node xyz not found'),
        ])
        mock_queue.purge_dead = AsyncMock(return_value=2)

        plan = RemediationPlan(queue=mock_queue)
        result = await plan.execute(drain_timeout=2.0, drain_poll_interval=0.05)

        assert isinstance(result, RemediationResult)
        assert result.baseline_dead == 5
        assert result.replayed == 5
        assert result.post_replay_dead == 2
        assert result.purged_node_not_found == 2
        assert result.final_dead == 0

        # Verify call order semantics
        mock_queue.replay_dead.assert_awaited_once()
        mock_queue.get_dead_items.assert_awaited_once()
        mock_queue.purge_dead.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drain_timeout_respected(self):
        """execute() returns within drain_timeout even when queue never drains."""
        import time as _time

        mock_queue = AsyncMock()
        # Stats always report dead=5 — queue never drains
        mock_queue.get_stats = AsyncMock(
            return_value={'counts': {'dead': 5}, 'oldest_pending_age_seconds': None}
        )
        mock_queue.replay_dead = AsyncMock(return_value=5)
        mock_queue.get_dead_items = AsyncMock(return_value=[])  # no NodeNotFoundError items
        mock_queue.purge_dead = AsyncMock(return_value=0)

        plan = RemediationPlan(queue=mock_queue)
        start = _time.monotonic()
        result = await plan.execute(drain_timeout=0.1, drain_poll_interval=0.02)
        elapsed = _time.monotonic() - start

        assert elapsed <= 0.3, f'Expected to return within 0.3s, took {elapsed:.2f}s'
        assert result.post_replay_dead == 5
        assert result.purged_node_not_found == 0


class TestRemediateDlqMain:
    @pytest.mark.asyncio
    async def test_main_entrypoint_uses_maintenance_service(self):
        """main() uses maintenance_service and returns a RemediationResult."""
        from fused_memory.maintenance.remediate_dlq import main

        mock_queue = AsyncMock()
        mock_queue.get_stats = AsyncMock(
            return_value={'counts': {'dead': 0}, 'oldest_pending_age_seconds': None}
        )
        mock_queue.replay_dead = AsyncMock(return_value=0)
        mock_queue.get_dead_items = AsyncMock(return_value=[])
        mock_queue.purge_dead = AsyncMock(return_value=0)

        mock_service = MagicMock()
        mock_service.durable_queue = mock_queue

        mock_config = MagicMock()

        import contextlib

        @contextlib.asynccontextmanager
        async def fake_maintenance_service(config_path):
            yield mock_config, mock_service

        with patch(
            'fused_memory.maintenance.remediate_dlq.maintenance_service',
            fake_maintenance_service,
        ):
            result = await main(config_path=None)

        assert isinstance(result, RemediationResult)
        assert result.baseline_dead == 0
        assert result.final_dead == 0
