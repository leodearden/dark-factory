"""Regression tests for GraphitiBackend.retrieve_episodes ordering (task 2055).

get_episodes(last_n=N) must return the N *most-recent* episodes, ordered by
created_at descending. graphiti_core's EpisodicNode.get_by_group_ids truncates
via ``ORDER BY uuid DESC LIMIT $limit`` — a UUID ordering unrelated to recency —
so retrieve_episodes must fetch the group's full episode set and sort/truncate
by created_at itself rather than delegating truncation to that call.
"""
from __future__ import annotations

import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest


def _episode(uuid: str, created_at: datetime | None) -> types.SimpleNamespace:
    """Build a minimal EpisodicNode-shaped stand-in for mocking get_by_group_ids."""
    return types.SimpleNamespace(
        uuid=uuid,
        name=f'episode-{uuid}',
        content=f'content-{uuid}',
        source='message',
        group_id='dark_factory',
        created_at=created_at,
    )


class TestRetrieveEpisodesOrdering:
    """GraphitiBackend.retrieve_episodes must sort by created_at DESC before truncating."""

    @pytest.mark.asyncio
    async def test_retrieve_episodes_returns_n_most_recent_descending(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        # Deliberately non-insertion / non-monotonic created_at order.
        episodes = [
            _episode('uuid-apr', datetime(2026, 4, 7, tzinfo=UTC)),
            _episode('uuid-jun', datetime(2026, 6, 22, tzinfo=UTC)),
            _episode('uuid-may', datetime(2026, 5, 17, tzinfo=UTC)),
            _episode('uuid-feb', datetime(2026, 2, 1, tzinfo=UTC)),
            _episode('uuid-mar', datetime(2026, 3, 10, tzinfo=UTC)),
        ]
        with patch(
            'fused_memory.backends.graphiti_client.EpisodicNode.get_by_group_ids',
            AsyncMock(return_value=episodes),
        ):
            result = await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=3)

        # (a) exactly last_n returned
        assert len(result) == 3
        # (b) selection correctness — the 3 largest created_at, not the first 3 fetched
        assert {ep.uuid for ep in result} == {'uuid-jun', 'uuid-may', 'uuid-mar'}
        # (c) strictly non-increasing created_at order
        created_ats = [ep.created_at for ep in result]
        assert created_ats == sorted(created_ats, reverse=True)
