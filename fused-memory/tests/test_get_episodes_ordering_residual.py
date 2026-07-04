"""Regression tests for residual intermittent non-monotonic get_episodes ordering
(task 2079, follow-up to task 2055).

Task 2055 fixed GraphitiBackend.retrieve_episodes to fetch the group's full
episode set and sort/truncate by created_at in Python (rather than delegating
truncation to EpisodicNode.get_by_group_ids' ``ORDER BY uuid DESC LIMIT``,
which truncates on the wrong key). That fix is correct and is NOT reopened
here — see test_get_episodes_ordering.py for its regression tests.

This file hardens the same sort against two additive gaps that can produce an
*apparent* non-monotonic result while 2055's instant-order sort remains
correct:

1. No deterministic tie-breaker: when >=2 episodes share created_at, Python's
   stable sort merely preserves upstream (get_by_group_ids) order, which is
   not guaranteed reproducible across executions.
2. Naive-vs-aware datetime comparison: a tzinfo-less created_at raises
   TypeError when compared against timezone-aware ones, turning the read into
   an error instead of a correctly-ordered (or at least non-crashing) result.
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


class TestRetrieveEpisodesTieBreaker:
    """retrieve_episodes must apply a deterministic secondary key when created_at ties."""

    @pytest.mark.asyncio
    async def test_tied_created_at_orders_deterministically_regardless_of_upstream_order(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        tied_at = datetime(2026, 5, 1, tzinfo=UTC)
        recent = _episode('uuid-recent', datetime(2026, 6, 1, tzinfo=UTC))
        tie_a = _episode('uuid-tie-a', tied_at)
        tie_b = _episode('uuid-tie-b', tied_at)
        created_at_by_uuid = {ep.uuid: ep.created_at for ep in (recent, tie_a, tie_b)}

        async def _uuids_for(upstream_order: list[types.SimpleNamespace]) -> list[str]:
            with patch(
                'fused_memory.backends.graphiti_client.EpisodicNode.get_by_group_ids',
                AsyncMock(return_value=upstream_order),
            ):
                result = await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=3)
            return [ep.uuid for ep in result]

        order_from_b_first = await _uuids_for([tie_b, tie_a, recent])
        order_from_a_first = await _uuids_for([tie_a, tie_b, recent])

        # (a) order is invariant to the upstream (get_by_group_ids) input order.
        assert order_from_b_first == order_from_a_first

        # (b) within the tied group, order follows the deterministic secondary
        # key (uuid, descending) rather than upstream input order.
        assert order_from_b_first == ['uuid-recent', 'uuid-tie-b', 'uuid-tie-a']

        # (c) created_at values remain non-increasing overall.
        created_ats = [created_at_by_uuid[u] for u in order_from_b_first]
        assert created_ats == sorted(created_ats, reverse=True)
