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


class TestRetrieveEpisodesNaiveDatetimeRobustness:
    """retrieve_episodes must not crash when a naive datetime is mixed with aware ones."""

    @pytest.mark.asyncio
    async def test_naive_datetime_does_not_crash_and_sorts_as_utc(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        episodes = [
            _episode('uuid-aware-early', datetime(2026, 1, 1, tzinfo=UTC)),
            # Naive (tzinfo-less) — must be treated as UTC for ordering purposes.
            _episode('uuid-naive-mid', datetime(2026, 3, 1)),
            _episode('uuid-aware-late', datetime(2026, 6, 1, tzinfo=UTC)),
            _episode('uuid-none', None),
        ]
        with patch(
            'fused_memory.backends.graphiti_client.EpisodicNode.get_by_group_ids',
            AsyncMock(return_value=episodes),
        ):
            # Must not raise TypeError from comparing a naive datetime against
            # timezone-aware ones in the sort key.
            result = await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=len(episodes))

        assert len(result) == len(episodes)
        # The naive value lands at its correct chronological position (treated
        # as UTC, so it sorts between the Jan and June aware episodes), and the
        # None-created_at episode still sorts last (preserving 2055's behavior).
        assert [ep.uuid for ep in result] == [
            'uuid-aware-late',
            'uuid-naive-mid',
            'uuid-aware-early',
            'uuid-none',
        ]
