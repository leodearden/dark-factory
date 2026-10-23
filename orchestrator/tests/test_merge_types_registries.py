"""InFlightMergeRegistry and MergeBounceRegistry behaviour, pinned through their public methods."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from orchestrator.merge_lane.types import InFlightMergeRegistry


@pytest.fixture
def event_loop_for_futures() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestAcquireSeedsTheEntry:
    def test_entry_records_the_branch_it_was_acquired_under(
        self, event_loop_for_futures: asyncio.AbstractEventLoop,
    ) -> None:
        registry = InFlightMergeRegistry()
        fut = event_loop_for_futures.create_future()

        registry.acquire('591', 'T-591', fut, request_id='mr-1', snapshot_tip='sha-a')

        entry = registry.entry('591')
        assert entry is not None
        assert entry.branch == '591'
        assert entry.task_id == 'T-591'
        assert entry.request_id == 'mr-1'
        assert entry.snapshot_tip == 'sha-a'
        assert entry.primary_future is fut

    def test_dispatcher_without_a_request_id_waits_under_the_empty_id(
        self, event_loop_for_futures: asyncio.AbstractEventLoop,
    ) -> None:
        registry = InFlightMergeRegistry()
        fut = event_loop_for_futures.create_future()

        registry.acquire('591', 'T-591', fut)

        entry = registry.entry('591')
        assert entry is not None
        assert entry.request_id is None
        assert [w.request_id for w in entry.waiters] == ['']
