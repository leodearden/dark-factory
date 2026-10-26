"""InFlightMergeRegistry and MergeBounceRegistry behaviour, pinned through their public methods."""

from __future__ import annotations

import asyncio
import gc
import time
import weakref
from collections.abc import Iterator

import pytest

from orchestrator.merge_lane.types import InFlightMergeRegistry, MergeBounceRegistry


@pytest.fixture
def event_loop_for_futures() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class ManualClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


@pytest.fixture
def frozen_monotonic(monkeypatch: pytest.MonkeyPatch) -> ManualClock:
    clock = ManualClock(1000.0)
    monkeypatch.setattr(time, 'monotonic', clock)
    return clock


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


def _dispatch_591(loop: asyncio.AbstractEventLoop) -> tuple[InFlightMergeRegistry, int]:
    registry = InFlightMergeRegistry()
    registry.acquire('591', 'T-591', loop.create_future(), request_id='mr-1')
    at_dispatch = registry.eta_seconds('591')
    assert at_dispatch is not None and at_dispatch > 1, f'estimate too small to probe: {at_dispatch}'
    return registry, at_dispatch


class TestEtaSeconds:
    def test_eta_counts_down_one_second_per_elapsed_second(
        self, event_loop_for_futures: asyncio.AbstractEventLoop, frozen_monotonic: ManualClock,
    ) -> None:
        registry, at_dispatch = _dispatch_591(event_loop_for_futures)

        frozen_monotonic.advance(1)

        assert registry.eta_seconds('591') == at_dispatch - 1

    def test_eta_still_reports_the_last_whole_second(
        self, event_loop_for_futures: asyncio.AbstractEventLoop, frozen_monotonic: ManualClock,
    ) -> None:
        registry, at_dispatch = _dispatch_591(event_loop_for_futures)

        frozen_monotonic.advance(at_dispatch - 1)

        assert registry.eta_seconds('591') == 1

    def test_eta_is_none_once_the_estimate_is_used_up(
        self, event_loop_for_futures: asyncio.AbstractEventLoop, frozen_monotonic: ManualClock,
    ) -> None:
        registry, at_dispatch = _dispatch_591(event_loop_for_futures)

        frozen_monotonic.advance(at_dispatch)
        assert registry.eta_seconds('591') is None

        frozen_monotonic.advance(1)
        assert registry.eta_seconds('591') is None


class TestDetachLastWaiter:
    def test_detaching_after_the_primary_failed_leaves_its_error_reportable(
        self, event_loop_for_futures: asyncio.AbstractEventLoop,
    ) -> None:
        """Detaching every waiter from an already-failed merge must not mark its failure as handled."""
        loop = event_loop_for_futures
        messages: list[str] = []
        loop.set_exception_handler(lambda _loop, context: messages.append(context['message']))
        registry = InFlightMergeRegistry()
        primary = loop.create_future()
        registry.acquire('591', 'T-591', primary, request_id='mr-1')
        primary.set_exception(RuntimeError('verify crashed'))

        assert registry.detach('591', 'mr-1') == 0
        loop.run_until_complete(asyncio.sleep(0))
        assert registry.entry('591') is None, 'the registry still holds the failed primary'
        assert primary.done() and not primary.cancelled(), 'the failed primary must stay failed'

        loop.close()
        primary_ref = weakref.ref(primary)
        del primary
        gc.collect()
        assert primary_ref() is None, 'something still holds the failed primary, so asyncio cannot report it'
        assert messages == ['Future exception was never retrieved']


class TestMergeBounceRegistryClear:
    def test_clearing_a_branch_that_never_bounced_is_a_no_op(self) -> None:
        registry = MergeBounceRegistry()
        registry.record_bounce('task/592')

        registry.clear('task/591')

        assert registry.count('task/591') == 0
        assert registry.count('task/592') == 1

    def test_clear_resets_a_bounced_branch_to_zero(self) -> None:
        registry = MergeBounceRegistry()
        registry.record_bounce('task/591')
        registry.record_bounce('task/591')

        registry.clear('task/591')

        assert registry.count('task/591') == 0
        assert registry.record_bounce('task/591') == 1
