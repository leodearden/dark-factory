"""Contract of shared/src/shared/testing_virtual_clock.py.

The loop clock advances only while the loop is idle, and by exactly the
timeout it would have slept — so blocking the host thread never moves it.
``virtual_clock_test`` is pinned by decorated tests that pytest itself runs.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from shared.testing_virtual_clock import run_on_virtual_clock, virtual_clock_test


def test_an_idle_loop_jumps_straight_to_its_next_timer():
    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.sleep(3600)
        return loop.time() - t0

    assert run_on_virtual_clock(scenario()) == pytest.approx(3600.0)


def test_a_host_stall_does_not_move_the_loop_clock():
    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        time.sleep(0.02)
        return loop.time() - t0

    assert run_on_virtual_clock(scenario()) == 0.0


def test_a_deadline_cannot_expire_during_a_host_stall():
    finished = object()

    async def scenario() -> object:
        async with asyncio.timeout(0.05):
            time.sleep(0.1)
            await asyncio.sleep(0.001)
        return finished

    assert run_on_virtual_clock(scenario()) is finished


def test_the_earliest_deadline_fires_first():
    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.sleep(0.5), timeout=0.05)
        return loop.time() - t0

    assert run_on_virtual_clock(scenario()) == pytest.approx(0.05)


def test_tasks_still_pending_when_the_scenario_returns_are_cancelled():
    async def scenario() -> asyncio.Task[bool]:
        return asyncio.create_task(asyncio.Event().wait())

    returned_task = run_on_virtual_clock(scenario())

    assert returned_task.cancelled()


def test_an_exception_raised_by_the_scenario_propagates_unchanged():
    async def scenario() -> None:
        raise ValueError('x')

    with pytest.raises(ValueError, match='x'):
        run_on_virtual_clock(scenario())


def test_executor_work_is_rejected_loudly():
    async def scenario() -> None:
        await asyncio.to_thread(time.sleep, 0)

    with pytest.raises(RuntimeError, match='executor'):
        run_on_virtual_clock(scenario())


def test_name_resolution_is_rejected_loudly():
    async def scenario() -> None:
        await asyncio.get_running_loop().getaddrinfo('localhost', 80)

    with pytest.raises(RuntimeError, match='executor'):
        run_on_virtual_clock(scenario())


def _assert_a_host_stall_leaves_the_running_loop_clock_still() -> None:
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    time.sleep(0.02)
    assert loop.time() - t0 == 0.0


@virtual_clock_test
async def test_a_decorated_test_runs_on_the_virtual_clock_with_its_fixtures(
    tmp_path: Path,
) -> None:
    _assert_a_host_stall_leaves_the_running_loop_clock_still()
    assert tmp_path.is_dir()


class TestADecoratedMethod:
    @virtual_clock_test
    async def test_runs_on_the_virtual_clock_with_self_and_its_fixtures(
        self, tmp_path: Path,
    ) -> None:
        _assert_a_host_stall_leaves_the_running_loop_clock_still()
        assert isinstance(self, TestADecoratedMethod)
        assert tmp_path.is_dir()
