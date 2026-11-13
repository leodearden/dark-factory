"""Tests for _held_loop_clock: a loop clock that stands still until released."""

from __future__ import annotations

import asyncio
import time

import pytest
from _held_loop_clock import HeldClock, held_loop_clock


def _fake_base(start: float) -> tuple[list[float], HeldClock]:
    now = [start]
    return now, HeldClock(lambda: now[0])


def test_a_held_clock_reads_the_held_value_while_its_base_advances():
    now, clock = _fake_base(100.0)
    now[0] = 250.0
    assert clock() == 100.0


def test_release_resumes_from_the_held_value_at_the_base_rate():
    now, clock = _fake_base(100.0)
    now[0] = 250.0
    clock.release()
    assert clock() == 100.0, 'release must resume from the held value, not jump to the base'
    now[0] += 7.0
    assert clock() == 107.0, 'after release the clock must advance at the base rate'


def test_a_second_release_does_not_reset_the_lag():
    now, clock = _fake_base(100.0)
    now[0] = 250.0
    clock.release()
    now[0] += 7.0
    before = clock()
    clock.release()
    assert clock() == before == 107.0


@pytest.mark.asyncio
async def test_a_deadline_set_while_held_survives_a_real_stall_past_its_budget():
    with held_loop_clock() as release:
        never = asyncio.get_running_loop().create_future()
        waiter = asyncio.ensure_future(asyncio.wait_for(never, timeout=0.05))
        await asyncio.to_thread(time.sleep, 0.2)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not waiter.done(), (
            'a 0.05s deadline set while the clock is held must not expire during a '
            '0.2s off-loop stall: the budget counts from release, not from the stall'
        )
        release()
        with pytest.raises(TimeoutError):
            await waiter


@pytest.mark.asyncio
async def test_leaving_the_block_restores_the_loops_own_clock():
    loop = asyncio.get_running_loop()
    with held_loop_clock() as release:
        await asyncio.to_thread(time.sleep, 0.05)
        release()
    before = time.monotonic()
    now = loop.time()
    after = time.monotonic()
    assert before <= now <= after, (
        f'loop.time() must be time.monotonic() again after the block, but read {now!r} '
        f'outside [{before!r}, {after!r}]: the lagged held clock was left installed'
    )
