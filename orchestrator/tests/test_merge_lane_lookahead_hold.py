"""The merger's wait, before its look-ahead pop, for the verifier's chain decision (task 7720).

Unit tests for ``merge_lane/lookahead_hold.py::LookaheadHold``.
"""
from __future__ import annotations

import asyncio

import pytest

from orchestrator.merge_lane.lookahead_hold import LookaheadHold

pytestmark = pytest.mark.asyncio


def _alive() -> asyncio.Future[None]:
    """A stand-in for the verifier task: pending until the test finishes it."""
    return asyncio.get_running_loop().create_future()


async def test_decided_tip_is_delivered_and_forgotten() -> None:
    alive = _alive()
    hold = LookaheadHold()
    hold.hold('mr-s')
    waiter = asyncio.create_task(hold.wait('mr-s', alive=alive))
    await asyncio.sleep(0)
    assert not waiter.done()
    hold.decide('mr-s', 'TIP')
    assert await asyncio.wait_for(waiter, 5) == 'TIP'
    assert not hold.is_held('mr-s')


async def test_decision_before_wait_is_kept() -> None:
    alive = _alive()
    hold = LookaheadHold()
    hold.hold('mr-s')
    hold.decide('mr-s', 'TIP')
    assert await asyncio.wait_for(hold.wait('mr-s', alive=alive), 5) == 'TIP'
    assert not hold.is_held('mr-s')


async def test_decide_on_an_unheld_id_is_a_no_op() -> None:
    hold = LookaheadHold()
    hold.decide('mr-nobody', 'TIP')
    assert not hold.is_held('mr-nobody')


async def test_wait_on_an_unheld_id_returns_none_at_once() -> None:
    alive = _alive()
    hold = LookaheadHold()
    assert await asyncio.wait_for(hold.wait('mr-nobody', alive=alive), 5) is None


async def test_release_all_unblocks_every_wait() -> None:
    alive = _alive()
    hold = LookaheadHold()
    hold.hold('mr-a')
    hold.hold('mr-b')
    waiters = [
        asyncio.create_task(hold.wait(rid, alive=alive)) for rid in ('mr-a', 'mr-b')
    ]
    await asyncio.sleep(0)
    hold.release_all()
    assert await asyncio.wait_for(asyncio.gather(*waiters), 5) == [None, None]
    assert not hold.is_held('mr-a')
    assert not hold.is_held('mr-b')


async def test_a_finished_verifier_never_wedges_the_wait() -> None:
    alive = _alive()
    hold = LookaheadHold()
    hold.hold('mr-s')
    waiter = asyncio.create_task(hold.wait('mr-s', alive=alive))
    await asyncio.sleep(0)
    assert not waiter.done()
    alive.set_result(None)
    assert await asyncio.wait_for(waiter, 5) is None
    assert not hold.is_held('mr-s')


async def test_holding_twice_is_refused_naming_the_id() -> None:
    hold = LookaheadHold()
    hold.hold('mr-s')
    with pytest.raises(ValueError, match='mr-s'):
        hold.hold('mr-s')
