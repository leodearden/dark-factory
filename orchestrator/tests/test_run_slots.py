"""Task 8480: run slots split into a general pool and a reserve only a main-health fix may take."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.main_health_fix_owner import SIGNATURE_KEY
from orchestrator.run_slots import DispatchScope, RunSlotInvariantError, RunSlots

PLAIN_TASK = {'id': 7, 'metadata': {'files': ['a.py']}}
FIX_TASK = {'id': 9, 'metadata': {SIGNATURE_KEY: 'sig-1', 'merge_lane': 'high'}}


async def _claim(slots: RunSlots, scope: DispatchScope, task: dict) -> asyncio.Semaphore:
    return await asyncio.wait_for(slots.claim(scope, task), timeout=0.1)


class TestDispatchScopeAdmits:
    @pytest.mark.parametrize('task', [PLAIN_TASK, FIX_TASK, {'id': 1}])
    def test_any_admits_every_task(self, task: dict) -> None:
        assert DispatchScope.ANY.admits(task)

    def test_main_health_fix_admits_only_a_signed_fix(self) -> None:
        assert DispatchScope.MAIN_HEALTH_FIX.admits(FIX_TASK)
        assert not DispatchScope.MAIN_HEALTH_FIX.admits(PLAIN_TASK)


class TestFreeScope:
    @pytest.mark.asyncio
    async def test_scope_narrows_as_the_general_pool_then_the_reserve_fill(self) -> None:
        slots = RunSlots(general=2, main_health_fix_reserve=1)
        assert slots.free_scope() is DispatchScope.ANY

        await _claim(slots, DispatchScope.ANY, PLAIN_TASK)
        assert slots.free_scope() is DispatchScope.ANY
        await _claim(slots, DispatchScope.ANY, PLAIN_TASK)
        assert slots.free_scope() is DispatchScope.MAIN_HEALTH_FIX

        await _claim(slots, DispatchScope.MAIN_HEALTH_FIX, FIX_TASK)
        assert slots.free_scope() is None

    @pytest.mark.asyncio
    async def test_releasing_a_claim_frees_the_pool_it_came_from(self) -> None:
        slots = RunSlots(general=2, main_health_fix_reserve=1)
        general_first = await _claim(slots, DispatchScope.ANY, PLAIN_TASK)
        general_second = await _claim(slots, DispatchScope.ANY, PLAIN_TASK)
        reserve = await _claim(slots, DispatchScope.MAIN_HEALTH_FIX, FIX_TASK)

        assert reserve is not general_first
        assert reserve is not general_second

        reserve.release()
        assert slots.free_scope() is DispatchScope.MAIN_HEALTH_FIX

        await _claim(slots, DispatchScope.MAIN_HEALTH_FIX, FIX_TASK)
        general_first.release()
        assert slots.free_scope() is DispatchScope.ANY

    @pytest.mark.asyncio
    async def test_a_zero_reserve_leaves_nothing_free_once_the_general_pool_is_full(self) -> None:
        slots = RunSlots(general=1, main_health_fix_reserve=0)

        await _claim(slots, DispatchScope.ANY, PLAIN_TASK)

        assert slots.free_scope() is None

    @pytest.mark.asyncio
    async def test_eighteen_merge_waiters_leave_the_reserve_free(self) -> None:
        slots = RunSlots(general=18, main_health_fix_reserve=1)

        for _ in range(18):
            await _claim(slots, DispatchScope.ANY, PLAIN_TASK)

        assert slots.free_scope() is DispatchScope.MAIN_HEALTH_FIX


class TestClaimInvariants:
    @pytest.mark.asyncio
    async def test_the_reserve_refuses_a_task_its_scope_does_not_admit(self) -> None:
        slots = RunSlots(general=1, main_health_fix_reserve=1)

        with pytest.raises(RunSlotInvariantError) as raised:
            await _claim(slots, DispatchScope.MAIN_HEALTH_FIX, PLAIN_TASK)

        message = str(raised.value)
        assert str(PLAIN_TASK['id']) in message
        assert DispatchScope.MAIN_HEALTH_FIX.value in message

    @pytest.mark.asyncio
    async def test_a_claim_on_a_full_general_pool_is_refused(self) -> None:
        slots = RunSlots(general=1, main_health_fix_reserve=1)
        await _claim(slots, DispatchScope.ANY, PLAIN_TASK)

        with pytest.raises(RunSlotInvariantError) as raised:
            await _claim(slots, DispatchScope.ANY, PLAIN_TASK)

        message = str(raised.value)
        assert DispatchScope.ANY.value in message
        assert 'general' in message
        assert 'full' in message

    @pytest.mark.parametrize('kwargs, argument', [
        ({'general': -1, 'main_health_fix_reserve': 1}, 'general'),
        ({'general': 1, 'main_health_fix_reserve': -1}, 'main_health_fix_reserve'),
    ])
    def test_negative_counts_are_rejected(self, kwargs: dict, argument: str) -> None:
        with pytest.raises(ValueError, match=argument):
            RunSlots(**kwargs)
