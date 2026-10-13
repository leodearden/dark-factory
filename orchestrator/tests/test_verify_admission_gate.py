"""The in-process task-verify admission gate (plans/verify-admission-fifo-prd.md, task α).

Driven only through the gate's public API — ``gate_for(tmp_path)``, its value
types, ``admit`` and ``snapshot`` — with the REAL flock from
``shared.verify_admission`` taken inside every granted body. The gate is not
behind the autouse ``_neutralize_verify_admission`` seam, so no marker is needed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from shared.verify_admission import acquire_task_slot, is_gated_role

from orchestrator.verify_admission_gate import (
    ROLE_CLASSES,
    AdmissionGrant,
    AdmissionOrder,
    AdmissionPlan,
    AdmissionPolicy,
    AdmissionRequest,
    gate_for,
)


async def _until(
    predicate: Callable[[], bool], *, watching: asyncio.Task | None = None, timeout: float = 5.0,
) -> None:
    """Poll *predicate* under a deadline, surfacing *watching*'s own failure if it ends first."""
    async with asyncio.timeout(timeout):
        while not predicate():
            if watching is not None and watching.done():
                watching.result()
                raise AssertionError('the leg finished before reaching the awaited state')
            await asyncio.sleep(0.005)


async def _expect_cancelled(task: asyncio.Task) -> None:
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(5):
            await task


def _request(label: str, role: str = 'task') -> AdmissionRequest:
    return AdmissionPlan.begin(role=role, task_id=label).request(module=label, scope_kind=None)


class _Legs:
    """Admitted legs on one gate. Each granted body takes the REAL flock without
    waiting, asserts it got a slot, records its turn, then parks until released."""

    def __init__(self, slots_dir: Path, policy: AdmissionPolicy) -> None:
        self.gate = gate_for(slots_dir)
        self.slots_dir = slots_dir
        self.policy = policy
        self.granted: list[str] = []
        self.grants: dict[str, AdmissionGrant] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self._release: dict[str, asyncio.Event] = {}

    def read_policy(self) -> AdmissionPolicy:
        return self.policy

    def start(self, label: str, request: AdmissionRequest, *, hold: bool = False) -> asyncio.Task:
        release = asyncio.Event()
        if not hold:
            release.set()
        self._release[label] = release
        task = asyncio.create_task(self._leg(label, request, release))
        self.tasks[label] = task
        return task

    async def _leg(self, label: str, request: AdmissionRequest, release: asyncio.Event) -> None:
        async with self.gate.admit(request, self.read_policy) as grant:
            with acquire_task_slot(
                request.role, slots_dir=self.slots_dir, n=self.policy.slots, wait=False,
            ) as held:
                assert held is True, f'{label}: granted a turn the flock had no slot for'
                self.granted.append(label)
                self.grants[label] = grant
                await release.wait()

    async def hold(self, label: str, request: AdmissionRequest | None = None) -> asyncio.Task:
        task = self.start(label, request or _request(label), hold=True)
        await _until(lambda: label in self.granted, watching=task)
        return task

    async def queue(
        self, label: str, request: AdmissionRequest | None = None, *, hold: bool = False,
    ) -> asyncio.Task:
        depth = self.gate.snapshot().depth
        task = self.start(label, request or _request(label), hold=hold)
        await _until(lambda: self.gate.snapshot().depth == depth + 1, watching=task)
        return task

    def release(self, label: str) -> None:
        self._release[label].set()

    def forget(self, label: str) -> asyncio.Task:
        return self.tasks.pop(label)

    async def finish(self) -> None:
        for release in self._release.values():
            release.set()
        async with asyncio.timeout(5):
            await asyncio.gather(*self.tasks.values())


def _policy(
    *, slots: int = 1, order: AdmissionOrder = 'arrival', promote_after_secs: float = 7200.0,
) -> AdmissionPolicy:
    return AdmissionPolicy(slots=slots, order=order, promote_after_secs=promote_after_secs)


class TestRoleClasses:
    @pytest.mark.parametrize('role', ['merge', 'task', 'background', 'offline', 'no-such-role'])
    def test_role_classes_agree_with_the_gated_role_authority(self, role):
        assert (role in ROLE_CLASSES) == is_gated_role(role)

    def test_task_ranks_ahead_of_background(self):
        assert ROLE_CLASSES['task'] < ROLE_CLASSES['background']


class TestAdmissionPlan:
    def test_begin_stamps_a_unique_plan(self):
        first = AdmissionPlan.begin(role='task', task_id='42')
        second = AdmissionPlan.begin(role='task', task_id='42')

        assert first.plan_id != second.plan_id
        assert first.arrival <= second.arrival
        assert first.arrived_at.utcoffset() == timedelta(0)
        assert first.role == 'task'
        assert first.task_id == '42'

    def test_requests_of_one_plan_share_its_identity_and_arrival(self):
        plan = AdmissionPlan.begin(role='background', task_id=None)

        a = plan.request(module='a', scope_kind=None)
        b = plan.request(module='b', scope_kind='file_scoped')

        assert a.plan_id == b.plan_id == plan.plan_id
        assert a.plan_arrival == b.plan_arrival == plan.arrival
        assert (a.role, a.module, a.task_id) == ('background', 'a', None)
        assert (b.module, b.scope_kind) == ('b', 'file_scoped')

    def test_an_ungated_role_cannot_begin_a_plan(self):
        with pytest.raises(ValueError, match=r'only gated verify roles.*merge'):
            AdmissionPlan.begin(role='merge', task_id='1')

    def test_an_ungated_role_cannot_build_a_request(self):
        with pytest.raises(ValueError, match=r'only gated verify roles.*merge'):
            AdmissionRequest(
                role='merge', plan_id='p', plan_arrival=0.0,
                task_id=None, module=None, scope_kind=None,
            )


class TestAdmissionPolicy:
    def test_a_valid_policy_constructs(self):
        policy = AdmissionPolicy(slots=1, order='arrival', promote_after_secs=0.0)
        assert (policy.slots, policy.order, policy.promote_after_secs) == (1, 'arrival', 0.0)

    def test_slots_below_one_is_rejected(self):
        with pytest.raises(ValueError, match=r'slots.*0'):
            AdmissionPolicy(slots=0, order='unordered', promote_after_secs=1.0)

    def test_an_unknown_order_is_rejected(self):
        with pytest.raises(ValueError, match=r'order.*fifo'):
            AdmissionPolicy(slots=1, order=cast(Any, 'fifo'), promote_after_secs=1.0)

    def test_a_negative_promote_age_is_rejected(self):
        with pytest.raises(ValueError, match=r'promote_after_secs.*-1'):
            AdmissionPolicy(slots=1, order='arrival', promote_after_secs=-1.0)


class TestRegistry:
    def test_one_gate_per_slots_dir(self, tmp_path):
        assert gate_for(tmp_path) is gate_for(tmp_path)

    def test_the_key_is_the_resolved_path(self, tmp_path):
        assert gate_for(tmp_path) is gate_for(tmp_path / 'x' / '..')

    def test_distinct_dirs_get_distinct_gates(self, tmp_path):
        assert gate_for(tmp_path / 'a') is not gate_for(tmp_path / 'b')


class TestFreshSnapshot:
    def test_a_fresh_gate_is_empty(self, tmp_path):
        snap = gate_for(tmp_path).snapshot()

        assert snap.depth == 0
        assert snap.waiters == ()
        assert snap.holders == ()
        assert snap.oldest is None
        assert snap.oldest_wait_secs is None

    def test_the_snapshot_is_frozen(self, tmp_path):
        snap = gate_for(tmp_path).snapshot()

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, snap).waiters = ()


@pytest.mark.asyncio
class TestArrivalOrder:
    async def test_same_class_plans_are_granted_in_arrival_order(self, tmp_path):
        legs = _Legs(tmp_path, _policy())
        await legs.hold('H')
        for label in ('A', 'B', 'C'):
            await legs.queue(label)

        legs.release('H')
        await legs.finish()

        assert legs.granted == ['H', 'A', 'B', 'C']

    async def test_the_plan_stamp_not_the_gate_arrival_orders_same_class_legs(self, tmp_path):
        legs = _Legs(tmp_path, _policy())
        requests = {label: _request(label) for label in ('A', 'B', 'C')}
        await legs.hold('H')
        for label in ('C', 'A', 'B'):
            await legs.queue(label, requests[label])

        legs.release('H')
        await legs.finish()

        assert legs.granted == ['H', 'A', 'B', 'C']

    async def test_unordered_grants_every_permutation(self, tmp_path):
        seen: set[tuple[str, ...]] = set()
        all_orders = set(itertools.permutations('ABC'))
        for trial in range(400):
            slots_dir = tmp_path / str(trial)
            slots_dir.mkdir()
            legs = _Legs(slots_dir, _policy(order='unordered'))
            await legs.hold('H')
            for label in ('A', 'B', 'C'):
                await legs.queue(label)
            legs.release('H')
            await legs.finish()
            seen.add(tuple(legs.granted[1:]))
            if seen == all_orders and trial >= 59:
                break

        assert seen == all_orders

    async def test_a_task_plan_is_granted_before_an_earlier_background_plan(self, tmp_path):
        legs = _Legs(tmp_path, _policy())
        background = _request('G', role='background')
        task = _request('T')
        await legs.hold('H')
        await legs.queue('G', background)
        await legs.queue('T', task)

        legs.release('H')
        await legs.finish()

        assert legs.granted == ['H', 'T', 'G']

    @pytest.mark.parametrize(
        ('promote_after_secs', 'expected'),
        [(0.2, ['H', 'T1', 'G', 'T2']), (3600.0, ['H', 'T1', 'T2', 'G'])],
    )
    async def test_an_aged_background_waiter_ranks_with_tasks_on_its_own_stamp(
        self, tmp_path, promote_after_secs, expected,
    ):
        legs = _Legs(tmp_path, _policy(promote_after_secs=promote_after_secs))
        await legs.hold('H')
        await legs.queue('T1')
        await legs.queue('G', _request('G', role='background'))
        await asyncio.sleep(0.3)
        await legs.queue('T2')

        legs.release('H')
        await legs.finish()

        assert legs.granted == expected


@pytest.mark.asyncio
class TestGrantFacts:
    async def test_a_free_slot_grants_at_once(self, tmp_path):
        legs = _Legs(tmp_path, _policy())

        await legs.hold('H')

        grant = legs.grants['H']
        assert legs.gate.snapshot().depth == 0
        assert grant.depth_at_arrival == 0
        assert 0 <= grant.waited_secs < 1.0
        await legs.finish()

    @pytest.mark.parametrize('order', ['arrival', 'unordered'])
    async def test_a_waiter_grant_records_its_queue_and_wait(self, tmp_path, order):
        legs = _Legs(tmp_path, _policy(order=order))
        await legs.hold('H')
        for label in ('A', 'B', 'C'):
            await legs.queue(label)

        legs.release('H')
        await legs.finish()

        for depth, label in enumerate(('A', 'B', 'C')):
            grant = legs.grants[label]
            assert grant.depth_at_arrival == depth
            assert grant.waited_secs > 0
            assert grant.granted_at >= grant.arrived_at
            assert grant.order == order
            assert grant.slot_held is False
            assert grant.request.module == label


@pytest.mark.asyncio
class TestPolicyIsReadAtEachGrant:
    async def test_raising_slots_admits_a_second_holder_at_the_next_grant(self, tmp_path):
        legs = _Legs(tmp_path, _policy(slots=1))
        await legs.hold('H')
        await legs.queue('A', hold=True)
        await legs.queue('B', hold=True)

        legs.policy = _policy(slots=2)
        legs.release('H')
        await _until(lambda: len(legs.gate.snapshot().holders) == 2)

        assert {h.request.module for h in legs.gate.snapshot().holders} == {'A', 'B'}
        await legs.finish()

    async def test_switching_to_arrival_order_applies_to_the_next_grant(self, tmp_path):
        legs = _Legs(tmp_path, _policy(order='unordered'))
        await legs.hold('H')
        for label in ('A', 'B', 'C'):
            await legs.queue(label, hold=True)

        legs.release('H')
        await _until(lambda: len(legs.granted) == 2)
        first = legs.granted[1]
        legs.policy = _policy(order='arrival')
        legs.release(first)
        await legs.finish()

        assert legs.granted[2:] == sorted({'A', 'B', 'C'} - {first})
        assert legs.grants[first].order == 'unordered'
        assert {legs.grants[label].order for label in legs.granted[2:]} == {'arrival'}


def _loop_callback_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if 'Exception in callback' in r.getMessage()
        or 'InvalidStateError' in r.getMessage()
        or (r.exc_info is not None and isinstance(r.exc_info[1], asyncio.InvalidStateError))
    ]


@pytest.mark.asyncio
class TestCancellation:
    async def test_a_waiter_cancelled_in_the_queue_leaves_it(self, tmp_path):
        legs = _Legs(tmp_path, _policy())
        await legs.hold('H')
        await legs.queue('A')
        await legs.queue('B')

        await _expect_cancelled(_cancel(legs.forget('A')))

        assert [w.request.module for w in legs.gate.snapshot().waiters] == ['B']
        legs.release('H')
        await legs.finish()
        assert legs.granted == ['H', 'B']
        assert legs.gate.snapshot().depth == 0
        assert legs.gate.snapshot().holders == ()

    @pytest.mark.parametrize('release_first', [True, False])
    async def test_a_cancel_racing_its_grant_leaks_no_holder(self, tmp_path, caplog, release_first):
        caplog.set_level(logging.WARNING)
        legs = _Legs(tmp_path, _policy())
        holder = legs.gate.admit(_request('H'), legs.read_policy)
        await holder.__aenter__()
        await legs.queue('A')
        waiter = legs.forget('A')

        if release_first:
            await holder.__aexit__(None, None, None)
            waiter.cancel()
        else:
            waiter.cancel()
            await holder.__aexit__(None, None, None)
        await _expect_cancelled(waiter)

        assert legs.gate.snapshot().holders == ()
        assert legs.gate.snapshot().depth == 0
        async with asyncio.timeout(1):
            async with legs.gate.admit(_request('N'), legs.read_policy) as grant:
                assert grant.depth_at_arrival == 0
        assert _loop_callback_errors(caplog) == []

    async def test_a_body_that_raises_hands_its_turn_on(self, tmp_path):
        legs = _Legs(tmp_path, _policy())
        fail = asyncio.Event()

        async def failing_holder() -> None:
            async with legs.gate.admit(_request('H'), legs.read_policy):
                await fail.wait()
                raise RuntimeError('the admitted body failed')

        holder = asyncio.create_task(failing_holder())
        await _until(lambda: len(legs.gate.snapshot().holders) == 1, watching=holder)
        await legs.queue('A')

        fail.set()
        with pytest.raises(RuntimeError, match='the admitted body failed'):
            async with asyncio.timeout(5):
                await holder
        await legs.finish()

        assert legs.granted == ['A']
        assert legs.gate.snapshot().holders == ()

    async def test_dispatch_skips_a_cancelled_waiter_it_reaches_first(self, tmp_path):
        legs = _Legs(tmp_path, _policy())
        holder = legs.gate.admit(_request('H'), legs.read_policy)
        await holder.__aenter__()
        await legs.queue('A')
        await legs.queue('B')

        cancelled = legs.forget('A')
        cancelled.cancel()
        await holder.__aexit__(None, None, None)
        await _expect_cancelled(cancelled)
        await legs.finish()

        assert legs.granted == ['B']
        assert legs.gate.snapshot().depth == 0
        assert legs.gate.snapshot().holders == ()


def _cancel(task: asyncio.Task) -> asyncio.Task:
    task.cancel()
    return task
