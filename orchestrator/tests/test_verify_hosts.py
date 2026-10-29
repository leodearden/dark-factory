"""HostPool: the task path's view of the worker-lifetime HostAllocator.

Every case drives a REAL :class:`~orchestrator.verify_runner.HostAllocator`
over tiny fake remotes that expose only the public surface the allocator
calls (``name``, ``cancel_verify()``, ``probe_clean()``).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import pytest

from orchestrator.verify_hosts import HostPool, Refusal, RefusalReason
from orchestrator.verify_runner import HostAllocator, HostLease, RunnerUnavailable


class _FakeRemote:
    """A remote host whose cancel rc and probe verdict are fixed per test."""

    def __init__(
        self, name: str, *, cancel_rc: int = 0, probe_clean: bool = True,
        cancel_raises: Exception | None = None,
    ) -> None:
        self.name = name
        self.cancel_rc = cancel_rc
        self.cancel_raises = cancel_raises
        self.probe_verdict = probe_clean
        self.cancel_calls = 0
        self.probe_calls = 0

    async def cancel_verify(self) -> int:
        self.cancel_calls += 1
        if self.cancel_raises is not None:
            raise self.cancel_raises
        return self.cancel_rc

    async def probe_clean(self) -> bool:
        self.probe_calls += 1
        return self.probe_verdict


class _Hooks:
    """Recording merge-demand / unavailable / recovered callables."""

    def __init__(self, *, demand: list[bool] | None = None) -> None:
        self.demand_answers = list(demand or [False])
        self.demand_calls = 0
        self.unavailable: list[tuple[str, RunnerUnavailable]] = []
        self.recovered: list[str] = []

    def merge_demand(self) -> bool:
        answer = self.demand_answers[min(self.demand_calls, len(self.demand_answers) - 1)]
        self.demand_calls += 1
        return answer

    def on_unavailable(self, name: str, exc: RunnerUnavailable) -> None:
        self.unavailable.append((name, exc))

    def on_recovered(self, name: str) -> None:
        self.recovered.append(name)


class _RecordingUnwind:
    """An async teardown that records what *observe* returns each time it is awaited."""

    def __init__(self, observe: Callable[[], object]) -> None:
        self._observe = observe
        self.snapshots: list[object] = []

    async def __call__(self) -> None:
        self.snapshots.append(self._observe())


async def _no_sleep(_seconds: float) -> None:
    return None


def _pool(
    remotes: list[_FakeRemote],
    hooks: _Hooks,
    *,
    quarantine: set[str] | None = None,
) -> tuple[HostPool, HostAllocator]:
    allocator = HostAllocator(list(remotes), quarantine=quarantine)
    pool = HostPool(
        allocator,
        merge_demand=hooks.merge_demand,
        on_unavailable=hooks.on_unavailable,
        on_recovered=hooks.on_recovered,
        probe_sleep=_no_sleep,
    )
    return pool, allocator


def _lease(outcome: HostLease | Refusal) -> HostLease:
    assert isinstance(outcome, HostLease), outcome
    return outcome


def _held_by_a_merge_lease(allocator: HostAllocator, quarantine: set[str]) -> None:
    assert allocator.acquire_remote() is not None


def _benched(allocator: HostAllocator, quarantine: set[str]) -> None:
    quarantine.add('laptop')


_HOST_STATES = [
    pytest.param(_held_by_a_merge_lease, 'host_busy', id='busy'),
    pytest.param(_benched, 'host_quarantined', id='quarantined'),
]


async def _park_through_a_failed_cancel(pool: HostPool, allocator: HostAllocator, name: str) -> None:
    lease = _lease(pool.acquire_for_task('42'))
    assert lease.name == name
    assert await pool.cancel_and_release(lease) is False
    assert allocator.is_parked(name) is True


class TestAcquireForTask:
    def test_free_remote_is_leased_and_held_for_the_task(self) -> None:
        pool, allocator = _pool([_FakeRemote('laptop')], _Hooks())

        lease = _lease(pool.acquire_for_task('42'))

        assert lease.is_local is False
        assert lease.name == 'laptop'
        assert allocator.is_busy('laptop') is True
        assert allocator.is_busy('local') is False
        assert pool.holders() == {'laptop': 'task:42'}

    def test_no_remotes_refuses_no_host_without_reading_merge_demand(self) -> None:
        hooks = _Hooks(demand=[True])
        pool, _ = _pool([], hooks)

        assert pool.acquire_for_task('42') == Refusal('no_host')
        assert hooks.demand_calls == 0

    def test_merge_demand_refuses_and_leaves_the_slot_free(self) -> None:
        pool, allocator = _pool([_FakeRemote('laptop')], _Hooks(demand=[True]))

        assert pool.acquire_for_task('42') == Refusal('merge_demand')
        assert allocator.is_busy('laptop') is False
        assert pool.holders() == {}

    def test_merge_demand_is_read_on_every_call(self) -> None:
        hooks = _Hooks(demand=[True, False])
        pool, _ = _pool([_FakeRemote('laptop')], hooks)

        assert pool.acquire_for_task('42') == Refusal('merge_demand')
        assert _lease(pool.acquire_for_task('42')).name == 'laptop'
        assert hooks.demand_calls == 2

    def test_remote_busy_with_a_merge_lease_refuses_host_busy(self) -> None:
        pool, allocator = _pool([_FakeRemote('laptop')], _Hooks())
        assert allocator.acquire_remote() is not None

        assert pool.acquire_for_task('42') == Refusal('host_busy')
        assert pool.holders() == {}

    def test_quarantined_remote_refuses_host_quarantined(self) -> None:
        pool, allocator = _pool([_FakeRemote('laptop')], _Hooks(), quarantine={'laptop'})

        assert pool.acquire_for_task('42') == Refusal('host_quarantined')
        assert allocator.is_busy('laptop') is False

    @pytest.mark.parametrize(('enter_state', 'reason'), _HOST_STATES)
    def test_a_host_state_refusal_is_given_without_reading_merge_demand(
        self, enter_state: Callable[[HostAllocator, set[str]], None], reason: RefusalReason,
    ) -> None:
        hooks = _Hooks(demand=[True])
        quarantine: set[str] = set()
        pool, allocator = _pool([_FakeRemote('laptop')], hooks, quarantine=quarantine)
        enter_state(allocator, quarantine)

        assert pool.acquire_for_task('42') == Refusal(reason)
        assert hooks.demand_calls == 0

    def test_a_busy_remote_outranks_a_quarantined_one(self) -> None:
        pool, allocator = _pool(
            [_FakeRemote('A'), _FakeRemote('B')], _Hooks(), quarantine={'A'},
        )
        merge_lease = allocator.acquire_remote()
        assert merge_lease is not None
        assert merge_lease.name == 'B'

        assert pool.acquire_for_task('42') == Refusal('host_busy')
        assert pool.holders() == {}

    def test_allocator_is_the_one_passed_in(self) -> None:
        allocator = HostAllocator([_FakeRemote('laptop')])
        hooks = _Hooks()
        pool = HostPool(
            allocator,
            merge_demand=hooks.merge_demand,
            on_unavailable=hooks.on_unavailable,
            on_recovered=hooks.on_recovered,
        )

        assert pool.allocator is allocator


class TestHoldersAndRefusal:
    def test_holders_returns_a_copy(self) -> None:
        pool, _ = _pool([_FakeRemote('laptop')], _Hooks())
        _lease(pool.acquire_for_task('42'))

        snapshot = pool.holders()
        assert isinstance(snapshot, dict)
        snapshot['laptop'] = 'task:999'
        snapshot['other'] = 'task:1'

        assert pool.holders() == {'laptop': 'task:42'}

    def test_refusal_is_frozen(self) -> None:
        refusal = Refusal('no_host')

        pytest.raises(dataclasses.FrozenInstanceError, setattr, refusal, 'reason', 'merge_demand')

    @pytest.mark.parametrize('reason', ['busy_elsewhere', 'NO_HOST', '', 'pressure_hold'])
    def test_refusal_rejects_a_reason_outside_the_vocabulary(self, reason) -> None:
        with pytest.raises(ValueError, match=f"'{reason}'"):
            Refusal(reason)


@pytest.mark.asyncio
class TestReleaseAndCancel:
    async def test_parked_remote_refuses_host_parked(self) -> None:
        remote = _FakeRemote('laptop', cancel_rc=1, probe_clean=False)
        pool, allocator = _pool([remote], _Hooks())
        lease = _lease(pool.acquire_for_task('42'))

        assert await pool.cancel_and_release(lease) is False

        assert allocator.is_parked('laptop') is True
        assert pool.holders() == {}
        assert pool.acquire_for_task('43') == Refusal('host_parked')

    async def test_a_parked_refusal_is_given_without_reading_merge_demand(self) -> None:
        hooks = _Hooks(demand=[False, True])
        pool, allocator = _pool([_FakeRemote('laptop', cancel_rc=1, probe_clean=False)], hooks)
        await _park_through_a_failed_cancel(pool, allocator, 'laptop')
        demand_reads = hooks.demand_calls

        assert pool.acquire_for_task('43') == Refusal('host_parked')
        assert hooks.demand_calls == demand_reads

    async def test_a_quarantined_remote_outranks_a_parked_one(self) -> None:
        quarantine: set[str] = set()
        pool, allocator = _pool(
            [_FakeRemote('A', cancel_rc=1, probe_clean=False), _FakeRemote('B')], _Hooks(),
            quarantine=quarantine,
        )
        await _park_through_a_failed_cancel(pool, allocator, 'A')
        quarantine.add('B')

        assert pool.acquire_for_task('43') == Refusal('host_quarantined')
        assert pool.holders() == {}

    async def test_release_frees_the_slot_and_drops_the_hold(self) -> None:
        pool, allocator = _pool([_FakeRemote('laptop')], _Hooks())
        lease = _lease(pool.acquire_for_task('42'))

        await pool.release(lease)

        assert allocator.is_busy('laptop') is False
        assert pool.holders() == {}

    async def test_second_release_does_not_free_a_slot_reacquired_elsewhere(self) -> None:
        pool, allocator = _pool([_FakeRemote('laptop')], _Hooks())
        lease = _lease(pool.acquire_for_task('42'))
        await pool.release(lease)
        merge_lease = allocator.acquire_remote()
        assert merge_lease is not None

        await pool.release(lease)

        assert allocator.is_busy('laptop') is True
        assert pool.holders() == {}

    async def test_release_of_a_lease_the_pool_did_not_hand_out_is_a_no_op(self) -> None:
        pool, allocator = _pool([_FakeRemote('laptop')], _Hooks())
        merge_lease = allocator.acquire_remote()
        assert merge_lease is not None

        await pool.release(merge_lease)

        assert allocator.is_busy('laptop') is True

    async def test_clean_cancel_frees_the_slot_and_drops_the_hold(self) -> None:
        remote = _FakeRemote('laptop', cancel_rc=0)
        pool, allocator = _pool([remote], _Hooks())
        lease = _lease(pool.acquire_for_task('42'))

        assert await pool.cancel_and_release(lease) is True

        assert allocator.is_busy('laptop') is False
        assert pool.holders() == {}
        assert remote.cancel_calls == 1

    async def test_mark_unavailable_reports_without_quarantining(self) -> None:
        hooks = _Hooks()
        pool, allocator = _pool([_FakeRemote('laptop')], hooks)
        lease = _lease(pool.acquire_for_task('42'))

        unreachable = RunnerUnavailable('ssh exited 255', kind='transport')

        pool.mark_unavailable(lease, unreachable)

        assert hooks.unavailable == [('laptop', unreachable)]
        assert allocator.is_quarantined('laptop') is False
        assert allocator.is_busy('laptop') is True
        await pool.release(lease)
        assert allocator.is_busy('laptop') is False

    async def test_mark_recovered_reports_the_host(self) -> None:
        hooks = _Hooks()
        pool, _ = _pool([_FakeRemote('laptop')], hooks)
        lease = _lease(pool.acquire_for_task('42'))

        pool.mark_recovered(lease)

        assert hooks.recovered == ['laptop']
        await pool.release(lease)

    async def test_clean_cancel_unwinds_after_the_host_cancel_while_the_slot_is_held(self) -> None:
        remote = _FakeRemote('laptop', cancel_rc=0)
        pool, allocator = _pool([remote], _Hooks())
        lease = _lease(pool.acquire_for_task('42'))
        unwind = _RecordingUnwind(lambda: (
            remote.cancel_calls,
            allocator.is_busy('laptop'),
            allocator.is_parked('laptop'),
            dict(pool.holders()),
        ))

        assert await pool.cancel_and_release(lease, unwind=unwind) is True

        assert unwind.snapshots == [(1, True, False, {'laptop': 'task:42'})]
        assert allocator.is_busy('laptop') is False
        assert pool.holders() == {}

    async def test_failed_cancel_unwinds_before_the_park_and_the_probe(self) -> None:
        remote = _FakeRemote('laptop', cancel_rc=1, probe_clean=False)
        pool, allocator = _pool([remote], _Hooks())
        lease = _lease(pool.acquire_for_task('42'))
        unwind = _RecordingUnwind(lambda: (
            remote.cancel_calls,
            remote.probe_calls,
            allocator.is_parked('laptop'),
            dict(pool.holders()),
        ))

        assert await pool.cancel_and_release(lease, unwind=unwind) is False

        assert unwind.snapshots == [(1, 0, False, {'laptop': 'task:42'})]
        assert allocator.is_parked('laptop') is True
        assert pool.holders() == {}
        assert pool.acquire_for_task('43') == Refusal('host_parked')

    async def test_unwind_runs_once_for_a_lease_the_pool_does_not_hold(self) -> None:
        remote = _FakeRemote('laptop')
        pool, allocator = _pool([remote], _Hooks())
        merge_lease = allocator.acquire_remote()
        assert merge_lease is not None
        unwind = _RecordingUnwind(lambda: 'unwound')

        assert await pool.cancel_and_release(merge_lease, unwind=unwind) is True

        assert unwind.snapshots == ['unwound']
        assert remote.cancel_calls == 0
        assert allocator.is_busy('laptop') is True

    @pytest.mark.parametrize(('probe_clean', 'parked'), [(True, False), (False, True)])
    async def test_a_cancel_that_raises_is_a_failed_cancel_not_a_busy_leak(
        self, probe_clean: bool, parked: bool, caplog: pytest.LogCaptureFixture,
    ) -> None:
        remote = _FakeRemote('laptop', probe_clean=probe_clean, cancel_raises=RuntimeError('ssh died'))
        pool, allocator = _pool([remote], _Hooks())
        lease = _lease(pool.acquire_for_task('42'))
        unwind = _RecordingUnwind(lambda: 'unwound')
        caplog.set_level(logging.WARNING, logger='orchestrator.verify_hosts')

        assert await pool.cancel_and_release(lease, unwind=unwind) is False

        assert unwind.snapshots == ['unwound']
        assert remote.probe_calls >= 1
        assert allocator.is_parked('laptop') is parked
        assert allocator.is_busy('laptop') is parked
        assert pool.holders() == {}
        [warning] = [r for r in caplog.records if r.name == 'orchestrator.verify_hosts']
        assert warning.levelno == logging.WARNING
        assert 'laptop' in warning.getMessage()
