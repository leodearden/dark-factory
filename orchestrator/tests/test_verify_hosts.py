"""HostPool: one admission gate per remote host, and the task path's hold records.

PRD ``plans/remote-task-leg-verify-prd.md`` decisions 14 and 16, §5.4.  Driven
through the pool's public API with REAL AdmissionGates; the cancel path is
fed small runners that implement only ``CancellableRunner``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _gate_fixtures import gated_remote_runner

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import admission_policy
from orchestrator.verify_admission_gate import AdmissionPlan, AdmissionPolicy
from orchestrator.verify_hosts import HostPool
from orchestrator.verify_runner import RemoteRunner, RunnerUnavailable


class _Runner:
    """A remote runner whose cancel rc and probe answers are fixed per test;
    every call is appended to the shared *log*."""

    def __init__(
        self, name: str, log: list[str], *, cancel_rc: int = 0,
        probes: list[bool] | None = None, cancel_raises: Exception | None = None,
    ) -> None:
        self.name = name
        self.log = log
        self.cancel_rc = cancel_rc
        self.cancel_raises = cancel_raises
        self.probes = list(probes or [])

    async def cancel_verify(self) -> int:
        self.log.append('cancel')
        if self.cancel_raises is not None:
            raise self.cancel_raises
        return self.cancel_rc

    async def probe_clean(self) -> bool:
        self.log.append('probe')
        return self.probes.pop(0) if self.probes else False


class _Hooks:
    def __init__(self) -> None:
        self.unavailable: list[tuple[str, RunnerUnavailable]] = []

    def on_unavailable(self, name: str, exc: RunnerUnavailable) -> None:
        self.unavailable.append((name, exc))


async def _no_sleep(_seconds: float) -> None:
    return None


def _remote_runner(name: str) -> RemoteRunner:
    return gated_remote_runner(name, name, f'ssh://{name}/repo.git', Path('/nonexistent'))


def _pool(
    hosts: tuple[str, ...] = ('laptop',),
    *,
    quarantine: set[str] | None = None,
    hooks: _Hooks | None = None,
    runner_for: Callable[[str], RemoteRunner] = _remote_runner,
    probe_attempts: int = 3,
) -> HostPool:
    return HostPool(
        hosts,
        runner_for=runner_for,
        quarantine=quarantine if quarantine is not None else set(),
        on_unavailable=(hooks or _Hooks()).on_unavailable,
        probe_sleep=_no_sleep,
        probe_attempts=probe_attempts,
    )


def _config(tmp_path: Path, **overrides: Any) -> OrchestratorConfig:
    fields: dict[str, Any] = {
        'project_root': tmp_path,
        'verify_admission_slots_dir': str(tmp_path / 'slots'),
        'verify_admission_task_slots': 1,
        'verify_admission_order': 'arrival',
    }
    fields.update(overrides)
    return OrchestratorConfig(**fields)


class TestHosts:
    def test_hosts_are_the_tuple_given_in_order(self):
        assert _pool(('laptop', 'desk')).hosts == ('laptop', 'desk')

    def test_no_hosts_is_allowed(self):
        assert _pool(()).hosts == ()

    def test_a_duplicate_host_is_refused(self):
        with pytest.raises(ValueError, match='laptop'):
            _pool(('laptop', 'desk', 'laptop'))

    def test_the_local_name_is_not_a_remote_host(self):
        with pytest.raises(ValueError, match="'local'"):
            _pool(('laptop', 'local'))

    def test_the_local_name(self):
        assert HostPool.local_name == 'local'
        assert _pool().local_name == 'local'

    @pytest.mark.parametrize('attempts', [0, -1])
    def test_a_cancel_must_be_probed_at_least_once(self, attempts: int):
        with pytest.raises(ValueError, match=f'probe_attempts.*{attempts}'):
            _pool(probe_attempts=attempts)


class TestGates:
    def test_one_gate_per_host(self):
        pool = _pool(('laptop', 'desk'))

        assert pool.gate('laptop') is pool.gate('laptop')
        assert pool.gate('laptop') is not pool.gate('desk')

    def test_an_unknown_host_has_no_gate(self):
        with pytest.raises(ValueError, match=r"'nope'.*'laptop'.*'desk'"):
            _pool(('laptop', 'desk')).gate('nope')


@pytest.mark.usefixtures('code_default_config')
class TestPolicy:
    def test_an_admitted_host_reads_the_local_policy(self, tmp_path):
        config = _config(
            tmp_path, verify_admission_task_slots=2, verify_admission_background_promote_secs=9.0,
        )

        assert _pool().policy('laptop', config) == admission_policy(config)

    def test_a_benched_host_reads_zero_slots(self, tmp_path):
        config = _config(tmp_path, verify_admission_task_slots=2)
        quarantine: set[str] = set()
        pool = _pool(quarantine=quarantine)

        quarantine.add('laptop')

        policy = pool.policy('laptop', config)
        assert policy == AdmissionPolicy(
            slots=0, order='arrival', promote_after_secs=policy.promote_after_secs,
        )
        assert policy.promote_after_secs == admission_policy(config).promote_after_secs

    def test_the_pool_never_writes_the_quarantine_set(self, tmp_path):
        quarantine = {'desk'}
        hooks = _Hooks()
        pool = _pool(('laptop', 'desk'), quarantine=quarantine, hooks=hooks)

        pool.policy('laptop', _config(tmp_path))
        pool.mark_unavailable('laptop', RunnerUnavailable('down', kind='transport'))
        pool.note_readmitted('desk', _config(tmp_path))

        assert quarantine == {'desk'}


class TestRunnerForTurn:
    def test_each_turn_gets_a_fresh_runner(self):
        built: list[str] = []

        def runner_for(host: str) -> RemoteRunner:
            built.append(host)
            return _remote_runner(host)

        pool = _pool(runner_for=runner_for)

        first, second = pool.runner_for_turn('laptop'), pool.runner_for_turn('laptop')

        assert first is not second
        assert built == ['laptop', 'laptop']

    def test_a_runner_named_otherwise_is_refused(self):
        pool = _pool(runner_for=lambda host: _remote_runner('desk'))

        with pytest.raises(ValueError, match=r"'desk'.*'laptop'|'laptop'.*'desk'"):
            pool.runner_for_turn('laptop')

    def test_an_unknown_host_has_no_runner(self):
        with pytest.raises(ValueError, match='nope'):
            _pool().runner_for_turn('nope')


async def _until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
class TestReadmission:
    async def test_a_readmitted_host_grants_its_oldest_waiter_at_once(self, tmp_path):
        config = _config(tmp_path)
        quarantine = {'laptop'}
        pool = _pool(quarantine=quarantine)
        gate = pool.gate('laptop')
        older = AdmissionPlan.begin(role='task', task_id='old')
        younger = AdmissionPlan.begin(role='task', task_id='young')
        release = asyncio.Event()

        async def wait_turn(plan: AdmissionPlan) -> None:
            request = plan.request(module=None, scope_kind=None)
            async with gate.admit(request, lambda: pool.policy('laptop', config)):
                await release.wait()

        waiters = [asyncio.create_task(wait_turn(p)) for p in (younger, older)]
        await _until(lambda: gate.snapshot().depth == 2)
        await asyncio.sleep(0.02)
        assert gate.snapshot().holders == ()

        quarantine.discard('laptop')
        pool.note_readmitted('laptop', config)

        assert [h.request.task_id for h in gate.snapshot().holders] == ['old']
        release.set()
        async with asyncio.timeout(5):
            await asyncio.gather(*waiters)

    async def test_readmission_while_still_benched_grants_nothing(self, tmp_path):
        config = _config(tmp_path)
        pool = _pool(quarantine={'laptop'})
        gate = pool.gate('laptop')
        request = AdmissionPlan.begin(role='task', task_id='7').request(module=None, scope_kind=None)

        async def wait_turn() -> None:
            async with gate.admit(request, lambda: pool.policy('laptop', config)):
                pass

        waiter = asyncio.create_task(wait_turn())
        await _until(lambda: gate.snapshot().depth == 1)

        pool.note_readmitted('laptop', config)

        assert gate.snapshot().holders == ()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter


class TestMarkUnavailable:
    def test_the_hook_is_called_once_with_the_host_and_error(self):
        hooks = _Hooks()
        quarantine: set[str] = set()
        exc = RunnerUnavailable('down', kind='transport')

        _pool(quarantine=quarantine, hooks=hooks).mark_unavailable('laptop', exc)

        assert hooks.unavailable == [('laptop', exc)]
        assert quarantine == set()


class TestHolding:
    def test_a_hold_shows_while_its_context_runs(self):
        pool = _pool()

        with pool.holding('laptop', '7'):
            assert pool.holders() == {'laptop': 'task:7'}
            with pool.holding('laptop', '8'):
                assert pool.holders() == {'laptop': 'task:7,task:8'}
            assert pool.holders() == {'laptop': 'task:7'}

        assert pool.holders() == {}

    def test_a_hold_ends_when_its_body_raises(self):
        pool = _pool()

        with pytest.raises(RuntimeError), pool.holding('laptop', '7'):
            raise RuntimeError('leg failed')

        assert pool.holders() == {}

    def test_holders_is_a_fresh_mapping(self):
        pool = _pool()

        with pool.holding('laptop', '7'):
            first = pool.holders()
            first['laptop'] = 'task:forged'

            assert pool.holders() == {'laptop': 'task:7'}
            assert pool.holders() is not pool.holders()


@pytest.mark.asyncio
class TestCancelInFlight:
    async def _cancel(
        self, *, cancel_rc: int = 0, probes: list[bool] | None = None,
        cancel_raises: Exception | None = None, probe_attempts: int = 3,
    ) -> tuple[bool, list[str], _Hooks, list[str]]:
        hooks = _Hooks()
        log: list[str] = []
        runner = _Runner(
            'laptop', log, cancel_rc=cancel_rc, probes=probes, cancel_raises=cancel_raises,
        )
        pool = _pool(hooks=hooks, probe_attempts=probe_attempts)

        async def unwind() -> None:
            log.append('unwind')

        clean = await pool.cancel_in_flight('laptop', runner, unwind=unwind)
        return clean, log, hooks, [name for name, _ in hooks.unavailable]

    async def test_a_clean_cancel_unwinds_and_probes_nothing(self):
        clean, log, _, benched = await self._cancel(cancel_rc=0)

        assert clean is True
        assert log == ['cancel', 'unwind']
        assert benched == []

    async def test_a_failed_cancel_whose_request_ends_is_clean(self):
        clean, log, _, benched = await self._cancel(cancel_rc=1, probes=[False, True])

        assert clean is True
        assert log == ['cancel', 'unwind', 'probe', 'probe']
        assert benched == []

    async def test_a_failed_cancel_never_probed_clean_benches_the_host(self):
        clean, log, hooks, benched = await self._cancel(cancel_rc=1, probe_attempts=4)

        assert clean is False
        assert log == ['cancel', 'unwind', 'probe', 'probe', 'probe', 'probe']
        assert benched == ['laptop']
        [(_, exc)] = hooks.unavailable
        assert exc.kind == 'transport'
        assert 'laptop' in str(exc) and 'rc=1' in str(exc)

    async def test_a_cancel_that_raises_is_a_failed_cancel(self, caplog):
        caplog.set_level(logging.WARNING)

        clean, log, _, benched = await self._cancel(
            cancel_raises=OSError('no route'), probes=[True],
        )

        assert clean is True
        assert log == ['cancel', 'unwind', 'probe']
        assert benched == []
        assert any('laptop' in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
