"""Composed-path admission scenarios of plans/verify-admission-fifo-prd.md, task α.

The record carriers (the test CheckRun's ``admission``, VerifyResult's
``admission_legs`` and the three EventTypes), then the gate composed with the
REAL flock and real subprocess commands through ``run_verification``,
``run_scoped_verification`` and ``run_full_verification``, and the plan row
each records.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from shared.verify_admission import acquire_task_slot

from orchestrator import verify as verify_module
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.verify import (
    VerifyInfraError,
    VerifyResult,
    run_full_verification,
    run_scoped_verification,
    run_verification,
)
from orchestrator.verify_admission_gate import gate_for
from orchestrator.verify_admission_record import AdmissionLeg
from orchestrator.verify_runner import result_from_json, result_to_json

_BLOCK_SRC = (
    'import os, sys, time\n'
    "open(sys.argv[2], 'w').close()\n"
    'deadline = time.monotonic() + 60\n'
    'while not os.path.exists(sys.argv[1]) and time.monotonic() < deadline:\n'
    '    time.sleep(0.02)\n'
)


def _config(tmp_path: Path, **overrides: Any) -> OrchestratorConfig:
    fields: dict[str, Any] = {
        'project_root': tmp_path,
        'verify_admission_slots_dir': str(tmp_path / 'slots'),
        'verify_admission_task_slots': 1,
        'verify_clock_stop_enabled': False,
        'verify_admission_nice_task': 'nice -n 10',
        'verify_admission_nice_background': 'nice -n 10',
    }
    fields.update(overrides)
    return OrchestratorConfig(**fields)


def _sleep_cmd(secs: float = 0.05) -> str:
    return f'{shlex.quote(sys.executable)} -c "import time; time.sleep({secs})"'


def _worker_crash_cmd(tmp_path: Path) -> str:
    """A test command whose output is a bare xdist worker crash, which verify
    raises as a VerifyInfraError for the workflow's infra retry."""
    script = tmp_path / 'crash.py'
    script.write_text("print('[gw0] node down: Not properly terminated')\nraise SystemExit(1)\n")
    return shlex.join([sys.executable, str(script)])


class _Blocker:
    """A test command that marks its start, then runs until the test releases it
    (bounded at 60s by the command itself, and by each test's own timeout)."""

    def __init__(self, tmp_path: Path, name: str) -> None:
        self._script = tmp_path / 'block.py'
        self._script.write_text(_BLOCK_SRC)
        self.started = tmp_path / f'{name}.started'
        self.release_file = tmp_path / f'{name}.release'

    @property
    def cmd(self) -> str:
        return shlex.join([sys.executable, str(self._script), str(self.release_file), str(self.started)])

    def release(self) -> None:
        self.release_file.touch()


def _module(prefix: str, test_command: str) -> ModuleConfig:
    return ModuleConfig(
        prefix=prefix, test_command=test_command, lint_command=None, type_check_command=None,
    )


def _worktree(tmp_path: Path, name: str) -> Path:
    worktree = tmp_path / name
    worktree.mkdir()
    return worktree


async def _until(
    predicate: Callable[[], bool], *, watching: asyncio.Task | None = None, timeout: float = 10.0,
) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            if watching is not None and watching.done():
                watching.result()
                raise AssertionError('the verify finished before reaching the awaited state')
            await asyncio.sleep(0.01)


async def _expect_cancelled(task: asyncio.Task) -> None:
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(10):
            await task


def _holders(slots_dir: Path) -> list[str | None]:
    return [h.request.module for h in gate_for(slots_dir).snapshot().holders]


def _slot_free_now(slots_dir: Path) -> bool:
    with acquire_task_slot('task', slots_dir=slots_dir, n=1, wait=False) as held:
        return held


async def _assert_no_slot_leaked(slots_dir: Path) -> None:
    """Probe without waiting until the slot is free, twice, a few poll periods
    apart, so a flock acquired late by a cancelled leg's thread is seen released."""
    await _until(lambda: _slot_free_now(slots_dir))
    await asyncio.sleep(0.3)
    await _until(lambda: _slot_free_now(slots_dir))

_LEG = AdmissionLeg(
    plan_id='p', module='pkg', scope_kind=None, order='arrival',
    arrived_at='2026-10-12T00:00:00+00:00', granted_at='2026-10-12T00:00:01+00:00',
    waited_secs=1.0, depth_at_arrival=0, slot_held=True, flock_wait_secs=0.5, held_secs=2.5,
    rc=0, timed_out=False,
)


def _result(**overrides) -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='All checks passed',
        **overrides,
    )


class TestRecordCarriers:
    def test_admission_legs_default_to_none(self):
        assert _result().admission_legs is None

    def test_admission_legs_do_not_take_part_in_equality(self):
        assert _result(admission_legs=(_LEG,)) == _result(admission_legs=None)

    def test_admission_legs_round_trip_through_the_wire_codec_as_records(self):
        result = _result(admission_legs=(_LEG, replace(_LEG, module='other')))

        decoded = result_from_json(result_to_json(result))

        assert decoded.admission_legs == result.admission_legs

    def test_the_admission_event_types_exist(self):
        assert {
            EventType('verify_admission_plan'),
            EventType('verify_admission_heartbeat'),
            EventType('verify_admission_wait_aged'),
        }


@pytest.mark.real_verify_admission
@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
class TestRunVerificationAdmission:
    async def test_a_task_verify_returns_its_admission_leg(self, tmp_path):
        config = _config(tmp_path)

        async with asyncio.timeout(30):
            result = await run_verification(
                _worktree(tmp_path, 'wt'), config, _module('pkg', _sleep_cmd()), task_id='7',
            )

        assert result.passed
        assert result.admission_legs is not None
        [leg] = result.admission_legs
        assert leg.module == 'pkg'
        assert leg.slot_held is True
        assert (leg.rc, leg.timed_out) == (0, False)
        assert leg.held_secs > 0
        assert leg.order == 'unordered'
        assert leg.plan_id

    @pytest.mark.skipif(os.geteuid() == 0, reason='root ignores directory permissions')
    async def test_an_unusable_slots_dir_fails_open_and_says_so(self, tmp_path):
        locked = tmp_path / 'locked'
        locked.mkdir()
        locked.chmod(0o500)
        config = _config(tmp_path, verify_admission_slots_dir=str(locked / 'slots'))
        try:
            async with asyncio.timeout(30):
                result = await run_verification(
                    _worktree(tmp_path, 'wt'), config, _module('pkg', _sleep_cmd()), task_id='9',
                )
        finally:
            locked.chmod(0o700)

        assert result.passed
        assert result.admission_legs is not None
        [leg] = result.admission_legs
        assert leg.slot_held is False

    async def test_a_merge_verify_never_touches_the_gate(self, tmp_path):
        config = _config(tmp_path)
        slots_dir = Path(config.verify_admission_slots_dir)
        blocker = _Blocker(tmp_path, 'merge')

        run = asyncio.create_task(run_verification(
            _worktree(tmp_path, 'wt'), config, _module('m', blocker.cmd), role='merge',
        ))
        try:
            await _until(blocker.started.exists, watching=run)
            snapshot = gate_for(slots_dir).snapshot()
            assert (snapshot.depth, snapshot.holders) == (0, ())
        finally:
            blocker.release()
        async with asyncio.timeout(30):
            result = await run

        assert result.passed
        assert result.admission_legs is None
        assert not slots_dir.exists()

    async def test_a_queued_verify_cancelled_leaves_and_the_next_is_granted(self, tmp_path):
        config = _config(tmp_path)
        slots_dir = Path(config.verify_admission_slots_dir)
        gate = gate_for(slots_dir)
        blocker = _Blocker(tmp_path, 'H')

        def start(name: str, cmd: str) -> asyncio.Task:
            return asyncio.create_task(run_verification(
                _worktree(tmp_path, name), config, _module(name, cmd), task_id=name,
            ))

        holder = start('H', blocker.cmd)
        try:
            await _until(lambda: _holders(slots_dir) == ['H'], watching=holder)
            cancelled = start('A', _sleep_cmd())
            await _until(lambda: gate.snapshot().depth == 1, watching=cancelled)
            survivor = start('B', _sleep_cmd())
            await _until(lambda: gate.snapshot().depth == 2, watching=survivor)

            cancelled.cancel()
            await _expect_cancelled(cancelled)
            await _until(lambda: gate.snapshot().depth == 1)
            assert [w.request.module for w in gate.snapshot().waiters] == ['B']
        finally:
            blocker.release()
        async with asyncio.timeout(30):
            results = await asyncio.gather(holder, survivor)

        assert all(r.passed for r in results)
        assert gate.snapshot().holders == ()
        await _assert_no_slot_leaked(slots_dir)

    async def test_an_admitted_verify_cancelled_mid_flock_poll_passes_its_turn_on(self, tmp_path):
        config = _config(tmp_path)
        slots_dir = Path(config.verify_admission_slots_dir)
        slots_dir.mkdir()
        gate = gate_for(slots_dir)

        def start(name: str) -> asyncio.Task:
            return asyncio.create_task(run_verification(
                _worktree(tmp_path, name), config, _module(name, _sleep_cmd()), task_id=name,
            ))

        with contextlib.ExitStack() as external:
            assert external.enter_context(
                acquire_task_slot('task', slots_dir=slots_dir, n=1, wait=False),
            ) is True
            polling = start('A')
            await _until(lambda: _holders(slots_dir) == ['A'], watching=polling)
            survivor = start('B')
            await _until(lambda: gate.snapshot().depth == 1, watching=survivor)

            polling.cancel()
            await _expect_cancelled(polling)
            await _until(lambda: _holders(slots_dir) == ['B'], watching=survivor)
        async with asyncio.timeout(30):
            result = await survivor

        assert result.passed
        await _assert_no_slot_leaked(slots_dir)

    async def test_a_legs_flock_wait_is_inside_its_held_turn(self, tmp_path):
        config = _config(tmp_path)
        slots_dir = Path(config.verify_admission_slots_dir)
        slots_dir.mkdir()

        with contextlib.ExitStack() as external:
            assert external.enter_context(
                acquire_task_slot('task', slots_dir=slots_dir, n=1, wait=False),
            ) is True
            run = asyncio.create_task(run_verification(
                _worktree(tmp_path, 'wt'), config, _module('pkg', _sleep_cmd()), task_id='f',
            ))
            await _until(lambda: _holders(slots_dir) == ['pkg'], watching=run)
            await asyncio.sleep(0.5)
        async with asyncio.timeout(30):
            result = await run

        assert result.admission_legs is not None
        [leg] = result.admission_legs
        assert leg.waited_secs < 0.5 <= leg.flock_wait_secs < leg.held_secs

    async def test_a_timeout_retry_is_a_second_leg_of_the_same_plan(self, tmp_path):
        config = _config(
            tmp_path, verify_command_timeout_secs=1, verify_cold_command_timeout_secs=1,
        )

        async with asyncio.timeout(60):
            result = await run_verification(
                _worktree(tmp_path, 'wt'), config, _module('pkg', _sleep_cmd(5)),
                task_id='r', max_retries=1,
            )

        assert result.timed_out
        assert result.admission_legs is not None
        assert len(result.admission_legs) == 2
        assert all(leg.timed_out for leg in result.admission_legs)
        assert len({leg.plan_id for leg in result.admission_legs}) == 1


def _plan_rows(store: EventStore) -> list[dict]:
    return store.fetch_events_by_type(EventType.verify_admission_plan)


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


@pytest.mark.real_verify_admission
@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
class TestPlanEvent:
    async def test_a_task_plan_emits_one_row_whose_legs_are_its_grants(self, tmp_path):
        config = _config(tmp_path, verify_admission_order='arrival')
        store = EventStore(tmp_path / 'runs.db', run_id='t')
        gate = gate_for(Path(config.verify_admission_slots_dir))
        blocker = _Blocker(tmp_path, 'both')
        plan = asyncio.create_task(run_scoped_verification(
            _worktree(tmp_path, 'wt'), config,
            [_module('a', blocker.cmd), _module('b', blocker.cmd)],
            task_files=[], task_id='11', role='task', event_store=store,
        ))
        try:
            await _until(lambda: gate.snapshot().depth == 1, watching=plan)
        finally:
            blocker.release()
        async with asyncio.timeout(30):
            result = await plan

        assert result.passed
        [row] = _plan_rows(store)
        assert row['task_id'] == '11'
        data = row['data']
        assert (data['role'], data['order'], data['task_id']) == ('task', 'arrival', '11')
        assert result.admission_legs is not None
        assert data['legs'] == [leg.to_dict() for leg in result.admission_legs]
        assert data['outcome'] == 'returned'
        assert {leg['plan_id'] for leg in data['legs']} == {data['plan_id']}
        assert {leg['module'] for leg in data['legs']} == {'a', 'b'}
        first, second = sorted(data['legs'], key=lambda leg: _at(leg['granted_at']))
        assert (first['depth_at_arrival'], second['depth_at_arrival']) == (0, 0)
        assert second['waited_secs'] > 0
        assert all(leg['slot_held'] and leg['held_secs'] > 0 for leg in data['legs'])
        assert all(_at(data['plan_arrival_at']) <= _at(leg['arrived_at']) for leg in data['legs'])
        assert _at(data['completed_at']) >= max(_at(leg['granted_at']) for leg in data['legs'])

    @pytest.mark.parametrize(
        ('role', 'overrides'),
        [('merge', {}), ('task', {'verify_admission_enabled': False})],
    )
    async def test_a_plan_with_no_gated_leg_emits_no_row(self, tmp_path, role, overrides):
        config = _config(tmp_path, **overrides)
        store = EventStore(tmp_path / 'runs.db', run_id='t')

        async with asyncio.timeout(30):
            result = await run_scoped_verification(
                _worktree(tmp_path, 'wt'), config, [_module('a', _sleep_cmd())],
                task_files=[], task_id='12', role=role, event_store=store,
            )

        assert result.passed
        assert result.admission_legs is None
        assert _plan_rows(store) == []

    @pytest.mark.parametrize(('overrides', 'rows'), [({}, 1), ({'verify_admission_enabled': False}, 0)])
    async def test_a_plan_that_raises_records_that_its_admission_went_unmeasured(
        self, tmp_path, overrides, rows,
    ):
        store = EventStore(tmp_path / 'runs.db', run_id='t')

        with pytest.raises(VerifyInfraError):
            async with asyncio.timeout(30):
                await run_scoped_verification(
                    _worktree(tmp_path, 'wt'), _config(tmp_path, **overrides),
                    [_module('a', _worker_crash_cmd(tmp_path))],
                    task_files=[], task_id='14', role='task', event_store=store,
                )

        assert len(_plan_rows(store)) == rows
        for row in _plan_rows(store):
            assert row['task_id'] == '14'
            data = row['data']
            assert (data['outcome'], data['legs'], data['order']) == ('raised', [], None)

    async def test_a_plan_without_an_event_store_still_records_its_legs(self, tmp_path):
        async with asyncio.timeout(30):
            result = await run_scoped_verification(
                _worktree(tmp_path, 'wt'), _config(tmp_path), [_module('a', _sleep_cmd())],
                task_files=[], task_id='13', role='task', event_store=None,
            )

        assert result.passed
        assert result.admission_legs is not None
        assert len(result.admission_legs) == 1

    async def test_a_plans_later_legs_keep_its_place_ahead_of_a_later_plan(self, tmp_path):
        config = _config(tmp_path, verify_admission_order='arrival', max_concurrent_module_verifies=4)
        gate = gate_for(Path(config.verify_admission_slots_dir))
        blocker = _Blocker(tmp_path, 'all')
        p_modules = [_module(f'p{i}', blocker.cmd) for i in range(1, 7)]

        p = asyncio.create_task(run_scoped_verification(
            _worktree(tmp_path, 'wt-p'), config, p_modules, task_files=[], task_id='P', role='task',
        ))
        q: asyncio.Task | None = None
        try:
            await _until(
                lambda: (len(gate.snapshot().holders), gate.snapshot().depth) == (1, 3), watching=p,
            )
            q = asyncio.create_task(run_scoped_verification(
                _worktree(tmp_path, 'wt-q'), config, [_module('q', blocker.cmd)],
                task_files=[], task_id='Q', role='task',
            ))
            await _until(lambda: gate.snapshot().depth == 4, watching=q)
        finally:
            blocker.release()
        assert q is not None
        async with asyncio.timeout(60):
            p_result, q_result = await asyncio.gather(p, q)

        assert p_result.passed and q_result.passed
        assert p_result.admission_legs is not None and q_result.admission_legs is not None
        assert len(p_result.admission_legs) == 6
        [q_leg] = q_result.admission_legs
        assert max(_at(leg.granted_at) for leg in p_result.admission_legs) < _at(q_leg.granted_at)

    async def test_a_file_scoped_leg_records_its_scope_kind(self, tmp_path):
        worktree = _worktree(tmp_path, 'wt')
        (worktree / 'pkg').mkdir()
        (worktree / 'pkg' / 'test_one.py').write_text('def test_ok():\n    pass\n')
        pytest_cmd = f'{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider'

        async with asyncio.timeout(60):
            result = await run_scoped_verification(
                worktree, _config(tmp_path), [_module('pkg', pytest_cmd)],
                task_files=['pkg/test_one.py'], task_id='sk', role='task',
            )

        assert result.admission_legs is not None
        [leg] = result.admission_legs
        assert leg.scope_kind == 'file_scoped'

    async def test_a_background_full_verification_emits_its_plan(self, tmp_path):
        config = _config(tmp_path)
        config._module_configs = {
            'a': _module('a', _sleep_cmd()), 'b': _module('b', _sleep_cmd()),
        }
        store = EventStore(tmp_path / 'runs.db', run_id='t')

        async with asyncio.timeout(30):
            result = await run_full_verification(
                tmp_path, config, role='background', event_store=store,
            )

        assert result.passed
        [row] = _plan_rows(store)
        assert row['data']['role'] == 'background'
        assert row['data']['task_id'] is None
        assert {leg['module'] for leg in row['data']['legs']} == {'a', 'b'}

    async def test_the_main_tip_sweep_hands_its_event_store_on(self, tmp_path):
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(
                main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees',
            ),
        )
        git_ops = GitOps(config.git, tmp_path)
        git_ops.get_main_sha = AsyncMock(return_value='a' * 40)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        store = EventStore(tmp_path / 'runs.db', run_id='t')
        seen: list[dict] = []

        async def fake_run(cmd, **kwargs):
            return (0, '', '')

        async def fake_full_verification(project_root, cfg, **kwargs):
            seen.append(kwargs)
            return VerifyResult(
                passed=True, test_output='', lint_output='', type_output='', summary='ok',
            )

        with (
            patch('orchestrator.git_ops._run', side_effect=fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=fake_full_verification),
        ):
            await verify_module.run_main_tip_sweep(config, git_ops, event_store=store)

        assert seen
        assert all(kwargs['event_store'] is store for kwargs in seen)
