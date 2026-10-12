"""Composed-path admission scenarios of plans/verify-admission-fifo-prd.md, task α.

The record carriers (the test CheckRun's ``admission``, VerifyResult's
``admission_legs`` and the three EventTypes), then the gate composed with the
REAL flock and real subprocess commands through ``run_verification``,
``run_scoped_verification`` and ``run_full_verification``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from shared.verify_admission import acquire_task_slot

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.verify import VerifyResult, run_verification
from orchestrator.verify_admission_gate import gate_for

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

_LEG = {
    'plan_id': 'p', 'module': 'pkg', 'scope_kind': None, 'order': 'arrival',
    'arrived_at': '2026-10-12T00:00:00+00:00', 'granted_at': '2026-10-12T00:00:01+00:00',
    'waited_secs': 1.0, 'depth_at_arrival': 0, 'slot_held': True, 'held_secs': 2.5,
    'rc': 0, 'timed_out': False,
}


def _result(**overrides) -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='All checks passed',
        **overrides,
    )


class TestRecordCarriers:
    def test_admission_legs_default_to_none(self):
        assert _result().admission_legs is None

    def test_admission_legs_do_not_take_part_in_equality(self):
        assert _result(admission_legs=[_LEG]) == _result(admission_legs=None)

    def test_admission_legs_round_trip_through_the_wire_codec(self):
        result = _result(admission_legs=[_LEG, {**_LEG, 'module': 'other'}])

        decoded = VerifyResult(**dataclasses.asdict(result))

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
        assert leg['module'] == 'pkg'
        assert leg['slot_held'] is True
        assert (leg['rc'], leg['timed_out']) == (0, False)
        assert leg['held_secs'] > 0
        assert leg['order'] == 'unordered'
        assert leg['plan_id']

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
        assert leg['slot_held'] is False

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
        assert all(leg['timed_out'] for leg in result.admission_legs)
        assert len({leg['plan_id'] for leg in result.admission_legs}) == 1
