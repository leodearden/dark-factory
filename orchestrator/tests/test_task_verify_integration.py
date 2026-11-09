"""ι (task 6672): the remote task-verify integration gate, with every seam joined.

PRD ``plans/remote-task-leg-verify-prd.md`` §6 (the boundary rows) and §14
manifest resolutions 2-3.  Each producer pinned its own rows behind fakes;
here the same rows run against the real objects together: a real Harness
with its own HostPool, TaskVerifyDispatcher and (where a row needs one) merge
worker, real TaskWorkflows, and, as the remote host ``laptop``, the real
``orchestrator verify-task`` CLI on a fixture repo behind one scripted
ssh/git transport (``_task_verify_integration_rig``).

Row groups:

* A, ``TestLaptopCli``: the laptop CLI and its transport, rows 1, 2, 3, 4, 19, 24.
* B, ``TestHarnessDispatcher``: the harness dispatcher with no merge worker,
  rows 5, 8, 11, 15, 18, 21, 23, 25, 34.
* C, ``TestHostHealthWithTheMergeWorker``: host health with the real merge
  worker, rows 6, 7, 9, 12, 14, 20, 28, 32, 48.
* D, ``TestThroughTheTaskWorkflow``: through a real TaskWorkflow, rows 10, 13,
  16, 17, 22, 26, 30, 31.

Rows 8a, 27, 29, 33, 35-47 and 49-51 are the dispatcher's own race and stay
with ``test_task_verify_host_race.py``.

Three PRD-text drifts are asserted as shipped:

* Row 3: the CLI's exit 1 is RunnerUnavailable kind ``gate``, not
  ``transport``.  ``remote_gate.py::classify_remote_exit`` (the 6565 Q1
  ruling) keeps ``transport`` for ssh's own 255 and a spawn failure.
* Row 32: readmission goes through ``SpeculativeMergeWorker.readmit_host``,
  which replaced ``_readmit_probed_host`` (task 5290).
* Row 25: the per-module override refusal is logged once per dispatcher, at
  the first dispatch that meets it (``TaskVerifyDispatcher._warn_once``), not
  at config load.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT
from _task_verify_integration_rig import (
    MODULE,
    SECOND_MODULE,
    Exit,
    Hold,
    Subprocess,
    open_scene,
)
from _xdist_crash_fixtures import XDIST_WORKER_CRASH_OUTPUT

from orchestrator.task_verify_run import prepare_task_verify_worktree
from orchestrator.task_verify_spec import (
    build_task_verify_spec,
    task_spec_from_json,
    task_spec_to_json,
)
from orchestrator.verify import run_scoped_verification
from orchestrator.verify_cancel import pgid_file, read_worktree_path, worktree_path_file
from orchestrator.verify_runner import (
    ConcurrentDispatchError,
    RunnerUnavailable,
    build_merge_verify_spec,
    result_from_json,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('code_default_config')]

_CLI_TIMEOUT = pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)


def _verdict_fields(result) -> tuple[bool, str, str, bool]:
    return result.passed, result.category, result.summary, result.timed_out


def _pgid_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class TestLaptopCli:
    async def test_row1_a_spec_carrying_a_verify_env_round_trips_and_refuses_an_extra_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch) as scene:
            spec = build_task_verify_spec(
                scene.config, [scene.module(MODULE), scene.module(SECOND_MODULE)], (),
                force_workspace=False, attempt_id=1, task_id=scene.task_id, tip_sha=scene.tip,
            )

        assert spec.verify_env == scene.config.effective_verify_env
        assert spec.verify_env
        assert task_spec_from_json(task_spec_to_json(spec)) == spec
        wire = json.loads(task_spec_to_json(spec))
        wire['surprise'] = 1
        with pytest.raises(TypeError):
            task_spec_from_json(json.dumps(wire))

    @_CLI_TIMEOUT
    @pytest.mark.parametrize('passes', [True, False], ids=['green', 'red'])
    async def test_row2_the_laptop_verdict_is_the_workstations_verdict_on_the_same_sha(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passes: bool,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch) as scene:
            modules = [scene.module(MODULE, test='true' if passes else 'false')]
            request_id = scene.laptop.upcoming_request_id
            runner = scene.pool.runner_for_turn('laptop')

            remote = await runner.run_task_verify(
                scene.spec(modules), task_id=scene.task_id, archive_root=None,
            )

            second = tmp_path / 'second'
            subprocess.run(
                ['git', '-C', str(scene.repo), 'worktree', 'add', '--detach', str(second), scene.tip],
                check=True, capture_output=True,
            )
            prepare_task_verify_worktree(second)
            local = await run_scoped_verification(
                second, scene.config, modules, list(scene.task_files),
                attempt_id=1, task_id=scene.task_id, archive_root=None, role='task',
            )

        assert remote.passed is passes
        assert _verdict_fields(remote) == _verdict_fields(local)
        wire = result_from_json(scene.laptop.stdout_of(request_id))
        assert (wire.worktree_log_paths, wire.archive_log_paths) == ([], [])

    @_CLI_TIMEOUT
    @pytest.mark.parametrize(
        'workstation_provision,expected_marker',
        [(True, 'spec.marker'), (False, 'host.marker')],
        ids=['spec_command_replaces_host', 'empty_spec_command_keeps_host'],
    )
    async def test_row2_a_cold_laptop_worktree_runs_the_effective_provision_and_is_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        workstation_provision: bool, expected_marker: str,
    ) -> None:
        markers = tmp_path / 'markers'
        markers.mkdir()
        spec_provision = f'touch {shlex.quote(str(markers / "spec.marker"))}'
        async with open_scene(
            tmp_path, monkeypatch,
            provision=spec_provision if workstation_provision else '',
            laptop_provision=f'touch {shlex.quote(str(markers / "host.marker"))}',
        ) as scene:
            request_id = scene.laptop.upcoming_request_id
            pgf = pgid_file(scene.laptop_worktree_base, request_id)
            seen = tmp_path / 'seen.record'
            copy_record = (
                f'cp {shlex.quote(str(worktree_path_file(pgf)))} {shlex.quote(str(seen))}'
            )
            runner = scene.pool.runner_for_turn('laptop')

            result = await runner.run_task_verify(
                scene.spec([scene.module(MODULE, test=copy_record)]),
                task_id=scene.task_id, archive_root=None,
            )

        assert result.passed, result.summary
        assert sorted(p.name for p in markers.iterdir()) == [expected_marker]
        recorded = Path(seen.read_text().strip())
        assert recorded.parent == scene.laptop_worktree_base
        assert recorded.name.startswith('_merge-')
        assert not recorded.exists()
        assert not pgf.exists()
        assert not worktree_path_file(pgf).exists()

    @_CLI_TIMEOUT
    async def test_row3_an_xdist_crash_exits_1_and_is_a_gate_unavailability(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        crash = f'printf %s {shlex.quote(XDIST_WORKER_CRASH_OUTPUT)} && exit 1'
        async with open_scene(tmp_path, monkeypatch) as scene:
            request_id = scene.laptop.upcoming_request_id
            runner = scene.pool.runner_for_turn('laptop')

            with pytest.raises(RunnerUnavailable) as raised:
                await runner.run_task_verify(
                    scene.spec([scene.module(MODULE, test=crash)]),
                    task_id=scene.task_id, archive_root=None,
                )

        assert raised.value.kind == 'gate'
        assert scene.laptop.exit_code_of(request_id) == 1
        with pytest.raises(ValueError):
            result_from_json(scene.laptop.stdout_of(request_id))
        assert scene.laptop_merge_entries() == []

    @_CLI_TIMEOUT
    async def test_row4_a_signal_killed_leg_is_no_verdict_on_both_runner_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch) as scene:
            modules = [scene.module(MODULE, test='kill -9 $$')]
            request_id = scene.laptop.upcoming_request_id
            task_runner = scene.pool.runner_for_turn('laptop')

            with pytest.raises(RunnerUnavailable) as task_raised:
                await task_runner.run_task_verify(
                    scene.spec(modules), task_id=scene.task_id, archive_root=None,
                )

            stdout = scene.laptop.stdout_of(request_id)
            assert scene.laptop.exit_code_of(request_id) == 0
            await scene.start_worker()
            merge_runner = scene.worker.ensure_host_allocator(scene.config).remote_runner('laptop')
            assert merge_runner is not None
            scene.laptop.answers['verify-merge'] = Exit(0, stdout)

            with pytest.raises(RunnerUnavailable) as merge_raised:
                await merge_runner.run_merge_verify(
                    scene.tip, build_merge_verify_spec(scene.config, modules, scene.task_files),
                )

        assert task_raised.value.kind == 'no_verdict'
        assert merge_raised.value.kind == 'no_verdict'

    async def test_row19_a_second_dispatch_on_a_busy_runner_is_refused_before_any_push(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch) as scene:
            scene.laptop.answers['verify-task'] = Hold()
            runner = scene.pool.runner_for_turn('laptop')
            spec = scene.spec([scene.module(MODULE, test='true')])
            first = asyncio.create_task(
                runner.run_task_verify(spec, task_id=scene.task_id, archive_root=None),
            )
            request_id = await scene.laptop.until_verifying('verify-task')

            with pytest.raises(ConcurrentDispatchError):
                await runner.run_task_verify(spec, task_id=scene.task_id, archive_root=None)

            assert scene.laptop.labels().count('push') == 1
            assert runner.dispatch_in_flight
            scene.laptop.release(request_id)
            result = await asyncio.wait_for(first, 30)

        assert result.passed
        assert not runner.dispatch_in_flight

    @_CLI_TIMEOUT
    async def test_row24_a_cancelled_dispatch_kills_the_real_verify_task_and_reaps_its_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(
            tmp_path, monkeypatch, verify_task_host_policy='remote_all',
        ) as scene:
            scene.laptop.answers['verify-task'] = Subprocess()
            call = scene.start([scene.module(MODULE, test='sleep 300')])
            request_id = await scene.laptop.until_verifying('verify-task')
            pgf = pgid_file(scene.laptop_worktree_base, request_id)
            worktree = await scene.until_recorded_cold_worktree(pgf, watching=call)
            pgid = int(pgf.read_text().strip())

            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            await scene.until(lambda: _pgid_gone(pgid), what='the verify-task process group to die')
            await scene.until(lambda: scene.unwound(), what='the laptop turn to be left')

            assert scene.laptop.labels().count('cancel_verify') == 1
            assert not (worktree / '.task').exists()
            assert not worktree.exists()
            assert not pgf.exists()
            assert not worktree_path_file(pgf).exists()
            assert read_worktree_path(pgf) is None
