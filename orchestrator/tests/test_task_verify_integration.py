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
import fnmatch
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _host_health_helpers import transport_down
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT, wait_responsive
from _task_verify_integration_rig import (
    LAPTOP,
    MODULE,
    PROVISION_LEG,
    SECOND_MODULE,
    SSH_CONNECT_FAILURE,
    WORKFLOW_TASK_ID,
    Cli,
    Ensure,
    Exit,
    Hold,
    Raises,
    Scene,
    Subprocess,
    Turn,
    Verdict,
    WorkflowScene,
    open_scene,
    open_workflow_scene,
)
from _workflow_helpers import (
    PLAN,
    _derive_meta_root_like_production,
)
from _xdist_crash_fixtures import XDIST_WORKER_CRASH_OUTPUT

from orchestrator.config import load_config
from orchestrator.event_store import EventType
from orchestrator.remote_gate import local_gate_identity
from orchestrator.task_verify_run import prepare_task_verify_worktree
from orchestrator.task_verify_spec import (
    build_task_verify_spec,
    task_spec_from_json,
    task_spec_to_json,
)
from orchestrator.verify import VerifyResult, admission_active, run_scoped_verification
from orchestrator.verify_cancel import pgid_file, read_worktree_path, worktree_path_file
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
from orchestrator.verify_checkpoint import green_checkpoint_at_tip
from orchestrator.verify_runner import (
    ConcurrentDispatchError,
    RunnerUnavailable,
    build_merge_verify_spec,
    result_from_json,
)
from orchestrator.workflow_types import TerminalReport, WorkflowOutcome, WorkflowState

# Autouse; named here so the agent-side doubles resolve the meta_root production does.
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.usefixtures('code_default_config', _derive_meta_root_like_production.__name__),
]

_CLI_TIMEOUT = pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
_DISPATCH_LOGGER = 'orchestrator.task_verify_dispatch'
_REMOTE_MINE = {'verify_task_host_policy': 'remote_modules', 'verify_task_remote_modules': [MODULE]}
_TEST_LEG = f'test-{MODULE}'


def _verdict_fields(result) -> tuple[bool, str, str, bool]:
    return result.passed, result.category, result.summary, result.timed_out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _matches(names: list[str], *patterns: str) -> bool:
    """Each name matches exactly one pattern, and each pattern exactly one name."""
    return (
        len(names) == len(patterns)
        and all(len(fnmatch.filter(names, pattern)) == 1 for pattern in patterns)
        and all(sum(fnmatch.fnmatch(name, p) for p in patterns) == 1 for name in names)
    )


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


class TestHarnessDispatcher:
    """The harness's own dispatcher with no merge worker; policy from the config's verify_task_* keys."""

    @_CLI_TIMEOUT
    async def test_row5_an_eligible_verify_runs_on_the_laptop_and_never_starts_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            result = await scene.call([scene.module(MODULE)])

            assert result.passed, result.summary
            assert scene.legs.spawned_at('local') == []
            assert scene.legs.spawned_at('laptop') == sorted(
                f'{kind}-{MODULE}' for kind in ('lint', 'test', 'type')
            )
            sidecar = scene.sidecar()
            assert (sidecar['decision'], sidecar['runner'], sidecar['reason']) == (
                'remote', LAPTOP, 'remote',
            )
            assert sidecar['remote_wait_secs'] >= 0
            assert result.archive_log_paths
            assert all(
                Path(path).is_relative_to(scene.archive_root) for path in result.archive_log_paths
            )
            assert _matches(
                scene.archived_files(),
                'attempt-1.dispatch-*.json', f'attempt-1.remote-{LAPTOP}.pass-summary-*.json',
            ), scene.archived_files()
            assert scene.unwound(), scene.turns()

    @pytest.mark.parametrize('ineligible', ['other_module', 'force_workspace'])
    async def test_row8_an_ineligible_verify_runs_local_and_never_meets_the_laptop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ineligible: str,
    ) -> None:
        policy = (
            {'verify_task_host_policy': 'remote_modules', 'verify_task_remote_modules': [SECOND_MODULE]}
            if ineligible == 'other_module' else _REMOTE_MINE
        )
        async with open_scene(tmp_path, monkeypatch, **policy) as scene:
            result = await scene.call(
                [scene.module(MODULE)], force_workspace=ineligible == 'force_workspace',
            )

            assert result.passed, result.summary
            assert scene.legs.spawned_at('local')
            assert scene.legs.spawned_at('laptop') == []
            assert scene.sidecar()['reason'] == 'not_eligible'
            assert scene.laptop.calls == []
            assert scene.runners.built == []
            assert scene.unwound(), scene.turns()

    async def test_row11_a_dirty_tree_runs_local_before_anything_reaches_the_laptop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            (scene.worktree / 'untracked.txt').write_text('not committed\n')

            result = await scene.call([scene.module(MODULE)])

            assert result.passed, result.summary
            assert scene.legs.spawned_at('local')
            sidecar = scene.sidecar()
            assert (sidecar['decision'], sidecar['reason']) == ('local', 'dirty_tree')
            assert scene.laptop.calls == []
            assert scene.unwound(), scene.turns()

    @_CLI_TIMEOUT
    async def test_row15_every_second_remote_verdict_is_cross_checked_locally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(
            tmp_path, monkeypatch, verify_task_remote_cross_check_every_n=2, **_REMOTE_MINE,
        ) as scene:
            scene.legs.fail(_TEST_LEG, 'local')
            modules = [scene.module(MODULE)]

            first = await scene.call(modules, attempt_id=1)

            assert first.passed, first.summary
            assert scene.legs.spawned_at('local') == []
            assert (scene.sidecar(1)['reason'], scene.sidecar(1)['cross_check']) == ('remote', None)

            second = await scene.call(modules, attempt_id=2)

            assert not second.passed
            assert scene.legs.spawned(_TEST_LEG, 'local')
            assert len(scene.legs.pids(_TEST_LEG, 'laptop')) == 2
            sidecar = scene.sidecar(2)
            assert (sidecar['runner'], sidecar['reason']) == ('local', 'cross_check')
            assert sidecar['cross_check'] == {
                'remote_passed': True, 'local_passed': False, 'agree': False,
            }

    @_CLI_TIMEOUT
    async def test_row18_a_reload_flips_the_next_dispatch_and_an_empty_module_list_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch) as scene:
            modules = [scene.module(MODULE)]

            flip = scene.reload(**_REMOTE_MINE)

            assert flip['applied']['verify_task_host_policy'] == {
                'old': 'local', 'new': 'remote_modules',
            }
            await scene.call(modules, attempt_id=1)
            assert scene.sidecar(1)['reason'] == 'remote'

            back = scene.reload(verify_task_host_policy='local', verify_task_remote_modules=[])

            assert back['applied']['verify_task_host_policy']['new'] == 'local'
            await scene.call(modules, attempt_id=2)
            assert scene.sidecars(2) == []
            assert scene.laptop.requests('verify_task') == ['req1']

        refused = tmp_path / 'refused.yaml'
        refused.write_text(
            'verify_task_host_policy: remote_modules\nverify_task_remote_modules: []\n',
        )
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(refused))
        with pytest.raises(ValueError) as raised:
            load_config(refused)
        assert 'verify_task_host_policy' in str(raised.value)
        assert 'verify_task_remote_modules' in str(raised.value)

    async def test_row21_a_remote_verify_that_raises_propagates_and_leaves_the_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            scene.laptop.answers['verify-task'] = Raises(RuntimeError('the laptop shim blew up'))

            with pytest.raises(RuntimeError, match='the laptop shim blew up'):
                await scene.call([scene.module(MODULE)])

            assert scene.sidecar()['reason'] == 'dispatcher_error'
            assert scene.unwound(), scene.turns()

    async def test_row23_a_local_policy_forwards_the_call_unchanged_and_records_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch) as scene:
            result = await scene.call([scene.module(MODULE)], attempt_id=3)

            assert result.passed, result.summary
            assert scene.legs.spawned_at('local') == sorted(
                f'{kind}-{MODULE}' for kind in ('lint', 'test', 'type')
            )
            assert scene.sidecars(3) == []
            assert scene.dispatcher.decision_for(scene.task_id, 3) is None
            assert scene.laptop.calls == []
            assert scene.runners.built == []
            assert _matches(
                scene.archived_files(),
                f'attempt-3.{MODULE}.summary-*.json', 'attempt-3.plan-*.json',
            ), scene.archived_files()

    async def test_row25_a_per_module_verify_env_keeps_every_dispatch_local_and_warns_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        async with open_scene(
            tmp_path, monkeypatch, verify_task_host_policy='remote_all',
        ) as scene:
            modules = [scene.module(MODULE, verify_env={'MOD_ONLY': '1'})]

            with caplog.at_level(logging.WARNING, logger=_DISPATCH_LOGGER):
                await scene.call(modules, attempt_id=1)
                await scene.call(modules, attempt_id=2)

            assert [scene.sidecar(n)['reason'] for n in (1, 2)] == ['not_eligible'] * 2
            warnings = [
                r for r in caplog.records
                if r.name == _DISPATCH_LOGGER and r.levelno == logging.WARNING
            ]
            assert len(warnings) == 1
            assert repr(MODULE) in warnings[0].getMessage()
            assert scene.laptop.calls == []

    async def test_row34_an_ungated_local_leg_keeps_the_verify_once_the_laptop_frees(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(
            tmp_path, monkeypatch, verify_task_host_policy='remote_all',
        ) as scene:
            assert not admission_active(scene.config)
            holder = scene.hold()
            await wait_responsive(holder.granted.wait(), label='the test to hold the laptop turn')
            scene.legs.block(_TEST_LEG, 'local')
            call = scene.start([scene.module(MODULE)])
            await scene.until(
                lambda: scene.legs.spawned(_TEST_LEG, 'local'), what='the local test leg to spawn',
            )

            await holder.leave()
            await scene.until(
                lambda: scene.turns()['laptop_waiters'] == [], what='the attempt to leave the laptop',
            )

            [test_pid] = scene.legs.pids(_TEST_LEG, 'local')
            assert _alive(test_pid)
            assert scene.laptop.requests('verify_task') == []
            scene.legs.release(_TEST_LEG, 'local')
            result = await asyncio.wait_for(call, 60)

            assert result.passed, result.summary
            assert scene.sidecar()['reason'] == 'local_granted_first'
            assert scene.laptop.calls == []
            assert scene.unwound(), scene.turns()


_REAL = pytest.mark.real_verify_admission
_SSH_DOWN = Exit(255, '', SSH_CONNECT_FAILURE)
_UNREACHABLE = 'verify_host_unreachable'


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
class TestHostHealthWithTheMergeWorker:
    """The harness's own merge worker, started, beside its task-path pool."""

    async def test_row6_a_busy_merge_slot_leaves_the_task_turn_free(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            await scene.start_worker()
            allocator = scene.worker.ensure_host_allocator(scene.config)
            lease = allocator.acquire_remote()
            assert lease is not None and lease.name == LAPTOP
            try:
                result = await scene.call([scene.module(MODULE)])

                assert result.passed, result.summary
                assert scene.sidecar()['reason'] == 'remote'
                assert allocator.is_busy(LAPTOP)
                assert scene.laptop.requests('verify_merge') == []
            finally:
                await allocator.release(lease)

    async def test_row7_a_merge_verify_runs_on_the_host_a_task_holds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            await scene.start_worker()
            scene.laptop.answers['verify-task'] = Hold()
            scene.laptop.answers['verify-merge'] = Hold()
            call = scene.start([scene.module(MODULE)])
            task_request = await scene.laptop.until_verifying('verify-task')

            merge = await scene.merge('m7')
            merge_request = await scene.laptop.until_verifying('verify-merge')

            assert merge_request != task_request
            row = scene.laptop_row()
            assert (row['slot_state'], row['task_held_by']) == ('busy', f'task:{scene.task_id}')
            scene.laptop.release(merge_request)
            scene.laptop.release(task_request)
            outcome = await scene.end_merge(merge)
            result = await asyncio.wait_for(call, 60)

            assert outcome == 'done'
            assert result.passed, result.summary
            assert scene.sidecar()['reason'] == 'remote'

    @pytest.mark.parametrize('escalate_after_n', [1, 2])
    async def test_row9_an_unreachable_laptop_benches_ru_and_the_sweep_readmits_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, escalate_after_n: int,
    ) -> None:
        async with open_scene(
            tmp_path, monkeypatch,
            verify_host_unreachable_escalate_after_n=escalate_after_n, **_REMOTE_MINE,
        ) as scene:
            await scene.start_worker()
            scene.laptop.healthy = False
            scene.laptop.answers['verify-task'] = _SSH_DOWN

            result = await scene.call([scene.module(MODULE)], attempt_id=1)

            assert result.passed, result.summary
            assert scene.legs.spawned_at('local')
            assert scene.sidecar(1)['reason'] == 'runner_unavailable'
            row = scene.laptop_row()
            assert (row['quarantined'], row['quarantine_class'], row['streak']) == (True, 'ru', 1)
            assert bool(scene.escalations(_UNREACHABLE)) is (escalate_after_n == 1)

            scene.laptop.answers['verify-task'] = Cli()
            scene.laptop.healthy = True
            await scene.until(
                lambda: not scene.laptop_row()['quarantined'], what='the sweep to readmit the laptop',
            )
            await scene.call([scene.module(MODULE)], attempt_id=2)

            assert scene.sidecar(2)['reason'] == 'remote'
            assert scene.laptop.requests('verify_merge') == []

    @pytest.mark.parametrize('failure', ['ensure_rc_1', 'identity', 'ensure_rc_255', 'publish'])
    async def test_row12_a_gate_that_cannot_be_ensured_benches_by_kind_before_any_verify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
    ) -> None:
        skew = failure in {'ensure_rc_1', 'identity'}
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            await scene.start_worker()
            scene.laptop.healthy = False
            if failure == 'publish':
                scene.laptop.publish_connect_fails = True
            else:
                scene.laptop.ensure = {
                    'ensure_rc_1': Ensure(rc=1),
                    'identity': Ensure(identity=local_gate_identity('e' * 40)),
                    'ensure_rc_255': Ensure(rc=255),
                }[failure]

            result = await scene.call([scene.module(MODULE)], attempt_id=1)

            assert result.passed, result.summary
            assert scene.sidecar(1)['reason'] == 'runner_unavailable'
            assert scene.laptop.requests('verify_task') == []
            row = scene.laptop_row()
            assert (row['quarantined'], row['quarantine_class']) == (True, 'skew' if skew else 'ru')
            assert scene.events(EventType.runner_stale) == []

            since_recovery = len(scene.laptop.calls)
            scene.laptop.publish_connect_fails = False
            scene.laptop.ensure = Ensure()
            if not skew:
                scene.laptop.healthy = True
            await scene.until(
                lambda: not scene.laptop_row()['quarantined'], what='the sweep to readmit the laptop',
            )
            readmitting_probe = 'ensure' if skew else 'health'
            assert readmitting_probe in scene.laptop.labels()[since_recovery:], scene.laptop.labels()
            assert scene.events(EventType.runner_stale) == []
            scene.laptop.ensure = Ensure(built=True)
            await scene.call([scene.module(MODULE)], attempt_id=2)

            assert scene.sidecar(2)['reason'] == 'remote'
            assert len(scene.events(EventType.runner_stale)) == 1

    async def test_row14_a_failed_cancel_holds_the_turn_through_the_probe_then_benches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            await scene.start_worker()
            scene.laptop.healthy = False
            scene.laptop.answers['verify-task'] = Hold()
            scene.laptop.cancel_rc, scene.laptop.pgrep_rc = 1, 0
            scene.laptop.on_pgrep = scene.pool.holders
            call = scene.start([scene.module(MODULE)], attempt_id=1)
            await scene.laptop.until_verifying('verify-task')

            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            await scene.until(
                lambda: scene.laptop_row()['quarantined'], what='the failed cancel to bench the laptop',
            )
            await scene.until(scene.unwound, what='the laptop turn to be left')

            assert scene.laptop.pgrep_seen
            assert all(seen == {LAPTOP: f'task:{scene.task_id}'} for seen in scene.laptop.pgrep_seen)
            assert scene.laptop_row()['quarantine_class'] == 'ru'
            await scene.call([scene.module(MODULE)], attempt_id=2)
            assert scene.sidecar(2)['reason'] == 'local_granted_first'

    async def test_row20_one_quarantine_set_survives_a_worker_rebuild_and_a_refused_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            await scene.start_worker()
            scene.laptop.healthy = False

            scene.pool.mark_unavailable(LAPTOP, transport_down())

            assert scene.worker.ensure_host_allocator(scene.config).is_quarantined(LAPTOP)
            assert scene.pool.policy(LAPTOP, scene.config).slots == 0
            await scene.stop_worker()
            await scene.start_worker()
            assert scene.laptop_row()['quarantined'] is True
            await scene.stop_worker()

            with patch('orchestrator.merge_lane.liveness.INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS', 1.0), \
                 pytest.raises(Exception, match='enforce_merge_liveness_margin: startup refused'):
                await scene.start_worker()

            assert scene.pool.hosts == (LAPTOP,)
            assert scene.pool.policy(LAPTOP, scene.config).slots == 0
            result = await scene.call([scene.module(MODULE)])
            assert result.passed, result.summary
            assert scene.sidecar()['reason'] == 'local_granted_first'

    @_REAL
    @pytest.mark.parametrize('cancelled', ['task', 'merge'])
    async def test_row28_a_merge_and_a_task_verify_run_on_the_laptop_at_once_and_cancel_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: str,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            await scene.start_worker()
            scene.laptop.answers['verify-task'] = Hold()
            scene.laptop.answers['verify-merge'] = Hold()
            merge = await scene.merge('m28')
            merge_request = await scene.laptop.until_verifying('verify-merge')
            call = scene.start([scene.module(MODULE)])
            task_request = await scene.laptop.until_verifying('verify-task')
            merge_runner = scene.worker.ensure_host_allocator(scene.config).remote_runner(LAPTOP)
            assert merge_runner is not None
            task_runner = scene.runners.built[-1]

            assert task_runner is not merge_runner
            assert task_request != merge_request
            assert merge_runner.dispatch_in_flight and task_runner.dispatch_in_flight

            if cancelled == 'task':
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
                await scene.until(scene.unwound, what='the task turn to be left')
                assert merge_runner.dispatch_in_flight
                assert merge_request not in scene.laptop.requests('hold_ended')
                scene.laptop.release(merge_request)
                await scene.end_merge(merge)
            else:
                assert scene.worker.cancel_request(merge.request_id).refusal is None
                await scene.until(
                    lambda: merge_request in scene.laptop.requests('hold_ended'),
                    what='the merge verify to unwind',
                )
                assert task_runner.dispatch_in_flight
                assert task_request not in scene.laptop.requests('hold_ended')
                scene.laptop.release(task_request)
                assert (await asyncio.wait_for(call, 60)).passed

    @_REAL
    async def test_row32_a_readmitted_host_gives_its_oldest_waiter_the_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            await scene.start_worker()
            scene.laptop.healthy = False
            scene.laptop.answers['verify-task'] = Hold(Cli())
            local_holder = scene.hold_local()
            await wait_responsive(local_holder.granted.wait(), label='the test to hold the local turn')
            scene.pool.mark_unavailable(LAPTOP, transport_down())
            assert scene.pool.policy(LAPTOP, scene.config).slots == 0
            call = scene.start([scene.module(MODULE)])
            await scene.until(
                lambda: scene.task_id in scene.turns()['laptop_waiters']
                and scene.task_id in scene.turns()['local_waiters'],
                what='the call to queue on both gates',
            )
            younger = scene.hold(task_id='younger')
            await scene.until(
                lambda: 'younger' in scene.turns()['laptop_waiters'],
                what='a younger laptop waiter behind the call',
            )

            scene.laptop.healthy = True
            request_id = await scene.laptop.until_verifying('verify-task')

            assert scene.turns()['laptop_waiters'] == ['younger']
            assert not younger.granted.is_set()
            scene.laptop.release(request_id)
            result = await asyncio.wait_for(call, 60)
            await wait_responsive(younger.granted.wait(), label='the younger waiter to get the freed turn')
            await younger.leave()

            assert result.passed, result.summary
            assert scene.sidecar()['reason'] == 'remote'
            assert not scene.legs.spawned(_TEST_LEG, 'local')
            assert scene.legs.spawned(_TEST_LEG, 'laptop')
            assert not scene.laptop_row()['quarantined']
            assert scene.unwound(local_holders=['local-holder']), scene.turns()
            await local_holder.leave()

    @_REAL
    async def test_row48_a_merge_bench_during_a_task_verify_outlives_the_tasks_green(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_scene(tmp_path, monkeypatch, **_REMOTE_MINE) as scene:
            await scene.start_worker()
            scene.laptop.healthy = False
            scene.laptop.answers['verify-task'] = Hold()
            scene.laptop.answers['verify-merge'] = _SSH_DOWN
            call = scene.start([scene.module(MODULE)])
            task_request = await scene.laptop.until_verifying('verify-task')

            merge = await scene.merge('m48')
            await scene.until(
                lambda: scene.laptop_row()['quarantined'], what='the merge to bench the laptop',
            )
            scene.laptop.release(task_request)
            result = await asyncio.wait_for(call, 60)

            assert result.passed, result.summary
            assert scene.sidecar()['reason'] == 'remote'
            row = scene.laptop_row()
            assert (row['quarantined'], row['quarantine_class'], row['streak']) == (True, 'ru', 1)
            await scene.end_merge(merge)
            scene.laptop.healthy = True
            await scene.until(
                lambda: not scene.laptop_row()['quarantined'], what='the sweep to readmit the laptop',
            )


_REMOTE_ALL: dict[str, Any] = {'verify_task_host_policy': 'remote_all'}
_LINT, _TEST = 'lint', 'test'
"""The workstation-wide legs a workflow verify runs (the workflow config defines no modules)."""
_INFRA = VerifyResult(
    passed=False, test_output='', lint_output='', type_output='', summary='host out of space',
    category=sorted(INFRA_TRANSIENT_CATEGORIES)[0],
)
_GREEN = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='', summary='All checks passed',
)
_RACE_WORKFLOW: dict[str, Any] = {'concurrent_verify': True, 'verify_admission_enabled': True, **_REMOTE_ALL}
_ENSURE_FAILS = Ensure(rc=1)
"""Keeps a skew-benched laptop benched: the reprobe sweep readmits one only once ensure succeeds."""


async def _hold_both(scene: Scene) -> tuple[Turn, Turn]:
    """The laptop's turn and the local turn, both held by the test."""
    laptop, local = scene.hold(), scene.hold_local()
    await wait_responsive(laptop.granted.wait(), label='the test to hold the laptop turn')
    await wait_responsive(local.granted.wait(), label='the test to hold the local turn')
    return laptop, local


def _local_waiter(scene: Scene, task_id: str = WORKFLOW_TASK_ID) -> tuple[str, float]:
    """The plan id and arrival *task_id* waits under on the local gate."""
    [request] = [
        view.request for view in scene.local_gate.snapshot().waiters
        if view.request.task_id == task_id
    ]
    return request.plan_id, request.plan_arrival


async def _end(flow: WorkflowScene) -> TerminalReport:
    """Soft-cancel a run parked at MERGE; its report."""
    flow.soft_cancel()
    return await flow.report()


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
class TestThroughTheTaskWorkflow:
    """A real TaskWorkflow driven by AgentStub into its real VERIFY, built with the harness's dispatcher."""

    async def test_row10_an_infra_transient_remote_verdict_reruns_locally_inside_the_same_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_workflow_scene(tmp_path, monkeypatch, **_REMOTE_ALL) as flow:
            scene = flow.scene
            await scene.start_worker()
            scene.laptop.answers['verify-task'] = Hold(Verdict(_INFRA))
            flow.start()
            request_id = await scene.laptop.until_verifying('verify-task')
            scene.laptop.ensure = _ENSURE_FAILS
            scene.laptop.release(request_id)

            await flow.until_parked_at_merge()

            [sidecar] = scene.sidecars(1)
            assert (sidecar['reason'], sidecar['runner']) == ('infra_transient', 'local')
            assert scene.laptop.requests('verify_task') == [request_id]
            assert scene.legs.spawned(_TEST, 'local')
            [row] = flow.verify_rows()
            assert (row['passed'], row['runner']) == (True, 'local')
            laptop = scene.laptop_row()
            assert (
                laptop['quarantined'], laptop['quarantine_class'], laptop['kind'], laptop['streak'],
            ) == (True, 'skew', 'no_verdict', 1)
            await _end(flow)

    async def test_row13_a_soft_cancel_mid_remote_verify_cancels_it_on_the_laptop_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_workflow_scene(tmp_path, monkeypatch, **_REMOTE_ALL) as flow:
            scene = flow.scene
            scene.laptop.answers['verify-task'] = Hold()
            flow.start()
            request_id = await scene.laptop.until_verifying('verify-task')

            flow.soft_cancel()
            report = await flow.report()

            assert (report.outcome, report.phase) == (
                WorkflowOutcome.SOFT_CANCELLED, WorkflowState.CANCELLED,
            )
            calls = [(call.label, call.request_id) for call in scene.laptop.calls]
            assert calls.index(('cancel_verify', request_id)) < calls.index(
                ('hold_ended', request_id),
            )
            assert scene.unwound(), scene.turns()

    async def test_row16_a_workflow_without_the_dispatcher_calls_the_module_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_workflow_scene(tmp_path, monkeypatch, **_REMOTE_ALL) as flow:
            scene = flow.scene
            module_verify = AsyncMock(return_value=_GREEN)
            monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', module_verify)
            flow.start(with_dispatcher=False)

            await flow.until_parked_at_merge()

            assert flow.workflow is not None
            [call] = module_verify.await_args_list
            assert (call.args, call.kwargs) == ((flow.workflow.worktree, scene.config, []), {
                'task_files': PLAN['files'],
                'attempt_id': 1,
                'task_id': WORKFLOW_TASK_ID,
                'archive_root': scene.archive_root,
                'force_workspace': False,
                'role': 'task',
                'event_store': scene.harness.event_store,
            })
            assert scene.dispatcher.decision_for(WORKFLOW_TASK_ID, 1) is None
            assert scene.sidecars(1) == []
            assert scene.laptop.calls == []
            assert scene.runners.built == []
            assert scene.unwound(), scene.turns()
            await _end(flow)

    async def test_row17_a_remote_green_names_the_laptop_beside_the_worktree_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_workflow_scene(tmp_path, monkeypatch, **_REMOTE_ALL) as flow:
            scene = flow.scene
            flow.start()

            await flow.until_parked_at_merge()

            head = scene.tip
            [row] = flow.verify_rows()
            assert (row['passed'], row['runner'], row['tip_sha'], row['site']) == (
                True, LAPTOP, head, 'implement',
            )
            assert green_checkpoint_at_tip(
                scene.harness.event_store, EventType.workflow_verify, WORKFLOW_TASK_ID, head,
            ) is True
            assert scene.sidecar(1)['reason'] == 'remote'
            assert scene.legs.spawned_at('local') == []
            assert scene.legs.spawned_at('laptop') == [_LINT, _TEST, 'type']
            assert (await _end(flow)).outcome == WorkflowOutcome.SOFT_CANCELLED

    async def test_row22_a_laptop_red_confirmed_green_locally_never_reaches_the_debugger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_workflow_scene(
            tmp_path, monkeypatch, verify_task_remote_red_confirms_locally=True, **_REMOTE_ALL,
        ) as flow:
            scene = flow.scene
            scene.legs.fail(_TEST, 'laptop')
            flow.start()

            await flow.until_parked_at_merge()

            assert 'debugger' not in flow.agents.calls
            assert scene.legs.spawned(_TEST, 'laptop') and scene.legs.spawned(_TEST, 'local')
            sidecar = scene.sidecar(1)
            assert (sidecar['runner'], sidecar['reason']) == ('local', 'red_confirm')
            assert sidecar['red_confirmed_locally'] == {'remote_passed': False, 'local_passed': True}
            [row] = flow.verify_rows()
            assert (row['passed'], row['runner']) == (True, 'local')
            await _end(flow)

    @_REAL
    async def test_row26_a_laptop_turn_freed_while_queued_withdraws_the_local_verify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_workflow_scene(tmp_path, monkeypatch, **_RACE_WORKFLOW) as flow:
            scene = flow.scene
            scene.legs.block(_LINT)
            laptop_holder, local_holder = await _hold_both(scene)
            flow.start()
            await flow.until_queued_on_both()
            await flow.until(lambda: scene.legs.spawned(_LINT), what='the local lint leg to spawn')
            [lint_pid] = scene.legs.pids(_LINT)

            await laptop_holder.leave()
            await flow.until_parked_at_merge()

            assert not scene.legs.spawned(_TEST)
            assert not _alive(lint_pid)
            assert WORKFLOW_TASK_ID not in scene.turns()['local_waiters']
            assert scene.events(EventType.verify_admission_plan) == []
            sidecar = scene.sidecar(1)
            assert (sidecar['decision'], sidecar['runner']) == ('remote', LAPTOP)
            assert sidecar['remote_wait_secs'] > 0
            assert scene.unwound(local_holders=['local-holder']), scene.turns()
            await _end(flow)
            await local_holder.leave()

    @_REAL
    async def test_row30_an_unreachable_laptop_after_withdrawal_reenters_at_the_original_arrival(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async with open_workflow_scene(tmp_path, monkeypatch, **_RACE_WORKFLOW) as flow:
            scene = flow.scene
            await scene.start_worker()
            scene.laptop.healthy = False
            scene.laptop.answers['verify-task'] = _SSH_DOWN
            scene.legs.block(_TEST)
            laptop_holder, local_holder = await _hold_both(scene)
            flow.start()
            await flow.until_queued_on_both()
            arrival = _local_waiter(scene)
            younger = scene.hold_local(task_id='younger')
            await flow.until(
                lambda: 'younger' in scene.turns()['local_waiters'], what='a younger local waiter',
            )

            await laptop_holder.leave()
            await flow.until(
                lambda: scene.laptop_row()['quarantined']
                and WORKFLOW_TASK_ID in scene.turns()['local_waiters'],
                what='the call to re-enter the local queue',
            )

            assert _local_waiter(scene) == arrival
            await local_holder.leave()
            await flow.until(
                lambda: scene.legs.spawned(_TEST) or younger.granted.is_set(),
                what='the local turn to go to the older plan',
            )
            assert scene.legs.spawned(_TEST)
            assert not younger.granted.is_set()
            scene.legs.release(_TEST)
            await flow.until_parked_at_merge()
            await wait_responsive(younger.granted.wait(), label='the younger plan to be granted the local turn')

            assert len(scene.laptop.requests('verify_task')) == 1
            sidecar = scene.sidecar(1)
            assert (sidecar['reason'], sidecar['runner']) == ('runner_unavailable', 'local')
            laptop = scene.laptop_row()
            assert (laptop['quarantined'], laptop['quarantine_class']) == (True, 'ru')
            await _end(flow)
            await younger.leave()

    @_REAL
    @pytest.mark.parametrize('kind', ['soft', 'hard'])
    @pytest.mark.parametrize('stage', ['queued_on_both', 'withdrawing_local', 'rederiving_scope'])
    async def test_row31_a_cancel_at_any_stage_of_the_race_unwinds_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, stage: str,
    ) -> None:
        async with open_workflow_scene(
            tmp_path, monkeypatch, term_ignoring_provision=stage == 'withdrawing_local',
            **_RACE_WORKFLOW,
        ) as flow:
            scene = flow.scene
            laptop_held, local_held = await _stage_the_race(flow, stage)

            if kind == 'soft':
                flow.soft_cancel()
            else:
                flow.hard_cancel()
            if stage == 'withdrawing_local':
                await asyncio.sleep(0.1)
                scene.legs.release(PROVISION_LEG)
            report = await flow.report()

            expected = WorkflowOutcome.SOFT_CANCELLED if kind == 'soft' else WorkflowOutcome.CANCELLED
            assert (report.outcome, report.phase) == (expected, WorkflowState.CANCELLED)
            assert scene.unwound(
                laptop_holders=list(laptop_held), local_holders=list(local_held),
            ), scene.turns()
            await flow.until_no_task_outlives_the_run()
            pids = {label: scene.legs.pids(label) for label in scene.legs.spawned_at('local')}
            assert pids and all(len(spawns) == 1 for spawns in pids.values()), pids
            assert not any(_alive(pid) for [pid] in pids.values())
            assert scene.laptop.requests('verify_task') == []
            for turn in (*laptop_held.values(), *local_held.values()):
                await turn.leave()


async def _stage_the_race(flow: WorkflowScene, stage: str) -> tuple[dict[str, Turn], dict[str, Turn]]:
    """Start the run and bring its verify to *stage*; the laptop and local turns the test still holds."""
    scene = flow.scene
    if stage == 'withdrawing_local':
        scene.legs.block(PROVISION_LEG)
        laptop_holder = scene.hold()
        await wait_responsive(laptop_holder.granted.wait(), label='the test to hold the laptop turn')
        flow.start()
        await flow.until(
            lambda: scene.legs.spawned(PROVISION_LEG)
            and WORKFLOW_TASK_ID in scene.turns()['laptop_waiters'],
            what='the local provision to spawn while the call waits for the laptop',
        )
        await laptop_holder.leave()
        await flow.until(
            lambda: scene.pool.holders() == {LAPTOP: f'task:{WORKFLOW_TASK_ID}'},
            what='the call to claim the laptop turn',
        )
        return {}, {}
    laptop_holder, local_holder = await _hold_both(scene)
    flow.start()
    await flow.until_queued_on_both()
    await flow.until(lambda: scene.legs.spawned(_LINT), what='the ungated local lint leg to spawn')
    if stage == 'queued_on_both':
        return {'holder': laptop_holder}, {'local-holder': local_holder}
    flow.merge_base.arm()
    await laptop_holder.leave()
    await flow.until(lambda: flow.merge_base.paused.is_set(), what='the scope re-derive to pause')
    assert scene.pool.holders() == {LAPTOP: f'task:{WORKFLOW_TASK_ID}'}
    return {}, {'local-holder': local_holder}
