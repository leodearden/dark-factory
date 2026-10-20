"""`orchestrator verify-task` and the remote-verify worktree hygiene it shares.

A remote verify (``verify-merge`` or ``verify-task``) records its worktree
beside its pgid file; ``cancel-verify`` SIGKILLs the tree, which skips the
subcommand's own ``finally``, so it reaps the recorded worktree instead.
PRD: plans/remote-task-leg-verify-prd.md §4.2, §5.3.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT
from _xdist_crash_fixtures import XDIST_WORKER_CRASH_OUTPUT
from click.testing import CliRunner, Result
from test_cli import (
    _communicate_or_empty,
    _patched_verify_merge,
    _setup_verify_repo,
    _wait_for_pgid_file_or_report_crash,
    _wait_pgid_gone,
)

import orchestrator.cli as cli_module
from orchestrator.cli import main
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.task_verify_run import prepare_task_verify_worktree
from orchestrator.task_verify_spec import (
    TaskVerifySpec,
    apply_task_verify_spec,
    task_spec_to_json,
)
from orchestrator.verify import VerifyResult, run_scoped_verification
from orchestrator.verify_cancel import (
    PGID_DIR_NAME,
    WatchdogTrigger,
    acquire_merge_verify_flock,
    kill_process_tree,
    lane_lock_path,
    pgid_file,
    read_worktree_path,
    release_merge_verify_flock,
    worktree_path_file,
    write_pgid_file,
    write_worktree_path,
)
from orchestrator.verify_runner import is_flock_contention_failure, result_from_json
from orchestrator.verify_spec_shared import VerifyCommand

NONEXISTENT_PID = 2 ** 22  # no process can hold this pid: it is at pid_max
FAKE_PGID = 55555


def _invoke_cancel_verify(tmp_path: Path, request_id: str) -> Result:
    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')
    return CliRunner().invoke(
        main, ['cancel-verify', '--request-id', request_id, '--config', str(cfg_file)],
    )


def _registered_worktrees(repo: Path) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), 'worktree', 'list', '--porcelain'],
        check=True, capture_output=True, text=True,
    ).stdout


class TestCancelVerifyReapsRecordedWorktree:
    @pytest.fixture
    def repo_setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        repo, head_sha = _setup_verify_repo(tmp_path)
        config = OrchestratorConfig(project_root=repo)
        monkeypatch.setattr(cli_module, 'load_config', lambda _: config)
        return repo, head_sha, GitOps(config.git, repo)

    def _record_dead_run(self, git_ops: GitOps, request_id: str, worktree: Path) -> Path:
        pgf = pgid_file(git_ops.worktree_base, request_id)
        write_pgid_file(pgf, NONEXISTENT_PID)
        write_worktree_path(pgf, worktree)
        return pgf

    def test_reaps_recorded_ephemeral_merge_worktree(self, tmp_path: Path, repo_setup) -> None:
        repo, head_sha, git_ops = repo_setup
        wt = asyncio.run(git_ops.create_ephemeral_verify_worktree(head_sha))
        pgf = self._record_dead_run(git_ops, 'req-reap', wt)

        r = _invoke_cancel_verify(tmp_path, 'req-reap')

        assert r.exit_code == 0, r.output
        assert not wt.exists()
        assert str(wt) not in _registered_worktrees(repo)
        assert not pgf.exists()
        assert not worktree_path_file(pgf).exists()

    def test_refuses_to_reap_a_task_shaped_path(self, tmp_path: Path, repo_setup) -> None:
        _, _, git_ops = repo_setup
        task_dir = git_ops.worktree_base / '1234'
        task_dir.mkdir(parents=True)
        sentinel = task_dir / 'sentinel'
        sentinel.write_text('keep me')
        pgf = self._record_dead_run(git_ops, 'req-task', task_dir)

        r = _invoke_cancel_verify(tmp_path, 'req-task')

        assert r.exit_code == 0, r.output
        assert sentinel.read_text() == 'keep me'
        assert str(task_dir) in r.stderr
        assert read_worktree_path(pgf) is None

    def test_refuses_a_merge_named_path_outside_worktree_base(
        self, tmp_path: Path, repo_setup,
    ) -> None:
        _, _, git_ops = repo_setup
        stray = tmp_path / 'elsewhere' / '_merge-stray'
        stray.mkdir(parents=True)
        sentinel = stray / 'sentinel'
        sentinel.write_text('keep me')
        self._record_dead_run(git_ops, 'req-stray', stray)

        r = _invoke_cancel_verify(tmp_path, 'req-stray')

        assert r.exit_code == 0, r.output
        assert sentinel.read_text() == 'keep me'
        assert str(stray) in r.stderr


class TestVerifyMergeRecordsItsWorktree:
    def test_records_acquired_worktree_mid_run_and_clears_it_on_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invoke = _patched_verify_merge(tmp_path, monkeypatch)
        pgf = pgid_file(tmp_path / '.worktrees', 'req-merge')
        recorded_mid_run: list[Path | None] = []

        async def fake_run_merge_verify(wt, cfg, spec, merge_sha=None):
            recorded_mid_run.append(read_worktree_path(pgf))
            return object()

        monkeypatch.setattr(
            'orchestrator.verify_runner.run_merge_verify_on_worktree', fake_run_merge_verify,
        )

        r = invoke('--request-id', 'req-merge')

        assert r.exit_code == 0, r.output
        assert recorded_mid_run == [tmp_path / '_merge-verify']
        assert not worktree_path_file(pgf).exists()
        assert not pgf.exists()

    def test_without_request_id_records_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invoke = _patched_verify_merge(tmp_path, monkeypatch)

        r = invoke()

        assert r.exit_code == 0, r.output
        assert not (tmp_path / '.worktrees' / PGID_DIR_NAME).exists()


# ---------------------------------------------------------------------------
# `orchestrator verify-task`
# ---------------------------------------------------------------------------


def _task_spec(
    sha: str, test_command: str, *, preprovision: str = '', tip_sha: str | None = None,
) -> TaskVerifySpec:
    return TaskVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command=test_command),),
        global_verify_command=None,
        task_files=('mod/test_x.py',),
        force_workspace=False,
        verify_env={},
        scope_cargo=False,
        verify_cold_preprovision_command=preprovision,
        attempt_id=1,
        task_id='t1',
        tip_sha=sha if tip_sha is None else tip_sha,
    )


def _use_host_config(
    monkeypatch: pytest.MonkeyPatch, repo: Path, *, preprovision: str = '',
    git: GitConfig | None = None,
) -> OrchestratorConfig:
    """Hand verify-task this host config; the preprovision knob is ALWAYS explicit.

    The autouse ORCH_CONFIG_PATH pin would otherwise hand every config the
    operational `uv sync … && npm ci …`, and a verify-task worktree is cold.
    """
    config = OrchestratorConfig(project_root=repo, verify_cold_preprovision_command=preprovision)
    if git is not None:
        config = config.model_copy(update={'git': git})
    monkeypatch.setattr(cli_module, 'load_config', lambda _: config)
    return config


def _invoke_verify_task(tmp_path: Path, sha: str, spec_json: str, *extra: str) -> Result:
    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')
    return CliRunner().invoke(main, [
        'verify-task', '--sha', sha, '--spec', spec_json, '--config', str(cfg_file), *extra,
    ])


def _fake_dispatch_seams(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Keep setsid off the test process and the watchdog off its stdin; return watchdog pgids."""
    watchdog_pgids: list[int] = []

    def fake_start_stdin_watchdog(pgid: int, *args, **kwargs) -> MagicMock:
        watchdog_pgids.append(pgid)
        return MagicMock()

    monkeypatch.setattr(cli_module, 'start_own_process_group', lambda: FAKE_PGID)
    monkeypatch.setattr(cli_module, 'start_stdin_watchdog', fake_start_stdin_watchdog)
    return watchdog_pgids


def _copy_record_command(git_ops: GitOps, request_id: str, seen: Path) -> str:
    """A test command proving the worktree record existed mid-run, by copying it to *seen*."""
    record = worktree_path_file(pgid_file(git_ops.worktree_base, request_id))
    return f'cp {shlex.quote(str(record))} {shlex.quote(str(seen))}'


def _merge_entries(worktree_base: Path) -> list[str]:
    if not worktree_base.exists():
        return []
    return sorted(p.name for p in worktree_base.iterdir() if p.name.startswith('_merge-'))


async def _in_process_task_verify(config: OrchestratorConfig, spec: TaskVerifySpec) -> VerifyResult:
    git_ops = GitOps(config.git, config.project_root)
    wt = await git_ops.create_ephemeral_verify_worktree(spec.tip_sha)
    try:
        prepare_task_verify_worktree(wt)
        applied, modules = apply_task_verify_spec(config, spec)
        result = await run_scoped_verification(
            wt, applied, modules, list(spec.task_files),
            attempt_id=spec.attempt_id, task_id=spec.task_id, archive_root=None,
            force_workspace=False, role='task',
        )
        return dataclasses.replace(result, worktree_log_paths=[], archive_log_paths=[])
    finally:
        await git_ops.cleanup_merge_worktree(wt)


@pytest.fixture
def verify_repo(tmp_path: Path) -> tuple[Path, str]:
    return _setup_verify_repo(tmp_path)


class TestVerifyTaskVerdict:
    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    @pytest.mark.parametrize('test_command,expect_pass', [('true', True), ('false', False)])
    def test_cli_result_equals_the_in_process_task_verify(
        self, tmp_path, monkeypatch, verify_repo, test_command, expect_pass,
    ) -> None:
        repo, head_sha = verify_repo
        config = _use_host_config(monkeypatch, repo)
        spec = _task_spec(head_sha, test_command)
        baseline = asyncio.run(_in_process_task_verify(config, spec))

        r = _invoke_verify_task(tmp_path, head_sha, task_spec_to_json(spec))

        assert r.exit_code == 0, r.output
        assert result_from_json(r.stdout) == baseline
        assert baseline.passed is expect_pass

    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    @pytest.mark.parametrize(
        'spec_carries_command,expected_marker',
        [(True, 'spec.marker'), (False, 'host.marker')],
        ids=['spec_command_replaces_host', 'empty_spec_command_keeps_host'],
    )
    def test_cold_worktree_runs_the_effective_preprovision_command(
        self, tmp_path, monkeypatch, verify_repo, spec_carries_command, expected_marker,
    ) -> None:
        repo, head_sha = verify_repo
        _use_host_config(
            monkeypatch, repo, preprovision=f'touch {shlex.quote(str(tmp_path / "host.marker"))}',
        )
        spec_command = (
            f'touch {shlex.quote(str(tmp_path / "spec.marker"))}' if spec_carries_command else ''
        )
        spec = _task_spec(head_sha, 'true', preprovision=spec_command)

        r = _invoke_verify_task(tmp_path, head_sha, task_spec_to_json(spec))

        assert r.exit_code == 0, r.output
        assert sorted(p.name for p in tmp_path.glob('*.marker')) == [expected_marker]


class TestVerifyTaskRequestId:
    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    def test_records_its_ephemeral_worktree_mid_run_and_removes_both_on_exit(
        self, tmp_path, monkeypatch, verify_repo,
    ) -> None:
        repo, head_sha = verify_repo
        config = _use_host_config(monkeypatch, repo)
        watchdog_pgids = _fake_dispatch_seams(monkeypatch)
        git_ops = GitOps(config.git, repo)
        seen = tmp_path / 'seen.record'
        spec = _task_spec(head_sha, _copy_record_command(git_ops, 'req-task', seen))

        r = _invoke_verify_task(
            tmp_path, head_sha, task_spec_to_json(spec), '--request-id', 'req-task',
        )

        assert r.exit_code == 0, r.output
        assert result_from_json(r.stdout).passed
        assert watchdog_pgids == [FAKE_PGID]
        wt = Path(seen.read_text().strip())
        assert wt.name.startswith('_merge-')
        assert wt.name != '_merge-verify'
        assert not wt.exists()
        assert str(wt) not in _registered_worktrees(repo)
        pgf = pgid_file(git_ops.worktree_base, 'req-task')
        assert not pgf.exists()
        assert not worktree_path_file(pgf).exists()

    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    def test_a_fired_watchdog_owns_the_exit(self, tmp_path, monkeypatch, verify_repo) -> None:
        repo, head_sha = verify_repo
        _use_host_config(monkeypatch, repo)
        log: list[str] = []

        class _FakeWatchdogThread:
            def join(self, timeout=None) -> None:
                log.append('watchdog_joined')

        def fake_start_stdin_watchdog(pgid, *, fire, **kwargs) -> _FakeWatchdogThread:
            fire(WatchdogTrigger.EOF)
            return _FakeWatchdogThread()

        monkeypatch.setattr(cli_module, 'start_own_process_group', lambda: FAKE_PGID)
        monkeypatch.setattr(
            cli_module, 'fire_watchdog_kill', lambda pgid, **kw: log.append('escalation_started'),
        )
        monkeypatch.setattr(cli_module, 'start_stdin_watchdog', fake_start_stdin_watchdog)

        r = _invoke_verify_task(
            tmp_path, head_sha, task_spec_to_json(_task_spec(head_sha, 'true')),
            '--request-id', 'req-watchdog',
        )

        assert log == ['escalation_started', 'watchdog_joined']
        assert r.exit_code == 1, r.output
        with pytest.raises(ValueError):
            result_from_json(r.stdout)


class TestVerifyTaskIsAlwaysEphemeral:
    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    def test_runs_while_the_merge_lane_lock_is_held(
        self, tmp_path, monkeypatch, verify_repo,
    ) -> None:
        repo, head_sha = verify_repo
        config = _use_host_config(
            monkeypatch, repo, git=GitConfig(persistent_merge_worktree=True),
        )
        _fake_dispatch_seams(monkeypatch)
        git_ops = GitOps(config.git, repo)
        seen = tmp_path / 'seen.record'
        spec = _task_spec(head_sha, _copy_record_command(git_ops, 'req-lane', seen))
        lane_fd = acquire_merge_verify_flock(
            lane_lock_path(git_ops.persistent_merge_worktree_path), 1.0,
        )
        assert lane_fd is not None
        try:
            r = _invoke_verify_task(
                tmp_path, head_sha, task_spec_to_json(spec), '--request-id', 'req-lane',
            )
        finally:
            release_merge_verify_flock(lane_fd)

        assert r.exit_code == 0, r.output
        result = result_from_json(r.stdout)
        assert result.passed
        assert not is_flock_contention_failure(result)
        wt_name = Path(seen.read_text().strip()).name
        assert wt_name.startswith('_merge-')
        assert wt_name != '_merge-verify'


def _malformed_json(sha: str) -> tuple[str, str]:
    return sha, '{not json'


def _unknown_spec_key(sha: str) -> tuple[str, str]:
    spec = json.loads(task_spec_to_json(_task_spec(sha, 'true')))
    spec['surprise'] = 1
    return sha, json.dumps(spec)


def _tip_differs_from_sha(sha: str) -> tuple[str, str]:
    return sha, task_spec_to_json(_task_spec(sha, 'true', tip_sha='f' * 40))


def _absent_sha(sha: str) -> tuple[str, str]:
    absent = '0' * 40
    return absent, task_spec_to_json(_task_spec(absent, 'true'))


class TestVerifyTaskRefusals:
    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    @pytest.mark.parametrize(
        'bad_request',
        [_malformed_json, _unknown_spec_key, _tip_differs_from_sha, _absent_sha],
    )
    def test_bad_request_exits_nonzero_without_a_result(
        self, tmp_path, monkeypatch, verify_repo,
        bad_request: Callable[[str], tuple[str, str]],
    ) -> None:
        repo, head_sha = verify_repo
        config = _use_host_config(monkeypatch, repo)
        sha, spec_json = bad_request(head_sha)

        r = _invoke_verify_task(tmp_path, sha, spec_json)

        assert r.exit_code != 0
        with pytest.raises(ValueError):
            result_from_json(r.stdout)
        assert r.stderr.strip()
        assert r.exception is None or isinstance(r.exception, SystemExit)
        assert _merge_entries(GitOps(config.git, repo).worktree_base) == []

    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    def test_xdist_worker_crash_exits_1_without_a_result(
        self, tmp_path, monkeypatch, verify_repo,
    ) -> None:
        repo, head_sha = verify_repo
        config = _use_host_config(monkeypatch, repo)
        crash = f'printf %s {shlex.quote(XDIST_WORKER_CRASH_OUTPUT)} && exit 1'

        r = _invoke_verify_task(
            tmp_path, head_sha, task_spec_to_json(_task_spec(head_sha, crash)),
        )

        assert r.exit_code == 1, r.output
        with pytest.raises(ValueError):
            result_from_json(r.stdout)
        assert _merge_entries(GitOps(config.git, repo).worktree_base) == []


def _verify_task_child_env() -> dict[str, str]:
    """The env of test_cli.py::test_verify_merge_cancel_end_to_end, plus a hermetic cold verify."""
    env = dict(os.environ)
    worktree_src = str(Path(__file__).parent.parent / 'src')
    existing_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{worktree_src}:{existing_pp}' if existing_pp else worktree_src
    env.pop('ORCH_PROJECT_ROOT', None)
    env.pop('ORCH_VERIFY_COLD_PREPROVISION_COMMAND', None)
    env['ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS'] = '600'
    return env


def _wait_for_recorded_cold_worktree(
    pgf: Path, child: subprocess.Popen, *, timeout: float = 30.0,
) -> Path:
    """Poll until the child has recorded its worktree and given it the cold ``.task/``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and child.poll() is None:
        wt = read_worktree_path(pgf)
        if wt is not None and (wt / '.task').is_dir():
            return wt
        time.sleep(0.1)
    stdout, stderr = _communicate_or_empty(child) if child.poll() is not None else (b'', b'')
    pytest.fail(
        f'verify-task never recorded a worktree holding .task/ (poll={child.poll()!r})\n'
        f'STDOUT: {stdout.decode()[:2000]!r}\nSTDERR: {stderr.decode()[:2000]!r}'
    )


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_cancel_verify_kills_a_real_verify_task_and_reaps_its_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verify_repo,
) -> None:
    repo, head_sha = verify_repo
    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text(f"project_root: {repo}\nverify_cold_preprovision_command: ''\n")
    config = OrchestratorConfig(project_root=repo)
    pgf = pgid_file(GitOps(config.git, repo).worktree_base, 'req-e2e')
    child = subprocess.Popen(
        [sys.executable, '-c', 'from orchestrator.cli import main; main()',
         'verify-task',
         '--sha', head_sha,
         '--spec', task_spec_to_json(_task_spec(head_sha, 'sleep 300')),
         '--config', str(cfg_file),
         '--request-id', 'req-e2e'],
        env=_verify_task_child_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pgid: int | None = None
    try:
        _wait_for_pgid_file_or_report_crash(pgf, child, timeout=30)
        pgid = int(pgf.read_text().strip())
        wt = _wait_for_recorded_cold_worktree(pgf, child)
        monkeypatch.setattr(cli_module, 'load_config', lambda _: config)

        r = _invoke_cancel_verify(tmp_path, 'req-e2e')

        assert r.exit_code == 0, r.output
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail('verify-task did not exit within 20s of cancel-verify')
        _wait_pgid_gone(pgid, timeout=20)
        assert not (wt / '.task').exists()
        assert not wt.exists()
        assert str(wt) not in _registered_worktrees(repo)
        assert not pgf.exists()
        assert not worktree_path_file(pgf).exists()
    finally:
        if child.stdin is not None:
            child.stdin.close()
        if child.poll() is None:
            kill_process_tree(child.pid, backstop_pgid=pgid)
            child.wait(timeout=5)
