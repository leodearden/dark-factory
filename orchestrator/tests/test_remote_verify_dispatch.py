"""Transport tests for ``RemoteRunner._dispatch``, reached only through the
public ``run_merge_verify`` / ``run_task_verify`` / ``probe_clean``.

PRD: plans/remote-task-leg-verify-prd.md §4.3, §5.2, §6 rows 3/4/19.

The no-verdict results these tests feed the transport are PRODUCED by
executing the public ``verify.run_verification`` against real subprocesses,
so a drift in the producer's summary wording fails here, at the boundary,
rather than passing against a hand-typed marker string.
"""

import asyncio
import dataclasses
import json
import shlex
import shutil
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from _gate_fixtures import gate_argv, gated_remote_runner
from _xdist_crash_fixtures import XDIST_IN_FLIGHT_NODEID, XDIST_SESSION_ABORTED_OUTPUT

from orchestrator import verify
from orchestrator.config import OrchestratorConfig
from orchestrator.task_verify_spec import TaskVerifySpec, task_spec_from_json, task_spec_to_json
from orchestrator.verify import (
    SIGNAL_KILL_SUMMARY_MARKER,
    WORKER_DEATH_SUMMARY_MARKER,
    VerifyResult,
)
from orchestrator.verify_runner import (
    REMOTE_VERIFY_PROCESS_PATTERN,
    ConcurrentDispatchError,
    MergeVerifySpec,
    RemoteRunner,
    RunnerUnavailable,
    RunnerUnavailableKind,
    UnscopedTypecheckSpec,
    result_from_json,
    result_to_json,
)
from orchestrator.verify_spec_shared import VerifyCommand


async def _executed_result(worktree: Path, test_command: str) -> VerifyResult:
    config = OrchestratorConfig(
        project_root=worktree,
        test_command=test_command,
        lint_command='true',
        type_check_command='true',
        verify_timeout_retries=0,
    )
    return await verify.run_verification(worktree, config, max_retries=0)


async def _signal_killed_result(worktree: Path) -> VerifyResult:
    return await _executed_result(worktree, 'kill -9 $$')


async def _xdist_worker_death_result(worktree: Path) -> VerifyResult:
    output = worktree / 'xdist.out'
    output.write_text(XDIST_SESSION_ABORTED_OUTPUT)
    return await _executed_result(worktree, f'cat {shlex.quote(str(output))}; exit 1')


async def _genuine_red_result(worktree: Path) -> VerifyResult:
    return await _executed_result(worktree, 'echo plain failure; exit 1')


class TestIsNoVerdictSummary:
    @pytest.mark.asyncio
    async def test_executed_signal_kill_is_no_verdict(self, tmp_path: Path):
        result = await _signal_killed_result(tmp_path)

        assert not result.passed
        assert SIGNAL_KILL_SUMMARY_MARKER in result.summary
        assert verify.is_no_verdict_summary(result.summary) is True

    @pytest.mark.asyncio
    async def test_executed_xdist_worker_death_is_no_verdict(self, tmp_path: Path):
        result = await _xdist_worker_death_result(tmp_path)

        assert not result.passed
        assert WORKER_DEATH_SUMMARY_MARKER in result.summary
        assert verify.is_no_verdict_summary(result.summary) is True

    @pytest.mark.asyncio
    async def test_executed_genuine_red_is_a_verdict(self, tmp_path: Path):
        result = await _genuine_red_result(tmp_path)

        assert not result.passed
        assert result.summary == 'Failures: tests failed'
        assert verify.is_no_verdict_summary(result.summary) is False

    def test_green_summary_is_a_verdict(self):
        assert verify.is_no_verdict_summary('All checks passed') is False

    def test_empty_summary_is_a_verdict(self):
        assert verify.is_no_verdict_summary('') is False


class _FakeTransport:
    """Records every (argv, cwd) the runner issues and answers it.

    Git pushes answer *push_rc* (or raise *push_error*); ref deletes always
    succeed; the ssh dispatch answers (*ssh_rc*, *ssh_stdout*, *ssh_stderr*)
    or raises *ssh_error*.  When *hold_ssh* is set, every ssh first signals
    ``ssh_entered`` and then waits for ``ssh_release``.
    """

    def __init__(
        self,
        *,
        ssh_stdout: str = '',
        ssh_rc: int = 0,
        ssh_stderr: str = '',
        push_rc: int = 0,
        push_stderr: str = 'push rejected',
        push_error: OSError | None = None,
        ssh_error: OSError | None = None,
        hold_ssh: bool = False,
    ) -> None:
        self.calls: list[tuple[list[str], object]] = []
        self.ssh_stdout = ssh_stdout
        self.ssh_rc = ssh_rc
        self.ssh_stderr = ssh_stderr
        self.push_rc = push_rc
        self.push_stderr = push_stderr
        self.push_error = push_error
        self.ssh_error = ssh_error
        self.hold_ssh = hold_ssh
        self.ssh_entered = asyncio.Event()
        self.ssh_release = asyncio.Event()

    async def run(self, argv: list[str], *, cwd: object = None) -> tuple[int, str, str]:
        self.calls.append((list(argv), cwd))
        if argv[0] == 'git':
            if '--delete' in argv:
                return 0, '', ''
            if self.push_error is not None:
                raise self.push_error
            return self.push_rc, '', self.push_stderr if self.push_rc else ''
        if self.ssh_error is not None:
            raise self.ssh_error
        if self.hold_ssh:
            self.ssh_entered.set()
            await self.ssh_release.wait()
        return self.ssh_rc, self.ssh_stdout, self.ssh_stderr

    def pushed_refspecs(self) -> list[str]:
        return [
            argv[3] for argv, _ in self.calls
            if argv[:2] == ['git', 'push'] and '--delete' not in argv
        ]

    def deleted_refs(self) -> list[str]:
        return [argv[-1] for argv, _ in self.calls if '--delete' in argv]

    def ssh_calls(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls if argv[0] == 'ssh']


def _ids(*request_ids: str) -> Callable[[], str]:
    remaining: Iterator[str] = iter(request_ids)
    return lambda: next(remaining)


def _remote_runner(
    transport: _FakeTransport,
    *,
    id_factory: Callable[[], str] | None = None,
    main_branch: str | None = None,
    config_path: str | None = None,
) -> RemoteRunner:
    return gated_remote_runner(
        name='laptop',
        ssh_host='laptop.local',
        git_remote='origin',
        cwd='/repo',
        config_path=config_path,
        main_branch=main_branch,
        run=transport.run,
        id_factory=id_factory or (lambda: 'fixed-id'),
    )


def _merge_spec() -> MergeVerifySpec:
    return MergeVerifySpec(
        verify_commands=(VerifyCommand(prefix='m', test_command='pytest'),),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=('m/a.py',),
        verify_env={},
        cold_timeout_secs=60.0,
    )


def _green() -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='',
        summary='All checks passed', category='passed',
    )


_GREEN_WITH_NULL_SUMMARY = json.dumps({**json.loads(result_to_json(_green())), 'summary': None})


class TestRunnerUnavailableKind:
    def test_default_kind_is_transport_and_message_is_kept(self):
        exc = RunnerUnavailable('host down')

        assert exc.kind == 'transport'
        assert str(exc) == 'host down'

    def test_bare_construction_is_transport(self):
        assert RunnerUnavailable().kind == 'transport'

    @pytest.mark.parametrize('kind', ['transport', 'gate', 'unparseable', 'no_verdict', 'wire_shape'])
    def test_every_vocabulary_kind_is_accepted(self, kind: RunnerUnavailableKind):
        assert RunnerUnavailable('x', kind=kind).kind == kind

    def test_unknown_kind_is_rejected_naming_it_and_the_vocabulary(self):
        excinfo = pytest.raises(ValueError, RunnerUnavailable, 'x', kind='bogus')

        message = str(excinfo.value)
        assert 'bogus' in message
        assert 'no_verdict' in message
        assert 'wire_shape' in message


class TestDispatchFailureKinds:
    async def _kind_of(self, transport: _FakeTransport) -> str:
        with pytest.raises(RunnerUnavailable) as excinfo:
            await _remote_runner(transport).run_merge_verify('abc123', _merge_spec())
        return excinfo.value.kind

    @pytest.mark.asyncio
    async def test_push_rejection_is_gate_and_no_verify_ssh_is_issued(self):
        """A host that refuses the verify ref is reached: 'gate' (task 6565, Q1 ruling)."""
        transport = _FakeTransport(push_rc=1)

        assert await self._kind_of(transport) == 'gate'
        assert transport.ssh_calls() == []

    @pytest.mark.asyncio
    async def test_push_connect_failure_is_transport(self):
        transport = _FakeTransport(
            push_rc=128,
            push_stderr='ssh: connect to host laptop.local port 22: Connection timed out\n'
                        'fatal: Could not read from remote repository.',
        )

        assert await self._kind_of(transport) == 'transport'

    @pytest.mark.asyncio
    async def test_push_spawn_failure_is_transport(self):
        assert await self._kind_of(_FakeTransport(push_error=OSError('no git'))) == 'transport'

    @pytest.mark.asyncio
    async def test_ssh_exit_255_is_transport(self):
        assert await self._kind_of(_FakeTransport(ssh_rc=255, ssh_stderr='refused')) == 'transport'

    @pytest.mark.asyncio
    async def test_ssh_exit_other_than_255_is_gate(self):
        """The host was reached and the verify command failed there (design D5)."""
        assert await self._kind_of(_FakeTransport(ssh_rc=1, ssh_stderr='Error: x')) == 'gate'

    @pytest.mark.asyncio
    async def test_ssh_spawn_failure_is_transport(self):
        assert await self._kind_of(_FakeTransport(ssh_error=OSError('no ssh'))) == 'transport'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('stdout', ['', 'not json!!!'])
    async def test_non_json_stdout_is_unparseable(self, stdout: str):
        assert await self._kind_of(_FakeTransport(ssh_stdout=stdout)) == 'unparseable'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('stdout', ['{"unexpected": 1}', '[1, 2]', _GREEN_WITH_NULL_SUMMARY])
    async def test_json_of_the_wrong_shape_is_wire_shape(self, stdout: str):
        assert await self._kind_of(_FakeTransport(ssh_stdout=stdout)) == 'wire_shape'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('produce', [_signal_killed_result, _xdist_worker_death_result])
    async def test_executed_no_verdict_result_is_unavailable(
        self, tmp_path: Path, produce: Callable[[Path], Awaitable[VerifyResult]],
    ):
        executed = await produce(tmp_path)
        transport = _FakeTransport(ssh_stdout=result_to_json(executed))
        runner = _remote_runner(transport)

        with pytest.raises(RunnerUnavailable) as excinfo:
            await runner.run_merge_verify('abc123', _merge_spec())

        assert excinfo.value.kind == 'no_verdict'
        assert executed.summary in str(excinfo.value)
        assert transport.deleted_refs() == ['refs/merge-verify/fixed-id']
        assert runner.dispatch_in_flight is False

    @pytest.mark.asyncio
    async def test_executed_genuine_red_is_returned_not_raised(self, tmp_path: Path):
        wire = result_to_json(await _genuine_red_result(tmp_path))

        result = await _remote_runner(_FakeTransport(ssh_stdout=wire)).run_merge_verify(
            'abc123', _merge_spec(),
        )

        assert result == result_from_json(wire)
        assert result.summary == 'Failures: tests failed'


class TestExclusiveDispatch:
    @pytest.mark.asyncio
    async def test_second_dispatch_on_a_busy_runner_is_refused(self):
        transport = _FakeTransport(ssh_stdout=result_to_json(_green()), hold_ssh=True)
        runner = _remote_runner(transport, id_factory=_ids('id-a', 'id-b', 'id-c'))
        first = asyncio.create_task(runner.run_merge_verify('abc123', _merge_spec()))
        await transport.ssh_entered.wait()

        with pytest.raises(ConcurrentDispatchError) as excinfo:
            await runner.run_merge_verify('def456', _merge_spec())

        message = str(excinfo.value)
        assert 'id-a' in message
        assert 'id-b' in message
        assert 'laptop' in message
        assert 'def456:refs/merge-verify/id-b' not in transport.pushed_refspecs()
        assert runner.dispatch_in_flight is True

        transport.ssh_release.set()
        assert await first == _green()
        assert 'refs/merge-verify/id-a' in transport.deleted_refs()
        assert runner.dispatch_in_flight is False

        assert await runner.run_merge_verify('fed789', _merge_spec()) == _green()
        assert 'fed789:refs/merge-verify/id-c' in transport.pushed_refspecs()

    def test_a_broken_invariant_is_never_read_as_an_outage(self):
        assert not issubclass(ConcurrentDispatchError, RunnerUnavailable)


_TIP = 'c' * 40


def _task_spec(**overrides) -> TaskVerifySpec:
    fields = {
        'verify_commands': (VerifyCommand(prefix='orchestrator', test_command='pytest'),),
        'global_verify_command': None,
        'task_files': ('orchestrator/src/orchestrator/x.py',),
        'force_workspace': False,
        'verify_env': {'A': '1'},
        'scope_cargo': True,
        'verify_cold_preprovision_command': 'uv sync',
        'attempt_id': 4,
        'task_id': '6667',
        'tip_sha': _TIP,
    }
    return TaskVerifySpec(**{**fields, **overrides})


def _red() -> VerifyResult:
    return VerifyResult(
        passed=False,
        test_output='FAILED TEST',
        lint_output='',
        type_output='TYPE ERRORS',
        summary='Failures: tests failed, type errors',
        category='test_failure',
        worktree_log_paths=['/laptop/wt/.task/verify/attempt-4.test.log'],
    )


def _one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    assert len(matches) == 1, f'{pattern}: {matches}'
    return matches[0]


class TestRunTaskVerifyTransport:
    @pytest.mark.asyncio
    async def test_happy_path_ships_the_task_spec_and_deletes_the_ref(self):
        spec = _task_spec()
        transport = _FakeTransport(ssh_stdout=result_to_json(_green()))
        runner = _remote_runner(transport, main_branch='main', config_path='/etc/orch.yaml')

        result = await runner.run_task_verify(spec, task_id='6667', archive_root=None)

        assert result == _green()
        assert transport.calls[0] == (
            ['git', 'push', 'origin', f'{_TIP}:refs/merge-verify/fixed-id'], '/repo',
        )
        [ssh_argv] = transport.ssh_calls()
        assert ssh_argv[-2] == 'laptop.local'
        parsed = gate_argv(ssh_argv[-1])
        assert parsed == [
            'verify-task', '--sha', _TIP,
            '--spec', task_spec_to_json(spec),
            '--config', '/etc/orch.yaml',
            '--request-id', 'fixed-id',
        ]
        assert task_spec_from_json(parsed[4]) == spec
        assert transport.deleted_refs() == ['refs/merge-verify/fixed-id']

    @pytest.mark.asyncio
    async def test_no_main_mirror_push_for_a_task_verify(self):
        transport = _FakeTransport(ssh_stdout=result_to_json(_green()))

        await _remote_runner(transport, main_branch='main').run_task_verify(
            _task_spec(), task_id='6667', archive_root=None,
        )

        assert transport.pushed_refspecs() == [f'{_TIP}:refs/merge-verify/fixed-id']
        assert not any('refs/heads/main' in ' '.join(argv) for argv, _ in transport.calls)

    @pytest.mark.asyncio
    async def test_dispatch_is_in_flight_only_while_the_ssh_runs(self):
        transport = _FakeTransport(ssh_stdout=result_to_json(_green()), hold_ssh=True)
        runner = _remote_runner(transport)
        verifying = asyncio.create_task(
            runner.run_task_verify(_task_spec(), task_id='6667', archive_root=None),
        )
        await transport.ssh_entered.wait()

        assert runner.dispatch_in_flight is True

        transport.ssh_release.set()
        await verifying
        assert runner.dispatch_in_flight is False

    async def _kind_of(self, transport: _FakeTransport, archive_root: Path | None = None) -> str:
        with pytest.raises(RunnerUnavailable) as excinfo:
            await _remote_runner(transport).run_task_verify(
                _task_spec(), task_id='6667', archive_root=archive_root,
            )
        return excinfo.value.kind

    @pytest.mark.asyncio
    async def test_ssh_exit_255_is_transport(self):
        assert await self._kind_of(_FakeTransport(ssh_rc=255, ssh_stderr='refused')) == 'transport'

    @pytest.mark.asyncio
    async def test_push_rejection_is_gate_and_no_verify_ssh_is_issued(self):
        """A host that refuses the verify ref is reached: 'gate' (task 6565, Q1 ruling)."""
        transport = _FakeTransport(push_rc=1)

        assert await self._kind_of(transport) == 'gate'
        assert transport.ssh_calls() == []

    @pytest.mark.asyncio
    async def test_push_connect_failure_is_transport(self):
        transport = _FakeTransport(
            push_rc=128,
            push_stderr='ssh: connect to host laptop.local port 22: Connection timed out\n'
                        'fatal: Could not read from remote repository.',
        )

        assert await self._kind_of(transport) == 'transport'

    @pytest.mark.asyncio
    async def test_non_json_stdout_is_unparseable(self):
        assert await self._kind_of(_FakeTransport(ssh_stdout='not json!!!')) == 'unparseable'

    @pytest.mark.asyncio
    async def test_json_of_the_wrong_shape_is_wire_shape(self):
        assert await self._kind_of(_FakeTransport(ssh_stdout='{"unexpected": 1}')) == 'wire_shape'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('produce', [_signal_killed_result, _xdist_worker_death_result])
    async def test_executed_no_verdict_result_is_unavailable_and_archived(
        self, tmp_path: Path, produce: Callable[[Path], Awaitable[VerifyResult]],
    ):
        worktree = tmp_path / 'worktree'
        worktree.mkdir()
        archive_root = tmp_path / 'archive'
        transport = _FakeTransport(ssh_stdout=result_to_json(await produce(worktree)))

        assert await self._kind_of(transport, archive_root) == 'no_verdict'

        task_dir = archive_root / '6667'
        record = json.loads(_one(task_dir, 'attempt-4.remote-laptop.no-verdict-*.json').read_text())
        assert record['sha'] == _TIP
        assert record['attempt_id'] == 4
        assert record['role'] == 'task'
        assert list(task_dir.glob('*pass-summary*')) == []

    @pytest.mark.asyncio
    async def test_task_id_disagreeing_with_the_spec_is_refused_before_any_push(self):
        transport = _FakeTransport(ssh_stdout=result_to_json(_green()))

        with pytest.raises(ValueError) as excinfo:
            await _remote_runner(transport).run_task_verify(
                _task_spec(task_id='6667'), task_id='other', archive_root=None,
            )

        assert 'other' in str(excinfo.value)
        assert '6667' in str(excinfo.value)
        assert transport.calls == []


class TestRunTaskVerifyArchive:
    @pytest.mark.asyncio
    async def test_red_archives_streams_summary_and_stderr_under_the_spec_attempt(
        self, tmp_path: Path,
    ):
        transport = _FakeTransport(ssh_stdout=result_to_json(_red()), ssh_stderr='LAPTOP LOG')

        result = await _remote_runner(transport).run_task_verify(
            _task_spec(), task_id='6667', archive_root=tmp_path,
        )

        task_dir = tmp_path / '6667'
        expected = {
            _one(task_dir, 'attempt-4.remote-laptop.test-*.log'),
            _one(task_dir, 'attempt-4.remote-laptop.type-*.log'),
            _one(task_dir, 'attempt-4.remote-laptop.summary-*.json'),
            _one(task_dir, 'attempt-4.remote-laptop-*.stderr.log'),
        }
        assert set(task_dir.iterdir()) == expected
        summary = json.loads(_one(task_dir, 'attempt-4.remote-laptop.summary-*.json').read_text())
        assert summary['role'] == 'task'
        assert summary['category'] == 'test_failure'
        assert _one(task_dir, 'attempt-4.remote-laptop-*.stderr.log').read_text() == 'LAPTOP LOG'
        assert set(result.archive_log_paths) == {str(path) for path in expected}
        assert len(result.archive_log_paths) == len(expected)
        assert result.worktree_log_paths == []
        assert result == dataclasses.replace(
            _red(), worktree_log_paths=[], archive_log_paths=result.archive_log_paths,
        )

    @pytest.mark.asyncio
    async def test_green_archives_one_pass_summary(self, tmp_path: Path):
        spec = _task_spec()
        transport = _FakeTransport(ssh_stdout=result_to_json(_green()))

        result = await _remote_runner(transport).run_task_verify(
            spec, task_id='6667', archive_root=tmp_path,
        )

        task_dir = tmp_path / '6667'
        pass_summary = _one(task_dir, 'attempt-4.remote-laptop.pass-summary-*.json')
        assert list(task_dir.iterdir()) == [pass_summary]
        payload = json.loads(pass_summary.read_text())
        duration_ms = payload.pop('duration_ms')
        assert isinstance(duration_ms, int)
        assert duration_ms >= 0
        assert payload == {
            'tip_sha': _TIP,
            'task_id': '6667',
            'attempt_id': 4,
            'runner': 'laptop',
            'passed': True,
            'category': 'passed',
            'scope': {
                'task_files': ['orchestrator/src/orchestrator/x.py'],
                'force_workspace': False,
                'scope_cargo': True,
                'modules': ['orchestrator'],
            },
        }
        assert result.archive_log_paths == [str(pass_summary)]

    @pytest.mark.asyncio
    async def test_no_archive_root_writes_nothing_and_drops_laptop_paths(self, tmp_path: Path):
        laptop_red = dataclasses.replace(_red(), archive_log_paths=['/laptop/archive/x.log'])
        transport = _FakeTransport(ssh_stdout=result_to_json(laptop_red), ssh_stderr='LAPTOP LOG')

        result = await _remote_runner(transport).run_task_verify(
            _task_spec(), task_id='6667', archive_root=None,
        )

        assert list(tmp_path.rglob('*')) == []
        assert result.archive_log_paths == []
        assert result.worktree_log_paths == []

    @pytest.mark.asyncio
    async def test_merge_red_keeps_attempt_one_and_role_merge(self, tmp_path: Path):
        transport = _FakeTransport(ssh_stdout=result_to_json(_red()), ssh_stderr='LAPTOP LOG')

        await _remote_runner(transport).run_merge_verify(
            'abc123', _merge_spec(), task_id='6667', archive_root=tmp_path,
        )

        task_dir = tmp_path / '6667'
        _one(task_dir, 'attempt-1.remote-laptop-*.stderr.log')
        summary = json.loads(_one(task_dir, 'attempt-1.remote-laptop.summary-*.json').read_text())
        assert summary['role'] == 'merge'
        assert list(task_dir.glob('attempt-4.*')) == []


class TestNoVerdictArchive:
    @pytest.mark.asyncio
    async def test_merge_no_verdict_archives_the_worker_death_line(self, tmp_path: Path):
        worktree = tmp_path / 'worktree'
        worktree.mkdir()
        executed = await _xdist_worker_death_result(worktree)
        transport = _FakeTransport(ssh_stdout=result_to_json(executed), ssh_stderr='LAPTOP LOG')

        with pytest.raises(RunnerUnavailable) as excinfo:
            await _remote_runner(transport).run_merge_verify(
                'abc123', _merge_spec(), task_id='6667', archive_root=tmp_path / 'archive',
            )

        assert excinfo.value.kind == 'no_verdict'
        task_dir = tmp_path / 'archive' / '6667'
        test_log = _one(task_dir, 'attempt-1.remote-laptop.test-*.log').read_text()
        assert f"crashed while running '{XDIST_IN_FLIGHT_NODEID}'" in test_log
        assert _one(task_dir, 'attempt-1.remote-laptop-*.stderr.log').read_text() == 'LAPTOP LOG'
        summary = json.loads(_one(task_dir, 'attempt-1.remote-laptop.summary-*.json').read_text())
        assert summary['role'] == 'merge'
        record = json.loads(_one(task_dir, 'attempt-1.remote-laptop.no-verdict-*.json').read_text())
        assert {key: record[key] for key in (
            'sha', 'runner', 'ssh_host', 'task_id', 'attempt_id', 'role', 'summary',
        )} == {
            'sha': 'abc123',
            'runner': 'laptop',
            'ssh_host': 'laptop.local',
            'task_id': '6667',
            'attempt_id': 1,
            'role': 'merge',
            'summary': executed.summary,
        }
        assert list(task_dir.glob('*pass-summary*')) == []
        assert transport.deleted_refs() == ['refs/merge-verify/fixed-id']
        message = str(excinfo.value)
        for path in task_dir.iterdir():
            assert str(path) in message

    @pytest.mark.asyncio
    async def test_unwritable_archive_root_still_raises_no_verdict(self, tmp_path: Path):
        worktree = tmp_path / 'worktree'
        worktree.mkdir()
        archive_root = tmp_path / 'archive'
        archive_root.write_text('a file, so no archive directory can be made under it')
        executed = await _xdist_worker_death_result(worktree)
        transport = _FakeTransport(ssh_stdout=result_to_json(executed), ssh_stderr='LAPTOP LOG')
        runner = _remote_runner(transport)

        with pytest.raises(RunnerUnavailable) as excinfo:
            await runner.run_merge_verify(
                'abc123', _merge_spec(), task_id='6667', archive_root=archive_root,
            )

        assert excinfo.value.kind == 'no_verdict'
        assert 'archived at' not in str(excinfo.value)
        assert runner.dispatch_in_flight is False


class TestExclusiveDispatchAcrossPaths:
    @pytest.mark.asyncio
    async def test_task_dispatch_is_refused_while_a_merge_is_in_flight(self):
        transport = _FakeTransport(ssh_stdout=result_to_json(_green()), hold_ssh=True)
        runner = _remote_runner(transport, id_factory=_ids('id-a', 'id-b'))
        merging = asyncio.create_task(runner.run_merge_verify('abc123', _merge_spec()))
        await transport.ssh_entered.wait()

        with pytest.raises(ConcurrentDispatchError):
            await runner.run_task_verify(_task_spec(), task_id='6667', archive_root=None)

        assert f'{_TIP}:refs/merge-verify/id-b' not in transport.pushed_refspecs()
        transport.ssh_release.set()
        assert await merging == _green()

    @pytest.mark.asyncio
    async def test_merge_dispatch_is_refused_while_a_task_is_in_flight(self):
        transport = _FakeTransport(ssh_stdout=result_to_json(_green()), hold_ssh=True)
        runner = _remote_runner(transport, id_factory=_ids('id-a', 'id-b'))
        verifying = asyncio.create_task(
            runner.run_task_verify(_task_spec(), task_id='6667', archive_root=None),
        )
        await transport.ssh_entered.wait()

        with pytest.raises(ConcurrentDispatchError):
            await runner.run_merge_verify('abc123', _merge_spec())

        assert 'abc123:refs/merge-verify/id-b' not in transport.pushed_refspecs()
        transport.ssh_release.set()
        assert await verifying == _green()


def _sleeper(*argv: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', *argv])


class TestRemoteVerifyProcessPattern:
    def test_pgrep_finds_both_subcommands_and_nothing_else(self):
        if shutil.which('pgrep') is None:
            pytest.skip('pgrep is not installed')
        request_id = f'--request-id={uuid.uuid4().hex}'
        task = _sleeper('orchestrator', 'verify-task', request_id)
        merge = _sleeper('orchestrator', 'verify-merge', request_id)
        other = _sleeper('orchestrator', 'verify-other', request_id)
        try:
            listed = subprocess.run(
                ['pgrep', '-f', REMOTE_VERIFY_PROCESS_PATTERN],
                capture_output=True, text=True, check=False,
            ).stdout.split()

            assert str(task.pid) in listed
            assert str(merge.pid) in listed
            assert str(other.pid) not in listed
        finally:
            for child in (task, merge, other):
                child.kill()
                child.wait()

    @pytest.mark.asyncio
    async def test_probe_issues_one_ssh_running_the_pattern(self):
        transport = _FakeTransport(ssh_rc=1)

        await _remote_runner(transport).probe_clean()

        [ssh_argv] = transport.ssh_calls()
        assert (ssh_argv[0], ssh_argv[-2]) == ('ssh', 'laptop.local')
        assert shlex.split(ssh_argv[-1]) == ['pgrep', '-f', REMOTE_VERIFY_PROCESS_PATTERN]
        assert len(transport.calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('ssh_rc', 'clean'), [(1, True), (0, False), (2, False)])
    async def test_probe_is_clean_only_when_pgrep_finds_nothing(self, ssh_rc: int, clean: bool):
        assert await _remote_runner(_FakeTransport(ssh_rc=ssh_rc)).probe_clean() is clean

    @pytest.mark.asyncio
    async def test_probe_transport_failure_is_not_clean(self):
        transport = _FakeTransport(ssh_error=OSError('no ssh'))

        assert await _remote_runner(transport).probe_clean() is False
