"""Transport tests for ``RemoteRunner._dispatch``, reached only through the
public ``run_merge_verify`` / ``run_task_verify`` / ``probe_clean``.

PRD: plans/remote-task-leg-verify-prd.md §4.3, §5.2, §6 rows 3/4/19.

The no-verdict results these tests feed the transport are PRODUCED by
executing the public ``verify.run_verification`` against real subprocesses,
so a drift in the producer's summary wording fails here, at the boundary,
rather than passing against a hand-typed marker string.
"""

import asyncio
import shlex
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from _xdist_crash_fixtures import XDIST_SESSION_ABORTED_OUTPUT

from orchestrator import verify
from orchestrator.config import OrchestratorConfig
from orchestrator.verify import (
    SIGNAL_KILL_SUMMARY_MARKER,
    WORKER_DEATH_SUMMARY_MARKER,
    VerifyResult,
)
from orchestrator.verify_runner import (
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
        push_error: OSError | None = None,
        ssh_error: OSError | None = None,
        hold_ssh: bool = False,
    ) -> None:
        self.calls: list[tuple[list[str], object]] = []
        self.ssh_stdout = ssh_stdout
        self.ssh_rc = ssh_rc
        self.ssh_stderr = ssh_stderr
        self.push_rc = push_rc
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
            return self.push_rc, '', 'push rejected' if self.push_rc else ''
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
    return RemoteRunner(
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


class TestRunnerUnavailableKind:
    def test_default_kind_is_transport_and_message_is_kept(self):
        exc = RunnerUnavailable('host down')

        assert exc.kind == 'transport'
        assert str(exc) == 'host down'

    def test_bare_construction_is_transport(self):
        assert RunnerUnavailable().kind == 'transport'

    @pytest.mark.parametrize('kind', ['transport', 'unparseable', 'no_verdict', 'wire_shape'])
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
    async def test_push_rejection_is_transport_and_no_ssh_is_issued(self):
        transport = _FakeTransport(push_rc=1)

        assert await self._kind_of(transport) == 'transport'
        assert transport.ssh_calls() == []

    @pytest.mark.asyncio
    async def test_push_spawn_failure_is_transport(self):
        assert await self._kind_of(_FakeTransport(push_error=OSError('no git'))) == 'transport'

    @pytest.mark.asyncio
    async def test_ssh_exit_255_is_transport(self):
        assert await self._kind_of(_FakeTransport(ssh_rc=255, ssh_stderr='refused')) == 'transport'

    @pytest.mark.asyncio
    async def test_ssh_spawn_failure_is_transport(self):
        assert await self._kind_of(_FakeTransport(ssh_error=OSError('no ssh'))) == 'transport'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('stdout', ['', 'not json!!!'])
    async def test_non_json_stdout_is_unparseable(self, stdout: str):
        assert await self._kind_of(_FakeTransport(ssh_stdout=stdout)) == 'unparseable'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('stdout', ['{"unexpected": 1}', '[1, 2]'])
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
