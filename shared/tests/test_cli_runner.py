"""The public CLI runner contract: ``shared.cli_invoke.run_cli_subprocess``.

Every spawning test drives a REAL child written into ``tmp_path``. The
group-kill tests use a fake that Popen-s a ``sleep 30`` grandchild into its
own process group, so "the group is gone" cannot pass with a leader-only
signal. Each test passes its env explicitly, so no ambient ``DF_AGENT_CPU_*``
prefix applies.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    SubprocessResult,
    TranscriptWatch,
    invoke_claude_agent,
    run_cli_subprocess,
)
from shared.proc_group import pid_running, process_group_terminated
from shared.testing_fake_claude import install_fake_claude_cli

_STATE_WAIT_SECS = 10.0
_GROUP_GONE_SECS = 10.0
_POLL_STEP_SECS = 0.05


def _bare_env(**extra: str) -> dict[str, str]:
    return {'PATH': os.environ['PATH'], **extra}


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(f'#!{sys.executable} -I\n{body}')
    path.chmod(0o755)
    return path


def _echo_fake(tmp_path: Path) -> Path:
    return _write_executable(tmp_path / 'echo-fake', (
        'import sys\n'
        'data = sys.stdin.buffer.read()\n'
        "sys.stdout.write(f'stdout:{len(data)}')\n"
        "sys.stderr.write('stderr-text')\n"
        'sys.exit(3)\n'
    ))


def _grandchild_fake(tmp_path: Path, state_file: Path) -> Path:
    return _write_executable(tmp_path / 'grandchild-fake', (
        'import json, os, subprocess, time\n'
        "child = subprocess.Popen(['sleep', '30'])\n"
        f'state = {str(state_file)!r}\n'
        "tmp = state + '.tmp'\n"
        "with open(tmp, 'w') as f:\n"
        "    json.dump({'pid': os.getpid(), 'grandchild': child.pid}, f)\n"
        'os.replace(tmp, state)\n'
        'time.sleep(30)\n'
    ))


async def _wait_for_state(state_file: Path) -> dict[str, int]:
    deadline = asyncio.get_running_loop().time() + _STATE_WAIT_SECS
    while not state_file.exists():
        assert asyncio.get_running_loop().time() < deadline, (
            f'the fake never wrote {state_file} within {_STATE_WAIT_SECS}s'
        )
        await asyncio.sleep(_POLL_STEP_SECS)
    return json.loads(state_file.read_text())


async def _group_terminated_within(pgid: int) -> bool:
    deadline = asyncio.get_running_loop().time() + _GROUP_GONE_SECS
    while asyncio.get_running_loop().time() < deadline:
        if process_group_terminated(pgid):
            return True
        await asyncio.sleep(_POLL_STEP_SECS)
    return process_group_terminated(pgid)


def _kill_leftover_group(state_file: Path) -> None:
    if not state_file.exists():
        return
    pgid = json.loads(state_file.read_text())['pid']
    if not process_group_terminated(pgid):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


class TestWrapArgv:
    async def test_wrap_is_applied_once_before_spawn(self, tmp_path: Path) -> None:
        fake = install_fake_claude_cli(tmp_path, ['out'])

        result = await run_cli_subprocess(
            ['inner', 'x'], tmp_path, _bare_env(),
            backend='claude', model='m', timeout_seconds=30.0,
            wrap_argv=lambda argv: [str(fake.executable), 'WRAPPED', *argv],
        )

        assert fake.recorded_argv() == [['WRAPPED', 'inner', 'x']]
        assert result.stdout == 'out'

    async def test_a_raising_wrap_propagates_and_spawns_nothing(self, tmp_path: Path) -> None:
        fake = install_fake_claude_cli(tmp_path, ['out'])

        def refuse(argv: list[str]) -> list[str]:
            raise RuntimeError('wrap refused')

        with pytest.raises(RuntimeError, match='wrap refused'):
            await run_cli_subprocess(
                [str(fake.executable)], tmp_path, _bare_env(),
                backend='claude', model='m', stdin_data=b'payload', wrap_argv=refuse,
            )

        assert fake.recorded_argv() == []


class TestFlatArm:
    @pytest.mark.timeout(30)
    async def test_normal_exit_result_carries_only_the_process_fields(
        self, tmp_path: Path,
    ) -> None:
        env = _bare_env(DF_AGENT_CPU_NICE='5')

        result = await run_cli_subprocess(
            [str(_echo_fake(tmp_path))], tmp_path, env,
            backend='codex', model='m', timeout_seconds=30.0, stdin_data=b'hello',
        )

        assert result == SubprocessResult(
            stdout='stdout:5', stderr='stderr-text', returncode=3,
            duration_ms=result.duration_ms,
        )
        assert env['DF_AGENT_CPU_NICE'] == '5'

    @pytest.mark.timeout(30)
    async def test_timeout_kills_the_whole_group_with_the_flat_result(
        self, tmp_path: Path,
    ) -> None:
        state_file = tmp_path / 'state.json'
        try:
            result = await run_cli_subprocess(
                [str(_grandchild_fake(tmp_path, state_file))], tmp_path, _bare_env(),
                backend='pi', model='m', timeout_seconds=2.0,
            )
            state = json.loads(state_file.read_text())

            assert result == SubprocessResult(
                stdout='',
                stderr='Process killed after 2.0s timeout (SIGTERM+SIGKILL)',
                returncode=1,
                duration_ms=result.duration_ms,
                timed_out=True,
            )
            assert await _group_terminated_within(state['pid'])
            assert not pid_running(state['grandchild'])
        finally:
            _kill_leftover_group(state_file)


class TestCancellation:
    @pytest.mark.timeout(30)
    async def test_cancel_reaps_the_whole_group(self, tmp_path: Path) -> None:
        state_file = tmp_path / 'state.json'
        try:
            task = asyncio.ensure_future(run_cli_subprocess(
                [str(_grandchild_fake(tmp_path, state_file))], tmp_path, _bare_env(),
                backend='codex', model='m', timeout_seconds=30.0,
            ))
            state = await _wait_for_state(state_file)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert await _group_terminated_within(state['pid'])
            assert not pid_running(state['grandchild'])
        finally:
            _kill_leftover_group(state_file)

    async def test_cancel_during_timeout_kill_still_reaps_group(self, tmp_path: Path) -> None:
        """A cancel landing inside the flat timeout's group kill retries the kill.

        Fully mocked: the fake terminate_process_group hangs on its first call,
        so the test synchronises on "the task is suspended inside the kill"
        without guessing at a sleep duration.
        """
        proc = MagicMock()
        proc.pid = 12345
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.wait = AsyncMock()
        proc.returncode = None

        async def fake_exec(*args: Any, **kwargs: Any) -> MagicMock:
            return proc

        calls: list[tuple[tuple, dict]] = []
        entered = asyncio.Event()
        hang_forever = asyncio.Event()

        async def fake_terminate_process_group(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))
            if len(calls) == 1:
                entered.set()
                await hang_forever.wait()

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch(
                'shared.cli_invoke.terminate_process_group',
                side_effect=fake_terminate_process_group,
            ),
        ):
            task = asyncio.ensure_future(run_cli_subprocess(
                ['fake'], tmp_path, {}, backend='codex', model='m', timeout_seconds=0.01,
            ))
            await asyncio.wait_for(entered.wait(), timeout=5)

            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert len(calls) == 2, calls
        second_args, second_kwargs = calls[1]
        assert second_args[1] == 12345, calls[1]
        assert second_kwargs.get('grace_secs') == 5.0, calls[1]


def test_subprocess_result_is_frozen() -> None:
    result = SubprocessResult(stdout='', stderr='', returncode=0, duration_ms=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.returncode = 1  # type: ignore[misc]


@pytest.fixture
def pinned_claude(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    exec_file = tmp_path / 'claude'
    exec_file.write_text('#!/bin/sh\nexit 0\n')
    exec_file.chmod(0o755)
    monkeypatch.setenv('CLAUDE_BINARY', str(exec_file))
    return str(exec_file)


def _canned_claude_result(**overrides: Any) -> SubprocessResult:
    stdout = json.dumps({
        'result': 'canned-output',
        'is_error': False,
        'subtype': 'success',
        'num_turns': 1,
        'session_id': 'canned-session',
    })
    fields: dict[str, Any] = {
        'stdout': stdout, 'stderr': '', 'returncode': 0, 'duration_ms': 100,
    }
    fields.update(overrides)
    return SubprocessResult(**fields)


class _RecordingRunner:
    def __init__(self, result: SubprocessResult) -> None:
        self.result = result
        self.calls: list[tuple[list[str], Path, dict[str, str], dict[str, Any]]] = []

    async def __call__(
        self, cmd: list[str], cwd: Path, env: dict[str, str], **kwargs: Any,
    ) -> SubprocessResult:
        self.calls.append((list(cmd), cwd, dict(env), kwargs))
        return self.result


class TestInvokeClaudeAgentRunnerSeam:
    async def _invoke(
        self, tmp_path: Path, runner: _RecordingRunner, **overrides: Any,
    ):
        kwargs: dict[str, Any] = {
            'prompt': 'p',
            'system_prompt': 's',
            'cwd': tmp_path,
            'session_id': 's1',
            'config_dir': tmp_path,
            'startup_grace_secs': 7.0,
            'working_idle_secs': 60.0,
            'absolute_cap_secs': 600.0,
            'timeout_seconds': 30.0,
            'runner': runner,
        }
        kwargs.update(overrides)
        return await invoke_claude_agent(**kwargs)

    async def test_the_runner_receives_the_unwrapped_argv_and_the_spawn_contract(
        self, tmp_path: Path, pinned_claude: str,
    ) -> None:
        runner = _RecordingRunner(_canned_claude_result())

        def wrap(argv: list[str]) -> list[str]:
            return ['SANDBOX', *argv]

        result = await self._invoke(tmp_path, runner, sandbox_wrap=wrap)

        assert len(runner.calls) == 1
        cmd, _cwd, _env, kw = runner.calls[0]
        assert cmd[0] == pinned_claude
        assert kw['wrap_argv'] is wrap
        assert kw['backend'] == 'claude'
        assert kw['stdin_data'] == b'p'
        assert kw['timeout_seconds'] == 30.0
        assert kw['transcript'] == TranscriptWatch(
            session_id='s1', config_dir=tmp_path, startup_grace_secs=7.0,
            working_idle_secs=60.0, absolute_cap_secs=600.0,
        )
        assert result.output == 'canned-output'
        assert result.session_id == 'canned-session'

    async def test_resume_session_id_is_the_watched_transcript(
        self, tmp_path: Path, pinned_claude: str,
    ) -> None:
        runner = _RecordingRunner(_canned_claude_result())

        await self._invoke(tmp_path, runner, resume_session_id='r1')

        _cmd, _cwd, _env, kw = runner.calls[0]
        assert kw['transcript'].session_id == 'r1'
        assert kw['wrap_argv'] is None

    async def test_a_timed_out_runner_result_reaches_the_agent_result(
        self, tmp_path: Path, pinned_claude: str,
    ) -> None:
        runner = _RecordingRunner(_canned_claude_result(timed_out=True))

        result = await self._invoke(tmp_path, runner)

        assert result.timed_out is True
