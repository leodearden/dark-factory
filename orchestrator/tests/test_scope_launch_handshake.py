"""ScopeLaunchHandshake against real subprocesses: did a launcher exec the payload it was handed."""

from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from orchestrator.scope_launch_handshake import ScopeLaunchHandshake

_NEVER_EXECS = ('/bin/sh', '-c', 'echo refused >&2; exit 1', 'sh')


def _run(handshake: ScopeLaunchHandshake, argv: list[str] | tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, pass_fds=handshake.pass_fds, capture_output=True, text=True, check=False
        )
    finally:
        handshake.release_writer()


def _run_payload(handshake: ScopeLaunchHandshake, cmd: str) -> subprocess.CompletedProcess[str]:
    return _run(handshake, ['/bin/bash', '-c', handshake.payload(cmd)])


def _open_fds() -> list[str]:
    return sorted(os.listdir('/proc/self/fd'))


class TestPayloadThatRan:
    def test_failing_command_is_not_a_launch_refusal(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            proc = _run_payload(handshake, 'exit 3')
            assert proc.returncode == 3
            assert handshake.launcher_exited_before_exec() is False

    def test_syntax_error_in_the_command_still_proves_the_payload_started(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            proc = _run_payload(handshake, 'if then fi')
            assert proc.returncode == 2
            assert 'syntax error' in proc.stderr
            assert handshake.launcher_exited_before_exec() is False

    def test_command_keeps_its_own_stdout_and_stderr(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            proc = _run_payload(handshake, 'echo ok; echo err >&2')
            assert proc.stdout == 'ok\n'
            assert proc.stderr == 'err\n'

    def test_command_does_not_inherit_the_handshake_pipe(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            (writer,) = handshake.pass_fds
            pipe_target = f'pipe:[{os.fstat(writer).st_ino}]'
            proc = _run_payload(handshake, 'ls -l /proc/self/fd')
            assert proc.returncode == 0
            assert pipe_target not in proc.stdout

    @pytest.mark.asyncio
    async def test_asyncio_spawn_carries_the_handshake(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            try:
                proc = await asyncio.create_subprocess_exec(
                    '/bin/bash', '-c', handshake.payload('exit 5'), pass_fds=handshake.pass_fds
                )
            finally:
                handshake.release_writer()
            assert await proc.wait() == 5
            assert handshake.launcher_exited_before_exec() is False


class TestLauncherThatNeverExecs:
    def test_launcher_exit_without_exec_is_a_refusal(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            proc = _run(handshake, [*_NEVER_EXECS, '/bin/bash', '-c', handshake.payload('true')])
            assert proc.returncode == 1
            assert proc.stderr == 'refused\n'
            assert handshake.launcher_exited_before_exec() is True

    def test_verdict_releases_the_parents_writer_itself(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            proc = subprocess.run(
                [*_NEVER_EXECS, '/bin/bash', '-c', handshake.payload('true')],
                pass_fds=handshake.pass_fds,
                capture_output=True,
                check=False,
            )
            assert proc.returncode == 1
            assert handshake.launcher_exited_before_exec() is True

    def test_writer_still_held_by_a_live_process_is_not_proof(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            holder = subprocess.Popen(['/bin/sleep', '30'], pass_fds=handshake.pass_fds)
            try:
                handshake.release_writer()
                assert handshake.launcher_exited_before_exec() is False
            finally:
                holder.kill()
                holder.wait()


class TestLifecycle:
    def test_close_is_idempotent_and_leaks_no_fd(self) -> None:
        before = _open_fds()
        handshake = ScopeLaunchHandshake()
        _run_payload(handshake, 'true')
        handshake.close()
        handshake.close()
        assert _open_fds() == before

    def test_failed_spawn_leaks_no_fd(self, tmp_path) -> None:
        before = _open_fds()
        with ScopeLaunchHandshake() as handshake:
            with pytest.raises(FileNotFoundError):
                subprocess.run(
                    ['/bin/bash', '-c', handshake.payload('true')],
                    pass_fds=handshake.pass_fds,
                    cwd=tmp_path / 'missing',
                    check=False,
                )
        assert _open_fds() == before

    def test_release_writer_is_idempotent(self) -> None:
        with ScopeLaunchHandshake() as handshake:
            handshake.release_writer()
            handshake.release_writer()
            assert handshake.launcher_exited_before_exec() is True
