"""Run a command as a process group that cannot outlive the run, nor hold it open.

The command's stdout and stderr go to unlinked temporary files, not pipes, so
the run ends when the command itself (the group LEADER) exits, whatever its
descendants still hold. When the run ends, however it ends, every descendant
that stayed in the leader's process group is SIGKILLed.

The leader gets a process group of its own, not a new session: verify reaps a
defeated command's subtree by its session as well as its process group
(``orchestrator/src/orchestrator/verify.py::_reap_command_subtree``), and a new
session would take this command out of that reap.

Task 9478: a nested pytest probe that had already printed its summary was held
open to its budget by a pipe a descendant still held.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

_EXIT_POLL_SECS = 0.05


class ProcessGroupOverrun(Exception):
    """The group leader was still running when its budget ran out."""

    def __init__(self, budget_secs: float, leader_pid: int, stdout: str, stderr: str) -> None:
        super().__init__(budget_secs, leader_pid, stdout, stderr)
        self.budget_secs = budget_secs
        self.leader_pid = leader_pid
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        return (
            f'process group {self.leader_pid} was still running after {self.budget_secs}s\n'
            f'stdout:\n{self.stdout}\nstderr:\n{self.stderr}'
        )


def run_process_group(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    budget_secs: float,
) -> subprocess.CompletedProcess[str]:
    """Run *argv* until its leader exits; raise :class:`ProcessGroupOverrun` past *budget_secs*."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        leader = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            process_group=0,
        )
        try:
            exited = _await_exit_unreaped(leader.pid, deadline=time.monotonic() + budget_secs)
        finally:
            # Still unreaped here, so the leader's pid pins the pgid and cannot
            # have been recycled into someone else's group.
            _kill_group(leader.pid)
            leader.wait()
        stdout, stderr = _read_text(out), _read_text(err)
    if not exited:
        raise ProcessGroupOverrun(budget_secs, leader.pid, stdout, stderr)
    return subprocess.CompletedProcess(list(argv), leader.returncode, stdout, stderr)


def _await_exit_unreaped(pid: int, *, deadline: float) -> bool:
    """Poll until child *pid* has exited, leaving it unreaped; False at *deadline*."""
    while True:
        if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_EXIT_POLL_SECS)


def _kill_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _read_text(stream: IO[bytes]) -> str:
    stream.seek(0)
    return stream.read().decode('utf-8', errors='replace')
