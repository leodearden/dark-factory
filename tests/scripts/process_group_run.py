"""Run a command as a process group that cannot outlive the run, nor hold it open.

The command's stdout and stderr go to unlinked temporary files, not pipes, so
the run ends when the command itself (the group LEADER) exits, whatever its
descendants still hold. When the run ends, however it ends, every descendant
that stayed in the leader's process group is SIGKILLed.

The leader gets a process group of its own, not a new session: verify finds
the cgroup scopes a defeated command's subtree moved into through the members
of that command's session
(``orchestrator/src/orchestrator/relocated_scopes.py::relocated_scopes``), and a
new session would hide this command from that discovery.

Task 9478: a nested pytest probe that had printed its summary overran its
budget. With files instead of pipes no descendant can be the cause, and an
overrun lists the live group to show what was.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

from shared.proc_group import ProcessGroupMember, process_group_members

_EXIT_POLL_SECS = 0.05
_PROC = Path('/proc')
UNREADABLE = '?'


@dataclasses.dataclass(frozen=True)
class GroupMember:
    """A live process-group member, with its command line and each thread's wait channel.

    ``thread_wchans`` lists the kernel wait channel of the main thread first,
    then of every other thread by tid. An unreadable command line, thread list
    or wait channel reads as :data:`UNREADABLE`.
    """

    process: ProcessGroupMember
    cmdline: str
    thread_wchans: tuple[str, ...]


class ProcessGroupOverrun(Exception):
    """The group leader was still running when its budget ran out."""

    def __init__(
        self,
        budget_secs: float,
        leader_pid: int,
        members: tuple[GroupMember, ...],
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(budget_secs, leader_pid, members, stdout, stderr)
        self.budget_secs = budget_secs
        self.leader_pid = leader_pid
        self.members = members
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        return '\n'.join([
            f'process group {self.leader_pid} was still running after '
            f'{self.budget_secs}s; live members at the deadline:',
            *(self._member_line(member) for member in self.members),
            f'stdout:\n{self.stdout}',
            f'stderr:\n{self.stderr}',
        ])

    def _member_line(self, member: GroupMember) -> str:
        process = member.process
        role = ' (leader)' if process.pid == self.leader_pid else ''
        return (
            f'  pid {process.pid}{role} ppid {process.ppid} state {process.state} '
            f'wchan {",".join(member.thread_wchans)}: {member.cmdline}'
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
            members = () if exited else _group_members(leader.pid)
        finally:
            # Still unreaped here, so the leader's pid pins the pgid and cannot
            # have been recycled into someone else's group.
            _kill_group(leader.pid)
            leader.wait()
        stdout, stderr = _read_text(out), _read_text(err)
    if not exited:
        raise ProcessGroupOverrun(budget_secs, leader.pid, members, stdout, stderr)
    return subprocess.CompletedProcess(list(argv), leader.returncode, stdout, stderr)


def _await_exit_unreaped(pid: int, *, deadline: float) -> bool:
    """Poll until child *pid* has exited, leaving it unreaped; False at *deadline*."""
    while True:
        if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_EXIT_POLL_SECS)


def _group_members(pgid: int) -> tuple[GroupMember, ...]:
    """Every live process in group *pgid*, the leader first.

    The walk is :func:`shared.proc_group.process_group_members`, which skips a
    pid that vanishes or whose stat is malformed, so the snapshot never masks
    the overrun it reports.
    """
    live = (process for process in process_group_members(pgid) if not process.terminated)
    ordered = sorted(live, key=lambda process: (process.pid != pgid, process.pid))
    return tuple(
        GroupMember(process, _cmdline(process.pid), _thread_wchans(process.pid))
        for process in ordered
    )


def _cmdline(pid: int) -> str:
    try:
        raw = (_PROC / str(pid) / 'cmdline').read_bytes()
    except OSError:
        return UNREADABLE
    return raw.rstrip(b'\0').replace(b'\0', b' ').decode('utf-8', errors='replace')


def _thread_wchans(pid: int) -> tuple[str, ...]:
    tasks = _PROC / str(pid) / 'task'
    try:
        tids = sorted((int(tid) for tid in os.listdir(tasks)), key=lambda tid: (tid != pid, tid))
    except OSError:
        return (UNREADABLE,)
    return tuple(_wchan(tasks / str(tid)) for tid in tids)


def _wchan(task: Path) -> str:
    try:
        return (task / 'wchan').read_bytes().decode('ascii', errors='replace').strip() or UNREADABLE
    except OSError:
        return UNREADABLE


def _kill_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _read_text(stream: IO[bytes]) -> str:
    stream.seek(0)
    return stream.read().decode('utf-8', errors='replace')
