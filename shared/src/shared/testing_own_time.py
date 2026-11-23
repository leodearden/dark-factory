"""Waits on a child process bounded by its own time, with the wall clock only a backstop.

A child's own time is the wall-clock time since a wait began, less the run
delay of the child's main thread: field 2 of Linux's ``/proc/PID/schedstat``, in
nanoseconds, the time the kernel credits it in queue stints. Time queued for a
CPU, here, is time the child spends runnable but off a CPU. The kernel credits
it as run delay only once its stint ends, so while the child is runnable own
time also leaves out any time it may have spent queued for a CPU that is not
credited yet. A host too crowded to run the child therefore fails a wait only at
its wall-clock limit, by default many budgets long, while time the child spends
running, or waiting on something of its own, counts, time blocked in the kernel
(state D) included. The kernel can also credit, as run delay, time in which the
child was not runnable; once it does, own time leaves that time out too, so such
a credit can only make a wait more lenient.

A wait reads the wall clock and its child, and sleeps, through a
:class:`WaitHost`, by default :data:`LINUX_HOST`, so it is Linux-only:
:data:`LINUX_HOST` reads ``/proc/PID/schedstat`` and ``/proc/PID/stat``.
:func:`run_on_own_time` runs a command to its exit under such a wait, as
``subprocess.run`` would with its output captured.

This is the sanctioned bound on a real child process under the
timing-dependence lint (fused-memory/scripts/_timing_dependence.py): use
:func:`run_on_own_time` in place of ``subprocess.run(..., timeout=<literal>)``,
and :func:`wait_on_own_time` for a ``Popen``. It is a testing module, so it is
never re-exported from ``shared/__init__.py`` and no src module imports it,
which keeps it out of the shipped package root. Pure stdlib, and registered as
such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Protocol

__all__ = [
    "LINUX_HOST",
    "ChildSnapshot",
    "OwnTimeExceeded",
    "Reading",
    "WaitHost",
    "WallLimitExceeded",
    "exited_unreaped",
    "run_on_own_time",
    "state_letter",
    "wait_on_own_time",
]

_DEFAULT_WALL_LIMIT_IN_BUDGETS = 20


def state_letter(stat: str) -> str:
    """The state in the text of a ``/proc/PID/stat``: its first token after the
    last ``)``, as the process name before that may hold spaces and parentheses."""
    return stat.rpartition(")")[2].split()[0]


@dataclass(frozen=True)
class ChildSnapshot:
    """A child process's time and state at one reading.

    *on_cpu* is how long the child's main thread has run on a CPU since it
    started, and *run_delay* its run delay since then, both in seconds; *state*
    is its proc(5) state letter. *run_delay* lags by any queue stint still in
    progress, which :class:`Reading` bounds as its *uncredited* while the child
    is runnable.
    """

    on_cpu: float
    run_delay: float
    state: str

    @property
    def runnable(self) -> bool:
        """Whether the child is on a CPU or queued for one."""
        return self.state == "R"


class WaitHost(Protocol):
    """What a wait reads and spends time through: the host, or a scripted host."""

    def wall(self) -> float: ...

    def child(self, pid: int) -> ChildSnapshot:
        """The time and state of the child *pid*. The state is read before the
        times, so the run delay in the times counts all the time a child seen
        not runnable had spent queued for a CPU by that read."""
        ...

    def sleep(self, seconds: float) -> None: ...


class _LinuxHost:
    """The host's wall clock and sleep, and the child's state and then its times
    from its ``/proc`` files, in the order :meth:`WaitHost.child` requires."""

    def wall(self) -> float:
        return time.monotonic()

    def child(self, pid: int) -> ChildSnapshot:
        state = state_letter(Path(f"/proc/{pid}/stat").read_text())
        on_cpu_ns, run_delay_ns, _ = Path(f"/proc/{pid}/schedstat").read_text().split()
        return ChildSnapshot(
            on_cpu=int(on_cpu_ns) / 1e9,
            run_delay=int(run_delay_ns) / 1e9,
            state=state,
        )

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


LINUX_HOST = _LinuxHost()


@dataclass(frozen=True)
class Reading:
    """How a wait's child had spent the wait, at one poll.

    *elapsed* is how long the wait had lasted, *on_cpu* how much of that the
    child's main thread spent on a CPU, *run_delay* the run delay the kernel
    credited it in that time, and *state* the child's proc(5) state letter.
    *uncredited* is how much of the wait, up to this reading, the child may have
    spent queued for a CPU that its run delay does not count yet: while the
    child is runnable, the time since the later of the reading before its latest
    progress and its latest reading in which it was not runnable; while it is
    not, 0.
    """

    elapsed: float
    on_cpu: float
    run_delay: float
    state: str
    uncredited: float

    @property
    def own(self) -> float:
        """The child's own time: the wait's, less *run_delay* and *uncredited*. It
        never goes below 0: run delay credited during the wait can cover time
        before it began, and *uncredited* can overlap run delay already
        credited."""
        return max(0.0, self.elapsed - self.run_delay - self.uncredited)

    def __str__(self) -> str:
        return (
            f"after {self.elapsed:.2f} s: {self.on_cpu:.2f} s on a CPU, "
            f"{self.run_delay:.2f} s of run delay and up to {self.uncredited:.2f} s more "
            f"queued for a CPU but not yet credited, leaving {self.own:.2f} s as its own, "
            f"the child in state {self.state}"
        )


@dataclass(frozen=True)
class _OwnTimeMeter:
    """The readings of one wait, each against the *baseline* the wait took of its
    child; *reading* is the latest."""

    baseline: ChildSnapshot
    reading: Reading

    @classmethod
    def starting_from(cls, baseline: ChildSnapshot) -> _OwnTimeMeter:
        """The meter when the wait takes its *baseline*, before any time has passed."""
        return cls(
            baseline,
            Reading(elapsed=0.0, on_cpu=0.0, run_delay=0.0, state=baseline.state, uncredited=0.0),
        )

    def after_reading(self, child: ChildSnapshot, *, elapsed: float) -> _OwnTimeMeter:
        """This meter once the wait has read *child*, *elapsed* seconds in.

        A child queued for a CPU stays runnable until it gets one, which ends
        its queue stint, so a child that is not runnable has no time queued for
        a CPU that its run delay leaves out. A runnable child whose times have
        moved since the latest reading is in no queue stint that began before
        it, and one whose times have not may be in the stint it was already in.
        """
        latest = self.reading
        on_cpu = child.on_cpu - self.baseline.on_cpu
        run_delay = child.run_delay - self.baseline.run_delay
        since_latest = elapsed - latest.elapsed
        progressed = (on_cpu, run_delay) != (latest.on_cpu, latest.run_delay)
        if not child.runnable:
            uncredited = 0.0
        elif progressed:
            uncredited = since_latest
        else:
            uncredited = latest.uncredited + since_latest
        return replace(
            self,
            reading=Reading(
                elapsed=elapsed,
                on_cpu=on_cpu,
                run_delay=run_delay,
                state=child.state,
                uncredited=uncredited,
            ),
        )


class OwnTimeExceeded(TimeoutError):
    """A wait that outlasted its budget of the child's own time.

    A :class:`TimeoutError` whose message leads with the reading it keeps as
    ``reading``; a caller prefixes its own context.
    """

    def __init__(self, *, budget: float, reading: Reading) -> None:
        super().__init__(f"{reading}, past its {budget:g} s budget")
        self.budget = budget
        self.reading = reading


class WallLimitExceeded(TimeoutError):
    """A wait that outlasted its wall-clock limit while the child's own time
    stayed within its budget, as when the host keeps the child queued for a CPU.

    A :class:`TimeoutError` whose message leads with the reading it keeps as
    ``reading``; a caller prefixes its own context.
    """

    def __init__(self, *, wall_limit: float, reading: Reading) -> None:
        super().__init__(f"{reading}, past its {wall_limit:g} s wall-clock limit")
        self.wall_limit = wall_limit
        self.reading = reading


def exited_unreaped(pid: int) -> bool:
    """Whether this process's child *pid* has exited, observed without reaping it.

    The exit stays for the child's Popen to reap, and until then the child's
    ``/proc/PID/schedstat`` and ``/proc/PID/stat`` stay readable.
    """
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def wait_on_own_time(
    pid: int,
    done: Callable[[], bool],
    *,
    budget: float,
    wall_limit: float | None = None,
    poll_interval: float = 0.02,
    host: WaitHost = LINUX_HOST,
) -> None:
    """Poll until *done* holds, failing once the child *pid* has spent more
    than *budget* seconds of its own time on the wait, or once the wait has
    lasted *wall_limit* seconds, by default _DEFAULT_WALL_LIMIT_IN_BUDGETS
    budgets.

    *pid* must be this process's child, and stay unreaped for the whole wait.
    *done* is checked after each poll's reading and before that reading is held
    against either limit, so a condition that comes true while this process is
    itself starved of a CPU never fails the wait. The wait reads and spends all
    its time through *host*. It reads the child before it starts its clock, and
    the clock before the child at every poll, so a stall of this process can
    only understate the child's own time. It also leaves the *uncredited* bound
    of the child's :class:`Reading` out of that time, so a queue stint the
    kernel has not yet credited can only understate it too.

    Raises:
        OwnTimeExceeded: *done* was still false once the child's own time
            exceeded *budget*.
        WallLimitExceeded: *done* was still false once the wait had lasted
            *wall_limit* seconds, with the child's own time within *budget*.
    """
    if wall_limit is None:
        wall_limit = _DEFAULT_WALL_LIMIT_IN_BUDGETS * budget
    meter = _OwnTimeMeter.starting_from(host.child(pid))
    started = host.wall()
    while True:
        elapsed = host.wall() - started
        meter = meter.after_reading(host.child(pid), elapsed=elapsed)
        reading = meter.reading
        if done():
            return
        if reading.own > budget:
            raise OwnTimeExceeded(budget=budget, reading=reading)
        if reading.elapsed > wall_limit:
            raise WallLimitExceeded(wall_limit=wall_limit, reading=reading)
        host.sleep(poll_interval)


def run_on_own_time(
    args: Sequence[str],
    *,
    budget: float,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the command *args*, in *cwd* and with *env* as ``subprocess.run``
    would, until it exits, waiting on it as :func:`wait_on_own_time` does with
    *budget*.

    Returns what ``subprocess.run`` returns with ``capture_output=True,
    text=True, errors="backslashreplace"``: the command, its exit status, and
    everything it wrote to its stdout and to its stderr, with any bytes that do
    not decode, such as a character a kill cut short, escaped. That output goes
    to unnamed temporary files, not pipes, because a child blocked on a full
    pipe would be spending its own time. On a failed wait, it kills and reaps
    the child before reading those files: a child still running shares each
    file's offset with the read, and could write over the output being read.
    The wait's exception then propagates, noted with the command and everything
    the child wrote.

    Raises:
        OwnTimeExceeded: as :func:`wait_on_own_time` raises it.
        WallLimitExceeded: as :func:`wait_on_own_time` raises it.
    """
    with (
        tempfile.TemporaryFile("w+", errors="backslashreplace") as stdout,
        tempfile.TemporaryFile("w+", errors="backslashreplace") as stderr,
        subprocess.Popen(args, cwd=cwd, env=env, stdout=stdout, stderr=stderr) as child,
    ):
        try:
            wait_on_own_time(child.pid, lambda: exited_unreaped(child.pid), budget=budget)
        except (OwnTimeExceeded, WallLimitExceeded) as exceeded:
            child.kill()
            child.wait()
            exceeded.add_note(
                f"running {shlex.join(args)}\n"
                f"its stdout:\n{_written_to(stdout)}\n"
                f"its stderr:\n{_written_to(stderr)}"
            )
            raise
        finally:
            if child.poll() is None:
                child.kill()
        return subprocess.CompletedProcess(
            list(args), child.wait(), _written_to(stdout), _written_to(stderr)
        )


def _written_to(file: IO[str]) -> str:
    """Everything written to *file* so far."""
    file.seek(0)
    return file.read()
