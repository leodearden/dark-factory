"""Contract of ``shared.testing_own_time``: waits bounded by a child process's own time.

A child's own time is as ``shared.testing_own_time`` defines it. Time the child
spends blocked on something of its own counts against a wait's budget, so a
child that hangs still fails the wait. Time it spends ready to run but queued
behind other work does not, so a host too crowded to run the child fails the
wait only at its wall-clock limit. A wait that sees the child exit leaves the
exit status for the child's Popen to collect. A command run to its exit under
such a wait gives back its exit status and everything it wrote, and is killed
if the wait fails. A clock on a child reads its own time since the clock was
made. The wait's and the clock's contracts run on a scripted host, so no verdict
depends on how busy the real one is; real processes check only what the real
host provides, and what a run gives back.
"""

from __future__ import annotations

import contextlib
import math
import re
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from shared.testing_own_time import (
    LINUX_HOST,
    ChildSnapshot,
    OwnTimeClock,
    OwnTimeExceeded,
    Reading,
    WallLimitExceeded,
    exited_unreaped,
    run_on_own_time,
    state_letter,
    wait_on_own_time,
)

_on_the_host = pytest.mark.skipif(
    sys.platform != "linux", reason="LINUX_HOST reads /proc/PID/schedstat and /proc/PID/stat"
)

_WAITS_ON_ITS_OWN_FOR_S = 60.0

_ANY_PID = 1
_POLL_INTERVAL_S = 1 / 16
_ENDLESS_WAIT_S = 1000.0

_STALL_POSITIONS = [pytest.param(k, id=f"before-reading-{k}") for k in range(6)]

_STATES_WHILE_ITS_RUN_DELAY_GROWS = [
    pytest.param("S", id="credited-while-not-runnable"),
    pytest.param("R", id="queued-while-runnable"),
]

_DIES_ON_SIGTERM = "import time\nprint('ready', flush=True)\ntime.sleep(60)\n"

_READY_THEN_AWAITS_SIGTERM = (
    "import signal\n"
    "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})\n"
    "print('ready', flush=True)\n"
    "signal.sigwait({signal.SIGTERM})\n"
)

_WAITS_ON_ITS_OWN_AFTER_SIGTERM = _READY_THEN_AWAITS_SIGTERM + (
    f"import threading\nthreading.Event().wait({_WAITS_ON_ITS_OWN_FOR_S})\n"
)

_NAMES_ITSELF_WITH_A_SPACE_AND_A_PARENTHESIS = (
    "import ctypes\n"
    "PR_SET_NAME = 15\n"
    "if ctypes.CDLL(None).prctl(PR_SET_NAME, b'a) b', 0, 0, 0) != 0:\n"
    "    raise SystemExit('prctl(PR_SET_NAME) failed')\n"
)

_EXITS_ONCE_READY = "print('ready', flush=True)\n"

_SHORT_SCRIPT_BUDGET_S = 30.0
_MORE_THAN_A_PIPE_HOLDS = 256 * 1024
_EXIT_STATUS = 3

_PRINTS_ITS_CWD_AND_WHO_THEN_OVERFILLS_A_PIPE_ON_EACH_STREAM = (
    "import os, sys\n"
    "print(os.getcwd())\n"
    "print(os.environ['WHO'])\n"
    f"sys.stdout.write('o' * {_MORE_THAN_A_PIPE_HOLDS})\n"
    f"sys.stderr.write('e' * {_MORE_THAN_A_PIPE_HOLDS})\n"
    f"raise SystemExit({_EXIT_STATUS})\n"
)

_A_CHARACTER_CUT_SHORT = "\N{EURO SIGN}".encode()[:-1]
_A_CHARACTER_CUT_SHORT_ESCAPED = r"\xe2\x82"

_PRINTS_ITS_PID_AND_A_CHARACTER_CUT_SHORT_THEN_WAITS_ON_ITS_OWN_BEFORE_MAKING_ITS_ARGUMENT = (
    "import os, pathlib, sys, threading\n"
    "print(f'pid {os.getpid()}', flush=True)\n"
    "print(f'pid {os.getpid()} on stderr', file=sys.stderr, flush=True)\n"
    "for stream in (sys.stdout, sys.stderr):\n"
    f"    stream.buffer.write({_A_CHARACTER_CUT_SHORT!r})\n"
    "    stream.buffer.flush()\n"
    f"threading.Event().wait({_WAITS_ON_ITS_OWN_FOR_S})\n"
    "pathlib.Path(sys.argv[1]).touch()\n"
)


@dataclass
class _ScriptedHost:
    """A scripted host, structurally a ``shared.testing_own_time.WaitHost``.

    Fake time starts at *now* and moves only when the wait sleeps, or by *stall*
    seconds just before the wait's reading number *stall_before_reading*,
    counted from 0 over its readings of the wall clock and of the child. Whichever
    pid is asked about, the child has spent *on_cpu_share* of the time since fake
    time 0 on a CPU and been credited *run_delay_share* of it as run delay. It is in
    *state* until the earliest fake time in *state_changes*, and from each such
    time on in the state paired with it, until the next such time. Its times move
    until *progress_until* and then stay still, as in a queue stint the kernel has
    not credited while it is runnable.
    """

    on_cpu_share: float = 0.0
    run_delay_share: float = 0.0
    state: str = "S"
    state_changes: tuple[tuple[float, str], ...] = ()
    progress_until: float = math.inf
    stall: float = 0.0
    stall_before_reading: int | None = None
    now: float = 0.0
    _readings: int = field(default=0, init=False)

    def wall(self) -> float:
        self._take_reading()
        return self.now

    def child(self, pid: int) -> ChildSnapshot:
        self._take_reading()
        credited = min(self.now, self.progress_until)
        return ChildSnapshot(
            on_cpu=self.on_cpu_share * credited,
            run_delay=self.run_delay_share * credited,
            state=self._state_now(),
        )

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.now > _ENDLESS_WAIT_S:
            pytest.fail(f"the wait outlasted {_ENDLESS_WAIT_S:g} s of scripted time")

    def _take_reading(self) -> None:
        if self._readings == self.stall_before_reading:
            self.now += self.stall
        self._readings += 1

    def _state_now(self) -> str:
        latest_first = sorted(self.state_changes, reverse=True)
        return next((state for since, state in latest_first if since <= self.now), self.state)


def _wait(
    host: _ScriptedHost,
    done: Callable[[], bool],
    *,
    budget: float,
    wall_limit: float | None = None,
) -> None:
    """Wait on the scripted child, polling every _POLL_INTERVAL_S of scripted time."""
    wait_on_own_time(
        _ANY_PID,
        done,
        budget=budget,
        wall_limit=wall_limit,
        poll_interval=_POLL_INTERVAL_S,
        host=host,
    )


@contextlib.contextmanager
def _started(script: str) -> Iterator[subprocess.Popen[str]]:
    """A child running *script*, once it has printed its ready line; killed when the block ends."""
    proc = subprocess.Popen([sys.executable, "-I", "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline() == "ready\n"
        yield proc
    finally:
        proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


@_on_the_host
def test_a_wait_returns_once_its_condition_holds_leaving_the_exit_for_popen() -> None:
    with _started(_DIES_ON_SIGTERM) as proc:
        proc.send_signal(signal.SIGTERM)

        wait_on_own_time(proc.pid, lambda: exited_unreaped(proc.pid), budget=5.0)

        assert proc.wait(timeout=0) == -signal.SIGTERM


@_on_the_host
def test_time_a_process_spends_waiting_on_its_own_counts_against_the_budget() -> None:
    """A blocked process is not queued for a CPU, so the time it waits is its
    own. It gives up after _WAITS_ON_ITS_OWN_FOR_S, so a wait that did not
    count that time fails here rather than hanging."""
    budget = 0.5
    with _started(_WAITS_ON_ITS_OWN_AFTER_SIGTERM) as proc:
        proc.send_signal(signal.SIGTERM)

        with pytest.raises(OwnTimeExceeded) as raised:
            wait_on_own_time(proc.pid, lambda: exited_unreaped(proc.pid), budget=budget)

    assert raised.value.budget == budget


def test_run_delay_credited_to_a_process_does_not_count_against_the_budget() -> None:
    """The child, seen not runnable, is on a CPU for an eighth of the wait and
    credited the rest as run delay, so its condition comes true 2.4 s in, after
    0.3 s of running: the wait outlasts its 1 s budget and returns rather than
    fails."""
    host = _ScriptedHost(on_cpu_share=1 / 8, run_delay_share=7 / 8, state="S")

    _wait(host, lambda: host.now >= 2.4, budget=1.0)


def test_a_queue_stint_the_kernel_has_not_credited_does_not_count_against_the_budget() -> None:
    """The child is runnable and its times never move, which is one queue stint
    that the kernel credits only once it ends. The condition comes true 3.5 s in,
    past the 1 s budget, so the wait returns rather than fails."""
    host = _ScriptedHost(state="R")

    _wait(host, lambda: host.now >= 3.5, budget=1.0)


@pytest.mark.parametrize("state", ["S", "D"])
def test_time_a_blocked_process_spends_without_progress_counts_against_the_budget(
    state: str,
) -> None:
    """The wait attributes a child's still times to a queue stint the kernel has
    not credited only while the child is runnable. So a blocked child's time is
    its own however long its times stay still, time blocked in the kernel
    (state D) included."""
    host = _ScriptedHost(state=state)

    with pytest.raises(OwnTimeExceeded):
        _wait(host, lambda: False, budget=1.0)


@pytest.mark.parametrize("state", _STATES_WHILE_ITS_RUN_DELAY_GROWS)
@pytest.mark.parametrize(
    ("wall_limit", "expected_limit"),
    [
        pytest.param(1.5, 1.5, id="given"),
        pytest.param(None, 20.0, id="twenty-budgets-by-default"),
    ],
)
def test_a_process_credited_all_the_wait_as_run_delay_fails_the_wait_at_its_wall_limit(
    state: str, wall_limit: float | None, expected_limit: float
) -> None:
    """A child credited all of the wait as run delay has no own time, seen
    runnable (on a host too crowded to run it) or not, so only the wall limit
    ends the wait."""
    host = _ScriptedHost(run_delay_share=1, state=state)

    with pytest.raises(WallLimitExceeded) as raised:
        _wait(host, lambda: False, budget=1.0, wall_limit=wall_limit)

    assert raised.value.wall_limit == expected_limit
    assert expected_limit < host.now <= expected_limit + _POLL_INTERVAL_S


@pytest.mark.parametrize("stalled_before_reading", _STALL_POSITIONS)
def test_a_condition_that_comes_true_while_the_waiting_process_stalls_ends_the_wait(
    stalled_before_reading: int,
) -> None:
    """The child is blocked throughout and credited no run delay, so all the
    wait is its own time. A 5 s stall takes the wait past its 1 s budget and its
    2 s wall limit, and the condition comes true during the stall: the wait
    returns rather than fails."""
    host = _ScriptedHost(stall=5.0, stall_before_reading=stalled_before_reading)

    _wait(host, lambda: host.now >= 2.5, budget=1.0, wall_limit=2.0)


@pytest.mark.parametrize("state", _STATES_WHILE_ITS_RUN_DELAY_GROWS)
@pytest.mark.parametrize("stalled_before_reading", _STALL_POSITIONS)
def test_a_stall_of_the_waiting_process_never_counts_its_childs_run_delay_as_its_own(
    state: str,
    stalled_before_reading: int,
) -> None:
    """The child is credited all of the wait as run delay, seen runnable or not,
    so none of it is its own time. A 3 s stall, three budgets long, wherever it
    falls in the wait's first readings, leaves only the wall limit able to end
    the wait."""
    host = _ScriptedHost(
        run_delay_share=1, state=state, stall=3.0, stall_before_reading=stalled_before_reading
    )

    with pytest.raises(WallLimitExceeded):
        _wait(host, lambda: False, budget=1.0)


@pytest.mark.parametrize("stalled_before_reading", _STALL_POSITIONS)
def test_a_queue_stint_the_kernel_has_not_credited_never_counts_as_the_childs_own_time(
    stalled_before_reading: int,
) -> None:
    """The child runs for its first 0.25 s, then waits runnable in one stint the
    kernel never credits. A 3 s stall, three budgets long, wherever it falls in
    the wait's first readings, leaves only the wall limit able to end the wait."""
    host = _ScriptedHost(
        on_cpu_share=1,
        state="R",
        progress_until=0.25,
        stall=3.0,
        stall_before_reading=stalled_before_reading,
    )

    with pytest.raises(WallLimitExceeded):
        _wait(host, lambda: False, budget=1.0)


@pytest.mark.parametrize(
    ("failure", "on_cpu_share", "run_delay_share", "state", "progress_until", "wall_limit", "reading"),
    [
        pytest.param(
            OwnTimeExceeded,
            1 / 4,
            1 / 2,
            "D",
            math.inf,
            None,
            Reading(elapsed=2.0625, on_cpu=0.515625, run_delay=1.03125, state="D", uncredited=0.0),
            id="own-time",
        ),
        pytest.param(
            WallLimitExceeded,
            0.0,
            1,
            "R",
            math.inf,
            1.5,
            Reading(elapsed=1.5625, on_cpu=0.0, run_delay=1.5625, state="R", uncredited=0.0625),
            id="wall-limit",
        ),
        pytest.param(
            WallLimitExceeded,
            1,
            0.0,
            "R",
            0.25,
            1.5,
            Reading(elapsed=1.5625, on_cpu=0.25, run_delay=0.0, state="R", uncredited=1.375),
            id="uncredited-stint",
        ),
    ],
)
def test_a_failed_wait_reports_how_its_child_spent_the_time(
    failure: type[OwnTimeExceeded | WallLimitExceeded],
    on_cpu_share: float,
    run_delay_share: float,
    state: str,
    progress_until: float,
    wall_limit: float | None,
    reading: Reading,
) -> None:
    host = _ScriptedHost(
        on_cpu_share=on_cpu_share,
        run_delay_share=run_delay_share,
        state=state,
        progress_until=progress_until,
    )

    with pytest.raises(failure) as raised:
        _wait(host, lambda: False, budget=1.0, wall_limit=wall_limit)

    exceeded = raised.value
    assert isinstance(exceeded, failure)
    assert exceeded.reading == reading
    assert str(exceeded).startswith(str(reading))


def test_the_uncredited_bound_restarts_at_the_childs_latest_reading_not_runnable() -> None:
    """The child runs for its first 0.25 s, sleeps until 0.75 s, and then waits
    runnable in one stint the kernel never credits. A child seen not runnable is
    in no queue stint, so the bound counts from its latest reading asleep, 0.6875 s
    in, not from the reading before its last progress, 0.1875 s in."""
    host = _ScriptedHost(
        on_cpu_share=1,
        state="R",
        progress_until=0.25,
        state_changes=((0.25, "S"), (0.75, "R")),
    )

    with pytest.raises(WallLimitExceeded) as raised:
        _wait(host, lambda: False, budget=1.0, wall_limit=1.5)

    assert raised.value.reading == Reading(
        elapsed=1.5625, on_cpu=0.25, run_delay=0.0, state="R", uncredited=1.5625 - 0.6875
    )


def test_a_failed_wait_reports_only_what_its_child_spent_since_the_wait_began() -> None:
    """The child had spent a quarter of its first 100 s on a CPU and been
    credited half of them as run delay when the wait began, so the reading shows
    those shares of the wait alone."""
    host = _ScriptedHost(on_cpu_share=1 / 4, run_delay_share=1 / 2, state="D", now=100.0)

    with pytest.raises(OwnTimeExceeded) as raised:
        _wait(host, lambda: False, budget=1.0)

    reading = raised.value.reading
    assert reading.on_cpu == pytest.approx(reading.elapsed / 4)
    assert reading.run_delay == pytest.approx(reading.elapsed / 2)


def test_a_clock_leaves_out_a_queue_stint_the_kernel_has_not_credited() -> None:
    """The child is runnable and its times never move, which is one queue stint
    that the kernel credits only once it ends, so none of it is the child's own
    time however long it lasts."""
    host = _ScriptedHost(state="R")
    clock = OwnTimeClock(_ANY_PID, host=host)

    readings = []
    for _ in range(4):
        host.sleep(1.0)
        readings.append(clock())

    assert readings == [0.0, 0.0, 0.0, 0.0]


def test_a_clock_leaves_out_run_delay_credited_to_a_child_not_runnable() -> None:
    """The child, seen not runnable, is on a CPU for an eighth of the time and
    credited the rest as run delay, so 8 s hold 1 s of its own time."""
    host = _ScriptedHost(on_cpu_share=1 / 8, run_delay_share=7 / 8, state="S")
    clock = OwnTimeClock(_ANY_PID, host=host)

    host.sleep(8.0)

    assert clock() == pytest.approx(1.0)


@pytest.mark.parametrize("state", ["S", "D"])
def test_a_clock_counts_all_the_time_a_blocked_child_spends_without_progress(
    state: str,
) -> None:
    """A blocked child is in no queue stint, so its still times are its own,
    time blocked in the kernel (state D) included."""
    host = _ScriptedHost(state=state)
    clock = OwnTimeClock(_ANY_PID, host=host)

    host.sleep(2.0)

    assert clock() == pytest.approx(2.0)


def test_a_clock_counts_only_from_when_it_was_made() -> None:
    """The child had been credited half of its first 100 s as run delay when
    the clock was made, so a reading 2 s later holds those 2 s alone: 1 s of
    run delay and 1 s of its own, not the 50 s of its whole life."""
    host = _ScriptedHost(run_delay_share=1 / 2, state="S", now=100.0)
    clock = OwnTimeClock(_ANY_PID, host=host)

    host.sleep(2.0)

    assert clock.reading() == Reading(
        elapsed=2.0, on_cpu=0.0, run_delay=1.0, state="S", uncredited=0.0
    )
    assert clock() == pytest.approx(1.0)


def test_a_reading_never_reports_a_negative_own_time() -> None:
    """The bound on an uncredited stint can overlap run delay already credited,
    but own time cannot go below 0."""
    reading = Reading(elapsed=1.5625, on_cpu=0.0, run_delay=1.5625, state="R", uncredited=0.0625)

    assert reading.own == 0.0


def test_a_reading_reports_every_figure_it_holds() -> None:
    """The figures differ, so each shows in the report as itself and none can
    stand in for another."""
    reading = Reading(elapsed=9.0, on_cpu=1.0, run_delay=2.0, state="D", uncredited=3.0)
    figures = {
        "elapsed": reading.elapsed,
        "on_cpu": reading.on_cpu,
        "run_delay": reading.run_delay,
        "uncredited": reading.uncredited,
        "own": reading.own,
    }

    report = str(reading)

    assert [name for name, seconds in figures.items() if f"{seconds:.2f} s" not in report] == []
    assert reading.state in report


@_on_the_host
def test_the_host_reads_the_state_of_a_child_named_with_a_space_and_a_parenthesis() -> None:
    """/proc/PID/stat gives the process's name, in parentheses, before its state, so
    the child names itself with a space and a ')'. Exited but unreaped, it is a zombie."""
    with _started(_NAMES_ITSELF_WITH_A_SPACE_AND_A_PARENTHESIS + _EXITS_ONCE_READY) as proc:
        wait_on_own_time(proc.pid, lambda: exited_unreaped(proc.pid), budget=5.0)

        assert LINUX_HOST.child(proc.pid).state == "Z"


@_on_the_host
def test_a_run_returns_the_childs_exit_status_and_everything_it_wrote(tmp_path: Path) -> None:
    """The child writes more to each stream than a pipe holds, so a run that
    read a pipe only once the child had exited would leave the child blocked on
    a full pipe, spending its own time, until the wait failed."""
    args = [sys.executable, "-I", "-c", _PRINTS_ITS_CWD_AND_WHO_THEN_OVERFILLS_A_PIPE_ON_EACH_STREAM]

    run = run_on_own_time(
        args, budget=_SHORT_SCRIPT_BUDGET_S, cwd=tmp_path, env={"WHO": "the-test"}
    )

    assert run.returncode == _EXIT_STATUS
    assert run.args == args
    assert run.stdout == f"{tmp_path}\nthe-test\n" + "o" * _MORE_THAN_A_PIPE_HOLDS
    assert run.stderr == "e" * _MORE_THAN_A_PIPE_HOLDS


@_on_the_host
def test_a_run_whose_child_outspends_its_budget_is_killed_and_reports_what_it_wrote(
    tmp_path: Path,
) -> None:
    """The child prints its pid to each stream, ends each with a character cut
    short, as a child killed mid-write may leave it, and then waits on its own,
    which counts against its budget. Its start-up counts too, so the budget is a
    short script's, and the child prints long before that budget is spent. Had
    it waited out _WAITS_ON_ITS_OWN_FOR_S, it would then have made the file its
    argument names, so that file's absence shows the child was killed, and its
    pid naming no process that it was reaped."""
    outlived_its_wait = tmp_path / "outlived-its-wait"
    args = [
        sys.executable,
        "-I",
        "-c",
        _PRINTS_ITS_PID_AND_A_CHARACTER_CUT_SHORT_THEN_WAITS_ON_ITS_OWN_BEFORE_MAKING_ITS_ARGUMENT,
        str(outlived_its_wait),
    ]

    with pytest.raises(OwnTimeExceeded) as raised:
        run_on_own_time(args, budget=_SHORT_SCRIPT_BUDGET_S)

    assert raised.value.budget == _SHORT_SCRIPT_BUDGET_S
    notes = "\n".join(raised.value.__notes__)
    assert shlex.join(args) in notes
    pid_line = re.search(r"^pid (\d+)$", notes, re.MULTILINE)
    assert pid_line is not None, notes
    pid = int(pid_line[1])
    assert f"pid {pid}\n{_A_CHARACTER_CUT_SHORT_ESCAPED}" in notes
    assert f"pid {pid} on stderr\n{_A_CHARACTER_CUT_SHORT_ESCAPED}" in notes
    assert not outlived_its_wait.exists()
    assert not Path(f"/proc/{pid}").exists()


@pytest.mark.parametrize(
    ("stat", "state"),
    [
        pytest.param("42 (python3) S 1 42 42 0 -1 4194560\n", "S", id="plain-name"),
        pytest.param("42 (a b) R 1 42 42 0 -1 4194560\n", "R", id="name-holding-a-space"),
        pytest.param(
            "42 (a) S b) Z 1 42 42 0 -1 4194560\n",
            "Z",
            id="name-holding-a-parenthesis-and-a-state",
        ),
    ],
)
def test_the_state_is_the_first_token_after_the_last_closing_parenthesis(
    stat: str, state: str
) -> None:
    """The process name, in parentheses before the state, may hold spaces,
    parentheses, and even a token that looks like a state."""
    assert state_letter(stat) == state
