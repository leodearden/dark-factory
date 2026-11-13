#!/usr/bin/env python3
"""Where did fused-memory's event loop stop, for how long, and what was it doing?

Reads a journal capture of fused-memory, taken as Usage below shows, and
reports the windows in which the unit's own journal went COMPLETELY SILENT — the shared
signature of the 2026-09-15 23:55 and 2026-09-16 12:04 wedge episodes (task
5544), in which fused-memory stopped serving every HTTP route while the
process stayed up and its listening socket stayed bound.

Built because that diagnosis is expensive by hand and the failure recurs.
Reconstructing either episode meant a long sequence of ad-hoc
`journalctl | awk` pipelines over a ~6700- and ~8900-line window, and the
wedge went root-cause-unexamined twice for exactly that reason. The
measurements it recovers, on the two episodes it was built from:

    episode 1 (2026-09-15)  stall 23:55:10 -> 23:57:10 = 120s
                            teardown 50s, startup 39s, stopped cleanly
    episode 2 (2026-09-16)  stall 12:04:01 -> 12:06:52 = 171s
                            teardown 90s (SIGTERM never serviced -> SIGKILL),
                            startup 41s

against a normal per-project reconciliation cadence of 61s (measured
consecutive `Project reconciliation loop started for dark_factory` lines at
11:58:46, 11:59:47, 12:00:48, 12:01:49, 12:02:50, 12:03:51).

Usage — pipe a capture in:

    journalctl --user -u fused-memory.service -u fused-memory.socket \\
        --since "2026-09-16 12:03" --until "2026-09-16 12:10" \\
        -o short-iso --no-pager | scripts/fm_wedge_forensics.py

Include the socket unit. Under socket activation its `Closed` line is the only
witness of a deliberate stop, so without it a down-unit silence reports no
watchdog verdict.

`--user` IS MANDATORY. fused-memory.service is a systemd --user unit, so a
system-scope `journalctl -u fused-memory.service` prints "-- No entries --"
and looks exactly like an absence of evidence. That mistake is why this
failure went root-cause-unexamined twice, so an input in which nothing parses
is a LOUD failure here, never a quiet "0 episodes".

Read-only. Stdlib only, and deliberately importing nothing from
`fused_memory`, `orchestrator` or `shared`: this runs in a mid-incident shell
where fused-memory itself is the thing that is down, and must not depend on
a synced venv to say so.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import enum
import json
import re
import sys

# Above the 61s normal per-project harness cadence, below the 120s shorter of
# the two observed stalls — so it separates a stall from healthy idle with
# ~29s of margin on one side and ~30s on the other. Both endpoints are
# measured, and the module docstring above records them; do not restate the
# derivation anywhere else.
DEFAULT_STALL_THRESHOLD_SECONDS = 90.0

# The capture --help and every input-rejection message recommend.
CAPTURE_COMMAND = (
    "journalctl --user -u fused-memory.service -u fused-memory.socket "
    "--since ... --until ... -o short-iso --no-pager"
)


# How a stall ENDED, and what it says about the loop.
#
# asyncio installs its signal handlers ON the event loop, so servicing SIGTERM
# is itself proof the loop ran. A process that never services it for the whole
# stop timeout therefore had a BLOCKED loop, not a merely descheduled one — a
# descheduled process is scheduled and exits well inside that budget.
SIGTERM_UNSERVICED = "sigterm-unserviced"
STOPPED_ON_SIGNAL = "stopped-on-signal"
SELF_RECOVERED = "self-recovered"
# The fourth answer is that the question does not apply: the three above all
# describe how a BLOCKED LOOP answered a stop signal, and a silence that begins
# with the unit already stopped had no loop to block. Named rather than folded
# into 'self-recovered', which would tell a reader the loop resumed on its own.
UNIT_DOWN = "unit-down"

# scripts/orchestrator-watchdog.py::_fused_memory_liveness_verdict's own words,
# reused verbatim so this output collates with its journal lines instead of
# needing a translation step. 'wedged' is port-open-but-/alive-unanswered;
# 'port-down' is the socket gone, which under socket activation is a deliberate
# stop or disable rather than merely a down unit (see that function's docstring).
WEDGED = "wedged"
PORT_DOWN = "port-down"

# fused-memory.socket's first commit (8ac33458c4). No host ran the socket before it, so an older
# silence is from the regime in which the process bound port 8002 itself (OPERATIONS.md §"Socket activation").
FUSED_MEMORY_SOCKET_FIRST_COMMITTED_AT = dt.datetime.fromisoformat("2026-09-25T13:06:26+01:00")


class SocketState(enum.Enum):
    """fused-memory.socket's state, as its own latest lifecycle line reports it."""

    LISTENING = "listening"
    CLOSED = "closed"


# How many lines before the silence to keep as context. Enough to reach past
# the HTTP access-log chatter into the application's own loggers on the
# REDUCED captures this is normally fed; raise --context on a full window,
# where fm emits hundreds of access lines per second.
DEFAULT_CONTEXT_LINES = 10

# systemd's stop-time accounting, e.g.
#   Consumed 2h 29min 21.259s CPU time, 3.2G memory peak, 934.4M memory swap peak.
# CPU time is the only mandatory clause, and every optional one is really seen
# on this host: the hours drop out of a shorter-lived invocation (`20min
# 30.697s`), and a unit without memory accounting prints CPU time alone
# (`df-verify-reify-…scope: Consumed 8min 25.125s CPU time.`). Requiring the
# memory clause would discard the CPU time systemd DID measure and report the
# whole line as unmeasured.
_CONSUMED_PATTERN = re.compile(
    r"Consumed (?:(?P<hours>\d+)h )?(?:(?P<minutes>\d+)min )?(?P<seconds>[\d.]+)s CPU time"
    r"(?:, (?P<memory>[\d.]+)(?P<memory_unit>[BKMGT]) memory peak)?"
    r"(?:, (?P<swap>[\d.]+)(?P<swap_unit>[BKMGT]) memory swap peak)?"
)
# systemd prints these with 1024-based units, down to a bare `B` for a size
# below one kibibyte — which is how it renders a measured zero (`0B memory swap
# peak`, real). Without `B` here that zero would read as "not measured".
_SIZE_MULTIPLIER = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

_KILLED_PATTERN = re.compile(r"Killing process (?P<pid>\d+) \((?P<name>[^)]+)\) with signal SIGKILL")
# `<identifier>[<pid>]:` — the journal's own attribution of each line.
_WRITER_PATTERN = re.compile(r"^\S+ \S+ (?P<identifier>\S+)\[(?P<pid>\d+)\]:")

_SIGKILL_MARKERS = ("State 'stop-sigterm' timed out", "with signal SIGKILL")
_STOP_MARKERS = ("Stopping fused-memory", "Stopped fused-memory")
_SOCKET_UNIT = "fused-memory.socket"
_SOCKET_CLOSED_MARKERS = (f"Closed {_SOCKET_UNIT}",)
_SOCKET_STATE_MARKERS = (
    (SocketState.LISTENING, (f"Listening on {_SOCKET_UNIT}",)),
    (SocketState.CLOSED, _SOCKET_CLOSED_MARKERS),
)
# A stall that BEGINS here began with the unit already down.
# The stop-time accounting earns its place: systemd emits it as the unit
# deactivates, which makes that the LAST line of a real stop sequence, so
# without it a gap opening after a stop would still read 'wedged'.
# The socket closes only after the service's own stop, so in a capture that
# includes the socket unit its `Closed` line is the last line of a deliberate stop.
_UNIT_DOWN_MARKERS = (
    "Stopped fused-memory",
    "Main process exited",
    "with signal SIGKILL",
    _CONSUMED_PATTERN,
    *_SOCKET_CLOSED_MARKERS,
)

# The observed boundaries of the two recovery terms. Teardown ends at
# whichever came first of a clean stop, a SIGKILL and systemd's stop-time
# accounting, so an ignored SIGTERM is measured rather than assumed away; the
# accounting is the only witness when systemd goes straight from `Consumed` to
# `Starting` (scripts/tests/test_fm_wedge_forensics.py::NO_STOPPED_LINE_JOURNAL).
_TEARDOWN_START = ("Stopping fused-memory",)
_TEARDOWN_END = ("Stopped fused-memory", "with signal SIGKILL", _CONSUMED_PATTERN)
_STARTUP_START = ("Starting fused-memory",)
_STARTUP_END = ("Started fused-memory",)


@dataclasses.dataclass(frozen=True)
class ResourceTotals:
    """systemd's CUMULATIVE per-invocation accounting, emitted once at stop.

    Cumulative over the whole invocation, NOT over the stall — systemd offers
    no per-window resource history, and no sampler recorded one
    (scripts/install-load-sampler.sh is still a stub). So these numbers say
    what the process had used by the time it died, and nothing about what it
    was using while it was blocked. Do not read them as a stall measurement.

    Byte counts are converted from systemd's own 1024-based G/M rendering,
    which is already rounded to two significant figures; they carry that
    precision and no more.

    Each byte count is None when systemd printed no clause for it — a unit
    with memory accounting off prints CPU time alone — and that is NOT MEASURED,
    distinct from the measured zero systemd spells `0B`. CPU time is never
    None: a line that does not carry it is not a `Consumed` line at all.
    """

    cpu_seconds: float
    memory_peak_bytes: float | None
    swap_peak_bytes: float | None


@dataclasses.dataclass(frozen=True)
class CostBreakdown:
    """What one episode cost, in intervals that actually elapsed.

    Every term is measured off this capture's own timestamps — no constant is
    read from the unit file or from scripts/orchestrator-watchdog.py, so
    there is no second home for a number that drifts (heuristic 11), and an
    ignored SIGTERM shows up as the 90s it really took instead of the clean
    stop a config-derived estimate would have predicted.

    Absent terms are None, never 0: a self-recovered stall was never torn
    down or started, and zeros would read as an instantaneous restart.
    """

    stall_seconds: float
    teardown_seconds: float | None
    startup_seconds: float | None
    total_seconds: float | None

    @property
    def detection_seconds(self) -> None:
        """Always None: not witnessable from a fused-memory capture.

        The watchdog's consecutive-failure streak is logged in the
        orchestrator-watchdog journal, so the interval between its first
        failed probe and its restart decision simply is not in this input.
        Reported as a known absence rather than silently omitted, so a reader
        totalling the terms knows one is missing and where to find it.
        """
        return None

    @property
    def dominant_recovery_term(self) -> str | None:
        """Which of the two recovery terms cost more, or None if neither ran.

        The stall itself is deliberately not a candidate: it is the outage
        being recovered from, not part of the recovery.
        """
        terms = {
            name: seconds
            for name, seconds in (
                ("teardown", self.teardown_seconds),
                ("startup", self.startup_seconds),
            )
            if seconds is not None
        }
        return max(terms, key=terms.__getitem__) if terms else None


@dataclasses.dataclass(frozen=True)
class KilledProcess:
    """One process SIGKILLed out of the unit's cgroup when systemd gave up on it.

    *wrote_journal_lines* says only what it says: this capture credits output
    to this process, by pid or by journal identifier. It is an ANNOTATION and
    never a filter, because it is NOT the same question as "is this the unit's
    own runtime". A child the unit spawned inherits the unit's stderr, so
    anything it writes is journaled under this same unit — a writer set can
    therefore contain the very child we are looking for. Filtering on it would
    drop that child QUIETLY, as an empty list indistinguishable from "nothing
    outlived the stop", and that absence-of-evidence reading is the one this
    module exists to refuse.
    """

    pid: int
    name: str
    wrote_journal_lines: bool


def parse_consumed(line: str) -> ResourceTotals | None:
    """Read one systemd `Consumed ...` line, or None if *line* is not one."""
    match = _CONSUMED_PATTERN.search(line)
    if match is None:
        return None
    cpu_seconds = (
        int(match["hours"] or 0) * 3600
        + int(match["minutes"] or 0) * 60
        + float(match["seconds"])
    )
    return ResourceTotals(
        cpu_seconds=cpu_seconds,
        memory_peak_bytes=_size_bytes(match["memory"], match["memory_unit"]),
        swap_peak_bytes=_size_bytes(match["swap"], match["swap_unit"]),
    )


def _size_bytes(size: str | None, unit: str | None) -> float | None:
    """One of systemd's `<number><unit>` sizes in bytes, or None if unprinted."""
    if size is None or unit is None:
        return None
    return float(size) * _SIZE_MULTIPLIER[unit]


def _carries_marker(line: str, markers: tuple[str | re.Pattern[str], ...]) -> bool:
    """Does *line* carry any of *markers*?

    A marker is a literal substring, or a pattern for a line no substring can
    pin: the bare word "Consumed" also turns up in application lines, so
    systemd's stop-time accounting is recognised by `_CONSUMED_PATTERN` alone,
    the same definition :func:`parse_consumed` reads it with.
    """
    return any(
        marker in line if isinstance(marker, str) else marker.search(line) is not None
        for marker in markers
    )


@dataclasses.dataclass(frozen=True)
class StallEpisode:
    """One window in which the unit emitted nothing at all.

    Bounded by real log lines on both sides: *stall_started_at* is the last
    line before the silence and *stall_ended_at* the first line after it, so
    the duration is measured rather than inferred from any configured timeout.

    *restart_sequence* is the systemd restart THIS stall triggered, and is
    empty when it triggered none. *pre_stall_context* is the tail of what came
    before the silence. *socket_state_before_stall* is what fused-memory.socket's
    latest `Listening on`/`Closed` line at or before the silence reports, None
    when the capture carries none.
    """

    stall_started_at: dt.datetime
    stall_ended_at: dt.datetime
    last_line_before_stall: str
    socket_state_before_stall: SocketState | None
    pre_stall_context: tuple[str, ...]
    restart_sequence: tuple[tuple[dt.datetime, str], ...]
    unit_process_names: frozenset[str]
    unit_process_pids: frozenset[int]

    @property
    def stall_seconds(self) -> float:
        return (self.stall_ended_at - self.stall_started_at).total_seconds()

    @property
    def began_with_unit_down(self) -> bool:
        """Was the unit already stopped when this silence began?

        The fact :attr:`outcome` turns on and :attr:`watchdog_verdict` asks
        first, asked once so the two can never disagree about it. It does not
        settle the verdict alone: a down unit says nothing about its port.
        """
        return _carries_marker(self.last_line_before_stall, _UNIT_DOWN_MARKERS)

    @property
    def outcome(self) -> str:
        """Whether the loop was still blocked when systemd tried to stop it."""
        if self.began_with_unit_down:
            return UNIT_DOWN
        if any(_carries_marker(line, _SIGKILL_MARKERS) for line in self.aftermath):
            return SIGTERM_UNSERVICED
        if any(_carries_marker(line, _STOP_MARKERS) for line in self.aftermath):
            return STOPPED_ON_SIGNAL
        return SELF_RECOVERED

    @property
    def watchdog_verdict(self) -> str | None:
        """The verdict orchestrator-watchdog.py's probe would have returned
        DURING this stall, or None when this capture cannot witness it.

        A stall that begins while the unit is running leaves the listening
        socket bound, so the port probe passes and only the /alive fetch —
        served by the same blocked loop — fails: that is 'wedged'.

        A silence that begins with the unit already down is 'port-down' only
        where :attr:`port_witnessed_closed`. There are two witnesses. One is the
        capture predating socket activation, when the stopped process took its
        own socket with it. The other is fused-memory.socket's own latest
        lifecycle line reporting it closed. Otherwise this is None: under
        socket activation a restart or crash keeps the socket bound and logs
        nothing on it, so the probe could have read either verdict
        (OPERATIONS.md §"fused-memory liveness revive").
        """
        if not self.began_with_unit_down:
            return WEDGED
        return PORT_DOWN if self.port_witnessed_closed else None

    @property
    def port_witnessed_closed(self) -> bool:
        """Does this capture witness port 8002 closed when a down-unit silence began?"""
        if not self.began_with_unit_down:
            return False
        return (
            self.stall_started_at < FUSED_MEMORY_SOCKET_FIRST_COMMITTED_AT
            or self.socket_state_before_stall is SocketState.CLOSED
        )

    @property
    def aftermath(self) -> tuple[str, ...]:
        return tuple(line for _, line in self.restart_sequence)

    @property
    def costs(self) -> CostBreakdown:
        """Decompose this episode into the intervals its own journal witnesses."""
        teardown_start = self._first_time(_TEARDOWN_START)
        teardown_end = self._first_time(_TEARDOWN_END)
        startup_start = self._first_time(_STARTUP_START)
        recovered_at = self._first_time(_STARTUP_END)
        return CostBreakdown(
            stall_seconds=self.stall_seconds,
            teardown_seconds=_elapsed(teardown_start, teardown_end),
            startup_seconds=_elapsed(startup_start, recovered_at),
            total_seconds=_elapsed(self.stall_started_at, recovered_at),
        )

    def _first_time(self, markers: tuple[str | re.Pattern[str], ...]) -> dt.datetime | None:
        for timestamp, line in self.restart_sequence:
            if _carries_marker(line, markers):
                return timestamp
        return None

    @property
    def resources(self) -> ResourceTotals | None:
        """systemd's stop-time accounting, or None when it emitted none.

        None means NOT MEASURED, and is deliberately not a zero-filled record:
        a self-recovered stall never produced a `Consumed` line at all, and
        reporting it as 0 would read as "consumed nothing".
        """
        for line in self.aftermath:
            totals = parse_consumed(line)
            if totals is not None:
                return totals
        return None

    @property
    def killed_processes(self) -> tuple[KilledProcess, ...]:
        """Every process SIGKILLed out of the cgroup, in the order systemd killed them.

        Reported in full, annotated rather than filtered (see
        :class:`KilledProcess`). The annotation is read off the capture itself
        rather than a hardcoded name list: for episode 2 it marks `python3`
        (pid 3655756 wrote the unit's log) and `uv` (the identifier every one
        of those lines carries), leaving the `git` child as the one entry with
        no journal output of its own — which is the evidence that a
        synchronous subprocess was still in flight on the blocked loop.
        """
        killed = []
        for line in self.aftermath:
            match = _KILLED_PATTERN.search(line)
            if match is None:
                continue
            pid, name = int(match["pid"]), match["name"]
            killed.append(
                KilledProcess(
                    pid=pid,
                    name=name,
                    wrote_journal_lines=(
                        pid in self.unit_process_pids or name in self.unit_process_names
                    ),
                )
            )
        return tuple(killed)


class OffsetlessTimestampError(ValueError):
    """A capture line leads with a timestamp that carries no UTC offset."""


def timestamped_lines(journal_text: str) -> list[tuple[dt.datetime, str]]:
    """Pair every journal line with the timestamp it leads with, in order.

    `-o short-iso` prefixes each line with an offset-bearing ISO timestamp
    (`2026-09-16T12:04:01+01:00`), which `datetime.fromisoformat` accepts
    directly. A line whose first token is not such a timestamp is SKIPPED
    rather than fatal — a capture legitimately carries `-- Boot ... --`
    markers, blank lines and `-- No entries --`. Skipping them silently is
    safe only because the caller can still tell a capture that parsed from
    one that did not: an all-unparseable input yields an empty list, which
    the CLI reports as a loud failure rather than as "no episodes".

    A timestamp WITHOUT an offset raises :class:`OffsetlessTimestampError`
    instead. No short-iso capture prints one, and it cannot be ordered against
    an offset-bearing time such as FUSED_MEMORY_SOCKET_FIRST_COMMITTED_AT.
    """
    parsed = []
    for number, line in enumerate(journal_text.splitlines(), start=1):
        leading, _, _ = line.partition(" ")
        try:
            timestamp = dt.datetime.fromisoformat(leading)
        except ValueError:
            continue
        if timestamp.utcoffset() is None:
            raise OffsetlessTimestampError(
                f"line {number} leads with a timestamp that carries no UTC offset: {line!r}"
            )
        parsed.append((timestamp, line))
    return parsed


def restart_sequence_after(lines: list[tuple[dt.datetime, str]], stall_end: int) -> tuple[tuple[dt.datetime, str], ...]:
    """The systemd restart a stall triggered, or empty if it triggered none.

    A restart belongs to the silence it ENDS, not to an earlier one. The
    discriminator is that systemd's `Stopping` is itself the line that broke
    the silence: the watchdog decided while the loop was still unresponsive.
    Once the loop has resumed and is logging again, any later restart is a
    separate event.

    Without that bound, an episode's aftermath ran to the next stall — and on
    the real 2026-09-16 12:10-18:00 capture that made the self-recovered 160s
    stall at 12:16 swallow an unrelated restart 2h51m later, reporting it as
    `stopped-on-signal` with a 3s teardown and a 10300s total: three wrong
    numbers a reader had no way to doubt.

    Errs toward under-attribution. A restart whose `Stopping` lands just after
    the loop resumes reads as self-recovered, which understates a cost rather
    than inventing one.
    """
    if not _carries_marker(lines[stall_end][1], _TEARDOWN_START):
        return ()
    sequence = []
    for entry in lines[stall_end:]:
        sequence.append(entry)
        if _carries_marker(entry[1], _STARTUP_END):
            break
    return tuple(sequence)


def _socket_states_before(lines: list[tuple[dt.datetime, str]]) -> list[SocketState | None]:
    """fused-memory.socket's latest reported state before each line, in one forward pass.

    Element *i* is what the last lifecycle line among the first *i* lines
    reports, None until there is one, so the list is one longer than *lines*.
    """
    state: SocketState | None = None
    states: list[SocketState | None] = [state]
    for _, line in lines:
        reported = _socket_state_reported_by(line)
        if reported is not None:
            state = reported
        states.append(state)
    return states


def _socket_state_reported_by(line: str) -> SocketState | None:
    if _SOCKET_UNIT not in line:
        return None
    for state, markers in _SOCKET_STATE_MARKERS:
        if _carries_marker(line, markers):
            return state
    return None


def _elapsed(start: dt.datetime | None, end: dt.datetime | None) -> float | None:
    """Seconds between two observed timestamps, or None if either never happened."""
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def journal_writers(lines: list[tuple[dt.datetime, str]]) -> tuple[frozenset[str], frozenset[int]]:
    """The identifiers and pids the capture attributes its own lines to.

    Read out of the `<identifier>[<pid>]:` prefix journalctl stamps on every
    line, so "which processes are the service's own" is answered by the input
    rather than by a name list this script would have to keep in step.
    """
    identifiers, pids = set(), set()
    for _, line in lines:
        match = _WRITER_PATTERN.match(line)
        if match is not None:
            identifiers.add(match["identifier"])
            pids.add(int(match["pid"]))
    return frozenset(identifiers), frozenset(pids)


def analyze(
    journal_text: str,
    stall_threshold_seconds: float = DEFAULT_STALL_THRESHOLD_SECONDS,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> list[StallEpisode]:
    """Report every silence longer than *stall_threshold_seconds*, in order."""
    lines = timestamped_lines(journal_text)
    names, pids = journal_writers(lines)
    socket_states = _socket_states_before(lines)
    starts = [
        index
        for index in range(1, len(lines))
        if (lines[index][0] - lines[index - 1][0]).total_seconds() > stall_threshold_seconds
    ]
    episodes = []
    for index in starts:
        episodes.append(
            StallEpisode(
                stall_started_at=lines[index - 1][0],
                stall_ended_at=lines[index][0],
                last_line_before_stall=lines[index - 1][1],
                socket_state_before_stall=socket_states[index],
                pre_stall_context=tuple(
                    line for _, line in lines[max(0, index - context_lines):index]
                ),
                restart_sequence=restart_sequence_after(lines, index),
                unit_process_names=names,
                unit_process_pids=pids,
            )
        )
    return episodes


def as_dict(episode: StallEpisode) -> dict:
    """The episode as plain JSON-able data, absences preserved as null."""
    resources = episode.resources
    costs = episode.costs
    socket_state = episode.socket_state_before_stall
    return {
        "stall_started_at": episode.stall_started_at.isoformat(),
        "stall_ended_at": episode.stall_ended_at.isoformat(),
        "stall_seconds": episode.stall_seconds,
        "outcome": episode.outcome,
        "watchdog_verdict": episode.watchdog_verdict,
        "socket_state_before_stall": socket_state.value if socket_state is not None else None,
        "pre_stall_context": list(episode.pre_stall_context),
        "resources": dataclasses.asdict(resources) if resources else None,
        "killed_processes": [
            dataclasses.asdict(process) for process in episode.killed_processes
        ],
        "costs": {
            **dataclasses.asdict(costs),
            "detection_seconds": costs.detection_seconds,
            "dominant_recovery_term": costs.dominant_recovery_term,
        },
    }


def format_report(episodes: list[StallEpisode]) -> str:
    """The same findings as --json, for a human mid-incident."""
    if not episodes:
        return "no stalls above the threshold in this window"
    blocks = []
    for number, episode in enumerate(episodes, start=1):
        costs = episode.costs
        lines = [
            f"stall {number}: {episode.stall_seconds:.0f}s silent "
            f"({episode.stall_started_at:%H:%M:%S} -> {episode.stall_ended_at:%H:%M:%S})",
            f"  outcome          {episode.outcome} ({_show_verdict(episode)})",
            f"  teardown         {_show_seconds(costs.teardown_seconds)}",
            f"  startup          {_show_seconds(costs.startup_seconds)}",
            f"  total            {_show_seconds(costs.total_seconds)}",
            "  detection        unavailable (logged in the orchestrator-watchdog journal)",
        ]
        if costs.dominant_recovery_term:
            lines.append(f"  dominant term    {costs.dominant_recovery_term}")
        if episode.resources:
            lines.append(
                f"  consumed         {episode.resources.cpu_seconds:.0f}s CPU, "
                f"{_show_gigabytes(episode.resources.memory_peak_bytes)} peak "
                "(cumulative for the whole invocation, not this stall)"
            )
        for process in episode.killed_processes:
            attribution = (
                "logged under this unit" if process.wrote_journal_lines
                else "NO journal output of its own in this capture"
            )
            lines.append(
                f"  sigkilled        {process.name} (pid {process.pid}) — {attribution}"
            )
        lines.append("  last lines before the silence:")
        lines.extend(f"    {line}" for line in episode.pre_stall_context)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _show_seconds(seconds: float | None) -> str:
    return "not measured" if seconds is None else f"{seconds:.0f}s"


def _show_gigabytes(byte_count: float | None) -> str:
    return "not measured" if byte_count is None else f"{byte_count / 1024**3:.1f}G"


def _show_verdict(episode: StallEpisode) -> str:
    verdict = episode.watchdog_verdict
    if verdict is not None:
        return f"watchdog would report {verdict}"
    if episode.socket_state_before_stall is SocketState.LISTENING:
        return (
            "watchdog verdict not witnessed: fused-memory.socket was last seen listening, "
            "so port 8002 stayed bound and the verdict turned on how fast a "
            "probe-activated unit answered /alive"
        )
    return (
        "watchdog verdict not witnessed: port 8002 may have stayed bound in "
        "fused-memory.socket; add -u fused-memory.socket to the capture"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report fused-memory event-loop stalls from a journal capture on stdin. "
            f"Capture it with: {CAPTURE_COMMAND}  "
            "(--user is mandatory: system scope prints '-- No entries --')."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_STALL_THRESHOLD_SECONDS,
        help="seconds of silence that count as a stall (default: %(default)s)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=DEFAULT_CONTEXT_LINES,
        help="lines to keep from before each silence (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="emit the episode records as JSON")
    args = parser.parse_args(argv)

    journal_text = sys.stdin.read()
    try:
        parsed = timestamped_lines(journal_text)
    except OffsetlessTimestampError as error:
        print(f"{error}. Capture with `{CAPTURE_COMMAND}`.", file=sys.stderr)
        return 2
    if not parsed:
        print(
            "no journal lines parsed: every line lacked a leading short-iso timestamp. "
            f"Capture with `{CAPTURE_COMMAND}` — "
            "`--user` is mandatory, and system scope prints '-- No entries --'.",
            file=sys.stderr,
        )
        return 2

    episodes = analyze(journal_text, args.threshold, args.context)
    if args.json:
        print(json.dumps({"episodes": [as_dict(episode) for episode in episodes]}, indent=2))
    else:
        print(format_report(episodes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
