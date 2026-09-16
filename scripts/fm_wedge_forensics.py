#!/usr/bin/env python3
"""Where did fused-memory's event loop stop, for how long, and what was it doing?

Reads a `journalctl --user -u fused-memory.service` capture and reports the
windows in which the unit's own journal went COMPLETELY SILENT — the shared
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

Read-only. Stdlib only, and deliberately importing nothing from
`fused_memory`, `orchestrator` or `shared`: this runs in a mid-incident shell
where fused-memory itself is the thing that is down, and must not depend on
a synced venv to say so.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import re

# Above the 61s normal per-project harness cadence, below the 120s shorter of
# the two observed stalls — so it separates a stall from healthy idle with
# ~29s of margin on one side and ~30s on the other. Both endpoints are
# measured, and the module docstring above records them; do not restate the
# derivation anywhere else.
DEFAULT_STALL_THRESHOLD_SECONDS = 90.0


# How a stall ENDED, and what it says about the loop.
#
# asyncio installs its signal handlers ON the event loop, so servicing SIGTERM
# is itself proof the loop ran. A process that never services it for the whole
# stop timeout therefore had a BLOCKED loop, not a merely descheduled one — a
# descheduled process is scheduled and exits well inside that budget.
SIGTERM_UNSERVICED = "sigterm-unserviced"
STOPPED_ON_SIGNAL = "stopped-on-signal"
SELF_RECOVERED = "self-recovered"

# scripts/orchestrator-watchdog.py::_fused_memory_liveness_verdict's own words,
# reused verbatim so this output collates with its journal lines instead of
# needing a translation step. 'wedged' is port-open-but-/alive-unanswered;
# 'port-down' is the socket gone.
WEDGED = "wedged"
PORT_DOWN = "port-down"

# How many lines before the silence to keep as context. Enough to reach past
# the HTTP access-log chatter into the application's own loggers on the
# REDUCED captures this is normally fed; raise --context on a full window,
# where fm emits hundreds of access lines per second.
DEFAULT_CONTEXT_LINES = 10

# systemd's stop-time accounting, e.g.
#   Consumed 2h 29min 21.259s CPU time, 3.2G memory peak, 934.4M memory swap peak.
# The hours group is optional: a shorter-lived invocation prints `20min 30.697s`.
_CONSUMED_PATTERN = re.compile(
    r"Consumed (?:(?P<hours>\d+)h )?(?:(?P<minutes>\d+)min )?(?P<seconds>[\d.]+)s CPU time"
    r", (?P<memory>[\d.]+)(?P<memory_unit>[KMGT]) memory peak"
    r"(?:, (?P<swap>[\d.]+)(?P<swap_unit>[KMGT]) memory swap peak)?"
)
# systemd prints these with 1024-based units.
_SIZE_MULTIPLIER = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

_KILLED_PATTERN = re.compile(r"Killing process (?P<pid>\d+) \((?P<name>[^)]+)\) with signal SIGKILL")
# `<identifier>[<pid>]:` — the journal's own attribution of each line.
_WRITER_PATTERN = re.compile(r"^\S+ \S+ (?P<identifier>\S+)\[(?P<pid>\d+)\]:")

_SIGKILL_MARKERS = ("State 'stop-sigterm' timed out", "with signal SIGKILL")
_STOP_MARKERS = ("Stopping fused-memory", "Stopped fused-memory")
# A stall that BEGINS here began with the unit already down, so its socket was
# unbound and a watchdog probe would have seen 'port-down' rather than 'wedged'.
_UNIT_DOWN_MARKERS = ("Stopped fused-memory", "Main process exited", "with signal SIGKILL")


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
    """

    cpu_seconds: float
    memory_peak_bytes: float
    swap_peak_bytes: float | None


@dataclasses.dataclass(frozen=True)
class KilledProcess:
    pid: int
    name: str


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
    swap = match["swap"]
    return ResourceTotals(
        cpu_seconds=cpu_seconds,
        memory_peak_bytes=float(match["memory"]) * _SIZE_MULTIPLIER[match["memory_unit"]],
        swap_peak_bytes=float(swap) * _SIZE_MULTIPLIER[match["swap_unit"]] if swap else None,
    )


@dataclasses.dataclass(frozen=True)
class StallEpisode:
    """One window in which the unit emitted nothing at all.

    Bounded by real log lines on both sides: *stall_started_at* is the last
    line before the silence and *stall_ended_at* the first line after it, so
    the duration is measured rather than inferred from any configured timeout.

    *aftermath* is every line from the end of the stall up to the next stall
    (or the end of the capture) — the evidence for how this one resolved.
    *pre_stall_context* is the tail of what came before it.
    """

    stall_started_at: dt.datetime
    stall_ended_at: dt.datetime
    last_line_before_stall: str
    pre_stall_context: tuple[str, ...]
    aftermath: tuple[str, ...]
    unit_process_names: frozenset[str]
    unit_process_pids: frozenset[int]

    @property
    def stall_seconds(self) -> float:
        return (self.stall_ended_at - self.stall_started_at).total_seconds()

    @property
    def outcome(self) -> str:
        """Whether the loop was still blocked when systemd tried to stop it."""
        if any(marker in line for line in self.aftermath for marker in _SIGKILL_MARKERS):
            return SIGTERM_UNSERVICED
        if any(marker in line for line in self.aftermath for marker in _STOP_MARKERS):
            return STOPPED_ON_SIGNAL
        return SELF_RECOVERED

    @property
    def watchdog_verdict(self) -> str:
        """The verdict orchestrator-watchdog.py's probe would have returned
        DURING this stall.

        A stall that begins while the unit is running leaves the listening
        socket bound, so the port probe passes and only the /alive fetch —
        served by the same blocked loop — fails: that is 'wedged'. A gap that
        begins after the unit is already down is 'port-down' instead.
        """
        if any(marker in self.last_line_before_stall for marker in _UNIT_DOWN_MARKERS):
            return PORT_DOWN
        return WEDGED

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
    def foreign_killed_processes(self) -> tuple[KilledProcess, ...]:
        """Processes SIGKILLed out of the cgroup that are NOT the unit's own.

        The unit's own runtime is identified from the capture itself rather
        than from a hardcoded name list: a process belongs to it if its pid
        wrote journal lines under this unit, or if its name is one of the
        journal identifiers those lines carry. For episode 2 that excludes
        `python3` (pid 3655756 wrote the unit's log) and `uv` (the identifier
        every one of those lines is tagged with), leaving the `git` child —
        which is the whole signal, and would be diluted by the other two.
        """
        killed = []
        for line in self.aftermath:
            match = _KILLED_PATTERN.search(line)
            if match is None:
                continue
            pid, name = int(match["pid"]), match["name"]
            if pid in self.unit_process_pids or name in self.unit_process_names:
                continue
            killed.append(KilledProcess(pid=pid, name=name))
        return tuple(killed)


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
    """
    parsed = []
    for line in journal_text.splitlines():
        leading, _, _ = line.partition(" ")
        try:
            timestamp = dt.datetime.fromisoformat(leading)
        except ValueError:
            continue
        parsed.append((timestamp, line))
    return parsed


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
    starts = [
        index
        for index in range(1, len(lines))
        if (lines[index][0] - lines[index - 1][0]).total_seconds() > stall_threshold_seconds
    ]
    episodes = []
    for position, index in enumerate(starts):
        next_stall = starts[position + 1] - 1 if position + 1 < len(starts) else len(lines)
        episodes.append(
            StallEpisode(
                stall_started_at=lines[index - 1][0],
                stall_ended_at=lines[index][0],
                last_line_before_stall=lines[index - 1][1],
                pre_stall_context=tuple(
                    line for _, line in lines[max(0, index - context_lines):index]
                ),
                aftermath=tuple(line for _, line in lines[index:next_stall]),
                unit_process_names=names,
                unit_process_pids=pids,
            )
        )
    return episodes
