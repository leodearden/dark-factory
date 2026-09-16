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

Usage — pipe a capture in:

    journalctl --user -u fused-memory.service \\
        --since "2026-09-16 12:03" --until "2026-09-16 12:10" \\
        -o short-iso --no-pager | scripts/fm_wedge_forensics.py

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
import json
import re
import sys

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

# The observed boundaries of the two recovery terms. Teardown ends at
# whichever came first of a clean stop and a SIGKILL, so an ignored SIGTERM
# is measured rather than assumed away.
_TEARDOWN_START = ("Stopping fused-memory",)
_TEARDOWN_END = ("Stopped fused-memory", "with signal SIGKILL")
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
    """

    cpu_seconds: float
    memory_peak_bytes: float
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

    *restart_sequence* is the systemd restart THIS stall triggered, and is
    empty when it triggered none. *pre_stall_context* is the tail of what came
    before the silence.
    """

    stall_started_at: dt.datetime
    stall_ended_at: dt.datetime
    last_line_before_stall: str
    pre_stall_context: tuple[str, ...]
    restart_sequence: tuple[tuple[dt.datetime, str], ...]
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

    def _first_time(self, markers: tuple[str, ...]) -> dt.datetime | None:
        for timestamp, line in self.restart_sequence:
            if any(marker in line for marker in markers):
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
    if not any(marker in lines[stall_end][1] for marker in _TEARDOWN_START):
        return ()
    sequence = []
    for entry in lines[stall_end:]:
        sequence.append(entry)
        if any(marker in entry[1] for marker in _STARTUP_END):
            break
    return tuple(sequence)


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
    return {
        "stall_started_at": episode.stall_started_at.isoformat(),
        "stall_ended_at": episode.stall_ended_at.isoformat(),
        "stall_seconds": episode.stall_seconds,
        "outcome": episode.outcome,
        "watchdog_verdict": episode.watchdog_verdict,
        "pre_stall_context": list(episode.pre_stall_context),
        "resources": dataclasses.asdict(resources) if resources else None,
        "foreign_killed_processes": [
            dataclasses.asdict(process) for process in episode.foreign_killed_processes
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
            f"  outcome          {episode.outcome} (watchdog would report "
            f"{episode.watchdog_verdict})",
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
                f"{episode.resources.memory_peak_bytes / 1024**3:.1f}G peak "
                "(cumulative for the whole invocation, not this stall)"
            )
        for process in episode.foreign_killed_processes:
            lines.append(
                f"  still in cgroup  {process.name} (pid {process.pid}) SIGKILLed at teardown"
            )
        lines.append("  last lines before the silence:")
        lines.extend(f"    {line}" for line in episode.pre_stall_context)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _show_seconds(seconds: float | None) -> str:
    return "not measured" if seconds is None else f"{seconds:.0f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report fused-memory event-loop stalls from a journal capture on stdin. "
            "Capture it with: journalctl --user -u fused-memory.service "
            "--since ... --until ... -o short-iso --no-pager  "
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
    if not timestamped_lines(journal_text):
        print(
            "no journal lines parsed: every line lacked a leading short-iso timestamp. "
            "Capture with `journalctl --user -u fused-memory.service -o short-iso` — "
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
