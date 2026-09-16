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
import itertools

# Above the 61s normal per-project harness cadence, below the 120s shorter of
# the two observed stalls — so it separates a stall from healthy idle with
# ~29s of margin on one side and ~30s on the other. Both endpoints are
# measured, and the module docstring above records them; do not restate the
# derivation anywhere else.
DEFAULT_STALL_THRESHOLD_SECONDS = 90.0


@dataclasses.dataclass(frozen=True)
class StallEpisode:
    """One window in which the unit emitted nothing at all.

    Bounded by real log lines on both sides: *stall_started_at* is the last
    line before the silence and *stall_ended_at* the first line after it, so
    the duration is measured rather than inferred from any configured timeout.
    """

    stall_started_at: dt.datetime
    stall_ended_at: dt.datetime

    @property
    def stall_seconds(self) -> float:
        return (self.stall_ended_at - self.stall_started_at).total_seconds()


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


def analyze(
    journal_text: str,
    stall_threshold_seconds: float = DEFAULT_STALL_THRESHOLD_SECONDS,
) -> list[StallEpisode]:
    """Report every silence longer than *stall_threshold_seconds*, in order."""
    lines = timestamped_lines(journal_text)
    episodes = []
    for (before, _), (after, _) in itertools.pairwise(lines):
        if (after - before).total_seconds() > stall_threshold_seconds:
            episodes.append(StallEpisode(stall_started_at=before, stall_ended_at=after))
    return episodes
