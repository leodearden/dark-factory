"""The orchestrator watchdog's crash-loop detector: NRestarts samples, the restart-rate / start-limit-hit alarm, and its page (task 8270).

Why it exists, and why it samples before every liveness gate, is told once in
OPERATIONS.md §8 "Fleet redeploy & watchdog", under "Crash loop".

Inputs are injected: *repo_dir* is the checkout under which the detector's
state and escalation queue live, *log* is the caller's journal sink so every
line lands on the CALLER's log, and the systemd status reader is passed in.
Nothing here reads a module global of scripts/orchestrator-watchdog.py. It
imports only _watchdog_state, downward, and is stdlib-only like its importer.

Import-resolution contract, the same as scripts/_watched_module_removals.py's:
this stays a flat sibling in scripts/, which a directly executed script finds
because its own directory is sys.path[0].
"""

from __future__ import annotations

import dataclasses
import enum
import os

START_LIMIT_HIT_RESULT = "start-limit-hit"

# Env-with-fallback: a typo must not crash the oneshot. Both clamp to >= 1:
# a count of 0 would alarm on zero growth for every unit, and a window of 0
# would prune every sample before it is counted.
try:
    RESTART_RATE_ALARM_COUNT = max(1, int(os.environ["ORCH_RESTART_RATE_ALARM_COUNT"]))
except (KeyError, ValueError):
    RESTART_RATE_ALARM_COUNT = 5

try:
    RESTART_RATE_ALARM_WINDOW_SECS = max(
        1, int(os.environ["ORCH_RESTART_RATE_ALARM_WINDOW_SECS"])
    )
except (KeyError, ValueError):
    RESTART_RATE_ALARM_WINDOW_SECS = 1800


@dataclasses.dataclass(frozen=True)
class UnitRestartStatus:
    """One read of a unit's systemd ``NRestarts`` and ``Result``; None where unreadable."""

    n_restarts: int | None
    result: str | None

    @property
    def start_limit_hit(self) -> bool:
        return self.result == START_LIMIT_HIT_RESULT

    def render(self) -> str:
        """``N/Result`` as one whitespace-free token, ``?`` for an unreadable field."""
        n = "?" if self.n_restarts is None else str(self.n_restarts)
        return f"{n}/{self.result or '?'}"


UNKNOWN_RESTART_STATUS = UnitRestartStatus(n_restarts=None, result=None)


class AlarmKind(enum.StrEnum):
    RESTART_RATE = "restart-rate"
    START_LIMIT_HIT = "start-limit-hit"


@dataclasses.dataclass(frozen=True)
class CrashLoopAlarm:
    """A page the detector decided on: which unit, why, and the in-window growth."""

    unit: str
    kind: AlarmKind
    restarts: int
    window_secs: int

    @property
    def task_sentinel(self) -> str:
        return f"orchestrator-crash-loop-{self.unit}"

    def summary(self) -> str:
        if self.kind is AlarmKind.START_LIMIT_HIT:
            return (
                f"orchestrator crash loop: {self.unit} is in start-limit-hit; "
                "systemd stopped restarting it and the watchdog is leaving it down"
            )
        return (
            f"orchestrator crash loop: {self.unit} — {self.restarts} restarts in "
            f"{self.window_secs}s (systemd NRestarts)"
        )

    def detail(self) -> str:
        return (
            f"scripts/orchestrator-watchdog.py saw {self.unit} {self._finding()}. "
            "Find the cause: "
            f"`journalctl --user -u {self.unit}` for each start's last lines, then "
            "the `runs` table in the project's runs.db, whose `exit_reason` and "
            "`startup_completed_at` columns (startup-hardening task γ) say whether "
            "and where each start died. A unit in start-limit-hit is left down by "
            "the watchdog: it runs neither reset-failed nor restart on it. Once the "
            "cause is fixed, restore it with "
            f"`systemctl --user reset-failed {self.unit} && "
            f"systemctl --user start {self.unit}`. Rationale: OPERATIONS.md §8, "
            "\"Crash loop\"."
        )

    def _finding(self) -> str:
        if self.kind is AlarmKind.START_LIMIT_HIT:
            return f"in Result={START_LIMIT_HIT_RESULT}"
        return (
            f"restart {self.restarts} times within {self.window_secs}s "
            "(NRestarts growth, counted through counter flushes)"
        )


@dataclasses.dataclass(frozen=True)
class UnitRestartHistory:
    """The detector's persisted memory of one unit.

    *samples* are ``(epoch, NRestarts)`` pairs pruned to the alarm window;
    *sampling_since* is never pruned, so it shows how long sampling has run.
    *last_escalation_epoch* caps pages to one per window, whichever kind;
    *start_limit_paged* caps start-limit-hit pages to one per episode.
    """

    samples: tuple[tuple[int, int], ...] = ()
    sampling_since: int | None = None
    last_escalation_epoch: int | None = None
    start_limit_paged: bool = False


def restart_growth(samples: tuple[tuple[int, int], ...]) -> int:
    """Total restarts across consecutive *samples*, counting through counter flushes.

    Any manual start (a watchdog revive, a sweep) resets NRestarts to 0, so a
    decrease from a to b means the counter was flushed and b restarts happened
    since: b is added, not b - a.
    """
    counts = [n for _epoch, n in samples]
    return sum(b - a if b >= a else b for a, b in zip(counts, counts[1:], strict=False))


@dataclasses.dataclass(frozen=True)
class Observation:
    """observe_unit's verdict: the new history, plus a page to file or one held back."""

    history: UnitRestartHistory
    alarm: CrashLoopAlarm | None
    held: CrashLoopAlarm | None


def observe_unit(
    unit: str,
    history: UnitRestartHistory,
    status: UnitRestartStatus,
    *,
    now: float,
    alarm_count: int,
    window_secs: int,
) -> Observation:
    """Fold one status read into *history* and decide whether *unit* pages.

    A page is filed only when no page for *unit* landed within *window_secs*;
    otherwise the alarm comes back as *held*. A filed page stamps
    last_escalation_epoch on attempt, and a start-limit-hit page also marks
    its episode paged until a different known Result clears it.
    """
    history = _record_sample(history, status, now=now, window_secs=window_secs)
    candidate = _alarm_candidate(
        unit, history, status, alarm_count=alarm_count, window_secs=window_secs
    )
    if status.result is not None and not status.start_limit_hit:
        history = dataclasses.replace(history, start_limit_paged=False)
    if candidate is None:
        return Observation(history, alarm=None, held=None)
    last = history.last_escalation_epoch
    if last is not None and now - last < window_secs:
        return Observation(history, alarm=None, held=candidate)
    filed = dataclasses.replace(
        history,
        last_escalation_epoch=int(now),
        start_limit_paged=history.start_limit_paged
        or candidate.kind is AlarmKind.START_LIMIT_HIT,
    )
    return Observation(filed, alarm=candidate, held=None)


def _record_sample(
    history: UnitRestartHistory,
    status: UnitRestartStatus,
    *,
    now: float,
    window_secs: int,
) -> UnitRestartHistory:
    samples = history.samples
    sampling_since = history.sampling_since
    if status.n_restarts is not None:
        samples = (*samples, (int(now), status.n_restarts))
        sampling_since = int(now) if sampling_since is None else sampling_since
    kept = tuple(s for s in samples if now - s[0] <= window_secs)
    return dataclasses.replace(history, samples=kept, sampling_since=sampling_since)


def _alarm_candidate(
    unit: str,
    history: UnitRestartHistory,
    status: UnitRestartStatus,
    *,
    alarm_count: int,
    window_secs: int,
) -> CrashLoopAlarm | None:
    growth = restart_growth(history.samples)
    if status.start_limit_hit and not history.start_limit_paged:
        return CrashLoopAlarm(unit, AlarmKind.START_LIMIT_HIT, growth, window_secs)
    if growth >= alarm_count:
        return CrashLoopAlarm(unit, AlarmKind.RESTART_RATE, growth, window_secs)
    return None
