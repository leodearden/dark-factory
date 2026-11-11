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
import math
import os
import subprocess
import time
from collections.abc import Callable

from _watchdog_state import LogSink, atomic_write_json, read_json_state, within_min_interval

START_LIMIT_HIT_RESULT = "start-limit-hit"

# The detector's own state, separate from every clock the watchdog keeps.
# ORCH_RESTART_RATE_STATE overrides it (see restart_rate_state_path).
RESTART_RATE_STATE_RELPATH = os.path.join(
    "data", "orchestrator", "watchdog-restart-rate.json"
)

# The page's identity. escalation/src/escalation/submit.py enforces the
# severity and role boundary these must satisfy.
CRASH_LOOP_SENTINEL_PREFIX = "orchestrator-crash-loop-"
# Kept exactly: the watcher-design ruling's rider (a) keys host-role records on it.
CRASH_LOOP_AGENT_ROLE = "orchestrator-watchdog"
# Born at L2: submit.py accepts only BORN_AT_L2_SEVERITIES.
CRASH_LOOP_SEVERITY = "critical"
CRASH_LOOP_CATEGORY = "infra_issue"
CRASH_LOOP_SUBMIT_TIMEOUT_SECS = 60

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
        return f"{CRASH_LOOP_SENTINEL_PREFIX}{self.unit}"

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
    *last_escalation_epoch* caps page attempts to one per window, whichever
    kind; *start_limit_paged* is set only once a start-limit-hit page has
    landed, which caps it to one landed page per episode.
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

    A page is filed only when no page for *unit* was attempted within
    *window_secs*; otherwise the alarm comes back as *held*. The page epoch
    is stamped on attempt, so any page, landed or not, is capped to one
    attempt per window. The episode flag is not observe_unit's to set:
    mark_page_landed sets it after a successful submit, and a different
    known Result clears it here.
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
    if within_min_interval(window_secs, lambda: last, now=now):
        return Observation(history, alarm=None, held=candidate)
    attempted = dataclasses.replace(history, last_escalation_epoch=int(now))
    return Observation(attempted, alarm=candidate, held=None)


def mark_page_landed(history: UnitRestartHistory, alarm: CrashLoopAlarm) -> UnitRestartHistory:
    """A landed start-limit-hit page closes its episode; a failed one leaves it open, so it retries at window cadence."""
    if alarm.kind is AlarmKind.START_LIMIT_HIT:
        return dataclasses.replace(history, start_limit_paged=True)
    return history


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


def restart_rate_state_path(repo_dir: str) -> str:
    """Return the detector's state path: ORCH_RESTART_RATE_STATE, else under *repo_dir*."""
    return os.environ.get(
        "ORCH_RESTART_RATE_STATE", os.path.join(repo_dir, RESTART_RATE_STATE_RELPATH)
    )


def crash_loop_queue_dir(repo_dir: str) -> str:
    """Return the escalation queue pages go to: ORCH_CRASH_LOOP_QUEUE_DIR, else *repo_dir*'s."""
    return os.environ.get(
        "ORCH_CRASH_LOOP_QUEUE_DIR", os.path.join(repo_dir, "data", "escalations")
    )


def watchdog_uv_bin() -> str:
    """Return the uv that runs ``escalation submit``: absolute, as the watchdog unit sets no PATH.

    Mirrors scripts/dashboard-watchdog.py::UV_BIN; ORCH_WATCHDOG_UV_BIN overrides it.
    """
    return os.environ.get("ORCH_WATCHDOG_UV_BIN", "/home/leo/.local/bin/uv")


def load_restart_histories(path: str, log: LogSink) -> dict[str, UnitRestartHistory]:
    """Read every unit's history from *path*; {} when there is none or it is unreadable.

    A hand-edited entry loses only its unusable parts. An unusable
    last_escalation_epoch reads as None, which fails toward paging.
    """
    state = read_json_state(path, log) or {}
    histories: dict[str, UnitRestartHistory] = {}
    for unit, entry in state.items():
        if not isinstance(entry, dict):
            log(f"ignoring non-object restart-rate entry for {unit} at {path}")
            continue
        histories[unit] = _history_from_json(entry)
    return histories


def save_restart_histories(
    path: str, histories: dict[str, UnitRestartHistory], log: LogSink
) -> bool:
    """Write every unit's history to *path*; True iff the write landed."""
    payload = {unit: dataclasses.asdict(history) for unit, history in histories.items()}
    return atomic_write_json(path, payload, log)


def _history_from_json(entry: dict) -> UnitRestartHistory:
    raw_samples = entry.get("samples")
    samples = raw_samples if isinstance(raw_samples, list) else []
    return UnitRestartHistory(
        samples=tuple(s for s in map(_as_sample, samples) if s is not None),
        sampling_since=_as_count(entry.get("sampling_since")),
        last_escalation_epoch=_as_count(entry.get("last_escalation_epoch")),
        start_limit_paged=entry.get("start_limit_paged") is True,
    )


def _as_sample(raw: object) -> tuple[int, int] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    epoch, n = _as_count(raw[0]), _as_count(raw[1])
    if epoch is None or n is None:
        return None
    return epoch, n


def _as_count(raw: object) -> int | None:
    """A non-negative finite number as an int; None for anything else, bools included."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    if not math.isfinite(raw) or raw < 0:
        return None
    return int(raw)


def crash_loop_submit_argv(
    alarm: CrashLoopAlarm, *, uv_bin: str, repo_dir: str, queue_dir: str
) -> list[str]:
    """The ``escalation submit`` argv that files *alarm* as a born-at-L2 record."""
    return [
        uv_bin,
        "run",
        "--project",
        os.path.join(repo_dir, "escalation"),
        "escalation",
        "submit",
        "--queue-dir",
        queue_dir,
        "--task",
        alarm.task_sentinel,
        "--severity",
        CRASH_LOOP_SEVERITY,
        "--category",
        CRASH_LOOP_CATEGORY,
        "--summary",
        alarm.summary(),
        "--detail",
        alarm.detail(),
        "--agent-role",
        CRASH_LOOP_AGENT_ROLE,
    ]


def file_crash_loop_escalation(alarm: CrashLoopAlarm, *, repo_dir: str, log: LogSink) -> bool:
    """File *alarm* into *repo_dir*'s escalation queue; True iff submit succeeded.

    Never raises: a missing uv, a timeout, undecodable output or a non-zero
    exit is logged and reported as False. Deduplication is the caller's:
    observe_unit stamps the page epoch on attempt and mark_page_landed sets
    the episode flag only on a True return, so a broken submit of either kind
    retries at window cadence.
    """
    argv = crash_loop_submit_argv(
        alarm,
        uv_bin=watchdog_uv_bin(),
        repo_dir=repo_dir,
        queue_dir=crash_loop_queue_dir(repo_dir),
    )
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=CRASH_LOOP_SUBMIT_TIMEOUT_SECS,
        )
    except Exception as exc:
        log(
            f"crash-loop page {alarm.task_sentinel} not confirmed filed: "
            f"`escalation submit` raised {type(exc).__name__}: {exc}"
        )
        return False
    if result.returncode != 0:
        log(
            f"crash-loop page {alarm.task_sentinel} NOT filed: `escalation submit` "
            f"exited {result.returncode}: {result.stderr.strip()[:500]!r}"
        )
        return False
    log(f"filed born-at-L2 crash-loop page {alarm.task_sentinel} ({alarm.kind})")
    return True


def crash_loop_pass(
    units: list[str],
    read_status: Callable[[str], UnitRestartStatus],
    *,
    repo_dir: str,
    log: LogSink,
    now: float | None = None,
    alarm_count: int = RESTART_RATE_ALARM_COUNT,
    window_secs: int = RESTART_RATE_ALARM_WINDOW_SECS,
) -> dict[str, UnitRestartStatus]:
    """Sample every unit's restart status once, page on a crash loop, and return the statuses read.

    The watchdog runs this before every liveness gate, enabled and startup
    grace included, because the grace window is what blinded it to the
    2026-10-26 loop; OPERATIONS.md §8 "Crash loop" has the rationale. Each
    unit is read inside its own guard, so one failing read cannot blind the
    rest, and a unit whose read failed is absent from the result. A unit's
    observed history is kept before its page is filed, so a filing that
    raises still leaves the attempt stamped. The state is loaded and saved
    once per pass. Never raises.
    """
    statuses: dict[str, UnitRestartStatus] = {}
    try:
        now = time.time() if now is None else now
        path = restart_rate_state_path(repo_dir)
        histories = load_restart_histories(path, log)
        for unit in units:
            try:
                status = read_status(unit)
                statuses[unit] = status
                obs = observe_unit(
                    unit,
                    histories.get(unit, UnitRestartHistory()),
                    status,
                    now=now,
                    alarm_count=alarm_count,
                    window_secs=window_secs,
                )
                histories[unit] = obs.history
                histories[unit] = _act_on(obs, now=now, repo_dir=repo_dir, log=log)
            except Exception as exc:
                log(f"crash-loop detector error for {unit}: {exc!r}")
        save_restart_histories(path, histories, log)
    except Exception as exc:
        log(f"crash-loop detector pass error: {exc!r}")
    return statuses


def _act_on(
    obs: Observation, *, now: float, repo_dir: str, log: LogSink
) -> UnitRestartHistory:
    """File or log *obs*'s page and return the history to persist."""
    if obs.alarm is not None:
        filed = file_crash_loop_escalation(obs.alarm, repo_dir=repo_dir, log=log)
        return mark_page_landed(obs.history, obs.alarm) if filed else obs.history
    if obs.held is not None:
        age = now - (obs.history.last_escalation_epoch or now)
        log(
            f"crash loop on {obs.held.unit} ({obs.held.kind}): the last page attempt "
            f"was {age:.0f}s ago, inside the {obs.held.window_secs}s window; not re-filing"
        )
    return obs.history
