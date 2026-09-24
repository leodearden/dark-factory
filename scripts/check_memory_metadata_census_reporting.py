#!/usr/bin/env python3
"""Predicate: is the nightly memory-metadata coverage census producing
readings? Exit 0 = yes; non-zero = escalate.

Usage: check_memory_metadata_census_reporting.py <hours>

It reads both the census service's systemd state and the newest row of the
history artifact, because the census wrapper always exits 0 and a
healthy-looking unit can still have stopped the trend. OPERATIONS.md §12
"Nightly canonical/topic coverage census (05:00)" carries the full argument.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

UNIT = "memory-metadata-coverage-census.service"
INSTALLER = "scripts/install-memory-metadata-coverage-census-timer.sh"
UNIT_FIELDS = ("LoadState", "Result", "ExecMainStatus", "ExecMainExitTimestamp")
SYSTEMD_UTC_TIMESTAMP = "%a %Y-%m-%d %H:%M:%S UTC"
NEVER_RAN_TIMESTAMPS = frozenset({"", "n/a"})
DEFAULT_REPO = "/home/leo/src/dark-factory"
HISTORY_RELPATH = Path("plans") / "memory-metadata-coverage-history.json"
USAGE = (
    "usage: check_memory_metadata_census_reporting.py <hours>\n"
    "  <hours>: the freshness window, a finite positive number"
)


class NotReporting(Exception):
    """A named reason the census is not producing readings."""


def read_unit_state() -> dict[str, str]:
    # systemctl renders timestamps in the caller's local zone; pin it so the
    # format parsed below is the only one that can arrive.
    try:
        completed = subprocess.run(
            ["systemctl", "--user", "show", "-p", ",".join(UNIT_FIELDS), UNIT],
            env={**os.environ, "TZ": "UTC"},
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NotReporting(
            f"UNREADABLE UNIT STATE: could not read unit state of {UNIT} ({exc})"
        ) from None
    if completed.returncode != 0:
        raise NotReporting(
            f"UNREADABLE UNIT STATE: could not read unit state of {UNIT}: "
            f"systemctl exited {completed.returncode} ({completed.stderr.strip()!r})"
        )
    return dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )


def require_installed(state: dict[str, str]) -> None:
    load_state = state.get("LoadState", "")
    if load_state != "loaded":
        raise NotReporting(
            f"NOT INSTALLED: {UNIT} has LoadState={load_state!r}; "
            f"arm the 05:00 timer with {INSTALLER}"
        )


def last_run_time(state: dict[str, str]) -> datetime:
    raw = state.get("ExecMainExitTimestamp", "")
    if raw in NEVER_RAN_TIMESTAMPS:
        raise NotReporting(
            f"NEVER RAN: {UNIT} is installed but has never run "
            f"(ExecMainExitTimestamp={raw!r}); its first reading lands at the next 05:00"
        )
    try:
        return datetime.strptime(raw, SYSTEMD_UTC_TIMESTAMP).replace(tzinfo=UTC)
    except ValueError:
        raise NotReporting(
            f"UNREADABLE LAST RUN: {UNIT} reported ExecMainExitTimestamp={raw!r}, "
            f"not a systemd UTC timestamp"
        ) from None


def last_run_age(state: dict[str, str], window_hours: float) -> float:
    age = age_hours(last_run_time(state))
    if age > window_hours:
        raise NotReporting(
            f"STALE LAST RUN: {UNIT} last ran {age:.1f}h ago, "
            f"outside the {window_hours:g}h window"
        )
    return age


def require_succeeded(state: dict[str, str]) -> None:
    result = state.get("Result", "")
    status = state.get("ExecMainStatus", "")
    if result != "success" or status != "0":
        raise NotReporting(
            f"FAILED LAST RUN: {UNIT} last run ended Result={result!r} "
            f"ExecMainStatus={status!r}; read journalctl --user -u {UNIT} -n 100"
        )


def history_path() -> Path:
    return Path(os.environ.get("REPO") or DEFAULT_REPO) / HISTORY_RELPATH


def load_history(path: Path) -> object:
    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        raise NotReporting(f"NO HISTORY: {path} does not exist") from None
    except ValueError as exc:
        raise NotReporting(
            f"UNREADABLE HISTORY: {path} is not valid JSON ({exc})"
        ) from None
    except OSError as exc:
        raise NotReporting(
            f"UNREADABLE HISTORY: {path} could not be read ({exc})"
        ) from None


def newest_stamp(path: Path) -> datetime:
    history = load_history(path)
    runs = history.get("runs") if isinstance(history, dict) else None
    if not isinstance(runs, list):
        raise NotReporting(f"UNREADABLE HISTORY: {path} has no 'runs' list")
    if not runs:
        raise NotReporting(f"NO READINGS: 'runs' is empty in {path}")
    newest = runs[-1]
    stamp = newest.get("stamp") if isinstance(newest, dict) else None
    if stamp is None:
        raise NotReporting(
            f"UNREADABLE HISTORY: the newest row in {path} has no 'stamp'"
        )
    return parse_row_stamp(stamp, path)


def parse_row_stamp(stamp: object, path: Path) -> datetime:
    try:
        parsed = datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
    except ValueError:
        parsed = None
    if parsed is None:
        raise NotReporting(
            f"UNREADABLE HISTORY: the newest row in {path} has stamp={stamp!r}, "
            f"not an ISO-8601 timestamp"
        )
    if parsed.tzinfo is None:
        raise NotReporting(
            f"UNREADABLE HISTORY: the newest row in {path} has stamp={stamp!r} "
            f"with no UTC offset"
        )
    return parsed


def history_row_age(window_hours: float) -> float:
    path = history_path()
    age = age_hours(newest_stamp(path))
    if age > window_hours:
        raise NotReporting(
            f"STALE HISTORY ROW: the newest reading in {path} is {age:.1f}h old, "
            f"outside the {window_hours:g}h window, although {UNIT} reports a "
            f"recent run; read commit-step= on the wrapper's 'done (...)' journal line"
        )
    return age


def age_hours(then: datetime) -> float:
    return (datetime.now(UTC) - then).total_seconds() / 3600


def assess(window_hours: float) -> str:
    state = read_unit_state()
    require_installed(state)
    run_age = last_run_age(state, window_hours)
    require_succeeded(state)
    row_age = history_row_age(window_hours)
    return (
        f"OK: {UNIT} last ran {run_age:.1f}h ago; "
        f"newest history row {row_age:.1f}h ago (window {window_hours:g}h)"
    )


def parse_window(args: list[str]) -> float | None:
    if len(args) != 1:
        return None
    try:
        hours = float(args[0])
    except ValueError:
        return None
    return hours if math.isfinite(hours) and hours > 0 else None


def main(argv: list[str]) -> int:
    window_hours = parse_window(argv[1:])
    if window_hours is None:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        verdict = assess(window_hours)
    except NotReporting as reason:
        print(reason, file=sys.stderr)
        return 1
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
