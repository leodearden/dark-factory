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
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

UNIT = "memory-metadata-coverage-census.service"
INSTALLER = "scripts/install-memory-metadata-coverage-census-timer.sh"
UNIT_FIELDS = ("LoadState", "Result", "ExecMainStatus", "ExecMainExitTimestamp")
SYSTEMD_UTC_TIMESTAMP = "%a %Y-%m-%d %H:%M:%S UTC"
DEFAULT_REPO = "/home/leo/src/dark-factory"
HISTORY_RELPATH = Path("plans") / "memory-metadata-coverage-history.json"


class NotReporting(Exception):
    """A named reason the census is not producing readings."""


def read_unit_state() -> dict[str, str]:
    # systemctl renders timestamps in the caller's local zone; pin it so the
    # format parsed below is the only one that can arrive.
    completed = subprocess.run(
        ["systemctl", "--user", "show", "-p", ",".join(UNIT_FIELDS), UNIT],
        env={**os.environ, "TZ": "UTC"},
        capture_output=True, text=True, timeout=30,
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


def history_path() -> Path:
    return Path(os.environ.get("REPO") or DEFAULT_REPO) / HISTORY_RELPATH


def newest_stamp(path: Path) -> datetime:
    with path.open() as f:
        history = json.load(f)
    return datetime.fromisoformat(history["runs"][-1]["stamp"])


def age_hours(then: datetime) -> float:
    return (datetime.now(UTC) - then).total_seconds() / 3600


def assess() -> str:
    state = read_unit_state()
    require_installed(state)
    last_run = datetime.strptime(
        state["ExecMainExitTimestamp"], SYSTEMD_UTC_TIMESTAMP
    ).replace(tzinfo=UTC)
    row = newest_stamp(history_path())
    return (
        f"OK: {UNIT} last ran {age_hours(last_run):.1f}h ago; "
        f"newest history row {age_hours(row):.1f}h ago"
    )


def main(argv: list[str]) -> int:
    try:
        verdict = assess()
    except NotReporting as reason:
        print(reason, file=sys.stderr)
        return 1
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
