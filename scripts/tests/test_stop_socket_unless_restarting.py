"""Tests for scripts/stop-socket-unless-restarting.sh, the ExecStopPost hook of
every socket-activated service.

The hook runs against a fake `systemctl` that answers `list-jobs` from a canned
job table (the column layout `systemctl --user list-jobs --no-legend` prints)
and records every other call.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

HOOK = pathlib.Path(__file__).resolve().parents[1] / "stop-socket-unless-restarting.sh"
SERVICE = "orchestrator-reify.service"

_FAKE_SYSTEMCTL = """#!/bin/sh
case "$*" in
  *list-jobs*) cat "$JOBS_FILE" ;;
  *) echo "$*" >> "$CALLS_FILE" ;;
esac
"""


def _run_hook(tmp_path: pathlib.Path, jobs: str) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "systemctl"
    fake.write_text(_FAKE_SYSTEMCTL)
    fake.chmod(0o755)
    jobs_file = tmp_path / "jobs"
    jobs_file.write_text(jobs)
    calls_file = tmp_path / "calls"
    calls_file.touch()
    subprocess.run(
        [str(HOOK), SERVICE],
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "JOBS_FILE": str(jobs_file),
            "CALLS_FILE": str(calls_file),
        },
        check=True,
        timeout=30,
    )
    return calls_file.read_text().splitlines()


def test_a_deliberate_stop_closes_the_socket(tmp_path):
    calls = _run_hook(tmp_path, f"7300211 {SERVICE}  stop running\n")
    assert calls == ["--user stop --no-block orchestrator-reify.socket"]


@pytest.mark.parametrize(
    "jobs",
    [
        pytest.param(f"7300065 {SERVICE}  restart running\n", id="restart"),
        pytest.param("", id="crash-awaiting-auto-restart"),
        pytest.param(
            "6810620 orchestrator-dark-factory.service  stop running\n",
            id="another-unit-stopping",
        ),
    ],
)
def test_anything_but_a_stop_of_this_service_keeps_the_socket(tmp_path, jobs):
    assert _run_hook(tmp_path, jobs) == []
