"""Tests for scripts/legibility/check_transcript_check_liveness.sh — the
liveness predicate for the legibility-transcript-check timer (task 2901,
follow-up to task 2893's lost-transcript detector).

Mirrors scripts/tests/test_check_trickle_liveness.py, on the same harness:
scripts/tests/systemd_show_fakes.py.

THE ONE PRINCIPLED DIVERGENCE from the mirrored trickle probe (DD4): for the
trickle, exit 1 means the pipeline broke (fail-loud), so its probe requires
ExecMainStatus=0. For the DETECTOR, exit 1 is the NORMAL, expected lost-
transcript ALARM -- it found a missing transcript and escalated -- so a probe
that flagged exit 1 as unhealthy would conflate "detector working and firing"
with "detector broken". The liveness question is purely "did it run on
schedule?" => freshness + not-never-ran. So Result in {success, exit-code}
AND ExecMainStatus in {0,1} is ALIVE; an abnormal Result (timeout / signal /
core-dump / ...) or ExecMainStatus >= 2 (argparse / malformed ExecStart) or
never-ran or staleness still FAIL.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from systemd_show_fakes import run_unit_probe, systemd_timestamp

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / "legibility" / "check_transcript_check_liveness.sh"


# ---------------------------------------------------------------------------
# step-3: RED -- check_transcript_check_liveness.sh
# ---------------------------------------------------------------------------

def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_alive_exit0_within_window_exits_zero(tmp_path):
    recent = datetime.now(UTC) - timedelta(hours=2)
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecMainExitTimestamp": systemd_timestamp(recent),
        },
    )

    assert run.result.returncode == 0, (
        f"stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )

    calls = run.systemctl_calls
    assert any(
        c[0] == "show" and "legibility-transcript-check@proj_a.service" in c for c in calls
    ), f"Expected a `systemctl show ... legibility-transcript-check@proj_a.service` probe; calls={calls!r}"

    assert not run.git_called, "check_transcript_check_liveness.sh must never invoke git"


def test_alive_exit1_finding_within_window_exits_zero(tmp_path):
    """THE DIVERGENCE CASE (DD4): a run that exited 1 within the window is a
    lost-transcript FINDING (the detector fired and escalated) -- a healthy,
    expected run. The probe MUST treat it as ALIVE (exit 0), unlike the
    trickle probe which requires ExecMainStatus=0."""
    recent = datetime.now(UTC) - timedelta(hours=1)
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={
            "Result": "exit-code",
            "ExecMainStatus": "1",
            "ExecMainExitTimestamp": systemd_timestamp(recent),
        },
    )

    assert run.result.returncode == 0, (
        f"A within-window exit-1 detector run (a lost-transcript finding) is "
        f"ALIVE and must exit 0 (DD4); stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )
    assert not run.git_called, "check_transcript_check_liveness.sh must never invoke git"


def test_older_than_window_exits_nonzero(tmp_path):
    stale = datetime.now(UTC) - timedelta(hours=48)
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecMainExitTimestamp": systemd_timestamp(stale),
        },
    )

    assert run.result.returncode != 0, (
        f"Expected a non-zero exit for a run older than the window; "
        f"stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )
    assert not run.git_called, "check_transcript_check_liveness.sh must never invoke git"


def test_never_ran_exits_nonzero(tmp_path):
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={"Result": "", "ExecMainStatus": "", "ExecMainExitTimestamp": ""},
    )

    assert run.result.returncode != 0, (
        f"Expected a non-zero exit when the service has never run; "
        f"stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )
    assert not run.git_called, "check_transcript_check_liveness.sh must never invoke git"


def test_unparseable_timestamp_exits_nonzero(tmp_path):
    """A non-empty but unparseable ExecMainExitTimestamp must FAIL loud, not be
    silently coerced into a benign (fresh) age. The never-ran guard only
    catches empty / "n/a"; a garbage value like 'not-a-date' clears the Result
    and ExecMainStatus checks and reaches the `date -d` parse, which must fail
    the probe rather than let GNU date treat it as parseable. Guards the
    parse-failure fail-loud branch (`if ! epoch="$(date -d ...)"`)."""
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecMainExitTimestamp": "not-a-date",
        },
    )

    assert run.result.returncode != 0, (
        f"Expected a non-zero exit for an unparseable ExecMainExitTimestamp "
        f"(the parse-failure branch must fail loud, not treat it as fresh); "
        f"stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )
    assert not run.git_called, "check_transcript_check_liveness.sh must never invoke git"


def test_abnormal_result_exits_nonzero(tmp_path):
    """A within-window run whose Result is abnormal (e.g. timeout / signal /
    core-dump) did NOT run to a clean exit, so it is NOT alive -- even with a
    fresh timestamp and an ExecMainStatus that looks benign."""
    recent = datetime.now(UTC) - timedelta(hours=1)
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={
            "Result": "timeout",
            "ExecMainStatus": "0",
            "ExecMainExitTimestamp": systemd_timestamp(recent),
        },
    )

    assert run.result.returncode != 0, (
        f"Expected a non-zero exit for an abnormal Result (process did not run "
        f"to a clean exit); stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )
    assert not run.git_called, "check_transcript_check_liveness.sh must never invoke git"


def test_exit_status_two_exits_nonzero(tmp_path):
    """ExecMainStatus >= 2 (argparse error / malformed ExecStart) is NOT the
    detector's normal exit-1 alarm -- it means the timer is genuinely broken,
    so the probe must FAIL even though Result=exit-code and the run is
    fresh."""
    recent = datetime.now(UTC) - timedelta(hours=1)
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={
            "Result": "exit-code",
            "ExecMainStatus": "2",
            "ExecMainExitTimestamp": systemd_timestamp(recent),
        },
    )

    assert run.result.returncode != 0, (
        f"Expected a non-zero exit for ExecMainStatus=2 (argparse / malformed "
        f"ExecStart); stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )
    assert not run.git_called, "check_transcript_check_liveness.sh must never invoke git"
