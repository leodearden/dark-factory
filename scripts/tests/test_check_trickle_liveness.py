"""Tests for scripts/legibility/check_trickle_liveness.sh — the real
liveness predicate that replaces the fail-loud stub committed with
plans/confusion-reduction-prd.md (PRD task epsilon, decision 7: quiet
nights commit nothing, so liveness must probe systemd UNIT STATE, never
git history).

SCOPE: this probe answers "did the unit run" only. Whether the pipeline
actually PRODUCED anything is a separate question answered by
check_trickle_progress.py (see scripts/tests/test_check_trickle_progress.py,
task 3340). This file is deliberately unchanged in substance: its passing
unchanged is the proof that task 3340 altered no liveness semantics, which
tasks 2587/2615 already depend on.

Harness: scripts/tests/systemd_show_fakes.py.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from systemd_show_fakes import run_unit_probe, systemd_timestamp

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / "legibility" / "check_trickle_liveness.sh"


# ---------------------------------------------------------------------------
# step-27: RED -- check_trickle_liveness.sh
# ---------------------------------------------------------------------------

def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_alive_within_window_exits_zero(tmp_path):
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
        c[0] == "show" and "legibility-trickle@proj_a.service" in c for c in calls
    ), f"Expected a `systemctl show ... legibility-trickle@proj_a.service` probe; calls={calls!r}"

    assert not run.git_called, "check_trickle_liveness.sh must never invoke git (decision 7)"


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
    assert not run.git_called, "check_trickle_liveness.sh must never invoke git (decision 7)"


def test_failed_run_exits_nonzero(tmp_path):
    recent = datetime.now(UTC) - timedelta(hours=1)
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={
            "Result": "exit-code",
            "ExecMainStatus": "1",
            "ExecMainExitTimestamp": systemd_timestamp(recent),
        },
    )

    assert run.result.returncode != 0, (
        f"Expected a non-zero exit for a failed last run; "
        f"stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )
    assert not run.git_called, "check_trickle_liveness.sh must never invoke git (decision 7)"


def test_never_ran_exits_nonzero(tmp_path):
    run = run_unit_probe(
        SCRIPT, tmp_path, "proj_a", "24",
        fields={"Result": "", "ExecMainStatus": "", "ExecMainExitTimestamp": ""},
    )

    assert run.result.returncode != 0, (
        f"Expected a non-zero exit when the service has never run; "
        f"stdout={run.result.stdout!r} stderr={run.result.stderr!r}"
    )
    assert not run.git_called, "check_trickle_liveness.sh must never invoke git (decision 7)"
