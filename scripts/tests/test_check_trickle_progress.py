"""Tests for scripts/legibility/check_trickle_progress.py — the PROGRESS
predicate ("did signal flow?"), sibling to check_trickle_liveness.sh's
LIVENESS predicate ("did the unit run?").

Driven by SUBPROCESS with ``XDG_STATE_HOME`` pointed at tmp_path and a
FAKE ``git`` shimmed onto PATH that only leaves a marker if ever invoked
— lifted from scripts/tests/test_check_trickle_liveness.py (COPIED, not
imported, matching how that file itself copies the fake-`systemctl`
convention from test_deploy_w5_recon_reliability.py). Every case asserts
the marker is absent, so PRD decision 7's "never a git-history probe"
stays an ASSERTION for the new probe too, not an assumption.

State files are seeded by calling ``trickle_state.record_run`` directly,
so the probe is exercised against REAL writer output rather than a
hand-rolled fixture that could drift from it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from legibility import trickle_state

SCRIPT = Path(__file__).parent.parent / "legibility" / "check_trickle_progress.py"

_FAKE_GIT_SRC = '''#!/usr/bin/env bash
# Fake `git` for testing check_trickle_progress.py -- decision 7 forbids
# inferring pipeline health from the repo's contents. Any invocation drops
# a marker file so the test can assert git was NEVER called.
set -euo pipefail
: > "$FAKE_GIT_CALLED_MARKER"
exit 0
'''


def _bin_dir(tmp_path):
    """Write an executable fake `git` into <tmp_path>/bin/. Returns
    (bin_dir, git_marker_path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    fake_git = bin_dir / "git"
    fake_git.write_text(_FAKE_GIT_SRC)
    fake_git.chmod(0o755)

    return bin_dir, tmp_path / "git_was_called"


def _run_probe(tmp_path, *args, extra_env=None, cwd=None):
    """Run the probe by subprocess with XDG_STATE_HOME at tmp_path and the
    fake git on PATH. Returns (CompletedProcess, git_marker_path)."""
    bin_dir, git_marker = _bin_dir(tmp_path)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    env["FAKE_GIT_CALLED_MARKER"] = str(git_marker)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        env=env, capture_output=True, text=True, timeout=30, cwd=cwd,
    )
    return result, git_marker


def _seed(tmp_path, monkeypatch, *, outcomes, project_id="dark_factory",
          recorded_at=None, exit_code=0):
    """Build a real state file by driving record_run for each entry in
    *outcomes* (one of 'productive' / 'quiet' / 'barren-budget' /
    'barren-cut'). The LAST entry's recorded_at is *recorded_at* (default:
    now), so freshness is exercised against the real writer."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    stamp = recorded_at or datetime.now(UTC)

    doc = None
    for i, outcome in enumerate(outcomes):
        # Space earlier runs a day apart, ending on `stamp`.
        at = stamp - timedelta(days=(len(outcomes) - 1 - i))
        counters = {
            "productive": dict(selected_count=2, total_records=2),
            "quiet": dict(zero_signal_dropped=5, total_records=5),
            "barren-budget": dict(budget_skipped=4, total_records=4),
            "barren-cut": dict(below_sampling_cut=3, total_records=3),
        }[outcome]
        # Annotated: without it the counter values infer as a narrow union
        # that pyright then checks positionally against record_run's later
        # keyword parameters when splatted as **full.
        full: dict[str, Any] = dict(
            zero_signal_dropped=0, dedupe_collapsed=0, below_sampling_cut=0,
            budget_skipped=0, selected_count=0,
        )
        full.update(counters)
        doc = trickle_state.record_run(
            project_id,
            target_date=at.date() if hasattr(at, "date") else date(2026, 7, 1),
            recorded_at=at,
            exit_code=exit_code,
            **full,
        )
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    return doc


def _assert_no_git(git_marker):
    assert not git_marker.exists(), (
        "check_trickle_progress.py must never invoke git (PRD decision 7): "
        "it reads what the PIPELINE recorded, never what the repo contains"
    )


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------

def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"deterministic_runner._default_run_script EXECs a bound predicate "
        f"directly, so a non-executable probe is a dead probe. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_streak_below_threshold_exits_zero(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["productive", "barren-budget"])
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout
    _assert_no_git(git_marker)


def test_streak_at_threshold_exits_nonzero(tmp_path, monkeypatch):
    _seed(
        tmp_path, monkeypatch,
        outcomes=["barren-budget", "barren-budget", "barren-budget"],
    )
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "3" in result.stderr, "must name the streak count"
    assert "barren" in result.stderr, "must name the outcome"
    assert "trickle-state.json" in result.stderr, "must name the state file"
    _assert_no_git(git_marker)


def test_streak_above_threshold_exits_nonzero(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["barren-budget"] * 5)
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    _assert_no_git(git_marker)


def test_budget_door_streak_names_the_budget_remedy(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["barren-budget"] * 3)
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    assert "max_daily_digest_bytes" in result.stderr
    assert "top_fraction" not in result.stderr, (
        "the two doors have DIFFERENT remedies and must never be conflated "
        "(SampleResult's own docstring)"
    )
    _assert_no_git(git_marker)


def test_sampling_cut_streak_names_the_sampling_remedy(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["barren-cut"] * 3)
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    assert "top_fraction" in result.stderr or "per_stratum_min" in result.stderr
    assert "max_daily_digest_bytes" not in result.stderr, (
        "raising the byte budget does nothing for a below_sampling_cut record"
    )
    _assert_no_git(git_marker)


def test_a_quiet_night_never_fails_the_progress_probe(tmp_path, monkeypatch):
    """THE false-alarm guard. A genuinely quiet night must pass — this is
    PRD decision 7's guarantee in executable form at the probe boundary,
    and it is what lets this probe exist at all."""
    _seed(tmp_path, monkeypatch, outcomes=["quiet"] * 6)
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"a quiet night must never alarm; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    _assert_no_git(git_marker)


def test_a_productive_night_exits_zero(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["productive"])
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "productive" in result.stdout
    _assert_no_git(git_marker)


def test_missing_state_file_is_its_own_verdict(tmp_path):
    """"Never recorded a run" is a DIFFERENT failure from a barren streak
    — a probe that cannot say WHICH absence it found is the trap this task
    exists to close."""
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    assert "never recorded a run" in result.stderr.lower()
    assert "streak" not in result.stderr.lower()
    _assert_no_git(git_marker)


def test_corrupt_state_file_is_its_own_verdict(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["productive"])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    trickle_state.trickle_state_path("dark_factory").write_text("{corrupt")
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    lowered = result.stderr.lower()
    assert "unreadable" in lowered or "corrupt" in lowered
    assert "never recorded a run" not in lowered
    _assert_no_git(git_marker)


def test_stale_recorder_exits_nonzero(tmp_path, monkeypatch):
    """The pipeline stopped writing state at all — distinct from a barren
    streak, and the recorded outcome alone would look healthy."""
    old = datetime.now(UTC) - timedelta(hours=100)
    _seed(tmp_path, monkeypatch, outcomes=["productive"], recorded_at=old)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3, 72)

    assert result.returncode != 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "72" in result.stderr, "must name the window"
    assert "100" in result.stderr or "h" in result.stderr, "must name the age"
    _assert_no_git(git_marker)


def test_just_inside_the_freshness_window_exits_zero(tmp_path, monkeypatch):
    recent = datetime.now(UTC) - timedelta(hours=71)
    _seed(tmp_path, monkeypatch, outcomes=["productive"], recorded_at=recent)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3, 72)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    _assert_no_git(git_marker)


def test_max_age_hours_defaults_to_seventy_two(tmp_path, monkeypatch):
    """Third arg is optional and defaults to kappa's 72h window."""
    old = datetime.now(UTC) - timedelta(hours=100)
    _seed(tmp_path, monkeypatch, outcomes=["productive"], recorded_at=old)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    assert "72" in result.stderr
    _assert_no_git(git_marker)


def test_wrong_arity_prints_usage(tmp_path):
    result, git_marker = _run_probe(tmp_path, "dark_factory")

    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
    _assert_no_git(git_marker)


def test_too_many_args_prints_usage(tmp_path):
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3, 72, "extra")

    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
    _assert_no_git(git_marker)


def test_non_integer_max_barren_prints_usage(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["productive"])
    result, git_marker = _run_probe(tmp_path, "dark_factory", "three")

    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
    _assert_no_git(git_marker)


def test_non_integer_max_age_prints_usage(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["productive"])
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3, "soon")

    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
    _assert_no_git(git_marker)


def test_probe_imports_only_stdlib_under_bare_python(tmp_path, monkeypatch):
    """Pins the constraint that deterministic_runner._default_run_script
    EXECs this file directly — no `uv run`, no project venv, no package
    context. Run with PYTHONNOUSERSITE=1 from OUTSIDE the repo: it must
    still produce a VERDICT, not an ImportError traceback."""
    _seed(tmp_path, monkeypatch, outcomes=["productive"])

    outside = tmp_path / "elsewhere"
    outside.mkdir()

    result, git_marker = _run_probe(
        tmp_path, "dark_factory", 3,
        extra_env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": ""},
        cwd=str(outside),
    )

    assert "Traceback" not in result.stderr, (
        f"the probe must not depend on a project venv or package context; "
        f"stderr={result.stderr!r}"
    )
    assert result.returncode in (0, 1), (
        f"expected a verdict exit code; got {result.returncode} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    _assert_no_git(git_marker)
