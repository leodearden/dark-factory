"""Tests for scripts/legibility/check_trickle_progress.py — the PROGRESS
predicate ("did signal flow?"), sibling to check_trickle_liveness.sh's
LIVENESS predicate ("did the unit run?").

Driven by SUBPROCESS with ``trickle_state.STATE_ROOT_ENV`` pointed at
tmp_path and a
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
    """Run the probe by subprocess with the legibility state root at
    tmp_path and the fake git on PATH. Returns (CompletedProcess,
    git_marker_path)."""
    bin_dir, git_marker = _bin_dir(tmp_path)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env[trickle_state.STATE_ROOT_ENV] = str(tmp_path / "state")
    env["FAKE_GIT_CALLED_MARKER"] = str(git_marker)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        env=env, capture_output=True, text=True, timeout=30, cwd=cwd,
    )
    return result, git_marker


# Counters AND exit code per entry, so a failed streak can follow
# productive nights in one seed. exit_code is per-ENTRY rather than
# per-seed because that is the only shape that can express the history the
# regression needs: productive nights, then a crash.
_SEED_OUTCOMES = {
    "productive": (0, dict(selected_count=2)),
    "quiet": (0, dict(zero_signal_dropped=5)),
    "barren-budget": (0, dict(budget_skipped=4)),
    "barren-cut": (0, dict(below_sampling_cut=3)),
    # Signal in, run broke downstream -- the 2026-08-18 reify shape.
    "failed": (1, dict(selected_count=1)),
}


def _seed(tmp_path, monkeypatch, *, outcomes, project_id="dark_factory",
          recorded_at=None):
    """Build a real state file by driving record_run for each entry in
    *outcomes* (a key of :data:`_SEED_OUTCOMES`). The LAST entry's
    recorded_at is *recorded_at* (default: now), so freshness is exercised
    against the real writer."""
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path / "state"))
    stamp = recorded_at or datetime.now(UTC)

    doc = None
    for i, outcome in enumerate(outcomes):
        # Space earlier runs a day apart, ending on `stamp`.
        at = stamp - timedelta(days=(len(outcomes) - 1 - i))
        entry_exit_code, counters = _SEED_OUTCOMES[outcome]
        # Annotated: without it the counter values infer as a narrow union
        # that pyright then checks positionally against record_run's later
        # keyword parameters when splatted as **full.
        full: dict[str, Any] = dict(
            zero_signal_dropped=0, dedupe_collapsed=0, below_sampling_cut=0,
            budget_skipped=0, selected_count=0,
        )
        # Derived, so every seeded night satisfies SampleResult's
        # conservation identity by construction.
        full.update(counters)
        full["total_records"] = sum(full.values())
        doc = trickle_state.record_run(
            project_id,
            target_date=at.date() if hasattr(at, "date") else date(2026, 7, 1),
            recorded_at=at,
            exit_code=entry_exit_code,
            **full,
        )
    monkeypatch.delenv(trickle_state.STATE_ROOT_ENV, raising=False)
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

def test_probe_reads_the_path_the_writer_wrote_under_a_divergent_environment(
    tmp_path, monkeypatch
):
    """The writer and the reader are DIFFERENT PROCESSES with
    independently-sourced environments; this pins that they still agree on
    one file.

    The writer is ``legibility-trickle@<project>.service`` under the
    ``systemd --user`` manager. The reader is the health timer, an
    orchestrator-EXEC'd ``before_done`` predicate inheriting whatever
    shell launched the orchestrator, or a dev shell. Under the
    pre-task-4514 code this exact shape returned ``('missing', None)``:
    the probe took branch 1 and exited 1 PERMANENTLY, which for a
    milestone binding is a born-at-L2 ``milestone_check_failed`` for a
    pipeline that is running perfectly — the outcome the 2026-09-14
    triage predicted would arrive the moment the probe was bound.
    """
    # The root _run_probe pins into the CHILD env; the writer below is
    # pointed at the same one, so the only thing left disagreeing between
    # the two sides is the ambient environment.
    state_root = tmp_path / "state"
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(state_root))

    # The writer's ambient environment.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "writer-xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "writer-home"))
    trickle_state.record_run(
        "dark_factory",
        target_date=date(2026, 7, 1),
        recorded_at=datetime.now(UTC),
        exit_code=0,
        total_records=2, zero_signal_dropped=0, dedupe_collapsed=0,
        below_sampling_cut=0, budget_skipped=0, selected_count=2,
    )

    # Asserted, not assumed: both sides must land on ONE tmp root. That is
    # the only thing keeping this test off the operator's real state file,
    # and a silent disagreement here would make the assertion below pass or
    # fail for the wrong reason.
    assert os.environ[trickle_state.STATE_ROOT_ENV] == str(state_root)

    # The reader's ambient environment, disagreeing on both levers.
    result, git_marker = _run_probe(
        tmp_path, "dark_factory", 3,
        extra_env={
            "XDG_STATE_HOME": str(tmp_path / "reader-xdg"),
            "HOME": str(tmp_path / "reader-home"),
        },
    )

    assert result.returncode == 0, (
        f"the probe resolved a different file than the writer wrote; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "productive" in result.stdout, (
        f"the probe must report the outcome the writer recorded; "
        f"stdout={result.stdout!r}"
    )
    _assert_no_git(git_marker)


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
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path / "state"))
    trickle_state.trickle_state_path("dark_factory").write_text("{corrupt")
    monkeypatch.delenv(trickle_state.STATE_ROOT_ENV, raising=False)

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
    """Arity widened from 2-3 to 2-4 in task 4514 (the new optional
    ``[max_failed_runs]``), so the over-arity case moves from 4 args to 5.

    Safe to change: tasks 2587/2615 bound only ``check_trickle_liveness.sh``,
    so no ``done_provenance`` rests on THIS script's arity."""
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3, 72, 2, "extra")

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


# ---------------------------------------------------------------------------
# The `failed` verdict (task 4514)
# ---------------------------------------------------------------------------


def test_a_failed_streak_exits_nonzero(tmp_path, monkeypatch):
    """THE regression. Before task 4514 this identical history recorded
    ``outcome=productive``, ``consecutive_barren_runs=0`` and exited 0
    FOREVER — a permanently broken coder reading as a healthy pipeline."""
    _seed(tmp_path, monkeypatch, outcomes=["failed", "failed"])
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    _assert_no_git(git_marker)


def test_the_failed_verdict_is_distinct_from_the_barren_one(
    tmp_path, monkeypatch
):
    """This file's "every failure verdict is DISTINCT" contract. A probe
    that cannot say WHICH absence it found is the trap it exists to close
    — and here the barren doors' remedies are actively WRONG."""
    _seed(tmp_path, monkeypatch, outcomes=["failed", "failed"])
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    stderr = result.stderr
    assert "failed" in stderr, "must name the outcome"
    assert "2" in stderr, "must name the consecutive count"
    assert "exit_code" in stderr or "exit code" in stderr
    assert "last_productive_at" in stderr
    assert "journalctl" in stderr, "must name where to read the crash"

    assert "max_daily_digest_bytes" not in stderr, (
        "raising the byte budget does nothing for a run that crashed"
    )
    assert "top_fraction" not in stderr, (
        "the sampling cut is not the problem when the pipeline broke "
        "downstream of it"
    )
    _assert_no_git(git_marker)


def test_the_failed_verdict_takes_precedence_over_the_barren_one(
    tmp_path, monkeypatch
):
    """record_run CARRIES the barren streak forward across failed runs, so
    a barren streak read during a failure window is stale by construction.
    Report the failure."""
    _seed(
        tmp_path, monkeypatch,
        outcomes=["barren-budget", "barren-budget", "failed", "failed"],
    )
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    assert "journalctl" in result.stderr, "the FAILED verdict must be the one"
    assert "max_daily_digest_bytes" not in result.stderr
    _assert_no_git(git_marker)


def test_a_sub_threshold_failed_night_reads_honestly(tmp_path, monkeypatch):
    """The "OK: last run was productive 0h ago" lie. Below the threshold
    the probe still exits 0 — one crash is already owned elsewhere — but
    it must not claim the last run was productive when it was not."""
    _seed(tmp_path, monkeypatch, outcomes=["productive", "failed"])
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "failed" in result.stdout, "the last run's outcome, reported"
    assert "consecutive_failed_runs=1" in result.stdout
    assert "was productive" not in result.stdout, (
        "the last run was NOT productive; reporting it as such is the "
        "exact lie this task closes"
    )
    _assert_no_git(git_marker)


def test_max_failed_runs_is_a_fourth_optional_positional(
    tmp_path, monkeypatch
):
    _seed(tmp_path, monkeypatch, outcomes=["failed", "failed"])
    loose, git_marker = _run_probe(tmp_path, "dark_factory", 3, 72, 5)
    assert loose.returncode == 0, (
        f"stdout={loose.stdout!r} stderr={loose.stderr!r}"
    )
    _assert_no_git(git_marker)

    _seed(tmp_path, monkeypatch, outcomes=["failed"])
    tight, git_marker = _run_probe(tmp_path, "dark_factory", 3, 72, 1)
    assert tight.returncode != 0
    _assert_no_git(git_marker)


def test_max_failed_runs_defaults_to_the_module_constant(
    tmp_path, monkeypatch
):
    """Pins ``trickle_state.DEFAULT_MAX_FAILED_RUNS`` as the default under
    BOTH shorter arities, so neither can drift from it."""
    assert trickle_state.DEFAULT_MAX_FAILED_RUNS == 2

    _seed(tmp_path, monkeypatch, outcomes=["failed", "failed"])
    two_arg, git_marker = _run_probe(tmp_path, "dark_factory", 3)
    assert two_arg.returncode != 0
    _assert_no_git(git_marker)

    three_arg, git_marker = _run_probe(tmp_path, "dark_factory", 3, 72)
    assert three_arg.returncode != 0
    _assert_no_git(git_marker)


def test_non_integer_max_failed_prints_usage(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, outcomes=["productive"])
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3, 72, "twice")

    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
    _assert_no_git(git_marker)
