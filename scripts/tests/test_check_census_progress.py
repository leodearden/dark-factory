"""Tests for scripts/legibility/check_census_progress.py — the CENSUS
progress predicate: did the census each trigger FIRE launched actually
advance census-state's ``last_census_at``?

Driven by SUBPROCESS with ``trickle_state.STATE_ROOT_ENV`` pointed at
tmp_path and a FAKE ``git`` on PATH that leaves a marker if ever invoked
(COPIED from scripts/tests/test_check_trickle_progress.py, not imported,
per this directory's copy-not-share convention). The trickle record is
seeded through the REAL ``trickle_state.record_run``, and census-state.json
is a real file at the path the writer recorded.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from legibility import trickle_state

SCRIPT = Path(__file__).parent.parent / "legibility" / "check_census_progress.py"

_FAKE_GIT_SRC = '''#!/usr/bin/env bash
# Fake `git` for testing check_census_progress.py -- the probe reads what
# the pipeline recorded, never the repo. Any invocation drops a marker.
set -euo pipefail
: > "$FAKE_GIT_CALLED_MARKER"
exit 0
'''

_FIRST_NIGHT = date(2026, 9, 1)


def _bin_dir(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    fake_git = bin_dir / "git"
    fake_git.write_text(_FAKE_GIT_SRC)
    fake_git.chmod(0o755)

    return bin_dir, tmp_path / "git_was_called"


def _run_probe(tmp_path, *args, extra_env=None, cwd=None):
    """Run the probe by subprocess with the legibility state root at
    tmp_path/state and the fake git on PATH. Returns (CompletedProcess,
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


def _census_state_path(tmp_path) -> Path:
    return tmp_path / "proj" / "docs" / "legibility" / "census-state.json"


def _write_census_state(path: Path, last_census_at: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_census_at": last_census_at}))


# night kind -> (exit_code, fired); fired None = trigger not evaluated.
_NIGHTS = {
    "fire": (0, True),
    "no-fire": (0, False),
    "not-evaluated": (1, None),
}


def _seed(tmp_path, monkeypatch, *, nights, anchor: str | None = "2026-08-02",
          census_state_path=None, project_id="dark_factory",
          recorded_at=None):
    """Drive the real record_run once per night in *nights* (keys of
    :data:`_NIGHTS`), each observing *anchor* at *census_state_path*.

    Night i's target_date is ``_FIRST_NIGHT + i``, so a streak that starts
    on the first night has ``since == '2026-09-01'``. The state root stays
    set for the rest of the test (see test_check_trickle_progress._seed for
    why it must not be unset)."""
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path / "state"))
    path = census_state_path or _census_state_path(tmp_path)
    stamp = recorded_at or datetime.now(UTC)

    state = None
    for i, night in enumerate(nights):
        exit_code, fired = _NIGHTS[night]
        census = None if fired is None else trickle_state.CensusObservation(
            fired=fired, last_census_at=anchor, census_state_path=str(path),
        )
        state = trickle_state.record_run(
            project_id,
            trickle_state.NightRun(
                target_date=_FIRST_NIGHT + timedelta(days=i),
                recorded_at=stamp - timedelta(days=len(nights) - 1 - i),
                exit_code=exit_code,
                counters=trickle_state.RunCounters(
                    total_records=1, zero_signal_dropped=1, dedupe_collapsed=0,
                    below_sampling_cut=0, budget_skipped=0, selected_count=0,
                ),
                census=census,
            ),
            barren_escalated=False,
        )
    return state


def _assert_no_git(git_marker):
    assert not git_marker.exists(), (
        "check_census_progress.py must never invoke git: it reads what the "
        "PIPELINE recorded plus census-state.json, never the repo history"
    )


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def test_n_fires_without_an_advance_are_flagged(tmp_path, monkeypatch):
    state = _census_state_path(tmp_path)
    _write_census_state(state, "2026-08-02")
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    err = result.stderr
    lowered = err.lower()
    assert "3 consecutive" in err
    assert "2026-09-01" in err, "must name when the streak began"
    assert "2026-08-02" in err, "must name the unmoved last census date"
    assert str(state) in err
    assert "journalctl" in err
    assert "headroom" in lowered and "counts" in lowered, (
        "must say a census deferred for usage headroom counts"
    )
    for other_verdict in ("barren", "max_daily_digest_bytes", "top_fraction"):
        assert other_verdict not in lowered
    _assert_no_git(git_marker)


def test_a_census_that_advanced_is_not_flagged(tmp_path, monkeypatch):
    state = _census_state_path(tmp_path)
    _write_census_state(state, "2026-08-02")
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3)
    _write_census_state(state, "2026-09-24")

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "advanced" in result.stdout.lower()
    assert "2026-08-02" in result.stdout
    assert "2026-09-24" in result.stdout
    _assert_no_git(git_marker)


def test_a_quiet_no_fire_streak_is_never_flagged(tmp_path, monkeypatch):
    _write_census_state(_census_state_path(tmp_path), "2026-08-02")
    _seed(tmp_path, monkeypatch, nights=["no-fire"] * 6)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    _assert_no_git(git_marker)


# ---------------------------------------------------------------------------
# The streak
# ---------------------------------------------------------------------------

def test_a_streak_below_the_threshold_passes(tmp_path, monkeypatch):
    _write_census_state(_census_state_path(tmp_path), "2026-08-02")
    _seed(tmp_path, monkeypatch, nights=["fire"] * 2)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "fires=2" in result.stdout
    assert "threshold 3" in result.stdout
    _assert_no_git(git_marker)


def test_nights_that_did_not_evaluate_the_trigger_do_not_break_the_streak(
    tmp_path, monkeypatch,
):
    _write_census_state(_census_state_path(tmp_path), "2026-08-02")
    _seed(tmp_path, monkeypatch,
          nights=["fire", "not-evaluated", "fire", "fire"])

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "3 consecutive" in result.stderr
    _assert_no_git(git_marker)


def test_a_never_censused_project_is_flagged(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3, anchor=None)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "never censused" in result.stderr.lower()
    assert "vanished" not in result.stderr.lower()
    _assert_no_git(git_marker)


def test_a_first_census_landing_counts_as_an_advance(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3, anchor=None)
    _write_census_state(_census_state_path(tmp_path), "2026-09-04")

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "never censused" in result.stdout.lower()
    assert "2026-09-04" in result.stdout
    _assert_no_git(git_marker)


def test_a_stale_record_is_still_assessed(tmp_path, monkeypatch):
    """Freshness is check_trickle_progress.py's verdict; an unadvanced
    streak at or over the threshold stays a true finding however old."""
    _write_census_state(_census_state_path(tmp_path), "2026-08-02")
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3,
          recorded_at=datetime.now(UTC) - timedelta(days=30))

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    assert "3 consecutive" in result.stderr
    _assert_no_git(git_marker)


# ---------------------------------------------------------------------------
# Distinct verdicts for what cannot be assessed
# ---------------------------------------------------------------------------

def test_a_vanished_census_state_is_its_own_verdict(tmp_path, monkeypatch):
    state = _census_state_path(tmp_path)
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3)
    assert not state.exists()

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    lowered = result.stderr.lower()
    assert "vanished" in lowered
    assert str(state) in result.stderr
    assert "advanced" not in lowered
    _assert_no_git(git_marker)


def test_an_unreadable_census_state_is_its_own_verdict(tmp_path, monkeypatch):
    state = _census_state_path(tmp_path)
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{corrupt")

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    lowered = result.stderr.lower()
    assert "unreadable" in lowered or "corrupt" in lowered
    assert str(state) in result.stderr
    assert "vanished" not in lowered
    _assert_no_git(git_marker)


def test_a_non_object_census_state_is_unreadable(tmp_path, monkeypatch):
    state = _census_state_path(tmp_path)
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("[1, 2]")

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    assert "unreadable" in result.stderr.lower()
    _assert_no_git(git_marker)


def test_a_missing_trickle_record_names_the_progress_probe(tmp_path):
    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    assert "check_trickle_progress.py" in result.stderr
    assert "never recorded" in result.stderr.lower()
    _assert_no_git(git_marker)


def test_a_corrupt_trickle_record_is_its_own_verdict(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, nights=["fire"])
    trickle_state.trickle_state_path("dark_factory").write_text("{corrupt")

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode != 0
    lowered = result.stderr.lower()
    assert "unreadable" in lowered or "corrupt" in lowered
    assert "never recorded" not in lowered
    _assert_no_git(git_marker)


def test_a_record_without_census_facts_passes(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, nights=["not-evaluated"] * 4)

    result, git_marker = _run_probe(tmp_path, "dark_factory", 3)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "fires=0" in result.stdout
    _assert_no_git(git_marker)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def test_the_threshold_defaults_to_the_module_constant(tmp_path, monkeypatch):
    default = trickle_state.DEFAULT_MAX_UNADVANCED_CENSUS_FIRES
    _write_census_state(_census_state_path(tmp_path), "2026-08-02")

    _seed(tmp_path, monkeypatch, nights=["fire"] * (default - 1))
    below, _ = _run_probe(tmp_path, "dark_factory")
    _seed(tmp_path, monkeypatch, nights=["fire"])
    at, git_marker = _run_probe(tmp_path, "dark_factory")

    assert below.returncode == 0, f"stderr={below.stderr!r}"
    assert at.returncode != 0, f"stdout={at.stdout!r}"
    _assert_no_git(git_marker)


def test_wrong_arity_prints_usage(tmp_path):
    for args in ((), ("dark_factory", 3, "extra")):
        result, git_marker = _run_probe(tmp_path, *args)

        assert result.returncode != 0
        assert "usage" in result.stderr.lower()
        _assert_no_git(git_marker)


def test_a_bad_threshold_prints_usage(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, nights=["fire"])
    for threshold in ("three", 0, -1):
        result, git_marker = _run_probe(tmp_path, "dark_factory", threshold)

        assert result.returncode != 0, threshold
        assert "usage" in result.stderr.lower(), threshold
        _assert_no_git(git_marker)


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"deterministic_runner._default_run_script EXECs a bound predicate "
        f"directly, so a non-executable probe is a dead probe. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_probe_imports_only_stdlib_under_bare_python(tmp_path, monkeypatch):
    _write_census_state(_census_state_path(tmp_path), "2026-08-02")
    _seed(tmp_path, monkeypatch, nights=["fire"] * 3)
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    result, git_marker = _run_probe(
        tmp_path, "dark_factory", 3,
        extra_env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": ""},
        cwd=str(outside),
    )

    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode in (0, 1), (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    _assert_no_git(git_marker)
