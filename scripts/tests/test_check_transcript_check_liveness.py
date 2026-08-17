"""Tests for scripts/legibility/check_transcript_check_liveness.sh — the
liveness predicate for the legibility-transcript-check timer (task 2901,
follow-up to task 2893's lost-transcript detector).

Mirrors scripts/tests/test_check_trickle_liveness.py: drives the script via
subprocess with a FAKE `systemctl` shimmed onto PATH (answers `show -p ...`
property lines from a JSON state file) and a FAKE `git` that only leaves a
marker if ever invoked, so "never consults git" is an assertion, not an
assumption. ExecMainExitTimestamp values are built from the REAL current
wall-clock time in systemd's own `Tue YYYY-MM-DD HH:MM:SS UTC` format, which
the script's real `date -d` parses natively — age math is exercised for real
without needing a faked `date`.

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

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "legibility" / "check_transcript_check_liveness.sh"


# ---------------------------------------------------------------------------
# Fake systemctl (canned `show -p ...` property lines from a JSON state file)
# ---------------------------------------------------------------------------

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing check_transcript_check_liveness.sh.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE, and answers `show -p FIELD1,FIELD2,...  <unit>`
(also accepts a repeated `-p FIELD` / `--property=FIELD`) from that
file's "fields" mapping -- a requested field absent from "fields" answers
as an empty string, mirroring real systemd's behavior for a unit that has
never run.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    if not args:
        return 1
    verb, rest = args[0], args[1:]

    state = _load()
    state.setdefault("calls", []).append(args)
    _save(state)

    if verb == "show":
        fields = None
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "-p":
                fields = rest[i + 1]
                i += 2
            elif tok.startswith("--property="):
                fields = tok.split("=", 1)[1]
                i += 1
            else:
                i += 1  # unit name or unrecognized flag

        current = state.get("fields", {})
        keys = fields.split(",") if fields else list(current.keys())
        for k in keys:
            print(f"{k}={current.get(k, '')}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

_FAKE_GIT_SRC = '''#!/usr/bin/env bash
# Fake `git` for testing check_transcript_check_liveness.sh -- like the
# trickle probe (decision 7) the liveness predicate must probe systemd UNIT
# STATE, never git history. Any invocation drops a marker file so the test
# can assert git was NEVER called.
set -euo pipefail
: > "$FAKE_GIT_CALLED_MARKER"
exit 0
'''


def _fake_bins(tmp_path, *, fields):
    """Write executable fake `systemctl` + `git` into <tmp_path>/bin/ and
    the systemctl fake's backing JSON state file (seeded with *fields*, the
    canned `show` property mapping). Returns (bin_dir, state_path,
    git_marker_path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    fake_systemctl = bin_dir / "systemctl"
    fake_systemctl.write_text(_FAKE_SYSTEMCTL_SRC)
    fake_systemctl.chmod(0o755)

    fake_git = bin_dir / "git"
    fake_git.write_text(_FAKE_GIT_SRC)
    fake_git.chmod(0o755)

    state_path = tmp_path / "systemctl_state.json"
    state_path.write_text(json.dumps({"calls": [], "fields": fields}))

    git_marker_path = tmp_path / "git_was_called"
    return bin_dir, state_path, git_marker_path


def _systemctl_calls(tmp_path):
    state_path = tmp_path / "systemctl_state.json"
    if not state_path.is_file():
        return []
    return json.loads(state_path.read_text())["calls"]


def _systemd_timestamp(dt):
    """Format a UTC datetime the way `systemctl show` emits a timestamp
    property (e.g. 'Tue 2026-07-14 04:00:12 UTC') -- the script's real
    `date -d` parses this natively, so age math is exercised for real
    without needing a faked `date`."""
    return dt.strftime('%a %Y-%m-%d %H:%M:%S UTC')


def _run_script(tmp_path, project_id, hours, *, fields):
    bin_dir, state_path, git_marker_path = _fake_bins(tmp_path, fields=fields)

    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    full_env["FAKE_GIT_CALLED_MARKER"] = str(git_marker_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), project_id, str(hours)],
        env=full_env, capture_output=True, text=True, timeout=30,
    )
    return result, git_marker_path


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
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecMainExitTimestamp": _systemd_timestamp(recent),
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _systemctl_calls(tmp_path)
    assert any(
        c[0] == "show" and "legibility-transcript-check@proj_a.service" in c for c in calls
    ), f"Expected a `systemctl show ... legibility-transcript-check@proj_a.service` probe; calls={calls!r}"

    assert not git_marker.exists(), "check_transcript_check_liveness.sh must never invoke git"


def test_alive_exit1_finding_within_window_exits_zero(tmp_path):
    """THE DIVERGENCE CASE (DD4): a run that exited 1 within the window is a
    lost-transcript FINDING (the detector fired and escalated) -- a healthy,
    expected run. The probe MUST treat it as ALIVE (exit 0), unlike the
    trickle probe which requires ExecMainStatus=0."""
    recent = datetime.now(UTC) - timedelta(hours=1)
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={
            "Result": "exit-code",
            "ExecMainStatus": "1",
            "ExecMainExitTimestamp": _systemd_timestamp(recent),
        },
    )

    assert result.returncode == 0, (
        f"A within-window exit-1 detector run (a lost-transcript finding) is "
        f"ALIVE and must exit 0 (DD4); stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not git_marker.exists(), "check_transcript_check_liveness.sh must never invoke git"


def test_older_than_window_exits_nonzero(tmp_path):
    stale = datetime.now(UTC) - timedelta(hours=48)
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecMainExitTimestamp": _systemd_timestamp(stale),
        },
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit for a run older than the window; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not git_marker.exists(), "check_transcript_check_liveness.sh must never invoke git"


def test_never_ran_exits_nonzero(tmp_path):
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={"Result": "", "ExecMainStatus": "", "ExecMainExitTimestamp": ""},
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when the service has never run; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not git_marker.exists(), "check_transcript_check_liveness.sh must never invoke git"


def test_unparseable_timestamp_exits_nonzero(tmp_path):
    """A non-empty but unparseable ExecMainExitTimestamp must FAIL loud, not be
    silently coerced into a benign (fresh) age. The never-ran guard only
    catches empty / "n/a"; a garbage value like 'not-a-date' clears the Result
    and ExecMainStatus checks and reaches the `date -d` parse, which must fail
    the probe rather than let GNU date treat it as parseable. Guards the
    parse-failure fail-loud branch (`if ! epoch="$(date -d ...)"`)."""
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecMainExitTimestamp": "not-a-date",
        },
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit for an unparseable ExecMainExitTimestamp "
        f"(the parse-failure branch must fail loud, not treat it as fresh); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not git_marker.exists(), "check_transcript_check_liveness.sh must never invoke git"


def test_abnormal_result_exits_nonzero(tmp_path):
    """A within-window run whose Result is abnormal (e.g. timeout / signal /
    core-dump) did NOT run to a clean exit, so it is NOT alive -- even with a
    fresh timestamp and an ExecMainStatus that looks benign."""
    recent = datetime.now(UTC) - timedelta(hours=1)
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={
            "Result": "timeout",
            "ExecMainStatus": "0",
            "ExecMainExitTimestamp": _systemd_timestamp(recent),
        },
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit for an abnormal Result (process did not run "
        f"to a clean exit); stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not git_marker.exists(), "check_transcript_check_liveness.sh must never invoke git"


def test_exit_status_two_exits_nonzero(tmp_path):
    """ExecMainStatus >= 2 (argparse error / malformed ExecStart) is NOT the
    detector's normal exit-1 alarm -- it means the timer is genuinely broken,
    so the probe must FAIL even though Result=exit-code and the run is
    fresh."""
    recent = datetime.now(UTC) - timedelta(hours=1)
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={
            "Result": "exit-code",
            "ExecMainStatus": "2",
            "ExecMainExitTimestamp": _systemd_timestamp(recent),
        },
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit for ExecMainStatus=2 (argparse / malformed "
        f"ExecStart); stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not git_marker.exists(), "check_transcript_check_liveness.sh must never invoke git"
