"""Tests for scripts/legibility/check_trickle_liveness.sh — the real
liveness predicate that replaces the fail-loud stub committed with
plans/confusion-reduction-prd.md (PRD task epsilon, decision 7: quiet
nights commit nothing, so liveness must probe systemd UNIT STATE, never
git history).

Drives the script via subprocess with a FAKE `systemctl` shimmed onto PATH
(answers `show -p ...` property lines from a JSON state file — mirrors
scripts/tests/test_deploy_w5_recon_reliability.py /
test_deploy_w11_lane_lifecycle.py and restart-orchestrator.sh's own
read_field/get_state `-p "$FIELDS"` convention) and a FAKE `git` that only
leaves a marker if ever invoked, so "never consults git" is an assertion,
not an assumption. ExecMainExitTimestamp values are built from the REAL
current wall-clock time in systemd's own `Tue YYYY-MM-DD HH:MM:SS UTC`
format, which the script's real `date -d` parses natively — age math is
exercised for real without needing a faked `date`.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "legibility" / "check_trickle_liveness.sh"


# ---------------------------------------------------------------------------
# Fake systemctl (canned `show -p ...` property lines from a JSON state file)
# ---------------------------------------------------------------------------

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing check_trickle_liveness.sh.

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
# Fake `git` for testing check_trickle_liveness.sh -- decision 7 forbids
# consulting git for liveness (quiet nights commit nothing, so a
# git-history probe would false-alarm on a healthy but quiet timer). Any
# invocation drops a marker file so the test can assert git was NEVER
# called.
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
    property (e.g. 'Tue 2026-07-14 03:00:12 UTC') -- the script's real
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
# step-27: RED -- check_trickle_liveness.sh
# ---------------------------------------------------------------------------

def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_alive_within_window_exits_zero(tmp_path):
    recent = datetime.now(timezone.utc) - timedelta(hours=2)
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
        c[0] == "show" and "legibility-trickle@proj_a.service" in c for c in calls
    ), f"Expected a `systemctl show ... legibility-trickle@proj_a.service` probe; calls={calls!r}"

    assert not git_marker.exists(), "check_trickle_liveness.sh must never invoke git (decision 7)"


def test_older_than_window_exits_nonzero(tmp_path):
    stale = datetime.now(timezone.utc) - timedelta(hours=48)
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
    assert not git_marker.exists(), "check_trickle_liveness.sh must never invoke git (decision 7)"


def test_failed_run_exits_nonzero(tmp_path):
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={
            "Result": "exit-code",
            "ExecMainStatus": "1",
            "ExecMainExitTimestamp": _systemd_timestamp(recent),
        },
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit for a failed last run; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not git_marker.exists(), "check_trickle_liveness.sh must never invoke git (decision 7)"


def test_never_ran_exits_nonzero(tmp_path):
    result, git_marker = _run_script(
        tmp_path, "proj_a", 24,
        fields={"Result": "", "ExecMainStatus": "", "ExecMainExitTimestamp": ""},
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when the service has never run; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not git_marker.exists(), "check_trickle_liveness.sh must never invoke git (decision 7)"
