"""Tests for deploy-w5-recon-reliability.sh — drives the script via
subprocess against fake `systemctl`, `curl`, and `journalctl` shimmed onto
PATH (extends test_deploy_w11_lane_lifecycle.py's fake-systemctl-on-PATH
design to the trio this script needs), recording every invocation into a
shared JSON state file so tests can assert the restart unit, the
restart -> health -> recon-serving ordering, and exit codes — never
touching a live systemd/fused-memory.

HEALTH_TIMEOUT/RECON_VERIFY_TIMEOUT are env-injected short (see
`_run_script`) so failure-path tests exit fast instead of hanging on the
production 30s/180s defaults.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "deploy-w5-recon-reliability.sh"
UNIT = "fused-memory.service"

# The test controls BOTH sides of the marker contract (this constant is
# handed to the script via the RECON_MARKER env var AND to the fake
# journalctl via the shared state file's "journalctl_marker" field), so the
# suite is agnostic to the real production marker string chosen in the
# script header.
RECON_MARKER = "test-recon-marker-xyz"


# ---------------------------------------------------------------------------
# Fake systemctl / curl / journalctl (shared-JSON-state-file + canned
# responses, mirroring test_deploy_w11_lane_lifecycle.py's _fake_systemctl)
# ---------------------------------------------------------------------------

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing deploy-w5-recon-reliability.sh. Records
every invocation (minus `--user`) into the shared JSON state file; always
"succeeds" -- a real restart failure is out of this script's control (the
DeterministicRunner's own baseline/fresh-PID verify covers that scenario
independently, per deterministic_runner.py); this script only needs to
prove IT invoked the restart, on the right unit, before verifying.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_STATE_PATH"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    state = _load()
    state.setdefault("systemctl_calls", []).append(
        [a for a in argv[1:] if a != "--user"]
    )
    _save(state)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

_FAKE_CURL_SRC = '''#!/usr/bin/env python3
"""Fake `curl` for testing deploy-w5-recon-reliability.sh's health gate.
Records every invocation into the shared JSON state file, plus an ordering
witness (`restart_called_before_first_curl`) snapshotted at the FIRST curl
invocation only -- proving the restart already ran before verify started,
not merely that both happened somewhere in the run.

Fails exactly `curl_fail_remaining` times (decrementing per call), then
succeeds on every call after -- lets a test simulate "healthy after K
polls" (curl_fail_remaining=K) or "never healthy" (a fail count that
outlasts the test's bounded timeout).
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_STATE_PATH"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    state = _load()
    if state.get("restart_called_before_first_curl") is None:
        state["restart_called_before_first_curl"] = bool(state.get("systemctl_calls"))
    state.setdefault("curl_calls", []).append(argv[1:])
    remaining = state.get("curl_fail_remaining", 0)
    if remaining > 0:
        state["curl_fail_remaining"] = remaining - 1
        _save(state)
        return 1
    state["curl_success_count"] = state.get("curl_success_count", 0) + 1
    _save(state)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

_FAKE_JOURNALCTL_SRC = '''#!/usr/bin/env python3
"""Fake `journalctl` for testing deploy-w5-recon-reliability.sh's
recon-serving gate. Records every invocation into the shared JSON state
file, plus two ordering witnesses snapshotted at the FIRST invocation only:
`restart_called_before_first_journalctl` (restart already ran) and
`health_passed_before_first_journalctl` (the health gate already succeeded
at least once) -- proving the gate ordering restart -> health ->
recon-serving, not merely that all three happened somewhere in the run.

Prints the shared state's `journalctl_marker` string to stdout on every
call when it is non-empty -- this fake owns BOTH sides of the marker
contract together with `_run_script` (which sets the script's real
RECON_MARKER env var to the same value), so the test is agnostic to the
real production marker string. An empty `journalctl_marker` means "never
emits" (the marker-absent/timeout case).
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_STATE_PATH"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    state = _load()
    if state.get("restart_called_before_first_journalctl") is None:
        state["restart_called_before_first_journalctl"] = bool(state.get("systemctl_calls"))
    if state.get("health_passed_before_first_journalctl") is None:
        state["health_passed_before_first_journalctl"] = state.get("curl_success_count", 0) > 0
    state.setdefault("journalctl_calls", []).append(argv[1:])
    marker = state.get("journalctl_marker", "")
    _save(state)
    if marker:
        print(marker)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _state_path(tmp_path):
    """The fakes' shared JSON state file for a given pytest `tmp_path` --
    doubles as the invocation marker the `_state`/`_systemctl_calls` readers
    inspect. Single derivation point so writer and reader can never drift
    apart."""
    return tmp_path / "fake_state.json"


def _write_fakes(tmp_path, *, curl_fail_remaining=0, journalctl_marker=""):
    """Write executable fake `systemctl`/`curl`/`journalctl` into
    <tmp_path>/bin/ plus their shared JSON state file.

    Returns (bin_dir, state_path).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, src in (
        ("systemctl", _FAKE_SYSTEMCTL_SRC),
        ("curl", _FAKE_CURL_SRC),
        ("journalctl", _FAKE_JOURNALCTL_SRC),
    ):
        fake = bin_dir / name
        fake.write_text(src)
        fake.chmod(0o755)

    state_path = _state_path(tmp_path)
    state_path.write_text(json.dumps({
        "systemctl_calls": [],
        "curl_calls": [],
        "journalctl_calls": [],
        "curl_fail_remaining": curl_fail_remaining,
        "journalctl_marker": journalctl_marker,
    }))
    return bin_dir, state_path


def _state(tmp_path):
    """Read back the fakes' shared JSON state file -- the full call log plus
    any ordering witnesses recorded so far."""
    return json.loads(_state_path(tmp_path).read_text())


def _systemctl_calls(tmp_path):
    return _state(tmp_path)["systemctl_calls"]


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_script(tmp_path, *args, env=None, curl_fail_remaining=0, journalctl_marker=""):
    """Run deploy-w5-recon-reliability.sh via subprocess with fresh fake
    `systemctl`/`curl`/`journalctl` on PATH (state reset each call; see
    `_write_fakes`) so the real script never touches a live systemd,
    fused-memory /health, or journal.

    HEALTH_TIMEOUT/RECON_VERIFY_TIMEOUT default to a short 3s -- irrelevant
    to a run that succeeds on its first poll, but keeps a regression bounded
    instead of hanging up to the production 30s/180s defaults.
    RECON_MARKER is always set to the shared `RECON_MARKER` test constant,
    which `journalctl_marker` (when set) tells the fake journalctl to emit.
    """
    bin_dir, state_path = _write_fakes(
        tmp_path,
        curl_fail_remaining=curl_fail_remaining,
        journalctl_marker=journalctl_marker,
    )

    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_STATE_PATH"] = str(state_path)
    full_env.setdefault("HEALTH_TIMEOUT", "3")
    full_env.setdefault("RECON_VERIFY_TIMEOUT", "3")
    full_env["RECON_MARKER"] = RECON_MARKER
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# step-1: RED -- executable bit + --check/--dry-run no-op smoke
# ---------------------------------------------------------------------------

def test_script_is_executable():
    """The working-tree script must carry the executable bit (mode 100755)
    -- pins the os.X_OK requirement enforced by deterministic_task_guard at
    submit_task time for before_done.script (CLAUDE.md "Deterministic task
    kind": before_done.script "must exist & be executable")."""
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SCRIPT}"
    )


@pytest.mark.parametrize("flag", ["--check", "--dry-run"])
def test_check_mode_is_noop(tmp_path, flag):
    """`--check`/`--dry-run` must exit 0, print an intended-actions line
    naming `fused-memory.service` plus a verify mention, and invoke NONE of
    systemctl/curl/journalctl -- check-mode must short-circuit before any
    side effect."""
    result = _run_script(tmp_path, flag)

    assert result.returncode == 0, (
        f"Expected {flag} to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert UNIT in result.stdout, (
        f"Expected an intended-actions line naming {UNIT!r} in stdout; "
        f"got: {result.stdout!r}"
    )
    assert "verify" in result.stdout.lower(), (
        f"Expected a verify mention in stdout; got: {result.stdout!r}"
    )

    state = _state(tmp_path)
    assert state["systemctl_calls"] == [], f"state={state!r}"
    assert state["curl_calls"] == [], f"state={state!r}"
    assert state["journalctl_calls"] == [], f"state={state!r}"
