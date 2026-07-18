"""Tests for scripts/restart-fused-memory.sh — the cycle-aware restart
(task 2703 δ). Drives the script via subprocess against fake
`systemctl`/`curl`/`journalctl` shimmed onto PATH, recording every
invocation into a shared JSON state file so tests can assert the recon-busy
defer gate, the --drain SIGUSR1 signal + drained-marker wait, and the
post-start /health verification — never touching a live systemd or
fused-memory. Template: test_deploy_w5_recon_reliability.py.

The test owns BOTH sides of the /health-body contract (the fake curl serves
a busy/idle body it controls) and the journalctl drained-marker contract
(the fake journalctl emits 'Harness fully drained' after K polls), so the
suite drives the REAL recon_busy_check.py gate logic with no HTTP server.
Short RECON_GATE_TIMEOUT/RECON_GATE_POLL_INTERVAL/DRAIN_TIMEOUT/HEALTH_TIMEOUT
are env-injected so the cap/timeout paths exercise in seconds, not 35 min.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "restart-fused-memory.sh"
UNIT = "fused-memory"

# The drained marker the script greps journalctl for (task 2702's converging
# drain). The fake journalctl emits this exact literal, so the test pins the
# contract with the harness's real marker.
DRAIN_MARKER = "Harness fully drained"


_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake systemctl: record every call (minus --user); snapshot the curl-call
count at the (single) restart so a test can prove the recon gate polled
before the restart. Always succeeds."""
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
    call = [a for a in argv[1:] if a != "--user"]
    state.setdefault("systemctl_calls", []).append(call)
    if call and call[0] == "restart" and "curl_calls_at_restart" not in state:
        state["curl_calls_at_restart"] = len(state.get("curl_calls", []))
    _save(state)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

_FAKE_CURL_SRC = '''#!/usr/bin/env python3
"""Fake curl. Two phases, discriminated by whether a restart has happened:

BEFORE restart -> recon-gate body fetch (`curl -s --max-time N URL`):
  - curl_unreachable True  -> emit nothing, exit 1 (endpoint unreachable)
  - recon_busy_remaining>0 -> emit a BUSY /health body (decrement)
  - else                   -> emit an IDLE /health body

AFTER restart  -> post-start health verify (`curl -sf URL`):
  - health_fail_remaining>0 -> exit 1 (not healthy yet; decrement)
  - else                    -> exit 0 (healthy), count the success
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_STATE_PATH"]

BUSY_BODY = (
    '{"status":"ok","graphiti":true,"mem0":true,"recon_busy":'
    '[{"project_id":"dark_factory","run_id":"run-xyz",'
    '"stage":"stage1_memory_consolidation",'
    '"started_at":"2026-07-18T06:00:00+00:00"}]}'
)
IDLE_BODY = '{"status":"ok","graphiti":true,"mem0":true,"recon_busy":[]}'


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    state = _load()
    state.setdefault("curl_calls", []).append(argv[1:])
    restart_done = any(
        c and c[0] == "restart" for c in state.get("systemctl_calls", [])
    )
    if not restart_done:
        if state.get("curl_unreachable"):
            _save(state)
            return 1
        remaining = state.get("recon_busy_remaining", 0)
        if remaining > 0:
            state["recon_busy_remaining"] = remaining - 1
            _save(state)
            sys.stdout.write(BUSY_BODY)
            return 0
        _save(state)
        sys.stdout.write(IDLE_BODY)
        return 0
    hfr = state.get("health_fail_remaining", 0)
    if hfr > 0:
        state["health_fail_remaining"] = hfr - 1
        _save(state)
        return 1
    state["health_success_count"] = state.get("health_success_count", 0) + 1
    _save(state)
    sys.stdout.write(IDLE_BODY)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

_FAKE_JOURNALCTL_SRC = '''#!/usr/bin/env python3
"""Fake journalctl for the --drain drained-marker wait. Records every call;
emits the shared `journalctl_marker` string once the call count exceeds
`journalctl_marker_after` (so the wait must poll a few times first). An
empty marker means "never emits" (drain-timeout path)."""
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
    state.setdefault("journalctl_calls", []).append(argv[1:])
    marker = state.get("journalctl_marker", "")
    after = state.get("journalctl_marker_after", 0)
    _save(state)
    if marker and len(state["journalctl_calls"]) > after:
        print(marker)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _state_path(tmp_path):
    return tmp_path / "fake_state.json"


def _write_fakes(
    tmp_path,
    *,
    recon_busy_remaining=0,
    curl_unreachable=False,
    health_fail_remaining=0,
    journalctl_marker="",
    journalctl_marker_after=0,
):
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
        "recon_busy_remaining": recon_busy_remaining,
        "curl_unreachable": curl_unreachable,
        "health_fail_remaining": health_fail_remaining,
        "journalctl_marker": journalctl_marker,
        "journalctl_marker_after": journalctl_marker_after,
    }))
    return bin_dir, state_path


def _state(tmp_path):
    return json.loads(_state_path(tmp_path).read_text())


def _run_script(
    tmp_path,
    *args,
    env=None,
    recon_busy_remaining=0,
    curl_unreachable=False,
    health_fail_remaining=0,
    journalctl_marker="",
    journalctl_marker_after=0,
):
    bin_dir, state_path = _write_fakes(
        tmp_path,
        recon_busy_remaining=recon_busy_remaining,
        curl_unreachable=curl_unreachable,
        health_fail_remaining=health_fail_remaining,
        journalctl_marker=journalctl_marker,
        journalctl_marker_after=journalctl_marker_after,
    )
    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_STATE_PATH"] = str(state_path)
    # Short, bounded defaults; individual tests override as needed.
    full_env.setdefault("RECON_GATE_TIMEOUT", "100")
    full_env.setdefault("RECON_GATE_POLL_INTERVAL", "0")
    full_env.setdefault("DRAIN_TIMEOUT", "10")
    full_env.setdefault("DRAIN_POLL_INTERVAL", "0")
    full_env.setdefault("HEALTH_TIMEOUT", "3")
    full_env.setdefault("CURL_MAX_TIME", "2")
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
# (a) --now bypasses the recon gate
# ---------------------------------------------------------------------------

def test_now_bypasses_gate_and_restarts(tmp_path):
    result = _run_script(tmp_path, "--now", recon_busy_remaining=5)
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    assert ["restart", UNIT] in state["systemctl_calls"], f"state={state!r}"
    # Gate skipped entirely: no gate polling before the restart, no defer output.
    assert state.get("curl_calls_at_restart", 0) == 0, f"state={state!r}"
    assert "recon_gate:" not in result.stdout, (
        f"--now must not run the recon gate; stdout={result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# (b) DEFAULT defers while busy, then proceeds once idle
# ---------------------------------------------------------------------------

def test_default_defers_while_busy_then_proceeds(tmp_path):
    result = _run_script(
        tmp_path,
        recon_busy_remaining=2,  # busy for 2 polls, idle on the 3rd
        env={"RECON_GATE_TIMEOUT": "100", "RECON_GATE_POLL_INTERVAL": "0"},
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    # The restart happened only AFTER the gate cleared (>=3 gate polls first).
    assert state.get("curl_calls_at_restart", 0) >= 3, f"state={state!r}"
    assert ["restart", UNIT] in state["systemctl_calls"], f"state={state!r}"
    # Structured defer output: recon_gate: lines with elapsed/cap + per-cycle detail.
    assert "recon_gate:" in result.stdout, f"stdout={result.stdout!r}"
    assert "deferring" in result.stdout, f"stdout={result.stdout!r}"
    assert "recon_busy_cycle" in result.stdout, (
        f"expected per-cycle detail from recon_busy_check; stdout={result.stdout!r}"
    )
    assert "run_id=run-xyz" in result.stdout, f"stdout={result.stdout!r}"


# ---------------------------------------------------------------------------
# (c) DEFAULT proceeds once the cap elapses (busy forever)
# ---------------------------------------------------------------------------

def test_default_proceeds_after_cap(tmp_path):
    result = _run_script(
        tmp_path,
        recon_busy_remaining=100_000,  # never goes idle
        env={"RECON_GATE_TIMEOUT": "1", "RECON_GATE_POLL_INTERVAL": "0.3"},
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "cap reached" in result.stdout.lower(), (
        f"expected a structured cap-reached line; stdout={result.stdout!r}"
    )
    state = _state(tmp_path)
    assert ["restart", UNIT] in state["systemctl_calls"], f"state={state!r}"


# ---------------------------------------------------------------------------
# (d) DEFAULT proceeds fail-safe when /health is unreachable
# ---------------------------------------------------------------------------

def test_default_proceeds_when_unreachable(tmp_path):
    result = _run_script(tmp_path, curl_unreachable=True)
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "unreachable" in result.stdout.lower(), (
        f"expected the gate to report unreachable and proceed; "
        f"stdout={result.stdout!r}"
    )
    state = _state(tmp_path)
    assert ["restart", UNIT] in state["systemctl_calls"], f"state={state!r}"


# ---------------------------------------------------------------------------
# (e) --drain signals SIGUSR1 via systemctl kill + waits for the drained marker
# ---------------------------------------------------------------------------

def test_drain_signals_sigusr1_and_waits_for_marker(tmp_path):
    result = _run_script(
        tmp_path,
        "--drain",
        journalctl_marker=DRAIN_MARKER,
        journalctl_marker_after=2,  # marker appears on the 3rd poll
        env={"DRAIN_TIMEOUT": "10", "DRAIN_POLL_INTERVAL": "0"},
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    assert ["kill", "--kill-who=main", "--signal=SIGUSR1", UNIT] in state["systemctl_calls"], (
        f"expected a `systemctl --user kill --kill-who=main --signal=SIGUSR1 "
        f"{UNIT}` call; state={state!r}"
    )
    assert len(state["journalctl_calls"]) >= 3, (
        f"expected the drained-marker wait to poll journalctl until the marker "
        f"appears; state={state!r}"
    )
    assert DRAIN_MARKER.lower() in result.stdout.lower(), f"stdout={result.stdout!r}"
    assert ["restart", UNIT] in state["systemctl_calls"], f"state={state!r}"


# ---------------------------------------------------------------------------
# (f) post-start /health never healthy -> exit 1 (retained verify behaviour)
# ---------------------------------------------------------------------------

def test_exits_nonzero_when_health_never_ready(tmp_path):
    result = _run_script(
        tmp_path,
        recon_busy_remaining=0,        # gate clears immediately
        health_fail_remaining=100_000,  # health verify never passes
        env={"HEALTH_TIMEOUT": "1"},
    )
    assert result.returncode != 0, (
        f"expected non-zero exit when /health never becomes healthy; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stderr.strip(), f"expected a diagnostic on stderr; got {result.stderr!r}"
    state = _state(tmp_path)
    assert ["restart", UNIT] in state["systemctl_calls"], (
        f"the restart must still have occurred; the failure is the verify gate; "
        f"state={state!r}"
    )
