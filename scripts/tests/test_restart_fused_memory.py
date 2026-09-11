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
  - recon_busy_remaining>0 -> emit a BUSY /health body with
    `recon_busy_entry_count` in-flight cycles (decrement remaining). Entry 0
    always has run_id "run-xyz" so existing assertions keep working; a large
    entry_count drives recon_busy_check.py's output past 64KiB, the pipe
    buffer size that made `... | head -n1` race against SIGPIPE (task 3838).
  - else                   -> emit an IDLE /health body

AFTER restart  -> post-start health verify (`curl -sf URL`):
  - health_fail_remaining>0 -> exit 1 (not healthy yet; decrement)
  - else                    -> exit 0 (healthy), count the success
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_STATE_PATH"]

IDLE_BODY = '{"status":"ok","graphiti":true,"mem0":true,"recon_busy":[]}'


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def _busy_body(entry_count):
    entries = [
        {
            "project_id": "dark_factory",
            "run_id": "run-xyz" if i == 0 else f"run-{i}",
            "stage": "stage1_memory_consolidation",
            "started_at": "2026-07-18T06:00:00+00:00",
        }
        for i in range(entry_count)
    ]
    return json.dumps(
        {"status": "ok", "graphiti": True, "mem0": True, "recon_busy": entries}
    )


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
            entry_count = state.get("recon_busy_entry_count", 1)
            _save(state)
            sys.stdout.write(_busy_body(entry_count))
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
empty marker means "never emits" (drain-timeout path). When
`journalctl_marker_padding_bytes` is set, a large block of filler is
written AFTER the marker line so a still-reading real `grep -q` consumer
(which exits the instant it matches) would race SIGPIPE against this
process for the trailing bytes it never reads — task 3838 companion
regression for the drain-wait's journalctl|grep pipeline."""
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
    padding_bytes = state.get("journalctl_marker_padding_bytes", 0)
    call_count = len(state["journalctl_calls"])
    _save(state)
    if marker and call_count > after:
        out = marker + "\\n"
        if padding_bytes:
            line = "x" * 200 + "\\n"
            out += line * (padding_bytes // len(line) + 1)
        sys.stdout.write(out)
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
    recon_busy_entry_count=1,
    curl_unreachable=False,
    health_fail_remaining=0,
    journalctl_marker="",
    journalctl_marker_after=0,
    journalctl_marker_padding_bytes=0,
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
        "recon_busy_entry_count": recon_busy_entry_count,
        "curl_unreachable": curl_unreachable,
        "health_fail_remaining": health_fail_remaining,
        "journalctl_marker": journalctl_marker,
        "journalctl_marker_after": journalctl_marker_after,
        "journalctl_marker_padding_bytes": journalctl_marker_padding_bytes,
    }))
    return bin_dir, state_path


def _state(tmp_path):
    return json.loads(_state_path(tmp_path).read_text())


def _run_script(
    tmp_path,
    *args,
    env=None,
    recon_busy_remaining=0,
    recon_busy_entry_count=1,
    curl_unreachable=False,
    health_fail_remaining=0,
    journalctl_marker="",
    journalctl_marker_after=0,
    journalctl_marker_padding_bytes=0,
):
    bin_dir, state_path = _write_fakes(
        tmp_path,
        recon_busy_remaining=recon_busy_remaining,
        recon_busy_entry_count=recon_busy_entry_count,
        curl_unreachable=curl_unreachable,
        health_fail_remaining=health_fail_remaining,
        journalctl_marker=journalctl_marker,
        journalctl_marker_after=journalctl_marker_after,
        journalctl_marker_padding_bytes=journalctl_marker_padding_bytes,
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
# (c2) DEFAULT survives a large multi-line busy verdict without SIGPIPE
# (regression, task 3838)
# ---------------------------------------------------------------------------

def test_default_survives_large_busy_output_without_sigpipe(tmp_path):
    """With the old `verdict="$(printf '%s\\n' "$gate_output" | head -n1)"`
    pipeline, `head -n1` can close the pipe before `printf` finishes writing
    a large gate_output, taking SIGPIPE; under `set -euo pipefail` that
    would abort the whole script mid-gate, before the restart, with no
    diagnostic — a pipeline that can SIGPIPE at all has no business in a
    `set -euo pipefail` gate. A single busy poll with 2000
    `recon_busy_cycle` lines (~260KB, well past the 64KiB pipe buffer) makes
    the race deterministic instead of load-sensitive; this is a worst-case
    hardening drill, not a reproduction of the real gate_output size behind
    the observed task 3838 exit-141 incident, which remains unconfirmed
    (recon_busy_snapshot() emits one entry per concurrently in-flight full
    cycle — realistically 1-5, ~650 bytes — see task 3838 follow-up). The
    fix (`verdict="${gate_output%%$'\\n'*}"`) is pure bash and cannot
    SIGPIPE regardless of size."""
    result = _run_script(
        tmp_path,
        recon_busy_remaining=1,  # one large-busy poll, then idle
        recon_busy_entry_count=2000,
        env={"RECON_GATE_TIMEOUT": "100", "RECON_GATE_POLL_INTERVAL": "0"},
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    assert ["restart", UNIT] in state["systemctl_calls"], f"state={state!r}"
    assert "recon_busy_cycle" in result.stdout, f"stdout={result.stdout!r}"
    assert "run_id=run-xyz" in result.stdout, f"stdout={result.stdout!r}"


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
# (e2) --drain survives a large post-marker journal burst without SIGPIPE
# (regression, task 3838 amendment)
# ---------------------------------------------------------------------------

def test_drain_survives_large_journal_output_without_sigpipe(tmp_path):
    """With the old
    `journalctl ... | grep -q "$DRAIN_MARKER"` pipeline, `grep -q` exits
    the instant it matches, so a burst larger than the 64KiB pipe buffer
    queued behind the matching line (plausible during an active recon
    cycle — exactly when --drain is used) can take journalctl SIGPIPE
    after grep has already found the marker and exited 0. Under
    `pipefail`, bash's own rule ("the last, i.e. rightmost, command to
    exit non-zero") still surfaces journalctl's non-zero status even
    though the rightmost command (grep) succeeded, so the pipeline as a
    whole reads as failed. Because this pipeline is an `if` condition,
    `set -e` does not abort — the condition just silently evaluates
    false, so the old code would treat a marker it actually saw as
    not-seen and spin polling until DRAIN_TIMEOUT. A marker on the very
    first poll followed by 200KB of padding (well past the 64KiB pipe
    buffer) makes the race deterministic instead of load-sensitive. The
    fix (`journal="$(journalctl ... || true)"` then a pure-bash
    `[[ "$journal" == *"$DRAIN_MARKER"* ]]`) captures the whole journal
    via command substitution — bash reads to EOF rather than exiting
    early — so it cannot SIGPIPE."""
    result = _run_script(
        tmp_path,
        "--drain",
        journalctl_marker=DRAIN_MARKER,
        journalctl_marker_after=0,  # marker appears on the very first poll
        journalctl_marker_padding_bytes=200_000,  # >> 64KiB pipe buffer
        env={"DRAIN_TIMEOUT": "10", "DRAIN_POLL_INTERVAL": "0"},
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    # The marker was present on the FIRST poll: the drain wait must detect
    # it right away rather than (falsely, under the old SIGPIPE-prone
    # pipeline) missing it and spinning toward DRAIN_TIMEOUT.
    assert len(state["journalctl_calls"]) == 1, (
        f"expected the drain wait to stop as soon as the marker was seen "
        f"on the first poll, not spin looking for it again; state={state!r}"
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
