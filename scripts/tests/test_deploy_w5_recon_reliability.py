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

# Trailing bytes the fake journalctl writes AFTER the marker, to provoke the
# SIGPIPE half of the `| grep -q` misread.
#
# THE MEASUREMENTS AND THE "DO NOT LOWER THIS" ARGUMENT LIVE IN ONE PLACE:
# tests/scripts/shell_sections.py::SIGPIPE_BULK_BYTES. This is a duplicated
# VALUE, deliberately not a duplicated RATIONALE — restating the trial counts
# here is how three copies of the same number drift into three different
# numbers, and a future measurement that moves the threshold should have to
# edit exactly one comment.
#
# It is a duplicated value only because scripts/tests/ cannot import a
# tests/scripts/ helper: scripts/tests/conftest.py puts scripts/,
# scripts/legibility/ and scripts/local-model-serving/ on sys.path and NOT
# tests/scripts/, and widening that conftest to make one constant importable is
# a directory-boundary change well outside what this constant is worth. Should
# the boundary ever open, this becomes an import.
BULK_BYTES = 262144


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

Two further knobs simulate the `producer | grep -q` misread the
recon-serving gate is subject to, both defaulting to today's behaviour:

  `journalctl_exit_code` (default 0) -- the status returned AFTER the
  marker has been printed, i.e. a journalctl that showed the marker and
  then failed for its own reasons. `pipefail` hands that status to the
  script's `if`, losing a marker the journal plainly carried.

  `journalctl_bulk_bytes` (default 0) -- that many bytes written after the
  marker, then an explicit flush. `grep -q` exits on its first match and
  closes the read end, so the producer dies mid-write.

NOTE ON MECHANISM for the bulk case: this fake is PYTHON, which ignores
SIGPIPE and raises BrokenPipeError instead, so the interpreter exits
non-zero (120) on the failed shutdown flush rather than dying of signal
13. Either way the producer's status is non-zero and `pipefail` hands it
to the `if` -- which IS the defect under test. Do NOT "fix" this by
catching BrokenPipeError: that would hide the very status the test needs.
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
    bulk = state.get("journalctl_bulk_bytes", 0)
    _save(state)
    if marker:
        print(marker)
    if bulk:
        sys.stdout.write("x" * bulk)
        sys.stdout.flush()
    return state.get("journalctl_exit_code", 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _state_path(tmp_path):
    """The fakes' shared JSON state file for a given pytest `tmp_path` --
    doubles as the invocation marker the `_state`/`_systemctl_calls` readers
    inspect. Single derivation point so writer and reader can never drift
    apart."""
    return tmp_path / "fake_state.json"


def _write_fakes(tmp_path, *, curl_fail_remaining=0, journalctl_marker="",
                 journalctl_exit_code=0, journalctl_bulk_bytes=0):
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
        "journalctl_exit_code": journalctl_exit_code,
        "journalctl_bulk_bytes": journalctl_bulk_bytes,
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

def _run_script(tmp_path, *args, env=None, curl_fail_remaining=0,
                journalctl_marker="", journalctl_exit_code=0,
                journalctl_bulk_bytes=0):
    """Run deploy-w5-recon-reliability.sh via subprocess with fresh fake
    `systemctl`/`curl`/`journalctl` on PATH (state reset each call; see
    `_write_fakes`) so the real script never touches a live systemd,
    fused-memory /health, or journal.

    HEALTH_TIMEOUT/RECON_VERIFY_TIMEOUT default to a short 3s -- irrelevant
    to a run that succeeds on its first poll, but keeps a regression bounded
    instead of hanging up to the production 30s/180s defaults.
    RECON_MARKER is always set to the shared `RECON_MARKER` test constant,
    which `journalctl_marker` (when set) tells the fake journalctl to emit.

    `journalctl_exit_code` / `journalctl_bulk_bytes` both default to today's
    behaviour, so every pre-existing test is unaffected; see
    `_FAKE_JOURNALCTL_SRC` for what each simulates.
    """
    bin_dir, state_path = _write_fakes(
        tmp_path,
        curl_fail_remaining=curl_fail_remaining,
        journalctl_marker=journalctl_marker,
        journalctl_exit_code=journalctl_exit_code,
        journalctl_bulk_bytes=journalctl_bulk_bytes,
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


# ---------------------------------------------------------------------------
# step-3: RED -- apply restarts the correct unit BEFORE verifying, and
# rejects unknown args without restarting
# ---------------------------------------------------------------------------

def test_apply_restarts_fused_memory_then_verifies(tmp_path):
    """A full apply run (no flags), with fake curl answering /health 200 and
    fake journalctl emitting the RECON_MARKER, must exit 0 and record a
    `systemctl --user restart fused-memory.service` call. The ordering
    witnesses prove verify runs strictly AFTER restart -- not merely that
    both happened somewhere in the run."""
    result = _run_script(tmp_path, curl_fail_remaining=0, journalctl_marker=RECON_MARKER)

    assert result.returncode == 0, (
        f"Expected apply to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    state = _state(tmp_path)
    assert ["restart", UNIT] in state["systemctl_calls"], (
        f"Expected a `systemctl --user restart {UNIT}` call; state={state!r}"
    )
    assert state.get("restart_called_before_first_curl") is True, (
        f"Expected the restart to already have run by the first curl call "
        f"(health verify strictly after restart); state={state!r}"
    )
    assert state.get("restart_called_before_first_journalctl") is True, (
        f"Expected the restart to already have run by the first journalctl "
        f"call (serving-sanity verify strictly after restart); state={state!r}"
    )


def test_unknown_argument_is_rejected_without_restarting(tmp_path):
    """An unrecognized argument (e.g. --bogus) exits non-zero with a stderr
    message and performs no restart or verify call at all."""
    result = _run_script(tmp_path, "--bogus")

    assert result.returncode != 0, (
        f"Expected --bogus to be rejected with a non-zero exit; got 0\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stderr.strip(), (
        f"Expected a non-empty stderr message for the rejected argument; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    state = _state(tmp_path)
    assert state["systemctl_calls"] == [], f"state={state!r}"
    assert state["curl_calls"] == [], f"state={state!r}"
    assert state["journalctl_calls"] == [], f"state={state!r}"


# ---------------------------------------------------------------------------
# step-5: RED -- post-restart health gate (both directions)
# ---------------------------------------------------------------------------

def test_apply_waits_for_health(tmp_path):
    """When fused-memory's /health is not yet ready, the script must keep
    polling `$HEALTH_URL` (rather than giving up after one failed attempt)
    and exit 0 once it reports healthy -- and the health poll(s) must have
    started only after the restart."""
    result = _run_script(
        tmp_path,
        curl_fail_remaining=2,
        journalctl_marker=RECON_MARKER,
        env={"HEALTH_TIMEOUT": "10"},
    )

    assert result.returncode == 0, (
        f"Expected apply to eventually exit 0 once healthy; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    assert len(state["curl_calls"]) >= 3, (
        f"Expected the script to retry curl past the first 2 failures "
        f"(3+ calls total); state={state!r}"
    )
    assert state.get("restart_called_before_first_curl") is True, (
        f"Expected polling to have started only after the restart; state={state!r}"
    )


def test_apply_fails_when_health_never_ready(tmp_path):
    """When /health never becomes ready, with a short injected
    HEALTH_TIMEOUT the script must exit non-zero with a diagnostic on
    stderr -- and the restart must still have occurred (the failure belongs
    to the verify gate, not the restart)."""
    result = _run_script(
        tmp_path,
        curl_fail_remaining=10_000,
        env={"HEALTH_TIMEOUT": "1"},
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when health never becomes ready; got 0\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stderr.strip(), (
        f"Expected a diagnostic on stderr; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    assert ["restart", UNIT] in state["systemctl_calls"], (
        f"Expected the restart to still have occurred despite the health "
        f"gate failing; state={state!r}"
    )
    assert state["journalctl_calls"] == [], (
        f"Expected the recon-serving gate to never run when the health gate "
        f"fails first; state={state!r}"
    )


def test_apply_confirms_recon_serving(tmp_path):
    """When fused-memory's journal shows the RECON_MARKER after a restart,
    the serving-sanity gate observes it and the run exits 0 -- and the
    recon-serving check must run only after the health gate has already
    passed at least once (restart -> health -> recon-serving ordering)."""
    result = _run_script(tmp_path, curl_fail_remaining=0, journalctl_marker=RECON_MARKER)

    assert result.returncode == 0, (
        f"Expected apply to exit 0 once recon-serving is confirmed; got "
        f"{result.returncode}\nstdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    assert state["journalctl_calls"], (
        f"Expected at least one journalctl call observing the recon-serving "
        f"marker; state={state!r}"
    )
    assert state.get("health_passed_before_first_journalctl") is True, (
        f"Expected the recon-serving check to run only after the health "
        f"gate had already passed; state={state!r}"
    )
    assert state.get("restart_called_before_first_journalctl") is True, (
        f"Expected the recon-serving check to run only after the restart; "
        f"state={state!r}"
    )


# --- the `producer | grep -q` misread at the recon-serving gate -------------
# `journalctl ... | grep -q "$RECON_MARKER"` reports the PRODUCER's status
# under `set -o pipefail`, not grep's verdict, so the gate can time out on a
# journal that plainly CARRIED the marker -- failing a deploy whose service is
# in fact serving reconciliation. Driven through the WHOLE shipped script, so
# the restart -> health -> recon-serving ordering witnesses still apply.
#
# `test_apply_fails_when_recon_marker_absent` above is already this site's
# anti-regression guard: it pins that a journal which never emits the marker
# STILL times out non-zero, so a fix cannot buy these two tests by making the
# gate unconditionally pass.

_MARKER_TIMEOUT_ERROR = "did not show ledger-backed recon serving"


def test_apply_confirms_recon_serving_when_journalctl_exits_nonzero_after_the_marker(
    tmp_path,
):
    """A journal that SHOWED the marker proves recon is serving, whatever journalctl's own status was.

    `journalctl` reports on its own invocation; the marker having appeared is a
    fact about the OUTPUT. Conflating the two fails a healthy deploy.
    """
    result = _run_script(
        tmp_path,
        curl_fail_remaining=0,
        journalctl_marker=RECON_MARKER,
        journalctl_exit_code=1,
    )

    assert result.returncode == 0, (
        f"Expected apply to exit 0: the journal DID show the recon-serving "
        f"marker, and journalctl's own exit status is a different question; "
        f"got {result.returncode}\nstdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert _MARKER_TIMEOUT_ERROR not in result.stderr, (
        f"Expected no marker-timeout diagnostic for a journal that carried "
        f"the marker; stderr={result.stderr!r}"
    )


def test_apply_confirms_recon_serving_when_journalctl_is_sigpiped_after_the_marker(
    tmp_path,
):
    """A producer still writing when grep matches dies mid-write; the marker was still shown.

    `grep -q` exits on its first match and closes the read end, so a journalctl
    still streaming fails its next write and `pipefail` turns that into "marker
    never seen" on a journal that plainly carried it.
    """
    result = _run_script(
        tmp_path,
        curl_fail_remaining=0,
        journalctl_marker=RECON_MARKER,
        journalctl_bulk_bytes=BULK_BYTES,
    )

    assert result.returncode == 0, (
        f"Expected apply to exit 0: the journal DID show the recon-serving "
        f"marker before the read end was closed; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert _MARKER_TIMEOUT_ERROR not in result.stderr, (
        f"Expected no marker-timeout diagnostic for a journal that carried "
        f"the marker; stderr={result.stderr!r}"
    )


def test_apply_fails_when_recon_marker_absent(tmp_path):
    """When the journal never shows the RECON_MARKER, with a short injected
    RECON_VERIFY_TIMEOUT the script must exit non-zero with a diagnostic on
    stderr (so the DeterministicRunner escalates) -- while the restart and
    health gate must still have occurred; the failure belongs to the
    recon-serving gate alone."""
    result = _run_script(
        tmp_path,
        curl_fail_remaining=0,
        journalctl_marker="",
        env={"RECON_VERIFY_TIMEOUT": "1"},
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when the recon-serving marker never "
        f"appears; got 0\nstdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stderr.strip(), (
        f"Expected a diagnostic on stderr; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    state = _state(tmp_path)
    assert ["restart", UNIT] in state["systemctl_calls"], (
        f"Expected the restart to still have occurred despite the "
        f"recon-serving gate failing; state={state!r}"
    )
    assert state.get("curl_success_count", 0) > 0, (
        f"Expected the health gate to still have passed despite the "
        f"recon-serving gate failing; state={state!r}"
    )
    assert state["journalctl_calls"], (
        f"Expected the recon-serving gate to have polled at least once "
        f"before timing out; state={state!r}"
    )
