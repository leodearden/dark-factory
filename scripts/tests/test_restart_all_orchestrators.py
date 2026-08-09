"""Tests for restart-all-orchestrators.sh's `--drain` merge-drain gate
(task 2397, γ of the orchestrator fleet-redeploy PRD).

Drives the script via subprocess against a fake `systemctl` shimmed onto
PATH -- adapted from test_restart_orchestrator.py's single-unit harness,
EXTENDED with `list-units` support and per-unit state so the gate's
per-unit poll loop can be exercised offline. drain_check.py itself is
NOT mocked -- it runs for real against heartbeat JSON files the tests
write directly into a tmp fleet dir (ORCH_FLEET_DIR).
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# The repo-wide deploy-clock guard's OWN snapshot/compare (task 3797), reused
# rather than re-implemented here: two copies of the same "(bytes, mtime_ns) or
# absent" comparison would drift the moment the guard tightens what it looks at.
# Importable because conftest.py appends REPO_ROOT to sys.path for this
# directory. Functions only -- importing one of that module's FIXTURES into a
# test module would bind a module-scoped copy that shadows the conftest's.
from df_pytest_isolation import (
    PIPE_CLOSING_LEAKER_SRC,
    deploy_clock_snapshot,
    deploy_clock_violation_reason,
    read_leaked_pid,
    run_in_new_session,
    wait_pid_gone,
    wait_proof_grace_secs,
)

SCRIPT = Path(__file__).parent.parent / "restart-all-orchestrators.sh"
UNIT_R = "orchestrator-reify.service"

FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing restart-all-orchestrators.sh.

Extends test_restart_orchestrator.py's single-unit fake with `list-units`
support and PER-UNIT state. State shape:
    {
      "running_units": [<unit>, ...],
      "units": {<unit>: {"MainPID": int, "ActiveState": str,
                          "ActiveEnterTimestamp": str,
                          "ActiveEnterTimestampMonotonic": int,
                          "scenario": "fresh"|"stale"|"delayed-fresh",
                          "fresh_after": int, "restarted": bool,
                          "post_restart_shows": int}},
      "calls": [...],
    }
`list-units` prints one line per entry in running_units (first
whitespace-delimited field is the unit name, matching the real script's
`awk '{print $1}'` parse). `show -p FIELDS UNIT` and `restart UNIT`
operate on state["units"][UNIT]; a unit with no configured scenario
defaults to "stale" (restart is a no-op, mirroring the single-unit fake's
default). Scenario "delayed-fresh" (task 2967) reports stale for the
first `fresh_after` post-restart `show` calls, then flips fresh --
mirroring tests/scripts/test_restart_all_orchestrators.py's bash
"delayed-fresh" fake's semantics, but keyed by per-unit state
(`restarted`, `post_restart_shows`, `fresh_after`) rather than an env var,
since this fake is already per-unit-state-based.
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
    state.setdefault("calls", []).append(argv[1:])

    if verb == "list-units":
        for unit in state.get("running_units", []):
            print(f"{unit} loaded active running Orchestrator")
        _save(state)
        return 0

    if verb == "restart":
        unit = rest[0] if rest else ""
        units = state.setdefault("units", {})
        ustate = units.setdefault(unit, {})
        scenario = ustate.get("scenario", "stale")
        if scenario == "fresh":
            ustate["MainPID"] = ustate.get("MainPID", 1000) + 1
            ustate["ActiveState"] = "active"
            ustate["ActiveEnterTimestampMonotonic"] = (
                ustate.get("ActiveEnterTimestampMonotonic", 0) + 5_000_000
            )
            ustate["ActiveEnterTimestamp"] = "restarted"
        elif scenario == "delayed-fresh":
            ustate["restarted"] = True
            ustate["post_restart_shows"] = 0
        _save(state)
        return 0

    if verb == "show":
        fields = None
        unit = None
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "-p":
                fields = rest[i + 1]
                i += 2
            elif tok.startswith("--property="):
                fields = tok.split("=", 1)[1]
                i += 1
            elif tok.startswith("-"):
                i += 1
            else:
                unit = tok
                i += 1
        ustate = state.get("units", {}).get(unit, {})
        if ustate.get("scenario") == "delayed-fresh" and ustate.get("restarted"):
            ustate["post_restart_shows"] = ustate.get("post_restart_shows", 0) + 1
            if ustate["post_restart_shows"] > ustate.get("fresh_after", 0):
                ustate["MainPID"] = ustate.get("MainPID", 1000) + 1
                ustate["ActiveState"] = "active"
                ustate["ActiveEnterTimestampMonotonic"] = (
                    ustate.get("ActiveEnterTimestampMonotonic", 0) + 5_000_000
                )
                ustate["ActiveEnterTimestamp"] = "restarted"
        current = {
            "MainPID": str(ustate.get("MainPID", 0)),
            "ActiveState": ustate.get("ActiveState", "active"),
            "ActiveEnterTimestamp": ustate.get("ActiveEnterTimestamp", "baseline"),
            "ActiveEnterTimestampMonotonic": str(ustate.get("ActiveEnterTimestampMonotonic", 0)),
        }
        keys = fields.split(",") if fields else list(current.keys())
        for k in keys:
            print(f"{k}={current.get(k, '')}")
        _save(state)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_systemctl(tmp_path, *, running_units, units=None):
    """Write a fake multi-unit `systemctl` into <tmp_path>/bin/.

    Returns (bin_dir, state_path).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "systemctl"
    fake.write_text(FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "systemctl_state.json"
    state_path.write_text(json.dumps({
        "running_units": list(running_units),
        "units": units or {},
        "calls": [],
    }))
    return bin_dir, state_path


def _write_heartbeat(fleet_dir, unit, **overrides):
    fleet_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": unit,
        "merge_idle": True,
        "depth": 0,
        "queue_empty": True,
        "ts_epoch": time.time(),
    }
    payload.update(overrides)
    (fleet_dir / f"{unit}.json").write_text(json.dumps(payload))


def _run_script(bin_dir, state_path, fleet_dir, *extra_args, env=None, timeout=20):
    """Run restart-all-orchestrators.sh with the fake systemctl on PATH.

    ORCH_FLEET_DEPLOY_CLOCK is NOT set here: conftest.py's session-scoped
    autouse `_df_fleet_deploy_clock_redirect` already points it at a tmp file
    for every spawner in this directory (task 3797). A per-test override still
    wins via `env=`, which is applied after the os.environ copy below.

    The spawn is SESSION-ISOLATED via run_in_new_session (task 3798), not a
    plain subprocess.run: subprocess.run's timeout kill()s the direct child
    only, and this script forks poll loops that outlived it by up to 27.8h,
    reparented to systemd --user. `_decode` below still applies -- the re-raised
    TimeoutExpired carries the partial output the timeout tests assert on.
    """
    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    full_env["ORCH_FLEET_DIR"] = str(fleet_dir)
    if env:
        full_env.update(env)
    return run_in_new_session(
        ["bash", str(SCRIPT), *extra_args],
        env=full_env,
        timeout=timeout,
    )


def _load_state(state_path):
    return json.loads(state_path.read_text())


# ---------------------------------------------------------------------------
# Process-group containment (task 3798).
#
# The mirror of this test lives in tests/scripts/test_orchestrator_watchdog.py
# as test_boundary_run_drain_script_timeout_kills_the_whole_process_group. The
# two are deliberately NOT cross-imported: these directories cannot import each
# other's test modules, which is the same constraint that forced
# test_boundary_fake_systemctl_matches_unit_suite_verbatim into existence. What
# they DO share is the one thing that matters -- a single spawn implementation
# in df_pytest_isolation.run_in_new_session, and (since the amendment pass) a
# single set of probes: PIPE_CLOSING_LEAKER_SRC / read_leaked_pid /
# wait_pid_gone. Copies of those here and in the mirror were byte-identical
# under cosmetic renames, which is the same "which of the copies did I fix"
# hazard one function over from the one the shared spawn exists to close.
# ---------------------------------------------------------------------------


def test_run_script_timeout_kills_the_whole_process_group(tmp_path, monkeypatch):
    """_run_script's timeout must reach the poll loops the script forks.

    subprocess.run's timeout path kill()s the DIRECT CHILD only, so a
    backgrounded grandchild survives, is reparented to systemd --user, and
    spends its grace unattended -- 86 concurrent orphans on 2026-08-06 and 82
    more on 2026-08-07 (task 3798).

    WHY THE LEAKER REDIRECTS ITS BACKGROUND CHILD'S STDIO, AND WHY THAT MUST
    NOT BE "SIMPLIFIED" AWAY: a background child that KEPT the inherited
    stdout/stderr pipes would hold their write ends open, so any drain run
    against it after the kill never sees EOF. This test drives the REAL
    spawner; with the stdio redirected to /dev/null nothing holds the pipe, the
    timeout is raised on schedule, and a regression fails CLEANLY on the
    surviving pid below. Dropping the `>/dev/null 2>&1` makes this test's
    behaviour depend on internals of whatever the spawner does after its kill,
    which is not what it is here to pin.

    Points SCRIPT at the synthetic leaker rather than the real script: it is
    read at call time inside _run_script, and the production script must never
    be driven by a test that exists to observe a timeout.
    """
    pidfile = tmp_path / "leaked.pid"
    leaker = tmp_path / "leaker.sh"
    leaker.write_text(PIPE_CLOSING_LEAKER_SRC)
    monkeypatch.setattr(sys.modules[__name__], "SCRIPT", leaker)

    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )

    leaked_pid = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_script(
                bin_dir, state_path, fleet_dir, "--drain",
                env={"LEAK_PIDFILE": str(pidfile)},
                timeout=2,
            )

        leaked_pid = read_leaked_pid(pidfile)
        assert wait_pid_gone(leaked_pid), (
            f"pid {leaked_pid} -- a grandchild backgrounded by the spawned "
            "script -- is STILL ALIVE after _run_script timed out. The timeout "
            "killed only the direct child, so every poll loop the script forked "
            "is now an orphan free to spend its grace and then issue a REAL "
            "systemctl restart. Fix: spawn via "
            "df_pytest_isolation.run_in_new_session."
        )
    finally:
        if leaked_pid is not None:
            with contextlib.suppress(OSError):
                os.kill(leaked_pid, signal.SIGKILL)


def _decode(maybe_bytes):
    """subprocess.run's TimeoutExpired attaches partial output as bytes
    even when text=True was passed to the original call -- normalize."""
    if maybe_bytes is None:
        return ""
    if isinstance(maybe_bytes, bytes):
        return maybe_bytes.decode(errors="replace")
    return maybe_bytes


# ---------------------------------------------------------------------------
# step-7: RED -- DEFER+FORCE (I3 two-way), plus idle-is-transparent
# ---------------------------------------------------------------------------

def test_force_restarts_busy_unit_when_grace_is_zero(tmp_path):
    """FORCE: a busy unit force-restarts immediately when
    ORCH_RESTART_FORCE_FIRE_AFTER_SECS=0 -- a force line is printed AND the
    restart is actually recorded AND the script exits 0."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=False, ts_epoch=time.time())

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_RESTART_FORCE_FIRE_AFTER_SECS": "0",
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "force-restarting" in result.stdout.lower(), (
        f"expected a force-restart line; got stdout={result.stdout!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


def test_defer_withholds_restart_while_busy(tmp_path):
    """DEFER: a busy unit with a large grace is NOT restarted before the
    test's own short subprocess timeout fires -- proving the script is
    still polling, not that it happened to finish quickly."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=False, ts_epoch=time.time())

    # ONE binding feeding BOTH the grace and the timeout, so they cannot drift:
    # the grace must outlast the timeout (or the script force-fires mid-test and
    # the assertion below fails), and must stay small enough that a poller which
    # escapes the kill self-terminates in seconds rather than 27.8h (task 3798).
    spawn_timeout = 3

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _run_script(
            bin_dir, state_path, fleet_dir, "--drain",
            env={
                "RESTART_VERIFY_TIMEOUT": "5",
                "ORCH_RESTART_FORCE_FIRE_AFTER_SECS": str(
                    wait_proof_grace_secs(spawn_timeout)
                ),
                "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
            },
            timeout=spawn_timeout,
        )

    stdout = _decode(exc_info.value.stdout)
    assert "deferring restart of orchestrator-reify.service: mid-merge" in stdout, (
        f"expected a stable defer-prefix line; got stdout={stdout!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] not in state["calls"], (
        f"restart must NOT have been recorded yet; got calls={state['calls']!r}"
    )


def test_idle_unit_restarts_transparently_with_no_defer_line(tmp_path):
    """A fresh, idle unit restarts with no defer/force line at all --
    the gate is a no-op when the unit isn't mid-merge."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=True, ts_epoch=time.time())

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={"RESTART_VERIFY_TIMEOUT": "5"},
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "deferring" not in result.stdout.lower(), (
        f"expected no defer line for an idle unit; got stdout={result.stdout!r}"
    )
    assert "force-restarting" not in result.stdout.lower(), (
        f"expected no force line for an idle unit; got stdout={result.stdout!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


# ---------------------------------------------------------------------------
# Hermeticity: this suite must never stamp the REAL fleet-deploy clock (3797)
# ---------------------------------------------------------------------------

def test_suite_never_stamps_the_repo_fleet_deploy_clock(
    tmp_path, _df_fleet_deploy_clock_redirect,
):
    """Driving the script must not touch the live checkout's deploy clock.

    ``restart-all-orchestrators.sh`` resolves ``CLOCK_FILE`` from
    ``$ORCH_FLEET_DEPLOY_CLOCK``, falling back to
    ``$REPO_DIR/data/orchestrator/last_redeploy_orchestrator.json`` where
    ``REPO_DIR`` is the checkout the SCRIPT lives in -- i.e. this worktree.
    Every exit-0 verified-fresh run in this file reaches
    ``stamp_fleet_deploy_clock``, so without a redirect a fake-systemctl test
    writes a REAL "the fleet just redeployed" stamp that
    ``scripts/orchestrator-watchdog.py`` then believes, suppressing its
    staleness backstop for ``ORCH_RESTART_MIN_INTERVAL_SECS`` (8h default).

    Both halves matter. The first asserts the repo clocks are untouched, reusing
    the repo-wide guard's own comparison (``deploy_clock_snapshot`` /
    ``deploy_clock_violation_reason``) rather than a second hand-rolled copy of
    it that could drift. The second asserts the script DID stamp somewhere
    harmless, so a future "fix" that merely stops stamping cannot pass.

    That second half is deliberately TIME-RELATIVE -- "the redirected clock
    changed across this run" -- not "it exists afterwards". The redirect is a
    session-scoped shared file, so earlier exit-0 tests in this file have
    already stamped it by the time this runs; an existence check would be
    satisfied by THEIR stamp and would stay green if the exit-0 path stopped
    stamping altogether. The redirect is taken from the fixture rather than
    ``os.environ`` so that deleting the fixture is a collection error naming it,
    not a bare KeyError.
    """
    repo_root = SCRIPT.parent.parent
    before = deploy_clock_snapshot(repo_root)

    redirected = Path(_df_fleet_deploy_clock_redirect)
    stamped_before = (
        (redirected.read_bytes(), redirected.stat().st_mtime_ns)
        if redirected.exists() else None
    )

    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=True, ts_epoch=time.time())

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={"RESTART_VERIFY_TIMEOUT": "5"},
    )

    # Non-vacuity: the assertions below are only meaningful if the run actually
    # reached the stamp, which happens only on the all-verified-fresh exit 0.
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    reason = deploy_clock_violation_reason(
        before, deploy_clock_snapshot(repo_root), root=repo_root,
    )
    assert reason is None, (
        f"{reason}\nThe write came from THIS test spawning "
        "restart-all-orchestrators.sh against a fake systemctl: point "
        "ORCH_FLEET_DEPLOY_CLOCK at a tmp file "
        "(scripts/tests/conftest.py::_df_fleet_deploy_clock_redirect)."
    )

    stamped_after = (
        (redirected.read_bytes(), redirected.stat().st_mtime_ns)
        if redirected.exists() else None
    )
    assert stamped_after is not None and stamped_after != stamped_before, (
        f"the script stamped nothing on this run: {redirected} is unchanged "
        f"({stamped_before!r} -> {stamped_after!r}). The clock must be "
        "REDIRECTED, not disabled -- stamping on a verified fleet restart is "
        "the correct production behaviour (task 2396 I2)."
    )
    stamp = json.loads(redirected.read_text())
    assert isinstance(stamp, dict) and "ts" in stamp and "iso" in stamp, (
        f"redirected clock has the wrong schema: {stamp!r}; expected the "
        "{ts, iso} shape both the coordinator and the watchdog read."
    )


# ---------------------------------------------------------------------------
# step-9: RED -- I4 (fail-toward-convergence), stale/absent branch
# ---------------------------------------------------------------------------

def test_absent_heartbeat_restarts_after_zero_grace(tmp_path):
    """ABSENT (I4 fail-toward-convergence): a unit with no heartbeat file at
    all still restarts once ORCH_DRAIN_UNKNOWN_GRACE_SECS elapses -- here 0,
    so immediately."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    # No heartbeat file written for UNIT_R at all.

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_UNKNOWN_GRACE_SECS": "0",
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


def test_stale_heartbeat_restarts_after_zero_grace(tmp_path):
    """STALE (I4 fail-toward-convergence): a unit whose heartbeat exists but
    is older than the freshness window still restarts once
    ORCH_DRAIN_UNKNOWN_GRACE_SECS elapses -- here 0, so immediately."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=True, ts_epoch=time.time() - 99999)

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_UNKNOWN_GRACE_SECS": "0",
            "ORCH_DRAIN_FRESH_WINDOW_SECS": "120",
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


def test_unknown_grace_withholds_restart_while_absent(tmp_path):
    """Proves the unknown-grace wait is a REAL bounded wait, not an instant
    fail-open: an absent heartbeat with a large ORCH_DRAIN_UNKNOWN_GRACE_SECS
    is NOT restarted before the test's own short subprocess timeout fires.
    This is the opposite fail-direction from the busy/defer path (which
    protects an in-flight merge), but unknown status still gets a bounded
    grace rather than restarting on the very first check."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    # No heartbeat file written for UNIT_R at all.

    # ONE binding feeding BOTH the grace and the timeout -- see
    # test_defer_withholds_restart_while_busy above. This is the test named in
    # every sampled orphan's PYTEST_CURRENT_TEST (task 3798).
    spawn_timeout = 3

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _run_script(
            bin_dir, state_path, fleet_dir, "--drain",
            env={
                "RESTART_VERIFY_TIMEOUT": "5",
                "ORCH_DRAIN_UNKNOWN_GRACE_SECS": str(
                    wait_proof_grace_secs(spawn_timeout)
                ),
                "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
            },
            timeout=spawn_timeout,
        )

    stdout = _decode(exc_info.value.stdout)
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] not in state["calls"], (
        f"restart must NOT have been recorded yet; got calls={state['calls']!r} "
        f"stdout={stdout!r}"
    )
