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
import tempfile
import threading
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
    assert_synthetic_units,
    deploy_clock_snapshot,
    deploy_clock_violation_reason,
    fleet_dir_redirect_violation_reason,
    read_leaked_pid,
    run_in_new_session,
    synthetic_unit,
    wait_pid_gone,
    wait_proof_grace_secs,
)

SCRIPT = Path(__file__).parent.parent / "restart-all-orchestrators.sh"
# SYNTHETIC, not the real `orchestrator-reify.service` this used to be (task
# 3799). The fake systemctl shadows the real one only for as long as its tmpdir
# sits on PATH; an orphaned poll loop that outlives it -- 27.8h in the worst
# case measured for task 3798 -- resolves /usr/bin/systemctl and restarts
# whatever name it was handed. `reify` still names what the fixture stands in
# for, so the tests below keep reading as being about the reify orchestrator.
UNIT_R = synthetic_unit("reify")

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

    Every unit name handed in must be SYNTHETIC (task 3799). This is the
    PATH-shimming seam -- the point where a name starts being answerable by a
    fake that only shadows `systemctl` while its tmpdir lives -- so checking it
    here covers every caller, including the ones nobody has written yet. See
    test_fake_systemctl_rejects_a_real_unit_name for the hazard.
    """
    assert_synthetic_units(
        [*running_units, *(units or {})],
        where="scripts/tests/test_restart_all_orchestrators.py::_make_fake_systemctl",
    )
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
    """Write <fleet_dir>/<unit>.json, replacing it ATOMICALLY.

    Written to a sibling tempfile in the same directory and moved into place
    with `os.replace`, mirroring the script's own `stamp_fleet_deploy_clock`
    mktemp + `mv -f` idiom -- rather than `Path.write_text`, which truncates
    the target file before writing its new content. That truncate-then-write
    window is a torn read for any concurrent reader: this function is also
    called from a background `threading.Timer` thread by `_heartbeat_timeline`
    (below) WHILE the spawned script polls this same file every
    ORCH_DRAIN_POLL_INTERVAL_SECS, and a poll landing inside the window would
    see a zero-length file -- drain_check.py's `_read_heartbeat` turns a
    `ValueError` from the empty/partial JSON into "absent", a verdict no
    timeline scheduled. `os.replace` is a same-filesystem rename, atomic on
    POSIX, so a concurrent reader always observes either the old content or
    the full new content, never a partial write (reviewer_comprehensive #2).
    """
    fleet_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": unit,
        "merge_idle": True,
        "depth": 0,
        "queue_empty": True,
        "ts_epoch": time.time(),
    }
    payload.update(overrides)
    target = fleet_dir / f"{unit}.json"
    fd, tmp_name = tempfile.mkstemp(
        dir=fleet_dir, prefix=f".{unit}.", suffix=".json.tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload))
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


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


def test_fleet_dir_is_redirected_away_from_the_live_checkout(
    _df_fleet_dir_redirect, tmp_path_factory,
):
    """ORCH_FLEET_DIR must point somewhere hermetic for the WHOLE session.

    THE CONSEQUENCE of it being unset, which is what this pins: this file's
    _run_script sets ORCH_FLEET_DIR per call, but the defect class is "a spawner
    that forgets" -- and restart-all-orchestrators.sh's FLEET_DIR default and
    drain_check.DEFAULT_FLEET_DIR
    both resolve their fleet dir from `${ORCH_FLEET_DIR:-...}`, so an unset (or
    EMPTY -- `${VAR:-...}` treats those identically) value falls through to the
    machine-global /home/leo/src/dark-factory/data/fleet. A test-spawned drain
    gate then reads five other projects' LIVE production heartbeats and decides
    the real fleet's drain state from them.

    This directory's OWN proof. tests/scripts/test_fleet_dir_isolation.py has
    the mirror: the two roots are wired separately (this one does not load the
    repo-root conftest at all), so a green test in one says nothing about the
    other. Only the WIRING differs between the two copies, so only the wiring is
    duplicated -- the comparison and its messages live once, in
    df_pytest_isolation.fleet_dir_redirect_violation_reason, for the same reason
    deploy_clock_snapshot is imported above rather than re-implemented here. Two
    copies of the assertion body had already drifted in message text before they
    were a day old.

    Takes the redirect from the fixture BY NAME rather than reading os.environ
    bare, so deleting the fixture fails collection with a message naming
    `_df_fleet_dir_redirect` instead of passing off a leftover env var.
    """
    value = os.environ.get("ORCH_FLEET_DIR")
    reason = fleet_dir_redirect_violation_reason(value, tmp_path_factory.getbasetemp())
    assert reason is None, reason

    # The genuinely per-root half: this proves THIS rootdir's conftest bound the
    # fixture that set the variable checked above.
    assert Path(_df_fleet_dir_redirect).resolve() == Path(value or "").resolve()


# ---------------------------------------------------------------------------
# Fixture-unit-name containment (task 3799).
#
# The mirror of this test lives in tests/scripts/test_orchestrator_watchdog.py
# as test_boundary_fake_systemctl_rejects_a_real_unit_name. Two copies for the
# same reason as the process-group pair below: these directories cannot import
# each other's test modules, so each root must prove its OWN factory validates.
# What they share is the rule itself -- df_pytest_isolation.assert_synthetic_units.
# ---------------------------------------------------------------------------


def test_fake_systemctl_rejects_a_real_unit_name(tmp_path):
    """_make_fake_systemctl must refuse a genuinely installed unit name.

    THE HAZARD, in the terms the incident established: the fake shadows
    `systemctl` only for as long as its tmpdir sits on PATH. A poll loop that
    outlives the test -- task 3798 measured orphans surviving 27.8 HOURS, well
    past pytest's tmpdir GC -- resolves /usr/bin/systemctl instead and issues a
    REAL restart of whatever unit name this factory handed it.
    `orchestrator-reify.service` is INSTALLED on this box, so that worst case is
    a real fleet restart; a synthetic name makes it a no-op against a unit that
    does not exist.

    Checked at the FACTORY, not by grepping test sources: this file's siblings
    hold ~40 real unit-name literals that are CONTRACT PINS against real
    production configuration, which a source-text guard would false-positive on.

    pytest.raises(pytest.fail.Exception) rather than AssertionError -- pytest.fail
    raises Failed, a BaseException, deliberately so a fixture's own
    `except Exception` cannot swallow it.
    """
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _make_fake_systemctl(tmp_path, running_units=["orchestrator-reify.service"])
    message = str(excinfo.value)
    assert "orchestrator-reify.service" in message, message
    assert "_make_fake_systemctl" in message, message


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
    # Interpolated, never a second copy of the literal: a hardcoded unit name
    # here is how the task-3799 rename would silently half-land -- the defer
    # assertion would just stop matching the name the fixture actually used.
    assert f"deferring restart of {UNIT_R}: mid-merge" in stdout, (
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


# ---------------------------------------------------------------------------
# task 4077: RED -- drain_gate's busy->idle resume path (both the in-loop
# site at 378-381 and the post-drain_await_fresh site at 388-390) and its
# stale-during-defer re-classification (382-404), including the deliberate
# non-reset of the force-fire anchor on a busy<->stale/absent oscillation.
# ---------------------------------------------------------------------------

_HB_BUSY = {"merge_idle": False}  # fresh + mid-merge
_HB_IDLE = {"merge_idle": True}  # fresh + drained
# Aged out no matter when it's actually written -- mirrors
# test_stale_heartbeat_restarts_after_zero_grace's ts_epoch spelling above.
# Computed once at import time rather than at write time: it only needs to
# stay further in the past than any fresh_window this file uses (default
# 120s), and a plain module-level dict constant (matching _HB_BUSY/_HB_IDLE)
# can't carry a call-time value.
_HB_STALE = {"merge_idle": True, "ts_epoch": time.time() - 99999}

# Every timeline below polls at this cadence; named so the offsets DERIVED
# from it (immediately below) move together with it instead of each test
# repeating a bare "1" string for ORCH_DRAIN_POLL_INTERVAL_SECS
# (reviewer_comprehensive #1).
_TIMELINE_POLL_INTERVAL_SECS = 1
# The delay before a timeline's FIRST transition. It must clear both the
# subprocess's own startup and drain_gate's first heartbeat read -- fast under
# no load (~0.2s observed) but NOT bounded -- with margin to spare: if a
# loaded box pushes that first read out past this point, the read observes
# the ALREADY-flipped heartbeat instead of the pre-timeline value, and every
# assertion that depends on the pre-flip behaviour (starting with the initial
# "deferring" line every one of these tests asserts on) fails for a reason
# unrelated to the drain_gate branch under test. Three poll intervals rather
# than a bare `2.0`, so the margin scales if the poll interval ever does.
_FIRST_TRANSITION_DELAY_SECS = 3 * _TIMELINE_POLL_INTERVAL_SECS


@contextlib.contextmanager
def _heartbeat_timeline(fleet_dir, unit, timeline):
    """Rewrite <fleet_dir>/<unit>.json on a schedule while `_run_script` blocks.

    `drain_check.py` classifies a verdict purely from the on-disk heartbeat
    JSON (scripts/drain_check.py `classify`), so a single static file can
    never exercise a mid-poll verdict CHANGE -- busy->idle, busy->stale->idle,
    or a stale<->busy oscillation. This helper drives those transitions by
    rewriting the file on a schedule while the spawned script polls it.

    `timeline` is an ordered ``(label, delay_secs, overrides)`` sequence. Each
    entry arms one ``threading.Timer(delay_secs, ...)``, all started together
    at context entry so every delay is an offset from the SAME t0 -- callers'
    timings are relative to script start, and `_run_script` must be invoked
    INSIDE this block. ``overrides`` is forwarded to `_write_heartbeat` as
    kwargs; ``overrides is None`` instead UNLINKS <fleet_dir>/<unit>.json
    (missing_ok=True), driving the verdict to "absent".

    Yields a `fired` list that each transition appends its label to on
    success -- callers' non-vacuity handle in BOTH directions: a transition a
    test depends on asserts its label IS in `fired`; a counterfactual "trap"
    transition the correct code must never reach asserts its label is NOT in
    `fired`.

    Cancels and joins every timer on the way out, and asserts that no
    transition raised -- collected into a list rather than left to escape
    silently on a background thread, so a failed rewrite can never masquerade
    as a passing test. If the with-BODY also raised (e.g. `_run_script`
    raising `subprocess.TimeoutExpired` during a RED-proof mutant run), that
    exception is the more diagnostic of the two and is left to propagate
    as-is -- any collected transition errors are folded into it as a note
    instead of being raised as a separate `AssertionError` that would bump
    the body's own failure down to `__context__` (reviewer_comprehensive #3).

    Rewriting real heartbeat JSON, rather than shimming a fake `python3` onto
    PATH to script drain_check.py's own output, is deliberate: this module's
    own docstring records that drain_check.py is NOT mocked here -- it runs
    for real against heartbeat files the tests write -- and a scripted-verdict
    shim would stop exercising classify()'s fresh-window arithmetic. It would
    also collide with `_make_fake_systemctl`'s fake, whose shebang is
    `#!/usr/bin/env python3`: bin_dir is already first on PATH, so a fake
    `python3` placed there would shadow it too.
    """
    fired = []
    errors = []

    def _apply(label, overrides):
        try:
            if overrides is None:
                (Path(fleet_dir) / f"{unit}.json").unlink(missing_ok=True)
            else:
                _write_heartbeat(fleet_dir, unit, **overrides)
            fired.append(label)
        except Exception as exc:  # collected, not raised -- see docstring
            errors.append((label, exc))

    timers = []
    for label, delay_secs, overrides in timeline:
        timer = threading.Timer(delay_secs, _apply, args=(label, overrides))
        timer.daemon = True
        timers.append(timer)
    for timer in timers:
        timer.start()

    try:
        yield fired
    finally:
        for timer in timers:
            timer.cancel()
        for timer in timers:
            timer.join(timeout=5)
        if errors:
            in_flight = sys.exc_info()[1]
            if in_flight is not None:
                in_flight.add_note(
                    f"ALSO: heartbeat timeline transition(s) raised: {errors!r}"
                )
            else:
                raise AssertionError(
                    f"heartbeat timeline transition(s) raised: {errors!r}"
                )


def _busy_unit_drain_run(tmp_path, timeline, *, spawn_timeout, **knobs):
    """Shared preamble + spawn for the busy-unit drain-gate timeline tests
    below (reviewer_comprehensive #4).

    Every one of them starts UNIT_R busy, drives one or more scheduled
    heartbeat transitions across a run of restart-all-orchestrators.sh
    --drain via `_heartbeat_timeline`, and inspects the result -- only the
    timeline and a couple of env knobs actually differ between them. This
    factors out the rest (fake systemctl setup, the initial busy heartbeat,
    and the timeline-wrapped `_run_script` call) so a caller is left with
    just its own timeline, knobs, and assertions.

    `knobs` are merged into `_run_script`'s env on top of
    {"RESTART_VERIFY_TIMEOUT": "5", "ORCH_DRAIN_POLL_INTERVAL_SECS":
    str(_TIMELINE_POLL_INTERVAL_SECS)}, both of which every caller wants and
    none of them varies.

    Returns (result, state, fired) so a caller can assert directly on all
    three without re-deriving any of them: `state` is
    _load_state(state_path) (the fake systemctl's recorded calls) and
    `fired` is the timeline's own non-vacuity list (see
    _heartbeat_timeline's docstring).
    """
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, **_HB_BUSY)

    env = {
        "RESTART_VERIFY_TIMEOUT": "5",
        "ORCH_DRAIN_POLL_INTERVAL_SECS": str(_TIMELINE_POLL_INTERVAL_SECS),
    }
    env.update(knobs)

    with _heartbeat_timeline(fleet_dir, UNIT_R, timeline) as fired:
        result = _run_script(
            bin_dir, state_path, fleet_dir, "--drain", env=env, timeout=spawn_timeout,
        )

    return result, _load_state(state_path), fired


def test_busy_unit_that_drains_mid_defer_resumes_and_restarts(tmp_path):
    """The ordinary successful outcome of a --drain redeploy: drain_gate's
    IN-LOOP idle verdict (scripts/restart-all-orchestrators.sh:378-381). A
    unit that goes busy, then drains WHILE deferred, must resume the restart
    from inside the poll loop rather than waiting out the full busy grace."""
    # ONE binding feeding both the grace and the timeout -- see
    # test_defer_withholds_restart_while_busy. Here the grace is a
    # MUST-NEVER-BE-REACHED bound rather than a wait-proving one: if the
    # in-loop resume regresses, the run silently consumes the spawn timeout
    # instead of force-firing early and looking like a pass.
    spawn_timeout = 15

    result, state, fired = _busy_unit_drain_run(
        tmp_path, [("idle", _FIRST_TRANSITION_DELAY_SECS, _HB_IDLE)],
        spawn_timeout=spawn_timeout,
        ORCH_RESTART_FORCE_FIRE_AFTER_SECS=str(wait_proof_grace_secs(spawn_timeout)),
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Non-vacuity: proves the gate really deferred first, rather than sailing
    # through an idle first read.
    assert f"deferring restart of {UNIT_R}: mid-merge" in result.stdout, (
        f"expected a defer line before the resume; got stdout={result.stdout!r}"
    )
    assert f"resuming restart of {UNIT_R}: drained" in result.stdout, (
        f"expected the in-loop idle resume line; got stdout={result.stdout!r}"
    )
    assert "force-restarting" not in result.stdout.lower(), (
        f"expected a resume, not a force-fire; got stdout={result.stdout!r}"
    )
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )
    assert "idle" in fired, (
        f"the scheduled idle transition never landed: fired={fired!r}"
    )


@pytest.mark.parametrize(
    "verdict_label, overrides", [("stale", _HB_STALE), ("absent", None)],
    ids=["stale", "absent"],
)
def test_unit_that_stops_heartbeating_mid_defer_drops_into_the_shorter_grace(
    tmp_path, verdict_label, overrides,
):
    """A unit that stops heartbeating WHILE deferred (e.g. it crashed
    mid-merge) must drop into the shorter bounded stale/absent grace instead
    of waiting out the rest of the busy grace -- drain_gate's in-loop
    stale/absent re-classification (scripts/restart-all-orchestrators.sh:
    382-404) and its else arm (400-402). Parametrized over the two ways a
    dead unit stops reporting: its heartbeat ages out ("stale") or its file
    disappears entirely ("absent") -- drain_gate treats both identically.

    The busy grace (60s, via wait_proof_grace_secs) and the unknown grace (0s)
    are deliberately asymmetric, and that asymmetry IS the assertion: this
    run only completes in a few seconds because ORCH_DRAIN_UNKNOWN_GRACE_SECS
    is 0. A regression that kept waiting out the 60s busy grace instead would
    still eventually restart with the same "proceeding" text, but only after
    consuming the whole busy grace -- which is exactly what the RED proof
    (mutant MB) demonstrates via `_run_script`'s own timeout.
    """
    # ONE binding feeding both the grace and the timeout -- see
    # test_defer_withholds_restart_while_busy. Deliberately unreachable here:
    # the point is that the run finishes long before this busy grace would.
    spawn_timeout = 15

    result, state, fired = _busy_unit_drain_run(
        tmp_path, [(verdict_label, _FIRST_TRANSITION_DELAY_SECS, overrides)],
        spawn_timeout=spawn_timeout,
        ORCH_RESTART_FORCE_FIRE_AFTER_SECS=str(wait_proof_grace_secs(spawn_timeout)),
        ORCH_DRAIN_UNKNOWN_GRACE_SECS="0",
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # THE LOAD-BEARING ASSERTION: the top-level stale/absent path (line 359)
    # prints a byte-identical "proceeding" line but returns BEFORE any defer
    # line, so this is what proves the IN-LOOP re-classification block ran,
    # not the top-level path with the unit simply starting stale/absent.
    assert f"deferring restart of {UNIT_R}: mid-merge" in result.stdout, (
        f"expected a defer line before the re-classification; got "
        f"stdout={result.stdout!r}"
    )
    assert (
        f"proceeding with restart of {UNIT_R}: heartbeat {verdict_label} "
        "after 0s grace"
    ) in result.stdout, (
        f"expected the shorter-grace proceed line; got stdout={result.stdout!r}"
    )
    assert "force-restarting" not in result.stdout.lower(), (
        f"expected the shorter unknown grace, not the busy grace; got "
        f"stdout={result.stdout!r}"
    )
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )
    assert verdict_label in fired, (
        f"the scheduled {verdict_label} transition never landed: fired={fired!r}"
    )


def test_unit_that_stops_heartbeating_mid_defer_proceeds_after_a_nonzero_grace_elapses(
    tmp_path,
):
    """Sibling of test_unit_that_stops_heartbeating_mid_defer_drops_into_the_
    shorter_grace, covering the arm that test's ORCH_DRAIN_UNKNOWN_GRACE_SECS=0
    cannot reach (reviewer_comprehensive #5): with a grace of 0,
    drain_await_fresh's `while` loop (scripts/restart-all-orchestrators.sh:
    291-298) breaks on its FIRST elapsed-check, before ever sleeping or
    re-reading the heartbeat -- so the "poll at least once with a nonzero
    grace, never see a fresh reading, then proceed" combination, entered from
    the MID-DEFER handoff at line 385, had no test at all (only the top-level
    entry point had a same-shaped zero-grace test; neither entry had a
    nonzero one). A heartbeat that STAYS stale (no further scheduled
    transition -- unlike this file's other timelines, this one deliberately
    lets the grace genuinely elapse instead of racing a later flip) with a
    small nonzero grace forces at least one real sleep-and-recheck cycle
    before the grace elapses.
    """
    # ONE binding feeding both the grace and the timeout -- see
    # test_defer_withholds_restart_while_busy. Deliberately unreachable here.
    spawn_timeout = 15
    unknown_grace = 3

    result, state, fired = _busy_unit_drain_run(
        tmp_path, [("stale", _FIRST_TRANSITION_DELAY_SECS, _HB_STALE)],
        spawn_timeout=spawn_timeout,
        ORCH_RESTART_FORCE_FIRE_AFTER_SECS=str(wait_proof_grace_secs(spawn_timeout)),
        ORCH_DRAIN_UNKNOWN_GRACE_SECS=str(unknown_grace),
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert f"deferring restart of {UNIT_R}: mid-merge" in result.stdout, (
        f"expected a defer line before the re-classification; got "
        f"stdout={result.stdout!r}"
    )
    assert (
        f"proceeding with restart of {UNIT_R}: heartbeat stale "
        f"after {unknown_grace}s grace"
    ) in result.stdout, (
        f"expected the nonzero-grace proceed line; got stdout={result.stdout!r}"
    )
    assert "force-restarting" not in result.stdout.lower(), (
        f"expected the shorter unknown grace, not the busy grace; got "
        f"stdout={result.stdout!r}"
    )
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )
    assert "stale" in fired, (
        f"the scheduled stale transition never landed: fired={fired!r}"
    )


@pytest.mark.parametrize(
    "verdict_label, overrides", [("stale", _HB_STALE), ("absent", None)],
    ids=["stale", "absent"],
)
def test_unit_that_drains_during_the_unknown_grace_resumes_after_the_await(
    tmp_path, verdict_label, overrides,
):
    """drain_gate's SECOND resume site: the post-drain_await_fresh idle
    verdict (scripts/restart-all-orchestrators.sh:388-390). A unit that stops
    heartbeating mid-defer -- via either way a dead unit stops reporting,
    "stale" or "absent" (parametrized to match its sibling,
    test_unit_that_stops_heartbeating_mid_defer_drops_into_the_shorter_grace;
    reviewer_comprehensive #5 -- this test previously covered only "stale")
    -- then comes back drained WHILE still inside the bounded unknown grace,
    must resume the restart from there.

    THE DISCRIMINATOR: this site's resume line
    (f"resuming restart of {UNIT_R}: drained") is byte-identical to the
    in-loop resume site's (378-381, pinned by
    test_busy_unit_that_drains_mid_defer_resumes_and_restarts), so the two
    can't be told apart by text. They're told apart by an ORDERING
    INEQUALITY instead: ORCH_RESTART_FORCE_FIRE_AFTER_SECS=5 is deliberately
    SMALLER than the scheduled idle flip at t=8, and drain_await_fresh never
    consults the force-fire clock. A resume observed at ~t=8 is therefore
    only reachable from INSIDE drain_await_fresh -- had control stayed in
    the outer busy loop, it would have force-fired at t=5 instead. Do not
    "simplify" the 5-vs-8 relationship; it is the assertion.
    """
    # ORCH_DRAIN_UNKNOWN_GRACE_SECS is the must-never-elapse bound here (the
    # unit resumes on its own at t=8, well inside it); ONE binding still
    # feeds it from spawn_timeout, per test_defer_withholds_restart_while_busy.
    spawn_timeout = 20

    result, state, fired = _busy_unit_drain_run(
        tmp_path,
        [
            (verdict_label, _FIRST_TRANSITION_DELAY_SECS, overrides),
            ("idle", 8.0, _HB_IDLE),
        ],
        spawn_timeout=spawn_timeout,
        ORCH_RESTART_FORCE_FIRE_AFTER_SECS="5",
        ORCH_DRAIN_UNKNOWN_GRACE_SECS=str(wait_proof_grace_secs(spawn_timeout)),
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert f"deferring restart of {UNIT_R}: mid-merge" in result.stdout, (
        f"expected a defer line before the stale interlude; got "
        f"stdout={result.stdout!r}"
    )
    assert f"resuming restart of {UNIT_R}: drained" in result.stdout, (
        f"expected the post-await idle resume line; got stdout={result.stdout!r}"
    )
    # THE SITE-2 PROOF -- see the docstring's ordering-inequality argument.
    assert "force-restarting" not in result.stdout.lower(), (
        f"expected the resume to come from inside drain_await_fresh, not a "
        f"force-fire; got stdout={result.stdout!r}"
    )
    # Fail-toward-convergence proof: the resume must come from the idle
    # verdict itself, not from the unknown grace merely elapsing.
    assert "proceeding with restart" not in result.stdout, (
        f"expected the idle verdict to resume the restart, not the unknown "
        f"grace elapsing; got stdout={result.stdout!r}"
    )
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )
    assert fired == [verdict_label, "idle"], (
        f"expected both scheduled transitions to land in order; got fired={fired!r}"
    )


def test_busy_stale_busy_oscillation_does_not_reset_the_force_fire_anchor(tmp_path):
    """The deliberate NON-reset of `start_secs` in drain_gate's
    busy-resumption arm (scripts/restart-all-orchestrators.sh:391-399, the
    arm carrying the "anchored once ... can't defer the forced restart
    indefinitely" comment): a unit that goes busy -> stale -> busy again
    must NOT get a fresh force-fire deadline on the second busy reading --
    the deadline is anchored to when the unit FIRST went busy.

    A scheduled idle "trap" at t=15 is what makes this a TEXT-level
    assertion rather than a wall-clock one, which is deliberately NOT
    "simplified" into a timing check. The trap must sit strictly BETWEEN two
    deadlines that differ only by whether the anchor reset, so the margin on
    BOTH sides is the point (reviewer_comprehensive #1: the original t=12
    trap, timed to sit just after an elapsed(8) >= FORCE_FIRE(6) force-fire,
    measured as little as ~1.5s clear of the correct-path exit under 2x CPU
    oversubscription). FORCE_FIRE=10 here (not 6) is what buys that margin:
    it holds the correct path's force-fire a couple of poll cycles AFTER
    busy is redetected at t~8 instead of on the very next check, which pushes
    the reset path's hypothetical deadline out to ~18-19 and opens a wider
    window to place the trap in.
      - Anchor PRESERVED (correct): busy is redetected at t~8-9, still short
        of the UNRESET deadline (start~0 + FORCE_FIRE(10) = ~10). The outer
        loop force-fires the first time elapsed reaches 10 -- around
        t~10-11 -- and never reads the heartbeat again, so the t=15 trap is
        unreachable: `fired` stays ["stale", "busy"].
      - Anchor RESET (the regression): start_secs restarts at t~8-9, so the
        new deadline is ~8-9 + FORCE_FIRE(10) = ~18-19 -- AFTER the trap.
        The loop keeps polling past t=15, its own (unguarded-by-FORCE_FIRE)
        idle check reads the trap's heartbeat, and it prints "resuming
        restart of <unit>: drained" with NO force line (the reset deadline
        would not have been reached until t~18-19).
    The two counterfactuals differ in OUTPUT, not merely in duration, so the
    assertions below are text-level: exactly two defer lines (the initial
    one plus the re-defer after the stale interlude -- itself independent
    corroboration that the oscillation happened), a force line present, a
    resume line absent, and "idle-trap" absent from `fired`.
    """
    # ORCH_DRAIN_UNKNOWN_GRACE_SECS is a must-never-elapse bound here (the
    # unit resumes busy on its own at t=8, well inside it). Capped at 22
    # (rather than pushed higher for even more trap margin) because
    # wait_proof_grace_secs(22)=88 is the largest multiple of this spawn
    # timeout that still stays inside LEAK_SELF_TERMINATION_CEILING_SECS
    # (90s, df_pytest_isolation.py) -- see design_decisions in this task's
    # plan.json for why that ceiling is load-bearing.
    spawn_timeout = 22

    result, state, fired = _busy_unit_drain_run(
        tmp_path,
        [
            ("stale", _FIRST_TRANSITION_DELAY_SECS, _HB_STALE),
            ("busy", 8.0, _HB_BUSY),
            ("idle-trap", 15.0, _HB_IDLE),
        ],
        spawn_timeout=spawn_timeout,
        ORCH_RESTART_FORCE_FIRE_AFTER_SECS="10",
        ORCH_DRAIN_UNKNOWN_GRACE_SECS=str(wait_proof_grace_secs(spawn_timeout)),
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    defer_count = result.stdout.count(f"deferring restart of {UNIT_R}: mid-merge")
    assert defer_count == 2, (
        f"expected exactly two defer lines (initial + re-defer after the "
        f"stale interlude), which also independently catches a disabled "
        f"stale/absent handoff; got count={defer_count} stdout={result.stdout!r}"
    )
    assert f"force-restarting {UNIT_R}" in result.stdout, (
        f"expected the anchor to force-fire once elapsed(~8-9) >= 10; got "
        f"stdout={result.stdout!r}"
    )
    # THE ANCHOR PROOF -- see the docstring's two counterfactuals.
    assert f"resuming restart of {UNIT_R}: drained" not in result.stdout, (
        f"expected NO resume line -- one here means start_secs was reset "
        f"on the busy-resumption arm; got stdout={result.stdout!r}"
    )
    assert "idle-trap" not in fired, (
        f"the idle-trap transition fired, meaning the script was still "
        f"running at t=15 -- the anchor must have been reset: fired={fired!r}"
    )
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )
