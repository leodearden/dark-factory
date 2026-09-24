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
import dataclasses
import json
import os
import re
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
    WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS,
    assert_synthetic_units,
    deploy_clock_snapshot,
    deploy_clock_violation_reason,
    fleet_dir_redirect_violation_reason,
    load_scaled_grace,
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
import time

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def _observe_lease(args):
    """Snapshot the in-flight fleet-redeploy lease as of THIS call (task 4755).

    This fake is the only vantage point a test has on the lease while it is
    actually held: it runs as a descendant of restart-all-orchestrators.sh,
    mid-sweep, whereas the test process only ever sees the before and after.

    `pid_cmdline` is what turns "pid is a plausible integer" into "pid is the
    sweep's own": the lease names a pid, and the process wearing that pid
    right now is read straight out of /proc. Resolved HERE rather than in the
    test because the pid is only guaranteed to still be running while the
    sweep that recorded it is mid-flight.
    """
    obs = {"args": args, "observed_at": time.time(), "lease": None, "pid_cmdline": None}
    try:
        with open(os.environ.get("ORCH_FLEET_LEASE", "")) as f:
            obs["lease"] = json.load(f)
    except (OSError, ValueError):
        return obs
    pid = obs["lease"].get("pid") if isinstance(obs["lease"], dict) else None
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read()
        except OSError:
            return obs
        obs["pid_cmdline"] = raw.replace(b"\\x00", b" ").decode(errors="replace").strip()
    return obs


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    if not args:
        return 1
    verb, rest = args[0], args[1:]

    state = _load()
    state.setdefault("calls", []).append(argv[1:])
    state.setdefault("lease_observations", []).append(_observe_lease(args))

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


def _read_poll_trace(path):
    """Read the drain poll ledger, as one ``(verdict, unit)`` pair per poll.

    The ledger is written by restart-all-orchestrators.sh's
    ``drain_check_verdict`` when ``ORCH_DRAIN_POLL_TRACE_FILE`` is set: one
    append-only TSV line per drain poll, so its LENGTH is a load-independent
    count of what the spawned script actually did inside the gate.

    Returns the EMPTY LIST when the file does not exist, so a test whose
    ledger never appeared fails on its own assertion message rather than on a
    bare FileNotFoundError that says nothing about what was being proven.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for raw_line in path.read_text().splitlines():
        fields = raw_line.split("\t")
        assert len(fields) == 2, (
            f"poll-trace records are <verdict>\\t<unit>, exactly two fields; "
            f"got {raw_line!r} in {path}"
        )
        records.append((fields[0], fields[1]))
    return records


def _assert_poll_ledger(polls, *, at_least, verdict, unit, too_few, context):
    """Assert the poll ledger's COUNT and its VOCABULARY, as two assertions.

    Never one fused assertion: a short ledger means the script was killed or
    returned before polling that often, a wrong record means
    drain_check_verdict wrote the wrong thing. Unrelated causes, so a fused
    message describing only the first sends the next reader chasing a timeout
    for a defect that has nothing to do with timing.
    """
    assert len(polls) >= at_least, (
        f"drain poll ledger holds {len(polls)} record(s), expected >= "
        f"{at_least}. {too_few} ledger={polls!r} {context}"
    )
    assert all(p == (verdict, unit) for p in polls), (
        f"the ledger's COUNT is fine ({len(polls)} >= {at_least}); its "
        f"vocabulary or field order is not. Every record must be "
        f"({verdict!r}, {unit!r}) -- look at drain_check_verdict's trace "
        f"write in restart-all-orchestrators.sh, NOT at any timeout. "
        f"ledger={polls!r} {context}"
    )


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

    # The genuinely per-root half: this proves THIS rootdir's conftest bound a
    # LIVE instance of the fixture -- which then either ESTABLISHED the value
    # checked above or ADOPTED one another root's instance had already
    # established (df_pytest_isolation.fleet_dir_redirect_target, task 4890).
    # The two are indistinguishable from here, deliberately: what this pins is
    # that the yielded path and the env var AGREE, which is exactly what
    # stopped holding in a two-root session before adoption existed.
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
    # That design is what makes the load-scaling below a ONE-line change:
    # wait_proof_grace_secs derives from this same (now scaled) binding, so the
    # grace-outlasts-timeout invariant holds automatically at every load -- at
    # base 3 the grace is the 30s floor, at the 22 cap it is 88, always >= 4x.
    #
    # BASE 3 IS PRESERVED DELIBERATELY, not widened (task 4890): load_scaled_grace
    # floors at its base, so an unloaded run is byte-identical and this test still
    # takes ~3s there. The cap is WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS because the
    # grace DERIVED from this binding must stay inside
    # LEAK_SELF_TERMINATION_CEILING_SECS -- see that constant's derivation.
    #
    # THE MEASURED MECHANISM (task 4218): a flat 3s had to cover bash start +
    # script parse + the fake systemctl's python3 (list-units) + drain_check.py's
    # own python3 before the defer `echo` in restart-all-orchestrators.sh is even
    # reached. The one observed failure's stdout ended exactly at the preceding
    # "Restarting 1 orchestrator unit(s)" line, which is the signature of the
    # budget expiring before that echo. The MIRROR test in
    # tests/scripts/test_orchestrator_watchdog.py::
    # test_boundary4_defers_busy_unit_while_others_proceed carries a comment
    # recording that 8s was already measured as insufficient for the same defer
    # line under load and was raised to 20 -- this site's 3s was 2.7x tighter
    # still. Freshness is NOT the mechanism: `classify` needs now - ts_epoch > 120
    # for stale, unreachable inside a 3s budget.
    spawn_timeout = load_scaled_grace(3, cap_secs=WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS)

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
        f"expected a stable defer-prefix line; got stdout={stdout!r} "
        f"stderr={_decode(exc_info.value.stderr)!r} "
        f"(spawn_timeout={spawn_timeout}s, load-scaled from base 3). "
        f"TWO mechanisms produce this, and their stdout is byte-identical, so "
        f"check both before touching a number: (1) the budget expired before "
        f"the defer `echo` was reached -- stdout ending at the preceding "
        f"'Restarting N orchestrator unit(s)' line is the tell, and the "
        f"remedy is the load scaling already applied here, so log the RESOLVED "
        f"budget above and check whether it hit the cap rather than widening "
        f"the base; (2) restart-all-orchestrators.sh coerces ANY non-zero exit "
        f"of drain_check.py to raw=\"absent\", after which drain_await_fresh "
        f"polls SILENTLY for up to ORCH_DRAIN_UNKNOWN_GRACE_SECS (default 120, "
        f"deliberately left unset by this test) and prints nothing at all. "
        f"Do NOT pin that grace to disambiguate -- it trades a silent stall "
        f"for an equally confusing spurious restart."
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


def test_absent_heartbeat_polls_through_a_nonzero_grace_then_restarts(tmp_path):
    """ABSENT + BOUNDED: the unknown grace TERMINATES, and the restart follows.

    The complement of test_unknown_grace_withholds_restart_while_absent below,
    which proves the wait is REAL by being killed inside it and so can never
    observe its end. Nothing else covers that end: the only other top-level
    absent elapse test uses a grace of 0, where drain_await_fresh breaks on
    its FIRST elapsed-check and the loop never sleeps. Here the grace is a
    small NONZERO 3s and the run COMPLETES -- >= 2 ledger polls catch the loop
    iterating, the proceed line shows the grace elapsed, and the restart shows
    the gate opened anyway (I4 fail-toward-convergence).
    """
    fleet_dir = tmp_path / "fleet"
    trace_path = tmp_path / "drain-poll-trace.tsv"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    # No heartbeat file written for UNIT_R at all.

    unknown_grace = 3
    # This run must FINISH, so the budget covers the grace, its polls, the
    # python3 spawns and the restart verify. load_scaled_grace with its DEFAULT
    # cap, NOT WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS: that cap exists to bound a
    # grace DERIVED from a spawn timeout via wait_proof_grace_secs, so an
    # escaped poller self-terminates inside LEAK_SELF_TERMINATION_CEILING_SECS.
    # Nothing here derives a grace from this number and nothing is killed, so
    # borrowing that cap would only under-budget a loaded host.
    timeout = load_scaled_grace(20)

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_UNKNOWN_GRACE_SECS": str(unknown_grace),
            "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
            "ORCH_DRAIN_POLL_TRACE_FILE": str(trace_path),
        },
        timeout=timeout,
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert (
        f"proceeding with restart of {UNIT_R}: heartbeat absent "
        f"after {unknown_grace}s grace"
    ) in result.stdout, (
        f"the grace must ELAPSE and say so: that line is the only "
        f"operator-visible evidence the bounded wait ended rather than never "
        f"having started. got stdout={result.stdout!r}"
    )
    _assert_poll_ledger(
        _read_poll_trace(trace_path),
        at_least=2, verdict="absent", unit=UNIT_R,
        too_few=(
            f"the {unknown_grace}s grace must have been POLLED through rather "
            f"than slept through in one shot: >= 2 polls is what proves "
            f"drain_await_fresh's loop body ran."
        ),
        context=f"timeout={timeout}s stdout={result.stdout!r}",
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"a bounded wait must END in the restart it was holding back; got "
        f"calls={state['calls']!r} stdout={result.stdout!r}"
    )


def test_an_unwritable_poll_trace_never_aborts_the_redeploy(tmp_path):
    """A mis-typed ORCH_DRAIN_POLL_TRACE_FILE must not take down the fleet.

    The ledger is an OBSERVABILITY knob; nothing about a fleet redeploy
    depends on it succeeding. The hazard is that restart-all-orchestrators.sh
    runs under `set -euo pipefail` while drain_check_verdict is only ever
    invoked via command substitution assigned to a plain variable --
    `_DRAIN_VERDICT="$(...)"` in drain_await_fresh, `verdict="$(...)"` in
    drain_gate -- so a failed `>>` that lands as that function's LAST command
    fails the assignment and aborts the ENTIRE run. An operator who
    fat-fingers a path would lose the redeploy, not just the trace.

    What this pins is the OBSERVABLE property -- a mis-typed path costs the
    operator the trace and nothing else -- not one spelling of the guard, so
    it holds however drain_check_verdict is ordered internally.

    Fresh + IDLE heartbeat, so the run completes on drain_gate's shortest path
    -- which still makes drain_await_fresh's ONE opening drain_check_verdict
    call, so the trace write IS exercised. The knob points inside a directory
    that does not exist: the likeliest operator typo, and no permissions games
    needed to reproduce it.
    """
    fleet_dir = tmp_path / "fleet"
    trace_path = tmp_path / "no-such-dir" / "drain-poll-trace.tsv"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=True)

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_POLL_TRACE_FILE": str(trace_path),
        },
    )

    assert result.returncode == 0, (
        f"an unwritable trace path must never take down a fleet redeploy; "
        f"got rc={result.returncode} stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"the redeploy must not merely survive the failed trace write, it "
        f"must still do its JOB; got calls={state['calls']!r} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # LOUD, not silent: the operator set the knob and gets no trace, so the
    # reason must reach stderr (no-silent-fail-soft). Asserted on the PATH
    # rather than on libc's wording, which is locale- and bash-version
    # dependent.
    assert str(trace_path) in result.stderr, (
        f"a swallowed trace-write failure leaves the operator with no trace "
        f"and no reason; expected {str(trace_path)!r} to appear in stderr. "
        f"got stderr={result.stderr!r}"
    )
    assert not trace_path.exists(), (
        f"nothing should have been created at {str(trace_path)!r} -- if it "
        f"exists, this test is no longer exercising the unwritable path"
    )


def test_the_poll_ledger_records_idle_on_the_gate_fast_path(tmp_path):
    """Pins the ledger's field ORDER, and a token other than "absent".

    Every other ledger assertion drives the ABSENT path, where drain_check.py
    prints the very token the gate acts on and the two fields never coincide
    by accident but the vocabulary is never exercised beyond one word. An
    idle unit takes drain_gate's shortest path: drain_await_fresh's one
    opening poll reads idle, its `while` body is never entered, drain_gate
    returns. So exactly one record, and the count pins that shape too.
    """
    fleet_dir = tmp_path / "fleet"
    trace_path = tmp_path / "drain-poll-trace.tsv"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=True)

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_POLL_TRACE_FILE": str(trace_path),
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    polls = _read_poll_trace(trace_path)
    assert polls == [("idle", UNIT_R)], (
        f"the fast path is exactly one poll, recorded <verdict>\\t<unit> in "
        f"that order; got {polls!r} stdout={result.stdout!r}"
    )


def test_a_drain_check_that_cannot_run_is_traced_as_the_coerced_absent(tmp_path):
    """The ledger records what the GATE acted on, not what drain_check.py said.

    Here drain_check.py says nothing at all: ORCH_DRAIN_FRESH_WINDOW_SECS is
    passed straight through to its --fresh-window, so a non-numeric value
    makes argparse exit 2 with an empty stdout and drain_check_verdict's
    `|| raw="absent"` fail-toward-convergence fallback supplies the verdict.
    The heartbeat on disk is fresh and IDLE, so an "absent" in the ledger can
    only have come from that coercion -- and the run must still complete and
    restart, since an unreadable verdict is exactly the case the gate is
    built to fail forward through.

    This is the REACHABLE half of the coercion. The other arm -- exit 0 with
    an unrecognized token -- cannot be driven from the script's public
    surface: drain_check.py's main() prints one of four literals and returns
    0, and this harness deliberately does not shim `python3` (a fake first on
    bin_dir would also shadow the fake systemctl's own `#!/usr/bin/env
    python3` shebang, see _heartbeat_timeline). It stays a defensive backstop
    against a future drain_check.py change.
    """
    fleet_dir = tmp_path / "fleet"
    trace_path = tmp_path / "drain-poll-trace.tsv"
    bad_window = "not-a-number"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=True)

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            # 0, so the grace breaks on its first elapsed-check: one poll.
            "ORCH_DRAIN_UNKNOWN_GRACE_SECS": "0",
            "ORCH_DRAIN_FRESH_WINDOW_SECS": bad_window,
            "ORCH_DRAIN_POLL_TRACE_FILE": str(trace_path),
        },
    )

    assert result.returncode == 0, (
        f"a drain_check.py that cannot run must not abort the redeploy; got "
        f"rc={result.returncode} stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    polls = _read_poll_trace(trace_path)
    assert polls == [("absent", UNIT_R)], (
        f"the heartbeat is fresh and idle, so 'absent' here can only be the "
        f"coerced verdict the gate acted on -- anything else means the "
        f"ledger is recording drain_check.py's raw reading instead. got "
        f"{polls!r} stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert bad_window in result.stderr, (
        f"drain_check_verdict leaves the subprocess's stderr unsuppressed so "
        f"an unreadable verdict is not silent; expected {bad_window!r} in "
        f"stderr. got stderr={result.stderr!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"fail-toward-convergence: an unreadable verdict still restarts; got "
        f"calls={state['calls']!r} stdout={result.stdout!r}"
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
    is NOT restarted before the test's own short subprocess timeout fires,
    AND the script is caught POLLING while it withholds that restart. This is
    the opposite fail-direction from the busy/defer path (which protects an
    in-flight merge), but unknown status still gets a bounded grace rather
    than restarting on the very first check.

    The negative assertion alone proves nothing -- a subprocess killed before
    it ever reached the gate satisfies it just as well -- and neither does one
    poll, which is exactly what an instant fail-open (check once, return)
    produces. TWO polls prove `drain_await_fresh`'s `while` body ran: sleep
    ORCH_DRAIN_POLL_INTERVAL_SECS, then re-poll. The ledger witnesses that
    LOAD-INDEPENDENTLY -- it counts what the script did, not how long the
    test's own clock happened to run.
    """
    fleet_dir = tmp_path / "fleet"
    trace_path = tmp_path / "drain-poll-trace.tsv"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    # No heartbeat file written for UNIT_R at all.

    # ONE binding feeding BOTH the grace and the timeout -- see
    # test_defer_withholds_restart_while_busy above. This is the test named in
    # every sampled orphan's PYTEST_CURRENT_TEST (task 3798).
    #
    # Load-scaled for the same measured reason as that sibling, and it matters
    # MORE here (task 4890 amendment). This test's only behavioural assertion
    # is NEGATIVE, so a flat 3s budget that expires before the script even
    # reaches the drain gate makes it pass VACUOUSLY -- proving nothing about
    # the bounded wait it is named for, and proving nothing SILENTLY, which is
    # a worse outcome than the flake fixed next door. Base 3 is preserved, not
    # widened, so an unloaded run stays byte-identical; the cap is
    # WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS because the grace derived from this
    # same binding must stay inside LEAK_SELF_TERMINATION_CEILING_SECS.
    spawn_timeout = load_scaled_grace(3, cap_secs=WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS)

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _run_script(
            bin_dir, state_path, fleet_dir, "--drain",
            env={
                "RESTART_VERIFY_TIMEOUT": "5",
                "ORCH_DRAIN_UNKNOWN_GRACE_SECS": str(
                    wait_proof_grace_secs(spawn_timeout)
                ),
                "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
                "ORCH_DRAIN_POLL_TRACE_FILE": str(trace_path),
            },
            timeout=spawn_timeout,
        )

    stdout = _decode(exc_info.value.stdout)
    # NON-VACUITY, part 1 of 2: the DOORSTEP. Without it the negative
    # assertion below is satisfied just as well by a subprocess killed before
    # it ever entered the wait (task 4890 amendment). There is no drain line
    # to assert on and that is by contract, not oversight:
    # restart-all-orchestrators.sh's `drain_await_fresh` "prints nothing to
    # stdout" while it polls, and the one line the absent path ever emits --
    # "proceeding with restart of <unit>: heartbeat absent after Ns grace" --
    # is printed when the grace ELAPSES, which must never happen here. So the
    # strongest witness AVAILABLE ON STDOUT is the last line before the gate:
    # it is echoed immediately ahead of the per-unit loop that calls
    # `drain_gate`, so reaching it proves the subprocess cleared bash start,
    # script parse and the fake systemctl's `list-units`.
    assert f"Restarting 1 orchestrator unit(s): {UNIT_R}" in stdout, (
        f"the script was killed before it reached the per-unit drain gate, so "
        f"the no-restart assertion below would hold VACUOUSLY. Raise nothing "
        f"by hand: spawn_timeout={spawn_timeout}s is already load-scaled from "
        f"base 3, so check whether it hit the "
        f"WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS cap before touching a number. "
        f"stdout={stdout!r} stderr={_decode(exc_info.value.stderr)!r}"
    )
    # NON-VACUITY, part 2 of 2: INSIDE the gate (task 4486). The doorstep
    # witness above is silent about everything past it -- a python3 spawn, a
    # poll interval, another spawn -- so a kill landing there still proves no
    # bounded wait. The ledger closes that window; >= 2 rather than 1 is
    # argued in the docstring.
    # HEADROOM, measured here over 3 runs at loadavg 579 on 32 cores (factor
    # 18.1, so the budget sat at the 22s cap): poll 2 landed at 1.6 / 2.4 /
    # 3.9s. Only the spawns scale with load -- ~1.0s of that is the fixed
    # ORCH_DRAIN_POLL_INTERVAL_SECS below. So the thin end is a burst
    # load_scaled_grace's 1-minute loadavg has not registered yet: factor 1.0
    # budgets 3.0s, which the 3.9s run above would have missed.
    _assert_poll_ledger(
        _read_poll_trace(trace_path),
        at_least=2, verdict="absent", unit=UNIT_R,
        too_few=(
            f"the script was killed without polling the drain gate twice, so "
            f"'a REAL bounded wait, not an instant fail-open' is UNPROVEN: "
            f"one poll (or none) is exactly what a fail-open produces. Raise "
            f"nothing by hand -- spawn_timeout={spawn_timeout}s is already "
            f"load-scaled from base 3, so check whether it hit the "
            f"WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS "
            f"(={WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS}s) cap first."
        ),
        context=(
            f"stdout={stdout!r} stderr={_decode(exc_info.value.stderr)!r}"
        ),
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] not in state["calls"], (
        f"restart must NOT have been recorded yet; got calls={state['calls']!r} "
        f"stdout={stdout!r}"
    )


# ---------------------------------------------------------------------------
# task 4077: RED -- drain_gate's busy->idle resume path (both the in-loop
# `verdict == "idle"` arm of its busy poll loop and the post-await
# `verdict == "idle"` arm nested inside that loop's stale/absent handoff)
# and its stale-during-defer re-classification (the in-loop stale/absent
# block that hands off to drain_await_fresh), including the deliberate
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
# ONE binding feeding both the grace and the timeout -- see
# test_defer_withholds_restart_while_busy. Here the grace is a
# MUST-NEVER-BE-REACHED bound rather than a wait-proving one: if the
# in-loop resume regresses, the run silently consumes the spawn timeout
# instead of force-firing early and looking like a pass.
_IN_LOOP_RESUME_SPAWN_TIMEOUT_SECS = 15
# Longer than the worst spawn-to-first-poll latency measured under load (2.80s
# at loadavg ~90 on 32 cores, task 5838) and than the 3s wall-clock offset
# (_FIRST_TRANSITION_DELAY_SECS) at which the first rewrite currently fires.
_SLOW_START_SECS = 4


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
    success. READ THAT LIST FOR WHAT IT IS: a wall-clock observation of THIS
    process, never a record of what the spawned script reached. Every timer is
    armed here at context entry and cancelled only once the with-BODY returns,
    so a label lands in `fired` iff the script's TOTAL wall clock outran that
    label's delay -- whatever the script did or did not observe.

    So only POSITIVE `<label> in fired` checks are legitimate, and what they
    assert is non-vacuity: "the rewrite this test depends on did happen before
    the script exited". A NEGATIVE `<label> not in fired` is FORBIDDEN in any
    form (task 4890). It reads as "the correct code never reached that state"
    and is in fact "this host was fast enough", so it fails on correct code
    under load: one stood in
    test_busy_stale_busy_oscillation_does_not_reset_the_force_fire_anchor and
    failed 2/10 isolated reruns at loadavg 90 on 32 cores, where the same run's
    wall clock was measured varying 11.3s-39.7s. Observe a counterfactual
    "trap" transition through the SUBPROCESS'S STDOUT instead -- that records
    what the script actually reached, and no amount of host load can perturb
    it. `test_fired_records_elapsed_wall_clock_not_script_reachability` pins
    this premise directly, and is the test to read before adding a timeline
    test that wants to assert a negative.

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


def test_fired_records_elapsed_wall_clock_not_script_reachability(tmp_path):
    """`fired` is a WALL-CLOCK observation of the TEST process -- not a record
    of what the spawned script actually reached.

    `_heartbeat_timeline` arms one `threading.Timer` per transition IN THIS
    process and cancels them only once the with-block exits, so a label
    lands in `fired` iff the BODY was still inside the block when that
    timer's delay elapsed -- whether or not anything ever read the heartbeat
    the transition wrote. The body below proves the decoupling: it finishes
    everything it cares about in its first statement (reading the
    pre-transition heartbeat), then merely LINGERS past the transition's
    delay, standing in for a subprocess still running under host load. The
    label lands anyway.

    That is why a NEGATIVE `assert <label> not in fired` cannot be a
    behavioural assertion: it asserts only "the with-body returned in under
    <delay> seconds", which on a contended host is a property of the LOAD,
    not of the code under test (task 4890). The POSITIVE `<label> in fired`
    non-vacuity checks elsewhere in this file are a different claim and are
    unaffected -- they assert a transition a test depends on did land.

    Load-independent in the direction that matters: it asserts a label IS
    present after lingering PAST the delay, so extra host load can only make
    it more true, never flaky. In-process and sub-second; spawns no
    subprocess and shims no PATH.
    """
    fleet_dir = tmp_path / "fleet"
    _write_heartbeat(fleet_dir, UNIT_R, **_HB_BUSY)
    trap_delay_secs = 0.2

    with _heartbeat_timeline(
        fleet_dir, UNIT_R, [("trap", trap_delay_secs, _HB_IDLE)],
    ) as fired:
        # The body's OWN business, complete in one statement: it reads the
        # heartbeat and gets the pre-transition value. Nothing below ever
        # looks at the file again, so nothing here observes the trap.
        observed = json.loads((fleet_dir / f"{UNIT_R}.json").read_text())
        # From here the body only LINGERS -- the stand-in for `_run_script`
        # still blocking on a child that host load has slowed down.
        time.sleep(trap_delay_secs * 2)
        # Bounded top-up wait: under heavy load the timer THREAD may not have
        # been scheduled by the time that sleep returns. Waiting on the
        # CONDITION rather than trusting one fixed sleep is what keeps this
        # test's own assertion load-independent -- extra load makes it wait
        # longer, never fail. The bound only caps a genuine hang.
        deadline = time.monotonic() + 30
        while not fired and time.monotonic() < deadline:
            time.sleep(0.05)

    assert "trap" in fired, (
        f"the trap label must land purely because the BODY lingered past "
        f"{trap_delay_secs}s, with nothing having read the heartbeat it "
        f"wrote; got fired={fired!r} body_observed={observed!r}"
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


@dataclasses.dataclass(frozen=True)
class _Rewrite:
    """Rewrite the heartbeat to `to` once the gate has polled `after` `polls`
    times since the previous rewrite."""

    after: str  # a drain verdict: "busy" | "idle" | "stale" | "absent"
    to: dict | None  # `_write_heartbeat` kwargs; None unlinks, driving "absent"
    polls: int = 1


# How often the rewrite watcher re-reads the ledger. Responsiveness only: a
# rewrite that lands late just adds polls of the state it replaces.
_REWRITE_WATCH_INTERVAL_SECS = 0.02


def _complete_poll_records(trace_path):
    """The ledger's ``(verdict, unit)`` records so far, newline-terminated only.

    The script appends while the watcher reads, so an unterminated tail is a
    record still being written and must never count.
    """
    try:
        text = Path(trace_path).read_text()
    except FileNotFoundError:
        return []
    return [tuple(line.split("\t")) for line in text.split("\n")[:-1]]


@contextlib.contextmanager
def _rewrites_on_gate_polls(fleet_dir, unit, trace_path, rewrites):
    """Apply `rewrites` to <fleet_dir>/<unit>.json in order, each once the
    drain gate has polled the verdict it waits for, while `_run_script` blocks.

    `drain_check.py` classifies a verdict purely from the on-disk heartbeat
    JSON (scripts/drain_check.py::classify), so a single static file can
    never exercise a mid-poll verdict CHANGE -- busy->idle, busy->stale->idle,
    or a stale<->busy oscillation. One watcher thread drives those
    transitions, triggered by the gate's own poll ledger at `trace_path`
    (the script's ORCH_DRAIN_POLL_TRACE_FILE).

    WHY THE LEDGER AND NOT A CLOCK. restart-all-orchestrators.sh::
    drain_check_verdict appends a record only AFTER python3 has READ the
    heartbeat, and records the verdict the gate then acts on. A record
    therefore proves the gate has acted on the state that produced it, so a
    rewrite it triggers can never land before the read it depends on,
    however loaded the host. An offset from spawn can: nothing bounds when
    the script's first poll lands (see _SLOW_START_SECS).

    THE CALLER INVARIANT. Each rewrite counts the records appended since the
    previous rewrite fired, and a poll already in flight when that rewrite
    landed may still have read the state it replaced. So `after` must be a
    verdict the PREVIOUS heartbeat state cannot produce: then every counted
    record is a read of the state just written.

    A rewrite whose trigger never arrives is not an error: a trap rewrite
    must never fire on correct code, and the callers' ledger assertions
    report everything else. An exception on the watcher thread is collected
    and surfaced on exit, so a failed rewrite can never masquerade as a
    passing test: as an AssertionError, or, if the with-body raised too (a
    `subprocess.TimeoutExpired` from `_run_script`, say), as a note on that
    exception, which is the more diagnostic of the two and stays primary.

    Rewriting real heartbeat JSON, rather than shimming a fake `python3` onto
    PATH to script drain_check.py's own output, is deliberate: this module's
    own docstring records that drain_check.py is NOT mocked here -- it runs
    for real against heartbeat files the tests write -- and a scripted-verdict
    shim would stop exercising classify()'s fresh-window arithmetic. It would
    also collide with `_make_fake_systemctl`'s fake, whose shebang is
    `#!/usr/bin/env python3`: bin_dir is already first on PATH, so a fake
    `python3` placed there would shadow it too.
    """
    stop = threading.Event()
    errors = []

    def _position_once_triggered(rewrite, since):
        while True:
            records = _complete_poll_records(trace_path)
            if records[since:].count((rewrite.after, unit)) >= rewrite.polls:
                return len(records)
            if stop.wait(_REWRITE_WATCH_INTERVAL_SECS):
                return None

    def _watch():
        try:
            position = 0
            for rewrite in rewrites:
                position = _position_once_triggered(rewrite, position)
                if position is None:
                    return
                if rewrite.to is None:
                    (Path(fleet_dir) / f"{unit}.json").unlink(missing_ok=True)
                else:
                    _write_heartbeat(fleet_dir, unit, **rewrite.to)
        except Exception as exc:  # collected, not raised -- see docstring
            errors.append(exc)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join(timeout=5)
        if errors:
            in_flight = sys.exc_info()[1]
            if in_flight is not None:
                in_flight.add_note(f"ALSO: heartbeat rewrite(s) raised: {errors!r}")
            else:
                raise AssertionError(f"heartbeat rewrite(s) raised: {errors!r}")


def _run_busy_unit_through(tmp_path, rewrites, *, spawn_timeout, **knobs):
    """Run restart-all-orchestrators.sh --drain on a busy UNIT_R, applying
    `rewrites` as the gate polls, and return (result, state, polls).

    `knobs` are merged into `_run_script`'s env over
    {"RESTART_VERIFY_TIMEOUT": "5", "ORCH_DRAIN_POLL_INTERVAL_SECS":
    str(_TIMELINE_POLL_INTERVAL_SECS)}. `state` is the fake systemctl's
    recorded state and `polls` the gate's finished poll ledger, as read by
    `_read_poll_trace`.
    """
    assert "ORCH_DRAIN_POLL_TRACE_FILE" not in knobs, (
        "_run_busy_unit_through OWNS ORCH_DRAIN_POLL_TRACE_FILE: its rewrite "
        "watcher must read the very ledger the script writes. Assert on the "
        "returned `polls` instead of passing the knob."
    )
    fleet_dir = tmp_path / "fleet"
    trace_path = tmp_path / "drain-poll-trace.tsv"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, **_HB_BUSY)

    env = {
        "RESTART_VERIFY_TIMEOUT": "5",
        "ORCH_DRAIN_POLL_INTERVAL_SECS": str(_TIMELINE_POLL_INTERVAL_SECS),
        "ORCH_DRAIN_POLL_TRACE_FILE": str(trace_path),
    }
    env.update(knobs)

    with _rewrites_on_gate_polls(fleet_dir, UNIT_R, trace_path, rewrites):
        result = _run_script(
            bin_dir, state_path, fleet_dir, "--drain", env=env, timeout=spawn_timeout,
        )

    return result, _load_state(state_path), _read_poll_trace(trace_path)


def _assert_resumed_from_the_busy_loop(result, state, polls):
    """Assert drain_gate's IN-LOOP resume: defer on a busy read, then resume
    on an idle read taken straight from the busy poll loop.

    `polls` is `_read_poll_trace` output. The ledger is what pins the ORDER
    in which the gate observed the unit's states; stdout only implies it.
    """
    context = (
        f"ledger={polls!r} stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.returncode == 0, context
    assert polls[:1] == [("busy", UNIT_R)], (
        f"the gate's FIRST poll must read {UNIT_R} busy. A first poll of "
        f"anything else means the drain landed before the gate ever saw the "
        f"unit busy -- the 2026-09-23 merge-gate red (task 5348 sighting 1): "
        f"no defer line, rc 0, restart recorded. An EMPTY ledger instead "
        f"means the script never wrote ORCH_DRAIN_POLL_TRACE_FILE: the knob "
        f"did not reach it, or the write failed (see stderr). {context}"
    )
    assert len(polls) >= 2, (
        f"an in-loop resume takes at least two polls: the busy read that "
        f"deferred and the idle read that resumed. Fewer means the script "
        f"left the gate without its busy loop ever polling. {context}"
    )
    assert polls == [("busy", UNIT_R)] * (len(polls) - 1) + [("idle", UNIT_R)], (
        f"the ledger's COUNT is fine; its shape is not. Every poll after the "
        f"first busy one must read busy until one final idle: a stale/absent "
        f"record is a detour through drain_await_fresh (the OTHER resume "
        f"site), and a ledger ending on busy is a force-fire. {context}"
    )
    assert f"deferring restart of {UNIT_R}: mid-merge" in result.stdout, (
        f"expected a defer line before the resume; {context}"
    )
    assert f"resuming restart of {UNIT_R}: drained" in result.stdout, (
        f"expected the in-loop idle resume line; {context}"
    )
    assert "force-restarting" not in result.stdout.lower(), (
        f"expected a resume, not a force-fire; {context}"
    )
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r} "
        f"{context}"
    )


def test_busy_unit_that_drains_mid_defer_resumes_and_restarts(tmp_path):
    """The ordinary successful outcome of a --drain redeploy: drain_gate's
    IN-LOOP idle verdict (scripts/restart-all-orchestrators.sh::drain_gate,
    the `verdict == "idle"` arm reached straight from its busy poll loop,
    before any stale/absent handoff). A unit that goes busy, then drains
    WHILE deferred, must resume the restart from inside the poll loop rather
    than waiting out the full busy grace. The drain lands only after the
    gate's first busy poll, so the defer-then-resume order does not depend on
    how fast the script starts."""
    spawn_timeout = _IN_LOOP_RESUME_SPAWN_TIMEOUT_SECS

    result, state, polls = _run_busy_unit_through(
        tmp_path, [_Rewrite(after="busy", to=_HB_IDLE)],
        spawn_timeout=spawn_timeout,
        ORCH_RESTART_FORCE_FIRE_AFTER_SECS=str(wait_proof_grace_secs(spawn_timeout)),
    )

    _assert_resumed_from_the_busy_loop(result, state, polls)


def test_a_slow_start_still_defers_before_the_drain_lands(tmp_path):
    """The test above with the gate's first poll landing AFTER the drain, as
    host load makes it land: the 2026-09-23 merge-gate red.

    BASH_ENV is sourced by bash before the script's first line, so the late
    start is charged against the same spawn budget real slowness is, and
    modelling it needs no change to FAKE_SYSTEMCTL_SRC, whose verbatim mirror
    lives in tests/scripts/test_orchestrator_watchdog.py. The budget is the
    test above's plus the deterministic stall, so the load-dependent part of
    the run keeps exactly that test's headroom.
    """
    slow_start = tmp_path / "slow-start.sh"
    slow_start.write_text(f"sleep {_SLOW_START_SECS}\n")
    spawn_timeout = _IN_LOOP_RESUME_SPAWN_TIMEOUT_SECS + _SLOW_START_SECS

    result, state, polls = _run_busy_unit_through(
        tmp_path, [_Rewrite(after="busy", to=_HB_IDLE)],
        spawn_timeout=spawn_timeout,
        ORCH_RESTART_FORCE_FIRE_AFTER_SECS=str(wait_proof_grace_secs(spawn_timeout)),
        BASH_ENV=str(slow_start),
    )

    _assert_resumed_from_the_busy_loop(result, state, polls)


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
    stale/absent re-classification (scripts/restart-all-orchestrators.sh::
    drain_gate, the `verdict == "stale"/"absent"` block inside its busy poll
    loop that hands off to drain_await_fresh) and that block's trailing
    `else` arm, which proceeds once the shorter grace elapses with the
    heartbeat still unfresh. Parametrized over the two ways a dead unit
    stops reporting: its heartbeat ages out ("stale") or its file
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

    result, state, polls = _run_busy_unit_through(
        tmp_path, [_Rewrite(after="busy", to=overrides)],
        spawn_timeout=spawn_timeout,
        ORCH_RESTART_FORCE_FIRE_AFTER_SECS=str(wait_proof_grace_secs(spawn_timeout)),
        ORCH_DRAIN_UNKNOWN_GRACE_SECS="0",
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # THE LOAD-BEARING ASSERTION: drain_gate's TOP-LEVEL stale/absent path
    # -- the early `return 0` it takes on the verdict from its very first
    # drain_await_fresh, above the defer line -- prints a byte-identical
    # "proceeding" line but returns BEFORE any defer line, so this is what
    # proves the IN-LOOP re-classification block ran, not the top-level path
    # with the unit simply starting stale/absent. The rewrite waits for the
    # gate's first busy poll, so correct code cannot reach that top-level path.
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
    assert (verdict_label, UNIT_R) in polls, (
        f"the gate never READ the {verdict_label} rewrite: ledger={polls!r} "
        f"stdout={result.stdout!r}"
    )


def test_unit_that_stops_heartbeating_mid_defer_proceeds_after_a_nonzero_grace_elapses(
    tmp_path,
):
    """Sibling of test_unit_that_stops_heartbeating_mid_defer_drops_into_the_
    shorter_grace, covering the arm that test's ORCH_DRAIN_UNKNOWN_GRACE_SECS=0
    cannot reach (reviewer_comprehensive #5): with a grace of 0,
    the stale/absent poll loop in
    scripts/restart-all-orchestrators.sh::drain_await_fresh breaks on its
    FIRST elapsed-check, before ever sleeping or re-reading the heartbeat --
    so the "poll at least once with a nonzero grace, never see a fresh
    reading, then proceed" combination, entered from the MID-DEFER
    drain_await_fresh call nested inside drain_gate's busy poll loop (rather
    than from drain_gate's own opening await), had no test at all (only the
    top-level entry point had a same-shaped zero-grace test; neither entry
    had a nonzero one). A heartbeat that STAYS stale once the gate has seen
    it busy (no further rewrite -- unlike the tests below, which hand the
    gate a fresh reading afterwards, this one deliberately lets the grace
    genuinely elapse) with a small nonzero grace forces at least one real
    sleep-and-recheck cycle before the grace elapses.
    """
    # ONE binding feeding both the grace and the timeout -- see
    # test_defer_withholds_restart_while_busy. Deliberately unreachable here.
    spawn_timeout = 15
    unknown_grace = 3

    result, state, polls = _run_busy_unit_through(
        tmp_path, [_Rewrite(after="busy", to=_HB_STALE)],
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
    assert ("stale", UNIT_R) in polls, (
        f"the gate never READ the stale rewrite: ledger={polls!r} "
        f"stdout={result.stdout!r}"
    )


@pytest.mark.parametrize(
    "verdict_label, overrides", [("stale", _HB_STALE), ("absent", None)],
    ids=["stale", "absent"],
)
def test_unit_that_drains_during_the_unknown_grace_resumes_after_the_await(
    tmp_path, verdict_label, overrides,
):
    """drain_gate's SECOND resume site: the post-drain_await_fresh idle
    verdict (scripts/restart-all-orchestrators.sh::drain_gate, the
    `verdict == "idle"` arm nested inside its in-loop stale/absent handoff,
    which reads _DRAIN_VERDICT rather than making a fresh
    drain_check_verdict call). A unit that stops
    heartbeating mid-defer -- via either way a dead unit stops reporting,
    "stale" or "absent" (parametrized to match its sibling,
    test_unit_that_stops_heartbeating_mid_defer_drops_into_the_shorter_grace;
    reviewer_comprehensive #5 -- this test previously covered only "stale")
    -- then comes back drained WHILE still inside the bounded unknown grace,
    must resume the restart from there.

    THE DISCRIMINATOR: this site's resume line
    (f"resuming restart of {UNIT_R}: drained") is byte-identical to the
    in-loop resume site's (drain_gate's `verdict == "idle"` arm reached
    straight from the busy poll loop, pinned by
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
    busy-resumption arm (scripts/restart-all-orchestrators.sh::drain_gate,
    the `elif verdict == "busy"` arm of its in-loop stale/absent handoff --
    the one carrying the "anchored once ... can't defer the forced restart
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
        unreachable BY THE SCRIPT: it prints the force line and no resume
        line. (The trap's timer may still FIRE in this process afterwards
        if the run is slow -- see below; that says nothing about the
        anchor.)
      - Anchor RESET (the regression): start_secs restarts at t~8-9, so the
        new deadline is ~8-9 + FORCE_FIRE(10) = ~18-19 -- AFTER the trap.
        The loop keeps polling past t=15, its own (unguarded-by-FORCE_FIRE)
        idle check reads the trap's heartbeat, and it prints "resuming
        restart of <unit>: drained" with NO force line (the reset deadline
        would not have been reached until t~18-19).
    The two counterfactuals differ in OUTPUT, not merely in duration, so
    every assertion below is text-level, read off the subprocess's STDOUT:
    exactly two defer lines (the initial one plus the re-defer after the
    stale interlude -- itself independent corroboration that the
    oscillation happened), a force line PRESENT, and a resume line ABSENT.
    That pair discriminates both counterfactuals completely, and stdout is
    a record of what the script actually reached, which no amount of host
    load can perturb.

    The trap is therefore observed through stdout and NEVER through this
    process's timer. A negative `assert "idle-trap" not in fired` used to
    stand here and was deleted (task 4890): `fired` is appended by a
    `threading.Timer` armed in the TEST process and cancelled only when
    `_run_script` returns, so it reports "the script's total wall clock
    exceeded 15s" -- a quantity measured varying 11.3s-39.7s across five
    runs at loadavg 90 on 32 cores, i.e. a property of the host, not of
    the anchor. It failed 2/10 isolated reruns while the code was correct.
    `test_fired_records_elapsed_wall_clock_not_script_reachability` (above)
    pins that premise directly.

    DO NOT, on a recurrence here: widen the trap delay, raise FORCE_FIRE,
    or re-add a negative `fired` assertion in any form. The trap ENTRY at
    t=15 stays -- it is load-bearing, being what makes the reset path print
    a resume line instead of merely force-firing later -- but its only
    legitimate observation is the stdout pair above.
    """
    # ORCH_DRAIN_UNKNOWN_GRACE_SECS is a must-never-elapse bound here (the
    # unit resumes busy on its own at t=8, well inside it), so this site wants
    # the LARGEST spawn timeout a wait-proving test may legally take -- which
    # is precisely what WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS is defined to be.
    # The 22 -> wait_proof_grace_secs(22)=88 <= 90 derivation, and why that
    # ceiling is load-bearing, live ONCE at that constant in
    # df_pytest_isolation and are deliberately not re-spelled here (task 4890
    # amendment): a bare literal would survive a move of
    # WAIT_PROOF_GRACE_MULTIPLIER or LEAK_SELF_TERMINATION_CEILING_SECS with a
    # silently-wrong comment, out of reach of
    # test_the_spawn_timeout_cap_is_the_largest_the_ceiling_permits, which
    # pins the constant but cannot see a copy of its arithmetic.
    spawn_timeout = WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS

    result, state, _ = _busy_unit_drain_run(
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
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


# ---------------------------------------------------------------------------
# The in-flight fleet-redeploy lease (task 4755), producer side.
#
# INVARIANT: for as long as a sweep is running, `$ORCH_FLEET_LEASE` names a
# live lease recording the sweep's own pid, the epoch it started, and the unit
# it is currently restarting -- and it is gone again on every exit path the
# shell can catch. The three readers (the watchdog's staleness backstop, the
# merge-landed coordinator, and the watchdog's liveness probe) all stand down
# while that file is live, which is how the fleet stops redeploying on top of
# its own in-flight sweep.
#
# This half of the suite owns MID-SWEEP observation and SIGNAL handling,
# matching how the two suites already divide; the exit-path lifecycle
# assertions that pair with the clock stamp live in
# tests/scripts/test_restart_all_orchestrators.py.
# ---------------------------------------------------------------------------

UNIT_S = synthetic_unit("solar")


def _lease_observations(state_path):
    """Every mid-sweep lease snapshot the fake systemctl recorded, in order."""
    return _load_state(state_path).get("lease_observations", [])


@contextlib.contextmanager
def _sweep_in_background(bin_dir, state_path, fleet_dir, lease_path, *extra_args, env=None):
    """Spawn the sweep, yield (popen, pgid) once its lease is on disk, always reap.

    A foreground `run_in_new_session` cannot serve the signal tests: they need
    to signal a sweep that is still running, and the only way to know one has
    got as far as acquiring its lease is to watch for the file. Session
    isolation and the group kill are kept -- `start_new_session=True` plus a
    pgid frozen at spawn is exactly `run_in_new_session`'s own defence, and for
    the same task-845 reason (`os.getpgid` on a reaped-and-recycled pid resolves
    the NEW owner's group). The `finally` is what honours the task-3798 leaked
    drain-process guard: whatever the test did or failed to do, the whole group
    is SIGKILLed before the test returns.
    """
    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    full_env["ORCH_FLEET_DIR"] = str(fleet_dir)
    full_env["ORCH_FLEET_LEASE"] = str(lease_path)
    if env:
        full_env.update(env)
    proc = subprocess.Popen(
        ["bash", str(SCRIPT), *extra_args],
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pgid = proc.pid
    try:
        deadline = time.monotonic() + load_scaled_grace(
            10, cap_secs=WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS,
        )
        while not lease_path.exists():
            assert proc.poll() is None, (
                f"the sweep exited (rc={proc.returncode}) before writing a lease at "
                f"{lease_path}"
            )
            assert time.monotonic() < deadline, (
                f"no lease appeared at {lease_path} while the sweep ran"
            )
            time.sleep(0.05)
        yield proc, pgid
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.communicate(timeout=10)


def test_lease_is_held_mid_sweep_and_names_the_sweeps_own_pid(tmp_path):
    """(a) While the sweep runs, the lease records {pid, started_ts, current_unit}.

    `pid` is asserted to be the SWEEP's pid, not merely a positive integer:
    the fake reads /proc/<pid>/cmdline at observation time, and a lease whose
    pid names some other process would silently defeat every reader -- all
    three of them stand down only while that pid is alive, so a wrong pid is
    either a lease that never expires or one that never holds.
    """
    fleet_dir = tmp_path / "fleet"
    lease_path = tmp_path / "lease.json"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )

    result = _run_script(
        bin_dir, state_path, fleet_dir,
        env={"RESTART_VERIFY_TIMEOUT": "5", "ORCH_FLEET_LEASE": str(lease_path)},
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    observations = [o for o in _lease_observations(state_path) if o["lease"] is not None]
    assert observations, (
        f"no systemctl call observed a lease on disk; the sweep must hold one "
        f"for its whole run. observations={_lease_observations(state_path)!r} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    for obs in observations:
        lease = obs["lease"]
        assert set(lease) == {"pid", "started_ts", "current_unit"}, (
            f"lease body must carry exactly pid/started_ts/current_unit; got {lease!r}"
        )
        assert isinstance(lease["pid"], int) and lease["pid"] > 0, (
            f"pid must be a positive int; got {lease['pid']!r}"
        )
        assert obs["pid_cmdline"] is not None and "restart-all-orchestrators.sh" in obs["pid_cmdline"], (
            f"the lease pid {lease['pid']} must be the sweep's own process; "
            f"/proc said {obs['pid_cmdline']!r}"
        )
        assert isinstance(lease["started_ts"], (int, float)), (
            f"started_ts must be numeric; got {lease['started_ts']!r}"
        )
        assert 0 <= obs["observed_at"] - lease["started_ts"] < 300, (
            f"started_ts {lease['started_ts']!r} is not a plausible epoch for a "
            f"sweep observed at {obs['observed_at']!r}"
        )


def test_lease_current_unit_advances_in_ordered_units_order(tmp_path):
    """(b) `current_unit` names the unit being restarted RIGHT NOW.

    This is the field the liveness probe scopes its suppression on, so a lease
    that named only "some sweep is running" would force a blanket liveness
    disable -- which I5 forbids. Two units, with SELF_UNIT among them, also
    pins the deferred-self ordering: ordered_units puts everything except
    SELF_UNIT first in enumeration order and appends SELF_UNIT last, so
    current_unit must walk [UNIT_S, UNIT_R] even though list-units reported
    [UNIT_R, UNIT_S].
    """
    fleet_dir = tmp_path / "fleet"
    lease_path = tmp_path / "lease.json"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path,
        running_units=[UNIT_R, UNIT_S],
        units={UNIT_R: {"scenario": "fresh"}, UNIT_S: {"scenario": "fresh"}},
    )

    result = _run_script(
        bin_dir, state_path, fleet_dir,
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_FLEET_LEASE": str(lease_path),
            "SELF_UNIT": UNIT_R,
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    observations = [o for o in _lease_observations(state_path) if o["lease"] is not None]
    # Dedup CONSECUTIVE repeats only: each unit is the subject of several
    # systemctl calls, and what is under test is the order of the transitions,
    # not how many calls each unit happened to make.
    walk = []
    for obs in observations:
        value = obs["lease"]["current_unit"]
        if not walk or walk[-1] != value:
            walk.append(value)
    assert walk == ["", UNIT_S, UNIT_R], (
        f"current_unit must start empty (the lease is acquired before any unit "
        f"is enumerated), then follow ordered_units with SELF_UNIT={UNIT_R} "
        f"last; got {walk!r} from observations={observations!r}"
    )
    # The gate runs BEFORE the restart, so a unit deferred for its whole busy
    # grace is still correctly named as current -- pinned by the fact that the
    # unit's very first observed call already sees its own name.
    first_for_self = next(
        o for o in observations if o["args"][:2] == ["restart", UNIT_R]
    )
    assert first_for_self["lease"]["current_unit"] == UNIT_R, (
        f"the lease must already name {UNIT_R} by the time it is restarted; "
        f"got {first_for_self!r}"
    )


def test_sigterm_mid_sweep_releases_the_lease(tmp_path):
    """(g) A SIGTERMed sweep releases its lease.

    The script had NO trap before this task, and an untrapped SIGTERM kills
    the shell WITHOUT running an EXIT trap -- so this fails against a naive
    `trap lease_release EXIT` alone. Operators stop a sweep this way
    (`systemctl --user stop`, a killed transient unit), and a lease left
    behind by one suppresses every redeploy tier until the max-age bound
    expires.
    """
    fleet_dir = tmp_path / "fleet"
    lease_path = tmp_path / "lease.json"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    # Busy heartbeat + a grace far longer than this test: the sweep parks in
    # drain_gate's defer loop, which is what gives us a sweep to signal.
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=False, ts_epoch=time.time())

    with _sweep_in_background(
        bin_dir, state_path, fleet_dir, lease_path, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_RESTART_FORCE_FIRE_AFTER_SECS": str(
                wait_proof_grace_secs(load_scaled_grace(
                    10, cap_secs=WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS,
                ))
            ),
            "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
        },
    ) as (proc, pgid):
        assert lease_path.exists()
        # The GROUP, so the foreground `sleep` dies too: bash defers a trap
        # until the running foreground command returns, so signalling the
        # shell alone would stall the release for a whole poll interval.
        os.killpg(pgid, signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=30)

    assert not lease_path.exists(), (
        f"a SIGTERMed sweep must release its lease; {lease_path} still holds "
        f"{lease_path.read_text()!r}. stdout={stdout!r} stderr={stderr!r}"
    )


def test_sigkill_mid_sweep_leaves_the_lease_behind_by_design(tmp_path):
    """(h) SIGKILL strands the lease -- BY DESIGN, and the reason for the bound.

    Not a defect and not a gap to be closed: SIGKILL is uncatchable, so no
    amount of trapping can make the producer clean up after one. This test
    exists so that a future reader finding a stranded lease reads it as the
    ANTICIPATED case rather than a bug, and so the reader-side max-age bound
    (ORCH_FLEET_LEASE_MAX_AGE_SECS) can never be deleted as redundant: it is
    the ONLY thing standing between a SIGKILLed sweep and a wedged fleet.
    """
    fleet_dir = tmp_path / "fleet"
    lease_path = tmp_path / "lease.json"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=False, ts_epoch=time.time())

    with _sweep_in_background(
        bin_dir, state_path, fleet_dir, lease_path, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_RESTART_FORCE_FIRE_AFTER_SECS": str(
                wait_proof_grace_secs(load_scaled_grace(
                    10, cap_secs=WAIT_PROOF_SPAWN_TIMEOUT_CAP_SECS,
                ))
            ),
            "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
        },
    ) as (proc, pgid):
        held = json.loads(lease_path.read_text())
        os.killpg(pgid, signal.SIGKILL)
        proc.communicate(timeout=30)
        assert wait_pid_gone(proc.pid), "the SIGKILLed sweep must actually be gone"

    assert lease_path.exists(), (
        "SIGKILL is uncatchable, so the lease MUST still be here -- if this "
        "ever starts passing by removal, something other than the script is "
        "deleting leases, and the reader-side max-age bound is being asked to "
        "cover a case it was never given"
    )
    assert json.loads(lease_path.read_text()) == held, (
        "the stranded lease must be byte-equal to what the sweep held"
    )


def test_every_exit_after_acquisition_is_covered_by_the_release_trap():
    """(f)+(j) Trap COVERAGE, asserted structurally -- the only way it can be.

    MEASURED, not assumed (2026-09-21): drain_gate's `_drain_validate_verdict`
    defensive `exit 1` is unreachable from outside the script, so it cannot be
    driven by a behavioural test at all. drain_check_verdict whitelists its own
    subprocess's output through a `case idle|busy|stale|absent` and coerces
    anything else to "absent", and drain_await_fresh sets _DRAIN_VERDICT only
    from that function -- so no heartbeat, no drain_check.py behaviour and no
    PATH shim can produce an unrecognised verdict. Driving the script with a
    `python3` on PATH that printed `bogus-verdict` and exited 0 produced
    `heartbeat absent after 0s grace` and exit 0, never the `BUG(task 3852)`
    abort. The branch is a backstop against a future IN-script refactor, which
    is precisely the change this test protects the lease against.

    So the checkable invariant is coverage rather than any one path: the EXIT
    trap is installed before execution can reach ANY function body, hence
    before every `exit` inside one. Two structural facts carry that:

    * the trap is installed before `mapfile -t running_units`, the first
      top-level statement after the definitions -- so no function has yet been
      CALLED, however early its `exit` sits in the file; and
    * the only `exit` textually before the trap is the argument-parse one,
      which is also (j): a rejected argument must leave NO lease, because
      acquisition happens after parsing.
    """
    lines = SCRIPT.read_text().splitlines()

    def _sole_index(pattern, what):
        hits = [i for i, line in enumerate(lines) if re.search(pattern, line)]
        assert len(hits) == 1, (
            f"expected exactly one {what} line matching {pattern!r}; got "
            f"{[(i + 1, lines[i]) for i in hits]!r}"
        )
        return hits[0]

    trap_at = _sole_index(r"^trap lease_release EXIT$", "EXIT-trap install")
    acquire_at = _sole_index(r"^lease_acquire$", "top-level lease_acquire call")
    enumerate_at = _sole_index(r"^mapfile -t running_units", "unit-enumeration")

    assert trap_at < acquire_at, (
        "the trap must be installed BEFORE the lease is acquired: an exit "
        "between the two is then harmless (rm -f is idempotent), whereas the "
        "reverse order strands a lease on any failure inside lease_acquire"
    )
    assert acquire_at < enumerate_at, (
        "the lease must be held for the whole sweep, so it is acquired before "
        "any unit is enumerated"
    )
    assert trap_at < enumerate_at, (
        "the trap must be installed before the first top-level statement that "
        "calls a function -- otherwise an `exit` inside a function body (e.g. "
        "_drain_validate_verdict's defensive abort) escapes the release"
    )

    pre_trap_exits = [
        (i + 1, lines[i]) for i, line in enumerate(lines[:trap_at])
        if re.match(r"^\s*exit \d+\s*$", line)
    ]
    assert len(pre_trap_exits) == 1, (
        f"exactly one exit may precede the trap -- the argument-parse "
        f"rejection, which must write no lease; got {pre_trap_exits!r}"
    )
    reject_at = _sole_index(r"unexpected argument", "argument-rejection")
    assert reject_at < trap_at < acquire_at, (
        f"the argument rejection (line {reject_at + 1}) must stay ahead of the "
        f"trap and the acquisition, so an unknown argument never writes a lease"
    )
