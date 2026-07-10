"""Tests for deploy-w11-lane-lifecycle.sh — drives the script via subprocess
against a real temp git repo standing in for `<worktree_base>` (real `git
worktree add` lanes, so the adopt emitter's `git symbolic-ref --short -q
HEAD` / `git rev-parse -q --verify HEAD` subprocess reads observe faithful
git reality) and a fake `systemctl` shimmed onto PATH (observe the
apply-mode restart delegation without a live orchestrator-dark-factory
.service).

Mirrors scripts/tests/test_enable_laptop_persistent_worktree.py's
self-contained shim-setup-inside-`_run_script` pattern (callers never manage
the shim directly) and scripts/tests/test_restart_orchestrator.py's fake
`systemctl`-via-JSON-state design: deploy-w11-lane-lifecycle.sh delegates
its own restart+verify step to restart-orchestrator.sh (PRD design decision
-- DRY, don't duplicate the verify-loop), so the same fake systemctl shim
shape lets that delegation run for real against a fake target instead of a
live systemd.

WORKTREE_DIR contract this suite pins for deploy-w11-lane-lifecycle.sh's
implementation (step-2 onward): an ABSOLUTE `WORKTREE_DIR` is used as
`WORKTREE_BASE` directly; a RELATIVE value (the production default,
`.worktrees`) is joined under the script's own `${BASH_SOURCE[0]}`-derived
`PROJECT_ROOT`. This suite always passes an absolute path (the fake
worktree_base), never the production tree.

Record-contract literals below are pinned against the merged spine
(orchestrator/src/orchestrator/lane_lifecycle.py, orchestrator/src/
orchestrator/config.py, orchestrator/src/orchestrator/artifacts.py) --
prerequisite-2's read-only confirm.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "deploy-w11-lane-lifecycle.sh"
UNIT = "orchestrator-dark-factory.service"

# orchestrator/src/orchestrator/lane_lifecycle.py:42 -- directory (a direct
# child of worktree_base) holding every lane's durable LaneRecord.
LANE_STATE_DIRNAME = ".lane-state"
# orchestrator/src/orchestrator/config.py:853 -- TaskArtifacts.meta_root_for
# (orchestrator/src/orchestrator/artifacts.py:180) joins this onto
# worktree_base + the lane name to get the NEW plan.json location, a
# SIBLING of the lane worktree itself: <worktree_base>/.task-meta/<lane>/.
TASK_META_DIRNAME = ".task-meta"
# LaneRecord's exact field set (lane_lifecycle.py:181-196) -- `state` is
# persisted as the lowercase LaneState.value (lane_lifecycle.py:198-201).
LANE_RECORD_FIELDS = frozenset({
    "state", "task_id", "title", "branch", "seeded_from_sha", "updated_at",
})
LANE_STATE_VALUES = frozenset({
    "seed", "registered", "assigned", "in_use", "released", "quarantined",
})


# ---------------------------------------------------------------------------
# Fake systemctl (marker-file + canned `show`/`restart` responses)
# ---------------------------------------------------------------------------

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing deploy-w11-lane-lifecycle.sh's restart
delegation. The script shells out to restart-orchestrator.sh, which itself
invokes `systemctl --user show|restart` -- this fake stands in for a live
systemd so that delegation runs for real without touching one.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE and answers `show` from that file's fields; a
`restart` call ALWAYS bumps MainPID/ActiveEnterTimestampMonotonic and
reports ActiveState=active, simulating an unconditionally clean restart.
restart-orchestrator.sh's OWN stale/failed-restart branches are already
covered by test_restart_orchestrator.py -- this fake only needs the
always-succeeds scenario to verify deploy-w11-lane-lifecycle.sh's wiring
(that it calls restart-orchestrator.sh at all, with the right unit, AFTER
adopt).
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
            i += 1  # unit name or unrecognized flag -- not needed by the fake

    state = _load()
    state.setdefault("calls", []).append(argv[1:])

    if verb == "restart":
        # Ordering witness for the PRD "adopt-then-restart" migration
        # caution (task 2263 step-11): snapshot -- AT THE MOMENT of this
        # restart call -- whether the adopt step already populated
        # .lane-state/, so a test can prove adopt ran BEFORE restart
        # rather than merely that both happened somewhere in the run.
        lane_state_dir = os.environ.get("FAKE_SYSTEMCTL_LANE_STATE_DIR")
        if lane_state_dir:
            state["lane_state_populated_at_restart"] = bool(
                os.path.isdir(lane_state_dir) and os.listdir(lane_state_dir)
            )
        state["MainPID"] = state.get("MainPID", 1000) + 1
        state["ActiveState"] = "active"
        state["ActiveEnterTimestampMonotonic"] = (
            state.get("ActiveEnterTimestampMonotonic", 0) + 5_000_000
        )
        state["ActiveEnterTimestamp"] = "restarted"
        _save(state)
        return 0

    if verb == "show":
        current = {
            "MainPID": str(state.get("MainPID", 0)),
            "ActiveState": state.get("ActiveState", "active"),
            "ActiveEnterTimestamp": state.get("ActiveEnterTimestamp", "baseline"),
            "ActiveEnterTimestampMonotonic": str(state.get("ActiveEnterTimestampMonotonic", 0)),
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


def _systemctl_state_path(tmp_path):
    """The fake systemctl's JSON state file for a given pytest `tmp_path` --
    doubles as the invocation marker `_systemctl_calls` inspects. Single
    derivation point shared by `_fake_systemctl` (writer) and
    `_systemctl_calls` (reader) so the two can never drift apart."""
    return tmp_path / "systemctl_state.json"


def _fake_systemctl(tmp_path, *, main_pid=1234):
    """Write an executable fake `systemctl` into <tmp_path>/bin/systemctl and
    its backing JSON state/marker file.

    Returns (bin_dir, state_path).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "systemctl"
    fake.write_text(_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = _systemctl_state_path(tmp_path)
    state_path.write_text(json.dumps({
        "MainPID": main_pid,
        "ActiveState": "active",
        "ActiveEnterTimestamp": "baseline",
        "ActiveEnterTimestampMonotonic": 1_000_000,
        "calls": [],
    }))
    return bin_dir, state_path


def _systemctl_calls(worktree_base):
    """Return the list of systemctl invocations (each argv minus `--user`)
    recorded by the fake systemctl `_run_script` wires onto PATH for a run
    against `worktree_base` -- empty if systemctl was never invoked (or
    `_run_script` was never called against this `worktree_base`)."""
    path = _systemctl_state_path(worktree_base.parent)
    if not path.is_file():
        return []
    return json.loads(path.read_text())["calls"]


# ---------------------------------------------------------------------------
# Real git worktree_base fixture
# ---------------------------------------------------------------------------

def _git(repo, *args):
    """Run a git command against *repo*, raising loudly on failure -- test
    setup must never silently produce a lane that doesn't match the spec."""
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


def _build_worktree_base(tmp_path, lanes):
    """Build a real temp git repo (<tmp_path>/origin, one initial commit on
    `main`) and `git worktree add` a real lane worktree dir directly under
    <tmp_path>/worktrees for each entry in *lanes* -- faithful git reality
    (real branches/commits, real detached HEADs) for the adopt emitter's
    git-subprocess reads to observe, rather than a synthetic fixture.

    Each `lanes` entry is a dict:
        name           -- lane dir name, e.g. "_lane-7" (required).
        branch         -- branch to check out, e.g. "task/1234"; falsy
                          (None/"") checks out a DETACHED HEAD instead
                          (required key, may be None).
        plan           -- dict to serialize as the lane's plan.json content,
                          or None to omit plan.json entirely (optional,
                          default None).
        plan_location  -- "new" (writes
                          <worktree_base>/.task-meta/<name>/plan.json, the
                          post-W11-beta sibling-dir path) or "legacy"
                          (writes <worktree_base>/<name>/.task/plan.json);
                          ignored when plan is None (optional, default
                          "new").

    Returns `worktree_base` (<tmp_path>/worktrees), always created even when
    *lanes* is empty (the empty-base smoke case).
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "Test")
    (origin / "README.md").write_text("seed\n")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-q", "-m", "initial")

    worktree_base = tmp_path / "worktrees"
    worktree_base.mkdir()

    for lane in lanes:
        name = lane["name"]
        branch = lane.get("branch")
        lane_dir = worktree_base / name
        if branch:
            _git(origin, "worktree", "add", "-q", "-b", branch, str(lane_dir), "main")
        else:
            _git(origin, "worktree", "add", "-q", "--detach", str(lane_dir), "main")

        plan = lane.get("plan")
        if plan is not None:
            location = lane.get("plan_location", "new")
            if location == "new":
                plan_dir = worktree_base / TASK_META_DIRNAME / name
            elif location == "legacy":
                plan_dir = lane_dir / ".task"
            else:
                raise ValueError(f"unknown plan_location {location!r}")
            plan_dir.mkdir(parents=True, exist_ok=True)
            (plan_dir / "plan.json").write_text(json.dumps(plan))

    return worktree_base


def _lane_state_path(worktree_base, lane_name):
    return worktree_base / LANE_STATE_DIRNAME / f"{lane_name}.json"


def _read_lane_record(worktree_base, lane_name):
    """Return the parsed `.lane-state/<lane_name>.json` record, or None if
    no record has been written for that lane."""
    path = _lane_state_path(worktree_base, lane_name)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_script(worktree_base, *args, env=None):
    """Run deploy-w11-lane-lifecycle.sh against `worktree_base` via
    subprocess.

    Injects WORKTREE_DIR=<worktree_base> (absolute -- see the module
    docstring's WORKTREE_DIR contract) and puts a fresh fake `systemctl` on
    PATH (state reset each call; see `_fake_systemctl`) so the apply-mode
    restart delegation never touches a real systemd. RESTART_VERIFY_TIMEOUT
    defaults to a short 5s -- irrelevant to a successful restart, which the
    fake answers on its very first poll, but keeps a regression bounded
    instead of hanging up to restart-orchestrator.sh's 30s production
    default. FAKE_SYSTEMCTL_LANE_STATE_DIR points at this worktree_base's
    own `.lane-state/` dir, so the fake's `restart` verb can witness
    whether adopt already populated it (the step-11 ordering proof).

    Systemctl invocations are inspectable afterward via
    `_systemctl_calls(worktree_base)`.
    """
    tmp_path = worktree_base.parent
    bin_dir, state_path = _fake_systemctl(tmp_path)

    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    full_env["WORKTREE_DIR"] = str(worktree_base)
    full_env["FAKE_SYSTEMCTL_LANE_STATE_DIR"] = str(worktree_base / LANE_STATE_DIRNAME)
    full_env.setdefault("RESTART_VERIFY_TIMEOUT", "5")
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
# step-1: RED -- executable bit + empty-base --check smoke
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


def test_check_mode_on_empty_base_is_noop(tmp_path):
    """`--check` against an EMPTY worktree_base (no `_lane-*` lanes -- the
    dark_factory pool-less-host shape per the plan's final design decision)
    exits 0, writes no `.lane-state/` records, and never invokes systemctl
    (adopt-then-restart ordering means restart must not even be attempted
    when --check short-circuits before it)."""
    worktree_base = _build_worktree_base(tmp_path, [])

    result = _run_script(worktree_base, "--check")

    assert result.returncode == 0, (
        f"Expected --check on an empty base to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    lane_state_dir = worktree_base / LANE_STATE_DIRNAME
    assert not lane_state_dir.exists(), (
        f"Expected no {LANE_STATE_DIRNAME}/ directory to be created by "
        f"--check; found contents={list(lane_state_dir.iterdir()) if lane_state_dir.exists() else None}"
    )
    assert _systemctl_calls(worktree_base) == [], (
        f"Expected --check to never invoke systemctl; got calls="
        f"{_systemctl_calls(worktree_base)!r}"
    )


# ---------------------------------------------------------------------------
# step-3: RED -- ADOPT writes an ASSIGNED record for a bound lane
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plan_location", ["new", "legacy"])
def test_adopt_writes_assigned_record_for_bound_lane(tmp_path, plan_location):
    """A lane on a `task/<id>` branch WITH a plan.json (checked at either the
    new `.task-meta/<lane>/` path or the legacy `<lane>/.task/` path) is
    ASSIGNED -- adopt must seed the durable record so LaneLifecycle's
    record-driven recovery only engages the new path (state in
    {assigned,in_use}) for genuinely-bound lanes."""
    worktree_base = _build_worktree_base(tmp_path, [
        {
            "name": "_lane-7",
            "branch": "task/1234",
            "plan": {"title": "Some task title"},
            "plan_location": plan_location,
        },
    ])

    result = _run_script(worktree_base)

    assert result.returncode == 0, (
        f"Expected apply to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    record = _read_lane_record(worktree_base, "_lane-7")
    assert record is not None, (
        f"Expected a .lane-state/_lane-7.json record to be written; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert record["state"] == "assigned", f"record={record!r}"
    assert record["task_id"] == "1234", f"record={record!r}"
    assert record["branch"] == "task/1234", f"record={record!r}"
    sha = record["seeded_from_sha"]
    assert sha and re.fullmatch(r"[0-9a-f]{40}", sha), (
        f"Expected a non-empty 40-hex seeded_from_sha; got {sha!r}"
    )
    assert record["title"] == "Some task title", f"record={record!r}"
    assert record["updated_at"], f"Expected a non-empty updated_at; record={record!r}"


# ---------------------------------------------------------------------------
# step-5: RED -- ADOPT writes REGISTERED for free lanes
# ---------------------------------------------------------------------------

def test_adopt_writes_registered_record_for_free_lanes(tmp_path):
    """(a) a lane retaining a `task/<id>` branch but with NO plan.json (the
    2098 re-poisoning guard case -- must NOT be mis-marked ASSIGNED) and
    (b) a detached-HEAD lane (no branch at all) both get REGISTERED
    records."""
    worktree_base = _build_worktree_base(tmp_path, [
        {"name": "_lane-8", "branch": "task/999", "plan": None},
        {"name": "_lane-9", "branch": None},
    ])

    result = _run_script(worktree_base)

    assert result.returncode == 0, (
        f"Expected apply to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    rec8 = _read_lane_record(worktree_base, "_lane-8")
    assert rec8 is not None, (
        f"Expected a _lane-8 record; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert rec8["state"] == "registered", f"record={rec8!r}"
    assert rec8["task_id"] is None, f"record={rec8!r}"
    assert rec8["branch"] == "task/999", f"record={rec8!r}"
    assert rec8["seeded_from_sha"], f"Expected a non-empty seeded_from_sha; record={rec8!r}"

    rec9 = _read_lane_record(worktree_base, "_lane-9")
    assert rec9 is not None, (
        f"Expected a _lane-9 record; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert rec9["state"] == "registered", f"record={rec9!r}"
    assert rec9["task_id"] is None, f"record={rec9!r}"
    assert not rec9["branch"], f"Expected a null/empty branch for a detached lane; record={rec9!r}"
    assert rec9["seeded_from_sha"], f"Expected a non-empty seeded_from_sha; record={rec9!r}"


# ---------------------------------------------------------------------------
# step-7: RED -- idempotency (adopt never clobbers an existing record)
# ---------------------------------------------------------------------------

def test_adopt_never_clobbers_an_existing_record(tmp_path):
    """adopt only SEEDS absent records -- the restarted new-code orchestrator
    is the authoritative writer, so an existing `.lane-state/<lane>.json`
    must survive byte-for-byte and mtime-for-mtime, while a lane with no
    record yet still gets one. A second full apply is then a total no-op on
    records."""
    worktree_base = _build_worktree_base(tmp_path, [
        {"name": "_lane-7", "branch": "task/1234", "plan": {"title": "T"}},
        {"name": "_lane-8", "branch": "task/999", "plan": None},
    ])

    lane_state_dir = worktree_base / LANE_STATE_DIRNAME
    lane_state_dir.mkdir(parents=True)
    hand_crafted = {
        "state": "in_use",
        "task_id": "1234",
        "title": "hand-crafted, must survive",
        "branch": "task/1234",
        "seeded_from_sha": "0" * 40,
        "updated_at": "2020-01-01T00:00:00+00:00",
    }
    lane7_path = _lane_state_path(worktree_base, "_lane-7")
    lane7_path.write_text(json.dumps(hand_crafted, indent=2))
    before_bytes = lane7_path.read_bytes()
    before_mtime_ns = lane7_path.stat().st_mtime_ns

    result = _run_script(worktree_base)
    assert result.returncode == 0, (
        f"Expected apply to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    assert lane7_path.read_bytes() == before_bytes, (
        "adopt must not touch an existing record's bytes"
    )
    assert lane7_path.stat().st_mtime_ns == before_mtime_ns, (
        "adopt must not touch an existing record's mtime"
    )

    rec8 = _read_lane_record(worktree_base, "_lane-8")
    assert rec8 is not None, (
        f"Expected a NEW _lane-8 record to be written; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert rec8["state"] == "registered", f"record={rec8!r}"

    # A second full apply is a total no-op on records.
    result2 = _run_script(worktree_base)
    assert result2.returncode == 0, (
        f"Expected the second apply to exit 0; got {result2.returncode}\n"
        f"stdout={result2.stdout!r} stderr={result2.stderr!r}"
    )
    assert lane7_path.read_bytes() == before_bytes
    assert lane7_path.stat().st_mtime_ns == before_mtime_ns
    assert _read_lane_record(worktree_base, "_lane-8") == rec8


# ---------------------------------------------------------------------------
# step-9: RED -- --check/--dry-run preview only, no mutation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--check", "--dry-run"])
def test_check_and_dry_run_preview_without_mutating(tmp_path, flag):
    """Against a base with an ASSIGNED-worthy lane, --check/--dry-run must
    print an intended-record line naming the lane + state, create NO
    .lane-state/ files, never invoke systemctl, and leave the base's
    on-disk state completely unchanged."""
    worktree_base = _build_worktree_base(tmp_path, [
        {"name": "_lane-7", "branch": "task/1234", "plan": {"title": "T"}},
    ])

    before_listing = sorted(p.relative_to(worktree_base) for p in worktree_base.rglob("*"))

    result = _run_script(worktree_base, flag)

    assert result.returncode == 0, (
        f"Expected {flag} to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "_lane-7" in result.stdout and "assigned" in result.stdout, (
        f"Expected an intended-record line naming the lane + state in "
        f"stdout; got: {result.stdout!r}"
    )
    assert "Traceback" not in result.stdout and "Traceback" not in result.stderr, (
        f"Unexpected traceback: stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    lane_state_dir = worktree_base / LANE_STATE_DIRNAME
    assert not lane_state_dir.exists(), (
        f"Expected no {LANE_STATE_DIRNAME}/ directory to be created by {flag}"
    )
    assert _systemctl_calls(worktree_base) == [], (
        f"Expected {flag} to never invoke systemctl; got calls="
        f"{_systemctl_calls(worktree_base)!r}"
    )

    after_listing = sorted(p.relative_to(worktree_base) for p in worktree_base.rglob("*"))
    assert after_listing == before_listing, (
        f"Expected the base's on-disk listing to be byte-for-byte unchanged; "
        f"before={before_listing!r} after={after_listing!r}"
    )


# ---------------------------------------------------------------------------
# step-11: RED -- apply performs restart+verify (adopt BEFORE restart), and
# unknown-arg rejection
# ---------------------------------------------------------------------------

def test_apply_restarts_and_verifies_after_adopt(tmp_path):
    """A full apply run (no flags) against a bound lane must, after writing
    that lane's `.lane-state/` record, delegate to restart-orchestrator.sh's
    blocking `systemctl --user restart <unit>` (verified fresh via the fake's
    always-succeeds ActiveEnterTimestampMonotonic bump). The fake's ordering
    witness (`lane_state_populated_at_restart`, set at the moment `restart`
    is invoked) must be True -- proving the PRD "adopt-then-restart"
    ordering (migration caution / decomposition kappa): read reality -> write
    records -> THEN flip, not merely that both steps happened somewhere in
    the run."""
    worktree_base = _build_worktree_base(tmp_path, [
        {"name": "_lane-7", "branch": "task/1234", "plan": {"title": "T"}},
    ])

    result = _run_script(worktree_base)

    assert result.returncode == 0, (
        f"Expected apply to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    # Calls are recorded as the raw argv (including the constant `--user`
    # flag) -- matching test_restart_orchestrator.py's own assertion
    # convention (e.g. test_invokes_systemctl_restart_on_correct_unit),
    # which this suite's fake systemctl mirrors.
    calls = _systemctl_calls(worktree_base)
    assert ["--user", "restart", UNIT] in calls, (
        f"Expected a `systemctl --user restart {UNIT}` call; calls={calls!r}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    state = json.loads(_systemctl_state_path(worktree_base.parent).read_text())
    assert state.get("lane_state_populated_at_restart") is True, (
        f"Expected .lane-state/ to already be populated at the moment "
        f"systemctl restart was invoked (adopt must run BEFORE restart); "
        f"state={state!r}"
    )
    assert _read_lane_record(worktree_base, "_lane-7") is not None, (
        "Expected the lane's record to exist after a successful apply"
    )


def test_unknown_argument_is_rejected_without_restarting(tmp_path):
    """An unrecognized argument (e.g. --bogus) exits non-zero with a stderr
    message and performs no restart -- arg-parsing must reject before adopt
    or restart run at all."""
    worktree_base = _build_worktree_base(tmp_path, [
        {"name": "_lane-7", "branch": "task/1234", "plan": {"title": "T"}},
    ])

    result = _run_script(worktree_base, "--bogus")

    assert result.returncode != 0, (
        f"Expected --bogus to be rejected with a non-zero exit; got 0\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stderr.strip(), (
        f"Expected a non-empty stderr message for the rejected argument; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert _systemctl_calls(worktree_base) == [], (
        f"Expected no systemctl invocation when an unknown arg is rejected; "
        f"got calls={_systemctl_calls(worktree_base)!r}"
    )


# ---------------------------------------------------------------------------
# step-13: RED -- schema-coherence (drift guard vs LaneRecord)
# ---------------------------------------------------------------------------

def test_written_record_matches_lane_record_schema_exactly(tmp_path):
    """A written record's JSON keys must be EXACTLY LANE_RECORD_FIELDS (no
    extra/missing keys), `state` must be one of the lowercase LaneState
    values, and `updated_at` must parse as an ISO-8601 timestamp -- guards
    the adopt emitter's hand-encoded schema against drifting away from
    orchestrator/src/orchestrator/lane_lifecycle.py's LaneRecord (the
    embedded python3 heredoc re-encodes this schema rather than importing
    it -- see the plan's design decision -- so this test is the drift
    guard). When orchestrator IS importable (not the case under this
    suite's `--project shared` run, but defensive for any environment
    where it is), the record is additionally round-tripped through
    LaneRecord.from_json/to_dict for an exact match."""
    worktree_base = _build_worktree_base(tmp_path, [
        {"name": "_lane-7", "branch": "task/1234", "plan": {"title": "T"}},
    ])

    result = _run_script(worktree_base)
    assert result.returncode == 0, (
        f"Expected apply to exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    raw = _lane_state_path(worktree_base, "_lane-7").read_text()
    record = json.loads(raw)

    assert set(record.keys()) == LANE_RECORD_FIELDS, (
        f"Expected exactly {sorted(LANE_RECORD_FIELDS)!r} keys; "
        f"got {sorted(record.keys())!r}"
    )
    assert record["state"] in LANE_STATE_VALUES, (
        f"Expected state to be one of {sorted(LANE_STATE_VALUES)!r}; "
        f"got {record['state']!r}"
    )
    # Must parse as an ISO-8601 timestamp (raises ValueError otherwise).
    datetime.fromisoformat(record["updated_at"])

    if importlib.util.find_spec("orchestrator.lane_lifecycle") is not None:
        from orchestrator.lane_lifecycle import LaneRecord

        round_tripped = LaneRecord.from_json(raw)
        assert round_tripped.to_dict() == record, (
            f"Expected a lossless LaneRecord round-trip; "
            f"round_tripped.to_dict()={round_tripped.to_dict()!r} "
            f"record={record!r}"
        )
