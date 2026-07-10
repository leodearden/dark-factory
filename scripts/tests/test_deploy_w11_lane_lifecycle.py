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

import json
import os
import subprocess
from pathlib import Path

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
    default.

    Systemctl invocations are inspectable afterward via
    `_systemctl_calls(worktree_base)`.
    """
    tmp_path = worktree_base.parent
    bin_dir, state_path = _fake_systemctl(tmp_path)

    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    full_env["WORKTREE_DIR"] = str(worktree_base)
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
