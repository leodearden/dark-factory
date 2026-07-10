#!/usr/bin/env bash
set -euo pipefail

# deploy-w11-lane-lifecycle.sh
#
# ADOPT + RESTART deploy capstone for the W11 mechanism-1+2 batch
# (LaneLifecycle durable `.lane-state/<lane>.json` records + the
# `.task-meta` sibling-dir relocation -- plans/worktree-lane-lifecycle-prd.md,
# tasks alpha..iota / 2254-2262, all merged to main ahead of this script).
#
# INTENDED CALLER: a task_kind='deterministic' deploy task's
# `before_done.script`, with `target_unit` set to THIS orchestrator's OWN
# unit (orchestrator-dark-factory.service). That makes deterministic_runner
# take the self-restart path (deterministic_runner.py
# _default_schedule_detached_restart, docstring section epsilon): it
# schedules a detached `systemd-run --user` transient unit that re-runs this
# script OUT OF the orchestrator's cgroup, after run() returns -- so this
# script's own blocking restart (below) survives the very orchestrator
# process it restarts being killed.
#
# ORDERING (PRD "migration caution" / decomposition kappa): read git reality
# -> write .lane-state records ("adopt") -> THEN restart. Never the other
# order -- the new code must find seeded records already in place the
# moment it starts serving.
#
# cwd GOTCHA (2064/2105): the detached `systemd-run --user` unit defaults
# its cwd to $HOME, not project_root. This script must never rely on `pwd`/
# $PWD -- PROJECT_ROOT is derived below from ${BASH_SOURCE[0]} instead
# (before_done.cwd is still required to be the absolute project_root at
# filing time, defensively, but this script does not trust it).
#
# Usage:
#   deploy-w11-lane-lifecycle.sh [--check|--dry-run]
#
# --check / --dry-run: print the intended `.lane-state/` records and exit 0
# WITHOUT writing anything and WITHOUT restarting the orchestrator.
#
# Env overrides (test-only knobs; production runs with the defaults):
#   WORKTREE_DIR - worktree_dir; relative values are joined under
#                  PROJECT_ROOT, absolute values are used as-is (default:
#                  .worktrees, matching GitConfig.worktree_dir's pydantic
#                  default -- dark_factory does not override it).
#   SERVICE      - the orchestrator unit to restart+verify (default:
#                  orchestrator-dark-factory.service).

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WORKTREE_DIR="${WORKTREE_DIR:-.worktrees}"
if [[ "$WORKTREE_DIR" = /* ]]; then
    WORKTREE_BASE="$WORKTREE_DIR"
else
    WORKTREE_BASE="$PROJECT_ROOT/$WORKTREE_DIR"
fi

SERVICE="${SERVICE:-orchestrator-dark-factory.service}"

MODE="apply"

for arg in "$@"; do
    case "$arg" in
        --check|--dry-run)
            MODE="check"
            ;;
        *)
            echo "ERROR: unexpected argument: $arg" >&2
            exit 1
            ;;
    esac
done

adopt() {
    # Seed absent `.lane-state/<lane>.json` records from live git reality
    # (or, in --check/--dry-run mode, print what would be seeded without
    # writing). Embedded stdlib-only python3 (no `uv`/orchestrator venv
    # guaranteed in the detached systemd-run unit this script may run
    # inside -- PRD design decision). Schema source of truth:
    # orchestrator/src/orchestrator/lane_lifecycle.py (LaneRecord fields
    # exactly state/task_id/title/branch/seeded_from_sha/updated_at, state
    # persisted as the lowercase LaneState.value, LANE_STATE_DIRNAME=
    # '.lane-state') and orchestrator/src/orchestrator/artifacts.py
    # TaskArtifacts.meta_root_for -> <worktree_base>/.task-meta/<lane>/
    # plan.json (config.py TASK_META_DIRNAME).
    WORKTREE_BASE="$WORKTREE_BASE" MODE="$MODE" python3 - <<'PYEOF'
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

worktree_base = Path(os.environ["WORKTREE_BASE"])
mode = os.environ["MODE"]

TASK_BRANCH_RE = re.compile(r"^task/(\d+)$")


def _git(lane, *args):
    result = subprocess.run(
        ["git", "-C", str(lane), *args],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _plan_json_path(lane_dir, lane_name):
    new_path = worktree_base / ".task-meta" / lane_name / "plan.json"
    if new_path.is_file():
        return new_path
    legacy_path = lane_dir / ".task" / "plan.json"
    if legacy_path.is_file():
        return legacy_path
    return None


def _title_from_plan(plan_path):
    try:
        data = json.loads(plan_path.read_text())
    except (OSError, ValueError):
        return None
    title = data.get("title") or data.get("context")
    return title if isinstance(title, str) else None


def _write_record(lane_name, record):
    state_dir = worktree_base / ".lane-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=state_dir, prefix=f".{lane_name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp_path, state_dir / f"{lane_name}.json")
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


lane_dirs = sorted(p for p in worktree_base.glob("_lane-*") if p.is_dir()) \
    if worktree_base.is_dir() else []

for lane_dir in lane_dirs:
    lane_name = lane_dir.name

    if (worktree_base / ".lane-state" / f"{lane_name}.json").exists():
        # Never clobber an existing record: the restarted new-code
        # orchestrator becomes the authoritative writer once it starts
        # serving, so adopt only SEEDS records that are still absent --
        # making a re-run (or a lane a live orchestrator already wrote)
        # a total no-op.
        continue

    branch = _git(lane_dir, "symbolic-ref", "--short", "-q", "HEAD")
    sha = _git(lane_dir, "rev-parse", "-q", "--verify", "HEAD")

    m = TASK_BRANCH_RE.match(branch) if branch else None
    plan_path = _plan_json_path(lane_dir, lane_name) if m else None

    if m is not None and plan_path is not None:
        record = {
            "state": "assigned",
            "task_id": m.group(1),
            "title": _title_from_plan(plan_path),
            "branch": branch,
            "seeded_from_sha": sha or None,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    else:
        # Free/detached lane -- including a lane that RETAINS a task/<id>
        # branch (branch-retention 1912/1914) but has no plan.json, which
        # must NOT be mis-marked ASSIGNED (dodges the 2098 re-poisoning
        # class: only the plan.json discriminant makes a lane ASSIGNED).
        record = {
            "state": "registered",
            "task_id": None,
            "title": None,
            "branch": branch or None,
            "seeded_from_sha": sha or None,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    if mode == "check":
        print(f"{lane_name}: {json.dumps(record)}")
        continue

    _write_record(lane_name, record)
PYEOF
}

adopt

if [[ "$MODE" == "check" ]]; then
    exit 0
fi

# RESTART (apply mode only, AFTER adopt -- PRD migration caution: read
# reality -> write .lane-state records -> THEN flip). Delegate to the
# existing restart-orchestrator.sh rather than duplicating its blocking
# `systemctl --user restart` + fresh-ActiveEnterTimestampMonotonic verify
# loop (DRY). It already targets orchestrator-dark-factory.service.
exec "$PROJECT_ROOT/scripts/restart-orchestrator.sh"
