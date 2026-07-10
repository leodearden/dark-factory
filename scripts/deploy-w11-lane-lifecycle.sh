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
    # writing). Stub for now -- filled in starting step-4.
    :
}

adopt

if [[ "$MODE" == "check" ]]; then
    exit 0
fi
