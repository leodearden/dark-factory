#!/usr/bin/env bash
# reclaim-orphaned-worktrees.sh -- the committed nightly reclaim action for
# .worktrees-orphaned/ parkings (task 2980). Invoked by
# scripts/reclaim-orphaned-worktrees.service's ExecStart (systemd .timer-driven,
# nightly) and directly by scripts/install-reclaim-orphaned-worktrees-timer.sh
# for the immediate one-time drain of the current backlog.
#
# Unlike scripts/fused-memory-flag-marker-sweep.sh, the reclaim script is
# STDLIB-ONLY: it does NOT import fused_memory/OrchestratorConfig, so this
# wrapper runs `python3 <script> --repo "$REPO"` DIRECTLY -- no `uv run --frozen
# --project`, no `.env` sourcing, and no CONFIG_PATH/PROJECT_ROOT/FALKORDB_URI
# service-env exports. That decoupling is the whole point of the standalone
# design (a bug here can't wedge the orchestrator, and it runs in any env).
#
# RECLAIM_ORPHANED_WT_CMD overrides the default `python3` interpreter prefix
# (tests inject a fake recorder here to assert the reclaim script is invoked
# correctly, without touching real worktrees) -- mirrors the flag-marker
# wrapper's FLAG_MARKER_SWEEP_CMD seam. REPO is similarly overridable so tests
# can point it at a tmp dir; it is passed through as --repo so the reclaim runs
# against that repo's <REPO>/.worktrees-orphaned quarantine base.
set -euo pipefail

REPO="${REPO:-/home/leo/src/dark-factory}"

# shellcheck disable=SC2206
CMD=(${RECLAIM_ORPHANED_WT_CMD:-python3})

exec "${CMD[@]}" "$REPO/scripts/reclaim_orphaned_worktrees.py" --repo "$REPO"
