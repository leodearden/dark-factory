#!/usr/bin/env bash
# fused-memory-flag-marker-sweep.sh -- the committed nightly DRAIN action for
# stage1_flag_marker dead-weight records (task 2693, follow-up to task 2596's
# previously-unwired sweep). Invoked by
# scripts/fused-memory-flag-marker-sweep.service's ExecStart (systemd
# .timer-driven, nightly) and directly by
# scripts/install-flag-marker-sweep-timer.sh for the immediate one-time
# drain of the current backlog.
#
# Sets up the fused-memory service environment (sourced .env + CONFIG_PATH /
# PROJECT_ROOT / FALKORDB_URI roots) then runs sweep_orphan_flag_markers.py
# under `uv run` so `fused_memory.*` imports resolve -- mirrors the
# fused-memory/scripts/cgl_eta_auto_apply.sh runbook lesson: a fused-memory
# maintenance action must run under the SERVICE env, not a bare shell, or
# the census silently narrows.
#
# Runs `--apply --terminal-drain` WITHOUT `--check` on purpose: the sweep's
# own docstring/WARNING notes undated markers can never be drained by
# find_stale_markers, so a `--check --max-backlog 0` recurring service would
# enter systemd `failed` state forever whenever any undated marker exists.
# Backlog visibility is instead left to the existing reconciliation
# Stage-1/2 re-flag net (the mechanism that filed this task).
#
# FLAG_MARKER_SWEEP_CMD overrides the default `uv run --frozen --project
# $FM python` interpreter prefix (tests inject a fake recorder here to
# assert the sweep is invoked correctly, without uv/live stores) --
# mirrors install-trickle-timer.sh's INSTALL_TRICKLE_TIMER_PYTHON /
# watcher-rearm.sh's WATCHER_REARM_PYTHON override convention. REPO is
# similarly overridable so tests can point it at a tmp dir with no `.env`
# (a no-op source).
set -euo pipefail

REPO="${REPO:-/home/leo/src/dark-factory}"
FM="$REPO/fused-memory"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"

# shellcheck disable=SC2206
SWEEP_CMD=(${FLAG_MARKER_SWEEP_CMD:-uv run --frozen --project "$FM" python})

exec "${SWEEP_CMD[@]}" "$FM/scripts/sweep_orphan_flag_markers.py" --apply --terminal-drain
