#!/usr/bin/env bash
# CGL-η auto-apply WRAPPER — the committed before_done=predicate action for the
# deterministic bulk-apply task (depends on task 2451). Sets up the fused-memory
# service environment (sourced .env + CONFIG/PROJECT roots), then runs the impl
# under `uv run` so fused_memory imports resolve. The impl's exit code is the
# predicate result: 0 == clean apply (task -> done); non-zero == escalate.
#
# Runbook lesson (ops scripts): must run under the SERVICE env, not a bare shell
# — source .env + PROJECT_ROOT + DASHBOARD_KNOWN_PROJECT_ROOTS or the census
# silently narrows. Idempotent: safe to re-run on predicate resume.
set -euo pipefail

REPO=/home/leo/src/dark-factory
FM="$REPO/fused-memory"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export DASHBOARD_KNOWN_PROJECT_ROOTS="${DASHBOARD_KNOWN_PROJECT_ROOTS:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"
# Fresh per-run stamp so predicate re-runs never clobber a prior run's artifacts.
export CGL_RUN_STAMP="${CGL_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

echo "[cgl-auto-apply] stamp=$CGL_RUN_STAMP config=$CONFIG_PATH"
exec uv run --project "$FM" python "$FM/scripts/cgl_eta_auto_apply_impl.py"
