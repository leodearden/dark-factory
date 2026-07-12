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

# Quiet the two big target graphs' write load: halt BOTH the reify and
# dark_factory orchestrator schedulers for the duration, and ALWAYS resume them
# on exit (any path but SIGKILL) via a trap. Halt is best-effort/non-fatal — the
# apply is safe under load regardless (see cgl_eta_scheduler_gate.py); this is
# defence-in-depth. If a hard SIGKILL (predicate timeout) skips the trap, the
# born-at-L2 timeout escalation surfaces the still-halted schedulers for manual
# resume. NOTE: this halts the dark_factory scheduler that dispatched THIS task
# too — safe, because this deterministic task is already dispatched and running;
# the halt only withholds OTHER tasks, and resume lands before the runner reads
# our exit code.
GATE="$FM/scripts/cgl_eta_scheduler_gate.py"
resume_schedulers() { uv run --project "$FM" python "$GATE" resume || true; }
trap resume_schedulers EXIT INT TERM

echo "[cgl-auto-apply] stamp=$CGL_RUN_STAMP config=$CONFIG_PATH"
uv run --project "$FM" python "$GATE" halt || true
# set -e: a non-zero impl exit terminates here (trap resumes schedulers, wrapper
# exits non-zero -> predicate escalates). The finalize line below is reached ONLY
# on a clean exit 0.
uv run --project "$FM" python "$FM/scripts/cgl_eta_auto_apply_impl.py"
# Clean apply only: auto-close the esc-2273-1 gate (best-effort; never flips the
# predicate verdict — a finalize miss leaves the L2 for the watcher/operator).
uv run --project "$FM" python "$FM/scripts/cgl_eta_finalize_gate.py" || true
# wrapper exits 0 -> predicate verdict = done (trap resumes schedulers first).
