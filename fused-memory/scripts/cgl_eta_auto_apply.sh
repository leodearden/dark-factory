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
#
# `uv` is resolved to an absolute path by
# scripts/lib/resolve_uv.sh::require_uv_bin rather than trusted to be on PATH
# (task 4591; see that lib for the ladder and the boot-catch-up cause). That
# matters more here than in either flag-marker wrapper: this script is WIRED
# as a before_done predicate action, so a 127 is a live predicate failure.
#
# ORDERING IS DELIBERATE (and was wrong in this file's first cut): the
# require_uv_bin call runs AFTER `set -a; source "$REPO/.env"`, as in the
# flag-marker wrappers, so a PATH or UV_BIN set in .env -- the remedy an
# operator reaches for after a boot-PATH 127 -- is honored here too.
set -euo pipefail

case "${BASH_SOURCE[0]}" in */*) _self_dir="${BASH_SOURCE[0]%/*}" ;; *) _self_dir=. ;; esac
_uv_lib="$_self_dir/../../scripts/lib/resolve_uv.sh"
# shellcheck source=../../scripts/lib/resolve_uv.sh
source "$_uv_lib" || { echo "${0##*/}: ERROR: cannot load the shared uv resolver $_uv_lib -- refusing to guess which uv to run." >&2; exit 127; }

REPO=/home/leo/src/dark-factory
FM="$REPO/fused-memory"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export DASHBOARD_KNOWN_PROJECT_ROOTS="${DASHBOARD_KNOWN_PROJECT_ROOTS:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"

# After the .env source above, on purpose -- see ORDERING IS DELIBERATE.
UV_BIN_RESOLVED="$(require_uv_bin)" || exit $?

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
resume_schedulers() { "$UV_BIN_RESOLVED" run --project "$FM" python "$GATE" resume || true; }
trap resume_schedulers EXIT INT TERM

echo "[cgl-auto-apply] stamp=$CGL_RUN_STAMP config=$CONFIG_PATH"
"$UV_BIN_RESOLVED" run --project "$FM" python "$GATE" halt || true
# set -e: a non-zero impl exit terminates here (trap resumes schedulers, wrapper
# exits non-zero -> predicate escalates). The finalize line below is reached ONLY
# on a clean exit 0.
"$UV_BIN_RESOLVED" run --project "$FM" python "$FM/scripts/cgl_eta_auto_apply_impl.py"
# Clean apply only: auto-close the esc-2273-1 gate (best-effort; never flips the
# predicate verdict — a finalize miss leaves the L2 for the watcher/operator).
"$UV_BIN_RESOLVED" run --project "$FM" python "$FM/scripts/cgl_eta_finalize_gate.py" || true
# wrapper exits 0 -> predicate verdict = done (trap resumes schedulers first).
