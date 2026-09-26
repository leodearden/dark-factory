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
# Task 4591 -- `uv` is resolved to an ABSOLUTE path below rather than trusted
# to be on PATH: the same bare-`uv`-from-PATH pattern that killed
# fused-memory-flag-marker-sweep.sh with `exec: uv: not found` / status=127
# on its systemd unit's Persistent=true boot catch-up run, which fires
# before the login session pushes the user PATH into the systemd user
# manager. That matters more here than on either flag-marker wrapper: this
# script is WIRED as a before_done predicate action, so a 127 is a live
# predicate failure, not a latent one.
#
# CITATION STATE, so nobody chases a reference that looks stale: task 2917's
# fix to that sweep wrapper is PENDING on branch task/2917 (commit
# 2e74b9f51d) and is NOT an ancestor of main -- on main's lineage the sweep
# wrapper still carries the bare `uv run --frozen --project "$FM" python`
# default. Finding bare `uv` there means 2917 has not landed, not that the
# sweep wrapper regressed.
#
# UV_BIN overrides the resolution outright (also the test seam); otherwise
# `command -v uv` wins, then the measured real location
# ($HOME/.local/bin/uv), then /usr/local/bin. TWO distinct failures are
# reported LOUDLY (an ERROR: line) rather than left as a bare shell 127: an
# unresolvable `uv`, and a UV_BIN that is SET but not executable. The latter
# deliberately does NOT fall through to the rest of the ladder -- silently
# ignoring an explicit operator/test pin would run a DIFFERENT uv than the
# one named.
#
# ORDERING IS DELIBERATE (and was wrong in this file's first cut): the
# resolution runs AFTER `set -a; source "$REPO/.env"`, matching
# scripts/fused-memory-flag-marker-check.sh. Resolving BEFORE the source
# would ignore a PATH or UV_BIN set in .env -- which is precisely the remedy
# an operator would reach for after a minimal-boot-PATH 127, and it would
# have failed here while succeeding in the sibling wrapper. Keep the two in
# the same order.
#
# DUPLICATION, stated so the next copy-paste is at least deliberate:
# resolve_uv_bin() below is one of three verbatim copies of this ladder
# (UV_BIN -> PATH -> $HOME/.local/bin -> /usr/local/bin) -- the others are
# scripts/fused-memory-flag-marker-check.sh (same task) and, on branch
# task/2917, fused-memory-flag-marker-sweep.sh; scripts/sync-orchestrator-
# env.sh hardcodes a fourth, shorter spelling (UV=/home/leo/.local/bin/uv).
# A host-layout change -- uv moving to /opt, say -- must be found and
# applied in every one of them. Extracting a single sourceable
# scripts/lib/resolve_uv.sh is the right fix and is DEFERRED, not declined:
# it requires editing the sweep wrapper, which task 4591 holds no lock for
# and whose own copy is still on an unmerged branch. Filed as a follow-up.
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

# Resolved AFTER the .env source above, on purpose -- see ORDERING IS
# DELIBERATE in the header.
# rc=0 -> resolved (path on stdout); rc=2 -> UV_BIN set but not executable;
# rc=1 -> nothing found anywhere. rc=2 is split out rather than folded into
# the generic miss so the caller can name the bad pin in its diagnostic.
resolve_uv_bin() {
  # Order: explicit override, then PATH, then the two known install roots.
  # A SET-but-not-executable UV_BIN (typo, stale path, lost exec bit) stops
  # here with rc=2 -- it must never fall through to the ladder and quietly
  # run some other uv behind the operator's back.
  if [ -n "${UV_BIN:-}" ]; then
    if [ -x "${UV_BIN}" ]; then
      printf '%s' "${UV_BIN}"
      return 0
    fi
    return 2
  fi
  local from_path
  if from_path="$(command -v uv 2>/dev/null)" && [ -x "$from_path" ]; then
    printf '%s' "$from_path"
    return 0
  fi
  local candidate
  for candidate in "$HOME/.local/bin/uv" /usr/local/bin/uv; do
    if [ -x "$candidate" ]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

# `|| resolve_rc=$?` (not `if ! ...`) so the function's DISTINCT rc survives:
# inside an `if !` body $? is the negation's status, not the callee's.
resolve_rc=0
UV_BIN_RESOLVED="$(resolve_uv_bin)" || resolve_rc=$?
if [ "$resolve_rc" -eq 2 ]; then
  echo "cgl_eta_auto_apply.sh: ERROR: \$UV_BIN is set to '${UV_BIN}' but that path is not executable (bad path, or the exec bit is missing). Refusing to silently fall back to PATH / \$HOME/.local/bin/uv / /usr/local/bin/uv, because that would run a DIFFERENT \`uv\` than the one you pinned. Fix the path or unset UV_BIN." >&2
  exit 127
elif [ "$resolve_rc" -ne 0 ]; then
  echo "cgl_eta_auto_apply.sh: ERROR: cannot resolve \`uv\` -- not at \$UV_BIN (${UV_BIN:-unset}), not on PATH (${PATH}), and not at \$HOME/.local/bin/uv or /usr/local/bin/uv. This is the \`exec: uv: not found\` / status=127 boot-catch-up failure task 2917 observed on the sibling sweep wrapper (fix pending on branch task/2917). Install uv, or set UV_BIN to its absolute path." >&2
  exit 127
fi

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
