#!/usr/bin/env bash
# fused-memory-flag-marker-check.sh -- read-only CHECK counterpart to
# fused-memory-flag-marker-sweep.sh, exposing sweep_orphan_flag_markers.py's
# --check/backlog_verdict predicate mode (task 2596).
#
# THE WATCH GATE IS RETIRED (task 3923): task 2902, the only before_done
# consumer this wrapper ever had, is done, so its remaining role is an
# ad-hoc, read-only census run by hand. Ruling, dated census and the
# remediation path if you trip the armed verdict below:
# docs/flag-marker-sweep-recurring.md §"Decision (task 3923)" -- the single
# copy, don't restate it here.
#
# Sets up the same service env as the nightly sweep wrapper (a fused-memory
# maintenance action must run under the SERVICE env, not a bare shell, or the
# census silently narrows), then passes caller args through -- e.g.
# --project-id reify --max-backlog 0.
# Exit code only: 0 = verdict holds; 1 = EITHER the residual source-tagged
# backlog exceeds --max-backlog OR an OBSERVED enumeration blind spot vetoed
# the verdict (see below). Both causes share rc=1 -- the emitted JSON's
# cross_check block is what distinguishes them, so a consumer that needs to
# tell them apart must read it rather than infer from the code alone. (A
# distinct code for the veto was considered and declined: rc is
# sweep_orphan_flag_markers.py's own, and the orchestrator's before_done
# path renders any rc!=0 identically as a predicate violation, so a third
# code would buy separability nowhere it is consumed.)
# The VERDICT is the exit code; the orchestrator parses no
# output to reach it. It does, however, READ the output: on rc=0 it extracts
# a bounded structured summary (a trailing JSON block, else one clean final
# line, log lines dropped) into done_provenance.note, and on rc!=0 it carries
# the output verbatim in the escalation detail. This header previously
# asserted the runner "reads no output" -- that stale belief is precisely why
# task 2902's server-log noise reached a task note and, from there, memory
# (task 3286). Emit a trailing JSON object or one clean final line if you
# want this script's verdict preserved.
#
# That extracted JSON now carries a "cross_check" block (task 3897), so an
# enumeration blind spot is PRESERVED in done_provenance.note rather than
# lost. This matters here specifically: the --check predicate reads
# before.total_source, which the source filter measures at 0 in every
# project probed, so backlog_verdict holds unconditionally and this gate
# structurally cannot fail on its own. cross_check.blind_spot=true is the
# signal that the 0 is "saw nothing", not "there was nothing".
#
# Since task 3923 you no longer have to remember to read it: an OBSERVED
# blind spot is rc=1 BY DEFAULT. Because this script's exec line hardcodes
# --check, that applies to every invocation here -- so re-wiring this
# wrapper as a before_done predicate today FAILS LOUDLY ON DAY ONE rather
# than passing silently forever, which is the point. The remedy is to fix
# the source/kind enumeration so it sees the real marker population BEFORE
# wiring a gate on it: --no-fail-on-blind-spot is census-only (it relaxes
# the vacuity check, never the backlog ceiling) and must not be used as a
# gate configuration, since silencing the veto restores the vacuous pass.
# --fail-on-blind-spot is still accepted as an explicit affirmation of the
# default. A failed census probe never trips the escalation, so a transient
# Qdrant blip cannot flap the verdict.
#
# Do NOT re-point this at the adjacent flag_for_stage2 pool expecting it to
# reach zero: that pool is a healthy rolling window, so a gate keyed on its
# emptiness fails forever -- the same footgun as --max-backlog 0 against
# undated markers, below.
#
# Unlike the nightly sweep wrapper this performs NO deletions (--check
# without --apply is a dry-run census + verdict), so resolve/resume re-runs
# are harmless. NOTE the sweep script's own caveat: undated markers can never
# be drained by age, so a --max-backlog 0 predicate against a population
# containing undated markers fails forever -- verify the census first or set
# the ceiling accordingly. Historical narrative for the retired 2902 watch:
# plans/reify-flag-marker-backlog-rca-2026-07-22.md §6a.
#
# Task 4591 -- `uv` is resolved to an ABSOLUTE path below rather than trusted
# to be on PATH. The failure this prevents was OBSERVED on the sibling
# nightly-drain wrapper (fused-memory-flag-marker-sweep.sh), which died
# `exec: uv: not found` / status=127 on its systemd unit's Persistent=true
# boot catch-up run -- that run fires before the login session pushes the
# user PATH into the systemd user manager.
#
# CITATION STATE, so nobody chases a reference that looks stale: task 2917's
# fix to that sweep wrapper is PENDING on branch task/2917 (commit
# 2e74b9f51d) and is NOT an ancestor of main. On main's lineage
# fused-memory-flag-marker-sweep.sh still carries the bare
# `uv run --frozen --project "$FM" python` default. So if you follow the
# citation to inspect the pattern being mirrored and find bare `uv` there,
# the sweep wrapper has not regressed -- 2917 simply has not landed yet.
#
# This wrapper is not currently wired to any systemd unit or before_done
# predicate (the watch gate is retired, see above), so the defect here is
# LATENT -- but whoever re-wires it inherits the identical failure mode
# unless it is fixed now. UV_BIN overrides the resolution outright (the test
# seam); otherwise `command -v uv` wins, then the measured real location
# ($HOME/.local/bin/uv), then /usr/local/bin. TWO distinct failures are
# reported LOUDLY (an ERROR: line naming uv, the PATH searched, and the
# boot-catch-up cause) rather than left as a bare shell 127 that says
# nothing about why: an unresolvable `uv`, and a UV_BIN that is SET but not
# executable. The latter deliberately does NOT fall through to the rest of
# the ladder -- silently ignoring an explicit operator/test pin would run a
# DIFFERENT uv than the one named, which is the exact silent degradation
# this resolution exists to prevent.
#
# DUPLICATION, stated so the next copy-paste is at least deliberate:
# resolve_uv_bin() below is the THIRD copy of this ladder (UV_BIN -> PATH ->
# $HOME/.local/bin -> /usr/local/bin). Copy #1 lives on branch task/2917 in
# fused-memory-flag-marker-sweep.sh; copy #2 is
# fused-memory/scripts/cgl_eta_auto_apply.sh (added by this same task); and
# scripts/sync-orchestrator-env.sh hardcodes a fourth, shorter spelling
# (UV=/home/leo/.local/bin/uv). A host-layout change -- uv moving to /opt,
# say -- must be found and applied in every one of them. Extracting a single
# sourceable scripts/lib/resolve_uv.sh is the right fix and is DEFERRED, not
# declined: it requires editing the sweep wrapper, which this task holds no
# lock for and whose own copy is still on an unmerged branch. Filed as a
# follow-up rather than done here.
set -euo pipefail

REPO="${REPO:-/home/leo/src/dark-factory}"
FM="$REPO/fused-memory"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"

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

if [ -n "${FLAG_MARKER_SWEEP_CMD:-}" ]; then
  # The documented test seam: an unquoted expansion so a multi-word prefix
  # word-splits into the array.
  # shellcheck disable=SC2206
  CHECK_CMD=(${FLAG_MARKER_SWEEP_CMD})
else
  # `|| resolve_rc=$?` (not `if ! ...`) so the function's DISTINCT rc survives:
  # inside an `if !` body $? is the negation's status, not the callee's.
  resolve_rc=0
  UV_RESOLVED="$(resolve_uv_bin)" || resolve_rc=$?
  if [ "$resolve_rc" -eq 2 ]; then
    echo "fused-memory-flag-marker-check.sh: ERROR: \$UV_BIN is set to '${UV_BIN}' but that path is not executable (bad path, or the exec bit is missing). Refusing to silently fall back to PATH / \$HOME/.local/bin/uv / /usr/local/bin/uv, because that would run a DIFFERENT \`uv\` than the one you pinned. Fix the path or unset UV_BIN." >&2
    exit 127
  elif [ "$resolve_rc" -ne 0 ]; then
    echo "fused-memory-flag-marker-check.sh: ERROR: cannot resolve \`uv\` -- not at \$UV_BIN (${UV_BIN:-unset}), not on PATH (${PATH}), and not at \$HOME/.local/bin/uv or /usr/local/bin/uv. This is the \`exec: uv: not found\` / status=127 boot-catch-up failure task 2917 observed on the sibling sweep wrapper (fix pending on branch task/2917). Install uv, or set UV_BIN to its absolute path." >&2
    exit 127
  fi
  # Built literally (not via a \${X:-...} default inside an unquoted array
  # expansion) so "$FM" survives verbatim even when the repo path contains
  # spaces.
  CHECK_CMD=("$UV_RESOLVED" run --frozen --project "$FM" python)
fi

exec "${CHECK_CMD[@]}" "$FM/scripts/sweep_orphan_flag_markers.py" --check "$@"
