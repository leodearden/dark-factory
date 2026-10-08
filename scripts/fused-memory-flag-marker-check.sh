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
# are harmless. It also means the verdict HERE compares before.total_source
# -- the WHOLE enumerated residual, since nothing is deleted -- so a ceiling
# tuned to the structural floor alone is not enough for this wrapper.
# NOTE that floor: part of the enumerated population cannot be drained by
# ANY invocation of the sweep, and since task 4436 the sweep CHECKS it
# (structural_floor.gate_unsatisfiable in the JSON, plus an ERROR naming the
# floor, the ceiling and the per-arm remedy) instead of merely warning. The
# arms, their remedies and the sufficient ceiling are stated once in
# docs/flag-marker-sweep-recurring.md -- don't restate them here.
# Historical narrative for the retired 2902 watch:
# plans/reify-flag-marker-backlog-rca-2026-07-22.md §6a.
#
# `uv` is resolved to an absolute path by
# scripts/lib/resolve_uv.sh::require_uv_bin rather than trusted to be on PATH
# (task 4591; see that lib for the ladder and the boot-catch-up cause). The
# call runs AFTER `source "$REPO/.env"` so a UV_BIN or PATH set in .env is
# honored -- the remedy an operator reaches for after a boot-PATH 127.
set -euo pipefail

case "${BASH_SOURCE[0]}" in */*) _self_dir="${BASH_SOURCE[0]%/*}" ;; *) _self_dir=. ;; esac
_uv_lib="$_self_dir/lib/resolve_uv.sh"
# shellcheck source=lib/resolve_uv.sh
source "$_uv_lib" || { echo "${0##*/}: ERROR: cannot load the shared uv resolver $_uv_lib -- refusing to guess which uv to run." >&2; exit 127; }

REPO="${REPO:-/home/leo/src/dark-factory}"
FM="$REPO/fused-memory"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"

if [ -n "${FLAG_MARKER_SWEEP_CMD:-}" ]; then
  # The documented test seam: an unquoted expansion so a multi-word prefix
  # word-splits into the array.
  # shellcheck disable=SC2206
  CHECK_CMD=(${FLAG_MARKER_SWEEP_CMD})
else
  UV_RESOLVED="$(require_uv_bin)" || exit $?
  # Built literally (not via a \${X:-...} default inside an unquoted array
  # expansion) so "$FM" survives verbatim even when the repo path contains
  # spaces.
  CHECK_CMD=("$UV_RESOLVED" run --frozen --project "$FM" python)
fi

exec "${CHECK_CMD[@]}" "$FM/scripts/sweep_orphan_flag_markers.py" --check "$@"
