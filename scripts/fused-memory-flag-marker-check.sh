#!/usr/bin/env bash
# fused-memory-flag-marker-check.sh -- read-only CHECK counterpart to
# fused-memory-flag-marker-sweep.sh, exposing sweep_orphan_flag_markers.py's
# --check/backlog_verdict predicate mode (task 2596) as a runnable
# before_done.script for task_kind='deterministic',
# before_done.kind='predicate' watch tasks. Sets up the same service env as
# the nightly sweep wrapper (a fused-memory maintenance action must run under
# the SERVICE env, not a bare shell, or the census silently narrows), then
# passes caller args through -- e.g. --project-id reify --max-backlog 0.
# Exit code only: 0 = residual source-tagged backlog within --max-backlog,
# 1 = violated. The VERDICT is the exit code; the orchestrator parses no
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
# signal that the 0 is "saw nothing", not "there was nothing" -- read it
# before treating an rc=0 here as a clean bill of health. Pass
# --fail-on-blind-spot (opt-in, default off) to turn an OBSERVED blind spot
# into rc=1; a failed census probe never trips it, so a transient Qdrant
# blip cannot flap the predicate. It only takes effect alongside --check,
# which this script's exec line already supplies -- the sweep rejects the
# flag without it. Do NOT wire it expecting the adjacent flag_for_stage2
# pool to reach zero: that pool is a healthy rolling window, so a gate keyed
# on its emptiness fails forever -- the same footgun as --max-backlog 0
# against undated markers, below. Dated census and full rationale live in
# docs/flag-marker-sweep-recurring.md (the single copy -- don't restate the
# numbers here).
#
# Unlike the nightly sweep wrapper this performs NO deletions (--check
# without --apply is a dry-run census + verdict), so resolve/resume re-runs
# are harmless. NOTE the sweep script's own caveat: undated markers can never
# be drained by age, so a --max-backlog 0 predicate against a population
# containing undated markers fails forever -- verify the census first or set
# the ceiling accordingly. First consumer: the esc-2866-1 O2 watch task; see
# plans/reify-flag-marker-backlog-rca-2026-07-22.md.
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
CHECK_CMD=(${FLAG_MARKER_SWEEP_CMD:-uv run --frozen --project "$FM" python})

exec "${CHECK_CMD[@]}" "$FM/scripts/sweep_orphan_flag_markers.py" --check "$@"
