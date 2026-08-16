#!/usr/bin/env bash
# memory-metadata-coverage-census.sh -- the committed nightly MEASURE-then-
# REHEARSE action for canonical/topic stamping coverage (task 4006). Invoked by
# scripts/memory-metadata-coverage-census.service's ExecStart (systemd
# .timer-driven, nightly 05:00).
#
# ONE CADENCE, TWO HALVES, IN SEQUENCE.
#
#   1. fused-memory/scripts/census_memory_metadata.py -- the GAUGE. A
#      deterministic paginated Qdrant count+scroll (never semantic search) that
#      regenerates plans/memory-metadata-census-report.{json,md} and APPENDS one
#      compact scalar row to plans/memory-metadata-coverage-history.json, which
#      is what turns a snapshot into a trend.
#
#   2. fused-memory/scripts/retro_stamp_topics.py -- the MECHANISM that closes
#      the gap the gauge just measured (PRD leaf θ, task 3201), run as a full
#      rehearsal.
#
# Pairing them on one timer is the point of the job: an operator reads the gap
# and the stamping that would close it in the SAME journal entry, instead of a
# number nobody is accountable for. The census runs FIRST so the rehearsal is
# read against a report generated moments before rather than last night's.
#
# THE REHEARSAL IS NEVER `--apply`. 3201 writes `canonical: true` in bulk. Under
# the shipped `memory_metadata.enforce: false` default those writes are not even
# rejected by 3198's write-time per-(project,topic) uniqueness check -- it
# censuses the violation and lets the write through -- so an unattended
# `--apply` could manufacture exactly the violations this census exists to
# count. Committing the stamps stays an operator decision:
#
#   uv run --frozen --project fused-memory python \
#       fused-memory/scripts/retro_stamp_topics.py --apply
#
# NEITHER HALF CAN ABORT THE OTHER, AND THE WRAPPER ALWAYS EXITS 0. A recurring
# `oneshot` that can fail enters systemd `failed` state and STAYS there,
# silently stopping the whole nightly job (the lesson already written into
# scripts/reify-closure-staleness-sweep.sh). That matters more here than for
# most jobs: the census exits 1 BY DESIGN whenever `coverage.complete` is false,
# which on a live corpus that orchestrators write to during the scroll is a
# routine outcome, not a fault. Propagating it would wedge the timer on the
# first churny night and quietly end the trend -- while the artifacts, which
# carry the evidence of the shortfall, were written anyway. Both exit codes are
# narrated instead, so exiting 0 never means the failure is invisible.
#
# RUNS UNDER THE SERVICE ENV (sourced .env + CONFIG_PATH / PROJECT_ROOT /
# FALKORDB_URI roots) and under `uv run` so `fused_memory.*` imports resolve --
# mirrors scripts/fused-memory-flag-marker-sweep.sh and the
# fused-memory/scripts/cgl_eta_auto_apply.sh runbook lesson: a fused-memory
# maintenance action must run under the SERVICE env, not a bare shell, or the
# census silently narrows to a different config and censuses a different
# collection than the one the artifacts claim.
#
# COVERAGE_CENSUS_CMD / RETRO_STAMP_CMD override the two interpreter prefixes
# (tests inject fake recorders here to assert argument construction and
# ordering without touching uv, live Qdrant or the live corpus) -- mirrors the
# FLAG_MARKER_SWEEP_CMD / REIFY_CLOSURE_SWEEP_CMD seam convention. REPO is
# similarly overridable so tests can point it at a tmp dir with no `.env` (a
# no-op source).
#
# `set -e` is deliberately NOT set: both halves are allowed to fail and are
# graded explicitly below.
set -uo pipefail

REPO="${REPO:-/home/leo/src/dark-factory}"
FM="$REPO/fused-memory"

set -a
# shellcheck disable=SC1091
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"

# --top-n 400 matches the committed artifact's recorded params, so a nightly
# regeneration diffs as CONTENT drift and never as a rendering-width change.
CENSUS_TOP_N="${CENSUS_TOP_N:-400}"

# shellcheck disable=SC2206
CENSUS_CMD=(${COVERAGE_CENSUS_CMD:-uv run --frozen --project "$FM" python})
# shellcheck disable=SC2206
STAMP_CMD=(${RETRO_STAMP_CMD:-uv run --frozen --project "$FM" python})

echo "memory-metadata-coverage-census: censusing $REPO (top-n $CENSUS_TOP_N)"
"${CENSUS_CMD[@]}" "$FM/scripts/census_memory_metadata.py" --top-n "$CENSUS_TOP_N"
census_rc=$?
if [ "$census_rc" -ne 0 ]; then
    echo "memory-metadata-coverage-census: census exited $census_rc — the" \
         "artifacts still carry the evidence; rehearsing the retro stamp anyway" >&2
fi

echo "memory-metadata-coverage-census: rehearsing the retro topic/canonical stamp (DRY RUN)"
"${STAMP_CMD[@]}" "$FM/scripts/retro_stamp_topics.py"
stamp_rc=$?
if [ "$stamp_rc" -ne 0 ]; then
    echo "memory-metadata-coverage-census: retro-stamp rehearsal exited $stamp_rc" >&2
fi

echo "memory-metadata-coverage-census: done (census=$census_rc stamp=$stamp_rc)"
# Always 0 — see the oneshot rationale in the header.
exit 0
