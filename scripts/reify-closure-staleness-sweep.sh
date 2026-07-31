#!/usr/bin/env bash
# reify-closure-staleness-sweep.sh -- the committed nightly sweep-then-consume
# action for reify's deterministic-gate closure-staleness audit (task 3102).
# Invoked by scripts/reify-closure-staleness-sweep.service's ExecStart
# (systemd .timer-driven, nightly 04:30).
#
# THE CROSS-REPO SEAM. reify task 5321 ships the PRIMITIVE:
# scripts/deterministic-gate-closure-staleness-sweep.sh, which is READ-ONLY on
# all task state (its invariant L6) and adjudicates stranded rows into request
# files under --emit-requests. dark-factory wires the INVOCATION -- this
# wrapper -- and performs the actual writes through the fused-memory MCP
# server, so every status transition goes through the reconciliation-triggering
# path CLAUDE.md requires. Neither half re-implements the other: the consumer
# reads the emitted files and never re-derives the sweep's predicates.
#
# WHY THE CADENCE IS THE TIMER'S, NOT A CONFIG KNOB'S. reify's sweep docs ask
# for a "standing-audit cadence"; in this repo that is the nightly systemd
# .timer family (legibility-trickle 03:00, fused-memory-flag-marker-sweep
# 03:30, reclaim-orphaned-worktrees + legibility-transcript-check 04:00, this
# job 04:30), each carrying its schedule in its own OnCalendar. Putting the
# cadence in dark-factory-orchestrator.yaml instead would be misplaced on the
# merits -- the orchestrator process is not the thing being scheduled -- and
# expensive in practice: that file is tracked in the reify repo, loads once at
# startup with no hot-reload, and is guarded by a dirty-start check, so a
# re-cadence would cost a cross-repo commit plus a fleet redeploy instead of a
# one-line unit edit and a re-run of the installer.
#
# ONE SERVICE, TWO HALVES, IN SEQUENCE. The brief asks for one recurring job
# rather than two parallel mechanisms. Sequencing here also gets the ordering
# right for free: the consumer drains a directory the sweep has JUST refreshed
# (and just retracted stale entries from), so it never acts on last night's
# snapshot. Two separately-timed units would need After=/Requires= plumbing to
# achieve the same and would still race on a caught-up Persistent=true firing.
#
# NEITHER HALF CAN ABORT THE OTHER, and the wrapper always exits 0. A recurring
# `oneshot` that can fail enters systemd `failed` state and STAYS there --
# silently stopping the whole nightly job (the lesson already written into
# scripts/fused-memory-flag-marker-sweep.sh). A sweep fault must not swallow
# the drain of requests emitted on a previous night, and a consumer fault must
# not mask the sweep's report in the journal. Both exit codes are reported
# instead.
#
# STDLIB-ONLY, like scripts/reclaim-orphaned-worktrees.sh: no `.env` sourcing,
# no CONFIG_PATH/PROJECT_ROOT/FALKORDB_URI exports, no `uv run --frozen
# --project`. A nightly timer that can fail on a lockfile or venv-resolution
# problem is a job that silently stops running.
#
# REIFY_CLOSURE_SWEEP_CMD / REIFY_REDISPATCH_CONSUMER_CMD override the two
# interpreter prefixes (tests inject fake recorders here to assert argument
# construction and ordering without touching the real sweep, the live task
# store, or systemd) -- mirrors the FLAG_MARKER_SWEEP_CMD /
# RECLAIM_ORPHANED_WT_CMD seam convention. REIFY_REPO and
# REIFY_GATE_STALENESS_REQUESTS_DIR are similarly overridable so tests can
# point at a tmp dir; the latter is the sweep's OWN env var name, so exporting
# it and passing it explicitly agree.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # dark-factory scripts/
REIFY_REPO="${REIFY_REPO:-/home/leo/src/reify}"

# Under reify's EXISTING data/ dir on purpose: the sweep creates the requests
# dir on demand only when its PARENT already exists, a deliberate guard so a
# typo'd --emit-requests surfaces as a warning rather than hiding as
# "0 requests emitted".
REQUESTS_DIR="${REIFY_GATE_STALENESS_REQUESTS_DIR:-$REIFY_REPO/data/redispatch-requests}"

SWEEP_SCRIPT="$REIFY_REPO/scripts/deterministic-gate-closure-staleness-sweep.sh"
CONSUMER_SCRIPT="$SCRIPT_DIR/consume_redispatch_requests.py"

# shellcheck disable=SC2206
SWEEP_CMD=(${REIFY_CLOSURE_SWEEP_CMD:-bash})
# shellcheck disable=SC2206
CONSUMER_CMD=(${REIFY_REDISPATCH_CONSUMER_CMD:-python3})

echo "reify-closure-staleness-sweep: sweeping $REIFY_REPO -> $REQUESTS_DIR"
"${SWEEP_CMD[@]}" "$SWEEP_SCRIPT" --emit-requests "$REQUESTS_DIR"
sweep_rc=$?
if [ "$sweep_rc" -ne 0 ]; then
    echo "reify-closure-staleness-sweep: sweep exited $sweep_rc — draining any" \
         "requests already on disk anyway" >&2
fi

echo "reify-closure-staleness-sweep: consuming requests in $REQUESTS_DIR"
"${CONSUMER_CMD[@]}" "$CONSUMER_SCRIPT" \
    --requests-dir "$REQUESTS_DIR" \
    --project-root "$REIFY_REPO"
consumer_rc=$?
if [ "$consumer_rc" -ne 0 ]; then
    echo "reify-closure-staleness-sweep: consumer exited $consumer_rc" >&2
fi

echo "reify-closure-staleness-sweep: done (sweep=$sweep_rc consumer=$consumer_rc)"
# Always 0 — see the oneshot rationale in the header.
exit 0
