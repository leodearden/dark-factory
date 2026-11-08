#!/usr/bin/env bash
# phase3.sh: the third pass on a bench pod, detached; leaves results/PHASE3_DONE or
# results/PHASE3_FAILED.
#   1. the two single-threaded probes again, on a now-quiet disk, into
#      probes.rerun.jsonl (the first set ran right after GBs of writes)
#   2. the reify gate, cold then warm, as an unprivileged user (reify_as_user.sh),
#      under eatmydata, without the infra pool (reify-spec-noinfra.json)
set -euo pipefail

export BENCH=$HOME/df-hwbench
export RESULTS=$BENCH/results
export QUICK=0 REFERENCE=0
export PATH=$HOME/.local/bin:$HOME/.local/node/bin:$HOME/.cargo/bin:$PATH
unset VIRTUAL_ENV
log() { echo "[phase3 $(date -u +%T)] $*"; }
trap 'rc=$?; ((rc == 0)) && touch "$RESULTS/PHASE3_DONE" || echo "rc=$rc" >"$RESULTS/PHASE3_FAILED"' EXIT
rm -f "$RESULTS/PHASE3_DONE" "$RESULTS/PHASE3_FAILED"

log "probes again on a quiet disk"
ONLY='py_collect pyright' PROBES_OUT=$RESULTS/probes.rerun.jsonl bash "$BENCH/bin/probes.sh"

SPEC_FILE=$BENCH/reify-spec-noinfra.json GATE_WRAPPER=eatmydata bash "$BENCH/bin/reify_as_user.sh"
log "finished"
