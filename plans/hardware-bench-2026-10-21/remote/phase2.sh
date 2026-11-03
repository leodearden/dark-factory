#!/usr/bin/env bash
# phase2.sh: the second pass on a bench pod, after run.sh. Detached like run.sh;
# leaves results/PHASE2_DONE or results/PHASE2_FAILED.
#   1. set aside probe batches with a non-zero exit (probes.failed.jsonl) and rerun
#      the reify compile probes, which need the tree-sitter CLI reify_gate.sh installs
#   2. diag.sh: user vs system CPU of a pytest collection, and a pure-interpreter loop
#   3. DF gates, 1 then GATE_CONCURRENCY at once, under eatmydata (fsync becomes a
#      no-op, so the pod's disk does not decide the result) with spec-deselect.json
#      (the live spec minus one test that hangs to its 300 s timeout on some hosts)
#   4. the reify gate, cold then warm, under eatmydata with reify-spec-noinfra.json
#      (REIFY_INFRA_SUITE_ACTIVE=1: verify.sh omits the infra pool and its release
#      prebuilds, whose shell tests cannot pass as root in a container and whose
#      failure skips every later step); an earlier attempt moves to reify-run1/
#   env: DF_SHA REIFY_STAGE [GATE_CONCURRENCY=2]
set -euo pipefail

export BENCH=$HOME/df-hwbench
export RESULTS=$BENCH/results
: "${DF_SHA:?}" "${GATE_CONCURRENCY:=2}"
export DF_SHA QUICK=0 REFERENCE=0
export PATH=$HOME/.local/bin:$HOME/.local/node/bin:$HOME/.cargo/bin:$PATH
unset VIRTUAL_ENV
log() { echo "[phase2 $(date -u +%T)] $*"; }
trap 'rc=$?; ((rc == 0)) && touch "$RESULTS/PHASE2_DONE" || echo "rc=$rc" >"$RESULTS/PHASE2_FAILED"' EXIT
rm -f "$RESULTS/PHASE2_DONE" "$RESULTS/PHASE2_FAILED"

command -v eatmydata >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq eatmydata >/dev/null

log "setting aside failed probe batches"
python3 -I - "$RESULTS/probes.jsonl" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
batches = [json.loads(line) for line in path.read_text().splitlines()]
failed = [b for b in batches if any(r['rc'] != 0 for r in b['runs'])]
with open(path.with_name('probes.failed.jsonl'), 'a') as f:
    f.writelines(json.dumps(b) + '\n' for b in failed)
path.write_text(''.join(json.dumps(b) + '\n' for b in batches if b not in failed))
print(f'set aside {len(failed)}: {sorted({b["probe"] for b in failed})}')
PY

log "reify compile probes (tree-sitter: $(command -v tree-sitter))"
ONLY='rustc_j1 rustc_jn' bash "$BENCH/bin/probes.sh"

bash "$BENCH/bin/diag.sh"

export GATE_SPEC=$BENCH/spec-deselect.json GATE_WRAPPER=eatmydata GATE_LABEL=eatmydata-deselect
bash "$BENCH/bin/gate.sh" 1
bash "$BENCH/bin/gate.sh" "$GATE_CONCURRENCY"

log "reify gates without the infra pool"
mkdir -p "$RESULTS/reify-run1"
for f in "$RESULTS"/reify-* "$RESULTS"/reify_*; do
	if [[ -e $f && $f != "$RESULTS/reify-run1" ]]; then
		mv "$f" "$RESULTS/reify-run1/"
	fi
done
SPEC_FILE=$BENCH/reify-spec-noinfra.json STAGE=${REIFY_STAGE:?} bash "$BENCH/bin/reify_gate.sh" all
log "finished"
