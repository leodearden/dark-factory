#!/usr/bin/env bash
# gate.sh <concurrency>: run that many real dark-factory merge gates at once,
# each `orchestrator verify-merge` in its own project clone, exactly as a
# verify host runs one for the dispatcher (spec built by make_spec.py).
# Appends one JSON line to $RESULTS/gates.jsonl.
set -euo pipefail

: "${BENCH:?}" "${RESULTS:?}" "${DF_SHA:?}"
concurrency=$1
ORCH=$BENCH/df-0/.venv/bin/orchestrator
# GATE_SPEC overrides the spec file; GATE_WRAPPER (e.g. eatmydata) prefixes each
# verify-merge; GATE_LABEL (default: the wrapper) keeps such batches apart.
spec=$(cat "${GATE_SPEC:-$BENCH/spec.json}")
label=${GATE_LABEL:-${GATE_WRAPPER:-}}
batch=$RESULTS/gate-c$concurrency${label:+-$label}
mkdir -p "$batch"

log() { echo "[gate $(date -u +%T)] $*"; }

slot_config() { # <slot dir>
	cat <<-YAML
		project_root: "$1"
		git:
		  main_branch: "main"
		  branch_prefix: "task/"
		  worktree_dir: ".worktrees"
		  persistent_merge_worktree: false
		  remote: "origin"
		fused_memory:
		  url: "http://127.0.0.1:8002"
		  project_id: "dark_factory"
		  server_command: []
		escalation:
		  queue_dir: "data/escalations"
		  port: 8102
		merge_verify_max_concurrent_modules: 1
		verify_use_cgroup_scope: false
		verify_cold_preprovision_command: "uv sync --all-packages && npm ci --no-audit --no-fund --prefer-offline"
	YAML
}

pids=()
for ((i = 1; i <= concurrency; i++)); do
	slot=$BENCH/gate-slot-$i
	if [[ ! -d $slot ]]; then
		git clone -q "$BENCH/df.git" "$slot"
	fi
	git -C "$slot" worktree prune
	slot_config "$slot" >"$slot.yaml"
done

log "starting $concurrency gate(s) at $DF_SHA"
t0=$(date +%s.%N)
for ((i = 1; i <= concurrency; i++)); do
	/usr/bin/time -q -f '%e %x' -o "$batch/time.$i" \
		${GATE_WRAPPER:-} "$ORCH" verify-merge --sha "$DF_SHA" --spec "$spec" --config "$BENCH/gate-slot-$i.yaml" \
		>"$batch/result.$i.json" 2>"$batch/stderr.$i.log" &
	pids+=($!)
done
wait "${pids[@]}" || true
t1=$(date +%s.%N)

python3 -I - "$concurrency" "$t0" "$t1" "$batch" >>"$RESULTS/gates.jsonl" <<'PY'
import json, os, pathlib, sys
concurrency, t0, t1, d = sys.argv[1:]
d = pathlib.Path(d)
gates = []
for i in range(1, int(concurrency) + 1):
    wall, rc = (d / f'time.{i}').read_text().split()[-2:]
    try:
        verdict = json.loads((d / f'result.{i}.json').read_text())
    except (OSError, ValueError):
        verdict = {}
    gates.append({
        'wall_s': float(wall), 'rc': int(rc),
        'passed': verdict.get('passed'), 'timed_out': verdict.get('timed_out'),
        'summary': (verdict.get('summary') or '')[:400],
    })
print(json.dumps({'concurrency': int(concurrency), 'wrapper': os.environ.get('GATE_WRAPPER', ''),
                  'label': d.name, 'spec': os.environ.get('GATE_SPEC', 'spec.json'), 'makespan_s': round(float(t1) - float(t0), 2), 'gates': gates}))
PY
log "done: $(tail -1 "$RESULTS/gates.jsonl")"
