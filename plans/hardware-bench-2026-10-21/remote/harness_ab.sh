#!/usr/bin/env bash
# harness_ab.sh: does probes.sh's own sampler (an awk over /proc/cpuinfo every 2 s,
# which on a 256-CPU host reads every CPU's frequency) slow the probe it watches?
# K=1 pytest collection plain vs with the sampler, interleaved x5; K=32 plain and with
# the sampler once each; then one rustc -j1 build alone. JSON lines to results/harness_ab.jsonl.
set -euo pipefail
cd "$HOME/df-hwbench/df-0/orchestrator"
PY=../.venv/bin/python
OUT=$HOME/df-hwbench/results/harness_ab.jsonl
export PATH=$HOME/.local/bin:$HOME/.local/node/bin:$HOME/.cargo/bin:$PATH
sampler() {
	while :; do
		awk '/^cpu MHz/ { s += $4; n++ } END { printf "%.0f ", n ? s / n : 0 }' /proc/cpuinfo >/dev/null
		cut -d' ' -f1 /proc/loadavg >/dev/null
		pgrep -fc 'orchestrator verify-(merge|task)' >/dev/null || true
		sleep 2
	done
}
collect() { "$PY" -m pytest --collect-only -q -n 0 -p no:cacheprovider tests >/dev/null 2>&1; }
batch() { # <k> <mode>: k copies together, prints per-copy walls
	local k=$1 mode=$2 sp=0 pids=() dir
	dir=$(mktemp -d)
	[[ $mode == sampler ]] && { sampler & sp=$!; }
	for ((i = 0; i < k; i++)); do
		/usr/bin/time -q -f '%e' -o "$dir/t.$i" bash -c "$(declare -f collect); PY=$PY collect" &
		pids+=($!)
	done
	wait "${pids[@]}"
	((sp == 0)) || { kill "$sp"; wait "$sp" 2>/dev/null || true; }
	cat "$dir"/t.* | tr '\n' ' '
	rm -rf "$dir"
}
emit() { printf '{"test": "%s", "k": %d, "mode": "%s", "walls": [%s], "host_load1": %s}\n' "$1" "$2" "$3" \
	"$(echo "$4" | xargs | tr ' ' ',')" "$(cut -d' ' -f1 /proc/loadavg)" >>"$OUT"; }
collect
for round in 1 2 3 4 5; do
	for mode in plain sampler; do emit py_collect 1 "$mode" "$(batch 1 "$mode")"; sleep 3; done
done
for mode in plain sampler; do emit py_collect 32 "$mode" "$(batch 32 "$mode")"; sleep 10; done
cd "$HOME/df-hwbench/reify"
rm -rf "$HOME/df-hwbench/scratch/target-ab"
w=$( { /usr/bin/time -q -f '%e' -o /dev/stdout env RUSTC_WRAPPER= CARGO_BUILD_RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
	CARGO_TARGET_DIR="$HOME/df-hwbench/scratch/target-ab" cargo build --locked --offline -j 1 -p reify-compiler >/dev/null 2>&1; } | tail -1)
emit rustc_j1 1 plain "$w"
rm -rf "$HOME/df-hwbench/scratch/target-ab"
echo done
