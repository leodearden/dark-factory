#!/usr/bin/env bash
# Fixed single-threaded workloads, timed alone and as K simultaneous copies.
# One copy measures the per-thread speed a serial gate leg sees; K = physical
# cores and K = hardware threads measure per-thread speed with the box busy.
# Then two cold reify compiles: -j1 (one rustc at a time) and -j<threads>.
# Appends one JSON line per batch to $RESULTS/probes.jsonl.
set -euo pipefail

: "${BENCH:?}" "${RESULTS:?}"
REFERENCE=${REFERENCE:-0}
QUICK=${QUICK:-0}
DF=$BENCH/df-0
REIFY=$BENCH/reify
OUT=${PROBES_OUT:-$RESULTS/probes.jsonl}
SCRATCH=$BENCH/scratch
FOREIGN_VERIFY='orchestrator verify-(merge|task)|scripts/verify\.sh'

# The CPU budget, not the host's thread count: in a container nproc and lscpu
# can disagree about what we may use (see container_limits.py).
LIMITS=$BENCH/bin/container_limits.py
threads=$(python3 -I "$LIMITS" cpu_budget)
cores=$(python3 -I "$LIMITS" cores)
allowed_cpus=$(python3 -I "$LIMITS" allowed_cpus)
quiet_load=$(awk -v t="$threads" 'BEGIN { q = t / 8; print (q < 1.5 ? 1.5 : q) }')

log() { echo "[probes $(date -u +%T)] $*"; }
log "cpu budget $threads ($(python3 -I "$LIMITS" cpu_budget_source)), k levels cores=$cores ($(python3 -I "$LIMITS" cores_rule)) threads=$threads"

# A reference host is a live verify host: start a batch only when none of its
# own verifies is running and its load has settled.
wait_quiet() {
	local waited=0
	while pgrep -f "$FOREIGN_VERIFY" >/dev/null ||
		awk -v q="$quiet_load" '{ exit !($1 >= q) }' /proc/loadavg; do
		((waited % 600 == 0)) && log "waiting for the host to go quiet (load $(cut -d' ' -f1 /proc/loadavg))"
		sleep 20
		waited=$((waited + 20))
	done
}

sample_loop() {
	while :; do
		# "<mean MHz over every CPU> <mean over the CPUs we may use> <max over them> "
		awk -v allowed="$allowed_cpus" '
			BEGIN { n = split(allowed, parts, ","); for (i = 1; i <= n; i++) { m = split(parts[i], r, "-"); for (c = r[1]; c <= r[m]; c++) ok[c] = 1 } n = 0 }
			/^processor/ { cpu = $3 }
			/^cpu MHz/ {
				s += $4; n++
				if (cpu in ok) { sa += $4; na++; if ($4 > mx) mx = $4 }
			}
			END { printf "%.0f %.0f %.0f ", n ? s / n : 0, na ? sa / na : 0, mx }' /proc/cpuinfo
		# The 1-min load of the whole host (a container's /proc/loadavg is the host's):
		# how busy the neighbours were.
		printf '%s ' "$(cut -d' ' -f1 /proc/loadavg)"
		pgrep -fc "$FOREIGN_VERIFY" || true
		sleep 2
	done
}

# run_batch <probe> <k> <rep> <command>: start k copies of <command> together.
run_batch() {
	local probe=$1 k=$2 rep=$3 cmd=$4
	local dir=$SCRATCH/batch-$probe-k$k-r$rep
	rm -rf "$dir"
	mkdir -p "$dir"
	[[ $REFERENCE == 1 ]] && wait_quiet
	log "$probe k=$k rep=$rep"
	sample_loop >"$dir/samples" &
	local sampler=$! pids=() t0 t1
	t0=$(date +%s.%N)
	for ((i = 0; i < k; i++)); do
		/usr/bin/time -q -f '%e %M %x' -o "$dir/time.$i" bash -c "$cmd" >/dev/null 2>"$dir/err.$i" &
		pids+=($!)
	done
	wait "${pids[@]}" || true
	t1=$(date +%s.%N)
	kill "$sampler"
	wait "$sampler" 2>/dev/null || true
	python3 -I - "$probe" "$k" "$rep" "$t0" "$t1" "$dir" >>"$OUT" <<'PY'
import json, pathlib, sys
probe, k, rep, t0, t1, d = sys.argv[1:]
d = pathlib.Path(d)
runs = []
for f in sorted(d.glob('time.*'), key=lambda p: int(p.suffix[1:])):
    wall, rss, rc = f.read_text().split()[-3:]
    runs.append({'wall_s': float(wall), 'maxrss_kb': int(rss), 'rc': int(rc)})
samples = [line.split() for line in (d / 'samples').read_text().splitlines()]
samples = [s for s in samples if len(s) == 5]
mhz = [float(s[0]) for s in samples]
mhz_allowed = [float(s[1]) for s in samples]
mhz_max_allowed = [float(s[2]) for s in samples]
host_load1 = [float(s[3]) for s in samples]
print(json.dumps({
    'probe': probe, 'k': int(k), 'rep': int(rep),
    'makespan_s': round(float(t1) - float(t0), 2), 'runs': runs,
    'mhz_mean': round(sum(mhz) / len(mhz)) if mhz else None,
    'mhz_mean_allowed': round(sum(mhz_allowed) / len(mhz_allowed)) if mhz_allowed else None,
    'mhz_max_allowed': round(sum(mhz_max_allowed) / len(mhz_max_allowed)) if mhz_max_allowed else None,
    'host_load1_mean': round(sum(host_load1) / len(host_load1), 1) if host_load1 else None,
    'foreign_verify_seen': any(int(s[4]) > 0 for s in samples),
}))
PY
	local bad
	bad=$(awk '$NF != 0' "$dir"/time.* | wc -l)
	((bad == 0)) || log "WARNING: $bad of $k copies exited non-zero; see $dir/err.*"
}

max_rss_kb() { # largest maxrss seen for <probe> so far
	python3 -I -c '
import json, sys
rss = [r["maxrss_kb"] for line in open(sys.argv[1]) for b in [json.loads(line)]
       if b["probe"] == sys.argv[2] for r in b["runs"]]
print(max(rss))' "$OUT" "$1"
}

# K levels for a probe: cores and the CPU budget, each capped so the copies
# fit in 60% of the memory available to this host or container.
k_levels() {
	local rss avail cap
	rss=$(max_rss_kb "$1")
	avail=$(python3 -I "$LIMITS" mem_available_kb)
	cap=$((avail * 6 / 10 / rss))
	if [[ $QUICK == 1 ]]; then
		echo 2
		return
	fi
	for k in "$cores" "$threads"; do
		((k <= cap)) || {
			log "capping k=$k to $cap by memory (maxrss ${rss} kB)" >&2
			k=$cap
		}
		echo "$k"
	done | sort -nu
}

probe_scaled() { # <probe> <command>
	local probe=$1 cmd=$2 reps=3
	[[ $QUICK == 1 ]] && reps=1
	log "$probe warm-up"
	bash -c "$cmd" >/dev/null 2>&1 || true
	for ((r = 1; r <= reps; r++)); do run_batch "$probe" 1 "$r" "$cmd"; done
	for k in $(k_levels "$probe"); do run_batch "$probe" "$k" 1 "$cmd"; done
}

mkdir -p "$SCRATCH" "$RESULTS"

# ONLY="<probe> ..." runs just those probes (e.g. a rerun of the compiles).
wanted() { [[ -z ${ONLY:-} || " $ONLY " == *" $1 "* ]]; }

wanted py_collect && probe_scaled py_collect \
	"cd '$DF/orchestrator' && '$DF/.venv/bin/python' -m pytest --collect-only -q -n 0 -p no:cacheprovider tests"
wanted pyright && probe_scaled pyright \
	"cd '$DF/orchestrator' && PATH='$DF/.venv/bin':\$PATH '$DF/.venv/bin/pyright' src"

[[ $QUICK == 1 ]] && crates=(-p reify-core) || crates=(-p reify-compiler)
rust_env="RUSTC_WRAPPER= CARGO_BUILD_RUSTC_WRAPPER= CARGO_INCREMENTAL=0"
rm -rf "$SCRATCH/target-j1"
wanted rustc_j1 && run_batch rustc_j1 1 1 \
	"cd '$REIFY' && $rust_env CARGO_TARGET_DIR='$SCRATCH/target-j1' cargo build --locked --offline -j 1 ${crates[*]}"
rm -rf "$SCRATCH/target-j1"

[[ $QUICK == 1 ]] || crates=(-p reify-compiler -p reify-stdlib -p reify-constraints -p reify-ir -p reify-expr)
rm -rf "$SCRATCH/target-jn"
wanted rustc_jn && run_batch rustc_jn 1 1 \
	"cd '$REIFY' && $rust_env CARGO_TARGET_DIR='$SCRATCH/target-jn' cargo build --locked --offline -j $threads ${crates[*]}"
rm -rf "$SCRATCH/target-jn"

log "done"
