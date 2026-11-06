#!/usr/bin/env bash
# The whole benchmark on one host, meant to run detached (bench.sh starts it
# with setsid and polls for results/DONE or results/FAILED).
#   MODE=target     probes, then the real DF gates (bench.sh has provisioned)
#   MODE=reference  a live verify host (the laptop): no provisioning, no gates,
#                   each probe batch waits for the host to go quiet
set -euo pipefail

export BENCH=$HOME/df-hwbench
export RESULTS=$BENCH/results
: "${MODE:?}" "${DF_SHA:?}" "${GATE_CONCURRENCY:=8}" "${SKIP_GATE:=0}" "${QUICK:=0}"
export DF_SHA QUICK
export REFERENCE=$([[ $MODE == reference ]] && echo 1 || echo 0)
export PATH=$HOME/.local/bin:$HOME/.local/node/bin:$HOME/.cargo/bin:$PATH
unset VIRTUAL_ENV

trap 'rc=$?; ((rc == 0)) && touch "$RESULTS/DONE" || echo "rc=$rc at line $LINENO" >"$RESULTS/FAILED"' EXIT
mkdir -p "$RESULTS"
rm -f "$RESULTS/DONE" "$RESULTS/FAILED"
log() { echo "[run $(date -u +%T)] $*"; }

log "host facts"
{
	echo "== lscpu"; lscpu
	echo "== nproc"; nproc
	echo "== meminfo"; head -3 /proc/meminfo
	echo "== uname"; uname -a
	echo "== os"; grep PRETTY_NAME /etc/os-release
	echo "== loadavg"; cat /proc/loadavg
} >"$RESULTS/host.txt"
log "dark-factory checkout at $DF_SHA"
if [[ ! -d $BENCH/df-0 ]]; then
	git clone -q "$BENCH/df.git" "$BENCH/df-0"
fi
git -C "$BENCH/df-0" checkout -q --detach "$DF_SHA"
(cd "$BENCH/df-0" && uv sync --all-packages --frozen -q && npm ci --no-audit --no-fund --prefer-offline --silent)

python3 -I - "$MODE" "$DF_SHA" "$QUICK" "$(python3 -I "$BENCH/bin/container_limits.py")" "$(python3 -I "$BENCH/bin/toolchain_versions.py" "$BENCH/df-0/.venv/bin/python")" >"$RESULTS/host.json" <<'PY'
import json, os, subprocess, sys
lscpu = dict(line.split(':', 1) for line in subprocess.run(
    ['lscpu'], capture_output=True, text=True).stdout.splitlines() if ':' in line)
lscpu = {k.strip(): v.strip() for k, v in lscpu.items()}
limits = json.loads(sys.argv[4])
toolchain = json.loads(sys.argv[5])
host_mem_kb = int(open('/proc/meminfo').readline().split()[1])
mem_kb = min(host_mem_kb, limits['mem_limit_kb'] or host_mem_kb)
# cores/threads are the usable ones (the K levels of the probes); on bare metal
# they are the host's physical cores and hardware threads.
print(json.dumps({
    'hostname': os.uname().nodename, 'mode': sys.argv[1], 'df_sha': sys.argv[2],
    'quick': sys.argv[3] == '1',
    'cpu_model': lscpu.get('Model name'), 'sockets': lscpu.get('Socket(s)'),
    'cores': limits['cores'], 'threads': limits['cpu_budget'], 'mem_gib': round(mem_kb / 2**20),
    'max_mhz': lscpu.get('CPU max MHz'),
    'cpu_budget': limits['cpu_budget'], 'cpu_budget_source': limits['cpu_budget_source'],
    'cores_rule': limits['cores_rule'], 'host_threads': limits['host_threads'],
    'allowed_cpus': limits['allowed_cpus'], 'cgroup_cpu_quota': limits['cgroup_cpu_quota'],
    'toolchain': toolchain,
    'mem_host_gib': round(host_mem_kb / 2**20), 'mem_limit_kb': limits['mem_limit_kb'],
}))
PY

log "reify crates"
(cd "$BENCH/reify" && cargo fetch --locked -q)

bash "$BENCH/bin/probes.sh"

if [[ $MODE == target && $SKIP_GATE == 0 ]]; then
	bash "$BENCH/bin/gate.sh" 1
	((GATE_CONCURRENCY > 1)) && bash "$BENCH/bin/gate.sh" "$GATE_CONCURRENCY"
fi

# REIFY_STAGE holds reify_stage.sh's output (the volume, or a local dir on a pod without one).
if [[ $MODE == target && ${REIFY_GATE:-0} == 1 ]]; then
	STAGE=${REIFY_STAGE:?} bash "$BENCH/bin/reify_gate.sh" all
fi
log "finished"
