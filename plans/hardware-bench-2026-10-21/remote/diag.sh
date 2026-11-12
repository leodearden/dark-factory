#!/usr/bin/env bash
# diag.sh: after a run, separate CPU from filesystem cost in the per-thread probes.
#   py_collect_split  one pytest collection, user vs system CPU seconds (x3)
#   interp_loop       a pure CPython loop with the DF venv's python: no I/O at all (x5)
# Appends JSON lines to $RESULTS/diag.jsonl. REFERENCE=1 waits for a quiet host first.
set -euo pipefail

BENCH=$HOME/df-hwbench
RESULTS=${RESULTS:-$BENCH/results}
DF=$BENCH/df-0
PY=$DF/.venv/bin/python
FOREIGN_VERIFY='orchestrator verify-(merge|task)|scripts/verify\.sh'
quiet_load=$(awk -v t="$(nproc)" 'BEGIN { q = t / 8; print (q < 1.5 ? 1.5 : q) }')
log() { echo "[diag $(date -u +%T)] $*"; }

wait_quiet() {
	while pgrep -f "$FOREIGN_VERIFY" >/dev/null || awk -v q="$quiet_load" '{ exit !($1 >= q) }' /proc/loadavg; do
		sleep 20
	done
}

emit() { # <probe> <rep> <wall> <user> <sys>
	printf '{"probe": "%s", "rep": %s, "wall_s": %s, "user_s": %s, "sys_s": %s, "host_load1": %s}\n' \
		"$1" "$2" "$3" "$4" "$5" "$(cut -d' ' -f1 /proc/loadavg)" >>"$RESULTS/diag.jsonl"
}

for rep in 1 2 3; do
	[[ ${REFERENCE:-0} == 1 ]] && wait_quiet
	log "py_collect_split rep $rep"
	read -r wall user sys < <(cd "$DF/orchestrator" && /usr/bin/time -q -f '%e %U %S' -o /dev/stdout \
		"$PY" -m pytest --collect-only -q -n 0 -p no:cacheprovider tests 2>/dev/null | tail -1)
	emit py_collect_split "$rep" "$wall" "$user" "$sys"
done

for rep in 1 2 3 4 5; do
	[[ ${REFERENCE:-0} == 1 ]] && wait_quiet
	log "interp_loop rep $rep"
	read -r wall user sys < <(/usr/bin/time -q -f '%e %U %S' -o /dev/stdout "$PY" -I -c '
def work():
    d, s = {}, 0
    for i in range(3_000_000):
        s += i * i % 7
        d[i & 1023] = s
    return s
for _ in range(4):
    work()
' | tail -1)
	emit interp_loop "$rep" "$wall" "$user" "$sys"
done
log "done"
