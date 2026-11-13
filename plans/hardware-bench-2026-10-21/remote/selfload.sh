#!/usr/bin/env bash
# selfload.sh: is a busy Milan core slow because of OTHER tenants or because of our own
# load? Alternates one probe copy alone with the same copy beside 31 CPU-burning
# copies of the interpreter loop (the pod's other 31 CPUs), recording host load.
set -euo pipefail
cd "$HOME/df-hwbench/df-0/orchestrator"
PY=../.venv/bin/python
OUT=$HOME/df-hwbench/results/selfload.jsonl
BURN='
def work():
    d, s = {}, 0
    for i in range(3_000_000):
        s += i * i % 7
        d[i & 1023] = s
while True:
    work()
'
probe() { # <name>: one timed copy, prints "wall user sys"
	case $1 in
	py_collect) /usr/bin/time -q -f '%e %U %S' -o /dev/stdout "$PY" -m pytest --collect-only -q -n 0 -p no:cacheprovider tests 2>/dev/null | tail -1 ;;
	pyright) PATH=../.venv/bin:$PATH /usr/bin/time -q -f '%e %U %S' -o /dev/stdout ../.venv/bin/pyright src 2>/dev/null | tail -1 ;;
	esac
}
for round in 1 2 3 4 5; do
	for name in py_collect pyright; do
		for mode in alone burners31; do
			pids=()
			if [[ $mode == burners31 ]]; then
				for _ in $(seq 31); do "$PY" -I -c "$BURN" & pids+=($!); done
				sleep 3
			fi
			load=$(cut -d' ' -f1 /proc/loadavg)
			read -r wall user sys < <(probe "$name")
			((${#pids[@]} == 0)) || { kill "${pids[@]}"; wait "${pids[@]}" 2>/dev/null || true; }
			printf '{"probe": "%s", "mode": "%s", "round": %d, "wall_s": %s, "user_s": %s, "sys_s": %s, "host_load1": %s}\n' \
				"$name" "$mode" "$round" "$wall" "$user" "$sys" "$load" >>"$OUT"
			sleep 5
		done
	done
done
echo done
