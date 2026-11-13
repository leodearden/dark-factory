#!/usr/bin/env bash
# reify_as_user.sh: run reify_gate.sh's gate phase as an unprivileged user (run as root,
# after `reify_gate.sh provision`). As root, reify's permission tests (a read-only
# directory, an unreadable db, the sudo branch) cannot fail the way they are written to,
# and a failing nextest pass fails fast and skips every later step of the gate. The
# laptop runs the gate as its own user; so does this.
#   env: SPEC_FILE GATE_WRAPPER RUNS (as for reify_gate.sh)
# Leaves the user's results in $BENCH/results (an earlier attempt moves to reify-run<N>/).
set -euo pipefail

U=leo
UHOME=/home/$U
BENCH=/root/df-hwbench
VIEW=$UHOME/bench
REIFY_ROOT=/home/leo/src/reify
AGENT_CARGO_HOME=/tmp/reify-agent-cargo-home
log() { echo "[reify-user $(date -u +%T)] $*"; }

id -u "$U" >/dev/null 2>&1 || useradd -M -d "$UHOME" -s /bin/bash "$U"

log "user-level toolchains for $U"
for d in .rustup .cargo .local/node .npm; do
	rm -rf "${UHOME:?}/$d"
	mkdir -p "$(dirname "$UHOME/$d")"
	cp -a "/root/$d" "$UHOME/$d"
done

# The DF CLI (its venv and uv's Python) stays root's, read-only to the user.
chmod 755 /root
rm -rf "$VIEW"
mkdir -p "$VIEW/results"
ln -s "$BENCH/df-0" "$VIEW/df-0"
cp "$BENCH/reify-host.yaml" "$VIEW/"
python3 -I - "$BENCH/reify-spec.pod.json" "$VIEW/reify-spec.pod.json" "$UHOME/.cache/sccache" <<'PY'
import json, sys
src, dst, sccache_dir = sys.argv[1:]
spec = json.load(open(src))
spec['verify_env']['SCCACHE_DIR'] = sccache_dir
json.dump(spec, open(dst, 'w'), sort_keys=True)
PY
chown -R "$U:" "$UHOME" "$AGENT_CARGO_HOME"

# The gate writes fixed /tmp paths (dispatch stamps, locks, run dirs); root's runs left
# them root-owned, and the user cannot touch them.
log "clearing root's /tmp leftovers"
find /tmp -mindepth 1 -maxdepth 1 -user root ! -path "$AGENT_CARGO_HOME" -exec rm -rf {} +

log "stopping root's sccache server (the user's must own the port)"
sccache --stop-server >/dev/null 2>&1 || true

log "gate as $U"
runuser -u "$U" -- env -i HOME="$UHOME" USER="$U" LOGNAME="$U" LANG=C.UTF-8 \
	PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
	BENCH="$VIEW" RESULTS="$VIEW/results" SCCACHE_DIR="$UHOME/.cache/sccache" \
	SPEC_FILE="${SPEC_FILE:?}" GATE_WRAPPER="${GATE_WRAPPER:-}" RUNS="${RUNS:-cold warm}" \
	bash "$BENCH/bin/reify_gate.sh" gate || log "gate exited $?"

n=1
while [[ -e $BENCH/results/reify-run$n ]]; do n=$((n + 1)); done
mkdir -p "$BENCH/results/reify-run$n"
for f in "$BENCH"/results/reify-gate-* "$BENCH"/results/reify-sccache-* "$BENCH"/results/reify_gates.jsonl \
	"$BENCH"/results/reify-verify-runs.jsonl "$BENCH"/results/reify-spec*.json "$BENCH"/results/reify-host.yaml; do
	if [[ -e $f ]]; then
		mv "$f" "$BENCH/results/reify-run$n/"
	fi
done
cp -a "$VIEW/results/." "$BENCH/results/"
log "results as $U copied into $BENCH/results (the earlier attempt is in reify-run$n/)"
