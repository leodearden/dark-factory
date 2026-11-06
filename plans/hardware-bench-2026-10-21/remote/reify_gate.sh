#!/usr/bin/env bash
# reify_gate.sh [provision|gate|all]: the bench-pod half of the reify gate bench.
#
# Runs the real reify merge gate (`orchestrator verify-merge` with reify's own
# MergeVerifySpec, built by make_spec.py from reify's live config) twice in one
# persistent _merge-verify worktree, as leo-laptop does:
#   cold  at COLD_SHA: empty sccache, no worktree, no target/, no verify-logs
#   warm  at REIFY_SHA: target/, sccache and verify-logs left by the cold run
# COLD_SHA defaults to the merge the laptop gated just before REIFY_SHA, so the
# warm gate starts from the state the laptop's production gate started from.
# COLD_SHA=$REIFY_SHA instead makes the warm gate a same-sha rerun (near-zero
# compile): a lower bound, not the laptop's situation.
#
# Inputs (env):
#   BENCH       default /root/df-hwbench; ORCH defaults to $BENCH/df-0/.venv/bin/orchestrator
#   RESULTS     default $BENCH/results
#   STAGE       default /workspace/hwbench: reads reify-stage.tar and src/reify-repo.tar
#   APT_ARCHIVES  the .debs reify_stage.sh downloaded (default $STAGE/apt-archives)
#   SPEC_FILE   default $BENCH/reify-spec.json (make_spec.py <reify yaml> <out>)
#   REIFY_SHA, COLD_SHA, REIFY_TREE (the tree the laptop's verify-runs record names)
#   REIFY_ROOT  default /home/leo/src/reify, the laptop's path, so the spec's
#               absolute ledger paths and every compile path match the laptop's
#   SCCACHE_DIR default /root/.cache/sccache (local disk); SCCACHE_CACHE_SIZE default 100G
#   RUNS        default "cold warm"
# Outputs in $RESULTS:
#   reify_gates.jsonl          one line per run: run, sha, wall_s, rc, passed,
#                              timed_out, summary, sccache_stats, started_at
#   reify-gate-<run>.{stdout.json,stderr.log}, reify-sccache-<run>.txt
#   reify-spec.json, reify-spec.pod.json, reify-host.yaml (what actually ran)
#   reify-verify-runs.jsonl    verify.sh's own per-step record of both gates
set -euo pipefail

BENCH=${BENCH:-/root/df-hwbench}
RESULTS=${RESULTS:-$BENCH/results}
STAGE=${STAGE:-/workspace/hwbench}
SPEC_FILE=${SPEC_FILE:-$BENCH/reify-spec.json}
ORCH=${ORCH:-$BENCH/df-0/.venv/bin/orchestrator}
REIFY_SHA=${REIFY_SHA:-57be3b937f8cddfb77408603f46a798710644d89}
REIFY_TREE=${REIFY_TREE:-b2cbab7750c0202d6a73babab3e030ab51bb7d08}
COLD_SHA=${COLD_SHA:-fad2d981951d606291196fd8c9f8781f6cfbf350}
REIFY_ROOT=${REIFY_ROOT:-/home/leo/src/reify}
export SCCACHE_DIR=${SCCACHE_DIR:-/root/.cache/sccache}
export SCCACHE_CACHE_SIZE=${SCCACHE_CACHE_SIZE:-100G}
export SCCACHE_IDLE_TIMEOUT=0
RUNS=${RUNS:-cold warm}
APT_ARCHIVES=${APT_ARCHIVES:-$STAGE/apt-archives}
HERE=$(cd "$(dirname "$0")" && pwd)
STAGED=/root/reify-stage
HOST_YAML=$BENCH/reify-host.yaml
POD_SPEC=$BENCH/reify-spec.pod.json
SUDO=$([[ $(id -u) == 0 ]] || echo sudo)
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$HOME/.local/node/bin:$PATH
unset VIRTUAL_ENV SCCACHE_REDIS

log() { echo "[reify-gate $(date -u +%T)] $*"; }

install_apt() {
	log "apt: OCCT PPA source, then reify's packages from $APT_ARCHIVES"
	$SUDO install -D -m644 "$STAGED/apt/occt-releases.asc" /etc/apt/keyrings/occt-releases.asc
	$SUDO install -D -m644 "$STAGED/apt/occt-releases.sources" /etc/apt/sources.list.d/occt-releases.sources
	$SUDO mkdir -p "$APT_ARCHIVES/partial"
	$SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
	mapfile -t packages <"$STAGED/apt/packages.txt"
	$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
		-o Dir::Cache::archives="$APT_ARCHIVES" -o APT::Sandbox::User=root "${packages[@]}" >/dev/null
}

merge_into() { # <src dir> <dest dir>: move, or merge into an existing dest
	if [[ -e $2 ]]; then
		cp -a "$1/." "$2/"
		rm -rf "$1"
	else
		mkdir -p "$(dirname "$2")"
		mv "$1" "$2"
	fi
}

install_staged() {
	log "staged tools and caches from $STAGE/reify-stage.tar"
	rm -rf "$STAGED"
	tar --no-same-owner -xf "$STAGE/reify-stage.tar" -C "$(dirname "$STAGED")"
	install_apt
	install -D -m755 -t "$HOME/.cargo/bin" "$STAGED"/bin/*
	# The spec pins CARGO_HOME here, as on the laptop (reify task 5836).
	merge_into "$STAGED/cargo-home" /tmp/reify-agent-cargo-home
	merge_into "$STAGED/npm-cache" "$HOME/.npm"
}

install_repo() {
	log "reify repo at $REIFY_ROOT"
	local parent
	parent=$(dirname "$REIFY_ROOT")
	rm -rf "$REIFY_ROOT" "$parent/.reify-unpack"
	mkdir -p "$parent/.reify-unpack"
	tar --no-same-owner -xf "$STAGE/src/reify-repo.tar" -C "$parent/.reify-unpack"
	mv "$parent/.reify-unpack/reify" "$REIFY_ROOT"
	rmdir "$parent/.reify-unpack"
	git -C "$REIFY_ROOT" checkout -q -f --detach "$REIFY_SHA"
	local tree
	tree=$(git -C "$REIFY_ROOT" rev-parse "$REIFY_SHA^{tree}")
	[[ $tree == "$REIFY_TREE" ]] || { log "REIFY_SHA tree $tree is not the laptop's $REIFY_TREE"; exit 1; }
	git -C "$REIFY_ROOT" cat-file -e "$COLD_SHA^{commit}"
	git config --global user.name >/dev/null || git config --global user.name hwbench
	git config --global user.email >/dev/null || git config --global user.email hwbench@localhost
}

write_host_config() {
	sed "s|@PROJECT_ROOT@|$REIFY_ROOT|" "$HERE/reify-host.template.yaml" >"$HOST_YAML"
}

# The spec as the workstation ships it, with only the pod's forced changes:
# sccache on this pod's local disk (the laptop's server is local disk too; the
# shipped SCCACHE_REDIS points at the workstation and never reaches the
# laptop's already-running server), and absolute /home/leo/src/reify paths
# moved if REIFY_ROOT is elsewhere.
write_pod_spec() {
	python3 -I - "$SPEC_FILE" "$POD_SPEC" "$REIFY_ROOT" "$SCCACHE_DIR" "$SCCACHE_CACHE_SIZE" <<-'PY'
		import json, sys
		src, dst, root, sc_dir, sc_size = sys.argv[1:]
		spec = json.load(open(src))
		env = spec['verify_env']
		env.pop('SCCACHE_REDIS', None)
		env.update(SCCACHE_DIR=sc_dir, SCCACHE_CACHE_SIZE=sc_size, SCCACHE_IDLE_TIMEOUT='0')
		laptop_root = '/home/leo/src/reify'
		for key, value in env.items():
		    if root != laptop_root and value.startswith(laptop_root + '/'):
		        env[key] = root + value[len(laptop_root):]
		json.dump(spec, open(dst, 'w'), sort_keys=True)
	PY
}

provision() {
	install_staged
	install_repo
	write_host_config
	write_pod_spec
	for tool in cargo cargo-nextest sccache tree-sitter node npm python3 jq; do
		command -v "$tool" >/dev/null || { log "missing $tool on PATH"; exit 1; }
	done
	log "provisioned: $(sccache --version), $(cargo-nextest --version | head -1), $(tree-sitter --version), node $(node --version)"
}

restart_sccache() { # [wipe]
	sccache --stop-server >/dev/null 2>&1 || true
	[[ ${1:-} != wipe ]] || rm -rf "$SCCACHE_DIR"
	mkdir -p "$SCCACHE_DIR"
	sccache --start-server >/dev/null
	sccache --zero-stats >/dev/null
}

reset_for_cold() {
	log "cold state: no worktrees, no verify-logs, empty sccache"
	rm -rf "$REIFY_ROOT/.worktrees" "$REIFY_ROOT/data"
	mkdir -p "$REIFY_ROOT/data/verify-logs"
	git -C "$REIFY_ROOT" worktree prune
	restart_sccache wipe
}

run_gate() { # <run name> <sha>
	local run=$1 sha=$2 t0 t1 rc=0
	if [[ $run == cold ]]; then
		reset_for_cold
	else
		sccache --zero-stats >/dev/null
	fi
	log "$run gate at $sha"
	t0=$(date +%s.%N)
	${GATE_WRAPPER:-} "$ORCH" verify-merge --sha "$sha" --spec "$(cat "$POD_SPEC")" --config "$HOST_YAML" \
		>"$RESULTS/reify-gate-$run.stdout.json" 2>"$RESULTS/reify-gate-$run.stderr.log" || rc=$?
	t1=$(date +%s.%N)
	sccache --show-stats >"$RESULTS/reify-sccache-$run.txt" 2>&1 || true
	sccache --show-stats --stats-format=json >"$RESULTS/reify-sccache-$run.json" 2>/dev/null || echo '{}' >"$RESULTS/reify-sccache-$run.json"
	python3 -I - "$run" "$sha" "$t0" "$t1" "$rc" "$RESULTS" >>"$RESULTS/reify_gates.jsonl" <<-'PY'
		import json, os, pathlib, sys
		run, sha, t0, t1, rc, results = sys.argv[1:]
		results = pathlib.Path(results)
		try:
		    verdict = json.loads((results / f'reify-gate-{run}.stdout.json').read_text())
		except (OSError, ValueError):
		    verdict = {}
		try:
		    stats = json.loads((results / f'reify-sccache-{run}.json').read_text())
		except (OSError, ValueError):
		    stats = {}
		print(json.dumps({
		    'run': run, 'sha': sha, 'started_at': float(t0), 'wrapper': os.environ.get('GATE_WRAPPER', ''),
		    'wall_s': round(float(t1) - float(t0), 1), 'rc': int(rc),
		    'passed': verdict.get('passed'), 'timed_out': verdict.get('timed_out'),
		    'summary': (verdict.get('summary') or '')[:2000],
		    'sccache_stats': stats.get('stats', stats),
		}))
	PY
	log "$run done: $(tail -1 "$RESULTS/reify_gates.jsonl" | cut -c1-300)"
}

gate() {
	mkdir -p "$RESULTS"
	cp "$SPEC_FILE" "$RESULTS/reify-spec.json"
	cp "$POD_SPEC" "$RESULTS/reify-spec.pod.json"
	cp "$HOST_YAML" "$RESULTS/reify-host.yaml"
	for run in $RUNS; do
		case $run in
		cold) run_gate cold "$COLD_SHA" ;;
		warm) run_gate warm "$REIFY_SHA" ;;
		*) log "unknown run $run"; exit 64 ;;
		esac
	done
	cp "$REIFY_ROOT/data/verify-logs/verify-runs.jsonl" "$RESULTS/reify-verify-runs.jsonl" 2>/dev/null || true
}

case ${1:-all} in
provision) provision ;;
gate) gate ;;
all) provision && gate ;;
*) echo "usage: reify_gate.sh [provision|gate|all]" >&2; exit 64 ;;
esac
