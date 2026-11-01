#!/usr/bin/env bash
# followup.sh <ssh-host> <local results dir> [--reference]
#
# After bench.sh's run on <ssh-host>: rerun the two reify compile probes (their
# first attempt fails on a pod without the tree-sitter CLI, which reify_gate.sh
# installs later), moving the failed batches to probes.failed.jsonl so no median
# mixes them in; then run remote/diag.sh (CPU vs filesystem split). Copies
# probes.jsonl, probes.failed.jsonl and diag.jsonl into <local results dir>.
# --reference: a live verify host; only diag.sh runs, and it waits for quiet.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
host=${1:?usage: followup.sh <ssh-host> <local results dir> [--reference]}
out=${2:?usage: followup.sh <ssh-host> <local results dir> [--reference]}
reference=$([[ ${3:-} == --reference ]] && echo 1 || echo 0)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
[[ -z ${BENCH_SSH_CONFIG:-} ]] || SSH_OPTS=(-F "$BENCH_SSH_CONFIG" "${SSH_OPTS[@]}")
remote() { ssh "${SSH_OPTS[@]}" "$host" "$@"; }
log() { echo "[followup $(date -u +%T)] $*"; }

scp -q "${SSH_OPTS[@]}" "$HERE"/remote/* "$host:df-hwbench/bin/"
env_prefix='cd df-hwbench && export BENCH=$HOME/df-hwbench RESULTS=$HOME/df-hwbench/results QUICK=0 PATH=$HOME/.local/bin:$HOME/.local/node/bin:$HOME/.cargo/bin:$PATH'

if [[ $reference == 0 ]]; then
	log "setting aside failed compile batches"
	remote "$env_prefix && python3 -I - \$RESULTS/probes.jsonl" <<-'PY'
		import json, pathlib, sys
		path = pathlib.Path(sys.argv[1])
		batches = [json.loads(line) for line in path.read_text().splitlines()]
		failed = [b for b in batches if any(r['rc'] != 0 for r in b['runs'])]
		kept = [b for b in batches if b not in failed]
		with open(path.with_name('probes.failed.jsonl'), 'a') as f:
		    f.writelines(json.dumps(b) + '\n' for b in failed)
		path.write_text(''.join(json.dumps(b) + '\n' for b in kept))
		print(f'kept {len(kept)}, set aside {len(failed)}: {sorted({b["probe"] for b in failed})}')
	PY
	log "rerunning the reify compile probes (tree-sitter CLI: $(remote 'command -v tree-sitter || ls ~/.cargo/bin/tree-sitter'))"
	remote "$env_prefix && REFERENCE=0 ONLY='rustc_j1 rustc_jn' bash bin/probes.sh" | grep -E '^\[probes '
fi

log "diag.sh"
remote "$env_prefix && REFERENCE=$reference bash bin/diag.sh" | grep -E '^\[diag '

for f in probes.jsonl probes.failed.jsonl diag.jsonl; do
	scp -q "${SSH_OPTS[@]}" "$host:df-hwbench/results/$f" "$out/" 2>/dev/null || log "no $f on $host"
done
log "copied into $out"
