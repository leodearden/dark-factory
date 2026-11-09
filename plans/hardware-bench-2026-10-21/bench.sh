#!/usr/bin/env bash
# bench.sh [--reference | --stage | --from-volume] [--skip-gate] [--quick] [--gates N] <ssh-host>
#
# Ships dark-factory (git push of DF_SHA) and reify (git archive of REIFY_SHA)
# to <ssh-host>, starts remote/run.sh there detached, polls until it finishes,
# then copies the results to results/<host>-<utc>/ and prints the summary.
#   --reference    the host is a live verify host (the laptop): no provisioning,
#                  no gates, every probe batch waits for the host to go quiet
#   --stage        the host is the cheap STAGING pod of a RunPod network volume:
#                  put both sources on the volume ($BENCH_STAGE/src) and run
#                  remote/stage.sh, which builds the toolchains and the apt/uv/npm/
#                  cargo caches on the pod's local disk and publishes them as a few
#                  tar files. Runs no benchmark.
#   --from-volume  the host is the BENCH pod, same volume mounted: remote/localize.sh
#                  extracts the tars to local disk, then the normal run (probes,
#                  gates) takes its sources from those copies. No provisioning,
#                  no push, no archive: the bench run touches the volume for nothing.
#   --skip-gate    probes only
#   --quick        a smoke run: one copy per probe, k=2, small compiles
#   --gates N      concurrency of the second gate batch (default 8)
#   --reify        after the DF gates, the real reify merge gate, cold then warm
#                  (remote/reify_gate.sh). On a volume pod it reads reify_stage.sh's
#                  output from the volume; otherwise this script stages it on the
#                  pod's local disk first (repo tar, /opt/reify-deps, reify_stage.sh).
# BENCH_SSH_CONFIG names an ssh config file to use instead of ~/.ssh/config.
# BENCH_STAGE is the volume's staging directory on the pods (default /workspace/hwbench).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
DF_ROOT=/home/leo/src/dark-factory
REIFY_ROOT=/home/leo/src/reify
# Main at 2026-10-21; the laptop gated this merge green in 862.8 s.
DF_SHA=${DF_SHA:-71bc561086e5116c5ef65bc91d5b4e44a77ef211}
REIFY_SHA=${REIFY_SHA:-57be3b937f8cddfb77408603f46a798710644d89}
STAGE=${BENCH_STAGE:-/workspace/hwbench}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
[[ -z ${BENCH_SSH_CONFIG:-} ]] || SSH_OPTS=(-F "$BENCH_SSH_CONFIG" "${SSH_OPTS[@]}")

mode=target skip_gate=0 quick=0 gates=8 reify=0
while [[ $# -gt 1 ]]; do
	case $1 in
	--reference) mode=reference ;;
	--stage) mode=stage ;;
	--from-volume) mode=volume ;;
	--skip-gate) skip_gate=1 ;;
	--quick) quick=1 ;;
	--gates) gates=$2; shift ;;
	--reify) reify=1 ;;
	*) echo "unknown option $1" >&2; exit 64 ;;
	esac
	shift
done
host=${1:?usage: bench.sh [--reference | --stage | --from-volume] [--skip-gate] [--quick] [--gates N] [--reify] <ssh-host>}

log() { echo "[bench $(date -u +%T)] $*"; }
remote() { ssh "${SSH_OPTS[@]}" "$host" "$@"; }
export GIT_SSH_COMMAND="ssh ${SSH_OPTS[*]}"

# push_sources <remote df.git> <remote reify dir>
push_sources() {
	log "dark-factory $DF_SHA"
	remote "test -d $1 || git init -q --bare -b main $1"
	git -C "$DF_ROOT" push -q -f "$host:$1" "$DF_SHA:refs/heads/main"

	log "reify $REIFY_SHA"
	if [[ $(remote "cat $2/.bench-sha 2>/dev/null || true") != "$REIFY_SHA" ]]; then
		remote "rm -rf $2 && mkdir -p $2"
		git -C "$REIFY_ROOT" archive --format=tar "$REIFY_SHA" | remote "tar -x -C $2"
		remote "echo $REIFY_SHA > $2/.bench-sha"
	fi
}

# stage_sources: onto the volume as few, large files: a bare repo that keeps the
# pushed pack whole, and the reify archive as one tar (written under a .tmp name).
stage_sources() {
	local repo=$STAGE/src/df.git reify_tar=$STAGE/src/reify-$REIFY_SHA.tar
	log "dark-factory $DF_SHA"
	remote "test -d $repo || git init -q --bare --template= -b main $repo
		git -C $repo config receive.unpackLimit 1; git -C $repo config gc.auto 0"
	git -C "$DF_ROOT" push -q -f "$host:$repo" "$DF_SHA:refs/heads/main"

	log "reify $REIFY_SHA"
	if ! remote "test -s $reify_tar"; then
		git -C "$REIFY_ROOT" archive --format=tar "$REIFY_SHA" | remote "cat >$reify_tar.tmp && mv $reify_tar.tmp $reify_tar"
	fi
}

# run_detached <script> <flag dir> <env assignments>: start bin/<script> on the
# host and poll until it leaves DONE or FAILED in <flag dir> (relative to
# df-hwbench), echoing its tagged log lines. Detached because a long step must
# survive a dropped ssh connection.
run_detached() {
	local script=$1 flags=$2 env=$3 log_file=${1%.sh}.log
	# Only the simple command is backgrounded: a backgrounded `cd && ...` list
	# would hold ssh's stdout open and block this call until the step ends.
	remote "cd df-hwbench || exit 1; rm -rf $flags; $env \
		setsid nohup bash bin/$script >$log_file 2>&1 </dev/null &"
	local seen=0 lines
	until remote "test -e df-hwbench/$flags/DONE -o -e df-hwbench/$flags/FAILED"; do
		lines=$(remote "tail -n +$((seen + 1)) df-hwbench/$log_file" || true)
		[[ -n $lines ]] && echo "$lines" | grep -E '^\[(run|probes|gate|provision|stage|localize|reify-gate|reify-stage) ' || true
		seen=$((seen + $(printf '%s' "$lines" | grep -c '' || true)))
		sleep "${BENCH_POLL_SECS:-60}"
	done
	remote "tail -n +$((seen + 1)) df-hwbench/$log_file" | grep -E '^\[(run|probes|gate|provision|stage|localize|reify-gate|reify-stage) ' || true
	if remote "test -e df-hwbench/$flags/FAILED"; then
		log "$script FAILED ($(remote "cat df-hwbench/$flags/FAILED")); log on $host:df-hwbench/$log_file"
		return 1
	fi
}

log "shipping scripts to $host:df-hwbench"
remote 'mkdir -p df-hwbench/bin && rm -rf df-hwbench/results'
scp -q "${SSH_OPTS[@]}" "$HERE"/remote/* "$host:df-hwbench/bin/"

if [[ $mode == stage ]]; then
	stage_sources
	log "staging onto the volume"
	run_detached stage.sh stage-status "STAGE=$STAGE DF_SHA=$DF_SHA REIFY_SHA=$REIFY_SHA STATUS_DIR=\$PWD/stage-status"
	out=$HERE/results/$host-stage-$(date -u +%Y%m%dT%H%MZ)
	mkdir -p "$out"
	scp -q "${SSH_OPTS[@]}" "$host:df-hwbench/stage.log" "$out/"
	remote "cat $STAGE/staged.json; ls -l $STAGE $STAGE/src"
	log "staged; log in $out"
	exit 0
fi

if [[ $mode == target ]]; then
	log "provisioning"
	remote 'bash df-hwbench/bin/provision.sh'
fi
if [[ $mode == volume ]]; then
	log "localizing the volume"
	run_detached localize.sh localize-status "STAGE=$STAGE DF_SHA=$DF_SHA REIFY_SHA=$REIFY_SHA STATUS_DIR=\$PWD/localize-status"
else
	push_sources df-hwbench/df.git df-hwbench/reify
fi

if [[ $mode != reference && $skip_gate == 0 ]]; then
	log "gate spec from the live config"
	only=()
	[[ $quick == 0 ]] || only=(sampler)
	"$DF_ROOT/.venv/bin/python" "$HERE/make_spec.py" "$DF_ROOT/dark-factory-orchestrator.yaml" "$HERE/.spec.json" "${only[@]}"
	scp -q "${SSH_OPTS[@]}" "$HERE/.spec.json" "$host:df-hwbench/spec.json"
fi

reify_stage_dir=$STAGE
if [[ $reify == 1 ]]; then
	log "reify gate spec from the live config"
	"$DF_ROOT/.venv/bin/python" "$HERE/make_spec.py" "$REIFY_ROOT/dark-factory-orchestrator.yaml" "$HERE/.reify-spec.json"
	scp -q "${SSH_OPTS[@]}" "$HERE/.reify-spec.json" "$host:df-hwbench/reify-spec.json"
	if [[ $mode == target ]]; then
		# No volume: stage reify on the pod's own disk, as the staging pod does for the volume.
		reify_stage_dir=/root/hwbench-stage
		[[ -s $HERE/.reify-repo.tar ]] || bash "$HERE/reify_bundle.sh" "$HERE/.reify-repo.tar"
		log "reify repo and /opt/reify-deps"
		remote "mkdir -p $reify_stage_dir/src"
		remote "cat > $reify_stage_dir/src/reify-repo.tar" <"$HERE/.reify-repo.tar"
		tar -C /opt -cf - reify-deps | remote 'mkdir -p /opt && tar -C /opt --no-same-owner -xf -'
		log "reify_stage.sh on the pod"
		remote "STAGE=$reify_stage_dir bash df-hwbench/bin/reify_stage.sh"
	fi
fi

log "starting the run (MODE=$mode)"
# run.sh knows two modes: a live reference host, or a target that gates.
run_mode=$([[ $mode == reference ]] && echo reference || echo target)
run_rc=0
run_detached run.sh results "MODE=$run_mode DF_SHA=$DF_SHA GATE_CONCURRENCY=$gates SKIP_GATE=$skip_gate QUICK=$quick \
	REIFY_GATE=$reify REIFY_STAGE=$reify_stage_dir" || run_rc=$?

out=$HERE/results/$host-$(date -u +%Y%m%dT%H%MZ)
mkdir -p "$out"
scp -q -r "${SSH_OPTS[@]}" "$host:df-hwbench/results/." "$out/"
scp -q "${SSH_OPTS[@]}" "$host:df-hwbench/run.log" "$out/"
[[ $mode != volume ]] || scp -q "${SSH_OPTS[@]}" "$host:df-hwbench/localize.log" "$out/"
if ((run_rc != 0)); then
	log "run FAILED ($(cat "$out/FAILED")); log: $out/run.log"
	exit 1
fi
log "results in $out"
python3 "$HERE/summarize.py" "$out"
