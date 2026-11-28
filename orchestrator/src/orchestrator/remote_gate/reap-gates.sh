#!/usr/bin/env bash
# Reap idle gate checkouts under $GATE_ROOT: sha dirs unused for 7 days and
# trash older than 1 day.
#
# Task 6565, design D2. Contract (orchestrator/remote_gate.py::reap_script_text):
#   GATE_ROOT  absolute directory holding one checkout per sha
#
# Only a root carrying .dark-factory-gate-root is touched, and only names that
# are exactly 40 hex digits or start with .trash-. A sha dir is reaped under its
# build lock (never while ensure-gate.sh builds or touches it), is renamed to
# trash first so an interrupted reap never leaves a half-dir at a sha path, and
# is aged by its mtime, to the minute, which every ensure refreshes.
#
# Build lock protocol, binding on every locker of $GATE_ROOT/.build-<sha>.lock
# (this script, ensure-gate.sh and any future one): this script unlinks a reaped
# sha's lock and ensure-gate.sh unlinks its lock when it fails, each under that
# lock. So a locker holds the lock only once a re-check after flock finds fd 9
# still the file at the lock path; one that finds it replaced retakes it
# (ensure-gate.sh) or skips the sha (here). Only that verified holder unlinks a
# lock, and no lock is aged, so a lock another process holds is never unlinked.
set -euo pipefail

[[ ${GATE_ROOT:-} == /* ]] || {
	echo "reap-gates: GATE_ROOT must be an absolute path, got '${GATE_ROOT:-}'" >&2
	exit 64
}
[[ -f $GATE_ROOT/.dark-factory-gate-root ]] || exit 0
idle_mins=$((7 * 24 * 60))

reap_sha_dir() {
	local dir=$1 lock trash=''
	lock="$GATE_ROOT/.build-$(basename "$dir").lock"
	exec 9>"$lock"
	if flock -n 9 && [[ /dev/fd/9 -ef $lock ]] &&
		[[ -n $(find "$dir" -maxdepth 0 -mmin "+$idle_mins" 2>/dev/null) ]]; then
		trash=$(mktemp -u -d "$GATE_ROOT/.trash-XXXXXXXX")
		mv -T "$dir" "$trash"
		rm -f "$lock"
	fi
	exec 9>&-
	[[ -z $trash ]] || rm -rf "$trash"
}

while IFS= read -r -d '' dir; do
	reap_sha_dir "$dir"
done < <(find "$GATE_ROOT" -mindepth 1 -maxdepth 1 -type d \
	-regextype posix-extended -regex '.*/[0-9a-f]{40}' -mmin "+$idle_mins" -print0)

find "$GATE_ROOT" -mindepth 1 -maxdepth 1 -name '.trash-*' -mmin +1440 \
	-exec rm -rf {} +
