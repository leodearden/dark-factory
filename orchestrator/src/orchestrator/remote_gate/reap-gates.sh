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
# is aged by its mtime, which every ensure refreshes. Lock files are never
# reaped: flock does not touch their mtime, so an age rule would unlink a held
# lock.
set -euo pipefail

[[ ${GATE_ROOT:-} == /* ]] || {
	echo "reap-gates: GATE_ROOT must be an absolute path, got '${GATE_ROOT:-}'" >&2
	exit 64
}
[[ -f $GATE_ROOT/.dark-factory-gate-root ]] || exit 0

reap_sha_dir() {
	local dir=$1 trash=''
	exec 9>"$GATE_ROOT/.build-$(basename "$dir").lock"
	if flock -n 9 && [[ -n $(find "$dir" -maxdepth 0 -mtime +7 2>/dev/null) ]]; then
		trash=$(mktemp -u -d "$GATE_ROOT/.trash-XXXXXXXX")
		mv -T "$dir" "$trash"
	fi
	exec 9>&-
	[[ -z $trash ]] || rm -rf "$trash"
}

while IFS= read -r -d '' dir; do
	reap_sha_dir "$dir"
done < <(find "$GATE_ROOT" -mindepth 1 -maxdepth 1 -type d \
	-regextype posix-extended -regex '.*/[0-9a-f]{40}' -mtime +7 -print0)

find "$GATE_ROOT" -mindepth 1 -maxdepth 1 -name '.trash-*' -mmin +1440 \
	-exec rm -rf {} +
