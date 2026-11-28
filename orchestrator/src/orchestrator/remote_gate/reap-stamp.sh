#!/usr/bin/env bash
# Print how long ago reap-gates.sh last completed under $GATE_ROOT, as one JSON
# line: {"gate_root": <bool>, "age_secs": <int or null>}.
#
# Task 8520. Contract (orchestrator/remote_gate.py::reap_stamp_script_text,
# parsed by remote_gate.py::parse_reap_stamp):
#   GATE_ROOT  absolute directory holding one checkout per sha
#
# gate_root is false when $GATE_ROOT carries no .dark-factory-gate-root;
# age_secs is null when no run has completed there (no .last-reap). The age is
# this host's clock minus the stamp's mtime on this host, so the dispatcher's
# clock never enters.
set -euo pipefail

[[ ${GATE_ROOT:-} == /* ]] || {
	echo "reap-stamp: GATE_ROOT must be an absolute path, got '${GATE_ROOT:-}'" >&2
	exit 64
}
stamp=$GATE_ROOT/.last-reap

if [[ ! -f $GATE_ROOT/.dark-factory-gate-root ]]; then
	echo '{"gate_root": false, "age_secs": null}'
elif [[ ! -f $stamp ]]; then
	echo '{"gate_root": true, "age_secs": null}'
else
	stamped=$(stat -c %Y "$stamp")
	now=$(date +%s)
	echo "{\"gate_root\": true, \"age_secs\": $((now - stamped))}"
fi
