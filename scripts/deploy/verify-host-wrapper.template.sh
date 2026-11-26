#!/bin/bash
# The orchestrator wrapper on a remote verify host, rendered from
# scripts/deploy/verify-host-wrapper.template.sh by verify_host_contract.py;
# edits here are overwritten. A non-interactive ssh session sources no profile,
# so this puts the verify toolchain on PATH, then runs $ORCH_GATE_BIN (the
# dispatcher's gate checkout) or else this host's Dark Factory checkout.
# Contract: SETUP.md §12, "The shim contract".
export @@TOOL_PATH_PRELUDE@@
@@WRAPPER_ENV@@
_LOG="${XDG_STATE_HOME:-$HOME/.local/state}/orchestrator/wrapper.log"
mkdir -p "$(dirname "$_LOG")"
echo "$(date -Is) $$ START $*" >> "$_LOG"
"${ORCH_GATE_BIN:-@@DF_CHECKOUT_PATH@@/.venv/bin/orchestrator}" "$@" 2> >(tee -a "$_LOG" >&2)
rc=$?
echo "$(date -Is) $$ EXIT rc=$rc" >> "$_LOG"
exit $rc
