#!/usr/bin/env bash
# Install (or re-install, idempotently) the lms-arm@ systemd user unit TEMPLATE.
#
# PRD-MARKER:local-memory-models-eval serving
#
# Task 3713 (LME-alpha) of plans/local-memory-models-eval-prd.md.
#
# Usage: install-lms-units.sh
#
# Copies lms-arm@.service VERBATIM into ${XDG_CONFIG_HOME:-$HOME/.config}/
# systemd/user/ and reloads the user daemon.  There is exactly one template and
# nothing is per-arm customized: systemd expands %i to the arm_id at
# instantiation time, so `systemctl --user start lms-arm@qwen3.5-9b.service`
# is all an arm needs.
#
# Deliberately does NOT enable or start anything.  These units hold the whole
# GPU; which arm runs when is a decision the operator (or lms_ctl, with its
# VRAM pre-flight) makes per run, never a side effect of installing.
#
# Self-verifies afterwards: daemon-reload can nominally succeed while the unit
# is unexpectedly absent from the user manager, and installing nothing
# observable must fail loud rather than quietly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_TEMPLATE="lms-arm@.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [ ! -f "$SCRIPT_DIR/$UNIT_TEMPLATE" ]; then
    echo "ERROR: $SCRIPT_DIR/$UNIT_TEMPLATE not found" >&2
    exit 1
fi

mkdir -p "$UNIT_DIR"
cp "$SCRIPT_DIR/$UNIT_TEMPLATE" "$UNIT_DIR/"
echo "install-lms-units.sh: installed ${UNIT_DIR}/${UNIT_TEMPLATE}"

systemctl --user daemon-reload

# LMS_INSTALL_SABOTAGE exists so the fail-loud path is OBSERVED by a test
# rather than asserted about. Without it the negative case is unreachable:
# a real absent-after-reload requires breaking the user manager.
if [ "${LMS_INSTALL_SABOTAGE:-0}" = "1" ]; then
    rm -f "$UNIT_DIR/$UNIT_TEMPLATE"
fi

echo "install-lms-units.sh: verifying ${UNIT_TEMPLATE} is present after reload..."
# Capture first, then match against a here-string. Do NOT pipe into `grep -q`:
# under `set -o pipefail` grep exits the instant it matches, closing the pipe,
# and the producer is SIGPIPE'd mid-write (exit 141), so the check reports
# "not found" on a MATCH -- documented at
# scripts/legibility/install-trickle-timer.sh:64-69.
installed_units="$(ls -1 "$UNIT_DIR")"
if ! grep -qF "$UNIT_TEMPLATE" <<<"$installed_units"; then
    echo "ERROR: ${UNIT_TEMPLATE} is absent from ${UNIT_DIR} after daemon-reload" >&2
    exit 1
fi

echo "install-lms-units.sh: ${UNIT_TEMPLATE} installed."
echo "install-lms-units.sh: start an arm with:"
echo "    uv run --project shared python scripts/local-model-serving/lms_ctl.py start <arm_id>"
