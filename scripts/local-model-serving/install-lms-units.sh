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
# Self-verifies afterwards, in TWO layers:
#
#   1. PRESENCE -- daemon-reload can nominally succeed while the unit is
#      unexpectedly absent from the user manager, and installing nothing
#      observable must fail loud rather than quietly.
#   2. EFFECTIVE CONFIGURATION -- via scripts/check_lms_unit_parity.py, which
#      compares the installed copy against this template AND asks systemd what
#      it would actually apply.
#
# Layer 2 exists because layer 1 is structurally blind to the thing that
# actually redirects a unit.  `systemctl --user edit` never modifies the unit
# file; it writes lms-arm@.service.d/override.conf beside it, and systemd
# merges that over the unit at load time.  The observed instance pinned
# WorkingDirectory at a worktree, so the file was present and byte-identical
# while every arm served a frozen tree -- and re-running this installer
# reported success without clearing it.  Presence was never the claim an
# operator needed; "the committed template is what systemd will apply" is.
#
# A drop-in found this way is NAMED AND REFUSED, never removed.  That is
# deliberate (task 3750): the observed drop-in was LOAD-BEARING while its
# worktree was unmerged, so deleting it would have pointed every arm at a
# directory with no launcher.  Removal has a correct owner already,
# scripts/remove-lms-arm-worktree-dropin.sh, which gates it behind
# preconditions an installer does not check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UNIT_TEMPLATE="lms-arm@.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
PARITY_CHECKER="$REPO_ROOT/scripts/check_lms_unit_parity.py"

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

# Layer 2: what will systemd ACTUALLY apply? Purely additive -- the presence
# check above still runs first and still exits first, so its observable
# failure path is unchanged.
#
# Under `set -euo pipefail` a bare invocation would abort here with no
# explanation at all, so the exit code is captured (the shape setup-host.sh
# uses for the same checkers).
echo "install-lms-units.sh: verifying the effective configuration..."
if python3 "$PARITY_CHECKER" --installed-dir "$UNIT_DIR" --repo-root "$REPO_ROOT"; then
    :
else
    parity_rc=$?
    echo "ERROR: the install did NOT produce a clean effective configuration" >&2
    echo "       (check_lms_unit_parity.py exited ${parity_rc}; see its" >&2
    echo "       [lms_unit_parity] report above for the specific finding)." >&2
    echo "" >&2
    echo "  An [override] or [effective] finding means a drop-in or a resolved" >&2
    echo "  WorkingDirectory is winning over ${UNIT_TEMPLATE}. A drop-in SURVIVES" >&2
    echo "  reinstallation, and is deliberately NOT removed here: it can be" >&2
    echo "  load-bearing, so deleting it blindly is the unsafe option." >&2
    echo "" >&2
    echo "  Inspect the merged result:" >&2
    echo "      systemctl --user cat lms-arm@<arm>.service" >&2
    echo "  Remove the known worktree drop-in, with its safety preconditions:" >&2
    echo "      scripts/remove-lms-arm-worktree-dropin.sh" >&2
    exit 1
fi

echo "install-lms-units.sh: ${UNIT_TEMPLATE} installed."
echo "install-lms-units.sh: start an arm with:"
echo "    uv run --project shared python scripts/local-model-serving/lms_ctl.py start <arm_id>"
