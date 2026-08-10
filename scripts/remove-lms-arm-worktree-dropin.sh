#!/usr/bin/env bash
# Remove the temporary lms-arm@ worktree drop-in installed for task 3713.
#
# Task 3750, run as a deterministic `before_done` DEPLOY action with
# always_escalates=false: it performs the removal and exits non-zero ONLY when
# something is genuinely wrong, so the orchestrator escalates on exception
# rather than on every run.
#
# CONTEXT
# -------
# Task 3713's steward installed
#     ~/.config/systemd/user/lms-arm@.service.d/10-worktree-3713.conf
# which overrides WorkingDirectory to
#     /home/leo/src/dark-factory/.worktrees/3713
# because the serving scripts did not yet exist on main.  The override applies
# to the TEMPLATE, so it affects EVERY lms-arm@<arm> instance, not one arm.
#
# WHY THIS MUST NOT RUN EARLY
# ---------------------------
# If the serving scripts are not on main, removing the drop-in points every arm
# at a directory with no launcher.  Task 3750 depends on 3713, and 3713's
# delivered-check already gates on its serving marker being present on main --
# but this script RE-CHECKS independently rather than trusting the scheduler
# (defence in depth: the scheduler proves the marker landed, P1 below proves
# the file this unit actually execs is there).
#
# DELIBERATELY NOT SPELLED OUT HERE
# ---------------------------------
# Task 3713's delivered-check greps scripts/ for its serving marker literal.
# Writing that literal into this file would FALSELY satisfy 3713's check on
# main and release task 3750's own dependency gate early -- the exact ordering
# failure this script exists to prevent.  Refer to the marker; never reproduce
# it in this directory.
#
# SCOPE OF VERIFICATION
# ---------------------
# This asserts systemd's RESOLVED configuration plus the syntactic loadability
# of the launcher from the merged path.  It deliberately does NOT start an arm:
# that loads multi-GB weights and holds the whole GPU, and would contend with
# whisper-writer.  A config-level pass is therefore necessary but not
# sufficient -- functional proof belongs to the consumers (tasks 3720/3721/
# 3722), which perform real runs.
#
# Idempotent: safe to re-run.  An already-absent drop-in is a no-op that still
# re-verifies the end state.

set -euo pipefail

# The three LMS_* overrides below are TEST SEAMS, not tuning knobs.  Their
# defaults are the production values; scripts/tests/test_remove_lms_dropin.sh
# uses them to exercise this script's success path end-to-end against a
# throwaway systemd template, because a script that runs UNSANDBOXED and
# UNATTENDED must not have its happy path first execute in production.
REPO="${LMS_REPO_ROOT:-/home/leo/src/dark-factory}"
TEMPLATE="${LMS_UNIT_TEMPLATE:-lms-arm@}"
DROPIN_NAME="${LMS_DROPIN_NAME:-10-worktree-3713.conf}"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/${TEMPLATE}.service"
DROPIN_DIR="$UNIT_DIR/${TEMPLATE}.service.d"
DROPIN="$DROPIN_DIR/$DROPIN_NAME"
LAUNCHER="$REPO/scripts/local-model-serving/lms_serve.py"
PROBE="${TEMPLATE}probe.service"

fail() { echo "remove-lms-arm-worktree-dropin: ERROR: $*" >&2; exit 1; }
note() { echo "remove-lms-arm-worktree-dropin: $*"; }

# --- Preconditions -------------------------------------------------------

# P1. The merged world must actually carry the launcher.  This is the check
# that makes removal safe; without it every arm loses its ExecStart target and
# the units break in exactly the world they are finally supposed to work in.
[ -f "$LAUNCHER" ] || fail \
  "launcher absent at $LAUNCHER -- task 3713 has not landed on main. REFUSING to remove the drop-in: doing so now would break every lms-arm@ instance."

# P2. The template must still be installed.  If it is gone, this task's premise
# no longer holds; a human should look rather than have us silently "succeed".
[ -f "$UNIT" ] || fail \
  "${TEMPLATE}.service template absent from $UNIT_DIR -- nothing to un-override; investigate before re-running."

# P3. The launcher must be loadable from the merged path.  Cheap syntactic
# proof that the merge produced a usable file, without starting an arm.
python3 -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')" "$LAUNCHER" \
  || fail "launcher at $LAUNCHER does not compile -- refusing to remove the drop-in."

# --- Action (idempotent) -------------------------------------------------

if [ -e "$DROPIN" ]; then
    rm -f "$DROPIN"
    note "removed $DROPIN"
else
    note "drop-in already absent at $DROPIN (idempotent no-op)"
fi

# Only removes the directory when it is empty, so a foreign drop-in somebody
# else added is left alone rather than swept up with ours.
rmdir --ignore-fail-on-non-empty "$DROPIN_DIR" 2>/dev/null || true

systemctl --user daemon-reload \
  || fail "systemctl --user daemon-reload failed -- is XDG_RUNTIME_DIR reachable from this context?"

# --- Verification --------------------------------------------------------

wd="$(systemctl --user show -p WorkingDirectory --value "$PROBE")" \
  || fail "could not read WorkingDirectory for $PROBE"
dp="$(systemctl --user show -p DropInPaths --value "$PROBE")" || dp=""

[ "$wd" = "$REPO" ] || fail \
  "WorkingDirectory resolves to '$wd', expected '$REPO' -- the override was not fully removed."

case "$dp" in
  *"${TEMPLATE}.service.d"*) fail "a drop-in still applies to the template: $dp" ;;
esac

note "OK: $PROBE resolves WorkingDirectory=$wd with no ${TEMPLATE} drop-in applied; template still installed at $UNIT"
