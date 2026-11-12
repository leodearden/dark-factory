#!/usr/bin/env bash
# install-transcript-secret-sweep-timer.sh -- install (or re-install,
# idempotently) the nightly transcript secret sweep systemd user timer (task
# 6820). Mirrors scripts/install-exception-owner-sweep-timer.sh.
#
# Usage: install-transcript-secret-sweep-timer.sh
#
# Copies the two transcript-secret-sweep.* unit files into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user systemd
# daemon, and enables+starts the timer (arming the nightly 02:30 recurrence).
# Idempotent: safe to re-run.
#
# DELIBERATELY DOES NOT KICK AN IMMEDIATE RUN. A run rewrites transcripts in
# place and files an escalation, so the first one belongs to the deployment
# that activates this timer, not to an install. For a run now:
# `systemctl --user start transcript-secret-sweep.service`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                            # unit files live here

SERVICE_NAME="transcript-secret-sweep.service"
TIMER_NAME="transcript-secret-sweep.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-transcript-secret-sweep-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "install-transcript-secret-sweep-timer.sh: verifying ${TIMER_NAME} is listed..."
# Capture the full listing first, THEN grep the captured text: a
# `systemctl ... | grep -qF` pipe can kill systemctl with SIGPIPE, which
# pipefail turns into a false "timer not listed" error. The full rationale is
# on the same step in scripts/install-memory-metadata-coverage-census-timer.sh.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_NAME" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_NAME} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-transcript-secret-sweep-timer.sh: ${TIMER_NAME} installed and enabled (nightly 02:30)."
echo "install-transcript-secret-sweep-timer.sh: NO immediate run was kicked. For a run now:"
echo "  systemctl --user start ${SERVICE_NAME}"
