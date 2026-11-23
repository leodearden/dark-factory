#!/usr/bin/env bash
# install-timeout-headroom-selfcheck-timer.sh -- install (or re-install,
# idempotently) the nightly timeout-headroom self-check systemd user timer
# (task 10940). Mirrors scripts/install-exception-owner-sweep-timer.sh.
#
# Usage: install-timeout-headroom-selfcheck-timer.sh
#
# Copies the two timeout-headroom-selfcheck.* unit files into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user systemd
# daemon, and enables+starts the timer (arming the nightly 01:30 recurrence).
# Idempotent: safe to re-run.
#
# DELIBERATELY DOES NOT KICK AN IMMEDIATE RUN. A run files an escalation for
# every breach it finds, so the first one belongs to the deployment that
# activates this timer, not to an install. An operator who wants a look right
# now runs the read-only
#
#   uv run --no-sync --project shared python scripts/timeout_headroom_selfcheck.py --dry-run
#
# and, for a real run, `systemctl --user start timeout-headroom-selfcheck.service`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                            # unit files live here

SERVICE_NAME="timeout-headroom-selfcheck.service"
TIMER_NAME="timeout-headroom-selfcheck.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-timeout-headroom-selfcheck-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "install-timeout-headroom-selfcheck-timer.sh: verifying ${TIMER_NAME} is listed..."
# Capture the full listing first, THEN grep the captured text: a
# `systemctl ... | grep -qF` pipe can kill systemctl with SIGPIPE, which
# pipefail turns into a false "timer not listed" error. The full rationale is
# on the same step in scripts/install-memory-metadata-coverage-census-timer.sh.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_NAME" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_NAME} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-timeout-headroom-selfcheck-timer.sh: ${TIMER_NAME} installed and enabled (nightly 01:30)."
echo "install-timeout-headroom-selfcheck-timer.sh: NO immediate run was kicked. For a read-only look:"
echo "  uv run --no-sync --project shared python scripts/timeout_headroom_selfcheck.py --dry-run"
echo "install-timeout-headroom-selfcheck-timer.sh: for a real run now:"
echo "  systemctl --user start ${SERVICE_NAME}"
