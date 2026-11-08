#!/usr/bin/env bash
# install-exception-owner-sweep-timer.sh -- install (or re-install,
# idempotently) the nightly INV-12 owner sweep systemd user timer (task 5608).
# Mirrors scripts/install-memory-metadata-coverage-census-timer.sh.
#
# Usage: install-exception-owner-sweep-timer.sh
#
# Copies the two exception-owner-sweep.* unit files into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user systemd
# daemon, and enables+starts the timer (arming the nightly 06:00 recurrence).
# Idempotent: safe to re-run.
#
# DELIBERATELY DOES NOT KICK AN IMMEDIATE RUN. A run files tasks and
# escalations for every dead owner it finds, so the first one belongs to the
# deployment that activates this timer, not to an install. An operator who
# wants a look right now runs the read-only
#
#   uv run --project shared python scripts/exception_owner_sweep.py --dry-run
#
# and, for a real run, `systemctl --user start exception-owner-sweep.service`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                            # unit files live here

SERVICE_NAME="exception-owner-sweep.service"
TIMER_NAME="exception-owner-sweep.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-exception-owner-sweep-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "install-exception-owner-sweep-timer.sh: verifying ${TIMER_NAME} is listed..."
# Capture the full listing first, THEN grep the captured text: a
# `systemctl ... | grep -qF` pipe can kill systemctl with SIGPIPE, which
# pipefail turns into a false "timer not listed" error. The full rationale is
# on the same step in scripts/install-memory-metadata-coverage-census-timer.sh.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_NAME" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_NAME} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-exception-owner-sweep-timer.sh: ${TIMER_NAME} installed and enabled (nightly 06:00)."
echo "install-exception-owner-sweep-timer.sh: NO immediate run was kicked. For a read-only look:"
echo "  uv run --project shared python scripts/exception_owner_sweep.py --dry-run"
echo "install-exception-owner-sweep-timer.sh: for a real run now:"
echo "  systemctl --user start ${SERVICE_NAME}"
