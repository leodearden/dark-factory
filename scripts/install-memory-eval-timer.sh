#!/usr/bin/env bash
# install-memory-eval-timer.sh -- install (or re-install, idempotently) the
# fused-memory-memory-eval systemd user timer (task 3211).
#
# Copies the .service and .timer into ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/,
# reloads the user daemon, enables the timer and verifies that list-timers
# shows it. It does NOT start the service: the first run is the grandfather
# snapshot and belongs on schedule. See OPERATIONS.md §"Scheduled memory evals".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE_NAME="fused-memory-memory-eval.service"
TIMER_NAME="fused-memory-memory-eval.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$SCRIPT_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$SCRIPT_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-memory-eval-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

# Captured, then grepped: `systemctl | grep -q` under pipefail can report a
# SIGPIPE'd producer as a missing timer.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_NAME" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_NAME} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi
echo "install-memory-eval-timer.sh: ${TIMER_NAME} installed and enabled (verified in list-timers)"
