#!/usr/bin/env bash
# install-link-heal-timer.sh -- install (or re-install, idempotently) the
# fused-memory-link-heal systemd user timer (task 7159).
#
# Copies the .service and .timer into ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/,
# reloads the user daemon, enables the timer and verifies that list-timers
# shows it. It does NOT start the service: the first sweep belongs on
# schedule. See OPERATIONS.md §"Nightly link-heal sweep (02:15)".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE_NAME="fused-memory-link-heal.service"
TIMER_NAME="fused-memory-link-heal.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$SCRIPT_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$SCRIPT_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-link-heal-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

# Captured, then grepped: `systemctl | grep -q` under pipefail can report a
# SIGPIPE'd producer as a missing timer.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_NAME" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_NAME} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi
echo "install-link-heal-timer.sh: ${TIMER_NAME} installed and enabled (verified in list-timers)"
