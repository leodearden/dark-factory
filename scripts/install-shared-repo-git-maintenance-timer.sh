#!/usr/bin/env bash
# install-shared-repo-git-maintenance-timer.sh -- install (or re-install,
# idempotently) the shared-repo-git-maintenance systemd user timer and kick
# its first run (task 8076). Mirrors scripts/install-reclaim-orphaned-worktrees-timer.sh.
#
# Usage: install-shared-repo-git-maintenance-timer.sh
#
# The units share their names with the 2026-11-01 hand install, so running
# this replaces that install in place rather than leaving two timers
# repacking the same repos.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                            # unit files live here

SERVICE_NAME="shared-repo-git-maintenance.service"
TIMER_NAME="shared-repo-git-maintenance.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-shared-repo-git-maintenance-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

# --no-block: a fresh host's first repack can take minutes.
echo "install-shared-repo-git-maintenance-timer.sh: kicking the first run via $SERVICE_NAME"
systemctl --user start --no-block "$SERVICE_NAME"

echo "install-shared-repo-git-maintenance-timer.sh: verifying ${TIMER_NAME} is listed..."
# Captured, then grepped: `systemctl ... | grep -qF` under pipefail can turn
# grep's early exit into a SIGPIPE failure of systemctl.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_NAME" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_NAME} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-shared-repo-git-maintenance-timer.sh: ${TIMER_NAME} installed and enabled; first run kicked."
