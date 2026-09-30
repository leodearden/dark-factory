#!/usr/bin/env bash
# install-return-brief-timer.sh -- install (or re-install, idempotently) the
# nightly 05:30 cross-project return brief systemd user timer (task 5376).
# Mirrors scripts/install-memory-metadata-coverage-census-timer.sh.
#
# Copies return-brief.{service,timer} into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user daemon and
# enables the timer. It kicks NO immediate run: the prepare step spends a Fable
# budget, and an install-time firing would spend it off-cadence. For a look now,
# `RETURN_BRIEF_SKIP_PREPARE=1 scripts/return-brief.sh` re-renders at no model cost.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                            # unit files live here

SERVICE_NAME="return-brief.service"
TIMER_NAME="return-brief.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-return-brief-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "install-return-brief-timer.sh: verifying ${TIMER_NAME} is listed..."
# Capture, then grep the captured text: `systemctl ... | grep -q` under pipefail
# can SIGPIPE the producer and turn a listed timer into a false failure.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_NAME" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_NAME} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-return-brief-timer.sh: ${TIMER_NAME} installed and enabled (nightly 05:30)."
echo "install-return-brief-timer.sh: NO immediate run was kicked — the prepare step spends a Fable budget; re-render now at no model cost with RETURN_BRIEF_SKIP_PREPARE=1 scripts/return-brief.sh. See OPERATIONS.md §12."
