#!/usr/bin/env bash
# install-reify-closure-staleness-sweep-timer.sh -- install (or re-install,
# idempotently) the nightly reify closure-staleness sweep + re-dispatch-drain
# systemd user timer (task 3102). Mirrors
# scripts/install-reclaim-orphaned-worktrees-timer.sh.
#
# Usage: install-reify-closure-staleness-sweep-timer.sh
#
# Copies the two reify-closure-staleness-sweep.* unit files into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user systemd
# daemon, and enables+starts the timer (arming the nightly 04:30 recurrence).
# Idempotent: safe to re-run.
#
# DELIBERATELY DOES NOT KICK AN IMMEDIATE RUN, unlike its two siblings. Those
# jobs drain a backlog that is safe to drain now; this one MUTATES the live
# reify task store (cancelling gate-closed rows, re-dispatching stranded ones).
# The first execution against that store should be an operator-run
#
#   python3 scripts/consume_redispatch_requests.py --dry-run \
#       --requests-dir /home/leo/src/reify/data/redispatch-requests
#
# so the planned writes can be read before any of them happen. See
# OPERATIONS.md §"Nightly reify closure-staleness sweep".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                            # unit files live here

SERVICE_NAME="reify-closure-staleness-sweep.service"
TIMER_NAME="reify-closure-staleness-sweep.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-reify-closure-staleness-sweep-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "install-reify-closure-staleness-sweep-timer.sh: verifying ${TIMER_NAME} is listed..."
# Capture the full listing first, THEN grep the captured text. Piping
# `systemctl ... | grep -qF` is fragile under `set -o pipefail`: `grep -q`
# exits on the first match and closes the pipe, so `systemctl` -- still
# writing its trailing summary line -- dies with SIGPIPE, and pipefail turns
# that producer failure into a false "timer not listed" error. Running
# systemctl to completion into a variable removes the pipe entirely.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_NAME" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_NAME} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-reify-closure-staleness-sweep-timer.sh: ${TIMER_NAME} installed and enabled (nightly 04:30)."
echo "install-reify-closure-staleness-sweep-timer.sh: NO immediate run was kicked — do a --dry-run first; see OPERATIONS.md."
