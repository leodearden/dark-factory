#!/usr/bin/env bash
# install-flag-marker-sweep-timer.sh -- install (or re-install, idempotently)
# the fused-memory-flag-marker-sweep systemd user timer + kick an immediate
# one-time drain of the current backlog (task 2693).
#
# Usage: install-flag-marker-sweep-timer.sh
#
# Unlike scripts/legibility/install-trickle-timer.sh, this is a single
# non-templated dark_factory job (no project_id argument, no per-project
# config resolution) -- copies the two fused-memory-flag-marker-sweep.*
# systemd user unit files into ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/
# user/, reloads the user systemd daemon, enables+starts the timer (arms
# nightly 03:30 recurrence), verifies via `list-timers` that the enable
# actually took, and only THEN starts the service once immediately (drains
# the current backlog now rather than waiting for the next scheduled run).
#
# The self-verify runs BEFORE the one-time drain kick, not after. The
# drain kick starts a `Type=oneshot` unit, so `systemctl start` blocks
# until the sweep finishes and propagates its exit status -- if backing
# stores are down or OPENAI_API_KEY is unset, that kick fails, and under
# `set -e` an ordering with the verify last would abort the script right
# there, skipping the self-verify even though the timer is already
# installed and enabled. Verifying first means the install's stated
# guarantee (the timer IS armed) is always checked, regardless of whether
# the immediate drain succeeds; a failed drain kick still exits non-zero,
# but loudly names itself as the cause rather than masquerading as an
# install failure.
# Idempotent: safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                            # unit files live here

SERVICE_NAME="fused-memory-flag-marker-sweep.service"
TIMER_NAME="fused-memory-flag-marker-sweep.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-flag-marker-sweep-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "install-flag-marker-sweep-timer.sh: verifying ${TIMER_NAME} is listed..."
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
echo "install-flag-marker-sweep-timer.sh: ${TIMER_NAME} installed and enabled (verified in list-timers)"

echo "install-flag-marker-sweep-timer.sh: kicking immediate one-time drain via $SERVICE_NAME"
if ! systemctl --user start "$SERVICE_NAME"; then
    echo "ERROR: ${SERVICE_NAME} one-time drain kick failed (the timer IS installed and enabled; the nightly run is armed). Inspect with: journalctl --user -u ${SERVICE_NAME}" >&2
    exit 1
fi

echo "install-flag-marker-sweep-timer.sh: ${SERVICE_NAME} one-time drain kick complete."
