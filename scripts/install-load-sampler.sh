#!/usr/bin/env bash
# install-load-sampler.sh -- install (or re-install, idempotently) the
# dark-factory-load-sampler systemd user timer, then verify a sample row
# actually landed (task 3592, leaf δ of plans/load-throttle-harmonisation-prd.md).
#
# Usage: install-load-sampler.sh
#
# Copies the two dark-factory-load-sampler.* unit files into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user systemd
# daemon, enables+starts the timer (arming the 5s recurrence), verifies via
# `list-timers` that the enable actually took, and only THEN starts the
# oneshot service once to prove end to end that a tick writes a row.
# Idempotent: safe to re-run.
#
# The unit files live in dashboard/, not scripts/ -- which is the only way
# this installer differs in shape from its siblings.
#
# DELIBERATELY NOT scripts/setup-host.sh. That script re-renders EVERY unit on
# the host and is the known clobber hazard; this one touches exactly the two
# units it owns. Task δ-prime is the deterministic deploy that runs this
# script; ε1/ε2 are the calibration gates that follow it.
#
# The self-verify runs BEFORE the one-time kick, not after -- the ordering
# scripts/install-flag-marker-sweep-timer.sh's header documents as the
# corrected one. The kick starts a `Type=oneshot` unit, so `systemctl start`
# blocks until the tick finishes and propagates its exit status; under
# `set -e` an ordering with the verify last would abort the script right
# there, skipping the self-verify even though the timer is already installed
# and enabled. Verifying first means the install's stated guarantee (the timer
# IS armed) is always checked.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/dashboard"                          # unit files live here

SERVICE_NAME="dark-factory-load-sampler.service"
TIMER_NAME="dark-factory-load-sampler.timer"

# The DB the sampler actually writes is resolved through the SAME env seam
# sampler/src/sampler/__main__.py honours, so operator, installer, sampler and
# calibration script cannot disagree about which file is the corpus. The
# unit's WorkingDirectory is %h/src/dark-factory, so a repo-relative path here
# would make the installer verify a file nothing writes when run from a
# worktree.
SAMPLE_DB="${DARK_FACTORY_ROOT:-$HOME/src/dark-factory}/data/load-samples.db"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-load-sampler.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "install-load-sampler.sh: verifying ${TIMER_NAME} is listed..."
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
echo "install-load-sampler.sh: ${TIMER_NAME} installed and enabled (verified in list-timers)"

# High-water mark BEFORE the kick, so the row check proves THIS run wrote
# something rather than finding an old row from a previous install.
pre_kick_ts="$(date +%s)"

echo "install-load-sampler.sh: kicking one immediate tick via $SERVICE_NAME"
if ! systemctl --user start "$SERVICE_NAME"; then
    echo "ERROR: ${SERVICE_NAME} kick failed (the timer IS installed and enabled; the 5s recurrence is armed). Inspect with: journalctl --user -u ${SERVICE_NAME}" >&2
    exit 1
fi

echo "install-load-sampler.sh: verifying a sample row landed in $SAMPLE_DB ..."
if [ ! -f "$SAMPLE_DB" ]; then
    echo "ERROR: no sample DB at ${SAMPLE_DB} after the kick. The timer is armed but the sampler wrote nothing. Inspect with: journalctl --user -u ${SERVICE_NAME}" >&2
    exit 1
fi
# The metric COUNT is deliberately not asserted: a tick writes 11 rows plus
# two per cgroup leaf discovered at collection time, so the number is
# host-dependent (25 here, with seven orchestrator-*.service leaves). What
# matters is that rows arrived at all.
row_count="$(sqlite3 "$SAMPLE_DB" "SELECT COUNT(*) FROM samples WHERE ts >= ${pre_kick_ts};")"
if [ "$row_count" -eq 0 ]; then
    echo "ERROR: no rows at or after ${pre_kick_ts} in ${SAMPLE_DB}. The timer is armed but the tick wrote nothing. Inspect with: journalctl --user -u ${SERVICE_NAME}" >&2
    exit 1
fi

echo "install-load-sampler.sh: ${row_count} sample rows written by the kick; install complete."
