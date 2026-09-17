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
# blocks until the tick finishes and propagates its exit status, and a failing
# kick ends the run: not via `set -e` (the `if ! systemctl ...` wrapper below
# catches the status so the message can name the unit) but by an explicit
# exit 1. Verifying first means the install's stated guarantee -- the timer IS
# armed -- is checked even on a host where the tick itself cannot run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/dashboard"                          # unit files live here

SERVICE_NAME="dark-factory-load-sampler.service"
TIMER_NAME="dark-factory-load-sampler.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-load-sampler.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

# The DB to verify is the one THE INSTALLED UNIT will write, asked of systemd
# rather than assumed -- `show --value` answers with the effective directory,
# %h already expanded and any drop-in applied.
#
# This deliberately does NOT read $DARK_FACTORY_ROOT. sampler/__main__.py
# resolves <DARK_FACTORY_ROOT or CWD>/data/load-samples.db, and the unit sets
# no Environment=, so under systemd the root is the CWD -- i.e. exactly
# WorkingDirectory. Reading the variable from the INVOKING shell instead made
# the two disagree precisely when it was set: run from a worktree with
# DARK_FACTORY_ROOT exported, the sampler still wrote $HOME/src/dark-factory
# while the installer checked the worktree and exited 1 with the misleading
# "the sampler wrote nothing". Propagating the variable INTO the unit was the
# other way to make them agree and is worse: the corpus is a 30-day artifact
# and a worktree is deleted when its task lands, so the collector would be
# aimed at a directory that disappears. Relocate the checkout by editing
# WorkingDirectory -- one knob, honoured by unit, sampler and this check.
unit_workdir="$(systemctl --user show -p WorkingDirectory --value "$SERVICE_NAME")"
if [ -z "$unit_workdir" ]; then
    echo "ERROR: systemd reported no WorkingDirectory for ${SERVICE_NAME}; cannot tell which DB the sampler writes. Inspect with: systemctl --user cat ${SERVICE_NAME}" >&2
    exit 1
fi
SAMPLE_DB="$unit_workdir/data/load-samples.db"

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
