#!/usr/bin/env bash
# install-memory-metadata-coverage-census-timer.sh -- install (or re-install,
# idempotently) the nightly canonical/topic stamping COVERAGE census +
# retro-stamp rehearsal systemd user timer (task 4006). Mirrors
# scripts/install-reify-closure-staleness-sweep-timer.sh.
#
# Usage: install-memory-metadata-coverage-census-timer.sh
#
# Copies the two memory-metadata-coverage-census.* unit files into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user systemd
# daemon, and enables+starts the timer (arming the nightly 05:00 recurrence).
# Idempotent: safe to re-run.
#
# DELIBERATELY DOES NOT KICK AN IMMEDIATE RUN, like the reify sweep installer
# and unlike the flag-marker and reclaim ones. Those drain a backlog that is
# safe to drain now. This job is a paginated full scroll of both live Qdrant
# collections that APPENDS a row to the committed trend history -- so an
# install-time firing would put an unreviewed, off-cadence row into the series
# the nightly timer owns, and would do it while an operator is watching the
# install rather than the report. An operator who wants a look right now runs
#
#   uv run --frozen --project fused-memory python \
#       fused-memory/scripts/census_memory_metadata.py --top-n 400 --no-history
#
# which measures and reports the trend without entering it. See OPERATIONS.md
# §"Nightly canonical/topic coverage census".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                            # unit files live here

SERVICE_NAME="memory-metadata-coverage-census.service"
TIMER_NAME="memory-metadata-coverage-census.timer"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/$SERVICE_NAME" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/$TIMER_NAME" "$UNIT_DIR/"

echo "install-memory-metadata-coverage-census-timer.sh: installing units into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

echo "install-memory-metadata-coverage-census-timer.sh: verifying ${TIMER_NAME} is listed..."
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

echo "install-memory-metadata-coverage-census-timer.sh: ${TIMER_NAME} installed and enabled (nightly 05:00)."
echo "install-memory-metadata-coverage-census-timer.sh: NO immediate run was kicked — the nightly run owns the trend history; use --no-history for an ad-hoc look. See OPERATIONS.md §12."
