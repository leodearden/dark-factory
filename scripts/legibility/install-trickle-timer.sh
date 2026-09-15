#!/usr/bin/env bash
# Install (or re-install, idempotently) the legibility-trickle systemd user
# timer for a single project.
#
# Usage: install-trickle-timer.sh <project_id>
#
# Resolves the project's docs/legibility/legibility.yaml by delegating to
# `nightly.py resolve-config <project_id>` (the single Python resolver --
# see legibility.nightly.resolve_config_path), so this script never
# re-implements project_id -> repo mapping and never guesses a directory
# name. Copies the two `legibility-trickle@.*` systemd user unit TEMPLATES
# (../legibility-trickle@.service/.timer -- %i is expanded by systemd
# itself at instantiation time, so the templates are copied verbatim, never
# per-project-customized) into ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/
# user/, reloads the user systemd daemon, and enables+starts the timer
# instance for <project_id>. Self-verifies the timer instance is actually
# listed afterward -- `enable --now` can nominally succeed while the unit
# is unexpectedly absent from `list-timers`, which must fail loud rather
# than silently install nothing observable. Idempotent: safe to re-run.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: install-trickle-timer.sh <project_id>" >&2
    exit 1
fi
PROJECT_ID="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/legibility
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                  # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                             # legibility-trickle@.service/.timer live here (sibling of legibility/)

# Interpreter prefix: INSTALL_TRICKLE_TIMER_PYTHON overrides the default `uv
# run` invocation (tests inject sys.executable to drive nightly.py's
# resolve-config directly, without uv/--frozen). Built as an array so the
# default branch keeps "$REPO_ROOT/shared" as a single argv token -- space
# -safe -- mirrors watcher-rearm.sh's WATCHER_REARM_PYTHON override
# (WATCHER_CMD=(uv run --project ...)).
if [ -n "${INSTALL_TRICKLE_TIMER_PYTHON:-}" ]; then
    # shellcheck disable=SC2206
    PYTHON_CMD=($INSTALL_TRICKLE_TIMER_PYTHON)
else
    PYTHON_CMD=(uv run --frozen --project "$REPO_ROOT/shared" python3)
fi

echo "install-trickle-timer.sh: resolving legibility.yaml for project_id=${PROJECT_ID}..."
if ! CONFIG_PATH="$("${PYTHON_CMD[@]}" "$SCRIPT_DIR/nightly.py" resolve-config "$PROJECT_ID")"; then
    echo "ERROR: could not resolve a legibility.yaml for project_id=${PROJECT_ID} (see above)" >&2
    exit 1
fi
echo "install-trickle-timer.sh: resolved config: ${CONFIG_PATH}"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/legibility-trickle@.service" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/legibility-trickle@.timer" "$UNIT_DIR/"

# Retire the 2026-09-14 account-pin stopgap (task 5488). NOT cleanup: that
# drop-in resets and re-spells ExecStart with one account's token inline, and a
# drop-in's `ExecStart=` reset REPLACES the unit's own -- so leaving it here
# keeps the trickle pinned to a single account forever while the unit file
# copied above says otherwise and every part of the multi-account change sits
# silently inert. Choosing an account is the gate's job now, per invocation.
#
# Exactly ONE named file, never the directory's contents: an operator's own
# override in the same .service.d must survive. `-f` and the `rmdir ... ||
# true` keep both steps no-ops under `set -euo pipefail` when there is nothing
# to remove, so a first-time install on a machine that never had the stopgap
# is not a hard failure. rmdir (not `rm -r`) is the guard that it only ever
# removes an EMPTY directory.
DROPIN_DIR="$UNIT_DIR/legibility-trickle@.service.d"
STALE_DROPIN="$DROPIN_DIR/10-account-pin.conf"
if [ -e "$STALE_DROPIN" ]; then
    rm -f "$STALE_DROPIN"
    rmdir "$DROPIN_DIR" 2>/dev/null || true
    echo "install-trickle-timer.sh: removed the stale 10-account-pin.conf drop-in (account choice is the usage gate's job now)"
fi

TIMER_UNIT="legibility-trickle@${PROJECT_ID}.timer"

systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_UNIT"

echo "install-trickle-timer.sh: verifying ${TIMER_UNIT} is listed..."
# Capture first, then match against a here-string. Do NOT pipe systemctl
# straight into `grep -q`: under `set -o pipefail` (line 20) grep exits the
# instant it matches, closing the pipe, and systemctl is then SIGPIPE'd
# mid-write -- a real systemctl exits 141, and the pipeline inherits that
# non-zero status, so the check reports "not found" even though grep MATCHED.
# A here-string is a simple command, not a pipeline, so pipefail cannot bite.
timers="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_UNIT" <<<"$timers"; then
    echo "ERROR: ${TIMER_UNIT} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-trickle-timer.sh: ${TIMER_UNIT} installed and enabled."
