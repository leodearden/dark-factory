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

TIMER_UNIT="legibility-trickle@${PROJECT_ID}.timer"

systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_UNIT"

echo "install-trickle-timer.sh: verifying ${TIMER_UNIT} is listed..."
if ! systemctl --user list-timers --all | grep -qF "$TIMER_UNIT"; then
    echo "ERROR: ${TIMER_UNIT} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-trickle-timer.sh: ${TIMER_UNIT} installed and enabled."
