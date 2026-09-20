#!/usr/bin/env bash
# Install (or re-install, idempotently) the legibility-trickle-health systemd
# user timer for a single project (task 4514, GAP 2 -- the thing that finally
# RUNS check_trickle_progress.py and check_trickle_liveness.sh, neither of
# which any unit, cron entry or config had ever bound).
#
# Usage: install-trickle-health-timer.sh <project_id>
#
# Resolves the project's docs/legibility/legibility.yaml by delegating to
# `nightly.py resolve-config <project_id>` (the single Python resolver --
# see legibility.nightly.resolve_config_path -- which the health probe's own
# --project-id path also resolves through, so a successful resolve here
# guarantees the ExecStart will resolve at runtime), so this script never
# re-implements project_id -> repo mapping and never guesses a directory
# name. Copies the two `legibility-trickle-health@.*` systemd user unit
# TEMPLATES (../legibility-trickle-health@.service/.timer -- %i is expanded
# by systemd itself at instantiation time, so the templates are copied
# verbatim, never per-project-customized) into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user systemd
# daemon, and enables+starts the timer instance for <project_id>.
# Self-verifies the timer instance is actually listed afterward -- `enable
# --now` can nominally succeed while the unit is unexpectedly absent from
# `list-timers`, which must fail loud rather than silently install nothing
# observable. That gap is exactly how a probe ends up shipped-but-unbound,
# which IS the defect this job closes. Idempotent: safe to re-run.
#
# Strictly ADDITIVE: this script never touches `legibility-trickle@.*`. The
# two jobs' unit names differ by one word and the nightly one is what actually
# produces the digests, so that separation is pinned by
# scripts/tests/test_install_trickle_health_timer.py::test_installer_does_not_touch_the_trickle_units.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: install-trickle-health-timer.sh <project_id>" >&2
    exit 1
fi
PROJECT_ID="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/legibility
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                  # dark-factory repo root
TEMPLATES_DIR="$REPO_ROOT/scripts"                             # legibility-trickle-health@.service/.timer live here (sibling of legibility/)

# Interpreter prefix: INSTALL_TRICKLE_HEALTH_TIMER_PYTHON overrides the
# default `uv run` invocation (tests inject sys.executable to drive
# nightly.py's resolve-config directly, without uv/--frozen). Built as an
# array so the default branch keeps "$REPO_ROOT/shared" as a single argv
# token -- space-safe -- mirrors install-trickle-timer.sh's
# INSTALL_TRICKLE_TIMER_PYTHON override.
if [ -n "${INSTALL_TRICKLE_HEALTH_TIMER_PYTHON:-}" ]; then
    # shellcheck disable=SC2206
    PYTHON_CMD=($INSTALL_TRICKLE_HEALTH_TIMER_PYTHON)
else
    PYTHON_CMD=(uv run --frozen --project "$REPO_ROOT/shared" python3)
fi

echo "install-trickle-health-timer.sh: resolving legibility.yaml for project_id=${PROJECT_ID}..."
if ! CONFIG_PATH="$("${PYTHON_CMD[@]}" "$SCRIPT_DIR/nightly.py" resolve-config "$PROJECT_ID")"; then
    echo "ERROR: could not resolve a legibility.yaml for project_id=${PROJECT_ID} (see above)" >&2
    exit 1
fi
echo "install-trickle-health-timer.sh: resolved config: ${CONFIG_PATH}"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"

cp "$TEMPLATES_DIR/legibility-trickle-health@.service" "$UNIT_DIR/"
cp "$TEMPLATES_DIR/legibility-trickle-health@.timer" "$UNIT_DIR/"

TIMER_UNIT="legibility-trickle-health@${PROJECT_ID}.timer"

systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_UNIT"

echo "install-trickle-health-timer.sh: verifying ${TIMER_UNIT} is listed..."
# Capture first, then match against a here-string. Do NOT pipe systemctl
# straight into `grep -q`: under `set -o pipefail` (line 31) grep exits the
# instant it matches, closing the pipe, and systemctl is then SIGPIPE'd
# mid-write -- a real systemctl exits 141, and the pipeline inherits that
# non-zero status, so the check reports "not found" even though grep MATCHED.
# A here-string is a simple command, not a pipeline, so pipefail cannot bite.
# Task 3527 (commit 033613c4b0) fixed this in both sibling installers.
timers="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_UNIT" <<<"$timers"; then
    echo "ERROR: ${TIMER_UNIT} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi

echo "install-trickle-health-timer.sh: ${TIMER_UNIT} installed and enabled."
