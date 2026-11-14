#!/usr/bin/env bash
# install-quality-snapshot-timer.sh -- install (or re-install, idempotently)
# the nightly quality-snapshot systemd user timer for one project (task 8464).
#
# Usage: install-quality-snapshot-timer.sh <project_root>
#
# Derives the project id from the root (shared.project_id.project_id_for_root)
# and has the cadence wrapper's own parser accept it. Refuses a root whose id is
# already installed for a different root. Writes the instance's root into
# ${XDG_CONFIG_HOME:-$HOME/.config}/dark-factory/quality-snapshot/<id>.env,
# copies the three quality-snapshot units verbatim into
# ${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/, reloads the user daemon,
# enables quality-snapshot@<id>.timer and verifies that list-timers shows it.
# It does NOT start the service: the first run belongs on schedule. See
# OPERATIONS.md §"Nightly quality snapshot (03:45)".
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: install-quality-snapshot-timer.sh <project_root>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# INSTALL_QUALITY_SNAPSHOT_TIMER_PYTHON overrides the interpreter (tests inject
# sys.executable), the INSTALL_TRICKLE_TIMER_PYTHON pattern.
if [ -n "${INSTALL_QUALITY_SNAPSHOT_TIMER_PYTHON:-}" ]; then
    # shellcheck disable=SC2206
    PYTHON_CMD=($INSTALL_QUALITY_SNAPSHOT_TIMER_PYTHON)
else
    PYTHON_CMD=("$REPO_ROOT/.venv/bin/python")
fi

if ! ROOT="$(cd "$1" 2>/dev/null && pwd -P)"; then
    echo "ERROR: $1 is not a directory" >&2
    exit 1
fi
if [[ ! "$ROOT" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "ERROR: $ROOT holds a character outside [A-Za-z0-9._/-]; an EnvironmentFile value is taken verbatim, so the unit could not be handed this root" >&2
    exit 1
fi
if ! git -C "$ROOT" rev-parse --verify --quiet 'main^{commit}' >/dev/null; then
    echo "ERROR: $ROOT is not a git checkout with a main commit to measure" >&2
    exit 1
fi
PROJECT_ID="$("${PYTHON_CMD[@]}" -c 'import sys; from shared.project_id import project_id_for_root; print(project_id_for_root(sys.argv[1]))' "$ROOT")"

# The unit hands the wrapper this id every night, so the wrapper's own parser
# judges it now: argparse converts --project-id before it reaches --help.
if ! "${PYTHON_CMD[@]}" "$SCRIPT_DIR/quality_snapshot_cadence.py" --root "$ROOT" --project-id "$PROJECT_ID" --help >/dev/null; then
    echo "ERROR: scripts/quality_snapshot_cadence.py refuses the project id ${PROJECT_ID} that $ROOT derives" >&2
    exit 1
fi

CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
UNIT_DIR="$CONFIG_HOME/systemd/user"
ENV_DIR="$CONFIG_HOME/dark-factory/quality-snapshot"
ENV_FILE="$ENV_DIR/${PROJECT_ID}.env"
ENV_LINE="QUALITY_SNAPSHOT_ROOT=$ROOT"
TIMER_UNIT="quality-snapshot@${PROJECT_ID}.timer"

if [ -e "$ENV_FILE" ] && [ "$(cat "$ENV_FILE")" != "$ENV_LINE" ]; then
    echo "ERROR: quality-snapshot@${PROJECT_ID} is already installed for another root: $ENV_FILE reads '$(cat "$ENV_FILE")', not '$ENV_LINE'. To move the instance to $ROOT, remove $ENV_FILE and re-run" >&2
    exit 1
fi

mkdir -p "$UNIT_DIR" "$ENV_DIR"
env_tmp="$(mktemp "$ENV_DIR/.${PROJECT_ID}.env.XXXXXX")"
printf '%s\n' "$ENV_LINE" >"$env_tmp"
chmod 0644 "$env_tmp"
mv -f "$env_tmp" "$ENV_FILE"

cp "$SCRIPT_DIR/quality-snapshot@.service" "$UNIT_DIR/"
cp "$SCRIPT_DIR/quality-snapshot@.timer" "$UNIT_DIR/"
cp "$SCRIPT_DIR/quality-snapshot-alarm@.service" "$UNIT_DIR/"

echo "install-quality-snapshot-timer.sh: installing ${TIMER_UNIT} for ${ROOT} into $UNIT_DIR"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_UNIT"

# Captured, then grepped: `systemctl | grep -q` under pipefail can report a
# SIGPIPE'd producer as a missing timer.
list_timers_out="$(systemctl --user list-timers --all)"
if ! grep -qF "$TIMER_UNIT" <<<"$list_timers_out"; then
    echo "ERROR: ${TIMER_UNIT} not found in 'systemctl --user list-timers --all' after enable" >&2
    exit 1
fi
echo "install-quality-snapshot-timer.sh: ${TIMER_UNIT} installed and enabled (verified in list-timers)"
