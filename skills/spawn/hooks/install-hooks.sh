#!/usr/bin/env bash
# install-hooks.sh — merge the SessionStart/Notification/Stop trio into the
# real ~/.claude/settings.json (Attention Rail T6).
#
# Idempotent: safe to re-run. Delegates entirely to
# orchestrator/src/orchestrator/session_hooks.py's `install` CLI verb, which
# reads the current settings, merges in the trio (adding ONLY the 3 new
# event keys; every existing event and other top-level key is left
# byte-identical), takes a timestamped backup of the pre-merge settings.json,
# and writes atomically (tmp file + os.replace) so a crash mid-write can
# never leave settings.json truncated or corrupt. See
# skills/spawn/hooks/README.md.
#
# Unlike the per-event hook entrypoints (session-start.sh/notification.sh/
# stop.sh), this is a human-invoked one-shot command, not a fail-soft hook:
# it propagates session_hooks.py's own exit code (0 on success, 1 if the
# install itself failed) so a failed install is never silently reported as
# a success.

set +e

# Absolute path to the repo root, computed from this script's own location
# (skills/spawn/hooks/install-hooks.sh -> hooks -> spawn -> skills -> repo
# root) so session_hooks.py can be invoked by absolute path with no venv or
# install required (same trick as skills/spawn/spawn-claude.sh).
REPO_ROOT="$(cd "$(dirname "$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")")" && pwd)" || exit 0

export PYTHONPATH="$REPO_ROOT/orchestrator/src${PYTHONPATH:+:$PYTHONPATH}"

command -v python3 >/dev/null 2>&1 || exit 0

PY="$REPO_ROOT/orchestrator/src/orchestrator/session_hooks.py"
python3 "$PY" install
exit $?
