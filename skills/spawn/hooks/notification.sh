#!/usr/bin/env bash
# notification.sh — Claude Code Notification hook entrypoint (Attention Rail T6).
#
# Fires when Claude Code needs the user's attention (awaiting input). Flips
# this session's session-registry record to status=awaiting-input and
# retitles the terminal tab via an emulator-agnostic OSC escape sequence.
#
# Thin by design (see skills/spawn/hooks/README.md): all logic lives in
# orchestrator/src/orchestrator/session_hooks.py, unit-tested in
# orchestrator/tests/test_session_hooks.py. This script only wires stdin
# passthrough + PYTHONPATH and writes the emitted OSC retitle to /dev/tty.
#
# Hard rule: this runs on every notification across every project, so it
# MUST be best-effort and MUST NOT block the turn — it always exits 0
# (mirrors ~/.claude/hooks/worktree-hookspath-capture.sh).

set +e

# Absolute path to the repo root, computed from this script's own location
# (skills/spawn/hooks/notification.sh -> hooks -> spawn -> skills -> repo
# root) so session_hooks.py can be invoked by absolute path with no venv or
# install required (same trick as skills/spawn/spawn-claude.sh).
REPO_ROOT="$(cd "$(dirname "$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")")" && pwd)" || exit 0

export PYTHONPATH="$REPO_ROOT/orchestrator/src${PYTHONPATH:+:$PYTHONPATH}"

command -v python3 >/dev/null 2>&1 || exit 0

osc="$(python3 "$REPO_ROOT/orchestrator/src/orchestrator/session_hooks.py" notification)"

printf '%s' "$osc" > /dev/tty 2>/dev/null || true

exit 0
