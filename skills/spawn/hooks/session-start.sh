#!/usr/bin/env bash
# session-start.sh — Claude Code SessionStart hook entrypoint (Attention Rail T6).
#
# Fires on every Claude Code session start, including Leo's hand-launched
# ones. Captures/refreshes this session's session-registry record
# (status=running).
#
# Thin by design (see skills/spawn/hooks/README.md): all logic lives in
# orchestrator/src/orchestrator/session_hooks.py, unit-tested in
# orchestrator/tests/test_session_hooks.py. This script only wires stdin
# passthrough + PYTHONPATH and writes any emitted OSC retitle to /dev/tty.
#
# Hard rule: this runs on every session start across every project, so it
# MUST be best-effort and MUST NOT block the session — it always exits 0
# (mirrors ~/.claude/hooks/worktree-hookspath-capture.sh).

set +e

# Absolute path to the repo root, computed from this script's own location
# (skills/spawn/hooks/session-start.sh -> hooks -> spawn -> skills -> repo
# root) so session_hooks.py can be invoked by absolute path with no venv or
# install required (same trick as skills/spawn/spawn-claude.sh).
REPO_ROOT="$(cd "$(dirname "$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")")" && pwd)" || exit 0

export PYTHONPATH="$REPO_ROOT/orchestrator/src${PYTHONPATH:+:$PYTHONPATH}"

command -v python3 >/dev/null 2>&1 || exit 0

osc="$(python3 "$REPO_ROOT/orchestrator/src/orchestrator/session_hooks.py" session-start)"

printf '%s' "$osc" > /dev/tty 2>/dev/null || true

exit 0
