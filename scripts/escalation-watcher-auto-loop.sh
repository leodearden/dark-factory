#!/usr/bin/env bash
# Supervisor loop for the autonomous L1 escalation-watcher-auto skill.
# Runs claude -p in a rotation, restarting with a fresh context each time
# it exits cleanly (its own designed behaviour once rotation limits are
# hit). Scoped to exactly the tools the skill's own SKILL.md Hard
# Constraints section allows — no bypass-permissions mode needed, and no
# non-root user needed either, once two things are done correctly:
#
# 1. MCP tool calls (mcp__fused-memory__*, mcp__escalation__*) must be
#    named explicitly in --allowedTools. --allow-dangerously-skip-permissions
#    does NOT cover MCP tool calls in non-interactive `-p` sessions —
#    confirmed by direct A/B test (bypass mode alone: denied; explicit
#    --allowedTools naming the tool: works).
# 2. Env vars needed by scripts/watcher-rearm.sh (DARK_FACTORY_ROOT,
#    WATCHER_REARM_PYTHON) MUST be exported in the calling shell, never
#    inlined as a per-command prefix (`VAR=val script ...`) — the Bash
#    permission pattern matches against the literal invoked command
#    string, and an inline env-var prefix changes what that string starts
#    with, breaking a `Bash(/path/to/script *)` prefix match even though
#    the same script+args match fine once the vars are pre-exported.
#
# Env (all set by the CALLER before running this script -- see the guard
# below; this script never assumes any particular operator's directory
# layout):
#   DARK_FACTORY_ROOT           required; repo root, same meaning as in
#                                scripts/watcher-rearm.sh.
#   PROJECT_ROOT                required; the target project this loop
#                                watches escalations for (becomes the cwd
#                                claude -p runs in, and feeds
#                                watcher-rearm.sh's --queue-dir default).
#   WATCHER_REARM_PYTHON         optional interpreter override, passed
#                                through to watcher-rearm.sh. Only needed
#                                on a host with no system-wide `uv` on
#                                PATH; see the auto-fallback below. Leave
#                                unset on a normal `uv`-equipped host.
#   CLAUDE_CODE_OAUTH_TOKEN      optional; if already set in the caller's
#                                environment it is used as-is and the
#                                file fallback below is skipped entirely.
#   CLAUDE_CODE_OAUTH_TOKEN_FILE optional; path to read
#                                CLAUDE_CODE_OAUTH_TOKEN from when it
#                                isn't already set (default:
#                                $HOME/.config/claude-code/token). Credential
#                                storage is inherently deployment-specific,
#                                so this is a convenience fallback, not a
#                                requirement -- an operator who manages the
#                                token another way can just export
#                                CLAUDE_CODE_OAUTH_TOKEN before calling and
#                                this file read never happens.
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Silent-no-op guard, mirroring scripts/watcher-rearm.sh's own convention:
# every required input is loudly diagnosed on stderr and exits non-zero
# rather than silently defaulting to some particular operator's paths.
if [ -z "${DARK_FACTORY_ROOT:-}" ] || [ ! -d "${DARK_FACTORY_ROOT:-/nonexistent}" ]; then
    echo "escalation-watcher-auto-loop.sh: DARK_FACTORY_ROOT must be set to a valid directory" >&2
    exit 1
fi
if [ -z "${PROJECT_ROOT:-}" ] || [ ! -d "${PROJECT_ROOT:-/nonexistent}" ]; then
    echo "escalation-watcher-auto-loop.sh: PROJECT_ROOT must be set to a valid directory (the target project to watch escalations for)" >&2
    exit 1
fi

# WATCHER_REARM_PYTHON: watcher-rearm.sh's own default is `uv run --project
# ... python`, which is correct and needs no override on a host with `uv`
# on PATH. Only fall back to the repo-local venv interpreter when `uv`
# itself is missing (e.g. a droplet provisioned without it) -- and even
# then, derive the path from DARK_FACTORY_ROOT rather than hardcoding one
# operator's absolute path.
if ! command -v uv >/dev/null 2>&1; then
    export WATCHER_REARM_PYTHON="$DARK_FACTORY_ROOT/.venv/bin/python"
fi

# CLAUDE_CODE_OAUTH_TOKEN: only read from a file if the caller hasn't
# already exported the token some other way (secrets manager, systemd
# EnvironmentFile, etc). The file path itself is configurable so this
# fallback doesn't presume any one operator's credential-storage layout.
CLAUDE_CODE_OAUTH_TOKEN_FILE="${CLAUDE_CODE_OAUTH_TOKEN_FILE:-$HOME/.config/claude-code/token}"
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -f "$CLAUDE_CODE_OAUTH_TOKEN_FILE" ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$CLAUDE_CODE_OAUTH_TOKEN_FILE")"
fi

cd "$PROJECT_ROOT"

ALLOWED_TOOLS=(
  'mcp__fused-memory__update_task'
  'mcp__fused-memory__add_dependency'
  'mcp__fused-memory__get_task'
  'mcp__fused-memory__get_tasks'
  'mcp__fused-memory__search'
  'mcp__escalation__resolve_issue'
  'mcp__escalation__promote_to_l2'
  'mcp__escalation__stamp_triage'
  'mcp__escalation__get_pending_escalations'
  'Read'
  'Glob'
  'Grep'
  'Bash(git log *)'
  'Bash(git diff *)'
  'Bash(git show *)'
  'Bash(git status *)'
  "Bash($DARK_FACTORY_ROOT/scripts/watcher-rearm.sh *)"
  'Bash(printenv DARK_FACTORY_ROOT)'
  'Bash(printenv WATCHER_REARM_PYTHON)'
)

while true; do
  echo "$(date -u +%FT%TZ) escalation-watcher-auto-loop: starting rotation"
  claude -p "/escalation-watcher-auto ROTATION_ESCALATIONS=20 ROTATION_HOURS=1 DARK_FACTORY_ROOT=$DARK_FACTORY_ROOT (the bounded-wait wrapper is at $DARK_FACTORY_ROOT/scripts/watcher-rearm.sh — invoke it by that absolute path, DARK_FACTORY_ROOT and WATCHER_REARM_PYTHON — if this host needed the uv-fallback override — are already exported in your shell environment)" \
    --allowedTools "${ALLOWED_TOOLS[@]}" \
    2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) escalation-watcher-auto-loop: rotation exited rc=$rc — restarting in 5s"
  sleep 5
done
