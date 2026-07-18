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
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export DARK_FACTORY_ROOT=/root/dark-factory
export WATCHER_REARM_PYTHON=/root/dark-factory/.venv/bin/python
export CLAUDE_CODE_OAUTH_TOKEN="$(cat /root/.config/claude-code/token)"
export PROJECT_ROOT=/root/riffchain

cd /root/riffchain

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
  'Bash(/root/dark-factory/scripts/watcher-rearm.sh *)'
  'Bash(printenv DARK_FACTORY_ROOT)'
  'Bash(printenv WATCHER_REARM_PYTHON)'
)

while true; do
  echo "$(date -u +%FT%TZ) escalation-watcher-auto-loop: starting rotation"
  claude -p '/escalation-watcher-auto ROTATION_ESCALATIONS=20 ROTATION_HOURS=1 DARK_FACTORY_ROOT=/root/dark-factory (the bounded-wait wrapper is at /root/dark-factory/scripts/watcher-rearm.sh — invoke it by that absolute path, DARK_FACTORY_ROOT and WATCHER_REARM_PYTHON are already exported in your shell environment)' \
    --allowedTools "${ALLOWED_TOOLS[@]}" \
    2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) escalation-watcher-auto-loop: rotation exited rc=$rc — restarting in 5s"
  sleep 5
done
