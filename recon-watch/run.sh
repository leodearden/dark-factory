#!/usr/bin/env bash
# Launch a dedicated Claude session that watches the fused-memory RECONCILIATION
# escalation queue (port 8103) via the /recon-escalation-watcher skill.
#
# MCP connections are per-process, so this leaves any orchestrator-queue
# (8102/8100) escalation-watcher session untouched: this session's `escalation`
# server is pinned to 8103 by --strict-mcp-config + mcp.json (which also wires
# `fused-memory` at 8002 for the watcher's fix/file actions).
#
# Prereqs: fused-memory.service running (hosts both 8103 and 8002).
#   systemctl --user status fused-memory.service
#
# Usage:
#   ./recon-watch/run.sh            # autonomous (skip permission prompts)
#   ./recon-watch/run.sh --interactive   # prompt for each tool use
#   extra args are passed through to `claude`.
set -euo pipefail

export DARK_FACTORY_ROOT="${DARK_FACTORY_ROOT:-/home/leo/src/dark-factory}"
CONFIG="$DARK_FACTORY_ROOT/recon-watch/mcp.json"

PERM=(--dangerously-skip-permissions)   # autonomous by default; long-running loop
if [[ "${1:-}" == "--interactive" ]]; then
  PERM=()
  shift
fi

cd "$DARK_FACTORY_ROOT"
exec claude --strict-mcp-config --mcp-config "$CONFIG" "${PERM[@]}" "$@" \
  "/recon-escalation-watcher"
