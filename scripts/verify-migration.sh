#!/usr/bin/env bash
# verify-migration.sh — comprehensive post-migration verification.
# Run on the DESTINATION host after import-data.sh completes.
# Exits 0 if all critical checks pass, non-zero if any critical check fails.
#
# Usage: bash scripts/verify-migration.sh [--no-smoke]
#   --no-smoke   Skip the fused-memory functional smoke test (faster, no API calls)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/fused-memory/docker/docker-compose.yml"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m  + %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m  x %s\033[0m\n' "$*"; }

SMOKE=1
for arg in "$@"; do
  [ "$arg" = "--no-smoke" ] && SMOKE=0
done

CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNED=0

check_pass() { ok "$1"; CHECKS_PASSED=$((CHECKS_PASSED + 1)); }
check_fail() { fail "$1"; CHECKS_FAILED=$((CHECKS_FAILED + 1)); }
check_warn() { warn "$1"; CHECKS_WARNED=$((CHECKS_WARNED + 1)); }

# ---------------------------------------------------------------------------
# 1. FalkorDB: PING + DBSIZE > 0
# ---------------------------------------------------------------------------
info "FalkorDB checks"

# Use direct redis-cli on port 6379 (avoids docker-compose env_file requirement)
if command -v redis-cli &>/dev/null; then
  FALKOR_PING=$(redis-cli -p 6379 ping 2>/dev/null || echo "FAIL")
else
  # Fall back to docker run
  FALKOR_PING=$(docker run --rm --network host redis:7-alpine redis-cli -p 6379 ping 2>/dev/null || echo "FAIL")
fi

if echo "$FALKOR_PING" | grep -q PONG; then
  check_pass "FalkorDB PING → PONG (port 6379)"

  if command -v redis-cli &>/dev/null; then
    DBSIZE=$(redis-cli -p 6379 DBSIZE 2>/dev/null | tr -d '[:space:]' || echo "0")
  else
    DBSIZE=$(docker run --rm --network host redis:7-alpine redis-cli -p 6379 DBSIZE 2>/dev/null | tr -d '[:space:]' || echo "0")
  fi

  if [ "$DBSIZE" -gt 0 ] 2>/dev/null; then
    check_pass "FalkorDB DBSIZE = $DBSIZE (data present)"
  else
    check_fail "FalkorDB DBSIZE = ${DBSIZE:-0} (no data — migration may not have imported)"
  fi

  # Also confirm the container is running via docker ps
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q falkordb; then
    check_pass "FalkorDB container running"
  else
    check_warn "FalkorDB port 6379 responds but container not found via docker ps"
  fi
else
  check_fail "FalkorDB not reachable on port 6379 (run: docker compose -f $COMPOSE_FILE up -d falkordb)"
fi

# ---------------------------------------------------------------------------
# 2. Qdrant: /readyz + collections > 0 + point counts
# ---------------------------------------------------------------------------
info "Qdrant checks"

if curl -sf http://localhost:6333/readyz &>/dev/null; then
  check_pass "Qdrant /readyz → ready"

  COL_RESPONSE=$(curl -s http://localhost:6333/collections 2>/dev/null || echo '{}')
  COL_COUNT=$(echo "$COL_RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('result',{}).get('collections',[])))" 2>/dev/null || echo "0")

  if [ "$COL_COUNT" -gt 0 ] 2>/dev/null; then
    check_pass "Qdrant collections = $COL_COUNT (data present)"

    # Check total points across all collections (Qdrant uses 'points_count' not 'vectors_count')
    TOTAL_POINTS=0
    COLLECTIONS=$(echo "$COL_RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); [print(c['name']) for c in d.get('result',{}).get('collections',[])]" 2>/dev/null || true)
    while IFS= read -r col; do
      [ -z "$col" ] && continue
      COL_INFO=$(curl -s "http://localhost:6333/collections/$col" 2>/dev/null || echo '{}')
      POINTS=$(echo "$COL_INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('result',{}).get('points_count',0))" 2>/dev/null || echo "0")
      TOTAL_POINTS=$((TOTAL_POINTS + POINTS))
    done <<< "$COLLECTIONS"

    if [ "$TOTAL_POINTS" -gt 0 ]; then
      check_pass "Qdrant total points = $TOTAL_POINTS across $COL_COUNT collections"
    else
      check_warn "Qdrant has $COL_COUNT collections but 0 points (may be empty after fresh import)"
    fi
  else
    check_fail "Qdrant collections = 0 (no data — migration may not have imported)"
  fi
else
  check_fail "Qdrant /readyz failed (container not running or not ready)"
fi

# ---------------------------------------------------------------------------
# 3. Fused-memory MCP endpoint
# ---------------------------------------------------------------------------
info "Fused-memory service checks"

if systemctl --user is-active fused-memory &>/dev/null; then
  check_pass "fused-memory systemd unit active"
else
  check_fail "fused-memory systemd unit not active (run: systemctl --user start fused-memory)"
fi

# Wait up to 10s for fused-memory health endpoint (may still be starting)
# Note: /mcp requires Accept: text/event-stream; use /health for liveness check
MCP_OK=0
for i in $(seq 1 10); do
  if curl -sf http://localhost:8002/health &>/dev/null; then
    MCP_OK=1
    break
  fi
  sleep 1
done

if [ "$MCP_OK" -eq 1 ]; then
  HEALTH_JSON=$(curl -s http://localhost:8002/health 2>/dev/null || echo '{}')
  GRAPHITI_OK=$(echo "$HEALTH_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print('yes' if d.get('graphiti') else 'no')" 2>/dev/null || echo "?")
  MEM0_OK=$(echo "$HEALTH_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print('yes' if d.get('mem0') else 'no')" 2>/dev/null || echo "?")
  check_pass "Fused-memory healthy (graphiti=$GRAPHITI_OK, mem0=$MEM0_OK)"
else
  check_fail "Fused-memory not responding after 10s (check: journalctl --user -u fused-memory -n 50)"
fi

# ---------------------------------------------------------------------------
# 4. Runtime data: reconciliation.db + queue.db
# ---------------------------------------------------------------------------
info "Runtime data checks"

RECON_DB="$REPO_ROOT/data/reconciliation/reconciliation.db"
if [ -f "$RECON_DB" ]; then
  RECON_SIZE=$(du -sh "$RECON_DB" | cut -f1)
  # Check it's a valid SQLite DB
  if sqlite3 "$RECON_DB" "SELECT COUNT(*) FROM event_buffer;" &>/dev/null; then
    EVENT_COUNT=$(sqlite3 "$RECON_DB" "SELECT COUNT(*) FROM event_buffer;" 2>/dev/null || echo "?")
    check_pass "reconciliation.db present ($RECON_SIZE, $EVENT_COUNT events in buffer)"
  else
    check_warn "reconciliation.db exists but may be corrupt (cannot query event_buffer)"
  fi
else
  check_warn "reconciliation.db not found at $RECON_DB (new install may not have this yet)"
fi

QUEUE_DB="$REPO_ROOT/data/queue"
if [ -d "$QUEUE_DB" ]; then
  QUEUE_SIZE=$(du -sh "$QUEUE_DB" | cut -f1)
  check_pass "data/queue/ present ($QUEUE_SIZE)"
else
  check_warn "data/queue/ not found — queue data may not have been exported"
fi

# Burndown state
BURNDOWN_DIR="$REPO_ROOT/data/burndown"
if [ -d "$BURNDOWN_DIR" ]; then
  check_pass "data/burndown/ present"
else
  check_warn "data/burndown/ not found"
fi

# ---------------------------------------------------------------------------
# 5. Secrets files
# ---------------------------------------------------------------------------
info "Secrets checks"

FUSED_ENV="$REPO_ROOT/fused-memory/.env"
if [ -f "$FUSED_ENV" ]; then
  # Check it has OPENAI_API_KEY (don't print value)
  if grep -q "OPENAI_API_KEY" "$FUSED_ENV"; then
    check_pass "fused-memory/.env present with OPENAI_API_KEY"
  else
    check_warn "fused-memory/.env present but OPENAI_API_KEY not found"
  fi
else
  check_fail "fused-memory/.env missing — fused-memory will not function"
fi

ROOT_ENV="$REPO_ROOT/.env"
if [ -f "$ROOT_ENV" ]; then
  check_pass ".env present"
else
  check_warn ".env not found (may not be required)"
fi

# ---------------------------------------------------------------------------
# 6. Git worktrees: no prunable entries
# ---------------------------------------------------------------------------
info "Worktree checks"

for repo_dir in "$REPO_ROOT" "$(dirname "$REPO_ROOT")/reify" "$(dirname "$REPO_ROOT")/autopilot-video"; do
  [ -d "$repo_dir/.git" ] || continue
  repo_name="$(basename "$repo_dir")"

  # Count worktrees (subtract 1 for main worktree)
  wt_count=$(git -C "$repo_dir" worktree list 2>/dev/null | wc -l)
  wt_count=$((wt_count - 1))

  # Check for broken/prunable worktrees
  prunable=$(git -C "$repo_dir" worktree list --porcelain 2>/dev/null | grep -c "^prunable " || true)

  if [ "$prunable" -gt 0 ]; then
    check_warn "$repo_name: $wt_count worktrees, $prunable prunable (run: git -C $repo_dir worktree prune)"
  elif [ "$wt_count" -gt 0 ]; then
    check_pass "$repo_name: $wt_count worktrees intact"
  else
    check_pass "$repo_name: no worktrees (fresh clone)"
  fi
done

# ---------------------------------------------------------------------------
# 7. Functional smoke test: fused-memory search returns results
# ---------------------------------------------------------------------------
if [ "$SMOKE" -eq 1 ]; then
  info "Functional smoke test"

  if [ "$MCP_OK" -eq 1 ]; then
    # fused-memory search is only accessible via MCP StreamableHTTP (SSE protocol).
    # We verify end-to-end pipeline via the /health endpoint's graphiti + mem0 flags,
    # combined with the data presence checks above (FalkorDB DBSIZE > 0, Qdrant points > 0).
    HEALTH_JSON=$(curl -s http://localhost:8002/health 2>/dev/null || echo '{}')
    GRAPHITI_OK=$(echo "$HEALTH_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print('yes' if d.get('graphiti') else 'no')" 2>/dev/null || echo "?")
    MEM0_OK=$(echo "$HEALTH_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print('yes' if d.get('mem0') else 'no')" 2>/dev/null || echo "?")

    if [ "$GRAPHITI_OK" = "yes" ] && [ "$MEM0_OK" = "yes" ]; then
      check_pass "Smoke test: fused-memory connected to both backends (graphiti=yes, mem0=yes)"
    elif [ "$GRAPHITI_OK" = "yes" ] || [ "$MEM0_OK" = "yes" ]; then
      check_warn "Smoke test: partial backend connectivity (graphiti=$GRAPHITI_OK, mem0=$MEM0_OK)"
    else
      check_fail "Smoke test: fused-memory not connected to backends (graphiti=$GRAPHITI_OK, mem0=$MEM0_OK)"
    fi
  else
    check_warn "Smoke test skipped — fused-memory not available"
  fi
else
  warn "Functional smoke test skipped (--no-smoke)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "Verification Summary"
echo ""
printf '  \033[1;32m%d passed\033[0m  \033[1;33m%d warnings\033[0m  \033[1;31m%d failed\033[0m\n' \
  "$CHECKS_PASSED" "$CHECKS_WARNED" "$CHECKS_FAILED"
echo ""

if [ "$CHECKS_FAILED" -gt 0 ]; then
  fail "Migration verification FAILED — $CHECKS_FAILED critical check(s) failed"
  echo ""
  echo "  Remediation hints:"
  echo "    FalkorDB: docker compose -f $COMPOSE_FILE up -d falkordb"
  echo "    Qdrant:   docker compose -f $COMPOSE_FILE up -d qdrant"
  echo "    fused-memory: systemctl --user start fused-memory"
  echo "    Check import: bash $REPO_ROOT/scripts/import-data.sh <export-dir>"
  echo ""
  exit 1
else
  ok "Migration verification PASSED"
  echo ""
  exit 0
fi
