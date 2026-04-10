#!/usr/bin/env bash
# import-data.sh — import fused-memory data from an export directory.
# Run on the DESTINATION host after setup-host.sh has been run.
# Usage: bash scripts/import-data.sh <export-dir>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/fused-memory/docker/docker-compose.yml"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m  + %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m  x %s\033[0m\n' "$*"; exit 1; }

EXPORT_DIR="${1:-}"
if [ -z "$EXPORT_DIR" ] || [ ! -d "$EXPORT_DIR" ]; then
  echo "Usage: bash scripts/import-data.sh <export-dir>"
  echo ""
  echo "  export-dir: directory created by export-data.sh (contains fused-memory-data/, data/, secrets/)"
  exit 1
fi

# Resolve to absolute path
EXPORT_DIR="$(cd "$EXPORT_DIR" && pwd)"

info "Importing from $EXPORT_DIR"
echo ""
du -sh "$EXPORT_DIR"/*/ 2>/dev/null | sed 's/^/    /'
echo ""

# ---------------------------------------------------------------------------
# 1. Stop services
# ---------------------------------------------------------------------------
info "Stopping services"

if systemctl --user is-active fused-memory &>/dev/null; then
  systemctl --user stop fused-memory
  ok "fused-memory stopped"
fi

# Stop Docker backing stores so we can replace their data
if docker compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -q falkordb; then
  docker compose -f "$COMPOSE_FILE" stop falkordb qdrant
  ok "FalkorDB + Qdrant containers stopped"
fi

# ---------------------------------------------------------------------------
# 2. Fix Docker-owned files (containers write as root inside volumes)
# ---------------------------------------------------------------------------
DATA_DIR="$REPO_ROOT/fused-memory/data"
if [ -d "$DATA_DIR" ]; then
  # Use a throwaway container to chown — avoids needing sudo
  docker run --rm -v "$DATA_DIR:/data" alpine chown -R "$(id -u):$(id -g)" /data 2>/dev/null && \
    ok "Fixed ownership on fused-memory/data/" || \
    warn "Could not fix ownership (may need: sudo chown -R $(id -u):$(id -g) $DATA_DIR)"
fi

# ---------------------------------------------------------------------------
# 3. Restore FalkorDB data
# ---------------------------------------------------------------------------
info "Restoring FalkorDB data"

FALKOR_EXPORT="$EXPORT_DIR/fused-memory-data/falkordb"
FALKOR_DEST="$REPO_ROOT/fused-memory/data/falkordb"
if [ -d "$FALKOR_EXPORT" ]; then
  mkdir -p "$FALKOR_DEST"
  cp -a "$FALKOR_EXPORT/." "$FALKOR_DEST/"
  ok "FalkorDB: $(du -sh "$FALKOR_DEST" | cut -f1)"
else
  warn "No FalkorDB data in export"
fi

# ---------------------------------------------------------------------------
# 4. Restore Qdrant data
# ---------------------------------------------------------------------------
info "Restoring Qdrant data"

QDRANT_EXPORT="$EXPORT_DIR/fused-memory-data/qdrant"
QDRANT_DEST="$REPO_ROOT/fused-memory/data/qdrant"
if [ -d "$QDRANT_EXPORT" ]; then
  mkdir -p "$QDRANT_DEST"
  cp -a "$QDRANT_EXPORT/." "$QDRANT_DEST/"
  ok "Qdrant: $(du -sh "$QDRANT_DEST" | cut -f1)"
else
  warn "No Qdrant data in export"
fi

# ---------------------------------------------------------------------------
# 5. Restore runtime data directories
# ---------------------------------------------------------------------------
info "Restoring runtime data"

for subdir in reconciliation queue burndown escalations; do
  SRC="$EXPORT_DIR/data/$subdir"
  DEST="$REPO_ROOT/data/$subdir"
  if [ -d "$SRC" ]; then
    mkdir -p "$DEST"
    cp -a "$SRC/." "$DEST/"
    ok "data/$subdir: $(du -sh "$DEST" | cut -f1)"
  fi
done

# Standalone files (e.g., reify-orphan-tasks.json)
for f in "$EXPORT_DIR"/data/*.json; do
  [ -f "$f" ] || continue
  mkdir -p "$REPO_ROOT/data"
  cp -a "$f" "$REPO_ROOT/data/"
  ok "data/$(basename "$f")"
done

# ---------------------------------------------------------------------------
# 6. Restore secrets
# ---------------------------------------------------------------------------
info "Restoring secrets"

SECRETS_DIR="$EXPORT_DIR/secrets"
if [ -d "$SECRETS_DIR" ]; then
  # .env -> repo root
  if [ -f "$SECRETS_DIR/.env" ]; then
    cp -a "$SECRETS_DIR/.env" "$REPO_ROOT/.env"
    ok ".env"
  fi
  # fused-memory_.env -> fused-memory/.env
  if [ -f "$SECRETS_DIR/fused-memory_.env" ]; then
    cp -a "$SECRETS_DIR/fused-memory_.env" "$REPO_ROOT/fused-memory/.env"
    ok "fused-memory/.env"
  fi
else
  warn "No secrets directory in export — you'll need to create .env files manually"
fi

# ---------------------------------------------------------------------------
# 7. Start services
# ---------------------------------------------------------------------------
info "Starting backing stores"

docker compose -f "$COMPOSE_FILE" up -d falkordb qdrant

# Wait for healthy
for i in $(seq 1 30); do
  if docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
    ok "FalkorDB healthy"
    break
  fi
  [ "$i" -eq 30 ] && warn "FalkorDB did not become healthy in 30s"
  sleep 1
done

for i in $(seq 1 30); do
  if curl -sf http://localhost:6333/readyz &>/dev/null; then
    ok "Qdrant healthy"
    break
  fi
  [ "$i" -eq 30 ] && warn "Qdrant did not become healthy in 30s"
  sleep 1
done

info "Starting fused-memory"

if [ -f "$REPO_ROOT/fused-memory/.env" ]; then
  systemctl --user restart fused-memory
  sleep 3

  if systemctl --user is-active fused-memory &>/dev/null; then
    ok "fused-memory running"
  else
    warn "fused-memory failed to start — check: journalctl --user -u fused-memory"
  fi
else
  warn "fused-memory/.env missing — not starting (create it first)"
fi

# ---------------------------------------------------------------------------
# 8. Health checks
# ---------------------------------------------------------------------------
info "Health checks"

# FalkorDB
if docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
  DBSIZE=$(docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli DBSIZE 2>/dev/null || echo "?")
  ok "FalkorDB: PONG ($DBSIZE)"
else
  warn "FalkorDB: not responding"
fi

# Qdrant
if curl -sf http://localhost:6333/readyz &>/dev/null; then
  COL_COUNT=$(curl -s http://localhost:6333/collections | jq '.result.collections | length' 2>/dev/null || echo "?")
  ok "Qdrant: ready ($COL_COUNT collections)"
else
  warn "Qdrant: not responding"
fi

# Fused-memory
if curl -sf http://localhost:8002/mcp &>/dev/null; then
  ok "Fused-memory: healthy (http://localhost:8002/mcp)"
else
  warn "Fused-memory: not responding (may still be starting)"
fi

# Event buffer status
RECON_DB="$REPO_ROOT/data/reconciliation/reconciliation.db"
if [ -f "$RECON_DB" ]; then
  BUFFERED=$(sqlite3 "$RECON_DB" "SELECT COUNT(*) FROM event_buffer WHERE status='buffered';" 2>/dev/null || echo "?")
  ok "Event buffer: $BUFFERED events pending reconciliation"
fi

# ---------------------------------------------------------------------------
# 9. Verify worktrees (if repos were rsync'd)
# ---------------------------------------------------------------------------
info "Verifying worktrees"

for repo_dir in "$REPO_ROOT" "$(dirname "$REPO_ROOT")/reify" "$(dirname "$REPO_ROOT")/autopilot-video"; do
  [ -d "$repo_dir/.git" ] || continue
  repo_name="$(basename "$repo_dir")"
  wt_count=$(git -C "$repo_dir" worktree list 2>/dev/null | wc -l)
  # Subtract 1 for the main worktree
  wt_count=$((wt_count - 1))

  if [ "$wt_count" -gt 0 ]; then
    # Check for broken worktrees
    broken=$(git -C "$repo_dir" worktree list --porcelain 2>/dev/null | grep -c "^prunable " || true)
    if [ "$broken" -gt 0 ]; then
      warn "$repo_name: $wt_count worktrees, $broken prunable (run: git -C $repo_dir worktree prune)"
    else
      ok "$repo_name: $wt_count worktrees intact"
    fi
  else
    ok "$repo_name: no worktrees (fresh clone — orchestrator will create as needed)"
  fi
done

echo ""
info "Import complete"
echo ""
