#!/usr/bin/env bash
# export-data.sh — export all fused-memory persistent data for migration.
# Run on the SOURCE host before transferring to the destination.
# Usage: bash scripts/export-data.sh [export-dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/fused-memory/docker/docker-compose.yml"
EXPORT_DIR="${1:-$REPO_ROOT/export-$(date +%Y-%m-%d-%H%M%S)}"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m  + %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m  x %s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------------------
info "Pre-flight checks"

# Check for active reconciliation locks
RECON_DB="$REPO_ROOT/data/reconciliation/reconciliation.db"
if [ -f "$RECON_DB" ]; then
  LOCK_COUNT=$(sqlite3 "$RECON_DB" "SELECT COUNT(*) FROM reconciliation_locks;" 2>/dev/null || echo "0")
  if [ "$LOCK_COUNT" -gt 0 ]; then
    warn "Active reconciliation lock detected ($LOCK_COUNT locks)"
    warn "Events will be recovered automatically on import via _recover_stale_runs()"
  fi

  BUFFERED=$(sqlite3 "$RECON_DB" "SELECT COUNT(*) FROM event_buffer WHERE status='buffered';" 2>/dev/null || echo "?")
  ok "Event buffer: $BUFFERED buffered events (will be preserved)"
else
  warn "No reconciliation database found at $RECON_DB"
fi

# ---------------------------------------------------------------------------
# 2. Stop fused-memory (clean WAL checkpoint)
# ---------------------------------------------------------------------------
info "Stopping fused-memory service"

if systemctl --user is-active fused-memory &>/dev/null; then
  systemctl --user stop fused-memory
  ok "fused-memory stopped"
else
  warn "fused-memory was not running"
fi

# ---------------------------------------------------------------------------
# 3. FalkorDB BGSAVE
# ---------------------------------------------------------------------------
info "Flushing FalkorDB to disk"

if docker compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -q falkordb; then
  BEFORE=$(docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli LASTSAVE 2>/dev/null || echo "0")
  docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli BGSAVE &>/dev/null

  for i in $(seq 1 30); do
    AFTER=$(docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli LASTSAVE 2>/dev/null || echo "0")
    if [ "$AFTER" != "$BEFORE" ]; then
      ok "FalkorDB BGSAVE completed"
      break
    fi
    [ "$i" -eq 30 ] && warn "FalkorDB BGSAVE did not complete in 30s (using existing dump.rdb)"
    sleep 1
  done
else
  warn "FalkorDB container not running — using existing dump.rdb"
fi

# ---------------------------------------------------------------------------
# 4. Create export directory and copy data
# ---------------------------------------------------------------------------
info "Exporting data to $EXPORT_DIR"

mkdir -p "$EXPORT_DIR"

# FalkorDB
FALKOR_SRC="$REPO_ROOT/fused-memory/data/falkordb"
if [ -d "$FALKOR_SRC" ]; then
  mkdir -p "$EXPORT_DIR/fused-memory-data/falkordb"
  cp -a "$FALKOR_SRC/." "$EXPORT_DIR/fused-memory-data/falkordb/"
  ok "FalkorDB: $(du -sh "$FALKOR_SRC" | cut -f1)"
fi

# Qdrant
QDRANT_SRC="$REPO_ROOT/fused-memory/data/qdrant"
if [ -d "$QDRANT_SRC" ]; then
  mkdir -p "$EXPORT_DIR/fused-memory-data/qdrant"
  cp -a "$QDRANT_SRC/." "$EXPORT_DIR/fused-memory-data/qdrant/"
  ok "Qdrant: $(du -sh "$QDRANT_SRC" | cut -f1)"
fi

# Runtime data directories (SQLite WAL triplets + JSON files)
for subdir in reconciliation queue burndown escalations; do
  SRC="$REPO_ROOT/data/$subdir"
  if [ -d "$SRC" ]; then
    mkdir -p "$EXPORT_DIR/data/$subdir"
    cp -a "$SRC/." "$EXPORT_DIR/data/$subdir/"
    ok "data/$subdir: $(du -sh "$SRC" | cut -f1)"
  fi
done

# Standalone files in data/
for f in "$REPO_ROOT"/data/*.json; do
  [ -f "$f" ] || continue
  mkdir -p "$EXPORT_DIR/data"
  cp -a "$f" "$EXPORT_DIR/data/"
  ok "data/$(basename "$f")"
done

# Stale reconciliation.db at fused-memory/data/ (if non-empty)
STALE_RECON="$REPO_ROOT/fused-memory/data/reconciliation.db"
if [ -f "$STALE_RECON" ] && [ -s "$STALE_RECON" ]; then
  cp -a "$STALE_RECON" "$EXPORT_DIR/fused-memory-data/"
  ok "fused-memory/data/reconciliation.db (legacy)"
fi

# ---------------------------------------------------------------------------
# 5. Secrets
# ---------------------------------------------------------------------------
info "Exporting secrets"

mkdir -p "$EXPORT_DIR/secrets"
for envfile in "$REPO_ROOT/.env" "$REPO_ROOT/fused-memory/.env"; do
  if [ -f "$envfile" ]; then
    REL="${envfile#$REPO_ROOT/}"
    cp -a "$envfile" "$EXPORT_DIR/secrets/$(echo "$REL" | tr '/' '_')"
    ok "$REL"
  fi
done

# ---------------------------------------------------------------------------
# 6. Push all local branches to remote (safety net for worktree WIP)
# ---------------------------------------------------------------------------
info "Pushing local branches to remote"

REPOS=(
  "$REPO_ROOT"
)
# Add sibling projects if they exist
for sibling in "$(dirname "$REPO_ROOT")/reify" "$(dirname "$REPO_ROOT")/autopilot-video"; do
  [ -d "$sibling/.git" ] && REPOS+=("$sibling")
done

for repo in "${REPOS[@]}"; do
  repo_name="$(basename "$repo")"
  LOCAL_ONLY=$(git -C "$repo" for-each-ref --format='%(refname:short)' refs/heads/ | while read -r branch; do
    git -C "$repo" rev-parse "refs/remotes/origin/$branch" &>/dev/null || echo "$branch"
  done | wc -l)

  if [ "$LOCAL_ONLY" -gt 0 ]; then
    git -C "$repo" push origin --all 2>/dev/null && \
      ok "$repo_name: pushed $LOCAL_ONLY local-only branches" || \
      warn "$repo_name: push failed — branches are still local-only"
  else
    ok "$repo_name: all branches already on remote"
  fi
done

# ---------------------------------------------------------------------------
# 7. Restart services on source host
# ---------------------------------------------------------------------------
info "Restarting fused-memory on source host"

if [ -f "$REPO_ROOT/fused-memory/.env" ]; then
  systemctl --user start fused-memory
  ok "fused-memory restarted"
else
  warn "fused-memory/.env missing — not restarting"
fi

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
echo ""
info "Export complete"
echo ""
du -sh "$EXPORT_DIR"
echo ""
echo "  Contents:"
du -sh "$EXPORT_DIR"/*/ 2>/dev/null | sed 's/^/    /'
echo ""

if [ -n "${BUFFERED:-}" ] && [ "$BUFFERED" != "?" ]; then
  echo "  Reconciliation: $BUFFERED buffered events (will resume on import)"
fi

echo ""
info "Migration steps"
echo ""
echo "  Step 1: Transfer fused-memory data"
echo "    rsync -avP $EXPORT_DIR/ leo-workstation:~/dark-factory-export/"
echo ""
echo "  Step 2: Transfer repos (with worktrees, excluding build artifacts)"
echo "    rsync -avP --exclude='target/' --exclude='.venv/' --exclude='__pycache__/' \\"
echo "      ~/src/dark-factory/ leo-workstation:~/src/dark-factory/"
for repo in "${REPOS[@]}"; do
  [ "$repo" = "$REPO_ROOT" ] && continue
  repo_name="$(basename "$repo")"
  echo "    rsync -avP --exclude='target/' --exclude='.venv/' --exclude='__pycache__/' \\"
  echo "      ~/src/$repo_name/ leo-workstation:~/src/$repo_name/"
done
echo ""
echo "  Step 3: On leo-workstation, run setup + import"
echo "    bash ~/src/dark-factory/scripts/setup-host.sh"
echo "    bash ~/src/dark-factory/scripts/import-data.sh ~/dark-factory-export"
echo ""
echo "  Step 4: Copy Claude credentials + project memory"
echo "    rsync -avP --exclude='projects/*worktrees*' ~/.claude/ leo-workstation:~/.claude/"
echo ""
