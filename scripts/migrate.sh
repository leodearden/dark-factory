#!/usr/bin/env bash
# migrate.sh — full host migration orchestrator.
# Run on the SOURCE host (Framework 16 laptop) to migrate to a destination.
# Ties together export → rsync → setup → import → verify.
#
# Usage: bash scripts/migrate.sh [OPTIONS]
#   -t TARGET     Destination host (default: leo-workstation)
#   -e EXPORT_DIR Use an existing export directory (skip export phase)
#   -p PHASE      Start from a specific phase (1-5) to resume after failure
#   -n            Dry-run: print rsync commands without executing
#   -h            Show this help
#
# Phases:
#   1  Export data locally (export-data.sh)
#   2  rsync export + repos + Claude credentials to target
#   3  SSH to target: run setup-host.sh
#   4  SSH to target: run import-data.sh
#   5  SSH to target: run verify-migration.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m  + %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m  x %s\033[0m\n' "$*"; exit 1; }
step()  { printf '\n\033[1;36m[Phase %s] %s\033[0m\n' "$1" "$2"; }

TARGET="leo-workstation"
EXPORT_DIR=""
START_PHASE=1
DRY_RUN=0

usage() {
  sed -n '2,/^set/p' "$0" | grep '^#' | sed 's/^# //' | sed 's/^#//'
  exit 0
}

while getopts "t:e:p:nh" opt; do
  case $opt in
    t) TARGET="$OPTARG" ;;
    e) EXPORT_DIR="$OPTARG" ;;
    p) START_PHASE="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    h) usage ;;
    *) fail "Unknown option: -$OPTARG (run with -h for help)" ;;
  esac
done

RSYNC_OPTS="-avP"
if [ "$DRY_RUN" -eq 1 ]; then
  RSYNC_OPTS="-avPn"
  warn "DRY RUN — rsync commands will be printed but not executed"
fi

# ---------------------------------------------------------------------------
# Pre-flight: verify SSH connectivity to target
# ---------------------------------------------------------------------------
info "Pre-flight: SSH connectivity check"

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$TARGET" echo "SSH OK" &>/dev/null; then
  fail "Cannot SSH to $TARGET — check Tailscale connectivity and SSH keys"
fi
ok "SSH to $TARGET is reachable"

# ---------------------------------------------------------------------------
# Phase 1: Export data locally
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 1 ]; then
  step "1" "Export data (source host)"

  if [ -n "$EXPORT_DIR" ]; then
    warn "Using existing export directory: $EXPORT_DIR"
  else
    if [ "$DRY_RUN" -eq 1 ]; then
      warn "DRY RUN: would run: bash $REPO_ROOT/scripts/export-data.sh"
      EXPORT_DIR="$REPO_ROOT/export-DRY-RUN"
    else
      bash "$REPO_ROOT/scripts/export-data.sh"
      # Find the most recently created export directory
      EXPORT_DIR=$(ls -dt "$REPO_ROOT"/export-*/  2>/dev/null | head -1)
      EXPORT_DIR="${EXPORT_DIR%/}"
      if [ -z "$EXPORT_DIR" ] || [ ! -d "$EXPORT_DIR" ]; then
        fail "export-data.sh ran but no export-* directory found in $REPO_ROOT"
      fi
    fi
    ok "Export directory: $EXPORT_DIR"
  fi
fi

# Ensure EXPORT_DIR is set for subsequent phases
if [ -z "$EXPORT_DIR" ] && [ "$START_PHASE" -gt 1 ]; then
  fail "No export directory specified. Use -e <export-dir> when skipping phase 1."
fi

# ---------------------------------------------------------------------------
# Phase 2: rsync data and repos to target
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 2 ]; then
  step "2" "rsync to $TARGET"

  # 2a. Export data → target home
  info "Rsyncing export data to $TARGET:~/dark-factory-export/"
  if [ "$DRY_RUN" -eq 0 ]; then
    rsync $RSYNC_OPTS "$EXPORT_DIR/" "$TARGET:~/dark-factory-export/"
    ok "Export data rsynced"
  else
    echo "  rsync $RSYNC_OPTS $EXPORT_DIR/ $TARGET:~/dark-factory-export/"
  fi

  # 2b. Main repo → target (exclude build artifacts, Python envs)
  info "Rsyncing dark-factory repo to $TARGET:~/src/dark-factory/"
  EXCLUDE_ARGS=(
    --exclude='target/'
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='*.egg-info/'
    --exclude='.task/'
    --exclude='export-*/'
  )
  if [ "$DRY_RUN" -eq 0 ]; then
    rsync $RSYNC_OPTS "${EXCLUDE_ARGS[@]}" "$REPO_ROOT/" "$TARGET:~/src/dark-factory/"
    ok "dark-factory repo rsynced"
  else
    echo "  rsync $RSYNC_OPTS ${EXCLUDE_ARGS[*]} $REPO_ROOT/ $TARGET:~/src/dark-factory/"
  fi

  # 2c. Sibling repos (if present)
  for sibling in "$(dirname "$REPO_ROOT")/reify" "$(dirname "$REPO_ROOT")/autopilot-video"; do
    [ -d "$sibling/.git" ] || continue
    repo_name="$(basename "$sibling")"
    info "Rsyncing $repo_name to $TARGET:~/src/$repo_name/"
    if [ "$DRY_RUN" -eq 0 ]; then
      rsync $RSYNC_OPTS "${EXCLUDE_ARGS[@]}" "$sibling/" "$TARGET:~/src/$repo_name/"
      ok "$repo_name rsynced"
    else
      echo "  rsync $RSYNC_OPTS ${EXCLUDE_ARGS[*]} $sibling/ $TARGET:~/src/$repo_name/"
    fi
  done

  # 2d. Claude credentials (exclude project worktree state which is host-specific)
  CLAUDE_DIR="$HOME/.claude"
  if [ -d "$CLAUDE_DIR" ]; then
    info "Rsyncing Claude credentials to $TARGET:~/.claude/"
    CLAUDE_EXCLUDE=(
      --exclude='projects/*worktrees*'
      --exclude='projects/*/todos/'
    )
    if [ "$DRY_RUN" -eq 0 ]; then
      rsync $RSYNC_OPTS "${CLAUDE_EXCLUDE[@]}" "$CLAUDE_DIR/" "$TARGET:~/.claude/"
      ok "Claude credentials rsynced"
    else
      echo "  rsync $RSYNC_OPTS ${CLAUDE_EXCLUDE[*]} $CLAUDE_DIR/ $TARGET:~/.claude/"
    fi
  else
    warn "~/.claude not found — skipping Claude credentials"
  fi
fi

# ---------------------------------------------------------------------------
# Phase 3: Setup target host
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 3 ]; then
  step "3" "setup-host.sh on $TARGET"

  info "Running setup-host.sh on $TARGET"
  if [ "$DRY_RUN" -eq 0 ]; then
    ssh "$TARGET" "bash ~/src/dark-factory/scripts/setup-host.sh" || {
      warn "setup-host.sh exited non-zero on $TARGET"
      warn "Check if this is recoverable. To resume from phase 4:"
      warn "  bash $0 -t $TARGET -e $EXPORT_DIR -p 4"
      exit 1
    }
    ok "setup-host.sh completed on $TARGET"
  else
    echo "  ssh $TARGET bash ~/src/dark-factory/scripts/setup-host.sh"
  fi
fi

# ---------------------------------------------------------------------------
# Phase 4: Import data on target
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 4 ]; then
  step "4" "import-data.sh on $TARGET"

  info "Running import-data.sh on $TARGET"
  if [ "$DRY_RUN" -eq 0 ]; then
    ssh "$TARGET" "bash ~/src/dark-factory/scripts/import-data.sh ~/dark-factory-export" || {
      warn "import-data.sh exited non-zero on $TARGET"
      warn "Check import logs. To re-run import:"
      warn "  bash $0 -t $TARGET -e $EXPORT_DIR -p 4"
      exit 1
    }
    ok "import-data.sh completed on $TARGET"
  else
    echo "  ssh $TARGET bash ~/src/dark-factory/scripts/import-data.sh ~/dark-factory-export"
  fi
fi

# ---------------------------------------------------------------------------
# Phase 5: Verify migration on target
# ---------------------------------------------------------------------------
if [ "$START_PHASE" -le 5 ]; then
  step "5" "verify-migration.sh on $TARGET"

  info "Running verify-migration.sh on $TARGET"
  if [ "$DRY_RUN" -eq 0 ]; then
    if ssh "$TARGET" "bash ~/src/dark-factory/scripts/verify-migration.sh"; then
      echo ""
      ok "Migration to $TARGET COMPLETE and VERIFIED"
      echo ""
      echo "  Next steps:"
      echo "    1. Update DNS/Tailscale routing if needed"
      echo "    2. Verify Claude Code sessions on $TARGET"
      echo "    3. Update any hardcoded host references in config"
      echo "    4. Consider decommissioning source host data (keep for 7 days)"
    else
      echo ""
      fail "Migration to $TARGET FAILED verification — see output above"
    fi
  else
    echo "  ssh $TARGET bash ~/src/dark-factory/scripts/verify-migration.sh"
    echo ""
    ok "DRY RUN complete — review commands above before executing"
  fi
fi
