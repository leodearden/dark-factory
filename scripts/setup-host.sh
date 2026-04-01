#!/usr/bin/env bash
# setup-host.sh — idempotent bootstrap for a dark-factory development host.
# Assumes the repo is already cloned (with --recurse-submodules).
# Run from anywhere: bash /path/to/dark-factory/scripts/setup-host.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/fused-memory/docker/docker-compose.yml"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m  ✗ %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
info "Checking prerequisites"

# Docker
if command -v docker &>/dev/null && docker compose version &>/dev/null; then
  ok "Docker + Compose v2"
else
  warn "Docker not found — installing via get.docker.com"
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  warn "Added $USER to docker group — you may need to log out and back in"
fi

# uv
if command -v uv &>/dev/null; then
  ok "uv ($(uv --version))"
else
  warn "uv not found — installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  ok "uv installed ($(uv --version))"
fi

# Node 22+
if command -v node &>/dev/null; then
  NODE_MAJOR=$(node --version | sed 's/v\([0-9]*\).*/\1/')
  if [ "$NODE_MAJOR" -ge 22 ]; then
    ok "Node $(node --version)"
  else
    warn "Node $(node --version) found but >= 22 required"
    warn "Install Node 22 via your preferred method (nvm, fnm, nodesource)"
  fi
else
  warn "Node not found — install Node 22 via nvm, fnm, or nodesource"
fi

# Claude Code
if command -v claude &>/dev/null; then
  ok "Claude Code ($(claude --version 2>/dev/null || echo 'installed'))"
else
  warn "Claude Code not found — install with: npm install -g @anthropic-ai/claude-code"
fi

# System packages
for pkg in curl jq bubblewrap; do
  if command -v "$pkg" &>/dev/null; then
    ok "$pkg"
  else
    warn "$pkg not found — installing"
    sudo apt-get update -qq && sudo apt-get install -y -qq "$pkg"
  fi
done

# ---------------------------------------------------------------------------
# 2. Docker Compose — start backing stores
# ---------------------------------------------------------------------------
info "Starting backing stores (FalkorDB + Qdrant)"

mkdir -p "$REPO_ROOT/fused-memory/data/falkordb"
mkdir -p "$REPO_ROOT/fused-memory/data/qdrant"

docker compose -f "$COMPOSE_FILE" up -d falkordb qdrant

# Wait for healthy
for i in $(seq 1 30); do
  if docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
    ok "FalkorDB healthy"
    break
  fi
  [ "$i" -eq 30 ] && fail "FalkorDB did not become healthy in 30s"
  sleep 1
done

for i in $(seq 1 30); do
  if curl -sf http://localhost:6333/readyz &>/dev/null; then
    ok "Qdrant healthy"
    break
  fi
  [ "$i" -eq 30 ] && fail "Qdrant did not become healthy in 30s"
  sleep 1
done

# ---------------------------------------------------------------------------
# 3. Python subprojects — uv sync (dependency order)
# ---------------------------------------------------------------------------
info "Syncing Python subprojects"

for proj in shared escalation fused-memory orchestrator dashboard; do
  (cd "$REPO_ROOT/$proj" && uv sync --quiet)
  ok "$proj"
done

# ---------------------------------------------------------------------------
# 4. Taskmaster AI — npm install + build
# ---------------------------------------------------------------------------
info "Building taskmaster-ai"

if [ -d "$REPO_ROOT/taskmaster-ai" ]; then
  (cd "$REPO_ROOT/taskmaster-ai" && npm install --silent && npm run build --silent)
  if [ -f "$REPO_ROOT/taskmaster-ai/dist/mcp-server.js" ]; then
    ok "taskmaster-ai built (dist/mcp-server.js)"
  else
    fail "taskmaster-ai build did not produce dist/mcp-server.js"
  fi
else
  warn "taskmaster-ai/ not found — did you clone with --recurse-submodules?"
fi

# ---------------------------------------------------------------------------
# 5. Systemd user unit for fused-memory
# ---------------------------------------------------------------------------
info "Installing fused-memory systemd unit"

UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

UV_PATH="$(command -v uv)"
sed \
  -e "s|__REPO_ROOT__|$REPO_ROOT|g" \
  -e "s|__UV_PATH__|$UV_PATH|g" \
  "$REPO_ROOT/scripts/fused-memory.service.template" \
  > "$UNIT_DIR/fused-memory.service"

systemctl --user daemon-reload
systemctl --user enable fused-memory

# Only start if .env exists (needs secrets)
if [ -f "$REPO_ROOT/fused-memory/.env" ]; then
  systemctl --user restart fused-memory
  ok "fused-memory unit installed and started"
else
  warn "fused-memory unit installed but NOT started (fused-memory/.env missing)"
fi

# ---------------------------------------------------------------------------
# 6. Skim — context compression for coding agents
# ---------------------------------------------------------------------------
info "Installing skim (context compression)"

if command -v skim &>/dev/null; then
  ok "skim already installed ($(skim --version 2>/dev/null))"
else
  if command -v cargo &>/dev/null; then
    cargo install rskim --quiet
    ok "skim installed via cargo"
  else
    warn "cargo not found — install Rust (rustup.rs) then: cargo install rskim"
  fi
fi

# Install global Claude Code hook (idempotent — skim init checks existing state)
if command -v skim &>/dev/null && command -v claude &>/dev/null; then
  if [ -f "$HOME/.claude/hooks/skim-rewrite.sh" ]; then
    ok "skim hook already installed"
  else
    skim init --yes
    ok "skim hook installed for Claude Code"
  fi
fi

# ---------------------------------------------------------------------------
# 7. Claude Code skill symlinks
# ---------------------------------------------------------------------------
info "Creating Claude Code skill symlinks"

COMMANDS_DIR="$HOME/.claude/commands"
mkdir -p "$COMMANDS_DIR"

declare -A SKILLS=(
  ["orchestrate.md"]="$REPO_ROOT/skills/orchestrate/SKILL.md"
  ["orchestrate-references"]="$REPO_ROOT/skills/orchestrate/references"
  ["reflect.md"]="$REPO_ROOT/skills/reflect/SKILL.md"
  ["semantic-merge.md"]="$REPO_ROOT/skills/semantic-merge/SKILL.md"
  ["unblock.md"]="$REPO_ROOT/skills/unblock/SKILL.md"
)

for name in "${!SKILLS[@]}"; do
  target="${SKILLS[$name]}"
  link="$COMMANDS_DIR/$name"
  if [ -e "$target" ] || [ -d "$target" ]; then
    ln -sfn "$target" "$link"
    ok "$name -> $(basename "$target")"
  else
    warn "Skipping $name — target does not exist: $target"
  fi
done

# ---------------------------------------------------------------------------
# 8. Git hooks
# ---------------------------------------------------------------------------
info "Setting up git hooks"

if [ -x "$REPO_ROOT/hooks/setup.sh" ]; then
  (cd "$REPO_ROOT" && bash hooks/setup.sh)
  ok "Git hooks configured"
else
  warn "hooks/setup.sh not found or not executable"
fi

# ---------------------------------------------------------------------------
# 9. Manual steps reminder
# ---------------------------------------------------------------------------
info "Manual steps (if migrating from another host)"
echo ""
echo "  1. Copy secrets:"
echo "     scp laptop:~/src/dark-factory/.env $REPO_ROOT/.env"
echo "     scp laptop:~/src/dark-factory/fused-memory/.env $REPO_ROOT/fused-memory/.env"
echo ""
echo "  2. Copy Claude credentials:"
echo "     scp laptop:~/.claude/.credentials.json ~/.claude/.credentials.json"
echo ""
echo "  3. Copy Claude settings + project memory:"
echo "     rsync -av --exclude='projects/*worktrees*' laptop:~/.claude/ ~/.claude/"
echo ""
echo "  4. Import data (after running export-data.sh on the source host):"
echo "     bash $REPO_ROOT/scripts/import-data.sh <export-dir>"
echo ""

# ---------------------------------------------------------------------------
# 10. Health checks
# ---------------------------------------------------------------------------
info "Health checks"

# FalkorDB
if docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
  ok "FalkorDB: PONG"
else
  fail "FalkorDB: not responding"
fi

# Qdrant
if curl -sf http://localhost:6333/readyz &>/dev/null; then
  col_count=$(curl -s http://localhost:6333/collections | jq '.result.collections | length' 2>/dev/null || echo "?")
  ok "Qdrant: ready ($col_count collections)"
else
  fail "Qdrant: not responding"
fi

# Fused-memory
if curl -sf http://localhost:8002/mcp 2>/dev/null; then
  ok "Fused-memory: healthy"
elif systemctl --user is-active fused-memory &>/dev/null; then
  warn "Fused-memory: unit active but health check failed (may still be starting)"
else
  warn "Fused-memory: not running (check: journalctl --user -u fused-memory)"
fi

echo ""
info "Setup complete"
