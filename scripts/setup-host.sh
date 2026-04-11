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
# 6. jCodeMunch — structured code retrieval for coding agents
# ---------------------------------------------------------------------------
info "Installing jCodeMunch (AST-based code indexing)"

# Global config (ignore patterns shared across all projects)
CODE_INDEX_DIR="$HOME/.code-index"
mkdir -p "$CODE_INDEX_DIR"

if [ ! -f "$CODE_INDEX_DIR/config.jsonc" ]; then
  cat > "$CODE_INDEX_DIR/config.jsonc" << 'JCEOF'
{
  "max_folder_files": 10000,
  "extra_ignore_patterns": [
    ".worktrees/",
    ".eval-worktrees/",
    ".claude/worktrees/",
    ".taskmaster/",
    ".playwright-mcp/",
    "node_modules/",
    "target/",
    "__pycache__/",
    ".venv/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    "*.png",
    "uv.lock",
    "Cargo.lock",
    "pnpm-lock.yaml",
    "package-lock.json"
  ],
  "staleness_days": 3
}
JCEOF
  ok "Global config written to $CODE_INDEX_DIR/config.jsonc"
else
  ok "Global config already exists"
fi

# Project-level config
if [ ! -f "$REPO_ROOT/.jcodemunch.jsonc" ]; then
  cat > "$REPO_ROOT/.jcodemunch.jsonc" << 'JCEOF'
{
  // dark-factory: Python monorepo (fused-memory, orchestrator, escalation, shared)
  "languages": ["python"],
  "max_folder_files": 5000,
  "disabled_tools": ["search_columns"],
  "staleness_days": 3
}
JCEOF
  ok "Project config written"
else
  ok "Project config already exists"
fi

# Add jcodemunch MCP to user-level Claude config (idempotent)
if command -v claude &>/dev/null; then
  if claude mcp list --scope user 2>/dev/null | grep -q jcodemunch; then
    ok "jcodemunch MCP already in user config"
  else
    claude mcp add --scope user jcodemunch -- uvx --python 3.12 jcodemunch-mcp
    ok "jcodemunch MCP added to user config"
  fi
fi

# Systemd watcher unit
sed \
  -e "s|__REPO_ROOT__|$REPO_ROOT|g" \
  -e "s|__UV_PATH__|$UV_PATH|g" \
  "$REPO_ROOT/scripts/jcodemunch-watcher.service.template" \
  > "$UNIT_DIR/jcodemunch-watcher.service"

systemctl --user daemon-reload
systemctl --user enable jcodemunch-watcher
systemctl --user restart jcodemunch-watcher
ok "jcodemunch-watcher unit installed and started"

# ---------------------------------------------------------------------------
# 7. Skim — context compression for coding agents
# ---------------------------------------------------------------------------
info "Installing skim (context compression)"

if command -v skim &>/dev/null; then
  ok "skim already installed ($(skim --version 2>/dev/null))"
else
  # Find cargo: may be on PATH, in ~/.cargo/bin, or only in a rustup toolchain
  CARGO=""
  if command -v cargo &>/dev/null; then
    CARGO="cargo"
  elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO="$HOME/.cargo/bin/cargo"
  else
    # Fall back to rustup stable toolchain
    RUSTUP_CARGO="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo"
    [ -x "$RUSTUP_CARGO" ] && CARGO="$RUSTUP_CARGO"
  fi

  if [ -n "$CARGO" ]; then
    $CARGO install rskim --quiet
    ok "skim installed via cargo ($CARGO)"
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

  # The hook rewrites commands to bare `skim`, which must be on PATH for all
  # shell types (login, interactive, non-interactive bash -c).  ~/.cargo/bin
  # is only added by profile/bashrc sourcing — symlink into /usr/local/bin
  # so it's on the base OS PATH unconditionally.
  SKIM_BIN="$HOME/.cargo/bin/skim"
  if [ ! -e /usr/local/bin/skim ]; then
    if [ -x "$SKIM_BIN" ]; then
      sudo ln -s "$SKIM_BIN" /usr/local/bin/skim
      ok "symlinked skim → /usr/local/bin/skim"
    fi
  else
    ok "skim already on system PATH (/usr/local/bin/skim)"
  fi
fi

# Verify skim is on PATH for all shell types an agent session might use.
# The hook rewrites commands to bare `skim`, so it must be findable without
# inheriting a profile-enhanced PATH.  Non-login, non-interactive shells
# (gnome-terminal -- bash -c '...', systemd ExecStart=, asyncio subprocesses)
# only get the base OS PATH unless ~/.cargo/env is sourced outside an
# interactivity guard.
if command -v skim &>/dev/null; then
  info "Checking skim PATH visibility across shell types"

  BASE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

  # Login shell (sources ~/.profile → ~/.cargo/env)
  if env -i HOME="$HOME" TERM="$TERM" bash --login -c 'command -v skim' &>/dev/null; then
    ok "skim on PATH: login shell"
  else
    fail "skim NOT on PATH: login shell"
  fi

  # Interactive shell (sources ~/.bashrc — needs to pass interactivity guard)
  # stderr suppressed: bash -ic warns about missing terminal/job-control
  if env -i HOME="$HOME" TERM="$TERM" bash -ic 'command -v skim' >/dev/null 2>&1; then
    ok "skim on PATH: interactive shell"
  else
    fail "skim NOT on PATH: interactive shell"
  fi

  # Non-interactive, non-login shell with base OS PATH only.
  # This simulates: gnome-terminal -- bash -c '...' when the parent env
  # was not profile-initialised, or asyncio.create_subprocess_exec with a
  # stripped env, or a systemd unit without Environment=PATH additions.
  if env -i HOME="$HOME" PATH="$BASE_PATH" bash -c 'command -v skim' &>/dev/null; then
    ok "skim on PATH: non-login non-interactive shell (base PATH)"
  else
    fail "skim NOT on PATH: non-login non-interactive shell (base PATH)"
    warn "  Agents spawned without profile init will fail on skim-rewritten commands"
    warn "  Fix: sudo ln -s $HOME/.cargo/bin/skim /usr/local/bin/skim"
  fi
fi

# ---------------------------------------------------------------------------
# 8. Dashboard systemd units
# ---------------------------------------------------------------------------
info "Installing dashboard systemd units"

sed \
  -e "s|__REPO_ROOT__|$REPO_ROOT|g" \
  -e "s|__UV_PATH__|$UV_PATH|g" \
  "$REPO_ROOT/scripts/dashboard.service.template" \
  > "$UNIT_DIR/dark-factory-dashboard.service"

# Watchdog service + timer (no templating needed — no repo-specific paths)
cp "$REPO_ROOT/dashboard/dark-factory-dashboard-watchdog.service" "$UNIT_DIR/"
cp "$REPO_ROOT/dashboard/dark-factory-dashboard-watchdog.timer" "$UNIT_DIR/"

systemctl --user daemon-reload
systemctl --user enable dark-factory-dashboard
systemctl --user enable dark-factory-dashboard-watchdog.timer
ok "Dashboard units installed (start manually when ready: systemctl --user start dark-factory-dashboard)"

# ---------------------------------------------------------------------------
# 9. Claude Code skill symlinks
# ---------------------------------------------------------------------------
info "Creating Claude Code skill symlinks"

COMMANDS_DIR="$HOME/.claude/commands"
mkdir -p "$COMMANDS_DIR"

declare -A SKILLS=(
  ["orchestrate.md"]="$REPO_ROOT/skills/orchestrate/SKILL.md"
  ["orchestrate-references"]="$REPO_ROOT/skills/orchestrate/references"
  ["reflect.md"]="$REPO_ROOT/skills/reflect/SKILL.md"
  ["unblock.md"]="$REPO_ROOT/skills/unblock/SKILL.md"
  ["review.md"]="$REPO_ROOT/skills/review/SKILL.md"
  ["review-references"]="$REPO_ROOT/skills/review/references"
  ["review-briefing.md"]="$REPO_ROOT/skills/review-briefing/SKILL.md"
  ["review-briefing-references"]="$REPO_ROOT/skills/review-briefing/references"
  ["escalation-watcher.md"]="$REPO_ROOT/skills/escalation-watcher/SKILL.md"
  ["merge-queue.md"]="$REPO_ROOT/skills/merge-queue/SKILL.md"
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
# 10. Git hooks
# ---------------------------------------------------------------------------
info "Setting up git hooks"

if [ -x "$REPO_ROOT/hooks/setup.sh" ]; then
  (cd "$REPO_ROOT" && bash hooks/setup.sh)
  ok "Git hooks configured"
else
  warn "hooks/setup.sh not found or not executable"
fi

# ---------------------------------------------------------------------------
# 11. Manual steps reminder
# ---------------------------------------------------------------------------
info "Manual steps (if migrating from another host)"
echo ""
echo "  On the SOURCE host, run:"
echo "    bash ~/src/dark-factory/scripts/export-data.sh"
echo ""
echo "  This exports fused-memory data, pushes all branches to remote,"
echo "  and prints rsync commands for transferring repos + data + credentials."
echo ""
echo "  On THIS host, after rsync completes:"
echo "    bash $REPO_ROOT/scripts/import-data.sh ~/dark-factory-export"
echo ""

# ---------------------------------------------------------------------------
# 12. Health checks
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

# jCodeMunch watcher
if systemctl --user is-active jcodemunch-watcher &>/dev/null; then
  ok "jCodeMunch watcher: running"
else
  warn "jCodeMunch watcher: not running (check: journalctl --user -u jcodemunch-watcher)"
fi

echo ""
info "Setup complete"
