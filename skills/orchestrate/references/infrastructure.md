# Infrastructure Setup & Troubleshooting

Read this when services are down, connections fail, or you need to set up the environment from scratch.

## Backing Stores

The orchestrator depends on two backing stores (FalkorDB for Graphiti knowledge graph, Qdrant for Mem0 vector memory) plus the fused-memory MCP server.

### Start backing stores

```bash
cd /home/leo/src/dark-factory/fused-memory/docker
docker compose up -d
```

This starts:
- **FalkorDB** on port 6379 (Redis-compatible graph database)
- **Qdrant** on port 6333/6334 (vector search engine)

The docker-compose also defines a `fused-mcp` service, but fused-memory runs as a systemd user service (port 8002) — you typically only need the two backing stores from docker-compose.

### Verify backing stores

```bash
# FalkorDB (Redis protocol — not HTTP)
(echo PING; sleep 0.5) | nc -w2 localhost 6379 | grep PONG
# Expected: +PONG
# If redis-cli is available, you can also use: redis-cli -p 6379 ping

# Qdrant (HTTP)
curl -sf http://localhost:6333/readyz
# Expected: (empty 200 response)
```

### Common issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `Connection refused :6379` | FalkorDB not running | `cd fused-memory/docker && docker compose up -d falkordb` |
| `Connection refused :6333` | Qdrant not running | `cd fused-memory/docker && docker compose up -d qdrant` |
| `Connection refused :8002` | fused-memory server not running | fused-memory runs as a systemd user service on port 8002. Check: `systemctl --user status fused-memory`. Do **not** start/restart/stop without explicit user permission. |
| `OPENAI_API_KEY not set` | Missing env var | Export it: `export OPENAI_API_KEY=sk-...` (needed for embeddings) |
| `ANTHROPIC_API_KEY not set` | Stale check — no longer required | Orchestrator agents use OAuth (Max subscription). If fused-memory's Graphiti extraction needs Anthropic models, set it in `fused-memory/config/config.yaml`, but this is not required for orchestrator runs. |
| Docker containers exit immediately | Port conflict or stale volume | `docker compose down -v && docker compose up -d` |
| `uv: command not found` | uv not installed | `curl -LsSf https://astral.sh/uv/install.sh | sh` |

## Python Environment

The orchestrator is a uv-managed Python project:

```bash
cd /home/leo/src/dark-factory/orchestrator
uv sync
```

This installs dependencies into a local `.venv`. The `uv run --project orchestrator` prefix handles this automatically.

## Fused-Memory Server

fused-memory runs as a **systemd user service** on port 8002. It must be running before launching the orchestrator. Do **not** start, restart, or stop it without explicit user permission.

```bash
# Check status
systemctl --user status fused-memory

# Health check
curl -sf http://localhost:8002/health
```

## Bubblewrap (bwrap) Sandbox

The orchestrator sandboxes implementer/debugger agents with bubblewrap. On Ubuntu 24.10+ / Linux 6.17+ with `apparmor_restrict_unprivileged_userns=1`, bwrap needs an AppArmor profile granting the `userns` permission.

### Setup

```bash
sudo ./orchestrator/scripts/setup-bwrap.sh
```

### Verify

```bash
bwrap --ro-bind / / --dev /dev --proc /proc -- /bin/true && echo OK
```

### Common issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `setting up uid map: Permission denied` | AppArmor blocks unprivileged user namespaces | Run `setup-bwrap.sh` to install the profile |
| `bwrap: command not found` | bubblewrap not installed | `sudo apt install bubblewrap` |
| Warning: "bwrap unavailable — running unsandboxed" | Probe failed, agents run without sandbox | Run `setup-bwrap.sh`, then restart orchestrator |

If bwrap can't be fixed (e.g., container environments), the orchestrator degrades gracefully — agents run unsandboxed with a logged warning.

## Full Reset

If everything is in a bad state:

```bash
# Stop all containers
cd /home/leo/src/dark-factory/fused-memory/docker
docker compose down

# Clear data volumes (destroys all stored memories and graph data)
docker compose down -v

# Restart fresh
docker compose up -d

# Re-sync Python deps
cd /home/leo/src/dark-factory/orchestrator && uv sync
cd /home/leo/src/dark-factory/fused-memory && uv sync
```
