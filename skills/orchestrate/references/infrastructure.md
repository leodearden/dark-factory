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

The docker-compose also defines a `fused-mcp` service, but the orchestrator manages its own fused-memory server process — you typically only need the two backing stores from docker-compose.

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
| `Connection refused :8000` | fused-memory server not running | The orchestrator starts this automatically. If running tools manually: `cd /home/leo/src/dark-factory && uv run --project fused-memory python -m fused_memory.server.main --transport http` |
| `OPENAI_API_KEY not set` | Missing env var | Export it: `export OPENAI_API_KEY=sk-...` (needed for embeddings) |
| `ANTHROPIC_API_KEY not set` | Missing env var | Export it: `export ANTHROPIC_API_KEY=sk-ant-...` (needed for agent invocations) |
| Docker containers exit immediately | Port conflict or stale volume | `docker compose down -v && docker compose up -d` |
| `uv: command not found` | uv not installed | `curl -LsSf https://astral.sh/uv/install.sh | sh` |

## Python Environment

The orchestrator is a uv-managed Python project:

```bash
cd /home/leo/src/dark-factory/orchestrator
uv sync
```

This installs dependencies into a local `.venv`. The `uv run --project orchestrator` prefix handles this automatically.

## Fused-Memory Server (manual)

The orchestrator starts and stops fused-memory automatically. If you need to run it manually (e.g., for interactive MCP tool usage):

```bash
cd /home/leo/src/dark-factory
uv run --project fused-memory python -m fused_memory.server.main --transport http --config fused-memory/config/config.yaml
```

This starts the HTTP server on port 8000. Health check: `curl http://localhost:8000/health`

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
