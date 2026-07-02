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

## Orchestrator Fleet Staleness — Two-Tier Restart

Per `plans/orchestrator-fleet-staleness-prd.md`: after a merge touching
orchestrator-core code (`orchestrator/src/**`, `escalation/src/**`, and the
sibling `pyproject.toml`/`uv.lock` manifests) lands on dark-factory main,
every running `orchestrator-*.service` user unit is restarted onto the new
code — not just `orchestrator-dark-factory.service`. This happens via two
independent tiers; understanding which one acted (or should have) matters
when triaging a stale-code incident (e.g. dispatch-time `ImportError` after a
rename).

### Tier 1 (primary): event-driven restart-all coordinator

The dark-factory daemon is the **unique observer** of orchestrator-core
merges (they only ever land through its own merge queue). Its
`StaleServiceRestartCoordinator` (`Harness._build_orchestrator_restart_coordinator`)
watches every landed merge's diff and, once armed, fires
`scripts/restart-all-orchestrators.sh` — which enumerates the running fleet
at runtime and restarts every unit, deferring the df daemon's own unit
(`SELF_UNIT`) to last so a mid-script df death can't strand the others
unrestarted.

The fire is gated to be polite: debounced (default 300 s quiet window),
only at the run-loop's agents-idle window, and only once the merge
queue/pipeline is fully drained (`Harness._merge_pipeline_idle`). A burst of
several core merges inside the debounce window coalesces into exactly one
restart-all fire.

### Tier 2 (backstop): watchdog staleness pass

`scripts/orchestrator-watchdog.py` runs as a oneshot on a 60 s systemd timer
(`orchestrator-watchdog.timer`). Every tick, after its liveness probe, it
also runs a fleet-wide staleness pass: any running `orchestrator-*.service`
unit whose realtime start time predates the newest commit touching the
watched paths is restarted, with a WARNING logged to the unit's journal
(`systemd-cat -t orchestrator-watchdog`).

This backstop exists because the event-driven tier has known blind spots —
the df daemon itself down/crashed (its in-memory pending flag is lost),
direct-to-main commits that bypass the merge worker entirely, a fire-time
failure inside the detached restart script, the coordinator giving up after
repeated transient failures, or the `orchestrator_restart_on_merge_enabled`
knob simply being off. The staleness pass converges all of these within one
grace period + one tick, and — because staleness is recomputed from live
systemd + git state on every call rather than stored — it self-clears as
soon as any restart (from either tier, a deploy capstone, or a manual
operator restart) lands: no stored state, no flap loop.

To avoid racing the polite event-driven tier, the backstop only acts once
the newest watched commit is older than `STALENESS_GRACE_SECS` (30 minutes)
— giving Tier 1 a head start — and additionally respects each unit's
startup grace window and `is_unit_enabled` (an operator-disabled unit is
left alone).

### When to run `--report`

```bash
python3 scripts/orchestrator-watchdog.py --report
```

Read-only doctor mode: prints a per-unit table (unit / start time / newest
watched commit / verdict) for the dynamically-enumerated running fleet, and
exits 0 if every unit is fresh or 1 if any unit is stale. It performs **zero
mutating systemctl calls** — safe to run at any time to check fleet state,
e.g. after a suspected stale-code incident or before/after a deploy
capstone.

Note the verdict is a raw `start_epoch` vs. newest-watched-commit comparison
only — it does **not** apply the restraint gates (`is_unit_enabled`,
startup grace, commit grace) that the actual staleness pass applies before
restarting. A unit reported `stale` may be one the backstop correctly leaves
alone this tick (e.g. still within commit-grace, giving Tier 1 time to act).
Treat `stale` as "not running code from the newest watched commit," not as a
prediction that a restart is imminent.

### Soak-watch pointers

- `journalctl --user -u 'orch-selfrestart-on-merge-*'` — the Tier-1
  coordinator's detached transient units. A fire-time failure of the restart
  script itself is journald-only (no in-process escalation); this is the
  only trace of it.
- Reconciliation's staleness findings remain an independent detector of
  residual gaps in both tiers — this is the same detection path that
  originally filed task 2003 for the incidents this two-tier design fixes.

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
