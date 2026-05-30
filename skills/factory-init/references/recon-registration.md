# Reconciliation Registration

Stage 6. This is the one step that touches **dark-factory itself** rather than the target repo, and the one destructive step (it restarts a shared service). Read all of it before acting.

## Why this is mandatory, and why order matters

fused-memory's write path (`add_memory`, `submit_task`, `set_task_status`, …) accepts **any** `project_id` matching `[A-Za-z0-9_-]`. But the reconciliation harness builds a registry of known `project_id → project_root` mappings at process start, and its project loop **hard-rejects** any id not in that registry (`UnknownProjectError`) — and does **not** quarantine the offending event. The practical consequences, both learned the hard way:

- If the project isn't registered, the first task status-change (`done`/`blocked`/`cancelled`) that triggers reconciliation fails pre-flight, and a mistyped or unknown id can **poison the recon event buffer permanently** — one bad id, and the buffer never drains.
- Therefore registration + restart must complete **before** any `/prd` run in Stage 7 files a task. Never queue against an unregistered id "to test it."

The registry is built from two sources at fused-memory startup:
1. `taskmaster.project_root` in `fused-memory/config/config.yaml` (dark-factory's own primary project).
2. The `DASHBOARD_KNOWN_PROJECT_ROOTS` env var — a comma-separated list of absolute paths for every *additional* project. This is what you extend.

## Procedure

Let `<DF>` be the dark-factory repo root and `<TARGET>` the absolute target path.

### 1. Edit the source of truth (the template)

`<DF>/scripts/fused-memory.service.template` has a line like:

```
Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=__REPO_ROOT__,/home/leo/src/reify,/home/leo/src/autopilot-video,/home/leo/src/autotrade,/home/leo/src/know-live
```

Append `,<TARGET>` to the end of that line. Editing the template (not just the live unit) means the registration survives the next time anyone re-runs `setup-host.sh`. `__REPO_ROOT__` is a placeholder for `<DF>` and is fine to leave as-is — the new entry is a literal absolute path.

**Keep `dashboard.service.template` in sync.** The same `DASHBOARD_KNOWN_PROJECT_ROOTS=` line appears in `<DF>/scripts/dashboard.service.template` (the comment in the fused-memory template says so explicitly). Make the identical append there, or the project won't show on the dashboard. The fused-memory copy is the one that governs reconciliation (the recon-storm hazard); the dashboard copy is cosmetic but should match.

### 2. Apply to the live unit (surgical edit — do NOT re-render)

The live unit at `$HOME/.config/systemd/user/fused-memory.service` already has every placeholder resolved. **Do not** re-render it from the template by substituting only `__REPO_ROOT__` — the template also contains `__UV_PATH__` (the `ExecStart` line), and a partial `sed` render would leave that placeholder literal and the restart would fail. (If you ever do re-render, substitute *both*: `sed -e 's|__REPO_ROOT__|<DF>|g' -e "s|__UV_PATH__|$(command -v uv)|g"`.)

Instead, append `,<TARGET>` to the single `DASHBOARD_KNOWN_PROJECT_ROOTS=` line in the live unit (and, for parity, the live dashboard unit `dark-factory-dashboard.service`), then reload:

```bash
systemctl --user daemon-reload
```

Confirm the live line is right before restarting:

```bash
systemctl --user show fused-memory.service -p Environment | tr ' ' '\n' | grep DASHBOARD_KNOWN_PROJECT_ROOTS
# expect the line to now end with ,<TARGET>
```

### 3. Restart (CONFIRM WITH THE USER FIRST)

```bash
systemctl --user restart fused-memory
```

⚠️ **This severs this session's fused-memory MCP connection** — the `mcp__fused-memory__*` tools will not reconnect for the rest of this session. That is acceptable here: nothing after Stage 6 needs MCP *in this session*, and the sessions spawned in Stage 7 open fresh connections to the restarted server. Still, get the user's explicit go-ahead (they authorised "full auto with confirm").

Do **not** restart fused-memory if a dark-factory orchestrator run is mid-flight against another project unless the user accepts the interruption — the restart is global to the shared server.

### 4. Verify

```bash
# Health probe — curl is broken on this host, so use python urllib:
python3 -c "import urllib.request,sys; \
print(urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=10).read().decode())"

# Confirm the project is recognised (registration detection is heuristic):
journalctl --user -u fused-memory.service --since '2 min ago' --no-pager | grep -iE 'known project|registr|<PROJECT_ID>'
```

If the health probe fails, check `systemctl --user status fused-memory` and the journal — a bad edit to the unit file will fail the restart. If the service is healthy but the journal doesn't mention the project, the registry may not have picked up the root (the prefix detection is heuristic); re-check the rendered `Environment` line and that the path is absolute and exists.
