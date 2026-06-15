# Capability manifest — Dashboard per-orchestrator last-update + fused-memory uptime PRD

Per-leaf capability→evidence bindings (mechanized G3+G6). Built at author time 2026-06-15.
PRD: `plans/dashboard-orch-update-and-fm-uptime-prd.md`. Verdict: **no FAIL bindings — batch clear to queue.**

## α — Orchestrators panel: per-orchestrator last-update (dashboard)

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| Tasks carry `updated_at` to take a max over | grep:dashboard/src/dashboard/data/tasks.py:66 `'updated_at': task.get('updatedAt')`; already the recency key at active_tasks.py:230 | wired |
| `discover_orchestrators` has the task list in scope where `summary` is built (place to compute `last_update`) | grep:dashboard/src/dashboard/data/orchestrator.py:300-318 — `tasks` unpacked from `project_cache`, `summary` built from it, emitted in the per-orch dict | wired |
| Shape layer can copy a new per-orch field onto the `ORCHESTRATORS` entry | grep:dashboard/src/dashboard/data/redux_api.py:53-62 `out_orchs.append({... 'started': ...})` — add `'last_update'` sibling | wired |
| Render surfaces exist (OrchTab + overview row) | OrchTab static/redux/tabs.jsx (renders `started`, summary per orch); overview row tab_overview.jsx (Orch/Project/Modules/Done/⏱ columns) | wired |
| Empty-tree fallback is honest, not synthetic | `last_update = None` when no dated tasks → frontend renders "—" (no-synthetic-data rule) | wired |

## β — fused-memory `get_status` returns uptime fields (fused-memory)

| Capability | Evidence | Verdict |
|---|---|---|
| `get_status()` exists and returns a mutable dict to extend | grep:fused-memory/src/fused_memory/services/memory_service.py:1918-1969 — builds `status: dict`, returns it | wired |
| `MemoryService` has a construction point to stamp process start | service is constructed once per process (the get_status method is an instance method on a long-lived service); capturing start time at `__init__` is the task's own deliverable, not an assumed substrate | producer:task-β (this task) |
| MCP `get_status` tool passes the dict through to callers whole | dashboard `get_memory_status` already receives graphiti/mem0/queue/projects from this same dict via the MCP tool → new sibling keys propagate by the same path | wired |
| Monotonic-increase signal is achievable | wall-clock/`time.monotonic()` delta from a fixed start stamp increases by construction across calls in one process lifetime | wired |

## γ — Memory panel shows fused-memory uptime (dashboard)

| Capability | Evidence | Verdict |
|---|---|---|
| `uptime_seconds` / `started_at` present in the get_status response | producer: **β upstream** (dep wired γ→β); seam contract pins field names + types | producer-upstream, wired |
| Dashboard memory data layer receives the get_status dict | grep target dashboard/src/dashboard/data/memory.py `get_memory_status` (calls MCP `get_status`, returns dict) — passes new fields through | wired |
| Shape layer threads fields into `MEMORY_STATUS` | dashboard/src/dashboard/data/redux_api.py `shape_memory` builds `MEMORY_STATUS` (graphiti/mem0/queue/projects/wal today) — add `uptime_seconds`/`started_at` | wired |
| Render surfaces (MemoryTab KPI + overview health block) | static/redux/tabs.jsx MemoryTab (KPI tiles); tab_overview.jsx memory health block (per-component status rows) | wired |
| Offline/absent fallback is honest | `MEMORY_STATUS.offline` / missing `uptime_seconds` → render "—" (existing offline path + no-synthetic-data rule) | wired |

## DAG-direction (anti-inversion) check

- α: no deps; producers (task.updated_at, fetch_tasks) all exist on main upstream. OK.
- γ depends_on β; the consumed capability (`uptime_seconds`) is produced by β which is **upstream** of γ. No inversion. OK.

## G6 premise notes

- α signal asserts no number — relative-time of an existing field; trivially true.
- β signal "monotonically increasing uptime_seconds" — true by construction of a delta from a fixed start stamp.
- γ signal asserts no number beyond formatting `uptime_seconds`; the value's truth is β's responsibility.
