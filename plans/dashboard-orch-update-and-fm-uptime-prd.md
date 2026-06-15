# Dashboard: per-orchestrator last-update + fused-memory uptime

**Status:** active — 2026-06-15. Two small, independent observability enhancements to the redux dashboard.

## Goal

Two at-a-glance freshness signals on the dashboard:

1. **Orchestrators section** — each orchestrator row shows when its task tree last changed ("updated 4m ago"), so a viewer can tell a *progressing* orchestrator from a *stalled* one without opening the task table. Today only the process `started` time is shown — a long-running orchestrator that's wedged looks identical to one that's actively closing tasks.
2. **Memory section** — show how long the fused-memory server has been up ("fused-memory up 3d 7h"). Today the memory panel shows connectivity/counts/WAL but no liveness duration; a recent restart (the watchdog SIGABRT recoveries are ~6/day per `recon_stale_run` note) is invisible.

## Background

- Dashboard: redux/JSX frontend served by uvicorn (`dark-factory-dashboard.service`); FastAPI app at `dashboard/src/dashboard/app.py`; data producers under `dashboard/src/dashboard/data/`; redux shaping in `dashboard/src/dashboard/data/redux_api.py`; frontend under `dashboard/src/dashboard/static/redux/`.
- Orchestrators: `discover_orchestrators()` (`data/orchestrator.py:243`) already fetches each project's task list via `fetch_tasks` and builds a `summary`; tasks carry `updated_at` (preserved from `updatedAt` in `data/tasks.py:66`, already used as the recency key in `data/active_tasks.py`). Shaped into the `ORCHESTRATORS` array by `shape_orchestrators()` (`data/redux_api.py:32`); rendered by `OrchTab` (`static/redux/tabs.jsx`) and the overview panel (`static/redux/tab_overview.jsx`).
- Memory: the panel's data comes from `get_memory_status()` (`data/memory.py`), which calls the fused-memory MCP `get_status` tool and returns its dict whole; shaped by `shape_memory()` (`data/redux_api.py`) into `MEMORY_STATUS`; rendered by `MemoryTab` (`tabs.jsx`) + the overview health block (`tab_overview.jsx`). `get_status` is defined server-side at `fused-memory/src/fused_memory/services/memory_service.py:1918`.

## Sketch of approach

**Feature 1 — per-orchestrator last update (dashboard only).** In `discover_orchestrators()`, alongside `summary`, compute `last_update = max(updated_at over the project's tasks)` (None when the tree is empty / no task carries `updated_at`). `shape_orchestrators()` copies `last_update` onto each `ORCHESTRATORS` entry (sibling of `started`). `OrchTab` and the overview row render it as a relative time ("updated 4m ago"), falling back to "—" when absent.

**Feature 2 — fused-memory uptime (server self-reports; resolved design decision).** The fused-memory server is the authority on its own uptime, so it reports it rather than the dashboard inferring it from systemd (which would couple the dashboard to the exact `--user` unit name and break for any non-systemd / remote instance). The `MemoryService` captures its process start time once at construction; `get_status()` returns `uptime_seconds` (int) and `started_at` (ISO-8601 UTC) as top-level fields. The dashboard already calls `get_status` for the memory panel, so `get_memory_status()` passes the new fields through, `shape_memory()` threads them into `MEMORY_STATUS.uptime_seconds` / `.started_at`, and the Memory tab + overview health block render a compact duration ("up 3d 7h"), falling back to "—" when offline/absent.

### Seam contract (Feature 2, intra-batch cross-package)

`get_status()` return dict gains two top-level keys (siblings of `graphiti`, `mem0`, `queue`, `projects`):

| Field | Type | Meaning |
|---|---|---|
| `uptime_seconds` | `int` | Whole seconds since the fused-memory server process started, captured once at `MemoryService` construction. |
| `started_at` | `str` | ISO-8601 UTC timestamp of process start. |

The dashboard reads `uptime_seconds` (primary; `started_at` is informational/hover). These are the only fields crossing the package boundary; β produces them, γ consumes them.

## Resolved design decisions

1. **Uptime source = server self-report via `get_status`** (not dashboard-side `systemctl`). Robust to unit-name drift and non-systemd/remote instances; reuses the get_status call the memory panel already makes. (User decision, 2026-06-15.)
2. **"Last update" = max task `updated_at`** across the orchestrator's tree — the cleanest available proxy for "is this orchestrator's work advancing", already fetched. Not a separate orchestrator heartbeat.
3. **Formatting lives in the frontend.** Backend passes raw values (`last_update` ISO string; `uptime_seconds` int); JSX formats relative-time / compact-duration, matching the existing pattern where the backend ships raw `started` and the frontend renders. Empty/absent → "—" (honours the no-synthetic-data rule).
4. **Feature 2 split into two single-package tasks** (β fused-memory, γ dashboard) per the multi-package-split rule; γ `depends_on` β.

## Pre-conditions for activating

None — no upstream PRDs or substrate prerequisites. All substrate verified present (see capability manifest beside this PRD).

## Cross-PRD relationship

No cross-PRD seams. One intra-batch cross-package seam (`get_status` uptime fields, β→γ), owned entirely within this PRD's decomposition and wired as a dependency.

## Decomposition plan

- **α — Orchestrators panel: per-orchestrator last-update timestamp** (modules: `dashboard`). Leaf. Compute `last_update` in `discover_orchestrators` (max task `updated_at`, None when empty); surface via `shape_orchestrators`; render relative-time in `OrchTab` + overview row, "—" when absent. **Signal:** an orchestrator row in the Orchestrators tab and the overview panel shows "updated &lt;relative&gt; ago" reflecting its most recently updated task, and "—" for an orchestrator whose tree has no dated tasks. No deps.

- **β — fused-memory `get_status` returns `uptime_seconds` + `started_at`** (modules: `fused-memory`). Intermediate (unlocks γ). Capture process start at `MemoryService` construction; add the two top-level fields to `get_status()`'s return per the seam contract. **Signal:** calling the `get_status` MCP tool returns an integer `uptime_seconds` and ISO `started_at` that increase monotonically across successive calls within one process lifetime. No deps. **Unlocks:** γ.

- **γ — Memory panel shows fused-memory uptime** (modules: `dashboard`). Leaf. Pass the new fields through `get_memory_status`; thread into `MEMORY_STATUS` via `shape_memory`; render a compact duration in `MemoryTab` + the overview health block, "—" when offline/absent. **Signal:** the Memory tab / overview health block shows "fused-memory up &lt;Nd Nh&gt;" derived from `uptime_seconds`, and "—" when fused-memory is unreachable. **Depends on:** β.

## Out of scope

- Per-orchestrator heartbeat independent of task activity (e.g. last scheduler-tick time) — `updated_at` is the v1 proxy.
- Uptime sparkline / restart-count history — single scalar only.
- Surfacing `started_at` as anything more than a hover/title on the uptime figure.
- Backing-store (Neo4j/Qdrant/docker) uptime — fused-memory server process only.

## Open questions (tactical, defer to implementation)

1. **Relative-time / duration formatter.** Reuse an existing JS time-format helper in `static/redux/` if one is present; otherwise add a small local helper. **Decide during α/γ.**
2. **Relative-time refresh granularity.** The data loader polls every ~3 s, so the rendered "Xm ago" updates on poll; no separate ticking timer needed unless it looks stale. **Decide during α.**
