# Curator dashboard section

## Mission
Add a top-level **Curator** tab to the redux dashboard so an operator can see:
- what's in the curator's queue right now,
- how fast it's moving when uncapped (blue latency sparkline + centiles),
- recent cap history (red 0/1 step sparkline),
- and cancel any individual queued ticket.

## Why
The task curator (fused-memory `TaskInterceptor` + `TaskCurator`) is the throughput bottleneck for task creation. When it stalls — a slow LLM, a wedged worker, an all-accounts cap — the user currently has no visual signal short of tailing `journalctl --user -u fused-memory.service`. We have already observed a single curator call blocking a project for **31+ min** (see `observation_curator_single_call_31min_blocking`). This section turns the curator into a first-class observability surface.

## Architecture overview

```
Curator UsageGate ─┐                       ┌─ list_tickets (pending)
                   ├─→ account_events ◄────┤
Orchestrator gate ─┘   (cap intervals)     │
                                           │
ticket_store ──→ tickets.db                ├─→ /api/v2/dashboard/curator
   (created_at,                            │
    resolved_at)  ──→ active latency       │
                       (wall − overlap)    │
                                           │
metrics sampler ─→ curator_snapshots ──────┘
                   (10-min ticks)
```

## Data model

### Reused
- `account_events` (existing) — `cap_hit` / `resumed` rows already populated by the orchestrator's `UsageGate`. **Will also need to flow for the curator's gate** (see [task 2]).
- `tickets.db` (existing) — `tickets` table with `created_at`, `resolved_at`, `status`, `project_id`.

### New
- `curator_snapshots` table in `dashboard/data/metrics.db`:
  ```sql
  CREATE TABLE IF NOT EXISTS curator_snapshots (
      ts              TEXT PRIMARY KEY,
      pending_total   INTEGER,         -- sum of pending tickets across projects
      capped_now      INTEGER,         -- 0 or 1: is the curator gate currently capped
      p50_active_ms   INTEGER,         -- centile of (wall − capped overlap) over last 1h
      p90_active_ms   INTEGER,
      p99_active_ms   INTEGER
  );
  ```

## Definitions

### "Active latency" per ticket
```
active = (resolved_at − created_at) − overlap(ticket_interval, all_accounts_capped_intervals)
```
`all_accounts_capped_intervals` is the union of `cap_hit → resumed` pairs across **every** account, intersected to the time when every account was capped simultaneously (the only condition that actually blocks `before_invoke`). Compute it once per sampler tick.

### Cap sparkline
A 0/1 step series at 10-min resolution. `1` ⇔ "all accounts capped" at that bucket boundary. Source: the merged interval set above, evaluated at each bucket's right edge.

### Pending queue
Union of `list_tickets(project_id, status='pending')` across every known project root. Default ordering newest-first.

## Decisions (locked)

1. **Cap data source**: re-use existing `account_events` (the Costs tab already reads it). Factor the cap-interval reader into a shared `dashboard/data/cap_history.py` helper and refactor `costs.py` to use it.
2. **Cancel semantics**: best-effort, queue-only. Cancel marks the ticket `cancelled` in the store; `_prepare_ticket` already skips non-pending rows so anything still queued is dropped. In-flight LLM calls win the race and may produce a `created` outcome — the cancel button on the UI just disappears when the ticket leaves pending. v1 trade-off accepted.
3. **Latency filter**: subtract capped overlap per ticket → "active latency" (option b in the design conversation). Uses every terminal ticket; no samples discarded.
4. **UI placement**: new top-level **Curator** tab on the rail, alongside Memory / Reconciliation.

## Components

### Fused-memory side (server tools + middleware)
- `TaskInterceptor.cancel_ticket(ticket_id)` → calls `TicketStore.mark_resolved(id, status='cancelled', reason='user_cancelled')`, then `self._signal_ticket_event(id)`. Idempotent: cancelling a terminal ticket returns `{status: <existing>, no_op: True}`.
- `server/tools.py` exposes `cancel_ticket` as an MCP tool.
- `server/main.py` wires a `CostStore` (path: `<config.data_dir>/curator_events.db` or share `fused-memory/data/curator_account_events.db`) into the curator `UsageGate` so cap events persist.

### Dashboard side
- `dashboard/data/cap_history.py` — new module. Exposes:
  - `read_cap_intervals(dbs, *, days)` → `list[CapInterval]` per account.
  - `merge_all_accounts_capped(intervals_by_account)` → `list[(start, end)]` of windows when every account was capped.
  - `bucketise_cap_sparkline(intervals, *, bucket_seconds=600, window_hours=24)` → `ChartData`.
  - `compute_overlap_ms(ticket_start, ticket_end, intervals)` → int.
  - `costs.py` refactored to import from the new helper (no behaviour change on the Costs tab).
- `dashboard/data/metrics.py` — add `curator_snapshots` to `METRICS_SCHEMA`, add a sampler that:
  1. counts pending tickets via fused-memory HTTP (`list_tickets`),
  2. reads cap intervals via the helper,
  3. computes p50/p90/p99 active latency over the last 1 h of terminal tickets in `tickets.db`,
  4. writes one row.
  Add a corresponding `get_curator_sparks(metrics_db, days=1)` reader returning `{pending, capped, p50, p90, p99}` ChartData bundles.
- `dashboard/app.py` —
  - `GET /api/v2/dashboard/curator`: aggregates pending tickets across projects (via fused-memory HTTP) + sparks from metrics.db + cap sparkline from the helper, shaped by `redux_api.shape_curator`.
  - `POST /api/v2/dashboard/curator/cancel`: body `{ticket_id}`, proxies to fused-memory MCP `cancel_ticket`.
- `dashboard/data/redux_api.py` — `shape_curator(...)` returns `{CURATOR_STATE: {pending, latency_spark: {p50,p90,p99}, capped_spark, state: {capped_now, paused_reason, pending_total}}}`.

### UI
- New `dashboard/src/dashboard/static/redux/tab_curator.jsx`:
  - State pill (top): green "Open" / red "Capped: <reason>".
  - Two stacked sparklines: blue multi-line (p50 light, p90 medium, p99 dark) active latency; red step 0/1 cap.
  - Queue table grouped by project: ticket id (short), title, files (chips), age, **Cancel** button.
- Register in `tabs.jsx` (`window.DF_TABS.CuratorTab`), `app.jsx` (add `'curator'` tab + render switch), `shell.jsx` (rail entry), `styles.css` (sparkline + table styles already present).
- Cancel button does optimistic UI: instantly removes the row, fires `POST /api/v2/dashboard/curator/cancel`, refetches on next tick.

## Out of scope (v1)
- Per-account cap visualisation (always rolls up to "all accounts capped").
- Hard cancel of in-flight LLM calls.
- Bulk cancel.
- Filtering / search inside the queue table.
- Retry buttons for failed tickets (the existing `curator_escalator` already handles failure reporting).

## Acceptance criteria
- Submitting a task while every account is capped shows it in the Curator queue table within ~5 s.
- Cancelling that ticket from the UI flips it to `cancelled` in `tickets.db` and removes it from the queue within ~5 s.
- After uncap, the active-latency sparkline reflects new terminal tickets at the next 10-min sampler tick.
- The cap sparkline shows a `1`-bucket spanning the capped window.
- Costs tab continues to render identically (refactor is behaviour-preserving).
- `pyright` and `pytest` clean for every touched package.
