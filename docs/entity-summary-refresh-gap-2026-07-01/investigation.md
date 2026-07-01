# Entity-summary post-invalidation refresh gap — investigation (2026-07-01)

Investigation-only write-up (task 1948). Audits every Graphiti edge-mutation
path in fused-memory to determine which ones invalidate/supersede/dedup
edges *without* triggering `refresh_entity_summary`, states the root cause
of the recurring entity-summary staleness observed by task 1946, and
determines the right fix among the three options that task raised.

**No runtime code was changed by this investigation, and the 2026-07-01
backfill (task 1946) was NOT re-run here.** This document is the sole
deliverable.

Staleness, as used throughout: a stored Graphiti entity-node `summary` that
differs from the canonical dedup of that entity's currently-valid
(`invalid_at IS NULL`) edge facts — i.e. what `refresh_entity_summary` would
produce if called right now.

## TL;DR

- Every **explicit** fused-memory edge-mutation seam (`update_edge`,
  `merge_entities`, `delete_entity`) already auto-refreshes the affected
  entity summaries at the backend. These are **not** the gap.
- The **primary gap** is ingestion: `GraphitiBackend.add_episode` (backed by
  `add_episode`/`add_memory(dual_write=True)`) hands off entirely to
  `graphiti_core`, which invalidates/supersedes/dedups edges internally as
  part of extraction — and nothing calls `refresh_entity_summary` afterward.
  Ingestion calls vastly outnumber explicit edge mutations, so this is the
  dominant driver of the staleness task 1946 measured (dark_factory ~61%,
  know_live ~53%, 6,437/10,918 entities combined, 0 rebuild errors).
- Task 126's Stage-1 prompt fix (2026-03-28) cannot close this gap — it's
  LLM/agent-driven and can only refresh entities an agent knows it changed.
  It's also largely redundant with the backend `update_edge` seam.
- **Recommendation: (b) a periodic `rebuild_entity_summaries` maintenance
  job as the primary fix, complemented by (a) a targeted best-effort
  post-ingestion refresh; reject (c) extending task 126 as insufficient
  alone.** See §4/§5.

## 1. Context / evidence recap

Task 1946 ran `rebuild_entity_summaries` project-wide on 2026-07-01 for the
two projects with confirmed-affected entities:

| Project | Stale (rebuilt) | Total entities | Stale rate | Errors |
|---|---:|---:|---:|---:|
| dark_factory | 5,145 | 8,492 | ~60.6% | 0 |
| know_live | 1,292 | 2,426 | ~53.3% | 0 |
| **Combined** | **6,437** | **10,918** | **~59.0%** | **0** |

All detected-stale entities rebuilt cleanly (0 errors) — this was a pure
operational backfill, no code change (task 1946's original AC1/AC2, which
proposed a fix to the rebuild query's `invalid_at` filter, were dropped as a
confirmed false premise: the filter was already correct; see §3 and §7).

The scale here — roughly half-to-majority of all entity nodes in both live
graphs — is why this investigation exists: task 126 (done, 2026-03-28)
already shipped a mitigation for post-invalidation staleness, yet staleness
recurred at large scale by 2026-07-01. This document audits every
edge-mutation path to explain why.

## 2. Edge-mutation-path audit

All line numbers verified against base branch `6135cf9`.

| # | Path | Effect on edges | Refreshes `refresh_entity_summary`? | Citation | Status |
|---|---|---|---|---|---|
| 1 | `GraphitiBackend.update_edge` (fact-text update, `invalid_at` supersede, or `clear_invalid_at` restore) | Updates/invalidates/restores one edge | **Yes** — unconditionally loops over `(edge.source_node_uuid, edge.target_node_uuid)` and refreshes both, even for a pure `invalid_at`-only supersede with no fact-text change. Best-effort per node (logged, non-fatal). | `fused-memory/src/fused_memory/backends/graphiti_client.py` — method L378-449; refresh loop L432-443 | **COVERED** |
| 2 | `merge_entities` | Redirects all of the deprecated node's edges onto the surviving node, deletes the deprecated node | **Yes** — refreshes the surviving node once after redirect+delete. (Deprecated node is gone; nothing to refresh there.) | same file — method L776-832; refresh call L815 | **COVERED** |
| 3 | `delete_entity` | Deletes a node (its edges vanish via `DETACH DELETE`) | **Yes** — collects neighbour UUIDs *before* delete, then refreshes each neighbour's summary in a best-effort loop (failures collected in `refresh_errors`, non-fatal, since the delete itself is already irreversible) | same file — method L834-909+; refresh loop L891-901 | **COVERED** |
| 4 | `add_episode` (backend) — driving both the `add_episode` MCP tool and `add_memory(dual_write=True)`'s Graphiti-side write | Extracts new edges from episode content; `graphiti_core` internally resolves each candidate against existing edges for the same entity pair, **invalidating/superseding duplicates and deduping near-identical facts as a side effect**, then regenerates the node summary by its own internal (non-fused-memory) logic | **No** — `GraphitiBackend.add_episode` is a thin wrapper: it `await`s `client.add_episode(...)` and returns the raw result. No fused-memory code observes which edges `graphiti_core` invalidated/superseded, so nothing calls the fused-memory canonical `refresh_entity_summary`. | same file — method L252-281 (no refresh call anywhere in the method) | **PRIMARY GAP** |
| 5 | Stage-1 task-126 prompt instruction | N/A (a *process*, not a code seam) — instructs the Stage-1 LLM agent to call `refresh_entity_summary` itself after *it* deletes/invalidates edges | Only when Stage 1 performs the invalidation and the agent remembers to follow the instruction — best-effort, LLM-driven, no code-level enforcement, Stage-1-only (Stage 2/3 and other MCP callers aren't covered by this prompt text) | `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py` L56-59 | **Best-effort / largely redundant with row 1 for the paths it *can* see; blind to row 4** |

Row 4 is materially different from rows 1–3: rows 1–3 are all **explicit**
fused-memory API calls where fused-memory code decides to invalidate/merge/
delete an edge or node, so fused-memory code can (and does) refresh
afterward. Row 4 delegates edge resolution to `graphiti_core`'s internal
extraction pipeline — the invalidation/supersession/dedup happens *inside*
a single awaited call that fused-memory does not introspect.

Two additional wiring details relevant to designing a fix (§4(a)):

- `MemoryService._dual_write_callback` (`fused-memory/src/fused_memory/services/memory_service.py` L565-601, registered for `callback_type='dual_write_episode'` at L182-184) already runs **after every `add_episode` service call completes**, and already receives the `graphiti_core` result's `.edges`/`.entity_edges` list — today only to enqueue each fact for a durable Mem0 write. It does not call `refresh_entity_summary`. This is the natural seam for a targeted post-ingestion refresh.
- `MemoryService.add_memory(..., dual_write=True)` (same file, ~L698-780) enqueues its Graphiti-side write as an `add_memory_graphiti` operation *without* passing any `callback_type` (unlike `add_episode`'s `dual_write_episode`, ~L656) — so today this path doesn't even get the existing Mem0-dual-write side effect, let alone a summary refresh. Any fix targeting row 4 needs to cover both enqueue sites.

## 3. Root cause

`graphiti_core`'s temporal edge-resolution pipeline — invoked internally by
`add_episode`/`add_memory(dual_write=True)` on every ingest — silently
invalidates, supersedes, and dedups edges as a side effect of extracting
new facts from episode content. This resolution happens entirely inside the
single `client.add_episode(...)` call that `GraphitiBackend.add_episode`
awaits and returns (row 4 above); no fused-memory code observes *which*
edges were touched, so no fused-memory code can call
`refresh_entity_summary` afterward.

Task 187 (done) corroborates this asymmetry from the opposite direction:
"Graphiti does not rebuild summaries on edge deletion — only on new edge
addition." In other words, `graphiti_core`'s own internal summary handling
is itself keyed off new-edge addition, not off the fused-memory canonical
dedup of currently-valid edges — so even when `graphiti_core` does
regenerate a summary internally, it isn't guaranteed to match what
`refresh_entity_summary`/`rebuild_entity_summaries` would produce.

Per procedural memory `ffe0f4f8-0e0a-4050-9ae5-f098cbd82068` (2026-07-01,
written during task 1946's investigation): edges are **directed**, so an
ingestion-time invalidation of a `know-live → Mem0` edge, say, can leave the
`Mem0` endpoint's summary stale even after the `know-live` endpoint happens
to be refreshed through some other path. That memory also independently
confirmed — empirically, against the live `dark_factory` graph — that the
rebuild query's `WHERE e.invalid_at IS NULL` filter (`get_valid_edges_for_node`
/ `get_all_valid_edges` in `graphiti_client.py`) is and was already correct.
This rules out a confounding explanation: the staleness is **not** a
filter/compaction bug, it is purely a refresh-**triggering** gap. Task 1946
dropped its original AC1/AC2 (a proposed filter fix) as a confirmed false
premise on exactly this basis.

Because `add_episode`/`add_memory` calls vastly outnumber explicit
`update_edge`/`merge_entities`/`delete_entity` calls in normal operation
(every reconciliation cycle and most agent writes go through ingestion),
this unobserved ingestion-time supersession is the dominant contributor to
the staleness task 1946 measured: ~61% of dark_factory's entities and ~53%
of know_live's were stale as of 2026-07-01, despite task 126 having shipped
over three months earlier (2026-03-28). Task 126's mitigation only reaches
row 5 of the audit table (Stage-1 agent-driven invalidations), which is
largely redundant with the already-covered backend `update_edge` seam (row
1) — it was never positioned to reach row 4, and the recurrence at scale
confirms it doesn't.

## 4. Fix determination

Task 1948 asked which of three options is the right fix:

**(c) Extend the task-126 prompt instruction beyond Stage 1 — REJECTED, insufficient alone.**
The primary gap (audit row 4) is not agent-driven: no agent observes, or is
even invoked for, `graphiti_core`'s internal ingestion-time edge resolution.
A prompt instruction can only tell an agent to refresh entities it *knows*
it changed; it has no visibility into `graphiti_core`'s internal dedup
decisions inside `add_episode`. Worse, the invalidations this style of fix
*can* reach — explicit, agent-initiated edge changes via `update_edge` — are
already auto-refreshed at the backend (row 1), so extending 126 further
would mostly duplicate existing coverage rather than close the actual gap.

**(a) Auto-refresh at the ingestion seam — right direction, complement not sole fix.**
Concretely: extend `MemoryService._dual_write_callback` (memory_service.py
L565-601) — already invoked post-hoc with the `graphiti_core` result's edge
list — to also collect `{edge.source_node_uuid, edge.target_node_uuid}` for
every edge in the result and call `refresh_entity_summary` on each
(deduplicated), mirroring `delete_entity`'s existing best-effort
neighbour-refresh pattern (log-and-continue on failure, non-fatal). The
`add_memory_graphiti` enqueue path needs the same (or an equivalent)
callback wired in, since it currently has none.
- *Pros*: shrinks the drift window at the source — near-real-time instead
  of stale until the next periodic rebuild; reuses a pattern already proven
  safe in `delete_entity`/`merge_entities`.
- *Cons*: touches the async ingestion hot path (extra FalkorDB round-trips
  per episode); `graphiti_core`'s internal resolution can affect entities
  beyond just the newly-added edges' own endpoints (e.g. resolving against
  a pre-existing edge on an entity not otherwise named in this episode), so
  a callback keyed only off the returned edge list cannot cheaply guarantee
  it enumerates *every* affected entity. It reduces the gap; it doesn't
  close it.

**(b) Periodic `rebuild_entity_summaries` maintenance job — RECOMMENDED PRIMARY.**
The infrastructure already exists and is proven at scale: `maintenance/
rebuild_summaries.py` (`RebuildSummariesManager` / `run_rebuild_summaries()`
CLI entrypoint, parameterized by `group_id`/`force`/`dry_run`) is exactly
what task 1946 just ran project-wide for dark_factory and know_live with 0
errors across ~10.9K entities.
- *Pros*: bounds staleness regardless of root cause — including causes no
  seam could cheaply observe (e.g. `graphiti_core` resolving against an
  entity not directly named in the triggering episode); low-risk (rebuild
  is idempotent and read-then-write, already exercised at this scale);
  needs no new code path, only a scheduling wrapper around an existing,
  tested entrypoint.
- *Cons*: eventually-consistent — summaries can be stale for up to one
  period between runs; doesn't help a consumer reading between runs.

## 5. Recommendation

Ship **(b) as the primary, reliable fix** — it bounds staleness regardless
of cause, using infrastructure that already exists and that task 1946 just
validated at scale — **complemented by (a)** to reduce drift latency at the
dominant ingestion seam. **Reject (c)**: it cannot reach the primary gap and
mostly duplicates coverage the backend already provides.

## 6. Proposed follow-up implementation tasks

1. **(Primary, fix b) Periodic `rebuild_entity_summaries` maintenance job.**
   Add a systemd timer/service unit (following this repo's existing
   `scripts/fused-memory.service.template` convention) that invokes
   `maintenance/rebuild_summaries.py`'s `run_rebuild_summaries()` per active
   project (`dark_factory`, `know_live`, and any others onboarded later) on
   a defined cadence. Scope includes: choosing the cadence, `force=False` by
   default, alerting/logging when `errors > 0`, and documenting how to add a
   newly-onboarded project to the sweep.
2. **(Complement, fix a) Best-effort post-ingestion entity-summary refresh.**
   Extend `MemoryService._dual_write_callback` (memory_service.py L565-601)
   to call `refresh_entity_summary` for the deduplicated set of source/target
   node UUIDs across every edge in the `graphiti_core` `add_episode` result,
   logged/non-fatal on failure (mirroring `delete_entity`'s L891-901
   pattern). Also wire the same (or an equivalent) callback into the
   `add_memory_graphiti` durable-queue operation (memory_service.py
   ~L742-754), which currently has no `callback_type` at all. Needs a design
   decision on batching to avoid redundant refreshes when many edges in one
   episode share endpoints.

## 7. References

- Task 126 (done, 2026-03-28) — "Fix: Add Graphiti entity summary refresh
  after bulk edge invalidation." Shipped the Stage-1 `memory_consolidator`
  prompt instruction audited as row 5 above.
- Task 187 (done) — "Stop writing task-count/status-distribution edges in
  reconciliation pipeline." Source of the "Graphiti does not rebuild
  summaries on edge deletion — only on new edge addition" finding cited in
  §3.
- Task 1946 (done, 2026-07-01) — "Operational: rebuild_entity_summaries
  project-wide for dark_factory + know_live." Source of the backfill
  counts in §1; also independently re-confirmed the `invalid_at` rebuild
  filter is correct, ruling out a filter/compaction-bug explanation (its
  own AC1/AC2 were dropped as a false premise on this basis). This
  investigation task (1948) is its named follow-up.
- Procedural memory `ffe0f4f8-0e0a-4050-9ae5-f098cbd82068` (2026-07-01) —
  establishes the filter-is-correct / refresh-triggering-gap distinction
  and the directed-edge nuance, both cited in §3.
- This task (1948) is investigation-only: no runtime code was changed, and
  the 2026-07-01 backfill (task 1946) was **not** re-run as part of this
  investigation.
