# Design-session brief — cross-process exclusion for fused-memory identity mutations

Source: `plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md` Appendix A (Leo's ruling of 2026-11-10: the follow-up is a human design session, not an architect task). Gate task: 9198. Here `fm/` abbreviates `fused-memory/src/fused_memory/`.

**Launch.** `/spawn` an interactive session that runs `/team --fable`. Investigate, discuss candidate designs
with Leo, and answer every open question below before any PRD is written. Output: a PRD committed through
`/prd`, whose tasks `/prd` files, then gate σ resumed with that PRD's path.

**Problem.** The identity lock that serialises entity merges, renames, deletes and node deduplication is an
in-memory asyncio lock inside one graph-client object (after the S26 public-lock task,
`fm/backends/graphiti_client.py::GraphitiBackend.identity_lock`), so it only works within one process.
Operator scripts run as separate processes, each with its own graph client. The cross-graph migration
scripts (`consolidate_namespace_families`, `migrate_cross_graph_leak`, through
`fm/maintenance/cross_graph_move.py`'s seven mutators) can therefore change entity identity in a graph while
the live server's queue worker is deduplicating nodes in the same graph. Nothing excludes them. That is the
class of duplicate-node damage investigated in tasks 2073 and 2081. Separately, task 6293 found that a
script's `MemoryService.initialize()` could attach to the live write queue and drain it; S26's lifecycle task
already gives scripts a no-attach opening, and 6293's ownership question waits on this session.

**Constraints already decided.**

- Exclusion is per graph group.
- Every group a call touches is taken in sorted order, as the in-process lock does.
- Waits are bounded and the wait limit is configuration.
- A refusal reuses the in-process lock's structured `IdentityLockBusy(group_id, waited_s)` error.
- The exclusion has a named owner that can be identified and cleaned up.
- The server's queue worker holds the in-process lock across graphiti_core's LLM extraction, up to the
  120 s write budget. Any mechanism the worker must take on every write sits on the hot path.

**Open questions it must answer.**

1. Which shared medium carries the exclusion? Candidates to weigh: a SQLite lease file (the queue directory
   is resolved relative to the working directory, which is task 6293's own hazard), an OS file lock, a key
   in the Redis instance FalkorDB already runs in, or no lock at all with cross-graph moves run inside the
   server as an MCP tool.
2. Must the server pay on every write? Or can it pay only when a maintenance process has announced itself,
   for example by an operator-presence marker that the queue worker checks cheaply?
3. Is the simplest correct answer refusal rather than waiting? A script could refuse to run identity
   mutations while a live server owns the queue or the graph group, which costs the server nothing.
4. How does the cross-process exclusion order against the in-process lock, so that neither can deadlock the
   other?
5. What happens when a script crashes while holding the exclusion? Who detects that it is stale, and after
   how long?
6. What is the scope? Only `cross_graph_move`, or also scripts' raw graph writes such as the whole-graph
   purges and edge invalidations? Measured on 2026-11-10: no script calls the service's identity mutators,
   but several write raw Cypher through the graph client.
7. Does it subsume task 6293's ownership question (ownership lock, no-attach mode or fixed queue
   directory), and should 6293 be closed or re-scoped to match? 6293 waits on this gate.
8. How is it tested? Measurement must be a two-process test in which a second process's move waits or
   refuses and never interleaves.

**Read first.**

- `plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md`: D8, C6 and λ (the public lock,
  the ownership and re-entry rule, the multi-group order).
- The identity-lock finding (`fk-9804ecc4567d`) in `plans/review-all-dark_factory-20261008.json`.
- Task 6293 and the "QUEUE-DIR ATTACH" ruling in `fm/utils/store_mutation_preflight.py`.
- `fm/backends/graphiti_client.py::GraphitiBackend.identity_lock` (or `_identity_lock_for` if read before the
  public-lock task has landed).
- `fm/maintenance/cross_graph_move.py` and the two scripts.
- `fm/services/durable_queue.py`.
- Tasks 2073 and 2081; `plans/cross-graph-entity-leak-rca.md`.
