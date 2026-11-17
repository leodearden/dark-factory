# S26 — Memory service write path and graph client (PRD)

**Status:** active. Authored 2026-11-01 by the S26 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S26, agent mode, wave 2, size L, risk high,
ruling R7); critic findings folded 2026-11-02; the rulings of the 2026-11-10 sitting applied the same day
(§Rulings) and every anchor re-checked against main; fold-verified the same evening (the lead's sequencing
ruling applied). B+H: a contract section (C1–C9) and a boundary-test
sketch. Twenty-three tasks: twenty leaves (the service write path, the graph client, the label grammar, and
the service split the sitting added), the integration gate π, the human design-session gate σ and the
follow-up φ.

**Code anchors** verified against main `38a200d213` (2026-11-10). Main moves fast — cite-by-symbol;
re-locate lines at implementation time. `git diff e634cfe658..38a200d213` touches none of the plan's paths;
the five fused-memory files it changes reach none of the counted names, and the headline counts were re-run
at `38a200d213` with identical results (`memory_service.py` 11,727 lines, 25 journal guards, 87 backend
lines, 241 `_driver._get_graph` stubs in 27 files, 29 + 1 script constructions), so every count labelled
`measured at e634cfe658` below holds at `38a200d213`. Earlier anchors: the run's `6551b75aba` (2026-10-08) and the
authoring `ce0f71a384`/`b688221a04` (2026-11-02). This fold ran `git log b688221a04..e634cfe658` over every
path the plan touches; 43 had changed (`services/memory_service.py` +90/−13, `services/durable_queue.py`
+10/−126, `backends/graphiti_client.py` +12/−20, `backends/mem0_client.py`, `models/memory.py`,
`server/main.py`, `server/tools.py`, `config/schema.py`, `services/write_journal.py`, two reconciliation
modules, `tests/_fm_helpers.py` (split by task 6454), `tests/conftest.py`, and test or script files). Every
count a done-criterion or delivered check rests on was re-measured and is labelled `measured at
e634cfe658`. The 2026-11-10 discussion round measured `memory_service.py`'s method groups at `f94f0e16c7`;
the file has no commit between `f94f0e16c7` and `e634cfe658`, so those figures carry the same label. Paths
are relative to the repo root; `fused-memory/src/fused_memory/` is abbreviated `fm/`.

## Goal

The fused-memory write path becomes one typed, explicitly started pipeline: a caller's write is a frozen
value with an operation enum from the moment it is enqueued, the service that executes it is fully wired
before its queue workers start, every Layer-1 journal row is written by one context manager, and the nine
post-write identity passes live in their own class behind a public `run()`. The graph client gets one
egress (`GraphAccess`) through which every Cypher call passes with a bounded timeout, one public identity
lock that every in-process identity mutator holds, and one home for what a RELATES_TO edge's stored
properties mean. Backends become private to `MemoryService`, which exposes the small read surface
reconciliation sweeps need and is, with the collaborators it owns, the only in-process module that mutates
the graph. `memory_service.py` splits at its measured seams: the graph mutators, the memory mutators, the
write-metadata policy and the store codec move into their own modules over one journaled-operations layer,
and `MemoryService` keeps one-line public delegates. Every operator entry point builds its service through
one factory. The label grammar is composed from shared fragments.

What a user observes when this lands:

- `add_memory` with a Graphiti leg answers `stores_queued=['graphiti']`, `stores_written=['mem0']` and a
  message that says what happened; nothing reports Graphiti as written before the queue lands it.
- `fused-memory.service` logs `MemoryService started` after the journal, project registry, task backend
  and recon collaborators are wired, and a backlog row drained at startup always leaves a `write_ops`
  terminal row. A script that constructs a `MemoryService` and calls only `initialize()` (or
  `initialize_for_reads()`) starts no queue worker against the live store, even when it enqueues or
  replays: its rows persist `pending` and `get_queue_stats` counts them until the server next starts
  workers for that group.
- A failed `update_edge`, `delete_memory` or `update_memory` leaves a `write_ops` row with
  `success=0`, exactly as a failed `merge_entities` does today. A degraded `search` journals
  `success=0`; a retried `mem0_classify_and_add` fact leaves one row, not one per attempt.
- A FalkorDB query that hangs raises `TimeoutError('Graphiti <op> timed out after <budget>s')` from every
  Cypher site in the backend, not only from the four `search`-family methods.
- A `merge_entities`, `rename_entity`, `delete_entity` or `reassign_edge` in the server process that races
  a queued write to the same group waits on the identity lock; one that cannot acquire it within
  `identity_lock.acquire_timeout_seconds` returns a structured `IdentityLockBusy` refusal. A script
  running beside the live server is not covered by this lock: the cross-process design is the human
  design session gate σ (S26-3), whose PRD files that work.
- The degenerate-node sweep's delete appears in `write_ops`.
- `python -m fused_memory.maintenance.task_family_census` opens the store for reads only: no Graphiti
  startup maintenance, no queue.
- A test for a Cypher path constructs `GraphitiBackend.with_driver(..., graph_access=FakeGraphAccess({...}))`
  and never spells `_driver`, `_get_graph` or `_graph_for`; a test for an identity pass constructs an
  `EpisodeIdentityReconciler` and calls `run()`; a test for a graph mutator constructs `GraphMutations`
  with a fake backend and journal; an operator script reaches its graph through
  `GraphitiBackend.graph_access`.
- Adding a queued-write field, a tenth identity pass, a journaled maintenance tool, a relocated edge
  property, a graph mutator or a new operator script is one edit in one module.

## Background

Run `review-all-dark_factory-20261008` (record `plans/review-all-dark_factory-20261008.json`) found the
memory service and its graph client carry four structural costs: an untyped wire contract on the durable
queue with pop-then-re-parse conventions; a two-phase construction whose queue goes live before seven
collaborators exist; a backend with no injectable seam and no Cypher egress, so 51 test files and 7
operator scripts reach two private levels down; and an 11,727-line service whose identity-reconcile third
separates cleanly but is tested through ≈900 private-name accesses (rule in the fk-50cc281018fb row).
Ruling R7 (2026-10-10) chose option 1 — the `EpisodeIdentityReconciler` — and approved the rest of the
remedy as proposed. The 2026-11-10 sitting ruled every open question of this plan (§Rulings): S26-2 added
three service-split leaves (ρ, τ, υ) and a follow-up (φ), carrying R7's partition past the identity third;
S26-3 replaced the out-of-process follow-up leaf with a human design-session gate (σ). This PRD implements
R7 and those rulings without relitigating them; R22 names no invariant for S26; R23 names no weakened key
for S26.

## Findings disposition (re-verified at `e634cfe658`, 2026-11-10; unchanged at `38a200d213`)

Every key of the stream appears exactly once. Counts are `measured at e634cfe658` by AST and grep unless a
row says otherwise; the rule is stated where a count is not a plain grep. No key's cost is gone: no row is
`fixed` or `stale`.

| Key | Sev | Disposition | Note |
|---|---|---|---|
| `fk-1215aacfa626` untyped queue payload | medium | leaf α | 4 producers (`add_episode` 15 keys with `project_id`; `add_memory` 10, no `project_id`; `_dual_write_callback` 7; `replay_from_store` 6, no `project_id`); 19 distinct consumer keys (AST over the 8 consumers plus `_decode_referents`, at `b688221a04`; `memory_service.py`'s diff since touches no producer or consumer); 3 pop-then-read-back keys; `durable_queue.py::DurableWriteQueue._process_item` still parses twice ("A FRESH parse" present); no payload type anywhere |
| `fk-bdc0427c8971` operation vocabulary as bare strings | medium | leaf α | `MemoryService._execute_durable_write` tests `'mem0_add'` and `'mem0_classify_and_add'` and sends any other string to `_execute_graphiti_write` (`_execute_graphiti_write(operation, payload)` present); literals in 4 modules; no enum. `DEFAULT_IDENTITY_PAYLOAD_KEYS` is gone (task 5474), and task 6369 (done, `a555a1ecac`, merged `e7fcf2bf0a`) deleted `identity_payload_keys=`/`_identity_uuid`, so `_classify_failure` is two-way (transient / normal) |
| `fk-b30cc4fa2899` add_memory reports Graphiti as written | medium | leaf β | `stores_written.append(SourceStore.graphiti)` right after `durable_queue.enqueue` ("report as written" present); message `Memory queued for [...]`; `AddMemoryResponse` has `stores_pending_retry` (task 7397) and, from task 7650 (done, `789efc6ad6`), the Mem0-only `mem0_write_state` (`Mem0WriteState`); no queued state |
| `fk-017442f273d2` two-phase construction, queue live before wiring | medium | leaves γ0, γ | `server/main.py::run_server` calls `initialize()` before `set_write_journal`, `taskmaster=`, `set_known_projects`, `set_recon_journal`, `set_recon_ledger`, `set_event_buffer`; `DurableWriteQueue.initialize` still recovers and starts workers, and `enqueue`/`enqueue_batch`/`replay_dead` each call `_ensure_workers`; 25 `if self._write_journal` guards (79 references; 24/77 at `b688221a04` — task 7650 added `check_mem0_retry_backlog`'s); 6 setters; 9 late-assigned attributes; no `start()`. A second opening mode, `MemoryService.initialize_for_reads` (task 6239), now exists. 31 construction sites (`(=|return) *(_service\.)?MemoryService\(`): 27 under `fused-memory/scripts/`, 2 under `scripts/`, `maintenance/_utils.py::maintenance_service`, `server/main.py`; plus one injected default, `fused-memory/scripts/memory_eval_write_after_miss.py::open_memory`'s `service=MemoryService`. No script calls `set_write_journal`/`set_known_projects` |
| `fk-d37d63bf0d11` write_ops journaling without a choke point | medium | leaf δ | 19 `log_write_op` calls in 4 shapes (9 linear, 2 finally, 7 finally+guarded try, 1 try-body); 187 finally-boilerplate lines (at `b688221a04`, spans unchanged since); 9 `success = True`; `_journaled_backend_call` (62 lines) exists for Layer 2, nothing for Layer 1 |
| `fk-abc3487a3528` post-write identity pass chain | high | leaf ε | 9 passes via a nested `_run_pass` closure in `_reconcile_episode_identity` (195 lines); the write-result extraction is repeated at 8 sites; 19-method closure reachable only from the chain; no `EpisodeIdentityReconciler` |
| `fk-da5b47d0493a` god module; identity third separates | medium | leaves ε, ρ, τ, υ; follow-up φ | 11,727 lines, 6,121 of them prose (52%); `MemoryService` spans 9,375 lines with 110 methods. Groups (discussion round, AST): identity chain 22 methods / 2,972 lines plus ≈400 module-level lines; graph mutators 14 methods / 1,413 lines; memory mutators 4 / 658 and hierarchy 6 / 264; write-metadata policy 883 module-level lines (70% prose); store codec and error helpers 552 (`_serialize_temporal` … `_store_failure_diagnostics`); search 3 / 618; other reads 16 / 897. After ε, ρ, τ and υ the file is estimated at ≈4,200 lines (not measured ahead of the work); φ carries the read-path split and a prose pass over the remainder |
| `fk-343c4a42be6b` heuristic-14 measurement | medium | leaves ε, ρ, τ, υ | complexipy module total for `memory_service.py` 1,087 over 152 functions, max 84 (`_verify_episode_referents`), measured at `e634cfe658` (1,079 / 148 at `b688221a04`); the partitions this PRD performs are the identity chain (ε), the write-metadata policy and store codec (ρ), the graph mutators (τ) and the memory mutators (υ), with search reduced by η and journaling boilerplate by α/δ; π records the post-state |
| `fk-50cc281018fb` tests reach private names | high | leaves ε, ρ, τ, υ | Rule: AST over `fused-memory/tests/**/*.py`, counting `Attribute` nodes, import aliases and string constants (patch targets, including `fused_memory.services.memory_service.<name>`) that name a single-underscore member of `MemoryService` (method or `self._x` attribute) or a private module-level name of `memory_service.py`. Result: 896 accesses over 59 names in 28 files. The same rule gives 895 / 58 / 27 at `b688221a04`, where the critic's spelling of it recorded 893 / 57 / 27. Top names: `_repair_episode_referents` 137, `_verify_episode_referents` 133, `_execute_graphiti_write` 81, `_write_journal` 51 (γ), `_reconcile_episode_identity` 37, `_known_projects` 37, `_normalize_task_node_names` 36, `_dedup_episode_nodes` 35. ε moves the pass references onto `run()`; ρ turns ≈150 policy and codec references in 18 files into imports of public names; τ and υ move ≈40 mutator references in ≈8 files onto directly constructed collaborators. `Mem0Backend._get_async_qdrant` is not a service name: its `tests/test_mem0_client.py` references test the backend's own getter (task 6950 owns that seam), and ν removes only the 3 script calls |
| `fk-41eaf65cc4d0` `_verify_episode_referents` prose | medium | leaf ζ | 729-line span, 142 docstring + 386 comment-only lines (74%), 188 executable, cognitive complexity 84 (file max); leaf narration present (`leaf iota` on 18 lines in `memory_service.py`, 12 of them in code ε moves — 8 in the chain, 2 in the `referent_*_counts` properties, 2 module-level beside `REFERENT_REPAIR_OUTCOMES`; `Resolved decision 7` ×2) |
| `fk-eb0d5fea9a52` `search` does seven things | medium | leaf η | 453 lines, cognitive 81 (file second); inline pin block 220 lines; imports `_ANCHOR_SCROLL_LIMIT` and `_MAX_ANCHOR_TOPICS`; `resolve_topic_anchor_enabled(memory_service)` takes the whole service and is evaluated on every call (the knob is green-tier hot-reloadable, `config/reload.py`) |
| `fk-9430031679c4` no injectable graph seam | medium | leaf θ (scripts: θ2; tests: ι, κ) | partly fixed by task 6233 (`d064e0181d`): `GraphitiBackend.with_driver` and `conftest::make_graph_driver` / `make_wired_backend` exist. Still present: `_graph_for` → `driver._get_graph` (graphiti_core private); tests hold 241 `_driver._get_graph` stubs (27 files), 65 `._graph_for =` (5 files), 4 `patch.object(…,'_graph_for')`, 23 `._driver =` (10 files) — 51 test files reach a graph private (243 / 65 / 4 / 25 in 50 files at `b688221a04`); 7 operator scripts call `._graph_for`/`._require_falkor_client`/`_paged_ro_query` (≈20 code sites); task 7781 (pending) is the unfinished test sweep |
| `fk-90f6fbb5daa0` no Cypher egress | high | leaf θ | timeout half fixed by task 5699 (`ed8c8fce34`, `26cbd2b884`): `search`, `search_nodes`, `retrieve_episodes`, `get_episode_by_uuid` raise a named `TimeoutError` via `_timed_read`; no read returns `[]` on timeout. Still present: 47 raw Cypher lines — 22 `graph.query`, 23 `graph.ro_query` (task 8246 turned the dup-uuid detect read into `ro_query`), 2 inline `_graph_for(...)` — 2 of them under `asyncio.wait_for` (45 unbudgeted); no socket timeout on the FalkorDB connection |
| `fk-9804ecc4567d` identity lock honoured at two sites | high | leaf λ (in-process); gate σ (out-of-process, human design session) | `GraphitiBackend._identity_lock_for` private (an `asyncio.Lock` per group in the per-instance dict `self._identity_locks`); acquired only by `_execute_graphiti_write` and `ensure_entity_node`; `merge_entities`, `rename_entity`, `delete_entity`, `reassign_edge` and `cross_graph_move` acquire nothing; 5 lines in `fm/` say callers "MUST hold" it. The finding's "out-of-process maintenance" half: the only importers of `cross_graph_move` are two scripts, each a separate process with its own backend, so an in-process lock never contends for them; σ carries that half in its `x_finding_key` |
| `fk-6e50b22357f2` backends public, reach-throughs | medium | leaf μ (scripts: ν) | 8 direct + 6 alias-form modules in `fm/` reach 20 backend methods (1 of 21 has a service counterpart; re-measured, unchanged since `b688221a04`); 16 scripts (15 under `fused-memory/scripts/` plus `scripts/sweep_uuid_prefixes.py`); `cross_graph_move` 16 `_graph_for` + 5 `_require_falkor_client`; 3 `_get_async_qdrant` script calls; the degenerate sweep's delete is unjournaled and unlocked. Raw maintenance writes with no service counterpart: 1 direct `mem0.update` (`tag_cgl_eta_rehome_scope.py`; the earlier "×5" counted five scripts that call the service's journaled `update_memory`), 3 `_get_async_qdrant` writers, the graph-key purges in `consolidate_namespace_families`/`purge_knowlive_namespace`, and `cross_graph_move`'s 7 mutators |
| `fk-265caf7191a5` four SET-list copies | medium | leaf ξ | 4 copies still present (`reassign_edge` ×2, `redirect_node_edges` ×2); `test_reassign_edge.py::_COPIED_PROPS` + 2 substring pins; a fifth list, `cross_graph_move._CARRIED_EDGE_PROPERTIES` (task 6616), exists; task 5612 (pending, branch 54 ahead / 9,138 behind main, last commit 2026-10-12) owned the in-file consolidation — cancelled into ξ (S26-6 ruled B) |
| `fk-62af616b8470` edge lifecycle as two nullable timestamps | high | leaf ξ | partly fixed by task 4714 (`8a8083ff98`, Leo ruling A): the active-edge filter is `invalid_at` only, named once as `graphiti_client._ACTIVE_EDGE_PREDICATE` (interpolated at 6 f-string Cypher lines) with a Python mirror `memory_service._edge_is_retired` (2 sites). Still present: 4 inline `getattr(edge, 'invalid_at')` spellings in the identity passes, two homes for one predicate, no typed state. The finding's "no transition table" half is reduced to nil cost: ruling A, plus every `invalid_at` write funnels through `GraphitiBackend.update_edge`'s single `SET e.invalid_at` |
| `fk-352d7b3fcad4` label regexes re-decide fragments | medium | leaf ο | `canonical_labels.py` unchanged since `b688221a04`: 5 compiled patterns; only `_TASK_WORD` shared (task 4850 landed it, which also closed the fixes tasks 4235 and 4479 describe); qualifier class spelled 3×, separator alternation 2×, digit capture 4×, right edge 2×; 144 code / 698 prose lines |

## Pre-conditions for activating

- No upstream stream (G4: S26 has no seam dependencies). Nothing is held.
- Substrate present at `e634cfe658` (G3, verified): `GraphitiBackend.with_driver` and
  `conftest::make_backend` / `make_graph_driver` / `make_wired_backend` (task 6233);
  `backends/timeout_budget.py::await_within`; `shared/src/shared/streak.py::KeyedStreaks`;
  `models/memory.py::AddMemoryResponse` with `stores_pending_retry` and `mem0_write_state` (task 7650);
  `MemoryService.initialize_for_reads` (task 6239); `maintenance/_utils.py::maintenance_service`;
  `models/scope.py::build_known_projects_map`; `WriteJournal.log_write_op` upserts on `id` (task 3582) and
  never raises; `falkordb.asyncio.FalkorDB.__init__` accepts `socket_timeout` and
  `graphiti_core.driver.falkordb_driver.FalkorDriver.__init__` accepts `falkor_db=` (signatures read from
  the root `.venv`); `_MultiTenantFalkorDriver.clone` and graphiti_core's `FalkorDriver.clone` reuse
  `self.client`, so the injected connection's `socket_timeout` survives clone.
- Tasks 6362, 6369 and 7650, which α and θ waited on in the 2026-11-02 draft, are done; α has no
  out-of-batch dependency left.
- The live queue store `data/queue/write_queue.db` holds 0 pending and 0 in-flight rows (read-only sqlite,
  2026-11-10: `add_memory_graphiti` 40,873 completed, `mem0_add` 1,511, `mem0_classify_and_add` 16,
  `add_episode` 1). Legacy-row fixtures are therefore synthesised (boundary row 3).
- Live tasks this PRD depends on, absorbs or re-points (contract §8 step 1 found no task carrying any S26
  key, re-run 2026-11-10; step 2 found these): see §Live tasks.

## Sketch of approach

Three chains, a service split and three gates. The service chain (α→β→γ0→γ→δ→λ→ε→{ζ, η}→ξ→μ→ν) edits
`memory_service.py` in series and continues into the split (ν→ρ→τ→υ). The graph chain (θ→θ2, θ→λ→…→ξ→μ)
edits `graphiti_client.py` in series and meets the service chain at γ0/γ and λ. The test sweeps (ι→κ) follow
θ2 and ε. The grammar leaf ο is independent. σ, the human design session for the out-of-process half, waits
only on λ. π is the integration gate; φ follows it. The full DAG is in §Decomposition plan.

**α — typed queued writes.** `fm/services/queued_writes.py` (new leaf module) defines `QueueOperation`,
`QueueCallback`, `QueueDecodeError` and one frozen value type per operation (C1). `DurableWriteQueue.enqueue`
takes a `QueuedWrite`, stores `operation=value.operation.value`, `payload=value.to_json()`; `_process_item`
decodes once with `decode_queued_write` and hands the same immutable value to the executor and to the
callback. The FRESH parse, the four producer dict literals, the eight pops, the three read-backs and the
dispatcher fall-through are deleted. A row that cannot be decoded raises a `QueueDecodeError`, which
`_classify_failure` makes permanent by type, so the row dead-letters on attempt 1 with its raw payload.
`dead_letter_escalator._REPORTED_TO_CALLER` keys on the enum.

**β — honest add_memory response.** `AddMemoryResponse.stores_queued` (C2). The Graphiti leg appends
there; `stores_written` means landed; the message is rendered from the three lists. Task 7650's
`mem0_write_state` stays the one Mem0 reading; β adds no second predicate (D3).

**γ0 — one operator factory (S26-4: A).** Every operator script builds its service through
`maintenance/_utils.py::maintenance_service(config_path=None, *, config=None, open_as=ServiceOpen.FULL)`;
a script that rewrites its config before construction passes `config=`; the two read-only eval scripts
open `ServiceOpen.READS` (today's `initialize_for_reads()`), `scripts/check_consolidation_closure.py`
opens `ServiceOpen.NONE`, and `task_family_census`'s CLI opens `READS` (absorbs task 6835). γ0 changes no
behaviour except the shared close and the census's read-only open: it gives γ one site to change.

**γ — constructed, then started.** `MemoryService(config, *, write_journal, known_projects, graphiti=None,
mem0=None, taskmaster=None, recon_journal=None, recon_ledger=None, planned_registry=None)` constructs its
`DurableWriteQueue`; `initialize()` opens backends and the queue store; `initialize_for_reads()` keeps its
read-only opening; `start()` runs `DurableWriteQueue.start()` (recover in-flight, then start workers) and is
the last lifecycle call before serving (C3). `set_event_buffer` is the one pre-start setter. Until
`start()` the queue persists and returns ids but runs no worker. `maintenance_service` opens the live
journal and project registry and never starts. The other setters, the 25 None-guards and the bound-method
"call-time resolution" docstrings go. `conftest::build_memory_service` is the one test factory; 77 test
files move onto it.

**δ — one Layer-1 choke point.** `services/journaled_ops.py::journaled_op` (new module; S26-5 ruled B, its
home set by S26-2) is an async context manager. It mints a `write_op_id` only when none is carried, records
success, failure (exception or an explicit `op.fail`) and cancellation, and journals in its exit.
`MemoryService._journaled_op` delegates to it. The 19 call sites state only params and result.

**ε — EpisodeIdentityReconciler.** `fm/services/episode_identity.py` holds the nine passes, their closure
(19 methods), `ReconcileStats`, `ReferentRepair`, `ReferentRepairStats`, the fact predicates and the
counters behind `run()` (C4). `MemoryService` owns one instance; `_execute_graphiti_write` calls it. The
reconciler mutates only through the backend it was built with, under the caller's lock, and never calls a
service mutator. The test files that reach the passes construct the reconciler. Absorbs task 6601
(`RepairOutcome` StrEnum).

**ζ — the verify pass in its own right.** `_verify_episode_referents` becomes four methods of the
reconciler, its leaf narration moves to this PRD (§Rationale rehomed), its placement constraint becomes the
call order in `run()`.

**η — search in three steps.** `topic_anchor.apply_topic_anchor` is a pure async function owning the pin's
commit/fail-open protocol; `search` is `_fan_out` → `_fuse_and_filter` → `apply_topic_anchor` →
`_journal_search`. η also deletes the `leaf iota` narration left in `memory_service.py`.

**θ — GraphAccess.** `fm/backends/graph_access.py` defines the `GraphHandle`/`GraphAccess` protocols and
`FalkorGraphAccess` (C5). It is the one place that reaches `driver._get_graph` and the one place that
applies a budget to raw Cypher. `GraphitiBackend` takes a `graph_access` at construction (`with_driver`
keyword, `initialize()` default) and exposes it read-only as `GraphitiBackend.graph_access`. The 47 raw
call lines go through it. `_graph_for` survives only as a one-line shim over `graph_access.graph` until θ2.
`_paged_ro_query` moves to `graph_access.paged_ro_query`, which also removes `falkor_read_only.py`'s upward
import. `cross_graph_move` keeps its `graphiti` parameter and resolves graphs through
`graphiti.graph_access`. The FalkorDB connection is built with a socket timeout.

**θ2 — operator scripts on the public provider.** The 7 scripts that call `_graph_for`,
`_require_falkor_client` or `_paged_ro_query` and their 5 test files move onto `GraphitiBackend.graph_access`;
the shim and the scripts' `# noqa: SLF001` go.

**ι, κ — tests on the public seam.** The test files stubbing `_driver._get_graph`, `_graph_for`, `_driver`,
`_MultiTenantFalkorDriver` or `_paged_ro_query` move onto `make_backend(cfg, graph_access=FakeGraphAccess(...))`
or `make_graph_driver`. Absorbs task 7781.

**λ — one public identity lock.** `GraphitiBackend.identity_lock(group_id, *, timeout)` (C6) records its
holder in a `ContextVar` (S26-7: C), refuses re-entry, and is held by `_execute_graphiti_write`,
`ensure_entity_node`, `merge_entities`, `rename_entity`, `delete_entity`, `reassign_edge` and
`cross_graph_move`'s seven public mutators. A call touching several groups takes every group's lock in
`sorted()` order. MCP and reconciliation mutators wait at most `identity_lock.acquire_timeout_seconds`; the
queue worker is the one unbounded holder. `EpisodeIdentityReconciler.run()` refuses to run unless the
calling task holds the lock.

**σ — the out-of-process half, as a human design session (S26-3: D, ruled as a session).** A deterministic,
always-escalating gate task (the pure-gate form, `docs/task-authoring.md` §5) that, once λ is done, files a
born-at-L2 escalation asking Leo to run an interactive `/spawn` session under `/team --fable`: investigate,
discuss candidate designs with him, answer every open question of the committed brief (Appendix A, to be
committed as `plans/fm-cross-process-identity-exclusion-session-brief.md`) before any PRD is written, then
commit a PRD through `/prd` and resume the gate with its path. Only that PRD's tasks are dispatched. It
absorbs task 6293's design question (queue ownership lock vs no-attach mode vs fixed queue directory), so
two cross-process ownership designs for the same stores cannot appear. It is not an architect task and
builds nothing itself.

**μ, ν — one graph mutator.** `MemoryService._graphiti`/`_mem0` are private; the service exposes the
sweeps' read surface and a journaled `remove_edges` (C7); the degenerate sweep deletes through
`delete_entity`; `GraphitiBackend.delete_entity_node` becomes the private primitive
`_detach_delete_entity_node`; maintenance mutations without a service counterpart are wrapped in
`journaled_ops.journaled_op` (S26-5: B); scripts construct the backends they read directly (ν, S26-1: A).

**ξ — what a RELATES_TO edge's stored properties mean.** `fm/backends/relates_to_schema.py` (C8) is the
one home of the active-edge predicate (Cypher fragment and Python predicate derived from one marker list)
and of the relocation rule; `graphiti_client`, `cross_graph_move` and the reconciler import it; the test
tuple and substring pins become a relocation parity test. Absorbs task 5612's two Leo-widened items
(S26-6: B).

**ο — one label grammar.** `canonical_labels` composes its five patterns from named fragments; the fixes
task 6826 describes become one-fragment edits.

**ρ, τ, υ — the service split (S26-2, ruled by Leo).** After μ and ν, three behaviour-preserving moves
take `memory_service.py` from ≈7,800 lines to ≈4,200 (estimates from the discussion round's group
measurements). ρ moves the write-metadata policy into `services/memory_metadata_policy.py` and the store
codec and error helpers into `services/store_records.py`, both as public functions that take their
collaborators as arguments, as today. τ completes the
journaled-operations layer (`journaled_backend_call` and the `EventSink` join δ's `journaled_op`) and
extracts the graph mutators into `services/graph_mutations.py::GraphMutations`. υ extracts the memory
mutators and the parent-child hierarchy into `services/memory_mutations.py::MemoryMutations`. `MemoryService`
builds both collaborators in its constructor and keeps one-line public delegates, so no caller changes (C9).

**π — integration gate.** One end-to-end test over a real SQLite queue, a fake `GraphAccess` and a real
`WriteJournal`: enqueue → typed executor → reconciler under the lock → journal rows → honest response;
plus the recorded heuristic-14/complexity re-measurement of every module the plan created or shrank.

**φ — follow-up: the read path and a prose pass.** Filed now, dispatched after π: re-measure
`memory_service.py` with this plan's changes in place, extract the read path (search and the other reads)
into a reader collaborator if it still reads as a clean cut, and run a Comments-stance prose pass over what
remains.

## Resolved design decisions

- **D1 — the queue imports the value types; the executor dispatches on type.** `DurableWriteQueue` stays a
  generic durable store of `(group_id, operation, payload)` rows, but its public `enqueue` takes a
  `QueuedWrite` and its `_process_item` decodes once (seam table: "durable_queue imports the value types").
  Decoding inside the queue is what deletes the FRESH parse: the executor and the callback receive the
  same frozen value, so there is nothing to pop and nothing to read back. `MemoryService._execute_durable_write`
  becomes a table `{type: executor}`; an unregistered type is a programming error raised at dispatch. A row
  that cannot be decoded (an unknown stored operation string, or a body `from_json` rejects) raises a
  `QueueDecodeError`; `DurableWriteQueue._classify_failure` returns `('permanent', 1)` for
  `isinstance(exc, QueueDecodeError)` — decided by type, never by message — so the row dead-letters on its
  first attempt (INV-11: today's fall-through runs it as a Graphiti write). Task 6369 (done) left the
  classifier two-way (transient / normal), so without this branch an undecodable row would retry
  `max_attempts` times. The `DeadLetterEvent` for an undecodable row carries the raw payload text, because
  `MemoryService._report_queue_dead_letter` and `dead_letter_escalator` cannot read typed fields off a row
  that has none. `QueueOperation` is the queue's vocabulary only. Write-journal `operation` strings
  (`reconciliation/stage_stats.py::_OP_TO_STAT` keys `'add_memory'`, `'delete_memory'`, `'update_edge'`, …)
  are a separate vocabulary and are not folded into it. Rejected: typing only in `MemoryService` with the
  queue left on dicts — the double parse and the pop convention would survive, which is the cost the
  finding names.
- **D2 — one envelope, per-operation bodies, legacy rows decode, the wire keeps its keys.** Every value
  carries a `WriteEnvelope` (`project_id`, `group_id`, `agent_id`, `session_id`, `causation_id`,
  `write_op_id`); the body fields are the operation's own. `to_json` writes today's flat key names
  (`_causation_id`, `_write_op_id`, `project_id`, `referents`, …), so a pre-α build (a rollback) can drain
  post-α rows and `server/tools.py::get_dead_letters`, which returns the raw payload, keeps its shape.
  `from_json` gives every field a legacy default so a row written before α decodes: `project_id` absent on
  `add_memory_graphiti` and `replay_from_store` rows derives from `group_id` by the rule
  `models/scope.py::Scope.graphiti_group_id` (`group_id` = `project_id`); the pre-task-3561 `uuid` key is
  ignored; an absent `referents` key decodes to `DecodedReferents((), 'none', None)`.
  `QueueOperation.MEM0_ADD` is kept as a decode-only member mapped to `_execute_mem0_write` (1,511
  completed rows, no producer; retiring it is tactical, t2).
- **D3 — `stores_queued` is a third list, not a flag; no second predicate.** `AddMemoryResponse` gains
  `stores_queued: list[SourceStore]`; `stores_written` keeps its meaning (landed, ids known) and
  `stores_pending_retry` keeps task 7397's. The MCP tool passes the model through unchanged
  (`server/tools.py::add_memory` returns `result.model_dump()`), so the wire shape is the model. The message
  is derived from the three lists by one renderer; no caller parses it (heuristic 12). `add_memory`'s
  `write_ops.result_summary` gains `stores_queued`; `stage_stats._count_graphiti_queued` reads it, or for
  rows journaled before β the legacy `stores` list containing `'graphiti'`. Task 7650 (done) shipped
  `Mem0WriteState` and `AddMemoryResponse.mem0_write_state`, the one Mem0 landed / pending-retry /
  not-landed reading that four src modules and one script route through; it did not ship the per-store
  `outcome_for` the 2026-11-02 draft assumed. β leaves `mem0_write_state` as it is and adds no per-store
  predicate: measured at `e634cfe658`, no code reads a Graphiti write state off the model. The readers of
  the store lists are the message renderer, `model_dump`, `stage_stats` through the journal, 7650's Mem0
  consumers, and `maintenance/seed_autopilot_video_triage_guardrails.py`, whose report copies
  `stores_written` per seed write; β gives that report the queued list too, so it keeps saying where each
  write went. The Graphiti leg's state is membership in `stores_queued`. A per-store predicate is added when
  its first consumer appears (t4).
- **D4 — required collaborators at construction; optional ones explicit; one `start()`.** `write_journal`
  and `known_projects` are required. Every path has them: the server builds them in `run_server`; operator
  paths get them from `maintenance_service` (γ0/γ, S26-4: A). `main.py` can construct both before the
  service — the journal needs only `config.reconciliation.data_dir`. `taskmaster`, `recon_journal`,
  `recon_ledger` and `planned_registry` are constructor keywords defaulting to `None`. `recon_journal` and
  `recon_ledger` are built from `config.reconciliation.data_dir` alone (measured in `run_server`), so they
  need no setter; a `None` `planned_registry` is opened by `initialize()` (read-write) or
  `initialize_for_reads()` (read-only), exactly as each does today when none is wired. `None` taskmaster is a
  real state (`config.taskmaster` may be absent) and is handled once: `_check_dependency_direction` logs
  `dependency_direction_check_skipped` with the reason the first time it runs without a backend (INV-11),
  instead of returning `[]` silently. `MemoryService.__init__` constructs the `DurableWriteQueue` object
  (pure; `initialize()` opens its store), so `durable_queue.get_stats` exists before `run_server` builds the
  `EventBuffer` whose `queue_stats_fn` it is. For the same reason the event buffer cannot be a constructor
  argument. It is wired by the one remaining pre-start setter, `set_event_buffer`, which raises
  `RuntimeError('event_buffer wired after start()')` once `start()` has run. `initialize()` opens owned
  resources (backends, `PlannedEpisodeRegistry`, overreach shadow, the queue store's schema);
  `initialize_for_reads()` (task 6239) keeps its narrower opening and never opens the queue store. `start()`
  is new on both `MemoryService` and `DurableWriteQueue`, requires a prior `initialize()`, and is the only
  call that recovers in-flight rows or starts workers. `DurableWriteQueue` holds `_started`, and
  `_ensure_workers` is a no-op until `start()`. Before `start()`, `enqueue`, `enqueue_batch` and
  `replay_dead` persist and return ids. `start()` recovers in-flight rows, then starts workers for every
  pending group. Consequence: a script's queued Graphiti leg lands when the server next starts workers for
  that group. That is a hold surfaced by `get_queue_stats`' oldest-pending age (INV-7), and
  `maintenance_service`'s docstring says so. `main.py` wires everything, then `start()`s, keeping today's
  order `_recover_mem0_intents_at_startup` → `prune_every_table_at_startup` before `start()` and the Mem0
  retry loop after it. `set_write_journal`, `set_known_projects`, `set_recon_journal`, `set_recon_ledger` and
  `set_planned_registry` (no production caller; 2 test files call it) are deleted; those tests and task
  8660's three dotted-path patches move onto the `planned_registry=` keyword. `set_known_projects`'s
  propagation to `_mem0_update_storm_escalator` moves into `__init__`. The factory's three opening modes are
  one enum, `ServiceOpen.FULL | READS | NONE` (heuristic 12): `FULL` awaits `journal.initialize()` then
  `service.initialize()`; `READS` awaits `journal.initialize()` then `service.initialize_for_reads()`, so a
  read-only script's searches journal their `kind='read'` rows like the server's (S26-4 ruled the live
  journal for every operator path); `NONE` constructs the journal and the service and awaits neither (the
  lean read `check_consolidation_closure` needs, no FalkorDB, embedder or `OPENAI_API_KEY`). Rejected:
  keeping setters and merely moving `initialize()` later — the None-guards would remain because the type
  would still be `X | None`.
- **D5 — `build_memory_service` is the one test factory, backends injected through it.** Measured at
  `e634cfe658`: 78 test files assign `.graphiti = …`/`.mem0 = …` on some receiver (67 on a service-named
  one); 25 call `.set_write_journal(`/`.set_known_projects(`, 33 call any of the six setters; 77 construct a
  service or call a setter (76 / 65 / 25 / 32 / 76 at `b688221a04` by the same regexes). Once backends are
  private (μ) those assignments are private reaches. So γ makes `graphiti=`/`mem0=` constructor keywords
  (None → built from config, as today) and gives tests one factory in `fused-memory/tests/conftest.py`:
  `build_memory_service(config, *, graphiti=None, mem0=None, write_journal=None, known_projects=None,
  taskmaster=None, planned_registry=None, …)`. Its defaults are spec'd doubles (`AsyncMock(spec=WriteJournal)`
  for the journal), and its `graphiti=None` builds the same MagicMock backend `make_backend` does. Tests
  that assert journaling pass a real `WriteJournal` over `tmp_path` (`test_journaling_integration.py` is the
  precedent). The sweep across 77 files is one uniform substitution; it crosses the overlay's 15-file review
  trigger deliberately, as S25 δ/ε did.
- **D6 — `journaled_op` is a context manager, uniform for all 19 sites, and journals failures.** Shape and
  home (S26-5: B; S26-2 put the shared layer in its own module, never inside the 1,618-line
  `write_journal.py`): `services/journaled_ops.py::journaled_op(journal, operation, params, *, source,
  kind='write', provenance=None, project_id, agent_id=None, session_id=None, causation_id=None,
  write_op_id=None)`; `MemoryService._journaled_op(operation, params, **kw)` delegates with
  `self._write_journal`. Use: `async with self._journaled_op('update_edge', params, causation_id=…) as op: … ;
  op.result_summary = …`. It mints `write_op_id` only when none is passed. Queue-executor sites pass
  `value.envelope.write_op_id`: the carried id is what task 3582's terminal write-back joins on, and
  re-minting it breaks the join. `_execute_mem0_classify_and_add` gets a per-fact id minted at enqueue in
  `QueuedMem0Fact.envelope`, so a retried fact upserts one row (`log_write_op` upserts on `id`) instead of
  writing one `success=0` row per attempt. Consequence: `mem0_classify_and_add` rows, which today carry no
  `_write_op_id`, now reach `DurableWriteQueue._notify_terminal` → `WriteJournal.record_terminal_outcome`
  and the dead-letter `write_op_id` path, both of which they skip today. `record_terminal_outcome` is an
  upsert, so a fact row that dead-letters before its executor ever logged leaves an `operation`-NULL stub
  row under that id. Row 8 asserts that shape, and δ rewrites `durable_queue.py`'s prose that says these
  rows carry no `_write_op_id`. The yielded `op` supports `op.fail(reason)` for a non-exception failure:
  `search` journals `kind='read'` with `success=0` when degraded, as it does today. The exit journals on any
  `BaseException`, `asyncio.CancelledError` included (as `add_episode` and `_execute_mem0_write` do today),
  and re-raises. It writes exactly one `write_ops` row; the 10 methods that today journal only on success
  gain failure rows (an operator querying `write_ops` for failures sees every operation kind). It never
  wraps `log_write_op` in `try/except Exception`: `WriteJournal.log_write_op`'s contract is "never raises",
  and the seven wrappers defended against nothing. `_journaled_backend_call` (Layer 2) is unchanged by δ and
  nests inside; τ moves it beside `journaled_op` as `journaled_backend_call`. `add_episode`'s enqueue-time
  row keeps task 3582's terminal write-back; the context manager carries the `write_op_id` into the queued
  value's envelope. The free function is also what maintenance mutators wrap their raw writes in (D11).
- **D7 — the reconciler is a class with injected collaborators; passes are methods; `run()` is the only
  entry.** Measured: the nineteen-method closure reads exactly seven service attributes (`graphiti`,
  `taskmaster`, `_write_journal`, `_known_projects`, `_referent_finding_counts`, `_referent_repair_counts`,
  `_referent_repair_streaks`). The last three move inside; the first four are constructor arguments. The
  module imports downward only (`backends.graphiti_client` for types, `services.write_journal`,
  `utils.referent_*`, `utils.canonical_labels`, `middleware.dependency_direction_check`,
  `middleware.referent_repair_storm_escalator`, `shared.streak`). It never imports or names
  `memory_service`/`MemoryService`, nor `graph_mutations`/`memory_mutations` (heuristic 13). It mutates only
  through the backend it was constructed with, under the caller's `identity_lock`, and never calls a service
  mutator — on `MemoryService` or, after τ, on `GraphMutations` — because those take the lock (D8). The
  write-result extraction (`getattr(result, 'edges') or getattr(result, 'entity_edges')`, 8 sites) becomes
  one `EpisodeWriteResult.from_graphiti(result)` value built once in `run()`. The pass order is `run()`'s
  body, not a comment. `_referent_source_counts` is incremented in `_execute_graphiti_write` before
  `add_episode`, outside the chain, and is INV-4's denominator. The caller keeps incrementing it at the same
  point through `EpisodeIdentityReconciler.record_referent_source(source)`, so every write counts, including
  those whose Graphiti add fails before `run()`, and the counter has one home.
  `MemoryService.referent_source_counts`/`referent_finding_counts`/`referent_repair_counts` have zero
  production callers (grep over `fm/`: prose mentions only) and move to `EpisodeIdentityReconciler.counts`.
  Rejected (R7 option 2): keeping the passes as methods.
- **D8 — `run()` requires the lock and says so structurally; the lock knows its holder (S26-7: C).** ε lands
  after λ, so `run()` checks `graphiti.identity_lock_held(group_id)` from its first commit and raises
  `IdentityLockNotHeld` (C4) naming the group (heuristic 10, redundant enforcement at use); ε owns that check
  and boundary row 10. The lock stays a per-group `asyncio.Lock` registry on the backend (task 2198's
  design). `Lock.locked()` is True when any task holds it, so ownership is recorded in a `ContextVar` of
  held group ids that `identity_lock` sets and `identity_lock_held` reads. A re-entry by the holding task
  raises `IdentityLockReentry` instead of deadlocking, so the deadlock class of D7 is a test failure, not a
  hang. asyncio copies the context into a task created while the lock is held, so such a child reads as a
  holder: correct for work the holder awaits (graphiti_core's gathered coroutines inside `add_episode`),
  where a lock-taking call would otherwise deadlock on its own parent and now raises instead; a detached
  task that outlives the hold is never spawned under the lock (λ's docstring states it; row 10c pins the
  awaited case). The public context manager adds the bounded acquire `ensure_entity_node` already performs
  and a structured `IdentityLockBusy(group_id, waited_s)` refusal. MCP and reconciliation-stage mutators
  pass `timeout=config.identity_lock.acquire_timeout_seconds`. That is a new field whose default lives in
  the schema only (R22 item 5), and `entity_mint.lock_timeout_seconds` folds into it (OQ-3). The queue worker
  passes `timeout=None` and is documented as the one unbounded holder: it holds the lock across
  graphiti_core's LLM extraction (write budget 120 s), so other mutators will routinely wait on it. A call
  that touches several groups acquires every group it touches in `sorted()` order: a cross-graph move takes
  source and target, and `cross_graph_move.recreate_subgraph_relationships(graphiti, specs)` takes the
  groups of every spec it is handed.
  Out-of-process maintenance (`cross_graph_move`, driven by two scripts, each a separate process with its own
  backend) holds the same in-process lock, which never contends with the server. The cross-process design is
  the human design session σ (S26-3); `IdentityLockBusy` is one of its fixed constraints.
- **D9 — `GraphAccess` is the Cypher egress; `with_driver` stays the driver seam.** Task 6233 made the
  driver injectable; the remaining cost is that every raw Cypher call still resolves its graph through a
  graphiti_core private and carries its own (usually absent) timeout. `GraphAccess.graph(group_id)` returns
  a `GraphHandle` whose `query`/`ro_query` are budgeted by `await_within` with the backend's read/write
  budgets (`config.queue.backend_read_timeout_seconds` / `config.queue.backend_write_timeout_seconds`, read
  in `GraphitiBackend.__init__` and already used by `_timed_read`/`_timed_write`) and whose
  `ro_query_compact` is the byte-exact `--compact` read `cross_graph_move` needs (its 5
  `_require_falkor_client` calls go). `FalkorGraphAccess` wraps a `FalkorDriver`; `GraphitiBackend.initialize`
  builds `_MultiTenantFalkorDriver(falkor_db=FalkorDB(host, port, password,
  socket_timeout=cfg.falkordb.socket_timeout_seconds))` — both signatures verified, and `clone` reuses the
  connection — so the connection has a socket bound for the first time. The new field's default lives in
  `FalkorDBProviderConfig` only (R22 item 5 discipline). `GraphitiBackend.graph_access` is a read-only
  property returning the provider passed to `with_driver` or built in `initialize()`. It is how
  maintenance code and operator scripts reach a graph (θ2), and the backend-to-provider link C5 otherwise
  lacks. graphiti_core's own calls (`add_episode`, `search`) keep `_timed_read`/`_timed_write`. Tests use
  `FakeGraphAccess(graphs: Mapping[str, GraphHandle])` from conftest; `make_graph_driver` remains for tests
  that exercise graphiti_core through the driver.
- **D10 — one home for the edge's stored meaning.** `relates_to_schema.py` declares the retirement marker
  columns once (`RETIREMENT_MARKERS = ('invalid_at',)` per Leo's ruling A on esc-4714-9; `expired_at` is the
  restored-edge signature and never an activity test), derives `ACTIVE_EDGE_PREDICATE` (Cypher, parameter
  alias `e`) and `edge_is_active(edge)` (Python) from it, and declares the relocation rule and
  `relocation_set_clause(*, new, old, overrides)`. OQ-1 is ruled (b) (lead, 2026-11-02, on the live probe
  against FalkorDB 4.18.0). The in-graph rule (`reassign_edge`, `redirect_node_edges`) is `SET new +=
  properties(old)`, which keeps every stored type — String, List, Integer, Float, Boolean and Vectorf32 —
  and leaves the relocated edge in the vector index. It is followed by re-asserting
  `RELOCATION_REASSERTED_PROPERTIES`: `source_uuid`, `target_uuid` (graphiti_core's FalkorDB save flattens
  them onto every RELATES_TO edge, `graphiti_core/edges.py::EntityEdge.save`) and `source_node_uuid`,
  `target_node_uuid` (fused-memory's own stamps, `GraphitiBackend.redirect_node_edges`, `reassign_edge`), all
  from the MATCHed new endpoints. Per site, `uuid`, `superseded_edge_uuid` and `reassigned_from_node_uuid`
  come in as `overrides`. `+=` alone would copy the old endpoint ids onto a relocated edge, which today's SET
  lists do not. `fact_embedding` is NOT in the in-graph re-assert list: `+=` already carries Vectorf32, and
  `vecf32(old.fact_embedding)` on a Vectorf32 errors (`Type mismatch: expected List or Null but was
  Vectorf32`). `cross_graph_move` copies between two graphs, so `properties(old)` cannot be read in its write
  query. It uses task 6680's parameter-map form for every property except the embedding: read
  `properties(e)` in the source graph, strip `fact_embedding` from the map, and write `SET r += $props` in the
  target. The embedding is then re-asserted from the byte-exact `--compact` transport the module already
  owns: `_embedding_set_clause('r.fact_embedding', <compact reply>)`, i.e. `_read_compact_vector` →
  `parse_compact_vector_reply` → `format_vecf32_literal`. Lead ruling, 2026-11-02, on critic3 N1 option (a).
  The re-assert is mandatory: a vector read through the client arrives as a List and, left in the map, would
  store the edge List-typed and invisible to the vector index (probe tests 10/H). It must come from the
  compact tokens, not `vecf32($props.fact_embedding)`, because the client's decoded path calls `float()` on
  each component (`falkordb/query_result.py::__parse_vectorf32`). The module docstring's claim that this path
  is lossy (`plans/cross-graph-entity-leak-rca.md` §6 Phase 1) stands, and node `name_embedding` and edge
  `fact_embedding` keep one transport (heuristic 11). Legacy edges whose `fact_embedding` is already a plain
  List are carried as a List by the in-graph rule (faithful, and as unindexed as their source).
  Cross-graph, `_read_compact_vector` refuses such a wrong-typed column with its existing `ValueError`, as
  today. Upgrading legacy embeddings is out of scope. `graphiti_client` deletes `_ACTIVE_EDGE_PREDICATE`,
  `memory_service` deletes `_edge_is_retired`, the reconciler's 4 inline spellings import the predicate,
  `cross_graph_move` deletes `_CARRIED_EDGE_PROPERTIES`, and the `reassign_edge` comment saying a property
  copy is needed "to preserve the vecf32 type" goes (false on 4.18.0). ξ also carries task 5612's two
  Leo-widened items (S26-6: B). First, one rule for `superseded_edge_uuid` on a re-redirected edge: keep the
  original when `old.superseded_edge_uuid` is already set, applied in `redirect_node_edges` and
  `reassign_edge` alike (task 6823, which lands before θ, already carries the stamp in `reassign_edge`).
  Second, `redirect_node_edges`' docstring and the active-predicate comment state exactly what `expired_at`
  protects (§Rationale rehomed).
- **D11 — `MemoryService` (with the collaborators it owns) is the only in-process graph mutator; sweeps read
  through it; maintenance journals its own raw writes.** In-process reconciliation sweeps and stages
  (`degenerate_task_node_sweep`, the two stale-edge sweeps, `task_knowledge_sync`, `standing_decision_writer`,
  `memory_consolidator`, `server/tools.py` health) call the C7 read surface and the journaled mutators
  (`delete_entity`, `update_edge`, new `remove_edges`). Maintenance library modules (`cleanup_stale_edges`,
  `reindex`, `task_family_census`, `verify_zombie_edges`, `cross_graph_move`) already take or now take the
  backend as a parameter. Their mutations that bypass the journal today (`bulk_remove_edges` in two modules)
  route through `MemoryService.remove_edges`. Maintenance mutations with no service counterpart (measured at
  `e634cfe658`: 1 direct `mem0.update` in `tag_cgl_eta_rehome_scope.py`, raw qdrant writes via
  `_get_async_qdrant` ×3, graph-key purges `graphiti._graph_for(key).delete()` in
  `consolidate_namespace_families`/`purge_knowlive_namespace`, and cross_graph_move's 7 mutators) wrap their
  raw write in `journaled_ops.journaled_op` with the journal `maintenance_service` opened (S26-5: B). They do
  not grow a `MemoryService` pass-through per kind. Scripts construct the backend they read
  (`Mem0Backend(config)`, `GraphitiBackend(config)` + `initialize(skip_maintenance=True)`) instead of reaching
  through a service (S26-1: A). `Mem0Backend` gains the public read the three `_get_async_qdrant` scripts
  need (a scroll of raw points), so the `# noqa: SLF001` in `audit_duplicate_memories.py` goes (INV-12).
- **D12 — composition over regex prose.** `canonical_labels` keeps its public API (`Referent`, `LabelScan`,
  `parse_node_name`, `scan_content`, `is_task_vocabulary_qualifier`) and its measured precision rules; the
  five patterns are built from named fragments (`_TASK_WORD` already, plus the padding, the separator
  alternation, the digit capture, the right edge, the qualifier class, the left-edge exclusion sets) by one
  `_compile` helper, so a rule has one spelling. The 698 prose lines are reduced to a module docstring that
  states the grammar and points at `tests/test_canonical_labels.py` for the per-clause measured cases
  (Comments stance); task archaeology goes to the commit message. `_TASK_VOCABULARY_QUALIFIER` stays its
  own lowercase-only fragment (`(sub_?)?tasks?`), fullmatched against a qualifier its caller lowercased.
  Deriving it from `_TASK_WORD` (`[Tt][Aa][Ss][Kk][Ss]?`) would make `is_task_vocabulary_qualifier('Task')`
  True for direct callers that do not lowercase; row 19 pins it False.
- **D13 — prose stance.** Every docstring a leaf rewrites follows the Comments stance in
  `docs/code-quality.md`: leaf narration (`leaf gamma … leaf iota`, `Resolved decision 7`, `PLACEMENT IS
  LOAD-BEARING`, task-number histories — 55 distinct task numbers in `memory_service.py` prose) moves to
  this PRD's §Rationale rehomed or the commit message, leaving a one-line pointer. Placement constraints
  become call order (D7, η). Measurements go to the task record. A leaf that moves a symbol retargets the
  tracked prose that cites it by `memory_service.py::…` (ε: `fm/utils/referent_verification.py`,
  `fm/backends/graphiti_client.py`, `fused-memory/scripts/audit_wrong_binding_edges.py`; ρ, τ and υ: every
  tracked citation of a symbol they move). That prose is invalidated by this change, so the pin rule's
  carve-out for untouched prose does not apply. The split leaves ρ, τ and υ are moves: they carry docstrings
  and comments verbatim apart from those retargets; the prose pass over the moved and remaining code is φ.
- **D14 — no invariant promotion.** R22 names none for S26. The guard against regression is structural:
  types (C1, C4), one home per fact (C8), the ownership-checked lock (C6), collaborators that cannot import
  their owner (C9) and the integration gate's re-measurement (π).
- **D15 — the service splits at its measured seams (S26-2, ruled by Leo).** The 2026-11-10 measurement found
  the graph mutators call only `_emit_event` and `_journaled_backend_call`, read only injected objects, and
  are called by nothing else in the file; the memory mutators are called only by the hierarchy group. The
  shared state the 2026-11-02 draft cited as a stitching risk stops being service state once δ makes
  `journaled_op` a free function and λ puts the lock on the backend. So the split needs one small shared
  layer, `services/journaled_ops.py` (`journaled_op` from δ; `journaled_backend_call` and an `EventSink`
  holder from τ, ≈200 lines), never `write_journal.py`, which is already 1,618 lines. `GraphMutations` (τ)
  takes the 14 graph-mutator methods, the entity-mint storm state and μ's `remove_edges`; `MemoryMutations`
  (υ) takes the hierarchy and memory-mutator groups and the Mem0 update-storm state, with the
  `known_projects` map that state's escalator reads (injected at construction; no setter survives γ). υ depends on ρ: without
  the policy and codec in their own modules, `memory_mutations.py` would import
  `_apply_memory_metadata_validation`, `_normalize_task_id_metadata` and `_mem0_content` back from
  `memory_service.py`, the stitching heuristic 14 forbids. The three thin Mem0 reads the memory mutators call
  (`get_memory_by_id`, `count_memories_by_metadata`, `get_memories_by_metadata`) become functions in
  `store_records.py` beside the decoders they return through; the service delegates to them. The
  metadata-storm detector is shared by `add_memory`, `add_system_record` and `update_memory`, so it is one
  object `MemoryService.__init__` builds and injects into `MemoryMutations`. `MemoryService` keeps one-line
  public delegates for the 16 public mutators (`git grep '\.delete_memory\b'`, which also counts the
  same-named backend method, finds 31 test files and 9 script lines), so no caller outside the moved code
  changes. The split goes after μ and ν, the last planned
  edits to the moved code, so no planned leaf rebases onto moved code; the 18 live tasks that name a moved
  mutator get a note on where its code went (§Live tasks). Rejected: stopping at the identity split (the
  file stays ≈7,800 lines and the merge-lane contention point for every memory-service task); a
  `MemoryMaintenance` collaborator reaching back into the service (stitching).

## Contracts

```python
# C1 — fm/services/queued_writes.py (α)                       durable_queue imports; memory_service imports
class QueueOperation(StrEnum):    # the queue's vocabulary only; write-journal operation strings are separate (D1)
    ADD_EPISODE = 'add_episode'; ADD_MEMORY_GRAPHITI = 'add_memory_graphiti'
    MEM0_CLASSIFY_AND_ADD = 'mem0_classify_and_add'; MEM0_ADD = 'mem0_add'   # decode-only legacy (D2)
class QueueCallback(StrEnum):
    DUAL_WRITE_EPISODE = 'dual_write_episode'; REFRESH_ENTITY_SUMMARIES = 'refresh_entity_summaries'

@dataclass(frozen=True)
class WriteEnvelope: project_id: str; group_id: str; agent_id: str | None = None; session_id: str | None = None
                     causation_id: str | None = None; write_op_id: str | None = None
@dataclass(frozen=True)
class DecodedReferents: refs: ReferentSet; source: str; ambiguous: ReferentSet | None
    # ambiguous None = pre-5262 row (re-derive from content); () = "believe: nothing ambiguous". Absent key → ((), 'none', None)

@dataclass(frozen=True)
class QueuedEpisodeWrite:          operation = QueueOperation.ADD_EPISODE;           envelope; content; name; source; source_description; reference_time; correlation_id; temporal_context; unverified_claim; referents: DecodedReferents; …   # no memory_category: the add_episode payload never carries one
@dataclass(frozen=True)
class QueuedGraphitiMemoryWrite:   operation = QueueOperation.ADD_MEMORY_GRAPHITI;   envelope; content; name; source; source_description; memory_category: str | None; referents: DecodedReferents; unverified_claim
    # no correlation_id (never on this operation); envelope.write_op_id/causation_id may be None (replay_from_store rows carry neither)
@dataclass(frozen=True)
class QueuedMem0Fact:              operation = QueueOperation.MEM0_CLASSIFY_AND_ADD; envelope; fact_text; temporal_context; unverified_claim
    # envelope.write_op_id minted at enqueue, one per fact, so retries upsert one write_ops row (D6)
@dataclass(frozen=True)
class QueuedMem0Add:               operation = QueueOperation.MEM0_ADD;              envelope; content; metadata   # exactly what _execute_mem0_write reads
QueuedWrite = QueuedEpisodeWrite | QueuedGraphitiMemoryWrite | QueuedMem0Fact | QueuedMem0Add
# each: to_json() -> str (today's flat key names, D2) ; @classmethod from_json(text, *, group_id) -> Self (legacy defaults, D2)
def decode_queued_write(operation: str, payload: str, *, group_id: str, row_id: int) -> QueuedWrite
class QueueDecodeError(ValueError): operation: str; row_id: int; raw_payload: str    # a body from_json rejects
class UnknownQueueOperation(QueueDecodeError): ...                                  # an operation string not in QueueOperation

# fm/services/durable_queue.py (α; start gating γ)
class DurableWriteQueue:
    async def enqueue(self, value: QueuedWrite, *, callback: QueueCallback | None = None) -> int
    async def enqueue_batch(self, values: Sequence[tuple[QueuedWrite, QueueCallback | None]]) -> list[int]
    # execute_write: Callable[[QueuedWrite], Awaitable[Any]]; callbacks: Callable[[QueuedWrite, Any], Awaitable[None]]
    # _classify_failure: ('permanent', 1) for isinstance(exc, QueueDecodeError) — by type (D1); else today's two-way choice
    async def initialize(self) -> None        # γ: schema only
    async def start(self) -> None             # γ: recover in-flight, then start workers for every pending group
```
Invariants: a row's `operation` column is always a `QueueOperation` value; a value is never mutated after
`enqueue`; the executor and the callback of one row observe the same value object; an undecodable row never
reaches an executor and dead-letters on attempt 1 with its raw payload; `_ensure_workers` starts nothing
before `start()` (γ), so `enqueue`, `enqueue_batch` and `replay_dead` before `start()` only persist.

```python
# C2 — fm/models/memory.py (β)                                 MCP clients, reconciliation/stage_stats consume
class AddMemoryResponse(BaseModel):
    memory_ids: list[str]; stores_written: list[SourceStore]      # landed, ids known
    stores_queued: list[SourceStore]                               # β: durably enqueued; lands via the queue
    stores_pending_retry: list[SourceStore]                        # task 7397
    category; message: str; classification_fallback
    @property
    def mem0_write_state(self) -> Mem0WriteState                   # task 7650, unchanged by β (D3)
```
Invariant: a store appears in at most one of the three lists (a failed Graphiti enqueue appears in none);
`message` is rendered from them and read by no code; `mem0_write_state` reads only `memory_ids` and
`stores_pending_retry`, as today.

```python
# C3 — fm/services/memory_service.py (γ)                       server/main.py, maintenance/_utils.py, tests consume
class MemoryService:
    def __init__(self, config: FusedMemoryConfig, *, write_journal: WriteJournal, known_projects: Mapping[str, str],
                 graphiti: GraphitiBackend | None = None, mem0: Mem0Backend | None = None,
                 taskmaster: TaskBackend | None = None, recon_journal=None, recon_ledger=None,
                 planned_registry: PlannedEpisodeRegistry | None = None) -> None   # constructs DurableWriteQueue (pure)
    def set_event_buffer(self, event_buffer: EventBuffer) -> None   # the one pre-start setter; RuntimeError after start()
    async def initialize(self) -> None            # opens owned resources; starts nothing
    async def initialize_for_reads(self) -> None  # task 6239's read-only opening; opens no queue store
    async def start(self) -> None                 # after initialize(); the only call that drains backlog and starts workers
    async def close(self) -> None
# fm/maintenance/_utils.py (γ0 shape; γ body)                  every operator script and maintenance module consumes
class ServiceOpen(StrEnum): FULL = 'full'; READS = 'reads'; NONE = 'none'
@asynccontextmanager
async def maintenance_service(config_path: str | None = None, *, config: FusedMemoryConfig | None = None,
                              open_as: ServiceOpen = ServiceOpen.FULL) -> AsyncIterator[tuple[FusedMemoryConfig, MemoryService]]
    # γ: constructs WriteJournal(Path(config.reconciliation.data_dir) if config.reconciliation else Path('./data'))
    #    and build_known_projects_map(primary_root=''); FULL awaits journal.initialize() then service.initialize();
    #    READS awaits journal.initialize() then service.initialize_for_reads(); NONE awaits neither (D4).
    #    Never start() (D4 consequence in its docstring).
# fused-memory/tests/conftest.py
def build_memory_service(config, *, graphiti=None, mem0=None, write_journal=None, known_projects=None, taskmaster=None,
                         planned_registry=None, **optional) -> MemoryService
```
Invariants: `write_journal` is never `None` on an instance (no guard anywhere); no queue worker runs before
`start()`, whatever is enqueued or replayed; `start()` is idempotent and refuses after `close()` and before
`initialize()`; `set_event_buffer` after `start()` raises; no production module outside `server/main.py` and
`maintenance/_utils.py` constructs a `MemoryService` (`(=|return) *([A-Za-z_]+\.)?MemoryService\(` and `=
MemoryService,` absent under `fused-memory/scripts/` and `scripts/`; tests use
`fused-memory/tests/conftest.py::build_memory_service`).

```python
# C4 — fm/services/episode_identity.py (ε; D8 in λ)             memory_service owns; tests construct
class EpisodeIdentityReconciler:
    def __init__(self, *, graphiti: GraphitiBackend, write_journal: WriteJournal, known_projects: Mapping[str, str],
                 taskmaster: TaskBackend | None, repair_streak_threshold: int = REFERENT_REPAIR_STREAK_THRESHOLD) -> None
    async def run(self, result: EpisodeWriteResult, *, group_id: str, referents: ReferentSet, ambiguous: ReferentSet | None,
                  content: str, referent_source: str) -> ReconcileStats      # caller holds graphiti.identity_lock(group_id)
    def record_referent_source(self, source: str) -> None                    # INV-4 denominator; caller-side, before add_episode
    @property
    def counts(self) -> IdentityCounts   # frozen snapshot: referent_source, finding, repair counters
class EpisodeWriteResult:  # built once from graphiti_core's AddEpisodeResults
    @classmethod from_graphiti(cls, result) -> Self ; episode_uuid: str | None; edges: tuple; nodes: tuple
class RepairOutcome(StrEnum): …            # task 6601's vocabulary
class IdentityLockNotHeld(RuntimeError): group_id: str   # run() called without the group's identity_lock (D8; ε lands after λ)
```
Invariants: the passes run in `run()`'s stated order, each best-effort with its failure recorded in
`ReconcileStats` (never swallowed silently; INV-11); `run()` raises `IdentityLockNotHeld` if the calling
task does not hold the group's lock; `ambiguous=None` re-derives from `content`, `()` believes
nothing ambiguous; the reconciler mutates only through the backend it was constructed with, under the
caller's `identity_lock`, and never calls a service mutator (those take the lock); no module imports
a pass by name; `episode_identity.py` has no reference to `MemoryService`, `GraphMutations` or
`MemoryMutations`.

```python
# C5 — fm/backends/graph_access.py (θ)                          graphiti_client owns; tests, cross_graph_move, falkor_read_only, scripts consume
class GraphHandle(Protocol):
    name: str
    async def query(self, cypher: str, params: Mapping[str, Any] | None = None, *, op: str) -> Any
    async def ro_query(self, cypher: str, params: Mapping[str, Any] | None = None, *, op: str) -> Any
    async def ro_query_compact(self, cypher: str, params: Mapping[str, Any] | None = None, *, op: str) -> Any   # byte-exact --compact
class GraphAccess(Protocol):
    def graph(self, group_id: str) -> GraphHandle
    async def list_graphs(self) -> list[str]
class FalkorGraphAccess(GraphAccess):
    def __init__(self, driver: FalkorDriver, *, read_budget_s: float, write_budget_s: float) -> None   # the one driver._get_graph site
async def paged_ro_query(graph: GraphHandle, cypher: str, params, *, op: str, page_size: int = …) -> PagedRead   # moved from graphiti_client
# fm/backends/graphiti_client.py
class GraphitiBackend:
    @classmethod
    def with_driver(cls, config, driver, *, registered_graph_ids, llm_client=None, embedder=None, cross_encoder=None,
                    graph_access: GraphAccess | None = None) -> GraphitiBackend
    @property
    def graph_access(self) -> GraphAccess      # read-only; the provider passed to with_driver or built in initialize()
# fused-memory/tests/conftest.py
class FakeGraphAccess(GraphAccess): def __init__(self, graphs: Mapping[str, Any]) -> None   # KeyError names an unwired group
```
Invariants: every Cypher the backend issues passes through a `GraphHandle`; every `query`/`ro_query`
raises a named `TimeoutError` at its budget; no module outside `graph_access.py` names `_get_graph`; after
θ2 `_graph_for` is gone and no module outside `graphiti_client.py` calls `_require_falkor_client` (it stays
the backend's private accessor for its own `list_graphs` reads; neither θ nor θ2 deletes it).

```python
# C6 — fm/backends/graphiti_client.py (λ)                        memory_service, graph_mutations (τ), episode_identity, cross_graph_move consume
class GraphitiBackend:
    def identity_lock(self, group_id: str, *, timeout: float | None = None) -> AbstractAsyncContextManager[None]
    def identity_lock_held(self, group_id: str) -> bool          # True only for the holding task and tasks it spawned under the hold (ContextVar, S26-7: C)
class IdentityLockBusy(Exception): group_id: str; waited_s: float      # structured refusal; MCP tools render error_type 'IdentityLockBusy' (OQ-3)
class IdentityLockReentry(RuntimeError): group_id: str                 # S26-7: C
# fm/config/schema.py
class IdentityLockConfig(BaseModel): acquire_timeout_seconds: float = Field(default=…)   # default here only; green-tier in config/reload.py
```
Invariants: every in-process mutator of entity identity (`_execute_graphiti_write`; `ensure_entity_node`,
`merge_entities`, `rename_entity`, `delete_entity`, `reassign_edge`, in `GraphMutations` after τ; and
`cross_graph_move`'s `create_moved_node`, `move_entity_across_graphs`, `recreate_subgraph_relationships`,
`delete_source_node`, `create_moved_episode`, `delete_source_episode`, `merge_foreign_duplicate`) holds it
for the whole mutation; MCP and reconciliation-stage callers pass `identity_lock.acquire_timeout_seconds`,
the queue worker passes `None` and is the one unbounded holder; every group a call touches is acquired in
`sorted()` order; re-entry by the holder raises; the lock is never held across a Mem0 write (task 2202's B3).

```python
# C7 — fm/services/memory_service.py (μ)                        reconciliation sweeps, stages, server/tools.py consume
class MemoryService:
    async def valid_edges_for_node(self, node_uuid: str, *, project_id: str) -> list[EdgeDict]
    async def enumerate_valid_edges(self, *, project_id: str) -> …                    # checked read, as rebuild_entity_summaries uses
    async def enumerate_valid_edges_mentioning(self, needle: str, *, project_id: str) -> …
    async def find_duplicate_entity_nodes(self, name: str, *, project_id: str) -> list[dict]
    async def store_inventory(self) -> StoreInventory                                  # graph ids + mem0 projects (health_check, get_status)
    async def list_memories(self, *, project_id: str, …) -> …                          # memory_consolidator's get_all
    async def remove_edges(self, uuids: Sequence[str], *, project_id: str, reason: str) -> RemoveEdgesResult   # journaled; cleanup/zombie maintenance; a delegate to GraphMutations after τ
    # backends are self._graphiti / self._mem0; no public attribute
# fm/backends/graphiti_client.py: delete_entity_node → private _detach_delete_entity_node (the primitive under merge_entities/delete_entity)
```
Invariant: outside `services/memory_service.py`, `services/episode_identity.py` and (after τ and υ)
`services/graph_mutations.py` and `services/memory_mutations.py`, no module under `fm/` calls a
`GraphitiBackend` mutator, except `maintenance/*`, whose raw writes are wrapped in
`journaled_ops.journaled_op` (S26-5: B), and `arm_harness/*`, which writes only scratch graphs built by
`arm_harness/arm_backend.py::scratch_graphiti_backend` (guarded by `arm_harness/scratch_guard.py::require_scratch_name`);
every MCP-reachable and reconciliation-stage graph mutation leaves a `write_ops` row.

```python
# C8 — fm/backends/relates_to_schema.py (ξ)                     graphiti_client, cross_graph_move, episode_identity consume
RETIREMENT_MARKERS: tuple[str, ...] = ('invalid_at',)          # ruling A, esc-4714-9
ACTIVE_EDGE_PREDICATE: str                                       # f'{alias}.invalid_at IS NULL' derived; alias parameter
def edge_is_active(edge: Any) -> bool
RELOCATION_REASSERTED_PROPERTIES: tuple[str, ...] = ('source_uuid', 'target_uuid', 'source_node_uuid', 'target_node_uuid')
    # OQ-1 ruled (b): in-graph SET new += properties(old) keeps every type incl. Vectorf32, so fact_embedding is NOT here
    # (vecf32() on a Vectorf32 errors); endpoint ids re-asserted from the MATCHed new endpoints
def relocation_set_clause(*, new: str, old: str, overrides: Mapping[str, str]) -> str
    # renders `SET {new} += properties({old}), {new}.source_uuid = …, …` + per-site overrides (uuid, superseded_edge_uuid, reassigned_from_node_uuid)
def cross_graph_embedding_reassert(compact_reply: str | None) -> str:     # CROSS_GRAPH_EMBEDDING_REASSERT rule (lead ruling on critic3 N1 (a))
    # cross_graph_move strips fact_embedding from $props and appends _embedding_set_clause('r.fact_embedding', compact_reply):
    # the byte-exact --compact tokens, never float(); '' for a null/absent source embedding. Node name_embedding uses the
    # same transport. Legacy List-typed embeddings: carried as List in-graph; refused cross-graph by _read_compact_vector
    # (wrong-typed column, as today); upgrading them is out of scope
```
Invariant: the Cypher predicate and the Python predicate agree by construction (one marker list); a
property stored on a RELATES_TO edge survives `reassign_edge`, `redirect_node_edges` and every
`cross_graph_move` recreate with its stored type unless named in that module as a re-asserted exception;
every endpoint-id property of a relocated edge names its new endpoint; a Vectorf32 `fact_embedding` is
value-equal at float32, Vectorf32 and indexed after every relocation, in-graph and cross-graph.

```python
# C9 — the service split (δ creates journaled_ops; τ, ρ, υ the rest)   S26-2; module names confirmed by the lead 2026-11-10 (t7)
# fm/services/journaled_ops.py — memory_service, graph_mutations, memory_mutations, maintenance/* and scripts consume
@asynccontextmanager
async def journaled_op(journal: WriteJournal, operation: str, params: Mapping[str, Any], *, source: str, kind: str = 'write',
                       provenance: str | None = None, project_id: str, agent_id=None, session_id=None, causation_id=None,
                       write_op_id: str | None = None) -> AsyncIterator[JournaledOp]          # δ (D6)
async def journaled_backend_call(journal: WriteJournal, *, write_op_id, causation_id, backend, operation, payload, coro) -> Any
    # τ: MemoryService._journaled_backend_call moved unchanged
class EventSink:                                                  # τ: the one holder of the optional EventBuffer
    def wire(self, buffer: EventBuffer) -> None                   # MemoryService.set_event_buffer forwards here (pre-start only)
    async def emit(self, event: ReconciliationEvent) -> None      # no-op until wired, as _emit_event is today
# fm/services/graph_mutations.py (τ)
class GraphMutations:
    def __init__(self, *, graphiti: GraphitiBackend, write_journal: WriteJournal, config: FusedMemoryConfig,
                 known_projects: Mapping[str, str], events: EventSink) -> None
    # update_edge, reassign_edge, ensure_entity_node, delete_episode, redact_episode_content, refresh_entity_summary,
    # set_entity_summary, rename_entity, rebuild_entity_summaries, merge_entities, delete_entity, remove_edges:
    # today's MemoryService signatures and journaling, unchanged; the entity-mint storm state lives here
# fm/services/memory_mutations.py (υ)
class MemoryMutations:
    def __init__(self, *, mem0: Mem0Backend, graphiti: GraphitiBackend, write_journal: WriteJournal, config: FusedMemoryConfig,
                 known_projects: Mapping[str, str], events: EventSink,
                 metadata_storm: UnknownKeyStormDetector) -> None   # the one detector MemoryService builds
    # delete_memory, update_memory and the parent-child hierarchy / cascade helpers; the Mem0 update-storm state lives here:
    # it builds its Mem0UpdateStormEscalator with known_projects (γ moved set_known_projects' propagation into __init__; no setter)
# fm/services/memory_metadata_policy.py (ρ) — public functions, today's bodies and sync/async shapes (collaborators passed in):
#   apply_memory_metadata_validation, normalize_task_id_metadata, infer_recon_pool, apply_cycle_summary_metadata_tagging,
#   check_canonical_uniqueness, strip_recon_pool_from_non_cycle_summary, stamp_unverified_claim, … (the 883-line block plus infer_recon_pool)
# fm/services/store_records.py (ρ; υ adds the three Mem0 reads) — MEM0_CONTENT_KEYS, mem0_content, metadata_row_text,
#   the record decoders (node_to_dict, edge_to_dict, serialize_temporal, …), the store-error helpers
#   (degrade_or_reraise, store_failure_diagnostics, …); υ: get_memory_by_id, count_memories_by_metadata, get_memories_by_metadata
```
Invariants: `graph_mutations.py`, `memory_mutations.py`, `memory_metadata_policy.py`, `store_records.py` and
`journaled_ops.py` import nothing from `memory_service`; `MemoryService` builds `GraphMutations` and
`MemoryMutations` in `__init__` from its constructor collaborators and keeps one-line public delegates with
today's signatures; every mutation that wrote a `write_ops` row before the move writes the same row after
it; `server/grouped_read.py` and `reconciliation/preservation_specimen_guard.py` import public names from
`store_records`, not privates from `memory_service`.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | queue (α) | enqueue a `QueuedEpisodeWrite`; drain | empty store | executor receives the same frozen object the callback receives; one row; `operation='add_episode'`; payload keys equal today's flat keys |
| 2 | queue (α) | a row whose `operation` is `'typo_op'`, and a row whose body `from_json` rejects | hand-inserted rows | each dead-lettered with `attempts == 1` and a `QueueDecodeError` subclass; no executor called; `DeadLetterEvent` carries the operation and the raw payload |
| 3 | queue (α) | legacy rows: `add_memory_graphiti` without `project_id`; pre-3561 row with `uuid`; a row with no `referents` key; a `referents` blob without `ambiguous` | synthesised rows in the `6551b75aba` shape | decode; `project_id` = `group_id`; `uuid` ignored; both referent cases decode to `ambiguous is None` |
| 4 | response (β) | `add_memory` dual-write, Mem0 lands, Graphiti enqueued | fake backends | `stores_written=['mem0']`, `stores_queued=['graphiti']`, `mem0_write_state == Mem0WriteState.LANDED`, message names both states; `write_ops.result_summary.stores_queued == ['graphiti']`; `_REPORTED_TO_CALLER` text agrees |
| 5 | lifecycle (γ) | construct + `initialize()` over a store with 3 pending rows; no `start()` | pending rows | no worker task exists; rows unchanged; `get_queue_stats` reports them pending |
| 5b | lifecycle (γ) | construct + `initialize()`, then `add_memory` with a Graphiti leg and a `replay_dead` | empty store, one dead row | the new row is persisted `pending`, the replayed row is `pending`, no worker task exists, `get_queue_stats` counts both; after `start()` both drain |
| 5c | lifecycle (γ0/γ) | `maintenance_service(open_as=READS)` and `(open_as=NONE)` over a store with pending rows | pending rows | READS: `initialize_for_reads()` ran, no Graphiti maintenance, no queue store opened, no worker; NONE: no backend opened; neither can `start()`; `task_family_census`'s CLI opens READS |
| 6 | lifecycle (γ) | `start()` then a backlog row drains | journal wired at construction | `write_ops` terminal row written (today lost when `_write_journal is None`) |
| 7 | lifecycle (γ) | `set_event_buffer` after `start()` | started | `RuntimeError('event_buffer wired after start()')` |
| 8 | journal (δ) | every journaled operation fails (parametrised over the 19 sites); `search` degraded; `mem0_classify_and_add` fails twice then lands; a queue executor cancelled | fake backend raising / degraded / cancelled | exactly one `write_ops` row per operation, `success=0`, `error` names the exception class; degraded `search` → one `kind='read'` row `success=0`; the retried fact → exactly one row (final state); the cancelled op → one row and `CancelledError` re-raised; queue sites journal under the carried `write_op_id`; a fact row dead-lettered before execution leaves one `operation`-NULL stub under its id |
| 9 | identity (ε) | `run()` over a fixture write result | reconciler built directly | `ReconcileStats` field per pass; a pass raising records its failure and the next pass runs |
| 10 | identity (ε, on λ's lock) | `run()` without the lock held, and `run()` while another task holds it | — | `IdentityLockNotHeld(group_id)` in both cases |
| 10b | identity (λ) | the lock holder calls a lock-taking mutator for the same group | holder inside `identity_lock` | `IdentityLockReentry(group_id)`, no hang |
| 10c | identity (λ) | a coroutine the holder awaits through `asyncio.gather` reads `identity_lock_held` and then calls a lock-taking mutator | holder inside `identity_lock` | `identity_lock_held` is True in the child; the mutator raises `IdentityLockReentry`, no hang |
| 11 | search (η) | pin displaces the lowest hit; pin lookup raises `TypeError` vs `RuntimeError`; the knob toggled between two calls | fixture results | `apply_topic_anchor` is pure: same input → same output; `TypeError` propagates, `RuntimeError` fails open with the window unchanged; the second call sees the reloaded knob |
| 12 | graph (θ) | a `GraphHandle` whose `ro_query` never returns | `FakeGraphAccess` with a hanging graph | every backend Cypher method raises `TimeoutError('Graphiti <op> timed out after <budget>s')` within budget (parametrised over the methods) |
| 13 | graph (θ) | `with_driver(..., graph_access=FakeGraphAccess({gid: g}))`; call a Cypher method for another group; read `backend.graph_access` | — | `KeyError` names the unwired group; `backend.graph_access` is the injected provider; no private attribute touched (AST check over the test) |
| 14 | lock (λ) | two coroutines: `merge_entities` and `_execute_graphiti_write` on one group; a third with `timeout=0.01`; a cross-graph move over two groups racing one over the same two in the other order | fake backend recording order | second waits; interleaving impossible; the third gets `IdentityLockBusy`; the two-group moves never deadlock (`sorted()` order) |
| 15 | mutator (μ) | degenerate sweep deletes a 0-edge node | fixture graph | `write_ops` row `delete_entity`; `GraphitiBackend` has no public `delete_entity_node` |
| 15b | mutator (μ) | degenerate sweep meets a held lock | lock held with `acquire_timeout_seconds` tiny | the node is skipped, counted as an `IdentityLockBusy` skip in the sweep stats, retried next cycle; a streak escalates (INV-4) |
| 16 | mutator (μ; τ, υ extend) | AST scan of `fm/`: (i) `.graphiti.`/`.mem0.` attribute chains on a service object; (ii) every call whose attribute name is in an enumerated set of `GraphitiBackend` mutator method names (the scan's own constant), on any receiver — a receiver's type is not visible to an AST scan | post-μ tree | (i) zero outside `memory_service.py`; (ii) zero outside the module allowlist `services/memory_service.py`, `services/episode_identity.py`, `backends/*`, `maintenance/*` (raw writes inside `journaled_op` only) and `arm_harness/*`, to which τ and υ add `services/graph_mutations.py` and `services/memory_mutations.py`; `episode_identity.py` has no reference to `MemoryService`; the scan flags a seeded fixture module (INV-10) |
| 17 | schema (ξ) | relocate an edge carrying a custom attribute through `reassign_edge`, `redirect_node_edges` and `cross_graph_move` | live FalkorDB (integration tier) + mock tier | every stored property survives with its type except the module's named exceptions; a Vectorf32 `fact_embedding` is value-equal — the target edge's `--compact`-read tokens equal the source's, token for token — Vectorf32, and returned by `db.idx.vector.queryRelationships` after the in-graph and the cross-graph move (integration tier at production dimension 1536; the probe measured dimension 4); a source edge with no `fact_embedding` is recreated without one; a legacy List-typed embedding stays List in-graph; every endpoint-id property (`source_uuid`, `target_uuid`, `source_node_uuid`, `target_node_uuid`) names the new endpoint; a twice-redirected edge keeps its original `superseded_edge_uuid` |
| 18 | schema (ξ) | an edge with `expired_at` set, `invalid_at` NULL | — | `edge_is_active` is True and the Cypher predicate returns it (ruling A preserved) |
| 19 | grammar (ο) | the existing parametrised precision fixtures | — | unchanged verdicts; `parse_node_name('reify:\n132')` is None; `is_task_vocabulary_qualifier('Task')` is False; `_QUALIFIED_REF_PATTERN` and `_QUALIFIED_NODE_NAME_PATTERN` share one qualifier fragment (identity assert on the fragment, not substring) |
| 20 | gate (π) | `add_memory` → real SQLite queue → `start()` → typed executor → reconciler under `identity_lock` → journal → response | real `WriteJournal`, `FakeGraphAccess`, fake Mem0 | response per row 4; `write_ops` rows for the enqueue and the terminal outcome under one `write_op_id`; `ReconcileStats` recorded; no private name in the test |
| 21 | policy (ρ) | the existing metadata-policy and codec tests, importing the public names from `memory_metadata_policy`/`store_records` | — | identical verdicts; `memory_service.py` defines none of the moved names; `grouped_read.py` and `preservation_specimen_guard.py` import no private from `memory_service` |
| 22 | mutators (τ) | each public graph mutator through `MemoryService` and through a directly built `GraphMutations`, same fakes; `set_event_buffer` before `start()` | fake backend, real `WriteJournal` over `tmp_path` | identical results and `write_ops` rows on both paths; events reach the buffer through the one `EventSink`; `graph_mutations.py` names no `MemoryService` |
| 23 | mutators (υ) | `delete_memory` with children (refusal, then `cascade=True`) and an `update_memory` content amend, through `MemoryService` and through a directly built `MemoryMutations` | fake Mem0, real `WriteJournal` | identical refusal, rows and storm-escalator calls on both paths; `memory_mutations.py` imports nothing from `memory_service` |

## Cross-PRD relationship

| Other PRD / stream / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| Task 7650 (done, `789efc6ad6`…`314e1a92dd`, merged `ca07a6ffb9`) | consumes | `AddMemoryResponse.mem0_write_state` / `Mem0WriteState`, the one Mem0 reading | landed | β leaves it unchanged and adds `stores_queued` (D3); the 2026-11-02 `amend` (an `outcome_for` deliverable) is void |
| Task 6362 (done) | consumed | `test_e2e_durable_queue.py` fixtures | landed | α rewrites them; no edge |
| Task 6369 (done, `a555a1ecac`, merged `e7fcf2bf0a`) | consumed | `identity_payload_keys` seam deleted; `_classify_failure` two-way | landed | α and θ no longer wait; D1 adds the decode branch |
| Task 6233 (done, `d064e0181d`) | consumes | `GraphitiBackend.with_driver`, `conftest::make_graph_driver`, `make_wired_backend` | landed | θ adds `graph_access=`; ι/κ build on the factories |
| Task 6239 (done) | consumes | `MemoryService.initialize_for_reads` | landed | γ keeps it; γ0's `ServiceOpen.READS` routes the two eval scripts and the census CLI through it |
| Task 7781 (pending; the ~37-file private-wiring sweep) | subsumed | the same test files, moved onto C5 instead of only the driver | S26 ι/κ | cancel with `x_acceptance_reason` naming ι/κ (no branch exists) |
| Task 5612 (pending, branch 54 ahead / 9,138 behind, last commit 2026-10-12) | subsumed (S26-6: B) | in-file consolidation of the four SET lists; Leo's 2026-10-19 widening (`superseded_edge_uuid` overwrite rule, docstring accuracy) | S26 ξ | cancel naming ξ, `x_acceptance_reason`; ξ inherits the two widened items (D10) |
| Task 6823 (pending, low, no branch; `reassign_edge` endpoint read measured at 1,042 ms against FalkorDB's 1,000 ms TIMEOUT) | upstream (S26-6: B) | `reassign_edge` carries `superseded_edge_uuid`; index-seek endpoint read; same SET clauses θ rewrites | 6823 | θ depends on 6823; lead `amend` drops its "after task 5612" sequencing and its "not in 5612's constant" clause |
| Task 6680 (pending) | subsumed | `cross_graph_move` carries every property (`properties(e)`) | S26 ξ | cancel naming ξ; C8/D10 adopt its parameter-map form for `cross_graph_move` |
| Task 6730 (pending, low; files `memory_service.py`, `graphiti_client.py`, `test_entity_mint_service.py`) | overlaps | public identity-gated mint surface; `ensure_entity_node` composes three private primitives | 6730 | λ closes only the lock-privacy clause of its problem 1; τ moves `ensure_entity_node` into `GraphMutations`; edge: 6730 depends on υ (the last in-batch leaf editing `memory_service.py`; μ is the last of `graphiti_client.py`, τ of `test_entity_mint_service.py`); the follow-up φ, which re-measures whatever has landed, is deliberately unordered against 6730; it does not depend on σ (lead ruling); note re-scopes it |
| Task 6293 (pending, medium; scripts attach to the live queue; files include `durable_queue.py`, `bake_off_storage_shape.py`) | absorbed into σ (design) | γ: `initialize()` no longer drains or starts workers (C3); σ settles its ownership question (S26-3, lead ruling) | σ decides; 6293 executes or closes | edges: 6293 depends on σ (its design is the session's) and on ν (the last in-batch editor of `bake_off_storage_shape.py`, after δ, the last of `durable_queue.py`); ν no longer depends on 6293 |
| Task 6835 (pending, low; `task_family_census` CLI runs startup writes) | subsumed | a read-only `maintenance_service` opening | S26 γ0 | cancel naming γ0 (`ServiceOpen.READS`, the census CLI opens it) |
| Task 8660 (pending, low; module-scope `PlannedEpisodeRegistry` import; tests inject via `set_planned_registry`) | subsumed | registry injection | S26 γ | cancel naming γ: γ deletes `set_planned_registry`, adds the `planned_registry=` keyword, moves the three dotted-path patches onto it and the import to module scope |
| Task 6950 (pending, low; `Mem0Backend` public async-Qdrant seam and ≈70 test patch sites; its `metadata.files` include `memory_service.py`, `test_memory_service.py`, `test_mem0_client.py`, `test_audit_duplicate_memories.py`, `test_census_memory_metadata.py`) | adjacent | `MemoryService` supplies its `Mem0Backend` (γ's `mem0=`); public raw-point scroll (μ); scripts' private calls removed (ν) | 6950 | edge: 6950 depends on υ — ρ, τ and υ edit five of its files after ν (`memory_service.py`, `test_memory_service.py`; ρ also `test_mem0_client.py`, `test_audit_duplicate_memories.py`, `test_census_memory_metadata.py`), so an edge on ν alone left them unordered; φ is deliberately unordered against it, as against 6730; note: γ delivers its "matching way for MemoryService to supply it", 6950 keeps the backend's client seam and the test sweep |
| Task 6636 (`fk-16d7023811f6`, pending; `DeadLetterEvent` fields; file `durable_queue.py`) | same files | `DeadLetterEvent` | 6636 | edge: 6636 depends on δ (the last in-batch editor of `durable_queue.py`) |
| Task 6601 (`fk-ec3de5b6cf2b`, pending) | subsumed | `RepairOutcome` StrEnum | S26 ε | cancel naming ε; ε carries `fk-ec3de5b6cf2b` |
| Task 6826 (pending; `_QUALIFIED_REF_PATTERN` prose precision) | consumes | the qualifier fragment (D12) | 6826 | edge: 6826 depends on ο |
| Task 6631 (low, pending; `graphiti_client.py::_construct_llm_client` prose) | same file | — | 6631 | edge: θ depends on 6631 |
| 18 live tasks naming a moved mutator, and 4857 (`_MEM0_CONTENT_KEYS`) | same file | `memory_service.py` | theirs | lead `note` on each naming where its code lives after ρ/τ/υ; no edge (§Live tasks) |
| Tasks 4235, 4479 (deferred) | stale | both fixes landed in task 4850 (`ea47bfb5ec`; `_TASK_WORD`, `[ \t]` padding) | — | lead action: cancel as landed (not S26 keys) |
| Tasks 4870, 4872 (pending) / 4812, 4770 (deferred) (timed-out add_memory, journal coverage) | adjacent | `AddMemoryResponse`; `_journaled_backend_call` | theirs | xref only: β's `stores_queued` does not address timeouts; 4870 is among the noted tasks (τ moves `_journaled_backend_call`) |
| Task 6635 (low, pending; `services/__init__` re-export) | same package | — | 6635 | not serialized: no S26 leaf edits `services/__init__.py` |
| Gate 6698 (pending; deps 6707, 6721, 6736, 6740, 6747, 6750; its 259,200 s delayed milestone from 2026-11-01 has elapsed) | downstream | trigger-chain gate | lead | decompose adds ε, θ, λ, ξ, σ as dependencies and confirms with `get_task`; σ makes the next review wait on a human sitting (S26-3 consequence) |
| S10 task-write contract (wave 2 sibling) | same files | `server/main.py::run_server` construction order; `server/tools.py` | each | S10 κ1 and μ2 depend on S26 γ (both need the constructed `MemoryService` with `WriteJournal` built first; S10-1 ruled as recommended); S26 λ, μ and ρ also edit `server/tools.py` (ρ: the `metadata_row_text` import), unordered with S10's `tools.py` chain; S26 μ edits `reconciliation/stages/task_knowledge_sync.py` and S26 γ0/θ2/ν edit 4 scripts S10 also touches (`invalidate_fabricated_shipping_edges`, `memory_eval_staleness_sweep`, `normalize_topic_slugs`, `sweep_orphan_flag_markers`) — the lead wires those orders at decompose |
| S5 GitOps ports (wave 2 sibling) | none now | — | — | no shared file: S5's fused-memory work moved with the probe reference into `fused-memory/tests/_fm_git_repos.py` (task 6454's split), which no S26 leaf edits; S26 ε and ι edit `_fm_helpers.py` alone (re-checked against S5's draft and return, 2026-11-10); no order needed |
| S19 package layering (R17, not authored) | adjacent | `fm` subpackage DAG | S19 | α, δ, ε, θ, ξ, ρ, τ, υ add leaf modules that import downward only (D7, D9, C9); nothing for S19 to redefine |
| S15 fused-memory MCP server boundary (R14, not authored) | adjacent | `server/tools.py::health_check` reads C7 `store_inventory` | S26 provides; S15 consumes if it moves health | xref |

## Live tasks (contract §8)

Step 1, re-run 2026-11-10 (twice: at the ruling fold and again at fold-verify, all 136 values read):
`find_tasks_by_metadata(key="x_finding_key")` over all 136 tasks carrying the key found none carrying any of
the 18 keys (6601 carries `fk-ec3de5b6cf2b`, which ε absorbs). Step 2
(`search_tasks` ≥ 0.6, confirmed by `get_task`) for the new leaves ρ, τ, υ, σ and φ returned nothing at or
above 0.6 (best: task 310, 0.59, done; 6293 scored 0.50 for σ's query). The same-file sweep the S26-2 ruling asks for —
`find_tasks_by_metadata(key="files", value="fm/services/memory_service.py")` over pending, in-progress and
blocked, 45 tasks, each read with `get_task` — found 18 that name a mutator ρ/τ/υ moves: 3010, 3233, 4493,
4521, 4859, 4861, 4870, 4876, 4898, 5065, 5154, 6054, 6325, 6459, 6497 (in progress), 6730, 7152, 7163; plus
4857, which names `_MEM0_CONTENT_KEYS` (ρ). It also found three that overlap planned leaves: 6835 (γ0),
8660 (γ), 6950 (μ/ν). Dispositions at decompose: no out-of-batch dependency remains for α (6362, 6369, 7650
done); dependencies 6631, 6823 (→ θ); `amend` on 6823 (drop the 5612 sequencing); `cancel` with
`x_acceptance_reason` for 7781 (→ ι/κ), 6680 (→ ξ), 6601 (→ ε, which carries `fk-ec3de5b6cf2b`), 5612 (→ ξ,
S26-6 ruled B), 6835 (→ γ0), 8660 (→ γ); `edge` for 6636 (→ δ), 6293 (→ σ and ν), 6730 (→ υ), 6826 (→ ο),
6950 (→ υ; ρ/τ/υ edit five of its files after ν); `note` on 6730, 6293, 6950, 6698 and on each of the 18 mutator-naming tasks and 4857 (where the
code moved; no edge — they keep their place in the scheduler's file serialization); `cancel` 4235 and 4479
as landed by 4850. Re-run step 1 immediately before filing (deliberation §Launching).

## Decomposition plan

Sizes are expected changed LOC. Every leaf is `task_kind="normal"` except σ (`deterministic`, the pure-gate
form); `priority` is its keys' highest severity; **ε, θ, λ, ξ and σ are `high`** and become dependencies of
gate 6698 at decompose. Same-file leaves are serialized by real edges (the DAG below was checked file by file
against every leaf's Files and every named task's `metadata.files`). ο is independent. Nothing is held (no
upstream stream). Each leaf's `metadata.files` is its Files list.

- **α — Typed queued-write values and a QueueOperation enum beside durable_queue**
  - Size and keys: ~900 LOC. Keys `fk-1215aacfa626`, `fk-bdc0427c8971`. Priority medium.
  - Files: `fm/services/queued_writes.py` (new), `fm/services/durable_queue.py`,
    `fm/services/memory_service.py`, `fm/middleware/dead_letter_escalator.py`,
    `fused-memory/tests/test_queued_writes.py` (new), `fused-memory/tests/test_durable_queue.py`,
    `fused-memory/tests/test_e2e_durable_queue.py`, `fused-memory/tests/middleware/test_dead_letter_escalator.py`.
  - Work: C1, D1, D2. Producers construct values; `_execute_durable_write` is a type table; `_process_item`
    decodes once with the row id; `_classify_failure` makes `QueueDecodeError` permanent by type ahead of its
    two-way choice; the undecodable row's `DeadLetterEvent` carries the raw payload; `to_json` keeps today's
    flat keys.
  - Deletes: 4 producer dict literals; the pop-then-read-back convention and `_process_item`'s second parse;
    the `_execute_graphiti_write(operation, payload)` fall-through; the `'add_episode'`/`'mem0_add'`/
    `'mem0_classify_and_add'`/`'add_memory_graphiti'` queue-operation literals in the 4 modules that hold them
    (write-journal operation strings are a separate vocabulary and stay).
  - Signal: boundary rows 1–3. A hand-inserted `typo_op` row in a test store dead-letters on attempt 1 with
    a structured reason (`get_dead_letters` shows it) instead of executing as a Graphiti write.
  - Consumer: β, γ (lifecycle), `dead_letter_escalator`, 6636.
  - Deps: none (6362, 6369 and 7650 are done).
  - G7: INV-11 holds (an undecodable row is loud and terminal); INV-5 holds (one queue vocabulary); INV-12:
    no suppression added.
- **β — add_memory reports Graphiti as queued, not written**
  - Size and keys: ~350 LOC. Key `fk-b30cc4fa2899`. Priority medium.
  - Files: `fm/models/memory.py`, `fm/services/memory_service.py`, `fm/middleware/dead_letter_escalator.py`,
    `fm/reconciliation/stage_stats.py`, `fm/maintenance/seed_autopilot_video_triage_guardrails.py`,
    `fused-memory/tests/test_seed_autopilot_video_triage_guardrails.py`, `fused-memory/tests/test_memory_service.py`,
    `fused-memory/tests/test_add_memory_response.py`, `fused-memory/tests/test_e2e_durable_queue.py`,
    `fused-memory/tests/test_mem0_intent_recovery.py`, `fused-memory/tests/test_stage_stats.py`,
    `fused-memory/tests/test_sweep_toolcall_xml_leak.py`,
    `fused-memory/tests/server/test_add_memory_write_triage_gate.py`,
    `fused-memory/tests/test_replay_failed_mem0_intents_script.py`.
  - Work: C2, D3. The 8 test files asserting `graphiti in stores_written` or `Memory queued for` (measured at
    `e634cfe658`) move to `stores_queued`/the rendered message; `add_memory`'s `result_summary` gains
    `stores_queued`; `stage_stats._count_graphiti_queued` reads it or the legacy `stores` list; the
    seed-guardrails report gains a queued-stores column beside `stores_written_by_name`. `mem0_write_state`
    is untouched.
  - On the size: 14 files; the test files are one-assertion edits.
  - Deletes: the `report as written` append and comment; the `Memory queued for [...]` f-string.
  - Signal: boundary row 4; `mcp__fused-memory__add_memory` against a dev server answers
    `stores_queued: ["graphiti"]`.
  - Consumer: every MCP client; `_REPORTED_TO_CALLER`; recon Stage 1's `graphiti_writes_queued`.
  - Deps: α.
  - G7: INV-2 holds (the outcome is a structured list, not prose); INV-9 (one renderer for the message, one
    Mem0 reading).
- **γ0 — Every operator entry point constructs MemoryService through `maintenance/_utils.py::maintenance_service`** (S26-4: A)
  - Size and keys: ~450 LOC. Key `fk-017442f273d2`. Priority medium.
  - Files: `fm/maintenance/_utils.py`, `fm/maintenance/task_family_census.py`,
    `fused-memory/tests/test_maintenance_utils.py`, `fused-memory/tests/test_task_family_census.py`, and the 30
    script files that construct a service (measured at `e634cfe658`: 29 matched by `(=|return)
    *([A-Za-z_]+\.)?MemoryService\(` plus `memory_eval_write_after_miss.py`'s `service=MemoryService` default):
    `fused-memory/scripts/{amend_stale_resume_cwd_records, audit_duplicate_memories,
    backfill_entity_standing_decision, bake_off_storage_shape, calibrate_write_triage, cleanup_count_snapshots,
    cleanup_pin_queue_edges, clear_false_dependency_invalidations, clear_malformed_empty_memory,
    consolidate_namespace_families, eval_write_triage_judge, eval_write_triage_reranker,
    freeze_write_triage_population, invalidate_fabricated_shipping_edges, investigate_cross_graph_duplication,
    memory_eval_retrieval_probe, memory_eval_staleness_sweep, memory_eval_write_after_miss,
    migrate_cross_graph_leak, normalize_topic_slugs, prune_recon_cycle_summaries, purge_knowlive_namespace,
    read_transform_selection, repair_recon_citation, retro_stamp_topics, sweep_orphan_flag_markers,
    sweep_toolcall_xml_leak, tag_cgl_eta_rehome_scope}.py`, `scripts/check_consolidation_closure.py`,
    `scripts/sweep_uuid_prefixes.py`.
  - On the size: 34 files, crossing the 15-file trigger deliberately — a uniform substitution
    (`MemoryService(cfg)` + `initialize()`/`close()` → `async with maintenance_service(None, config=cfg) as
    (cfg, service)`), the S25 δ precedent.
  - Work: `maintenance_service(config_path=None, *, config=None, open_as=ServiceOpen.FULL)` and the
    `ServiceOpen` enum (C3). The `config=` overload serves `bake_off_storage_shape.py` and
    `read_transform_selection.py`, which rewrite `config.queue.data_dir` (and, from γ,
    `config.reconciliation.data_dir`, so the journal follows the queue into their tmp dir) before
    construction. `ServiceOpen.READS` serves `memory_eval_retrieval_probe.py` and
    `memory_eval_write_after_miss.py`, which open with `initialize_for_reads()` today, and
    `task_family_census`'s CLI (task 6835's fix: no Graphiti startup maintenance, no queue). `ServiceOpen.NONE`
    serves `scripts/check_consolidation_closure.py`, whose `build_memory_service` deliberately never awaits
    `initialize()` (a read-only check that needs neither FalkorDB/Graphiti nor an embedder /
    `OPENAI_API_KEY`). Each script keeps its own error mapping around the context manager's entry (e.g.
    `memory_eval_retrieval_probe.py` maps `OSError`/`sqlite3.Error` from the opening to `ProbeRunError`;
    `memory_eval_write_after_miss.py::open_memory` keeps its injectable factory for its tests, defaulting to
    one that enters `maintenance_service`). No behaviour change except the shared close and the census's
    read-only opening.
  - Deletes: 29 script-side `MemoryService(` constructions (2 of them spelled `_service.MemoryService(`), the
    `service=MemoryService` default, and their hand-rolled `close()` finally blocks.
  - Signal: `(=|return) *([A-Za-z_]+\.)?MemoryService\(` and `= MemoryService,` are absent under
    `fused-memory/scripts/` and `scripts/` (29 + 1 lines in 30 files today, measured at `e634cfe658`); the
    scripts' tests pass unchanged; boundary row 5c's census case.
  - Consumer: γ (one site to change), every operator; the census CLI's operator.
  - Deps: β (service chain), θ2 (same script files). Absorbs 6835 (cancel).
  - G7: INV-9 (one construction site for operator paths); heuristic 12 (one opening enum, not two flags).
- **γ — MemoryService constructed with its collaborators; an explicit start() after wiring**
  - Size and keys: ~1,300 LOC (≈400 in `fm/`, ≈900 across 77 test files as one-line factory edits).
    Key `fk-017442f273d2`. Priority medium.
  - Files: `fm/services/memory_service.py`, `fm/services/durable_queue.py`, `fm/server/main.py`,
    `fm/maintenance/_utils.py`, `fm/reconciliation/context_assembler.py`, `fused-memory/tests/conftest.py`
    (`build_memory_service`), `fused-memory/tests/test_memory_service_lifecycle.py` (new),
    `fused-memory/tests/test_e2e_durable_queue.py`, `fused-memory/tests/test_journaling_integration.py`,
    `fused-memory/tests/test_maintenance_utils.py`, `fused-memory/tests/test_memory_service_initialize_for_reads.py`,
    `fused-memory/tests/test_overreach_shadow_wiring.py`, and the 77 test files that construct `MemoryService(`
    or call a deleted setter (measured at `e634cfe658`; the architect re-greps).
  - On the size: this crosses the 15-file review trigger deliberately; it is one uniform substitution
    (`MemoryService(cfg)` + attribute wiring → `build_memory_service(cfg, graphiti=…, mem0=…)`), the S25 δ/ε
    precedent.
  - Work: C3, D4, D5. `DurableWriteQueue.start()` and the `_ensure_workers` gate; `MemoryService.start()`;
    `MemoryService.__init__` constructs the queue and propagates `known_projects` to the storm escalator;
    `planned_registry=` keyword (the `PlannedEpisodeRegistry` import moves to module scope and the three
    dotted-path patches of task 8660 move onto the keyword); `main.py::run_server` builds the journal, project
    registry, recon journal and ledger first, constructs, wires `set_event_buffer`, runs
    `_recover_mem0_intents_at_startup` → `prune_every_table_at_startup`, then `start()`s, then starts the Mem0
    retry loop; `maintenance_service`'s body constructs `WriteJournal(...)` (awaiting `journal.initialize()` for
    `FULL` and `READS`) and `build_known_projects_map(primary_root='')`, and its docstring states the D4
    consequence; the taskmaster-absent branch logs once.
  - Deletes: 25 `if self._write_journal` guards (measured at `e634cfe658`, including
    `retry_due_mem0_intents`' and `check_mem0_retry_backlog`'s); the `MemoryService.write_journal` property's
    `None` arm; `context_assembler.py`'s "None means unwired" handling; `set_write_journal`,
    `set_known_projects`, `set_recon_journal`, `set_recon_ledger`, `set_planned_registry`; the two "BOUND
    METHOD … resolved at CALL time" docstrings and the two "Bound methods, not self" comments; the
    'permissive' INFO line (replaced by one construction-time log); the
    `assert memory_service.durable_queue is not None  # set by initialize()`.
  - Signal: boundary rows 5–7 (5b, 5c included); `journalctl -u fused-memory` shows `MemoryService started`
    after the `Task backend:` line.
  - Consumer: δ (journal always present), ε (collaborators injected), μ (backend injection for tests), τ/υ
    (collaborators built from the constructor), 6293, 6950, S10 κ1/μ2.
  - Deps: γ0, θ (both edit `conftest.py`). Absorbs 8660 (cancel).
  - G7: INV-11 (taskmaster absence logged; event-buffer wiring after start raises); INV-6
    `status-matches-liveness` holds — `start()` is the one place in-flight rows flip to pending, under the
    process that will run them; INV-7 — a pre-start enqueue is a hold whose age `get_queue_stats` reports.
- **δ — One `journaled_op` context manager for every Layer-1 write**
  - Size and keys: ~650 LOC, net negative. Key `fk-d37d63bf0d11`. Priority medium.
  - Files: `fm/services/journaled_ops.py` (new; `journaled_op`, S26-5: B in the home S26-2 set),
    `fm/services/memory_service.py`, `fm/services/queued_writes.py` (per-fact `write_op_id` at enqueue),
    `fm/services/durable_queue.py` (the prose saying `mem0_classify_and_add` rows carry no `_write_op_id`),
    `fused-memory/tests/test_journaled_ops.py` (new), `fused-memory/tests/test_journaling_integration.py`,
    `fused-memory/tests/test_write_journal.py`, `fused-memory/tests/test_memory_service.py`.
  - Work: D6; the 19 sites; queue-executor sites pass the carried id; `search` uses `op.fail` when degraded.
  - Deletes: 187 finally-block lines; 9 `success = True`; 7 `try/except Exception` wrappers around
    `log_write_op`; per-method `write_op_id = uuid4().hex` minting (the carried-id fallback
    `write_op_id or uuid4` included).
  - Signal: boundary row 8 — a parametrised test over every journaled operation's failure path asserts the
    `write_ops` row; today no such row exists for `update_edge`, `delete_memory`, `update_memory`,
    `delete_episode`, `redact_episode_content`.
  - Consumer: an operator querying `write_ops`; λ, μ (new mutators use it); maintenance (D11); τ (completes
    the module).
  - Deps: γ. Edges requested: 6293 (through ν) and 6636 depend on δ (the last in-batch editor of
    `durable_queue.py`).
  - G7: INV-10 holds (the test drives the public mutators); INV-11 (a failed or cancelled op is journaled as
    failed).
- **ε — EpisodeIdentityReconciler owns the nine identity passes behind run()**
  - Size and keys: ~2,400 LOC moved/changed (≈1,500 in `fm/`, ≈900 in tests). Keys `fk-abc3487a3528`,
    `fk-da5b47d0493a`, `fk-343c4a42be6b`, `fk-50cc281018fb`, and `fk-ec3de5b6cf2b` (task 6601, subsumed).
    Priority **high**.
  - Files: `fm/services/episode_identity.py` (new), `fm/services/memory_service.py`,
    `fm/utils/referent_verification.py` (prose retarget), `fm/backends/graphiti_client.py` (prose retarget),
    `fused-memory/scripts/audit_wrong_binding_edges.py` (prose retarget),
    `fused-memory/tests/test_episode_identity.py` (new), `fused-memory/tests/test_memory_service.py`,
    `fused-memory/tests/test_referent_repair.py`, `fused-memory/tests/test_referent_verification.py`,
    `fused-memory/tests/test_referent_queue_threading.py`, `fused-memory/tests/test_write_time_identity.py`,
    `fused-memory/tests/test_temporal_guards.py`, `fused-memory/tests/test_dependency_direction_check.py`,
    `fused-memory/tests/test_referent_repair_storm_escalator.py`, `fused-memory/tests/_fm_helpers.py`.
  - On the size: above the 1,500 target because the extraction and the test migration cannot land apart
    (the tests reference the moved names). The move is mechanical (the measured 19-method closure plus the
    module-level identity symbols); the design work is C4 and the test rewrite. Not split further on purpose:
    a half-moved chain is the stitching heuristic 14 forbids.
  - Work: C4, D7; `RepairOutcome` StrEnum (6601); `MemoryService.__init__` builds the reconciler from its
    constructor collaborators; `_execute_graphiti_write` calls `record_referent_source` before `add_episode`
    and `run(…, ambiguous=<decoded ReferentSet | None>)` after, and reads `ReconcileStats`. `run()` raises
    `IdentityLockNotHeld` unless `graphiti.identity_lock_held(group_id)` (D8; λ has landed). The reconciler
    mutates through its backend under the caller's lock and never calls a service mutator (C4). The new
    module's docstring states its purpose in one sentence; per-pass docstrings keep behaviour, not history
    (D13; ζ finishes the verify pass). The three files citing moved symbols as `memory_service.py::…` are
    retargeted (D13).
  - Deletes: `_reconcile_episode_identity` and its `_run_pass` closure, the nine passes and their nine
    closure helpers from `memory_service.py`; `ReconcileStats`/`ReferentRepair`/`ReferentRepairStats`,
    `_episode_uuid_of`, `_elapsed_ms_since`, `_is_dependency_fact`, `_priority_override_*`, their regexes
    from `memory_service.py`; the 8 repeated write-result extractions; `MemoryService.referent_*_counts`;
    `REFERENT_REPAIR_OUTCOMES`; the private test references to the passes (≈450; `_verify_episode_referents`
    alone 140 in 4 test files, measured at `e634cfe658`).
  - Signal: boundary rows 9 and 10; `memory_service.py` loses ≈3,300 lines; `test_memory_service.py` and
    `test_referent_verification.py` name no `_verify_episode_referents` (delivered check). The module-total
    cognitive complexity of `memory_service.py` + `episode_identity.py` is ≤ 1,087 (`memory_service.py`'s
    total measured at `e634cfe658`; no complexity added — `docs/code-quality.md` §What to measure).
  - Consumer: `_execute_graphiti_write`; ζ, λ (D8), ξ (predicate import); the MCP readers of the counters
    via `MemoryService.identity.counts` if any appear (none today).
  - Deps: λ (the lock is public before the reconciler asserts it; also serializes `memory_service.py` and
    `graphiti_client.py`).
  - G7: INV-11 (pass failures recorded in stats); INV-5 (one write-result extraction); INV-9 (counters have
    one home); INV-4 (the referent-source denominator still counts every write). No waiver.
- **ζ — The verify pass as four methods; its narration rehomed**
  - Size and keys: ~500 LOC, net negative. Key `fk-41eaf65cc4d0`. Priority medium.
  - Files: `fm/services/episode_identity.py`, `fused-memory/tests/test_episode_identity.py`,
    `fused-memory/tests/test_referent_verification.py`.
  - Work: `_verify_episode_referents` → `_ambiguity_set`, `_index_nodes`, `_walk_edge_endpoints`,
    `_verification_findings`, each ≤ 25 cognitive complexity and ≤ 60 lines; the `RUNS LAST … ordering is
    load-bearing` constraint is `run()`'s order; the 142-line docstring becomes one paragraph plus a pointer
    to §Rationale rehomed; `_repair_episode_referents` (complexity 16) likewise for its prose.
  - Deletes: ≈386 comment-only and ≈240 docstring lines; every `leaf <greek>` and `Resolved decision N`
    string in `episode_identity.py` (the 12 lines ε moved; η deletes the 6 left in `memory_service.py`).
  - Signal: complexipy on `episode_identity.py` reports each of the four successors of
    `_verify_episode_referents` ≤ 25 and the module max ≤ 45 — the untouched
    `_invalidate_stale_superseded_ttl_edges` (45); `_repair_edge_findings` (39) and `_cleanup_emptied_nodes`
    (26) are also untouched, measured at `e634cfe658`. The referent verification fixtures pass unchanged.
  - Consumer: the next agent changing a verification rule (heuristic 4).
  - Deps: ε.
  - G7: INV-9 (rationale has one home: this PRD).
- **η — search as fan-out / fuse-and-filter / pin / journal; the pin a pure function in topic_anchor**
  - Size and keys: ~700 LOC. Key `fk-eb0d5fea9a52`. Priority medium.
  - Files: `fm/services/memory_service.py`, `fm/services/topic_anchor.py`,
    `fused-memory/tests/test_topic_anchor.py`, `fused-memory/tests/test_memory_service.py`,
    `fused-memory/tests/test_search_topic_anchor.py` (new or merged into test_topic_anchor).
  - Work: `apply_topic_anchor(results, *, limit, lookup, include_planned, categories) -> list[MemoryResult]`
    owns the two-tier exception band and the local-copy commit; `resolve_topic_anchor_enabled(reconciliation_cfg)`
    takes the config object and is still evaluated on every `search` call, never captured at construction
    (the knob is green-tier hot-reloadable, `config/reload.py`); `_ANCHOR_SCROLL_LIMIT`/`_MAX_ANCHOR_TOPICS`
    stay private to `topic_anchor`; `search` becomes four calls in order.
  - Deletes: the 220-line inline pin block (149 comment lines); the two underscore imports; the
    `PLACEMENT IS LOAD-BEARING` header; the `leaf iota` narration left in `memory_service.py` after ε (6
    lines: module level, `__init__` ×2, `_execute_graphiti_write`, `add_episode`, `add_memory` ×2, …).
  - Signal: boundary row 11; `search` cognitive complexity ≤ 15 and each step ≤ 25 (complexipy; `search` is
    81 at `e634cfe658`), the module total not above today's; `leaf iota` absent from `memory_service.py`.
  - Consumer: every `search` caller (12 call sites, task 4656's census); write triage (task 5802 cancelled —
    unchanged `anchor_topics=False` opt-out preserved).
  - Deps: ε.
  - G7: INV-11 (fail-open arm still logs at WARNING with the topic set; the typed re-raise band preserved).
- **θ — GraphAccess protocol with bounded egress; GraphitiBackend takes it at construction**
  - Size and keys: ~1,000 LOC. Keys `fk-9430031679c4` (backend half), `fk-90f6fbb5daa0`. Priority **high**.
  - Files: `fm/backends/graph_access.py` (new), `fm/backends/graphiti_client.py`,
    `fm/backends/falkor_read_only.py`, `fm/maintenance/cross_graph_move.py`, `fm/config/schema.py`
    (`FalkorDBProviderConfig.socket_timeout_seconds`), `fused-memory/scripts/measure_plural_enum_guard_recall.py`
    (its `_paged_ro_query` import), `fused-memory/tests/conftest.py` (`FakeGraphAccess`),
    `fused-memory/tests/test_graph_access.py` (new), `fused-memory/tests/test_graphiti_backend_with_driver.py`,
    `fused-memory/tests/test_falkor_read_only.py`, `fused-memory/tests/test_cross_graph_move.py`,
    `fused-memory/tests/test_graph_read_pagination.py` and `fused-memory/tests/test_measure_plural_enum_guard_recall.py`
    (both import or identity-assert `graphiti_client._paged_ro_query`, which moves).
  - On the size: 13 files, one over the band; the two pagination test files follow the moved symbol.
  - Work: C5, D9. The 47 raw Cypher lines (22 `graph.query`, 23 `graph.ro_query`, 2 inline `_graph_for(...)`,
    measured at `e634cfe658`) call `self.graph_access.graph(gid).query/ro_query(..., op=<method name>)`;
    `paged_ro_query` moves; `GraphitiBackend.graph_access` property; `_graph_for` is re-implemented as
    `return self.graph_access.graph(group_id)` and `_require_falkor_client` is kept, both only until θ2;
    `cross_graph_move`'s public functions **keep** their `graphiti: GraphitiBackend` parameter (D11) and its
    16 `_graph_for` + 5 `_require_falkor_client` calls resolve through `graphiti.graph_access.graph(gid)` /
    `.ro_query_compact(...)`; the connection gets `socket_timeout`.
  - Deletes: `_graph_for`'s `driver._get_graph` body (the reach moves into `FalkorGraphAccess`);
    `falkor_read_only.py`'s upward import of `_paged_ro_query`; the 45 unbudgeted raw Cypher lines (47 minus
    the 2 under `asyncio.wait_for`) as unbudgeted calls.
  - Signal: boundary rows 12–13; against a dev FalkorDB with `GRAPH.CONFIG SET TIMEOUT 1`, a slow `get_entity`
    raises `TimeoutError('Graphiti get_entity timed out after 30.0s')`-shaped errors from every method, not
    `[]`.
  - Consumer: θ2, ι, κ (tests), λ, ξ, μ (cross_graph_move callers), `falkor_read_only`.
  - Deps: 6631, 6823 (out of batch; S26-6: B; 6369 is done); 6233 landed. Both are low; the scheduler's
    priority inheritance through the dependency graph lifts them behind high θ (no override requested).
  - G7: INV-11 (every timeout is raised, none swallowed); INV-2 (the error names op and budget);
    INV-1 (the egress contract is a Protocol pyright checks, not prose).
- **θ2 — Operator scripts and their tests onto `GraphitiBackend.graph_access`**
  - Size and keys: ~400 LOC, 14 files. Key `fk-9430031679c4`. Priority medium.
  - Files: `fm/backends/graphiti_client.py` (delete the shim),
    `fused-memory/scripts/{cgl_eta_auto_apply_impl, consolidate_namespace_families,
    invalidate_fabricated_shipping_edges, investigate_cross_graph_duplication, measure_plural_enum_guard_recall,
    migrate_cross_graph_leak, purge_knowlive_namespace}.py`, `fused-memory/tests/{test_consolidate_namespace_families,
    test_migrate_cross_graph_leak, test_purge_knowlive_namespace, test_investigate_cross_graph_duplication,
    test_cgl_eta_auto_apply_impl, test_reindex}.py`.
  - Work: every script `backend._graph_for(gid)` → `backend.graph_access.graph(gid)`, every
    `_require_falkor_client` → `ro_query_compact`; the scripts' tests stop assigning the instance attribute
    (`backend._graph_for = …`; 24 + 16 + 14 + 10 + 1 sites) and inject `FakeGraphAccess` through
    `with_driver(..., graph_access=)`. Without this leaf θ would land green with the scripts raising
    `AttributeError` in production, because their tests stub the attribute θ removes. `test_reindex.py`'s 4
    `patch.object(backend, '_graph_for', …)` sites (measured at `38a200d213`) move here too: `patch.object`
    raises `AttributeError` on an attribute the class no longer has, so they cannot outlive the shim; they
    inject `FakeGraphAccess` instead, and `test_delegates_to_graph_for` becomes a test that `node_count`
    resolves its graph through `graph_access`. ι keeps that file's `_driver._get_graph` stubs.
  - Deletes: `GraphitiBackend._graph_for`; the scripts' `_require_falkor_client` calls; the
    `# noqa: SLF001` in `measure_plural_enum_guard_recall.py` (INV-12); the 65 test-side `_graph_for` stubs in
    the 5 script-test files and the 4 `patch.object` sites in `test_reindex.py`; the prose
    `graphiti._graph_for(key).delete()` in `consolidate_namespace_families.py`'s module docstring (it matches
    the θ2 check).
  - Signal: `def _graph_for` absent from `graphiti_client.py`; `\._graph_for\(|\._require_falkor_client\(`
    absent under `fused-memory/scripts/`; `\._graph_for = |patch\.object\([^)]*_graph_for` absent under
    `fused-memory/tests/`; the 6 suites pass.
  - Consumer: operators running the cross-graph scripts; γ0 (same script files); ι.
  - Deps: θ.
  - G7: INV-12 (the suppression is removed, not re-owned); Tests stance (fakes via the public constructor).
- **ι — Tests on the public graph seam, cluster 1 (the heavy files)**
  - Size and keys: ~500 LOC. Key `fk-9430031679c4`. Priority medium.
  - Files: `fused-memory/tests/test_reindex.py` (25 driver sites; θ2 already moved its 4 `patch.object`),
    `test_cross_graph_move.py` (48), `test_refresh_entity_summary.py` (36), `test_write_time_identity.py`
    (25), `test_merge_entities.py` (22), `test_delete_entity.py` (13), `_fm_helpers.py` (counts measured at
    `e634cfe658` over `_driver._get_graph|._graph_for = |patch.object(…_graph_for|._driver = `).
  - Work: every `._driver._get_graph =` and `._driver =` in these files becomes
    `make_backend(cfg, graph_access=FakeGraphAccess({gid: graph}))` or `make_graph_driver`; an
    `_get_graph.assert_called_with(gid)` becomes wiring only that gid. (No `._graph_for =` assignment is left
    in these files: θ2 removes all 65 and the 4 `patch.object`.)
  - Deletes: ≈140 `_driver._get_graph` stubs (160 lines in these 7 files at `38a200d213`).
  - Signal: `_driver\._get_graph` absent from these 7 files; the suites pass unchanged in behaviour.
  - Consumer: the next change to how a graph handle is obtained (the finding's next_change).
  - Deps: θ2 (owns the script tests and `test_reindex.py`'s `patch.object` sites, so it edits that file
    first), ε (edits `test_write_time_identity.py`/`_fm_helpers.py`). Absorbs
    7781 (cancel).
  - G7: Tests stance (the point of the leaf).
- **κ — Tests on the public graph seam, cluster 2 (the long tail)**
  - Size and keys: ~600 LOC. Key `fk-9430031679c4`. Priority medium.
  - Files: the remaining files of the 51 (≈38, 1–12 stubs each; the architect re-greps
    `_driver\._get_graph|\._driver = |_MultiTenantFalkorDriver|_paged_ro_query` under `fused-memory/tests`),
    including `test_valid_edges_uuid_dedup.py`, `test_retrieve_episodes_bounded_read.py`,
    `test_reassign_edge.py`, `test_cleanup_stale_edges.py`, `test_startup_identity_scan.py`,
    `test_graphiti_group_arg_canonicalization.py`, `test_falkor_indices.py`, `test_rebuild_entity_summaries.py`,
    `test_memory_service.py` (9), and `fused-memory/tests/conftest.py` (the `make_graph_driver` docstring).
  - On the size: >15 files, each a one- to twelve-line uniform substitution (S25 δ precedent).
  - Work: as ι; the `_MultiTenantFalkorDriver` monkeypatches and the `_paged_ro_query` patches move onto
    `make_graph_driver` / `FakeGraphAccess`; conftest's own `_MultiTenantFalkorDriver` import is the one
    sanctioned private import left, and `make_backend_over_fake_graphiti`'s `_client_for` stub is decided
    here (task 7781's addendum: a per-group-client seam or an explicit `# debt:` owner);
    `conftest::make_graph_driver`'s docstring stops spelling `_driver._get_graph`.
  - Deletes: the remaining `_driver._get_graph` stubs (≈100 of the 241), `._driver =` (23), the private patches.
  - Signal: `grep -rE '_driver\._get_graph|\._driver = ' fused-memory/tests` is empty.
  - Consumer: as ι.
  - Deps: ι, η (both edit `test_memory_service.py`).
  - G7: INV-12 — any private read kept carries a `# debt: task <id>` owner.
- **λ — The identity lock as a public context manager held by every in-process identity mutator**
  - Size and keys: ~550 LOC. Key `fk-9804ecc4567d` (in-process half). Priority **high**.
  - Files: `fm/backends/graphiti_client.py`, `fm/services/memory_service.py`,
    `fm/maintenance/cross_graph_move.py`, `fm/server/tools.py` (render `IdentityLockBusy`),
    `fm/config/schema.py` (`IdentityLockConfig`), `fm/config/reload.py` (green tier), `fused-memory/config/config.yaml`
    (the lock-timeout comment), `fused-memory/tests/test_write_time_identity.py`, `fused-memory/tests/test_merge_entities.py`,
    `fused-memory/tests/test_delete_entity.py`, `fused-memory/tests/test_reassign_edge.py`,
    `fused-memory/tests/test_cross_graph_move.py`, `fused-memory/tests/test_entity_mint_service.py`,
    `fused-memory/tests/server/test_entity_mint_authz_gate.py`.
  - On the size: 14 files, over the band by the config triple and the wire rename's two test files.
  - Work: C6, D8 (S26-7: C). `identity_lock(group_id, *, timeout)` wraps the bounded acquire
    `ensure_entity_node` does today, records ownership in a `ContextVar` and raises `IdentityLockReentry` on
    re-entry; its docstring states the child-task rule (D8); `_execute_graphiti_write` (timeout `None`),
    `ensure_entity_node`, `merge_entities`, `rename_entity`, `delete_entity`, `reassign_edge` (timeout
    `identity_lock.acquire_timeout_seconds`) and `cross_graph_move`'s 7 public mutators (`create_moved_node`,
    `move_entity_across_graphs`, `recreate_subgraph_relationships`, `delete_source_node`, `create_moved_episode`,
    `delete_source_episode`, `merge_foreign_duplicate`; in-process only; every group the call touches
    (`recreate_subgraph_relationships(graphiti, specs)` spans any number), in `sorted()` order) hold it.
    `IdentityLockBusy` is rendered by the MCP tools as a structured refusal with `error_type:
    'IdentityLockBusy'`; `ensure_entity_node`'s wire value `EntityMintLockBusy` changes to it, and
    `entity_mint.lock_timeout_seconds` folds into the new field (OQ-3). The private `_identity_lock_for`
    registry stays under the public context manager; the 50 test references to it in 10 files (measured at
    `e634cfe658`) keep working.
  - Deletes: `self.graphiti._identity_lock_for` (2 service sites + the `wait_for(lock.acquire())` composition);
    the 5 "MUST hold" docstring clauses (replaced by the context manager); the lock-privacy clause of 6730's
    problem 1; the `EntityMintLockBusy` spelling (3 `fm/` modules).
  - Signal: boundary rows 10b, 10c and 14 (row 10 is ε's: `run()` does not exist until ε);
    `grep -rn '_identity_lock_for' fm/services fm/maintenance` is empty.
  - Consumer: ε (D8, row 10), μ (`delete_entity` locked before the sweep uses it), σ (designs on top of it), 6730.
  - Deps: θ, δ.
  - G7: INV-7 `holds-owned-and-bounded` — every MCP and reconciliation wait is bounded by
    `identity_lock.acquire_timeout_seconds`, the one unbounded holder is named, and the refusal names the
    group; INV-2 (`IdentityLockBusy` carries group and wait).
- **σ — Human design session: cross-process identity exclusion between the fused-memory server and operator scripts** (S26-3)
  - Form: `task_kind='deterministic'`, `always_escalates=true`, no `before_done` — the pure gate of
    `docs/task-authoring.md` §5 (a human gate under the one rule S10-6 ruled: deterministic and
    always-escalating). No code. Key `fk-9804ecc4567d` (out-of-process half) in `metadata.x_finding_key`,
    with the program's usual stamps. Priority **high**.
  - Files: none of its own. Its description points at the committed brief
    `plans/fm-cross-process-identity-exclusion-session-brief.md` (text: Appendix A; the lead commits it in the
    wave-2 commit, before filing σ — lead ruling, 2026-11-10), not at an escalation or a memory.
  - Work: when λ is done, the gate files a born-at-L2 escalation asking Leo to `/spawn` an interactive
    session running `/team --fable` that investigates, discusses candidate designs with him and answers every
    open question in the brief before any PRD is written; the session commits a PRD through `/prd` (whose
    tasks are filed by `/prd`) and resumes the gate with that PRD's path. It is not an architect task. It
    settles task 6293's ownership question in the same sitting (the brief's question 7).
  - Close: `done_provenance` `deterministic-gate` citing the resolving escalation, after `resume`.
  - Signal: the committed PRD's path in the resume; its two-process test (a second process's
    `cross_graph_move` waits or refuses with `IdentityLockBusy` and never interleaves) is that PRD's signal,
    not this plan's.
  - Consumer: Leo; the session's PRD; 6293; gate 6698 (the next review waits on this sitting).
  - Deps: λ only (lead ruling, 2026-11-10). Task 6730 must not depend on σ. 6293 depends on σ.
  - G7: INV-7 (the exclusion the session designs must be owned and bounded; a brief constraint).
- **μ — Backends private; the sweeps' read surface; every graph mutation through the service**
  - Size and keys: ~850 LOC. Key `fk-6e50b22357f2`. Priority medium.
  - Files: `fm/services/memory_service.py`, `fm/backends/graphiti_client.py` (`delete_entity_node` →
    private), `fm/backends/mem0_client.py` (public raw-point scroll for ν), `fm/services/id_resolver.py`,
    `fm/reconciliation/degenerate_task_node_sweep.py`, `fm/reconciliation/stale_status_snapshot_edge_sweep.py`,
    `fm/reconciliation/stale_priority_override_edge_sweep.py`, `fm/reconciliation/stages/task_knowledge_sync.py`,
    `fm/reconciliation/standing_decision_writer.py`, `fm/reconciliation/stages/memory_consolidator.py`,
    `fm/server/tools.py`, `fm/maintenance/cleanup_stale_edges.py`, `fm/maintenance/reindex.py`,
    `fm/maintenance/task_family_census.py`, `fm/maintenance/verify_zombie_edges.py`,
    `fused-memory/tests/reconciliation/test_degenerate_task_node_sweep.py`,
    `fused-memory/tests/test_memory_service_read_surface.py` (new; includes the boundary-16 AST scan),
    `fused-memory/tests/test_merge_entities.py`, `fused-memory/tests/test_delete_entity.py`, plus the tests of
    the repointed modules (including the 4 files with one `delete_entity_node` reference each:
    `test_graphiti_group_arg_canonicalization.py`, `test_referent_repair.py`, `test_task_family_census.py`,
    `test_audit_wrong_binding_edges.py`).
  - On the size: 16 `fm/` files; each repoint is small and uniform; the design is C7. The reach-through
    sites are unchanged since `b688221a04` (re-measured; tasks 6120 and 6375 restructured
    `memory_consolidator.py` and `stale_status_snapshot_edge_sweep.py` but left their reach lines in place).
  - Work: C7, D11. `self.graphiti`/`self.mem0` → `self._graphiti`/`self._mem0` (87 lines, measured at
    `e634cfe658`; tests already inject via `build_memory_service`); the degenerate sweep calls
    `delete_entity`, counts `IdentityLockBusy` skips in its stats with a streak escalation (INV-4), and retries
    them next cycle; `delete_entity_node` becomes `_detach_delete_entity_node`, still the primitive under
    `merge_entities` and `delete_entity`; maintenance modules take `graphiti: GraphitiBackend` (reads) and
    `memory_service: MemoryService` (mutations) parameters, and their raw writes with no service counterpart
    are wrapped in `journaled_ops.journaled_op` (S26-5: B).
  - Deletes: the public `GraphitiBackend.delete_entity_node`; 8 direct + 6 alias reach-through sites in
    `fm/`; the `bulk_remove_edges` bypasses (through `remove_edges`).
  - Signal: boundary rows 15, 15b, 16; `write_ops` shows a `delete_entity` row after a degenerate sweep on a
    dev server.
  - Consumer: the sweeps and stages; ν; τ (moves `remove_edges` with the graph mutators); S15 (health read).
  - Deps: ξ (the service chain and the graph chain both end in ξ; it also orders μ after ι for the two
    shared test files).
  - G7: INV-10 (the AST guard is exercised against a seeded fixture); INV-3 holds (`delete_entity`'s
    existence and active-edge checks are the corroboration the sweep's `edge_count==0` pre-read lacked);
    INV-4 (lock-busy skips streak-escalate).
- **ν — Scripts stop reaching through MemoryService's backends** (S26-1: A)
  - Size and keys: ~500 LOC. Key `fk-6e50b22357f2`. Priority medium.
  - Files: the 16 scripts measured at `e634cfe658` (`memory_eval_staleness_sweep`, `cleanup_count_snapshots`,
    `prune_recon_cycle_summaries`, `freeze_write_triage_population`, `clear_malformed_empty_memory`,
    `tag_cgl_eta_rehome_scope`, `audit_duplicate_memories`, `consolidate_namespace_families`,
    `investigate_cross_graph_duplication`, `purge_knowlive_namespace`, `migrate_cross_graph_leak`,
    `normalize_topic_slugs`, `invalidate_fabricated_shipping_edges`, `bake_off_storage_shape`,
    `read_transform_selection` under `fused-memory/scripts/`, and `scripts/sweep_uuid_prefixes.py`), their
    tests, and `fused-memory/tests/test_store_mutation_conformance.py` (the existing AST census pattern gains
    the "no `.graphiti.`/`.mem0.` reach on a service" rule).
  - On the size: 16 scripts, each a construction-site edit (S25 δ precedent).
  - Work: D11. Read-only scripts construct `Mem0Backend`/`GraphitiBackend` directly; mutating scripts keep
    `maintenance_service` and call the service's journaled mutators or wrap raw writes in `journaled_op`
    (S26-5: B; `tag_cgl_eta_rehome_scope.py`'s direct `mem0.update` is one); `_get_async_qdrant` callers use
    μ's public scroll.
  - Deletes: 3 `_get_async_qdrant` calls and the `# noqa: SLF001`; the `_get_async_qdrant` prose in
    `clear_malformed_empty_memory.py`'s module docstring and `consolidate_namespace_families.py`'s comment;
    16 scripts' `memory_service.graphiti`/`.mem0` reaches.
  - Signal: `grep -rE '(memory|service|svc|memory_service)\.(graphiti|mem0)\b' fused-memory/scripts scripts`
    is empty; `test_store_mutation_conformance.py`'s new rule passes the tree and flags its fixture.
  - Consumer: operators running the scripts; the conformance guard; 6293 (waits on ν and σ); 6950 (through υ).
  - Deps: μ. (No longer on 6293: 6293 now waits on ν and σ.)
  - G7: INV-12 (the `SLF001` suppression is removed, not re-owned).
- **ξ — One home for a RELATES_TO edge's stored meaning: the active predicate and the relocation rule**
  - Size and keys: ~550 LOC. Keys `fk-265caf7191a5`, `fk-62af616b8470`. Priority **high**.
  - Files: `fm/backends/relates_to_schema.py` (new), `fm/backends/graphiti_client.py`,
    `fm/maintenance/cross_graph_move.py`, `fm/services/memory_service.py`, `fm/services/episode_identity.py`,
    `fused-memory/tests/test_relates_to_schema.py` (new), `fused-memory/tests/test_relates_to_relocation_parity.py`
    (new; 5612 would have created it), `fused-memory/tests/test_reassign_edge.py`, `fused-memory/tests/test_merge_entities.py`,
    `fused-memory/tests/test_cross_graph_move_carried_edge_properties.py`.
  - Work: C8, D10, OQ-1 ruled (b) (lead, 2026-11-02, live probe on FalkorDB 4.18.0). `relocation_set_clause`
    renders `SET new += properties(old)` plus the `RELOCATION_REASSERTED_PROPERTIES` re-asserts (endpoint ids
    from the MATCHed new endpoints) and no `fact_embedding` re-assert — `+=` keeps Vectorf32, and
    `vecf32(old.fact_embedding)` errors on one. The per-site overrides (`uuid`, `superseded_edge_uuid`,
    `reassigned_from_node_uuid`) stay at the call sites as `overrides`. `cross_graph_move` uses 6680's
    `properties(e)` read with `fact_embedding` stripped from `$props`, then `SET r += $props` plus the
    `CROSS_GRAPH_EMBEDDING_REASSERT` rule (`cross_graph_embedding_reassert`, delegating to
    `_embedding_set_clause('r.fact_embedding', <compact reply>)`) — mandatory and byte-exact, from the
    `--compact` tokens, so node and edge embeddings keep one transport and the module docstring's lossy claim
    stands. Legacy List-typed embeddings stay List in-graph and are refused cross-graph as today; upgrading
    them is out of scope. The `reassign_edge` "preserve the vecf32 type" comment goes. The
    predicate pair replaces `_ACTIVE_EDGE_PREDICATE`, `_edge_is_retired` and the 4 inline spellings. S26-6
    (B): the `superseded_edge_uuid` keep-original rule and the `expired_at` docstring accuracy from 5612's
    widening.
  - Deletes: `graphiti_client._ACTIVE_EDGE_PREDICATE`; `memory_service._edge_is_retired`; the 4 inline
    `getattr(edge, 'invalid_at')` tests in the passes; `cross_graph_move._CARRIED_EDGE_PROPERTIES`;
    the four hand-written SET lists; `test_reassign_edge.py::_COPIED_PROPS` and the 2
    `new.expired_at = old.expired_at` substring pins (replaced by behaviour: boundary row 17).
  - Signal: boundary rows 17–18.
  - Consumer: every relocation and every active-edge read; the next property added to an edge.
  - Deps: ζ, η, κ (the service, identity and test chains; κ edits `test_reassign_edge.py` and
    `test_cross_graph_move_carried_edge_properties.py`). Absorbs 6680 and 5612 (cancels).
  - G7: INV-5 (one rule / one predicate); INV-10 (the parity test executes the relocation against a fixture
    graph rather than pinning Cypher text).
- **ο — canonical_labels' five patterns from one grammar**
  - Size and keys: ~400 LOC, net negative. Key `fk-352d7b3fcad4`. Priority medium.
  - Files: `fm/utils/canonical_labels.py`, `fused-memory/tests/test_canonical_labels.py`,
    `fused-memory/tests/test_task_naming.py`, `fused-memory/tests/test_cross_project_refs.py`.
  - Work: D12. Fragments `_TASK_WORD` (exists), `_PAD` (`[ \t]*`), `_SEPARATOR` (`(?:[ \t]*[#:][ \t]*|\s+)` and
    its mention variant), `_NUMBER` (`([0-9]+)`), `_NUMBER_END` (`(?!\d)`), `_QUALIFIER`
    (`[A-Za-z][A-Za-z0-9_-]{2,}`), the two left-edge exclusion sets, composed by one `_compile(*parts, anchored)`;
    `_TASK_VOCABULARY_QUALIFIER` stays lowercase-only. The private names tests reach (`_canonical_allowlist`,
    `_KIND_LABELS` — 3 sites) get public spellings or the tests move to behaviour.
  - Deletes: the duplicated qualifier class (3→1), separator (2→1), digit capture (4→1), right edge (2→1);
    ≈600 of 698 prose lines (the measured-case rationale moves into the parametrised tests' ids and this
    PRD's §Rationale rehomed).
  - Signal: boundary row 19; `complexipy`/line census: ≤ 250 lines, prose share < 40%.
  - Consumer: `referent_resolution`, `referent_verification`, `task_naming`, `cross_project_refs`,
    `entities_gate`, `entity_mint_authz`, `memory_service`/`episode_identity` (7 modules); task 6826.
  - Deps: none. Edge requested: 6826 depends on ο.
  - G7: INV-5 (one spelling per rule); INV-9 (rationale rehomed once).
- **ρ — Move the write-metadata policy and store-codec helpers out of memory_service.py** (S26-2)
  - Size and keys: ~1,600 LOC moved or retargeted, mechanical. Keys `fk-da5b47d0493a`, `fk-343c4a42be6b`,
    `fk-50cc281018fb`. Priority medium.
  - Files: `fm/services/memory_service.py`, `fm/services/memory_metadata_policy.py` (new),
    `fm/services/store_records.py` (new), `fm/server/grouped_read.py`, `fm/server/tools.py`,
    `fm/reconciliation/preservation_specimen_guard.py`, and the 18 test files that reference a moved private
    (measured at `e634cfe658`: `test_memory_service.py` 111, `test_summary_pool.py` 5,
    `reconciliation/test_preservation_specimen_guard.py` 4, `test_retro_stamp_topics.py` 4,
    `test_mem0_client.py` 3, and 13 files with 1–2 each, among them `server/test_grouped_read.py`,
    `test_memory_metadata.py`, `test_normalize_topic_slugs.py`, `test_referent_resolution.py`,
    `test_update_memory_tool.py`; the architect re-greps).
  - On the size: above the band in files and LOC, as a pure move with uniform import retargets (the S25 δ
    precedent); no behaviour changes.
  - Work: C9, D15. `_infer_recon_pool` and the block `_missing_cycle_summary_keys` …
    `_apply_cycle_summary_metadata_tagging` (883 lines at `e634cfe658`) move to `memory_metadata_policy.py` as
    public functions; `_MEM0_CONTENT_KEYS`, `_mem0_content`, `metadata_row_text` and what remains of
    `_serialize_temporal` … `_store_failure_diagnostics` after α, ε and ξ (552 lines at `e634cfe658`) move to
    `store_records.py` as public functions; `grouped_read.py`'s private import and `tools.py`'s and
    `preservation_specimen_guard.py`'s `metadata_row_text` imports repoint; tests import the public names.
    Prose moves verbatim (D13); tracked citations of moved symbols are retargeted.
  - Deletes: the moved definitions from `memory_service.py`; `grouped_read.py`'s
    `from fused_memory.services.memory_service import _MEM0_CONTENT_KEYS`; ≈150 private test references.
  - Signal: boundary row 21; `memory_service.py` loses ≈1,400 lines; `_apply_memory_metadata_validation`
    and `_MEM0_CONTENT_KEYS` absent from `memory_service.py`.
  - Consumer: υ (imports the policy and codec instead of reaching back into `memory_service.py`); the
    importers named; the next metadata-rule change (one module).
  - Deps: ν (the last in-batch editor before the split, after μ; ν also edits `test_audit_duplicate_memories.py`
    and `test_normalize_topic_slugs.py`, two of ρ's test files).
  - G7: INV-9 (one home per policy rule); heuristic 13 (each new file makes sense alone); Tests stance (public
    names, not patched privates).
- **τ — Extract GraphMutations over a shared journaled-operations module** (S26-2)
  - Size and keys: ~1,700 LOC touched (≈1,500 moved, ≈200 new). Keys `fk-da5b47d0493a`, `fk-343c4a42be6b`,
    `fk-50cc281018fb`. Priority medium.
  - Files: `fm/services/memory_service.py`, `fm/services/journaled_ops.py`, `fm/services/graph_mutations.py`
    (new), `fused-memory/tests/test_graph_mutations.py` (new), `fused-memory/tests/test_memory_service_read_surface.py`
    (the boundary-16 allowlist), `fused-memory/tests/test_merge_standing_decision_invalidation.py`,
    `fused-memory/tests/middleware/test_entity_mint_storm_escalator.py`, `fused-memory/tests/test_entity_mint_service.py`,
    `fused-memory/tests/test_reassign_edge.py`, `fused-memory/tests/test_memory_service.py`,
    `fused-memory/tests/test_consolidate_memories_tool.py`, `fused-memory/tests/test_consolidation_ops.py`,
    `fused-memory/tests/test_delete_memory_truncated_uuid.py` (the last five reference `_journaled_backend_call`
    or `_emit_event`).
  - On the size: 13 files and above the LOC band; one cohesive move whose two halves (the layer and the
    collaborator) cannot land apart without a stitched intermediate.
  - Work: C9, D15. `MemoryService._journaled_backend_call` moves unchanged to `journaled_ops.journaled_backend_call`;
    `_emit_event`'s buffer becomes the one `EventSink` (`set_event_buffer` forwards to `EventSink.wire`);
    `GraphMutations` takes `update_edge`, `reassign_edge`, `ensure_entity_node` with its two mint helpers,
    `delete_episode`, `redact_episode_content`, `refresh_entity_summary`, `set_entity_summary`,
    `rename_entity`, `rebuild_entity_summaries`, `merge_entities`, `_expire_standing_decisions_for_merge`,
    `delete_entity` and `remove_edges`, plus the entity-mint storm state; `MemoryService.__init__` builds it
    and keeps one-line delegates for the public ones; boundary 16's allowlist gains
    `services/graph_mutations.py`; tests that reached the moved privates construct `GraphMutations` with a
    fake backend and journal; dotted-path patch targets of names the move relocates (e.g.
    `fused_memory.services.memory_service.emit_entity_mint_storm_escalation` in `test_entity_mint_service.py`)
    are retargeted to `graph_mutations`, since a patch on the old path would patch nothing.
  - Deletes: the moved methods' bodies from `memory_service.py` (≈1,500 lines); `MemoryService._journaled_backend_call`;
    `MemoryService._emit_event`'s buffer handling.
  - Signal: boundary row 22; `class GraphMutations` present, `async def _expire_standing_decisions_for_merge`
    absent from `memory_service.py`.
  - Consumer: every graph-mutation caller (unchanged, through the delegates); the next graph mutator added
    (one module); 6730 (its `ensure_entity_node` code lives here).
  - Deps: ρ (same file; `test_memory_service.py`).
  - G7: INV-9 (journaling wrappers have one home); C7's invariant holds with the extended allowlist; heuristic
    13 (no import of `memory_service`).
- **υ — Extract MemoryMutations** (S26-2)
  - Size and keys: ~1,100 LOC touched (≈920 moved). Keys `fk-da5b47d0493a`, `fk-343c4a42be6b`,
    `fk-50cc281018fb`. Priority medium.
  - Files: `fm/services/memory_service.py`, `fm/services/memory_mutations.py` (new), `fm/services/store_records.py`
    (the three Mem0 reads), `fused-memory/tests/test_memory_mutations.py` (new),
    `fused-memory/tests/test_memory_service_read_surface.py` (the boundary-16 allowlist),
    `fused-memory/tests/test_memory_service.py` (27 references to the moved privates, measured at
    `e634cfe658`), `fused-memory/tests/test_update_memory_tool.py`.
  - Work: C9, D15. `MemoryMutations` takes `delete_memory`, `update_memory`, `_record_content_amend`,
    `_apply_metadata_delta` and the six parent-child hierarchy / cascade methods, plus the Mem0 update-storm
    state; `get_memory_by_id`, `count_memories_by_metadata` and `get_memories_by_metadata` become
    `store_records` functions over the Mem0 backend with one-line service delegates; the metadata-storm
    detector is built once and injected; `MemoryMutations` takes `known_projects` and builds the
    `Mem0UpdateStormEscalator` with it (γ deleted `set_known_projects`, so no setter carries it later);
    `MemoryService` keeps one-line delegates for the public mutators; boundary 16's allowlist gains
    `services/memory_mutations.py`. Dotted-path patch targets of names the move relocates are retargeted to
    `memory_mutations` (a patch on the old `memory_service` path would patch nothing).
  - Deletes: the moved methods from `memory_service.py` (≈920 lines); the 27 private test references in
    `test_memory_service.py` (the tests construct `MemoryMutations`).
  - Signal: boundary row 23; `class MemoryMutations` present, `def _apply_metadata_delta` absent from
    `memory_service.py`.
  - Consumer: every memory-mutation caller (unchanged, through the delegates); the delete and update
    hardening tasks (4859, 4861, 7163, …) whose code now lives here.
  - Deps: ρ (ruled; without it the module stitches back into `memory_service.py`) and τ (same file).
  - G7: INV-9; heuristic 13; Tests stance.
- **π — Integration gate: the write path end to end, and the re-measurement**
  - Size and keys: ~400 LOC. Keys: none closed here (signal leaf). Priority medium.
  - Files: `fused-memory/tests/test_write_path_end_to_end.py` (new), `fused-memory/tests/conftest.py`,
    `plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md` (§Post-state measurement, appended).
  - Work: boundary row 20 over a real SQLite queue in `tmp_path`, a real `WriteJournal`, `FakeGraphAccess`
    and a fake Mem0, through `build_memory_service` + `initialize()` + `start()`; the whole-tree scans the
    team-seat lesson names (silent-fallthrough and the INV-12 suppression ratchet) on the integrated tree;
    the heuristic-14 and complexity re-measurement of `memory_service.py`, `episode_identity.py`,
    `graph_mutations.py`, `memory_mutations.py`, `memory_metadata_policy.py`, `store_records.py`,
    `journaled_ops.py`, `queued_writes.py`, `topic_anchor.py`, `graphiti_client.py` and `canonical_labels.py`
    recorded in the task record and appended
    to this PRD.
  - Deletes: nothing.
  - Signal: the test passes with no private name referenced (an AST assertion inside the test); the recorded
    measurement shows the complexipy total over `memory_service.py` plus every module that received its code —
    the six split modules (`episode_identity.py`, `graph_mutations.py`, `memory_mutations.py`,
    `memory_metadata_policy.py`, `store_records.py`, `journaled_ops.py`), `queued_writes.py` (α's decode, which
    replaces the pop/read-back code) and the functions η adds to `topic_anchor.py` (the moved pin block) — ≤
    1,087 (today's `memory_service.py` total, measured at `e634cfe658`) with `memory_service.py`'s max ≤ 45
    (today 84), and `memory_service.py` below 5,000 lines (≈4,200 estimated; 11,727 today).
  - Consumer: the next `/review-all` run (Phase 0 reads the post-state); φ.
  - Deps: υ, ο (υ is the tail of the merged service, graph and test chains; every other leaf except σ
    reaches π through it).
  - G7: INV-13 `readers-prove-their-producer` — the test's rows are written by the real queue worker under
    `start()`, not seeded.
- **φ — Follow-up: the read-path split and a prose pass over memory_service.py** (S26-2)
  - Size and keys: ~1,500 LOC expected (≈1,300 for the read path after η; the prose pass is not measured).
    Key `fk-da5b47d0493a`. Priority low.
  - Files: `fm/services/memory_service.py`, `fm/services/memory_reads.py` (name confirmed by the lead,
    2026-11-10), and the tests reaching `_search_graphiti`/`_search_mem0` (79 references in 7 files at
    `e634cfe658`, `test_memory_service.py` 48 and `test_temporal_guards.py` 13 the largest); the architect
    re-greps.
  - Work: re-measure `memory_service.py` from π's recorded post-state first. If search and the other reads
    still read as one clean cut ("answer queries over both stores"), extract them into a reader collaborator
    with service delegates and move the tests to behaviour; then a Comments-stance prose pass over what
    remains (lifecycle, queue execution, write API; 40–55% prose at the discussion's measurement). If the
    re-measurement shows no clean cut, close with the measurement recorded and no code change. The architect
    may split it into two tasks along those halves.
  - Signal: the recorded re-measurement; on the split path, `memory_service.py` below 3,000 lines.
  - Consumer: the next `/review-all` run.
  - Deps: π.
  - G7: heuristic 14 (the file the plan leaves above the alarm gets an owner); Comments stance.

DAG: `α → β → γ0 → γ → δ → λ → ε`; `{6631, 6823} → θ → θ2 → γ0`; `θ → γ`; `θ → λ`; `ε → ζ`; `ε → η`;
`θ2 → ι`; `ε → ι`; `ι → κ`; `η → κ`; `{ζ, η, κ} → ξ → μ → ν → ρ → τ → υ`; `ρ → υ`; `{υ, ο} → π → φ`;
`λ → σ`. Out of batch, requested: 6293 depends on σ and ν; 6636 depends on δ; 6730 depends on υ (never on
σ); 6826 depends on ο; 6950 depends on υ (never only on ν: ρ/τ/υ edit five of its files). High leaves for gate 6698: ε, θ, λ, ξ, σ. Priority inheritance:
the high leaves wait behind medium α–δ, γ0, θ2 and behind low 6631/6823; the scheduler's priority
inheritance through the dependency graph lifts those ancestors, so no override is requested (contract §4;
Leo's note on S3-1).

## Rationale rehomed (the Comments stance)

Prose the leaves delete from code and that must persist lives here, one line each with its source; a leaf
leaves a pointer `plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md §Rationale rehomed`.

- Pass order (ε/ζ): referent verification runs after node dedup and name normalisation because it indexes
  the nodes those passes produce; repair runs after verification because it consumes its findings;
  dependency-direction check runs last because it reads repaired edges (task 3671 "leaf zeta", task 5262).
- Verification resolved decisions 7 and 8 (ζ): two nodes minted microseconds apart for the same label are
  a mint race, not two referents — the later one is the duplicate; a degraded target lookup defers the
  repair rather than minting off it (task 5191).
- `ambiguous` is None-or-a-set and the distinction is load-bearing (α/ε): `None` marks a pre-5262 row and
  re-derives from `content`; `()` means "believe: nothing ambiguous" (task 5262).
- `expired_at` is the restored-edge signature (ξ): `update_edge(clear_invalid_at=True)` clears `invalid_at`
  only; an edge with `expired_at` set and `invalid_at` NULL is active (Leo ruling A, esc-4714-9, commits
  `9a10182ab8`, `8a8083ff98`). No fused-memory code reads that shape; graphiti_core's
  `resolve_extracted_edge` self-expiry guard is its only reader, applying only when the edge is re-resolved
  as an LLM duplicate of a new fact (task 5612's 2026-10-19 widening).
- Why relocation is `+=` with endpoint re-asserts (ξ): on FalkorDB 4.18.0 `SET new += properties(old)` (and
  `= properties(old)`) keeps every type, Vectorf32 included, and the relocated edge stays in the vector index
  (lead's probe, 2026-11-02, OQ-1 ruled (b)); the older claim that a whole-map copy loses the vecf32 type
  (task 4986, 5612) does not hold on that version. `vecf32()` accepts a List or Null and errors on a
  Vectorf32, so the in-graph form never re-asserts the embedding. A cross-graph copy goes through the client,
  which returns a vector as a List, so `cross_graph_move` must strip it from the map and re-assert it from the
  byte-exact `--compact` tokens (`_embedding_set_clause`), never via the client's `float()` decode, which the
  cross-graph RCA measured as lossy (`plans/cross-graph-entity-leak-rca.md` §6 Phase 1). The
  endpoint-id properties are re-asserted because `+=` would carry the old endpoints' ids.
- `redirect_node_edges` mints fresh uuids and stamps `superseded_edge_uuid` (task 2207), keeping the
  original on a second redirect (5612 widening); `reassign_edge` keeps the uuid (task 3009).
- Topic pin placement (η): after RRF sort and the category filter so the pin displaces the lowest-ranked
  survivor and never a filtered row; the window stays exactly `limit` long (task 3111, esc-3111-2).
- The label grammar's precision rules (ο): `[ \t]` not `\s` around separators because `\s` matches a
  newline (tasks 4123, 4850); ASCII task word, no `IGNORECASE`, because Unicode case folding accepts
  U+017F/U+212A (task 4850); `[0-9]` not `\d` (task 4174); the task-vocabulary qualifier fullmatches a
  caller-lowercased string; per-clause measured false-positive cases live as parametrised test ids in
  `test_canonical_labels.py`. Direction of safety: a narrowing that removes a foreign ref is safe, but one
  that removes a bare mention removes a contest, which is why the whitespace gap and the local node-name
  padding stay `\s`. Anchored rules end in `\Z`, not `$`, because `$` tolerates a trailing newline. The
  right edge stays `(?!\d)` rather than `(?![0-9])`, so a mixed digit run yields nothing rather than a
  truncated real number.
- Why the queue worker and the journal were "call-time resolved" (γ): `initialize()` started workers before
  `set_write_journal`; `start()` after wiring makes the binding static.
- Why `MemoryService` keeps delegates after the split (ρ/τ/υ): its public mutators have callers in dozens of
  test files and scripts (`delete_memory` 31 test files, 9 script lines); a one-line delegate keeps every
  caller unchanged while the code lives in a module that makes sense alone (S26-2 discussion, 2026-11-10).

## Out of scope

- Cross-process identity locking and task 6293's queue-ownership design: the human design session σ
  (S26-3) designs both, and the PRD it commits owns the work.
- Retiring `QueueOperation.MEM0_ADD` and the `_execute_mem0_write` executor (t2).
- The read-path split and the prose pass over the remainder of `memory_service.py`: follow-up φ.
- The timed-out `add_memory` reporting and retry safety (tasks 4870, 4872, 4812, 4770).
- Changing which edges are active (ruling A stands) or the restore hooks' semantics. Four operator scripts
  spell the active-edge predicate inline (`audit_ruling_overreach`, `audit_wrong_binding_edges`,
  `cleanup_pin_queue_edges`, `invalidate_fabricated_shipping_edges`); `audit_wrong_binding_edges.py` spells it
  `invalid_at IS NULL AND expired_at IS NULL`, contrary to ruling A. They are recorded here, not repointed:
  they are audit scripts, not the server's read path.
- Upgrading legacy List-typed `fact_embedding` edges to Vectorf32 (carried as List in-graph; refused cross-graph by `_read_compact_vector`'s wrong-type `ValueError`, as today).
- `Mem0Backend`'s test seam for its async Qdrant client and the ≈70 test patch sites of `_get_async_qdrant`
  (task 6950, which waits on υ).
- Dashboard or orchestrator consumers (none reach these backends: measured zero in `dashboard/`,
  `orchestrator/`, `escalation/`, `shared/`).

## Rulings (sitting of 2026-11-10)

Every design question of this plan is ruled; none is open. "S26-n" is the sitting key; "OQ-n" the plan's
id. Detailed effects of the changed rulings are in the run's discussion digest for S26, measured on main the
same day, and are applied above.

- **OQ-2 (S26-1)** — scripts construct the backend they read; mutations go through the service's journaled
  mutators or `journaled_op` (option a, as recommended). Ruled by Leo. Applied in D11, ν.
- **OQ-6 (S26-2)** — three extra tasks after the backends-private task and before the integration test: ρ
  (write-metadata policy and store-codec helpers out of `memory_service.py`), τ (`GraphMutations` over a
  small shared journaled-operations module — `journaled_op`, `journaled_backend_call`, event sink — not
  inside `write_journal.py`), υ (`MemoryMutations`, after ρ); `MemoryService` keeps thin public delegates;
  a follow-up (φ) records the read-path split and a prose pass; notes on the live tasks that name moved
  mutators. Ruled by Leo (the recommendation was to stop at the identity split). It carries R7's
  2026-10-10 partition past the identity third, which R7 alone named; it does not change R7's choice of the
  reconciler. It also moves `journaled_op`'s home from `write_journal.py` to `journaled_ops.py` (an effect
  on OQ-9's placement). Applied in Goal, Sketch, D6, D13, D15, C7, C9, rows 16 and 21–23, ρ, τ, υ, φ, π.
- **OQ-7 (S26-3)** — option (d) as a human design session: the follow-up is a deterministic, always-escalating
  gate task asking for an interactive `/spawn` `/team --fable` session with a committed brief, not an
  architect task. Ruled by Leo. Sequencing ruled by the lead: σ depends only on λ and absorbs task 6293's
  design question; task 6730 must not depend on σ. Applied in Goal, Sketch, D8, σ, Appendix A, Out of scope;
  leaf λ2 is removed.
- **OQ-8 (S26-4)** — one factory opens the live journal and project registry for every operator path
  (option a, as recommended). Ruled by Leo. Applied in γ0, γ, C3, D4 (with the read-only opening task 6239
  added since).
- **OQ-9 (S26-5)** — share the journaling wrapper as a free function maintenance wraps its raw writes in
  (option b, as recommended), its module set by S26-2 (`journaled_ops.py`). Ruled by Leo. Applied in D6,
  D11, C7, C9.
- **OQ-10 (S26-6)** — θ depends on 6823; 5612 is cancelled into ξ, which inherits its two Leo-widened items
  (option b, as recommended). Ruled by Leo. Applied in θ, ξ, §Live tasks; 6823's text is amended.
- **OQ-11 (S26-7)** — the lock records its holder in a `ContextVar` and raises `IdentityLockReentry` on
  re-entry (option c, as recommended). Ruled by Leo. Applied in D8 (with the child-task reading), C6, λ,
  rows 10–10c.
- **OQ-1** — in-graph relocation is `SET new += properties(old)` with endpoint re-asserts and no in-graph
  embedding re-assert; cross-graph strips the embedding and re-asserts it from the byte-exact `--compact`
  tokens (option b, on the FalkorDB 4.18.0 probe, and option (a) of critic3 N1 for the transport). Ruled by
  the lead (2026-11-02). Applied in C8, D10, ξ.
- **Lead sequencing ruling (2026-11-10, after the ruling folds)** — the split's module names
  (`journaled_ops.py`, `graph_mutations.py`, `memory_mutations.py`, `memory_metadata_policy.py`,
  `store_records.py`, `memory_reads.py`) are confirmed; the lead commits
  `plans/fm-cross-process-identity-exclusion-session-brief.md` (Appendix A) in the wave-2 commit and σ is filed
  pointing at it. Applied in C9, φ, σ, Appendix A, t7.
- **OQ-3** — names and placement as drafted, with one refusal name `IdentityLockBusy` and
  `identity_lock.acquire_timeout_seconds` absorbing `entity_mint.lock_timeout_seconds`. Ruled by the lead
  (2026-11-02). The module names S26-2 adds (C9, φ) were confirmed by the lead on 2026-11-10 (below; t7).

Tactical (decide in the named leaf):

- **t1 (γ)** — answered at fold: `recon_journal`/`recon_ledger` are built from `config.reconciliation.data_dir`
  alone (constructor keywords); `event_buffer` needs `durable_queue.get_stats` and keeps the one pre-start
  setter (D4).
- **t2 (α/π)** — retire `QueueOperation.MEM0_ADD` once every registered project's queue store shows no
  non-completed row with that operation (dark_factory: 1,511 completed, 0 other, 2026-11-10).
- **t3 (θ)** — the `socket_timeout_seconds` default (proposal: `backend_write_timeout_seconds` + 10 s). The
  clone half is answered: `clone` reuses `self.client`, so the injected connection's options survive.
- **t4 (β)** — answered at this fold: no per-store predicate until a consumer reads a Graphiti write state
  off the model (D3); task 7650's `mem0_write_state` stays the one Mem0 reading.
- **t5 (ι/κ)** — whether `make_backend_over_fake_graphiti` gets a per-group-client injection point or a
  `# debt: task <id>` owner (7781's addendum).
- **t6 (μ)** — `store_inventory`'s shape and whether `get_status` and `health_check` share it.
- **t7 (ρ/τ/υ/φ)** — answered: the lead confirmed the module names S26-2 adds (`journaled_ops.py`,
  `graph_mutations.py`, `memory_mutations.py`, `memory_metadata_policy.py`, `store_records.py`, and φ's
  `memory_reads.py`) on 2026-11-10; the delivered checks anchor on these spellings.

## Appendix A — design-session brief (σ)

To be committed by the lead as `plans/fm-cross-process-identity-exclusion-session-brief.md` in the wave-2
commit, before σ is filed (lead ruling, 2026-11-10); σ's description points at that file.

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
