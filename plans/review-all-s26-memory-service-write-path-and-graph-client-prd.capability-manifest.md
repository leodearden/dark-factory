# Capability manifest — S26 memory service write path and graph client

PRD: `plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md`. YAML twin:
`plans/review-all-s26-memory-service-write-path-and-graph-client-prd.capability-manifest.yaml`.

**Code anchors** verified against main `38a200d213` (2026-11-10; no plan path changed since `e634cfe658`). Main moves fast — cite-by-symbol;
re-locate lines at implementation time. Bindings cite symbols, not lines. A binding marked *producer* FAILS
at today's tree and PASSES once its leaf lands; a binding marked *substrate* passes today and says so. Every
grep and path check in the twin was run at `38a200d213` through
`shared/src/shared/delivered_check_polarity.py::evaluate_grep_at_tree` / `evaluate_path_at_tree` after
loading the twin with `shared.capability_manifest.CapabilityManifestDoc`: all 62 producer checks fail and all
7 substrate checks pass, and `lint_delivered_checks` reports nothing. `fm/` abbreviates `fused-memory/src/fused_memory/`. The sitting of 2026-11-10 removed
λ2 and added ρ, τ, υ, σ and φ (PRD §Rulings); the fold-verify pass moved the `._graph_for` test check from
ι to θ2 (θ2's shim deletion is what breaks those sites) and added ε's `IdentityLockNotHeld` check.

## α — Typed queued-write values and a QueueOperation enum beside durable_queue

| Capability | Binding | Verdict |
|---|---|---|
| Queue rows carry `(operation TEXT, payload TEXT)` that a typed value can round-trip | substrate: `fm/services/durable_queue.py::QueueItem.parsed_payload`, `_CREATE_TABLE`; live store operations are exactly `add_episode`, `add_memory_graphiti`, `mem0_classify_and_add`, `mem0_add`, all completed (read-only sqlite, 2026-11-10) | PASS |
| The classifier is two-way before α adds the decode branch | substrate: task 6369 (done, `a555a1ecac`) deleted `identity_payload_keys=`/`_identity_uuid`; `DurableWriteQueue._classify_failure` returns transient or normal | PASS |
| One operation vocabulary | producer (α): `fm/services/queued_writes.py::QueueOperation` (queue vocabulary only; write-journal operation strings stay separate) | PASS (α delivers) |
| Executor and callback receive one frozen value; no second parse | producer (α): `DurableWriteQueue._process_item` decodes once; the "A FRESH parse" comment goes | PASS (α delivers) |
| An undecodable row is loud and terminal (INV-11), never a Graphiti write | producer (α): `QueueDecodeError`/`UnknownQueueOperation`; `_classify_failure` returns `('permanent', 1)` by type; boundary row 2 asserts `attempts == 1` | PASS (α delivers) |

## β — add_memory reports Graphiti as queued, not written

| Capability | Binding | Verdict |
|---|---|---|
| The response model is the wire shape | substrate: `fm/server/tools.py::add_memory` returns `result.model_dump()`; `fm/models/memory.py::AddMemoryResponse` has `stores_pending_retry` (task 7397) | PASS |
| One Mem0 reading already exists | substrate: `fm/models/memory.py::AddMemoryResponse.mem0_write_state` / `Mem0WriteState` (task 7650, `789efc6ad6`); β leaves it unchanged and adds no per-store predicate (PRD D3) | PASS |
| `stores_queued` list | producer (β): `AddMemoryResponse.stores_queued`; the `report as written` append goes | PASS (β delivers) |

## γ0 — Every operator entry point constructs MemoryService through `maintenance_service` (S26-4: A)

| Capability | Binding | Verdict |
|---|---|---|
| One factory to route through | substrate: `fm/maintenance/_utils.py::maintenance_service` (async context manager) | PASS |
| A read-only opening to route the eval scripts and the census through | substrate: `MemoryService.initialize_for_reads` (task 6239) | PASS |
| No script constructs a service directly | producer (γ0): `(=|return) *([A-Za-z_]+\.)?MemoryService\(` absent under `fused-memory/scripts/` and `scripts/` (29 lines in 29 files at `e634cfe658`; 2 spelled `_service.MemoryService(`) | PASS (γ0 delivers) |
| No injected `MemoryService` default | producer (γ0): `= MemoryService,` absent under the same roots (`memory_eval_write_after_miss.py::open_memory`, 1 line today) | PASS (γ0 delivers) |
| One opening enum | producer (γ0): `fm/maintenance/_utils.py::ServiceOpen` (`FULL`, `READS`, `NONE`) | PASS (γ0 delivers) |
| Read-only and lean openings kept | producer (γ0): `READS` for the two eval scripts and `task_family_census`'s CLI (task 6835 absorbed), `NONE` for `scripts/check_consolidation_closure.py`; boundary row 5c | PASS (γ0 delivers; manual) |

## γ — MemoryService constructed with its collaborators; an explicit start() after wiring

| Capability | Binding | Verdict |
|---|---|---|
| The journal, project registry, recon journal and ledger can be built before the service | substrate: `fm/server/main.py::run_server` builds `WriteJournal(wj_data_dir)`, `_known_projects_map`, `ReconciliationJournal(Path(config.reconciliation.data_dir))` and `_build_recon_ledger_store(...)`, none reading the service | PASS |
| The event buffer needs the queue's stats | substrate: `run_server` builds `EventBuffer(queue_stats_fn=memory_service.durable_queue.get_stats)`; so the queue is constructed in `__init__` and `set_event_buffer` is the one pre-start setter | PASS |
| Queue recovery and worker start are separable from schema open | substrate: `DurableWriteQueue.initialize` ends with `_recover_in_flight`, `_start_workers_for_pending_groups`; `enqueue`, `enqueue_batch`, `replay_dead` call `_ensure_workers` — all gated by `start()` | PASS |
| `start()` exists on both classes | producer (γ): `MemoryService.start`, `DurableWriteQueue.start` | PASS (γ delivers) |
| No worker before `start()` | producer (γ): boundary rows 5 and 5b | PASS (γ delivers; manual) |
| No None-guard on the journal | producer (γ): zero `if self._write_journal` in `memory_service.py` (25 at `e634cfe658`) | PASS (γ delivers) |
| Setters gone; one test factory | producer (γ): `def set_write_journal` absent; `fused-memory/tests/conftest.py::build_memory_service` present; `planned_registry=` keyword replaces `set_planned_registry` (task 8660 absorbed) | PASS (γ delivers) |

## δ — One `journaled_op` context manager for every Layer-1 write

| Capability | Binding | Verdict |
|---|---|---|
| `log_write_op` never raises and upserts on `id` | substrate: `fm/services/write_journal.py::WriteJournal.log_write_op` docstring "Fire-and-forget — never raises"; `_UPSERT_WRITE_OP_SQL` (task 3582) | PASS |
| Layer-2 choke point to nest inside | substrate: `fm/services/memory_service.py::MemoryService._journaled_backend_call` (moved by τ) | PASS |
| One Layer-1 context manager; no `success = True` sentinel | producer (δ): `services/journaled_ops.py::journaled_op` (S26-5: B; home per S26-2) and `MemoryService._journaled_op`; `success = True$` count 0 (9 today) | PASS (δ delivers) |
| Every operation's failure journaled, once, under the carried id | producer (δ): boundary row 8 (19 sites, degraded `search`, retried fact, cancellation) | PASS (δ delivers) |

## ε — EpisodeIdentityReconciler owns the nine identity passes behind run()

| Capability | Binding | Verdict |
|---|---|---|
| The chain's closure reads a bounded set of service state | substrate: AST measurement — 7 attributes (`graphiti`, `taskmaster`, `_write_journal`, `_known_projects`, `_referent_finding_counts`, `_referent_repair_counts`, `_referent_repair_streaks`); no config field | PASS |
| Streak primitive shared | substrate: `shared/src/shared/streak.py::KeyedStreaks` (S14 λ, task 6712 done) | PASS |
| Collaborators available at construction | producer: γ (C3) upstream of ε | PASS (γ upstream) |
| Public lock to assert in `run()` (D8) | producer: λ (C6) upstream of ε | PASS (λ upstream) |
| The class and its public entry | producer (ε): `fm/services/episode_identity.py::EpisodeIdentityReconciler`, `run` | PASS (ε delivers) |
| The chain leaves memory_service.py | producer (ε): `def _reconcile_episode_identity` absent from `memory_service.py` | PASS (ε delivers) |
| Repair vocabulary typed (task 6601, `fk-ec3de5b6cf2b`) | producer (ε): `RepairOutcome` StrEnum | PASS (ε delivers) |
| `run()` refuses without the lock (D8, row 10) | producer (ε): `fm/services/episode_identity.py::IdentityLockNotHeld` | PASS (ε delivers) |
| Tests drive `run()` not the passes | producer (ε): `_verify_episode_referents` absent from `test_memory_service.py` and `test_referent_verification.py` (137 lines in those 2 files today) | PASS (ε delivers) |

## ζ — The verify pass as four methods; its narration rehomed

| Capability | Binding | Verdict |
|---|---|---|
| The pass lives in the reconciler | producer: ε upstream | PASS (ε upstream) |
| The pass is split | producer (ζ): `def _walk_edge_endpoints` present in `episode_identity.py` | PASS (ζ delivers) |
| Complexity reduced, not moved | producer (ζ): the four successors each ≤ 25; module max ≤ 45 (the untouched `_invalidate_stale_superseded_ttl_edges`) — recorded in the task record | PASS (ζ delivers; manual) |
| Narration gone from the reconciler | producer (ζ): judged in review — an absent-grep on a file that does not yet exist passes vacuously | PASS (ζ delivers; manual) |

## η — search as fan-out / fuse-and-filter / pin / journal

| Capability | Binding | Verdict |
|---|---|---|
| The pin's inputs are all passable | substrate: the inline block reads `results`, `limit`, `self._mem0` scroll (`_ANCHOR_SCROLL_LIMIT`), `include_planned`, categories — no other service state | PASS |
| Pure pin function | producer (η): `fm/services/topic_anchor.py::apply_topic_anchor`; `_ANCHOR_SCROLL_LIMIT` absent from `memory_service.py` | PASS (η delivers) |
| Leaf narration gone from the service | producer (η): `leaf iota` absent from `memory_service.py` (18 lines today; ε moves 12, η deletes 6) | PASS (η delivers) |
| `search` cognitive complexity ≤ 15 | producer (η): complexipy (81 today), recorded in the task record | PASS (η delivers; manual) |

## θ — GraphAccess protocol with bounded egress

| Capability | Binding | Verdict |
|---|---|---|
| Driver injectable without `initialize()` | substrate: `fm/backends/graphiti_client.py::GraphitiBackend.with_driver` (task 6233, `d064e0181d`) | PASS |
| Named-timeout egress primitive | substrate: `fm/backends/timeout_budget.py::await_within` raising `TimeoutError('Graphiti <op> timed out after <budget>s')` | PASS |
| Budgets exist in config | substrate: `fm/config/schema.py::QueueConfig` `backend_read_timeout_seconds=30.0`, `backend_write_timeout_seconds=120.0`, read in `GraphitiBackend.__init__` as `config.queue.*` | PASS |
| A socket timeout can reach the connection | substrate: `falkordb.asyncio.FalkorDB.__init__(…, socket_timeout=None, …)` and `graphiti_core.driver.falkordb_driver.FalkorDriver.__init__(…, falkor_db=…)` (root `.venv`); `clone` reuses `self.client` | PASS |
| Upstream SET-clause fix settled | producer: task 6823 upstream of θ (S26-6: B) | PASS (6823 upstream) |
| The protocol and the Falkor provider | producer (θ): `fm/backends/graph_access.py::GraphAccess`, `FalkorGraphAccess` | PASS (θ delivers) |
| A public path from backend to provider | producer (θ): `GraphitiBackend.graph_access` property | PASS (θ delivers) |
| No private graph resolution left in the backend | producer (θ): `_get_graph` absent from `graphiti_client.py` (`_graph_for` survives as a shim until θ2) | PASS (θ delivers) |
| Every Cypher call budgeted | producer (θ): boundary row 12 parametrised over the backend's Cypher methods | PASS (θ delivers) |
| `falkor_read_only` imports downward | producer (θ): `_paged_ro_query` import from `graphiti_client` absent in `falkor_read_only.py` | PASS (θ delivers) |

## θ2 — Operator scripts and their tests onto `GraphitiBackend.graph_access`

| Capability | Binding | Verdict |
|---|---|---|
| Public provider to reach | producer: θ (`GraphitiBackend.graph_access`, `with_driver(graph_access=)`) upstream | PASS (θ upstream) |
| The shim goes | producer (θ2): `def _graph_for` absent from `graphiti_client.py` | PASS (θ2 delivers) |
| Scripts on the public provider | producer (θ2): `\._graph_for\(|\._require_falkor_client\(` absent under `fused-memory/scripts/` (19 lines in 7 scripts today) | PASS (θ2 delivers) |
| No `_graph_for` replacement in tests | producer (θ2): `\._graph_for = |patch\.object\([^)]*_graph_for` absent under `fused-memory/tests/` (65 assignments in 5 files plus `test_reindex.py`'s 4 `patch.object`, which cannot outlive the shim) | PASS (θ2 delivers) |

## ι — Tests on the public graph seam, cluster 1

| Capability | Binding | Verdict |
|---|---|---|
| Fake provider through the public constructor | producer: θ (`conftest::FakeGraphAccess`, `with_driver(graph_access=)`) upstream | PASS (θ upstream) |
| No driver stub in the heavy files | producer (ι): `_driver\._get_graph` absent from its seven files (`test_reindex`, `test_cross_graph_move`, `test_refresh_entity_summary`, `test_write_time_identity`, `test_merge_entities`, `test_delete_entity`, `_fm_helpers`; 160 lines at `38a200d213`) | PASS (ι delivers) |

## κ — Tests on the public graph seam, cluster 2

| Capability | Binding | Verdict |
|---|---|---|
| No driver stub in tests | producer (κ): `_driver\._get_graph` absent under `fused-memory/tests/` (241 in 27 files today, conftest's `make_graph_driver` docstring included) | PASS (κ delivers) |
| Remaining private reads owned | producer (κ): any kept private read carries `# debt: task <id>` (INV-12) — judged by the inline-suppression ratchet | PASS (κ delivers; manual) |

## λ — The identity lock as a public context manager held by every in-process identity mutator

| Capability | Binding | Verdict |
|---|---|---|
| Per-group lock registry exists | substrate: `GraphitiBackend._identity_lock_for` (task 2198) and the bounded acquire in `MemoryService.ensure_entity_node` (`entity_mint.lock_timeout_seconds`) | PASS |
| Public context manager | producer (λ): `GraphitiBackend.identity_lock` | PASS (λ delivers) |
| No private lock reach from the service layer | producer (λ): `graphiti._identity_lock_for` absent under `fm/services/` (3 lines today) | PASS (λ delivers) |
| Lock-busy is a structured refusal | producer (λ): `IdentityLockBusy`; boundary row 14 observes it fire | PASS (λ delivers) |
| Every MCP/reconciliation wait bounded by config (INV-7) | producer (λ): `acquire_timeout_seconds` present in `fm/config/schema.py` | PASS (λ delivers) |
| Re-entry refused (S26-7: C) | producer (λ): `IdentityLockReentry`; boundary rows 10b and 10c observe it fire | PASS (λ delivers) |

## σ — Human design session: cross-process identity exclusion (S26-3)

| Capability | Binding | Verdict |
|---|---|---|
| A public lock and refusal to design on | producer: λ (`GraphitiBackend.identity_lock`, `IdentityLockBusy`) upstream; σ depends on λ only (lead ruling) | PASS (λ upstream) |
| A pure-gate task form exists | substrate: `docs/task-authoring.md` §5 — `task_kind='deterministic'`, `always_escalates=true`, no `before_done` escalates born-at-L2 and closes `done` after `resume` | PASS |
| The session's PRD is committed and the gate resumed with its path | the session answers the committed brief (PRD Appendix A, `plans/fm-cross-process-identity-exclusion-session-brief.md`) with Leo; its two-process test is that PRD's signal | PASS (manual) |

## μ — Backends private; the sweeps' read surface; every graph mutation through the service

| Capability | Binding | Verdict |
|---|---|---|
| Backends injectable for tests | producer: γ (`graphiti=`/`mem0=` constructor keywords, `build_memory_service`) upstream | PASS (γ upstream) |
| `delete_entity` journals and refuses active edges | substrate: `MemoryService.delete_entity` → `GraphitiBackend.delete_entity` (`ActiveEdgesError`, neighbour refresh), journaled | PASS |
| Locked `delete_entity` | producer: λ upstream | PASS (λ upstream) |
| Backends private | producer (μ): `self\.(graphiti|mem0)([^A-Za-z0-9_]|$)` absent from `memory_service.py` (87 lines today) | PASS (μ delivers) |
| Sweep delete through the service | producer (μ): `\.delete_entity_node\(|def delete_entity_node` absent under `fm/reconciliation/` and in `graphiti_client.py` (the primitive survives as private `_detach_delete_entity_node`) | PASS (μ delivers) |
| Journaled bulk edge removal | producer (μ): `MemoryService.remove_edges` (a delegate to `GraphMutations` after τ); grep `async def remove_edges` in `memory_service.py` or `graph_mutations.py` | PASS (μ delivers) |
| Reach-through guard exercised | producer (μ): boundary row 16's AST scan flags a seeded fixture (INV-10) | PASS (μ delivers; manual) |

## ν — Scripts stop reaching through MemoryService's backends (S26-1: A)

| Capability | Binding | Verdict |
|---|---|---|
| Public raw-point scroll on Mem0 | producer: μ (`Mem0Backend` public read replacing `_get_async_qdrant`) upstream | PASS (μ upstream) |
| No `_get_async_qdrant` in scripts | producer (ν): absent under `fused-memory/scripts/` (3 calls + 2 prose mentions today) | PASS (ν delivers) |
| Conformance rule for the reach | producer (ν): `fused-memory/tests/test_store_mutation_conformance.py` gains the rule; flags its fixture | PASS (ν delivers; manual) |

## ξ — One home for a RELATES_TO edge's stored meaning

| Capability | Binding | Verdict |
|---|---|---|
| Ruling A semantics to preserve | substrate: `graphiti_client._ACTIVE_EDGE_PREDICATE = 'e.invalid_at IS NULL'` (task 4714, `8a8083ff98`); `update_edge(clear_invalid_at=True)` clears `invalid_at` only (`9a10182ab8`) | PASS |
| Upstream SET-clause work settled | producer: task 6823 upstream of θ (and so of ξ); 5612 cancelled into ξ (S26-6: B) | PASS (6823 upstream) |
| `SET new += properties(old)` is type-faithful on FalkorDB, Vectorf32 included (OQ-1 ruled (b)) | substrate: lead's live probe 2026-11-02, FalkorDB 4.18.0 — every type kept, vector index returns the moved edge; `vecf32()` errors on a Vectorf32; a parameter-map vector arrives as a List | PASS (manual) |
| Cross-graph moves keep the embedding value-equal and indexed | producer (ξ): `relates_to_schema.cross_graph_embedding_reassert(` (the `CROSS_GRAPH_EMBEDDING_REASSERT` rule) called in `cross_graph_move.py` — `fact_embedding` stripped from `$props`, re-asserted from the byte-exact `--compact` tokens via `_embedding_set_clause`; row 17 checks token-for-token equality | PASS (ξ delivers) |
| One home module | producer (ξ): `fm/backends/relates_to_schema.py::relocation_set_clause`, `RELOCATION_REASSERTED_PROPERTIES`, `ACTIVE_EDGE_PREDICATE`, `edge_is_active` | PASS (ξ delivers) |
| The copies go | producer (ξ): `_CARRIED_EDGE_PROPERTIES` absent from `cross_graph_move.py`; `_COPIED_PROPS` absent from `test_reassign_edge.py`; `def _edge_is_retired` absent from `memory_service.py` | PASS (ξ delivers) |
| Relocation parity by behaviour | producer (ξ): boundary row 17 (mock tier + `-m integration` live tier), endpoint ids included | PASS (ξ delivers) |

## ο — canonical_labels' five patterns from one grammar

| Capability | Binding | Verdict |
|---|---|---|
| Task-word fragment already shared | substrate: `fm/utils/canonical_labels.py::_TASK_WORD` (task 4850) | PASS |
| Public API unchanged and tested | substrate: `parse_node_name`, `scan_content`, `is_task_vocabulary_qualifier`, `Referent`, `LabelScan`; 16 test files reference the module | PASS |
| Fragments composed once | producer (ο): `_NUMBER = ` present in `canonical_labels.py`; qualifier class literal count 1 | PASS (ο delivers) |
| Precision fixtures unchanged | producer (ο): the parametrised suites pass with identical verdicts; `is_task_vocabulary_qualifier('Task')` stays False | PASS (ο delivers) |

## ρ — Move the write-metadata policy and store-codec helpers out of memory_service.py (S26-2)

| Capability | Binding | Verdict |
|---|---|---|
| The block separates without stitching | substrate: the discussion round's AST measurement (2026-11-10) — the policy block (883 lines) and codec helpers (552 lines) are module-level functions taking their collaborators as arguments; `server/grouped_read.py` and `reconciliation/preservation_specimen_guard.py` already import from it | PASS |
| The policy has its own module | producer (ρ): `fm/services/memory_metadata_policy.py::apply_memory_metadata_validation` (public) | PASS (ρ delivers) |
| The policy left the service | producer (ρ): `^(async )?def _apply_memory_metadata_validation\b` absent from `memory_service.py` | PASS (ρ delivers) |
| The codec has its own module | producer (ρ): `fm/services/store_records.py::MEM0_CONTENT_KEYS` (with `mem0_content`, `metadata_row_text`, the record decoders and store-error helpers) | PASS (ρ delivers) |
| No private import of the codec | producer (ρ): `memory_service import _MEM0_CONTENT_KEYS` absent from `fm/server/grouped_read.py` | PASS (ρ delivers) |

## τ — Extract GraphMutations over a shared journaled-operations module (S26-2)

| Capability | Binding | Verdict |
|---|---|---|
| The graph mutators call only the two journaling helpers and read only injected objects | substrate: the discussion round's AST measurement (2026-11-10): 14 methods, inbound calls from the rest of the file none | PASS |
| The layer's first member | producer: δ (`services/journaled_ops.py::journaled_op`) upstream | PASS (δ upstream) |
| The collaborator | producer (τ): `fm/services/graph_mutations.py::GraphMutations` | PASS (τ delivers) |
| The backend-call wrapper shared | producer (τ): `fm/services/journaled_ops.py::journaled_backend_call` | PASS (τ delivers) |
| The mutators left the service | producer (τ): `def _expire_standing_decisions_for_merge` and `def _journaled_backend_call` absent from `memory_service.py` | PASS (τ delivers) |
| Delegates behave identically | producer (τ): boundary row 22 | PASS (τ delivers; manual) |

## υ — Extract MemoryMutations (S26-2)

| Capability | Binding | Verdict |
|---|---|---|
| The policy and codec are importable from their own modules | producer: ρ upstream (ruled: υ depends on ρ) | PASS (ρ upstream) |
| The collaborator | producer (υ): `fm/services/memory_mutations.py::MemoryMutations` | PASS (υ delivers) |
| The memory mutators left the service | producer (υ): `def _apply_metadata_delta` absent from `memory_service.py` | PASS (υ delivers) |
| The three Mem0 reads beside the decoders | producer (υ): `fm/services/store_records.py::get_memory_by_id` (with the two metadata reads); the service delegates | PASS (υ delivers) |

## π — Integration gate

| Capability | Binding | Verdict |
|---|---|---|
| Every chain landed | producer: υ and ο upstream (every other leaf except σ reaches π through υ) | PASS (upstream) |
| The end-to-end test exists and names no private | producer (π): `fused-memory/tests/test_write_path_end_to_end.py` | PASS (π delivers) |
| Rows written by the real producer (INV-13) | producer (π): the queue worker under `start()` writes the rows the test reads | PASS (π delivers; manual) |
| Post-state measurement recorded | producer (π): §Post-state measurement appended to the PRD over `memory_service.py` and every module split from it; task record carries the numbers | PASS (π delivers; manual) |

## φ — Follow-up: the read-path split and a prose pass (S26-2)

| Capability | Binding | Verdict |
|---|---|---|
| A post-state to start from | producer: π (the recorded measurement) upstream | PASS (π upstream) |
| The read path decided | the re-measurement recorded; on the split path a reader collaborator with service delegates | PASS (manual) |
