# Capability manifest — recon-reliability (W5)

Mechanizes G3 + G6 for `plans/recon-reliability-prd.md`. One block per task; each asserted
capability bound to evidence (`grep:file:line` wired on main, `producer:task-X upstream` in the
transitive dependency closure, or `rejection-check` observed). Substrate re-verified 2026-07-06;
re-verify any file:line at implementation time (main moves fast). **No FAIL bindings remain** —
all resolve PASS or producer-upstream.

Legend: **PASS** = evidence found on main; **PRODUCER-UP** = delivered by an upstream task in the
DAG (dep wired); **BOUND** = premise deliberately bound to an upstream confirmation task rather
than asserted (G6 resolution (a)).

Cross-batch dep ids: **2144** M4-α, **2146** M4-β, **2150** M4-γ (task_knowledge_sync.py +
memory_consolidator.py scope), **2149** M5-γ (gather idiom in task_knowledge_sync.py / flag_dedup.py
/ summary_pool.py), **2140** recon_pool_map.py, **2158** W3-α (TaskMetadata).

---

## α — ReconLedgerStore + server wiring

| Capability asserted by signal | Check | Evidence | Verdict |
|---|---|---|---|
| Transactional SQLite store substrate (open/WAL/txn/UPSERT) | capability→producer (wired) | `shared/async_sqlite_base.py:57,89`; template `middleware/ticket_store.py:83`; UPSERT `journal.py:296`, `event_buffer.py:264` | **PASS** |
| Server startup wiring point | grep wired | `server/main.py:479,615,720` (stores constructed + `initialize()`d + checkpoint loop) | **PASS** |
| `INSERT … ON CONFLICT(pk) DO UPDATE` idempotency (row count stays 1) | capability→producer (self) | new code; SQLite ≥3.35 already used (`durable_queue.py:490` RETURNING) | **PASS** |

## β — recon_self_model.py

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| stage→recon_pool constants to import | capability→producer (upstream) | `producer:task-2140 upstream` (recon_pool_map.py, in-progress) — dep wired | **PRODUCER-UP** |
| Prompt sections to render (marker lifecycle, suppression schema, cycle summary, execution_class) | grep (source of prose to replace) | `prompts/stage1.py:489-589`, `prompts/stage2.py:31-66,103-129,196-344` | **PASS** |
| Fingerprint identity fields to export | grep wired | `_derive_affected_ids` `harness.py:177`; Stage-1 fp `flag_dedup.py:1441-1497` | **PASS** |

## γ — dedup-premise empirical confirmation (G6-critical)

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Premise "recon `infer=False` system writes are silently dedup-dropped" | premise validity (G6 branch, BOUND not asserted) | `infer=False` skips dedup `mem0/…/main.py:1417`; `_MEM0_ADD_INFER_PINNED_FALSE` `memory_service.py:57-65` (task-1974: empty=anomalous); only `task_knowledge_sync.py:2110` assumes drops. **This task confirms the outcome; downstream deletion (λ) is anchored on ledger-authority, NOT this premise.** | **BOUND** |
| `count_memories_by_metadata` to verify write landing | grep wired | `memory_service.py:1950` → `mem0_client.count_by_metadata:237` | **PASS** |

## δ — dedup-exempt system-write path

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Fresh-uuid direct insert bypassing Mem0 similar-memory check | capability→producer (wired) | `_create_memory` fresh uuid `mem0/…/main.py:2136-2154`; raw Qdrant `mem0_client._get_async_qdrant():213` | **PASS** |
| Server sees caller agent_id at add_memory time | grep wired | `_resolve_identity` `server/tools.py:409`; recon-stage gate precedent `tools.py:403,787,799` | **PASS** |
| Non-recon caller rejected with observed diagnostic | rejection-check (G6 branch 4) | template = existing recon-stage reject dicts `tools.py:790-807` (returned + FastMCP-serialized); premise confirmed | **PRODUCER-UP** (γ) |

## ε — caller agent_id on the task-write path

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `_resolve_identity` derives caller agent_id | grep wired | `server/tools.py:409-441` (reads `clientInfo.name`) | **PASS** |
| Interceptor methods to thread agent_id into | grep wired (net-new param) | `_apply_status_transition:619`, `update_task:3118` (no agent_id today — this task adds it) | **PASS** (gap is the deliverable) |
| W2 authority for actor/agent_id | DAG-direction (no inversion) | W2 unfiled; W5 builds minimal, W2 converges (PRD §7 decision #5) — **not** a downstream dep | **PASS** (no inversion) |

## ζ — ReconWritePolicy

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Caller agent_id available in interceptor | capability→producer (upstream) | `producer:task-ε upstream` (dep wired) | **PRODUCER-UP** |
| `live_workflow_detector.is_workflow_live_for_task` | grep wired | `services/live_workflow_detector.py:215`; consumed today `task_knowledge_sync.py:570` | **PASS** |
| `TERMINAL_STATUSES` server-side | grep wired | `task_interceptor.py:126` (`frozenset{'done','cancelled'}`) | **PASS** |
| Structured-error surfacing (returned dict, LLM reads mid-run) | rejection-check (G6 branch 4) | `DarkFactoryPathScopeViolation` `path_scope_guard.py:125-154`; returned at `tools.py:2697/3202` | **PASS** |
| Terminal/live-workflow/stale-snapshot writes are actually rejected on X | rejection-check | policy fires server-side pre-write; boundary tests P1/P2/P3 (ο) observe the diagnostic | **PRODUCER-UP** (ο verifies) |

## η — execution_class declaration+validation (coordinates with ratified 2085; does NOT supersede/cancel it)

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Enum-guard template at submit_task | grep wired | `deterministic_task_error` `deterministic_task_guard.py:79-154`; wired `tools.py:2804-2810`; `inject_task_kind:220` | **PASS** |
| Caller agent_id in submit_task (recon-stage gate) | capability→producer (upstream) | `producer:task-ε upstream` (ε covers submit_task) — dep wired | **PRODUCER-UP** |
| `task_kind='deterministic'` + `always_escalates` pure-gate coercion target | grep wired | CLAUDE.md deterministic kind; `deterministic_task_guard.py`; presets validated at submit_task | **PASS** |
| Recon submit_task without execution_class is rejected (observed) | rejection-check (G6 branch 4) | template proven by `deterministic_task_error` reject shape; boundary test E1 (ο) observes | **PRODUCER-UP** (ο verifies) |

## θ — computed derive_stage_stats

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Write journal ops queryable per run/stage | grep wired | `WriteJournal.get_ops_by_causation(run_id)` `write_journal.py:223`; op has `agent_id`,`operation`,`params` | **PASS** |
| Op→stat mapping to compute from | grep wired | `_OP_TO_STAT` `stats_verifier.py:31-42`; bucket-by-stage `:148` | **PASS** |
| Alias map + flag-counter checkers to delete | grep wired | `_STAT_ALIASES` `stats_verifier.py:47`; `_check_flag_counter_completeness` `task_knowledge_sync.py:591`, `_check_mem0_flag_counter_completeness:644` | **PASS** |

## ι — flag_dedup → ledger

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Ledger UPSERT + indexed suppression query | capability→producer (upstream) | `producer:task-α upstream` (dep wired) | **PRODUCER-UP** |
| Marker/suppression write sites + metadata shape to migrate | grep wired | `_write_and_confirm_marker` `flag_dedup.py:745-763`; `filter_suppressed:468-586`; `write_suppression_record:1284` | **PASS** |
| Compensations to delete (confirm/delete dance, circuit-breaker, memo) | grep wired | `flag_dedup.py:1023-1097` (HIT dance), `:319/849-936` (breaker), `:1004-1009/1194` (memo) | **PASS** |
| flag_dedup gather sites converted (avoid collision) | DAG-direction (file-serialize) | `producer:task-2149 upstream` (M5-γ flag_dedup gather) — dep wired | **PRODUCER-UP** |

## κ — TaskKnowledgeSync markers + GC collapse

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Ledger `gc()` DELETE pass | capability→producer (upstream) | `producer:task-α upstream` + markers-in-ledger `producer:task-ι upstream` — deps wired | **PRODUCER-UP** |
| The four sweeps to delete | grep wired | `_sweep_stale_fixc_markers:1192`, `_sweep_stale_flag_markers:1246`, `_sweep_terminal_task_flag_markers:1424`, `_sweep_stale_persistence_markers:1624` | **PASS** |
| ProjectScope-threaded TKS signatures (no rebase war) | DAG-direction (file-serialize) | `producer:task-2150 upstream` (M4-γ) + `producer:task-2149 upstream` (M5-γ) — deps wired | **PRODUCER-UP** |

## λ — deterministic cycle summaries + delete nonce/verify/repair/reconstruct

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Ledger to write the authoritative summary | capability→producer (upstream) | `producer:task-α upstream` — dep wired | **PRODUCER-UP** |
| Dedup-exempt Mem0 mirror write | capability→producer (upstream) | `producer:task-δ upstream` — dep wired | **PRODUCER-UP** |
| Premise that the nonce chain can be safely deleted | G6 (anchored on ledger, confirmed by γ) | deletion justified by ledger-authority (decision #1), NOT by dedup premise; `producer:task-γ upstream` (confirmation) | **PRODUCER-UP / BOUND** |
| Nonce + verify/repair/reconstruct code to delete | grep wired | `generate_summary_nonce` `cli_stage_runner.py:240`; `_verify_stage2_summary_written:1825`,`_repair…:1890`,`_reconstruct…:2040`; run() wiring `:2629-2680`; 4 stats keys `:2628,2673,2674,2678` | **PASS** |
| `StageReport` to source summary content | grep wired | recon report state → `report.stats`/summary (`stats_verifier` entry `:262`; harness `:1878`) | **PASS** |
| recon_pool constants | capability→producer (upstream) | `producer:task-2140 upstream` — dep wired | **PRODUCER-UP** |
| Scope-threaded memory_consolidator + summary_pool (no rebase war) | DAG-direction | `producer:task-2150 upstream` (memory_consolidator) + `producer:task-2149 upstream` (summary_pool) — deps wired | **PRODUCER-UP** |

## μ — post-flight guard shrink + flag-counter deletion

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Server-side rejection now enforces invariants (guards deletable) | capability→producer (upstream) | `producer:task-ζ upstream` — dep wired | **PRODUCER-UP** |
| Computed stats replace flag-counter checks | capability→producer (upstream) | `producer:task-θ upstream` — dep wired | **PRODUCER-UP** |
| Guards/helpers to delete | grep wired | `_apply_post_flight_guards:2730-2987`; `_classify_terminal_state_violations:266`, `_check_stall_guard_freshness:421`, `_verify_set_task_status_post_action:336`, `_classify_live_workflow_status_writes:510` | **PASS** |
| TKS file-serialize | DAG-direction | `producer:task-λ upstream` (intra-batch) — dep wired | **PRODUCER-UP** |

## ξ — prompt self-model cutover + premise-lint

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Rendered self-model sections to import | capability→producer (upstream) | `producer:task-β upstream` — dep wired | **PRODUCER-UP** |
| execution_class prompt requirement text | capability→producer (upstream) | `producer:task-η upstream` — dep wired | **PRODUCER-UP** |
| Premise-lint predicates | capability→producer (upstream) | `producer:task-β upstream` (predicates exported by recon_self_model) | **PRODUCER-UP** |
| Prompt files file-serialize | DAG-direction | `producer:task-λ upstream` (λ edits prompts first) — dep wired | **PRODUCER-UP** |

## ο — integration-gate (B+H two-way boundary tests)

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Every seam it exercises exists | capability→producer (upstream) | `producer:{α,δ,ζ,ι,κ,λ} upstream` — deps wired (transitive closure = the batch) | **PRODUCER-UP** |
| Two-layer integration-test precedent | grep wired | `test_merge_queue_two_layer_integration.py` (task 2001 pattern; cf. W7 ι task 2148) | **PASS** |

## π — deploy capstone (deferred-filer)

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| Out-of-cgroup fused-memory restart convention | grep wired | program decision #6 (`systemctl --user restart fused-memory.service`, NOT `--drain`/task 2090) | **PASS** |
| Deterministic before_done must exist at submit_task | DAG-direction (deferred-filer) | filed as **normal** task; commits the restart script THEN files the `task_kind='deterministic'` deploy (ε2 deferred-filer pattern; validation `tools.py:~2520`) — avoids chicken-egg | **PASS** |
| Deploy runs after all code lands | DAG-direction | `producer:{μ,ξ,ο,…all} upstream` — deps wired | **PRODUCER-UP** |

---

**Manifest verdict:** all bindings PASS / PRODUCER-UP / BOUND. The one shaky premise (the ~0.92
dedup) is **BOUND** to upstream task γ and the downstream deletion (λ) is anchored on
ledger-authority, so no leaf bakes a false premise into a RED test (G6 satisfied). The net-new
substrate gap (caller agent_id on the task-write path) is queued as task ε upstream of ζ/η (G3
resolution (b), dep wired). No FAIL binding blocks the batch.
