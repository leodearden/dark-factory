# Capability manifest — plans/memory-auto-consolidation-prd.md

Binds each task's asserted capabilities to evidence (G3+G6 mechanized). Verified 2026-09-09
against main `3a6bbc7dc4` by two evidence agents (tasks α–δ, tasks ε–θ) plus the decomposing
session; every binding cites `path::symbol` or a section banner, never `file:line`. Two FAIL
bindings surfaced and were resolved before filing (recorded under the task they hit). The
machine-readable twin is `memory-auto-consolidation-prd.capability-manifest.yaml`; explanations
live here because the sidecar stamper drops YAML comments on every batch flip.

Batch DAG: γ ← {α, β}; δ ← {γ, task 4808}; ε ← α; ζ ← δ; η ← {δ, ε}; θ ← δ.

## task α — config, predicate, validator, builder (intermediate → γ, δ, ε)

| Capability | Evidence | Verdict |
|---|---|---|
| Bare-mount submodel pattern makes leaves reloadable | `fused-memory/src/fused_memory/config/schema.py::Mem0UpdateConfig` docstring ("BARE (non-Optional) submodel so `_iter_leaves` descends"); `FusedMemoryConfig.mem0_update = Field(default_factory=Mem0UpdateConfig)` | PASS wired |
| Allowlist + apply path | `config/reload.py::RELOADABLE_FIELDS` (frozenset of dotted leaves) consumed by `apply_reload`; `server/tools.py::reload_config` calls it on the live config; list-leaf precedent `'reconciliation.procedural_knowledge_topic_guard_clusters'` | PASS wired |
| Green-tier test precedent | `fused-memory/tests/test_config_reload.py::TestMem0UpdateLeavesAreGreenTier`; live-read precedent `tests/server/test_update_memory_authz_gate.py::TestLiveRead` | PASS wired |
| One validator to extend | `server/consolidation.py::validate_consolidate_args(*, canonical_content, topic, supersedes, retain, run_id)` — slug + UUID shape + overlap today; no claim rules | PASS wired (extension = producer:this-task) |
| Slug regex home | `fused_memory/topic_slug.py::TOPIC_SLUG_RE`, `TOPIC_SLUG_MAX_LEN = 100`, `is_valid_topic_slug` — no tokenizer or stopword list (open question 4 → inline list) | PASS wired |
| Import-leaf pin precedent | `fused-memory/tests/test_consolidation_gate.py::TestImportLeafAndSingleHomes.test_module_imports_stay_leaf` (subprocess import, forbidden-module set) | PASS wired |
| `evaluate_auto_predicate`, `build_auto_canonical`, `ConsolidationAutoConfig`, the provenance-prefix leaf | absent on main | producer:this-task |

## task β — extract the retain arm (intermediate → γ, δ)

| Capability | Evidence | Verdict |
|---|---|---|
| Retain arm is a contiguous closure region | `server/tools.py::consolidate_memories`: inner `_patch_metadata` → `memory_service.update_memory(content=None, metadata_patch=…, metadata_mode='merge', agent_id, session_id, causation_id, _source)`; `for retain_id in retain_ids` reading `get_memory_by_id`; mint via `memory_service.add_memory(content=canonical_content, category, agent_id, metadata=canonical_meta)`; closure scroll via `get_memories_by_metadata`; envelope via `server/consolidation.py::build_consolidation_result` | PASS wired |
| Authz is tool-layer today | same tool → `resolve_mem0_update_authorization(memory_service, agent_id=…, content_amend=False, metadata_patch=True)`; defaults `['recon-stage-', 'curator-']` on both prefix lists (`config/schema.py::Mem0UpdateConfig`) so the executor's `recon-stage-memory_consolidator` identity clears it | PASS wired (helper must call it — D17 of the review) |
| Fixture cluster | `fused-memory/tests/test_consolidate_memories_tool.py::make_service`, `::call_consolidate`, `::TestRetainAndTagArm` (8 tests). No existing byte-identical-envelope assertion — β's signal adds it | PASS wired |
| `services/consolidation_ops.py::execute_retain_consolidation` | absent on main | producer:this-task |

## task γ — proposal tool, ledger reader, stage surface, prompt retirement (intermediate → δ)

| Capability | Evidence | Verdict |
|---|---|---|
| Recon-stage prefix gate precedent | `server/tools.py::add_system_record` refuses non-`recon-stage-` callers with `DedupExemptNotPermitted` | PASS wired |
| Ledger record shape | `reconciliation/recon_ledger.py::ReconLedgerRecord` (free-string `record_kind`/`state`), PK `(project_id, record_kind, task_id, flag_type, run_id)`, `upsert` last-write-wins, `mark_addressed`, `gc` deletes only `expires_at < now`, index `ix_recon_ledger_project_kind_state` | PASS wired |
| List-by-kind reader | none on main (`get_by_identity`, `list_suppressions`, `marker_task_ids`, … only) | producer:this-task |
| Additive deny-sublist pattern + CLI path | `reconciliation/cli_stage_runner.py::DISALLOW_RECON_REPORT_LEDGER_WRITES` folded into `STAGE1_DISALLOWED`; `STAGE3_DISALLOWED` alone folds `DISALLOW_MEMORY_WRITES` (holds `consolidate_memories`); lists reach the CLI via `stages/base.py::BaseStage.get_disallowed_tools` → `shared/src/shared/cli_invoke.py::invoke_with_cap_retry` (`--disallowed-tools`) | PASS wired |
| Pins to retire | `fused-memory/tests/test_stage1_consolidation_guidance.py::TestStage1AdvertisesTheConsolidationOp` (3), `::TestSharedNormNamesTheSanctionedPath.test_the_norm_names_the_consolidation_op`, `::TestStage1ExecutionContract`; `tests/test_stages.py::TestDisallowedToolLists.test_consolidate_memories_is_classified_as_a_memory_write` | PASS wired |
| Prompt sites | `prompts/stage1.py` (7 mentions); `prompts/__init__.py::STALE_KNOWLEDGE_ANNOTATION_NORM` clause (d) — NOT `AMEND_AND_EPISODE_TOOLS_BLOCK` (0 mentions; PRD corrected); `consolidation_gate.py::render_end_state_brief` step 5; `recon_self_model.py` is deleter provenance only (dropped from γ) | PASS wired (2 sites re-homed) |
| Markup guard covers the new tool automatically | `server/markup_guard.py::install_markup_guard` wraps `_tool_manager.call_tool`, `EXEMPT_TOOLS = {'scan_memory_content'}` | PASS wired |
| Every-tool-classified test | only `tests/test_stages.py::TestDisallowedToolLists.test_every_escalation_server_tool_is_classified` (escalation server); no fused-memory twin | producer:this-task |
| `propose_consolidation` | absent on main | producer:this-task |

## task δ — executor post-step, flood control, gate shape, reversal (leaf-bearing intermediate → ζ, η, θ)

| Capability | Evidence | Verdict |
|---|---|---|
| Post-step seam + attributes | `reconciliation/stages/memory_consolidator.py::MemoryConsolidator.run`: `super().run()` → remediation early-return → sweeps → `maybe_escalate_stalled_gate_backlog(escalation_queue=self._escalation_queue, …)` → `write_stage1_cycle_summary`; `self.memory`/`self.taskmaster` from `stages/base.py::BaseStage.__init__`; `_escalation_queue` assigned by the harness | PASS wired |
| Raw backend identity | `server/main.py::_build_task_backend` → `SqliteTaskBackend`, passed positionally to `ReconciliationHarness`; `SqliteTaskBackend.add_task(..., metadata: str | None, ...)` has NO `task_kind`/`always_escalates` params — markers travel in the metadata JSON (`task_interceptor._GATE_MARKER_KEYS`) | PASS wired, field-population caveat recorded in C6 |
| Stats reach the cycle summary | `reconciliation/summary_pool.py::write_cycle_summary` serialises `report.stats` whole | PASS wired |
| In-stage `info` escalation precedent | `reconciliation/stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog` builds `escalation.models.Escalation(..., severity='blocking', level=1, dedupe_fingerprint=…)` and calls `submit_or_dedupe`; `Escalation.severity` accepts `'info'`; `harness.py::ReconciliationHarness._escalate` has no severity param (hardcoded tuple) — hence not used | PASS wired |
| `info` never pins | `escalation/src/escalation/pins.py::_classify_record` → `NON_PINNING` for `sev == 'info'` | PASS wired |
| Open-gate scan | `reconciliation/curator_gate_resolution_sweep.py::extract_open_gate_task_ids` (ids only; topic read off `x_recon_consolidation_gate`) | PASS wired, extent noted |
| Gate builder + closure on main | `reconciliation/consolidation_gate.py::build_consolidation_gate_task(*, topic, rationale, observed_members, report_run, detector, authoritative, considered_and_kept, priority, title, description)`; `_default_description`; `evaluate_closure(..., unstamped_live_ids=())` already on main | PASS wired (extension = producer:this-task) |
| 4808 closure helper | `git show task/4808:…/consolidation_gate.py` defines `unstamped_candidates` + `async resolve_unstamped_live_ids`; absent on main; tip `60e05ccea3` not an ancestor | PASS producer-upstream (external dep task 4808; PRD pin corrected) |
| Category vocabulary | `escalation/src/escalation/server.py::CATEGORIES` is inert (consumed nowhere; `queue.py::EscalationQueue.submit` never rejects) | rejection-absent for the vocabulary — documented, not a signal premise |
| `MemoryService` surface | `services/memory_service.py::count_memories_by_metadata`, `update_memory(... metadata_delete_keys ...)`, `delete_memory(memory_id, store, ...)` (store positional — C4 corrected), `_check_canonical_uniqueness` (census) | PASS wired |
| Anchor selector ignores non-`True` canonical | `services/topic_anchor.py::select_canonical_payload` (`canonical is True`) — reversal must delete, not flip | PASS wired |
| Post-hoc guard target | `middleware/deterministic_task_guard.py` (`inject_task_kind`, `deterministic_task_error`); imported only by `server/tools.py` | PASS wired |
| Executor, flood control, revert | absent on main | producer:this-task |

## task ε — close the hand-passed-provenance hole (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Validator signature + existing recon-stage branch | `middleware/task_interceptor.py::_validate_done_provenance(task_id, raw, project_root, *, require, is_recon_stage=False)`; `kind == 'operational-verified' and is_recon_stage` → `_done_provenance_error(...)`; pinned by `fused-memory/tests/test_task_interceptor.py` | PASS rejection-fires |
| Call sites | `TaskInterceptor._apply_status_transition` computes `is_recon_stage_write = agent_id.startswith('recon-stage-')`; TWO validator call sites (fresh `done` and `_repair_done_provenance_same_status`) — ε covers both | PASS wired |
| Caller identity reaches the interceptor | `server/tools.py::set_task_status` → `_resolve_identity(agent_id, None, ctx)` (clientInfo fallback) → `task_interceptor.set_task_status(agent_id=…)`; `orchestrator/src/orchestrator/scheduler.py::Scheduler.set_task_status` sends no `agent_id`; `mcp_lifecycle.py::McpSession.initialize` advertises `'orchestrator'` | PASS wired |
| Journal discards the caller today | `TaskInterceptor._journal_around` → `write_journal.log_write_op(..., agent_id='task-interceptor', ...)`; `services/write_journal.py::log_write_op` already accepts `agent_id` | PASS wired (defect confirmed) |
| Runner arms | `orchestrator/src/orchestrator/deterministic_runner.py`: curator arm `_build_done_provenance('deterministic-gate', note=…, escalation_id=_gate_esc_id)`; pure arm omits it; `orchestrator/tests/test_deterministic_runner.py` pins `'pure gate resolved'` | PASS wired |
| Resolver shape to copy | `server/mem0_update_authz.py::resolve_mem0_update_allowed_prefixes` (non-list → deny) | PASS wired |
| Config leaf | `reconciliation.deterministic_provenance_allowed_agent_prefixes` | producer:task-α (upstream) |
| Docs table | `docs/task-authoring.md` §"Task statuses & transitions" → "`done_provenance` requirement" (7-row table) | PASS wired |
| Rejection scope (G6 branch 4) | producers of `deterministic-*` outside tests: `deterministic_runner.py` (6 sites) AND `fused-memory/scripts/cgl_eta_finalize_gate.py::_gate_done_provenance` (clientInfo `cgl-sched-gate`, one-shot for done task 2273). **FAIL→resolved:** not allowlisted by decision (PRD §7); ε documents it | PASS (resolved) |
| `DeterministicProvenanceCallerNotPermitted`, `middleware/done_provenance_authz.py` | absent on main | producer:this-task |

## task ζ — review surfaces, skills, operator docs, pointers (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Thin-script exemplar | `scripts/check_consolidation_closure.py` (exit 0/1/2; verdict from `evaluate_closure`); its test exists only on `task/4808` (`fused-memory/tests/test_check_consolidation_closure_cli.py`) — **FAIL test-absent on main → resolved** via ζ ← δ ← 4808; `scripts/tests/conftest.py` supplies `make_tasks_db` / sys.path | PASS producer-upstream |
| Skill step to amend | `skills/curate-fused-memories/SKILL.md` "Phase 3 — Execute" step 3 ("Close the gate by RESOLVING ITS ESCALATION… hand-passing `deterministic-gate` would be silently ACCEPTED. That is the trap") — ε makes the trap a refusal, ζ re-words + adds 3a; stale line "nothing in this sitting should assume a grouped or anchored read exists" under "The ratified corpus shape" contradicted by `services/topic_anchor.py` (task 3111) | PASS wired |
| Stamp key + its consumer | `shared/src/shared/task_metadata.py::HUMAN_CURATOR_ADJUDICATED_AT_KEY`; `deterministic_runner.py::_curator_adjudication_confirmed` | PASS wired (skill step = producer:this-task) |
| Watcher playbook | `skills/recon-escalation-watcher/SKILL.md` "Per-Category Playbook" (bullet list); `recon_auto_consolidation_refused` declared by δ | PASS producer-upstream (δ) |
| Ops paragraph | `OPERATIONS.md` "Fused-memory has its own, separate green tier." | PASS wired |
| Accretion-line home | pin caveat in both `server/tools.py::FUSED_MEMORY_INSTRUCTIONS` and the `search` docstring → ζ writes it once in `FUSED_MEMORY_INSTRUCTIONS` (open question 5 resolved) | PASS wired |
| Cross-PRD anchors | `docs/prds/memory-write-path-convergence.md` "C2 — Consolidation contract (thrust B)" + §9 ι bullet; `docs/prds/memory-metadata-vocabulary.md` "3. The corpus-shape decision" | PASS wired |
| Report inputs | `server/tools.py::get_memories_by_metadata` (limit 1000, `truncated`/`total`); `add_system_record` `record_kind` is caller convention; `list_by_kind` / `revert_auto_consolidation` from γ/δ | PASS producer-upstream |

## task η — end-to-end integration gate (leaf; its signal is §5 of the PRD)

| Capability | Evidence | Verdict |
|---|---|---|
| Fake `MemoryService` | `fused-memory/tests/test_consolidate_memories_tool.py::make_service` (topic_members, canonical_peers, update_errors, scroll_error); `tests/_fm_helpers.py::FakeMemoryLookup` | PASS wired |
| Running the consolidator in a test | `fused-memory/tests/reconciliation/consolidator_fixtures.py::make_consolidator` | PASS wired |
| Ledger + queue doubles | `ReconLedgerStore(tmp_path / 'ledger.db')` (no `:memory:` precedent); `escalation.queue.EscalationQueue(tmp_path / 'esc')` assigned to `_escalation_queue` (`tests/test_harness.py`) | PASS wired |
| Closure-seam precedent | `fused-memory/tests/test_consolidation_closure_seam.py` | PASS wired |
| Every producer upstream | α, β, γ, δ via η ← δ; **B11 needs ε** — **FAIL producer-not-in-closure → resolved** by adding η ← ε | PASS producer-upstream |

## task θ — migrate the open gates (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Direct `MemoryService` caller precedent | `fused-memory/scripts/retro_stamp_topics.py` (`FusedMemoryConfig()` → `MemoryService(config)` → `initialize()`; `--apply` gated by `assert_store_mutation_allowed`) | PASS wired |
| Gate enumeration | `curator_gate_resolution_sweep.py::extract_open_gate_task_ids` (ids only, includes non-consolidation gates) + `consolidation_gate.py::GATE_METADATA_KEY` (`x_recon_consolidation_gate`) for the topic; titles vary — never key on title | PASS wired, extent noted in PRD |
| Closure check | `consolidation_gate.py::evaluate_closure` on main; `resolve_unstamped_live_ids` via δ ← 4808 | PASS producer-upstream |
| Resume semantics | `escalation/src/escalation/server.py::resolve_issue` stamps `resolution_action` + `queue.resolve`; the status flip is orchestrator-side (`harness.py::_cascade_unblock_member` via `escalation/action_effects.py`); `granted_files` is the only metadata write | PASS wired |
| Open-gate count premise | live: **1** dark_factory gate (5183) — **FAIL bound (PRD said 7) → resolved**: PRD §6 D13 / §10 θ corrected; the test fixture supplies its own gates | PASS (resolved) |
| Predicate + ops | α/β via δ | PASS producer-upstream |
