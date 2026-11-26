# Capability manifest: live task facts at recall and at write (fused recall, plan 1)

**PRD:** `plans/live-task-facts-at-recall-prd.md` · **Sidecar:** `plans/live-task-facts-at-recall-prd.capability-manifest.yaml`
**Code anchors** verified against main `60db54fc74` (2026-11-14). Cited by symbol.

Each task's signal is bound to the capabilities it asserts. Every binding is PASS: none is declared-only, test-only,
a downstream producer or an absent producer. Each `grep:` pointer was resolved against main by parsing the file at
`main:<path>`. Each built-here pattern was confirmed absent on main with `git grep -E` over the stated paths, so its
check goes from failing to passing when its producer lands. The retirement checks run the other way: each names a
pattern present on main today that must become absent.

**Decompose-time correction.** η's row named `skills/memory/SKILL.md`, which does not exist. The `/memory`
reference is `.claude/commands/memory.md`; the PRD row was corrected in the same commit as this manifest.

**Symbols the grammar cannot point at.** Several anchors are functions nested inside another function's body, so
no `::<dotted.name>` pointer can name them; each is bound to its module with the name in prose:
`_batched_task_statuses`, `_claim_task_statuses`, `_verify_mint_referent`, `_completion_claim_gate` and
`_premature_completion_block` are closures inside the tool registration in `server/tools.py`.
`_format_memory_result` in the context assembler is module-level, not a method as the PRD's prose implies.

**G5 (B + H).** ω is the integration gate; its signal is the PRD's Boundary-test sketch (B1–B17) against the merged
tree, plus the snapshot deletion script run once per registered project. `signal_leaves` = ω.

**G7.** Walked against `docs/legibility/design-invariants.md` at decompose. The PRD's advisory walk stands
(`one-fact-one-home`, `no-silent-fail-soft`, `loop-thread-occupancy-bounded`, `readers-prove-their-producer`,
`guards-exercise-behaviour`, `no-lockstep-duplication`). Also checked: `contracts-machine-checked` (C3's shape is
pinned by B17's shared fixture across the two renderers); `structured-facts-at-failure` (`COULD_NOT_CHECK`
carries a `reason`); `storm-escape-required` (ι removes an alarm and κ adds none); `corroborate-before-acting`,
`status-matches-liveness`, `holds-owned-and-bounded`, `exceptions-owned-or-ratified` (C5's census lists every
code writer) and `sampler-vocabulary-one-table` raise nothing. No waiver.

**Checks adjusted at commit (2026-11-14).** The polarity linter warned that three removal patterns also matched
files outside their task's declared files. Each was narrowed to the definition or import site the code cannot keep
once the task lands (ι: the switch's definition, every import of the cadence module; κ: the snapshot gate's
`error_type` literal). ι's files gained `flag_dedup.py`, which imports the cadence module. Two Stage-1 prompt checks
moved here from task 9739, which no longer edits that prompt.

**Producers outside this batch.** None. Every leaf builds on code on main today; the cross-plan edges run the other
way (filed review-all tasks wait on this batch).

## α — one batched task-facts read
| Capability | Binding | Verdict |
|---|---|---|
| The task store to add the ids-scoped read to | `grep:fused-memory/src/fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend` | PASS |
| The protocol and pass-through that declare it | `grep:fused-memory/src/fused_memory/backends/task_backend_protocol.py::TaskBackendProtocol` `grep:fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor` | PASS |
| The nested status helper and its two callers to replace | `grep:fused-memory/src/fused_memory/server/tools.py` (`_batched_task_statuses`, `_claim_task_statuses`, `_verify_mint_referent`, nested in the tool registration) | PASS |
| `read_task_facts`, `SqliteTaskBackend.get_task_facts`; the nested helper gone | built here; two greps absent today, one grep that must become absent | PASS |

## β — search hits name their graph endpoints; entity edges carry dates
| Capability | Binding | Verdict |
|---|---|---|
| The search conversion that leaves `entities` empty | `grep:fused-memory/src/fused_memory/services/memory_service.py::MemoryService._search_graphiti` | PASS |
| The entity-edge normaliser that yields `temporal=None` | `grep:fused-memory/src/fused_memory/services/memory_service.py::_edge_to_dict` | PASS |
| The field β fills | `grep:fused-memory/src/fused_memory/models/memory.py::MemoryResult` (`entities`, `temporal`) | PASS |
| The valid-edges read and a batched node-name read | `grep:fused-memory/src/fused_memory/backends/graphiti_client.py` | PASS |
| Hits carry endpoint names; exact-name edges carry dates | built here; manual (the fix's shape is the implementer's; B2 and B9 test it) | PASS |

## γ — bind recalled results to tasks; fix the mention scan
| Capability | Binding | Verdict |
|---|---|---|
| The scan and the node-name parser | `grep:fused-memory/src/fused_memory/utils/canonical_labels.py::scan_content` `grep:fused-memory/src/fused_memory/utils/canonical_labels.py::parse_node_name` | PASS |
| The completion-record registry | `grep:fused-memory/src/fused_memory/memory_metadata.py::is_orchestrator_completion_record` | PASS |
| The write-path consumers of the scan whose suites γ runs | `grep:fused-memory/src/fused_memory/utils/referent_resolution.py::resolve_referents` `grep:fused-memory/src/fused_memory/server/entities_gate.py::entities_gate` | PASS |
| Read facts and filled endpoints | producer:α upstream, producer:β upstream | PASS |
| `bind_tasks`, `render_task_facts` | built here; two greps absent today | PASS |

## δ — read replies and reconciliation's stage context carry live task facts
| Capability | Binding | Verdict |
|---|---|---|
| The read tool bodies | `grep:fused-memory/src/fused_memory/server/tools.py` (`search`, `get_entity`, `get_entity_by_uuid`, registered inside the tool registration) | PASS |
| The context assembler and its task read | `grep:fused-memory/src/fused_memory/reconciliation/context_assembler.py::ContextAssembler` (holds `taskmaster`) `grep:fused-memory/src/fused_memory/reconciliation/context_assembler.py::_format_memory_result` | PASS |
| The two functions it calls | producer:α upstream, producer:γ upstream | PASS |
| The server and the assembler call `bind_tasks` | built here; grep absent today | PASS |

## ε — briefings print the live task status block
| Capability | Binding | Verdict |
|---|---|---|
| The briefing renderers and reply parser | `grep:orchestrator/src/orchestrator/agents/memory_recall.py::render_entity_block` `grep:orchestrator/src/orchestrator/agents/memory_recall.py::parse_search_reply` `grep:orchestrator/src/orchestrator/agents/memory_recall.py::filter_foreign_project_results` | PASS |
| The reply field it renders | producer:δ upstream | PASS |
| The orchestrator reads `task_facts` | built here; grep absent today | PASS |

## η — write replies echo live task facts; guidance
| Capability | Binding | Verdict |
|---|---|---|
| The write tool bodies | `grep:fused-memory/src/fused_memory/server/tools.py` (`add_memory`, `add_episode`, `update_memory`) | PASS |
| The guidance renderer | `grep:fused-memory/src/fused_memory/utils/referent_resolution.py::render_referent_declaration_guidance` | PASS |
| The docs it edits | `grep:.claude/commands/memory.md` `grep:CLAUDE.md` (memory section) | PASS |
| Binding and facts | producer:α upstream, producer:γ upstream, producer:δ upstream | PASS |
| Write replies carry `task_facts` | built here; manual (δ's grep cannot tell the write arm from the read arm; B12 tests it) | PASS |

## θ — code writers carry the task as structure
| Capability | Binding | Verdict |
|---|---|---|
| The decision-notes writer | `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._write_decisions_to_memory` | PASS |
| The completion-record marker the orchestrator writes | `grep:orchestrator/src/orchestrator/completion_record.py::completion_record_metadata` | PASS |
| Reconciliation writers | `grep:fused-memory/src/fused_memory/reconciliation/targeted.py::TargetedReconciler` `grep:fused-memory/src/fused_memory/reconciliation/scope_freshness.py` `grep:fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._reconcile_status_correction` | PASS |
| The status vocabulary the census forbids | `grep:shared/src/shared/task_statuses.py` | PASS |
| The registry the echo is marked with | producer:γ upstream | PASS |
| The per-writer tests and the census test | built here; manual (test names are the implementer's; B14 tests it) | PASS |

## ι — stop writing the task-count snapshot
| Capability | Binding | Verdict |
|---|---|---|
| The writer, switch and alarm to delete | `grep:fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py::_write_task_count_snapshot` `grep:fused-memory/src/fused_memory/reconciliation/policies/__init__.py::SNAPSHOT_WRITE_BLOCKED_PROJECTS` `grep:fused-memory/src/fused_memory/reconciliation/task_count_snapshot_cadence.py` | PASS |
| The consumers that hand the snapshot to a stage | `grep:fused-memory/src/fused_memory/reconciliation/prompts/stage3.py` `grep:fused-memory/src/fused_memory/reconciliation/prompts/judge.py` `grep:fused-memory/src/fused_memory/reconciliation/recon_self_model.py` `grep:fused-memory/src/fused_memory/reconciliation/standing_decision_constants.py` | PASS |
| Writer, switch and cadence module gone | built here; three greps that must become absent | PASS |
| The deletion script | built here; manual (its name is the implementer's; ω runs it) | PASS |

## κ — retire the detector, the sweep and four status write gates
| Capability | Binding | Verdict |
|---|---|---|
| The sweep, its Stage-1 call and its per-task call | `grep:fused-memory/src/fused_memory/reconciliation/stale_status_snapshot_edge_sweep.py::sweep_stale_status_snapshot_edges_for_task` `grep:fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py` `grep:fused-memory/src/fused_memory/reconciliation/targeted.py::TargetedReconciler._retire_stale_status_snapshots` | PASS |
| The status grammar and its importers | `grep:fused-memory/src/fused_memory/reconciliation/status_snapshot_extraction.py` `grep:fused-memory/src/fused_memory/reconciliation/citation_verifier.py` `grep:fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py` | PASS |
| The count grammar and filters | `grep:fused-memory/src/fused_memory/reconciliation/task_filter.py::is_count_snapshot` `grep:fused-memory/src/fused_memory/reconciliation/flag_dedup.py` `grep:fused-memory/src/fused_memory/reconciliation/policies/know_live.py` | PASS |
| The four gates | `grep:fused-memory/src/fused_memory/server/tools.py` (`ReconLiveTaskStatusWriteRejected`, `ReconConflictingTaskStatusWriteRejected`, `ReconSnapshotWriteRejected`, `_premature_completion_block`) | PASS |
| The Stage-1 prompt | `grep:fused-memory/src/fused_memory/reconciliation/prompts/stage1.py` | PASS |
| What stays | `grep:fused-memory/src/fused_memory/services/completion_claim_gate.py` (the completion-claim check; the mixed-framing gate `ReconMixedFramingWriteRejected` in `server/tools.py`) | PASS |
| The echo that replaces the gates; the join on every model read path | producer:η upstream, producer:δ upstream, producer:ε upstream, producer:θ upstream, producer:ι upstream | PASS |
| Sweep, gates and count grammar gone; the Stage-1 status-flip clause and task-count example gone | built here; seven greps that must become absent (the last two inherited from task 9739) | PASS |

## ω — integration gate
| Capability | Binding | Verdict |
|---|---|---|
| Everything it runs | producer:κ upstream (and through it every leaf) | PASS |
| B1–B17 pass on the merged tree; the deletion script ran per project | built here; manual (a live run) | PASS |
