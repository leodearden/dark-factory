# Capability manifest — S13 recon sweeps, flag identity and recon-report state

PRD: `plans/review-all-s13-recon-sweeps-flag-identity-and-recon-report-state-prd.md`. YAML twin:
`plans/review-all-s13-recon-sweeps-flag-identity-and-recon-report-state-prd.capability-manifest.yaml`
(every `task_id: null` until decompose; signal leaf ω). Folded 2026-11-11 (`critique/S13.fold.md`); ruling
fold 2026-11-13 (`critique/S13.rulingfold.md`): every question ruled, ξ unconditional and refusing, ε grown
by the refusal record, π and ρ added, η's Stage-1 prompt rewrite definite. Fold-verified 2026-11-13
(`critique/S13.foldverify.md`) at main `4a87282ee7`: ρ is filed by the lead once π is on main, π's script is
committed executable, η also retires the prompt's task-count example (one more producer check, 71 in all,
every one FAILS at `4a87282ee7`; no declared path changed since `a85cb1ad78`).

**Code anchors** re-verified for the ruling fold against main `a85cb1ad78` (2026-11-13); authored against
`1f240e6da2` (2026-11-10) and re-checked at `97be1d71fa` (2026-11-11). Bindings cite symbols, not lines.
Every `grep:` pointer below was resolved against main at `a85cb1ad78` (the path exists; the dotted name is a
module-level `def`/`class`/assignment or `Class.method`; script `scratch/S13-rulingfold/check_pointers.py`).
Every mechanical `delivered_check` in the twin was evaluated at `a85cb1ad78` and run through
`shared/src/shared/delivered_check_polarity.py::lint_delivered_checks` with the leaf's declared files
(script `scratch/S13-rulingfold/eval_sidecar.py`): 70 producer checks, every one FAILS at `a85cb1ad78` and
passes once its leaf lands; 0 rejects; two `absent_overbroad` warnings, both on names a predecessor moves
into the leaf's declared file before the leaf runs: δ2's `terminal-literal-gone` (it matches
`flag_dedup.py::TERMINAL_STATUSES` at main, which θ1 moves into `flag_filters/task_state_refutations.py`)
and κ3's `relay-kind-constant-gone` (it matches `prompts/__init__.py::FLAG_FOR_STAGE2_MARKER_KIND` at main,
which S12 ι2 moves into `prompts/fragments.py`). Substrate rows (`*-substrate`) carry no mechanical check,
so none is copied to a task as a producer check. Upstream producers (tasks 9203, 6721; S12 γ2 and δ) are
bound by id or label and gate through their own checks. No check is a `kind: script` check (π's script does
not exist on main). Checks on modules S19 moves into `fused_memory.recon_state` (`recon_ledger.py`,
`event_journal.py`, `event_queue.py`, `flag_record_contract.py`, `consolidation_gate.py`) scan
`fused-memory/src/` for a symbol pattern, so they hold under either order. Leaves κ1–κ3 are written under
OQ-7 (b), η under OQ-1 (b) and OQ-8 (b′), ε under OQ-2 (a) with Leo's additions, all ruled by Leo.


## α — Give recon flag identity one public module: fingerprint, normalizer, signatures, FlagTypeFamily, UUID extraction and the parsed flag subject

Closes fk-e30ecd1f36b8, fk-e64ee986eeed. High. No dependency. Persisted forms are frozen byte-for-byte (D1); the golden fixture captures today's outputs. 20 declared paths, 6 of them one-line test import repoints of the moved signature functions.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| flag-identity-module | built here — fused-memory/src/fused_memory/reconciliation/flag_identity.py::parse_flag_subject (C1) | PASS | grep present `^def parse_flag_subject\(` in `fused-memory/src/fused_memory/reconciliation/flag_identity.py` |
| private-fingerprint-imports-gone | anti-inversion — scope_freshness and task_knowledge_sync import content_fingerprint from flag_identity, never flag_dedup's private name | PASS | grep absent `[^A-Za-z0-9]_content_fingerprint` in `fused-memory/src/fused_memory/reconciliation/scope_freshness.py`, `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py` |
| recon-report-local-normalizer-gone | anti-inversion — server/recon_report.py imports the identity functions from flag_identity | PASS | grep absent `^def _normalize_description\(` in `fused-memory/src/fused_memory/server/recon_report.py` |
| persisted-forms-golden-fixture | built here — fused-memory/tests/reconciliation/test_flag_identity.py holds the B1/B2 fixture captured from today's code | PASS | path present: `fused-memory/tests/reconciliation/test_flag_identity.py` |
| flag-dedup-identity-substrate | `grep:fused-memory/src/fused_memory/reconciliation/flag_dedup.py::compute_marker_signature` substrate — is on main; α moves it | PASS | — |

## β — Make recon flag suppression match any component of a composite task id and record partial matches on the suppression row

Closes fk-5d00812d5256 under R13 option 1. Deps δ1 (through it α); S19 α (lead ruling OQ-4). filter_suppressed gains a run_id keyword; the 8 test calls of is_suppressed move or go.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| partial-hits-recorded | built here — flag_dedup.py::filter_suppressed records partial_hits on the matching row (D2) | PASS | grep present `partial_hits` in `fused-memory/src/fused_memory/reconciliation/flag_dedup.py` |
| verbatim-flag-compare-gone | anti-inversion — the flag side is parsed into a FlagSubject | PASS | grep absent `tid_str = str\(flag_tid\)` in `fused-memory/src/fused_memory/reconciliation/flag_dedup.py` |
| ledger-is-suppressed-deleted | anti-inversion — no second suppression-matching rule in the ledger | PASS | grep absent `def is_suppressed\(` in `fused-memory/src/` |
| conditional-upsert-substrate | `grep:fused-memory/src/fused_memory/reconciliation/recon_ledger.py::ReconLedgerStore.upsert_if_state` substrate — exists on main; β writes partial hits through it so a lifted row is never revived | PASS | — |

## γ — Put every recon flag_type vocabulary on FlagTypeFamily with a drift log, and guard against ad-hoc flag_type matching

Closes fk-65aeb80b161d. Dep β. Subsumes task 7388 (cancelled). EXACT mode for the absence filter, which gates delete_memory (D3).

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| family-match-modes | built here — flag_identity.py::FamilyMatch (C2) | PASS | grep present `^class FamilyMatch\(` in `fused-memory/src/fused_memory/reconciliation/flag_identity.py` |
| absence-frozenset-gone | anti-inversion — the absence vocabulary is a FlagTypeFamily(match=EXACT) | PASS | grep absent `ABSENCE_FLAG_TYPES` in `fused-memory/src/` |
| vocabulary-guard-test | built here — the AST guard test, exercised against a seeded fixture module (INV-10) | PASS | path present: `fused-memory/tests/reconciliation/test_flag_type_vocabulary_guard.py` |

## δ1 — Split task_filter.py: the recon prose grammar moves to recon_prose_grammar.py, task_filter keeps the task-tree renderer

Closes fk-d9bfbf342b5b. Dep α. 23 declared paths, 16 of them one-line import repoints. Placement (lead ruling OQ-4): S19 β2 moves recon_prose_grammar.py in place of task_filter.py.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| grammar-module | built here — reconciliation/recon_prose_grammar.py holds the grammar half verbatim | PASS | grep present `^COUNT_SNAPSHOT_RE` in `fused-memory/src/fused_memory/reconciliation/recon_prose_grammar.py` |
| task-filter-holds-no-regex | anti-inversion — the tree half compiles no regex (measured: all 17 re.compile sites are in the grammar half) | PASS | grep absent `re\.compile\(` in `fused-memory/src/fused_memory/reconciliation/task_filter.py` |

## δ2 — Derive the recon status lexicon and terminal sets from shared.task_statuses; make active-tree selection a typed value

Closes fk-51ec1e4010af, fk-9f66b6f4420c. Dep θ2; shares only server/tools.py import lines with S12 δ (rebase-only). status_rank keeps today's ranks (infra-hold 3, cancelled 7), pinned by an order test. Task 9285 depends on δ2 (lead action); S19 β2 depends on δ2 (lead ruling OQ-4).

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| lexicon-from-taskstatus | built here — recon_prose_grammar.py::STATUS_WORD_ALTERNATION generated from TaskStatus (C3) | PASS | grep present `^STATUS_WORD_ALTERNATION` in `fused-memory/src/fused_memory/reconciliation/recon_prose_grammar.py` |
| status-rank-public | built here — task_filter.py::status_rank, total over TaskStatus (D6) | PASS | grep present `^def status_rank\(` in `fused-memory/src/fused_memory/reconciliation/task_filter.py` |
| terminal-literal-gone | anti-inversion — terminal sets are shared.task_statuses.TERMINAL (the three named recon sets, 3 matching lines at 97be1d71fa) | PASS | grep absent `^(TERMINAL_STATUSES\|INACTIVE_TASK_STATUSES\|TERMINAL_TASK_STATUSES)( \|:\|=)` in `fused-memory/src/` |
| five-tuple-gone | anti-inversion — _select_visible_active_with_body returns a frozen VisibleActiveSelection | PASS | grep absent `-> tuple\[list\[dict\], str \\| None, str, str, str\]` in `fused-memory/src/fused_memory/reconciliation/task_filter.py` |
| task-statuses-substrate | `grep:shared/src/shared/task_statuses.py::TaskStatus` `grep:shared/src/shared/task_statuses.py::TERMINAL` substrate — are on main | PASS | — |

## ε — Make the recon finding category a closed enum: refuse off-list categories at add_finding, record every refusal for analysis, and generate the stage schema from it

Closes fk-7c13df0b988c. Deps ζ, θ3, S12 ι2 (prompts/fragments.py). OQ-2 ruled (a) by Leo plus his additions: the refusal is recorded in the append-only `rejected_findings` table (`server/recon_report_store.py`, 180-day retention, untouched by run cleanup), a same-run accepted retry stamps `retried_as`, and the stage report counts `findings_rejected_invalid_category` (C11, D18). The five producer category constants become FindingCategory members. 21 declared paths.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| finding-category-enum | built here — models/reconciliation.py::FindingCategory (C5) | PASS | grep present `^class FindingCategory\(` in `fused-memory/src/fused_memory/models/reconciliation.py` |
| stage-schema-enum-derived | built here — cli_stage_runner.py::FINDING_ITEM_SCHEMA category enum is list(FINDING_CATEGORY_VALUES) | PASS | grep present `list\(FINDING_CATEGORY_VALUES\)` in `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py` |
| prefix-rule-gone | anti-inversion — actionable default reads FindingCategory.informational | PASS | grep absent `startswith\('cross_project'\)` in `fused-memory/src/fused_memory/server/recon_report.py` |
| producer-category-literals-gone | anti-inversion — the producer category constants are FindingCategory members, not string literals | PASS | grep absent `^(GATE_RESOLUTION_FLAG_CATEGORY\|ORPHANED_ESCALATION_FLAG_CATEGORY\|UNLANDED_PATH_FLAG_CATEGORY)(: str)? = '` in `fused-memory/src/` |
| rejected-findings-table | built here — server/recon_report_store.py owns the rejected_findings table (C11) | PASS | grep present `CREATE TABLE IF NOT EXISTS rejected_findings` in `fused-memory/src/fused_memory/server/recon_report_store.py` |
| refusal-recorded-by-add-finding | built here — ReconReportState.add_finding records each off-list refusal through ReconReportStore.record_rejection (D18) | PASS | grep present `record_rejection\(` in `fused-memory/src/fused_memory/server/recon_report.py` |
| refusal-stage-stat | built here — the stage entry's findings_rejected_invalid_category stat reaches the stage report (D18) | PASS | grep present `findings_rejected_invalid_category` in `fused-memory/src/fused_memory/server/recon_report.py` |
| retry-stamp | built here — an accepted add_finding stamps retried_as on the same run's refusals of the same digest (C11) | PASS | grep present `stamp_retried_as\(` in `fused-memory/src/fused_memory/server/recon_report.py` |
| severity-precedent-substrate | `grep:fused-memory/src/fused_memory/models/reconciliation.py::FindingSeverity` `grep:fused-memory/src/fused_memory/models/reconciliation.py::FINDING_SEVERITY_VALUES` substrate — on main; the add_finding refusal mirrors ReconReportState.add_finding's invalid_severity arm | PASS | — |

## ζ — Give recon_report one RunFindingIndex per run that owns its session and dedup maps

Closes fk-0791e9817b84. High. Dep α.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| run-finding-index | built here — server/recon_finding_index.py::RunFindingIndex (C4) | PASS | grep present `^class RunFindingIndex` in `fused-memory/src/fused_memory/server/recon_finding_index.py` |
| hand-synced-run-maps-gone | anti-inversion — ReconReportState holds no run-scoped dedup map of its own | PASS | grep absent `self\._run_sig_index` in `fused-memory/src/fused_memory/server/recon_report.py` |
| install-entry-substrate | `grep:fused-memory/src/fused_memory/server/recon_report.py::ReconReportState._install_entry` substrate — is the existing single installer the index absorbs | PASS | — |

## η — Give both stale-edge sweeps one Graphiti edge-sweep primitive, take the agent id from StageId, and stop recon writing task status into memory (the sweep's replacement sentence and the Stage-1 prompt's status facts)

Closes fk-917fdf32a542, fk-cdb8c20a3d0f. OQ-1 (b) and OQ-8 (b′) ruled by Leo: the sweep invalidates and writes nothing; the Stage-1 prompt's two status sections tell the agent to invalidate and write nothing for a task's status (D19); the detector stays as a backstop. Deps κ2, δ2; out of batch 9203 (S26 μ, C7 read surface), 6721 (S14 μ), 6718 (S14 θ, same sweep module and test), 7941, 5272 (6598 done).

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| edge-sweep-primitive | built here — reconciliation/graph_edge_sweep.py::invalidate_edges (C6) | PASS | grep present `^async def invalidate_edges\(` in `fused-memory/src/fused_memory/reconciliation/graph_edge_sweep.py` |
| private-agent-id-gone | anti-inversion — the agent id comes from recon_stage_agent_id(StageId) | PASS | grep absent `_SWEEP_AGENT_ID` in `fused-memory/src/` |
| agent-id-literal-gone | anti-inversion — no hand-written stage agent id assignment in reconciliation | PASS | grep absent `= 'recon-stage-memory_consolidator'` in `fused-memory/src/fused_memory/reconciliation/` |
| replacement-status-prose-gone | anti-inversion (OQ-1 b) — the sweep invalidates and writes no status sentence | PASS | grep absent `_SUPERSEDE_FACT_TEMPLATE` in `fused-memory/src/` |
| sweep-writes-no-replacement-guidance | built here — prompts/stage1.py tells the agent the sweep writes no replacement status fact | PASS | grep present `do not write a replacement status fact` in `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py` |
| stage1-status-flip-two-step-gone | anti-inversion (OQ-8 b′) — the Stage-1 prompt no longer scopes the invalidate-and-write two-step to status flips | PASS | grep absent `status flips` in `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py` |
| supersede-stats-gone | anti-inversion (OQ-1 b) — the consolidator reports no superseding-write stats | PASS | grep absent `stale_blocked_edges_supersede` in `fused-memory/src/` |
| stage1-task-count-example-gone | anti-inversion (OQ-8 b′) — the Stage-1 prompt's effective-date example is no longer a task-status count (the recon-stage count-snapshot gate refuses that write anyway) | PASS | grep absent `3 blocked, 18 done` in `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py` |
| checked-read-surface-upstream | producer:task-9203 upstream (S26 μ, C7 MemoryService.enumerate_valid_edges); its own delivered checks gate it | PASS | — |
| recon-stage-agent-id-upstream | producer:task-6721 upstream (S14 μ, shared/agent_identity.py::recon_stage_agent_id) | PASS | — |
| stage-id-substrate | `grep:fused-memory/src/fused_memory/models/reconciliation.py::StageId` substrate — on main | PASS | — |

## θ1 — Move flag_dedup's task-state refutation and systemic-pattern filters, with their shared lookup helpers, into flag_filters modules

Part of fk-46514b4ef0cc. Dep γ; touches no file S12 edits. Subsumes task 5996 with θ2, θ3.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| refutation-filters-module | built here — reconciliation/flag_filters/task_state_refutations.py | PASS | grep present `^async def filter_false_absence_flags\(` in `fused-memory/src/fused_memory/reconciliation/flag_filters/task_state_refutations.py` |
| absence-filter-left-flag-dedup | anti-inversion — moved, not copied | PASS | grep absent `^async def filter_false_absence_flags\(` in `fused-memory/src/fused_memory/reconciliation/flag_dedup.py` |
| chain-order-substrate | `grep:fused-memory/src/fused_memory/reconciliation/stage1_flag_chain.py::RESOLVING_FILTERS_BEFORE_SUPPRESSION` substrate — is the one statement of filter order; θ1 only repoints its imports | PASS | — |

## θ2 — Move flag_dedup's count-snapshot, authorship and task-text accounting filters into flag_filters modules

Part of fk-46514b4ef0cc. Dep θ1.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| count-snapshot-module | built here — reconciliation/flag_filters/count_snapshot.py | PASS | grep present `^def filter_stale_count_snapshot_corrections\(` in `fused-memory/src/fused_memory/reconciliation/flag_filters/count_snapshot.py` |
| authorship-filter-left-flag-dedup | anti-inversion — moved, not copied | PASS | grep absent `^async def filter_style_only_authorship_flags\(` in `fused-memory/src/fused_memory/reconciliation/flag_dedup.py` |

## θ3 — Move the phantom-creation filter, Hook A and the Stage-3 finding guards out of flag_dedup, leave it the ledger lifecycle, and rehome its prose

Closes fk-46514b4ef0cc with θ1, θ2; carries fk-2868338d3cf3. Deps θ2, δ2, ι and S12 ι2 (lead ruling OQ-6). Repoints _select_proactive_sample onto task_filter.status_rank.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| stage3-guards-module | built here — reconciliation/flag_filters/stage3_findings.py | PASS | grep present `^def filter_blocked_snapshot_findings\(` in `fused-memory/src/fused_memory/reconciliation/flag_filters/stage3_findings.py` |
| ceiling-guard-left-flag-dedup | anti-inversion — flag_dedup keeps filter_suppressed as its only filter_* function | PASS | grep absent `^def filter_contamination_ceiling_findings\(` in `fused-memory/src/fused_memory/reconciliation/flag_dedup.py` |
| status-priority-private-import-gone | anti-inversion — _select_proactive_sample imports task_filter.status_rank at module top | PASS | grep absent `_STATUS_PRIORITY` in `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py` |
| flag-dedup-prose-rehomed | built here — docs/recon-flag-dedup.md holds the narrative the module docstring drops (D16) | PASS | path present: `docs/recon-flag-dedup.md` |

## ι — Split task_knowledge_sync.py: IntegrityCheck, the shared report formatters, snapshot I/O, the growth sweep and the reify briefing hook move to their own modules

Closes fk-b1cda781c60e. Deps α, 9203; lands before S12's stage-interface, streak-alarm and task-write tasks, which depend on it (lead ruling OQ-6). Subsumes task 6846.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| integrity-check-module | built here — reconciliation/stages/integrity_check.py::IntegrityCheck, imported by the harness from its own module (the path once existed and was removed by task 157, so the check reads the importer) | PASS | grep present `from fused_memory\.reconciliation\.stages\.integrity_check import` in `fused-memory/src/fused_memory/reconciliation/harness.py` |
| integrity-check-left-stage2-module | anti-inversion — moved, not copied | PASS | grep absent `^class IntegrityCheck\(` in `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py` |
| growth-sweep-module | built here — reconciliation/standing_decision_growth_sweep.py | PASS | path present: `fused-memory/src/fused_memory/reconciliation/standing_decision_growth_sweep.py` |

## κ1 — Delete Stage 2's dormant flag-persistence and escalation markers and the dormant human-operator stall marker; give recon marker GC its own module

Carries fk-9165ae3a1e72 (high) under OQ-7 (b), ruled by Leo. Dep ε; S19 β1 (lead ruling OQ-4). Subsumes task 7807. 16 declared paths, 4 of them prose-only.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| marker-gc-module | built here — reconciliation/marker_gc.py::gc_recon_markers (C7) | PASS | grep present `^async def gc_recon_markers\(` in `fused-memory/src/fused_memory/reconciliation/marker_gc.py` |
| fix-d-persistence-tracking-gone | anti-inversion (OQ-7 b) — FIX D's Mem0 persistence tracking is deleted, not migrated | PASS | grep absent `^async def _track_flag_persistence\(` in `fused-memory/src/` |
| persistence-mem0-sweep-gone | anti-inversion (OQ-7 b) — no Mem0 persistence-marker pool remains to sweep | PASS | grep absent `^async def _sweep_stale_persistence_markers\(` in `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py` |
| stall-marker-gone | anti-inversion (OQ-7 b) — the dormant human-operator stall marker and its escalation are deleted | PASS | grep absent `_STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE` in `fused-memory/src/` |
| ledger-store-substrate | `grep:fused-memory/src/fused_memory/reconciliation/recon_ledger.py::ReconLedgerStore.gc` `grep:fused-memory/src/fused_memory/reconciliation/recon_ledger.py::MARKER_KINDS` substrate — on main; gc_recon_markers keeps calling the ledger's own gc | PASS | — |

## κ2 — Delete the dormant Stage-1→Stage-2 Mem0 flag relay and the Mem0 marker-pool sweeper

Closes fk-9165ae3a1e72 with κ1, κ3, and fk-fa1fd33c7a9e, under OQ-7 (b), ruled by Leo. High. Dep κ1. 17 declared paths, almost all deletions.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| stage2-relay-read-gone | anti-inversion (OQ-7 b) — Stage 2 reads Stage 1's findings only through the recon-report channel | PASS | grep absent `^async def _query_stage2_flags\(` in `fused-memory/src/` |
| mem0-pool-sweeper-gone | anti-inversion — no Mem0 marker pool remains to sweep | PASS | grep absent `def _sweep_stale_mem0_pool\(` in `fused-memory/src/` |
| semantic-relay-search-gone | anti-inversion — no top-N semantic search for relayed flags | PASS | grep absent `query='stage 1 flag for stage 2'` in `fused-memory/src/` |
| terminal-task-relay-retirement-gone | anti-inversion — the latency layer that retired relay markers on task done goes with the pool | PASS | grep absent `def retire_flag_markers_for_terminal_task\(` in `fused-memory/src/` |
| recon-report-channel-substrate | `grep:fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py::_query_recon_report_findings` substrate — on main; the one Stage-1→Stage-2 channel that stays | PASS | — |

## κ3 — Refuse recon-stage writes to the retired Stage-1→Stage-2 Mem0 relay and retire the relay from guidance, vocabulary and operator scripts

Closes fk-9165ae3a1e72 with κ1, κ2, under OQ-7 (b), ruled by Leo. High. Dep κ2; S19 β2 (lead ruling OQ-4) (moves flag_record_contract.py and consolidation_gate.py, so checks scan fused-memory/src/).

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| relay-write-refusal | built here — flag_record_contract.py::FLAG_FOR_STAGE2_RELAY_RETIRED returned by recon_stage_flag_kind_refusal for recon-stage writers (C7) | PASS | grep present `flag_for_stage2_relay_retired` in `fused-memory/src/` |
| relay-kind-constant-gone | anti-inversion — no prompt-package constant names a kind no prompt writes | PASS | grep absent `^FLAG_FOR_STAGE2_MARKER_KIND = ` in `fused-memory/src/fused_memory/reconciliation/prompts/` |
| gate-hint-relay-gone | anti-inversion — the consolidation gate's hint names add_finding, not the retired relay | PASS | grep absent `Relay it to Stage 2 via the `flag_for_stage2`` in `fused-memory/src/` |
| flag-record-contract-substrate | `grep:fused-memory/src/fused_memory/reconciliation/flag_record_contract.py::recon_stage_flag_kind_refusal` substrate — is the recon-stage-only write boundary that gains the refusal | PASS | — |

## ν — Make TaskKnowledgeSync tests drive public seams: injected collaborators, no private imports or dotted-path patches

Closes fk-90f374fcce21 (high). Deps κ2, S12 γ2 (S12/C2 StageCollaborators); id wired at decompose. The draft's ν → S12 θ is not wired (they share only test files; lead's test-file rule). Subsumes task 7065.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| public-seam-guard | built here — the guard test that flags a seeded private import or patch (B16) | PASS | path present: `fused-memory/tests/reconciliation/test_tks_public_seam_guard.py` |
| dotted-path-patches-gone | anti-inversion — no test patches a task_knowledge_sync private by dotted path | PASS | grep absent `stages\.task_knowledge_sync\._` in `fused-memory/tests/` |
| stage-collaborator-injection-upstream | producer: S12 γ2 (S12/C2 StageCollaborators — TaskKnowledgeSync constructed with its collaborators, R12); S12 is authored in this wave and has no ids yet | OPEN | manual: a reviewer confirms at decompose that ν depends on the filed S12 γ2 task and that S12/C2 names TaskKnowledgeSync's injected collaborators |

## λ — Move per-project recon policy (snapshot-write block, contamination-ceiling retirement, briefing refresh) into config keyed by project id

Closes fk-fbbf580e4e1d. Dep ν (through it ι, θ3, ε, κ2). Read seam ReconciliationConfig.policy_for; reload applies at the next cycle.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| project-policies-config | built here — config/schema.py::ReconciliationConfig.project_policies (C9) | PASS | grep present `project_policies: dict\[str, ReconProjectPolicy\]` in `fused-memory/src/fused_memory/config/schema.py` |
| per-project-modules-gone | anti-inversion — per-project Python modules deleted | PASS | path absent: `fused-memory/src/fused_memory/reconciliation/policies/autopilot_video.py` |
| policy-read-seam | built here — config/schema.py::ReconciliationConfig.policy_for (C9 read seam) | PASS | grep present `def policy_for\(` in `fused-memory/src/fused_memory/config/schema.py` |
| project-policies-reloadable | built here — config/reload.py::RELOADABLE_FIELDS names reconciliation.project_policies | PASS | grep present `'reconciliation\.project_policies'` in `fused-memory/src/fused_memory/config/reload.py` |
| reconciliation-config-substrate | `grep:fused-memory/src/fused_memory/config/schema.py::ReconciliationConfig` substrate — on main, already keying per-project values by project_id | PASS | — |

## μ — Run EventJournal writes on one writer thread; EventQueue.enqueue awaits the durable append off the event loop

Closes fk-4c8d19c64d99. No intra-batch dependency; S19 α (lead ruling OQ-4). Durability shape (lead ruling OQ-5 (a)); the dead-letter replay runs off the loop thread only.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| journal-async-append | built here — event_journal.py::EventJournal.append runs on the writer thread (C8) | PASS | grep present `^    async def append\(self, event` in `fused-memory/src/` |
| enqueue-awaits-append | built here (OQ-5 a) — event_queue.py::EventQueue.enqueue is async | PASS | grep present `async def enqueue\(self, event: ReconciliationEvent` in `fused-memory/src/` |
| on-loop-replay-branch-gone | anti-inversion — replay never enqueues inline on the loop thread; _enqueue_on_loop awaits enqueue (D15) | PASS | grep absent `return self\.enqueue(\(event\))?$` in `fused-memory/src/` |
| writer-thread-precedent-substrate | `grep:fused-memory/src/fused_memory/reconciliation/dead_letter_writer.py::DeadLetterWriter` substrate — is the single-thread writer precedent | PASS | — |

## ξ — The recon-report server takes the run id from the stage connection, refuses a differing run_id argument, and the stage prompts stop asking for run_id

Unconditional (lead ruling S12/OQ-10 (a)) under Leo's S12-2 refusal rule: a `run_id` differing from `CAUSATION_ID_HEADER` is refused through S12/C3's `stage_identity_refusal` with `StageIdentityRefusal` (`field='run_id'`), one `recon_stage_identity_refused` log line, no override event; the echo-run_id prompt lines go in the same change. Consumes S12/C3; deps ζ, ε, κ3, η, λ and S12 δ. C10, D20.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| recon-report-reads-stage-headers | built here — server/recon_report.py resolves run_id from CAUSATION_ID_HEADER through _report_run_id for every tool whose signature takes run_id (C10) | PASS | grep present `CAUSATION_ID_HEADER` in `fused-memory/src/fused_memory/server/recon_report.py` |
| run-id-mismatch-refused | built here — _report_run_id returns S12/C3's StageIdentityRefusal with field='run_id' for a differing argument (D20) | PASS | grep present `field=['"]run_id['"]` in `fused-memory/src/fused_memory/server/recon_report.py` |
| recon-report-run-id-echo-gone | anti-inversion — no hand-written recon-report call example in the stage prompts passes the run id | PASS | grep absent `run_id=<from Reconciliation Context>` in `fused-memory/src/fused_memory/reconciliation/prompts/` |
| stage-identity-refusal-upstream | producer: S12 δ (S12/C3 stage_identity_refusal, StageIdentityRefusal, the headers on the recon-report connection); ξ → S12 δ is the gate | PASS | — |

## π — Add the recon finding-category candidate check script and its tests

Leo's addition to OQ-2. Dep ε (C11's `read_rejected_findings`). The deterministic half of the analysis (D17): exit 1 on a candidate label or a lost share above threshold, exit 0 with a verdict, exit 75 when the table is not there yet. Thresholds are script arguments. Read-only.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| candidate-check-script | built here — fused-memory/scripts/check_recon_category_candidates.py | PASS | path present: `fused-memory/scripts/check_recon_category_candidates.py` |
| candidate-check-tests | built here — the B18 tests seed rejected_findings rows and assert every exit (INV-10) | PASS | path present: `fused-memory/tests/test_check_recon_category_candidates.py` |
| candidate-check-executable | built here — committed with git mode 100755 and a `#!/usr/bin/env python3` line: the runner execs it directly (`orchestrator/src/orchestrator/before_done_script.py::run_script`) and `submit_task` refuses ρ unless it passes `os.X_OK` | PASS | manual: a reviewer confirms `git ls-files -s` shows 100755 on π's merge; a path check cannot see the mode |
| rejected-findings-reader-upstream | producer: ε (C11 read_rejected_findings and RejectedFindingsTableMissing); π → ε is the gate | PASS | — |

## ρ — Recon finding-category candidates: fortnightly check

Leo's addition to OQ-2: the seed link of a native recurrence chain (docs/task-authoring.md §6.1). Filed by the lead the day π's merge lands, not at decompose: `submit_task` refuses a `before_done` script that is not on main and executable (`fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_script_under`, planning mode included); the lead stamps the twin's `task_id` then. `task_kind=deterministic`; `before_done {kind: predicate, script: fused-memory/scripts/check_recon_category_candidates.py, args: the five thresholds, timeout_secs: 120, recheck_secs: 3600}` (`recheck_secs` is a fold addition, confirmed by the lead); dated milestone about 30 days after filing; `recurrence {key: recon-category-candidates, interval_secs: 1209600}`. Dep π. No files, no finding key. A firing link files a born-at-L2 `milestone_check_failed` escalation and blocks its chain until resolved; the first live recurring chain in this project.

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| recurrence-substrate | `grep:shared/src/shared/task_metadata.py::Recurrence` `grep:fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_recurrence` `grep:fused-memory/src/fused_memory/middleware/recurrence_mint.py::run_remint_sweep_loop` substrate — the recurrence carrier contract, mint and re-mint sweep are on main | PASS | — |
| predicate-runner-substrate | `grep:orchestrator/src/orchestrator/deterministic_runner.py::DeterministicRunner._run_predicate` substrate — the predicate exit-code contract (0 done, 75 re-arm with recheck_secs, other milestone_check_failed) is on main | PASS | — |
| candidate-script-upstream | producer: π (the script ρ runs); ρ → π is the gate, so no link fires before the script is on main | PASS | — |
| seed-link-shape | the seed link is filed with task_kind=deterministic, the predicate before_done, a dated milestone and the recurrence key, and is accepted by submit_task's carrier validation | PASS | manual: the lead, filing the link once π's script is on main, reads it back with get_task and confirms the three metadata keys and the edge onto π; a malformed carrier, or a script that is missing or not executable, is rejected at submit_task and never persisted |

## ω — Run the recon flag and report seams' two-way boundary suite and the live-run check (integration gate)

Signal leaf. Depends on every other leaf except ρ (the recurring chain's seed link; its first link fires about 30 days after filing and its links recur after the program).

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| boundary-suite | built here — the two-way boundary suite rows B3, B4, B7, B10, B11b, B12, B13, B14b, B17 | PASS | path present: `fused-memory/tests/reconciliation/test_recon_flag_identity_boundary.py` |
| live-cycle-check | one live dark_factory recon cycle after the last deploy shows findings reaching Stage 2 only through the recon-report channel, no recon-stage Mem0 write carrying a retired marker key or a 'has status' sentence, no accepted off-vocabulary category and a rejected_findings row per counted refusal, EdgeSweepStats keys for both sweeps, no unexplained recon_stage_identity_refused line | PASS | manual: live-run reads (the Stage-2 report stats, write_journal.db, recon_report_state.db, the Stage-1 report) after deploy; not a tree property |

## Acceptance record (not a sidecar block)

fk-9b76dd8a1cdb (OQ-1 and OQ-8, ruled by Leo): filed `cancelled` at decompose with `metadata.x_acceptance_reason` (text in the PRD's decomposition plan: the detector stays as a cleanup backstop for other writers' status prose until the recall-from-the-task-store design session sets the write and read rules; re-raise if it lands a write rule for every writer); `files: []`; no capability to bind.
