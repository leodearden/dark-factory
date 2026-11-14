# S13 — Recon sweeps, flag identity and recon-report state (PRD)

**Status:** draft, authored 2026-11-10 by the S13 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S13, wave 3, size L, risk medium);
critiqued and folded 2026-11-11 (`critique/S13.critic.md`, fold log `critique/S13.fold.md`). Ruling fold
2026-11-13: every open question was ruled at the wave-3 sitting (answers of 2026-11-12, residual answers of
2026-11-13); the text is authored under the rulings (§Rulings) and no question is open. Not yet decomposed.
B + H: contracts C1–C11, a two-way boundary-test sketch, and ω as the integration gate. G5 fired on blast
radius (reconciliation, server, middleware, config, prompts, scripts), on mechanism count (eleven
contracts) and on two consumer streams (S19 moves modules this plan creates; S12 edits files this plan
splits; both were authored concurrently, and S12's fold-verified draft of 2026-11-12 was read for the
seams below). Twenty-two tasks: twenty-one leaves (ξ is now unconditional; π, the category-candidate check
script, and ρ, the seed link of a fortnightly recurring check, were added by the ruling on OQ-2) and the
integration gate ω; plus one acceptance record; no held leaf. Twenty-one are filed at decompose; ρ is filed
by the lead once π's script is on main (§Decomposition plan, ρ). Ruling R13 governs, as amended for the
dormant Mem0 pools and narrowed for fk-9b76dd8a1cdb by this sitting (§Rulings); R22 (S13 promotes no
invariant) and R23 (S13 carries no weakened key) also apply.

**Code anchors** re-verified for the ruling fold against main `a85cb1ad78` (2026-11-13), and re-checked by
the fold-verifier at `4a87282ee7` (2026-11-13): none of the 179 paths this plan declares changed between the
two, so every count below holds there. Authored
against `1f240e6da2` (2026-11-10) and re-checked at `97be1d71fa` (2026-11-11). Between `1f240e6da2` and
`a85cb1ad78` the files this plan cites that changed are: `flag_dedup.py` (−2 lines: task 6621 rewrote
`_is_valid_marker_task_id`'s locality docstring), `stages/task_knowledge_sync.py` (tasks 6509 and 6621:
`TERMINAL_STATUSES` now imported from `shared.task_statuses`; still 5,467 lines), `targeted.py` (task 6234),
`citation_verifier.py` (task 6598's docstrings), `cli_stage_runner.py` (task 6553), `consolidation_gate.py`
(task 6621, one docstring line), `harness.py` (task 6158), `prompts/stage2.py` (task 4867),
`server/tools.py` (tasks 4867, 6621, 4695, 6956, 8558), `middleware/task_interceptor.py` (tasks 6621, 9276,
6236), `config/schema.py` and `config/reload.py` (tasks 8129, 6547), `config/config.yaml` (task 8129), two
scripts and their tests, and `shared/tests/loop_blocking_allowlist.py` (task 6158) (`git diff --stat
1f240e6da2..a85cb1ad78`). Every symbol this plan relies on in them was re-read and every count resting on
them re-measured at `a85cb1ad78`; all hold except the two this text now states (`flag_dedup.py` is 5,981
lines; the `_is_valid_marker_task_id` layering rationale is gone, §Findings disposition). Every other cited
file is byte-identical, so its counts hold. Main moves fast — cite by symbol; re-locate lines at
implementation time. The run pinned `6551b75aba` (2026-10-08). Counts are `measured at 1f240e6da2` and
hold at `a85cb1ad78` unless they name another tree or store; live-store figures added by the fold are
`measured at 97be1d71fa` (read-only sqlite on `data/reconciliation/reconciliation.db`, fused-memory
`count_memories_by_metadata`); figures from the sitting's discussion are `measured 2026-11-12` (main
`f1c3256ab3`, `digests/discussion/S13.md`).

## Goal

Recon decides "is this the same flag?", "is this flag's subject suppressed?", "is this status word a
task status?" and "which findings does this run already hold?" in one place each, and keeps its
control-plane state in its ledger instead of in a vector store. Today each of those questions has two to
eight answers in different files, kept equal by hand and by "keep in sync" comments.

What a user observes when this lands:

- **An operator's suppression reaches composite flags.** `write_suppression_record(task_id=3105)` drops a
  later Stage-1 flag carrying `task_id='3105,4223'` (today it is kept; measured below). The suppression's
  ledger row records the partial match (`partial_hits`), so an over-suppression is visible where the
  operator wrote it (R13).
- **Stage 1 hands flags to Stage 2 through one channel, and recon keeps no control-plane marker in
  Mem0.** The dormant Mem0 relay (`flag_for_stage2`, 0 flags carried since 2026-10-08 against
  2,431 findings through the recon-report channel), its persistence and escalation markers, the dormant
  human-operator stall marker and the 679-line Mem0 pool sweeper are deleted; a recon-stage write carrying
  `flag_for_stage2: true` is refused with a structured error naming the recon-report channel.
- **Reconciliation writes no task status into memory.** Leo's rule (2026-11-12): nothing writes task
  status into memory or reads it from there; the task store is its one point of truth. Recon's two status
  writers stop: the stale-status sweep only retires stale status edges and no longer writes "As of <date>,
  task N has status S." (30 such writes since 2026-10-08, measured 2026-11-12), and the Stage-1 prompt no
  longer asks the agent for a replacement status fact (for a task's status it invalidates the stale edge
  and writes nothing). The sweep's detector stays as a cleanup backstop for status prose other writers add
  (about two thirds of today's status prose comes from outside recon), until the separate
  recall-from-the-task-store design session sets a rule for every writer.
- **The recon-report MCP tool refuses an off-vocabulary finding category** with a structured
  `invalid_category` error listing the nine values, as it already does for severity; the stage schema's
  enum, the stage prompts' vocabulary and every category comparison come from one `FindingCategory` enum.
  Every refusal is recorded with its context in an append-only `rejected_findings` table, a later accepted
  retry of the same finding stamps the category the agent chose, and a fortnightly recurring check files
  an escalation when the refusals show a good candidate for a new category or too many lost findings.
- **The recon-report server takes a stage's run id from its connection.** A recon-report call whose
  `run_id` argument differs from the stage's run-id header is refused with a structured error naming both
  values (one log line per refusal), and the stage prompts no longer ask the model to echo `run_id`.
- **The recon-report state keeps a run's findings in one owner.** Its six run maps become one
  `RunFindingIndex` per run, with one register/unregister path and an invariant check the tests execute
  after every scenario, including a fused-memory restart in the middle of a run (the second filing of a
  signature still answers `duplicate_finding`).
- **A new task status reaches every recon lexicon by itself.** `is_count_snapshot('40 pending / 12
  infra-hold')` returns True (today False).
- **Event journaling no longer blocks the fused-memory event loop**: every task-status transition's
  journal fsync runs on a writer thread.
- **The next agent opens small files**: `flag_dedup.py` (5,981 lines, measured at `a85cb1ad78`), `task_knowledge_sync.py` (5,467)
  and `task_filter.py` (1,882) become modules that each make sense alone, and tests reach public names
  only.

## Background

The run `review-all-dark_factory-20261008` found the recon flag pipeline's identity spread across
`flag_dedup`, `server/recon_report`, `task_knowledge_sync` and four guards; its status vocabulary copied
out of `shared.task_statuses`; its two Graphiti edge sweeps sharing a copied skeleton; `task_filter` and
`task_knowledge_sync` each holding unrelated units; control-plane marker pools living in Mem0 (the fold
inventories five; four are dormant); and
`ReconReportState` holding one fact in six hand-synchronised maps. Program rulings this PRD implements
without relitigating: **R13** (all of it), R22 (no invariant here), R23 (no weakened key here).

**How this PRD reads R13's "Rest of S13 as proposed".** R13 lists the synthesis remedy items. Six of the
twenty keys (fk-b1cda781c60e, fk-90f374fcce21, fk-fa1fd33c7a9e, fk-fbbf580e4e1d, fk-4c8d19c64d99,
fk-2868338d3cf3) are not named in that list; each is carried here under its own finding's proposal, which
"as proposed" covers. Two keys (fk-9b76dd8a1cdb, fk-cdb8c20a3d0f) carry a proposal — a typed Graphiti
task-state claim — whose premise does not hold on re-measurement; the sitting ruled their disposition
(OQ-1, OQ-8 below): the typed claim is not built, because it would keep a copy of task status inside
memory.

## Rulings (sitting of 2026-11-12)

Every open question of this stream was ruled at the wave-3 sitting: Leo's answers of 2026-11-12
(`digests/wave3-leo-answers-2026-11-12.md`), his residual answers of 2026-11-13
(`digests/wave3-leo-residual-answers-2026-11-13.md`), the rulings file `digests/wave3-rulings-applied.md`
§S13 and its cross-stream block, the measured "Effect on the plan if adopted" in
`digests/discussion/S13.md` (part of each ruling that adds to the recommendation), and the lead's rulings
`digests/wave3-lead-rulings.md` (later blocks win). The text of this PRD is authored under them.

| Id (sitting key) | Ruling | Ruled by |
|---|---|---|
| OQ-1 (S13-1) | As recommended, option (b), through residual R-1 = A on 2026-11-13. Leo's principle: nothing puts task status into the memory store or reads it from there; the task store is its one point of truth. The stale-status sweep stops writing its replacement sentence and only retires stale facts (η); the typed status record (option (a)) is not built, because it would keep a copy of status inside memory. The acceptance record's reason says the detector stays as a cleanup backstop for status prose from other writers until the recall design session sets the write and read rules, to be re-raised if that session lands a write rule for every writer. The recall-from-the-task-store design session is SPAWNED by the lead as its own fable /team session (investigate, discuss and design); it is not a task of this plan, this plan does not design it, and nothing here waits on it (xref only). Narrows R13's treatment of fk-9b76dd8a1cdb: the typed claim the finding proposed is rejected, not deferred. | Leo |
| OQ-2 (S13-2) | As recommended, option (a): `add_finding` refuses an off-list category with `invalid_category`, listing the nine values, and the stage prompts list the vocabulary (ε). PLUS Leo: every refusal is recorded with adequate context for later analysis, and a delayed recurring task analyses the recorded refusals and escalates when a good candidate appears. Applied per `discussion/S13.md` §S13-2: ε gains the append-only `rejected_findings` table owned by `server/recon_report_store.py` (180-day retention, pruned on write, untouched by run cleanup), the `retried_as` stamp and the per-stage `findings_rejected_invalid_category` count (C11); new task π writes `fused-memory/scripts/check_recon_category_candidates.py` (code decides by thresholds; judgement happens at the escalation); new task ρ is the seed link of a native recurrence chain (`task_kind='deterministic'`, predicate `before_done`, dated milestone ~30 days after filing, `recurrence {key: recon-category-candidates, interval_secs: 1209600}`). The discussion said "filed at decompose"; `submit_task` refuses a `before_done.script` that is not on main and executable (`middleware/deterministic_task_guard.py::_validate_script_under`, planning mode included), so the lead files ρ the day π lands, as the ruling-fold rules put a follow-up that can only be filed after something lands (fold-verifier; a lead action). ε's post-deploy findings-per-run signal is replaced by the script's lost-finding condition. | Leo |
| OQ-3 | Moot: OQ-7 was ruled (b), so no ledger-backed relay exists to write through (lead S13/OQ-3, conditional on OQ-7). | the lead |
| OQ-4 | (a), per move, as drafted: S19's ledger, journal, queue and self-model moves land first and this plan edits at the new paths; this plan's task-tree split lands first and S19 β2 then moves `recon_prose_grammar.py` in place of `task_filter.py`; S19's move of the flag-record contract and consolidation gate lands before κ3. S19's move never waits on κ3. | the lead |
| OQ-5 | (a): queueing becomes awaitable and returns after the journal row is committed (μ, C8, D15). | the lead |
| OQ-6 | (a): ι, the verbatim split of `stages/task_knowledge_sync.py`, lands first and takes no S12 edge; S12 γ0, γ1, ε and κ2 wait on it; θ3 and ε wait on S12 ι2. Same ruling as S12/OQ-8. | the lead |
| OQ-7 (S13-3) | As recommended, option (b): the dormant Mem0 relay, FIX D's persistence and escalation markers and the human-operator stall marker with its escalation are deleted (κ1, κ2), and recon-stage relay writes are refused (κ3). Amends R13's "marker pools move from Mem0 to the recon ledger": the dormant pools are deleted rather than moved, which meets R13's intent (no control-plane state in the vector store) with less code. | Leo |
| OQ-8 (S13-4) | As recommended, option (b′), through residual R-1 = A, discussed with OQ-1 at Leo's request: η also rewrites the Stage-1 prompt's "Resulting-state-only temporal_facts" and "update_edge Temporal Limitation" sections, so for a task's status the agent invalidates the stale edge and writes nothing; other temporal facts keep the two-step. The acceptance record's reason is the one under OQ-1. | Leo |
| S12 OQ-10 (lead) with S12-2's cross-stream rule | ξ is unconditional: S12 δ puts the stage-identity and run-id headers on the recon-report connection, and ξ makes the recon-report server read them. Under Leo's S12-2 (option B, confirmed 2026-11-13) and the lead's run-id clause, ξ REFUSES a `run_id` argument that differs from `CAUSATION_ID_HEADER` with a structured error naming both values and one log line per refusal, and drops the prompts' "echo run_id" line in the same change. Supersedes the lead's 2026-11-11 alignment line ("header wins, one override event"); no `recon_report_identity_override` exists. | the lead (refusal: Leo's S12-2) |

Cross-stream rules from the same sitting that bind this plan:

- **`one-decision-partition-per-area` (INV-23)** (Leo, under S17-1; recorded as a memory and entering
  `docs/legibility/design-invariants.md` through a task of the S17 plan as INV-23, the last of the six
  wave-3 invariant numbers INV-18 to INV-23 taken in landing order; every check anchors on the slug, never
  the number). D17 tabulates every decision this plan's tables and sweeps make and who
  makes it. Code decides everything facts settle; the only LLM judgement is the category-candidate
  escalation (ρ's filing), which reaches a human.
- **Evals are hermetic** (Leo, under S1-2). ε adds a table to `recon_report_state.db`; no eval cell
  constructs a `ReconReportStore` (the store is built only by `server/main.py`'s recon-report wiring), and
  ρ's script reads the production store only from the orchestrator's deterministic runner, never from an
  eval. D17 records that the rule is satisfied and needs no further task here.
- **Edges that rest only on a shared test file are not wired** (lead, after the wave-3 fold-verifiers,
  restating R54; applies inside and across plans). The decomposition plan below wires no such edge.
- **Dependency cycles are refused by the task store** (task 9490, high, outside every plan). The decompose
  seat's wiring may keep a cycle pre-pass as an early check, not as the guard.

### Freshness (re-measured at `1f240e6da2`)

Commits since the run that touched these keys, and what is left:

- **Stale-status sweep (task 6375, `fe83201cd7`, `1ae0e8e1f8`).** The sweep was split into a 639-line
  orchestration module and a 651-line `status_snapshot_extraction.py`; its prose moved to
  `docs/stale-status-snapshot-sweep.md` (1,225 lines). Prose share fell from 76% to 42%. The grammar
  itself is unchanged: `status_snapshot_extraction.py::_build_snapshot_patterns` still compiles 5 regexes
  per family, two families at import (10 compiled), 11 `re.compile` sites in the module; the sweep still
  writes its replacement as prose (`stale_status_snapshot_edge_sweep.py::build_supersede_fact`,
  `'As of {date}, task {task_id} has status {status}.'`, into `add_memory(category='temporal_facts')`).
- **Per-task status sweep (task 4851, `547987780a`).** The status sweep's two entry points share one core
  (`_sweep_status_snapshot_edges` → `_sweep_under_lock` → `_read_corpus`). The priority-override sweep
  still holds its own copy of the skeleton: 32 distinct code lines shared with the status core (rule:
  docstrings and comments stripped, whitespace collapsed, set intersection; 44 at the run). It still
  imports `_SWEEP_AGENT_ID` and `flatten_dedup_edges` from the status module.
- **FlagTypeFamily adoption (task 6005).** Six `FlagTypeFamily(` instances now (three at the run); the
  three hand-rolled drift logs are gone (0 now, 3 at the run). Five matchers still test `flag_type`
  another way and log no drift (§Findings disposition, fk-65aeb80b161d).
- **Marker identity (task 5614).** `stage1_flag_marker` ledger rows are keyed by the new
  `flag_dedup.py::compute_marker_signature`; the Mem0 mirror was retired earlier (task 2406). The
  surface fk-e30ecd1f36b8 describes grew by one signature function.
- **Stage-1 chain (task 6120, `c7579e6d1c`).** The Stage-1 flag chain moved from `MemoryConsolidator.run`
  to `reconciliation/stage1_flag_chain.py`, with two ordered `ResolvingFilter` tuples as the one statement
  of order. The hotspot proposal's `FlagFilter` protocol is therefore already realised; what remains of
  fk-46514b4ef0cc is where the filters live.
- **Stage-2 accounting (tasks 4873, `0de5e07dde`, `110215e038`).** The tasks_created accounting cluster
  left `task_knowledge_sync.py` for `stage2_counter_grounding.py`. The other six clusters remain.
- **recon_report (tasks 6025, `0c54379ab6`; `c705050475`).** `_state` became a read-only view over
  `_resident_entries`, written only by `_install_entry`; a seventh run-scoped map, `_run_entries`, was
  added. The four dedup maps, two per-entry mirrors and two persisted slices are still kept in step by
  hand.
- **Severity (`66b665933e`).** `ReconReportState.add_finding` now refuses an off-vocabulary severity with
  `invalid_severity`; the category got no such treatment. It is the precedent C5 follows.

Untouched since the run: `scope_freshness.py`, `flag_task_ids.py`, `policies/`, `event_journal.py`'s
append path, `stale_priority_override_edge_sweep.py`.

Re-checked for the ruling fold at `a85cb1ad78` (2026-11-13; script `scratch/S13-rulingfold/measure.py`):

- **Task 6621 (`de3185b357`, `9cb7f23670`)** moved `TERMINAL_STATUSES`' import in `task_knowledge_sync.py`
  onto `shared.task_statuses.TERMINAL` (an alias, not a declaration: the three recon terminal-set
  declarations δ2 deletes are unchanged, 3 matching lines) and rewrote `flag_dedup.py::_is_valid_marker_task_id`'s
  docstring so it no longer claims a layering reason. The four "local copy / keep in sync" comments in
  `flag_dedup.py` are unchanged (the identity one at `_normalize_content_description` still cites the
  inversion), and `flag_dedup.py` is 5,981 lines.
- **Task 6598 is done** (`a16b38eec6`): `citation_verifier.py`'s repoint-stat docstrings agree with the
  code. η's edge onto it is satisfied; `REPOINT_AGENT_ID = 'recon-stage-memory_consolidator'` and the
  `INACTIVE_TASK_STATUSES` import, which η and δ2 change, are unchanged.
- **Task 6553** denied one more tool to every recon stage in `cli_stage_runner.py`; `FINDING_ITEM_SCHEMA`'s
  hand-written 9-value enum, which ε derives, is unchanged.
- **Task 6234** (`targeted.py`) left the two inline `('done', 'cancelled')` tuples δ2 deletes, the
  `retire_flag_markers_for_terminal_task` call κ2 deletes and `_SNAPSHOT_CONTRADICTING_TRANSITIONS` as they
  were.
- **Tasks 9276 and 6236** rebuilt the interceptor's done-provenance probes and moved them to
  `middleware/done_provenance.py`; `TaskInterceptor._journal` still calls `self.event_queue.enqueue(event)`
  synchronously (the one production call site μ awaits), and the replay path's inline
  `return self.enqueue(event)` (μ's check) is unchanged.
- **Tests**: the reach ν removes is unchanged (42 private names, 139 imports, 57 string patch targets in
  2 files, attribute reach 9 + 4 + 3); a fourth module, `tests/reconciliation/test_stage2_payload_section_parity.py`,
  imports the module as an alias but reads only `__file__`, which the guard does not count.
- **Prompts**: `run_id=<…>` call examples are 5 in `prompts/stage1.py`, 5 in `stage2.py`, 6 in `stage3.py`,
  2 in `prompts/__init__.py` and 1 in `policies/autopilot_video.py` (ξ's count; unchanged since
  `c9ab19766a`); `stages/base.py`'s `recon_context` paragraph still tells every stage to pass `agent_id`
  and `_causation_id` (S12 δ rewrites those lines; ξ the `run_id` one).
- **Prose shares** are unchanged: `flag_dedup.py` 58.4%, `task_filter.py` 59.6%, `server/recon_report.py`
  53.2%, `task_knowledge_sync.py` 49.0%.
- **Recon-report store**: `server/recon_report_store.py` still holds one table, `recon_report_state`, whose
  rows `ReconReportStore.delete_run` removes when a run goes quiet; no refusal is recorded anywhere today
  (the severity refusal `server/recon_report.py::_invalid_severity_error` only logs). No task in the store
  carries `metadata.recurrence` (all statuses), so ρ is the project's first live recurring chain.

## Findings disposition

| Key | Sev | Disposition | Note (measured at `1f240e6da2`; re-checked at `a85cb1ad78`) |
|---|---|---|---|
| `fk-51ec1e4010af` status lexicon copies | medium | leaf δ2 | 2 hand-written status-word alternations (`task_filter.py::COUNT_SNAPSHOT_RE`, which holds it twice, and `flag_dedup.py::_COUNT_GROUP_RE`), neither containing `infra-hold`; 3 terminal-set literals (`flag_dedup.py::TERMINAL_STATUSES`, `task_filter.py::INACTIVE_TASK_STATUSES`, `orphaned_recon_escalation_sweep.py::TERMINAL_TASK_STATUSES`) plus two inline `('done','cancelled')` tuples in `targeted.py::TargetedReconciler`; `task_filter.py::_STATUS_PRIORITY` lacks `cancelled` and `infra-hold` and is privately imported by `task_knowledge_sync.py::_select_proactive_sample`. Probe: `is_count_snapshot('40 pending / 12 infra-hold')` → False. No test covers any of these copies. |
| `fk-917fdf32a542` copied sweep skeleton | medium | leaf η | partly fixed by `547987780a` (the status sweep's two entry points share one core); 32 distinct lines still shared with the priority sweep; `_SWEEP_AGENT_ID` private reach and the literal `'recon-stage-memory_consolidator'` in two modules (`stale_status_snapshot_edge_sweep.py::_SWEEP_AGENT_ID`, `citation_verifier.py::REPOINT_AGENT_ID`) against five derived sites; two parallel stat-key pairs. |
| `fk-5d00812d5256` composite task_id | medium | leaf β (parser in α) | 8 independent comma-splitters of a finding's `task_id` (`flag_dedup.py::_decompose_suppression_task_id`, `::_canonicalize_suppression_task_id`, `::_task_union_signature`, `::_is_valid_marker_task_id`, `flag_task_ids.py::task_id_components`, `server/recon_report.py::_split_task_id_parts`, `finding_task_escalation.py::_sole_task_id_part`, `gate_owned_finding_phrasing.py::_flag_cited_task_ids`) plus the verbatim compare in `filter_suppressed._keep`. Probe (stub ledger): a row `3105` keeps flags `'3105,4223'`, `'3105, 4223'`, `'4223,3105'`; a row `'3105,4223'` keeps the identical flag `'3105,4223'`. |
| `fk-d9bfbf342b5b` task_filter two modules | medium | leaf δ1 | grammar half 774 lines (32–805), tree half 1,077 lines (806–1882); 11 production importers use only the grammar, 11 only the tree, 3 both; module docstring names only the tree. |
| `fk-65aeb80b161d` flag_type mechanisms | medium | leaf γ | partly fixed by task 6005; still outside `FlagTypeFamily`: `ABSENCE_FLAG_TYPES` (exact; its filter gates `delete_memory`), `PHANTOM_TASK_CREATION_FLAG_TYPES` (exact), `_is_systemic_pattern_candidate` (`== 'systemic_pattern'`), `_flag_type_in_grounds_family` (substring stems from `standing_decision_constants.py::GROUNDS_TOKEN_FAMILIES`), `preservation_specimen_guard.py::STRANDED_FLAG_TOKEN_FAMILY` (substring), `scope_freshness.py::is_cross_project_scope_correction` (`== 'cross_project'`). None logs drift. Task 7388 is cancelled into γ. |
| `fk-e64ee986eeed` private fingerprint reach | low | leaf α | `scope_freshness.py` imports `_content_fingerprint` and persists it as `subject_description_fingerprint` (compared next cycle in `snapshot_is_fresh`); `task_knowledge_sync.py` imports `_content_fingerprint` and `_normalize_content_description`; the `_CONTENT_FP_*` comments still name only the marker consumer. |
| `fk-9f66b6f4420c` 5-tuple render seam | low | leaf δ2 | `_select_visible_active_with_body` still returns `tuple[list, str|None, str, str, str]` (4 returns, 2 unpack sites); 4 string patch targets into `task_filter` (`_render_task_line` ×3, `_select_visible_active_with_body` ×1). |
| `fk-b1cda781c60e` task_knowledge_sync bundles | medium | leaf ι | partly fixed by `0de5e07dde` (accounting cluster moved). Still in the 5,467-line file: Stage 3 `IntegrityCheck` (217 lines) and its shared formatters (~145), marker GC (~1,570), task_count_snapshot I/O (~490), growth sweep + streak (~355), Stage-2 flag relay (~785 + 56), the reify briefing hook (~250). Production importers: `harness.py` (3 names), `targeted.py` (1). Two function-local imports break no cycle. Tasks 6846 and 7807 are cancelled into ι and κ1. |
| `fk-9165ae3a1e72` control-plane state in Mem0 | high | leaves κ1, κ2, κ3 (OQ-7 ruled (b)) | partly fixed (5614, 2406): `stage1_flag_marker`, `stage1_flag_suppression` and `cycle_summary` are ledger-authoritative. Still Mem0-only, five pools: `stage2_persistence_marker`, `flag_for_stage2` (LLM-written; read by a `limit=100` semantic search), `stage2_escalation_marker` (never deleted by anything), `reconciliation/stage1_stall_detector.py::_STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE` (written and counted through Mem0; its docstring says the markers "accumulate indefinitely") and the `stage2_suppress` guard memories. **Traffic, live store:** since 2026-10-08, 2,083 completed runs across 10 projects carried a Stage-2 report (of 13,289 completed runs); `stage1_mem0_flags_processed` (the Mem0 relay read) sums to 0 in every project, `stage1_analytical_findings_processed` (the recon-report channel, `stages/task_knowledge_sync.py::_query_recon_report_findings`) to 2,431; `stale_flags_escalated` 0; `stage2_flag_markers_acknowledged` 0; since 2026-09-01 the relay carried 13 flags against 4,737 findings (critic). The critic's write-journal read finds no `recon-stage-*` `add_memory` carrying `flag_for_stage2` since 2026-10-08. Mem0 now: `{'flag_for_stage2': true}` 1 row (dark_factory), 7 (reify); persistence, escalation and human-operator stall markers 0; legacy `{'source': 'stage1_flag_marker'}` 0 in dark_factory and reify. FIX D persistence tracking and the escalation markers act only on Mem0-relayed flags (`TaskKnowledgeSync.assemble_payload` feeds `_track_flag_persistence` from `_query_stage2_flags`' partition), so the unbounded escalation pool is latent. The human-operator stall alarm with its marker (an L1 escalation for a task flagged as needing a human operator for 5 cycles) cannot fire, because Stage 1's output schema has no field for that status and no Stage-1 report since 2026-06-01 carried one: `MemoryConsolidator._escalate_human_operator_stalls` acts only on a flag with `resolution_status == 'human_operator_required'`, `resolution_status` is not a property of `cli_stage_runner.FINDING_ITEM_SCHEMA`, no Stage-1 prompt asks for it, and `stage1_human_operator_stalled` (written whenever such a flag exists, stall or not) is absent from all 32,619 completed runs since 2026-06-01 (8,337 with a Stage-1 report), while `stage1_gate_backlog_escalated`, under the same escalation-queue guard, appears in 4,525 (critic, delta pass). Ledger: 0 `stage2_persistence_marker` or `flag_for_stage2` rows. **Disposition** under OQ-7 (b), ruled by Leo: the relay, FIX D's markers and the stall marker with the dormant escalation that counts it are deleted (κ1, κ2) and recon-stage relay writes are refused (κ3). `stage2_suppress` stays in Mem0: the guard is the completion-note memory itself (knowledge an LLM or human searches, not a marker), and its read is a deterministic Qdrant payload count, not a semantic search (3,730 rows in dark_factory; `memory_metadata.py` census 1,588). Residual of this key after κ1–κ3: none; `stage1_flag_suppression` and `cycle_summary` Mem0 mirrors are narrative beside an authoritative ledger (D11). |
| `fk-90f374fcce21` tests reach TKS privates | high | leaf ν (ι, κ1, κ2 contribute) | 42 private names imported 139 times across 8 test files (rule: AST `ImportFrom` of an underscore name from the module); 57 string dotted-path patch targets (10 names) in 2 files, `test_stages.py` 47 and `reconciliation/test_task_count_snapshot_cadence.py` 10 (rule: AST `str` constants starting `fused_memory.reconciliation.stages.task_knowledge_sync._`), re-measured at `97be1d71fa`; plus attribute reach on a module alias (`tks._x`, `patch.object(tks, '_x')`, `hasattr(mod, '_x')`): 11 in `test_stages.py`, 4 in `test_recon_self_model.py`, 3 in `reconciliation/test_standing_decision_growth_sweep.py`. Top patch targets: `_sweep_stale_persistence_markers` 10, `_sweep_stale_mem0_flag_markers` 10, `_gc_recon_markers` 8, `_verify_task_count_snapshot_written` 7. Task 7065 is cancelled into ν. |
| `fk-fa1fd33c7a9e` Mem0 pool sweeper contract | medium | leaf κ2 (deleted; OQ-7 ruled (b)) | `task_knowledge_sync.py::_sweep_stale_mem0_pool` grew to 679 lines (226-line docstring), 14 parameters, returns a bare `int`; 11 paths return a count indistinguishable from "nothing stale". Its three callers swept 28 residue rows since 2026-10-08 (`stale_persistence_markers_gc_swept` 9, `stale_mem0_flag_for_stage2_markers_gc_swept` 19, `stale_mem0_flag_markers_gc_swept` 0) and carried no relay traffic (row above). κ2 deletes it with the pools it sweeps. |
| `fk-9b76dd8a1cdb` status mirrored as Graphiti prose | high | accepted (reason; OQ-1 and OQ-8 ruled by Leo) | the grammar is unchanged (above). The proposed typed claim is rejected under Leo's rule that nothing writes task status into memory or reads it from there (it would be a typed copy of status inside memory), and it would not let the grammar be deleted anyway (G6). Measured 2026-11-12 (`discussion/S13.md` §S13-1): of 46,627 `add_memory` writes since 2026-10-08, 115 match the status-phrase rule; recon Stage 1 made 37 of them (30 are the sweep's sentence), every other writer about two thirds; none of those writers is gated, and graphiti-core can extract a status edge from text that states none. Recon stops both its status writers (η: the sweep's replacement sentence and the Stage-1 prompt's status facts); the detector stays as the cleanup backstop for other writers' status prose until the separate recall-from-the-task-store design session sets a rule for every writer. The acceptance record is filed at decompose. |
| `fk-cdb8c20a3d0f` status projected into Graphiti | medium | leaf η (OQ-1 ruled (b)) | η deletes the sweep's prose replacement write (`build_supersede_fact`, `_SUPERSEDE_FACT_TEMPLATE`, `_MAX_SUPERSEDE_WRITES_PER_CYCLE`) and rewrites the Stage-1 prompt's status sections (OQ-8 (b′)): the sweep invalidates, and the task store answers status. |
| `fk-2868338d3cf3` prose dominates | medium | leaf θ3 (every moving leaf contributes) | partly fixed by `1ae0e8e1f8` (sweep 76% → 42%). Still: `flag_dedup.py` 58.4% (344-line module docstring), `task_filter.py` 59.6%, `server/recon_report.py` 53.2%, `task_knowledge_sync.py` 49.0% (rule: whole-line COMMENT tokens plus AST docstring spans over physical lines). `harness.py` (47.6%) is S12's file and is out of this plan. |
| `fk-46514b4ef0cc` flag_dedup purpose | medium | leaves θ1, θ2, θ3 | 14 public `filter_*` functions beside the ledger lifecycle; `dedup_flags` 536 lines. Clean clusters measured by intra-module name graph; dedup and systemic-pattern discovery share the tracked-task matcher, which moves to a leaf first. Task 5996 is cancelled into θ1–θ3. |
| `fk-fbbf580e4e1d` per-project axis | low | leaf λ | unchanged since the run: 4 per-project modules folded into `SNAPSHOT_WRITE_BLOCKED_PROJECTS` / `CONTAMINATION_CEILING_RETIRED_PROJECTS`; a third mechanism, `task_knowledge_sync.py::_BRIEFING_REFRESH_PROJECT_ALLOWLIST`; `prompts/stage2.py` compares a hard-coded project id. Config already keys three per-project values by `project_id`. |
| `fk-4c8d19c64d99` journal fsync on the loop | medium | leaf μ | `event_queue.py::EventQueue.enqueue` is still sync and calls `EventJournal.append` (INSERT + commit, WAL, `synchronous=FULL`) inline on the loop thread, from `TaskInterceptor._journal` on every task-status transition; the docstrings still claim "never awaits SQLite" and "~1-5 ms"; no measurement exists. The loop-blocking gate's primitive table does not see sqlite commits, so no ledger row or task owns this. |
| `fk-e30ecd1f36b8` identity has no home | high | leaf α | 8 identity functions over 3 modules plus `task_knowledge_sync.py::_flag_content_fingerprint`; the two normalizers are byte-equal bodies (`' '.join(description.split()).casefold()`) kept equal by comment; 4 "local copy / keep in sync" comments, all in `flag_dedup.py` (rule: case-insensitive `keep in sync|local copy` over `flag_dedup.py`, `server/recon_report.py`, `task_knowledge_sync.py`, `scope_freshness.py`, re-measured at `a85cb1ad78`, unchanged; one justifies the identity copy, one the status lexicon copy, two the task-state extraction copy; task 6621 removed a fifth layering claim from `_is_valid_marker_task_id`'s docstring); the "server/ must not import from reconciliation" rule they cite is broken at module scope by `server/recon_report.py` (`citation_repair`, since `df0e8d1fac`). |
| `fk-0791e9817b84` six run maps | high | leaf ζ | writers per map: `_run_sig_index` 5 methods, `_run_finding_index` 4, `_run_desc_index` 4, `_run_cited_task_index` 4, `_active` 3, plus `_run_entries` and `_resident_entries` behind `_install_entry`; class 2,721 lines (`add_finding` 393, `cite_task` 353); tests reach `_run_sig_index` 15×, `_run_finding_index` 10×, `_run_desc_index` 9×, `_run_cited_task_index` 9×. |
| `fk-7c13df0b988c` category free string | medium | leaf ε (refusal and its record, OQ-2 ruled (a) plus Leo's additions); π and ρ carry the recurring analysis | `cli_stage_runner.py::FINDING_ITEM_SCHEMA` holds the 9-value enum by hand; `add_finding` only truncates the category; 3 matcher rules (prefix default, exact module constant, exact literal in `scope_freshness`). Live store (`data/reconciliation/recon_report_state.db`, read-only, live runs): 714 findings, 216 distinct categories, 207 of them (345 findings) off the enum; 13 findings carry a `cross_project_*` category the prefix rule silently makes non-actionable. Measured 2026-11-12: 345 off-list findings carry 207 distinct labels; the top three are `duplicate_procedural_knowledge` (24), `mem0_duplicate_cluster_consolidated` (23) and `memory_consolidation` (13). No refusal is recorded anywhere today. |

## Pre-conditions for activating

- **Substrate on main** (G3, verified at `1f240e6da2`): `shared/src/shared/task_statuses.py::TaskStatus`
  and `::TERMINAL`; `fused-memory/src/fused_memory/models/reconciliation.py::StageId` and
  `::FindingSeverity`; `reconciliation/recon_ledger.py::ReconLedgerStore` (`list_by_kind`,
  `upsert_if_state`, `MARKER_KINDS`, `gc`); `backends/graphiti_client.py::read_all_valid_edges_checked`;
  `reconciliation/flag_record_contract.py::recon_stage_flag_kind_refusal`;
  `reconciliation/dead_letter_writer.py::DeadLetterWriter` (the writer-thread precedent);
  `config/schema.py::ReconciliationConfig`.
- **Upstream leaves** (dependencies, not holds):
  - S14 μ, task **6721** (`shared/agent_identity.py::recon_stage_agent_id`) — η.
  - S26 μ, task **9203** (C7, `MemoryService.enumerate_valid_edges`, the sweeps' checked read surface; it
    also repoints the growth sweep's `memory_service.graphiti` reach in `task_knowledge_sync.py`) — η, and
    ι (lead ruling OQ-6).
  - S12 γ2 (S12/C2 `StageCollaborators`: stages constructed with their collaborators, R12) — ν. S12 is
    authored in this wave (fold-verified draft
    `plans/review-all-s12-reconciliation-run-lifecycle-and-stage-interface-prd.md`); ids are wired at
    decompose. The draft's ν → S12 θ is dropped under the lead's test-file rule (they share only test
    files).
  - S12 δ (S12/C3: the stage-identity and run-id headers on the recon-report connection, and C3's
    refusal rule `stage_identity_refusal`) — ξ (lead ruling S12/OQ-10 (a); ξ is unconditional).
  - S12 ι2 (S12's prompt-library move, which creates `prompts/fragments.py` (S12/C7) and is S12's last leaf on
    `stages/task_knowledge_sync.py` and `stages/memory_consolidator.py`) — θ3 and ε (lead ruling OQ-6).
  - S19 α, β1 and β2 (draft `plans/review-all-s19-package-layering-and-private-seams-prd.md`; they move
    `recon_ledger.py`, `event_journal.py`, `event_queue.py`, `recon_self_model.py`,
    `flag_record_contract.py` and `consolidation_gate.py` into `fused_memory.recon_state`) — β, μ (α); κ1
    (β1); κ3 (β2) (lead ruling OQ-4).
- **Existing tasks that land first in today's files**: 7941 and 5272 (both sweeps), 6598
  (`citation_verifier.py`; done 2026-11-13, `a16b38eec6`, so that edge is satisfied), 6718 (S14 θ, the scheduler-override store, which edits
  `stale_priority_override_edge_sweep.py` and its test) — η depends on each.
- **What the second half waits on** (forensic closure over the live `dependencies` table, re-measured
  2026-11-13 at `a85cb1ad78`; script `scratch/S13-rulingfold/` reuses S12's closure reader): S12's run
  chain starts with S12 α → 7876, 9254. Task **4867**, which the draft named as the long wait, is done
  (`daff96ef3c`, 2026-11-11). The union of the ancestors of every out-of-batch task an S13 leaf or its S12
  upstream waits on (9203, 6721, 6718, 7941, 5272, 7876, 9254, 6695, 9278, 9280, 9180, 9282, 7877, 8144) is
  99 tasks, none of them 8528, 7720 or 5040; 9254 has 33 live ancestors and 9282 has 53. κ1 and κ3 also
  wait on S19 β1 and β2, so on β2's own external dependencies (6599, 9240, 9204 and 9147 through S19 γ,
  6622): re-measured at `4a87282ee7`, the union with the list above is 109 tasks (62 live), still none of
  8528, 7720, 5040, 8526 or 8527. So every S13 leaf
  that follows an S12 leaf — θ3, ε, κ1, κ2, κ3, η, ν, λ, ξ, π, ρ, ω — waits on S12's chain and on those
  wave-2 tasks, including the high leaves κ1, κ2, κ3 and ν, and no S13 leaf reaches the merge-lane policy
  chain (8528, 7720, 8526, 8527; 8525 is done). 9285 (150 ancestors, 8528 among them) depends on δ2, never
  the reverse. S12 α also waits on task 7876 (pending, no ancestors, waiting on a live-store observation),
  which the lead watches. The first half (α, ζ, δ1, ι, β, γ, θ1, θ2, δ2, μ) waits on nothing beyond 9203
  (19 ancestors, 13 live, no long-chain sentinel) and S19 α.
- **File paths.** Every path below is today's. Where the lead's OQ-4 ruling puts an S19 move first, the clerk repaths the
  leaf's `metadata.files` from S19's §Moves table when the move lands (S19 D8); the delivered checks that
  name a moving module are written path-agnostic (`fused-memory/src/` plus a symbol pattern) so they hold
  under either order.

## Sketch of approach

1. **One identity module (α, β, γ).** `reconciliation/flag_identity.py` holds every function that decides
   "same flag": the description normalizer and its two digests, the content fingerprint, the three
   signatures, the parsed `FlagSubject`, the marker-key validator, `canonical_flag_type_family`,
   `FlagTypeFamily` and UUID extraction. Its module docstring is the persisted-format contract (C1 table):
   which outputs are stored, by whom, and what a change costs. `server/recon_report`, `scope_freshness`,
   `task_knowledge_sync` and the guards import it; their copies and their false layering comments go.
   β makes suppression match on any component (R13) and records partial matches on the ledger row. γ puts
   every flag_type vocabulary on `FlagTypeFamily` (with an exact mode for drop gates in front of an
   irreversible write) and adds a guard.
2. **flag_dedup keeps one job (θ1–θ3).** The fourteen filters move, whole, into
   `reconciliation/flag_filters/` modules grouped by what they decide; the shared task-lookup and
   tracked-task-term helpers move into leaves first, so nothing reaches back. `flag_dedup.py` keeps the
   marker and suppression ledger lifecycle. `stage1_flag_chain.py` stays the one statement of order.
3. **task_filter splits; the status vocabulary derives (δ1, δ2).** The prose grammar moves verbatim to
   `reconciliation/recon_prose_grammar.py` (S19 β2 then moves it; OQ-4), which builds the status-word alternation once from
   `TaskStatus` (C3). Every recon terminal set becomes `shared.task_statuses.TERMINAL`. The active-tree
   renderer returns a frozen value and takes its line renderer as a parameter.
4. **One closed category, its refusals recorded (ε, π, ρ).** `FindingCategory` beside `FindingSeverity`;
   parsed at `add_finding`, which refuses an off-list category and records the refusal with its context
   in a `rejected_findings` table (C11); every comparison and the stage schema read it; the prompts render
   it. π writes the check script that reads the refusals, and ρ seeds a fortnightly recurring chain that
   runs it and escalates a good candidate to a human.
5. **One run index (ζ).** `ReconReportState` keeps a `RunFindingIndex` per run that owns the run's
   entries, active stage and four dedup maps; every mutator calls it; persisted slices derive from it.
6. **One edge-sweep primitive (η).** Read through S26's checked read surface, record completeness before
   the policy can raise, invalidate in one loop, report typed stats; the two sweeps supply only their
   selection rule; the agent id comes from `StageId`; no replacement status prose is written, and the
   Stage-1 prompt asks for none (OQ-1, OQ-8).
7. **task_knowledge_sync splits and its Mem0 control plane goes (ι, κ1, κ2, κ3, ν).** ι moves five
   clusters verbatim, first on the file (OQ-6). Under OQ-7 (b) the dormant Mem0 machinery is
   deleted rather than migrated: κ1 moves the marker GC into its own module and deletes FIX D's
   persistence and escalation markers and the dormant human-operator stall marker; κ2 deletes the
   Stage-1→Stage-2 Mem0 relay and the Mem0 pool sweeper; κ3 refuses recon-stage relay writes and retires
   the relay from guidance, vocabulary and operator scripts. ν moves the tests onto public seams, using
   S12's collaborator injection.
8. **Smaller items.** λ moves the per-project policy flags into config. μ moves the event journal's
   writes onto one writer thread (OQ-5). ξ makes the recon-report server take the run id from the
   stage connection and refuse a differing argument (C10). ω runs the two-way boundary suite and the
   live-run check.

## Resolved design decisions

- **D1 — Persisted formats are frozen byte-for-byte.** Every identity output a store holds keeps its exact
  bytes (C1 table): marker signatures keep their lexicographic union order, the recon-report signature
  keeps its numeric-aware order, the content fingerprint keeps `fp:` + 32 hex. Unifying the two orders
  would orphan ledger marker rows (recurrence history) and re-key persisted recon-report state; neither
  buys anything once the parse is shared. The value of the leaf is one parse and one documented list of
  persisted forms, pinned by a golden fixture computed from today's code.
- **D2 — `FlagSubject` keeps every component (R13).** The composite is parsed once
  (`parse_flag_subject`, built on today's `task_id_components` rules: str or int, split on `,`, strip,
  drop blanks, first-position dedup, total over malformed input). Suppression matches when the flag's
  components and the row's components intersect, for a flag_type in the row's family (or a wildcard row).
  A partial match (some flag component not covered by any matching row) is still dropped, and is
  recorded on each matching row's payload as a bounded `partial_hits` list (newest 10: run id, the
  flag's raw subject, the uncovered components) plus a `partial_hit_count`, written with
  `upsert_if_state(…, existing_state='active')` so a lifted row is never revived. `filter_suppressed`
  gains a keyword `run_id: str`, which every caller passes (today only `dedup_flags`, measured at
  `97be1d71fa`). A failed `partial_hits` write is logged and never un-drops the flag: the drop stands. The
  payload read-modify-write can lose one hit to a concurrent run's write on the same row; that is accepted,
  because the list is a bounded diagnostic and `partial_hit_count` is not an accounting figure. The
  Stage-1 report gains one stat for partial drops. `ReconLedgerStore.is_suppressed` (no production caller;
  8 test calls in 3 files at `97be1d71fa`, repointed or deleted in β) is deleted rather than given a
  second matching rule.
- **D3 — FlagTypeFamily matching modes.** `FlagTypeFamily` gains `match: FamilyMatch` — `CANONICAL`
  (today's behaviour, the default), `EXACT` (spelling set only; used where a drop gates an irreversible
  write: `filter_false_absence_flags` gates `delete_memory`, the caveat task 7388 recorded), and `STEMS`
  (case-insensitive substring stems, for the grounds families and the stranded family). Every mode logs
  drift through `log_drift`. The ledger-side comparison of a flag against a suppression row's flag_type
  keeps `canonical_flag_type_family` directly: rows are data, not a vocabulary. `scope_freshness`'s
  category half is ε's. R13 is read as: every vocabulary is a `FlagTypeFamily` tested through `matches()`
  and `log_drift`; the exact and stem semantics survive only as declared modes (EXACT for the
  `delete_memory` gate, per task 7388).
- **D4 — The flag_dedup partition.** Modules under `reconciliation/flag_filters/` (package `__init__`
  holds no re-exports): `task_state_refutations.py` (terminal metadata, stale bulk get_statuses, false
  absence, false phantom creation — each drops a flag a live task read refutes), `systemic_pattern.py`,
  `count_snapshot.py`, `authorship.py`, `task_text_accounting.py` (cluster growth and recorded caveat,
  which share a core), `entity_standing_decision.py` (Hook A), `stage3_findings.py` (blocked snapshot,
  contamination ceiling, snapshot stat edge). Leaves first: `reconciliation/tracked_task_terms.py`
  (`_TrackedTask`, `_extract_tracked_task_terms`, `_match_extracted_tracked_tasks`, `_asserts_never_tracked`,
  `_significant_terms`, shared by dedup's discovery and the systemic filter) and the existing
  `reconciliation/task_corroboration.py` (gains `_safe_get_task_or_none` and `_titles_corroborate`).
  `flag_dedup.py` keeps `dedup_flags`, `filter_suppressed`, `write_suppression_record`,
  `build_suppression_payload`, `acknowledge_*` and the cross-project fix-task discovery `dedup_flags`
  calls. Names a sibling now imports become public; no alias shims (heuristic 14's no-cheating rule).
- **D5 — The task_filter split.** `recon_prose_grammar.py` takes the 32 grammar symbols verbatim and
  imports nothing from `task_filter`; `task_filter.py` keeps `FilteredTaskTree`, the census diagnostics and
  the renderer, and imports nothing from the grammar. `TestClauseSplitRe`'s consumer allowlist moves with
  `WIDE_CLAUSE_BOUNDARY_RE`. The ~120-line clause-boundary rationale moves to
  `docs/recon-prose-grammar.md` with a one-line pointer. Both modules are born in `reconciliation/`. S19 β2
  moves `recon_prose_grammar.py` into `fused_memory.recon_state` **in place of** `task_filter.py`, and
  `task_filter.py` (the tree renderer) stays in `reconciliation/` (lead ruling OQ-4): S19/C2 puts a module in
  `recon_state` when a unit below `reconciliation` imports it, and after δ1 and δ2 the only such importers
  (`middleware/dependency_direction_check.py`, `services/completion_claim_gate.py`) import grammar names
  only, `citation_verifier.py` and `stale_status_snapshot_edge_sweep.py` lose their
  `INACTIVE_TASK_STATUSES` import in δ2, and `server` sits above `reconciliation` in S19/C3's `LAYERS`.
- **D6 — The status lexicon (C3).** The alternation is generated from `TaskStatus` values with one
  spelling rule (`-` → `[-_ ]?`), plus two named non-status words beside the builder (`total`, and the
  `canceled` spelling); `COUNT_SNAPSHOT_RE` and the count-snapshot filter's group regex are built from it.
  The extractor in the count-snapshot filter keeps its optional-status-word grammar; only the word list is
  shared. `task_filter.py` gains the public `status_rank(status)`, total over `TaskStatus` (unknown
  strings rank last). It keeps today's ranks for the seven members `task_filter.py::_STATUS_PRIORITY` holds
  (in-progress 0, blocked 1, review 2, pending 3, done 4, deferred 5, merge-deferred 6) and places the two
  it lacks: infra-hold 3 (beside pending, a waiting state) and cancelled 7 (after done and every other
  member); an unknown string ranks 8. The table is an explicit map keyed by `TaskStatus` members, not the
  enum's declaration order (pending, in-progress, blocked, deferred, review, merge-deferred, infra-hold,
  done, cancelled), which would reorder the budget-truncated active-tree render and
  `_select_proactive_sample`; a δ2 test pins the full order and fails when a new member is unranked. δ2
  keeps the private name `_STATUS_PRIORITY` for the table, so the function-local import in
  `task_knowledge_sync.py::_select_proactive_sample` keeps working; θ3, the first leaf that already
  depends on both δ2 and ι, repoints that import to a module-top import of `status_rank`, after which
  nothing outside `task_filter` reads the table (ι imports nothing δ2 creates).
- **D7 — `RunFindingIndex` is per run and owns all six run maps (R13).** One object per live run holds the
  run's stage registry (today `_run_entries`), its resident stages (today the run's keys of
  `_resident_entries`), its active stage (today `_active[run]`) and the four dedup maps.
  `ReconReportState` keeps `dict[run_id, RunFindingIndex]`; its `_state` view derives from the indexes.
  Key derivation is one method (`keys_for`) built on C1; `unregister` re-derives from the stored finding
  and pops only entries that still name it. Run quiescence (no resident stage) drops the object, replacing
  the four `pop`s in `tick`. Per-entry mirrors and persisted slices are views the index computes; the
  persisted `entry_json` shape is unchanged so `hydrate_from_store` reads old rows. The index lives in its
  own module, `fused_memory/server/recon_finding_index.py`, which does not import `recon_report.py`; the
  record dataclasses it needs (`_Finding`, `_ReportEntry`) move with it or behind a Protocol (tactical).
- **D8 — `FindingCategory` (C5).** The closed set is the nine schema values; `informational` is True for
  `CROSS_PROJECT_ROUTING` only, which reproduces today's prefix default for the one in-vocabulary
  `cross_project_*` member. Unknown input is refused with the severity precedent's envelope (OQ-2), and the
  refusal is recorded (D18, C11); the
  routing guard, `scope_freshness`, the Stage-3 guards, the systemic filter and Stage 2's
  `get_findings_for_run` filter compare members; `FINDING_ITEM_SCHEMA`'s enum is
  `list(FINDING_CATEGORY_VALUES)`; the prompt fragment library (S12/C7: `prompts/fragments.py` once S12 ι2 has
  moved it out of `prompts/__init__.py`) renders the vocabulary into the add_finding guidance the way it
  renders severity, and `prompts/stage3.py`'s hand-written list goes.
- **D9 — The edge-sweep primitive (C6).** It consumes S26's checked read (C7 `MemoryService.enumerate_valid_edges`,
  task 9203), records `enumeration_complete` / `enumeration_incomplete_kind` into its typed stats before a
  structural incompleteness is re-raised (the "load-bearing" order, stated once), and refuses to
  invalidate anything from an incomplete read. That is a behaviour change, stated here on purpose: today
  both sweeps proceed on an EMPIRICAL incompleteness (see the comment in
  `stale_priority_override_edge_sweep.py::sweep_stale_priority_override_edges` and the status sweep's
  docstring) and invalidate from the partial read; after η such a cycle invalidates nothing and records
  `enumeration_complete` False. Measured cost (critic, live journal): 1 `short_read` per sweep in about 950
  completed runs since 2026-10-25, so roughly one skipped sweep per thousand cycles, retried by the next
  cycle. The invalidation loop re-raises `CancelledError`,
  `KeyboardInterrupt` and `SystemExit` and counts the rest. The agent id is
  `recon_stage_agent_id(StageId.memory_consolidator)` (S14 μ); `citation_verifier.py::REPOINT_AGENT_ID`
  derives the same way. Stats are one `EdgeSweepStats` value with `to_stats(prefix)`, which
  `MemoryConsolidator` merges wholesale (the two parallel stat-key constant pairs go). Under OQ-1 the sweep
  writes no replacement fact (D19).
- **D10 — The task_knowledge_sync split (ι).** Verbatim moves: `IntegrityCheck` →
  `stages/integrity_check.py`; `_format_report`, `_format_flagged`, `_FLAGGED_ITEMS_CHAR_BUDGET` →
  `reconciliation/stage_report_format.py` (shared by both stages, so a leaf rather than a reach-back);
  snapshot I/O (`_verify_task_count_snapshot_written`, `_prune_task_count_snapshots`,
  `_write_task_count_snapshot`, its source constants) → `reconciliation/task_count_snapshot_store.py`; the
  growth sweep → `reconciliation/standing_decision_growth_sweep.py`; the reify briefing hook →
  `reconciliation/policies/briefing_refresh.py`, as `run_briefing_refresh(…, briefing_gap_source=
  run_briefing_known_gaps_script)` — the one keyword added during the move, so tests pass a fake gap
  source instead of patching the script runner (task 7065's ask). `prompts.stage3` becomes a top-level
  import. Ordering against S12's edits to the moved clusters: the lead ruled OQ-6 (a), ι first, the same
  ruling as S12/OQ-8 (a): ι is a verbatim move with a short chain (α, 9203), and S12's
  stage test builder, stage-interface, streak-alarm and task-write leaves (S12 γ0, γ1, ε, κ2) then import
  or edit the moved modules —
  S12 ε deletes the growth-streak quartet in `standing_decision_growth_sweep.py` and S12 κ2 puts the
  briefing writer on the interceptor in `policies/briefing_refresh.py`. ι moves both clusters whole,
  growth-streak quartet included. ι does not touch `_select_proactive_sample`'s `_STATUS_PRIORITY`
  import (θ3 repoints it; D6), so ι imports nothing a sibling creates. A `stages/integrity_check.py`
  existed once as a re-export shim and task 157 deleted it (`7933f4fd0a`); the new module holds the class
  itself and `task_knowledge_sync.py` re-exports nothing.
- **D11 — No recon control-plane marker lives in Mem0; the dormant ones are deleted, not migrated (κ1,
  κ2, κ3; OQ-7 ruled (b)).** Five Mem0 pools are in the inventory (findings row `fk-9165ae3a1e72`), each with a
  disposition:
  - `flag_for_stage2` (the Stage-1→Stage-2 relay): deleted with its reader (κ2); recon-stage writes
    carrying the key are refused at the write contract (κ3). 0 flags relayed since 2026-10-08.
  - `stage2_persistence_marker` and `stage2_escalation_marker` (FIX D: count how many cycles a relayed flag
    persisted, escalate the stale ones): deleted (κ1). They act only on Mem0-relayed flags, so they have
    had nothing to count; 0 rows of either in Mem0, 0 `stale_flags_escalated` fleet-wide.
  - `stage1_human_operator_stall_marker` (`stage1_stall_detector.py`: count cycles a
    `human_operator_required` flag recurs, escalate after 5): deleted with the escalation that counts it
    (κ1). It is the human-operator stall alarm with its marker (an L1 escalation for a task flagged as
    needing a human operator for 5 cycles), and it cannot fire, because Stage 1's output schema has no field
    for that status and no Stage-1 report since 2026-06-01 carried one (`resolution_status` is not a property
    of `cli_stage_runner.FINDING_ITEM_SCHEMA`); `stage1_human_operator_stalled` is absent from all 32,619
    completed runs since 2026-06-01, and 0 markers exist. Deleting it changes nothing an operator sees. The stateless gate-backlog escalation beside it
    (`maybe_escalate_stalled_gate_backlog`, 34 escalations since 2026-10-08) stays.
  - `stage2_suppress` guard memories: stay in Mem0. The guard is the completion-note memory itself
    (knowledge an LLM or a human searches, not a marker), and Stage 2 reads it through a deterministic
    Qdrant payload count (`count_memories_by_metadata`), not a semantic search.
  Persistence signalling after the deletion rests on the ledger's `stage1_flag_marker` recurrence counts
  and on the recon-report channel's own cross-cycle dedup; no new ledger kind is added. The marker GC
  (`_gc_recon_markers`) moves to `reconciliation/marker_gc.py` as `gc_recon_markers(ledger, …)` (κ1).
  `recon_ledger.py::MARKER_KINDS` and `recon_self_model.py::MARKER_KINDS` both drop each retired kind, and
  `recon_self_model.py::MARKER_LIFECYCLE` and `::MEM0_TOMBSTONE_DELETERS` drop its entries, in the leaf
  that retires it (κ1 `stage2_persistence_marker`; κ2 `flag_for_stage2`): the self-model's tuple is the
  one its renderer reads into the recon prompts, and `tests/test_recon_self_model.py` cross-checks the
  two tuples, so both change in one commit per kind. The Mem0 residue — 1 `{'flag_for_stage2': true}` row
  in dark_factory and 7 in reify at `97be1d71fa` — is read once in κ3; rows a `recon-stage-*` agent wrote
  are deleted with a tombstone through `mem0_tombstone.record_mem0_deletion_tombstones`, and rows another
  writer stamped are left (task 4375 recorded non-relay writers putting the key on audit records); the
  read and its outcome go in κ3's completion note. No drain script and no new MCP tool. The
  `stage1_flag_suppression` and `cycle_summary` Mem0 mirrors stay: they are narrative an LLM or a human
  searches, the ledger is authoritative, and no gate reads them. Leo ruled OQ-7 (b) on 2026-11-12; the
  author pass's ledger-backed relay (option (a)) and its stage-gated tool (OQ-3) are not built.
- **D12 — One Stage-1→Stage-2 channel (OQ-7 (b)).** Stage 2 reads Stage 1's findings only through the
  recon-report channel (`stages/task_knowledge_sync.py::_query_recon_report_findings`), which carried
  every handed-over finding since 2026-10-08. The Mem0 relay read (`_query_stage2_flags`,
  `Stage2FlagPartition` and its carry-forward of markers from runs whose Stage-2 agent died, task 5984),
  FIX C (the Stage-2 agent deleting relayed Mem0 flags, and `_acknowledge_resolved_stage1_markers`, which
  joins those deletions back to ledger markers: `stage2_flag_markers_acknowledged` is 0 since
  2026-10-08), and the Stage-2 prompt's "Persistent Flags", "Mem0 Active-Query Flag Deletion (FIX C)" and
  "Stale Flags Requiring Escalation" sections are deleted (κ1 the FIX D half, κ2 the rest).
  `summary_pool.py::trailing_incomplete_agent_run_ids` loses its only production caller (the
  carry-forward) and goes with it. The Stage-1 prompt's relay instructions go in κ2; the Stage-1 agent
  hands findings over through `add_finding`, as it already does.
- **D13 — Tests drive public seams (ν).** Moved functions get public names in their new modules; the
  dotted-path patches of `TaskKnowledgeSync` internals retire onto collaborators injected through S12/C2
  (`StageCollaborators`, landed by S12 γ2) or onto the S13-owned seams below; nothing is added to
  S12/C2's field list (that would be a parallel edit of S12's contract). Each private patch target, at
  `97be1d71fa` (57 string sites, 10 names):

  | Patch target (sites) | Where it goes | Owner |
  |---|---|---|
  | `_sweep_stale_persistence_markers` (10) | deleted with FIX D (OQ-7 (b)) | κ1 |
  | `_sweep_stale_mem0_flag_markers` (10) | deleted with the Mem0 pool sweeper | κ2 |
  | `_gc_recon_markers` (8) | `marker_gc.gc_recon_markers(ledger, …)`, called by the stage with its ledger; tests pass a real `ReconLedgerStore` on a temp db | κ1 |
  | `_verify_task_count_snapshot_written` (7), `_write_task_count_snapshot` (3) | a `TaskCountSnapshotStore` in `reconciliation/task_count_snapshot_store.py` (ι moves the functions there), passed into `TaskKnowledgeSync` as a constructor keyword `snapshot_store` that S13 owns (not a `StageCollaborators` field) | ι module, ν class and keyword |
  | `_run_briefing_known_gaps_script` (5) | the `briefing_gap_source` keyword on `policies/briefing_refresh.py::run_briefing_refresh` | ι, then λ |
  | `_query_stage2_flags` (5), `_sweep_stale_mem0_flag_for_stage2_markers` (4) | deleted with the relay (OQ-7 (b)) | κ2 |
  | `_acknowledge_resolved_stage1_markers` (3) | deleted with FIX C (OQ-7 (b)) | κ2 |
  | `_git_show_name_only` (2) | no seam added: the tests build a temporary git repository and pass its root as the scope's `project_root`, which `_render_done_provenance_section` already reads | ν |

  The guard test fails on (1) an underscore name imported from `task_knowledge_sync` or a module ι, κ1
  or κ2 creates; (2) a string patch target `'…task_knowledge_sync._x'` (or into those modules); (3) an
  `Attribute` whose value is a `Name` bound by `import …task_knowledge_sync as X` (or
  `from …stages import task_knowledge_sync`) with an underscore attribute; (4) `patch.object(X, '_…')` and
  `monkeypatch.setattr(X, '_…')` on such a name. `hasattr(X, '_…')` absence assertions are not reach.
  Attribute-form reach at `97be1d71fa`: 9 in `tests/test_stages.py`, 4 in `tests/test_recon_self_model.py`
  (three of whose constants κ1 and κ2 delete; the fourth, `_STAGE2_CYCLE_SUMMARY_TRIM_SOURCE`, becomes
  public), 3 in `tests/reconciliation/test_standing_decision_growth_sweep.py` (patches of
  `_maybe_escalate_growth_sweep_failures`, which S12 ε deletes).
- **D14 — Per-project policy in config (λ).** `ReconciliationConfig.project_policies:
  dict[str, ReconProjectPolicy]` (default empty) with `snapshot_write_blocked`,
  `contamination_ceiling_retired`, `briefing_refresh` and `stage2_guardrail` (an id naming a prompt
  constant, never prompt text in yaml); `fused-memory/config/config.yaml` carries today's four projects
  and reify's briefing flag; the four per-project modules and the hard-coded project compare in
  `prompts/stage2.py` go; the `exception_register` entry for the briefing allowlist goes with it. The read
  seam is C9's `ReconciliationConfig.policy_for(project_id)`, passed to each consumer as a parameter; no
  module reads a process-global config. `reconciliation.project_policies` joins
  `config/reload.py::RELOADABLE_FIELDS`, so a `reload_config` edit applies at the next recon cycle: every
  consumer reads the policy per cycle from the shared config object, which is that module's
  reload-safety rule. `STAGE3_SYSTEM_PROMPT` stays a module constant that names no project: its two
  "(currently: …)" interpolations become a sentence pointing to the per-cycle payload, and
  `IntegrityCheck`'s payload renders this project's write-blocked and retired-ceiling notes from
  `policy_for(project_id)` each cycle; `prompts/stage2.py::build_stage2_system_prompt`, already called per
  project, gains a keyword-only argument with a no-guardrail default,
  `build_stage2_system_prompt(project_id, *, policy: ReconProjectPolicy = ReconProjectPolicy())`, and
  `TaskKnowledgeSync.get_system_prompt` passes `policy_for(self.project_id)`. No importer of
  `STAGE3_SYSTEM_PROMPT` changes.
- **D15 — The event journal (C8).** One writer thread owns the journal's sqlite connection; `append`,
  `mark_processed` and `load_unprocessed` run there. `EventQueue.enqueue` becomes `async` and awaits the
  durable append before `put_nowait` (lead ruling OQ-5), so the row is on disk before the MCP tool returns, as
  today, while the loop thread does no sqlite I/O. The dead-letter replay path changes with it:
  `EventQueue.replay_dead_letters` stays synchronous and runs only off the loop thread (its one production
  caller, the `server/tools.py::replay_event_dead_letters` MCP tool, already calls it through
  `asyncio.to_thread`); called on the loop thread it raises `RuntimeError` naming `asyncio.to_thread`.
  `_resolve_replay_enqueue`'s inline branch is deleted and `_enqueue_on_loop` awaits `enqueue`, so a
  replayed event is never counted `replayed` without being enqueued. The 4 direct calls in
  `tests/test_event_queue.py` move to `await asyncio.to_thread(queue.replay_dead_letters)`.
- **D16 — Prose stance.** Every leaf that moves or rewrites code applies the Comments stance in
  `docs/code-quality.md`: measured figures, ruled-out hypotheses and reviewer answers go to the commit
  message or to `docs/recon-flag-dedup.md` / `docs/recon-prose-grammar.md` with a one-line pointer; what
  stays is the invariant the code relies on, at its symbol. Each new module states its purpose in one
  sentence. Leaf letters never appear in code.
- **D17 — Who decides, per decision area (`one-decision-partition-per-area`, INV-23).** Every decision this plan's
  tables, guards and sweeps make, with the one place that makes it:

  | Decision | Decided by | Where | Escape hatch |
  |---|---|---|---|
  | Is this the same flag? | code | C1 identity functions (α) | — |
  | Is this flag's subject suppressed? | code: any component of the parsed subject matches a row (R13) | `filter_suppressed` (β) | an over-suppression is recorded on the row (`partial_hits`) for the operator who wrote it |
  | Does a flag_type belong to a vocabulary? | code | `FlagTypeFamily.matches`, drift logged (γ) | the drift log names unseen spellings |
  | Is a word a task status; which statuses are terminal? | code | `shared.task_statuses` (δ2) | — |
  | Which category does a finding belong to? | the stage agent (it classifies a finding it wrote) | the stage run | refused when off the list (next row) |
  | Is the category on the list? | code | `add_finding` refusal (ε, C5) | the refusal is recorded (C11) |
  | Is a recorded refusal a candidate for a new category, or are too many findings lost? | code, by thresholds that are script arguments | `check_recon_category_candidates.py` (π), run by ρ's chain | none needed: the script only decides whether to escalate |
  | Does a flagged candidate deserve a new category value, a prompt synonym, or nothing? | judgement | the born-at-L2 `milestone_check_failed` escalation ρ files (deny-listed from L2 auto-close) | it is already at the human level |
  | Is a status edge stale? | code: the sweep's selection, corroborated against the live task store (INV-3) | stale-status sweep (η) | — |
  | Does an edge the sweep's grammar cannot parse state a stale task status? | the Stage-1 agent (prose the grammar cannot enumerate) | Stage-1 prompt (η) | it invalidates and flags it in the cycle summary, so the grammar is corrected at the source |
  | What replaces a retired status edge? | nobody: nothing is written; the task store answers status | η, Stage-1 prompt | — |
  | May an incomplete edge read drive invalidation? | code: never | `read_edges_for_sweep` (C6) | `enumeration_complete` False in the stats |
  | Which run does a recon-report call belong to? | code: the connection's run-id header; a differing argument is refused | `_report_run_id` (ξ, C10) | the refusal names both values |
  | Which per-project policy applies? | config | `ReconciliationConfig.policy_for` (λ) | `reload_config` |
  | Which Mem0 relay residue rows are deleted? | code: the writer's `recon-stage-` prefix | κ3's one-time read | rows another writer stamped are left and listed in κ3's completion note |

  No decidable row is handed to an LLM; the two LLM rows are classification of the agent's own finding and
  reading prose the grammar cannot parse. The evals-are-hermetic rule is satisfied without a task here: no
  eval cell constructs a `ReconReportStore` (only `server/main.py` builds one) or the recon harness, and
  ρ's check reads the production store from the orchestrator's deterministic runner, not from an eval.
- **D18 — Refused categories are recorded where analysis can read them (ε, C11; Leo's addition to OQ-2).**
  `add_finding` refuses an off-list category before any truncation or dedup work, as it refuses an
  off-list severity, so no finding, index or other stat of the run changes. The refusal does three things:
  it returns `invalid_category` listing the nine values; it writes one row to the append-only
  `rejected_findings` table in `recon_report_state.db`, owned by `server/recon_report_store.py`; and it
  adds one to the stage entry's `findings_rejected_invalid_category` stat, which reaches the stage report
  through the assembled report like every recon-report stat, so the rate stays readable in the run
  records after the table is pruned. The row keeps what an analysis needs: when, project, run, stage, the
  raw category as sent (capped at 200 characters), the finding's flag type, severity and task id, 500-character
  excerpts of the description and suggested action, and the description digest (C1) of the description as
  `add_finding` caps it for dedup. A later `add_finding` in the same run whose digest matches and whose
  category is accepted stamps `retried_as` with that category on every unstamped row of that run and
  digest; a row never stamped is a lost finding. A label agents keep retrying as `other` is a candidate
  for its own value; a label retried as an existing value is a synonym the prompt can name. The table is
  not touched by `ReconReportStore.delete_run`; rows older than 180 days are pruned in the same transaction
  as each insert. A failed record write is logged and never changes the refusal the caller receives (the
  refusal stands; the stat still counts it). With persistence disabled (the store failed to open, which
  is logged today) the refusal is logged and counted only.
- **D19 — Recon writes no task status into memory (η; OQ-1 and OQ-8, Leo's rule).** The stale-status
  sweep invalidates the stale edges its selection finds and writes nothing in their place:
  `build_supersede_fact`, `_SUPERSEDE_FACT_TEMPLATE`, `_MAX_SUPERSEDE_WRITES_PER_CYCLE` and the three
  `stale_blocked_edges_supersede*` / `_superseded` stats go. The Stage-1 prompt changes in three places,
  all in η: the paragraph describing the sweep ("Already done for you") says it invalidates and writes no
  replacement and that the task store answers status; the "Resulting-state-only temporal_facts" section
  and the "update_edge Temporal Limitation" section (including its "Encoding effective dates" paragraph's
  list of snapshot kinds, which loses "task-count" and "task-status", and its worked example `"As of
  2026-05-09: project dark_factory has 3 blocked, 18 done, 42 total."`, a task-status count that the
  recon-stage count-snapshot gate in `server/tools.py::add_memory` already refuses with
  `count_snapshot_write_blocked`, replaced by a non-task snapshot) say that for a task's status the agent
  invalidates the stale edge and writes nothing, while every other temporal fact keeps the two-step; and the residual paragraph's "fix it AND
  flag it" becomes "invalidate it AND flag it". The detector (`status_snapshot_extraction.py`'s grammar)
  and the sweep's selection stay: they are the cleanup backstop for status prose other writers add, until
  the recall-from-the-task-store design session (spawned by the lead; not a task of this plan) sets a rule
  for every writer. This plan designs none of that session's questions: the read side, the write rule for
  non-recon writers, the Stage-2 task-count snapshot and the orchestrator's completion notes are its, and
  are out of scope here.
- **D20 — The recon-report run id comes from the connection; a differing argument is refused (ξ, C10;
  lead ruling S12/OQ-10 with Leo's S12-2).** S12 δ puts `CAUSATION_ID_HEADER` on the stage's
  recon-report connection. ξ resolves every run-taking tool's run through `_report_run_id`: a header with
  no argument, or with an equal one, gives the header's run; a header with a differing non-blank argument
  is refused before anything is read or written, the caller receiving S12/C3's `StageIdentityRefusal`
  shape with `field='run_id'` (`error_type='stage_identity_mismatch'`, naming both values and saying the
  connection already identifies this run), and the server writing one `recon_stage_identity_refused` log
  line, the same event name S12's resolvers write; no header falls back to the argument as today (tests,
  scripts, non-stage callers); neither gives `run_id_missing`. The decision is S12/C3's one predicate,
  `stage_identity_refusal(ctx, agent_id=None, causation_id=run_id)`, so the two servers cannot disagree
  about what a mismatch is; ξ widens C3's `field` literal by the one member `'run_id'` and re-labels the
  predicate's `'_causation_id'` refusal with `dataclasses.replace(refusal, field='run_id')`, so the error
  names the argument the caller actually sent. There is no override path and no `recon_report_identity_override`
  event. In the same change the stage prompts stop asking the model to pass `run_id` to recon-report
  calls. The `search` exclusion S12 applies to `agent_id` has no counterpart here: no recon-report tool has
  a filter-shaped `run_id`.

## Contracts

```python
# C1 — fused-memory/src/fused_memory/reconciliation/flag_identity.py   (α lands; β, γ extend)
# Imports: stdlib, shared.*; nothing from server/, stages/, services/, middleware/.
def normalize_description(text: str) -> str                 # ' '.join(text.split()).casefold()
def description_digest(text: str) -> str                    # sha256(normalize_description(text)) hex, 64 chars
def content_fingerprint(text: str) -> str                   # CONTENT_FP_PREFIX + description_digest(text)[:CONTENT_FP_HEXLEN]
CONTENT_FP_PREFIX: Final = 'fp:'; CONTENT_FP_HEXLEN: Final = 32; CONTENT_FP_FLAG_TYPE: Final = '__content_fp__'
def is_content_fingerprint_task_id(value: str) -> bool
@dataclass(frozen=True)
class FlagSubject:
    raw: str | None                         # the value as the finding carried it (str-coerced int), verbatim
    components: tuple[str, ...]             # split on ',', stripped, blanks dropped, first-position dedup
    @property
    def is_composite(self) -> bool
    def covered_by(self, others: Iterable[str]) -> frozenset[str]   # components also in others
def parse_flag_subject(raw: object) -> FlagSubject          # total: None/bool/float/list/... -> components == ()
def is_valid_marker_task_id(value: str) -> bool             # fp:<32 hex> | every component digits or <project>:<digits>
def recon_signature_task_id(subject: FlagSubject) -> str | None   # numeric-aware sorted join (recon_report's persisted form)
def compute_flag_signature(flag: Mapping) -> tuple[str, str] | None
def compute_marker_signature(flag: Mapping, *, project_id: str) -> tuple[str, str] | None   # keyword unchanged
def compute_content_fingerprint_signature(flag: Mapping) -> tuple[str, str] | None
def canonical_flag_type_family(flag_type: str) -> str
def extract_flag_uuids(flag: Mapping) -> set[str]
class FlagTypeFamily: ...                                   # moved verbatim in α; C2 extends in γ
```

Persisted-format contract (C1's module docstring states this table; the golden fixture pins it):

| Output | Stored by | A change requires |
|---|---|---|
| `content_fingerprint` | `recon_ledger` `stage1_flag_marker` rows (`task_id` = `fp:…`, `flag_type` = `CONTENT_FP_FLAG_TYPE`); `scope_freshness` snapshot metadata `subject_description_fingerprint` | a ledger re-key migration and a scope-snapshot re-stamp (else every snapshot reads changed for one cycle) |
| `description_digest` | `recon_report_state.db` `entry_json` `deschash_to_finding` | re-keying persisted runs, or waiting out their TTL |
| `compute_marker_signature` (lexicographic union, `project:id` for foreign citations) | `recon_ledger` `stage1_flag_marker` identity | a ledger re-key migration (else recurrence history is orphaned) |
| `recon_signature_task_id` (numeric-aware sort, keeps `fp:`) | `recon_report_state.db` `signature_to_finding`, `sig_anchor_slice` | re-keying persisted runs, or waiting out their TTL |

Invariants: every function is pure and total over LLM-authored input; `parse_flag_subject(x).components ==
task_id_components(x)` for every input today's function accepts; each stored output is byte-equal to
today's for the fixture corpus. Error semantics: none raise on malformed input; malformed yields an empty
subject or `None` signature.

```python
# C2 — FlagTypeFamily matching modes (γ), in flag_identity.py
class FamilyMatch(StrEnum): CANONICAL = 'canonical'; EXACT = 'exact'; STEMS = 'stems'
@dataclass(frozen=True)
class FlagTypeFamily:
    name: str; spellings: frozenset[str]; drift_tokens: frozenset[str]
    unseen_spelling_tokens: frozenset[str] = frozenset(); match: FamilyMatch = FamilyMatch.CANONICAL
    def matches(self, flag_type: object) -> bool             # False for non-str/empty, in every mode
    def log_drift(self, flags: Iterable[Mapping], *, log_event: str) -> None
```

Rule (enforced by γ's guard, an AST scan of `fused-memory/src/fused_memory/reconciliation/**/*.py`
excluding `flag_identity.py`): no operand that is a str constant, a set/tuple/list/frozenset of str
constants, or a same-module `Name` bound at module level to one of those, is compared with `==`, `!=`,
`in` or `not in` against an expression naming `flag_type`, or iterated in
`any(tok in <flag_type expr> for tok in …)`; the module calls a `FlagTypeFamily` instead. Ledger-row
comparisons through `canonical_flag_type_family` are not vocabulary tests and are allowed. The named-constant
and `any()` forms are the live cases (`flag_dedup.py::ABSENCE_FLAG_TYPES` and
`::PHANTOM_TASK_CREATION_FLAG_TYPES` tested with `flag_type in NAME`;
`preservation_specimen_guard.py::STRANDED_FLAG_TOKEN_FAMILY` iterated as substrings); an inline-literal-only
rule finds just 2 sites today and would pass the tree with the pattern still allowed.

```python
# C3 — the TaskStatus-derived lexicon (δ2), in reconciliation/recon_prose_grammar.py
STATUS_WORD_ALTERNATION: str      # built once at import from TaskStatus values; '-' -> '[-_ ]?'
COUNT_SNAPSHOT_EXTRA_WORDS: tuple[str, ...] = ('total', 'canceled')
COUNT_SNAPSHOT_RE: re.Pattern[str]  # built from STATUS_WORD_ALTERNATION
# fused-memory/src/fused_memory/reconciliation/task_filter.py
def status_rank(status: object) -> int   # total over TaskStatus; unknown -> len(TaskStatus)
```

Invariants: for every `TaskStatus` member `s`, `is_count_snapshot(f'40 pending / 12 {s.value}')` is True;
no reconciliation module holds its own terminal-status set (terminal sets are
`shared.task_statuses.TERMINAL`), checked by δ2's sidecar check `terminal-literal-gone` (the three named
sets absent from `fused-memory/src/`) and δ2's B6 test; `status_rank` ranks every `TaskStatus` member, pinned
by δ2's order test (D6). Other status-string sets that are not terminal sets (for example
`targeted.py::_SNAPSHOT_CONTRADICTING_TRANSITIONS`, `frozenset({'done', 'cancelled', 'deferred'})`) are
out of this invariant; S10's task-status literal guard (task 9285) covers comparisons only, not
declarations. S14 owns the source (R13).

```python
# C4 — fused-memory/src/fused_memory/server/recon_finding_index.py   (ζ)
@dataclass(frozen=True)
class FindingKeys:
    signature: tuple[str | None, str | None] | None   # (recon_signature_task_id, flag_type)
    desc_hash: str | None                              # description_digest, null-null findings only
    cited_task: str | None                             # 'project:task' of cited_tasks[0]
    derived_signature: tuple[str, str | None] | None   # cite_task's (cited task id, flag_type)
class RunFindingIndex:
    run_id: str
    def install(self, entry) -> None                    # start_report / hydrate: registry + resident
    def activate(self, stage: str) -> None
    @property active_stage(self) -> str | None
    def evict(self, stage: str) -> None                 # tick: not resident; registry keeps the entry
    @property quiescent(self) -> bool                   # no resident stage -> the owner drops this object
    def reachable_entries(self) -> list                 # live or evicted, for persistence reach
    def keys_for(self, finding) -> FindingKeys          # the only key derivation
    def register(self, finding, entry) -> FindingKeys   # add_finding
    def register_citation(self, finding, entry) -> None # cite_task's derived signature and cited-task key
    def unregister(self, finding) -> None               # _purge_finding: re-derives, pops only own entries
    def by_signature(self, sig) -> str | None; def by_desc(self, h) -> str | None
    def by_cited_task(self, key) -> str | None; def owner_of(self, finding_id) -> object | None
    def owned_slice(self, entry) -> 'PersistedSlice'    # what _persist_run writes for entry
    def absorb(self, entry, slice: 'PersistedSlice') -> None   # hydrate_from_store
    def violations(self) -> list[str]                   # empty when the invariant holds
```

Invariant: every index key names a finding id held by exactly one reachable entry of this run, and every
live, unsuperseded finding's keys are present. `violations()` reports breaches; the test suite asserts it
is empty after every scenario in the boundary sketch. Error semantics: `register` returns the keys and
never raises on a duplicate (the caller decides `duplicate_finding`); `unregister` of an unknown finding
is a no-op.

```python
# C5 — fused-memory/src/fused_memory/models/reconciliation.py   (ε)
class FindingCategory(StrEnum):
    MEMORY_STALE = 'memory_stale'; MEMORY_DUPLICATE = 'memory_duplicate'
    MEMORY_CONTRADICTION = 'memory_contradiction'; TASK_MEMORY_MISMATCH = 'task_memory_mismatch'
    MISSING_KNOWLEDGE = 'missing_knowledge'; CROSS_STORE_INCONSISTENCY = 'cross_store_inconsistency'
    SYSTEMIC_PATTERN = 'systemic_pattern'; CROSS_PROJECT_ROUTING = 'cross_project_routing'; OTHER = 'other'
    @property
    def informational(self) -> bool              # True only for CROSS_PROJECT_ROUTING
FINDING_CATEGORY_VALUES: tuple[str, ...]          # derived; never hand-declared elsewhere
# server/recon_report.py add_finding, on an unknown category (OQ-2 (a)): refused with nothing in the
#   run changed but the refusal stat, and recorded; the envelope and the record are C11
```

```python
# C6 — fused-memory/src/fused_memory/reconciliation/graph_edge_sweep.py   (η)
@dataclass
class EdgeSweepStats:
    enumeration_complete: bool | None = None; enumeration_incomplete_kind: str | None = None
    candidates: int = 0; invalidated: int = 0; errors: int = 0
    def to_stats(self, prefix: str) -> dict[str, object]
async def read_edges_for_sweep(memory_service, *, project_id: str, stats: EdgeSweepStats,
                               log: logging.Logger) -> list[EdgeDict]
    # C7 checked read; records completeness into stats BEFORE re-raising IncompleteEnumerationError;
    # an EMPIRICAL incompleteness returns the rows with enumeration_complete False and the caller
    # invalidates nothing (fail-closed: an incomplete read never becomes a destructive write)
async def invalidate_edges(memory_service, edges: Sequence[EdgeDict], *, project_id: str, stage: StageId,
                           run_id: str, invalid_at: datetime, stats: EdgeSweepStats,
                           log: logging.Logger) -> None
    # agent_id = recon_stage_agent_id(stage); re-raises CancelledError/KeyboardInterrupt/SystemExit
```

Each sweep is then `stats = EdgeSweepStats(); edges = await read_edges_for_sweep(...); stale =
<its own selection>(edges, …); if stats.enumeration_complete: await invalidate_edges(...)`.

```python
# C7 — recon marker GC and the retired Stage-2 Mem0 channel   (κ1, κ2, κ3)   OQ-7 (b), ruled by Leo
# reconciliation/marker_gc.py (κ1; imported by stages/task_knowledge_sync.py, never the reverse)
async def gc_recon_markers(ledger, *, project_id: str, taskmaster, run_id: str,
                           now: datetime | None = None) -> int     # today's _gc_recon_markers, public
# reconciliation/recon_ledger.py — one kind left after κ2 (κ1 drops stage2_persistence_marker,
#   κ2 drops flag_for_stage2; 0 ledger rows of either at 97be1d71fa)
MARKER_KINDS = ('stage1_flag_marker',)
# reconciliation/recon_self_model.py — the renderer's tuple, changed in the same leaf per kind
MARKER_KINDS = ('stage1_flag_marker', 'stage1_flag_suppression', 'cycle_summary')
#   MARKER_LIFECYCLE drops 'stage2_persistence_marker' (κ1) and 'flag_for_stage2' (κ2);
#   MEM0_TOMBSTONE_DELETERS drops 'stage2_persistence_marker_gc_sweep' (κ1), 'stage1_flag_marker_gc_sweep'
#   and 'flag_for_stage2_gc_sweep' (κ2); historic tombstone rows keep their tags as data
# reconciliation/prompts/fragments.py (κ3; S12 ι2 moved the fragment library there from prompts/__init__.py):
#   FLAG_FOR_STAGE2_MARKER_KIND is deleted (no prompt writes that kind after κ2)
# reconciliation/flag_record_contract.py (κ3)
FLAG_FOR_STAGE2_RELAY_RETIRED: Final = FlagKindRefusal(
    error='flag_for_stage2_relay_retired', error_type='ReconFlagForStage2RelayRetired',
    hint='Stage 1 hands findings to Stage 2 through the recon-report add_finding tool; '
         'the Mem0 flag_for_stage2 relay is retired')
def recon_stage_flag_kind_refusal(agent_id: object, metadata: object) -> FlagKindRefusal | None
#   unchanged guard: None unless agent_id starts with 'recon-stage-'; then, besides today's kind table,
#   returns FLAG_FOR_STAGE2_RELAY_RETIRED when metadata['flag_for_stage2'] is True or
#   metadata['kind'] == 'flag_for_stage2'. A non-recon writer stamping the key is NOT refused (task 4375
#   recorded non-relay LLM writers putting it on audit records); an explicit False (the completion-marker
#   convention flag_dedup.py::_is_completion_flag reads on report flags) is not affected.
```

Invariants: after κ2 no recon module reads or writes a Stage-2 marker or relay row in Mem0 (no
`search`, `count_memories_by_metadata`, `get_memories_by_metadata` or `delete_memory` keyed on
`flag_for_stage2`, `stage2_persistence_marker`, `stage2_escalation_marker` or
`stage1_human_operator_stall_marker`); every `MARKER_KINDS` row is reaped by `ReconLedgerStore.gc`
(expiry or terminal task); the two `MARKER_KINDS` tuples and `MARKER_LIFECYCLE` agree, as
`tests/test_recon_self_model.py` checks today.

```python
# C8 — fused-memory/src/fused_memory/reconciliation/event_journal.py   (μ)
class EventJournal:
    async def append(self, event) -> None             # awaits the writer thread's INSERT + FULL commit
    async def mark_processed(self, event_id: str) -> None
    async def load_unprocessed(self) -> list[ReconciliationEvent]
    def close(self) -> None                           # drains and joins the writer thread
# reconciliation/event_queue.py
class EventQueue:
    async def enqueue(self, event) -> bool            # await journal.append, then put_nowait (OQ-5 (a))
```

```python
# reconciliation/event_queue.py — the replay path (μ, D15)
class EventQueue:
    def replay_dead_letters(self, *, project_id: str | None = None) -> dict[str, int]
        # off the loop thread only; on the loop thread raises RuntimeError naming asyncio.to_thread
    async def _enqueue_on_loop(self, event) -> bool     # awaits self.enqueue(event)
    # _resolve_replay_enqueue's inline (on-loop) branch is deleted
```

Invariant (INV-8): the event-loop thread performs no sqlite call on the enqueue, drain or replay path; the
journal connection is used only by its writer thread. A replay either raises (on the loop thread) or
awaits each enqueue (off it); it never counts an event `replayed` that was not enqueued. Error semantics
otherwise unchanged: an append failure raises to the caller as today.

```python
# C9 — fused-memory/src/fused_memory/config/schema.py   (λ)
class ReconProjectPolicy(BaseModel):
    snapshot_write_blocked: bool = False; contamination_ceiling_retired: bool = False
    briefing_refresh: bool = False; stage2_guardrail: Literal['autopilot_video_contamination'] | None = None
class ReconciliationConfig(BaseModel):
    project_policies: dict[str, ReconProjectPolicy] = {}   # keyed by canonical project_id
    def policy_for(self, project_id: str) -> ReconProjectPolicy   # ReconProjectPolicy() for an unknown project
# fused-memory/src/fused_memory/config/reload.py: 'reconciliation.project_policies' joins RELOADABLE_FIELDS
```

Read seam (λ): each consumer receives the policy, or the project sets derived from it, as a parameter.
Today `prompts/stage3.py` interpolates `SNAPSHOT_WRITE_BLOCKED_PROJECTS` and
`CONTAMINATION_CEILING_RETIRED_PROJECTS` into `STAGE3_SYSTEM_PROMPT` at import time, and `prompts/stage2.py`
imports `AUTOPILOT_VIDEO_CONTAMINATION_GUARDRAIL`. After λ, `STAGE3_SYSTEM_PROMPT` stays a module constant
that names no project (the two "(currently: …)" interpolations become a sentence pointing to the per-cycle
payload); `stages/integrity_check.py::IntegrityCheck`'s payload renders this project's write-blocked and
retired-ceiling notes from `policy_for(project_id)` each cycle; `prompts/stage2.py::build_stage2_system_prompt`
(already per project) gains a keyword-only argument with a no-guardrail default,
`build_stage2_system_prompt(project_id, *, policy: ReconProjectPolicy = ReconProjectPolicy())`, and
`TaskKnowledgeSync.get_system_prompt` passes `policy_for(self.project_id)`; the two Stage-3 filters
(`flag_filters/stage3_findings.py::filter_blocked_snapshot_findings`, `::filter_contamination_ceiling_findings`,
which today reach `is_snapshot_write_blocked` / `is_contamination_ceiling_retired` through function-local
imports) take `policy`; `harness.py`, `stages/task_knowledge_sync.py` and
`task_count_snapshot_cadence.py` read `StageCollaborators.config` (S12/C2) or `self.config` and call
`policy_for`. No module reads a process-global config. A `reload_config` change applies at the next recon
cycle (every consumer reads per cycle from the shared config object).

```python
# C10 — recon-report run id from the stage connection; a differing argument is refused   (ξ)
#       (lead ruling S12/OQ-10 (a) under Leo's S12-2)
# fused-memory/src/fused_memory/server/recon_report.py
_ERR_RUN_ID_MISSING: dict[str, str] = {'error': 'run_id_missing', 'error_type': 'ReconReportRunIdMissing',
                                       'hint': ...}   # mirrors _ERR_RUN_UNKNOWN ('run_id_unknown')
def _report_run_id(ctx, run_id: str | None) -> str | StageIdentityRefusal | None
    # header present, argument absent or equal  -> the header's run id
    # header present, argument non-blank and different -> the refusal from S12/C3's one predicate,
    #   stage_identity_refusal(ctx, agent_id=None, causation_id=run_id), which names field '_causation_id';
    #   _report_run_id re-labels it dataclasses.replace(refusal, field='run_id') (header and both values
    #   unchanged); one recon_stage_identity_refused log line (both values, the tool, the field)
    # header absent -> the argument (today's behaviour); neither -> None (the tool returns _ERR_RUN_ID_MISSING)
# every recon-report tool whose signature takes run_id (12 of 13, measured at a85cb1ad78: start_report,
#   add_finding, delete_finding, set_stat, inc_stat, complete, the five cite_*, repair_memory_citation):
#   run_id moves to the end of the parameter list as run_id: str | None = None (MCP calls bind by name, so
#   no caller depends on its position) and the tool gains an injected FastMCP ctx; a refusal returns
#   StageIdentityRefusal.as_error() and reads or writes nothing
# fused-memory/src/fused_memory/server/tool_context.py (S12/C3's module): StageIdentityRefusal.field's
#   Literal gains 'run_id' (the one widening ξ makes to S12's contract; the predicate is unchanged)
# get_recon_report_tool_signatures / render_recon_report_tool_guidance (S12/C7, prompts/fragments.py):
#   both header-supplied parameters (run_id, ctx) are dropped from the rendered call examples
```

Invariants (C10): with no header the `run_id` argument still supplies the run (tests and non-stage callers
pass it as today); with a header, a differing argument is refused with nothing read or written and one
log line, and a matching or absent argument lands in the header's run; a call with neither header nor
argument returns the structured `run_id_missing` error (`ReconReportRunIdMissing`, shaped like
`run_id_unknown`) and mutates nothing; the decision is made only by S12/C3's `stage_identity_refusal`; no
override path or override event exists; the header is read from the MCP request context, as
`server/tool_context.py::_resolve_identity` (created by S15 α, S12/C3) reads it.

```python
# C11 — the rejected-findings record   (ε writes it; π reads it)
# fused-memory/src/fused_memory/server/recon_report_store.py — owns the table; recon_report.py never writes SQL
REJECTED_FINDINGS_RETENTION_SECS: Final = 180 * 86400
# CREATE TABLE IF NOT EXISTS rejected_findings (
#     rejected_at REAL NOT NULL, project_id TEXT NOT NULL, run_id TEXT NOT NULL, stage TEXT NOT NULL,
#     category_raw TEXT NOT NULL,            -- as sent, capped at 200 characters
#     flag_type TEXT, severity TEXT, task_id TEXT,
#     description_excerpt TEXT NOT NULL,     -- 500 characters
#     suggested_action_excerpt TEXT NOT NULL,
#     description_digest TEXT NOT NULL,      -- C1 description_digest of the dedup-capped description
#     retried_as TEXT);                      -- NULL until a same-run, same-digest finding is accepted
# CREATE INDEX ... ON rejected_findings (run_id, description_digest); ... (rejected_at)
@dataclass(frozen=True)
class RejectedFinding: ...                   # one row, the fields above
class ReconReportStore:
    def record_rejection(self, row: RejectedFinding) -> None
        # INSERT, then DELETE rows older than REJECTED_FINDINGS_RETENTION_SECS, one transaction
    def stamp_retried_as(self, *, run_id: str, description_digest: str, category: str) -> int
        # UPDATE ... SET retried_as=? WHERE run_id=? AND description_digest=? AND retried_as IS NULL
    def delete_run(self, run_id: str) -> None   # unchanged: touches recon_report_state only
def read_rejected_findings(db_path: Path, *, since: float) -> list[RejectedFinding]
    # opens the database read-only (sqlite URI mode=ro); the one reader π uses
# fused-memory/src/fused_memory/server/recon_report.py — ReconReportState.add_finding, off-list category:
#   returns {'error': 'invalid_category', 'error_type': 'ReconReportInvalidCategory', 'category': <raw>,
#            'known_categories': list(FINDING_CATEGORY_VALUES), 'hint': ...}
#   writes one RejectedFinding (when a store is open); adds 1 to the entry's
#   'findings_rejected_invalid_category' stat; changes no finding, index or other stat
#   an accepted add_finding calls stamp_retried_as(run_id, digest, category) (when a store is open)
```

Invariants (C11): every refusal returns `invalid_category` whatever the record write does; a store open
means exactly one row per refusal; rows outlive their run's `recon_report_state` rows and are pruned only by
age; `retried_as` is written only by an accepted finding of the same run and digest and is never
overwritten; `read_rejected_findings` never writes. Error semantics: a failed `record_rejection` or
`stamp_retried_as` is logged (`recon_report_rejection_record_failed`) and swallowed, because the caller's
answer does not depend on it; a missing table (a store older than ε) makes `read_rejected_findings` raise
`RejectedFindingsTableMissing`, which π maps to its "not yet" exit.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| B1 | identity → stores | golden corpus of 40 flags (int/str/composite/`fp:`/foreign-cited/malformed) | identity functions from C1 | every stored output byte-equal to the fixture's literals, captured from today's `flag_dedup`/`recon_report` code |
| B2 | store → identity | scope snapshot written before the change, read after | snapshot metadata from today's `build_scope_snapshot_metadata` | `snapshot_is_fresh` True for an unchanged description |
| B3 | suppression | ledger row `3105` (wildcard), flags `'3105'`, `3105`, `'3105,4223'`, `'3105, 4223'`, `'4223,3105'`; row `'3105,4223'`, flag `'3105,4223'` | real `ReconLedgerStore` on a temp db | all dropped; the row's `partial_hits` names `4223` for the composites; a lifted row gains nothing |
| B4 | suppression, end to end | operator `write_suppression_record(task_id=3105)` then a Stage-1 cycle emitting `'3105,4223'` | `run_stage1_flag_chain` with the real ledger | flag absent from the cycle's output; partial-drop stat 1 |
| B5 | vocabulary guard | seeded fixture module with `flag_type == 'x'`, `in {'a','b'}`, `'strand' in flag_type`, a module constant `_TYPES = frozenset({'a','b'})` tested as `flag_type in _TYPES`, and `any(tok in flag_type for tok in _STEMS)` over a module tuple | γ guard | guard names each of the five; clean tree passes |
| B6 | lexicon | every `TaskStatus` value in a count snapshot | C3 | `is_count_snapshot` True; the count filter extracts the groups |
| B7 | category, producer | add_finding with `duplicate_procedural_knowledge`, `cross_project_scope_mismatch`, `memory_stale` | MCP tool on a real `ReconReportState` with a real `ReconReportStore` on a temp db | first two refused `invalid_category` with the vocabulary, no finding or index changed; two `rejected_findings` rows with the raw categories, excerpts and digests; the stage's `findings_rejected_invalid_category` stat is 2 in the assembled report; third accepted |
| B7b | category, retry and retention | a refused finding re-filed in the same run as `other`; a refused finding never re-filed; a second run's identical text accepted; a store failing on insert; `delete_run` of the first run; a row older than 180 days present at the next insert | real store on a temp db | the first row's `retried_as` is `other`, the second stays NULL, the other run's acceptance stamps nothing; the failing insert still returns `invalid_category` and logs `recon_report_rejection_record_failed`; `delete_run` leaves every `rejected_findings` row; the old row is gone after the insert |
| B8 | category, consumers | a `CROSS_PROJECT_ROUTING` finding with and without `cited_tasks` | real state | actionable False by `informational`; the routing guard and `scope_freshness` agree |
| B9 | run index | Stage 1 files and completes; TTL evicts it; Stage 2 cites one of its findings, then a fold purges it | real `ReconReportStore` | no resurrection on hydrate; `violations()` empty at every step |
| B10 | run index | restart between two add_findings of one signature | fresh state hydrated from the store | the second returns `duplicate_finding` naming the first |
| B11 | edge sweep | checked read reports structural incompleteness / empirical incompleteness / complete | fake C7 surface | raise with completeness recorded / zero invalidations, `enumeration_complete` False / invalidations counted; agent id equals `recon_stage_agent_id(StageId.memory_consolidator)`; in every case the spying memory service records no `add_memory` (no replacement status fact) |
| B11b | Stage-1 prompt, no status writes (negative) | the rendered Stage-1 system prompt | `prompts/stage1.py::STAGE1_SYSTEM_PROMPT` as the stage renders it (after S12 ι1's capability rendering) | it states that the sweep writes no replacement and that for a task's status the agent invalidates and writes nothing; it contains no instruction to write a status fact (no `has status <S>` template, no "status flips" in the two-step's scope, no task-count snapshot example such as `3 blocked, 18 done`) |
| B12 | relay write refused (negative) | a `recon-stage-memory_consolidator` `add_memory` with metadata `flag_for_stage2: true`; the same with `kind: 'flag_for_stage2'`; a non-recon agent's write with `flag_for_stage2: true`; a recon-stage write with `flag_for_stage2: false` | real MCP `add_memory` on a hermetic memory service | the first two refused `flag_for_stage2_relay_retired` naming `add_finding`, nothing written to Mem0; the last two accepted unchanged |
| B13 | no Mem0 relay read, ledger-only GC (negative) | a Stage-2 cycle with a `{'flag_for_stage2': true}` memory and a `{'source': 'stage2_persistence_marker'}` memory present in Mem0; an expired `stage1_flag_marker` ledger row and one for a task that turns `done` | real `ReconLedgerStore`, a spying memory service | Stage 2's payload has no relay, "Persistent Flags" or "Stale Flags Requiring Escalation" section; the memory service records no read or delete keyed on a retired marker; `gc_recon_markers` reaps both ledger rows |
| B14 | journal | a journal whose commit blocks for 2 s; a loop heartbeat coroutine | injected slow connection | heartbeat keeps ticking at its period during `enqueue`; the row is durable when `enqueue` returns |
| B14b | journal replay | `replay_dead_letters` called on the loop thread; then through `asyncio.to_thread` with 3 dead-lettered events | real `EventQueue` and journal on a temp dir | the on-loop call raises `RuntimeError` naming `asyncio.to_thread` and replays nothing; the off-loop call returns `replayed == 3` and the queue holds exactly 3 events, each durable in the journal |
| B15 | policies | config with a policy for a project and none for another; then a `reload_config` that flips one flag | `ReconciliationConfig` | harness, cadence, Stage-3 guards, the Stage-3 payload's policy notes and the Stage-2 prompt builder read the same answers the four modules gave, and `STAGE3_SYSTEM_PROMPT` names no project, and `build_stage2_system_prompt('autopilot_video')` without a policy renders no guardrail; after the reload the next cycle's payload, builder and guards read the new value without a restart |
| B16 | tests stance | seeded test modules: importing `_x` from `task_knowledge_sync`; patching `'…task_knowledge_sync._x'`; `tks._x` on `import …task_knowledge_sync as tks`; `patch.object(tks, '_x')`; `monkeypatch.setattr(tks, '_x', …)` | ν guard | guard names each of the five; `hasattr(tks, '_x')` is not flagged; clean test tree passes |
| B17 | recon-report identity (S12/OQ-10 (a), Leo's S12-2) | an `add_finding` carrying `CAUSATION_ID_HEADER` and no `run_id`; one with an equal `run_id`; one with a differing `run_id`; a `set_stat` with a differing `run_id`; one with no header and a `run_id`; one with neither | real `ReconReportState` behind the MCP tool | the first two land in the header's run; the two differing calls are refused with `error_type='stage_identity_mismatch'`, `field='run_id'` and both values, nothing written, one `recon_stage_identity_refused` log line each; the no-header call behaves as today; the last returns `run_id_missing` (`ReconReportRunIdMissing`) and mutates nothing |
| B18 | category-candidate check | seeded `rejected_findings` rows: one label refused 10 times over 3 runs, mostly un-retried; the same label passed as `--ruled-label`; a window whose lost share is 25% of 40 refusals; a quiet window; a database without the table | `check_recon_category_candidates.py` on a temp db | exit 1 naming the label with counts, runs and retry mix; exit 0 when the label is ruled; exit 1 naming the lost share; exit 0 with a one-line verdict; exit 75 naming the missing table |

## Cross-PRD relationship

Edge notation: "X → Y" means X depends on Y (Y lands first). Wave-2 and wave-1 tasks are named by id. S12
labels are those of S12's fold-verified draft of 2026-11-12 (all 17 labels and C1–C7 unchanged by its ruling
fold), so each S12 leaf is also named by what it delivers.

Same-file rows across plans (standing alignment rule, restating R54). A pair that shares only a test file
takes no edge: the file locks order it and the later task rebases. A pair that shares a non-test file is
chained by the lead's per-file order at the join, built from each task's declared files, unless one side's
edit there only rewrites an import line or a prose citation (S19/Q-C6, rebase-only). Where a row below says
"no edge" or "rebase-only, different symbols" for a non-test file, it records that no semantic order is
needed and which symbols each side edits; it does not exempt the pair from the per-file order, which the
lead sets in either direction. The import-only and prose-only edits this plan makes are the decompose
chain's `exempt_files`: α's `fused-memory/scripts/sweep_orphan_flag_markers.py`; δ1's import and prose
repoints of the grammar move in thirteen non-test files (every non-test file δ1 declares except
`task_filter.py`, its two new files and `fused-memory/scripts/memory_eval_staleness_sweep.py`, whose call
site reads `task_filter.` as a module attribute); δ2's four one-line `INACTIVE_TASK_STATUSES` repoints
(`citation_verifier.py`, `stale_status_snapshot_edge_sweep.py`, `server/tools.py`,
`measure_stale_status_snapshot_sweep.py`); ι's `IntegrityCheck` import line in `harness.py`; κ1's prose-only edits of `orphaned_recon_escalation_sweep.py`,
`mem0_tombstone.py`, `summary_pool.py` and `targeted.py`.

| Other PRD / stream / task | Direction | Seam mechanism or shared file | Owner | Order |
|---|---|---|---|---|
| S14 (`plans/review-all-s14-shared-leaves-prd.md`) | consumes | `shared.task_statuses` (on main; C3 derives); `shared/agent_identity.py::recon_stage_agent_id` (μ, task 6721) | S14 | η → 6721 (so η also follows 6719, 6720 through 6721's chain) |
| S14 ξ, task 6719 | same file | `flag_dedup.py` (error class-name compare in the terminal-metadata extractor) | S14 | no edge; if θ1 lands first, 6719's edit applies to `flag_filters/task_state_refutations.py` (lead note) |
| S14 θ, task 6718 | same files | the scheduler-override and park-eviction store: 6718 edits `stale_priority_override_edge_sweep.py` and `tests/reconciliation/test_stale_priority_override_edge_sweep.py`, which η rewrites | S14 | **η → 6718** (6718 has no live ancestors, measured at `97be1d71fa`) |
| S26 μ, task 9203 (`plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md`) | consumes (C7) | `MemoryService.enumerate_valid_edges` checked read; 9203 also repoints the growth sweep's `memory_service.graphiti` reach in `task_knowledge_sync.py` | S26 | η → 9203; ι → 9203 (9203 repoints the growth sweep where it lives today, then ι moves the repointed code). 9203's 19 ancestors include no long-chain task |
| S12 (wave 3, fold-verified draft `plans/review-all-s12-reconciliation-run-lifecycle-and-stage-interface-prd.md`) | consumes | S12/C2 `StageCollaborators`, delivered by S12 γ2 ("reconciliation stages constructed with their collaborators and per-cycle inputs"): `TaskKnowledgeSync` built with its collaborators (R12) | S12 | ν → S12 γ2 (a contract seam; both edit `stages/task_knowledge_sync.py`). The draft's ν → S12 θ is NOT wired: it rested only on the shared test files `tests/test_stages.py`, `tests/test_harness.py` and `tests/reconciliation/test_task_count_snapshot_cadence.py` (S12 θ's one non-test file is `harness.py`, which ν does not declare), and the lead's test-file rule (after the wave-3 fold-verifiers, restating R54) leaves such pairs to the file locks; S12's draft still names it and drops it at S12's decompose under the same rule |
| S12 | does not consume | S12/C3, delivered by S12 δ ("recon stage identity and run id bound on the MCP transport"). η's in-process sweep writes take their agent id from S14 μ's `recon_stage_agent_id(StageId…)`, not from the transport | — | no edge (xref) |
| S12 | consumes (C3) | S12 δ puts the stage-identity and run-id headers (S12/C3 `stage_mcp_headers`, `STAGE_AGENT_ID_HEADER`, `CAUSATION_ID_HEADER`; names unchanged by S12's ruling fold) on the stage's recon-report connection; ξ makes the recon-report server read the run-id header, REFUSE a differing `run_id` argument through C3's one predicate `stage_identity_refusal(ctx, *, agent_id, causation_id)` with C3's `StageIdentityRefusal` shape (`error_type='stage_identity_mismatch'`, both values; ξ widens its `field` literal by `'run_id'`) and one `recon_stage_identity_refused` log line, and drop the prompts' "echo run_id" line in the same change (C10, D20). `recon_stage_identity_override` no longer exists in S12, and no `recon_report_identity_override` exists here | S13 (ξ; lead ruling S12/OQ-10 (a), unconditional) | **ξ → S12 δ**; ξ → ζ, ε (`recon_report.py`), κ3 (`prompts/fragments.py`), η (`prompts/stage1.py`), λ (`prompts/stage2.py`, `stage3.py`); ω → ξ. S12's other C3 changes reach no S13 file: `_extract_causation(metadata, agent_id, ctx)` gains `ctx` at its 17 call sites in `server/tools.py` (δ1, δ2 change only import lines there, rebase-only) and `CreationAttribution.from_submission(metadata, agent_id, *, bound_causation_id=None)` is outside S13 |
| S12 | same file `stages/task_knowledge_sync.py` (lead ruling OQ-6 (a) = S12/OQ-8 (a)) | S12 γ1 (stage payload interface), γ2, ε (consecutive-cycle recon alarms; deletes the growth-streak quartet, fk-90e488a174b7), ζ, κ2 (stage task writes only through the interceptor; the briefing writer, fk-875cca636438), ι2 (prompt fragment library to `prompts/fragments.py`) edit the file; ι moves clusters three of them then edit | S13 moves first; S12 edits the moved modules | **S12 γ0, γ1, ε, κ2 → S13 ι** (S12 γ0's test builder `stage_builders.py` imports `IntegrityCheck`; S12 names `stages/integrity_check.py`, `standing_decision_growth_sweep.py` and `policies/briefing_refresh.py` as created by ι); ι takes no edge on any S12 task. S13's later leaves on the file follow S12's last leaf on it: **θ3 → S12 ι2** and **ε → S12 ι2** (`prompts/fragments.py`, created by ι2). ι edits `harness.py`'s `IntegrityCheck` import: rebase-only against S12 α, β1, β2 (S12 γ1 → ι orders it before the rest of S12's run chain) |
| S12 | same files | `stages/memory_consolidator.py` (θ3 Hook A import, κ1 deletes the human-operator stall escalation, η `EdgeSweepStats` merge); `prompts/stage1.py`–`stage3.py` (S12 δ, which now edits `stage1.py` and `stage2.py` for its identity sweep, ι1, ι2; S13 ε, κ1, κ2, η, λ, ξ); `stages/base.py` (S12 γ1, γ2, δ, κ2; S13 ξ); `models/reconciliation.py` (S12 α, ζ; S13 ε); `recon_self_model.py` (S12 δ, ι1; S13 κ1, κ2); `server/tool_context.py` (S12 δ; S13 ξ widens one literal); `harness.py` (S13 ι, λ; S12's run chain and the η1/η2 partition); `server/tools.py` (S12 δ, its `_extract_causation` call sites under S12 OQ-11 (a1); S13 δ1, δ2 import lines); `targeted.py` (S12 κ1; S13 δ1, δ2 rebase-only; S13 κ1, κ2 ordered after it); `middleware/task_interceptor.py` (S12 κ1; S13 μ rebase-only) | each | every S13 leaf on these files except ι, δ1, δ2 and μ follows θ3 → S12 ι2, which follows S12 α … κ2, δ and ι1 (S12 ι2 → ι1 → δ), so S13's prompt and self-model tasks reach S12 δ through θ3 → S12 ι2; `harness.py` order between λ and S12 η1/η2 is the lead's per-file order; `server/tools.py` (δ1, δ2) and `targeted.py` (δ1) are import-only edits on S13's side (S19/Q-C6, rebase-only); `targeted.py` (δ2's tuples vs S12 κ1) and `middleware/task_interceptor.py` (μ vs S12 κ1) are different symbols with no semantic order, chained by the lead's per-file order at the join |
| S19 (wave 3, draft `plans/review-all-s19-package-layering-and-private-seams-prd.md`) | same modules | S19 moves into `fused_memory.recon_state`: α `recon_ledger.py`, `event_journal.py`, `event_queue.py`, `dead_letter_writer.py`; β1 `recon_self_model.py`, `standing_decision_constants.py`; β2 `task_filter.py`, `consolidation_gate.py`, `flag_record_contract.py`. `flag_identity.py`, `marker_gc.py`, `graph_edge_sweep.py`, `flag_filters/` are engine modules and stay | S19 owns `recon_state`; S13 populates | lead ruling OQ-4 (a) (= S19/Q-C5 (a)): β, μ → S19 α; κ1 → S19 β1; κ3 → S19 β2 (κ3 edits `flag_record_contract.py` and `consolidation_gate.py`); S19 β2 → δ2; S19's moves never wait on κ3. Both drafts now carry these edges. **This plan's position: S19 β2 moves `recon_prose_grammar.py` in place of `task_filter.py`, and `task_filter.py` (the tree renderer) stays in `reconciliation/`** (D5: under S19/C2's own rule nothing below `reconciliation` imports the renderer after δ1 and δ2). S19's fold-verified draft moves both ("joins"); the lead reconciles S19 β2's file list (Open for the lead). Either way no S13 edge changes |
| S10 ο, task 9285 | same files | `task_filter.py`, `targeted.py` task-status comparisons | S10 owns the literal guard; S13 owns the recon lexicon (R13) | **9285 → δ2** (lead action). No cycle: δ2's ancestors are α, δ1, β, γ, θ1, θ2 and S19 α; 9285's 150 ancestors (re-measured 2026-11-13) include 8528, so in practice δ2 lands first anyway |
| S10 α, task 9180 (high) | same files | `task_knowledge_sync.py` (`TaskKnowledgeSync.assemble_payload` and `_needs_hint_attention`: every `append=` and `update_task` line in the file), `targeted.py` | S10 | no edge: different symbols from ι, δ1 and δ2, rebase-only. S12's xref asks for ι → 9180; not wired: 9180's edits there (`assemble_payload`, `_needs_hint_attention`) are in no cluster ι moves; an edge would only make ι itself wait on 9180's 38 ancestors (22 live, re-measured 2026-11-13). S12 γ1 → 9180 stands in S12 |
| S10 κ2 9251, λ 9254 | same files | `orphaned_recon_escalation_sweep.py` (get_statuses_fresh), `gate_owned_finding_phrasing.py` | S10 | no edge: different symbols |
| S26 ρ, task 9243 | same file | `preservation_specimen_guard.py` (`metadata_row_text` import) | S26 | no edge: different symbols |
| S3 ε1, task 9188 | same file | `cli_stage_runner.py::DISALLOW_ESCALATION_MERGE_LANE_CONTROL` vs ε's `FINDING_ITEM_SCHEMA` | S3 | no edge: different symbols (9188 waits on 8528; ε must not) |
| S1 (wave 3, draft `plans/review-all-s1-one-escalation-filing-entry-point-prd.md`) | same files | S1 ι1 (recon escalation registry) edits `finding_task_escalation.py`, `orphaned_recon_escalation_sweep.py`, `harness.py`; S1 ι2 (the other recon filers) edits `stage1_stall_detector.py`, `task_knowledge_sync.py`, `targeted.py`, `preservation_specimen_guard.py`, `standing_decision_storm_escape.py` | each | different symbols, no semantic edge; the lead's per-file order chains each non-test pair at the join (S1 ι2 waits on 9285 and so on 8528; no S13 leaf may, so S1 ι2 goes after S13's leaves on its files). Under OQ-7 (b), ruled, κ1 deletes `stage1_stall_detector.py::maybe_escalate_stalled_tasks` and its stall marker, so S1 ι2's filer conversion in that module covers only `maybe_escalate_stalled_gate_backlog`; S1 ι2's `task_knowledge_sync.py` edit targets `_maybe_escalate_growth_sweep_failures`, which S12 ε deletes (S12's fold drops it from S1 ι2) |
| task 7941 | same files | public names in the sweep modules; its `_SWEEP_AGENT_ID` item is moot after η | 7941 | η → 7941; 7941 amended to drop the `_SWEEP_AGENT_ID` item (lead action) |
| task 5272 | same files | priority sweep selection rule (audit facts) | 5272 | η → 5272 |
| task 6598 (mechanical, fk-a1d78ddf0b10) | same file | `citation_verifier.py` docstrings | 6598 | done 2026-11-13 (`a16b38eec6`); the edge is satisfied and is not wired |
| task 6599 (mechanical, fk-5403085e02d0) | same file | `task_filter.py::nonterminal_completion_claim_task_ids` | 6599 | 6599 → δ1 if still pending at decompose; its file list then names `recon_prose_grammar.py` (lead actions) |
| task 5270 | same file | `shared/tests/loop_blocking_allowlist.py` rows for `_run_briefing_known_gaps_script` | 5270 owns the rows | no edge; ι re-keys the two rows when it moves the function (lead note) |
| task 6957 | same file | `recon_report.py` truncation constants | 6957 | no edge |
| task 4011 (deferred) | same file | unknown-stage `start_report` | 4011 | after ζ its fix belongs in `RunFindingIndex.install` / `activate` (lead note) |
| tasks 7388, 5996, 6846, 7807, 7065 | subsumed | — | γ; θ1–θ3; ι; κ1; ν | cancelled with `x_acceptance_reason` (lead actions) |
| the recall-from-the-task-store design session (S13-1, residual R-1 = A) | xref only | Leo's broader question: recall that draws task facts from the task store, and a write rule for every writer. SPAWNED by the lead as its own fable /team session (investigate, discuss, design), brief = the seven questions in `digests/discussion/S13.md` §S13-1 and S13-4 | that session | nothing in S13 waits on it, and it is not a task of this plan. Its outcome may re-raise the acceptance record for fk-9b76dd8a1cdb (a write rule for every writer is the stated trigger) and may retire the detector and the sweep's selection |
| task 3140 (pending, waits on 3139) | xref only | flags memories that cite a task as a blocker when that task reaches done or cancelled: a reader of task facts in memory the design session weighs (its question 6) | 3140 | no edge |
| task 9490 (high, outside every plan) | xref only | the task store refuses a cycle-closing dependency edge | 9490 | no edge; the decompose seat's wiring may keep a cycle pre-pass only as an early check |
| S15 (wave 3) | same file | `server/tool_context.py` (created by S15 α; S15's later tool-family tasks carry the resolver calls) | S15 | ξ's one-literal widening of `StageIdentityRefusal.field` and S15's later tasks on the file edit different symbols with no semantic order; the lead's per-file order chains them at the join (it is not an import-only edit); ξ already follows S15 α through S12 δ |

Wave-1/wave-2 tasks sharing files with S13 leaves, on different symbols with no semantic order (forensic
read of `metadata.files` for live tasks); test files are rebase-only, and each non-test pair is the lead's
per-file order at the join unless S13's edit is import-only or prose-only (the paragraph above):

| Task | Files shared with S13 | S13 leaves |
|---|---|---|
| 9151, 9208 (S26 γ0, ν) | `fused-memory/scripts/audit_duplicate_memories.py`, `cleanup_count_snapshots.py`, `memory_eval_staleness_sweep.py`, `sweep_orphan_flag_markers.py` | α, δ1, κ3 (import and comment lines) |
| 9248 (S10 κ1) | the three scripts above | δ1, α |
| 9240 (S10 η) | `stale_gate_citation_guard.py` | δ1 |
| 9204 (S10 δ) | `targeted.py` | δ1, δ2, κ1, κ2 |
| 9194 (S26 λ) | `config/schema.py`, `config/config.yaml` | λ |
| 9195, 9214, 9280, 9282 (9276 done 2026-11-12) | `middleware/task_interceptor.py` | μ's one `await` |
| 9278 (S10 μ1) | `tests/server/test_recon_report_store.py`, `test_recon_report_citations.py` | ζ (ε writes its store tests in a new module) |
| 9167, 9193 (S3 δ, ε2) | `tests/test_stages.py` | ι, κ1, κ2, ν |
| 9285 (S10 ο) | `stage1_stall_detector.py` | κ1 (deletes the human-operator stall path; 9285 rewrites status comparisons) |

Other out-of-batch pending tasks that share files with no semantic ordering need (per-file order at the
join, as above): 3390, 6834
(flag_dedup, stage1 prompt), 5336, 5852 (task_filter), 5451, 5714 (recon_report), 6938 (recon_ledger), 7468
(citation_verifier), 7894 (finding_task_escalation), 7508, 7856 (event_queue, event_journal), 6830, 6843,
7500, 7560, 4875, 5154 (task_knowledge_sync). Done since the draft: 6509 (merged `80cc1007f0`; its
`_format_flagged` edit is in the cluster ι moves) and 6621 (its `TERMINAL_STATUSES` import repoint is in
`task_knowledge_sync.py`'s header, which no S13 leaf moves).

## Decomposition plan

Every leaf: `planning_mode=True`, `metadata.x_finding_key` (list), `metadata.x_finding_run =
"review-all-dark_factory-20261008"`, `metadata.source = "review-all"`, file-level `metadata.files`,
`priority` = the highest severity among its keys; π and ρ, added by Leo's ruling on OQ-2, carry no finding
key and take medium priority, and ρ is a deterministic seed link whose shape its block states and which the
lead files once π is on main (a `before_done` script must exist and be executable at `submit_task`). Overlay bands: target 300–1,500 changed LOC and ≤ 10–12
files; α, δ1, δ2, ε, ι, κ1, κ2, κ3, ν, λ and θ3 exceed 12 declared paths and say why (most additions are
one-line import or test repoints of a moved name). Verbatim moves count both sides of the diff, so ι and
θ1–θ3 overstate their review cost. **High leaves (dependencies of trigger-chain gate 6698 at decompose):
α, ζ, κ1, κ2, κ3, ν.** No leaf declares `merge_lane/worker.py`, so none is parked behind task 8528 at
filing. What the high leaves inherit (closure re-measured 2026-11-13): α and ζ wait on nothing but α.
κ1, κ2, κ3 and ν wait, through θ3 → S12 ι2 (and ν → S12 γ2), on S12's whole run chain and through it on
the wave-2 tasks 9254, 9180, 9278, 9280 and 9282; task 4867, which the draft named, is done. They do
**not** inherit the merge-lane policy chain: S12's fold deleted its targeted-reconciler task's edge onto
9285, and no other S12 external dependency reaches 8528. Priority inversion, recorded and not overridden
(contract §4): the high leaves κ1, κ2, κ3 and ν wait on the medium leaves θ3 and ε and on S12's medium and
low tasks. ρ, the seed link of a recurring chain, is not a leaf of the 6698 gate: its links recur after the
program ends.

Intra-batch order, driven by the non-test files the leaves share (`flag_dedup.py`,
`task_knowledge_sync.py`, `recon_report.py`, `stage1_flag_chain.py`, `targeted.py`, `recon_ledger.py`,
`recon_self_model.py`, `mem0_tombstone.py`, `summary_pool.py`, `memory_consolidator.py`, the new filter
modules and the prompts), checked file by file so every pair of leaves sharing a non-test file is ordered.
Test files shared by two unordered leaves are rebase-only (file locks order them; the lead's test-file
rule, which wins over the overlay's same-file serialisation for test files); there is one,
`tests/reconciliation/test_task_count_snapshot_cadence.py` (δ1 repoints one `is_count_snapshot` import
line, ι repoints the snapshot-function patches), and no line is changed by both. η and ξ also edit
`tests/test_stages.py` with ι, κ1, κ2, ν and λ; each such pair is ordered through a non-test file anyway.

- α first; ζ → α; δ1 → α; ι → α, 9203 (and no S12 leaf; lead ruling OQ-6).
- β → δ1 (and S19 α; OQ-4); γ → β; θ1 → γ; θ2 → θ1; δ2 → θ2.
- θ3 → θ2, δ2, ι and S12 ι2 (OQ-6); ε → ζ, θ3 (and S12 ι2).
- κ1 → ε (and S19 β1; OQ-4); κ2 → κ1; κ3 → κ2 (and S19 β2; OQ-4).
- η → κ2, δ2, plus its out-of-batch dependencies; ν → κ2 (and S12 γ2); λ → ν.
- μ → S19 α (OQ-4), nothing in the batch; ξ → ζ, ε, κ3, η, λ and S12 δ.
- π → ε (the script reads C11's reader, which ε creates); ρ → π (a recurring link must never fire before
  its script is on main). ρ is not filed at decompose: `submit_task` refuses a `before_done.script` that
  does not exist and is not executable under `project_root`
  (`fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_script_under`, which runs
  for `planning_mode=True` too), so the lead files it, with the edge onto π's task id, the day π's merge
  lands (return `lead_actions`).
- ω → every leaf except ρ (ρ's first link fires about 30 days after filing, and its links recur after the
  program; ω checks π's script, not the chain).

Two halves. α, ζ, δ1, ι, β, γ, θ1, θ2, δ2 and μ can land while S12 runs: every file they share with S12
is ordered (α's, through ι), an import-only edit on S13's side (`server/tools.py` for δ1 and δ2 vs S12 δ,
`targeted.py` for δ1 vs S12 κ1; S19/Q-C6), a test module, or a different-symbol pair the lead's per-file
order chains at the join (`targeted.py`: δ2's two status tuples vs S12 κ1's write gate;
`middleware/task_interceptor.py`: μ's one `await` vs S12 κ1). If that order puts S12 κ1 first, δ2 or μ
waits on S12's chain to κ1 and leaves this half; and ι lands before S12's stage test builder and
stage-interface tasks (OQ-6). Everything from θ3 on touches
`task_knowledge_sync.py`, `memory_consolidator.py` or the prompts after S12's prompt-library move.
The chain is long because `task_knowledge_sync.py` and `flag_dedup.py` are each edited by six or more
leaves; the alternative (fewer, larger leaves) would put 3,000-line verbatim moves and semantic changes in
one diff, which a reviewer cannot read.

**α — Give recon flag identity one public module: fingerprint, normalizer, signatures, FlagTypeFamily, UUID extraction and the parsed flag subject**
(`task_kind=normal`, priority high, est. ~1,350 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/flag_identity.py (new),
  fused-memory/src/fused_memory/reconciliation/flag_dedup.py,
  fused-memory/src/fused_memory/reconciliation/flag_task_ids.py (deleted),
  fused-memory/src/fused_memory/server/recon_report.py,
  fused-memory/src/fused_memory/reconciliation/scope_freshness.py,
  fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py,
  fused-memory/src/fused_memory/reconciliation/sweep_deletion_guard.py,
  fused-memory/src/fused_memory/reconciliation/preservation_specimen_guard.py,
  fused-memory/src/fused_memory/reconciliation/stage1_flag_chain.py,
  fused-memory/scripts/sweep_orphan_flag_markers.py,
  fused-memory/tests/reconciliation/test_flag_identity.py (new),
  fused-memory/tests/reconciliation/test_flag_task_ids.py (deleted),
  fused-memory/tests/test_flag_dedup.py,
  fused-memory/tests/reconciliation/test_stage2_recon_report_channel.py,
  fused-memory/tests/reconciliation/test_curator_gate_resolution_sweep.py,
  fused-memory/tests/reconciliation/test_flag_dedup_citation_fallback.py,
  fused-memory/tests/reconciliation/test_flag_dedup_project_qualified_marker.py,
  fused-memory/tests/reconciliation/test_recon_cross_cycle_dedup_citations.py,
  fused-memory/tests/reconciliation/test_stage1.py,
  fused-memory/tests/reconciliation/test_unlanded_path_check.py
- Size note: 20 paths, above the >15 review trigger. The last six are one-line import repoints of the
  moved signature functions (AST scan of `ImportFrom` at `97be1d71fa`); leaving them would keep them
  working only through `flag_dedup`'s namespace, which α empties.
- Deps: none.
- Keys: fk-e30ecd1f36b8, fk-e64ee986eeed.
- Work: C1. Move the identity functions out of `flag_dedup` (public names; `compute_marker_signature`
  keeps its `project_id` keyword, used by `stage1_flag_chain.py` and the tests), fold `flag_task_ids` into
  `parse_flag_subject`, delete `recon_report`'s `_canonical_sig_field`, `_normalize_description`,
  `_description_hash`, `_canonicalize_task_id_string`, `_split_task_id_parts`, `_task_id_part_sort_key`
  and `task_knowledge_sync`'s `_flag_content_fingerprint` body (it calls C1). `flag_dedup` re-exports
  nothing it moved; every importer, test modules included, is repointed to `flag_identity`. Write the
  persisted-format table into the module docstring and the golden fixture (B1, B2) into
  `test_flag_identity.py`.
- DELETES: the two private `_content_fingerprint` imports, the second normalizer, the identity "local
  copy" comment and the false layering rationale, `flag_task_ids.py`.
- Observable signal: B1 and B2 pass in CI. After deploy, in the first post-deploy dark_factory cycle no
  new `stage1_flag_marker` ledger row is created for an identity that a pre-deploy row already holds under
  today's signature (read-only sqlite on `recon_ledger` in `data/reconciliation/reconciliation.db`: rows
  with `created_at` after the deploy, recomputed with the B1 fixture's pre-change signature function).
  `git grep -E 'def _content_fingerprint|def _normalize_description|def _canonical_sig_field' fused-memory/src`
  is empty.
- G7: INV-9 (one home per identity output), INV-5 (no lockstep copies), INV-10 (the fixture executes the
  functions against stored values, not their text).

**β — Make recon flag suppression match any component of a composite task id and record partial matches on the suppression row**
(priority medium, est. ~750 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/flag_identity.py,
  fused-memory/src/fused_memory/reconciliation/flag_dedup.py,
  fused-memory/src/fused_memory/reconciliation/recon_ledger.py,
  fused-memory/src/fused_memory/reconciliation/finding_task_escalation.py,
  fused-memory/src/fused_memory/reconciliation/gate_owned_finding_phrasing.py,
  fused-memory/src/fused_memory/reconciliation/stage1_flag_chain.py,
  fused-memory/tests/test_flag_dedup.py,
  fused-memory/tests/reconciliation/test_flag_suppression_composite.py (new),
  fused-memory/tests/reconciliation/test_gate_owned_finding_phrasing.py,
  fused-memory/tests/reconciliation/test_finding_task_escalation.py,
  fused-memory/tests/test_flag_record_write_seam.py,
  fused-memory/tests/test_recon_ledger.py
- Deps: δ1 (same file `flag_dedup.py`; through it α); S19 α (lead ruling OQ-4; it moves `recon_ledger.py`;
  the clerk repaths this file list).
- Keys: fk-5d00812d5256.
- Work: D2 under R13. `filter_suppressed` (gaining the `run_id` keyword `dedup_flags` passes),
  `write_suppression_record`'s coverage check, `_canonicalize_suppression_task_id`,
  `_decompose_suppression_task_id`, `finding_task_escalation`'s sole part and `gate_owned_finding_phrasing`'s
  cited ids read `FlagSubject`. The 8 test calls of `ReconLedgerStore.is_suppressed` (3 in
  `test_flag_dedup.py`, 4 in `test_recon_ledger.py`, 1 in `test_flag_record_write_seam.py`, measured at
  `97be1d71fa`) move onto `filter_suppressed` or `list_suppressions`, or go with the method.
- DELETES: the verbatim flag-side compare; 5 of the 8 splitters (the other 3 are C1's own);
  `ReconLedgerStore.is_suppressed`.
- Observable signal: B3 and B4 pass in CI; after deploy, a suppression row whose run shows a partial
  drop carries `partial_hits` readable through the existing ledger read path
  (`ReconLedgerStore.list_suppressions`).
- G7: INV-11 (an over-suppression is recorded, not silent; a failed record write never un-drops the
  flag); INV-4 does not fire (no escalation is filed; the per-row list is bounded at 10).

**γ — Put every recon flag_type vocabulary on FlagTypeFamily with a drift log, and guard against ad-hoc flag_type matching**
(priority medium, est. ~700 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/flag_identity.py,
  fused-memory/src/fused_memory/reconciliation/flag_dedup.py,
  fused-memory/src/fused_memory/reconciliation/preservation_specimen_guard.py,
  fused-memory/src/fused_memory/reconciliation/scope_freshness.py,
  fused-memory/tests/test_flag_dedup.py,
  fused-memory/tests/reconciliation/test_flag_identity.py,
  fused-memory/tests/reconciliation/test_preservation_specimen_guard.py,
  fused-memory/tests/reconciliation/test_scope_freshness.py,
  fused-memory/tests/reconciliation/test_unlanded_path_check.py,
  fused-memory/tests/reconciliation/test_flag_type_vocabulary_guard.py (new)
- Deps: β.
- Keys: fk-65aeb80b161d. Subsumes task 7388 (cancelled).
- Work: C2, D3. `ABSENCE_FLAG_TYPES` → `FlagTypeFamily(match=EXACT)` (its importer
  `test_unlanded_path_check.py` is repointed); phantom creation, the `systemic_pattern` flag_type test,
  `cross_project` → families; grounds and stranded stems → `match=STEMS`; every filter calls `log_drift`.
  `standing_decision_constants.py::GROUNDS_TOKEN_FAMILIES` stays data (token tuples): that module moves
  below reconciliation under S19 and must not import `flag_identity`, so the consuming filter builds its
  `FlagTypeFamily(match=STEMS)` values from it. The guard is C2's rule as an AST scan of
  `fused-memory/src/fused_memory/reconciliation/**/*.py` excluding `flag_identity.py`, covering literal
  operands, module-level named constants and the `any()` substring loop, tested against seeded fixture
  modules (B5).
- DELETES: four of the five matching mechanisms (exact frozenset, literal compare, substring stems,
  bare canonical-family set for a vocabulary).
- Observable signal: B5 passes in CI, and so does a per-family fixture test that feeds each converted
  filter a respelled flag_type (case, separator and word-order variants) and a spelling outside the
  family, and asserts the filter's declared mode (an EXACT family keeps the respelled flag, a CANONICAL
  family treats it as a member) and that the `*_filter_possible_drift` event fires for the outside
  spelling — today the absence, phantom and systemic filters log nothing for either.
- G7: INV-10 (the guard flags seeded fixtures for every operand form and passes the tree), INV-12 (no
  exemption list: the module that defines the mechanism is excluded by construction).

**δ1 — Split task_filter.py: the recon prose grammar moves to recon_prose_grammar.py, task_filter keeps the task-tree renderer**
(priority medium, est. ~1,900 LOC, mostly a verbatim move)
- Files: fused-memory/src/fused_memory/reconciliation/recon_prose_grammar.py (new),
  fused-memory/src/fused_memory/reconciliation/task_filter.py,
  fused-memory/src/fused_memory/reconciliation/flag_dedup.py,
  fused-memory/src/fused_memory/reconciliation/targeted.py,
  fused-memory/src/fused_memory/reconciliation/stage1_payload_format.py,
  fused-memory/src/fused_memory/reconciliation/stale_gate_citation_guard.py,
  fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py,
  fused-memory/src/fused_memory/reconciliation/status_snapshot_extraction.py,
  fused-memory/src/fused_memory/middleware/dependency_direction_check.py,
  fused-memory/src/fused_memory/services/completion_claim_gate.py,
  fused-memory/src/fused_memory/server/tools.py,
  fused-memory/scripts/audit_duplicate_memories.py, fused-memory/scripts/audit_ruling_overreach.py,
  fused-memory/scripts/cleanup_count_snapshots.py, fused-memory/scripts/memory_eval_staleness_sweep.py,
  fused-memory/scripts/measure_stale_status_snapshot_sweep.py,
  fused-memory/tests/test_task_filter.py, fused-memory/tests/test_recon_prose_grammar.py (new),
  fused-memory/tests/test_memory_eval_staleness_sweep.py, fused-memory/tests/test_audit_duplicate_memories.py,
  fused-memory/tests/test_harness.py,
  fused-memory/tests/reconciliation/test_task_count_snapshot_cadence.py,
  docs/recon-prose-grammar.md (new)
- Size note: 23 paths, above the >15 review trigger. 16 are one-line import repoints of the same move
  (`test_harness.py` and `test_task_count_snapshot_cadence.py` import `is_count_snapshot`); splitting them
  off would leave importers of a deleted symbol between two tasks.
- Deps: α (same file `flag_dedup.py`). Task 6599 depends on δ1 if still pending (lead action).
- Keys: fk-d9bfbf342b5b.
- Work: D5, with the placement the lead ruled under OQ-4 (a).
- DELETES: the grammar half of `task_filter.py` (774 lines move) and its 120-line rationale (to the doc).
- Observable signal: `uv run --project fused-memory pytest fused-memory/tests/test_recon_prose_grammar.py
  fused-memory/tests/test_task_filter.py` passes in CI; `task_filter.py` contains no `re.compile`; after
  deploy the MCP `add_memory` write gate still rejects a recon-stage count-snapshot write
  (`tests/server/test_recon_flag_marker_write_gate.py` path) and the stale-status sweep's next cycle
  reports the same candidate count.
- G7: INV-5; heuristic 13 (each file makes sense alone) and 14 (no shim stitching the halves).

**δ2 — Derive the recon status lexicon and terminal sets from shared.task_statuses; make active-tree selection a typed value**
(priority medium, est. ~750 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/recon_prose_grammar.py,
  fused-memory/src/fused_memory/reconciliation/task_filter.py,
  fused-memory/src/fused_memory/reconciliation/flag_filters/count_snapshot.py,
  fused-memory/src/fused_memory/reconciliation/flag_filters/task_state_refutations.py,
  fused-memory/src/fused_memory/reconciliation/orphaned_recon_escalation_sweep.py,
  fused-memory/src/fused_memory/reconciliation/targeted.py,
  fused-memory/src/fused_memory/reconciliation/citation_verifier.py,
  fused-memory/src/fused_memory/reconciliation/stale_status_snapshot_edge_sweep.py,
  fused-memory/src/fused_memory/server/tools.py,
  fused-memory/scripts/measure_stale_status_snapshot_sweep.py,
  fused-memory/tests/test_task_filter.py, fused-memory/tests/test_recon_prose_grammar.py,
  fused-memory/tests/test_status_vocab_consolidation.py,
  fused-memory/tests/reconciliation/test_orphaned_recon_escalation_sweep.py
- Size note: 14 paths. The first four after `targeted.py` import `INACTIVE_TASK_STATUSES` from
  `task_filter`, and the last test imports `TERMINAL_TASK_STATUSES`; each becomes a one-line repoint onto
  `shared.task_statuses.TERMINAL` (AST scan at `97be1d71fa`).
- Deps: θ2 (the count-snapshot filter moved in θ2, the terminal-metadata filter in θ1). Task 9285 depends
  on δ2 (lead action; no cycle: δ2's ancestry is α, δ1, β, γ, θ1, θ2 and S19 α). S19 β2 depends on δ2 under
  lead ruling OQ-4.
- Keys: fk-51ec1e4010af, fk-9f66b6f4420c.
- Work: C3, D6. `status_rank` keeps today's ranks for the seven present members and ranks infra-hold 3
  and cancelled 7 (an explicit map keyed by `TaskStatus`, kept under the private name `_STATUS_PRIORITY`
  until θ3 repoints its one outside reader); a test pins the full order and fails when a `TaskStatus`
  member is unranked. `_select_visible_active_with_body` returns a frozen `VisibleActiveSelection`;
  `select_visible_active`, `render_active_section` and `format_filtered_task_tree` take
  `render_line: Callable[[dict], str] = render_task_line`, so tests pass a cheap renderer.
- DELETES: `_COUNT_GROUP_RE`'s word list, `TERMINAL_STATUSES`, `INACTIVE_TASK_STATUSES`,
  `TERMINAL_TASK_STATUSES` (and their five outside importers' copies of the name), the two `targeted.py`
  tuples, the 5-tuple, the four string patch targets into `task_filter`. `_STATUS_PRIORITY`'s hand-written
  partial map becomes the full ranked map behind `status_rank`.
- Observable signal: B6 passes in CI, driving `strip_snapshot_lines` (its returned dropped-line count)
  and the count-snapshot filter with one line per `TaskStatus` member; after deploy,
  `.venv/bin/python -c "from fused_memory.reconciliation.recon_prose_grammar import is_count_snapshot;
  print(is_count_snapshot('40 pending / 12 infra-hold'))"` in the main checkout prints `True` (today
  `False`).
- G7: INV-9, INV-5; the parity test becomes a behaviour test over every member (INV-10).

**ε — Make the recon finding category a closed enum: refuse off-list categories at add_finding, record every refusal for analysis, and generate the stage schema from it**
(priority medium, est. ~1,150 LOC: ~850 drafted plus the refusal record, the retry stamp, the stage stat
and their tests; the record's share is the sitting's estimate, not measured)
- Files: fused-memory/src/fused_memory/models/reconciliation.py,
  fused-memory/src/fused_memory/server/recon_report.py,
  fused-memory/src/fused_memory/server/recon_report_store.py,
  fused-memory/src/fused_memory/server/recon_lifecycle_filer.py,
  fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py,
  fused-memory/src/fused_memory/reconciliation/scope_freshness.py,
  fused-memory/src/fused_memory/reconciliation/flag_filters/stage3_findings.py,
  fused-memory/src/fused_memory/reconciliation/flag_filters/systemic_pattern.py,
  fused-memory/src/fused_memory/reconciliation/curator_gate_resolution_sweep.py,
  fused-memory/src/fused_memory/reconciliation/dangling_citation.py,
  fused-memory/src/fused_memory/reconciliation/orphaned_recon_escalation_sweep.py,
  fused-memory/src/fused_memory/reconciliation/unlanded_path_check.py,
  fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py,
  fused-memory/src/fused_memory/reconciliation/prompts/fragments.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage1.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage2.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage3.py,
  fused-memory/tests/test_finding_category_vocabulary.py (new),
  fused-memory/tests/test_stage3_cross_project_routing_actionable.py,
  fused-memory/tests/server/test_recon_report.py,
  fused-memory/tests/server/test_recon_report_rejected_findings.py (new)
- Size note: 21 paths, over the band, deliberately: the ruling puts the refusal record in this task
  ("the category task grows"), because a refusal that lands without its record loses exactly the evidence
  Leo asked to keep. The prompt edits replace hand-typed category literals with the enum; five are one-line
  producer-constant conversions to `FindingCategory` members (behaviour-neutral: a `StrEnum` member equals
  its string). `prompts/fragments.py` is created by S12 ι2 (S12/C7, the fragment library that renders the
  severity vocabulary today in `prompts/__init__.py`). The store tests go in a new module so ε shares no
  test file with task 9278.
- Deps: ζ (`recon_report.py`), θ3 (`stage3_findings.py`, `task_knowledge_sync.py`; through it θ1 for
  `systemic_pattern.py`, γ for `scope_freshness.py`, δ2 for `orphaned_recon_escalation_sweep.py`), S12 ι2
  (`prompts/fragments.py`; also through θ3).
- Keys: fk-7c13df0b988c.
- Work: C5, C11, D8, D18 (OQ-2 (a) plus Leo's additions). `add_finding` refuses an off-list category
  before truncation or dedup with `invalid_category` listing the nine values, writes one
  `rejected_findings` row through `ReconReportStore.record_rejection`, and adds one to the entry's
  `findings_rejected_invalid_category` stat; an accepted `add_finding` stamps `retried_as` through
  `ReconReportStore.stamp_retried_as`. The store gains the table, its two indexes, the prune-on-insert
  retention (180 days), the read-only `read_rejected_findings` reader and `RejectedFindingsTableMissing`;
  `delete_run` is unchanged and a test pins that it leaves the table alone. The add_finding guidance
  rendered by the fragment library lists the vocabulary for every stage. The producer category constants
  become members: `curator_gate_resolution_sweep.py::GATE_RESOLUTION_FLAG_CATEGORY`,
  `dangling_citation.py::_CATEGORY`, `orphaned_recon_escalation_sweep.py::ORPHANED_ESCALATION_FLAG_CATEGORY`,
  `unlanded_path_check.py::UNLANDED_PATH_FLAG_CATEGORY` and the `category='task_memory_mismatch'` literal
  in `server/recon_lifecycle_filer.py`.
- DELETES: the hand-written schema enum, `_CROSS_PROJECT_ROUTING_CATEGORY`, the
  `startswith('cross_project')` rule, the `'cross_project_routing'` and `'systemic_pattern'` category
  literals, `prompts/stage3.py`'s hand list, the five producer category literals.
- Observable signal: B7, B7b and B8 pass; after deploy the MCP `add_finding` tool answers an
  off-vocabulary category with `invalid_category`, a read-only query of `recon_report_state.db` over runs
  started after the deploy finds no accepted category outside `FINDING_CATEGORY_VALUES`, and each refusal
  in those runs has a `rejected_findings` row whose count per run and stage equals that stage report's
  `findings_rejected_invalid_category`. The draft's 20-runs-before-and-after findings comparison is
  replaced, as ruled, by the lost-finding condition of π's script, which ρ's chain runs and escalates.
- G7: INV-1 (the vocabulary is enforced at the boundary), INV-2 (structured refusal), INV-11 (a refusal
  is never silent: it is recorded and counted, and a failed record write is logged).

**ζ — Give recon_report one RunFindingIndex per run that owns its session and dedup maps**
(priority high, est. ~1,300 LOC)
- Files: fused-memory/src/fused_memory/server/recon_finding_index.py (new),
  fused-memory/src/fused_memory/server/recon_report.py,
  fused-memory/tests/server/test_recon_finding_index.py (new),
  fused-memory/tests/server/test_recon_report.py,
  fused-memory/tests/server/test_recon_report_store.py,
  fused-memory/tests/server/test_recon_report_citations.py,
  fused-memory/tests/server/test_recon_report_start_idempotence.py
- Deps: α.
- Keys: fk-0791e9817b84.
- Work: C4, D7. Every mutator (`start_report`, `add_finding`, `cite_task`, `_purge_finding`,
  `hydrate_from_store`, `tick`, `_persist_run`) calls the index; tests assert behaviour and
  `violations() == []` instead of reading `_run_*_index`.
- DELETES: the five run-scoped maps and their twelve writer sites, the per-entry mirror maintenance, the
  slice re-derivation in `_persist_run`, the 43-line `_reachable_run_entries`, the "fourth index" comment.
- Observable signal: B9 and B10 pass through the MCP tools in CI; after deploy, a fused-memory restart
  during a recon run produces no duplicate finding for the same `(task_id, flag_type)` in that run's
  assembled report (`get_assembled_report`).
- G7: INV-9 (one owner of the run's findings), INV-10 (the invariant check runs in tests).

**η — Give both stale-edge sweeps one Graphiti edge-sweep primitive, take the agent id from StageId, and stop recon writing task status into memory (the sweep's replacement sentence and the Stage-1 prompt's status facts)**
(priority medium, est. ~1,050 LOC: ~950 drafted plus the Stage-1 prompt's two status sections)
- Files: fused-memory/src/fused_memory/reconciliation/graph_edge_sweep.py (new),
  fused-memory/src/fused_memory/reconciliation/stale_status_snapshot_edge_sweep.py,
  fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py,
  fused-memory/src/fused_memory/reconciliation/citation_verifier.py,
  fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage1.py,
  fused-memory/scripts/measure_stale_status_snapshot_sweep.py,
  docs/stale-status-snapshot-sweep.md,
  fused-memory/tests/reconciliation/test_graph_edge_sweep.py (new),
  fused-memory/tests/reconciliation/test_stale_status_snapshot_edge_sweep.py,
  fused-memory/tests/reconciliation/test_stale_priority_override_edge_sweep.py,
  fused-memory/tests/reconciliation/test_stage1.py,
  fused-memory/tests/test_stages.py,
  fused-memory/tests/reconciliation/test_stage1_no_status_writes.py (new)
- Size note: 14 paths. `tests/test_stages.py` pins the consolidator's stage stats, including the three
  `stale_blocked_edges_supersede*` / `_superseded` keys η deletes (measured at `a85cb1ad78`); the new test
  module is B11b.
- Deps: κ2 (`prompts/stage1.py`), δ2 (`citation_verifier.py`, `stale_status_snapshot_edge_sweep.py`; also
  through κ2); out of batch: 9203 (S26 μ, C7), 6721 (S14 μ), 6718 (S14 θ, the priority sweep and its
  test), 7941, 5272. (6598 is done.)
- Keys: fk-917fdf32a542, fk-cdb8c20a3d0f.
- Work: C6, D9 (including the stated behaviour change on an empirical incompleteness) and D19 (OQ-1 (b) and
  OQ-8 (b′), ruled by Leo). `test_stage1.py` imports the four `*_ENUMERATION_*_STAT_KEY` constants and the
  measurement script imports `flatten_dedup_edges`; both are repointed. The sweep invalidates and writes
  nothing; the Stage-1 prompt's paragraph describing what the sweep has "already done for you" becomes:
  "the sweep invalidates stale blocked-status snapshot edges and writes no replacement; the task store
  answers status; do not write a replacement status fact for an edge the sweep retired." The
  "Resulting-state-only temporal_facts" section and the "update_edge Temporal Limitation" section (with its
  note line, its scope carve-out, and the "Encoding effective dates" paragraph, whose list of snapshot kinds
  loses "task-count" and "task-status" and whose task-count example becomes a non-task snapshot, D19) say
  that for a task's status the agent invalidates the stale edge and writes nothing, while every other
  temporal fact keeps the two-step; the residual paragraph's "fix it AND flag it" becomes "invalidate it AND flag it".
  `docs/stale-status-snapshot-sweep.md` drops its "superseding write" section and says why. The detector
  and the selection rule stay (D19); nothing about the recall design session's questions is built here.
- DELETES: the priority sweep's copied skeleton, `_SWEEP_AGENT_ID`, the literal agent id in
  `citation_verifier`, the two `*_ENUMERATION_*_STAT_KEY` pairs, `build_supersede_fact`,
  `_SUPERSEDE_FACT_TEMPLATE`, `_MAX_SUPERSEDE_WRITES_PER_CYCLE`, the three supersede stats, and the Stage-1
  prompt's instructions to write a status fact (the status-flip clause of the two-step, the sweep's
  "writes the superseding resulting-state-only fact itself" sentence).
- Observable signal: B11 and B11b pass; after deploy the next dark_factory Stage-1 report carries the sweep
  stats from `EdgeSweepStats.to_stats` for both sweeps and no `stale_blocked_edges_superseded` key, and
  read-only sqlite on `write_journal.db` finds no `write_ops` row with `operation='add_memory'`,
  `agent_id='recon-stage-memory_consolidator'` and params containing `has status` after the deploy (30
  such rows since 2026-10-08, measured 2026-11-12, for scale).
- G7: INV-11 (an incomplete read never becomes a destructive write, stated in one place), INV-3 (selection
  still corroborates against live status), INV-5, INV-9 (the task store is task status's one home; recon
  stops writing a copy).

**θ1 — Move flag_dedup's task-state refutation and systemic-pattern filters, with their shared lookup helpers, into flag_filters modules**
(priority medium, est. ~1,400 LOC, verbatim move)
- Files: fused-memory/src/fused_memory/reconciliation/flag_filters/__init__.py (new),
  fused-memory/src/fused_memory/reconciliation/flag_filters/task_state_refutations.py (new),
  fused-memory/src/fused_memory/reconciliation/flag_filters/systemic_pattern.py (new),
  fused-memory/src/fused_memory/reconciliation/tracked_task_terms.py (new),
  fused-memory/src/fused_memory/reconciliation/task_corroboration.py,
  fused-memory/src/fused_memory/reconciliation/flag_dedup.py,
  fused-memory/src/fused_memory/reconciliation/stage1_flag_chain.py,
  fused-memory/tests/test_flag_dedup.py
- Deps: γ (`flag_dedup.py`). Touches no file S12 edits. Subsumes task 5996 with θ2 and θ3 (cancelled).
- Keys: fk-46514b4ef0cc.
- Work: D4 for `filter_terminal_metadata_flags`, `filter_stale_bulk_get_statuses_flags`,
  `filter_false_absence_flags`, `filter_already_tracked_systemic_patterns` and their private helpers,
  with `TERMINAL_STATUSES` (read only by `filter_terminal_metadata_flags`; δ2 then replaces it in
  `task_state_refutations.py`). The phantom-creation filter (imported by `task_knowledge_sync.py`) waits for θ3, so θ1 touches no S12
  file.
- DELETES: those filters and helpers from `flag_dedup.py` (~1,000 lines).
- Observable signal: `stage1_flag_chain.py`'s two `ResolvingFilter` tuples import from
  `flag_filters.*`; after deploy the next dark_factory Stage-1 report shows each moved filter's drop stat
  key present (value may be 0).
- G7: INV-5; heuristics 6, 13, 14.

**θ2 — Move flag_dedup's count-snapshot, authorship and task-text accounting filters into flag_filters modules**
(priority medium, est. ~1,000 LOC, verbatim move)
- Files: fused-memory/src/fused_memory/reconciliation/flag_filters/count_snapshot.py (new),
  fused-memory/src/fused_memory/reconciliation/flag_filters/authorship.py (new),
  fused-memory/src/fused_memory/reconciliation/flag_filters/task_text_accounting.py (new),
  fused-memory/src/fused_memory/reconciliation/flag_dedup.py,
  fused-memory/src/fused_memory/reconciliation/stage1_flag_chain.py,
  fused-memory/tests/test_flag_dedup.py
- Deps: θ1. Touches no file S12 edits.
- Keys: fk-46514b4ef0cc.
- Work: D4 for `filter_stale_count_snapshot_corrections`, `filter_style_only_authorship_flags`,
  `filter_accounted_cluster_growth_flags` and `filter_already_recorded_caveat_flags` (with their shared
  `_TaskTextAccountingGate` core).
- DELETES: those filters from `flag_dedup.py` (~850 lines).
- Observable signal: as θ1, for the moved filters.
- G7: INV-5; heuristics 6, 13, 14.

**θ3 — Move the phantom-creation filter, Hook A and the Stage-3 finding guards out of flag_dedup, leave it the ledger lifecycle, and rehome its prose**
(priority medium, est. ~1,350 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/flag_filters/task_state_refutations.py,
  fused-memory/src/fused_memory/reconciliation/flag_filters/entity_standing_decision.py (new),
  fused-memory/src/fused_memory/reconciliation/flag_filters/stage3_findings.py (new),
  fused-memory/src/fused_memory/reconciliation/flag_dedup.py,
  fused-memory/src/fused_memory/reconciliation/stage1_flag_chain.py,
  fused-memory/src/fused_memory/reconciliation/standing_decision_storm_escape.py,
  fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py,
  fused-memory/src/fused_memory/reconciliation/stages/integrity_check.py,
  fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py,
  docs/recon-flag-dedup.md (new),
  fused-memory/tests/test_flag_dedup.py,
  fused-memory/tests/test_fused_memory_governed_lists.py,
  fused-memory/tests/reconciliation/test_stage1_flag_chain.py,
  fused-memory/tests/test_standing_decision_storm_escape.py
- Size note: 14 paths; the last two tests import `EntityStandingSuppressionResult`, which moves.
- Deps: θ2, δ2 (`task_state_refutations.py`; `task_filter.status_rank`), ι (`stages/integrity_check.py`
  exists), S12 ι2 (S12's last leaf on `task_knowledge_sync.py` and `memory_consolidator.py`; lead ruling OQ-6).
- Keys: fk-46514b4ef0cc, fk-2868338d3cf3.
- Work: D4 for `filter_false_phantom_task_creation_flags` (into `task_state_refutations.py`),
  `filter_entity_standing_decisions` with `EntityStandingSuppressionResult` (Hook A, imported by
  `stage1_flag_chain`, `standing_decision_storm_escape` and `memory_consolidator`), and
  `filter_blocked_snapshot_findings`, `filter_contamination_ceiling_findings`,
  `filter_snapshot_stat_edge_findings` with their predicates; `flag_dedup.py`'s module docstring becomes
  its one-sentence purpose plus the ledger invariants; the 344-line narrative (Mem0-era compensations,
  incident history) moves to `docs/recon-flag-dedup.md` or the commit message (D16). Repoint
  `task_knowledge_sync.py::_select_proactive_sample`'s function-local `_STATUS_PRIORITY` import to a
  module-top import of `task_filter.status_rank` (D6).
- DELETES: those filters and ~320 docstring lines from `flag_dedup.py`; the last private import of
  `_STATUS_PRIORITY` outside `task_filter`.
- Observable signal: `flag_dedup.py` defines no `filter_*` function other than `filter_suppressed`
  (`git grep -nE "^(async )?def filter_" fused-memory/src/fused_memory/reconciliation/flag_dedup.py`
  prints one line), its module docstring opens with a one-sentence purpose naming the ledger lifecycle and
  carries no task-number narrative; the leaf records the before/after prose share of the four recon core
  files in its completion note (a report, not a gate — the Comments stance, not a line-count target).
  After deploy the next dark_factory Stage-3 report still applies the three moved guards (their stats keys
  present).
- G7: Comments stance; INV-5.

**ι — Split task_knowledge_sync.py: IntegrityCheck, the shared report formatters, snapshot I/O, the growth sweep and the reify briefing hook move to their own modules**
(priority medium, est. ~3,000 LOC counted both sides, verbatim move of ~1,460 lines)
- Files: fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py,
  fused-memory/src/fused_memory/reconciliation/stages/integrity_check.py (new),
  fused-memory/src/fused_memory/reconciliation/stage_report_format.py (new),
  fused-memory/src/fused_memory/reconciliation/task_count_snapshot_store.py (new),
  fused-memory/src/fused_memory/reconciliation/standing_decision_growth_sweep.py (new),
  fused-memory/src/fused_memory/reconciliation/policies/briefing_refresh.py (new),
  fused-memory/src/fused_memory/reconciliation/harness.py,
  scripts/exception_register.py, shared/tests/loop_blocking_allowlist.py,
  fused-memory/tests/test_stages.py,
  fused-memory/tests/reconciliation/test_standing_decision_growth_sweep.py,
  fused-memory/tests/reconciliation/test_task_knowledge_sync_citation_verify.py,
  fused-memory/tests/reconciliation/test_task_count_snapshot_cadence.py,
  fused-memory/tests/test_refresh_entity_summary.py
- Size note: 14 paths; `test_task_count_snapshot_cadence.py` (3 imports and 10 dotted-path patches of the
  snapshot functions) and `test_refresh_entity_summary.py` (imports `_format_report`) are repoints of the
  moved names to their new public homes.
- Deps: α, 9203 (the growth sweep's graphiti reach repointed in place first). No S12 dependency
  (lead ruling OQ-6): S12's stage test builder, stage-interface, streak-alarm and task-write tasks depend on ι
  instead. No edge onto 9180: 9180's edits there (`assemble_payload`, `_needs_hint_attention`) are in no
  cluster ι moves; an edge would only make ι itself wait on 9180's 22 live ancestors.
- Keys: fk-b1cda781c60e. Subsumes task 6846 (cancelled).
- Work: D10. Moved functions take public names in their new modules; the briefing hook gains the
  `briefing_gap_source` keyword, and the 5 tests that patch `_run_briefing_known_gaps_script` call
  `run_briefing_refresh` with a fake source instead; the snapshot tests' patches are repointed to the
  public names (ν then moves them onto `TaskCountSnapshotStore`); re-key the moved symbols'
  `loop_blocking_allowlist` rows and the exception-register `ListRef`. Re-measure the clusters at plan
  time (the 2026-11-11 `_format_flagged` edit of task 6509 is inside the formatter cluster).
- DELETES: five clusters (~1,460 lines measured at `1f240e6da2`) and the `prompts.stage3` function-local
  import from `task_knowledge_sync.py`.
- Observable signal: `harness.py` imports `IntegrityCheck` from `stages/integrity_check.py`; after deploy
  a full dark_factory cycle completes with Stage 3's report present in the run journal and the
  `task_count_snapshot_written` stat unchanged from the previous cycle.
- G7: heuristics 6, 13, 14; INV-5.

**κ1 — Delete Stage 2's dormant flag-persistence and escalation markers and the dormant human-operator stall marker; give recon marker GC its own module**
(priority high, est. ~700 LOC, mostly deletions)
- Files: fused-memory/src/fused_memory/reconciliation/marker_gc.py (new),
  fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py,
  fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py,
  fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py,
  fused-memory/src/fused_memory/reconciliation/orphaned_recon_escalation_sweep.py,
  fused-memory/src/fused_memory/reconciliation/recon_ledger.py,
  fused-memory/src/fused_memory/reconciliation/recon_self_model.py,
  fused-memory/src/fused_memory/reconciliation/mem0_tombstone.py,
  fused-memory/src/fused_memory/reconciliation/summary_pool.py,
  fused-memory/src/fused_memory/reconciliation/targeted.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage2.py,
  fused-memory/tests/reconciliation/test_marker_gc.py (new),
  fused-memory/tests/test_stages.py, fused-memory/tests/test_recon_self_model.py,
  fused-memory/tests/test_recon_reliability_integration.py,
  fused-memory/tests/reconciliation/test_stage1_stall_detector.py
- Size note: 16 paths. `orphaned_recon_escalation_sweep.py`, `mem0_tombstone.py`, `summary_pool.py` and
  `targeted.py` change only prose that names the deleted stall filer, the deleted persistence sweep or the
  renamed GC function; `test_recon_reliability_integration.py` imports `_gc_recon_markers`.
- Deps: ε (`task_knowledge_sync.py`, `prompts/stage2.py`; through it θ3 for `memory_consolidator.py`, β for
  `recon_ledger.py`, δ2 for `orphaned_recon_escalation_sweep.py` and `targeted.py`); S19 β1 (lead ruling OQ-4)
  (it moves `recon_self_model.py`, after S19 α moves `recon_ledger.py`; the clerk repaths this file list).
  Subsumes task 7807 (cancelled).
- Keys: fk-9165ae3a1e72.
- Work: D11 and C7 for the FIX D half and the stall marker, under OQ-7 (b) as ruled by Leo. `_gc_recon_markers` moves to
  `marker_gc.py` as `gc_recon_markers`; `task_knowledge_sync.py` imports it, and its 8 test patch sites
  drive it with a real ledger. Delete `_track_flag_persistence`, `_compute_stale_flags`,
  `_write_escalation_markers`, `_filter_already_escalated_flags`, `_sweep_stale_persistence_markers` and
  its deleter source constant, the "Stale Flags Requiring Escalation" payload section and its Stage-2
  prompt directive; `stage2_persistence_marker` leaves both `MARKER_KINDS` tuples, `MARKER_LIFECYCLE` and
  `MEM0_TOMBSTONE_DELETERS`. Delete the human-operator stall path: `stage1_stall_detector.py`'s
  `track_human_operator_stalls`, `compute_stalled_task_ids`, `maybe_escalate_stalled_tasks`,
  `extract_human_operator_task_ids`, `STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD`,
  `_STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE`, and `MemoryConsolidator._escalate_human_operator_stalls`.
  The gate-backlog escalation in the same module stays. The escalation category
  `reconciliation_stale_human_operator` stays in `escalation/src/escalation/categories.py`, and the
  orphaned-escalation sweep keeps closing records of that category already on the queue; its prose stops
  naming a live filer.
- DELETES: FIX D (four functions, the persistence sweep, one payload section, one prompt directive), the
  human-operator stall marker and its escalation (six names and one stage method), the
  `stage2_persistence_marker` kind.
- Observable signal: B13 passes; after deploy the next dark_factory Stage-2 report carries
  `recon_markers_gc_swept` and none of `stale_persistence_markers_gc_swept`, `stale_flags_escalated`,
  and the Stage-1 report carries `stage1_gate_backlog_escalated` (the stall module's surviving escalation)
  as before.
- G7: INV-9 (no control-plane marker outside the ledger); heuristic 6 (machinery with no traffic and no
  purpose goes); the two `MARKER_KINDS` tuples change in one commit (INV-5's cross-check test stays
  green).

**κ2 — Delete the dormant Stage-1→Stage-2 Mem0 flag relay and the Mem0 marker-pool sweeper**
(priority high, est. ~800 LOC, mostly deletions)
- Files: fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py,
  fused-memory/src/fused_memory/reconciliation/recon_ledger.py,
  fused-memory/src/fused_memory/reconciliation/recon_self_model.py,
  fused-memory/src/fused_memory/reconciliation/mem0_tombstone.py,
  fused-memory/src/fused_memory/reconciliation/summary_pool.py,
  fused-memory/src/fused_memory/reconciliation/targeted.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage1.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage2.py,
  fused-memory/tests/test_stages.py,
  fused-memory/tests/test_summary_pool.py,
  fused-memory/tests/test_recon_self_model.py,
  fused-memory/tests/test_targeted.py,
  fused-memory/tests/test_mem0_tombstone.py,
  fused-memory/tests/test_recon_marker_kind_prompt_guidance.py,
  fused-memory/tests/reconciliation/test_stage2_recon_report_channel.py,
  fused-memory/tests/reconciliation/test_stage2_flag_carry_forward.py (deleted),
  fused-memory/tests/_flag_for_stage2_pool_fake.py (deleted)
- Size note: 17 paths, almost all deletions: the relay's reader, sweeper and their tests leave together so
  no cycle runs with a half-removed channel. `test_summary_pool.py` imports
  `_sweep_stale_mem0_flag_markers`; `test_recon_self_model.py` reaches `_sweep_stale_mem0_pool`;
  `test_recon_marker_kind_prompt_guidance.py` pins the Stage-1 prompt's relay text.
- Deps: κ1.
- Keys: fk-9165ae3a1e72, fk-fa1fd33c7a9e.
- Work: D11, D12 and C7 for the relay half, under OQ-7 (b) as ruled by Leo. Delete `_query_stage2_flags`,
  `Stage2FlagPartition` and its carry-forward, FIX C (the Stage-2 prompt section and
  `_acknowledge_resolved_stage1_markers`), `_sweep_stale_mem0_pool` and its callers
  `_sweep_stale_mem0_flag_markers` and `_sweep_stale_mem0_flag_for_stage2_markers`,
  `_retire_flag_for_stage2_members`, `retire_flag_markers_for_terminal_task` with
  `targeted.py`'s latency-layer call to it (`TargetedReconciler._on_task_done`), `_warn_on_flag_for_stage2_type_drift`,
  `summary_pool.py::trailing_incomplete_agent_run_ids` (no other production caller), the Stage-1 prompt's
  relay instructions and the Stage-2 "Persistent Flags" directive. `flag_for_stage2` leaves both
  `MARKER_KINDS` tuples and `MARKER_LIFECYCLE`; `stage1_flag_marker_gc_sweep` and `flag_for_stage2_gc_sweep`
  leave `MEM0_TOMBSTONE_DELETERS`. Stage 2 keeps `_query_recon_report_findings` unchanged.
- DELETES: `_sweep_stale_mem0_pool` (679 lines) and every Mem0 relay read, write prompt and sweep; the
  pool fake and the carry-forward test module.
- Observable signal: B13 passes; after deploy the dark_factory Stage-2 report carries
  `stage1_analytical_findings_processed` and none of `stage1_mem0_flags_processed`,
  `stage2_markers_carried_forward`, `stale_mem0_flag_markers_gc_swept`,
  `stale_mem0_flag_for_stage2_markers_gc_swept`; and `stage1_analytical_findings_processed` summed over
  the 20 completed dark_factory runs after the deploy is not below 80% of the 20 before it (no handed-over
  finding was riding the deleted channel).
- G7: INV-11 (the sweeper's silent zero goes with it), INV-9 (one Stage-1→Stage-2 channel).

**κ3 — Refuse recon-stage writes to the retired Stage-1→Stage-2 Mem0 relay and retire the relay from guidance, vocabulary and operator scripts**
(priority high, est. ~450 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/flag_record_contract.py,
  fused-memory/src/fused_memory/reconciliation/consolidation_gate.py,
  fused-memory/src/fused_memory/reconciliation/prompts/fragments.py (created by S12 ι2),
  fused-memory/src/fused_memory/memory_metadata.py,
  fused-memory/scripts/sweep_orphan_flag_markers.py,
  fused-memory/scripts/freeze_write_triage_population.py,
  scripts/fused-memory-flag-marker-check.sh,
  docs/flag-marker-sweep-recurring.md,
  fused-memory/tests/server/test_add_memory_flag_record_schema_error.py,
  fused-memory/tests/server/test_add_memory_write_triage_gate.py,
  fused-memory/tests/reconciliation/test_flag_record_contract.py,
  fused-memory/tests/test_operational_routing_boundary_matrix.py,
  fused-memory/tests/test_sweep_orphan_flag_markers.py,
  fused-memory/tests/test_freeze_write_triage_population.py
- Size note: 14 paths; the scripts and the doc change only what they say about the relay (the
  `{'flag_for_stage2': True}` census probe and the deleted in-cycle collector they name).
- Deps: κ2 (the Stage-1 prompt stops asking for relay writes before the boundary refuses them); S19 β2
  (lead ruling OQ-4; it moves `flag_record_contract.py` and `consolidation_gate.py`; the clerk repaths this
  file list).
- Keys: fk-9165ae3a1e72.
- Work: C7's refusal (recon-stage writers only; task 4375's non-relay writers and an explicit `False` are
  unaffected). `consolidation_gate.py`'s gate hint stops telling agents to relay "via the
  `flag_for_stage2` / `flagged_items` channel" and names `add_finding`; the boundary-matrix test pins the
  new text. `prompts/fragments.py::FLAG_FOR_STAGE2_MARKER_KIND` (S12 ι2 moved the fragment library there)
  goes (no prompt writes that kind after κ2). Its three other importers, all in these Files, then take the
  kind name from wherever tactical question 3 puts it: `fused-memory/scripts/freeze_write_triage_population.py`
  (its `_RECON_MARKER_KINDS`) and `tests/test_freeze_write_triage_population.py` import it from
  `memory_metadata.py` if κ3 keeps it there, else the script holds `'flag_for_stage2'` as a data literal
  beside `_RECON_MARKER_FLAGS` and the test uses that literal; `tests/server/test_add_memory_write_triage_gate.py`
  drops its relay-marker-kind case (a recon-stage writer can no longer write that kind) or, if the case is
  kept for non-recon writers, imports the name from the same place.
  `memory_metadata.py`'s `flag_for_stage2` vocabulary entries say the key is refused for recon-stage
  writers; `freeze_write_triage_population.py` keeps excluding existing rows that carry the key. The
  operator scripts drop the `{'flag_for_stage2': True}` census probe and stop naming the deleted in-cycle
  collector. The Mem0 residue is read once (1 row in dark_factory, 7 in reify at `97be1d71fa`): rows a
  `recon-stage-*` agent wrote are deleted with a tombstone through
  `mem0_tombstone.record_mem0_deletion_tombstones`; rows another writer stamped are left; the read and its
  outcome go in the completion note.
- DELETES: the relay guidance in the consolidation gate, `FLAG_FOR_STAGE2_MARKER_KIND`, the scripts'
  relay census, the recon-written residue rows.
- Observable signal: B12 passes; after deploy, `add_memory` through MCP from a `recon-stage-*` agent with
  `flag_for_stage2: true` answers `flag_for_stage2_relay_retired`, and read-only sqlite on
  `write_journal.db` finds no `add_memory` by a `recon-stage-*` agent with `flag_for_stage2` true in its
  params after the deploy.
- G7: INV-1 (the boundary refuses a retired channel by a structured error), INV-2, INV-9.

**ν — Make TaskKnowledgeSync tests drive public seams: injected collaborators, no private imports or dotted-path patches**
(priority high, est. ~1,000 LOC, mostly tests)
- Files: fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py,
  fused-memory/src/fused_memory/reconciliation/task_count_snapshot_store.py,
  fused-memory/tests/test_stages.py, fused-memory/tests/test_harness.py,
  fused-memory/tests/test_memory_service.py,
  fused-memory/tests/test_recon_self_model.py,
  fused-memory/tests/reconciliation/test_stage2_recon_report_channel.py,
  fused-memory/tests/reconciliation/test_standing_decision_growth_sweep.py,
  fused-memory/tests/reconciliation/test_task_count_snapshot_cadence.py,
  fused-memory/tests/reconciliation/test_task_knowledge_sync_citation_verify.py,
  fused-memory/tests/test_stage2_tasks_created_accounting.py,
  fused-memory/tests/test_stage2_stats_audit.py,
  fused-memory/tests/reconciliation/test_tks_public_seam_guard.py (new)
- Size note: 13 paths, all but two tests; `test_memory_service.py` imports
  `_STAGE2_CYCLE_SUMMARY_RECON_POOL` and `test_recon_self_model.py` reaches `_STAGE2_CYCLE_SUMMARY_TRIM_SOURCE`
  as an attribute.
- Deps: κ2 (`task_knowledge_sync.py`); S12 γ2 (S12/C2 `StageCollaborators`; both edit
  `task_knowledge_sync.py`), out of batch, wired at decompose. No edge onto S12 θ: the two share only the
  test files `test_stages.py`, `test_harness.py` and `test_task_count_snapshot_cadence.py`, which the
  lead's test-file rule leaves to the file locks. Subsumes task 7065 (cancelled).
- Keys: fk-90f374fcce21.
- Work: D13 and its table. `TaskCountSnapshotStore` wraps the snapshot functions ι moved and enters
  `TaskKnowledgeSync` as the `snapshot_store` keyword; the remaining private names the tests need become
  public. The guard covers all four reach forms (B16). Re-measure the reach at plan time (at
  `97be1d71fa`: 42 names, 139 imports, 57 string patch targets in 2 files, 16 attribute reaches in 3).
- DELETES: every underscore import from, dotted-path patch into, and attribute reach on
  `task_knowledge_sync` in tests.
- Observable signal: B16 passes (the guard flags seeded test modules and passes the tree); `git grep -E
  "stages\.task_knowledge_sync\._|task_knowledge_sync import \(?_" fused-memory/tests` is empty.
- G7: Tests stance; INV-10.

**λ — Move per-project recon policy (snapshot-write block, contamination-ceiling retirement, briefing refresh) into config keyed by project id**
(priority low, est. ~700 LOC)
- Files: fused-memory/src/fused_memory/config/schema.py, fused-memory/src/fused_memory/config/reload.py,
  fused-memory/config/config.yaml,
  fused-memory/src/fused_memory/reconciliation/policies/__init__.py,
  fused-memory/src/fused_memory/reconciliation/policies/autopilot_video.py (deleted),
  fused-memory/src/fused_memory/reconciliation/policies/dark_factory.py (deleted),
  fused-memory/src/fused_memory/reconciliation/policies/know_live.py (deleted),
  fused-memory/src/fused_memory/reconciliation/policies/solar_challenge_platform.py (deleted),
  fused-memory/src/fused_memory/reconciliation/policies/briefing_refresh.py,
  fused-memory/src/fused_memory/reconciliation/harness.py,
  fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py,
  fused-memory/src/fused_memory/reconciliation/stages/integrity_check.py,
  fused-memory/src/fused_memory/reconciliation/task_count_snapshot_cadence.py,
  fused-memory/src/fused_memory/reconciliation/flag_filters/stage3_findings.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage2.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage3.py,
  scripts/exception_register.py,
  fused-memory/tests/reconciliation/test_policies.py,
  fused-memory/tests/test_config_reload.py,
  fused-memory/tests/test_stage2_narrative_reconstruction_retired.py,
  fused-memory/tests/test_stages.py,
  fused-memory/tests/_recon_prompt_write_scope.py,
  fused-memory/tests/test_recon_report_guidance_drift.py,
  fused-memory/tests/test_stage2_tasks_created_accounting.py,
  fused-memory/tests/test_graphiti_degradation_probe.py,
  fused-memory/tests/test_recon_amend_tool_advertisement.py,
  fused-memory/tests/test_recon_report_stage_gated_tool_advertisement.py,
  fused-memory/tests/test_duplicate_finding_salvage_guidance.py,
  fused-memory/tests/test_finding_provenance_prompt_guidance.py,
  fused-memory/tests/test_recon_gate_closure_guidance.py,
  fused-memory/tests/test_recon_gate_owned_action_phrasing_guidance.py
- Size note: 31 paths; four are deletions of per-project modules,
  `test_stage2_narrative_reconstruction_retired.py` imports `policies.autopilot_video`,
  `stages/integrity_check.py` renders the two per-cycle policy notes into the Stage-3 payload, and the last
  eleven test modules pass `policy=ReconProjectPolicy(stage2_guardrail='autopilot_video_contamination')`
  where their `'autopilot_video'` case exists for the guardrail splice (`test_stages.py`'s
  `TestStage2GuardrailProjectIdGating` among them). At `c9ab19766a`, 25 test modules call
  `build_stage2_system_prompt` and 16 exercise `'autopilot_video'`; re-measure at plan time.
  `test_stages.py` is already ordered: λ → ν.
- Deps: ν (`task_knowledge_sync.py`), θ3 (`stage3_findings.py`), κ2 (`prompts/stage2.py`), ι
  (`policies/briefing_refresh.py`), ε (`prompts/stage3.py`).
- Keys: fk-fbbf580e4e1d.
- Work: C9 and its read seam, D14. The guardrail prompt text moves into `prompts/stage2.py` as a constant
  named by the id; `STAGE3_SYSTEM_PROMPT` stays a constant that names no project, `IntegrityCheck`'s payload
  renders this project's two policy notes each cycle, and `build_stage2_system_prompt` gains the
  keyword-only `policy` argument (default `ReconProjectPolicy()`, no guardrail; `TaskKnowledgeSync.get_system_prompt`
  passes `policy_for(self.project_id)`) (no importer of `STAGE3_SYSTEM_PROMPT` changes); `reconciliation.project_policies` joins
  `RELOADABLE_FIELDS` with a live-consumer test in `test_config_reload.py`. Re-measure every caller of
  `is_snapshot_write_blocked` / `is_contamination_ceiling_retired` at plan time, after S12 ε. If the cadence
  check now lives in `reconciliation/run_streaks.py` (created by S12 ε), add it to Files and pass the policy
  through that module's streak evaluation, not a process-global.
- DELETES: four per-project modules, the two aggregated sets, `_BRIEFING_REFRESH_PROJECT_ALLOWLIST` and
  its exception-register entry, the hard-coded project compare, the import-time f-string interpolation of
  the project sets.
- Observable signal: B15 passes; after deploy, the Stage-3 payload for autopilot_video carries the
  write-blocked note and the retired-ceiling note, driven by `fused-memory/config/config.yaml`; the Stage-3
  system prompt names no project.
- G7: INV-9 (one mechanism for the per-project axis); INV-12 (an exception entry removed, none added).

**μ — Run EventJournal writes on one writer thread; EventQueue.enqueue awaits the durable append off the event loop**
(priority medium, est. ~600 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/event_journal.py,
  fused-memory/src/fused_memory/reconciliation/event_queue.py,
  fused-memory/src/fused_memory/middleware/task_interceptor.py,
  fused-memory/tests/test_event_journal.py, fused-memory/tests/test_event_queue.py,
  fused-memory/tests/_dead_letter_helpers.py, fused-memory/tests/test_queue_stats_tool.py,
  fused-memory/tests/test_sqlite_watchdog.py,
  fused-memory/tests/test_event_queue_loop_occupancy.py (new)
- Deps: S19 α (lead ruling OQ-4; it moves `event_journal.py` and `event_queue.py`; the clerk repaths this
  file list); nothing in the batch. Re-measure `EventQueue.enqueue` call sites at plan time (the
  interceptor's `_journal` and the replay path in `src/`; about 40 test sites).
- Keys: fk-4c8d19c64d99.
- Work: C8, D15 (lead ruling OQ-5 (a)), including the replay path (`replay_dead_letters` off the loop thread only;
  `_resolve_replay_enqueue`'s inline branch deleted; `_enqueue_on_loop` awaits `enqueue`); the three stale
  docstring claims are corrected.
- DELETES: the loop-thread sqlite calls on the enqueue, drain and replay paths; the on-loop replay branch.
- Observable signal: B14 and B14b pass in CI; after deploy, the fused-memory journal shows the writer
  thread name on journal commits (a debug log line per commit batch) and none from the loop thread, and a
  `replay_event_dead_letters` MCP call on an empty dead-letter file returns `replayed: 0` without error.
- G7: INV-8 (loop-thread occupancy bounded: worst case is now one queue hand-off); INV-11 (a replay never
  counts an event it did not enqueue).

**ξ — The recon-report server takes the run id from the stage connection, refuses a differing run_id argument, and the stage prompts stop asking for run_id**
(priority medium, est. ~950 LOC; unconditional under the lead's S12/OQ-10 (a), with Leo's S12-2 refusal rule)
- Files: fused-memory/src/fused_memory/server/recon_report.py,
  fused-memory/src/fused_memory/server/tool_context.py (S12/C3's module, created by S15 α),
  fused-memory/src/fused_memory/reconciliation/prompts/fragments.py (created by S12 ι2),
  fused-memory/src/fused_memory/reconciliation/prompts/stage1.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage2.py,
  fused-memory/src/fused_memory/reconciliation/prompts/stage3.py,
  fused-memory/src/fused_memory/reconciliation/stages/base.py,
  fused-memory/tests/server/test_recon_report.py,
  fused-memory/tests/server/test_recon_report_header_identity.py (new),
  fused-memory/tests/test_recon_report_guidance_drift.py,
  fused-memory/tests/test_stages.py
- Size note: 11 paths; re-measure the Files at plan time. The recon-report guidance lives in the fragment
  library S12 ι2 moves (S12/C7), and the `recon_context` paragraph is rendered in `stages/base.py`.
  Hand-written recon-report call examples carrying `run_id=<from Reconciliation Context>`, re-measured at
  `a85cb1ad78` (the fold corrects the draft's 5/5/6/2, which also counted fused-memory calls): 4 in
  `prompts/stage2.py`, 4 in `stage3.py`, 1 in the guardrail λ moves into `stage2.py` (today
  `policies/autopilot_video.py`), and in the fragment library 1 hand-written example
  (`repair_memory_citation`) plus the renderer's `'run_id': '<from Reconciliation Context>'` placeholder;
  `prompts/stage1.py` has none. The other `run_id=<…>` sites in the prompts belong to fused-memory calls
  (`get_cycle_summary_presence`, `propose_consolidation`, `search` filters) and marker metadata, and keep
  their `run_id`; so do `repair_memory_citation`'s `target_run_id` and `cite_run`'s `cited_run_id`, which
  name another run. `prompts/stage1.py` changes only its "Verifying add_finding responses" list, which
  gains the two new refusals. `tests/test_stages.py::test_recon_context_includes_run_id` pins `run_id` in
  the `recon_context` paragraph. `server/tool_context.py` takes one line (D20).
- Deps: ζ and ε (`recon_report.py`), κ3 (`prompts/fragments.py`), η (`prompts/stage1.py`), λ
  (`prompts/stage2.py`, `stage3.py`); out of batch, S12 δ (S12/C3: the headers on the recon-report
  connection, `stage_identity_refusal`, `StageIdentityRefusal`, the `recon_stage_identity_refused` log
  line; `stages/base.py`), and through δ S15 α (`server/tool_context.py`). Wired at decompose.
- Keys: none of S13's; it is the reader S12/C3 names for the recon-report connection (S12 δ carries S12's key
  fk-a3598aec7dca).
- Work: C10, D20. In every recon-report tool whose signature takes `run_id` (12 of 13 at `a85cb1ad78`),
  `run_id` moves to the end of the parameter list as `run_id: str | None = None` (MCP calls bind by name, so
  no caller depends on its position); the tool gains an injected `ctx` and resolves the run through
  `_report_run_id`: the header's run for an absent or equal argument, a refusal for a differing one
  (decided by S12/C3's `stage_identity_refusal(ctx, agent_id=None, causation_id=run_id)`, re-labelled with
  `dataclasses.replace(refusal, field='run_id')` and returned as `StageIdentityRefusal.as_error()`, one
  `recon_stage_identity_refused` log line,
  nothing read or written), the argument when no header is present, and `run_id_missing` when neither
  supplies it. `StageIdentityRefusal.field`'s literal gains `'run_id'`. `get_recon_report_tool_signatures` /
  `render_recon_report_tool_guidance` drop both header-supplied parameters from the rendered call examples.
  In the same change the hand-written recon-report call examples in `prompts/stage2.py`, `stage3.py` and
  `prompts/fragments.py` stop passing `run_id`, and the `recon_context` paragraph in `stages/base.py` keeps
  its `run_id` line as information for the fused-memory calls that take one but says recon-report calls
  take the run from the connection and must not pass it (the "echo run_id" instruction goes); the
  "Verifying add_finding responses" list in `prompts/stage1.py` (and its Stage-2 and Stage-3 counterparts,
  where present) names `stage_identity_mismatch` and `run_id_missing` beside `run_id_unknown`.
  `tests/test_recon_report_guidance_drift.py`'s two call-example `run_id` guards
  (`test_every_report_tool_call_example_shows_run_id`, `test_fallback_call_examples_all_carry_run_id`)
  invert to "no call example carries a header-supplied parameter";
  `test_shared_guidance_tool_takes_run_id_first` is deleted with the other inverted guards rather than kept,
  and `_FROZEN_RECON_REPORT_SIGNATURE_SPECS` drops `run_id`: this replaces task 2559's guard against the
  run_id-omission bug; it does not delete it. `tests/test_stages.py::test_recon_context_includes_run_id` is
  rewritten to pin that the paragraph still shows the run id and tells the agent not to pass it to
  recon-report calls.
- DELETES: the prompt instructions to echo `run_id` into recon-report calls; the draft's override path
  (no `recon_report_identity_override` event is ever written).
- Observable signal: B17 passes in CI; after deploy, a dark_factory cycle's recon-report findings land in
  that cycle's run while its stage prompts carry no `run_id` instruction for recon-report calls, and
  `journalctl -u fused-memory` shows no `recon_stage_identity_refused` line with `field=run_id` for that
  run, or each one names both values.
- G7: INV-1 (identity is a header the server reads and enforces, not prose), INV-2 (structured refusal),
  INV-9 (one source for the run id), `one-decision-partition-per-area` (INV-23; a decidable fact leaves the LLM).

**π — Add the recon finding-category candidate check script and its tests**
(`task_kind=normal`, priority medium, est. ~450 LOC with tests)
- Files: fused-memory/scripts/check_recon_category_candidates.py (new),
  fused-memory/tests/test_check_recon_category_candidates.py (new)
- Deps: ε (C11's `read_rejected_findings`, the table and `RejectedFindingsTableMissing`).
- Keys: none of S13's (Leo's addition to OQ-2; fk-7c13df0b988c is closed by ε).
- Work: the deterministic half of the category analysis (D17: code decides by thresholds; judgement
  happens at the escalation ρ files). The script reads `rejected_findings` through C11's reader, over a
  window ending now, and normalises each raw label once (lower case, runs of characters outside
  `[a-z0-9]` collapsed to `_`, stripped). It exits **1**, printing each finding it fires on, when either
  (a) one normalised label not passed as `--ruled-label` is refused at least `--min-refusals` times across
  at least `--min-runs` runs and more than half of its refusals are un-retried or retried as `other`
  (printed with its counts, runs, retry mix and two description excerpts), or (b) the share of refusals
  never retried exceeds `--max-lost-share` over at least `--min-sample` refusals. It exits **0** with a
  one-line verdict otherwise (the runner keeps it as the link's bounded completion note), and **75** when
  the store has no `rejected_findings` table yet (fused-memory not redeployed since ε). Defaults:
  `--window-days 14 --min-refusals 10 --min-runs 3 --max-lost-share 0.20 --min-sample 20`, the db path
  from the fused-memory config's `reconciliation.data_dir`, overridable with `--db`. `--ruled-label` is
  repeatable: a label a human has already ruled on (added to the vocabulary, named as a synonym in the
  prompt, or rejected) stops firing condition (a). Read-only throughout; no network, no task-store call.
  The script is committed executable (git mode `100755`) with a `#!/usr/bin/env python3` line: the
  deterministic runner execs it directly
  (`orchestrator/src/orchestrator/before_done_script.py::run_script`, `create_subprocess_exec(str(script))`
  with the orchestrator's own environment, whose `PATH` starts with the checkout's `.venv/bin`, so the
  import of C11's reader resolves), and `submit_task` refuses ρ unless the file exists and passes `os.X_OK`.
- DELETES: nothing.
- Observable signal: B18 passes in CI; on main, `git ls-files -s
  fused-memory/scripts/check_recon_category_candidates.py` shows mode `100755`, and
  `fused-memory/scripts/check_recon_category_candidates.py --db data/reconciliation/recon_report_state.db`
  executed directly from the checkout root under the checkout's `.venv` prints a verdict line and exits 0,
  1 or 75.
- G7: INV-10 (the tests seed rows and assert every exit), `one-decision-partition-per-area` (INV-23; one
  table of thresholds in code; judgement only at the escalation).

**ρ — Recon finding-category candidates: fortnightly check (seed link of a recurring chain)**
(`task_kind=deterministic`, priority medium, no code; filed by the lead the day π's merge lands, not at
decompose)
- Filing: `submit_task` validates `before_done.script` at submit, for `planning_mode=True` too: it must
  exist and be executable under `project_root`
  (`fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::_validate_script_under`). At
  decompose π's script is not on main, so ρ would be refused and never persisted. The lead files it once
  π is `done` and its merge is on main (a `lead_actions` entry with that trigger), reads it back with
  `get_task`, and stamps its id into the sidecar's ρ block.
- Files: none (`files: []`).
- Deps: π (already `done` when ρ is filed; the edge is still wired so the order is on record). A link must
  never fire before its script is on main: a delivered check against an uncommitted script has turned main
  red before.
- Keys: none (`source: review-all`, `x_finding_run: review-all-dark_factory-20261008`).
- Shape (docs/task-authoring.md §6.1, carrier contract): `before_done = {kind: 'predicate', script:
  'fused-memory/scripts/check_recon_category_candidates.py', args: ['--window-days', '14', '--min-refusals',
  '10', '--min-runs', '3', '--max-lost-share', '0.20', '--min-sample', '20'], timeout_secs: 120,
  recheck_secs: 3600}`; `metadata.milestone = {mode: 'dated', at: <filing time + 30 days, UTC>}`;
  `metadata.recurrence = {key: 'recon-category-candidates', interval_secs: 1209600}`. The thresholds sit
  in `args` so a human tunes them by editing the link, and every minted successor copies them. The dated
  milestone is about 30 days after ρ is filed, so after π lands and roughly when ε's refusals have
  accumulated in production. `recheck_secs`, the `--min-sample 20` argument and π's `--ruled-label` flag
  were added by the ruling fold and confirmed by the lead (lead rulings after the wave-3 fold-verifiers):
  `recheck_secs` lets the script's exit 75 ("no table yet") re-arm the link hourly
  instead of failing it, and the runner's stall cap (30 re-arms) turns a fused-memory that stays
  undeployed into a `milestone_check_stalled` escalation.
- Description (what the resolver reads): what the check measures, the two conditions and their
  arguments, and how to resolve a firing link. A firing link files a born-at-L2 `milestone_check_failed`
  escalation (deny-listed from L2 auto-close) carrying the script's output and blocks; no successor is
  minted while it is blocked. A resume re-runs the check (docs/deterministic-runner.md, section-1 resume),
  so after ruling on a candidate the resolver either adds `--ruled-label <label>` to the link's
  `before_done.args` and resumes, or files the follow-up (a new `FindingCategory` value, a prompt synonym)
  and resumes once the condition has cleared; `abandon` ends the chain.
- Observable signal: once filed, the link exists `pending` with the three metadata keys and the edge onto
  π; after its first fire it is
  `done` with `done_provenance.kind='deterministic-milestone'` and a successor `pending` titled with a
  `[due …]` run label, or `blocked` with a pending `milestone_check_failed` escalation naming a label or
  the lost share.
- Consequences stated as ruled: a firing link blocks the chain until its escalation is resolved, so an
  unattended escalation stops the analysis; this is the project's first live recurring chain (0 tasks
  carry `recurrence` on 2026-11-13), so its first runs also test the mechanism.

**ω — Run the recon flag and report seams' two-way boundary suite and the live-run check (integration gate)**
(priority medium, est. ~500 LOC)
- Files: fused-memory/tests/reconciliation/test_recon_flag_identity_boundary.py (new)
- Deps: α, β, γ, δ1, δ2, ε, ζ, η, θ1, θ2, θ3, ι, κ1, κ2, κ3, λ, μ, ν, ξ, π. Not ρ: ρ's first link fires about
  30 days after filing and its links recur after the program; π's tests cover the script it runs.
- Keys: none (verifies the leaves above).
- Work: the cross-module rows B3, B4, B7, B10, B11b, B12, B13, B14b and B17 driven through the MCP tools,
  the real ledger and the real recon-report store in one module; a checklist run against one live
  dark_factory cycle after the last deploy.
- Observable signal (the PRD's signal leaf): the suite passes in CI, and one live dark_factory recon
  cycle after deploy shows (a) Stage 1's findings reaching Stage 2 only through the recon-report channel
  (`stage1_analytical_findings_processed` present, no Mem0 relay stat key), (b) no recon-stage Mem0 write
  carrying a retired marker key and no recon-stage `add_memory` whose params contain `has status` in
  `write_journal.db`, (c) no accepted off-vocabulary category in `recon_report_state.db` for that run, and
  a `rejected_findings` row for each refusal the stage reports count, (d) `EdgeSweepStats` keys for both
  sweeps in the Stage-1 report, (e) no `recon_stage_identity_refused` line for that run's recon-report
  calls, or each one names both values.
- G7: INV-10, INV-13.

**Acceptance record — fk-9b76dd8a1cdb** (OQ-1 and OQ-8, ruled by Leo; filed `cancelled` at decompose,
priority high, `files: []`, `metadata.x_acceptance_reason`): "Recon's Graphiti task-status grammar (the
detector in `status_snapshot_extraction.py` and the stale-status sweep's selection) stays as a cleanup
backstop for task-status prose written by writers other than reconciliation, until the
recall-from-the-task-store design session (spawned 2026-11-13 by Leo's ruling R-1 = A, outside this plan)
decides the write and read rules. Under Leo's rule that nothing writes task status into memory or reads it
from there, the finding's typed task-status claim is not built: it would keep a copy of status inside
memory. Reconciliation itself stops writing status prose: the sweep writes no replacement sentence and the
Stage-1 prompt asks for no status fact (S13 η). Measured 2026-11-12: of 115 status-phrased add_memory
writes since 2026-10-08, about two thirds came from non-recon writers, none of them gated, and graphiti-core
can extract a status edge from text that states none, so detection is still needed. Re-raise this record
if that session lands a write rule for every writer."

### Existing tasks (contract §8 dedup, re-run 2026-11-11 at `97be1d71fa`; re-run 2026-11-13 for the ruling fold)

- Ruling-fold re-run (2026-11-13): step 1, a read-only sqlite scan of `x_finding_key` over every task (322
  carry it), finds none of the 20 keys. Step 2 for the new and grown work: `search_tasks` on "record refused
  recon-report findings … analyse them for category candidates" (top 0.55, tasks 2410, 1971, 2453, 2088 —
  all done or cancelled, none about a refusal record), on "recurring fortnightly check … recon finding
  categories" (top 0.54, done 5257/3709/2693/2664), on "Stage-1 prompt stops writing replacement task-status
  facts" (3079 0.66 done, 2694 done, 3888 0.63 **deferred**: status-transition-triggered stale-edge detection
  plus surfacing dropped sweep stats, a different change; 2628 done, the recon-only status gate) and on "the
  recon-report server reads the run id from the header and refuses a mismatch" (3988 done, 4011 deferred,
  named below): no live near-duplicate. Task 3936 (deferred, "durably record every soft-blocked add_memory
  write") is about the add_memory gates, not recon-report, and does not cover ε's record.

- Step 1 (`x_finding_key` over every task's metadata, forensic `json` read of the live store, all
  statuses): no task carries any of the 20 keys.
- Step 2 (`search_tasks` on each leaf's remedy, ≥ 0.6, confirmed with `get_task`): 6846 (0.70,
  IntegrityCheck extraction) → cancel into ι; 7807 (0.65, marker-pool GC extraction) → cancel into κ1;
  7065 (0.63, briefing mismatch source injectable) → cancel into ν; 2454, 3037, 2781, 2219, 2312
  (≥ 0.6) are `done`. Below 0.6 but confirmed by reading: 7388 names fk-65aeb80b161d → cancel into γ;
  5996 plans the same flag_dedup partition (76 declared files, no branch, no claimant) → cancel into θ1–θ3.
  Fold probes: "delete the dormant flag_for_stage2 relay, its markers and the pool sweeper" (top hits 1139,
  4863, 2406, 2312, 2966, 1660 `done`; 1376, 3762 `cancelled`) and "remove the human-operator stall marker
  and its escalation" (1201, 3017, 1204, 1192, 1517, 1154, 1169 `done`; 5793 `cancelled`): no live
  near-duplicate. Every task named in this PRD is still `pending` (4011 `deferred`).
- Phase-5 mechanical tasks in these files (none carries an S13 key): 6598 (`citation_verifier.py`) and
  6599 (`task_filter.py`), ordered in §Cross-PRD relationship.

## Out of scope

- A typed Graphiti task-state claim (the fk-9b76dd8a1cdb proposal): rejected under Leo's rule (OQ-1).
- The recall-from-the-task-store design (Leo's question under S13-1): recall that draws task facts from the
  task store, the write rule for every writer, what counts as task status (completion notes, "landed at
  commit X"), the Stage-2 task-count snapshot, the 2026-07-02 norm memory, task 3140's relation, and when the
  detector and the sweep's selection can retire. A separate fable /team session the lead spawned owns all of
  it; this plan neither designs it nor waits on it.
- Status prose from writers other than reconciliation (the orchestrator, task agents, interactive sessions,
  the escalation watcher): ungated by this plan, as the sitting's option 2 was not chosen.
- `harness.py` prose and structure, the recon run driver, `MemoryConsolidator`'s construction, the
  growth-streak semantics and the briefing hook's task-write path (S12); S12 δ's identity sweep of
  `CLAUDE.md`, `.claude/commands/memory.md`, `docs/task-authoring.md` and the stage `agent_id` /
  `_causation_id` lines (S12).
- Moving modules into `fused_memory.recon_state` (S19).
- The `stage1_flag_suppression` and `cycle_summary` Mem0 mirrors, and the `stage2_suppress` guard memories
  (D11).
- The `reconciliation_stale_human_operator` escalation category in `escalation/` (S1/S2 own the
  vocabulary; κ1 deletes only its recon filer).
- Widening `FindingCategory` itself: a new value is filed when ρ's escalation is ruled, not by this plan.
- `server/tools.py`'s guard chains (S15); the loop-blocking gate's primitive table (it does not see
  sqlite commits; μ fixes the one site this stream owns).

## Open questions

None. Every design question was ruled at the wave-3 sitting (§Rulings); the text is authored under the
rulings.

Tactical questions (decide in the named leaf):

1. Whether `RunFindingIndex` takes the moved record dataclasses or a Protocol over them. Decide in ζ.
2. The exact stat key for partial composite drops. Decide in β.
3. Where the `flag_for_stage2` kind name lives once no prompt writes it: deleted outright, or kept in
   `memory_metadata.py::KIND_REGISTRY` for the residue rows another writer stamped. Decide in κ3 from the
   residue read.
5. Whether a refusal's `retried_as` stamp also fires on a same-run retry that returns `duplicate_finding`
   (the finding exists, so the retry was accepted in substance). Default: yes. Decide in ε.
6. Whether π excludes refusals younger than about an hour (a still-running run may yet retry them).
   Default: no; the next link sees them. Decide in π.
4. Whether `EdgeSweepStats` lives in `graph_edge_sweep.py` or a shared stats leaf other sweeps can use.
   Default: `graph_edge_sweep.py`.
