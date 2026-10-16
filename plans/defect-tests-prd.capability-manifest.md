# Capability manifest — plans/defect-tests-prd.md

Per-leaf capability → evidence bindings (G3 + G6 mechanized), built at decompose on 2026-10-16 by seats M-A and
M-B and revised after the batch critic: β split into β1/β2, λ into λ1/λ2, δ1b added, ε gained the roles.py
exposure, η gained the train path. All bindings are PASS. Symbols are cited by name, never by line. The
machine-readable twin is `plans/defect-tests-prd.capability-manifest.yaml`; its task ids are stamped by
`commit_planning`.

## α1 — Defect vocabulary: shared/defects.py types, ref grammar, transition table, metadata registration

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| metadata-submodel-registration-api-with-list-cardinality | capability→producer (wired) — shared/src/shared/task_metadata.py::register_metadata_submodel (cardinality='list' opt-in; precedent shared/src/shared/capability_manifest.py registers delivered_checks list-valued; parse_metadata enforces wrong_cardinality) | PASS | — |
| exemption-owner-type-debt | capability→producer (wired) — shared/src/shared/governed_exceptions.py::Debt (owner: TaskRef \| TicketRef; frozen dataclasses, __post_init__ rejects other owners); consumed by roles.py via DECLARATION_FORMS. Debt has no JSON wire form, so the Exemption wire shape for owner is α1's own deliverable | PASS | — |
| task-authoring-metadata-vocabulary-section-exists | capability→producer (wired) — docs/task-authoring.md section '8. Task metadata vocabulary & census' (the entry site for the new keys) | PASS | — |
| defects-vocabulary-module-and-records | producer:α1 — the leaf itself builds shared/src/shared/defects.py; absent on main | PASS | path present: `shared/src/shared/defects.py` |
| defect-declaration-record-type | producer:α1 — frozen DefectDeclaration record (Contract Types); absent on main | PASS | grep `class DefectDeclaration` in `shared/src/shared/` |
| defect-test-retirements-key-registered | producer:α1 — defect_test_retirements registered beside defects (list cardinality); key absent from shared/src on main | PASS | grep `defect_test_retirements` in `shared/src/shared/` |
| task-authoring-documents-defect-keys | producer:α1 — docs/task-authoring.md section 8 entry for defects and defect_test_retirements; absent on main | PASS | grep `defect_test_retirements` in `docs/task-authoring.md` |
| transition-table-ref-grammar-single-home | producer:α1 — I2 transition table and decision-15 ref grammar live only in shared/src/shared/defects.py | PASS | manual: symbol names for the table and grammar are not fixed by the PRD; judged by α1's own table-driven tests (every legal transition accepted, every other refused) |
| issue-kind-accessor | producer:α1 — issue_kind_of(issue): fail-closed default defect, flagged (decision 16) | PASS | grep `def issue_kind_of` in `shared/src/shared/defects.py` |

## α2 — declare_defect / record_defect_outcome write path, creation floor, update_task rejection, authority drift test

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| per-ref-merge-inside-one-write-unit | capability→producer (wired) — fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._write_lock (held by update_task and submit paths) and SqliteTaskBackend._write_unit in fused-memory/src/fused_memory/backends/sqlite_task_backend.py | PASS | — |
| terminal-task-detection | capability→producer (wired) — fused-memory/src/fused_memory/middleware/live_task_write_guard.py::is_terminal_reopenable (done/cancelled), consumed by update_task gating | PASS | — |
| write-authority-rejection-precedent | capability→producer (wired) — fused-memory/src/fused_memory/backends/task_backend_errors.py::DoneProvenanceWriteAuthorityError raised from SqliteTaskBackend.update_task (verbatim-passthrough carve-out under metadata_mode='replace') and mapped by TaskInterceptor.update_task via to_error_dict; α2 generalises the guard to defects and defect_test_retirements | PASS | — |
| mcp-tool-registration-surface | capability→producer (wired) — fused-memory/src/fused_memory/server/tools.py registers update_task / submit_task / set_task_status the same way the new operations will be | PASS | — |
| transition-table-and-event-types | producer:α1 upstream — shared/src/shared/defects.py DeclareEvent, OutcomeEvent and the I2 table (α1 scope covers types, grammar and table) | PASS | — |
| role-allowed-tools-drift-test-substrate | capability→producer (wired) — orchestrator/src/orchestrator/agents/roles.py::AgentRole.allowed_tools; precedent orchestrator/tests/test_role_done_provenance_gate.py (the drift test must live in orchestrator/tests, which can import roles) | PASS | — |
| malformed-declaration-rejected-synchronously-naming-values | rejection-mechanism — built+bound by α2: validates through α1 models explicitly; must not lean on the enforce-gated parse_metadata write path (SqliteTaskBackend._validate_metadata_on_write only raises when _task_metadata_enforce is on) | PASS | manual: rejection quality is judged by α2's fixtures observing the diagnostic fire for each malformed shape |
| declare-defect-mcp-operation | producer:α2 — absent on main | PASS | grep `async def declare_defect` in `fused-memory/src/fused_memory/server/tools.py` |
| record-defect-outcome-mcp-operation | producer:α2 — absent on main | PASS | grep `async def record_defect_outcome` in `fused-memory/src/fused_memory/server/tools.py` |
| authority-drift-test-names-record-defect-outcome | producer:α2 — drift test asserting record_defect_outcome is in no role's allowed_tools; absent on main | PASS | grep `record_defect_outcome` in `orchestrator/tests/` |
| lookup-by-ref-and-update-task-rejection | producer:α2 — cross-task lookup by ref (no metadata-key query exists in SqliteTaskBackend on main) and update_task refusal of defects / defect_test_retirements for every caller | PASS | manual: symbol names are not fixed by the PRD; B17 and the round-trip through get_task are the observable proof |

## β1 — Injected pytest plugin, verify-leg injection, map archiving, defect_tests config, marker registration

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| pytest-leg-structured-injection-seam | capability→producer (wired) — orchestrator/src/orchestrator/verify_cmd.py::with_junitxml and with_pytest_timeout (structured PYTEST legs only; raw-retained and OPAQUE are no-ops), applied at orchestrator/src/orchestrator/verify.py inject-junit site; the INFO log on a suppressed injection is the B22 precedent | PASS | — |
| pythonpath-injection-site | capability→producer (wired) — orchestrator/src/orchestrator/verify.py::_target_subprocess_env (overlays caller extra LAST, so the new PYTHONPATH entry must be merged with, not replaced by, an existing value) | PASS | — |
| tool-classification-pytest-vs-opaque | capability→producer (wired) — orchestrator/src/orchestrator/verify_cmd.py::ToolKind and parse_config_command (PYTEST, OPAQUE, raw-retained chains) | PASS | — |
| marker-registration-drift-test | capability→producer (wired) — orchestrator/tests/test_marker_registration_drift.py (β extends it for the defect marker in member pyprojects) | PASS | — |
| defect-tests-mode-off-skips-injection | producer:β1 — the defect_tests: config group (mode, max_plan_repairs, max_proof_replans, proof_timeout_secs, streak_threshold, test_file_pattern; reload green tier) moved to β on 2026-10-16 so β, ε, ζ1, ζ2 and η all read it from an upstream leaf; injection is skipped when mode == off | PASS | grep `^[[:space:]]+defect_tests[[:space:]]*:` in `orchestrator/src/orchestrator/config.py` |
| plugin-file-alone-in-its-directory | producer:β1 — absent on main | PASS | path present: `shared/src/shared/_pytest_inject/_df_defect_plugin.py` |
| plugin-registers-markers-additively | producer:β1 — addinivalue_line registration so a target's own markers list is never overridden | PASS | grep `addinivalue_line` in `shared/src/shared/_pytest_inject/` |
| collection-map-written-next-to-junit | producer:β1 — <junit>.tests.json collection map (consumed by client-feed ε 6972) | PASS | grep `tests\.json` in `shared/src/shared/_pytest_inject/` |
| dash-p-flag-added-to-structured-pytest-legs | producer:β1 — orchestrator adds -p _df_defect_plugin; identifier absent from orchestrator/src on main | PASS | grep `_df_defect_plugin` in `orchestrator/src/orchestrator/` |
| plugin-directory-on-pythonpath | producer:β1 — only the _pytest_inject directory goes on PYTHONPATH; identifier absent from orchestrator/src on main | PASS | grep `_pytest_inject` in `orchestrator/src/orchestrator/` |
| no-shadowing-and-non-workspace-verification | rejection-mechanism — B21 and B18 fixtures (project with its own top-level testing module, filterwarnings=error) built by β and observed | PASS | manual: no-shadowing is a runtime property of the one-file directory, proven by fixture, not by grep |
| tests-json-archived-beside-the-archived-junit | producer:β1 — verify.py::_archive_junit_report (cost and attributing paths) also archives the sibling <junit>.tests.json under the junit's stem, so client-feed's archive reader (6969) and ι see the map | PASS | grep `tests\.json` in `orchestrator/src/orchestrator/verify.py` |
| defect-marker-registered-in-root | producer:β1 — marker registration (per member; one check per file because grep paths are disjunctive) | PASS | grep `defect[(:]` in `pyproject.toml` |
| defect-marker-registered-in-cockpit | producer:β1 — marker registration (per member; one check per file because grep paths are disjunctive) | PASS | grep `defect[(:]` in `cockpit/pyproject.toml` |
| defect-marker-registered-in-dashboard | producer:β1 — marker registration (per member; one check per file because grep paths are disjunctive) | PASS | grep `defect[(:]` in `dashboard/pyproject.toml` |
| defect-marker-registered-in-escalation | producer:β1 — marker registration (per member; one check per file because grep paths are disjunctive) | PASS | grep `defect[(:]` in `escalation/pyproject.toml` |
| defect-marker-registered-in-fused-memory | producer:β1 — marker registration (per member; one check per file because grep paths are disjunctive) | PASS | grep `defect[(:]` in `fused-memory/pyproject.toml` |
| defect-marker-registered-in-orchestrator | producer:β1 — marker registration (per member; one check per file because grep paths are disjunctive) | PASS | grep `defect[(:]` in `orchestrator/pyproject.toml` |
| defect-marker-registered-in-sampler | producer:β1 — marker registration (per member; one check per file because grep paths are disjunctive) | PASS | grep `defect[(:]` in `sampler/pyproject.toml` |
| defect-marker-registered-in-shared | producer:β1 — marker registration (per member; one check per file because grep paths are disjunctive) | PASS | grep `defect[(:]` in `shared/pyproject.toml` |
| outcome-recorder-reads-call-excinfo-xdist-safe | producer:β1 — outcomes from pytest_runtest_makereport hookwrapper and pytest_exception_interact (call.excinfo); ImportError module=e.name symbol=e.name_from; single designated writer, atomic replace (decision 9) | PASS | manual: judged by β1's fixtures including an -n 2 run and B4 |

## β2 — Static defect enumerator: enumerate_source, tree walker, completeness, skip/xfail, CLI

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| static-mark-grammar-primitives | capability→producer (wired) — shared/src/shared/pytest_mark_grammar.py::marker_name, mark_elements, pytestmark_value (consumed by orchestrator/src/orchestrator/pytest_markers.py). Extent: name-level only; decorator walk, argument extraction and skip/xfail detection are β2's own work | PASS | — |
| static-enumerator-module | producer:β2 — absent on main | PASS | path present: `shared/src/shared/defect_enumerator.py` |
| enumerate-source-entry-point | producer:β2 — pure per-file entry point enumerate_source(path, text) so η enumerates from git objects | PASS | grep `def enumerate_source` in `shared/src/shared/defect_enumerator.py` |

## γ — DEFECT_TEST_RULE constant and splice into the six fixing prompts; declared defects in task briefings

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| six-fixing-prompt-sites-exist | capability→producer (wired) — orchestrator/src/orchestrator/agents/roles.py ARCHITECT, SIMPLE_TASK, DEBUGGER, STEWARD roles; orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_replan_prompt and build_amender_prompt | PASS | — |
| task-record-rendering-hook | capability→producer (wired) — orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler._format_task, reached through _task_preamble by the architect, replan, plan-completion, plan-tightening, schema-repair, revalidation, simple-task and steward-initial builders; precedent _format_delivered_checks (task 5359) | PASS | — |
| rule-in-five-role-system-prompts | producer:γ — decision 17 (amended 2026-10-16): DEFECT_TEST_RULE is spliced into the system prompts of ARCHITECT (covers replan), IMPLEMENTER (covers amender), SIMPLE_TASK, DEBUGGER and STEWARD, one site per dispatch; declared defects are listed only in the task-preamble briefings via _format_task (the debugger and amender briefings carry no task record by design) | PASS | — |
| marker-binding-the-rule-names | producer:β1 upstream — @pytest.mark.defect registration and the static enumerator (β is a γ prerequisite) | PASS | — |
| defects-metadata-type-for-rendering | producer:α1 upstream — DefectDeclaration and the registered defects slice | PASS | — |
| single-architect-recipe-site-for-splice | producer:task-5532 upstream (pending, no deps) — collapses the duplicated architect recipe between roles.py ARCHITECT and briefing.py so the once-per-prompt sentinel test is meaningful for the architect path | PASS | — |
| roles-py-doctrine-edit-sequencing | producer:task-5753 upstream (pending; edits roles.py, adds ESCALATE_INFO_CONTRACT and the IMPLEMENTER doctrine) — sequencing prerequisite only, no symbol relied on | PASS | — |
| debugger-failure-text-sequencing | producer:task-5357 upstream (pending, no deps) — edits build_debugger_prompt and workflow call site; sequencing prerequisite only | PASS | — |
| defect-test-rule-constant | producer:γ — absent on main | PASS | grep `DEFECT_TEST_RULE` in `orchestrator/src/orchestrator/agents/roles.py` |
| sentinel-identity-guard-test | producer:γ — splice-by-identity guard asserting the rule appears exactly once per composed prompt | PASS | grep `DEFECT_TEST_RULE` in `orchestrator/tests/` |
| unblock-skill-carries-rule-pointer | producer:γ — pointer plus rendered excerpt in skills/unblock/SKILL.md | PASS | grep `DEFECT_TEST_RULE` in `skills/unblock/SKILL.md` |
| unblock-low-risk-skill-carries-rule-pointer | producer:γ — pointer plus rendered excerpt in skills/unblock-low-risk/SKILL.md | PASS | grep `DEFECT_TEST_RULE` in `skills/unblock-low-risk/SKILL.md` |
| contributing-carries-rule-pointer | producer:γ — pointer plus rendered excerpt in CONTRIBUTING.md | PASS | grep `DEFECT_TEST_RULE` in `CONTRIBUTING.md` |
| declared-defects-rendered-in-task-block | producer:γ — _format_task renders metadata.defects | PASS | manual: renderer symbol is not fixed by the PRD and a defects-word grep would be satisfied by the rule text alone; judged by the persisted-briefing test |

## δ1 — Reviewer contract: issue kind and defect_rulings, validation, legacy default, harness bump

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| reviewer-contract-template-and-harness-version | capability→producer (wired) — orchestrator/src/orchestrator/agents/roles.py::_REVIEWER_CONTRACT_TEMPLATE and _REVIEWER_PROMPT_HARNESS_VERSION; literal pin in orchestrator/tests/test_reviewer_prompt_split.py | PASS | — |
| retryable-status-error-rejection-path | capability→producer (wired) — orchestrator/src/orchestrator/mcp/verdict_tools.py::_submit_review_verdict already returns {'status': 'error', 'message': ...} for a reviewer mismatch and an unknown verdict | PASS | — |
| issue-validator-seam | producer:task-5494 upstream (pending, no deps) — per-issue shape and vocabulary validation derived from _REVIEWER_CONTRACT_TEMPLATE in _submit_review_verdict. Extent caveat: 5494 prefers normalise-and-flag and rejects only unusable issues, so δ1's hard refusal of a missing or unknown kind is δ1's own rule on top of that seam | PASS | — |
| verdict-contract-edit-sequencing | producer:task-5224 upstream (pending; deps 5221, 5223) — adds note_dispositions to the same verdict tool and contract template; one harness bump per landing | PASS | — |
| issue-section-renderer | capability→producer (wired) — orchestrator/src/orchestrator/artifacts.py::_format_issue_section | PASS | — |
| kind-and-ruling-enums-to-render | producer:α1 upstream — IssueKind and Ruling vocabulary in shared/src/shared/defects.py | PASS | — |
| defect-rulings-in-verdict-tool | producer:δ1 — absent on main | PASS | grep `defect_rulings` in `orchestrator/src/orchestrator/mcp/verdict_tools.py` |
| defect-rulings-in-reviewer-contract | producer:δ1 — contract template declares defect_rulings | PASS | grep `defect_rulings` in `orchestrator/src/orchestrator/agents/roles.py` |
| contract-refusal-names-allowed-values | rejection-mechanism — built+bound by δ1 (B7: missing kind, unknown kind, malformed defect_rulings entry each observed to return status error naming the enum) | PASS | manual: rejection quality is judged by B7 fixtures; the harness-version literal is not predictable because 5224 or 5494 may bump first |
| legacy-kind-default-via-accessor | producer:α1 upstream — issue_kind_of; δ1 no longer waits on 6739 | PASS | — |

## δ1b — Route the review-payload normaliser (6739) through issue_kind_of

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| review-payload-normaliser-for-legacy-default | producer:task-6739 upstream — review_payload.py::normalize_review (absent on main; 6739 creates it); δ1b routes its kind read through α1 issue_kind_of | PASS | — |
| normaliser-uses-accessor | producer:δ1b | PASS | grep `issue_kind_of` in `orchestrator/src/orchestrator/review_payload.py` |

## δ2 — Reviewer consumption: in-task defect declaration writer, ruling application, reviewer briefing

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| declare-defect-operation | producer:α2 upstream (α2 is a δ2 prerequisite) — declare_defect covering Declare, DeclareClosure, DeclareRetirement and Sighting | PASS | — |
| record-defect-outcome-rule-events | producer:α2 upstream — record_defect_outcome carrying Rule events | PASS | — |
| issue-kind-on-aggregated-blocking-issues | producer:δ1 upstream — required kind on every issue and defect_rulings on the verdict, carried through aggregate_reviews | PASS | — |
| declaration-before-cap-exit-site | capability→producer (wired) — orchestrator/src/orchestrator/workflow.py::TaskWorkflow._execute_verify_review_loop blocking branch: review_cycle += 1, set_review_counters, then the max_review_cycles check that calls _exit_review_cycles_exhausted; the declaration goes ahead of that check | PASS | — |
| orchestrator-to-store-write-seam | capability→producer (wired) — orchestrator/src/orchestrator/scheduler.py::SchedulerFacade.dispatch_tool (generic MCP call, used from TaskWorkflow) and SchedulerFacade.get_task | PASS | — |
| reviewer-briefing-task-block | producer:task-5369 upstream (pending; dep 5225) — wires the task record into BriefingAssembler.build_reviewer_prompt; today build_reviewer_prompt takes no task. Extent caveat: δ2's defects section rides that block only if 5369 renders through _format_task; if it hand-rolls a block, δ2 must add the defects section itself | PASS | — |
| seeding-pending-closures-for-reviewer-half-of-b8-b23 | producer:α2 upstream — DeclareClosure and DeclareRetirement through declare_defect seed the pending states; ε's plan tools (downstream of δ2) are not needed for δ2's signal | PASS | — |
| loop-exit-edit-sequencing-with-review-completed | producer:task-6967 upstream (pending, no deps) — adds the review_completed emit at the same loop exits; sequencing prerequisite only. Its description carries by_category counts but no by_kind (the paired amendment is a decompose-time edit) | PASS | — |
| in-task-declaration-writer-calls-declare-defect | producer:δ2 — orchestrator call to declare_defect absent on main | PASS | grep `declare_defect` in `orchestrator/src/orchestrator/` |
| rulings-applied-through-record-defect-outcome | producer:δ2 — orchestrator call to record_defect_outcome absent on main | PASS | grep `record_defect_outcome` in `orchestrator/src/orchestrator/` |
| reviewer-briefing-shows-defects-pending-closures-retirements | producer:δ2 — section in build_reviewer_prompt | PASS | manual: section heading is not fixed by the PRD; judged by the persisted reviewer briefing test |

## θ1 — Auto-filing sites declare defects: submit_only_instructions, steward and deep-reviewer templates, curator candidate and combine

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| submit-only-instructions-defects-parameter | capability→producer (wired) — orchestrator/src/orchestrator/agents/roles.py::submit_only_instructions, called by _FOLLOWUP_FILING_INSTRUCTIONS, the STEWARD templates and DEEP_REVIEWER, and by review_checkpoint.py::_build_prompt; the defects parameter is this leaf's deliverable | PASS | grep `^\s+defects\s*:` in `orchestrator/src/orchestrator/agents/roles.py` |
| review-checkpoint-prompt-declares-defects | capability→producer (wired) — orchestrator/src/orchestrator/review_checkpoint.py::_build_prompt calls submit_only_instructions; passing the defects block is this leaf's deliverable | PASS | grep `\bdefects\b` in `orchestrator/src/orchestrator/review_checkpoint.py` |
| agent-followup-ref-keys-already-in-the-template | capability→producer (wired) — _FOLLOWUP_FILING_INSTRUCTIONS metadata already carries spawned_from and suggestion_hash, the two inputs of ref followup:<spawned_from>:<suggestion_hash>; the ref grammar is producer:α1 upstream | PASS | — |
| submit-task-accepts-open-defects | producer:α2 upstream (submit_task creation floor: defects accepted only as open entries with no proof and no ruling) | PASS | — |
| curator-candidate-carries-declarations | capability→producer (wired) — fused-memory/src/fused_memory/middleware/task_interceptor.py::_build_candidate constructs task_curator.py::CandidateTask (no defects field on main); the field, its payload_hash inclusion and the metadata plumb are this leaf's deliverable | PASS | grep `\bdefects\b` in `fused-memory/src/fused_memory/middleware/task_curator.py` |
| combine-declares-only-onto-a-pending-target | capability→producer (wired) — task_curator.py::is_combine_eligible_status is pending-only and task_interceptor.py::_execute_combine already takes candidate_metadata and aborts to create on an ineligible target, so the in-progress half of B15 is existing behaviour; the declare_defect call on the pending branch is producer:α2 upstream plus this leaf | PASS | manual: B15 behaviour (pending target gains entries via declare_defect, in-progress target gets a separate task); no stable symbol is fixed by the PRD, and declare_defect may legitimately already appear in the interceptor once α2 lands |

## ε — Plan tools (reproduces, declare_not_reproducible, declare_withdrawn, retire_defect_test), plan check, defect_tests config

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| plan-site-hook-points-exist | capability→producer (wired) — orchestrator/src/orchestrator/workflow.py::TaskWorkflow._plan, _validate_prerequisites_or_block, _replan, _apply_revalidation_skip, _should_run_simple_task, _run_simple_task, _execute_verify_review_loop; orchestrator/src/orchestrator/artifacts.py::TaskArtifacts.stamp_plan_provenance and lock_plan | PASS | — |
| bounded-architect-repair-precedent | capability→producer (wired) — orchestrator/src/orchestrator/workflow.py::TaskWorkflow._repair_plan_schema with briefing.py::BriefingAssembler.build_plan_schema_repair_prompt | PASS | — |
| persisted-task-lifetime-counters | capability→producer (wired) — orchestrator/src/orchestrator/artifacts.py::TaskArtifacts.set_review_counters (task 2749), called from the review loop; ε adds defect_plan_repairs_total beside it | PASS | — |
| plan-tool-registration-and-step-add-replace-paths | capability→producer (wired) — orchestrator/src/orchestrator/mcp/plan_tools.py _add_plan_step, _replace_plan_step and the registered add_plan_step tool | PASS | — |
| step-type-enum-validation | producer:task-5752 upstream (in-progress; ε prerequisite) — STEP_TYPES and the structured refusal in _add_plan_step and _replace_plan_step (its item 4). ε's own scope line 'step_type is validated as an enum' duplicates this extent and should be dropped in favour of relying on 5752 | PASS | — |
| event-type-vocabulary | capability→producer (wired) — orchestrator/src/orchestrator/event_store.py::EventType and EventStore.emit | PASS | — |
| orchestrator-config-reload-green-tier | capability→producer (wired) — orchestrator/src/orchestrator/config.py::OrchestratorConfig and RELOADABLE_FIELDS (a whole-submodel group takes one RELOADABLE_FIELDS entry) | PASS | — |
| declare-defect-write-from-workflow | producer:α2 upstream — declare_defect; call seam SchedulerFacade.dispatch_tool | PASS | — |
| defects-and-plancheck-types | producer:α1 upstream — PlanCheck, DeclareEvent, Exemption, Withdrawal, Retirement | PASS | — |
| in-task-defects-declared-before-plan-check | producer:δ2 upstream (δ2 is an ε prerequisite) — review-found defects are declared before the replan whose plan ε checks | PASS | — |
| plan-reproduces-predicate | producer:ε — absent on main | PASS | grep `def plan_reproduces_declared_defects` in `orchestrator/src/orchestrator/` |
| reproduces-parameter-on-add-plan-step | producer:ε — the only 'reproduces' on main in plan_tools.py is prose without a colon | PASS | grep `reproduces:` in `orchestrator/src/orchestrator/mcp/plan_tools.py` |
| declare-not-reproducible-plan-tool | producer:ε — absent on main | PASS | grep `def declare_not_reproducible` in `orchestrator/src/orchestrator/mcp/plan_tools.py` |
| declare-withdrawn-plan-tool | producer:ε — absent on main | PASS | grep `def declare_withdrawn` in `orchestrator/src/orchestrator/mcp/plan_tools.py` |
| retire-defect-test-plan-tool | producer:ε — absent on main | PASS | grep `def retire_defect_test` in `orchestrator/src/orchestrator/mcp/plan_tools.py` |
| defect-plan-check-event-type | producer:ε — absent on main | PASS | grep `defect_plan_check` in `orchestrator/src/orchestrator/event_store.py` |
| persisted-plan-repair-counter | producer:ε — absent on main | PASS | grep `defect_plan_repairs_total` in `orchestrator/src/orchestrator/artifacts.py` |
| defect-tests-config-group | producer:β1 upstream — the defect_tests: config group (moved from ε to β, 2026-10-16) | PASS | — |
| simple-path-veto-and-shadow-recording | producer:ε — veto in _should_run_simple_task under enforce; post-hoc recording check in _run_simple_task under shadow | PASS | manual: behaviour of the veto depends on config mode; judged by B2 in shadow and enforce |
| defect-proof-event-type-defined | producer:ε — defines the event type in event_store.py::EventType (moved from ζ2 2026-10-16) | PASS | grep `defect_proof = 'defect_proof'` in `orchestrator/src/orchestrator/event_store.py` |
| defect-keep-check-event-type-defined | producer:ε — defines the event type in event_store.py::EventType (moved from η 2026-10-16) | PASS | grep `defect_keep_check = 'defect_keep_check'` in `orchestrator/src/orchestrator/event_store.py` |
| plan-tools-in-plan-creator-allowlist | producer:ε — the three plan tools join roles.py::_PLAN_CREATOR_TOOLS | PASS | grep `mcp__plan-tools__declare_not_reproducible` in `orchestrator/src/orchestrator/agents/roles.py` |
| declare-defect-exposed-to-fixing-roles | producer:ε — mcp__fused-memory__declare_defect in IMPLEMENTER, DEBUGGER, STEWARD, SIMPLE_TASK allowed_tools; DEFECT_TEST_RULE gains the declaration clause | PASS | grep `mcp__fused-memory__declare_defect` in `orchestrator/src/orchestrator/agents/roles.py` |
| theta1-roles-serialization | producer:θ1 upstream (roles.py serialization γ → δ1 → θ1 → ε) | PASS | — |

## ζ1 — Proof runner: PROOF_BASE worktree, test-side overlay, injected-plugin legs, DefectProof, scripts/prove_defect.py

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| proof-base-worktree-kind | capability→producer (wired) — orchestrator/src/orchestrator/git_ops.py::GitOps.ephemeral_worktree + WorktreeKind (mint, cleanup and the PROTECTED_PREFIXES band are in use by verify.py's main-probe and main-sweep); the PROOF_BASE member is this leaf's own deliverable | PASS | grep `PROOF_BASE` in `orchestrator/src/orchestrator/git_ops.py` |
| proof-base-band-registered-in-protected-prefixes | capability→producer (wired) — git_ops.py::PROTECTED_PREFIXES keys the MAIN_PROBE/MAIN_SWEEP bands by WorktreeKind.*.value (the reaper-protection precedent PROOF_BASE copies) | PASS | grep `WorktreeKind\.PROOF_BASE\.value` in `orchestrator/src/orchestrator/git_ops.py` |
| merge-base-recomputed-at-proof-time | capability→producer (wired) — orchestrator/src/orchestrator/git_ops.py::GitOps.merge_base_with_main (consumed by merge_lane/landing_evidence.py's base-vs-tip differential) | PASS | — |
| ratchet-cluster-raise-recorded-with-the-git-ops-edit | capability→producer (wired) — orchestrator/tests/merge_lane_ratchet_baseline.json params.cluster_paths lists git_ops.py; scripts/merge_lane_metrics.py --authorize-raise appends to orchestrator/tests/merge_lane_ratchet_authorized_raises.json | PASS | manual: the raise record carries the producing task's id, unknown until stamping; test_merge_lane_ratchet is the enforcing gate and goes red without it |
| pytest-classified-command-gate-for-unsupported-runner | capability→producer (wired) — orchestrator/src/orchestrator/verify_cmd.py::parse_config_command / VerifyCmd (ToolKind.PYTEST vs OPAQUE; raw-retained chains keep raw), the classification PROOF_UNSUPPORTED_RUNNER keys on | PASS | — |
| bounded-async-subprocess-runner-and-target-env | capability→producer (wired) — orchestrator/src/orchestrator/verify.py::_run_cmd (async, timeout) and verify.py::_target_subprocess_env (target-venv env scrub); base-tree venv via verify.py::_preprovision_shared_venv (wired at the verify pre-provision site) | PASS | — |
| injected-plugin-outcome-recorder-and-static-enumerator | producer:β1/β2 upstream — β scope covers the single-file plugin (markers, collection map, outcome recorder), the -p injection in verify_cmd.py and shared/src/shared/defect_enumerator.py; record-file location and the recorder's activation switch are not pinned in the PRD, see notes | PASS | — |
| defect-vocabulary-and-record-defect-outcome-write-path | producer:α2 upstream (record_defect_outcome applying Proof and Rule events) over producer:α1 (DefectProof, Ruling, transition table); orchestrator-side call via orchestrator/src/orchestrator/mcp_lifecycle.py::mcp_call (scripts) and scheduler.py SchedulerFacade.dispatch_tool (workflow) — generic tool passthrough, no allowlist | PASS | — |
| defect-tests-config-group-readable-by-the-proof | producer:β1/β2 upstream — the defect_tests: config group (moved to β, 2026-10-16) | PASS | — |
| scripts-prove-defect-cli | own deliverable — scripts/prove_defect.py <task> [--rule <ref> accept\|reject --reason ...] recording through record_defect_outcome | PASS | path present: `scripts/prove_defect.py` |
| operator-ruling-flag-records-ruled-by-operator | own deliverable — the --rule arm of scripts/prove_defect.py (B20) | PASS | grep `--rule` in `scripts/prove_defect.py` |
| proof-outcome-assembly-per-invariant-i4 | own deliverable — DefectProof assembly from the plugin's per-id outcomes (PROVEN, PASSES_AT_BASE, FAILS_AT_TIP, NOT_COLLECTED, MARKER_NOT_STATIC, PROOF_UNAVAILABLE, PROOF_UNSUPPORTED_RUNNER) | PASS | manual: judged by the B1, B3, B4 fixtures on a real fixture branch, not by a grep; the cold and warm proof duration is recorded in the completion note |

## ζ2 — Workflow proof integration: hook after both VERIFY arms, typed replan, persisted counter, streak escalation, soak report

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| proof-hook-site-after-both-verify-arms | capability→producer (wired) — orchestrator/src/orchestrator/workflow.py::TaskWorkflow._execute_verify_review_loop (the _verify_checkpoint_hit arm and the _verify_debugfix_loop arm both exist and are on the production path) | PASS | — |
| typed-replan-cause-rendered-by-build-replan-prompt | capability→producer (wired) — workflow.py::TaskWorkflow._replan and agents/briefing.py::build_replan_prompt (today fed reviews.format_for_replan() as a string) over artifacts.py::ReviewAggregation; ReplanCause = ReviewAggregation \| DefectProof is this leaf's deliverable and the _BriefingLike protocol in workflow.py must change in lockstep | PASS | grep `ReplanCause` in `orchestrator/src/orchestrator/` |
| persisted-task-lifetime-proof-replan-counter | capability→producer (wired) — artifacts.py::TaskArtifacts.set_review_counters / get_review_cycles_total (review_state.json read-modify-write, preserves sibling keys; the task-2749 precedent); ε (upstream) extends it with defect_plan_repairs_total | PASS | grep `defect_proof_replans_total` in `orchestrator/src/orchestrator/artifacts.py` |
| fleet-streak-primitive | capability→producer (wired) — shared/src/shared/streak.py::KeyedStreaks (failure/success/count/forget API complete; no production consumer on main yet, ζ2 is the first) | PASS | grep `KeyedStreaks` in `orchestrator/src/orchestrator/` |
| info-l1-and-l2-escalation-filing-from-workflow | capability→producer (wired) — workflow.py severity='info' escalation sites and the born-at-L2 filing (level=2) sites | PASS | — |
| record-defect-outcome-call-from-workflow | producer:α2 upstream; call path scheduler.py SchedulerFacade.dispatch_tool (generic fused-memory tool passthrough used by chronic_flake.py) | PASS | — |
| proof-runner-and-defect-proof-record | producer:ζ1 upstream (runner, DefectProof assembly, PROOF_BASE worktree) | PASS | — |
| defect-tests-config-and-plan-counter | producer:ε upstream (defect_tests: group with mode, max_proof_replans, proof_timeout_secs, streak_threshold; defect_plan_repairs_total) | PASS | — |
| defect-proof-event-type | producer:ε upstream — the event type is defined by ε (event_store.py has one editor, 2026-10-16); this leaf only emits it | PASS | — |
| soak-report-script-read-only-from-runs-db | capability→producer (wired) — scripts/analyze_speculation_depth.py opens runs.db through a mode=ro SQLite URI (the read-only precedent); the report itself is this leaf's deliverable; it reads defect_plan_check, defect_proof and defect_keep_check by their event_type strings, so η adds no rows | PASS | path present: `scripts/defect_tests_soak_report.py` |
| shadow-records-enforce-acts-governed-by-mode | own deliverable — I5: the proof-driven replan acts only in enforce; PROOF_UNAVAILABLE retry and escalations are mode-independent | PASS | manual: behavioural; judged by the B1, B3, B5 defect_proof events in runs.db and the soak report output |
| process-level-streak-in-harness | capability→producer (wired) — orchestrator/src/orchestrator/harness.py constructs workflows via workflow.py::build_workflow; ζ2 constructs one KeyedStreaks there and passes it in | PASS | — |

## η — Merge presence check and keep-ratchet: merge_lane/defect_checks.py, typed DefectMergeRefusal, defect_keep_check event

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| merge-gate-call-site-with-candidate-and-parent-trees | capability→producer (wired) — orchestrator/src/orchestrator/merge_lane/worker.py::classify_and_merge step 6 drop-guard (merge_lane/gates.py::_check_plan_targets_in_tree is handed merge_commit and base_sha, the speculative or main parent); MergeRequest.config carries OrchestratorConfig; AND merge_lane/worker.py::_do_train_merge per member (trains skip classify_and_merge), each member against its own parent | PASS | — |
| git-object-reads-for-both-trees | capability→producer (wired) — merge_lane/gates.py already runs git ls-tree -r and git show <rev>:<path> against commits, never a working copy | PASS | — |
| live-task-metadata-read-in-merge-lane | capability→producer (wired) — merge_lane/worker.py SpeculativeMergeWorker._scheduler (optional, None on bare-harness workers) with scheduler.get_task as used by reconcile_landed_row; the /unblock merge_request path (escalation/src/escalation/server.py) reaches the same worker; a None scheduler must fail closed | PASS | — |
| typed-merge-outcome-field-for-refusal | capability→producer (wired) — merge_lane/types.py::MergeOutcome (status Literal plus typed side fields such as disposition and failure_category; no defect field yet); the DefectMergeRefusal enum is producer:α1 upstream, the MergeOutcome field is this leaf's deliverable | PASS | grep `DefectMergeRefusal` in `orchestrator/src/orchestrator/merge_lane/types.py` |
| static-enumeration-from-file-contents | producer:β1/β2 upstream (shared/src/shared/defect_enumerator.py with per-file completeness and skip/xfail detection); the PRD does not pin a pure per-file source-text entry point, which η needs to enumerate from git objects, see notes | PASS | — |
| proof-records-and-retirement-records-readable | producer:α1 upstream (DefectProof.node_ids, Retirement, Ruling) and producer:α2 upstream (declare_defect/record_defect_outcome, usable to seed B23's accepted retirement); producer:ζ1 upstream for PROVEN records; event type defect_keep_check from producer:ε upstream | PASS | — |
| defect-tests-mode-readable-at-the-merge-gate | producer:β1/β2 upstream — the defect_tests: config group (moved to β, 2026-10-16) | PASS | — |
| defect-checks-module | own deliverable — orchestrator/src/orchestrator/merge_lane/defect_checks.py, every function at or below cognitive complexity 15 (merge_lane_ratchet_baseline.json params.new_function_cognitive_ceiling); merge_lane/**/*.py is in params.cluster_paths | PASS | path present: `orchestrator/src/orchestrator/merge_lane/defect_checks.py` |
| single-gate-call-into-defect-checks | own deliverable — one call from the merge gate (classify_and_merge or the merge_lane/gates.py chain) into merge_lane/defect_checks.py, its ratchet raise recorded in merge_lane_ratchet_authorized_raises.json | PASS | grep `defect_checks` in `orchestrator/src/orchestrator/merge_lane/worker.py`, `orchestrator/src/orchestrator/merge_lane/gates.py` |
| defect-keep-check-event-type | producer:ε upstream — the event type is defined by ε (event_store.py has one editor, 2026-10-16); this leaf only emits it | PASS | — |
| presence-check-and-keep-ratchet-semantics | own deliverable — I8 presence check, I6 keep-ratchet against the candidate's own parent, KEEP_CHECK_INCOMPLETE on unparseable files, skip/xfail-only carriers counted as removed | PASS | manual: behavioural; judged by B10 to B14, B16 and B23 at the merge lane, not by a grep |
| fleet-streak-injected-into-merge-worker | producer:ζ2 upstream — the process-level KeyedStreaks constructed in harness.py; η injects it into SpeculativeMergeWorker | PASS | — |

## θ2 — Docs and skills: findings-contract pointer, hand-filing recipes, flake-ledger reciprocal sentence

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| findings-contract-section-8-exists | capability→producer (wired) — docs/quality-findings-contract.md section '8. Filed-task metadata and the dedup protocol' | PASS | — |
| instrument-skill-filing-templates-exist | capability→producer (wired) — skills/review/SKILL.md and skills/review/references/phase3-triage.md, skills/hotspot-survey/SKILL.md (submit_task metadata block with x_finding_key), skills/review-all/SKILL.md (Filing paragraph) | PASS | — |
| flake-ledger-out-of-scope-section | capability→producer (wired) — plans/flake-ledger-prd.md section '6. Out of scope' (tracked) | PASS | — |
| task-authoring-hand-filing-home | capability→producer (wired) — docs/task-authoring.md (section 8 vocabulary plus the filing sections); no defects or red-main recipe exists there on main | PASS | — |
| red-main-signature-field-located | capability→producer (wired) — metadata.main_health_signature (with main_health_escalation_id and spawn_context='main_health_auto_heal') is written onto the fix-main task by orchestrator/src/orchestrator/merge_lane/worker.py and orchestrator/src/orchestrator/workflow.py, not onto an escalation record; no field named main_health_signature exists in escalation/src | PASS | — |
| submit-task-accepts-defects | producer:α2 upstream, transitive (θ2 → θ1 → α2) — creation floor accepting open entries; ref grammar finding:<fk>, human:..., redmain:... from producer:α1 upstream | PASS | — |
| filing-site-structured-defects-parameter | producer:θ1 upstream (θ2 prerequisite) — submit_only_instructions defects parameter and the curator carry | PASS | — |
| findings-contract-pointer-sentence | producer:θ2 — section 8 sentence: correctness findings only, ref = finding:<fk>; absent on main | PASS | grep `finding:<fk` in `docs/quality-findings-contract.md` |
| review-skill-points-at-contract | producer:θ2 — skills/review templates point at the findings-contract sentence | PASS | grep `finding:<fk` in `skills/review/` |
| hotspot-survey-skill-points-at-contract | producer:θ2 — skills/hotspot-survey template points at the sentence | PASS | grep `finding:<fk` in `skills/hotspot-survey/` |
| review-all-skill-points-at-contract | producer:θ2 — skills/review-all template points at the sentence | PASS | grep `finding:<fk` in `skills/review-all/` |
| task-authoring-red-main-recipe-names-signature-field | producer:θ2 — hand-filing recipe for red_main naming main_health_signature; absent from docs/task-authoring.md on main | PASS | grep `main_health_signature` in `docs/task-authoring.md` |
| flake-ledger-reciprocal-sentence | producer:θ2 — a defect-marked test is ineligible for any quarantine lane; absent from plans/flake-ledger-prd.md on main | PASS | grep `defect-marked` in `plans/flake-ledger-prd.md` |
| mirror-test-and-review-filing-validates | rejection-mechanism — the mirror test (INV-10 mirror tier) and the /review filing whose defects validates are θ2's own deliverables | PASS | manual: mirror-test file and name are not fixed by the PRD; the signal is a test-suite property |

## ι — Client-feed defect display: cases[].defects and the defects_results fold

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| client-feed-package-contract-and-fold | producer:task-6966 upstream (client-feed workspace member, contract.py, fold.py; status in-progress, not cancelled; reached through 6969 and 7002); client-feed/ does not exist on main yet | PASS | — |
| test-results-cases-from-archived-junit | producer:task-6969 upstream (sources/archive.py: junit to test_results cases [id, outcome, duration_ms, criterion, exception_type], not_recorded when absent; status pending) | PASS | — |
| publisher-loop-and-derived-event-emission | producer:task-6970 upstream through 6972 (publisher.py one-pass loop, feed_io, heartbeat degraded list); status pending | PASS | — |
| map-model-with-defects-field-in-the-junit-join | producer:task-6972 upstream (amended 2026-10-16 to read β's <junit>.tests.json map; 6972 depends on β) for the criterion half; ι itself reads the map's defects field in criteria.py | PASS | — |
| tests-json-map-written-beside-junit-by-the-plugin | producer:β1 upstream (plugin writes <junit>.tests.json beside --junitxml, mode not off) | PASS | — |
| tests-json-map-archived-beside-the-archived-junit | producer:β1 upstream — β archives the sibling <junit>.tests.json beside the archived junit (verify.py::_archive_junit_report) | PASS | — |
| merge-run-junit-archived-at-full-merge-breadth | capability→producer (wired) for dark-factory — dark-factory-orchestrator.yaml sets merge_verify_breadth full and verify.py::_JunitReportPlan kind 'attributing' archives the merge-run junit; for client projects producer:task-7008 (pending, flips the shipped default scoped to full) | PASS | — |
| defect-entries-readable-as-typed-fields | producer:α1 upstream (metadata.defects submodel in shared/src/shared/task_metadata.py, importable by client-feed per 6966's import rule); client-feed sources/tasks.py (6968, in closure via 6970) projects an EXACT metadata allowlist that excludes defects, so the typed defects projection is this leaf's own extension | PASS | grep `defects` in `client-feed/src/client_feed/sources/tasks.py` |
| defects-results-fold-and-cases-defects-field | own deliverable — contract.py cases[].defects and the defects_results event plus fold, typed fields only (ref, node ids, outcome, exemption or withdrawal reason, owner); never summary | PASS | grep `defects_results` in `client-feed/src/client_feed/` |
| not-recorded-distinct-from-zero-defects | own deliverable — a task or run with no recorded defect map renders not_recorded, distinct from an empty defects list (decision 18) | PASS | manual: judged by fixtures with and without a map; not_recorded already appears in 6969's archive reader, so a grep would not isolate this leaf's work |
| latest-real-run-reports-ref-or-explicit-zero | own signal — a run over the latest real dark-factory merge-run junit and map reports a real ref or an explicit zero distinct from not_recorded (the observed real row moved to ξ) | PASS | manual: judged by the run output; zero is legitimate before the first proven defect |

## κ — Debugger-edit fact to the reviewer: edited_failing_tests with could_not_determine

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| task-verify-cost-junit-written-and-archived | capability→producer (wired) — orchestrator/src/orchestrator/verify.py::_JunitReportPlan kind 'cost' (task role with archive_root; written at _task_junit_report_path, archived by _archive_junit_report, never parsed today); workflow.py passes archive_root=<project_root>/data/verify-logs at the task-verify call site | PASS | — |
| junit-failing-id-parser | capability→producer (wired) — verify.py::_extract_failing_test_ids_from_junit (private; ET.parse, no gzip, returns None on a missing or malformed report) and the id spelling _junit_testcase_id (classname::name, not a file path); the 'cost' report is parsed here for the first time | PASS | — |
| debugger-per-iteration-changed-files | capability→producer (wired) — git_ops.py::GitOps.get_changed_files(from_sha, to_sha) (used by workflow.py inter-iteration rebase) with workflow.py::_get_head_commit; debugger dispatch at workflow.py::_verify_debugfix_loop (self._invoke(DEBUGGER, ...)) | PASS | — |
| reviewer-prompt-builder-accepts-the-fact | capability→producer (wired) — agents/briefing.py::build_reviewer_prompt (keyword-only amendment_suggestions and task today) called from the single site workflow.py::_run_reviewer; the _BriefingLike protocol in workflow.py mirrors its signature and changes in lockstep; δ2 (upstream) reworks the same briefing | PASS | — |
| edited-failing-tests-parameter-and-state | own deliverable — edited_failing_tests with a could_not_determine state passed to build_reviewer_prompt and rendered in the persisted reviewer briefing | PASS | grep `edited_failing_tests` in `orchestrator/src/orchestrator/agents/briefing.py`, `orchestrator/src/orchestrator/workflow.py` |
| could-not-determine-state | own deliverable — the explicit unknown state when the cost junit is absent, malformed, or its classname ids cannot be mapped to the debugger's changed files | PASS | grep `could_not_determine` in `orchestrator/src/orchestrator/` |

## λ1 — Integration gate, workflow seam: B1–B9, B15, B17–B22

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| workflow-seam-hooks-for-b1-to-b9 | producer:δ2, producer:ε, producer:ζ2 upstream (in-task declarations, plan check and mode, proof hook and replan), over producer:ζ1 (proof runner) and producer:δ1 (reviewer contract, B7); all in λ's transitive closure | PASS | — |
| filing-seam-for-b15 | producer:θ1 upstream (curator combine onto a pending target) | PASS | — |
| injection-seam-for-b18-b19-b21-b22 | producer:β1 upstream (plugin, enumerator, -p injection, PYTHONPATH entry) and producer:ζ1 (MARKER_NOT_STATIC and PROOF_UNSUPPORTED_RUNNER outcomes); reached through ζ2/η | PASS | — |
| authority-seam-for-b17-and-b20 | producer:α2 upstream (record_defect_outcome in no role's allowed_tools drift test; update_task rejection) and producer:ζ1 (scripts/prove_defect.py --rule) | PASS | — |
| fake-agent-and-git-fixture-harness | capability→producer (wired) — orchestrator/tests/_workflow_helpers.py::FakeScheduler and the workflow test helpers; the fixture pytest workspace and the non-workspace pytest fixture project are this leaf's own deliverable | PASS | — |
| boundary-module-workflow | own deliverable | PASS | path present: `orchestrator/tests/test_defect_tests_boundary_workflow.py` |

## λ2 — Integration gate, merge seam: B10–B14, B16 (real train), B23

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| merge-seam-hooks-for-b10-to-b16-and-b23 | producer:η upstream (presence check, keep-ratchet, typed refusal); the speculative-train harness is orchestrator/tests/_merge_queue_harness.py (wired, in the ratchet cluster) | PASS | — |
| unblock-path-for-b14 | capability→producer (wired) — escalation/src/escalation/server.py builds MergeRequest for the merge_request MCP tool, which lands on the same merge worker η instruments | PASS | — |
| boundary-module-merge | own deliverable | PASS | path present: `orchestrator/tests/test_defect_tests_boundary_merge.py` |

## μ — Human gate: flip defect_tests.mode to enforce

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| soak-report-available-to-the-decider | producer:ζ2 upstream (scripts/defect_tests_soak_report.py) via λ1/λ2; metadata.milestone delayed 14 days | PASS | — |
| mode-flip-is-hot-reloadable | producer:ε upstream (defect_tests: group in the reload green tier) over orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS and the mcp__escalation__reload_config tool (wired) | PASS | — |
| enforce-flip-decision | own deliverable — a deterministic always_escalates gate; no code | PASS | manual: Leo decides whether to flip defect_tests.mode to enforce after the soak criteria (14 days and 10 proofs or 60 days, no unexplained PROOF_UNAVAILABLE streak, hand audit of 5 PROVEN records and every PASSES_AT_BASE); the gate has no code deliverable |

## ν — Human gate: author the client-feed PRD revision (defect_reported inbox event and privileged filer)

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| client-origin-declaration-surface | producer:α2 upstream (declare_defect and submit_task accept origin client with ref portal:<project>:<block>:<n>) over producer:α1 (origin and ref grammar) | PASS | — |
| client-feed-revision-and-portal-amendment | own deliverable — a deterministic always_escalates gate; no code | PASS | manual: Leo authors the client-feed PRD revision (defect_reported inbox event, privileged filer outside the publisher's read-only boundary, calling submit_task/declare_defect with origin client) and the portal PRD's one-line amendment exposing report id and criterion id; the gate has no code deliverable |

## ξ — Human gate: restore the LDE sheet's per-bug sentence?

| Capability | Binding | Verdict | Delivered check |
|---|---|---|---|
| soak-report-and-client-display-available-to-the-decider | producer:μ upstream (enforce flipped after the soak, report from ζ2), producer:ι upstream (client feed displays defect ref, test ids, outcome) and producer:task-7008 (pending; full merge verify as the shipped default, the precondition for 'run on every later version') | PASS | — |
| restore-sheet-sentence-decision | own deliverable — a deterministic always_escalates gate; no code | PASS | manual: Leo decides whether about-lde.md (scoping-intake repo) may again say 'every bug or gap found during the build gets its own test, kept and run on every later version'; the gate has no code deliverable |
| observed-real-defect-row-and-live-mode | own — proceed only if the live yaml shows defect_tests.mode enforce; check an observed (not seeded) real defect row on the client display | PASS | manual: human gate |
