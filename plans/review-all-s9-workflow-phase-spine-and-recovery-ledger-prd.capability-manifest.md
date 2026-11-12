# Capability manifest — S9 workflow phase spine and recovery ledger

PRD: `plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md`. YAML twin: `plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.capability-manifest.yaml`.

**Code anchors** verified against main `e634cfe658` (2026-11-10) and re-checked by the fold-verify pass at `bb1c659562` (same day; every producer still fails, every substrate still passes). Main moves fast — cite-by-symbol; re-locate lines at implementation time. Authored at `ce0f71a384`, folded three times on 2026-11-02 at `e87d551bf9`, and folded for the 2026-11-10 sitting's rulings, which added π and its C10 checks. Bindings cite symbols, not lines. Every producer check FAILS at `e634cfe658` and passes once its leaf lands; every substrate check passes today and says so (all 78 mechanical checks re-run with `git grep -E` / `git ls-tree` at `e634cfe658`). No producer check is satisfiable by an upstream sibling or an earlier S9 leaf alone: `no-finalization-backfill` is α's (α removes the backfill); λ1's check targets the OSError re-switch arm (S4 α repoints the `_is_infra_oserror` import). ι checks `corroborated_landing`, not the absence of `MergeProvenance` (S6 γ1 deletes it). μ1/μ2 check code shapes, not the word `MagicMock` (INV-10). λ2's `transitional-test-reader-gone` and ζ1's `storm-sampler-reads-restore-outcome` are manual: the helper does not exist until δ adds it, and S8 γ moving the restore set could turn a grep on the sampler green. ε's `escalation-scoped-envelope` became a substrate check when task 5563 landed (`3fd6ccb3dd`).

## π — The L0 settle check script: stranded and wrongly dismissed blocking L0s from the escalation archive

Lands C10. Contributes fk-bab6058b23c8, fk-e39550272df9 (measures ε; closes neither). No dependencies; ε depends on π. Ruled: OQ-5 plus Leo's check (sitting key S9-7). The lead files the 14-day check task once π is on main (lead action).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `l0-settle-check-script` | capability→producer (π) — scripts/check_l0_settle_outcomes.py, read-only over the escalation archive (C10) | path `scripts/check_l0_settle_outcomes.py` → present | PASS |
| `check-thresholds-cli` | capability→producer (π) — the ruled threshold flags the lead-filed check task passes (C10) | grep `max-reaper-wrong` `scripts/check_l0_settle_outcomes.py` → present | PASS |
| `check-script-tested` | capability→producer (π) — boundary row 26 over an archive fixture | path `scripts/tests/test_check_l0_settle_outcomes.py` → present | PASS |
| `escalation-archive-substrate` | substrate (PASS today) — escalation.archive keeps resolved records 30 days (prune_archive) | grep `def prune_archive` `escalation/src/escalation/archive.py` → present | PASS |
| `deterministic-guard-substrate` | substrate (PASS today) — the deterministic task guard refuses a missing or non-executable before_done.script, so the check task is filed only after π lands | grep `os\.X_OK` `fused-memory/src/fused_memory/middleware/deterministic_task_guard.py` → present | PASS |
| `check-task-filed` | manual — lead action: the deterministic delayed-milestone check task (C10 fields) is filed once π is on main, depending on ε | manual: a task-store fact, not a tree property; the lead files it | PASS |

## α — Typed Plan model; every plan.json writer through it; structural edits clear confirmation; the replan prompt confirms

Lands C1. Contributes fk-37190b541196 (β closes). No dependencies (task 3155, the draft's gate, landed ef96d32603). Ruled: OQ-1, Q-C1 (a).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `plan-state-type` | capability→producer (α) — artifacts.PlanState derived from the typed Plan lifecycle | grep `class PlanState` `orchestrator/src/orchestrator/artifacts.py` → present | PASS |
| `structural-edit-clears-confirmation` | capability→producer (α) — Plan.restructured, used by the step-set plan-tools mutators only (files-only edits keep confirmation) [Q-C1] | grep `def restructured` `orchestrator/src/orchestrator/artifacts.py` → present | PASS |
| `per-pass-confirmation-fact` | capability→producer (α) — Plan.confirmed_since(pre), the typed per-pass fact fingerprint minting reads (critic2 N5) | grep `def confirmed_since` `orchestrator/src/orchestrator/artifacts.py` → present | PASS |
| `non-repairing-plan-read` | capability→producer (α) — TaskArtifacts.peek_plan_model mirrors peek_plan (task 4052): normalized, never written | grep `def peek_plan_model` `orchestrator/src/orchestrator/artifacts.py` → present | PASS |
| `no-finalization-backfill` | anti-inversion — stamp_plan_provenance builds Plan.stamped; the setdefault backfill is gone (moved here from β: α removes it) | grep `setdefault\('_finalized_at'` `orchestrator/src/orchestrator/` → absent | PASS |
| `plan-tools-writes-through-model` | anti-inversion — plan_tools (incl. _read_plan_repaired) and plan_markup_stamp write plan.json only through write_plan_model (write_plan_model( does not match) | grep `\.write_plan\(` `orchestrator/src/orchestrator/mcp/plan_tools.py`, `orchestrator/src/orchestrator/mcp/plan_markup_stamp.py` → absent | PASS |
| `plan-tools-no-hand-marker-writes` | anti-inversion — plan_tools confirms through Plan.confirmed, never by subscript | grep `\['_finalized_at'\]` `orchestrator/src/orchestrator/mcp/plan_tools.py` → absent | PASS |
| `plan-tools-already-imports-artifacts` | substrate (PASS today) — plan_tools.py imports TaskArtifacts; the MCP process already loads the model module | grep `from orchestrator.artifacts import TaskArtifacts` `orchestrator/src/orchestrator/mcp/plan_tools.py` → present | PASS |
| `replan-prompt-confirms` | manual — BriefingAssembler.build_replan_prompt ends in confirm_plan (function-scoped; not greppable to one function) [Q-C1] | manual: read build_replan_prompt after merge; boundary row 4 drives the real tool sequence | PASS |
| `live-plan-carries-confirmed-by` | INV-13 producer proof — a live architect run writes _confirmed_by via confirm_plan | manual: live plan.json read after merge; not a tree property | PASS |

## β — Every plan reader on PlanState; finalization a precondition at every stamp site; recovered plans keep their lifecycle

Closes fk-37190b541196. Depends on α. Tasks 5645 and 7345 depend on β. Ruled: OQ-2.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `workflow-reads-plan-state` | anti-inversion — workflow.py reads Plan.state, not marker keys | grep `get\('_session_id'\)\|get\('_finalized_at'\)` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `harness-stops-predicting-plan-branch` | anti-inversion — Harness._recover_crashed_tasks reads Plan.state | grep `plan\.get\('_session_id'\)` `orchestrator/src/orchestrator/harness.py` → absent | PASS |
| `recovered-plan-origin` | capability→producer (β) — TaskWorkflow takes initial_plan_origin; a recovered plan keeps its lifecycle (D3) | grep `initial_plan_origin` `orchestrator/src/orchestrator/workflow.py` → present | PASS |
| `meta-root-locator-moved` | anti-inversion — evals import artifacts.meta_root_for_worktree, not a workflow.py private | grep `def _meta_root_for_worktree` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `plan-stamp-readers-folded` | anti-inversion — the module readers plan_base_commit/plan_task_text_fp become Plan.lifecycle fields (task 3157 added them after the draft) | grep `^def plan_(base_commit\|task_text_fp)\b` `orchestrator/src/orchestrator/artifacts.py` → absent | PASS |
| `unconfirmed-pass-not-executed` | rejection-mechanism — produced by α+β; boundary rows 4/4b drive the real replan/revalidation tool sequence cut off before confirm; 3c keeps a tightening pass STAMPED; 4c keeps the fingerprint | manual: behavioural: boundary rows 3c, 4, 4b, 4c, 5b, 5c, 5d | PASS |

## γ — IterationEntry union with writer-decided is_work; the completion judge reads typed entries

Closes fk-9973521f4f59. Depends on β; lands before S5 ν on read_iteration_log. Subsumes task 7391.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `iteration-entry-union` | capability→producer (γ) — artifacts.IterationEntry | grep `^(type )?IterationEntry\b[^=]*=` `orchestrator/src/orchestrator/artifacts.py` → present | PASS |
| `label-classifier-deleted` | anti-inversion — readers consume entry.is_work | grep `def _iteration_entry_is_work` `orchestrator/src/orchestrator/` → absent | PASS |
| `debugger-attempt-not-iteration` | anti-inversion — the debugger kind writes verify_attempt | grep `'iteration': verify_attempt` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `judge-prompt-excludes-agent-lines` | rejection-mechanism — produced by γ; boundary row 8b | manual: behavioural: boundary row 8b | PASS |
| `source-guard-substrate` | substrate (PASS today) — append_iteration_log refuses a source outside ORCHESTRATOR_ITERATION_SOURCES (task 5726) | — | PASS |

## δ — BlockReason and the one-argument _mark_blocked; report-handler and recovery families converted

Lands C3. Closes fk-d2ba7601c130; contributes fk-b8efaa973162 (λ2 closes). Depends on γ and S5 ν (typed exits) [Q-C2].

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `block-reason-type` | capability→producer (δ) — workflow_types.BlockReason | grep `class BlockReason` `orchestrator/src/orchestrator/workflow_types.py` → present | PASS |
| `block-route-replaces-two-flags` | capability→producer (δ) — BlockRoute makes the contradictory pair unrepresentable | grep `class BlockRoute` `orchestrator/src/orchestrator/workflow_types.py` → present | PASS |
| `registered-category-vocabulary` | substrate (PASS today) — escalation.categories.EscalationCategory holds every category the workflow files | grep `class EscalationCategory` `escalation/src/escalation/categories.py` → present | PASS |
| `typed-architect-exits-upstream` | producer: S5 ν (D13; _read_model and five *Exit dataclasses) — upstream of δ; edge at decompose [Q-C2] | — | PASS |
| `merge-phase-derivation-premise` | OPEN — δ proves (D7) every merge_phase=True path runs in WorkflowState.MERGE, else BlockReason gains a phase field | manual: reviewer confirms δ's tests cover every merge_phase=True path in WorkflowState.MERGE, or the phase field | OPEN |

## ε — The steward decides every published variant; the workflow believes it; one L0 settle

Lands C4. Closes fk-e39550272df9, fk-bab6058b23c8. Depends on δ and π (task 5563, the draft's other dependency, landed 3fd6ccb3dd). Ruled: OQ-5 (S9-7).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `consumer-override-removed` | anti-inversion — _await_steward_completion returns a published outcome verbatim | grep `_outcome_requeues\(` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `producer-publishes-terminal-decision` | capability→producer (ε) — TaskSteward publishes StewardTerminalDecision after a session that set the row terminal | grep `StewardTerminalDecision\(` `orchestrator/src/orchestrator/steward.py` → present | PASS |
| `one-l0-settle` | capability→producer (ε) — the one workflow dismissal site with a per-disposition scope | grep `def _settle_own_l0s` `orchestrator/src/orchestrator/workflow.py` → present | PASS |
| `degraded-never-terminal` | rejection-mechanism — a failed row read never yields StewardTerminalDecision; the open-L1 arm applies only to the _outcome_requeues set (budget exhaustion exits BLOCKED, row 14d); the reads follow a chained follow-on's pre-session give-up (row 15c) | manual: behavioural: boundary rows 14b, 14c, 14d, 15b, 15c, 16b | PASS |
| `escalation-scoped-envelope` | substrate (PASS today) — PublishedStewardOutcome(escalation_id, outcome) and the scoped wait landed with task 5563 (3fd6ccb3dd) | grep `class PublishedStewardOutcome` `orchestrator/src/orchestrator/workflow_types.py` → present | PASS |
| `fourteen-day-l0-check` | manual — S9-7 ruling: the lead-filed C10 check task, 14 days after ε is done, exits 0 | manual: a deterministic milestone task's outcome 14 days after landing; not a tree property | PASS |

## ζ1 — _invoke in three parts: ResumeArm, invocation, accounting

Lands C7. Closes fk-2c0594928a18; contributes fk-7d410939a337 and fk-f37f7e253cd5 (branch 5). Depends on ε.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `restore-outcome-enum` | capability→producer (ζ1) — RestoreOutcome replaces the four-valued string | grep `class RestoreOutcome` `orchestrator/src/orchestrator/` → present | PASS |
| `restore-by-design-one-home` | capability→producer (ζ1) — RestoreOutcome.is_by_design replaces the harness set | grep `def is_by_design` `orchestrator/src/orchestrator/` → present | PASS |
| `restore-set-has-one-home` | anti-inversion — no string set restates the by-design restore outcomes anywhere in orchestrator/src (scoped wide so S8 γ moving the set to session_resume_storm.py cannot turn it green) | grep `BY_DESIGN_RESTORE_OUTCOMES: frozenset` `orchestrator/src/orchestrator/` → absent | PASS |
| `restore-string-gone` | anti-inversion — _invoke reads ResumeArm.restore | grep `restore_outcome = 'disabled'` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `stand-in-result-getattr-gone` | anti-inversion — branch 5: accounting reads the typed AgentResult | grep `getattr\(result, 'resume_fallback_session_ids'` `orchestrator/src/orchestrator/` → absent | PASS |
| `storm-sampler-reads-restore-outcome` | rejection-mechanism — storm_window_bound.py stops importing the harness private and still carves out disabled and miss (missed by the draft; no grep: S8 γ moving the set could turn one green) | manual: behavioural: python -c import orchestrator.storm_window_bound; its sampler carve-out equals the by-design members | PASS |

## ζ2 — One route resolver with one scope-capacity read; MCP-injection helpers leave workflow.py

Closes fk-5ea52a0f89ad; contributes fk-a503ccb37b4b. Depends on ζ1 and task 6766 (S25 C4). Ruled: OQ-7 (lead).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `resolver-takes-usage-gate` | capability→producer (ζ2) — resolve_and_record_route computes scope_capacity from an optional usage_gate (D12) | grep `usage_gate` `orchestrator/src/orchestrator/routing_dispatch.py` → present | PASS |
| `in-band-route-copy-gone` | anti-inversion — _invoke calls routing_dispatch.resolve_and_record_route | grep `def _record_routing_decision` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `routing-helper-copies-gone` | anti-inversion — the mirrored helpers and the in-band capacity read exist only once | grep `def (_trailing_24h_window\|_routing_inputs_digest\|_ceiling_spend_by_model\|_scope_capacity_snapshot)` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `steward-no-workflow-private-import` | anti-inversion — steward imports _inject_verdict_tools_mcp from its new home at module level | grep `from orchestrator.workflow import _inject_verdict_tools_mcp` `orchestrator/src/orchestrator/steward.py` → absent | PASS |
| `mcp-composer-moved` | anti-inversion — compose_invocation_mcp_config (task 5386) moves to mcp_lifecycle.py with the inject helpers it composes | grep `def compose_invocation_mcp_config` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `out-of-band-resolver-substrate` | substrate (PASS today) — routing_dispatch.resolve_and_record_route exists | grep `async def resolve_and_record_route` `orchestrator/src/orchestrator/routing_dispatch.py` → present | PASS |

## η — Merge-side blocks from OutcomeKind; the merge stash goes

Contributes fk-d2ba7601c130, fk-44bc973b1cc4, fk-7d410939a337, fk-b8efaa973162. Depends on ζ2, S6 C1 (β1), S6 C3 (δ), S6 C7 (θ), task 4838.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `outcome-kind-block-table` | capability→producer (η) — one OutcomeKind → BlockReason table | grep `OUTCOME_KIND_BLOCK` `orchestrator/src/orchestrator/workflow.py` → present | PASS |
| `merge-stash-gone` | anti-inversion — the thrash signature reads MergeOutcome directly | grep `_last_merge_(block_reason\|failure_category\|failure_cause_hint)` `orchestrator/src/orchestrator/` → absent | PASS |
| `merge-outcome-kind-upstream` | producer: S6 C1 (MergeOutcome.kind, S6 β1) — upstream of η; edge at decompose | — | PASS |
| `halt-owner-upstream` | producer: S6 C7 (HaltState owner semantics, S6 θ) — upstream of η; edge at decompose | — | PASS |

## θ — Merge-done classification and the stranded_merge_failed filer in stranded_verified_green.py

Lands C5. Closes fk-108cdd00d122 (pairs 1–2). Depends on η, S6 C1 (β1), task 7277. 3354 and S8 ε2b depend on θ (ε2b then moves the re-pointed Harness._on_stranded_merge_done into stranded_reconcile.py). Subsumes 7985.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `shared-classifier` | capability→producer (θ) — classify_merge_future routes on MergeOutcome.kind | grep `def classify_merge_future` `orchestrator/src/orchestrator/stranded_verified_green.py` → present | PASS |
| `shared-filer` | capability→producer (θ) — one stranded_merge_failed filer | grep `async def file_stranded_merge_failed` `orchestrator/src/orchestrator/stranded_verified_green.py` → present | PASS |
| `workflow-twin-filer-gone` | anti-inversion — the workflow copy is deleted | grep `def _file_architect_merge_failed` `orchestrator/src/orchestrator/` → absent | PASS |
| `harness-twin-filer-gone` | anti-inversion — the harness copy is deleted | grep `def _file_stranded_merge_failed` `orchestrator/src/orchestrator/` → absent | PASS |
| `submit-path-substrate` | substrate (PASS today) — submit_verified_green_merge_request already shared by both callers | grep `async def submit_verified_green_merge_request` `orchestrator/src/orchestrator/stranded_verified_green.py` → present | PASS |

## ι — One corroborated landing reader for the recovery guards and the ground-truth journal rung

Lands C6. Closes fk-a152341e38ed. Depends on θ, S6 C4 (γ1), task 7277. Subsumes 7769.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `corroborated-landing-reader` | capability→producer (ι) — corroborated_landing checks is_ancestor_checked before trusting a row | grep `async def corroborated_landing` `orchestrator/src/orchestrator/` → present | PASS |
| `guards-call-the-reader` | capability→consumer wiring (ι) — the three workflow guards call it | grep `corroborated_landing\(` `orchestrator/src/orchestrator/workflow.py` → present | PASS |
| `ground-truth-rung-calls-the-reader` | capability→consumer wiring (ι) — task_ground_truth._resolve_branch_state journal rung | grep `corroborated_landing\(` `orchestrator/src/orchestrator/task_ground_truth.py` → present | PASS |
| `ancestry-substrate` | substrate (PASS today) — GitOps.is_ancestor_checked (True/False/None) | grep `async def is_ancestor_checked` `orchestrator/src/orchestrator/git_ops.py` → present | PASS |
| `injected-outbox-upstream` | producer: S6 C4 (LandedOutbox injected; MergeProvenance deleted, S6 γ1) — upstream of ι; edge at decompose | — | PASS |

## κ — Execute, verify and planning legs return BlockReason; the stash dicts go

Closes fk-44bc973b1cc4, fk-7d410939a337. Depends on ι, S4 C2 (α) and S4 C1 (β); keeps S4 α's bare-crash rule (S4-5 ruling).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `stash-dicts-gone` | anti-inversion — each leg returns its BlockReason | grep `_infra_hold_info\|_inherited_break_info\|_zero_output_hang_info\|_progress_resume_churn_info` `orchestrator/src/orchestrator/` → absent | PASS |
| `verify-transients-gone` | anti-inversion — the VERIFY→REVIEW emission takes the tip and the verdict runner as arguments (task 6670 added _verify_green_runner after the draft) | grep `_verify_green_result\|_verify_green_runner\|_verify_checkpoint_hit\|_last_verify_result` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `typed-verify-result-upstream` | producer: S4 C2 (α: infra_transient property, VerifyInfraError retired) and S4 C1 (β: VerifyContext at the two call sites) — upstream of κ; edges at decompose | — | PASS |

## λ1 — The _drive ladder onto disposition rows; MetadataWriteFailed blocks; two dead columns go

Closes fk-4f7a933c0dd4, fk-67b5971ed72d; contributes fk-b8efaa973162. Depends on κ, S5 C3 (δ), S10 C2 (β2), S4 C2 (α).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `oserror-arm-is-a-row` | anti-inversion — the infra-OSError re-switch arm becomes a table row (retargeted: S4 α repoints the import, so the import is not λ1-specific) | grep `isinstance\(e, OSError\) and _?is_infra_oserror` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `dead-block-class-column-dropped` | anti-inversion — BlockDisposition loses block_class (always AGENT_FAILURE) | grep `block_class=BlockClass` `orchestrator/src/orchestrator/workflow_types.py`, `orchestrator/src/orchestrator/block_disposition.py`, `orchestrator/src/orchestrator/git_ops.py` → absent | PASS |
| `terminal-report-dead-category-gone` | anti-inversion — TerminalReport.category (set from the dead column, read nowhere) is deleted | grep `category: FailureCategory \\| None` `orchestrator/src/orchestrator/workflow_types.py` → absent | PASS |
| `metadata-write-failed-upstream` | producer: S10 C2 (MetadataWriteFailed from Scheduler.update_task, S10 β2) — upstream of λ1; edge at decompose | — | PASS |
| `block-disposition-home-upstream` | producer: S5 C3 (block_disposition.py, derived warm-lane rows, S5 δ) — upstream of λ1; edge at decompose | — | PASS |
| `verify-vocabulary-upstream` | producer: S4 C2 (α: VerifyInfraError arm and row removed; is_infra_oserror public) — upstream of λ1; edge at decompose | — | PASS |

## λ2 — The keyword form is deleted; the remaining keyword-form tests migrate

Closes fk-b8efaa973162. Depends on λ1.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `keyword-form-deleted` | anti-inversion — _mark_blocked takes only BlockReason; the transitional adapter is gone | grep `escalate_to_human: bool = False\|skip_escalation: bool = False\|from_legacy_kwargs` `orchestrator/src/orchestrator/` → absent | PASS |
| `transitional-test-reader-gone` | anti-inversion — tests assert on TerminalReport and the filed record; δ's transitional blocked_with helper is deleted (no grep: the helper does not exist until δ, so an absent-check would pass today) | manual: read _workflow_helpers.py after λ2; the suite asserts on TerminalReport/the filed record | PASS |

## μ1 — Protocol-typed seams, hand-written fakes, conformance; fixture-tolerance branches deleted

Lands C8 ports. Contributes fk-f37f7e253cd5 (branches 3, 6, 7). Depends on λ2, S10 β1 (SchedulerFacade/FakeScheduler parity).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `steward-port` | capability→producer (μ1) — StewardPort Protocol | grep `class StewardPort` `orchestrator/src/orchestrator/` → present | PASS |
| `fixture-tolerance-gone` | anti-inversion — branches 3 and 6 (code shape, not comments) | grep `getattr\(self, '_process_run_id'\|getattr\(steward, 'metrics'` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `test-only-attribute-gone` | anti-inversion — branch 7: _merge_recovery_basis deleted; tests read mark_done kind | grep `_merge_recovery_basis` `orchestrator/src/orchestrator/` → absent | PASS |
| `dry-run-scheduler-typed` | anti-inversion — run_dry_run_unblock types its scheduler with DryRunScheduler | grep `scheduler: Any` `orchestrator/src/orchestrator/dry_run_unblock.py` → absent | PASS |
| `scheduler-facade-substrate` | substrate (PASS today) — scheduler.SchedulerFacade runtime_checkable Protocol with test-side conformance blocks | grep `class SchedulerFacade` `orchestrator/src/orchestrator/scheduler.py` → present | PASS |

## μ2 — A spec'd scheduler double replaces the bare mocks; the double-tolerance branches go

Contributes fk-f37f7e253cd5 (branches 4, 9, the dry-run trims, the is-True check; μ3 closes). Depends on μ1. Subsumes 6962. Ruled: Q-C4 (a).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `spec-scheduler-helper` | capability→producer (μ2) — orchestrator/tests/_orch_helpers.py::spec_scheduler | grep `def spec_scheduler` `orchestrator/tests/_orch_helpers.py` → present | PASS |
| `mock-tolerance-gone` | anti-inversion — the is-True identity check (code shape); branch 1 is μ3's | grep `blocked_writes_suppressed\(self\.task_id\) is True` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `liveness-mock-precedent-substrate` | substrate (PASS today) — _orch_helpers.wire_scheduler_liveness_mock is the spec'd-default precedent | grep `def wire_scheduler_liveness_mock` `orchestrator/tests/_orch_helpers.py` → present | PASS |
| `exit-contract-still-judges` | rejection-mechanism — a migrated suite's exit is judged (boundary row 23b); a bare MagicMock would blind it | manual: behavioural: boundary row 23b | PASS |

## μ3 — A spec'd config double replaces the bare config mocks; branch 1 goes

Closes fk-f37f7e253cd5 (branch 1, after μ1/μ2/ζ1 delete the rest). Depends on μ2 (same files). Ruled: Q-C4 (a).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `spec-config-helper` | capability→producer (μ3) — orchestrator/tests/_orch_helpers.py::spec_config (real OrchestratorConfig with overrides) | grep `def spec_config` `orchestrator/tests/_orch_helpers.py` → present | PASS |
| `mock-config-tolerance-gone` | anti-inversion — branch 1: the MagicMock-interval isinstance guard in _claimant_heartbeat_loop (code shape) | grep `isinstance\(interval, \(int, float\)\)` `orchestrator/src/orchestrator/workflow.py` → absent | PASS |
| `heartbeat-runs-on-real-config` | rejection-mechanism — boundary row 23c: a migrated suite runs the heartbeat on a real interval with no background TypeError | manual: behavioural: boundary row 23c | PASS |

## ν — WorkflowStateMachine is the single writer of phase and row

Lands C9. Closes fk-ee708be4436f; contributes fk-f37f7e253cd5 (branch 8). Depends on μ3. Ruled: OQ-3, Q-C5 (b).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `machine-enter` | capability→producer (ν) — WorkflowStateMachine.enter writes the projected row | grep `async def enter\(` `orchestrator/src/orchestrator/workflow_types.py` → present | PASS |
| `write-failure-recorded` | capability→producer (ν) — machine.last_write_failure feeds the judge's STORE_UNAVAILABLE relaxation (ruled: Q-C5 (b), record and keep going) | grep `def last_write_failure` `orchestrator/src/orchestrator/workflow_types.py` → present | PASS |
| `phase-comparison-gone` | anti-inversion — the machine holds report and state, so the judge's phase comparison is deleted (retargeted from _note_exit_status_write) | grep `report_phase != machine_state` `orchestrator/src/orchestrator/exit_contract.py` → absent | PASS |
| `force-set-and-stash-gone` | anti-inversion — tests pass initial_state=; run() returns machine.terminal_report | grep `def force_set\|self\._terminal_report = ` `orchestrator/src/orchestrator/` → absent | PASS |
| `write-only-on-change` | rejection-mechanism — enter writes the row only when the projection changes; a harness write during EXECUTE survives the move to VERIFY (row 24c, INV-6) | manual: behavioural: boundary row 24c | PASS |
| `done-write-outside-machine` | rejection-mechanism — done and cancelled writes stay outside the machine and keep Scheduler.mark_done's raise-on-failure (S9-5 contract sentence, row 24d) | manual: behavioural: boundary row 24d | PASS |

## ξ — Retry budgets: one run-length rule on shared.streak, one steward guard clear, a typed first-enqueue time

Closes fk-9eb149f0706d (narrow). Depends on ν, S10 C3 (η), task 6724. Ruled: OQ-4.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `one-run-length-rule` | capability→producer (ξ) — shared.streak.run_length beside KeyedStreaks | grep `def run_length` `shared/src/shared/streak.py` → present | PASS |
| `one-steward-guard-clear` | capability→producer (ξ) — TaskSteward._clear_guards | grep `def _clear_guards` `orchestrator/src/orchestrator/steward.py` → present | PASS |
| `verify-reach-back-gone` | anti-inversion — verify.py stops importing a workflow.py private | grep `from orchestrator.workflow import _normalize_cause_hint` `orchestrator/src/orchestrator/` → absent | PASS |
| `shared-streak-substrate` | substrate (PASS today) — shared.streak.KeyedStreaks (S14 λ, task 6712) | grep `class KeyedStreaks` `shared/src/shared/streak.py` → present | PASS |

## ο — DryRunProposal: one typed entry for metadata.dry_run_proposals

Closes fk-e43472e7d5d7. Depends on μ2 (same file dry_run_unblock.py).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `proposal-model` | capability→producer (ο) — unblock_types.DryRunProposal | grep `class DryRunProposal` `orchestrator/src/orchestrator/unblock_types.py` → present | PASS |
| `duplicate-timestamp-key-gone` | anti-inversion — one investigated_at plus reaffirmed_at | grep `'timestamp':` `orchestrator/src/orchestrator/dry_run_unblock.py` → absent | PASS |
| `b3-gate-reads-typed-status` | anti-inversion — b3_gate no longer reads absence of status as success | grep `'status' in entry` `orchestrator/src/orchestrator/b3_gate.py` → absent | PASS |
| `live-proposal-typed` | INV-13 producer proof — the live dry-run hook writes the typed entry | manual: live metadata read after a real block; not a tree property | PASS |

## ω — Integration gate: the boundary matrix on real fixtures; heuristic-14 re-measure of workflow.py

Signal leaf. Closes fk-a503ccb37b4b, fk-a71254176a81. Depends on π and α–ο.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| `integration-gate-module` | capability→producer (ω) — the boundary matrix as one gate module | path `orchestrator/tests/test_workflow_spine_integration_gate.py` → present | PASS |
| `h14-measurement-recorded` | manual — the heuristic-14 measurement is recorded in the task record | manual: judged from the task record, not the tree | PASS |
