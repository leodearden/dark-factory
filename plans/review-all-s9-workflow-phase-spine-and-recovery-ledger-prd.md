# S9 — Workflow phase spine and recovery ledger (PRD)

**Status:** draft, every question ruled. Author pass 2026-11-01; fold passes 2026-11-02 applying
`critique/S9.critic.md` and its deltas `critique/S9.critic2.md` and `critique/S9.critic3.md`; ruling fold
2026-11-10 applying the wave-2 sitting (§Rulings) and re-checking every anchor at today's main; fold-verify
pass 2026-11-10 (`critique/S9.foldverify.md`) applying the lead's sequencing rulings and re-checking at
`bb1c659562`. Written by
the S9 seat of the `/review-all` remediation program (`plans/review-all-program-dark_factory-2026-10-10.md`,
stream S9, wave 2, size L, risk high).

- **Ruling.** R2: option 1, the typed Plan model; the rest of the remedy as proposed. Every open question
  of this plan was ruled on 2026-11-10 (§Rulings). None amends R2.
- **Approach.** B+H: contracts plus a two-way boundary-test sketch. G5 fired on every axis: 9 modules, 8
  cross-PRD consumers, and the workflow spine is a load-bearing seam.
- **Shape.** Twenty-one leaves, none held. π, the L0 settle check script, was added by the S9-7 ruling.

**Code anchors** verified against main `e634cfe658` (2026-11-10). Main moves fast — cite-by-symbol;
re-locate lines at implementation time. The draft was measured at `d40ba048e9`/`e87d551bf9` (2026-11-02);
2,508 commits have landed since, and 81 of the 162 existing paths the plan names changed, 24 of them
source or skill files (§Freshness at e634cfe658 says what moved). Every count below is
`measured at e634cfe658` unless it carries another sha in parentheses; the fold-verify pass re-measured
them at main `bb1c659562` (2026-11-10) and only the `workflow.py` size counts moved (§Freshness).

## Rulings (sitting of 2026-11-10)

Sitting keys in brackets. None of these amends one of Leo's 2026-10-10 rulings.

| Question | Ruling | By |
|---|---|---|
| OQ-1 [S9-1] Plan model depth | Lifecycle markers typed, steps and body verbatim, every plan.json writer through `write_plan_model` (D1), as recommended | Leo |
| Q-C1 [S9-2] what "confirmed" means | (a): step-set edits clear confirmation, files-only edits keep it, every step-set prompt including the replan ends in `confirm_plan`, as recommended | Leo |
| OQ-2 [S9-3] a structuring pass left `PARTIAL` | Completion pass once, then block `planning_stalled` with the `PlanState` in the detail, as recommended | Leo |
| OQ-3 [S9-4] single writer of phase and row | Leaf ν (C9), as recommended | Leo |
| Q-C5 [S9-5] the machine's failed row write | (b) record the failure and keep going, as recommended. Added: done and cancelled writes stay outside the machine and keep `Scheduler.mark_done`'s raise-on-failure (C9). Follow-up outside this plan: a periodic landed-journal pass, an xref to S8 and the program | Leo |
| OQ-4 [S9-6] retry-budget scope | Narrow leaf ξ on `shared.streak`, depending on task 6724, as recommended | Leo |
| OQ-5 [S9-7] how far the L0 end of life moves | Option 1 (D11, C4), as recommended. Added: leaf π (`scripts/check_l0_settle_outcomes.py`) and a lead-filed 14-day deterministic check task after ε | Leo |
| OQ-6 [S9-8] `_mark_blocked` conversion | Expand/contract; λ2 deletes the adapter, as recommended | Leo |
| Q-C4 [S9-9] conforming doubles | (a): μ1 ports and fakes, μ2 `spec_scheduler`, μ3 `spec_config`, as recommended | Leo |
| OQ-7 inject-helper home | `mcp_lifecycle.py` | the lead |
| OQ-8 resume-outcome types | new `resume_outcome.py` | the lead |
| OQ-9 corroborating landing reader | new `corroborated_landing.py` | the lead |
| Q-C2 typed architect exits | ceded to S5 ν; δ depends on S5 ν | the lead |
| Q-C3 cross-stream same-file order | option 1: the lead builds one per-file serial order at decompose | the lead |
| S8↔S9 `_on_stranded_merge_done` | ends in S8's `StrandedReconciler`; θ re-points it onto C5 in `harness.py` first, and S8 ε2b depends on θ (the opposite edge closed θ → ε2b → … → δ → ι → θ) | the lead |
| Sequencing after the ruling folds (2026-11-10) | ε depends on π (the check script lands first), confirmed; π is priority high | the lead |

Cross-stream rulings of the same sitting that reach this plan: S4-5 (a bare worker crash is re-run once,
then a real failure; κ keeps it, row 12); S4-1 (`VerifyHost` resolved once at `TaskWorkflow`, in S4's
leaves); S10-3 (the backend is the one place a status transition is refused; ν's `StatusWriter` adapts);
S8-3 (S8's event-store task ι2 allow-lists `storm_window_bound.py`'s raw event read as an offline tool in
its guard test; the module does not move onto `EventStore`, and ι2 does not edit the file ζ1 edits).

S6-2 (task 5040 first) reaches S9 transitively on two paths, although no S9 leaf consumes S6 ζ or κ:
- S6 δ (`MainHealthAutoHeal`, C3) depends on 5040 in S6's draft, and S6 θ follows δ. η consumes both, so η
  and every S9 leaf after it in label order wait on 5040.
- S5 γ depends on S6 ζ, which depends on 5040, and S5's total order puts S5 ν (the typed exits δ reads)
  twentieth of 21, after γ. So δ, and everything after it in label order, also waits on 5040.

π, α, β and γ do not wait on 5040. This is the lead's open risk on 5040 (§Live tasks).

The lead's harness.py order (S3 ε1 depends on S8 ε2b) puts θ ahead of S3 ε1 through S8 ε2b. The invariant
order, S3-1, S7-1/S7-3 and task 4867 do not touch S9.

## Freshness at e634cfe658

`git log e87d551bf9..e634cfe658` touches 81 of the 162 existing paths this plan names (24 source or skill
files, the rest tests). No finding's cost is gone. What moved under the plan:

- **Landed:** task 3155 (`ARTIFACT_ROOTS`; α's dependency, dropped), task 5563 (envelope and scoped steward
  wait; ε's dependency, dropped), task 3157 (plan.json `_base_commit`; a sixth lifecycle field in C1),
  task 4052 (`peek_plan`; `task_runtime._count_iterations`), task 5386 (`invocation_scope` in `_invoke`;
  the public MCP composer ζ2 now moves too), task 7332 (2 more disposition rows), task 6670
  (`TaskVerifyPort`; a new `_verify_green_runner` stash κ deletes), tasks 3754, 3158, 3160, 5385, 5579 and
  7668 (edits inside the regions β, ε, ζ1 and λ2 rewrite; no premise changed).
- **Re-measured:** `workflow.py` 19,471 lines (+320), `TaskWorkflow` 241 methods, `_invoke` 755 lines;
  `_mark_blocked` call sites, keyword sets, families and parameters unchanged (70, 26, 18/28/15/9, 12);
  marker reads in `workflow.py` 13 by the `get('_finalized_at'|'_session_id')` rule, the same at the draft
  sha (the draft's 17 used a wider rule); disposition rows 30 + 2 defaults (draft: 28 + 2);
  `_mark_blocked` test files 77 (draft: 75); `.state =` test files 14 (draft: 13).
- **Corrected:** `storm_window_bound.py` deferred-imports `harness._BY_DESIGN_RESTORE_OUTCOMES` (it did at the
  draft too); ζ1 now declares it. `Harness._recover_crashed_tasks`'s marker reads now sit in
  `_recover_worktree_entry`. Sibling labels follow today's drafts: S6 β1/γ1, S10 β1/β2.
- **Every delivered check** in the sidecar was re-run with `git grep -E` at e634cfe658: each producer check
  fails and each substrate check passes.
- **Re-checked at bb1c659562 (fold-verify, 78 commits later).** Task 5224 (the reviewer leg of the info-L0
  router) added `TaskWorkflow._reviewer_notes` and `_apply_note_dispositions` and widened
  `ReviewAggregation`. No S9 premise moved. `workflow.py` is now 19,511 lines, and `TaskWorkflow` has 243
  methods over 18,119 lines. Unchanged at bb1c659562:
  - `_invoke` 755, `_drive` 591, `_mark_blocked` 540;
  - 70 `_mark_blocked` calls, 26 keyword sets, 7 `disposition=`;
  - 12 `scheduler.set_task_status`, 13 marker reads;
  - 18 `dismiss=True` calls in 7 files;
  - the 95/77/59 test-double counts, 77 `_mark_blocked` test files, and 48 `.state =` in 14 files.

  The 78 mechanical checks were re-run through `shared.delivered_check_polarity` (`evaluate_grep_at_tree` /
  `evaluate_path_at_tree`) at main. All 67 producer checks fail, all 11 substrate checks pass, and
  `lint_delivered_checks` over each leaf's declared files reports nothing on the producer checks.

## Goal

The per-task workflow stops passing its own facts around in raw dicts, instance stashes and probes.
Each fact gets one typed value with one writer, and every reader consumes that value.

- **The plan's lifecycle is a type.** A typed `Plan` in `artifacts.py` derives `PlanState` (shell /
  partial / confirmed / stamped).
  - Every plan.json writer goes through it: `mcp/plan_tools.py`, `stamp_plan_provenance`, the
    orchestrator-side writers and `plan_markup_stamp.py`.
  - A plan-tools edit to the step set clears confirmation; a files-only edit (`update_plan_metadata`,
    `drop_plan_file`, as the merge-phase tightening pass uses) keeps it.
  - Every prompt that has the architect change the step set ends in `confirm_plan`, including the replan.
  - Stamping an unconfirmed plan raises. It no longer backfills `_finalized_at`.
- **An iteration-log entry says what it is.** `IterationEntry` is a discriminated union. Each writer
  decides `is_work`. Recovery-DONE, the infra-resume progress signal and the completion-judge prompt read
  that field instead of pattern-matching labels.
- **A block is one value.** The phase loop returns a `BlockReason`, and `_mark_blocked(block)` takes only
  that. This removes:
  - its 12 parameters and 26 keyword sets;
  - the four stash dicts and their comment-enforced read order.
- **The steward's answer arrives on the channel and is believed.** The steward applies the facts it holds
  (the task row and an open L1) to every variant it publishes. The workflow returns a published outcome
  verbatim and probes only when nothing was published. The workflow dismisses its own L0s in one place.
- **`_invoke` has three parts and one route resolver.** These are resume arming (a typed `ResumeArm`),
  the invocation, and accounting. Route resolution goes through `routing_dispatch.resolve_and_record_route`
  for in-band and out-of-band calls alike, with one scope-capacity read.
- **The seams the doubles stand in for are typed, and the doubles really conform.**
  - Protocols and hand-written fakes cover the steward, the merge worker and the dry-run scheduler subset.
  - A spec'd scheduler double and a spec'd config double replace the bare `MagicMock()` schedulers and
    configs. Pyright cannot type-check a
    `MagicMock`, because typeshed's `NonCallableMock` subclasses `Any`.
  - Only after both is a double-tolerance branch deleted.
- **Workflow and harness share one merge-done classification and one `stranded_merge_failed` filer**,
  both in `stranded_verified_green.py`.

What a user observes when this lands:

- **A cut-off structuring pass is not executed.** This covers a replan, a revalidation, or a fresh plan
  whose architect is cut off before `confirm_plan`. The plan reads `PARTIAL`, gets the completion pass once,
  and if still unconfirmed blocks `planning_stalled` with the `PlanState` in the detail.
- **Block categories come from one table.** Every block's escalation record carries an
  `EscalationCategory` member chosen there; no call site writes a category string. An
  `escalate_to_human=True, skip_escalation=True` call cannot be written.
- **A drain-restart no longer files a spurious `stranded_merge_failed`.** After a merge-worker
  drain-restart, a stranded-reaper merge whose request was only interrupted no longer files a born-at-L2
  `stranded_merge_failed` (task 7985's hypothesis, closed by construction).
- **Moving the L0 settle to the steward does not strand or wrongly dismiss more L0s.** Fourteen days after
  ε lands, a lead-filed deterministic check runs π's `scripts/check_l0_settle_outcomes.py` against the
  escalation archive and exits 0; a breach files a born-at-L2 `milestone_check_failed` naming the joint L0
  end-of-life redesign with the harness plan (S8).
- **Journal-trusting readers corroborate before acting.** Both the three workflow recovery guards and
  `task_ground_truth` treat a landed-outbox row as landed only after `is_ancestor_checked` corroborates it.
  A stale write-ahead row retries the merge instead of reaching a human.
- **`workflow.py` sheds its duplicates.** It loses:
  - the four routing copies;
  - the twin filer and the merge-done classifier;
  - the stash dicts;
  - the label-matching classifier;
  - the plan-marker reads.

  The final leaf re-measures the file under heuristic 14 and records the measurement.

## Background

The `/review-all` run `review-all-dark_factory-20261008` found that `TaskWorkflow`'s phases talk through
shared mutable state:

- The plan lifecycle was marker keys.
- Iteration entries were writer-specific dicts.
- Block facts rode on stash attributes.
- The steward's outcome was inferred by probes.
- Route resolution existed twice.
- Workflow and harness duplicated the stranded-merge handling.

Most `workflow.py` keys came through prior instruments: the hotspot refresh
`hotspot-survey-dark_factory-20261009` and the review `review-dark_factory-20261009`. Their full statements
and proposals are in `plans/bug-hotspot-survey-2026-10-09-full-findings.json` and
`review/reports/review-dark_factory-20261009.json`.

Prior work this PRD builds on and does not redo:

- `plans/workflow-state-machine-prd.md` (W9) delivered `WorkflowStateMachine`, `TerminalReport`,
  `StewardOutcome`, `BlockDisposition`/`classify_failure` and `RetryLedger`. W9 typed only the outermost
  channel, so each S9 leaf finishes one of those seams inward.
- Task 5726 (`artifacts.py::iteration_entry_written_by_orchestrator`).
- Task 7096 (RC2): an L0 is promoted, never dismissed and then forgotten, and `_mark_blocked` raises on
  `escalate_to_human and skip_escalation`.
- Task 5563 (done; merged `3fd6ccb3dd`, 2026-11-07) wraps every published steward outcome in a
  `PublishedStewardOutcome(escalation_id, outcome)` envelope and scopes `_await_steward_completion` to its
  own escalation (`escalation_id` is a required argument; `_in_wait_scope` and
  `escalation/classify.py::closed_by_non_sweep_resolver` hold the rule). Leaf ε builds on it.
- Task 3155 (done; merged `ef96d32603`, 2026-11-02) made `ARTIFACT_ROOTS` the one artefact resolver
  (`TaskArtifacts.root` is gone; plan-tools take `--task-root`), and task 3157 (done, `dc443323ff`) gave
  plan.json its own `_base_commit` (`artifacts.py::PLAN_BASE_COMMIT_KEY`), which `stamp_plan_provenance` and
  `bump_revalidation_stamp` write and `TaskWorkflow._revalidation_base` reads. α types it as a lifecycle
  field.

Rulings this PRD implements without relitigating:

- R2 (this stream).
- R22: S9 owns no invariant.
- R23: the weakened key fk-108cdd00d122 is re-verified below.
- The G4 rows naming S9.

One element of R2's adopted remedy is delivered by another stream: "typed architect-exit reports via
`_read_model`". S5 ν (S5 D13) already authors `_read_model` and five frozen `*Exit` dataclasses under S5's
key `fk-78b757799cab`. S9 cedes that element and consumes the typed exits in δ (ruled: Q-C2).

## Findings disposition

There is one row per key of the stream (20), and every count is measured at e634cfe658 unless it carries an
older sha. No key's cost is gone: no row is `fixed:` or `stale`, and every leaf stands. Contract §8 step 1
(`find_tasks_by_metadata(key="x_finding_key")`, all statuses, re-run 2026-11-10) returned no task for any of
the 20 keys; 136 tasks carry the key.

| Key | Sev | Disposition | Re-verified |
|---|---|---|---|
| `fk-37190b541196` plan lifecycle has no type | high | leaf β (α lands the model and the writer side) | **Still present, but the mechanism is not the one the finding names.** (a) Confirmation is sticky across structural edits: no plan-tools structural mutator (`add_plan_step`, `replace_plan_step`, `remove_plan_step`, `add_prerequisite`, `update_plan_metadata`) clears `_finalized_at`. `TaskWorkflow._stamp_architect_plan` therefore compares against `finalized_before_invoke`. (b) `BriefingAssembler.build_replan_prompt` forbids `create_plan` and asks for no `confirm_plan` (commit `60988ec0be`, inside the run pin); every other architect prompt ends in `confirm_plan()`. A cut-off replan keeps its old `_finalized_at`, so the finding's create-then-cut-off scenario needs a disobedient architect. The live holes are a cut-off obedient replan or revalidation, both executed today. Still true: `TaskArtifacts.stamp_plan_provenance` `setdefault`s `_finalized_at`; `plan_tools._create_plan` drops every lifecycle key; 5 textual `stamp_plan_provenance(` sites in `workflow.py` (`_drive`'s `initial_plan`, `_stamp_architect_plan`, `_stamp_authored_plan`, the post-review replan in `_execute_verify_review_loop`, the `_execute_iterations` re-stamp), of which only `_plan`'s path (via `_run_architect_for_plan`) is gated. Marker-read lines (`get('_finalized_at')` / `get('_session_id')`): `workflow.py` 13 (6 in `_plan`), `harness.py` 3, `artifacts.py` 2, `plan_tools.py` 1, evals 2. Raw plan.json write calls (`write_plan(` / `_write_held_plan(`): `plan_tools.py` 12, `artifacts.py` 9, `workflow.py` 4, `plan_markup_stamp.py` 1. Since the draft, task 3157 added a sixth lifecycle key, `_base_commit`, with module readers `artifacts.py::plan_base_commit`/`plan_task_text_fp` (4 `workflow.py` call sites), and task 4052 added `TaskArtifacts.peek_plan` (normalized, never written back). `Harness._recover_worktree_entry` (called by `_recover_crashed_tasks`) still preserves on `plan.get('_session_id')`. |
| `fk-9973521f4f59` iteration entries untyped | medium | leaf γ | 7 `append_iteration_log` sites, all in `workflow.py`. `'iteration'` holds `verify_attempt` at the debugger site. `_iteration_entry_is_work` is unchanged. `_check_infra_resume_thrash` and `task_runtime._count_iterations` (task 4052 moved the count out of `_read_task_entry`) count raw `len(entries)`. `build_completion_judge_prompt` renders the last 5 raw entries with no provenance filter. `append_iteration_log` refuses a non-orchestrator `source` (task 5726), the only enforced key. |
| `fk-108cdd00d122` workflow/harness twins (**weakened**) | medium | leaf θ (pairs 1–2) | Kept, scoped to the verifier's reading. Pair 1 is genuine (same dedupe read, same `critical` L2 record, same `escalation_created` payload). Pair 2 shares the future classification and status-set dispatch. The other pairs are not taken: pair 3 is already SPOT (`delivered_checks.gate_mark_done_on_delivered_checks`); pair 6 diverges by design; pair 5's fix is task 5310's `delete_keys`. Pair 4 was re-verified and dropped as thin: 49 vs 59 lines, one `set_task_status(mid, 'pending')` plus two warnings. The harness copy's race guard is a behaviour difference, not duplication (xref S6, train-module owner). Sizes: workflow 112 + 91 + 123 lines, harness 80 + 54 (AST spans). No test asserts the two filings agree. |
| `fk-f37f7e253cd5` production branches tolerating doubles | medium | leaf μ3 (μ1, μ2 and ζ1 land their branches) | All 9 branches are present, plus 2 `dry_run_unblock.py` trims and the `_exit_status_facts` `is True` identity check (task 6962). Seams `steward_factory`/`merge_worker` are untyped, `self._steward: Any`, and `run_dry_run_unblock` takes 6 `Any` parameters. Of the 95 test files constructing `TaskWorkflow(`, 77 build a bare `MagicMock()`/`AsyncMock()` scheduler (146 test files overall). Typeshed's `NonCallableMock(Base, Any)` makes those assignable to any Protocol. Branch 1 tolerates a mock *config*: 59 of the 95 build `config = MagicMock(…)` (rule `config\s*=\s*(MagicMock|AsyncMock)\(`), 56 of them also μ2 suites, only 4 set `claimant_heartbeat_interval_secs`. |
| `fk-2c0594928a18` `_invoke` 6 concerns, 4-valued string | medium | leaf ζ1 | `_invoke` is 755 lines (task 5386 routed it through `agents/invocation_scope.py::invocation_scope`). Resume arming is still inline. `restore_outcome` is a local str. `ResumeFailure.stage`/`.restore` are str. 7 instance attributes are written. `harness.py::_BY_DESIGN_RESTORE_OUTCOMES = {'disabled', 'miss'}` is a second home of the restore vocabulary, and `storm_window_bound.py` deferred-imports it. |
| `fk-ee708be4436f` terminal state checkable, not single-sourced | high | leaf ν | 12 `scheduler.set_task_status` calls (AST), 10 with literal statuses, 3 in a function that also calls `_enter_phase`. `force_set` backs the `state` setter; tests assign `.state =` 48 times in 14 files. `self._terminal_report` is assigned at 4 sites. 26 direct `return WorkflowOutcome.BLOCKED/ESCALATED/REQUEUED` in `TaskWorkflow`. `exit_contract.judge_exit` judges the phase half (`report_phase != machine_state`) and the outcome/status half. The latter takes `_note_exit_status_write`'s write-failure record (11 calls) as its `STORE_UNAVAILABLE` relaxation. |
| `fk-4f7a933c0dd4` exception ladder re-classifies | medium | leaf λ1 | After `classify_failure`, `_drive`'s broad `except Exception` re-switches on `isinstance` in 6 arms (5 once S4 α removes the `VerifyInfraError` arm). 4 of the 6 pair `disp` with a hand-written `category=`. `BranchResetError` hard-codes `escalate_to_human=True`. The newer rows fall to the generic arm: 3 at the draft, 5 now that task 7332 added `GitQueryFailed` and `CitationPatternInvalid`. |
| `fk-a152341e38ed` journal row treated as landing | medium | leaf ι | Three workflow readers go from a journal hit to `_finalise_recovery_done(basis='journal')` with no ancestry check: `_recover_if_already_merged`, `_recover_before_execute` and `_recover_landed_branch`. `task_ground_truth._resolve_branch_state` is a fourth (`BranchState(ON_MAIN, row.advanced_sha)`, no git I/O). ι takes all four. `merge_lane/worker.py::reconcile_landed_row` remains the only corroborating reader. |
| `fk-44bc973b1cc4` inner phases return bare BLOCKED | high | leaf κ (η lands the merge half) | Bare `BLOCKED` returns: `_execute_iterations` 4, `_verify_debugfix_loop` 5. `_run_scoped_verification_with_infra_retry` stashes and returns `None`. The read order and its "Must be checked BEFORE" comment are unchanged. `_infra_hold_info` has no reset; it writes 6 keys, 4 of them never read (d40ba048e9). |
| `fk-d2ba7601c130` five block vocabularies | high | leaf δ (merge-lane prefix half: S6 β1, then η) | 70 `_mark_blocked` calls; 7 pass `disposition=`. 10 distinct `category=` expressions. 5 `suggested_action` values. `escalation.categories.EscalationCategory` (registered, enforced at `EscalationQueue.submit`) is used by no call site. |
| `fk-bab6058b23c8` no owner for an L0's end of life | high | leaf ε (the reaper half is S8's, asked by xref); π measures the change | 18 `resolve(..., dismiss=True)` call sites in 7 files: `workflow.py` 4, `steward.py` 2, `harness.py` 6, `escalation_router.py` 2, `l0_promotion.py` 1, `review_checkpoint.py` 1, `escalation/queue.py` 2 (code calls only; comment mentions excluded). `cc21530ae4` (task 7096) narrowed `_mark_blocked`'s dismissal to one handoff-covered helper. Baseline for the S9-7 check (`digests/discussion/S9.md` §S9-7, measured 2026-11-10 over 2026-10-11..11-10): 18 of 495 blocking L0s stranded (3.6 per 100), 1 wrongly dismissed by the reaper's dismiss arm. |
| `fk-e39550272df9` steward outcome inferred | high | leaf ε; π measures the change | Reduced: the timestamp-window probe is gone, `run()`'s mirror is `_honour_steward_terminal_decision`, and task 5563 scoped the wait to its own escalation. Remaining: `_await_steward_completion` reads `scheduler.get_status` after every wait and synthesizes `StewardTerminalDecision` from it, and its `has_open_l1` override converts `_outcome_requeues` outcomes, including `StewardInterrupted(wip_commits_present=True)`, into `StewardReescalatedL1`. Its docstring now names the override as the only layer that sees a chained follow-on L0 (task 2248), whose outcome a scoped wait discards as foreign. That override also dismisses the L0 because the WIP publisher skips `_auto_escalate_to_human`. The steward never publishes `StewardTerminalDecision`. |
| `fk-9eb149f0706d` retry budgets in five homes | medium | leaf ξ | `RetryLedger` (`shared/task_metadata.py`) is whole-blob. `merge_first_enqueued_at: str` holds `str(epoch)`. The steward keeps 4 guard-file maps, popping three in lockstep at 3 sites. `_failure_signature_history` is in memory. "N consecutive identical" is written 3 times: `_evaluate_merge_thrash`, the `_verify_debugfix_loop` window, and `streaks.py::StreakCounter.touch(cause=)`. `shared/streak.py::KeyedStreaks` exists (S14 λ, task 6712). Task 6724 (S14 ο, pending) rewrites `streaks.py` on it. The scheduler requeue dicts are S17's. |
| `fk-a503ccb37b4b` workflow.py purpose needs "and" | medium | leaf ω | Moved since: the offline-lane helpers are gone (`131622ebf5`, task 6862), and `_select_train_members` moved to `merge_lane/train_selection.py` (`825feb1caf`/`7ba39a7d8f`). Still present, each with the leaf that moves it: `steward.py` deferred-imports `_inject_verdict_tools_mcp` (ζ2); `evals/architect_eval.py` imports `_inject_plan_tools_mcp` (ζ2) and `_meta_root_for_worktree` (β); `verify.py` deferred-imports `_normalize_cause_hint` (ξ); `merge_lane/worker.py` deferred-imports `compute_preexisting_main_break_fingerprint` (S6 δ). New since the draft: task 5386 added the public `compose_invocation_mcp_config` and `PER_INVOCATION_MCP_SERVERS`, the composer over the two inject helpers, imported by `evals/shadow_isolation.py` (ζ2 moves them with the helpers). `TaskWorkflow`: 243 methods, 18,119 lines (measured at bb1c659562; 241 / 18,088 at e634cfe658). |
| `fk-a71254176a81` heuristic-14 measurement | medium | leaf ω | 19,511 lines (measured at bb1c659562); at e634cfe658, 19,471 lines with 8,558 prose lines (44.0%; rule: comment lines plus docstring lines); 374 `task NNNN` + 18 `esc-` mentions (rule `[Tt]asks? \d{3,4}`). `_drive` 591 lines (51% prose), `_mark_blocked` 540 (51%). Each leaf rehomes the prose of what it rewrites (Comments stance); ω re-measures. |
| `fk-b8efaa973162` `_mark_blocked` flags non-orthogonal | high | leaf λ2 (δ lands the value) | 12 parameters, 70 call sites, 26 distinct keyword sets. The RC2a contradictory pair is gone and rejected at run time (task 7096), not by construction. 24 calls thread `merge_phase=merge_phase` through 9 helpers. Test fan-out: 77 test files reference `_mark_blocked`; the two new ones are `test_mark_blocked_outcome_scoping.py` (task 5563, declared by ε) and `test_workflow_plan_files_gate_own_work_base.py` (task 7668, reads a patched mock's kwargs; joins λ2's list). At d40ba048e9, 36 of the then 75 passed keyword flags or read a patched mock's kwargs (AST: 11 pass flags, 28 read kwargs). |
| `fk-7d410939a337` stash idiom | medium | leaf κ (ζ1 and η land their halves) | 51 attributes are assigned in a `TaskWorkflow` method other than `__init__` (any `self.<attr>` assignment target). `_verify_green_result` is new (task 5651), and task 6670 added its sibling `_verify_green_runner`. Test references: `_infra_hold_info` 55, `_zero_output_hang_info` 32, `_last_merge_block_reason` 23, `_pending_resume_session_id` 43. 10 test files reference the κ stash attributes. |
| `fk-67b5971ed72d` two dead disposition columns | low | leaf λ1 | 32 `BlockDisposition(` constructions in `workflow_types.py` (30 rows plus 2 defaults; task 7332 added the `GitQueryFailed` and `CitationPatternInvalid` rows). `category` is `FailureCategory.NONE` ×32 and `block_class` is `BlockClass.AGENT_FAILURE` ×32. 5 rows are documented unreachable or completeness-only (d40ba048e9); λ1 re-measures after S4 α deletes the `VerifyInfraError` row. `TerminalReport.category` is set from `disposition.category` and read nowhere. |
| `fk-e43472e7d5d7` dry-run proposal dicts | medium | leaf ο | 6 dict sites. All write `investigated_at` and `timestamp`; `_reaffirm_latest_proposal` re-stamps only `timestamp`. The success entry omits `status`/`cost_usd`. `b3_gate.check_proposal` keys on `'status' in entry`. `skills/escalation-watcher/SKILL.md` hand-builds an entry. |
| `fk-5ea52a0f89ad` route resolution in two copies | medium | leaf ζ2 | All 4 mirrored pairs are present. `RouteInputs.scope_capacity: Mapping[str, bool] | None` is set in-band only (`TaskWorkflow._scope_capacity_snapshot` → `UsageGate.scope_capacity_snapshot`). The five out-of-band callers pass `None`, which skips the capacity fail-safe. There is no parity test. |

Mechanical tasks: the report's `## Mechanical filings` carries none of this stream's 20 keys (step 1
confirmed). Live tasks overlapping a leaf are listed in §Live tasks.

## Consumers and user-observable surface (G1)

| Mechanism (home) | In-batch consumers | Other-stream consumers (G4) | User-observable surface |
|---|---|---|---|
| `Plan`, `PlanState`, `PlanNotConfirmed` — `artifacts.py` (C1) | β, ω | S8 ε1 (asked, xref), S18 (evals `plan_confirmed`) | a cut-off structuring pass is not executed; `confirm_plan`'s response is unchanged |
| `IterationEntry` — `artifacts.py` (C2) | γ, ε, κ | dashboard runtime view via `task_runtime`; the completion judge | recovery-DONE vs re-plan follows writer-stated `is_work`; infra-resume progress ignores bookkeeping; the judge sees orchestrator entries only |
| `BlockReason` — `workflow_types.py` (C3) | δ, ε, η, κ, λ1, λ2, ν | `dry_run_unblock` via `_spawn_dry_run_unblock`'s scalar fields; harness `TaskReport` mapping (S8); S10's `MetadataWriteFailed` | every block's category is an `EscalationCategory` member from one table |
| `StewardOutcome` as published (C4) | ε | — | a steward that set the task terminal or left an L1 is honoured from its own report |
| `classify_merge_future`, `file_stranded_merge_failed` — `stranded_verified_green.py` (C5) | θ | task 7277's `recovered_merge_failure.py` (if landed) | one `stranded_merge_failed` shape from either path; a shutdown outcome files nothing |
| `corroborated_landing` — `corroborated_landing.py` (ruled: OQ-9) (C6) | ι | `task_ground_truth._resolve_branch_state` (edited by ι; S8 δ follows) | a stale journal row retries instead of reaching a human |
| `ResumeArm`, `RestoreOutcome`, `ResumeStage`, `ResumeFailure` — `resume_outcome.py` (ruled: OQ-8) (C7) | ζ1 | S8 C6 `SessionResumeStorm` (asked, xref) | `session_resume_failed` payload strings unchanged |
| `StewardPort`, `MergeWorkerPort`, `DryRunScheduler`; the spec'd scheduler and config doubles (C8) | μ1, μ2, μ3, ν | S10 (`SchedulerFacade` parity) | double-tolerance branches gone; a non-conforming hand-written fake fails pyright |
| `WorkflowStateMachine.enter`, `StatusWriter` (C9) | ν | harness `_run_slot` | one row write per exit, equal to the state's projection |
| `scripts/check_l0_settle_outcomes.py` (C10) | π; ε's signal | the lead-filed 14-day check task (deterministic predicate); S8, named by the script's breach line | the check exits 0, or a born-at-L2 `milestone_check_failed` carries the script's JSON |

## Pre-conditions and substrate (G3, verified at `e634cfe658`)

- **The plan-tools process.** `orchestrator/src/orchestrator/mcp/plan_tools.py` imports
  `orchestrator.artifacts.TaskArtifacts` and runs as a separate stdio MCP process per invocation, launched
  `-m orchestrator.mcp.plan_tools` from `mcp_lifecycle.plan_tools_mcp_server`. The Plan model lives in a
  module that process already imports.
- **The plan mutex.** `TaskArtifacts.plan_mutex` / `_write_held_plan` (task 5745) serialize every plan.json
  write.
- **The confirm instruction.** `BriefingAssembler.build_architect_prompt`, the revalidation prompt,
  `build_plan_completion_prompt` and `build_plan_schema_repair_prompt` end in `confirm_plan()`.
  `build_replan_prompt` does not; α adds it. `build_plan_tightening_prompt` offers `confirm_plan()` only for
  an unchanged plan; its narrowing options use the files-only tools.
- **The category vocabulary.** `escalation/src/escalation/categories.py::EscalationCategory` is a registered
  StrEnum importing only stdlib. It holds every category the workflow files: `task_failure`, `infra_issue`,
  `wip_conflict`, `branch_reset`, `design_concern`, `integration_skew`, `already_landed`,
  `cross_repo_deliverable`, `preexisting_main_break`, `merge_cancelled_by_operator`,
  `stranded_merge_failed`, `planning_stalled`.
- **The state machine.** `WorkflowStateMachine.transition` delegates to
  `shared.task_transitions.is_legal_transition`.
- **Scope capacity.** `routing.py::RouteInputs.scope_capacity` is `Mapping[str, bool] | None`, and
  `shared/usage_gate.py::UsageGate.scope_capacity_snapshot` produces it. `resolve_and_record_route` is
  keyword-only and lacks both.
- **The steward channel.** `TaskSteward._publish_outcome` / `set_outcome_channel` exist.
  `TaskWorkflow._pending_steward_l0s` is today's give-up sweep scope. The envelope
  (`workflow_types.py::PublishedStewardOutcome`) and the scoped wait landed with task 5563.
- **The escalation archive.** Resolved records are kept 30 days under `data/escalations/archive/<day>/`
  (`escalation/src/escalation/archive.py::prune_archive`), each with `level`, `severity`, `status`,
  `resolved_by`, `resolution`, `resolution_class`, `timestamp`, `resolved_at`, `filing_claimant_run_id` and
  `late_resolutions`. π reads only these and the live pending queue files.
- **Deterministic checks.** `fused-memory/src/fused_memory/middleware/deterministic_task_guard.py` refuses a
  `before_done.script` that does not exist or is not executable at submit time, so the check task can be
  filed only once π's script is on main. A `delayed` milestone fires `after_secs` after the task's own
  dependencies are satisfied (`docs/task-authoring.md` §6).
- **Done writes.** `Scheduler.mark_done` retries a transient failure and then raises; a refusal raises at
  once as `SetTaskStatusRejected`. C9 leaves both as they are.
- **Ancestry.** `GitOps.is_ancestor_checked` returns `True`/`False`/`None` (None on a git error).
  `GitOps.is_ancestor` folds errors to `False`.
- **The dry-run port.** The merge lane calls `run_dry_run_unblock` through
  `merge_lane/ports.py::…dry_run_unblock` with scalar `reason`/`detail` and
  `block_class=BlockClass.MERGE_VERIFY_RED`. That scalar signature stays the lane port.
- **Streaks.** `shared/src/shared/streak.py::KeyedStreaks` and
  `orchestrator/streaks.py::StreakCounter.touch(cause=)` exist.
- **Spec'd doubles.** `orchestrator/tests/_orch_helpers.py::wire_scheduler_liveness_mock` is the precedent
  for a spec'd liveness default. Pyright includes `tests` (`orchestrator/pyproject.toml`).
- **Upstream contracts that do not exist yet,** each a dependency of the leaf that consumes it:
  - S4: C2 in α (failure vocabulary; the `infra_transient` property; `VerifyInfraError` retired;
    `verify_categories.is_infra_oserror` public; under the S4-5 ruling a bare worker crash is re-run once
    and a second crash is a real failure, not infra-transient) and C1 in β (`VerifyContext`).
  - S5: C3 in δ (`block_disposition.py` and the derived warm-lane rows) and ν (D13: typed architect exits).
  - S6: C1 in β1 (`MergeOutcome.kind`), C3 in δ (`MainHealthAutoHeal`), C4 in γ1 (the injected outbox) and
    C7 in θ (`HaltState` owner semantics).
  - S10: C2 in β2 (`MetadataWriteFailed`), `SchedulerFacade`/`FakeScheduler` parity in β1, and C3 in η
    (the slice table).
  - S25: C4, task 6766.
  - S14: task 6724 (`streaks.py` on `KeyedStreaks`).

## Sketch of approach

**Plan (α, β).** `Plan` is a frozen dataclass. Its lifecycle markers are typed in `PlanLifecycle`. Its
steps and every other key are carried verbatim, so a round-trip is byte-stable for keys the model does not
name. `PlanState` is derived, and confirmation and ownership are separate facts.

Every plan.json writer goes through `write_plan_model`:
- plan-tools' 12 sites;
- `artifacts.py`'s methods;
- the 4 workflow writers;
- `plan_markup_stamp.py`.

Value methods:
- `Plan.confirmed` (sets `confirmed_at` and `revalidated_at`, as `_confirm_plan` does);
- `Plan.restructured(**changes)` (used by every step-set plan-tools mutator; clears confirmation, so
  the state becomes `PARTIAL`);
- `Plan.confirmed_since(pre) -> bool` (`confirmed_at` changed since `pre`; the per-pass fact fingerprint
  minting needs);
- `Plan.with_lifecycle(lifecycle) -> Plan` (another value's lifecycle over this value's steps and body);
- `Plan.stamped` (raises `PlanNotConfirmed` unless confirmed; carries the task-text fingerprint and the
  plan's `_base_commit`, each left untouched when `None`, as `stamp_plan_provenance` does today);
- `Plan.recreated`;
- `Plan.with_files`;
- `Plan.revalidated` (advances `_base_commit`, as `bump_revalidation_stamp` does today).

`TaskArtifacts.read_plan_model` mirrors `read_plan` (normalizes and repairs on disk) and
`peek_plan_model` mirrors `peek_plan` (task 4052: normalizes, never writes). The module readers
`plan_base_commit`/`plan_task_text_fp` become lifecycle fields.

`build_replan_prompt` ends with "call `confirm_plan()` as your FINAL action".

β moves every reader onto `Plan.state` and the lifecycle fields:
- `_plan` dispatches on the state with `match`; the revalidation branch also needs a revalidation base
  (`TaskWorkflow._revalidation_base`, unchanged).
- After an architect pass, `_stamp_architect_plan` reads `post.confirmed_since(pre)` instead of the raw
  `finalized_before_invoke` marker read, and uses it only to decide whether to mint a fresh task-text
  fingerprint (task 4031); executability reads the state.
- The harness's recovery reads the state instead of predicting `_plan`'s branch.
- `_drive`'s `initial_plan` takes an explicit `initial_plan_origin`. A recovered plan keeps its own
  lifecycle; only eval and SIMPLE_TASK injections confirm `INJECTED`.
- The per-iteration re-stamp captures the plan before the implementer runs and, if the implementer's raw
  edit dropped the whole lifecycle block, writes the implementer's steps and body under the captured
  lifecycle. It never writes the captured body, and it never raises.

**Iteration log (γ).** `IterationEntry` is a union of seven frozen kinds, one per writer, each with a
`kind` tag and an `is_work` that the constructor sets from the kind's own fields. The reader parses each
line once:
- A tagged line parses to its kind.
- A legacy untagged line goes through the old rule, moved once into the parser.
- A line that is not orchestrator-written parses as `AgentAuthored(is_work=False)`, the ask in task 7391.

Recovery, `_execute_iterations`' re-entry check and the infra-resume progress signal read `is_work`. The
completion-judge prompt renders orchestrator kinds only, and counts what it excludes.

**Blocks (δ, η, κ, λ1, λ2).** `BlockReason` carries the facts of one block, one field per dimension:
`route`, `category`, `row_status`, `suggested_action`, `dry_run`, `dedupe_fingerprint`, `block_class`,
`counts_against_requeue_cap`, `reason`, `detail` and `root_cause`. Each field has a default, and named
constructors cover the recurring shapes.

`_mark_blocked(block)` derives the merge-phase dimension from the machine state (§D7). The 70 call sites
move in four families, each in the leaf that already rewrites its producing code:

| Leaf | Family | Sites | Waits for |
|---|---|---|---|
| δ | agent-report handlers, recovery and finalise paths; reads S5 ν's typed exits | 18 | — |
| η | the merge side | 28 | S6's `MergeOutcome.kind` |
| κ | execute, verify and planning legs | 15 | S4 C2 / C1 |
| λ1 | the `_drive` ladder | 9 | — |

Until λ2, `_mark_blocked` accepts the legacy keyword form through one adapter, `BlockReason.from_legacy_kwargs`,
which raises on the contradictory pair. λ2 deletes the adapter, the keyword parameters and the
transitional test reader.

Tests move onto the public seams: `TaskWorkflow.run()`'s returned `TerminalReport` (block reason and
category) and the filed escalation record. A patched `_mark_blocked` is not that seam, so δ's
`blocked_with(mock)` helper is transitional and λ2 deletes it.

**Steward (ε).** After each session, the steward applies the row/L1 facts to every variant it publishes
(D11). `_await_steward_completion` returns a published outcome verbatim and consults the row and the L1
queue only when nothing was published (timeout or cancel). The four workflow-side dismissal sites become
one `_settle_own_l0s(disposition)` with a per-disposition scope (ruled: OQ-5, S9-7).

**The L0 settle check (π).** Before ε lands, π puts a read-only script on main that counts, over two
N-day windows of the escalation archive, blocking L0s filed, stranded (reaper promotions plus `stale-strand`
restart dismissals), wrongly dismissed by the reaper's dismiss arm, wrongly dismissed by the startup sweep
(reported, not judged), and blocking L0s pending longer than 24 hours (C10). Once π is on main the lead
files a deterministic check task that runs it 14 days after ε is done. ε depends on π, so the check is
filed before ε lands and its pre-landing window is clean.

**`_invoke` (ζ1, ζ2).**

ζ1 splits the work into three parts:
- `arm_resume(pending, ctx) -> ResumeArm`, which consumes the single `PendingResume` value that replaces
  the three `_pending_resume_*` attributes;
- `_run_invocation(...)`;
- `_account_invocation(result, arm, decision)`, which reads the typed result, so branch 5's `getattr`
  goes.

`RestoreOutcome.is_by_design` replaces `harness.py`'s `_BY_DESIGN_RESTORE_OUTCOMES`, and
`storm_window_bound.py`'s deferred import of that set moves onto `RestoreOutcome`.

ζ2 gives `resolve_and_record_route` an optional `usage_gate` and has it compute
`scope_capacity_snapshot()` itself. `_invoke` and the five out-of-band callers pass their gate, and the
four `workflow.py` routing copies are deleted. `_inject_plan_tools_mcp` and `_inject_verdict_tools_mcp`
move to `mcp_lifecycle.py` (ruled: OQ-7), together with the composer task 5386 built over them
(`compose_invocation_mcp_config`, `PER_INVOCATION_MCP_SERVERS`), so `evals/shadow_isolation.py` imports
them from there.

**Merge-done classification (θ).** `stranded_verified_green.py` gains:
- `classify_merge_future(fut) -> MergeDone`, which routes on `MergeOutcome.kind` (`SHUTDOWN` is
  transient);
- `file_stranded_merge_failed(...)`.

The workflow and the stranded reconciler each keep only their own scheduling and side effects.
θ re-points `Harness._on_stranded_merge_done` onto C5 in `harness.py`, where the method lives at θ's
dispatch. S8 ε2b then moves it into S8's `StrandedReconciler` (`stranded_reconcile.py`); ε2b depends on θ
(lead ruling 2026-11-02, direction fixed in fold 3). If task 7277's
`recovered_merge_failure.py` exists at dispatch, its classification calls `classify_merge_future` too.

**Landing authority (ι).** `corroborated_landing(task_id, *, outbox, git_ops, main_sha)` returns a row only
once `is_ancestor_checked(row.advanced_sha, main_sha) is True`. It returns a miss on `False`, and logs at
WARNING and returns a miss on `None` (indeterminate). Otherwise it falls back to the existing
landing-evidence producers. The three workflow guards and `task_ground_truth._resolve_branch_state`'s
journal rung call it.

**Ports and doubles (μ1, μ2, μ3).**
- μ1: Protocols for what `TaskWorkflow` calls on the steward and the merge worker, and for what
  `run_dry_run_unblock` calls on the scheduler. It adds hand-written fakes with pyright conformance
  assignments, deletes the branches whose doubles are fakes or fixtures (3, 6, 7), and moves the 7
  `object.__new__(TaskWorkflow)` files onto the public builder.
- μ2: a spec'd scheduler double in `_orch_helpers.py`: `create_autospec(Scheduler, instance=True)` with
  falsy `blocked_writes_suppressed`/`is_paused` defaults, like `wire_scheduler_liveness_mock`. It migrates
  the 77 suites with a bare `MagicMock()`/`AsyncMock()` scheduler onto it, then deletes branches 4 and 9,
  the two dry-run trims and the `is True` identity check.
- μ3: a spec'd config double, `spec_config(**overrides)`: a real `OrchestratorConfig` with overrides. It
  migrates the suites that build a bare `MagicMock()` config onto it, then deletes branch 1 (the
  `_claimant_heartbeat_loop` interval `isinstance`). Branch 1 tolerates a mock config, not a mock
  scheduler.

Branch 5 goes with ζ1, branch 2 stays (D15), and branch 8 goes with ν.

**Single writer (ν) (ruled: OQ-3, Q-C5 (b)).** The contract is C9. `WorkflowStateMachine.enter(to, *,
report, row)` validates the move, writes the projected row through an injected `StatusWriter` only when it
differs from the row the machine last wrote, then commits the state. A failed write is recorded as
`machine.last_write_failure` and the run keeps going. Done and cancelled rows are not the machine's: they
stay with the store's done and cancel paths, and `Scheduler.mark_done` keeps raising when its retry
window runs out. `run()` returns `machine.terminal_report`. ν deletes:
- the phase comparison of `exit_contract.judge_exit`;
- the stash;
- the own-row literal writes;
- `force_set` and the setter.

The outcome/status half and its `STORE_UNAVAILABLE` relaxation stay, fed by `machine.last_write_failure`.

**Retry budgets (ξ), dry-run proposals (ο), gate (ω)** are as described in their leaves.

## Resolved design decisions

- **D1 — Plan model shape (ruled: OQ-1, Q-C1 (a)).**
  - Lifecycle markers are typed, `_base_commit` (task 3157) among them; `steps` and the body round-trip
    verbatim.
  - Every writer goes through `write_plan_model`, which keeps R2's "write through it" literal. Plan-tools'
    step-set mutators change only in their lifecycle side; files-only edits keep confirmation.
  - On-disk key names are unchanged (`_finalized_at` stays the confirmation marker), so live plans parse
    with no migration.
  - A new `_confirmed_by` key records `ARCHITECT` or `INJECTED`; a legacy plan without it reads as
    `ARCHITECT`.
- **D2 — `PlanState` derivation.**
  - No file → `read_plan_model() -> None`. No steps → `SHELL`. Steps, not confirmed → `PARTIAL`
    (ownership does not change this). Confirmed, not owned → `CONFIRMED`. Confirmed and owned
    (`session_id`) → `STAMPED`.
  - The revalidation branch needs `STAMPED` plus a revalidation base: `TaskWorkflow._revalidation_base`
    returns the plan's own `_base_commit` when `is_ancestor_checked` puts it on main, else the
    workflow-side `_old_plan_base` (task 3157). β reads the base from `plan.lifecycle.base_commit` and
    leaves the rule unchanged.
  - An owned, unconfirmed plan (`set_plan_files`' ownership transfer onto a partial) is now `PARTIAL` and
    gets the completion pass. Today it reaches revalidation or fresh planning.
  - `Plan.with_files` keeps today's `_created_at` backfill, so `_can_skip_revalidation`'s age read is
    unchanged. `created_at` plays no part in the state.
- **D3 — finalization is a precondition, enforced at the writer and at the reader (heuristic 10).**
  - **At the writer.** `Plan.stamped` raises `PlanNotConfirmed`, whose message names the task, the state
    and the markers present (INV-2).
  - **At the reader.** `_plan`, the replan path and `_run_architect_for_plan` check `plan.state` before
    stamping. Step-set edits clear confirmation, so a cut-off pass that changed the step set reads
    `PARTIAL` whatever its starting state. Whether a pass re-confirmed (for fingerprint minting)
    is `post.confirmed_since(pre)`, a per-pass fact the state alone cannot carry.
  - **A `PARTIAL` result of a structuring pass (ruled: OQ-2).** The replan, a revalidation or a fresh
    plan left `PARTIAL` gets `_plan`'s completion pass once in-process; still `PARTIAL`, it blocks
    `planning_stalled` with the `PlanState` in the detail. The post-review replan keeps restoring the
    captured fingerprint and `_base_commit` it reads before `_replan` (task 3157), now as
    `stamped(..., task_text_fp=pre.task_text_fp, base_commit=pre.base_commit)`.
  - **Origin of an injected plan.** `TaskWorkflow` takes `initial_plan_origin:
    Literal['injected', 'recovered']`. `Harness._run_slot` passes `'recovered'` for
    `_recovered_plans`; `build_workflow` from evals passes `'injected'`.
    - A recovered plan keeps its own lifecycle (`Plan.from_json(initial_plan)`). If `CONFIRMED`/`STAMPED`
      it is stamped as today.
    - If `PARTIAL` (for example, steps marked done at authoring by `mark_step_committed` before confirm),
      it goes through `_plan`'s completion pass and is never auto-confirmed.
    - Only `'injected'` and `_run_simple_task` confirm `INJECTED`.
  - **The per-iteration re-stamp** in `_execute_iterations` exists because an implementer may edit
    plan.json directly, and `_execute_iterations` re-reads `self.plan` after the implementer runs. So:
    - capture `pre = read_plan_model()` before the implementer `_invoke`;
    - afterwards, if the on-disk plan's lifecycle block is entirely absent (a raw edit), write
      `post.with_lifecycle(pre.lifecycle).stamped(...)`: the implementer's steps and body (its
      `mark_step_done` progress included) under the captured lifecycle. Log a WARNING and emit
      `plan_lifecycle_restored`. Never write `pre`'s body;
    - if other lifecycle keys are present but confirmation is cleared (a `restructured` edit), do not
      restore: the plan routes to the architect through the state check;
    - it never raises.
  - `PlanNotConfirmed` is raised only at the two orchestrator stamp sites that act on a freshly authored
    plan. It is not caught; it is a programming error there.
- **D4 — `IterationEntry` on disk.** Each entry gains a `kind` tag and an explicit `is_work`. The debugger
  kind writes `verify_attempt`. Legacy untagged entries classify once in the parser by the old rule (moved
  verbatim as `_legacy_is_work`), which counts how many legacy lines it classified. Agent-written lines
  parse as `AgentAuthored(is_work=False)`.
- **D5 — `BlockReason.category` is `escalation.categories.EscalationCategory`**, the registered,
  queue-enforced vocabulary; no enum is added. `suggested_action` stays `str` until S2's typed escalation
  vocabulary lands (xref).
- **D6 — `BlockRoute` replaces two booleans:**
  - `STEWARD`: file an L0 and wait.
  - `HUMAN`: L1.
  - `OWNED_ELSEWHERE`: the caller has already filed or promoted its own escalation.

  The contradictory combination cannot be constructed.
- **D7 — the merge-phase dimension is derived, not threaded.** `_mark_blocked` reads
  `self.state is WorkflowState.MERGE`. δ first proves the premise: it traces the 24
  `merge_phase=merge_phase` call sites through their 9 threading helpers and adds a test that drives each
  helper's merge-phase path and asserts the machine state. If any helper blocks with `merge_phase=True`
  outside `MERGE`, `BlockReason` gains a `phase: BlockPhase` field and the derivation is dropped. The leaf
  records which way it went.
- **D8 — phase legs return values (κ).** `_execute_iterations`, `_verify_debugfix_loop` and
  `_run_scoped_verification_with_infra_retry` return `BlockReason | Literal[...]`. For the verify leg that
  is `VerifyResult | BlockReason`, built from S4 C2's `infra_transient` property. The caller dispatches with
  `match`. The VERIFY→REVIEW emission takes the green tip and its verdict runner (task 6670's
  `_verify_green_runner`) as arguments.
- **D9 — the disposition row carries the escalation category (λ1).**
  - `BlockDisposition.category: FailureCategory` (always NONE) is replaced by
    `escalation_category: EscalationCategory`.
  - `block_class` (always AGENT_FAILURE) is dropped, and so is `TerminalReport.category` (read nowhere).
  - `disposition.block_reason(exc) -> BlockReason` builds the reason text once.
  - `_drive` keeps one clause per outcome kind.
  - The rows documented unreachable are deleted; λ1 re-measures their count after S4 α.
  - `TestBD2Completeness` is scoped to exceptions that can reach `_drive`; any exemption carries a reason
    (INV-12).
- **D10 — `MetadataWriteFailed` becomes a block in one place (λ1, R3).** It gets a disposition row
  (`INFRA_ISSUE`, route `HUMAN`). The `update_task` arms whose documented policy is "block" let it
  propagate. Arms whose policy is "log and continue" narrow to `except MetadataWriteFailed`.
- **D11 — the steward decides every published variant (ruled: OQ-5, S9-7).**
  - After the agent session for L0 *X*, `TaskSteward` reads the task row once and the task's open L1 once,
    then applies them to the variant its handler produced. The reads come after the session and before
    the publish, so they see every L1 the session filed (row 15b).
    - Every variant: a terminal/deferred row → `StewardTerminalDecision(new_status)`.
    - Only today's `_outcome_requeues` set (`StewardResolved` and `StewardInterrupted(wip=True)`): an open
      L1 → `StewardReescalatedL1(l1.id)`. `_give_up_with_wip` keeps its dismiss-then-publish (task 3170).
    - `StewardInterrupted(wip=False)` and `StewardBudgetExhausted` are published as produced. Their give-up
      path files the steward's own L1 first (`_auto_escalate_to_human`), and today they exit through
      `_mark_blocked`'s fall-through to BLOCKED; converting them would change the row, the routing-tier bump
      and the slot exit.
  - **A chained follow-on L0.** Today's consumer override covers one more case: a follow-on L0 *Y* that
    the steward chained and handed to a human (task 2248; foreign to a scoped wait since task 5563).
    `TaskSteward._run_loop` is serial (fold-verify, measured at bb1c659562). After `_handle_escalation(X)`
    publishes with `put_nowait`, the loop takes *Y* from `_next_escalation` with no awaited I/O. A
    pre-session give-up on *Y* then files *Y*'s L1 synchronously, before the steward first yields; the
    likely case is `_give_up_if_lifetime_budget_spent`, when *X*'s session spent the last of the task's
    budget. The workflow wakes only after that, so today's override sees *Y*'s L1. A read taken before
    *X*'s publish would not see it.
  - **So ε takes the reads later.** It publishes *X*'s applied outcome at the steward's next yield point:
    just before it starts another agent session, awaits the WIP probe or the watcher, or goes idle. The
    row and L1 reads come immediately before that publish, so a give-up on *Y* that needs no session is
    applied to *X*'s variant, as today (row 15c). A *Y* that runs its own agent session is seen by neither
    today's consumer nor ε. That case is unchanged and is the reaper's.
  - A failed row read skips the terminal arm only. The L1 read still runs, the envelope carries
    `degraded=True`, and a WARNING names the escalation (INV-11).
  - The workflow consults the row and the L1 queue only when no outcome was published, which is where task
    7370's timeout half lands.
  - `_settle_own_l0s(disposition)` keeps today's scope (`_pending_steward_l0s`, every pending steward L0 of
    the task) for `ABANDONED_ON_TIMEOUT` and `SUPERSEDED_BY_L1`, and narrows to own-incarnation
    (`filing_claimant_run_id`) only for `HANDED_OFF`.
- **D12 — one route resolver and one scope-capacity read (ζ2).** `resolve_and_record_route` gains
  `usage_gate: UsageGate | None = None` and computes `scope_capacity: Mapping[str, bool] | None` from it.
  Every caller holding a gate passes it: `_invoke`, `steward.py` ×2, `review_checkpoint.py`, `harness.py`
  and `dry_run_unblock.py`. `TaskWorkflow._scope_capacity_snapshot` goes. The digest is the out-of-band
  one.
- **D13 — merge-done classification by kind (θ).**
  - `MergeDoneClass` is `NOOP`, `LANDED(merge_sha)`, `TRANSIENT`, `DURABLE_FAILURE` or `UNCLASSIFIED`.
    `UNCLASSIFIED` is logged at ERROR and treated as durable.
  - `OutcomeKind.SHUTDOWN` → `TRANSIENT`.
  - The two status sets stay as the fallback for an outcome without a kind, which θ asserts cannot occur
    after S6 β1.
- **D14 — landing authority placement (ruled: OQ-9).** A new `orchestrator/corroborated_landing.py`, named for
  what it returns so it does not compete with `merge_lane/landing_evidence.py` (heuristic 1). The
  alternatives are a corroborating method on S6 C4's `LandedOutbox`, or a function in `landing_evidence.py`.
- **D15 — doubles conform before production stops tolerating them (μ1, μ2, μ3).**
  - Each Protocol is the subset the consumer calls, measured by AST.
  - `DryRunScheduler` is a narrower Protocol that `SchedulerFacade` satisfies.
  - Hand-written fakes get pyright conformance assignments.
  - `MagicMock` cannot be checked by pyright, so the 77 bare-mock-scheduler suites move onto a spec'd
    scheduler double (μ2) before branches 4 and 9 go, and the bare-mock-config suites move onto a spec'd
    config double (μ3) before branch 1 goes.
  - `_record_routing_decision`'s failure-isolation `try` (branch 2) is telemetry isolation, not double
    tolerance, and stays.
- **D16 — prose rehoming.** Every docstring or comment a leaf rewrites follows the Comments stance:
  measurements, rejected alternatives and git-log investigations go to the commit message. PRD letters
  never appear in code.
- **D17 — S9 promotes no invariant** (R22).
- **D18 — typed architect exits are S5's (ruled: Q-C2).** S5 ν delivers `_read_model` and the five `*Exit`
  dataclasses, and δ reads them. On the shared symbol `read_iteration_log`, γ owns
  `read_iteration_entries` and lands first; S5 ν routes only the non-iteration readers through
  `_read_model`.

## Contracts

### C1 — the Plan model (`orchestrator/src/orchestrator/artifacts.py`, α; finalization precondition live in β)

```python
class PlanState(StrEnum):
    SHELL = 'shell'; PARTIAL = 'partial'; CONFIRMED = 'confirmed'; STAMPED = 'stamped'

class PlanConfirmation(StrEnum):
    ARCHITECT = 'architect'; INJECTED = 'injected'

@dataclass(frozen=True)
class PlanLifecycle:
    confirmed_at: str | None          # on disk: '_finalized_at'
    confirmed_by: PlanConfirmation | None   # '_confirmed_by' (absent on a legacy confirmed plan -> ARCHITECT)
    session_id: str | None            # '_session_id' (ownership)
    created_at: str | None            # '_created_at'
    revalidated_at: str | None        # '_revalidated_at'
    revalidated_by_session: str | None
    task_text_fp: str | None          # PLAN_TASK_TEXT_FP_KEY
    base_commit: str | None           # PLAN_BASE_COMMIT_KEY '_base_commit' (task 3157)
    schema_version: int

@dataclass(frozen=True)
class Plan:
    lifecycle: PlanLifecycle
    steps: tuple[Mapping[str, Any], ...]
    body: Mapping[str, Any]           # every other key, verbatim
    @property
    def state(self) -> PlanState: ...
    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Plan: ...
    def to_json(self) -> dict[str, Any]: ...   # from_json(d).to_json() == d for any d this repo writes
    def recreated(self, *, task_id: str, title: str, analysis: str, files: list[str],
                  carried: Mapping[str, Any]) -> Plan: ...        # create_plan: lifecycle cleared -> SHELL
    def restructured(self, **changes: Any) -> Plan: ...          # step-set edit; clears confirmed_at/_by -> PARTIAL
    def confirmed(self, *, at: str, by: PlanConfirmation) -> Plan: ...   # sets confirmed_at AND revalidated_at; raises PlanNotConfirmed if no steps
    def stamped(self, *, session_id: str, at: str, task_text_fp: str | None,
                base_commit: str | None) -> Plan: ...
        # raises PlanNotConfirmed unless state in {CONFIRMED, STAMPED}; a None fp or base leaves the existing value
    def with_files(self, files: list[str], *, owner: str) -> Plan: ...   # orchestrator-side; keeps confirmation; backfills created_at as today
    def revalidated(self, *, at: str, by_session: str, base_commit: str | None) -> Plan: ...
    def confirmed_since(self, pre: Plan) -> bool: ...                    # lifecycle.confirmed_at != pre.lifecycle.confirmed_at
    def with_lifecycle(self, lifecycle: PlanLifecycle) -> Plan: ...      # this steps/body under another lifecycle

class PlanNotConfirmed(ValueError): ...   # message: task id, state, markers present (INV-2)

class TaskArtifacts:
    def read_plan_model(self) -> Plan | None: ...       # None iff plan.json absent; mirrors read_plan (repairs on disk)
    def peek_plan_model(self) -> Plan | None: ...       # mirrors peek_plan (task 4052): normalized, never written
    def write_plan_model(self, plan: Plan, *, mutex_held: bool = False) -> None: ...  # the ONLY plan.json writer
    def stamp_plan_provenance(self, session_id: str, *, task_text_fp: str | None = None,
                              base_commit: str | None = None) -> Plan: ...
```

Invariants:
- `state` is a pure function of `lifecycle` and `steps`.
- Only `confirmed` sets `confirmed_at`, and only `recreated`/`restructured` clear it.
- Step-set plan-tools mutators (`add_plan_step`, `replace_plan_step`, `remove_plan_step`,
  `add_prerequisite`, `record_descoped_step`) go through `restructured`.
- Files-only edits keep confirmation: `update_plan_metadata` and `drop_plan_file`. The merge-phase
  tightening pass (`BriefingAssembler.build_plan_tightening_prompt`, consumed by
  `TaskWorkflow._try_narrow_plan`) uses them without `confirm_plan`, and `_set_task_scope` and the
  merge-gate re-check verify the narrowed files.
- Other writers that keep confirmation: `mark_step_done`, `mark_step_committed`,
  `record_step_deviation`, `add_design_decision`, `add_reuse_item`, the markup stamp, and the orchestrator
  writers `with_files`/`revalidated`.
- Every plan.json write goes through `write_plan_model` under the existing mutex.
- A `Plan` crosses no process boundary as a Python object; the MCP process and the workflow share only
  the JSON file.

### C2 — `IterationEntry` (`artifacts.py`, γ)

```python
@dataclass(frozen=True)
class _EntryBase:
    timestamp: str; source: Literal['orchestrator', 'amendment']; summary: str
    # is_work is a derived field each kind sets in __post_init__ from its own fields

@dataclass(frozen=True) class ImplementerIteration(_EntryBase): kind = 'implementer'; iteration: int; steps_attempted: tuple[str, ...]; steps_completed: tuple[str, ...]; commit: str | None; committed: bool | None; dirty: bool
@dataclass(frozen=True) class DebuggerAttempt(_EntryBase):     kind = 'debugger'; verify_attempt: int; commit: str | None; committed: bool | None; dirty: bool
@dataclass(frozen=True) class JudgeEarlyExit(_EntryBase):      kind = 'judge_early_exit'; iteration: int; complete: bool; substantive_work: bool; uncovered_plan_steps: tuple[str, ...]
@dataclass(frozen=True) class ExecuteSkipped(_EntryBase):      kind = 'execute_skipped'; iteration: int; steps_completed: tuple[str, ...]
@dataclass(frozen=True) class Amendment(_EntryBase):           kind = 'amendment'; iteration: int; amendment_round: int; suggestions_count: int; commit: str | None; committed: bool | None; dirty: bool
@dataclass(frozen=True) class InterIterationRebase(_EntryBase): kind = 'inter_iteration_rebase'; iteration: int; old_base: str; new_base: str; files_changed_on_main: tuple[str, ...]; distance_commits: int; cohort: str
@dataclass(frozen=True) class PlanStepRederive(_EntryBase):    kind = 'plan_step_rederive'; iteration: int; rederived_steps: tuple[str, ...]
@dataclass(frozen=True) class AgentAuthored:                   raw: Mapping[str, Any]; is_work: bool = False

IterationEntry = (ImplementerIteration | DebuggerAttempt | JudgeEarlyExit | ExecuteSkipped
                  | Amendment | InterIterationRebase | PlanStepRederive | AgentAuthored)

class TaskArtifacts:
    def append_iteration_entry(self, entry: IterationEntry) -> None: ...   # refuses AgentAuthored
    def read_iteration_entries(self) -> tuple[list[IterationEntry], list[str]]: ...  # (entries, corrupt lines)
```

`is_work` per kind matches `workflow.py::_iteration_entry_is_work` for the kinds it recognises:

| Kind | `is_work` |
|---|---|
| `ImplementerIteration` | `committed is not False and bool(steps_completed)` |
| `DebuggerAttempt` | `committed is not False` |
| `Amendment` | `committed is not False` |
| `JudgeEarlyExit` | `substantive_work` |
| `ExecuteSkipped` | `bool(steps_completed)` |
| `InterIterationRebase` | `False` |
| `PlanStepRederive` | `False` |
| `AgentAuthored` | `False` |

`append_iteration_log(dict)` is deleted.

Consumers:
- `workflow.py`: `_has_prior_implementation`, `_check_infra_resume_thrash` (counts work entries),
  `_execute_iterations`, `_rederive_step_status_from_branch_state`, `_amend` and `_run_completion_judge`.
- `agents/briefing.py::build_completion_judge_prompt`, which renders orchestrator kinds only and states how
  many `AgentAuthored` entries it excluded.
- `task_runtime._count_iterations` (feeds `_read_task_entry`'s `loops`).

### C3 — `BlockReason` (`orchestrator/src/orchestrator/workflow_types.py`, δ)

```python
class BlockRoute(enum.Enum):
    STEWARD = 'steward'; HUMAN = 'human'; OWNED_ELSEWHERE = 'owned_elsewhere'

class BlockRowStatus(StrEnum):
    BLOCKED = 'blocked'; INFRA_HOLD = 'infra-hold'

@dataclass(frozen=True)
class BlockReason:
    reason: str
    detail: str = ''
    route: BlockRoute = BlockRoute.STEWARD
    category: EscalationCategory = EscalationCategory.TASK_FAILURE
    row_status: BlockRowStatus = BlockRowStatus.BLOCKED
    suggested_action: str = 'investigate_and_retry'      # typed by S2 later (D5)
    dry_run: DryRunSpawn = DryRunSpawn.OUTSIDE_MERGE_PHASE
    dedupe_fingerprint: str | None = None
    block_class: BlockClass | None = None                # None -> classify_block_reason until η, then AGENT_FAILURE
    counts_against_requeue_cap: bool = True
    root_cause: str = ''
    @classmethod
    def infra(cls, reason: str, detail: str = '', **kw) -> BlockReason: ...
    @classmethod
    def human(cls, reason: str, detail: str = '', **kw) -> BlockReason: ...
    @classmethod
    def owned_elsewhere(cls, reason: str, detail: str = '') -> BlockReason: ...
    @classmethod
    def from_legacy_kwargs(cls, reason: str, **kw) -> BlockReason: ...   # transitional; deleted by λ2

# TaskWorkflow
async def _mark_blocked(self, block: BlockReason) -> WorkflowOutcome: ...   # accepts the legacy form until λ2
```

Invariants:
- `route is OWNED_ELSEWHERE` implies no escalation is filed.
- Until λ2, the legacy adapter raises `ValueError` on the contradictory pair.
- `category` is a registered member by type.
- `TerminalReport.block_category` is `block.category.value`.

Consumers:
- `TaskWorkflow._spawn_dry_run_unblock` passes `block.reason`, `block.detail` and `block.block_class` into
  the unchanged scalar `run_dry_run_unblock(...)`, which stays the lane port. `dry_run_unblock` and
  `merge_lane/` do not import `workflow_types` for this.
- The harness maps `TerminalReport` as today.

### C4 — the published steward outcome (ε; on the envelope and scoped wait task 5563 landed)

```python
# steward.py — the producer decides the variant (D11)
async def _publish_for(self, esc: Escalation, produced: StewardOutcome) -> None:
    # after the agent session for esc: one row read, one open-L1 read, applied to `produced`, taken at the
    #   steward's next yield point (after any pre-session give-up of a chained follow-on L0; D11, row 15c)
    #   row in TERMINAL or 'deferred' (any variant)                       -> StewardTerminalDecision(TaskStatus(row))
    #   elif produced in {StewardResolved, StewardInterrupted(wip=True)}
    #        and open L1 for the task                                     -> StewardReescalatedL1(l1.id)
    #        (_give_up_with_wip keeps its dismiss-then-publish)
    #   else (incl. StewardInterrupted(wip=False), StewardBudgetExhausted) -> produced
    # row read raised -> skip the terminal arm, still read L1, publish with degraded=True, WARNING names esc.id
    # published as PublishedStewardOutcome(escalation_id=esc.id, outcome=..., degraded=...)
    #   (escalation_id/outcome landed with task 5563; ε adds degraded: bool = False)

# workflow.py (the signature task 5563 landed; ε changes only the body)
async def _await_steward_completion(self, *, escalation_id: str | None,
                                    skip_if_idle: bool = False) -> StewardOutcome: ...
    # a published outcome in the wait's scope is returned verbatim, with no get_status / has_open_l1 read;
    # only the no-outcome path (timeout / cancel / closed by a non-sweep resolver) reads the row and the L1 queue
def _settle_own_l0s(self, disposition: L0Disposition) -> None: ...   # the one workflow dismissal site

class L0Disposition(enum.Enum):
    CONSUMED = 'consumed'; HANDED_OFF = 'handed_off'; SUPERSEDED_BY_L1 = 'superseded_by_l1'
    ABANDONED_ON_TIMEOUT = 'abandoned_on_timeout'
# scope: ABANDONED_ON_TIMEOUT and SUPERSEDED_BY_L1 -> _pending_steward_l0s (every pending steward L0 of the task)
#        HANDED_OFF -> L0s whose filing_claimant_run_id is this incarnation's
```

Invariants:
- `StewardTerminalDecision` is published only by the steward; the docstring saying otherwise is inverted.
- A consumer probe never overrides a published outcome.
- A `degraded` outcome is never the terminal variant.
- The row and L1 reads for *X* are taken at the steward's first yield after *X*'s session, never earlier
  (D11, row 15c).
- The consumer's terminal-absorption guard (`SetTaskStatusRejected` → `_handle_cancelled_terminal_exit`)
  still catches a row a human changed after the publish.

### C5 — merge-done classification and the stranded filer (`orchestrator/src/orchestrator/stranded_verified_green.py`, θ)

```python
class MergeDoneClass(enum.Enum):
    NOOP = 'noop'; LANDED = 'landed'; TRANSIENT = 'transient'
    DURABLE_FAILURE = 'durable_failure'; UNCLASSIFIED = 'unclassified'

@dataclass(frozen=True)
class MergeDone:
    cls: MergeDoneClass; outcome: MergeOutcome | None; merge_sha: str | None

def classify_merge_future(fut: asyncio.Future[MergeOutcome]) -> MergeDone: ...
async def file_stranded_merge_failed(
    *, escalation_queue: EscalationQueue, event_store: EventStore | None, task_id: str,
    agent_role: str, reason_tag: str, detail: str, filing_claimant_run_id: str | None,
) -> str | None: ...   # the escalation id, or None when an open one already exists (dedupe)
```

Invariants:
- One dedupe read, one `Escalation` shape (`critical`, level 2, `manual_intervention`) and one
  `escalation_created` payload.
- `classify_merge_future` routes on `outcome.kind`, never on reason text.
- The filer submits to `EscalationQueue` directly. It is a future migration site for S1's one filing entry
  point (xref).

Consumers:
- `TaskWorkflow._on_architect_merge_done`, which keeps its landing arm and its events.
- `Harness._on_stranded_merge_done`, re-pointed onto C5 by θ in `harness.py`. It then moves into S8's
  `StrandedReconciler` (`orchestrator/src/orchestrator/stranded_reconcile.py`) in S8 ε2b, which depends on
  θ. From then on the composition root passes `stranded_verified_green.file_stranded_merge_failed` to
  `StrandedReconciler`.
- If it exists at dispatch, `recovered_merge_failure.py` (task 7277).

### C6 — corroborated landing (`orchestrator/src/orchestrator/corroborated_landing.py` (ruled: OQ-9), ι)

```python
@dataclass(frozen=True)
class Landing:
    sha: str; basis: Literal['journal', 'git']

async def corroborated_landing(task_id: str, *, outbox: LandedOutbox, git_ops: GitOps,
                               main_sha: str) -> Landing | None: ...
```

Invariants:
- A journal row counts only when `await git_ops.is_ancestor_checked(row.advanced_sha, main_sha) is True`
  (INV-3).
- `False` means a miss, logged at INFO with `pruned_not_landed` semantics.
- `None` (git error) means indeterminate: logged at WARNING naming the task and the sha, and treated as a
  miss (INV-11).
- The call never consumes or rewrites the row; pruning stays `reconcile_landed_row`'s.
- The git fallback reuses the existing landing-evidence producers.

Consumers:
- the three workflow guards;
- `task_ground_truth._resolve_branch_state`'s journal rung, edited by ι. S8 δ's condition-row rewrite
  follows ι (xref).

### C7 — resume arming (`orchestrator/src/orchestrator/resume_outcome.py` (ruled: OQ-8), ζ1)

```python
class RestoreOutcome(StrEnum):
    DISABLED = 'disabled'; MISS = 'miss'; FAULT = 'fault'; PUBLISHED = 'published'
    @property
    def is_by_design(self) -> bool: ...   # DISABLED, MISS — replaces harness._BY_DESIGN_RESTORE_OUTCOMES
class ResumeStage(StrEnum):
    PRE_FLIGHT = 'pre_flight'; CLI = 'cli'

@dataclass(frozen=True)
class PendingResume:
    session_id: str; role: str; count: int

@dataclass(frozen=True)
class ResumeArm:
    session_id: str | None; resume_session_id: str | None; resume_count: int
    restore: RestoreOutcome; archive_root: Path | None

@dataclass(frozen=True, slots=True)
class ResumeFailure: ...            # moved from workflow.py; stage: ResumeStage; restore: RestoreOutcome | None
class ResumeOutcomeSink(Protocol): ...   # moved from workflow.py

async def arm_resume(pending: PendingResume | None, ctx: ResumeContext) -> ResumeArm: ...
```

Invariants:
- `arm_resume` is the only place resume eligibility, restore and veto are decided.
- `session_resume_failed` payload strings are unchanged.
- `_invoke` reads `arm.*` and the typed invocation result, never `getattr(result, ...)`.

Consumers:
- `TaskWorkflow._invoke`.
- The harness sink, which reads `report.restore.is_by_design`.
- S8 C6 `SessionResumeStorm`, asked by xref to drop `BY_DESIGN_RESTORE_OUTCOMES` and read
  `RestoreOutcome.is_by_design`, importing `ResumeOutcomeSink`/`ResumeFailure` from `resume_outcome.py`.

### C8 — collaborator ports and the spec'd scheduler and config doubles (μ1, μ2, μ3)

```python
class StewardPort(Protocol): ...       # the subset TaskWorkflow calls on TaskSteward, measured by AST in μ1
StewardFactory = Callable[[TaskAssignment], StewardPort]
class MergeWorkerPort(Protocol): ...   # the subset TaskWorkflow calls on SpeculativeMergeWorker
class DryRunScheduler(Protocol): ...   # the subset run_dry_run_unblock calls (get_task, update_task, get_status, ...)

# orchestrator/tests/_orch_helpers.py (μ2)
def spec_scheduler(**overrides: Any) -> Scheduler: ...
    # create_autospec(Scheduler, instance=True); blocked_writes_suppressed / is_paused default falsy;
    # awaitable methods return AsyncMock results; the one helper every workflow suite builds its scheduler with
def spec_config(**overrides: Any) -> OrchestratorConfig: ...   # μ3: a real OrchestratorConfig with overrides
```

Invariants:
- Each Protocol is the measured call subset.
- `TaskSteward`, `SpeculativeMergeWorker` and every hand-written fake have a pyright conformance
  assignment in `orchestrator/tests`.
- A spec'd double raises `AttributeError` on a method `Scheduler` does not have, and returns falsy liveness
  answers by default.

### C9 — single writer (ν)

```python
class StatusWriter(Protocol):
    async def write(self, task_id: str, status: TaskStatus) -> None: ...   # scheduler.set_task_status

class WorkflowState(enum.Enum): ...  # gains REQUEUED

class WorkflowStateMachine:
    def __init__(self, *, writer: StatusWriter, task_id: str,
                 initial_state: WorkflowState = WorkflowState.PLAN) -> None: ...
    async def enter(self, to: WorkflowState, *, report: TerminalReport | None = None,
                    row: TaskStatus | None = None) -> None: ...
    @property
    def terminal_report(self) -> TerminalReport | None: ...
    @property
    def last_write_failure(self) -> BaseException | None: ...
```

Projection (the row `enter` writes):

| State | Row |
|---|---|
| PLAN, EXECUTE, VERIFY, REVIEW, MERGE | `in-progress` |
| MERGE_DEFERRED | `merge-deferred` |
| REQUEUED | `pending` |
| BLOCKED | `report`'s block `row_status` (`blocked` or `infra-hold`) |
| ESCALATED | `blocked` |
| DONE, CANCELLED | none (the store's done/cancel paths own those rows, as today) |

`row=` overrides only for the documented re-pend paths.

Ordering: `enter` validates the move (`is_legal_transition`), then writes the projected row only when it
differs from the row the machine last wrote, then commits the state and the report. The dispatch claim
writes `in-progress` once, so working-phase moves (PLAN → EXECUTE → VERIFY → REVIEW → MERGE) write nothing,
as `_enter_phase` writes nothing today. A harness-owned write during a working phase
(`Harness._action_teardown_and_set_status`, an operator `set_task_status`) is therefore never overwritten by
the next phase move (INV-6), and `_exit_status_facts`' hard-cancel relaxation keeps its premise.

Failure semantics (ruled: Q-C5 (b), record the failure and keep going, S9-5):
- A rejected write (`SetTaskStatusRejected`) propagates to the existing terminal-exit handlers, unchanged.
- Any other write failure is swallowed, recorded on `last_write_failure` with a WARNING, and the state is
  committed. `StatusWriter` adapts whatever `Scheduler.set_task_status` takes at dispatch (S10's
  transition task makes the backend the one place a transition is refused, S10-3 ruling); the refusal
  still arrives as `SetTaskStatusRejected`.
- Done and cancelled writes stay outside the machine and keep `Scheduler.mark_done`'s raise-on-failure:
  the record-and-continue rule governs only the `in-progress`, `merge-deferred`, `pending`, `blocked` and
  `infra-hold` writes the projection table makes (row 24d).
- No failed write is queued for a later retry: replaying an old `blocked` or `pending` over a row the
  harness, an operator or a steward has moved since is the stale overwrite this contract removes.
- `exit_contract.judge_exit` keeps its outcome/status half and `STORE_UNAVAILABLE`, now fed from
  `machine.last_write_failure`.
- The phase comparison (`report_phase != machine_state`) is deleted because the machine holds both.

Invariants: exiting `run()` without a terminal report raises a named invariant error. Rows of other tasks
(`_revert_withheld_member`) are written directly, not through the machine.

### C10 — the L0 settle check (`scripts/check_l0_settle_outcomes.py`, π; ruled: OQ-5 plus Leo's check, S9-7)

```text
scripts/check_l0_settle_outcomes.py --queue-dir DIR --window-days N
    --baseline-stranded-per-100 F --ratio R --min-stranded K
    --max-reaper-wrong M --max-pending-hours H
reads:   DIR/archive/<day>/*.json (resolved records, 30-day retention) and DIR's live pending records; writes nothing
windows: post = the last N days; pre = the N days before them
counts per window, blocking L0s on numeric task ids only (info L0s excluded):
  filed; stranded = reaper promotions ('Auto-promoted to level 1') + startup-sweep 'stale-strand' dismissals;
  reaper_wrong = reaper dismiss-arm closes later resolved late, or followed within 24 h by a same-task,
                 same-category escalation;
  startup_wrong = the same rule over startup-sweep dismissals (reported, never judged);
  pending_over_H = blocking L0s pending longer than H hours now
stdout:  human lines, then one trailing JSON object carrying every count and rate
exit 1:  post stranded per 100 >= R * max(pre per 100, F) and post stranded >= K
         or post reaper_wrong >= M
         or pending_over_H > 0
         the last line then names the follow-up: open the joint L0 end-of-life redesign with the harness plan (S8)
exit 0:  otherwise; exit 2 on an unreadable queue directory (INV-11: never a silent pass)
         the runner treats any rc != 0 (75 aside) as a verdict, so exit 2 also files milestone_check_failed;
         its last line names the unreadable path, not the redesign
```

Invariants: the script is read-only and executable (the deterministic guard refuses a non-executable
`before_done.script`); its rules are the ones `digests/discussion/S9.md` §S9-7 measured the 3.6-per-100
baseline with, so the baseline and the check count the same thing.

The lead-filed check task (a lead action, filed once π is on main): title "Check stranded and wrongly
dismissed L0s 14 days after the steward-decides-outcome change landed"; `task_kind='deterministic'`;
`priority='medium'`; `dependencies=[ε]`; `always_escalates=false`;
`metadata.milestone={"mode": "delayed", "after_secs": 1209600}`;
`metadata.before_done={"kind": "predicate", "script": "scripts/check_l0_settle_outcomes.py", "args":
["--queue-dir", "data/escalations", "--window-days", "14", "--baseline-stranded-per-100", "3.6", "--ratio",
"2.0", "--min-stranded", "8", "--max-reaper-wrong", "3", "--max-pending-hours", "24"], "timeout_secs": 120}`;
`metadata.source='review-all'`, `metadata.x_finding_run` this run, `metadata.x_finding_key=[fk-bab6058b23c8,
fk-e39550272df9]`. A non-zero exit makes the runner file a born-at-L2 `milestone_check_failed` carrying the
script's JSON and blocks the check task; the human files the redesign on resolving it.

## Boundary-test sketch

Each row faces one side of a seam. "Real" means:
- a real `TaskArtifacts` on `tmp_path`;
- a real tmp git repo;
- the real plan-tools functions called against the same `TaskArtifacts`;
- the real `TaskSteward` with a scripted agent;
- the real `EscalationQueue`.

| # | Seam / side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | C1 writer (plan_tools) | create_plan → add_plan_step ×2 → confirm_plan | empty task meta | `state is CONFIRMED`; bytes on disk equal today's for the same calls except `_confirmed_by` (`_revalidated_at` set by confirm, as today) |
| 2 | C1 writer | create_plan over a stamped plan | stamped plan | `SHELL`; carried `_markup_rejections`/descope preserved |
| 3 | C1 precondition | `stamp_plan_provenance` on a PARTIAL plan | steps, no confirm | raises `PlanNotConfirmed` naming task, `partial`, markers; plan.json unchanged |
| 3b | C1 writer | `add_plan_step` on a STAMPED plan | stamped plan | `PARTIAL` (confirmation cleared) |
| 3c | C1 writer | merge-phase tightening pass via (a) `update_plan_metadata(files=…)` or (c) `drop_plan_file`, no `confirm_plan` | stamped plan at MERGE | state stays `STAMPED` |
| 4 | C1 reader (workflow) | real replan tool sequence (`add_plan_step` ×N via the replan prompt's tools), architect cut off before `confirm_plan` | stamped plan, review loop | state `PARTIAL`; completion pass invoked once; still PARTIAL → block `planning_stalled`; plan never stamped; no EXECUTE entry |
| 4b | C1 reader | revalidation pass cut off mid-edit | stamped plan, rebased main | same as row 4 |
| 4c | C1 reader | cap-exit revalidation with no confirm and no step-set edit | stamped plan | `confirmed_since(pre)` False; the old `_task_text_fp` is kept |
| 5 | C1 reader (harness) | crash with STAMPED zero-done plan / PARTIAL plan (`Harness._recover_worktree_entry`) | tmp worktree | STAMPED preserved for revalidation; PARTIAL per §Open questions 1; no code reads `_session_id` |
| 5b | C1 reader (`initial_plan`) | crash-recovered PARTIAL plan with authoring-time done steps | `_recovered_plans` | `initial_plan_origin='recovered'`; completion pass, never auto-confirmed |
| 5c | C1 re-stamp | implementer strips the whole lifecycle block from plan.json mid-execute | executing task | the re-stamp writes post's steps/body under the captured lifecycle; WARNING + `plan_lifecycle_restored`; no raise |
| 5d | C1 re-stamp | implementer marks a step done *and* drops the lifecycle markers | executing task | the step stays done after the re-stamp; the captured body is never written |
| 6 | C1 round-trip | every plan fixture in `orchestrator/tests` + a live-shaped plan | — | `Plan.from_json(d).to_json() == d` |
| 7 | C2 writer | each of the 7 writers appends | real artifacts | each line has `kind` and `is_work` per C2's table; debugger writes `verify_attempt` |
| 8 | C2 reader | legacy log + an agent hand-appended `implementer` line | fixture log | legacy lines classify as the old rule did; agent line `AgentAuthored(is_work=False)`; `_has_prior_implementation` agrees with today on legacy-only logs |
| 8b | C2 consumer (judge) | completion judge over a log with an agent-written line | fixture log | prompt's "Recent Iterations" omits it and states the excluded count |
| 9 | C2 consumer | infra-resume thrash with only rebase/rederive entries between resumes | ledger at N−1 | thrash trips (today it does not) |
| 10 | C3 producer→consumer | each δ-family handler blocks (reading S5 ν's typed exits) | per-handler stub | `TerminalReport.block_category` and the filed record's `category` equal today's string (golden table) |
| 11 | C3 | legacy adapter called with `escalate_to_human=True, skip_escalation=True` | δ–λ2 window | `ValueError` naming both |
| 12 | C3 / κ | verify leg exhausts on infra-transient (S4 C2 property), then a real failure, then a bare worker crash twice | scripted verifier | first: `infra_issue`, HUMAN, today's `row_status`; second: `task_failure`; third: one re-run, then a real failure to the debugger whose report names the crash, never `infra_issue` (S4-5 ruling, built by S4 α); no instance attribute carries any of them |
| 13 | C3 / λ1 | each `_drive` ladder exception class (incl. Recut ×2, BranchCheckedOutElsewhere, `MetadataWriteFailed`) | raise from a phase | block category from the row; one `except` per outcome kind |
| 14 | C4 producer | steward session sets the task `done` / files an L1 / resolves plainly | real `TaskSteward`, scripted agent | publishes `StewardTerminalDecision(done)` / `StewardReescalatedL1(id)` / `StewardResolved` for that escalation id |
| 14b | C4 producer | steward interrupted with WIP while an L1 for the task is open | real queue | own L0 dismissed, then `StewardReescalatedL1(l1.id)` published |
| 14c | C4 producer | the row read raises after the session | failing scheduler fake | terminal arm skipped; L1 read runs; envelope `degraded=True`; WARNING names the escalation |
| 14d | C4 both sides | budget exhaustion with the steward's own L1 open | real queue | `StewardBudgetExhausted` published as produced; the workflow exits BLOCKED, as today |
| 15 | C4 both sides | steward resolves its L0 while an L1 for the task is open | real queue with an open L1 | producer publishes `StewardReescalatedL1`; consumer returns it verbatim and makes no `get_status`/`has_open_l1` call on the published path |
| 15b | C4 producer | the steward's session for L0 *X* files an L1 for the task itself, then resolves *X* | real `TaskSteward`, scripted agent, real queue | the open-L1 read after the session sees that L1; `StewardReescalatedL1` published for *X*'s id |
| 15c | C4 producer | the session for L0 *X* chains a follow-on L0 *Y* and resolves *X*; the task's lifetime budget is spent, so *Y* gives up before any session | real `TaskSteward`, scripted agent, real queue | *Y*'s L1 is filed before *X*'s outcome is published; `StewardReescalatedL1(Y's L1)` is published for *X*'s id; the workflow returns it verbatim and does not requeue |
| 16 | C4 consumer | no outcome before `steward_completion_timeout`, row `done` | fake clock | `StewardTerminalDecision(done)` |
| 16b | C4 consumer | timeout with WIP; one pending steward L0 has `filing_claimant_run_id=None` | real queue | `_settle_own_l0s(ABANDONED_ON_TIMEOUT)` dismisses it (today's `_pending_steward_l0s` scope) |
| 17 | C5 | merge future resolves `MergeOutcome(kind=SHUTDOWN)` for both callers | real worker stop | no `stranded_merge_failed` filed |
| 18 | C5 | durable failure via workflow then harness for one task | real queue | one record (second deduped); identical but `agent_role`/reason tag |
| 19 | C6 | journal row whose `advanced_sha` is not an ancestor of main | real repo + outbox | all three guards and `_resolve_branch_state` miss → merge retried |
| 19b | C6 | `is_ancestor_checked` returns `None` (git error injected through the public runner) | real outbox | WARNING names task and sha; treated as miss |
| 20 | C6 | journal row on main | real repo | `Landing(basis='journal')`; provenance `kind='merged'` as today |
| 21 | C7 | resume armed → CLI "No conversation found" | stub CLI | `ResumeFailure(stage=CLI, restore=None)` to the sink; payload strings unchanged; `RestoreOutcome.MISS.is_by_design` |
| 22 | ζ2 | in-band `_invoke` and out-of-band steward invoke for the same role/task, both holding a gate | real `routing`, real `UsageGate` | identical `routing_decision` payload keys and `inputs_digest`; `scope_capacity` present in both |
| 23 | C8 | pyright over a seeded non-conforming fake kept outside pyright's `include` | — | a test runs pyright on it as a subprocess and expects rc ≠ 0 |
| 23b | C8 / μ2 | a spec'd scheduler double in a workflow suite | — | `blocked_writes_suppressed()` falsy; the run()-exit contract judges the exit; calling a non-`Scheduler` method raises `AttributeError` |
| 23c | C8 / μ3 | a workflow suite on `spec_config()` runs to exit with the claimant heartbeat started | — | the heartbeat sleeps on a real interval; no background `TypeError`; branch 1 is gone |
| 24 | C9 | each exit kind (DONE, BLOCKED, BLOCKED infra-hold, REQUEUED, CANCELLED, ESCALATED, MERGE_DEFERRED) | real machine + recording `StatusWriter` | exactly one row write per exit, equal to the projection table; `run()` returns the machine's report |
| 24b | C9 | the writer raises a non-rejection error on the BLOCKED write | failing writer | state committed; `last_write_failure` set; `judge_exit` returns `STORE_UNAVAILABLE` |
| 24c | C9 | a harness write to the row during EXECUTE, then the move to VERIFY | real machine + recording writer | the move writes nothing; the harness-written row is not overwritten |
| 24d | C9 | the run reaches DONE and `Scheduler.mark_done` raises after its retry window | failing scheduler fake | the error propagates as today (not recorded on `last_write_failure`, no `pending`/`blocked` write from the machine); `merge-deferred`/`blocked` write failures still record and continue (ruled: Q-C5 (b)) |
| 25 | ο | b3_gate / briefing read a legacy entry and a typed entry | fixture metadata | same decisions as today for legacy; typed success entry gates identically |
| 26 | C10 / π | archive fixture with a known count of reaper promotions, `stale-strand` dismissals, dismiss-arm closes with and without a late resolution, an info L0 and a pending L0 older than 24 h; then the same fixture under each exit-1 threshold | tmp queue dir | trailing JSON equals the fixture's counts; info L0 excluded; exit 0 below every threshold, 1 above each, with the redesign line; no file under the queue dir changes |

## Cross-PRD relationship (G4)

| Other PRD / stream | Direction | Seam mechanism | Owner | Resolution |
|---|---|---|---|---|
| S10 task-write contract | consumes | C2 `MetadataWriteFailed` (β2); `SchedulerFacade`/`FakeScheduler` parity (β1); C3 slice table (η); the status-write refusal shape after the S10-3 ruling (κ1: one verdict, backend only, `actor` required) | S10 | λ1 depends on S10 β2; μ1 on S10 β1; ξ on S10 η; ν's `StatusWriter` adapts `Scheduler.set_task_status` as it stands at dispatch |
| S4 verify run context | consumes | C2 (α): failure vocabulary, `infra_transient` property, `VerifyInfraError` retired, public `is_infra_oserror`, and (S4-5 ruling) a bare worker crash re-run once then a real failure, not infra-transient; C1 (β): `VerifyContext` at the two `run_scoped_verification` call sites; the S4-1 `VerifyHost` resolved once at `TaskWorkflow` (S4's leaves) | S4 | κ depends on S4 α and β; λ1 on S4 α; κ keeps S4 α's bare-crash rule (row 12) |
| S5 GitOps ports | consumes | C3 (δ): `block_disposition.py` + derived rows; D13 (ν): typed architect exits (ruled: Q-C2) | S5 | λ1 depends on S5 δ; δ depends on S5 ν; γ before S5 ν on `read_iteration_log`. S5 ν is 20th of 21 in S5's total order, after S5 γ, which waits on S6 ζ and so on task 5040. S5 ν's own prerequisite is only that chain position, so the lead may slot it earlier at decompose (Q-C3) |
| S6 merge-lane lifecycle | consumes | C1 `MergeOutcome.kind` (β1), C3 `MainHealthAutoHeal` (δ), C4 injected `LandedOutbox` (γ1), C7 `HaltState` owner semantics (θ) | S6 | η on S6 β1, δ, θ; θ on S6 β1; ι on S6 γ1. Under the S6-2 ruling (task 5040 first), S6 δ depends on 5040 and S6 θ follows it, so η and the S9 leaves after it wait on 5040. S6 ζ and κ also wait on it; no S9 leaf consumes them (§Rulings) |
| S8 harness composition root | produces (to) and consumes | C1 (ε1 crash recovery), C5 (`_on_stranded_merge_done`, re-pointed onto C5 by θ, moves into S8's `StrandedReconciler` in ε2b; ε2b depends on θ), C6 (δ `task_ground_truth`), C7 (γ `SessionResumeStorm`, which also owns `storm_window_bound.py`'s reach-back, fk-d0c02dc767dd), `_invoke` region (κ, μ); the harness sweeps π's check measures | S9 (C1, C5–C7, C10); S8 (`stranded_reconcile.py`, the sweeps) | S8 ε2b depends on θ (S8 filed second takes the edge); asks in xrefs: ε1 after β, δ after ι, γ after ζ1, κ after ζ1, μ after ζ2. Program follow-ups listed with S8's sweeps, not tasks here: the periodic landed-journal pass (S9-5) and, only if π's check breaches, the joint L0 end-of-life redesign (S9-7) |
| S14 shared leaves | consumes | `shared.streak` (λ, landed); task 6724 (ο) rewrites `streaks.py` | S14 | ξ puts `run_length` on `shared.streak` and depends on 6724 |
| S25 scripts instruments | consumes | C4 `shared.route_rejection` (task 6766) | S25 | ζ2 depends on 6766 |
| S1 / S2 escalation | — | θ's filer is a future S1 filing-entry migration site; `suggested_action` vocabulary is S2's | S1 / S2 | xrefs |
| S3 escalation server | same file | S3 α, δ, η edit `workflow.py` (η: `_emit_workflow_verify`) | S3 | §Cross-stream same-file order |
| S7 merge gates | same file | S7 γ settle sites in `workflow.py` | S7 | §Cross-stream same-file order |
| S17 scheduler seams | — | scheduler requeue counters (part 3 of `fk-9eb149f0706d`) | S17 | out of scope here (xref) |

## Decomposition plan

**Sizing.** Overlay bands are respected except where noted. Each leaf is roughly 300–1,500 LOC with
≤ 12 declared files. π is smaller (~200 LOC, above the ~100 floor): it is an instrument the ruling needs
on main before ε lands, and folding it into ε would delay the check's filing. β and ζ2 declare 13 files,
each one over the band for a file that arrived after the draft. ζ1 also declares 13: two prose-pointer
files and the conditional `session_resume_storm.py`. These leaves cross the 15-file review trigger with
one mechanical shape each, and say so:
- λ2: 22 files;
- μ1: 16 files;
- μ2: 77 suites;
- μ3: about 59 suites;
- ν: 21 files, 11 of them the `.state =` test files.

**Ordering.** `workflow.py` is touched by β through ξ (μ3 included). Within S9 they are serialized in label order, and
the lead may reorder any S9 leaf whose only dependency is same-file (ruled: Q-C3). Cross-stream order is in
§Cross-stream same-file order. π touches no file another S9 leaf touches and has no dependency; ε depends
on it so the lead can file the 14-day check before ε lands (C10).

**Priority.** A leaf's `priority` is the highest severity in its `x_finding_key` list. The `high` leaves
are π, α, β, δ, ε, η, κ, λ1, λ2 and ν; they become dependencies of trigger-chain gate 6698 at decompose.
π carries the two L0 keys it measures (both high) as contributes; it closes neither.

**Filing metadata.** Every leaf carries `metadata.x_finding_run = review-all-dark_factory-20261008` and
`metadata.source = review-all`.

**Test seam.** Every leaf moves the tests it touches onto public seams: `TaskWorkflow.run()`'s returned
`TerminalReport`, the filed escalation record, the plan file read through `read_plan_model`, and the
events. It names what it retires in its task record.

**π — The L0 settle check script: stranded and wrongly dismissed blocking L0s from the escalation archive**
(`task_kind=normal`, priority **high**, est. ~200 LOC; ruled: OQ-5 plus Leo's check, S9-7)
- Files:
  - `scripts/check_l0_settle_outcomes.py` (new, executable)
  - `scripts/tests/test_check_l0_settle_outcomes.py` (new; row 26)
- Keys: `fk-bab6058b23c8`, `fk-e39550272df9` (contributes: measures ε; closes neither).
- Deps: none. ε depends on π.
- Seams: produces C10.
- Work:
  - C10: read-only over the archive records and the live pending queue files; two N-day windows; the
    counts and rules `digests/discussion/S9.md` §S9-7 measured the baseline with; trailing JSON; exit 0 or 1
    on the ruled thresholds, the last line naming the S8 redesign on a breach.
  - Run it once against the dark-factory queue directory with `--window-days 14` and put the output in the
    task record; that is the pre-landing reference the check compares against.
- Signal:
  - Row 26.
  - `scripts/check_l0_settle_outcomes.py --queue-dir data/escalations --window-days 14` on the live queue
    directory prints the trailing JSON, and its 30-day-equivalent stranded rate is within reach of the S9-7
    baseline (3.6 per 100); the task record carries the output.
  - Lead action: once π is on main, the lead files the C10 check task depending on ε (it cannot be filed
    earlier: the deterministic guard refuses a missing script).
- Deletes: nothing.
- G7: INV-11 (an unreadable queue is a non-zero exit, never a pass); INV-13 (the producer is the archive
  the harness sweeps already write).

**α — Typed Plan model; every plan.json writer through it; structural edits clear confirmation; the replan prompt confirms**
(`task_kind=normal`, priority **high**, est. ~1,100 LOC)
- Files:
  - `orchestrator/src/orchestrator/artifacts.py`
  - `orchestrator/src/orchestrator/mcp/plan_tools.py`
  - `orchestrator/src/orchestrator/mcp/plan_markup_stamp.py`
  - `orchestrator/src/orchestrator/agents/briefing.py` (`build_replan_prompt` ends in `confirm_plan`)
  - `orchestrator/tests/test_plan_model.py` (new; boundary rows 1–3c, 6)
  - `orchestrator/tests/test_artifacts.py`
  - `orchestrator/tests/test_plan_tools_server.py`
  - `orchestrator/tests/test_plan_tools_plan_mutex.py`
  - `orchestrator/tests/test_plan_text_fp_preserving_stamps.py`
  - `orchestrator/tests/test_briefing.py`
- Keys: `fk-37190b541196` (contributes; β closes).
- Deps: none. Task 3155, the draft's gate on α, is done (merged `ef96d32603`): α builds on its
  `ARTIFACT_ROOTS` resolver and the plan-tools `--task-root`.
- Seams: produces C1.
- Work:
  - C1 and D1, with `_base_commit` a lifecycle field (`stamped`/`revalidated` carry it) and
    `peek_plan_model` beside `read_plan_model`.
  - `_create_plan` → `recreated`; the step-set mutators → `restructured`; `update_plan_metadata` and
    `drop_plan_file` keep confirmation; `_confirm_plan` → `confirmed(by=ARCHITECT)`.
  - The 12 plan-tools writes and the markup stamp → `write_plan_model`.
  - `stamp_plan_provenance` builds `Plan.stamped`. Its `setdefault` backfill is gone. Until β lands it
    confirms an unconfirmed plan `INJECTED` with a WARNING, so no caller breaks.
  - `set_plan_files` / `bump_revalidation_stamp` → `with_files` / `revalidated`.
  - `build_replan_prompt` gains the final `confirm_plan()` instruction.
- Signal:
  - Boundary rows 1–3c and 6 pass against the real plan-tools functions.
  - A live architect run after merge leaves `_confirmed_by: "architect"` in plan.json.
  - A live replan's transcript ends in a `confirm_plan` call.
- Deletes:
  - `stamp_plan_provenance`'s `setdefault('_finalized_at', …)`.
  - every raw `.write_plan(` call in `plan_tools.py` and `plan_markup_stamp.py` (13), folded into
    `write_plan_model` (including `_read_plan_repaired`'s write-back).
  - the hand-built marker writes.
- G7: INV-2 (`PlanNotConfirmed` names its values); INV-9; INV-13 (the producer is the plan-tools process
  the architect already runs).

**β — Every plan reader on `PlanState`; finalization a precondition at every stamp site; recovered plans keep their lifecycle**
(`task_kind=normal`, priority **high**, est. ~1,200 LOC)
- Files:
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/harness.py`
  - `orchestrator/src/orchestrator/artifacts.py`
  - `orchestrator/src/orchestrator/evals/metrics.py`
  - `orchestrator/src/orchestrator/evals/plan_predicates.py`
  - `orchestrator/src/orchestrator/evals/architect_eval.py`
  - `orchestrator/src/orchestrator/evals/runner.py`
  - `orchestrator/tests/test_workflow_plan_repair_replan.py`
  - `orchestrator/tests/test_crash_recovery.py` (rows 5, 5b)
  - `orchestrator/tests/test_eval_plan_predicates.py`
  - `orchestrator/tests/test_plan_revalidation_skip.py`
  - `orchestrator/tests/test_plan_base_commit.py` (task 3157's; reads the base through the lifecycle)
  - `orchestrator/tests/test_workflow_plan_lifecycle.py` (new; rows 4, 4b, 4c, 5c, 5d)

  Thirteen files, one over the band: the one added since the draft is task 3157's test of the base it
  moves.
- Keys: `fk-37190b541196`.
- Deps: α. (Tasks 5645 and 7345 depend on β; 6141 depends on α.)
- Seams: consumes C1.
- Work:
  - D2 and D3.
  - `_plan` matches on `plan.state`. `_stamp_architect_plan`'s raw `finalized_before_invoke` marker read
    becomes `post.confirmed_since(pre)`, used only for fingerprint minting.
  - `_run_architect_for_plan`'s gate and salvage read `state`.
  - The 5 stamp sites follow D3, including `initial_plan_origin` on `TaskWorkflow`/`build_workflow` and
    the captured-lifecycle re-stamp in `_execute_iterations` (`pre` captured before the implementer runs).
  - The replan site handles a PARTIAL result per D3 (ruled: OQ-2), and keeps restoring the captured
    fingerprint and `_base_commit` through `stamped`.
  - `_repair_plan_schema`, `_check_plan_text_drift`, `_can_skip_revalidation`, `_revalidation_base`,
    `_stamped_task_text_fp` and `_escalate_plan_overwrite` read the model; the 4 call sites of
    `plan_base_commit`/`plan_task_text_fp` read `plan.lifecycle`, and the two module readers are deleted.
  - `Harness._recover_worktree_entry` (2 sites; called by `_recover_crashed_tasks`), `_fold_granted_files`
    and `_run_slot` (passes the origin) read `state` / `with_files`.
  - The evals predicates read `Plan.from_json(...).state`.
  - `_meta_root_for_worktree` becomes `artifacts.meta_root_for_worktree`.
  - α's tolerant path is removed.
- Signal:
  - Boundary rows 4, 4b, 4c, 5, 5b, 5c and 5d.
  - `git grep -E "get\('_finalized_at'\)|get\('_session_id'\)"` is empty over `workflow.py`,
    `harness.py` and `evals/`.
  - On a live task, a cap-truncated replan's block record reads `planning_stalled` with
    `PlanState partial`, and no EXECUTE entry follows.
- Deletes:
  - marker-read lines: 13 in `workflow.py`, 3 in `harness.py`, 2 in evals;
  - `artifacts.py::plan_base_commit`, `plan_task_text_fp` and `_plan_stamp`;
  - the "`_plan()` will then take the revalidation branch" prediction;
  - `workflow._meta_root_for_worktree`.
- G7: INV-3; INV-6 (an unconfirmed pass blocks with its row written); INV-11 (`PlanNotConfirmed` is
  never caught, and the re-stamp's restoration is loud).

**γ — `IterationEntry` union with writer-decided `is_work`; the completion judge reads typed entries**
(`task_kind=normal`, priority medium, est. ~1,000 LOC)
- Files:
  - `orchestrator/src/orchestrator/artifacts.py`
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/task_runtime.py`
  - `orchestrator/src/orchestrator/agents/briefing.py`
  - `orchestrator/tests/test_iteration_entry.py` (new; rows 7–9)
  - `orchestrator/tests/test_artifacts.py`
  - `orchestrator/tests/test_briefing.py`
  - `orchestrator/tests/test_workflow_resume_on_progress.py`
  - `orchestrator/tests/test_workflow_infra_thrash.py`
  - `orchestrator/tests/test_harness_plan_step_rederive.py`
  - `orchestrator/tests/test_task_runtime.py`
  - `orchestrator/tests/test_workflow_truthful_commit_stamp.py`
- Keys: `fk-9973521f4f59`.
- Deps: β. Lands before S5 ν on `read_iteration_log` (D18). Subsumes task 7391.
- Seams: produces C2.
- Work:
  - C2 and D4.
  - The 7 writers construct their kind.
  - The readers listed in C2 read `entry.is_work`/kinds.
  - `_check_infra_resume_thrash` counts work entries.
  - `_run_completion_judge` passes typed entries, and `build_completion_judge_prompt` renders orchestrator
    kinds only (`AgentAuthored` excluded and counted).
  - `task_runtime._count_iterations` (`_read_task_entry`'s `loops`) counts implementer and debugger kinds;
    its INV-2 corrupt-line refusal (task 4052) stays.
  - `iteration_entry_written_by_orchestrator` folds into the parser.
- Signal:
  - Boundary rows 7–9.
  - `git grep _iteration_entry_is_work orchestrator/src` is empty.
  - A live `iterations.jsonl` after merge has `kind` and `is_work` on every orchestrator line.
- Deletes: `_iteration_entry_is_work`, the `agent='architect'` labelling and its comment, the 7 writer dict
  literals, `append_iteration_log(dict)`, `iteration_entry_written_by_orchestrator`.
- G7: INV-1; INV-3.

**δ — `BlockReason` and the one-argument `_mark_blocked`; report-handler and recovery families converted**
(`task_kind=normal`, priority **high**, est. ~1,200 LOC)
- Files:
  - `orchestrator/src/orchestrator/workflow_types.py`
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/tests/test_block_reason.py` (new; rows 10–11, the D7 premise test)
  - `orchestrator/tests/_workflow_helpers.py` (the transitional `blocked_with(mock) -> BlockReason`)
  - `orchestrator/tests/test_workflow_already_done.py`
  - `orchestrator/tests/test_workflow_unactionable.py`
  - `orchestrator/tests/test_workflow_false_premise.py`
  - `orchestrator/tests/test_workflow_blocking_dep.py`
  - `orchestrator/tests/test_workflow_mark_blocked_l0_coverage.py`
  - `orchestrator/tests/test_workflow_terminal_report.py`
  - `orchestrator/tests/test_workflow_dry_run_hook.py`
- Keys: `fk-d2ba7601c130`; contributes `fk-b8efaa973162`.
- Deps: γ; S5 ν (D13, typed exits; same handlers and test files) (ruled: Q-C2).
- Seams: produces C3.
- Work:
  - C3 and D5–D7.
  - `_mark_blocked`'s body reads `block.*`; the legacy keyword form goes through `from_legacy_kwargs` only.
  - The 18 call sites become `BlockReason` constructions reading S5 ν's typed exit fields:
    `_handle_already_done_report` 4, `_block_on_architect_escalations` 2,
    `_handle_blocking_dep_report` 2, `_handle_unactionable_task_report` 2,
    `_handle_false_premise_report` 2, and one each in `_hold_unattributed_already_done`, `_reject`,
    `_finalise_recovery_done`, `_settle_unmergeable_branch`, `_finalise_merged_done` and
    `_finalise_without_merge_sha`.
  - The D7 premise test.
  - `_spawn_dry_run_unblock` passes the block's scalar fields to the unchanged `run_dry_run_unblock`.
  - `_mark_blocked`'s docstring prose rehomed (D16).
- Signal:
  - Boundary rows 10–11.
  - A live block from an `already_done` report files an L1 whose category equals today's.
- Deletes: the 18 sites' category, escalate and skip literals; `_mark_blocked`'s 69-line docstring (a
  pointer stays).
- G7: INV-1 (the contradictory pair is unrepresentable); INV-5; INV-2.

**ε — The steward decides every published variant; the workflow believes it; one L0 settle**
(`task_kind=normal`, priority **high**, est. ~1,000 LOC)
- Files:
  - `orchestrator/src/orchestrator/steward.py`
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/workflow_types.py`
  - `orchestrator/tests/test_steward_outcome.py`
  - `orchestrator/tests/test_steward.py`
  - `orchestrator/tests/test_mark_blocked_outcome_scoping.py` (task 5563's; its `_mark_blocked` calls move
    to `BlockReason`)
  - `orchestrator/tests/test_workflow_escalated_steward_stall.py`
  - `orchestrator/tests/test_workflow_mark_blocked_l0_coverage.py`
  - `orchestrator/tests/test_steward_publish_boundary.py` (new; rows 14–16b, 14d, 15b, 15c)
- Keys: `fk-e39550272df9`, `fk-bab6058b23c8`.
- Deps: δ; π (the check is filed before ε lands). Task 5563, the draft's other dependency, is done
  (merged `3fd6ccb3dd`): ε builds on its envelope and scoped wait.
- Seams: produces C4.
- Work:
  - C4 and D11.
  - `TaskSteward._publish_for` applies the terminal-row arm to every variant and the open-L1 arm only to
    today's `_outcome_requeues` set; `_give_up_with_wip` keeps its dismiss-then-publish; the envelope is
    `degraded` on a row-read failure.
  - The reads and the publish move to the steward's next yield point after the session, so a chained
    follow-on's pre-session give-up is seen (D11, row 15c).
  - `_await_steward_completion` returns published outcomes verbatim, keeping task 5563's scope rule
    (`_in_wait_scope`, `closed_by_non_sweep_resolver`) and its `escalation_id`/`skip_if_idle` signature.
  - The 4 workflow dismissal sites → `_settle_own_l0s` with the per-disposition scope.
  - The `StewardTerminalDecision` docstring is inverted.
  - `_outcome_requeues`' override use is removed.
  - `_outcome_severity` / `_await_steward_completion` prose rehomed (subsumes task 8027).
- Signal:
  - Boundary rows 14–16b, 14d, 15b and 15c.
  - `git grep '_outcome_requeues(' orchestrator/src/orchestrator/workflow.py` is empty.
  - The lead-filed C10 check task, 14 days after ε is done, exits 0 (ruled: S9-7).
  - `git grep -c 'dismiss=True' orchestrator/src/orchestrator/workflow.py` falls from 4 to 1.
  - On a live task where the steward sets `done`, the workflow exit follows the published
    `terminal_decision` with no `get_status` on the published path (debug log line).
- Deletes: the consumer-side status/L1 override and its L0 dismissal; 3 of the 4 dismissal sites;
  `_dismiss_l0s_covered_by_handoff` (folded in); the docstring-only producer invariant (it becomes a test).
- G7: INV-3; INV-6; INV-11 (`degraded` never fails open into the terminal variant); INV-4.

**ζ1 — `_invoke` in three parts: `ResumeArm`, invocation, accounting**
(`task_kind=normal`, priority medium, est. ~1,000 LOC, ~400 moved)
- Files:
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/resume_outcome.py` (new) (ruled: OQ-8)
  - `orchestrator/src/orchestrator/harness.py` (sink reads `RestoreOutcome.is_by_design`; `_BY_DESIGN_RESTORE_OUTCOMES` deleted)
  - `orchestrator/src/orchestrator/session_resume_storm.py` (if present at dispatch: S8 γ's set dropped in favour of `RestoreOutcome.is_by_design`)
  - `orchestrator/src/orchestrator/storm_window_bound.py` (its deferred import of `harness._BY_DESIGN_RESTORE_OUTCOMES` reads `RestoreOutcome`'s by-design members; missed by the draft, present since task 3733; S8 γ later owns the module's reach-back, fk-d0c02dc767dd)
  - `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/event_store.py` (their two prose pointers at `harness.py::_BY_DESIGN_RESTORE_OUTCOMES` repoint to `RestoreOutcome.is_by_design`)
  - `orchestrator/tests/test_resume_outcome.py` (new; row 21)
  - `orchestrator/tests/test_workflow_resume_on_progress.py`
  - `orchestrator/tests/test_session_resume_integration_gate.py`
  - `orchestrator/tests/test_transcript_archival_boundary_gate.py`
  - `orchestrator/tests/test_liveness_boundary_gate.py`
  - `orchestrator/tests/test_crash_recovery.py`
- Keys: `fk-2c0594928a18`; contributes `fk-7d410939a337` (`_pending_resume_*`) and `fk-f37f7e253cd5`
  (branch 5).
- Deps: ε.
- Work:
  - C7.
  - `arm_resume`, `_run_invocation` and `_account_invocation`.
  - `PendingResume` replaces 3 attributes.
  - The accounting reads the typed `AgentResult` fields.
  - Tests that read `_pending_resume_session_id` (43 refs, 4 files) move onto `arm_resume`'s returned
    `ResumeArm` and the `session_resume_failed` event.
  - `_run_invocation` keeps task 5386's `invocation_scope(self.config, role)` call.
- Signal:
  - Row 21.
  - `TaskWorkflow._invoke` is ≤ 300 lines (AST; 755 today).
  - `python -c 'import orchestrator.storm_window_bound'` still imports, and its sampler still carves out
    `disabled` and `miss`.
  - `git grep "getattr(result, 'resume_fallback_session_ids'" orchestrator/src` is empty.
  - A live `session_resume_failed` event's payload strings are unchanged.
- Deletes:
  - the `restore_outcome` string and the 55-line comment run above it;
  - the three `_pending_resume_*` attributes;
  - branch 5's `getattr`s;
  - `harness._BY_DESIGN_RESTORE_OUTCOMES`.
- G7: INV-9 (one restore vocabulary). The new module imports downward only (heuristic 13).

**ζ2 — One route resolver with one scope-capacity read; MCP-injection helpers leave `workflow.py`**
(`task_kind=normal`, priority medium, est. ~800 LOC)
- Files:
  - `orchestrator/src/orchestrator/routing_dispatch.py`
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/mcp_lifecycle.py` (ruled: OQ-7)
  - `orchestrator/src/orchestrator/steward.py`
  - `orchestrator/src/orchestrator/review_checkpoint.py`
  - `orchestrator/src/orchestrator/harness.py`
  - `orchestrator/src/orchestrator/dry_run_unblock.py`
  - `orchestrator/src/orchestrator/evals/architect_eval.py`
  - `orchestrator/src/orchestrator/evals/shadow_isolation.py` (imports the moved composer; task 5386's module)
  - `orchestrator/tests/test_routing_dispatch.py`
  - `orchestrator/tests/test_workflow_routing_decision.py`
  - `orchestrator/tests/test_out_of_band_routing.py`
  - `orchestrator/tests/test_routing_parity.py` (new; row 22)

  Thirteen files, one over the band: `evals/shadow_isolation.py` arrived with task 5386 after the draft.
- Keys: `fk-5ea52a0f89ad`; contributes `fk-a503ccb37b4b` (part c).
- Deps: ζ1; task 6766 (S25 C4, same files).
- Work:
  - D12.
  - `_invoke` calls `resolve_and_record_route(..., usage_gate=self.usage_gate, plan_shape=...)`, and the
    five out-of-band callers pass their gate.
  - The two inject helpers move to `mcp_lifecycle.py` with their composer `compose_invocation_mcp_config`
    and `PER_INVOCATION_MCP_SERVERS` (task 5386); steward and evals import them at module level.
- Signal:
  - Row 22.
  - `git grep -E 'def (_trailing_24h_window|_routing_inputs_digest|_ceiling_spend_by_model|_record_routing_decision|_scope_capacity_snapshot)' orchestrator/src/orchestrator/workflow.py`
    is empty.
  - Live `routing_decision` events from an architect invocation and from a steward invocation both carry
    `scope_capacity`.
- Deletes: the 4 routing copies; `_scope_capacity_snapshot`; `steward.py`'s function-local import of a
  `workflow.py` private.
- G7: INV-5; INV-11 (telemetry isolation kept and logged).

**η — Merge-side blocks from `OutcomeKind`; the merge stash goes**
(`task_kind=normal`, priority **high**, est. ~1,200 LOC)
- Files:
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/workflow_types.py`
  - `orchestrator/src/orchestrator/unblock_types.py`
  - `orchestrator/tests/test_workflow_merge_block_table.py` (new)
  - `orchestrator/tests/test_workflow_merge_thrash.py`
  - `orchestrator/tests/test_workflow_train_attribution.py`
  - `orchestrator/tests/test_workflow_train_halt_owner.py`
  - `orchestrator/tests/test_workflow_merge_phase_block_status.py`
  - `orchestrator/tests/test_workflow_merge_blocked_surfacing.py`
  - `orchestrator/tests/test_merge_queue_auto_heal.py`
  - `orchestrator/tests/test_workflow_ready_to_merge.py`
- Keys: `fk-d2ba7601c130` (merge half), `fk-44bc973b1cc4` (merge half), `fk-7d410939a337` (merge half);
  contributes `fk-b8efaa973162`.
- Deps: ζ2; S6 C1 (β1), S6 C3 (δ), S6 C7 (θ: `HaltState`; θ deletes `_submit_halt_owning_escalation` and
  edits `test_workflow_train_halt_owner.py`); task 4838.
- Work:
  - One `OUTCOME_KIND_BLOCK` table.
  - The 28 merge-family sites: `_submit_to_merge_queue` 9, `_handle_merge_gate_escalations` 3,
    `_resolve_and_resubmit` 2, `_attribute_train_failure` 2, `_check_merge_outcome_thrash` 2,
    `_auto_heal_main_health` 2 (or S6 δ's mapping), and one each in `_block_on_operator_merge_cancel`,
    `_maybe_enqueue_group_merge`, `_run_merge_phase`, `_block_on_owned_main_break`,
    `_escalate_train_halt`, `_handle_wip_recovery_no_advance`, `_handle_unmerged_state` and
    `_handle_stash_failed`.
  - `MergeOutcome` feeds the thrash signature directly.
  - `_spawn_dry_run_unblock` reads `block.block_class`. η deletes the prose sniff if `dry_run_unblock`'s
    defensive fallback is its last caller.
- Signal:
  - A golden table per `OutcomeKind`.
  - `git grep -E '_last_merge_(block_reason|failure_category|failure_cause_hint)' orchestrator/src` is empty.
  - A live merge-verify-red block records `block_class=merge_verify_red` from the table.
- Deletes: the three `_last_merge_*` attributes (15 refs); the merge families' category literals; the
  `'ff' in reason` fallback (unless S6 β1 already removed it).
- G7: INV-5; INV-12.

**θ — Merge-done classification and the `stranded_merge_failed` filer in `stranded_verified_green.py`**
(`task_kind=normal`, priority medium, est. ~650 LOC)
- Files:
  - `orchestrator/src/orchestrator/stranded_verified_green.py`
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/harness.py` (`Harness._on_stranded_merge_done` re-pointed onto C5; `Harness._file_stranded_merge_failed` deleted)
  - `orchestrator/src/orchestrator/recovered_merge_failure.py` (only if task 7277 has landed it)
  - `orchestrator/tests/test_stranded_verified_green.py`
  - `orchestrator/tests/test_workflow_ready_to_merge.py`
  - `orchestrator/tests/test_stranded_merge_done_parity.py` (new; rows 17–18)
- Keys: `fk-108cdd00d122`.
- Deps: η; S6 C1 (β1); task 7277 (same files; pending, branch `task/7277` 0 commits ahead at e634cfe658). Task 3354 and S8 ε2b depend on θ (same-file; ε2b then moves the re-pointed method into `stranded_reconcile.py`). Subsumes 7985.
- Seams: produces C5.
- Work: C5 and D13. `_on_architect_merge_done` and `Harness._on_stranded_merge_done` keep scheduling and
  their own side effects only. If `recovered_merge_failure.py` exists, its classification calls
  `classify_merge_future`.
- Signal:
  - Rows 17–18.
  - `git grep -E 'def _file_(architect|stranded)_merge_failed' orchestrator/src` is empty.
  - After a live drain-restart with a stranded-reaper merge in flight, no new `stranded_merge_failed`
    appears in `get_pending_escalations`.
- Deletes: the two filers (112 + 80 lines) and the two future classifications.
- G7: INV-5; INV-12 (UNCLASSIFIED stays loud).

**ι — One corroborated landing reader for the recovery guards and the ground-truth journal rung**
(`task_kind=normal`, priority medium, est. ~650 LOC)
- Files:
  - `orchestrator/src/orchestrator/corroborated_landing.py` (new) (ruled: OQ-9)
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/task_ground_truth.py` (the `_resolve_branch_state` journal rung only)
  - `orchestrator/tests/test_corroborated_landing.py` (new; rows 19–20)
  - `orchestrator/tests/test_workflow_merge_provenance.py`
  - `orchestrator/tests/test_workflow_recovery_landing_evidence.py`
  - `orchestrator/tests/test_workflow_finalise_without_merge_sha.py`
  - `orchestrator/tests/test_task_ground_truth.py`
- Keys: `fk-a152341e38ed`.
- Deps: θ; S6 C4 (γ1); task 7277 (`task_ground_truth.py`). Subsumes 7769.
- Seams: produces C6.
- Work:
  - C6.
  - The three guards and `_resolve_branch_state` call `corroborated_landing`.
  - `_finalise_recovery_done` takes the `Landing`.
  - The unreachable `NOT_LANDED`/`is_ancestor` breadcrumb in `_recover_before_merge` goes.
- Signal:
  - Rows 19–20.
  - A stale write-ahead row on a live restart retries the merge (`merge_enqueued`), not
    `escalation_created`.
- Deletes: the journal-first-then-git policy restated in four readers; the breadcrumb.
- G7: INV-3; INV-11 (indeterminate is logged); INV-13.

**κ — Execute, verify and planning legs return `BlockReason`; the stash dicts go**
(`task_kind=normal`, priority **high**, est. ~1,200 LOC)
- Files:
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/tests/test_workflow_verify_infra_resume.py`
  - `orchestrator/tests/test_workflow_zero_output_hang.py`
  - `orchestrator/tests/test_workflow_resume_on_progress.py`
  - `orchestrator/tests/test_execute_server_error_requeue.py`
  - `orchestrator/tests/test_workflow_preexisting_main_break.py`
  - `orchestrator/tests/test_architect_all_committed_plan.py`
  - `orchestrator/tests/test_liveness_boundary_gate.py`
  - `orchestrator/tests/test_verify_scope_inversion_boundary.py`
  - `orchestrator/tests/test_workflow_chronic_flake.py`
  - `orchestrator/tests/test_workflow_skew_surfacing.py`
  - `orchestrator/tests/test_phase_legs_boundary.py` (new; row 12)
- Keys: `fk-44bc973b1cc4`, `fk-7d410939a337`.
- Deps: ι; S4 C2 (α) and S4 C1 (β), which edit `_run_scoped_verification_with_infra_retry` and the two
  `run_scoped_verification` call sites first.
- Work:
  - D8.
  - The 15 sites: `_execute_verify_review_loop` 7, `_handle_no_plan_failure` 3,
    `_check_infra_resume_thrash` 2, and one each in `_run_architect_for_plan`,
    `_validate_prerequisites_or_block` and `_block_on_inherited_break`.
  - The verify leg returns `VerifyResult | BlockReason` from the `infra_transient` property, and keeps S4
    α's bare-crash rule (one re-run, then a real failure for the debugger; never infra-transient; S4-5
    ruling).
  - The VERIFY→REVIEW emission takes the tip and the verdict runner (task 6670's `TaskVerifyPort` path).
  - The 10 test files reading the stash move onto `TerminalReport` and the filed record.
- Signal:
  - Row 12.
  - `git grep -E '_infra_hold_info|_inherited_break_info|_zero_output_hang_info|_progress_resume_churn_info' orchestrator/src`
    is empty.
  - `git grep -E '_verify_green_result|_verify_green_runner|_verify_checkpoint_hit|_last_verify_result' orchestrator/src/orchestrator/workflow.py`
    is empty.
- Deletes: the 4 stash dicts, `_last_verify_result`, `_verify_green_result`, `_verify_green_runner` (task
  6670, new since the draft), `_verify_checkpoint_hit`, the
  reset-at-entry code, the precedence comments, and `_infra_hold_info`'s 4 never-read keys.
- G7: INV-2; heuristic 5.

**λ1 — The `_drive` ladder onto disposition rows; `MetadataWriteFailed` blocks; two dead columns go**
(`task_kind=normal`, priority **high**, est. ~900 LOC)
- Files:
  - `orchestrator/src/orchestrator/workflow_types.py`
  - `orchestrator/src/orchestrator/block_disposition.py` (S5's)
  - `orchestrator/src/orchestrator/git_ops.py` (one field per `WarmLaneUnavailable` member)
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/dry_run_unblock.py`
  - `orchestrator/tests/test_block_disposition.py`
  - `orchestrator/tests/test_workflow_e2e.py`
  - `orchestrator/tests/test_ladder_rows_boundary.py` (new; row 13)
- Keys: `fk-4f7a933c0dd4`, `fk-67b5971ed72d`; contributes `fk-b8efaa973162`.
- Deps: κ; S5 C3 (δ); S10 C2 (β2); S4 C2 (α).
- Work:
  - D9 and D10.
  - `_drive`'s 9 sites; one clause per outcome kind.
  - 5 isinstance re-switch arms (after S4 α removes the `VerifyInfraError` arm) become rows.
  - `TerminalReport.category` deleted.
  - `dry_run_unblock`'s cap path reads `BlockClass.AGENT_FAILURE` directly.
  - The unreachable-row count is re-measured at dispatch.
- Signal:
  - Row 13.
  - `git grep -E 'isinstance\(e, OSError\) and _?is_infra_oserror' orchestrator/src/orchestrator/workflow.py`
    is empty.
  - `git grep -E 'block_class=BlockClass' orchestrator/src/orchestrator/workflow_types.py orchestrator/src/orchestrator/block_disposition.py orchestrator/src/orchestrator/git_ops.py`
    is empty.
- Deletes: the re-switch arms; 2 dead columns; `TerminalReport.category`; the unreachable rows.
- G7: INV-1; INV-12; INV-11.

**λ2 — The keyword form is deleted; the remaining keyword-form tests migrate**
(`task_kind=normal`, priority **high**, est. ~900 LOC; one mechanical shape across ~21 test files, so the
review trigger is acknowledged)
- Files:
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/workflow_types.py`
  - `orchestrator/tests/_workflow_helpers.py` (deletes `blocked_with`)
  - the keyword-form test files no earlier leaf declares, measured at d40ba048e9 by AST plus kwargs reads,
    plus one added since (task 7668):
    - `test_auto_eval_block_category.py`
    - `test_convert_to_blocked.py`
    - `test_harness_action_dispatch.py`
    - `test_info_l0_router_workflow.py`
    - `test_main_health_fix_owner.py`
    - `test_rebase_branch_reset_guard.py`
    - `test_reconcile_stranded.py`
    - `test_recut_ruling_dispatch.py`
    - `test_scheduler.py`
    - `test_simple_task_workflow.py`
    - `test_suggestion_triage.py`
    - `test_workflow_merge_retry_pending.py`
    - `test_workflow_merge_superseded.py`
    - `test_workflow_status_on_resume.py`
    - `test_workflow_train_completion.py`
    - `test_workflow_train_state_escalation.py`
    - `test_workflow.py`
    - `test_workflow_state_machine_boundary.py`
    - `test_workflow_plan_files_gate_own_work_base.py` (task 7668; reads a patched mock's kwargs)

    The leaf re-measures the list at dispatch and drops any file whose hit is a false positive.
- Keys: `fk-b8efaa973162`.
- Deps: λ1.
- Work:
  - Delete `from_legacy_kwargs`, `_mark_blocked`'s 11 keyword parameters and `blocked_with`.
  - Every remaining test asserts on `TerminalReport` or the filed record instead of a patched
    `_mark_blocked`'s kwargs.
- Signal: `git grep -E 'escalate_to_human: bool = False|skip_escalation: bool = False|from_legacy_kwargs' orchestrator/src`
  is empty; the suite is green.
- Deletes: the adapter, 11 parameters and the transitional reader.
- G7: INV-1; tests stance.

**μ1 — Protocol-typed seams, hand-written fakes, conformance; fixture-tolerance branches deleted**
(`task_kind=normal`, priority medium, est. ~1,000 LOC)
- Files:
  - `orchestrator/src/orchestrator/workflow_types.py` (ports)
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/dry_run_unblock.py` (`DryRunScheduler` annotation)
  - `orchestrator/tests/_workflow_helpers.py` (public builder)
  - `orchestrator/tests/test_port_conformance.py` (new; row 23)
  - `orchestrator/tests/fixtures/port_conformance_negative.py.txt` (new; a non-`.py` data file so it stays outside pyright's `include`, which covers `tests`; row 23's test copies it to `tmp_path` and runs pyright on it as a subprocess)
  - `orchestrator/tests/test_workflow_merge_provenance.py`
  - `orchestrator/tests/test_workflow_finalise_without_merge_sha.py`
  - `orchestrator/tests/test_workflow_state_machine_boundary.py` (15 `_merge_recovery_basis` refs)
  - the 7 `object.__new__(TaskWorkflow)` files:
    - `test_eval_boundary_suite.py`
    - `test_eval_memory_isolation.py`
    - `test_mcp_post_transport.py`
    - `test_workflow_chronic_flake.py`
    - `test_workflow_completion_memory.py`
    - `test_workflow_sandbox_events.py`
    - `test_workflow_sandbox_refusal.py`

  This is 16 files: one mechanical shape, review trigger acknowledged.
- Keys: `fk-f37f7e253cd5` (branches 3, 6, 7).
- Deps: λ2; S10 β1 (`SchedulerFacade`/`FakeScheduler` parity).
- Seams: produces C8's ports.
- Work:
  - Ports and fakes with conformance.
  - Delete branch 3 (`_filing_claimant_run_id`'s `getattr` for `object.__new__` fixtures), branch 6
    (`_steward_progress_counter`'s older-fake tolerance) and branch 7 (`_merge_recovery_basis`; 39 test
    refs move onto `mark_done`'s `kind`).
  - The 7 files build through the public builder.
- Signal:
  - Row 23.
  - `git grep -E "getattr\(self, '_process_run_id'|getattr\(steward, 'metrics'" orchestrator/src/orchestrator/workflow.py`
    is empty.
  - `git grep _merge_recovery_basis orchestrator/src` is empty.
  - `git grep 'scheduler: Any' orchestrator/src/orchestrator/dry_run_unblock.py` is empty.
- G7: INV-1; tests stance.

**μ2 — A spec'd scheduler double replaces the bare mocks; the double-tolerance branches go**
(`task_kind=normal`, priority medium, est. ~1,300 LOC; one mechanical shape across the 77 suites, so the
review trigger is acknowledged)
- Files:
  - `orchestrator/tests/_orch_helpers.py` (`spec_scheduler`)
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/dry_run_unblock.py`
  - `orchestrator/tests/test_spec_scheduler.py` (new; row 23b)
  - the 77 test files that construct `TaskWorkflow(` with a bare `MagicMock()`/`AsyncMock()` scheduler.
    They are measured at e634cfe658 (unchanged from the draft) by `git grep -l 'TaskWorkflow('` intersected with a bare-mock
    scheduler assignment, and listed in the task record at decompose.
- Keys: `fk-f37f7e253cd5` (contributes: branches 4, 9, the two dry-run trims, the `is True` check; μ3 closes).
- Deps: μ1. Subsumes 6962.
- Work:
  - C8's helper.
  - The 77 suites use it.
  - Then delete branch 4 (`_exit_status_facts`' catch of a raising `get_status`) and the `is True`
    identity check. Branch 1 is μ3's: it tolerates a mock config, not a mock scheduler.
  - Delete branch 9 (`dry_run_unblock`'s non-awaitable `get_task` except) and the two "bare-MagicMock
    scheduler" trims.
- Signal:
  - Row 23b.
  - `git grep -E 'blocked_writes_suppressed\(self\.task_id\) is True' orchestrator/src/orchestrator/workflow.py`
    is empty.
  - The exit-contract guard judges the exit in a sampled migrated suite (row 23b).
- G7: INV-10 (the guard stays able to fail); tests stance.

**μ3 — A spec'd config double replaces the bare config mocks; branch 1 goes**
(`task_kind=normal`, priority medium, est. ~700 LOC; one mechanical shape across ~59 suites, so the
review trigger is acknowledged)
- Files:
  - `orchestrator/tests/_orch_helpers.py` (`spec_config`)
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/tests/conftest.py` (its note on the heartbeat guard)
  - `orchestrator/tests/test_spec_config.py` (new; row 23c)
  - the test files that construct `TaskWorkflow(` with a bare `MagicMock(…)` config: 59 measured at
    `e634cfe658` by `config\s*=\s*(MagicMock|AsyncMock)\(`; 56 of them are also
    μ2 suites and 4 set `claimant_heartbeat_interval_secs`. The leaf re-measures and lists them in the task
    record.
- Keys: `fk-f37f7e253cd5` (closes: branch 1, after μ1 and μ2 delete the rest).
- Deps: μ2 (same files).
- Work:
  - `spec_config(**overrides) -> OrchestratorConfig`: real defaults with overrides.
  - Migrate the bare config doubles onto it.
  - Then delete branch 1, the `isinstance(interval, (int, float))` guard in `_claimant_heartbeat_loop`,
    and its comments in `conftest.py` and `_orch_helpers.py`.
- Signal:
  - Row 23c.
  - `git grep -E 'isinstance\(interval, \(int, float\)\)' orchestrator/src/orchestrator/workflow.py` is empty.
- G7: INV-10; tests stance.

**ν — `WorkflowStateMachine` is the single writer of phase and row**
(`task_kind=normal`, priority **high**, est. ~1,400 LOC; ruled: OQ-3, Q-C5 (b))
- Files:
  - `orchestrator/src/orchestrator/workflow_types.py`
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/exit_contract.py`
  - `orchestrator/src/orchestrator/harness.py` (`_run_slot`'s defensive `getattr(workflow.state)`)
  - `orchestrator/tests/test_workflow_state_machine.py`
  - `orchestrator/tests/test_workflow_state_machine_boundary.py`
  - `orchestrator/tests/test_workflow_terminal_report.py`
  - `orchestrator/tests/test_exit_contract.py`
  - `orchestrator/tests/test_repend_state_machine.py`
  - `orchestrator/tests/test_single_writer_boundary.py` (new; rows 24, 24b, 24c, 24d)
  - the 11 further `.state =` files (48 assignments in 14 files at e634cfe658):
    - `test_auto_eval_block_category.py`
    - `test_harness_cancel_workflow.py`
    - `test_harness_warm_lane_wiring.py`
    - `test_info_l0_router_workflow.py`
    - `test_suggestion_triage.py`
    - `test_task_completed_block_context.py`
    - `test_workflow_cancellation.py`
    - `test_workflow_skew_surfacing.py`
    - `test_workflow_truthful_exits.py`
    - `test_workflow_worktree_missing.py`
    - `test_workflow_task_verify_port.py` (task 6670, new since the draft)

  That is 21 files: one mechanical shape (`.state =` → `initial_state=`), review trigger acknowledged. The
  leaf adds `test_workflow_e2e.py` if its refs remain after μ2.
- Keys: `fk-ee708be4436f`; contributes `fk-f37f7e253cd5` (branch 8).
- Deps: μ3.
- Seams: produces C9.
- Work:
  - C9.
  - The workflow's own-row `set_task_status` calls route through `machine.enter`.
  - `run()` returns `machine.terminal_report`.
  - `initial_state=` replaces `force_set` and the setter (48 `.state =` assignments in 14 test files).
  - `_note_exit_status_write`'s 11 calls collapse into `machine.last_write_failure`.
  - `judge_exit`'s phase comparison is deleted; its outcome/status half reads `last_write_failure`.
  - Done and cancelled writes stay on the store's done and cancel paths; `Scheduler.mark_done` keeps
    raising past its retry window (the S9-5 contract sentence). No failed write is queued for retry.
- Signal:
  - Rows 24, 24b, 24c and 24d.
  - `git grep 'report_phase != machine_state' orchestrator/src/orchestrator/exit_contract.py` is empty.
  - `git grep -E 'def force_set|self\._terminal_report = ' orchestrator/src` is empty.
- Deletes: the `_terminal_report` stash (4 sites); the hand pairs (3); 10 literal own-row writes;
  `force_set` and the setter; the phase comparison.
- G7: INV-6; heuristic 10; INV-11 (a failed write is recorded and judged, never silent).

**ξ — Retry budgets: one run-length rule on `shared.streak`, one steward guard clear, a typed first-enqueue time**
(`task_kind=normal`, priority medium, est. ~450 LOC)
- Files:
  - `shared/src/shared/streak.py`
  - `shared/tests/test_streak.py`
  - `orchestrator/src/orchestrator/workflow.py`
  - `orchestrator/src/orchestrator/steward.py`
  - `shared/src/shared/task_metadata.py` (or its S10 η home)
  - `orchestrator/src/orchestrator/verify.py` (one import)
  - `orchestrator/tests/test_workflow_retry_ledger.py`
  - `orchestrator/tests/test_steward.py`
  - `orchestrator/tests/test_workflow_merge_thrash.py`
- Keys: `fk-9eb149f0706d`.
- Deps: ν; S10 C3 (η); task 6724 (S14 ο rewrites `streaks.py`'s cause-change layer onto `KeyedStreaks`).
- Work:
  - A pure `run_length(prev_signature, prev_count, signature) -> int` beside `KeyedStreaks`, consumed by
    `_evaluate_merge_thrash`, the verify failure-signature window and 6724's cause-change layer.
  - The steward's three maps are cleared by one `_clear_guards(esc_id)`.
  - `RetryLedger.merge_first_enqueued_at: float | None`. Legacy `str(epoch)` values coerce on read; the
    lax model accepts them: 2,981 of the 3,245 stored values in the task store are strings today (read-only
    store read, 2026-11-10).
  - `_normalize_cause_hint` / `_compute_merge_outcome_signature` delegators deleted.
- Signal:
  - `git grep 'from orchestrator.workflow import _normalize_cause_hint' orchestrator/src` is empty.
  - `def run_length` exists once (in `shared/src/shared/streak.py`).
  - A live ledger serializes `merge_first_enqueued_at` as a JSON number.
- Deletes: two copies of the rule; the triple-pop at 3 sites; the two delegators.
- G7: INV-5.

**ο — `DryRunProposal`: one typed entry for `metadata.dry_run_proposals`**
(`task_kind=normal`, priority medium, est. ~700 LOC)
- Files:
  - `orchestrator/src/orchestrator/unblock_types.py`
  - `orchestrator/src/orchestrator/dry_run_unblock.py`
  - `orchestrator/src/orchestrator/b3_gate.py`
  - `orchestrator/src/orchestrator/agents/briefing.py`
  - `skills/escalation-watcher/SKILL.md`
  - `skills/unblock-low-risk/SKILL.md`
  - `orchestrator/tests/test_dry_run_unblock.py`
  - `orchestrator/tests/test_b3_gate.py`
  - `orchestrator/tests/test_briefing.py`
  - `orchestrator/tests/test_merge_queue_dry_run_unblock.py`
- Keys: `fk-e43472e7d5d7`.
- Deps: μ2 (same file `dry_run_unblock.py`).
- Work:
  - A frozen `DryRunProposal` with `status: ProposalStatus` (explicit success member), one
    `investigated_at`, a `reaffirmed_at`, an always-present `cost_usd`, and an embedded `BlockRecord`.
  - `to_metadata()`/`from_metadata()`; legacy entries parse.
  - `b3_gate.check_proposal` and `_format_prior_proposal` read the model.
  - The skill's hand-built entry is replaced by a pointer to the writer's shape.
- Signal:
  - Row 25.
  - `git grep "'timestamp':" orchestrator/src/orchestrator/dry_run_unblock.py` is empty.
  - `git grep "'status' in entry" orchestrator/src/orchestrator/b3_gate.py` is empty.
  - A live block's newest proposal carries `status` and one timestamp key.
- G7: INV-9; INV-13.

**ω — Integration gate: the boundary matrix on real fixtures; heuristic-14 re-measure of `workflow.py`**
(`task_kind=normal`, priority medium, est. ~900 LOC, tests only plus a docstring pointer)
- Files:
  - `orchestrator/tests/test_workflow_spine_integration_gate.py` (new)
  - `orchestrator/tests/_spine_fixtures.py` (new)
  - `orchestrator/src/orchestrator/workflow.py` (module docstring pointer only)
- Keys: `fk-a503ccb37b4b`, `fk-a71254176a81`.
- Deps: π and α–ο (all).
- Work:
  - All boundary rows run together on real fixtures. A row that fails because production is wrong is
    escalated naming the row and the owning leaf; it is never weakened.
  - Re-measure `workflow.py` per heuristic 14: prose home first, then a named partition with the h13
    symptom per candidate. Record it in the task record and leave a one-line pointer.
  - Perform no split.
- Signal: the gate module passes in the merge gate, and the recorded measurement names line count, prose
  share and each candidate's symptom or "none".
- G7: INV-10; INV-13.

### Cross-stream same-file order (ruled: Q-C3)

Measured over the sibling drafts at fold time; awk over each draft's leaf blocks, so a few hits are
mentions rather than edits. Under Q-C3 (option 1) the lead builds one per-file serial order at decompose, in seam
order. The rules:

- An upstream leaf whose contract an S9 leaf consumes precedes it. Those are the semantic edges named in
  each leaf's Deps.
- An S9 leaf that only shares the file is slotted by readiness.
- S8's leaves follow the S9 leaf named in its xref.

In the Edge rule column, `A → B` reads "A lands before B" (B depends on A), the reverse of the program's
edge notation; "X after Y" means X depends on Y. In program notation the named edges are δ → S5 ν,
S8 ε2b → θ and ε → π.

| Shared file | Sibling leaves (draft at fold time) | Edge rule |
|---|---|---|
| `orchestrator/src/orchestrator/workflow.py` | S3 α, δ, η; S4 α, β, ι1, μ0, ξ; S5 α, ε, λ, ν; S6 β1, γ1, γ2, δ, θ; S7 γ; S8 α, κ; S10 β1, β2, ε, ξ | semantic: S4 α/β → κ; S4 α → λ1; S5 ν → δ; S6 β1 → η, θ; S6 δ, θ → η; S6 γ1 → ι; S10 β2 → λ1; S10 β1 → μ1; S8 α, κ after ζ1/ζ2; the rest by readiness |
| `orchestrator/src/orchestrator/workflow_types.py` | S4 α; S5 α, δ | S4 α, S5 δ → λ1; S5 α by readiness |
| `orchestrator/src/orchestrator/artifacts.py` | S4 λ; S5 ν, ξ | γ → S5 ν (D18); S4 λ, S5 ξ by readiness |
| `orchestrator/src/orchestrator/steward.py` | S5 α; S8 θ | by readiness |
| `orchestrator/src/orchestrator/stranded_verified_green.py` | S5 ι; S8 ι, ω; S10 β1/β2 | by readiness; S8's after θ |
| `orchestrator/src/orchestrator/task_runtime.py` | S5 ι | by readiness |
| `orchestrator/src/orchestrator/dry_run_unblock.py` | S5 δ; S10 β1/β2 | S10 β2 → λ1; S10 β1 → μ1; S5 δ by readiness |
| `orchestrator/src/orchestrator/routing_dispatch.py` | S10 β1/β2 | by readiness with ζ2 |
| `orchestrator/src/orchestrator/mcp_lifecycle.py` | S8 μ | S8 μ after ζ2 |
| `orchestrator/src/orchestrator/unblock_types.py` | S6 β1 | S6 β1 → η |
| `orchestrator/src/orchestrator/storm_window_bound.py` (S9 ζ1, added in this fold) | S8 γ (owns the module's reach-back, fk-d0c02dc767dd); not S8 ι2, the event-store task (S8-3 ruling: its guard test allow-lists the module's raw `FROM events` read as an offline tool; the module does not move onto `EventStore`) | ζ1 → S8 γ (S8's xref: γ after ζ1); no edge with S8 ι2 |
| `orchestrator/src/orchestrator/evals/shadow_isolation.py` (S9 ζ2, added in this fold) | none in the sibling drafts | by readiness |
| `orchestrator/src/orchestrator/harness.py` (S9 β, ζ1, ζ2, θ, ν) | S3 β, δ, ε1, ε2, ζ, κ; S4 α, ι3, ξ; S5 γ, ε, ι, κ, λ0, ξ0, π; S6 α, β1, γ1, θ, ι, κ, μ; S8 most leaves (ε1a/b, ε2a/b, γ, κ, μ1 …); S10 β1, β2, ε, κ2, ξ | S8 ε1 after β; S8's other harness leaves after the S9 harness leaf named in each xref; S8 ε2b after θ (S8 filed second takes the edge), and S3 ε1 after S8 ε2b (lead, 2026-11-10), so θ also precedes S3 ε1; the lead's harness.py order (S6 α, S3 β/δ/κ, S6 β1 … μ, S3 ε1/ε2/ζ) slots the S9 leaves by readiness |
| `orchestrator/src/orchestrator/task_ground_truth.py` (S9 ι) | S5 α; S6 γ1; S8 δ, ω; task 7277; task 6615 (in progress 2026-11-10, `_pid_alive`) | 7277 → ι → S8 δ; S6 γ1 → ι (C4); 6615 lands first; the rest by readiness |
| `orchestrator/src/orchestrator/agents/briefing.py` (S9 α, γ, ο) | S8 μ2; S10 η | by readiness |
| `orchestrator/src/orchestrator/stranded_reconcile.py` (not S9's; listed for the ruling) | S8 ε2a, ε2b (creates it) | θ → S8 ε2b (S8 filed second takes the edge; θ edits the method in `harness.py` before it moves) |

## Live tasks (contract §8) and lead actions

Step 2 (`search_tasks` ≥ 0.6, confirmed by `get_task` or a read-only store read; re-run 2026-11-10 for π,
α/β, ε and the S9-5 follow-up) found no live task carrying any leaf's whole remedy. The closest matches are
already done: 2248, 2249, 2246 and 2245 (W9), 5563 and 3157. Statuses below were re-read on 2026-11-10.

| Task | Status | Relation | Action at decompose |
|---|---|---|---|
| 7391 | pending, low | subsumed by γ: typed parse plus the judge prompt, its consumer (2) | cancel, `x_acceptance_reason` naming γ |
| 7985 | pending, low | subsumed by θ (`SHUTDOWN` → transient) | cancel naming θ |
| 7769 | pending, low | subsumed by ι | cancel naming ι |
| 6962 | pending, low | subsumed by μ2 (spec'd scheduler double) | cancel naming μ2 |
| 8027 | pending, low | subsumed by ε | cancel naming ε |
| 3155 | done (merged `ef96d32603`, 2026-11-02) | `artifacts.py`, `plan_tools.py` | none: the draft's edge α → 3155 is dropped |
| 5563 | done (merged `3fd6ccb3dd`, 2026-11-07) | ε builds on its envelope and scoped wait | none: the draft's edge ε → 5563 is dropped |
| 7370 | pending, low (a task-5563 follow-up: progress-refreshed wait bound; a synthesized timeout honours an open L1) | same functions as ε | 7370 depends on ε |
| 3174 | pending, low | `_wait_for_resolution` | 3174 depends on ε |
| 6766 | pending, medium (S25 λ) | same files; C4 | ζ2 depends on 6766 |
| 4838 | pending, high | merge-halt handler region | η depends on 4838 |
| 3354 | pending, medium | same files: `workflow.py`, `harness.py`, `test_workflow_ready_to_merge.py` (its open items are the tree-hash check and the routing-tier bump, item 3 still awaiting a design call; its cancelled-future fix landed with task 3031) | 3354 depends on θ |
| 7277 | pending, high (branch `task/7277`, 0 commits) | plans `recovered_merge_failure.py` (a third merge-outcome classifier) and edits `task_ground_truth.py` | θ and ι depend on 7277; its classification calls C5 if it lands first |
| 6724 | pending, medium (S14 ο) | `streaks.py` | ξ depends on 6724 |
| 5645 | pending, medium | the recovered-plan path β edits | 5645 depends on β (rebases onto the typed state; avoids a priority inversion on the high chain head) |
| 7345 | pending, low | `_plan` wipe paths, `briefing.py` | 7345 depends on β (same reason) |
| 6141 | pending, low | `_confirm_plan`, `artifacts.py`, `plan_tools.py` | 6141 depends on α |
| 4163, 7319 | pending, low / medium | `stranded_verified_green.py` submit path, which θ leaves untouched | both depend on θ |
| 7052, 7054 | pending, medium (defect-tests ε, ζ2) | `plan_tools.py`; `_execute_verify_review_loop` | same-file edges with α / β, κ; direction by status at decompose |
| 5310 / 3151 | pending / deferred | pair 5 of fk-108cdd00d122 | none |
| 5543, 5200 | pending, medium | the harness's stale-L0-strand sweep (a periodic mid-run sweep; its threshold): not π's work, but either one landing inside the check's windows moves what C10 counts as stranded | none; the lead names them in the check task's description |
| 4646, 4651 | pending, high | the landed-merge journal (keep the row for parked tasks; the periodic landed-not-done reconciler). 4651 carries a temporary parking edge on 5040 (lead, 2026-11-10), which S8's decompose replaces with 4651 → S8 ε2a | none here: the S9-5 follow-up (a periodic landed-journal pass after 4646, narrowing 4651) is a program-level follow-up listed with S8's sweeps |
| 5040 | pending (half-built branch; lock-starved since 2026-11-04) | not S9's. S6 δ and (through S5 γ → S6 ζ) S5 ν depend on it, so η and δ do too, transitively (§Rulings) | none here; the lead's open risk |

Lead actions outside the table:

- **File the C10 check task** once π is on main, with the fields C10 lists, depending on ε (ruled: S9-7).
  The deterministic guard refuses it before the script exists, and ε depends on π, so it is filed before
  ε lands and its 14-day clock starts when ε is done.
- **Route the S9-5 follow-up** (a periodic landed-journal pass with a one-hour give-up to an L1, after task
  4646; task 4651 narrows to landed parked tasks without a journal row) to the harness plan's sweeps or the
  program's follow-up list. It is not a task of this plan.

## Held leaves

None. Every upstream that S9 consumes is in wave 2 and decomposes before S9 in the seam order, or is
already filed (6766, 6724, 7277; 3155 and 5563 have since landed). Not held is not unblocked: at dispatch, δ
(through S5 ν) and η (through S6 δ and θ) wait on task 5040 transitively, and so does each S9 leaf after
them in label order (§Rulings).

## Out of scope

- Splitting `TaskWorkflow` into modules. ω measures.
- The harness's orphan-L0 reaper freshness arms (S8, asked by xref). The workflow's `merge_phase_liveness`
  stamp writers become deletable once S8 drops their reader.
- Scheduler requeue counters (S17), and a full `RetryBudget` registry.
- Pair 4 and pair 5 of `fk-108cdd00d122`.
- Typing `suggested_action` (S2), and typing every plan body field.
- Typed architect-exit reports (S5 ν, D18) (ruled: Q-C2).
- A queue-and-retry of failed status writes (asked at the sitting, S9-5): it would replay a stale row over
  one the harness, an operator or a steward has moved, and no dark-factory done write was lost to a store
  outage in the 30 days measured. The narrow form, a periodic pass over the landed-merge journal with a
  one-hour give-up to an L1 after task 4646, is a program-level follow-up listed with S8's sweeps.
- The joint L0 end-of-life redesign with S8 (option B of OQ-5). It is opened only if the C10 check breaches.

## Open questions (tactical — decided in the named leaf)

1. **Harness preservation of a `PARTIAL` plan** (β). Today only a stamped plan is preserved. **Suggested:**
   preserve `PARTIAL` too, since `_plan`'s completion pass resumes it. Add a crash-recovery row.
2. **Success status spelling on disk** (ο). `status: 'proposed'` or `'ok'`. **Suggested:** `'proposed'`.
3. **Where the moved prose lands** (every leaf). The commit message by default.
4. **`TaskWorkflow._invoke` size target** (ζ1). ≤ 300 lines; a fourth helper is acceptable.
5. **`DryRunScheduler`'s member list** (μ1). Measured by AST.
6. **λ2's file list** (λ2). Re-measured at dispatch; the AST/kwargs rule is the starting list.
