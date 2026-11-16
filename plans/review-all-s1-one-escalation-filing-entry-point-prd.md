# S1 — One escalation filing entry point (PRD)

**Status:** draft (author pass, critic fold 2026-11-11, ruling fold 2026-11-13 after the sitting of
2026-11-12; §Rulings), stream S1 of the `/review-all` remediation program
`plans/review-all-program-dark_factory-2026-10-10.md` (wave 3, size L, risk medium). Implements ruling
**R8** and promotes R22 invariant 1. B + H: a contract section and a two-way boundary-test sketch (G5
fired on every arm: four packages, more than 8 mechanisms, a load-bearing seam, and S2/S12/S3/S6/S7
consume it). Twenty-five leaves and the integration gate ω, which mints the invariant; λ1, λ2 and μ (eval
hermeticity and the corpus-cell sandbox) were added by Leo's rulings and λ2 mints a second invariant.

**Code anchors** verified against main `a85cb1ad78` (2026-11-13). The run's anchors were verified at
`6551b75aba`; the author pass re-read every key at `1f240e6da2` and the critic fold at `ba17934861`. The
ruling fold re-read every symbol on the 47 named paths that changed between `ba17934861` and
`a85cb1ad78` (§Rulings, last bullet) and re-measured each count with one AST scan run on both trees, so a
count is labelled `measured at a85cb1ad78` and, where it moved, carries its `ba17934861` value beside it.
The fold-verifier re-checked at main `4a87282ee7` (2026-11-13, 19 commits later): three paths the plan
names changed (`server.py`'s `merge_status` adds `merge_sha` to an answer, `dashboard/data/memory_evals.py`
was refactored and still joins open `eval_regression` escalations on `dedupe_fingerprint`, and
`evals/judge.py` lost a dead function); no count, symbol or check moved, and the 62 producer checks still
fail there.
Main moves fast: cite by symbol and re-locate at implementation time. Measurement scripts live outside
the repo (the program's state home); the rule each count used is stated beside it.

Edge notation everywhere: "X → Y" means X depends on Y (Y lands first).

## Goal

Every escalation record any process writes into a queue directory is written by one function,
`escalation/src/escalation/filing.py::EscalationFiler.file` (or its two context-bound twins), and the
filer's choices along the four axes of variability (heuristic 3) are made once, in one typed value:
how it dedupes, which level it is born at, what identity a system-level filing uses, and whether the
blocking queue I/O leaves the event loop. The ladder rules that today live only in the MCP server's
closure (`escalation/src/escalation/server.py::create_server` → `_chokepoint_or_submit`, measured 184
lines) run for every filer: eval-lane containment, the filing-claimant stamp, note-to-file, the
born-at-L2 rule with its agent-role downgrade, terminal-task auto-resolve and dedupe.

What a user (an operator, an agent, the next implementer) observes when this lands:

- An agent's `escalate_blocker`/`escalate_info` call returns exactly the response shapes it returns
  today (`queued`, `dedup_skipped`, the auto-resolved shape, `accepted_unpersisted`), now produced by
  `FilingOutcome.tool_response()`.
- A harness storm alarm, a merge-lane alarm, a scheduler watchdog, a workflow block, a steward
  give-up, a recon finding and a fused-memory write-path alarm each pass through the same gates: a
  born-at-L2 severity filed by a role that is not a sentinel role is downgraded to `blocking`
  everywhere, not only at the MCP door; a watcher's `promote_to_l2` still creates a level-2 record,
  because promotion is a named policy, not a severity; an eval-lane task's record at level 1 or 2
  is contained everywhere, and the door does not contain an in-process level-0 record (a backstop for
  the future: at main an eval cell's workflow has no queue, so it files nothing); a filing on the
  orchestrator's event loop never runs the queue's flock'd scan or fsync on the loop thread.
- An eval run writes no production state (Leo's hermeticity rule, 2026-11-12): its agent transcripts
  and verify logs land under the eval worktree root, the eval cell factory refuses a production queue,
  filer, event store or cost store, corpus cells run under the production sandbox, and a boundary row
  shows a recorded eval cell leaving the production queue directory, `runs.db`, the transcript archive,
  the verify-log directory and the task store unchanged (λ1, λ2, μ).
- Every `TaskWorkflow` filing emits `escalation_created` (6 of 17 do not today), so `runs.db`
  consumers such as `evals/reviewer_trial/mining.py` stop under-counting.
- A recon escalation category is declared once, with its severity, fold rule and identity; the
  `_RECON_DEDUP_CONFIG` tuple and `_escalate`'s inline severity tuple no longer exist.
- `uv run --project shared pytest scripts/tests/test_design_invariants_consistency.py` passes with the
  new invariant `escalation-filing-one-door` in every pinned site, and its guard
  (`escalation/tests/test_filing_one_door.py`) fails on a seeded fixture that constructs an
  `Escalation` outside `escalation/filing.py`, and on one that resubmits a record rebuilt with
  `Escalation.from_dict` or `dataclasses.replace`.
- The next storm alarm is written by building an `EscalationDraft` and choosing a `FilingPolicy`. The
  agent writing it does not need to know `BORN_AT_L2_SEVERITIES`, the server closure, which
  `DedupeConfig` categories exist or which call is off-loop, because none of those choices is left to
  the call site.

## Background

The run `review-all-dark_factory-20261008` found (fk-7d1f3a0d12ca, `dimensions.1`) that filing an
escalation varies along four independent axes and that every filing site decides each by hand; and
(fk-333277fb4bd7, fk-0d31f1114911) that the store is written through two doors with different rules:
the MCP chokepoint with five gates, and `EscalationQueue.submit`, a field-agnostic atomic write
reached directly by every in-process filer. The per-area findings are slices of the same axis bundle:
the merge lane's one-method port (fk-2e0a4419290b), the scheduler's three mechanisms
(fk-cd446ff3da74), the workflow's 17 hand-built filings (fk-9c72e4399b4e), fused-memory's on-loop
emitters (fk-1bf4bfa8be04), the recon category vocabulary (fk-b7b0be25c6b4), the server's site-by-site
`to_thread` hops (fk-32903f2e64e0), and the copy-pasted "file then block, swallow" tail
(fk-0ee68c0948ab).

Main has moved ~10,000 commits since the run, and the axis has grown, not shrunk. Constructions are
counted by an alias-resolved AST scan of every Load-context call of `escalation.models.Escalation` in
tracked non-test Python, so `from escalation.models import Escalation as EscModel` counts. The ruling
fold ran one scan (the program's state home, `scratch/S1-rulingfold/scan.py`) over both trees, scoped
to `*/src`, `fused-memory/scripts`, `scripts` and `shared/src`, tests excluded, innermost def; where its
rule differs from the author pass's, both values are given.

| Measure | At `6551b75aba` (run) | At `ba17934861` | At `a85cb1ad78` (this fold) |
|---|---|---|---|
| `Escalation(...)` constructions | 112 (orch 83, fm 21, esc 8) in 38 modules | 123 (orch 95, fm 17 + 1 script, esc 10) in 51 modules | **122 (orch 93, fm 17 + 1 script, esc 11) in 51 modules**: task 8926 moved two orchestrator constructions into `escalation/file_or_refresh.py` |
| `.submit(`/`.submit_resolved(` calls on a queue-named receiver | 95 (orch 82, fm 13) | 115 by the author's rule; 114 by this fold's | **112** (orch 91, esc 9, fm 9, fm scripts 2, scripts 1), plus the steward's re-escalation submit |
| defs with each dedup read: `has_open_l1` / `get_by_task` / `submit_or_dedupe` / `find_pending_l2_by_root_cause` / `file_or_refresh` | 33 / 28 / 13 / 5 / — | 36 / 27 / 13 / 5 / — (author's rule); 41 / 65 / 14 / 8 / — (this fold's) | **41 / 61 / 14 / 8 / 2** (this fold's rule) |
| `level=` keyword at construction: 1 / 2 / default / 0 / computed | 56 / 29 / 16 / 6 / 5 | 69 / 28 / 15 / 3 / 8 | **68 / 27 / 15 / 3 / 9** |
| non-escalation constructions stamping `filing_claimant_run_id` | 48 of 104 | 56 of 113 (orch 56/95, fm 0/18) | **54 of 111** (orch 54/93, fm 0/18) |
| non-test `.has_open_l1(` calls (AST) | — | 41 in 19 modules (the author's "43 lines" counted two prose lines) | **41 in 19 modules** |
| `EscalationQueue(...)` constructors in `*/src` | 17 | 18 (esc 1, fm 14, orch 3) | **16** (esc 1, fm 12, orch 3): task 6234 injected the queue factory into `TargetedReconciler` (`c79f2a3332`); plus 7 in scripts |
| distinct dunder task-id literals | ~29 | 39 in 24 modules (author's exclusions); 44 in 25 (this fold's) | **44 in 26 modules** (this fold's rule: quoted `__[a-z][a-z0-9_-]*__`, Python dunders excluded); α measures the registry at dispatch |
| `file_folded_escalation` callers | 8 | 12 in 11 fm modules | **12 in 11** (11 direct calls, plus the ticket janitor's `asyncio.to_thread(file_folded_escalation, …)`) |
| `asyncio.to_thread(` in `server.py` | 11 → 10 | 11 | **11** |
| draft keyword union over the constructions | — | 21 filer fields | **22**: `review_due` (task 6390, `e87228b063`) |

Since the run, five more single-policy filing helpers appeared, each owning one point of the dedup
axis: `orchestrator/src/orchestrator/sentinel_l1.py::file_sentinel_l1` (open-L1-per-sentinel, 3
callers), `orchestrator/src/orchestrator/born_at_l2.py::file_deduped_born_at_l2` (one-L2-per-root-
cause, 1 caller), `escalation/src/escalation/dedupe.py::file_or_fold_l1` (content fingerprint, 5
callers), `escalation/src/escalation/guard_storm_alarm.py::file_storm_once` (one open alarm per
category+role, 2 callers) and, since the draft, `escalation/src/escalation/file_or_refresh.py::file_or_refresh`
(task 8926, done 2026-11-12: one pending record per sentinel and level, refreshed in place; 2 callers,
`startup_steps.py::_file_or_refresh_l1` and `harness.py::Harness._file_dirty_tree_escalation`). Each is
the right shape for one policy; together they are the 2^N-flag growth heuristic 3 warns about. They
become `FilingPolicy` values here.

Program rulings applied without relitigating: **R8** (this stream: option 1, the queue-layer filing
function with injected lookups, owning the five chokepoint gates and the `to_thread` hop;
`EscalationQueue.submit` becomes its private write step; the 95 submit sites migrate here; its wording
is clarified by OQ-1 and its argument name amended by OQ-10, §Rulings), **R22** item 1 (the invariant is
promoted; this PRD mints it in ω), **R10** (S3 owns the server/queue boundary; S1's filer sits beside
S3's registration modules), **R11** (S7's `GateMeasurement` reporter files through a sink S1 absorbs,
per the lead's addendum), **R54** (one per-file order, stated here as edges onto wave-2 task ids; edges
that rest only on a shared test file are not wired, lead 2026-11-12), **R55** (new
`merge_lane/worker.py` tasks park behind 8528). No S1 key is in R23's weakened list. Leo's two standing
rules of 2026-11-12 bind it: evals are realistic within their scope and hermetic (λ1, λ2, μ), and each
decision area has one partition between code and LLM judgement (invariant
`one-decision-partition-per-area`, INV-23, minted by S17 ν; C2: every step is code's).

## Findings disposition

One row per key of the stream; the dedup protocol (contract §8) ran on 2026-11-11 and again at the
ruling fold on 2026-11-13: step 1 (`find_tasks_by_metadata(key="x_finding_key", value=<key>)`) returned
no task for any of the 11 keys (the carried key fk-14e1510916ab returns task 6609, pending); step 2's
hits are dispositioned in §Live tasks. Every count was re-measured at `a85cb1ad78`; a count that moved
since `ba17934861` says so.

| Key | Sev | Disposition | Note (re-verified at `a85cb1ad78`) |
|---|---|---|---|
| `fk-7d1f3a0d12ca` four axes chosen per site | high | leaf α (mechanism) + ω (closes) | Grown: 122 constructions in 51 modules (123 at `ba17934861`), 5 new single-policy helpers (table above; task 8926 added the fifth since the draft). Every migration leaf contributes; ω's guard proves zero constructions outside `filing.py`. |
| `fk-333277fb4bd7` two doors, rules only at the MCP door | medium | leaf α (+ ω) | Still present: `EscalationQueue.submit` now refuses an unregistered category and a note-to-file record (`queue.py::EscalationQueue.submit`), but none of the five ladder gates. 54 of 111 non-escalation constructions stamp the claimant (56 of 113 at `ba17934861`; the two that moved into `file_or_refresh` take the claimant as an argument); 1 filing-time eval-lane re-implementation (`orchestrator/mcp/markup_sink.py::make_escalation_sink`; the two harness reads are consumers of the containment outcome, see κ2/η1/η2). |
| `fk-0d31f1114911` ladder invariants live only in the closure | high | leaf β (+ α) | Still present: `_file_markup_residue` and `promote_to_l2`'s create path call `queue.submit` directly, and `submit.py` keeps its own born-at-L2/role refusal. The prior row's "own copy of `_HARNESS_SENTINEL_ROLE_PREFIXES`" sub-claim is fixed (`submit.py` imports `models.HARNESS_SENTINEL_ROLE_PREFIXES`); `_is_harness_sentinel_role` still lives in `server.py` (task 6609's residue, folded into α). |
| `fk-32903f2e64e0` loop-hosted blocking queue, hops chosen per site | medium | leaf α | Still present: 11 `asyncio.to_thread` in `server.py`; the flock'd writes stay inline on the orchestrator loop by design until the notify callback is thread-safe (`dedupe.py` comment above `submit_or_dedupe_off_loop`). α's filer owns the hop for every filing; S3 β (9157) supplies the thread-safe callback delivery α needs. Read/mutate tool hops stay per tool in S3's modules (out of scope, §Out of scope). |
| `fk-2e0a4419290b` `EscalationPort` too narrow | medium | leaves ε1, ε2a, ε2b (closes), ε3 | Still present: 1 filing through the port (`SpeculativeMergeWorker` → `EscalationRecord`); 16 hand-built filings outside it (worker 5, seven satellites 11), one of which dedupes through `submit_or_dedupe`; 16 `escalation_queue.submit(` lines; 13 code-line `has_open_l1(` calls (14 grep lines, one in a `liveness.py` docstring); `get_by_task(` 5 grep lines, of which 3 are escalation-queue calls (`liveness.py::_resolve_pending_alarm`, a resolve sweep; `skew_tripwire.py::emit_pipeline_landing_tripwire`; `worker.py::_alarm_verify_worktree_contention`), one is a docstring and one an unrelated `get_by_task` method in `types.py`. S7 α1 (9136) adds `StormEscalationSink`, fed the raw queue; ε1 absorbs it (lead addendum). S6's filed C2/C3/C7 (θ 9176, δ 9171, φ 9186) move three more lane filings into `halt.py`, `main_health.py` and `item_lifecycle.py` on the raw queue; ε3 absorbs them. |
| `fk-cd446ff3da74` scheduler's three mechanisms | medium | leaves ζ1, ζ2 | Grown to four: 7 `SchedulerCallbacks` escalation hooks at 11 await sites; 3 inline `Escalation(...)` (`_file_fm_read_storm_escalation`, `_maybe_escalate_backfill_overstay_rate`, `trigger_retry_cap_exhausted`); and 3 `file_sentinel_l1` calls (new since the run), plus `work_product.py`'s `SentinelL1` builder handed an `escalate` callable by `scheduler.py`. |
| `fk-9c72e4399b4e` 17 hand-built workflow filings | medium | leaf δ (+ ε3 for the halt-owner filings S6 moves into the lane) | Still present: 17 constructions in 17 defs, 6 keyword sets (was 5); the same 6 file with no `escalation_created` (`_escalate_corruption`, `_escalate_unreconciled_done_step`, `_file_main_health_l1` (renamed from `_auto_heal_main_health`), `_attribute_train_failure`, `_file_halt_owning_l1`, `_file_non_owning_halt_blocked_record`); `_file_architect_merge_failed` still omits `workflow_state` and `worktree`, and is the one born-at-L2 workflow filing (role `orchestrator-architect-desync-merge`). S6 θ 9176 moves the WIP-halt filings (`_handle_wip_conflict`, `_handle_wip_recovery`, `_file_halt_owning_l1` through `_submit_halt_owning_escalation`) into the lane and S6 δ 9171 moves `_file_main_health_l1`'s path into `MainHealthAutoHeal`; δ re-derives the set that remains in `workflow.py` at dispatch. |
| `fk-1bf4bfa8be04` fm emitters on the loop | medium | leaves θ1a, θ1b | Still present: `server/tools.py::_report_unverified_claims` (sync) is called from async `add_memory`/`add_episode` and files through `file_folded_escalation`; `server/markup_guard.py`'s `_file_residue` (sync) is reached from `_guarded_call_tool`. 3 "ASYNC ON PURPOSE — do not re-inline" comments (tools.py 1, memory_service.py 2) plus 1 "to_thread IS LOAD-BEARING". `file_folded_escalation` has 12 call sites in 11 modules (the ticket janitor passes it to `asyncio.to_thread`); whether each runs on a loop thread is measured by θ1a. |
| `fk-b7b0be25c6b4` recon category vocabulary in three homes | medium | leaf ι1 | The run's live drift is fixed (`recon_resume_failure_storm` left with the resume pass, task 5514, `e95931e5de`), but the structure stands: 8 call-site categories over 10 `_escalate` calls, the 8-entry `_RECON_DEDUP_CONFIG.infra_dedupe_categories`, the 3-entry severity tuple inside `ReconciliationHarness._escalate`, and the comment above the config still says "all four". A ninth category still needs three edits. |
| `fk-0ee68c0948ab` swallows around `set_task_status` | medium | leaves γ (helper + runner 8), δ (workflow 4), ζ2 (2), η1 (3), η2 (6), η3 (1) | Grown: 24 non-reraising `except Exception` arms around `set_task_status` (deterministic_runner 8, workflow 4, harness 12; AST: `Try` whose body calls `set_task_status`, handler `Exception`/bare, no `Raise`). 12 are file-then-block tails, 12 are bare status writes. Every one becomes a typed `StatusWriteOutcome` (C6); task 3887's race (a block write after recon disposed the task) is closed by the same composition. Arms per leaf, by the module each lives in after wave 2: γ — the runner's five `_file_*_and_block` tails, `_rearm_predicate`, `_writeback_deploy_success`, `run`; δ — `_handle_terminal_exit_on_block`, `_persist_blocked_row`, `_repend_for_requeue`, workflow `_revert_withheld_member`; ζ2 — `_block_and_escalate_external_dep`, `_block_and_escalate_delivered_check` (two of the seven callbacks, in `dispatch_gate_filers.py` after S8 ζ1a); η1 — `_block_and_escalate_cross_repo`, `_block_and_escalate_substrate_flip`, `_block_and_file_dispatch_gate_l1` (`dispatch_gate_filers.py`); η3 — `_reconcile_one_stranded` (`stranded_reconcile.py` after S8 ε2a); η2 — `_block_if_still_pending`, `_cascade_unblock_member` ×2, `_maybe_auto_eval` ×2, harness `_revert_withheld_member` (they stay in `harness.py`). |
| `fk-1b573c2428c2` withheld train member stranded silently | high | `fixed:2ef04c8241` | Task 4737's `AbandonedMemberSweep` (`orchestrator/merge_deferred_recovery.py`, wired in `Harness._recover_abandoned_merge_deferred` on the stranded-reconcile pass, `2ef04c8241` 2026-10-15) owns `merge-deferred`: a member with no train custody, live request, workflow, claimant/pin or active declared peer is re-driven after two consecutive sightings and escalated. A member left `merge-deferred` by a failed `_revert_withheld_member` is such a member, so it is no longer silent or unbounded. The residual swallow in `_revert_withheld_member` is one of fk-0ee68c0948ab's 24 arms (η2 / δ). The two test docstrings still claiming "unreachable by every recovery path" are corrected by η2/δ, which touch those files' subjects. No leaf for this key. |

Fold note (critic B2, B3, M5, M6, M13 and the S6 contracts): the per-module assignment was re-derived at
`ba17934861`, and re-checked at `a85cb1ad78` (only `harness.py::Harness._file_dirty_tree_escalation` and
`startup_steps.py::_file_or_refresh_l1` moved, onto `file_or_refresh`), from the alias-resolved scan and from where each filed wave-2 task moves each filer
(S8 γ/ε2a/ε2b/ε3/ζ1a/ζ1b/ζ2, S9 θ, S6 δ/θ/φ). Every construction and every swallow arm above names the
leaf that converts it in the module it will live in; §Decomposition plan's "Harness accounting" table
lists the 30 harness constructions one by one.

Carried mechanical key (not one of the 11): `fk-14e1510916ab` (Phase-5 task 6609, "move
`is_harness_sentinel_role` into `models.py`") — the prefix tuple moved since; the predicate did not. α
needs it in `models.py` for the filer's downgrade gate, so α carries the key and 6609 is cancelled into
α (§Live tasks).

## Pre-conditions for activating

- **S3 β (task 9157) lands before α.** α moves the queue write off the event loop. `queue.submit` fires
  `_notify_callback`, which in production is `harness.py::Harness._on_escalation` →
  `asyncio.Event.set()`, not thread-safe (`dedupe.py`'s comment above `submit_or_dedupe_off_loop`
  states the hazard). S3 β's C4 makes `set_notify_callback(cb, *, deliver_on=loop)` deliver with
  `call_soon_threadsafe`, and wires the harness's registration with its loop. α depends on S3 ι 9181
  (server.py order), which depends on β through θ and γ; the PRD names 9157 so the substrate is
  explicit (G3). Task **4567** (the seven in-process archive-inclusive `get_by_task` reads off the
  loop), which the draft listed upstream through S3 θ 9177, is done (`b255f96b3e`, 2026-11-12); S1
  moves only the filing writes and their dedup lookups off the loop, so nothing of 4567 is redone here.
- **S2's vocabulary, cascade-parent and callback leaves land before α** (Leo's S2-3 (a), applied to
  S1/OQ-4 by the lead). α edits `escalation/models.py` after S2 α (status, severity and resolve-action
  enums; S2 α itself waits on S19 θ0's `shared/escalation_layout.py`), and after S2 β1 and β2
  transitively through 9181 → 9177 → S2 β2 (the lead keeps 9177 → S2 β2 under S2-3 (a)). S2's leaves that
  change what the door writes (its filer-declared facts, its resolve-action table, its caller-identity
  leaf with the `filed_via` stamp, and its `DedupeConfig` rename) land after S1 β (§Cross-PRD).
- **S7 α1 (task 9136) lands before ε1.** It creates `merge_lane/gate_measurement.py` and the
  three-method `StormEscalationSink` that ε1 deletes (absorbed into the widened port).
- **S6 θ, δ and φ (9176, 9171, 9186) land before ε3.** They create `merge_lane/halt.py`'s
  `HaltState.hold_with_escalation`/`file_holder_escalation`, `merge_lane/main_health.py::MainHealthAutoHeal`
  and `merge_lane/item_lifecycle.py::LifecycleRefusalResponder`, each filing on the raw queue (S6 C7,
  C3, C2); ε3 puts them on the widened port (C10).
- **S8 ζ1a/ζ1b/ζ2 (9226, 9227, 9228) and the rest of S8 land before ζ2 and η1/η2/η3.** S8 moves the
  harness filers into `dispatch_gate_filers.py`, `infra_filers.py` and `escalation_record_sweeps.py`;
  S8's PRD names those modules as "the repoint site for `escalation/filing.py::file`". S8 γ, ε2a/ε2b and
  ε3 (9220, 9223, 9224, 9225) and S9 θ (9213) move five more harness filings and one swallow arm into
  `session_resume_storm.py`, `stranded_reconcile.py`, `deterministic_recon.py` and
  `stranded_verified_green.py`; η3 converts them. ζ2 and η depend on S8 ω (9236), the last wave-2 task
  on `harness.py`.
- **S9 ε (9190) and S8 θ (9229) land before κ4.** They are the pending wave-2 tasks on `steward.py`;
  S9 ε's boundary row 15c relies on the steward's level-1 give-up landing before the steward publishes
  its previous outcome, which κ4 keeps under an awaited filing (S1/OQ-9, lead).
- **Task 5040 and 8528.** Every S1 leaf that waits on S8 ω, S3 ε1, S4 μ0/ρ, S5 ο0, S6 ζ/μ or S9 ν waits
  on 5040 transitively (Program status item 1). ε1, ε2a, ε2b and ε3 declare `merge_lane/worker.py` and
  so wait on 8528 (R55), stated as an edge. α, β, γ, θ1a, θ1b, θ2, ι1, ι2, κ1 and κ2 do not wait on
  5040 (γ, θ1a, ι1 and ι2 wait on S10 tasks; the hand carry of task 4867 that held that chain (R47) is
  done, `daff96ef3c`). λ1, λ2 and μ wait on no door leaf. They chain among themselves on
  `evals/runner.py` (μ → λ1, λ2 → μ, λ2 → λ1; lead 2026-11-13, same-file rule); λ1 and μ wait on no
  live task, and λ2 waits on the end of the wave-2 `workflow.py` chain (9232, which is behind 5040 and
  8528).
- No novel substrate otherwise: every other capability α assumes is on main (§Contracts cite them;
  the manifest binds them). The eval tasks' substrate is on main too: `evals/runner.py::build_eval_orch_config`,
  `evals/runner.py::_cell_workflow`, `evals/profile.py::EVAL_PROFILE` and its parity tripwire,
  `evals/snapshots.py::eval_worktree_root`, `orchestrator/tests/test_eval_boundary_suite.py` (B1–B10)
  and `evals/shadow_invocation.py::build_shadow_orch_config`, which already turns the sandbox on.

## Sketch of approach

**α — the door.** `escalation/filing.py` (new) owns the input value (`EscalationDraft`), the policy
value (`FilingPolicy` with a `Dedup` sum type), the outcome value (`FilingOutcome`) and the filer
(`EscalationFiler`). The gate bodies leave `server.py`'s closure and become the filer's ordered,
private steps; `escalate_info`/`escalate_blocker` build a draft and choose a policy (the
`review_suggestions` and `terminal_state_is_the_bug` bypasses become policy choices of the tool, not
category tests inside the gates). `EscalationQueue.has_open(task_id, *, level, ...)` is the
level-parameterised lookup (`has_open_l1` forwards to it until ω). The sentinel registry
(`escalation/models.py::Sentinel`) holds every system-identity task id byte-identical to today's
literal, dunder and hyphenated prefix forms alike, so open records keep matching; `subject_for(task_id)`
turns a string from an MCP tool or the CLI into the typed subject. The filer mints ids itself
(`make_id` moves into the hop). `RefreshOpen` refreshes an open record in place through S3 C4's
`_mutate_pending` (D13), and `dedupe.py::attach_or_submit` splits so the fallback write is the filer's
(D6).

**β — the rest of the escalation package.** The server's other filers (amendment-truncation and
root-cause over-fold storms, markup residue and storm, `promote_to_l2`'s create step under the named
promotion policy, the `merge_cancel` workflowless-cancel L1), `note_to_file.py`,
`guard_storm_alarm.py`, `submit.py`'s CLI and `mismatch_streak.py` file through the filer. After β,
`escalation/src` constructs `Escalation` only in `filing.py` (and `dedupe.file_or_fold_l1`, kept until
ι2 migrates its last fused-memory callers).

**γ — the status-write outcome.** `orchestrator/status_write.py` (new): `write_status` returns a typed
`StatusWriteOutcome` (written / rejected / failed), and `file_and_block` composes a filing with the
block write, re-reading status first (INV-3) and resolving the just-filed record when the task was
disposed meanwhile (task 3887's race). `DeterministicRunner`'s five `_file_*_and_block` tails and its
three other swallow arms use them; `status_write.py` imports nothing from `deterministic_runner.py`, so
it is the lower-module filer task 8355 asked for.

**δ — the workflow.** `TaskWorkflow._file_escalation(draft_fields…, policy, agent_role=…)` fills the
workflow's fields once (task id, worktree, workflow state, claimant), files, and emits
`escalation_created` for every outcome that wrote a record. The workflow filings that remain in
`workflow.py` after S6 θ/δ and the 4 swallow arms route through it and γ.

**ε1/ε2a/ε2b/ε3 — the merge lane.** `EscalationPort` gains `file(record, policy)` (async) and
`file_soon(record, policy)` (sync, for code running on the loop thread such as S7's
`GateUnmeasuredReporter.settle`); `ProductionEscalations` wraps one `EscalationFiler`;
`RecordingEscalations` implements the port and records `(draft, policy)` pairs, so lane tests assert on
the policy the lane chose. ε1 converts the worker's own filings and absorbs `StormEscalationSink`; ε2a
and ε2b convert the satellite modules and delete the raw `escalation_queue` threading for filing; ε3
puts S6's halt-owner, auto-heal and lifecycle-refusal filings on the port (C10) and deletes
`EscalationRecord`.

**ζ1/ζ2 — the scheduler.** ζ1 turns the three inline sites, the three `file_sentinel_l1` calls and
`work_product.py`'s builder into drafts + policies over a lookup-less filer and deletes
`sentinel_l1.py`. ζ2 replaces the seven escalation hooks with one `SchedulerCallbacks.escalations` port
(C5) and deletes their harness/S8 implementations.

**η1/η2/η3 — the harness.** After S8's and S9's moves, every filer in `dispatch_gate_filers.py`,
`infra_filers.py` and `escalation_record_sweeps.py` (η1), the strata that stay in `harness.py` and
`verify_runner.py` (η2), and the filings S8/S9 move into `session_resume_storm.py`,
`stranded_reconcile.py`, `deterministic_recon.py` and `stranded_verified_green.py` (η3) builds a draft
and chooses a policy; `born_at_l2.py` is deleted (its template is `FilingPolicy.born_at_l2(root_cause)`);
the 12 harness swallow arms go through γ.

**θ1a/θ1b/θ2 — fused-memory filers.** `file_folded_escalation` (blocking) and `afile_folded_escalation`
(async) become policy adapters over the filer (θ1a); the on-loop callers (`_report_unverified_claims`,
the markup guard's residue/storm sinks, and every other caller θ1a measures on a loop thread) use the
async one; `file_blocking`'s on-loop refusal makes the mistake impossible to repeat. θ1b moves the
remaining callers onto typed anchors. θ2 converts the six escalators that construct their own queue,
including TicketJanitor's dedup-outage alarm (task 6689). All three work at the paths the fused-memory
package-layering moves leave (S19 γ).

**ι1/ι2 — reconciliation.** `fused_memory/reconciliation/recon_escalations.py` (new) is the typed
registry (`ReconEscalation(category, severity, folds, identity)`) and the one declaration of a recon
alarm's severity and fold rule; `ReconciliationHarness._escalate` takes a registry member;
`_RECON_DEDUP_CONFIG` and the severity tuple are derived from it. ι2 converts the other recon filers,
routes the backfill script's resubmit through a queue mutator and deletes `dedupe.file_or_fold_l1`.

**κ1/κ2/κ3/κ4 — the remaining orchestrator modules**, grouped by which wave-2 chains they wait on; κ4
is the steward's level-1 re-escalation.

**λ1/λ2 — eval hermeticity (Leo, 2026-11-12/13).** The containment above keeps an eval record from
reaching a human or the task store if one ever reaches a production queue; it does not keep it out of
the production queue directory, which the dashboard's escalation analytics count with no eval filter.
Hermeticity is enforced where an eval cell is built: the eval profile (`evals/profile.py::EVAL_PROFILE`,
the single declared divergence surface, guarded by its parity tripwire) and the eval cell factory
(`evals/runner.py::_cell_workflow`). λ1 (eval package only) adds `transcript_archive.root` to the
profile, resolved under `evals/snapshots.py::eval_worktree_root`; makes the cell factory refuse a
workflow built with a production escalation queue, steward factory, event store or cost store; and adds
an eval-side zero-write row to the eval↔production boundary suite. λ2 gives the workflow's verify-log
archive root a seam (today hard-coded as `project_root / 'data' / 'verify-logs'` at the workflow's
verify call) that the eval factory sets under the eval worktree root, widens the zero-write row to the
verify-log directory, and mints the invariant `eval-writes-no-production-state`.

**μ — corpus eval cells under the production sandbox (Leo, R-7 = A).** `build_eval_orch_config` stops
overriding `sandbox` with `SandboxConfig(enabled=False)`, so a corpus cell inherits the base config's
sandbox (same backend and grants as production); `build_shadow_orch_config`'s own override becomes
redundant and goes. Setup steps that write outside the worktree get the grants they need, measured by
the task; each cell's metrics carry the sandbox backend it ran under, and the task records that corpus
results from before the change are not comparable and the next campaign needs a fresh baseline.
The separate eval clone (the residual's option C) is not part of this plan; the shared-`.git` path stays
a known residual (D14).

**ω — the invariant and the gate.** The four-site lockstep edit for `escalation-filing-one-door`, the
guard (behavioural, construction and resubmit halves), the boundary suite's integration half, the
deletion of `has_open_l1` and of the public `submit_or_dedupe`/`submit_or_dedupe_off_loop`, and the
ARCHITECTURE.md escalation paragraph.

## Resolved design decisions

- **D1 — One filer type, three context-bound entries; offload is not a policy field (OQ-1, ruled by
  Leo; a clarification of R8's single `file(record, policy)`).** The
  offload axis is decided by the caller's execution context, which the type system already knows: a
  coroutine awaits `file()`, which awaits the lookups on the loop and hops the blocking body to a
  thread; sync code running on the loop thread calls `file_soon()`, which schedules `file()` on the
  running loop and returns; code not on a loop thread calls `file_blocking()`, which refuses with
  `FilingOnLoopThread` when a loop is running on the calling thread. A policy field could be set wrong;
  an entry point cannot. This is how fk-1bf4bfa8be04's "async callers cannot reach the blocking body"
  becomes mechanical, and it deletes the comments that state the rule (Comments stance). The other three
  axes are policy (dedup, level) or record identity (subject), D2–D3. There is no fourth, inline entry
  (OQ-9, lead): a synchronous on-loop caller whose correctness needs the write to land before it yields
  (the steward's level-1 give-up, κ4; S6's halt-owner filing, ε3) becomes async and awaits `file()`. The
  await comes after the hold (S6) and before the publish (S9 ε publishes at its named yield points, and
  a filing await is not one of them), so both orders are kept; boundary rows 23 and 24 reproduce S9's
  row 15c and S6's row 13b.
  Accepted consequence (Leo, OQ-1): three synchronous on-loop alarm callers that write inline today
  become fire-and-forget through `file_soon()` and can lose a record if the process dies between the
  call and the worker thread's write: the reconciliation harness's alarm method
  `ReconciliationHarness._escalate` (10 call sites, measured at `a85cb1ad78`; ι1), the scheduler's
  work-product probe alarm (`work_product.py::_failure_escalation` through the scheduler's `escalate`
  callable; ζ1), and the merge-gate unmeasured-storm reporter `GateUnmeasuredReporter.settle` that task
  9136 adds (ε1). A clean stop loses nothing: the harness and the recon harness drain their filers on
  shutdown (`EscalationFiler.drain`, η2 and ι1), and `file_soon` logs every failed filing at ERROR.
- **D2 — Identity is the record's typed subject, backed by one registry (OQ-1, Leo).** `EscalationDraft.subject`
  is `TaskSubject(task_id)` or `SentinelSubject(sentinel, qualifier)`. `Sentinel` (a `StrEnum` in
  `escalation/models.py`, R8's seam: "one registry in `escalation/models.py`") lists every system
  identity, values byte-identical to today's literals (open records are matched by `task_id`; a new
  spelling would hide them and refile — the S3 α precedent for `__merge_coalesce_live_original__`).
  Prefix forms (`<sentinel><qualifier>`, e.g. `__merge_coalesce_live_original__<request_id>`) are a
  qualifier. The registry holds the non-dunder system ids too, as prefix members: `review-<id>`
  (`review_checkpoint.py`), `orchestrator-startup-<step>` (`startup_steps.py`), `main-sweep-<hex>`,
  `graph:<group_id>` (`index_drift_detector.py`), `inv12-finding:<key>` (`scripts/exception_owner_sweep.py`),
  `validation-cadence-<lane>` (an external producer through the CLI), `recon-<run>` (recon `_escalate`),
  `pipeline-landing-tripwire-<sha>` (`skew_tripwire.py`), `store-unavailable-startup` (S8 C3b) and the
  hyphenated anchors (`mcp-markup-residue`, `markup-tripwire`, `dirty-project-root-startup`, …); α
  measures the full set at dispatch. `TaskSubject` refuses a dunder-shaped id, a registered value or an
  id carrying a registered prefix at construction (INV-2 message naming the value and the `Sentinel`
  member to use), so the registry is enforced, not advisory. `subject_for(task_id)` is the one parser for
  ids that arrive as strings (the ladder MCP tools, `promote_to_l2`, the submit CLI): a registered value
  or prefix form becomes a `SentinelSubject`, any other id a `TaskSubject`. Lookups (status, claimant)
  and terminal auto-resolve apply to `TaskSubject` only.
- **D3 — Dedup is a sum type, level a policy field, born-at-L2 derived [R8].** `Dedup = NoDedup |
  OpenAtLevel(level=None, category=None, agent_role=None, same_fingerprint=False, any_level=False) |
  Fingerprint(DedupeConfig) | RootCause() | RefreshOpen(...)` — R8's `none | open_at_level |
  fingerprint | root_cause`, plus the refresh member (D13). `FilingPolicy.level ∈ {0, 1}` is the entry
  level for a non-born-at-L2 severity; a born-at-L2 severity that survives the role gate is stamped
  level 2. A policy's dedup level defaults to the level the filer stamps: `OpenAtLevel()` on a record
  stamped level 2 matches open level-2 records, so a level-2 filing is never deduped against level 1
  (the latent bug in the three `merge_lane/shadow.py` filings, ε2b). `OpenAtLevel(any_level=True)`
  matches a pending record at any level under the subject (today's folded-anchor rule, C9). So
  "born-at-L2 ⇔ severity ∈ `BORN_AT_L2_SEVERITIES` ⇔ a sentinel role" (fk-0d31f1114911) holds by
  construction at the one door for filings. The measured exception is promotion (OQ-7, lead): a watcher's
  `promote_to_l2` creates an L2 with the cluster's computed severity (often `blocking`) under the role
  `escalation-watcher-auto`, which is not a sentinel role, so it is level 2 by promotion, not by
  severity. `FilingPolicy.promoted_l2(root_cause)` sets `promotion=True`: level 2 is stamped
  explicitly, the role downgrade and the claimant lookup are skipped (promoted L2s are unstamped by
  design today, `pins.py` Link 3), terminal auto-resolve does not run, and the dedup must be
  `RootCause` (any other dedup with `promotion=True` raises `ValueError` naming both, INV-2). Measured
  otherwise: every literal `level=2` construction today uses `critical`/`urgent` and a
  sentinel-prefixed or computed sentinel role, except `workflow.py::_file_architect_merge_failed`, whose
  role `orchestrator-architect-desync-merge` is a sentinel role δ passes through (C7); the 20
  constructions with the bare role `'orchestrator'` are all `blocking`/`info`, so no in-process filing
  changes level under the uniform rule.
- **D4 — The draft is a frozen projection of `Escalation`, and the door is the only constructor of
  `Escalation` (OQ-2, lead).** R8's static form ("greps `Escalation(` constructions outside filing.py") needs
  callers that never construct the record. `EscalationDraft` (frozen, keyword-only) holds the 22 fields
  production filers set today (measured at `a85cb1ad78`: the union of keywords over the 122 constructions:
  `task_id`→`subject`, `agent_role`, `severity`, `category`, `summary`, `detail`, `suggested_action`,
  `filing_claimant_run_id`, `workflow_state`, `worktree`, `dedupe_fingerprint`, `root_cause`,
  `evidence`, `project_id`, `options`, `train_state`, `note_to_file`, `members`, `index_health`,
  `delivered_check_owner`, `citation_sha`, and `review_due` (the human-gate payload task 6390 added,
  `e87228b063`, filed only through `deterministic_runner.py::build_gate_escalation_fields`); plus
  `id_namespace` for the 5 deliberate `make_id` key
  divergences; `id` and `level` are the filer's, and the one site that passes `timestamp` gets the
  filer's stamp). The filer constructs `Escalation` once from the draft. A parity test pins each draft
  field to the `Escalation` field of the same name, type and default under one projection rule: a
  draft `tuple[X, ...]` with default `()` projects to the record's `list[X]` with `default_factory=list`
  (`members`, `options`, `evidence`), and every other field projects unchanged (INV-5's drift-test
  house pattern for a deliberate projection; the two types have different purposes, heuristic 6: a
  frozen filer input vs the mutable persisted record the queue's mutators edit). The draft is frozen
  deeply: `__post_init__` stores `train_state`, `index_health`, `delivered_check_owner`, `review_due` and
  each `EvidenceEntry` (all `TypedDict`s) as `MappingProxyType` copies, so a caller keeping a reference
  cannot change a filed draft. The filer never mutates a record in place: the downgrade, the level and
  the claimant are computed into the one construction (heuristic 8; the closure mutates `esc` in place
  today).
- **D5 — `EscalationQueue.submit` stays named `submit`; after ω, filing is its only production caller
  (OQ-5, lead).** R8: "becomes its private write step". 125 test files carry 754 `.submit(`/`.submit_resolved(`
  calls (any receiver; most seed reader fixtures through the queue; measured); renaming would be a
  125-file test codemod with no production benefit and the lock-starvation profile R57 warns about.
  Today the statement is false in two non-construction places: `fused-memory/scripts/backfill_recon_escalations.py::apply_plan`
  and `orchestrator/src/orchestrator/startup_steps.py::_file_or_refresh_l1` mutate a loaded record and
  call `queue.submit` on it; ι2 routes the first through a public queue mutator built on S3 C4's
  `_mutate_pending`, and κ1 moves the second onto `RefreshOpen` (D13). After ω, privacy is enforced by
  three checks: no production module constructs an `Escalation` (row 21); every production caller of
  `submit`/`submit_resolved` is `escalation.filing` (row 20, R8's fixture form); and no code outside
  `escalation/` passes `submit`/`submit_resolved` a record built by `Escalation.from_dict`,
  `Escalation.from_json` or `dataclasses.replace` (row 22). S3 β's `_mutate_pending`/`_mutate_terminal`
  remain the only rewriters of an existing record (S3 C4); the filer creates, and refreshes only
  through them.
- **D6 — Dedup and write are one critical section per subject.** Today the on-loop, await-free sequence
  "check open → mint id → submit" is atomic only against other coroutines. Off the loop it would race,
  so the filer holds `subject_lock(queue_dir, task_id)` (a flock on a sidecar beside the per-id locks
  `queue.py::escalation_id_lock` already uses) from the dedup check to the write. `RootCause` takes its
  lock on the canonical root cause instead (`root_cause_lock(queue_dir, canonical)`), because two
  promotions of one root cause can arrive under different subjects. It also closes the
  cross-process duplicate window `dedupe.py`'s contract tolerates today. `Fingerprint` keeps
  `resolve_dedupe_parent` and its TOCTOU guard inside the section; `dedupe.py::attach_or_submit` splits
  into `attach_to_parent(queue, parent_id, esc) -> bool` (the fold, a mutation through S3 C4 that
  returns `False` when the parent vanished) and the filer's own fallback write, so every write of a new
  record is the filer's (row 20 sees `escalation.filing`, never `escalation.dedupe`).
- **D7 — The lookups are awaited on the loop, before the hop.** `FilingLookups(task_status,
  task_claimant)` are async callables (today's `create_server` parameters). `file()` awaits them first,
  only when the subject is a task and the draft has no claimant, then passes the results into the
  thread. `file_blocking()` takes no lookups: a filer built with lookups refuses `file_blocking` with
  `FilingMisuse` (construction decides, not the call). In-process filers build lookup-less filers, so
  terminal auto-resolve and the claimant lookup stay disabled for them exactly as today: in-process
  orchestrator drafts carry `filing_claimant_run_id` themselves (54 of 93 orchestrator constructions do
  at `a85cb1ad78`, and `file_or_refresh`'s two callers pass it; the converting leaves keep each one), and
  fused-memory filers stay unstamped (0 of 18
  today), so `escalation/src/escalation/pins.py`'s Link 4 reads the same records as before. The MCP
  server's filer is the one built with lookups (from S3 ζ's `LadderLookups.task_status`/`task_claimant`
  once ζ lands).
- **D8 — Outcomes are typed; the wire shape is a projection.** `FilingOutcome = Filed | Folded |
  Suppressed | Refreshed | Contained | AutoResolved | NotedToFile | Unpersisted`. `tool_response()` renders
  today's MCP dicts byte-for-byte (boundary row 1 pins them). A write failure raises `FilingFailed`
  (naming subject, category, cause; INV-2) from `file`/`file_blocking`; `file_soon` never raises and
  logs an ERROR with the draft's identity (INV-11: the drop is visible), and
  `EscalationFiler.drain(timeout)` awaits scheduled filings (tests, and the harness's shutdown in η2).
  `async EscalationFiler.resolve_filed(outcome, resolution, *, resolution_class)` resolves a record this
  filer just wrote, delegating to `EscalationQueue.resolve` off the loop (it hops like `file()`, INV-8)
  with `resolved_by` naming the filer's caller; it is how `file_and_block` closes a moot filing (D10).
- **D9 — Per-area adapters only choose a policy (R8).** The lane port, the scheduler port,
  `TaskWorkflow._file_escalation`, the recon registry and `file_folded_escalation` hold no gate logic:
  each fills the fields its area owns and names a `FilingPolicy`. Named constructors keep the common
  choices one call: `FilingPolicy.open_l1(…)`, `FilingPolicy.born_at_l2()` (with `RootCause`),
  `FilingPolicy.promoted_l2(root_cause)` (D3), `FilingPolicy.folded(config)`,
  `FilingPolicy.refresh_open(…)` (D13), `FilingPolicy.terminal_is_the_subject()` (today's
  `terminal_state_is_the_bug=True`).
- **D10 — The status write gets one typed outcome (fk-0ee68c0948ab).** `write_status` catches
  `SetTaskStatusRejected` and other exceptions separately and returns
  `Written | Rejected(err: SetTaskStatusRejected) | Failed(exc)`, logging both failures at WARNING with
  `task_id`, target status, the error code or exception type, and the site (INV-2). A caller matches on
  the outcome; none of the 24 arms keeps a bare `except Exception`. `file_and_block` re-reads the task's
  status before the block write and skips it when the task left the expected status, then resolves the
  just-filed record as moot by awaiting `EscalationFiler.resolve_filed(outcome, …, resolution_class='benign')`
  when the task is terminal (task 3887).
- **D11 — Recon categories are one registry (fk-b7b0be25c6b4); it is the one declaration of a recon
  alarm.** `ReconEscalation` members carry `category: EscalationCategory`, `severity`, `folds: bool`
  and an identity factory (finding dict or a stable `_*_FINDING` constant, or none). The recon
  `DedupeConfig`'s category set (today's field `infra_dedupe_categories`, which S2's `DedupeConfig`
  rename renames) is the set of members with `folds=True`; the severity is the member's;
  `_finding_recently_resolved` reads the same fold set. `_escalate(member, run_id, summary, *,
  finding=None)`; a `str` category no longer type-checks. S12 consumes it (seam table: "recon category
  registry — S12 consumes S1's FilingPolicy"): an S12 `RunStreak` names a `ReconEscalation` member and
  declares no severity, level or dedupe mode of its own (heuristic 11: one declaration per fact).
- **D12 — S7's sink is absorbed by the port, not adapted.** `GateUnmeasuredReporter` takes the
  widened `EscalationPort`; `settle` (S7 C2: synchronous, on the loop) calls `file_soon(draft,
  FilingPolicy.open_l1(category=MERGE_GATE_UNMEASURED_STORM))` on the per-gate sentinel. The "no await
  between the counter check, `make_id`, `has_open_l1` and `submit`" clause of S7 C2 is replaced by D6's
  critical section: dedup is atomic in the filer, not by loop occupancy. `StormEscalationSink` is
  deleted (lead addendum: "S1 absorbs it when S1 widens `EscalationPort`").
- **D13 — Refreshing an open record is a dedup member, not a second write path.** Three filers keep
  one pending record per sentinel and refresh it in place: `startup_steps.py::_file_or_refresh_l1`, the
  dirty-tree filer (`harness.py::Harness._file_dirty_tree_escalation`, in `infra_filers.py` after S8
  ζ1b) and S8 C3b's `STORE_UNAVAILABLE_SENTINEL` (S8 β 9218, pending). Task 8926 (done 2026-11-12,
  `ed4ceb9480`) gave the first two one helper, `escalation/src/escalation/file_or_refresh.py::file_or_refresh`
  over a frozen `RefreshableAlarm` spec, and 9218 will file through it too. Its rules become this
  member's: a refresh rewrites `summary`, `detail` and `timestamp` and increments `dedupe_count`; it
  keeps the id, every other field and `filing_claimant_run_id` (the incarnation that filed the record;
  a refresh is not a filing, task 3550); alarms at different levels on one subject never merge; a
  record no longer pending is never refreshed; a refreshable alarm below level 1 is refused (an L0 whose
  filing incarnation is dead stops pinning, `escalation.pins`). `RefreshOpen(level, refresh)` matches an
  open record on the subject (at `level`, or the stamped level when `None`) and, inside D6's subject
  lock, rewrites only the named `refresh` fields through S3 C4's `_mutate_pending`, returning
  `Refreshed(id, level)`; with no open record it files a new one. The subject lock closes the race the
  helper's own docstring records (a record resolved between its read and its write comes back as
  pending; two concurrent refreshes can lose a `dedupe_count` increment), and the filer never
  resubmits a record (D5): the helper does `dataclasses.replace` + `queue.submit` today. κ1 and η1 move
  the two landed callers onto `RefreshOpen`, η2 moves 9218's, and η2 then deletes `file_or_refresh.py`
  and its test.
- **D14 — Eval-lane containment covers every level-1 and level-2 filing; the door does not contain an
  in-process level-0 record (OQ-8, ruled by Leo, with his hermeticity rule).** Today only the MCP door
  contains, at every level (task 3096). At main an eval cell's workflow has no escalation queue and no
  steward (`evals/runner.py::_cell_workflow` passes neither to `workflow.py::build_workflow`, and every
  workflow filing is guarded by the queue's presence), so an eval cell files nothing in process: keeping
  level 0 out of containment is a backstop for the future, not a protection of current behaviour.
  `FilingPolicy.eval_lane: EvalLaneScope` is `ABOVE_L0` by default and `ALL_LEVELS` for the ladder MCP
  tools' policies (today's door scope), so no eval-lane record can reach a human at level 1 or 2 ("never
  a production signal", `shared/src/shared/eval_lane.py`). Should an eval level-0 record ever reach a
  production queue, no automated filer turns it into a task: the info-level-0 router excludes eval-lane
  records (`escalation_router.py::router_owns`, κ1 keeps the exclusion) and the orphan reaper dismisses
  them (`Harness._reap_orphan_l0_escalations`, in `escalation_record_sweeps.py` after S8 ζ2; η1 keeps the
  read); the startup sweep only dismisses level-0 records. Measured 2026-11-12 over all 11 project
  queues: 0 eval-lane records in the last 30 days of about 8,700; the 109 older ones are reify's, from
  March and April 2026. Containment is not hermeticity: a contained record is still written into the
  production queue directory and counted by `dashboard/src/dashboard/data/escalation_analytics.py`,
  which has no eval filter. Leo's rule ("the filed level zeros from evals [must not] end up leaking into
  the task store via any automated filer, or indeed leaking out of the eval in any way") is therefore
  enforced where an eval cell is built, not at the door: λ1 and λ2 keep eval records and every other
  eval product out of production stores, μ puts corpus cells' agents under the production sandbox, and
  the invariant `eval-writes-no-production-state` (λ2) states it. Out of scope by the lead's reading,
  shown to Leo: the daily memory-eval runner's `eval_regression` escalations
  (`fused-memory/scripts/memory_eval_runner.py`) are meant to reach a human, so the rule does not apply
  to them; the eval worktree's shared `.git` (refs, the single `refs/stash`, hooks, objects) stays a
  known residual, since the separate eval clone was not filed (R-7). μ narrows it for agents: under the
  production write set (`agents/write_set.py::compute_write_set`, read at `4a87282ee7`) an eval agent
  can write the main `.git`'s `objects/`, `refs/heads/task/` and its own `worktrees/<name>/`, not
  `refs/stash`, hooks or config; μ's host run confirms it.
- **D15 — S6's lane filings take the widened port (C10).** S6's filed contracts build three lane
  filings on the raw queue: `HaltState`'s `submit_halt_owner` (C7, plain `EscalationQueue.submit`, with
  a caller-minted id), `MainHealthAutoHeal`'s bound `submit_or_dedupe` (C3) and
  `LifecycleRefusalResponder`'s queue (C2). ε3 gives all three the `EscalationPort`. Because the filer
  mints ids (D4), a halt owner holds under a provisional holder id first (S6's invariant "a holder is
  in place before its escalation is filed" is kept) and re-keys on the filed id, which S6 C7 already
  does "if id differs". S6's own value types `Filed`/`FilingFailed` collide with C1's names, and S6's
  `file_holder_escalation` returns a value where C1 raises; ε3 renames them `HolderFiled`/
  `HolderFilingFailed` (named in the lead's note on 9176; the lead may instead collapse them into C1's
  outcome).

## Contracts

S1 owns C1–C11 (C10 added by the critic fold, C11 by the ruling fold; C1–C10 keep their numbers and
meaning). A dependent stream holds a leaf on `S1/C<n>`.

```python
# C1 — escalation/src/escalation/filing.py (α)
@dataclass(frozen=True)
class TaskSubject:
    task_id: str                                  # refuses a dunder-shaped id, a Sentinel value or a Sentinel prefix form (ValueError, INV-2)
@dataclass(frozen=True)
class SentinelSubject:
    sentinel: Sentinel                            # C3
    qualifier: str | None = None                  # prefix forms: task_id = f'{sentinel.value}{qualifier}'
Subject = TaskSubject | SentinelSubject           # .task_id renders the persisted string
def subject_for(task_id: str) -> Subject: ...     # string ids at the boundaries (MCP tools, promote_to_l2, CLI); D2

@dataclass(frozen=True, kw_only=True)
class EscalationDraft:
    subject: Subject
    agent_role: str
    severity: str                                 # S2 retypes to its Severity StrEnum
    category: EscalationCategory                  # escalation.categories; checked in __post_init__
    summary: str                                  # non-empty; the filer cuts to the queue's limit, never the caller
    detail: str = ''
    suggested_action: str = ''
    worktree: str | None = None
    workflow_state: str | None = None
    filing_claimant_run_id: str | None = None
    dedupe_fingerprint: str | None = None         # still written by this plan (lead, 2026-11-13); S2 γ3 deletes it after ω, S2 γ2 moves its readers first (§Rulings)
    root_cause: str = ''
    options: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    evidence: tuple[EvidenceEntry, ...] = ()
    project_id: str | None = None
    train_state: TrainState | None = None
    index_health: IndexHealthState | None = None
    delivered_check_owner: DeliveredCheckOwner | None = None
    citation_sha: str | None = None
    review_due: ReviewDueState | None = None      # task 6390's human-gate payload (models.py::ReviewDueState)
    note_to_file: bool = False
    id_namespace: str | None = None               # make_id key when it deliberately differs from task_id
    # __post_init__: TypedDict-valued fields and each EvidenceEntry stored as MappingProxyType copies (D4)
    # S2 ε adds filed_via: CallerChannel = CallerChannel.IN_PROCESS (the MCP tools set the parsed channel),
    # so every record the door writes carries the channel stamp; S2 γ adds its four record facts

@dataclass(frozen=True) class NoDedup: ...
@dataclass(frozen=True)
class OpenAtLevel:                                # skip when an open record matches
    level: Literal[0, 1, 2] | None = None         # None = the level the filer stamps (D3)
    category: EscalationCategory | None = None
    agent_role: str | None = None
    same_fingerprint: bool = False                # match draft.dedupe_fingerprint
    any_level: bool = False                       # match a pending record at any level; refuses with level set
@dataclass(frozen=True)
class Fingerprint:                                # fold into a pending parent (dedupe.py mechanics)
    config: DedupeConfig
@dataclass(frozen=True) class RootCause: ...      # one pending L2 per canonical draft.root_cause
RefreshField = Literal['summary', 'detail', 'timestamp', 'dedupe_count']   # never the claimant (D13, task 8926)
@dataclass(frozen=True)
class RefreshOpen:                                # refresh the open record in place (D13), else file
    level: Literal[0, 1, 2] | None = None
    refresh: frozenset[RefreshField] = frozenset({'summary', 'detail', 'timestamp', 'dedupe_count'})
Dedup = NoDedup | OpenAtLevel | Fingerprint | RootCause | RefreshOpen

class TerminalTask(StrEnum):
    AUTO_RESOLVE = 'auto_resolve'                 # today's gate 4 (only when a status lookup is wired)
    FILE = 'file'                                 # today's terminal_state_is_the_bug / review_suggestions bypass

class EvalLaneScope(StrEnum):                     # D14 (OQ-8, Leo)
    ABOVE_L0 = 'above_l0'                         # default: contain level 1 and 2; the door does not contain an in-process L0
    ALL_LEVELS = 'all_levels'                     # the ladder MCP tools' policies (today's door scope)

@dataclass(frozen=True)
class FilingPolicy:
    dedup: Dedup
    level: Literal[0, 1] = 1                      # entry level; a surviving born-at-L2 severity is stamped 2
    terminal_task: TerminalTask = TerminalTask.AUTO_RESOLVE
    eval_lane: EvalLaneScope = EvalLaneScope.ABOVE_L0
    promotion: bool = False                       # D3 (OQ-7): stamp level 2, no downgrade, no claimant lookup;
                                                  # requires dedup=RootCause() (ValueError otherwise, INV-2)
    # named constructors: open_l1(...), born_at_l2(), promoted_l2(root_cause), folded(config),
    # refresh_open(...), none(level), terminal_is_the_subject(...)
    # promoted_l2() sets promotion=True, dedup=RootCause() and terminal_task=TerminalTask.FILE, so steps 4
    # and 7 never look up or auto-resolve a promotion's task (today's create path bypasses that gate; D3)

@dataclass(frozen=True) class Filed:        id: str; level: int; status: str        # observed post-write status
@dataclass(frozen=True) class Folded:       parent_id: str; child_id: str; level: int
@dataclass(frozen=True) class Suppressed:   open_id: str; level: int                # OpenAtLevel / RootCause matched
@dataclass(frozen=True) class Refreshed:    id: str; level: int                     # RefreshOpen matched
@dataclass(frozen=True) class Contained:    id: str; reason: str                    # eval-lane, filed resolved close_only
@dataclass(frozen=True) class AutoResolved: id: str; task_status: str
@dataclass(frozen=True) class NotedToFile:  id: str
@dataclass(frozen=True) class Unpersisted:  id: str; persist_check: str; level: int
FilingOutcome = Filed | Folded | Suppressed | Refreshed | Contained | AutoResolved | NotedToFile | Unpersisted
def tool_response(outcome: FilingOutcome) -> dict[str, Any]: ...   # today's MCP shapes, byte-for-byte

@dataclass(frozen=True)
class FilingLookups:
    task_status: Callable[[str], Awaitable[str | None]] | None = None
    task_claimant: Callable[[str], Awaitable[str | None]] | None = None

class FilingFailed(RuntimeError): subject: str; category: str      # the write raised; cause chained
class FilingOnLoopThread(RuntimeError): ...                         # file_blocking on a loop thread
class FilingMisuse(TypeError): ...                                  # file_blocking on a filer with lookups

class EscalationFiler:
    def __init__(self, queue: EscalationQueue, *, lookups: FilingLookups | None = None) -> None: ...
    @classmethod
    def for_queue_dir(cls, queue_dir: Path) -> EscalationFiler: ...  # the fused-memory per-project shape
    async def file(self, draft: EscalationDraft, policy: FilingPolicy) -> FilingOutcome: ...
    def file_soon(self, draft: EscalationDraft, policy: FilingPolicy) -> None: ...   # needs a running loop on this thread
    def file_blocking(self, draft: EscalationDraft, policy: FilingPolicy) -> FilingOutcome: ...
    async def drain(self, timeout_secs: float) -> int: ...          # awaits file_soon tasks; returns how many were left
    async def resolve_filed(self, outcome: FilingOutcome, resolution: str, *, resolution_class: str) -> bool: ...
        # resolves the record `outcome` names (Filed/Folded/Refreshed/Unpersisted) via EscalationQueue.resolve,
        # hopping the flock'd resolve off the loop like file() (INV-8); D8
```

**C2 — the door's behaviour (α).** One ordered pipeline, identical for the three entries. Steps 1–3 are
pure (no I/O) and run on the calling thread; step 4 is `file()`'s awaited lookups; steps 5–9 run in one
worker thread under `subject_lock(queue_dir, subject.task_id)`, or `root_cause_lock(queue_dir, canonical)`
for `RootCause` (D6):

1. Draft validated at construction (category registered, summary non-empty, subject legal).
2. Severity and level. Skipped for a `note_to_file` draft (today's note-to-file path runs no
   downgrade). Role downgrade: severity ∈ `BORN_AT_L2_SEVERITIES`, not `policy.promotion`, and not
   `models.is_harness_sentinel_role(role)` → severity `blocking`, WARNING, and today's downgrade marker
   (the summary suffix `[downgraded:<severity>]`; S2's filer-declared-facts leaf, after β, writes
   `original_severity` here instead, S2-3 (a)). Level: 2 when `policy.promotion` or the (post-downgrade)
   severity ∈ `BORN_AT_L2_SEVERITIES`, else `policy.level`.
3. Eval-lane decision: `shared.eval_lane.eval_lane_provenance(task_id, worktree)` not `None`, and the
   step-2 level is ≥ 1 or `policy.eval_lane is ALL_LEVELS` (OQ-8) → the draft is contained (step 5) and
   no lookup runs. So no argument can mint an eval-lane record pending at level 1 or 2.
4. `file()` only, not contained: claimant lookup when `TaskSubject`, lookup wired, draft claimant `None`
   and not `policy.promotion` (fail-open: a raising lookup leaves it `None` and logs WARNING, never drops
   the filing); status lookup when `TaskSubject`, lookup wired and `policy.terminal_task is
   AUTO_RESOLVE` (same fail-open).
5. Contained → write resolved `close_only` via `submit_resolved` (class `note_to_file` or `benign` as
   today, `resolved_by='escalation-eval-lane-containment'`) → `Contained`. The contained record is
   written from the draft as filed (caller severity, no marker, `policy.level`); step 2's level is used
   only for step 3's decision.
6. `note_to_file` → the resolved note-to-file write (today's `note_to_file.py::file_note_to_file`) →
   `NotedToFile`.
7. Terminal: status from step 4 ∈ {done, cancelled} → `submit_resolved` benign, with today's
   `resolved_by='escalation-mcp-pre-submit-check'` unchanged (resolver-tier readers key on it) →
   `AutoResolved`.
8. Dedup per `policy.dedup`: `OpenAtLevel` → `queue.has_open(...)` at the policy's level, the stamped
   level when `None`, or any level when `any_level` → `Suppressed`; `RootCause` →
   `find_pending_l2_by_root_cause` match → `Suppressed`; `Fingerprint` → applies only at level < 2 (a
   level-2 record is never folded into a lower parent — today's L2 bypass), folds via
   `attach_to_parent` → `Folded`, and falls through to step 9 when no parent remains; `RefreshOpen` →
   an open match is rewritten through S3 C4 `_mutate_pending` → `Refreshed` (D13).
9. Mint the id (`make_id(draft.id_namespace or task_id)`), construct the one `Escalation`, `submit`,
   return `Filed`/`Unpersisted` from `observed_submit_response`.

Invariants: every record written to a queue directory by production code passed steps 1–9 (ω); the
filer never mutates an `Escalation` after construction and rewrites an existing record only through S3
C4; `file_soon` never raises; `file()` holds the loop thread only for its awaits (INV-8). The notify
callback fires from the worker thread and is delivered on the loop by S3 C4 `deliver_on`. Once S2 ε
lands, step 9 writes the draft's `filed_via` on every record (cross-stream identity ruling).

Who decides (invariant `one-decision-partition-per-area`, INV-23, minted by S17 ν): every step above
is decided by code from facts code holds, in this one pipeline; no step asks an LLM. Judgement starts
only after a record is written, at the ladder's existing consumers (the steward at level 0, the
auto-watcher at level 1, a human at level 2), each with its path to a human. A filer's choice of
`FilingPolicy` is code too, made once in the adapter that owns the area (D9).

```python
# C3 — escalation/src/escalation/models.py (α) and queue.py (α)
class Sentinel(StrEnum):            # values byte-identical to today's literals; α lands all of them
    FM_READ_STORM = '__fm_read_storm__'
    BACKFILL_OVERSTAY_RATE = '__backfill_overstay_rate__'
    MERGE_COALESCE_LIVE_ORIGINAL = '__merge_coalesce_live_original__'   # prefix form (qualifier = request id)
    REVIEW = 'review-'                                                  # prefix form (qualifier = task id); D2
    RECON_RUN = 'recon-'                                                # prefix form (qualifier = run id[:8])
    ...                             # every system task id measured at a85cb1ad78: 44 dunder literals by the ruling fold's rule, 39 by the author's (the
                                    # non-escalation ones excluded by α's measurement), the hyphenated anchors
                                    # ('mcp-markup-residue', 'markup-tripwire', 'dirty-project-root-startup', …)
                                    # and the non-dunder prefix forms D2 lists
def is_harness_sentinel_role(agent_role: str) -> bool: ...  # moved from server.py; HARNESS_SENTINEL_ROLE_PREFIXES stays here

class EscalationQueue:
    def has_open(self, task_id: str, *, level: int | None, category: str | None = None,
                 agent_role: str | None = None, dedupe_fingerprint: str | None = None) -> bool: ...
        # level None = any level (OpenAtLevel.any_level); the filer resolves OpenAtLevel.level=None to the stamped level
    def has_open_l1(self, task_id: str, *, category=None, dedupe_fingerprint=None) -> bool: ...  # forwards; deleted in ω
# escalation/src/escalation/dedupe.py (α): attach_or_submit splits (D6)
def attach_to_parent(queue: EscalationQueue, parent_id: str, esc: Escalation) -> bool: ...  # through S3 C4; False = parent gone
```

```python
# C4 — orchestrator/src/orchestrator/merge_lane/ports.py (ε1)
class EscalationPort(Protocol):
    async def file(self, record: EscalationDraft, policy: FilingPolicy) -> FilingOutcome | None: ...
    def file_soon(self, record: EscalationDraft, policy: FilingPolicy) -> None: ...
# ProductionEscalations(filer: EscalationFiler); DiscardingEscalations returns None / does nothing.
# EscalationRecord is deleted by ε3 once S6's halt.py stops taking it (EscalationDraft replaces it). R8's
# "file(record, dedupe)": the second argument is the whole FilingPolicy because the level is chosen there
# too (OQ-10, lead; amends R8's spelling). The lane keeps the raw queue only for resolve and reads (task 8665, amended).
```

```python
# C5 — orchestrator/src/orchestrator/scheduler_escalations.py (ζ2)
class SchedulerEscalations(Protocol):            # one SchedulerCallbacks field, `escalations` (S8 owns SchedulerCallbacks)
    def file(self, record: EscalationDraft, policy: FilingPolicy) -> None: ...          # file_soon semantics
    async def file_and_block(self, task_id: str, record: EscalationDraft, policy: FilingPolicy,
                             *, expected_status: str) -> FileAndBlockOutcome: ...      # C6
    async def resolve_open(self, subject: Subject, *, category: EscalationCategory, resolution: str) -> int: ...
        # hops the flock'd resolve off the loop like file() (INV-8)
# Replaces on_external_dep_block, on_delivered_check_block, on_starvation_warn, on_starvation_resolve,
# on_warm_base_warn, on_warm_base_promote_l2, on_warm_base_resolve. None in bare-Scheduler tests -> a
# DiscardingSchedulerEscalations, never an attribute set after construction.
```

```python
# C6 — orchestrator/src/orchestrator/status_write.py (γ)
@dataclass(frozen=True) class Written: ...
@dataclass(frozen=True) class Rejected:  error: SetTaskStatusRejected       # .row_already_terminal, .error_code
@dataclass(frozen=True) class Failed:    exc: BaseException
StatusWriteOutcome = Written | Rejected | Failed
async def write_status(scheduler: Scheduler, task_id: str, status: str, *, site: str) -> StatusWriteOutcome: ...
@dataclass(frozen=True)
class FileAndBlockOutcome:
    filing: FilingOutcome
    block: StatusWriteOutcome | None          # None = skipped: the task left expected_status before the write
async def file_and_block(filer: EscalationFiler, scheduler: Scheduler, task_id: str, record: EscalationDraft,
                         policy: FilingPolicy, *, expected_status: str, site: str) -> FileAndBlockOutcome: ...
```

C7 — `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._file_escalation(*, severity, category,
summary, detail, suggested_action='', policy, evidence=(), root_cause='', train_state=None,
agent_role='orchestrator') -> FilingOutcome` (δ). Its keyword list grows with the draft fields S2 adds
(S2 θ1 passes `filing_site` and `failure_signature` on the drafts δ builds, and extends this signature
to carry them). It fills `subject=TaskSubject(self.task_id)`,
`worktree`, `workflow_state`, `filing_claimant_run_id`; passes `agent_role` through (the one born-at-L2
workflow filing, `_file_architect_merge_failed`, passes its sentinel role
`orchestrator-architect-desync-merge`, so step 2 keeps it at level 2). Emits
`EventType.escalation_created` with today's data keys plus an `outcome` key for every outcome that
wrote or rewrote a record (`Filed`, `Folded`, `Refreshed`, `Unpersisted`, `Contained`, `AutoResolved`,
`NotedToFile`), so `runs.db` counts every record written; never for `Suppressed`.

C8 — `fused-memory/src/fused_memory/reconciliation/recon_escalations.py` (ι1): `ReconEscalation`
(an `Enum` whose values are frozen `ReconEscalationSpec(category: EscalationCategory, severity: str,
folds: bool, identity: Callable[..., dict | None] | None)`); `RECON_FOLD_CATEGORIES` and
`recon_dedupe_config()` derived from it; `policy_for(member) -> FilingPolicy`. It is the one declaration
of a recon alarm's severity and fold rule (D11): S12's `RunStreak` names a member and adds nothing.

C9 — `fused-memory/src/fused_memory/escalators/folded_escalation.py` (S19/OQ-4 (b)) (θ1a; today
`middleware/_folded_escalation.py`, moved by S19 γ): `file_folded_escalation(project_root, *, anchor:
SentinelSubject, …) -> str | None` (blocking, refuses on a loop thread through C1) and
`async def afile_folded_escalation(...)` with the same parameters; both choose
`FilingPolicy(dedup=OpenAtLevel(any_level=True))`, today's fold: any pending record under the anchor,
whatever its level or category, suppresses a new one (`get_by_task(anchor_task_id, status='pending')`),
so a promoted L2 under the anchor keeps suppressing new L1s. The "every filer's anchor is unique" test
is kept. The content-fingerprint family stays separate (θ2) as today's module docstring requires.

```python
# C10 — S6's lane collaborators take the widened port (ε3); S6 C2, C3, C7 land as written, ε3 applies this after them
class HaltState:                                  # merge_lane/halt.py (S6 θ creates)
    def __init__(self, lanes: tuple[str, ...], *, on_resume: Callable[[], None],
                 escalations: EscalationPort) -> None: ...      # replaces submit_halt_owner: Callable[[EscalationRecord], str]
    async def hold_with_escalation(self, lane: str, reason: str, draft: EscalationDraft,
                                   kind: HaltKind) -> HaltHolder: ...
        # holds FIRST under a provisional holder id (no await before the hold), then awaits
        # escalations.file(draft, FilingPolicy.none(level=1)) and re-keys the holder on the filed id
    async def file_holder_escalation(self, holder: HaltHolder, draft: EscalationDraft
                                     ) -> HolderFiled | HolderFilingFailed: ...
        # S6's Filed/FilingFailed renamed (D15); never raises into the lane, as S6 C7 states
class MainHealthAutoHeal:                         # merge_lane/main_health.py (S6 δ creates)
    def __init__(self, *, registry, halt: HaltState, escalations: EscalationPort, spawn_fix_task, max_attempts: int) -> None: ...
        # files with FilingPolicy.folded(<the main-health DedupeConfig>) in place of a bound submit_or_dedupe
class LifecycleRefusalResponder:                  # merge_lane/item_lifecycle.py (S6 ζ/φ create)
    def __init__(self, *, policy, halt: HaltState, escalations: EscalationPort, event_store) -> None: ...
# Invariants kept from S6: a holder is in place before its escalation is filed; a filing failure releases
# that holder only; a refusal never retires the request. Halt-owner filings use NoDedup (today's plain
# submit), so a minted record is never folded into a parent.
```

```python
# C11 — the eval cell is built hermetic (λ1, λ2, μ; Leo 2026-11-12/13). orchestrator/src/orchestrator/evals/
# profile.py (λ1): EVAL_PROFILE gains 'transcript_archive.root'; resolve_eval_profile_update resolves it under
#   evals/snapshots.py::eval_worktree_root(project_root), so the archive leaves project_root and the parity
#   tripwire (boundary B1) covers the divergence like every other declared one.
class EvalHermeticityError(ValueError): ...      # names the field and the production object it was handed (INV-2)
# runner.py::_cell_workflow (λ1): after build_workflow, refuses with EvalHermeticityError a workflow holding an
#   escalation queue, a steward factory, an event store or a cost store; the escalation filer δ adds is
#   derived from the queue, so refusing the queue refuses it too. "No queue, no filer, nothing written".
# workflow.py (λ2): build_workflow(..., verify_log_root: Path | None = None) -> TaskWorkflow; None keeps today's
#   project_root / 'data' / 'verify-logs'; _cell_workflow passes a root under eval_worktree_root(project_root).
# runner.py::build_eval_orch_config (μ): no 'sandbox' key in its update; a corpus cell inherits
#   base_config.sandbox (backend and grants); the cell's metrics carry the sandbox backend it ran under.
# Invariant kept by all three: an eval run writes no production queue, runs.db row, transcript, verify log or
#   task; the eval profile and the cell factory are the only places an eval cell diverges from production.
```

## Boundary-test sketch

Facing both sides: the filer (producer) and each adapter (consumer). Real queue directories under
`tmp_path`, real `EscalationQueue`, no private patch targets (Tests stance). Rows 1–9b are α's, 10–11
β's, 12–13 γ's, 14 and 14b δ's, 15–16 ε1/ε2a/ε2b's, 15b ε2b's, 17 ζ2's, 18 θ1a's, 19 ι1's, 23 κ4's,
24–25 ε3's, 26–27 λ1's (27 widened by λ2), 28 μ's; ω runs rows 1–25 plus 20–22 (λ1, λ2 and μ are
independent of the door and keep their rows in the eval boundary suite).

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | filer→MCP | `escalate_blocker`/`escalate_info` for each of today's response branches (queued, dedup_skipped, auto-resolved terminal, accepted_unpersisted, contained, note-to-file), the contained branch also with a non-sentinel role filing `critical` | golden responses and records recorded on main before α lands | `tool_response(outcome)` equals the golden dict for every branch; the contained record keeps the caller's severity and level, with no downgrade marker (C2 step 5) |
| 2 | filer | agent role files `critical` | `TaskSubject('5')`, role `implementer` | `Filed(level=policy.level)`, severity `blocking`, the record carries step 2's downgrade marker (today's summary suffix; after S2's filer-declared-facts leaf, `original_severity='critical'`, and that leaf updates this row), WARNING names role and task |
| 3 | filer | sentinel role files `urgent` with `policy.level=1` | role `orchestrator-x` | level 2, severity `urgent`, never folded under a `Fingerprint` policy |
| 4 | filer | eval-lane subject at level 1 and at level 2 (any severity), and any level under `EvalLaneScope.ALL_LEVELS` | `task_id` an eval fixture id | `Contained`; record resolved `close_only`; no notify callback fired; no lookup ran; no other step ran |
| 4b | filer (OQ-8) | eval-lane subject, in-process policy (default `ABOVE_L0`), level 0 | `task_id` an eval fixture id | `Filed(level=0)`, pending: the door does not contain an in-process level-0 record (a backstop for the future; at main an eval cell files nothing in process, row 14b); the router and the orphan reaper keep it from becoming a task (D14) |
| 5 | filer | `file()` on a running loop, queue I/O made slow (a queue dir on a FUSE/sleep wrapper or a 200 ms injected `get_by_task` delay via a subclass, never a patch of a private) | loop heartbeat task ticking every 10 ms | heartbeat never misses > 1 tick during the filing; notify callback ran on the loop thread (S3 C4) |
| 6 | filer | `file_blocking()` called from a coroutine | running loop | raises `FilingOnLoopThread`; nothing written |
| 7 | filer | two concurrent `file()` calls, same sentinel, `OpenAtLevel(1)` | empty queue | exactly one `Filed`, one `Suppressed` (D6); same across two processes (spawned interpreter) |
| 8 | filer | status lookup returns `done` / raises | `TaskSubject`, lookup wired | `AutoResolved` / `Filed` + WARNING (fail-open) |
| 9 | filer | `TaskSubject('__fm_read_storm__')`, `TaskSubject(Sentinel.FM_READ_STORM.value)`, `TaskSubject('review-42')`; `subject_for('__fm_read_storm__')`, `subject_for('42')` | — | the three constructions raise `ValueError` naming the value and the `Sentinel` member; `subject_for` returns `SentinelSubject(FM_READ_STORM)` and `TaskSubject('42')` |
| 9b | filer | `RefreshOpen` twice on one sentinel, then once after the open record is resolved | empty queue | first `Filed`; second `Refreshed(id)` with the same id, `dedupe_count` 1, summary/detail/timestamp rewritten and every other field byte-identical, written through S3 C4 (no `submit` recorded by row 20's recorder); third `Filed` with a new id |
| 10 | escalation pkg | `promote_to_l2` create, markup residue, markup storm, amendment-truncation storm, `submit.py` CLI | pre-S1 golden records | identical persisted fields except `id` sequence; each passed C2 (row 20's recorder) |
| 10b | escalation pkg (OQ-7) | watcher-role (`escalation-watcher-auto`), `blocking`-severity `promote_to_l2` create through `FilingPolicy.promoted_l2(root_cause)`, the tool driven with the watcher's identity header so the row stays green when S2 ο refuses a header-less promotion (S2 ε only warns); then the same root cause again; then two concurrent promotions of one new root cause under different subjects (two threads, and two spawned interpreters); then a promotion whose subject task is `done` | no pending L2 for the root cause | first `Filed(level=2)`, severity `blocking` unchanged, no downgrade marker, no `filing_claimant_run_id`; second `Suppressed(open_id=<first>)` and the tool appends the members to it as today; the concurrent pair yields exactly one `Filed` and one `Suppressed` (D6's `root_cause_lock`); the subject task is `done`, the L2 is still `Filed(level=2)` (no auto-resolve, `terminal_task=FILE`); `FilingPolicy(dedup=NoDedup(), promotion=True)` raises `ValueError` naming both |
| 11 | escalation pkg | `submit.py` with a non-sentinel role and `critical` | CLI argv | refused as today (exit code and message unchanged), now by the shared predicate; task 7300's pinned external-producer call still files one pending L2 |
| 12 | status write | block write rejected with `TerminalExitRejection` after filing | task cancelled between filing and block | `FileAndBlockOutcome(block=Rejected)`, filed record resolved benign, WARNING with error code; no `except Exception` path |
| 13 | status write | task moved `pending → deferred` before the block write | pure-gate task | block skipped (`block=None`), record kept open, INFO names observed status (task 3887) |
| 14 | workflow | each workflow filing path that remains in `workflow.py` at δ's dispatch, driven through its public trigger; `_file_architect_merge_failed` twice | real queue, real `EventStore` | one `escalation_created` row (with `outcome`) per written record, `workflow_state`/`worktree` present on all; the architect desync filing is `Filed(level=2)` with role `orchestrator-architect-desync-merge` and severity `critical`, and its second firing is `Suppressed` |
| 14b | workflow (eval build) | the eval cell's workflow built by `evals/runner.py::_cell_workflow` for a corpus fixture | real factory, `tmp_path` project root | the workflow holds no escalation queue and no filer; driving a workflow filing path writes nothing to any queue directory; a production queue handed to the factory is refused (row 26), which refuses the filer derived from it |
| 15 | lane (consumer) | each lane alarm fired against `RecordingEscalations` | fake port | recorded policy equals the alarm's documented policy (open-L1 per sentinel, open at the stamped level for the four level-2 filings, fingerprint for main-health) |
| 15b | lane (producer) `[M3]` | the same warm/cold shadow divergence twice, then an unparseable shadow twice, over a real filer | empty queue | one pending L2 per sentinel, not one per divergence (today each repeat files another L2, because `has_open_l1` never matches an L2): fewer human pages, a stated behaviour change |
| 16 | lane (producer) | `GateUnmeasuredReporter.settle` storm over a real filer | threshold+1 unmeasured | one L1 on `__merge_gate_unmeasured_storm__:<gate>`; settle returned synchronously; the filing completed after `drain()` |
| 17 | scheduler | each of the 7 former hooks' triggers with a real `SchedulerEscalations` | bare `Scheduler` with the port injected at construction | same records as today's harness methods (golden), no `escalation_queue` attribute write after construction |
| 18 | fused-memory | `add_memory` flagged by the completion-claim gate, on a running loop; each of the 12 `file_folded_escalation` call sites (the twelfth is the ticket janitor's `asyncio.to_thread(file_folded_escalation, …)` call) θ1a measured as on-loop, through its public trigger | real queue under the project | the escalation lands; `file_blocking` was never reached on a loop thread (row 6's refusal would raise); a pending L2 under an anchor suppresses a new L1 under it (C9) |
| 19 | recon | every `ReconEscalation` member filed twice | real recon queue | members with `folds=True` fold, others file twice; severity equals the member's; `_escalate` rejects a `str` at pyright |
| 20 | invariant | every adapter's production path, with the public class attributes `EscalationQueue.submit` and `EscalationQueue.submit_resolved` monkeypatched (they are public, so this is not a private patch target) to record the calling module, which also covers queues built inside `EscalationFiler.for_queue_dir` | all leaves landed | every recorded caller is `escalation.filing` |
| 21 | invariant | AST guard over `*/src`, `scripts`, `*/scripts` (tests excluded) and seeded fixture modules (a direct construction; an `import … as` alias; the class passed as a value, today's `markup_sink.py` `escalation_cls` shape) | — | outside `escalation/filing.py` and `escalation/models.py`, zero Load-context references to `escalation.models.Escalation` (alias-resolved) other than annotations, `isinstance`/`issubclass` second arguments and attribute access (`Escalation.from_json`); each seeded fixture is flagged with its path and line |
| 22 | invariant (OQ-5) | AST guard over the same scope plus seeded fixtures: `queue.submit(Escalation.from_dict({...}))`, `queue.submit(dataclasses.replace(rec, …))`, and a name assigned from either then submitted | — | outside `escalation/`, zero `.submit(`/`.submit_resolved(` calls whose argument is, or is assigned in the same function from, `Escalation.from_dict`, `Escalation.from_json` or `dataclasses.replace`; each seeded fixture is flagged with its path and line |
| 23 | steward (OQ-9) | S9 ε's row 15c: the session for L0 *X* chains a follow-on L0 *Y* and resolves *X*; the task's lifetime budget is spent, so *Y* gives up before any session; the give-up now awaits `file()` | real `TaskSteward`, scripted agent, real queue | *Y*'s L1 is in the queue before *X*'s outcome is published; `StewardReescalatedL1(Y's L1)` is published for *X*'s id; the original L0 is dismissed after the L1 is written |
| 24 | lane (OQ-9) | S6's row 13b: a declared train's advance fails with `wip_overlap` while the port's `file` raises; auto-heal holds `normal` | `normal` held by auto-heal; a second request queued | the train's provisional holder is in place before `file` is awaited; on the raise that holder alone is released, `merge_halt_owner_filing_failed` emitted, nothing re-raised; the auto-heal holder is untouched |
| 25 | lane | a halt owner filed through the port | lane unheld | the holder is placed under a provisional id, re-keyed to `HolderFiled.escalation_id` after the write, and resolving that escalation releases the lane; `MainHealthAutoHeal` and `LifecycleRefusalResponder` record their policies on `RecordingEscalations` |
| 26 | eval factory (λ1) | `_cell_workflow` handed, in turn, a production escalation queue, a steward factory, an event store and a cost store | real factory | each raises `EvalHermeticityError` naming the field; with none, the cell builds as today |
| 27 | eval boundary (λ1, widened by λ2) | one recorded eval cell (the suite's deterministic stub agents, the real factory and workflow) whose run archives a transcript, writes a verify log and takes the steps that file or submit today | `tmp_path` project root holding a production queue directory, `runs.db`, `data/orchestrator/agent-transcripts`, `data/verify-logs` and the eval stub task store | zero-write delta on all five production stores (bytes and directory listings unchanged); every task write the cell attempts reaches the eval scheduler's `_StubMcpSession` and is refused (`EvalStubRefusal`), and every curator post (review suggestions, the main-fix fallback) targets B5's null sentinel URL, so a task-store leak fails the row instead of passing silently; the transcript and the verify log are under `eval_worktree_root(project_root)`; the parity tripwire B1 lists `transcript_archive.root` |
| 28 | eval config (μ) | `build_eval_orch_config` over a base whose sandbox is `enabled=True, backend='landlock'`; and `build_shadow_orch_config` over the same base | — | the corpus config's `sandbox` equals the base's (enabled, backend and grants); the shadow config's equals it too, with no override of its own; a cell's metrics record the backend |

## Design invariants (ω and λ2 each perform a four-site lockstep edit)

Minted by ω: **`escalation-filing-one-door`**. The six wave-3 minting tasks (S1 ω, S1 λ2, S2 ι, S16 ι,
S19 ε and S17 ν) take INV-18 to INV-23 in landing order, S17 ν's `one-decision-partition-per-area` last
as INV-23 (lead, 2026-11-13; it amends R51's numbering clause, after R51's wave-2 chain 9140, 9242 and
9154 takes INV-15 to INV-17). No wave-3 invariant leaf waits on 9154 for a number (lead S2/OQ-10 (b)).
The lead wires the invariant-file chain at decompose and corrects the numbers then if the order changes;
every check anchors on the slug, never on the number.
Sites (`scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`):
`docs/legibility/design-invariants.md` (section, header dating paragraph, `## Fixtures` date list),
`docs/legibility/design-invariants-fixtures.md` (`INV-n-PRD`, `INV-n-CODE`, a dated addendum walk with its
2-row rehearsal table and the cumulative match count), `skills/prd/references/gates.md` (the family row
and the `inv-trigger-shapes` span); `CONTRIBUTING.md` gets nothing (pinned as an absence). Draft text:

> **Rule**: Every escalation record a production process writes into a queue directory is created by
> `escalation.filing`. No module outside `escalation/filing.py` (and `escalation/models.py`'s
> deserialisers) constructs an `Escalation`; filers build an `EscalationDraft` and choose a
> `FilingPolicy`. The ladder rules (eval-lane containment, claimant stamp, note-to-file, born-at-L2 and
> its role downgrade, terminal auto-resolve, dedupe) and the event-loop offload run at that one door.
>
> **Checkable design question(s)**: Does this feature create an escalation record? Does it pass an
> `EscalationDraft` and a `FilingPolicy` to `EscalationFiler`, or does it construct `Escalation` or call
> `EscalationQueue.submit` itself? Does it choose its dedup, level and identity in the policy and the
> draft's subject, or by hand?
>
> **Evidence**: at `6551b75aba` 112 filing functions in 38 modules chose four axes by hand and only the
> MCP door enforced the ladder; at `a85cb1ad78` 122 in 51 (fk-7d1f3a0d12ca, fk-333277fb4bd7, run
> `review-all-dark_factory-20261008`).
>
> **House pattern**: `escalation/filing.py::EscalationFiler`; the guard
> `escalation/tests/test_filing_one_door.py`.
>
> **Family boundary**: INV-5 governs duplicated logic in general; this governs one store's write path.
> INV-8 governs loop occupancy in general; the door's three entries are its house pattern for filing.

Trigger shape for gates.md: *an escalation record created outside `escalation.filing` — an
`Escalation(...)` constructed, or `EscalationQueue.submit` called, by a filer
(`escalation-filing-one-door`)?*

The guard has three halves, each exercised against fixture inputs (INV-10): behavioural (row 20, R8's
fixture form), structural for constructions (row 21, R8's static form) and structural for resubmitted
records (row 22, the known non-construction write shapes, because `submit` stays public, OQ-5).
Exclusions are derived (the two allowed modules for row 21, the `escalation/` package for row 22), not
hand-listed, so INV-12 has nothing to own. A novel write shape that neither static half names is caught
only by row 20's behavioural half when a test drives it; the PRD records that hole rather than claiming
the static form is complete.


Minted by λ2: **`eval-writes-no-production-state`** (Leo's hermeticity rule, 2026-11-12; text proposed
in `digests/discussion/S1.md` §S1-2 and carried here). Same four sites, same numbering rule (one of
INV-18 to INV-22, by landing order). Draft text:

> **Rule**: An eval run writes no production state. Every store an eval cell can write (escalation
> queue, event and cost stores, transcript archive, verify-log archive, task store, memory) is either
> eval-owned (under the project's eval worktree root, or a recording sink) or a null sink; agents in an
> eval cell run under the production sandbox. The shared resources are account quota and, by Leo's
> ruling of 2026-11-13 (the separate eval clone is not built), the live repository's `.git` that an
> eval worktree is linked to.
>
> **Checkable design question(s)**: Does this feature add or move a store, sink, filer, queue or
> archive that an eval cell can reach? Is the eval cell's handle on it set in `evals/profile.py`'s
> `EVAL_PROFILE` or refused by `evals/runner.py::_cell_workflow`, and does a row of the eval boundary
> suite fail when the cell writes to it (B11's zero-write delta for queues, `runs.db`, transcripts,
> verify logs and task writes; B5 for memory; B10 for the agents' MCP servers)?
>
> **Evidence**: task 4757 measured one 53-cell campaign writing 36 memories, 7 escalations and 3 task
> submissions into production; at `07aaa5be80` 172 eval transcripts sat in the production transcript
> archive and 3 fixture directories in `data/verify-logs` (`digests/discussion/S1.md` §S1-2, run
> `review-all-dark_factory-20261008`).
>
> **House pattern**: `evals/profile.py::EVAL_PROFILE` with its parity tripwire, `_cell_workflow`'s
> refusal, and the zero-write row of `orchestrator/tests/test_eval_boundary_suite.py`.
>
> **Family boundary**: `escalation-filing-one-door`'s eval-lane containment is the backstop when an
> eval record reaches a production queue anyway; this invariant keeps it from getting there. The daily
> memory-eval runner's `eval_regression` escalations are meant for a human and are not an eval cell's
> output.

Trigger shape for gates.md: *a store, sink, filer, queue or archive an eval cell can reach, not
declared in `EVAL_PROFILE` or refused by the eval cell factory (`eval-writes-no-production-state`)?*
## Cross-PRD relationship

Edge notation: "X → Y" means X depends on Y (Y lands first). Wave-2 tasks are named by id; wave-3
sibling leaves (S2, S12, S17, S19) by stream and label, wired at decompose. An edge that rests only on a
shared test file is not wired, inside this plan or across plans (lead, 2026-11-12, restating R54); the
scheduler's file locks serialize such tasks and the later one rebases.

| Other PRD / stream | Direction | Seam mechanism | Owner | Order (X → Y: X depends on Y) |
|---|---|---|---|---|
| S3 escalation server and queue boundary (`plans/review-all-s3-…-prd.md`) | consumes | C4 `set_notify_callback(…, deliver_on=loop)` (S3 β) — the thread-safe wake α's hop needs; C4 `_mutate_pending` for `RefreshOpen` and `attach_to_parent` | S3 | α → 9181 (S3 ι; `server.py` order; transitively 9157 β, 9177 θ, 9173 κ, 9162 γ, 9167 δ, 9152 α; task 4567, once upstream through 9177, is done) |
| S3 | produces (to) | C1/C2 — the merge tools' filing and the ladder tools' gates. S3's PRD planned `filing_tools.py::submit_or_dedupe` (ε1) and moving `_chokepoint_or_submit` into `filing_tools.py` (ζ) "until S1's `filing.py` takes them"; under this order α/β land first and S3 moves thin callers instead | S1 | 9188 (S3 ε1) → β [lead edge]; 9188 and 9200 (S3 ζ) amended, and the S3 PRD's §Contracts C7a text with them: `register_merge_tools(filer: EscalationFiler, …)` replaces `file_escalation=`/`mint_escalation_id=`; ζ registers `escalate_*` over the filer and builds `FilingLookups` from `LadderLookups.task_status/task_claimant` (OQ-3, lead) |
| S3 | same files | `escalation/src/escalation/queue.py` (in depends-on form: S3 λ → θ → γ → β), `models.py` (S3 κ → γ); `orchestrator/src/orchestrator/stranded_reconcile.py` (S3 ε1 → S8 ε2b → S8 ε2a) | — | α after θ 9177 and κ 9173 (through 9181); 9205 (S3 λ) after α through ζ → ε1 → β; η3 → 9188 (end of the `stranded_reconcile.py` chain) |
| S10 β1/β2, S5 λ0/λ/ξ0, S7 δ (`escalation/server.py`) | same file | R54's per-file chain, in depends-on form: S3 ε1 9188 → S5 ξ0 9268 → S5 λ 9267 → S7 δ 9127 → S5 λ0 9256 → S10 β2 9191 → S10 β1 9187 → S3 ι 9181 | — | R54 applied (OQ-6, lead, flagged to Leo with S2-3): α → 9181, β → α, and 9187 (S10 β1) → β [lead edge], so α and β slot between S3 ι and S10 β1 and the chain resumes at 9187. Cost: S10 β1 (high) waits on α and β (~2,300 LOC), on S2's vocabulary, cascade-parent and callback leaves and task 8562 through α (S2-3 (a)), and on task 7300 (low, a test-only pin) through β |
| S7 merge-lane gates (`plans/review-all-s7-…-prd.md`) | consumes / absorbs | `merge_lane/gate_measurement.py::StormEscalationSink` (S7 α1, lead ruling OQ-CQ4: "S1 absorbs it when S1 widens `EscalationPort`"); S7 C2's await-free clause replaced by D6 | S1 (absorption) | ε1 → 9136 (S7 α1), 9154 (S7 ζ, last S7 task; `gate_measurement.py` settled) |
| S6 merge-lane lifecycle (`plans/review-all-s6-…-prd.md`) | consumes / absorbs | C7 `HaltState(…, submit_halt_owner=)`, `hold_with_escalation`, `file_holder_escalation` (θ 9176); C3 `MainHealthAutoHeal(submit=submit_or_dedupe)` (δ 9171); C2 `LifecycleRefusalResponder(escalation_queue=)` (ζ 9182, φ 9186) — each files on the raw queue, with a caller-minted id, a bound dedupe callable and S6's own `Filed`/`FilingFailed` names | S1 (absorption, C10) | ε3 → 9176, 9171, 9186; lead notes on 9176, 9171 and 9186 name C10 without changing their work (D15); same files: ε1 → 9188 (end of the `worker.py` chain); ε3 → 9193 (S3 ε2, end of `merge_lane/types.py`), 9201 (S6 μ, end of `merge_lane/__init__.py`); ε2b → 9201 (`landing_evidence.py`), 9182 (S6 ζ, `intake_buffers.py`) |
| S4 verify run context | same files | `merge_lane/ports.py`, `shadow.py`, `drift.py`, `review_checkpoint.py`, `flake_recorder.py`, `verify_runner.py` | — | ε1 → 9150 (S4 μ0, end of `ports.py`); ε2b → 9150 (`shadow.py`); κ3 → 9150 (`review_checkpoint.py`, `flake_recorder.py`); η2 → 9166 (S4 ρ, end of `verify_runner.py` and of `merge_lane/drift.py`) |
| S5 GitOps ports | same files | `merge_lane/liveness.py`, `skew_tripwire.py`, `lane_lifecycle.py` | — | ε2a → 9270 (S5 ο0), 9255 (S5 α); κ3 → 9262 (S5 κ) |
| S8 harness composition root | consumes | `SchedulerCallbacks` (S8 owns; ζ2 adds one field, removes seven); S8's filer modules `dispatch_gate_filers.py`, `infra_filers.py`, `escalation_record_sweeps.py` ("the repoint site for `escalation/filing.py::file`", S8 §Cross-PRD); S8's extracted `session_resume_storm.py`, `stranded_reconcile.py`, `deterministic_recon.py`; S8 β's store-unavailable L2 through `escalation/file_or_refresh.py::file_or_refresh` (task 8926, done) | S8 (modules), S1 (filing) | ζ2 → 9236 (S8 ω, end of `harness.py`); ζ1 → 9234 (S8 μ1, end of `scheduler.py`); η1 → ζ2; η3 → 9220 (S8 γ), 9224 (S8 ε2b), 9225 (S8 ε3); κ3 → 9224 (S8 ε2b, `recovery_emission.py`), 9216 (S8 α, `slot_lease.py`); κ4 → 9229 (S8 θ, end of `steward.py`); no edge onto 9236 for the ratchet baseline (R54 leaves test-data files to the scheduler's file locks) |
| S9 workflow phase spine | same files | `workflow.py` (S9's 21 leaves), `exit_contract.py`, `steward.py` (S9 ε), `stranded_verified_green.py` (S9 θ, C5 `file_stranded_merge_failed`) | — | δ → 9232 (S8 κ1, end of the `workflow.py` chain, after S9 ω 9289); κ3 → 9284 (S9 ν); κ4 → 9190 (S9 ε; κ4 keeps its row 15c, boundary row 23); η3 → 9213 (S9 θ) |
| S10 task-write contract | same files | `deterministic_runner.py`, `server/tools.py`, recon `harness.py`, `orphaned_recon_escalation_sweep.py`, `targeted.py`, `stage1_stall_detector.py`, `provenance_conflict.py` | — | γ → 9191 (S10 β2); θ1a → 9280 (S10 μ2); ι1 → 9254 (S10 λ), 9251 (S10 κ2); ι2 → 9285 (S10 ο); κ3 → 9283 (S10 ξ) |
| S6 α, S10 θ, S7 ζ invariant leaves (R51), and the wave-3 invariant leaves S2 ι, S16 ι, S19 ε and S17 ν (`one-decision-partition-per-area`) | same files | `design-invariants.md`, `-fixtures.md`, `gates.md` | lead | lead S2/OQ-10 (b): no wave-3 invariant leaf waits on 9154 for a number (the draft's ω → 9154 is dropped). Lead 2026-11-13: the six wave-3 minting tasks take INV-18 to INV-23 in landing order, S17 ν last as INV-23; the lead wires the invariant-file chain at decompose, with S2 ι → ω as both drafts have it and S17 ν → ω (ν last). S1 orders its own two: ω → λ2 (same three files; λ2 still waits on no door leaf). Every check anchors on its slug, never on an INV number |
| S3 μ | same file | `ARCHITECTURE.md` | — | ω → 9210 |
| S2 typed escalation record facts (wave 3, concurrent): vocabulary | produces (to S1) | S2 C1 (`EscalationStatus`, `Severity`, `ResolveAction`) in `escalation/models.py`; S1 lands the `Sentinel` registry and the moved role predicate in the same file after it | S2 (vocabulary), S1 (`Sentinel`, filing) | Leo's S2-3 (a), applied to S1/OQ-4 by the lead: α → S2 α (direct); α → S2 β1 and S2 β2 through 9181 → 9177 → S2 β2 → S2 β1 (the lead keeps 9177 → S2 β2). S2 α itself waits on S19 θ0 (`shared/escalation_layout.py`, lead S19/Q-C8) |
| S2: what the door writes | consumes (from S1) | S2's filer-declared facts (`failure_signature`, `filing_site`, `hypothesis`, `original_severity`) and S2 ε's `filed_via` join C1's `EscalationDraft` and its parity test; S2 writes `original_severity` in C2 step 2 in `filing.py`, not in the server closure (which α deletes); S2's resolve-action table and `promote_to_l2` header changes edit `server.py` after β. Caller identity (Leo S2-2 B widened by R-6 B): the door is the submit path, so every record it writes carries S2 ε's `filed_via` stamp (the draft's default `CallerChannel.IN_PROCESS`, the MCP tools' parsed channel otherwise); filings stay open to header-less callers, so no S1 filer changes; S1's boundary row 10b drives `promote_to_l2` with the watcher's identity header | S1 (filing), S2 (fields, identity) | S2 γ → β; S2 δ → β; S2 ε (the caller-identity leaf) → β (S2-3 (a); "the caller-identity task follows the door's other-filers task"); S2's identity deploy task and the later denial take no S1 edge |
| S2: one failure identity (lead ruling of 2026-11-13, "FINGERPRINT", which amends lead S2/OQ-3 (a); split into two S2 tasks by the lead the same day) | produces (to S2 γ2 and γ3) | S1 KEEPS writing `dedupe_fingerprint`: five readers neither plan's door work moves read the stored field (the dashboard's memory-eval panel, `dashboard/data/memory_evals.py`, which joins open `eval_regression` escalations on it, re-checked at `4a87282ee7`; fused-memory's `reconciliation/escalation_archive.py`; the recon harness's pending and resolved re-file checks; `EscalationQueue.has_open_l1`'s fingerprint match, S1 C3's `has_open(dedupe_fingerprint=)`; and the fold keys `dedupe.py::content_fingerprint_key`/`gate_backlog_fingerprint_key`), and S1's own `OpenAtLevel(same_fingerprint=True)` and `Fingerprint` policies rest on it. The retirement is carried by the typed-records plan (S2) as TWO tasks (lead 2026-11-13, for size; it amends the lead's "one task" wording): S2 γ2 moves those readers onto S2's declared failure identity, including the comparison inside S1 C3's `has_open` and the two fold keys, and changes no write, so it lands early; S2 γ3, after ω, deletes the field and edits S1's contracts: C1 `EscalationDraft.dedupe_fingerprint`, C1 `OpenAtLevel.same_fingerprint`, C3 `EscalationQueue.has_open(dedupe_fingerprint=)` and the `Fingerprint` member's fold key. S1 has no fingerprint-retiring leaf | S2 (γ2, γ3), S1 (the field until γ3) | S2 γ2 → S2 γ (the failure-signature task); S2 γ3 → S2 γ2 and → S1 ω (S1's closing task); no S1 leaf depends on γ2 or γ3. γ2 shares `dedupe.py` and `queue.py` with ι2 and ω, and recon `harness.py` with ι1, with no edge stated in either draft; that order is the lead's at decompose |
| S2: workflow, steward and other orchestrator readers | same files | S2 θ1 (`workflow.py`, `harness.py` readers), S2 θ2 (vocabulary compares in nine orchestrator modules, among them `deterministic_runner.py`, `escalation_router.py`, `l0_promotion.py`, `provenance_conflict.py`, `steward.py`) | — | S2 θ1 → δ and → η2 (S1's last `harness.py` leaf); S2 θ2 → γ, κ1, κ3, κ4 (S2 θ2 waits on 9287, 9231 and 9270, all behind 5040) |
| S2: fused-memory reader and the `DedupeConfig` rename | same files | S2 η (`reconciliation/backlog_policy.py`); S2 κ renames `DedupeConfig`'s fields in 33 files, among them `dedupe.py`, `worker.py`, both `harness.py` files, `workflow.py`, `provenance_conflict.py`, the fused-memory escalators, `backlog_policy.py`, `preservation_specimen_guard.py` and `task_count_snapshot_cadence.py` | — | ι1 → S2 η; S2 κ → ω (κ last on every shared file; ω follows S1's last leaf on each); D11 names the field by role, so the rename edits no S1 sentence |
| S12 recon run lifecycle (wave 3) | produces (to) | C8 `ReconEscalation` registry and C1 `FilingPolicy`; C8 is the one declaration of a recon alarm (D11): an S12 `RunStreak` names a member | S1 | S12 ε → ι1 (it declares its streak members in C8); only ε waits on ι1 (lead S12/OQ-7): S12 η2, which moves `_file_finding_task_escalation` and `BacklogIterator` out of recon `harness.py`, and S12 θ reach ι1 through S12's own chain (θ → η2 → η1 → κ2 → κ1 → ζ → ε), so neither takes a direct edge; S12 β1 and ι1 both edit recon `harness.py` and are unordered (lead S12/OQ-7: the file locks serialize them, the later rebases). S12 ε deletes `task_knowledge_sync.py::_maybe_escalate_growth_sweep_failures` and files that alarm through C8, so ι2 does not convert it and ω → S12 ε. The S12 author return's xref asked for the opposite direction ("S1's harness migration should depend on S12 ε"); wiring both would close a cycle, which `add_dependency` refuses once task 9490 lands |
| S17 scheduler seams (wave 3) | same files / consumes | S17 θ moves `_maybe_escalate_backfill_overstay_rate` into `backfill_admission.py::BackfillAdmission` and the warm-base hooks' producer into `warm_base_watchdog.py::WarmBaseHardDownWatchdog`; S17 ξ1 splits `DeterministicRunner.run` and names task 8355 as its precondition; S17 δ2 (the delivered-check gate) and λ (the slow-phase alarm) file through C5 `SchedulerEscalations`; S17 μ bundles `open_escalation_channel` from `mcp/markup_sink.py`, which κ2 edits | S1 (C5, C6, the `Sentinel` registry) | S17 θ → ζ2 (S1 first: S17 moves already-converted code; ζ2 → ζ1 covers ζ1); S17 δ2 → ζ2; S17 λ reaches ζ2 through δ2; S17 ξ1 → γ (γ subsumes 8355; S17 ξ1's "Task 8355" precondition is amended to γ); S17 μ → κ2 (`markup_sink.py`). S17's two scheduler filers (lead 2026-11-13; S17/OQ-9 = A) are both born at level 2 through S1's entry point: δ2 files `dependency_capability` per (dependent, dep) with `FilingPolicy.born_at_l2()`, subject the dependent task; λ files `infra_issue` at severity `critical` (the floor of `BORN_AT_L2_SEVERITIES`) with `FilingPolicy.born_at_l2()` on the `Sentinel` member `__delivered_check_phase_slow__`, which λ adds to S1's registry in `escalation/src/escalation/models.py`, one open record per project. The gate files no self-resolving level-1. Because δ2 → ζ2 and λ → δ2, neither lands before the door, so ζ1 has nothing of S17's to convert |
| S19 package layering (wave 3) | same files (moves) | S19 γ moves `middleware/_folded_escalation.py` → `escalators/folded_escalation.py` and the entity-mint, mem0-update, referent-repair and candidate-key escalators into `fused_memory/escalators/` (S19/OQ-4 (b), lead), and the dead-letter escalator into `services/dead_letter_escalator.py` (S19 D5: it imports `services.durable_queue` at module level since task 9143; lead correction of S1's draft path, 2026-11-12); S19 α/β1/β2 move recon state modules (`backlog_policy.py` → `recon_state/`) and repoint importers in `server/tools.py`, recon `harness.py`, `targeted.py`, `task_knowledge_sync.py`, `preservation_specimen_guard.py`, `standing_decision_storm_escape.py`; S19's own row reads "S1's leaves on the escalators → γ" | S19 (moves), S1 (filing) | θ1b, θ2 → S19 γ; θ1a, ι1, ι2 → S19 β2 (the last S19 move leaf on their files; through it γ, β1, α). Files below are written at the post-move paths, and every fused-memory check anchors on a symbol over a glob path that matches both layouts |
| S15 fused-memory server boundary (wave 3) | same file | `server/tools.py` (S15 ε4 and task 5270 also declare it); S15 ε4 moves `_report_unverified_claims` with `add_memory`/`add_episode` as θ1a leaves it | — | θ1a → 5270 (by readiness: 5270's one dependency, 4484, is done); S15 ε4 → θ1a (S15's draft; the same-file rule) |
| S26 memory service | not touched | `services/memory_service.py`'s four offloaded emitters stay correct under C1 (`file_blocking` called from `to_thread`); their "ASYNC ON PURPOSE"/"LOAD-BEARING" comments become redundant with the refusal | S26 | xref only; no S1 edit of `memory_service.py` |
| `plans/merge-lane-policy-prd.md` chain (8525–8528, 7720; 8525 and 7720 done) | upstream | new `worker.py` tasks park behind 8528 (R55; pending) | — | ε1, ε2a, ε2b, ε3 → 8528 |
| Eval framework, filed tasks on the eval files (8499 pending low: `build_eval_orch_config`'s routing pins; 6748 pending low, S18 κ: an import line in `evals/runner.py`; 7765 pending low: `_cell_metrics`; S8 κ1 9232: the invocation carries its sandbox backend, `evals/shadow_isolation.py`) | same files | `evals/runner.py`, `evals/profile.py`, `evals/shadow_invocation.py`, `workflow.py` | — | Leo ruled λ1 and μ independent of the other S1 tasks: no edge onto the filed tasks 8499, 6748 or 7765 (disjoint keys of one function, an import line, a different function; the file locks serialize them, the later rebases; 6748's edit is import-only, rebase-only under S19/Q-C6). Inside this plan the overlay's same-file rule chains the three tasks that edit `evals/runner.py` (lead 2026-11-13): μ → λ1, λ2 → μ (and λ2 → λ1). λ2 → 9232 (end of the wave-2 `workflow.py` chain, as the ruling places it); δ → λ2 (λ2 lands first on `workflow.py`, so λ2 waits on no door leaf). S2 μ (the identity configuration-deploy task) adds one-line edits to `evals/runner.py` and `workflow.py`; it takes no edge onto λ1, μ, λ2 or δ (lead 2026-11-13: the deploy task's one-line edits in those files are covered by the one-line exception; the file locks serialize, the later rebases). μ reads the sandbox backend the way the shadow cells do today, and after 9232 the way κ1 threads it |
| Daily memory-eval runner (`fused-memory/scripts/memory_eval_runner.py`) | not covered | its `eval_regression` escalations are meant to reach a human (lead reading, shown to Leo); λ1, λ2 and μ do not touch it; ι2 still moves its filing onto `file_blocking` | — | xref only |

## Decomposition plan

Overlay bands: every leaf targets 300–1,500 changed LOC and ≤ 12 declared files; same-file leaves are
serialized by real edges. Files are repo-relative; a fused-memory path that S19 moves is written at its
post-move path (§Cross-PRD S19 row). "Deletes" accounts for the synthesis's deletable list. Priority =
the highest severity of the keys the leaf closes; **high** leaves (α, β, ω) become dependencies of gate
6698 at decompose. λ1, λ2 and μ carry no finding key (they implement Leo's rulings on S1-2 and R-7) and
are medium by his ruling; they are not gate dependencies. Every task carries `x_finding_run = review-all-dark_factory-20261008`,
`source = review-all` and the keys listed. A leaf that converts a module also migrates the tests of that
module that call `has_open_l1`, `submit_or_dedupe` or define a raw-queue fake (measured at
`a85cb1ad78`, unchanged since `ba17934861`: 18 test files call `.has_open_l1(` or `submit_or_dedupe(`, 14 define a `has_open_l1`
fake, 30 distinct); each is declared by the leaf named below, and ω takes the five that test the
deleted queue methods themselves.

The synthesis's `deletable` list, accounted for (re-measured at `a85cb1ad78`; no row moved since `ba17934861`):

| Deletable (synthesis) | Leaf | At `a85cb1ad78` |
|---|---|---|
| `_chokepoint_or_submit`'s gate body | α | deleted (and `_submit_or_dedupe`) |
| 15 raw `Escalation()`+submit sites and raw queue params in merge_lane | ε1, ε2a, ε2b | 16 filings (worker 5, satellites 11), all deleted; the raw `escalation_queue` parameters that exist only to file go (the lane keeps the queue for resolve and reads) |
| scheduler's 2 inline sentinels | ζ1 | 3 inline sites + 3 `file_sentinel_l1` calls + `sentinel_l1.py`, all deleted |
| 17 hand-built workflow constructions | δ (+ ε3) | 17; those S6 moves into the lane are deleted by ε3, the rest by δ |
| `_RECON_DEDUP_CONFIG` and `_escalate`'s severity tuple | ι1 | deleted (derived from C8) |
| 5 "ASYNC ON PURPOSE" comments | θ1a | the `tools.py` one deleted; the `memory_service.py` ones are S26's file and stay correct under C1 (xref S26) |
| 3 `eval_lane_provenance` re-implementations | κ2, η1, η2 | 1 is a filing-time re-implementation (`markup_sink.py`), deleted by κ2; the 2 harness reads consume the containment outcome (the orphan-L0 backstop in `escalation_record_sweeps.py`, η1; the resolve-callback wake filter in `harness.py`, η2) and stay |

Harness accounting (critic B3): the 29 `Escalation(...)` constructions in `harness.py` at `a85cb1ad78`
(30 at `ba17934861`; task 8926 moved `_file_dirty_tree_escalation`'s onto `file_or_refresh`), plus that
one `file_or_refresh` call, by the module each lives in after wave 2 and the S1 leaf that converts it.

| Destination after wave 2 | Constructions (enclosing def) | S1 leaf |
|---|---|---|
| `dispatch_gate_filers.py` (S8 ζ1a) | `_block_and_escalate_external_dep`, `_block_and_escalate_delivered_check`, `_file_starvation_info` (three of the seven callbacks) | ζ2 |
| `dispatch_gate_filers.py` (S8 ζ1a) | `_block_and_escalate_cross_repo`, `_block_and_escalate_substrate_flip`, `_file_harness_l1` | η1 |
| `infra_filers.py` (S8 ζ1b) | `_file_warm_base_hard_down_notice`, `_promote_warm_base_hard_down_l2` (two of the seven callbacks) | ζ2 |
| `infra_filers.py` (S8 ζ1b) | `_file_archival_storm_escalation`, `_file_config_dir_ambiguous_escalation`, `_file_pool_storage_absent_escalation`, `_file_no_landings_info_escalation`, `_file_lane_record_drift_l2`, `_file_structural_exhaustion_l2`, `_file_config_unknown_keys_escalation`, `_file_watcher_outage_l2`, `_file_reblock_guard_l2`, `_escalate_reconcile_failure`, `_escalate_cascade_status_rejection` | η1 |
| `escalation_record_sweeps.py` (S8 ζ2) | `_reap_orphan_l0_escalations`, `_file_stale_l0_strand_escalation` | η1 |
| `session_resume_storm.py` (S8 γ) | `_file_session_resume_storm_escalation` | η3 |
| `stranded_reconcile.py` (S8 ε2a/ε2b) | `_reconcile_one_stranded`, `_maybe_submit_stranded_verified_green` | η3 |
| `deterministic_recon.py` (S8 ε3) | `_recover_stranded_deterministic_task`, `_recover_stranded_deterministic_gate` | η3 |
| `stranded_verified_green.py` (S9 θ, C5 `file_stranded_merge_failed`) | `_file_stranded_merge_failed` | η3 |
| `harness.py` (stays) | `_file_scheduler_pause_escalation`, `_run_main_tip_sweep` | η2 |

Totals: ζ2 5, η1 16, η3 6, η2 2 = 29 constructions, plus `_file_dirty_tree_escalation`'s
`file_or_refresh` call (`infra_filers.py` after S8 ζ1b), which η1 moves onto `RefreshOpen`. S8 β adds
one more after it lands (the store-unavailable L2 in `harness.py`, S8 C3b, filed through
`file_or_refresh`); η2 moves it onto `RefreshOpen`.

**α — The filing entry point: escalation/filing.py, the registry of system escalation ids and the level-parameterised open-record lookup**
(`task_kind=normal`, priority **high**, est. ~1,500 LOC)
- Files: escalation/src/escalation/filing.py (new), escalation/src/escalation/server.py,
  escalation/src/escalation/queue.py, escalation/src/escalation/models.py, escalation/src/escalation/dedupe.py,
  escalation/tests/test_filing.py (new), escalation/tests/test_server_chokepoint.py,
  escalation/tests/test_server_eval_lane.py, escalation/tests/test_queue.py, escalation/tests/test_models.py,
  escalation/tests/test_dedupe.py, escalation/tests/test_server.py.
- Deps: 9181 (S3 ι; `server.py` per-file order; transitively 9157 S3 β, the `deliver_on` and `_mutate_pending` substrate, and S2 β2 through 9177 under S2-3 (a)); S2 α (wave 3; the vocabulary in `models.py`; S2-3 (a)).
- Keys: fk-7d1f3a0d12ca, fk-333277fb4bd7, fk-32903f2e64e0, fk-14e1510916ab (carried; task 6609).
- Does: C1, C2, C3; `subject_for`; `RefreshOpen` (D13, with `file_or_refresh`'s rules); `promoted_l2`
  (D3); `EvalLaneScope` (D14); the `attach_or_submit` split (D6); `escalate_info`/`escalate_blocker` build drafts and choose
  policies with `eval_lane=ALL_LEVELS`; `create_server`'s `task_status_lookup`/`task_claimant_lookup`
  build the server's `FilingLookups`. `escalation/tests/test_server.py` imports
  `escalation.server._is_harness_sentinel_role`, which α moves; α repoints it.
- Deletes: `_chokepoint_or_submit`'s gate body and `_submit_or_dedupe` (server closure), the two
  `make_id` hops in the escalate tools, `server.py::_is_harness_sentinel_role` (→ `models.py`),
  `dedupe.py::attach_or_submit` (→ `attach_to_parent` + the filer's write).
- Signal: boundary rows 1–9b green; an `escalate_blocker` call on the running orchestrator returns the
  same shape as before and its record carries the claimant stamp; no `async def _chokepoint_or_submit`
  remains under `escalation/src`.
- G7: INV-8 (the hop; row 5), INV-2 (every refusal names values), INV-11 (`file_soon` drop logged,
  fail-open lookups logged), INV-3 n/a. INV-4: a filing failure storm has no escalation channel by
  construction (the queue is the channel); logged ERROR is the floor — recorded, not waived.

**β — Every other escalation-package filer, including the watcher's promotion to level 2, files through the entry point**
(`task_kind=normal`, priority **high**, est. ~900 LOC)
- Files: escalation/src/escalation/server.py, escalation/src/escalation/note_to_file.py,
  escalation/src/escalation/guard_storm_alarm.py, escalation/src/escalation/submit.py,
  escalation/src/escalation/mismatch_streak.py, escalation/tests/test_note_to_file.py,
  escalation/tests/test_guard_storm_alarm.py, escalation/tests/test_mismatch_streak.py,
  escalation/tests/test_server.py, escalation/tests/test_submit_cli.py.
- Deps: α; 7300 (pins the external producer's whole `escalation submit` call, which β reroutes). Lead edges: 9188 (S3 ε1) → β; 9187 (S10 β1) → β (OQ-6); S2 γ, δ and ε → β (S2-3 (a)).
- Keys: fk-0d31f1114911.
- Does: `_report_amendment_truncation_storm`, `_report_root_cause_overfold` (`Fingerprint`),
  `_file_markup_residue` (`NoDedup`; residue records are never deduped) and the markup storm
  (`file_storm_once` → `OpenAtLevel(1, category, agent_role)` + its existing amend-on-open note),
  `promote_to_l2`'s create step through `FilingPolicy.promoted_l2(root_cause)` with the subject from
  `subject_for(task_id)` (a sentinel-class cluster is promoted under its own sentinel id, as today's
  `sentinel_task_id_required` refusal requires) and `members`/`options` carried in the draft;
  the tool's own eval-lane refusal stays ahead of the filer; `merge_cancel`'s
  `_escalate_workflowless_cancel`, `file_note_to_file` (becomes C2 step 6), `submit.py` (`file_blocking`
  with `subject_for`; its role/severity refusal calls `models.is_harness_sentinel_role`),
  `mismatch_streak.py` (`Fingerprint` via the filer).
- Deletes: every `queue.submit`/`queue.submit_resolved` code line in `server.py` (5), the "Bypasses
  chokepoint" docstring claim on `promote_to_l2`, `note_to_file.py::file_note_to_file` (moved).
- Signal: rows 10, 10b and 11 green; `escalation/src` constructs `Escalation` only in `filing.py` and
  `dedupe.py::file_or_fold_l1` (ι2 removes the latter); a live watcher promotion of a `blocking` cluster
  on dark-factory creates a level-2 record (`get_escalation` shows `level: 2`).
- G7: INV-5 (the markup residue/storm record shapes are not unified here — task 5258's scope, amended
  to build drafts; §Live tasks).

**γ — A typed outcome for task status writes, and one file-then-block helper for the deterministic runner**
(`task_kind=normal`, priority medium, est. ~700 LOC)
- Files: orchestrator/src/orchestrator/status_write.py (new), orchestrator/src/orchestrator/deterministic_runner.py,
  orchestrator/tests/test_status_write.py (new), orchestrator/tests/test_deterministic_runner.py,
  orchestrator/tests/test_deterministic_runner_predicate_recheck.py.
- Deps: α; 9191 (S10 β2, end of the `deterministic_runner.py` chain).
- Keys: fk-0ee68c0948ab (helper + the runner's 8 arms).
- Does: C6. The five `_file_*_and_block` tails become `file_and_block` with
  `expected_status='pending'` or `'in-progress'` as each path requires; `_rearm_predicate`,
  `_writeback_deploy_success` and `run`'s done write use `write_status` and match on the outcome.
  `DeterministicRunner` files through `EscalationFiler(self.escalation_queue)` (lookup-less, D7);
  `file_deterministic_escalation` and `file_illegal_deploy_transition` build drafts with
  `FilingPolicy.born_at_l2()`; `build_gate_escalation_fields` (the one gate-fields builder for
  `milestone_gate` and `review_due` since task 6390, read by the runner and by the harness's
  `_recover_stranded_deterministic_gate`, η3) returns draft fields. `status_write.py` imports nothing from `deterministic_runner.py`, which
  is task 8355's lower-module filer (8355 cancelled into γ; S17 ξ1 owns its predicate-path split).
- Deletes: the five copies of the "file, advance deploy phase, block, swallow" tail; 8 bare
  `except Exception` arms around `set_task_status`; the runner's 5 `Escalation(...)`.
- Signal: rows 12–13 green; a pure gate cancelled by recon between filing and blocking stays
  `cancelled` and its gate escalation reads resolved benign (task 3887's scenario, driven end to end).
- G7: INV-3 (re-read before the block write), INV-6 (no status implies ownership after the write is
  skipped), INV-11 (each outcome distinguishable).

**δ — One workflow filing helper for the task workflow's escalations, each emitting escalation_created**
(`task_kind=normal`, priority medium, est. ~1,100 LOC)
- Files: orchestrator/src/orchestrator/workflow.py, orchestrator/tests/test_workflow_escalation_filing.py (new),
  orchestrator/tests/test_workflow_escalation_warning.py, orchestrator/tests/test_workflow_escalated_steward_stall.py,
  orchestrator/tests/test_workflow_train_completion.py, orchestrator/tests/test_workflow_e2e.py,
  orchestrator/tests/test_workflow_merge_gating_strand.py, orchestrator/tests/test_workflow_preexisting_main_break.py,
  orchestrator/tests/test_workflow_status_on_resume.py.
- Deps: α, γ, λ2 (`workflow.py`: λ2 takes the end of the wave-2 order on that file, as Leo's ruling
  places it, and δ follows); 9232 (S8 κ1, end of the wave-2 `workflow.py` chain; after S9 ω 9289 and S6
  θ 9176), through λ2.
- Keys: fk-9c72e4399b4e, fk-0ee68c0948ab (the workflow's 4 arms).
- Does: C7; every workflow filing that remains in `workflow.py` at dispatch (re-derived with the
  alias-resolved scan: 17 at `a85cb1ad78`, unchanged since `ba17934861`, less the WIP-halt filings S6 θ moves into the lane and
  `_file_main_health_l1`'s path S6 δ moves into `MainHealthAutoHeal`, which ε3 converts);
  `_file_architect_merge_failed` passes `agent_role=role` (its sentinel role) with
  `OpenAtLevel(category='stranded_merge_failed', agent_role=role)`, keeping it a deduped level-2
  filing; `_handle_terminal_exit_on_block` via `file_and_block`; `_repend_for_requeue`,
  `_persist_blocked_row`, `_revert_withheld_member` via `write_status`; the 5 `has_open_l1` calls
  (measured at `a85cb1ad78`) become `has_open(level=1)` or a policy; 6348's two `workflow.py` sites use
  `Fingerprint(DedupeConfig.for_content_fingerprint(…))`. The eval cell factory passes no filer, and a
  test asserts it (Leo's S1-2 ruling): the workflow derives its filer from its escalation queue, the
  eval build passes none (`evals/runner.py::_cell_workflow`), so λ1's refusal of a production queue
  covers the filer too, and boundary row 14b pins "no queue, no filer, nothing written".
- Deletes: the workflow's remaining hand-built `Escalation(...)` constructions and their per-site
  emits; the silent filings among them; the test docstring claiming merge-deferred is "unreachable by
  every recovery path" (`test_workflow_train_completion.py`).
- Signal: rows 14 and 14b green; in `runs.db`, the `escalation_created` count equals the workflow-filed
  record count over the boundary fixture (every written record, was 11 of 17).
- G7: INV-5 (one construction site), INV-13 (the event's reader, `evals/reviewer_trial/mining.py`, reads
  rows the real producer writes).

**ε1 — The merge lane's escalation port takes a draft and a policy; the gate-measurement storm sink folds into it**
(`task_kind=normal`, priority medium, est. ~800 LOC)
- Files: orchestrator/src/orchestrator/merge_lane/ports.py, orchestrator/src/orchestrator/merge_lane/worker.py,
  orchestrator/src/orchestrator/merge_lane/gate_measurement.py, orchestrator/tests/_merge_lane_fakes.py,
  orchestrator/tests/test_gate_measurement.py, orchestrator/tests/test_merge_lane_escalation_port.py (new),
  orchestrator/tests/merge_lane_ratchet_baseline.json, orchestrator/tests/test_merge_queue_concurrent_verify.py,
  orchestrator/tests/test_merge_queue_duplicate_submission.py, orchestrator/tests/test_merge_queue_invariant_integration_gate.py,
  orchestrator/tests/test_merge_queue_resource_audit.py.
- Deps: α; 9136 (S7 α1), 9154 (S7 ζ), 9188 (S3 ε1, end of `worker.py`), 9150 (S4 μ0, end of `ports.py`), 8528 (R55; parked behind it at filing).
- Keys: fk-2e0a4419290b (port half).
- Does: C4; `ProductionEscalations` over an `EscalationFiler`; `RecordingEscalations` implements the
  port; the worker's `_alarm_verify_worktree_contention` (level 2, `OpenAtLevel()`),
  `_run_post_merge_verify`'s cross-check filing, `_alarm_resource_audit`, `_submit_loop_escalation` and
  the `self._escalation.file(EscalationRecord(...))` site file through the port with drafts;
  `GateUnmeasuredReporter` takes the port (D12); the worker's 6348 site uses
  `Fingerprint(DedupeConfig.for_content_fingerprint(…))`. The four worker test files' raw-queue fakes
  (each defines `has_open_l1`) become `RecordingEscalations` or a real queue under `tmp_path`.
  `_file_main_health_escalation` and `_alarm_illegal_lifecycle_transition` are not ε1's: S6 δ and S6 ζ/φ
  move them out of `worker.py` (ε3).
- Deletes: `StormEscalationSink`, the worker's raw `escalation_queue.submit` sites that remain in
  `worker.py` at dispatch.
- Signal: rows 15 (worker half) and 16 green; on the running lane a forced verify-worktree contention
  files one L2 and a repeat files none (open-at-stamped-level policy), observed through
  `get_pending_escalations`.
- G7: INV-8 (lane filings leave the loop), INV-10 (the fake records policies, so tests assert behaviour).

**ε2a — The merge lane's liveness, request-ledger, verify-base and skew-tripwire alarms file through the port**
(`task_kind=normal`, priority medium, est. ~550 LOC)
- Files: orchestrator/src/orchestrator/merge_lane/liveness.py, orchestrator/src/orchestrator/merge_lane/request_ledger.py,
  orchestrator/src/orchestrator/merge_lane/verify_base_detector.py, orchestrator/src/orchestrator/merge_lane/skew_tripwire.py,
  orchestrator/src/orchestrator/merge_lane/worker.py, orchestrator/tests/test_merge_queue_multihost_wiring.py,
  orchestrator/tests/test_multihost_verify_integration.py, orchestrator/tests/test_merge_queue_host_observability.py,
  orchestrator/tests/test_merge_request_ledger.py, orchestrator/tests/merge_lane_ratchet_baseline.json.
- Deps: ε1; 9270 (S5 ο0, `liveness.py`), 9255 (S5 α, `skew_tripwire.py`), 8493 (`verify_base_detector.py`'s event attribution), 8528.
- Keys: fk-2e0a4419290b (contributes).
- Does: `_alarm_verify_host_unreachable`, `_alarm_verify_host_skew`, `_alarm_merge_request_stuck`,
  `alarm_verify_base_divergence` choose `FilingPolicy.open_l1()` on `Sentinel` members;
  `emit_pipeline_landing_tripwire` chooses `OpenAtLevel(any_level=True)` (today's pending-at-any-level
  `get_by_task` read) on its `pipeline-landing-tripwire-<sha>` prefix sentinel; the worker passes the
  port. `liveness.py::_resolve_pending_alarm` keeps the raw queue for its resolve sweep (task 8665,
  amended).
- Deletes: 6 raw `Escalation(...)` + `escalation_queue.submit` sites and their `has_open_l1`/`get_by_task`
  dedup reads in the four modules.
- Signal: row 15 green for these alarms; a verify host made unreachable on the running lane files one
  L1 and its recovery resolves it, as today.
- G7: INV-5, INV-8.

**ε2b — The merge lane's shadow, landing-evidence, member re-drive and intake-buffer alarms file through the port**
(`task_kind=normal`, priority medium, est. ~600 LOC)
- Files: orchestrator/src/orchestrator/merge_lane/shadow.py, orchestrator/src/orchestrator/merge_lane/landing_evidence.py,
  orchestrator/src/orchestrator/merge_lane/member_redrive.py, orchestrator/src/orchestrator/merge_lane/intake_buffers.py,
  orchestrator/src/orchestrator/merge_lane/worker.py, orchestrator/tests/test_merge_shadow.py,
  orchestrator/tests/test_landing_evidence.py, orchestrator/tests/test_branch_work_landed.py,
  orchestrator/tests/test_merge_queue.py, orchestrator/tests/merge_lane_ratchet_baseline.json.
- Deps: ε2a (`worker.py`); 9150 (S4 μ0, `shadow.py`), 9201 (S6 μ, `landing_evidence.py`), 9182 (S6 ζ, `intake_buffers.py`), 8528.
- Keys: fk-2e0a4419290b (closes).
- Does: the three `shadow.py` level-2 filings (`_submit_shadow_divergence_escalation`,
  `_alarm_warm_shadow_unparseable`, `_submit_coarse_shadow_divergence_escalation`) choose `OpenAtLevel()`
  (the stamped level 2), fixing today's level-1 dedup that never matches an L2 (D3); the two
  `landing_evidence.py` filings and `_alarm_coalesce_live_original` choose `FilingPolicy.open_l1()`;
  `member_redrive.py` passes the port instead of `self.escalation_queue` to
  `landing_evidence.file_unattributed_landing_escalation`.
- Deletes: 6 raw `Escalation(...)` + `escalation_queue.submit` sites and their `has_open_l1` reads; the
  `escalation_queue` parameters that existed only to file.
- Signal: rows 15 and 15b green; no `escalation_queue.submit(` remains in the lane's satellite
  modules (ε3 brings the lane-wide count to 0 once S6's new modules take the port); a repeated warm/cold
  divergence on the running lane leaves one pending L2, not one per divergence (a behaviour change:
  fewer human pages).
- G7: INV-5, INV-8, INV-4 (the repeat suppression is the storm escape the level-1 dedup never gave).

**ε3 — The merge lane's halt-owner, auto-heal and lifecycle-refusal escalations file through the port**
(`task_kind=normal`, priority medium, est. ~650 LOC)
- Files: orchestrator/src/orchestrator/merge_lane/halt.py, orchestrator/src/orchestrator/merge_lane/main_health.py,
  orchestrator/src/orchestrator/merge_lane/item_lifecycle.py, orchestrator/src/orchestrator/merge_lane/types.py,
  orchestrator/src/orchestrator/merge_lane/__init__.py, orchestrator/src/orchestrator/merge_lane/worker.py,
  orchestrator/src/orchestrator/harness.py, orchestrator/tests/test_merge_lane_halt.py,
  orchestrator/tests/test_halt_owner.py, orchestrator/tests/test_merge_lane_main_health.py,
  orchestrator/tests/test_merge_lane_lifecycle_refusal_policy.py, orchestrator/tests/merge_lane_ratchet_baseline.json.
- Deps: ε2b (`worker.py`), ζ2 (`harness.py`; ζ2 is S1's first leaf on that file); 9176 (S6 θ, C7), 9171 (S6 δ, C3), 9186 (S6 φ, C2), 9193 (S3 ε2, end of `merge_lane/types.py`), 9201 (S6 μ, end of `merge_lane/__init__.py`), 9236 (S8 ω, end of `harness.py`), 8528.
- Keys: fk-2e0a4419290b (contributes), fk-9c72e4399b4e (the WIP-halt and main-health filings S6 moves out of `workflow.py`).
- Does: C10 (D15). `HaltState` takes the port; `hold_with_escalation` and `file_holder_escalation` hold
  under a provisional holder id, await `file()` with `NoDedup` and re-key on the filed id
  (ε3 renames S6's values `HolderFiled`/`HolderFilingFailed`; the lead's note on 9176 names it); `MainHealthAutoHeal` files with
  `FilingPolicy.folded(<main-health DedupeConfig>)`; `LifecycleRefusalResponder` files its refusal L1
  through the port; whichever of `worker.py`/`harness.py` constructs `HaltState` after S6 θ passes the
  port (measured at dispatch). Every other caller of `hold_with_escalation`/`file_holder_escalation`
  at dispatch is converted here (neither symbol exists at `a85cb1ad78`).
- Deletes: `merge_lane/types.py::EscalationRecord` and its façade re-export in `merge_lane/__init__.py`;
  `submit_halt_owner`; the bound `submit_or_dedupe` callable; S6's `Filed`/`FilingFailed` names.
- Signal: rows 24 and 25 green; on the running lane `get_merge_halt_status` lists a WIP holder keyed by
  the escalation id `get_pending_escalations` shows, and resolving it releases the lane.
- G7: INV-7 (every holder keeps a machine-readable owner), INV-11 (a filing failure releases its holder
  and emits `merge_halt_owner_filing_failed`), INV-8.

**ζ1 — Scheduler sentinel alarms built as drafts and policies; sentinel_l1.py deleted**
(`task_kind=normal`, priority medium, est. ~600 LOC)
- Files: orchestrator/src/orchestrator/scheduler.py, orchestrator/src/orchestrator/anchor_coverage_gate.py,
  orchestrator/src/orchestrator/starvation_watchdog.py, orchestrator/src/orchestrator/work_product.py,
  orchestrator/src/orchestrator/sentinel_l1.py (deleted), orchestrator/tests/test_recovery_emission.py,
  orchestrator/tests/test_scheduler_continuity.py, orchestrator/tests/test_scheduler_scoring.py,
  orchestrator/tests/test_work_product.py, orchestrator/tests/test_scheduler_park_backfill.py.
- Deps: α; 9234 (S8 μ1, end of `scheduler.py`); 7718.
- Keys: fk-cd446ff3da74 (contributes).
- Does: the scheduler holds one lookup-less `EscalationFiler` over its `escalation_queue`;
  `_file_fm_read_storm_escalation`, `_maybe_escalate_backfill_overstay_rate`,
  `trigger_retry_cap_exhausted` (its queue parameter goes) and the three `file_sentinel_l1` calls choose
  `FilingPolicy.open_l1()` on `Sentinel` members; `work_product.py::_failure_escalation` builds a draft
  and its `escalate` callable takes `(draft, policy)`, wired by `scheduler.py` to the filer's
  `file_soon`. Carries task 6942's intent (one shape for the two scheduler L1 filers; 6942 cancelled into
  ζ1).
- Deletes: `sentinel_l1.py` (`SentinelL1`, `file_sentinel_l1`), 3 inline `Escalation(...)` and 3
  function-local `from escalation.models import Escalation`, `Scheduler._FM_READ_STORM_SENTINEL` and
  `_BACKFILL_OVERSTAY_SENTINEL`.
- Signal: an fm read storm on a bare `Scheduler` fixture files one L1 on `Sentinel.FM_READ_STORM` and a
  repeat files none; `sentinel_l1.py` is gone and its four test importers pass.
- G7: INV-5, INV-11.

**ζ2 — One scheduler escalation port replacing seven escalation callbacks**
(`task_kind=normal`, priority medium, est. ~800 LOC)
- Files: orchestrator/src/orchestrator/scheduler_escalations.py (new), orchestrator/src/orchestrator/scheduler.py,
  orchestrator/src/orchestrator/harness.py, orchestrator/src/orchestrator/dispatch_gate_filers.py,
  orchestrator/src/orchestrator/infra_filers.py, orchestrator/tests/test_scheduler_escalations.py (new),
  orchestrator/tests/test_scheduler_callbacks.py, orchestrator/tests/test_harness_starvation_watchdog.py,
  orchestrator/tests/test_dispatch_gate_filers.py.
- Deps: ζ1, γ; 9236 (S8 ω, end of `harness.py`; S8 ζ1a/ζ1b moved the callbacks).
- Keys: fk-cd446ff3da74 (closes), fk-0ee68c0948ab (2 arms).
- Does: C5; the external-dep and delivered-check blocks use `file_and_block`; starvation and
  warm-base warn/resolve/promote become drafts + the async `resolve_open`; the scheduler's ζ1 filer
  becomes the injected port.
- Deletes: 7 `SchedulerCallbacks` fields and their harness wiring, the 7 implementations in S8's two
  filer modules (5 constructions; §Harness accounting), 2 swallow arms.
- Signal: row 17 green; a bare `Scheduler` built with a `RecordingSchedulerEscalations` records the
  starvation L0 a stalled fixture triggers (today it logs "no escalation_queue wired" and files nothing).
- G7: INV-5, INV-7 (the resolve path keeps one owner), INV-8 (`resolve_open` hops), INV-11 (no silent
  no-op when unwired: the discarding port logs once per kind).

**η1 — Harness dispatch-gate, infrastructure and escalation-sweep filers through the entry point; born_at_l2.py deleted**
(`task_kind=normal`, priority medium, est. ~1,300 LOC)
- Files: orchestrator/src/orchestrator/dispatch_gate_filers.py, orchestrator/src/orchestrator/infra_filers.py,
  orchestrator/src/orchestrator/escalation_record_sweeps.py, orchestrator/src/orchestrator/born_at_l2.py (deleted),
  orchestrator/src/orchestrator/lane_record_divergence.py, orchestrator/tests/test_dispatch_gate_filers.py,
  orchestrator/tests/test_born_at_l2.py (deleted), orchestrator/tests/test_harness_escalation_filing_identity.py,
  orchestrator/tests/test_stale_escalation_cleanup.py, orchestrator/tests/test_pool_storage_guard.py,
  orchestrator/tests/test_orphan_l0_reaper.py.
- Deps: ζ2 (same files); through it 9236. (Task 8926, the draft's edge, is done: its helper is on main.)
- Keys: fk-7d1f3a0d12ca (contributes), fk-0ee68c0948ab (3 arms).
- Does: the 16 constructions §Harness accounting assigns to η1 build drafts and choose policies; the
  four born-at-L2 filers task 8826 moved onto `file_deduped_born_at_l2` use `FilingPolicy.born_at_l2()`;
  `_file_dirty_tree_escalation` moves from `file_or_refresh(…, _DIRTY_TREE_ALARM, …)` to `RefreshOpen`
  (D13; its clean-tree resolve keeps reading the alarm's subject and level from one place); the three
  dispatch-gate block-and-file tails use `file_and_block`; the orphan-L0 reaper's eval-lane read stays
  (it consumes the containment outcome and is half of Leo's "no automated filer" rule, D14); the
  filing-identity tripwire asserts the claimant on drafts instead of on constructions and
  `file_or_refresh` calls (task 8926 taught it to count those).
- Deletes: `born_at_l2.py` (`BornAtL2Alarm`, `file_deduped_born_at_l2`), every `Escalation(...)` in the
  three modules, 3 swallow arms.
- Signal: rows 15/17's harness-side goldens; `test_harness_escalation_filing_identity.py` passes with
  the same site count; a dirty project root at startup leaves one pending record whose `dedupe_count`
  rises on each restart.
- G7: INV-5, INV-9.

**η2 — The filings that stay in harness.py, the drift detectors and the harness's remaining swallowed status writes**
(`task_kind=normal`, priority medium, est. ~800 LOC)
- Files: orchestrator/src/orchestrator/harness.py, orchestrator/src/orchestrator/verify_runner.py,
  orchestrator/src/orchestrator/merge_lane/drift.py, escalation/src/escalation/file_or_refresh.py (deleted),
  escalation/tests/test_file_or_refresh.py (deleted), orchestrator/tests/test_harness_train_callbacks.py,
  orchestrator/tests/test_harness_escalation_filing_identity.py, orchestrator/tests/test_merge_drift.py,
  orchestrator/tests/test_verify_runner.py.
- Deps: η1 and κ1 (they convert the two landed callers of `escalation/file_or_refresh.py`, which η2 deletes), ε3 (`harness.py`: η2 is S1's last leaf on it); 9166 (S4 ρ, end of `verify_runner.py` and `merge_lane/drift.py`); 9218 (S8 β, the third caller; already upstream through ε3 → 9236).
- Keys: fk-0ee68c0948ab (6 arms), fk-7d1f3a0d12ca (contributes).
- Does: `_file_scheduler_pause_escalation`, `_run_main_tip_sweep`, S8 β's store-unavailable L2
  (`RefreshOpen` in place of `file_or_refresh`), `verify_runner.py`'s two drift-detector filings
  (`DriftDetector.check`, `check_env`) and the `escalation_queue` argument `drift.py` threads into
  `DriftDetector`; `_revert_withheld_member`, `_maybe_auto_eval` ×2, `_cascade_unblock_member` ×2,
  `_block_if_still_pending` via `write_status`/`file_and_block`; `Harness` drains its long-lived filers
  on shutdown (`EscalationFiler.drain`), so a clean stop loses no `file_soon` filing (D1). The
  resolve-callback wake filter's eval-lane read stays.
- Deletes: harness's remaining `Escalation(...)` constructions and `queue.submit` calls (2
  constructions at `a85cb1ad78`, of 22 `_escalation_queue.submit(` lines today, the rest moved by S8 and
  converted by ζ2, η1 and η3; plus S8 β's one), 6 swallow arms; task 8926's helper module
  `escalation/src/escalation/file_or_refresh.py` and its test `escalation/tests/test_file_or_refresh.py`
  (η2 converts the last of its three callers; κ1 and η1 converted the other two); the test docstring claiming merge-deferred is
  "unreachable by every recovery path" (`test_harness_train_callbacks.py`).
- Signal: a forced revert failure in the train fixture yields `Failed` with a WARNING naming the member
  and site, and the member is re-driven by the abandoned-member sweep on its second pass.
- G7: INV-6, INV-11.

**η3 — Harness filings in the modules the composition-root refactor creates: session-resume storm, stranded reconcile, deterministic recon, stranded verified-green**
(`task_kind=normal`, priority medium, est. ~550 LOC)
- Files: orchestrator/src/orchestrator/session_resume_storm.py, orchestrator/src/orchestrator/stranded_reconcile.py,
  orchestrator/src/orchestrator/deterministic_recon.py, orchestrator/src/orchestrator/stranded_verified_green.py,
  orchestrator/tests/test_session_resume_storm.py, orchestrator/tests/test_stranded_reconciler.py,
  orchestrator/tests/test_deterministic_recon.py, orchestrator/tests/test_stranded_verified_green.py,
  orchestrator/tests/test_reconcile_stranded.py.
- Deps: α, γ; 9220 (S8 γ, `session_resume_storm.py`), 9224 (S8 ε2b), 9225 (S8 ε3, `deterministic_recon.py`), 9213 (S9 θ, `stranded_verified_green.py`), 9188 (S3 ε1, end of the `stranded_reconcile.py` chain), 9236 (S8 ω).
- Keys: fk-7d1f3a0d12ca (contributes), fk-0ee68c0948ab (1 arm).
- Does: the 6 constructions §Harness accounting assigns to η3 build drafts and choose policies (the
  two level-2 recovery filings `OpenAtLevel()`, the storm L1 `FilingPolicy.open_l1()`); the
  `_reconcile_one_stranded` arm uses `file_and_block`; `test_reconcile_stranded.py`'s raw-queue fake
  becomes a real queue.
- Deletes: every `Escalation(...)` in the four modules, 1 swallow arm.
- Signal: the stranded-reconcile sweep's summary log line and the session-resume storm L1 text are
  byte-identical to the pre-S1 goldens S8's leaves record, filed through the filer (row 20's recorder).
- G7: INV-5.

**θ1a — Fused-memory's folded escalation filer gets an async twin; the on-loop emitters await it**
(`task_kind=normal`, priority medium, est. ~700 LOC)
- Files: fused-memory/src/fused_memory/escalators/folded_escalation.py (post-move path; S19 γ (S19/OQ-4 (b))), fused-memory/src/fused_memory/server/tools.py,
  fused-memory/src/fused_memory/server/markup_guard.py, fused-memory/src/fused_memory/server/markup_tripwire.py,
  fused-memory/src/fused_memory/services/completion_claim_gate.py, fused-memory/tests/test_folded_escalation.py,
  fused-memory/tests/server/test_markup_tripwire.py, fused-memory/tests/test_markup_guard_fused_memory.py,
  fused-memory/tests/test_completion_claim_gate.py, fused-memory/tests/server/test_markup_tripwire_gate.py.
- Deps: α, β; 4809 (pending, `markup_guard.py`/`tools.py`); 9280 (S10 μ2, end of `tools.py`); 5270 (`tools.py`, by readiness); S19 β2 (wave 3; the last S19 move leaf on these files).
- Keys: fk-1bf4bfa8be04.
- Does: C9 with `OpenAtLevel(any_level=True)`; the async `afile_folded_escalation`;
  `_report_unverified_claims` and the markup guard's residue/storm sinks become async and await it;
  `_file_triage_fail_open_storm`'s `to_thread` wrapper goes. θ1a records, for each of the 12
  `file_folded_escalation` call sites (the twelfth is the ticket janitor's curation-backlog alarm,
  `asyncio.to_thread(file_folded_escalation, …)` in `middleware/ticket_janitor.py`), whether it runs on a
  loop thread (measured by driving its public trigger under a running loop), and converts the on-loop
  ones in its own files; its three call sites (`markup_tripwire.py` ×2, `completion_claim_gate.py`) pass
  a typed `anchor=`; `markup_tripwire`'s public anchor parameter takes the typed `SentinelSubject` too,
  and the gate test's fake of `emit_markup_storm_escalation` follows it; θ1b and θ2 convert the rest. So the eight callers θ1b owns and the janitor's call θ2
  owns keep working in between, C9 keeps a transitional `anchor_task_id: str` keyword (parsed through
  `subject_for`) that θ2 removes.
- Deletes: the "ASYNC ON PURPOSE — do not re-inline" docstring in `tools.py`; `file_folded_escalation`'s
  own `Escalation(...)` + `queue.submit` and its `get_by_task` fold.
- Signal: row 18 green; on the live fused-memory server an `add_memory` carrying an unverified
  completion claim files the per-ref escalation and the loop-lag monitor (task 3778) records no stall
  for the call.
- G7: INV-8, INV-5.

**θ1b — The other eight fused-memory folded-escalation callers pass typed anchors**
(`task_kind=normal`, priority medium, est. ~400 LOC)
- Files: fused-memory/src/fused_memory/maintenance/link_heal_executor.py, fused-memory/src/fused_memory/escalators/candidate_key_escalation.py (post-move path; S19 γ (S19/OQ-4 (b))),
  fused-memory/src/fused_memory/escalators/referent_repair_storm_escalator.py (post-move path; S19 γ (S19/OQ-4 (b))), fused-memory/src/fused_memory/server/write_triage.py,
  fused-memory/src/fused_memory/services/classification_fallback_alarm.py, fused-memory/src/fused_memory/services/journal_growth_alarm.py,
  fused-memory/src/fused_memory/services/mem0_retry_backlog_alarm.py, fused-memory/src/fused_memory/services/memory_metadata_census.py,
  fused-memory/tests/test_journal_growth_alarm.py, fused-memory/tests/test_classification_fallback_alarm.py,
  fused-memory/tests/test_referent_repair_storm_escalator.py, fused-memory/tests/test_candidate_key_escalation.py,
  fused-memory/tests/server/test_write_triage.py, fused-memory/tests/test_memory_metadata_census.py.
- Deps: θ1a; S19 γ (wave 3).
- Keys: fk-1bf4bfa8be04 (contributes).
- Does: each caller passes a `SentinelSubject` anchor from the registry instead of an `anchor_task_id`
  string; every caller θ1a measured as on-loop awaits `afile_folded_escalation`. 14 files, two over the
  band: the two spies of converted callers (`test_write_triage.py`'s `TestDelegatesToTheSharedHelper`,
  `test_memory_metadata_census.py`'s spies) read the keyword θ1b replaces with `anchor=`.
- Deletes: the 8 modules' string anchors.
- Signal: row 18's anchor-uniqueness test passes over the registry members; a mem0 retry backlog on the
  live server files one record and a repeat folds into it.
- G7: INV-8, INV-9.

**θ2 — Fused-memory escalators that build their own queue file through the entry point; the ticket janitor's outage alarm folds**
(`task_kind=normal`, priority medium, est. ~650 LOC)
- Files: fused-memory/src/fused_memory/middleware/curator_escalator.py, fused-memory/src/fused_memory/services/dead_letter_escalator.py (post-move path; S19 γ, D5),
  fused-memory/src/fused_memory/escalators/entity_mint_storm_escalator.py (post-move path; S19 γ (S19/OQ-4 (b))), fused-memory/src/fused_memory/escalators/mem0_update_storm_escalator.py (post-move path; S19 γ (S19/OQ-4 (b))),
  fused-memory/src/fused_memory/middleware/scope_violation_escalator.py, fused-memory/src/fused_memory/middleware/ticket_janitor.py,
  fused-memory/src/fused_memory/escalators/folded_escalation.py (post-move path; S19 γ (S19/OQ-4 (b))),
  fused-memory/tests/test_curator_escalator.py, fused-memory/tests/middleware/test_dead_letter_escalator.py,
  fused-memory/tests/middleware/test_entity_mint_storm_escalator.py, fused-memory/tests/middleware/test_mem0_update_storm_escalator.py,
  fused-memory/tests/test_ticket_janitor.py, fused-memory/tests/test_folded_escalation.py.
- Deps: α, θ1b (θ2 converts the last caller of C9's transitional keyword and removes it); S19 γ (wave 3).
- Keys: fk-7d1f3a0d12ca (contributes), fk-333277fb4bd7 (contributes: 7 of the 12 fm queue constructors in `*/src` at `a85cb1ad78`).
- Does: each builds an `EscalationFiler.for_queue_dir`; the content-fingerprint family uses
  `Fingerprint(DedupeConfig.for_content_fingerprint(...))` (task 6348's adoption, cancelled into θ2);
  curator/janitor keep their `id_namespace` divergences (the 5 `make_id` keys); TicketJanitor's
  dedup-outage alarm and its probe-defect alarm file on a stable per-project `Sentinel` anchor with
  `OpenAtLevel(any_level=True)`, so a sustained outage and a janitor re-instantiated over the same queue
  directory leave one pending record (task 6689, cancelled into θ2). θ2 converts the ticket janitor's
  curation-backlog alarm (`asyncio.to_thread(file_folded_escalation, …)`, the twelfth C9 caller) to a
  `SentinelSubject` anchor, and with that last caller converted removes C9's transitional
  `anchor_task_id` keyword. 13 files, one over the band, because the leaf that converts the last caller
  removes the keyword it defines (and migrates `test_folded_escalation.py`, which passes it).
- Deletes: 7 `EscalationQueue(...)` constructions used only to file, 7 `Escalation(...)`; the janitor's
  curation-backlog string anchor; C9's transitional `anchor_task_id` keyword in
  `escalators/folded_escalation.py`.
- Signal: a dead-letter backlog in the fixture files one folded record across three triggers, through
  the filer (row 20's recorder sees `escalation.filing`); two consecutive dedup-outage windows and a
  restarted janitor yield one pending record.
- G7: INV-5, INV-4 (6689's storm on one root cause gets its escape).

**ι1 — Recon escalation registry: one declaration per recon alarm category**
(`task_kind=normal`, priority medium, est. ~900 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/recon_escalations.py (new), fused-memory/src/fused_memory/reconciliation/harness.py,
  fused-memory/src/fused_memory/recon_state/backlog_policy.py (post-move path; S19 β1), fused-memory/src/fused_memory/reconciliation/finding_task_escalation.py,
  fused-memory/src/fused_memory/reconciliation/orphaned_recon_escalation_sweep.py, fused-memory/src/fused_memory/reconciliation/task_count_snapshot_cadence.py,
  fused-memory/tests/reconciliation/test_recon_escalations.py (new), fused-memory/tests/test_harness.py,
  fused-memory/tests/reconciliation/test_finding_task_escalation.py, fused-memory/tests/test_backlog_policy.py.
- Deps: α; 9254 (S10 λ, end of recon `harness.py`), 9251 (S10 κ2, `orphaned_recon_escalation_sweep.py`); 5270 (`_escalate`'s and `_file_finding_task_escalation`'s offload, by readiness); S19 β2 (wave 3); S2 η (wave 3; `backlog_policy.py`).
- Keys: fk-b7b0be25c6b4.
- Does: D11; `_escalate` takes a `ReconEscalation`, stays synchronous and fire-and-forget, and files
  with `file_soon` on the `recon-<run>` prefix sentinel (D1: sync code on the loop thread);
  `_finding_recently_resolved` reads the derived fold set; the BacklogIterator's
  `recon_backlog_overflow` call passes the member; `_file_finding_task_escalation` and
  `BacklogPolicy`'s judge-halt record file through the filer; the recon filer drains at harness stop;
  `backlog_policy.py`'s 6348 site uses `Fingerprint(DedupeConfig.for_content_fingerprint(…))`.
- Deletes: `_RECON_DEDUP_CONFIG`, `_escalate`'s inline severity tuple, the "all four" comment, the
  recon harness's own `Escalation(...)` constructions; the prose pointers at `_RECON_DEDUP_CONFIG` in
  `orphaned_recon_escalation_sweep.py` (3) and `task_count_snapshot_cadence.py` (1) are repointed at C8.
- Signal: row 19 green; the existing storm test that mocked `_escalate` is replaced by one that files
  twice into a real recon queue and observes the fold (the run's evidence that the old test could not).
- G7: INV-5, INV-9, INV-8 (the recon queue write leaves the loop).

**ι2 — The other recon filers and the recon backfill script through the entry point; dedupe.file_or_fold_l1 deleted**
(`task_kind=normal`, priority medium, est. ~750 LOC)
- Files: fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py, fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py,
  fused-memory/src/fused_memory/reconciliation/index_drift_detector.py, fused-memory/src/fused_memory/reconciliation/targeted.py,
  fused-memory/src/fused_memory/reconciliation/preservation_specimen_guard.py, fused-memory/src/fused_memory/reconciliation/standing_decision_storm_escape.py,
  fused-memory/scripts/memory_eval_runner.py, fused-memory/scripts/backfill_recon_escalations.py,
  escalation/src/escalation/dedupe.py, fused-memory/tests/reconciliation/test_stage1.py,
  fused-memory/tests/test_standing_decision_storm_escape.py, fused-memory/tests/test_backfill_recon_escalations.py.
- Deps: ι1, β; 9285 (S10 ο, end of `stage1_stall_detector.py` and `targeted.py`); S19 β2 (wave 3).
- Keys: fk-7d1f3a0d12ca (contributes).
- Does: every recon filer files through a filer with a registry member or a policy; the two
  `file_or_fold_l1` callers use `Fingerprint`; `index_drift_detector.py`'s `graph:<group_id>` id becomes
  a `Sentinel` prefix subject and its REFACTOR TRIGGER docstring is retired (task 7070's question,
  answered by the typed subject); `targeted.py`'s reconciler-L1 filing is one offloaded helper since
  task 6234 (`TargetedReconciler._file_reconciler_l1`, its queue from an injected factory, journalled as
  `escalation/submit` rows), so ι2 converts that one helper and keeps the injected factory and the journal
  rows; `memory_eval_runner.py` uses `file_blocking` (its `eval_regression` filings are meant for a human
  and keep their `dedupe_fingerprint`, the dashboard's join key); `backfill_recon_escalations.py::apply_plan` stamps the canonical through a public queue mutator built
  on S3 C4's `_mutate_pending` (`attach_dedupe_child` per child, or a new public `stamp_dedupe_group`
  when the fingerprint cannot ride it; measured at dispatch), never `queue.submit(existing)`. The
  growth-sweep alarm in `stages/task_knowledge_sync.py` is not ι2's: S12 ε deletes it and files it
  through C8.
- Deletes: `dedupe.py::file_or_fold_l1` (and its `Escalation(...)`), 6 recon `Escalation(...)`, the
  script's one, the 6 `has_open_l1` dedup calls in `stage1_stall_detector.py`, the function-local
  `escalation.dedupe` imports guarded by `None` fallbacks, `apply_plan`'s resubmit.
- Signal: a stalled-gate fixture's L1 folds across Stage-1 cycles through the filer
  (`dedupe_count` rises; one pending record); `backfill_recon_escalations.py --apply` on a fixture queue
  writes no record through `submit` (row 20's recorder).
- G7: INV-11 (the `file_or_fold_l1 is None` silent no-op branches go: escalation is a declared
  dependency of fused-memory).

**κ1 — Escalation router, L0 promotion, zero-progress requeue, startup steps and process supervision file through the entry point**
(`task_kind=normal`, priority medium, est. ~550 LOC)
- Files: orchestrator/src/orchestrator/escalation_router.py, orchestrator/src/orchestrator/l0_promotion.py,
  orchestrator/src/orchestrator/zero_progress_requeue.py, orchestrator/src/orchestrator/startup_steps.py,
  orchestrator/src/orchestrator/proc_supervision.py, orchestrator/tests/test_zero_progress_requeue.py,
  orchestrator/tests/test_escalation_router.py, orchestrator/tests/test_startup_steps.py.
- Deps: α. (Task 8926, the draft's edge, is done: `_file_or_refresh_l1` already calls
  `escalation/file_or_refresh.py::file_or_refresh`.)
- Keys: fk-7d1f3a0d12ca (contributes).
- Does: drafts + policies; `startup_steps.py::_file_or_refresh_l1` moves from `file_or_refresh` to
  `FilingPolicy.refresh_open(…)` on the `orchestrator-startup-<step>` prefix sentinel (D13);
  `escalation_router.py::router_owns` keeps its eval-lane exclusion (half of Leo's "no automated filer"
  rule, D14); `proc_supervision.py`'s `_file_inprocess_escalation` keeps its computed role/severity, now
  through the downgrade gate (all its roles are sentinel roles; row 20).
- Deletes: 5 `Escalation(...)` (6 at `ba17934861`; 8926 moved `startup_steps.py`'s into the helper), 1
  `EscalationQueue(...)` used only to file, `_file_or_refresh_l1`'s `file_or_refresh` call.
- Signal: row 20's recorder over these modules' production triggers; a startup step failing on three
  consecutive boots leaves one pending L1 whose `dedupe_count` is 2.
- G7: INV-5.

**κ2 — Markup sink, flake-debt drain, offline-lane and merge-deferred recovery file through the entry point**
(`task_kind=normal`, priority medium, est. ~500 LOC)
- Files: orchestrator/src/orchestrator/mcp/markup_sink.py, orchestrator/src/orchestrator/flake_debt_drain.py,
  orchestrator/src/orchestrator/offline_lane_escalations.py, orchestrator/src/orchestrator/merge_deferred_recovery.py,
  orchestrator/tests/test_markup_sink_eval_lane.py, orchestrator/tests/test_markup_sink_lazy_imports.py (new),
  orchestrator/tests/test_flake_debt_drain.py.
- Deps: α, β; 7128 (pending, `markup_sink.py`).
- Keys: fk-333277fb4bd7 (contributes: the filing-time eval-lane re-implementation).
- Does: `make_escalation_sink`'s `file_record` files through the filer (containment is C2 step 3).
  The sink runs inside per-agent stdio MCP subprocesses (plan-tools, verdict-tools) and imports
  `escalation` lazily today (its channel hands it an `escalation_cls`); the filer import stays lazy, and
  the new `test_markup_sink_lazy_imports.py` pins it (task 5258 also names that file; κ2 creates it).
- Deletes: `markup_sink.py`'s own `eval_lane_provenance` check, its `EscalationQueue(...)` and the
  `escalation_cls` value it threads through the channel (two filings built through it); 4 more
  `Escalation(...)` in the other three modules.
- Signal: `test_markup_sink_eval_lane.py` passes with the containment observed on the filed record
  (resolved `close_only`) rather than as a skipped filing.

**κ3 — Exit contract, recovery emission, slot lease, review checkpoint, provenance conflict, flake recorder and lane lifecycle file through the entry point**
(`task_kind=normal`, priority medium, est. ~600 LOC)
- Files: orchestrator/src/orchestrator/exit_contract.py, orchestrator/src/orchestrator/recovery_emission.py,
  orchestrator/src/orchestrator/slot_lease.py, orchestrator/src/orchestrator/review_checkpoint.py,
  orchestrator/src/orchestrator/provenance_conflict.py, orchestrator/src/orchestrator/flake_recorder.py,
  orchestrator/src/orchestrator/lane_lifecycle.py, orchestrator/tests/test_exit_contract.py,
  orchestrator/tests/test_provenance_conflict_sink.py, orchestrator/tests/test_recovery_emission.py.
- Deps: α; 9284 (S9 ν), 9224 (S8 ε2b), 9216 (S8 α), 9150 (S4 μ0), 9283 (S10 ξ), 9262 (S5 κ). (The draft's
  κ3 → ζ1 rested only on the shared test file `test_recovery_emission.py` and is dropped under the
  lead's test-file rule; the file locks serialize the two.)
- Keys: fk-7d1f3a0d12ca (contributes).
- Does: drafts + policies; `review-<id>` becomes a `Sentinel` prefix subject; `provenance_conflict.py`'s
  6348 site uses `Fingerprint(DedupeConfig.for_content_fingerprint(…))`; `lane_lifecycle.py`'s inert
  born-at-L2 filing takes a filer (task 4746 wires it; §Live tasks).
- Deletes: 7 `Escalation(...)`.
- Signal: row 20's recorder over these modules' production triggers.

**κ4 — The steward's level-1 re-escalation through the entry point**
(`task_kind=normal`, priority medium, est. ~400 LOC)
- Files: orchestrator/src/orchestrator/steward.py, orchestrator/tests/test_steward.py,
  orchestrator/tests/test_steward_outcome.py, orchestrator/tests/test_steward_escalation_filing.py (new).
- Deps: α; 9190 (S9 ε), 9229 (S8 θ, end of the `steward.py` chain).
- Keys: fk-7d1f3a0d12ca (contributes).
- Does: `TaskSteward._auto_escalate_to_human` (the single choke point of every steward give-up; 8 call
  sites) becomes `async` and awaits `file()` with `TaskSubject`, `FilingPolicy.none(level=1)` and the
  steward's role (OQ-9); `_give_up_if_lifetime_budget_spent` becomes `async` and its callers await it;
  the L0 is dismissed after the L1 is written, as today; S9 ε's row 15c is reproduced as boundary row
  23. `_pre_triage_suggestions` builds its in-memory copy with `dataclasses.replace(escalation,
  detail=…, summary=…)` instead of `Escalation(**{...})`. Measured at `a85cb1ad78` (unchanged since `ba17934861`): of the 8 test files
  that mention `_auto_escalate_to_human`, only `test_steward.py` calls it (4 calls, which κ4 awaits); the
  other seven (`test_workflow_e2e.py`, `test_workflow_escalated_steward_stall.py`,
  `test_workflow_merge_gating_strand.py`, `test_workflow_merge_phase_block_status.py`,
  `test_workflow_state_machine_boundary.py`, `test_steward_outcome.py`,
  `scripts/tests/test_escalation_ladder.py`) cite it in prose and need no edit. Every production caller
  is already inside an `async def` (`_run_loop`, `_handle_escalation`) except
  `_give_up_if_lifetime_budget_spent`, which κ4 makes async.
- Deletes: the steward's two `Escalation(...)` constructions and its `escalation_queue.submit`.
- Signal: row 23 green; a steward give-up on the running orchestrator files one L1 that
  `get_pending_escalations` shows before the workflow's outcome is published.
- G7: INV-8 (the steward's filing leaves the loop), INV-3 (the L1 lands before the publish).

**λ1 — Eval cells write nothing to production stores: transcripts under the eval root, a factory that refuses production stores, and a zero-write boundary check**
(`task_kind=normal`, priority medium, est. ~450 LOC; Leo's S1-2 ruling, parts 1, 3 and 4)
- Files: orchestrator/src/orchestrator/evals/profile.py, orchestrator/src/orchestrator/evals/runner.py,
  orchestrator/tests/test_eval_boundary_suite.py, orchestrator/tests/test_eval_profile.py,
  orchestrator/tests/test_eval_cell_hermeticity.py (new).
- Deps: none (Leo: "depending on no other S1 task"; parts 1, 3 and 4 touch only the eval package and
  can land at once). It opens this plan's `evals/runner.py` chain: μ → λ1 and λ2 → μ (lead 2026-11-13,
  same-file rule). The filed same-file neighbours take no edge: 8499 (`build_eval_orch_config`'s routing
  pins) and 7765 (`_cell_metrics`) edit other functions of `evals/runner.py`; the file locks serialize them.
- Keys: none (rulings task; `source: review-all`, `x_finding_run: review-all-dark_factory-20261008`).
- Does: C11's profile and factory parts. `EVAL_PROFILE` gains `transcript_archive.root`, resolved by
  `resolve_eval_profile_update` under `evals/snapshots.py::eval_worktree_root(project_root)`, so the
  parity tripwire (B1) declares it; `_cell_workflow` raises `EvalHermeticityError` when the built
  workflow holds an escalation queue, a steward factory, an event store or a cost store; the eval
  boundary suite gains row B11 (boundary row 27, λ1's half): one recorded cell over a `tmp_path` project
  root leaves the production queue directory, `runs.db`, the transcript archive and the eval stub task
  store unchanged, and its transcripts land under the eval worktree root.
- Signal: boundary rows 26 and 27 (λ1's half) green; after a corpus cell runs on the host,
  `data/orchestrator/agent-transcripts/` gains no fixture-id directory and the cell's transcripts are
  under `<project>-eval-worktrees/` (172 eval transcripts under 4 fixture ids sat in the production
  archive at `07aaa5be80`).
- G7: INV-10 (the zero-write row is exercised against a cell that would write), INV-2 (the refusal names
  the field), INV-12 (no allow-list: the row names production stores, it exempts nothing).

**λ2 — Eval verify logs under the eval root, and the design invariant that an eval run writes no production state**
(`task_kind=normal`, priority medium, est. ~400 LOC; Leo's S1-2 ruling, parts 2 and 5)
- Files: orchestrator/src/orchestrator/workflow.py, orchestrator/src/orchestrator/evals/runner.py,
  orchestrator/tests/test_eval_boundary_suite.py, orchestrator/tests/test_workflow_verify_log_root.py (new),
  docs/legibility/design-invariants.md, docs/legibility/design-invariants-fixtures.md,
  skills/prd/references/gates.md.
- Deps: λ1 (it widens λ1's B11 row and passes the seam from λ1's factory); μ (`evals/runner.py`, the
  last of this plan's three tasks on that file; lead 2026-11-13, same-file rule; μ waits on no live task,
  so this adds no wait); 9232 (S8 κ1, the end of the wave-2 `workflow.py` chain: part 2 "goes at the end
  of that file's per-file order"). S1 δ follows λ2
  on `workflow.py` (δ → λ2), so λ2 waits on no door leaf. The invariant files: the lead wires the
  wave-3 invariant-file chain at decompose (INV-18 to INV-23 in landing order, S17 ν last); S1's ω → λ2.
- Keys: none (rulings task).
- Does: C11's workflow part: `build_workflow(…, verify_log_root=None)` with today's
  `project_root / 'data' / 'verify-logs'` as the default, and `_cell_workflow` passing a root under
  `eval_worktree_root(project_root)`; B11 widened to the verify-log directory (and any other path under
  `data/verify-logs` the eval path writes, measured by the row); the four-site lockstep edit minting
  `eval-writes-no-production-state` (§Design invariants), numbered by landing order among INV-18 to
  INV-22 (S17 ν takes INV-23); its checks anchor on the slug.
- Signal: row 27 green in full; `scripts/tests/test_design_invariants_consistency.py` passes with the
  slug in every pinned site; after a corpus cell runs on the host, `data/verify-logs/` gains no fixture
  directory (3 empty fixture directories sat there at `07aaa5be80`).
- G7: INV-9 (the root has one home per run, set at construction), INV-10.

**μ — Run corpus eval cells under the production sandbox, give their setup the grants it needs, and mark the new baseline**
(`task_kind=normal`, priority medium, est. ~350 LOC; Leo's residual R-7 = A, 2026-11-13)
- Files: orchestrator/src/orchestrator/evals/runner.py, orchestrator/src/orchestrator/evals/shadow_invocation.py,
  orchestrator/src/orchestrator/evals/metrics.py, orchestrator/src/orchestrator/agents/write_set.py (only
  if the measurement below needs a grant), orchestrator/tests/test_eval_profile.py,
  orchestrator/tests/test_eval_shadow_invocation.py, orchestrator/tests/test_eval_collect_metrics.py,
  orchestrator/tests/test_eval_corpus_sandbox.py (new).
- Deps: λ1 (`evals/runner.py`: the overlay's same-file rule chains this plan's three tasks on that file,
  lead 2026-11-13; λ1 waits on no live task, so μ stays dispatchable as soon as λ1 lands, which keeps
  Leo's "independent of the other S1 tasks" as far as the same-file rule allows); λ2 follows μ. The filed
  same-file neighbours take no edge: 8499 (other keys of the same update dict) and 6748 (an import line,
  rebase-only under S19/Q-C6) are serialized by the file locks.
- Keys: none (rulings task).
- Does: C11's sandbox part. `build_eval_orch_config` drops its `'sandbox': SandboxConfig(enabled=False)`
  override, so a corpus cell inherits `base_config.sandbox`, the same backend and grants as production;
  `build_shadow_orch_config`'s own sandbox override goes with it (shadow cells keep running sandboxed,
  now by inheritance). The grants are production's: the only sandboxed processes are agent invocations
  (`agents/sandbox_dispatch.py::wrap_command`), and their writable set is derived in one place,
  `agents/write_set.py::compute_write_set`: the cell's worktree, its `.task-meta/<name>/` and task root,
  the main `.git`'s `objects/`, `refs/heads/task/`, `logs/refs/heads/task/` and `worktrees/<name>/`,
  `~/.cache/uv/`, `~/.claude/fleet/`, `/tmp/` and `/dev/` (read at `4a87282ee7`). The worktree's
  `setup_commands` (`evals/snapshots.py::create_eval_worktree`, the `uv sync`) run in the eval process,
  not in an agent, as production's worktree preparation does. The task measures which eval agent writes
  (environment syncs, cache fills, anything a cell's verify writes from inside an agent) fall outside
  that set: one corpus cell is run on the host under the configured backend, every sandbox refusal is
  recorded, and each needed write gets an entry in `compute_write_set` (so production and eval keep one
  grant list), never an eval-only switch-off; a write that only an eval needs is refused and fixed in the
  eval instead. It resolves the
  backend the way shadow cells do today (before S8 κ1 9232 the backend is a process global the harness
  sets; an eval process has no harness), and the new test pins it. `EvalMetrics` gains
  `sandbox_backend: str | None` (`None` on every result recorded before μ), so a reader can tell the two
  populations apart. The task records, in its description and commit, that corpus results from before
  the change are not comparable with later ones and that the next campaign needs a fresh baseline.
  The separate eval clone (residual option C) is not part of it.
- Signal: row 28 green; `test_eval_profile.py`'s `assert result.sandbox.enabled is False` becomes
  equality with the base config's sandbox; a corpus cell run on the host after landing completes with its
  agents under the configured backend, records `sandbox_backend`, and logs no sandbox refusal of a
  setup step.
- G7: INV-11 (a refused write is recorded, never swallowed), INV-9.

**ω — Integration gate: invariant escalation-filing-one-door, its guard, and the level-1 and dedupe helpers deleted**
(`task_kind=normal`, priority **high**, est. ~700 LOC)
- Files: escalation/tests/test_filing_one_door.py (new), escalation/tests/fixtures/filing_one_door/seeded_violation.py (new),
  escalation/src/escalation/queue.py, escalation/src/escalation/dedupe.py, escalation/tests/test_queue.py,
  escalation/tests/test_dedupe.py, escalation/tests/test_category_census.py, escalation/tests/test_category_contract.py,
  orchestrator/tests/test_cross_task_dedupe_reengagement.py, docs/legibility/design-invariants.md,
  docs/legibility/design-invariants-fixtures.md, skills/prd/references/gates.md, ARCHITECTURE.md.
- Deps: β, γ, δ, ε3, η2, η3, θ1b, θ2, ι2, κ1, κ2, κ3, κ4 (and through them every door leaf); λ2 (the
  three invariant files; the lead wires the other wave-3 invariant leaves at decompose, INV-18 to INV-23
  in landing order with S17 ν last, and no edge onto 9154 is wired, lead S2/OQ-10 (b)); 9210 (S3 μ, `ARCHITECTURE.md`); S12 ε (wave 3; deletes the last recon
  construction outside S1's leaves).
- Keys: fk-7d1f3a0d12ca, fk-333277fb4bd7 (close).
- Does: the four-site lockstep edit; rows 20–22; deletes `EscalationQueue.has_open_l1` (every caller
  converted by the leaves above; the four escalation test files and the one orchestrator test that
  exercise it or `submit_or_dedupe` directly move to `has_open`/the filer) and the public
  `dedupe.submit_or_dedupe`/`submit_or_dedupe_off_loop` (filing-private); ARCHITECTURE.md's escalation
  section names the one door.
  13 files, one over the band, because the residual test migrations are the gate's own (measured: 5).
- Signal: `scripts/tests/test_design_invariants_consistency.py` passes with the slug in every pinned
  site; the guard flags its seeded fixtures and passes the tree; rows 1–25 green in one run.
- G7: INV-10 (every half exercised against fixtures), INV-12 (derived exclusions; no allow-list).

## Live tasks (contract §8 step 2; 2026-11-11, re-run by the critic fold and at the ruling fold 2026-11-13)

| Task | Status | Relation | Action |
|---|---|---|---|
| 6609 (`fk-14e1510916ab`, Phase-5) | pending, low | subsumed by α (the predicate moves to `models.py` as α's downgrade gate needs) | cancel with `x_acceptance_reason` naming α; α carries the key |
| 3887 (pure-gate blocked write after recon disposed the task) | pending, low | subsumed by γ (`file_and_block`'s re-read and moot-resolve, row 13) | cancel naming γ |
| 8355 (one born-at-L2 filer for the deterministic runner in a lower module) | pending, low | subsumed by γ (`status_write.py::file_and_block` is that filer; it imports nothing from the runner, so the cycle 8355 describes is gone); its predicate-path split is S17 ξ1's | cancel naming γ; S17 ξ1's "Task 8355" precondition becomes γ (xref S17) |
| 6942 (route the scheduler's fm-read-storm and backfill-overstay L1 filers through `file_sentinel_l1`) | pending, low | subsumed by ζ1 (both filers become drafts + `FilingPolicy.open_l1()`; `sentinel_l1.py` is deleted) | cancel naming ζ1 |
| 6348 (adopt `DedupeConfig.for_content_fingerprint` at the hand-built sites) | pending, low; depends on 8528 | subsumed by θ2 (the five escalators), with δ, κ3, ε1 and ι1 adopting `Fingerprint(DedupeConfig.for_content_fingerprint(…))` at their sites | cancel naming θ2 (S2 κ's dependency on 6348 goes with it; xref S2) |
| 7070 (first-class dedup key instead of a third synthetic task id) | pending, low; depends on 9177 | subsumed by α (the typed `Subject` and the `Sentinel` prefix forms are the first-class key; `graph:<group_id>` and `inv12-finding:<key>` become registered prefixes; ι2 retires `index_drift_detector.py`'s REFACTOR TRIGGER) | cancel naming α |
| 6689 (TicketJanitor's dedup-outage escalation re-files every window and restart) | pending, medium | subsumed by θ2 (`OpenAtLevel(any_level=True)` on a stable per-project anchor) | cancel naming θ2 |
| 8926 (one file-or-refresh helper for `_file_or_refresh_l1` and the dirty-tree filer) | **done** 2026-11-12 (`ed4ceb9480`; the helper landed as `escalation/src/escalation/file_or_refresh.py::file_or_refresh`, not the `orchestrator/escalation_refresh.py` the draft's amendment proposed) | its rules are D13's `RefreshOpen`'s; S8 β 9218 will file through it | no action: the draft's amend and its edges κ1 → 8926, η1 → 8926 are withdrawn (satisfied); κ1, η1 and η2 move its three callers onto `RefreshOpen` and η2 deletes the module and `escalation/tests/test_file_or_refresh.py` |
| 5270 (offload the remaining inline blocking calls in fused-memory, incl. recon `_escalate`'s archive walk) | pending, medium; its one dependency, 4484, is done | same recon `harness.py` and `server/tools.py` as ι1 and θ1a; complementary (5270 offloads reads and the archive walk; S1 moves the queue writes) | edge ι1 → 5270 and θ1a → 5270 (by readiness: 5270 is dispatchable now) |
| 8493 (`verify_base_detector.py`'s `escalation_created` attribution) | pending, low | same file as ε2a | edge ε2a → 8493 |
| 7300 (pin the external producer's whole `escalation submit` call) | pending, low | β reroutes `submit.py` | edge β → 7300 |
| 8665 (Protocol-type the lane's `escalation_queue`) | pending, low; depends on 8528 | after ε1/ε2a/ε2b the lane keeps the raw queue only for `resolve` (`liveness.py`) and reads; the Protocol is still worth having over that narrower surface | amend 8665 to that surface; edge 8665 → ε2b |
| 8826 (harness born-at-L2 filers onto `born_at_l2.py`) | pending, low | upstream (S8 ζ1b depends on it); η1 then deletes `born_at_l2.py` | keep; no edge (implied through S8) |
| 7170 (block+L1 copies onto `_block_and_file_dispatch_gate_l1`) | pending, low | upstream (S8 ζ1a depends on it) | keep |
| 7718 (starvation cap L1 detail drift) | pending, low | same `scheduler.py` filing as ζ1; content fix | edge ζ1 → 7718 |
| 5258 (unify the three markup filers; push ceiling) | pending, low | overlaps β, θ1a, κ2 on `server.py`, `markup_tripwire.py`, `markup_sink.py` | amend: its record unification builds `EscalationDraft`s filed through C1; edge 5258 → β, θ1a, κ2 [lead] |
| 4809 (markup guard residue preservation) | pending, medium | same files as θ1a | edge θ1a → 4809 |
| 7128 (markup sink loss durability) | pending, low | same file as κ2 | edge κ2 → 7128 |
| 4746 (wire a queue into `LaneLifecycle`) | pending, low | κ3 makes `lane_lifecycle.py` take a filer | note on 4746: wire an `EscalationFiler`, not a queue |
| 9176, 9171, 9186 (S6 θ, δ, φ) | pending (wave 2) | their contracts C7, C3, C2 file on the raw queue; ε3 absorbs them (C10, D15) | note on each, starting "No change to this task's work:", naming the C10 shape ε3 applies later: `HaltState(…, escalations: EscalationPort)` with a provisional holder id re-keyed on the filed id, and `Filed`/`FilingFailed` renamed `HolderFiled`/`HolderFilingFailed` (9176); `MainHealthAutoHeal(…, escalations=)` with a folded policy (9171); `LifecycleRefusalResponder(…, escalations=)` (9186). S6 builds as written; ε3 converts |
| 9188, 9200 (S3 ε1, ζ) | pending (wave 2) | OQ-3 (lead) | edge 9188 → β; amend both, and the S3 PRD's C7a text |
| 9187 (S10 β1) | pending (wave 2), high | `server.py` R54 order (OQ-6, lead) | edge 9187 → β |
| 5897 (completion-claim escalation L0 vs L1) | pending, low | θ1a preserves today's level 0; 5897 decides | xref only |
| 4603 (deferred; older copy of 5258's first half) | deferred | not S1's | xref only |
| 4567 (seven in-process archive-inclusive `get_by_task` calls on the loop) | **done** 2026-11-12 (`b255f96b3e`) | complementary (reads vs S1's filing writes) | no action |
| 4757, 2471, 3096, 5386 (eval MCP isolation, eval memory sink, eval-lane containment, shadow sandbox plumbing) | done | the hermeticity λ1/λ2 build on; step 2 found no live task carrying λ1, λ2 or μ (queries: "Make eval cells hermetic…", "Run corpus eval cells under the production sandbox…", hits ≥ 0.5 all done) | no action |
| 8499, 6748, 7765 (eval runner: routing pins, the eval CLI move, `_cell_metrics` shape) | pending, low | same `evals/runner.py` as λ1, λ2 and μ, other functions or an import line | no edge (Leo ruled λ1 and μ independent; the file locks serialize them) |
| 9232 (S8 κ1, the invocation carries its sandbox backend) | pending (wave 2), low | end of the wave-2 `workflow.py` chain; it threads the sandbox backend μ inherits | edge λ2 → 9232 (the ruling's per-file place for λ2's `workflow.py` part) |

## Out of scope

- The read and mutate tools' own `to_thread` hops (`get_pending_escalations`, `amend_escalation`,
  `promote_to_l2`'s combined read): they are reads/mutations, not filings; S3's `read_tools.py` and
  `resolve_tools.py` own them.
- Rewriting existing records beyond `RefreshOpen` (S3 C4 `_mutate_pending`/`_mutate_terminal`).
- Typing `severity`/`status`/`category` vocabularies beyond what `EscalationCategory` already does (S2).
- `memory_service.py`'s four offloaded emitters (S26's file; correct under C1).
- `DeterministicRunner.run`'s predicate/deploy/gate split (S17, R16); γ only replaces its status-write arms
  and its filers.
- Unifying the three markup residue/storm record shapes (task 5258, amended).
- The recon growth-sweep alarm's evaluation (S12 ε).
- Retiring `dedupe_fingerprint`: S1 keeps writing it; its retirement is two tasks of the typed-records
  plan (S2): S2 γ2 moves the readers after S2's failure-signature task, and S2 γ3 deletes the field
  after S1 ω (§Rulings, lead 2026-11-13).
- A separate eval clone instead of a linked worktree (Leo, R-7: not filed); the daily memory-eval
  runner's `eval_regression` escalations, which are meant to reach a human.

## Rulings (sitting of 2026-11-12)

Leo answered on 2026-11-12 and the residual questions on 2026-11-13; lead rulings are in the program's
`digests/wave3-lead-rulings.md` (later blocks win). No question is open.

- **OQ-1** — Option 1: three entry points chosen by where the caller runs (await; fire-and-forget on
  the loop; blocking off the loop) and a typed subject on the record; accepted consequence: three
  synchronous on-loop alarm callers become fire-and-forget and can lose a record on a crash (D1).
  Ruled by Leo. A clarification of R8's wording ("file(record, policy)").
- **OQ-2** — Option 1: callers build a frozen draft and only `filing.py` constructs the record; a parity
  test pins the field lists. Ruled by the lead.
- **OQ-3** — Option 1: α and β land before the server split; the lead amends 9188, 9200 and the S3 PRD's
  C7a. Ruled by the lead.
- **OQ-4** — Follows Leo's S2-3 (a): S2's vocabulary, cascade-parent and callback leaves, then the door
  with the `Sentinel` registry, then S2's leaves that change what the door writes. Ruled by the lead
  (duplicate of S2/OQ-7, which Leo ruled).
- **OQ-5** — Option 1: a third static check for resubmitted records; `submit` stays public for the test
  fixtures. Ruled by the lead.
- **OQ-6** — Option 1: α and β sit right after 9181 and 9187 waits on β. Ruled by the lead (flagged to
  Leo with S2-3).
- **OQ-7** — Option 1: the named promotion policy `FilingPolicy.promoted_l2`. Ruled by the lead.
- **OQ-8** — Option 2: eval-task filings contained at levels 1 and 2 through the policy's two-value eval
  scope; the agent tools contain every level; the door does not contain an in-process level-0 record,
  reworded as a backstop for the future (D14, row 4b). PLUS Leo's hermeticity rule: nothing an eval files
  reaches the task store through any automated filer, nor leaves the eval in any other way. Applied as δ's
  "no filer" sentence and row 14b, the kept router and reaper exclusions (κ1, η1), and two new tasks of
  this plan, λ1 and λ2 (ruled by Leo 2026-11-13: the eval-hermeticity task is carried here; it is split in
  two because Leo's ruling puts its `workflow.py` part at the end of that file's order while its eval-
  package parts "can land at once"). Ruled by Leo.
- **OQ-9** — Option 1: synchronous callers that need the write first become async and await it. Ruled
  by the lead.
- **OQ-10** — Option 1: the port's second argument is `policy`. Ruled by the lead. Amends R8's spelling
  `file(record, dedupe)`.
- **R-7 (residual, S1-2)** — A: corpus eval cells run under the production sandbox, same backend and
  grants; a fresh baseline for the next campaign; the separate eval clone is not filed, and the shared
  `.git` stays a known residual. Applied as task μ. Ruled by Leo 2026-11-13.
- **Cross-stream, applied** — caller identity (S2-2 B widened by R-6 B): the door writes S2 ε's
  `filed_via` on every record, S2 ε → β, row 10b sends the watcher's header (Leo); order S2 → door → S2
  (S2-3 (a), Leo); test-file-only edges are not wired, so κ3 → ζ1 is dropped (lead); the dead-letter
  escalator's path is S19's `services/dead_letter_escalator.py` (lead); no wave-3 invariant leaf waits on
  9154 for a number (lead S2/OQ-10 (b)); the S12 row names task 9490's cycle refusal (Leo, S2-3);
  the decision-partition invariant `one-decision-partition-per-area` (INV-23, S17 ν) is checked against
  C2, where every step is code's (Leo, S17-1); the six wave-3 minting tasks take INV-18 to INV-23 in
  landing order, S17 ν last (lead).
- **Fingerprint** — this plan keeps writing `dedupe_fingerprint`, because five readers that neither
  plan's door work moves still read it and S1's own open-at-level key rests on it. Its retirement is
  carried by the typed-records plan (S2) as two tasks, for size: S2 γ2 moves those readers onto the
  typed failure signature (after S2's failure-signature task; it changes no write), and S2 γ3 deletes
  the field (after γ2 and S1 ω) (§Cross-PRD, "one failure identity"). Ruled by the lead 2026-11-13;
  amends lead S2/OQ-3 (a)'s clause "S1 retires new fingerprint writes", and the lead's later
  "one task" wording.
- **Eval hermeticity split and the eval runner's file order** — the two-task split (λ1 for the eval
  package at once; λ2 for the `workflow.py` part and the invariant entry at the end of that file's order)
  is confirmed. Tasks of this plan that edit the same non-test file are chained by a real edge (the
  overlay's same-file rule; only test-file-only edges are dropped): λ1, μ and λ2 all edit
  `evals/runner.py`, so μ → λ1 and λ2 → μ (λ2 → λ1 stands), the order that holds fewer tasks: μ waits
  behind λ1 alone (no live task upstream), and λ2 already waits behind 256 live tasks through 9232, so μ
  ahead of it adds none; the other order would put μ behind those 256, 5040 and 8528 among them. Ruled
  by the lead 2026-11-13 (the order picked by the fold-verifier, as the ruling asks, then confirmed by
  the lead: "the same-file order on `evals/runner.py` as the fold-verifier read it stands").
- **S17's two new scheduler filers** — both are born at level 2 through this plan's entry point (C5):
  S17 δ2 files `dependency_capability` with `FilingPolicy.born_at_l2()`, and S17 λ files `infra_issue`
  at severity `critical` with `FilingPolicy.born_at_l2()` on the `Sentinel` member
  `__delivered_check_phase_slow__`, which λ adds to this plan's registry. S17 δ2 → ζ2 and S17 λ → δ2, so
  neither lands before the door and ζ1 converts nothing of S17's (§Cross-PRD S17 row). Ruled by the lead
  2026-11-13 (S17/OQ-9 = A: no self-resolving level-1 is filed).
- **Freshness (ruling fold, `ba17934861..a85cb1ad78`, 1,210 commits)** — 47 named paths changed. What
  moved the plan: task 8926 landed `escalation/file_or_refresh.py` (D13, κ1, η1, η2; its amend and edges
  withdrawn); task 6390 added `Escalation.review_due` (D4, C1, γ); task 6234 made `targeted.py`'s
  reconciler-L1 filing one offloaded helper over an injected queue factory (ι2); tasks 4567, 4867, 7720,
  8525 and 4484 are done (pre-conditions, §Cross-PRD, §Live tasks). Re-measured and unchanged: the 24
  swallow arms, the 17 workflow constructions and their 6 silent filings, the 16 lane filings, the 3
  scheduler sites, the 10 recon `_escalate` calls over 8 categories, the 12 folded-escalation callers,
  the 11 `to_thread` hops in `server.py` and the 41 `has_open_l1` calls.

## Tactical choices (decided in the leaf)

1. **Sentinel member names.** Upper-snake of the literal (default) or grouped by owning area. Decide in α.
2. **`subject_lock` granularity.** Per task id (default) or per (task id, category). Decide in α from
   the row-7 measurement.
3. **Where the moved measurements land.** Commit messages (default). Decide per leaf.
4. **`drain` timeout at harness shutdown.** Default 5 s; η2 measures the slowest filing on the
   live queue first.
5. **`HolderFiled` vs collapsing S6's values into C1's outcome.** Rename (default, done by ε3; the
   lead's note on 9176 names it); the lead may collapse instead.
6. **How `transcript_archive.root` resolves under the eval root.** A value `resolve_eval_profile_update`
   resolves against `eval_worktree_root(project_root)` (default) or a per-run override beside it; either
   way the parity tripwire must list it. Decide in λ1.
