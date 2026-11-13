# Dispatch-time riders: held low follow-ups offered to the host task that already holds their locks

**Status:** active PRD, authored 2026-11-04 (`/prd` under `/team`, author seat of the task-shape
PRD 3 session, from Leo's rulings of 2026-11-03/04 on
`plans/portfolio-review-2026-11-01/followups/task-shape.md` option C and the lead's decisions,
all recorded in §3; critic pass `task-shape-seats/PRD3-C.md` folded in, see the closing section).
B+H: three packages touched, fourteen mechanisms, two load-bearing seams (the plan boundary's scope
reconcile and the task-status chokepoint in the interceptor).
**Code anchors** cite symbols; verified against main `94d620f988` (2026-11-04). Main moves fast;
re-locate at implementation time.
**Evidence:** `plans/portfolio-review-2026-11-01/followups/task-shape.md` (§5, §7 option C) and
the design scout `task-shape-seats/PRD3-R.md` (substrate §1, measured limit §2, offers per host
§2.2, host class mix §2.3, observation series §5, interactions §6, risks §7); critic `PRD3-C.md`.
**Sibling PRDs:** `plans/follow-up-filing-shape-prd.md` (PRD 1, filed as 8413–8421) produces the
records this PRD consumes; `plans/followup-packing-sweep-prd.md` (PRD 2, option B) owns the
14-day rider reservation and is the drain this PRD yields to (§5).

## 1. Goal

A held low follow-up whose declared files all sit inside the lock-key set a medium-or-higher host
task has just planned is offered to that host's architect, after the plan exists, as a rider: the
architect accepts it as one or more rider-tagged plan steps or declines it with a typed reason; an
accepted rider is carried by the host (holder edge kept, a dependency edge on the host,
`metadata.rider_of`) and is closed `done` with the host's landing sha, inside the host's own done
transition, when the host lands with every rider-tagged step done; a rider whose steps were not
done is released back to the held pool before the host lands; a cancelled host releases its riders
inside its cancel transition; a carried rider whose host has not landed in 14 days is released so
no parked host can hold it out of both drains. Nothing about the host's lock set, `metadata.files`
or the implementer's scope boundary changes, because a rider is only ever offered when it needs no
lock the host does not already hold. The measured premise (task-shape §7 C, PRD3-R §2.1): 29%
(df) and 36% (reify) of held follow-ups with files are a key-subset of some pending medium+ host;
on the memo's flow proxy (df 230, reify 150 low follow-ups a month) that is about **df 66 and
reify 55 avoided landings a month** before architect declines, each worth the host-local verify
minutes of a full landing (df 19 to 56, reify 86 to 110 minutes), against a one-key-and-test-only
limit that would have left 4 to 5% of the pool rideable and about 9 landings a month.

### What an operator observes when this lands

- In `riders.mode: observe`, every df dispatch whose plan-scope reconcile succeeded writes
  `rider_offered{host, rider, rank, mode: observe}` rows to runs.db for up to five held tasks, and
  each offered held task's `get_task` shows `metadata.rider_ledger.offers` incremented with
  `last_host` and `last_offered_at`; the architect prompt is unchanged.
- In `riders.mode: shadow`, a sampled fraction (`riders.shadow_sample`, default 0.2) of
  offer-bearing dispatches runs the rider pass and applies nothing: `rider_pass{mode: shadow,
  cost_usd, duration_ms, turns}` and `rider_verdict{mode: shadow}` rows exist, `plan.json` is
  restored, no task changes; the report prints the measured pass cost and accept rate before
  anything is carried.
- In `riders.mode: offer`, a host dispatched with one eligible rider runs one extra architect
  continuation after `confirm_plan`; `plan.json` gains a `_riders` list with one typed verdict per
  offer; an accepted rider's `get_task` shows `dependencies: [<holder>, <host>]`,
  `metadata.rider_of == host` and `rider_ledger.carried_at`; the host's `followup_items` contains
  the rider's items and its `details` ends with a generated `Riders:` block; `rider_verdict{verdict:
  accept}` and `rider_carried` events exist for the pair.
- When that host goes `done` with a merged-family provenance, by the lane, by a train, by its
  steward or by a human's hand carry, the same `set_task_status` call closes the rider:
  `get_task(rider)` shows `status: done`, `done_provenance {kind: merged, commit: <host landing
  sha>}`, `followup_hold.promoted.by == 'rider'`, and a `rider_closed` event names the host; the
  rider never had a verify, a worktree or a merge request of its own, and no scheduler tick ever
  saw it ready.
- When the implementer leaves a rider-tagged step undone, the rider is back in the held pool
  before the host's merge request is submitted: `rider_of` absent, the host edge gone, the holder
  edge intact, `rider_released{reason: steps_undone}` emitted, and the plan no longer carries the
  undone step.
- An offer the architect never answered is `rider_verdict{verdict: decline, reason: no_verdict}`
  in runs.db and `rider_ledger.last_decline_reason == 'no_verdict'` on the held task; silence is
  never the record.
- `set_task_status(host, 'cancelled')` leaves every carried rider pending and held with
  `rider_released{reason: host_cancelled}` and no L1 from the cancelled-descendant sweep.
- A carried rider whose carry is 14 days old while its host is not `in-progress` (pending after a
  requeue, blocked, deferred) shows `rider_released{reason: host_stalled}` and is offered again at
  the next matching dispatch; a host still executing, re-verifying or waiting in the merge queue
  keeps its riders whatever their age.
- A rider that cannot be closed when its host goes `done` (a human re-statused it meanwhile, its
  own provenance write is refused) is released in that same transaction with
  `rider_close_failed{rider, error}`; the host's `done` lands regardless and no escalation is
  filed about a rider.
- A `scope_violation` escalation whose task is a rider host is counted by the observation report
  as a defect (expected zero under the subset rule).
- Fourteen days after the integration gate lands, one dated report under
  `plans/portfolio-review-2026-11-01/followups/` gives the seven series of PRD3-R §5 per project,
  the rider-pass cost (turns, USD) per dispatch from shadow and offer rows, and a `KILL` line when
  matched rider hosts pay more than +0.5 verifies per landing at n ≥ 30, each series
  distinguishing "no producer has written" from zero.

### Consumers (G1)

| mechanism | consumer |
|---|---|
| `select_rider_offers` pure selector and offer-side `offerable` predicate (α) | the workflow's plan boundary (`TaskWorkflow._plan` after `_reconcile_plan_scope`); δ |
| `riders.*` config section, green-tier reloadable (α, β) | the selector and the pass gate at every dispatch; the operator through `reload_config` (the `KILL` line is applied by flipping `riders.mode`) |
| `rider_offered` / `rider_pass` / `rider_verdict` / `rider_carried` / `rider_released` / `rider_closed` / `rider_reconciled` events (α, β, γ) | the observation report (δ); the L2 watcher; `scripts/` readers of runs.db |
| `metadata.rider_ledger` registered submodel on the held task (α; `carried_at`, `deliverable_at`, `deferred_promotion` fields β, γ) | PRD 2's `followup_packing.exclusion` (reads it when present, companion correction §5); the selector's ranking (fewest declines); the interceptor's close decision (`deliverable_at`); δ |
| `build_rider_offer_prompt` and the rider pass in `shadow` and `offer` modes (β) | the architect at every sampled or offer-bearing dispatch |
| plan-tools `accept_rider`, `decline_rider`, `add_plan_step(rider_id=)`, typed `RiderVerdict` in `plan._riders` (β) | the architect; the harness, which reads the verdicts and applies them; the implementer briefing, which renders rider-tagged steps last; the carry reconciler (γ) |
| `apply_rider_verdicts` fused-memory verb and `metadata.rider_of` (β) | the harness after the pass; the selector (a carried rider is never offered); PRD 2's packer (`PackExclusion.RIDER`); cancel-with-children, which enumerates the rider through its host edge |
| `Riders:` block in the host's `details` (β) | the D2 context-complete reviewer and the completion judge, whose only channel is the task text |
| `stamp_rider_deliverability` verb and `_reconcile_rider_carry` (γ) | the interceptor's done transition (reads `deliverable_at`); the plan (undone rider steps removed) |
| close and release inside `TaskInterceptor._apply_status_transition`, the `release_rider` verb, the periodic `reconcile_carries` sweep (γ) | every writer of a host's `done` or `cancelled`: the lane, trains, the steward, `/merge-queue` hand carries, `/unblock`; the fused-memory server at startup and on its sweep tick; the operator (`release_rider` is the manual unfold) |
| the rider promotion guard registered through PRD 1 ζ's `followup_hold.register_promotion_guard` (γ; the generic hook is 8418's companion correction) | `promote` in `followup_hold.py` (consults it before removing a holder edge); `release_rider` (applies the deferred cause); the close path (ignores it); the race in §3 decision 11 |
| integration gate (γ) and observation report (δ) | the operator; δ is the INV-13 proof that every producer above wrote in production and the only input to the `shadow` → `offer` flip and the kill decision |

## 2. Background (substrate as it is)

- **Dispatch path.** `orchestrator/src/orchestrator/scheduler.py::Scheduler._get_modules` gives
  a task its lock keys from `metadata.files` through
  `orchestrator/src/orchestrator/module_charter.py::derive_modules` (directory strip, then
  `shared/src/shared/locking.py::normalize_lock`) at the project's `lock_depth` (df 12, reify
  10), with a `task-<id>` fallback lock and `[]` for deterministic tasks. At depth 12 a key is in
  effect a file path. Nothing computes host ∩ held-pool today (PRD3-R §1.1). Task ids are
  INTEGER in the store (`scripts/tasks_db_schema.py`); every id in this PRD's records is `int`.
- **Plan boundary.** `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._plan` invokes the
  architect (`::TaskWorkflow._run_architect_for_plan`), stamps the plan
  (`_stamp_architect_plan`), reads `plan.files`, derives modules, and calls
  `::TaskWorkflow._reconcile_plan_scope` → `::TaskWorkflow._reconcile_scope_locks`, the
  scope-reconciliation choke point (task 2505). That derives the module set from the plan files
  and, when it differs from `self.modules`, asks
  `Scheduler.handle_blast_radius_expansion(task_id, current, needed, persist_files=plan_files)`
  (`lock_table.try_acquire_additional`, `release_subset`, and `_persist_files_metadata`, the
  one plan-time writer of `metadata.files`); on success `self.modules` is updated; on a lock
  conflict the task is requeued through `_requeue_on_lock_conflict`. When the module set is
  unchanged the call is a no-op and persists nothing. After success `_plan` writes decisions to
  memory and returns `PLANNED`. **The post-reconcile `self.modules` is the plan's lock-key set and
  the only key set this PRD ever compares against** (PRD3-R §1.3).
  `orchestrator/src/orchestrator/artifacts.py::TaskArtifacts.lock_plan` is an `O_CREAT|O_EXCL`
  ownership marker read by `TaskGroundTruth`; `mcp/plan_tools.py` never consults it, so a
  plan-tools write after it is not refused.
- **Three architect plan-writing invocations exist**, and the rider records must survive all
  three: the fresh plan in `_plan`; `::TaskWorkflow._replan(reviews)`, the review-phase re-plan
  that rewrites steps after a review; and the merge-phase narrowing pass
  `::TaskWorkflow._run_plan_tightening` (one-shot flag, plan.json snapshot restored on failure or
  decline, post-pass subset check refusing new files), briefed by
  `orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_plan_tightening_prompt`.
  `mcp/plan_tools.py::_create_plan` rewrites plan.json wholesale and carries forward only the
  keys in `plan_descope.DESCOPE_PLAN_KEYS`; `update_plan_metadata` is a whole-plan
  read-modify-write that keeps unknown keys. The rider pass copies the narrowing pass's shape at
  the plan boundary; `_replan` is the closer cost analogue for a plan-edit continuation.
- **C-A1 anti-anchoring.** `::BriefingAssembler.build_architect_prompt` calls
  `_task_preamble(task, 'architect', context, include_files=False)`: the architect is deliberately
  not shown the host's `metadata.files`. Offering riders in that prompt would leak the host's
  footprint through the riders' files, so offers come after the plan (PRD3-R §1.2, §6).
- **Architect tools.** `orchestrator/src/orchestrator/agents/roles.py::ARCHITECT` allows `Read,
  Glob, Grep, Bash`, the escalation and memory tools, `mcp__fused-memory__submit_task`, the
  jcodemunch tools and `_PLAN_CREATOR_TOOLS`; it disallows `Edit`, `Write` and
  `_NO_TASK_STATUS_WRITE` (`mcp__fused-memory__set_task_status`). It holds no `update_task`,
  `add_dependency` or `set_task_status`, so it cannot carry or close a task itself; the harness
  is the writer (PRD3-R §1.4, headline 5). The plan-tools FastMCP stdio server
  `orchestrator/src/orchestrator/mcp/plan_tools.py` (`create_server`, launched per worktree by
  `mcp_lifecycle.py::plan_tools_mcp_server`) holds `create_plan`, `add_plan_step(step_id,
  step_type, description)`, `drop_plan_file`, `remove_plan_step`, `replace_plan_step`,
  `confirm_plan`, `mark_step_done(step_id, commit_sha)` (ancestor-checked commit per step) and
  `record_descoped_step(step_id, commit, disposition, authority, note)`, which writes a typed,
  frozen `orchestrator/src/orchestrator/plan_descope.py::DescopedStep` under
  `plan_descope.DESCOPED_STEPS_KEY` (`_descoped_steps`). That record is the precedent for a typed
  verdict inside plan.json; its flat-string parameters are deliberate so the markup guard covers
  them. Every plan field has a `plan_tools.py::_PlanField` fact (`tool`, `also_written_by`) and
  every collection a `_COLLECTION_SCHEMA_TOOL` entry; the repair tests assert them, so a new tool
  registers both.
- **Scope enforcement and `scope_violation` emitters.** The lock charter is `metadata.files`
  (creation guard `fused-memory/src/fused_memory/middleware/lock_charter_guard.py`); the executed
  scope is the plan files; the implementer, debugger and simple_task are told to stay inside the
  plan's files and self-report otherwise (`roles.py::_SCOPE_BOUNDARY_RECOURSE`,
  `::_SCOPE_BOUNDARY_RECOURSE_SIMPLE`). `escalation/src/escalation/categories.py::SCOPE_VIOLATION`
  is filed by (1) that self-report, an L0 the steward grants with `granted_files`; (2)
  `orchestrator/src/orchestrator/harness.py::Harness._block_and_escalate_cross_repo`; (3)
  fused-memory's `middleware/scope_violation_escalator.py::ScopeViolationEscalator` at submit.
  The reviewer files none; the D2 reviewer (`::build_context_complete_reviewer_prompt`) reads the
  task block, so rider work not named in the task text reads as unexplained churn; the completion
  judge (`::build_completion_judge_prompt`) reads the plan, so rider-tagged steps count as in
  scope. Under a key-subset rule emitter 1 cannot fire on rider content (PRD3-R §1.5).
- **The done chokepoint is the interceptor, not the scheduler.** The orchestrator's own merged
  stamps go through `scheduler.py::Scheduler.mark_done(task_id, *, kind, sha, note)` (the
  post-merge stamp in `workflow.py::_merge_and_finalise`, both train-member stamps in
  `workflow.py` and `harness.py`, `workflow.py::_finalise_recovery_done`, and the advanced-merge
  stamp in `orchestrator/src/orchestrator/merge_lane/worker.py`), but two production writers do
  not: the hand-carry procedure (`skills/merge-queue/SKILL.md`, "Update the task":
  `set_task_status(..., 'done', done_provenance={'kind': 'merged', ...})`, Leo's SOP for every
  hand carry and `/unblock`) and the steward (`agents/roles.py` `kind="merged"` text), which call
  `set_task_status` directly and run no `Scheduler`. The chokepoint every writer passes is
  `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor.set_task_status`
  → `_apply_status_transition`, under `_write_lock(project_id)`, where PRD 1 ζ already hooks its
  promotion triggers (PRD 1 decision 14) and where `_validate_done_provenance` rev-parses and
  `merge-base --is-ancestor`s the commit. A fast-forward landing (21% of df) advances main to the
  rebased tip; the host's landing sha is what the lane records as `merge_sha`, with or without a
  merge commit. `EventType.merge_finalized` is emitted by
  `orchestrator/src/orchestrator/merge_lane/intake.py::enqueue_merge_request`'s `_on_finalized`
  callback on every terminal lane outcome (done, blocked, abandoned, superseded) and on no hand
  landing, so it is neither a sufficient nor a necessary close hook. `shared/src/shared/task_metadata.py::DoneProvenance`
  kinds: `merged` (commit required), `found_on_main` (stamped after a full execute whose branch
  landed by hand, by a sibling or by a citation commit: `harness.py`'s ground-truth path,
  `escalation/src/escalation/git_authority.py::found_on_main_response` as called from
  `escalation/src/escalation/server.py`), the deterministic kinds and
  `operational-verified`; a cancel carries no provenance kind (PRD3-R §1.6).
  `fused-memory/src/fused_memory/middleware/done_provenance_authz.py` authorises done writes by
  identity; a rider close runs under the host's write identity (§7).
- **Cancelled-descendant sweep.** `fused-memory/src/fused_memory/reconciliation/targeted.py::TargetedReconciler._sweep_cancelled_descendants`
  enumerates descendants of a cancelled task as `spawned_from == A` or `A ∈ dependencies`, so a
  carried rider qualifies by its host edge, and routes each to auto-cancel (child with
  `escalation_id` whose declared files are absent), a blocking `scope_violation` L1 when the
  orchestrator is live, or `blocked` with `reopen_reason='parent_cancelled:<A>'` when it is not.
  A rider still edge-linked when that sweep runs is blocked or cancelled, never "pending and
  held". **Cancel-with-children** (`plans/cancel-with-children-prd.md`, 8142–8148, all `pending`
  on 2026-11-04) §4.3 enumerates children and dependents by edge (inverted
  `get_dependency_edges`) and §4.2 `keep` reads a cancelled dependency as satisfied.
- **PRD 1 records this PRD consumes** (`plans/follow-up-filing-shape-prd.md` §4, filed 8413–8421,
  none landed on 2026-11-04): `metadata.followup_items` (`shared/followup_items.py`:
  `FollowupItem`, `merge_items`, `legacy_item`, `MAX_FOLLOWUP_ITEMS = 5` enforced on create
  only) and `agents/briefing.py::_format_followup_items` (β 8414); `metadata.followup_hold`
  with `holder`, `held_at`, `promoted: PromotionRecord | None` and the hold rule (ε 8417);
  `promote(tm, project_root, held_id, *, by, ref)` with `by='rider'` in its enum, fired by
  `add_dependency` from a pending task, every `deferred → pending` landing of a dependent, an
  `update_task` adding `x_finding_key` or `prd_path`, `_execute_combine`, and a human
  `remove_dependency` (ζ 8418; PRD 1 decision 14). `pending_since` is stamped by
  `fused-memory/src/fused_memory/backends/sqlite_task_backend.py::stamp_pending_since` and is not
  re-stamped by a hold or a promotion. Registered submodels go through
  `shared/task_metadata.py::register_metadata_submodel(key, model, cardinality=)`; a
  caller-supplied value for a server-stamped key is refused at the interceptor's public
  `submit_task` / `update_task` (PRD 1 decision 7, the `DeclaredScopeWriteAuthorityError` idiom).
- **PRD 2 records this PRD consumes** (`plans/followup-packing-sweep-prd.md` §4.1, draft):
  `fused-memory/src/fused_memory/middleware/followup_packing.py::RIDER_RESERVE_DAYS = 14`,
  `PackMember.rider_of: int | None`, `PackExclusion.RIDER` / `RIDER_RESERVED` and the pure
  `exclusion(member, now=)` predicate (decision 13 there: a carried rider is never packed; a held
  task with files is reserved 14 days from `pending_since`; fileless tasks go straight to the
  packer; the reservation applies from P1 on, before this PRD lands).
- **Fused-memory write discipline.** MCP verbs are exposed in
  `fused-memory/src/fused_memory/server/tools.py` and delegate to interceptor methods
  (`set_task_claimant`, `add_dependency`, `update_task` are the shapes). `update_task` accepts
  `details` as a structured column write.
- **Config and cost rows.** Orchestrator sections are pydantic submodels in
  `orchestrator/src/orchestrator/config.py` (`UnblockAutoConfig`, `ReviewConfig`, ...) with
  hot-reloadable leaves registered in `config.py::RELOADABLE_FIELDS` through
  `_submodel_leaf_paths`. Per-invocation cost rows are `EventType.invocation_end` with `role`,
  `phase`, `cost_usd`, `duration_ms` and `turns` from `shared/cli_invoke.py::AgentResult`
  (`turns` is 0 when the backend does not track it).
- **Measured limit** (PRD3-R §2, stores read 2026-11-04, riders = pending tasks matching PRD 1's
  hold rule, hosts = pending medium+ non-deterministic, keys = `derive_modules` at project
  depth): riders with ≥ 1 key df 1,019 of 1,400 (381 fileless, 375 of them reviewer
  suggestions), reify 1,007 of 1,106 (99 fileless); any key shared with a host 87% / 91%;
  exactly one key 14% / 18%; **subset rule, no key cap: 295 (28.9%) / 367 (36.4%)**; subset ≤ 2
  keys 226 (22.2%) / 272 (27.0%); one-key with test/doc-only 40 (3.9%) / 52 (5.2%). Flow proxy
  under the ruled no-cap rule: 230 × 28.9% ≈ **66 (df)** and 150 × 36.4% ≈ **55 (reify)**
  avoided landings a month before architect declines; under the joint rule about 9 and 8.
- **Offers per host** (PRD3-R §2.2): any-overlap offering floods (p90 142 to 160 riders per
  host); every subset variant has p50 4 to 7, p90 13 to 18, max 29 to 36 per host, so a cap of
  5 ranked offers bounds the prompt at about five item lines; hub keys under the subset rule are
  mild (df `OPERATIONS.md` 4, `orchestrator/tests/test_merge_queue.py` 3). Hosts offered ≥ 1
  rider under subset: df 423 of 470 (90%), reify 594 of 814 (73%), so the rider pass is one
  extra architect continuation on about 90% (df) and 73% (reify) of dispatches in `mode: offer`.
- **Host class mix** (PRD3-R §2.3): about 30% of offered hosts are PRD tasks in both projects
  (merge-fail df 25.4%, reify 33.9% against 2 to 19% for other classes); excluding PRD hosts
  keeps df 243 of 295 and reify 234 of 367 subset riders (82% / 64%), and only 50% of reify's
  joint-rule riders. Ranking PRD hosts last is unmeasured because dispatch order decides, and the
  other pending hosts' key sets come from their declared `metadata.files`, which under-declare on
  44% of df dispatches (PRD3-R §3.1).

## 3. Resolved design decisions

Leo, 2026-11-03/04: **riders at dispatch, with a limit.** The limit is **subset with no key
cap**: every rider lock key must already be inside the host's post-plan lock-key set; no
test-only or doc-only term; at most 5 offers per dispatch; the host's merged `followup_items`
stay ≤ `MAX_FOLLOWUP_ITEMS = 5`, so an accept that would exceed it is refused. The close shape is
**carry then close**. Leo's original wording was "folded and cancelled" (accepted riders folded
`fold{into: host}` and cancelled at accept). Carry-then-close changes *when* the rider leaves
the pool, not *whether*: under cancel-at-accept a host that later fails, is cancelled or lands
without the rider's steps loses the rider's work silently, because cancel-with-children covers
only edge-linked dependents and a cancelled rider has no edge to its host, and un-cancelling a
task is a transition the store does not offer today (PRD3-R headline 4, §1.6, §3.4); with
8142–8148 all pending, the `superseded_by` stamp cancel-at-accept would need does not exist
either. Carry keeps the rider's holder edge, adds a dependency edge on the host and stamps
`metadata.rider_of = host`; when the host lands with every rider-tagged step done,
`promote(by='rider', ref=host)` then `done` with `done_provenance {kind: merged, commit: <host
landing sha>}`; a rider step not done releases the rider to the held pool; a cancelled host
releases its riders in its own cancel transition and, under cancel-with-children, carries them
as dependents with a default disposition of `keep` plus release.

Lead's decisions under those rulings:

1. **Offers are computed after the plan, against the plan's lock keys, never in the first
   briefing (C-A1).** The call sits in `TaskWorkflow._plan` after `_reconcile_plan_scope`
   succeeds and before `_stamp_architect_plan`'s lock, against `self.modules`. Simple-path hosts
   (`_run_simple_task`) and deterministic tasks get no offers; a host re-dispatched with an
   existing plan gets offers again, excluding riders it already carries.
2. **The rider pass is one bounded architect continuation modelled on
   `build_plan_tightening_prompt`**, with three modes: `observe` computes offers and writes the
   ledger only; `shadow` runs the pass on a sampled fraction of offer-bearing dispatches
   (`riders.shadow_sample`, default 0.2), records cost, turns and verdicts with `mode: shadow`,
   restores the plan snapshot and applies nothing to the store; `offer` runs it on every
   offer-bearing dispatch and applies the verdicts. The pass runs once per plan, with the
   plan.json snapshot restored if the invocation raises, and every unanswered offer recorded as
   declined with reason `no_verdict` by the harness, so an unanswered offer is a record, not
   silence.
3. **The architect decides through typed plan-tools records**: `accept_rider(rider_id, step_ids)`
   and `decline_rider(rider_id, reason, note)` write `plan._riders: list[RiderVerdict]`, frozen
   dataclasses like `plan_descope.DescopedStep`, with flat-string parameters for the markup guard;
   `add_plan_step` gains an optional `rider_id`. `accept_rider` refuses, typed, an unoffered rider,
   a missing step, a step already claimed by another rider, a rider whose files are not all under
   the plan's lock keys, and a rider whose items would push the host over `MAX_FOLLOWUP_ITEMS`
   (the offer record carries the host's headroom and each rider's item count, so the check is
   local to plan.json). Rider steps are added last and tagged. Both tools register `_PlanField`
   facts (`accept_rider` writes `files` and `_riders`, `also_written_by` naming each other and
   `create_plan`) and `_COLLECTION_SCHEMA_TOOL` entries; `_riders` and `_rider_offers` join the
   `create_plan` carry-forward set beside `DESCOPE_PLAN_KEYS`, so `_replan` and a revalidation
   `create_plan` keep them. A carried rider whose `_riders` record is nevertheless missing from
   the plan at reconcile time is released (`records_lost`), never silently kept.
4. **Accept adds the rider's files to `plan.files`; it never widens the lock set, and the check
   is a typed path, not an assert.** The key-subset rule makes the post-pass
   `_reconcile_scope_locks(plan_files)` a module no-op when the architect only accepted and
   declined; if the module set differs (the architect also called `drop_plan_file` or
   `update_plan_metadata` during the pass, invalidating `accept_rider`'s check against the
   recorded offer keys), the workflow restores the snapshot, records every verdict as declined
   `pass_invalid`, emits `rider_pass_failed{reason}` and continues the dispatch with the pre-pass
   plan; `metadata.files` is not written on either path. The added files are recorded on the
   verdict (`added_files`) so a release can take back exactly those the branch never touched.
5. **One fused-memory writer for every rider effect on the store** (heuristics 7, 10, 11):
   `apply_rider_verdicts(project_root, host_id, offers, verdicts, run_id, mode)` under
   `_write_lock`. Accept (`mode: offer` only): `merge_items(host.followup_items, rider items)` (a
   legacy single is one item through `legacy_item`), the `Riders:` block regenerated in the
   host's `details` from the set of carried riders, the host edge (`add_dependency(rider,
   host)`, skipped when 6072's parent edge already is that edge), `rider_of = host`,
   `rider_ledger.carried_at`. Decline, shadow verdict and offer: the ledger only. **Never a write
   to `metadata.files`** (task 3260). The verb re-validates under the lock (rider still held,
   `promoted is None`, not carried, host headroom unchanged) and records a verdict that fails
   re-validation as declined `stale_offer` (`corroborate-before-acting`).
6. **PRD hosts are allowed, ranked last; `riders.exclude_prd_hosts` exists and defaults false.**
   A rider is not offered to a PRD host (`metadata.prd_path` present) while a non-PRD pending
   host whose declared key set covers it exists (declared keys under-declare on 44% of df
   dispatches, so this rule errs toward offering to the PRD host; δ reports offers by host
   class). Ranking of the remaining candidates: same `spawned_from` as the host or spawned from
   the host itself first, then fewest `rider_ledger.declines`, then oldest `pending_since`. Cap
   `riders.max_offers` (default 5).
7. **`metadata.rider_ledger` is a registered submodel on the held task, a separate key, written
   only by this PRD's verbs**: the lead's five fields `{offers, declines, last_offered_at,
   last_host, last_decline_reason}` plus three the fold-in added for the carry: `carried_at`,
   `deliverable_at` and `deferred_promotion` (§4.4). Not a field on PRD 1's server-stamped
   `FollowupHold`, so PRD 1's write authority is untouched; a caller-supplied `rider_ledger` or
   `rider_of` is refused with `RiderWriteAuthorityError`. `deferred_promotion` is written by the
   rider promotion guard, which is γ's code registered into PRD 1 ζ's generic hook (decision 11),
   so no PRD 1 module writes or reads this key. Observe-mode offers count as offers (the report
   splits by event `mode`).
8. **The reservation is PRD 2's; this PRD owns `rider_of`, the ledger and the offer side.** PRD
   2's `followup_packing.RIDER_RESERVE_DAYS = 14` from `pending_since` and its
   `PackExclusion.RIDER_RESERVED` / `RIDER` exclusions in `exclusion()` are the single home of
   "reserved for riders" and "carried, never packed"; this PRD imports the constant for the carry
   clock (decision 18) and defines nothing of its own about reservation. The offer-side predicate
   `riders.py::offerable(task, *, now)` (held, `promoted is None`, `status == pending`, ≥ 1
   declared file, `rider_of` absent, no dependents) is the selector's, not the packer's; the two
   agree by construction because the packer excludes `rider_of` presence and this PRD guarantees
   (decisions 10, 11, 18) that `rider_of` is never left set on a rider whose host is terminal or
   stalled. Two companion corrections to PRD 2 are filed at decompose (§5): its `RIDER`
   exclusion names this definition, and its `exclusion()` reads `rider_ledger` when present
   (`last_decline_reason == 'different_intent'` steers a task to same-parent packs). Consequence
   at deploy: the backfilled stock is already older than 14 days by `pending_since`, so it is the
   packer's; riders serve the fresh flow.
9. **Deliverability is decided by the workflow from the plan and persisted to the store; the
   close decision reads the store and never the plan.** `TaskWorkflow._reconcile_rider_carry`
   runs at every execute → review transition, after `_replan`, and after a narrowing commit,
   idempotently: a carried rider is deliverable iff ≥ 1 plan step carries its `rider_id` and
   every such step is done (`mark_step_done` commit recorded); the workflow calls
   `stamp_rider_deliverability(host, rider, deliverable)`, which sets or clears
   `rider_ledger.deliverable_at`. A rider that is not deliverable at that point is released at
   once (`release_rider(reason='steps_undone' | 'step_removed' | 'file_dropped' |
   'records_lost')`), its undone rider-tagged steps are removed from the plan, and its
   `added_files` the branch never touched are taken back from `plan.files` (the merge gate's own
   not-touched computation, reused). So any rider still carried when the host's merge request is
   submitted is deliverable and stamped so; plan.json, which lives in the worktree and is reaped
   after landing, is not needed afterwards.
10. **Close and release live in `TaskInterceptor._apply_status_transition`, inside the host's
    own transition, riders first.** On a host `done` whose provenance is merged-family
    (`merged`, or `found_on_main` with a commit), every task with `rider_of == host` is handled
    in the same `_write_lock` unit before the host's row flips: a rider with `deliverable_at`
    set is closed (`promote(by='rider', ref=host)` if its holder edge is still present, tolerating
    `promoted is not None`; then `done` with `{kind: merged, commit: <host commit>}`, the ancestor
    check applying unchanged); a rider without it is released (`not_deliverable`). On a host
    `done` of any other kind (`operational-verified`, a deterministic kind) riders are released
    (`host_not_merged`). On a host `cancelled`, riders are released (`host_cancelled`) in the
    cancel transaction, before `_sweep_cancelled_descendants` can enumerate them, so they stay
    pending and held with no L1; under 8147 a carried rider's default disposition is `keep` plus
    this release. **The close is total: a rider that cannot be closed is released, never a
    reason to fail the host.** The orchestrator's post-merge stamp maps a refused
    `set_task_status` to `_mark_blocked(escalate_to_human=True)` and the landed-reconcile path
    retries every tick, so a rider-side refusal (a human re-statused the carried rider, `promote`
    refuses, the rider's own `_validate_done_provenance` fails) propagating out of the unit would
    turn a landed host into a false-blocked task with an L2 about a rider. Instead the unit
    releases that rider (`rider_of`, host edge, `carried_at`, `deliverable_at` cleared, holder
    edge kept, deferred promotion applied) and emits `rider_close_failed{rider, error}`; the
    host's transition proceeds. Only the lock unit itself failing to write fails the host. The
    invariant "no status write on a host leaves a rider carried by a terminal host" holds by
    release-on-failure. A rider's done kind is `merged` by design even when the host was stamped
    `found_on_main`: the rider's work landed through a merge, whichever way the host's landing
    was attributed, and the validator checks the commit, not the host's kind. The unit resolves
    and ancestor-checks the host's commit once and reuses that resolution for every rider (up to
    five), so the lock is held for two git subprocesses, not ten. `Scheduler.mark_done` and
    `merge_finalized` are not hooked. `found_on_main` after a full execute closes deliverable
    riders because their steps are on main in exactly that case.
11. **The carry race is closed by a promotion guard; PRD 1 ζ learns nothing about riders.**
    8418's companion correction is one generic hook in `followup_hold.py`:
    `register_promotion_guard(fn)` (the `register_metadata_submodel` idiom), where a guard is
    `fn(task, *, by, ref) -> PromotionDeferral | None`; `promote` consults every registered guard
    before removing a holder edge and, on a deferral, skips the edge removal and emits
    `followup_promotion_deferred{task_id, by, ref, guard}`. γ of this PRD registers the one
    guard that exists: it reads `rider_of`, defers `by ∈ {dependent, citation, combine, pack}`
    while the rider is carried, and writes `rider_ledger.deferred_promotion = {by, ref, at,
    guard: 'rider'}` (owned here, `DeferredPromotion` in `shared/rider_ledger.py`); γ imports
    nothing from ζ beyond the registration function. `release_rider` applies the deferred
    promotion (the rider leaves the hold with the cause it earned); the close path ignores it. A
    human `remove_dependency(rider, holder)` is not deferred: it promotes `by=human` and the
    carry survives it, which is the human's stated intent; if that host is then cancelled or
    stalls, the release leaves the rider with no edge at all, i.e. ready, again the human's
    intent, and the release event records it. `_execute_combine` skips tasks with `rider_of`.
    The selector's "no dependents" check at offer plus this guard mean a carried rider is never a
    dispatch candidate between the host's done write and its own close.
12. **The host edge is the authority for "carried"; `rider_of` is the index.** Both are written
    in one `_write_lock` unit by the verbs; the integration gate asserts they agree; the selector
    and PRD 2's packer read `rider_of` (`find_tasks_by_metadata('rider_of', host)` for the
    carried set), the scheduler reads only the edge (`one-fact-one-home`). Ids are `int`.
13. **The implementer is told what a rider step is.** The plan render marks rider-tagged steps
    `[rider <id>]`, places them after host steps, and `roles.py` says they may be left undone when
    the host's own steps consumed the budget, with no self-report: an undone rider step releases
    the rider. The implementer's scope-boundary text is unchanged.
14. **Config `riders.{enabled, mode, shadow_sample, exclude_prd_hosts, max_offers}`**,
    green-tier reloadable, `mode ∈ {observe, shadow, offer}` defaulting to `observe`;
    `enabled: false` computes nothing and writes nothing. The kill decision is a human config edit
    on the report's `KILL` line, not an automatic flip.
15. **The rider pass's cost is measured in `shadow` mode before `offer`.** β's `rider_pass{mode,
    host, offers, accepted, declined, cost_usd, duration_ms, turns}` rows from the sampled
    shadow passes give the per-dispatch cost, the accept rate and the decline-reason mix
    directly; δ reports them beside the avoided fixed charges (task-shape §7 A: USD 2.3
    implementer + 2.5 to 5.2 architect per avoided tiny task) and the offer-mode rows once
    flipped. The observe-mode fallback line (dispatches with ≥ 1 offer × the median
    `invocation_end` cost of `_replan`, the plan-edit continuation closest in shape) is printed
    only while no shadow row exists, labelled as an estimate.
16. **Observation is one reader-shaped leaf** rendering each of the seven PRD3-R §5 series, the
    pass cost and the carried-behind-a-stalled-host count as "producer has not written" when its
    event count is zero (INV-13), with the kill threshold evaluated on matched pairs (origin class
    and S2 lines tercile) at n ≥ 30.
17. **Items merge keeps the host's bundle whole.** Rider items enter `followup_items` by key; on
    release they are removed by key unless the host already carried that key before the accept
    (the verdict records `merged_keys`); the `Riders:` block is regenerated from the carried set
    on every verb call, never edited in place (heuristic 12).
18. **The carry has the same 14-day clock as the reservation, so no parked host holds a rider out
    of both drains.** A carried rider is released `host_stalled` (host edge and `rider_of`
    removed, holder edge kept, deferred promotion applied) iff `now − carried_at >
    RIDER_RESERVE_DAYS` **and** the host's status at sweep time is not `in-progress` (pending
    after a requeue, blocked, deferred). A host that is in-progress, however old the carry, keeps
    its riders: their steps are on its branch and land with it, and a release would re-dispatch
    work that is about to land and strip `merged_keys` from a host still carrying it
    (blocked → unblocked → re-verify loops run for weeks on both projects). The sweep re-checks
    each tick, so a host that leaves in-progress past day 14 releases its riders at the next
    tick. The owner of that bound is the fused-memory sweep `reconcile_carries`, run at server
    start after PRD 1's backfill and every `riders.carry_sweep_interval_secs` (fused-memory
    `config/schema.py`, default 3,600) thereafter; it also closes or releases riders of hosts
    that became terminal while the store was down and un-blocks a rider whose `reopen_reason` is
    `parent_cancelled:<its rider_of>`. With this clock the only open-ended carry is behind a host
    that stays in-progress, which is the host's own liveness, supervised by its claimant
    heartbeat and the L2 ladder; after release the rider is PRD 1's held task again, under PRD 1
    ε's waiver and PRD 2's drain. No G7 waiver is recorded here.

## 4. Contract (B+H)

### 4.1 Selector (`orchestrator/src/orchestrator/riders.py`, pure)

```python
@dataclass(frozen=True)
class HostView:                 # built by the workflow from the scheduler's task snapshot
    task_id: int
    keys: frozenset[str]        # derive_modules(metadata.files, lock_depth) for pending hosts;
                                # the live host passes its post-plan self.modules
    is_prd: bool                # metadata.prd_path present
    spawned_from: int | None

@dataclass(frozen=True)
class RiderOffer:
    rider_id: int
    rank: int
    keys: frozenset[str]        # ⊆ host.keys by construction
    files: tuple[str, ...]      # the rider's declared files
    item_count: int             # len(followup_items) or 1 for a legacy single
    relation: Literal['sibling', 'child', 'none']
    declines: int

def offerable(task: Mapping, *, now: datetime) -> bool
    # followup_hold present and promoted is None; status == pending; >= 1 declared file;
    # rider_of absent; no dependents (keeps cancel-with-children recursion out; dependents are
    # 1.6% / 3.0% of low follow-ups)

def select_rider_offers(
    host: HostView, held_pool: Sequence[Mapping], pending_hosts: Sequence[HostView],
    *, lock_depth: int, max_offers: int, exclude_prd_hosts: bool, now: datetime,
) -> tuple[RiderOffer, ...]
```

After `offerable`, the limit: `derive_modules(rider.files, lock_depth)` non-empty and ⊆
`host.keys`. Then the PRD rule: if `host.is_prd` and some non-PRD `pending_hosts` entry covers
the rider, skip; if `exclude_prd_hosts` and `host.is_prd`, return `()`. Ranking: relation
(`sibling` = same `spawned_from`, `child` = `rider.spawned_from == host.task_id`) first, then
`rider_ledger.declines` ascending, then `pending_since` ascending, then id. Truncate to
`max_offers`. A selector that cannot read the pool raises; the workflow emits
`rider_offers_skipped{reason}` and continues the dispatch (never an empty offer list standing in
for a failure).

Config (`config.py::RidersConfig`, registered in `RELOADABLE_FIELDS`): `enabled: bool = False`,
`mode: Literal['observe', 'shadow', 'offer'] = 'observe'`, `shadow_sample: float = 0.2`,
`exclude_prd_hosts: bool = False`, `max_offers: int = 5`. The reservation constant is not
config: it is PRD 2's `followup_packing.RIDER_RESERVE_DAYS`.

### 4.2 Plan records and plan tools (`orchestrator/src/orchestrator/plan_riders.py`, `mcp/plan_tools.py`)

```python
RIDER_OFFERS_KEY = '_rider_offers'      # written by the harness before the pass; read by accept_rider
RIDERS_KEY = '_riders'                  # list[RiderVerdict]; both keys in the create_plan carry-forward set
DECLINE_REASONS = frozenset({'different_intent', 'outside_plan_keys', 'too_large', 'already_done',
                             'other', 'no_verdict', 'stale_offer', 'items_cap', 'pass_invalid'})

@dataclasses.dataclass(frozen=True, kw_only=True)
class RiderVerdict:                     # plan_descope.DescopedStep idiom; non-blank strings enforced
    rider_id: int
    verdict: Literal['accept', 'decline']
    step_ids: tuple[str, ...]           # accept: ≥ 1, each an existing step tagged rider_id == rider_id
    reason: str                         # decline: ∈ DECLINE_REASONS; accept: ''
    note: str
    added_files: tuple[str, ...]        # accept: rider files appended to plan.files by accept_rider
    merged_keys: tuple[str, ...]        # filled by apply_rider_verdicts: item keys new to the host
    authority: Literal['architect', 'harness']   # no_verdict / stale_offer / pass_invalid are harness-authored

# plan-tools (flat-string parameters, MarkupGuardMiddleware coverage as record_descoped_step):
def accept_rider(rider_id: str, step_ids: str) -> dict      # comma-separated step ids
def decline_rider(rider_id: str, reason: str, note: str) -> dict
def add_plan_step(step_id, step_type, description, rider_id: str = '') -> dict
```

`accept_rider` refuses typed (`{'status': 'error', 'code': ...}`): `not_offered`,
`unknown_step`, `step_claimed`, `outside_plan_keys` (rider keys ⊄ the offer's recorded host
keys), `items_cap` (`host_items + item_count > MAX_FOLLOWUP_ITEMS`). On success it appends the
rider's files to `plan.files` and records `added_files`. A second verdict for the same rider
replaces the first. `_PlanField` facts: `_riders` written by `accept_rider` / `decline_rider`;
`files` gains `accept_rider` in `also_written_by`; `_COLLECTION_SCHEMA_TOOL['_riders']`.

### 4.3 Rider pass (`agents/briefing.py`, `workflow.py`)

`BriefingAssembler.build_rider_offer_prompt(task, plan, offers, context=None) -> str`: the
architect preamble (`_task_preamble(task, 'architect', context)`, files still excluded), the
plan's current steps and files, then one block per offer: rider id, title, items via
`_format_followup_items` (or the legacy single), declared files, relation, declines so far, and
the rule: accept only a rider that is the same intent as a step area the plan already has,
whose files are within the plan's lock keys; add its steps last with `rider_id`; otherwise
`decline_rider` with a reason; finish with `confirm_plan()`. The prompt never names the mode.

`TaskWorkflow._run_rider_pass(offers, mode)`: writes `_rider_offers` (ids, item counts, host
headroom, host keys) into plan.json, snapshots plan.json, invokes `ARCHITECT` once with the
prompt builder, fills `no_verdict` declines for unanswered offers, then branches: if the
invocation raised or `derive_modules(plan.files)` differs from `self.modules`, restore the
snapshot, record every verdict `pass_invalid`, emit `rider_pass_failed{reason}`; in `shadow`
mode restore the snapshot unconditionally and hand the verdicts to `apply_rider_verdicts(...,
mode='shadow')` (ledger and events only); in `offer` mode keep the plan, re-run
`_stamp_architect_plan`, and hand `(offers, verdicts)` to `apply_rider_verdicts(..., mode='offer')`.
Always emits `rider_pass{mode, host, offers, accepted, declined, cost_usd, duration_ms, turns}`
(`turns` from `AgentResult.turns`, 0 when untracked). One-shot per plan (`_rider_pass_done`);
`shadow` samples by `hash(run_id, host) < shadow_sample`.

### 4.4 Store verbs and records (`fused-memory/src/fused_memory/middleware/rider_carry.py`; MCP in `server/tools.py`; the interceptor's status transition)

```python
class DeferredPromotion(BaseModel):     # frozen; written only by the rider guard (γ)
    by: Literal['dependent', 'citation', 'combine', 'pack']   # human is never deferred (decision 11)
    ref: str | None
    at: str
    guard: Literal['rider']

# PRD 1 ζ's generic hook (8418 companion correction), in followup_hold.py; riders are not named there:
#   class PromotionDeferral(BaseModel): guard: str; reason: str
#   PromotionGuard = Callable[[Mapping, str, str | None], PromotionDeferral | None]   # (task, by, ref)
#   def register_promotion_guard(fn: PromotionGuard) -> None
#   promote() consults every guard before removing the holder edge; a deferral skips the removal
#   and emits followup_promotion_deferred{task_id, by, ref, guard}
def rider_promotion_guard(task, by, ref) -> PromotionDeferral | None
    # registered explicitly in fused-memory/src/fused_memory/server/main.py beside the other startup
    # registrations (γ), never as an import-time side effect of rider_carry.py
    # defers iff rider_of is set and by != 'human'; writes rider_ledger.deferred_promotion

class RiderLedger(BaseModel):           # metadata.rider_ledger on the held task; frozen, extra='forbid'
    offers: int = 0
    declines: int = 0
    last_offered_at: str | None = None  # ISO-8601 UTC
    last_host: int | None = None
    last_decline_reason: str | None = None
    carried_at: str | None = None       # set at accept, cleared at release and close
    deliverable_at: str | None = None   # set/cleared by stamp_rider_deliverability; read by the close
    deferred_promotion: DeferredPromotion | None = None   # PRD 1 ζ seam, decision 11
register_metadata_submodel('rider_ledger', RiderLedger, cardinality='dict')
# metadata.rider_of: int — the host id; written only by the verbs below; a caller-supplied
# rider_of or rider_ledger on submit_task / update_task is refused with RiderWriteAuthorityError

@dataclass(frozen=True)
class RiderOutcome:
    rider_id: int
    applied: Literal['carried', 'declined', 'stale_offer', 'shadow']
    reason: str | None

@dataclass(frozen=True)
class RiderApplyResult:
    host_id: int
    outcomes: tuple[RiderOutcome, ...]
    items_after: int

@dataclass(frozen=True)
class ReconcileReport:
    project_id: str
    closed: tuple[int, ...]
    released: tuple[tuple[int, str], ...]      # (rider_id, reason)
    unblocked: tuple[int, ...]

async def apply_rider_verdicts(tm, project_root, host_id, offers, verdicts, run_id, mode) -> RiderApplyResult
    # under _write_lock: ledger offers for every offer; mode='offer': for each accept that re-validates,
    # merge_items, Riders: block, host edge (unless the 6072 parent edge is that edge), rider_of,
    # carried_at; for each decline (incl. harness no_verdict / stale_offer / pass_invalid): ledger declines.
    # mode in {'observe', 'shadow'}: ledger only.
async def stamp_rider_deliverability(tm, project_root, host_id, rider_id, deliverable: bool) -> None
    # under _write_lock: sets or clears rider_ledger.deliverable_at; refuses if rider_of != host_id
async def release_rider(tm, project_root, rider_id, *, reason, host_id) -> bool
    # under _write_lock: re-reads the rider and NO-OPS (returns False, no event) unless rider_of == host_id,
    # so a decision taken outside the lock (the sweep's enumeration) cannot act on a rider the host's own
    # transition already closed or released. The rider's STATUS is deliberately not a condition: a rider a
    # human deferred or blocked by hand still has its rider_of and host edge cleared when its host lands,
    # fails or is cancelled, so no terminal host ever leaves a stale rider_of. Otherwise remove host edge (never
    # the holder edge, never a 6072 parent edge), clear rider_of, carried_at, deliverable_at; remove
    # merged_keys from host.followup_items; regenerate Riders: block; apply deferred_promotion if any
    # (promote with its cause); emit rider_released{reason}; return True
async def close_or_release_riders_for(transition) -> None   # awaited by _apply_status_transition on the
    # host, inside the host's _write_lock unit, before the host row flips: done + merged-family provenance →
    # resolve and ancestor-check the host commit ONCE, then close each deliverable rider (promote by='rider'
    # if the holder edge is present; done {merged, host commit}), release the rest (not_deliverable);
    # done of another kind → release (host_not_merged); cancelled → release (host_cancelled).
    # TOTAL: a rider whose close raises is released in the same unit with rider_close_failed{rider, error};
    # the host's transition proceeds. Only the lock unit itself raising fails the host.
async def reconcile_carries(tm, project_root, *, now) -> ReconcileReport
    # server start and every carry_sweep_interval_secs: close/release riders of hosts terminal while the
    # store was down; release host_stalled iff carry age > RIDER_RESERVE_DAYS AND host status is not
    # in-progress AND the rider's status == pending (the one arm that tests the rider's status: a rider a
    # human took out of pending is the human's, not the clock's); un-block
    # reopen_reason == f'parent_cancelled:{rider_of}'; every per-rider write goes
    # through release_rider / the close helper, which re-validate under the lock; one rider_reconciled
    # event per run
class RiderWriteAuthorityError(TaskBackendError)
```

`Riders:` block in `details`: delimited `<!-- riders:begin -->` … `<!-- riders:end -->`, one line
per carried rider (`- rider <id>: <title> — items <keys>`), regenerated wholesale from the carried
set on every verb call; an empty set removes the block.

### 4.5 Events (runs.db through `EventStore.emit`; store-side through the interceptor's event path)

`rider_offered{host, rider, rank, relation, mode}`, `rider_offers_skipped{host, reason}`,
`rider_pass{mode, host, offers, accepted, declined, cost_usd, duration_ms, turns}`,
`rider_pass_failed{host, reason ∈ {invocation_error, modules_changed}}`,
`rider_verdict{mode, host, rider, verdict, reason, authority}`, `rider_carried{host, rider}`,
`followup_promotion_deferred{task_id, by, ref, guard}` (PRD 1 ζ's event, `guard: rider`),
`rider_released{host, rider, reason ∈ {steps_undone, step_removed, file_dropped, records_lost,
not_deliverable, host_not_merged, host_cancelled, host_stalled, close_failed, human}}`,
`rider_close_failed{host, rider, error}`, `rider_closed{host, rider, commit}`,
`rider_reconciled{project, closed, released, unblocked}`.

Invariants: a rider is carried by at most one live host; a carried rider has the holder edge, the
host edge, `rider_of` and `carried_at`, and they agree; a rider's keys ⊆ the host's lock keys at
accept and the accept never changes the host's lock set or `metadata.files`; every offer ends in
exactly one verdict record; a rider that is carried when the host's merge request is submitted
has `deliverable_at` set; a carried rider's holder edge is removed only by the close or by a
human; no status write on a host leaves a rider carried by a terminal host (by close, else by
release-on-failure), whatever the rider's own status, so a terminal host never leaves a stale
`rider_of`; a rider's failure never changes a host's outcome; a rider leaves `pending`
only through the close (done) or, after release, through PRD 1's promotion causes.

### 4.6 Boundary-test sketch (the integration gate γ's signal; every scenario drives the public interfaces: a dispatch through the harness, plan-tools calls, `get_task`, `set_task_status`, `add_dependency`, `reload_config`, runs.db rows, never internals)

| # | scenario | preconditions | postconditions |
|---|---|---|---|
| 1 | observe mode, host plans with keys {a, b}; held riders R1 {a}, R2 {a, b}, R3 {a, c}, R4 fileless | `riders.enabled`, `mode: observe` | `rider_offered` for R1 and R2 only (R3 needs c, R4 has no key); `rider_ledger.offers == 1` on R1, R2; architect prompt unchanged (one invocation) |
| 2 | same host in `mode: offer`, architect accepts R1 with a tagged step, declines R2 `different_intent` | scenario 1 state | two architect invocations; `plan._riders` has two verdicts; R1 `dependencies == [holder, host]`, `rider_of == host`, `carried_at` set; host `followup_items` ⊇ R1 items; host `details` carries the `Riders:` block; R2 ledger `declines == 1`, `last_decline_reason == 'different_intent'`; `rider_pass{mode: offer}` row carries `cost_usd`, `turns` |
| 2s | same host in `mode: shadow` with `shadow_sample: 1.0` | scenario 1 state | two invocations; `rider_pass{mode: shadow}` and two `rider_verdict{mode: shadow}` rows; plan.json equals the pre-pass snapshot; no edge, no `rider_of`, no items change; ledgers `offers` incremented, `declines` not |
| 3 | architect answers neither offer | `mode: offer` | both verdicts `decline/no_verdict`, `authority == 'harness'`; ledgers updated; dispatch proceeds |
| 4 | architect calls `accept_rider(R3, ...)` (unoffered) and `accept_rider(R1, 'step-9')` (unknown step) | offers R1, R2 | both refused typed; `_riders` unchanged |
| 5 | host already carries 4 items, R2 has 2 | `mode: offer` | `accept_rider(R2)` refused `items_cap`; the offer was still made (headroom is checked at accept, not at offer) |
| 6 | the accept re-validates stale: R1 was promoted by citation between offer and apply (`rider_of` is not yet set, so no guard defers it) | scenario 2 inputs | verdict demoted to `decline/stale_offer`; no edge, no `rider_of`; `rider_verdict{reason: stale_offer}` |
| 6h | as 6, promoted by a human `remove_dependency` | | same postconditions |
| 7 | the architect accepts R1 and also `drop_plan_file`s a lock-bearing file during the pass | `mode: offer` | plan restored to the snapshot; both verdicts `pass_invalid`; `rider_pass_failed{reason: modules_changed}`; no `lock_acquired{reason: plan_refinement}`; `metadata.files` unchanged; dispatch proceeds |
| 7b | accepted R1's lock set vs the host's | scenario 2 | `lock_acquired{reason: plan_refinement}` count unchanged by the accept; `metadata.files` unchanged by the accept; `plan.files` ⊇ R1 files |
| 8 | implementer completes host steps and R1's step; host lands through the lane | scenario 2 | R1 `deliverable_at` set at execute → review; on the host's `done`, in the same `set_task_status` call, R1 `status == done`, `done_provenance == {kind: merged, commit: <host landing sha>}`, `followup_hold.promoted.by == 'rider'`, `ref == host`; `rider_closed`; R1 never appears in `verify_admission_plan`, merge events or a dispatch |
| 8h | same as 8 but a human stamps the host `done/merged` by hand (`/merge-queue` SOP) with no orchestrator attached | carried deliverable R1 | identical postconditions; no sweep needed |
| 8r | R1 carried and deliverable; a pending task adds an edge to R1; then the host lands | scenario 8 | `followup_promotion_deferred{by: dependent, guard: rider}`; `rider_ledger.deferred_promotion.by == 'dependent'`; R1's holder edge intact until the close; R1 `done/merged`; never dispatched |
| 8f | R1 carried and deliverable; a human set R1 `deferred` by hand; the host lands through the lane | scenario 8 | the host is `done/merged` with no `SetTaskStatusRejected`, no `_mark_blocked`, no L2; R1 is closed or released in the same transaction, never left carried (`rider_of` absent, host edge gone, holder edge intact on a release); a release emits `rider_close_failed{rider: R1}` |
| 8x | the host's done write itself fails inside the lock unit (backend error) | scenario 8 | the host's transition fails as today; R1 untouched (still carried); nothing half-written |
| 9 | implementer leaves R1's step undone | scenario 2 | at execute → review R1 is released: `rider_of` absent, host edge gone, holder edge intact, `rider_released{reason: steps_undone}`, R1's step absent from the plan, R1 items absent from host `followup_items`, `Riders:` block gone; host proceeds; the merge gate sees no untouched R1 file |
| 9d | as 9, but R1 had a deferred `citation` promotion | | on release R1 is promoted `by == 'citation'` with the deferred `ref`; it is ready next tick |
| 10 | a narrowing pass drops R1's only file | carried R1; merge-phase gate flags it | R1 released `file_dropped` |
| 10r | `_replan` after review rewrites the steps and drops R1's tagged step | carried R1 | R1 released `step_removed`; `_riders` still present in the re-planned plan.json (carry-forward) |
| 10c | a revalidation `create_plan` writes a plan whose `_riders` lacks R1 (carry-forward bypassed) | carried R1 | R1 released `records_lost` at the next reconcile |
| 11 | the host closes `found_on_main` after a full execute with R1's step on main | carried deliverable R1 | R1 closed with the host's commit |
| 11b | the host closes `found_on_main` with no deliverability stamp (no execute ran) | carried R1 | R1 released `not_deliverable`, back in the held pool |
| 12 | the host is cancelled | carried R1; 8147 not enforcing | in the cancel transaction R1 released `host_cancelled`; R1 still pending and held; no L1 and no `blocked` from `_sweep_cancelled_descendants`; the selector offers R1 again |
| 12e | the host is cancelled under 8147 enforcement | carried R1 | the disposition map lists R1 with default `keep`; the cancel releases it the same way |
| 13 | fused-memory restarts with R1 carried by a host already `done/merged` (host written while riders code was absent) | seeded | `reconcile_carries` closes R1 with the host's `done_provenance.commit`; `rider_closed`; one `rider_reconciled` event |
| 13s | R1 carried 15 days, host `pending` (requeued, parked by dependency) or `blocked` | sweep tick | R1 released `host_stalled`; holder edge intact; offered again at the next matching dispatch |
| 13i | R1 carried 20 days, host `in-progress` (re-verify loop or merge-queue wait) | sweep tick | R1 still carried; no event; released at the first tick after the host leaves `in-progress` without landing |
| 13r | the sweep enumerates R1 as stalled, then the host lands before the sweep's `release_rider` runs | both under `_write_lock` | the close runs first and clears `rider_of`; `release_rider` re-reads, finds `rider_of` absent, no-ops, emits nothing; R1 `done/merged` |
| 13h | R1 carried; a human set R1 `deferred`; 15 days later the host is `blocked` | sweep tick | R1 is not released by the clock (its status is the human's); when the host later lands or is cancelled, R1's `rider_of` and host edge are cleared in that transition (`close_failed` or `host_cancelled`), holder edge intact |
| 14 | PRD host H_p and non-PRD host H_n both cover R1; H_p dispatches first | `exclude_prd_hosts: false` | R1 not offered to H_p; offered to H_n |
| 15 | same as 14 with no non-PRD host for R1 | | R1 offered to H_p |
| 16 | `exclude_prd_hosts: true` via `reload_config` | H_p dispatches | zero offers; `applied` lists the leaf |
| 17 | R1 is a child of the host (`spawned_from == host`) with a 6072 parent edge | `mode: offer`, accept | no duplicate edge; `rider_of` set; release and close never remove the parent edge |
| 18 | a `submit_task` or `update_task` supplies `rider_of` or `rider_ledger` | any | refused `RiderWriteAuthorityError` |
| 19 | PRD 2's packer enumerates candidates | carried R1; eligible R5 with `pending_since` 3 d; R6 fileless | R1 `PackExclusion.RIDER`, R5 `RIDER_RESERVED`, R6 eligible (PRD 2's `exclusion()`; this row is shared with PRD 2's gate) |
| 20 | a simple-path host and a deterministic task dispatch | any | no `rider_offered` rows |
| 21 | the selector's pool read raises | any | `rider_offers_skipped{reason}`; dispatch continues; no ledger write |
| 22 | a `scope_violation` is filed with a rider host as task | δ fixture | δ counts it under series 4 |

## 5. Cross-PRD relationship and seams (G4)

| other PRD / document | direction | seam mechanism | owner | status |
|---|---|---|---|---|
| `plans/follow-up-filing-shape-prd.md` (PRD 1; 8413–8421 pending) | consumes | `followup_items`, `merge_items`, `legacy_item`, `_format_followup_items` (β 8414); `followup_hold`, hold rule (ε 8417); `promote(by='rider', ref=host)` (ζ 8418); `pending_since` unchanged by hold or promotion | PRD 1 owns the records; this PRD owns every rider reader and writer; `merge_items` is applied here under the host's `≤ MAX_FOLLOWUP_ITEMS` headroom check so PRD 1's create-only cap needs no change | queued (deps 8414, 8417, 8418) |
| PRD 1 ζ (8418) promotion triggers | PRD 1 ζ produces for | `followup_hold.register_promotion_guard(fn)`: `promote` consults registered guards before removing a holder edge and, on a deferral, skips the removal and emits `followup_promotion_deferred{by, ref, guard}`; ζ names no rider, no `rider_of`, no `rider_ledger`. γ registers the single rider guard (defers `by ≠ human` while `rider_of` is set, writes `rider_ledger.deferred_promotion`); `release_rider` applies the deferral, the close ignores it; `_execute_combine` skips `rider_of` tasks (decision 11) | ζ provides the hook; γ owns the guard. The hook is 8418's companion correction at this PRD's decompose (a generic registry with zero registrants until γ); 8418 carries no dependency on this PRD | companion correction |
| `plans/followup-packing-sweep-prd.md` (PRD 2, draft) | consumes | `followup_packing.RIDER_RESERVE_DAYS`, `PackExclusion.RIDER_RESERVED` / `RIDER`, `exclusion()`; `PackMember.rider_of: int | None` | PRD 2 owns the reservation and the exclusions; this PRD owns `rider_of`, `rider_ledger` and the offer side (decision 8); companion corrections to PRD 2 at decompose: `RIDER` names "rider_of set" as this PRD defines it (never left set on a terminal or stalled host), and `exclusion()` reads `rider_ledger.last_decline_reason` when present as a same-parent pack hint | PRD 2 lands first (its P0/P1 do not depend on riders) |
| `plans/cancel-with-children-prd.md` (8142–8148 pending) | consumes | a carried rider is a dependent of its host by edge, so a host cancel enumerates it; the cancel transition releases it first (decision 10) and its default disposition is `keep` plus release; one sentence added to its §4.2 seam list as a companion correction at decompose | that PRD owns dispositions; this PRD owns the edge and the in-transaction release | queued |
| task 3260 (`metadata.files` owned field) | coexists | no rider path writes `metadata.files`; the accept is a module no-op by the subset rule, and a non-no-op is a typed failure | 3260 | pending; annotate at decompose |
| task 6072 (parent edge for unmerged parents) | coexists | a rider spawned from its host already carries the host edge; the verb detects it and adds none; close and release never remove a parent edge | 6072 | pending; annotate at decompose |
| task 6322 (re-dedupe sweep) | coexists | `_execute_combine` skips `rider_of` tasks (the PRD 1 ζ row) | 6322 | pending; annotate at decompose |
| `plans/config-hot-reload-prd.md` | consumes | `riders.*` leaves registered in `RELOADABLE_FIELDS` | this PRD (α) | queued |
| `docs/legibility/design-invariants.md` INV-13 | consumes | δ renders "producer has not written" per series | this PRD (δ) | queued |
| reify alpha-spine Ω 9739 | coexists | a reify rider behind Ω and the holder is still `offerable` (the hold rule, not Ω, decides); its carry adds a third edge, all AND | reify spine PRD | wired |

No reciprocal-ownership statements remain: PRD 2 §5 says "this PRD owns the reservation; that PRD
owns `rider_of`", which decision 8 here now mirrors; PRD 1 §5 names this PRD as the owner of
`promote(by='rider')`'s caller and of "held pool ∩ host lock set", which §4.1 and §4.4 hold.

## 6. Decomposition plan

Overlay bands: 300–1,500 LOC and ≤ 10–12 files per leaf. Labels are Greek; ids are assigned at
decompose. Same-file leaves are serialized by edges (α, β, γ all touch `workflow.py` and
`rider_carry.py`).

| label | title | modules | observable signal | prereqs |
|---|---|---|---|---|
| α | Rider offers in observe mode: pure selector, config, events, rider ledger | `orchestrator/src/orchestrator/riders.py` (new: `HostView`, `RiderOffer`, `offerable`, `select_rider_offers`), `config.py` (`RidersConfig`, `RELOADABLE_FIELDS`), `defaults.yaml`, `event_store.py` (`rider_offered`, `rider_offers_skipped`), `workflow.py` (call after `_reconcile_plan_scope` success; `HostView` from the scheduler snapshot; `_module_cache` reuse for pool keys), `shared/src/shared/rider_ledger.py` (new: `RiderLedger`, `DeferredPromotion`, registration), `fused-memory/.../middleware/rider_carry.py` (new: `apply_rider_verdicts` observe arm, `RiderApplyResult`), `server/tools.py` (verb), `docs/task-authoring.md` §8, tests ×3 (`riders.py`, `rider_ledger.py`, `rider_carry.py`) — 12 files | sketch rows 1, 20, 21 on a live df dispatch in observe mode: `rider_offered` rows with rank and mode, `rider_ledger.offers` on the offered held tasks, and a `reload_config` of `riders.max_offers` reported `applied` | PRD 1 ε 8417 (`followup_hold`) |
| β | Architect rider pass in shadow and offer modes: prompt builder, plan tools, typed verdicts, carry writes, write-authority refusal | `orchestrator/src/orchestrator/plan_riders.py` (new: `RiderVerdict`, keys, reasons), `mcp/plan_tools.py` (`accept_rider`, `decline_rider`, `add_plan_step(rider_id)`, `_PlanField` / `_COLLECTION_SCHEMA_TOOL` facts, carry-forward keys beside `DESCOPE_PLAN_KEYS`), `agents/roles.py` (two tool names in `_PLAN_CREATOR_TOOLS`; implementer rider-step sentence; plan render tag), `agents/briefing.py` (`build_rider_offer_prompt`; plan render `[rider <id>]`), `workflow.py` (`_run_rider_pass`, one-shot flag, snapshot/restore, typed `modules_changed` path, shadow sampling, `rider_pass` event), `rider_carry.py` (offer and shadow arms: re-validation, `merge_items`, `Riders:` block, host edge with 6072 detection, `rider_of`, `carried_at`), `task_interceptor.py` (`RiderWriteAuthorityError` on submit and update), `event_store.py` (`rider_pass`, `rider_pass_failed`, `rider_verdict`, `rider_carried`), tests ×4 — 12 files | sketch rows 2, 2s, 3–7b, 14–18 through a df dispatch: in `mode: shadow`, `rider_pass{mode: shadow}` rows with `cost_usd` and `turns` and an unchanged store; in `mode: offer`, a plan with a rider-tagged step, `_riders` verdicts, the rider's `rider_of` and `carried_at`, the host's merged items and `Riders:` block | α, PRD 1 β 8414 (`merge_items`, `_format_followup_items`) |
| γ | Deliverability stamp, close and release in the status transition, carry sweep, integration gate | `workflow.py` (`_reconcile_rider_carry` at execute → review, after `_replan`, after a narrowing commit; untouched `added_files` taken back through the merge gate's not-touched helper), `rider_carry.py` (`stamp_rider_deliverability`, `release_rider`, `close_or_release_riders_for`, `reconcile_carries`, `ReconcileReport`), `rider_carry.py::rider_promotion_guard` registered explicitly in `fused-memory/.../server/main.py` beside the other startup registrations through PRD 1 ζ's `followup_hold.register_promotion_guard` (γ owns the guard and `deferred_promotion`; ζ's hook is its companion correction and the only ζ symbol γ imports), `task_interceptor.py` (`_apply_status_transition` host hook for `done` and `cancelled`, awaited inside the host's `_write_lock` unit, release-on-failure; `_execute_combine` skip), `fused-memory/.../server/main.py` (sweep at start after PRD 1's backfill, then periodic), `config/schema.py` (`riders.carry_sweep_interval_secs`), `server/tools.py` (`release_rider` human-callable; `stamp_rider_deliverability`), `event_store.py` (`rider_released`, `rider_close_failed`, `rider_closed`, `rider_reconciled`), `fused-memory/tests/integration/test_followup_riders.py` (new; drives public interfaces only; the directory is PRD 1 ι 8420's home, create it if γ lands first), `plans/cancel-with-children-prd.md` §4.2 one-sentence seam, `OPERATIONS.md` (riders section: modes, the `KILL` line, `release_rider`, the sweep) — 11 files | every §4.6 row passes against the integrated tree (rows 8–13s are this leaf's own, 19 shared with PRD 2's gate); any row failing is a RED gate | β, PRD 1 ζ 8418 (`promote(by='rider')`) |
| δ | Observation report, delayed milestone 14 days after γ | `scripts/followup_riders_report.py` (new, read-only over tasks.db and runs.db; matched-pair verify delta by origin class and lines tercile), its test, one dated report under `plans/portfolio-review-2026-11-01/followups/` | the report prints, per project: offers / accepts / declines by reason / `no_verdict` per dispatch, split by mode and host class; the shadow-mode pass cost (USD, turns, duration p50 / p90) and accept rate, then the offer-mode rows, beside the avoided fixed charges, with the observe-mode `_replan` analogy printed only while no shadow row exists; host verify count and merge-fail rate for rider hosts against matched controls with n and the `KILL` verdict at +0.5 verifies per landing, n ≥ 30; rider-attributed `scope_violation` count by emitter; rider-attributed lines from `mark_step_done` commits; riders closed via host per month against the §2 bound (df 66, reify 55) and releases by reason; held-pool age, carried riders by age and the `host_stalled` count; each series "producer has not written" when its event count is zero | γ (`metadata.milestone` delayed 14 d) |

Companion corrections at decompose: 8418's description gains the generic
`register_promotion_guard` hook of decision 11 (no rider vocabulary; γ is the only registrant and
carries no edge from 8418 beyond the one §7 already lists); PRD 2's draft
gains the two sentences of decision 8; annotate 3260 (no rider path writes `metadata.files`);
annotate 6072 (a parent edge doubles as the host edge; close and release never touch it); annotate
6322 (`_execute_combine` skips `rider_of`); add the one-sentence rider seam to
`plans/cancel-with-children-prd.md` §4.2 (γ carries the edit).

G7 walk (walked in author mode; blocking at decompose, re-walked there): `holds-owned-and-
bounded`: a carried rider's owner is its host (`rider_of` + edge, machine-readable) and its bound
is `RIDER_RESERVE_DAYS` from `carried_at`, enforced by `reconcile_carries` on a periodic tick
(decision 18); the hold a released rider returns to is PRD 1's, under PRD 1 ε's recorded waiver;
no waiver is recorded in this PRD. `contracts-machine-checked`: `RidersConfig`, `RiderLedger`,
`DeferredPromotion`, `RiderVerdict`, `DECLINE_REASONS`, `RiderApplyResult`, `ReconcileReport`
and the typed plan-tools refusals; no verdict lives in prose. `corroborate-before-acting`:
`apply_rider_verdicts` re-reads the rider under `_write_lock` and demotes a stale accept
(decision 5); the close reads `deliverable_at` from the store, never a plan file. `storm-escape-
required`: `reconcile_carries` emits one `rider_reconciled` per run, not per rider; the rider pass
is one-shot per plan, sampled in shadow, and capped at `max_offers`. `no-lockstep-duplication`:
one offer-side predicate (`offerable`) and PRD 2's one packer-side `exclusion()`, agreeing by the
`rider_of` guarantee rather than by two copies of a reservation; one `Riders:` renderer; one
close path (`close_or_release_riders_for`) shared by the transition hook and the sweep. `status-
matches-liveness`: a rider is written `done` in the same unit as its host, before the host row
flips, and a rider that cannot be closed is released in that unit rather than left carried by a
done host (decision 10); a cancelled host never leaves a rider edge-linked for the descendant
sweep; the stall clock reads the host's live status, so an in-progress host is never treated as
stalled (decision 18). `loop-thread-occupancy-bounded`: the selector
is set algebra over the scheduler's in-memory snapshot with `_module_cache` keys, O(pool) per
dispatch (df 1,400); the store verbs are one locked write unit each; the sweep pages
`find_tasks_by_metadata('rider_of')`. `one-fact-one-home`: the host edge is the authority for
carried, `rider_of` the index (decision 12); the reservation constant has one home, in PRD 2.
`guards-exercise-behaviour`: the gate drives real dispatches, plan-tools calls and a hand
`set_task_status`; the authority refusal is asserted by submitting the key. `no-silent-fail-
soft`: `no_verdict`, `stale_offer`, `pass_invalid`, `records_lost`, `rider_offers_skipped` and
`rider_pass_failed` make every degraded outcome a record. `exceptions-owned-or-ratified`: none
added. `readers-prove-their-producer`: δ (decision 16). `structured-facts-at-failure`: every
refusal and release carries a reason enum, never a message to scrape. `sampler-vocabulary-one-
table`: not touched.

## 7. Pre-conditions

- PRD 1 leaves 8414 (β: `followup_items`, `merge_items`, `legacy_item`,
  `_format_followup_items`), 8417 (ε: `followup_hold`, the hold rule) and 8418 (ζ:
  `promote(by='rider')`, plus the decision 11 deferral as a companion correction) are the hard
  prerequisites and are wired as out-of-batch edges: α depends on 8417, β on 8414, γ on 8418.
  None had landed on 2026-11-04.
- PRD 2 P0/P1 (`followup_packing.py` with `RIDER_RESERVE_DAYS`, `PackExclusion`, `exclusion()`)
  land before γ: γ imports the constant for the carry clock and row 19 drives PRD 2's exclusions.
  Wired as an out-of-batch edge from γ to PRD 2's P1 at decompose.
- No other novel substrate: every mechanism extends a site named in §2 (G3 verified against main
  `94d620f988`, symbol by symbol, 2026-11-04). `_reconcile_scope_locks`,
  `build_plan_tightening_prompt`, `_replan`, `_create_plan` and `DESCOPE_PLAN_KEYS`,
  `record_descoped_step`, `plan_descope.DescopedStep`, `mark_step_done`, `_PlanField`,
  `_COLLECTION_SCHEMA_TOOL`, `_apply_status_transition`, `_sweep_cancelled_descendants`,
  `AgentResult.turns`, `stamp_pending_since` and `_NO_TASK_STATUS_WRITE` were each located by
  grep.
- γ verifies that `done_provenance_authz.py` authorises a rider's `done/merged` write under the
  identity of the host's own transition (the rider has no claimant and no branch); if the
  authoriser keys on those, γ adds the in-transaction rider close to its authorised set rather
  than widening any caller's rights.
- Cancel-with-children (8142–8148) is not a prerequisite: carry-then-close needs only the edge
  it already enumerates, and decision 10 covers the pre-enforcement window.
- `riders.enabled` defaults false and `mode` defaults `observe`; the flips to `shadow` and
  `offer` are config edits after δ's first report, or earlier by Leo on the shadow-mode
  measurements (decision 15).

## 8. Out of scope

- The packing sweep itself (PRD 2), its reservation and its exclusions; this PRD only fixes the
  seam (§5) and the `rider_of` guarantee the packer relies on.
- Riders for simple-path hosts, deterministic hosts, or hosts dispatched without an architect
  plan.
- A non-subset rule (one-key, `max_keys`, test-or-doc-only, any-overlap): the measured pool and
  the scope-escalation argument reject them (PRD3-R §0, §2.1); no `riders.rule` knob exists.
- Cancel-at-accept, `superseded_by` stamps on riders, and any un-cancel transition.
- A rider that is itself a bundle of more than `MAX_FOLLOWUP_ITEMS - host items` items: refused
  at accept, never split.
- Automatic flipping of `riders.mode` on the kill threshold (decision 14).
- Per-item closure of a host's bundle (PRD 1 §8).
- Hooking `Scheduler.mark_done` or `merge_finalized`: neither is a chokepoint (§2).
- Changing `metadata.files` ownership (task 3260), the lock table, or the scope-boundary prompt.

## 9. Open questions (tactical)

1. **Pending-host snapshot for the PRD-last rule.** The selector needs pending hosts' key sets;
   the scheduler's `_module_cache` covers hosts it has seen, and declared files under-declare on
   44% of df dispatches. **Suggested resolution:** α derives from `metadata.files` for cache
   misses, caps the scan at the medium+ pending set (df 492, reify 963), and δ reports the
   PRD-host share of offers so the rule's error is visible. Decide in α.
2. **Exact call sites of `_reconcile_rider_carry`.** It must run before the completion judge
   reads the plan and after `_replan` returns. **Suggested resolution:** the top of the review
   phase entry, the tail of `_replan`, and the narrowing commit path in `_run_plan_tightening`.
   Decide in γ.
3. **Shadow sample default.** 0.2 of offer-bearing dispatches reaches n = 30 shadow passes a
   week only at about 170 offer-bearing dispatches a week, which is an assumption about df's
   dispatch rate, not a measurement; δ prints the actual offer-bearing dispatch count beside n,
   and the sample is raised (to 1.0 on sc/scp) when a week yields fewer than 30. Decide at the
   `shadow` flip.
4. **Observe-mode fallback denominator.** `_replan` runs only after a review sends work back, so
   its sample may be small on sc/scp; δ reports n and falls back to df's median when a project's
   n < 10, and drops the line once a shadow row exists. Decide in δ.
5. **Human release.** `release_rider(reason='human')` is exposed as an MCP verb; whether the L2
   watcher skill should name it in its runbook is an `OPERATIONS.md` sentence in γ.
6. **Sweep interval.** 3,600 s is a placeholder bound for a clock measured in days; γ may lengthen
   it to the fused-memory reconciliation trickle's cadence if that loop already exists for PRD 2.

## Appendix A. G1 / G3 / G6 self-check

- G1: every mechanism in §1's table names a consumer wired on main or a leaf in this batch; PRD 2
  is named as a consumer of `rider_of` and the ledger, never as the only one (the selector, the
  interceptor hook and δ consume them here).
- G3: no novel substrate beyond PRD 1's three filed leaves and PRD 2's P1, each a hard edge in §7;
  the cancel-at-accept un-cancel transition the scout flagged as unverified is not used; the
  `done_provenance_authz` question is a γ verification with a named fallback, not an assumption.
- G6: one numeric threshold, the kill line at +0.5 verifies per landing, n ≥ 30; basis: S4 §5
  measured 2.2 → 3.2 task verifies per landing across lines terciles, so +0.5 is half the tercile
  spread and well above the per-landing noise the matched-pair design removes; it is a stopping
  rule, not an accuracy bound. The monthly bound δ reports against (df 66, reify 55) is derived
  from the ruled no-cap shares (28.9% / 36.4%) on the memo's flow proxy. Rejection-style signals
  (rows 4, 5, 7, 16, 18, 20) name the mechanism that fires: typed plan-tools refusals (β), the
  typed `modules_changed` path (β), `RiderWriteAuthorityError` (β, interceptor-level), the
  `enabled`/`mode`/`exclude_prd_hosts` gates (α), and the `_plan`-only call site (α).
  Reader-shaped δ renders the no-producer state. End-to-end row 8's capabilities (promotion,
  done provenance `merged`, the in-transaction close, the persisted deliverability stamp) are all
  upstream of or inside γ; row 11 no longer asserts that `found_on_main` implies no execute ran.

## Critic fold-in (2026-11-04)

`task-shape-seats/PRD3-C.md` found eight answer-changing errors and fourteen lesser ones, all
applied with the lead's decisions: the close and release moved from a `Scheduler.mark_done` hook,
which hand carries and the steward bypass, into `TaskInterceptor._apply_status_transition` inside
the host's own `done` or `cancelled` unit, riders first (findings 1, 2, 5, 6); deliverability is
now stamped to the store at execute → review, after `_replan` and after narrowing, so
`found_on_main` after a full execute closes riders and no close reads a plan file (finding 6);
the promotion race is closed by a PRD 1 ζ deferral while `rider_of` is set, filed as a companion
correction to 8418 (finding 2); the reservation constant and predicate are PRD 2's
(`followup_packing.RIDER_RESERVE_DAYS`, `exclusion()`), this PRD keeping `rider_of`, the ledger
and the offer-side `offerable` (finding 3); the carry gained the same 14-day clock, owned by the
periodic `reconcile_carries` sweep, so the β G7 waiver is gone (finding 8); a `shadow` mode
measures the pass's cost and accept rate before anything is carried (finding 7); the bound is the
ruled no-cap figure, df 66 / reify 55 a month, 82% / 64% and 90% / 73% corrected (finding 4);
and the lesser items: `stamp_pending_since`'s home, the moot plan-lock question, `AgentResult.turns`,
`_replan` as a third rewriting call with `step_removed`, `create_plan` carry-forward beside
`DESCOPE_PLAN_KEYS` with `records_lost`, the typed `modules_changed` path replacing an assert, α
at 12 files with the authority refusal moved to β, `rider_reconciled` and the two result shapes in
§4, `_PlanField` / `_COLLECTION_SCHEMA_TOOL` registration in β, `int` ids, the under-declaring
PRD-last comparison named, `_replan` as the cost analogue, row 12 rewritten, and no hook on either
`mark_done` definition. Header sha re-verified at `94d620f988`.

Delta pass 1 (same day) found three more answer-changing items and seven lesser, applied with
the lead's decisions: the close is total, a rider that cannot be closed is released inside the
host's unit with `rider_close_failed` and only the lock unit raising fails the host, so a landed
host can never become a false-blocked task over a rider (D1; decision 10, rows 8f, 8x, the
invariants); PRD 1 ζ learns nothing about riders, 8418's companion correction is the generic
`followup_hold.register_promotion_guard` hook and γ alone registers the rider guard and owns
`deferred_promotion`, with the §5 ζ row reading "ζ provides the hook; γ owns the guard" and the
"ζ, else γ" clause removed from §6 (D2; decisions 7, 11); the carry clock fires only when the carry
is ≥ 14 days old and the host is not `in-progress`, so an executing, re-verifying or
merge-queued host keeps its riders at any age (D3; decision 18, rows 13s, 13i); and the lesser
items: `release_rider` re-validates `rider_of == host_id` under the lock and no-ops otherwise (row
13r), row 6 restored to citation with a human twin, the rider's `merged`
kind under a `found_on_main` host stated and reasoned, one commit resolution per host reused for
every rider, `close_or_release_riders_for` made `async`, δ printing the real dispatch count beside
the shadow n, and the status line's mechanism count. Delta pass 2 (D4) narrowed `release_rider`'s
corroboration to `rider_of == host_id` alone, moving the `status == pending` test to the
`host_stalled` arm, so a rider a human deferred still has `rider_of` and the host edge cleared
when its host lands or fails (rows 8f, 13h, 13r), and the promotion guard is registered explicitly
in `server/main.py` rather than at `rider_carry.py` import time.

## Decompose record (2026-11-04)

Decomposed against main `c393a67c54` by the PRD 3 decompose seat. The manifest and sidecar were committed
at `8bba1402ab` and the stamped sidecar at `59ac0ac4ba`, both as
`plans/followup-riders-prd.capability-manifest.{md,yaml}`, with `signal_leaves: [γ]`. There are 41 grep
checks (no `script`, no `path`) and 5 manual ones. The local polarity lint returned 0 findings.

| label | id | prereqs (edges) | prio |
|---|---|---|---|
| α | 8438 | 8417 (PRD 1 ε) | high |
| β | 8439 | 8438, 8414 (PRD 1 β) | high |
| γ | 8440 | 8439, 8418 (PRD 1 ζ); **PRD 2 P1 pending** | high |
| δ | 8441 | 8440; `milestone {delayed, 1209600}` (clock starts at γ's landing) | low |

Edges: 3 intra-batch, 3 out-of-batch. Every task is `task_kind="normal"` and pending. No task depends on
a blocked task. `commit_planning` stamped α, β, γ and δ with no missing labels or errors.

**Pending edge.** PRD 2 (`plans/followup-packing-sweep-prd.md`) is not committed or decomposed, so the
§7 edge γ → PRD 2 P1 is not wired. Add `add_dependency(8440, <P1 id>)` once PRD 2 is filed. Until then,
the manifest's γ carry-clock row stays OPEN, and 8440's description tells γ to escalate if
`RIDER_RESERVE_DAYS` is absent.

**Resolved at decompose** (no contract change):

1. No leaf owned turning observe mode on (`defaults.yaml` keeps `enabled: false`), so δ's observe series
   could never be written. α (8438) now commits `riders: {enabled: true, mode: observe}` to
   `dark-factory-orchestrator.yaml`. The `shadow` and `offer` flips stay Leo's. Reify needs the same stanza
   in its own config, by an operator; until then δ's reify series reads "producer has not written".
2. Name corrections:
   - the narrowing pass is `TaskWorkflow._try_narrow_plan` (§2, §3 decision 9, §9 question 2 say
     `_run_plan_tightening`);
   - `RiderWriteAuthorityError` subclasses `TaskmasterError` in `backends/task_backend_errors.py`, because
     there is no `TaskBackendError`;
   - in `_plan`, `lock_plan` runs before `_reconcile_plan_scope`, so decision 1's offer call sits after the
     lock, which is harmless (§2).
3. Design notes carried into the descriptions:
   - β samples shadow mode with a stable digest of `(run_id, host)`, not the per-process salted `hash()`;
   - α runs the PRD-host check only on subset-passing riders (INV-8);
   - γ runs its git calls through the non-blocking provenance path (INV-8).

G7: no waivers. §6 records none, and the re-walk found no unresolved hit.

**Companion corrections:**

- 8418's description now carries the generic `register_promotion_guard` hook. It uses no rider
  vocabulary and has zero registrants; 8440 is the only registrant.
- 3260, 6072 and 6322 carry riders notes in their `details`.
- Left as doc edits for the lead:
  - the one-sentence rider seam in `plans/cancel-with-children-prd.md` §4.2. γ's description says it may
    already be present;
  - PRD 2's two decision-8 sentences.
