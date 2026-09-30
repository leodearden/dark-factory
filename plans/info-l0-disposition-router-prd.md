# PRD — Info-L0 disposition router: every open L0 has a consumer, and info never gates

**Status:** active — authored 2026-09-08 (interactive `/prd` session with Leo;
rulings recorded inline in §Resolved design decisions). Implements the
disposition router Leo **ratified on 2026-08-11**
(`plans/info-l0-blocking-l2-spurious-escalation-analysis.md` Addendum B) and
resolves the accreted gaps measured by the 2026-09-08 investigation
(memory `project_orphan_l0_steward_invariant_investigated_2026_09_08`).
**Approach:** **B + H** (blast radius 4 packages — orchestrator, escalation,
skills, fused-memory-adjacent curator intake; ~10 mechanisms; touches the
escalation ladder and the merge gate, both load-bearing). Contract + boundary
tests below.
**Extends/completes:** `plans/escalation-repend-state-machine-prd.md` (D4/C7
gating predicate — narrowed for info), `plans/task-escalation-state-graph-prd.md`
(η = task 3541 keeps the recovery-veto sites; this PRD takes the two
workflow-side predicates 3541 item (3) excluded), the 2026-08-10/11 analysis
doc (Options A/B/C/D → this design). Invariants: INV-7 `holds-owned-and-bounded`,
INV-11 `no-silent-fail-soft`, INV-4 `storm-escape-required`, INV-9
`one-fact-one-home` (`docs/legibility/design-invariants.md`).
**Write-tag:** `agent_id="claude-info-l0-router-prd"`.

**Code anchors** verified against main `a24b1fb665` (2026-09-08). Main moves
fast — cite-by-symbol; re-locate at implementation time.

## Goal (user-observable behaviour)

The escalation ladder's invariant becomes:

> Every open L0 has a designated consumer that has the branch in front of it.
> A **blocking** L0 implies a live steward (existing gating, unchanged). An
> **info** L0 is dispositioned by the reviewer while the workflow lives, by a
> steward if one is already running, and otherwise by the deterministic router
> at workflow exit or when orphaned. No record reaches L1 by age alone, and no
> info record at any level gates a merge, vetoes recovery, or enters the L2
> queue unless an LLM tier explicitly raised its severity.

What Leo observes once this lands:

1. The reviewer's briefing carries a `# Implementer notes (open info escalations)`
   block for its own task, and the reviewer's verdict carries a
   `note_dispositions` list; each note ends `resolution_class` ∈
   {`addressed`, `observation-consumed`} or stays open marked `work`.
2. A task that exits by coalesce park (`merge-deferred`), merged-done, blocked,
   or cancelled leaves **zero** pending info L0s behind; each is closed
   `addressed` / `observation-consumed` / `converted` / `status-info`, or (loud
   fallback) promoted to L1 with its own hint preserved — never as
   `manual_intervention`.
3. A green, reviewed branch whose only open records are info-severity (at any
   level) merges; the merge-entry gate no longer parks it, and `blocked` is
   never written with reason "Steward re-escalated to human" when zero stewards
   ran.
4. `promote_to_l2` with an all-info member set and no explicit severity returns
   `{'status': 'refused', 'code': 'info_only_cluster'}`; the auto-watcher closes
   the members at L1 with a digest. The L2 queue holds no inherited-info
   records.
5. A mechanical-filer storm (the 2026-09-07 starvation-watchdog burst: 489 info
   L0s in two minutes) produces **one** info L1 aggregate per class per sweep,
   not 489 L1s and a blocking L2 pinning 489 tasks.
6. Every disposition is an `info_l0_dispositioned` event
   (`{escalation_id, class, by, exit_kind, ticket, task_id, decided_at}`) and
   the dashboard's escalation analytics bucket `dispositioned` counts them.

## Background

Measured 2026-09-08 over 3,937 escalation records (2026-08-08 → 09-08): a
steward targeted 69% of blocking L0s and **24% of info L0s**; 558 L0s (44% of
non-watchdog) were filed mid-workflow and never saw a steward; all 19 L0s
pending at measurement time were info notes on live workflows with no
steward. The dominant info lifecycle: filed mid-workflow → ignored → workflow
exits → `Harness._reap_orphan_l0_escalations` promotes the text verbatim to L1
on the first tick after `lock_released` (median 77 s) → auto-watcher promotes to
L2 with template options → a human or watcher session closes `close_only`
(77% of 126 terminal reaper-born L2s).

Three different consumer-less exits produced the weekend's three L2s:

- **Coalesce park.** `TaskWorkflow._handle_superseded` writes `merge-deferred`
  and exits the slot with no steward tail (tasks 5050, 4314 run 1).
- **Steward tail killed.** `TaskWorkflow._merge_and_finalise` did start a
  steward for task 5131's info L0 (`_ensure_steward_started` has no severity
  filter), but the merge worker had already flipped the row `done`, so
  `Harness._scan_for_terminal_active_tasks` soft-cancelled the workflow 3 min
  later at zero steward invocations.
- **Info L2 gates the merge.** `workflow.py::_is_gating_escalation` disjunct 3
  gates "any level ≥ 2 … including 'info'" (deliberate, task 1619 / repend
  PRD D4), and `_wait_for_resolution` raises `_StewardReescalated` immediately on
  any L2, so task 4314 run 2's green branch was parked `blocked` by an **info**
  L2 with `steward_invocations=0`.

Two related facts the design relies on. `escalation.pins.classify_pins`
already says an info record is `NON_PINNING` at any level, but the recovery
veto sites still use `bool(open_escalations)` (task 3541 owns their rewire),
so an info L2 vetoes recovery today; the resulting veto streak files a
**blocking** alarm which the auto-watcher folds into the info L2, and the
upward-only floor from task 3976 turns the L2 blocking (4 instances since
3976 landed, including the 489-task storm). And the reviewer never sees an
L0: `AgentBriefing.build_reviewer_prompt` takes no escalation input and the
reviewer role holds only read-only + verdict tools, so "reporting so the
reviewer sees it" in an L0's text has been a false hope.

Prior rulings this PRD keeps: 2026-07-19 (Leo present) — work-loss judgement
on **blocking** orphans belongs in the LLM tier, not a deterministic reaper
predicate; 2026-07-17 — no status-only auto-close of `design_concern`
(flag-not-close, task 3587); 2026-08-11 — consume-on-terminal is acceptable
only for pure observations, work-shaped notes must convert to tasks via the
curator (the router), and `escalate_info` stays reserved for information, not
work items (task 2640).

## Substrate reality check (G3) — verified against working tree `a24b1fb665`

| Assumed capability | Evidence | Verdict |
|---|---|---|
| Reviewer verdict channel accepts a new field | `orchestrator/src/orchestrator/mcp/verdict_tools.py::submit_review_verdict` (`reviewer, verdict, issues, summary`) and `_submit_review_verdict`; consumed by `workflow.py::TaskWorkflow._run_reviewer` → `_review` → `artifacts.py::TaskArtifacts.aggregate_reviews` → `_execute_verify_review_loop`. Exactly one reviewer holds the channel (`roles.py::ALL_REVIEWERS = [REVIEWER_COMPREHENSIVE]`); adjudicator and deep reviewer do not. | PASS — field is additive at four named sites; `_salvageable_verdict_payload` drops a malformed payload whole, so the router at exit remains the backstop |
| Reviewer briefing can carry the notes | `briefing.py::AgentBriefing.build_reviewer_prompt(reviewer_type, diff, context=None, *, amendment_suggestions=None)`; `context` is never passed by its sole caller; the `# Amendment Re-Review Scope` block is the precedent for an optional appended section. `EscalationQueue.get_by_task(task_id, status=, level=, agent_role=)` supports the filter. Reviewer cwd is the worktree. | PASS |
| One exit choke point runs on every slot exit before `lock_released` | `workflow.py::TaskWorkflow._on_terminal_cleanups` list, executed in `workflow_types.py::CancellationScope.supervise`'s `finally` (shielded, every exit kind); `lock_released` is emitted from `scheduler.release` in `Harness._run_slot`'s `finally`, strictly after `workflow.run()` returns | PASS |
| Terminal-status watcher has no steward-tail exemption | `harness.py::Harness._scan_for_terminal_active_tasks` — inputs are `TERMINAL_STATUSES` and `_workflow_cancel_events` only | PASS (motivates placing the router in `_on_terminal_cleanups`, not the tail) |
| Gating predicates are unowned by 3541 | `workflow.py::_is_gating_escalation` (4 call sites) and `_wait_for_resolution`'s `level >= 2` short-circuit; task 3541 item (3) says `_is_gating_escalation` "stays separate"; `Harness._already_landed_dispatch_gate` already consumes `classify_pins` (task 3534) | PASS — this PRD claims both |
| Severity-aware pin classifier exists and is importable | `escalation/src/escalation/pins.py::classify_pins(task_id, records, *, live_claimant, live_claimant_id=None) -> PinReport`; precedence link 1: info → `NON_PINNING`; lazily imported today in `harness.py` and `recovery_emission.py`, never yet in `workflow.py` | PASS |
| Resolution-class vocabulary is one-place, validated | `escalation/src/escalation/models.py::RESOLUTION_CLASSES` (frozenset; validated in `EscalationQueue.resolve`, `EscalationQueue.submit_resolved`, `server.py::resolve_issue`); `EscalationQueue.resolve(..., resolution_class=)` accepts it | PASS |
| `promote_to_l2` has a refusal point and a structured error precedent | `server.py::promote_to_l2`: `derived = _derive_l2_severity(...)` then `effective_severity`; errors already return `{'error', 'code'}` (e.g. `level_forbidden`); no tool returns `'refused'` today | PASS — `status: refused` is a coherent third status |
| Curator intake from the orchestrator that follows the 307 | `scheduler.py::Scheduler.dispatch_tool` → `mcp_call(f'{memory_url}/mcp', …)` (no trailing slash; used by `harness.py` today); `mcp_lifecycle.py` uses `follow_redirects=True`. The five raw `/mcp/` POSTs in `workflow.py`/`merge_queue.py` are the task-4023 defect and are **not** used here | PASS — no dependency on 4023 |
| fused-memory `submit_task(planning_mode=False)` passes the curator and is idempotent per escalation | `fused_memory/server/tools.py::submit_task` → `middleware/task_interceptor.py` returns `{'ticket': 'tkt_…'}`; `resolve_ticket` yields the task id; `_check_escalation_idempotency` keys on `(escalation_id, suggestion_hash)`; `spawn_context` is a free string defaulting to `'manual'` | PASS |
| Event type is a one-site enum edit | `event_store.py::EventType` (StrEnum, name == value); `EventStore.emit(...)` never raises; `event_store` may be `None` | PASS |
| Mechanical info filers are enumerable by role | Six `severity='info', level=0` filing sites (`orchestrator-starvation-watchdog`, `orchestrator-merge-skew-tripwire`, `orchestrator-no-landings-breaker`, `orchestrator-warm-base-hard-down` notice, `orchestrator-verify-host-monitor` recovery notice, `orchestrator-offline-lane` notice) plus the bare-`orchestrator` done-step tripwire (`suggested_action='verify_wip_reconciliation'`, discriminated by `harness.py::_is_done_step_commit_orphan`) and scope-divergence (`_is_scope_divergence_orphan`); `escalation/classify.py::_REAPER_SWEEP_RESOLVERS` is the existing role registry precedent | PASS |
| Startup amnesty is a single site | `harness.py::Harness._dismiss_stale_escalations` → `EscalationQueue.dismiss_all_pending(resolution, strand_age_secs=)` | PASS |

No novel substrate — every mechanism composes existing, wired capabilities.

## Resolved design decisions

- **D1 — Invariant reframed (Leo, 2026-09-08).** Not "a steward per open L0"
  and not "a steward whenever any L0 is open". A blocking L0 implies a live
  steward (unchanged); an info L0 has a designated consumer chain: reviewer →
  already-running steward → router. Steward invocation triggers are unchanged.
- **D2 — Reviewer sees agent-filed info notes and dispositions them via the
  verdict (Leo).** Only `severity='info'`, `level=0`, `status='pending'`,
  own task, agent roles (implementer/architect/debugger/deep_reviewer — never
  the mechanical roles in D8). Dispositions: `addressed` (verified fixed on
  this branch), `no_action` (pure observation), `work` (leave open; the router
  converts). Carrier: a `note_dispositions` field on `submit_review_verdict`,
  applied by the **workflow** deterministically after `_review` returns; the
  reviewer keeps read-only + verdict tools. `work` never closes a note.
- **D3 — Info never gates a merge, vetoes recovery, or short-circuits the
  escalated wait (Leo, "exactly the intended distinction between info and
  blocking").** `_is_gating_escalation` disjunct 3 and the `_wait_for_resolution`
  short-circuit consume `classify_pins` (a record gates at level ≥ 2 iff the
  classifier calls it `QUEUE_HANDOFF`). Task 3541 keeps the recovery-veto
  sites; `_already_landed_dispatch_gate` is already info-safe. Explicit
  severity raises still gate: an LLM tier that judged a cluster blocking wins.
- **D4 — All-info clusters never enter L2 (Leo).** `promote_to_l2` **refuses**
  when `severity is None` and the derived member severity is `info`, returning
  `{'status': 'refused', 'code': 'info_only_cluster', 'members', 'derived_severity'}`.
  The auto-watcher closes the members at L1 with a digest, or passes an
  explicit severity if it judges the cluster blocking (3976's escape hatch
  stays). The server judges nothing; it only enforces the contract.
- **D5 — Router fallback is loud, not a hold (Leo).** A work-shaped note the
  curator cannot take (ticket `failed`/`refused`, MCP unreachable, or the
  ticket unresolved past `info_l0_router_ticket_timeout_secs`) is promoted to
  L1 with `suggested_action='convert_to_task'` and the author's original hint
  preserved in `detail` — never `manual_intervention`. Task 3726 (L1 filing
  grant) is the consumer of those L1s.
- **D6 — Blocking orphan L0s are unchanged (Leo).** The reaper's deferral,
  `has_open_l1` dedup, class arms and verbatim promotion for `severity !=
  'info'` are untouched (2026-07-19 ruling; hollow-done constraint, task 2729).
- **D7 — Router placement.** One function, `escalation.disposition.route_info_l0`
  (pure decision) + one applier, called from three sites: (a) the workflow's
  `_on_terminal_cleanups` entry `disposition_info_l0s` on terminal exits
  (DONE, MERGE_DEFERRED, BLOCKED, CANCELLED — soft or hard); (b) the reaper's
  info branch, replacing the verbatim-promotion fallthrough for
  `severity == 'info'`; (c) the startup amnesty, which routes info records
  instead of dismissing them unread. REQUEUED exits leave info notes pending
  for the next incarnation's reviewer; the reaper backstops if no incarnation
  comes. The post-merge steward tail starts a steward only when a **blocking**
  L0 is pending; its "suggestion triage" rationale is dead in production
  (`_escalate_suggestions` is reachable only when `self.mcp is None`).
- **D8 — STATUS_INFO leg is deterministic per class with one aggregate per
  sweep (Leo).** A registry `escalation/classify.py::INFO_L0_MECHANICAL_ROLES`
  (frozenset of the six info-filing roles + the done-step and scope-divergence
  discriminators) → `resolution_class='status-info'`; per sweep and class, one
  info L1 aggregate (`agent_role='harness-info-l0-router'`, deduped by a
  per-class sentinel through `has_open_l1(sentinel, category=)`), listing the
  dismissed ids. An unknown mechanical role is **not** status-info: it falls to
  the curator leg (fail toward loud, never toward silent dismissal).
- **D9 — `work` conversion is a reaper-owned bounded hold, not a wait in the
  finalizer.** At exit the router files the curator ticket
  (`spawn_context='info-l0-router'`, `escalation_id`, `spawned_from`) through
  `Scheduler.dispatch_tool`, stamps the record `triaged_by='info-l0-router'`,
  `triage_note=<ticket id + filed_at>`, and leaves it pending. The reaper's
  info branch finishes it on a later tick: ticket `created` → resolve
  `converted` citing the task id; `failed`/`refused` or timeout → D5 fallback.
  Lock release is never delayed by curator latency (INV-7: owner = reaper,
  bound = the timeout, visible on the record).
- **D10 — In-workflow blanket L0 dismissals become severity-aware.**
  `_mark_blocked`'s fall-through dismissal, the `StewardInterrupted(wip=True)`
  paths and `steward_wait_timeout` dismiss blocking L0s exactly as today and
  leave info notes for the finalizer router.
- **D11 — Disposition facts have one home.** `resolution_class` +
  `resolution` on the escalation record (ticket/task id in the resolution
  text and `triage_note`); the `info_l0_dispositioned` event is telemetry that
  names the record id; the created task points back via `metadata.escalation_id`
  / `spawned_from` (existing convention, e.g. tasks 4988, 5195).
- **D12 — Storm escape.** Conversions are capped per sweep
  (`info_l0_router_max_conversions_per_sweep`); overflow records are promoted
  under D5 and one aggregate L1 names the overflow. The status leg's aggregate
  is the escape for mechanical storms.
- **D13 — Reason text is truthful.** A merge-entry or escalated-wait exit that
  blocked without a steward invocation writes reason `Gated by open L2 <ids>;
  no steward applicable` — and, post-D3, that path is reachable only for
  non-info L2s.
- **D14 — Vocabulary additions are machine-checked.** `RESOLUTION_CLASSES`
  gains `addressed`, `observation-consumed`, `converted`, `status-info`; the
  dashboard analytics buckets them as `dispositioned`. No new escalation
  category (the aggregate uses `risk_identified`; the prose-only `CATEGORIES`
  refactor trigger is not pulled by this PRD).

## Contract (B+H §1)

- `escalation.pins.is_queue_handoff(record) -> bool` — thin, pure wrapper over
  `classify_pins(record.task_id, [record], live_claimant=True)`; True iff the
  record lands in `queue_handoff`. Consumed by `workflow.py::_is_gating_escalation`
  (disjunct 3 becomes `e.level >= 2 and is_queue_handoff(e)`) and by
  `_wait_for_resolution`'s short-circuit (`severity in BORN_AT_L2_SEVERITIES or
  (level >= 2 and is_queue_handoff(e))`). Disjuncts 1–2 unchanged. No other
  severity predicate is hand-rolled in `workflow.py` (INV-5).
- `escalation.disposition.route_info_l0(record, *, task_status, exit_kind,
  mechanical_roles, reviewer_disposition=None) -> Disposition` — pure. Returns
  one of `Addressed`, `ObservationConsumed`, `StatusInfo(class_key)`,
  `ConvertViaCurator(payload)`, `PromoteWithHint(reason)`, `Defer(reason)`.
  Precedence: reviewer disposition (if the record carries one from D2 that the
  workflow failed to apply) → mechanical role → terminal-subject observation
  (author's `suggested_action`/`detail` says no action, or the subject is
  `done`/`cancelled` and the note is not work-shaped by the curator's own
  admission) → convert → promote-with-hint. `Defer` only for REQUEUED exits.
  Never returns a silent drop; every branch names its class.
- `orchestrator.escalation_router.apply(disposition, record, queue,
  event_store, dispatch_tool)` — the single applier used by all three callers
  (D7). Writes exactly one of: `queue.resolve(..., resolution_class=…)`,
  `queue.submit(L1 with hint)`, ticket file + `stamp` (D9), or nothing
  (`Defer`). Emits `EventType.info_l0_dispositioned` with
  `{escalation_id, class, by, exit_kind, ticket, task_id, decided_at}`. Never
  raises into the caller; a failure is itself a `PromoteWithHint('router_error')`
  (INV-11).
- Reviewer verdict: `submit_review_verdict(reviewer, verdict, issues, summary,
  note_dispositions: list[{escalation_id, disposition: 'addressed'|'no_action'|
  'work', rationale}] = [])`; `artifacts.ReviewAggregation.note_dispositions`
  carries it through the review mirror; applied in
  `_execute_verify_review_loop` immediately after `reviews = await self._review(...)`
  with `resolved_by='reviewer-comprehensive/workflow'`. Unknown ids or
  non-info/non-own records are ignored with a WARNING and an event
  (`class='rejected_disposition'`), never applied.
- Reviewer briefing: `build_reviewer_prompt(..., open_info_notes=[...])`
  appends `# Implementer notes (open info escalations)` — one entry per note:
  id, filer role, filed-at, summary, `suggested_action`, the first
  `info_l0_note_detail_chars` of `detail`, and the instruction that the
  reviewer must disposition each id in its verdict. Absent notes → no block.
- `promote_to_l2`: after `derived` is computed, `if severity is None and
  derived == 'info': return {'status': 'refused', 'code': 'info_only_cluster',
  'members': [...], 'derived_severity': 'info'}`. No record written, no
  dedup fold. An explicit `severity` bypasses the guard (validated upstream as
  today). `add_members_to_l2` is unchanged (upward-only floor stays).
- Reaper (`_reap_orphan_l0_escalations`): for `esc.severity == 'info'`, the
  branch runs **before** the `has_open_l1` dedup and after the liveness
  deferrals; it (i) finishes a D9 hold if the record is router-stamped,
  (ii) otherwise calls `route_info_l0(..., exit_kind='orphan')` and applies.
  Blocking records take the existing path byte-for-byte.
- Startup amnesty (`_dismiss_stale_escalations`): info records are routed
  (`exit_kind='restart'`) before `dismiss_all_pending`; blocking records keep
  the strand sweep. Strand telemetry (task 3172) therefore counts blocking
  strands only — see companion amendment to task 5200.
- Config (all hot-reloadable, `orchestrator/config.py`):
  `info_l0_router_enabled: bool = True`, `info_l0_router_ticket_timeout_secs:
  float = 900`, `info_l0_router_max_conversions_per_sweep: int = 20`,
  `info_l0_note_detail_chars: int = 1200`.
- Ordering invariants: the finalizer entry runs after `stop_steward` and
  before `release_lane`; the router never touches task status; a record is
  resolved at most once (resolve is idempotent on an already-terminal record
  and the applier re-reads status before writing — INV-3).

## Boundary-test sketch (B+H §2) — the ω integration-gate signal

| # | Scenario | Pre | Post |
|---|---|---|---|
| 1 | Reviewer addresses a note | implementer files info L0; reviewer verdict `addressed` | record `resolution_class='addressed'`, `resolved_by='reviewer-comprehensive/workflow'`; event emitted; no reaper L1 ever |
| 2 | Reviewer marks `work` | as #1 with `work` | record stays pending; at merged-done exit the router files a curator ticket, stamps `triaged_by='info-l0-router'`; reaper resolves `converted:<task>` after `resolve_ticket` returns created |
| 3 | Curator refuses | as #2, ticket `refused` | reaper promotes L1 `suggested_action='convert_to_task'`, `detail` starts with the author's text; no `manual_intervention` |
| 4 | Ticket timeout | as #2, ticket still queued after `info_l0_router_ticket_timeout_secs` | same as #3 with reason `ticket_timeout`; hold visible on the record until then |
| 5 | Coalesce park (task 5050 shape) | info L0 pending; merge superseded by coalesce | `_handle_superseded` exit → finalizer routes; zero `harness-orphan-reaper` info L1 for the task |
| 6 | Merged-done + terminal-status cancel (task 5131 shape) | info L0 pending; merge worker flips `done` during the tail | finalizer still runs (shielded); no steward started for an info-only set; record dispositioned before `lock_released` |
| 7 | Info L2 at merge entry (task 4314 run-2 shape) | pending L2 with `severity='info'`; branch green and reviewed | `_is_gating_escalation` false; task merges; no `blocked` row; no `_StewardReescalated` |
| 8 | Blocking L2 at merge entry | as #7 with `severity='blocking'` | gates exactly as today; reason text names the L2 ids and "no steward applicable" |
| 9 | Explicit-severity promote | auto-watcher passes `severity='blocking'` for an all-info cluster | L2 created blocking (escape hatch intact) |
| 10 | Silent all-info promote | members all info, severity omitted | `{'status': 'refused', 'code': 'info_only_cluster'}`; no record; watcher closes members at L1 with digest |
| 11 | Mechanical storm (esc-4851 shape) | 489 `orchestrator-starvation-watchdog` info L0s aged past threshold | 489 records `status-info`; exactly one aggregate info L1 per sweep listing them; nothing promoted individually |
| 12 | Unknown mechanical role | info L0 from a role not in the registry, no workflow | routed to the curator leg, not dismissed |
| 13 | Blocking orphan unchanged | blocking L0, workflow gone, no open L1 | verbatim L1 promotion exactly as today (existing tests unchanged) |
| 14 | BLOCKED exit keeps info notes | blocking L0 + info L0; steward re-escalates; `_mark_blocked` fall-through | blocking L0 dismissed as today; info L0 routed by the finalizer |
| 15 | REQUEUED exit defers | info L0 pending; workflow requeues | note stays pending; next incarnation's reviewer sees it; reaper routes it if no incarnation returns within `orphan_l0_timeout_secs` |
| 16 | Restart amnesty routes | info L0 pending across restart | routed with `exit_kind='restart'`; blocking strands swept as today; no `auto-dismissed` unread info record |
| 17 | Lost verdict | reviewer verdict discarded by `_salvageable_verdict_payload` | no disposition applied; note still pending; routed at exit (backstop) |
| 18 | Deliberate hold unaffected | task parked via `park` / a blocking L2 | untouched by every router path |
| 19 | Storm overflow | more work-shaped notes than the per-sweep cap | cap honoured; overflow promoted under D5; one aggregate names the overflow |
| 20 | Router failure | queue write raises inside the applier | `PromoteWithHint('router_error')` L1 filed if possible; error logged with structured data; task exit unaffected |

## Cross-PRD relationship (G4)

| Other PRD / artifact | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/task-escalation-state-graph-prd.md` (η = task 3541) | shares predicate home | `classify_pins` consumption: 3541 rewires the recovery-veto sites; **this PRD** owns `_is_gating_escalation` disjunct 3 and the `_wait_for_resolution` short-circuit (3541 item (3) explicitly left them separate) | split as stated; companion amendment to 3541 recorded at decompose | queued here (γ); 3541 pending |
| `plans/task-escalation-state-graph-prd.md` (task 3587 moot sweep) | narrows | info moot records are now the router's; 3587 keeps flag-not-close for non-info records and adds `cancelled` + merged-not-done signals | 3587 (amended) | pending |
| `plans/escalation-repend-state-machine-prd.md` (D4/C7) | supersedes for info | disjunct 3 of the gating predicate | **this PRD**; dated addendum note in the repend PRD (ε) | queued here (ε) |
| `plans/info-l0-blocking-l2-spurious-escalation-analysis.md` (Addendum B ratified router) | implements | the router + reviewer leg | **this PRD**; pointer addendum (ε) | queued here |
| Task 3726 (L1 filing grant) | consumes | D5 fallback L1s (`convert_to_task`) and the D8 aggregates | 3726 (unchanged, stays the "L1 half") | pending |
| Task 4023 (307 transport) | none | the router uses `Scheduler.dispatch_tool`; 4023 remains the fix for reviewer suggestions and memory writes | 4023 | pending, behind 5036 by file overlap — not a prerequisite here |
| Task 5200 (strand-sweep threshold) | narrows | strand telemetry counts blocking strands only once info records are routed at startup | 5200 (amended) | pending |
| Task 4988 (held info L0 unreachable) | superseded for dark-factory | D2/D7 give held info L0s a mid-run consumer and an exit consumer; option (b) content preservation is met by the router's records | 4988 (amend: narrow to reify-strand telemetry or close on ε landing) | pending |
| fused-memory curator intake | consumes | `submit_task(planning_mode=False, metadata.spawn_context='info-l0-router', escalation_id=…)` — additive, no vocabulary edit | fused-memory (no change) | wired |

No reciprocal-ownership ambiguity: every code seam lands in this batch or in
3541's, with the boundary written into both.

## Decomposition plan

File-lock spine: `workflow.py` linear γ → δ → β; `harness.py` δ only;
`escalation/` α (models, pins, classify, disposition) then γ (server); skills
γ; docs ε. All code leaves `task_kind="normal"`; `force_full_path=true` on the
god-file leaves (γ, δ, β). Sizing per the overlay bands.

- **α — Disposition vocabulary, pin helper, mechanical-role registry, event
  type** (`escalation/models.py`, `escalation/pins.py`, `escalation/classify.py`,
  new `escalation/disposition.py` (pure `route_info_l0`), `orchestrator/event_store.py`,
  `orchestrator/config.py`). Intermediate; unlocks β/γ/δ. Signal for its own
  tests: `route_info_l0` verdict table over the record classes (boundary #11,
  #12 decision halves); `RESOLUTION_CLASSES` accepts the four new classes at all
  three validators.

- **γ — Severity-aware gating + all-info L2 refusal + watcher rung**
  (`orchestrator/workflow.py` `_is_gating_escalation`, `_wait_for_resolution`,
  reason text D13; `escalation/server.py::promote_to_l2`;
  `skills/escalation-watcher-auto/SKILL.md` "info-only cluster: close at L1 with
  digest, or raise explicitly"). Prereq α. Leaf. Signal: boundary #7, #8, #9,
  #10; operationally, an info-L2 fixture task merges and `promote_to_l2` on an
  all-info member set returns `refused`.

- **δ — The router: exit finalizer, reaper info branch, startup routing,
  status-info aggregates, curator convert leg, D9 hold, analytics bucket**
  (`orchestrator/workflow.py` `_on_terminal_cleanups` + D7 tail change + D10
  severity-aware dismissals; `orchestrator/harness.py` reaper branch + amnesty;
  new `orchestrator/escalation_router.py` applier; `orchestrator/scheduler.py`
  reuse of `dispatch_tool`; `dashboard/src/dashboard/data/escalation_analytics.py`
  `dispositioned` bucket). Prereq α, γ (workflow.py spine). Leaf. Signal:
  boundary #2–#6, #11–#16, #19, #20 in tests; operationally, replaying the
  5050 / 5131 / 4851 fixtures yields zero `harness-orphan-reaper` info L1s and
  one aggregate for the storm.

- **β — Reviewer sees notes and dispositions them via the verdict**
  (`orchestrator/mcp/verdict_tools.py`, `orchestrator/agents/roles.py`
  `_REVIEWER_CONTRACT_TEMPLATE`, `orchestrator/agents/briefing.py`,
  `orchestrator/artifacts.py` `ReviewAggregation`, `orchestrator/workflow.py`
  `_run_reviewer` + `_execute_verify_review_loop`). Prereq α, δ (spine).
  Leaf. Signal: boundary #1, #2 (first half), #17; operationally a reviewer
  transcript shows the notes block and the verdict JSON carries
  `note_dispositions`.

- **ε — Docs and companion corrections** (`ARCHITECTURE.md` §3.6/§6: reaper
  predicate as built, restart routing, the consumer chain; `OPERATIONS.md`:
  holds are explicit (`park` or non-info record), router classes, watcher
  rung; dated addendum in `plans/escalation-repend-state-machine-prd.md`
  (D4 narrowed for info); pointer addendum in the analysis doc). Prereq γ, δ.
  Leaf (docs). Signal: the named sections exist and describe the landed
  behaviour; `delivered_check` greps on the section banners.

- **ω — Integration gate: boundary sketch executed end-to-end**
  (`orchestrator/tests/`, `escalation/tests/`; fixtures replaying the four
  weekend shapes through the real harness/workflow/reaper/server code paths,
  not mocks of the router). Prereq β, γ, δ. Leaf. Signal: all 20 boundary rows
  green as executed fixtures (INV-10), including #13 and #18 pinning that
  blocking behaviour and deliberate holds are unchanged.

Companion task amendments (applied at decompose via `update_task`, text
ratified in-session): 3541 item (3) → "the gating predicate's severity
awareness is owned by `plans/info-l0-disposition-router-prd.md` leaf γ, which
consumes `classify_pins`"; 3587 → drop info-record moot handling, add
`cancelled` subjects and the merged-not-done signal; 4988 → narrow to strand
content telemetry for reify, or close when ε lands; 5200 → strand telemetry
counts blocking strands only.

## Out of scope for this PRD

- Blocking-L0 reaper behaviour, including category-scoping its `has_open_l1`
  dedup (D6; see open question 4).
- The recovery-veto sites (task 3541) and `_already_landed_dispatch_gate`
  (already info-safe).
- Reviewer non-blocking suggestions → tasks (task 4023, the 307 transport) and
  the L1 watcher's filing grant (task 3726) — consumers of this PRD's outputs,
  not built here.
- Promoting `CATEGORIES` from prose to an enum (models.py refactor trigger).
- Any change to `escalate_info`'s filing contract or to which roles may file.
- Retro-closing already-terminal records; the two pending L2s that descend
  from info sources are blocking today and stay with their owners.

## Open questions (tactical; defaults taken)

1. **Ticket timeout default.** `info_l0_router_ticket_timeout_secs=900`
   matches `steward_completion_timeout`. Decide during δ against curator
   queue latency.
2. **Per-sweep conversion cap.** Default 20; the 08-11 corpus puts work-shaped
   notes at ~22% of ~15/day. Decide during δ.
3. **Note detail length in the reviewer block.** Default 1200 chars of
   `detail`; full text reachable by id. Decide during β.
4. **Category-scoped `has_open_l1` for blocking orphans.** 44 of 70 dedup
   dismissals were cross-category; D6 says unchanged. Suggested resolution:
   leave; revisit with 3541 item (4).
5. **Aggregate granularity.** One L1 per (sweep, class) vs one per sweep.
   Default per class so the watcher's digest stays legible.
6. **Reviewer block for the deep reviewer.** It files L0s but holds no verdict
   channel; its notes reach the router only. Suggested resolution: accept.
