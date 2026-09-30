# Capability manifest — info-l0-disposition-router-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized), verified against
working tree `a24b1fb665` (2026-09-08). YAML sidecar twin:
`info-l0-disposition-router-prd.capability-manifest.yaml`. Anchors are cited
by symbol; every `delivered_check` is a pattern grep, never a line.

G7 walk (decompose Step 2.3) recorded at the end. No waivers.

## α — Disposition vocabulary, pin helper, mechanical-role registry, event type (intermediate)
- `RESOLUTION_CLASSES` is a one-place frozenset validated at three sites → `escalation/models.py::RESOLUTION_CLASSES`; validators `EscalationQueue.resolve`, `EscalationQueue.submit_resolved`, `server.py::resolve_issue` — PASS
- `classify_pins` exists, pure, info → NON_PINNING at any level → `escalation/pins.py::classify_pins`, precedence link 1 — PASS
- role-registry precedent → `escalation/classify.py::_REAPER_SWEEP_RESOLVERS` — PASS
- `EventType` is a StrEnum with name == value and a payload-shape comment convention → `orchestrator/event_store.py::EventType` (`stale_l0_strand_dismissed` exemplar); `EventStore.emit` never raises — PASS
- config keys are hot-reloadable green-tier leaves → `orchestrator/config.py::OrchestratorConfig` (`orphan_l0_timeout_secs` sibling pattern) — PASS
- deliverable (not premise): `escalation/disposition.py::route_info_l0` does not exist yet — built by α — PASS (deliverable)

## γ — Severity-aware gating + all-info L2 refusal + watcher rung (leaf)
- gating predicate to narrow → `orchestrator/workflow.py::_is_gating_escalation` (four call sites: merge-entry gate, post-wait re-check, post-implementer, post-debugger) — PASS
- escalated-wait short-circuit to narrow → `orchestrator/workflow.py::TaskWorkflow._wait_for_resolution` (`BORN_AT_L2_SEVERITIES or level >= 2` filter) — PASS
- pin helper upstream → `producer:α` (`escalation/pins.py::is_queue_handoff`) — PASS (producer upstream)
- refusal insertion point → `escalation/server.py::promote_to_l2`, after `derived = _derive_l2_severity(...)`; structured error precedent `{'error', 'code': 'level_forbidden'}`; no existing `'refused'` status — PASS
- explicit-severity escape hatch preserved → `promote_to_l2` validates an explicit `severity` before the derive branch (task 3976) — PASS
- rejection-mechanism (G6 branch 4): "an all-info silent promote is refused" is BUILT by γ and pinned by boundary #10 as an executed test → PASS (test-bound)
- rejection-mechanism (G6 branch 4): "an info L2 does not gate merge entry" is observed by boundary #7 executing the real gate → PASS (test-bound)
- watcher rung home → `skills/escalation-watcher-auto/SKILL.md` §"Severity of a promoted L2" and the "When unsure, PROMOTE" rule — PASS
- reason text site → `orchestrator/workflow.py` `_mark_blocked('Steward re-escalated to human', ...)` at the merge-entry gate and `run()`'s ESCALATED branch — PASS
- 3541 boundary: recovery-veto sites are NOT touched here → task 3541 item (1); item (3) leaves `_is_gating_escalation` to this PRD — PASS (ownership recorded in both)

## δ — The router: exit finalizer, reaper info branch, startup routing, status-info aggregates, curator convert leg, D9 hold, analytics bucket (leaf)
- exit choke point runs shielded on every exit before `lock_released` → `orchestrator/workflow.py::TaskWorkflow._on_terminal_cleanups` executed in `workflow_types.py::CancellationScope.supervise` `finally`; `lock_released` emitted from `scheduler.release` in `Harness._run_slot` `finally` — PASS (wired)
- post-merge tail to narrow → `orchestrator/workflow.py::TaskWorkflow._merge_and_finalise` (`_ensure_steward_started` + `_await_steward_completion(skip_if_idle=True)`); `_ensure_steward_started` reads `get_by_task(level=0)` with no severity filter — PASS
- suggestion-triage rationale is dead → `TaskWorkflow._escalate_suggestions` reachable only under `if self.mcp is None` — PASS (verified)
- in-workflow blanket dismissals to make severity-aware → `TaskWorkflow._mark_blocked` fall-through, `StewardInterrupted(wip=True)` paths, `steward_wait_timeout` — PASS
- reaper info branch site → `orchestrator/harness.py::Harness._reap_orphan_l0_escalations` (info branch before the `has_open_l1` dedup; blocking path byte-identical) — PASS
- startup amnesty site → `orchestrator/harness.py::Harness._dismiss_stale_escalations` → `EscalationQueue.dismiss_all_pending` — PASS
- redirect-safe curator client exists and is wired → `orchestrator/scheduler.py::Scheduler.dispatch_tool` → `mcp_call(f'{memory_url}/mcp', ...)`, used by `harness.py` today — PASS (wired; the five raw `/mcp/` POSTs of task 4023 are NOT used)
- curator intake is idempotent per escalation and accepts free `spawn_context` → `fused_memory/server/tools.py::submit_task` (planning_mode=False → `{'ticket'}`), `middleware/task_interceptor.py::_check_escalation_idempotency`, `spawn_context` default `'manual'` (free string) — PASS
- ticket follow-up primitive → `fused_memory` `resolve_ticket` — PASS
- record stamp fields for the D9 hold → `escalation/models.py::Escalation.triaged_by`, `.triage_note`, `.triaged_at` — PASS
- mechanical info filers enumerable → six `severity='info', level=0` sites (roles `orchestrator-starvation-watchdog`, `orchestrator-merge-skew-tripwire`, `orchestrator-no-landings-breaker`, `orchestrator-warm-base-hard-down`, `orchestrator-verify-host-monitor`, `orchestrator-offline-lane`) + `harness.py::_is_done_step_commit_orphan` + `_is_scope_divergence_orphan` — PASS
- aggregate dedup primitive → `EscalationQueue.has_open_l1(sentinel, category=)` (recovery-veto-streak precedent in `recovery_emission.py::emit_recovery_veto_streak_escalation`) — PASS
- analytics bucket site → `dashboard/src/dashboard/data/escalation_analytics.py` (buckets `benign`/`actionable` today) — PASS
- vocabulary, event type, pure router upstream → `producer:α` — PASS (producer upstream)
- gating narrowed upstream (so a routed task can merge) → `producer:γ` — PASS (producer upstream)
- storm fixture (G6 branch 3): 489 `orchestrator-starvation-watchdog` records → exactly one aggregate per class per sweep; capability delivered by δ itself, asserted by boundary #11 — PASS (test-bound)

## β — Reviewer sees notes and dispositions them via the verdict (leaf)
- verdict schema site → `orchestrator/mcp/verdict_tools.py::submit_review_verdict` + `_submit_review_verdict` (`reviewer, verdict, issues, summary`) — PASS
- contract prose site → `orchestrator/agents/roles.py::_REVIEWER_CONTRACT_TEMPLATE` — PASS
- carrier through the review mirror → `orchestrator/artifacts.py::ReviewAggregation` / `TaskArtifacts.aggregate_reviews` (reads `verdict`, `issues` only today — new key must be carried) — PASS
- application point → `orchestrator/workflow.py::TaskWorkflow._execute_verify_review_loop` immediately after `reviews = await self._review(...)` — PASS
- briefing extension → `orchestrator/agents/briefing.py::AgentBriefing.build_reviewer_prompt` (`context` unused by its sole caller `_run_reviewer`; `# Amendment Re-Review Scope` appended-block precedent) — PASS
- filtered read → `escalation/queue.py::EscalationQueue.get_by_task(task_id, status=, level=, agent_role=)` — PASS
- exactly one reviewer holds the channel → `roles.py::ALL_REVIEWERS = [REVIEWER_COMPREHENSIVE]` — PASS
- reviewer cwd is the worktree → `_run_reviewer` → `_invoke(role, prompt, self.worktree)` — PASS
- lost-verdict backstop (G6 branch 3): a discarded verdict (`_salvageable_verdict_payload`) loses dispositions; the note is routed at exit by δ → `producer:δ` upstream — PASS
- resolution classes upstream → `producer:α` — PASS

## ε — Docs and companion corrections (leaf, docs)
- doc homes exist → `ARCHITECTURE.md` §3.6/§6, `OPERATIONS.md`, `plans/escalation-repend-state-machine-prd.md`, `plans/info-l0-blocking-l2-spurious-escalation-analysis.md` — PASS
- describes landed behaviour → `producer:γ`, `producer:δ` upstream — PASS

## ω — Integration gate: boundary sketch executed end-to-end (leaf)
- all twenty boundary rows have producers upstream → `producer:β`, `producer:γ`, `producer:δ` — PASS
- fixtures replay real code paths (INV-10) → `orchestrator/tests/test_orphan_l0_reaper.py`, `test_steward.py`, `test_workflow_escalated_steward_stall.py` are the existing harnesses to extend — PASS
- rows #13 and #18 pin unchanged behaviour (blocking orphan promotion; deliberate `park`) → existing tests `test_aged_orphan_l0_promoted`, `test_live_holder_l0_not_promoted` stay green — PASS

## G7 walk (every task, dark-factory family)
- `contracts-machine-checked`: vocabulary in `RESOLUTION_CLASSES` and `INFO_L0_MECHANICAL_ROLES` frozensets; verdict field declared in the tool signature and contract template; refusal is a structured `{'status': 'refused', 'code'}` — no hit.
- `structured-facts-at-failure`: `info_l0_dispositioned` carries `{escalation_id, class, by, exit_kind, ticket, task_id, decided_at}`; refusal carries `members`, `derived_severity` — no hit.
- `corroborate-before-acting`: the applier re-reads record status before writing; the reaper's liveness deferrals precede routing; ticket state is re-read on each tick — no hit.
- `storm-escape-required`: status-info aggregate per class per sweep; conversions capped per sweep with an aggregate naming the overflow (D8, D12) — no hit.
- `no-lockstep-duplication`: one gating helper (`is_queue_handoff`) consumed at both workflow sites; one router + one applier with three callers — no hit.
- `status-matches-liveness`: the router never writes task status — no hit.
- `holds-owned-and-bounded`: the D9 hold is owned by the reaper, bounded by `info_l0_router_ticket_timeout_secs`, visible on the record (`triaged_by`/`triage_note`), and survives restart because the amnesty routes instead of dismissing — no hit.
- `loop-thread-occupancy-bounded`: the finalizer performs one bounded `mcp_call` and queue writes; no polling wait — no hit.
- `one-fact-one-home`: disposition home is the escalation record; the event and the created task's `escalation_id` are pointers — no hit.
- `guards-exercise-behaviour`: ω executes fixtures through real paths; `delivered_check`s are presence greps on new symbols, not prose — no hit.
- `no-silent-fail-soft`: every router branch names a class; failure → `PromoteWithHint('router_error')`; refusal is a distinct status; unknown mechanical role goes to the curator leg — no hit.
