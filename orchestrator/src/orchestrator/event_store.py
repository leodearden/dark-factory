"""Append-only SQLite event store for orchestrator observability.

Captures structured events throughout execution — agent invocations, phase
transitions, escalations, waste, and infrastructure events.  All writes are
fire-and-forget: failures are logged but never propagate to callers.

Modeled on fused-memory's WriteJournal pattern.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from shared.sqlite_sync_base import CheckpointResult, apply_full_durability_pragmas_sync

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_events_run  ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(timestamp);
"""


class EventType(StrEnum):
    """Orchestrator event taxonomy."""

    # Agent invocations
    invocation_end = 'invocation_end'
    # Adaptive model routing (plans/adaptive-model-routing-prd.md, task γ) —
    # emitted once per LLM invocation by TaskWorkflow._invoke, immediately
    # after invocation_end.  Payload: the resolved model/effort/budget_usd/
    # max_turns, source_layer (which layer decided — e.g. 'config' or
    # 'policy_rule'), rule_id (the specific rule that fired, if any),
    # rejected (list of alternatives considered and turned down),
    # routing_tier (the current adaptive-retry tier counter), and
    # inputs_digest (a stable hash of the salient resolution inputs).  Pre-ε
    # this records a coarse approximation of source_layer/rule_id (only the
    # existing Rust-heuristic model upgrade is distinguishable from plain
    # config lookup); task ε swaps in the full RoutingDecision.
    routing_decision = 'routing_decision'

    # Workflow phases
    phase_enter = 'phase_enter'
    phase_exit = 'phase_exit'

    # Escalations
    escalation_created = 'escalation_created'
    escalation_resolved = 'escalation_resolved'

    # Waste detection
    waste_detected = 'waste_detected'

    # Infrastructure
    cap_hit = 'cap_hit'
    lock_acquired = 'lock_acquired'
    lock_released = 'lock_released'
    # Durable persist of the plan-tightened lock set on the success branch of
    # handle_blast_radius_expansion. Payload: {files, released, acquired,
    # persisted: bool}. Complements lock_released (reason='plan_refinement')
    # by evidencing the DURABLE write; persisted=False means the in-memory
    # narrowing applied but the metadata write failed (non-critical).
    set_to_plan = 'set_to_plan'
    merge_attempt = 'merge_attempt'
    # The merge worker's POST-rebase verify verdict (task_id-keyed).  data:
    # {runner, merge_sha, passed, duration_ms, attempt, depth, speculative}
    # (depth/speculative always-present, None when absent — task 2340), PLUS
    # two always-present retry-scope keys (task 2837, PRD verify-retry-
    # failed-only D5) derived from spec.verify_env — the D2 failed-only retry
    # contract the orchestrator ITSELF built and shipped to reify, NOT a reify
    # stdout marker:
    #   retry_scope        None | 'failed_only'.  None is unambiguously a full
    #                      green gate; 'failed_only' is a D2-narrowed retry.
    #   retry_subset_sizes None (full/legacy verify) | the per-suite subset
    #                      sizes {run_all, gui, nextest_debug, nextest_release}
    #                      (nextest_* are None if their filter file is absent/
    #                      unreadable — INV-2 honest degrade, never a crash).
    # Both keys are ALWAYS present (None when not narrowed) so the survey's
    # runtime mining (milestone 5254) reads a uniform data.get('retry_scope')
    # and can NEVER miscount a narrowed retry as a full green gate.
    merge_verify = 'merge_verify'
    # A merge-role scoped-verify red that the isolated-rerun-confirm gate
    # (verify.apply_merge_flake_suppression, PRD task α) demonstrated was a
    # CPU-starvation flake and suppressed so the merge could proceed.
    # task_id-keyed; data carries {node_ids: list[str], merge_sha: str,
    # measured_at: ISO-8601 str} — the suppressed pytest node-ids, the merge
    # commit whose verify was suppressed, and when the suppression was decided.
    merge_flake_suppressed = 'merge_flake_suppressed'
    # The branch's OWN pre-merge verify verdict, recorded by the orchestrator
    # workflow VERIFY phase (branch-vs-its-merge-base) — distinct from
    # merge_verify, which is the merge worker's POST-rebase verify (branch
    # already merged onto CURRENT main).  task_id-keyed; data carries at least
    # {passed: bool, base_sha: str, branch: str}.  Consumed by the merge-skew
    # attribution classifier's I5 branch-green fact (merge_disposition.
    # _branch_pre_merge_verify_green).  Task 2381 α adds this member + the
    # reader; task 2383 β emits it at the workflow VERIFY-pass site.  Its
    # ABSENCE is honest, not a bug: non-orchestrator paths (/merge-queue,
    # /unblock, /do) submit branches without emitting it -> classifier
    # degrades to INDETERMINATE by design.
    workflow_verify = 'workflow_verify'
    verdict_parity_ok = 'verdict_parity_ok'
    # Land-time remote-green cross-check (task 2822, fix b) telemetry. The
    # AGREE case reuses verdict_parity_ok above; these two give the divergence
    # and local-infra-inconclusive outcomes their own auditable types.
    # verify_cross_check_mismatch: remote PASS vs local trust-anchor FAIL — the
    # land is withheld (fail-closed), the remote runner quarantined, and a
    # blocking escalation filed. verify_cross_check_inconclusive: the local
    # cross-check runner was unavailable (RunnerUnavailable) so the remote green
    # is trusted (fail-safe) — the land proceeds but the skipped second opinion
    # is recorded.
    verify_cross_check_mismatch = 'verify_cross_check_mismatch'
    verify_cross_check_inconclusive = 'verify_cross_check_inconclusive'
    # data.queue_depth = depth-at-enqueue INCLUDING the newly-queued item
    # (CAS-retry: qsize()+len(urgent)+1; plain enqueue: qsize() after put)
    merge_queued = 'merge_queued'
    # data.queue_depth = remaining depth AFTER removal (excludes the dequeued item)
    merge_dequeued = 'merge_dequeued'
    merge_coalesced = 'merge_coalesced'
    # Periodic queue-depth heartbeat: emitted while items wait in the pipeline.
    # Queue-scoped (task_id=None). Payload: {depth, oldest_age_secs, head_of_line,
    # verify_in_progress}.  Closes the journalctl blind spot during long queue waits.
    merge_heartbeat = 'merge_heartbeat'
    # Emitted when a remote verify host becomes persistently unreachable (after
    # crossing the streak or time threshold).  Payload: {host, reason}.
    verify_host_unreachable = 'verify_host_unreachable'
    # Emitted when a previously-unreachable remote verify host passes health()
    # and is cleared back into the live pool.  Payload: {host, downtime_s}.
    verify_host_recovered = 'verify_host_recovered'
    # Emitted from enqueue_merge_request's terminal done-callback when a
    # MergeRequest future reaches its final state (resolved, cancelled, or
    # exception).  Payload shape: {request_id, branch, state, snapshot_tip,
    # merge_sha}.  state is one of MergeOutcome.status values, 'abandoned'
    # (cancelled future), or 'error' (unexpected exception).
    merge_finalized = 'merge_finalized'
    # Unconditional observability of the generic merge-blocked fall-through in
    # TaskWorkflow._submit_to_merge_queue.  Emitted BEFORE _mark_blocked and
    # ungated by has_open_l1, so `merge_finalized state=blocked` can never again
    # be followed by total silence when an unrelated open L1 suppresses the
    # escalation (task 2757; Reify 5120 RCA RC-2).  task_id-keyed; payload shape:
    # {reason, category, failure_category, cause_hint}.
    merge_blocked = 'merge_blocked'
    speculative_merge = 'speculative_merge'
    speculative_discard = 'speculative_discard'
    # Emitted by verify.run_scoped_verification when the merge gate (the
    # adoptable post-merge verdict path: role='merge' AND is_merge_verify=True)
    # would otherwise return a "nothing to run" TRIVIAL PASS — no source files,
    # empty existing_files (e.g. an ENOENT-clobbered worktree), or an empty
    # command set.  INV-1 (plans/merge-verdict-integrity-prd.md §1, §3.2)
    # forbids a no-evidence PASS from a merge-role verify: the would-be trivial
    # pass is ESCALATED to the project's full merge gate, or — if no full-gate
    # command exists — FAILs loud (VerifyResult category 'merge_no_evidence').
    # task_id-keyed; data carries {reason, resolution, measured_at}.  reason ∈
    # {no_source_files, empty_existing_files, empty_command_set}; resolution ∈
    # {full_gate, loud_fail}; measured_at is an ISO-8601 str.  Only the
    # dispatch-side LocalRunner threads an event_store, so the remote
    # in-worktree path is None-safe (correctness fix still applies; only the
    # event is dispatch-local, mirroring merge_flake_suppressed).
    trivial_pass_escalated = 'trivial_pass_escalated'
    # Contract-currency (INV-2, plans/merge-verdict-integrity-prd.md §1, §3.1,
    # §3.7) telemetry for the RemoteRunner auto-sync-at-dispatch gate.  A
    # RemoteRunner's adoptable post-merge verdict is trustworthy only if the
    # runner executed CURRENT gate logic; incident bb834dd42a is the laptop's
    # Dark-Factory *code* checkout frozen ~5 weeks, rubber-stamping trivial
    # PASSes (966f23a6).  Both are queue/runner-scoped (task_id optional; the
    # sync happens before/around dispatch, not necessarily bound to one task).
    #
    # runner_stale: emitted by RemoteRunner.sync_if_stale when the remote
    # DF-code checkout HEAD differs from the dispatcher's local DF HEAD
    # (staleness detected, BEFORE any sync/pull is attempted).  data carries
    # {runner, local_head, remote_head}.
    #
    # runner_synced: emitted when the remote git state was brought current with
    # the dispatcher — either the DF-code checkout (git pull --ff-only + uv
    # sync, kind='df_checkout') or the project checkout's main ref (mirror
    # force-push after a non-fast-forward, kind='project_main_mirror').  data
    # carries {runner, kind, from_head, to_head, forced}; kind ∈ {'df_checkout',
    # 'project_main_mirror'}; forced is True only on the project-main mirror
    # force-push arm.
    runner_stale = 'runner_stale'
    runner_synced = 'runner_synced'
    # INV-3 chain-intact enforcement (PRD plans/merge-verdict-integrity-prd.md
    # §3.7 new vocab).  Emitted whenever a verdict (PASS or FAIL) is VOIDED
    # because its verified tree no longer chains onto a still-live prefix — the
    # item's base_sha points at a dead commit (failed/ejected/superseded/
    # re-merged since verify dispatch).  A distinct type (not speculative_discard)
    # so the phantom-block-avoided signal is separately countable via the event
    # census.  data: {dead_link, reason ('chain_dead'), point ('dispatch'|'adoption')}.
    verdict_voided = 'verdict_voided'

    # Task lifecycle
    task_started = 'task_started'
    task_completed = 'task_completed'

    # Zero-progress requeue backstop (task 3068, origin incident reify
    # esc-5556-1).  Emitted when a task has been burning dispatch slots with
    # nothing to show for it — a class the requeue cap cannot see by design;
    # see orchestrator/src/orchestrator/zero_progress_requeue.py (module
    # docstring) for why.  This event is its only telemetry.  Both are keyed on
    # the REAL task_id (the escalation uses a synthetic sentinel id, but the
    # events must stay joinable against task_completed).
    # data: {streak, threshold, span_seconds, reason, block_phase}
    zero_progress_requeue = 'zero_progress_requeue'
    # ...and its recovery half, emitted when the streak breaks and the filed
    # blocking L1 is auto-resolved.  data: {streak, resolved}
    zero_progress_requeue_recovered = 'zero_progress_requeue_recovered'

    # Warm-lane session resume (task γ, plans/warm-lane-session-resume-prd.md).
    # Emitted by the _run_slot eligibility guard, only when a recovered agent
    # session was present for the dispatched task:
    #   session_resume          — an eligible session was injected as --resume.
    #   session_resume_fallback — an ineligible session degraded to fresh
    #                             dispatch; data.reason ∈ {stale, no_transcript,
    #                             reseeded}.
    #   session_resume_capped   — resume_count reached max_resumes_per_task;
    #                             by-design throttling, degrades to fresh dispatch.
    # (enabled=False degrades silently — no event.)
    #
    # Of the fallback reasons, `reseeded` is EXPECTED, not a failure (task
    # 3256): warm-lane acquire ALWAYS re-seeds a lane from base, wiping
    # <lane>/.task/ and the whole claude-config transcript store with it, so a
    # session adopted at boot routinely finds its store gone by re-dispatch. It
    # therefore does NOT feed the fallback-storm streak (like
    # session_resume_capped); only {stale, no_transcript} do. The event is still
    # emitted so the rate stays measurable (PRD open question 3 — lane-collision
    # rate is read off these reasons post-deploy).
    #
    # Ratio recipe: there is no separate "attempt" row — attempts are the SUM of
    # the three outcome events (session_resume + session_resume_fallback +
    # session_resume_capped) for a window, since the guard emits exactly one per
    # dispatch that carried a recovered session. Read the fallback RATE as a
    # ratio against that denominator rather than as an absolute count, and split
    # the numerator by json_extract(data, '$.reason') to separate expected
    # reseeds from genuine corroboration failures. (enabled=False emits nothing,
    # so a zero total means either no recovered sessions or the kill switch.)
    session_resume = 'session_resume'
    session_resume_fallback = 'session_resume_fallback'
    session_resume_capped = 'session_resume_capped'

    # Scheduler fairness
    task_skipped = 'task_skipped'
    reservation_installed = 'reservation_installed'
    reservation_expired = 'reservation_expired'
    reservation_evicted = 'reservation_evicted'
    reservation_shadowed = 'reservation_shadowed'
    reservation_restored = 'reservation_restored'
    reservation_used = 'reservation_used'
    reservation_force_evicted = 'reservation_force_evicted'
    reservation_force_evict_refused = 'reservation_force_evict_refused'
    # Emitted once per acquire_next tick when the fused-memory task read
    # FAILED (distinct from a genuinely empty project) and the park-eviction
    # drain was therefore SKIPPED (fail-safe, survey finding C3).
    # Scheduler-scoped (task_id=None). Payload: {consecutive_failures}.
    park_eviction_deferred_fm_unavailable = 'park_eviction_deferred_fm_unavailable'
    scheduler_tier_cap_idle = 'scheduler_tier_cap_idle'

    # Scheduler priority overrides
    #
    # Pin-order observation contract (decided in task 1290):
    #
    # These events are observability-only and may lag OverrideStore / MCP
    # writes by one scheduler tick.  ``pin_queue_reordered`` fires ONLY on a
    # pure reorder (tasks already pinned shifting position via
    # reorder_pin_queue) and carries the complete new ordering in
    # ``data['new_order']``.  It is intentionally NOT emitted on pin add
    # (task_pinned) or remove (task_unpinned), since those events already
    # fully describe the change.
    #
    # Consumer strategies for current pin order:
    #   (i)  Event recomposition — task_pinned → append task_id;
    #        task_unpinned → remove task_id;
    #        pin_queue_reordered → replace list with new_order.
    #   (ii) Authoritative snapshot (preferred) — call
    #        OverrideStore.get_pin_queue(project_root) or read
    #        snapshot.pin_queue from the scheduler's public API directly.
    #
    # See also: _emit_override_diff_events docstring in scheduler.py
    # (producer side) for full rationale and example consumer code.
    priority_override_set = 'priority_override_set'
    priority_override_cleared = 'priority_override_cleared'
    task_pinned = 'task_pinned'
    task_unpinned = 'task_unpinned'
    pin_queue_reordered = 'pin_queue_reordered'

    # Scheduler park-and-stop pause (AFK hardening, task 1322).
    # scheduler_paused: emitted when Harness.pause_scheduler() is called
    #   (whether by the park-stop trip, a sibling cost-ceiling watcher, or a
    #   human operator).  Data payload: {reason, threshold, window_hours}.
    # scheduler_resumed: emitted when Harness.resume_scheduler() clears the pause.
    # scheduler_pause_restored: emitted on orchestrator restart when a persisted
    #   pause is loaded from runs.db.  Distinct from scheduler_paused so the
    #   event timeline self-documents cross-run continuity.  Data payload:
    #   {reason, pause_at, restored_from_run_id}.
    scheduler_paused = 'scheduler_paused'
    scheduler_resumed = 'scheduler_resumed'
    scheduler_pause_restored = 'scheduler_pause_restored'

    # Plan revalidation
    plan_revalidated = 'plan_revalidated'

    # Config hot-reload (plans/config-hot-reload-prd.md, task beta) — emitted
    # by Harness.reload_config() on every call, success or failure. Data
    # payload = the full reload report: {reloaded, config_path, applied,
    # restart_required, unchanged, error}.
    config_reload = 'config_reload'

    # Plan salvaged — the architect CLI run reported failure (e.g. budget/turn
    # cap) but had already written a finalized, valid plan to disk; the
    # workflow uses that plan instead of discarding it and re-planning.
    plan_salvaged = 'plan_salvaged'

    # Architect merge-landing-desync exit (plans/architect-already-complete-
    # exits.md §β) — emitted once per `.task/ready_to_merge.json` report by
    # TaskWorkflow._handle_ready_to_merge_report, on BOTH dispositions, so the
    # decision is legible without scraping logs:
    #   accepted  → data: {decision: 'enqueued', tip, main_sha, request_id,
    #                      evidence}
    #   duplicate → data: {decision: 'duplicate', tip, main_sha, marker}
    #   rejected  → data: {decision: 'rejected', predicate: <name>,
    #                      measured: {...}}  — `predicate` names the FIRST
    #               desync predicate that failed and `measured` carries the
    #               first-hand values it was judged against.
    architect_desync_merge = 'architect_desync_merge'

    # Phase skipped — emitted when an optimistic-path optimisation
    # short-circuits a workflow phase (B: revalidation skipped on overlap=0;
    # C: full architect+implementer skipped via SIMPLE_TASK).
    phase_skipped = 'phase_skipped'

    # Auto-eval — original task's branch + worktree renamed and a sibling
    # redo task dispatched on the full-architect path so the optimistic
    # path's outcome can be ground-truthed against a baseline.
    auto_eval_dispatched = 'auto_eval_dispatched'

    # Retry cap — per-task REQUEUED counter exceeded; emitted once per
    # cap-exhaustion event by Scheduler.trigger_retry_cap_exhausted.
    retry_cap_exhausted = 'retry_cap_exhausted'

    # Reserve-now lifecycle — emitted when a reserve_now flag transitions
    # False→True (armed) or when parks are installed for a reserve_now task
    # and the flag is cleared (consumed).
    #
    # BREAKING RENAME (task 1230, commits 4d45eecd9b / deb8f426ab):
    # reserve_now_consumed replaced the former reserve-now short-circuit form
    # of reservation_installed (which used data.reason == 'reserve_now').
    # The two reservation events are discriminated by EVENT NAME, not a
    # 'reason' field:
    #   threshold-based install  → reservation_installed
    #                               data: {modules, skip_count, priority}
    #                               no 'reason' key — UNCHANGED by task 1230
    #   reserve-now short-circuit → reserve_now_consumed
    #                               data: {modules, priority}
    # Audit (task 1333): exhaustive repo search found ZERO in-repo consumers
    # that ever filtered reservation_installed by data.reason == 'reserve_now'.
    # Downstream consumers outside this repo must migrate; see CHANGELOG.md.
    reserve_now_armed = 'reserve_now_armed'
    reserve_now_consumed = 'reserve_now_consumed'

    # Worktree hygiene (Fix B/C) — emitted when an orphaned or identity-
    # mismatched worktree is relocated to the sibling ``-orphaned`` base
    # (quarantine, WIP preserved) or removed (reap, provably-empty/clean).
    worktree_quarantined = 'worktree_quarantined'
    worktree_reaped = 'worktree_reaped'

    # Train formation (β former — emitted when the former selects a train)
    train_formed = 'train_formed'

    # Atomic train lifecycle (PRD 13.7)
    train_started = 'train_started'
    train_member_deferred = 'train_member_deferred'
    train_merged = 'train_merged'
    train_derailed = 'train_derailed'
    # Retroactive coalescing pass (γ/1719): waiting singles merged into one train.
    # data keys: train_id, absorbed_request_ids, member_task_ids, tip_task_id,
    # ejected (stacking-conflict ejects), size.
    train_coalesced = 'train_coalesced'

    # Service lifecycle — emitted when the post-merge staleness hook triggers
    # a restart of a backing service (e.g. fused-memory.service after a merge
    # whose landed diff touched fused-memory/src/).
    service_restart = 'service_restart'

    # Cross-project external-dep gate held — emitted when a pending task's
    # external deps have been holding dispatch for ``threshold`` consecutive
    # ticks.  Makes the hold DASHBOARD-VISIBLE (telemetry) without escalating
    # (which would be premature for a legitimately-slow upstream).
    # cause values: 'resolver_degraded' (MCP resolver returned an unusable
    #   result) or 'deps_live' (deps returned a live non-done status).
    # data keys: {cause, ticks, detail}.
    external_dep_gate_held = 'external_dep_gate_held'

    # Delivered-check dep-gate held — emitted when a pending task's done/
    # cancelled local dep carries metadata.delivered_checks and its
    # per-(dep_task_id, main_sha) cache entry is NOT True (either the check
    # ran-and-FAILED, or it is absent — errored/over-budget/not-yet-run).
    # Unlike external_dep_gate_held, this fires on EVERY held tick (no
    # threshold gate) — task 2580 (delta) has no grace_cycles config yet;
    # task 2583 (epsilon) layers threshold-gated escalation on top with its
    # own counter. Pure dashboard visibility (capability-delivered-checks
    # PRD, plans/capability-delivered-checks-prd.md, task delta).
    # data keys: {ticks, detail}.
    #   detail: dict naming the failed check (name/dep_id/main_sha/kind) or
    #           None when _note_delivered_hold was called without one.
    delivered_check_gate_held = 'delivered_check_gate_held'

    # Dispatch-admission gate deferred a HEAVY (normal-kind) candidate this
    # tick because host PSI pressure is saturated (config.psi_admission) and
    # this orchestrator's own in-flight count (len(Scheduler._dispatched)) is
    # at/above the configured floor (task 2328, DA3 of
    # docs/prds/dispatch-admission-load-cap.md).  Deterministic-kind
    # candidates are exempt (DA-D4) and never produce this event.  Emitted at
    # most once per ``acquire_next`` tick — on the FIRST heavy candidate
    # skipped, across both the pinned and scored dispatch loops.
    # data keys: {metric, value, in_flight, floor}
    #   metric:    the PSI field name that tripped saturation, chosen in
    #              ranked order cpu_some_avg10 > mem_some_avg10 >
    #              mem_full_avg10 > io_some_avg10 (DA-D1)
    #   value:     that metric's sampled avg10 value (float)
    #   in_flight: len(Scheduler._dispatched) at decision time (int)
    #   floor:     config.psi_admission.min_inflight_floor (int)
    dispatch_deferred = 'dispatch_deferred'

    # Paired (rebase → immediately-following verify) cost record.
    # Emitted once per real rebase (not short-circuit) in _verify_debugfix_loop.
    # Data payload: {
    #   old_base: str,           # SHA before rebase
    #   new_base: str,           # SHA after rebase (current main)
    #   distance_commits: int,   # git rev-list --count old_base..new_base (-1 = unknown)
    #   files_changed_on_main: int,  # len(changed_files) from the rebase return
    #   next_verify_wall_secs: float,  # VerifyResult.duration_secs of the following verify
    #   verify_scope: {          # context for cost normalisation
    #     n_task_files: int,
    #     n_modules: int,
    #     workspace: bool,       # True when self._train is not None
    #   },
    #   cohort: str,             # 'continuous' | 'post-unblock' | 'big-jump' | 'unknown'
    # }
    rebase_verify_cost = 'rebase_verify_cost'

    # Emitted by the post-amendment WIP guard (TaskWorkflow._commit_amendment_wip,
    # task 2760) when it rescues UNCOMMITTED amendment work by auto-committing it
    # as `wip: amendment round N` before the loop re-enters VERIFY/REVIEW — those
    # phases (and the merge queue) see only committed branch state, so a reviewed,
    # verified amendment can never silently fail to land.  Phase 'review'.
    # data keys: {amendment_round:int, recovery:'auto_commit', wip_sha:str|None}
    amendment_uncommitted_recovered = 'amendment_uncommitted_recovered'

    # OS-sandbox worktree containment (plans/os-sandbox-worktree-containment-prd.md
    # task β2, §Goal 3 / INV-2). Gives operators (and γ1's soak predicate) a
    # mechanical, per-invocation record of whether OS containment was active for
    # each sandboxed dispatch: every sandboxed invocation emits exactly one of
    # these two — a real/none-escape wrap emits sandbox_applied, a fail-closed
    # refusal emits sandbox_unavailable.
    #
    # sandbox_unavailable — emitted in TaskWorkflow._guard_sandbox's
    #   except-SandboxUnavailable branch on the β1 fail-closed refusal, ONCE PER
    #   REFUSAL (deliberately NOT deduped — unlike the accompanying blocking
    #   escalation, which stays deduped per backend-state change, INV-4). The
    #   event store is the per-invocation structured record γ1 queries ("0
    #   sandbox-attributed blocks" must see EACH refusal), so suppressing repeat
    #   events would blind the soak predicate. role/task_id are first-class
    #   columns. Payload: {backend} — the configured backend that failed to
    #   resolve (SandboxUnavailable.backend_state).
    sandbox_unavailable = 'sandbox_unavailable'
    # sandbox_applied — emitted once per sandboxed invocation by
    #   TaskWorkflow._invoke on the NON-refusing guard path: a real backend
    #   (landlock/bwrap) OR the explicit ``backend=none`` operator escape hatch
    #   (still emitted — data.backend lets γ1 distinguish deliberately-
    #   unsandboxed from real containment without a third event type).
    #   role/task_id are first-class columns. Payload: {backend, digest} —
    #   ``digest`` is WriteSet.digest(), the STABLE sha256 of the resolved
    #   writable set so an operator can diff exactly what a given invocation
    #   could touch (INV-2 structured facts). Consumed by γ1's soak predicate.
    sandbox_applied = 'sandbox_applied'


class EventStore:
    """Append-only SQLite event store.

    One instance per orchestrator run, created in the harness and propagated
    to workflows, merge worker, steward, and scheduler.  Every ``emit()``
    call is fire-and-forget — failures log a warning but never raise.
    """

    def __init__(self, db_path: Path, run_id: str):
        self.db_path = db_path
        self.run_id = run_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the store's DB with the full durability pragma triad applied."""
        conn = sqlite3.connect(str(self.db_path))
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
        return conn

    def checkpoint(self) -> CheckpointResult:
        """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` and return the result.

        Returns:
            A :class:`CheckpointResult` named-tuple ``(busy, log, checkpointed)``.
        """
        conn = self._connect()
        try:
            row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
        finally:
            conn.close()
        if row is None:
            raise RuntimeError('PRAGMA wal_checkpoint returned no rows')
        return CheckpointResult(int(row[0]), int(row[1]), int(row[2]))

    def _ensure_schema(self) -> None:
        try:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
            finally:
                conn.close()
        except Exception:
            logger.warning('event_store: schema init failed', exc_info=True)

    def emit(
        self,
        event_type: EventType,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict | None = None,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Write a single event row.  Never raises."""
        try:
            conn = self._connect()
            try:
                conn.execute(
                    'INSERT INTO events '
                    '(timestamp, run_id, task_id, event_type, phase, role, '
                    ' data, cost_usd, duration_ms) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        datetime.now(UTC).isoformat(),
                        self.run_id,
                        task_id,
                        event_type.value,
                        phase,
                        role,
                        json.dumps(data) if data else '{}',
                        cost_usd,
                        duration_ms,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.warning('event_store.emit failed', exc_info=True)

    def latest_merge_finalized(
        self,
        request_id: str | None = None,
        branch: str | None = None,
        task_id: str | None = None,
    ) -> dict | None:
        """Return the most-recent merge_finalized event row matching the given key.

        Lookup precedence: request_id > branch > task_id.  When no key is
        provided, returns None immediately.

        Queries are scoped to the current run (``self.run_id``).  branch= and
        task_id= lookups therefore return the most-recent outcome for the
        *current* orchestrator run only; results from previous runs that
        reused the same branch or task_id are not returned.  This prevents
        merge_status from silently surfacing stale prior-run terminal outcomes.

        The returned dict has keys:
            request_id, task_id, branch, state, snapshot_tip, merge_sha,
            reason, finished_at (ISO-8601 timestamp string from the events table).

        Returns None on miss, on unknown key, or if all keys are None.
        Errors are logged and return None (read path is fire-safe).
        """
        if request_id is None and branch is None and task_id is None:
            return None
        try:
            conn = self._connect()
            try:
                if request_id is not None:
                    row = conn.execute(
                        "SELECT task_id, data, timestamp FROM events "
                        "WHERE event_type='merge_finalized' "
                        "  AND run_id = ? "
                        "  AND json_extract(data,'$.request_id')=? "
                        "ORDER BY id DESC LIMIT 1",
                        (self.run_id, request_id),
                    ).fetchone()
                elif branch is not None:
                    row = conn.execute(
                        "SELECT task_id, data, timestamp FROM events "
                        "WHERE event_type='merge_finalized' "
                        "  AND run_id = ? "
                        "  AND json_extract(data,'$.branch')=? "
                        "ORDER BY id DESC LIMIT 1",
                        (self.run_id, branch),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT task_id, data, timestamp FROM events "
                        "WHERE event_type='merge_finalized' "
                        "  AND run_id = ? "
                        "  AND task_id=? "
                        "ORDER BY id DESC LIMIT 1",
                        (self.run_id, task_id),
                    ).fetchone()
            finally:
                conn.close()
            if row is None:
                return None
            db_task_id, raw_data, timestamp = row
            data = json.loads(raw_data) if raw_data else {}
            return {
                'request_id': data.get('request_id'),
                'task_id': db_task_id,
                'branch': data.get('branch'),
                'state': data.get('state'),
                'snapshot_tip': data.get('snapshot_tip'),
                'merge_sha': data.get('merge_sha'),
                'superseded_by': data.get('superseded_by'),
                'reason': data.get('reason'),
                'finished_at': timestamp,
            }
        except Exception:
            logger.warning('event_store.latest_merge_finalized failed', exc_info=True)
            return None

    def fetch_events_by_type(self, event_type: str | EventType) -> list[dict]:
        """Return all events of *event_type* emitted in the current run, ordered by id.

        The ``data`` column of every returned row is parsed from JSON back to a
        dict.  Returns an empty list when no rows match.  Errors are logged and
        return [] (read path is fire-safe).
        """
        try:
            type_str = event_type.value if isinstance(event_type, EventType) else str(event_type)
            conn = self._connect()
            try:
                rows = conn.execute(
                    'SELECT id, timestamp, run_id, task_id, event_type, phase, role, '
                    '       data, cost_usd, duration_ms '
                    'FROM events '
                    'WHERE event_type = ? AND run_id = ? '
                    'ORDER BY id',
                    (type_str, self.run_id),
                ).fetchall()
            finally:
                conn.close()
            result = []
            for row in rows:
                (row_id, timestamp, run_id, task_id, evt_type, phase,
                 role, raw_data, cost_usd, duration_ms) = row
                result.append({
                    'id': row_id,
                    'timestamp': timestamp,
                    'run_id': run_id,
                    'task_id': task_id,
                    'event_type': evt_type,
                    'phase': phase,
                    'role': role,
                    'data': json.loads(raw_data) if raw_data else {},
                    'cost_usd': cost_usd,
                    'duration_ms': duration_ms,
                })
            return result
        except Exception:
            logger.warning('event_store.fetch_events_by_type failed', exc_info=True)
            return []

    def fetch_events_by_type_all_runs(
        self, event_type: str | EventType, *, task_id: str | None = None,
    ) -> list[dict]:
        """Return all events of *event_type* across ALL runs, ordered by id.

        The restart-durable (run-agnostic) counterpart to
        ``fetch_events_by_type``: it drops the ``run_id`` scoping so a row
        emitted in a PRIOR orchestrator run (e.g. before a fleet redeploy) is
        still visible.  This is what the durable verified-green checkpoint
        (task 2752) relies on — the run-scoped ``fetch_events_by_type`` cannot
        see a prior run's ``workflow_verify`` green.

        When *task_id* is given, an additional ``task_id = ?`` predicate bounds
        the result set to that one task's rows (a handful, not the whole
        cross-run history).  The ``data`` column of every returned row is
        parsed from JSON back to a dict.  Returns an empty list when no rows
        match.  Errors are logged and return [] (read path is fire-safe).
        """
        try:
            type_str = event_type.value if isinstance(event_type, EventType) else str(event_type)
            sql = (
                'SELECT id, timestamp, run_id, task_id, event_type, phase, role, '
                '       data, cost_usd, duration_ms '
                'FROM events '
                'WHERE event_type = ?'
            )
            params: list = [type_str]
            if task_id is not None:
                sql += ' AND task_id = ?'
                params.append(task_id)
            sql += ' ORDER BY id'
            conn = self._connect()
            try:
                rows = conn.execute(sql, tuple(params)).fetchall()
            finally:
                conn.close()
            result = []
            for row in rows:
                (row_id, timestamp, run_id, row_task_id, evt_type, phase,
                 role, raw_data, cost_usd, duration_ms) = row
                result.append({
                    'id': row_id,
                    'timestamp': timestamp,
                    'run_id': run_id,
                    'task_id': row_task_id,
                    'event_type': evt_type,
                    'phase': phase,
                    'role': role,
                    'data': json.loads(raw_data) if raw_data else {},
                    'cost_usd': cost_usd,
                    'duration_ms': duration_ms,
                })
            return result
        except Exception:
            logger.warning(
                'event_store.fetch_events_by_type_all_runs failed', exc_info=True
            )
            return []
