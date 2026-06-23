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
    merge_verify = 'merge_verify'
    verdict_parity_ok = 'verdict_parity_ok'
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
    speculative_merge = 'speculative_merge'
    speculative_discard = 'speculative_discard'

    # Task lifecycle
    task_started = 'task_started'
    task_completed = 'task_completed'

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

    # Plan salvaged — the architect CLI run reported failure (e.g. budget/turn
    # cap) but had already written a finalized, valid plan to disk; the
    # workflow uses that plan instead of discarding it and re-planning.
    plan_salvaged = 'plan_salvaged'

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
            finished_at (ISO-8601 timestamp string from the events table).

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
