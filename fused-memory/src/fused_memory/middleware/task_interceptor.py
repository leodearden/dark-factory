"""Intercepts task state transitions for targeted reconciliation."""

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import uuid as uuid_mod
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from shared.cli_invoke import AllAccountsCappedException  # type: ignore[import]
except ImportError:
    # Sentinel that can never be raised by real code, so the
    # `except (CuratorFailureError, AllAccountsCappedException)` clause
    # below still has tuple-compatible semantics without silently catching
    # every Exception (which would collapse the cap-retry fallback path
    # into the generic decisions=[None]*N degrade path and make the
    # subsequent `except Exception` branch dead code).
    class _UnavailableAllAccountsCapped(Exception):
        """Placeholder used only when shared.cli_invoke is not importable."""

    AllAccountsCappedException = _UnavailableAllAccountsCapped  # type: ignore[assignment,misc]

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.middleware.task_curator import (
    CandidateTask,
    CuratorDecision,
    CuratorFailureError,
    TaskCurator,
    _to_pool_entry,
    flatten_task_tree,
    normalize_title,
)
from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer

if TYPE_CHECKING:
    from shared.usage_gate import UsageGate

    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.middleware.curator_escalator import CuratorEscalator
    from fused_memory.middleware.task_file_committer import TaskFileCommitter
    from fused_memory.middleware.ticket_store import TicketStore
    from fused_memory.reconciliation.backlog_policy import BacklogPolicy
    from fused_memory.reconciliation.bulk_reset_guard import BulkResetGuard
    from fused_memory.reconciliation.event_queue import EventQueue
    from fused_memory.reconciliation.targeted import TargetedReconciler

logger = logging.getLogger(__name__)

# Terminal statuses the server refuses to exit without a reopen_reason.
# Duplicated from orchestrator.task_status.TERMINAL_STATUSES — the server
# and orchestrator are independent modules, and the set is effectively
# ossified; duplication is cheaper than cross-package coupling.
TERMINAL_STATUSES: frozenset[str] = frozenset({'done', 'cancelled'})


def _is_ticket_id(value: object) -> bool:
    """Return True when *value* looks like a two-phase ticket id (``tkt_…``)."""
    return isinstance(value, str) and value.startswith('tkt_')


def _format_ticket_result(row: dict) -> dict:
    """Format a terminal ticket row as the public resolve_ticket response dict.

    Exposes only ``{status, task_id?, reason?}`` — does NOT leak ``result_json``.
    """
    result: dict = {'status': row['status']}
    if row.get('task_id') is not None:
        result['task_id'] = row['task_id']
    if row.get('reason') is not None:
        result['reason'] = row['reason']
    return result


class TaskInterceptor:
    """Wraps Taskmaster operations, intercepts state transitions for targeted reconciliation."""

    STATUS_TRIGGERS = {'done', 'blocked', 'cancelled', 'deferred'}
    BULK_TRIGGERS = {'parse_prd', 'expand_task'}

    def __init__(
        self,
        taskmaster: TaskmasterBackend | None,
        targeted_reconciler: 'TargetedReconciler | None',
        event_buffer: EventBuffer,
        task_committer: 'TaskFileCommitter | None' = None,
        config: 'FusedMemoryConfig | None' = None,
        escalator: 'CuratorEscalator | None' = None,
        event_queue: 'EventQueue | None' = None,
        backlog_policy: 'BacklogPolicy | None' = None,
        usage_gate: 'UsageGate | None' = None,
        ticket_store: 'TicketStore | None' = None,
        bulk_reset_guard: 'BulkResetGuard | None' = None,
    ):
        self.taskmaster = taskmaster
        self.reconciler = targeted_reconciler
        self.buffer = event_buffer
        # WP-B: when an event queue is wired in, journalling is fire-and-forget
        # via the background drainer. If None, fall back to inline buffer.push
        # — preserves the legacy call pattern for tests that haven't yet been
        # updated to construct a queue.
        self.event_queue = event_queue
        self.task_committer = task_committer
        # WP-D: bounded-backlog enforcement. Each mutating public method calls
        # ``_backlog_policy.check(project_id, project_root)`` before acquiring
        # the project lock; a rejection verdict short-circuits to a structured
        # error dict with no taskmaster mutation.
        self._backlog_policy = backlog_policy
        # Task 918: defence-in-depth bulk-reset circuit-breaker.  When set,
        # _apply_status_transition calls observe_attempt after the same-status
        # no-op check and before the terminal-exit gate.  Rejections
        # short-circuit without any taskmaster mutation, event emission, or
        # reconciliation scheduling.
        self._bulk_reset_guard = bulk_reset_guard
        self._background_tasks: set[asyncio.Task] = set()

        # Task curator: LLM-judged drop/combine/create gate. Lazy-initialized in
        # _get_curator() because it pulls in a Qdrant client + embedder.
        self._config = config
        self._curator: TaskCurator | None = None
        self._escalator = escalator
        # Forwarded to ``TaskCurator`` for cap-aware LLM invocation across the
        # shared account pool. ``None`` falls back to the legacy single-shot
        # path with no cap retry — preserved for tests.
        self._usage_gate = usage_gate
        # Split per-project locks (2026-04-20; updated 2026-04-22 for ticket queue):
        #
        # ``_write_locks`` (short, high-frequency) serialises tasks.json
        # mutations for every write op (set_task_status, update_task,
        # add_dependency, remove_dependency, and the actual tm.add_task
        # write inside the worker). Held only across the fast Taskmaster
        # stdio call. Every mutation takes this lock.
        #
        # ``_curator_locks`` (long, curator-family) serialises the
        # add_subtask / remove_task flow so concurrent candidates can't both
        # decide "create" for a duplicate. Held across ``curator.curate()``
        # (an LLM round-trip, 25-35s tail) plus the synchronous
        # ``note_created`` + awaited ``record_task`` so the NEXT waiter on
        # the curator lock sees the new entry on its pre-LLM check (R3).
        #
        # NOTE: add_task's *queueing* is now provided by per-project ticket
        # queues and workers (_curator_worker) — one asyncio.Queue and one
        # asyncio.Task per project_id — so a slow curator round-trip on
        # project A does not delay submissions on project B.  The worker
        # STILL acquires ``_curator_lock(project_id)`` around curator.curate()
        # through note_created + record_task to preserve R3: add_subtask /
        # remove_task still enter _curator_lock directly, and without this
        # acquisition their curate() calls could race against the worker's
        # on a stale pre-note_created snapshot and both decide "create" for
        # the same candidate.  Within a project, the lock provides
        # single-threaded curator execution; across projects, the per-project
        # queue+worker+lock architecture preserves fairness.
        #
        # Lock ordering is always curator_lock BEFORE write_lock to avoid
        # deadlock. Write-only ops take just write_lock, so a long curator
        # call on a project no longer blocks status updates on the same
        # project — fixing the symptom that caused 50+ spurious "Failed to
        # set task X status to Y: " errors in the reify orchestrator run
        # on 2026-04-20.
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._curator_locks: dict[str, asyncio.Lock] = {}
        # One-shot flag: prevents redundant auto-backfill checks on subsequent calls.
        self._backfill_triggered: bool = False
        # Set by close(); prevents _get_curator() from re-creating a curator.
        self._closed: bool = False
        # Set on first start() call; makes a second start() a true no-op so
        # flush_pending_on_startup cannot flip in-flight tickets to failed.
        self._started: bool = False
        # Two-phase ticket store: persists submitted tickets across restarts.
        self._ticket_store = ticket_store
        # Per-project in-memory queues of ticket_ids pending worker processing.
        # Each project_id gets its own Queue so a slow curator on project A
        # does not delay submissions on project B.  Queues are created lazily
        # on first submit_task call (via setdefault) — no empty-queue overhead
        # for projects that never use submit_task.
        self._ticket_queues: dict[str, asyncio.Queue[str]] = {}
        # Per-project asyncio.Task handles for the curator workers.  Keyed by
        # project_id, lazily started on first submit_task for that project.
        self._worker_tasks: dict[str, asyncio.Task] = {}
        # Per-ticket asyncio.Event lists: resolve_ticket appends one per caller
        # so multiple concurrent waiters (e.g. reconnect/retry patterns) each
        # get their own event.  _signal_ticket_event sets and removes all of them.
        self._ticket_events: dict[str, list[asyncio.Event]] = {}

    async def _ensure_taskmaster(self) -> TaskmasterBackend:
        """Return a connected TaskmasterBackend, or raise with a structured error."""
        if self.taskmaster is None:
            raise RuntimeError('Taskmaster is not configured.')
        await self.taskmaster.ensure_connected()
        return self.taskmaster

    async def _backlog_gate(self, project_root: str) -> dict | None:
        """WP-D guard. Returns a structured error dict if the policy rejects;
        otherwise ``None`` and the caller proceeds.
        """
        if self._backlog_policy is None:
            return None
        project_id = resolve_project_id(project_root)
        verdict = await self._backlog_policy.check(project_id, project_root)
        if verdict.is_rejection:
            return verdict.to_error_dict()
        return None

    def _make_event(
        self, event_type: EventType, project_root: str, payload: dict
    ) -> ReconciliationEvent:
        payload = {**payload, '_project_root': project_root}
        return ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=event_type,
            source=EventSource.agent,
            project_id=resolve_project_id(project_root),
            timestamp=datetime.now(UTC),
            payload=payload,
        )

    async def _journal(self, event: ReconciliationEvent) -> None:
        """Hand off an event for persistence.

        WP-B: when a fire-and-forget ``EventQueue`` is configured, enqueue
        synchronously and return immediately — the MCP hot path never
        awaits SQLite. Without a queue (legacy / test setups), fall back
        to an inline ``buffer.push``.
        """
        if self.event_queue is not None:
            self.event_queue.enqueue(event)
            return
        # Legacy inline push (no queue wired) — tests and degraded setups.
        buffer = self.buffer
        await buffer.push(event)

    @staticmethod
    def _on_reconciliation_done(task: asyncio.Task) -> None:
        """Callback for fire-and-forget reconciliation tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f'Background reconciliation failed: {exc}')

    def _schedule_commit(self, project_root: str, operation: str) -> None:
        """Fire-and-forget auto-commit of tasks.json."""
        if self.task_committer is None:
            return
        task = asyncio.create_task(
            self.task_committer.commit(project_root, operation),
            name=f'auto-commit-{operation}',
        )
        self._background_tasks.add(task)
        task.add_done_callback(lambda t: self._background_tasks.discard(t))
        task.add_done_callback(self._on_commit_done)

    @staticmethod
    def _on_commit_done(task: asyncio.Task) -> None:
        """Callback for fire-and-forget commit tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f'Background auto-commit failed: {exc}')

    async def _await_commit(self, project_root: str, operation: str) -> None:
        """Await commit directly (used by bulk ops that must capture full batch)."""
        if self.task_committer is None:
            return
        await self.task_committer.commit(project_root, operation)

    async def drain(self) -> None:
        """Await all pending background tasks (commits + reconciliation).

        Call at shutdown or when you need to guarantee all fire-and-forget
        work has completed.
        """
        if not self._background_tasks:
            return
        tasks = list(self._background_tasks)
        logger.info('Draining %d background tasks', len(tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error('Background task failed during drain: %s', result)

    async def close(self) -> None:
        """Release resources held by the interceptor.

        1. Sets the ``_closed`` flag so ``submit_task`` rejects new work.
        2. Cancels and awaits the curator worker task (if running).  Note:
           queued tickets that have not yet been dequeued remain ``pending``
           in the store and will be marked ``failed/server_restart`` on the
           next ``start()`` call.  In-flight callers waiting in
           ``resolve_ticket`` are woken immediately (step 3) so they do not
           hang indefinitely.
        3. Signals all registered ``_ticket_events`` so any ``resolve_ticket``
           callers that are blocked on ``event.wait()`` unblock and re-read
           the (still-pending) ticket row, returning the timeout sentinel.
        4. Drains any remaining fire-and-forget background tasks.
        5. Closes the :class:`TaskCurator`'s Qdrant connection.
        6. Closes the :class:`TicketStore`'s SQLite connection.

        Idempotent: safe to call multiple times.
        """
        # Set closed first so submit_task rejects new tickets before we cancel
        # the worker — prevents a TOCTOU race between close() and submit_task().
        self._closed = True

        # Cancel all per-project curator workers so they stop blocking on
        # curator.curate().  Gather all tasks regardless of done-state so we
        # can detect any task that raised unexpectedly.
        running_workers = [
            t for t in self._worker_tasks.values() if not t.done()
        ]
        for t in running_workers:
            t.cancel()
        if running_workers:
            await asyncio.gather(*running_workers, return_exceptions=True)

        # Wake all resolve_ticket callers that are waiting on an event so they
        # don't block forever after the worker has been cancelled.  The callers
        # will re-read the ticket row (still 'pending') and return the
        # {status: failed, reason: server_closed} sentinel.
        for events in list(self._ticket_events.values()):
            for event in events:
                event.set()

        await self.drain()
        if self._curator is not None:
            await self._curator.close()
            self._curator = None

        if self._ticket_store is not None:
            await self._ticket_store.close()

    async def start(self) -> None:
        """Initialise runtime state after construction.

        Call exactly once after constructing the interceptor and before
        accepting submit_task traffic.  Guarded by an ``_started`` flag so a
        second invocation is a true no-op — without the guard,
        ``flush_pending_on_startup`` would flip any in-flight tickets (rows
        submitted by the current run but not yet resolved by the worker) to
        ``failed/server_restart`` and corrupt their waiters.

        Performs:
        - ``flush_pending_on_startup``: marks any pending tickets left from a
          previous server run as ``failed/server_restart``.
        - ``sweep_expired``: marks any expired pending tickets as
          ``failed/expired``.

        If no ``ticket_store`` is wired in, this is a no-op.
        """
        if self._started:
            return
        self._started = True
        if self._ticket_store is None:
            return
        flushed = await self._ticket_store.flush_pending_on_startup()
        if flushed:
            logger.warning('start(): flushed %d orphaned pending ticket(s) from prior run', flushed)
        swept = await self._ticket_store.sweep_expired()
        if swept:
            logger.info('start(): swept %d expired ticket(s)', swept)

    # ── Status transitions (with targeted reconciliation) ──────────────

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
        done_provenance: dict | None = None,
        reopen_reason: str | None = None,
    ) -> dict:
        """Proxy to Taskmaster, then fire-and-forget targeted reconciliation if triggered.

        ``task_id`` may be a single id (``"12"``) or a comma-separated list
        (``"12,13,14"``). CSV input runs each id through the same gates
        independently and returns ``{'success': bool, 'results': [...]}``;
        single-id input returns the raw per-id result dict for backwards
        compatibility.
        """
        if err := await self._backlog_gate(project_root):
            return err
        await self._ensure_taskmaster()

        if ',' in task_id:
            ids = [t.strip() for t in task_id.split(',') if t.strip()]
            results: list[dict] = []
            for tid in ids:
                per_result = await self._apply_status_transition(
                    task_id=tid,
                    status=status,
                    project_root=project_root,
                    tag=tag,
                    done_provenance=done_provenance,
                    reopen_reason=reopen_reason,
                )
                results.append({'task_id': tid, 'result': per_result})
            all_ok = all(
                isinstance(r.get('result'), dict) and r['result'].get('error') is None
                for r in results
            )
            return {'success': all_ok, 'results': results}

        return await self._apply_status_transition(
            task_id=task_id,
            status=status,
            project_root=project_root,
            tag=tag,
            done_provenance=done_provenance,
            reopen_reason=reopen_reason,
        )

    async def _apply_status_transition(
        self,
        *,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None,
        done_provenance: dict | None,
        reopen_reason: str | None,
    ) -> dict:
        """Single-id status transition with all gates + event emission.

        Extracted so the public ``set_task_status`` can loop over CSV ids
        and apply the gates per-id. Holds the write lock across
        read→check→write, emits the event outside the lock.
        """
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        resolved_provenance: dict | None = None
        resolved_reopen_reason: str | None = None
        async with self._write_lock(project_id):
            # 1. Get before-state
            before = await tm.get_task(task_id, project_root, tag)

            # 2. Same-status guard: no-op if nothing changed
            old_status = _extract_status(before)
            if status == old_status:
                return {'success': True, 'no_op': True, 'task_id': task_id}

            # 2a-pre. Bulk-reset circuit-breaker (task 918, refined task 1016):
            # observe the attempt before the terminal-exit gate so the guard
            # catches reversal patterns regardless of whether individual attempts
            # carry a reopen_reason.  The 2026-04 bulk resets DID carry
            # reopen_reason — the guard fires on the *pattern*, not the per-id
            # legitimacy check.
            #
            # Split-counter design (task 1016): the two reversal kinds
            # (done→pending and in-progress→pending) are tracked against
            # independent thresholds, so a legitimate startup stranded-task
            # reconcile (e.g. the 2026-04-24 reify incident,
            # esc-bulk-reset-reify-2026-04-24T070944_6456580000: 27
            # in-progress→pending reversals in ~2 s) does not trip the
            # done→pending counter, which is tuned to catch data-loss patterns
            # (task 918, default threshold 10/60 s).  The in-progress→pending
            # counter has a higher default (100/60 s) to allow large startup
            # reconciles while still catching pathological runaways.
            #
            # Trade-off: placing the guard BEFORE the terminal-exit gate means
            # that done→pending attempts *without* a reopen_reason also consume
            # window slots, even though the terminal-exit gate below would
            # reject them anyway.  In a pathological burst of unauthorised
            # (no-reopen_reason) done→pending attempts, the guard will trip and
            # emit an escalation even though no actual reversals were allowed.
            # This is intentional: the *attempt pattern* is the signal.
            # False-positive trips from such bursts are acceptable because
            # (a) the escalation text names the affected task IDs so a steward
            # can distinguish legitimate from illegitimate reversals, and
            # (b) the guard's primary goal is limiting blast radius, not
            # distinguishing authorised from unauthorised reversals.
            #
            # IMPORTANT: This block MUST remain outside (before) the
            # ``if old_status in TERMINAL_STATUSES:`` check below.  Moving it
            # inside that branch would leave in-progress→pending reversals
            # (which are non-terminal and bypass that gate entirely) completely
            # unguarded.  The guard's _reversal_kind classifier handles its own
            # filtering; the interceptor must not pre-filter by old_status.
            if self._bulk_reset_guard is not None:
                _brg_verdict = await self._bulk_reset_guard.observe_attempt(
                    project_id=project_id,
                    task_id=task_id,
                    old_status=old_status,
                    new_status=status,
                    project_root=project_root,
                )
                if _brg_verdict.is_rejection:
                    return _brg_verdict.to_error_dict()

            # 2a. Terminal-exit gate: reject done→non-done and cancelled→non-done
            # unless a non-empty reopen_reason is provided. Moves the terminal
            # FSM server-side so two independent writers (scheduler + steward)
            # can no longer race past a stale client cache.
            if old_status in TERMINAL_STATUSES:
                reason = (reopen_reason or '').strip()
                if not reason:
                    return _terminal_exit_error(task_id, old_status, status)
                resolved_reopen_reason = reason
                try:
                    await tm.update_task(
                        task_id=task_id,
                        metadata=json.dumps({
                            'reopen_reason': reason,
                            'reopen_from': old_status,
                            'reopen_at': datetime.now(UTC).isoformat(),
                        }),
                        project_root=project_root,
                        tag=tag,
                    )
                except Exception as e:
                    logger.warning(
                        'Failed to persist reopen_reason for task %s: %s', task_id, e,
                    )

            # 2b. Phantom-done gate: if transitioning to done and the task
            # advertises concrete files in metadata.files, refuse when any of
            # those files is absent at project_root. Catches set_task_status
            # calls that bypass the orchestrator's merge gate (reify tasks
            # 1746/1747/1749, 2026-04-19).
            if status == 'done':
                declared = _extract_metadata_files(before)
                if declared:
                    missing = _missing_files(project_root, declared)
                    if missing:
                        return _done_gate_error(task_id, declared, missing)

            # 2c. Done-provenance gate: require done_provenance={commit?, note?}
            # so Stage-2 reconciliation has verified evidence to reference
            # instead of fabricating 'shipped via X' edges from metadata.modules.
            if status == 'done':
                validation_err, resolved_provenance = await _validate_done_provenance(
                    task_id, done_provenance, project_root,
                    require=self._require_done_provenance(),
                )
                if validation_err is not None:
                    return validation_err
                if resolved_provenance is not None:
                    try:
                        await tm.update_task(
                            task_id=task_id,
                            metadata=json.dumps({'done_provenance': resolved_provenance}),
                            project_root=project_root,
                            tag=tag,
                        )
                    except Exception as e:
                        logger.warning(
                            'Failed to persist done_provenance for task %s: %s', task_id, e,
                        )

            # 3. Execute status change. Convert the typed DTO to a plain
            # dict so callers can tack on the reconciliation key below.
            result: dict[str, Any] = dict(
                await tm.set_task_status(task_id, status, project_root, tag)
            )

        # 5. Emit event
        payload: dict[str, Any] = {
            'task_id': task_id, 'old_status': old_status, 'new_status': status,
        }
        if resolved_provenance is not None:
            payload['done_provenance'] = resolved_provenance
        if resolved_reopen_reason is not None:
            payload['reopen_reason'] = resolved_reopen_reason
            payload['reopen_from'] = old_status
        event = self._make_event(
            EventType.task_status_changed,
            project_root,
            payload,
        )
        await self._journal(event)
        self._schedule_commit(project_root, f'set_task_status({task_id}={status})')

        # 6. Targeted reconciliation for trigger statuses (fire-and-forget)
        if status in self.STATUS_TRIGGERS and self.reconciler:
            task = asyncio.create_task(
                self.reconciler.reconcile_task(
                    task_id=task_id,
                    transition=status,
                    project_id=resolve_project_id(project_root),
                    project_root=project_root,
                    task_before=before,
                ),
                name=f'targeted-recon-{task_id}-{status}',
            )
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))
            task.add_done_callback(self._on_reconciliation_done)
            result['reconciliation'] = {'status': 'async', 'task_id': task_id}

        return result

    # ── Bulk operations (with targeted reconciliation) ─────────────────

    async def expand_task(
        self,
        task_id: str,
        project_root: str,
        num: str | None = None,
        prompt: str | None = None,
        force: bool = False,
        tag: str | None = None,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        # WP-E: serialise the pre-snapshot + bulk mutation + commit so a
        # concurrent add_task can't slip a task into the snapshot gap and
        # get misclassified as newly-bulk-created.
        async with self._write_lock(project_id):
            # Pre-snapshot: capture the task tree before Taskmaster generates subtasks
            try:
                pre_snapshot = await tm.get_tasks(project_root)
            except Exception as pre_exc:
                logger.warning(
                    'bulk_dedup: pre-snapshot failed for expand_task(%s): %s', task_id, pre_exc,
                )
                pre_snapshot = None

            result: dict[str, Any] = dict(await tm.expand_task(
                task_id, project_root, num=num, prompt=prompt, force=force, tag=tag,
            ))
            await self._await_commit(project_root, f'expand_task({task_id})')
        event = self._make_event(
            EventType.tasks_bulk_created,
            project_root,
            {'parent_task_id': task_id, 'operation': 'expand_task'},
        )
        await self._journal(event)

        # Post-hoc dedup: remove newly-created tasks that duplicate pre-existing ones
        try:
            if pre_snapshot is not None:
                dedup = await self._dedupe_bulk_created(
                    project_root, pre_snapshot, parent_task_id=task_id,
                )
            else:
                dedup = {
                    'removed': [], 'kept': [], 'errors': [],
                    'skipped_reason': 'pre_snapshot_failed',
                }
            result['dedup'] = dedup
        except Exception as dedup_exc:
            logger.warning(
                'bulk_dedup: dedup block failed for expand_task(%s): %s', task_id, dedup_exc,
            )
            result['dedup'] = {'skipped_reason': f'exception: {dedup_exc}'}

        if self.reconciler:
            task = asyncio.create_task(
                self.reconciler.reconcile_bulk_tasks(
                    parent_task_id=task_id,
                    project_id=resolve_project_id(project_root),
                    project_root=project_root,
                ),
                name=f'bulk-recon-expand-{task_id}',
            )
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))
            task.add_done_callback(self._on_reconciliation_done)
            result['reconciliation'] = {'status': 'async', 'task_id': task_id}

        return result

    async def parse_prd(
        self,
        input_path: str,
        project_root: str,
        num_tasks: str | None = None,
        tag: str | None = None,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        # WP-E: serialise pre-snapshot + bulk mutation + commit; see
        # expand_task for the rationale.
        async with self._write_lock(project_id):
            # Pre-snapshot: capture the task tree before Taskmaster generates tasks
            try:
                pre_snapshot = await tm.get_tasks(project_root)
            except Exception as pre_exc:
                logger.warning('bulk_dedup: pre-snapshot failed for parse_prd: %s', pre_exc)
                pre_snapshot = None

            result: dict[str, Any] = dict(await tm.parse_prd(
                input_path, project_root, num_tasks=num_tasks, tag=tag,
            ))
            await self._await_commit(project_root, 'parse_prd')
        event = self._make_event(
            EventType.tasks_bulk_created,
            project_root,
            {'input_path': input_path, 'operation': 'parse_prd'},
        )
        await self._journal(event)

        # Post-hoc dedup: remove newly-created tasks that duplicate pre-existing ones
        try:
            if pre_snapshot is not None:
                dedup = await self._dedupe_bulk_created(project_root, pre_snapshot)
            else:
                dedup = {
                    'removed': [], 'kept': [], 'errors': [],
                    'skipped_reason': 'pre_snapshot_failed',
                }
            result['dedup'] = dedup
        except Exception as dedup_exc:
            logger.warning('bulk_dedup: dedup block failed for parse_prd: %s', dedup_exc)
            result['dedup'] = {'skipped_reason': f'exception: {dedup_exc}'}

        if self.reconciler:
            task = asyncio.create_task(
                self.reconciler.reconcile_bulk_tasks(
                    parent_task_id=None,
                    project_id=resolve_project_id(project_root),
                    project_root=project_root,
                ),
                name='bulk-recon-parse-prd',
            )
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))
            task.add_done_callback(self._on_reconciliation_done)
            result['reconciliation'] = {'status': 'async', 'operation': 'parse_prd'}

        return result

    # ── Curator helpers ────────────────────────────────────────────────

    async def _get_curator(self) -> TaskCurator | None:
        """Lazily construct the task curator gate.

        On first construction, triggers a background backfill check via
        ``_maybe_backfill_corpus()`` so pre-existing tasks are indexed into the
        curator corpus without blocking the caller.

        Returns ``None`` after :meth:`close` to prevent re-creating resources.
        """
        if self._closed:
            return None
        if self._curator is not None:
            return self._curator
        if self._config is None or not self._config.curator.enabled:
            return None
        try:
            from pathlib import Path as _Path

            cwd = None
            project_root: str | None = None
            if self.taskmaster is not None:
                pr = getattr(self.taskmaster.config, 'project_root', None)
                if pr:
                    cwd = _Path(pr)
                    project_root = str(pr)
            self._curator = TaskCurator(
                config=self._config,
                taskmaster=self.taskmaster,
                cwd=cwd,
                escalator=self._escalator,
                usage_gate=self._usage_gate,
            )
            # Trigger the one-shot backfill check as a background task so the
            # caller is not delayed by the Qdrant count() round-trip.
            if project_root is not None:
                bg = asyncio.create_task(
                    self._maybe_backfill_corpus(self._curator, project_root),
                    name='curator-backfill-check',
                )
                self._background_tasks.add(bg)
                bg.add_done_callback(lambda t: self._background_tasks.discard(t))
            return self._curator
        except Exception:
            logger.warning('Failed to create TaskCurator', exc_info=True)
            return None

    async def _maybe_backfill_corpus(self, curator: TaskCurator, project_root: str) -> None:
        """Trigger a one-shot background backfill if the collection is empty.

        Called after the curator is constructed (or lazily on first use).
        Checks collection point count via Qdrant. If count is 0 (or the
        collection doesn't exist), fetches the full task tree, flattens it,
        and awaits ``curator.backfill_corpus()`` inline (this method itself
        runs as a background task spawned by ``_get_curator``).

        The ``_backfill_triggered`` flag prevents re-triggering on subsequent
        calls. All failures degrade silently — nothing here must ever block
        task creation.
        """
        if self._backfill_triggered:
            return
        self._backfill_triggered = True

        try:
            project_id = resolve_project_id(project_root)

            # Check whether the collection already has tasks via the public API.
            count = await curator.corpus_count(project_id)

            if count > 0:
                logger.debug(
                    'task_curator: corpus for project %s has %d points — skipping auto-backfill',
                    project_id, count,
                )
                return

            # Fetch the task tree so we can backfill.
            if self.taskmaster is None:
                return
            try:
                tasks_result = await self.taskmaster.get_tasks(project_root)
                flat_tasks = flatten_task_tree(tasks_result)
            except Exception:
                logger.warning(
                    'task_curator: auto-backfill: get_tasks failed', exc_info=True,
                )
                return

            if not flat_tasks:
                return

            try:
                result = await curator.backfill_corpus(flat_tasks, project_id)
                logger.info(
                    'task_curator: auto-backfill complete — '
                    'upserted=%d skipped=%d errors=%d',
                    result.upserted, result.skipped, result.errors,
                )
            except Exception:
                logger.warning(
                    'task_curator: auto-backfill failed', exc_info=True,
                )

        except Exception:
            logger.warning('task_curator: _maybe_backfill_corpus raised', exc_info=True)

    @staticmethod
    def _build_candidate(kwargs: dict[str, Any]) -> CandidateTask | None:
        """Extract a CandidateTask from add_task / add_subtask kwargs.

        Returns None if there's no title (e.g. pure prompt-only add_task) —
        the curator cannot judge a candidate it cannot read.
        """
        title = str(kwargs.get('title') or '').strip()
        if not title:
            return None

        meta = kwargs.get('metadata') or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        files = meta.get('files_to_modify') or meta.get('modules') or []
        if isinstance(files, str):
            files = [files]
        files = [str(f) for f in files if f]

        return CandidateTask(
            title=title,
            description=str(kwargs.get('description') or ''),
            details=str(kwargs.get('details') or ''),
            files_to_modify=files,
            priority=str(kwargs.get('priority') or 'medium'),
            spawned_from=meta.get('spawned_from'),
            spawn_context=str(meta.get('spawn_context') or 'manual'),
        )

    async def _execute_combine(
        self,
        project_root: str,
        decision: CuratorDecision,
    ) -> dict | None:
        """Apply a curator combine decision to the target task.

        Returns the update_task result on success, None on failure (caller
        falls back to create). Combine is implemented via Taskmaster's
        ``update_task`` with a ``prompt`` that instructs a verbatim replacement
        plus metadata carrying the combine marker.

        Before writing, verifies the target's fingerprint and status. A
        mismatched fingerprint (curator targeted the wrong task) or terminal
        status (done/cancelled) aborts the write and returns None so the
        caller degrades to ``create`` instead of silently clobbering an
        unrelated task.
        """
        if decision.rewritten_task is None or decision.target_id is None:
            return None
        rt = decision.rewritten_task
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        # ── Guard: fetch live target and verify fingerprint + status ──
        try:
            raw_target = await tm.get_task(decision.target_id, project_root)
        except Exception as exc:
            logger.warning(
                'combine-guard: get_task failed for target=%s: %s — aborting combine',
                decision.target_id, exc,
            )
            return None

        target = _extract_task_dict(raw_target)
        if target is None:
            logger.warning(
                'combine-guard: target %s returned no task dict — aborting combine',
                decision.target_id,
            )
            return None

        target_status = str(target.get('status', '') or '')
        if target_status in {'done', 'cancelled'}:
            logger.warning(
                'combine-guard: target %s has terminal status %r — aborting '
                'combine to avoid silently losing candidate work',
                decision.target_id, target_status,
            )
            return None

        target_title = str(target.get('title', '') or '')
        expected_fp = normalize_title(target_title)
        got_fp = normalize_title(decision.target_fingerprint or '')
        if not got_fp or expected_fp != got_fp:
            logger.warning(
                'combine-guard: fingerprint mismatch for target=%s: '
                'expected=%r got=%r — aborting combine',
                decision.target_id,
                target_title[:80],
                (decision.target_fingerprint or '')[:80],
            )
            return None

        files_block = '\n'.join(f'  - {f}' for f in rt.files_to_modify)
        combine_prompt = (
            'Replace this task with EXACTLY the following fields, verbatim. '
            'Do not paraphrase, do not merge with existing content, do not add '
            'commentary. The task curator already produced this coherent '
            "rewrite that subsumes a duplicate task's work.\n\n"
            f'TITLE: {rt.title}\n\n'
            f'DESCRIPTION: {rt.description}\n\n'
            f'PRIORITY: {rt.priority}\n\n'
            f'FILES_TO_MODIFY:\n{files_block}\n\n'
            f'DETAILS:\n{rt.details}\n'
        )
        combine_metadata = json.dumps({
            'curator_action': 'combine',
            'curator_justification': decision.justification[:500],
            'combined_at': datetime.now(UTC).isoformat(),
        })

        # ── Audit: persist old-vs-new BEFORE the mutation so we can
        # recover if the write crashes mid-flight.
        _append_combine_audit(
            project_id=project_id,
            target_id=decision.target_id,
            old_title=target_title,
            old_description=str(target.get('description', '') or ''),
            old_status=target_status,
            new_title=rt.title,
            new_description=rt.description,
            justification=decision.justification,
        )

        try:
            return dict(await tm.update_task(
                task_id=decision.target_id,
                project_root=project_root,
                prompt=combine_prompt,
                metadata=combine_metadata,
                append=False,
            ))
        except Exception as exc:
            logger.warning(
                'task_curator: combine update failed for target=%s: %s',
                decision.target_id, exc,
            )
            return None

    async def _dedupe_bulk_created(
        self,
        project_root: str,
        pre_snapshot: Mapping[str, Any],
        parent_task_id: str | None = None,
    ) -> dict:
        """Post-hoc deduplication after a bulk task-creation operation.

        Reads the current task tree, diffs against pre_snapshot, then runs
        a two-pass deduplication on the newly-created tasks:

        **Pass 1 — intra-batch dedup (pre-pass)**:
        Groups new tasks by normalised (title, description) hash.  First
        occurrence wins; each subsequent duplicate is removed immediately
        under ``_write_lock`` with ``reason='intra_batch_duplicate'``.
        This is cheap (no LLM) and prevents duplicate curator calls.

        **Pass 2 — cross-task curator pass**:
        Invokes ``curator.curate()`` on each unique survivor from pass 1.
        Drop → remove the new task.  Combine → rewrite the target then
        remove the new task.  Create → keep and record.

        Called by both ``expand_task`` and ``parse_prd``; the two-pass
        structure is path-agnostic (``parent_task_id`` is only used as
        ``spawned_from`` metadata for curator candidates).

        Returns ``{'removed': [...], 'kept': [...], 'errors': []}``.
        Each ``removed`` entry carries a ``reason`` field that is one of:

        - ``'intra_batch_duplicate'`` — removed in pass 1 (mechanical,
          no LLM); entry has ``matched_task_id`` (first-occurrence id)
          but no ``justification``.
        - ``'curator_drop'`` — removed in pass 2 by the curator.
        - ``'curator_combine'`` — merged into an existing task in pass 2.
        """
        removed: list[dict] = []
        kept: list[dict] = []
        errors: list[dict] = []

        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        pre_ids = {
            str(t.get('id', '')) for t in flatten_task_tree(pre_snapshot)
            if t.get('id')
        }
        post_snapshot = await tm.get_tasks(project_root)
        new_task_dicts = [
            t for t in flatten_task_tree(post_snapshot)
            if str(t.get('id', '')) and str(t.get('id', '')) not in pre_ids
        ]
        if not new_task_dicts:
            return {'removed': removed, 'kept': kept, 'errors': errors}

        # ── Intra-batch dedup pre-pass ──────────────────────────────────────
        # Detect and remove tasks that are near-identical duplicates of another
        # task created in the SAME batch (e.g. the LLM emits 3 variants of the
        # same subtask).  This runs BEFORE the per-task curator loop so we
        # avoid wasting LLM calls on tasks we are about to remove.
        #
        # Key: normalised (title, description) hash — case + whitespace
        # insensitive, excluding files_to_modify (often absent).
        # First occurrence wins (matches curate_batch hash_to_first_idx
        # convention).
        #
        # Two-phase discipline:
        #   Phase A (outside the lock): hash each task, update seen_keys /
        #   unique_new_tasks, and collect duplicates into dups_to_remove.
        #   All work here is pure local-state; no I/O.
        #   Phase B (inside ONE _write_lock scope): issue all tm.remove_task
        #   calls with per-item try/except so a transient backend failure on
        #   one removal does not abort the rest of the batch.  The
        #   dual-append-on-error fall-through (failing task → both errors and
        #   unique_new_tasks/kept) is preserved inside the lock.
        seen_keys: dict[str, str] = {}   # key → first-occurrence task_id
        unique_new_tasks: list[dict] = []
        dups_to_remove: list[tuple] = []  # (tid, title, t, first_id)
        for t in new_task_dicts:
            tid = str(t.get('id', ''))
            title = str(t.get('title', ''))
            description = str(t.get('description', '') or '')

            # Guard: tasks with a blank title get an identical
            # _intra_batch_key('', '') hash regardless of description,
            # which would incorrectly collapse all malformed subtasks
            # into the first one.  Pass them straight through and let
            # the curator path's own empty-title guard decide.
            if not title.strip():
                unique_new_tasks.append(t)
                continue

            key = TaskCurator._intra_batch_key(title, description)
            if key not in seen_keys:
                seen_keys[key] = tid
                unique_new_tasks.append(t)
            else:
                first_id = seen_keys[key]
                dups_to_remove.append((tid, title, t, first_id))

        # Phase B: issue all removals inside a single lock scope to amortise
        # lock-handoff cost.  Per-item try/except inside the lock preserves
        # the partial-failure dual-append fall-through: a transient backend
        # error on one removal does not abort the rest of the batch.
        if dups_to_remove:
            async with self._write_lock(project_id):
                for tid, title, t, first_id in dups_to_remove:
                    try:
                        await tm.remove_task(tid, project_root)
                        removed.append({
                            'task_id': tid,
                            'title': title,
                            'reason': 'intra_batch_duplicate',
                            'matched_task_id': first_id,
                        })
                        logger.info(
                            'bulk_dedup: intra-batch duplicate %s removed (matches %s)',
                            tid, first_id,
                        )
                    except Exception as exc:
                        errors.append({'task_id': tid, 'title': title, 'error': str(exc)})
                        # Removal failed transiently — fall through to curator so
                        # this task still appears in `kept` rather than silently
                        # disappearing from both `removed` and `kept`.
                        unique_new_tasks.append(t)

        curator = await self._get_curator()
        if curator is None:
            for t in unique_new_tasks:
                kept.append({
                    'task_id': str(t.get('id', '')),
                    'title': str(t.get('title', '')),
                })
            return {'removed': removed, 'kept': kept, 'errors': errors}

        for t in unique_new_tasks:
            tid = str(t.get('id', ''))
            title = str(t.get('title', ''))
            pool_entry = _to_pool_entry(t, source='module', lock_depth=2)
            files = pool_entry.files_to_modify if pool_entry is not None else []
            candidate = CandidateTask(
                title=title,
                description=str(t.get('description', '') or ''),
                details=str(t.get('details', '') or ''),
                files_to_modify=files,
                priority=str(t.get('priority', 'medium')),
                spawned_from=parent_task_id,
                spawn_context='expand' if parent_task_id else 'parse_prd',
            )
            if not candidate.title.strip():
                kept.append({'task_id': tid, 'title': title})
                continue

            try:
                decision = await curator.curate(candidate, project_id, project_root)
            except Exception as exc:
                logger.warning('bulk_curator: curate() failed for %s: %s', tid, exc)
                kept.append({'task_id': tid, 'title': title})
                continue

            if decision.action == 'drop' and decision.target_id:
                try:
                    # WP-E: serialise the tasks.json mutation.
                    async with self._write_lock(project_id):
                        await tm.remove_task(tid, project_root)
                    removed.append({
                        'task_id': tid,
                        'title': title,
                        'reason': 'curator_drop',
                        'matched_task_id': decision.target_id,
                        'justification': decision.justification[:200],
                    })
                    logger.warning(
                        'bulk_curator: dropped duplicate task %s (matched %s): %s',
                        tid, decision.target_id, decision.justification[:80],
                    )
                except Exception as exc:
                    errors.append({'task_id': tid, 'title': title, 'error': str(exc)})
                continue

            if decision.action == 'combine' and decision.target_id:
                # WP-E: combine writes to the target *and* removes the new
                # task — hold the lock across both so no concurrent writer
                # can observe the intermediate "new still present" state.
                try:
                    async with self._write_lock(project_id):
                        combine_result = await self._execute_combine(project_root, decision)
                        if combine_result is None:
                            kept.append({'task_id': tid, 'title': title})
                            continue
                        await tm.remove_task(tid, project_root)
                except Exception as exc:
                    errors.append({'task_id': tid, 'title': title, 'error': str(exc)})
                    continue
                removed.append({
                    'task_id': tid,
                    'title': title,
                    'reason': 'curator_combine',
                    'matched_task_id': decision.target_id,
                    'justification': decision.justification[:200],
                })
                if decision.rewritten_task is not None:
                    rt_candidate = CandidateTask(
                        title=decision.rewritten_task.title,
                        description=decision.rewritten_task.description,
                        details=decision.rewritten_task.details,
                        files_to_modify=decision.rewritten_task.files_to_modify,
                        priority=decision.rewritten_task.priority,
                    )
                    bg = asyncio.create_task(
                        curator.reembed_task(
                            decision.target_id, rt_candidate, project_id,
                        ),
                        name=f'curator-reembed-{decision.target_id}',
                    )
                    self._background_tasks.add(bg)
                    bg.add_done_callback(lambda t: self._background_tasks.discard(t))
                continue

            # action == 'create' or degenerate fall-through
            kept.append({'task_id': tid, 'title': title})
            bg = asyncio.create_task(
                curator.record_task(tid, candidate, project_id),
                name=f'curator-record-{tid}',
            )
            self._background_tasks.add(bg)
            bg.add_done_callback(lambda t: self._background_tasks.discard(t))

        return {'removed': removed, 'kept': kept, 'errors': errors}

    # ── Write pass-throughs (emit event, no targeted reconciliation) ───

    def _write_lock(self, project_id: str) -> asyncio.Lock:
        """Per-project lock for tasks.json mutations.

        Short-held; covers only the Taskmaster stdio write. Every mutating
        op (set_task_status, update_task, add_dependency, remove_dependency,
        the actual tm.add_task/tm.add_subtask/tm.remove_task calls, and the
        bulk expand/parse_prd pre-snapshot+mutation) takes this lock.
        """
        lock = self._write_locks.get(project_id)
        if lock is None:
            lock = asyncio.Lock()
            self._write_locks[project_id] = lock
        return lock

    def _curator_lock(self, project_id: str) -> asyncio.Lock:
        """Per-project lock for curator-family operations.

        Long-held; covers ``curator.curate()`` (LLM round-trip) plus the
        post-write ``note_created``/``record_task`` steps so the next waiter
        on this lock sees the new entry on its pre-LLM check.

        Taken by: ``add_subtask``, ``remove_task`` (conservative: protects
        concurrent combine-target integrity) AND by ``_process_add_ticket``
        (the add_task worker path) so curator decisions for the three families
        are mutually exclusive within a project.  Without the worker taking
        this lock, concurrent add_subtask + add_task on the same project
        could both call curator.curate() against a stale snapshot and both
        decide "create" for what is effectively the same candidate.
        Lock order: curator_lock BEFORE write_lock. Short writes
        (set_task_status etc.) never take this lock and so are not blocked
        by an in-flight curator decision.

        NOTE: Routing ``add_subtask`` / ``remove_task`` through the per-project
        worker queue is a planned follow-up (out-of-scope for this task); until
        then the worker must acquire this lock to preserve R3.
        """
        lock = self._curator_locks.get(project_id)
        if lock is None:
            lock = asyncio.Lock()
            self._curator_locks[project_id] = lock
        return lock

    @staticmethod
    def _extract_metadata_dict(metadata) -> dict | None:
        """Best-effort parse of ``metadata`` into a dict, or None."""
        if metadata is None:
            return None
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except (json.JSONDecodeError, ValueError):
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    async def _check_escalation_idempotency(
        self, *, project_root: str, metadata,
    ) -> dict | None:
        """Return an existing task's identity if ``(escalation_id,
        suggestion_hash)`` in ``metadata`` matches a non-cancelled task.

        R4: authoritative and cheap — no LLM, no embedding lookup.
        Works even when the curator is disabled or broken, which is
        exactly the regime that triggered the duplicate Type::Error
        tasks (esc-1912-179 → esc-1912-190).
        """
        meta = self._extract_metadata_dict(metadata)
        if not meta:
            return None
        esc_id = meta.get('escalation_id')
        sug_hash = meta.get('suggestion_hash')
        if not isinstance(esc_id, str) or not isinstance(sug_hash, str):
            return None
        if not esc_id or not sug_hash:
            return None

        if self.taskmaster is None:
            return None
        try:
            tasks_result = await self.taskmaster.get_tasks(project_root)
        except Exception:
            logger.debug(
                'r4: get_tasks failed during idempotency check', exc_info=True,
            )
            return None
        for task in flatten_task_tree(tasks_result):
            tmeta = task.get('metadata')
            if not isinstance(tmeta, dict):
                continue
            if tmeta.get('escalation_id') != esc_id:
                continue
            if tmeta.get('suggestion_hash') != sug_hash:
                continue
            if str(task.get('status', '')) == 'cancelled':
                continue
            tid = str(task.get('id', ''))
            if not tid:
                continue
            logger.warning(
                'r4: idempotency hit — returning existing task %s for '
                'escalation_id=%s suggestion_hash=%s',
                tid, esc_id, sug_hash,
            )
            return {
                'id': tid,
                'title': str(task.get('title', '')),
                'deduplicated': True,
                'action': 'idempotency_hit',
                'reason': 'escalation+suggestion matched',
            }
        return None

    async def submit_task(self, project_root: str, **kwargs: Any) -> dict:
        """Phase-1 of the two-phase add: persist a ticket and return its id immediately.

        The curator decision (drop / combine / create) is deferred to the
        single-worker queue so concurrent callers never contend on a long LLM
        round-trip. Use ``resolve_ticket`` to block until the worker decides.

        Returns ``{'ticket': 'tkt_<id>'}`` so callers can poll or block via
        ``resolve_ticket``. Does NOT call ``tm.add_task``.

        Cancellation caveat
        -------------------
        If ``close()`` cancels the worker while ``tm.add_task`` is already
        mid-flight, the task may be written to ``tasks.json`` but the ticket
        row stays ``pending`` (the worker was interrupted before
        ``mark_resolved``).  On the next ``start()``, that ticket is marked
        ``failed/server_restart`` — a caller that retries without passing
        ``escalation_id`` + ``suggestion_hash`` metadata may therefore create a
        duplicate.  Callers in contexts where duplication is unacceptable
        should supply idempotency metadata so the R4 gate short-circuits the
        retry.
        """
        if self._closed:
            return {'error': 'TaskInterceptor is closed; cannot submit new tasks', 'error_type': 'ClosedError'}

        if self._ticket_store is None:
            return {'error': 'ticket_store not configured; cannot use submit_task', 'error_type': 'ConfigError'}

        if err := await self._backlog_gate(project_root):
            return err

        project_id = resolve_project_id(project_root)

        # Serialise the full call payload so the worker can reconstruct it.
        # Stored as a canonical JSON blob: {project_root, kwargs, metadata}.
        # No default=str: non-JSON-native values (e.g. datetime, Path, enum)
        # must fail fast here with a TypeError so the caller gets a structured
        # ValidationError rather than silently storing a mangled blob that the
        # worker cannot faithfully execute.
        metadata = kwargs.pop('metadata', None)
        try:
            blob = json.dumps({
                'project_root': project_root,
                'kwargs': kwargs,
                'metadata': metadata,
            })
        except (TypeError, ValueError) as exc:
            return {
                'error': f'submit_task: non-serialisable argument — {exc}',
                'error_type': 'ValidationError',
            }

        ticket_id = await self._ticket_store.submit(
            project_id=project_id,
            candidate_json=blob,
            ttl_seconds=600,
        )
        queue = self._ticket_queues.setdefault(project_id, asyncio.Queue())
        await queue.put(ticket_id)
        self._start_worker_if_needed(project_id)
        return {'ticket': ticket_id}

    async def _resolve_ticket_raw(
        self,
        ticket: str,
        timeout_seconds: float | None = None,
    ) -> tuple[dict, dict | None]:
        """Core resolve logic called by ``resolve_ticket``.

        Returns ``(public_result, raw_row)`` where:

        - *public_result* is ``{status, task_id?, reason?}`` suitable for MCP
          callers.
        - *raw_row* is the full ticket-store row dict (including ``result_json``)
          when the ticket reached a terminal state, or ``None`` for sentinel
          outcomes (unknown ticket, timeout, store not configured, server-closed).

        Lost-wakeup safety
        ------------------
        The per-ticket :class:`asyncio.Event` is registered in
        ``_ticket_events`` BEFORE the first ``ticket_store.get()`` call so the
        worker cannot signal between the terminal-status check and the event
        registration.

        Multiple waiters
        ----------------
        Multiple concurrent callers may await the same ticket; each appends its
        own event to the per-ticket list so no caller's ``event.wait()`` is
        stranded by a later registration.  ``_signal_ticket_event`` sets and
        removes all events atomically.

        Cleanup is centralised in a single ``finally`` block that removes just
        *this* event from the list (leaving other waiters' events untouched).

        Shutdown race
        -------------
        ``close()`` signals all pending ticket events then calls
        ``TicketStore.close()``.  Between the signal and the store close the
        event loop may schedule woken waiters, which then call
        ``_ticket_store.get()`` on a closed store; ``_require_db()`` raises
        ``RuntimeError`` in that window.  Both ``get()`` call-sites are wrapped
        in ``try/except RuntimeError`` so callers always receive the
        ``server_closed`` sentinel rather than a bare exception.
        """
        if self._ticket_store is None:
            return (
                {'status': 'failed', 'reason': 'ticket_store not configured', 'task_id': None},
                None,
            )

        # (1) Register the event BEFORE reading the row (lost-wakeup safety).
        event = asyncio.Event()
        self._ticket_events.setdefault(ticket, []).append(event)
        try:
            # (2) Read the ticket row.  Wrap in try/except RuntimeError so a
            # store that was closed concurrently returns the server_closed
            # sentinel rather than propagating the exception.
            try:
                row = await self._ticket_store.get(ticket)
            except RuntimeError as exc:
                logger.debug(
                    'resolve_ticket: ticket_store closed during initial read for %s: %s',
                    ticket, exc,
                )
                return ({'status': 'failed', 'reason': 'server_closed', 'task_id': None}, None)
            if row is None:
                return ({'status': 'failed', 'reason': 'unknown_ticket', 'task_id': None}, None)

            # (3) Terminal already — return immediately (finally cleans up).
            if row['status'] != 'pending':
                return (_format_ticket_result(row), row)

            # (4) Still pending — wait for the worker to signal.  The event
            # may already be set if the worker raced us between steps (1) and (2).
            try:
                if timeout_seconds is not None:
                    await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
                else:
                    await event.wait()
            except TimeoutError:
                return ({'status': 'failed', 'reason': 'timeout', 'task_id': None}, None)

            # (5) Re-load the (hopefully) now-terminal row.  Same RuntimeError
            # guard: close() may have run between the event signal and this
            # re-read (the shutdown race flagged by reviewer_comprehensive).
            try:
                row = await self._ticket_store.get(ticket)
            except RuntimeError as exc:
                logger.debug(
                    'resolve_ticket: ticket_store closed during post_wake read for %s: %s',
                    ticket, exc,
                )
                return ({'status': 'failed', 'reason': 'server_closed', 'task_id': None}, None)
            if row is None:
                return ({'status': 'failed', 'reason': 'unknown_ticket', 'task_id': None}, None)
            # Guard: if the row is still 'pending' after the event was set, the
            # worker was cancelled (close()) or raised before mark_resolved could
            # run.  Return a server_closed failure so the caller always receives a
            # terminal response — never a 'pending' state that could loop.
            if row['status'] == 'pending':
                return ({'status': 'failed', 'reason': 'server_closed', 'task_id': None}, None)
            return (_format_ticket_result(row), row)
        finally:
            # (6) Remove just this event from the per-ticket list; others stay.
            # _signal_ticket_event may have already popped the entire list —
            # the get() + remove() is a harmless no-op in that case.
            evs = self._ticket_events.get(ticket)
            if evs is not None:
                with contextlib.suppress(ValueError):
                    evs.remove(event)  # already removed by _signal_ticket_event
                if not evs:
                    self._ticket_events.pop(ticket, None)

    async def resolve_ticket(
        self,
        ticket: str,
        project_root: str,
        timeout_seconds: float | None = None,
    ) -> dict:
        """Phase-2 of the two-phase add: block until the worker decides.

        If the ticket is already terminal, returns immediately.  Otherwise
        registers an asyncio.Event and awaits it (optionally with
        *timeout_seconds*).  On timeout, returns a synthetic failed dict
        WITHOUT mutating the ticket row — the worker may still resolve it later,
        and the TTL sweep cleans truly abandoned rows.

        Returns ``{status, task_id?, reason?}`` — does NOT expose ``result_json``.

        Multiple concurrent callers may await the same ticket; each receives
        its own event so no caller is stranded by a later registration.

        See :meth:`_resolve_ticket_raw` for implementation details.
        """
        result, _ = await self._resolve_ticket_raw(ticket, timeout_seconds)
        return result

    def _start_worker_if_needed(self, project_id: str) -> None:
        """Lazily start the per-project curator worker asyncio.Task if not running.

        One worker per project_id so a slow LLM on project A does not delay
        add_task traffic on project B.  Within a project, tickets are still
        serialised (one curator.curate() at a time).

        Workers are intentionally NOT added to ``_background_tasks``: they are
        long-running infinite-loop tasks.  Adding them would cause ``drain()``
        to block forever.  Lifecycle is managed explicitly by ``close()``.

        TOCTOU safety: ``_closed`` is re-checked here (not only in
        ``submit_task``) so a worker created in the window between
        ``submit_task``'s closed check and ``close()`` completing is never
        left running untracked.  When closed, we simply don't start a new
        worker; the already-queued ticket will be marked
        ``failed/server_restart`` by the next ``start()`` call.
        """
        if self._ticket_store is None:
            return
        if self._closed:
            return
        existing = self._worker_tasks.get(project_id)
        if existing is None or existing.done():
            self._worker_tasks[project_id] = asyncio.create_task(
                self._curator_worker(project_id),
                name=f'task-interceptor-curator-worker-{project_id}',
            )

    async def _curator_worker(self, project_id: str) -> None:
        """Drain pending tickets for *project_id* from its per-project queue.

        Processes tickets in batches: blocks on ``queue.get()`` for the first
        ticket, then opportunistically drains up to ``batch_max - 1`` more via
        ``queue.get_nowait()`` (non-blocking).  This preserves low-latency for
        single submitters (no wait for batch fill) while amortising LLM cost
        when tickets arrive back-to-back under orchestrator load.

        Independent workers per project restore the per-project fairness of the
        old curator_lock approach: project B is never blocked by a slow LLM
        call for project A.

        Lifecycle: cancellable; ``close()`` cancels all worker tasks and awaits
        them so in-flight work is not silently dropped.
        """
        queue = self._ticket_queues.setdefault(project_id, asyncio.Queue())
        # Determine batch_max from config; fall back to the documented default (5)
        # when no config is available so that tests without explicit config still
        # exercise the batch path rather than silently falling back to serial.
        _raw = getattr(getattr(self._config, 'curator', None), 'batch_max', None)
        batch_max: int = (
            _raw
            if isinstance(_raw, int) and not isinstance(_raw, bool) and _raw >= 1
            else 5
        )  # falls back to CuratorConfig default (5) for None, bool, non-int, or ≤0

        try:
            while True:
                # Block on the first ticket — never spin.
                first_ticket = await queue.get()
                batch: list[str] = [first_ticket]

                # Opportunistic drain: grab more tickets without waiting.
                while len(batch) < batch_max:
                    try:
                        batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                try:
                    await self._process_add_tickets_batch(batch)
                except Exception:
                    logger.exception(
                        '_curator_worker: unhandled error processing batch of %d '
                        'tickets for project %s',
                        len(batch), project_id,
                    )
                    # Signal all ticket waiters so resolve_ticket callers are not
                    # blocked forever if _process_add_tickets_batch raised before
                    # calling _signal_ticket_event for each ticket.
                    for tid in batch:
                        self._signal_ticket_event(tid)
                finally:
                    # Mark all drained tickets as done in the queue.
                    for _ in batch:
                        queue.task_done()
        except asyncio.CancelledError:
            pass

    async def _dispatch_ticket_decision(
        self,
        *,
        ticket_id: str,
        project_root: str,
        project_id: str,
        candidate: CandidateTask | None,
        decision: CuratorDecision | None,
        kwargs: dict,
        metadata: Any,
        curator: Any,  # TaskCurator | None
        curator_degrade_reason: str | None = None,
    ) -> tuple[str, str | None, str | None, dict | None, str | None]:
        """Execute the drop/combine/create dispatch for one curator decision.

        Returns (status, task_id, reason, result_dict, curator_degrade_reason).
        Does NOT call mark_resolved, _signal_ticket_event, journal, or commit —
        those are the caller's responsibility.
        """
        if decision is not None and decision.action == 'drop' and decision.target_id:
            # Drop: fold candidate into the existing target task (status='combined').
            # Preserves legacy result shape for the add_task facade.
            result_dict = {
                'id': decision.target_id,
                'title': candidate.title if candidate else '',
                'deduplicated': True,
                'action': 'drop',
                'justification': decision.justification,
            }
            return (
                'combined',
                decision.target_id,
                f'drop: {decision.justification}',
                result_dict,
                None,
            )

        if decision is not None and decision.action == 'combine' and decision.target_id:
            # Combine: rewrite target task under write_lock, spawn reembed background task.
            async with self._write_lock(project_id):
                combine_result = await self._execute_combine(project_root, decision)
            if combine_result is not None:
                if decision.rewritten_task is not None and curator is not None:
                    rt_candidate = CandidateTask.from_rewritten_task(decision.rewritten_task)
                    bg = asyncio.create_task(
                        curator.reembed_task(
                            decision.target_id, rt_candidate, project_id,
                        ),
                        name=f'curator-reembed-{decision.target_id}',
                    )
                    self._background_tasks.add(bg)
                    bg.add_done_callback(lambda t: self._background_tasks.discard(t))
                result_dict = {
                    'id': decision.target_id,
                    'title': (
                        decision.rewritten_task.title
                        if decision.rewritten_task else (candidate.title if candidate else '')
                    ),
                    'deduplicated': True,
                    'action': 'combine',
                    'justification': decision.justification,
                }
                return (
                    'combined',
                    decision.target_id,
                    f'combine: {decision.justification}',
                    result_dict,
                    None,
                )
            # combine failed → fall through to create

        # ── Create task ───────────────────────────────────────────────
        status: str = 'failed'
        task_id: str | None = None
        reason: str | None = None
        result_dict: dict | None = None

        tm = await self._ensure_taskmaster()
        metadata_json: str | None = None
        if metadata:
            metadata_json = (
                metadata if isinstance(metadata, str) else json.dumps(metadata)
            )

        async with self._write_lock(project_id):
            try:
                result = await tm.add_task(
                    project_root=project_root,
                    metadata=metadata_json,
                    **kwargs,
                )
                atomic_metadata_written = metadata_json is not None
            except TypeError:
                result = await tm.add_task(project_root=project_root, **kwargs)
                atomic_metadata_written = False

            # TaskmasterBackend.add_task is contractually guaranteed to return
            # an AddTaskResult DTO with a non-empty `id` — anything else raises
            # TaskmasterError, which propagates out of the try block above.
            task_id_str = str(result['id'])
            # ── Latch success immediately — post-create errors must NOT
            # downgrade this to 'failed'. Once tm.add_task returns, the
            # task exists in tasks.json; marking the ticket 'failed' would
            # strand the task and cause duplicate-on-retry.
            task_id = task_id_str
            status = 'created'
            result_dict = dict(result)
            if curator_degrade_reason is not None:
                result_dict = {**result_dict, 'curator_degrade_reason': curator_degrade_reason}

            if metadata_json and task_id_str and not atomic_metadata_written:
                try:
                    await tm.update_task(
                        task_id=task_id_str,
                        metadata=metadata_json,
                        project_root=project_root,
                    )
                except Exception as e:
                    logger.warning(
                        '_dispatch_ticket_decision: metadata follow-up for %s failed: %s',
                        task_id_str, e,
                    )

            # note_created + record_task under curator_lock + write_lock
            # (R3 invariant): by holding curator_lock across curate() →
            # note_created, concurrent add_subtask curator calls cannot
            # race on a stale in-memory snapshot.
            # Each is wrapped independently: a failure appends to
            # post_create_warnings and is logged, but does NOT flip status.
            if curator is not None and candidate is not None and task_id_str:
                # task_id_str truthy ↔ we took the else-branch above, so
                # result_dict was assigned; assert to satisfy the type checker.
                assert result_dict is not None
                try:
                    curator.note_created(project_id, candidate, task_id_str)
                except Exception as exc:
                    logger.warning(
                        '_dispatch_ticket_decision: curator.note_created failed for %s: %s',
                        task_id_str, exc,
                    )
                    result_dict.setdefault('post_create_warnings', []).append(
                        {'stage': 'note_created', 'error': str(exc)}
                    )
                try:
                    await curator.record_task(task_id_str, candidate, project_id)
                except Exception as exc:
                    logger.warning(
                        '_dispatch_ticket_decision: curator.record_task failed for %s',
                        task_id_str, exc_info=True,
                    )
                    result_dict.setdefault('post_create_warnings', []).append(
                        {'stage': 'record_task', 'error': str(exc)}
                    )

        return (status, task_id, reason, result_dict, curator_degrade_reason)

    async def _process_add_ticket(self, ticket_id: str) -> None:
        """Run the full curator + write pipeline for a single ticket.

        Mirrors ``_add_task_locked`` but persists outcomes to the ticket store
        instead of returning them directly.  The worker calls this and the result
        is read back by ``resolve_ticket``.

        Terminal status values written:
        - ``created``  — task was created via ``tm.add_task``
        - ``combined`` — candidate was folded into an existing task (drop/combine/idempotency)
        - ``failed``   — an unrecoverable error occurred

        Decision order: R4 idempotency check → curator decision (drop /
        combine / create) → atomic write under ``_write_lock``.  On any
        unhandled exception in the write section, ticket is marked ``failed``
        and no ``task_created`` journal event is emitted.
        """
        if self._ticket_store is None:
            return

        row = await self._ticket_store.get(ticket_id)
        if row is None:
            logger.warning('_process_add_ticket: ticket %s not found in store', ticket_id)
            return
        if row['status'] != 'pending':
            logger.warning(
                '_process_add_ticket: ticket %s already terminal (%s), skipping',
                ticket_id, row['status'],
            )
            return

        try:
            blob = json.loads(row['candidate_json'])
        except Exception:
            logger.exception('_process_add_ticket: bad candidate_json for %s', ticket_id)
            await self._ticket_store.mark_resolved(
                ticket_id, status='failed', reason='bad_candidate_json',
            )
            self._signal_ticket_event(ticket_id)
            return

        project_root: str = blob['project_root']
        kwargs: dict = dict(blob.get('kwargs', {}))
        metadata = blob.get('metadata')
        project_id = resolve_project_id(project_root)

        # Rebuild candidate for curator
        kwargs_for_candidate = dict(kwargs)
        if metadata is not None:
            kwargs_for_candidate['metadata'] = metadata
        candidate = self._build_candidate(kwargs_for_candidate)

        status = 'failed'
        task_id: str | None = None
        reason: str | None = None
        result_dict: dict | None = None
        curator_degrade_reason: str | None = None

        try:
            # ── R3 invariant: serialise curator+write within this project ─
            # The worker runs per-project, but add_subtask / remove_task still
            # take _curator_lock in their own paths.  Without this acquisition,
            # a concurrent add_subtask could call curator.curate() against a
            # stale snapshot (note_created not yet published) and both paths
            # could independently decide 'create' for the same candidate.
            # Lock ordering: curator_lock (outer) → write_lock (inner), matching
            # remove_task and _add_subtask_locked.
            async with self._curator_lock(project_id):
                # ── R4: escalation-level idempotency ─────────────────────────
                # Short-circuits curator when (escalation_id, suggestion_hash) in
                # metadata matches a non-cancelled existing task — avoids duplicate
                # tasks when reconciliation retries an escalation suggestion.
                idempotency_hit = await self._check_escalation_idempotency(
                    project_root=project_root, metadata=metadata,
                )
                if idempotency_hit is not None:
                    existing_task_id = str(idempotency_hit.get('id', ''))
                    await self._ticket_store.mark_resolved(
                        ticket_id,
                        status='combined',
                        task_id=existing_task_id or None,
                        reason='idempotency_hit',
                        result_json=json.dumps(idempotency_hit),
                    )
                    self._signal_ticket_event(ticket_id)
                    return

                # ── Curator gate ─────────────────────────────────────────────
                curator = await self._get_curator()
                decision: CuratorDecision | None = None
                if curator is not None and candidate is not None:
                    try:
                        decision = await curator.curate(candidate, project_id, project_root)
                    except CuratorFailureError as exc:
                        logger.warning(
                            '_process_add_ticket: curator.curate raised CuratorFailureError '
                            'for %s; degrading to create. reason=%s',
                            ticket_id, exc,
                        )
                        # Degrade gracefully to create — avoids losing the task on
                        # transient LLM failures.  Record the failure so the facade
                        # and callers can see it in result_json.
                        curator_degrade_reason = str(exc)
                        decision = CuratorDecision(action='create')

                status, task_id, reason, result_dict, _ = (
                    await self._dispatch_ticket_decision(
                        ticket_id=ticket_id,
                        project_root=project_root,
                        project_id=project_id,
                        candidate=candidate,
                        decision=decision,
                        kwargs=kwargs,
                        metadata=metadata,
                        curator=curator,
                        curator_degrade_reason=curator_degrade_reason,
                    )
                )

        except asyncio.CancelledError:
            # Worker task cancelled (close() path) while inside _process_add_ticket.
            # Ensure the ticket reaches a terminal state even if mark_resolved wasn't
            # reached yet — cancellation may have interrupted the flow AFTER
            # tm.add_task already mutated tasks.json (status latched to 'created').
            # asyncio.shield() protects the mark_resolved call itself from being
            # interrupted by the propagated CancelledError.
            # TicketStore.mark_resolved() is idempotent (returns False when row is
            # already terminal), so calling it twice is harmless.
            logger.debug(
                '_process_add_ticket: cancelled for ticket %s; persisting status=%s',
                ticket_id, status,
            )
            try:
                await asyncio.shield(
                    self._ticket_store.mark_resolved(
                        ticket_id,
                        status=status,
                        task_id=task_id,
                        reason=reason if reason else (
                            'cancelled_during_write' if status != 'created' else None
                        ),
                        result_json=(
                            json.dumps(result_dict) if result_dict is not None else None
                        ),
                    )
                )
                self._signal_ticket_event(ticket_id)
            except Exception:
                logger.exception(
                    '_process_add_ticket: mark_resolved failed after cancellation '
                    'for ticket %s; ticket remains pending and will be cleaned up '
                    'by flush_pending_on_startup on next restart',
                    ticket_id,
                )
            raise

        except Exception as exc:
            logger.exception(
                '_process_add_ticket: unexpected error for ticket %s', ticket_id,
            )
            if status == 'created':
                # Failure happened after tm.add_task succeeded — preserve the
                # 'created' resolution; demote the error to a warning so the
                # facade and callers can surface it without losing the task.
                logger.warning(
                    '_process_add_ticket: post-create error for ticket %s, task %s: %s',
                    ticket_id, task_id, exc,
                )
                if result_dict is None:
                    result_dict = {}
                result_dict.setdefault('post_create_warnings', []).append(
                    {'stage': 'post_create', 'error': str(exc)}
                )
            else:
                reason = str(exc)
                status = 'failed'
                result_dict = None

        # ── Persist terminal state ────────────────────────────────────────
        # Shield from cancellation so that a close()-triggered CancelledError
        # that arrives here (after the try/except block) does not prevent the
        # ticket from reaching a terminal state.
        await asyncio.shield(
            self._ticket_store.mark_resolved(
                ticket_id,
                status=status,
                task_id=task_id,
                reason=reason,
                result_json=json.dumps(result_dict) if result_dict is not None else None,
            )
        )

        # ── Emit journal event and schedule commit (create path only) ────
        if status == 'created' and task_id:
            event = self._make_event(
                EventType.task_created,
                project_root,
                {'operation': 'add_task', 'task_id': task_id},
            )
            await self._journal(event)
            self._schedule_commit(project_root, 'add_task')

        self._signal_ticket_event(ticket_id)

    async def _process_add_tickets_batch(self, ticket_ids: list[str]) -> None:
        """Run the full curator + write pipeline for a batch of tickets.

        Calls ``curator.curate_batch`` once for all candidates, then dispatches
        each per-ticket decision under a single ``_curator_lock`` acquisition
        (R3 invariant).  All tickets in a batch must share the same project_id
        (guaranteed by the per-project queue draining in ``_curator_worker``).

        Terminal status values written (per ticket):
        - ``created``  — task was created via ``tm.add_task``
        - ``combined`` — candidate was folded into an existing task (drop/combine)
        - ``failed``   — an unrecoverable error occurred for that ticket

        Per-ticket failures are isolated: one ticket failing does not prevent
        other tickets from being processed.
        """
        if self._ticket_store is None:
            return
        if not ticket_ids:
            return

        # ── Load and validate all tickets ─────────────────────────────────────
        # Build per-ticket data; skip missing or already-terminal rows.
        ticket_data: list[tuple[str, dict, dict, str, dict, Any, CandidateTask | None]] = []
        # (ticket_id, row, blob, project_root, kwargs, metadata, candidate)

        project_id: str | None = None
        project_root: str | None = None

        for ticket_id in ticket_ids:
            row = await self._ticket_store.get(ticket_id)
            if row is None:
                logger.warning(
                    '_process_add_tickets_batch: ticket %s not found in store, skipping',
                    ticket_id,
                )
                continue
            if row['status'] != 'pending':
                logger.warning(
                    '_process_add_tickets_batch: ticket %s already terminal (%s), skipping',
                    ticket_id, row['status'],
                )
                continue

            try:
                blob = json.loads(row['candidate_json'])
            except Exception:
                logger.exception(
                    '_process_add_tickets_batch: bad candidate_json for %s', ticket_id,
                )
                await self._ticket_store.mark_resolved(
                    ticket_id, status='failed', reason='bad_candidate_json',
                )
                self._signal_ticket_event(ticket_id)
                continue

            t_project_root: str = blob['project_root']
            t_kwargs: dict = dict(blob.get('kwargs', {}))
            t_metadata = blob.get('metadata')
            t_project_id = resolve_project_id(t_project_root)

            # All tickets in the batch must share project_id (per-project queue invariant).
            if project_id is None:
                project_id = t_project_id
                project_root = t_project_root
            elif t_project_id != project_id:
                logger.warning(
                    '_process_add_tickets_batch: ticket %s has project_id=%s, '
                    'expected %s; skipping',
                    ticket_id, t_project_id, project_id,
                )
                continue

            # Rebuild candidate for curator.
            kwargs_for_candidate = dict(t_kwargs)
            if t_metadata is not None:
                kwargs_for_candidate['metadata'] = t_metadata
            candidate = self._build_candidate(kwargs_for_candidate)

            ticket_data.append((
                ticket_id, row, blob, t_project_root, t_kwargs, t_metadata, candidate,
            ))

        if not ticket_data or project_id is None or project_root is None:
            return

        # ── Curator + dispatch under a single curator_lock ─────────────────────
        async with self._curator_lock(project_id):
            # Get curator once for the whole batch.
            curator = await self._get_curator()

            # ── R4 idempotency checks (per-ticket, before curator call) ──────
            # Same short-circuit that _process_add_ticket performs.  Tickets
            # that match an existing non-cancelled task are resolved 'combined'
            # immediately and excluded from the curate_batch call.
            active_ticket_data: list[tuple] = []
            for entry in ticket_data:
                tid, _, _, t_pr, _, t_meta, _ = entry
                idempotency_hit = await self._check_escalation_idempotency(
                    project_root=t_pr, metadata=t_meta,
                )
                if idempotency_hit is not None:
                    existing_id = str(idempotency_hit.get('id', ''))
                    await self._ticket_store.mark_resolved(
                        tid,
                        status='combined',
                        task_id=existing_id or None,
                        reason='idempotency_hit',
                        result_json=json.dumps(idempotency_hit),
                    )
                    self._signal_ticket_event(tid)
                else:
                    active_ticket_data.append(entry)
            ticket_data = active_ticket_data

            if not ticket_data:
                return  # All tickets short-circuited by idempotency

            # Build candidates list for curate_batch.
            candidates: list[CandidateTask | None] = [
                entry[6] for entry in ticket_data
            ]
            non_none_candidates = [c for c in candidates if c is not None]
            # Per-ticket curator degrade reason (populated on fallback path).
            curator_degrade_reasons: list[str | None] = [None] * len(ticket_data)

            # Call curate_batch — one LLM round-trip for all candidates.
            decisions: list[CuratorDecision | None]
            if curator is not None and non_none_candidates:
                try:
                    batch_decisions = await curator.curate_batch(
                        non_none_candidates, project_id, project_root,
                    )
                    # Map decisions back to ticket_data-space (some candidates are None).
                    # batch_target_index emitted by curate_batch is in non_none-space
                    # (positions within non_none_candidates, which is what the LLM saw).
                    # We must remap it to ticket_data-space (positions in candidates)
                    # before the topological dispatch loop, which keys resolved_task_ids
                    # by ticket_data-space indices.
                    non_none_to_ticket_data = [
                        i for i, c in enumerate(candidates) if c is not None
                    ]
                    batch_idx = 0
                    decisions = []
                    for c in candidates:
                        if c is None:
                            decisions.append(None)
                        else:
                            bd = (
                                batch_decisions[batch_idx]
                                if batch_idx < len(batch_decisions)
                                else None
                            )
                            # Remap batch_target_index: non_none-space → ticket_data-space.
                            # Single-item curate() calls in the fallback path cannot emit
                            # batch_target_index, so no remap is needed there.
                            if bd is not None and bd.batch_target_index is not None:
                                local_bti = bd.batch_target_index
                                if 0 <= local_bti < len(non_none_to_ticket_data):
                                    remapped_bti = non_none_to_ticket_data[local_bti]
                                    bd = dataclasses.replace(
                                        bd, batch_target_index=remapped_bti,
                                    )
                                else:
                                    # Out-of-range in non_none-space: degrade to create
                                    # to avoid substituting a wrong task_id.
                                    logger.warning(
                                        '_process_add_tickets_batch: batch_target_index=%d '
                                        'out of non_none range [0, %d) for ticket %d; '
                                        'degrading to create',
                                        local_bti, len(non_none_to_ticket_data), batch_idx,
                                    )
                                    bd = CuratorDecision(
                                        action='create',
                                        justification=(
                                            f'batch-target-out-of-range: '
                                            f'local={local_bti}'
                                        ),
                                    )
                            decisions.append(bd)
                            batch_idx += 1
                except (CuratorFailureError, AllAccountsCappedException) as exc:
                    # Whole batch LLM failure: fall back to individual curate() calls.
                    # This path preserves per-ticket curator_degrade_reason so that
                    # result_json records the failure reason (same as _process_add_ticket).
                    logger.warning(
                        '_process_add_tickets_batch: curate_batch raised %s for '
                        'project %s; falling back to %d individual curate() calls',
                        type(exc).__name__, project_id, len(non_none_candidates),
                    )
                    batch_idx = 0
                    decisions = []
                    for i, c in enumerate(candidates):
                        if c is None:
                            decisions.append(None)
                        else:
                            try:
                                d = await curator.curate(c, project_id, project_root)
                                decisions.append(d)
                            except CuratorFailureError as e:
                                decisions.append(CuratorDecision(action='create'))
                                curator_degrade_reasons[i] = str(e)
                            batch_idx += 1
                except Exception:  # pyright: ignore[reportUnusedExcept]
                    logger.exception(
                        '_process_add_tickets_batch: curate_batch failed for project %s; '
                        'degrading all %d candidates to create',
                        project_id, len(non_none_candidates),
                    )
                    decisions = [None] * len(ticket_data)
            else:
                decisions = [None] * len(ticket_data)

            # ── Topologically-ordered dispatch ────────────────────────────────
            # Items with batch_target_index wait for their target to be
            # dispatched first so we can substitute the target's task_id into
            # the dependent's mark_resolved.  We use an iterative pass that
            # drains items that are ready (target already resolved or no
            # batch_target_index).  After each full pass we check for progress;
            # if none is made (all remaining items form a cycle), we coerce them
            # to create and drain.  Cycles should not occur in practice because
            # _parse_batch_decisions already degrades cycles, but we handle them
            # defensively here too.

            # resolved_task_ids[i] = task_id string (or None on failure) once item i is done.
            resolved_task_ids: dict[int, str | None] = {}
            pending_indices: list[int] = list(range(len(ticket_data)))

            while pending_indices:
                made_progress = False
                still_pending: list[int] = []
                for i in pending_indices:
                    dec = decisions[i] if i < len(decisions) else None
                    batch_target = dec.batch_target_index if dec is not None else None

                    # Can dispatch if: no batch_target_index, OR target already resolved.
                    if batch_target is not None and batch_target not in resolved_task_ids:
                        still_pending.append(i)
                        continue

                    # Resolve the decision: if this is a within-batch drop, substitute
                    # the sibling's task_id (or degrade to create if sibling failed).
                    effective_decision = dec
                    if (
                        dec is not None
                        and dec.action == 'drop'
                        and batch_target is not None
                    ):
                        sibling_task_id = resolved_task_ids.get(batch_target)
                        if sibling_task_id:
                            # Sibling created or combined — use its task_id.
                            effective_decision = CuratorDecision(
                                action='drop',
                                target_id=sibling_task_id,
                                justification=(
                                    f'batch_target_index={batch_target}: '
                                    f'{dec.justification}'
                                ),
                            )
                        else:
                            # Sibling failed — degrade this item to create.
                            logger.warning(
                                '_process_add_tickets_batch: batch_target_index=%d '
                                'for ticket %s had no task_id; degrading to create',
                                batch_target, ticket_data[i][0],
                            )
                            effective_decision = CuratorDecision(
                                action='create',
                                justification='batch-sibling-failed: degraded to create',
                            )

                    ticket_id, _, _, t_project_root, t_kwargs, t_metadata, candidate = ticket_data[i]
                    status = 'failed'
                    task_id: str | None = None
                    reason: str | None = None
                    result_dict: dict | None = None

                    try:
                        status, task_id, reason, result_dict, _ = (
                            await self._dispatch_ticket_decision(
                                ticket_id=ticket_id,
                                project_root=t_project_root,
                                project_id=project_id,
                                candidate=candidate,
                                decision=effective_decision,
                                kwargs=t_kwargs,
                                metadata=t_metadata,
                                curator=curator,
                                curator_degrade_reason=(
                                    curator_degrade_reasons[i]
                                    if i < len(curator_degrade_reasons)
                                    else None
                                ),
                            )
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.exception(
                            '_process_add_tickets_batch: dispatch failed for ticket %s',
                            ticket_id,
                        )
                        status = 'failed'
                        reason = str(exc)
                        task_id = None
                        result_dict = None

                    # Persist terminal state.
                    await asyncio.shield(
                        self._ticket_store.mark_resolved(
                            ticket_id,
                            status=status,
                            task_id=task_id,
                            reason=reason,
                            result_json=json.dumps(result_dict) if result_dict is not None else None,
                        )
                    )

                    # Emit journal event and schedule commit (create path only).
                    if status == 'created' and task_id:
                        event = self._make_event(
                            EventType.task_created,
                            t_project_root,
                            {'operation': 'add_task', 'task_id': task_id},
                        )
                        await self._journal(event)
                        self._schedule_commit(t_project_root, 'add_task')

                    self._signal_ticket_event(ticket_id)

                    # Record this item's result for downstream sibling drops.
                    resolved_task_ids[i] = task_id
                    made_progress = True

                pending_indices = still_pending

                if pending_indices and not made_progress:
                    # Remaining items form a cycle (should not happen after parser
                    # cycle detection, but handle defensively).
                    logger.warning(
                        '_process_add_tickets_batch: cycle in batch_target_index '
                        'graph for project %s; coercing %d items to create',
                        project_id, len(pending_indices),
                    )
                    for i in pending_indices:
                        dec = decisions[i] if i < len(decisions) else None
                        decisions[i] = CuratorDecision(
                            action='create',
                            justification='batch-cycle: coerced to create',
                        )
                    # Don't update pending_indices — the next iteration will drain them.

    def _signal_ticket_event(self, ticket_id: str) -> None:
        """Wake all callers waiting on resolve_ticket for this ticket.

        Pops the entire event list for *ticket_id* atomically so no caller is
        signalled twice.  Invariant: after this call there is no entry for
        *ticket_id* in ``_ticket_events``; ``resolve_ticket``'s ``finally``
        block harmlessly no-ops its subsequent ``evs.remove()`` attempt on the
        already-popped (and discarded) list via ``contextlib.suppress(ValueError)``.
        This pop-then-set ordering is intentional: asyncio is single-threaded, so
        there is no true data race, but the explicit pop makes the ownership
        transfer clear and prevents a future refactor from accidentally re-adding
        an already-signalled event.
        """
        events = self._ticket_events.pop(ticket_id, None)
        if events:
            for event in events:
                event.set()

    async def update_task(
        self, task_id: str, project_root: str, **kwargs: Any,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # WP-E: serialise the write; re-embed below reads only and stays
        # outside the lock.
        async with self._write_lock(project_id):
            result: dict[str, Any] = dict(await tm.update_task(
                task_id=task_id, project_root=project_root, **kwargs,
            ))
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'operation': 'update_task'},
        )
        await self._journal(event)
        self._schedule_commit(project_root, f'update_task({task_id})')

        # Re-embed if any corpus-relevant field changed. Taskmaster's update_task
        # accepts a free-form ``prompt`` for AI-driven edits, so we can't know
        # exactly what changed without re-fetching. Re-embed unconditionally when
        # the caller passed any of these hints.
        should_reembed = any(
            k in kwargs for k in ('prompt', 'title', 'description', 'details')
        )
        if should_reembed:
            curator = await self._get_curator()
            if curator is not None:
                try:
                    refreshed = await tm.get_task(task_id, project_root)
                    candidate = CandidateTask(
                        title=str(refreshed.get('title', '') or ''),
                        description=str(refreshed.get('description', '') or ''),
                        details=str(refreshed.get('details', '') or ''),
                        files_to_modify=[],
                        priority=str(refreshed.get('priority', 'medium')),
                    )
                    if candidate.title:
                        project_id = resolve_project_id(project_root)
                        bg = asyncio.create_task(
                            curator.reembed_task(
                                task_id, candidate, project_id,
                            ),
                            name=f'curator-reembed-{task_id}',
                        )
                        self._background_tasks.add(bg)
                        bg.add_done_callback(
                            lambda t: self._background_tasks.discard(t),
                        )
                except Exception:
                    logger.debug(
                        'task_curator: reembed_task on update failed for %s',
                        task_id, exc_info=True,
                    )
        return result

    async def add_subtask(
        self, parent_id: str, project_root: str, **kwargs: Any,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        candidate = self._build_candidate(kwargs)
        project_id = resolve_project_id(project_root)
        # curator_lock across curator.curate + note_created/record_task;
        # write_lock acquired internally for the brief tm.add_subtask call.
        async with self._curator_lock(project_id):
            return await self._add_subtask_locked(
                parent_id=parent_id,
                project_root=project_root,
                project_id=project_id,
                candidate=candidate,
                kwargs=kwargs,
            )

    async def _add_subtask_locked(
        self,
        *,
        parent_id: str,
        project_root: str,
        project_id: str,
        candidate,
        kwargs: dict,
    ) -> dict:
        # Curator gate for subtasks — previously bypassed entirely.
        curator = await self._get_curator()
        if curator is not None and candidate is not None:
            candidate.spawned_from = str(parent_id)
            candidate.spawn_context = candidate.spawn_context or 'manual'
            decision = await curator.curate(candidate, project_id, project_root)
            if decision.action == 'drop' and decision.target_id:
                logger.warning(
                    'task_curator: drop (subtask) — returning existing task %s '
                    'instead of creating duplicate: %s',
                    decision.target_id, candidate.title[:80],
                )
                return {
                    'id': decision.target_id,
                    'title': candidate.title,
                    'deduplicated': True,
                    'action': 'drop',
                    'justification': decision.justification,
                }
            if decision.action == 'combine' and decision.target_id:
                # _execute_combine writes; wrap in write_lock (curator_lock
                # is already held by the caller).
                async with self._write_lock(project_id):
                    combine_result = await self._execute_combine(project_root, decision)
                if combine_result is not None:
                    logger.warning(
                        'task_curator: combine (subtask) — folded into task %s',
                        decision.target_id,
                    )
                    return {
                        'id': decision.target_id,
                        'title': (
                            decision.rewritten_task.title
                            if decision.rewritten_task else candidate.title
                        ),
                        'deduplicated': True,
                        'action': 'combine',
                        'justification': decision.justification,
                    }

        tm = await self._ensure_taskmaster()
        # write_lock across the actual tm.add_subtask call so concurrent
        # set_task_status / update_task on the same project see a consistent
        # tasks.json view.
        async with self._write_lock(project_id):
            result = dict(await tm.add_subtask(
                parent_id=parent_id, project_root=project_root, **kwargs,
            ))
        event = self._make_event(
            EventType.task_created,
            project_root,
            {'parent_id': parent_id, 'operation': 'add_subtask'},
        )
        await self._journal(event)
        self._schedule_commit(project_root, f'add_subtask({parent_id})')

        # Record the new subtask in the curator corpus (synchronous cache
        # update + awaited Qdrant upsert — see add_task for the rationale).
        # AddSubtaskResult DTO guarantees a non-empty `id` on success.
        if curator is not None and candidate is not None:
            new_id = str(result['id'])
            curator.note_created(project_id, candidate, new_id)
            try:
                await curator.record_task(new_id, candidate, project_id)
            except Exception:
                logger.warning(
                    'add_subtask: curator.record_task awaited path failed for %s',
                    new_id, exc_info=True,
                )
        return result

    async def remove_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # curator_lock (conservative): a concurrent add_task / add_subtask
        # curator decision of ``combine target=task_id`` would hold the
        # curator_lock; blocking remove_task here prevents the target
        # from vanishing between the curator's decision and
        # _execute_combine's guarded write. _execute_combine also has a
        # fingerprint-and-status guard that degrades to ``create`` on
        # mismatch, so this is belt-and-braces, not load-bearing.
        async with (
            self._curator_lock(project_id),
            self._write_lock(project_id),
        ):
            result = dict(await tm.remove_task(task_id, project_root, tag))
        event = self._make_event(
            EventType.task_deleted,
            project_root,
            {'task_id': task_id, 'operation': 'remove_task'},
        )
        await self._journal(event)
        self._schedule_commit(project_root, f'remove_task({task_id})')
        return result

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # WP-E: serialise against concurrent mutations on the same project.
        async with self._write_lock(project_id):
            result = dict(await tm.add_dependency(
                task_id, depends_on, project_root, tag,
            ))
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'depends_on': depends_on, 'operation': 'add_dependency'},
        )
        await self._journal(event)
        self._schedule_commit(project_root, f'add_dependency({task_id}<-{depends_on})')
        return result

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # WP-E: serialise against concurrent mutations on the same project.
        async with self._write_lock(project_id):
            result = dict(await tm.remove_dependency(
                task_id, depends_on, project_root, tag,
            ))
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'depends_on': depends_on, 'operation': 'remove_dependency'},
        )
        await self._journal(event)
        self._schedule_commit(project_root, f'remove_dependency({task_id}<-{depends_on})')
        return result

    # ── Pure reads (direct pass-through) ───────────────────────────────

    async def get_tasks(
        self, project_root: str, tag: str | None = None
    ) -> dict:
        tm = await self._ensure_taskmaster()
        return dict(await tm.get_tasks(project_root, tag))

    async def get_statuses(
        self,
        project_root: str,
        ids: list[str] | None = None,
        tag: str | None = None,
    ) -> dict[str, str]:
        """Return a ``{id_str: status_str}`` mapping for tasks in *project_root*.

        This is a pure read — no events are emitted, no journal entry is written.

        Args:
            project_root: Absolute path to project root.
            ids: When given, only these task ids are returned (unknown ids are
                 silently omitted).  ``None`` returns all tasks.  ``[]`` returns
                 ``{}``.
            tag: Tag context forwarded to ``get_tasks`` (optional).

        Notes:
            Tasks whose dict is missing the ``'status'`` key are included with
            the sentinel value ``'unknown'``.  Callers that need to distinguish
            a genuine ``'unknown'`` status from a missing field should treat any
            ``'unknown'`` value as indeterminate.
        """
        tm = await self._ensure_taskmaster()
        raw = await tm.get_tasks(project_root, tag)
        task_list = raw.get('tasks', [])

        ids_set: set[str] | None = (
            {str(i) for i in ids} if ids is not None else None
        )
        mapping: dict[str, str] = {}
        for t in task_list:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            # Filter early: if the caller supplied an ids list, skip tasks
            # that are not in it rather than building the full mapping first.
            if ids_set is not None and tid not in ids_set:
                continue
            mapping[tid] = str(t.get('status', 'unknown'))

        return mapping

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict:
        tm = await self._ensure_taskmaster()
        return await tm.get_task(task_id, project_root, tag)

    def _require_done_provenance(self) -> bool:
        """True when the provenance gate is enforcing (reject missing/invalid).

        False during phased rollout — in that mode a missing/malformed
        provenance payload logs a warning and the transition proceeds.
        """
        cfg = self._config
        if cfg is None:
            return False
        recon = getattr(cfg, 'reconciliation', None)
        return bool(getattr(recon, 'require_done_provenance', False))


def _extract_status(task_data: dict) -> str:
    """Extract status from Taskmaster get_task response."""
    if 'status' in task_data:
        return task_data['status']
    data = task_data.get('data', {})
    if isinstance(data, dict):
        return data.get('status', 'unknown')
    return 'unknown'


def _extract_metadata_files(task_data: Any) -> list[str]:
    """Return ``metadata.files`` as a list[str] from a Taskmaster task dict.

    Silently returns ``[]`` when the field is absent, empty, or malformed —
    the phantom-done gate only fires when files is a non-empty list of
    strings, so defensive behaviour here means "do not gate".
    """
    if not isinstance(task_data, dict):
        return []
    metadata = task_data.get('metadata')
    if not isinstance(metadata, dict):
        return []
    files = metadata.get('files')
    if not isinstance(files, list):
        return []
    return [f for f in files if isinstance(f, str) and f]


def _missing_files(project_root: str, declared: list[str]) -> list[str]:
    """Return the subset of ``declared`` that does not exist under ``project_root``."""
    root = Path(project_root)
    return [f for f in declared if not (root / f).exists()]


def _terminal_exit_error(task_id: str, from_status: str, to_status: str) -> dict:
    """Structured error returned when the terminal-exit gate trips.

    Mirrors :func:`_done_gate_error` in shape so MCP callers (scheduler,
    steward, human un-defer scripts) can handle the rejection uniformly.
    """
    return {
        'success': False,
        'error': 'terminal_exit_rejected',
        'task_id': task_id,
        'from_status': from_status,
        'to_status': to_status,
        'hint': (
            f'Cannot transition task from {from_status!r} to {to_status!r} '
            'without reopen_reason. Terminal statuses (done, cancelled) are '
            'server-side frozen; pass reopen_reason="<short explanation>" to '
            "override — e.g. 'un-defer script', 'manual re-scope', "
            "'reconciliation: re-implementation required'. The reason is "
            'persisted as task metadata for audit.'
        ),
    }


def _done_gate_error(task_id: str, declared: list[str], missing: list[str]) -> dict:
    """Structured error returned when the phantom-done gate trips."""
    return {
        'success': False,
        'error': 'done_gate_missing_files',
        'task_id': task_id,
        'missing_files': missing,
        'files_checked': declared,
        'hint': (
            'Cannot mark task done: files listed in metadata.files do not exist '
            "at project_root. Either the implementation wasn't committed, or "
            'metadata.files is stale. Land the implementation (or fix '
            'metadata.files via update_task) before retrying.'
        ),
    }


async def _validate_done_provenance(
    task_id: str,
    raw: object,
    project_root: str,
    *,
    require: bool,
) -> tuple[dict | None, dict | None]:
    """Validate + resolve done_provenance for set_task_status(done).

    Returns ``(error_payload, resolved_provenance)``. Error payload is a
    structured dict suitable for returning to the MCP caller; when it is
    non-None the transition must be aborted. Resolved provenance is a dict
    with normalised ``commit`` (full 40-char SHA) and/or ``note`` keys, plus
    the original ``commit_input`` when a short hash or ref was resolved.

    When ``require`` is False, missing/empty provenance logs a warning but
    returns ``(None, None)`` so the transition proceeds. Malformed provenance
    (wrong type, unresolvable commit) still errors regardless of ``require``
    — we never want to record corrupt provenance on the task.
    """
    if raw is None or raw == {}:
        if require:
            return _done_provenance_missing_error(task_id), None
        logger.warning(
            'set_task_status(%s, done) called without done_provenance; '
            'Stage-2 reconciliation will treat this task as provenance-unknown. '
            'Pass done_provenance={"commit": "..."} or {"note": "..."} '
            'to record verified evidence.',
            task_id,
        )
        return None, None

    if not isinstance(raw, dict):
        return (
            _done_provenance_error(
                task_id,
                'done_provenance must be an object with keys "commit" and/or "note"',
            ),
            None,
        )

    commit_input = raw.get('commit')
    note = raw.get('note')

    if commit_input is not None and not isinstance(commit_input, str):
        return _done_provenance_error(task_id, 'commit must be a string'), None
    if note is not None and not isinstance(note, str):
        return _done_provenance_error(task_id, 'note must be a string'), None

    commit_input = (commit_input or '').strip() or None
    note = (note or '').strip() or None

    if commit_input is None and note is None:
        return (
            _done_provenance_error(
                task_id,
                'done_provenance requires at least one non-empty commit or note',
            ),
            None,
        )

    resolved: dict = {}
    if commit_input is not None:
        sha_or_err = await _resolve_commit_sha(project_root, commit_input)
        if isinstance(sha_or_err, dict):
            return _done_provenance_error(
                task_id,
                f'commit {commit_input!r} not found in {project_root}: '
                f'{sha_or_err.get("reason", "rev-parse failed")}',
            ), None
        resolved['commit'] = sha_or_err
        if sha_or_err != commit_input:
            resolved['commit_input'] = commit_input
    if note is not None:
        resolved['note'] = note

    return None, resolved


async def _resolve_commit_sha(project_root: str, commit: str) -> str | dict:
    """Resolve a tree-ish to a 40-char SHA via ``git rev-parse --verify``.

    Returns the SHA on success or a ``{'reason': ...}`` dict on failure.
    Uses a commit-peeling suffix (``^{commit}``) so a tag that points at a
    commit resolves to the commit SHA directly.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'rev-parse', '--verify',
            f'{commit}^{{commit}}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except TimeoutError:
            proc.kill()
            return {'reason': 'git rev-parse timed out'}
    except FileNotFoundError:
        return {'reason': 'git binary not found'}
    except Exception as e:
        return {'reason': f'{type(e).__name__}: {e}'}

    if proc.returncode != 0:
        msg = (stderr.decode('utf-8', errors='replace') or '').strip() or 'no such ref'
        return {'reason': msg}
    sha = stdout.decode('utf-8', errors='replace').strip()
    if not sha or len(sha) < 7:
        return {'reason': 'empty rev-parse output'}
    return sha


def _done_provenance_missing_error(task_id: str) -> dict:
    return {
        'success': False,
        'error': 'done_provenance_required',
        'task_id': task_id,
        'hint': (
            'Cannot mark task done: done_provenance is required when '
            'reconciliation.require_done_provenance is enabled. Pass '
            'done_provenance={"commit": "<sha-or-ref>"} with the merge '
            'commit that landed the implementation, or '
            'done_provenance={"note": "<explanation>"} when no commit applies '
            '(fast-forward merge, covered by sibling task, interactive session).'
        ),
    }


def _done_provenance_error(task_id: str, reason: str) -> dict:
    return {
        'success': False,
        'error': 'done_provenance_invalid',
        'task_id': task_id,
        'reason': reason,
    }


def _extract_task_dict(raw: Any) -> dict | None:
    """Return ``raw`` if it's a dict, else None.

    Retained as a thin guard for callers that still want defensive
    None-handling; the TaskmasterBackend adapter now returns a flat task
    dict from ``get_task``, so the legacy ``{'data': {...}}`` unwrap
    path is no longer needed.
    """
    return raw if isinstance(raw, dict) else None


def _combine_audit_path() -> Path:
    """Resolve the combine-audit log path.

    Honours ``DARK_FACTORY_DATA_DIR``; defaults to ``data/`` relative to the
    process CWD (which for the shared systemd server is the repo root).
    """
    return Path(os.getenv('DARK_FACTORY_DATA_DIR', 'data')) / 'combine_audit.jsonl'


def _append_combine_audit(
    *,
    project_id: str,
    target_id: str,
    old_title: str,
    old_description: str,
    old_status: str,
    new_title: str,
    new_description: str,
    justification: str,
) -> None:
    """Append a one-line JSON record documenting an about-to-happen combine.

    Append-only, best-effort. Failures log WARN but never propagate —
    a flaky audit write should not block task-merge progress.
    """
    record = {
        'ts': datetime.now(UTC).isoformat(),
        'project_id': project_id,
        'target_id': target_id,
        'curator_decision_id': str(uuid_mod.uuid4()),
        'old': {
            'title': old_title,
            'description_truncated': old_description[:500],
            'status': old_status,
        },
        'new': {
            'title': new_title,
            'description_truncated': new_description[:500],
        },
        'justification_truncated': justification[:500],
    }
    path = _combine_audit_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
    except Exception as exc:
        logger.warning(
            'combine-audit: failed to append audit record for target=%s: %s',
            target_id, exc,
        )
