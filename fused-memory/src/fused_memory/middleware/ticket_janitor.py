"""Periodic background sweep that surfaces failed tickets to the orchestrator.

Replaces the per-call ``resolve_ticket`` wait that the steward / deep_reviewer
used to chain after every ``submit_task``. With those waits removed (drop
plan), candidates that the curator rejects (or that expire while pending)
would otherwise terminalise silently in :class:`TicketStore`. The janitor
runs on a fixed interval, batches such failures by
``(project_id, task_id, escalation_id)``, and submits one info-severity
``ticket_failure`` escalation per batch so the next steward loop can re-stamp
metadata and retriage if it chooses.

Routing parallels :mod:`curator_escalator`:

* Per-project :class:`EscalationQueue` instances built lazily.
* Orchestrator liveness probed via ``flock(LOCK_SH | LOCK_NB)`` on
  ``{project_root}/data/orchestrator/orchestrator.lock``. No live orchestrator
  → log + retry on the next tick (failures stay flagged until somebody runs).
* Per-group rolling cooldown to prevent escalation storms when the curator
  flaps. When the cooldown suppresses a fresh group, the rows still get
  ``escalated_at`` stamped — accepted loss-of-signal in exchange for no spam.

Notes:
    * ``failed/server_restart`` rows from
      :meth:`TicketStore.flush_pending_on_startup` are picked up the same
      way as any other failure.
    * ``failed/bad_candidate_json`` falls back to the ``_unparseable_``
      pseudo-group; still emits one escalation.
    * ``combined/idempotency_hit`` is happy-path; the SQL filter excludes
      it via ``status='failed'`` and again via ``reason != 'idempotency_hit'``.
"""

from __future__ import annotations

import fcntl
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from fused_memory.models.scope import build_known_projects_map

if TYPE_CHECKING:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

    from fused_memory.middleware.ticket_store import TicketStore

# Match curator_escalator's defensive import so the janitor degrades to a
# log-only no-op in minimal envs without the escalation package.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped,no-redef]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


_LOCK_FILENAME = 'data/orchestrator/orchestrator.lock'
_QUEUE_DIRNAME = 'data/escalations'

# Sentinel project_id used when the candidate_json is unparseable, so we can
# still group those rows and emit a single escalation for the batch.
_UNPARSEABLE_PROJECT = '_unparseable_project_'

# Sentinel for grouping when metadata.escalation_id is absent — these rows
# came from non-orchestrator callers (interactive MCP, ad-hoc scripts) and
# earn a separate per-project group rather than collapsing across reviews.
_NO_ESCALATION = '_no_escalation_'

# Sentinel used to form the cooldown key for probe-defect escalations so the
# existing _escalation_log/_cooldown_blocks mechanism rate-limits them without
# a second structure.  Underscore-wrapped style matches _NO_ESCALATION and
# _UNPARSEABLE_PROJECT; cannot collide with real (project_id, task_id,
# escalation_id) triples derived from candidate metadata.
_PROBE_DEFECT = '_probe_defect_'



def _orchestrator_running(project_root: str) -> bool:
    """Return True if the project's orchestrator holds its exclusive lock.

    Mirrors :meth:`CuratorEscalator._orchestrator_running` so the janitor
    does not double-escalate when no orchestrator is around to handle the
    queue.
    """
    lock_path = Path(project_root) / _LOCK_FILENAME
    if not lock_path.exists():
        return False
    try:
        fd = lock_path.open('rb')
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError as exc:
            import errno
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return True
            raise
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        fd.close()


class TicketJanitor:
    """Periodic worker that escalates failed tickets to the orchestrator.

    Constructed once per fused-memory process and ticked on a fixed interval
    by the lifespan-managed task in :func:`server.main.run_server`.

    The ``{project_id → project_root}`` registry is snapshotted at ``__init__``
    time (either from the injected ``known_projects`` kwarg or via
    :func:`~fused_memory.models.scope.build_known_projects_map`) — no per-tick
    rebuild occurs.  This matches :class:`ReconciliationHarness` behaviour and
    means a process restart is required to pick up
    ``DASHBOARD_KNOWN_PROJECT_ROOTS`` changes (task 1164).

    .. note::

        Prior to task 1164, ``DASHBOARD_KNOWN_PROJECT_ROOTS`` was re-read on
        every ``tick()`` call, giving apparent SIGHUP-propagation behaviour.
        That dynamic re-read was an inconsistency — :class:`ReconciliationHarness`
        has always snapshotted at init — and was intentionally removed to
        homogenise the two consumers.  Operators who previously relied on
        SIGHUP-triggered project-map refresh must now restart the process instead.
        See ``plans/deep-squishing-lagoon.md`` for the trade-off discussion.
    """

    # Emit the startup discoverability nudge once per process, not once per
    # construction — the test suite creates many janitors and would otherwise
    # spam INFO for every test that instantiates one.
    _registry_log_emitted: bool = False

    def __init__(
        self,
        store: TicketStore,
        *,
        cooldown_secs: float = 3600.0,
        batch_limit: int = 100,
        primary_project_root: str = '',
        liveness_probe: Callable[[str], bool] | None = None,
        known_projects: dict[str, str] | None = None,
        probe_defect_threshold: int = 3,
    ) -> None:
        self._store = store
        self._cooldown_secs = cooldown_secs
        self._batch_limit = batch_limit
        self._primary_root = primary_project_root
        # Worker-liveness reaper: when set, ``tick()`` asks this probe whether
        # each project with pending tickets has a live curator worker; rows
        # for projects whose worker is dead are terminalised as
        # ``failed/worker_dead``. None disables the reaper (legacy behaviour).
        self._liveness_probe = liveness_probe
        # Consecutive probe raises before a probe-defect escalation is surfaced.
        # Default 3 keeps a single transient blip silent.  Reset per-project
        # whenever the probe returns cleanly.
        self._probe_defect_threshold = probe_defect_threshold
        # project_id → count of consecutive liveness-probe raises.  Incremented
        # on each raise; cleared via pop() when the probe returns without raising.
        self._probe_failures: dict[str, int] = defaultdict(int)
        # Registry snapshotted at init — no per-tick rebuild (task 1164).
        # When known_projects is injected (server startup), store a defensive
        # copy so external mutation doesn't affect routing after init.
        self._known_projects: dict[str, str] = (
            dict(known_projects) if known_projects is not None
            else build_known_projects_map(primary_project_root)
        )
        # group_key → monotonic timestamps of recent escalations, used for
        # rolling-window cooldown. group_key is (project_id, task_id,
        # escalation_id). Reset on process restart by design.
        self._escalation_log: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        self._queues: dict[str, EscalationQueue] = {}
        if not TicketJanitor._registry_log_emitted:
            TicketJanitor._registry_log_emitted = True
            logger.info(
                'ticket_janitor: project registry snapshotted at init (%d project(s)); '
                'restart to pick up DASHBOARD_KNOWN_PROJECT_ROOTS changes (task 1164)',
                len(self._known_projects),
            )

    def _queue_for(self, project_root: str) -> EscalationQueue:
        q = self._queues.get(project_root)
        if q is None:
            q = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
            self._queues[project_root] = q
        return q

    def _parse_candidate(self, row: dict) -> dict:
        """Extract the candidate metadata block, gracefully handling bad JSON.

        Returns a dict with ``task_id``, ``escalation_id``, ``suggestion_hash``,
        ``candidate_title``. Unparseable rows fall back to ``_unparseable_``
        sentinels so they still group together for one batched escalation.
        """
        try:
            blob = json.loads(row['candidate_json'])
            metadata = blob.get('metadata') or {}
            kwargs = blob.get('kwargs') or {}
            # ``task_id`` is the canonical key per the plan; fall back to
            # ``spawned_from`` (steward's convention) before defaulting.
            task_id = metadata.get('task_id') or metadata.get('spawned_from') or 'task-curator'
            return {
                'task_id': str(task_id),
                'escalation_id': metadata.get('escalation_id') or _NO_ESCALATION,
                'suggestion_hash': metadata.get('suggestion_hash'),
                'candidate_title': kwargs.get('title') or '<no-title>',
                'parseable': True,
            }
        except (ValueError, TypeError, KeyError):
            return {
                'task_id': '_unparseable_',
                'escalation_id': '_unparseable_',
                'suggestion_hash': None,
                'candidate_title': '<unparseable candidate_json>',
                'parseable': False,
            }

    def _cooldown_blocks(self, key: tuple[str, str, str], now: float) -> bool:
        """Return True if this group has escalated within the cooldown window."""
        cutoff = now - self._cooldown_secs
        log = [t for t in self._escalation_log[key] if t >= cutoff]
        self._escalation_log[key] = log
        return bool(log)

    def _surface_probe_defect(self, pid: str, count: int) -> None:
        """Surface an infra_issue escalation when the liveness probe has raised consecutively.

        Mirrors the routing guard ladder used by tick() step 4:
        cooldown → HAS_ESCALATION → _known_projects → _orchestrator_running → submit.
        Bail-out branches (no orchestrator / unresolved root / submit failure) do NOT
        record the cooldown so the next tick retries — identical to the ticket_failure
        flow.  Only a successful queue.submit records the cooldown timestamp.
        """
        now = time.monotonic()
        key = (pid, _PROBE_DEFECT, _PROBE_DEFECT)
        if self._cooldown_blocks(key, now):
            logger.info(
                'ticket_janitor: cooldown suppressed probe-defect escalation '
                'for %s (consecutive raises: %d)', pid, count,
            )
            return

        if not HAS_ESCALATION:
            logger.warning(
                'ticket_janitor: probe defect for %s (count=%d) but escalation '
                'package unavailable; defect not surfaced', pid, count,
            )
            return

        project_root = self._known_projects.get(pid)
        if project_root is None:
            logger.warning(
                'ticket_janitor: probe defect for %s (count=%d) but cannot '
                'resolve project_root; unable to surface escalation', pid, count,
            )
            return

        if not _orchestrator_running(project_root):
            logger.info(
                'ticket_janitor: probe defect for %s (count=%d) but orchestrator '
                'not running; will retry on next tick', pid, count,
            )
            return

        queue = self._queue_for(project_root)
        escalation = Escalation(
            id=queue.make_id('ticket-janitor'),
            task_id='task-curator',
            agent_role='fused-memory/ticket-janitor',
            severity='info',
            category='infra_issue',
            level=1,
            summary=(
                f'liveness probe raised {count} consecutive time(s) for '
                f'project {pid}; pending tickets may be silently stranded'
            ),
            detail=json.dumps({
                'project_id': pid,
                'consecutive_probe_failures': count,
                'stranded_ticket_risk': (
                    'pending tickets for this project are neither reaped nor '
                    'escalated while the probe raises'
                ),
            }),
        )
        try:
            queue.submit(escalation)
        except Exception:
            logger.warning(
                'ticket_janitor: failed to submit probe-defect escalation for '
                'project %s (count=%d); will retry next tick', pid, count,
            )
            return

        # Record in the cooldown log only after a successful submit.
        self._escalation_log[key].append(now)
        # Reset the per-project counter so the next post-cooldown escalation
        # reports failures-since-last-surface rather than failures-since-start.
        self._probe_failures.pop(pid, None)
        logger.warning(
            'ticket_janitor: surfaced probe-defect escalation %s for '
            'project %s (consecutive probe raises: %d)',
            escalation.id, pid, count,
        )

    async def tick(self) -> None:
        """One janitor pass.

        Best-effort: every step is wrapped so the asyncio task that drives the
        loop never crashes the server. Background work for an MCP server
        should be allergic to unhandled exceptions.
        """
        # 1) terminalise pending tickets whose project has no live curator
        #    worker — replaces the old wall-clock TTL janitor with an honest
        #    "is the worker actually processing?" signal.
        if self._liveness_probe is not None:
            _pending_list_ok = False
            try:
                pending_projects = await self._store.list_projects_with_pending()
                _pending_list_ok = True
            except Exception:
                logger.exception('ticket_janitor: list_projects_with_pending failed')
                pending_projects = []
            for pid in pending_projects:
                try:
                    alive = self._liveness_probe(pid)
                except Exception:
                    logger.exception(
                        'ticket_janitor: liveness_probe raised for %s; '
                        'treating as alive (fail-open)', pid,
                    )
                    alive = True
                    self._probe_failures[pid] += 1
                    if self._probe_failures[pid] >= self._probe_defect_threshold:
                        try:
                            self._surface_probe_defect(pid, self._probe_failures[pid])
                        except Exception:
                            logger.exception(
                                'ticket_janitor: _surface_probe_defect raised for %s', pid,
                            )
                else:
                    # Probe returned cleanly — reset consecutive-failure counter so
                    # intermittent blips never accumulate toward the threshold.
                    self._probe_failures.pop(pid, None)
                if not alive:
                    try:
                        reaped_ids = await self._store.mark_pending_failed_for_project(
                            pid, reason='worker_dead',
                        )
                    except Exception:
                        logger.exception(
                            'ticket_janitor: mark_pending_failed_for_project '
                            'failed for %s', pid,
                        )
                        continue
                    if reaped_ids:
                        logger.warning(
                            'ticket_janitor: reaped %d pending ticket(s) for '
                            'dead worker on project %s', len(reaped_ids), pid,
                        )
            # Prune project IDs that no longer appear in the pending list so
            # _probe_failures doesn't grow without bound when projects are
            # deleted, renamed, or have all tickets terminalised elsewhere.
            # Only prune when list_projects_with_pending() succeeded; skip
            # pruning on the exception-fallback empty list to avoid falsely
            # resetting counters for all projects on a transient DB error.
            if _pending_list_ok:
                _stale = set(self._probe_failures) - set(pending_projects)
                for _stale_pid in _stale:
                    self._probe_failures.pop(_stale_pid, None)

        # 2) collect rows that haven't been escalated yet
        try:
            rows = await self._store.fetch_unescalated_failures(
                limit=self._batch_limit,
            )
        except Exception:
            logger.exception('ticket_janitor: fetch_unescalated_failures failed')
            return

        if not rows:
            return

        # 3) group by (project_id, task_id, escalation_id)
        groups: dict[
            tuple[str, str, str],
            list[tuple[dict, dict]],
        ] = defaultdict(list)
        for row in rows:
            project_id = row.get('project_id') or _UNPARSEABLE_PROJECT
            cand = self._parse_candidate(row)
            key = (project_id, cand['task_id'], cand['escalation_id'])
            groups[key].append((row, cand))

        # 4) emit one escalation per group (or stamp + skip on cooldown)
        now = time.monotonic()
        for key, items in groups.items():
            project_id, task_id, escalation_id = key
            ticket_ids = [r['ticket_id'] for r, _ in items]

            if not HAS_ESCALATION:
                logger.warning(
                    'ticket_janitor: escalation package unavailable; '
                    'leaving %d failed tickets for project %s un-escalated',
                    len(items), project_id,
                )
                # Don't stamp escalated_at — we want the next attempt (after
                # the missing import is fixed) to retry.
                continue

            project_root = self._known_projects.get(project_id)
            if project_root is None:
                logger.warning(
                    'ticket_janitor: cannot resolve project_root for '
                    'project_id=%s; %d ticket(s) left un-escalated',
                    project_id, len(items),
                )
                continue

            if not _orchestrator_running(project_root):
                logger.info(
                    'ticket_janitor: orchestrator not running for project %s; '
                    'leaving %d ticket(s) for retry on next tick',
                    project_id, len(items),
                )
                continue

            if self._cooldown_blocks(key, now):
                # Stamp the rows so we don't re-evaluate them every tick.
                # Accepted loss-of-signal in favour of no escalation spam:
                # the original group already has a live escalation in flight.
                logger.info(
                    'ticket_janitor: cooldown suppressed escalation for %s; '
                    'stamping %d ticket(s) to prevent re-evaluation',
                    key, len(items),
                )
                try:
                    await self._store.mark_escalated(ticket_ids)
                except Exception:
                    logger.exception(
                        'ticket_janitor: mark_escalated failed for cooldown'
                        ' batch %s', key,
                    )
                continue

            # Build the escalation
            queue = self._queue_for(project_root)
            detail_payload = []
            for row, cand in items:
                detail_payload.append({
                    'ticket_id': row['ticket_id'],
                    'reason': row.get('reason'),
                    'candidate_title': cand['candidate_title'],
                    'suggestion_hash': cand['suggestion_hash'],
                    'created_at': row.get('created_at'),
                    'resolved_at': row.get('resolved_at'),
                })
            escalation = Escalation(
                id=queue.make_id('ticket-janitor'),
                task_id=task_id if task_id and task_id != '_unparseable_' else 'task-curator',
                agent_role='fused-memory/ticket-janitor',
                severity='info',
                category='ticket_failure',
                summary=(
                    f'{len(items)} ticket(s) failed for task {task_id} '
                    f'escalation {escalation_id}'
                ),
                detail=json.dumps(detail_payload, indent=2),
                level=1,
            )
            try:
                queue.submit(escalation)
            except Exception:
                logger.exception(
                    'ticket_janitor: failed to submit escalation for group %s; '
                    'leaving %d ticket(s) for retry on next tick',
                    key, len(items),
                )
                # Do NOT stamp — we want a retry next tick.
                continue

            # Record in the cooldown log AND stamp the rows.
            self._escalation_log[key].append(now)
            try:
                await self._store.mark_escalated(ticket_ids)
            except Exception:
                logger.exception(
                    'ticket_janitor: mark_escalated failed after submit for %s',
                    key,
                )

            logger.info(
                'ticket_janitor: queued escalation %s for project %s '
                '(task=%s escalation=%s, %d ticket(s))',
                escalation.id, project_id, task_id, escalation_id, len(items),
            )

    async def run_loop(self, interval_seconds: float) -> None:
        """Long-running tick loop. Cancellation is the only exit path."""
        import asyncio

        while True:
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                raise
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('ticket_janitor: tick raised; continuing')
