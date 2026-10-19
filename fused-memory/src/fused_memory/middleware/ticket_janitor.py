"""Periodic background sweep over the tickets no request is waiting on.

Replaces the per-call ``resolve_ticket`` wait that the steward / deep_reviewer
used to chain after every ``submit_task``. With those waits removed (drop
plan), tickets would otherwise sit or terminalise silently in
:class:`TicketStore`. Each :meth:`TicketJanitor.tick` runs, in order:

0. **Drain.** Every due ``awaiting_curation`` ticket, oldest first, is handed
   back to curation through the injected ``requeue_ticket``. It runs first
   because a requeue registers worker intent, which the reaper must see. It is
   the only retrier of parked tickets.
1. **Reap.** The ``pending`` tickets of a project whose curator worker is gone
   are recovered by the store's one rule (re-parked if already deferred, else
   ``failed/worker_dead``) and their waiters woken.
2. **Dedup outage.** A blocking ``infra_issue`` for the curator dedup-outage
   signature (see ``plans/curator-dedup-outage-2026-08-15-rca.md``).
3. **Failed tickets.** Failures are batched by
   ``(project_id, task_id, escalation_id)`` into one info-severity
   ``ticket_failure`` escalation per batch, so the next steward loop can
   re-stamp metadata and retriage if it chooses.
4. **Curation backlog.** One folded ``infra_issue`` per
   ``(project_id, defer_reason)`` once a parked ticket there has survived a
   retry, under :func:`curation_backlog_anchor`, with the reason's
   :func:`~fused_memory.middleware.curation_deferral.backlog_severity`.

Routing parallels :mod:`curator_escalator`:

* Per-project :class:`EscalationQueue` instances built lazily.
* Orchestrator liveness via
  ``fused_memory/services/orchestrator_detector.py::is_orchestrator_lock_held``.
  No live orchestrator → log + retry on the next tick (failures stay flagged
  until somebody runs).
* Per-group rolling cooldown to prevent escalation storms when the curator
  flaps. When the cooldown suppresses a fresh group, the rows still get
  ``escalated_at`` stamped — accepted loss-of-signal in exchange for no spam.

The curation backlog is the exception: it files through
:func:`~fused_memory.middleware._folded_escalation.file_folded_escalation`,
whose fold into the pending record under its anchor survives restarts, and it
does not wait for a live orchestrator. After the record is resolved, the
cooldown spaces a refile.

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

import asyncio
import json
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from shared.project_layout import escalation_queue_dir

from fused_memory.middleware._folded_escalation import file_folded_escalation
from fused_memory.middleware.curation_deferral import DeferReason, backlog_severity
from fused_memory.middleware.ticket_store import AWAITING_CURATION
from fused_memory.models.scope import build_known_projects_map
from fused_memory.services.orchestrator_detector import is_orchestrator_lock_held

if TYPE_CHECKING:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

    from fused_memory.config.schema import DedupOutageDetectorConfig
    from fused_memory.middleware.ticket_store import CurationBacklog, DedupHealth, TicketStore

# Match curator_escalator's defensive import so the janitor degrades to a
# log-only no-op in minimal envs without the escalation package.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped,no-redef]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


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

# Cooldown-key sentinel for dedup-outage escalations, in the same style.
_DEDUP_OUTAGE = '_dedup_outage_'

_CURATION_BACKLOG = '_curation_backlog_'
CURATION_BACKLOG_ANCHOR: Final = 'curation-backlog'
_CURATION_BACKLOG_DETAIL_TICKETS: Final = 20


def curation_backlog_anchor(reason: DeferReason) -> str:
    return f'{CURATION_BACKLOG_ANCHOR}-{reason}'


def is_dedup_outage(health: DedupHealth, cfg: DedupOutageDetectorConfig) -> bool:
    """True when a window holds enough curated tickets, none combined, resolved implausibly fast."""
    return (
        health.resolved >= cfg.min_samples
        and health.combined == 0
        and health.median_resolve_seconds is not None
        and health.median_resolve_seconds <= cfg.max_median_resolve_seconds
    )



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
        signal_ticket_resolved: Callable[[str], None] | None = None,
        known_projects: dict[str, str] | None = None,
        probe_defect_threshold: int = 3,
        dedup_outage: DedupOutageDetectorConfig | None = None,
        requeue_ticket: Callable[[str, str], Awaitable[bool]] | None = None,
        curation_backlog_min_attempts: int = 2,
    ) -> None:
        self._store = store
        self._requeue_ticket = requeue_ticket
        self._curation_backlog_min_attempts = curation_backlog_min_attempts
        # None (the default) leaves the dedup-outage detector off.
        self._dedup_outage_cfg = dedup_outage
        self._cooldown_secs = cooldown_secs
        self._batch_limit = batch_limit
        self._primary_root = primary_project_root
        # Worker-liveness reaper: when set, ``tick()`` asks this probe whether
        # each project with pending tickets has a live curator worker; rows
        # for projects whose worker is dead are terminalised as
        # ``failed/worker_dead``. None disables the reaper (legacy behaviour).
        self._liveness_probe = liveness_probe
        # Optional callback invoked once per reaped ticket id after the
        # terminalising UPDATE commits.  Mirrors the liveness_probe injection
        # pattern.  Wired in server/main.py to
        # task_interceptor._signal_ticket_event so blocked resolve_ticket
        # waiters are woken promptly instead of waiting for their own timeout.
        # None = legacy no-signal behaviour (default).
        self._signal_ticket_resolved = signal_ticket_resolved
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
            q = EscalationQueue(escalation_queue_dir(project_root))
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

    def _cooldown_blocks(
        self,
        key: tuple[str, str, str],
        now: float,
        cooldown_secs: float | None = None,
    ) -> bool:
        """Return True if this group has escalated within the cooldown window.

        *cooldown_secs* overrides the janitor-wide cooldown for this key.
        """
        cutoff = now - (self._cooldown_secs if cooldown_secs is None else cooldown_secs)
        log = [t for t in self._escalation_log[key] if t >= cutoff]
        self._escalation_log[key] = log
        return bool(log)

    async def _submit_infra_escalation(
        self, pid: str, *, severity: str, summary: str, detail: str,
    ) -> bool:
        """Route one janitor ``infra_issue`` escalation to *pid*'s queue.

        The routing ladder used by tick() step 4:
        HAS_ESCALATION → _known_projects → is_orchestrator_lock_held → submit.
        Returns True only on a successful ``queue.submit``. A caller records its
        rate limit only on True, so every bail-out (no escalation package,
        unresolved root, no orchestrator, submit failure) retries next tick.
        """
        if not HAS_ESCALATION:
            logger.warning(
                'ticket_janitor: escalation package unavailable; not surfaced '
                'for %s: %s', pid, summary,
            )
            return False

        project_root = self._known_projects.get(pid)
        if project_root is None:
            logger.warning(
                'ticket_janitor: cannot resolve project_root for %s; not '
                'surfaced: %s', pid, summary,
            )
            return False

        if not await asyncio.to_thread(is_orchestrator_lock_held, project_root):
            logger.info(
                'ticket_janitor: orchestrator not running for %s; will retry '
                'on next tick: %s', pid, summary,
            )
            return False

        queue = self._queue_for(project_root)
        escalation = Escalation(
            id=queue.make_id('ticket-janitor'),
            task_id='task-curator',
            agent_role='fused-memory/ticket-janitor',
            severity=severity,
            category='infra_issue',
            level=1,
            summary=summary,
            detail=detail,
        )
        try:
            queue.submit(escalation)
        except Exception:
            logger.warning(
                'ticket_janitor: failed to submit infra escalation for %s; '
                'will retry next tick: %s', pid, summary,
            )
            return False
        logger.warning(
            'ticket_janitor: surfaced infra escalation %s for %s: %s',
            escalation.id, pid, summary,
        )
        return True

    async def _surface_probe_defect(self, pid: str, count: int) -> None:
        """Surface an infra_issue escalation when the liveness probe has raised consecutively.

        Only a successful submit records the cooldown timestamp and resets the
        per-project counter; see :meth:`_submit_infra_escalation`.
        """
        now = time.monotonic()
        key = (pid, _PROBE_DEFECT, _PROBE_DEFECT)
        if self._cooldown_blocks(key, now):
            logger.info(
                'ticket_janitor: cooldown suppressed probe-defect escalation '
                'for %s (consecutive raises: %d)', pid, count,
            )
            return

        submitted = await self._submit_infra_escalation(
            pid,
            severity='info',
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
        if not submitted:
            return

        self._escalation_log[key].append(now)
        # Reset the per-project counter so the next post-cooldown escalation
        # reports failures-since-last-surface rather than failures-since-start.
        self._probe_failures.pop(pid, None)

    async def _check_dedup_outage(self, cfg: DedupOutageDetectorConfig) -> None:
        """Escalate each known project whose recent tickets show the dedup-outage signature.

        Rate-limited per project on ``cfg.window_seconds``, so a sustained
        outage yields at most one escalation per window of fresh evidence.
        """
        for pid in self._known_projects:
            try:
                await self._check_project_dedup_outage(pid, cfg)
            except Exception:
                logger.exception('ticket_janitor: dedup-outage check failed for %s', pid)

    async def _check_project_dedup_outage(
        self, pid: str, cfg: DedupOutageDetectorConfig,
    ) -> None:
        since = datetime.now(UTC) - timedelta(seconds=cfg.window_seconds)
        health = await self._store.dedup_health(pid, since=since)
        if not is_dedup_outage(health, cfg):
            return
        now = time.monotonic()
        key = (pid, _DEDUP_OUTAGE, _DEDUP_OUTAGE)
        if self._cooldown_blocks(key, now, cooldown_secs=cfg.window_seconds):
            return
        median = health.median_resolve_seconds
        submitted = await self._submit_infra_escalation(
            pid,
            severity='blocking',
            summary=(
                f'curator dedup outage signature for project {pid}: '
                f'{health.resolved} tickets resolved, {health.combined} combined, '
                f'median resolve {median:.1f}s'
            ),
            detail=json.dumps({
                'project_id': pid,
                'window_seconds': cfg.window_seconds,
                'resolved': health.resolved,
                'combined': health.combined,
                'median_resolve_seconds': median,
                'top_create_reasons': [
                    [reason, count] for reason, count in health.top_create_reasons
                ],
            }),
        )
        if submitted:
            self._escalation_log[key].append(now)

    async def _drain_awaiting_curation(self, requeue_ticket: Callable[[str, str], Awaitable[bool]]) -> None:
        try:
            due = await self._store.fetch_due_awaiting_curation(
                now=datetime.now(UTC), limit=self._batch_limit,
            )
        except Exception:
            logger.exception('ticket_janitor: fetch_due_awaiting_curation failed')
            return
        for row in due:
            try:
                await requeue_ticket(row['project_id'], row['ticket_id'])
            except Exception:
                logger.exception(
                    'ticket_janitor: requeue of parked ticket %s failed; it stays '
                    'parked for the next tick', row['ticket_id'],
                )

    async def _escalate_curation_backlog(self) -> None:
        """File one folded escalation per ``(project, defer_reason)`` backlog that survived a retry."""
        try:
            backlogs = await self._store.summarise_awaiting_curation()
        except Exception:
            logger.exception('ticket_janitor: summarise_awaiting_curation failed')
            return
        parked_per_project: defaultdict[str, int] = defaultdict(int)
        for backlog in backlogs:
            parked_per_project[backlog.project_id] += backlog.count
        for backlog in backlogs:
            if backlog.max_attempts < self._curation_backlog_min_attempts:
                continue
            try:
                await self._escalate_one_curation_backlog(
                    backlog, parked_in_project=parked_per_project[backlog.project_id],
                )
            except Exception:
                logger.exception(
                    'ticket_janitor: curation-backlog escalation failed for %s/%s',
                    backlog.project_id, backlog.defer_reason,
                )

    async def _escalate_one_curation_backlog(
        self, backlog: CurationBacklog, *, parked_in_project: int,
    ) -> None:
        now = time.monotonic()
        key = (backlog.project_id, _CURATION_BACKLOG, backlog.defer_reason.value)
        if self._cooldown_blocks(key, now):
            return
        severity = backlog_severity(backlog.defer_reason)
        parked = await self._store.list_tickets(
            backlog.project_id,
            status=AWAITING_CURATION,
            since=datetime.fromisoformat(backlog.oldest_created_at),
            limit=parked_in_project,
        )
        tickets = [
            {
                'ticket_id': row['ticket_id'],
                'candidate_title': self._parse_candidate(row)['candidate_title'],
                'created_at': row['created_at'],
                'attempts': row['attempts'],
                'next_attempt_at': row['next_attempt_at'],
            }
            for row in parked
            if row['defer_reason'] == backlog.defer_reason.value
        ][:_CURATION_BACKLOG_DETAIL_TICKETS]
        esc_id = await asyncio.to_thread(
            file_folded_escalation,
            self._known_projects.get(backlog.project_id),
            anchor_task_id=curation_backlog_anchor(backlog.defer_reason),
            agent_role='fused-memory/ticket-janitor',
            category='infra_issue',
            severity=severity,
            summary=(
                f'{backlog.count} task submission(s) parked awaiting curation for project '
                f'{backlog.project_id}: {backlog.defer_reason} (oldest {backlog.oldest_created_at}, '
                f'up to {backlog.max_attempts} deferrals)'
            ),
            detail=json.dumps({
                'project_id': backlog.project_id,
                'defer_reason': backlog.defer_reason.value,
                'severity': severity,
                'count': backlog.count,
                'oldest_created_at': backlog.oldest_created_at,
                'max_attempts': backlog.max_attempts,
                'tickets': tickets,
            }),
            suggested_action=(
                "The task curator's account pool cannot judge these submissions, so they "
                'are parked and retried automatically on backoff until it can; nothing is '
                'lost and nothing should be re-submitted. A blocking record means the pool '
                'will not recover by itself: restore the curator accounts named by the '
                'defer reason.'
            ),
            logger=logger,
            log_label='ticket_janitor',
            context=f'curation backlog {backlog.project_id}/{backlog.defer_reason}',
        )
        if esc_id is not None:
            self._escalation_log[key].append(now)

    async def tick(self) -> None:
        """One janitor pass.

        Best-effort: every step is wrapped so the asyncio task that drives the
        loop never crashes the server. Background work for an MCP server
        should be allergic to unhandled exceptions.
        """
        if self._requeue_ticket is not None:
            await self._drain_awaiting_curation(self._requeue_ticket)
        await self._reap_dead_workers()
        cfg = self._dedup_outage_cfg
        if cfg is not None and cfg.enabled:
            await self._check_dedup_outage(cfg)
        await self._escalate_failed_tickets()
        await self._escalate_curation_backlog()

    async def _reap_dead_workers(self) -> None:
        """Recover the pending tickets of every project whose curator worker is gone."""
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
                            await self._surface_probe_defect(pid, self._probe_failures[pid])
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
                        recovery = await self._store.recover_pending_for_dead_worker(
                            pid, reason='worker_dead',
                        )
                    except Exception:
                        logger.exception(
                            'ticket_janitor: recover_pending_for_dead_worker '
                            'failed for %s', pid,
                        )
                        continue
                    reaped_ids = recovery.failed_ids + recovery.reparked_ids
                    if reaped_ids:
                        logger.warning(
                            'ticket_janitor: reaped %d pending ticket(s) for '
                            'dead worker on project %s', len(reaped_ids), pid,
                        )
                        # Signal blocked resolve_ticket waiters so they wake
                        # promptly with failed/worker_dead (or
                        # awaiting_curation, if re-parked) instead of waiting
                        # for their own timeout.  Strictly after the
                        # recovering UPDATE so the woken caller re-reads a
                        # row that is no longer pending.  Best-effort:
                        # a buggy callback must not crash the janitor loop.
                        if self._signal_ticket_resolved is not None:
                            for tid in reaped_ids:
                                try:
                                    self._signal_ticket_resolved(tid)
                                except Exception:
                                    logger.exception(
                                        'ticket_janitor: signal_ticket_resolved '
                                        'raised for %s', tid,
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

    async def _escalate_failed_tickets(self) -> None:
        """Surface each group of un-escalated failed tickets as one ``ticket_failure``."""
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

            if not await asyncio.to_thread(is_orchestrator_lock_held, project_root):
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
