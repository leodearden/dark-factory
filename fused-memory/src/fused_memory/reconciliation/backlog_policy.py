"""WP-D reconciliation-backlog escalation/rejection policy.

Bounds the backlog of unprocessed reconciliation events per project. When the
count of buffered events plus the in-flight queue exceeds a hard limit, the
policy routes to one of two outcomes:

* **Orchestrator live for project** → write an L1 escalation JSON under
  ``<project_root>/data/escalations/``. Rate-limited per project so a hot
  backlog doesn't spam the queue.
* **No orchestrator** → return a structured ``ReconciliationBacklogExceeded``
  error that callers convert to MCP responses. Reads stay unaffected.

Also exposes callbacks that the :class:`SqliteWatchdog` and the judge-halt
code path invoke when they detect a fault that implies the same drain-now
outcome — route to escalation or structured rejection so nothing goes silent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from fused_memory.reconciliation.event_buffer import EventBuffer

# The escalation package is a declared workspace dependency, but keep the
# import guarded exactly as targeted.py / ticket_janitor.py do so a minimal
# env degrades to a logged no-op rather than failing to import the policy.
try:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]
    _HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    EscalationQueue = None  # type: ignore[assignment,misc]
    _HAS_ESCALATION = False


class EventQueueLike(Protocol):
    """Structural interface for the event queue; only ``stats()`` is used."""

    def stats(self) -> dict: ...

logger = logging.getLogger(__name__)


OrchestratorDetector = Callable[[str], bool]
"""Callable that takes a project_root path and returns True iff orchestrator is live."""

TimeProvider = Callable[[], float]

# Escalation-id prefix per fault kind. A judge halt and a drainer wedge each get
# a DISTINCT prefix so they are never mis-identified as (or absorbed into)
# generic backlog-overflow noise. See task 2920 deliverable (a).
_ESC_ID_PREFIXES: dict[str, str] = {
    'backlog': 'esc-reconciliation-backlog-',
    'judge_halt': 'esc-reconciliation-halt-',
    'wedge': 'esc-reconciliation-wedge-',
}

# Keys BacklogPolicy stamps onto its escalation records that are NOT
# ``Escalation`` dataclass fields. ``EscalationQueue.resolve()`` rewrites the
# file from ``Escalation.to_json()`` (== ``asdict(Escalation)``), so closing a
# record DESTROYS them unless they are re-merged afterwards — and with
# ``project_id``/``error_type`` gone the archived halt is no longer
# attributable to a project, breaking the exact forensic query that diagnosed
# the 48h reify incident ('0 of 96 escalation files carried
# ReconciliationJudgeHalted'). See BacklogPolicy._restore_policy_keys.
_POLICY_ONLY_KEYS: tuple[str, ...] = ('project_id', 'error_type', 'backlog', 'threshold')


@dataclass(frozen=True)
class BacklogVerdict:
    """Outcome of a policy check."""

    outcome: Literal['ok', 'rejection', 'escalated']
    backlog: int = 0
    threshold: int = 0
    project_id: str = ''
    error_type: str = 'ReconciliationBacklogExceeded'
    escalation_path: str | None = None

    @property
    def is_rejection(self) -> bool:
        return self.outcome == 'rejection'

    def to_error_dict(self) -> dict:
        """Structured error dict for MCP callers. Empty on ok/escalated."""
        if self.outcome != 'rejection':
            return {}
        return {
            'error': (
                f'{self.error_type}: backlog {self.backlog} > limit '
                f'{self.threshold} for project {self.project_id}; '
                f'drain before retrying.'
            ),
            'error_type': self.error_type,
            'backlog': self.backlog,
            'threshold': self.threshold,
            'project_id': self.project_id,
        }


@dataclass
class _PolicyState:
    """Per-project mutable state: per-kind last-escalation timestamps + root cache.

    ``last_escalation_ts`` is keyed by escalation kind ('backlog' | 'judge_halt'
    | 'wedge') so each fault class is rate-limited on an INDEPENDENT clock — a
    hot backlog can never suppress a judge-halt or wedge escalation, and
    vice-versa (task 2920 deliverable (a)).
    """

    last_escalation_ts: dict[str, float] = field(default_factory=dict)
    project_root: str | None = None


class BacklogPolicy:
    """Bounded-backlog policy with escalation or rejection.

    ``escalations_fallback_dir`` is used when ``check`` is invoked for a
    project_id whose project_root hasn't been registered yet (e.g. a memory
    write for a project that's never run a task op). Callers pass the
    project_root explicitly where they have it (task interceptor paths);
    memory-tool paths rely on the policy's cached mapping.
    """

    def __init__(
        self,
        event_buffer: EventBuffer,
        event_queue: EventQueueLike | None,
        orchestrator_detector: OrchestratorDetector,
        *,
        hard_limit: int = 500,
        hard_limit_overrides: dict[str, int] | None = None,
        rate_limit_seconds: float = 900.0,
        time_provider: TimeProvider = time.time,
    ) -> None:
        self._event_buffer = event_buffer
        self._event_queue = event_queue
        self._detector = orchestrator_detector
        self._hard_limit = hard_limit
        self._hard_limit_overrides: dict[str, int] = dict(hard_limit_overrides or {})
        self._rate_limit_seconds = rate_limit_seconds
        self._now = time_provider
        self._state: dict[str, _PolicyState] = {}
        # Project ids whose root was seeded at STARTUP from the known-projects
        # map rather than observed on a real mutating call.  See
        # register_known_project_roots for why the provenance is tracked.
        self._startup_seeded: set[str] = set()
        # Throttle clock for the rejection-branch WARNING, keyed by
        # (project_id, kind).  Deliberately NOT a _PolicyState field: writing
        # one would setdefault a _state entry for an id that has no registered
        # root, which _projects_with_backlog would then fan a wedge out to.
        self._last_reject_log_ts: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

    @property
    def hard_limit(self) -> int:
        return self._hard_limit

    def hard_limit_for(self, project_id: str) -> int:
        """Return the effective hard limit for ``project_id``.

        Returns the per-project override if one is configured, otherwise
        falls back to the global default ``hard_limit``.
        """
        return self._hard_limit_overrides.get(project_id, self._hard_limit)

    def register_project_root(self, project_id: str, project_root: str) -> None:
        """Cache the project_root for a project_id.

        Invoked by the task interceptor on every mutating call. Memory tools
        that only know project_id read from this cache to locate the
        escalation directory.

        A call here is evidence of REAL activity for ``project_id``, so it
        clears any startup-seeded provenance: the project is promoted from
        "merely known" to "active" and participates unconditionally in
        ``_projects_with_backlog`` again.
        """
        state = self._state.setdefault(project_id, _PolicyState())
        state.project_root = project_root
        self._startup_seeded.discard(project_id)

    def register_known_project_roots(self, roots: Mapping[str, str]) -> None:
        """Seed the project_root cache for every KNOWN project at startup.

        Without this, ``register_project_root``'s only caller is ``check()``,
        so a halt rehydrated by ``Judge.initialize()`` fires
        ``_notify_judge_halt`` before any mutating MCP call has registered a
        root — ``_route_over_limit`` then falls to the rejection branch and
        NOTHING is written (task 2998 GAP A; the 48h reify incident in which
        0 of 96 escalation files carried ``ReconciliationJudgeHalted``).

        Each seeded id is recorded in ``_startup_seeded``.  That provenance
        matters because seeding puts an entry in ``_state`` for every known
        project, and ``_projects_with_backlog`` derives its fan-out set from
        ``_state`` — without the marker a single drainer wedge would emit one
        escalation per KNOWN project instead of per ACTIVE one.  The marker is
        consumed in ``_projects_with_backlog`` (skip startup-seeded ids whose
        backlog is 0) and cleared by ``register_project_root``.
        """
        for project_id, project_root in roots.items():
            if not project_id or not project_root:
                continue
            state = self._state.setdefault(project_id, _PolicyState())
            state.project_root = project_root
            self._startup_seeded.add(project_id)

    def project_root_for(self, project_id: str) -> str | None:
        state = self._state.get(project_id)
        return state.project_root if state else None

    def _queue_pressure(self) -> int:
        """Global (un-sharded) queue depth + retries in flight."""
        if self._event_queue is None:
            return 0
        stats = self._event_queue.stats()
        return int(stats.get('queue_depth') or 0) + int(stats.get('retry_in_flight') or 0)

    async def current_backlog(self, project_id: str) -> int:
        """Count buffered events + in-flight queue for ``project_id``.

        Queue stats are global (single drainer), but ``queue_depth`` still
        represents unprocessed work that will soon be attributed to one
        project or another; include it in every project's view rather than
        trying to shard it. The binding signal is still per-project buffered.
        """
        db_count = await self._event_buffer.count_buffered(project_id)
        return db_count + self._queue_pressure()

    async def check(
        self, project_id: str, project_root: str | None = None,
    ) -> BacklogVerdict:
        """Enforce the backlog bound for ``project_id``.

        Returns ``ok`` if under threshold, ``escalated`` if over and an
        orchestrator is live (escalation JSON written), otherwise
        ``rejection`` with a structured error payload.
        """
        if project_root is not None:
            self.register_project_root(project_id, project_root)

        backlog = await self.current_backlog(project_id)
        limit = self.hard_limit_for(project_id)
        if backlog <= limit:
            return BacklogVerdict(outcome='ok', project_id=project_id)

        return await self._route_over_limit(
            project_id=project_id,
            backlog=backlog,
            kind='backlog',
            error_type='ReconciliationBacklogExceeded',
            summary=(
                f'Reconciliation backlog exceeded for {project_id}: '
                f'{backlog}/{limit}'
            ),
            detail=(
                f'reconciliation_backlog (buffered events + queue depth + '
                f'retries) = {backlog} vs threshold {limit}. Drain the backlog '
                f'(run reconciliation or trigger_reconciliation), then confirm '
                f"the drain via get_queue_stats(project_id='{project_id}')."
                f'reconciliation_backlog — NOT the durable-write-queue counts, '
                f'a different subsystem that stays ~0.'
            ),
            suggested_action='drain_reconciliation',
        )

    async def on_judge_halt(self, project_id: str, reason: str) -> BacklogVerdict:
        """Invoked by the harness when the judge halts reconciliation.

        Routes through the same escalation-or-reject path so the halt doesn't
        rot silently: orchestrator operators see an L1 escalation; non-orchestrator
        writers see a structured error on their next mutating call.
        """
        backlog = await self.current_backlog(project_id)
        return await self._route_over_limit(
            project_id=project_id,
            backlog=backlog,
            kind='judge_halt',
            error_type='ReconciliationJudgeHalted',
            summary=f'Reconciliation HALTED for {project_id}: {reason}',
            detail=(
                f'Judge halted reconciliation for project {project_id}: '
                f'{reason}. Backlog at halt: {backlog}.'
            ),
            suggested_action='inspect_judge_halt',
        )

    async def on_judge_unhalt(self, project_id: str) -> list[str]:
        """Close the halt escalation(s) this policy opened for ``project_id``.

        Invoked by the harness's unhalt callback (manual
        ``unhalt_reconciliation`` or auto-unhalt-after-cooldown). Before task
        2998 nothing closed these records: the judge's in-memory + journal
        state cleared, but the ``esc-reconciliation-halt-*.json`` stayed
        pending forever, so the dashboard kept showing a halt that no longer
        existed.

        Write and close stay with the class that owns the record. This is NOT
        a violation of the A7b "harness never calls queue.resolve()" invariant
        — that invariant is scoped to the recon escalation queue at
        ``config.escalation_queue_dir``, closed solely by the port-8103
        watcher. The halt record lives in a different queue,
        ``<project_root>/data/escalations/``.

        Only ``judge_halt``-prefixed records that are still ``pending`` AND
        carry a matching ``project_id`` are touched — backlog/wedge records
        and other projects' halts are left alone. The project filter reads the
        RAW json because ``Escalation.from_dict`` keeps only dataclass fields
        and therefore DROPS the ``project_id`` key this policy writes, making
        ``queue.get_pending()`` unfilterable by project.

        Returns the ids actually resolved (empty when there is nothing to do),
        so the caller can report them to the operator. An id is reported ONLY
        when ``queue.resolve()`` confirms the record reached a terminal status
        — reporting a close that did not happen would be the same silent rot
        this method exists to remove, merely relabelled as success.
        """
        project_root = self.project_root_for(project_id)
        if project_root is None:
            logger.info(
                'backlog_policy: no halt escalation closed for %s — '
                'project_root not registered',
                project_id,
            )
            return []
        if not _HAS_ESCALATION or EscalationQueue is None:  # pragma: no cover
            logger.warning(
                'backlog_policy: cannot close halt escalation for %s — '
                'escalation package unavailable',
                project_id,
            )
            return []

        esc_dir = Path(project_root) / 'data' / 'escalations'
        if not esc_dir.is_dir():
            logger.info(
                'backlog_policy: no halt escalation closed for %s — %s absent',
                project_id, esc_dir,
            )
            return []

        queue = EscalationQueue(esc_dir)
        resolved: list[str] = []
        for path in sorted(esc_dir.glob(f"{_ESC_ID_PREFIXES['judge_halt']}*.json")):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                if (
                    record.get('status') != 'pending'
                    or record.get('project_id') != project_id
                ):
                    continue
                esc_id = record.get('id') or path.stem
                closed = queue.resolve(
                    esc_id,
                    resolution=(
                        f'Reconciliation halt cleared for {project_id} '
                        f'(unhalt_reconciliation / auto-unhalt-after-cooldown).'
                    ),
                    resolved_by='fused-memory-judge-unhalt',
                )
                # resolve() returns None when the id cannot be located (its
                # `id` key disagrees with the filename, or the `or path.stem`
                # fallback landed on a stem that is not the id), and returns
                # the UNCHANGED record when another resolver already moved it
                # out of `pending`. Reporting either as a successful close
                # tells the operator 'auto-resolved' about a record that stays
                # pending forever — loud-over-silent inverted.
                if closed is None or closed.status not in ('resolved', 'dismissed'):
                    logger.warning(
                        'backlog_policy: resolve() was a no-op for %s (%s) — '
                        'record not closed (id/filename mismatch, or a '
                        'non-terminal status: %s); NOT reporting it resolved',
                        esc_id, path, getattr(closed, 'status', None),
                    )
                    continue
                self._restore_policy_keys(esc_dir, esc_id, record)
            except (OSError, json.JSONDecodeError) as exc:
                # One unreadable record must never block closing the rest.
                logger.warning(
                    'backlog_policy: could not close halt escalation %s: %s',
                    path, exc,
                )
                continue
            resolved.append(esc_id)
            logger.info(
                'backlog_policy: resolved halt escalation %s on unhalt of %s',
                esc_id, project_id,
            )
        return resolved

    @staticmethod
    def _locate_persisted(esc_dir: Path, esc_id: str) -> Path | None:
        """Find a record on disk AFTER ``resolve()`` has run.

        ``resolve()`` archives into ``<esc_dir>/archive/<YYYY-MM-DD>/<id>.json``
        but deliberately leaves the file in the queue root when the archive
        move fails (``_archive_resolved`` is best-effort), so both are probed —
        root first, then the newest dated archive subdir.
        """
        root = esc_dir / f'{esc_id}.json'
        if root.exists():
            return root
        # Dated subdir names sort chronologically, so the last match is newest.
        archived = sorted(esc_dir.glob(f'archive/*/{esc_id}.json'))
        return archived[-1] if archived else None

    def _restore_policy_keys(
        self, esc_dir: Path, esc_id: str, original: Mapping[str, Any],
    ) -> None:
        """Re-merge this policy's non-schema keys onto a just-closed record.

        ``resolve()`` persists ``Escalation.to_json()``, and ``from_dict``
        keeps only dataclass fields, so closing a halt silently strips
        ``project_id``/``error_type``/``backlog``/``threshold`` — see
        ``_POLICY_ONLY_KEYS``. Re-merging keeps an auto-closed halt
        attributable to its project and its fault kind.

        Best-effort by construction: every failure is logged and swallowed.
        A record that IS closed but lost its forensic keys must never be
        misreported as un-closed, so this can never fail the caller.
        """
        keys = {k: original[k] for k in _POLICY_ONLY_KEYS if k in original}
        if not keys:
            return
        path = self._locate_persisted(esc_dir, esc_id)
        if path is None:
            logger.warning(
                'backlog_policy: closed %s but could not locate the persisted '
                'record under %s to restore %s — the archived record will not '
                'identify its project',
                esc_id, esc_dir, sorted(keys),
            )
            return
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            stripped = {k: v for k, v in keys.items() if record.get(k) != v}
            if not stripped:
                return
            record.update(stripped)
            tmp = path.with_name(f'{path.name}.tmp')
            tmp.write_text(json.dumps(record, indent=2), encoding='utf-8')
            tmp.replace(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                'backlog_policy: could not restore %s on closed record %s: %s',
                sorted(keys), path, exc,
            )

    async def on_watchdog_wedge(self, payload: dict) -> list[BacklogVerdict]:
        """Invoked by :class:`SqliteWatchdog` when the drainer is wedged.

        Writes one escalation per project with a non-zero buffered count (or
        per registered project, if the buffered count is unavailable). Returns
        the verdicts produced, primarily for test introspection.
        """
        verdicts: list[BacklogVerdict] = []
        # One stats() snapshot for the whole fan-out, and the per-project
        # buffered counts are REUSED from the filter below rather than
        # re-queried by current_backlog — a wedge is exactly when sqlite is
        # least able to serve 2N counts. ``buffered is None`` means the count
        # raised: report the global pressure alone rather than dropping the
        # alert (see _projects_with_backlog).
        queue_pressure = self._queue_pressure()
        for project_id, buffered in await self._projects_with_backlog():
            backlog = queue_pressure + (buffered or 0)
            verdict = await self._route_over_limit(
                project_id=project_id,
                backlog=backlog,
                kind='wedge',
                error_type='SqliteDrainerWedged',
                summary=f'SQLite drainer wedged for {project_id}',
                detail=(
                    'The reconciliation event-queue drainer has not committed '
                    f'within the stall threshold. Diagnostic payload: '
                    f'{json.dumps(payload, default=str, sort_keys=True)[:2000]}'
                ),
                suggested_action='inspect_sqlite_drainer',
            )
            verdicts.append(verdict)
        return verdicts

    async def _projects_with_backlog(self) -> list[tuple[str, int | None]]:
        """Project ids a fleet-wide fault (e.g. a drainer wedge) should escalate.

        Returns ``(project_id, buffered_count)`` pairs. The count is ``None``
        when the per-project query FAILED (see the failure policy below); the
        caller reuses it so the wedge path counts each project once, not twice.

        Every EXPLICITLY registered id (one observed on a real mutating call,
        via ``register_project_root``) is returned unconditionally, including
        at zero backlog — an active project deserves the alert even when its
        buffered count happens to be 0 right now.

        Ids known only from startup seeding (``register_known_project_roots``,
        tracked in ``_startup_seeded``) are returned ONLY when the project's
        OWN buffered count is non-zero.  Without that filter, seeding every
        known project at startup (task 2998 GAP A) would turn a single drainer
        wedge into one escalation per KNOWN project rather than per ACTIVE one.

        The idle test deliberately reads ``count_buffered`` directly rather
        than ``current_backlog``.  ``current_backlog`` folds in the GLOBAL
        ``queue_depth`` + ``retry_in_flight`` (intentionally — a global
        backlog is real pressure on every project), but the only caller of
        this method is ``on_watchdog_wedge``, which the watchdog fires ONLY
        when ``queue_depth + retry_in_flight > 0``.  Using it here would make
        ``current_backlog(pid) >= outstanding > 0`` hold for EVERY project by
        construction, so the filter would be inert and the fan-out regression
        would be live.  The signal must be per-project only.

        Failure policy — LOUD and INCLUSIVE.  ``count_buffered`` raises when
        the db is not initialised and on any sqlite error ('database is
        locked', disk I/O): precisely the conditions under which the drainer
        wedge that calls this fires.  An unguarded raise would propagate out of
        ``on_watchdog_wedge`` (whose caller's ``except Exception`` only logs
        'wedge_callback raised') and write ZERO wedge escalations for ANY
        project, including explicitly-registered active ones.  So a failed
        count warns and KEEPS the project in the fan-out: an unavailable count
        must never silently suppress a wedge alert.
        """
        candidates: list[tuple[str, int | None]] = []
        for project_id in sorted(self._state.keys()):
            buffered: int | None
            try:
                buffered = await self._event_buffer.count_buffered(project_id)
            except Exception as exc:
                logger.warning(
                    'backlog_policy: count_buffered failed for %s (%s) — keeping '
                    'it in the wedge fan-out; an unavailable count must not '
                    'suppress the alert',
                    project_id, exc,
                )
                buffered = None
            if buffered == 0 and project_id in self._startup_seeded:
                continue
            candidates.append((project_id, buffered))
        return candidates

    async def _route_over_limit(
        self,
        *,
        project_id: str,
        backlog: int,
        kind: str,
        error_type: str,
        summary: str,
        detail: str,
        suggested_action: str,
    ) -> BacklogVerdict:
        """Either write an escalation (if orchestrator live) or return a rejection.

        ``kind`` ('backlog' | 'judge_halt' | 'wedge') selects the escalation-id
        prefix and the independent per-kind rate-limit bucket (task 2920 (a)).
        """
        limit = self.hard_limit_for(project_id)
        project_root = self.project_root_for(project_id)
        if project_root is not None and self._detector(project_root):
            path = await self._maybe_write_escalation(
                project_id=project_id,
                project_root=project_root,
                backlog=backlog,
                threshold=limit,
                kind=kind,
                error_type=error_type,
                summary=summary,
                detail=detail,
                suggested_action=suggested_action,
            )
            return BacklogVerdict(
                outcome='escalated',
                backlog=backlog,
                threshold=limit,
                project_id=project_id,
                error_type=error_type,
                escalation_path=str(path) if path else None,
            )

        # Loud-over-silent (task 2998): the reject branch writes no escalation
        # file, so without this line the drop is invisible — the defining
        # symptom of the 48h reify incident was NO backlog_policy log of any
        # kind.
        #
        # THROTTLED to the escalation rate-limit window, on the same
        # per-(project, kind) granularity. The GAP-B fix deliberately stopped
        # burning the dedupe token on a failed write, so a halted project with
        # no live orchestrator now re-enters this branch every ~5s harness
        # tick — unthrottled that is ~17k WARNING lines/day/project, which
        # drowns the very signal this line exists to raise. The FIRST
        # rejection always warns; the repeats inside the window drop to DEBUG,
        # and a rejection that is still happening a window later warns again.
        cause = (
            'project_root not registered'
            if project_root is None
            else f'no live orchestrator for {project_root}'
        )
        now = self._now()
        last_logged = self._last_reject_log_ts.get((project_id, kind))
        if last_logged is None or (now - last_logged) >= self._rate_limit_seconds:
            self._last_reject_log_ts[(project_id, kind)] = now
            emit = logger.warning
        else:
            emit = logger.debug
        emit(
            'backlog_policy: no escalation written for %s (kind=%s, '
            'error_type=%s, backlog=%d) — %s',
            project_id, kind, error_type, backlog, cause,
        )
        return BacklogVerdict(
            outcome='rejection',
            backlog=backlog,
            threshold=limit,
            project_id=project_id,
            error_type=error_type,
        )

    async def _maybe_write_escalation(
        self,
        *,
        project_id: str,
        project_root: str,
        backlog: int,
        threshold: int,
        kind: str,
        error_type: str,
        summary: str,
        detail: str,
        suggested_action: str,
    ) -> Path | None:
        """Write an escalation JSON unless rate-limited. Returns path on write.

        Rate-limiting is per-(project, ``kind``): a backlog escalation never
        suppresses a judge-halt or wedge escalation within the window, and
        vice-versa (task 2920 (a)). The id prefix is derived from ``kind`` so a
        halt/wedge is never mis-filed as 'backlog'.
        """
        async with self._lock:
            state = self._state.setdefault(project_id, _PolicyState())
            now = self._now()
            last = state.last_escalation_ts.get(kind, 0.0)
            if (now - last) < self._rate_limit_seconds:
                logger.info(
                    'backlog_policy: rate-limited %s escalation for %s (%.0fs since last)',
                    kind, project_id, now - last,
                )
                return None
            state.last_escalation_ts[kind] = now

        esc_dir = Path(project_root) / 'data' / 'escalations'
        esc_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.fromtimestamp(self._now(), tz=UTC).isoformat()
        safe_ts = ts.replace(':', '').replace('+', '').replace('.', '_')
        prefix = _ESC_ID_PREFIXES.get(kind, _ESC_ID_PREFIXES['backlog'])
        esc_id = f'{prefix}{safe_ts}'
        path = esc_dir / f'{esc_id}.json'

        record = {
            'id': esc_id,
            'task_id': None,
            'agent_role': 'fused-memory',
            'severity': 'blocking',
            'category': 'infra_issue',
            'summary': summary,
            'detail': detail,
            'suggested_action': suggested_action,
            'timestamp': ts,
            'status': 'pending',
            'level': 1,
            'workflow_state': 'infra',
            'backlog': backlog,
            'threshold': threshold,
            'project_id': project_id,
            'error_type': error_type,
        }
        try:
            path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            logger.warning(
                'backlog_policy: wrote L1 escalation %s (backlog=%d, threshold=%d)',
                path, backlog, threshold,
            )
            return path
        except OSError as exc:
            logger.error(
                'backlog_policy: failed to write escalation %s: %s', path, exc,
            )
            return None
