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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from fused_memory.reconciliation.event_buffer import EventBuffer


class _QueueLike(Protocol):
    """Structural interface for objects that expose queue statistics.

    Accepted by :class:`BacklogPolicy` instead of the concrete
    :class:`~fused_memory.reconciliation.event_queue.EventQueue` so that
    lightweight test stubs (which implement only ``stats()``) satisfy the
    type checker without inheriting from the full queue class.
    """

    def stats(self) -> dict: ...

logger = logging.getLogger(__name__)


OrchestratorDetector = Callable[[str], bool]
"""Callable that takes a project_root path and returns True iff orchestrator is live."""

TimeProvider = Callable[[], float]


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
    """Per-project mutable state: last-escalation timestamp + root cache."""

    last_escalation_ts: float = 0.0
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
        event_queue: _QueueLike | None,
        orchestrator_detector: OrchestratorDetector,
        *,
        hard_limit: int = 500,
        rate_limit_seconds: float = 900.0,
        time_provider: TimeProvider = time.time,
    ) -> None:
        self._event_buffer = event_buffer
        self._event_queue = event_queue
        self._detector = orchestrator_detector
        self._hard_limit = hard_limit
        self._rate_limit_seconds = rate_limit_seconds
        self._now = time_provider
        self._state: dict[str, _PolicyState] = {}
        self._lock = asyncio.Lock()

    @property
    def hard_limit(self) -> int:
        return self._hard_limit

    def register_project_root(self, project_id: str, project_root: str) -> None:
        """Cache the project_root for a project_id.

        Invoked by the task interceptor on every mutating call. Memory tools
        that only know project_id read from this cache to locate the
        escalation directory.
        """
        state = self._state.setdefault(project_id, _PolicyState())
        state.project_root = project_root

    def project_root_for(self, project_id: str) -> str | None:
        state = self._state.get(project_id)
        return state.project_root if state else None

    async def current_backlog(self, project_id: str) -> int:
        """Count buffered events + in-flight queue for ``project_id``.

        Queue stats are global (single drainer), but ``queue_depth`` still
        represents unprocessed work that will soon be attributed to one
        project or another; include it in every project's view rather than
        trying to shard it. The binding signal is still per-project buffered.
        """
        db_count = await self._event_buffer.count_buffered(project_id)
        queue_depth = 0
        retry_in_flight = 0
        if self._event_queue is not None:
            stats = self._event_queue.stats()
            queue_depth = int(stats.get('queue_depth') or 0)
            retry_in_flight = int(stats.get('retry_in_flight') or 0)
        return db_count + queue_depth + retry_in_flight

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
        if backlog <= self._hard_limit:
            return BacklogVerdict(outcome='ok', project_id=project_id)

        return await self._route_over_limit(
            project_id=project_id,
            backlog=backlog,
            error_type='ReconciliationBacklogExceeded',
            summary=(
                f'Reconciliation backlog exceeded for {project_id}: '
                f'{backlog}/{self._hard_limit}'
            ),
            detail=(
                f'Buffered events + queue depth = {backlog}, threshold = '
                f'{self._hard_limit}. Drain the backlog (run reconciliation '
                f'or trigger_reconciliation) before retrying.'
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
            error_type='ReconciliationJudgeHalted',
            summary=f'Reconciliation judge halted for {project_id}',
            detail=(
                f'Judge halted reconciliation for project {project_id}: '
                f'{reason}. Backlog at halt: {backlog}.'
            ),
            suggested_action='inspect_judge_halt',
        )

    async def on_watchdog_wedge(self, payload: dict) -> list[BacklogVerdict]:
        """Invoked by :class:`SqliteWatchdog` when the drainer is wedged.

        Writes one escalation per project with a non-zero buffered count (or
        per registered project, if the buffered count is unavailable). Returns
        the verdicts produced, primarily for test introspection.
        """
        verdicts: list[BacklogVerdict] = []
        project_ids = await self._projects_with_backlog()
        for project_id in project_ids:
            backlog = await self.current_backlog(project_id)
            verdict = await self._route_over_limit(
                project_id=project_id,
                backlog=backlog,
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

    async def _projects_with_backlog(self) -> list[str]:
        """Project ids with any registered state or buffered events.

        Falls back to the cached set so wedge alerts fire even when the
        buffer count query is unavailable.
        """
        return sorted(self._state.keys())

    async def _route_over_limit(
        self,
        *,
        project_id: str,
        backlog: int,
        error_type: str,
        summary: str,
        detail: str,
        suggested_action: str,
    ) -> BacklogVerdict:
        """Either write an escalation (if orchestrator live) or return a rejection."""
        project_root = self.project_root_for(project_id)
        if project_root is not None and self._detector(project_root):
            path = await self._maybe_write_escalation(
                project_id=project_id,
                project_root=project_root,
                backlog=backlog,
                error_type=error_type,
                summary=summary,
                detail=detail,
                suggested_action=suggested_action,
            )
            return BacklogVerdict(
                outcome='escalated',
                backlog=backlog,
                threshold=self._hard_limit,
                project_id=project_id,
                error_type=error_type,
                escalation_path=str(path) if path else None,
            )

        return BacklogVerdict(
            outcome='rejection',
            backlog=backlog,
            threshold=self._hard_limit,
            project_id=project_id,
            error_type=error_type,
        )

    async def _maybe_write_escalation(
        self,
        *,
        project_id: str,
        project_root: str,
        backlog: int,
        error_type: str,
        summary: str,
        detail: str,
        suggested_action: str,
    ) -> Path | None:
        """Write an escalation JSON unless rate-limited. Returns path on write."""
        async with self._lock:
            state = self._state.setdefault(project_id, _PolicyState())
            now = self._now()
            if (now - state.last_escalation_ts) < self._rate_limit_seconds:
                logger.info(
                    'backlog_policy: rate-limited escalation for %s (%.0fs since last)',
                    project_id, now - state.last_escalation_ts,
                )
                return None
            state.last_escalation_ts = now

        esc_dir = Path(project_root) / 'data' / 'escalations'
        esc_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.fromtimestamp(self._now(), tz=UTC).isoformat()
        safe_ts = ts.replace(':', '').replace('+', '').replace('.', '_')
        esc_id = f'esc-reconciliation-backlog-{safe_ts}'
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
            'threshold': self._hard_limit,
            'project_id': project_id,
            'error_type': error_type,
        }
        try:
            path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            logger.warning(
                'backlog_policy: wrote L1 escalation %s (backlog=%d, threshold=%d)',
                path, backlog, self._hard_limit,
            )
            return path
        except OSError as exc:
            logger.error(
                'backlog_policy: failed to write escalation %s: %s', path, exc,
            )
            return None
