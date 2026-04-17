"""Route curator LLM failures to the orchestrator's escalation queue.

The :class:`TaskCurator` used to silently degrade to ``action='create'`` on
any LLM error, which meant a broken curator shipped for five days without
anyone noticing (see plans/floating-snuggling-pebble.md, R2). The curator
now raises :class:`CuratorFailureError` on LLM failure; this module decides
what happens next.

Routing policy (keyed off orchestrator liveness):

* **Orchestrator is running** for the target project — submit a level-1
  escalation to the project's queue and return; the escalation watcher
  runs ``/unblock`` against it. We degrade to ``action='create'`` so the
  current ``add_task`` call still succeeds.

* **No orchestrator** (typical interactive MCP usage) — re-raise the
  failure so the MCP caller sees a loud error instead of a silent
  curator outage.

Liveness is probed via ``flock(LOCK_SH | LOCK_NB)`` on
``{project_root}/data/orchestrator/orchestrator.lock`` (the orchestrator
holds ``LOCK_EX`` on startup). Treat a missing file as "no orchestrator".

Per-project 1 h cooldown keeps a stuck curator from flooding the queue
with a near-identical escalation on every ``add_task`` call.
"""

from __future__ import annotations

import fcntl
import logging
import time
from pathlib import Path

from fused_memory.middleware.task_curator import CuratorFailureError

# ``escalation`` is a sibling workspace package. The main reconciliation
# harness also imports it defensively (harness.py:38-46) because historical
# deployments could lack the package. Mirror that pattern here so the
# curator still functions (without escalation routing) in minimal envs.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover - exercised only in minimal envs
    HAS_ESCALATION = False
    Escalation = None  # type: ignore[assignment,misc]
    EscalationQueue = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


_DEFAULT_COOLDOWN_SECS = 3600.0
_LOCK_FILENAME = 'data/orchestrator/orchestrator.lock'
_QUEUE_DIRNAME = 'data/escalations'


class CuratorEscalator:
    """Route :class:`CuratorFailureError` to the orchestrator or back to the caller."""

    def __init__(self, cooldown_secs: float = _DEFAULT_COOLDOWN_SECS) -> None:
        self._cooldown_secs = cooldown_secs
        self._last_escalation: dict[str, float] = {}
        self._queues: dict[str, EscalationQueue] = {}

    def _orchestrator_running(self, project_root: str) -> bool:
        """Return True if the project's orchestrator holds its exclusive lock.

        We probe with a *shared* non-blocking lock so we don't perturb the
        orchestrator's lock state — a successful acquisition means nobody
        holds LOCK_EX, so no orchestrator is running; a block/EAGAIN means
        it is.
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
                # EAGAIN / EWOULDBLOCK on some platforms map to OSError
                import errno
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return True
                raise
            # Lock acquired → orchestrator is not running. Release promptly.
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            return False
        finally:
            fd.close()

    def _queue_for(self, project_root: str) -> EscalationQueue:
        q = self._queues.get(project_root)
        if q is None:
            q = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
            self._queues[project_root] = q
        return q

    async def report_failure(
        self,
        *,
        project_root: str,
        project_id: str,
        justification: str,
        candidate_title: str,
    ) -> None:
        """Route a curator failure. Raises :class:`CuratorFailureError` when no
        orchestrator is running so the MCP caller sees a loud error.

        When an orchestrator *is* running for this project, submit a level-1
        escalation (once per project per cooldown window) and return — the
        caller will fall back to ``action='create'`` so the current
        ``add_task`` still succeeds.
        """
        if not HAS_ESCALATION:
            # No escalation package available — fall back to a loud raise so
            # operators don't miss a silent curator outage.
            raise CuratorFailureError(
                f'TaskCurator LLM failed and escalation package is unavailable. '
                f'No dedupe was applied for project {project_id!r}. '
                f'justification={justification!r} candidate_title={candidate_title!r}',
            )

        if not self._orchestrator_running(project_root):
            raise CuratorFailureError(
                f'TaskCurator LLM failed and no orchestrator is running for '
                f'project {project_id!r}. No dedupe was applied. '
                f'justification={justification!r} candidate_title={candidate_title!r}',
            )

        now = time.monotonic()
        last = self._last_escalation.get(project_id, 0.0)
        if now - last < self._cooldown_secs:
            logger.warning(
                'curator_escalator: suppressing escalation for project %s '
                '(cooldown %.0fs remaining); failure=%s',
                project_id,
                self._cooldown_secs - (now - last),
                justification[:200],
            )
            return

        queue = self._queue_for(project_root)
        escalation = Escalation(
            id=queue.make_id('curator'),
            task_id='task-curator',
            agent_role='fused-memory/task-curator',
            severity='blocking',
            category='curator_failure',
            summary='TaskCurator LLM failing; dedupe disabled',
            detail=(
                f'candidate_title={candidate_title!r}\n'
                f'project_id={project_id!r}\n'
                f'justification={justification}'
            ),
            level=1,
        )
        try:
            queue.submit(escalation)
        except Exception:
            logger.exception(
                'curator_escalator: failed to submit escalation for project %s',
                project_id,
            )
            # Do not re-raise — falling through to action='create' is safer
            # than failing the add_task just because queue I/O broke.
            return

        self._last_escalation[project_id] = now
        logger.warning(
            'curator_escalator: queued L1 escalation %s for project %s',
            escalation.id, project_id,
        )
