"""Route path-scope-guard rejections to the project's escalation queue.

When :mod:`fused_memory.middleware.path_scope_guard` rejects a candidate
task because it cites paths owned by another project, the structured
``DarkFactoryPathScopeViolation`` error dict is returned to the caller —
but historically that's the end of the line.  An MCP caller may or may
not surface the rejection; an LLM agent may not retry; and the operator
never sees that a misroute was attempted.

This escalator writes a parallel ``scope_violation`` escalation alongside
the rejection so the operator's queue surfaces the misroute even when
the calling agent never reports it.

Design mirrors :class:`fused_memory.middleware.curator_escalator.CuratorEscalator`:

* Defensive import of the optional ``escalation`` workspace package.  When
  the package is missing (minimal envs, tests without escalation infra),
  ``report_rejection`` becomes a logged no-op — the rejection error dict
  is still returned by the guard, escalation is purely additive.
* Per-project ``EscalationQueue`` cache keyed by ``project_root``.
* Escalations land in ``{project_root}/data/escalations`` — the *rejecting*
  project's queue (the place the agent was operating against).  This
  matches the existing esc-2240-series scope_violation pattern referenced
  in task 1088.

No burst control in v1: observed misroute volume is in the single digits
per day, well below the noise floor a per-project rate limiter would be
designed around.  Add when (if) volume warrants.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Mirror the defensive-import pattern from curator_escalator.py:50-55 so
# the escalator silently no-ops when the escalation package is unavailable
# (minimal CI / unit-test envs, deployments that haven't installed it yet).
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


_QUEUE_DIRNAME: str = 'data/escalations'

# Anchor task_id used by ``EscalationQueue.make_id`` so the resulting
# escalation IDs (e.g. ``esc-task-path-guard-37``) are easily greppable.
_ANCHOR_TASK_ID: str = 'task-path-guard'

_AGENT_ROLE: str = 'fused-memory/path-guard'
_CATEGORY: str = 'scope_violation'

# Budget-misconfig escalation constants — deliberately distinct from the
# scope_violation family so operators can immediately tell these apart.
_BUDGET_MISCONFIG_ANCHOR_TASK_ID: str = 'adjudicator-budget-defect'
_BUDGET_MISCONFIG_AGENT_ROLE: str = 'fused-memory/path-scope-adjudicator'
_BUDGET_MISCONFIG_CATEGORY: str = 'adjudicator_config_defect'

# Burst-suppression dedup window (mirrors curator_escalator._ZOT_DEDUP_WINDOW_SECS).
# A sustained per-call budget exhaustion files ONE escalation per window, not one
# per hit, so the operator queue doesn't flood.  Intentionally short — the condition
# resolves only via a config change, so we want one reminder per window, not silence.
_BUDGET_MISCONFIG_DEDUP_WINDOW_SECS: float = 300.0


class ScopeViolationEscalator:
    """File a ``scope_violation`` escalation for each rejected misroute."""

    def __init__(
        self,
        budget_misconfig_dedup_window_secs: float = _BUDGET_MISCONFIG_DEDUP_WINDOW_SECS,
    ) -> None:
        self._queues: dict[str, EscalationQueue] = {}
        self._budget_misconfig_dedup_window_secs = budget_misconfig_dedup_window_secs
        # project_id → monotonic timestamp of the last submitted budget-misconfig
        # escalation.  Prevents a sustained per-call exhaustion from flooding the
        # operator queue with one entry per adjudicator hit.
        self._budget_misconfig_last_submitted: dict[str, float] = {}

    def _queue_for(self, project_root: str) -> EscalationQueue | None:
        """Return (cached) :class:`EscalationQueue` for *project_root*.

        Returns ``None`` when the escalation package is unavailable so
        callers can silently skip without conditional plumbing.
        """
        if not HAS_ESCALATION:
            return None
        q = self._queues.get(project_root)
        if q is None:
            q = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
            self._queues[project_root] = q
        return q

    def report_rejection(
        self,
        *,
        project_root: str,
        project_id: str,
        candidate_title: str,
        matched_paths: tuple[str, ...],
        suggested_project: str | None,
        suggested_root: str | None = None,
        llm_reason: str | None = None,
    ) -> str | None:
        """File a ``scope_violation`` escalation for a guard rejection.

        Returns the escalation id when one was filed, ``None`` otherwise
        (escalation package missing, queue write failed, etc.).  Never
        raises — escalation is additive to the existing rejection error
        dict, so a queue write failure must not turn a guard rejection
        into a guard exception.

        Sync because :meth:`escalation.queue.EscalationQueue.submit` is a
        synchronous filesystem write (atomic ``rename``); no await needed,
        and keeping this sync lets the existing sync ``_path_guard_or_skip``
        in :class:`fused_memory.middleware.task_interceptor.TaskInterceptor`
        call it without changing the call-site signature.

        Args:
            llm_reason: When set (Stage-2 adjudicator returned reject/uncertain
                or failed), this string is the LLM's stated reason (or a
                fail-safe marker).  Appended to the escalation detail so the
                operator can see why the LLM judged this a genuine/uncertain
                misroute.  None (default) preserves the existing detail format
                for callers that don't use the Stage-2 adjudicator.
        """
        queue = self._queue_for(project_root)
        if queue is None:
            logger.debug(
                'scope_violation_escalator: escalation package unavailable; '
                'rejection of %r in project %r will not be escalated',
                candidate_title[:80], project_id,
            )
            return None

        paths_str = ', '.join(matched_paths) or '<none>'
        target = suggested_project or '<unknown — multiple or no owner>'
        suggested_action = (
            f'resubmit_to_{suggested_project}' if suggested_project else 'manual_route'
        )

        detail_lines = [
            f'candidate_title={candidate_title!r}',
            f'rejecting_project_id={project_id!r}',
            f'rejecting_project_root={project_root!r}',
            f'matched_paths={list(matched_paths)}',
            f'suggested_project={suggested_project!r}',
        ]
        if suggested_root:
            detail_lines.append(f'suggested_project_root={suggested_root!r}')
        if llm_reason is not None:
            detail_lines.append(f'llm_adjudicator_reason={llm_reason!r}')
        detail_lines.append('')
        detail_lines.append(
            'A task creation request was rejected because its text or files '
            'reference paths owned by another project.  See suggested_project '
            'above for the intended target; resubmit there or, if no clear '
            'owner is known, route the task manually.',
        )
        detail = '\n'.join(detail_lines)

        try:
            esc = Escalation(  # type: ignore[possibly-unbound]
                id=queue.make_id(_ANCHOR_TASK_ID),
                task_id=_ANCHOR_TASK_ID,
                agent_role=_AGENT_ROLE,
                severity='info',
                category=_CATEGORY,
                summary=(
                    f'Misrouted task rejected: cites {paths_str} '
                    f'(suggested target: {target})'
                ),
                detail=detail,
                suggested_action=suggested_action,
                level=1,
            )
            esc_id = queue.submit(esc)
        except Exception:
            # Queue I/O failure must not propagate — the rejection error
            # dict is still returned by the guard, the operator just
            # doesn't get the heads-up.  Mirror curator_escalator's
            # tolerance so a broken filesystem can't break task creation.
            logger.exception(
                'scope_violation_escalator: failed to submit escalation '
                'for project %s (candidate=%r)',
                project_id, candidate_title[:80],
            )
            return None

        logger.warning(
            'scope_violation_escalator: queued %s for project %s '
            '(candidate=%r, suggested=%s)',
            esc_id, project_id, candidate_title[:80], target,
        )
        return esc_id

    def report_budget_misconfig(
        self,
        *,
        project_root: str,
        project_id: str,
        cost_usd: float,
        turns: int,
        max_budget_usd: float,
        model: str,
    ) -> str | None:
        """File an ``adjudicator_config_defect`` escalation for a budget-misconfig hit.

        Called when the path-scope adjudicator's LLM call returns
        ``error_max_budget_usd`` with ``cost_usd > 0`` — a DETERMINISTIC config
        defect: the per-call budget (``config.path_scope_adjudicator.max_budget_usd``)
        is too low for the call shape.  This is NOT a transient hang.

        Returns the escalation id when one was filed, ``None`` otherwise
        (escalation package missing, within dedup window, queue write failed, etc.).
        Never raises — escalation is purely additive; the adjudicator still returns
        its fail-safe verdict regardless of whether the escalation succeeds.

        Burst-suppressed: a second call for the same *project_id* within
        ``budget_misconfig_dedup_window_secs`` logs and returns ``None`` so that a
        sustained per-call exhaustion files ONE escalation per window, not one per hit.
        """
        # Burst-suppression dedup (mirrors curator_escalator._zot_last_submitted).
        now_mono = time.monotonic()
        last = self._budget_misconfig_last_submitted.get(project_id)
        if last is not None and (now_mono - last) < self._budget_misconfig_dedup_window_secs:
            logger.info(
                'scope_violation_escalator: deduplicating budget-misconfig escalation '
                'for project %s (last submitted %.1fs ago < dedup window %.0fs)',
                project_id,
                now_mono - last,
                self._budget_misconfig_dedup_window_secs,
            )
            return None

        queue = self._queue_for(project_root)
        if queue is None:
            logger.debug(
                'scope_violation_escalator: escalation package unavailable; '
                'budget-misconfig for project %r will not be escalated',
                project_id,
            )
            return None

        detail_lines = [
            f'project_id={project_id!r}',
            f'project_root={project_root!r}',
            f'cost_usd={cost_usd}',
            f'turns={turns}',
            f'max_budget_usd={max_budget_usd}',
            f'model={model!r}',
            '',
            'The path-scope adjudicator LLM call returned error_max_budget_usd with '
            'cost_usd > 0, indicating the per-call budget is too low for the call '
            'shape.  This is a DETERMINISTIC CONFIG DEFECT — not a transient hang.',
            '',
            'FIX: raise config.path_scope_adjudicator.max_budget_usd above '
            f'{max_budget_usd} (current value) so the adjudicator call can complete.',
        ]
        detail = '\n'.join(detail_lines)

        try:
            esc = Escalation(  # type: ignore[possibly-unbound]
                id=queue.make_id(_BUDGET_MISCONFIG_ANCHOR_TASK_ID),
                task_id=_BUDGET_MISCONFIG_ANCHOR_TASK_ID,
                agent_role=_BUDGET_MISCONFIG_AGENT_ROLE,
                severity='blocking',
                category=_BUDGET_MISCONFIG_CATEGORY,
                summary=(
                    f'Adjudicator budget misconfiguration: error_max_budget_usd '
                    f'cost_usd={cost_usd} turns={turns} '
                    f'(max_budget_usd={max_budget_usd} is too low)'
                ),
                detail=detail,
                suggested_action=(
                    f'raise path_scope_adjudicator.max_budget_usd above {max_budget_usd}'
                ),
                level=1,
            )
            esc_id = queue.submit(esc)
        except Exception:
            logger.exception(
                'scope_violation_escalator: failed to submit budget-misconfig escalation '
                'for project %s (cost_usd=%s, max_budget_usd=%s)',
                project_id, cost_usd, max_budget_usd,
            )
            return None

        # Record the submission timestamp AFTER a successful submit only.
        self._budget_misconfig_last_submitted[project_id] = now_mono

        logger.warning(
            'scope_violation_escalator: queued budget-misconfig escalation %s '
            'for project %s (cost_usd=%s, turns=%d, max_budget_usd=%s, model=%r)',
            esc_id, project_id, cost_usd, turns, max_budget_usd, model,
        )
        return esc_id
