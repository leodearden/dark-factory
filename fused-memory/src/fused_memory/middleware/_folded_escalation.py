"""The pending-anchor-fold escalation filer skeleton, in one place (INV-5).

:func:`file_folded_escalation` owns the defensive optional-``escalation``
import, the guarded ``EscalationQueue`` construction, the
``get_by_task(anchor, status='pending', agent_role=role)`` dedupe fold, and
the never-raise ``Escalation(...)`` + ``queue.submit(...)``.  The fold is
role-scoped, the same scope :func:`pending_folded_escalations` reads and
:func:`resolve_folded_escalations` closes, so another role's pending record
under the anchor cannot silence the filer.  Each caller keeps its own
``_ANCHOR_TASK_ID`` / ``_AGENT_ROLE`` / ``_CATEGORY`` constants and builds its
own summary and detail.  :func:`pending_folded_escalations` and
:func:`resolve_folded_escalations` read and close what a caller filed, for an
alarm that clears itself, and :func:`refresh_folded_escalation` rewrites and
counts a sustained condition into its one open record, under the same guard.

EVERY FILER'S ANCHOR MUST BE UNIQUE.  Role scoping stops another ROLE's
record under the anchor from silencing a filer, but a second filer under the
same anchor and role would fold into the first one's open record and never
file again, and that silence is indistinguishable from health.  Queue
consumers outside this module also match on ``task_id`` without the role, e.g.
``escalation/src/escalation/queue.py::EscalationQueue.dismiss_if_covered``.
Measured before the fold was role-scoped: the L1 escalation watcher squatted
the ``markup-tripwire`` anchor, and the tripwire filed nothing from 2026-08-16
to 2026-08-19 while 41 rejections occurred.  So ``anchor_task_id`` is required
and keyword-only with no default, which makes a shared anchor impossible to
introduce by omission.  Each caller keeps its anchor in its own module, where
``tests/test_folded_escalation.py::TestNoTwoFilersShareAnAnchor`` reads it and
fails on a collision.

NOT THE HOME FOR the ``submit_or_dedupe`` content-fingerprint family:
:mod:`fused_memory.middleware.scope_violation_escalator`,
:mod:`fused_memory.middleware.mem0_update_storm_escalator` and
:mod:`fused_memory.middleware.entity_mint_storm_escalator`.  They dedupe on
``compute_content_fingerprint`` + ``DedupeConfig`` over a cached per-project
queue, and ``ScopeViolationEscalator._submit`` is already that family's one
home.  Routing them through here would replace content-fingerprint dedupe with
pending-anchor dedupe, which is a behaviour change, not a consolidation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from shared.project_layout import escalation_queue_dir

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# The `escalation` workspace package is optional (minimal CI envs, deployments
# that have not installed it); without it every filer is a WARNING-logged no-op.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False


def _suffix(context: str) -> str:
    """Render the caller's *context* as a trailing log clause, or nothing."""
    return f' ({context})' if context else ''


def _open_queue(
    project_root: str | None,
    *,
    logger: logging.Logger,
    log_label: str,
    outcome: str,
    context: str = '',
) -> EscalationQueue | None:
    """Open *project_root*'s queue, or log why not and return None.

    *outcome* completes each no-op line, e.g. ``'nothing escalated'``.
    """
    if project_root is None:
        logger.debug(
            '%s: no project_root, so there is no project queue; %s%s',
            log_label, outcome, _suffix(context),
        )
        return None

    if not HAS_ESCALATION:
        logger.warning(
            '%s: escalation package unavailable; %s%s',
            log_label, outcome, _suffix(context),
        )
        return None

    try:
        return EscalationQueue(escalation_queue_dir(project_root))
    except Exception:
        # Constructing the queue creates its directory; a read-only or missing
        # project_root must not turn an alarm into a crash on a write path.
        logger.exception(
            '%s: could not open the escalation queue at project_root=%r; %s%s',
            log_label, project_root, outcome, _suffix(context),
        )
        return None


def file_folded_escalation(
    project_root: str | None,
    *,
    anchor_task_id: str,
    agent_role: str,
    category: str,
    severity: str,
    summary: str,
    detail: str,
    suggested_action: str,
    logger: logging.Logger,
    log_label: str,
    level: int = 1,
    dedupe: bool = True,
    context: str = '',
    on_fold: Callable[[Any], None] | None = None,
) -> str | None:
    """File one escalation under *anchor_task_id* into *project_root*'s queue.

    Args:
        project_root: The affected project's root; the escalation lands in that
            project's own ``data/escalations`` queue.  ``None`` is a
            DEBUG-logged no-op.
        anchor_task_id: The synthetic task id the record is filed under and the
            fold is keyed on.  Required, with no default, and never shared with
            another filer: see the module docstring.
        agent_role: The filer's role string.
        category: The escalation category.
        severity: ``'blocking'`` or ``'info'``; the caller's choice.  A
            born-at-L2 severity (``escalation.models.BORN_AT_L2_SEVERITIES``,
            ``'critical'``/``'urgent'``) pages a human directly, and only an
            *agent_role* with a ``harness-``/``orchestrator-`` sentinel prefix
            may file one.
        summary: One-line operator-facing summary.
        detail: The INV-2 structured evidence, already rendered by the caller.
        suggested_action: What a triager should do.
        logger: The caller's logger, so records are attributed to the caller's
            module rather than to this helper.
        log_label: The caller's grep token, prefixed onto every emitted
            message.
        level: Escalation level; defaults to 1.  Callers file under synthetic
            anchors that no steward ever claims, so an L0 record is left to
            ``orchestrator/src/orchestrator/harness.py::Harness._reap_orphan_l0_escalations``.
            While ``info_l0_router_enabled`` is on (the default), an
            ``'info'`` L0 goes to the info-L0 router
            (``orchestrator/src/orchestrator/escalation_router.py::router_owns``),
            and ``escalation/src/escalation/disposition.py::route_info_l0``
            decides it, converting a note with content into a curator task.
            Any other L0 is promoted to L1 after a delay.
        dedupe: Fold into an already-open escalation *agent_role* filed under
            *anchor_task_id* instead of filing a duplicate.  Another role's
            pending record under the anchor is not folded into.  That
            includes the L1 the orphan reaper re-files under its own role when
            it promotes an L0, so after a promotion each recurrence files a
            fresh L0, which the reaper then dismisses as covered by that L1.
            That churn is bounded and accepted.  Pass false when each record
            holds a distinct payload that a fold would discard.
        context: The caller's description of the subject, appended to every
            emitted message, including the no-op arms.
        on_fold: Called with the existing escalation in place of the default
            INFO fold line, so a caller can log the fold on its own logger at
            its own level.  An exception from it is logged, never raised.

    Returns the escalation id — freshly filed, or the id of the already-open
    escalation *agent_role* filed under this anchor when one exists — or
    ``None`` when filing was not possible. NEVER raises.

    Without the ``escalation`` package the alarm is LOST, so that arm logs at
    WARNING for every caller; the rationale lives in
    ``tests/test_folded_escalation.py::TestTheEmittedLevels``.
    """
    queue = _open_queue(
        project_root, logger=logger, log_label=log_label,
        outcome='nothing escalated', context=context,
    )
    if queue is None:
        return None

    # While a condition persists every event breaches again, so filing per
    # breach would bury the queue; a pending record THIS ROLE filed under the
    # anchor is the caller's open alarm. A read failure falls THROUGH to
    # filing: a possible duplicate is far cheaper than a silenced alarm.
    existing = _pending_in(
        queue, project_root, anchor_task_id=anchor_task_id, agent_role=agent_role,
        logger=logger, log_label=log_label,
        outcome='filing a new one rather than silencing the alarm',
        context=context,
    ) if dedupe else []
    if existing:
        if on_fold is not None:
            try:
                on_fold(existing[0])
            except Exception:
                logger.exception(
                    '%s: the on_fold hook failed while folding into %s%s',
                    log_label, existing[0].id, _suffix(context),
                )
        else:
            logger.info(
                '%s: %s already open; folding into it rather than filing a '
                'duplicate%s', log_label, existing[0].id, _suffix(context),
            )
        return existing[0].id

    # Constructed inside the guard so a malformed payload degrades to no
    # escalation. The event being complained about has already committed; a
    # failure here must cost the operator a heads-up, never the write.
    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(anchor_task_id),
            task_id=anchor_task_id,
            agent_role=agent_role,
            severity=severity,
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            level=level,
        )
        esc_id = queue.submit(esc)
    except Exception:
        logger.exception(
            '%s: failed to submit the alarm under anchor %r in '
            'project_root=%r%s',
            log_label, anchor_task_id, project_root, _suffix(context),
        )
        return None

    logger.warning('%s: queued %s%s', log_label, esc_id, _suffix(context))
    return esc_id


def pending_folded_escalations(
    project_root: str | None,
    *,
    anchor_task_id: str,
    agent_role: str,
    logger: logging.Logger,
    log_label: str,
) -> list[Escalation]:
    """Return the pending records *agent_role* filed under *anchor_task_id*.

    ``[]`` when there are none or the queue cannot be read; a no-op is logged
    as :func:`file_folded_escalation` logs it. NEVER raises.
    """
    queue = _open_queue(
        project_root, logger=logger, log_label=log_label, outcome='nothing read',
    )
    if queue is None:
        return []
    return _pending_in(
        queue, project_root, anchor_task_id=anchor_task_id, agent_role=agent_role,
        logger=logger, log_label=log_label, outcome='treating it as nothing pending',
    )


def _pending_in(
    queue: EscalationQueue,
    project_root: str | None,
    *,
    anchor_task_id: str,
    agent_role: str,
    logger: logging.Logger,
    log_label: str,
    outcome: str,
    context: str = '',
) -> list[Escalation]:
    """The pending records *agent_role* filed under *anchor_task_id*.

    ``[]`` on a read failure, logged with *outcome* as :func:`_open_queue`
    logs its no-op lines.
    """
    try:
        return queue.get_by_task(anchor_task_id, status='pending', agent_role=agent_role)
    except Exception:
        logger.exception(
            '%s: failed to read pending records under anchor %r in project_root=%r; %s%s',
            log_label, anchor_task_id, project_root, outcome, _suffix(context),
        )
        return []


def refresh_folded_escalation(
    project_root: str | None,
    *,
    escalation_id: str,
    level: int,
    summary: str,
    detail: str,
    logger: logging.Logger,
    log_label: str,
) -> bool:
    """Rewrite the pending record *escalation_id* at *level* and count the repeat.

    Over ``EscalationQueue.refresh_pending``: the id is kept and
    ``dedupe_count`` rises by one. Returns whether a pending record was
    refreshed; False for one that is absent, resolved or at another level, or
    when the queue cannot be opened. NEVER raises.
    """
    queue = _open_queue(
        project_root, logger=logger, log_label=log_label, outcome='nothing refreshed',
    )
    if queue is None:
        return False
    try:
        refreshed = queue.refresh_pending(
            escalation_id, level=level, summary=summary, detail=detail,
        )
    except Exception:
        logger.exception(
            '%s: failed to refresh %s in project_root=%r', log_label, escalation_id, project_root,
        )
        return False
    return refreshed is not None


def resolve_folded_escalations(
    project_root: str | None,
    *,
    anchor_task_id: str,
    agent_role: str,
    resolution: str,
    logger: logging.Logger,
    log_label: str,
) -> list[str]:
    """Resolve every record :func:`pending_folded_escalations` would return.

    Resolved as *agent_role*, so the record shows its filer closed it.  A
    failure on one record is logged and the next is still tried.  Returns the
    resolved ids. NEVER raises.
    """
    queue = _open_queue(
        project_root, logger=logger, log_label=log_label, outcome='nothing resolved',
    )
    if queue is None:
        return []
    pending = _pending_in(
        queue, project_root, anchor_task_id=anchor_task_id, agent_role=agent_role,
        logger=logger, log_label=log_label, outcome='nothing resolved',
    )
    resolved: list[str] = []
    for esc in pending:
        try:
            queue.resolve(esc.id, resolution, resolved_by=agent_role)
        except Exception:
            logger.exception('%s: failed to resolve %s', log_label, esc.id)
            continue
        resolved.append(esc.id)
    if resolved:
        logger.info('%s: resolved %s', log_label, ', '.join(resolved))
    return resolved
