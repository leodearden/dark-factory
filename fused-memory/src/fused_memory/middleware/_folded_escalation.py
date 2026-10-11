"""The pending-anchor-fold escalation filer skeleton, in one place (INV-5).

:func:`file_folded_escalation` owns the defensive optional-``escalation``
import, the guarded ``EscalationQueue`` construction, the
``get_by_task(anchor, status='pending')`` dedupe fold, and the never-raise
``Escalation(...)`` + ``queue.submit(...)``.  Each caller keeps its own
``_ANCHOR_TASK_ID`` / ``_AGENT_ROLE`` / ``_CATEGORY`` constants and builds its
own summary and detail.

EVERY FILER'S ANCHOR MUST BE UNIQUE.  A filer that dedupes against an anchor
somebody else keeps open never files again, and that silence is
indistinguishable from health.  Measured: the L1 escalation watcher squatted
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# The `escalation` workspace package is optional (minimal CI envs, deployments
# that have not installed it); without it every filer is a logged no-op.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

_QUEUE_DIRNAME: str = 'data/escalations'


def _suffix(context: str) -> str:
    """Render the caller's *context* as a trailing log clause, or nothing."""
    return f' ({context})' if context else ''


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
    no_escalation_level: int = logging.DEBUG,
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
        severity: ``'blocking'`` or ``'info'``; the caller's choice.
        summary: One-line operator-facing summary.
        detail: The INV-2 structured evidence, already rendered by the caller.
        suggested_action: What a triager should do.
        logger: The caller's logger, so records are attributed to the caller's
            module rather than to this helper.
        log_label: The caller's grep token, prefixed onto every emitted
            message.
        level: Escalation level; defaults to 1.  Callers file under synthetic
            anchors that no steward ever claims, so an L0 record only waits
            for ``HarnessRunner._reap_orphan_l0_escalations`` to promote it.
        dedupe: Fold into an already-open escalation under *anchor_task_id*
            instead of filing a duplicate.  Pass false when each record holds
            a distinct payload that a fold would discard.
        context: The caller's description of the subject, appended to every
            emitted message, including the no-op arms.
        on_fold: Called with the existing escalation in place of the default
            INFO fold line, so a caller can log the fold on its own logger at
            its own level.  An exception from it is logged, never raised.
        no_escalation_level: Log level for the package-unavailable arm;
            defaults to DEBUG.

    Returns the escalation id — freshly filed, or the id of the already-open
    escalation under this anchor when one exists — or ``None`` when filing was
    not possible. NEVER raises.
    """
    if project_root is None:
        logger.debug(
            '%s: no project_root, so there is no project queue to file into; '
            'nothing escalated%s', log_label, _suffix(context),
        )
        return None

    if not HAS_ESCALATION:
        logger.log(
            no_escalation_level,
            '%s: escalation package unavailable; nothing will be escalated%s',
            log_label, _suffix(context),
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        # Constructing the queue creates its directory; a read-only or missing
        # project_root must not turn an alarm into a crash on a write path.
        logger.exception(
            '%s: could not open the escalation queue at project_root=%r; '
            'nothing escalated%s', log_label, project_root, _suffix(context),
        )
        return None

    # While a condition persists every event breaches again, so filing per
    # breach would bury the queue; any pending record under the anchor IS this
    # caller's open alarm. A read failure falls THROUGH to filing: a possible
    # duplicate is far cheaper than a silenced alarm.
    existing = []
    if dedupe:
        try:
            existing = queue.get_by_task(anchor_task_id, status='pending')
        except Exception:
            logger.exception(
                '%s: failed to check for an already-open alarm under anchor %r '
                'in project_root=%r; proceeding to file a new one rather than '
                'silencing the alarm%s',
                log_label, anchor_task_id, project_root, _suffix(context),
            )
            existing = []
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
