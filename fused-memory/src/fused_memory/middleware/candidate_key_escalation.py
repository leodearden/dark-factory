"""File a residual-duplicate escalation for the sqlite backend's v3->v4
self-gating migration (fm-task-dedup W8 task A2).

When ``_migrate_v3_to_v4`` (:mod:`fused_memory.backends.sqlite_task_backend`)
finds non-cancelled duplicate ``candidate_key`` groups still present at
connection-open, it skips building the partial UNIQUE index and calls
:func:`emit_residual_candidate_key_escalation` so an operator sees the
residuals surfaced as an escalation, not just a log line.

Mirrors the defensive import + never-raise submit pattern established by
:mod:`fused_memory.middleware.scope_violation_escalator`:

* Defensive import of the optional ``escalation`` workspace package — when
  missing (minimal envs, tests without escalation infra), this becomes a
  logged no-op so the migration's fail-safe guarantee is never at risk.
* Escalations land in ``{project_root}/data/escalations`` — the affected
  project's own queue.
* ``submit`` is wrapped so a queue I/O failure never raises — the migration's
  self-gating skip has already happened by the time this is called;
  escalation is purely additive.
* Dedups against an already-open escalation for the same anchor before
  filing a new one (review amendment) — a connection (and therefore this
  migration step) runs at most once per project_root per process, so a
  residual-dup condition that outlives a single process restart would
  otherwise mint a brand-new escalation on every restart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Mirror the defensive-import pattern from scope_violation_escalator.py so
# this module silently no-ops when the escalation package is unavailable
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
# escalation IDs (e.g. ``esc-candidate-key-migration-1``) are easily
# greppable and distinct from the path-guard's ``task-path-guard`` series.
_ANCHOR_TASK_ID: str = 'candidate-key-migration'

_AGENT_ROLE: str = 'fused-memory/candidate-key-migration'
_CATEGORY: str = 'candidate_key_residual_duplicates'


def emit_residual_candidate_key_escalation(
    project_root: str,
    residual_groups: list[dict[str, Any]],
) -> str | None:
    """File a ``candidate_key_residual_duplicates`` escalation naming the
    residual non-cancelled duplicate ``candidate_key`` groups a v3->v4
    connection-open audit found.

    ``residual_groups`` entries carry ``tag``, ``candidate_key``,
    ``task_ids`` (list[str]), and ``count`` — the same shape the migration's
    ``GROUP_CONCAT``-based audit produces.

    Returns the escalation id — either a freshly filed one, or the id of an
    already-open escalation for this condition when one exists (dedup, see
    below) — or ``None`` when neither is possible (the ``escalation``
    package is unavailable, or the queue write failed). NEVER raises: this
    is called from connection-open migration code, and a raise here would
    defeat the self-gating step's own fail-safe guarantee.
    """
    if not HAS_ESCALATION:
        logger.debug(
            'candidate_key_escalation: escalation package unavailable; '
            '%d residual duplicate group(s) in project_root=%r will not be '
            'escalated',
            len(residual_groups), project_root,
        )
        return None

    queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)

    # Dedup against an already-open escalation (review amendment): connections
    # are cached per project_root, so this migration step runs at most once
    # per project_root per process — but a residual-dup condition that
    # outlives a single process restart would otherwise mint a brand-new
    # esc-candidate-key-migration-N escalation on every restart, flooding the
    # operator queue with near-identical entries. `_ANCHOR_TASK_ID` is a
    # stable per-project anchor, so any still-pending (including
    # parked-to-L2) escalation filed under it IS the open residual-duplicates
    # escalation for this project — reuse its id instead of filing a
    # duplicate. Best-effort: a read failure here just falls through to
    # filing a new escalation rather than blocking the migration step.
    try:
        existing = queue.get_by_task(_ANCHOR_TASK_ID, status='pending')
    except Exception:
        logger.exception(
            'candidate_key_escalation: failed to check for an existing open '
            'escalation for project_root=%r; proceeding to file a new one',
            project_root,
        )
        existing = []
    if existing:
        logger.info(
            'candidate_key_escalation: %s already open for project_root=%r '
            '(%d residual duplicate group(s) now); not filing a duplicate',
            existing[0].id, project_root, len(residual_groups),
        )
        return existing[0].id

    groups_desc = '; '.join(
        f'tag={g.get("tag")!r} candidate_key={g.get("candidate_key")!r} '
        f'task_ids={g.get("task_ids")!r} count={g.get("count")!r}'
        for g in residual_groups
    )
    detail_lines = [
        f'project_root={project_root!r}',
        f'residual_group_count={len(residual_groups)}',
        f'groups={groups_desc}',
        '',
        'The v3->v4 schema migration (fm-task-dedup W8 task A2) found '
        'non-cancelled rows sharing the same (tag, candidate_key) at '
        'connection-open, so it skipped building the partial UNIQUE index '
        'ux_tasks_candidate_key. Clean up the residual duplicates (cancel '
        'or merge the extras in each group above) — the next '
        'connection-open will re-audit and land the index automatically.',
    ]
    detail = '\n'.join(detail_lines)

    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(_ANCHOR_TASK_ID),
            task_id=_ANCHOR_TASK_ID,
            agent_role=_AGENT_ROLE,
            severity='blocking',
            category=_CATEGORY,
            summary=(
                f'{len(residual_groups)} residual duplicate candidate_key '
                f'group(s) blocking the UNIQUE index build'
            ),
            detail=detail,
            suggested_action='clean up residual duplicates (cancel or merge)',
            level=1,
        )
        esc_id = queue.submit(esc)
    except Exception:
        # Queue I/O failure must not propagate — the migration's self-gating
        # skip has already happened; the operator just doesn't get the
        # heads-up. Mirror scope_violation_escalator's tolerance so a broken
        # filesystem can't turn a connection-open migration into a crash.
        logger.exception(
            'candidate_key_escalation: failed to submit escalation for '
            'project_root=%r (%d residual group(s))',
            project_root, len(residual_groups),
        )
        return None

    logger.warning(
        'candidate_key_escalation: queued %s for project_root=%r '
        '(%d residual duplicate group(s))',
        esc_id, project_root, len(residual_groups),
    )
    return esc_id
