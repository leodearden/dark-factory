"""is_stranded predicate — Table C4 of the task-status-authority contract.

See PRD ``plans/task-status-authority-prd.md`` (contract C4, decision D4):
"stranded" becomes a queryable predicate backed by first-class
``claimant_run_id``/``heartbeat_at`` columns (task 2182 / rho2), replacing
``plan.lock``/owner-pid forensics.

NOT-infra-held rationale
-------------------------
:func:`is_stranded` represents "NOT infra-held" primarily through its
``status == 'in-progress'`` gate: decision D3 makes ``infra-hold`` a
first-class status, so an in-progress task is, by construction, never
infra-held. The supplementary ``metadata.infra_hold`` check exists only as
defensive safety for the pre-omega4 migration window, when some tasks may
still carry the legacy "in-progress + metadata.infra_hold=True" overload
instead of the new first-class status. Once omega4 completes that migration,
the metadata check becomes permanently dead but harmless.

This module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path
(``from shared.task_claimant import is_stranded``), consistent with the
``task_statuses``/``timestamps``/``mcp_envelope`` sub-module convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from shared.task_statuses import TaskStatus
from shared.timestamps import parse_timestamp_or_warn

__all__ = ['is_stranded']


def is_stranded(task: Mapping, now: datetime, ttl: timedelta) -> bool:
    """Return True when *task* is a claimed-but-abandoned in-progress task.

    A task is stranded when it is ``in-progress`` AND either:
      - it has no live claimant (``claimant_run_id`` is ``None``/blank), OR
      - its ``heartbeat_at`` is missing/unparseable, OR
      - its ``heartbeat_at`` is older than *now* - *ttl*,

    AND it is not infra-held (see module docstring).

    Parameters
    ----------
    task:
        A task-dict-like mapping. Reads ``status``, ``claimant_run_id``,
        ``heartbeat_at``, and (defensively) ``metadata``.
    now:
        The reference "current time". A naive (tz-less) value is tolerated —
        it is normalized to UTC rather than raising ``TypeError`` when
        compared against the tz-aware heartbeat.
    ttl:
        Heartbeat staleness threshold. A heartbeat older than ``now - ttl``
        is considered stale.
    """
    status = str(task.get('status'))
    if status != TaskStatus.IN_PROGRESS.value:
        return False

    metadata = task.get('metadata')
    if status == TaskStatus.INFRA_HOLD.value or (
        isinstance(metadata, Mapping) and metadata.get('infra_hold')
    ):
        return False

    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    claimant = task.get('claimant_run_id')
    if claimant is None or (isinstance(claimant, str) and not claimant.strip()):
        return True

    heartbeat, ok = parse_timestamp_or_warn(
        task.get('heartbeat_at'), context='task_claimant.is_stranded',
    )
    if not ok:
        return True

    return heartbeat < now - ttl
