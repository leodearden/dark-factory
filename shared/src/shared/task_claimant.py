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

__all__ = [
    'compose_claimant_run_id',
    'has_live_claimant',
    'is_stranded',
    'is_stranded_blocked',
]


def compose_claimant_run_id(run_id: str, session_id: str, owner_pid: int) -> str:
    """Compose the ``claimant_run_id`` identity stamped at dispatch (task 2188 / omega1).

    PRD ``plans/task-status-authority-prd.md`` contract C4 / decision D4: the
    dispatch-time claimant write embeds three components so the resulting
    string is unique per process-run and per workflow-session:

    - ``run_id``: the orchestrator process's run id (``Harness._run_id``).
    - ``session_id``: the workflow's per-task session id (``TaskWorkflow.session_id``).
    - ``owner_pid``: ``os.getpid()`` — deliberately the *same* value that
      ``artifacts.py`` records as ``plan.lock`` owner_pid, so this durable DB
      field mirrors the existing plan.lock/``_pid_alive`` liveness signal and
      W10 can cross-consume both without this module ripping out the
      plan.lock forensics.

    The format is a stable, greppable, labeled string:
    ``f'{run_id}/{session_id}/pid={owner_pid}'``. This is the identity
    consumed by :func:`is_stranded` via the ``claimant_run_id`` column.
    """
    return f'{run_id}/{session_id}/pid={owner_pid}'


def _claimant_liveness_stranded(task: Mapping, now: datetime, ttl: timedelta) -> bool:
    """Status-agnostic claimant/heartbeat liveness core shared by
    :func:`is_stranded`, :func:`is_stranded_blocked`, and (negated) by
    :func:`has_live_claimant`.

    Returns True when *task* has no live claimant — i.e. either:
      - it has no claimant at all (``claimant_run_id`` is ``None``/blank), OR
      - its ``heartbeat_at`` is missing/unparseable, OR
      - its ``heartbeat_at`` is older than *now* - *ttl*.

    Carries no status gate and no infra_hold check — those are the callers'
    responsibility (see :func:`is_stranded` / :func:`is_stranded_blocked`).

    Parameters
    ----------
    task:
        A task-dict-like mapping. Reads ``claimant_run_id`` and
        ``heartbeat_at``.
    now:
        The reference "current time". A naive (tz-less) value is tolerated —
        it is normalized to UTC rather than raising ``TypeError`` when
        compared against the tz-aware heartbeat.
    ttl:
        Heartbeat staleness threshold. A heartbeat older than ``now - ttl``
        is considered stale.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    claimant = task.get('claimant_run_id')
    if claimant is None or (isinstance(claimant, str) and not claimant.strip()):
        return True

    heartbeat, ok = parse_timestamp_or_warn(
        task.get('heartbeat_at'),
        context='task_claimant._claimant_liveness_stranded',
    )
    if not ok:
        return True

    return heartbeat < now - ttl


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

    # A first-class TaskStatus.INFRA_HOLD status can never reach this line —
    # the in-progress gate above already returned False for it. Only the
    # legacy metadata.infra_hold overload (pre-omega4 migration window, see
    # module docstring) is checked here.
    metadata = task.get('metadata')
    if isinstance(metadata, Mapping) and metadata.get('infra_hold'):
        return False

    return _claimant_liveness_stranded(task, now, ttl)


def is_stranded_blocked(task: Mapping, now: datetime, ttl: timedelta) -> bool:
    """Return True when *task* is a claimed-but-abandoned ``blocked`` task.

    :func:`is_stranded`'s blocked-status sibling (task 2408 mechanism 2,
    consumed by the scheduler's blocked-redispatch sweep). A sibling function
    is needed — rather than parameterizing ``is_stranded`` by status —
    because ``is_stranded``'s ``in-progress`` gate makes it unconditionally
    return False for blocked tasks.

    Same claimant/heartbeat/infra_hold truth table as :func:`is_stranded`,
    gated on ``status == 'blocked'`` instead of ``'in-progress'``.

    Note: this predicate answers only "is there no live claimant" — it does
    NOT distinguish a genuine crash-strand from a deliberate park (e.g. a
    human ``/unblock`` session or a deterministic born-at-L2 gate), both of
    which present identically here (null/stale claimant). Callers that flip
    status on this signal must apply their own additional park-protection
    guards (see the scheduler's stranded-blocked-redispatch sweep).

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
    if status != TaskStatus.BLOCKED.value:
        return False

    metadata = task.get('metadata')
    if isinstance(metadata, Mapping) and metadata.get('infra_hold'):
        return False

    return _claimant_liveness_stranded(task, now, ttl)


def has_live_claimant(task: Mapping, now: datetime, ttl: timedelta) -> bool:
    """Return True when *task* currently has a LIVE claimant, regardless of
    its status.

    Status-agnostic dispatch-gate primitive (task 2408 mechanism 1, consumed
    by the scheduler's ``_eligible_for_dispatch`` to refuse dispatching into
    a task that is currently claimed by a live workflow). Simply the negation
    of the shared liveness core — no status gate, no infra_hold check.

    Parameters
    ----------
    task:
        A task-dict-like mapping. Reads ``claimant_run_id`` and
        ``heartbeat_at``.
    now:
        The reference "current time". A naive (tz-less) value is tolerated —
        it is normalized to UTC rather than raising ``TypeError`` when
        compared against the tz-aware heartbeat.
    ttl:
        Heartbeat staleness threshold. A heartbeat older than ``now - ttl``
        is considered stale (i.e. not live).
    """
    return not _claimant_liveness_stranded(task, now, ttl)
