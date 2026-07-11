"""Detect whether a live workflow (orchestrator worktree) is active for a given task.

Provides a per-task/branch liveness signal that complements the project-level
``is_orchestrator_live_for`` check.  Three OR-ed signals bias toward 'live' so
recon does not race a working pipeline:

1. **worktree_registered** — ``git worktree list --porcelain`` lists a worktree
   whose branch is ``task/<task_id>``.
2. **recent_commit** — the tip of ``task/<task_id>`` is newer than
   *max_commit_age_hours* (default: 6 h), detected via ``git log -1 --format=%cI``.
3. **orchestrator_live** — ``is_orchestrator_live_for(project_root)`` returns True
   (the project-level orchestrator-lock signal).

``is_live = worktree_registered OR recent_commit OR orchestrator_live``

Fail-safe contract: a missing branch, subprocess error, or unparseable timestamp
makes *that individual signal* False without raising.  OR-aggregation means a
transient git glitch never flips a genuinely live task to 'not live'.

**Important — orchestrator_live is project-wide, not per-task.**
``is_orchestrator_live_for(project_root)`` holds one PID lock for the entire
project.  When it is True, *every* task in the project will have ``is_live=True``
regardless of its worktree or commit recency.  This is intentional: while any
task in the project is being actively dispatched, recon cannot distinguish which
specific task the orchestrator is acting on without deeper orchestrator coupling,
so it conservatively treats all tasks as owned.  Consequence: while the project
orchestrator is alive, stranded-work escalations are suppressed project-wide,
not just for the task the orchestrator is currently processing.

**Status scoping.** Callers that know the task's current status may pass it via
the ``status`` keyword.  When ``status`` is one of ``ORCH_LIVE_INELIGIBLE_STATUSES``
(statuses that are never actively dispatched: ``deferred``, ``done``,
``cancelled``), the project-wide ``orchestrator_live`` signal is forced to
``False`` for that call — the orchestrator being alive elsewhere is not evidence
that *this* task is live.  The per-task ``worktree_registered`` and
``recent_commit`` signals are unaffected by ``status``, so genuine per-branch
evidence still marks the task live.  ``status=None`` (the default) preserves the
prior, status-blind behavior.

**Blocked deterministic tasks.** Callers may also pass ``task_kind`` (the task's
``metadata.task_kind``).  Deliberately, ``'blocked'`` is NOT added to
``ORCH_LIVE_INELIGIBLE_STATUSES`` wholesale — a normal blocked task may
auto-unblock and be legitimately mid-pipeline, so the project-wide orchestrator
lock remains real per-task evidence for it (task 2031).  However, a
**deterministic** task (``task_kind == 'deterministic'``) never acquires a
worktree/branch — it is routed to ``DeterministicRunner`` instead — so for a
*blocked deterministic* task the bare orchestrator lock is not task-specific
evidence either.  ``detect_live_workflow`` therefore also forces
``orchestrator_live`` to ``False`` when ``status == 'blocked' AND task_kind ==
'deterministic'`` (see :func:`_orchestrator_signal_ineligible`).  This is a
compound condition, not a status addition: non-deterministic blocked tasks
(``task_kind`` absent or not ``'deterministic'``) keep the signal.  The
per-task ``worktree_registered``/``recent_commit`` signals remain unaffected,
so a blocked deterministic task that somehow did acquire a worktree is still
live.

**Blocked normal tasks (task 2409).** The same bare-orchestrator-lock problem
also affects **normal** (``task_kind`` absent or ``'normal'``) blocked tasks:
a blocked, normal-kind task with satisfied dependencies and no genuine live
pipeline can show only the project-wide ``orchestrator`` signal and be treated
as owned/live indefinitely, so reconciliation/redispatch never fires — the
repeated re-deferral loop observed for tasks 2335/2196.  ``detect_live_workflow``
therefore also forces ``orchestrator_live`` to ``False`` when ``status ==
'blocked' AND task_kind in (None, 'normal') AND NOT worktree_registered AND NOT
recent_commit``.  Unlike the deterministic rule, this one is guarded on the
per-task git signals: a normal blocked task *can* legitimately hold a
registered worktree or a recent commit while auto-unblocking mid-pipeline
(task 2031's original concern), so the guard only suppresses the bare case —
whenever real per-task evidence exists, ``orchestrator_live`` is reported
honestly and the task remains live via that evidence (``is_live`` is
unaffected either way, since it already ORs in ``worktree_registered``/
``recent_commit``).

The legitimate stranded case (orchestrator down, no worktree, no recent commits)
has all three signals False, so recon still escalates it.

Branch convention: ``task/<task_id>`` (matches the orchestrator's worktree naming).
Injectable ``now`` for deterministic tests.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fused_memory.services.orchestrator_detector import is_orchestrator_live_for

logger = logging.getLogger(__name__)

# Default age threshold for the recent-commit signal.  A commit newer than this
# value (relative to ``now``) counts as evidence of an active workflow.
DEFAULT_MAX_COMMIT_AGE_HOURS: float = 6.0

# Orchestrator branch naming convention: ``task/<id>``.
DEFAULT_BRANCH_PREFIX: str = 'task/'

# Timeout for each individual git subprocess call (seconds).
_GIT_TIMEOUT: int = 10

# Statuses that are never actively dispatched by the orchestrator.  For a task
# in one of these statuses, the project-wide orchestrator PID-lock signal is not
# evidence of a live pipeline for *that* task, so it is dropped (see `status`
# param on detect_live_workflow).  Deliberately excludes 'blocked' (may
# auto-unblock), 'review'/'merge-deferred' (in the verify/review/merge
# pipeline), and 'pending'/'in-progress' (dispatch-eligible or active) — for
# those statuses a live orchestrator legitimately means "don't race the
# pipeline."  A blocked *deterministic* task is handled separately by
# `_orchestrator_signal_ineligible` below — see its docstring — rather than by
# adding 'blocked' here, since that would wrongly suppress normal blocked tasks.
ORCH_LIVE_INELIGIBLE_STATUSES: frozenset[str] = frozenset({'deferred', 'done', 'cancelled'})

# task_kind value used by the orchestrator/scheduler for deterministic tasks
# (DeterministicRunner-routed: no worktree/branch is ever acquired). Mirrors
# Scheduler._is_deterministic's metadata.task_kind == 'deterministic' check.
DETERMINISTIC_TASK_KIND: str = 'deterministic'

# task_kind value (and documented default when metadata.task_kind is unset) for
# ordinary LLM-agent-pipeline tasks. Used by rule 3 of
# _orchestrator_signal_ineligible (task 2409) to scope the blocked-normal
# bare-orchestrator suppression.
NORMAL_TASK_KIND: str = 'normal'


@dataclass(frozen=True)
class WorkflowLiveness:
    """Result of :func:`detect_live_workflow`.

    Attributes:
        is_live: True when any of the three signals indicates an active workflow.
        worktree_registered: True when a git worktree is registered for ``branch``.
        recent_commit: True when the tip of ``branch`` is newer than the threshold.
        orchestrator_live: True when the project-level orchestrator lock is live.
        branch: The branch name inspected (e.g. ``task/4321``).
        last_commit_at: Parsed commit timestamp when ``recent_commit`` was evaluated.
            ``None`` when the branch was absent or the timestamp was unparseable.
    """

    is_live: bool
    worktree_registered: bool
    recent_commit: bool
    orchestrator_live: bool
    branch: str
    last_commit_at: datetime | None


def detect_live_workflow(
    task_id: str,
    project_root: str | Path,
    *,
    now: datetime | None = None,
    max_commit_age_hours: float = DEFAULT_MAX_COMMIT_AGE_HOURS,
    branch_prefix: str = DEFAULT_BRANCH_PREFIX,
    status: str | None = None,
    task_kind: str | None = None,
    _orchestrator_live: bool | None = None,
) -> WorkflowLiveness:
    """Detect whether a live workflow is active for *task_id*.

    Args:
        task_id: Task identifier (numeric string, e.g. ``"4321"``).
        project_root: Absolute path to the git repository root.
        now: Reference time for the recent-commit age check.  Defaults to
            ``datetime.now(UTC)`` when ``None``.  Pass a fixed datetime in
            tests for determinism.
        max_commit_age_hours: Commits newer than this many hours count as recent.
        branch_prefix: Branch name prefix; combined with *task_id* to form the
            branch name (e.g. ``"task/4321"``).
        status: The task's current status, when known.  When this is a member
            of :data:`ORCH_LIVE_INELIGIBLE_STATUSES` (statuses never actively
            dispatched: ``deferred``, ``done``, ``cancelled``), the project-wide
            ``orchestrator_live`` signal is forced to ``False`` for this call,
            short-circuiting before ``_orchestrator_live``/
            ``is_orchestrator_live_for`` are consulted.  ``None`` (default)
            leaves the orchestrator_live computation unaffected — fully
            backward compatible.
        task_kind: The task's ``metadata.task_kind``, when known.  When
            ``status == 'blocked'`` and ``task_kind == DETERMINISTIC_TASK_KIND``,
            the project-wide ``orchestrator_live`` signal is also forced to
            ``False`` — a deterministic task never acquires a worktree/branch,
            so the bare project lock is not task-specific evidence for it while
            blocked.  Likewise (task 2409), when ``status == 'blocked'`` and
            ``task_kind`` is ``None`` or ``NORMAL_TASK_KIND`` and there is
            neither a registered worktree nor a recent commit for the task's
            own branch, the signal is forced ``False`` too — see
            :func:`_orchestrator_signal_ineligible` for both rules.  For a
            blocked task with genuine per-task evidence (a registered worktree
            or a recent commit), or for any non-blocked status, the
            orchestrator_live computation is unaffected by ``task_kind``.
        _orchestrator_live: Pre-computed project-level orchestrator-lock result.
            When provided, skips the ``is_orchestrator_live_for(project_root)``
            call — use this to hoist the constant project-level check out of
            per-task loops (e.g. in :func:`_render_live_workflow_section`).
            ``None`` (default) triggers a fresh ``is_orchestrator_live_for``
            call.  Tests monkeypatch the module attribute directly; this
            parameter is only for performance hoisting, not test isolation.
            Ignored when :func:`_orchestrator_signal_ineligible` returns True
            for the given *status*/*task_kind* pair.

    Returns:
        A :class:`WorkflowLiveness` dataclass with all signals populated.
    """
    branch = f'{branch_prefix}{task_id}'
    root = str(project_root)

    worktree_registered = _check_worktree_registered(root, branch)
    last_commit_at, recent_commit = _check_recent_commit(
        root, branch, now=now, max_commit_age_hours=max_commit_age_hours
    )
    # orchestrator_live is the project-level lock signal (True when the
    # orchestrator process holds an active lock for this project_root, regardless
    # of which task it is currently dispatching).  Pre-computed callers may pass
    # it via _orchestrator_live to avoid redundant per-task subprocess calls.
    # Ineligible status/task_kind combinations (see
    # _orchestrator_signal_ineligible) force this signal False — the
    # project-wide lock is not evidence of liveness for a task that will never
    # be dispatched (or, for blocked deterministic tasks, never acquires a
    # worktree/branch of its own; or, for blocked normal tasks with no
    # per-task git evidence, task 2409).  worktree_registered/recent_commit
    # (already computed above) are threaded through so rule 3 can require
    # their absence.
    if _orchestrator_signal_ineligible(status, task_kind, worktree_registered, recent_commit):
        orchestrator_live = False
    else:
        orchestrator_live = (
            _orchestrator_live
            if _orchestrator_live is not None
            else is_orchestrator_live_for(project_root)
        )

    is_live = worktree_registered or recent_commit or orchestrator_live

    return WorkflowLiveness(
        is_live=is_live,
        worktree_registered=worktree_registered,
        recent_commit=recent_commit,
        orchestrator_live=orchestrator_live,
        branch=branch,
        last_commit_at=last_commit_at,
    )


def is_workflow_live_for_task(
    task_id: str,
    project_root: str | Path,
    **kwargs,
) -> bool:
    """Convenience wrapper — returns :attr:`WorkflowLiveness.is_live`.

    Accepts the same keyword arguments as :func:`detect_live_workflow`,
    including the ``_orchestrator_live`` performance hint.
    """
    return detect_live_workflow(task_id, project_root, **kwargs).is_live


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _orchestrator_signal_ineligible(
    status: str | None,
    task_kind: str | None,
    worktree_registered: bool = False,
    recent_commit: bool = False,
) -> bool:
    """Return True when the project-wide ``orchestrator_live`` signal must be
    forced False for this *status*/*task_kind* (/*worktree_registered*/
    *recent_commit*) combination.

    Three independent rules are centralized here:

    1. ``status`` is a member of :data:`ORCH_LIVE_INELIGIBLE_STATUSES`
       (``deferred``, ``done``, ``cancelled``) — statuses that are never
       actively dispatched, so the project-wide lock is not evidence for this
       task (task 2031).
    2. ``status == 'blocked' and task_kind == DETERMINISTIC_TASK_KIND`` — a
       blocked deterministic task, which never acquires a worktree/branch (it
       is routed to ``DeterministicRunner`` instead), so the bare project lock
       is not task-specific evidence for it either (task 2067). Unconditional:
       a deterministic task never has git evidence to lose.
    3. ``status == 'blocked' and task_kind in (None, NORMAL_TASK_KIND) and not
       worktree_registered and not recent_commit`` — a blocked normal (or
       task_kind-absent) task with NO per-task git evidence, which is the
       bare-orchestrator false positive that caused tasks 2335/2196 to loop
       through repeated re-deferral (task 2409). Guarded on the git signals
       (unlike rule 2): a normal blocked task can legitimately hold a
       worktree/recent commit while auto-unblocking mid-pipeline (task 2031's
       concern), so this rule only suppresses the bare case, leaving
       ``is_live`` (which already ORs in ``worktree_registered``/
       ``recent_commit``) unaffected whenever real per-task evidence exists.

    ``'blocked'`` is deliberately NOT added to ``ORCH_LIVE_INELIGIBLE_STATUSES``
    wholesale: a normal blocked task (``task_kind`` absent or not
    ``'deterministic'``) may auto-unblock and be legitimately mid-pipeline, so
    the project-wide orchestrator lock remains real per-task evidence for it
    *when there is other evidence to corroborate it*. Hence the compound
    (status AND task_kind [AND NOT git-evidence]) conditions for rules 2 and 3,
    rather than an unconditional status addition.
    """
    if status is not None and status in ORCH_LIVE_INELIGIBLE_STATUSES:
        return True
    if status != 'blocked':
        return False
    if task_kind == DETERMINISTIC_TASK_KIND:
        return True
    return (
        task_kind in (None, NORMAL_TASK_KIND)
        and not worktree_registered
        and not recent_commit
    )


def _check_worktree_registered(project_root: str, branch: str) -> bool:
    """Return True iff a git worktree is registered for *branch*.

    Parses ``git -C <root> worktree list --porcelain`` output.  Any subprocess
    error or unexpected output silently returns False (fail-safe).
    """
    try:
        result = subprocess.run(
            ['git', '-C', project_root, 'worktree', 'list', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug('live_workflow_detector: worktree list failed: %s', exc)
        return False

    if result.returncode != 0:
        logger.debug(
            'live_workflow_detector: worktree list returned %d: %s',
            result.returncode, result.stderr.strip(),
        )
        return False

    target = f'branch refs/heads/{branch}'
    return any(line.strip() == target for line in result.stdout.splitlines())


def _check_recent_commit(
    project_root: str,
    branch: str,
    *,
    now: datetime | None,
    max_commit_age_hours: float,
) -> tuple[datetime | None, bool]:
    """Return (last_commit_at, recent_commit) for *branch*.

    ``last_commit_at`` is the parsed commit timestamp, or ``None`` if unavailable.
    ``recent_commit`` is True when the commit is within *max_commit_age_hours*
    of *now* (or ``datetime.now(UTC)`` when *now* is ``None``).
    """
    try:
        result = subprocess.run(
            ['git', '-C', project_root, 'log', '-1', '--format=%cI', branch],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug('live_workflow_detector: git log failed for %s: %s', branch, exc)
        return None, False

    if result.returncode != 0:
        logger.debug(
            'live_workflow_detector: git log returned %d for branch %s (branch may not exist)',
            result.returncode, branch,
        )
        return None, False

    ts_str = result.stdout.strip()
    if not ts_str:
        return None, False

    last_commit_at = _parse_iso_timestamp(ts_str)
    if last_commit_at is None:
        logger.debug(
            'live_workflow_detector: cannot parse commit timestamp %r for branch %s',
            ts_str, branch,
        )
        return None, False

    reference = now if now is not None else datetime.now(UTC)
    age = reference - last_commit_at
    recent_commit = age <= timedelta(hours=max_commit_age_hours)
    return last_commit_at, recent_commit


def _parse_iso_timestamp(ts_str: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string from ``git log --format=%cI``.

    Returns ``None`` on any parse error.
    """
    try:
        dt = datetime.fromisoformat(ts_str)
        # Ensure UTC-aware for consistent comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None
