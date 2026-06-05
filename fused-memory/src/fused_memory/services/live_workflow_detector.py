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
from typing import Optional

from fused_memory.services.orchestrator_detector import is_orchestrator_live_for

logger = logging.getLogger(__name__)

# Default age threshold for the recent-commit signal.  A commit newer than this
# value (relative to ``now``) counts as evidence of an active workflow.
DEFAULT_MAX_COMMIT_AGE_HOURS: float = 6.0

# Orchestrator branch naming convention: ``task/<id>``.
DEFAULT_BRANCH_PREFIX: str = 'task/'

# Timeout for each individual git subprocess call (seconds).
_GIT_TIMEOUT: int = 10


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
    last_commit_at: Optional[datetime]


def detect_live_workflow(
    task_id: str,
    project_root: str | Path,
    *,
    now: Optional[datetime] = None,
    max_commit_age_hours: float = DEFAULT_MAX_COMMIT_AGE_HOURS,
    branch_prefix: str = DEFAULT_BRANCH_PREFIX,
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

    Returns:
        A :class:`WorkflowLiveness` dataclass with all signals populated.
    """
    branch = f'{branch_prefix}{task_id}'
    root = str(project_root)

    worktree_registered = _check_worktree_registered(root, branch)
    last_commit_at, recent_commit = _check_recent_commit(
        root, branch, now=now, max_commit_age_hours=max_commit_age_hours
    )
    # NOTE: orchestrator_live is wired in step-4; left False here so step-1/2
    # tests (which monkeypatch the name) pass before step-4.
    orchestrator_live = is_orchestrator_live_for(project_root)

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

    Accepts the same keyword arguments as :func:`detect_live_workflow`.
    """
    return detect_live_workflow(task_id, project_root, **kwargs).is_live


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    for line in result.stdout.splitlines():
        if line.strip() == target:
            return True
    return False


def _check_recent_commit(
    project_root: str,
    branch: str,
    *,
    now: Optional[datetime],
    max_commit_age_hours: float,
) -> tuple[Optional[datetime], bool]:
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


def _parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
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
