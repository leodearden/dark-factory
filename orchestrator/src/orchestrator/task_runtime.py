"""Harness.task_runtime_snapshot() accessor core (task 2634, PRD
plans/dashboard-task-runtime-endpoint-prd.md task alpha).

Per-task runtime state — ``{task_id, has_worktree, loops, attempts, started,
lane, phase, lane_state}`` — for every active task on the local host, read
directly from disk via the orchestrator's OWN format owners (
``TaskArtifacts``, ``LaneLifecycle``, ``GitOps``).

The heavy logic lives here as a free function,
:func:`build_task_runtime_snapshot`, so it is testable against a
*constructible* ``GitOps`` (tmp git repo) rather than a full ``Harness``.
``Harness.task_runtime_snapshot()`` is a thin one-line delegator to this
function. This module is the *intermediate* producer a later MCP tool
projects to the dashboard's wire schema.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from orchestrator.artifacts import TaskArtifacts
from orchestrator.lane_lifecycle import LaneState

logger = logging.getLogger(__name__)


@dataclass
class TaskRuntimeState:
    """One task's runtime snapshot — the unit ``build_task_runtime_snapshot``
    returns a (task_id-sorted) list of.

    ``error`` is ``None`` on a successful artifact read. On a per-task
    artifact READ FAILURE (as opposed to honest-empty artifacts), ``error``
    carries a non-empty diagnostic string and ``loops``/``attempts``/
    ``phase``/``started`` are all ``None`` — never a fabricated ``0`` (INV-2,
    structured-facts-at-failure). ``lane``/``lane_state``/``has_worktree``
    are sourced independently (from the lane record / worktree dir, not the
    artifact read) and stay populated even when ``error`` is set.
    """

    task_id: int
    has_worktree: bool
    loops: int | None
    attempts: int | None
    started: str | None
    lane: str | None
    phase: str | None
    lane_state: str | None
    error: str | None = None


def _derive_phase(plan: dict) -> str:
    """Coarse PLAN/EXECUTE/DONE derivation from a ``plan.json`` dict's
    ``steps`` — reproduces the dashboard's exact rule
    (``dashboard/src/dashboard/data/orchestrator.py``'s
    ``read_task_artifacts``, lines 279-296) for parity with today (Open-Q1:
    ship coarse, not event_store phase).
    """
    steps = plan.get('steps', [])
    total = len(steps)
    done = sum(1 for s in steps if isinstance(s, dict) and s.get('status') == 'done')
    if total == 0:
        return 'PLAN'
    if done == total:
        return 'DONE'
    return 'EXECUTE'


# The 6-state durable LaneState maps onto the contract's 3 task-relevant
# lane_state values (PRD decision 5 / resolved design decision above);
# SEED/REGISTERED (task-less states) are absent from this map and resolve to
# None via .get().
_LANE_STATE_MAP: dict[LaneState, str] = {
    LaneState.ASSIGNED: 'assigned',
    LaneState.IN_USE: 'assigned',
    LaneState.QUARANTINED: 'quarantined',
    LaneState.RELEASED: 'released',
}


def _map_lane_state(state: LaneState) -> str | None:
    """Map a durable ``LaneState`` onto the contract's 3 task-relevant
    ``lane_state`` values, or ``None`` for a task-less state (SEED/REGISTERED).
    """
    return _LANE_STATE_MAP.get(state)


# A worktree/lane dirname yields a task id as either the bare id ('42') or
# 'task-<id>' ('task-42').
_TASK_ID_RE = re.compile(r'^(?:task-)?(\d+)$')


def _extract_task_id(dirname: str) -> int | None:
    """Parse a task id out of a worktree/lane dirname, or ``None`` if the
    name doesn't match either recognised shape (e.g. '_lane-9', 'random-dir',
    '_merge-verify').
    """
    match = _TASK_ID_RE.match(dirname)
    if match is None:
        return None
    return int(match.group(1))


def _read_task_entry(
    *,
    task_id: int,
    worktree: Path,
    meta_root: Path,
    has_worktree: bool,
    lane: str | None,
    lane_state: str | None,
    event_store=None,
) -> TaskRuntimeState:
    """Read one task's artifacts into a :class:`TaskRuntimeState`.

    Shared by both the pooled and non-pooled enumeration branches so B4
    honest-zero and phase parity hold identically for both layouts.
    """
    artifacts = TaskArtifacts(worktree, meta_root)
    entries, _corrupted = artifacts.read_iteration_log()
    loops = len(entries)
    attempts = len(artifacts.read_reviews())
    phase = _derive_phase(artifacts.read_plan())
    started = artifacts.read_created_at()
    return TaskRuntimeState(
        task_id=task_id,
        has_worktree=has_worktree,
        loops=loops,
        attempts=attempts,
        started=started,
        lane=lane,
        phase=phase,
        lane_state=lane_state,
        error=None,
    )


def _build_non_pooled_snapshot(*, git_ops, event_store) -> list[TaskRuntimeState]:
    """Enumerate <worktree_base>/<task-id-dir> — per-task worktrees, no
    lane concept (dark_factory-style non-pooled layout).
    """
    worktree_base = git_ops.worktree_base
    results: list[TaskRuntimeState] = []
    if not worktree_base.is_dir():
        return results
    for entry in worktree_base.iterdir():
        if not entry.is_dir():
            continue
        task_id = _extract_task_id(entry.name)
        if task_id is None:
            continue
        meta_root = TaskArtifacts.meta_root_for(worktree_base, entry.name)
        results.append(_read_task_entry(
            task_id=task_id,
            worktree=entry,
            meta_root=meta_root,
            has_worktree=True,
            lane=None,
            lane_state=None,
            event_store=event_store,
        ))
    results.sort(key=lambda r: r.task_id)
    return results


def build_task_runtime_snapshot(*, git_ops, event_store=None) -> list[TaskRuntimeState]:
    """Per-task runtime snapshot for every active task on this host.

    Branches on ``git_ops.pool_in_use()`` (the existing pooled-vs-per-task
    discriminator): pooled projects (reify, autopilot-video, …) enumerate
    durable ``.lane-state`` records; non-pooled projects (dark_factory)
    enumerate per-task worktree dirs directly. Returns a list sorted by
    ``task_id``.
    """
    if not git_ops.pool_in_use():
        return _build_non_pooled_snapshot(git_ops=git_ops, event_store=event_store)
    return []
