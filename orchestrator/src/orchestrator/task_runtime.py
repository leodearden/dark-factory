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
from dataclasses import dataclass

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
