"""LaneLifecycle — the single durable-record writer for warm-lane state.

PRD W11 (``plans/worktree-lane-lifecycle-prd.md``), task alpha (mechanism 1):
gives the warm-lane pool one authoritative durable state record per lane,
written through one writer. ``WarmLanePool``'s in-memory map becomes a cache
of these records (consumed by task gamma, the GitOps acquire/release writer,
and task delta, the Harness crash-recovery reader).

Deliberately a SEPARATE module from ``warm_lane_pool.py``: that module is a
pure, git/escalation-free in-memory state machine (its own docstring: "No git
I/O"). This module owns file I/O, escalation filing, and async quarantine —
mixing those in would break the pool's purity and its existing 2-value
``LaneState{FREE,ASSIGNED}`` (PRD Open Q3; resolved: new module).
"""

from __future__ import annotations

from enum import Enum

# Escalation sentinel role for illegal-transition escalations. Matches the
# 'harness-' prefix in escalation.server._HARNESS_SENTINEL_ROLE_PREFIXES so
# the born-at-L2 record is exempt from the agent-role downgrade gate and
# stays L2 (routes straight to a human). PRD Open Q4.
ESCALATION_SENTINEL_ROLE = 'harness-lane-lifecycle'


class LaneState(Enum):
    """Lifecycle states for a single warm lane (PRD W11 Contract)."""

    SEED = 'seed'
    REGISTERED = 'registered'
    ASSIGNED = 'assigned'
    IN_USE = 'in_use'
    RELEASED = 'released'
    QUARANTINED = 'quarantined'


# Legal (from, to) edges. ``from`` is ``None`` for the pre-record "—" origin
# (a lane with no durable record yet). Built as the explicit table from the
# PRD's "Lane state transition table" plus a comprehension adding
# (state, QUARANTINED) for every state INCLUDING the None origin (recovery
# divergence can quarantine a lane at any point, even before a record exists).
LEGAL_TRANSITIONS: frozenset[tuple[LaneState | None, LaneState]] = frozenset(
    {
        (None, LaneState.SEED),
        (LaneState.SEED, LaneState.REGISTERED),
        (LaneState.REGISTERED, LaneState.ASSIGNED),
        (LaneState.RELEASED, LaneState.ASSIGNED),
        (LaneState.ASSIGNED, LaneState.IN_USE),
        (LaneState.IN_USE, LaneState.RELEASED),
        (LaneState.ASSIGNED, LaneState.RELEASED),
    }
    | {(origin, LaneState.QUARANTINED) for origin in [*list(LaneState), None]}
)


class IllegalLaneTransition(Exception):
    """Raised when a caller attempts a (from, to) edge not in LEGAL_TRANSITIONS.

    Never silent-heal (PRD I2): the durable record is left unchanged when this
    is raised.
    """
