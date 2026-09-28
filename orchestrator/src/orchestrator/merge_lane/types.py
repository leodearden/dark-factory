"""Value types that cross the merge lane's ports.

PRD ``plans/merge-lane-quality-prd.md`` task ζ1. Each wraps exactly what the
collaborator behind a port produces or consumes today; nothing here is new
information. Frozen, so a value handed across a port is never mutated by the
other side.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DiskGuardOutcome:
    """What the pre-verify disk guard decided.

    ``reason`` is the blocked reason when free space on the merge worktree's
    volume stayed below the floor after pruning stale merge worktrees, and
    ``None`` to proceed with the verify.
    """

    reason: str | None


@dataclasses.dataclass(frozen=True)
class EscalationRecord:
    """An escalation as the merge lane files it.

    Everything ``escalation.models.Escalation`` needs except the id, which
    the queue mints at filing time.
    """

    task_id: str
    agent_role: str
    severity: str
    level: int
    category: str
    summary: str
    detail: str
    suggested_action: str
