"""Per-project reconciliation policies.

Each sub-module contains project-specific guardrail configuration that would
otherwise pollute shared reconciliation infrastructure.  The policy modules
expose only public names so callers don't need to cross private-name boundaries.

Cross-project registry
----------------------
``SNAPSHOT_WRITE_BLOCKED_PROJECTS`` is a :class:`frozenset` of project_ids whose
task-count snapshot write paths are blocked-by-design.  For these projects the
ABSENCE of a task-count snapshot temporal_fact edge is the CORRECT state, so
Stage 3 findings asserting the edge is missing or stale are false positives.

``is_snapshot_write_blocked(project_id)`` is the public predicate; callers
should use it rather than querying the set directly.

To register a new project: add a ``<PROJECT>_SNAPSHOT_WRITES_BLOCKED: bool = True``
constant to its policy sub-module, then join it into the frozenset below.
"""

from __future__ import annotations

# Local import kept inside the module body (not package-level __init__ star-import)
# to avoid circular-import risk if sub-modules ever grow shared-infrastructure deps.
from fused_memory.reconciliation.policies.autopilot_video import (
    AUTOPILOT_VIDEO_PROJECT_ID,
    AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED,
)

# ---------------------------------------------------------------------------
# Snapshot-write-blocked registry
# ---------------------------------------------------------------------------

#: frozenset of project_ids whose task-count snapshot write paths are
#: blocked-by-design (see each project's policy module for the rationale).
#: Built from per-project flag constants so the registry is always consistent
#: with the project's own documented posture.
SNAPSHOT_WRITE_BLOCKED_PROJECTS: frozenset[str] = frozenset(
    ([AUTOPILOT_VIDEO_PROJECT_ID] if AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED else [])
)


def is_snapshot_write_blocked(project_id: str | None) -> bool:
    """Return True iff *project_id* is in :data:`SNAPSHOT_WRITE_BLOCKED_PROJECTS`.

    ``None`` and ``''`` safely return ``False`` (fail-open).

    Args:
        project_id: Project identifier string, or ``None``.

    Returns:
        ``True`` if snapshot writes are blocked-by-design for this project;
        ``False`` for any project not explicitly registered (fail-open).
    """
    if not project_id:
        return False
    return project_id in SNAPSHOT_WRITE_BLOCKED_PROJECTS
