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
constant to its policy sub-module, import it in the ``_PROJECT_SNAPSHOT_FLAGS``
list below, and append ``(<PROJECT_ID>, <PROJECT>_SNAPSHOT_WRITES_BLOCKED)`` to
the list.  The frozenset is derived automatically — no other edits required.
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

# Each entry is (project_id, blocked_flag).  The frozenset is derived from this
# list so adding a new project requires only a single append here, not editing a
# conditional expression.  The blocked_flag gate ensures that a project whose flag
# is later set to False is automatically removed from the set.
_PROJECT_SNAPSHOT_FLAGS: list[tuple[str, bool]] = [
    (AUTOPILOT_VIDEO_PROJECT_ID, AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED),
]

#: frozenset of project_ids whose task-count snapshot write paths are
#: blocked-by-design (see each project's policy module for the rationale).
#: Derived from :data:`_PROJECT_SNAPSHOT_FLAGS` so the registry is always
#: consistent with each project's own documented posture.
SNAPSHOT_WRITE_BLOCKED_PROJECTS: frozenset[str] = frozenset(
    pid for pid, blocked in _PROJECT_SNAPSHOT_FLAGS if blocked
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
