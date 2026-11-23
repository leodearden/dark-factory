"""Per-project reconciliation policies.

Each sub-module contains project-specific guardrail configuration that would
otherwise pollute shared reconciliation infrastructure.  The policy modules
expose only public names so callers don't need to cross private-name boundaries.

Contamination-ceiling-retired registry
---------------------------------------
``CONTAMINATION_CEILING_RETIRED_PROJECTS`` is a :class:`frozenset` of project_ids
whose Stage-1 task-ID "contamination ceiling" has been retired-by-design (task
2818).  For these projects the ABSENCE or staleness of a contamination-ceiling
guardrail memory is the CORRECT state — real cross-project-contamination
protection is now structural (project isolation + path-scope guard +
content-based cross_project routing), never task-ID magnitude — so Stage 3
findings asserting such a guardrail memory is missing or stale are false
positives and must be suppressed.

``is_contamination_ceiling_retired(project_id)`` is the public predicate; callers
should use it rather than querying the set directly.  It is consumed by
``filter_contamination_ceiling_findings`` (``flag_dedup.py``) as the load-bearing
code-side backstop.  The registry is intentionally a single entry today (only
autopilot_video); it uses the full three-part shape rather than a hardcoded
``project ==`` check so a future second project is a one-line append.

To register a new project: add a ``<PROJECT>_CONTAMINATION_CEILING_RETIRED: bool =
True`` constant to its policy sub-module, import it in the
``_PROJECT_CEILING_RETIRED_FLAGS`` list below, and append
``(<PROJECT_ID>, <PROJECT>_CONTAMINATION_CEILING_RETIRED)`` to the list.  The
frozenset is derived automatically — no other edits required.
"""

from __future__ import annotations

# Local import kept inside the module body (not package-level __init__ star-import)
# to avoid circular-import risk if sub-modules ever grow shared-infrastructure deps.
from fused_memory.reconciliation.policies.autopilot_video import (
    AUTOPILOT_VIDEO_CONTAMINATION_CEILING_RETIRED,
    AUTOPILOT_VIDEO_PROJECT_ID,
)

# ---------------------------------------------------------------------------
# Contamination-ceiling-retired registry
# ---------------------------------------------------------------------------

# Each entry is (project_id, retired_flag).  The frozenset is derived from this
# list so adding a new project requires only a single append here, not editing a
# conditional expression.  The registry is intentionally a single entry today
# (only autopilot_video's Stage-1 task-ID contamination ceiling has been retired,
# per task 2818); a future second project is a one-line append, not a refactor.
# The retired_flag gate ensures that a project whose flag is later set to False
# is automatically removed from the set.
_PROJECT_CEILING_RETIRED_FLAGS: list[tuple[str, bool]] = [
    (AUTOPILOT_VIDEO_PROJECT_ID, AUTOPILOT_VIDEO_CONTAMINATION_CEILING_RETIRED),
]

#: frozenset of project_ids whose Stage-1 task-ID "contamination ceiling" is
#: retired-by-design (see each project's policy module for the rationale).  For
#: these projects the ABSENCE or staleness of a contamination-ceiling guardrail
#: memory is the CORRECT state, so Stage 3 findings asserting the memory is
#: missing or stale are false positives.  Derived from
#: :data:`_PROJECT_CEILING_RETIRED_FLAGS` so the registry is always consistent
#: with each project's own documented posture.
CONTAMINATION_CEILING_RETIRED_PROJECTS: frozenset[str] = frozenset(
    pid for pid, retired in _PROJECT_CEILING_RETIRED_FLAGS if retired
)


def is_contamination_ceiling_retired(project_id: str | None) -> bool:
    """Return True iff *project_id* is in :data:`CONTAMINATION_CEILING_RETIRED_PROJECTS`.

    ``True`` means the Stage-1 task-ID contamination ceiling is retired-by-design
    for this project (task 2818): real cross-project-contamination protection is
    now structural (project isolation + path-scope guard + content-based
    cross_project routing), never task-ID magnitude.  The absence/staleness of a
    contamination-ceiling guardrail memory is therefore the correct, non-actionable
    state, and Stage 3 "missing/stale guardrail" findings must be suppressed (see
    ``filter_contamination_ceiling_findings`` in ``flag_dedup.py``).

    ``None`` and ``''`` safely return ``False`` (fail-open).

    Args:
        project_id: Project identifier string, or ``None``.

    Returns:
        ``True`` if the contamination ceiling is retired-by-design for this
        project; ``False`` for any project not explicitly registered (fail-open).
    """
    if not project_id:
        return False
    return project_id in CONTAMINATION_CEILING_RETIRED_PROJECTS
