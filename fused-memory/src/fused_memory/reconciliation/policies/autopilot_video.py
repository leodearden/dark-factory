"""Autopilot-video-specific reconciliation guardrail policy.

All autopilot_video-specific constants and helpers live here so that
shared reconciliation infrastructure (stage2.py, task_knowledge_sync.py)
does not embed project-specific magic values.
"""

from __future__ import annotations

from collections.abc import Iterable

# Canonical project identifier — single source of truth across stage2 prompt
# injection and task_knowledge_sync.py's programmatic warning check.
AUTOPILOT_VIDEO_PROJECT_ID: str = 'autopilot_video'

# Legacy ceiling constant — retained for backward-compat imports in
# task_knowledge_sync.py until step-4 removes the code gate entirely.
# The prompt guardrail (AUTOPILOT_VIDEO_CONTAMINATION_GUARDRAIL) no longer
# references this value; it has been replaced by content-based detection.
AUTOPILOT_VIDEO_TASK_CEILING: int = 606

# Prompt fragment injected by build_stage2_system_prompt() only when
# project_id == AUTOPILOT_VIDEO_PROJECT_ID.  Content-based: cross-project
# contamination is detected by cited file paths/modules belonging to another
# repo, NOT by numeric task-ID magnitude (high IDs are normal project growth).
AUTOPILOT_VIDEO_CONTAMINATION_GUARDRAIL: str = """\
## Cross-Project Contamination Guardrail (Pre-flight)
There is NO task-ID ceiling for autopilot_video. High task IDs (e.g. 607, 700, \
1000+) are normal project growth and are never evidence of cross-project contamination \
on their own. Do NOT abort or suppress task actions based on task-ID magnitude alone.

Cross-project contamination is identified by **content**: cited file paths or modules \
that belong to a different repository. The path-scope guard (``DarkFactoryPathScopeViolation``) \
already rejects mis-routed task writes at the API level, and the \
**## Cross-Project Routing** section in this prompt explains how to handle findings \
whose scope belongs to another project.

When a finding appears to belong to another project:
- If the target project is listed in the payload's "Known Projects" section, \
route the task there via that project's ``project_root``; the path-scope guard will \
validate the routing.
- If no matching project is in "Known Projects", add a ``cross_project_findings`` \
entry to your structured report (with ``summary``, ``target_project_hint``, and \
``evidence``) so the operator can route it manually. Do NOT suppress normal \
reconciliation work for this project.

"""


def excessive_autopilot_video_ids(tasks: Iterable[dict]) -> list[int]:
    """Return sorted list of task IDs above AUTOPILOT_VIDEO_TASK_CEILING.

    Legacy helper — retained for backward-compat import in task_knowledge_sync.py
    until step-4 removes the code gate.  After step-4 this function is deleted.

    Args:
        tasks: Iterable of task dicts (each with an ``'id'`` field).

    Returns:
        Sorted list of integer task IDs that exceed the ceiling.  Empty list
        when the guardrail condition is not met.
    """
    # Deferred import avoids a circular dependency at module load time.
    from fused_memory.reconciliation.task_filter import id_key  # noqa: PLC0415

    return sorted({
        pid
        for pid in (id_key(t) for t in tasks)
        if pid > AUTOPILOT_VIDEO_TASK_CEILING
    })
