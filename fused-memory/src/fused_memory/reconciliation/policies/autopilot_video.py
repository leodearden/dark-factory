"""Autopilot-video-specific reconciliation guardrail policy.

All autopilot_video-specific constants and helpers live here so that
shared reconciliation infrastructure (stage2.py, task_knowledge_sync.py)
does not embed project-specific magic values.

To update the ceiling when autopilot_video grows:
1. Bump ``AUTOPILOT_VIDEO_TASK_CEILING`` below.
2. Update ``tests/fixtures/autopilot_video_max_id.json`` to ``{"max_id": <new>}``.
3. Re-evaluate whether 'task IDs > ceiling ⟹ cross-project contamination' still holds.
4. Run ``TestAutopilotVideoTaskCeilingFreshness`` to confirm both values are consistent.
"""

from __future__ import annotations

from collections.abc import Iterable

# Canonical project identifier — single source of truth across stage2 prompt
# injection and task_knowledge_sync.py's programmatic warning check.
AUTOPILOT_VIDEO_PROJECT_ID: str = 'autopilot_video'

# Max task id observed in autopilot_video/.taskmaster/tasks/tasks.json at
# task-1192 commit time.  The TestAutopilotVideoTaskCeilingFreshness test fails
# when autopilot_video grows past this value — see update instructions above.
AUTOPILOT_VIDEO_TASK_CEILING: int = 606

# Prompt fragment injected by build_stage2_system_prompt() only when
# project_id == AUTOPILOT_VIDEO_PROJECT_ID.  Interpolates the named ceiling
# constant so it remains the single source of truth.
AUTOPILOT_VIDEO_CONTAMINATION_GUARDRAIL: str = f"""\
## Cross-Project Contamination Guardrail (Pre-flight)
When any task ID in the reconciliation payload exceeds {AUTOPILOT_VIDEO_TASK_CEILING} \
(the autopilot_video task ceiling), Stage 2 must take ZERO task actions — including \
memory_hints updates, set_task_status, add_subtask, or any other task write — for the \
remainder of that cycle. This is not a matter of judgment; it is an unconditional gate.

Required behaviour when the guardrail fires:
- Take ZERO task actions (no set_task_status, no add_subtask, no update_task, \
no add_dependency, no remove_task, no resolve_ticket).
- Write a single `add_memory(category='observations_and_summaries')` to the \
autopilot_video project documenting the abort: which task IDs exceeded the ceiling \
and that Stage 2 halted without acting.
- File cross-project task suggestions in the structured report's \
`cross_project_findings` section rather than as live task writes.
- Exit immediately after writing the summary memory; do not continue to the normal \
reconciliation loop.

"""


def excessive_autopilot_video_ids(tasks: Iterable[dict]) -> list[int]:
    """Return sorted list of task IDs above AUTOPILOT_VIDEO_TASK_CEILING.

    Uses ``id_key`` from ``task_filter`` to handle dotted subtask IDs like
    '450.2' (extracts parent id 450).  Returns 0 for unparseable ids, which
    always falls below the ceiling and is ignored.  The result is deduped via
    set so a parent + its subtasks both yielding the parent id count once.

    Args:
        tasks: Iterable of task dicts (each with an ``'id'`` field).

    Returns:
        Sorted list of integer task IDs that exceed the ceiling.  Empty list
        when the guardrail condition is not met.
    """
    # Deferred import avoids a circular dependency at module load time:
    # task_filter imports nothing from reconciliation.policies, so importing
    # id_key here is safe.
    from fused_memory.reconciliation.task_filter import id_key  # noqa: PLC0415

    return sorted({
        pid
        for pid in (id_key(t) for t in tasks)
        if pid > AUTOPILOT_VIDEO_TASK_CEILING
    })
