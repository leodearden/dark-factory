"""The fields the orchestrator writes for one plan design decision's fused-memory note.

This is the writer's half of a wire contract. The orchestrator does not depend
on fused_memory, so fused-memory owns the other half
(fused_memory/server/entities_gate.py::entities_gate), and
fused-memory/tests/server/test_orchestrator_decision_note_entities.py
validates :func:`decision_note_fields` against it.

The note names its task in its text as well as declaring it: an architect's
decision routinely names an adjacent sibling task, and the gate reads a
one-digit-apart number as a typo of the declaration unless the declared
number is written in the content too.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _is_task_number(task_id: str) -> bool:
    return task_id.isascii() and task_id.isdecimal()


def decision_note_fields(task_id: str | int, decision: Mapping[str, Any]) -> dict[str, Any]:
    """The ``add_memory`` content, and the task declaration when *task_id* is a task number.

    A non-numeric id (an eval run's ``'t'``) declares nothing, because the gate
    rejects a declaration that is not a run of digits and the write would be lost.
    The declared id is the exact-string form, which the gate keeps verbatim.
    """
    task = str(task_id)
    fields: dict[str, Any] = {
        'content': f"Decision (task {task}): {decision['decision']}\nRationale: {decision['rationale']}",
    }
    if _is_task_number(task):
        fields['entities'] = [{'kind': 'task', 'id': task}]
    return fields
