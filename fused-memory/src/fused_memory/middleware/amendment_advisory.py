"""Decides whether a task-text write amended the prose a plan was authored against.

The plan's fingerprint is read from the written row's ``metadata.plan_text_fp``, where
the orchestrator mirrors it; never from plan.json or git, which this process cannot see.
Contract: plans/task-amendment-delivery-prd.md D10 and §9 C7.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import TERMINAL
from shared.task_text_fingerprint import TaskTextDrift, task_text_drift, task_text_fingerprint

from fused_memory.backends.task_backend_types import AmendmentAdvisory

__all__ = [
    'TASK_TEXT_FIELDS',
    'AmendmentWriter',
    'amendment_advisory',
    'text_fields_written',
]

TASK_TEXT_FIELDS: tuple[str, ...] = ('prompt', 'title', 'description', 'details')


class AmendmentWriter(StrEnum):
    UPDATE_TASK = 'update_task'
    CURATOR_COMBINE = 'curator_combine'


def text_fields_written(
    write: Mapping[str, Any], *, description_rewritten: bool = False
) -> tuple[str, ...]:
    supplied = {field for field in TASK_TEXT_FIELDS if write.get(field) is not None}
    if description_rewritten:
        supplied.add('description')
    return tuple(field for field in TASK_TEXT_FIELDS if field in supplied)


def amendment_advisory(
    task_id: str, written_task: object, fields_changed: Sequence[str]
) -> AmendmentAdvisory | None:
    if not fields_changed or not isinstance(written_task, Mapping):
        return None
    metadata = coerce_task_metadata(written_task.get('metadata'))
    if metadata is None:
        return None
    plan_fp = metadata.get('plan_text_fp')
    text_fp = task_text_fingerprint(written_task.get('description'), written_task.get('details'))
    if not isinstance(plan_fp, str) or task_text_drift(plan_fp, text_fp) is not TaskTextDrift.AMENDED:
        return None
    authored_at = metadata.get('plan_authored_at')
    return AmendmentAdvisory(
        task_id=task_id,
        plan_authored_at=authored_at if isinstance(authored_at, str) else None,
        fields_changed=list(fields_changed),
        will_replan=written_task.get('status') not in TERMINAL,
        plan_text_fp_before=plan_fp,
        task_text_fp_after=text_fp,
    )
