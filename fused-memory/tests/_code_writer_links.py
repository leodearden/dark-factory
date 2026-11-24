"""Read one recorded memory write back as task links, for the C5 per-writer tests of plans/live-task-facts-at-recall-prd.md.

A structural link is whatever ``middleware/recall_annotation.py::bind_tasks``
binds STRUCTURAL over the written record, together with the task referents
``utils/referent_resolution.py::resolve_referents`` resolves from the write's
``declared_referents``. Both are read through the real code, so these tests
never restate the binding rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.testing_code_writer_contract import task_status_words_in

from fused_memory.middleware.recall_annotation import Binding, bind_tasks
from fused_memory.middleware.task_facts import TaskRef
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.utils.referent_resolution import resolve_referents
from fused_memory.utils.validation import canonicalize_project_id


def subject_task(project_id: str, task_id: str | int) -> TaskRef:
    """The TaskRef a write about ``task_id`` in ``project_id`` must link."""
    return TaskRef(canonicalize_project_id(project_id), str(task_id))


def _declared_task_refs(call_kwargs: Mapping[str, Any], project: str) -> set[TaskRef]:
    declared = call_kwargs.get('declared_referents')
    if declared is None:
        return set()
    resolution = resolve_referents(
        declared=declared,
        metadata=call_kwargs.get('metadata'),
        content=call_kwargs.get('content'),
        group_id=project,
    )
    assert resolution.source == 'declared', resolution
    return {
        TaskRef(referent.project_id or project, referent.number)
        for referent in resolution.referents
        if referent.kind == 'task'
    }


def structural_task_links(call_kwargs: Mapping[str, Any], *, project_id: str) -> frozenset[TaskRef]:
    """The tasks one recorded write links structurally, read from its keyword arguments."""
    project = canonicalize_project_id(project_id)
    written = MemoryResult(
        id='written',
        content=call_kwargs.get('content') or '',
        source_store=SourceStore.mem0,
        metadata=dict(call_kwargs.get('metadata') or {}),
    )
    bindings = bind_tasks([written], project_id=project, known_projects={})
    bound = {task.ref for task in bindings.tasks if task.binding is Binding.STRUCTURAL}
    return frozenset(bound | _declared_task_refs(call_kwargs, project))


def assert_writes_c5(
    call_kwargs: Mapping[str, Any], *, project_id: str, subject: TaskRef | frozenset[TaskRef] | None,
) -> None:
    """The write links exactly its subject task(s), or none, and its text names no task status."""
    if subject is None:
        expected: frozenset[TaskRef] = frozenset()
    elif isinstance(subject, TaskRef):
        expected = frozenset({subject})
    else:
        expected = subject
    assert structural_task_links(call_kwargs, project_id=project_id) == expected, call_kwargs
    content = call_kwargs.get('content') or ''
    assert task_status_words_in(content) == (), content
