"""Bind recalled results to tasks and render their live facts: plans/live-task-facts-at-recall-prd.md C2 and C4."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fused_memory.memory_metadata import is_completion_record
from fused_memory.middleware.task_facts import TaskRef
from fused_memory.models.memory import MemoryResult
from fused_memory.utils.canonical_labels import Referent, parse_node_name, scan_content
from fused_memory.utils.validation import canonicalize_project_id

__all__ = [
    'MAX_RESULTS_SCANNED',
    'MAX_TASKS',
    'Binding',
    'BoundTask',
    'TaskBindings',
    'bind_tasks',
]

MAX_TASKS = 40
MAX_RESULTS_SCANNED = 100


class Binding(StrEnum):
    STRUCTURAL = 'structural'
    MENTIONED = 'mentioned'


@dataclass(frozen=True)
class BoundTask:
    ref: TaskRef
    binding: Binding
    result_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        broken = _bound_task_defect(self)
        if broken is not None:
            raise ValueError(f'BoundTask for {self.ref!r} breaks an invariant: {broken}')


def _bound_task_defect(bound: BoundTask) -> str | None:
    if not bound.result_ids:
        return f'result_ids is non-empty (got {bound.result_ids!r})'
    return None


@dataclass(frozen=True)
class TaskBindings:
    tasks: tuple[BoundTask, ...]
    ambiguous: tuple[str, ...]
    omitted_tasks: int
    results_not_scanned: int

    def __post_init__(self) -> None:
        broken = _bindings_defect(self)
        if broken is not None:
            raise ValueError(f'TaskBindings breaks an invariant: {broken}')


def _bindings_defect(bindings: TaskBindings) -> str | None:
    count = len(bindings.tasks)
    if count > MAX_TASKS:
        return f'at most MAX_TASKS={MAX_TASKS} tasks (got {count})'
    if bindings.omitted_tasks < 0:
        return f'omitted_tasks is not negative (got {bindings.omitted_tasks})'
    if bindings.results_not_scanned < 0:
        return f'results_not_scanned is not negative (got {bindings.results_not_scanned})'
    if bindings.omitted_tasks and count < MAX_TASKS:
        return (
            f'omitted_tasks is 0 unless MAX_TASKS={MAX_TASKS} tasks are kept '
            f'(omitted_tasks={bindings.omitted_tasks}, tasks={count})'
        )
    duplicates = [ref for ref, seen in Counter(bound.ref for bound in bindings.tasks).items() if seen > 1]
    if duplicates:
        return f'each ref appears once (duplicates: {duplicates!r})'
    kinds = [bound.binding for bound in bindings.tasks]
    if kinds != sorted(kinds, key=_is_mentioned):
        return f'STRUCTURAL entries come before MENTIONED ones (got {[kind.value for kind in kinds]})'
    return None


def _is_mentioned(binding: Binding) -> bool:
    return binding is Binding.MENTIONED


_Link = tuple[TaskRef, Binding]


def bind_tasks(
    results: Sequence[MemoryResult], *, project_id: str, known_projects: Mapping[str, str],
) -> TaskBindings:
    """The tasks the first MAX_RESULTS_SCANNED *results* are about or mention:
    structural first, then by first result rank, the first MAX_TASKS kept."""
    project = canonicalize_project_id(project_id)
    links: list[tuple[str, _Link]] = []
    ambiguous: dict[str, None] = {}
    for result in results[:MAX_RESULTS_SCANNED]:
        scan = scan_content(result.content, group_id=project, known_project_ids=known_projects)
        links.extend((result.id, link) for link in _result_links(result, scan.refs, project))
        ambiguous.update(dict.fromkeys(referent.number for referent in scan.ambiguous))
    tasks = _merge(links)
    return TaskBindings(
        tasks=tuple(tasks[:MAX_TASKS]),
        ambiguous=tuple(ambiguous),
        omitted_tasks=max(0, len(tasks) - MAX_TASKS),
        results_not_scanned=max(0, len(results) - MAX_RESULTS_SCANNED),
    )


def _result_links(result: MemoryResult, mentions: Sequence[Referent], project: str) -> Iterator[_Link]:
    for name in result.entities:
        referent = parse_node_name(name)
        if referent is not None:
            yield _task_ref(referent, project), Binding.STRUCTURAL
    completed = _completion_task_id(result.metadata)
    if completed is not None:
        yield TaskRef(project, completed), Binding.STRUCTURAL
    for referent in mentions:
        yield _task_ref(referent, project), Binding.MENTIONED


def _task_ref(referent: Referent, project: str) -> TaskRef:
    return TaskRef(referent.project_id or project, referent.number)


def _completion_task_id(metadata: Mapping[str, Any]) -> str | None:
    if not is_completion_record(metadata):
        return None
    task_id = metadata.get('task_id')
    if isinstance(task_id, int) and not isinstance(task_id, bool):
        return str(task_id)
    return task_id if isinstance(task_id, str) and task_id else None


def _merge(links: Sequence[tuple[str, _Link]]) -> list[BoundTask]:
    """One entry per ref: STRUCTURAL if any link says so, result ids de-duplicated in rank order."""
    result_ids: dict[TaskRef, dict[str, None]] = {}
    structural: set[TaskRef] = set()
    for result_id, (ref, binding) in links:
        result_ids.setdefault(ref, {})[result_id] = None
        if binding is Binding.STRUCTURAL:
            structural.add(ref)
    ordered = sorted(result_ids, key=lambda ref: ref not in structural)
    return [
        BoundTask(
            ref, Binding.STRUCTURAL if ref in structural else Binding.MENTIONED, tuple(result_ids[ref]),
        )
        for ref in ordered
    ]
