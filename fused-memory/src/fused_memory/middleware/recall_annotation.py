"""Bind recalled results to tasks and render their live facts: plans/live-task-facts-at-recall-prd.md C2 and C4."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fused_memory.memory_metadata import is_completion_record
from fused_memory.middleware.task_facts import Lookup, TaskFacts, TaskRef
from fused_memory.models.memory import MemoryResult
from fused_memory.utils.canonical_labels import Referent, parse_node_name, scan_content
from fused_memory.utils.clip_disclosure import fit_at_word
from fused_memory.utils.validation import canonicalize_project_id

__all__ = [
    'MAX_RESULTS_SCANNED',
    'MAX_TASKS',
    'Binding',
    'BoundTask',
    'FactsUnavailable',
    'TaskBindings',
    'bind_tasks',
    'render_task_facts',
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
    return _ambiguous_defect(bindings)


def _ambiguous_defect(bindings: TaskBindings) -> str | None:
    if len(set(bindings.ambiguous)) != len(bindings.ambiguous):
        return f'each ambiguous number appears once (got {bindings.ambiguous!r})'
    looked_up = sorted(set(bindings.ambiguous) & _bound_numbers(bindings.tasks))
    if looked_up:
        return f'an ambiguous number is never a bound task_id (got {looked_up!r})'
    return None


def _bound_numbers(tasks: Sequence[BoundTask]) -> set[str]:
    return {bound.ref.task_id for bound in tasks}


def _is_mentioned(binding: Binding) -> bool:
    return binding is Binding.MENTIONED


_Link = tuple[TaskRef, Binding]


def bind_tasks(
    results: Sequence[MemoryResult], *, project_id: str, known_projects: Mapping[str, str],
) -> TaskBindings:
    """The tasks the first MAX_RESULTS_SCANNED *results* are about or mention:
    structural first, then by first result rank, the first MAX_TASKS kept.

    *project_id* always counts as registered, so an empty or unusable
    *known_projects* narrows the scan instead of letting any text qualifier
    ('localhost:6379') mint a task.
    """
    project = canonicalize_project_id(project_id)
    registry = (*known_projects, project)
    links: list[tuple[str, _Link]] = []
    ambiguous: dict[str, None] = {}
    for result in results[:MAX_RESULTS_SCANNED]:
        scan = scan_content(result.content, group_id=project, known_project_ids=registry)
        links.extend((result.id, link) for link in _result_links(result, scan.refs, project))
        ambiguous.update(dict.fromkeys(referent.number for referent in scan.ambiguous))
    tasks = _merge(links)
    kept = tasks[:MAX_TASKS]
    looked_up = _bound_numbers(kept)
    return TaskBindings(
        tasks=tuple(kept),
        ambiguous=tuple(number for number in ambiguous if number not in looked_up),
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


@dataclass(frozen=True)
class FactsUnavailable:
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(f'FactsUnavailable.reason must be non-empty, got {self.reason!r}')


_TITLE_LIMIT = 60
_ITEM_JOIN = ' · '


def render_task_facts(
    bindings: TaskBindings,
    facts: Mapping[TaskRef, TaskFacts] | FactsUnavailable,
    *,
    project_id: str,
    read_at: datetime,
) -> str:
    """The C4 briefing block for *bindings*, or '' when there is nothing to report."""
    if read_at.tzinfo is None or read_at.utcoffset() is None:
        raise ValueError(f'render_task_facts needs an aware read_at, got {read_at!r}')
    if isinstance(facts, FactsUnavailable):
        return f'**Live task status** could not be read from the task store: {facts.reason}'
    missing = [bound.ref for bound in bindings.tasks if bound.ref not in facts]
    if missing:
        raise ValueError(f'render_task_facts has no facts for bound refs {missing!r}')
    project = canonicalize_project_id(project_id)
    lines = [
        *_group_lines(bindings.tasks, facts, project),
        *_trailing_lines(bindings),
    ]
    if not lines:
        return ''
    return '\n'.join([_header(read_at), *lines])


def _header(read_at: datetime) -> str:
    return (
        f'**Live task status** (read from the task store at {read_at.astimezone(UTC):%H:%M}Z; '
        'status words inside the memory text below are history)'
    )


def _group_lines(
    tasks: Sequence[BoundTask], facts: Mapping[TaskRef, TaskFacts], project: str,
) -> Iterator[str]:
    by_status: dict[str, list[str]] = {}
    mentioned: list[str] = []
    no_such: list[str] = []
    unchecked: list[str] = []
    for bound in tasks:
        fact = facts[bound.ref]
        label = _label(bound.ref, project)
        if fact.lookup is Lookup.NO_SUCH_TASK:
            no_such.append(label)
        elif fact.lookup is Lookup.COULD_NOT_CHECK:
            unchecked.append(f'{label} ({fact.reason})')
        elif bound.binding is Binding.STRUCTURAL:
            by_status.setdefault(str(fact.status), []).append(_found_item(fact, project))
        else:
            mentioned.append(f'{_titled(label, fact.title)} ({fact.status})')
    for status, items in by_status.items():
        yield _line(status, items)
    for heading, items in (
        ('mentioned in the text only, check the title', mentioned),
        ('no such task', no_such),
        ('could not check', unchecked),
    ):
        if items:
            yield _line(heading, items)


def _found_item(fact: TaskFacts, project: str) -> str:
    item = _titled(_label(fact.ref, project), fact.title) + _finished(fact)
    if fact.waits_on:
        waits = ', '.join(f'{_label(ref, project)} ({status})' for ref, status in fact.waits_on)
        item += f', waits on {waits}'
    return item


def _finished(fact: TaskFacts) -> str:
    parts: list[str] = []
    if fact.finished_how:
        parts.append(fact.finished_how.replace('_', ' '))
    if fact.finished_commit:
        parts.append(fact.finished_commit[:10])
    return f' ({" ".join(parts)})' if parts else ''


def _titled(label: str, title: str | None) -> str:
    if title is None:
        return label
    one_line = ' '.join(title.split()).replace('"', "'")
    return f'{label} "{fit_at_word(one_line, _TITLE_LIMIT)}"'


def _label(ref: TaskRef, project: str) -> str:
    return ref.task_id if ref.project_id == project else f'{ref.project_id}:{ref.task_id}'


def _line(heading: str, items: Sequence[str]) -> str:
    return f'- {heading}: {_ITEM_JOIN.join(items)}'


def _trailing_lines(bindings: TaskBindings) -> Iterator[str]:
    if bindings.ambiguous:
        yield f'- ambiguous, not looked up: {", ".join(bindings.ambiguous)}'
    overflow: list[str] = []
    if bindings.omitted_tasks:
        overflow.append(_count(bindings.omitted_tasks, 'more task', 'more tasks'))
    if bindings.results_not_scanned:
        overflow.append(_count(bindings.results_not_scanned, 'result not scanned', 'results not scanned'))
    if overflow:
        yield f'- not shown: {"; ".join(overflow)}'


def _count(n: int, singular: str, plural: str) -> str:
    return f'{n} {singular if n == 1 else plural}'
