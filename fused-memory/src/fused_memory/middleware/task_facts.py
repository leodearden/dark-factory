"""One batched read of live task facts; the contract is plans/live-task-facts-at-recall-prd.md C1."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from shared.task_statuses import TERMINAL, TaskStatus

from fused_memory.backends.task_backend_types import DependencyFact, TaskFactsRow

logger = logging.getLogger(__name__)

__all__ = ['Lookup', 'TaskFacts', 'TaskFactsReader', 'TaskRef', 'read_task_facts']


@dataclass(frozen=True)
class TaskRef:
    project_id: str
    task_id: str

    def __post_init__(self) -> None:
        if not self.project_id:
            raise ValueError(f'TaskRef.project_id must be non-empty, got {self.project_id!r}')
        if not self.task_id:
            raise ValueError(f'TaskRef.task_id must be non-empty, got {self.task_id!r}')


class Lookup(StrEnum):
    FOUND = 'found'
    NO_SUCH_TASK = 'no_such_task'
    COULD_NOT_CHECK = 'could_not_check'


WaitsOn = tuple[tuple[TaskRef, str], ...]


@dataclass(frozen=True)
class TaskFacts:
    ref: TaskRef
    lookup: Lookup
    reason: str | None
    status: str | None
    title: str | None
    waits_on: WaitsOn
    finished_how: str | None
    finished_commit: str | None

    def __post_init__(self) -> None:
        broken = _broken_invariant(self)
        if broken is not None:
            raise ValueError(f'TaskFacts for {self.ref!r} breaks an invariant: {broken}')

    @classmethod
    def found(
        cls, ref: TaskRef, *, status: str, title: str | None, waits_on: WaitsOn = (),
        finished_how: str | None = None, finished_commit: str | None = None,
    ) -> TaskFacts:
        return cls(
            ref=ref, lookup=Lookup.FOUND, reason=None, status=status, title=title,
            waits_on=waits_on, finished_how=finished_how, finished_commit=finished_commit,
        )

    @classmethod
    def no_such_task(cls, ref: TaskRef) -> TaskFacts:
        return cls(
            ref=ref, lookup=Lookup.NO_SUCH_TASK, reason=None, status=None, title=None,
            waits_on=(), finished_how=None, finished_commit=None,
        )

    @classmethod
    def could_not_check(cls, ref: TaskRef, reason: str) -> TaskFacts:
        return cls(
            ref=ref, lookup=Lookup.COULD_NOT_CHECK, reason=reason, status=None, title=None,
            waits_on=(), finished_how=None, finished_commit=None,
        )


def _broken_invariant(facts: TaskFacts) -> str | None:
    found = facts.lookup is Lookup.FOUND
    finished = (facts.finished_how, facts.finished_commit)
    if (facts.reason is not None) != (facts.lookup is Lookup.COULD_NOT_CHECK):
        return f'reason is set iff COULD_NOT_CHECK (lookup={facts.lookup}, reason={facts.reason!r})'
    if (facts.status is not None) != found:
        return f'status is set iff FOUND (lookup={facts.lookup}, status={facts.status!r})'
    if not found and (facts.title is not None or facts.waits_on or finished != (None, None)):
        return (
            f'a {facts.lookup} entry has no title, waits_on or finished_* '
            f'(title={facts.title!r}, waits_on={facts.waits_on!r}, finished={finished!r})'
        )
    if facts.status != TaskStatus.DONE and finished != (None, None):
        return f'finished_* are None unless status is done (status={facts.status!r}, finished={finished!r})'
    if facts.status in TERMINAL and facts.waits_on:
        return f'waits_on is () for a terminal status (status={facts.status!r}, waits_on={facts.waits_on!r})'
    return None


class TaskFactsReader(Protocol):
    """The slice of ``TaskBackendProtocol`` that :func:`read_task_facts` needs."""

    async def get_task_facts(
        self, ids: Sequence[str], project_root: str,
    ) -> Mapping[str, TaskFactsRow]: ...


async def read_task_facts(
    refs: Collection[TaskRef], *, reader: TaskFactsReader | None, known_projects: Mapping[str, str],
) -> dict[TaskRef, TaskFacts]:
    """Facts for every ref in *refs*, from one read per registered project.

    ``reader=None`` means no task store is configured. A project's failure
    is contained to that project's refs, as ``COULD_NOT_CHECK``.
    """
    refs_by_project: dict[str, dict[TaskRef, None]] = {}
    for ref in refs:
        refs_by_project.setdefault(ref.project_id, {})[ref] = None
    facts: dict[TaskRef, TaskFacts] = {}
    for project_id, project_refs in refs_by_project.items():
        facts.update(await _read_project(
            project_id, list(project_refs), reader=reader, root=known_projects.get(project_id),
        ))
    return facts


async def _read_project(
    project_id: str, refs: list[TaskRef], *, reader: TaskFactsReader | None, root: str | None,
) -> dict[TaskRef, TaskFacts]:
    if not root:
        return _could_not_check_all(refs, 'project not registered')
    if reader is None:
        return _could_not_check_all(refs, 'no task store configured')
    try:
        rows = await reader.get_task_facts(
            ids=sorted({ref.task_id for ref in refs}), project_root=root,
        )
    except Exception as exc:
        logger.warning(
            'read_task_facts: task store read failed for project %r (%d refs)',
            project_id, len(refs), exc_info=True,
        )
        return _could_not_check_all(refs, f'task store read failed: {type(exc).__name__}: {exc}')
    if not isinstance(rows, Mapping):
        return _could_not_check_all(refs, f'task store returned a {type(rows).__name__}, not a mapping')
    return {ref: _facts_from_row(ref, rows.get(ref.task_id)) for ref in refs}


def _could_not_check_all(refs: Iterable[TaskRef], reason: str) -> dict[TaskRef, TaskFacts]:
    return {ref: TaskFacts.could_not_check(ref, reason) for ref in refs}


def _facts_from_row(ref: TaskRef, row: TaskFactsRow | None) -> TaskFacts:
    if row is None:
        return TaskFacts.no_such_task(ref)
    status = row.get('status') if isinstance(row, Mapping) else None
    if not isinstance(status, str):
        return TaskFacts.could_not_check(ref, 'task store returned no usable status')
    done = status == TaskStatus.DONE
    return TaskFacts.found(
        ref, status=status, title=row.get('title'),
        waits_on=() if status in TERMINAL else _waits_on(ref.project_id, row.get('dependencies', ())),
        finished_how=row.get('done_provenance_kind') if done else None,
        finished_commit=row.get('done_provenance_commit') if done else None,
    )


def _waits_on(project_id: str, dependencies: Iterable[DependencyFact]) -> WaitsOn:
    return tuple(
        (TaskRef(project_id, dep['id']), Lookup.NO_SUCH_TASK.value if dep['status'] is None else dep['status'])
        for dep in dependencies
        if dep['status'] not in TERMINAL
    )
