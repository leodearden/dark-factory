"""An in-memory ``legibility.task_store.TaskStore`` shared by the disposition
and census suites.

A plain module rather than a conftest fixture, the same convention as
``legibility_carrier_fixtures``: conftest appends this directory to
``sys.path``, so the test files import it by bare name.

It has no write method, so nothing driven through it can write task metadata
(plans/census-incremental-prd.md T4).
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from legibility import task_store

StoreAnswer = Sequence[Any] | task_store.StoreUnreachable


def _answer(answer: Any) -> Any:
    if isinstance(answer, task_store.StoreUnreachable):
        raise answer
    return answer


class FakeTaskStore:
    """*metadata_hits* maps ``(key, value)`` to the TaskHits that lookup finds;
    *search_hits* maps a needle to the SearchHits of every query containing it;
    *resolve_outcomes* maps a ticket to its TicketOutcome. Any of those values
    may instead be a StoreUnreachable, which the call raises. A tool named in
    *unreachable* raises on every call. An unknown ticket resolves to a new
    task, numbered from *first_created_id*. *on_resolve* runs on every
    resolve_ticket call, after it is recorded."""

    def __init__(
        self,
        *,
        tasks: Mapping[int, task_store.TaskRecord] | None = None,
        metadata_hits: Mapping[tuple[str, str | int], StoreAnswer] | None = None,
        search_hits: Mapping[str, StoreAnswer] | None = None,
        resolve_outcomes: Mapping[str, task_store.TicketOutcome | task_store.StoreUnreachable] | None = None,
        unreachable: Sequence[str] = (),
        first_created_id: int = 1000,
        on_resolve: Callable[[], None] | None = None,
    ) -> None:
        self.tasks = dict(tasks or {})
        self.metadata_hits = dict(metadata_hits or {})
        self.search_hits = dict(search_hits or {})
        self.resolve_outcomes = dict(resolve_outcomes or {})
        self.unreachable = frozenset(unreachable)
        self._created_ids = itertools.count(first_created_id)
        self._on_resolve = on_resolve
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, tool: str, **arguments: Any) -> None:
        self.calls.append((tool, arguments))
        if tool in self.unreachable:
            raise task_store.StoreUnreachable(f"{tool}: the fake store is unreachable")

    def tools_called(self) -> list[str]:
        return [tool for tool, _ in self.calls]

    def arguments_of(self, tool: str) -> list[dict[str, Any]]:
        return [arguments for called, arguments in self.calls if called == tool]

    def find_tasks_by_metadata(
        self, project_root: str, *, key: str, value: str | int,
    ) -> tuple[task_store.TaskHit, ...]:
        self._record("find_tasks_by_metadata", project_root=project_root, key=key, value=value)
        return tuple(_answer(self.metadata_hits.get((key, value), ())))

    def search_tasks(
        self, project_root: str, *, query: str, score_threshold: float,
    ) -> tuple[task_store.SearchHit, ...]:
        self._record("search_tasks", project_root=project_root, query=query, score_threshold=score_threshold)
        hits: list[task_store.SearchHit] = []
        for needle, answer in self.search_hits.items():
            if needle in query:
                hits.extend(_answer(answer))
        return tuple(hit for hit in hits if hit.score >= score_threshold)

    def get_task(self, project_root: str, task_id: int) -> task_store.TaskRecord:
        self._record("get_task", project_root=project_root, task_id=task_id)
        if task_id not in self.tasks:
            raise task_store.StoreUnreachable(f"get_task: NotFound: task {task_id} does not exist")
        return self.tasks[task_id]

    def resolve_ticket(self, project_root: str, ticket: str, *, timeout_secs: float) -> task_store.TicketOutcome:
        self._record("resolve_ticket", project_root=project_root, ticket=ticket, timeout_secs=timeout_secs)
        if self._on_resolve is not None:
            self._on_resolve()
        if ticket not in self.resolve_outcomes:
            self.resolve_outcomes[ticket] = task_store.TicketOutcome(
                task_store.TicketStatus.CREATED, next(self._created_ids), None,
            )
        return _answer(self.resolve_outcomes[ticket])
