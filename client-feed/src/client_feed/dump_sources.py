"""Every store source of one policy's root, read once, so an operator sees exactly what the publisher would read."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from pydantic import TypeAdapter

from client_feed.contract import NotRecorded, ReviewCounts
from client_feed.policy import Policy
from client_feed.sources.events import (
    EventBatch,
    FactoryStatus,
    LandedCommit,
    read_events,
    read_factory_status,
    read_open_escalations,
    read_review_counts,
)
from client_feed.sources.git import main_head
from client_feed.sources.prds import PrdsRead, read_prds
from client_feed.sources.tasks import TasksRead, read_tasks


@dataclass(frozen=True)
class SourcesDump:
    """main's head, the factory status, every task, their PRDs, one event batch and its landed tasks' reviews."""

    head: str | None
    factory: FactoryStatus
    tasks: TasksRead
    prds: PrdsRead
    events: EventBatch
    reviews: Mapping[int, ReviewCounts | NotRecorded]


def _landed_task_ids(events: EventBatch) -> set[int]:
    return {
        task_id
        for event in events.events
        if isinstance(event, LandedCommit)
        for task_id in event.task_ids
    }


def dump_sources(policy: Policy, *, now: datetime, after_id: int, limit: int) -> SourcesDump:
    """Read every source under `policy.project_root`; the event batch starts after `after_id`."""
    root = policy.project_root
    head = main_head(root)
    tasks = read_tasks(root, since=None, escalations=read_open_escalations(root))
    prd_paths = {row.prd_path for row in tasks.rows if row.prd_path is not None}
    prds = (
        PrdsRead(found=(), absent=tuple(sorted(prd_paths)))
        if head is None
        else read_prds(root, commit=head, prd_paths=prd_paths)
    )
    events = read_events(root, after_id=after_id, limit=limit)
    return SourcesDump(
        head=head,
        factory=read_factory_status(root, now=now),
        tasks=tasks,
        prds=prds,
        events=events,
        reviews=read_review_counts(root, _landed_task_ids(events)),
    )


_DUMP: Final = TypeAdapter(SourcesDump)


def render_dump(dump: SourcesDump) -> str:
    return _DUMP.dump_json(dump, indent=2).decode()
