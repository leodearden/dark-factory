"""One publisher pass: each source read as its own unit, its events built in a fixed order, a heartbeat last."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from client_feed.contract import (
    Branch,
    Degradation,
    EventHeader,
    FeedEvent,
    FeedLine,
    FeedState,
    Heartbeat,
    NonNegativeInt,
)
from client_feed.feed_io import Cursor, OutDir
from client_feed.fold import fold
from client_feed.policy import Policy
from client_feed.sources.events import (
    FactoryStatus,
    OpenEscalations,
    StreamEvent,
    read_events,
    read_factory_status,
    read_open_escalations,
)
from client_feed.sources.git import main_head, task_branches
from client_feed.sources.prds import PrdText, read_prds
from client_feed.sources.tasks import read_tasks
from client_feed.sources.tree import VisibleTree, visible_tree
from client_feed.stream_events import Identity, identity, stream_events
from client_feed.value_events import (
    branch_event,
    factory_event,
    prd_events,
    task_events,
    tree_event,
)

EVENTS_BATCH: Final = 1000


class PassReport(BaseModel):
    """What one pass appended, the watermark it committed, and the sources it could not read."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    appended: NonNegativeInt
    watermark_seq: NonNegativeInt
    degraded: tuple[Degradation, ...]


@dataclass(frozen=True)
class _EventsStore:
    """runs.db as one pass reads it: the open escalations, every stream row past the cursor, the factory status."""

    escalations: OpenEscalations
    stream: tuple[StreamEvent, ...]
    cursor: int
    factory: FactoryStatus


def _drained_stream(root: Path, after_id: int) -> tuple[tuple[StreamEvent, ...], int]:
    """Every stream row past `after_id`, batch after batch until none is left, and the last id read."""
    stream: list[StreamEvent] = []
    cursor = after_id
    while True:
        batch = read_events(root, after_id=cursor, limit=EVENTS_BATCH)
        stream.extend(batch.events)
        cursor = batch.cursor
        if not batch.has_more:
            return tuple(stream), cursor


def _read_events_store(root: Path, *, after_id: int, now: datetime) -> _EventsStore:
    escalations = read_open_escalations(root)
    stream, cursor = _drained_stream(root, after_id)
    return _EventsStore(
        escalations=escalations,
        stream=stream,
        cursor=cursor,
        factory=read_factory_status(root, now=now),
    )


@dataclass(frozen=True)
class _Git:
    """main's head and the task branches; the visible tree only when the head moved past the cursor's."""

    head: str | None
    branches: tuple[Branch, ...]
    tree: VisibleTree | None


def _read_git(policy: Policy, *, cursor: Cursor, view: FeedState) -> _Git:
    root = policy.project_root
    head = main_head(root)
    branches = task_branches(root)
    tree = None
    if head is not None and head != cursor.head:
        tree = visible_tree(root, policy, head, previous=view.tree, line_counts=cursor.line_counts)
    return _Git(head=head, branches=branches, tree=tree)


def _prd_paths(view: FeedState) -> set[str]:
    return {task.prd_path for task in view.tasks.values() if task.prd_path is not None}


def _read_summaries(root: Path, git: _Git, view: FeedState) -> Mapping[str, PrdText] | None:
    """Every PRD's summary at a new head, else only the PRDs the state has never summarized."""
    if git.head is None:
        return None
    paths = _prd_paths(view)
    requested = paths if git.tree is not None else paths - view.prds.keys()
    read = read_prds(root, commit=git.head, prd_paths=requested)
    return {summary.prd_path: summary for summary in read.found}


def _published(lines: Iterable[FeedLine]) -> frozenset[Identity]:
    """The identities of the stream events a stopped writer appended but never recorded in its cursor."""
    keys = (identity(line) for line in lines if not isinstance(line, EventHeader))
    return frozenset(key for key in keys if key is not None)


def _advanced(cursor: Cursor, *, events: _EventsStore, tasks_cursor: str | None, git: _Git) -> Cursor:
    """Each source position past what this pass read; the head and its line counts move with the tree."""
    update: dict[str, object] = {'events_id': events.cursor, 'tasks_updated_at': tasks_cursor}
    if git.tree is not None:
        update |= {'head': git.tree.commit, 'line_counts': git.tree.line_counts}
    return cursor.model_copy(update=update)


def run_pass(
    policy: Policy, out: OutDir, *, now: datetime, rescan_tasks: bool = True
) -> PassReport:
    """Read every source, append what is news in the order tasks, stream, factory, branches, tree, PRDs, heartbeat."""
    root = policy.project_root
    cursor = out.cursor
    stamp = out.stamp()
    store = _read_events_store(root, after_id=cursor.events_id, now=now)
    tasks = read_tasks(
        root,
        since=None if rescan_tasks else cursor.tasks_updated_at,
        escalations=store.escalations,
    )
    upserts = task_events(
        tasks.rows, state=out.state, escalations=store.escalations, stamp=stamp, now=now
    )
    view = fold(upserts, project_id=out.state.project_id, start=out.state)
    stream = stream_events(
        root,
        store.stream,
        state=view,
        max_cases=policy.max_cases_per_run,
        published=_published(out.uncommitted),
        stamp=stamp,
    )
    factory = factory_event(store.factory, state=view, stamp=stamp, now=now)
    git = _read_git(policy, cursor=cursor, view=view)
    branches = branch_event(git.branches, state=view, stamp=stamp, now=now)
    tree = None if git.tree is None else tree_event(git.tree, policy=policy, stamp=stamp, now=now)
    prds = prd_events(
        tasks=view.tasks.values(),
        state=view,
        summaries=_read_summaries(root, git, view),
        stamp=stamp,
        now=now,
    )
    degraded = stream.degraded
    heartbeat = Heartbeat(
        **stamp(now),
        head=git.head,
        as_of=now,
        engagement_version=None,
        watermark_seq=out.state.watermark_seq,
        degraded=degraded,
    )
    events: list[FeedEvent] = [
        *upserts,
        *stream.events,
        *(event for event in (factory, branches, tree) if event is not None),
        *prds,
        heartbeat,
    ]
    out.commit(events, _advanced(cursor, events=store, tasks_cursor=tasks.cursor, git=git))
    return PassReport(appended=len(events), watermark_seq=out.state.watermark_seq, degraded=degraded)
