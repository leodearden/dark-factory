"""The publisher: one pass reads each source as its own unit and appends its events in a fixed order, a heartbeat last."""

from __future__ import annotations

import itertools
import logging
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeVar

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
    Source,
)
from client_feed.feed_io import Cursor, OutDir, Stamp
from client_feed.fold import fold
from client_feed.policy import Policy
from client_feed.redact import redact_paths
from client_feed.sources.archive import ArchiveReadError
from client_feed.sources.events import (
    FactoryStatus,
    OpenEscalations,
    SkippedRow,
    StreamEvent,
    read_events,
    read_factory_status,
    read_open_escalations,
)
from client_feed.sources.git import GitError, main_head, task_branches
from client_feed.sources.prds import PrdText, read_prds
from client_feed.sources.store import StoreReadError
from client_feed.sources.tasks import TasksRead, read_tasks
from client_feed.sources.tree import VisibleTree, visible_tree
from client_feed.stream_events import Identity, StreamOutput, identity, stream_events
from client_feed.value_events import (
    branch_event,
    factory_event,
    prd_events,
    task_events,
    tree_event,
)

EVENTS_BATCH: Final = 1000
TASKS_RESCAN_PASSES: Final = 60

_NO_ESCALATIONS: Final = OpenEscalations(by_task=MappingProxyType({}))
_SOURCE_ERRORS: Final = (StoreReadError, GitError, ArchiveReadError)

_log = logging.getLogger(__name__)

_T = TypeVar('_T')


class PassReport(BaseModel):
    """What one pass appended, the watermark it committed, and the sources it could not read."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    appended: NonNegativeInt
    watermark_seq: NonNegativeInt
    degraded: tuple[Degradation, ...]


def _source_of(exc: StoreReadError | GitError | ArchiveReadError) -> Source:
    if isinstance(exc, StoreReadError):
        return exc.source
    return Source.GIT if isinstance(exc, GitError) else Source.ARCHIVE


def _guarded(unit: Callable[[], _T], degraded: list[Degradation]) -> _T | None:
    """The unit's result; a typed source error is named in `degraded` and logged instead, any other raises."""
    try:
        return unit()
    except _SOURCE_ERRORS as exc:
        source = _source_of(exc)
        _log.warning(
            'source %s degraded this pass: %s', source, exc.reason if isinstance(exc, GitError) else exc
        )
        degraded.append(Degradation(source=source, reason=redact_paths(str(exc))))
        return None


def _skipped(source: Source, rows: Iterable[tuple[int, StrEnum]]) -> tuple[Degradation, ...]:
    """Rows of `source` that could not be projected, counted by reason; none is no degradation."""
    rows = tuple(rows)
    if not rows:
        return ()
    tally = Counter(reason for _, reason in rows)
    counts = ', '.join(f'{reason}={count}' for reason, count in sorted(tally.items()))
    reason = f'skipped {len(rows)} rows: {counts}'
    _log.warning('source %s %s (ids %s)', source, reason, ', '.join(str(key) for key, _ in rows))
    return (Degradation(source=source, reason=reason),)


@dataclass(frozen=True)
class _EventsStore:
    """runs.db as one pass reads it: the open escalations, every stream row past the cursor, the factory status."""

    escalations: OpenEscalations
    stream: tuple[StreamEvent, ...]
    skipped: tuple[SkippedRow, ...]
    cursor: int
    factory: FactoryStatus


def _read_events_store(root: Path, *, after_id: int, now: datetime) -> _EventsStore:
    """Every stream row past `after_id`, batch after batch until none is left, with the store's other facts."""
    escalations = read_open_escalations(root)
    stream: list[StreamEvent] = []
    skipped: list[SkippedRow] = []
    cursor = after_id
    while True:
        batch = read_events(root, after_id=cursor, limit=EVENTS_BATCH)
        stream.extend(batch.events)
        skipped.extend(batch.skipped)
        cursor = batch.cursor
        if not batch.has_more:
            break
    return _EventsStore(
        escalations=escalations,
        stream=tuple(stream),
        skipped=tuple(skipped),
        cursor=cursor,
        factory=read_factory_status(root, now=now),
    )


def _events_store_unit(
    root: Path, cursor: Cursor, now: datetime, degraded: list[Degradation]
) -> _EventsStore | None:
    store = _guarded(lambda: _read_events_store(root, after_id=cursor.events_id, now=now), degraded)
    if store is not None:
        degraded.extend(_skipped(Source.EVENTS, ((row.event_id, row.reason) for row in store.skipped)))
    return store


def _tasks_unit(
    root: Path, since: str | None, store: _EventsStore | None, degraded: list[Degradation]
) -> TasksRead | None:
    """The task rows; while runs.db is unreadable they are read as if no escalation were open."""
    escalations = _NO_ESCALATIONS if store is None else store.escalations
    tasks = _guarded(lambda: read_tasks(root, since=since, escalations=escalations), degraded)
    if tasks is not None:
        degraded.extend(_skipped(Source.TASKS, ((task.task_id, task.reason) for task in tasks.skipped)))
    return tasks


def _published(lines: Iterable[FeedLine]) -> frozenset[Identity]:
    """The identities of the stream events a stopped writer appended but never recorded in its cursor."""
    keys = (identity(line) for line in lines if not isinstance(line, EventHeader))
    return frozenset(key for key in keys if key is not None)


def _stream_unit(
    policy: Policy,
    out: OutDir,
    store: _EventsStore | None,
    view: FeedState,
    stamp: Stamp,
    degraded: list[Degradation],
) -> StreamOutput | None:
    """The stream's events, all or none: a failure reading the landed merges holds the whole batch."""
    if store is None:
        return None
    stream = _guarded(
        lambda: stream_events(
            policy.project_root,
            store.stream,
            state=view,
            max_cases=policy.max_cases_per_run,
            published=_published(out.uncommitted),
            stamp=stamp,
        ),
        degraded,
    )
    if stream is not None:
        degraded.extend(stream.degraded)
    return stream


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


def _read_summaries(root: Path, head: str, *, moved: bool, view: FeedState) -> Mapping[str, PrdText]:
    """Every PRD's summary at a new head, else only the PRDs the state has never summarized."""
    paths = _prd_paths(view)
    requested = paths if moved else paths - view.prds.keys()
    read = read_prds(root, commit=head, prd_paths=requested)
    return {summary.prd_path: summary for summary in read.found}


def _summaries_unit(
    root: Path, git: _Git | None, view: FeedState, degraded: list[Degradation]
) -> Mapping[str, PrdText] | None:
    """The PRD summaries at main's head; None, so no PRD is newly summarized, when they cannot be read."""
    if git is None or git.head is None:
        return None
    head = git.head
    return _guarded(
        lambda: _read_summaries(root, head, moved=git.tree is not None, view=view), degraded
    )


def _advanced(
    cursor: Cursor, *, events_id: int | None, tasks: TasksRead | None, tree: VisibleTree | None
) -> Cursor:
    """Each source position past what this pass read, where its unit succeeded; head and line counts move with the tree."""
    update: dict[str, object] = {}
    if events_id is not None:
        update['events_id'] = events_id
    if tasks is not None:
        update['tasks_updated_at'] = tasks.cursor
    if tree is not None:
        update |= {'head': tree.commit, 'line_counts': tree.line_counts}
    return cursor.model_copy(update=update)


def run_pass(
    policy: Policy, out: OutDir, *, now: datetime, rescan_tasks: bool = True
) -> PassReport:
    """Read every source, append what is news in the order tasks, stream, factory, branches, tree, PRDs, heartbeat.

    A source that cannot be read is named in the heartbeat's degraded and holds only its own cursor fields.
    """
    root = policy.project_root
    cursor = out.cursor
    stamp = out.stamp()
    degraded: list[Degradation] = []
    store = _events_store_unit(root, cursor, now, degraded)
    tasks = _tasks_unit(root, None if rescan_tasks else cursor.tasks_updated_at, store, degraded)
    upserts = task_events(
        () if tasks is None else tasks.rows,
        state=out.state,
        escalations=None if store is None else store.escalations,
        stamp=stamp,
        now=now,
    )
    view = fold(upserts, project_id=out.state.project_id, start=out.state)
    stream = _stream_unit(policy, out, store, view, stamp, degraded)
    factory = None if store is None else factory_event(store.factory, state=view, stamp=stamp, now=now)
    git = _guarded(lambda: _read_git(policy, cursor=cursor, view=view), degraded)
    branches = None if git is None else branch_event(git.branches, state=view, stamp=stamp, now=now)
    tree = None if git is None or git.tree is None else git.tree
    snapshot = None if tree is None else tree_event(tree, policy=policy, stamp=stamp, now=now)
    prds = prd_events(
        tasks=view.tasks.values(),
        state=view,
        summaries=_summaries_unit(root, git, view, degraded),
        stamp=stamp,
        now=now,
    )
    heartbeat = Heartbeat(
        **stamp(now),
        head=None if git is None else git.head,
        as_of=now,
        engagement_version=None,
        watermark_seq=out.state.watermark_seq,
        degraded=tuple(degraded),
    )
    events: list[FeedEvent] = [
        *upserts,
        *(() if stream is None else stream.events),
        *(event for event in (factory, branches, snapshot) if event is not None),
        *prds,
        heartbeat,
    ]
    events_id = None if store is None or stream is None else store.cursor
    out.commit(events, _advanced(cursor, events_id=events_id, tasks=tasks, tree=tree))
    return PassReport(
        appended=len(events), watermark_seq=out.state.watermark_seq, degraded=heartbeat.degraded
    )


def _open(policy: Policy) -> OutDir:
    return OutDir.open(policy.out_dir, project_id=policy.project_id)


def snapshot(policy: Policy, *, now: datetime) -> PassReport:
    """One pass over every source, every task re-read."""
    return run_pass(policy, _open(policy), now=now, rescan_tasks=True)


def publish(policy: Policy, *, stop: threading.Event, clock: Callable[[], datetime]) -> None:
    """A pass every interval_seconds until `stop` is set; a pass in progress always completes.

    Every TASKS_RESCAN_PASSES-th pass, the first included, re-reads every task, so a row committed
    behind the tasks cursor is still published.
    """
    out = _open(policy)
    for passes in itertools.count():
        if stop.is_set():
            return
        report = run_pass(
            policy, out, now=clock(), rescan_tasks=passes % TASKS_RESCAN_PASSES == 0
        )
        _log.log(
            logging.INFO if report.appended > 1 else logging.DEBUG,
            'pass %d appended %d lines through seq %d',
            passes, report.appended, report.watermark_seq,
        )
        stop.wait(policy.interval_seconds)
