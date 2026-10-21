"""Current values that differ from the folded state, as upsert and snapshot events."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from typing import Any, Final

from client_feed.contract import (
    Branch,
    BranchSnapshot,
    FactoryState,
    FeedEvent,
    FeedState,
    FileKind,
    PrdUpserted,
    RevealBudget,
    TaskStatus,
    TaskUpserted,
    TreeFile,
    TreeSnapshot,
)
from client_feed.feed_io import Stamp, payload
from client_feed.policy import Policy, matches_any
from client_feed.sources.events import FactoryStatus, OpenEscalations
from client_feed.sources.prds import PrdText, summarize_prd
from client_feed.sources.tasks import TaskRow, waiting_on
from client_feed.sources.tree import VisibleTree

IDLE_RESOLUTION_SECONDS: Final = 60


def _differs(recorded: FeedEvent | None, said: Mapping[str, Any]) -> bool:
    """Whether `said` is news against the state's record of the same thing."""
    return recorded is None or payload(recorded) != said


def _read_task_events(
    rows: Iterable[TaskRow], state: FeedState, stamp: Stamp
) -> list[TaskUpserted]:
    events: list[TaskUpserted] = []
    for row in rows:
        said = asdict(row)
        if _differs(state.tasks.get(row.task_id), said):
            events.append(TaskUpserted(**stamp(row.updated_at), **said))
    return events


def _rewaited_task_events(
    state: FeedState, unread: Iterable[int], escalations: OpenEscalations, stamp: Stamp, now: datetime
) -> list[TaskUpserted]:
    events: list[TaskUpserted] = []
    for task_id in unread:
        recorded = state.tasks[task_id]
        waiting = waiting_on(recorded.status, escalations.levels(task_id))
        if waiting != recorded.waiting_on:
            events.append(TaskUpserted(**stamp(now), **{**payload(recorded), 'waiting_on': waiting}))
    return events


def task_events(
    rows: Iterable[TaskRow],
    *,
    state: FeedState,
    escalations: OpenEscalations | None,
    stamp: Stamp,
    now: datetime,
) -> tuple[TaskUpserted, ...]:
    """Each read row the state does not already say; then, given the open escalations, each unread task whose waiting_on they change."""
    rows = tuple(rows)
    events = _read_task_events(rows, state, stamp)
    if escalations is not None:
        read_ids = {row.task_id for row in rows}
        unread = [task_id for task_id in state.tasks if task_id not in read_ids]
        events += _rewaited_task_events(state, unread, escalations, stamp, now)
    return tuple(events)


def _task_counts(tasks: Iterable[TaskUpserted]) -> dict[str, dict[TaskStatus, int]]:
    by_prd: dict[str, Counter[TaskStatus]] = {}
    for task in tasks:
        if task.prd_path is not None:
            by_prd.setdefault(task.prd_path, Counter())[task.status] += 1
    return {path: dict(sorted(counts.items())) for path, counts in by_prd.items()}


def _summary(
    prd_path: str, state: FeedState, summaries: Mapping[str, PrdText] | None
) -> PrdText | None:
    """A fresh summary; else the state's title and excerpt; else, when it was requested and not found, a PRD with no text."""
    if summaries is not None and prd_path in summaries:
        return summaries[prd_path]
    recorded = state.prds.get(prd_path)
    if recorded is not None:
        return PrdText(prd_path=prd_path, title=recorded.title, excerpt=recorded.excerpt)
    return None if summaries is None else summarize_prd(prd_path, '')


def prd_events(
    *,
    tasks: Iterable[TaskUpserted],
    state: FeedState,
    summaries: Mapping[str, PrdText] | None,
    stamp: Stamp,
    now: datetime,
) -> tuple[PrdUpserted, ...]:
    """Each PRD whose title, excerpt or task counts differ from the state's; one never summarized waits for its summary."""
    counts = _task_counts(tasks)
    events: list[PrdUpserted] = []
    for prd_path in sorted(counts.keys() | state.prds.keys()):
        summary = _summary(prd_path, state, summaries)
        if summary is None:
            continue
        said = {**asdict(summary), 'task_counts': counts.get(prd_path, {})}
        if _differs(state.prds.get(prd_path), said):
            events.append(PrdUpserted(**stamp(now), **said))
    return tuple(events)


def _by_the_minute(when: datetime | None) -> int | None:
    return None if when is None else math.floor(when.timestamp() / IDLE_RESOLUTION_SECONDS)


def _quiet_time_key(status: FactoryStatus | FactoryState) -> tuple[object, ...]:
    """The scheduler facts exactly, the quiet time to the minute: idle_seconds moves on every pass."""
    idle = None if status.idle_seconds is None else status.idle_seconds // IDLE_RESOLUTION_SECONDS
    return (status.scheduler, status.since, _by_the_minute(status.last_event_at), idle)


def factory_event(
    status: FactoryStatus, *, state: FeedState, stamp: Stamp, now: datetime
) -> FactoryState | None:
    """The factory status, unless the state already says it to the minute."""
    if state.factory is not None and _quiet_time_key(state.factory) == _quiet_time_key(status):
        return None
    return FactoryState(**stamp(now), **asdict(status))


def branch_event(
    branches: Sequence[Branch], *, state: FeedState, stamp: Stamp, now: datetime
) -> BranchSnapshot | None:
    """Every task branch, unless the state already holds exactly these."""
    branches = tuple(branches)
    if state.branches is not None and state.branches.branches == branches:
        return None
    return BranchSnapshot(**stamp(now), branches=branches)


def _budget(policy: Policy, files: Sequence[TreeFile]) -> RevealBudget:
    """PRD §4.12 with no delivered commit and no reveal granted: the proportion of every counted line."""
    uncounted = (*policy.free_globs, *policy.never_reveal_globs)
    counted = sum(
        file.lines
        for file in files
        if file.kind is FileKind.PRODUCTION and not matches_any(uncounted, file.path)
    )
    return RevealBudget(
        proportion=policy.reveal_proportion,
        counted_lines=counted,
        budget_lines=math.floor(policy.reveal_proportion * counted),
        spent_lines=0,
        floor_lines=policy.reveal_floor_lines,
    )


def tree_event(tree: VisibleTree, *, policy: Policy, stamp: Stamp, now: datetime) -> TreeSnapshot:
    """The visible tree at a new main head, with its reveal budget."""
    return TreeSnapshot(
        **stamp(now),
        commit=tree.commit,
        totals=tree.totals,
        files=tree.files,
        budget=_budget(policy, tree.files),
    )
