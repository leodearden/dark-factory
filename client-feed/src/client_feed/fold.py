"""The one reducer of the client feed: feed lines to FeedState (plans/client-feed-prd.md §4.2, §7.3)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from client_feed.contract import (
    FEED_VERSION,
    BranchSnapshot,
    CriteriaResults,
    Degradation,
    EventHeader,
    FactoryState,
    FeedError,
    FeedEvent,
    FeedLine,
    FeedState,
    FileRevealed,
    Heartbeat,
    InFlightTask,
    MergeLanded,
    PrdUpserted,
    RevealedFile,
    RevealRefused,
    RunKind,
    TaskPhase,
    TaskUpserted,
    TreeSnapshot,
    VerifyRun,
    VerifyTestResults,
)

RECENT_LIMIT: Final = 200


class CorruptFeed(FeedError):
    """A feed whose seq does not strictly increase."""

    def __init__(self, *, seq: int, previous_seq: int) -> None:
        self.seq = seq
        self.previous_seq = previous_seq
        super().__init__(
            f'corrupt feed: seq must strictly increase, but seq {seq} follows seq {previous_seq}'
        )


class ProjectMismatch(FeedError):
    """A feed line, or a start state, naming another project than the one being folded."""

    def __init__(self, *, seq: int | None, found: str, expected: str) -> None:
        self.seq = seq
        self.found = found
        self.expected = expected
        where = 'the start state' if seq is None else f'seq {seq}'
        super().__init__(
            f'project mismatch at {where}: found project_id {found!r}, expected {expected!r}'
        )


_RecentKey = tuple[str, str | int]


def _recent_key(event: FeedEvent) -> _RecentKey:
    key = event.natural_key
    return (event.type, event.seq if key is None else key)


@dataclass
class _Accumulator:
    project_id: str
    watermark: int
    as_of: datetime | None
    head: str | None
    engagement_version: int | None
    degraded: tuple[Degradation, ...]
    factory: FactoryState | None
    branches: BranchSnapshot | None
    tree: TreeSnapshot | None
    prds: dict[str, PrdUpserted]
    tasks: dict[int, TaskUpserted]
    in_flight: dict[int, InFlightTask]
    revealed: dict[str, RevealedFile]
    criteria: dict[RunKind, CriteriaResults]
    recent: OrderedDict[_RecentKey, FeedEvent]

    @classmethod
    def resuming(cls, start: FeedState) -> _Accumulator:
        return cls(
            project_id=start.project_id,
            watermark=start.watermark_seq,
            as_of=start.as_of,
            head=start.head,
            engagement_version=start.engagement_version,
            degraded=start.degraded,
            factory=start.factory,
            branches=start.branches,
            tree=start.tree,
            prds=dict(start.prds),
            tasks=dict(start.tasks),
            in_flight={task.task_id: task for task in start.in_flight},
            revealed={file.path: file for file in start.revealed},
            criteria=dict(start.criteria),
            recent=OrderedDict((_recent_key(event), event) for event in start.recent),
        )

    def frozen(self) -> FeedState:
        return FeedState(
            project_id=self.project_id,
            feed_version=FEED_VERSION,
            watermark_seq=self.watermark,
            as_of=self.as_of,
            head=self.head,
            engagement_version=self.engagement_version,
            degraded=self.degraded,
            factory=self.factory,
            prds=dict(sorted(self.prds.items())),
            tasks=dict(sorted(self.tasks.items())),
            in_flight=tuple(task for _, task in sorted(self.in_flight.items())),
            branches=self.branches,
            tree=self.tree,
            revealed=tuple(file for _, file in sorted(self.revealed.items())),
            criteria=dict(sorted(self.criteria.items())),
            recent=tuple(self.recent.values()),
        )


def _empty_state(project_id: str) -> FeedState:
    return FeedState(
        project_id=project_id,
        feed_version=FEED_VERSION,
        watermark_seq=0,
        as_of=None,
        head=None,
        engagement_version=None,
        degraded=(),
        factory=None,
        prds={},
        tasks={},
        in_flight=(),
        branches=None,
        tree=None,
        revealed=(),
        criteria={},
        recent=(),
    )


def _apply_heartbeat(acc: _Accumulator, event: Heartbeat) -> None:
    acc.as_of = event.as_of
    acc.head = event.head
    acc.engagement_version = event.engagement_version
    acc.degraded = event.degraded


def _apply_factory_state(acc: _Accumulator, event: FactoryState) -> None:
    acc.factory = event


def _apply_prd_upserted(acc: _Accumulator, event: PrdUpserted) -> None:
    acc.prds[event.prd_path] = event


def _apply_task_upserted(acc: _Accumulator, event: TaskUpserted) -> None:
    acc.tasks[event.task_id] = event


def _apply_task_phase(acc: _Accumulator, event: TaskPhase) -> None:
    if event.entered:
        acc.in_flight[event.task_id] = InFlightTask(
            task_id=event.task_id, phase=event.phase, since=event.at
        )
        return
    current = acc.in_flight.get(event.task_id)
    if current is not None and current.phase == event.phase:
        del acc.in_flight[event.task_id]


def _apply_branch_snapshot(acc: _Accumulator, event: BranchSnapshot) -> None:
    acc.branches = event


def _apply_tree_snapshot(acc: _Accumulator, event: TreeSnapshot) -> None:
    acc.tree = event


def _apply_file_revealed(acc: _Accumulator, event: FileRevealed) -> None:
    if event.renamed_from is not None:
        acc.revealed.pop(event.renamed_from, None)
    if event.deleted:
        acc.revealed.pop(event.path, None)
    else:
        acc.revealed[event.path] = RevealedFile(
            path=event.path, commit=event.commit, lines=event.lines
        )


def _apply_criteria_results(acc: _Accumulator, event: CriteriaResults) -> None:
    acc.criteria[event.run_kind] = event


def _timeline_only(acc: _Accumulator, event: FeedEvent) -> None:
    """The event's only trace in the state is its entry in `recent`."""


_APPLIERS: Mapping[type, Callable[[_Accumulator, Any], None]] = {
    Heartbeat: _apply_heartbeat,
    FactoryState: _apply_factory_state,
    PrdUpserted: _apply_prd_upserted,
    TaskUpserted: _apply_task_upserted,
    TaskPhase: _apply_task_phase,
    VerifyRun: _timeline_only,
    VerifyTestResults: _timeline_only,
    MergeLanded: _timeline_only,
    BranchSnapshot: _apply_branch_snapshot,
    TreeSnapshot: _apply_tree_snapshot,
    FileRevealed: _apply_file_revealed,
    RevealRefused: _timeline_only,
    CriteriaResults: _apply_criteria_results,
}

_NOT_IN_TIMELINE: Final = frozenset({Heartbeat, FactoryState, BranchSnapshot, TreeSnapshot})


def _remember(acc: _Accumulator, event: FeedEvent) -> None:
    key = _recent_key(event)
    acc.recent.pop(key, None)
    acc.recent[key] = event
    while len(acc.recent) > RECENT_LIMIT:
        acc.recent.popitem(last=False)


def _apply(acc: _Accumulator, line: FeedLine) -> None:
    if isinstance(line, EventHeader):
        return
    _APPLIERS[type(line)](acc, line)
    if type(line) not in _NOT_IN_TIMELINE:
        _remember(acc, line)


def _checked_start(start: FeedState | None, project_id: str) -> FeedState:
    if start is None:
        return _empty_state(project_id)
    if start.project_id != project_id:
        raise ProjectMismatch(seq=None, found=start.project_id, expected=project_id)
    return start


def _check_line(acc: _Accumulator, line: FeedLine) -> None:
    if line.project_id != acc.project_id:
        raise ProjectMismatch(seq=line.seq, found=line.project_id, expected=acc.project_id)
    if line.seq <= acc.watermark:
        raise CorruptFeed(seq=line.seq, previous_seq=acc.watermark)


def fold(
    lines: Iterable[FeedLine],
    *,
    project_id: str,
    upto_seq: int | None = None,
    start: FeedState | None = None,
) -> FeedState:
    """The state after `lines`, resumed from `start`, stopping after the last line at or below `upto_seq`."""
    acc = _Accumulator.resuming(_checked_start(start, project_id))
    for line in lines:
        _check_line(acc, line)
        if upto_seq is not None and line.seq > upto_seq:
            break
        _apply(acc, line)
        acc.watermark = line.seq
    return acc.frozen()
