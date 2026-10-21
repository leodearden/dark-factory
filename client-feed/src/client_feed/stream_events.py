"""Each runs.db stream row as its feed events, once."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, TypeAlias

from client_feed.contract import (
    NOT_RECORDED,
    Degradation,
    FeedEvent,
    FeedState,
    MergeLanded,
    NotRecorded,
    ReviewCounts,
    Source,
    TaskPhase,
    VerifyKind,
    VerifyRun,
    VerifyTestResults,
)
from client_feed.envelope import Stamp
from client_feed.redact import redact_paths
from client_feed.sources.archive import (
    ArchivedRun,
    ArchivedTests,
    ArchiveReadError,
    read_archived_run,
)
from client_feed.sources.events import (
    LandedCommit,
    PhaseChange,
    StreamEvent,
    VerifyVerdict,
    read_review_counts,
)
from client_feed.sources.git import GitError, MergeStats, merge_stats

VERIFY_PHASE: Final = 'verify'  # orchestrator/src/orchestrator/workflow_types.py::WorkflowState.VERIFY
_MERGE_LANDED: Final = MergeLanded.model_fields['type'].default

Identity: TypeAlias = tuple[str, str | int]

_NOTHING_ARCHIVED: Final = ArchivedRun(legs=NOT_RECORDED, tests=None)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamOutput:
    """The feed events of a run of stream rows, and the archive reads that failed."""

    events: tuple[FeedEvent, ...]
    degraded: tuple[Degradation, ...]


_EMPTY: Final = StreamOutput(events=(), degraded=())


def identity(event: FeedEvent) -> Identity | None:
    """What makes a stream event the same event on a later pass: its natural key, or for a task_phase all but its seq."""
    if isinstance(event, TaskPhase):
        return event.type, event.model_dump_json(exclude={'seq'})
    if isinstance(event, VerifyRun | VerifyTestResults | MergeLanded):
        return event.type, event.natural_key
    return None


def _joined(outputs: Iterable[StreamOutput]) -> StreamOutput:
    outputs = tuple(outputs)
    return StreamOutput(
        events=tuple(event for output in outputs for event in output.events),
        degraded=tuple(reason for output in outputs for reason in output.degraded),
    )


def _archived(
    root: Path, task_id: int, started_at: datetime, ended_at: datetime, max_cases: int
) -> ArchivedRun | Degradation:
    try:
        return read_archived_run(
            root, task_id, started_at=started_at, ended_at=ended_at, max_cases=max_cases
        )
    except ArchiveReadError as err:
        reason = redact_paths(str(err))
        _log.warning('verify archive unreadable for task %d: %s', task_id, reason)
        return Degradation(source=Source.ARCHIVE, reason=reason)


def _test_results(
    run: VerifyRun, tests: ArchivedTests | None, stamp: Stamp
) -> tuple[VerifyTestResults, ...]:
    if tests is None:
        return ()
    return (
        VerifyTestResults(
            **stamp(run.at),
            run_key=run.run_key,
            counts=tests.counts,
            duration_ms=tests.duration_ms,
            cases=tests.cases,
        ),
    )


def _task_run(
    root: Path, exit_: PhaseChange, started_at: datetime, max_cases: int, stamp: Stamp
) -> StreamOutput:
    """The verify run a verify-phase exit closes; with no summary in its window, whether it passed is not recorded."""
    if started_at > exit_.at:
        return _EMPTY
    archived = _archived(root, exit_.task_id, started_at, exit_.at, max_cases)
    if isinstance(archived, Degradation):
        return StreamOutput(events=(), degraded=(archived,))
    if archived.legs == NOT_RECORDED or not archived.legs:
        return _EMPTY
    run = VerifyRun(
        **stamp(exit_.at),
        run_key=f'task-{exit_.event_id}',
        task_id=exit_.task_id,
        kind=VerifyKind.TASK,
        passed=all(leg.rc == 0 for leg in archived.legs),
        duration_ms=None,
        commit=None,
        legs=archived.legs,
    )
    return StreamOutput(events=(run, *_test_results(run, archived.tests, stamp)), degraded=())


def _phase_output(
    root: Path,
    change: PhaseChange,
    verify_starts: dict[int, datetime],
    max_cases: int,
    stamp: Stamp,
) -> StreamOutput:
    """The change as a task_phase; a verify exit also closes the run its task's verify entry opened."""
    phase = TaskPhase(
        **stamp(change.at), task_id=change.task_id, phase=change.phase, entered=change.entered
    )
    if change.phase != VERIFY_PHASE:
        return StreamOutput(events=(phase,), degraded=())
    if change.entered:
        verify_starts[change.task_id] = change.at
        return StreamOutput(events=(phase,), degraded=())
    started_at = verify_starts.pop(change.task_id, None)
    run = _EMPTY if started_at is None else _task_run(root, change, started_at, max_cases, stamp)
    return StreamOutput(events=(phase, *run.events), degraded=run.degraded)


def _merge_archive(
    root: Path, verdict: VerifyVerdict, max_cases: int
) -> tuple[ArchivedRun, tuple[Degradation, ...]]:
    """What the archive holds for a merge verify: it is recorded after its run, `duration_ms` long."""
    if verdict.task_id is None or verdict.duration_ms is None:
        return _NOTHING_ARCHIVED, ()
    started_at = verdict.at - timedelta(milliseconds=verdict.duration_ms)
    archived = _archived(root, verdict.task_id, started_at, verdict.at, max_cases)
    if isinstance(archived, Degradation):
        return _NOTHING_ARCHIVED, (archived,)
    return archived, ()


def _merge_run(root: Path, verdict: VerifyVerdict, max_cases: int, stamp: Stamp) -> StreamOutput:
    """A merge verdict's run, emitted whatever the archive holds: the verdict records whether it passed."""
    archived, degraded = _merge_archive(root, verdict, max_cases)
    run = VerifyRun(
        **stamp(verdict.at),
        run_key=f'merge-{verdict.event_id}',
        task_id=verdict.task_id,
        kind=VerifyKind.MERGE,
        passed=verdict.passed,
        duration_ms=verdict.duration_ms,
        commit=verdict.commit,
        legs=archived.legs,
    )
    return StreamOutput(events=(run, *_test_results(run, archived.tests, stamp)), degraded=degraded)


def _first_landings(stream: Sequence[StreamEvent]) -> dict[str, LandedCommit]:
    """Each landed commit at its first landing, carrying every landing's tasks in first-seen order."""
    first: dict[str, LandedCommit] = {}
    for landing in stream:
        if isinstance(landing, LandedCommit):
            seen = first.get(landing.commit, landing)
            task_ids = tuple(dict.fromkeys((*seen.task_ids, *landing.task_ids)))
            first[landing.commit] = replace(seen, task_ids=task_ids)
    return first


def _landing_identity(landing: LandedCommit) -> Identity:
    """The identity of the merge_landed a landing becomes, known before it is built."""
    return _MERGE_LANDED, landing.commit


def _stats(root: Path, landing: LandedCommit, skip_unreadable_before: datetime | None) -> MergeStats | None:
    """The landing's merge stats; None for a merge git cannot read that landed before the cutoff."""
    try:
        return merge_stats(root, landing.commit)
    except GitError as exc:
        if skip_unreadable_before is None or landing.at >= skip_unreadable_before:
            raise
        _log.error(
            'skipping the landing of %s (event %d) at %s: git cannot read it (%s)',
            landing.commit, landing.event_id, landing.at.isoformat(), exc.reason,
        )
        return None


@dataclass(frozen=True)
class _Landed:
    """The merges a stream lands, each at its first landing; stats only for those to emit, which git could read."""

    first: Mapping[str, LandedCommit]
    stats: Mapping[str, MergeStats]
    reviews: Mapping[int, ReviewCounts | NotRecorded]
    skipped: int


def _landed(
    root: Path,
    stream: Sequence[StreamEvent],
    published: AbstractSet[Identity],
    skip_unreadable_before: datetime | None,
) -> _Landed:
    """Git read in landing order, before any archive, and only for landings not yet published."""
    first = _first_landings(stream)
    stats: dict[str, MergeStats] = {}
    skipped = 0
    for commit, landing in first.items():
        if _landing_identity(landing) in published:
            continue
        read = _stats(root, landing, skip_unreadable_before)
        if read is None:
            skipped += 1
        else:
            stats[commit] = read
    task_ids = {task_id for commit in stats for task_id in first[commit].task_ids}
    return _Landed(
        first=first, stats=stats, reviews=read_review_counts(root, task_ids), skipped=skipped
    )


def _skipped_landings(landed: _Landed) -> tuple[Degradation, ...]:
    if not landed.skipped:
        return ()
    noun = 'landing' if landed.skipped == 1 else 'landings'
    reason = f'skipped {landed.skipped} {noun} whose merge git cannot read'
    return (Degradation(source=Source.GIT, reason=reason),)


def _merge_landed(
    landing: LandedCommit, landed: _Landed, state: FeedState, stamp: Stamp
) -> StreamOutput:
    """A commit's merge_landed at its first landing; only a single task's merge has one review to report."""
    first = landed.first[landing.commit]
    if first.event_id != landing.event_id or first.commit not in landed.stats:
        return _EMPTY
    titled = (state.tasks.get(task_id) for task_id in first.task_ids)
    merge = MergeLanded(
        **stamp(first.at),
        commit=first.commit,
        task_ids=first.task_ids,
        titles=tuple('' if task is None else task.title for task in titled),
        **asdict(landed.stats[first.commit]),
        review=landed.reviews[first.task_ids[0]] if len(first.task_ids) == 1 else NOT_RECORDED,
    )
    return StreamOutput(events=(merge,), degraded=())


def _verify_starts(state: FeedState) -> dict[int, datetime]:
    """The verify windows a previous pass left open, by task."""
    return {task.task_id: task.since for task in state.in_flight if task.phase == VERIFY_PHASE}


def stream_events(
    root: Path,
    stream: Sequence[StreamEvent],
    *,
    state: FeedState,
    max_cases: int,
    published: AbstractSet[Identity],
    stamp: Stamp,
    skip_unreadable_before: datetime | None,
) -> StreamOutput:
    """The feed events of `stream` in id order, less those already `published`.

    A workflow_verify verdict adds none: its verify-phase exit reports the run. A GitError or StoreReadError
    reading the landed merges propagates, so the caller holds the whole batch; but a merge git cannot read
    that landed before `skip_unreadable_before` emits no merge_landed and is named in `degraded` instead.
    """
    landed = _landed(root, stream, published, skip_unreadable_before)
    verify_starts = _verify_starts(state)
    outputs: list[StreamOutput] = []
    for item in stream:
        if isinstance(item, PhaseChange):
            outputs.append(_phase_output(root, item, verify_starts, max_cases, stamp))
        elif isinstance(item, LandedCommit):
            outputs.append(_merge_landed(item, landed, state, stamp))
        elif item.kind is VerifyKind.MERGE:
            outputs.append(_merge_run(root, item, max_cases, stamp))
    built = _joined(outputs)
    return StreamOutput(
        events=tuple(event for event in built.events if identity(event) not in published),
        degraded=(*_skipped_landings(landed), *built.degraded),
    )
