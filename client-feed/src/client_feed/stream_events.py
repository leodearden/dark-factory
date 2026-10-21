"""Each runs.db stream row as its feed events, once."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, TypeAlias

from client_feed.contract import (
    NOT_RECORDED,
    Degradation,
    FeedEvent,
    FeedState,
    Source,
    TaskPhase,
    VerifyKind,
    VerifyRun,
    VerifyTestResults,
)
from client_feed.feed_io import Stamp
from client_feed.redact import redact_paths
from client_feed.sources.archive import (
    ArchivedRun,
    ArchivedTests,
    ArchiveReadError,
    read_archived_run,
)
from client_feed.sources.events import PhaseChange, StreamEvent, VerifyVerdict

VERIFY_PHASE: Final = 'verify'  # orchestrator/src/orchestrator/workflow_types.py::WorkflowState.VERIFY

Identity: TypeAlias = tuple[str, str | int]

_NOTHING_ARCHIVED: Final = ArchivedRun(legs=NOT_RECORDED, tests=None)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamOutput:
    """The feed events of a run of stream rows, and the archive reads that failed."""

    events: tuple[FeedEvent, ...]
    degraded: tuple[Degradation, ...]


_EMPTY: Final = StreamOutput(events=(), degraded=())


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
) -> StreamOutput:
    """The feed events of `stream`, in id order; a workflow_verify verdict adds none, its verify-phase exit reports the run."""
    verify_starts = _verify_starts(state)
    outputs: list[StreamOutput] = []
    for item in stream:
        if isinstance(item, PhaseChange):
            outputs.append(_phase_output(root, item, verify_starts, max_cases, stamp))
        elif isinstance(item, VerifyVerdict) and item.kind is VerifyKind.MERGE:
            outputs.append(_merge_run(root, item, max_cases, stamp))
    return _joined(outputs)
