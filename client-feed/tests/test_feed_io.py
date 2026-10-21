"""out/ as durable storage with one writer: append, fold, atomic state and cursor (PRD §4.1-§4.3, §4.15)."""

import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from client_feed.contract import (
    FEED_VERSION,
    FactoryState,
    FeedEvent,
    Heartbeat,
    SchedulerState,
    TaskStatus,
    TaskUpserted,
    parse_feed_line,
    parse_state,
    split_jsonl,
)
from client_feed.feed_io import Cursor, FeedIoError, FeedIoProblem, OutDir, Stamp
from client_feed.fold import ProjectMismatch, fold
from client_feed.validate import validate_out_dir

PROJECT = 'acme'
EPOCH = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
WRITTEN = ('cursor.json', 'feed.jsonl', 'state.json')


def at(minutes: float) -> datetime:
    return EPOCH + timedelta(minutes=minutes)


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'out'
    path.mkdir()
    return path


def heartbeat(stamp: Stamp, minutes: float = 0, *, watermark_seq: int = 0) -> Heartbeat:
    return Heartbeat(
        **stamp(at(minutes)),
        head=None,
        as_of=at(minutes),
        engagement_version=None,
        watermark_seq=watermark_seq,
        degraded=(),
    )


def task(stamp: Stamp, task_id: int = 7, minutes: float = 0) -> TaskUpserted:
    return TaskUpserted(
        **stamp(at(minutes)),
        task_id=task_id,
        title=f'Task {task_id}',
        status=TaskStatus.PENDING,
        waiting_on=None,
        updated_at=at(minutes),
        depends_on=(),
        prd_path=None,
        prd_task_label=None,
        landed_commit=None,
    )


def factory(stamp: Stamp, minutes: float = 0) -> FactoryState:
    return FactoryState(
        **stamp(at(minutes)),
        scheduler=SchedulerState.RUNNING,
        since=None,
        last_event_at=at(minutes),
        idle_seconds=0,
    )


def three_events(out: OutDir) -> list[FeedEvent]:
    stamp = out.stamp()
    return [task(stamp), factory(stamp), heartbeat(stamp)]


def raw_heartbeat(seq: int, *, project_id: str = PROJECT) -> Heartbeat:
    return Heartbeat(
        seq=seq,
        at=at(seq),
        project_id=project_id,
        feed_version=FEED_VERSION,
        head=None,
        as_of=at(seq),
        engagement_version=None,
        watermark_seq=0,
        degraded=(),
    )


def feed_texts(out_dir: Path) -> tuple[str, ...]:
    return split_jsonl((out_dir / 'feed.jsonl').read_text())


def feed_lines(out_dir: Path) -> list:
    return [parse_feed_line(text, line_no=n) for n, text in enumerate(feed_texts(out_dir), 1)]


def stored_cursor(out_dir: Path) -> Cursor:
    return Cursor.model_validate_json((out_dir / 'cursor.json').read_text())


def test_an_empty_out_dir_opens_at_the_start(out_dir: Path) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)

    assert out.state.watermark_seq == 0
    assert out.state == fold([], project_id=PROJECT)
    assert out.cursor == Cursor()
    assert out.cursor == Cursor(
        through_seq=0,
        feed_bytes=0,
        events_id=0,
        tasks_updated_at=None,
        head=None,
        inbox_offset=0,
        line_counts=None,
    )
    assert out.uncommitted == ()
    assert out.next_seq == 1


def test_a_stamp_allocates_seq_in_call_order(out_dir: Path) -> None:
    stamp = OutDir.open(out_dir, project_id=PROJECT).stamp()

    envelopes = [stamp(at(n)) for n in range(3)]

    assert [envelope['seq'] for envelope in envelopes] == [1, 2, 3]
    assert envelopes[0] == {
        'seq': 1,
        'at': at(0),
        'project_id': PROJECT,
        'feed_version': FEED_VERSION,
    }
    assert heartbeat(stamp).seq == 4


def test_a_commit_appends_one_line_per_event_and_writes_state_and_cursor(out_dir: Path) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)
    events = three_events(out)

    out.commit(events, Cursor(events_id=12))

    assert feed_texts(out_dir) == tuple(event.model_dump_json() for event in events)
    assert (out_dir / 'feed.jsonl').read_text().endswith('\n')
    lines = feed_lines(out_dir)
    assert lines == events
    state = parse_state((out_dir / 'state.json').read_text())
    assert state == fold(lines, project_id=PROJECT)
    assert out.state == state
    cursor = stored_cursor(out_dir)
    assert cursor.through_seq == 3
    assert cursor.feed_bytes == (out_dir / 'feed.jsonl').stat().st_size
    assert cursor.events_id == 12
    assert out.cursor == cursor
    assert out.next_seq == 4
    assert validate_out_dir(out_dir).ok


def test_reopening_gives_the_same_state_and_cursor(out_dir: Path) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)
    out.commit(three_events(out), Cursor(events_id=12, tasks_updated_at='2026-10-21T12:00:00Z'))

    reopened = OutDir.open(out_dir, project_id=PROJECT)

    assert reopened.state == out.state
    assert reopened.cursor == out.cursor
    assert reopened.uncommitted == ()
    assert reopened.next_seq == 4


def _refused_seq(out_dir: Path, events: list[FeedEvent]) -> FeedIoError:
    out = OutDir.open(out_dir, project_id=PROJECT)
    before = (out_dir / 'feed.jsonl').read_bytes()
    with pytest.raises(FeedIoError) as caught:
        out.commit(events, Cursor())
    assert (out_dir / 'feed.jsonl').read_bytes() == before
    assert caught.value.problem is FeedIoProblem.SEQ_NOT_INCREASING
    assert 'seq_not_increasing' in str(caught.value)
    return caught.value


def test_a_seq_not_above_the_last_written_one_is_refused(out_dir: Path) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)
    out.commit(three_events(out), Cursor())

    refusal = _refused_seq(out_dir, [raw_heartbeat(3)])

    assert 'seq 3' in str(refusal)
    assert refusal.name == 'feed.jsonl'


def test_a_batch_whose_seqs_do_not_increase_is_refused(out_dir: Path) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)
    out.commit(three_events(out), Cursor())

    refusal = _refused_seq(out_dir, [raw_heartbeat(5), raw_heartbeat(4)])

    assert 'seq 4' in str(refusal)
    assert 'seq 5' in str(refusal)


def test_a_seq_gap_is_accepted(out_dir: Path) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)
    out.commit(three_events(out), Cursor())

    out.commit([raw_heartbeat(5)], Cursor())

    assert [line.seq for line in feed_lines(out_dir)] == [1, 2, 3, 5]
    assert out.next_seq == 6
    assert validate_out_dir(out_dir).ok


def test_an_event_of_another_project_is_refused_before_any_byte(out_dir: Path) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)

    with pytest.raises((ProjectMismatch, FeedIoError)):
        out.commit([raw_heartbeat(1, project_id='other')], Cursor())

    assert sorted(os.listdir(out_dir)) == []


def _supplementary_group(path: Path) -> int | None:
    current = path.stat().st_gid
    return next((group for group in os.getgroups() if group != current), None)


def test_written_files_are_group_readable_whatever_the_umask(out_dir: Path) -> None:
    group = _supplementary_group(out_dir)
    if group is not None:
        os.chown(out_dir, -1, group)
    os.chmod(out_dir, 0o2750)
    previous = os.umask(0o077)
    try:
        out = OutDir.open(out_dir, project_id=PROJECT)
        out.commit(three_events(out), Cursor())
        out.commit([heartbeat(out.stamp(), 1)], Cursor())
    finally:
        os.umask(previous)

    directory = out_dir.stat()
    assert directory.st_mode & stat.S_ISGID
    for name in WRITTEN:
        status = (out_dir / name).stat()
        assert status.st_mode & 0o777 == 0o640, name
        assert status.st_gid == directory.st_gid, name


def test_a_commit_leaves_only_the_three_files(out_dir: Path) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)
    out.commit(three_events(out), Cursor())
    out.commit([heartbeat(out.stamp(), 1)], Cursor())

    assert tuple(sorted(os.listdir(out_dir))) == WRITTEN
