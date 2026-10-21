"""out/ as durable storage with one writer: append, fold, atomic state and cursor (PRD §4.1-§4.3, §4.15)."""

import os
import stat
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from _git_repo import make_repo

from client_feed.contract import (
    FEED_VERSION,
    ContractViolation,
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
from client_feed.policy import OutDirRefused, PolicyError, PolicyProblem, load_policy
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


def _contents(directory: Path) -> dict[str, str | bytes]:
    """Every entry under `directory`: a symlink as its target text, a file as its bytes."""
    return {
        str(path.relative_to(directory)): (
            f'-> {os.readlink(path)}' if path.is_symlink() else path.read_bytes()
        )
        for path in sorted(directory.rglob('*'))
        if path.is_symlink() or path.is_file()
    }


def _refused_out_dir(path: Path) -> OutDirRefused:
    with pytest.raises(OutDirRefused) as caught:
        OutDir.open(path, project_id=PROJECT)
    assert str(path) in str(caught.value)
    return caught.value


@pytest.mark.skipif(os.geteuid() == 0, reason='root owns every directory it could be refused')
def test_an_out_dir_owned_by_another_user_is_refused() -> None:
    assert _refused_out_dir(Path('/')).problem is PolicyProblem.OUT_DIR_NOT_OWNED


def test_an_out_dir_that_is_a_symlink_is_refused(tmp_path: Path, out_dir: Path) -> None:
    link = tmp_path / 'linked-out'
    link.symlink_to(out_dir)

    assert _refused_out_dir(link).problem is PolicyProblem.OUT_DIR_UNUSABLE
    assert _contents(out_dir) == {}


def test_a_missing_out_dir_is_refused(tmp_path: Path) -> None:
    missing = tmp_path / 'absent'

    assert _refused_out_dir(missing).problem is PolicyProblem.OUT_DIR_UNUSABLE
    assert not missing.exists()


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    path = tmp_path / 'elsewhere' / 'target.txt'
    path.parent.mkdir()
    path.write_text('not the publisher\'s\n')
    return path


def _assert_symlink_refusal(caught: pytest.ExceptionInfo[FeedIoError], name: str) -> None:
    assert caught.value.problem is FeedIoProblem.SYMLINK
    assert caught.value.name == name
    assert name in str(caught.value)
    assert 'symlink' in str(caught.value)


def test_a_symlinked_feed_is_refused_and_its_target_unchanged(
    tmp_path: Path, out_dir: Path, outside: Path
) -> None:
    (out_dir / 'feed.jsonl').symlink_to(outside)
    before = _contents(tmp_path)

    with pytest.raises(FeedIoError) as caught:
        out = OutDir.open(out_dir, project_id=PROJECT)
        out.commit(three_events(out), Cursor())

    _assert_symlink_refusal(caught, 'feed.jsonl')
    assert _contents(tmp_path) == before


@pytest.mark.parametrize('name', ['state.json', 'cursor.json'])
def test_a_symlinked_state_or_cursor_is_refused_at_open(
    tmp_path: Path, out_dir: Path, outside: Path, name: str
) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)
    out.commit(three_events(out), Cursor())
    (out_dir / name).unlink()
    (out_dir / name).symlink_to(outside)
    before = _contents(tmp_path)

    with pytest.raises(FeedIoError) as caught:
        OutDir.open(out_dir, project_id=PROJECT)

    _assert_symlink_refusal(caught, name)
    assert _contents(tmp_path) == before


def test_a_symlink_planted_after_open_is_refused_not_replaced(
    out_dir: Path, outside: Path
) -> None:
    out = OutDir.open(out_dir, project_id=PROJECT)
    out.commit(three_events(out), Cursor())
    (out_dir / 'state.json').unlink()
    (out_dir / 'state.json').symlink_to(outside)
    target_before = outside.read_bytes()

    with pytest.raises(FeedIoError) as caught:
        out.commit([heartbeat(out.stamp(), 1)], Cursor())

    _assert_symlink_refusal(caught, 'state.json')
    assert (out_dir / 'state.json').is_symlink()
    assert os.readlink(out_dir / 'state.json') == str(outside)
    assert outside.read_bytes() == target_before
    assert not any(name.endswith('.tmp') for name in os.listdir(out_dir))


def _policy_config(tmp_path: Path, out_dir: Path) -> Path:
    root = tmp_path / 'root'
    repo = make_repo(root)
    repo.write('src/a.py', 'a = 1\n')
    repo.commit('one', when=EPOCH)
    config = tmp_path / 'policy.yaml'
    config.write_text(
        yaml.safe_dump(
            {
                'project_id': PROJECT,
                'project_root': str(root),
                'out_dir': str(out_dir),
                'in_dir': str(tmp_path / 'in'),
                'visible_globs': ['src/**'],
            }
        )
    )
    return config


def _linked_out(tmp_path: Path) -> Path:
    real = tmp_path / 'real-out'
    real.mkdir()
    link = tmp_path / 'linked-out'
    link.symlink_to(real)
    return link


@pytest.mark.parametrize(
    ('out_dir_of', 'problem'),
    [
        pytest.param(
            lambda tmp_path: Path('/'),
            PolicyProblem.OUT_DIR_NOT_OWNED,
            marks=pytest.mark.skipif(os.geteuid() == 0, reason='root owns every directory'),
            id='not-owned',
        ),
        pytest.param(_linked_out, PolicyProblem.OUT_DIR_UNUSABLE, id='symlink'),
        pytest.param(lambda tmp_path: tmp_path / 'absent', PolicyProblem.OUT_DIR_UNUSABLE, id='missing'),
    ],
)
def test_load_policy_and_the_writer_share_one_out_dir_check(
    tmp_path: Path, out_dir_of: Callable[[Path], Path], problem: PolicyProblem
) -> None:
    out_dir = out_dir_of(tmp_path)
    config = _policy_config(tmp_path, out_dir)

    with pytest.raises(PolicyError) as policy_refusal:
        load_policy(config)
    with pytest.raises(OutDirRefused) as writer_refusal:
        OutDir.open(out_dir, project_id=PROJECT)

    assert policy_refusal.value.problem is problem
    assert writer_refusal.value.problem is problem
    assert writer_refusal.value.detail in str(policy_refusal.value)


def _committed(out_dir: Path, *batches: Callable[[Stamp], list[FeedEvent]]) -> list[FeedEvent]:
    """Each batch built with a fresh stamp and committed; every event written, in order."""
    written: list[FeedEvent] = []
    for batch in batches:
        out = OutDir.open(out_dir, project_id=PROJECT)
        events = batch(out.stamp())
        out.commit(events, Cursor())
        written.extend(events)
    return written


def first_batch(stamp: Stamp) -> list[FeedEvent]:
    return [task(stamp, 7), factory(stamp), heartbeat(stamp)]


def second_batch(stamp: Stamp) -> list[FeedEvent]:
    return [task(stamp, 8, 1), heartbeat(stamp, 1)]


def _files(out_dir: Path) -> dict[str, bytes]:
    return {name: (out_dir / name).read_bytes() for name in sorted(os.listdir(out_dir))}


def test_a_torn_final_line_is_cut_and_logged(
    out_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _committed(out_dir, first_batch)
    complete = (out_dir / 'feed.jsonl').read_bytes()
    torn = b'{"seq":9,"at"'
    with (out_dir / 'feed.jsonl').open('ab') as feed:
        feed.write(torn)

    with caplog.at_level('WARNING', logger='client_feed.feed_io'):
        out = OutDir.open(out_dir, project_id=PROJECT)

    assert (out_dir / 'feed.jsonl').read_bytes() == complete
    assert out.next_seq == 4
    assert [record.name for record in caplog.records] == ['client_feed.feed_io']
    assert f'{len(torn)} bytes' in caplog.records[0].getMessage()
    assert validate_out_dir(out_dir).ok


def _uncommit_second_batch(out_dir: Path, *, stale_state: bool) -> list[FeedEvent]:
    first = _committed(out_dir, first_batch)
    saved = {name: (out_dir / name).read_bytes() for name in ('cursor.json', 'state.json')}
    second = _committed(out_dir, second_batch)
    (out_dir / 'cursor.json').write_bytes(saved['cursor.json'])
    if stale_state:
        (out_dir / 'state.json').write_bytes(saved['state.json'])
    return first + second


@pytest.mark.parametrize('stale_state', [False, True], ids=['current-state', 'stale-state'])
def test_lines_past_the_cursor_are_the_uncommitted_tail(out_dir: Path, stale_state: bool) -> None:
    written = _uncommit_second_batch(out_dir, stale_state=stale_state)

    out = OutDir.open(out_dir, project_id=PROJECT)

    assert out.uncommitted == tuple(written[3:])
    assert out.state == fold(written, project_id=PROJECT)
    assert out.cursor.through_seq == 3
    assert out.next_seq == 6


def test_a_deleted_state_is_refolded_from_the_feed(out_dir: Path) -> None:
    written = _committed(out_dir, first_batch, second_batch)
    (out_dir / 'state.json').unlink()

    out = OutDir.open(out_dir, project_id=PROJECT)

    assert out.state == fold(feed_lines(out_dir), project_id=PROJECT)
    assert out.state == fold(written, project_id=PROJECT)
    assert out.uncommitted == ()


def test_a_deleted_cursor_makes_the_whole_feed_uncommitted(out_dir: Path) -> None:
    written = _committed(out_dir, first_batch, second_batch)
    (out_dir / 'cursor.json').unlink()

    out = OutDir.open(out_dir, project_id=PROJECT)

    assert out.cursor == Cursor()
    assert out.uncommitted == tuple(written)
    assert out.state == fold(written, project_id=PROJECT)
    assert out.next_seq == 6


def _cursor_beyond_the_feed(out_dir: Path) -> tuple[str, ...]:
    size = (out_dir / 'feed.jsonl').stat().st_size
    (out_dir / 'cursor.json').write_text(Cursor(through_seq=5, feed_bytes=size + 10).model_dump_json())
    return (f'feed_bytes {size + 10}', str(size))


def _cursor_behind_its_seq(out_dir: Path) -> tuple[str, ...]:
    size = (out_dir / 'feed.jsonl').stat().st_size
    (out_dir / 'cursor.json').write_text(Cursor(through_seq=4, feed_bytes=size).model_dump_json())
    return ('through_seq 4', 'seq 5')


def _cursor_mid_line(out_dir: Path) -> tuple[str, ...]:
    size = (out_dir / 'feed.jsonl').stat().st_size
    (out_dir / 'cursor.json').write_text(Cursor(through_seq=5, feed_bytes=size - 3).model_dump_json())
    return (f'feed_bytes {size - 3}',)


@pytest.mark.parametrize(
    'misplace', [_cursor_beyond_the_feed, _cursor_behind_its_seq, _cursor_mid_line],
    ids=['beyond-the-feed', 'seq-mismatch', 'mid-line'],
)
def test_a_cursor_ahead_of_the_feed_is_refused_untouched(
    out_dir: Path, misplace: Callable[[Path], tuple[str, ...]]
) -> None:
    _committed(out_dir, first_batch, second_batch)
    named = misplace(out_dir)
    before = _files(out_dir)

    with pytest.raises(FeedIoError) as caught:
        OutDir.open(out_dir, project_id=PROJECT)

    assert caught.value.problem is FeedIoProblem.CURSOR_AHEAD
    assert caught.value.name == 'cursor.json'
    for value in named:
        assert value in str(caught.value)
    assert _files(out_dir) == before


def test_a_malformed_complete_line_in_the_tail_is_refused_untouched(out_dir: Path) -> None:
    _committed(out_dir, first_batch)
    with (out_dir / 'feed.jsonl').open('ab') as feed:
        feed.write(b'{"seq": 4, "not": "a feed line"}\n')
    before = _files(out_dir)

    with pytest.raises(ContractViolation):
        OutDir.open(out_dir, project_id=PROJECT)

    assert _files(out_dir) == before


def test_a_state_of_another_project_is_refused_untouched(out_dir: Path) -> None:
    _committed(out_dir, first_batch)
    (out_dir / 'state.json').write_text(fold([], project_id='other').model_dump_json())
    before = _files(out_dir)

    with pytest.raises(ProjectMismatch):
        OutDir.open(out_dir, project_id=PROJECT)

    assert _files(out_dir) == before


def test_leftover_temp_files_of_an_interrupted_write_are_removed(out_dir: Path) -> None:
    _committed(out_dir, first_batch)
    (out_dir / '.state.json.k3j_x9.tmp').write_text('{"half": ')
    (out_dir / '.cursor.json.a1b2c3.tmp').write_text('')

    OutDir.open(out_dir, project_id=PROJECT)

    assert tuple(sorted(os.listdir(out_dir))) == WRITTEN
