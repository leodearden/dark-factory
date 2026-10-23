"""The inbox tail: complete reveal-request lines past an offset, a bad line rejected, a rewind refused (PRD §7.4)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from client_feed.contract import RevealRequested
from client_feed.feed_io import FeedIoError, FeedIoProblem
from client_feed.inbox import read_inbox

EPOCH = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return EPOCH + timedelta(minutes=minutes)


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    path = tmp_path / 'elsewhere' / 'target.txt'
    path.parent.mkdir()
    path.write_text('not the publisher\'s\n')
    return path


def request_line(seq: int, path: str = 'core/src/a.py') -> bytes:
    request = RevealRequested(seq=seq, at=at(seq), path=path)
    return f'{request.model_dump_json()}\n'.encode()


@pytest.fixture
def in_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'in'
    path.mkdir()
    return path


def test_the_inbox_is_read_to_its_last_complete_line(in_dir: Path) -> None:
    first, second, third = request_line(1), request_line(2, 'core/src/b.py'), request_line(3)
    inbox = in_dir / 'inbox.jsonl'
    inbox.write_bytes(first + second + third[:10])

    read = read_inbox(in_dir, offset=0)

    assert [request.seq for request in read.requests] == [1, 2]
    assert read.requests[1].path == 'core/src/b.py'
    assert read.rejected == ()
    assert read.offset == len(first + second)

    inbox.write_bytes(first + second + third)
    again = read_inbox(in_dir, offset=read.offset)

    assert [request.seq for request in again.requests] == [3]
    assert again.offset == len(first + second + third)


def test_a_malformed_inbox_line_is_rejected_by_offset_and_the_tail_moves_on(in_dir: Path) -> None:
    first, bad, last = request_line(1), b'{"seq": 2, "type": "reveal_requested"}\n', request_line(3)
    (in_dir / 'inbox.jsonl').write_bytes(first + bad + last)

    read = read_inbox(in_dir, offset=0)

    assert [request.seq for request in read.requests] == [1, 3]
    assert len(read.rejected) == 1
    assert read.rejected[0].offset == len(first)
    assert 'at' in read.rejected[0].detail
    assert read.offset == len(first + bad + last)


def test_a_line_that_is_not_utf8_is_rejected_by_offset(in_dir: Path) -> None:
    first, bad, last = request_line(1), b'{"seq": 2, "path": "\xff"}\n', request_line(3)
    (in_dir / 'inbox.jsonl').write_bytes(first + bad + last)

    read = read_inbox(in_dir, offset=0)

    assert [request.seq for request in read.requests] == [1, 3]
    assert [(line.offset, 'UTF-8' in line.detail) for line in read.rejected] == [(len(first), True)]
    assert read.offset == len(first + bad + last)


def test_a_missing_inbox_or_in_dir_has_no_requests(tmp_path: Path, in_dir: Path) -> None:
    for directory in (in_dir, tmp_path / 'absent'):
        read = read_inbox(directory, offset=7)

        assert (read.requests, read.rejected, read.offset) == ((), (), 7)


def test_a_symlinked_inbox_is_refused(in_dir: Path, outside: Path) -> None:
    (in_dir / 'inbox.jsonl').symlink_to(outside)

    with pytest.raises(FeedIoError) as caught:
        read_inbox(in_dir, offset=0)

    assert caught.value.problem is FeedIoProblem.SYMLINK
    assert caught.value.name == 'inbox.jsonl'
    assert 'symlink' in str(caught.value)


def test_an_offset_beyond_the_inbox_is_a_rewound_inbox(in_dir: Path) -> None:
    (in_dir / 'inbox.jsonl').write_bytes(request_line(1))
    size = (in_dir / 'inbox.jsonl').stat().st_size

    with pytest.raises(FeedIoError) as caught:
        read_inbox(in_dir, offset=size + 50)

    assert caught.value.problem is FeedIoProblem.INBOX_REWOUND
    assert caught.value.name == 'inbox.jsonl'
    assert f'offset {size + 50}' in str(caught.value)
    assert f'{size} bytes' in str(caught.value)
