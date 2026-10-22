"""out_dir as durable storage with one writer: the feed appended, state.json and cursor.json replaced atomically (PRD §4.1-§4.3, §4.15)."""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import stat
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from client_feed.contract import (
    FEED_FILENAME,
    FEED_MAJOR,
    STATE_FILENAME,
    CommitSha,
    ContractViolation,
    FeedEvent,
    FeedLine,
    FeedState,
    NonNegativeInt,
    UnsupportedFeedVersion,
    feed_major,
    parse_feed_line,
    parse_state,
    split_jsonl,
)
from client_feed.envelope import Stamp
from client_feed.fold import fold
from client_feed.policy import require_usable_out_dir
from client_feed.sources.tree import LineCounts

CURSOR_FILENAME: Final = 'cursor.json'
FILE_MODE: Final = 0o640

_APPEND_FLAGS: Final = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
_TRUNCATE_FLAGS: Final = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
_REPLACED: Final = (STATE_FILENAME, CURSOR_FILENAME)
_TEMP_SUFFIX: Final = '.tmp'
_SCAN_CHUNK: Final = 1 << 16

_log = logging.getLogger(__name__)


class Cursor(BaseModel):
    """cursor.json: where the committed feed ends and how far each source has been published."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    through_seq: NonNegativeInt = 0
    feed_bytes: NonNegativeInt = 0
    events_id: NonNegativeInt = 0
    tasks_updated_at: str | None = None
    head: CommitSha | None = None
    inbox_offset: NonNegativeInt = 0
    line_counts: LineCounts | None = None
    reveals_head: CommitSha | None = None


class FeedIoProblem(StrEnum):
    SEQ_NOT_INCREASING = 'seq_not_increasing'
    SYMLINK = 'symlink'
    CURSOR_AHEAD = 'cursor_ahead'
    ROTATION_TARGET_EXISTS = 'rotation_target_exists'
    INBOX_REWOUND = 'inbox_rewound'
    UNREADABLE = 'unreadable'


class FeedIoError(Exception):
    """A write or read of out_dir, or a read of in_dir, refused because it would break one of the feed's invariants."""

    def __init__(self, *, problem: FeedIoProblem, name: str, detail: str) -> None:
        self.problem = problem
        self.name = name
        self.detail = detail
        super().__init__(f'{name} refused ({problem}): {detail}')


def _symlink_refused(name: str) -> FeedIoError:
    return FeedIoError(
        problem=FeedIoProblem.SYMLINK,
        name=name,
        detail=f'{name} is a symlink; the writer never follows or replaces one',
    )


@contextlib.contextmanager
def _opened(path: Path, flags: int, mode: int = FILE_MODE) -> Iterator[int]:
    """An fd on `path`, opened with flags that carry O_NOFOLLOW; a symlink is refused, never followed."""
    try:
        fd = os.open(path, flags, mode)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _symlink_refused(path.name) from exc
        raise
    try:
        yield fd
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(fd, 1 << 20):
        chunks.append(chunk)
    return b''.join(chunks)


def _append(path: Path, data: bytes) -> int:
    """Append `data` in one write sequence and fsync it; the file's size afterwards."""
    with _opened(path, _APPEND_FLAGS) as fd:
        os.fchmod(fd, FILE_MODE)
        _write_all(fd, data)
        os.fsync(fd)
        return os.fstat(fd).st_size


def _require_not_symlink(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        if stat.S_ISLNK(os.lstat(path).st_mode):
            raise _symlink_refused(path.name)


def _temp_prefix(name: str) -> str:
    return f'.{name}.'


def _atomic_write(out_dir: Path, name: str, text: str) -> None:
    """Replace out_dir/name by a complete, fsynced file of mode FILE_MODE, or leave it as it was."""
    fd, temp = tempfile.mkstemp(dir=out_dir, prefix=_temp_prefix(name), suffix=_TEMP_SUFFIX)
    try:
        try:
            os.fchmod(fd, FILE_MODE)
            _write_all(fd, text.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        _require_not_symlink(out_dir / name)
        os.replace(temp, out_dir / name)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)
        raise


def _decoded(data: bytes, name: str) -> str:
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ContractViolation(line_no=None, detail=f'{name} is not UTF-8: {exc}') from exc


def _read_text(path: Path) -> str | None:
    try:
        with _opened(path, _READ_FLAGS) as fd:
            data = _read_all(fd)
    except FileNotFoundError:
        return None
    return _decoded(data, path.name)


@dataclass(frozen=True)
class FileTail:
    """A file's size when it was opened, and its bytes from the offset it was read from."""

    size: int
    data: bytes


def read_tail(path: Path, *, offset: int) -> FileTail | None:
    """`path` read from `offset` through one fd that refuses a symlink; None when there is no file."""
    try:
        with _opened(path, _READ_FLAGS) as fd:
            size = os.fstat(fd).st_size
            os.lseek(fd, offset, os.SEEK_SET)
            return FileTail(size=size, data=_read_all(fd))
    except FileNotFoundError:
        return None


def read_feed_lines(out_dir: Path, *, from_byte: int = 0) -> tuple[FeedLine, ...]:
    """The complete lines of out_dir's feed from `from_byte`, numbered from 1 there; a partial last line is left unread."""
    with _opened(out_dir / FEED_FILENAME, _READ_FLAGS) as fd:
        os.lseek(fd, from_byte, os.SEEK_SET)
        data = _read_all(fd)
    texts = split_jsonl(_decoded(data[: data.rfind(b'\n') + 1], FEED_FILENAME))
    return tuple(parse_feed_line(text, line_no=line_no) for line_no, text in enumerate(texts, 1))


def _lines_if_any(out_dir: Path, from_byte: int) -> tuple[FeedLine, ...]:
    try:
        return read_feed_lines(out_dir, from_byte=from_byte)
    except FileNotFoundError:
        return ()


def _line_start(fd: int, end: int) -> int:
    """The offset just past the last newline before `end`, or 0 when there is none."""
    position = end
    while position > 0:
        start = max(0, position - _SCAN_CHUNK)
        newline = os.pread(fd, position - start, start).rfind(b'\n')
        if newline != -1:
            return start + newline + 1
        position = start
    return 0


def _feed_extent(out_dir: Path) -> tuple[int, int]:
    """The feed's size and the end of its last complete line; (0, 0) when there is no feed."""
    try:
        with _opened(out_dir / FEED_FILENAME, _READ_FLAGS) as fd:
            size = os.fstat(fd).st_size
            return size, _line_start(fd, size)
    except FileNotFoundError:
        return 0, 0


def _first_line(out_dir: Path) -> str | None:
    """The feed's first complete line, or None when it has none."""
    buffer = bytearray()
    try:
        with _opened(out_dir / FEED_FILENAME, _READ_FLAGS) as fd:
            while (newline := buffer.find(b'\n')) == -1:
                chunk = os.read(fd, _SCAN_CHUNK)
                if not chunk:
                    return None
                buffer += chunk
    except FileNotFoundError:
        return None
    return _decoded(bytes(buffer[:newline]), FEED_FILENAME)


def _exists(path: Path) -> bool:
    """Whether anything, a dangling symlink included, is at `path`."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _major_on_disk(out_dir: Path) -> int:
    text = _first_line(out_dir)
    if text is None:
        return FEED_MAJOR
    try:
        parse_feed_line(text, line_no=1)
    except UnsupportedFeedVersion as exc:
        return feed_major(exc.found)
    return FEED_MAJOR


def _rotate_another_major(out_dir: Path) -> None:
    """Move a feed of another major aside, dropping its cursor first, so the new feed starts from empty cursors."""
    major = _major_on_disk(out_dir)
    if major == FEED_MAJOR:
        return
    rotated = f'feed.{major}.jsonl'
    if _exists(out_dir / rotated):
        raise FeedIoError(
            problem=FeedIoProblem.ROTATION_TARGET_EXISTS,
            name=rotated,
            detail=f'{FEED_FILENAME} of major {major} cannot be rotated to {rotated}, which exists',
        )
    for name in (CURSOR_FILENAME, STATE_FILENAME):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(out_dir / name)
    os.rename(out_dir / FEED_FILENAME, out_dir / rotated)
    with _opened(out_dir / rotated, _READ_FLAGS) as fd:
        os.fchmod(fd, FILE_MODE)
    _log.info('rotated %s of major %d to %s', FEED_FILENAME, major, rotated)


def _is_leftover_temp(entry: str) -> bool:
    return entry.endswith(_TEMP_SUFFIX) and any(
        entry.startswith(_temp_prefix(name)) for name in _REPLACED
    )


def _remove_leftover_temps(out_dir: Path) -> None:
    for entry in os.listdir(out_dir):
        if _is_leftover_temp(entry):
            os.unlink(out_dir / entry)


def _stored_cursor(out_dir: Path) -> Cursor:
    text = _read_text(out_dir / CURSOR_FILENAME)
    if text is None:
        return Cursor()
    try:
        return Cursor.model_validate_json(text)
    except ValidationError as exc:
        _log.warning('%s is not a cursor (%s); every source restarts from the beginning', CURSOR_FILENAME, exc)
        return Cursor()


def _cursor_ahead(detail: str) -> FeedIoError:
    return FeedIoError(problem=FeedIoProblem.CURSOR_AHEAD, name=CURSOR_FILENAME, detail=detail)


def _cursor_line_start(out_dir: Path, cursor: Cursor, complete_end: int) -> int:
    """Where the committed line ending at cursor.feed_bytes starts; refused unless a complete line ends there."""
    end = cursor.feed_bytes
    if end > complete_end:
        raise _cursor_ahead(
            f'cursor.json records feed_bytes {end} but {FEED_FILENAME} holds {complete_end}'
            ' bytes of complete lines'
        )
    if end == 0:
        return 0
    with _opened(out_dir / FEED_FILENAME, _READ_FLAGS) as fd:
        if os.pread(fd, 1, end - 1) != b'\n':
            raise _cursor_ahead(f'cursor.json records feed_bytes {end}, which is not the end of a feed line')
        return _line_start(fd, end - 1)


def _uncommitted_tail(out_dir: Path, cursor: Cursor, line_start: int) -> tuple[FeedLine, ...]:
    """The lines past the cursor, once the line ending at feed_bytes is shown to carry through_seq."""
    lines = _lines_if_any(out_dir, line_start)
    at_cursor, tail = (lines[0].seq, lines[1:]) if cursor.feed_bytes else (0, lines)
    if at_cursor != cursor.through_seq:
        raise _cursor_ahead(
            f'cursor.json records through_seq {cursor.through_seq} but the line ending at'
            f' feed_bytes {cursor.feed_bytes} carries seq {at_cursor}'
        )
    return tail


def _stored_state(out_dir: Path, project_id: str) -> FeedState | None:
    """state.json when it is a state of this project; None when it is absent or not a state."""
    text = _read_text(out_dir / STATE_FILENAME)
    if text is None:
        return None
    try:
        stored = parse_state(text)
    except ContractViolation as exc:
        _log.warning('%s is not a state (%s); it will be refolded', STATE_FILENAME, exc)
        return None
    return fold((), project_id=project_id, start=stored)


def _recovered_state(
    out_dir: Path, project_id: str, cursor: Cursor, tail: tuple[FeedLine, ...]
) -> FeedState:
    """The fold of every complete line: state.json plus the tail past it, else the whole feed refolded."""
    stored = _stored_state(out_dir, project_id)
    last_seq = tail[-1].seq if tail else cursor.through_seq
    if stored is not None and cursor.through_seq <= stored.watermark_seq <= last_seq:
        past = (line for line in tail if line.seq > stored.watermark_seq)
        return fold(past, project_id=project_id, start=stored)
    lines = _lines_if_any(out_dir, 0)
    if lines:
        why = 'absent' if stored is None else f'at watermark {stored.watermark_seq}'
        _log.warning(
            '%s %s, outside [%d, %d]: refolding the %d lines of %s',
            STATE_FILENAME, why, cursor.through_seq, last_seq, len(lines), FEED_FILENAME,
        )
    return fold(lines, project_id=project_id)


def _cut_torn_tail(out_dir: Path, size: int, complete_end: int) -> None:
    if size == complete_end:
        return
    with _opened(out_dir / FEED_FILENAME, _TRUNCATE_FLAGS) as fd:
        os.ftruncate(fd, complete_end)
        os.fsync(fd)
    _log.warning(
        '%s: cut a torn final line of %d bytes at byte %d',
        FEED_FILENAME, size - complete_end, complete_end,
    )


def _require_increasing(events: Sequence[FeedEvent], last_seq: int) -> None:
    previous = last_seq
    for event in events:
        if event.seq <= previous:
            raise FeedIoError(
                problem=FeedIoProblem.SEQ_NOT_INCREASING,
                name=FEED_FILENAME,
                detail=f'seq {event.seq} follows seq {previous}; each seq must be above the one before',
            )
        previous = event.seq


class OutDir:
    """One writer's out_dir: the folded state, the cursor, and the only path that appends to the feed."""

    def __init__(
        self,
        path: Path,
        *,
        state: FeedState,
        cursor: Cursor,
        uncommitted: tuple[FeedLine, ...],
        feed_bytes: int,
    ) -> None:
        self._path = path
        self._state = state
        self._cursor = cursor
        self._uncommitted = uncommitted
        self._feed_bytes = feed_bytes

    @classmethod
    def open(cls, out_dir: Path, *, project_id: str) -> OutDir:
        """out_dir recovered: every complete line folded, lines past the cursor uncommitted, a torn tail cut last."""
        require_usable_out_dir(out_dir)
        _remove_leftover_temps(out_dir)
        _rotate_another_major(out_dir)
        size, complete_end = _feed_extent(out_dir)
        cursor = _stored_cursor(out_dir)
        tail = _uncommitted_tail(out_dir, cursor, _cursor_line_start(out_dir, cursor, complete_end))
        state = _recovered_state(out_dir, project_id, cursor, tail)
        _cut_torn_tail(out_dir, size, complete_end)
        return cls(
            out_dir, state=state, cursor=cursor, uncommitted=tail, feed_bytes=complete_end
        )

    @property
    def state(self) -> FeedState:
        return self._state

    @property
    def cursor(self) -> Cursor:
        return self._cursor

    @property
    def uncommitted(self) -> tuple[FeedLine, ...]:
        """Feed lines appended by a writer that stopped before recording them in cursor.json."""
        return self._uncommitted

    @property
    def next_seq(self) -> int:
        return self._state.watermark_seq + 1

    def stamp(self) -> Stamp:
        return Stamp(project_id=self._state.project_id, first_seq=self.next_seq)

    def commit(self, events: Sequence[FeedEvent], cursor: Cursor) -> None:
        """Append `events`, then replace state.json, then cursor.json; through_seq and feed_bytes are the writer's."""
        _require_increasing(events, self._state.watermark_seq)
        state = fold(events, project_id=self._state.project_id, start=self._state)
        feed_bytes = self._feed_bytes
        if events:
            data = ''.join(f'{event.model_dump_json()}\n' for event in events).encode()
            feed_bytes = _append(self._path / FEED_FILENAME, data)
            _atomic_write(self._path, STATE_FILENAME, state.model_dump_json())
        committed = cursor.model_copy(
            update={'through_seq': state.watermark_seq, 'feed_bytes': feed_bytes}
        )
        _atomic_write(self._path, CURSOR_FILENAME, committed.model_dump_json())
        self._state = state
        self._cursor = committed
        self._uncommitted = ()
        self._feed_bytes = feed_bytes
