"""out_dir as durable storage with one writer: the feed appended, state.json and cursor.json replaced atomically (PRD §4.1-§4.3, §4.15)."""

from __future__ import annotations

import contextlib
import itertools
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, TypedDict

from pydantic import BaseModel, ConfigDict

from client_feed.contract import (
    FEED_FILENAME,
    FEED_VERSION,
    STATE_FILENAME,
    CommitSha,
    FeedEvent,
    FeedLine,
    FeedState,
    NonNegativeInt,
    parse_state,
)
from client_feed.fold import fold
from client_feed.sources.tree import LineCounts

CURSOR_FILENAME: Final = 'cursor.json'
FILE_MODE: Final = 0o640

_APPEND_FLAGS: Final = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC


class Envelope(TypedDict):
    """The envelope fields of one feed line, as keyword arguments to an event model."""

    seq: int
    at: datetime
    project_id: str
    feed_version: str


class Stamp:
    """The envelopes of one pass: seq allocated in call order from the first free seq."""

    def __init__(self, *, project_id: str, first_seq: int) -> None:
        self._project_id = project_id
        self._seqs = itertools.count(first_seq)

    def __call__(self, at: datetime) -> Envelope:
        return Envelope(
            seq=next(self._seqs), at=at, project_id=self._project_id, feed_version=FEED_VERSION
        )


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


class FeedIoProblem(StrEnum):
    SEQ_NOT_INCREASING = 'seq_not_increasing'


class FeedIoError(Exception):
    """A write or read of out_dir refused because it would break one of the feed's invariants."""

    def __init__(self, *, problem: FeedIoProblem, name: str, detail: str) -> None:
        self.problem = problem
        self.name = name
        self.detail = detail
        super().__init__(f'{name} refused ({problem}): {detail}')


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _append(path: Path, data: bytes) -> int:
    """Append `data` in one write sequence and fsync it; the file's size afterwards."""
    fd = os.open(path, _APPEND_FLAGS, FILE_MODE)
    try:
        os.fchmod(fd, FILE_MODE)
        _write_all(fd, data)
        os.fsync(fd)
        return os.fstat(fd).st_size
    finally:
        os.close(fd)


def _atomic_write(out_dir: Path, name: str, text: str) -> None:
    """Replace out_dir/name by a complete, fsynced file of mode FILE_MODE, or leave it as it was."""
    fd, temp = tempfile.mkstemp(dir=out_dir, prefix=f'.{name}.', suffix='.tmp')
    try:
        try:
            os.fchmod(fd, FILE_MODE)
            _write_all(fd, text.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, out_dir / name)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)
        raise


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None


def _feed_size(out_dir: Path) -> int:
    try:
        return os.lstat(out_dir / FEED_FILENAME).st_size
    except FileNotFoundError:
        return 0


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
        """out_dir's state and cursor as the last commit left them."""
        cursor_text = _read_text(out_dir / CURSOR_FILENAME)
        cursor = Cursor() if cursor_text is None else Cursor.model_validate_json(cursor_text)
        state_text = _read_text(out_dir / STATE_FILENAME)
        stored = None if state_text is None else parse_state(state_text)
        state = fold((), project_id=project_id, start=stored)
        return cls(
            out_dir,
            state=state,
            cursor=cursor,
            uncommitted=(),
            feed_bytes=_feed_size(out_dir),
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
