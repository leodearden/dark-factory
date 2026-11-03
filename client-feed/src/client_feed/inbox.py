"""The tail of in/inbox.jsonl, the portal's reveal requests and defect reports to the publisher (PRD §7.4, §14.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from client_feed.contract import (
    INBOX_FILENAME,
    ContractViolation,
    DefectReported,
    InboxLine,
    RevealRequested,
    parse_inbox_line,
)
from client_feed.feed_io import FeedIoError, FeedIoProblem, FileTail, read_tail


@dataclass(frozen=True)
class RejectedInboxLine:
    """A complete inbox line that fails the inbox contract: where it starts, and why it was refused."""

    offset: int
    detail: str


@dataclass(frozen=True)
class InboxRead:
    """The valid lines among the inbox's complete lines past an offset, in inbox order, and the offset just after them."""

    lines: tuple[InboxLine, ...]
    rejected: tuple[RejectedInboxLine, ...]
    offset: int

    @property
    def requests(self) -> tuple[RevealRequested, ...]:
        """The reveal requests among the lines."""
        return tuple(line for line in self.lines if isinstance(line, RevealRequested))

    @property
    def reports(self) -> tuple[DefectReported, ...]:
        """The defect reports among the lines."""
        return tuple(line for line in self.lines if isinstance(line, DefectReported))


class InboxUnusable(FeedIoError):
    """An inbox that cannot be read this pass; its lines wait at the held offset."""


def _inbox_tail(in_dir: Path, offset: int) -> FileTail | None:
    try:
        return read_tail(in_dir / INBOX_FILENAME, offset=offset)
    except FeedIoError as exc:
        raise InboxUnusable(problem=exc.problem, name=exc.name, detail=exc.detail) from exc
    except OSError as exc:
        raise InboxUnusable(
            problem=FeedIoProblem.UNREADABLE,
            name=INBOX_FILENAME,
            detail=exc.strerror or type(exc).__name__,
        ) from exc


def _inbox_bytes(in_dir: Path, offset: int) -> bytes | None:
    """The inbox's bytes from `offset` to its end; None when there is no inbox."""
    tail = _inbox_tail(in_dir, offset)
    if tail is None:
        return None
    if offset > tail.size:
        raise InboxUnusable(
            problem=FeedIoProblem.INBOX_REWOUND,
            name=INBOX_FILENAME,
            detail=f'offset {offset} lies beyond its {tail.size} bytes; the portal replaced it',
        )
    return tail.data


def _line(raw: bytes, *, line_no: int) -> InboxLine:
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ContractViolation(line_no=line_no, detail=f'{INBOX_FILENAME} line is not UTF-8: {exc}') from exc
    return parse_inbox_line(text, line_no=line_no)


def read_inbox(in_dir: Path, *, offset: int) -> InboxRead:
    """The valid inbox lines past `offset`; a line off the contract is rejected, never raised, so it cannot stall the tail."""
    data = _inbox_bytes(in_dir, offset)
    if data is None:
        return InboxRead(lines=(), rejected=(), offset=offset)
    lines: list[InboxLine] = []
    rejected: list[RejectedInboxLine] = []
    position = offset
    for line_no, raw in enumerate(data[: data.rfind(b'\n') + 1].split(b'\n')[:-1], 1):
        try:
            lines.append(_line(raw, line_no=line_no))
        except ContractViolation as exc:
            rejected.append(RejectedInboxLine(offset=position, detail=exc.detail))
        position += len(raw) + 1
    return InboxRead(lines=tuple(lines), rejected=tuple(rejected), offset=position)
