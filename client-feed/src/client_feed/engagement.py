"""in/engagement.json read as this project's engagement projection (PRD §4.14, §7.5)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    ContractViolation,
    EngagementProjection,
    FeedError,
    parse_engagement,
)
from client_feed.feed_io import FeedIoError, read_tail
from client_feed.sources.git import has_commit


class EngagementProblem(StrEnum):
    MISSING = 'missing'
    UNREADABLE = 'unreadable'
    INVALID = 'invalid'
    UNKNOWN_DELIVERED_COMMIT = 'unknown_delivered_commit'


class EngagementUnusable(Exception):
    """An engagement.json the publisher cannot apply, so reveals are disabled; its str names no absolute path."""

    def __init__(self, *, problem: EngagementProblem, detail: str) -> None:
        self.problem = problem
        self.detail = detail
        super().__init__(f'{ENGAGEMENT_FILENAME} is {problem}: {detail}; reveals are disabled')


class EngagementProjectMismatch(FeedError):
    """An engagement.json naming another project than the publisher's: its in_dir is wired to the wrong project."""

    def __init__(self, *, found: str, expected: str) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f'{ENGAGEMENT_FILENAME} names project_id {found!r}, expected {expected!r}'
        )


def _unreadable(detail: str) -> EngagementUnusable:
    return EngagementUnusable(problem=EngagementProblem.UNREADABLE, detail=detail)


def _engagement_text(in_dir: Path) -> str:
    try:
        tail = read_tail(in_dir / ENGAGEMENT_FILENAME, offset=0)
    except FeedIoError as exc:
        raise _unreadable(exc.detail) from exc
    except OSError as exc:
        raise _unreadable(exc.strerror or type(exc).__name__) from exc
    if tail is None:
        raise EngagementUnusable(problem=EngagementProblem.MISSING, detail='there is no such file')
    try:
        return tail.data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise _unreadable(f'not UTF-8 at byte {exc.start}: {exc.reason}') from exc


def read_engagement(root: Path, in_dir: Path, *, project_id: str) -> EngagementProjection:
    """This project's projection, its delivered commit one the repository holds; EngagementUnusable naming why not."""
    try:
        projection = parse_engagement(_engagement_text(in_dir))
    except ContractViolation as exc:
        raise EngagementUnusable(problem=EngagementProblem.INVALID, detail=exc.detail) from exc
    if projection.project_id != project_id:
        raise EngagementProjectMismatch(found=projection.project_id, expected=project_id)
    delivered = projection.delivered_commit
    if delivered is not None and not has_commit(root, delivered):
        raise EngagementUnusable(
            problem=EngagementProblem.UNKNOWN_DELIVERED_COMMIT,
            detail=f'delivered_commit {delivered} is no commit of this repository',
        )
    return projection
