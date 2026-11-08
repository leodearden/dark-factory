"""The landing datum: one Landing per task landed on main, read from runs.db and placed on git's line.

Contract and its α correction: plans/quality-dashboard-prd.md §Contract "Landing record".
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from shared.project_id import project_id_for_root
from shared.quality_series.git_lines import git_output

__all__ = [
    'CursorNotOnMain',
    'GitCrossCheck',
    'Landing',
    'LandingBatch',
    'LandingSource',
    'LinesBase',
    'landings',
    'parse_utc',
]


class LandingSource(StrEnum):
    MERGE_FINALIZED = 'merge_finalized'
    TRAIN_MERGED = 'train_merged'


class LinesBase(StrEnum):
    """What a landing's lines are measured against.

    FIRST_PARENT: the landing commit's first parent (a merge, or a first landing with no earlier
    one). PREVIOUS_LANDING: the nearest older first-parent commit that is itself a landing (a
    fast-forward landing, whose first parent is the task's own previous commit). UNMEASURABLE:
    the sha is not on main's first-parent line, or is a root commit.
    """

    FIRST_PARENT = 'first_parent'
    PREVIOUS_LANDING = 'previous_landing'
    UNMEASURABLE = 'unmeasurable'


class CursorNotOnMain(Exception):
    """The cursor sha is not a commit on main's first-parent line."""


def parse_utc(text: str) -> datetime:
    """A timezone-aware ISO timestamp; a naive one is refused rather than guessed."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f'a landing timestamp must carry its offset (timestamp={text!r})')
    return parsed


@dataclass(frozen=True)
class Landing:
    """One task landed on main; train members share merge_sha == unit_id and unit_size."""

    project_id: str
    merge_sha: str
    task_id: str
    landed_at: str
    ordinal: int
    source: LandingSource
    unit_id: str
    unit_size: int
    git_first_parent: bool
    prev_landed_at: str | None
    lines_base: LinesBase
    base_sha: str | None

    def __post_init__(self) -> None:
        where = f'task_id={self.task_id!r}, merge_sha={self.merge_sha!r}'
        if self.unit_size < 1:
            raise ValueError(f'a unit has at least one member (unit_size={self.unit_size!r}, {where})')
        if (self.base_sha is None) != (self.lines_base is LinesBase.UNMEASURABLE):
            raise ValueError(
                'base_sha is None exactly when lines are unmeasurable '
                f'(lines_base={self.lines_base!r}, base_sha={self.base_sha!r}, {where})'
            )
        parse_utc(self.landed_at)
        if self.prev_landed_at is not None:
            parse_utc(self.prev_landed_at)

    @property
    def landed_time(self) -> datetime:
        return parse_utc(self.landed_at)

    @property
    def prev_landed_time(self) -> datetime | None:
        return None if self.prev_landed_at is None else parse_utc(self.prev_landed_at)


@dataclass(frozen=True)
class GitCrossCheck:
    """Where runs.db and git disagree over the emitted range; served, never dropped."""

    off_first_parent: int
    merges_without_landing: int

    @property
    def disagreement(self) -> int:
        return self.off_first_parent + self.merges_without_landing


@dataclass(frozen=True)
class LandingBatch:
    landings: tuple[Landing, ...]
    git_check: GitCrossCheck


@dataclass(frozen=True)
class _Recorded:
    landed: datetime
    merge_sha: str
    position: int
    task_id: str
    source: LandingSource
    unit_size: int
    event_id: int

    @property
    def order(self) -> tuple[datetime, str, int, str]:
        return self.landed, self.merge_sha, self.position, self.task_id


_DONE_SQL = """
SELECT id, timestamp, task_id, json_extract(data, '$.merge_sha')
  FROM events
 WHERE event_type = 'merge_finalized'
   AND json_extract(data, '$.state') = 'done'
   AND json_extract(data, '$.merge_sha') IS NOT NULL
"""

_TRAIN_SQL = """
SELECT id, timestamp, json_extract(data, '$.merge_commit_sha'), data
  FROM events
 WHERE event_type = 'train_merged'
"""


def _train_members(event_id: int, data: str) -> list[str]:
    members = json.loads(data).get('member_task_ids')
    if not isinstance(members, list) or not members:
        raise ValueError(f'a train_merged event names its members (event id={event_id})')
    return [str(member) for member in members]


def _recorded_history(conn: sqlite3.Connection) -> list[_Recorded]:
    """Every landing in the store, deduplicated on (merge_sha, task_id) by earliest event."""
    found: list[_Recorded] = []
    for event_id, ts, task_id, sha in conn.execute(_DONE_SQL):
        found.append(_Recorded(parse_utc(ts), sha, 0, str(task_id),
                               LandingSource.MERGE_FINALIZED, 1, event_id))
    for event_id, ts, sha, data in conn.execute(_TRAIN_SQL):
        members = _train_members(event_id, data)
        found.extend(
            _Recorded(parse_utc(ts), sha, position, member, LandingSource.TRAIN_MERGED,
                      len(members), event_id)
            for position, member in enumerate(members)
        )
    earliest: dict[tuple[str, str], _Recorded] = {}
    for recorded in sorted(found, key=lambda r: (r.landed, r.event_id)):
        earliest.setdefault((recorded.merge_sha, recorded.task_id), recorded)
    return sorted(earliest.values(), key=lambda r: r.order)


@dataclass(frozen=True)
class _FirstParentLine:
    """main's first-parent commits oldest first, with each commit's parents."""

    shas: tuple[str, ...]
    parents: tuple[tuple[str, ...], ...]

    def position(self) -> dict[str, int]:
        return {sha: index for index, sha in enumerate(self.shas)}


def _first_parent_line(repo: str | os.PathLike[str], main_ref: str, timeout_s: float) -> _FirstParentLine:
    stdout = git_output(repo, ['rev-list', '--first-parent', '--parents', main_ref], timeout_s,
                        objects=(main_ref,))
    rows = [line.split() for line in reversed(stdout.decode().splitlines()) if line]
    return _FirstParentLine(tuple(row[0] for row in rows), tuple(tuple(row[1:]) for row in rows))


def _resolve_cursor(line: _FirstParentLine, cursor: str) -> int:
    matches = [index for index, sha in enumerate(line.shas) if sha.startswith(cursor)]
    if len(matches) != 1 or not cursor:
        raise CursorNotOnMain(
            f'cursor {cursor!r} names {len(matches)} commits on the first-parent line, not one'
        )
    return matches[0]


def _lines_bases(
    line: _FirstParentLine, landing_shas: set[str]
) -> dict[str, tuple[LinesBase, str | None]]:
    bases: dict[str, tuple[LinesBase, str | None]] = {}
    previous_landing: str | None = None
    for sha, parents in zip(line.shas, line.parents, strict=True):
        if len(parents) >= 2:
            bases[sha] = (LinesBase.FIRST_PARENT, parents[0])
        elif len(parents) == 1 and previous_landing is not None:
            bases[sha] = (LinesBase.PREVIOUS_LANDING, previous_landing)
        elif len(parents) == 1:
            bases[sha] = (LinesBase.FIRST_PARENT, parents[0])
        else:
            bases[sha] = (LinesBase.UNMEASURABLE, None)
        if sha in landing_shas:
            previous_landing = sha
    return bases


def landings(
    runs_db: sqlite3.Connection,
    repo: str | os.PathLike[str],
    since_cursor: str | None,
    *,
    project_id: str | None = None,
    main_ref: str = 'main',
    git_timeout_s: float = 60.0,
) -> LandingBatch:
    """The landings after ``since_cursor``, ordered by ordinal, with git's disagreement count.

    ``since_cursor=None`` means the full history: every landing in the store, ordered by
    landed_at then merge_sha (train members tied on both by their member_task_ids position).
    A sha cursor emits first-parent landings after it on main's first-parent line, and
    off-first-parent landings recorded after the newest landing at or below it; ordinals are
    numbered over the full history, so they are stable across calls. ``project_id`` defaults
    to the repo root's project id. The call runs one blocking ``git rev-list`` bounded by
    ``git_timeout_s``, so a coroutine calls it through ``asyncio.to_thread``. The return is a
    LandingBatch whose ``.landings`` are the rows.

    Raises CursorNotOnMain, and the git_lines errors (GitUnavailable for a non-repository).
    """
    project = project_id if project_id is not None else project_id_for_root(repo)
    line = _first_parent_line(repo, main_ref, git_timeout_s)
    position = line.position()
    history = _recorded_history(runs_db)
    bases = _lines_bases(line, {recorded.merge_sha for recorded in history})

    on_line = [position[r.merge_sha] for r in history if r.merge_sha in position]
    if since_cursor is None:
        range_start = min(on_line, default=len(line.shas))
        horizon: datetime | None = None
    else:
        cursor_position = _resolve_cursor(line, since_cursor)
        range_start = cursor_position + 1
        horizon = max(
            (r.landed for r in history
             if r.merge_sha in position and position[r.merge_sha] <= cursor_position),
            default=None,
        )

    def emitted(recorded: _Recorded) -> bool:
        if recorded.merge_sha in position:
            return position[recorded.merge_sha] >= range_start
        return horizon is None or recorded.landed > horizon

    rows: list[Landing] = []
    previous: dict[str, str] = {}
    for ordinal, recorded in enumerate(history):
        landed_at = recorded.landed.isoformat()
        lines_base, base_sha = bases.get(recorded.merge_sha, (LinesBase.UNMEASURABLE, None))
        landing = Landing(
            project_id=project,
            merge_sha=recorded.merge_sha,
            task_id=recorded.task_id,
            landed_at=landed_at,
            ordinal=ordinal,
            source=recorded.source,
            unit_id=recorded.merge_sha,
            unit_size=recorded.unit_size,
            git_first_parent=recorded.merge_sha in position,
            prev_landed_at=previous.get(recorded.task_id),
            lines_base=lines_base,
            base_sha=base_sha,
        )
        previous[recorded.task_id] = landed_at
        if emitted(recorded):
            rows.append(landing)

    landing_shas = {r.merge_sha for r in history}
    merges_without_landing = sum(
        1 for index in range(range_start, len(line.shas))
        if len(line.parents[index]) >= 2 and line.shas[index] not in landing_shas
    )
    off_first_parent = sum(1 for row in rows if not row.git_first_parent)
    return LandingBatch(tuple(rows), GitCrossCheck(off_first_parent, merges_without_landing))
