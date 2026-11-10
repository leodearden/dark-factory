"""The landing datum: one Landing per task landed on main, read from runs.db and placed on git's line.

Contract and its α correction: plans/quality-dashboard-prd.md §Contract "Landing record".
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, replace
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
    'UnrecognisedTrainEvent',
    'landings',
    'parse_utc',
    'recorded_landing_keys',
]


class LandingSource(StrEnum):
    """Which runs.db event recorded the landing.

    TRAIN_MERGED: a coalesced train's member (``$.member_task_ids``), sharing its train's merge
    commit. TRAIN_ATTRIBUTED: a train member that passed solo after the train's union verify
    failed (``$.verdict = 'attributed'``); it advanced main alone, so it is a unit of one.
    """

    MERGE_FINALIZED = 'merge_finalized'
    TRAIN_MERGED = 'train_merged'
    TRAIN_ATTRIBUTED = 'train_attributed'


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


class UnrecognisedTrainEvent(Exception):
    """A train_merged event carries neither the coalesced nor the attributed payload."""

    def __init__(self, event_ids: list[int]) -> None:
        self.event_ids = tuple(event_ids)
        super().__init__(
            'train_merged events with neither member_task_ids + merge_commit_sha nor '
            f"verdict='attributed' + merge_sha (event ids={list(self.event_ids)})"
        )


def parse_utc(text: str) -> datetime:
    """A runs.db timestamp as a timezone-aware datetime; a naive one is refused rather than guessed."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f'a runs.db timestamp must carry its offset (timestamp={text!r})')
    return parsed


@dataclass(frozen=True)
class Landing:
    """One task landed on main; train members share merge_sha == unit_id and unit_size.

    landed_at and prev_landed_at are timezone-aware; ISO text is rendered only where a row is
    written or printed.

    A late claim names a sha an earlier event already claimed for another task: it is a unit of
    one under an opaque unit_id (compared for equality, never parsed) with unmeasurable lines.
    """

    project_id: str
    merge_sha: str
    task_id: str
    landed_at: datetime
    ordinal: int
    source: LandingSource
    unit_id: str
    unit_size: int
    late_claim: bool
    git_first_parent: bool
    prev_landed_at: datetime | None
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
        if self.late_claim and (
            self.unit_size != 1
            or self.lines_base is not LinesBase.UNMEASURABLE
            or self.unit_id == self.merge_sha
        ):
            raise ValueError(
                'a late_claim is its own unit of one with unmeasurable lines '
                f'(unit_id={self.unit_id!r}, unit_size={self.unit_size!r}, '
                f'lines_base={self.lines_base!r}, {where})'
            )
        if not self.late_claim and self.unit_id != self.merge_sha:
            raise ValueError(
                f'a first claim\'s unit_id is its merge_sha (unit_id={self.unit_id!r}, {where})'
            )
        for name, at in (('landed_at', self.landed_at), ('prev_landed_at', self.prev_landed_at)):
            if at is not None and at.tzinfo is None:
                raise ValueError(f'{name} carries its offset ({name}={at!r}, {where})')


@dataclass(frozen=True)
class GitCrossCheck:
    """Where runs.db and git disagree over the emitted range; served, never dropped."""

    off_first_parent: int
    merges_without_landing: int
    late_claims: int

    @property
    def disagreement(self) -> int:
        return self.off_first_parent + self.merges_without_landing + self.late_claims


@dataclass(frozen=True)
class LandingBatch:
    """The emitted landings, git's disagreement over their range, and the cursor to resume from.

    next_cursor is the merge_sha of the newest first-claim landing emitted, or the call's cursor
    in full when none was; None only for a full-history call that found no landing on main.
    """

    landings: tuple[Landing, ...]
    git_check: GitCrossCheck
    next_cursor: str | None


@dataclass(frozen=True)
class _Recorded:
    landed: datetime
    merge_sha: str
    position: int
    task_id: str
    source: LandingSource
    unit_size: int
    event_id: int
    late_claim: bool = False

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

_COALESCED_TRAIN = """
json_type(data, '$.member_task_ids') = 'array'
AND json_array_length(data, '$.member_task_ids') > 0
AND json_extract(data, '$.merge_commit_sha') IS NOT NULL
"""

_ATTRIBUTED_TRAIN = """
json_extract(data, '$.verdict') = 'attributed'
AND json_extract(data, '$.merge_sha') IS NOT NULL
"""

_COALESCED_TRAIN_SQL = f"""
SELECT events.id, events.timestamp, json_extract(data, '$.merge_commit_sha'),
       member.key, member.value, json_array_length(data, '$.member_task_ids')
  FROM events, json_each(events.data, '$.member_task_ids') AS member
 WHERE event_type = 'train_merged' AND {_COALESCED_TRAIN}
"""

_ATTRIBUTED_TRAIN_SQL = f"""
SELECT id, timestamp, task_id, json_extract(data, '$.merge_sha')
  FROM events
 WHERE event_type = 'train_merged' AND {_ATTRIBUTED_TRAIN}
"""

_UNRECOGNISED_TRAIN_SQL = f"""
SELECT id
  FROM events
 WHERE event_type = 'train_merged'
   AND NOT COALESCE(({_COALESCED_TRAIN}), 0)
   AND NOT COALESCE(({_ATTRIBUTED_TRAIN}), 0)
 ORDER BY id
"""


def _recorded_history(conn: sqlite3.Connection) -> list[_Recorded]:
    """Every landing in the store, deduplicated on (merge_sha, task_id) by earliest event."""
    unrecognised = [event_id for (event_id,) in conn.execute(_UNRECOGNISED_TRAIN_SQL)]
    if unrecognised:
        raise UnrecognisedTrainEvent(unrecognised)
    found: list[_Recorded] = []
    for event_id, ts, task_id, sha in conn.execute(_DONE_SQL):
        found.append(_Recorded(parse_utc(ts), sha, 0, str(task_id),
                               LandingSource.MERGE_FINALIZED, 1, event_id))
    for event_id, ts, sha, position, member, size in conn.execute(_COALESCED_TRAIN_SQL):
        found.append(_Recorded(parse_utc(ts), sha, position, str(member),
                               LandingSource.TRAIN_MERGED, size, event_id))
    for event_id, ts, task_id, sha in conn.execute(_ATTRIBUTED_TRAIN_SQL):
        found.append(_Recorded(parse_utc(ts), sha, 0, str(task_id),
                               LandingSource.TRAIN_ATTRIBUTED, 1, event_id))
    earliest: dict[tuple[str, str], _Recorded] = {}
    for recorded in sorted(found, key=lambda r: (r.landed, r.event_id)):
        earliest.setdefault((recorded.merge_sha, recorded.task_id), recorded)
    return sorted(_mark_late_claims(list(earliest.values())), key=lambda r: r.order)


def recorded_landing_keys(
    runs_db: sqlite3.Connection, *, recorded_after: datetime | None,
) -> frozenset[tuple[str, str]]:
    """(merge_sha, task_id) of each landing recorded after *recorded_after*; reads runs.db, never git."""
    return frozenset(
        (recorded.merge_sha, recorded.task_id)
        for recorded in _recorded_history(runs_db)
        if recorded_after is None or recorded.landed > recorded_after
    )


def _mark_late_claims(records: list[_Recorded]) -> list[_Recorded]:
    """A record is a late claim when an earlier event already claimed its sha for another task.

    A train's members share their event's id, so they all stay with the claiming event.
    """
    claiming_event: dict[str, int] = {}
    for recorded in sorted(records, key=lambda r: (r.landed, r.event_id)):
        claiming_event.setdefault(recorded.merge_sha, recorded.event_id)
    return [replace(r, late_claim=r.event_id != claiming_event[r.merge_sha]) for r in records]


def _unit(recorded: _Recorded) -> tuple[str, int]:
    """(unit_id, unit_size); a late claim's unit_id is opaque, compared only for equality."""
    if recorded.late_claim:
        return f'{recorded.merge_sha}:{recorded.task_id}', 1
    return recorded.merge_sha, recorded.unit_size


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
    A sha cursor emits first claims after it on main's first-parent line by position; late
    claims and off-first-parent landings it emits by landed_at, those recorded after the newest
    first-claim landing at or below the cursor, so a later cursor can re-emit one and consumers
    upsert on (project_id, merge_sha, task_id). Ordinals are numbered over the full history, so
    they are stable across calls.

    The next call's ``since_cursor`` is the batch's ``next_cursor``, never main's HEAD. A
    landing's runs.db event trails main's ref update (α measured 38 s to 3m52s on df), so a
    commit at or below HEAD may not be a recorded landing yet, and a cursor past it would skip
    that landing for good. Until its event is written, such a merge counts in
    ``git_check.merges_without_landing``.

    A late claim (a done row naming a sha an earlier event already claimed for another task)
    stays a landing, because the task left the merge queue with main advanced, but it forms its
    own unit of one with unmeasurable lines, and git_check counts it. ``project_id`` defaults
    to the repo root's project id. The call runs one blocking ``git rev-list`` bounded by
    ``git_timeout_s``, so a coroutine calls it through ``asyncio.to_thread``. The return is a
    LandingBatch whose ``.landings`` are the rows and ``.next_cursor`` the cursor to resume from.

    Raises CursorNotOnMain, UnrecognisedTrainEvent, and the git_lines errors (GitUnavailable for a non-repository).
    """
    project = project_id if project_id is not None else project_id_for_root(repo)
    line = _first_parent_line(repo, main_ref, git_timeout_s)
    position = line.position()
    history = _recorded_history(runs_db)
    bases = _lines_bases(line, {recorded.merge_sha for recorded in history})

    def first_claim_on_line(recorded: _Recorded) -> bool:
        return not recorded.late_claim and recorded.merge_sha in position

    on_line = [position[r.merge_sha] for r in history if r.merge_sha in position]
    cursor_position: int | None = None
    if since_cursor is None:
        range_start = min(on_line, default=len(line.shas))
        horizon: datetime | None = None
    else:
        cursor_position = _resolve_cursor(line, since_cursor)
        range_start = cursor_position + 1
        horizon = max(
            (r.landed for r in history
             if first_claim_on_line(r) and position[r.merge_sha] <= cursor_position),
            default=None,
        )

    def emitted(recorded: _Recorded) -> bool:
        if first_claim_on_line(recorded):
            return position[recorded.merge_sha] >= range_start
        return horizon is None or recorded.landed > horizon

    rows: list[Landing] = []
    previous: dict[str, datetime] = {}
    for ordinal, recorded in enumerate(history):
        unit_id, unit_size = _unit(recorded)
        lines_base, base_sha = (
            (LinesBase.UNMEASURABLE, None) if recorded.late_claim
            else bases.get(recorded.merge_sha, (LinesBase.UNMEASURABLE, None))
        )
        landing = Landing(
            project_id=project,
            merge_sha=recorded.merge_sha,
            task_id=recorded.task_id,
            landed_at=recorded.landed,
            ordinal=ordinal,
            source=recorded.source,
            unit_id=unit_id,
            unit_size=unit_size,
            late_claim=recorded.late_claim,
            git_first_parent=recorded.merge_sha in position,
            prev_landed_at=previous.get(recorded.task_id),
            lines_base=lines_base,
            base_sha=base_sha,
        )
        previous[recorded.task_id] = recorded.landed
        if emitted(recorded):
            rows.append(landing)

    landing_shas = {r.merge_sha for r in history}
    merges_without_landing = sum(
        1 for index in range(range_start, len(line.shas))
        if len(line.parents[index]) >= 2 and line.shas[index] not in landing_shas
    )
    off_first_parent = sum(1 for row in rows if not row.git_first_parent)
    late_claims = sum(1 for row in rows if row.late_claim)
    newest = max(
        (position[row.merge_sha] for row in rows if row.git_first_parent and not row.late_claim),
        default=cursor_position,
    )
    return LandingBatch(
        tuple(rows),
        GitCrossCheck(off_first_parent, merges_without_landing, late_claims),
        next_cursor=None if newest is None else line.shas[newest],
    )
