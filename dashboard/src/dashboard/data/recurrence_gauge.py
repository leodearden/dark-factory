"""The R-D5/C-4 chain-state gauge of ``docs/prds/recurring-deterministic-tasks.md``.

Every recurrence chain gets exactly one state, read from its current link. The
current link is ``shared.recurrence_chains.current_links``' one definition, taken
one tag at a time, so ``broken`` is exactly the set the fused-memory re-mint
sweep repairs.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import aiosqlite
from shared.recurrence_chains import current_links
from shared.task_metadata import Milestone, Recurrence
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import TaskStatus

from dashboard.data.escalation_corpus import PendingEscalations

OVERDUE_GRACE_CAP = timedelta(hours=24)
_OVERDUE_GRACE_CAP_SECS = int(OVERDUE_GRACE_CAP.total_seconds())
# A necessary-condition prefilter: every chain link carries metadata.recurrence.
# recurrence_key still decides each fetched row. json_extract would raise on one
# malformed metadata cell and blind the whole root.
_CHAIN_ROWS_SQL = (
    "SELECT tag, id, status, metadata FROM tasks WHERE metadata LIKE '%recurrence%' "
    'ORDER BY tag, id'
)
_DATA_VERSION_SQL = 'PRAGMA data_version'


class ChainState(StrEnum):
    SCHEDULED = 'scheduled'
    OVERDUE = 'overdue'
    PAUSED = 'paused'
    BROKEN = 'broken'
    ENDED = 'ended'


_STATE_BY_STATUS: Mapping[str, ChainState] = {
    TaskStatus.BLOCKED: ChainState.PAUSED,
    TaskStatus.DONE: ChainState.BROKEN,
    TaskStatus.CANCELLED: ChainState.ENDED,
}


def overdue_grace(metadata: Mapping[str, Any]) -> timedelta:
    """How long past its fire time a live link may stay unfired before it is overdue."""
    try:
        recurrence = Recurrence.model_validate(metadata.get('recurrence'))
    except ValueError:
        return OVERDUE_GRACE_CAP
    return timedelta(seconds=min(recurrence.interval_secs, _OVERDUE_GRACE_CAP_SECS))


def _fire_time(metadata: Mapping[str, Any]) -> datetime | None:
    try:
        milestone = Milestone.model_validate(metadata.get('milestone'))
    except ValueError:
        return None
    if milestone.mode != 'dated' or milestone.at is None:
        return None
    at = datetime.fromisoformat(milestone.at)
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


def chain_state(link: Mapping[str, Any], *, now: datetime) -> ChainState:
    """The one state of the chain whose current link is *link* (a decoded task row)."""
    state = _STATE_BY_STATUS.get(link['status'])
    if state is not None:
        return state
    metadata = link['metadata']
    fire_time = _fire_time(metadata)
    if fire_time is None or now - overdue_grace(metadata) > fire_time:
        return ChainState.OVERDUE
    return ChainState.SCHEDULED


@dataclass(frozen=True)
class ChainEntry:
    tag: str
    key: str
    state: ChainState
    link_id: int
    link_status: str
    at: str | None
    pending_escalation_ids: tuple[str, ...] | None

    def to_wire(self) -> dict[str, Any]:
        return {
            'tag': self.tag,
            'key': self.key,
            'state': self.state.value,
            'link_id': self.link_id,
            'link_status': self.link_status,
            'at': self.at,
            'pending_escalation_ids': (
                None if self.pending_escalation_ids is None else list(self.pending_escalation_ids)
            ),
        }


def _as_link(row: Mapping[str, Any]) -> dict[str, Any]:
    return {**row, 'metadata': coerce_task_metadata(row['metadata']) or {}}


def _raw_fire_time(metadata: Mapping[str, Any]) -> str | None:
    milestone = metadata.get('milestone')
    at = milestone.get('at') if isinstance(milestone, Mapping) else None
    return at if isinstance(at, str) else None


def _entry(
    tag: str,
    key: str,
    link: Mapping[str, Any],
    *,
    now: datetime,
    pending_escalations: PendingEscalations | None,
) -> ChainEntry:
    return ChainEntry(
        tag=tag,
        key=key,
        state=chain_state(link, now=now),
        link_id=link['id'],
        link_status=link['status'],
        at=_raw_fire_time(link['metadata']),
        pending_escalation_ids=(
            None if pending_escalations is None else pending_escalations.pinning(tag, link['id'])
        ),
    )


def classify_chains(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
    pending_escalations: PendingEscalations | None,
) -> tuple[ChainEntry, ...]:
    """One entry per (tag, chain key), sorted; *pending_escalations* None means unknown."""
    links_by_tag: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        links_by_tag[row['tag']].append(_as_link(row))
    entries = [
        _entry(tag, key, link, now=now, pending_escalations=pending_escalations)
        for tag, links in links_by_tag.items()
        for key, link in current_links(links).items()
    ]
    return tuple(sorted(entries, key=lambda entry: (entry.tag, entry.key)))


@dataclass(frozen=True)
class _ChainRowsRead:
    data_version: int
    rows: tuple[Mapping[str, Any], ...]


_last_read_by_conn: weakref.WeakKeyDictionary[aiosqlite.Connection, _ChainRowsRead] = (
    weakref.WeakKeyDictionary()
)


async def read_chain_rows(conn: aiosqlite.Connection) -> tuple[Mapping[str, Any], ...]:
    """Every row that may be a chain link, rescanned only after another connection commits.

    The scan reads every metadata cell, so *conn*'s last scan is served again
    while ``PRAGMA data_version`` says no other connection has committed since.
    """
    (version_row,) = await conn.execute_fetchall(_DATA_VERSION_SQL)
    data_version = version_row[0]
    last = _last_read_by_conn.get(conn)
    if last is not None and last.data_version == data_version:
        return last.rows
    rows = tuple(
        MappingProxyType(dict(row)) for row in await conn.execute_fetchall(_CHAIN_ROWS_SQL)
    )
    _last_read_by_conn[conn] = _ChainRowsRead(data_version, rows)
    return rows
