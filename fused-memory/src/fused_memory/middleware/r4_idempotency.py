"""R4: which of a submission's follow-up work the live store already carries.

Contract: plans/follow-up-filing-shape-prd.md §4.1. No LLM and no embedding, so it holds while the
curator is down. Only a live task (``shared.task_statuses.ACTIVE``) owns work. A lookup error
propagates: the caller chooses how to degrade. Every check reads through an :class:`R4Lookup`, which
one pass of checks shares.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from shared.followup_items import (
    FOLLOWUP_ITEMS_METADATA_KEY,
    FollowupItem,
    parse_item,
    read_items,
    with_derived_bundle_keys,
)
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import ACTIVE

from fused_memory.backends.task_backend_errors import TaskNotFoundError
from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
from fused_memory.backends.task_backend_types import MetadataLookupRow

__all__ = [
    'LIVE_STATUSES',
    'DroppedItem',
    'LiveOwner',
    'R4Lookup',
    'R4Match',
    'R4Verdict',
    'check_r4',
    'metadata_keeping',
]

LIVE_STATUSES: Final = tuple(sorted(status.value for status in ACTIVE))

# One lookup page; a value shared past it is paged, not missed.
_R4_LOOKUP_PAGE: Final = 100


@dataclass(frozen=True)
class LiveOwner:
    task_id: int
    title: str


@dataclass(frozen=True)
class DroppedItem:
    key: str
    owner: LiveOwner

    def to_record(self) -> dict[str, str]:
        return {'key': self.key, 'task_id': str(self.owner.task_id)}


class R4Match(StrEnum):
    ESCALATION_PAIR = 'escalation+suggestion matched'
    FOLLOWUP_ITEMS = 'followup items already live'


@dataclass(frozen=True)
class R4Verdict:
    """``hit`` names the live task the whole submission resolves onto; otherwise ``kept`` is filed."""

    hit: LiveOwner | None = None
    match: R4Match | None = None
    kept: tuple[FollowupItem, ...] = ()
    dropped: tuple[DroppedItem, ...] = ()

    def __post_init__(self) -> None:
        if (self.hit is None) != (self.match is None):
            raise ValueError(f'R4Verdict needs hit and match together, got {self.hit!r} and {self.match!r}')

    @property
    def partial(self) -> bool:
        return bool(self.dropped) and bool(self.kept)

    def hit_result(self) -> dict[str, Any]:
        if self.hit is None or self.match is None:
            raise ValueError('a verdict without a hit has no hit result')
        result: dict[str, Any] = {
            'id': str(self.hit.task_id),
            'title': self.hit.title,
            'deduplicated': True,
            'action': 'idempotency_hit',
            'reason': self.match.value,
        }
        if self.dropped:
            result['idempotency_dropped_items'] = [dropped.to_record() for dropped in self.dropped]
        return result


class R4Lookup:
    """R4's reads of the live store, shared by one pass of checks that writes no task between them.

    The live item owners come from a scan of every live bundle, so it runs once per project root,
    on the first check that needs it, and the pass's later checks reuse it. A scan that raised is
    not kept: the next check scans again.
    """

    def __init__(self, backend: TaskBackendProtocol) -> None:
        self.backend: Final = backend
        self._item_owners: dict[str, Mapping[str, LiveOwner]] = {}

    async def item_owner(self, project_root: str, key: str) -> LiveOwner | None:
        """The lowest-id live task in *project_root* carrying a follow-up item keyed *key*."""
        owners = self._item_owners.get(project_root)
        if owners is None:
            owners = await _live_item_owners(self.backend, project_root)
            self._item_owners[project_root] = owners
        return owners.get(key)


async def check_r4(lookup: R4Lookup, project_root: str, metadata: Mapping[str, Any]) -> R4Verdict:
    """The verdict on one submission's metadata.

    1. Its ``(escalation_id, suggestion_hash)`` pair on a live task: a hit.
    2. A bundle: each item a live task owns is dropped, by a live ``followup_items`` key or a live
       ``suggestion_hash`` equal to the item key; with every item dropped, a hit on the first owner.
    3. A pre-bundle single whose ``suggestion_hash`` is a live item key: a hit.
    """
    pair_owner = await _escalation_pair_owner(lookup.backend, project_root, metadata)
    if pair_owner is not None:
        return R4Verdict(hit=pair_owner, match=R4Match.ESCALATION_PAIR)
    if FOLLOWUP_ITEMS_METADATA_KEY in metadata:
        return await _bundle_verdict(lookup, project_root, read_items(metadata))
    suggestion_hash = metadata.get('suggestion_hash')
    if not isinstance(suggestion_hash, str) or not suggestion_hash.strip():
        return R4Verdict()
    owner = await lookup.item_owner(project_root, suggestion_hash)
    if owner is None:
        return R4Verdict()
    return R4Verdict(
        hit=owner, match=R4Match.FOLLOWUP_ITEMS, dropped=(DroppedItem(suggestion_hash, owner),),
    )


def metadata_keeping(metadata: Mapping[str, Any], verdict: R4Verdict) -> dict[str, Any]:
    """``metadata`` as the bundle of the verdict's kept items, with their derived R4 keys."""
    return with_derived_bundle_keys({
        **metadata,
        FOLLOWUP_ITEMS_METADATA_KEY: [item.model_dump(mode='json') for item in verdict.kept],
    })


async def _bundle_verdict(
    lookup: R4Lookup, project_root: str, items: Sequence[FollowupItem],
) -> R4Verdict:
    kept: list[FollowupItem] = []
    dropped: list[DroppedItem] = []
    for item in items:
        owner = (
            await lookup.item_owner(project_root, item.key)
            or await _live_hash_owner(lookup.backend, project_root, item.key)
        )
        if owner is None:
            kept.append(item)
        else:
            dropped.append(DroppedItem(item.key, owner))
    if dropped and not kept:
        return R4Verdict(hit=dropped[0].owner, match=R4Match.FOLLOWUP_ITEMS, dropped=tuple(dropped))
    return R4Verdict(kept=tuple(kept), dropped=tuple(dropped))


async def _live_item_owners(backend: TaskBackendProtocol, project_root: str) -> dict[str, LiveOwner]:
    """Every live item key and the lowest-id live task carrying it."""
    owners: dict[str, LiveOwner] = {}
    offset = 0
    while True:
        page = await backend.find_tasks_by_metadata(
            project_root, key=FOLLOWUP_ITEMS_METADATA_KEY, statuses=list(LIVE_STATUSES),
            limit=_R4_LOOKUP_PAGE, offset=offset, include_value=True,
        )
        rows = list(page['results'])
        for row in rows:
            stored = row.get('matched_value')
            for element in stored if isinstance(stored, list) else ():
                item = parse_item(element)
                if item is not None:
                    owners.setdefault(item.key, _owner(row))
        offset += len(rows)
        if not rows or offset >= page['total']:
            return owners


async def _live_hash_owner(
    backend: TaskBackendProtocol, project_root: str, suggestion_hash: str,
) -> LiveOwner | None:
    page = await backend.find_tasks_by_metadata(
        project_root, key='suggestion_hash', value=suggestion_hash,
        statuses=list(LIVE_STATUSES), limit=1,
    )
    rows = list(page['results'])
    return _owner(rows[0]) if rows else None


async def _escalation_pair_owner(
    backend: TaskBackendProtocol, project_root: str, metadata: Mapping[str, Any],
) -> LiveOwner | None:
    esc_id = metadata.get('escalation_id')
    sug_hash = metadata.get('suggestion_hash')
    if not isinstance(esc_id, str) or not isinstance(sug_hash, str) or not esc_id or not sug_hash:
        return None
    row = await _escalation_match(backend, project_root, esc_id, sug_hash)
    return None if row is None else _owner(row)


async def _escalation_match(
    backend: TaskBackendProtocol, project_root: str, esc_id: str, sug_hash: str,
) -> MetadataLookupRow | None:
    """The first live task carrying *sug_hash* whose ``escalation_id`` is *esc_id*."""
    offset = 0
    while True:
        page = await backend.find_tasks_by_metadata(
            project_root, key='suggestion_hash', value=sug_hash,
            statuses=list(LIVE_STATUSES), limit=_R4_LOOKUP_PAGE, offset=offset,
        )
        rows = list(page['results'])
        if not rows:
            return None
        for row in rows:
            if await _carries_escalation(backend, project_root, row['task_id'], esc_id):
                return row
        offset += len(rows)
        if offset >= page['total']:
            return None


async def _carries_escalation(
    backend: TaskBackendProtocol, project_root: str, task_id: int, esc_id: str,
) -> bool:
    """Whether task *task_id* still exists and its metadata names *esc_id*."""
    try:
        task = await backend.get_task(str(task_id), project_root)
    except TaskNotFoundError:
        return False
    task_meta = coerce_task_metadata(task.get('metadata'))
    return task_meta is not None and task_meta.get('escalation_id') == esc_id


def _owner(row: MetadataLookupRow) -> LiveOwner:
    return LiveOwner(task_id=row['task_id'], title=str(row['title']))
