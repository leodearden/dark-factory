"""Resolve a truncated uuid prefix to the ids it could mean (PRD C2).

``plans/uuid-prefix-resolution-prd.md`` §4-C2 is normative. The resolver
enumerates every id in one project that starts with a prefix — Mem0 point
ids, Graphiti Entity node uuids, RELATES_TO edge uuids — and classifies the
prefix as ``unique``, ``ambiguous`` or ``none``. It never mutates a store.

INV-11: a prefix that could not be checked is never ``none``. Any failure to
read a store raises ``ResolverUnavailable(store)``; the one exception is a
Mem0 collection that does not exist, which is the store answering that it
holds no ids.

Each store's reads run under one ``read_timeout_seconds`` deadline, so a
resolution is bounded per store, not per Qdrant page.

D9: no cache. Every call walks. A per-collection map would have to be
invalidated at ``Mem0Backend.add``, ``Mem0Backend.add_system_record`` and
``Mem0Backend.delete`` (plus a TTL rebuild for out-of-band writes); add one
only on a measurement.

The result and error types belong to the guard's port and live in
``shared/src/shared/uuid_prefix_guard.py``; they are imported, never copied.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

from shared.uuid_prefix import find_prefix_tokens
from shared.uuid_prefix_guard import (
    NAMESPACES,
    Candidate,
    Namespace,
    Resolution,
    ResolutionOutcome,
    ResolverUnavailable,
)

from fused_memory.backends.mem0_client import extract_payload_text, is_missing_collection_error
from fused_memory.models.scope import Scope
from fused_memory.utils.validation import is_full_uuid

if TYPE_CHECKING:
    from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

ID_WALK_PAGE_SIZE = 10_000
PREVIEW_CHARS = 80
MEM0_STORE = 'mem0'
GRAPHITI_STORE = 'graphiti'

TombstoneSource = Callable[[str, str], Awaitable[list[dict[str, Any]] | None]]


class Mem0Ids(Protocol):
    """The Mem0 reads the resolver makes: an ids-only walk, then one read of the matches."""

    def scroll_collection_pages(
        self, collection_name: str, *, with_payload: bool = ..., page_size: int = ...,
    ) -> AsyncIterator[Any]: ...

    async def get_points_by_ids(
        self, memory_ids: Sequence[str], scope: Scope,
    ) -> dict[str, dict[str, Any]]: ...


class GraphitiIds(Protocol):
    """The two STARTS WITH reads the resolver makes on a project graph."""

    async def entity_uuids_with_prefix(self, prefix: str, *, group_id: str) -> list[tuple[str, str]]: ...

    async def edge_uuids_with_prefix(self, prefix: str, *, group_id: str) -> list[tuple[str, str]]: ...


RESOLVABLE_PREFIX_GRAMMAR = (
    '8-31 lowercase hex characters standing alone, or a full 36-character uuid'
)


def is_uuid_prefix(value: object) -> bool:
    """True iff *value* is, in its entirety, one C1 prefix token (8-31 lowercase hex)."""
    if not isinstance(value, str):
        return False
    tokens = find_prefix_tokens({'value': value})
    return len(tokens) == 1 and tokens[0].start == 0 and tokens[0].end == len(value)


def is_resolvable_prefix(value: object) -> TypeGuard[str]:
    """True iff :meth:`IdResolver.resolve_prefix` accepts *value* (see RESOLVABLE_PREFIX_GRAMMAR)."""
    return is_uuid_prefix(value) or is_full_uuid(value)


def _normalized_prefix(prefix: str) -> str:
    if not is_resolvable_prefix(prefix):
        raise ValueError(f'{prefix!r} is not a uuid prefix: expected {RESOLVABLE_PREFIX_GRAMMAR}')
    return prefix.lower()


def _outcome(candidate_count: int) -> ResolutionOutcome:
    if candidate_count == 0:
        return 'none'
    if candidate_count == 1:
        return 'unique'
    return 'ambiguous'


def _candidate_order(candidate: Candidate) -> tuple[int, str]:
    return NAMESPACES.index(candidate.namespace), candidate.id


def _mem0_preview(payload: dict[str, Any] | None) -> str:
    text = extract_payload_text(payload) if payload else None
    return (text or '')[:PREVIEW_CHARS]


class IdResolver:
    """Enumerates and classifies the ids a prefix could mean; holds no state between calls."""

    def __init__(
        self,
        *,
        mem0: Mem0Ids,
        graphiti: GraphitiIds,
        tombstones: TombstoneSource,
        collection_prefix: str,
        read_timeout_seconds: float,
    ) -> None:
        self._mem0 = mem0
        self._graphiti = graphiti
        self._tombstones = tombstones
        self._collection_prefix = collection_prefix
        self._read_timeout = read_timeout_seconds

    @classmethod
    def for_memory_service(cls, memory_service: MemoryService) -> IdResolver:
        """Wire a resolver to a service's stores. Reads attributes only; no I/O."""
        return cls(
            mem0=memory_service.mem0,
            graphiti=memory_service.graphiti,
            tombstones=memory_service.find_mem0_deletion_tombstones,
            collection_prefix=memory_service.config.mem0.collection_prefix,
            read_timeout_seconds=memory_service.config.queue.backend_read_timeout_seconds,
        )

    async def resolve_prefix(self, project_id: str, prefix: str) -> Resolution:
        """Classify *prefix* against every id in *project_id*, Mem0 first, then Graphiti.

        Raises:
            ValueError: *prefix* is neither a C1 prefix token nor a full uuid.
            ResolverUnavailable: A store could not be read in full.
        """
        prefix = _normalized_prefix(prefix)
        candidates = [
            *await self._mem0_candidates(project_id, prefix),
            *await self._graphiti_candidates(project_id, prefix),
        ]
        candidates.sort(key=_candidate_order)
        return Resolution(_outcome(len(candidates)), tuple(candidates))

    async def miss_report(
        self, project_id: str, prefix: str, resolution: Resolution,
    ) -> dict[str, Any]:
        """What each namespace holds for *prefix*: the resolution's live ids, plus tombstones.

        Reads only the tombstone source; the live lists are the resolution's
        own candidates, so nothing the resolution just walked is walked again.
        ``mem0_tombstone`` is ``None`` when the source could not read its
        store: unchecked, never an empty list (INV-11).
        """

        def live(namespace: Namespace) -> list[dict[str, str]]:
            return [
                {'id': candidate.id, 'preview': candidate.preview}
                for candidate in resolution.candidates
                if candidate.namespace == namespace
            ]

        return {
            'mem0': live('mem0'),
            'mem0_tombstone': await self._tombstones(project_id, prefix),
            'graphiti_node': live('graphiti_node'),
            'graphiti_edge': live('graphiti_edge'),
        }

    async def _mem0_candidates(self, project_id: str, prefix: str) -> list[Candidate]:
        scope = Scope(project_id=project_id)
        collection = scope.mem0_collection_name(self._collection_prefix)
        try:
            async with asyncio.timeout(self._read_timeout):
                matched: list[str] = []
                async for point in self._mem0.scroll_collection_pages(
                    collection, with_payload=False, page_size=ID_WALK_PAGE_SIZE,
                ):
                    point_id = str(point.id)
                    if point_id.startswith(prefix):
                        matched.append(point_id)
                payloads = await self._mem0.get_points_by_ids(matched, scope) if matched else {}
        except Exception as exc:
            if is_missing_collection_error(exc):
                logger.info(
                    'resolve_prefix: Mem0 collection %s does not exist; it holds no ids',
                    collection,
                )
                return []
            raise ResolverUnavailable(MEM0_STORE) from exc
        return [
            Candidate('mem0', memory_id, _mem0_preview(payloads.get(memory_id)))
            for memory_id in matched
        ]

    async def _graphiti_candidates(self, project_id: str, prefix: str) -> list[Candidate]:
        try:
            async with asyncio.timeout(self._read_timeout):
                nodes = await self._graphiti.entity_uuids_with_prefix(prefix, group_id=project_id)
                edges = await self._graphiti.edge_uuids_with_prefix(prefix, group_id=project_id)
        except Exception as exc:
            raise ResolverUnavailable(GRAPHITI_STORE) from exc
        return [
            *(Candidate('graphiti_node', uuid, name[:PREVIEW_CHARS]) for uuid, name in nodes),
            *(Candidate('graphiti_edge', uuid, fact[:PREVIEW_CHARS]) for uuid, fact in edges),
        ]
