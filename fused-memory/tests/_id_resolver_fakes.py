"""In-memory stores for driving the REAL IdResolver in tests (task 5323).

One set of fakes, shared by the resolver, route and read-tool suites. Each
fake answers only the members ``services/id_resolver.py::IdResolver`` reads,
records every call so a test can prove which stores were (not) touched, and
can be told to fail or hang so the INV-11 arms are reachable without a live
store. The resolver itself is never stubbed: ``make_resolver`` builds the
real class over these fakes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from fused_memory.services.id_resolver import IdResolver


class FakeMem0:
    """Qdrant-shaped id walk plus batched point reads over an ``id -> payload`` map.

    A ``None`` payload is a point deleted after the walk saw it, so the batch
    read omits it. ``scroll_error`` is raised after every id has been yielded,
    i.e. a walk that fails part-way; ``hang`` stalls the walk before its
    first id; ``point_error`` is raised by every point read. ``point_reads``
    holds one tuple of ids per batch read.
    """

    def __init__(
        self,
        points: Mapping[str, dict[str, Any] | None] | None = None,
        *,
        scroll_error: BaseException | None = None,
        point_error: BaseException | None = None,
        hang: bool = False,
    ) -> None:
        self.points = dict(points or {})
        self.scroll_error = scroll_error
        self.point_error = point_error
        self.hang = hang
        self.scroll_calls: list[tuple[str, dict[str, Any]]] = []
        self.point_reads: list[tuple[str, ...]] = []

    async def scroll_collection_pages(self, collection_name: str, **kwargs: Any) -> AsyncIterator[Any]:
        self.scroll_calls.append((collection_name, kwargs))
        if self.hang:
            await asyncio.Event().wait()
        for point_id in self.points:
            yield SimpleNamespace(id=point_id)
        if self.scroll_error is not None:
            raise self.scroll_error

    async def get_points_by_ids(self, memory_ids: Sequence[str], scope: Any) -> dict[str, dict[str, Any]]:
        self.point_reads.append(tuple(memory_ids))
        if self.point_error is not None:
            raise self.point_error
        return {
            memory_id: payload
            for memory_id in memory_ids
            if (payload := self.points.get(memory_id)) is not None
        }


class FakeGraphiti:
    """The two STARTS WITH reads over ``uuid -> name`` nodes and ``uuid -> fact`` edges."""

    def __init__(
        self,
        nodes: Mapping[str, str] | None = None,
        edges: Mapping[str, str] | None = None,
        *,
        error: BaseException | None = None,
        hang: bool = False,
    ) -> None:
        self.nodes = dict(nodes or {})
        self.edges = dict(edges or {})
        self.error = error
        self.hang = hang
        self.calls: list[tuple[str, str, str]] = []

    async def entity_uuids_with_prefix(self, prefix: str, *, group_id: str) -> list[tuple[str, str]]:
        return await self._read('entity_uuids_with_prefix', self.nodes, prefix, group_id)

    async def edge_uuids_with_prefix(self, prefix: str, *, group_id: str) -> list[tuple[str, str]]:
        return await self._read('edge_uuids_with_prefix', self.edges, prefix, group_id)

    async def _read(
        self, method: str, rows: Mapping[str, str], prefix: str, group_id: str,
    ) -> list[tuple[str, str]]:
        self.calls.append((method, prefix, group_id))
        if self.hang:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return [(uuid, label) for uuid, label in sorted(rows.items()) if uuid.startswith(prefix)]


class RecordingTombstones:
    """An async ``(project_id, prefix) -> list[dict] | None`` tombstone source.

    ``unreadable`` answers ``None``: the tombstone store could not be read.
    """

    def __init__(self, found: list[dict[str, Any]] | None = None, *, unreadable: bool = False) -> None:
        self.found = list(found or [])
        self.unreadable = unreadable
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, project_id: str, prefix: str) -> list[dict[str, Any]] | None:
        self.calls.append((project_id, prefix))
        return None if self.unreadable else list(self.found)


def make_resolver(
    *,
    mem0: FakeMem0 | None = None,
    graphiti: FakeGraphiti | None = None,
    tombstones: RecordingTombstones | None = None,
    collection_prefix: str = 'fused',
    read_timeout_seconds: float = 5.0,
) -> IdResolver:
    """The real IdResolver over fakes; omitted stores are empty."""
    return IdResolver(
        mem0=mem0 if mem0 is not None else FakeMem0(),
        graphiti=graphiti if graphiti is not None else FakeGraphiti(),
        tombstones=tombstones if tombstones is not None else RecordingTombstones(),
        collection_prefix=collection_prefix,
        read_timeout_seconds=read_timeout_seconds,
    )
