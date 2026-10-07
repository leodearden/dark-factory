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
from collections.abc import AsyncIterator, Mapping
from types import SimpleNamespace
from typing import Any

from fused_memory.services.id_resolver import IdResolver


class FakeMem0:
    """Qdrant-shaped id walk plus point reads over an ``id -> payload`` map.

    ``scroll_error`` is raised after every id has been yielded, i.e. a walk
    that fails part-way; ``point_error`` is raised by every point read.
    """

    def __init__(
        self,
        points: Mapping[str, dict[str, Any] | None] | None = None,
        *,
        scroll_error: BaseException | None = None,
        point_error: BaseException | None = None,
    ) -> None:
        self.points = dict(points or {})
        self.scroll_error = scroll_error
        self.point_error = point_error
        self.scroll_calls: list[tuple[str, dict[str, Any]]] = []
        self.point_reads: list[str] = []

    async def scroll_collection_pages(self, collection_name: str, **kwargs: Any) -> AsyncIterator[Any]:
        self.scroll_calls.append((collection_name, kwargs))
        for point_id in self.points:
            yield SimpleNamespace(id=point_id)
        if self.scroll_error is not None:
            raise self.scroll_error

    async def get_point_by_id(self, memory_id: str, scope: Any) -> dict[str, Any] | None:
        self.point_reads.append(memory_id)
        if self.point_error is not None:
            raise self.point_error
        return self.points.get(memory_id)


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
    """An async ``(project_id, prefix) -> list[dict]`` tombstone source."""

    def __init__(self, found: list[dict[str, Any]] | None = None) -> None:
        self.found = list(found or [])
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, project_id: str, prefix: str) -> list[dict[str, Any]]:
        self.calls.append((project_id, prefix))
        return list(self.found)


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
