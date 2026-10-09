"""A frozen snapshot of one Mem0 collection, and per-arm re-embedded replicas of it.

The live collection grows continuously, so it is read once, read-only, into a JSONL
snapshot pinned by the sha256 of its bytes, and every arm's replica is built from
that one snapshot. A replica keeps each record's id and payload verbatim and
replaces only its vector. Its distance is pinned to Cosine here and nowhere else.
"""

import hashlib
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from shared.safe_io import atomic_write_text

from fused_memory.arm_harness.arm_embedder import DocumentEmbedder
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.normalization import NormStats, norm_stats
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name

REPLICA_DISTANCE = Distance.COSINE
MEM0_TEXT_KEY = 'data'
MEM0_PROJECT_KEY = 'user_id'
"""mem0 scopes every search by this payload key equal to the project id (mem0_client.py::Mem0Backend.search)."""

SNAPSHOT_PAGE_SIZE = 256
DEFAULT_WRITE_BATCH_SIZE = 64


class Mem0Record(FrozenModel):
    id: str
    data: str
    payload: dict[str, Any]


class Mem0Snapshot(FrozenModel):
    source: str
    excluded_empty: int = Field(ge=0)
    """Records in the source with no text to embed, left out rather than embedded as blanks."""
    records: tuple[Mem0Record, ...]


class _SnapshotHeader(FrozenModel):
    source: str
    excluded_empty: int = Field(ge=0)
    record_count: int = Field(ge=0)


class ReplicaReembed(FrozenModel):
    points: int
    written: int
    failures: tuple[tuple[str, str], ...]
    raw_norms: NormStats | None
    embed_seconds: float
    write_seconds: float


@dataclass(frozen=True)
class ReplicaHit:
    """One replica search result: the id and content E1's canonical_hit matches on."""

    id: str
    content: str


class CollectionReader(Protocol):
    """The read-only slice of AsyncQdrantClient a snapshot uses."""

    async def scroll(
        self,
        collection_name: str,
        *,
        limit: int = ...,
        offset: Any = ...,
        with_payload: bool = ...,
        with_vectors: bool = ...,
    ) -> tuple[list[Any], Any]: ...


class ReplicaWriter(Protocol):
    """The slice of AsyncQdrantClient a replica build uses."""

    async def collection_exists(self, collection_name: str) -> bool: ...

    async def create_collection(
        self, collection_name: str, vectors_config: VectorParams | None = ...
    ) -> object: ...

    async def upsert(
        self, collection_name: str, points: list[PointStruct], wait: bool = ...
    ) -> object: ...


class ReplicaSearcher(Protocol):
    """The slice of AsyncQdrantClient a replica search uses."""

    async def query_points(
        self,
        collection_name: str,
        *,
        query: list[float] | None = ...,
        query_filter: Filter | None = ...,
        limit: int = ...,
        with_payload: bool = ...,
    ) -> Any: ...


async def snapshot_collection(
    qdrant: CollectionReader, source: str, *, page_size: int = SNAPSHOT_PAGE_SIZE
) -> Mem0Snapshot:
    records: list[Mem0Record] = []
    excluded_empty = 0
    async for point in _scroll_all(qdrant, source, page_size):
        payload = dict(point.payload or {})
        data = payload.get(MEM0_TEXT_KEY)
        if isinstance(data, str) and data.strip():
            records.append(Mem0Record(id=point.id, data=data, payload=payload))
        else:
            excluded_empty += 1
    return Mem0Snapshot(
        source=source,
        excluded_empty=excluded_empty,
        records=tuple(sorted(records, key=lambda record: record.id)),
    )


async def _scroll_all(
    qdrant: CollectionReader, source: str, page_size: int
) -> AsyncIterator[Any]:
    offset = None
    while True:
        points, offset = await qdrant.scroll(
            source, limit=page_size, offset=offset, with_payload=True, with_vectors=False
        )
        for point in points:
            yield point
        if offset is None:
            return


def write_snapshot(path: Path, snapshot: Mem0Snapshot) -> None:
    """One header line, then one canonical JSON line per record."""
    header = _SnapshotHeader(
        source=snapshot.source,
        excluded_empty=snapshot.excluded_empty,
        record_count=len(snapshot.records),
    )
    lines = [_json_line(header), *(_json_line(record) for record in snapshot.records)]
    atomic_write_text(path, ''.join(line + '\n' for line in lines), mkdir=True)


def _json_line(model: FrozenModel) -> str:
    return json.dumps(model.model_dump(mode='json'), sort_keys=True, ensure_ascii=False)


def load_snapshot(path: Path) -> Mem0Snapshot:
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if not lines:
        raise ValueError(f'{path}: empty, so it holds no snapshot header')
    header = _SnapshotHeader.model_validate_json(lines[0])
    records = tuple(Mem0Record.model_validate_json(line) for line in lines[1:])
    if len(records) != header.record_count:
        raise ValueError(
            f'{path}: the header declares {header.record_count} records but {len(records)} '
            'follow, so the snapshot is truncated or edited'
        )
    return Mem0Snapshot(
        source=header.source, excluded_empty=header.excluded_empty, records=records
    )


def snapshot_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def build_replica(
    qdrant: ReplicaWriter,
    name: str,
    records: Sequence[Mem0Record],
    embedder: DocumentEmbedder,
    *,
    dim: int,
    write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
    clock: Callable[[], float] = time.perf_counter,
) -> ReplicaReembed:
    require_scratch_name(name, checkpoint=GuardCheckpoint.REPLICA_BUILD)
    if await qdrant.collection_exists(name):
        raise ValueError(
            f'replica {name!r} already exists; tear it down first, because a stale replica '
            'is never reused'
        )
    embedded = await embedder.embed_documents([(record.id, record.data) for record in records])
    await qdrant.create_collection(
        name, vectors_config=VectorParams(size=dim, distance=REPLICA_DISTANCE)
    )
    points = [
        PointStruct(id=record.id, vector=list(embedded.vectors[record.id]), payload=record.payload)
        for record in records
        if record.id in embedded.vectors
    ]
    started = clock()
    for start in range(0, len(points), write_batch_size):
        await qdrant.upsert(name, points=points[start : start + write_batch_size], wait=True)
    write_seconds = clock() - started
    norms = list(embedded.raw_norms.values())
    return ReplicaReembed(
        points=len(records),
        written=len(points),
        failures=embedded.failures,
        raw_norms=norm_stats(norms) if norms else None,
        embed_seconds=embedded.embed_seconds,
        write_seconds=write_seconds,
    )


async def search_replica(
    qdrant: ReplicaSearcher,
    name: str,
    vector: Sequence[float],
    *,
    limit: int,
    project_id: str,
) -> tuple[ReplicaHit, ...]:
    require_scratch_name(name, checkpoint=GuardCheckpoint.SEARCH)
    response = await qdrant.query_points(
        name,
        query=list(vector),
        query_filter=Filter(
            must=[FieldCondition(key=MEM0_PROJECT_KEY, match=MatchValue(value=project_id))]
        ),
        limit=limit,
        with_payload=True,
    )
    return tuple(
        ReplicaHit(id=str(point.id), content=point.payload[MEM0_TEXT_KEY])
        for point in response.points
    )
