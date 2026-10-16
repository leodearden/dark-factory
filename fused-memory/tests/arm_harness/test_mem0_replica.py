"""A frozen snapshot of one Mem0 collection, and per-arm re-embedded replicas of it (arm_harness/mem0_replica.py)."""

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from arm_harness._fakes import PROTECTED_GRAPHS
from fused_memory.arm_harness.arm_embedder import DocumentEmbeddings
from fused_memory.arm_harness.mem0_replica import (
    Mem0Record,
    Mem0Snapshot,
    ReplicaHit,
    ReplicaReembed,
    build_replica,
    load_snapshot,
    search_replica,
    snapshot_collection,
    snapshot_sha,
    write_snapshot,
)
from fused_memory.arm_harness.normalization import NormStats
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError

SOURCE = 'fused_dark_factory'
REPLICA = 'evalmem_lme_emb_granite'
PROJECT = 'dark_factory'
WRITES = frozenset({'create_collection', 'upsert'})

ID_A = '0b6f8a52-3a50-4c11-9d39-0f5d1c1e7a01'
ID_B = '4c1d2e3f-5a6b-4c7d-8e9f-0a1b2c3d4e02'
ID_C = '9e8d7c6b-5a49-4382-9170-6f5e4d3c2b03'


def _payload(data: Any, **extra: Any) -> dict[str, Any]:
    payload = {'user_id': PROJECT, 'hash': 'h', 'category': 'procedural_knowledge', **extra}
    if data is not None:
        payload['data'] = data
    return payload


def _point(point_id: str, data: Any, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, payload=_payload(data, **extra), vector=None)


class FakeQdrant:
    """The AsyncQdrantClient slice the replica uses; refuses any write to a non-``evalmem_`` name.

    ``scroll`` hands back at most ``page_cap`` points per page whatever ``limit`` asks,
    as a server-side cap would, in the stored order rather than by id.
    """

    def __init__(
        self,
        collections: Mapping[str, Sequence[SimpleNamespace]] | None = None,
        *,
        page_cap: int = 1000,
    ) -> None:
        self.points: dict[str, list[SimpleNamespace]] = {
            name: list(points) for name, points in (collections or {}).items()
        }
        self.page_cap = page_cap
        self.vector_params: dict[str, VectorParams] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _refuse_live(self, collection_name: str) -> None:
        if not collection_name.startswith('evalmem_'):
            raise AssertionError(f'write to live collection {collection_name!r}')

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    async def scroll(
        self,
        collection_name: str,
        *,
        limit: int = 10,
        offset: int | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[SimpleNamespace], int | None]:
        self.calls.append(('scroll', {
            'collection_name': collection_name,
            'limit': limit,
            'offset': offset,
            'with_payload': with_payload,
            'with_vectors': with_vectors,
        }))
        start = offset or 0
        stop = start + min(limit, self.page_cap)
        points = self.points[collection_name]
        return points[start:stop], (stop if stop < len(points) else None)

    async def collection_exists(self, collection_name: str) -> bool:
        self.calls.append(('collection_exists', {'collection_name': collection_name}))
        return collection_name in self.points

    async def create_collection(
        self, collection_name: str, vectors_config: VectorParams | None = None
    ) -> bool:
        self._refuse_live(collection_name)
        self.calls.append(('create_collection', {
            'collection_name': collection_name, 'vectors_config': vectors_config,
        }))
        assert vectors_config is not None
        self.points[collection_name] = []
        self.vector_params[collection_name] = vectors_config
        return True

    async def upsert(
        self, collection_name: str, points: list[PointStruct], wait: bool = True
    ) -> SimpleNamespace:
        self._refuse_live(collection_name)
        self.calls.append(('upsert', {
            'collection_name': collection_name, 'points': list(points), 'wait': wait,
        }))
        self.points[collection_name].extend(
            SimpleNamespace(id=point.id, payload=point.payload, vector=point.vector)
            for point in points
        )
        return SimpleNamespace(status='completed')

    async def query_points(
        self,
        collection_name: str,
        query: list[float] | None = None,
        query_filter: Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> SimpleNamespace:
        self.calls.append(('query_points', {
            'collection_name': collection_name,
            'query': query,
            'query_filter': query_filter,
            'limit': limit,
            'with_payload': with_payload,
        }))
        assert query is not None
        scored = sorted(
            self.points[collection_name],
            key=lambda point: -sum(a * b for a, b in zip(point.vector, query, strict=True)),
        )
        return SimpleNamespace(points=scored[:limit])


class FakeEmbedder:
    """A DocumentEmbedder whose vectors are fixed per key; ``failing`` keys come back as failures."""

    def __init__(self, vectors: Mapping[str, tuple[float, ...]], *, failing: frozenset[str] = frozenset()):
        self.vectors = vectors
        self.failing = failing
        self.items: list[tuple[str, str]] = []

    async def embed_documents(self, items: Sequence[tuple[str, str]]) -> DocumentEmbeddings:
        self.items.extend(items)
        kept = [key for key, _ in items if key not in self.failing]
        return DocumentEmbeddings(
            vectors=MappingProxyType({key: self.vectors[key] for key in kept}),
            raw_norms=MappingProxyType({key: 30.0 + index for index, key in enumerate(kept)}),
            embed_seconds=1.5,
            failures=tuple(
                (key, 'VectorInvariantError') for key, _ in items if key in self.failing
            ),
        )


UNIT = {
    ID_A: (1.0, 0.0, 0.0),
    ID_B: (0.0, 1.0, 0.0),
    ID_C: (0.0, 0.0, 1.0),
}


def _records() -> tuple[Mem0Record, ...]:
    return tuple(
        Mem0Record(id=point_id, data=f'memory {point_id[:4]}', payload=_payload(f'memory {point_id[:4]}'))
        for point_id in (ID_A, ID_B, ID_C)
    )


def _ticking(*ticks: float) -> Callable[[], float]:
    remaining = iter(ticks)
    return lambda: next(remaining)


# --- snapshot_collection -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_snapshot_pages_the_whole_source_without_vectors():
    qdrant = FakeQdrant(
        {SOURCE: [_point(ID_C, 'c'), _point(ID_A, 'a'), _point(ID_B, 'b')]}, page_cap=2
    )

    snapshot = await snapshot_collection(qdrant, SOURCE)

    scrolls = [call for name, call in qdrant.calls if name == 'scroll']
    assert len(scrolls) == 2
    assert [call['offset'] for call in scrolls] == [None, 2]
    assert all(call['collection_name'] == SOURCE for call in scrolls)
    assert all(call['with_payload'] is True and call['with_vectors'] is False for call in scrolls)
    assert [record.id for record in snapshot.records] == [ID_A, ID_B, ID_C]


@pytest.mark.asyncio
async def test_a_snapshot_keeps_each_payload_verbatim_and_reads_its_text_from_data():
    point = _point(ID_A, 'remember the merge lane', created_at='2026-10-01T00:00:00Z')
    qdrant = FakeQdrant({SOURCE: [point]})

    snapshot = await snapshot_collection(qdrant, SOURCE)

    assert snapshot.source == SOURCE
    (record,) = snapshot.records
    assert record.data == 'remember the merge lane'
    assert record.payload == point.payload


@pytest.mark.asyncio
async def test_a_snapshot_excludes_records_with_no_text_and_counts_them():
    qdrant = FakeQdrant({SOURCE: [
        _point(ID_A, 'kept'),
        _point(ID_B, '   \n\t'),
        _point(ID_C, None),
        _point('d2f1c3b4-0000-4000-8000-000000000004', ''),
    ]})

    snapshot = await snapshot_collection(qdrant, SOURCE)

    assert [record.id for record in snapshot.records] == [ID_A]
    assert snapshot.excluded_empty == 3


@pytest.mark.asyncio
async def test_a_snapshot_never_writes_to_its_source():
    qdrant = FakeQdrant({SOURCE: [_point(ID_A, 'a'), _point(ID_B, 'b')]}, page_cap=1)

    await snapshot_collection(qdrant, SOURCE)

    assert set(qdrant.names) == {'scroll'}


# --- write_snapshot / load_snapshot / snapshot_sha ---------------------------------------


def _snapshot() -> Mem0Snapshot:
    return Mem0Snapshot(source=SOURCE, excluded_empty=2, records=_records())


def test_a_written_snapshot_loads_back_identical(tmp_path: Path):
    path = tmp_path / 'mem0-snapshot.jsonl'

    write_snapshot(path, _snapshot())

    assert load_snapshot(path) == _snapshot()


def test_the_snapshot_file_is_one_header_line_then_one_line_per_record(tmp_path: Path):
    path = tmp_path / 'mem0-snapshot.jsonl'

    write_snapshot(path, _snapshot())

    assert len(path.read_bytes().splitlines()) == 1 + len(_records())


def test_the_snapshot_sha_is_the_sha256_of_the_file_bytes_and_is_stable(tmp_path: Path):
    first, second = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'

    write_snapshot(first, _snapshot())
    write_snapshot(second, _snapshot())

    assert snapshot_sha(first) == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.read_bytes() == second.read_bytes()


def test_a_truncated_snapshot_is_refused(tmp_path: Path):
    path = tmp_path / 'mem0-snapshot.jsonl'
    write_snapshot(path, _snapshot())
    path.write_bytes(b'\n'.join(path.read_bytes().splitlines()[:-1]) + b'\n')

    with pytest.raises(ValueError, match='mem0-snapshot.jsonl'):
        load_snapshot(path)


# --- build_replica -----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('protected', (*PROTECTED_GRAPHS, SOURCE))
async def test_a_replica_on_a_live_name_is_refused_before_any_client_call(protected: str):
    qdrant = FakeQdrant()

    with pytest.raises(ScratchGuardError) as caught:
        await build_replica(qdrant, protected, _records(), FakeEmbedder(UNIT), dim=3)

    assert caught.value.checkpoint is GuardCheckpoint.REPLICA_BUILD
    assert qdrant.calls == []


@pytest.mark.asyncio
async def test_an_existing_replica_is_refused_and_left_alone():
    qdrant = FakeQdrant({REPLICA: [_point(ID_A, 'stale')]})
    embedder = FakeEmbedder(UNIT)

    with pytest.raises(ValueError, match=REPLICA):
        await build_replica(qdrant, REPLICA, _records(), embedder, dim=3)

    assert not WRITES & set(qdrant.names)
    assert embedder.items == []


@pytest.mark.asyncio
async def test_a_replica_is_created_cosine_at_the_arms_dimension():
    qdrant = FakeQdrant()

    await build_replica(qdrant, REPLICA, _records(), FakeEmbedder(UNIT), dim=3)

    assert qdrant.vector_params[REPLICA] == VectorParams(size=3, distance=Distance.COSINE)


@pytest.mark.asyncio
async def test_a_replica_holds_the_original_ids_and_payloads_with_the_arms_vectors():
    qdrant = FakeQdrant()
    embedder = FakeEmbedder(UNIT)

    await build_replica(qdrant, REPLICA, _records(), embedder, dim=3)

    assert embedder.items == [(record.id, record.data) for record in _records()]
    stored = {point.id: point for point in qdrant.points[REPLICA]}
    assert set(stored) == {ID_A, ID_B, ID_C}
    for record in _records():
        assert stored[record.id].payload == record.payload
        assert tuple(stored[record.id].vector) == UNIT[record.id]


@pytest.mark.asyncio
async def test_a_replica_upserts_in_batches():
    qdrant = FakeQdrant()

    await build_replica(
        qdrant, REPLICA, _records(), FakeEmbedder(UNIT), dim=3, write_batch_size=2
    )

    upserts = [call for name, call in qdrant.calls if name == 'upsert']
    assert [len(call['points']) for call in upserts] == [2, 1]
    assert all(call['collection_name'] == REPLICA for call in upserts)


@pytest.mark.asyncio
async def test_a_record_the_arm_cannot_embed_is_skipped_and_listed():
    qdrant = FakeQdrant()

    reembed = await build_replica(
        qdrant, REPLICA, _records(), FakeEmbedder(UNIT, failing=frozenset({ID_B})), dim=3
    )

    assert {point.id for point in qdrant.points[REPLICA]} == {ID_A, ID_C}
    assert reembed.failures == ((ID_B, 'VectorInvariantError'),)
    assert reembed.points == 3
    assert reembed.written == 2


@pytest.mark.asyncio
async def test_a_replica_reports_its_norms_and_both_timings():
    qdrant = FakeQdrant()

    reembed = await build_replica(
        qdrant, REPLICA, _records(), FakeEmbedder(UNIT), dim=3, clock=_ticking(10.0, 12.5)
    )

    assert reembed == ReplicaReembed(
        points=3,
        written=3,
        failures=(),
        raw_norms=NormStats(count=3, min=30.0, median=31.0, max=32.0),
        embed_seconds=1.5,
        write_seconds=2.5,
    )


# --- search_replica ----------------------------------------------------------------------


async def _built_replica() -> FakeQdrant:
    qdrant = FakeQdrant()
    await build_replica(qdrant, REPLICA, _records(), FakeEmbedder(UNIT), dim=3)
    return qdrant


@pytest.mark.asyncio
async def test_a_replica_search_is_scoped_to_the_project_as_mem0_scopes_it():
    qdrant = await _built_replica()

    await search_replica(qdrant, REPLICA, (0.0, 1.0, 0.0), limit=2, project_id=PROJECT)

    ((_, call),) = [(name, call) for name, call in qdrant.calls if name == 'query_points']
    assert call['collection_name'] == REPLICA
    assert call['query'] == [0.0, 1.0, 0.0]
    assert call['limit'] == 2
    assert call['query_filter'] == Filter(
        must=[FieldCondition(key='user_id', match=MatchValue(value=PROJECT))]
    )


@pytest.mark.asyncio
async def test_a_replica_search_returns_hits_in_rank_order_with_their_text():
    qdrant = await _built_replica()
    query = (0.0, 1.0 / math.sqrt(2), 1.0 / math.sqrt(2) + 0.01)

    hits = await search_replica(qdrant, REPLICA, query, limit=2, project_id=PROJECT)

    by_id = {record.id: record for record in _records()}
    assert hits == (
        ReplicaHit(id=ID_C, content=by_id[ID_C].data),
        ReplicaHit(id=ID_B, content=by_id[ID_B].data),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('protected', (*PROTECTED_GRAPHS, SOURCE))
async def test_a_search_on_a_live_collection_is_refused(protected: str):
    qdrant = FakeQdrant({protected: [_point(ID_A, 'live')]})

    with pytest.raises(ScratchGuardError) as caught:
        await search_replica(qdrant, protected, (1.0, 0.0, 0.0), limit=1, project_id=PROJECT)

    assert caught.value.checkpoint is GuardCheckpoint.SEARCH
    assert qdrant.calls == []
