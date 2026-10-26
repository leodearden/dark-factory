"""The by-id cosine lookup: the Mem0 cosine of a text to named memory ids (task 7158).

Write triage's attach guard compares a parent's cosine with the slate's
``store_score`` values, so the lookup must report the SAME metric search
does: the query embedded by the mem0 instance's own embedder and scored by
Qdrant's cosine. These tests run against a real in-memory Qdrant so the
metric itself is pinned, not a mocked call shape.
"""

import math
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from _qdrant_responses import missing_collection_404
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.models.scope import Scope
from fused_memory.services.memory_service import MemoryService

PROJECT = 'p'
QUERY_VECTOR = [1.0, 0.5, 0.0]
VECTORS = {
    'A': [1.0, 0.0, 0.0],
    'B': [1.0, 1.0, 0.0],
    'C': [0.0, 0.0, 1.0],
}
IDS = {name: str(uuid.uuid5(uuid.NAMESPACE_URL, f'cosine-to-points/{name}')) for name in VECTORS}
ABSENT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, 'cosine-to-points/absent'))


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    return dot / (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))


class _RecordingEmbedder:
    """Embeds every text as QUERY_VECTOR and remembers what it was asked to embed."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        self.texts.append(text)
        return list(QUERY_VECTOR)


def _collection(config) -> str:
    return Scope(project_id=PROJECT).mem0_collection_name(config.mem0.collection_prefix)


@pytest_asyncio.fixture
async def qdrant(mock_config):
    client = AsyncQdrantClient(location=':memory:')
    collection = _collection(mock_config)
    await client.create_collection(
        collection,
        vectors_config=qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE),
    )
    await client.upsert(
        collection,
        points=[
            qmodels.PointStruct(id=IDS[name], vector=vector, payload={'data': name})
            for name, vector in VECTORS.items()
        ],
    )
    yield client
    await client.close()


@pytest.fixture
def embedder() -> _RecordingEmbedder:
    return _RecordingEmbedder()


def _bind(backend: Mem0Backend, client, embedder: _RecordingEmbedder):
    """Patch *backend*'s Qdrant client and mem0 instance (it has no public injection seam)."""
    instance = types.SimpleNamespace(embedding_model=embedder)
    return (
        patch.object(backend, '_get_async_qdrant', AsyncMock(return_value=client)),
        patch.object(backend, '_get_instance', AsyncMock(return_value=instance)),
    )


class TestCosineToPoints:
    @pytest.mark.asyncio
    async def test_one_id_reads_its_cosine_to_the_query(self, mock_config, qdrant, embedder):
        backend = Mem0Backend(mock_config)
        qdrant_patch, instance_patch = _bind(backend, qdrant, embedder)
        with qdrant_patch, instance_patch:
            scores = await backend.cosine_to_points('the write', [IDS['B']], Scope(project_id=PROJECT))

        assert scores == {IDS['B']: pytest.approx(_cosine(QUERY_VECTOR, VECTORS['B']), rel=1e-6)}

    @pytest.mark.asyncio
    async def test_the_score_is_on_the_scale_an_unfiltered_search_reports(
        self, mock_config, qdrant, embedder,
    ):
        backend = Mem0Backend(mock_config)
        qdrant_patch, instance_patch = _bind(backend, qdrant, embedder)
        with qdrant_patch, instance_patch:
            scores = await backend.cosine_to_points('the write', [IDS['B']], Scope(project_id=PROJECT))
        unfiltered = await qdrant.query_points(_collection(mock_config), query=QUERY_VECTOR, limit=3)

        search_scores = {str(point.id): point.score for point in unfiltered.points}
        assert scores[IDS['B']] == pytest.approx(search_scores[IDS['B']], rel=1e-9)

    @pytest.mark.asyncio
    async def test_only_the_asked_ids_that_exist_are_returned(self, mock_config, qdrant, embedder):
        backend = Mem0Backend(mock_config)
        qdrant_patch, instance_patch = _bind(backend, qdrant, embedder)
        with qdrant_patch, instance_patch:
            scores = await backend.cosine_to_points(
                'the write', [IDS['A'], IDS['B'], ABSENT_ID], Scope(project_id=PROJECT),
            )

        assert scores == {
            IDS['A']: pytest.approx(_cosine(QUERY_VECTOR, VECTORS['A']), rel=1e-6),
            IDS['B']: pytest.approx(_cosine(QUERY_VECTOR, VECTORS['B']), rel=1e-6),
        }

    @pytest.mark.asyncio
    async def test_no_ids_is_no_embedding_and_no_scores(self, mock_config, qdrant, embedder):
        backend = Mem0Backend(mock_config)
        qdrant_patch, instance_patch = _bind(backend, qdrant, embedder)
        with qdrant_patch, instance_patch:
            scores = await backend.cosine_to_points('the write', [], Scope(project_id=PROJECT))

        assert scores == {}
        assert embedder.texts == []

    @pytest.mark.asyncio
    async def test_the_embedder_scores_the_query_text(self, mock_config, qdrant, embedder):
        backend = Mem0Backend(mock_config)
        qdrant_patch, instance_patch = _bind(backend, qdrant, embedder)
        with qdrant_patch, instance_patch:
            await backend.cosine_to_points('the write', [IDS['A']], Scope(project_id=PROJECT))

        assert embedder.texts == ['the write']

    @pytest.mark.asyncio
    async def test_a_missing_collection_reads_as_no_scores(self, mock_config, embedder):
        backend = Mem0Backend(mock_config)
        client = MagicMock()
        client.query_points = AsyncMock(side_effect=missing_collection_404(_collection(mock_config)))
        qdrant_patch, instance_patch = _bind(backend, client, embedder)
        with qdrant_patch, instance_patch:
            scores = await backend.cosine_to_points('the write', [IDS['A']], Scope(project_id=PROJECT))

        assert scores == {}


class TestCosineToMemories:
    @pytest.mark.asyncio
    async def test_the_service_scopes_the_lookup_to_the_project(self, mock_config, qdrant, embedder):
        service = MemoryService(mock_config)
        qdrant_patch, instance_patch = _bind(service.mem0, qdrant, embedder)
        with qdrant_patch, instance_patch:
            scores = await service.cosine_to_memories(
                query='the write', memory_ids=[IDS['B']], project_id=PROJECT,
            )

        assert scores == {IDS['B']: pytest.approx(_cosine(QUERY_VECTOR, VECTORS['B']), rel=1e-6)}
