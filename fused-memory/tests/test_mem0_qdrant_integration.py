"""Integration tests for Mem0 ↔ Qdrant — requires a running Qdrant instance.

These tests verify that the real qdrant-client version is compatible with
mem0's vector store operations, including metadata-only updates that
previously broke with qdrant-client 1.17+ and mem0ai <1.0.10.

Skip automatically when Qdrant is not reachable.
"""

from __future__ import annotations

import contextlib
from typing import cast

import pytest
from _fm_helpers import QDRANT_URL, ensure_fresh_collection, qdrant_skipif
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

_COLLECTION_PREFIX = '_test_mem0_qdrant_integration'
VECTOR_DIM = 8  # tiny vectors for speed


pytestmark = [
    qdrant_skipif(),
    pytest.mark.integration,
    pytest.mark.timeout(30),
]


@pytest.fixture
def test_collection(worker_id):
    """Per-worker collection name to avoid races under pytest-xdist."""
    return f'{_COLLECTION_PREFIX}_{worker_id}'


@pytest.fixture
def qdrant(test_collection):
    """Provide a QdrantClient and clean up the test collection after each test."""
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    # Ensure clean state — idempotent + 409-tolerant so a prior run's swallowed
    # teardown self-heals (delete + recreate) rather than 409-failing this run.
    ensure_fresh_collection(client, test_collection, size=VECTOR_DIM, distance=Distance.COSINE)
    yield client
    with contextlib.suppress(Exception):
        client.delete_collection(test_collection)
    client.close()


class TestPointStructCompat:
    """Verify qdrant-client PointStruct accepts the patterns mem0 uses."""

    def test_upsert_with_vector(self, qdrant: QdrantClient, test_collection: str):
        """Standard upsert with id + vector + payload (mem0 insert path)."""
        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=1,
            vector=[0.1] * VECTOR_DIM,
            payload={'text': 'hello', 'user_id': 'u1'},
        )
        qdrant.upsert(collection_name=test_collection, points=[point])

        result = qdrant.retrieve(test_collection, ids=[1], with_payload=True)
        assert len(result) == 1
        assert result[0].payload is not None
        assert result[0].payload['text'] == 'hello'

    def test_set_payload_without_vector(self, qdrant: QdrantClient, test_collection: str):
        """Payload-only update via set_payload (mem0 >=1.0.10 update path).

        This is the fix for the PointStruct(vector=None) crash.
        """
        from qdrant_client.models import PointStruct

        # Insert initial point
        qdrant.upsert(
            collection_name=test_collection,
            points=[PointStruct(id=1, vector=[0.1] * VECTOR_DIM, payload={'v': '1'})],
        )

        # Update payload only — no vector
        qdrant.set_payload(
            collection_name=test_collection,
            payload={'v': '2', 'agent_id': 'new-agent'},
            points=[1],
        )

        result = qdrant.retrieve(test_collection, ids=[1], with_payload=True, with_vectors=True)
        assert result[0].payload is not None
        assert result[0].payload['v'] == '2'
        assert result[0].payload['agent_id'] == 'new-agent'
        # Vector unchanged
        assert result[0].vector is not None
        assert len(result[0].vector) == VECTOR_DIM


class TestPayloadPrimitivesPreserveIdentity:
    """Live round-trip: all three payload primitives preserve id + created_at.

    Task 3088's metadata-only arms exist so tagging a record does NOT re-embed
    or perturb its identity. Asserted here against a REAL Qdrant, because the
    unit tests can only prove which client method was called, not that the
    server actually leaves the point id and created_at alone.
    """

    CREATED_AT = '2026-01-01T00:00:00+00:00'

    def _seed(self, qdrant: QdrantClient, test_collection: str):
        from qdrant_client.models import PointStruct

        qdrant.upsert(
            collection_name=test_collection,
            points=[PointStruct(id=1, vector=[0.1] * VECTOR_DIM, payload={
                'data': 'original', 'created_at': self.CREATED_AT,
                'kind': 'canonical', 'src_project': 'reify',
            })],
        )

    def _payload(self, qdrant: QdrantClient, test_collection: str) -> dict:
        result = qdrant.retrieve(test_collection, ids=[1], with_payload=True)
        assert len(result) == 1, f'point id not preserved: {result!r}'
        assert result[0].id == 1
        return cast(dict, result[0].payload)

    def test_set_payload_merges_and_preserves_created_at(
        self, qdrant: QdrantClient, test_collection: str,
    ):
        self._seed(qdrant, test_collection)
        qdrant.set_payload(
            collection_name=test_collection, payload={'topic': 'cluster-a'}, points=[1],
        )
        payload = self._payload(qdrant, test_collection)
        assert payload['topic'] == 'cluster-a'
        assert payload['created_at'] == self.CREATED_AT
        # A genuine storage-layer partial merge: unlisted keys survive.
        assert payload['kind'] == 'canonical'
        assert payload['src_project'] == 'reify'

    def test_delete_payload_removes_only_named_keys(
        self, qdrant: QdrantClient, test_collection: str,
    ):
        self._seed(qdrant, test_collection)
        qdrant.delete_payload(collection_name=test_collection, keys=['kind'], points=[1])
        payload = self._payload(qdrant, test_collection)
        assert 'kind' not in payload
        assert payload['created_at'] == self.CREATED_AT
        assert payload['src_project'] == 'reify'

    def test_overwrite_payload_replaces_whole_payload(
        self, qdrant: QdrantClient, test_collection: str,
    ):
        self._seed(qdrant, test_collection)
        # overwrite_payload replaces the WHOLE payload -- created_at survives
        # only because it is re-attached, which is exactly why the service's
        # replace arm must read-modify-write rather than write blind.
        qdrant.overwrite_payload(
            collection_name=test_collection,
            payload={'data': 'original', 'created_at': self.CREATED_AT, 'topic': 'cluster-b'},
            points=[1],
        )
        payload = self._payload(qdrant, test_collection)
        assert payload['created_at'] == self.CREATED_AT
        assert payload['topic'] == 'cluster-b'
        assert 'kind' not in payload, 'overwrite must not silently retain dropped keys'


class TestMem0VectorStoreUpdate:
    """Exercise mem0's own Qdrant vector store update() method against real Qdrant.

    This is the code path that broke: mem0/vector_stores/qdrant.py update()
    was passing vector=None to PointStruct, which qdrant-client 1.17+ rejects.
    """

    def test_update_payload_only(self, qdrant: QdrantClient, test_collection: str):
        """mem0 update(vector_id, vector=None, payload={...}) must not raise."""
        from mem0.vector_stores.qdrant import Qdrant as Mem0Qdrant

        store = Mem0Qdrant.__new__(Mem0Qdrant)
        store.client = qdrant
        store.collection_name = test_collection

        # Insert a point first
        from qdrant_client.models import PointStruct

        qdrant.upsert(
            collection_name=test_collection,
            points=[PointStruct(
                id=42,
                vector=[0.5] * VECTOR_DIM,
                payload={'text': 'original', 'agent_id': 'old'},
            )],
        )

        # This is the exact call that crashed with mem0ai <1.0.10 + qdrant-client >=1.17
        # cast: mem0's update() signature declares vector: list but accepts None at runtime
        store.update(vector_id=42, vector=cast(list, None), payload={'text': 'original', 'agent_id': 'new'})

        result = qdrant.retrieve(test_collection, ids=[42], with_payload=True, with_vectors=True)
        assert result[0].payload is not None
        assert result[0].payload['agent_id'] == 'new'
        assert result[0].payload['text'] == 'original'
        # Embedding preserved (Qdrant normalizes cosine vectors, so check non-None + length)
        assert result[0].vector is not None
        assert len(result[0].vector) == VECTOR_DIM

    def test_update_vector_only(self, qdrant: QdrantClient, test_collection: str):
        """mem0 update(vector_id, vector=[...], payload=None) must not raise."""
        from mem0.vector_stores.qdrant import Qdrant as Mem0Qdrant

        store = Mem0Qdrant.__new__(Mem0Qdrant)
        store.client = qdrant
        store.collection_name = test_collection

        from qdrant_client.models import PointStruct

        qdrant.upsert(
            collection_name=test_collection,
            points=[PointStruct(
                id=43,
                vector=[0.1] * VECTOR_DIM,
                payload={'text': 'keep me'},
            )],
        )

        new_vec = [0.9] * VECTOR_DIM
        # cast: mem0's update() signature declares payload: dict but accepts None at runtime
        store.update(vector_id=43, vector=new_vec, payload=cast(dict, None))

        result = qdrant.retrieve(test_collection, ids=[43], with_payload=True, with_vectors=True)
        assert result[0].vector is not None
        assert len(result[0].vector) == VECTOR_DIM
        assert result[0].payload is not None
        assert result[0].payload['text'] == 'keep me'

    def test_update_both(self, qdrant: QdrantClient, test_collection: str):
        """mem0 update(vector_id, vector=[...], payload={...}) uses upsert."""
        from mem0.vector_stores.qdrant import Qdrant as Mem0Qdrant

        store = Mem0Qdrant.__new__(Mem0Qdrant)
        store.client = qdrant
        store.collection_name = test_collection

        from qdrant_client.models import PointStruct

        qdrant.upsert(
            collection_name=test_collection,
            points=[PointStruct(id=44, vector=[0.1] * VECTOR_DIM, payload={'v': '1'})],
        )

        store.update(vector_id=44, vector=[0.8] * VECTOR_DIM, payload={'v': '2'})

        result = qdrant.retrieve(test_collection, ids=[44], with_payload=True, with_vectors=True)
        assert result[0].vector is not None
        assert len(result[0].vector) == VECTOR_DIM
        assert result[0].payload is not None
        assert result[0].payload['v'] == '2'
