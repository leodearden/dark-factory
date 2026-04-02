"""Integration tests for Mem0 ↔ Qdrant — requires a running Qdrant instance.

These tests verify that the real qdrant-client version is compatible with
mem0's vector store operations, including metadata-only updates that
previously broke with qdrant-client 1.17+ and mem0ai <1.0.10.

Skip automatically when Qdrant is not reachable.
"""

from __future__ import annotations

import contextlib

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = 'http://localhost:6333'
TEST_COLLECTION = '_test_mem0_qdrant_integration'
VECTOR_DIM = 8  # tiny vectors for speed


def _qdrant_available() -> bool:
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2)
        client.get_collections()
        client.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _qdrant_available(), reason='Qdrant not reachable'),
    pytest.mark.timeout(30),
]


@pytest.fixture
def qdrant():
    """Provide a QdrantClient and clean up the test collection after each test."""
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    # Ensure clean state
    with contextlib.suppress(ResponseHandlingException, UnexpectedResponse):
        client.delete_collection(TEST_COLLECTION)
    client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    yield client
    with contextlib.suppress(Exception):
        client.delete_collection(TEST_COLLECTION)
    client.close()


class TestPointStructCompat:
    """Verify qdrant-client PointStruct accepts the patterns mem0 uses."""

    def test_upsert_with_vector(self, qdrant: QdrantClient):
        """Standard upsert with id + vector + payload (mem0 insert path)."""
        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=1,
            vector=[0.1] * VECTOR_DIM,
            payload={'text': 'hello', 'user_id': 'u1'},
        )
        qdrant.upsert(collection_name=TEST_COLLECTION, points=[point])

        result = qdrant.retrieve(TEST_COLLECTION, ids=[1], with_payload=True)
        assert len(result) == 1
        assert result[0].payload['text'] == 'hello'

    def test_set_payload_without_vector(self, qdrant: QdrantClient):
        """Payload-only update via set_payload (mem0 >=1.0.10 update path).

        This is the fix for the PointStruct(vector=None) crash.
        """
        from qdrant_client.models import PointStruct

        # Insert initial point
        qdrant.upsert(
            collection_name=TEST_COLLECTION,
            points=[PointStruct(id=1, vector=[0.1] * VECTOR_DIM, payload={'v': '1'})],
        )

        # Update payload only — no vector
        qdrant.set_payload(
            collection_name=TEST_COLLECTION,
            payload={'v': '2', 'agent_id': 'new-agent'},
            points=[1],
        )

        result = qdrant.retrieve(TEST_COLLECTION, ids=[1], with_payload=True, with_vectors=True)
        assert result[0].payload['v'] == '2'
        assert result[0].payload['agent_id'] == 'new-agent'
        # Vector unchanged
        assert len(result[0].vector) == VECTOR_DIM


class TestMem0VectorStoreUpdate:
    """Exercise mem0's own Qdrant vector store update() method against real Qdrant.

    This is the code path that broke: mem0/vector_stores/qdrant.py update()
    was passing vector=None to PointStruct, which qdrant-client 1.17+ rejects.
    """

    def test_update_payload_only(self, qdrant: QdrantClient):
        """mem0 update(vector_id, vector=None, payload={...}) must not raise."""
        from mem0.vector_stores.qdrant import Qdrant as Mem0Qdrant

        store = Mem0Qdrant.__new__(Mem0Qdrant)
        store.client = qdrant
        store.collection_name = TEST_COLLECTION

        # Insert a point first
        from qdrant_client.models import PointStruct

        qdrant.upsert(
            collection_name=TEST_COLLECTION,
            points=[PointStruct(
                id=42,
                vector=[0.5] * VECTOR_DIM,
                payload={'text': 'original', 'agent_id': 'old'},
            )],
        )

        # This is the exact call that crashed with mem0ai <1.0.10 + qdrant-client >=1.17
        store.update(vector_id=42, vector=None, payload={'text': 'original', 'agent_id': 'new'})

        result = qdrant.retrieve(TEST_COLLECTION, ids=[42], with_payload=True, with_vectors=True)
        assert result[0].payload['agent_id'] == 'new'
        assert result[0].payload['text'] == 'original'
        # Embedding preserved (Qdrant normalizes cosine vectors, so check non-None + length)
        assert result[0].vector is not None
        assert len(result[0].vector) == VECTOR_DIM

    def test_update_vector_only(self, qdrant: QdrantClient):
        """mem0 update(vector_id, vector=[...], payload=None) must not raise."""
        from mem0.vector_stores.qdrant import Qdrant as Mem0Qdrant

        store = Mem0Qdrant.__new__(Mem0Qdrant)
        store.client = qdrant
        store.collection_name = TEST_COLLECTION

        from qdrant_client.models import PointStruct

        qdrant.upsert(
            collection_name=TEST_COLLECTION,
            points=[PointStruct(
                id=43,
                vector=[0.1] * VECTOR_DIM,
                payload={'text': 'keep me'},
            )],
        )

        new_vec = [0.9] * VECTOR_DIM
        store.update(vector_id=43, vector=new_vec, payload=None)

        result = qdrant.retrieve(TEST_COLLECTION, ids=[43], with_payload=True, with_vectors=True)
        assert result[0].vector is not None
        assert len(result[0].vector) == VECTOR_DIM
        assert result[0].payload['text'] == 'keep me'

    def test_update_both(self, qdrant: QdrantClient):
        """mem0 update(vector_id, vector=[...], payload={...}) uses upsert."""
        from mem0.vector_stores.qdrant import Qdrant as Mem0Qdrant

        store = Mem0Qdrant.__new__(Mem0Qdrant)
        store.client = qdrant
        store.collection_name = TEST_COLLECTION

        from qdrant_client.models import PointStruct

        qdrant.upsert(
            collection_name=TEST_COLLECTION,
            points=[PointStruct(id=44, vector=[0.1] * VECTOR_DIM, payload={'v': '1'})],
        )

        store.update(vector_id=44, vector=[0.8] * VECTOR_DIM, payload={'v': '2'})

        result = qdrant.retrieve(TEST_COLLECTION, ids=[44], with_payload=True, with_vectors=True)
        assert result[0].vector is not None
        assert len(result[0].vector) == VECTOR_DIM
        assert result[0].payload['v'] == '2'
