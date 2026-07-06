"""Empirical probe (task 2221, W5-γ): does a recon infer=False system write
ever get silently dedup-dropped by Mem0 (§2.1's '~0.92 cosine' premise)?

FINDING — PREMISE FALSE. Mem0Backend.add pins infer=False, and mem0's
infer=False write path never runs a similarity search or an update-vs-add
choice — it unconditionally mints a fresh id and inserts a new point per
message. A message is skipped only if it is malformed or has role=='system',
which recon's plain-string add() calls never produce (mem0 always
normalizes a str payload to a single role='user' message). So a repeated,
byte-identical infer=False write is never dropped for being a near-duplicate.

Confirmed empirically below against a real Qdrant, twice: once with a
stubbed, identical-vector embedder (cosine=1.0 — a strictly stronger worst
case than the survey's ~0.92), and again with real OpenAI embeddings when
OPENAI_API_KEY is available. In both, every one of N byte-identical
infer=False writes lands as a distinct point (N distinct ids and
backend.count(scope)==N). See task 2221's memory notes for the full
code-trace citations backing this finding.

Consequence for downstream tasks: δ's add_system_record is a no-op
hardening (the premise it "fixes" is already false), and λ's deletion
logic should stay ledger-authority-anchored rather than assuming Mem0
dedup as a safety net either way.
"""
from __future__ import annotations

import contextlib
import os

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import VectorParams

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.models.scope import Scope

QDRANT_URL = 'http://localhost:6333'

FIXED_RECON_SUMMARY = (
    'Stage 2 cycle summary for run task-2221-premise-probe — '
    'byte-identical fixture content issued N times to test whether '
    'Mem0.add(infer=False) ever dedup-drops a repeated recon system write.'
)


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
    pytest.mark.timeout(60),
]


@pytest.fixture
def recon_scope(worker_id) -> Scope:
    """A recon-stage scope isolated per xdist worker."""
    return Scope(
        project_id=f'_test_recon_dedup_{worker_id}',
        agent_id='recon-stage-task_knowledge_sync',
    )


@pytest.fixture
def clean_collection(recon_scope, mock_config):
    """Delete the recon-scope's Qdrant collection before AND after the test."""
    collection = recon_scope.mem0_collection_name(mock_config.mem0.collection_prefix)
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    with contextlib.suppress(ResponseHandlingException, UnexpectedResponse):
        client.delete_collection(collection)
    yield collection
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    client.close()


def _collection_vector_size(collection: str) -> int:
    """Read back the vector dimension Qdrant actually created for *collection*.

    This assumes mem0's Qdrant vector store creates the collection eagerly —
    mem0.vector_stores.qdrant.QdrantDB.__init__ calls create_col()
    unconditionally, so by the time _build_recon_backend has awaited
    backend._get_instance(scope), the collection already exists. If a future
    mem0 release makes collection creation lazy (e.g. deferred to the first
    add()), fail loudly with a message that explains the assumption instead
    of leaking a raw 404 traceback.
    """
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        try:
            info = client.get_collection(collection)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise AssertionError(
                f'collection {collection!r} does not exist yet. This harness '
                "assumes mem0's AsyncMemory.from_config() eagerly creates the "
                'Qdrant collection during _get_instance(); if a mem0 upgrade '
                'made collection creation lazy instead, this stub-dimension '
                'lookup needs reworking (e.g. force a real add() before '
                'reading the dimension back).'
            ) from exc
        vectors = info.config.params.vectors
        # mem0's Qdrant vector store always creates collections with a single
        # unnamed vector config (see mem0.vector_stores.qdrant.create_col), never
        # named vectors or none — assert that so a future mem0 change that breaks
        # this assumption fails loudly here instead of silently misreading dims.
        assert isinstance(vectors, VectorParams), (
            f'expected an unnamed VectorParams config, got {vectors!r}'
        )
        return vectors.size
    finally:
        client.close()


async def _build_recon_backend(mock_config, scope, monkeypatch, *, real_embedder=False):
    """Construct a real Mem0Backend and force-create its AsyncMemory instance.

    Stubs out everything that isn't the dedup question itself: mem0's
    telemetry capture (offline safety), and the shared SQLite history writer
    (xdist-contended, irrelevant to vector-store dedup). Unless
    *real_embedder* is set, also stubs the embedder to return one constant
    vector for every call — the strongest-possible (cosine=1.0) duplicate
    input the infer=False path could ever see.
    """
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)

    config = mock_config.model_copy(deep=True)
    if real_embedder:
        # mock_config's embedder.providers.openai.api_key is a fake fixture
        # value ('test-key'); clearing it here makes mem0's OpenAIEmbedding
        # fall back to the real OPENAI_API_KEY env var instead.
        config.embedder.providers.openai.api_key = None

    backend = Mem0Backend(config)
    inst = await backend._get_instance(scope)
    inst.db.add_history = lambda *a, **kw: None

    if not real_embedder:
        collection = scope.mem0_collection_name(config.mem0.collection_prefix)
        dim = _collection_vector_size(collection)
        stub_vector = [0.1] * dim
        inst.embedding_model.embed = lambda *a, **kw: stub_vector

    return backend


async def _assert_all_writes_land_distinct(backend, scope, *, n: int, run_id: str) -> None:
    """Issue *n* byte-identical infer=False writes and assert none dedup-drop.

    Shared by both tests below: asserts each write returns exactly one
    result (the infer=False shape), that all *n* returned ids are distinct
    (no in-place update in lieu of a fresh point), and that the collection's
    exact point count is exactly *n* (no silent drop).
    """
    metadata = {
        'kind': 'cycle_summary',
        'stage': 'task_knowledge_sync',
        'run_id': run_id,
    }
    ids = []
    for _ in range(n):
        response = await backend.add(
            content=FIXED_RECON_SUMMARY,
            scope=scope,
            metadata=metadata,
        )
        results = response.get('results') or []
        assert len(results) == 1, (
            f'expected exactly one result under infer=False, got {results!r}'
        )
        assert 'id' in results[0]
        ids.append(results[0]['id'])

    assert len(set(ids)) == n, (
        f'expected {n} distinct ids (no dedup drop), got {len(set(ids))} distinct: {ids!r}'
    )
    assert await backend.count(scope) == n


@pytest.mark.asyncio
async def test_identical_infer_false_writes_all_land_distinct(
    mock_config, recon_scope, clean_collection, monkeypatch,
):
    """N byte-identical recon-stage infer=False writes must all land distinct.

    Issues N=8 identical Mem0Backend.add(...) calls (content + metadata both
    fixed — no nonce) through the real production write path (infer=False
    pinned in Mem0Backend.add) against a real, isolated Qdrant collection.
    If the §2.1 premise ('~0.92 cosine dedup can drop a repeat recon write')
    were true, at least one of these 8 writes would return zero results
    and/or the collection would end up with fewer than 8 points.
    """
    backend = await _build_recon_backend(mock_config, recon_scope, monkeypatch)
    try:
        await _assert_all_writes_land_distinct(
            backend, recon_scope, n=8, run_id='task-2221-premise-probe-hermetic',
        )
    finally:
        await backend.close()


@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason='real embedder needs OPENAI_API_KEY')
@pytest.mark.asyncio
async def test_identical_writes_land_with_real_openai_embeddings(
    mock_config, recon_scope, clean_collection, monkeypatch,
):
    """Real-OpenAI-embedder confirmation: identical text still never dedups.

    Same byte-identical-write probe as
    test_identical_infer_false_writes_all_land_distinct, but with NO
    embedder stub — genuine OpenAI embeddings for identical input text
    land at real cosine≈1.0. This directly, empirically rebuts the
    survey's '~0.92 cosine similarity dedup' observation: even the
    strongest realistic near-duplicate signal never triggers a drop,
    because the infer=False path never consults embeddings for dedup.
    """
    backend = await _build_recon_backend(mock_config, recon_scope, monkeypatch, real_embedder=True)
    try:
        await _assert_all_writes_land_distinct(
            backend, recon_scope, n=5, run_id='task-2221-premise-probe-real-embedder',
        )
    finally:
        await backend.close()
