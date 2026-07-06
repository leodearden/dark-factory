"""Empirical probe (task 2221, W5-γ): does a recon infer=False system write
ever get silently dedup-dropped by Mem0 (§2.1's '~0.92 cosine' premise)?

FINDING — PREMISE FALSE. Mem0Backend.add pins infer=False (mem0_client.py:84-112),
and mem0's AsyncMemory._add_to_vector_store infer=False branch (vendored
mem0/memory/main.py:1589-1624) iterates messages, skips ONLY a message dict
that is malformed or has role=='system', and otherwise calls _create_memory
UNCONDITIONALLY for every remaining message. Async _create_memory (main.py:
2356-2393) mints a fresh memory_id=str(uuid.uuid4()) and inserts a new Qdrant
point every time — it stores an md5 hash of the content but never searches,
never compares cosine similarity, and never chooses update-vs-add. That
near-duplicate/update logic exists ONLY in the infer=True branch (main.py:
1626+), which this write path never takes. So a successful infer=False write
always returns exactly one result with a fresh id — precisely the
_MEM0_ADD_INFER_PINNED_FALSE invariant (memory_service.py:68, task 1974).

The ONLY way an infer=False write drops a message is if that message dict
has role=='system' or is malformed (missing role/content) — mem0's own
str-content normalization (main.py:1542-1543) always wraps a plain string
in {'role': 'user', 'content': ...}, which recon's string-content call sites
(e.g. task_knowledge_sync.py's add_memory calls) always produce. So that
drop condition is unreachable from recon's actual call shape.

Empirically confirmed below by a hermetic harness with a stubbed,
IDENTICAL-vector embedder (cosine=1.0 — a strictly stronger worst case than
the survey's ~0.92) against a real Qdrant: every one of N byte-identical
infer=False writes lands as a distinct point. (A second, real-OpenAI-
embedder confirmation is added alongside this — see that test's docstring
for the additional empirical result once it lands.)

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
    """Read back the vector dimension Qdrant actually created for *collection*."""
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        info = client.get_collection(collection)
        return info.config.params.vectors.size
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

    backend = Mem0Backend(mock_config)
    inst = await backend._get_instance(scope)
    inst.db.add_history = lambda *a, **kw: None

    if not real_embedder:
        collection = scope.mem0_collection_name(mock_config.mem0.collection_prefix)
        dim = _collection_vector_size(collection)
        stub_vector = [0.1] * dim
        inst.embedding_model.embed = lambda *a, **kw: stub_vector

    return backend


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
    n = 8
    backend = await _build_recon_backend(mock_config, recon_scope, monkeypatch)
    try:
        ids = []
        metadata = {
            'kind': 'cycle_summary',
            'stage': 'task_knowledge_sync',
            'run_id': 'task-2221-premise-probe-hermetic',
        }
        for _ in range(n):
            response = await backend.add(
                content=FIXED_RECON_SUMMARY,
                scope=recon_scope,
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
        assert await backend.count(recon_scope) == n
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
    n = 5
    backend = await _build_recon_backend(mock_config, recon_scope, monkeypatch, real_embedder=True)
    try:
        ids = []
        metadata = {
            'kind': 'cycle_summary',
            'stage': 'task_knowledge_sync',
            'run_id': 'task-2221-premise-probe-real-embedder',
        }
        for _ in range(n):
            response = await backend.add(
                content=FIXED_RECON_SUMMARY,
                scope=recon_scope,
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
        assert await backend.count(recon_scope) == n
    finally:
        await backend.close()
