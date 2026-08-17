"""Seeded end-to-end probe for the RRF cross-store merge (task 3658).

THE α USER-OBSERVABLE SIGNAL. Before this task, ``MemoryService.search``
sorted ``(is_primary, -relevance_score)``: the router's primary store came
first WHOLESALE, so when Graphiti filled ``limit`` the ``results[:limit]``
truncation discarded every Mem0 hit no matter how good its cosine. A user
asking a Graphiti-primary question could not reach a perfectly-matching Mem0
memory at all. This module asserts, against a REAL embedder and a REAL
Qdrant, that a seeded Mem0 needle now reaches the merged top-5 — the thing an
agent actually notices. It fails on main.

It is NOT the merge-lane gate. ``fused-memory/pyproject.toml``'s
``addopts = -m 'not integration'`` deselects this whole module, so the
ordering contract is pinned by the unit tests in ``tests/test_memory_service.py``
(``TestSearchRrfMerge``, ``TestSearchMem0RrfFields``, and the Graphiti RRF
tests). This module corroborates that contract with real cosines and real
ranks; it cannot substitute for it.

Why Mem0 is seeded for real and Graphiti is stubbed: under RRF, Graphiti
contributes only a RANK. Its public ``search()`` exposes no scores at all —
that is the very reason RRF was chosen over score calibration — so a real
FalkorDB plus LLM entity extraction would add slowness and flake while
contributing exactly the ordinal a stubbed edge list already provides. Every
property asserted here (real cosine, real rank, real fusion) survives that
choice.

Isolation, in the order it matters:
  - ``collection_prefix`` is ``_test_mem0_qdrant_integration``, the only
    prefix ``scripts/cleanup_test_collections.py`` reaps. Seeding under the
    default ``fused`` prefix would leak the collection forever. Asserted
    below against that script's own ``PREFIX`` constant, not a restated
    string.
  - the collection is deleted BEFORE and AFTER, so a swallowed teardown
    self-heals on the next run instead of poisoning it.
  - ``project_id`` is per-xdist-worker, so concurrent workers cannot collide.
"""

from __future__ import annotations

import contextlib
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import QDRANT_URL, MockEdge, qdrant_skipif
from qdrant_client import QdrantClient

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.models.enums import QueryType, SourceStore
from fused_memory.models.memory import ReadRouteResult
from fused_memory.models.scope import Scope
from fused_memory.services.memory_service import RRF_K, MemoryService

pytestmark = [
    qdrant_skipif(),
    pytest.mark.timeout(120),
    # Real Qdrant + real OpenAI embedder: the integration/offline lane per task
    # 3019 / the integration-test-lane PRD, never a merge-lane gate.
    pytest.mark.integration,
]

# The only prefix scripts/cleanup_test_collections.py reaps.
EPHEMERAL_COLLECTION_PREFIX = '_test_mem0_qdrant_integration'

# The needle: a distinctive, self-contained fact with a phrasing no distractor
# shares, so a real embedder ranks it first for the query below.
NEEDLE = (
    'The pre-commit hook runs pyright only for packages with staged Python '
    'changes, so a docs-only commit prints "pyright skipped" and returns '
    'quickly.'
)
NEEDLE_QUERY = 'why did pre-commit skip pyright on my docs-only commit'

# Well-separated distractors: different subject matter, so they neither crowd
# the needle out nor tie with it.
DISTRACTORS = (
    'Qdrant stores the vector embeddings for the Mem0 half of fused memory.',
    'The merge worker consumes the advance path in the shared git directory.',
    'Escalations promote to L2 when the auto-watcher cannot resolve them.',
)


@pytest.fixture
def rrf_project_id(worker_id):
    """Per-xdist-worker so concurrent workers cannot share a collection."""
    return f'rrf_merge_{worker_id}'


@pytest.fixture
def rrf_config(mock_config, rrf_project_id):
    """mock_config pointed at an ephemeral collection with a REAL embedder.

    Clearing the fake api_key makes mem0's OpenAIEmbedding fall back to the
    real OPENAI_API_KEY. A stub constant vector would make this probe
    meaningless: the whole claim is that a genuinely-matching Mem0 memory now
    reaches the merged top-5, which requires a genuine cosine.
    """
    config = mock_config.model_copy(deep=True)
    config.mem0.collection_prefix = EPHEMERAL_COLLECTION_PREFIX
    config.embedder.providers.openai.api_key = None
    return config


@pytest.fixture
def clean_collection(rrf_config, rrf_project_id):
    """Delete the seeded collection before AND after the test."""
    collection = Scope(project_id=rrf_project_id).mem0_collection_name(
        rrf_config.mem0.collection_prefix,
    )
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    yield collection
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    client.close()


def _service(config, mem0_backend, *, graphiti_edges: int, primary: SourceStore):
    """A MemoryService over the real Mem0 backend and a stubbed Graphiti.

    Built via ``__new__`` rather than the constructor: ``search`` touches only
    the attributes set here, and the real constructor would stand up a durable
    queue and its SQLite files for nothing.
    """
    svc = MemoryService.__new__(MemoryService)
    svc.config = config
    svc.mem0 = mem0_backend
    svc.planned_episode_registry = None
    svc._write_journal = None

    svc.graphiti = MagicMock()
    svc.graphiti.search = AsyncMock(return_value=[
        MockEdge(fact=f'Graphiti edge {n} about unrelated fleet redeploy', uuid=f'g-{n}')
        for n in range(1, graphiti_edges + 1)
    ])

    svc.router = MagicMock()
    svc.router.route = AsyncMock(return_value=ReadRouteResult(
        query_type=QueryType.broad,
        stores=[SourceStore.graphiti, SourceStore.mem0],
        primary_store=primary,
    ))
    return svc


def test_the_ephemeral_collection_is_one_the_reaper_can_reclaim(
    monkeypatch, rrf_config, rrf_project_id,
):
    """A leaked collection under the default `fused` prefix would live forever.

    Deliberately does NOT take ``clean_collection``: that fixture opens a real
    QdrantClient, and this assertion is about a NAME.
    ``mem0_collection_name`` is pure, so ask it directly.
    """
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path

    collection = Scope(project_id=rrf_project_id).mem0_collection_name(
        rrf_config.mem0.collection_prefix,
    )

    path = Path(__file__).resolve().parent.parent / 'scripts' / 'cleanup_test_collections.py'
    spec = _ilu.spec_from_file_location('cleanup_test_collections', path)
    assert spec is not None and spec.loader is not None
    cleanup = _ilu.module_from_spec(spec)
    # setitem, not a bare assignment: exec_module needs the module visible in
    # sys.modules, but leaving it there leaks into the rest of the session.
    monkeypatch.setitem(_sys.modules, 'cleanup_test_collections', cleanup)
    spec.loader.exec_module(cleanup)

    assert collection.startswith(cleanup.PREFIX), (
        f'{collection!r} is not reapable by scripts/cleanup_test_collections.py'
    )


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='the seeded probe needs a real embedder',
)
@pytest.mark.asyncio
async def test_seeded_mem0_needle_reaches_the_merged_top5(
    rrf_config, rrf_project_id, clean_collection, monkeypatch,
):
    """The α signal: a real Mem0 hit survives a Graphiti-primary merge.

    FAILS ON MAIN. With five Graphiti edges and ``limit=5``, the old
    ``(is_primary, -relevance_score)`` sort put all five first and truncation
    dropped the needle — 0/3 on the boundary the task was filed for.

    Opt-in via ``-m integration``: this test makes genuine OpenAI network
    calls. The skipif above is belt-and-braces for the addopts-cleared
    recovery/flaky-rerun path (verify_cmd.serial_pytest).
    """
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)

    scope = Scope(project_id=rrf_project_id)
    backend = Mem0Backend(rrf_config)
    inst = await backend._get_instance(scope)
    # Shared SQLite history writer: xdist-contended and irrelevant here.
    inst.db.add_history = lambda *a, **kw: None

    try:
        for content in (*DISTRACTORS, NEEDLE):
            await backend.add(
                content=content,
                scope=scope,
                metadata={'category': 'observations_and_summaries'},
            )

        svc = _service(
            rrf_config, backend, graphiti_edges=5, primary=SourceStore.graphiti,
        )
        merged = await svc.search(
            NEEDLE_QUERY, project_id=rrf_project_id, limit=5,
        )

        mem0_hits = [r for r in merged if r.source_store == SourceStore.mem0]
        assert mem0_hits, (
            'Mem0 was shut out of the merged top-5 entirely — the exact defect '
            'task 3658 fixes'
        )

        needle = next((r for r in merged if r.content == NEEDLE), None)
        assert needle is not None, (
            f'the seeded needle is missing from the merged top-5; got '
            f'{[(r.source_store.value, r.content[:40]) for r in merged]}'
        )

        # It is the Mem0 top hit, and its cosine is a genuine similarity.
        assert needle.metadata['store_rank'] == 1, (
            f'the needle should be Mem0 rank-1 for a self-phrased query, got '
            f'{needle.metadata["store_rank"]}'
        )
        cosine = needle.metadata['store_score']
        assert isinstance(cosine, float) and cosine > 0.5, (
            f'store_score must carry the real cosine, got {cosine!r}'
        )
        # ...and that cosine is NOT what ordered the merge.
        assert needle.relevance_score == pytest.approx(1.0 / (RRF_K + 1))
        assert needle.relevance_score != pytest.approx(cosine)

        graphiti_hits = [r for r in merged if r.source_store == SourceStore.graphiti]
        assert graphiti_hits, 'the stubbed Graphiti side should still contribute'
        for r in graphiti_hits:
            assert r.metadata['store_score'] is None, (
                'Graphiti exposes no scores; store_score must be None'
            )
            for stale in (1.0, 0.95, 0.90):
                assert r.relevance_score != pytest.approx(stale), (
                    f'the synthesized score {stale} is still being produced'
                )

        for r in merged:
            assert isinstance(r.metadata['store_rank'], int), (
                'every result must expose an int store_rank'
            )
    finally:
        await backend.close()
