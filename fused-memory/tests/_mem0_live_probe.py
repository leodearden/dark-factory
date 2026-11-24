"""A real Mem0Backend against live Qdrant, run-isolated in a reused,
reaper-reclaimable collection.

A probe's collection is shared by every xdist worker and session and reused
across runs; scripts/cleanup_test_collections.py reaps it, and each run is
isolated by its own run_id.

Lives outside conftest.py, imported by name, under the convention stated in
_fm_helpers' module docstring.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator

from _fm_qdrant_live import (
    EPHEMERAL_COLLECTION_PREFIX,
    QDRANT_URL,
    assert_reapable,
    collection_vector_size,
)
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.scope import Scope


def reapable_config(config: FusedMemoryConfig) -> FusedMemoryConfig:
    """A copy of *config* with the collection prefix scripts/cleanup_test_collections.py reaps."""
    reapable = config.model_copy(deep=True)
    reapable.mem0.collection_prefix = EPHEMERAL_COLLECTION_PREFIX
    return reapable


def probe_collection(scope: Scope, config: FusedMemoryConfig) -> str:
    return scope.mem0_collection_name(config.mem0.collection_prefix)


def isolated_run(scope: Scope, config: FusedMemoryConfig, label: str) -> Iterator[str]:
    """Yield a fresh run_id, then best-effort delete that run's points; the collection stays."""
    run_id = f'{label}-{uuid.uuid4().hex}'
    yield run_id
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    with contextlib.suppress(Exception):
        client.delete(
            collection_name=probe_collection(scope, config),
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key='run_id', match=MatchValue(value=run_id))],
                ),
            ),
        )
    client.close()


async def build_live_backend(
    scope: Scope, config: FusedMemoryConfig, monkeypatch,
) -> Mem0Backend:
    """Construct a real Mem0Backend and build its AsyncMemory instance for *scope*.

    Refuses, before anything is written, a config whose collection the reaper
    would not reclaim.
    """
    assert_reapable(probe_collection(scope, config))
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)

    backend = Mem0Backend(config.model_copy(deep=True))
    await backend._get_instance(scope)
    return backend


async def build_stub_embedder_backend(
    scope: Scope, config: FusedMemoryConfig, monkeypatch,
) -> Mem0Backend:
    """:func:`build_live_backend` with a constant-vector embedder: every write is a cosine=1.0 duplicate."""
    backend = await build_live_backend(scope, config, monkeypatch)
    inst = await backend._get_instance(scope)

    # Race-safe vector-dim read: under `-n auto` load the just-created
    # collection can transiently 404 before it is durably readable, so retry
    # the read via the shared bounded-poll helper rather than a single get.
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        dim = await collection_vector_size(client, probe_collection(scope, config))
    finally:
        client.close()
    stub_vector = [0.1] * dim
    inst.embedding_model.embed = lambda *a, **kw: stub_vector

    return backend
