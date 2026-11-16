"""mem0's vector-store Qdrant client runs under Mem0Backend's own budgets.

Measured in the qdrant-integration lane pass 20261116T021649-ead5ad (task
10348): three tests logged ``Mem0 write failed: timed out`` for mem0's point
upsert (``PUT /collections/<c>/points?wait=true``), which Qdrant answered 200 a
little over 5s later.  mem0 had issued it through a QdrantClient it built with
no timeout, so httpx's 5s library default fired beneath the configured write
budget, and the failure named no budget at all.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import httpx
import pytest
from _mock_openai_server import mock_openai_server
from _stub_qdrant_server import stub_qdrant_server

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.scope import Scope

SCOPE = Scope(project_id='vector_store_client_probe')
MEMORY_ID = '00000000-0000-4000-8000-000000000001'
CONTENT = 'A slow Qdrant leg fails on its configured budget, never on a hidden one.'


@pytest.fixture
def config(mock_config, monkeypatch) -> Iterator[FusedMemoryConfig]:
    """mem0 telemetry off, and mem0's embedder served by a local mock."""
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)
    monkeypatch.setattr('mem0.memory.main.MEM0_TELEMETRY', False)
    config = mock_config.model_copy(deep=True)
    embedder = config.embedder.providers.openai
    assert embedder is not None
    with mock_openai_server() as server:
        embedder.api_url = server.base_url
        yield config


@pytest.fixture
def target(config) -> str:
    return SCOPE.mem0_collection_name(config.mem0.collection_prefix)


@pytest.fixture
def default_read() -> float:
    """httpx's library-default read timeout: the hidden deadline mem0's own client carried."""
    with httpx.Client() as client:
        read = client.timeout.read
    assert read is not None
    return read


@asynccontextmanager
async def backend_against(qdrant_url: str, config: FusedMemoryConfig) -> AsyncIterator[Mem0Backend]:
    config.mem0.qdrant_url = qdrant_url
    backend = Mem0Backend(config)
    try:
        yield backend
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_an_add_slower_than_the_http_client_default_still_lands(config, target, default_read):
    config.queue.backend_read_timeout_seconds = 1.0
    with stub_qdrant_server(upsert_delay_s={target: default_read + 0.5}) as stub:
        async with backend_against(stub.url, config) as backend:
            result = await backend.add(CONTENT, SCOPE)
        [added] = result['results']
        assert added['id']
        assert stub.upserts(target) == 1


@pytest.mark.asyncio
async def test_a_get_slower_than_the_http_client_default_still_answers(config, target, default_read):
    config.queue.backend_write_timeout_seconds = 1.0
    with stub_qdrant_server(retrieve_delay_s={target: default_read + 0.5}) as stub:
        async with backend_against(stub.url, config) as backend:
            assert await backend.get(MEMORY_ID, SCOPE) is None


@pytest.mark.asyncio
async def test_an_add_overrunning_its_write_budget_raises_a_timeout_naming_it(config, target, default_read):
    budget = default_read + 0.5
    config.queue.backend_write_timeout_seconds = budget
    with stub_qdrant_server(upsert_delay_s={target: 60.0}) as stub:
        async with backend_against(stub.url, config) as backend:
            with pytest.raises(TimeoutError, match=re.escape(f'Mem0 add timed out after {budget}s')):
                await backend.add(CONTENT, SCOPE)
            stub.release_event.set()
            assert stub.wait_upserted(target, timeout=10)
