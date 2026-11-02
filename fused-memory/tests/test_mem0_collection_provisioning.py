"""Mem0Backend provisions a project's Qdrant collection under its own budget.

Measured in the qdrant-integration lane passes 20261030T180141-9fb982 and
20261102T040927-1770f9 (task 8190): Qdrant runs collection creates one after
another, and under 8 workers it answered 200 to the failing creates at 5.31s
and 5.86s.  mem0 had issued them through a QdrantClient it built with no
timeout, so httpx's 5s library default gave up first and aborted a create the
server went on to complete.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from _stub_qdrant_server import stub_qdrant_server

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.scope import Scope

SCOPE = Scope(project_id='provisioning_probe')
MEMORY_ID = '00000000-0000-4000-8000-000000000001'


@pytest.fixture
def config(mock_config, monkeypatch) -> FusedMemoryConfig:
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)
    return mock_config.model_copy(deep=True)


@pytest.fixture
def target(config) -> str:
    return SCOPE.mem0_collection_name(config.mem0.collection_prefix)


@asynccontextmanager
async def backend_against(qdrant_url: str, config: FusedMemoryConfig) -> AsyncIterator[Mem0Backend]:
    config.mem0.qdrant_url = qdrant_url
    backend = Mem0Backend(config)
    try:
        yield backend
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_a_collection_create_slower_than_the_http_client_default_still_provisions(config, target):
    with httpx.Client() as client:
        default_read = client.timeout.read
    assert default_read is not None
    with stub_qdrant_server(create_delay_s={target: default_read + 0.5}) as stub:
        async with backend_against(stub.url, config) as backend:
            assert await backend.get(MEMORY_ID, SCOPE) is None
        assert stub.creates(target) == 1


@pytest.mark.asyncio
async def test_a_create_overrunning_the_write_budget_raises_a_timeout_naming_it_and_the_next_call_adopts_the_server_completed_collection(
    config, target,
):
    config.queue.backend_write_timeout_seconds = 0.5
    with stub_qdrant_server(create_delay_s={target: 3.0}) as stub:
        async with backend_against(stub.url, config) as backend:
            with pytest.raises(TimeoutError, match=r'Mem0 create_collection timed out after 0\.5s'):
                await backend.get(MEMORY_ID, SCOPE)
            stub.release_event.set()
            assert stub.wait_created(target, timeout=10)
            assert await backend.get(MEMORY_ID, SCOPE) is None
        assert stub.creates(target) == 1
