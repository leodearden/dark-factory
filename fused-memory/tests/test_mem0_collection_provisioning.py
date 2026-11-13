"""Mem0Backend provisions a project's Qdrant collection under its own budget.

Measured in the qdrant-integration lane passes 20261030T180141-9fb982 and
20261102T040927-1770f9 (task 8190): Qdrant runs collection creates one after
another, and under 8 workers it answered 200 to the failing creates at 5.31s
and 5.86s.  mem0 had issued them through a QdrantClient it built with no
timeout, so httpx's 5s library default gave up first and aborted a create the
server went on to complete.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from _stub_qdrant_server import stub_qdrant_server
from qdrant_client.http.exceptions import UnexpectedResponse

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.scope import Scope

SCOPE = Scope(project_id='provisioning_probe')
MEMORY_ID = '00000000-0000-4000-8000-000000000001'


@pytest.fixture
def config(mock_config, monkeypatch) -> FusedMemoryConfig:
    """mem0 telemetry off: no posthog call and no mem0migrations collection on the stub."""
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)
    monkeypatch.setattr('mem0.memory.main.MEM0_TELEMETRY', False)
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
async def test_a_new_collection_is_created_at_the_configured_width_under_cosine_distance(config, target):
    config.embedder.dimensions = 768
    with stub_qdrant_server() as stub:
        async with backend_against(stub.url, config) as backend:
            assert await backend.get(MEMORY_ID, SCOPE) is None
        [create_body] = stub.create_bodies(target)
    assert create_body['vectors'] == {'size': 768, 'distance': 'Cosine'}


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


@pytest.mark.asyncio
async def test_a_collection_another_creator_made_first_is_adopted_not_failed(config, target):
    with stub_qdrant_server(conflict_on_create={target}) as stub:
        async with backend_against(stub.url, config) as backend:
            assert await backend.get(MEMORY_ID, SCOPE) is None
            assert await backend.get(MEMORY_ID, SCOPE) is None
        assert stub.creates(target) == 1
        assert stub.exists_checks(target) == 1


@pytest.mark.asyncio
async def test_a_create_the_server_rejects_propagates_and_leaves_nothing_cached(config, target):
    with stub_qdrant_server(create_status={target: 500}) as stub:
        async with backend_against(stub.url, config) as backend:
            for attempt in (1, 2):
                with pytest.raises(UnexpectedResponse) as excinfo:
                    await backend.get(MEMORY_ID, SCOPE)
                assert excinfo.value.status_code == 500
                assert stub.creates(target) == attempt


@pytest.mark.asyncio
async def test_an_existing_collection_is_adopted_and_never_recreated(config, target):
    with stub_qdrant_server(preexisting={target}) as stub:
        async with backend_against(stub.url, config) as backend:
            assert await backend.get(MEMORY_ID, SCOPE) is None
        assert stub.creates(target) == 0


@pytest.mark.asyncio
async def test_concurrent_first_calls_for_one_project_provision_once(config, target):
    with stub_qdrant_server(create_delay_s={target: 0.3}) as stub:
        async with backend_against(stub.url, config) as backend:
            results = await asyncio.gather(backend.get(MEMORY_ID, SCOPE), backend.get(MEMORY_ID, SCOPE))
            assert results == [None, None]
        assert stub.creates(target) == 1


@pytest.mark.asyncio
async def test_concurrent_first_calls_share_one_failed_create_and_the_next_call_retries(config, target):
    with stub_qdrant_server(create_status={target: 500}) as stub:
        async with backend_against(stub.url, config) as backend:
            results = await asyncio.gather(
                backend.get(MEMORY_ID, SCOPE), backend.get(MEMORY_ID, SCOPE), return_exceptions=True,
            )
            assert [type(result) for result in results] == [UnexpectedResponse, UnexpectedResponse]
            assert stub.creates(target) == 1
            with pytest.raises(UnexpectedResponse):
                await backend.get(MEMORY_ID, SCOPE)
            assert stub.creates(target) == 2


@pytest.mark.asyncio
async def test_a_cancelled_first_caller_leaves_the_shared_creation_running_for_the_others(config, target):
    with stub_qdrant_server(create_delay_s={target: 10.0}) as stub:
        async with backend_against(stub.url, config) as backend:
            first = asyncio.create_task(backend.get(MEMORY_ID, SCOPE))
            second = asyncio.create_task(backend.get(MEMORY_ID, SCOPE))
            async with asyncio.timeout(10):
                while stub.creates(target) == 0:
                    await asyncio.sleep(0.01)
            first.cancel()
            stub.release_event.set()
            assert await second is None
            with pytest.raises(asyncio.CancelledError):
                await first
        assert stub.creates(target) == 1
        assert stub.exists_checks(target) == 1
