"""mem0's vector-store Qdrant client runs under Mem0Backend's own budgets.

Measured in the qdrant-integration lane pass 20261116T021649-ead5ad (task
10348): three tests logged ``Mem0 write failed: timed out`` for mem0's point
upsert (``PUT /collections/<c>/points?wait=true``), which Qdrant answered 200 a
little over 5s later.  mem0 had issued it through a QdrantClient it built with
no timeout, so httpx's 5s library default fired beneath the configured write
budget, and the failure named no budget at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Iterator

import pytest
from _fm_polling import poll_until
from _mem0_stub_backend import (
    MEMORY_ID,
    backend_against,
    httpx_default_read_timeout,
    stub_ready_config,
)
from _stub_qdrant_server import POOLING_LOOPBACK_HOST, stub_qdrant_server

from fused_memory.backends import mem0_client
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.scope import Scope

SCOPE = Scope(project_id='vector_store_client_probe')
CONTENT = 'A slow Qdrant leg fails on its configured budget, never on a hidden one.'


@pytest.fixture
def config(mock_config, monkeypatch) -> Iterator[FusedMemoryConfig]:
    with stub_ready_config(mock_config, monkeypatch) as config:
        yield config


@pytest.fixture
def target(config) -> str:
    return SCOPE.mem0_collection_name(config.mem0.collection_prefix)


@pytest.fixture
def default_read() -> float:
    return httpx_default_read_timeout()


@pytest.mark.asyncio
async def test_an_add_slower_than_the_http_client_default_still_lands(config, target, default_read):
    with stub_qdrant_server(upsert_delay_s={target: default_read + 0.5}) as stub:
        async with backend_against(stub.url, config) as backend:
            result = await backend.add(CONTENT, SCOPE)
        [added] = result['results']
        assert added['id']
        assert stub.upserts(target) == 1


@pytest.mark.asyncio
async def test_a_get_slower_than_the_http_client_default_still_answers(config, target, default_read):
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


@pytest.mark.asyncio
async def test_close_releases_every_connection_the_backend_opened(config):
    with stub_qdrant_server(host=POOLING_LOOPBACK_HOST) as stub:
        async with backend_against(stub.url, config) as backend:
            assert await backend.get(MEMORY_ID, SCOPE) is None
            await backend.close()
            await poll_until(
                lambda: not stub.open_connections(),
                message='the stub still holds a connection the closed backend opened',
            )


@pytest.mark.asyncio
async def test_a_backend_used_again_after_close_still_serves(config):
    with stub_qdrant_server() as stub:
        async with backend_against(stub.url, config) as backend:
            assert await backend.get(MEMORY_ID, SCOPE) is None
            await backend.close()
            assert await backend.get(MEMORY_ID, SCOPE) is None


@pytest.mark.asyncio
async def test_a_close_while_mem0_builds_its_indexes_leaves_the_backend_serving(config, target):
    with stub_qdrant_server(index_delay_s={target: 30.0}) as stub:
        async with backend_against(stub.url, config) as backend:
            first_touch = asyncio.create_task(backend.get(MEMORY_ID, SCOPE))
            try:
                await poll_until(lambda: stub.index_creates_in_flight(target) == 1)
                await backend.close()
            finally:
                stub.release_event.set()
            assert await first_touch is None
            assert await backend.get(MEMORY_ID, SCOPE) is None


@pytest.mark.asyncio
async def test_a_first_touch_leaves_the_event_loop_free_while_mem0_lists_collections(config):
    with stub_qdrant_server(list_collections_delay_s=30.0) as stub:
        async with backend_against(stub.url, config) as backend:
            creation = asyncio.create_task(backend.get(MEMORY_ID, SCOPE))
            try:
                await poll_until(
                    lambda: stub.collection_lists_in_flight() == 1,
                    message='the event loop never ran while mem0 listed collections at first touch',
                )
            finally:
                stub.release_event.set()
                result = await creation
            assert result is None


@pytest.mark.asyncio
async def test_a_first_touch_overrunning_its_write_budget_raises_a_timeout_naming_it(config, target):
    budget = 1.0
    config.queue.backend_write_timeout_seconds = budget
    overrun = re.escape(f'Mem0 create_instance timed out after {budget}s')
    with stub_qdrant_server(list_collections_delay_s=60.0, preexisting={target}) as stub:
        async with backend_against(stub.url, config) as backend:
            try:
                with pytest.raises(TimeoutError, match=overrun):
                    await backend.get(MEMORY_ID, SCOPE)
                with pytest.raises(TimeoutError, match=overrun):
                    await backend.get(MEMORY_ID, SCOPE)
                await poll_until(
                    lambda: stub.collection_lists_in_flight() == 2,
                    message='the call after a timed-out first touch never began a fresh mem0 build',
                )
            finally:
                stub.release_event.set()
            await poll_until(lambda: not stub.collection_lists_in_flight())


@pytest.mark.parametrize(
    ('read_budget', 'write_budget'),
    [pytest.param(1.0, 30.0, id='read-narrower'), pytest.param(30.0, 1.0, id='write-narrower')],
)
def test_the_vector_store_client_outlasts_whichever_budget_is_wider(
    config, target, default_read, read_budget, write_budget,
):
    config.queue.backend_read_timeout_seconds = read_budget
    config.queue.backend_write_timeout_seconds = write_budget
    with stub_qdrant_server(retrieve_delay_s={target: default_read + 0.5}) as stub:
        config.mem0.qdrant_url = stub.url
        with contextlib.closing(mem0_client.build_vector_store_client(config)) as client:
            assert client.retrieve(collection_name=target, ids=[MEMORY_ID]) == []
