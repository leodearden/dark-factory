"""A stalled /v1/embeddings response costs one bounded attempt, never the write (task 10241).

Every embedder client gets a per-attempt deadline and a bounded retry, sized
to finish inside the backend write budget.  Driven through the backends' public
seams against a real loopback socket, so the SDK's own timeout and retry run.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import openai
import pytest
from _mem0_stub_backend import backend_against
from _mock_openai_server import MockOpenAIServer, mock_openai_server
from _stub_qdrant_server import stub_qdrant_server
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver

from fused_memory.backends import graphiti_client
from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.request_attempts import RequestAttemptsConfig
from fused_memory.config.schema import FusedMemoryConfig, OpenAIProviderConfig
from fused_memory.models.scope import Scope
from fused_memory.services.mem0_write_retry import (
    TransientProviderFailure,
    classify_transient_provider_failure,
)

pytestmark = [
    pytest.mark.timeout(60),
    pytest.mark.asyncio,
    pytest.mark.timing(
        reason='the per-attempt embedder deadline and the backend write budget are the behaviour under test',
    ),
]

#: A 1.5 s stalled-schedule ceiling: 0.5 s, the SDK's 0.5 s backoff, 0.5 s.
ATTEMPTS = RequestAttemptsConfig(per_attempt_timeout_seconds=0.5, max_attempts=2)
WRITE_BUDGET_S = 5.0
#: Just above the ceiling, so only a stall the per-attempt deadline cannot see reaches it.
TRICKLE_BUDGET_S = 2.0
SCOPE = Scope(project_id='bounded')

ConfigFactory = Callable[[float], FusedMemoryConfig]


@pytest.fixture
def endpoint() -> Iterator[MockOpenAIServer]:
    with mock_openai_server() as server:
        yield server


@pytest.fixture
def bounded_config(mock_config, endpoint, monkeypatch) -> ConfigFactory:
    """A config embedding through *endpoint* under ATTEMPTS, at a given write budget.

    Re-validated whole, as a reload is, so every schedule here passes the
    rule that it fits the write budget.
    """
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)
    monkeypatch.setattr('mem0.memory.main.MEM0_TELEMETRY', False)

    def build(write_budget_s: float) -> FusedMemoryConfig:
        config = mock_config.model_copy(deep=True)
        config.embedder.providers.openai = OpenAIProviderConfig(
            api_key='test-key', api_url=endpoint.base_url,
        )
        assert config.llm.providers.openai is not None
        config.llm.providers.openai.api_url = endpoint.base_url
        config.embedder.request_attempts = ATTEMPTS
        config.queue.backend_write_timeout_seconds = write_budget_s
        return FusedMemoryConfig.model_validate(config.model_dump())

    return build


@pytest.fixture
def offline_graphiti(monkeypatch) -> object:
    """A real Graphiti over a spec'd driver double whose add_episode embeds its body, as upstream does.

    Returns the value that add_episode answers with.
    """
    added = object()

    def driver_factory(**_kwargs):
        driver = MagicMock(spec=FalkorDriver)
        driver.clone.return_value = MagicMock(spec=FalkorDriver)
        return driver

    async def add_episode_embedding_its_body(self, *, episode_body, **_kwargs):
        await self.embedder.create_batch([episode_body])
        return added

    monkeypatch.setattr(graphiti_client, '_MultiTenantFalkorDriver', driver_factory)
    monkeypatch.setattr(graphiti_client, 'check_openai_responses_api', lambda: None)
    monkeypatch.setattr(Graphiti, 'add_episode', add_episode_embedding_its_body)
    return added


@asynccontextmanager
async def mem0_backend(config: FusedMemoryConfig) -> AsyncIterator[Mem0Backend]:
    with stub_qdrant_server() as stub:
        async with backend_against(stub.url, config) as backend:
            yield backend


@asynccontextmanager
async def graphiti_backend(config: FusedMemoryConfig) -> AsyncIterator[GraphitiBackend]:
    backend = GraphitiBackend(config)
    await backend.initialize(skip_maintenance=True)
    try:
        yield backend
    finally:
        await backend.close()


async def warmed_mem0_backend_embeds(backend: Mem0Backend, endpoint: MockOpenAIServer) -> int:
    """Build and cache the backend's instance outside the timed window; return the embeds so far."""
    await backend.add('warm-up', SCOPE)
    return len(endpoint.requests_to('/embeddings'))


async def test_a_mem0_write_survives_one_stalled_embedding(bounded_config, endpoint):
    config = bounded_config(WRITE_BUDGET_S)
    endpoint.stall('/embeddings', first=1)

    async with mem0_backend(config) as backend:
        started = time.monotonic()
        result = await backend.add('x', SCOPE)
        elapsed = time.monotonic() - started

    assert [(r['memory'], r['event']) for r in result['results']] == [('x', 'ADD')]
    assert len(endpoint.requests_to('/embeddings')) == 2
    assert elapsed < WRITE_BUDGET_S


async def test_a_graphiti_write_survives_one_stalled_embedding(
    bounded_config, endpoint, offline_graphiti,
):
    config = bounded_config(WRITE_BUDGET_S)
    endpoint.stall('/embeddings', first=1)

    async with graphiti_backend(config) as backend:
        started = time.monotonic()
        result = await backend.add_episode(name='e', content='x', group_id='bounded')
        elapsed = time.monotonic() - started

    assert result is offline_graphiti
    assert len(endpoint.requests_to('/embeddings')) == 2
    assert elapsed < WRITE_BUDGET_S


async def test_a_mem0_write_whose_every_embedding_stalls_fails_on_its_attempts_before_the_budget(
    bounded_config, endpoint,
):
    config = bounded_config(WRITE_BUDGET_S)

    async with mem0_backend(config) as backend:
        embeds_before = await warmed_mem0_backend_embeds(backend, endpoint)
        endpoint.stall('/embeddings')
        started = time.monotonic()
        with pytest.raises(openai.APITimeoutError) as raised:
            await backend.add('x', SCOPE)
        elapsed = time.monotonic() - started

    assert len(endpoint.requests_to('/embeddings')) - embeds_before == ATTEMPTS.max_attempts
    assert elapsed < WRITE_BUDGET_S
    assert classify_transient_provider_failure(raised.value) is TransientProviderFailure.CONNECTION


async def test_a_graphiti_write_whose_every_embedding_stalls_fails_on_its_attempts_before_the_budget(
    bounded_config, endpoint, offline_graphiti,
):
    config = bounded_config(WRITE_BUDGET_S)
    endpoint.stall('/embeddings')

    async with graphiti_backend(config) as backend:
        started = time.monotonic()
        with pytest.raises(openai.APITimeoutError):
            await backend.add_episode(name='e', content='x', group_id='bounded')
        elapsed = time.monotonic() - started

    assert len(endpoint.requests_to('/embeddings')) == ATTEMPTS.max_attempts
    assert elapsed < WRITE_BUDGET_S


async def test_a_trickling_embedding_still_ends_at_the_write_budget(bounded_config, endpoint):
    """Each read arrives within the per-attempt deadline, so only the outer budget can end it."""
    config = bounded_config(TRICKLE_BUDGET_S)

    async with mem0_backend(config) as backend:
        await warmed_mem0_backend_embeds(backend, endpoint)
        endpoint.trickle('/embeddings', interval_s=0.1)
        started = time.monotonic()
        with pytest.raises(TimeoutError) as raised:
            await backend.add('x', SCOPE)
        elapsed = time.monotonic() - started

    assert str(raised.value) == f'Mem0 add timed out after {TRICKLE_BUDGET_S}s'
    assert TRICKLE_BUDGET_S <= elapsed < TRICKLE_BUDGET_S + 1.0
