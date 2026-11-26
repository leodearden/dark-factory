"""GraphitiBackend.initialize(llm_client=...): a per-arm caller injects the shared LLM client.

No FalkorDB I/O. ``falkordb.asyncio.FalkorDB.__init__`` sends INFO at construction
(its cluster check), so the driver class is stubbed with the idiom
tests/test_per_group_client_cache.py uses; the stub instance is spec'd on the real
driver class because ``Graphiti`` (kept real) checks it is a ``GraphDriver``.
GraphitiBackend has no driver seam, so this one private-name patch is tolerated;
everything asserted is read through the backend's public ``client``.
"""

from unittest.mock import MagicMock

import pytest
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig

import fused_memory.backends.graphiti_client as graphiti_client_module
from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.backends.llm_clients import TokenRecordingOpenAIGenericClient
from fused_memory.backends.llm_token_usage import AttributingTokenUsageTracker

_UNROUTABLE_FALKOR = 'redis://127.0.0.1:1'


@pytest.fixture
def offline_config(mock_config, monkeypatch):
    real_driver_cls = graphiti_client_module._MultiTenantFalkorDriver
    driver_cls = MagicMock(return_value=MagicMock(spec=real_driver_cls))
    monkeypatch.setattr(graphiti_client_module, '_MultiTenantFalkorDriver', driver_cls)
    config = mock_config.model_copy(deep=True)
    config.graphiti.falkordb.uri = _UNROUTABLE_FALKOR
    return config


def _injected_client() -> TokenRecordingOpenAIGenericClient:
    client = TokenRecordingOpenAIGenericClient(
        config=GraphitiLLMConfig(api_key='k', model='m', base_url='http://127.0.0.1:9/v1'),
    )
    client.token_tracker = AttributingTokenUsageTracker()
    return client


@pytest.mark.asyncio
async def test_injected_client_is_the_one_the_backend_uses_and_probes(offline_config):
    injected = _injected_client()
    backend = GraphitiBackend(offline_config)

    await backend.initialize(skip_maintenance=True, llm_client=injected)
    try:
        assert backend.client is not None
        assert backend.client.llm_client is injected
        async with backend.token_probe() as m:
            injected.token_tracker.record('extract_nodes', 7, 3)
    finally:
        await backend.close()

    assert m.usage is not None
    assert m.usage.total_tokens == 10


@pytest.mark.asyncio
async def test_without_injection_the_backend_builds_its_own_client(offline_config):
    backend = GraphitiBackend(offline_config)

    await backend.initialize(skip_maintenance=True)
    try:
        assert backend.client is not None
        built = backend.client.llm_client
    finally:
        await backend.close()

    assert built is not None
    assert built.model == offline_config.llm.model
