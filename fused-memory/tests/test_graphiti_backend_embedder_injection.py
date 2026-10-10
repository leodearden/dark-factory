"""build_embedder(cfg) and GraphitiBackend.initialize(embedder=...): a per-arm caller injects the embedder.

No FalkorDB I/O. The driver class is stubbed with the idiom
tests/test_graphiti_backend_llm_injection.py uses, for the same reason (its
constructor sends INFO). Search is observed through a fake embedder whose
``create`` records its input and then stops the search, so no driver query runs.
The per-group client is read through ``_client_for``, the one accessor that
exposes it (as tests/test_per_group_client_cache.py does); nothing private is
patched by dotted path.
"""

from unittest.mock import MagicMock

import pytest
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder

import fused_memory.backends.graphiti_client as graphiti_client_module
from fused_memory.backends.graphiti_client import GraphitiBackend, build_embedder
from fused_memory.config.schema import OpenAIProviderConfig

_UNROUTABLE_FALKOR = 'redis://127.0.0.1:1'
_ARM_URL = 'http://127.0.0.1:8414/v1'


@pytest.fixture
def offline_config(mock_config, monkeypatch):
    real_driver_cls = graphiti_client_module._MultiTenantFalkorDriver
    driver = MagicMock(spec=real_driver_cls)
    driver.clone.return_value = MagicMock(spec=real_driver_cls)
    driver_cls = MagicMock(return_value=driver)
    monkeypatch.setattr(graphiti_client_module, '_MultiTenantFalkorDriver', driver_cls)
    config = mock_config.model_copy(deep=True)
    config.graphiti.falkordb.uri = _UNROUTABLE_FALKOR
    return config


class _EmbedderReached(Exception):
    """Raised by the fake embedder so a search stops before any driver query."""


class _RecordingEmbedder(EmbedderClient):
    def __init__(self) -> None:
        self.inputs: list[object] = []

    async def create(self, input_data):
        self.inputs.append(input_data)
        raise _EmbedderReached

    async def create_batch(self, input_data_list):
        raise AssertionError('search embeds one query, never a batch')


def test_build_embedder_carries_the_configured_model_dimension_and_endpoint(mock_config):
    config = mock_config.model_copy(deep=True)
    config.embedder.model = 'qwen3-embedding-0.6b'
    config.embedder.dimensions = 1024
    config.embedder.providers.openai = OpenAIProviderConfig(api_key='arm-key', api_url=_ARM_URL)

    embedder = build_embedder(config)

    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.config.embedding_model == 'qwen3-embedding-0.6b'
    assert embedder.config.embedding_dim == 1024
    assert embedder.config.base_url == _ARM_URL
    assert embedder.config.api_key == 'arm-key'


def test_build_embedder_is_none_without_an_openai_provider(mock_config):
    config = mock_config.model_copy(deep=True)
    config.embedder.providers.openai = None

    assert build_embedder(config) is None


@pytest.mark.parametrize('api_key', [None, ''])
def test_build_embedder_is_none_without_an_api_key(mock_config, api_key):
    config = mock_config.model_copy(deep=True)
    config.embedder.providers.openai = OpenAIProviderConfig(api_key=api_key)

    assert build_embedder(config) is None


@pytest.mark.asyncio
async def test_the_injected_embedder_is_shared_by_every_client_and_embeds_the_search(
    offline_config,
):
    injected = _RecordingEmbedder()
    backend = GraphitiBackend(offline_config)

    await backend.initialize(skip_maintenance=True, embedder=injected)
    try:
        assert backend.client is not None
        assert backend.client.embedder is injected
        assert backend._client_for('evalmem_probe').embedder is injected
        with pytest.raises(_EmbedderReached):
            await backend.search('which arm\nembeds this?', group_ids=['evalmem_probe'])
    finally:
        await backend.close()

    assert injected.inputs == [['which arm embeds this?']]


@pytest.mark.asyncio
async def test_without_injection_the_backend_builds_what_build_embedder_builds(offline_config):
    backend = GraphitiBackend(offline_config)

    await backend.initialize(skip_maintenance=True)
    try:
        assert backend.client is not None
        built = backend.client.embedder
    finally:
        await backend.close()

    expected = build_embedder(offline_config)
    assert isinstance(expected, OpenAIEmbedder)
    assert isinstance(built, OpenAIEmbedder)
    assert built.config == expected.config
