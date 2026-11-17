"""The routing layer's one AsyncOpenAI: injected or built lazily, released by close()."""

import pytest
from _sdk_client_stubs import completion_client
from openai import AsyncOpenAI

from fused_memory.config.schema import RoutingConfig
from fused_memory.routing.openai_client import LazyOpenAIClient


class TestRoutingEndpointLeaves:
    def test_both_leaves_default_to_inherit(self):
        routing = RoutingConfig()

        assert routing.openai_api_url is None
        assert routing.openai_api_key is None

    @pytest.mark.parametrize('blank', ['', '   '])
    def test_a_blank_leaf_is_stored_as_inherit(self, blank):
        routing = RoutingConfig(openai_api_url=blank, openai_api_key=blank)

        assert routing.openai_api_url is None
        assert routing.openai_api_key is None

    def test_a_set_leaf_is_kept_verbatim(self):
        routing = RoutingConfig(
            openai_api_url='http://127.0.0.1:8317/v1', openai_api_key='sk-routing',
        )

        assert routing.openai_api_url == 'http://127.0.0.1:8317/v1'
        assert routing.openai_api_key == 'sk-routing'


class TestLazyOpenAIClient:
    def test_an_injected_client_is_returned_as_is(self, mock_config):
        client = completion_client('{}')
        holder = LazyOpenAIClient(mock_config, client)

        assert holder.get() is client
        assert holder.get() is client

    @pytest.mark.asyncio
    async def test_without_an_injected_client_one_is_built_once_and_cached(self, mock_config):
        holder = LazyOpenAIClient(mock_config)

        built = holder.get()

        assert holder.get() is built
        assert isinstance(built, AsyncOpenAI)
        assert built.api_key == 'test-key'

        await holder.close()

        assert built.is_closed()

    @pytest.mark.asyncio
    async def test_the_endpoint_comes_from_openai_base_url_not_the_configured_api_url(
        self, mock_config, monkeypatch,
    ):
        mock_config.llm.providers.openai.api_url = 'https://configured.example/v1'
        monkeypatch.setenv('OPENAI_BASE_URL', 'http://127.0.0.1:9/v1')
        holder = LazyOpenAIClient(mock_config)

        try:
            built = holder.get()
            assert str(built.base_url).rstrip('/') == 'http://127.0.0.1:9/v1'
        finally:
            await holder.close()

    @pytest.mark.asyncio
    async def test_an_injected_client_is_released_exactly_once(self, mock_config):
        client = completion_client('{}')
        holder = LazyOpenAIClient(mock_config, client)

        await holder.close()
        await holder.close()

        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closing_a_never_used_holder_is_safe_and_a_later_get_builds_afresh(
        self, mock_config,
    ):
        holder = LazyOpenAIClient(mock_config)

        await holder.close()
        await holder.close()

        built = holder.get()
        assert not built.is_closed()
        await holder.close()
        assert built.is_closed()
