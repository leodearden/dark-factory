"""The routing layer's one AsyncOpenAI: injected or built lazily, released by close()."""

import contextlib
import logging

import pytest
from _mock_openai_server import mock_openai_server
from _sdk_client_stubs import completion_client
from openai import AsyncOpenAI

from fused_memory.config.env_precedence import reset_warning_cache
from fused_memory.config.schema import RoutingConfig
from fused_memory.routing.openai_client import LazyOpenAIClient

AMBIENT_URL = 'http://127.0.0.1:9/v1'
UNROUTABLE_SENTINEL = 'http://127.0.0.1:1/v1'
CONFIGURED_URL = 'https://configured.example/v1'
ROUTING_URL = 'http://127.0.0.1:8317/v1'
ROUTING_KEY = 'sk-routing'


@pytest.fixture(autouse=True)
def _no_ambient_base_url(monkeypatch):
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('OPENAI_API_BASE', raising=False)
    reset_warning_cache()
    yield
    reset_warning_cache()


def _endpoint(client: AsyncOpenAI) -> str:
    return str(client.base_url).rstrip('/')


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


@contextlib.asynccontextmanager
async def _built_from(config):
    holder = LazyOpenAIClient(config)
    try:
        yield holder.get()
    finally:
        await holder.close()


class TestTheRoutingEndpoint:
    @pytest.mark.asyncio
    async def test_the_endpoint_is_the_configured_llm_api_url_not_openai_base_url(
        self, mock_config, monkeypatch,
    ):
        mock_config.llm.providers.openai.api_url = CONFIGURED_URL
        monkeypatch.setenv('OPENAI_BASE_URL', AMBIENT_URL)

        async with _built_from(mock_config) as built:
            assert _endpoint(built) == CONFIGURED_URL
            assert built.api_key == 'test-key'

    @pytest.mark.asyncio
    async def test_overriding_an_ambient_base_url_is_reported_for_the_routing_llm(
        self, mock_config, monkeypatch, caplog,
    ):
        mock_config.llm.providers.openai.api_url = CONFIGURED_URL
        monkeypatch.setenv('OPENAI_BASE_URL', AMBIENT_URL)

        with caplog.at_level(logging.WARNING):
            async with _built_from(mock_config):
                pass

        assert any(
            'routing LLM' in record.message
            and 'OPENAI_BASE_URL' in record.message
            and CONFIGURED_URL in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_the_routing_leaves_override_the_llm_provider(self, mock_config):
        mock_config.routing.openai_api_url = ROUTING_URL
        mock_config.routing.openai_api_key = ROUTING_KEY

        async with _built_from(mock_config) as built:
            assert _endpoint(built) == ROUTING_URL
            assert built.api_key == ROUTING_KEY

        provider = mock_config.llm.providers.openai
        assert provider is not None
        assert provider.api_key == 'test-key'
        assert provider.api_url == 'https://api.openai.com/v1'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('leaf', 'value', 'expected_url', 'expected_key'),
        [
            pytest.param('openai_api_url', ROUTING_URL, ROUTING_URL, 'test-key', id='url-only'),
            pytest.param(
                'openai_api_key', ROUTING_KEY, 'https://api.openai.com/v1', ROUTING_KEY,
                id='key-only',
            ),
        ],
    )
    async def test_each_routing_leaf_inherits_independently(
        self, mock_config, leaf, value, expected_url, expected_key,
    ):
        setattr(mock_config.routing, leaf, value)

        async with _built_from(mock_config) as built:
            assert _endpoint(built) == expected_url
            assert built.api_key == expected_key

    @pytest.mark.asyncio
    async def test_a_blank_routing_leaf_assigned_after_construction_still_inherits(
        self, mock_config,
    ):
        mock_config.llm.providers.openai.api_url = CONFIGURED_URL
        mock_config.routing.openai_api_url = ''

        async with _built_from(mock_config) as built:
            assert _endpoint(built) == CONFIGURED_URL

    @pytest.mark.asyncio
    async def test_with_no_openai_provider_and_no_routing_leaves_the_sdk_env_fallback_applies(
        self, mock_config, monkeypatch,
    ):
        mock_config.llm.providers.openai = None
        monkeypatch.setenv('OPENAI_BASE_URL', AMBIENT_URL)
        monkeypatch.setenv('OPENAI_API_KEY', 'env-key')

        async with _built_from(mock_config) as built:
            assert _endpoint(built) == AMBIENT_URL
            assert built.api_key == 'env-key'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('source', ['routing', 'llm'])
    async def test_the_built_client_sends_to_the_endpoint_set_in_config(
        self, mock_config, monkeypatch, source,
    ):
        with mock_openai_server() as server:
            monkeypatch.setenv('OPENAI_BASE_URL', UNROUTABLE_SENTINEL)
            monkeypatch.setenv('OPENAI_API_BASE', UNROUTABLE_SENTINEL)
            if source == 'routing':
                mock_config.routing.openai_api_url = server.base_url
            else:
                mock_config.llm.providers.openai.api_url = server.base_url

            async with _built_from(mock_config) as built:
                await built.chat.completions.create(
                    model='gpt-4o-mini', messages=[{'role': 'user', 'content': 'x'}],
                )

            assert len(server.requests_to('/chat/completions')) == 1
