"""Live proof that a configured local base_url actually RECEIVES the traffic.

Every other assertion in this task is on constructor kwargs, and kwarg
assertions structurally cannot detect the failure class this task exists to
prevent: a base_url accepted and then ignored, with traffic silently falling
back to api.openai.com (graphiti-#912). ``OpenAIEmbedderConfig`` is a pydantic
model with the default ``extra='ignore'`` — a misspelled kwarg is dropped
without a word — and both mem0 and the openai SDK fall back to
``OPENAI_BASE_URL`` / api.openai.com when unset. Only a real socket disproves
it.

Every test here points ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` /
``OPENAI_API_URL`` at an unroutable sentinel, so any code path that falls back
to an env var (or to api.openai.com) fails LOUDLY with a connection error
instead of silently succeeding against the wrong host.

Deliberately NOT marked ``integration``: that marker is deselected by the
default ``-m 'not integration'`` addopts, and this signal must run in CI. These
tests need no external service — just loopback sockets on ephemeral ports, so
they are xdist-safe.

HAZARD compliance: no FalkorDriver / _MultiTenantFalkorDriver / Graphiti
construction, no real graph. The LLM path is exercised through the extracted
``build_llm_client`` seam, never ``initialize()``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _mock_openai_server import mock_openai_server
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.config.schema import (
    EmbedderConfig,
    EmbedderProvidersConfig,
    FusedMemoryConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
)
from fused_memory.maintenance.reindex import run_reindex

# Unroutable: port 1 on loopback. Any fallback to an ambient env var lands here
# and fails loudly with a connection error rather than reaching a real host.
UNROUTABLE_SENTINEL = 'http://127.0.0.1:1/v1'

pytestmark = pytest.mark.timeout(60)


class _Inner(BaseModel):
    name: str


class _Extraction(BaseModel):
    """Nested submodel → model_json_schema() emits $defs/$ref."""

    items: list[_Inner]


@pytest.fixture
def no_ambient_openai_env(monkeypatch):
    """Point every ambient OpenAI endpoint var at the unroutable sentinel."""
    for var in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_URL'):
        monkeypatch.setenv(var, UNROUTABLE_SENTINEL)
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key-must-not-be-used')


def _config(llm_url: str, embedder_url: str) -> FusedMemoryConfig:
    return FusedMemoryConfig(
        llm=LLMConfig(
            provider='openai',
            client_class='openai_generic',
            structured_output_mode='json_object',
            model='mock-model',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key', api_url=llm_url),
            ),
        ),
        embedder=EmbedderConfig(
            provider='openai',
            model='mock-embedding-model',
            providers=EmbedderProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key', api_url=embedder_url),
            ),
        ),
    )


class TestGraphitiLLMPathReachesConfiguredEndpoint:
    """Test A — the graphiti LLM client."""

    @pytest.mark.asyncio
    async def test_llm_request_lands_on_the_configured_local_endpoint(
        self, no_ambient_openai_env,
    ):
        with mock_openai_server() as server:
            cfg = _config(server.base_url, server.base_url)
            client = build_llm_client(cfg)
            assert client is not None

            await client.generate_response(
                [Message(role='user', content='Alice knows Bob.')],
                response_model=_Extraction,
            )

            chat_requests = server.requests_to('/chat/completions')
            assert len(chat_requests) == 1, (
                f'expected exactly one chat request on the configured endpoint, '
                f'got {[r["path"] for r in server.requests]}'
            )
            assert chat_requests[0]['method'] == 'POST'
            assert chat_requests[0]['headers'].get('Authorization') == 'Bearer test-key'

    @pytest.mark.asyncio
    async def test_forced_json_object_survives_a_real_round_trip(
        self, no_ambient_openai_env,
    ):
        """End-to-end proof of the forcing wrapper over real HTTP — not just a
        mocked create() kwarg. A response_model IS passed, yet the body on the
        wire asks for json_object."""
        with mock_openai_server() as server:
            cfg = _config(server.base_url, server.base_url)
            client = build_llm_client(cfg)
            assert client is not None

            await client.generate_response(
                [Message(role='user', content='Alice knows Bob.')],
                response_model=_Extraction,
            )

            body = server.requests_to('/chat/completions')[0]['json_body']
            assert body['response_format'] == {'type': 'json_object'}

    @pytest.mark.asyncio
    async def test_configured_max_tokens_reaches_the_wire(
        self, no_ambient_openai_env,
    ):
        """The configured ``llm.max_tokens`` must actually leave the process.

        ``OpenAIGenericClient.__init__`` re-assigns ``self.max_tokens`` from its
        own parameter (default 16384) right after ``super().__init__`` took the
        configured value off the config object, and ``_generate_response`` sends
        ``self.max_tokens``. A construction-kwarg assertion alone cannot catch
        that class of drop — same reason this whole file exists for base_url —
        so this pins the recorded request body.
        """
        with mock_openai_server() as server:
            cfg = _config(server.base_url, server.base_url)
            cfg.llm.max_tokens = 2048  # distinct from schema default 4096 and upstream 16384
            client = build_llm_client(cfg)
            assert client is not None

            await client.generate_response(
                [Message(role='user', content='Alice knows Bob.')],
                response_model=_Extraction,
            )

            body = server.requests_to('/chat/completions')[0]['json_body']
            assert body['max_tokens'] == 2048

    @pytest.mark.asyncio
    async def test_auto_mode_sends_json_schema_over_the_wire(
        self, no_ambient_openai_env,
    ):
        """Control: with structured_output_mode='auto' the same live path sends
        json_schema, so the previous test is measuring the wrapper and not some
        incidental property of the client."""
        with mock_openai_server() as server:
            cfg = _config(server.base_url, server.base_url)
            cfg.llm.structured_output_mode = 'auto'
            client = build_llm_client(cfg)
            assert client is not None

            await client.generate_response(
                [Message(role='user', content='Alice knows Bob.')],
                response_model=_Extraction,
            )

            body = server.requests_to('/chat/completions')[0]['json_body']
            assert body['response_format']['type'] == 'json_schema'


class TestReindexEmbedderPathReachesConfiguredEndpoint:
    """Test B — the reindex embedder, with OpenAIEmbedderConfig/OpenAIEmbedder
    left REAL so the extra='ignore' silent-drop is actually exercised."""

    @pytest.mark.asyncio
    async def test_embedder_request_lands_on_the_configured_local_endpoint(
        self, no_ambient_openai_env, make_fake_maintenance_service,
    ):
        from fused_memory.maintenance.reindex import ReindexResult

        with mock_openai_server() as server:
            cfg = _config(server.base_url, server.base_url)

            mock_service = AsyncMock()
            mock_service.durable_queue = MagicMock()
            mock_service.graphiti = MagicMock()
            mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0}

            with (
                patch(
                    'fused_memory.maintenance.reindex.maintenance_service',
                    side_effect=make_fake_maintenance_service(cfg, mock_service),
                ),
                patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
            ):
                mock_mgr = MagicMock()
                mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
                mock_mgr_cls.return_value = mock_mgr

                await run_reindex()

            # Capture the REAL embedder run_reindex built and constructed.
            embedder = mock_mgr_cls.call_args.kwargs['embedder']
            assert server.requests == [], 'no traffic expected before create()'

            await embedder.create(['some stale content'])

            embedding_requests = server.requests_to('/embeddings')
            assert len(embedding_requests) == 1, (
                f'expected exactly one embeddings request on the configured '
                f'endpoint, got {[r["path"] for r in server.requests]}'
            )
            assert embedding_requests[0]['method'] == 'POST'


class TestIndependentLLMAndEmbedderEndpoints:
    """Test C — the task's explicit requirement: per-block YAML values give
    genuinely independent endpoints, not one reused for both."""

    @pytest.mark.asyncio
    async def test_each_block_receives_only_its_own_traffic(
        self, no_ambient_openai_env, make_fake_maintenance_service,
    ):
        from fused_memory.maintenance.reindex import ReindexResult

        with mock_openai_server() as llm_server, mock_openai_server() as embedder_server:
            assert llm_server.base_url != embedder_server.base_url
            cfg = _config(llm_server.base_url, embedder_server.base_url)

            client = build_llm_client(cfg)
            assert client is not None
            await client.generate_response(
                [Message(role='user', content='Alice knows Bob.')],
                response_model=_Extraction,
            )

            mock_service = AsyncMock()
            mock_service.durable_queue = MagicMock()
            mock_service.graphiti = MagicMock()
            mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0}

            with (
                patch(
                    'fused_memory.maintenance.reindex.maintenance_service',
                    side_effect=make_fake_maintenance_service(cfg, mock_service),
                ),
                patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
            ):
                mock_mgr = MagicMock()
                mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
                mock_mgr_cls.return_value = mock_mgr
                await run_reindex()

            await mock_mgr_cls.call_args.kwargs['embedder'].create(['stale'])

            # Exact counts, not "contains" — a base_url that is plumbed but
            # ignored would show up as a server that saw nothing.
            assert len(llm_server.requests) == 1
            assert llm_server.requests[0]['path'].endswith('/chat/completions')
            assert len(embedder_server.requests) == 1
            assert embedder_server.requests[0]['path'].endswith('/embeddings')
