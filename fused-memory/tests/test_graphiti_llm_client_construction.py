"""Tests for ``build_llm_client`` — the driver-free graphiti LLM construction seam.

Every test here calls ``build_llm_client(cfg)`` DIRECTLY and never
``GraphitiBackend.initialize()``. That is deliberate HAZARD compliance, not
style: ``initialize()`` constructs ``_MultiTenantFalkorDriver`` ~50 lines after
the LLM block, and ``FalkorDriver.__init__`` fire-and-forgets
``build_indices_and_constraints()`` — which would create indices on a real
graph and destroy the protected no-index evidence owned by
``docs/prds/falkordb-index-provisioning.md``. Extracting the builder makes that
structurally unreachable from these tests rather than merely monkeypatched
away. No FalkorDriver, no Graphiti client, no real graph is touched here.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

import fused_memory.backends.graphiti_client as graphiti_client_module
from fused_memory.backends.graphiti_client import build_llm_client


def _record_llm_config(monkeypatch) -> list[dict]:
    """Patch the module-level ``GraphitiLLMConfig`` with a recording wrapper.

    Returns the list that each construction's kwargs are appended to. The real
    class is still constructed and returned, so the client under test receives
    a genuine config object.
    """
    calls: list[dict] = []

    def recording_config(**kwargs):
        calls.append(kwargs)
        return GraphitiLLMConfig(**kwargs)

    monkeypatch.setattr(graphiti_client_module, 'GraphitiLLMConfig', recording_config)
    return calls


class TestBuildLLMClientDefaultOpenAIPath:
    """The shipped default: client_class='openai' → graphiti's OpenAIClient."""

    def test_returns_openai_client(self, mock_config, monkeypatch):
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', lambda: None,
        )
        client = build_llm_client(mock_config)
        assert isinstance(client, OpenAIClient)

    def test_plumbs_configured_base_url(self, mock_config, monkeypatch):
        """The core gap this task closes.

        The embedder path (graphiti_client.py:533-538) and the reranker path
        (:584-589) have always passed ``base_url=cfg....providers.openai.api_url``;
        the LLM config at :502-508 never did. A configured LLM endpoint was
        therefore silently discarded in favour of the openai SDK default.
        """
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', lambda: None,
        )
        mock_config.llm.providers.openai.api_url = 'https://llm.example.invalid/v1'
        calls = _record_llm_config(monkeypatch)

        build_llm_client(mock_config)

        assert len(calls) == 1
        assert calls[0]['base_url'] == 'https://llm.example.invalid/v1'

    def test_preserves_existing_llm_config_kwargs(self, mock_config, monkeypatch):
        """Behaviour preservation across the extraction: every kwarg that
        graphiti_client.py:502-508 passed today is still passed identically."""
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', lambda: None,
        )
        mock_config.llm.model = 'gpt-4o-mini'
        mock_config.llm.max_tokens = 4096
        mock_config.llm.temperature = None  # exercises the `or 0.0` coercion
        calls = _record_llm_config(monkeypatch)

        build_llm_client(mock_config)

        assert calls[0]['api_key'] == 'test-key'
        assert calls[0]['model'] == 'gpt-4o-mini'
        assert calls[0]['small_model'] == 'gpt-4o-mini'
        assert calls[0]['temperature'] == 0.0
        assert calls[0]['max_tokens'] == 4096

    def test_passes_explicit_temperature_through(self, mock_config, monkeypatch):
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', lambda: None,
        )
        mock_config.llm.temperature = 0.7
        calls = _record_llm_config(monkeypatch)

        build_llm_client(mock_config)

        assert calls[0]['temperature'] == 0.7

    def test_runs_responses_api_preflight(self, mock_config, monkeypatch):
        """The preflight (task 2053) must still run on the 'openai' arm — that
        arm builds OpenAIClient, which calls client.responses.create."""
        mock_preflight = MagicMock()
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', mock_preflight,
        )

        build_llm_client(mock_config)

        mock_preflight.assert_called_once()

    def test_preflight_failure_propagates(self, mock_config, monkeypatch):
        """A failing preflight must abort construction, not be swallowed."""
        def sentinel_check():
            raise RuntimeError('PREFLIGHT_SENTINEL')

        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', sentinel_check,
        )

        with pytest.raises(RuntimeError, match='PREFLIGHT_SENTINEL'):
            build_llm_client(mock_config)

    def test_no_max_tokens_kwarg_on_the_default_arm(self, mock_config, monkeypatch):
        """Byte-identical regression guard for the default (shipped) arm.

        The openai_generic arm passes ``max_tokens=cfg.llm.max_tokens`` to the
        constructor because ``OpenAIGenericClient.__init__`` re-assigns
        ``self.max_tokens`` from its own parameter (default 16384), discarding
        the value ``super().__init__`` just took from the config.
        ``BaseOpenAIClient.__init__`` performs the IDENTICAL override, so the
        default arm has always sent upstream's default too. Adding the kwarg
        here would change the wire request of every existing shipped-config
        deployment — which this task must not do. Assert the absence of the
        kwarg rather than pinning upstream's literal default, so a future wheel
        bump that legitimately changes that constant does not break this guard.
        """
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', lambda: None,
        )
        fake_openai_cls = MagicMock(name='OpenAIClient')
        monkeypatch.setattr(graphiti_client_module, 'OpenAIClient', fake_openai_cls)
        mock_config.llm.max_tokens = 2048

        build_llm_client(mock_config)

        fake_openai_cls.assert_called_once()
        assert 'max_tokens' not in fake_openai_cls.call_args.kwargs
        assert fake_openai_cls.call_args.args == ()


class TestBuildLLMClientOpenAIGenericPath:
    """client_class='openai_generic' → graphiti's OpenAIGenericClient.

    Construction only — no request is issued, so the unroutable api_url below
    is never dialled.
    """

    @pytest.fixture
    def generic_config(self, mock_config):
        mock_config.llm.client_class = 'openai_generic'
        mock_config.llm.providers.openai.api_url = 'http://127.0.0.1:9/v1'
        return mock_config

    def test_returns_openai_generic_client(self, generic_config):
        client = build_llm_client(generic_config)
        assert isinstance(client, OpenAIGenericClient)

    def test_auto_mode_returns_stock_client_not_the_forcing_subclass(
        self, generic_config,
    ):
        """structured_output_mode defaults to 'auto' → stock upstream client."""
        from fused_memory.backends.llm_clients import ForceJsonObjectOpenAIGenericClient

        assert generic_config.llm.structured_output_mode == 'auto'
        client = build_llm_client(generic_config)
        assert isinstance(client, OpenAIGenericClient)
        assert not isinstance(client, ForceJsonObjectOpenAIGenericClient)

    def test_plumbs_base_url_and_preserves_other_kwargs(
        self, generic_config, monkeypatch,
    ):
        generic_config.llm.model = 'qwen3-30b-a3b'
        generic_config.llm.max_tokens = 8192
        generic_config.llm.temperature = 0.2
        calls = _record_llm_config(monkeypatch)

        build_llm_client(generic_config)

        assert calls[0]['base_url'] == 'http://127.0.0.1:9/v1'
        assert calls[0]['model'] == 'qwen3-30b-a3b'
        assert calls[0]['small_model'] == 'qwen3-30b-a3b'
        assert calls[0]['temperature'] == 0.2
        assert calls[0]['max_tokens'] == 8192
        assert calls[0]['api_key'] == 'test-key'

    def test_does_not_run_responses_api_preflight(self, generic_config, monkeypatch):
        """The preflight must NOT run on this arm.

        check_openai_responses_api guards graphiti's OpenAIClient, which calls
        client.responses.create. OpenAIGenericClient uses chat.completions and
        never touches that surface, so running the check here would abort a
        perfectly valid local-endpoint arm on an SDK requirement it does not
        have. A raising sentinel pins the ordering so it cannot silently
        regress.
        """
        def sentinel_check():
            raise RuntimeError('PREFLIGHT_SENTINEL')

        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', sentinel_check,
        )

        client = build_llm_client(generic_config)  # must NOT raise
        assert isinstance(client, OpenAIGenericClient)

    @pytest.mark.parametrize('mode', ['json_object', 'auto'])
    def test_configured_max_tokens_reaches_the_constructed_client(
        self, generic_config, mode,
    ):
        """``cfg.llm.max_tokens`` must survive construction on BOTH sub-branches.

        Measured against the installed graphiti-core 0.28.2 wheel:
        ``OpenAIGenericClient.__init__`` (openai_generic_client.py:61-66,88)
        declares its own ``max_tokens: int = 16384`` and, immediately after
        ``super().__init__(config, cache)`` has correctly set
        ``self.max_tokens = config.max_tokens`` (client.py:80), unconditionally
        re-assigns ``self.max_tokens = max_tokens``. ``_generate_response`` then
        sends ``max_tokens=self.max_tokens`` (:126), ignoring the per-call
        argument — so the constructor kwarg is the ONLY lever that reaches the
        wire, and a client built with ``config=`` alone asks for 16384 output
        tokens no matter what the operator configured.

        2048 is deliberately distinct from both the schema default (4096,
        config/schema.py:214) and upstream's 16384, so a pass cannot be a
        coincidence of either.
        """
        generic_config.llm.structured_output_mode = mode
        generic_config.llm.max_tokens = 2048

        client = build_llm_client(generic_config)

        assert isinstance(client, OpenAIGenericClient)
        assert client.max_tokens == 2048

    def test_missing_api_key_still_returns_none(self, generic_config):
        """The api_key guard is shared, not duplicated per client_class."""
        generic_config.llm.providers.openai.api_key = ''
        assert build_llm_client(generic_config) is None

    def test_does_not_hijack_the_anthropic_branch(self, generic_config, monkeypatch):
        """client_class selects among OpenAI-shaped clients ONLY."""
        from fused_memory.config.schema import AnthropicProviderConfig

        generic_config.llm.provider = 'anthropic'
        generic_config.llm.providers.anthropic = AnthropicProviderConfig(
            api_key='anthropic-test-key',
        )

        fake_anthropic_cls = MagicMock(name='AnthropicClient')
        with patch(
            'graphiti_core.llm_client.anthropic_client.AnthropicClient',
            fake_anthropic_cls,
        ):
            client = build_llm_client(generic_config)

        assert client is fake_anthropic_cls.return_value
        assert not isinstance(client, OpenAIGenericClient)


class _Inner(BaseModel):
    """Nested submodel — its presence is what makes model_json_schema() emit
    ``$defs``/``$ref``, which is the exact shape llama.cpp#21228 silently
    ignores."""

    name: str


class _Outer(BaseModel):
    items: list[_Inner]
    note: str


def _fake_openai_client(content: str = '{"ok": true}') -> MagicMock:
    """A stand-in for AsyncOpenAI, injected via OpenAIGenericClient(client=...).

    That constructor argument bypasses AsyncOpenAI construction entirely, so
    the client under test issues no network call.
    """
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=response)
    return fake


def _messages() -> list[Message]:
    return [
        Message(role='system', content='You extract entities.'),
        Message(role='user', content='Alice knows Bob.'),
    ]


class TestForceJsonObjectOpenAIGenericClient:
    """The json_object forcing wrapper.

    Driven through the client's own ``client=`` injection seam rather than the
    network — construction and request shaping only.
    """

    @pytest.mark.asyncio
    async def test_forces_json_object_despite_a_response_model(self):
        """The whole point: a response_model is passed, yet the request still
        asks for json_object."""
        from fused_memory.backends.llm_clients import ForceJsonObjectOpenAIGenericClient

        fake = _fake_openai_client()
        client = ForceJsonObjectOpenAIGenericClient(
            config=GraphitiLLMConfig(api_key='k', model='m'), client=fake,
        )

        await client.generate_response(_messages(), response_model=_Outer)

        kwargs = fake.chat.completions.create.call_args.kwargs
        assert kwargs['response_format'] == {'type': 'json_object'}

    @pytest.mark.asyncio
    async def test_stock_client_sends_json_schema_with_refs(self):
        """Control assertion — pins that the wrapper changes REAL upstream
        behaviour, and that the schema genuinely contains $defs/$ref.

        Without this, a future graphiti-core wheel that inlined the schema (or
        stopped using json_schema at all) would make the wrapper silently
        pointless while every other assertion here stayed green.
        """
        fake = _fake_openai_client()
        client = OpenAIGenericClient(
            config=GraphitiLLMConfig(api_key='k', model='m'), client=fake,
        )

        await client.generate_response(_messages(), response_model=_Outer)

        response_format = fake.chat.completions.create.call_args.kwargs['response_format']
        assert response_format['type'] == 'json_schema'
        schema = response_format['json_schema']['schema']
        assert '$defs' in schema, (
            'expected a nested-model schema to carry $defs — if upstream started '
            'inlining it, the forcing wrapper is no longer load-bearing'
        )
        assert '$ref' in json.dumps(schema)

    @pytest.mark.asyncio
    async def test_override_changes_nothing_but_the_response_format(self):
        """Everything else the request carries is untouched by the override —
        so retry, tracing and message construction are unaffected."""
        from fused_memory.backends.llm_clients import ForceJsonObjectOpenAIGenericClient

        config = GraphitiLLMConfig(api_key='k', model='m', temperature=0.3)

        stock_fake = _fake_openai_client()
        forced_fake = _fake_openai_client()
        stock = OpenAIGenericClient(config=config, client=stock_fake)
        forced = ForceJsonObjectOpenAIGenericClient(config=config, client=forced_fake)

        await stock.generate_response(_messages(), response_model=_Outer)
        await forced.generate_response(_messages(), response_model=_Outer)

        stock_kwargs = dict(stock_fake.chat.completions.create.call_args.kwargs)
        forced_kwargs = dict(forced_fake.chat.completions.create.call_args.kwargs)
        stock_kwargs.pop('response_format')
        forced_kwargs.pop('response_format')
        assert stock_kwargs == forced_kwargs

    @pytest.mark.asyncio
    async def test_returns_the_parsed_payload(self):
        """The response is still json.loads'd and returned normally."""
        from fused_memory.backends.llm_clients import ForceJsonObjectOpenAIGenericClient

        fake = _fake_openai_client('{"ok": true}')
        client = ForceJsonObjectOpenAIGenericClient(
            config=GraphitiLLMConfig(api_key='k', model='m'), client=fake,
        )

        result = await client.generate_response(_messages(), response_model=_Outer)

        assert result == {'ok': True}


class TestStructuredOutputModeSelection:
    """build_llm_client wiring for llm.structured_output_mode."""

    def test_json_object_mode_selects_the_forcing_subclass(self, mock_config):
        from fused_memory.backends.llm_clients import ForceJsonObjectOpenAIGenericClient

        mock_config.llm.client_class = 'openai_generic'
        mock_config.llm.structured_output_mode = 'json_object'

        client = build_llm_client(mock_config)

        assert isinstance(client, ForceJsonObjectOpenAIGenericClient)

    def test_auto_mode_selects_the_stock_client(self, mock_config):
        from fused_memory.backends.llm_clients import ForceJsonObjectOpenAIGenericClient

        mock_config.llm.client_class = 'openai_generic'
        mock_config.llm.structured_output_mode = 'auto'

        client = build_llm_client(mock_config)

        assert isinstance(client, OpenAIGenericClient)
        assert not isinstance(client, ForceJsonObjectOpenAIGenericClient)

    def test_mode_is_inert_on_the_default_openai_arm(self, mock_config, monkeypatch):
        """structured_output_mode applies ONLY to the generic client."""
        from fused_memory.backends.llm_clients import ForceJsonObjectOpenAIGenericClient

        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', lambda: None,
        )
        mock_config.llm.client_class = 'openai'
        mock_config.llm.structured_output_mode = 'json_object'

        client = build_llm_client(mock_config)

        assert isinstance(client, OpenAIClient)
        assert not isinstance(client, ForceJsonObjectOpenAIGenericClient)


class TestBuildLLMClientBranchPreservation:
    """The non-openai-happy-path branches must survive the extraction intact."""

    def test_missing_api_key_returns_none(self, mock_config, monkeypatch):
        """No api_key → no client, and the preflight does not run either
        (review fix, task 2053: it must not run ahead of the api_key check)."""
        mock_config.llm.providers.openai.api_key = ''
        mock_preflight = MagicMock(side_effect=RuntimeError('PREFLIGHT_SENTINEL'))
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', mock_preflight,
        )

        assert build_llm_client(mock_config) is None
        mock_preflight.assert_not_called()

    def test_none_provider_block_returns_none(self, mock_config, monkeypatch):
        mock_config.llm.providers.openai = None
        mock_preflight = MagicMock(side_effect=RuntimeError('PREFLIGHT_SENTINEL'))
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', mock_preflight,
        )

        assert build_llm_client(mock_config) is None
        mock_preflight.assert_not_called()

    def test_anthropic_provider_builds_anthropic_client(self, mock_config, monkeypatch):
        """provider='anthropic' takes its own branch, with its own lazy import."""
        from fused_memory.config.schema import AnthropicProviderConfig

        mock_config.llm.provider = 'anthropic'
        mock_config.llm.providers.anthropic = AnthropicProviderConfig(
            api_key='anthropic-test-key',
        )
        mock_preflight = MagicMock(side_effect=RuntimeError('PREFLIGHT_SENTINEL'))
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', mock_preflight,
        )

        fake_anthropic_cls = MagicMock(name='AnthropicClient')
        with patch(
            'graphiti_core.llm_client.anthropic_client.AnthropicClient',
            fake_anthropic_cls,
        ):
            client = build_llm_client(mock_config)

        assert client is fake_anthropic_cls.return_value
        # The OpenAI-only Responses preflight has no business on this branch.
        mock_preflight.assert_not_called()

    def test_anthropic_provider_without_api_key_returns_none(
        self, mock_config, monkeypatch,
    ):
        from fused_memory.config.schema import AnthropicProviderConfig

        mock_config.llm.provider = 'anthropic'
        mock_config.llm.providers.anthropic = AnthropicProviderConfig(api_key='')
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', lambda: None,
        )

        assert build_llm_client(mock_config) is None
