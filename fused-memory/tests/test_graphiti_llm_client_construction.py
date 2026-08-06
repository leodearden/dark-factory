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

from unittest.mock import MagicMock, patch

import pytest
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig

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
