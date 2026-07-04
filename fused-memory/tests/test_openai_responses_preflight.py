"""Tests for the openai Responses-API startup preflight (task 2053).

graphiti-core's OpenAIClient issues ``client.responses.create(...)``, which
lazily resolves ``openai.resources.responses`` — a submodule introduced in
openai 1.66.0. An installed openai older than that raises
``ModuleNotFoundError`` deep inside Graphiti write-path LLM extraction,
where the durable queue treats it as non-retriable and dead-letters
silently after exhausting retries.

``check_openai_responses_api()`` converts that into an immediate, actionable
boot-time ``RuntimeError`` instead. These tests are hermetic: no FalkorDB,
no network.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend, check_openai_responses_api


class TestCheckOpenaiResponsesApi:
    """Unit tests for the standalone capability-check function."""

    def test_raises_actionable_error_when_responses_module_missing(self, monkeypatch):
        """When openai.resources.responses can't be found, raise a RuntimeError
        naming the missing module and the uv sync remediation."""
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == 'openai.resources.responses':
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)

        with pytest.raises(RuntimeError) as exc_info:
            check_openai_responses_api()

        message = str(exc_info.value).lower()
        assert 'openai.resources.responses' in message
        assert 'uv sync' in message

    def test_does_not_raise_when_module_present(self):
        """In the real environment, graphiti-core hard-requires openai>=1.91.0,
        which ships openai.resources.responses — so the check is a no-op."""
        assert check_openai_responses_api() is None

    def test_raises_actionable_error_when_version_below_floor(self, monkeypatch):
        """Module presence alone isn't enough: openai in [1.66.0, 1.91.0) ships
        the submodule but is still below graphiti-core's declared floor, so
        the version-floor guard must fire the same actionable error (review
        hardening, task 2053)."""
        import openai

        monkeypatch.setattr(openai, '__version__', '1.80.0')

        with pytest.raises(RuntimeError) as exc_info:
            check_openai_responses_api()

        message = str(exc_info.value).lower()
        assert 'openai.resources.responses' in message
        assert 'uv sync' in message
        assert '1.80.0' in str(exc_info.value)

    def test_does_not_raise_for_prerelease_version_at_floor(self, monkeypatch):
        """A prerelease suffix (e.g. 'rc1') must not defeat the floor parse:
        1.91.0rc1 parses to (1, 91, 0), which is not below the floor."""
        import openai

        monkeypatch.setattr(openai, '__version__', '1.91.0rc1')

        assert check_openai_responses_api() is None

    def test_raises_actionable_error_when_find_spec_raises(self, monkeypatch):
        """If resolving the submodule spec itself raises (e.g. a broken parent
        openai/openai.resources import), that must be converted into the same
        actionable RuntimeError rather than escaping as a raw
        Import/ModuleNotFoundError that buries the remediation (review
        hardening, task 2053)."""

        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == 'openai.resources.responses':
                raise ModuleNotFoundError("No module named 'openai.resources'")
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)

        with pytest.raises(RuntimeError) as exc_info:
            check_openai_responses_api()

        message = str(exc_info.value).lower()
        assert 'openai.resources.responses' in message
        assert 'uv sync' in message


class TestGraphitiBackendInitializePreflightWiring:
    """Wiring test: initialize() must invoke the preflight before touching FalkorDB."""

    @pytest.mark.asyncio
    async def test_initialize_runs_preflight_before_falkordb_driver(
        self, mock_config, monkeypatch,
    ):
        """The preflight must run on the OpenAI-LLM path before any FalkorDB
        driver / Graphiti client construction, so an incompatible openai
        aborts startup immediately instead of reaching a real connection."""
        import fused_memory.backends.graphiti_client as graphiti_client_module

        def sentinel_check():
            raise RuntimeError('PREFLIGHT_SENTINEL')

        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', sentinel_check,
        )
        # Guard against the RED run reaching a real FalkorDB connection.
        monkeypatch.setattr(
            graphiti_client_module, '_MultiTenantFalkorDriver', MagicMock(),
        )
        monkeypatch.setattr(
            graphiti_client_module, 'Graphiti', MagicMock(return_value=AsyncMock()),
        )

        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='PREFLIGHT_SENTINEL'):
            await backend.initialize()

    @pytest.mark.asyncio
    async def test_initialize_skips_preflight_when_api_key_missing(
        self, mock_config, monkeypatch,
    ):
        """When the openai provider is configured but has no api_key, no
        OpenAIClient is ever constructed (llm_client stays None) — so the
        preflight, which only guards the path that actually constructs
        OpenAIClient, must not run either (review fix, task 2053: it
        previously ran unconditionally on the openai-provider branch, before
        the api_key check)."""
        import fused_memory.backends.graphiti_client as graphiti_client_module

        mock_config.llm.providers.openai.api_key = ''

        mock_preflight = MagicMock(side_effect=RuntimeError('PREFLIGHT_SENTINEL'))
        monkeypatch.setattr(
            graphiti_client_module, 'check_openai_responses_api', mock_preflight,
        )
        monkeypatch.setattr(
            graphiti_client_module, '_MultiTenantFalkorDriver', MagicMock(),
        )
        monkeypatch.setattr(
            graphiti_client_module, 'Graphiti', MagicMock(return_value=AsyncMock()),
        )

        backend = GraphitiBackend(mock_config)
        await backend.initialize()  # must complete without the sentinel firing

        mock_preflight.assert_not_called()
