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
