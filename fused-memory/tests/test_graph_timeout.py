"""Tests for GraphitiBackend.ensure_graph_timeout()."""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.config.schema import FalkorDBProviderConfig, GraphitiBackendConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(config) -> GraphitiBackend:
    """Build a GraphitiBackend with a mock Graphiti client and mock _driver."""
    backend = GraphitiBackend(config)
    backend.client = MagicMock()
    # Inject a mock driver whose .client.execute_command is an AsyncMock
    mock_driver = MagicMock()
    mock_driver.client = MagicMock()
    mock_driver.client.execute_command = AsyncMock()
    backend._driver = mock_driver
    return backend


# ---------------------------------------------------------------------------
# step-3: ensure_graph_timeout — happy path
# ---------------------------------------------------------------------------

class TestEnsureGraphTimeout:
    """GraphitiBackend.ensure_graph_timeout() sends GRAPH.CONFIG SET and GET."""

    @pytest.mark.asyncio
    async def test_sends_graph_config_set_timeout(self, mock_config):
        backend = _make_backend(mock_config)
        # Mock GET response: [b'TIMEOUT', b'30000']
        backend._driver.client.execute_command.return_value = [b'TIMEOUT', b'30000']

        await backend.ensure_graph_timeout(30000)

        # First call should be SET
        set_call = backend._driver.client.execute_command.call_args_list[0]
        assert set_call.args == ('GRAPH.CONFIG', 'SET', 'TIMEOUT', 30000)

    @pytest.mark.asyncio
    async def test_sends_graph_config_get_timeout_after_set(self, mock_config):
        backend = _make_backend(mock_config)
        backend._driver.client.execute_command.return_value = [b'TIMEOUT', b'30000']

        await backend.ensure_graph_timeout(30000)

        # Two calls total: SET then GET
        assert backend._driver.client.execute_command.call_count == 2
        get_call = backend._driver.client.execute_command.call_args_list[1]
        assert get_call.args == ('GRAPH.CONFIG', 'GET', 'TIMEOUT')

    @pytest.mark.asyncio
    async def test_logs_confirmed_timeout_at_info(self, mock_config, caplog):
        backend = _make_backend(mock_config)
        backend._driver.client.execute_command.return_value = [b'TIMEOUT', b'30000']

        with caplog.at_level(logging.INFO, logger='fused_memory.backends.graphiti_client'):
            await backend.ensure_graph_timeout(30000)

        assert any('30000' in record.message for record in caplog.records
                   if record.levelno == logging.INFO)


# ---------------------------------------------------------------------------
# step-5: error resilience
# ---------------------------------------------------------------------------

class TestEnsureGraphTimeoutErrorResilience:
    """ensure_graph_timeout logs warning but does NOT raise on errors."""

    @pytest.mark.asyncio
    async def test_does_not_raise_when_set_fails(self, mock_config, caplog):
        backend = _make_backend(mock_config)
        backend._driver.client.execute_command.side_effect = Exception('connection refused')

        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'):
            # Should not raise
            await backend.ensure_graph_timeout(30000)

        assert any('warning' in record.levelname.lower() or record.levelno == logging.WARNING
                   for record in caplog.records)

    @pytest.mark.asyncio
    async def test_does_not_raise_when_get_fails_after_set(self, mock_config, caplog):
        backend = _make_backend(mock_config)
        # First call (SET) succeeds, second (GET) fails
        backend._driver.client.execute_command.side_effect = [
            None,  # SET succeeds
            Exception('get failed'),  # GET fails
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'):
            # Should not raise
            await backend.ensure_graph_timeout(30000)

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_driver_is_none(self, mock_config):
        backend = GraphitiBackend(mock_config)
        backend._driver = None  # not initialized

        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.ensure_graph_timeout(30000)


# ---------------------------------------------------------------------------
# step-7: opt-out (timeout_ms=0)
# ---------------------------------------------------------------------------

class TestEnsureGraphTimeoutOptOut:
    """When graph_timeout_ms is 0, ensure_graph_timeout skips GRAPH.CONFIG SET."""

    @pytest.mark.asyncio
    async def test_skips_execute_command_when_zero(self, mock_config, caplog):
        backend = _make_backend(mock_config)

        with caplog.at_level(logging.DEBUG, logger='fused_memory.backends.graphiti_client'):
            await backend.ensure_graph_timeout(0)

        backend._driver.client.execute_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_logs_debug_when_opt_out(self, mock_config, caplog):
        backend = _make_backend(mock_config)

        with caplog.at_level(logging.DEBUG, logger='fused_memory.backends.graphiti_client'):
            await backend.ensure_graph_timeout(0)

        assert any(record.levelno == logging.DEBUG for record in caplog.records)


# ---------------------------------------------------------------------------
# step-9: initialize() wires ensure_graph_timeout
# ---------------------------------------------------------------------------

class TestInitializeCallsEnsureGraphTimeout:
    """GraphitiBackend.initialize() calls ensure_graph_timeout with configured value."""

    @pytest.mark.asyncio
    async def test_initialize_calls_ensure_graph_timeout_with_configured_value(self, mock_config):
        """initialize() should await ensure_graph_timeout(config.graphiti.falkordb.graph_timeout_ms)."""
        # Use a custom timeout value to distinguish it from any default
        from fused_memory.config.schema import FalkorDBProviderConfig, GraphitiBackendConfig
        from dataclasses import replace as dc_replace

        # Build config with custom timeout
        mock_config.graphiti = GraphitiBackendConfig(
            falkordb=FalkorDBProviderConfig(
                uri='redis://localhost:6379',
                graph_timeout_ms=12345,
            )
        )

        backend = GraphitiBackend(mock_config)

        with (
            patch('fused_memory.backends.graphiti_client.FalkorDriver') as mock_driver_cls,
            patch('fused_memory.backends.graphiti_client.Graphiti') as mock_graphiti_cls,
            patch.object(backend, 'ensure_graph_timeout', new=AsyncMock()) as mock_ensure,
        ):
            mock_driver_inst = MagicMock()
            mock_driver_cls.return_value = mock_driver_inst
            mock_graphiti_inst = MagicMock()
            mock_graphiti_inst.build_indices_and_constraints = AsyncMock()
            mock_graphiti_cls.return_value = mock_graphiti_inst

            await backend.initialize()

        mock_ensure.assert_awaited_once_with(12345)
