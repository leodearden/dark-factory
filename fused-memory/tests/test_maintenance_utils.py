"""Tests for fused_memory.maintenance._utils: override_config_path and maintenance_service."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.maintenance._utils import override_config_path


class TestOverrideConfigPath:
    """override_config_path sets/restores CONFIG_PATH env var around a block."""

    def test_sets_config_path(self, monkeypatch):
        """Sets CONFIG_PATH when config_path is provided and it was previously unset."""
        monkeypatch.delenv('CONFIG_PATH', raising=False)

        with override_config_path('/some/path/config.yaml'):
            assert os.environ.get('CONFIG_PATH') == '/some/path/config.yaml'

    def test_restores_original_value(self, monkeypatch):
        """Restores the previous CONFIG_PATH value after the block exits."""
        monkeypatch.setenv('CONFIG_PATH', '/original/config.yaml')

        with override_config_path('/new/path/config.yaml'):
            assert os.environ.get('CONFIG_PATH') == '/new/path/config.yaml'

        assert os.environ.get('CONFIG_PATH') == '/original/config.yaml'

    def test_removes_env_var_when_previously_unset(self, monkeypatch):
        """Removes CONFIG_PATH after block when it was not set before entry."""
        monkeypatch.delenv('CONFIG_PATH', raising=False)

        with override_config_path('/some/path/config.yaml'):
            pass  # set during block

        assert 'CONFIG_PATH' not in os.environ

    def test_noop_when_config_path_is_none(self, monkeypatch):
        """Does not touch CONFIG_PATH at all when config_path is None."""
        # Case 1: env var already set — should remain unchanged
        monkeypatch.setenv('CONFIG_PATH', '/existing/config.yaml')
        with override_config_path(None):
            assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'
        assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'

        # Case 2: env var not set — should remain absent
        monkeypatch.delenv('CONFIG_PATH', raising=False)
        with override_config_path(None):
            assert 'CONFIG_PATH' not in os.environ
        assert 'CONFIG_PATH' not in os.environ

    def test_restores_on_exception(self, monkeypatch):
        """Restores CONFIG_PATH even when the body raises an exception."""
        monkeypatch.delenv('CONFIG_PATH', raising=False)

        with pytest.raises(RuntimeError, match='boom'), override_config_path('/some/path/config.yaml'):
            raise RuntimeError('boom')

        assert 'CONFIG_PATH' not in os.environ

    def test_restores_prior_value_on_exception(self, monkeypatch):
        """Restores previous CONFIG_PATH value when the body raises an exception."""
        monkeypatch.setenv('CONFIG_PATH', '/original/config.yaml')

        with pytest.raises(RuntimeError, match='boom'), override_config_path('/new/path/config.yaml'):
            raise RuntimeError('boom')

        assert os.environ.get('CONFIG_PATH') == '/original/config.yaml'


# ---------------------------------------------------------------------------
# TestMaintenanceService: maintenance_service async context manager
# ---------------------------------------------------------------------------

class TestMaintenanceService:
    """maintenance_service encapsulates the full service lifecycle as an async context manager."""

    @pytest.mark.asyncio
    async def test_yields_config_and_service_tuple(self):
        """maintenance_service yields a (FusedMemoryConfig, MemoryService) tuple."""
        from fused_memory.maintenance._utils import maintenance_service

        mock_cfg = MagicMock()
        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance._utils.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service(None) as (cfg, svc):
                assert cfg is mock_cfg
                assert svc is mock_service

    @pytest.mark.asyncio
    async def test_calls_initialize_before_yield(self):
        """maintenance_service calls service.initialize() before yielding."""
        from fused_memory.maintenance._utils import maintenance_service

        call_order = []
        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock(side_effect=lambda: call_order.append('initialize'))

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance._utils.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service(None):
                call_order.append('body')

        assert call_order == ['initialize', 'body']

    @pytest.mark.asyncio
    async def test_calls_close_on_normal_exit(self):
        """maintenance_service calls service.close() when the body exits normally."""
        from fused_memory.maintenance._utils import maintenance_service

        mock_cfg = MagicMock()
        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance._utils.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service(None):
                pass

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_calls_close_on_exception(self):
        """maintenance_service calls service.close() even when the body raises."""
        from fused_memory.maintenance._utils import maintenance_service

        mock_cfg = MagicMock()
        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance._utils.MemoryService', return_value=mock_service),
            pytest.raises(RuntimeError, match='body error'),
        ):
            async with maintenance_service(None):
                raise RuntimeError('body error')

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_exception_does_not_mask_original_error(self):
        """When both the body and close() raise, the original body error propagates."""
        from fused_memory.maintenance._utils import maintenance_service

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.close = AsyncMock(side_effect=RuntimeError('close error'))

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.maintenance._utils.MemoryService', return_value=mock_service),
            pytest.raises(RuntimeError, match='original'),
        ):
            async with maintenance_service(None):
                raise RuntimeError('original')

    @pytest.mark.asyncio
    async def test_skips_close_when_config_fails(self):
        """When FusedMemoryConfig raises, service is never created and close() is not called."""
        from fused_memory.maintenance._utils import maintenance_service

        with (
            patch(
                'fused_memory.maintenance._utils.FusedMemoryConfig',
                side_effect=RuntimeError('config failed'),
            ),
            patch('fused_memory.maintenance._utils.MemoryService') as mock_svc_cls,
            pytest.raises(RuntimeError, match='config failed'),
        ):
            async with maintenance_service(None):
                pass  # pragma: no cover

        mock_svc_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_config_path_set_during_config_construction(self, monkeypatch):
        """FusedMemoryConfig is called with CONFIG_PATH set to config_path."""
        from fused_memory.maintenance._utils import maintenance_service

        monkeypatch.delenv('CONFIG_PATH', raising=False)
        captured: dict[str, str | None] = {}

        def capture_config(*args, **kwargs):
            captured['value'] = os.environ.get('CONFIG_PATH')
            return MagicMock()

        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', side_effect=capture_config),
            patch('fused_memory.maintenance._utils.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service('/tmp/custom.yaml'):
                pass

        assert captured['value'] == '/tmp/custom.yaml'

    @pytest.mark.asyncio
    async def test_config_path_restored_on_normal_exit(self, monkeypatch):
        """CONFIG_PATH is restored to its original value after the context manager exits."""
        from fused_memory.maintenance._utils import maintenance_service

        monkeypatch.delenv('CONFIG_PATH', raising=False)
        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=MagicMock()),
            patch('fused_memory.maintenance._utils.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service('/tmp/custom.yaml'):
                pass

        assert 'CONFIG_PATH' not in os.environ

    @pytest.mark.asyncio
    async def test_config_path_none_does_not_touch_env(self, monkeypatch):
        """When config_path is None, CONFIG_PATH is not modified."""
        from fused_memory.maintenance._utils import maintenance_service

        monkeypatch.setenv('CONFIG_PATH', '/existing/config.yaml')
        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=MagicMock()),
            patch('fused_memory.maintenance._utils.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service(None):
                assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'

        assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'
