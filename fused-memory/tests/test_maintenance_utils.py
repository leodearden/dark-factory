"""Tests for fused_memory.maintenance._utils: override_config_path and maintenance_service."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import pydantic_spec

from fused_memory.config.schema import FusedMemoryConfig
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

        mock_cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.services.memory_service.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service(None) as (cfg, svc):
                assert cfg is mock_cfg
                assert svc is mock_service

    @pytest.mark.asyncio
    async def test_calls_initialize_before_yield(self):
        """maintenance_service calls service.initialize() before yielding."""
        from fused_memory.maintenance._utils import maintenance_service

        call_order = []
        mock_cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock(side_effect=lambda: call_order.append('initialize'))

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.services.memory_service.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service(None):
                call_order.append('body')

        assert call_order == ['initialize', 'body']

    @pytest.mark.asyncio
    async def test_calls_close_on_normal_exit(self):
        """maintenance_service calls service.close() when the body exits normally."""
        from fused_memory.maintenance._utils import maintenance_service

        mock_cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.services.memory_service.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service(None):
                pass

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_calls_close_on_exception(self):
        """maintenance_service calls service.close() even when the body raises."""
        from fused_memory.maintenance._utils import maintenance_service

        mock_cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        mock_service = AsyncMock()

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.services.memory_service.MemoryService', return_value=mock_service),
            pytest.raises(RuntimeError, match='body error'),
        ):
            async with maintenance_service(None):
                raise RuntimeError('body error')

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_exception_does_not_mask_original_error(self):
        """When both the body and close() raise, the original body error propagates."""
        from fused_memory.maintenance._utils import maintenance_service

        mock_cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        mock_service = AsyncMock()
        mock_service.close = AsyncMock(side_effect=RuntimeError('close error'))

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=mock_cfg),
            patch('fused_memory.services.memory_service.MemoryService', return_value=mock_service),
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
            patch('fused_memory.services.memory_service.MemoryService') as mock_svc_cls,
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
            patch('fused_memory.services.memory_service.MemoryService', return_value=mock_service),
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
            patch('fused_memory.services.memory_service.MemoryService', return_value=mock_service),
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
            patch('fused_memory.services.memory_service.MemoryService', return_value=mock_service),
        ):
            async with maintenance_service(None):
                assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'

        assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'


# ---------------------------------------------------------------------------
# TestServiceOpen: the opening modes and the config= overload
# ---------------------------------------------------------------------------


def _recording_service(calls: list[str], *, failing: str | None = None) -> AsyncMock:
    """An AsyncMock service recording initialize / initialize_for_reads / close in *calls*.

    When *failing* names an opening method, that method records itself and raises.
    """
    service = AsyncMock()

    def recorder(name: str):
        def record(*args, **kwargs):
            calls.append(name)
            if name == failing:
                raise RuntimeError(f'{name} failed')

        return record

    for name in ('initialize', 'initialize_for_reads', 'close'):
        setattr(service, name, AsyncMock(side_effect=recorder(name)))
    return service


def _as_an_unchecked_caller(**kwargs):
    """Enter maintenance_service the way a caller outside pyright's gate can."""
    from fused_memory.maintenance._utils import maintenance_service

    return maintenance_service(**kwargs)


class TestServiceOpen:
    """maintenance_service opens the service as its open_as mode says, and closes it once."""

    @pytest.mark.asyncio
    async def test_default_opening_awaits_initialize_only(self):
        from fused_memory.maintenance._utils import maintenance_service

        calls: list[str] = []
        service = _recording_service(calls)

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=MagicMock()),
            patch('fused_memory.services.memory_service.MemoryService', return_value=service),
        ):
            async with maintenance_service():
                assert calls == ['initialize']

        service.initialize.assert_awaited_once_with()
        service.initialize_for_reads.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reads_opening_awaits_initialize_for_reads_only(self):
        from fused_memory.maintenance._utils import ServiceOpen, maintenance_service

        calls: list[str] = []
        service = _recording_service(calls)

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig', return_value=MagicMock()),
            patch('fused_memory.services.memory_service.MemoryService', return_value=service),
        ):
            async with maintenance_service(open_as=ServiceOpen.READS):
                assert calls == ['initialize_for_reads']

        service.initialize_for_reads.assert_awaited_once_with()
        service.initialize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_opening_awaits_nothing_and_yields_the_constructed_service(self):
        from fused_memory.maintenance._utils import ServiceOpen, maintenance_service

        calls: list[str] = []
        service = _recording_service(calls)
        config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))

        with patch(
            'fused_memory.services.memory_service.MemoryService', return_value=service,
        ) as service_cls:
            async with maintenance_service(config=config, open_as=ServiceOpen.NONE) as (_, svc):
                assert calls == []
                assert svc is service

        service_cls.assert_called_once_with(config)
        service.initialize.assert_not_awaited()
        service.initialize_for_reads.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('open_as', ['full', 'reads', 'none'])
    async def test_close_is_awaited_once_on_a_normal_exit(self, open_as):
        from fused_memory.maintenance._utils import ServiceOpen, maintenance_service

        calls: list[str] = []
        service = _recording_service(calls)

        with patch('fused_memory.services.memory_service.MemoryService', return_value=service):
            async with maintenance_service(config=MagicMock(), open_as=ServiceOpen(open_as)):
                pass

        service.close.assert_awaited_once_with()
        assert calls[-1] == 'close'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('open_as', 'opening'),
        [('full', 'initialize'), ('reads', 'initialize_for_reads')],
    )
    async def test_a_failed_opening_still_closes_and_propagates_its_own_error(
        self, open_as, opening,
    ):
        from fused_memory.maintenance._utils import ServiceOpen, maintenance_service

        calls: list[str] = []
        service = _recording_service(calls, failing=opening)

        with (
            patch('fused_memory.services.memory_service.MemoryService', return_value=service),
            pytest.raises(RuntimeError, match=f'{opening} failed'),
        ):
            async with maintenance_service(config=MagicMock(), open_as=ServiceOpen(open_as)):
                calls.append('body')

        assert calls == [opening, 'close']

    @pytest.mark.asyncio
    async def test_a_given_config_is_used_untouched(self, monkeypatch):
        from fused_memory.maintenance._utils import maintenance_service

        monkeypatch.setenv('CONFIG_PATH', '/existing/config.yaml')
        config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        service = _recording_service([])

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig') as config_cls,
            patch(
                'fused_memory.services.memory_service.MemoryService', return_value=service,
            ) as service_cls,
        ):
            async with maintenance_service(config=config) as (cfg, svc):
                assert cfg is config
                assert svc is service
                assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'

        config_cls.assert_not_called()
        service_cls.assert_called_once_with(config)
        assert os.environ.get('CONFIG_PATH') == '/existing/config.yaml'

    @pytest.mark.asyncio
    async def test_both_config_sources_are_rejected_before_construction(self):
        from fused_memory.maintenance._utils import maintenance_service

        with (
            patch('fused_memory.services.memory_service.MemoryService') as service_cls,
            pytest.raises(ValueError, match='config_path') as excinfo,
        ):
            async with maintenance_service('/tmp/custom.yaml', config=MagicMock()):
                pytest.fail('the body must not run')

        assert 'config=' in str(excinfo.value)
        assert '/tmp/custom.yaml' in str(excinfo.value)
        service_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unknown_opening_mode_is_rejected_before_construction(self):

        with (
            patch('fused_memory.maintenance._utils.FusedMemoryConfig') as config_cls,
            patch('fused_memory.services.memory_service.MemoryService') as service_cls,
            pytest.raises(ValueError, match='write'),
        ):
            async with _as_an_unchecked_caller(open_as='write'):
                pytest.fail('the body must not run')

        config_cls.assert_not_called()
        service_cls.assert_not_called()
