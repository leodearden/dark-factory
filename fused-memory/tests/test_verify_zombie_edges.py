"""Tests for verify_zombie_edges maintenance: run_verify_zombie_edges() edge cases."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# step-7: run_verify_zombie_edges() env-var / close() edge cases
# ---------------------------------------------------------------------------

class TestRunVerifyZombieEdgesEdgeCases:
    """run_verify_zombie_edges() correctly saves/restores CONFIG_PATH and handles close() edge cases.

    All tests pass uuids=['dummy-uuid'] to bypass the ValueError guard that
    rejects empty uuid lists.
    """

    @pytest.mark.asyncio
    async def test_restores_preexisting_env_var_when_constructor_fails(self):
        """When CONFIG_PATH already set and FusedMemoryConfig raises, old value is restored."""
        import os
        from fused_memory.maintenance.verify_zombie_edges import run_verify_zombie_edges

        old = os.environ.get('CONFIG_PATH')
        os.environ['CONFIG_PATH'] = 'preexisting.yaml'
        try:
            with patch(
                'fused_memory.maintenance.verify_zombie_edges.FusedMemoryConfig',
                side_effect=RuntimeError('config load failed'),
            ):
                with pytest.raises(RuntimeError, match='config load failed'):
                    await run_verify_zombie_edges(
                        uuids=['dummy-uuid'], config_path='test.yaml'
                    )

            assert os.environ.get('CONFIG_PATH') == 'preexisting.yaml'
        finally:
            if old is None:
                os.environ.pop('CONFIG_PATH', None)
            else:
                os.environ['CONFIG_PATH'] = old

    @pytest.mark.asyncio
    async def test_does_not_touch_env_var_when_config_path_is_none(self):
        """When config_path is None, CONFIG_PATH env var is left completely untouched."""
        import os
        from fused_memory.maintenance.verify_zombie_edges import run_verify_zombie_edges

        old = os.environ.get('CONFIG_PATH')
        os.environ['CONFIG_PATH'] = 'existing.yaml'
        try:
            with patch(
                'fused_memory.maintenance.verify_zombie_edges.FusedMemoryConfig',
                side_effect=RuntimeError('config load failed'),
            ):
                with pytest.raises(RuntimeError, match='config load failed'):
                    await run_verify_zombie_edges(uuids=['dummy-uuid'])  # no config_path

            assert os.environ.get('CONFIG_PATH') == 'existing.yaml'
        finally:
            if old is None:
                os.environ.pop('CONFIG_PATH', None)
            else:
                os.environ['CONFIG_PATH'] = old

    @pytest.mark.asyncio
    async def test_close_exception_does_not_mask_original_error(self):
        """When both verifier.cleanup and close() raise, the original error propagates."""
        from fused_memory.maintenance.verify_zombie_edges import run_verify_zombie_edges

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.close = AsyncMock(side_effect=RuntimeError('close error'))

        with (
            patch(
                'fused_memory.maintenance.verify_zombie_edges.FusedMemoryConfig',
                return_value=mock_cfg,
            ),
            patch(
                'fused_memory.maintenance.verify_zombie_edges.MemoryService',
                return_value=mock_service,
            ),
            patch(
                'fused_memory.maintenance.verify_zombie_edges.ZombieEdgeVerifier'
            ) as mock_verifier_cls,
        ):
            mock_verifier = MagicMock()
            mock_verifier.cleanup = AsyncMock(side_effect=RuntimeError('original'))
            mock_verifier_cls.return_value = mock_verifier

            with pytest.raises(RuntimeError, match='original'):
                await run_verify_zombie_edges(uuids=['dummy-uuid'])

    @pytest.mark.asyncio
    async def test_skips_close_when_service_never_created(self):
        """When FusedMemoryConfig raises, service is None and close() is never called."""
        from fused_memory.maintenance.verify_zombie_edges import run_verify_zombie_edges

        with (
            patch(
                'fused_memory.maintenance.verify_zombie_edges.FusedMemoryConfig',
                side_effect=RuntimeError('config failed'),
            ),
            patch(
                'fused_memory.maintenance.verify_zombie_edges.MemoryService'
            ) as mock_svc_cls,
        ):
            with pytest.raises(RuntimeError, match='config failed'):
                await run_verify_zombie_edges(uuids=['dummy-uuid'])

        mock_svc_cls.assert_not_called()  # MemoryService never instantiated
        mock_svc_cls.return_value.close.assert_not_called()
