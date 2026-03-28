"""Tests for verify_zombie_edges maintenance: run_verify_zombie_edges() delegation tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.maintenance.verify_zombie_edges import VerifyResult, run_verify_zombie_edges

# ---------------------------------------------------------------------------
# step-5: run_verify_zombie_edges delegates to maintenance_service
# ---------------------------------------------------------------------------

class TestRunVerifyZombieEdgesDelegation:
    """run_verify_zombie_edges() delegates service lifecycle to maintenance_service()."""

    @pytest.mark.asyncio
    async def test_delegates_to_maintenance_service(self, make_fake_maintenance_service):
        """run_verify_zombie_edges() calls maintenance_service(config_path) and uses yielded service.graphiti."""
        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_result = VerifyResult(found=['u1'], missing=[], deleted=0)

        with (
            patch(
                'fused_memory.maintenance.verify_zombie_edges.maintenance_service',
                side_effect=make_fake_maintenance_service(mock_cfg, mock_service),
            ),
            patch('fused_memory.maintenance.verify_zombie_edges.ZombieEdgeVerifier') as mock_verifier_cls,
        ):
            mock_verifier = MagicMock()
            mock_verifier.cleanup = AsyncMock(return_value=mock_result)
            mock_verifier_cls.return_value = mock_verifier

            result = await run_verify_zombie_edges(
                uuids=['u1'],
                config_path='/tmp/config.yaml',
            )

        mock_verifier_cls.assert_called_once_with(backend=mock_service.graphiti)
        assert result is mock_result
