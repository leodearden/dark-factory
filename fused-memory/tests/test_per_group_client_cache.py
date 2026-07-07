"""Tests for the per-group Graphiti client cache in GraphitiBackend (task 2266, 2115 Phase 0a).

Covers the fix for an ACTIVE data-integrity bug: graphiti_core's
Graphiti.add_episode mutates the SHARED client's driver in place when
group_id != self.driver._database (graphiti.py:887-890), so concurrent
cross-group add_episode calls on one shared client misroute writes into the
wrong FalkorDB graph. The fix builds one Graphiti client PER group_id, each
pinned to its own _driver_for(group_id) so the driver's _database already
equals group_id and the upstream mutation branch is never taken.

- GraphitiBackend.initialize() hoisting shared llm_client/embedder/cross_encoder
  to instance attrs (step-1/2).
- GraphitiBackend._client_for(group_id): per-group cached Graphiti client (step-3/4).
- GraphitiBackend.add_episode routing through _client_for + concurrency isolation (step-5/6).
- GraphitiBackend.build_communities passing the per-group driver (step-7/8).
- GraphitiBackend.close() accounting for per-group clients (step-9/10).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import fused_memory.backends.graphiti_client as graphiti_client_module
from fused_memory.backends.graphiti_client import GraphitiBackend
from graphiti_core.cross_encoder.client import CrossEncoderClient

# ---------------------------------------------------------------------------
# step-1/2: GraphitiBackend.initialize() hoists shared sub-clients
# ---------------------------------------------------------------------------


class TestInitializeHoistsSharedSubClients:
    """initialize() must hoist llm_client/embedder to instance attrs and build
    ONE shared cross_encoder, wiring all three into the base Graphiti(...)."""

    @pytest.mark.asyncio
    async def test_hoists_llm_embedder_and_shared_cross_encoder(self, mock_config, monkeypatch):
        captured: dict = {}

        def fake_graphiti(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        cross_encoder_sentinel = MagicMock(spec=CrossEncoderClient)
        mock_reranker_cls = MagicMock(return_value=cross_encoder_sentinel)

        mock_driver_cls = MagicMock()
        mock_driver_cls.return_value.client.list_graphs = AsyncMock(return_value=[])

        monkeypatch.setattr(graphiti_client_module, 'Graphiti', fake_graphiti)
        monkeypatch.setattr(graphiti_client_module, 'OpenAIRerankerClient', mock_reranker_cls)
        monkeypatch.setattr(graphiti_client_module, '_MultiTenantFalkorDriver', mock_driver_cls)
        monkeypatch.setattr(graphiti_client_module, 'check_openai_responses_api', lambda: None)

        backend = GraphitiBackend(mock_config)
        await backend.initialize()

        assert backend._llm_client is not None
        assert backend._embedder is not None
        assert captured['llm_client'] is backend._llm_client
        assert captured['embedder'] is backend._embedder

        mock_reranker_cls.assert_called_once()
        assert backend._cross_encoder is cross_encoder_sentinel
        assert captured['cross_encoder'] is backend._cross_encoder
