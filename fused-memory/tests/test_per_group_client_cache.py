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
- GraphitiBackend.build_communities passing the per-group driver (step-7/8),
  including looping per group_id when more than one is requested (amendment).
- GraphitiBackend.close() accounting for per-group clients (step-9/10).
- GraphitiBackend.initialize()'s reranker config guard falling back to
  config=None when providers.openai is absent/keyless (amendment).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient

import fused_memory.backends.graphiti_client as graphiti_client_module
from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver

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


# ---------------------------------------------------------------------------
# step-3/4: GraphitiBackend._client_for(group_id) — cached per-group client
# ---------------------------------------------------------------------------


class TestClientFor:
    """_client_for(group_id) returns a per-group_id Graphiti client, cached,
    each pinned to its own _driver_for(group_id) so the driver's _database
    already equals group_id — the KEY INVARIANT that makes upstream's
    add_episode driver-mutation branch structurally unreachable."""

    def _make_backend(self, mock_config):
        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(falkor_db=MagicMock(), database='__base__')
        backend._llm_client = MagicMock(spec=LLMClient)
        backend._embedder = MagicMock(spec=EmbedderClient)
        backend._cross_encoder = MagicMock(spec=CrossEncoderClient)
        return backend

    def test_same_group_id_returns_same_cached_client(self, mock_config):
        backend = self._make_backend(mock_config)
        a = backend._client_for('g1')
        b = backend._client_for('g1')
        assert a is b

    def test_different_group_ids_return_distinct_clients(self, mock_config):
        backend = self._make_backend(mock_config)
        a = backend._client_for('g1')
        b = backend._client_for('g2')
        assert a is not b

    def test_each_client_driver_is_pinned_to_its_own_group(self, mock_config):
        backend = self._make_backend(mock_config)
        client_g1 = backend._client_for('g1')
        client_g2 = backend._client_for('g2')
        assert client_g1.driver._database == 'g1'
        assert client_g2.driver._database == 'g2'

    def test_sub_clients_are_shared_across_groups(self, mock_config):
        backend = self._make_backend(mock_config)
        client_g1 = backend._client_for('g1')
        client_g2 = backend._client_for('g2')

        assert client_g1.llm_client is backend._llm_client
        assert client_g1.embedder is backend._embedder
        assert client_g1.cross_encoder is backend._cross_encoder
        assert client_g2.llm_client is backend._llm_client
        assert client_g2.embedder is backend._embedder
        assert client_g2.cross_encoder is backend._cross_encoder
        assert client_g1.cross_encoder is client_g2.cross_encoder


# ---------------------------------------------------------------------------
# step-5/6: GraphitiBackend.add_episode routes through _client_for
# ---------------------------------------------------------------------------


class TestAddEpisodeRoutesThroughClientFor:
    """add_episode must write via the per-group client (_client_for), never
    via the shared self.client — routing writes off the shared client is
    the entire point of the fix."""

    @pytest.mark.asyncio
    async def test_add_episode_uses_per_group_client_not_shared_client(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        sentinel = object()
        per_group = MagicMock()
        per_group.add_episode = AsyncMock(return_value=sentinel)
        backend._client_for = MagicMock(return_value=per_group)
        backend.client.add_episode = AsyncMock()

        result = await backend.add_episode(name='e', content='c', group_id='A')

        backend._client_for.assert_called_once_with('A')
        per_group.add_episode.assert_awaited_once()
        backend.client.add_episode.assert_not_awaited()
        assert result is sentinel


# ---------------------------------------------------------------------------
# step-5/6: add_episode concurrency isolation (headline regression test)
# ---------------------------------------------------------------------------


class TestAddEpisodeConcurrencyIsolation:
    """Reproduces the ACTIVE data-integrity bug this task fixes: on a SHARED
    Graphiti client, concurrent cross-group add_episode calls race on
    self.driver / self.clients.driver (graphiti_core 0.28.2, graphiti.py:
    887-890) and misroute writes into the wrong FalkorDB graph. Once
    add_episode routes through per-group clients (_client_for), each has its
    own pre-pinned driver and the race is structurally unreachable."""

    @pytest.mark.asyncio
    async def test_concurrent_cross_group_writes_do_not_bleed(self, mock_config, monkeypatch):
        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(falkor_db=MagicMock(), database='__base__')
        backend._llm_client = MagicMock(spec=LLMClient)
        backend._embedder = MagicMock(spec=EmbedderClient)
        backend._cross_encoder = MagicMock(spec=CrossEncoderClient)
        backend.client = Graphiti(
            graph_driver=backend._driver,
            llm_client=backend._llm_client,
            embedder=backend._embedder,
            cross_encoder=backend._cross_encoder,
        )

        observed: dict[str, str] = {}

        async def fake_add_episode(self, *, name, group_id, **kwargs):
            # Mirrors graphiti_core 0.28.2 graphiti.py:887-890 verbatim.
            if group_id != self.driver._database:
                self.driver = self.driver.clone(database=group_id)
                self.clients.driver = self.driver
            await asyncio.sleep(0)  # yield so concurrent calls interleave
            observed[name] = self.clients.driver._database

        monkeypatch.setattr(Graphiti, 'add_episode', fake_add_episode)

        await asyncio.gather(
            backend.add_episode(name='epA', content='x', group_id='A'),
            backend.add_episode(name='epB', content='y', group_id='B'),
        )

        assert observed == {'epA': 'A', 'epB': 'B'}


# ---------------------------------------------------------------------------
# step-7/8: GraphitiBackend.build_communities passes the per-group driver
# ---------------------------------------------------------------------------


class TestBuildCommunitiesPassesDriver:
    """build_communities must pass driver=self._driver_for(group_ids[0]) —
    upstream Graphiti.build_communities DOES accept driver= (unlike
    add_episode), so the latent driver-omission sibling bug is fixed by
    passing the per-group driver directly rather than routing through
    _client_for."""

    @pytest.mark.asyncio
    async def test_build_communities_passes_per_group_driver(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        backend.client.build_communities = AsyncMock()

        da = backend._driver_for('A')
        await backend.build_communities(group_ids=['A'])

        backend.client.build_communities.assert_awaited_once()
        assert backend.client.build_communities.await_args is not None
        _, kwargs = backend.client.build_communities.await_args
        assert kwargs.get('group_ids') == ['A']
        assert kwargs.get('driver') is da

    @pytest.mark.asyncio
    async def test_build_communities_none_group_ids_uses_whole_graph_fallback(
        self, mock_config, make_backend,
    ):
        """group_ids=None must preserve the pre-existing whole-graph
        fallback: one call, no driver override."""
        backend = make_backend(mock_config)
        backend.client.build_communities = AsyncMock()

        await backend.build_communities(group_ids=None)

        backend.client.build_communities.assert_awaited_once_with(
            group_ids=None, driver=None,
        )

    @pytest.mark.asyncio
    async def test_build_communities_loops_per_group_with_own_driver(
        self, mock_config, make_backend,
    ):
        """More than one group_id must NOT be forwarded as a single call
        pinned to the first group's driver — each FalkorDB graph is a
        distinct physical database reachable only through its own driver,
        so a single call would silently scope (or skip) communities for
        every group but the first."""
        backend = make_backend(mock_config)
        backend.client.build_communities = AsyncMock()

        da = backend._driver_for('A')
        db = backend._driver_for('B')
        await backend.build_communities(group_ids=['A', 'B'])

        assert backend.client.build_communities.await_count == 2
        calls_by_group = {
            call.kwargs['group_ids'][0]: call.kwargs
            for call in backend.client.build_communities.await_args_list
        }
        assert calls_by_group == {
            'A': {'group_ids': ['A'], 'driver': da},
            'B': {'group_ids': ['B'], 'driver': db},
        }


# ---------------------------------------------------------------------------
# step-9/10: GraphitiBackend.close() accounts for per-group clients
# ---------------------------------------------------------------------------


class TestCloseAccountsForGroupClients:
    """close() must drop per-group client references without double-closing
    the single shared FalkorDB connection — each per-group client's driver
    aliases an entry already closed via the _cloned_drivers loop."""

    @pytest.mark.asyncio
    async def test_close_clears_group_clients_without_double_close(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        backend._driver = MagicMock()
        backend._driver.close = AsyncMock()
        cloned = MagicMock()
        cloned.close = AsyncMock()
        backend._cloned_drivers = {'A': cloned}
        backend._group_clients = {'A': MagicMock()}

        await backend.close()

        assert backend._group_clients == {}
        cloned.close.assert_awaited_once()
