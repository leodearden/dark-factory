"""Shared test fixtures."""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.config.schema import (
    EmbedderConfig,
    EmbedderProvidersConfig,
    FusedMemoryConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
    QueueConfig,
    RoutingConfig,
)


@dataclass
class MockNode:
    """Simulates a Graphiti entity node (source/target of an edge)."""

    name: str
    uuid: str = ''
    labels: list[str] = field(default_factory=list)


@dataclass
class MockEdge:
    """Simulates a Graphiti entity edge returned from add_episode or search."""

    fact: str
    uuid: str = ''
    source_node: MockNode | None = None
    target_node: MockNode | None = None
    source_node_uuid: str = ''
    target_node_uuid: str = ''
    episodes: list[str] = field(default_factory=list)
    valid_at: Any = None
    invalid_at: Any = None


@dataclass
class MockAddEpisodeResult:
    """Simulates the AddEpisodeResults returned by Graphiti's add_episode.

    The real AddEpisodeResults class uses 'edges' as the field name.
    We keep 'entity_edges' for backward compat with existing tests that
    construct MockAddEpisodeResult(entity_edges=[...]).
    """

    entity_edges: list[MockEdge] = field(default_factory=list)
    edges: list[MockEdge] = field(default_factory=list)

    def __post_init__(self) -> None:
        # If only entity_edges was provided, mirror it to edges so both fields
        # are populated.  This preserves backward compat while ensuring the
        # 'edges' field (used by the real AddEpisodeResults) is accessible.
        if self.edges == [] and self.entity_edges:
            self.edges = list(self.entity_edges)


@pytest.fixture(autouse=True)
def preserve_config_path():
    """Save and restore os.environ['CONFIG_PATH'] around every test.

    This is a safety net: if a test (or the code under test) modifies CONFIG_PATH,
    it won't leak into subsequent tests.  The fixture is autouse so all tests in this
    package are covered without needing to request it explicitly.
    """
    original = os.environ.get('CONFIG_PATH')
    yield
    if original is None:
        os.environ.pop('CONFIG_PATH', None)
    else:
        os.environ['CONFIG_PATH'] = original


@pytest.fixture
def standard_mock_config() -> MagicMock:
    """MagicMock config pre-configured with common 1536-dim embedder attributes.

    Used by run_* entrypoint tests (TestRunReindex, TestRunCleanup, etc.) that
    need a config mock but don't want to construct a full FusedMemoryConfig.
    Tests needing non-default values (e.g., 768-dim) can override in-place:

        def test_something(self, standard_mock_config):
            standard_mock_config.embedder.dimensions = 768
    """
    cfg = MagicMock()
    cfg.embedder.dimensions = 1536
    cfg.embedder.providers.openai = None
    cfg.embedder.model = 'text-embedding-3-small'
    return cfg


@pytest.fixture
def make_backend():
    """Factory fixture: returns a callable(config) -> GraphitiBackend with mock client.

    Usage::

        def test_foo(self, mock_config, make_backend):
            backend = make_backend(mock_config)
            backend.client.some_method.return_value = ...
    """
    def _factory(config) -> GraphitiBackend:
        backend = GraphitiBackend(config)
        backend.client = MagicMock()
        return backend

    return _factory


@pytest.fixture
def make_graph_mock():
    """Factory fixture: returns a callable(rows) -> MagicMock graph with AsyncMock queries.

    The returned mock has both .query and .ro_query as AsyncMocks returning a result
    object whose .result_set is *rows*.  This is a superset of the helpers in
    test_reindex.py and test_cleanup_stale_edges.py.

    Usage::

        def test_foo(self, make_graph_mock):
            graph = make_graph_mock([['uuid-1', 'label']])
            cast_target._get_graph.return_value = graph
    """
    def _factory(rows: list[list]) -> MagicMock:
        result = MagicMock()
        result.result_set = rows
        graph_mock = MagicMock()
        graph_mock.query = AsyncMock(return_value=result)
        graph_mock.ro_query = AsyncMock(return_value=result)
        return graph_mock

    return _factory


@pytest.fixture
def make_fake_maintenance_service():
    """Factory fixture: returns a callable(mock_cfg, mock_service) -> async context manager.

    The returned async context manager yields (mock_cfg, mock_service) and is
    suitable for use as the side_effect of a patched maintenance_service.

    Usage::

        def test_foo(self, make_fake_maintenance_service):
            mock_cfg = MagicMock()
            mock_service = AsyncMock()
            with patch(
                'fused_memory.maintenance.reindex.maintenance_service',
                side_effect=make_fake_maintenance_service(mock_cfg, mock_service),
            ):
                ...
    """
    def _factory(mock_cfg, mock_service):
        @asynccontextmanager
        async def fake(config_path):
            yield mock_cfg, mock_service

        return fake

    return _factory


@pytest.fixture
def mock_config(tmp_path) -> FusedMemoryConfig:
    """A FusedMemoryConfig that doesn't require real API keys or services."""
    return FusedMemoryConfig(
        llm=LLMConfig(
            provider='openai',
            model='gpt-4o-mini',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key'),
            ),
        ),
        embedder=EmbedderConfig(
            provider='openai',
            model='text-embedding-3-small',
            providers=EmbedderProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key'),
            ),
        ),
        routing=RoutingConfig(
            use_heuristics=True,
            llm_fallback=False,
            confidence_threshold=0.7,
        ),
        queue=QueueConfig(
            semaphore_limit=5,
            workers_per_group=2,
            max_attempts=3,
            retry_base_seconds=0.05,
            write_timeout_seconds=2.0,
            data_dir=str(tmp_path / 'queue'),
        ),
    )
