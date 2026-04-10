"""Shared test fixtures and test helper utilities."""

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Make this file directly importable by test modules via `from conftest import ...`.
# Needed because the tests/ directory has __init__.py (package layout), so pytest
# adds fused-memory/ (the parent) to sys.path rather than tests/ itself.
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from fused_memory.backends.graphiti_client import GraphitiBackend  # noqa: E402
from fused_memory.config.schema import (  # noqa: E402
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
        backend._driver = MagicMock()
        return backend

    return _factory


@pytest.fixture
def make_graph_mock():
    """Factory fixture: returns a callable(rows, *, ro_rows, q_rows) -> MagicMock graph.

    The returned mock has both .query and .ro_query as AsyncMocks.

    Basic usage (backward-compatible): both .query and .ro_query return the same
    result whose .result_set is *rows*::

        graph = make_graph_mock([['uuid-1', 'label']])

    Split usage: supply *ro_rows* and/or *q_rows* to give each path a distinct
    result_set.  This is useful when ro_query and query must return different data
    (e.g. delete_entity_node: pre-check returns a row, DETACH DELETE returns [])::

        graph = make_graph_mock(ro_rows=[['NodeName', 'summary']], q_rows=[])

    The returned graph mock can be wired up as::

        backend._driver._get_graph = MagicMock(return_value=graph)
    """
    def _factory(
        rows: list[list] | None = None,
        *,
        ro_rows: list[list] | None = None,
        q_rows: list[list] | None = None,
    ) -> MagicMock:
        if ro_rows is not None or q_rows is not None:
            # Split mode: create separate result objects for each path.
            ro_result = MagicMock()
            ro_result.result_set = ro_rows if ro_rows is not None else (rows or [])
            q_result = MagicMock()
            q_result.result_set = q_rows if q_rows is not None else (rows or [])
        else:
            # Shared mode (backward-compatible): both paths use the same result.
            ro_result = MagicMock()
            ro_result.result_set = rows if rows is not None else []
            q_result = ro_result

        graph_mock = MagicMock()
        graph_mock.query = AsyncMock(return_value=q_result)
        graph_mock.ro_query = AsyncMock(return_value=ro_result)
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


# ---------------------------------------------------------------------------
# call_args extraction helpers (task-435)
#
# Graph query methods (graph.query / graph.ro_query) may be called with the
# Cypher string and params dict either positionally or as keyword arguments.
# These two helpers extract the relevant value from a mock call_args object
# regardless of calling convention, eliminating fragile bare args[N] accesses
# that throw opaque IndexError when the implementation switches to keyword-passing.
#
# Usage in tests:
#   call_args = graph.ro_query.call_args
#   cypher = extract_cypher(call_args)   # str
#   params = extract_params(call_args)   # dict
# ---------------------------------------------------------------------------


async def assert_ro_query_only(
    backend,
    make_graph_mock_fn,
    rows: list[list],
    method_name: str,
    *args,
    **kwargs,
) -> MagicMock:
    """Assert that a backend method uses ro_query and never calls query.

    Creates a graph mock via *make_graph_mock_fn*, wires it into
    *backend._driver._get_graph*, invokes the named method, then asserts:
      - graph.ro_query was awaited exactly once
      - graph.query was not awaited at all

    Returns the graph mock so callers can add additional assertions (e.g.
    inspecting Cypher content via graph.ro_query.call_args).

    Usage::

        graph = await assert_ro_query_only(
            backend, make_graph_mock, [['Node', 'Summary']],
            'get_node_text', 'uuid-1', group_id='test',
        )
    """
    graph = make_graph_mock_fn(rows)
    backend._driver._get_graph = MagicMock(return_value=graph)
    await getattr(backend, method_name)(*args, **kwargs)
    graph.ro_query.assert_awaited_once()
    graph.query.assert_not_awaited()
    return graph


def extract_cypher(call_args: Any) -> str:
    """Return the Cypher query string from a mock call_args object.

    Checks positional args[0] first, then falls back to the 'query' keyword
    argument. Returns '' if neither is present.
    """
    if call_args.args:
        return call_args.args[0]
    return call_args.kwargs.get('query', '')


def extract_params(call_args: Any) -> dict:
    """Return the Cypher params dict from a mock call_args object.

    Checks positional args[1] first, then falls back to the 'params' keyword
    argument. Returns {} if neither is present.
    """
    if len(call_args.args) > 1:
        return call_args.args[1]
    return call_args.kwargs.get('params', {})
