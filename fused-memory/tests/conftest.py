"""pytest fixtures for fused-memory tests.

Non-fixture helpers (MockEdge, make_rebuild_detail, extract_cypher, …)
live in `_fm_helpers.py` — a uniquely-named sibling module — so they can
be imported from test files without conflicting with sibling subprojects'
conftests under `sys.modules['conftest']`.
"""

import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

# Make this file's directory importable by test modules so
# `from _fm_helpers import ...` resolves regardless of whether pytest
# is invoked from the subproject root or the workspace root.
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Make the sibling 'escalation' workspace package importable without installing it.
# curator_escalator.py uses a try/except guard (HAS_ESCALATION) — adding the src
# path here (before test files are collected) ensures the guard resolves to True so
# tests that exercise the escalation-routing branch can actually run.
_escalation_src = os.path.join(
    os.path.dirname(os.path.dirname(_tests_dir)),  # workspace root
    'escalation', 'src',
)
if _escalation_src not in sys.path:
    sys.path.insert(0, _escalation_src)

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
    """Factory fixture: returns a callable(config) -> GraphitiBackend with mock client."""
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
    """
    def _factory(
        rows: list[list] | None = None,
        *,
        ro_rows: list[list] | None = None,
        q_rows: list[list] | None = None,
    ) -> MagicMock:
        if ro_rows is not None or q_rows is not None:
            ro_result = MagicMock()
            ro_result.result_set = ro_rows if ro_rows is not None else (rows or [])
            q_result = MagicMock()
            q_result.result_set = q_rows if q_rows is not None else (rows or [])
        else:
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
    """Factory fixture: returns a callable(mock_cfg, mock_service) -> async context manager."""
    def _factory(mock_cfg, mock_service):
        @asynccontextmanager
        async def fake(config_path):
            yield mock_cfg, mock_service

        return fake

    return _factory


@pytest.fixture
def make_edge_backend():
    """Factory fixture: returns a callable(backend, *, nodes, edges) -> backend."""
    def _factory(backend, *, nodes, edges):
        backend.list_entity_nodes = AsyncMock(return_value=nodes)
        backend.get_all_valid_edges = AsyncMock(return_value=edges)
        return backend

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
