"""Tests that validate shared conftest fixtures added for task-147 DRY refactor.

These tests are written BEFORE the fixtures exist (TDD step-1) and will fail
until conftest.py is extended in step-2.
"""
from __future__ import annotations

import inspect
import os
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from _fm_helpers import extract_cypher, extract_params, make_rebuild_detail

# ---------------------------------------------------------------------------
# preserve_config_path fixture tests
# ---------------------------------------------------------------------------

class TestPreserveConfigPath:
    """preserve_config_path autouse fixture saves/restores CONFIG_PATH around each test."""

    def test_absent_key_is_absent(self, preserve_config_path):
        """When CONFIG_PATH is not set, the fixture doesn't interfere."""
        # Remove CONFIG_PATH if present so we start clean
        os.environ.pop('CONFIG_PATH', None)
        assert os.environ.get('CONFIG_PATH') is None

    def test_can_set_config_path_during_test(self, preserve_config_path):
        """Setting CONFIG_PATH during a test is visible within the test."""
        os.environ['CONFIG_PATH'] = '/tmp/inside_test.yaml'
        assert os.environ['CONFIG_PATH'] == '/tmp/inside_test.yaml'
        # Cleanup is the fixture's responsibility; we just verify it's set here

    def test_fixture_accepts_pre_set_value(self, preserve_config_path):
        """The fixture can be requested explicitly even when CONFIG_PATH was set before."""
        os.environ['CONFIG_PATH'] = '/tmp/pre_set.yaml'
        # Fixture should save this value on entry; test can see it
        assert os.environ['CONFIG_PATH'] == '/tmp/pre_set.yaml'

    def test_is_autouse_so_no_explicit_request_needed(self):
        """preserve_config_path is autouse; tests don't need to request it by name.

        This test requests no fixture by name but still passes when autouse is active.
        If the fixture is broken (e.g., raises on setup) this test will fail.
        """
        # No CONFIG_PATH interaction; just confirms autouse doesn't break normal tests
        assert True


# ---------------------------------------------------------------------------
# standard_mock_config fixture tests
# ---------------------------------------------------------------------------

class TestStandardMockConfig:
    """standard_mock_config returns a MagicMock with the common 1536-dim embedder attrs."""

    def test_embedder_dimensions_is_1536(self, standard_mock_config):
        assert standard_mock_config.embedder.dimensions == 1536

    def test_embedder_providers_openai_is_none(self, standard_mock_config):
        assert standard_mock_config.embedder.providers.openai is None

    def test_embedder_model_is_text_embedding_3_small(self, standard_mock_config):
        assert standard_mock_config.embedder.model == 'text-embedding-3-small'

    def test_can_override_dimensions(self, standard_mock_config):
        """MagicMock supports attribute assignment for 768-dim test variants."""
        standard_mock_config.embedder.dimensions = 768
        assert standard_mock_config.embedder.dimensions == 768

    def test_can_override_model(self, standard_mock_config):
        """MagicMock supports attribute assignment for alternate model names."""
        standard_mock_config.embedder.model = 'text-embedding-ada-002'
        assert standard_mock_config.embedder.model == 'text-embedding-ada-002'


# ---------------------------------------------------------------------------
# extract_cypher helper tests (task-435 step-1)
# ---------------------------------------------------------------------------

class TestExtractCypher:
    """extract_cypher(call_args) extracts the Cypher query string from a mock call_args.

    Handles both positional (args[0]) and keyword ('query' kwarg) calling conventions
    so tests remain robust if the implementation switches between the two.
    """

    def test_positional_args_returns_first_arg(self):
        """When query is passed positionally, returns args[0]."""
        call_args = call('MATCH (n) RETURN n', {})
        assert extract_cypher(call_args) == 'MATCH (n) RETURN n'

    def test_keyword_query_returns_kwarg(self):
        """When query is passed as keyword argument, returns kwargs['query']."""
        call_args = call(query='MATCH (n) RETURN n')
        assert extract_cypher(call_args) == 'MATCH (n) RETURN n'

    def test_keyword_query_returns_query_among_other_kwargs(self):
        """When 'query' kwarg is passed alongside other kwargs, returns the 'query' value specifically, not an arbitrary kwarg value."""
        call_args = call(query='MATCH (n) RETURN n', params={'uuid': 'x'})
        assert extract_cypher(call_args) == 'MATCH (n) RETURN n'

    def test_keyword_query_returns_query_regardless_of_insertion_order(self):
        """When 'query' kwarg is passed after other kwargs, still returns 'query' value.

        Regression guard: proves the helper looks up by key name, not dict iteration order.
        (params= inserted first, query= inserted second — reversed from test_keyword_query_returns_query_among_other_kwargs)
        """
        call_args = call(params={'uuid': 'x'}, query='MATCH (n) RETURN n')
        assert extract_cypher(call_args) == 'MATCH (n) RETURN n'

    def test_empty_call_returns_empty_string(self):
        """When neither positional nor keyword query is present, returns empty string."""
        call_args = call()
        assert extract_cypher(call_args) == ''


# ---------------------------------------------------------------------------
# extract_params helper tests (task-435 step-2)
# ---------------------------------------------------------------------------

class TestExtractParams:
    """extract_params(call_args) extracts the Cypher params dict from a mock call_args.

    Handles both positional (args[1]) and keyword ('params' kwarg) calling conventions
    so tests remain robust if the implementation switches between the two.
    """

    def test_positional_params_returns_second_arg(self):
        """When params is passed positionally as second arg, returns args[1]."""
        call_args = call('MATCH (n) RETURN n', {'uuid': 'x'})
        assert extract_params(call_args) == {'uuid': 'x'}

    def test_keyword_params_returns_kwarg(self):
        """When params is passed as keyword argument, returns kwargs['params']."""
        call_args = call('MATCH (n) RETURN n', params={'uuid': 'x'})
        assert extract_params(call_args) == {'uuid': 'x'}

    def test_no_params_returns_empty_dict(self):
        """When no params argument is present, returns empty dict."""
        call_args = call('MATCH (n) RETURN n')
        assert extract_params(call_args) == {}


# ---------------------------------------------------------------------------
# make_edge_backend factory fixture tests (task-445 step-1)
# ---------------------------------------------------------------------------

class TestMakeEdgeBackend:
    """make_edge_backend(backend, *, nodes, edges) wires the two-mock surface.

    Canonical usage::

        backend = make_edge_backend(make_backend(mock_config), nodes=[...], edges={...})
    """

    def test_returns_same_backend_object(self, make_edge_backend):
        """make_edge_backend returns the same backend object that was passed in."""
        backend = MagicMock()
        result = make_edge_backend(backend, nodes=[], edges={})
        assert result is backend

    def test_list_entity_nodes_is_async_mock(self, make_edge_backend):
        """backend.list_entity_nodes is replaced with an AsyncMock."""
        backend = MagicMock()
        make_edge_backend(backend, nodes=[], edges={})
        assert isinstance(backend.list_entity_nodes, AsyncMock)

    def test_get_all_valid_edges_is_async_mock(self, make_edge_backend):
        """backend.get_all_valid_edges is replaced with an AsyncMock."""
        backend = MagicMock()
        make_edge_backend(backend, nodes=[], edges={})
        assert isinstance(backend.get_all_valid_edges, AsyncMock)

    def test_list_entity_nodes_return_value_equals_nodes_arg(self, make_edge_backend):
        """list_entity_nodes.return_value equals the nodes kwarg passed to the factory."""
        nodes = [{'uuid': 'u1', 'name': 'Alice', 'summary': 'fact'}]
        backend = MagicMock()
        make_edge_backend(backend, nodes=nodes, edges={})
        assert backend.list_entity_nodes.return_value == nodes

    def test_get_all_valid_edges_return_value_equals_edges_arg(self, make_edge_backend):
        """get_all_valid_edges.return_value equals the edges kwarg passed to the factory."""
        edges = {'u1': [{'uuid': 'e1', 'fact': 'fact', 'name': 'rel'}]}
        backend = MagicMock()
        make_edge_backend(backend, nodes=[], edges=edges)
        assert backend.get_all_valid_edges.return_value == edges

    def test_nodes_is_keyword_only(self, make_edge_backend):
        """Calling with nodes as a positional argument raises TypeError."""
        backend = MagicMock()
        with pytest.raises(TypeError):
            make_edge_backend(backend, [], {})  # type: ignore[call-arg]

    def test_edges_is_keyword_only(self, make_edge_backend):
        """edges parameter has kind KEYWORD_ONLY in the factory signature."""
        sig = inspect.signature(make_edge_backend)
        assert sig.parameters['edges'].kind == inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.asyncio
    async def test_awaiting_list_entity_nodes_yields_nodes(self, make_edge_backend):
        """Awaiting backend.list_entity_nodes() returns the nodes list."""
        nodes = [{'uuid': 'u1', 'name': 'Alice', 'summary': 'fact'}]
        backend = MagicMock()
        make_edge_backend(backend, nodes=nodes, edges={})
        result = await backend.list_entity_nodes()
        assert result == nodes

    @pytest.mark.asyncio
    async def test_awaiting_get_all_valid_edges_yields_edges(self, make_edge_backend):
        """Awaiting backend.get_all_valid_edges() returns the edges dict."""
        edges = {'u1': [{'uuid': 'e1', 'fact': 'fact', 'name': 'rel'}]}
        backend = MagicMock()
        make_edge_backend(backend, nodes=[], edges=edges)
        result = await backend.get_all_valid_edges()
        assert result == edges


# ---------------------------------------------------------------------------
# make_rebuild_detail factory tests (task-505 step-1)
# ---------------------------------------------------------------------------

class TestMakeRebuildDetail:
    """make_rebuild_detail(uuid, name, *, ...) builds a rebuild-detail dict.

    The factory produces the canonical 6-key dict consumed by rebuild pipeline
    code and test assertions. uuid and name are positional; all other parameters
    are keyword-only with sensible defaults.
    """

    def test_defaults(self):
        """Positional args are set, all keyword defaults are correct, result is a plain dict."""
        result = make_rebuild_detail('u1', 'Alice')
        assert type(result) is dict
        assert result == {
            'uuid': 'u1',
            'name': 'Alice',
            'old_summary': '',
            'new_summary': '',
            'edge_count': 0,
            'status': 'rebuilt',
        }

    def test_all_kwargs_set_together(self):
        """All keyword overrides applied simultaneously produce the correct dict."""
        result = make_rebuild_detail(
            'node-1', 'Bob',
            old_summary='old B', new_summary='rebuilt B',
            edge_count=3, status='skipped',
        )
        assert result == {
            'uuid': 'node-1',
            'name': 'Bob',
            'old_summary': 'old B',
            'new_summary': 'rebuilt B',
            'edge_count': 3,
            'status': 'skipped',
        }

    def test_error_param_included_when_provided(self):
        """Passing error=None adds an 'error' key; omitting it keeps the dict to 6 keys."""
        without_error = make_rebuild_detail('u1', 'Alice')
        assert 'error' not in without_error

        with_error = make_rebuild_detail('u1', 'Alice', error=None)
        assert with_error['error'] is None
        assert set(with_error.keys()) == {'uuid', 'name', 'old_summary', 'new_summary', 'edge_count', 'status', 'error'}

    def test_keyword_args_are_keyword_only(self):
        """old_summary, new_summary, edge_count, status, error have kind KEYWORD_ONLY in signature."""
        sig = inspect.signature(make_rebuild_detail)
        for param_name in ('old_summary', 'new_summary', 'edge_count', 'status', 'error'):
            assert sig.parameters[param_name].kind == inspect.Parameter.KEYWORD_ONLY, (
                f'{param_name} should be KEYWORD_ONLY'
            )


# ---------------------------------------------------------------------------
# make_graph_mock: cypher-aware dispatch (task 4340)
# ---------------------------------------------------------------------------


class TestMakeGraphMockCypherDispatch:
    """make_graph_mock must dispatch on the cypher, not answer everything alike.

    Task 4340 paginated two whole-graph reads. A paginated read issues a
    single-row ``count(*)`` census probe and then a series of
    ``SKIP n LIMIT m`` pages. A fixture that answers EVERY query with the same
    rows would hand the census probe a page of edge rows, whose first column
    is a uuid string — ``int('node-1')`` is unusable as a count — so every
    existing caller would silently flip to ``complete=False`` plus a WARNING.

    Patching around that per-test would leave a shared fixture that actively
    lies about the read shape it stands in for, and the next person to
    paginate something would rediscover the same trap.
    """

    @pytest.mark.asyncio
    async def test_census_query_answers_with_a_single_row_count(self, make_graph_mock):
        """A ``count(`` query gets ONE row holding the row count, never page rows."""
        rows = [['node-1', 'e1', 'f', 'n'], ['node-2', 'e2', 'f', 'n']]
        graph = make_graph_mock(rows)
        result = await graph.ro_query('MATCH (n:Entity) RETURN count(*)')
        assert result.result_set == [[2]]
        assert len(result.result_set) == 1
        # The census and the pages agree BY CONSTRUCTION.
        assert result.result_set[0][0] == len(rows)

    @pytest.mark.asyncio
    async def test_a_count_column_among_others_is_not_a_census(self, make_graph_mock):
        """The census dispatch is NARROW: only a bare `RETURN count(*)` projection.

        A loose `'count(' in cypher` test also captures ordinary queries that
        return a count as one column among several — find_duplicate_entity_nodes
        issues `RETURN n.uuid, ..., count(e)` — and handing those a
        single-column [[n]] row raises IndexError deep inside the method under
        test, nowhere near the fixture.
        """
        rows = [['dup-uuid-1', 200, 2]]
        graph = make_graph_mock(rows)
        result = await graph.ro_query(
            'MATCH (n:Entity {name: $name})-[e:RELATES_TO]-() '
            'RETURN n.uuid, n.created_at, count(e) AS edge_count'
        )
        assert result.result_set == rows

    @pytest.mark.asyncio
    async def test_paged_query_answers_with_the_requested_slice(self, make_graph_mock):
        rows = [[f'n{i}'] for i in range(10)]
        graph = make_graph_mock(rows)
        first = await graph.ro_query('MATCH (n) RETURN n.uuid SKIP 0 LIMIT 4')
        second = await graph.ro_query('MATCH (n) RETURN n.uuid SKIP 4 LIMIT 4')
        assert first.result_set == rows[0:4]
        assert second.result_set == rows[4:8]

    @pytest.mark.asyncio
    async def test_skip_past_the_end_yields_no_rows(self, make_graph_mock):
        """The boundary that terminates a page loop."""
        rows = [[f'n{i}'] for i in range(10)]
        graph = make_graph_mock(rows)
        result = await graph.ro_query('MATCH (n) RETURN n.uuid SKIP 40 LIMIT 4')
        assert result.result_set == []

    @pytest.mark.asyncio
    async def test_plain_query_still_answers_with_every_row(self, make_graph_mock):
        """BACK-COMPAT: a non-paged, non-census query behaves exactly as before."""
        rows = [['node-1', 'e1', 'f', 'n'], ['node-2', 'e2', 'f', 'n']]
        graph = make_graph_mock(rows)
        result = await graph.ro_query('MATCH (n:Entity)-[e:RELATES_TO]-() RETURN n.uuid')
        assert result.result_set == rows

    @pytest.mark.asyncio
    async def test_resultset_cap_reproduces_the_server_truncation(self, make_graph_mock):
        """``resultset_cap`` is opt-in and truncates EVERY result set, silently."""
        rows = [[f'n{i}'] for i in range(25)]
        graph = make_graph_mock(rows, resultset_cap=10)
        unpaginated = await graph.ro_query('MATCH (n) RETURN n.uuid')
        assert len(unpaginated.result_set) == 10

        collected: list[list] = []
        for skip in range(0, 30, 5):
            page = await graph.ro_query(f'MATCH (n) RETURN n.uuid SKIP {skip} LIMIT 5')
            collected.extend(page.result_set)
        assert len(collected) == 25

    @pytest.mark.asyncio
    async def test_resultset_cap_defaults_to_no_truncation(self, make_graph_mock):
        """Opt-in: without the kwarg, nothing is capped."""
        rows = [[f'n{i}'] for i in range(25)]
        graph = make_graph_mock(rows)
        result = await graph.ro_query('MATCH (n) RETURN n.uuid')
        assert len(result.result_set) == 25

    @pytest.mark.asyncio
    async def test_ro_rows_and_q_rows_still_split_the_two_paths(self, make_graph_mock):
        """BACK-COMPAT: .query stays a separate AsyncMock with its own rows."""
        graph = make_graph_mock(ro_rows=[['ro']], q_rows=[['q']])
        assert isinstance(graph.ro_query, AsyncMock)
        assert isinstance(graph.query, AsyncMock)
        assert (await graph.ro_query('MATCH (n) RETURN n')).result_set == [['ro']]
        assert (await graph.query('MATCH (n) RETURN n')).result_set == [['q']]

    @pytest.mark.asyncio
    async def test_header_still_applies_to_every_dispatch_branch(self, make_graph_mock):
        """BACK-COMPAT: by-name column resolution keeps working on every branch."""
        header = [[1, 'label'], [1, 'properties']]
        graph = make_graph_mock([['a', 'b']], header=header)
        for cypher in (
            'CALL db.indexes()',
            'MATCH (n) RETURN count(*)',
            'MATCH (n) RETURN n SKIP 0 LIMIT 5',
        ):
            assert (await graph.ro_query(cypher)).header == header

    @pytest.mark.asyncio
    async def test_default_rows_is_empty(self, make_graph_mock):
        """BACK-COMPAT: make_graph_mock() with no args answers with no rows."""
        graph = make_graph_mock()
        assert (await graph.ro_query('MATCH (n) RETURN n')).result_set == []
        assert (await graph.ro_query('MATCH (n) RETURN count(*)')).result_set == [[0]]
