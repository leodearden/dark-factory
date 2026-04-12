"""Tests for code-quality improvements to graphiti_client.py rebuild/refresh pipeline.

Task 433: 8 code-quality improvements deferred from task-419 review.
- StaleSummaryResult NamedTuple (steps 1-2)
- _canonical_facts() @staticmethod (steps 3-4)
- refresh_entity_summary optional name/old_summary params (steps 5-6)
- rebuild_entity_summaries force+dry_run edge-fetch skip (steps 7-8)
- rebuild_entity_summaries variable scoping / data-flow clarity (steps 9-10)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.backends.graphiti_client import (
    EdgeDict,
    GraphitiBackend,
    StaleSummaryResult,
)

# ---------------------------------------------------------------------------
# task-507: make_stale_list factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def make_stale_list():
    """Factory fixture for the standard Alice/Bob two-entity stale-list shape.

    Follows the factory-fixture convention used throughout this project
    (make_backend, make_graph_mock, …): returns a callable that produces the
    test object, allowing keyword overrides at call time.

    Entity conventions:
      - Alice: uuid='u1', name='Alice'
      - Bob:   uuid='u2', name='Bob'

    Summary defaults ('summary A' / 'summary B') match the most common values
    used across the six consumer tests.  Pass keyword arguments to override:

        make_stale_list(alice_summary='old A', bob_summary='old B')
    """

    def _factory(alice_summary: str = 'summary A', bob_summary: str = 'summary B'):
        return [
            {'uuid': 'u1', 'name': 'Alice', 'summary': alice_summary},
            {'uuid': 'u2', 'name': 'Bob', 'summary': bob_summary},
        ]

    return _factory


# ---------------------------------------------------------------------------
# step-1: StaleSummaryResult named tuple with backward-compat tuple unpacking
# ---------------------------------------------------------------------------


class TestStaleSummaryResult:
    """StaleSummaryResult has named attrs and supports 3-tuple unpacking."""

    def test_named_attribute_stale(self):
        """StaleSummaryResult.stale holds the stale list."""
        stale_list = [{'uuid': 'u1', 'name': 'Alice'}]
        edges: dict[str, list[EdgeDict]] = {'u1': [{'uuid': 'e-1', 'fact': 'fact1', 'name': 'knows'}]}
        result = StaleSummaryResult(stale=stale_list, all_edges=edges, total_count=5)
        assert result.stale is stale_list

    def test_named_attribute_all_edges(self):
        """StaleSummaryResult.all_edges holds the edges dict."""
        stale_list: list[dict] = []
        edges: dict[str, list[EdgeDict]] = {'u1': [{'uuid': 'e-1', 'fact': 'fact1', 'name': 'knows'}]}
        result = StaleSummaryResult(stale=stale_list, all_edges=edges, total_count=3)
        assert result.all_edges is edges

    def test_named_attribute_total_count(self):
        """StaleSummaryResult.total_count holds the total entity count."""
        result = StaleSummaryResult(stale=[], all_edges={}, total_count=42)
        assert result.total_count == 42

    def test_tuple_unpacking_backward_compat(self):
        """StaleSummaryResult supports 3-tuple unpacking (backward compat)."""
        stale_list = [{'uuid': 'u1'}]
        edges = {'u1': []}
        result = StaleSummaryResult(stale=stale_list, all_edges=edges, total_count=7)
        a, b, c = result
        assert a is stale_list
        assert b is edges
        assert c == 7

    def test_is_tuple_subclass(self):
        """StaleSummaryResult compares value-equal to a plain 3-tuple (backward-compat promise)."""
        stale_list = [{'uuid': 'u1'}]
        edges: dict = {}
        result = StaleSummaryResult(stale=stale_list, all_edges=edges, total_count=1)
        # Value-equality with a plain tuple proves the NamedTuple backward-compat promise:
        # only actual tuple subclasses compare equal to plain tuples this way.
        assert result == (stale_list, edges, 1)

    def test_no_legacy_edges_attribute(self):
        """StaleSummaryResult must NOT expose the old 'edges' field name.

        StaleSummaryResult uses all_edges (not edges) as the field name; this test
        prevents silent re-aliasing.
        """
        result = StaleSummaryResult(stale=[], all_edges={}, total_count=0)
        assert not hasattr(result, 'edges')

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_returns_named_result(self, mock_config, make_backend):
        """_detect_stale_summaries_with_edges returns StaleSummaryResult with named access."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(
            return_value=[
                {'uuid': 'u1', 'name': 'Alice', 'summary': 'stale summary'},
            ]
        )
        backend.get_all_valid_edges = AsyncMock(
            return_value={
                'u1': [{'fact': 'fresh fact'}],
            }
        )
        result = await backend._detect_stale_summaries_with_edges(group_id='test')

        # Named access
        assert isinstance(result, StaleSummaryResult)
        assert result.total_count == 1
        assert isinstance(result.stale, list)
        assert isinstance(result.all_edges, dict)

        # Positional (backward compat)
        stale, edges, total = result
        assert total == 1
        assert isinstance(stale, list)
        assert isinstance(edges, dict)


# ---------------------------------------------------------------------------
# step-3: _canonical_facts() @staticmethod
# ---------------------------------------------------------------------------


class TestCanonicalFacts:
    """GraphitiBackend._canonical_facts deduplicates facts preserving order."""

    def test_deduplicates_preserving_insertion_order(self):
        """Duplicate facts are removed but original order is preserved."""
        edges = [
            {'fact': 'A knows B'},
            {'fact': 'B lives in London'},
            {'fact': 'A knows B'},  # duplicate
            {'fact': 'C works at Acme'},
        ]
        result = GraphitiBackend._canonical_facts(edges)
        assert result == ['A knows B', 'B lives in London', 'C works at Acme']

    def test_empty_edge_list_returns_empty_list(self):
        """Empty input returns an empty list."""
        assert GraphitiBackend._canonical_facts([]) == []

    def test_edges_missing_fact_key_are_skipped(self):
        """Edges without 'fact' key are silently skipped."""
        edges = [
            {'name': 'edge1'},  # no 'fact' key
            {'fact': 'A knows B'},
            {'uuid': 'some-uuid'},  # no 'fact' key
        ]
        result = GraphitiBackend._canonical_facts(edges)
        assert result == ['A knows B']

    def test_edges_with_falsy_fact_value_are_skipped(self):
        """Edges with empty string fact value are skipped."""
        edges = [
            {'fact': ''},  # falsy
            {'fact': 'A knows B'},
            {'fact': None},  # falsy
        ]
        result = GraphitiBackend._canonical_facts(edges)
        assert result == ['A knows B']

    def test_returns_list_not_set(self):
        """Return type is list, not set (order matters)."""
        edges = [{'fact': 'A'}, {'fact': 'B'}]
        result = GraphitiBackend._canonical_facts(edges)
        assert isinstance(result, list)
        assert result == ['A', 'B']

    def test_whitespace_only_fact_is_filtered(self):
        """Whitespace-only facts are filtered out, not included in results.

        The filter uses ``if e.get('fact', '').strip()`` so that a string like
        '   ' strips to '' (falsy) and is excluded.  Only facts with real
        non-whitespace content pass through.
        """
        edges = [
            {'fact': '   '},  # whitespace-only — filtered out
            {'fact': 'A knows B'},
        ]
        result = GraphitiBackend._canonical_facts(edges)
        assert result == ['A knows B']

    def test_whitespace_variants_all_filtered(self):
        """All whitespace-only variants are filtered; content with surrounding whitespace is kept.

        Tabs, newlines, mixed whitespace, and single spaces are all falsy after
        .strip() and must be excluded.  A fact with real content but leading/
        trailing whitespace (e.g. '  hello  ') is truthy after strip and must
        be preserved with its original value.
        """
        edges = [
            {'fact': '\t\t'},  # tabs only — filtered
            {'fact': '\n'},  # newline only — filtered
            {'fact': '  \t\n  '},  # mixed whitespace — filtered
            {'fact': ' '},  # single space — filtered
            {'fact': '  hello  '},  # real content with surrounding space — kept (raw value)
            {'fact': 'A knows B'},  # plain fact — kept
        ]
        result = GraphitiBackend._canonical_facts(edges)
        assert result == ['  hello  ', 'A knows B']

    def test_all_whitespace_edges_returns_empty_list(self):
        """All-whitespace input returns an empty list — the boundary case.

        All-whitespace input is the boundary case — if every edge is
        whitespace-only, _canonical_facts must return an empty list (no
        spurious empty string, no pre-strip value leakage).

        This is distinct from test_whitespace_only_fact_is_filtered (which
        mixes whitespace-only and valid edges) and from
        test_whitespace_variants_all_filtered (which also mixes).  This test
        covers the pure all-whitespace case where no valid fact is present,
        ensuring the result is [] rather than ['', '  ', etc.].
        """
        edges = [{'fact': '   '}, {'fact': '\t'}]
        result = GraphitiBackend._canonical_facts(edges)
        assert result == []


# ---------------------------------------------------------------------------
# step-5: refresh_entity_summary optional name/old_summary params
# ---------------------------------------------------------------------------


class TestRefreshEntitySummaryOptionalParams:
    """refresh_entity_summary optional name+old_summary params."""

    @pytest.mark.asyncio
    async def test_raises_if_name_without_old_summary(self, mock_config, make_backend):
        """ValueError raised when name is provided without old_summary."""
        backend = make_backend(mock_config)
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])
        backend.update_node_summary = AsyncMock()
        with pytest.raises(ValueError, match='both'):
            await backend.refresh_entity_summary('u1', group_id='test', name='Alice')

    @pytest.mark.asyncio
    async def test_raises_if_old_summary_without_name(self, mock_config, make_backend):
        """ValueError raised when old_summary is provided without name."""
        backend = make_backend(mock_config)
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])
        backend.update_node_summary = AsyncMock()
        with pytest.raises(ValueError, match='both'):
            await backend.refresh_entity_summary(
                'u1', group_id='test', old_summary='old summary text'
            )

    @pytest.mark.asyncio
    async def test_get_node_text_not_called_when_both_provided(self, mock_config, make_backend):
        """When both name and old_summary provided, get_node_text is NOT called."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('should-not-be-called', ''))
        backend.get_valid_edges_for_node = AsyncMock(
            return_value=[
                {'fact': 'Alice knows Bob'},
            ]
        )
        backend.update_node_summary = AsyncMock()

        result = await backend.refresh_entity_summary(
            'u1', group_id='test', name='Alice', old_summary='stale summary'
        )

        backend.get_node_text.assert_not_called()
        assert result['name'] == 'Alice'
        assert result['old_summary'] == 'stale summary'
        # Verify the summary actually written matches the joined canonical facts.
        # _canonical_facts([{'fact': 'Alice knows Bob'}]) == ['Alice knows Bob'],
        # joined with '\n' gives 'Alice knows Bob'.
        backend.update_node_summary.assert_awaited_once_with(
            'u1', 'Alice knows Bob', group_id='test'
        )

    @pytest.mark.asyncio
    async def test_get_node_text_called_when_neither_provided(self, mock_config, make_backend):
        """When neither name nor old_summary provided, get_node_text IS called (backward compat)."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'old summary'))
        backend.get_valid_edges_for_node = AsyncMock(
            return_value=[
                {'fact': 'Alice knows Bob'},
            ]
        )
        backend.update_node_summary = AsyncMock()

        result = await backend.refresh_entity_summary('u1', group_id='test')

        backend.get_node_text.assert_called_once()
        assert result['name'] == 'Alice'
        assert result['old_summary'] == 'old summary'


# ---------------------------------------------------------------------------
# step-7: rebuild_entity_summaries(force=True, dry_run=True) skips edge fetch
# ---------------------------------------------------------------------------


class TestRebuildEntitySummariesForceDryRun:
    """Pins the force=True, dry_run=True skip behaviour in rebuild_entity_summaries.

    Narrowed scope claim
    --------------------
    get_all_valid_edges is NOT awaited when rebuild_entity_summaries is called
    with force=True AND dry_run=True. This skip only applies to the force=True
    code path (the force branch); it does NOT apply to the force=False path.

    Two production-code guards in graphiti_client.rebuild_entity_summaries
    protect this behaviour (referenced by semantic role, not by line number):

    1. **edge-fetch guard in the force branch** — inside the ``if force:`` block,
       an inner ``if not dry_run:`` gate controls whether
       ``get_all_valid_edges(...)`` is awaited. When dry_run=True that gate is
       closed, so no edges are fetched at all.

    2. **dry_run early-return block before the semaphore loop** — after the
       edge-fetch guard, a subsequent ``if dry_run:`` block marks every target
       entity as ``'skipped_dry_run'`` and returns immediately, before the
       ``asyncio.Semaphore``-based rebuild loop executes. This is why
       ``_rebuild_entity_from_edges`` is also never awaited.

    force=False contrast (updated by task 526)
    ------------------------------------------
    The force=False path branches on ``dry_run`` at the call site in
    ``rebuild_entity_summaries``:

    - ``force=False, dry_run=True`` → calls ``_detect_stale_summaries_dry_run``,
      which fetches edges per-entity via ``get_valid_edges_for_node`` and does
      **NOT** call ``get_all_valid_edges``. This is the cheap-probe path added
      by task 526 to avoid materialising the O(E) edge dict when the result is
      never passed to ``_rebuild_entity_from_edges``.

    - ``force=False, dry_run=False`` → calls ``_detect_stale_summaries_with_edges``,
      which still issues a single bulk ``get_all_valid_edges`` query. That full
      edge map is needed because the actual rebuild loop (``_rebuild_entity_from_edges``)
      will consume it.

    Therefore the edge-fetch skip behaviour pinned by this class applies to both
    the force=True path and the force=False dry_run=True path (see also
    ``TestRebuildEntitySummariesDataFlow`` for tests specific to the force=False
    branching).
    """

    @pytest.mark.asyncio
    async def test_force_dry_run_does_not_call_get_all_valid_edges(
        self, mock_config, make_backend, make_stale_list
    ):
        """When force=True and dry_run=True, get_all_valid_edges is NOT called."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=make_stale_list())
        backend.get_all_valid_edges = AsyncMock(return_value={})
        backend._rebuild_entity_from_edges = AsyncMock()

        await backend.rebuild_entity_summaries(group_id='test', force=True, dry_run=True)

        backend.get_all_valid_edges.assert_not_awaited()
        backend._rebuild_entity_from_edges.assert_not_awaited()
        backend.list_entity_nodes.assert_awaited_once()

    async def _make_dry_run_result(self, mock_config, make_backend):
        """Shared setup for force=True dry_run=True tests.

        Returns (backend, entities, result) with a standard 3-entity fixture
        and all relevant methods mocked.  Extracted to avoid duplicating
        identical mock-wiring and production-call boilerplate across tests.
        """
        backend = make_backend(mock_config)
        entities = [
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'summary A'},
            {'uuid': 'u2', 'name': 'Bob', 'summary': 'summary B'},
            {'uuid': 'u3', 'name': 'Carol', 'summary': 'summary C'},
        ]
        backend.list_entity_nodes = AsyncMock(return_value=entities)
        backend.get_all_valid_edges = AsyncMock(return_value={})
        backend._rebuild_entity_from_edges = AsyncMock()
        result = await backend.rebuild_entity_summaries(group_id='test', force=True, dry_run=True)
        return backend, entities, result

    @pytest.mark.asyncio
    async def test_force_dry_run_returns_correct_aggregate(self, mock_config, make_backend):
        """When force=True and dry_run=True, result has correct structure."""
        backend, entities, result = await self._make_dry_run_result(mock_config, make_backend)

        assert result['total_entities'] == len(entities)
        assert result['stale_entities'] == result['total_entities']  # force=True treats every entity as stale
        assert result['skipped'] == len(entities)  # dry_run=True skips all
        assert result['rebuilt'] == 0
        assert result['errors'] == 0
        expected_details = [
            {'uuid': e['uuid'], 'name': e['name'], 'status': 'skipped_dry_run'} for e in entities
        ]
        assert len(result['details']) == len(entities)  # explicit length guard for clearer failure diagnostics
        # Schema pinning: assert the exact key set on [0] to catch accidental schema leaks (e.g. extra fields
        # added to the dry_run dict in graphiti_client.py).  All detail dicts are built by the same code path
        # so checking [0] is sufficient.
        assert set(result['details'][0].keys()) == {'uuid', 'name', 'status'}
        # Value + order check: projection tolerates nothing extra (guarded above) and verifies sequential order
        # from the 'for t in targets' loop.
        assert [{k: d[k] for k in ('uuid', 'name', 'status')} for d in result['details']] == expected_details
        assert result['errors'] + result['rebuilt'] + result['skipped'] == result['stale_entities']
        backend._rebuild_entity_from_edges.assert_not_awaited()
        backend.get_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_force_dry_run_detail_tolerates_extra_keys(self, mock_config, make_backend):
        """Pattern-regression test: projected-key assertion tolerates additive detail schema changes.

        This is a test-suite regression guard, not a production-behavior test. It
        manually injects an extra field into the returned detail dicts (simulating a
        future additive schema change) and verifies that:
          1. The projected-key assertion ({k: d[k] for k in keys}) correctly validates
             uuid/name/status while ignoring extra fields.
          2. The old exact-dict equality would fail with extra keys — documenting why
             the schema-tolerant projection pattern is needed.

        For a production-behavior test covering the full rebuild aggregate, see
        test_force_dry_run_returns_correct_aggregate.
        """
        _, entities, result = await self._make_dry_run_result(mock_config, make_backend)

        # Simulate an additive schema change: inject an extra field into each detail,
        # as if production had added 'group_id' to the skipped_dry_run record.
        for d in result['details']:
            d['group_id'] = 'extra_key_test'

        expected_details = [
            {'uuid': e['uuid'], 'name': e['name'], 'status': 'skipped_dry_run'} for e in entities
        ]

        # Confirm the extra key is present (simulation worked)
        assert all('group_id' in d for d in result['details'])

        # Old exact-dict comparison fails when extra keys are present
        with pytest.raises(AssertionError):
            assert result['details'] == expected_details

        # dry_run path appends details sequentially via 'for t in targets', so order is stable
        # Projected-key comparison tolerates extra keys while preserving order check
        assert [{k: d[k] for k in ('uuid', 'name', 'status')} for d in result['details']] == expected_details

    @pytest.mark.asyncio
    async def test_force_no_dry_run_calls_get_all_valid_edges(
        self, mock_config, make_backend, make_stale_list
    ):
        """Positive complement: force=True, dry_run=False calls get_all_valid_edges exactly once.

        This is the paired positive case for test_force_dry_run_does_not_call_get_all_valid_edges.
        When dry_run=False the edges ARE needed for the actual rebuild, so
        get_all_valid_edges must be called before processing entities.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=make_stale_list())
        backend.get_all_valid_edges = AsyncMock(return_value={})
        # Mock the inner rebuild to avoid touching real write path
        backend._rebuild_entity_from_edges = AsyncMock(
            return_value={
                'uuid': 'u1',
                'name': 'Alice',
                'old_summary': '',
                'new_summary': '',
                'edge_count': 0,
            }
        )

        await backend.rebuild_entity_summaries(group_id='test', force=True, dry_run=False)

        backend.get_all_valid_edges.assert_awaited_once_with(group_id='test')
        assert backend._rebuild_entity_from_edges.await_count == 2


# ---------------------------------------------------------------------------
# step-9: regression – rebuild_entity_summaries(force=False) data flow
# ---------------------------------------------------------------------------


class TestRebuildEntitySummariesDataFlow:
    """rebuild_entity_summaries(force=False) correctly flows total_entities from detect step."""

    @pytest.mark.asyncio
    async def test_total_entities_flows_from_detect_step(self, mock_config, make_backend):
        """total_entities in result matches _detect_stale_summaries_with_edges.total_count."""
        backend = make_backend(mock_config)
        stale_list = [{'uuid': 'u1', 'name': 'Alice', 'summary': 'old'}]
        all_edges: dict[str, list[EdgeDict]] = {
            'u1': [{'uuid': 'e-1', 'fact': 'Alice knows Bob', 'name': 'knows'}]
        }
        # total_count=10 means 10 entities exist but only 1 is stale
        detect_result = StaleSummaryResult(stale=stale_list, all_edges=all_edges, total_count=10)
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=detect_result)
        backend._rebuild_entity_from_edges = AsyncMock(
            return_value={
                'uuid': 'u1',
                'name': 'Alice',
                'old_summary': 'old',
                'new_summary': 'Alice knows Bob',
                'edge_count': 1,
            }
        )

        result = await backend.rebuild_entity_summaries(group_id='test', force=False)

        assert result['total_entities'] == 10  # flows from total_count=10
        assert result['stale_entities'] == 1  # only 1 stale
        assert result['rebuilt'] == 1
        assert result['skipped'] == 0
        assert result['errors'] == 0

    @pytest.mark.asyncio
    async def test_force_false_dry_run_total_entities_from_detect(
        self, mock_config, make_backend, make_stale_list
    ):
        """force=False, dry_run=True: total_entities flows from the cheap dry_run probe (task-526).

        After task-526 the force=False dry_run=True path routes through
        _detect_stale_summaries_dry_run (not _detect_stale_summaries_with_edges).
        The probe returns a plain (stale_list, total_count) tuple; total_entities
        in the final result must still come from total_count, not from len(stale_list).
        """
        backend = make_backend(mock_config)
        stale_list = make_stale_list(alice_summary='old A', bob_summary='old B')
        # Mock the new cheap-probe directly: (stale_list, total_count)
        backend._detect_stale_summaries_dry_run = AsyncMock(return_value=(stale_list, 7))

        result = await backend.rebuild_entity_summaries(group_id='test', force=False, dry_run=True)

        assert result['total_entities'] == 7
        assert result['stale_entities'] == 2
        assert result['skipped'] == 2
        assert result['rebuilt'] == 0

    @pytest.mark.asyncio
    async def test_force_false_dry_run_does_not_fetch_edge_map(self, mock_config, make_backend):
        """force=False, dry_run=True: get_all_valid_edges is NOT awaited (task-526).

        The force=False dry_run=True path should NOT pre-fetch the bulk O(E) edge
        map via get_all_valid_edges because the edges are never used — the dry_run
        block short-circuits before the rebuild loop that would consume them.

        Under current code this test FAILS: _detect_stale_summaries_with_edges
        unconditionally awaits get_all_valid_edges regardless of dry_run.
        After the fix (adding _detect_stale_summaries_dry_run), this test passes.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(
            return_value=[
                {'uuid': 'u1', 'name': 'Alice', 'summary': 'some summary'},
            ]
        )
        backend.get_all_valid_edges = AsyncMock(return_value={})
        # Also mock get_valid_edges_for_node so the dry_run probe can run
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])

        await backend.rebuild_entity_summaries(group_id='test', force=False, dry_run=True)

        backend.get_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_force_false_no_dry_run_still_fetches_edge_map(self, mock_config, make_backend):
        """force=False, dry_run=False: get_all_valid_edges IS awaited (positive contrast).

        Positive-contrast companion to test_force_false_dry_run_does_not_fetch_edge_map.
        When dry_run=False the force=False path still routes through
        _detect_stale_summaries_with_edges, which needs the full edge map for the
        actual rebuild (edges are passed into _rebuild_entity_from_edges).

        Guards against a future refactor that accidentally routes ALL force=False
        calls through the cheap dry_run probe — the non-dry-run path must still
        call get_all_valid_edges to obtain the edges used for rebuilding.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(
            return_value=[
                {'uuid': 'u1', 'name': 'Alice', 'summary': 'some summary'},
            ]
        )
        backend.get_all_valid_edges = AsyncMock(return_value={})
        backend._rebuild_entity_from_edges = AsyncMock(
            return_value={
                'uuid': 'u1',
                'name': 'Alice',
                'old_summary': 'some summary',
                'new_summary': '',
                'edge_count': 0,
            }
        )

        await backend.rebuild_entity_summaries(group_id='test', force=False, dry_run=False)

        backend.get_all_valid_edges.assert_awaited_once_with(group_id='test')

    @pytest.mark.asyncio
    async def test_force_false_dry_run_fetches_edges_per_entity(self, mock_config, make_backend, make_stale_list):
        """force=False, dry_run=True: get_valid_edges_for_node awaited once per non-empty-summary entity (post-526).

        Positive complement to test_force_false_dry_run_does_not_fetch_edge_map.
        Task 526 introduced _detect_stale_summaries_dry_run which fetches edges
        per-entity via get_valid_edges_for_node rather than the bulk
        get_all_valid_edges.  This test pins the positive claim in the updated
        docstring: with two non-empty-summary entities, the probe must issue
        exactly two get_valid_edges_for_node awaits — one per entity.

        Regression guard: a future refactor that accidentally short-circuited
        _detect_stale_summaries_dry_run into a no-op (returning an empty stale
        list without ever querying edges) would pass the existing negative test
        but fail here, surfacing the regression immediately.

        This is a characterization test — assertions match current production
        behaviour and should pass on first run with no production changes.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(
            return_value=make_stale_list(alice_summary='some summary 1', bob_summary='some summary 2')
        )
        backend.get_all_valid_edges = AsyncMock(return_value={})
        # Per-entity probe returns no edges → summaries are stale (non-empty summary,
        # canonical facts = '', summary != canonical).
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])

        result = await backend.rebuild_entity_summaries(group_id='test', force=False, dry_run=True)

        # Positive: per-entity fetch ran for each non-empty-summary entity
        assert backend.get_valid_edges_for_node.await_count == 2
        backend.get_valid_edges_for_node.assert_any_await('u1', group_id='test')
        backend.get_valid_edges_for_node.assert_any_await('u2', group_id='test')
        # Negative: bulk fetch was NOT issued (post-526 invariant)
        backend.get_all_valid_edges.assert_not_awaited()
        # Result reflects dry_run short-circuit: both stale entities skipped, none rebuilt
        assert result['total_entities'] == 2
        assert result['rebuilt'] == 0


# ---------------------------------------------------------------------------
# step-5 (task 443): error-accumulation path in rebuild_entity_summaries
# ---------------------------------------------------------------------------


class TestRebuildEntitySummariesErrorHandling:
    """rebuild_entity_summaries records per-entity errors without raising."""

    @pytest.mark.asyncio
    async def test_rebuild_entity_error_recorded_in_result(self, mock_config, make_backend):
        """When _rebuild_entity_from_edges raises, errors counter increments and details record it.

        rebuild_entity_summaries uses asyncio.gather(return_exceptions=True) so a
        per-entity failure does not abort the whole batch. Each exception is captured
        into result['errors'] and result['details'] with status='error'.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(
            return_value=[
                {'uuid': 'u1', 'name': 'Alice', 'summary': 'stale summary'},
            ]
        )
        backend.get_all_valid_edges = AsyncMock(return_value={})
        # Simulate _rebuild_entity_from_edges failing for this entity
        backend._rebuild_entity_from_edges = AsyncMock(side_effect=RuntimeError('boom'))

        result = await backend.rebuild_entity_summaries(group_id='test', force=True, dry_run=False)

        assert result['errors'] == 1
        assert result['rebuilt'] == 0
        assert result['skipped'] == 0
        assert result['total_entities'] == 1
        assert result['stale_entities'] == 1
        assert len(result['details']) == 1
        detail = result['details'][0]
        assert detail['status'] == 'error'
        assert detail['error'] == 'boom'
        assert detail['uuid'] == 'u1'
        assert detail['name'] == 'Alice'

    @pytest.mark.asyncio
    async def test_partial_success_one_error_one_rebuilt(self, mock_config, make_backend, make_stale_list):
        """asyncio.gather returns a mix of exceptions and successes without aborting.

        With two entities, the first raising and the second succeeding, the result
        should record errors=1 and rebuilt=1 with details in target order (u1 first,
        u2 second). This exercises the zip(targets, results, strict=True) accumulator
        loop that is the core value of return_exceptions=True.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(
            return_value=make_stale_list(alice_summary='stale summary', bob_summary='stale summary 2')
        )
        u2_edges: list[EdgeDict] = [
            {'uuid': 'e-1', 'fact': 'fact1', 'name': 'knows'},
            {'uuid': 'e-2', 'fact': 'fact2', 'name': 'knows'},
            {'uuid': 'e-3', 'fact': 'fact3', 'name': 'knows'},
        ]
        backend.get_all_valid_edges = AsyncMock(return_value={'u2': u2_edges})
        backend._rebuild_entity_from_edges = AsyncMock(
            side_effect=[
                RuntimeError('boom'),
                {
                    'uuid': 'u2',
                    'name': 'Bob',
                    'old_summary': '<echoed-old-summary>',
                    'new_summary': 'Bob summary v2',
                    'edge_count': 3,
                },
            ]
        )

        result = await backend.rebuild_entity_summaries(group_id='test', force=True, dry_run=False)

        assert result['errors'] == 1
        assert result['rebuilt'] == 1
        assert result['skipped'] == 0
        assert result['total_entities'] == 2
        assert result['stale_entities'] == 2
        assert len(result['details']) == 2

        err_detail = result['details'][0]
        assert err_detail['status'] == 'error'
        assert err_detail['uuid'] == 'u1'
        assert err_detail['name'] == 'Alice'
        assert err_detail['error'] == 'boom'

        ok_detail = result['details'][1]
        assert ok_detail['status'] == 'rebuilt'
        assert ok_detail['uuid'] == 'u2'
        assert ok_detail['name'] == 'Bob'
        assert ok_detail['old_summary'] == '<echoed-old-summary>'
        assert ok_detail['new_summary'] == 'Bob summary v2'
        assert ok_detail['edge_count'] == 3

        # Two independent forwarding paths are pinned below:
        # 1. mock→detail: ok_detail['old_summary'] == '<echoed-old-summary>' (above) proves
        #    the mock return value flows through the detail-assembly path.
        # 2. fixture→kwarg: assert_any_call(old_summary='stale summary 2') below proves
        #    list_entity_nodes()[i]['summary'] is forwarded as the old_summary kwarg to
        #    _rebuild_entity_from_edges — independent of whatever the mock returns.
        backend._rebuild_entity_from_edges.assert_any_call(
            'u2', 'Bob', u2_edges, group_id='test', old_summary='stale summary 2'
        )
        backend._rebuild_entity_from_edges.assert_any_call(
            'u1', 'Alice', [], group_id='test', old_summary='stale summary'
        )

    @pytest.mark.asyncio
    async def test_force_false_partial_error_uses_detect_total(
        self, mock_config, make_backend, make_stale_list
    ):
        """force=False error path: total_entities flows from StaleSummaryResult.total_count.

        This exercises the force=False bookkeeping path where total_entities comes
        from _detect_stale_summaries_with_edges (result.total_count=5), which is
        independent of stale_entities (=len(targets)=2). This path is not reachable
        via force=True — in that branch total_entities = len(list_entity_nodes()).

        With two stale entities and _rebuild_entity_from_edges raising for the first
        and succeeding for the second, the gather/zip accumulator must record
        errors=1 and rebuilt=1, with details in target order (u1 first, u2 second).
        """
        backend = make_backend(mock_config)
        stale_list = make_stale_list(alice_summary='old A', bob_summary='old B')
        detect_result = StaleSummaryResult(
            stale=stale_list,
            all_edges={'u1': [], 'u2': []},
            total_count=5,
        )
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=detect_result)
        backend._rebuild_entity_from_edges = AsyncMock(
            side_effect=[
                RuntimeError('boom'),
                {
                    'uuid': 'u2',
                    'name': 'Bob',
                    'old_summary': 'old B',
                    'new_summary': 'rebuilt B',
                    'edge_count': 0,
                },
            ]
        )

        result = await backend.rebuild_entity_summaries(group_id='test', force=False)

        assert result['total_entities'] == 5  # flows from total_count=5, not len(stale)
        assert result['stale_entities'] == 2  # len(targets) = len(stale_list)
        assert result['errors'] == 1
        assert result['rebuilt'] == 1
        assert result['skipped'] == 0
        assert len(result['details']) == 2

        err_detail = result['details'][0]
        assert err_detail['status'] == 'error'
        assert err_detail['uuid'] == 'u1'
        assert err_detail['name'] == 'Alice'
        assert err_detail['error'] == 'boom'

        ok_detail = result['details'][1]
        assert ok_detail['status'] == 'rebuilt'
        assert ok_detail['uuid'] == 'u2'
        assert ok_detail['name'] == 'Bob'
        assert ok_detail['new_summary'] == 'rebuilt B'
        assert ok_detail['old_summary'] == 'old B'
        assert ok_detail['edge_count'] == 0

        backend._detect_stale_summaries_with_edges.assert_awaited_once_with(group_id='test')
        assert backend._rebuild_entity_from_edges.await_count == 2

        # Symmetrical to the force=True test: pin the force=False forwarding path
        # where the targets list-comp sets t['old_summary'] = s['summary'] and
        # _rebuild_one forwards it as the old_summary kwarg to
        # _rebuild_entity_from_edges. Edges are [] because detect_result.all_edges
        # maps both uuids to empty lists.
        backend._rebuild_entity_from_edges.assert_any_call(
            'u2', 'Bob', [], group_id='test', old_summary='old B'
        )
        backend._rebuild_entity_from_edges.assert_any_call(
            'u1', 'Alice', [], group_id='test', old_summary='old A'
        )


# ---------------------------------------------------------------------------
# step-4: regression — whitespace-only fact must not cause false stale detection
# ---------------------------------------------------------------------------


class TestCanonicalFactsStalenessRegression:
    """Whitespace-only facts must not make a current entity appear stale."""

    @pytest.mark.asyncio
    async def test_whitespace_fact_does_not_cause_false_stale_detection(
        self, mock_config, make_backend
    ):
        """Entity with summary 'A knows B' and edges ['   ', 'A knows B'] is not stale.

        Before the fix, _canonical_facts(['   ', 'A knows B']) returned
        ['   ', 'A knows B'], so the canonical string became '   \nA knows B',
        which does not match the stored summary 'A knows B' and the entity was
        falsely flagged as stale.

        After the fix, _canonical_facts filters out '   ', returns ['A knows B'],
        and the joined canonical string 'A knows B' matches the stored summary —
        so the entity is NOT stale.

        The total_count assertion (added in task-492) strengthens this regression
        guard: stale==[] alone could pass trivially if the entity loop never ran
        (e.g. if list_entity_nodes returned zero entities).  total_count=1 pins
        that exactly one entity was scanned and _canonical_facts was actually
        exercised on its edges.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(
            return_value=[
                {'uuid': 'u1', 'name': 'Alice', 'summary': 'A knows B'},
            ]
        )
        backend.get_all_valid_edges = AsyncMock(
            return_value={
                'u1': [{'fact': '   '}, {'fact': 'A knows B'}],
            }
        )

        result = await backend._detect_stale_summaries_with_edges(group_id='test')

        assert result.stale == [], (
            'Entity should NOT be flagged stale when its only non-whitespace '
            'fact matches the stored summary.'
        )
        assert result.total_count == 1, (
            'total_count must confirm the entity was actually scanned; '
            'stale==[] alone could pass if the loop never ran.'
        )


# ---------------------------------------------------------------------------
# task-492: caller-coverage — _rebuild_entity_from_edges whitespace filter
# ---------------------------------------------------------------------------


class TestCanonicalFactsCallerCoverage:
    """_canonical_facts is exercised correctly via the bulk-rebuild call site.

    Caller-coverage gap (task-492): _canonical_facts is called by
    refresh_entity_summary, _detect_stale_summaries_with_edges,
    _detect_stale_summaries_dry_run, and _rebuild_entity_from_edges.  Only
    the stale-detection path had a whitespace-regression test before this
    task.  This class pins that the bulk-rebuild path also filters
    whitespace-only facts — a regression in _canonical_facts would corrupt
    every summary written through the batch path, not just stale-detection
    results.
    """

    @pytest.mark.asyncio
    async def test_rebuild_entity_from_edges_filters_whitespace_only_facts(
        self, mock_config, make_backend
    ):
        """_rebuild_entity_from_edges must not write whitespace-only lines to the summary."""
        backend = make_backend(mock_config)
        backend.update_node_summary = AsyncMock()

        edges = [
            {'uuid': 'e1', 'fact': '   ', 'name': 'edge1'},
            {'uuid': 'e2', 'fact': '\t', 'name': 'edge2'},
            {'uuid': 'e3', 'fact': 'Alice knows Bob', 'name': 'knows'},
            {'uuid': 'e4', 'fact': '\n  \t', 'name': 'edge4'},
        ]

        result = await backend._rebuild_entity_from_edges(
            'uuid-1', 'Alice', edges, group_id='test', old_summary='old'
        )

        # Exact match — only the one real fact, no whitespace-only lines
        assert result['new_summary'] == 'Alice knows Bob', (
            'new_summary must contain only real-content facts; whitespace-only '
            'edges must be filtered by _canonical_facts before joining.'
        )

        # update_node_summary was called with the clean summary
        backend.update_node_summary.assert_awaited_once_with(
            'uuid-1', 'Alice knows Bob', group_id='test'
        )

        # edge_count is the raw count (includes whitespace-filtered edges),
        # matching the documented contract in TestRebuildEntityFromEdgesOldSummary.
        assert result['edge_count'] == 4, (
            'edge_count must reflect the raw number of edges supplied, '
            'not the filtered count.'
        )


# ---------------------------------------------------------------------------
# task-507: make_stale_list fixture contract tests
# ---------------------------------------------------------------------------


class TestMakeStaleListFixture:
    """Direct contract tests for the make_stale_list factory fixture."""

    def test_default_shape(self, make_stale_list):
        """make_stale_list() with no args returns the standard Alice/Bob two-entity list."""
        result = make_stale_list()
        assert result == [
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'summary A'},
            {'uuid': 'u2', 'name': 'Bob', 'summary': 'summary B'},
        ]

    def test_custom_summaries(self, make_stale_list):
        """make_stale_list(alice_summary=..., bob_summary=...) returns overridden summary values."""
        result = make_stale_list(alice_summary='old A', bob_summary='old B')
        assert result == [
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'old A'},
            {'uuid': 'u2', 'name': 'Bob', 'summary': 'old B'},
        ]
