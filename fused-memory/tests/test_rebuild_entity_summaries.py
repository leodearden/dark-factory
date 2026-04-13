"""Tests for rebuild_entity_summaries across backends, service, MCP tool, and CLI.

Covers:
- GraphitiBackend.list_entity_nodes()           (step 1)
- GraphitiBackend.detect_stale_summaries()      (step 2)
- GraphitiBackend.rebuild_entity_summaries()    (step 3)
- MemoryService.rebuild_entity_summaries()      (step 4)
- MCP tool rebuild_entity_summaries             (step 5)
- DISALLOW_MEMORY_WRITES list                   (step 6)
- RebuildSummariesManager / run_rebuild_summaries (step 7)
- GraphitiBackend.get_all_valid_edges()           → see test_refresh_entity_summary.py
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend, StaleSummaryResult


def _make_rebuild_spy(backend) -> tuple[list[dict], Callable]:
    """Install a strict spy on backend._rebuild_entity_from_edges.

    Captures each call's uuid and old_summary, then delegates to the original.
    Returns (captured_calls, original_rebuild).

    The spy uses a required (no-default) old_summary kwarg to mirror the
    production signature — any caller that drops old_summary will raise TypeError
    instead of silently capturing None.
    """
    captured_calls: list[dict] = []
    original_rebuild = backend._rebuild_entity_from_edges

    async def capture_rebuild(uuid, name, edges, *, group_id, old_summary):
        captured_calls.append({'uuid': uuid, 'old_summary': old_summary})
        return await original_rebuild(uuid, name, edges, group_id=group_id, old_summary=old_summary)

    backend._rebuild_entity_from_edges = capture_rebuild
    return captured_calls, original_rebuild


# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.list_entity_nodes
# ---------------------------------------------------------------------------

class TestListEntityNodes:
    """GraphitiBackend.list_entity_nodes() returns all Entity nodes."""

    @pytest.mark.asyncio
    async def test_returns_entity_nodes(self, mock_config, make_backend, make_graph_mock):
        """Returns list of dicts with uuid/name/summary keys."""
        backend = make_backend(mock_config)
        rows = [
            ['uuid-1', 'Alice', 'Alice knows Bob'],
            ['uuid-2', 'Bob', 'Bob lives in London'],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.list_entity_nodes(group_id='test')
        assert len(result) == 2
        assert result[0] == {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'Alice knows Bob'}
        assert result[1] == {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'Bob lives in London'}

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_list(self, mock_config, make_backend, make_graph_mock):
        """Returns empty list when no Entity nodes exist."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.list_entity_nodes(group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_null_summary_defaults_to_empty_string(self, mock_config, make_backend, make_graph_mock):
        """Nodes with NULL summary field return empty string."""
        backend = make_backend(mock_config)
        rows = [['uuid-1', 'Alice', None]]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.list_entity_nodes(group_id='test')
        assert result[0]['summary'] == ''

    @pytest.mark.asyncio
    async def test_uses_ro_query(self, mock_config, make_backend, make_graph_mock):
        """Uses ro_query (read-only) for the list operation."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.list_entity_nodes(group_id='test')
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.list_entity_nodes(group_id='test')


# ---------------------------------------------------------------------------
# step-2: GraphitiBackend.detect_stale_summaries
# ---------------------------------------------------------------------------

class TestDetectStaleSummaries:
    """GraphitiBackend.detect_stale_summaries() flags entities with stale summaries."""

    @pytest.mark.asyncio
    async def test_clean_entity_not_flagged(self, mock_config, make_backend, make_edge_backend):
        """Entity whose summary exactly matches deduped valid edge facts is not returned."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'Alice knows Bob'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'Alice knows Bob', 'name': 'knows'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_duplicate_lines_detected(self, mock_config, make_backend, make_edge_backend):
        """Entity with duplicate summary lines is flagged with correct duplicate_count."""
        # Summary has 'A\nA\nB' — 'A' appears twice → duplicate_count=1
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factA\nfactA\nfactB'},
        ], edges={
            'uuid-1': [
                {'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'},
                {'uuid': 'e2', 'fact': 'factB', 'name': 'edge2'},
            ],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        assert result[0]['uuid'] == 'uuid-1'
        assert result[0]['duplicate_count'] == 1

    @pytest.mark.asyncio
    async def test_stale_lines_detected(self, mock_config, make_backend, make_edge_backend):
        """Entity with lines not in any valid edge fact is flagged with stale_line_count."""
        # 'old stale fact' is in summary but NOT a valid edge
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'current fact\nold stale fact'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        assert result[0]['stale_line_count'] == 1
        assert result[0]['valid_fact_count'] == 1

    @pytest.mark.asyncio
    async def test_mixed_staleness(self, mock_config, make_backend, make_edge_backend):
        """Entity with both duplicates and stale lines reports both counts."""
        # 'factA' duplicated (1 extra), 'old' is stale
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factA\nfactA\nold'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        assert result[0]['duplicate_count'] == 1
        # 'factA' is in valid_fact_set so it's not counted as stale even when duplicated;
        # only 'old' (not backed by any valid edge) is stale.
        assert result[0]['stale_line_count'] == 1

    @pytest.mark.asyncio
    async def test_empty_summary_not_flagged(self, mock_config, make_backend, make_edge_backend):
        """Entity with empty summary is not flagged as stale."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': ''},
        ], edges={})
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []
        # get_all_valid_edges still called once (before the loop), even if no edges
        backend.get_all_valid_edges.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_entities_returns_empty(self, mock_config, make_backend, make_edge_backend):
        """Empty graph returns empty stale list."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[], edges={})
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_result_includes_summary_line_count(self, mock_config, make_backend, make_edge_backend):
        """Stale entity dict includes summary_line_count."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'lineA\nlineB\nlineC'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'lineA', 'name': 'edge1'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert result[0]['summary_line_count'] == 3

    @pytest.mark.asyncio
    async def test_same_facts_different_order_triggers_rebuild(self, mock_config, make_backend, make_edge_backend):
        """Identical facts in a different order flag the entity as stale.

        This is a known limitation of the current implementation, not a deliberate
        design invariant: canonical summary follows edge-result order, which is
        non-deterministic. Future work to sort facts would require updating this test.
        """
        # Edges return factA first, factB second → canonical = 'factA\nfactB'
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factB\nfactA'},
        ], edges={
            'uuid-1': [
                {'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'},
                {'uuid': 'e2', 'fact': 'factB', 'name': 'edge2'},
            ],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        entity = result[0]
        assert entity['uuid'] == 'uuid-1'
        # Both lines are present in valid_fact_set → stale_line_count == 0
        assert entity['stale_line_count'] == 0
        # No duplicate lines → duplicate_count == 0
        assert entity['duplicate_count'] == 0
        # 2 edges with facts 'factA' and 'factB' were consulted
        assert entity['valid_fact_count'] == 2
        # Entity is stale due to order mismatch (summary != canonical), not content issues

    @pytest.mark.asyncio
    async def test_entity_with_zero_valid_edges_flagged_stale(self, mock_config, make_backend, make_edge_backend):
        """Entity with non-empty summary but zero valid edges is flagged stale.

        When get_all_valid_edges returns no edges for the entity, the canonical
        summary is '' (empty). Since summary != canonical, the entity is stale.
        All summary lines are counted as stale_line_count because none appear
        in the empty valid_fact_set. valid_fact_count=0, duplicate_count=0.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'old fact A\nold fact B'},
        ], edges={})
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        entity = result[0]
        assert entity['uuid'] == 'uuid-1'
        assert entity['stale_line_count'] == 2
        assert entity['valid_fact_count'] == 0
        assert entity['duplicate_count'] == 0
        assert entity['summary_line_count'] == 2


# ---------------------------------------------------------------------------
# task-526: GraphitiBackend._detect_stale_summaries_dry_run (cheap probe)
# ---------------------------------------------------------------------------

class TestDetectStaleSummariesDryRun:
    """Unit tests for GraphitiBackend._detect_stale_summaries_dry_run.

    This private method is the memory-cheaper alternative to
    ``_detect_stale_summaries_with_edges`` used by the force=False, dry_run=True
    code path in ``rebuild_entity_summaries``.

    Unlike the bulk variant, this probe issues up-to-N targeted
    ``get_valid_edges_for_node`` calls (one per non-empty-summary entity)
    and never materialises the O(E) all-edges dict.  The stale dict schema
    it produces is identical to ``_detect_stale_summaries_with_edges`` so
    that downstream consumers (e.g. the result['details'] field) see no
    surface-level change.

    These tests live in test_rebuild_entity_summaries.py (the canonical file
    for rebuild_entity_summaries coverage) rather than in
    test_graphiti_rebuild_pipeline.py (rebuild/refresh pipeline quality improvements) per the
    established layout where each backend method gets a dedicated test class.
    See task 526 for the full design rationale.
    """

    @pytest.mark.asyncio
    async def test_returns_stale_list_and_total_count(self, mock_config, make_backend):
        """Returns a (list, int) tuple where int is the total entity count.

        Three entities:
        - Alice: stale (summary != canonical)
        - Bob: clean (summary matches canonical)
        - Charlie: empty summary (cheap-skipped)

        Expected: stale_list has 1 entry (Alice), total_count == 3.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-alice', 'name': 'Alice', 'summary': 'old fact'},
            {'uuid': 'uuid-bob', 'name': 'Bob', 'summary': 'current fact'},
            {'uuid': 'uuid-charlie', 'name': 'Charlie', 'summary': ''},
        ])

        def edges_for_node(node_uuid, *, group_id):
            if node_uuid == 'uuid-alice':
                return [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]
            if node_uuid == 'uuid-bob':
                return [{'uuid': 'e2', 'fact': 'current fact', 'name': 'edge2'}]
            return []

        backend.get_valid_edges_for_node = AsyncMock(side_effect=edges_for_node)

        result = await backend._detect_stale_summaries_dry_run(group_id='test')

        assert isinstance(result, tuple)
        stale_list, total_count = result
        assert isinstance(stale_list, list)
        assert isinstance(total_count, int)
        assert total_count == 3
        assert len(stale_list) == 1
        assert stale_list[0]['uuid'] == 'uuid-alice'

    @pytest.mark.asyncio
    async def test_flags_stale_entity_with_diagnostic_counts(self, mock_config, make_backend):
        """Stale entity dict has the same schema as _detect_stale_summaries_with_edges.

        Keys: uuid, name, summary, duplicate_count, stale_line_count,
              valid_fact_count, summary_line_count.
        """
        backend = make_backend(mock_config)
        # Summary has 'factA' duplicated and 'old' which is not a valid edge fact.
        # valid edges: factA only.
        # duplicate_count=1 ('factA' appears twice → 2-1=1 extra)
        # stale_line_count=1 ('old' is not in valid_fact_set, 'factA' appears twice
        #   but the duplicate is also 'factA' which IS in valid_fact_set, so only 'old' is stale)
        # valid_fact_count=1 ('factA')
        # summary_line_count=3 ('factA', 'factA', 'old')
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factA\nfactA\nold'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'},
        ])

        stale_list, total_count = await backend._detect_stale_summaries_dry_run(group_id='test')

        assert total_count == 1
        assert len(stale_list) == 1
        entity = stale_list[0]

        # Schema keys must exactly match _detect_stale_summaries_with_edges output
        expected_keys = {
            'uuid', 'name', 'summary',
            'duplicate_count', 'stale_line_count', 'valid_fact_count', 'summary_line_count',
        }
        assert set(entity.keys()) == expected_keys

        assert entity['uuid'] == 'uuid-1'
        assert entity['name'] == 'Alice'
        assert entity['summary'] == 'factA\nfactA\nold'
        assert entity['duplicate_count'] == 1   # 'factA' appears twice → 1 extra
        assert entity['stale_line_count'] == 1  # 'old' not in valid_fact_set
        assert entity['valid_fact_count'] == 1  # deduped: only 'factA'
        assert entity['summary_line_count'] == 3

    @pytest.mark.asyncio
    async def test_skips_clean_entity(self, mock_config, make_backend):
        """Entity whose summary exactly matches canonical facts is not flagged as stale.

        Summary 'current fact' matches the single valid edge fact → clean.
        stale_list should be empty, but total_count is still 1.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'current fact'},
        ])
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'},
        ])

        stale_list, total_count = await backend._detect_stale_summaries_dry_run(group_id='test')

        assert stale_list == []
        assert total_count == 1

    @pytest.mark.asyncio
    async def test_does_not_call_get_all_valid_edges(self, mock_config, make_backend):
        """The dry_run probe never awaits get_all_valid_edges.

        This is the defining property of the cheap probe: it fetches edges
        per-entity via get_valid_edges_for_node rather than materialising the
        O(E) bulk all-edges dict.  The assertion style follows the project's
        AsyncMock convention (assert_not_awaited, not assert_not_called) as
        enforced in test_graphiti_rebuild_pipeline.py.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale fact'},
        ])
        backend.get_all_valid_edges = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'},
        ])

        await backend._detect_stale_summaries_dry_run(group_id='test')

        backend.get_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_empty_summary_entity_without_edge_query(self, mock_config, make_backend):
        """Empty-summary entities are cheap-skipped: no get_valid_edges_for_node call.

        Empty summaries are 'not stale by definition' (matches the semantics
        of _detect_stale_summaries_with_edges at graphiti_client.py:905-907).
        The probe additionally saves the edge query for those entities.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Charlie', 'summary': ''},
        ])
        backend.get_valid_edges_for_node = AsyncMock()

        stale_list, total_count = await backend._detect_stale_summaries_dry_run(group_id='test')

        # Empty summary is not stale
        assert stale_list == []
        assert total_count == 1
        # get_valid_edges_for_node must NOT have been called for the empty-summary entity
        backend.get_valid_edges_for_node.assert_not_awaited()


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.rebuild_entity_summaries
# ---------------------------------------------------------------------------

class TestRebuildEntitySummaries:
    """GraphitiBackend.rebuild_entity_summaries() batch-rebuilds stale entities."""

    @pytest.mark.asyncio
    async def test_rebuilds_only_stale_entities(self, mock_config, make_backend, make_edge_backend):
        """Only stale entities are rebuilt; clean entities are skipped.

        Alice has summary='stale fact' but her only valid edge has fact='current fact',
        so the canonical is 'current fact' != 'stale fact' → stale.
        Bob has summary='current fact' matching his edge canonical → clean.
        Total entities=2, stale=1, rebuilt=1.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale fact'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'current fact'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'current fact', 'name': 'edge2'}],
        })
        backend.update_node_summary = AsyncMock()
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert result['total_entities'] == 2
        assert result['stale_entities'] == 1
        assert result['rebuilt'] == 1
        backend.update_node_summary.assert_awaited_once_with('uuid-1', ANY, group_id='test')
        backend.list_entity_nodes.assert_awaited_once_with(group_id='test')
        backend.get_all_valid_edges.assert_awaited_once_with(group_id='test')
        assert result['details'][0]['old_summary'] == 'stale fact'

    @pytest.mark.asyncio
    async def test_force_rebuilds_all(self, mock_config, make_backend, make_edge_backend):
        """With force=True, rebuilds all entities using get_all_valid_edges (no refresh_entity_summary)."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'ok'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'also ok'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'fact1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'fact2', 'name': 'edge2'}],
        })
        backend.update_node_summary = AsyncMock()
        result = await backend.rebuild_entity_summaries(group_id='test', force=True)
        assert result['total_entities'] == 2
        assert result['stale_entities'] == 2
        assert result['rebuilt'] == 2
        assert backend.update_node_summary.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_aggregate_result(self, mock_config, make_backend):
        """Returns dict with total_entities, stale_entities, rebuilt, skipped, errors, details."""
        backend = make_backend(mock_config)
        backend._detect_stale_summaries_with_edges = AsyncMock(
            return_value=StaleSummaryResult(stale=[], all_edges={}, total_count=0)
        )
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert set(result.keys()) == {'total_entities', 'stale_entities', 'rebuilt', 'skipped', 'errors', 'details'}

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, mock_config, make_backend, make_edge_backend):
        """Both entities are detected stale because their summaries differ from the canonical facts derived from their valid edges.

        Alice's update_node_summary raises RuntimeError; Bob's succeeds.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale1'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'stale2'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'current2', 'name': 'edge2'}],
        })
        # side_effect order matches list_entity_nodes return order: Alice first, Bob second
        backend.update_node_summary = AsyncMock(side_effect=[
            RuntimeError('FalkorDB timeout'),
            None,
        ])
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert result['errors'] == 1
        assert result['rebuilt'] == 1
        assert len(result['details']) == 2
        error_detail = next(d for d in result['details'] if d['status'] == 'error')
        assert 'FalkorDB timeout' in error_detail['error']
        assert error_detail['uuid'] == 'uuid-1'
        backend.list_entity_nodes.assert_awaited_once_with(group_id='test')
        backend.get_all_valid_edges.assert_awaited_once_with(group_id='test')

    @pytest.mark.asyncio
    async def test_empty_graph_returns_zero_counts(self, mock_config, make_backend, make_edge_backend):
        """No entities means all counts are 0."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[], edges={})
        result = await backend.rebuild_entity_summaries(group_id='test')
        assert result['total_entities'] == 0
        assert result['stale_entities'] == 0
        assert result['rebuilt'] == 0
        assert result['skipped'] == 0
        assert result['errors'] == 0
        assert result['details'] == []

    @pytest.mark.asyncio
    async def test_dry_run_returns_stale_without_rebuilding(self, mock_config, make_backend):
        """With dry_run=True, detects stale entities but does not call update_node_summary.

        Alice has summary='stale fact' while her edge canonical is 'current fact' → stale.
        force=False + dry_run=True routes through _detect_stale_summaries_dry_run
        (task 526: the cheap per-entity probe that avoids the O(E) bulk edge fetch).
        Mocks that probe directly so the data-flow assertion holds without requiring
        a real graph driver.
        Explicitly mocks update_node_summary to document that the dry_run
        guarantee holds even if rebuild_entity_summaries were refactored
        to bypass _rebuild_entity_from_edges and call those methods directly.
        """
        backend = make_backend(mock_config)
        stale_list = [{'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale',
                       'duplicate_count': 0, 'stale_line_count': 1, 'valid_fact_count': 0,
                       'summary_line_count': 1}]
        backend._detect_stale_summaries_dry_run = AsyncMock(return_value=(stale_list, 1))
        backend.update_node_summary = AsyncMock()
        result = await backend.rebuild_entity_summaries(group_id='test', dry_run=True)
        assert result['stale_entities'] == 1
        assert result['rebuilt'] == 0
        assert result['skipped'] == 1
        backend.update_node_summary.assert_not_awaited()
        assert result['details'][0]['status'] == 'skipped_dry_run'
        # task 526 refactor: dry_run path no longer calls list_entity_nodes /
        # get_all_valid_edges directly; assert group_id propagates to the probe instead.
        backend._detect_stale_summaries_dry_run.assert_awaited_once_with(group_id='test')


# ---------------------------------------------------------------------------
# step-4: MemoryService.rebuild_entity_summaries
# ---------------------------------------------------------------------------

class TestMemoryServiceRebuildEntitySummaries:
    """MemoryService.rebuild_entity_summaries() delegates to graphiti backend."""

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with mocked backends (no real DB needed)."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 5,
            'stale_entities': 2,
            'rebuilt': 2,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_backend(self, service):
        """Calls graphiti.rebuild_entity_summaries with correct group_id, force, dry_run."""
        result = await service.rebuild_entity_summaries(
            project_id='dark_factory',
            force=False,
            dry_run=False,
        )
        service.graphiti.rebuild_entity_summaries.assert_awaited_once_with(
            group_id='dark_factory', force=False, dry_run=False
        )
        assert result['total_entities'] == 5

    @pytest.mark.asyncio
    async def test_journal_logs_on_success(self, service):
        """Write journal records operation with params and result_summary."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.rebuild_entity_summaries(
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'rebuild_entity_summaries'
        assert call_kwargs.get('project_id') == 'dark_factory'
        assert call_kwargs.get('success') is True

    @pytest.mark.asyncio
    async def test_journal_logs_on_failure(self, service):
        """Backend exception is re-raised; journal records success=False with error message."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        service.graphiti.rebuild_entity_summaries = AsyncMock(
            side_effect=RuntimeError('FalkorDB unavailable')
        )
        with pytest.raises(RuntimeError, match='FalkorDB unavailable'):
            await service.rebuild_entity_summaries(project_id='dark_factory')
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('success') is False
        assert 'FalkorDB unavailable' in call_kwargs.get('error', '')

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_success(self, service):
        """If journal.log_write_op raises, the successful result is still returned."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal full'))
        service.set_write_journal(mock_journal)
        result = await service.rebuild_entity_summaries(project_id='dark_factory')
        # Should NOT raise — journal failure must not mask the successful operation
        assert result['total_entities'] == 5

    @pytest.mark.asyncio
    async def test_journal_result_summary_is_condensed(self, mock_config):
        """Journal result_summary must contain only aggregate fields, not 'details'.

        The full backend result includes a 'details' list (one entry per rebuilt
        entity with uuid, name, status, old_summary, new_summary, edge_count, error).
        Passing this verbatim bloats the journal for large graphs. The service must
        condense result_summary to only the five aggregate fields.
        """
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        # Backend returns a result with a non-empty 'details' list (the bloat source)
        svc.graphiti.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 10,
            'stale_entities': 4,
            'rebuilt': 3,
            'skipped': 1,
            'errors': 0,
            'details': [
                {
                    'uuid': 'u1', 'name': 'Alice', 'status': 'rebuilt',
                    'old_summary': 'old text', 'new_summary': 'new text',
                    'edge_count': 5, 'error': None,
                },
                {
                    'uuid': 'u2', 'name': 'Bob', 'status': 'rebuilt',
                    'old_summary': 'old bob', 'new_summary': 'new bob',
                    'edge_count': 3, 'error': None,
                },
            ],
        })
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        svc.set_write_journal(mock_journal)
        await svc.rebuild_entity_summaries(project_id='dark_factory', agent_id='test-agent')
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        result_summary = call_kwargs.get('result_summary')
        assert result_summary is not None, 'result_summary must be set on success'
        # Must contain the five aggregate fields
        assert 'total_entities' in result_summary
        assert 'stale_entities' in result_summary
        assert 'rebuilt' in result_summary
        assert 'errors' in result_summary
        # Must NOT contain the verbose 'details' list
        assert 'details' not in result_summary, (
            'result_summary must not include the per-entity details list — '
            'it bloats the journal for large graphs'
        )
        # Values must match the backend result
        assert result_summary['total_entities'] == 10
        assert result_summary['stale_entities'] == 4
        assert result_summary['rebuilt'] == 3
        assert result_summary['errors'] == 0
        # Must contain 'skipped' aggregate field
        assert 'skipped' in result_summary
        assert result_summary['skipped'] == 1
        # Key set must be exactly these five aggregate fields — no more, no less
        assert set(result_summary.keys()) == {
            'total_entities', 'stale_entities', 'rebuilt', 'skipped', 'errors'
        }


# ---------------------------------------------------------------------------
# step-5: MCP tool rebuild_entity_summaries
# ---------------------------------------------------------------------------

class TestRebuildEntitySummariesMcpTool:
    """MCP tool rebuild_entity_summaries is registered and delegates correctly."""

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 3,
            'stale_entities': 1,
            'rebuilt': 1,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_registered(self, mcp_server):
        """Tool appears in MCP server's tool list."""
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'rebuild_entity_summaries' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_service(self, mcp_server, mock_service):
        """Calls memory_service.rebuild_entity_summaries with correct args."""
        await mcp_server._tool_manager.call_tool(
            'rebuild_entity_summaries',
            {'project_id': 'dark_factory', 'force': False, 'dry_run': True},
        )
        mock_service.rebuild_entity_summaries.assert_awaited_once()
        call_kwargs = mock_service.rebuild_entity_summaries.call_args[1]
        assert call_kwargs.get('project_id') == 'dark_factory'
        assert call_kwargs.get('dry_run') is True

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_error(self, mcp_server, mock_service):
        """Returns validation error dict for invalid project_id."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'rebuild_entity_summaries',
            {'project_id': ''},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.rebuild_entity_summaries.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        """Backend exception returns {error, error_type} dict, not raw exception."""
        import json
        mock_service.rebuild_entity_summaries = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'rebuild_entity_summaries',
            {'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']


# ---------------------------------------------------------------------------
# step-6: DISALLOW_MEMORY_WRITES
# ---------------------------------------------------------------------------

class TestDisallowListForRebuildEntitySummaries:
    """rebuild_entity_summaries must be in DISALLOW_MEMORY_WRITES (not in STAGE1_DISALLOWED)."""

    def test_in_disallow_memory_writes(self):
        """'mcp__fused-memory__rebuild_entity_summaries' must be in DISALLOW_MEMORY_WRITES
        so Stage 3 (read-only) cannot call it."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__rebuild_entity_summaries' in DISALLOW_MEMORY_WRITES

    def test_not_in_stage1_disallowed(self):
        """Stage 1 must be able to call rebuild_entity_summaries (not in STAGE1_DISALLOWED)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__rebuild_entity_summaries' not in STAGE1_DISALLOWED

    def test_in_stage3_disallowed(self):
        """Stage 3 must NOT be able to call rebuild_entity_summaries."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__rebuild_entity_summaries' in STAGE3_DISALLOWED


# ---------------------------------------------------------------------------
# step-7: Maintenance CLI — RebuildSummariesManager + run_rebuild_summaries
# ---------------------------------------------------------------------------

class TestRebuildSummariesManager:
    """RebuildSummariesManager.run() delegates to backend.rebuild_entity_summaries."""

    @pytest.mark.asyncio
    async def test_manager_delegates_to_backend(self, mock_config):
        """RebuildSummariesManager.run() calls backend.rebuild_entity_summaries."""
        from fused_memory.maintenance.rebuild_summaries import RebuildSummariesManager
        mock_backend = MagicMock()
        mock_backend.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 4,
            'stale_entities': 2,
            'rebuilt': 2,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        manager = RebuildSummariesManager(backend=mock_backend, group_id='test_project')
        result = await manager.run()
        mock_backend.rebuild_entity_summaries.assert_awaited_once_with(
            group_id='test_project', force=False, dry_run=False
        )
        assert result.total_entities == 4
        assert result.rebuilt == 2

    @pytest.mark.asyncio
    async def test_manager_passes_force_and_dry_run(self, mock_config):
        """force and dry_run params are forwarded correctly to the backend."""
        from fused_memory.maintenance.rebuild_summaries import RebuildSummariesManager
        mock_backend = MagicMock()
        mock_backend.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 2,
            'stale_entities': 2,
            'rebuilt': 0,
            'skipped': 2,
            'errors': 0,
            'details': [],
        })
        manager = RebuildSummariesManager(backend=mock_backend, group_id='test_project')
        result = await manager.run(force=True, dry_run=True)
        mock_backend.rebuild_entity_summaries.assert_awaited_once_with(
            group_id='test_project', force=True, dry_run=True
        )
        assert result.skipped == 2

    @pytest.mark.asyncio
    async def test_run_entrypoint_uses_maintenance_service(self, make_fake_maintenance_service):
        """run_rebuild_summaries() uses maintenance_service context manager."""
        from fused_memory.maintenance.rebuild_summaries import run_rebuild_summaries
        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_service.graphiti.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 0,
            'stale_entities': 0,
            'rebuilt': 0,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        fake_ctx = make_fake_maintenance_service(mock_cfg, mock_service)
        with patch(
            'fused_memory.maintenance.rebuild_summaries.maintenance_service',
            side_effect=fake_ctx,
        ):
            result = await run_rebuild_summaries(config_path='/fake/config.yaml', group_id='test')
        assert result.total_entities == 0

    @pytest.mark.asyncio
    async def test_run_entrypoint_returns_result(self, make_fake_maintenance_service):
        """Returns the RebuildResult from the manager."""
        from fused_memory.maintenance.rebuild_summaries import RebuildResult, run_rebuild_summaries
        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_service.graphiti.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 7,
            'stale_entities': 3,
            'rebuilt': 3,
            'skipped': 0,
            'errors': 0,
            'details': [{'uuid': 'u1', 'name': 'Alice', 'status': 'rebuilt',
                         'old_summary': 'old', 'new_summary': 'new', 'edge_count': 2}],
        })
        fake_ctx = make_fake_maintenance_service(mock_cfg, mock_service)
        with patch(
            'fused_memory.maintenance.rebuild_summaries.maintenance_service',
            side_effect=fake_ctx,
        ):
            result = await run_rebuild_summaries(group_id='test')
        assert isinstance(result, RebuildResult)
        assert result.total_entities == 7
        assert result.stale_entities == 3
        assert result.rebuilt == 3
        assert len(result.details) == 1

    @pytest.mark.asyncio
    async def test_run_entrypoint_emits_exactly_one_summary_log(
        self, make_fake_maintenance_service, caplog
    ):
        """run_rebuild_summaries() emits exactly one 'run_rebuild_summaries complete' log.

        This is a regression guard: after removing the manager's summary log, the
        entrypoint must still emit its own single summary line (no double-logging,
        no silent dropping).
        """
        import logging
        from unittest.mock import patch

        from fused_memory.maintenance.rebuild_summaries import run_rebuild_summaries

        mock_cfg = MagicMock()
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_service.graphiti.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 5,
            'stale_entities': 3,
            'rebuilt': 3,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        fake_ctx = make_fake_maintenance_service(mock_cfg, mock_service)
        with caplog.at_level(logging.INFO, logger='fused_memory.maintenance.rebuild_summaries'), patch(
            'fused_memory.maintenance.rebuild_summaries.maintenance_service',
            side_effect=fake_ctx,
        ):
            await run_rebuild_summaries(config_path='/fake/config.yaml', group_id='test')
        matching = [r for r in caplog.records if 'run_rebuild_summaries complete' in r.message]
        assert len(matching) == 1, (
            f'Expected exactly 1 summary log from run_rebuild_summaries, got {len(matching)}: '
            f'{[r.message for r in matching]}'
        )

    @pytest.mark.asyncio
    async def test_manager_does_not_emit_summary_log(self, mock_config, caplog):
        """RebuildSummariesManager.run() must NOT emit a summary log line.

        Following the CleanupManager precedent, the inner manager emits only
        operational per-item logs. The summary line belongs exclusively in the
        outer run_rebuild_summaries() entrypoint. This test guards against
        duplicate summary lines appearing in production logs.
        """
        import logging

        from fused_memory.maintenance.rebuild_summaries import RebuildSummariesManager
        mock_backend = MagicMock()
        mock_backend.rebuild_entity_summaries = AsyncMock(return_value={
            'total_entities': 3,
            'stale_entities': 1,
            'rebuilt': 1,
            'skipped': 0,
            'errors': 0,
            'details': [],
        })
        manager = RebuildSummariesManager(backend=mock_backend, group_id='test')
        with caplog.at_level(logging.INFO, logger='fused_memory.maintenance.rebuild_summaries'):
            await manager.run()
        # The manager must NOT emit a 'run complete' summary — that belongs in
        # the outer entrypoint run_rebuild_summaries() only.
        matching = [r for r in caplog.records if 'RebuildSummariesManager.run complete' in r.message]
        assert matching == [], (
            f'RebuildSummariesManager.run() emitted an unexpected summary log: {matching}'
        )


# ---------------------------------------------------------------------------
# N+1 fix step-7: detect_stale_summaries uses bulk get_all_valid_edges
# ---------------------------------------------------------------------------

class TestDetectStaleSummariesBulk:
    """detect_stale_summaries uses get_all_valid_edges (one query) not N per-entity queries."""

    @pytest.mark.asyncio
    async def test_calls_get_all_valid_edges_once_not_per_entity(self, mock_config, make_backend, make_edge_backend):
        """detect_stale_summaries calls get_all_valid_edges exactly once (not N times)."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factA'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'factB'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'factB', 'name': 'edge2'}],
        })
        backend.get_valid_edges_for_node = AsyncMock()
        await backend.detect_stale_summaries(group_id='test')
        backend.get_all_valid_edges.assert_awaited_once()
        backend.get_valid_edges_for_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_correctly_identifies_stale_with_bulk_edges(self, mock_config, make_backend, make_edge_backend):
        """Stale entity is returned; clean entity is not — using bulk-fetched edges."""
        # uuid-1 is stale (summary has old fact), uuid-2 is clean
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'old fact'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'current fact'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'current fact', 'name': 'edge2'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert len(result) == 1
        assert result[0]['uuid'] == 'uuid-1'

    @pytest.mark.asyncio
    async def test_empty_summary_entities_skipped_with_bulk_path(self, mock_config, make_backend, make_edge_backend):
        """Empty-summary entity is skipped even with bulk edges data source."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': ''},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'some fact', 'name': 'edge1'}],
        })
        result = await backend.detect_stale_summaries(group_id='test')
        assert result == []


# ---------------------------------------------------------------------------
# N+1 fix step-9: rebuild_entity_summaries parallel + no re-fetch
# ---------------------------------------------------------------------------

@pytest.fixture
def two_entity_backend(mock_config, make_backend, make_edge_backend):
    """Shared backend pre-configured with the canonical Alice/Bob two-entity setup.

    Provides:
      - make_backend(mock_config) instantiation (GraphitiBackend with mocked client)
      - list_entity_nodes returning Alice (uuid-1/stale1) and Bob (uuid-2/stale2)
      - get_all_valid_edges returning current1/current2 edges for each entity

    Function-scoped (pytest default) so each test gets a fresh backend with fresh
    AsyncMock.await_count counters — important for await_count assertions in both
    TestRebuildEntitySummariesParallel and TestRebuildEntitySummariesCancellation.

    The list_entity_nodes and get_all_valid_edges mocks are exercised by force=True
    consumers (TestRebuildEntitySummariesParallel.test_partial_failure_in_update_does_not_cancel_others
    and all TestRebuildEntitySummariesCancellation tests except test_cancelled_error_propagates_force_false).
    force=False consumers may intentionally supersede those lower-level mocks by stubbing
    _detect_stale_summaries_with_edges directly (see test_cancelled_error_propagates_force_false).

    Tests supply their own update_node_summary side_effect to exercise the specific
    scenario under test.
    """
    return make_edge_backend(make_backend(mock_config), nodes=[
        {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale1'},
        {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'stale2'},
    ], edges={
        'uuid-1': [{'uuid': 'e1', 'fact': 'current1', 'name': 'edge1'}],
        'uuid-2': [{'uuid': 'e2', 'fact': 'current2', 'name': 'edge2'}],
    })


class TestRebuildEntitySummariesParallel:
    """rebuild_entity_summaries uses _rebuild_entity_from_edges (no re-fetch) + asyncio.gather."""

    @pytest.mark.asyncio
    async def test_non_force_does_not_call_get_valid_edges_for_node(self, mock_config, make_backend, make_edge_backend):
        """Non-force path: get_valid_edges_for_node is never called (edges come from bulk fetch)."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale1'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'stale2'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'current2', 'name': 'edge2'}],
        })
        backend.update_node_summary = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock()

        result = await backend.rebuild_entity_summaries(group_id='test')

        backend.get_valid_edges_for_node.assert_not_awaited()
        assert result['rebuilt'] == 2

    @pytest.mark.asyncio
    async def test_force_calls_get_all_valid_edges_once_not_refresh_entity_summary(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Force path: get_all_valid_edges called once; refresh_entity_summary never called."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'ok'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'also ok'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'fact1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'fact2', 'name': 'edge2'}],
        })
        backend.update_node_summary = AsyncMock()
        backend.refresh_entity_summary = AsyncMock()

        result = await backend.rebuild_entity_summaries(group_id='test', force=True)

        backend.get_all_valid_edges.assert_awaited_once_with(group_id='test')
        backend.refresh_entity_summary.assert_not_awaited()
        assert result['total_entities'] == 2
        assert result['rebuilt'] == 2

    @pytest.mark.asyncio
    async def test_force_bulk_fetch_called_exactly_once_and_updates_all_entities(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Force path: bulk fetches run exactly once and every entity is updated.

        Guards three independent invariants under the force path:
        1. list_entity_nodes is awaited exactly once (single bulk node fetch).
        2. get_all_valid_edges is awaited exactly once with the forwarded group_id
           (single bulk edge fetch, correctly scoped).
        3. update_node_summary is awaited once per entity and each entity receives
           its own edge data — uses three distinct entities with distinct facts so
           a regression that reused uuid-1 data for every call would fail the
           per-uuid new_summary assertions below.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'summary-1'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'summary-2'},
            {'uuid': 'uuid-3', 'name': 'Carol', 'summary': 'summary-3'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'fact-1', 'name': 'rel-1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'fact-2', 'name': 'rel-2'}],
            'uuid-3': [{'uuid': 'e3', 'fact': 'fact-3', 'name': 'rel-3'}],
        })
        backend.update_node_summary = AsyncMock()

        result = await backend.rebuild_entity_summaries(group_id='test', force=True)

        # list_entity_nodes must be called exactly once — no per-entity re-fetch
        backend.list_entity_nodes.assert_awaited_once()
        backend.get_all_valid_edges.assert_awaited_once_with(group_id='test')
        assert result['total_entities'] == 3
        assert result['rebuilt'] == 3
        assert backend.update_node_summary.await_count == 3

        # Each entity must carry its own edge data, not uuid-1's data for every entry
        by_uuid = {d['uuid']: d for d in result['details']}
        assert by_uuid['uuid-1']['new_summary'] == 'fact-1'
        assert by_uuid['uuid-2']['new_summary'] == 'fact-2'
        assert by_uuid['uuid-3']['new_summary'] == 'fact-3'

    @pytest.mark.asyncio
    async def test_concurrent_all_five_entities_processed(self, mock_config, make_backend, make_edge_backend):
        """With 5 target entities, all 5 get update_node_summary calls (parallel processing)."""
        n = 5
        entities = [
            {'uuid': f'uuid-{i}', 'name': f'Entity{i}', 'summary': f'stale{i}'}
            for i in range(n)
        ]
        edges = {
            f'uuid-{i}': [{'uuid': f'e{i}', 'fact': f'current{i}', 'name': 'edge'}]
            for i in range(n)
        }
        backend = make_edge_backend(make_backend(mock_config), nodes=entities, edges=edges)
        backend.update_node_summary = AsyncMock()

        result = await backend.rebuild_entity_summaries(group_id='test')

        assert backend.update_node_summary.await_count == n
        assert result['rebuilt'] == n
        assert result['errors'] == 0

    @pytest.mark.asyncio
    async def test_force_path_passes_old_summary_no_get_node_text(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Force path: old_summary from list_entity_nodes is passed to
        _rebuild_entity_from_edges and get_node_text is never called."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'alice summary'},
            {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'bob summary'},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'fact1', 'name': 'edge1'}],
            'uuid-2': [{'uuid': 'e2', 'fact': 'fact2', 'name': 'edge2'}],
        })
        backend.get_node_text = AsyncMock()
        backend.update_node_summary = AsyncMock()

        captured_calls, _ = _make_rebuild_spy(backend)

        result = await backend.rebuild_entity_summaries(group_id='test', force=True)

        assert result['rebuilt'] == 2
        # old_summary must be threaded from list_entity_nodes summary field
        by_uuid = {c['uuid']: c for c in captured_calls}
        assert by_uuid['uuid-1']['old_summary'] == 'alice summary'
        assert by_uuid['uuid-2']['old_summary'] == 'bob summary'
        # get_node_text must never be called
        backend.get_node_text.assert_not_awaited()
        # end-to-end: details dict surfaces old_summary and new_summary per entity
        by_detail = {d['uuid']: d for d in result['details']}
        assert by_detail['uuid-1']['old_summary'] == 'alice summary'
        assert by_detail['uuid-1']['new_summary'] == 'fact1'
        assert by_detail['uuid-2']['old_summary'] == 'bob summary'
        assert by_detail['uuid-2']['new_summary'] == 'fact2'

    @pytest.mark.asyncio
    async def test_force_path_passes_empty_string_old_summary(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Force path: old_summary='' (FalkorDB NULL→'' normalisation) is passed
        as empty string — NOT None — to _rebuild_entity_from_edges."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': ''},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'new fact', 'name': 'edge1'}],
        })
        backend.get_node_text = AsyncMock()
        backend.update_node_summary = AsyncMock()

        captured_calls, _ = _make_rebuild_spy(backend)

        result = await backend.rebuild_entity_summaries(group_id='test', force=True)

        assert result['rebuilt'] == 1
        assert captured_calls[0]['old_summary'] == ''
        backend.get_node_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_force_path_passes_old_summary_no_get_node_text(
        self, mock_config, make_backend
    ):
        """Non-force path: old_summary from _detect_stale_summaries_with_edges stale dict
        is passed to _rebuild_entity_from_edges and get_node_text is never called."""
        backend = make_backend(mock_config)
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=StaleSummaryResult(
            stale=[
                {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'alice stale',
                 'duplicate_count': 0, 'stale_line_count': 1, 'valid_fact_count': 1,
                 'summary_line_count': 1},
                {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'bob stale',
                 'duplicate_count': 0, 'stale_line_count': 1, 'valid_fact_count': 1,
                 'summary_line_count': 1},
            ],
            all_edges={
                'uuid-1': [{'uuid': 'e1', 'fact': 'alice current', 'name': 'edge1'}],
                'uuid-2': [{'uuid': 'e2', 'fact': 'bob current', 'name': 'edge2'}],
            },
            total_count=2,
        ))
        backend.get_node_text = AsyncMock()
        backend.update_node_summary = AsyncMock()

        captured_calls, _ = _make_rebuild_spy(backend)

        result = await backend.rebuild_entity_summaries(group_id='test', force=False)

        assert result['rebuilt'] == 2
        by_uuid = {c['uuid']: c for c in captured_calls}
        assert by_uuid['uuid-1']['old_summary'] == 'alice stale'
        assert by_uuid['uuid-2']['old_summary'] == 'bob stale'
        backend.get_node_text.assert_not_awaited()
        # end-to-end: details dict surfaces old_summary and new_summary per entity
        by_detail = {d['uuid']: d for d in result['details']}
        assert by_detail['uuid-1']['old_summary'] == 'alice stale'
        assert by_detail['uuid-1']['new_summary'] == 'alice current'
        assert by_detail['uuid-2']['old_summary'] == 'bob stale'
        assert by_detail['uuid-2']['new_summary'] == 'bob current'

    @pytest.mark.asyncio
    async def test_partial_failure_in_update_does_not_cancel_others(
        self, two_entity_backend
    ):
        """If update_node_summary fails for one entity, others still complete.

        asyncio.gather with return_exceptions=True ensures partial failures are
        captured rather than propagated, so the gather completes for all entities.
        """
        backend = two_entity_backend
        # First entity's write fails, second succeeds
        backend.update_node_summary = AsyncMock(side_effect=[
            RuntimeError('FalkorDB timeout'),
            None,
        ])

        result = await backend.rebuild_entity_summaries(group_id='test')

        assert result['rebuilt'] == 1
        assert result['errors'] == 1


# ---------------------------------------------------------------------------
# task-432 step-1: _rebuild_entity_from_edges accepts old_summary kwarg
# ---------------------------------------------------------------------------

class TestRebuildEntityFromEdgesOldSummary:
    """_rebuild_entity_from_edges accepts old_summary kwarg and skips get_node_text."""

    @pytest.mark.asyncio
    async def test_rebuild_entity_from_edges_uses_passed_old_summary(
        self, mock_config, make_backend
    ):
        """When old_summary kwarg is provided, it is used in the result dict and
        get_node_text is NOT called."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock()
        backend.update_node_summary = AsyncMock()

        edges = [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]
        result = await backend._rebuild_entity_from_edges(
            'uuid-1', 'Alice', edges, group_id='test', old_summary='prior summary'
        )

        assert result['old_summary'] == 'prior summary'
        assert result['new_summary'] == 'current fact'
        assert result['uuid'] == 'uuid-1'
        assert result['name'] == 'Alice'
        assert result['edge_count'] == 1
        backend.get_node_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_duplicate_edges_deduplicated_in_summary_but_counted_in_edge_count(
        self, mock_config, make_backend
    ):
        """edge_count reflects raw edge count; new_summary deduplicates via _canonical_facts.

        3 raw edges with 1 duplicate → edge_count=3, but only 2 unique facts in new_summary.
        This documents that edge_count and unique-fact count are intentionally different
        concepts so future maintainers do not misread edge_count as a deduplicated count.
        """
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock()
        backend.update_node_summary = AsyncMock()

        edges = [
            {'uuid': 'e1', 'fact': 'Alice knows Bob', 'name': 'knows'},
            {'uuid': 'e2', 'fact': 'Alice knows Bob', 'name': 'knows'},   # duplicate
            {'uuid': 'e3', 'fact': 'Alice works at Acme', 'name': 'works_at'},
        ]
        result = await backend._rebuild_entity_from_edges(
            'uuid-1', 'Alice', edges, group_id='test', old_summary='prior'
        )

        assert result['edge_count'] == 3
        assert result['new_summary'] == 'Alice knows Bob\nAlice works at Acme'


# ---------------------------------------------------------------------------
# task-484: CancelledError must propagate, not be swallowed as per-entity error
# ---------------------------------------------------------------------------

class TestRebuildEntitySummariesCancellation:
    """CancelledError raised inside gather must propagate, not be silently captured.

    asyncio.gather(return_exceptions=True) captures *all* BaseException subclasses as
    result values — including asyncio.CancelledError, which is a BaseException but NOT
    an Exception in Python 3.8+. The current (buggy) guard ``isinstance(r, BaseException)``
    therefore treats cancellation signals as ordinary per-entity failures and converts them
    into ``{'status': 'error'}`` detail entries, silently swallowing the shutdown signal.

    The fix uses the shared 'two-tier check' via propagate_cancellations
    (fused_memory.utils.async_utils):
      - Pass 1 (propagate_cancellations): re-raises any value that is a BaseException
        but NOT an Exception (CancelledError, KeyboardInterrupt, SystemExit).
      - The per-entity accumulator loop then uses ``isinstance(r, Exception)`` so only
        application-level failures are recorded as error detail entries.
    """

    @staticmethod
    def _assert_single_cancellation_warning(
        caplog, group_id, rebuilt_so_far=None, errors_so_far=None
    ):
        """Assert exactly one WARNING from graphiti_client contains expected substrings.

        Returns the warning message. rebuilt_so_far and errors_so_far, if provided,
        are asserted against the rendered message values.
        """
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'fused_memory.backends.graphiti_client'
        ]
        assert len(warning_records) == 1, (
            f'Expected 1 WARNING from graphiti_client, got {len(warning_records)}: {warning_records}'
        )
        msg = warning_records[0].getMessage()
        assert 'rebuild_entity_summaries' in msg
        assert 'cancellation' in msg
        assert f'group={group_id}' in msg
        if rebuilt_so_far is not None:
            assert f'rebuilt_so_far={rebuilt_so_far}' in msg
        if errors_so_far is not None:
            assert f'errors_so_far={errors_so_far}' in msg
        return msg

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self, two_entity_backend):
        """CancelledError raised by one entity must propagate out of rebuild_entity_summaries.

        If asyncio.CancelledError is captured by gather(return_exceptions=True) and the
        accumulator treats it as a per-entity error (the current bug), the method returns a
        dict and pytest.raises will report DID NOT RAISE — confirming the test fails before
        the fix and passes after.

        Uses force=True (simpler mock surface: list_entity_nodes + get_all_valid_edges)
        matching the pattern from TestRebuildEntitySummariesParallel.
        """
        backend = two_entity_backend
        # First entity's rebuild raises CancelledError; second would succeed
        backend.update_node_summary = AsyncMock(
            side_effect=[asyncio.CancelledError(), None]
        )

        with pytest.raises(asyncio.CancelledError):
            await backend.rebuild_entity_summaries(group_id='grp-cancel-warn', force=True)

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_alongside_other_errors(
        self, two_entity_backend, caplog
    ):
        """CancelledError must take precedence over per-entity RuntimeErrors in the same batch.

        When gather(return_exceptions=True) returns a mix of Exception subclasses and
        CancelledError, the propagation pass (which runs before per-entity accumulation)
        must still re-raise the CancelledError even if a RuntimeError appears first in
        the results list. This guards against a regression where someone reorders the
        guard branches and accidentally promotes RuntimeError accounting before the
        cancellation check.

        The propagation pass scans *all* results before any per-entity bookkeeping, so
        even if RuntimeError occupies the first slot, CancelledError in the second slot
        is still detected and re-raised.

        The WARNING log must also be emitted on this mixed-error path — a regression
        dropping the warning from this branch would otherwise go undetected.
        """
        backend = two_entity_backend
        # First entity raises a normal RuntimeError; second raises CancelledError
        backend.update_node_summary = AsyncMock(
            side_effect=[RuntimeError('per-entity failure'), asyncio.CancelledError()]
        )

        with (
            caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'),
            pytest.raises(asyncio.CancelledError),
        ):
            await backend.rebuild_entity_summaries(group_id='grp-cancel-warn', force=True)

        # After CancelledError propagated, verify both update_node_summary calls were
        # attempted by gather.  Pass 1 (propagate_cancellations in
        # fused_memory.utils.async_utils) scans the full results list before any
        # per-entity bookkeeping, so the RuntimeError in slot 1 is captured by
        # gather(return_exceptions=True) but NEVER reaches the per-entity error
        # accumulator (Pass 2) — Pass 1 raises first.  The await_count==2 assertion
        # proves gather scheduled both coroutines and that we are testing the
        # post-gather propagation path, not a pre-gather short-circuit.
        assert backend.update_node_summary.await_count == 2

        self._assert_single_cancellation_warning(
            caplog, group_id='grp-cancel-warn', rebuilt_so_far=0, errors_so_far=0
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_logs_warning_before_propagating(
        self, two_entity_backend, caplog
    ):
        """A WARNING is emitted with group_id and progress counters before CancelledError propagates.

        The warning must contain 'rebuild_entity_summaries', 'cancellation', and the group_id
        so operators can identify mid-flight cancellations in logs. rebuilt_so_far=0 and
        errors_so_far=0 because counters are only incremented in Pass 2, which never executes
        when CancelledError is found in Pass 1.
        """
        backend = two_entity_backend
        # First entity's rebuild raises CancelledError; second would succeed
        backend.update_node_summary = AsyncMock(
            side_effect=[asyncio.CancelledError(), None]
        )

        with (
            caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'),
            pytest.raises(asyncio.CancelledError),
        ):
            await backend.rebuild_entity_summaries(group_id='grp-cancel-warn', force=True)

        self._assert_single_cancellation_warning(
            caplog, group_id='grp-cancel-warn', rebuilt_so_far=0, errors_so_far=0
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_in_second_slot_still_logs_warning(
        self, two_entity_backend, caplog
    ):
        """CancelledError in the second slot is detected by propagate_cancellations scan.

        All prior CancelledError tests place the error in slot 0 (side_effect=[CancelledError(),
        None]).  This test exercises slot 1 (side_effect=[None, CancelledError()]) where the
        first entity succeeds and CancelledError appears only in the second position.

        propagate_cancellations scans *all* results before any per-entity bookkeeping, so
        CancelledError in any slot must still be detected and re-raised.  Explicit coverage
        here guards against a regression where the scan is shortened to check only the first
        result.

        rebuilt_so_far=0 and errors_so_far=0 because counters are only incremented in Pass 2,
        which never executes when CancelledError is found in Pass 1.
        """
        backend = two_entity_backend
        # First entity's rebuild succeeds; second raises CancelledError
        backend.update_node_summary = AsyncMock(
            side_effect=[None, asyncio.CancelledError()]
        )

        with (
            caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'),
            pytest.raises(asyncio.CancelledError),
        ):
            await backend.rebuild_entity_summaries(group_id='grp-cancel-warn', force=True)

        self._assert_single_cancellation_warning(
            caplog, group_id='grp-cancel-warn', rebuilt_so_far=0, errors_so_far=0
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_force_false(self, two_entity_backend):
        """CancelledError must propagate through the force=False code path.

        The two existing cancellation tests (test_cancelled_error_propagates and
        test_cancelled_error_propagates_alongside_other_errors) both use force=True
        (simpler mock surface: list_entity_nodes + get_all_valid_edges).  This test
        exercises the force=False branch, which routes through
        _detect_stale_summaries_with_edges instead, to confirm that both code paths
        share the same gather + two-tier propagation guarantee.

        Even though both branches converge on the same asyncio.gather + two-tier check,
        an explicit test guards against future divergence — e.g. if the force=False path
        were ever refactored to bypass gather entirely.

        _detect_stale_summaries_with_edges is mocked directly (rather than relying on
        the natural detection path through the fixture's list_entity_nodes / get_all_valid_edges)
        to isolate the test surface to the gather + two-tier check, mirroring the pattern
        in TestRebuildEntitySummaries.test_rebuilds_only_stale_entities (line ~259).
        """
        backend = two_entity_backend
        # Intentionally supersede the fixture's lower-level list_entity_nodes /
        # get_all_valid_edges mocks by stubbing _detect_stale_summaries_with_edges
        # directly.  This isolates the test surface to the gather + two-tier check
        # path, mirroring the pattern in
        # TestRebuildEntitySummaries.test_rebuilds_only_stale_entities (line ~259).
        backend._detect_stale_summaries_with_edges = AsyncMock(
            return_value=StaleSummaryResult(
                stale=[
                    {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale1',
                     'duplicate_count': 0, 'stale_line_count': 1,
                     'valid_fact_count': 1, 'summary_line_count': 1},
                    {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'stale2',
                     'duplicate_count': 0, 'stale_line_count': 1,
                     'valid_fact_count': 1, 'summary_line_count': 1},
                ],
                all_edges={
                    'uuid-1': [{'uuid': 'e1', 'fact': 'current1', 'name': 'edge1'}],
                    'uuid-2': [{'uuid': 'e2', 'fact': 'current2', 'name': 'edge2'}],
                },
                total_count=2,
            )
        )
        # First entity's rebuild raises CancelledError; second would succeed
        backend.update_node_summary = AsyncMock(
            side_effect=[asyncio.CancelledError(), None]
        )

        with pytest.raises(asyncio.CancelledError):
            await backend.rebuild_entity_summaries(group_id='grp-cancel-warn', force=False)

        # After CancelledError propagated, verify the force=False gather path was actually
        # exercised: _detect_stale_summaries_with_edges was awaited exactly once (bulk fetch
        # of stale entities + their edges) and update_node_summary was awaited twice (one per
        # entity, scheduled by asyncio.gather before the Pass 1 propagation check).  This
        # guards against a future refactor where CancelledError could escape from a different
        # code path (e.g. before gather is reached) while the pytest.raises still reports DID
        # RAISE — mirroring the rationale in test_cancelled_error_propagates_alongside_other_errors.
        assert backend._detect_stale_summaries_with_edges.await_count == 1
        assert backend.update_node_summary.await_count == 2

    @pytest.mark.asyncio
    async def test_runtime_error_still_accumulates_in_errors(self, two_entity_backend):
        """Regression guard: Exception subclasses must still accumulate after the BaseException→Exception narrowing.

        The task-484 fix changed the per-entity accumulator guard from
        ``isinstance(r, BaseException)`` to ``isinstance(r, Exception)`` so that
        CancelledError (a BaseException but NOT an Exception) is excluded from the
        per-entity error bookkeeper and handled only by Pass 1 (propagate_cancellations
        in fused_memory.utils.async_utils).

        This narrowing is intentional but must not accidentally exclude ordinary
        application-level failures (ValueError, etc.) from Pass 2.
        If someone over-narrows the guard — e.g. to
        ``isinstance(r, RuntimeError)`` — ordinary failures from other Exception subclasses
        would silently disappear from the result dict.

        Uses ValueError (not RuntimeError) to prove the guard is isinstance(r, Exception)
        and is not over-narrowed to RuntimeError.  RuntimeError accumulation is already
        covered by test_partial_failure_in_update_does_not_cancel_others.

        Uses force=True (same mock surface as the other tests in this class) because
        both branches converge on the same Pass 2 accumulator.  The force=False branch's
        ValueError accumulation is already covered by test_partial_failure_continues
        (TestRebuildEntitySummaries class).
        """
        backend = two_entity_backend
        # First entity's rebuild raises ValueError; second succeeds
        backend.update_node_summary = AsyncMock(
            side_effect=[ValueError('per-entity boom'), None]
        )

        result = await backend.rebuild_entity_summaries(group_id='test', force=True)

        assert result['errors'] == 1
        assert result['rebuilt'] == 1
        # Pin down the full result counter shape: without these, a regression where both
        # counters incremented for the same entity (e.g. total_entities=1/skipped=1) would
        # still satisfy errors==1/rebuilt==1 and slip through.
        assert result['total_entities'] == 2
        assert result['skipped'] == 0
        error_detail = next((d for d in result['details'] if d['status'] == 'error'), None)
        assert error_detail is not None, f"expected an 'error' entry in details; got {result['details']}"
        assert 'per-entity boom' in error_detail['error']


# ---------------------------------------------------------------------------
# Task-511: replace bare assert isinstance(r, dict) with explicit TypeError guard
# ---------------------------------------------------------------------------

class TestRebuildEntitySummariesTypeGuard:
    """Regression tests for the explicit TypeError guard in the per-entity accumulation loop.

    The production code in graphiti_client.py's rebuild_entity_summaries contains a loop
    that gathers results from _rebuild_entity_from_edges.  In the else branch (non-exception
    result) the code originally used ``assert isinstance(r, dict)`` purely for Pyright type
    narrowing.  That assert is stripped when Python runs with ``-O``/``PYTHONOPTIMIZE``,
    leaving a silent footgun: a non-dict return would propagate silently until an
    AttributeError surfaced on ``r.get(...)`` far from the origin.

    Task-511 replaces the assert with an explicit ``if not isinstance(r, dict): raise TypeError(...)``
    so the contract check is unconditional.  Pyright still narrows ``r`` to ``dict`` after
    a conditional raise, so the type-narrowing motivation is preserved.

    See: fused-memory/src/fused_memory/backends/graphiti_client.py – per-entity accumulation loop.
    """

    @pytest.mark.asyncio
    async def test_non_dict_return_raises_typeerror(self, two_entity_backend):
        """_rebuild_entity_from_edges returning a non-dict value must raise TypeError.

        Patches _rebuild_entity_from_edges to return None (a non-dict) and asserts
        that rebuild_entity_summaries raises TypeError whose message contains:
          - the type name ('NoneType')
          - the entity uuid ('uuid-1')
          - the entity name ('Alice')

        Also verifies fail-fast behaviour: asyncio.gather awaits all tasks (both
        entities are gathered), then the processing loop raises on the first bad
        result without accumulating any details.  The await_count of 2 pins this
        contract — if the guard were moved inside _rebuild_one the count would drop
        to 1 (short-circuit before gather completes).
        """
        backend = two_entity_backend
        backend._rebuild_entity_from_edges = AsyncMock(return_value=None)

        with pytest.raises(TypeError) as exc_info:
            await backend.rebuild_entity_summaries(group_id='test', force=True)

        msg = str(exc_info.value)
        assert 'NoneType' in msg
        assert 'uuid-1' in msg
        assert 'Alice' in msg
        # Both entities are gathered before the processing loop runs — the TypeError
        # is raised during result accumulation, not during the gather phase.
        assert backend._rebuild_entity_from_edges.await_count == 2

    @pytest.mark.asyncio
    async def test_non_none_non_dict_return_raises_typeerror(self, two_entity_backend):
        """A non-None non-dict return (e.g. a list) must also trigger the TypeError guard.

        This prevents a future contributor from accidentally special-casing ``None``
        in the guard and missing other invalid return types.  The type name in the
        message must reflect the actual type of the returned value.
        """
        backend = two_entity_backend
        backend._rebuild_entity_from_edges = AsyncMock(return_value=[])

        with pytest.raises(TypeError) as exc_info:
            await backend.rebuild_entity_summaries(group_id='test', force=True)

        msg = str(exc_info.value)
        assert 'list' in msg
        assert 'uuid-1' in msg
        assert 'Alice' in msg
        # Gather still runs both tasks; TypeError is raised in the result loop.
        assert backend._rebuild_entity_from_edges.await_count == 2
