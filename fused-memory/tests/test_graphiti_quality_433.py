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
    GraphitiBackend,
    StaleSummaryResult,
)

# ---------------------------------------------------------------------------
# step-1: StaleSummaryResult named tuple with backward-compat tuple unpacking
# ---------------------------------------------------------------------------

class TestStaleSummaryResult:
    """StaleSummaryResult has named attrs and supports 3-tuple unpacking."""

    def test_named_attribute_stale(self):
        """StaleSummaryResult.stale holds the stale list."""
        stale_list = [{'uuid': 'u1', 'name': 'Alice'}]
        edges = {'u1': [{'fact': 'fact1'}]}
        result = StaleSummaryResult(stale=stale_list, all_edges=edges, total_count=5)
        assert result.stale is stale_list

    def test_named_attribute_all_edges(self):
        """StaleSummaryResult.all_edges holds the edges dict."""
        stale_list: list[dict] = []
        edges = {'u1': [{'fact': 'fact1'}]}
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

        The rename edges → all_edges was applied in task 438. This test locks
        in that the old alias is truly absent, so any accidental re-exposure
        would be caught immediately.
        """
        result = StaleSummaryResult(stale=[], all_edges={}, total_count=0)
        assert not hasattr(result, 'edges')

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_returns_named_result(self, mock_config, make_backend):
        """_detect_stale_summaries_with_edges returns StaleSummaryResult with named access."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'stale summary'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'u1': [{'fact': 'fresh fact'}],
        })
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
            {'fact': '   '},    # whitespace-only — filtered out
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
            {'fact': '\t\t'},          # tabs only — filtered
            {'fact': '\n'},            # newline only — filtered
            {'fact': '  \t\n  '},     # mixed whitespace — filtered
            {'fact': ' '},             # single space — filtered
            {'fact': '  hello  '},    # real content with surrounding space — kept (raw value)
            {'fact': 'A knows B'},    # plain fact — kept
        ]
        result = GraphitiBackend._canonical_facts(edges)
        assert result == ['  hello  ', 'A knows B']


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
            await backend.refresh_entity_summary(
                'u1', group_id='test', name='Alice'
            )

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
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'fact': 'Alice knows Bob'},
        ])
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
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'fact': 'Alice knows Bob'},
        ])
        backend.update_node_summary = AsyncMock()

        result = await backend.refresh_entity_summary('u1', group_id='test')

        backend.get_node_text.assert_called_once()
        assert result['name'] == 'Alice'
        assert result['old_summary'] == 'old summary'


# ---------------------------------------------------------------------------
# step-7: rebuild_entity_summaries(force=True, dry_run=True) skips edge fetch
# ---------------------------------------------------------------------------

class TestRebuildEntitySummariesForceDryRun:
    """rebuild_entity_summaries(force=True, dry_run=True) skips get_all_valid_edges."""

    @pytest.mark.asyncio
    async def test_force_dry_run_does_not_call_get_all_valid_edges(self, mock_config, make_backend):
        """When force=True and dry_run=True, get_all_valid_edges is NOT called."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'summary A'},
            {'uuid': 'u2', 'name': 'Bob', 'summary': 'summary B'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={})

        await backend.rebuild_entity_summaries(
            group_id='test', force=True, dry_run=True
        )

        backend.get_all_valid_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_dry_run_returns_correct_aggregate(self, mock_config, make_backend):
        """When force=True and dry_run=True, result has correct structure."""
        backend = make_backend(mock_config)
        entities = [
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'summary A'},
            {'uuid': 'u2', 'name': 'Bob', 'summary': 'summary B'},
            {'uuid': 'u3', 'name': 'Carol', 'summary': 'summary C'},
        ]
        backend.list_entity_nodes = AsyncMock(return_value=entities)
        backend.get_all_valid_edges = AsyncMock(return_value={})

        result = await backend.rebuild_entity_summaries(
            group_id='test', force=True, dry_run=True
        )

        assert result['total_entities'] == 3
        assert result['stale_entities'] == 3  # force=True targets all
        assert result['skipped'] == 3  # dry_run=True skips all
        assert result['rebuilt'] == 0
        assert result['errors'] == 0
        assert len(result['details']) == 3
        for detail in result['details']:
            assert detail['status'] == 'skipped_dry_run'

    @pytest.mark.asyncio
    async def test_force_no_dry_run_calls_get_all_valid_edges(self, mock_config, make_backend):
        """Positive complement: force=True, dry_run=False calls get_all_valid_edges exactly once.

        This is the paired positive case for test_force_dry_run_does_not_call_get_all_valid_edges.
        When dry_run=False the edges ARE needed for the actual rebuild, so
        get_all_valid_edges must be called before processing entities.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'summary A'},
            {'uuid': 'u2', 'name': 'Bob', 'summary': 'summary B'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={})
        # Mock the inner rebuild to avoid touching real write path
        backend._rebuild_entity_from_edges = AsyncMock(return_value={
            'uuid': 'u1', 'name': 'Alice',
            'old_summary': '', 'new_summary': '', 'edge_count': 0,
        })

        await backend.rebuild_entity_summaries(
            group_id='test', force=True, dry_run=False
        )

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
        from fused_memory.backends.graphiti_client import StaleSummaryResult

        backend = make_backend(mock_config)
        stale_list = [{'uuid': 'u1', 'name': 'Alice', 'summary': 'old'}]
        all_edges = {'u1': [{'fact': 'Alice knows Bob'}]}
        # total_count=10 means 10 entities exist but only 1 is stale
        detect_result = StaleSummaryResult(
            stale=stale_list, all_edges=all_edges, total_count=10
        )
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=detect_result)
        backend._rebuild_entity_from_edges = AsyncMock(return_value={
            'uuid': 'u1', 'name': 'Alice',
            'old_summary': 'old', 'new_summary': 'Alice knows Bob', 'edge_count': 1,
        })

        result = await backend.rebuild_entity_summaries(group_id='test', force=False)

        assert result['total_entities'] == 10   # flows from total_count=10
        assert result['stale_entities'] == 1    # only 1 stale
        assert result['rebuilt'] == 1
        assert result['skipped'] == 0
        assert result['errors'] == 0

    @pytest.mark.asyncio
    async def test_force_false_dry_run_total_entities_from_detect(self, mock_config, make_backend):
        """force=False, dry_run=True: total_entities still comes from detect step."""
        from fused_memory.backends.graphiti_client import StaleSummaryResult

        backend = make_backend(mock_config)
        stale_list = [
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'old A'},
            {'uuid': 'u2', 'name': 'Bob', 'summary': 'old B'},
        ]
        detect_result = StaleSummaryResult(
            stale=stale_list, all_edges={}, total_count=7
        )
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=detect_result)

        result = await backend.rebuild_entity_summaries(
            group_id='test', force=False, dry_run=True
        )

        assert result['total_entities'] == 7
        assert result['stale_entities'] == 2
        assert result['skipped'] == 2
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
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'stale summary'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={})
        # Simulate _rebuild_entity_from_edges failing for this entity
        backend._rebuild_entity_from_edges = AsyncMock(
            side_effect=RuntimeError('boom')
        )

        result = await backend.rebuild_entity_summaries(
            group_id='test', force=True, dry_run=False
        )

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
    async def test_partial_success_one_error_one_rebuilt(self, mock_config, make_backend):
        """asyncio.gather returns a mix of exceptions and successes without aborting.

        With two entities, the first raising and the second succeeding, the result
        should record errors=1 and rebuilt=1 with details in target order (u1 first,
        u2 second). This exercises the zip(targets, results, strict=True) accumulator
        loop that is the core value of return_exceptions=True.
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'stale summary'},
            {'uuid': 'u2', 'name': 'Bob', 'summary': 'stale summary 2'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={})
        backend._rebuild_entity_from_edges = AsyncMock(
            side_effect=[
                RuntimeError('boom'),
                {'uuid': 'u2', 'name': 'Bob', 'old_summary': '', 'new_summary': 'rebuilt', 'edge_count': 0},
            ]
        )

        result = await backend.rebuild_entity_summaries(
            group_id='test', force=True, dry_run=False
        )

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
        assert ok_detail['old_summary'] == ''
        assert ok_detail['new_summary'] == 'rebuilt'
        assert ok_detail['edge_count'] == 0


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
        """
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'A knows B'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'u1': [{'fact': '   '}, {'fact': 'A knows B'}],
        })

        result = await backend._detect_stale_summaries_with_edges(group_id='test')

        assert result.stale == [], (
            "Entity should NOT be flagged stale when its only non-whitespace "
            "fact matches the stored summary."
        )
