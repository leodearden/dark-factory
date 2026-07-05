"""Tests for the degenerate "tasks N" placeholder Graphiti node sweep (task 2107).

Stage 1 (MemoryConsolidator) has no deterministic sweep that keeps "tasks N"
placeholder entity nodes in sync for terminal tasks — those nodes are an
unintended byproduct of graphiti-core's LLM entity extraction. This module
adds a small deterministic post-processor that DELETES degenerate ("tasks
{id}" nodes with zero valid edges) placeholder nodes for terminal (done AND
cancelled) tasks.

Covers:
- extract_terminal_task_ids: pure helper that reads FilteredTaskTree.done_tasks
  + cancelled_tasks (never active_tasks) and returns deduped str task ids.
- sweep_degenerate_task_nodes: async helper that finds exact-name "tasks {id}"
  nodes via GraphitiBackend.find_duplicate_entity_nodes and deletes only the
  ones with edge_count == 0, best-effort.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.reconciliation.degenerate_task_node_sweep import (
    extract_terminal_task_ids,
    sweep_degenerate_task_nodes,
)
from fused_memory.reconciliation.task_filter import FilteredTaskTree


def _make_memory_service() -> MagicMock:
    """MagicMock memory_service with an AsyncMock .graphiti (mirrors test_merge_entities.py)."""
    memory_service = MagicMock()
    memory_service.graphiti = MagicMock()
    memory_service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
    memory_service.graphiti.delete_entity_node = AsyncMock()
    return memory_service


class TestExtractTerminalTaskIds:
    """extract_terminal_task_ids(tree) combines done_tasks + cancelled_tasks ids.

    active_tasks is never a source (only terminal tasks are swept); ids are
    deduped preserving first-seen order; non-dict elements and dicts missing
    an 'id' key are silently skipped.
    """

    def test_combines_done_and_cancelled_ids_excludes_active(self):
        """Result is done_tasks ids followed by cancelled_tasks ids; active_tasks excluded."""
        tree = FilteredTaskTree(
            active_tasks=[{'id': 999, 'status': 'pending'}],
            done_tasks=[{'id': 148, 'status': 'done'}],
            cancelled_tasks=[{'id': 142, 'status': 'cancelled'}, {'id': 144, 'status': 'cancelled'}],
        )

        result = extract_terminal_task_ids(tree)

        assert result == ['148', '142', '144'], (
            f'Expected done_tasks ids before cancelled_tasks ids, got {result!r}'
        )
        assert '999' not in result, 'active_tasks must never contribute to terminal ids'

    def test_dedups_repeated_ids_preserving_first_seen_order(self):
        """An id appearing in both done_tasks and cancelled_tasks is deduped to its first occurrence."""
        tree = FilteredTaskTree(
            done_tasks=[{'id': 148}],
            cancelled_tasks=[{'id': 148}, {'id': 144}],
        )

        result = extract_terminal_task_ids(tree)

        assert result == ['148', '144'], f'Expected dedup preserving first-seen order, got {result!r}'

    def test_skips_non_dict_elements(self):
        """Non-dict elements in done_tasks/cancelled_tasks are silently skipped."""
        tree = FilteredTaskTree(
            done_tasks=[{'id': 148}, 'not-a-dict', None],  # type: ignore[list-item]
            cancelled_tasks=[142, {'id': 144}],  # type: ignore[list-item]
        )

        result = extract_terminal_task_ids(tree)

        assert result == ['148', '144'], f'Expected non-dict elements skipped, got {result!r}'

    def test_skips_tasks_missing_id_key(self):
        """A dict lacking an 'id' key is skipped — never defaulted to a spurious '0'."""
        tree = FilteredTaskTree(
            done_tasks=[{'id': 148}, {'status': 'done'}],
            cancelled_tasks=[{'title': 'no id here'}, {'id': 144}],
        )

        result = extract_terminal_task_ids(tree)

        assert result == ['148', '144'], f'Expected id-less dicts skipped, got {result!r}'
        assert '0' not in result, "A missing 'id' key must never be coerced to '0'"

    def test_skips_unparseable_id(self):
        """An 'id' value that cannot be coerced to int is skipped (mirrors id_key parse/skip)."""
        tree = FilteredTaskTree(
            done_tasks=[{'id': 148}, {'id': 'abc'}],
            cancelled_tasks=[{'id': None}, {'id': 144}],
        )

        result = extract_terminal_task_ids(tree)

        assert result == ['148', '144'], f'Expected unparseable ids skipped, got {result!r}'

    def test_returns_empty_list_when_tree_is_none(self):
        """A None tree (e.g. harness didn't set filtered_task_tree) returns []."""
        assert extract_terminal_task_ids(None) == []

    def test_returns_empty_list_when_no_done_or_cancelled_tasks(self):
        """A tree with only active_tasks (no done/cancelled) returns []."""
        tree = FilteredTaskTree(active_tasks=[{'id': 1, 'status': 'pending'}])

        assert extract_terminal_task_ids(tree) == []


# --------------------------------------------------------------------------- #
# sweep_degenerate_task_nodes — core behavior
# --------------------------------------------------------------------------- #


class TestSweepDegenerateTaskNodesCore:
    """sweep_degenerate_task_nodes finds exact-name 'tasks {id}' nodes per task id
    and deletes ONLY the ones with edge_count == 0 — a node with >= 1 valid edge
    is never touched (safety invariant)."""

    @pytest.mark.asyncio
    async def test_deletes_only_the_degenerate_node_never_the_healthy_one(self):
        """One degenerate (edge_count=0) + one healthy (edge_count=3) node.

        delete_entity_node must be awaited exactly once, for the degenerate
        node's uuid with group_id=project_id, and never for the healthy node.
        """
        memory_service = _make_memory_service()

        async def fake_find(name, *, group_id):
            if name == 'tasks 142':
                return [{'uuid': 'node-142', 'created_at': 100, 'edge_count': 0}]
            if name == 'tasks 148':
                return [{'uuid': 'node-148', 'created_at': 100, 'edge_count': 3}]
            return []

        memory_service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find)

        stats = await sweep_degenerate_task_nodes(memory_service, 'test_project', ['142', '148'])

        memory_service.graphiti.delete_entity_node.assert_awaited_once_with(
            'node-142', group_id='test_project',
        )
        deleted_uuids = [c.args[0] for c in memory_service.graphiti.delete_entity_node.await_args_list]
        assert 'node-148' not in deleted_uuids, 'A node with edge_count > 0 must never be deleted'
        assert stats == {'scanned': 2, 'degenerate': 1, 'deleted': 1, 'errors': 0}, (
            f'Expected exactly the degenerate node counted+deleted, got stats={stats!r}'
        )


# --------------------------------------------------------------------------- #
# sweep_degenerate_task_nodes — best-effort resilience / edge cases
# --------------------------------------------------------------------------- #


class TestSweepDegenerateTaskNodesBestEffort:
    """A transient backend error scanning or deleting one task id/name must not
    abort the sweep for the remaining ids — it is tallied into stats['errors']
    and the loop continues (mirrors MemoryService._dedup_episode_nodes)."""

    @pytest.mark.asyncio
    async def test_delete_failure_is_swallowed_and_second_delete_still_attempted(self):
        """delete_entity_node raises for the first degenerate node; the second
        degenerate node's delete is still attempted; both count as scanned+degenerate,
        but only the successful one counts as deleted, and the failure counts as an error."""
        memory_service = _make_memory_service()

        async def fake_find(name, *, group_id):
            if name == 'tasks 142':
                return [{'uuid': 'node-142', 'created_at': 100, 'edge_count': 0}]
            if name == 'tasks 144':
                return [{'uuid': 'node-144', 'created_at': 100, 'edge_count': 0}]
            return []

        memory_service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find)
        memory_service.graphiti.delete_entity_node = AsyncMock(
            side_effect=[RuntimeError('lock contention'), None],
        )

        stats = await sweep_degenerate_task_nodes(memory_service, 'test_project', ['142', '144'])

        assert memory_service.graphiti.delete_entity_node.await_count == 2, (
            'The second degenerate node must still be attempted after the first delete fails'
        )
        assert stats == {'scanned': 2, 'degenerate': 2, 'deleted': 1, 'errors': 1}, (
            f'Expected the failed delete tallied as an error without blocking the second, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_find_failure_for_one_task_id_does_not_block_next_task_id(self):
        """find_duplicate_entity_nodes raises for the first task id; the second
        task id is still looked up and its degenerate node still deleted."""
        memory_service = _make_memory_service()

        async def fake_find(name, *, group_id):
            if name == 'tasks 142':
                raise RuntimeError('transient read timeout')
            if name == 'tasks 148':
                return [{'uuid': 'node-148', 'created_at': 100, 'edge_count': 0}]
            return []

        memory_service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find)

        stats = await sweep_degenerate_task_nodes(memory_service, 'test_project', ['142', '148'])

        assert memory_service.graphiti.find_duplicate_entity_nodes.await_count == 2, (
            'Both task ids must be looked up even though the first raised'
        )
        memory_service.graphiti.delete_entity_node.assert_awaited_once_with(
            'node-148', group_id='test_project',
        )
        assert stats == {'scanned': 1, 'degenerate': 1, 'deleted': 1, 'errors': 1}, (
            f'Expected the find failure tallied as an error and the second id still processed, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_empty_task_ids_yields_all_zero_stats_with_no_graphiti_calls(self):
        """Empty task_ids -> all-zero stats; neither graphiti method is awaited."""
        memory_service = _make_memory_service()

        stats = await sweep_degenerate_task_nodes(memory_service, 'test_project', [])

        assert stats == {'scanned': 0, 'degenerate': 0, 'deleted': 0, 'errors': 0}
        memory_service.graphiti.find_duplicate_entity_nodes.assert_not_awaited()
        memory_service.graphiti.delete_entity_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_matches_and_all_healthy_matches_both_yield_zero_deletions(self):
        """A name with no matches and a name with only healthy (edge_count>0)
        matches both contribute zero degenerate/deleted counts."""
        memory_service = _make_memory_service()

        async def fake_find(name, *, group_id):
            if name == 'tasks 150':
                return []
            if name == 'tasks 160':
                return [{'uuid': 'node-160', 'created_at': 100, 'edge_count': 5}]
            return []

        memory_service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find)

        stats = await sweep_degenerate_task_nodes(memory_service, 'test_project', ['150', '160'])

        memory_service.graphiti.delete_entity_node.assert_not_awaited()
        assert stats == {'scanned': 1, 'degenerate': 0, 'deleted': 0, 'errors': 0}, (
            f'Expected only the healthy node scanned and nothing deleted, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_malformed_edge_count_is_tallied_as_error_never_deleted(self):
        """A match with a missing or None edge_count is uncertain, not degenerate:
        it must be tallied into stats['errors'] (never assumed to be 0) and never
        deleted, while the sweep still proceeds to a later, well-formed match."""
        memory_service = _make_memory_service()

        async def fake_find(name, *, group_id):
            if name == 'tasks 142':
                return [{'uuid': 'node-142-missing-key', 'created_at': 100}]  # no edge_count key
            if name == 'tasks 144':
                return [{'uuid': 'node-144-none', 'created_at': 100, 'edge_count': None}]
            if name == 'tasks 148':
                return [{'uuid': 'node-148', 'created_at': 100, 'edge_count': 0}]
            return []

        memory_service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find)

        stats = await sweep_degenerate_task_nodes(
            memory_service, 'test_project', ['142', '144', '148'],
        )

        memory_service.graphiti.delete_entity_node.assert_awaited_once_with(
            'node-148', group_id='test_project',
        )
        assert stats == {'scanned': 3, 'degenerate': 1, 'deleted': 1, 'errors': 2}, (
            'Expected both the missing-key and None edge_count matches tallied as '
            f'errors (never deleted) and the well-formed match still processed, got {stats!r}'
        )


# --------------------------------------------------------------------------- #
# sweep_degenerate_task_nodes — name_templates parameterization
# --------------------------------------------------------------------------- #


class TestSweepDegenerateTaskNodesNameTemplates:
    """name_templates is the module's stated extension point for future
    casings/singular forms — every template must be queried per task id, and
    counts must aggregate across templates."""

    @pytest.mark.asyncio
    async def test_multiple_name_templates_are_all_queried_and_aggregated(self):
        """With 2 name_templates, both are queried for the task id and
        scanned/degenerate/deleted counts aggregate across both."""
        memory_service = _make_memory_service()

        async def fake_find(name, *, group_id):
            if name == 'tasks 142':
                return [{'uuid': 'node-lower-142', 'created_at': 100, 'edge_count': 0}]
            if name == 'Task 142':
                return [{'uuid': 'node-title-142', 'created_at': 100, 'edge_count': 0}]
            return []

        memory_service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find)

        stats = await sweep_degenerate_task_nodes(
            memory_service,
            'test_project',
            ['142'],
            name_templates=('tasks {id}', 'Task {id}'),
        )

        called_names = [
            c.args[0] for c in memory_service.graphiti.find_duplicate_entity_nodes.await_args_list
        ]
        assert called_names == ['tasks 142', 'Task 142'], (
            f'Expected both templates queried (in order) for the task id, got {called_names!r}'
        )
        deleted_uuids = {c.args[0] for c in memory_service.graphiti.delete_entity_node.await_args_list}
        assert deleted_uuids == {'node-lower-142', 'node-title-142'}, (
            f'Expected the degenerate node from each template deleted, got {deleted_uuids!r}'
        )
        assert stats == {'scanned': 2, 'degenerate': 2, 'deleted': 2, 'errors': 0}, (
            f'Expected counts aggregated across both templates, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_second_template_still_queried_when_first_has_no_matches(self):
        """A break/continue regression in the per-template loop would stop after
        the first template's empty result; this pins that the second is still
        queried and its degenerate node still deleted."""
        memory_service = _make_memory_service()

        async def fake_find(name, *, group_id):
            if name == 'tasks 148':
                return []
            if name == 'Task 148':
                return [{'uuid': 'node-title-148', 'created_at': 100, 'edge_count': 0}]
            return []

        memory_service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find)

        stats = await sweep_degenerate_task_nodes(
            memory_service,
            'test_project',
            ['148'],
            name_templates=('tasks {id}', 'Task {id}'),
        )

        assert memory_service.graphiti.find_duplicate_entity_nodes.await_count == 2, (
            'Expected both templates queried even though the first returned no matches'
        )
        memory_service.graphiti.delete_entity_node.assert_awaited_once_with(
            'node-title-148', group_id='test_project',
        )
        assert stats == {'scanned': 1, 'degenerate': 1, 'deleted': 1, 'errors': 0}, (
            f'Expected only the second template contributing counts, got {stats!r}'
        )
