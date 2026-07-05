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

from fused_memory.reconciliation.degenerate_task_node_sweep import (
    extract_terminal_task_ids,
)
from fused_memory.reconciliation.task_filter import FilteredTaskTree


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
