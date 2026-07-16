"""Tests for the stale task-status snapshot Graphiti edge sweep (task 2613).

Reconciliation Stage 1 (MemoryConsolidator) has no deterministic sweep that
invalidates VALID (invalid_at IS NULL) task-status-snapshot Graphiti edges
once the task(s) they reference reach a terminal status (done/cancelled).
This module adds a small deterministic post-processor: enumerate valid
edges, cross-reference each referenced task_id's CURRENT status via a direct
status lookup (taskmaster.get_statuses — NOT semantic search), and
invalidate any edge whose asserted non-terminal (active/pending/in-progress)
status now contradicts a terminal task.

Covers:
- extract_snapshot_edge_task_ids: pure lexical extractor — returns the
  specific task ids a status-snapshot edge asserts as active/pending/
  in-progress; returns the empty set for text with no such status marker,
  and for pure count-only snapshots with no specific task-id (the
  stale-by-design audit trail this sweep must never touch).
- flatten_dedup_edges: flattens get_all_valid_edges' dict[entity_uuid,
  list[EdgeDict]] shape and dedups by edge uuid (handles the backend's
  double-attribution of each directed edge under both endpoint entities).
- select_stale_status_snapshot_edges: pure decision core — selects edges
  whose asserted status contradicts a positively-terminal current status.
- sweep_stale_status_snapshot_edges: async orchestrator — enumerates,
  cross-references, and invalidates, best-effort throughout.
"""

from __future__ import annotations

from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (
    extract_snapshot_edge_task_ids,
)


class TestExtractSnapshotEdgeTaskIds:
    """extract_snapshot_edge_task_ids(fact) returns the specific task ids a
    status-snapshot edge asserts as active/pending/in-progress.

    Returns the empty set both when no active/pending/in-progress status
    marker is present at all, AND for pure count-only snapshots with no
    specific task-id reference — the out-of-scope, stale-by-design audit
    trail (Snapshot Discipline) that this sweep must never touch.
    """

    def test_individual_form_active_pending(self):
        """'Task 142 is an active pending task' -> {142}."""
        assert extract_snapshot_edge_task_ids('Task 142 is an active pending task') == {142}

    def test_individual_form_in_progress(self):
        """'Task 148 is in-progress' -> {148}."""
        assert extract_snapshot_edge_task_ids('Task 148 is in-progress') == {148}

    def test_aggregate_list_form_brackets(self):
        """'The active pending tasks are [142, 148, 150]' -> {142, 148, 150}."""
        result = extract_snapshot_edge_task_ids(
            'The active pending tasks are [142, 148, 150]'
        )
        assert result == {142, 148, 150}

    def test_aggregate_list_form_colon(self):
        """'active/in-progress tasks: 142, 148' -> {142, 148}."""
        result = extract_snapshot_edge_task_ids('active/in-progress tasks: 142, 148')
        assert result == {142, 148}

    def test_count_only_no_specific_id_returns_empty(self):
        """'There are 8 tasks in progress' -> set() (out-of-scope count-only snapshot)."""
        assert extract_snapshot_edge_task_ids('There are 8 tasks in progress') == set()

    def test_count_pair_returns_empty(self):
        """'1505 done / 148 cancelled' -> set() ('148' is a count operand, not an id)."""
        assert extract_snapshot_edge_task_ids('1505 done / 148 cancelled') == set()

    def test_status_gate_done_returns_empty(self):
        """'Task 5 is done' -> set() (no active/pending/in-progress marker)."""
        assert extract_snapshot_edge_task_ids('Task 5 is done') == set()

    def test_status_gate_merge_commit_returns_empty(self):
        """'Task 7 landed as merge commit' -> set() (no active/pending/in-progress marker)."""
        assert extract_snapshot_edge_task_ids('Task 7 landed as merge commit') == set()

    def test_incidental_numbers_excluded(self):
        """Date/commit-hash digits near a valid reference must not be swept in as ids."""
        result = extract_snapshot_edge_task_ids(
            'Task 142 is pending as of 2026-07-14 (commit ab12cd)'
        )
        assert result == {142}
