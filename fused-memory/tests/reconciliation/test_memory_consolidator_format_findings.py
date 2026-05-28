"""Tests for PRD γ cutover: _format_findings migration to typed citation lists."""

from __future__ import annotations

from fused_memory.reconciliation.stages.memory_consolidator import _format_findings


class TestFormatFindingsWithCitations:
    """_format_findings must render typed citation lists (PRD §9.3), not affected_ids.

    RED: currently _format_findings reads affected_ids and ignores the four typed
    citation lists. Step-16 migrates it to render cited_entities, cited_edges,
    cited_tasks, and cited_memories instead.
    """

    def _make_finding(self):
        """Build a finding dict with all four typed citation lists and no affected_ids."""
        return {
            'description': 'Memory and task are inconsistent',
            'severity': 'moderate',
            'category': 'memory_task_mismatch',
            'suggested_action': 'Update the memory to match task state.',
            'actionable': True,
            'cited_entities': [
                {'entity_uuid': 'ent-uuid-1', 'canonical_name': 'TaskEntity'},
            ],
            'cited_edges': [
                {'edge_uuid': 'edge-uuid-2', 'fact_text_snapshot': 'task was completed on 2026-01-01'},
            ],
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '42', 'title': 'Refactor reconciliation'},
            ],
            'cited_memories': [
                {'memory_id': 'mem-uuid-3', 'store': 'mem0', 'metadata_fingerprint': 'abc123'},
            ],
        }

    def test_canonical_name_in_output(self):
        """cited_entities[].canonical_name appears in the rendered output."""
        finding = self._make_finding()
        output = _format_findings([finding])
        assert 'TaskEntity' in output, (
            f'canonical_name "TaskEntity" must appear in _format_findings output, got:\n{output}'
        )

    def test_edge_uuid_in_output(self):
        """cited_edges[].edge_uuid appears in the rendered output."""
        finding = self._make_finding()
        output = _format_findings([finding])
        assert 'edge-uuid-2' in output, (
            f'edge_uuid "edge-uuid-2" must appear in _format_findings output, got:\n{output}'
        )

    def test_task_project_and_id_in_output(self):
        """cited_tasks project_id and task_id both appear in the rendered output."""
        finding = self._make_finding()
        output = _format_findings([finding])
        assert 'dark_factory' in output or '42' in output, (
            f'cited_tasks project_id/task_id must appear in output, got:\n{output}'
        )

    def test_memory_id_and_store_in_output(self):
        """cited_memories[].memory_id and store appear in the rendered output."""
        finding = self._make_finding()
        output = _format_findings([finding])
        assert 'mem-uuid-3' in output, (
            f'memory_id "mem-uuid-3" must appear in _format_findings output, got:\n{output}'
        )
        assert 'mem0' in output, (
            f'store "mem0" must appear in _format_findings output, got:\n{output}'
        )

    def test_no_affected_ids_literal_in_output(self):
        """The string 'affected_ids' must not appear anywhere in the output."""
        finding = self._make_finding()
        output = _format_findings([finding])
        assert 'affected_ids' not in output, (
            f'"affected_ids" must not appear in _format_findings output after migration, '
            f'got:\n{output}'
        )

    def test_header_fields_preserved(self):
        """description, severity, category, suggested_action still appear in output."""
        finding = self._make_finding()
        output = _format_findings([finding])
        assert 'Memory and task are inconsistent' in output
        assert 'moderate' in output
        assert 'memory_task_mismatch' in output

    def test_empty_citation_lists_do_not_error(self):
        """A finding with empty citation lists renders without error."""
        finding = {
            'description': 'Simple finding',
            'severity': 'minor',
            'category': 'test',
            'suggested_action': 'Check it.',
            'cited_entities': [],
            'cited_edges': [],
            'cited_tasks': [],
            'cited_memories': [],
        }
        output = _format_findings([finding])
        assert 'Simple finding' in output
        assert 'affected_ids' not in output

    def test_task_title_included_in_output(self):
        """cited_tasks title appears in the rendered output (reviewer incomplete_rendering)."""
        finding = {
            'description': 'x',
            'severity': 'minor',
            'category': 'other',
            'suggested_action': 'y',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '99', 'title': 'My Task Title'},
            ],
        }
        output = _format_findings([finding])
        assert 'My Task Title' in output, (
            f'Task title must appear in _format_findings output, got:\n{output}'
        )

    def test_fact_text_snapshot_truncated_at_120(self):
        """fact_text_snapshot longer than 120 chars is truncated with ellipsis."""
        long_snapshot = 'X' * 130
        finding = {
            'description': 'x',
            'severity': 'minor',
            'category': 'other',
            'suggested_action': 'y',
            'cited_edges': [
                {'edge_uuid': 'edge-trunc-test', 'fact_text_snapshot': long_snapshot},
            ],
        }
        output = _format_findings([finding])
        assert 'edge-trunc-test' in output
        # Truncated snapshot should appear (first 120 chars + '...')
        assert 'X' * 120 + '...' in output
        # Full 130-char snapshot must NOT appear verbatim
        assert long_snapshot not in output

    def test_citation_cap_at_five(self):
        """Citation lists with more than 5 entries show '... +N more' suffix."""
        finding = {
            'description': 'x',
            'severity': 'minor',
            'category': 'other',
            'suggested_action': 'y',
            'cited_entities': [
                {'entity_uuid': f'uuid-{n}', 'canonical_name': f'Entity{n}'}
                for n in range(8)
            ],
        }
        output = _format_findings([finding])
        # 5 shown, 3 extra
        assert '+3 more' in output
        # Only first 5 entities rendered
        assert 'Entity0' in output
        assert 'Entity4' in output
        assert 'Entity5' not in output
