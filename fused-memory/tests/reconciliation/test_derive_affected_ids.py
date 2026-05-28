"""Tests for harness._derive_affected_ids (reviewer finding missing_test_coverage).

_derive_affected_ids is a pure helper that flattens typed citation lists
(cited_tasks, cited_entities, cited_edges, cited_memories) into a flat list of
identity strings for escalation dedup and recurrence fingerprinting.

Non-trivial logic: legacy-field precedence, per-type field selection
(canonical_name OR entity_uuid for entities), order guarantee, and silent
skipping of malformed entries.  A regression here would silently weaken
cross-run recurrence counting against the _INTEGRITY_FINDING_RECURRENCE_THRESHOLD=4
gate without any other test failing.
"""

from __future__ import annotations

from fused_memory.reconciliation.harness import _derive_affected_ids


class TestDeriveAffectedIds:
    """_derive_affected_ids must flatten typed citations into a stable identity list."""

    # ------------------------------------------------------------------ #
    # Legacy compatibility
    # ------------------------------------------------------------------ #

    def test_legacy_affected_ids_takes_precedence(self):
        """When affected_ids is present (pre-cutover shape), it wins over typed citations."""
        finding = {
            'affected_ids': ['legacy1', 'legacy2'],
            'cited_tasks': [{'task_id': '99', 'project_id': 'p', 'title': 'T'}],
            'cited_entities': [{'canonical_name': 'should-be-ignored', 'entity_uuid': 'x'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['legacy1', 'legacy2'], (
            f'Legacy affected_ids must take precedence, got {result!r}'
        )

    def test_legacy_affected_ids_coerced_to_str(self):
        """affected_ids entries are coerced to str (robustness for int task_ids)."""
        finding = {'affected_ids': [42, 99]}
        result = _derive_affected_ids(finding)
        assert result == ['42', '99']

    # ------------------------------------------------------------------ #
    # Empty / absent citations
    # ------------------------------------------------------------------ #

    def test_empty_citations_returns_empty_list(self):
        """Finding with all-empty citation lists returns empty list."""
        finding = {
            'cited_tasks': [],
            'cited_entities': [],
            'cited_edges': [],
            'cited_memories': [],
        }
        result = _derive_affected_ids(finding)
        assert result == []

    def test_no_citations_at_all(self):
        """Finding with no citation keys at all returns empty list."""
        result = _derive_affected_ids({})
        assert result == []

    def test_none_citation_list_tolerated(self):
        """None values for citation lists are tolerated (treated as absent)."""
        finding = {
            'cited_tasks': None,
            'cited_entities': None,
            'cited_edges': None,
            'cited_memories': None,
        }
        result = _derive_affected_ids(finding)
        assert result == []

    # ------------------------------------------------------------------ #
    # Order: tasks → entities → edges → memories
    # ------------------------------------------------------------------ #

    def test_mixed_citation_types_preserve_order(self):
        """Results follow tasks → entities → edges → memories order."""
        finding = {
            'cited_tasks': [{'task_id': 'task-1', 'project_id': 'p', 'title': 'T'}],
            'cited_entities': [{'canonical_name': 'Entity1', 'entity_uuid': 'ent-uuid'}],
            'cited_edges': [{'edge_uuid': 'edge-uuid-1'}],
            'cited_memories': [{'memory_id': 'mem-uuid-1', 'store': 'mem0'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['task-1', 'Entity1', 'edge-uuid-1', 'mem-uuid-1'], (
            f'Expected tasks→entities→edges→memories order, got {result!r}'
        )

    # ------------------------------------------------------------------ #
    # Field selection: entities use canonical_name or entity_uuid
    # ------------------------------------------------------------------ #

    def test_canonical_name_preferred_over_entity_uuid(self):
        """canonical_name is used when both fields are present."""
        finding = {
            'cited_entities': [
                {'canonical_name': 'MyEntity', 'entity_uuid': 'uuid-should-not-appear'},
            ],
        }
        result = _derive_affected_ids(finding)
        assert result == ['MyEntity']
        assert 'uuid-should-not-appear' not in result

    def test_entity_uuid_fallback_when_no_canonical_name(self):
        """entity_uuid is used when canonical_name is absent."""
        finding = {
            'cited_entities': [{'entity_uuid': 'uuid-fallback'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['uuid-fallback']

    def test_entity_with_neither_field_skipped(self):
        """Entity citation with neither canonical_name nor entity_uuid is skipped."""
        finding = {
            'cited_entities': [
                {'unrelated_key': 'foo'},
                {'canonical_name': 'ValidEntity', 'entity_uuid': 'x'},
            ],
        }
        result = _derive_affected_ids(finding)
        assert result == ['ValidEntity']

    # ------------------------------------------------------------------ #
    # Non-dict entries are skipped silently
    # ------------------------------------------------------------------ #

    def test_non_dict_cited_tasks_skipped(self):
        """Non-dict entries in cited_tasks are silently skipped."""
        finding = {
            'cited_tasks': ['not-a-dict', None, {'task_id': '5', 'project_id': 'p', 'title': 'T'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['5']

    def test_non_dict_cited_entities_skipped(self):
        """Non-dict entries in cited_entities are silently skipped."""
        finding = {
            'cited_entities': [42, {'canonical_name': 'E1', 'entity_uuid': 'e1'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['E1']

    def test_non_dict_cited_edges_skipped(self):
        """Non-dict entries in cited_edges are silently skipped."""
        finding = {
            'cited_edges': ['bad-entry', {'edge_uuid': 'edge-ok'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['edge-ok']

    def test_non_dict_cited_memories_skipped(self):
        """Non-dict entries in cited_memories are silently skipped."""
        finding = {
            'cited_memories': [None, {'memory_id': 'mem-ok', 'store': 'mem0'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['mem-ok']

    # ------------------------------------------------------------------ #
    # Edge cases: missing task_id / edge_uuid / memory_id
    # ------------------------------------------------------------------ #

    def test_cited_task_without_task_id_skipped(self):
        """cited_tasks entry without task_id field is skipped."""
        finding = {
            'cited_tasks': [
                {'project_id': 'p', 'title': 'no task_id here'},
                {'task_id': 'valid', 'project_id': 'p', 'title': 'T'},
            ],
        }
        result = _derive_affected_ids(finding)
        assert result == ['valid']

    def test_cited_edge_without_edge_uuid_skipped(self):
        """cited_edges entry without edge_uuid field is skipped."""
        finding = {
            'cited_edges': [
                {'fact_text_snapshot': 'no uuid here'},
                {'edge_uuid': 'valid-edge'},
            ],
        }
        result = _derive_affected_ids(finding)
        assert result == ['valid-edge']

    def test_cited_memory_without_memory_id_skipped(self):
        """cited_memories entry without memory_id field is skipped."""
        finding = {
            'cited_memories': [
                {'store': 'mem0'},  # no memory_id
                {'memory_id': 'valid-mem', 'store': 'mem0'},
            ],
        }
        result = _derive_affected_ids(finding)
        assert result == ['valid-mem']
