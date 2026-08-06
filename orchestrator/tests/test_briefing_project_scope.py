"""Tests for project-scoping the dispatched-agent briefing ``# Context`` block.

Task 3609 (census R5): a hardcoded ``search(project_id=self.project_id)`` call
in ``BriefingAssembler._get_memory_context`` still surfaces cross-project
memory facts (metadata carries no project tag on the dominant, Graphiti-backed
leak channel) into a dispatched agent's prompt. This file covers the pure
``filter_foreign_project_results`` helper — the metadata-tag filter half of
the fix — plus (once added) the assembly-level end-to-end behaviour and the
standing provenance caveat that covers the untagged leak channel the filter
cannot reach on its own.

See ``orchestrator/src/orchestrator/agents/briefing.py``:
``filter_foreign_project_results`` / ``_canonical_project`` /
``_scoped_search`` / ``_get_memory_context``.
"""

from __future__ import annotations

import json

from orchestrator.agents.briefing import filter_foreign_project_results


def _result(id_: str, content: str, metadata: dict | None = None, source_store: str = 'graphiti') -> dict:
    """Build a dict matching the wire shape of ``fused_memory.models.memory.MemoryResult``.

    Mirrors the real result schema (id/content/category/source_store/
    relevance_score/provenance/temporal/entities/metadata/created_at) so the
    filter is exercised against the actual payload shape, not an invented one.
    """
    return {
        'id': id_,
        'content': content,
        'category': None,
        'source_store': source_store,
        'relevance_score': 0.9,
        'provenance': [],
        'temporal': None,
        'entities': [],
        'metadata': {} if metadata is None else metadata,
        'created_at': None,
    }


class TestFilterForeignProjectResults:
    """Unit coverage for the pure metadata-tag filter."""

    def test_foreign_tagged_result_is_dropped(self):
        payload = json.dumps({
            'results': [_result('1', 'Foreign fact about reify.', metadata={'project_id': 'reify'})],
        })

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Foreign fact about reify.' not in text
        assert dropped == 1

    def test_own_project_tagged_result_is_kept(self):
        payload = json.dumps({
            'results': [_result('1', 'Own project fact.', metadata={'project_id': 'dark_factory'})],
        })

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Own project fact.' in text
        assert dropped == 0

    def test_untagged_result_is_kept_and_not_counted(self):
        payload = json.dumps({
            'results': [_result('1', 'Untagged fact.', metadata={})],
        })

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Untagged fact.' in text
        assert dropped == 0

    def test_all_foreign_returns_empty_text_and_full_drop_count(self):
        payload = json.dumps({
            'results': [
                _result('1', 'Foreign one.', metadata={'project_id': 'reify'}),
                _result('2', 'Foreign two.', metadata={'project_id': 'other_proj'}),
            ],
        })

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert text == ''
        assert dropped == 2

    def test_sibling_top_level_keys_survive(self):
        payload = json.dumps({
            'results': [_result('1', 'Own fact.', metadata={'project_id': 'dark_factory'})],
            'degraded': True,
            'failed_stores': ['mem0'],
        })

        text, _dropped = filter_foreign_project_results(payload, 'dark_factory')
        parsed = json.loads(text)

        assert parsed['degraded'] is True
        assert parsed['failed_stores'] == ['mem0']

    def test_kept_results_round_trip_in_original_order(self):
        payload = json.dumps({
            'results': [
                _result('1', 'First own fact.', metadata={'project_id': 'dark_factory'}),
                _result('2', 'Foreign fact.', metadata={'project_id': 'reify'}),
                _result('3', 'Second own fact.', metadata={}),
            ],
        })

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')
        parsed = json.loads(text)

        assert [r['id'] for r in parsed['results']] == ['1', '3']
        assert dropped == 1
