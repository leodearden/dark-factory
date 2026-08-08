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
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.agents.briefing import BriefingAssembler, filter_foreign_project_results
from orchestrator.config import GitConfig, OrchestratorConfig


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    return BriefingAssembler(config)


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


class TestFilterFailsOpen:
    """The filter must never destroy context on a malformed payload.

    It fails OPEN — returns the original text unchanged and drops nothing —
    and logs loudly (WARNING) rather than silently blanking the # Context
    block on a serialisation surprise.
    """

    def test_non_json_text_fails_open(self, caplog):
        text_in = 'not json at all'

        with caplog.at_level(logging.WARNING):
            text, dropped = filter_foreign_project_results(text_in, 'dark_factory')

        assert (text, dropped) == (text_in, 0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_json_array_not_object_fails_open(self, caplog):
        text_in = '[1, 2, 3]'

        with caplog.at_level(logging.WARNING):
            text, dropped = filter_foreign_project_results(text_in, 'dark_factory')

        assert (text, dropped) == (text_in, 0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_missing_results_key_fails_open(self, caplog):
        text_in = json.dumps({'degraded': True})

        with caplog.at_level(logging.WARNING):
            text, dropped = filter_foreign_project_results(text_in, 'dark_factory')

        assert (text, dropped) == (text_in, 0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_non_list_results_fails_open(self, caplog):
        text_in = json.dumps({'results': 'oops'})

        with caplog.at_level(logging.WARNING):
            text, dropped = filter_foreign_project_results(text_in, 'dark_factory')

        assert (text, dropped) == (text_in, 0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_non_dict_entry_is_kept_without_raising(self):
        payload = json.dumps({
            'results': ['stray', _result('1', 'Own fact.', metadata={'project_id': 'dark_factory'})],
        })

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')
        parsed = json.loads(text)

        assert 'stray' in parsed['results']
        assert dropped == 0

    def test_non_dict_metadata_is_kept_without_raising(self):
        entry_none_meta = _result('1', 'Fact with null metadata.')
        entry_none_meta['metadata'] = None
        entry_list_meta = _result('2', 'Fact with list metadata.')
        entry_list_meta['metadata'] = ['oops']
        payload = json.dumps({'results': [entry_none_meta, entry_list_meta]})

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact with null metadata.' in text
        assert 'Fact with list metadata.' in text
        assert dropped == 0


class TestForeignTagKeysAndSpelling:
    """Tag extraction must recognise more than the bare ``project_id`` key,
    prefer ``src_project`` for CGL-eta rehomed entries, and canonicalise
    divergent spellings so ``dark-factory`` isn't mistaken for a foreign
    project.
    """

    def test_group_id_tag_is_dropped(self):
        payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'group_id': 'reify'})]})

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' not in text
        assert dropped == 1

    def test_project_tag_is_dropped(self):
        payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'project': 'reify'})]})

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' not in text
        assert dropped == 1

    def test_cgl_eta_rehome_production_case_is_dropped(self):
        """The one foreign-tag channel demonstrably present in production data.

        Task-2273 CGL-eta rehomed entries physically live in dst_project's
        Mem0 collection but reference src_project's task numbers — src_project
        is the authoritative origin.
        """
        payload = json.dumps({'results': [_result(
            '1',
            "A park on 'crates/reify-compiler/src' blocks acquire of 'crates/reify-compiler'.",
            metadata={
                'kind': 'cgl_eta_cross_target_rehome',
                'src_project': 'reify',
                'dst_project': 'dark_factory',
                'src_entity': 'Task 2310',
            },
        )]})

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'crates/reify-compiler' not in text
        assert dropped == 1

    def test_src_project_wins_over_same_entry_project_id(self):
        payload = json.dumps({'results': [_result(
            '1', 'Fact.', metadata={'src_project': 'reify', 'project_id': 'dark_factory'},
        )]})

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' not in text
        assert dropped == 1

    def test_dst_project_alone_is_never_consulted_and_is_kept(self):
        payload = json.dumps({'results': [_result(
            '1', 'Fact.', metadata={'dst_project': 'dark_factory'},
        )]})

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' in text
        assert dropped == 0

    def test_first_present_key_precedence(self):
        payload = json.dumps({'results': [_result(
            '1', 'Fact.', metadata={'project_id': 'dark_factory', 'group_id': 'reify'},
        )]})

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' in text
        assert dropped == 0

    def test_canonicalisation_of_divergent_spellings(self):
        for spelling in ('dark-factory', 'Dark_Factory', '  dark_factory  '):
            payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'project_id': spelling})]})

            text, dropped = filter_foreign_project_results(payload, 'dark_factory')

            assert 'Fact.' in text, f'spelling {spelling!r} should be treated as own-project'
            assert dropped == 0

        payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'project_id': 'reify'})]})
        text, dropped = filter_foreign_project_results(payload, 'dark_factory')
        assert 'Fact.' not in text
        assert dropped == 1

    def test_non_string_tag_is_treated_as_untagged(self):
        payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'project_id': 123})]})

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' in text
        assert dropped == 0


def _mcp_search_envelope(results: list[dict]) -> dict:
    """Build the real ``tools/call`` response envelope ``_mcp_search`` reads.

    Mirrors ``BriefingAssembler._mcp_search`` (briefing.py:1250-1275): FastMCP
    returns ``{'result': {'content': [{'type': 'text', 'text': ...}]}}`` where
    ``text`` is the JSON-serialised ``search`` tool payload. Used to patch
    ``orchestrator.agents.briefing.mcp_call`` directly (unlike every other
    briefing test, which patches ``_get_memory_context`` itself away to a
    stub) so the real ``_get_memory_context`` / ``_scoped_search`` /
    ``filter_foreign_project_results`` pipeline actually runs end-to-end.
    """
    return {
        'result': {
            'content': [
                {'type': 'text', 'text': json.dumps({'results': results})},
            ],
        },
    }


@pytest.mark.asyncio
class TestGetMemoryContextFiltersForeignFacts:
    """``_get_memory_context`` drops foreign-tagged results end-to-end.

    ``_mcp_search`` is called once per hardcoded query (project overview,
    conventions, decisions, and — since a ``task_id`` is passed — task
    context: four calls). The stub answers every call identically, so a
    single foreign result per query yields a filtered count of 4 in the
    assembled block, not 1 — the count must reflect all four queries, not
    just one.
    """

    async def test_filters_foreign_facts_and_announces_the_drop(
        self, briefing: BriefingAssembler, caplog,
    ):
        envelope = _mcp_search_envelope([
            _result(
                '1',
                "A park on 'crates/reify-compiler/src' blocks acquire of "
                "'crates/reify-compiler'.",
                metadata={'project_id': 'reify'},
            ),
            _result(
                '2',
                'Own project fact about dark_factory.',
                metadata={'project_id': 'dark_factory'},
            ),
        ])
        foreign_per_query = 1
        queries_fired = 4  # overview, conventions, decisions, task-context (task_id given below)
        expected_dropped = foreign_per_query * queries_fired

        with caplog.at_level(logging.INFO), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            context = await briefing._get_memory_context('3609')

        assert context.splitlines()[0] == '# Context'
        assert 'Own project fact about dark_factory.' in context
        assert 'crates/reify-compiler' not in context
        assert "A park on 'crates/reify-compiler/src'" not in context
        assert f'{expected_dropped} memory result(s) tagged to another project were filtered out' in context
        assert 'dark_factory' in caplog.text
        assert 'filtered' in caplog.text.lower()
        assert any(r.levelno == logging.INFO for r in caplog.records)
