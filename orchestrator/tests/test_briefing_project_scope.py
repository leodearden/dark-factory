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

from orchestrator.agents.briefing import (
    MEMORY_CONTEXT_CAVEAT,
    BriefingAssembler,
    filter_foreign_project_results,
)
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

    def test_no_op_when_nothing_dropped_returns_original_text_unchanged(self):
        """The common case (nothing filtered) must not pay for a re-serialise.

        Asserted via identity (``is``), not just equality: a rewrite that
        stopped short-circuiting but still round-tripped equal-looking JSON
        would not be caught by an equality check alone.
        """
        payload = json.dumps({
            'results': [_result('1', 'Own — café fact.', metadata={'project_id': 'dark_factory'})],
        })

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert dropped == 0
        assert text is payload

    def test_non_ascii_content_survives_round_trip_without_escaping(self):
        """When something IS dropped, the re-serialise must not mangle unicode.

        ``json.dumps`` defaults to ``ensure_ascii=True``, which would turn a
        literal em dash / accented character into a ``\\uXXXX`` escape —
        dark-factory memory content is dense with both.
        """
        payload = json.dumps({
            'results': [
                _result('1', 'Own — café fact.', metadata={'project_id': 'dark_factory'}),
                _result('2', 'Foreign fact.', metadata={'project_id': 'reify'}),
            ],
        })

        text, dropped = filter_foreign_project_results(payload, 'dark_factory')

        assert dropped == 1
        assert 'Own — café fact.' in text
        assert '\\u2014' not in text
        assert '\\u00e9' not in text

    def test_dropped_entry_is_logged_at_debug_with_id_key_and_value(self, caplog):
        """A false-positive drop must be diagnosable from an existing run's logs."""
        payload = json.dumps({
            'results': [_result('42', 'Foreign fact.', metadata={'project_id': 'reify'})],
        })

        with caplog.at_level(logging.DEBUG):
            filter_foreign_project_results(payload, 'dark_factory')

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(
            '42' in r.getMessage() and 'project_id' in r.getMessage() and 'reify' in r.getMessage()
            for r in debug_records
        )


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
        # The message names BOTH numbers (slots and queries) rather than
        # just `expected_dropped` — one distinct foreign fact matching all
        # four queries must not read as "4 memory results", which would
        # overstate the leak volume by 4x (task 3609 amendment).
        assert (
            f'{expected_dropped} memory result slot(s) across {queries_fired} queries were '
            'tagged to another project and filtered out'
        ) in context
        assert 'dark_factory' in caplog.text
        assert 'filtered' in caplog.text.lower()
        assert any(r.levelno == logging.INFO for r in caplog.records)

    async def test_all_foreign_results_surface_drop_count_in_empty_context(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The most consequential outcome — every recalled result is
        foreign — must not degrade silently: the drop count is visible in
        both the rendered message and the INFO log, not just discarded
        along with the (empty) recalled sections.
        """
        envelope = _mcp_search_envelope([
            _result('1', 'Foreign fact.', metadata={'project_id': 'reify'}),
        ])

        with caplog.at_level(logging.INFO), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            context = await briefing._get_memory_context('3609')

        assert context.splitlines()[0] == '# Context'
        assert 'Foreign fact.' not in context
        assert '_No memory context available' in context
        assert '4 memory result slot(s) across 4 queries' in context
        assert any(r.levelno == logging.INFO for r in caplog.records)
        assert 'filtered' in caplog.text.lower()


@pytest.mark.asyncio
class TestMemoryContextProvenanceCaveat:
    """A standing caveat covers the leak channel the tag filter cannot reach.

    ``filter_foreign_project_results`` only ever fires on Mem0-sourced
    results that carry a project tag. Every Graphiti-sourced result is
    untagged today (metadata == {}), so it survives the filter unclassified
    — the caveat is what converts "agent cd's/reads into a recalled
    'crates/reify-compiler' path" into "agent verifies the path first" for
    that untagged channel.
    """

    async def test_caveat_present_and_precedes_first_section(
        self, briefing: BriefingAssembler,
    ):
        envelope = _mcp_search_envelope([
            _result('1', 'Own project fact.', metadata={'project_id': 'dark_factory'}),
        ])

        with patch('orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope)):
            context = await briefing._get_memory_context('3609')

        assert context.splitlines()[0] == '# Context'
        assert briefing.project_id in context

        # Identity check against the actual constant, rather than pinning a
        # handful of substrings from its prose: any rewording that keeps the
        # constant's own text intact still passes, and any drift in what's
        # actually rendered is caught exactly (task 3609 amendment).
        assert MEMORY_CONTEXT_CAVEAT.format(project_id=briefing.project_id) in context

        # The caveat must precede the first '## ' section heading. Locate the
        # project_id mention that establishes the caveat is present (the
        # earliest one after the heading) and confirm it comes before that
        # heading — the JSON section body below also names the project, but
        # only as a later occurrence.
        caveat_mention = context.index(briefing.project_id, len('# Context'))
        first_section = context.index('\n## ')
        assert caveat_mention < first_section

    async def test_caveat_absent_when_no_sections_survive(self, briefing: BriefingAssembler):
        envelope = {'result': {'content': []}}

        with patch('orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope)):
            context = await briefing._get_memory_context('3609')

        assert context == '# Context\n\n_No memory context available._'

    async def test_caveat_present_alongside_untagged_foreign_path_content(
        self, briefing: BriefingAssembler,
    ):
        """The census failure mode: untagged content naming a foreign path.

        The result carries no project tag at all (metadata == {}), so the
        filter cannot classify — let alone drop — it; it renders verbatim.
        The caveat must still be present so the agent is warned to verify
        the path before treating it as real.
        """
        envelope = _mcp_search_envelope([
            _result(
                '1',
                "A park on 'crates/reify-compiler/src' blocks acquire of "
                "'crates/reify-compiler'.",
                metadata={},
            ),
        ])

        with patch('orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope)):
            context = await briefing._get_memory_context('3609')

        assert 'crates/reify-compiler' in context
        assert briefing.project_id in context
        assert 'verify' in context.lower()

    async def test_caveat_covers_recalled_sections_after_partial_failure(
        self, briefing: BriefingAssembler,
    ):
        """A later query raising must not blank the caveat for sections that
        were already successfully recalled.

        The caveat used to be gated on `memory_unavailable`, which — because
        it is set by a `try`/`except` wrapping all four searches — suppressed
        the caveat for the WHOLE block even when earlier queries had already
        returned real facts. This patches `_scoped_search` directly (rather
        than `mcp_call`, as the other tests in this module do) because
        `_mcp_search` itself catches every exception internally and returns
        `None` — a `mcp_call` failure can never reach `_get_memory_context`
        as a raised exception, only as an empty result. `_scoped_search` is
        the seam where `_get_memory_context`'s own `try`/`except` can
        actually observe a failure.
        """
        async def scoped_search_side_effect(query):
            if 'decisions' in query:
                raise TimeoutError('memory service unreachable')
            return f'## recalled for: {query}', 0

        with patch.object(
            briefing, '_scoped_search', new=AsyncMock(side_effect=scoped_search_side_effect),
        ):
            context = await briefing._get_memory_context('3609')

        assert context.splitlines()[0] == '# Context'
        assert '## Project Context' in context
        assert '## Conventions' in context
        assert '## Recent Decisions' not in context
        assert MEMORY_CONTEXT_CAVEAT.format(project_id=briefing.project_id) in context


@pytest.mark.asyncio
class TestScopedSearch:
    """Direct coverage of ``_scoped_search``, the seam between
    ``_mcp_search`` and ``_get_memory_context``.
    """

    async def test_returns_none_with_no_drop_when_underlying_search_is_empty(
        self, briefing: BriefingAssembler,
    ):
        with patch.object(briefing, '_mcp_search', new=AsyncMock(return_value=None)):
            result = await briefing._scoped_search('anything')

        assert result == (None, 0)

    async def test_multi_text_block_response_fails_open_known_limitation(
        self, briefing: BriefingAssembler, caplog,
    ):
        """``_mcp_search`` joins every MCP text block with ``'\\n'`` before
        returning (unchanged by this task — see its allowlist entry in
        ``shared/tests/silent_fallthrough_allowlist.py``). If the search
        tool ever answers with more than one text block, the joined text is
        not a single valid JSON document, so the filter fails open — the
        cross-project filter silently stops working for that query, though
        the block still renders (nothing is lost from the prompt, just the
        filtering). This test PINS that known limitation so a change to the
        response shape — or a future per-block fix — is caught rather than
        drifting unnoticed.
        """
        envelope = {
            'result': {
                'content': [
                    {'type': 'text', 'text': json.dumps({'results': [
                        _result('1', 'Foreign fact.', metadata={'project_id': 'reify'}),
                    ]})},
                    {'type': 'text', 'text': json.dumps({'results': [
                        _result('2', 'Own fact.', metadata={'project_id': 'dark_factory'}),
                    ]})},
                ],
            },
        }

        with caplog.at_level(logging.WARNING), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            text, dropped = await briefing._scoped_search('anything')

        # Known limitation: the joined multi-block text isn't valid JSON, so
        # the filter fails open rather than filtering each block — the
        # foreign fact is NOT removed.
        assert dropped == 0
        assert text is not None
        assert 'Foreign fact.' in text
        assert 'Own fact.' in text
        assert any(r.levelno == logging.WARNING for r in caplog.records)
