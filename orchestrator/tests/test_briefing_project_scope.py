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

import importlib.util
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.agents.briefing import (
    FOREIGN_PROJECT_TAG_KEYS,
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

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Foreign fact about reify.' not in text
        assert dropped == 1

    def test_own_project_tagged_result_is_kept(self):
        payload = json.dumps({
            'results': [_result('1', 'Own project fact.', metadata={'project_id': 'dark_factory'})],
        })

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Own project fact.' in text
        assert dropped == 0

    def test_untagged_result_is_kept_and_not_counted(self):
        payload = json.dumps({
            'results': [_result('1', 'Untagged fact.', metadata={})],
        })

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Untagged fact.' in text
        assert dropped == 0

    def test_all_foreign_returns_empty_text_and_full_drop_count(self):
        payload = json.dumps({
            'results': [
                _result('1', 'Foreign one.', metadata={'project_id': 'reify'}),
                _result('2', 'Foreign two.', metadata={'project_id': 'other_proj'}),
            ],
        })

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert text == ''
        assert dropped == 2

    def test_sibling_top_level_keys_survive(self):
        payload = json.dumps({
            'results': [_result('1', 'Own fact.', metadata={'project_id': 'dark_factory'})],
            'degraded': True,
            'failed_stores': ['mem0'],
        })

        text, _dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')
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

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')
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

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

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

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

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
            text, dropped, _nested = filter_foreign_project_results(text_in, 'dark_factory')

        assert (text, dropped) == (text_in, 0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_json_array_not_object_fails_open(self, caplog):
        text_in = '[1, 2, 3]'

        with caplog.at_level(logging.WARNING):
            text, dropped, _nested = filter_foreign_project_results(text_in, 'dark_factory')

        assert (text, dropped) == (text_in, 0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_missing_results_key_fails_open(self, caplog):
        text_in = json.dumps({'degraded': True})

        with caplog.at_level(logging.WARNING):
            text, dropped, _nested = filter_foreign_project_results(text_in, 'dark_factory')

        assert (text, dropped) == (text_in, 0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_non_list_results_fails_open(self, caplog):
        text_in = json.dumps({'results': 'oops'})

        with caplog.at_level(logging.WARNING):
            text, dropped, _nested = filter_foreign_project_results(text_in, 'dark_factory')

        assert (text, dropped) == (text_in, 0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_non_dict_entry_is_kept_without_raising(self):
        payload = json.dumps({
            'results': ['stray', _result('1', 'Own fact.', metadata={'project_id': 'dark_factory'})],
        })

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')
        parsed = json.loads(text)

        assert 'stray' in parsed['results']
        assert dropped == 0

    def test_non_dict_metadata_is_kept_without_raising(self):
        entry_none_meta = _result('1', 'Fact with null metadata.')
        entry_none_meta['metadata'] = None
        entry_list_meta = _result('2', 'Fact with list metadata.')
        entry_list_meta['metadata'] = ['oops']
        payload = json.dumps({'results': [entry_none_meta, entry_list_meta]})

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

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

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' not in text
        assert dropped == 1

    def test_project_tag_is_dropped(self):
        payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'project': 'reify'})]})

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

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

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'crates/reify-compiler' not in text
        assert dropped == 1

    def test_src_project_wins_over_same_entry_project_id(self):
        payload = json.dumps({'results': [_result(
            '1', 'Fact.', metadata={'src_project': 'reify', 'project_id': 'dark_factory'},
        )]})

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' not in text
        assert dropped == 1

    def test_dst_project_alone_is_never_consulted_and_is_kept(self):
        payload = json.dumps({'results': [_result(
            '1', 'Fact.', metadata={'dst_project': 'dark_factory'},
        )]})

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' in text
        assert dropped == 0

    def test_first_present_key_precedence(self):
        payload = json.dumps({'results': [_result(
            '1', 'Fact.', metadata={'project_id': 'dark_factory', 'group_id': 'reify'},
        )]})

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' in text
        assert dropped == 0

    def test_canonicalisation_of_divergent_spellings(self):
        for spelling in ('dark-factory', 'Dark_Factory', '  dark_factory  '):
            payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'project_id': spelling})]})

            text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

            assert 'Fact.' in text, f'spelling {spelling!r} should be treated as own-project'
            assert dropped == 0

        payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'project_id': 'reify'})]})
        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')
        assert 'Fact.' not in text
        assert dropped == 1

    def test_non_string_tag_is_treated_as_untagged(self):
        payload = json.dumps({'results': [_result('1', 'Fact.', metadata={'project_id': 123})]})

        text, dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        assert 'Fact.' in text
        assert dropped == 0


class TestOriginTagKeyDriftGuard:
    """The two hand-copied tag tuples must stay identical, key for key.

    ``fused_memory.server.grouped_read.ORIGIN_PROJECT_TAG_KEYS`` is what
    STAMPS an origin tag onto a nested child; :data:`FOREIGN_PROJECT_TAG_KEYS`
    is what READS it back here. Neither side can import the other — the
    orchestrator declares no runtime dependency on fused-memory (it appears
    only in ``orchestrator/pyproject.toml``'s ``[tool.pyright] extraPaths``, a
    type-checking-only reference) — so the coupling is by COPY, and until this
    test existed it rested entirely on two prose comments.

    Drift is SILENT and one-directional: a key added to the reader alone is
    harmless, but a key added to the reader that grouped_read never stamps (or
    a key dropped from the stamper) makes the nested safeguard stop firing on
    it, and a safeguard that never fires returns ``nested_dropped == 0`` —
    indistinguishable from "nothing foreign was found". That is exactly the
    failure this file otherwise cannot detect.

    The stamper is loaded BY PATH from this worktree, not through the normal
    import machinery, so the assertion compares THIS branch's stamper against
    THIS branch's reader. Resolving ``fused_memory`` by import name instead
    picks up whichever editable install the active virtualenv points at — on
    a task worktree that is the MAIN checkout, whose ``grouped_read`` predates
    this branch and carries no ``ORIGIN_PROJECT_TAG_KEYS`` at all. Measured:
    under ``/home/leo/src/dark-factory/.venv`` the import resolved to
    ``<main checkout>/fused-memory/.../grouped_read.py`` and this test raised
    ``AttributeError``, while under the worktree's own ``.venv`` it resolved
    in-tree and passed — i.e. by-name resolution made the guard's verdict a
    property of the ENVIRONMENT rather than of the two constants, and would
    equally report a spurious GREEN once this work lands on main.
    """

    def test_grouped_read_mirrors_the_briefing_tag_keys(self):
        # Load the IN-TREE stamper by file location. parents[2] of
        # ``orchestrator/tests/<this file>`` is the worktree root, mirroring
        # ``tests/conftest.py``'s documented "local src takes precedence"
        # intent, which front-loads orchestrator/shared/escalation src but
        # deliberately not fused-memory/src. Kept inside the test body rather
        # than at module scope: grouped_read's absolute ``fused_memory.*``
        # imports drag in graphiti_core, and every xdist worker would pay that
        # cost merely to COLLECT the other tests in this file.
        fm_grouped_read = (
            Path(__file__).resolve().parents[2]
            / 'fused-memory'
            / 'src'
            / 'fused_memory'
            / 'server'
            / 'grouped_read.py'
        )
        if not fm_grouped_read.exists():
            pytest.skip(f'fused-memory sibling package not present at {fm_grouped_read}')

        spec = importlib.util.spec_from_file_location('_fm_grouped_read', fm_grouped_read)
        assert spec is not None and spec.loader is not None
        grouped_read = importlib.util.module_from_spec(spec)
        try:
            # ImportError ONLY. That covers the legitimate gap this guard is
            # allowed to skip on — fused-memory's own dependencies absent in an
            # orchestrator-only environment. It must NOT cover AttributeError
            # on the constant itself: a missing stamper key IS the drift this
            # test exists to catch, and skipping on it would rebuild the
            # original defect in a quieter form. The module is never inserted
            # into ``sys.modules``, so the rest of the session is unaffected.
            spec.loader.exec_module(grouped_read)
        except ImportError as exc:
            pytest.skip(f'fused-memory not importable in this environment: {exc}')

        # Provenance: pins that the tuple just read really is this worktree's.
        # Without it, an editable install rooted at the MAIN checkout silently
        # shadows the branch's source — the exact way this test first went red.
        # Bound to a local first: ``ModuleType.__file__`` is typed
        # ``str | None`` (a namespace/builtin module has none), so feeding it
        # straight to ``Path()`` is a type error even though
        # ``spec_from_file_location`` always populates it. Asserting rather
        # than defaulting keeps an unpopulated ``__file__`` a FAILURE of the
        # provenance check, never a silent pass.
        loaded_file = grouped_read.__file__
        assert loaded_file is not None, (
            'the loaded module has no __file__, so its provenance cannot be '
            'established and the branch-against-branch comparison below would '
            'be unverifiable.'
        )
        assert Path(loaded_file).resolve() == fm_grouped_read.resolve(), (
            'the stamper was loaded from outside this worktree, so the '
            'comparison below would not be branch-against-branch. '
            f'loaded={loaded_file!r} expected={str(fm_grouped_read)!r}'
        )

        assert grouped_read.ORIGIN_PROJECT_TAG_KEYS == FOREIGN_PROJECT_TAG_KEYS, (
            'grouped_read.ORIGIN_PROJECT_TAG_KEYS (the STAMPER) and '
            'briefing.FOREIGN_PROJECT_TAG_KEYS (the READER) are hand-copies of '
            'one another and have drifted. Tuple equality is asserted, not set '
            'equality, so PRECEDENCE order is pinned too: src_project must stay '
            'first, or a CGL-eta rehomed record whose co-present project_id '
            'names the local project would be certified as native. Add the new '
            f'key to BOTH sides. stamper={grouped_read.ORIGIN_PROJECT_TAG_KEYS!r} '
            f'reader={FOREIGN_PROJECT_TAG_KEYS!r}'
        )


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

    async def test_a_nested_only_drop_is_reported_as_nested_records(
        self, briefing: BriefingAssembler, caplog,
    ):
        """The rendered note must not call a dropped CHILD a dropped result slot.

        Task 4008 amendment. Here every top-level result survives and only
        children inside them are removed, so a note reading "N memory result
        slot(s) ... filtered out" would tell an operator that N recalled facts
        vanished when in fact none did. The note is the ONLY signal a human
        gets that a leak was blocked, so it names the two quantities apart.
        """
        envelope = _mcp_search_envelope([_grouped_parent()])
        # One foreign amendment + one foreign pinned body per query, and the
        # stub answers all four queries identically.
        expected_nested = 2 * 4

        with caplog.at_level(logging.INFO), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            context = await briefing._get_memory_context('3609')

        assert 'FOREIGN AMENDMENT BODY' not in context
        assert 'FOREIGN SIGHTING BODY' not in context
        assert 'Native canonical.' in context, (
            'The parent itself was never foreign and must still be recalled'
        )
        assert (
            f'{expected_nested} nested memory record(s) across 4 queries were '
            'tagged to another project and filtered out'
        ) in context, (
            f'a nested-only drop must be reported as nested records, got {context!r}'
        )
        assert 'memory result slot(s)' not in context, (
            'No top-level result was dropped, so the note must not claim a '
            f'result slot was vacated, got {context!r}'
        )
        assert any(r.levelno == logging.INFO for r in caplog.records)

    async def test_mixed_drops_name_both_quantities_separately(
        self, briefing: BriefingAssembler, caplog,
    ):
        """A foreign result AND a foreign child are reported as distinct counts."""
        envelope = _mcp_search_envelope([
            _result('f1', 'Foreign fact.', metadata={'project_id': 'reify'}),
            _grouped_parent(),
        ])

        with caplog.at_level(logging.INFO), patch(
            'orchestrator.agents.briefing.mcp_call', new=AsyncMock(return_value=envelope),
        ):
            context = await briefing._get_memory_context('3609')

        assert 'Foreign fact.' not in context
        assert 'FOREIGN AMENDMENT BODY' not in context
        assert (
            '4 memory result slot(s) and 8 nested memory record(s) across 4 '
            'queries were tagged to another project and filtered out'
        ) in context, (
            f'both quantities must be named, and named apart, got {context!r}'
        )


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
            return f'## recalled for: {query}', 0, 0

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

        assert result == (None, 0, 0)

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
            text, dropped, _nested = await briefing._scoped_search('anything')

        # Known limitation: the joined multi-block text isn't valid JSON, so
        # the filter fails open rather than filtering each block — the
        # foreign fact is NOT removed.
        assert dropped == 0
        assert text is not None
        assert 'Foreign fact.' in text
        assert 'Own fact.' in text
        assert any(r.levelno == logging.WARNING for r in caplog.records)


def _grouped_parent(id_: str = 'p1', *, grouped: dict | None = None) -> dict:
    """A NATIVE canonical hit carrying a ``grouped`` block, as the server nests it.

    Mirrors what ``fused_memory.server.grouped_read.group_search_results``
    emits: the block is hung on a KEPT parent entry at ``entry['grouped']``
    (grouped_read.py:716-718), with bounded amendment digests under
    ``amendments`` and full swallowed bodies under ``matched_children``.
    """
    entry = _result('x', 'placeholder', metadata=None, source_store='mem0')
    entry['id'] = id_
    entry['content'] = 'Native canonical.'
    entry['metadata'] = {'project_id': 'dark_factory'}
    entry['grouped'] = _grouped_block() if grouped is None else grouped
    return entry


def _grouped_block() -> dict:
    return {
        'amendments': [
            {'id': 'a1', 'digest': 'FOREIGN AMENDMENT BODY', 'created_at': None,
             'kind': 'amendment', 'metadata': {'src_project': 'reify'}},
            {'id': 'a2', 'digest': 'NATIVE AMENDMENT BODY', 'created_at': None,
             'kind': 'amendment', 'metadata': {'project_id': 'dark_factory'}},
            {'id': 'a3', 'digest': 'UNTAGGED AMENDMENT BODY', 'created_at': None,
             'kind': 'amendment'},
        ],
        'matched_children': [
            {'id': 's1', 'content': 'FOREIGN SIGHTING BODY', 'created_at': None,
             'kind': 'sighting', 'matched': True, 'metadata': {'src_project': 'reify'}},
        ],
        'amendment_count': 3,
        'sighting_count': 1,
    }


class TestGroupedChildrenAreFiltered:
    """task 4008: a foreign child nested under a NATIVE canonical must not survive.

    ``group_search_results`` nests child data inside a KEPT parent entry, and
    ``_get_memory_context`` appends the filtered JSON verbatim into the
    ``# Context`` block — so a ``grouped`` sub-object renders as raw JSON in a
    dispatched agent's prompt, foreign digest text and foreign pinned bodies
    included.  Reading only the TOP-LEVEL tag lets every one of them through.

    SCOPE: ``src_project`` is the task-2273 CGL-eta rehome shape (see
    :data:`FOREIGN_PROJECT_TAG_KEYS`) — a record physically in dark_factory's
    collection naming a different ORIGIN project.  That is the ONLY reachable
    leak shape here, because ``_read_grouped_document``
    (``fused_memory/server/grouped_read.py``:275-305) already scopes every
    child read by ``project_id``, so a child from a foreign COLLECTION cannot
    appear in a block at all.
    """

    def test_foreign_nested_children_are_dropped_from_a_kept_parent(self):
        payload = json.dumps({'results': [_grouped_parent()]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        # (a) THE SUBSTANTIVE PIN: neither foreign body reaches the prompt.
        assert 'FOREIGN AMENDMENT BODY' not in text, (
            'A foreign-tagged amendment digest nested under a native canonical '
            f'must not survive into the # Context block, got {text!r}'
        )
        assert 'FOREIGN SIGHTING BODY' not in text, (
            'A foreign-tagged pinned body — the FULL text, not a digest — must '
            f'not survive into the # Context block, got {text!r}'
        )
        # (b) The correctly-tagged parent is NOT collateral damage.
        assert [r['id'] for r in json.loads(text)['results']] == ['p1']
        # (c) A native child survives.
        assert 'NATIVE AMENDMENT BODY' in text
        # (d) Nested entries inherit the top-level KEEP-UNTAGGED policy.
        assert 'UNTAGGED AMENDMENT BODY' in text, (
            'An untagged nested child must be KEPT, exactly as an untagged '
            f'top-level result is — not held to a stricter policy, got {text!r}'
        )
        # (e) Exactly the foreign children are gone.
        grouped = json.loads(text)['results'][0]['grouped']
        assert [c['id'] for c in grouped['amendments']] == ['a2', 'a3']
        assert grouped['matched_children'] == []
        # (f) Nested drops are COUNTED — in their own counter, so they reach
        # the drop note as nested records rather than as vacated result slots.
        assert nested == 2, (
            'Nested drops must be counted and returned, so a blocked leak is '
            f'reported rather than silently swallowed, got {nested}'
        )
        assert dropped == 0, (
            'No TOP-LEVEL result was foreign here — the parent survived — so '
            f'the result-slot counter must stay at 0, got {dropped}'
        )

    def test_a_nested_only_drop_defeats_the_no_op_fast_path(self):
        """Every top-level entry is native or untagged; only a CHILD is foreign.

        Pins that the ``dropped == 0`` fast path (which returns ``payload_text``
        byte-for-byte unchanged) cannot swallow a nested-only drop.
        """
        payload = json.dumps({
            'results': [
                _result('u1', 'Untagged graphiti fact.'),
                _grouped_parent(),
            ],
        })

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert (dropped, nested) == (0, 2)
        assert 'FOREIGN AMENDMENT BODY' not in text
        assert 'FOREIGN SIGHTING BODY' not in text
        assert text != payload, (
            'The payload must be genuinely re-serialised when only a nested '
            'entry was dropped — returning the unfiltered text would discard '
            'the drop entirely'
        )
        assert [r['id'] for r in json.loads(text)['results']] == ['u1', 'p1']

    def test_a_single_nested_drop_is_reported_and_re_serialised(self):
        """The minimal nested-only case: exactly ONE foreign child, nothing else."""
        payload = json.dumps({
            'results': [_grouped_parent(grouped={
                'amendments': [
                    {'id': 'a1', 'digest': 'FOREIGN AMENDMENT BODY', 'created_at': None,
                     'kind': 'amendment', 'metadata': {'src_project': 'reify'}},
                ],
                'amendment_count': 1,
                'sighting_count': 0,
            })],
        })

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert (dropped, nested) == (0, 1)
        assert 'FOREIGN AMENDMENT BODY' not in text
        assert json.loads(text)['results'][0]['grouped']['amendments'] == []


    def test_nested_children_canonicalise_divergent_spellings(self):
        """The descent must use ``_canonical_project``, not a raw string compare.

        fused-memory explicitly PERMITS divergent project-id spellings (see
        ``plans/cross-graph-entity-leak-prd.md`` decision 1 / S1), and the
        top-level loop already canonicalises before comparing
        (``TestForeignTagKeysAndSpelling``). If the nested loop compared raw
        tags instead, a native child tagged ``dark-factory`` would be
        FALSE-POSITIVE DROPPED — silent context loss on real corpus data,
        which is strictly worse than the leak this task set out to close, and
        every other test in this file would still pass.
        """
        for spelling in ('dark-factory', 'Dark_Factory', '  dark_factory  '):
            payload = json.dumps({'results': [_grouped_parent(grouped={
                'amendments': [
                    {'id': 'a1', 'digest': 'NATIVE AMENDMENT BODY', 'created_at': None,
                     'kind': 'amendment', 'metadata': {'project_id': spelling}},
                ],
                'amendment_count': 1,
                'sighting_count': 0,
            })]})

            text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

            assert (dropped, nested) == (0, 0), (
                f'nested spelling {spelling!r} must be treated as own-project, '
                f'got dropped={dropped} nested={nested}'
            )
            assert 'NATIVE AMENDMENT BODY' in text, (
                f'A native child spelled {spelling!r} must survive, got {text!r}'
            )

    def test_a_foreign_nested_child_is_dropped_whatever_its_casing(self):
        """The same canonicalisation must not let a FOREIGN child through either."""
        payload = json.dumps({'results': [_grouped_parent(grouped={
            'amendments': [
                {'id': 'a1', 'digest': 'FOREIGN AMENDMENT BODY', 'created_at': None,
                 'kind': 'amendment', 'metadata': {'src_project': '  REIFY  '}},
            ],
            'amendment_count': 1,
            'sighting_count': 0,
        })]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert (dropped, nested) == (0, 1)
        assert 'FOREIGN AMENDMENT BODY' not in text, (
            'Canonicalisation must normalise the FOREIGN side too — a tag of '
            f"'  REIFY  ' still names another project, got {text!r}"
        )

    def test_a_dropped_parent_is_counted_once_not_once_per_child(self):
        """A foreign parent takes its whole subtree with it — for exactly 1 drop.

        The descent runs only for a KEPT entry. If the
        ``_filter_grouped_children`` call were ever moved above the
        ``continue`` that drops a foreign parent, this parent would score
        1 + 3 = 4 instead of 1 — and ``dropped`` is not internal bookkeeping,
        it renders into the operator-visible drop note, so an over-count is a
        wrong number shown to a human.
        """
        entry = _result('f1', 'FOREIGN CANONICAL.', metadata={'src_project': 'reify'},
                        source_store='mem0')
        entry['grouped'] = {
            'amendments': [
                {'id': 'a1', 'digest': 'FOREIGN CHILD ONE', 'kind': 'amendment',
                 'metadata': {'src_project': 'reify'}},
                {'id': 'a2', 'digest': 'FOREIGN CHILD TWO', 'kind': 'amendment',
                 'metadata': {'src_project': 'reify'}},
            ],
            'matched_children': [
                {'id': 's1', 'content': 'FOREIGN CHILD THREE', 'kind': 'sighting',
                 'matched': True, 'metadata': {'src_project': 'reify'}},
            ],
            'amendment_count': 2,
            'sighting_count': 1,
        }
        payload = json.dumps({'results': [entry, _result('n1', 'Native fact.')]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert dropped == 1, (
            'A dropped parent is ONE drop however many children hung off it, '
            f'got {dropped}'
        )
        assert nested == 0, (
            'A dropped parent must not be descended into: its children are '
            f'already gone, so counting them would double-count, got {nested}'
        )
        assert [r['id'] for r in json.loads(text)['results']] == ['n1']
        for body in ('FOREIGN CANONICAL.', 'FOREIGN CHILD ONE',
                     'FOREIGN CHILD TWO', 'FOREIGN CHILD THREE'):
            assert body not in text, f'{body!r} must go with its parent'


class TestGroupedDescentIsSurgicalAndFailsOpen:
    """The descent rewrites the child LISTS and nothing else, and never raises.

    Surgical: the server's ``grouped`` block carries EXACT counts from
    ``count_memories_by_metadata``; a briefing that recomputed them after a
    drop would be fabricating a number the store never returned, which is a
    worse lie than a visibly-short list.

    Fails open: mirroring the malformed-payload arms of
    ``filter_foreign_project_results``, a shape surprise must never blank the
    ``# Context`` block — a silent capability loss across every prompt builder
    is the worse failure direction than one unfiltered entry.
    """

    def test_the_server_counts_are_never_rewritten(self):
        """(a) A shortened list keeps the count API's EXACT value."""
        payload = json.dumps({'results': [_grouped_parent()]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        grouped = json.loads(text)['results'][0]['grouped']
        assert nested == 2
        assert len(grouped['amendments']) == 2, 'PRECONDITION: the list really did shrink'
        assert grouped['amendment_count'] == 3, (
            'amendment_count is the count API\'s exact value; recomputing it here '
            f'would fabricate a number the store never returned, got {grouped!r}'
        )
        assert grouped['sighting_count'] == 1, (
            f'sighting_count must survive a matched_children drop verbatim, got {grouped!r}'
        )

    def test_sibling_grouped_keys_survive_verbatim(self):
        """(b) Only the child lists are touched."""
        block = _grouped_block()
        block['truncated'] = True
        block['children_unavailable'] = True
        block['error_type'] = 'TimeoutError'
        payload = json.dumps({'results': [_grouped_parent(grouped=block)]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        grouped = json.loads(text)['results'][0]['grouped']
        assert nested == 2
        assert grouped['truncated'] is True
        assert grouped['children_unavailable'] is True
        assert grouped['error_type'] == 'TimeoutError'

    def test_the_parents_own_fields_are_untouched(self):
        """(c) The descent edits the subtree, never the entry that carries it."""
        payload = json.dumps({'results': [_grouped_parent()]})

        text, _dropped, _nested = filter_foreign_project_results(payload, 'dark_factory')

        parent = json.loads(text)['results'][0]
        assert parent['id'] == 'p1'
        assert parent['content'] == 'Native canonical.'
        assert parent['metadata'] == {'project_id': 'dark_factory'}
        assert parent['relevance_score'] == 0.9

    def test_an_entry_with_no_grouped_key_is_a_no_op(self):
        """(d) Today's overwhelmingly common shape: zero child records in the corpus.

        ``build_grouped_document`` returns None on a zero-child canonical, so
        ``group_search_results`` sets no ``grouped`` key at all.
        """
        payload = json.dumps({
            'results': [_result('n1', 'Native fact.', metadata={'project_id': 'dark_factory'})],
        })

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert (dropped, nested) == (0, 0)
        assert 'Native fact.' in text

    def test_a_non_dict_grouped_value_fails_open(self):
        """(e)"""
        entry = _result('p1', 'Native canonical.', metadata={'project_id': 'dark_factory'})
        entry['grouped'] = 'not a dict at all'
        payload = json.dumps({'results': [entry]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert (dropped, nested) == (0, 0), (
            f'A shape surprise must not be counted as a drop, got {dropped}/{nested}'
        )
        assert [r['id'] for r in json.loads(text)['results']] == ['p1'], (
            f'The entry must SURVIVE a malformed grouped block, got {text!r}'
        )

    def test_a_non_list_child_collection_fails_open(self):
        """(f)"""
        entry = _result('p1', 'Native canonical.', metadata={'project_id': 'dark_factory'})
        entry['grouped'] = {'amendments': 'not a list', 'amendment_count': 1}
        payload = json.dumps({'results': [entry]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert (dropped, nested) == (0, 0)
        assert json.loads(text)['results'][0]['grouped']['amendments'] == 'not a list', (
            'A child collection of the wrong type is left EXACTLY as received — '
            'never coerced, never emptied'
        )

    def test_a_non_dict_nested_entry_is_kept_not_dropped(self):
        """(g) Same treatment the top-level loop gives a stray non-dict entry."""
        entry = _result('p1', 'Native canonical.', metadata={'project_id': 'dark_factory'})
        entry['grouped'] = {
            'amendments': [
                None,
                'a bare string',
                {'id': 'a1', 'digest': 'FOREIGN AMENDMENT BODY', 'kind': 'amendment',
                 'metadata': {'src_project': 'reify'}},
                {'id': 'a2', 'digest': 'NATIVE AMENDMENT BODY', 'kind': 'amendment'},
            ],
            'amendment_count': 4,
        }
        payload = json.dumps({'results': [entry]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert (dropped, nested) == (0, 1), (
            f'Only the classifiably-foreign child may be dropped, got {dropped}/{nested}'
        )
        amendments = json.loads(text)['results'][0]['grouped']['amendments']
        assert amendments[:2] == [None, 'a bare string'], (
            f'An unclassifiable nested entry is KEPT, not dropped, got {amendments!r}'
        )
        assert 'FOREIGN AMENDMENT BODY' not in text
        assert 'NATIVE AMENDMENT BODY' in text

    def test_a_non_dict_nested_metadata_is_kept(self):
        """(h) ``_result_project`` already returns None for that — pinned end to end."""
        entry = _result('p1', 'Native canonical.', metadata={'project_id': 'dark_factory'})
        entry['grouped'] = {
            'amendments': [
                {'id': 'a1', 'digest': 'ODDLY SHAPED BODY', 'kind': 'amendment',
                 'metadata': 'not a dict'},
            ],
            'amendment_count': 1,
        }
        payload = json.dumps({'results': [entry]})

        text, dropped, nested = filter_foreign_project_results(payload, 'dark_factory')

        assert (dropped, nested) == (0, 0)
        assert 'ODDLY SHAPED BODY' in text, (
            'A nested entry whose metadata is unreadable is untagged, and '
            f'untagged means KEPT, got {text!r}'
        )
