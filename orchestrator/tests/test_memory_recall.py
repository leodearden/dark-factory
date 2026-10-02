"""Pure tests for ``orchestrator.agents.memory_recall``.

The data layer parses one ``search`` reply once into a typed result, drops
cross-project facts from it, and renders what survives; the composition layer
turns one dispatch's tally of sections, notices and outcomes into the
``# Context`` block. Every test here drives those functions directly; the
end-to-end behaviour, through the public prompt builders, lives in
``test_briefing_project_scope.py``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from enum import Enum
from pathlib import Path

import pytest
from _briefing_helpers import _edge, _grouped_block, _grouped_parent, _node, _result

from orchestrator.agents.memory_recall import (
    FOREIGN_PROJECT_TAG_KEYS,
    MEMORY_CONTEXT_CAVEAT,
    MEMORY_EMPTY_NOTICE,
    MEMORY_OUTAGE_NOTICE,
    FilteredResults,
    MemoryFailure,
    MemoryQueryOutcome,
    RecallTally,
    SearchReply,
    UnparsedReply,
    filter_foreign_project_results,
    parse_search_reply,
    render_context_block,
    render_entity_block,
    render_memory_results,
)

PROJECT = 'dark_factory'


def _filter(results: list) -> FilteredResults:
    return filter_foreign_project_results(results, PROJECT)


def _ids(kept: tuple) -> list:
    return [entry['id'] for entry in kept]


def _all_text(kept: tuple) -> str:
    """Everything the kept results carry, for asserting a body is gone at any depth."""
    return json.dumps(list(kept), ensure_ascii=False)


class TestMemoryFailure:
    def test_the_reason_classes_are_an_enum_whose_values_the_notices_render(self):
        assert issubclass(MemoryFailure, Enum)
        assert {member.name: member.value for member in MemoryFailure} == {
            'TIMEOUT': 'timeout',
            'TRANSPORT': 'transport',
            'MALFORMED': 'malformed',
        }


class TestParseSearchReply:
    """One ``json.loads`` per reply, into one of three genuinely different outcomes."""

    def test_a_results_list_is_a_search_reply(self):
        r1 = _result('1', 'First.')
        r2 = _result('2', 'Second.')

        reply = parse_search_reply(json.dumps({'results': [r1, r2]}))

        assert isinstance(reply, SearchReply)
        assert reply.results == (r1, r2)
        assert reply.failed_stores == ()

    def test_a_degraded_reply_names_its_failed_string_stores(self):
        reply = parse_search_reply(json.dumps({
            'results': [], 'degraded': True, 'failed_stores': ['graphiti', 3, ''],
        }))

        assert isinstance(reply, SearchReply)
        assert reply.failed_stores == ('graphiti',)

    @pytest.mark.parametrize('degraded', [None, False])
    def test_failed_stores_without_degraded_report_nothing(self, degraded):
        payload: dict = {'results': [], 'failed_stores': ['graphiti']}
        if degraded is not None:
            payload['degraded'] = degraded

        reply = parse_search_reply(json.dumps(payload))

        assert isinstance(reply, SearchReply)
        assert reply.failed_stores == ()

    def test_a_non_list_failed_stores_reports_nothing(self):
        reply = parse_search_reply(json.dumps({
            'results': [], 'degraded': True, 'failed_stores': 'graphiti',
        }))

        assert isinstance(reply, SearchReply)
        assert reply.failed_stores == ()

    def test_the_reported_stores_survive_a_filter_that_keeps_nothing(self):
        """A partial store outage is owed to the reader even when every
        result was foreign, so it is read from the reply, not the survivors."""
        reply = parse_search_reply(json.dumps({
            'results': [_result('1', 'Foreign.', metadata={'project_id': 'reify'})],
            'degraded': True,
            'failed_stores': ['mem0'],
        }))

        assert isinstance(reply, SearchReply)
        assert _filter(list(reply.results)).kept == ()
        assert reply.failed_stores == ('mem0',)

    @pytest.mark.parametrize(
        'text',
        [
            '[1, 2, 3]',
            json.dumps({'degraded': True}),
            json.dumps({'results': 'oops'}),
            json.dumps({'error': "Invalid project_id 'foo-bar'", 'error_type': 'ValidationError'}),
        ],
        ids=['json-array', 'no-results-key', 'non-list-results', 'server-rejection'],
    )
    def test_json_that_is_no_results_list_is_malformed(self, text):
        assert parse_search_reply(text) is MemoryFailure.MALFORMED

    @pytest.mark.parametrize(
        'text',
        [
            'not json at all',
            json.dumps({'results': [_result('1', 'Foreign fact.')]})
            + '\n'
            + json.dumps({'results': [_result('2', 'Own fact.')]}),
        ],
        ids=['plain-text', 'two-joined-text-blocks'],
    )
    def test_text_that_is_not_json_fails_open_unchanged(self, text):
        assert parse_search_reply(text) == UnparsedReply(text)


class TestFilterForeignProjectResults:
    """Unit coverage for the pure metadata-tag filter."""

    def test_foreign_tagged_result_is_dropped(self):
        filtered = _filter([
            _result('1', 'Foreign fact about reify.', metadata={'project_id': 'reify'}),
        ])

        assert filtered.kept == ()
        assert filtered.dropped == 1

    def test_own_project_tagged_result_is_kept(self):
        filtered = _filter([_result('1', 'Own project fact.', metadata={'project_id': PROJECT})])

        assert _ids(filtered.kept) == ['1']
        assert filtered.dropped == 0

    def test_untagged_result_is_kept_and_not_counted(self):
        filtered = _filter([_result('1', 'Untagged fact.', metadata={})])

        assert _ids(filtered.kept) == ['1']
        assert filtered.dropped == 0

    def test_all_foreign_keeps_nothing_and_counts_every_drop(self):
        filtered = _filter([
            _result('1', 'Foreign one.', metadata={'project_id': 'reify'}),
            _result('2', 'Foreign two.', metadata={'project_id': 'other_proj'}),
        ])

        assert filtered == FilteredResults(kept=(), dropped=2, nested_dropped=0)

    def test_kept_results_keep_their_original_order(self):
        filtered = _filter([
            _result('1', 'First own fact.', metadata={'project_id': PROJECT}),
            _result('2', 'Foreign fact.', metadata={'project_id': 'reify'}),
            _result('3', 'Second own fact.', metadata={}),
        ])

        assert _ids(filtered.kept) == ['1', '3']
        assert filtered.dropped == 1

    def test_nothing_dropped_keeps_every_entry_as_received(self):
        results = [_result('1', 'Own — café fact.', metadata={'project_id': PROJECT})]

        filtered = _filter(results)

        assert filtered == FilteredResults(kept=tuple(results), dropped=0, nested_dropped=0)

    def test_dropped_entry_is_logged_at_debug_with_id_key_and_value(self, caplog):
        """A false-positive drop must be diagnosable from an existing run's logs."""
        with caplog.at_level(logging.DEBUG):
            _filter([_result('42', 'Foreign fact.', metadata={'project_id': 'reify'})])

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(
            '42' in r.getMessage() and 'project_id' in r.getMessage() and 'reify' in r.getMessage()
            for r in debug_records
        )

    def test_non_dict_entry_is_kept_without_raising(self):
        filtered = _filter(['stray', _result('1', 'Own fact.', metadata={'project_id': PROJECT})])

        assert 'stray' in filtered.kept
        assert filtered.dropped == 0

    def test_non_dict_metadata_is_kept_without_raising(self):
        entry_none_meta = _result('1', 'Fact with null metadata.')
        entry_none_meta['metadata'] = None
        entry_list_meta = _result('2', 'Fact with list metadata.')
        entry_list_meta['metadata'] = ['oops']

        filtered = _filter([entry_none_meta, entry_list_meta])

        assert _ids(filtered.kept) == ['1', '2']
        assert filtered.dropped == 0


class TestForeignTagKeysAndSpelling:
    """Tag extraction must recognise more than the bare ``project_id`` key,
    prefer ``src_project`` for CGL-eta rehomed entries, and canonicalise
    divergent spellings so ``dark-factory`` isn't mistaken for a foreign
    project.
    """

    def test_group_id_tag_is_dropped(self):
        filtered = _filter([_result('1', 'Fact.', metadata={'group_id': 'reify'})])

        assert (filtered.kept, filtered.dropped) == ((), 1)

    def test_project_tag_is_dropped(self):
        filtered = _filter([_result('1', 'Fact.', metadata={'project': 'reify'})])

        assert (filtered.kept, filtered.dropped) == ((), 1)

    def test_cgl_eta_rehome_production_case_is_dropped(self):
        """The one foreign-tag channel demonstrably present in production data.

        Task-2273 CGL-eta rehomed entries physically live in dst_project's
        Mem0 collection but reference src_project's task numbers — src_project
        is the authoritative origin.
        """
        filtered = _filter([_result(
            '1',
            "A park on 'crates/reify-compiler/src' blocks acquire of 'crates/reify-compiler'.",
            metadata={
                'kind': 'cgl_eta_cross_target_rehome',
                'src_project': 'reify',
                'dst_project': PROJECT,
                'src_entity': 'Task 2310',
            },
        )])

        assert 'crates/reify-compiler' not in _all_text(filtered.kept)
        assert filtered.dropped == 1

    def test_src_project_wins_over_same_entry_project_id(self):
        filtered = _filter([_result(
            '1', 'Fact.', metadata={'src_project': 'reify', 'project_id': PROJECT},
        )])

        assert (filtered.kept, filtered.dropped) == ((), 1)

    def test_dst_project_alone_is_never_consulted_and_is_kept(self):
        filtered = _filter([_result('1', 'Fact.', metadata={'dst_project': PROJECT})])

        assert (_ids(filtered.kept), filtered.dropped) == (['1'], 0)

    def test_first_present_key_precedence(self):
        filtered = _filter([_result(
            '1', 'Fact.', metadata={'project_id': PROJECT, 'group_id': 'reify'},
        )])

        assert (_ids(filtered.kept), filtered.dropped) == (['1'], 0)

    def test_canonicalisation_of_divergent_spellings(self):
        for spelling in ('dark-factory', 'Dark_Factory', '  dark_factory  '):
            filtered = _filter([_result('1', 'Fact.', metadata={'project_id': spelling})])

            assert _ids(filtered.kept) == ['1'], (
                f'spelling {spelling!r} should be treated as own-project'
            )
            assert filtered.dropped == 0

        filtered = _filter([_result('1', 'Fact.', metadata={'project_id': 'reify'})])
        assert (filtered.kept, filtered.dropped) == ((), 1)

    def test_non_string_tag_is_treated_as_untagged(self):
        filtered = _filter([_result('1', 'Fact.', metadata={'project_id': 123})])

        assert (_ids(filtered.kept), filtered.dropped) == (['1'], 0)


class TestOriginTagKeyDriftGuard:
    """The two hand-copied tag tuples must stay identical, key for key.

    ``fused_memory.server.grouped_read.ORIGIN_PROJECT_TAG_KEYS`` is what
    STAMPS an origin tag onto a nested child; :data:`FOREIGN_PROJECT_TAG_KEYS`
    is what READS it back here. Neither side can import the other — the
    orchestrator declares no runtime dependency on fused-memory — so the
    coupling is by COPY.

    Drift is SILENT and one-directional: a key the reader expects that the
    stamper never writes (or a key dropped from the stamper) makes the nested
    safeguard stop firing on it, and a safeguard that never fires returns
    ``nested_dropped == 0`` — indistinguishable from "nothing foreign was
    found".

    The stamper is loaded BY PATH from this worktree, not through the normal
    import machinery, so the assertion compares THIS branch's stamper against
    THIS branch's reader. Resolving ``fused_memory`` by import name instead
    picks up whichever editable install the active virtualenv points at,
    which on a task worktree can be the MAIN checkout — making the guard's
    verdict a property of the environment rather than of the two constants.
    """

    def test_grouped_read_mirrors_the_reader_tag_keys(self):
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
            'memory_recall.FOREIGN_PROJECT_TAG_KEYS (the READER) are hand-copies '
            'of one another and have drifted. Tuple equality is asserted, not set '
            'equality, so PRECEDENCE order is pinned too: src_project must stay '
            'first, or a CGL-eta rehomed record whose co-present project_id '
            'names the local project would be certified as native. Add the new '
            f'key to BOTH sides. stamper={grouped_read.ORIGIN_PROJECT_TAG_KEYS!r} '
            f'reader={FOREIGN_PROJECT_TAG_KEYS!r}'
        )


class TestGroupedChildrenAreFiltered:
    """A foreign child nested under a NATIVE canonical must not survive (task 4008).

    ``group_search_results`` nests child data inside a KEPT parent entry and
    every child renders into the ``# Context`` block, so reading only the
    TOP-LEVEL tag would let a foreign digest or pinned body through.

    SCOPE: ``src_project`` is the task-2273 CGL-eta rehome shape — a record
    physically in this project's collection naming a different ORIGIN
    project. That is the only reachable leak shape here, because the server
    already scopes every child read by ``project_id``.
    """

    def test_foreign_nested_children_are_dropped_from_a_kept_parent(self):
        filtered = _filter([_grouped_parent()])
        text = _all_text(filtered.kept)

        # (a) THE SUBSTANTIVE PIN: neither foreign body survives.
        assert 'FOREIGN AMENDMENT BODY' not in text
        assert 'FOREIGN SIGHTING BODY' not in text
        # (b) The correctly-tagged parent is NOT collateral damage.
        assert _ids(filtered.kept) == ['p1']
        # (c)+(d) A native child survives, and so does an untagged one: nested
        # entries inherit the top-level KEEP-UNTAGGED policy.
        grouped = filtered.kept[0]['grouped']
        assert [c['id'] for c in grouped['amendments']] == ['a2', 'a3']
        # (e) Exactly the foreign children are gone.
        assert grouped['matched_children'] == []
        # (f) Nested drops are COUNTED in their own counter, so they reach the
        # drop note as nested records rather than as vacated result slots.
        assert filtered.nested_dropped == 2
        assert filtered.dropped == 0

    def test_a_nested_only_drop_is_still_reported(self):
        """Every top-level entry is native or untagged; only a CHILD is foreign."""
        filtered = _filter([_result('u1', 'Untagged graphiti fact.'), _grouped_parent()])

        assert (filtered.dropped, filtered.nested_dropped) == (0, 2)
        assert 'FOREIGN AMENDMENT BODY' not in _all_text(filtered.kept)
        assert 'FOREIGN SIGHTING BODY' not in _all_text(filtered.kept)
        assert _ids(filtered.kept) == ['u1', 'p1']

    def test_a_single_nested_drop_is_reported(self):
        """The minimal nested-only case: exactly ONE foreign child, nothing else."""
        filtered = _filter([_grouped_parent(grouped={
            'amendments': [
                {'id': 'a1', 'digest': 'FOREIGN AMENDMENT BODY', 'created_at': None,
                 'kind': 'amendment', 'metadata': {'src_project': 'reify'}},
            ],
            'amendment_count': 1,
            'sighting_count': 0,
        })])

        assert (filtered.dropped, filtered.nested_dropped) == (0, 1)
        assert filtered.kept[0]['grouped']['amendments'] == []

    def test_the_callers_entry_is_never_mutated(self):
        """The reply is parsed once and shared, so the filter must not edit it."""
        parent = _grouped_parent()

        filtered = _filter([parent])

        assert filtered.nested_dropped == 2
        assert [c['id'] for c in parent['grouped']['amendments']] == ['a1', 'a2', 'a3']
        assert len(parent['grouped']['matched_children']) == 1
        assert filtered.kept[0] is not parent
        assert filtered.kept[0]['grouped'] is not parent['grouped']

    def test_nested_children_canonicalise_divergent_spellings(self):
        """The descent must canonicalise, not compare raw strings.

        fused-memory explicitly permits divergent project-id spellings, so a
        raw compare would FALSE-POSITIVE DROP a native child tagged
        ``dark-factory`` — silent context loss, which is strictly worse than
        the leak the descent exists to close.
        """
        for spelling in ('dark-factory', 'Dark_Factory', '  dark_factory  '):
            filtered = _filter([_grouped_parent(grouped={
                'amendments': [
                    {'id': 'a1', 'digest': 'NATIVE AMENDMENT BODY', 'created_at': None,
                     'kind': 'amendment', 'metadata': {'project_id': spelling}},
                ],
                'amendment_count': 1,
                'sighting_count': 0,
            })])

            assert (filtered.dropped, filtered.nested_dropped) == (0, 0), (
                f'nested spelling {spelling!r} must be treated as own-project'
            )
            assert [c['id'] for c in filtered.kept[0]['grouped']['amendments']] == ['a1']

    def test_a_foreign_nested_child_is_dropped_whatever_its_casing(self):
        """The same canonicalisation must not let a FOREIGN child through either."""
        filtered = _filter([_grouped_parent(grouped={
            'amendments': [
                {'id': 'a1', 'digest': 'FOREIGN AMENDMENT BODY', 'created_at': None,
                 'kind': 'amendment', 'metadata': {'src_project': '  REIFY  '}},
            ],
            'amendment_count': 1,
            'sighting_count': 0,
        })])

        assert (filtered.dropped, filtered.nested_dropped) == (0, 1)
        assert 'FOREIGN AMENDMENT BODY' not in _all_text(filtered.kept)

    def test_a_dropped_parent_is_counted_once_not_once_per_child(self):
        """A foreign parent takes its whole subtree with it — for exactly 1 drop.

        ``dropped`` renders into the operator-visible drop note, so descending
        into a dropped parent would show a human a wrong number.
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

        filtered = _filter([entry, _result('n1', 'Native fact.')])

        assert filtered.dropped == 1
        assert filtered.nested_dropped == 0
        assert _ids(filtered.kept) == ['n1']
        text = _all_text(filtered.kept)
        for body in ('FOREIGN CANONICAL.', 'FOREIGN CHILD ONE',
                     'FOREIGN CHILD TWO', 'FOREIGN CHILD THREE'):
            assert body not in text, f'{body!r} must go with its parent'


class TestGroupedDescentIsSurgicalAndFailsOpen:
    """The descent rewrites the child LISTS and nothing else, and never raises.

    Surgical: the server's ``grouped`` block carries EXACT counts; recomputing
    them after a drop would fabricate a number the store never returned.

    Fails open: a shape surprise must never blank the ``# Context`` block.
    """

    def test_the_server_counts_are_never_rewritten(self):
        filtered = _filter([_grouped_parent()])

        grouped = filtered.kept[0]['grouped']
        assert filtered.nested_dropped == 2
        assert len(grouped['amendments']) == 2, 'PRECONDITION: the list really did shrink'
        assert grouped['amendment_count'] == 3
        assert grouped['sighting_count'] == 1

    def test_sibling_grouped_keys_survive_verbatim(self):
        block = _grouped_block()
        block['truncated'] = True
        block['children_unavailable'] = True
        block['error_type'] = 'TimeoutError'

        filtered = _filter([_grouped_parent(grouped=block)])

        grouped = filtered.kept[0]['grouped']
        assert filtered.nested_dropped == 2
        assert grouped['truncated'] is True
        assert grouped['children_unavailable'] is True
        assert grouped['error_type'] == 'TimeoutError'

    def test_the_parents_own_fields_are_untouched(self):
        filtered = _filter([_grouped_parent()])

        parent = filtered.kept[0]
        assert parent['id'] == 'p1'
        assert parent['content'] == 'Native canonical.'
        assert parent['metadata'] == {'project_id': PROJECT}
        assert parent['relevance_score'] == 0.9

    def test_an_entry_with_no_grouped_key_is_a_no_op(self):
        entry = _result('n1', 'Native fact.', metadata={'project_id': PROJECT})

        filtered = _filter([entry])

        assert filtered == FilteredResults(kept=(entry,), dropped=0, nested_dropped=0)

    def test_a_non_dict_grouped_value_fails_open(self, caplog):
        entry = _result('p1', 'Native canonical.', metadata={'project_id': PROJECT})
        entry['grouped'] = 'not a dict at all'

        with caplog.at_level(logging.WARNING):
            filtered = _filter([entry])

        assert (filtered.dropped, filtered.nested_dropped) == (0, 0)
        assert _ids(filtered.kept) == ['p1']
        assert filtered.kept[0]['grouped'] == 'not a dict at all'
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_a_non_list_child_collection_fails_open(self, caplog):
        entry = _result('p1', 'Native canonical.', metadata={'project_id': PROJECT})
        entry['grouped'] = {'amendments': 'not a list', 'amendment_count': 1}

        with caplog.at_level(logging.WARNING):
            filtered = _filter([entry])

        assert (filtered.dropped, filtered.nested_dropped) == (0, 0)
        assert filtered.kept[0]['grouped']['amendments'] == 'not a list', (
            'A child collection of the wrong type is left EXACTLY as received'
        )
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_a_non_dict_nested_entry_is_kept_not_dropped(self):
        entry = _result('p1', 'Native canonical.', metadata={'project_id': PROJECT})
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

        filtered = _filter([entry])

        assert (filtered.dropped, filtered.nested_dropped) == (0, 1)
        amendments = filtered.kept[0]['grouped']['amendments']
        assert amendments[:2] == [None, 'a bare string']
        assert [a['id'] for a in amendments[2:]] == ['a2']

    def test_a_non_dict_nested_metadata_is_kept(self):
        entry = _result('p1', 'Native canonical.', metadata={'project_id': PROJECT})
        entry['grouped'] = {
            'amendments': [
                {'id': 'a1', 'digest': 'ODDLY SHAPED BODY', 'kind': 'amendment',
                 'metadata': 'not a dict'},
            ],
            'amendment_count': 1,
        }

        filtered = _filter([entry])

        assert (filtered.dropped, filtered.nested_dropped) == (0, 0)
        assert [a['id'] for a in filtered.kept[0]['grouped']['amendments']] == ['a1']


class TestRenderMemoryResults:
    def test_a_result_renders_as_a_tagged_bullet(self):
        entry = _result('1', 'Never run git stash.', source_store='mem0')
        entry['category'] = 'preferences_and_norms'
        entry['created_at'] = '2026-08-15T22:22:49+00:00'

        assert render_memory_results((entry,)) == (
            '- [preferences_and_norms · 2026-08-15 · mem0] Never run git stash.'
        )

    def test_grouped_children_render_indented_under_their_parents_store(self):
        assert render_memory_results((_grouped_parent(),)).splitlines() == [
            '- [uncategorized · undated · mem0] Native canonical.',
            '  - [amendment · undated · mem0] FOREIGN AMENDMENT BODY',
            '  - [amendment · undated · mem0] NATIVE AMENDMENT BODY',
            '  - [amendment · undated · mem0] UNTAGGED AMENDMENT BODY',
            '  - [sighting · undated · mem0] FOREIGN SIGHTING BODY',
        ]

    def test_entries_with_nothing_to_read_render_nothing(self):
        silent = _result('1', '   ', source_store='mem0')

        assert render_memory_results(()) == ''
        assert render_memory_results(('stray', silent)) == ''


class TestRenderEntityBlock:
    """Only an exactly-named node is admitted: the server's fuzzy fallback
    answers a miss with a neighbouring task, whose facts would read as ours."""

    def test_an_exactly_named_node_renders_its_summary_and_dated_edges(self):
        payload = {
            'nodes': [_node('Task 3609', 'Project-scopes the briefing context block.')],
            'edges': [
                _edge('Task 3609 landed as commit abc123.',
                      valid_at='2026-09-14T07:58:01.179808+00:00'),
                _edge('Task 3609 is related to task 3212.'),
            ],
        }

        assert render_entity_block(payload, 'Task 3609').splitlines() == [
            '**Task 3609** — Project-scopes the briefing context block.',
            '- Task 3609 landed as commit abc123. (2026-09-14)',
            '- Task 3609 is related to task 3212.',
        ]

    def test_a_fuzzy_neighbour_renders_nothing(self):
        payload = {
            'nodes': [_node('Task 3212', 'A DIFFERENT TASK ENTIRELY.')],
            'edges': [_edge('Task 3212 threads caller identity through search.')],
        }

        assert render_entity_block(payload, 'Task 3609') == ''

    def test_an_empty_node_list_renders_nothing(self):
        assert render_entity_block({'nodes': [], 'edges': []}, 'Task 3609') == ''

    @pytest.mark.parametrize(
        ('degraded', 'level'),
        [(True, logging.WARNING), (False, logging.DEBUG)],
    )
    def test_a_missing_node_is_logged_louder_when_the_reply_is_degraded(
        self, caplog, degraded, level,
    ):
        payload = {'nodes': [], 'edges': [], 'degraded': degraded}

        with caplog.at_level(logging.DEBUG):
            render_entity_block(payload, 'Task 3609')

        missing = [r for r in caplog.records if 'Task 3609' in r.getMessage()]
        assert [r.levelno for r in missing] == [level]


_SECTION = '## Conventions & Gotchas\n\n- [uncategorized · undated · mem0] A recalled fact.'
_TIMEOUT = MemoryQueryOutcome(failure=MemoryFailure.TIMEOUT)
_TRANSPORT = MemoryQueryOutcome(failure=MemoryFailure.TRANSPORT)
_HEALTHY = MemoryQueryOutcome(rendered='- a fact')


class TestRecallTally:
    """One dispatch's recall, frozen so the outage verdict is a pure function of it."""

    @pytest.mark.parametrize(
        ('tally', 'outage'),
        [
            (RecallTally(searches=(_TIMEOUT, _TRANSPORT)), True),
            (RecallTally(searches=(_TIMEOUT, MemoryQueryOutcome())), False),
            (RecallTally(searches=(), loop_failure=MemoryFailure.TRANSPORT), True),
            (
                RecallTally(
                    sections=(_SECTION,),
                    searches=(_HEALTHY,),
                    loop_failure=MemoryFailure.TRANSPORT,
                ),
                False,
            ),
            (RecallTally(searches=(MemoryQueryOutcome(), MemoryQueryOutcome())), False),
        ],
        ids=[
            'every-search-failed',
            'one-of-two-failed',
            'loop-broke-before-anything',
            'loop-broke-after-a-recalled-section',
            'nothing-failed-nothing-recalled',
        ],
    )
    def test_an_outage_means_nothing_worked(self, tally, outage):
        assert tally.is_outage is outage

    def test_reasons_are_deduplicated_in_order_then_the_loop_failure(self):
        tally = RecallTally(
            searches=(
                _TIMEOUT,
                MemoryQueryOutcome(failure=MemoryFailure.MALFORMED),
                _TIMEOUT,
                _HEALTHY,
            ),
            loop_failure=MemoryFailure.TRANSPORT,
        )

        assert tally.reasons == (
            MemoryFailure.TIMEOUT, MemoryFailure.MALFORMED, MemoryFailure.TRANSPORT,
        )

    @pytest.mark.parametrize(
        ('searches', 'note'),
        [
            (
                (MemoryQueryOutcome(dropped=2), MemoryQueryOutcome()),
                '2 memory result slot(s) across 2 queries were tagged to another '
                'project and filtered out',
            ),
            (
                (MemoryQueryOutcome(nested_dropped=4),),
                '4 nested memory record(s) across 1 query were tagged to another '
                'project and filtered out',
            ),
            (
                (
                    MemoryQueryOutcome(dropped=1, nested_dropped=2),
                    MemoryQueryOutcome(dropped=1, nested_dropped=2),
                ),
                '2 memory result slot(s) and 4 nested memory record(s) across 2 '
                'queries were tagged to another project and filtered out',
            ),
            ((MemoryQueryOutcome(), MemoryQueryOutcome()), ''),
        ],
        ids=['top-level-only', 'nested-only-one-query', 'both-named-apart', 'no-drops'],
    )
    def test_the_drop_note_names_each_quantity_apart(self, searches, note):
        assert RecallTally(searches=searches).drop_note == note


class TestRenderContextBlock:
    """The three shapes of the ``# Context`` block."""

    def test_nothing_recalled_and_no_outage_reads_as_an_empty_corpus(self):
        tally = RecallTally(
            notices=('_notice one_', '_notice two_'),
            searches=(MemoryQueryOutcome(dropped=1), MemoryQueryOutcome()),
        )

        assert render_context_block(tally, PROJECT) == (
            '# Context\n\n'
            + MEMORY_EMPTY_NOTICE
            + '\n\n_notice one_\n\n_notice two_'
            + f'\n\n_Note: {tally.drop_note}._'
        )

    def test_an_outage_leads_with_its_reasons_then_the_section_notices(self):
        tally = RecallTally(
            notices=('_section notice_',),
            searches=(_TIMEOUT, _TRANSPORT),
        )

        assert render_context_block(tally, PROJECT) == (
            '# Context\n\n'
            + MEMORY_OUTAGE_NOTICE.format(reasons='timeout, transport')
            + '\n\n_section notice_'
        )

    def test_recalled_sections_carry_the_caveat_and_the_loop_failure_line(self):
        tally = RecallTally(
            sections=(_SECTION, '## Task Context\n\n- another fact'),
            notices=('_section notice_',),
            searches=(MemoryQueryOutcome(dropped=1), _HEALTHY),
            loop_failure=MemoryFailure.TRANSPORT,
        )

        assert render_context_block(tally, PROJECT) == (
            '# Context\n\n'
            + MEMORY_CONTEXT_CAVEAT.format(project_id=PROJECT)
            + f'\n\n_In total, {tally.drop_note}._'
            + '\n\n'
            + _SECTION
            + '\n\n---\n\n## Task Context\n\n- another fact'
            + '\n\n---\n\n_section notice_\n'
            + '_Memory unavailable for the remaining queries — proceed with '
            'codebase exploration for anything not covered above._'
        )

    def test_recalled_sections_without_notices_or_drops_are_just_the_caveat_and_sections(self):
        tally = RecallTally(sections=(_SECTION,), searches=(_HEALTHY,))

        assert render_context_block(tally, PROJECT) == (
            '# Context\n\n' + MEMORY_CONTEXT_CAVEAT.format(project_id=PROJECT) + '\n\n' + _SECTION
        )
