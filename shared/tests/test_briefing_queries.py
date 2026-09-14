"""Tests for shared.briefing_queries — the single home for briefing memory queries.

Task 3659 (PRD lane β, D9/INV-5). The briefing assembler used to spell its
memory queries inline as four hardcoded strings; this module now owns WHAT is
asked, so that the orchestrator's renderer and PRD-γ's registry pinning test
read the same phrasings instead of two hand-copied copies that can drift.

This file covers the scope object and the area-term ladder (D2/D3: path
components -> task title -> repo-generic). The query-spec table itself is
covered further down the same module's surface by ``TestQuerySpecs``.
"""
from __future__ import annotations

import dataclasses

import pytest

from shared.briefing_queries import BriefingScope, derive_area_terms


class TestBriefingScopeFromTask:
    """``from_task`` reads the queue-time task record."""

    def test_reads_id_title_and_metadata_files(self):
        scope = BriefingScope.from_task({
            'id': '3659',
            'title': 'Briefing memory rescope',
            'metadata': {'files': [
                'shared/src/shared/briefing_queries.py',
                'orchestrator/src/orchestrator/agents/briefing.py',
            ]},
        })

        assert scope.task_id == '3659'
        assert scope.title == 'Briefing memory rescope'
        assert scope.files == (
            'shared/src/shared/briefing_queries.py',
            'orchestrator/src/orchestrator/agents/briefing.py',
        )

    def test_non_string_id_is_normalised_to_str(self):
        """Task ids reach the assembler as both ``'3659'`` and ``3659``; the
        caller-identity string built from one must not vary by wire type."""
        assert BriefingScope.from_task({'id': 3659}).task_id == '3659'

    def test_missing_keys_yield_an_empty_scope_without_raising(self):
        scope = BriefingScope.from_task({})

        assert scope.task_id is None
        assert scope.title == ''
        assert scope.files == ()

    def test_none_valued_keys_are_tolerated(self):
        scope = BriefingScope.from_task({'id': None, 'title': None, 'metadata': None})

        assert scope.task_id is None
        assert scope.title == ''
        assert scope.files == ()

    def test_non_dict_metadata_is_tolerated(self):
        assert BriefingScope.from_task({'metadata': 'not-a-dict'}).files == ()

    def test_a_bare_string_files_value_yields_no_files(self):
        """A string is iterable, so a permissive ``tuple(...)`` would shred it
        into single characters and ask memory about the letter ``s``."""
        assert BriefingScope.from_task({'metadata': {'files': 'a/b.py'}}).files == ()

    def test_non_string_file_entries_are_dropped(self):
        scope = BriefingScope.from_task({'metadata': {'files': ['a/b.py', None, 7]}})

        assert scope.files == ('a/b.py',)


class TestBriefingScopeFromPlan:
    """``from_plan`` reads the frozen plan, whose keys differ from a task's."""

    def test_reads_task_id_title_and_files(self):
        scope = BriefingScope.from_plan({
            'task_id': '3659',
            'title': 'Briefing memory rescope',
            'files': ['orchestrator/src/orchestrator/agents/briefing.py'],
        })

        assert scope.task_id == '3659'
        assert scope.title == 'Briefing memory rescope'
        assert scope.files == ('orchestrator/src/orchestrator/agents/briefing.py',)

    def test_missing_keys_yield_an_empty_scope_without_raising(self):
        scope = BriefingScope.from_plan({})

        assert scope.task_id is None
        assert scope.title == ''
        assert scope.files == ()

    def test_none_valued_keys_are_tolerated(self):
        scope = BriefingScope.from_plan({'task_id': None, 'title': None, 'files': None})

        assert scope == BriefingScope.from_plan({})


class TestBriefingScopeIsImmutable:
    def test_fields_cannot_be_reassigned(self):
        scope = BriefingScope.from_task({'id': '3659'})

        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.task_id = '3212'

    def test_files_are_a_tuple_not_a_list(self):
        scope = BriefingScope.from_task({'metadata': {'files': ['a/b.py']}})

        assert isinstance(scope.files, tuple)


class TestDeriveAreaTerms:
    """The D2/D3 ladder: ``metadata.files`` -> task title -> repo-generic."""

    def test_path_components_become_deduplicated_prose_terms(self):
        scope = BriefingScope(files=(
            'shared/src/shared/briefing_queries.py',
            'orchestrator/src/orchestrator/agents/briefing.py',
        ))

        # Interleaved, not path-by-path: each file contributes its first
        # word before any file contributes a second, so a term budget spent
        # on the earliest-listed package cannot hide the rest of the
        # footprint.
        assert derive_area_terms(scope) == (
            'shared', 'orchestrator', 'briefing', 'agents', 'queries',
        )

    def test_every_declared_file_is_represented_within_the_budget(self):
        from shared.briefing_queries import AREA_TERM_LIMIT

        scope = BriefingScope(files=(
            'shared/src/shared/briefing_queries.py',
            'shared/tests/test_briefing_queries.py',
            'shared/tests/silent_fallthrough_allowlist.py',
            'orchestrator/src/orchestrator/agents/briefing.py',
            'scripts/legibility/digest.py',
        ))

        terms = derive_area_terms(scope)

        assert len(terms) <= AREA_TERM_LIMIT
        for package in ('shared', 'orchestrator', 'scripts'):
            assert package in terms, (
                f'{package!r} is declared in the footprint but never reached '
                f'the query: {terms}'
            )

    def test_structural_path_noise_and_extensions_are_dropped(self):
        terms = derive_area_terms(BriefingScope(files=(
            'shared/src/shared/briefing_queries.py',
            'orchestrator/tests/_workflow_helpers.py',
        )))

        assert 'src' not in terms
        assert 'py' not in terms
        assert all('/' not in t and '.' not in t and '_' not in t for t in terms)
        assert 'workflow' in terms and 'helpers' in terms

    def test_terms_keep_first_seen_order_and_never_repeat(self):
        terms = derive_area_terms(BriefingScope(files=(
            'orchestrator/src/orchestrator/agents/briefing.py',
            'orchestrator/tests/test_briefing.py',
        )))

        assert terms[0] == 'orchestrator'
        assert len(terms) == len(set(terms))

    def test_title_is_the_fallback_when_there_are_no_files(self):
        scope = BriefingScope(title='Briefing memory rescope: shared query templates')

        assert derive_area_terms(scope) == (
            'briefing', 'memory', 'rescope', 'shared', 'query', 'templates',
        )

    def test_files_win_over_the_title_when_both_are_present(self):
        scope = BriefingScope(
            title='Some unrelated wording',
            files=('orchestrator/src/orchestrator/agents/briefing.py',),
        )

        assert derive_area_terms(scope) == ('orchestrator', 'agents', 'briefing')

    def test_neither_files_nor_title_is_the_repo_generic_last_resort(self):
        """No terms at all — the query table answers this with the
        repo-generic conventions spec, which needs none."""
        assert derive_area_terms(BriefingScope()) == ()

    def test_term_count_is_bounded(self):
        from shared.briefing_queries import AREA_TERM_LIMIT

        many = tuple(f'pkg{n}/src/pkg{n}/mod_{n}.py' for n in range(AREA_TERM_LIMIT + 5))

        assert len(derive_area_terms(BriefingScope(files=many))) == AREA_TERM_LIMIT

    def test_derivation_is_pure(self):
        scope = BriefingScope(files=('shared/src/shared/briefing_queries.py',))

        assert derive_area_terms(scope) == derive_area_terms(scope)


class TestQuerySpecs:
    """The table of queries the briefing fires (D1/D2/D3, PRD-γ's D9 slugs)."""

    def test_exactly_three_specs_with_the_registry_slugs(self):
        from shared.briefing_queries import QUERY_SPECS

        assert tuple(spec.slug for spec in QUERY_SPECS) == (
            'briefing-conventions-generic',
            'briefing-conventions-area',
            'briefing-task-semantic',
        )

    def test_slugs_are_unique(self):
        from shared.briefing_queries import QUERY_SPECS

        assert len({spec.slug for spec in QUERY_SPECS}) == len(QUERY_SPECS)

    def test_conventions_specs_are_store_and_category_scoped(self):
        from shared.briefing_queries import CONVENTIONS_AREA, CONVENTIONS_GENERIC

        for spec in (CONVENTIONS_GENERIC, CONVENTIONS_AREA):
            assert spec.stores == ('mem0',)
            assert spec.categories == ('preferences_and_norms', 'procedural_knowledge')

    def test_every_spec_asks_for_five_results(self):
        from shared.briefing_queries import QUERY_SPECS

        assert [spec.limit for spec in QUERY_SPECS] == [5, 5, 5]

    def test_the_retired_queries_appear_in_no_template(self):
        """D1 retires these two outright rather than rewording them."""
        from shared.briefing_queries import QUERY_SPECS

        rendered = ' '.join(spec.text for spec in QUERY_SPECS).lower()
        assert 'project overview architecture goals' not in rendered
        assert 'recent decisions and rationale' not in rendered

    def test_specs_are_immutable(self):
        from shared.briefing_queries import QUERY_SPECS, TASK_SEMANTIC

        with pytest.raises(dataclasses.FrozenInstanceError):
            TASK_SEMANTIC.limit = 50
        assert isinstance(QUERY_SPECS, tuple)
        for spec in QUERY_SPECS:
            assert isinstance(spec.stores, tuple)
            assert isinstance(spec.categories, tuple)


class TestQueriesForScope:
    """``queries_for`` — the ordered ``(spec, query text)`` pairs to fire."""

    def _scope(self) -> BriefingScope:
        return BriefingScope(
            task_id='3659',
            title='Briefing memory rescope',
            files=('orchestrator/src/orchestrator/agents/briefing.py',),
        )

    def test_a_task_scoped_dispatch_fires_the_area_and_semantic_specs(self):
        from shared.briefing_queries import queries_for

        assert [spec.slug for spec, _text in queries_for(self._scope())] == [
            'briefing-conventions-area',
            'briefing-task-semantic',
        ]

    def test_a_scope_with_nothing_to_go_on_fires_the_generic_spec_alone(self):
        from shared.briefing_queries import queries_for

        assert [spec.slug for spec, _text in queries_for(BriefingScope())] == [
            'briefing-conventions-generic',
        ]

    def test_a_title_only_scope_still_reaches_the_area_spec(self):
        from shared.briefing_queries import queries_for

        scope = BriefingScope(task_id='3659', title='Briefing memory rescope')

        assert [spec.slug for spec, _text in queries_for(scope)] == [
            'briefing-conventions-area',
            'briefing-task-semantic',
        ]

    def test_the_task_semantic_query_carries_the_title_and_the_area_terms(self):
        from shared.briefing_queries import queries_for

        text = dict((spec.slug, q) for spec, q in queries_for(self._scope()))['briefing-task-semantic']

        assert 'Briefing memory rescope' in text
        # Case-insensitive: a term the title already spells (here "Briefing")
        # is carried by the title's own spelling rather than repeated.
        for term in ('orchestrator', 'agents', 'briefing'):
            assert term in text.lower()

    def test_the_task_semantic_query_never_carries_the_bare_task_id(self):
        """The measured 0/5 failure mode: a bare task number embeds close to
        OTHER task numbers, so it is kept out of every query text."""
        from shared.briefing_queries import queries_for

        for _spec, text in queries_for(self._scope()):
            assert '3659' not in text

    def test_the_conventions_query_is_phrased_from_the_area_terms(self):
        from shared.briefing_queries import queries_for

        text = dict((spec.slug, q) for spec, q in queries_for(self._scope()))['briefing-conventions-area']

        assert 'conventions' in text
        assert 'orchestrator agents briefing' in text

    def test_a_title_only_scope_does_not_echo_its_title_twice(self):
        """With no files the area ladder falls back to the title, which must
        not read back into the semantic query as a doubled phrase."""
        from shared.briefing_queries import queries_for

        scope = BriefingScope(task_id='3659', title='Briefing memory rescope')
        text = dict((spec.slug, q) for spec, q in queries_for(scope))['briefing-task-semantic']

        assert text == 'Briefing memory rescope'

    def test_rendering_is_pure(self):
        from shared.briefing_queries import queries_for

        assert queries_for(self._scope()) == queries_for(self._scope())

    def test_no_unfilled_slot_survives_rendering(self):
        from shared.briefing_queries import queries_for

        for scope in (self._scope(), BriefingScope()):
            for _spec, text in queries_for(scope):
                assert '{' not in text and '}' not in text
                assert text == text.strip()
                assert '  ' not in text
