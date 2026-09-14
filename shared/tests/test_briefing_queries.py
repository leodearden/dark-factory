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

        assert derive_area_terms(scope) == (
            'shared', 'briefing', 'queries', 'orchestrator', 'agents',
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
