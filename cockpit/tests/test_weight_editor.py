"""Tests for cockpit.panes.weight_editor — pure weight-edit merge + known-projects
helpers (Fleet Cockpit C9b, PRD §9).

Pure, deterministic unit tests only -- no Textual import, no pilot. The
WeightEditorScreen(ModalScreen) widget itself is covered by test_app.py's
pilot tests instead, mirroring test_decision_queue.py/test_spawn_bar.py's own
split between fast pure-helper tests here and slower app-level pilot tests.
"""

from __future__ import annotations

from dataclasses import replace

from orchestrator import session_registry as sr


def _make_session(**overrides):
    """Mirrors test_app.py/test_decision_queue.py's _make_record convention."""
    fields: dict = {
        'session_slug': 'unblock-df-2085-4242',
        'status': sr.Status.RUNNING,
        'title': 'unblock:df#2085 slug',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'start_ts': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


def _make_decision(**overrides):
    """Mirrors test_decision_queue.py's _make_decision convention."""
    fields: dict = {
        'id': 'dec-1',
        'project': 'df',
        'text': 'Which port?',
        'filed_at': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.DecisionRecord(**fields)


class TestMergeWeightEdits:
    def test_updates_only_category_and_project_weights(self):
        from cockpit.panes.weight_editor import merge_weight_edits
        from cockpit.priority import Priorities

        base = Priorities.default()

        result = merge_weight_edits(
            base,
            category_edits={'bug': '4.0'},
            project_edits={'df': '9.0', 'newproj': '2.5'},
        )

        assert result.category_weights == {**base.category_weights, 'bug': 4.0}
        assert result.project_weights == {'df': 9.0, 'newproj': 2.5}
        assert result.severity_weights == base.severity_weights
        assert result.defaults == base.defaults
        assert result.age_curve == base.age_curve
        assert result.manual_boost == base.manual_boost

    def test_unparseable_or_empty_edits_are_skipped_and_never_raise(self):
        from cockpit.panes.weight_editor import merge_weight_edits
        from cockpit.priority import Priorities

        base = replace(Priorities.default(), project_weights={'df': 3.0})

        unparseable = merge_weight_edits(base, category_edits={}, project_edits={'df': 'xx'})
        empty = merge_weight_edits(base, category_edits={}, project_edits={'df': ''})

        assert unparseable.project_weights == {'df': 3.0}
        assert empty.project_weights == {'df': 3.0}

    def test_non_finite_edits_are_skipped_and_never_raise(self):
        """float() itself accepts 'inf'/'-inf'/'nan' -- a non-finite weight
        would poison priority.score()'s urgency into inf/nan and corrupt the
        DecisionQueue's sort order, so both loops must reject these exactly
        like any other unparseable input (not just skip ValueError/TypeError).
        """
        from cockpit.panes.weight_editor import merge_weight_edits
        from cockpit.priority import Priorities

        base = replace(
            Priorities.default(),
            category_weights={'bug': 1.0},
            project_weights={'df': 3.0},
        )

        for bad in ('inf', '-inf', 'nan', 'Infinity'):
            result = merge_weight_edits(base, category_edits={'bug': bad}, project_edits={'df': bad})
            assert result.category_weights == {'bug': 1.0}, bad
            assert result.project_weights == {'df': 3.0}, bad


class TestKnownProjects:
    def test_sorted_distinct_union_of_records_decisions_and_existing(self):
        from cockpit.panes.weight_editor import known_projects

        records = [_make_session(project='df')]
        decisions = [_make_decision(project='other')]
        existing = {'zeta': 1.0}

        result = known_projects(records, decisions, existing)

        assert result == ['df', 'other', 'zeta']

    def test_dedupes_across_records_decisions_and_existing(self):
        from cockpit.panes.weight_editor import known_projects

        records = [_make_session(project='df')]
        decisions = [_make_decision(project='df')]
        existing = {'df': 1.0}

        result = known_projects(records, decisions, existing)

        assert result == ['df']

    def test_empty_records_and_decisions_returns_existing_alone(self):
        from cockpit.panes.weight_editor import known_projects

        result = known_projects([], [], {'df': 1.0})

        assert result == ['df']

    def test_defaults_existing_to_empty(self):
        from cockpit.panes.weight_editor import known_projects

        assert known_projects([], []) == []

    def test_empty_project_names_excluded_and_never_raise(self):
        from cockpit.panes.weight_editor import known_projects

        records = [_make_session(project='')]
        decisions = [_make_decision(project='')]

        result = known_projects(records, decisions, {'': 1.0})

        assert result == []


class TestKnownProjectsOffersCanonicalNamesOnly:
    """The picker offers ONE name per project (task 3812).

    known_projects is the operator-facing half of the same invariant
    registry_reader and load_priorities enforce on the scorer's side: the
    name offered here becomes a project_weights KEY, and priority.score()
    looks that key up with an already-canonical item.project. Offering a raw
    spelling would therefore let an operator set a weight that silently never
    applies -- the exact new silent failure task 3807's design decision named
    as the reason it stopped short of this work.

    The fold is applied to all three input channels locally and is
    deliberately redundant with the upstream folds: normalize_project_token
    is idempotent, so it costs nothing, and it makes the guarantee LOCAL to
    the picker rather than contingent on three remote callers (CockpitApp's
    scanner is a DI seam -- a fake or future scanner can hand this function
    raw records).
    """

    def test_one_name_from_three_spellings_across_all_three_channels(self):
        from cockpit.panes.weight_editor import known_projects

        result = known_projects(
            [_make_session(project='dark-factory')],
            [_make_decision(project='df')],
            {'DARK_FACTORY': 1.0},
        )

        assert result == ['dark_factory']

    def test_distinct_tokens_stay_distinct(self):
        """Collapse guard: the fold merges SPELLINGS, never distinct
        projects. A cwd-basename token and a synthetic one each stay their
        own offered entry."""
        from cockpit.panes.weight_editor import known_projects

        records = [
            _make_session(session_slug='s-1', project='orchestrator'),
            _make_session(session_slug='s-2', project='fm-neutral-classifier-cwd-_3yp2s4h'),
            _make_session(session_slug='s-3', project='dark-factory'),
        ]

        result = known_projects(records, [], {})

        assert result == [
            'dark_factory',
            'fm_neutral_classifier_cwd_3yp2s4h',
            'orchestrator',
        ]

    def test_result_stays_sorted_and_deduped_after_folding(self):
        from cockpit.panes.weight_editor import known_projects

        records = [
            _make_session(session_slug='s-1', project='ZETA'),
            _make_session(session_slug='s-2', project='zeta'),
            _make_session(session_slug='s-3', project='alpha-one'),
        ]

        result = known_projects(records, [_make_decision(project='alpha_one')], {'zeta': 1.0})

        assert result == ['alpha_one', 'zeta']

    def test_whitespace_only_name_folds_to_empty_and_is_excluded(self):
        """Fold FIRST, then drop empties: a whitespace-only name is truthy as
        a raw string but folds to the '' unset sentinel, and must be excluded
        rather than offered as a nameless picker row."""
        from cockpit.panes.weight_editor import known_projects

        result = known_projects(
            [_make_session(project='   ')], [_make_decision(project='')], {'  ': 1.0}
        )

        assert result == []
