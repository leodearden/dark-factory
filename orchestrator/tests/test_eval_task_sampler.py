"""Tests for orchestrator.evals.task_sampler — stratified eval fixture sampler.

Mirrors the hermetic-DI style of ``test_curator_corpus``: pure classifiers and
sampling are tested over inline synthetic candidates; the git glue (later
steps) is tested against a real temp git repo (the ``tmp_repo`` pattern from
``test_snapshots.py``). No live data / no network.
"""

from __future__ import annotations

from collections import Counter

import pytest
from _eval_sampler_fixtures import make_candidate, rich_corpus

from orchestrator.evals.task_sampler import (
    CompletedTaskCandidate,
    SampleResult,
    classify_kind,
    classify_path,
    repo_of,
    sample_stratified,
    stratify,
)


def _cand(**overrides) -> CompletedTaskCandidate:
    """Build a ``CompletedTaskCandidate`` with sane defaults, overriding fields.

    Only the fields a given classifier reads matter; the rest default to
    empty so a test can name just the axis it exercises.
    """
    base = dict(
        task_id='df_1',
        project='dark_factory',
        project_root='/home/leo/src/dark-factory',
        title='',
        description='',
        complexity=None,
        modules=[],
        pre_commit='',
        post_commit='',
        merge_sha='',
    )
    base.update(overrides)
    return CompletedTaskCandidate(**base)


# ---------------------------------------------------------------------------
# repo_of — the repo axis (df / reify)
# ---------------------------------------------------------------------------

class TestRepoOf:
    def test_dark_factory_project_id_maps_to_df(self) -> None:
        assert repo_of(_cand(project='dark_factory')) == 'df'

    def test_dark_factory_hyphenated_fixture_spelling_maps_to_df(self) -> None:
        # Existing fixtures spell it 'dark-factory' (project string) while
        # fused-memory uses 'dark_factory' (project_id) — both are df.
        assert repo_of(_cand(project='dark-factory')) == 'df'

    def test_reify_project_maps_to_reify(self) -> None:
        assert repo_of(_cand(project='reify')) == 'reify'

    def test_unknown_project_raises_loudly(self) -> None:
        # Loud-over-silent: repo is a hard stratification axis; an
        # unrecognised project is an error, not a silent default.
        with pytest.raises(ValueError):
            repo_of(_cand(project='some-other-project'))


# ---------------------------------------------------------------------------
# classify_kind — the kind axis (bugfix / feature / refactor)
# ---------------------------------------------------------------------------

class TestClassifyKind:
    def test_bug_prefix_is_bugfix(self) -> None:
        assert classify_kind(
            _cand(title="Bug: verify.py fails with 'source: not found'")
        ) == 'bugfix'

    def test_fix_keyword_is_bugfix(self) -> None:
        assert classify_kind(
            _cand(title='fix crash in scheduler dispatch')
        ) == 'bugfix'

    def test_add_keyword_is_feature(self) -> None:
        assert classify_kind(
            _cand(title='Add stratified task-set sampler')
        ) == 'feature'

    def test_implement_keyword_is_feature(self) -> None:
        assert classify_kind(
            _cand(title='implement warm-start protocol with LRU pool')
        ) == 'feature'

    def test_new_keyword_is_feature(self) -> None:
        assert classify_kind(
            _cand(title='new eval-sample subcommand')
        ) == 'feature'

    def test_refactor_keyword_is_refactor(self) -> None:
        assert classify_kind(
            _cand(title='Refactor: extract diff helper')
        ) == 'refactor'

    def test_rename_keyword_is_refactor(self) -> None:
        assert classify_kind(
            _cand(title='rename foo to bar across the module')
        ) == 'refactor'

    def test_keyword_found_in_description_body(self) -> None:
        # The heuristic scans title + description, not just the title.
        assert classify_kind(
            _cand(title='Scheduler work',
                  description='refactor the retry loop by extracting a helper')
        ) == 'refactor'

    def test_documented_default_is_feature_when_no_keyword(self) -> None:
        assert classify_kind(
            _cand(title='Tweak scheduler tick cadence',
                  description='adjust the cadence of the periodic sweep')
        ) == 'feature'

    def test_precedence_bugfix_beats_refactor(self) -> None:
        # Documented precedence: bugfix > refactor > feature. 'Fix bug' is the
        # primary intent even though 'extract' also appears.
        assert classify_kind(
            _cand(title='Fix bug: extract method returns the wrong value')
        ) == 'bugfix'


# ---------------------------------------------------------------------------
# classify_path — the path axis (simple / full), mirroring production routing
# ---------------------------------------------------------------------------

class TestClassifyPath:
    def test_non_simple_complexity_is_full(self) -> None:
        assert classify_path(_cand(complexity='high')) == 'full'

    def test_none_complexity_is_full(self) -> None:
        assert classify_path(_cand(complexity=None)) == 'full'

    def test_small_complexity_is_full(self) -> None:
        # Only the literal 'simple' triggers the simple path — the legacy
        # fixtures' 'small'/'high' values are full-path.
        assert classify_path(_cand(complexity='small')) == 'full'

    def test_simple_with_no_blocker_is_simple(self) -> None:
        assert classify_path(
            _cand(complexity='simple',
                  title='Fix typo in docstring',
                  description='fix a spelling error in the module docstring')
        ) == 'simple'

    def test_simple_complexity_case_insensitive(self) -> None:
        # Mirrors production is_declared_simple_task: strip + lower.
        assert classify_path(
            _cand(complexity='  Simple ',
                  title='doc tweak', description='reword a comment')
        ) == 'simple'

    def test_simple_but_migration_blocker_is_full(self) -> None:
        assert classify_path(
            _cand(complexity='simple',
                  description='requires a schema migration to backfill rows')
        ) == 'full'

    def test_simple_but_integration_test_blocker_is_full(self) -> None:
        assert classify_path(
            _cand(complexity='simple',
                  description='add an integration test covering the merge lane')
        ) == 'full'

    def test_simple_but_architecture_blocker_is_full(self) -> None:
        assert classify_path(
            _cand(complexity='simple',
                  description='rework the architecture of the dispatch layer')
        ) == 'full'


# ---------------------------------------------------------------------------
# stratify — bucket candidates into (repo, kind, path) cells
# ---------------------------------------------------------------------------

class TestStratify:
    def test_buckets_keyed_by_three_tuple(self) -> None:
        cells = stratify(rich_corpus())
        assert all(isinstance(k, tuple) and len(k) == 3 for k in cells)

    def test_over_supplied_and_thin_cell_counts(self) -> None:
        cells = stratify(rich_corpus())
        assert len(cells[('df', 'feature', 'full')]) == 10
        assert len(cells[('reify', 'bugfix', 'full')]) == 2

    def test_partition_preserves_every_candidate(self) -> None:
        cands = rich_corpus()
        cells = stratify(cands)
        assert sum(len(v) for v in cells.values()) == len(cands)

    def test_empty_cells_are_absent_not_zero_valued(self) -> None:
        # A cell with no candidates simply does not appear as a key.
        cells = stratify(rich_corpus())
        assert ('reify', 'refactor', 'simple') not in cells


# ---------------------------------------------------------------------------
# sample_stratified — deterministic, band-clamped, round-robin, loud-on-thin
# ---------------------------------------------------------------------------

class TestSampleStratified:
    def test_rich_corpus_selection_within_band(self) -> None:
        result = sample_stratified(rich_corpus(), target_low=10, target_high=14, seed=1)
        assert isinstance(result, SampleResult)
        assert 10 <= len(result.selected) <= 14

    def test_thin_corpus_takes_all_and_notes_shortfall(self) -> None:
        # Fewer than target_low: no silent truncation — take ALL and record
        # the shortfall loudly in notes.
        cands = [
            make_candidate(f't{i}', repo='df', kind='bugfix', path='full')
            for i in range(5)
        ]
        result = sample_stratified(cands, target_low=10, target_high=14, seed=1)
        assert len(result.selected) == 5
        assert {c.task_id for c in result.selected} == {c.task_id for c in cands}
        assert any('shortfall' in n.lower() for n in result.notes), result.notes

    def test_thin_corpus_notes_empty_cells(self) -> None:
        # The single-cell thin corpus leaves 11 of the 12 axis cells empty;
        # each empty cell is surfaced (loud-over-silent thin coverage).
        cands = [
            make_candidate(f't{i}', repo='df', kind='bugfix', path='full')
            for i in range(5)
        ]
        result = sample_stratified(cands, seed=1)
        empty_notes = [n for n in result.notes if 'empty cell' in n.lower()]
        assert len(empty_notes) == 11, result.notes

    def test_reproducible_for_a_fixed_seed(self) -> None:
        cands = rich_corpus()
        r1 = sample_stratified(cands, seed=7)
        r2 = sample_stratified(cands, seed=7)
        assert [c.task_id for c in r1.selected] == [c.task_id for c in r2.selected]

    def test_different_seed_still_yields_a_valid_in_band_selection(self) -> None:
        cands = rich_corpus()
        r1 = sample_stratified(cands, target_low=10, target_high=14, seed=1)
        r2 = sample_stratified(cands, target_low=10, target_high=14, seed=2)
        assert 10 <= len(r1.selected) <= 14
        assert 10 <= len(r2.selected) <= 14

    def test_round_robin_fairness_no_cell_dominates(self) -> None:
        # ('df','feature','full') is over-supplied (10 of 22) but must NOT
        # dominate a bounded selection when six other cells have supply.
        result = sample_stratified(rich_corpus(), target_low=10, target_high=14, seed=3)
        cell_counts = Counter(
            (repo_of(c), classify_kind(c), classify_path(c)) for c in result.selected
        )
        assert cell_counts[('df', 'feature', 'full')] <= 4
        assert len(cell_counts) >= 5

    def test_cell_counts_match_selected(self) -> None:
        result = sample_stratified(rich_corpus(), seed=5)
        assert sum(result.cell_counts.values()) == len(result.selected)
