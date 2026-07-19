"""Tests for orchestrator.evals.task_sampler — stratified eval fixture sampler.

Mirrors the hermetic-DI style of ``test_curator_corpus``: pure classifiers and
sampling are tested over inline synthetic candidates; the git glue (later
steps) is tested against a real temp git repo (the ``tmp_repo`` pattern from
``test_snapshots.py``). No live data / no network.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest
from _eval_sampler_fixtures import make_candidate, rich_corpus

from orchestrator.evals.task_sampler import (
    CompletedTaskCandidate,
    ReferenceCapture,
    SampleResult,
    build_fixture_record,
    capture_reference,
    classify_kind,
    classify_path,
    default_verify_commands,
    discover_completed_tasks,
    materialize_reference_diff,
    pin_eval_branch,
    repo_of,
    resolve_task_commits_from_merge,
    sample_stratified,
    stratify,
)


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command in *cwd* and return stripped stdout."""
    return subprocess.run(
        ['git', *args], cwd=str(cwd), check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def merge_repo(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    """A temp repo with a real ``Merge task/777 into main`` no-ff merge commit.

    Returns ``(repo, task_id, merge_sha, pre_sha, post_sha)`` where
    ``post_sha == merge_sha`` (the landed state) and ``pre_sha == merge_sha^1``
    (prior main / eval baseline). The task branch adds 2 lines to a.txt and a
    new b.txt (+1), so the net ``pre..post`` diff is 2 files / 3 insertions /
    0 deletions.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(['init', '-q', '-b', 'main'], repo)
    _git(['config', 'user.email', 'test@example.com'], repo)
    _git(['config', 'user.name', 'Test User'], repo)
    _git(['config', 'commit.gpgsign', 'false'], repo)

    (repo / 'a.txt').write_text('base\n')
    _git(['add', '.'], repo)
    _git(['commit', '-q', '-m', 'base'], repo)

    _git(['checkout', '-q', '-b', 'task/777'], repo)
    (repo / 'a.txt').write_text('base\nline2\nline3\n')
    (repo / 'b.txt').write_text('new file\n')
    _git(['add', '.'], repo)
    _git(['commit', '-q', '-m', 'work on 777'], repo)

    _git(['checkout', '-q', 'main'], repo)
    _git(['merge', '--no-ff', 'task/777', '-m', 'Merge task/777 into main'], repo)
    merge_sha = _git(['rev-parse', 'HEAD'], repo)
    pre = _git(['rev-parse', 'HEAD^1'], repo)
    return repo, '777', merge_sha, pre, merge_sha


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


# ---------------------------------------------------------------------------
# default_verify_commands — per-repo standard gate command set
# ---------------------------------------------------------------------------

class TestDefaultVerifyCommands:
    def test_df_uses_pytest_ruff_pyright(self) -> None:
        cmds = default_verify_commands('df')
        assert 'pytest' in cmds['test']
        assert 'ruff' in cmds['lint']
        assert 'pyright' in cmds['typecheck']

    def test_reify_uses_cargo_test_and_clippy(self) -> None:
        cmds = default_verify_commands('reify')
        assert 'cargo test' in cmds['test']
        assert 'clippy' in cmds['lint']

    def test_unknown_repo_raises_loudly(self) -> None:
        with pytest.raises(ValueError):
            default_verify_commands('some-other-repo')


# ---------------------------------------------------------------------------
# build_fixture_record — canonical fixture schema + new cohort/provenance/etc
# ---------------------------------------------------------------------------

def _ref() -> ReferenceCapture:
    return ReferenceCapture(
        post_task_commit='postSHA', files=2, insertions=10, deletions=3
    )


class TestBuildFixtureRecord:
    def _df_candidate(self) -> CompletedTaskCandidate:
        return make_candidate(
            '12', repo='df', kind='bugfix', path='full',
            title='Bug: verify.py fails on dash',
            description='fix the shell executable',
            modules=['orchestrator/src/orchestrator'],
            pre_commit='preSHA', post_commit='postSHA', merge_sha='mergeSHA',
        )

    def test_canonical_schema_fields_present(self) -> None:
        rec = build_fixture_record(
            self._df_candidate(), _ref(), default_verify_commands('df'),
            plan={'task_id': 'x', 'steps': []},
            cohort='revival-zeta', sampled_at='2026-07-19T00:00:00+00:00', seed=42,
        )
        assert rec['id'] == 'df_task_12'
        assert rec['name'] == 'Bug: verify.py fails on dash'
        assert rec['project'] == 'dark-factory'
        assert rec['project_root'] == '/home/leo/src/dark-factory'
        assert rec['pre_task_commit'] == 'preSHA'
        assert rec['post_task_commit'] == 'postSHA'
        assert rec['task_definition'] == {
            'title': 'Bug: verify.py fails on dash',
            'description': 'fix the shell executable',
        }
        assert rec['modules'] == ['orchestrator/src/orchestrator']
        assert rec['complexity'] == 'high'
        assert 'pytest' in rec['verify_commands']['test']

    def test_new_cohort_provenance_reference_and_verify_outcome(self) -> None:
        rec = build_fixture_record(
            self._df_candidate(), _ref(), default_verify_commands('df'),
            plan={'task_id': 'x', 'steps': []},
            cohort='revival-zeta', sampled_at='2026-07-19T00:00:00+00:00', seed=42,
        )
        assert rec['cohort'] == 'revival-zeta'
        assert rec['provenance']['merge_sha'] == 'mergeSHA'
        assert rec['provenance']['sampled_at'] == '2026-07-19T00:00:00+00:00'
        assert rec['provenance']['seed'] == 42
        assert rec['reference'] == {
            'post_task_commit': 'postSHA',
            'diff_stat': {'files': 2, 'insertions': 10, 'deletions': 3},
        }
        assert rec['verify_outcome']['source'] == 'landed'
        assert rec['verify_outcome']['passed'] is True
        assert rec['verify_outcome']['commands'] == default_verify_commands('df')

    def test_plan_present_embedded_verbatim(self) -> None:
        plan = {'task_id': 'x', 'steps': [{'id': 'step-1'}]}
        rec = build_fixture_record(
            self._df_candidate(), _ref(), default_verify_commands('df'),
            plan=plan, cohort='c', sampled_at='t',
        )
        assert rec['plan'] == plan
        assert rec['provenance']['plan_source'] == 'embedded'

    def test_plan_absent_records_unavailable(self) -> None:
        cand = make_candidate('99', repo='reify', kind='feature', path='full')
        rec = build_fixture_record(
            cand, _ref(), default_verify_commands('reify'),
            plan=None, cohort='c', sampled_at='t',
        )
        assert rec['plan'] is None
        assert rec['provenance']['plan_source'] == 'unavailable'
        assert rec['id'] == 'reify_task_99'
        assert rec['project'] == 'reify'

    def test_record_is_json_serializable(self) -> None:
        # No dataclass / non-JSON types must leak into the emitted fixture.
        rec = build_fixture_record(
            self._df_candidate(), _ref(), default_verify_commands('df'),
            plan=None, cohort='c', sampled_at='t',
        )
        round_tripped = json.loads(json.dumps(rec))
        assert round_tripped['id'] == 'df_task_12'


# ---------------------------------------------------------------------------
# Git glue — merge-log discovery, commit resolution, reference capture, pin
# ---------------------------------------------------------------------------

class TestGitGlue:
    def test_resolve_task_commits_from_merge(
        self, merge_repo: tuple[Path, str, str, str, str]
    ) -> None:
        repo, _tid, merge_sha, pre, post = merge_repo
        got_pre, got_post = asyncio.run(
            resolve_task_commits_from_merge(repo, merge_sha)
        )
        assert got_pre == pre
        assert got_post == post

    def test_capture_reference_diff_stat(
        self, merge_repo: tuple[Path, str, str, str, str]
    ) -> None:
        repo, _tid, _msha, pre, post = merge_repo
        ref = asyncio.run(capture_reference(repo, pre, post))
        assert ref.post_task_commit == post
        assert ref.files == 2
        assert ref.insertions == 3
        assert ref.deletions == 0

    def test_materialize_reference_diff_equals_git_diff(
        self, merge_repo: tuple[Path, str, str, str, str]
    ) -> None:
        repo, _tid, _msha, pre, post = merge_repo
        got = asyncio.run(materialize_reference_diff(repo, pre, post))
        expected = _git(['diff', f'{pre}..{post}'], repo)
        assert got == expected
        # The net landed change is non-empty (the judge grades a real diff).
        assert 'b.txt' in got

    def test_pin_eval_branch_creates_and_is_idempotent(
        self, merge_repo: tuple[Path, str, str, str, str]
    ) -> None:
        repo, _tid, _msha, _pre, post = merge_repo
        branch = asyncio.run(pin_eval_branch(repo, 'df_task_777', post))
        assert branch == 'evals/df_task_777'
        assert _git(['rev-parse', 'evals/df_task_777'], repo) == post
        # Second call leaves the same ref and does not raise.
        branch2 = asyncio.run(pin_eval_branch(repo, 'df_task_777', post))
        assert branch2 == 'evals/df_task_777'
        assert _git(['rev-parse', 'evals/df_task_777'], repo) == post

    def test_discover_completed_tasks_parses_merge_lines(
        self, merge_repo: tuple[Path, str, str, str, str]
    ) -> None:
        repo, task_id, merge_sha, pre, post = merge_repo
        cands = asyncio.run(
            discover_completed_tasks(repo, since=None, project='dark_factory')
        )
        assert len(cands) == 1
        c = cands[0]
        assert c.task_id == task_id
        assert c.merge_sha == merge_sha
        assert c.post_commit == post
        assert c.pre_commit == pre
        assert c.project == 'dark_factory'
