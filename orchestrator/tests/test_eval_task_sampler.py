"""Tests for orchestrator.evals.task_sampler — stratified eval fixture sampler.

Mirrors the hermetic-DI style of ``test_curator_corpus``: pure classifiers and
sampling are tested over inline synthetic candidates; the git glue (later
steps) is tested against a real temp git repo (the ``tmp_repo`` pattern from
``test_snapshots.py``). No live data / no network.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

import pytest
from _eval_sampler_fixtures import make_candidate, rich_corpus
from click.testing import CliRunner

from orchestrator.cli import main
from orchestrator.evals.task_sampler import (
    CompletedTaskCandidate,
    FixtureAuditReport,
    ReferenceCapture,
    SampleResult,
    StratificationCounts,
    audit_fixture_corpus,
    build_fixture_record,
    capture_reference,
    classify_kind,
    classify_path,
    default_verify_commands,
    discover_completed_tasks,
    enrich_candidates_from_task_db,
    format_stratification_table,
    git_ref_exists,
    materialize_reference_diff,
    pin_eval_branch,
    repo_of,
    resolve_task_commits_from_merge,
    sample_stratified,
    stratification_counts,
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
    base: dict[str, object] = dict(
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
    return CompletedTaskCandidate(**base)  # type: ignore[arg-type]


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


# ---------------------------------------------------------------------------
# Listing + audit surface — stratification counts, table, corpus audit
# ---------------------------------------------------------------------------

def _fixture(
    task_id: object,
    *,
    repo: str = 'df',
    kind: str = 'feature',
    path: str = 'full',
    cohort: str = 'revival-zeta',
    plan: dict | None = None,
    with_reference: bool = True,
    with_verify_outcome: bool = True,
) -> dict:
    """Build a canonical fixture dict (via ``build_fixture_record``) in a cell.

    Optionally drops the ``reference`` / ``verify_outcome`` blocks so the audit
    tests can exercise the per-fixture completeness failures.
    """
    cand = make_candidate(
        task_id, repo=repo, kind=kind, path=path,
        pre_commit=f'pre{task_id}', post_commit=f'post{task_id}',
        merge_sha=f'merge{task_id}',
    )
    rec = build_fixture_record(
        cand,
        ReferenceCapture(
            post_task_commit=f'post{task_id}', files=1, insertions=2, deletions=0,
        ),
        default_verify_commands(repo),
        plan=plan, cohort=cohort, sampled_at='t', seed=0,
    )
    if not with_reference:
        rec.pop('reference')
    if not with_verify_outcome:
        rec.pop('verify_outcome')
    return rec


# Six populated cells the healthy-corpus builder cycles through (matching the
# cells make_candidate can deterministically synthesise).
_HEALTHY_CELLS = [
    ('df', 'bugfix', 'full'),
    ('df', 'feature', 'full'),
    ('df', 'refactor', 'full'),
    ('df', 'feature', 'simple'),
    ('reify', 'bugfix', 'full'),
    ('reify', 'feature', 'full'),
]


def _healthy_corpus(n: int = 12) -> list[dict]:
    """A band-plausible corpus of *n* complete fixtures spread across cells."""
    out: list[dict] = []
    for i in range(n):
        repo, kind, path = _HEALTHY_CELLS[i % len(_HEALTHY_CELLS)]
        out.append(_fixture(i, repo=repo, kind=kind, path=path, plan={'steps': []}))
    return out


class TestStratificationCounts:
    def test_per_cell_counts_and_total(self) -> None:
        fixtures = [
            _fixture('1', repo='df', kind='bugfix', path='full'),
            _fixture('2', repo='df', kind='bugfix', path='full'),
            _fixture('3', repo='reify', kind='feature', path='full'),
        ]
        counts = stratification_counts(fixtures)
        assert isinstance(counts, StratificationCounts)
        assert counts.total == 3
        assert counts.cells[('df', 'bugfix', 'full')] == 2
        assert counts.cells[('reify', 'feature', 'full')] == 1

    def test_plan_captured_counts_only_fixtures_with_a_plan(self) -> None:
        fixtures = [
            _fixture('1', plan={'steps': []}),
            _fixture('2', plan=None),
            _fixture('3', plan={'steps': [{'id': 'step-1'}]}),
        ]
        counts = stratification_counts(fixtures)
        assert counts.total == 3
        assert counts.plan_captured == 2

    def test_cohort_filter_scopes_counts(self) -> None:
        fixtures = [
            _fixture('1', repo='df', kind='bugfix', path='full', cohort='revival-zeta'),
            _fixture('2', repo='df', kind='bugfix', path='full', cohort='legacy-april'),
            _fixture('3', repo='reify', kind='feature', path='full', cohort='revival-zeta'),
        ]
        counts = stratification_counts(fixtures, cohort='revival-zeta')
        assert counts.total == 2
        assert counts.cohort == 'revival-zeta'
        assert counts.cells[('df', 'bugfix', 'full')] == 1
        assert counts.cells[('reify', 'feature', 'full')] == 1

    def test_cohort_none_counts_all_cohorts(self) -> None:
        fixtures = [
            _fixture('1', cohort='revival-zeta'),
            _fixture('2', cohort='legacy-april'),
        ]
        counts = stratification_counts(fixtures, cohort=None)
        assert counts.total == 2
        assert counts.cohort is None


class TestFormatStratificationTable:
    def test_table_is_a_stable_string(self) -> None:
        fixtures = [
            _fixture('1', repo='df', kind='bugfix', path='full'),
            _fixture('2', repo='reify', kind='feature', path='simple'),
        ]
        counts = stratification_counts(fixtures)
        table = format_stratification_table(counts)
        assert isinstance(table, str)
        # Deterministic render: same counts -> byte-identical table.
        assert format_stratification_table(counts) == table

    def test_table_reports_cells_totals_and_plan_capture(self) -> None:
        fixtures = [
            _fixture(str(i), repo='df', kind='bugfix', path='full', plan={'steps': []})
            for i in range(3)
        ]
        counts = stratification_counts(fixtures)
        table = format_stratification_table(counts)
        assert 'df' in table
        assert 'bugfix' in table
        assert 'reify' in table  # the full axis product renders (empty cells too)
        assert 'refactor' in table
        # Grand total and the plan-captured tally both surface in the table.
        assert '3' in table
        assert 'plan' in table.lower()


class TestAuditFixtureCorpus:
    def test_healthy_corpus_is_ok(self) -> None:
        fixtures = _healthy_corpus(12)
        report = audit_fixture_corpus(fixtures, ref_exists=lambda root, branch: True)
        assert isinstance(report, FixtureAuditReport)
        assert report.ok is True
        assert report.count == 12
        assert report.failures == []

    def test_band_violation_too_few(self) -> None:
        report = audit_fixture_corpus(_healthy_corpus(5), ref_exists=lambda *_: True)
        assert report.ok is False
        assert any('band' in f for f in report.failures), report.failures

    def test_band_violation_too_many(self) -> None:
        report = audit_fixture_corpus(_healthy_corpus(15), ref_exists=lambda *_: True)
        assert report.ok is False
        assert any('band' in f for f in report.failures), report.failures

    def test_missing_reference_block_flagged_by_id(self) -> None:
        fixtures = _healthy_corpus(12)
        fixtures[3].pop('reference')
        report = audit_fixture_corpus(fixtures, ref_exists=lambda *_: True)
        assert report.ok is False
        assert any(
            'reference' in f and fixtures[3]['id'] in f for f in report.failures
        ), report.failures

    def test_missing_verify_outcome_flagged_by_id(self) -> None:
        fixtures = _healthy_corpus(12)
        fixtures[2].pop('verify_outcome')
        report = audit_fixture_corpus(fixtures, ref_exists=lambda *_: True)
        assert report.ok is False
        assert any(
            'verify_outcome' in f and fixtures[2]['id'] in f for f in report.failures
        ), report.failures

    def test_missing_pinned_branch_flagged_per_fixture(self) -> None:
        fixtures = _healthy_corpus(12)
        report = audit_fixture_corpus(fixtures, ref_exists=lambda root, branch: False)
        assert report.ok is False
        # Every fixture is flagged as missing its pinned branch.
        assert sum('branch' in f for f in report.failures) == 12
        assert any(fixtures[0]['id'] in f for f in report.failures), report.failures

    def test_ref_exists_none_skips_branch_check(self) -> None:
        # Structural-only audit (no git checkout available): the branch-presence
        # check is skipped, so a band-and-completeness-clean corpus still passes.
        report = audit_fixture_corpus(_healthy_corpus(12), ref_exists=None)
        assert report.ok is True
        assert report.count == 12
        assert report.failures == []


# ---------------------------------------------------------------------------
# enrich_candidates_from_task_db — best-effort title/description/complexity fill
# ---------------------------------------------------------------------------

def _make_tasks_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """Build a minimal read-only taskmaster tasks.db with *rows*.

    Each row is ``(id, title, description, metadata_json)`` — the columns
    :func:`enrich_candidates_from_task_db` reads.
    """
    db = tmp_path / 'tasks.db'
    conn = sqlite3.connect(db)
    conn.execute(
        'CREATE TABLE tasks (id INTEGER, title TEXT, description TEXT, metadata TEXT)'
    )
    conn.executemany('INSERT INTO tasks VALUES (?, ?, ?, ?)', rows)
    conn.commit()
    conn.close()
    return db


class TestEnrichCandidatesFromTaskDb:
    def test_fills_title_description_complexity(self, tmp_path: Path) -> None:
        db = _make_tasks_db(tmp_path, [
            (12, 'Bug: fix the thing', 'a crash occurs', '{"complexity": "simple"}'),
        ])
        enriched = enrich_candidates_from_task_db(
            [_cand(task_id='12', title='', description='', complexity=None)], db,
        )
        assert enriched[0].title == 'Bug: fix the thing'
        assert enriched[0].description == 'a crash occurs'
        assert enriched[0].complexity == 'simple'

    def test_missing_db_returns_candidates_unchanged(self, tmp_path: Path) -> None:
        enriched = enrich_candidates_from_task_db(
            [_cand(task_id='12', title='orig')], tmp_path / 'nope.db',
        )
        assert enriched[0].title == 'orig'

    def test_unknown_id_left_as_stub(self, tmp_path: Path) -> None:
        db = _make_tasks_db(tmp_path, [(12, 'known', 'd', '{}')])
        enriched = enrich_candidates_from_task_db(
            [_cand(task_id='999', title='')], db,
        )
        assert enriched[0].title == ''

    def test_non_numeric_id_left_as_stub(self, tmp_path: Path) -> None:
        # Legacy fixtures may carry a non-numeric id; enrichment skips them
        # rather than raising on the int() lookup.
        db = _make_tasks_db(tmp_path, [(12, 'known', 'd', '{}')])
        enriched = enrich_candidates_from_task_db(
            [_cand(task_id='df_12', title='')], db,
        )
        assert enriched[0].title == ''


# ---------------------------------------------------------------------------
# CLI — eval-list-fixtures and eval-sample --dry-run (CliRunner, hermetic)
# ---------------------------------------------------------------------------

def _seed_fixture_dir(tasks_dir: Path, specs: list[tuple[str, dict]]) -> None:
    """Write one fixture JSON per spec into *tasks_dir*."""
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for task_id, kw in specs:
        rec = _fixture(task_id, **kw)
        (tasks_dir / f'{rec["id"]}.json').write_text(json.dumps(rec))


def _repo_with_merges(tmp_path: Path, task_ids: list[str]) -> Path:
    """A temp repo with one ``Merge task/<id> into main`` no-ff commit per id."""
    repo = tmp_path / 'df-repo'
    repo.mkdir()
    _git(['init', '-q', '-b', 'main'], repo)
    _git(['config', 'user.email', 'test@example.com'], repo)
    _git(['config', 'user.name', 'Test User'], repo)
    _git(['config', 'commit.gpgsign', 'false'], repo)
    (repo / 'base.txt').write_text('base\n')
    _git(['add', '.'], repo)
    _git(['commit', '-q', '-m', 'base'], repo)
    for tid in task_ids:
        _git(['checkout', '-q', '-b', f'task/{tid}'], repo)
        (repo / f'f{tid}.txt').write_text(f'work {tid}\n')
        _git(['add', '.'], repo)
        _git(['commit', '-q', '-m', f'work {tid}'], repo)
        _git(['checkout', '-q', 'main'], repo)
        _git(['merge', '--no-ff', f'task/{tid}', '-m', f'Merge task/{tid} into main'], repo)
    return repo


class TestEvalListFixturesCLI:
    def test_lists_table_and_exits_zero(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / 'tasks'
        _seed_fixture_dir(tasks_dir, [
            ('1', dict(repo='df', kind='bugfix', path='full')),
            ('2', dict(repo='reify', kind='feature', path='full')),
        ])
        result = CliRunner().invoke(
            main, ['eval-list-fixtures', '--tasks-dir', str(tasks_dir)],
        )
        assert result.exit_code == 0, result.output
        assert 'stratification' in result.output
        assert 'bugfix' in result.output

    def test_needs_no_orchestrator_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The listing reads only the tasks dir — no ORCH_CONFIG_PATH / --config.
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        tasks_dir = tmp_path / 'tasks'
        _seed_fixture_dir(tasks_dir, [('1', dict(repo='df', kind='bugfix', path='full'))])
        result = CliRunner().invoke(
            main, ['eval-list-fixtures', '--tasks-dir', str(tasks_dir)],
        )
        assert result.exit_code == 0, result.output

    def test_cohort_option_scopes_output(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / 'tasks'
        _seed_fixture_dir(tasks_dir, [
            ('1', dict(repo='df', kind='bugfix', path='full', cohort='revival-zeta')),
            ('2', dict(repo='df', kind='feature', path='full', cohort='legacy-april')),
        ])
        result = CliRunner().invoke(
            main,
            ['eval-list-fixtures', '--tasks-dir', str(tasks_dir), '--cohort', 'revival-zeta'],
        )
        assert result.exit_code == 0, result.output
        assert 'revival-zeta' in result.output
        assert 'TOTAL: 1' in result.output


class TestEvalSampleDryRunCLI:
    def test_dry_run_prints_selection_and_makes_no_changes(self, tmp_path: Path) -> None:
        repo = _repo_with_merges(tmp_path, ['101', '102', '103'])
        tasks_dir = tmp_path / 'tasks'
        tasks_dir.mkdir()
        result = CliRunner().invoke(main, [
            'eval-sample', '--dry-run',
            '--df-root', str(repo),
            '--reify-root', str(tmp_path / 'no-reify'),  # absent -> skipped loudly
            '--tasks-dir', str(tasks_dir),
            '--seed', '1',
        ])
        assert result.exit_code == 0, result.output
        assert 'dry-run' in result.output.lower()
        # Discovered all three completed-task merges.
        assert '3' in result.output
        # No fixture files written and no evals/* branches pinned.
        assert list(tasks_dir.glob('*.json')) == []
        assert _git(['branch', '--list', 'evals/*'], repo) == ''


# ---------------------------------------------------------------------------
# git_ref_exists — the production branch-presence predicate (real git)
# ---------------------------------------------------------------------------

def _bare_commit_repo(repo: Path) -> Path:
    """Init a temp git repo with a single base commit; return its path."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(['init', '-q', '-b', 'main'], repo)
    _git(['config', 'user.email', 'test@example.com'], repo)
    _git(['config', 'user.name', 'Test User'], repo)
    _git(['config', 'commit.gpgsign', 'false'], repo)
    (repo / 'base.txt').write_text('base\n')
    _git(['add', '.'], repo)
    _git(['commit', '-q', '-m', 'base'], repo)
    return repo


class TestGitRefExists:
    def test_present_ref_returns_true(self, tmp_path: Path) -> None:
        repo = _bare_commit_repo(tmp_path / 'repo')
        _git(['branch', 'evals/df_task_1', 'HEAD'], repo)
        assert git_ref_exists(str(repo), 'evals/df_task_1') is True

    def test_absent_ref_returns_false(self, tmp_path: Path) -> None:
        repo = _bare_commit_repo(tmp_path / 'repo')
        # No branch pinned -> the ref does not resolve.
        assert git_ref_exists(str(repo), 'evals/df_task_missing') is False

    def test_non_repo_returns_false_not_raises(self, tmp_path: Path) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()
        # A non-repo dir surfaces as False (a named audit failure), never a crash.
        assert git_ref_exists(str(plain), 'evals/df_task_1') is False


# ---------------------------------------------------------------------------
# CLI — eval-list-fixtures --audit (real-git glue + exit-code behaviour)
# ---------------------------------------------------------------------------

def _seed_zeta_corpus(
    tasks_dir: Path, repo_root: Path, n: int, *,
    cohort: str = 'revival-zeta', start: int = 0,
) -> list[str]:
    """Seed *n* complete fixtures rooted at *repo_root*; return their ids.

    Each fixture is a band-plausible ``_HEALTHY_CELLS`` cell with a plan (so its
    ``reference`` / ``verify_outcome`` blocks are present), but its
    ``project_root`` is rewritten to the temp *repo_root* so the CLI audit's
    real ``git_ref_exists`` probe hits that hermetic repo rather than the live
    checkout ``make_candidate`` hardcodes.

    *start* offsets the task-id space (and thus the fixture filenames) so a
    second cohort seeded into the same dir does NOT overwrite the first.
    """
    tasks_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    for i in range(start, start + n):
        repo, kind, path = _HEALTHY_CELLS[i % len(_HEALTHY_CELLS)]
        rec = _fixture(i, repo=repo, kind=kind, path=path,
                       cohort=cohort, plan={'steps': []})
        rec['project_root'] = str(repo_root)
        (tasks_dir / f'{rec["id"]}.json').write_text(json.dumps(rec))
        ids.append(rec['id'])
    return ids


class TestEvalListFixturesAuditCLI:
    def test_clean_corpus_audit_exits_zero(self, tmp_path: Path) -> None:
        repo = _bare_commit_repo(tmp_path / 'df-repo')
        tasks_dir = tmp_path / 'tasks'
        ids = _seed_zeta_corpus(tasks_dir, repo, 12)
        for fid in ids:  # pin every evals/<id> branch -> clean
            _git(['branch', f'evals/{fid}', 'HEAD'], repo)
        result = CliRunner().invoke(
            main, ['eval-list-fixtures', '--tasks-dir', str(tasks_dir), '--audit'],
        )
        assert result.exit_code == 0, result.output
        assert 'audit (revival-zeta)' in result.output
        assert 'ok=True' in result.output

    def test_missing_branch_audit_exits_nonzero(self, tmp_path: Path) -> None:
        repo = _bare_commit_repo(tmp_path / 'df-repo')
        tasks_dir = tmp_path / 'tasks'
        ids = _seed_zeta_corpus(tasks_dir, repo, 12)
        for fid in ids[:-1]:  # leave the last id's branch UNpinned
            _git(['branch', f'evals/{fid}', 'HEAD'], repo)
        result = CliRunner().invoke(
            main, ['eval-list-fixtures', '--tasks-dir', str(tasks_dir), '--audit'],
        )
        assert result.exit_code == 1, result.output
        assert 'ok=False' in result.output
        assert f'missing_branch:{ids[-1]}' in result.output

    def test_band_violation_audit_exits_nonzero(self, tmp_path: Path) -> None:
        repo = _bare_commit_repo(tmp_path / 'df-repo')
        tasks_dir = tmp_path / 'tasks'
        ids = _seed_zeta_corpus(tasks_dir, repo, 5)  # below the [10,14] floor
        for fid in ids:
            _git(['branch', f'evals/{fid}', 'HEAD'], repo)
        result = CliRunner().invoke(
            main, ['eval-list-fixtures', '--tasks-dir', str(tasks_dir), '--audit'],
        )
        assert result.exit_code == 1, result.output
        assert any('band' in line for line in result.output.splitlines()), result.output

    def test_bare_audit_ignores_retained_legacy_cohort(self, tmp_path: Path) -> None:
        # The fix under test: a bare --audit (no --cohort) scopes to revival-ζ,
        # so retained legacy fixtures (different cohort, no pinned branch) do
        # NOT push the corpus out of band or trip missing_branch.
        repo = _bare_commit_repo(tmp_path / 'df-repo')
        tasks_dir = tmp_path / 'tasks'
        ids = _seed_zeta_corpus(tasks_dir, repo, 12)
        for fid in ids:
            _git(['branch', f'evals/{fid}', 'HEAD'], repo)
        # Add unpinned legacy-cohort fixtures (disjoint id space) alongside the
        # clean zeta corpus.
        _seed_zeta_corpus(tasks_dir, repo, 3, cohort='legacy-april', start=100)
        result = CliRunner().invoke(
            main, ['eval-list-fixtures', '--tasks-dir', str(tasks_dir), '--audit'],
        )
        assert result.exit_code == 0, result.output
        assert 'audit (revival-zeta)' in result.output
        assert 'ok=True' in result.output

    def test_explicit_cohort_scopes_the_audit(self, tmp_path: Path) -> None:
        # --cohort overrides the revival-ζ default: auditing the legacy cohort
        # (5 fixtures, unpinned) fails band + missing_branch, not the zeta set.
        repo = _bare_commit_repo(tmp_path / 'df-repo')
        tasks_dir = tmp_path / 'tasks'
        ids = _seed_zeta_corpus(tasks_dir, repo, 12)
        for fid in ids:
            _git(['branch', f'evals/{fid}', 'HEAD'], repo)
        _seed_zeta_corpus(tasks_dir, repo, 5, cohort='legacy-april', start=100)
        result = CliRunner().invoke(
            main,
            ['eval-list-fixtures', '--tasks-dir', str(tasks_dir),
             '--cohort', 'legacy-april', '--audit'],
        )
        assert result.exit_code == 1, result.output
        assert 'audit (legacy-april)' in result.output
        assert any('band' in line for line in result.output.splitlines()), result.output
