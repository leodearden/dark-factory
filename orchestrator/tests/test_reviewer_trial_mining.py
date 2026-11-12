"""Tests for reviewer_trial FN-candidate mining (task 2495).

All tests use synthetic fixtures (see _reviewer_trial_mining_fixtures.py) —
none touch the real, gitignored data/orchestrator/runs.db or
data/escalations/.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Collection
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _reviewer_trial_mining_fixtures import (
    build_synthetic_runs_db,
    make_git_repo_with_merge,
    write_archived_escalations,
    write_sample_escalations,
)
from click.testing import CliRunner
from shared.cli_invoke import AgentResult

from orchestrator.evals.reviewer_trial.__main__ import cli
from orchestrator.evals.reviewer_trial.adjudication import AdjudicationLog, select_spot_check_subset
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest, GroundTruthIssue
from orchestrator.evals.reviewer_trial.mining import (
    EscalationRef,
    FnCandidate,
    RecoveredCandidate,
    admit_mined_diff,
    assign_split,
    audit_corpus,
    backfill_hand_authored,
    candidate_pool,
    fetch_task_titles,
    frontier_description,
    infer_language,
    mine_escalation_refs,
    mine_fn_candidates,
    mined_corpus_diff,
    propose_labels_frontier,
    recover_candidates,
    recover_diff,
    resolve_merge_sha_by_task_id,
    save_corpus,
)


class TestMineFnCandidates:
    """FN heuristic: review PASSed (review_cycles>=1) but a downstream bug
    signal follows — outcome in {requeued,blocked} OR verify_attempts>=min
    OR a later escalation_created event for the task."""

    def test_excludes_clean_done_task(self, tmp_path: Path) -> None:
        """A clean pass (done, low verify_attempts, no escalation) is not an FN candidate."""
        db_path = build_synthetic_runs_db(tmp_path, [
            {'task_id': '1', 'outcome': 'done', 'review_cycles': 1, 'verify_attempts': 1},
        ])
        candidates = mine_fn_candidates(db_path)
        assert candidates == []

    def test_includes_requeued_outcome(self, tmp_path: Path) -> None:
        db_path = build_synthetic_runs_db(tmp_path, [
            {'task_id': '2', 'outcome': 'requeued', 'review_cycles': 1, 'verify_attempts': 1},
        ])
        candidates = mine_fn_candidates(db_path)
        assert len(candidates) == 1
        assert candidates[0].task_id == '2'
        assert any('requeued' in r for r in candidates[0].signal_reason)

    def test_includes_blocked_outcome(self, tmp_path: Path) -> None:
        db_path = build_synthetic_runs_db(tmp_path, [
            {'task_id': '3', 'outcome': 'blocked', 'review_cycles': 1, 'verify_attempts': 1},
        ])
        candidates = mine_fn_candidates(db_path)
        assert len(candidates) == 1
        assert candidates[0].task_id == '3'
        assert any('blocked' in r for r in candidates[0].signal_reason)

    def test_includes_high_verify_attempts(self, tmp_path: Path) -> None:
        db_path = build_synthetic_runs_db(tmp_path, [
            {'task_id': '4', 'outcome': 'done', 'review_cycles': 1, 'verify_attempts': 3},
        ])
        candidates = mine_fn_candidates(db_path, min_verify_attempts=2)
        assert len(candidates) == 1
        assert candidates[0].task_id == '4'
        assert any('verify_attempts' in r for r in candidates[0].signal_reason)

    def test_verify_attempts_threshold_is_configurable(self, tmp_path: Path) -> None:
        db_path = build_synthetic_runs_db(tmp_path, [
            {'task_id': '5', 'outcome': 'done', 'review_cycles': 1, 'verify_attempts': 1},
        ])
        assert mine_fn_candidates(db_path, min_verify_attempts=2) == []
        candidates = mine_fn_candidates(db_path, min_verify_attempts=1)
        assert len(candidates) == 1
        assert candidates[0].task_id == '5'

    def test_includes_task_with_escalation_created_event(self, tmp_path: Path) -> None:
        db_path = build_synthetic_runs_db(tmp_path, [
            {
                'task_id': '6', 'outcome': 'done', 'review_cycles': 1,
                'verify_attempts': 1, 'escalation_created': True,
            },
        ])
        candidates = mine_fn_candidates(db_path)
        assert len(candidates) == 1
        assert candidates[0].task_id == '6'
        assert any('escalation' in r for r in candidates[0].signal_reason)

    def test_excludes_unreviewed_task_even_with_downstream_signal(self, tmp_path: Path) -> None:
        """review_cycles == 0 means the task never went through review at all —
        it cannot be an FN (reviewer never had a chance to PASS it)."""
        db_path = build_synthetic_runs_db(tmp_path, [
            {'task_id': '7', 'outcome': 'requeued', 'review_cycles': 0, 'verify_attempts': 1},
        ])
        candidates = mine_fn_candidates(db_path)
        assert candidates == []

    def test_carries_task_id_project_id_merge_sha_and_signal_reason(self, tmp_path: Path) -> None:
        db_path = build_synthetic_runs_db(tmp_path, [
            {
                'task_id': '8', 'project_id': 'reify', 'outcome': 'blocked',
                'review_cycles': 2, 'verify_attempts': 1, 'merge_sha': 'deadbeef123',
            },
        ])
        candidates = mine_fn_candidates(db_path)
        assert len(candidates) == 1
        cand = candidates[0]
        assert isinstance(cand, FnCandidate)
        assert cand.task_id == '8'
        assert cand.project_id == 'reify'
        assert cand.merge_sha == 'deadbeef123'
        assert cand.signal_reason  # non-empty

    def test_merge_sha_is_none_when_no_merge_finalized_event(self, tmp_path: Path) -> None:
        db_path = build_synthetic_runs_db(tmp_path, [
            {'task_id': '9', 'outcome': 'requeued', 'review_cycles': 1, 'verify_attempts': 1},
        ])
        candidates = mine_fn_candidates(db_path)
        assert candidates[0].merge_sha is None

    def test_multiple_rows_mixed(self, tmp_path: Path) -> None:
        db_path = build_synthetic_runs_db(tmp_path, [
            {'task_id': 'clean', 'outcome': 'done', 'review_cycles': 1, 'verify_attempts': 1},
            {'task_id': 'fn1', 'outcome': 'requeued', 'review_cycles': 1, 'verify_attempts': 1},
            {'task_id': 'fn2', 'outcome': 'done', 'review_cycles': 1, 'verify_attempts': 4},
            {'task_id': 'unreviewed', 'outcome': 'blocked', 'review_cycles': 0, 'verify_attempts': 1},
        ])
        candidates = mine_fn_candidates(db_path)
        ids = {c.task_id for c in candidates}
        assert ids == {'fn1', 'fn2'}


class TestMineEscalationRefs:
    """mine_escalation_refs parses esc-<id>.json records, skipping noise files
    and tolerating malformed JSON, keyed by task_id for cross-referencing."""

    def test_parses_valid_escalation_records(self, tmp_path: Path) -> None:
        esc_dir = tmp_path / 'escalations'
        write_sample_escalations(esc_dir)

        refs = mine_escalation_refs(esc_dir)

        assert '101' in refs
        assert '202' in refs
        ref = refs['101'][0]
        assert isinstance(ref, EscalationRef)
        assert ref.task_id == '101'
        assert ref.category == 'review_suggestions'
        assert ref.severity == 'blocking'
        assert ref.summary == 'Sample escalation summary 1'
        assert ref.level == 0
        assert ref.path.name == 'esc-101-1.json'

    def test_skips_lock_and_state_files(self, tmp_path: Path) -> None:
        esc_dir = tmp_path / 'escalations'
        write_sample_escalations(esc_dir)

        refs = mine_escalation_refs(esc_dir)

        all_paths = [r.path.name for group in refs.values() for r in group]
        assert not any(p.endswith('.lock') for p in all_paths)
        assert 'b3-state.json' not in all_paths

    def test_tolerates_malformed_json_without_raising(self, tmp_path: Path) -> None:
        esc_dir = tmp_path / 'escalations'
        write_sample_escalations(esc_dir)  # includes esc-303-1.json = malformed

        refs = mine_escalation_refs(esc_dir)  # must not raise

        all_paths = [r.path.name for group in refs.values() for r in group]
        assert 'esc-303-1.json' not in all_paths

    def test_empty_dir_returns_empty_dict(self, tmp_path: Path) -> None:
        esc_dir = tmp_path / 'empty_escalations'
        esc_dir.mkdir()
        assert mine_escalation_refs(esc_dir) == {}

    def test_recurses_into_dated_archive_subdirectories(self, tmp_path: Path) -> None:
        """The real data/escalations/ holds only *.lock/*.seq sentinels for
        OPEN escalations at the top level; resolved records rotate into
        dated archive/<YYYY-MM-DD>/esc-*.json subdirectories (discovered
        running the miner against the real tree for step-18). Confirmed via
        `find data/escalations/archive -name 'esc-*.json' | wc -l` = 1292
        real records live there, invisible to a non-recursive glob."""
        esc_dir = tmp_path / 'escalations'
        write_archived_escalations(esc_dir)

        refs = mine_escalation_refs(esc_dir)

        assert '404' in refs
        ref = refs['404'][0]
        assert isinstance(ref, EscalationRef)
        assert ref.task_id == '404'
        assert ref.path.name == 'esc-404-1.json'
        # The archived lock sentinel alongside it must still be skipped.
        all_paths = [r.path.name for group in refs.values() for r in group]
        assert 'esc-404-1.json.lock' not in all_paths


class TestRecoverDiff:
    """recover_diff shells out to git to recover the unified diff for a merge_sha."""

    def test_recovers_diff_for_merge_commit(self, tmp_path: Path) -> None:
        repo, merge_sha = make_git_repo_with_merge(tmp_path)

        diff_text = recover_diff(merge_sha, repo)

        assert diff_text is not None
        assert diff_text.strip() != ''
        assert '+++' in diff_text
        assert '---' in diff_text

    def test_returns_none_for_unknown_sha(self, tmp_path: Path) -> None:
        repo, _merge_sha = make_git_repo_with_merge(tmp_path)

        diff_text = recover_diff('0' * 40, repo)

        assert diff_text is None


class TestResolveMergeShaByTaskId:
    """Fallback recovery: greps git log for the 'Merge task/<id> into ...'
    commit-message convention when a candidate has no merge_finalized event."""

    def test_resolves_sha_from_merge_commit_message_convention(self, tmp_path: Path) -> None:
        repo, _fixture_merge_sha = make_git_repo_with_merge(tmp_path)
        subprocess.run(
            ['git', 'commit', '--allow-empty', '-q', '-m', 'Merge task/9999 into main'],
            cwd=repo, check=True,
        )
        expected_sha = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()

        sha = resolve_merge_sha_by_task_id('9999', repo, branch_prefix='task/', main_branch='main')

        assert sha == expected_sha

    def test_returns_none_when_no_matching_commit(self, tmp_path: Path) -> None:
        repo, _merge_sha = make_git_repo_with_merge(tmp_path)

        sha = resolve_merge_sha_by_task_id('no-such-task', repo, branch_prefix='task/', main_branch='main')

        assert sha is None

    def test_returns_none_for_nonexistent_repo(self, tmp_path: Path) -> None:
        sha = resolve_merge_sha_by_task_id('1234', tmp_path / 'does-not-exist', branch_prefix='task/', main_branch='main')

        assert sha is None


class TestAssignSplit:
    """assign_split deterministically buckets diff_ids into train/selection/test."""

    def test_maps_every_id_to_exactly_one_valid_split(self) -> None:
        diff_ids = [f'd{i}' for i in range(20)]
        result = assign_split(diff_ids, seed='seed1')
        assert set(result.keys()) == set(diff_ids)
        assert all(v in {'train', 'selection', 'test'} for v in result.values())

    def test_deterministic_for_fixed_seed(self) -> None:
        diff_ids = [f'd{i}' for i in range(50)]
        result1 = assign_split(diff_ids, seed='seed-fixed')
        result2 = assign_split(diff_ids, seed='seed-fixed')
        assert result1 == result2

    def test_varies_with_seed(self) -> None:
        diff_ids = [f'd{i}' for i in range(50)]
        result1 = assign_split(diff_ids, seed='seed-a')
        result2 = assign_split(diff_ids, seed='seed-b')
        assert result1 != result2

    def test_ratios_approximate_2_1_7_for_large_n(self) -> None:
        diff_ids = [f'd{i}' for i in range(300)]
        result = assign_split(diff_ids, seed='seed-ratio')
        counts = Counter(result.values())
        n = len(diff_ids)

        assert abs(counts['train'] / n - 0.2) < 0.05
        assert abs(counts['selection'] / n - 0.1) < 0.05
        assert abs(counts['test'] / n - 0.7) < 0.05


def _make_frontier_result(
    cost_usd: float = 0.15,
    structured: dict | None = None,
    output: str = '',
) -> AgentResult:
    """Build a real AgentResult for invoke_agent in propose_labels_frontier.

    Mirrors _make_matcher_result in test_reviewer_trial_scorer.py — uses the
    actual AgentResult dataclass so field-shape drift is caught at test time.
    """
    return AgentResult(
        success=True,
        output=output,
        cost_usd=cost_usd,
        structured_output=structured,
    )


class TestProposeLabelsFrontier:
    """propose_labels_frontier maps a frontier model's structured output into
    GroundTruthIssue objects tagged with adjudication provenance (an id
    namespaced 'frontier_*' rather than a hand-authored corpus id), and never
    raises on unparseable output."""

    @pytest.mark.asyncio
    async def test_maps_structured_output_to_ground_truth_issues(self) -> None:
        fake_result = _make_frontier_result(
            cost_usd=0.15,
            structured={
                'issues': [
                    {
                        'location': 'src/foo.py:10',
                        'category': 'off_by_one',
                        'severity': 'blocking',
                        'description': 'Loop bound is off by one.',
                        'mutation_type': 'logic',
                    },
                ],
            },
        )

        with patch('orchestrator.evals.reviewer_trial.mining.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = fake_result
            issues, cost = await propose_labels_frontier(
                'diff text', model='opus', description='a mined diff',
            )

        assert cost == pytest.approx(0.15)
        assert len(issues) == 1
        issue = issues[0]
        assert isinstance(issue, GroundTruthIssue)
        assert issue.location == 'src/foo.py:10'
        assert issue.category == 'off_by_one'
        assert issue.severity == 'blocking'
        assert issue.description == 'Loop bound is off by one.'
        assert issue.mutation_type == 'logic'
        # id is synthesized (never hallucinated by the model) and tagged so
        # a frontier-proposed label is distinguishable from a hand-authored one.
        assert issue.id.startswith('frontier_')

    @pytest.mark.asyncio
    async def test_maps_multiple_issues_with_distinct_ids(self) -> None:
        fake_result = _make_frontier_result(structured={
            'issues': [
                {
                    'location': 'a.py:1', 'category': 'bug', 'severity': 'blocking',
                    'description': 'First.', 'mutation_type': 'logic',
                },
                {
                    'location': 'b.py:2', 'category': 'style', 'severity': 'suggestion',
                    'description': 'Second.', 'mutation_type': 'style',
                },
            ],
        })

        with patch('orchestrator.evals.reviewer_trial.mining.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = fake_result
            issues, _cost = await propose_labels_frontier('diff text', model='opus', description='d')

        assert len(issues) == 2
        assert len({i.id for i in issues}) == 2  # ids are distinct

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_unparseable_output_without_raising(self) -> None:
        fake_result = _make_frontier_result(cost_usd=0.05, structured=None, output='not json at all')

        with patch('orchestrator.evals.reviewer_trial.mining.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = fake_result
            issues, cost = await propose_labels_frontier('diff text', model='opus', description='d')

        assert issues == []
        # Cost is still reported — the tokens were billed regardless (mirrors
        # match_issues' unparseable-output-still-reports-cost behaviour).
        assert cost == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_issues_key_missing(self) -> None:
        fake_result = _make_frontier_result(structured={'not_issues': []})

        with patch('orchestrator.evals.reviewer_trial.mining.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = fake_result
            issues, _cost = await propose_labels_frontier('diff text', model='opus', description='d')

        assert issues == []


class TestAuditCorpus:
    """audit_corpus(manifest, adjudication_log, min_diffs) reports ok=True
    only when every corpus-integrity signal holds, naming which check(s)
    failed otherwise (task 2495 / PRD D-6)."""

    @staticmethod
    def _make_diff(
        diff_id: str,
        source: str = 'synthetic',
        split: str | None = None,
        provenance: dict | None = None,
    ) -> CorpusDiff:
        return CorpusDiff(
            diff_id=diff_id,
            language='python',
            source=source,
            diff_text='--- a\n+++ b\n',
            description='d',
            ground_truth=[],
            split=split,
            provenance=provenance,
        )

    @classmethod
    def _passing_manifest_and_log(cls, n: int = 30) -> tuple[CorpusManifest, AdjudicationLog]:
        """A manifest + adjudication log that satisfies all five audit_corpus
        checks: >=n diffs, every diff split (~2:1:7 via assign_split), every
        mined diff has provenance, adjudication covers every diff_id, and a
        non-empty spot-check subset is flagged."""
        diff_ids = [f'd{i}' for i in range(n)]
        splits = assign_split(diff_ids, seed='audit-test')

        diffs = []
        for i, diff_id in enumerate(diff_ids):
            source = 'mined' if i % 2 == 0 else 'synthetic'
            provenance = {'merge_sha': 'deadbeef'} if source == 'mined' else None
            diffs.append(cls._make_diff(diff_id, source=source, split=splits[diff_id], provenance=provenance))
        manifest = CorpusManifest(diffs=diffs, split_seed='audit-test')

        log = AdjudicationLog()
        for i, diff_id in enumerate(diff_ids):
            log.append(diff_id, frontier_proposal=[], in_spot_check_subset=(i == 0))

        return manifest, log

    def test_passes_when_all_signals_hold(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)

        report = audit_corpus(manifest, log, min_diffs=30)

        assert report.ok is True
        assert report.failures == []
        assert report.diff_count == 30

    def test_fails_when_below_min_diffs(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=10)

        report = audit_corpus(manifest, log, min_diffs=50)

        assert report.ok is False
        assert 'diff_count' in report.failures

    def test_fails_when_split_missing(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        manifest.diffs[0].split = None

        report = audit_corpus(manifest, log, min_diffs=30)

        assert report.ok is False
        assert 'missing_split' in report.failures

    def test_fails_when_split_ratio_off(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        for d in manifest.diffs:
            d.split = 'train'  # every diff in one bucket -> ratio way off 2:1:7

        report = audit_corpus(manifest, log, min_diffs=30)

        assert report.ok is False
        assert 'split_ratio' in report.failures

    def test_fails_when_mined_diff_missing_provenance(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        mined_diff = next(d for d in manifest.diffs if d.source == 'mined')
        mined_diff.provenance = None

        report = audit_corpus(manifest, log, min_diffs=30)

        assert report.ok is False
        assert 'missing_provenance' in report.failures

    def test_fails_when_adjudication_coverage_incomplete(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        log.entries.pop()  # last diff_id now has no adjudication entry

        report = audit_corpus(manifest, log, min_diffs=30)

        assert report.ok is False
        assert 'adjudication_coverage' in report.failures

    def test_fails_when_spot_check_subset_empty(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        for entry in log.entries:
            entry.in_spot_check_subset = False

        report = audit_corpus(manifest, log, min_diffs=30)

        assert report.ok is False
        assert 'spot_check_subset_empty' in report.failures


class TestInferLanguage:
    """``infer_language`` -- majority-vote language sniff from a
    unified diff's ``diff --git`` headers."""

    def test_majority_language_wins(self) -> None:
        diff_text = (
            'diff --git a/a.py b/a.py\n'
            'diff --git a/b.py b/b.py\n'
            'diff --git a/c.rs b/c.rs\n'
        )
        assert infer_language(diff_text) == 'python'

    def test_tie_breaks_by_first_encountered_file(self) -> None:
        diff_text = (
            'diff --git a/c.rs b/c.rs\n'
            'diff --git a/a.py b/a.py\n'
        )
        assert infer_language(diff_text) == 'rust'

    def test_unrecognized_extension_defaults_to_python(self) -> None:
        diff_text = 'diff --git a/README.md b/README.md\n'
        assert infer_language(diff_text) == 'python'

    def test_empty_diff_defaults_to_python(self) -> None:
        assert infer_language('') == 'python'

    def test_typescript_extension_recognized(self) -> None:
        diff_text = (
            'diff --git a/a.tsx b/a.tsx\n'
            'diff --git a/b.tsx b/b.tsx\n'
        )
        assert infer_language(diff_text) == 'typescript'


class TestMineRunsDbValidation:
    """`mine` fails fast with a clear message when --runs-db doesn't point at
    a real file, instead of letting sqlite3 silently create an empty
    database that then blows up deeper in the pipeline with a confusing
    ``OperationalError: no such table``.

    The missing-file check runs before `_load_corpus()` / any real I/O, so
    this never touches the real committed corpus (consistent with this
    file's synthetic-fixtures-only policy)."""

    def test_missing_runs_db_exits_with_clear_error(self, tmp_path: Path) -> None:
        missing_db = tmp_path / 'does-not-exist.db'
        runner = CliRunner()
        result = runner.invoke(cli, ['mine', '--runs-db', str(missing_db)])

        assert result.exit_code == 1
        assert 'runs.db not found' in result.output
        assert str(missing_db) in result.output


_SEED = 'test-seed'
_MINED_IDS = [f'mined_{i:02d}' for i in range(12)]


def _issue(diff_id: str) -> GroundTruthIssue:
    return GroundTruthIssue(
        id=f'{diff_id}_gt', location='a.py:1', category='bug', severity='blocking',
        description=f'a bug in {diff_id}', mutation_type='logic',
    )


def _corpus_and_log(*, mined_flagged: Collection[str] = ()) -> tuple[CorpusManifest, AdjudicationLog]:
    """3 synthetic plus 12 mined diffs, each with a log entry; *mined_flagged* start in the spot-check subset."""
    synthetic = [
        CorpusDiff(diff_id=f'syn_{i}', language='python', source='synthetic',
                   diff_text=f'syn {i}\n', description=f'syn {i}', ground_truth=[_issue(f'syn_{i}')])
        for i in range(3)
    ]
    mined = [
        CorpusDiff(diff_id=diff_id, language='python', source='mined', diff_text=f'{diff_id}\n',
                   description=diff_id, ground_truth=[_issue(diff_id)], provenance={'kind': 'mined'})
        for diff_id in _MINED_IDS
    ]
    manifest = CorpusManifest(diffs=[*synthetic, *mined])
    log = AdjudicationLog()
    for diff in manifest.diffs:
        log.append(diff.diff_id, diff.ground_truth, in_spot_check_subset=diff.diff_id in mined_flagged)
    return manifest, log


def _flagged(log: AdjudicationLog) -> set[str]:
    return {e.diff_id for e in log.spot_check_subset()}


class TestSaveCorpus:
    """save_corpus grows the spot-check subset over mined diffs, re-splits every diff, and saves both."""

    @staticmethod
    def _save(tmp_path: Path, manifest: CorpusManifest, log: AdjudicationLog) -> tuple[Path, Path]:
        manifest_path = tmp_path / 'corpus' / 'manifest.json'
        log_path = tmp_path / 'corpus' / 'adjudication_log.jsonl'
        save_corpus(manifest, log, seed=_SEED, manifest_path=manifest_path, log_path=log_path)
        return manifest_path, log_path

    def test_every_diff_is_split_under_the_seed(self, tmp_path: Path) -> None:
        manifest, log = _corpus_and_log()

        self._save(tmp_path, manifest, log)

        assert {d.diff_id: d.split for d in manifest.diffs} == assign_split(
            [d.diff_id for d in manifest.diffs], seed=_SEED,
        )
        assert manifest.split_seed == _SEED

    def test_the_spot_check_subset_is_sampled_over_mined_diffs_only(self, tmp_path: Path) -> None:
        manifest, log = _corpus_and_log()

        self._save(tmp_path, manifest, log)

        assert _flagged(log) == select_spot_check_subset(_MINED_IDS)

    def test_what_it_saves_reads_back(self, tmp_path: Path) -> None:
        manifest, log = _corpus_and_log()

        manifest_path, log_path = self._save(tmp_path, manifest, log)

        loaded = CorpusManifest.load(manifest_path)
        assert {d.diff_id: d.split for d in loaded.diffs} == {d.diff_id: d.split for d in manifest.diffs}
        assert loaded.split_seed == _SEED
        assert _flagged(AdjudicationLog.load(log_path)) == _flagged(log)

    def test_a_flagged_mined_entry_stays_flagged(self, tmp_path: Path) -> None:
        unsampled = sorted(set(_MINED_IDS) - select_spot_check_subset(_MINED_IDS))[0]
        manifest, log = _corpus_and_log(mined_flagged={unsampled})

        self._save(tmp_path, manifest, log)

        assert unsampled in _flagged(log)


class TestBackfillHandAuthored:
    """backfill_hand_authored logs each unlogged hand-authored diff with its own ground truth."""

    def test_each_unlogged_hand_authored_diff_gets_one_entry(self) -> None:
        manifest, _ = _corpus_and_log()
        log = AdjudicationLog()

        backfill_hand_authored(manifest, log)

        assert [
            (e.diff_id, e.frontier_model, e.frontier_proposal, e.in_spot_check_subset) for e in log.entries
        ] == [
            (d.diff_id, 'hand-authored', [asdict(gt) for gt in d.ground_truth], False)
            for d in manifest.diffs if d.source != 'mined'
        ]

    def test_mined_and_already_logged_diffs_are_skipped(self) -> None:
        manifest, _ = _corpus_and_log()
        log = AdjudicationLog()
        log.append('syn_0', [], in_spot_check_subset=False, frontier_model='opus')

        backfill_hand_authored(manifest, log)

        assert [(e.diff_id, e.frontier_model) for e in log.entries] == [
            ('syn_0', 'opus'), ('syn_1', 'hand-authored'), ('syn_2', 'hand-authored'),
        ]

    def test_it_returns_how_many_it_added(self) -> None:
        manifest, _ = _corpus_and_log()
        log = AdjudicationLog()

        assert backfill_hand_authored(manifest, log) == 3
        assert backfill_hand_authored(manifest, log) == 0


def _candidate(task_id: str, merge_sha: str | None) -> FnCandidate:
    return FnCandidate(
        task_id=task_id, project_id='dark_factory', outcome='requeued', review_cycles=2,
        verify_attempts=3, merge_sha=merge_sha, signal_reason=['outcome=requeued', 'verify_attempts=3'],
    )


def _commit_task_merge(repo: Path, task_id: str, content: str) -> str:
    """Commit *content* under the ``Merge task/<id> into main`` message convention; return its sha."""
    (repo / f'{task_id}.txt').write_text(content)
    subprocess.run(['git', 'add', f'{task_id}.txt'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', f'Merge task/{task_id} into main'], cwd=repo, check=True)
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


class TestFetchTaskTitles:
    """fetch_task_titles reads task_results.title for the requested task ids."""

    def test_returns_the_requested_titles_only(self, tmp_path: Path) -> None:
        db = build_synthetic_runs_db(tmp_path, [
            {'task_id': '1', 'title': 'one'}, {'task_id': '2', 'title': 'two'}, {'task_id': '3', 'title': 'three'},
        ])

        assert fetch_task_titles(db, ['1', '3']) == {'1': 'one', '3': 'three'}

    def test_no_ids_returns_no_titles(self, tmp_path: Path) -> None:
        db = build_synthetic_runs_db(tmp_path, [{'task_id': '1', 'title': 'one'}])

        assert fetch_task_titles(db, []) == {}

    def test_more_ids_than_one_chunk_still_returns_every_title(self, tmp_path: Path) -> None:
        rows = [{'task_id': str(i), 'title': f'title {i}'} for i in range(1201)]
        db = build_synthetic_runs_db(tmp_path, rows)

        assert fetch_task_titles(db, [r['task_id'] for r in rows]) == {r['task_id']: r['title'] for r in rows}


class TestRecoverCandidates:
    """recover_candidates pairs each new candidate with its merge sha and diff, smallest diff first."""

    def test_a_candidate_with_a_merge_sha_is_recovered_from_it(self, tmp_path: Path) -> None:
        repo, merge_sha = make_git_repo_with_merge(tmp_path)

        [recovered] = recover_candidates(
            [_candidate('1', merge_sha)], existing_ids=set(), repo_path=repo, branch_prefix='task/', main_branch='main',
        )

        assert (recovered.candidate.task_id, recovered.merge_sha) == ('1', merge_sha)
        assert recovered.diff_text.strip()

    def test_a_candidate_already_in_the_corpus_is_skipped(self, tmp_path: Path) -> None:
        repo, merge_sha = make_git_repo_with_merge(tmp_path)
        candidate = _candidate('1', merge_sha)

        assert candidate.diff_id == 'mined_1'
        assert recover_candidates(
            [candidate], existing_ids={'mined_1'}, repo_path=repo, branch_prefix='task/', main_branch='main',
        ) == []

    def test_an_unknown_sha_is_skipped(self, tmp_path: Path) -> None:
        repo, _merge_sha = make_git_repo_with_merge(tmp_path)

        assert recover_candidates(
            [_candidate('1', '0' * 40)], existing_ids=set(), repo_path=repo, branch_prefix='task/', main_branch='main',
        ) == []

    def test_a_candidate_without_a_sha_is_found_by_its_merge_message(self, tmp_path: Path) -> None:
        repo, _merge_sha = make_git_repo_with_merge(tmp_path)
        sha = _commit_task_merge(repo, '77', 'fallback\n')

        [recovered] = recover_candidates(
            [_candidate('77', None)], existing_ids=set(), repo_path=repo, branch_prefix='task/', main_branch='main',
        )

        assert recovered.merge_sha == sha
        assert '+fallback' in recovered.diff_text

    def test_the_smallest_diff_comes_first(self, tmp_path: Path) -> None:
        repo, merge_sha = make_git_repo_with_merge(tmp_path)
        big = _commit_task_merge(repo, '88', ''.join(f'line {i}\n' for i in range(200)))

        recovered = recover_candidates(
            [_candidate('88', big), _candidate('1', merge_sha)], existing_ids=set(), repo_path=repo,
            branch_prefix='task/', main_branch='main',
        )

        assert [r.candidate.task_id for r in recovered] == ['1', '88']


def _recovered(n: int) -> list[RecoveredCandidate]:
    return [RecoveredCandidate(_candidate(str(i), f'sha{i}'), f'sha{i}', 'x' * (i + 1)) for i in range(n)]


class TestCandidatePool:
    """candidate_pool attempts what the corpus still needs plus an attrition buffer."""

    @pytest.mark.parametrize('corpus_size', [100, 120])
    def test_a_full_corpus_attempts_nothing(self, corpus_size: int) -> None:
        assert candidate_pool(_recovered(10), corpus_size=corpus_size, target_total=100, limit=None) == []

    @pytest.mark.parametrize(('corpus_size', 'pool_size'), [(90, 15), (60, 50)])
    def test_the_pool_is_what_is_needed_plus_a_buffer(self, corpus_size: int, pool_size: int) -> None:
        recovered = _recovered(100)

        assert candidate_pool(recovered, corpus_size=corpus_size, target_total=100, limit=None) == (
            recovered[:pool_size]
        )

    def test_limit_caps_the_pool(self) -> None:
        recovered = _recovered(100)

        assert candidate_pool(recovered, corpus_size=60, target_total=100, limit=7) == recovered[:7]


class TestMinedCorpusDiff:
    """mined_corpus_diff builds a mined dark-factory diff whose provenance says where it came from."""

    def test_it_is_a_mined_dark_factory_diff(self, tmp_path: Path) -> None:
        diff_text = 'diff --git a/x.rs b/x.rs\n+fn main() {}\n'
        issues = [_issue('mined_42')]

        diff = mined_corpus_diff(
            RecoveredCandidate(_candidate('42', None), 'abc123', diff_text),
            issues=issues, description='desc', escalation_refs=[], runs_db=tmp_path / 'runs.db',
        )

        assert (diff.diff_id, diff.source, diff.split, diff.project) == ('mined_42', 'mined', None, 'dark-factory')
        assert diff.language == infer_language(diff_text) == 'rust'
        assert (diff.diff_text, diff.description, diff.ground_truth) == (diff_text, 'desc', issues)

    def test_its_provenance_names_the_candidate_merge_runs_db_and_escalations(self, tmp_path: Path) -> None:
        candidate = _candidate('42', None)
        ref = EscalationRef(
            task_id='42', category='review_suggestions', severity='info', summary='s', level=1,
            path=tmp_path / 'esc-42-1.json',
        )
        runs_db = tmp_path / 'runs.db'

        diff = mined_corpus_diff(
            RecoveredCandidate(candidate, 'abc123', 'diff'),
            issues=[_issue('mined_42')], description='desc', escalation_refs=[ref], runs_db=runs_db,
        )

        provenance = diff.provenance
        assert provenance is not None
        candidate_fields = {k: v for k, v in asdict(candidate).items() if k != 'merge_sha'}
        assert {k: provenance[k] for k in candidate_fields} == candidate_fields
        assert (provenance['kind'], provenance['merge_sha'], provenance['runs_db_path']) == (
            'mined', 'abc123', str(runs_db),
        )
        assert provenance['escalation_refs'] == [{**asdict(ref), 'path': str(ref.path)}]


class TestFrontierDescription:
    def test_it_names_the_task_project_title_and_every_signal(self) -> None:
        text = frontier_description(_candidate('42', None), 'Fix the widget')

        for part in ('42', 'dark_factory', 'Fix the widget', 'outcome=requeued', 'verify_attempts=3'):
            assert part in text, (part, text)


class TestAdmitMinedDiff:
    def test_it_appends_the_diff_and_its_log_entry(self) -> None:
        manifest, log = CorpusManifest(diffs=[]), AdjudicationLog()
        diff = CorpusDiff(
            diff_id='mined_42', language='python', source='mined', diff_text='d', description='d',
            ground_truth=[_issue('mined_42')], provenance={'kind': 'mined'},
        )

        admit_mined_diff(manifest, log, diff, frontier_model='opus')

        assert manifest.diffs == [diff]
        [entry] = log.entries
        assert (entry.diff_id, entry.frontier_model, entry.frontier_proposal, entry.in_spot_check_subset) == (
            'mined_42', 'opus', [asdict(gt) for gt in diff.ground_truth], False,
        )
        assert '42' in entry.notes
