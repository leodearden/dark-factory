"""Tests for reviewer_trial FN-candidate mining (task 2495).

All tests use synthetic fixtures (see _reviewer_trial_mining_fixtures.py) —
none touch the real, gitignored data/orchestrator/runs.db or
data/escalations/.
"""

from __future__ import annotations

import subprocess
from collections import Counter
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

from orchestrator.evals.reviewer_trial.__main__ import (
    _infer_language,
    _select_spot_check_subset,
    cli,
)
from orchestrator.evals.reviewer_trial.adjudication import AdjudicationLog
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, CorpusManifest, GroundTruthIssue
from orchestrator.evals.reviewer_trial.mining import (
    EscalationRef,
    FnCandidate,
    assign_split,
    audit_corpus,
    mine_escalation_refs,
    mine_fn_candidates,
    propose_labels_frontier,
    recover_diff,
    resolve_merge_sha_by_task_id,
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

        sha = resolve_merge_sha_by_task_id('9999', repo)

        assert sha == expected_sha

    def test_returns_none_when_no_matching_commit(self, tmp_path: Path) -> None:
        repo, _merge_sha = make_git_repo_with_merge(tmp_path)

        sha = resolve_merge_sha_by_task_id('no-such-task', repo)

        assert sha is None

    def test_returns_none_for_nonexistent_repo(self, tmp_path: Path) -> None:
        sha = resolve_merge_sha_by_task_id('1234', tmp_path / 'does-not-exist')

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


class TestSelectSpotCheckSubset:
    """``__main__._select_spot_check_subset`` -- deterministic, evenly-spread
    spot-check sampling with a floor and sticky (grow-only) membership.

    Colocated here (rather than in test_reviewer_trial_main.py) because
    these are CLI helpers that back the `mine` pipeline machinery this file
    already tests end to end."""

    def test_empty_input_returns_empty_set(self) -> None:
        assert _select_spot_check_subset([]) == set()

    def test_below_minimum_floor_selects_everything(self) -> None:
        # n=3 < minimum=5 -> k is capped at n (not padded past the pool size).
        diff_ids = ['a', 'b', 'c']
        result = _select_spot_check_subset(diff_ids, minimum=5)
        assert result == {'a', 'b', 'c'}

    def test_fraction_rounds_to_target_size(self) -> None:
        diff_ids = [f'd{i:03d}' for i in range(100)]
        result = _select_spot_check_subset(diff_ids, fraction=0.1, minimum=5)
        assert len(result) == 10

    def test_deterministic_across_calls_and_input_order(self) -> None:
        diff_ids = [f'd{i:03d}' for i in range(37)]
        first = _select_spot_check_subset(diff_ids)
        reordered = _select_spot_check_subset(list(reversed(diff_ids)))
        again = _select_spot_check_subset(diff_ids)
        assert first == reordered == again

    def test_sticky_ids_survive_even_when_sampling_would_drop_them(self) -> None:
        """A diff flagged in a smaller corpus must stay flagged after the
        corpus grows, even though the deterministic even-spread sample over
        the larger id set no longer lands on it -- this is the bug the
        `sticky` param fixes: prior human sign-off silently disappearing on
        a later `mine` resave."""
        small = [f'd{i:02d}' for i in range(10)]
        prior_subset = _select_spot_check_subset(small)
        assert prior_subset == {'d00', 'd02', 'd04', 'd06', 'd08'}

        grown = [f'd{i:02d}' for i in range(40)]
        naive = _select_spot_check_subset(grown)  # no sticky: simulates the pre-fix bug
        assert not prior_subset <= naive, 'test premise: naive resample must drop some prior ids'

        result = _select_spot_check_subset(grown, sticky=prior_subset)
        assert prior_subset <= result, 'sticky ids must never be dropped'
        assert result - prior_subset, 'new ids should still be sampled to grow the subset'

    def test_sticky_ids_no_longer_in_corpus_are_dropped(self) -> None:
        """A sticky id that no longer exists in diff_ids (e.g. removed from
        the corpus) must not leak into the result."""
        diff_ids = ['a', 'b', 'c']
        result = _select_spot_check_subset(diff_ids, sticky={'zzz-not-present'})
        assert 'zzz-not-present' not in result


class TestInferLanguage:
    """``__main__._infer_language`` -- majority-vote language sniff from a
    unified diff's ``diff --git`` headers."""

    def test_majority_language_wins(self) -> None:
        diff_text = (
            'diff --git a/a.py b/a.py\n'
            'diff --git a/b.py b/b.py\n'
            'diff --git a/c.rs b/c.rs\n'
        )
        assert _infer_language(diff_text) == 'python'

    def test_tie_breaks_by_first_encountered_file(self) -> None:
        diff_text = (
            'diff --git a/c.rs b/c.rs\n'
            'diff --git a/a.py b/a.py\n'
        )
        assert _infer_language(diff_text) == 'rust'

    def test_unrecognized_extension_defaults_to_python(self) -> None:
        diff_text = 'diff --git a/README.md b/README.md\n'
        assert _infer_language(diff_text) == 'python'

    def test_empty_diff_defaults_to_python(self) -> None:
        assert _infer_language('') == 'python'

    def test_typescript_extension_recognized(self) -> None:
        diff_text = (
            'diff --git a/a.tsx b/a.tsx\n'
            'diff --git a/b.tsx b/b.tsx\n'
        )
        assert _infer_language(diff_text) == 'typescript'


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
