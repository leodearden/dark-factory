"""Tests for reviewer_trial FN-candidate mining (task 2495).

All tests use synthetic fixtures (see _reviewer_trial_mining_fixtures.py) —
none touch the real, gitignored data/orchestrator/runs.db or
data/escalations/.
"""

from __future__ import annotations

from pathlib import Path

from _reviewer_trial_mining_fixtures import (
    build_synthetic_runs_db,
    make_git_repo_with_merge,
    write_sample_escalations,
)

from orchestrator.evals.reviewer_trial.mining import (
    EscalationRef,
    FnCandidate,
    mine_escalation_refs,
    mine_fn_candidates,
    recover_diff,
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
