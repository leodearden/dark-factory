"""Tests for reviewer_trial FN-candidate mining (task 2495).

All tests use synthetic fixtures (see _reviewer_trial_mining_fixtures.py) —
none touch the real, gitignored data/orchestrator/runs.db or
data/escalations/.
"""

from __future__ import annotations

from pathlib import Path

from _reviewer_trial_mining_fixtures import build_synthetic_runs_db

from orchestrator.evals.reviewer_trial.mining import FnCandidate, mine_fn_candidates


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
