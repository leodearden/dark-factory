"""Tests for elo.is_pairwise_data_dense — the Open-Q3 density gate (task 2477 λ).

Elo is DEMOTED (decision 10): the primary leaderboard is now
report.build_composite_report; the pairwise Elo path is retained only when the
pairwise judge data is dense, gated by this predicate ("drop Elo unless pairwise
judge data is dense").
"""

from __future__ import annotations


def _pool(ratings, pair_counts=None):
    from orchestrator.evals.elo import TaskPool

    return TaskPool(ratings=dict(ratings), pair_counts=dict(pair_counts or {}))


class TestIsPairwiseDataDense:
    """Density predicate over pool.ratings / pool.pair_counts (_pair_key keys)."""

    def test_sparse_pool_is_not_dense(self):
        from orchestrator.evals.elo import is_pairwise_data_dense

        # 3 configs but only ONE pair judged once; A|C and B|C are absent (0).
        pool = _pool(
            {'A': 1500.0, 'B': 1500.0, 'C': 1500.0},
            {'A|B': 1},
        )
        assert is_pairwise_data_dense(pool) is False

    def test_dense_pool_is_dense(self):
        from orchestrator.evals.elo import is_pairwise_data_dense

        # Every unordered pair at the default threshold (3).
        pool = _pool(
            {'A': 1500.0, 'B': 1500.0, 'C': 1500.0},
            {'A|B': 3, 'A|C': 3, 'B|C': 3},
        )
        assert is_pairwise_data_dense(pool) is True

    def test_fewer_than_two_configs_is_not_dense(self):
        from orchestrator.evals.elo import is_pairwise_data_dense

        # No pair is possible with <2 configs.
        assert is_pairwise_data_dense(_pool({'A': 1500.0}, {})) is False
        assert is_pairwise_data_dense(_pool({}, {})) is False

    def test_one_underfilled_pair_breaks_density(self):
        from orchestrator.evals.elo import is_pairwise_data_dense

        # Every pair at 3 except B|C at 2 → a single sparse pair → not dense.
        pool = _pool(
            {'A': 1500.0, 'B': 1500.0, 'C': 1500.0},
            {'A|B': 3, 'A|C': 3, 'B|C': 2},
        )
        assert is_pairwise_data_dense(pool) is False

    def test_custom_min_matches_per_pair(self):
        from orchestrator.evals.elo import is_pairwise_data_dense

        pool = _pool(
            {'A': 1500.0, 'B': 1500.0, 'C': 1500.0},
            {'A|B': 2, 'A|C': 2, 'B|C': 2},
        )
        # Default min=3 → not dense; a relaxed min=2 → dense.
        assert is_pairwise_data_dense(pool) is False
        assert is_pairwise_data_dense(pool, min_matches_per_pair=2) is True

    def test_two_configs_single_dense_pair(self):
        from orchestrator.evals.elo import is_pairwise_data_dense

        pool = _pool({'A': 1500.0, 'B': 1500.0}, {'A|B': 3})
        assert is_pairwise_data_dense(pool) is True
