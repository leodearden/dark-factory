"""Tests for orchestrator.evals.prompt_opt.splits — the seeded train/selection/test partition (T6).

See plans/tier1-prompt-optimization-prd.md T6: the corpus is deterministically
partitioned once per run (`split_seed`), with TEST held out separately for a
single end-of-run scoring.
"""

from __future__ import annotations

from orchestrator.evals.prompt_opt.splits import CorpusSplit, split_corpus


def _items(n: int = 20) -> list[str]:
    return [f'item-{i}' for i in range(n)]


class TestSplitCorpus:
    def test_returns_corpus_split(self) -> None:
        result = split_corpus(_items(), seed=1)
        assert isinstance(result, CorpusSplit)

    def test_disjoint_and_exhaustive(self) -> None:
        items = _items(20)
        result = split_corpus(items, seed=1)

        train_set = set(result.train)
        selection_set = set(result.selection)
        test_set = set(result.test)

        assert train_set & selection_set == set()
        assert train_set & test_set == set()
        assert selection_set & test_set == set()

        assert train_set | selection_set | test_set == set(items)
        assert len(result.train) + len(result.selection) + len(result.test) == len(items)

    def test_ratio_2_1_7_test_is_largest(self) -> None:
        items = _items(20)
        result = split_corpus(items, seed=1, ratios=(2, 1, 7))

        assert len(result.train) == 4
        assert len(result.selection) == 2
        assert len(result.test) == 14
        assert len(result.test) > len(result.train)
        assert len(result.test) > len(result.selection)

    def test_deterministic_same_items_and_seed(self) -> None:
        items = _items(20)
        r1 = split_corpus(items, seed=42)
        r2 = split_corpus(items, seed=42)

        assert r1.train == r2.train
        assert r1.selection == r2.selection
        assert r1.test == r2.test

    def test_different_seed_changes_partition(self) -> None:
        items = _items(20)
        r1 = split_corpus(items, seed=1)
        r2 = split_corpus(items, seed=2)

        assert (r1.train, r1.selection, r1.test) != (r2.train, r2.selection, r2.test)

    def test_test_split_is_a_distinct_list_from_train_and_selection(self) -> None:
        result = split_corpus(_items(), seed=1)
        assert result.test is not result.train
        assert result.test is not result.selection
        assert result.train is not result.selection
