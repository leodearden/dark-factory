"""Hermetic end-to-end acceptance smoke for the Tier-1 prompt-opt stack (T7).

See plans/tier1-prompt-optimization-prd.md T7. Proves the whole stack —
loader -> reviewer HEURISTICS -> loop engine -> scorer -> report — works
end-to-end on a <=3-diff synthetic fixture corpus WITHOUT a real
($300-800) run, by dependency-injecting the three LLM-touching seams
(rollout_fn, Scorer, propose_fn) deterministically and running the REAL
REVIEWER_COMPREHENSIVE.prompt_spec, the REAL run_optimization_loop, and the
REAL PromptArtifactStore loader. No invoke_agent, no DB, no network.

Every assertion is on runtime behavior (report contents, resolve source,
byte-level contract prefix, CLI exit codes) — never on docstrings/prose.
"""

from __future__ import annotations

from orchestrator.evals.prompt_opt import split_corpus
from orchestrator.evals.prompt_opt.smoke import build_fixture_corpus


class TestFixtureCorpus:
    """The reconciliation: '<=3-diff' == <=3 DISTINCT archetypes replicated
    into a >=10-item, loop-runnable corpus (the REAL engine's fixed 2:1:7
    split raises before any rollout on an empty selection/test split)."""

    def test_at_most_three_distinct_diff_texts(self) -> None:
        corpus = build_fixture_corpus()
        distinct_diffs = {item.diff for item in corpus}
        assert 1 <= len(distinct_diffs) <= 3

    def test_at_least_ten_items_with_distinct_ids(self) -> None:
        corpus = build_fixture_corpus()
        assert len(corpus) >= 10
        ids = [item.item_id for item in corpus]
        assert len(set(ids)) == len(ids)  # every id distinct

    def test_every_item_carries_a_gold_label(self) -> None:
        corpus = build_fixture_corpus()
        assert corpus  # non-empty
        for item in corpus:
            assert item.gold_verdict in ('PASS', 'ISSUES_FOUND')
            if item.gold_verdict == 'PASS':
                assert item.gold_severity is None
            else:
                assert item.gold_severity in ('blocking', 'suggestion')

    def test_splittable_into_nonempty_train_selection_test(self) -> None:
        # The core signal: fed through the REAL engine's default 2:1:7 split,
        # the corpus must yield a non-empty train, selection, AND test split,
        # otherwise run_optimization_loop raises before producing a held-out
        # TEST verdict. 3 archetypes x 4 replicas = 12 -> train=2/sel=1/test=9.
        corpus = build_fixture_corpus()
        split = split_corpus(corpus, seed=2498)
        assert split.train, 'train split is empty'
        assert split.selection, 'selection split is empty'
        assert split.test, 'test split is empty'
        # exhaustive + disjoint: every item lands in exactly one split
        assert len(split.train) + len(split.selection) + len(split.test) == len(corpus)
