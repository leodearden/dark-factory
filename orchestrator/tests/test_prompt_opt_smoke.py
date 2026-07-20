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

import pytest
from shared.prompt_artifact import compose_prompt

from orchestrator.evals.prompt_opt import measure_repeatability_band, split_corpus
from orchestrator.evals.prompt_opt.smoke import (
    _SMOKE_EXECUTOR_MODEL,
    SmokeReviewerScorer,
    _smoke_rollout_fn,
    build_fixture_corpus,
)


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


class TestHermeticSeams:
    """The two LLM-free seams: the executor-model-asserting rollout_fn and the
    verdict-vs-gold Scorer with a bounded occurrence-keyed jitter cycle."""

    @pytest.mark.asyncio
    async def test_rollout_fn_requires_executor_model_and_is_deterministic(self) -> None:
        item = build_fixture_corpus()[0]
        composed = compose_prompt('CONTRACT', 'HEUR SMOKE_QUALITY=0.60')

        out = await _smoke_rollout_fn(composed, item, _SMOKE_EXECUTOR_MODEL)
        # derived only from (item, composed_prompt) — never invoke_agent
        assert out == (
            f'rollout::{item.item_id}::gold={item.gold_verdict}/{item.gold_severity}::{composed}'
        )
        # deterministic: identical inputs -> byte-identical rollout
        assert out == await _smoke_rollout_fn(composed, item, _SMOKE_EXECUTOR_MODEL)
        # must ALWAYS be called with the executor model (structural guarantee)
        with pytest.raises(AssertionError):
            await _smoke_rollout_fn(composed, item, 'not-the-executor-model')

    @pytest.mark.asyncio
    async def test_scorer_returns_float_in_unit_interval(self) -> None:
        scorer = SmokeReviewerScorer()
        item = build_fixture_corpus()[0]
        composed = compose_prompt('CONTRACT', 'HEUR SMOKE_QUALITY=0.60')
        rollout = await _smoke_rollout_fn(composed, item, _SMOKE_EXECUTOR_MODEL)
        score = await scorer.score(item, rollout)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_agreement_scores_higher_than_disagreement(self) -> None:
        # For a FIXED heuristics/quality, a rollout whose reported verdict
        # AGREES with the item's gold scores strictly higher than one that
        # disagrees.
        scorer = SmokeReviewerScorer()
        item = next(
            i for i in build_fixture_corpus()
            if i.gold_verdict == 'ISSUES_FOUND' and i.gold_severity == 'blocking'
        )
        composed = compose_prompt('CONTRACT', 'HEUR SMOKE_QUALITY=0.50')
        # the hermetic rollout reports gold == item's true gold -> agreement
        agree_rollout = await _smoke_rollout_fn(composed, item, _SMOKE_EXECUTOR_MODEL)
        # a crafted rollout reporting a DIFFERENT verdict -> disagreement (same quality)
        disagree_rollout = agree_rollout.replace(
            f'gold={item.gold_verdict}/{item.gold_severity}::', 'gold=PASS/None::'
        )
        assert disagree_rollout != agree_rollout
        agree_score = await scorer.score(item, agree_rollout)
        disagree_score = await scorer.score(item, disagree_rollout)
        assert agree_score > disagree_score

    @pytest.mark.asyncio
    async def test_higher_sentinel_yields_higher_mean_score(self) -> None:
        item = build_fixture_corpus()[0]

        async def mean_score(sentinel: str, n: int = 6) -> float:
            scorer = SmokeReviewerScorer()
            composed = compose_prompt('CONTRACT', f'HEUR {sentinel}')
            rollout = await _smoke_rollout_fn(composed, item, _SMOKE_EXECUTOR_MODEL)
            return sum([await scorer.score(item, rollout) for _ in range(n)]) / n

        low = await mean_score('SMOKE_QUALITY=0.50')
        high = await mean_score('SMOKE_QUALITY=0.70')
        assert high > low

    @pytest.mark.asyncio
    async def test_absent_sentinel_defaults_to_baseline_quality(self) -> None:
        # The REAL reviewer baseline heuristics carry no SMOKE_QUALITY sentinel
        # -> default 0.50, so the real baseline reads as baseline quality.
        item = build_fixture_corpus()[0]
        scorer = SmokeReviewerScorer()
        composed = compose_prompt('CONTRACT', 'real reviewer heuristics — no sentinel here')
        rollout = await _smoke_rollout_fn(composed, item, _SMOKE_EXECUTOR_MODEL)
        score = await scorer.score(item, rollout)
        # agreement path at the 0.50 default, within the +-0.03 jitter band
        assert abs(score - 0.50) <= 0.05

    @pytest.mark.asyncio
    async def test_jitter_cycle_yields_positive_reproducible_variance_band(self) -> None:
        item = build_fixture_corpus()[0]
        scorer = SmokeReviewerScorer()
        composed = compose_prompt('CONTRACT', 'HEUR SMOKE_QUALITY=0.50')
        rollout = await _smoke_rollout_fn(composed, item, _SMOKE_EXECUTOR_MODEL)

        # repeated scoring of IDENTICAL inputs disagrees by a small amount
        repeats = [[await scorer.score(item, rollout)] for _ in range(3)]
        flat = [batch[0] for batch in repeats]
        assert len(set(flat)) > 1, f'jitter produced no variation: {flat}'
        # -> a positive repeatability band (acceptance gate genuinely exercised)
        assert measure_repeatability_band(repeats) > 0.0
        # reproducible: a fresh scorer reproduces the same sequence byte-for-byte
        scorer2 = SmokeReviewerScorer()
        flat2 = [await scorer2.score(item, rollout) for _ in range(3)]
        assert flat2 == flat
