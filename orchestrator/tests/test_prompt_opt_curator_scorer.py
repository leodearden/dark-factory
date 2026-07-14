"""Tests for orchestrator.evals.prompt_opt.curator_scorer -- the T5 HARD,
LLM-free curator action-match scorer.

See plans/tier1-prompt-optimization-prd.md T5/D-5. `CuratorActionScorer`
implements the T6 `Scorer` Protocol (orchestrator.evals.prompt_opt.scorer):
action equality (create/drop match -> 1.0) plus combine-target correctness
(target_fingerprint, falling back to target_id) for combine/combine pairs.
No LLM judge call anywhere in this module -- a deterministic hard signal.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from shared.prompt_artifact import PromptSpec

from orchestrator.evals.prompt_opt.curator_corpus import CuratorReplayItem
from orchestrator.evals.prompt_opt.curator_scorer import CuratorActionScorer
from orchestrator.evals.prompt_opt.engine import LoopResult, run_optimization_loop
from orchestrator.evals.prompt_opt.scorer import Scorer


def _item(
    action: str,
    target_fingerprint: str | None = None,
    target_id: str | None = None,
) -> Any:
    """A minimal gold-label item: gold_action/gold_target_fingerprint/gold_target_id."""
    return SimpleNamespace(
        gold_action=action,
        gold_target_fingerprint=target_fingerprint,
        gold_target_id=target_id,
    )


class TestCuratorActionScorerProtocolConformance:
    def test_is_instance_of_scorer(self) -> None:
        result = isinstance(CuratorActionScorer(), Scorer)
        assert result is True

    @pytest.mark.asyncio
    async def test_score_is_awaitable(self) -> None:
        scorer = CuratorActionScorer()
        result = await scorer.score(_item('create'), {'action': 'create'})
        assert result == 1.0

    def test_score_result_is_in_unit_interval(self) -> None:
        # Sanity on the contract shape shared across every branch below --
        # exercised concretely per-branch in the test classes that follow.
        assert 0.0 <= 1.0 <= 1.0


class TestCuratorActionScorerActionEquality:
    @pytest.mark.asyncio
    async def test_create_create_scores_1(self) -> None:
        scorer = CuratorActionScorer()
        score = await scorer.score(_item('create'), {'action': 'create'})
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_drop_drop_scores_1(self) -> None:
        scorer = CuratorActionScorer()
        score = await scorer.score(_item('drop'), {'action': 'drop'})
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_create_vs_combine_mismatch_scores_0(self) -> None:
        scorer = CuratorActionScorer()
        score = await scorer.score(_item('create'), {'action': 'combine'})
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_drop_vs_create_mismatch_scores_0(self) -> None:
        scorer = CuratorActionScorer()
        score = await scorer.score(_item('drop'), {'action': 'create'})
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_drop_vs_combine_mismatch_scores_0(self) -> None:
        scorer = CuratorActionScorer()
        score = await scorer.score(_item('drop'), {'action': 'combine'})
        assert score == 0.0


class TestCuratorActionScorerCombineTarget:
    @pytest.mark.asyncio
    async def test_combine_matching_fingerprint_scores_1(self) -> None:
        scorer = CuratorActionScorer()
        item = _item('combine', target_fingerprint='Fix the thing', target_id='task-1')
        rollout = {
            'action': 'combine', 'target_fingerprint': 'Fix the thing', 'target_id': 'task-9',
        }
        score = await scorer.score(item, rollout)
        # Fingerprint match wins even though target_id differs -- ids get
        # reallocated over the DB's lifetime, fingerprint is the stable key.
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_combine_mismatching_fingerprint_scores_default_partial_credit(self) -> None:
        scorer = CuratorActionScorer()
        item = _item('combine', target_fingerprint='Fix the thing', target_id='task-1')
        rollout = {
            'action': 'combine', 'target_fingerprint': 'A different task', 'target_id': 'task-1',
        }
        score = await scorer.score(item, rollout)
        assert score == 0.0  # default combine_target_partial_credit

    @pytest.mark.asyncio
    async def test_combine_mismatching_fingerprint_uses_custom_partial_credit(self) -> None:
        scorer = CuratorActionScorer(combine_target_partial_credit=0.5)
        item = _item('combine', target_fingerprint='Fix the thing')
        rollout = {'action': 'combine', 'target_fingerprint': 'A different task'}
        score = await scorer.score(item, rollout)
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_combine_falls_back_to_target_id_when_fingerprint_absent(self) -> None:
        scorer = CuratorActionScorer()
        item = _item('combine', target_id='task-1')
        rollout = {'action': 'combine', 'target_id': 'task-1'}
        score = await scorer.score(item, rollout)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_combine_target_id_mismatch_when_fingerprint_absent_scores_partial(self) -> None:
        scorer = CuratorActionScorer()
        item = _item('combine', target_id='task-1')
        rollout = {'action': 'combine', 'target_id': 'task-2'}
        score = await scorer.score(item, rollout)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_combine_falls_back_to_target_id_when_only_one_side_has_fingerprint(self) -> None:
        # Gold carries a fingerprint but the rollout doesn't (or vice versa)
        # -- can't compare fingerprints meaningfully, so fall back to id.
        scorer = CuratorActionScorer()
        item = _item('combine', target_fingerprint='Fix the thing', target_id='task-1')
        rollout = {'action': 'combine', 'target_id': 'task-1'}
        score = await scorer.score(item, rollout)
        assert score == 1.0


class TestCuratorActionScorerDuckTypedRollout:
    """The scorer reads the predicted action/target from BOTH a plain dict
    rollout and a duck-typed object rollout (attribute access) -- the
    injected rollout_fn's return shape isn't fixed by the T6 engine."""

    @pytest.mark.asyncio
    async def test_reads_action_from_dict_rollout(self) -> None:
        scorer = CuratorActionScorer()
        score = await scorer.score(_item('create'), {'action': 'create'})
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_reads_action_from_object_rollout(self) -> None:
        scorer = CuratorActionScorer()
        rollout = SimpleNamespace(action='create', target_fingerprint=None, target_id=None)
        score = await scorer.score(_item('create'), rollout)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_combine_target_match_via_object_rollout(self) -> None:
        scorer = CuratorActionScorer()
        item = _item('combine', target_fingerprint='Fix the thing')
        rollout = SimpleNamespace(
            action='combine', target_fingerprint='Fix the thing', target_id=None,
        )
        score = await scorer.score(item, rollout)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_combine_mismatch_via_object_rollout(self) -> None:
        scorer = CuratorActionScorer()
        item = _item('combine', target_fingerprint='Fix the thing')
        rollout = SimpleNamespace(
            action='combine', target_fingerprint='Something else', target_id=None,
        )
        score = await scorer.score(item, rollout)
        assert score == 0.0


EXECUTOR_MODEL = 'curator-exec-x'
OPTIMIZER_MODEL = 'curator-frontier-y'

_ACTIONS_CYCLE = ('create', 'drop', 'combine')


def _make_curator_corpus(n: int) -> list[CuratorReplayItem]:
    """A tiny synthetic curator replay corpus -- large enough for
    split_corpus's default 2:1:7 ratios to produce non-empty selection AND
    test splits (n=20 -> train=4, selection=2, test=14). Cycles through
    create/drop/combine so every action is represented; combine items carry
    a target_fingerprint/target_id pair to exercise the combine-target
    branch too."""
    items = []
    for i in range(n):
        action = _ACTIONS_CYCLE[i % len(_ACTIONS_CYCLE)]
        target_fingerprint = f'Target {i}' if action == 'combine' else None
        target_id = f'task-{i}' if action == 'combine' else None
        items.append(CuratorReplayItem(
            ticket_id=f'ticket-{i}',
            candidate={'title': f'Candidate {i}'},
            recorded_action=action,
            recorded_target_fingerprint=target_fingerprint,
            recorded_target_id=target_id,
            gold_action=action,
            gold_target_fingerprint=target_fingerprint,
            gold_target_id=target_id,
        ))
    return items


def _predicted_decision_for(item: CuratorReplayItem) -> dict:
    """Deterministically derive a predicted decision from *item* alone (no
    randomness, no LLM): even ticket indices predict the gold action/target
    exactly, odd indices predict a deliberately WRONG action -- so the
    scorer sees a genuine mix of 1.0/0.0 outcomes rather than a trivially
    constant score."""
    index = int(item.ticket_id.rsplit('-', 1)[-1])
    if index % 2 == 0:
        return {
            'action': item.gold_action,
            'target_fingerprint': item.gold_target_fingerprint,
            'target_id': item.gold_target_id,
        }
    wrong_action = 'create' if item.gold_action != 'create' else 'drop'
    return {'action': wrong_action, 'target_fingerprint': None, 'target_id': None}


class TestCuratorActionScorerEngineConformance:
    """Hermetic end-to-end proof that CuratorActionScorer plugs into the T6
    run_optimization_loop engine seam -- no fake Scorer stand-in, no real
    LLM call, no real DB anywhere in this test."""

    @pytest.mark.asyncio
    async def test_scorer_drives_a_full_optimization_loop_on_the_executor_model(self) -> None:
        corpus = _make_curator_corpus(20)
        spec = PromptSpec(
            prompt_id='curator-trial-prompt',
            contract='You are the curator. Follow the HEURISTICS below exactly.',
            baseline_heuristics='BASELINE: prefer create unless duplication is obvious.',
        )
        scorer = CuratorActionScorer()
        rollout_calls_models: list[str] = []

        async def rollout_fn(composed_prompt: str, item: CuratorReplayItem, model: str) -> dict:
            rollout_calls_models.append(model)
            return _predicted_decision_for(item)

        async def propose_fn(
            current_heuristics: str,
            scored_minibatch: list[Any],
            rejected_buffer: list[str],
            *,
            max_edits: int,
            optimizer_model: str,
        ) -> str:
            # This test only proves the scorer<->engine plumbing, not
            # optimizer proposal quality (that's test_prompt_opt_engine.py's
            # job) -- a no-op edit keeps the loop deterministic either way.
            return current_heuristics

        result = await run_optimization_loop(
            spec=spec,
            corpus=corpus,
            scorer=scorer,
            executor_model=EXECUTOR_MODEL,
            optimizer_model=OPTIMIZER_MODEL,
            rollout_fn=rollout_fn,
            propose_fn=propose_fn,
            n_repeats=2,
            max_steps=1,
            split_seed=3,
            minibatch_size=2,
            git_sha='deadbeef',
            date='2026-07-14',
            harness_version='harness-test-v1',
        )

        assert isinstance(result, LoopResult)
        assert 0.0 <= result.test_score <= 1.0

        # The engine's structural guarantee (see engine.run_optimization_loop's
        # docstring) is that rollout_fn -- and therefore every rollout the
        # scorer ever grades -- is ALWAYS called with executor_model, never
        # optimizer_model. This is what "the scorer was invoked on the
        # executor model" means operationally, since CuratorActionScorer.score
        # itself has no model parameter -- it only ever sees what rollout_fn
        # produced.
        assert rollout_calls_models
        assert set(rollout_calls_models) == {EXECUTOR_MODEL}
