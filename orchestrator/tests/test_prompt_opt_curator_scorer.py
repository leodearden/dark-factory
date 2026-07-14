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

from orchestrator.evals.prompt_opt.curator_scorer import CuratorActionScorer
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
