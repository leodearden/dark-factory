"""The pairwise judge and the Elo tournament above it: an unmeasured verdict is never a tie."""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from _eval_refusal import answered
from shared.cli_invoke import AgentResult

from orchestrator.evals import judge
from orchestrator.evals.judge import JudgeVerdict, run_judge
from orchestrator.evals.measurement import (
    MeasurementStage,
    Unmeasured,
    UnmeasuredCause,
)

_TASK = {'id': 't', 'name': 't', 'pre_task_commit': 'BASE'}


def _contender(name: str, diff: str | None = None) -> dict:
    return {'config_name': name, 'diff': diff or f'diff {name}', 'worktree_path': ''}


def _returning(result: AgentResult) -> Callable[..., Awaitable[AgentResult]]:
    async def fake(**_kwargs: Any) -> AgentResult:
        return result

    return fake


@pytest.mark.asyncio
async def test_an_unparseable_answer_is_unmeasured_never_a_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        judge, 'invoke_agent', _returning(AgentResult(success=True, output='not json', cost_usd=0.5)),
    )

    verdict = await run_judge(_contender('A'), _contender('B'), _TASK)

    assert isinstance(verdict.unmeasured, Unmeasured)
    assert verdict.unmeasured.stage is MeasurementStage.PAIRWISE_JUDGE
    assert verdict.unmeasured.cause is UnmeasuredCause.UNPARSEABLE
    assert verdict.unmeasured.cost_usd == 0.5
    assert verdict.winner is None


@pytest.mark.parametrize('output', [
    json.dumps({'confidence': 0.9, 'reasoning': 'r'}),
    json.dumps({'winner': 'C', 'confidence': 0.9, 'reasoning': 'r'}),
    json.dumps({'winner': 'A', 'reasoning': 'r'}),
    json.dumps({'winner': 'A', 'confidence': 'high', 'reasoning': 'r'}),
    '{"winner": "A", "confidence": NaN, "reasoning": "r"}',
], ids=['no-winner', 'unknown-winner', 'no-confidence', 'word-confidence', 'nan-confidence'])
@pytest.mark.asyncio
async def test_an_answer_without_a_valid_verdict_is_unparseable(
    output: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        judge, 'invoke_agent', _returning(AgentResult(success=True, output=output, cost_usd=0.2)),
    )

    verdict = await run_judge(_contender('A'), _contender('B'), _TASK)

    assert verdict.unmeasured is not None
    assert verdict.unmeasured.cause is UnmeasuredCause.UNPARSEABLE
    assert verdict.winner is None


@pytest.mark.asyncio
async def test_the_winner_is_reported_for_the_config_whose_diff_won(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orders: set[str] = set()

    async def picks_config_a(**kwargs: Any) -> AgentResult:
        prompt = str(kwargs['prompt'])
        section_a = prompt.split("## Agent A's implementation")[1].split("## Agent B's implementation")[0]
        shown_first = 'A' if 'DIFF_OF_A' in section_a else 'B'
        orders.add(shown_first)
        return answered({'winner': shown_first, 'confidence': 0.8, 'reasoning': 'better'})

    monkeypatch.setattr(judge, 'invoke_agent', picks_config_a)
    random.seed(6736)

    verdicts = [
        await run_judge(_contender('A', 'DIFF_OF_A'), _contender('B', 'DIFF_OF_B'), _TASK)
        for _ in range(20)
    ]

    assert orders == {'A', 'B'}, 'both prompt orders must be exercised'
    for verdict in verdicts:
        assert verdict.winner == 'A'
        assert verdict.confidence == 0.8
        assert verdict.reasoning == 'better'
        assert verdict.unmeasured is None


def test_a_measured_verdict_must_name_its_winner_and_an_unmeasured_one_must_not() -> None:
    unmeasured = Unmeasured(
        MeasurementStage.PAIRWISE_JUDGE, UnmeasuredCause.CAP_HIT, 'r', cost_usd=0.1,
    )

    with pytest.raises(ValueError) as winner_and_unmeasured:
        JudgeVerdict('t', 'A', 'B', winner='A', confidence=0.0, reasoning='', unmeasured=unmeasured)
    with pytest.raises(ValueError) as neither:
        JudgeVerdict('t', 'A', 'B', winner=None, confidence=0.0, reasoning='', unmeasured=None)

    assert "winner='A'" in str(winner_and_unmeasured.value)
    assert 'unmeasured=Unmeasured(' in str(winner_and_unmeasured.value)
    assert 'winner=None' in str(neither.value)
    assert 'unmeasured=None' in str(neither.value)
    verdict: Any = JudgeVerdict('t', 'A', 'B', winner='tie', confidence=0.5, reasoning='r')
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.winner = 'A'
