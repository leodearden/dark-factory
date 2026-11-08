"""The pairwise judge and the Elo tournament above it: an unmeasured verdict is never a tie."""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from _eval_refusal import answered, refused_agent_result
from shared.cli_invoke import AgentResult

from orchestrator.evals import elo, judge
from orchestrator.evals.elo import (
    DEFAULT_RATING,
    MAX_PER_PAIR,
    TaskPool,
    next_matchup,
    record_match,
)
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


# ── The Elo tournament: an unmeasured verdict moves no rating and no count ──

_TIE = answered({'winner': 'tie', 'confidence': 0.5, 'reasoning': 'even'})


def _scripted(first: list[AgentResult], then: AgentResult, calls: list[str]):
    queue = list(first)

    async def fake(**kwargs: Any) -> AgentResult:
        calls.append(str(kwargs['prompt']))
        return queue.pop(0) if queue else then

    return fake


def _contenders(*names: str) -> list[dict]:
    return [_contender(name) for name in names]


@pytest.mark.asyncio
async def test_every_refused_call_leaves_the_pool_untouched_and_is_bounded_per_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(judge, 'invoke_agent', _scripted([], refused_agent_result(), calls))
    pool = TaskPool()

    tally = await judge.run_elo_tournament(_contenders('A', 'B', 'C'), _TASK, pool, max_rounds=50)

    bound = elo.MAX_UNMEASURED_PER_PAIR
    assert tally == judge.TournamentTally(judged=0, unmeasured=bound * 3)
    assert set(pool.ratings.values()) == {DEFAULT_RATING}
    assert pool.matches == []
    assert pool.pair_counts == {}
    assert len(calls) == bound * 3


@pytest.mark.asyncio
async def test_an_unmeasured_verdict_does_not_count_toward_max_per_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(judge, 'invoke_agent', _scripted([refused_agent_result()], _TIE, calls))
    pool = TaskPool()

    tally = await judge.run_elo_tournament(_contenders('A', 'B'), _TASK, pool)

    assert tally == judge.TournamentTally(judged=MAX_PER_PAIR, unmeasured=1)
    assert pool.pair_counts == {'A|B': MAX_PER_PAIR}
    assert len(pool.matches) == MAX_PER_PAIR


@pytest.mark.asyncio
async def test_max_unmeasured_per_pair_is_a_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(judge, 'invoke_agent', _scripted([], refused_agent_result(), calls))

    await judge.run_elo_tournament(
        _contenders('A', 'B', 'C'), _TASK, TaskPool(), max_unmeasured_per_pair=1,
    )

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_unmeasured_calls_spend_the_round_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(judge, 'invoke_agent', _scripted([], refused_agent_result(), calls))

    tally = await judge.run_elo_tournament(_contenders('A', 'B', 'C'), _TASK, TaskPool(), max_rounds=2)

    assert len(calls) == 2
    assert tally.unmeasured == 2


@pytest.mark.asyncio
async def test_a_pair_with_a_missing_contender_is_skipped_without_a_fabricated_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(judge, 'invoke_agent', _scripted([], _TIE, calls))
    pool = TaskPool(ratings={'C': DEFAULT_RATING})

    tally = await judge.run_elo_tournament(_contenders('A', 'B'), _TASK, pool)

    assert tally.judged == MAX_PER_PAIR
    assert pool.pair_counts == {'A|B': MAX_PER_PAIR}
    assert 'A|C' not in pool.pair_counts and 'B|C' not in pool.pair_counts


def test_next_matchup_skips_the_given_pairs() -> None:
    pool = TaskPool(ratings={'A': 1500.0, 'B': 1510.0, 'C': 1600.0})

    assert next_matchup(pool) == ('A', 'B')
    assert next_matchup(pool, skip={'A|B'}) == ('B', 'C')
    assert next_matchup(pool, skip={'A|B', 'A|C', 'B|C'}) is None


def test_record_match_rejects_an_unknown_winner() -> None:
    pool = TaskPool(ratings={'A': 1500.0, 'B': 1500.0})
    before = copy.deepcopy(pool.to_dict())

    with pytest.raises(ValueError) as rejected:
        record_match(pool, 'A', 'B', 'C', 0.5, 'r')

    assert "'C'" in str(rejected.value)
    assert all(winner in str(rejected.value) for winner in ('A', 'B', 'tie'))
    assert pool.to_dict() == before


def test_max_unmeasured_per_pair_defaults_to_two() -> None:
    assert elo.MAX_UNMEASURED_PER_PAIR == 2
