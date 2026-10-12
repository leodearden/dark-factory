"""B2 (plans/review-all-s18-evals-measurement-prd.md §6): one refused AgentResult through every scoring site and every reducer above it."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from typing import Any

import pytest
from _eval_refusal import REFUSAL_COST_USD, refused_agent_result
from shared.cli_invoke import AgentResult

from orchestrator.evals import judge
from orchestrator.evals.measurement import (
    MeasurementStage,
    Unmeasured,
    UnmeasuredCause,
)
from orchestrator.evals.reviewer_trial import campaign, scorer
from orchestrator.evals.reviewer_trial.corpus import CorpusDiff, GroundTruthIssue
from orchestrator.evals.reviewer_trial.runner import PanelRunResult

pytestmark = pytest.mark.asyncio

_REVIEWER_ISSUE = {'location': 'a.py:1', 'category': 'bug', 'description': 'A bug'}
_GROUND_TRUTH = GroundTruthIssue(
    id='gt1', location='a.py:1', category='bug', severity='blocking',
    description='A bug', mutation_type='logic',
)


async def _refuse(**_kwargs: Any) -> AgentResult:
    return refused_agent_result()


def _refused_panel_cell() -> tuple[PanelRunResult, CorpusDiff]:
    run = PanelRunResult(
        variant_name='v1', diff_id='d1', total_cost_usd=2.0, wall_clock_ms=300,
        reviews={
            'r1': {'verdict': 'NEEDS_CHANGES', 'issues': [_REVIEWER_ISSUE]},
            'r2': {'verdict': 'ERROR', 'issues': []},
        },
    )
    diff = CorpusDiff(
        diff_id='d1', language='python', source='synthetic', diff_text='diff',
        description='d', ground_truth=[_GROUND_TRUTH],
    )
    return run, diff


async def _matcher(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(scorer, 'invoke_agent', _refuse)
    return await scorer.match_issues([_REVIEWER_ISSUE], [_GROUND_TRUTH], 'diff')


async def _plan_judge(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(judge, 'invoke_agent', _refuse)
    plan = {'steps': [{'id': 'step-1', 'type': 'test'}]}
    return await judge.judge_plan_quality(plan, 'diff', {'id': 't'})


# A new scoring site → add a ROW.
_SITES = [
    pytest.param(_matcher, MeasurementStage.MATCHER, id='matcher'),
    pytest.param(_plan_judge, MeasurementStage.JUDGE, id='plan-judge'),
]


@pytest.mark.parametrize(('drive', 'stage'), _SITES)
async def test_every_scoring_site_answers_the_refusal_unmeasured(
    drive: Callable[[pytest.MonkeyPatch], Awaitable[object]],
    stage: MeasurementStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unmeasured = await drive(monkeypatch)

    assert isinstance(unmeasured, Unmeasured)
    assert unmeasured.stage is stage
    assert unmeasured.cause is UnmeasuredCause.CAP_HIT
    assert unmeasured.cost_usd == REFUSAL_COST_USD


async def test_score_panel_run_carries_the_matchers_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scorer, 'invoke_agent', _refuse)
    run, diff = _refused_panel_cell()

    score = await scorer.score_panel_run(run, diff)

    assert isinstance(score.unmeasured, Unmeasured)
    assert score.unmeasured.stage is MeasurementStage.MATCHER
    assert score.unmeasured.cause is UnmeasuredCause.CAP_HIT
    assert score.matches == []
    assert score.false_positives == []
    assert score.unmatched_gt == []
    assert score.cost_usd == run.total_cost_usd
    assert score.match_cost_usd == REFUSAL_COST_USD


def _measured_score(variant: str, diff: str, f1: float) -> scorer.ScoringResult:
    return scorer.ScoringResult(
        variant_name=variant, diff_id=diff, recall=f1, precision=f1, f1=f1,
        blocking_recall=f1, cost_usd=1.0, match_cost_usd=0.05, wall_clock_ms=100,
    )


async def test_summarize_campaign_counts_the_refusal_and_keeps_every_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scorer, 'invoke_agent', _refuse)
    run, diff = _refused_panel_cell()
    refused = asdict(await scorer.score_panel_run(run, diff)) | {'trial': 1}
    measured = [
        asdict(_measured_score(variant, d, f1)) | {'trial': t}
        for variant, f1 in (('inc', 0.5), ('v1', 0.7))
        for d in ('d1', 'd2') for t in (1, 2, 3)
    ]
    blocking = {'d1': {'gt1'}, 'd2': set()}

    without_refusal, _ = campaign.summarize_campaign(measured, 'inc', blocking)
    with_refusal, _ = campaign.summarize_campaign([*measured, refused], 'inc', blocking)

    for counted, clean in zip(with_refusal, without_refusal, strict=True):
        assert replace(counted, unmeasured=0, unmeasured_cost_usd=0.0) == clean
    assert next(a for a in with_refusal if a.name == 'v1').unmeasured == 1
