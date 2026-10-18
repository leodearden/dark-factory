"""B2 (plans/review-all-s18-evals-measurement-prd.md §6): one refused AgentResult through every scoring site and every reducer above it."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from _eval_refusal import REFUSAL_COST_USD, answered, refused_agent_result
from shared.cli_invoke import AgentResult

from orchestrator.evals import compare, judge
from orchestrator.evals.compare import ASSESSOR_SCHEMA, ComparisonReport
from orchestrator.evals.elo import DEFAULT_RATING, TaskPool
from orchestrator.evals.measurement import (
    MeasurementStage,
    Unmeasured,
    UnmeasuredCause,
)
from orchestrator.evals.results import EvalResult
from orchestrator.evals.reviewer_trial import campaign, scorer
from orchestrator.evals.reviewer_trial.corpus import (
    CorpusDiff,
    CorpusManifest,
    GroundTruthIssue,
)
from orchestrator.evals.reviewer_trial.report import build_trial_report
from orchestrator.evals.reviewer_trial.runner import PanelRunResult
from orchestrator.evals.reviewer_trial.variants import ReviewerSpec, VariantConfig

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


async def _pairwise_judge(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(judge, 'invoke_agent', _refuse)
    verdict = await judge.run_judge(
        {'config_name': 'A', 'diff': 'diff A'}, {'config_name': 'B', 'diff': 'diff B'},
        {'id': 't', 'pre_task_commit': 'BASE'},
    )
    return verdict.unmeasured


_A_WINS_ASSESSMENT = {
    'winner': 'A', 'confidence': 0.8, 'summary': 's',
    'strengths_a': [], 'weaknesses_a': [], 'strengths_b': [], 'weaknesses_b': [],
    'circumstances': '',
}


async def _compare(
    monkeypatch: pytest.MonkeyPatch,
    task_ids: list[str],
    invoke: Callable[..., Awaitable[AgentResult]],
) -> ComparisonReport:
    monkeypatch.setattr(compare, 'invoke_agent', invoke)
    results = [
        EvalResult(
            task_id=task_id, config_name=config, outcome='done',
            metrics={'composite_score': score},
            worktree_path=str(Path('/nonexistent-6736') / task_id / config),
        )
        for task_id in task_ids
        for config, score in (('cfgA', 0.9), ('cfgB', 0.6))
    ]
    tasks = {task_id: {'id': task_id, 'name': f'Task {task_id}'} for task_id in task_ids}
    return await compare.compare_models(results, ['cfgA'], ['cfgB'], tasks)


async def _assessor(monkeypatch: pytest.MonkeyPatch) -> object:
    report = await _compare(monkeypatch, ['t1'], _refuse)
    return report.assessments[0].unmeasured


async def _synthesizer(monkeypatch: pytest.MonkeyPatch) -> object:
    async def assessor_answers(**kwargs: Any) -> AgentResult:
        if kwargs['output_schema'] is ASSESSOR_SCHEMA:
            return answered(_A_WINS_ASSESSMENT)
        return refused_agent_result()

    report = await _compare(monkeypatch, ['t1'], assessor_answers)
    return report.synthesis


# A new scoring site → add a ROW.
_SITES = [
    pytest.param(_matcher, MeasurementStage.MATCHER, id='matcher'),
    pytest.param(_plan_judge, MeasurementStage.JUDGE, id='plan-judge'),
    pytest.param(_pairwise_judge, MeasurementStage.PAIRWISE_JUDGE, id='pairwise-judge'),
    pytest.param(_assessor, MeasurementStage.ASSESSOR, id='assessor'),
    pytest.param(_synthesizer, MeasurementStage.SYNTHESIZER, id='synthesizer'),
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


async def test_build_trial_report_counts_the_refusal_and_keeps_every_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scorer, 'invoke_agent', _refuse)
    run, diff = _refused_panel_cell()
    refused = await scorer.score_panel_run(run, diff)
    other = CorpusDiff(
        diff_id='d2', language='python', source='synthetic', diff_text='diff',
        description='d', ground_truth=[],
    )
    corpus = CorpusManifest(diffs=[diff, other])
    variants = [
        VariantConfig(name=name, description=name, reviewers=[
            ReviewerSpec(name='r1', model='sonnet', specialization='Testing.'),
        ])
        for name in ('v1', 'v2')
    ]
    measured = [
        _measured_score(variant, d, f1)
        for variant, f1 in (('v1', 0.7), ('v2', 0.4)) for d in ('d1', 'd2')
    ]

    clean = build_trial_report(measured, variants, corpus)
    counted = build_trial_report([*measured, refused], variants, corpus)

    clean_v1 = next(s for s in clean.summaries if s.name == 'v1')
    counted_v1 = next(s for s in counted.summaries if s.name == 'v1')
    assert replace(counted_v1, unmeasured=0, unmeasured_cost_usd=0.0) == clean_v1
    assert counted_v1.unmeasured == 1


async def test_elo_tournament_counts_the_refusal_and_leaves_every_rating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contenders = [{'config_name': name, 'diff': f'diff {name}'} for name in ('A', 'B')]
    task = {'id': 't', 'pre_task_commit': 'BASE'}
    pool = TaskPool()

    async def a_wins(**_kwargs: Any) -> AgentResult:
        return answered({'winner': 'A', 'confidence': 0.9, 'reasoning': 'r'})

    monkeypatch.setattr(judge, 'invoke_agent', a_wins)
    await judge.run_elo_tournament(contenders, task, pool, max_rounds=2)
    prior = copy.deepcopy(pool.to_dict())
    assert pool.matches and set(pool.ratings.values()) != {DEFAULT_RATING}

    monkeypatch.setattr(judge, 'invoke_agent', _refuse)
    tally = await judge.run_elo_tournament(contenders, task, pool)

    assert pool.to_dict() == prior
    assert tally.unmeasured > 0
    assert tally.judged == 0


async def test_compare_models_counts_the_refusal_and_keeps_every_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def t2_answers(t2: AgentResult) -> Callable[..., Awaitable[AgentResult]]:
        async def invoke(**kwargs: Any) -> AgentResult:
            if kwargs['output_schema'] is not ASSESSOR_SCHEMA:
                return answered({
                    'overall_winner': 'A', 'executive_summary': '',
                    'circumstantial_analysis': '', 'cost_quality_tradeoff': '',
                    'workflow_recommendations': '',
                })
            if '**Task t2**' in str(kwargs['prompt']):
                return t2
            return answered(_A_WINS_ASSESSMENT)

        return invoke

    refused = await _compare(monkeypatch, ['t1', 't2'], t2_answers(refused_agent_result()))
    tied = await _compare(
        monkeypatch, ['t1', 't2'], t2_answers(answered(_A_WINS_ASSESSMENT | {'winner': 'tie'})),
    )
    absent = await _compare(monkeypatch, ['t1'], t2_answers(refused_agent_result()))

    assert refused.unmeasured == 1
    assert (refused.wins_a, refused.wins_b, refused.ties) == (absent.wins_a, absent.wins_b, absent.ties)
    assert (refused.avg_score_a, refused.avg_score_b) == (tied.avg_score_a, tied.avg_score_b)
