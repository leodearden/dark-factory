"""The assessor and synthesizer measure or say they did not; compare_models counts the difference."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from _eval_refusal import answered
from shared.cli_invoke import AgentResult

from orchestrator.evals import compare
from orchestrator.evals.compare import (
    ASSESSOR_SCHEMA,
    SYNTHESIZER_SCHEMA,
    ComparisonReport,
    Synthesis,
    TaskAssessment,
    compare_models,
    format_comparison_markdown,
)
from orchestrator.evals.measurement import (
    MeasurementStage,
    Unmeasured,
    UnmeasuredCause,
)
from orchestrator.evals.results import EvalResult

_SYNTHESIS = {
    'overall_winner': 'A',
    'executive_summary': 'EXECUTIVE_TEXT',
    'circumstantial_analysis': 'CIRCUMSTANCES_TEXT',
    'cost_quality_tradeoff': 'TRADEOFF_TEXT',
    'workflow_recommendations': 'WORKFLOW_TEXT',
}
_NOT_JSON = AgentResult(success=True, output='not json', cost_usd=0.4)


def _assessment(winner: str = 'A', **overrides: Any) -> dict[str, Any]:
    payload = {
        'winner': winner, 'confidence': 0.8, 'summary': f'SUMMARY_{winner}',
        'strengths_a': [], 'weaknesses_a': [], 'strengths_b': [], 'weaknesses_b': [],
        'circumstances': '',
    }
    return payload | overrides


class _ScriptedCompare:
    """An assessor answer per task id, one synthesizer answer, and the synthesizer prompts seen."""

    def __init__(
        self, assessments: dict[str, AgentResult], synthesis: AgentResult | None = None,
    ) -> None:
        self.assessments = assessments
        self.synthesis = synthesis or answered(_SYNTHESIS)
        self.synthesizer_prompts: list[str] = []

    async def __call__(self, **kwargs: Any) -> AgentResult:
        prompt = str(kwargs['prompt'])
        if kwargs['output_schema'] is SYNTHESIZER_SCHEMA:
            self.synthesizer_prompts.append(prompt)
            return self.synthesis
        assert kwargs['output_schema'] is ASSESSOR_SCHEMA
        return next(
            answer for task_id, answer in self.assessments.items()
            if f'**Task {task_id}**' in prompt
        )


def _results(tmp_path: Path, task_ids: list[str]) -> list[EvalResult]:
    return [
        EvalResult(
            task_id=task_id, config_name=config, outcome='done',
            metrics={'composite_score': score},
            worktree_path=str(tmp_path / 'absent' / f'{task_id}-{config}'),
        )
        for task_id in task_ids
        for config, score in (('cfgA', 0.9), ('cfgB', 0.6))
    ]


async def _compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: _ScriptedCompare,
) -> ComparisonReport:
    monkeypatch.setattr(compare, 'invoke_agent', script)
    task_ids = sorted(script.assessments)
    tasks = {
        task_id: {'id': task_id, 'name': f'Task {task_id}', 'pre_task_commit': 'BASE'}
        for task_id in task_ids
    }
    return await compare_models(_results(tmp_path, task_ids), ['cfgA'], ['cfgB'], tasks)


def _assessment_for(report: ComparisonReport, task_id: str) -> TaskAssessment:
    return next(a for a in report.assessments if a.task_id == task_id)


@pytest.mark.asyncio
async def test_an_unparseable_assessment_is_unmeasured_and_counted_beside_wins_and_ties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': answered(_assessment('A')), 't2': _NOT_JSON})

    report = await _compare(tmp_path, monkeypatch, script)

    assert [a.task_id for a in report.assessments] == ['t1', 't2']
    refused = _assessment_for(report, 't2')
    assert isinstance(refused.unmeasured, Unmeasured)
    assert refused.unmeasured.stage is MeasurementStage.ASSESSOR
    assert refused.unmeasured.cause is UnmeasuredCause.UNPARSEABLE
    assert refused.unmeasured.cost_usd == 0.4
    assert refused.winner is None
    assert (report.wins_a, report.wins_b, report.ties, report.unmeasured) == (1, 0, 0, 1)


@pytest.mark.parametrize('payload', [
    {k: v for k, v in _assessment('A').items() if k != 'winner'},
    _assessment('C'),
    {k: v for k, v in _assessment('A').items() if k != 'confidence'},
], ids=['no-winner', 'unknown-winner', 'no-confidence'])
@pytest.mark.asyncio
async def test_an_assessment_without_a_valid_winner_is_unparseable(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': answered(payload)})

    report = await _compare(tmp_path, monkeypatch, script)

    [assessment] = report.assessments
    assert assessment.unmeasured is not None
    assert assessment.unmeasured.cause is UnmeasuredCause.UNPARSEABLE
    assert assessment.winner is None
    assert (report.wins_a, report.wins_b, report.ties, report.unmeasured) == (0, 0, 0, 1)


@pytest.mark.asyncio
async def test_the_synthesizer_sees_only_measured_assessments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': answered(_assessment('A')), 't2': _NOT_JSON})

    await _compare(tmp_path, monkeypatch, script)

    [prompt] = script.synthesizer_prompts
    assert '"task_id": "t1"' in prompt
    assert '"task_id": "t2"' not in prompt
    assert 'Wins: A=1, B=0, Ties=0' in prompt


@pytest.mark.asyncio
async def test_an_unparseable_synthesis_is_unmeasured_not_an_overall_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': answered(_assessment('A'))}, synthesis=_NOT_JSON)

    report = await _compare(tmp_path, monkeypatch, script)

    assert isinstance(report.synthesis, Unmeasured)
    assert report.synthesis.stage is MeasurementStage.SYNTHESIZER
    assert report.synthesis.cause is UnmeasuredCause.UNPARSEABLE


@pytest.mark.asyncio
async def test_a_measured_synthesis_is_carried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': answered(_assessment('A'))})

    report = await _compare(tmp_path, monkeypatch, script)

    assert report.synthesis == Synthesis(**_SYNTHESIS)


@pytest.mark.asyncio
async def test_no_measured_assessment_means_no_synthesis_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': _NOT_JSON, 't2': _NOT_JSON})

    report = await _compare(tmp_path, monkeypatch, script)

    assert script.synthesizer_prompts == []
    assert report.synthesis is None
    assert report.unmeasured == 2


def test_a_measured_assessment_must_name_its_winner_and_an_unmeasured_one_must_not() -> None:
    unmeasured = Unmeasured(
        MeasurementStage.ASSESSOR, UnmeasuredCause.CAP_HIT, 'r', cost_usd=0.1,
    )

    with pytest.raises(ValueError) as winner_and_unmeasured:
        TaskAssessment('t', 't', winner='A', confidence=0.0, summary='', unmeasured=unmeasured)
    with pytest.raises(ValueError) as neither:
        TaskAssessment('t', 't', winner=None, confidence=0.0, summary='', unmeasured=None)

    assert "winner='A'" in str(winner_and_unmeasured.value)
    assert 'cap_hit' in str(winner_and_unmeasured.value)
    assert 'winner=None' in str(neither.value)
    assert 'unmeasured=None' in str(neither.value)
    measured: Any = TaskAssessment('t', 't', winner='tie', confidence=0.5, summary='s')
    with pytest.raises(dataclasses.FrozenInstanceError):
        measured.winner = 'A'


def _heading(markdown: str, task_id: str) -> str:
    return next(line for line in markdown.splitlines() if line.startswith(f'### {task_id}:'))


def _overview_row(markdown: str, label: str) -> list[str]:
    return next(
        line.split() for line in markdown.splitlines() if line.strip().startswith(label)
    )


@pytest.mark.asyncio
async def test_markdown_counts_and_heads_an_unmeasured_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': answered(_assessment('A')), 't2': _NOT_JSON})

    markdown = format_comparison_markdown(await _compare(tmp_path, monkeypatch, script))

    assert _overview_row(markdown, 'Unmeasured') == ['Unmeasured', '1']
    heading = _heading(markdown, 't2')
    assert 'UNMEASURED' in heading
    assert 'assessor:unparseable' in heading
    assert 'Winner:' not in heading
    assert 'Winner:' in _heading(markdown, 't1')


@pytest.mark.asyncio
async def test_markdown_states_an_unmeasured_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': answered(_assessment('A'))}, synthesis=_NOT_JSON)

    markdown = format_comparison_markdown(await _compare(tmp_path, monkeypatch, script))

    section = markdown.split('## Synthesis', 1)[1]
    assert 'UNMEASURED' in section
    assert 'synthesizer:unparseable' in section
    assert '## Executive Summary' not in markdown


@pytest.mark.asyncio
async def test_markdown_renders_a_measured_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _ScriptedCompare({'t1': answered(_assessment('A'))})

    markdown = format_comparison_markdown(await _compare(tmp_path, monkeypatch, script))

    for heading, text in (
        ('## Executive Summary', 'EXECUTIVE_TEXT'),
        ('## Circumstantial Analysis', 'CIRCUMSTANCES_TEXT'),
        ('## Cost / Quality Tradeoff', 'TRADEOFF_TEXT'),
        ('## Workflow Recommendations', 'WORKFLOW_TEXT'),
    ):
        assert text in markdown.split(heading, 1)[1]
    assert _overview_row(markdown, 'Unmeasured') == ['Unmeasured', '0']
