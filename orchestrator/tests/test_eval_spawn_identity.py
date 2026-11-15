"""Eval-harness LLM calls declare their fleet identity (task 6211).

The judges and compare assessors run with ``cwd=/tmp``, so a session that
declares no ``CLAUDE_SPAWN_PROJECT`` is filed by the SessionStart hook under
the session registry's ``tmp`` bucket. ``measured_call`` is the one seam every
eval call passes through; it declares role ``eval_<stage>`` and project
``dark_factory`` itself.

These tests read only what each dispatch was handed, so a call is answered
with a refusal wherever its answer does not gate a later dispatch.

Hermetic: no LLM, no network.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _eval_probe_fixture import a_wins_assessment, eval_result, fixture_task
from _eval_refusal import answered, refused_agent_result
from _orch_helpers import build_usage_gate
from shared.cli_invoke import AgentResult, apply_spawn_env
from shared.config_models import AccountConfig

from orchestrator.config import OrchestratorConfig
from orchestrator.evals.compare import compare_models
from orchestrator.evals.judge import judge_plan_quality, run_judge
from orchestrator.evals.measurement import (
    MeasurementStage,
    Scored,
    json_answer,
    measured_call,
)
from orchestrator.session_registry import parse_spawn_identity

_TMP = Path('/tmp')


def _eval_spawn_env_for(stage: MeasurementStage) -> dict[str, str]:
    return {'CLAUDE_SPAWN_ROLE': f'eval_{stage.value}', 'CLAUDE_SPAWN_PROJECT': 'dark_factory'}


def _recording_invoke(
    *results: AgentResult,
) -> tuple[Callable[..., Awaitable[AgentResult]], list[dict[str, Any]]]:
    """An invoke double answering *results* in order and recording its kwargs."""
    pending = list(results)
    calls: list[dict[str, Any]] = []

    async def invoke(**kwargs: Any) -> AgentResult:
        calls.append(kwargs)
        return pending.pop(0)

    return invoke, calls


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', [
    MeasurementStage.JUDGE,
    MeasurementStage.PAIRWISE_JUDGE,
    MeasurementStage.ASSESSOR,
    MeasurementStage.SYNTHESIZER,
    MeasurementStage.MATCHER,
])
async def test_measured_call_declares_the_eval_identity(stage: MeasurementStage) -> None:
    invoke, calls = _recording_invoke(answered({'x': 1}))

    outcome = await measured_call(invoke, json_answer, stage=stage, backend='claude', prompt='P')

    assert isinstance(outcome, Scored)
    assert len(calls) == 1
    assert calls[0]['spawn_env'] == _eval_spawn_env_for(stage)


@pytest.mark.asyncio
async def test_eval_calls_file_under_the_operational_project_id() -> None:
    """The project TaskWorkflow agents file under: the operational yaml's, loaded by the autouse config fixture."""
    invoke, calls = _recording_invoke(answered({'x': 1}))

    await measured_call(invoke, json_answer, stage=MeasurementStage.JUDGE, backend='claude', prompt='P')

    assert calls[0]['spawn_env']['CLAUDE_SPAWN_PROJECT'] == OrchestratorConfig().fused_memory.project_id


@pytest.mark.asyncio
async def test_a_caller_supplied_spawn_env_is_refused_before_dispatch() -> None:
    invoke, calls = _recording_invoke(answered({'x': 1}))

    with pytest.raises(TypeError, match='measured_call declares the eval fleet identity itself'):
        await measured_call(
            invoke, json_answer, stage=MeasurementStage.JUDGE, backend='claude', prompt='P',
            spawn_env={'CLAUDE_SPAWN_ROLE': 'caller'},
        )

    assert calls == []


@pytest.mark.asyncio
async def test_gated_measured_call_declares_the_identity_on_every_attempt() -> None:
    gate = build_usage_gate(
        [
            AccountConfig(name='max-a', oauth_token_env='CLAUDE_OAUTH_A'),
            AccountConfig(name='max-b', oauth_token_env='CLAUDE_OAUTH_B'),
        ],
        ['tok-a', 'tok-b'],
    )
    invoke, calls = _recording_invoke(refused_agent_result(), answered({'x': 2}))

    with patch('shared.cli_invoke.asyncio.sleep', AsyncMock()):
        outcome = await measured_call(
            invoke, json_answer, stage=MeasurementStage.JUDGE,
            backend='claude', usage_gate=gate, prompt='P',
        )

    assert isinstance(outcome, Scored)
    assert len(calls) == 2
    assert calls[1]['oauth_token'] == 'tok-b'
    for call in calls:
        assert call['spawn_env'] == _eval_spawn_env_for(MeasurementStage.JUDGE)


# ---------------------------------------------------------------------------
# The four /tmp-cwd eval spawn sites, resolved through the SessionStart hook's
# own identity cascade.
# ---------------------------------------------------------------------------

async def _pairwise_judge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    invoke, calls = _recording_invoke(refused_agent_result())
    monkeypatch.setattr('orchestrator.evals.judge.invoke_agent', invoke)
    await run_judge({'diff': '+a', 'config_name': 'A'}, {'diff': '+b', 'config_name': 'B'}, fixture_task())
    return calls


async def _plan_judge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    invoke, calls = _recording_invoke(refused_agent_result())
    monkeypatch.setattr('orchestrator.evals.judge.invoke_agent', invoke)
    plan = {'steps': [{'id': 'step-1', 'type': 'test', 'description': 'write the test'}]}
    await judge_plan_quality(plan, '+landed', fixture_task())
    return calls


async def _compare_models_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> list[dict[str, Any]]:
    """One fixture compared once: the assessor dispatch, then the synthesizer's.

    The assessment must measure, because only a measured assessment reaches
    the synthesizer.
    """
    invoke, calls = _recording_invoke(answered(a_wins_assessment()), refused_agent_result())
    monkeypatch.setattr('orchestrator.evals.compare.invoke_agent', invoke)
    monkeypatch.setattr('orchestrator.evals.compare.get_diff', AsyncMock(return_value='+x'))
    task = fixture_task()
    await compare_models(
        [eval_result('A', tmp_path / 'wt_a'), eval_result('B', tmp_path / 'wt_b')],
        ['A'], ['B'], {task['id']: task},
    )
    assert len(calls) == 2, 'expected one assessor and one synthesizer dispatch'
    return calls


async def _assessor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    return (await _compare_models_calls(monkeypatch, tmp_path))[:1]


async def _synthesizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    return (await _compare_models_calls(monkeypatch, tmp_path))[1:]


@pytest.mark.asyncio
@pytest.mark.parametrize(('drive', 'role'), [
    (_pairwise_judge, 'eval_pairwise_judge'),
    (_plan_judge, 'eval_judge'),
    (_assessor, 'eval_assessor'),
    (_synthesizer, 'eval_synthesizer'),
], ids=['pairwise_judge', 'plan_judge', 'assessor', 'synthesizer'])
async def test_tmp_cwd_eval_calls_resolve_off_the_tmp_bucket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    drive: Callable[[pytest.MonkeyPatch, Path], Awaitable[list[dict[str, Any]]]],
    role: str,
) -> None:
    calls = await drive(monkeypatch, tmp_path)

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs['cwd'] == _TMP
    env: dict[str, str] = {}
    apply_spawn_env(env, kwargs['spawn_env'])
    identity = parse_spawn_identity(env, '', '', str(kwargs['cwd']))
    assert identity.project == 'dark_factory'
    assert identity.role == role
