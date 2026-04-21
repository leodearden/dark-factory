"""Tests for Harness._tag_task_modules()."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.agents.invoke import AgentResult
from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from shared.cli_invoke import AllAccountsCappedException


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


@pytest.fixture
def harness(config: OrchestratorConfig) -> Harness:
    return Harness(config)


SAMPLE_TASKS = [
    {
        'id': '1',
        'title': 'Add health endpoint',
        'description': 'Add GET /health to the server',
        'status': 'pending',
        'metadata': {},
        'dependencies': [],
    },
    {
        'id': '2',
        'title': 'Refactor config schema',
        'description': 'Split config into sub-models',
        'status': 'pending',
        'metadata': {},
        'dependencies': [],
    },
    {
        'id': '3',
        'title': 'Already tagged task',
        'description': 'This one has modules already',
        'status': 'pending',
        'metadata': {'modules': ['src/server']},
        'dependencies': [],
    },
]

AGENT_MODULE_RESPONSE = {
    'tasks': [
        {'id': '1', 'modules': ['src/server', 'tests']},
        {'id': '2', 'modules': ['src/config']},
    ],
}


@pytest.mark.asyncio
async def test_tag_task_modules_calls_update_for_untagged(harness, config):
    """Should invoke agent and call update_task for each untagged task."""
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output=json.dumps(AGENT_MODULE_RESPONSE),
        structured_output=AGENT_MODULE_RESPONSE,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)) as mock_invoke:
        await harness._tag_task_modules()

        # Agent should be called once with all untagged tasks
        mock_invoke.assert_called_once()
        prompt = mock_invoke.call_args.kwargs.get('prompt', mock_invoke.call_args[0][0] if mock_invoke.call_args[0] else '')
        # Should include task 1 and 2 but not task 3 (already tagged)
        assert '"1"' in prompt
        assert '"2"' in prompt

    # update_task called for tasks 1 and 2 only
    assert harness.scheduler.update_task.call_count == 2
    calls = harness.scheduler.update_task.call_args_list
    call_ids = {c.args[0] for c in calls}
    assert call_ids == {'1', '2'}

    # Verify module metadata content
    for call in calls:
        task_id, metadata_json = call.args
        metadata = json.loads(metadata_json)
        assert 'modules' in metadata
        assert isinstance(metadata['modules'], list)


@pytest.mark.asyncio
async def test_tag_task_modules_skips_when_all_tagged(harness):
    """Should skip agent invocation if all tasks already have modules."""
    all_tagged = [
        {
            'id': '1',
            'title': 'Task',
            'description': '',
            'status': 'pending',
            'metadata': {'modules': ['src/server']},
            'dependencies': [],
        },
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=all_tagged)
    harness.scheduler.update_task = AsyncMock()

    with patch('orchestrator.harness.invoke_agent', AsyncMock()) as mock_invoke:
        await harness._tag_task_modules()
        mock_invoke.assert_not_called()

    harness.scheduler.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_tag_task_modules_handles_agent_failure(harness):
    """Should log warning and continue if agent fails."""
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    # Non-zero cost/turns/duration avoid the zero-cost instant-exit heuristic
    # that would otherwise classify this as a cap hit in the unified loop.
    agent_result = AgentResult(
        success=False, output='Agent error',
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    harness.scheduler.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_tag_task_modules_handles_bad_json(harness):
    """Should handle unparseable agent output gracefully."""
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output='not valid json',
        structured_output=None,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    harness.scheduler.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_tag_task_modules_uses_correct_config(harness, config):
    """Should use module_tagger model/budget/turns from config."""
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output='{}',
        structured_output={'tasks': []},
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)) as mock_invoke:
        await harness._tag_task_modules()

        call_kwargs = mock_invoke.call_args.kwargs
        assert call_kwargs['model'] == config.models.module_tagger
        assert call_kwargs['max_turns'] == config.max_turns.module_tagger
        assert call_kwargs['max_budget_usd'] == config.budgets.module_tagger


@pytest.mark.asyncio
async def test_tag_task_modules_handles_all_accounts_capped(harness, caplog):
    """AllAccountsCappedException from invoke_with_cap_retry must return None gracefully.

    Before the explicit handler is added (step-4), the exception propagates and
    crashes _tag_task_modules, causing it to raise rather than return None.
    """
    import logging

    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    cap_exc = AllAccountsCappedException(
        retries=5, elapsed_secs=60.0, label='Module tagging'
    )

    with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
        with patch(
            'orchestrator.harness.invoke_with_cap_retry',
            AsyncMock(side_effect=cap_exc),
        ):
            result = await harness._tag_task_modules()

    # Must return None (no exception propagation)
    assert result is None

    # Must NOT have tagged any tasks
    harness.scheduler.update_task.assert_not_called()

    # Must emit a warning containing 'all accounts capped' (case-insensitive)
    warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'all accounts capped' in t.lower() for t in warning_texts
    ), f'Expected warning containing "all accounts capped", got: {warning_texts}'
