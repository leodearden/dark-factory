"""Tests for Harness._tag_task_modules()."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.agents.invoke import AgentResult
from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness


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

    agent_result = AgentResult(success=False, output='Agent error')

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
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)) as mock_invoke:
        await harness._tag_task_modules()

        call_kwargs = mock_invoke.call_args.kwargs
        assert call_kwargs['model'] == config.models.module_tagger
        assert call_kwargs['max_turns'] == config.max_turns.module_tagger
        assert call_kwargs['max_budget_usd'] == config.budgets.module_tagger
