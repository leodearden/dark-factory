"""Tests for the agent loop."""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.agent_loop import (
    AgentLoop,
    CircuitBreakerError,
    ToolDefinition,
)


def _make_config(**overrides) -> ReconciliationConfig:
    defaults = {
        'agent_max_steps': 10,
        'agent_max_tokens': 4096,
        'max_mutations_per_stage': 5,
        'agent_llm_provider': 'anthropic',
        'agent_llm_model': 'claude-sonnet-4-20250514',
    }
    defaults.update(overrides)
    return ReconciliationConfig(**defaults)


@dataclass
class FakeToolUse:
    """Fake tool_use block that doesn't have MagicMock's special 'name' handling."""

    type: str = 'tool_use'
    id: str = 'call_1'
    name: str = 'stage_complete'
    input: dict | None = None


@dataclass
class FakeText:
    type: str = 'text'
    text: str = ''


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class FakeResponse:
    content: list = None
    usage: FakeUsage = None

    def __post_init__(self):
        if self.content is None:
            self.content = []
        if self.usage is None:
            self.usage = FakeUsage()


@pytest.mark.asyncio
async def test_terminal_tool_ends_loop():
    """Agent loop stops when terminal tool is called."""
    config = _make_config()

    async def my_tool(x: int = 0):
        return {'value': x}

    tools = {
        'my_tool': ToolDefinition(
            name='my_tool',
            description='Test tool',
            parameters={'type': 'object', 'properties': {'x': {'type': 'integer'}}},
            function=my_tool,
        ),
        'stage_complete': ToolDefinition(
            name='stage_complete',
            description='Complete',
            parameters={'type': 'object', 'properties': {'report': {'type': 'object'}}},
            function=lambda **kw: kw,
        ),
    }

    agent = AgentLoop(
        config=config,
        system_prompt='You are a test agent.',
        tools=tools,
        terminal_tool='stage_complete',
    )

    mock_response = FakeResponse(
        content=[
            FakeToolUse(
                id='call_1',
                name='stage_complete',
                input={'report': {'stats': {'tested': True}}},
            )
        ],
    )

    async def mock_llm(messages, tool_schemas):
        return mock_response

    agent._call_llm = mock_llm

    result, entries = await agent.run('test payload')

    assert result == {'report': {'stats': {'tested': True}}}
    assert len(entries) == 0


@pytest.mark.asyncio
async def test_mutation_is_journaled():
    """Mutation tools produce journal entries."""
    config = _make_config()

    call_count = 0

    async def mutating_tool(content: str = ''):
        nonlocal call_count
        call_count += 1
        return {'id': f'mem_{call_count}'}

    tools = {
        'add_memory': ToolDefinition(
            name='add_memory',
            description='Write memory',
            parameters={'type': 'object', 'properties': {'content': {'type': 'string'}}},
            function=mutating_tool,
            is_mutation=True,
            target_system='mem0',
        ),
        'stage_complete': ToolDefinition(
            name='stage_complete',
            description='Complete',
            parameters={'type': 'object', 'properties': {'report': {'type': 'object'}}},
            function=lambda **kw: kw,
        ),
    }

    agent = AgentLoop(
        config=config,
        system_prompt='Test',
        tools=tools,
        terminal_tool='stage_complete',
    )

    # Step 1: call add_memory
    response1 = FakeResponse(
        content=[FakeToolUse(id='call_1', name='add_memory', input={'content': 'test'})],
    )

    # Step 2: call stage_complete
    response2 = FakeResponse(
        content=[FakeToolUse(id='call_2', name='stage_complete', input={'report': {}})],
    )

    call_idx = 0
    responses = [response1, response2]

    async def mock_llm(messages, tool_schemas):
        nonlocal call_idx
        resp = responses[call_idx]
        call_idx += 1
        return resp

    agent._call_llm = mock_llm

    result, entries = await agent.run('test')
    assert len(entries) == 1
    assert entries[0].operation == 'add_memory'
    assert entries[0].target_system == 'mem0'


@pytest.mark.asyncio
async def test_circuit_breaker():
    """Exceeding max mutations raises CircuitBreakerError."""
    config = _make_config(max_mutations_per_stage=2)

    async def mutating_tool(**kwargs):
        return {'ok': True}

    tools = {
        'mutate': ToolDefinition(
            name='mutate',
            description='Mutate',
            parameters={'type': 'object', 'properties': {}},
            function=mutating_tool,
            is_mutation=True,
            target_system='test',
        ),
        'stage_complete': ToolDefinition(
            name='stage_complete',
            description='Complete',
            parameters={'type': 'object', 'properties': {}},
            function=lambda **kw: kw,
        ),
    }

    agent = AgentLoop(
        config=config,
        system_prompt='Test',
        tools=tools,
        terminal_tool='stage_complete',
    )

    call_idx = 0

    async def mock_llm(messages, tool_schemas):
        nonlocal call_idx
        call_idx += 1
        return FakeResponse(
            content=[FakeToolUse(id=f'call_{call_idx}', name='mutate', input={})],
        )

    agent._call_llm = mock_llm

    with pytest.raises(CircuitBreakerError):
        await agent.run('test')


@pytest.mark.asyncio
async def test_max_steps_reached():
    """Agent stops after max_steps with a warning."""
    config = _make_config(agent_max_steps=2)

    async def noop(**kwargs):
        return {'ok': True}

    tools = {
        'noop': ToolDefinition(
            name='noop',
            description='No-op',
            parameters={'type': 'object', 'properties': {}},
            function=noop,
        ),
        'stage_complete': ToolDefinition(
            name='stage_complete',
            description='Complete',
            parameters={'type': 'object', 'properties': {}},
            function=lambda **kw: kw,
        ),
    }

    agent = AgentLoop(
        config=config,
        system_prompt='Test',
        tools=tools,
        terminal_tool='stage_complete',
    )

    call_idx = 0

    async def mock_llm(messages, tool_schemas):
        nonlocal call_idx
        call_idx += 1
        return FakeResponse(
            content=[FakeToolUse(id=f'call_{call_idx}', name='noop', input={})],
        )

    agent._call_llm = mock_llm

    result, entries = await agent.run('test')
    assert result.get('warning') == 'max_steps_reached'


@pytest.mark.asyncio
async def test_no_tool_calls_ends_loop():
    """Agent with no tool calls in response ends gracefully."""
    config = _make_config()

    tools = {
        'stage_complete': ToolDefinition(
            name='stage_complete',
            description='Complete',
            parameters={'type': 'object', 'properties': {}},
            function=lambda **kw: kw,
        ),
    }

    agent = AgentLoop(
        config=config,
        system_prompt='Test',
        tools=tools,
        terminal_tool='stage_complete',
    )

    async def mock_llm(messages, tool_schemas):
        return FakeResponse(content=[FakeText(text='I am done thinking.')])

    agent._call_llm = mock_llm

    result, entries = await agent.run('test')
    assert result.get('warning') == 'no_tool_calls'
    assert 'I am done thinking.' in result.get('text', '')
