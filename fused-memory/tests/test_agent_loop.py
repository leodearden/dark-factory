"""Tests for the agent loop."""

import json
import logging
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import AgentResult
from shared.testing import make_gate_mock

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.agent_loop import (
    CLI_WARNING_ORIGINS,
    AgentLoop,
    CircuitBreakerError,
    ToolDefinition,
    _OpenAIResponseAdapter,
    _TextBlock,
    _ToolUseBlock,
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
    content: list = field(default_factory=list)
    usage: FakeUsage = field(default_factory=FakeUsage)


# --- Fake OpenAI SDK response dataclasses ---


@dataclass
class FakeOpenAIFunction:
    """Fake function object nested inside a tool call."""
    name: str
    arguments: str  # JSON string


@dataclass
class FakeOpenAIToolCall:
    """Fake tool_call object on an OpenAI message."""
    id: str
    type: str
    function: FakeOpenAIFunction


@dataclass
class FakeOpenAIMessage:
    """Fake message object from OpenAI chat completion choice."""
    content: str | None
    tool_calls: list | None = None


@dataclass
class FakeOpenAIChoice:
    """Fake single choice in an OpenAI response."""
    message: FakeOpenAIMessage


@dataclass
class FakeOpenAIUsage:
    """Fake usage stats for OpenAI response."""
    total_tokens: int = 150


@dataclass
class FakeOpenAIResponse:
    """Fake OpenAI chat completion response."""
    choices: list = field(default_factory=list)
    usage: FakeOpenAIUsage = field(default_factory=FakeOpenAIUsage)


# --- OpenAI content block dispatch test ---


@pytest.mark.asyncio
async def test_openai_content_block_dispatch():
    """_call_openai correctly dispatches _TextBlock/_ToolUseBlock (actual adapter types).

    Uses _TextBlock and _ToolUseBlock directly (not FakeText/FakeToolUse) to establish
    baseline coverage before refactoring hasattr dispatch to isinstance.  Should pass
    with current hasattr code because both types carry a .type attribute.
    """
    config = _make_config(agent_llm_provider='openai', agent_llm_model='gpt-4o')

    agent = AgentLoop(
        config=config,
        system_prompt='Dispatch test agent.',
        tools={},
        terminal_tool='stage_complete',
    )

    # Messages using the actual runtime adapter types produced by _OpenAIResponseAdapter
    messages = [
        {'role': 'user', 'content': 'initial string message'},
        {'role': 'assistant', 'content': [
            _TextBlock(text='thinking about it'),
            _ToolUseBlock(id='tc1', name='my_tool', input={'x': 1}),
        ]},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'tc1', 'content': '{"result": 42}'},
        ]},
    ]

    tool_schemas = [
        {
            'name': 'my_tool',
            'description': 'A tool',
            'input_schema': {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
        }
    ]

    captured: dict = {}

    async def fake_create(**kwargs):
        captured['messages'] = list(kwargs['messages'])
        return FakeOpenAIResponse(
            choices=[
                FakeOpenAIChoice(
                    message=FakeOpenAIMessage(content='done', tool_calls=None)
                )
            ],
            usage=FakeOpenAIUsage(total_tokens=50),
        )

    mock_completions = MagicMock()
    mock_completions.create = fake_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch('openai.AsyncOpenAI', return_value=mock_client):
        await agent._call_openai(messages, tool_schemas)  # type: ignore[arg-type]

    sent = captured['messages']

    # Index 0: system message injected by _call_openai
    assert sent[0] == {'role': 'system', 'content': 'Dispatch test agent.'}

    # Index 1: user string passthrough
    assert sent[1] == {'role': 'user', 'content': 'initial string message'}

    # _TextBlock → role=assistant, content=text string
    text_msg = next(
        m for m in sent if m.get('role') == 'assistant' and isinstance(m.get('content'), str)
    )
    assert text_msg['content'] == 'thinking about it'

    # _ToolUseBlock → role=assistant, tool_calls=[{id, type, function}]
    tool_call_msg = next(
        m for m in sent if m.get('role') == 'assistant' and 'tool_calls' in m
    )
    assert tool_call_msg['tool_calls'][0]['id'] == 'tc1'
    assert tool_call_msg['tool_calls'][0]['type'] == 'function'
    assert tool_call_msg['tool_calls'][0]['function']['name'] == 'my_tool'
    assert json.loads(tool_call_msg['tool_calls'][0]['function']['arguments']) == {'x': 1}

    # tool_result dict → role=tool
    tool_result_msg = next(m for m in sent if m.get('role') == 'tool')
    assert tool_result_msg['tool_call_id'] == 'tc1'
    assert tool_result_msg['content'] == '{"result": 42}'


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


# --- _OpenAIResponseAdapter tests ---


def test_openai_adapter_text_only():
    """_OpenAIResponseAdapter with text content and no tool_calls produces one _TextBlock."""
    response = FakeOpenAIResponse(
        choices=[
            FakeOpenAIChoice(
                message=FakeOpenAIMessage(content='Hello, world!', tool_calls=None)
            )
        ],
        usage=FakeOpenAIUsage(total_tokens=42),
    )
    adapter = _OpenAIResponseAdapter(response)

    assert len(adapter.content) == 1
    block = adapter.content[0]
    assert block.type == 'text'
    assert block.text == 'Hello, world!'


def test_openai_adapter_tool_calls_only():
    """_OpenAIResponseAdapter with tool_calls and no content produces _ToolUseBlock objects."""
    tc1 = FakeOpenAIToolCall(
        id='call_abc',
        type='function',
        function=FakeOpenAIFunction(name='add_memory', arguments='{"content": "test"}'),
    )
    tc2 = FakeOpenAIToolCall(
        id='call_def',
        type='function',
        function=FakeOpenAIFunction(name='stage_complete', arguments='{"report": {}}'),
    )
    response = FakeOpenAIResponse(
        choices=[
            FakeOpenAIChoice(
                message=FakeOpenAIMessage(content=None, tool_calls=[tc1, tc2])
            )
        ],
    )
    adapter = _OpenAIResponseAdapter(response)

    assert len(adapter.content) == 2

    b0 = adapter.content[0]
    assert b0.type == 'tool_use'
    assert b0.id == 'call_abc'
    assert b0.name == 'add_memory'
    assert b0.input == {'content': 'test'}

    b1 = adapter.content[1]
    assert b1.type == 'tool_use'
    assert b1.id == 'call_def'
    assert b1.name == 'stage_complete'
    assert b1.input == {'report': {}}


def test_openai_adapter_mixed_text_and_tool_calls():
    """_OpenAIResponseAdapter with both content and tool_calls produces text + tool_use blocks."""
    tc = FakeOpenAIToolCall(
        id='call_xyz',
        type='function',
        function=FakeOpenAIFunction(
            name='search_memory',
            arguments='{"query": "recent facts", "limit": 10}',
        ),
    )
    response = FakeOpenAIResponse(
        choices=[
            FakeOpenAIChoice(
                message=FakeOpenAIMessage(
                    content='Let me search for that.',
                    tool_calls=[tc],
                )
            )
        ],
    )
    adapter = _OpenAIResponseAdapter(response)

    assert len(adapter.content) == 2

    text_blocks = [b for b in adapter.content if b.type == 'text']
    tool_blocks = [b for b in adapter.content if b.type == 'tool_use']

    assert len(text_blocks) == 1
    assert text_blocks[0].text == 'Let me search for that.'

    assert len(tool_blocks) == 1
    assert tool_blocks[0].id == 'call_xyz'
    assert tool_blocks[0].name == 'search_memory'
    assert tool_blocks[0].input == {'query': 'recent facts', 'limit': 10}


def test_openai_adapter_tool_arguments_json_parsed():
    """_OpenAIResponseAdapter parses tool arguments JSON string into a dict."""
    tc = FakeOpenAIToolCall(
        id='call_1',
        type='function',
        function=FakeOpenAIFunction(
            name='write_entity',
            arguments='{"entity": "Project", "fact": "Active", "nested": {"key": "value"}}',
        ),
    )
    response = FakeOpenAIResponse(
        choices=[FakeOpenAIChoice(message=FakeOpenAIMessage(content=None, tool_calls=[tc]))],
    )
    adapter = _OpenAIResponseAdapter(response)

    block = adapter.content[0]
    assert isinstance(block.input, dict)
    assert block.input['entity'] == 'Project'
    assert block.input['nested'] == {'key': 'value'}


# --- OpenAI provider _call_openai tests ---


def test_to_anthropic_schema_returns_tool_param_shape():
    """to_anthropic_schema() returns a dict with exactly the keys ToolParam requires.

    Validates structural compatibility before type annotation is tightened to ToolParam.
    """
    tool = ToolDefinition(
        name='search_memory',
        description='Search for memories by query string.',
        parameters={
            'type': 'object',
            'properties': {'query': {'type': 'string'}, 'limit': {'type': 'integer'}},
            'required': ['query'],
        },
        function=lambda **kw: kw,
    )

    schema = tool.to_anthropic_schema()

    # Must have exactly the three keys ToolParam requires
    assert isinstance(schema, dict)
    assert 'name' in schema
    assert 'description' in schema
    assert 'input_schema' in schema

    # Correct values
    assert schema['name'] == 'search_memory'
    assert schema['description'] == 'Search for memories by query string.'
    assert schema['input_schema'] == tool.parameters
    assert isinstance(schema['input_schema'], dict)


@pytest.mark.asyncio
async def test_openai_tool_schema_conversion():
    """_call_openai converts Anthropic tool schemas (input_schema) to OpenAI format (parameters)."""
    config = _make_config(agent_llm_provider='openai', agent_llm_model='gpt-4o')

    async def noop(**kwargs):
        return {'ok': True}

    tools = {
        'search_memory': ToolDefinition(
            name='search_memory',
            description='Search memories by query',
            parameters={
                'type': 'object',
                'properties': {'query': {'type': 'string'}, 'limit': {'type': 'integer'}},
                'required': ['query'],
            },
            function=noop,
        ),
        'stage_complete': ToolDefinition(
            name='stage_complete',
            description='Complete the stage',
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

    # Response triggers immediate terminal result
    captured = {}

    async def fake_create(**kwargs):
        captured['messages'] = kwargs['messages']
        captured['tools'] = kwargs['tools']
        return FakeOpenAIResponse(
            choices=[
                FakeOpenAIChoice(
                    message=FakeOpenAIMessage(
                        content=None,
                        tool_calls=[
                            FakeOpenAIToolCall(
                                id='tc1',
                                type='function',
                                function=FakeOpenAIFunction(
                                    name='stage_complete',
                                    arguments='{"report": {}}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=FakeOpenAIUsage(total_tokens=100),
        )

    mock_completions = MagicMock()
    mock_completions.create = fake_create

    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch('openai.AsyncOpenAI', return_value=mock_client):
        result, entries = await agent.run('initial payload')

    assert result == {'report': {}}

    # Verify OpenAI tool schema format
    sent_tools = captured['tools']
    assert len(sent_tools) == 2
    by_name = {t['function']['name']: t for t in sent_tools}

    # Anthropic input_schema → OpenAI parameters
    search_tool = by_name['search_memory']
    assert search_tool['type'] == 'function'
    assert 'parameters' in search_tool['function']
    assert 'input_schema' not in search_tool['function']
    assert search_tool['function']['parameters']['properties']['query']['type'] == 'string'
    assert search_tool['function']['description'] == 'Search memories by query'

    # System prompt is first message with role='system'
    sent_messages = captured['messages']
    assert sent_messages[0]['role'] == 'system'
    assert sent_messages[0]['content'] == 'You are a test agent.'


@pytest.mark.asyncio
async def test_openai_message_conversion():
    """_call_openai correctly converts all Anthropic message formats to OpenAI messages."""
    config = _make_config(agent_llm_provider='openai', agent_llm_model='gpt-4o')

    async def noop(**kwargs):
        return {'ok': True}

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
        system_prompt='System prompt here.',
        tools=tools,
        terminal_tool='stage_complete',
    )

    # Build a messages list that exercises all conversion branches:
    # 1. String content (initial user message)
    # 2. List with text block (assistant thinking)
    # 3. List with tool_use block (assistant tool call)
    # 4. List with tool_result dicts (user follow-up)
    # Use _TextBlock/_ToolUseBlock (the actual runtime adapter types) because
    # _call_openai dispatches via isinstance, not hasattr.
    messages = [
        {'role': 'user', 'content': 'initial payload string'},
        {'role': 'assistant', 'content': [
            _TextBlock(text='Thinking about this...'),
            _ToolUseBlock(id='call_a', name='noop', input={'x': 1}),
        ]},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'call_a', 'content': '{"ok": true}'},
        ]},
    ]

    captured = {}

    async def fake_create(**kwargs):
        captured['messages'] = list(kwargs['messages'])
        return FakeOpenAIResponse(
            choices=[
                FakeOpenAIChoice(
                    message=FakeOpenAIMessage(
                        content=None,
                        tool_calls=[
                            FakeOpenAIToolCall(
                                id='tc_final',
                                type='function',
                                function=FakeOpenAIFunction(
                                    name='stage_complete',
                                    arguments='{}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=FakeOpenAIUsage(total_tokens=50),
        )

    mock_completions = MagicMock()
    mock_completions.create = fake_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch('openai.AsyncOpenAI', return_value=mock_client):
        await agent._call_openai(messages, [])

    sent = captured['messages']

    # Index 0: system prompt injected by _call_openai
    assert sent[0] == {'role': 'system', 'content': 'System prompt here.'}

    # Index 1: string content → passthrough
    assert sent[1] == {'role': 'user', 'content': 'initial payload string'}

    # Indices 2 and 3: text block and tool_use block from assistant message
    # text block → role=assistant, content=text
    text_msg = next(m for m in sent[2:] if m.get('role') == 'assistant' and 'content' in m
                    and isinstance(m.get('content'), str))
    assert text_msg['content'] == 'Thinking about this...'

    # tool_use block → role=assistant, tool_calls=[{id, type, function}]
    tool_call_msg = next(
        m for m in sent[2:] if m.get('role') == 'assistant' and 'tool_calls' in m
    )
    assert tool_call_msg['tool_calls'][0]['id'] == 'call_a'
    assert tool_call_msg['tool_calls'][0]['type'] == 'function'
    assert tool_call_msg['tool_calls'][0]['function']['name'] == 'noop'
    assert json.loads(tool_call_msg['tool_calls'][0]['function']['arguments']) == {'x': 1}

    # tool_result dict → role=tool, tool_call_id, content
    tool_result_msg = next(m for m in sent if m.get('role') == 'tool')
    assert tool_result_msg['tool_call_id'] == 'call_a'
    assert tool_result_msg['content'] == '{"ok": true}'


@pytest.mark.asyncio
async def test_openai_tools_omitted_when_empty():
    """When tool_schemas=[], 'tools' key must NOT appear in kwargs to create().

    This test FAILS with current code because line 273 passes tools=None to the
    OpenAI SDK when openai_tools is empty — None is type-invalid for the tools param
    (the SDK uses NOT_GIVEN sentinel internally).  The fix is a conditional kwargs dict.
    """
    config = _make_config(agent_llm_provider='openai', agent_llm_model='gpt-4o')

    agent = AgentLoop(
        config=config,
        system_prompt='Tool omission test.',
        tools={},
        terminal_tool='stage_complete',
    )

    captured_kwargs: dict = {}

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeOpenAIResponse(
            choices=[
                FakeOpenAIChoice(
                    message=FakeOpenAIMessage(content='done', tool_calls=None)
                )
            ],
            usage=FakeOpenAIUsage(total_tokens=10),
        )

    mock_completions = MagicMock()
    mock_completions.create = fake_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch('openai.AsyncOpenAI', return_value=mock_client):
        await agent._call_openai([{'role': 'user', 'content': 'hi'}], [])

    # When tool_schemas is empty, 'tools' must NOT be sent to the API at all.
    assert 'tools' not in captured_kwargs, (
        f"'tools' key should be absent when no tool schemas provided, got: {captured_kwargs.get('tools')!r}"
    )


@pytest.mark.asyncio
async def test_openai_round_trip_two_turns():
    """Full AgentLoop round-trip with OpenAI provider: tool call then terminal.

    Verifies:
    - Agent executes the tool function on first response
    - Tool results are sent back in OpenAI tool format on second call
    - Terminal result is returned correctly
    - llm_call_count and token_count are updated
    """
    config = _make_config(agent_llm_provider='openai', agent_llm_model='gpt-4o')

    executed_calls = []

    async def my_tool(value: int = 0):
        executed_calls.append(value)
        return {'doubled': value * 2}

    tools = {
        'my_tool': ToolDefinition(
            name='my_tool',
            description='Doubles a number',
            parameters={
                'type': 'object',
                'properties': {'value': {'type': 'integer'}},
                'required': ['value'],
            },
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
        system_prompt='You are a round-trip test agent.',
        tools=tools,
        terminal_tool='stage_complete',
    )

    call_count = 0
    second_call_messages: list[dict] | None = None

    async def fake_create(**kwargs):
        nonlocal call_count, second_call_messages
        call_count += 1
        if call_count == 1:
            # First call: request my_tool
            return FakeOpenAIResponse(
                choices=[
                    FakeOpenAIChoice(
                        message=FakeOpenAIMessage(
                            content=None,
                            tool_calls=[
                                FakeOpenAIToolCall(
                                    id='call_tool_1',
                                    type='function',
                                    function=FakeOpenAIFunction(
                                        name='my_tool',
                                        arguments='{"value": 7}',
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=FakeOpenAIUsage(total_tokens=80),
            )
        else:
            # Second call: terminal
            second_call_messages = list(kwargs['messages'])
            return FakeOpenAIResponse(
                choices=[
                    FakeOpenAIChoice(
                        message=FakeOpenAIMessage(
                            content=None,
                            tool_calls=[
                                FakeOpenAIToolCall(
                                    id='call_terminal',
                                    type='function',
                                    function=FakeOpenAIFunction(
                                        name='stage_complete',
                                        arguments='{"report": {"doubled": 14}}',
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=FakeOpenAIUsage(total_tokens=60),
            )

    mock_completions = MagicMock()
    mock_completions.create = fake_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch('openai.AsyncOpenAI', return_value=mock_client):
        result, entries = await agent.run('run the test')

    # Agent returned terminal tool input
    assert result == {'report': {'doubled': 14}}
    assert len(entries) == 0  # my_tool is not a mutation

    # Tool was actually executed
    assert executed_calls == [7]

    # LLM was called twice
    assert agent.llm_call_count == 2
    assert agent.token_count == 80 + 60

    # Second call messages include a 'tool' role message with tool results
    assert second_call_messages is not None
    tool_result_msgs = [m for m in second_call_messages if m.get('role') == 'tool']
    assert len(tool_result_msgs) == 1
    assert tool_result_msgs[0]['tool_call_id'] == 'call_tool_1'
    result_content = json.loads(tool_result_msgs[0]['content'])
    assert result_content['doubled'] == 14


# --- Claude CLI provider tests ---
#
# On claude_cli the whole investigation is ONE CLI invocation (task 4344): the
# CLI runs its own built-in ``cli_tools`` and delivers the terminal tool's
# payload through ``--json-schema``.  Every test below drives the public
# ``run()`` and patches only the ``invoke_with_cap_retry`` seam (or, for the
# forwarding test, ``invoke_claude_agent`` one level lower).

_P = {
    'type': 'object',
    'properties': {
        'verdict': {'type': 'string', 'enum': ['confirmed', 'contradicted']},
        'summary': {'type': 'string'},
    },
    'required': ['verdict', 'summary'],
}

_CLI_TOOLS = ('Read', 'Grep', 'Glob')

_MEASURED_REFUSAL_OUTPUT = (
    "API Error: Sonnet 5.5's safeguards flagged this message "
    '(https://www.anthropic.com/legal/aup). This sometimes happens with safe, '
    "normal conversations. Claude Code can't respond to this message with "
    'Sonnet 5.5.\n\nRequest ID: req_x'
)


def _make_cli_config(**overrides) -> ReconciliationConfig:
    defaults = {
        'agent_max_steps': 10,
        'agent_max_tokens': 4096,
        'max_mutations_per_stage': 5,
        'agent_llm_provider': 'claude_cli',
        'agent_llm_model': 'sonnet',
    }
    defaults.update(overrides)
    return ReconciliationConfig(**defaults)


def _cli_tool_defs() -> dict[str, ToolDefinition]:
    return {
        'read_file': ToolDefinition(
            name='read_file',
            description='Read file contents from the codebase.',
            parameters={'type': 'object', 'properties': {'path': {'type': 'string'}}},
            function=lambda **kw: kw,
        ),
        'verification_complete': ToolDefinition(
            name='verification_complete',
            description='Deliver your findings.',
            parameters=_P,
            function=lambda **kw: kw,
        ),
    }


def _cli_agent(cli_tools=_CLI_TOOLS, **kw) -> AgentLoop:
    kw.setdefault('config', _make_cli_config())
    kw.setdefault('usage_gate', make_gate_mock())
    return AgentLoop(
        system_prompt='Caller system prompt.',
        tools=_cli_tool_defs(),
        terminal_tool='verification_complete',
        cli_tools=cli_tools,
        **kw,
    )


def _verdict_result(**overrides) -> AgentResult:
    fields = {
        'success': True,
        'output': '',
        'session_id': 'sess-ok',
        'structured_output': {'verdict': 'confirmed', 'summary': 's'},
    }
    fields.update(overrides)
    return AgentResult(**fields)


async def _run_cli(agent: AgentLoop, result: AgentResult, payload: str = 'payload'):
    """Run *agent* against one mocked CLI result; return (run output, mock)."""
    with patch(
        'fused_memory.reconciliation.agent_loop.invoke_with_cap_retry',
        new_callable=AsyncMock,
    ) as mock_invoke:
        mock_invoke.return_value = result
        out = await agent.run(payload)
    return out, mock_invoke


@pytest.mark.asyncio
async def test_cli_run_is_one_invocation_returning_the_terminal_payload():
    from pathlib import Path

    from fused_memory.reconciliation import _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS

    config = _make_cli_config()
    agent = _cli_agent(config=config)
    (payload, journal), mock_invoke = await _run_cli(
        agent, _verdict_result(input_tokens=30, output_tokens=12),
    )

    assert payload == {'verdict': 'confirmed', 'summary': 's'}
    assert journal == []
    mock_invoke.assert_called_once()
    kwargs = mock_invoke.call_args.kwargs
    assert kwargs['prompt'] == 'payload'
    assert kwargs['output_schema'] == _P
    assert kwargs['available_tools'] == ['Read', 'Grep', 'Glob']
    assert kwargs['permission_mode'] == 'dontAsk'
    assert kwargs.get('disallowed_tools') is None
    assert kwargs['mcp_config'] == {'mcpServers': {}}
    assert kwargs['strict_mcp_config'] is True
    assert kwargs['model'] == config.agent_llm_model
    assert kwargs['timeout_seconds'] == float(config.agent_cli_timeout_seconds)
    assert kwargs['cap_wait_sanity_secs'] == _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS
    assert kwargs['cwd'] == Path(config.explore_codebase_root)
    assert not kwargs.get('resume_session_id')
    assert not kwargs.get('resume_delivers_prompt')
    assert agent.llm_call_count == 1
    assert agent.token_count == 42


@pytest.mark.asyncio
async def test_cli_run_passes_a_workable_max_turns():
    """The cap bounds the WHOLE investigation, and it can never be 1: the model
    emits a prose turn before it calls StructuredOutput, and a cap of 1 leaves
    no room for it.

    Pinned as the INVARIANT (>= 3, the floor both migrated siblings use — see
    test_judge.py's and test_task_curator.py's identical pins) rather than the
    tuned constant, so retuning _AGENT_CLI_MAX_TURNS does not churn this test.
    """
    _out, mock_invoke = await _run_cli(_cli_agent(), _verdict_result())
    assert mock_invoke.call_args.kwargs['max_turns'] >= 3, (
        'max_turns=1 leaves no room for the prose turn the model emits before '
        'calling StructuredOutput; see _AGENT_CLI_MAX_TURNS.'
    )


@pytest.mark.asyncio
async def test_cli_system_prompt_cannot_read_as_a_pseudo_tool_registry():
    """No in-process tool name, heading or parameter schema reaches the CLI.

    The model calls whatever the prompt presents as a tool NATIVELY, and the
    CLI rejects each such call: the measured attractor that the JSON
    pseudo-tool protocol produced is recorded in
    fused-memory/scripts/probe_schema_max_turns.py's docstring.
    """
    _out, mock_invoke = await _run_cli(_cli_agent(), _verdict_result())
    system_prompt = mock_invoke.call_args.kwargs['system_prompt']

    assert system_prompt.startswith('Caller system prompt.')
    for name in (*_CLI_TOOLS, 'StructuredOutput'):
        assert name in system_prompt, f'{name!r} missing from: {system_prompt!r}'
    assert 'read_file' not in system_prompt
    assert 'verification_complete' not in system_prompt
    assert not any(line.startswith('### ') for line in system_prompt.splitlines())
    assert 'Available Tools' not in system_prompt
    assert '"properties"' not in system_prompt
    assert 'tool_calls' not in system_prompt


@pytest.mark.asyncio
async def test_cli_run_without_cli_tools_offers_an_empty_registry():
    """An AgentLoop that names no CLI tools gets ``--tools ''``, never the CLI's
    default registry.
    """
    agent = AgentLoop(
        config=_make_cli_config(),
        system_prompt='Caller system prompt.',
        tools=_cli_tool_defs(),
        terminal_tool='verification_complete',
        usage_gate=make_gate_mock(),
    )
    _out, mock_invoke = await _run_cli(agent, _verdict_result())
    assert mock_invoke.call_args.kwargs['available_tools'] == []


@pytest.mark.asyncio
async def test_cli_run_requires_the_terminal_tool():
    """The terminal tool's parameters ARE the CLI output schema, so a run
    without one cannot be started.
    """
    tools = _cli_tool_defs()
    del tools['verification_complete']
    agent = AgentLoop(
        config=_make_cli_config(),
        system_prompt='Caller system prompt.',
        tools=tools,
        terminal_tool='verification_complete',
        cli_tools=_CLI_TOOLS,
        usage_gate=make_gate_mock(),
    )
    with patch(
        'fused_memory.reconciliation.agent_loop.invoke_with_cap_retry',
        new_callable=AsyncMock,
    ) as mock_invoke, pytest.raises(ValueError, match='verification_complete'):
        await agent.run('payload')
    mock_invoke.assert_not_called()


_CLI_FAILURE_CASES = [
    pytest.param(
        AgentResult(
            success=False,
            output=_MEASURED_REFUSAL_OUTPUT,
            subtype='success',
            stop_reason='refusal',
            session_id='sess-r',
        ),
        'api_refusal',
        id='refusal',
    ),
    pytest.param(
        AgentResult(success=False, output='', subtype='error_max_turns'),
        'cli_max_turns',
        id='max_turns',
    ),
    pytest.param(
        _verdict_result(structured_output='not valid json {'),
        'cli_output_unparseable',
        id='unparseable',
    ),
    pytest.param(_verdict_result(structured_output=None), 'cli_output_empty', id='none'),
    pytest.param(_verdict_result(structured_output={}), 'cli_output_empty', id='empty_dict'),
    pytest.param(_verdict_result(structured_output='null'), 'cli_output_empty', id='json_null'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('result', 'origin'), _CLI_FAILURE_CASES)
async def test_cli_failures_end_the_run_with_a_closed_origin(result, origin, caplog):
    """Each recognised CLI failure ends the run as an audited no-tool-call exit
    whose ``warning_origin`` is a closed-vocabulary census token, so verify()
    writes an agent_failed row rather than a prose-only error row.
    """
    with caplog.at_level(logging.WARNING):
        (payload, journal), _mock = await _run_cli(_cli_agent(), result)

    assert payload['warning'] == 'no_tool_calls'
    assert payload['warning_origin'] == origin
    assert journal == []
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(origin in m for m in warnings), f'no WARNING naming {origin}: {warnings}'
    if origin == 'api_refusal':
        assert any('req_x' in m for m in warnings), f'request id not logged: {warnings}'


@pytest.mark.asyncio
async def test_cli_warning_origins_is_exactly_what_the_cli_path_emits():
    """The closed vocabulary must not drift from its only producer, in either
    direction: every token the CLI path emits is a member, and every member is
    emitted by some failure.
    """
    emitted = set()
    for case in _CLI_FAILURE_CASES:
        result, _origin = case.values
        (payload, _journal), _mock = await _run_cli(_cli_agent(), result)
        emitted.add(payload['warning_origin'])
    assert emitted == set(CLI_WARNING_ORIGINS)


@pytest.mark.asyncio
async def test_cli_max_turns_is_distinct_from_refusal_and_crash():
    """error_max_turns is its own audited token, never the refusal token, and
    never a raise; any other CLI failure still raises with its diagnostics.
    Both calls reached the model, so both are counted.
    """
    agent = _cli_agent()
    (payload, _journal), _mock = await _run_cli(
        agent, AgentResult(success=False, output='', subtype='error_max_turns'),
    )
    assert payload['warning_origin'] == 'cli_max_turns'

    crashed = AgentResult(
        success=False,
        output='',
        stderr='ENOENT: claude CLI binary not found',
        subtype='error_unexpected',
    )
    with pytest.raises(RuntimeError) as excinfo:
        await _run_cli(agent, crashed)
    msg = str(excinfo.value)
    assert msg.startswith('Claude CLI agent failed:'), f'unexpected prefix: {msg!r}'
    assert 'ENOENT: claude CLI binary not found' in msg, f'stderr missing from: {msg!r}'
    assert "subtype='error_unexpected'" in msg, f'subtype missing from: {msg!r}'
    assert agent.llm_call_count == 2


@pytest.mark.asyncio
async def test_cli_structured_output_json_string_is_parsed():
    (payload, _journal), _mock = await _run_cli(
        _cli_agent(),
        _verdict_result(structured_output='{"verdict": "confirmed", "summary": "s"}'),
    )
    assert payload == {'verdict': 'confirmed', 'summary': 's'}


@pytest.mark.asyncio
async def test_agent_loop_explicit_cwd_overrides_config_explore_root(tmp_path):
    """An explicit `cwd=` wins over config.explore_codebase_root (task 4722, PRD D5).

    The verifier runs against the TASK's own project root, not the
    process-global explore root.  The `!=` assertion is not redundant: a
    regression that silently ignores the parameter would still produce a Path,
    and pinning the inequality against a deliberately DISTINCT global root
    makes that failure mode loud.
    """
    from pathlib import Path

    global_root = tmp_path / 'global'
    global_root.mkdir()
    target = tmp_path / 'target'
    target.mkdir()
    config = _make_cli_config(explore_codebase_root=str(global_root))

    _out, mock_invoke = await _run_cli(_cli_agent(config=config, cwd=target), _verdict_result())

    assert mock_invoke.call_args.kwargs['cwd'] == target
    assert mock_invoke.call_args.kwargs['cwd'] != Path(config.explore_codebase_root)


@pytest.mark.asyncio
async def test_agent_loop_cwd_defaults_to_config_explore_root(tmp_path):
    """Without an explicit cwd, AgentLoop falls back to config.explore_codebase_root.

    `AgentLoop(` has exactly ONE call site outside tests, verify.py, and it
    always passes `cwd`, so nothing in production depends on this fallback.
    This pins it while this file's construction sites still rely on it.
    """
    from pathlib import Path

    global_root = tmp_path / 'global'
    global_root.mkdir()
    config = _make_cli_config(explore_codebase_root=str(global_root))

    _out, mock_invoke = await _run_cli(_cli_agent(config=config), _verdict_result())

    assert mock_invoke.call_args.kwargs['cwd'] == Path(config.explore_codebase_root)


@pytest.mark.asyncio
async def test_cli_scoping_survives_forwarding_to_invoke_claude_agent(tmp_path):
    """Every scoping kwarg survives invoke_with_cap_retry's forwarding layer.

    ``invoke_with_cap_retry`` takes ``**invoke_kwargs`` and forwards them blind.
    ``usage_gate=None`` takes its single-invocation fast path, so the REAL
    forwarding code runs, and ``autospec=True`` is load-bearing: a bare
    ``AsyncMock`` swallows any keyword, while the autospec'd mock raises
    TypeError on one the real ``invoke_claude_agent`` does not accept.

    Deliberately NOT asserted: that the CLI honours the flags.  That is the
    external CLI's contract; fused-memory/scripts/probe_schema_max_turns.py
    measures it live.
    """
    target = tmp_path / 'target'
    target.mkdir()

    with patch('shared.cli_invoke.invoke_claude_agent', autospec=True) as mock_agent:
        mock_agent.return_value = _verdict_result()
        await _cli_agent(usage_gate=None, cwd=target).run('payload')

    mock_agent.assert_called_once()
    kwargs = mock_agent.call_args.kwargs
    assert kwargs['available_tools'] == ['Read', 'Grep', 'Glob']
    assert kwargs['permission_mode'] == 'dontAsk'
    assert kwargs['mcp_config'] == {'mcpServers': {}}
    assert kwargs['strict_mcp_config'] is True
    assert kwargs['output_schema'] == _P
    assert kwargs['cwd'] == target
