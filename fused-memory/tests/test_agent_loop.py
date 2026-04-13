"""Tests for the agent loop."""

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.agent_loop import (
    AgentLoop,
    CircuitBreakerError,
    ToolDefinition,
    _CLIResponseAdapter,
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


def _cli_result_json(structured_output: dict, session_id: str = 'sess-1') -> bytes:
    """Build a fake CLI JSON response."""
    return json.dumps({
        'result': '',
        'session_id': session_id,
        'num_input_tokens': 1000,
        'num_output_tokens': 200,
        'structured_output': structured_output,
    }).encode()


@pytest.mark.asyncio
async def test_claude_cli_provider_first_call():
    """First CLI call uses --session-id and --system-prompt with tool schemas."""
    config = _make_cli_config()

    tools = {
        'stage_complete': ToolDefinition(
            name='stage_complete',
            description='Complete',
            parameters={'type': 'object', 'properties': {'report': {'type': 'object'}}},
            function=lambda **kw: kw,
        ),
    }

    agent = AgentLoop(
        config=config,
        system_prompt='Test system prompt',
        tools=tools,
        terminal_tool='stage_complete',
    )

    cli_output = _cli_result_json({
        'thinking': 'All done.',
        'tool_calls': [{'id': 'tc1', 'name': 'stage_complete', 'input': {'report': {}}}],
    })

    captured_cmd = []

    async def fake_subprocess(*args, **kwargs):
        captured_cmd.extend(args)
        proc = MagicMock()
        proc.returncode = 0

        async def communicate():
            return cli_output, b''
        proc.communicate = communicate
        return proc

    with patch('asyncio.create_subprocess_exec', side_effect=fake_subprocess):
        result, entries = await agent.run('initial payload')

    assert result == {'report': {}}
    cmd = captured_cmd
    assert 'claude' in cmd[0]
    assert '--session-id' in cmd
    assert '--system-prompt' in cmd
    # System prompt should include tool schemas
    sp_idx = cmd.index('--system-prompt') + 1
    assert 'stage_complete' in cmd[sp_idx]
    assert '--tools' in cmd
    # Prompt is last after --
    assert cmd[-1] == 'initial payload'


@pytest.mark.asyncio
async def test_claude_cli_provider_resume():
    """Second CLI call uses --resume instead of --session-id."""
    config = _make_cli_config(agent_max_steps=5)

    async def my_tool(**kwargs):
        return {'ok': True}

    tools = {
        'my_tool': ToolDefinition(
            name='my_tool',
            description='Test tool',
            parameters={'type': 'object', 'properties': {}},
            function=my_tool,
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

    call_count = 0
    captured_cmds = []

    async def fake_subprocess(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_cmds.append(list(args))
        proc = MagicMock()
        proc.returncode = 0

        if call_count == 1:
            output = _cli_result_json({
                'thinking': 'Calling tool.',
                'tool_calls': [{'id': 'tc1', 'name': 'my_tool', 'input': {}}],
            })
        else:
            output = _cli_result_json({
                'thinking': 'Done.',
                'tool_calls': [{'id': 'tc2', 'name': 'stage_complete', 'input': {}}],
            })

        async def communicate():
            return output, b''
        proc.communicate = communicate
        return proc

    with patch('asyncio.create_subprocess_exec', side_effect=fake_subprocess):
        result, entries = await agent.run('payload')

    assert call_count == 2
    # First call has --session-id
    assert '--session-id' in captured_cmds[0]
    assert '--resume' not in captured_cmds[0]
    # Second call has --resume
    assert '--resume' in captured_cmds[1]
    assert '--session-id' not in captured_cmds[1]


@pytest.mark.asyncio
async def test_claude_cli_response_adapter():
    """_CLIResponseAdapter produces correct _TextBlock/_ToolUseBlock."""
    structured = {
        'thinking': 'I should consolidate memories.',
        'tool_calls': [
            {'id': 'tc1', 'name': 'search_memory', 'input': {'query': 'test'}},
            {'id': 'tc2', 'name': 'delete_memory', 'input': {'id': 'mem-1'}},
        ],
    }

    adapter = _CLIResponseAdapter(structured)

    text_blocks = [b for b in adapter.content if b.type == 'text']
    tool_blocks = [b for b in adapter.content if b.type == 'tool_use']

    assert len(text_blocks) == 1
    assert text_blocks[0].text == 'I should consolidate memories.'
    assert len(tool_blocks) == 2
    assert tool_blocks[0].name == 'search_memory'
    assert tool_blocks[0].input == {'query': 'test'}
    assert tool_blocks[1].name == 'delete_memory'
    assert tool_blocks[1].id == 'tc2'


@pytest.mark.asyncio
async def test_claude_cli_response_adapter_no_thinking():
    """_CLIResponseAdapter handles empty thinking."""
    structured = {
        'thinking': '',
        'tool_calls': [{'id': 'tc1', 'name': 'stage_complete', 'input': {}}],
    }
    adapter = _CLIResponseAdapter(structured)
    text_blocks = [b for b in adapter.content if b.type == 'text']
    tool_blocks = [b for b in adapter.content if b.type == 'tool_use']
    assert len(text_blocks) == 0
    assert len(tool_blocks) == 1


@pytest.mark.asyncio
async def test_claude_cli_tool_results_serialization():
    """_serialize_tool_results formats tool results as text."""
    tool_results = [
        {
            'type': 'tool_result',
            'tool_use_id': 'tc1',
            'content': '{"id": "mem-1"}',
        },
        {
            'type': 'tool_result',
            'tool_use_id': 'tc2',
            'content': '{"error": "not found"}',
            'is_error': True,
        },
    ]

    text = AgentLoop._serialize_tool_results(tool_results)

    assert '[Tool Result: tc1] (OK)' in text
    assert '{"id": "mem-1"}' in text
    assert '[Tool Result: tc2] (ERROR)' in text
    assert '{"error": "not found"}' in text


@pytest.mark.asyncio
async def test_claude_cli_not_installed():
    """Clear error when Claude CLI is not installed."""
    config = _make_cli_config()

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

    with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError), \
         pytest.raises(RuntimeError, match='Claude CLI not found'):
        await agent.run('test')


# ---------------------------------------------------------------------------
# Cap-Hit Backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_claude_cli_sleeps_on_cap_hit():
    """_call_claude_cli sleeps before retrying on cap hit."""
    from unittest.mock import AsyncMock

    from fused_memory.reconciliation.agent_loop import _CAP_HIT_COOLDOWN_SECS

    config = _make_config(agent_llm_provider='claude-cli', agent_llm_model='opus')

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

    # Wire up a mock usage gate
    gate = MagicMock()
    gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
    cap_call_count = 0

    def detect_side_effect(*args, **kwargs):
        nonlocal cap_call_count
        cap_call_count += 1
        return cap_call_count == 1  # cap hit first time only

    gate.detect_cap_hit = MagicMock(side_effect=detect_side_effect)
    agent._usage_gate = gate

    # Build a JSON result the CLI would produce
    cli_result = json.dumps({
        'result': '',
        'session_id': 'sess-1',
        'num_input_tokens': 100,
        'num_output_tokens': 50,
        'structured_output': json.dumps({
            'thinking': 'done',
            'tool_calls': [{'name': 'stage_complete', 'arguments': {}}],
        }),
    })

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(
        cli_result.encode(), b'',
    ))

    messages = [{'role': 'user', 'content': 'test prompt'}]
    tool_schemas: list = []

    with (
        patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=mock_proc),
        patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
    ):
        await agent._call_claude_cli(messages, tool_schemas)

        mock_sleep.assert_called_once_with(_CAP_HIT_COOLDOWN_SECS)
        assert gate.before_invoke.call_count == 2


# ---------------------------------------------------------------------------
# UsageGate lifecycle: confirm_account_ok / on_agent_complete / release_probe_slot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_claude_cli_confirms_account_ok_on_success():
    """_call_claude_cli calls confirm_account_ok and on_agent_complete on success."""
    from unittest.mock import AsyncMock

    config = _make_config(agent_llm_provider='claude-cli', agent_llm_model='opus')

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

    gate = MagicMock()
    gate.before_invoke = AsyncMock(return_value='token-a')
    gate.detect_cap_hit = MagicMock(return_value=False)
    agent._usage_gate = gate

    # Include cost_usd in the CLI JSON response
    cli_result = json.dumps({
        'result': '',
        'session_id': 'sess-1',
        'num_input_tokens': 100,
        'num_output_tokens': 50,
        'cost_usd': 0.0042,
        'structured_output': json.dumps({
            'thinking': 'done',
            'tool_calls': [{'name': 'stage_complete', 'arguments': {}}],
        }),
    })

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(
        cli_result.encode(), b'',
    ))

    messages = [{'role': 'user', 'content': 'test prompt'}]
    tool_schemas: list = []

    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=mock_proc):
        await agent._call_claude_cli(messages, tool_schemas)

    gate.confirm_account_ok.assert_called_once_with('token-a')
    gate.on_agent_complete.assert_called_once_with(0.0042)
