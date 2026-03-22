"""Tests for the LLM-as-judge module (judge.py)."""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.judge import Judge
from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT


def _make_judge_config(**overrides) -> ReconciliationConfig:
    defaults = {
        'judge_llm_provider': 'anthropic',
        'judge_llm_model': 'claude-3-5-sonnet-20241022',
        'agent_max_steps': 10,
        'agent_max_tokens': 4096,
        'max_mutations_per_stage': 5,
    }
    defaults.update(overrides)
    return ReconciliationConfig(**defaults)


# --- Fake Anthropic SDK response dataclasses ---


@dataclass
class FakeAnthropicTextBlock:
    """Fake Anthropic TextBlock."""
    type: str = 'text'
    text: str = ''


@dataclass
class FakeAnthropicToolUseBlock:
    """Fake Anthropic tool_use block (non-text content block)."""
    type: str = 'tool_use'
    id: str = 'toolu_01'
    name: str = 'some_tool'
    input: dict = field(default_factory=dict)


@dataclass
class FakeAnthropicUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class FakeAnthropicResponse:
    """Fake Anthropic messages.create response."""
    content: list = field(default_factory=list)
    usage: FakeAnthropicUsage = field(default_factory=FakeAnthropicUsage)


# --- Fake OpenAI SDK response dataclasses for judge ---


@dataclass
class FakeJudgeOpenAIMessage:
    """Fake OpenAI message in a chat completion choice."""
    content: str | None


@dataclass
class FakeJudgeOpenAIChoice:
    """Fake single choice in an OpenAI response."""
    message: FakeJudgeOpenAIMessage


@dataclass
class FakeJudgeOpenAIResponse:
    """Fake OpenAI chat completion response."""
    choices: list = field(default_factory=list)


@pytest.fixture
def mock_journal():
    """A mock ReconciliationJournal that returns empty data."""
    journal = MagicMock()
    journal.get_run = AsyncMock(return_value=None)
    journal.get_entries = AsyncMock(return_value=[])
    journal.get_recent_verdicts = AsyncMock(return_value=[])
    journal.add_verdict = AsyncMock(return_value=None)
    return journal


# --- Judge._call_llm anthropic branch tests ---


@pytest.mark.asyncio
async def test_judge_call_llm_anthropic_normal(mock_journal):
    """_call_llm with anthropic provider extracts text from response content."""
    config = _make_judge_config(judge_llm_provider='anthropic')
    judge = Judge(config=config, journal=mock_journal)

    verdict_text = '{"severity": "ok", "findings": [], "summary": "All good."}'
    fake_response = FakeAnthropicResponse(
        content=[FakeAnthropicTextBlock(text=verdict_text)],
    )

    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.messages = mock_messages

    with patch('anthropic.AsyncAnthropic', return_value=mock_client):
        result = await judge._call_llm('Evaluate this run.')

    assert result == verdict_text

    # Verify SDK called with JUDGE_SYSTEM_PROMPT and correct model
    call_kwargs = mock_messages.create.call_args.kwargs
    assert call_kwargs['system'] == JUDGE_SYSTEM_PROMPT
    assert call_kwargs['model'] == config.judge_llm_model
    assert call_kwargs['max_tokens'] == 4096
    assert call_kwargs['messages'] == [{'role': 'user', 'content': 'Evaluate this run.'}]


@pytest.mark.asyncio
async def test_judge_call_llm_anthropic_no_text_blocks(mock_journal):
    """_call_llm with anthropic provider returns empty string when no text blocks."""
    config = _make_judge_config(judge_llm_provider='anthropic')
    judge = Judge(config=config, journal=mock_journal)

    # Response with only a tool_use block — no TextBlocks
    fake_response = FakeAnthropicResponse(
        content=[FakeAnthropicToolUseBlock(type='tool_use', id='t1', name='some_tool')],
    )

    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.messages = mock_messages

    with patch('anthropic.AsyncAnthropic', return_value=mock_client):
        result = await judge._call_llm('Evaluate this run.')

    assert result == ''


@pytest.mark.asyncio
async def test_judge_call_llm_anthropic_mixed_content(mock_journal):
    """_call_llm with anthropic provider returns only first text block when mixed content."""
    config = _make_judge_config(judge_llm_provider='anthropic')
    judge = Judge(config=config, journal=mock_journal)

    # Mixed: tool_use block first, then text block
    fake_response = FakeAnthropicResponse(
        content=[
            FakeAnthropicToolUseBlock(),
            FakeAnthropicTextBlock(text='First text block'),
            FakeAnthropicTextBlock(text='Second text block'),
        ],
    )

    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(return_value=fake_response)
    mock_client = MagicMock()
    mock_client.messages = mock_messages

    with patch('anthropic.AsyncAnthropic', return_value=mock_client):
        result = await judge._call_llm('Evaluate this.')

    # Only first text block's text is returned
    assert result == 'First text block'


# --- Judge._call_llm openai/else branch tests ---


@pytest.mark.asyncio
async def test_judge_call_llm_openai_normal(mock_journal):
    """_call_llm with openai provider returns message.content string."""
    config = _make_judge_config(
        judge_llm_provider='openai',
        judge_llm_model='gpt-4o',
    )
    judge = Judge(config=config, journal=mock_journal)

    verdict_text = '{"severity": "minor", "findings": [], "summary": "Minor issues."}'
    fake_response = FakeJudgeOpenAIResponse(
        choices=[
            FakeJudgeOpenAIChoice(
                message=FakeJudgeOpenAIMessage(content=verdict_text)
            )
        ],
    )

    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(return_value=fake_response)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch('openai.AsyncOpenAI', return_value=mock_client):
        result = await judge._call_llm('Evaluate this run.')

    assert result == verdict_text

    # Verify call args: system message with JUDGE_SYSTEM_PROMPT and user message with prompt
    call_kwargs = mock_completions.create.call_args.kwargs
    assert call_kwargs['model'] == 'gpt-4o'
    messages = call_kwargs['messages']
    assert messages[0] == {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT}
    assert messages[1] == {'role': 'user', 'content': 'Evaluate this run.'}


@pytest.mark.asyncio
async def test_build_review_prompt_handles_error_dict_entries(mock_journal):
    """_build_review_prompt must not crash when stage_reports contains plain dict entries.

    The harness injects {'_error': {...}} dicts into stage_reports on failure
    and during stale-run recovery.  Without the isinstance guard those dicts
    raise AttributeError because they lack .items_flagged / .stats etc.
    """
    from datetime import UTC, datetime

    from fused_memory.models.reconciliation import (
        ReconciliationRun,
        StageId,
        StageReport,
    )

    config = _make_judge_config()
    judge = Judge(config=config, journal=mock_journal)

    # Build a run whose stage_reports contains both a real StageReport and a
    # plain-dict _error entry (as injected by harness on failure / stale recovery)
    now = datetime.now(UTC)
    good_report = StageReport(
        stage=StageId.memory_consolidator,
        started_at=now,
        completed_at=now,
        items_flagged=[],
        stats={'processed': 5},
        llm_calls=2,
        tokens_used=100,
    )
    run = ReconciliationRun(
        id='run-bug1-test',
        project_id='test-project',
        run_type='full',
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status='failed',
        stage_reports={
            'memory_consolidator': good_report,
            '_error': {
                'error_type': 'RuntimeError',
                'error_message': 'stage exploded',
                'failed_stage': 'task_knowledge_sync',
                'traceback': 'Traceback ...',
            },
        },
    )

    # Should NOT raise AttributeError
    prompt = judge._build_review_prompt(run, entries=[], recent_verdicts=[])

    # The prompt should contain data from the valid stage report
    assert 'memory_consolidator' in prompt
    # The prompt should also contain the error dict key
    assert '_error' in prompt


@pytest.mark.asyncio
async def test_judge_call_llm_openai_none_content(mock_journal):
    """_call_llm with openai provider returns empty string when message.content is None."""
    config = _make_judge_config(judge_llm_provider='openai', judge_llm_model='gpt-4o-mini')
    judge = Judge(config=config, journal=mock_journal)

    fake_response = FakeJudgeOpenAIResponse(
        choices=[
            FakeJudgeOpenAIChoice(
                message=FakeJudgeOpenAIMessage(content=None)
            )
        ],
    )

    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(return_value=fake_response)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch('openai.AsyncOpenAI', return_value=mock_client):
        result = await judge._call_llm('Evaluate this run.')

    assert result == ''
