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
