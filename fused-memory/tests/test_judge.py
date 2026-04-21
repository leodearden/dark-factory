"""Tests for the LLM-as-judge module (judge.py)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    JudgeVerdict,
    VerdictAction,
    VerdictSeverity,
)
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
    journal.get_halt_states = AsyncMock(return_value=[])
    journal.set_halt = AsyncMock(return_value=None)
    journal.clear_halt = AsyncMock(return_value=None)
    journal.decrement_unhalt_grace = AsyncMock(return_value=0)
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

    from fused_memory.models.reconciliation import (
        ReconciliationRun,
        RunStatus,
        RunType,
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
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.failed,
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
async def test_build_review_prompt_excludes_recent_verdicts(mock_journal):
    """Regression: verdict history must not be serialized into the LLM prompt.

    Including prior verdicts caused a feedback loop where the LLM generated
    'systemic_trend' findings whenever recent verdicts were moderate, keeping
    projects halted indefinitely. The code-side _check_error_trends is the
    intended trend-detection mechanism.
    """

    from fused_memory.models.reconciliation import (
        ReconciliationRun,
        RunStatus,
        RunType,
        StageId,
        StageReport,
    )

    config = _make_judge_config()
    judge = Judge(config=config, journal=mock_journal)
    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-feedback-loop',
        project_id='reify',
        run_type=RunType.full,
        trigger_reason='test',
        started_at=now,
        events_processed=0,
        status=RunStatus.completed,
        stage_reports={
            'memory_consolidator': StageReport(
                stage=StageId.memory_consolidator,
                started_at=now,
                completed_at=now,
                stats={'memories_added': 1},
            ),
        },
    )

    prior_verdicts = [
        JudgeVerdict(
            run_id=f'prior-{i}',
            reviewed_at=now,
            severity=VerdictSeverity.moderate,
            findings=[{'issue': f'finding-{i}'}],
            action_taken=VerdictAction.rollback,
        )
        for i in range(10)
    ]

    prompt = judge._build_review_prompt(
        run, entries=[], recent_verdicts=prior_verdicts,
    )

    assert 'Recent Judge Verdicts' not in prompt
    assert 'trend context' not in prompt
    for v in prior_verdicts:
        assert v.run_id not in prompt


@pytest.mark.asyncio
async def test_verdict_action_taken_persisted_after_mutation_moderate(mock_journal):
    """journal.add_verdict() must be called AFTER action_taken is set to 'rollback'.

    Bug 2: add_verdict is currently called before the severity-based mutation,
    so the DB receives action_taken='none' instead of 'rollback' for moderate severity.
    We capture action_taken at call-time via a side_effect to avoid the reference aliasing
    that would mask the bug.
    """
    from datetime import UTC, datetime
    from unittest.mock import patch

    from fused_memory.models.reconciliation import (
        ReconciliationRun,
        RunStatus,
        RunType,
        StageId,
        StageReport,
    )

    config = _make_judge_config()
    judge = Judge(config=config, journal=mock_journal)

    # Set up journal to return a run and entries
    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-bug2-moderate',
        project_id='test-project',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.completed,
        stage_reports={
            'memory_consolidator': StageReport(
                stage=StageId.memory_consolidator,
                started_at=now,
                completed_at=now,
                items_flagged=[],
                stats={},
                llm_calls=1,
                tokens_used=50,
            ),
        },
    )
    mock_journal.get_run = AsyncMock(return_value=run)
    mock_journal.get_run_actions_combined = AsyncMock(return_value=[])

    # Capture action_taken AT CALL TIME, not after reference mutation
    captured_action_taken = []

    async def capture_verdict(verdict):
        captured_action_taken.append(verdict.action_taken)

    mock_journal.add_verdict = AsyncMock(side_effect=capture_verdict)

    # LLM returns a 'moderate' severity verdict
    moderate_response = '{"severity": "moderate", "findings": [{"issue": "bad data"}], "summary": "Moderate problems found."}'

    with patch.object(judge, '_call_llm', AsyncMock(return_value=moderate_response)):
        await judge.review_run('run-bug2-moderate')

    # At call time, action_taken must already be 'rollback' (post-mutation)
    assert len(captured_action_taken) == 1
    assert captured_action_taken[0] == 'rollback', (
        f"Expected action_taken='rollback' at add_verdict call time, "
        f"but got '{captured_action_taken[0]}'. "
        "add_verdict is being called before the severity mutation."
    )


@pytest.mark.asyncio
async def test_verdict_action_taken_persisted_after_mutation_serious(mock_journal):
    """journal.add_verdict() must be called AFTER action_taken is set to 'halt' for serious severity."""
    from datetime import UTC, datetime
    from unittest.mock import patch

    from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType

    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-bug2-serious',
        project_id='test-project',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.completed,
        stage_reports={},
    )
    mock_journal.get_run = AsyncMock(return_value=run)
    mock_journal.get_run_actions_combined = AsyncMock(return_value=[])

    captured_action_taken = []

    async def capture_verdict(verdict):
        captured_action_taken.append(verdict.action_taken)

    mock_journal.add_verdict = AsyncMock(side_effect=capture_verdict)

    serious_response = '{"severity": "serious", "findings": [{"issue": "critical failure"}], "summary": "Serious problems found."}'

    with patch.object(judge, '_call_llm', AsyncMock(return_value=serious_response)):
        await judge.review_run('run-bug2-serious')

    assert len(captured_action_taken) == 1
    assert captured_action_taken[0] == 'halt', (
        f"Expected action_taken='halt' at add_verdict call time, "
        f"but got '{captured_action_taken[0]}'. "
        "add_verdict is being called before the severity mutation."
    )


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


# --- Error trend detection tests ---


def _make_verdicts(severities: list[str], *, spacing_seconds: float = 60.0):
    """Build a list of JudgeVerdict objects, oldest first.

    Successive verdicts are spaced ``spacing_seconds`` apart so tests exercising
    time-windowed logic have a well-defined ordering.
    """
    from datetime import timedelta as _td
    base = datetime.now(tz=UTC)
    return [
        JudgeVerdict(
            run_id=f'run-{i}',
            reviewed_at=base + _td(seconds=i * spacing_seconds),
            severity=VerdictSeverity(s),
        )
        for i, s in enumerate(severities)
    ]


@pytest.mark.asyncio
async def test_error_trend_minor_only_does_not_halt(mock_journal):
    """10 minor verdicts should NOT trigger a halt (minor counts as ok for trend)."""
    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['minor'] * 10)
    await judge._check_error_trends('proj', verdicts)

    assert not judge.is_halted('proj')


@pytest.mark.asyncio
async def test_error_trend_moderates_consecutive_most_recent_halt(mock_journal):
    """5 moderates, with the 3 most recent all moderate, should halt."""
    config = _make_judge_config(
        halt_on_judge_serious=True,
        halt_trend_moderate_count=5,
        halt_trend_consecutive_required=3,
        halt_trend_window_hours=24.0,
    )
    judge = Judge(config=config, journal=mock_journal)

    # ok, ok, ok, ok, ok, M, M, M, M, M (oldest → newest)
    # newest three are moderate → consecutive check passes; 5 moderates in window.
    verdicts = _make_verdicts(['ok'] * 5 + ['moderate'] * 5)
    await judge._check_error_trends('proj', verdicts)

    assert judge.is_halted('proj')
    mock_journal.set_halt.assert_called_once()


@pytest.mark.asyncio
async def test_error_trend_moderates_old_does_not_halt(mock_journal):
    """5 moderates followed by ok should NOT halt — the ok breaks the consecutive streak."""
    config = _make_judge_config(
        halt_on_judge_serious=True,
        halt_trend_moderate_count=5,
        halt_trend_consecutive_required=3,
        halt_trend_window_hours=24.0,
    )
    judge = Judge(config=config, journal=mock_journal)

    # Moderates are old, ok is newest → consecutive-most-recent check bails.
    # This is the core of the self-latching-halt fix.
    verdicts = _make_verdicts(['moderate'] * 5 + ['ok'] * 1)
    await judge._check_error_trends('proj', verdicts)

    assert not judge.is_halted('proj')


@pytest.mark.asyncio
async def test_error_trend_mixed_moderate_serious_halt(mock_journal):
    """Mix of moderate+serious, consecutive most recent, triggers halt."""
    config = _make_judge_config(
        halt_on_judge_serious=True,
        halt_trend_moderate_count=5,
        halt_trend_consecutive_required=3,
        halt_trend_window_hours=24.0,
    )
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['ok'] * 5 + ['moderate'] * 3 + ['serious'] * 2)
    await judge._check_error_trends('proj', verdicts)

    assert judge.is_halted('proj')


@pytest.mark.asyncio
async def test_error_trend_under_count_no_halt(mock_journal):
    """4 moderate + 6 ok should NOT trigger a halt (below count threshold)."""
    config = _make_judge_config(
        halt_on_judge_serious=True,
        halt_trend_moderate_count=5,
        halt_trend_consecutive_required=3,
        halt_trend_window_hours=24.0,
    )
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['ok'] * 6 + ['moderate'] * 4)
    await judge._check_error_trends('proj', verdicts)

    assert not judge.is_halted('proj')


@pytest.mark.asyncio
async def test_error_trend_disabled_config_no_halt(mock_journal):
    """Even 10 moderate verdicts should not halt when halt_on_judge_serious=False."""
    config = _make_judge_config(halt_on_judge_serious=False)
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['moderate'] * 10)
    await judge._check_error_trends('proj', verdicts)

    assert not judge.is_halted('proj')


@pytest.mark.asyncio
async def test_error_trend_outside_window_no_halt(mock_journal):
    """Moderates outside the time window are ignored even if consecutive-most-recent."""
    from datetime import timedelta as _td
    config = _make_judge_config(
        halt_on_judge_serious=True,
        halt_trend_moderate_count=5,
        halt_trend_consecutive_required=3,
        halt_trend_window_hours=6.0,
    )
    judge = Judge(config=config, journal=mock_journal)

    # All verdicts 48 hours old — outside the 6h window → no halt
    old = datetime.now(tz=UTC) - _td(hours=48)
    verdicts = [
        JudgeVerdict(
            run_id=f'run-{i}',
            reviewed_at=old + _td(seconds=i * 60),
            severity=VerdictSeverity.moderate,
        )
        for i in range(10)
    ]
    await judge._check_error_trends('proj', verdicts)

    assert not judge.is_halted('proj')


@pytest.mark.asyncio
async def test_grace_cycles_suppress_trend_check(mock_journal):
    """After unhalt seeds grace, trend check is skipped until grace is consumed."""
    config = _make_judge_config(
        halt_on_judge_serious=True,
        halt_trend_moderate_count=5,
        halt_trend_consecutive_required=3,
        halt_trend_window_hours=24.0,
        halt_grace_cycles=2,
    )
    judge = Judge(config=config, journal=mock_journal)

    # Seed: halt → unhalt → grace=2
    await judge._apply_halt('proj', reason='seed')
    assert judge.is_halted('proj')

    mock_journal.decrement_unhalt_grace = AsyncMock(side_effect=[1, 0])
    await judge.unhalt('proj')
    assert not judge.is_halted('proj')
    assert judge.unhalt_grace_remaining('proj') == 2

    # Even with a full trend-condition set of verdicts, no halt while grace > 0
    verdicts = _make_verdicts(['ok'] * 5 + ['moderate'] * 5)
    await judge._check_error_trends('proj', verdicts)
    assert not judge.is_halted('proj')

    # Consume one cycle — grace now 1 — still suppressed
    remaining = await judge.consume_grace_cycle('proj')
    assert remaining == 1
    await judge._check_error_trends('proj', verdicts)
    assert not judge.is_halted('proj')

    # Consume the second cycle — grace now 0 — trend check resumes
    remaining = await judge.consume_grace_cycle('proj')
    assert remaining == 0
    await judge._check_error_trends('proj', verdicts)
    assert judge.is_halted('proj')


@pytest.mark.asyncio
async def test_cooldown_suppresses_immediate_rehalt(mock_journal):
    """After a halt, trend check is suppressed during the cooldown window."""
    config = _make_judge_config(
        halt_on_judge_serious=True,
        halt_trend_moderate_count=5,
        halt_trend_consecutive_required=3,
        halt_trend_window_hours=24.0,
        halt_grace_cycles=0,
        halt_cooldown_seconds=3600.0,
    )
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['ok'] * 5 + ['moderate'] * 5)
    await judge._check_error_trends('proj', verdicts)
    assert judge.is_halted('proj')
    first_set_halt = mock_journal.set_halt.call_count

    # Simulate operator unhalt (no grace), cooldown still active from _apply_halt
    judge._halted_projects.discard('proj')
    # Cooldown is still in _halt_cooldown_until
    await judge._check_error_trends('proj', verdicts)
    # No second halt because cooldown suppresses re-halt
    assert not judge.is_halted('proj')
    assert mock_journal.set_halt.call_count == first_set_halt


@pytest.mark.asyncio
async def test_unhalt_invokes_callback_and_journal(mock_journal):
    """unhalt() clears halt, writes journal, and invokes on_unhalt_cb."""
    called_with: list[str] = []

    def cb(project_id: str) -> None:
        called_with.append(project_id)

    config = _make_judge_config(
        halt_on_judge_serious=True, halt_grace_cycles=3, halt_cooldown_seconds=100.0,
    )
    judge = Judge(config=config, journal=mock_journal, on_unhalt_cb=cb)

    await judge._apply_halt('proj', reason='seed')
    assert judge.is_halted('proj')

    await judge.unhalt('proj')
    assert not judge.is_halted('proj')
    assert judge.unhalt_grace_remaining('proj') == 3
    mock_journal.clear_halt.assert_called_once()
    assert called_with == ['proj']


@pytest.mark.asyncio
async def test_initialize_rehydrates_halt_state(mock_journal):
    """Judge.initialize() restores halt state from the journal after a restart."""
    from datetime import timedelta as _td

    now = datetime.now(tz=UTC)
    mock_journal.get_halt_states = AsyncMock(return_value=[
        {
            'project_id': 'halted-proj',
            'halted_at': now,
            'cooldown_until': now + _td(seconds=300),
            'reason': 'persistent halt',
            'unhalted_at': None,
            'unhalt_grace_remaining': 0,
        },
        {
            'project_id': 'grace-proj',
            'halted_at': now - _td(hours=1),
            'cooldown_until': None,
            'reason': '',
            'unhalted_at': now - _td(minutes=5),
            'unhalt_grace_remaining': 2,
        },
    ])

    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)
    await judge.initialize()

    assert judge.is_halted('halted-proj')
    assert not judge.is_halted('grace-proj')
    assert judge.unhalt_grace_remaining('grace-proj') == 2


# ---------------------------------------------------------------------------
# Delegation to invoke_with_cap_retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_judge_cli_delegates_to_invoke_with_cap_retry(mock_journal):
    """_call_judge_cli delegates to invoke_with_cap_retry (no output_schema — free-form text).

    Verifies the essential delegation contract: prompt, system_prompt, model,
    usage_gate, and timeout are wired through correctly.  Fine-grained knobs
    (max_turns, permission_mode, disallowed_tools) are implementation details
    covered by shared/tests/test_cli_invoke.py.
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    fake_gate = MagicMock()
    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
    )
    judge = Judge(config=config, journal=mock_journal, usage_gate=fake_gate)

    fake_result = AgentResult(
        success=True,
        output='{"severity": "ok", "findings": []}',
        session_id='jsess-1',
    )

    with patch(
        'fused_memory.reconciliation.judge.invoke_with_cap_retry',
        new_callable=AsyncMock,
    ) as mock_invoke:
        mock_invoke.return_value = fake_result

        result = await judge._call_judge_cli('Evaluate this run.')

    mock_invoke.assert_called_once()
    call_kwargs = mock_invoke.call_args.kwargs
    call_positional = mock_invoke.call_args.args

    from pathlib import Path

    # Essential contract assertions
    assert call_kwargs['prompt'] == 'Evaluate this run.'
    assert call_kwargs['system_prompt'] == JUDGE_SYSTEM_PROMPT
    assert call_kwargs['model'] == config.judge_llm_model
    assert call_kwargs['timeout_seconds'] == float(config.judge_cli_timeout_seconds)
    assert 'output_schema' not in call_kwargs  # judge output is free-form, no schema
    assert call_kwargs['cwd'] == Path(config.explore_codebase_root)

    # usage_gate may be positional or keyword — accept either
    if 'usage_gate' in call_kwargs:
        assert call_kwargs['usage_gate'] is fake_gate
    else:
        assert call_positional[0] is fake_gate

    assert result == '{"severity": "ok", "findings": []}'


@pytest.mark.asyncio
async def test_call_judge_cli_empty_output_returns_empty_string(mock_journal):
    """Empty-stdout from CLI (error_empty_output subtype) returns '' instead of raising.

    _parse_claude_output in shared maps empty stdout → success=False /
    subtype='error_empty_output'.  The judge preserves the prior subprocess
    contract: exit-0 + empty stdout was treated as a valid empty verdict.
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    fake_gate = MagicMock()
    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
    )
    judge = Judge(config=config, journal=mock_journal, usage_gate=fake_gate)

    # Simulate what shared returns when the CLI produces no output
    empty_output_result = AgentResult(
        success=False,
        output='Agent produced no output',
        subtype='error_empty_output',
        session_id='jsess-1',
    )

    with patch(
        'fused_memory.reconciliation.judge.invoke_with_cap_retry',
        new_callable=AsyncMock,
        return_value=empty_output_result,
    ):
        result = await judge._call_judge_cli('Evaluate this run.')

    assert result == ''


@pytest.mark.asyncio
async def test_call_judge_cli_forwards_cwd_to_invoke_claude_agent(mock_journal, tmp_path):
    """_call_judge_cli passes cwd all the way through to invoke_claude_agent.

    Step-9 (b): patches invoke_claude_agent (one level below invoke_with_cap_retry)
    so the kwargs-forwarding layer is exercised.  Omitting cwd from the
    invoke_with_cap_retry call would raise TypeError at runtime; this test
    catches that regression without requiring a live Claude CLI.
    """
    from pathlib import Path
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    # Use tmp_path so the cwd Path actually exists on disk.
    explore_root = str(tmp_path)
    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
        explore_codebase_root=explore_root,
    )

    fake_result = AgentResult(
        success=True,
        output='{"severity": "ok"}',
        session_id='jsess-cwd',
    )

    # Patch at the level below invoke_with_cap_retry — this exercises the
    # kwargs-forwarding path that the higher-level mock skips.
    # usage_gate=None takes the fast path in invoke_with_cap_retry
    # (single invocation, no cap retry), so the real forwarding code runs.
    with patch(
        'shared.cli_invoke.invoke_claude_agent',
        new_callable=AsyncMock,
    ) as mock_agent:
        mock_agent.return_value = fake_result

        judge = Judge(config=config, journal=mock_journal, usage_gate=None)
        await judge._call_judge_cli('Evaluate this run.')

    mock_agent.assert_called_once()
    call_kwargs = mock_agent.call_args.kwargs
    assert call_kwargs['cwd'] == Path(explore_root)


# ---------------------------------------------------------------------------
# CLI failure path surfaces stderr + summary in RuntimeError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_judge_cli_failure_surfaces_stderr_and_summary_in_runtime_error(
    mock_journal,
):
    """_call_judge_cli embeds stderr and classify_agent_failure summary in the RuntimeError.

    Red test (step-3): the current code raises RuntimeError(f'Claude CLI judge
    failed: {result.output[:500]}').  When result.output is '' (e.g. a JSON
    parse crash), the message is empty and the diagnostic signal lives in
    result.stderr.  This test asserts the RuntimeError message after the fix:
      - starts with 'Claude CLI judge failed:'
      - contains the stderr content ('Traceback: JSONDecodeError in line 42')
      - contains 'error_unexpected' (the subtype, present in diagnostic_detail)
    The subtype is NOT 'error_empty_output', so the early-return on lines
    264-265 does not apply — the not-result.success branch must fire.
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    fake_gate = MagicMock()
    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
    )
    judge = Judge(config=config, journal=mock_journal, usage_gate=fake_gate)

    failing_result = AgentResult(
        success=False,
        output='',
        stderr='Traceback: JSONDecodeError in line 42',
        subtype='error_unexpected',
    )

    with patch(
        'fused_memory.reconciliation.judge.invoke_with_cap_retry',
        new_callable=AsyncMock,
    ) as mock_invoke:
        mock_invoke.return_value = failing_result

        with pytest.raises(RuntimeError) as excinfo:
            await judge._call_judge_cli('Evaluate this run.')

    msg = str(excinfo.value)
    assert msg.startswith('Claude CLI judge failed:'), f'unexpected prefix: {msg!r}'
    assert 'Traceback: JSONDecodeError in line 42' in msg, f'stderr missing from: {msg!r}'
    assert 'error_unexpected' in msg, f'subtype missing from: {msg!r}'
