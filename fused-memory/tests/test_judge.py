"""Tests for the LLM-as-judge module (judge.py)."""

import json
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


def _make_verdicts(severities: list[str]):
    """Build a list of JudgeVerdict objects from severity strings."""
    return [
        JudgeVerdict(
            run_id=f'run-{i}',
            reviewed_at=datetime.now(tz=UTC),
            severity=VerdictSeverity(s),
        )
        for i, s in enumerate(severities)
    ]


@pytest.mark.asyncio
async def test_error_trend_minor_only_does_not_halt(mock_journal):
    """5+ minor verdicts should NOT trigger a halt (the bug fix)."""
    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['minor'] * 10)
    await judge._check_error_trends('proj', verdicts)

    assert not judge.is_halted('proj')


@pytest.mark.asyncio
async def test_error_trend_moderate_triggers_halt(mock_journal):
    """5+ moderate verdicts should trigger a halt."""
    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['moderate'] * 5 + ['ok'] * 5)
    await judge._check_error_trends('proj', verdicts)

    assert judge.is_halted('proj')


@pytest.mark.asyncio
async def test_error_trend_mixed_moderate_serious_triggers_halt(mock_journal):
    """Mix of moderate+serious at threshold triggers halt."""
    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['moderate'] * 3 + ['serious'] * 2 + ['ok'] * 5)
    await judge._check_error_trends('proj', verdicts)

    assert judge.is_halted('proj')


@pytest.mark.asyncio
async def test_error_trend_under_threshold_no_halt(mock_journal):
    """4 moderate + 6 ok should NOT trigger a halt."""
    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    verdicts = _make_verdicts(['moderate'] * 4 + ['ok'] * 6)
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


# ---------------------------------------------------------------------------
# UsageGate lifecycle: confirm_account_ok / on_agent_complete / release_probe_slot
# ---------------------------------------------------------------------------


def _make_gated_judge(mock_journal, token: str = 'token-j'):
    """Return (judge, gate) with a mock usage gate pre-wired."""
    config = _make_judge_config(judge_llm_provider='claude-cli', judge_llm_model='sonnet')
    judge = Judge(config=config, journal=mock_journal)
    gate = MagicMock()
    gate.before_invoke = AsyncMock(return_value=token)
    gate.detect_cap_hit = MagicMock(return_value=False)
    judge._usage_gate = gate
    return judge, gate


def _make_judge_success_proc(result_text: str = 'The verdict is ok.', cost_usd: float = 0.0055):
    """Return a mock subprocess that exits 0 with a valid judge CLI JSON response."""
    cli_result = json.dumps({
        'result': result_text,
        'session_id': 'sess-j1',
        'num_input_tokens': 80,
        'num_output_tokens': 40,
        'cost_usd': cost_usd,
    })
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(cli_result.encode(), b''))
    return mock_proc


@pytest.mark.asyncio
async def test_call_judge_cli_confirms_account_ok_on_success(mock_journal):
    """_call_judge_cli calls confirm_account_ok and on_agent_complete on success."""
    judge, gate = _make_gated_judge(mock_journal, token='token-j')
    mock_proc = _make_judge_success_proc(result_text='The verdict is ok.', cost_usd=0.0055)

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
        result = await judge._call_judge_cli('Evaluate this.')

    assert result == 'The verdict is ok.'
    gate.confirm_account_ok.assert_called_once_with('token-j')
    gate.on_agent_complete.assert_called_once_with(0.0055)


@pytest.mark.asyncio
async def test_call_judge_cli_releases_probe_on_file_not_found(mock_journal):
    """release_probe_slot is called when create_subprocess_exec raises FileNotFoundError."""
    judge, gate = _make_gated_judge(mock_journal, token='token-jx')

    with (
        patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError),
        pytest.raises(RuntimeError, match='Claude CLI not found'),
    ):
        await judge._call_judge_cli('Evaluate this.')

    gate.release_probe_slot.assert_called_once_with('token-jx')


@pytest.mark.asyncio
async def test_call_judge_cli_releases_probe_on_timeout(mock_journal):
    """release_probe_slot is called when wait_for raises TimeoutError."""
    judge, gate = _make_gated_judge(mock_journal, token='token-jy')
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(side_effect=TimeoutError())

    with (
        patch('asyncio.create_subprocess_exec', return_value=mock_proc),
        pytest.raises(RuntimeError, match='timed out'),
    ):
        await judge._call_judge_cli('Evaluate this.')

    gate.release_probe_slot.assert_called_once_with('token-jy')


@pytest.mark.asyncio
async def test_call_judge_cli_no_confirm_on_cap_hit(mock_journal):
    """confirm_account_ok is NOT called on the cap-hit iteration, only on success."""
    judge, gate = _make_gated_judge(mock_journal)
    gate.before_invoke = AsyncMock(side_effect=['token-ja', 'token-jb'])

    cap_call_count = 0

    def detect_side_effect(*args, **kwargs):
        nonlocal cap_call_count
        cap_call_count += 1
        return cap_call_count == 1  # cap hit first time only

    gate.detect_cap_hit = MagicMock(side_effect=detect_side_effect)
    mock_proc = _make_judge_success_proc(result_text='Judge output.', cost_usd=0.0099)

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
        result = await judge._call_judge_cli('Evaluate this.')

    assert result == 'Judge output.'
    # Exactly once — for the success iteration, NOT for the cap-hit iteration
    gate.confirm_account_ok.assert_called_once_with('token-jb')
    gate.on_agent_complete.assert_called_once_with(0.0099)
    gate.release_probe_slot.assert_not_called()


@pytest.mark.asyncio
async def test_call_judge_cli_releases_probe_on_nonzero_exit(mock_journal):
    """release_probe_slot is called when the CLI exits with a non-zero return code."""
    judge, gate = _make_gated_judge(mock_journal, token='token-jz')
    mock_proc = AsyncMock()
    mock_proc.returncode = 2
    mock_proc.communicate = AsyncMock(return_value=(b'', b'judge error output'))

    with (
        patch('asyncio.create_subprocess_exec', return_value=mock_proc),
        pytest.raises(RuntimeError, match='exited with code 2'),
    ):
        await judge._call_judge_cli('Evaluate this.')

    gate.release_probe_slot.assert_called_once_with('token-jz')
    gate.confirm_account_ok.assert_not_called()


@pytest.mark.asyncio
async def test_call_judge_cli_empty_stdout_confirms_gate(mock_journal):
    """Empty stdout (exit 0) calls confirm_account_ok and on_agent_complete but not release_probe_slot."""
    judge, gate = _make_gated_judge(mock_journal, token='token-je')
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b'', b''))

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
        result = await judge._call_judge_cli('Evaluate this.')

    assert result == ''
    gate.confirm_account_ok.assert_called_once_with('token-je')
    gate.on_agent_complete.assert_called_once_with(0.0)
    gate.release_probe_slot.assert_not_called()
    mock_journal.write.assert_not_called()


@pytest.mark.asyncio
async def test_call_judge_cli_empty_stdout_with_stderr_confirms_gate(mock_journal):
    """Empty stdout (exit 0) still confirms gate even when stderr contains warnings."""
    judge, gate = _make_gated_judge(mock_journal, token='token-jw')
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b'', b'some warning'))

    with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
        result = await judge._call_judge_cli('Evaluate this.')

    assert result == ''
    gate.confirm_account_ok.assert_called_once_with('token-jw')
    gate.on_agent_complete.assert_called_once_with(0.0)
    gate.release_probe_slot.assert_not_called()
