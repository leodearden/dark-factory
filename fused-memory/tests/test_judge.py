"""Tests for the LLM-as-judge module (judge.py)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.testing import make_gate_mock

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    JudgeVerdict,
    VerdictAction,
    VerdictSeverity,
)
from fused_memory.reconciliation.judge import Judge, is_phantom_verdict
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


# --- Judge._parse_verdict tests ---


def test_parse_verdict_unparseable_response_is_loud_not_fabricated(mock_journal):
    """Unparseable judge output must produce a loud, honest verdict, not a
    fabricated fail-soft neutral one.

    A systemic judge/CLI outage should not hide behind a default 'minor'
    verdict with a generic 'Manual review recommended' recommendation —
    that lets the outage silently degrade review quality. It must surface
    as severity=serious (routing through the existing halt path) with a
    finding that honestly names the parse failure.
    """
    judge = Judge(config=_make_judge_config(), journal=mock_journal)

    verdict = judge._parse_verdict('not valid json at all {{{', 'run-unparseable')

    assert verdict.severity == VerdictSeverity.serious
    assert len(verdict.findings) == 1
    assert verdict.findings[0].get('recommendation') != 'Manual review recommended'
    assert 'Judge response could not be parsed' in verdict.findings[0].get('issue', '')


def test_parse_verdict_text_path_invalid_severity_is_loud_not_silent(mock_journal):
    """A semantically-invalid text payload must fail LOUD, not vanish silently.

    Valid JSON, invalid enum member.  JudgeVerdict construction raises
    pydantic.ValidationError, which is NOT in _parse_verdict's caught tuple, so
    today it escapes to review_run's broad `except Exception` and the run silently
    returns None — no verdict, no halt, no signal.  That violates the repo's
    loud-over-silent-degradation norm.  It must instead get the same fail-closed
    treatment as malformed text: severity=serious with the
    code='unparseable_judge_response' marker (which review_run turns into the
    truthful 'Unparseable judge response' halt reason).

    Scoped to the RETAINED anthropic/openai text fallback — the claude_cli
    provider classifies this same condition as cli_output_unparseable infra in
    _call_judge_cli and never reaches here.
    """
    judge = Judge(config=_make_judge_config(), journal=mock_journal)

    verdict = judge._parse_verdict('{"severity": "catastrophic", "findings": []}', 'run-z')

    assert verdict.severity == VerdictSeverity.serious
    assert len(verdict.findings) == 1
    assert verdict.findings[0].get('code') == 'unparseable_judge_response'


@pytest.mark.parametrize('response', [
    '[{"severity": "ok", "findings": []}]',   # verdict wrapped in a list
    '"ok"',                                   # bare JSON string
    '5',                                      # bare JSON number
    'null',                                   # JSON null
])
def test_parse_verdict_text_path_non_object_json_is_loud_not_silent(mock_journal, response):
    """A top-level JSON value that is not an OBJECT must fail loud, not vanish.

    Same silent-degradation class as the invalid-severity case above, reached via
    a different exception type: these all parse cleanly at ``json.loads``, then
    die on ``data.get(...)`` with an ``AttributeError`` — which was NOT in
    _parse_verdict's caught tuple, so it escaped to review_run's broad
    ``except Exception`` and the run returned None with no verdict, no halt and
    no signal.

    Every shape here is plausible output from a schema-less SDK provider asked
    for JSON (list-wrapping is a routine LLM habit), and the ``anthropic`` /
    ``openai`` branches have no ``--json-schema`` mechanism to prevent it.  The
    ``claude_cli`` provider classifies the same condition as
    cli_output_unparseable infra in _call_judge_cli and never reaches here.
    """
    judge = Judge(config=_make_judge_config(), journal=mock_journal)

    verdict = judge._parse_verdict(response, 'run-nonobj')

    assert verdict.severity == VerdictSeverity.serious
    assert len(verdict.findings) == 1
    assert verdict.findings[0].get('code') == 'unparseable_judge_response'


def test_parse_verdict_accepts_structured_payload_dict(mock_journal):
    """_parse_verdict accepts an already-structured payload dict and skips text parsing.

    The claude_cli provider now hands _parse_verdict the validated
    ``structured_output`` payload instead of prose, so the fence-stripping +
    json.loads text path must not be involved.  Proven by the ``findings`` value:
    a list-of-dicts survives verbatim, which no text-parsing route could produce
    from a dict input (it would raise AttributeError on ``.strip()``).  The
    ``summary`` key is dropped on construction, exactly as on the text path.
    """
    judge = Judge(config=_make_judge_config(), journal=mock_journal)

    verdict = judge._parse_verdict(
        {'severity': 'moderate', 'findings': [{'issue': 'y'}], 'summary': 'ignored'},
        'run-x',
    )

    assert verdict.severity == VerdictSeverity.moderate
    assert verdict.findings == [{'issue': 'y'}]
    assert verdict.action_taken == VerdictAction.none
    assert verdict.run_id == 'run-x'


@pytest.mark.parametrize('empty_text', ['', '   ', '\n\t '])
def test_parse_verdict_empty_response_stays_benign_not_serious(mock_journal, empty_text):
    """Empty (or whitespace-only) judge output must NOT be treated as a loud
    parse failure.

    _call_judge_cli returns '' for two documented benign cases: exit-0/
    empty-stdout CLI runs (legacy 'empty stdout = valid empty verdict'
    semantics) and the anthropic branch when the response has no text
    blocks. json.loads('') raises JSONDecodeError just like genuinely
    malformed text, so without an explicit short-circuit this benign,
    successful case would be indistinguishable from a systemic judge/CLI
    outage and incorrectly escalate to severity=serious (which halts the
    project when halt_on_judge_serious=True). It must stay non-halting.
    """
    judge = Judge(config=_make_judge_config(), journal=mock_journal)

    verdict = judge._parse_verdict(empty_text, 'run-empty')

    assert verdict.severity == VerdictSeverity.minor
    assert verdict.action_taken == VerdictAction.none


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
async def test_review_run_unparseable_response_halts_when_enabled(mock_journal):
    """A single unparseable judge response halts the project immediately when
    halt_on_judge_serious=True.

    This is the behaviorally significant consequence of the loud
    severity=serious parse-failure verdict: review_run() acts on it the
    same way it would a genuine serious verdict, without waiting for
    _check_error_trends' multi-occurrence window. One malformed judge/CLI
    response is deliberately treated as sufficient grounds to halt — see
    the comment in Judge._parse_verdict's except block.
    """
    from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType

    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-parse-halt',
        project_id='test-project-parse-halt',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.completed,
        stage_reports={},
    )
    mock_journal.get_run = AsyncMock(return_value=run)
    mock_journal.get_run_actions_combined = AsyncMock(return_value=[])

    with patch.object(
        judge, '_call_llm', AsyncMock(return_value='not valid json at all {{{')
    ):
        verdict = await judge.review_run('run-parse-halt')

    assert verdict is not None
    assert verdict.severity == VerdictSeverity.serious
    assert verdict.action_taken == VerdictAction.halt
    assert judge.is_halted('test-project-parse-halt')
    mock_journal.set_halt.assert_called_once()


@pytest.mark.asyncio
async def test_review_run_unparseable_response_no_halt_when_disabled(mock_journal):
    """An unparseable judge response still yields a loud serious verdict when
    halt_on_judge_serious=False, but does NOT halt the project.

    The severity itself stays honest/serious regardless of config — only
    the *action* (halting) is gated on halt_on_judge_serious, same as for
    a genuine serious verdict.
    """
    from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType

    config = _make_judge_config(halt_on_judge_serious=False)
    judge = Judge(config=config, journal=mock_journal)

    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-parse-no-halt',
        project_id='test-project-parse-no-halt',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.completed,
        stage_reports={},
    )
    mock_journal.get_run = AsyncMock(return_value=run)
    mock_journal.get_run_actions_combined = AsyncMock(return_value=[])

    with patch.object(
        judge, '_call_llm', AsyncMock(return_value='not valid json at all {{{')
    ):
        verdict = await judge.review_run('run-parse-no-halt')

    assert verdict is not None
    assert verdict.severity == VerdictSeverity.serious
    assert verdict.action_taken == VerdictAction.none
    assert not judge.is_halted('test-project-parse-no-halt')
    mock_journal.set_halt.assert_not_called()


@pytest.mark.asyncio
async def test_review_run_empty_response_does_not_halt_even_when_enabled(mock_journal):
    """An empty judge response must NOT halt the project, even with
    halt_on_judge_serious=True.

    Unlike a genuinely unparseable response (tested above), empty output is
    a documented benign case (see Judge._call_judge_cli's 'empty stdout =
    valid empty verdict' comment and _parse_verdict's empty-text
    short-circuit) — it must stay severity=minor and non-halting instead of
    being conflated with the loud severity=serious parse-failure path.
    """
    from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType

    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-empty-no-halt',
        project_id='test-project-empty-no-halt',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.completed,
        stage_reports={},
    )
    mock_journal.get_run = AsyncMock(return_value=run)
    mock_journal.get_run_actions_combined = AsyncMock(return_value=[])

    with patch.object(judge, '_call_llm', AsyncMock(return_value='')):
        verdict = await judge.review_run('run-empty-no-halt')

    assert verdict is not None
    assert verdict.severity == VerdictSeverity.minor
    assert verdict.action_taken == VerdictAction.none
    assert not judge.is_halted('test-project-empty-no-halt')
    mock_journal.set_halt.assert_not_called()


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


# --- is_phantom_verdict tests ---


class TestIsPhantomVerdict:
    """is_phantom_verdict(verdict) — True iff *verdict* is a fabricated
    stand-in for a review that never happened (built by
    Judge._parse_verdict's except-block fallback when the judge output
    could not be parsed), as opposed to a genuine judge finding.

    This step covers only the structured-marker branch
    (``code == 'unparseable_judge_response'``, the shape written for every
    run after task 2947 landed). The legacy unmarked shape (pre-2947 rows)
    is added by a later step.
    """

    def test_marked_unparseable_finding_is_phantom(self):
        """Real d741ffe3 / 33581299 / ad861d89 / c7d84e9f shape: severity=serious,
        one finding carrying the structured code marker."""
        verdict = JudgeVerdict(
            run_id='run-marked',
            reviewed_at=datetime.now(UTC),
            severity=VerdictSeverity.serious,
            findings=[{
                'issue': (
                    'Judge response could not be parsed: Expecting value: '
                    'line 1 column 1 (char 0)'
                ),
                'severity': 'serious',
                'code': 'unparseable_judge_response',
                'recommendation': (
                    'This verdict was not a real review — the judge output was '
                    'unparseable. Investigate the judge LLM/CLI output before '
                    'trusting subsequent verdicts.'
                ),
            }],
        )
        assert is_phantom_verdict(verdict) is True

    def test_genuine_serious_with_five_findings_and_no_code_is_not_phantom(self):
        """Real bc9459b8 shape: severity=serious, five substantive findings,
        none carrying the code marker."""
        verdict = JudgeVerdict(
            run_id='run-genuine',
            reviewed_at=datetime.now(UTC),
            severity=VerdictSeverity.serious,
            findings=[
                {
                    'issue': f'Substantive content issue #{i}',
                    'severity': 'serious',
                    'recommendation': f'Fix issue #{i}',
                }
                for i in range(5)
            ],
        )
        assert is_phantom_verdict(verdict) is False

    def test_empty_findings_is_not_phantom(self):
        verdict = JudgeVerdict(
            run_id='run-empty-findings',
            reviewed_at=datetime.now(UTC),
            severity=VerdictSeverity.serious,
            findings=[],
        )
        assert is_phantom_verdict(verdict) is False

    def test_ordinary_ok_verdict_is_not_phantom(self):
        verdict = JudgeVerdict(
            run_id='run-ok',
            reviewed_at=datetime.now(UTC),
            severity=VerdictSeverity.ok,
            findings=[],
        )
        assert is_phantom_verdict(verdict) is False


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


@pytest.mark.asyncio
async def test_initialize_rehydrates_halt_reason_and_halted_at(mock_journal):
    """A halt survives a restart WITH its reason and start time (task 3050).

    Both observed halt incidents spanned a restart, so without this the new
    read surface would report a halted project with halt_reason=None and
    halted_at=None — the exact "I can see something is wrong but not why or
    since when" failure the task exists to end. The journal row already
    carries both; initialize() previously dropped them.
    """
    from datetime import timedelta as _td

    now = datetime.now(tz=UTC)
    halted_at = now - _td(hours=48)
    mock_journal.get_halt_states = AsyncMock(return_value=[
        {
            'project_id': 'halted-proj',
            'halted_at': halted_at,
            'cooldown_until': now - _td(hours=47),
            'reason': 'judge-unreachable after 3 consecutive infra failures',
            'unhalted_at': None,
            'unhalt_grace_remaining': 0,
        },
        {
            'project_id': 'unhalted-proj',
            'halted_at': now - _td(hours=2),
            'cooldown_until': None,
            'reason': 'old reason',
            'unhalted_at': now - _td(minutes=5),
            'unhalt_grace_remaining': 1,
        },
        {
            'project_id': 'blank-reason-proj',
            'halted_at': halted_at,
            'cooldown_until': None,
            'reason': '',
            'unhalted_at': None,
            'unhalt_grace_remaining': 0,
        },
    ])

    judge = Judge(config=_make_judge_config(), journal=mock_journal)
    await judge.initialize()

    assert judge.halt_reason('halted-proj') == (
        'judge-unreachable after 3 consecutive infra failures'
    )
    snap = judge.halt_snapshot('halted-proj')
    assert snap['halted'] is True
    assert snap['halted_at'] == halted_at.isoformat()
    assert snap['halt_reason'] == 'judge-unreachable after 3 consecutive infra failures'
    # The incident's signature: halted for 48h with the cooldown long past.
    assert snap['cooldown_expired'] is True

    # An already-unhalted row leaves no halt residue.
    assert judge.halt_reason('unhalted-proj') is None
    assert judge.halt_snapshot('unhalted-proj')['halted_at'] is None

    # A stored empty reason means "unknown", not '' — never dress it up.
    assert judge.halt_reason('blank-reason-proj') is None
    assert judge.halt_snapshot('blank-reason-proj')['halt_reason'] is None
    assert judge.halt_snapshot('blank-reason-proj')['halted'] is True


class TestJudgeCooldownExpired:
    """Judge.cooldown_expired(project_id) — the harness uses this to decide
    auto-resume-after-cooldown when auto_unhalt_after_cooldown is enabled
    (task 2920 deliverable c). True iff the project is halted AND a
    cooldown_until is recorded AND it is now in the past."""

    def test_true_when_halted_and_cooldown_in_past(self, mock_journal):
        from datetime import timedelta

        judge = Judge(config=_make_judge_config(), journal=mock_journal)
        judge._halted_projects.add('proj')
        judge._halt_cooldown_until['proj'] = datetime.now(UTC) - timedelta(hours=1)
        assert judge.cooldown_expired('proj') is True

    def test_false_when_halted_but_cooldown_in_future(self, mock_journal):
        from datetime import timedelta

        judge = Judge(config=_make_judge_config(), journal=mock_journal)
        judge._halted_projects.add('proj')
        judge._halt_cooldown_until['proj'] = datetime.now(UTC) + timedelta(hours=1)
        assert judge.cooldown_expired('proj') is False

    def test_false_when_not_halted_even_with_stale_cooldown(self, mock_journal):
        from datetime import timedelta

        judge = Judge(config=_make_judge_config(), journal=mock_journal)
        # A stale cooldown row exists but the project is NOT halted.
        judge._halt_cooldown_until['proj'] = datetime.now(UTC) - timedelta(hours=1)
        assert judge.cooldown_expired('proj') is False

    def test_false_when_halted_but_no_cooldown_recorded(self, mock_journal):
        judge = Judge(config=_make_judge_config(), journal=mock_journal)
        judge._halted_projects.add('proj')  # no _halt_cooldown_until entry
        assert judge.cooldown_expired('proj') is False


class TestJudgeHaltSnapshot:
    """Judge.halt_snapshot()/halted_projects() — the operator/watcher read
    surface (task 3050 deliverable A).

    Before these existed, a halt was only observable by grepping harness logs:
    a large ``reconciliation_backlog`` looked identical whether the project was
    HALTED (remedy: unhalt_reconciliation) or merely unable to keep up (remedy:
    capacity). The snapshot is JSON-ready — datetimes render as ISO-8601
    strings — because its only consumers are MCP tool payloads.
    """

    @pytest.mark.asyncio
    async def test_snapshot_after_apply_halt_carries_reason_and_times(self, mock_journal):
        judge = Judge(config=_make_judge_config(), journal=mock_journal)
        before = datetime.now(UTC)
        await judge._apply_halt('proj', reason='Serious verdict in run r1')

        snap = judge.halt_snapshot('proj')

        assert snap['project_id'] == 'proj'
        assert snap['halted'] is True
        assert snap['halt_reason'] == 'Serious verdict in run r1'
        assert snap['cooldown_expired'] is False
        assert snap['unhalt_grace_remaining'] == 0
        # JSON-ready: ISO-8601 strings, NOT datetimes.
        assert isinstance(snap['halted_at'], str)
        assert isinstance(snap['cooldown_until'], str)
        halted_at = datetime.fromisoformat(snap['halted_at'])
        assert before <= halted_at <= datetime.now(UTC)
        # cooldown_until is halted_at + the configured cooldown.
        cooldown_until = datetime.fromisoformat(snap['cooldown_until'])
        assert cooldown_until > halted_at

    def test_unknown_project_reports_not_halted_without_raising(self, mock_journal):
        judge = Judge(config=_make_judge_config(), journal=mock_journal)

        snap = judge.halt_snapshot('never-seen')

        assert snap['project_id'] == 'never-seen'
        assert snap['halted'] is False
        assert snap['halt_reason'] is None
        assert snap['halted_at'] is None
        assert snap['cooldown_until'] is None
        assert snap['cooldown_expired'] is False
        assert snap['unhalt_grace_remaining'] == 0

    @pytest.mark.asyncio
    async def test_snapshot_after_unhalt_clears_halt_fields_and_seeds_grace(
        self, mock_journal
    ):
        config = _make_judge_config()
        judge = Judge(config=config, journal=mock_journal)
        await judge._apply_halt('proj', reason='Serious verdict in run r1')

        await judge.unhalt('proj')
        snap = judge.halt_snapshot('proj')

        assert snap['halted'] is False
        assert snap['halt_reason'] is None
        assert snap['halted_at'] is None
        assert snap['cooldown_until'] is None
        assert snap['cooldown_expired'] is False
        assert snap['unhalt_grace_remaining'] == config.halt_grace_cycles

    @pytest.mark.asyncio
    async def test_halted_projects_is_sorted_and_tracks_unhalt(self, mock_journal):
        judge = Judge(config=_make_judge_config(), journal=mock_journal)

        assert judge.halted_projects() == []

        await judge._apply_halt('b', reason='r-b')
        await judge._apply_halt('a', reason='r-a')
        assert judge.halted_projects() == ['a', 'b']

        await judge.unhalt('a')
        assert judge.halted_projects() == ['b']

    @pytest.mark.asyncio
    async def test_snapshot_reports_cooldown_expired(self, mock_journal):
        """The incident's actual signature: halted, cooldown long past, and
        nothing acted on it."""
        from datetime import timedelta

        judge = Judge(config=_make_judge_config(), journal=mock_journal)
        await judge._apply_halt('proj', reason='judge-unreachable')
        # Seed a past cooldown directly, mirroring TestJudgeCooldownExpired.
        judge._halt_cooldown_until['proj'] = datetime.now(UTC) - timedelta(hours=1)

        snap = judge.halt_snapshot('proj')

        assert snap['halted'] is True
        assert snap['cooldown_expired'] is True
        assert snap['halt_reason'] == 'judge-unreachable'


# ---------------------------------------------------------------------------
# Delegation to invoke_with_cap_retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_judge_cli_delegates_to_invoke_with_cap_retry(mock_journal):
    """_call_judge_cli delegates to invoke_with_cap_retry, carrying the verdict schema.

    Verifies the essential delegation contract: prompt, system_prompt, model,
    usage_gate, output_schema, and timeout are wired through correctly.
    Fine-grained knobs (max_turns, permission_mode, disallowed_tools) are
    implementation details covered by shared/tests/test_cli_invoke.py and, for
    the schema-carrying invariants specifically, by
    test_call_judge_cli_passes_judge_verdict_schema below.
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    from fused_memory.reconciliation.judge import JUDGE_VERDICT_SCHEMA

    fake_gate = make_gate_mock()
    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
    )
    judge = Judge(config=config, journal=mock_journal, usage_gate=fake_gate)

    fake_result = AgentResult(
        success=True,
        output='{"severity": "ok", "findings": []}',
        structured_output={'severity': 'ok', 'findings': []},
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
    assert call_kwargs['output_schema'] is JUDGE_VERDICT_SCHEMA
    assert call_kwargs['cwd'] == Path(config.explore_codebase_root)

    # usage_gate may be positional or keyword — accept either
    if 'usage_gate' in call_kwargs:
        assert call_kwargs['usage_gate'] is fake_gate
    else:
        assert call_positional[0] is fake_gate

    # The verdict comes from the structured payload, not the text channel.
    assert result == {'severity': 'ok', 'findings': []}


@pytest.mark.asyncio
async def test_call_judge_cli_passes_judge_verdict_schema(mock_journal):
    """The judge's verdict contract rides ``output_schema``, not the system prompt.

    This is the whole point of the migration: ``build_claude_argv`` emits
    ``--json-schema`` UNCONDITIONALLY (cli_invoke.py:1552-1553) while it drops
    ``--system-prompt-file`` on the resume path (:1501-1503).  A prose-only
    contract is therefore droppable across a cap-retry resume; a schema-carried
    one is not.  Pinned here at the invocation boundary:

    - ``output_schema is JUDGE_VERDICT_SCHEMA`` — the contract is attached.
    - ``disallowed_tools == ['*']`` — the judge keeps passing the wildcard
      VERBATIM.  Expanding it into the ``StructuredOutput``-preserving
      real-builtins deny-list is cli_invoke's job (:1533-1536), not the
      caller's, so the judge inherits future central fixes instead of pinning a
      stale copy of the CLI's built-in list.
    - ``max_turns >= 3`` — the schema mechanism burns a tool-use turn, so
      ``max_turns=1`` is incompatible with ``--json-schema``
      (task_curator.py:2366-2372); 3 is the floor both migrated siblings use.
    - ``system_prompt`` is unchanged — JUDGE_SYSTEM_PROMPT's "## Output Format"
      block stays because it is the ONLY output contract the anthropic/openai
      provider branches have (they never see ``--json-schema``).
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    from fused_memory.reconciliation.judge import JUDGE_VERDICT_SCHEMA

    fake_gate = make_gate_mock()
    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
    )
    judge = Judge(config=config, journal=mock_journal, usage_gate=fake_gate)

    fake_result = AgentResult(
        success=True,
        output='',
        structured_output={'severity': 'ok', 'findings': []},
        session_id='jsess-1',
    )

    with patch(
        'fused_memory.reconciliation.judge.invoke_with_cap_retry',
        new_callable=AsyncMock,
        return_value=fake_result,
    ) as mock_invoke:
        await judge._call_judge_cli('Evaluate this run.')

    call_kwargs = mock_invoke.call_args.kwargs
    assert call_kwargs['output_schema'] is JUDGE_VERDICT_SCHEMA
    assert call_kwargs['disallowed_tools'] == ['*']
    assert call_kwargs['max_turns'] >= 3, (
        'max_turns=1 is incompatible with --json-schema: the schema mechanism '
        'burns a tool-use turn and the CLI returns error_max_turns even when '
        'the payload is attached (task_curator.py:2366-2372)'
    )
    assert call_kwargs['system_prompt'] == JUDGE_SYSTEM_PROMPT


def test_judge_verdict_schema_shape():
    """JUDGE_VERDICT_SCHEMA matches the verdict the judge is asked to produce.

    The severity enum is pinned against a LITERAL set, not against
    ``VerdictSeverity``: the schema builds its enum from that very enum
    (``_VERDICT_SEVERITY_VALUES``), so asserting the two agree can never fail and
    is not coverage.  Pinning the literal is what actually catches something —
    either a silent widening of VerdictSeverity or the schema quietly ceasing to
    derive from it.
    """
    from fused_memory.reconciliation.judge import JUDGE_VERDICT_SCHEMA

    assert JUDGE_VERDICT_SCHEMA['type'] == 'object'
    assert 'severity' in JUDGE_VERDICT_SCHEMA['required']
    assert 'findings' in JUDGE_VERDICT_SCHEMA['required']

    properties = JUDGE_VERDICT_SCHEMA['properties']
    assert set(properties['severity']['enum']) == {'ok', 'minor', 'moderate', 'serious'}
    assert properties['findings']['type'] == 'array'


@pytest.mark.asyncio
async def test_call_judge_cli_returns_structured_payload_ignoring_prose(mock_journal):
    """The verdict is read from structured_output — model prose is NOT the verdict.

    ``output`` here is the literal 2026-07-25 transcript shape: conversational
    prose with an italic stage-direction, produced after a cap-retry resume
    dropped the system prompt.  Parsing THAT as the verdict is what fabricated a
    ``severity=serious`` halt for a run nobody had reviewed.  The payload wins.
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
    )
    judge = Judge(config=config, journal=mock_journal, usage_gate=make_gate_mock())

    payload = {'severity': 'moderate', 'findings': [{'issue': 'x', 'severity': 'moderate'}]}
    result = AgentResult(
        success=True,
        output="I'll check the repo for context...\n\n*Searches project files*",
        structured_output=payload,
    )

    with patch(
        'fused_memory.reconciliation.judge.invoke_with_cap_retry',
        new=AsyncMock(return_value=result),
    ):
        assert await judge._call_judge_cli('p') == payload


@pytest.mark.asyncio
async def test_call_judge_cli_json_string_structured_output_is_loaded(mock_journal):
    """A structured_output delivered as a JSON *string* is json.loads-ed to a dict.

    Parity with agent_loop.py:388-392, the sibling that already consumes
    ``structured_output`` from this same shared path.
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
    )
    judge = Judge(config=config, journal=mock_journal, usage_gate=make_gate_mock())

    result = AgentResult(
        success=True,
        output='',
        structured_output='{"severity": "ok", "findings": []}',
    )

    with patch(
        'fused_memory.reconciliation.judge.invoke_with_cap_retry',
        new=AsyncMock(return_value=result),
    ):
        assert await judge._call_judge_cli('p') == {'severity': 'ok', 'findings': []}


@pytest.mark.asyncio
async def test_call_judge_cli_empty_output_returns_empty_string(mock_journal):
    """Empty-stdout from CLI (error_empty_output subtype) returns '' instead of raising.

    _parse_claude_output in shared maps empty stdout → success=False /
    subtype='error_empty_output'.  The judge preserves the prior subprocess
    contract: exit-0 + empty stdout was treated as a valid empty verdict.
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    fake_gate = make_gate_mock()
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
async def test_call_judge_cli_input_rejected_raises_infra_error(mock_journal):
    """A pre-turn CLI rejection is INFRA, not a benign empty verdict (task 3143).

    The legacy "empty stdout = valid empty verdict" contract is scoped to a
    genuine EMPTY_OUTPUT: we asked the judge something and it returned nothing.
    A rejection means the judge was never asked at all — returning '' there
    would have _parse_verdict fabricate a benign severity=minor verdict for a
    review that never happened, exactly the silent degradation this module
    exists to prevent.  invoke_with_cap_retry has already spent its one
    automatic retry by the time we get here, so raising is the right escalation.
    """
    from unittest.mock import AsyncMock

    from shared.cli_invoke import AgentResult

    from fused_memory.reconciliation.judge import JudgeInfraError

    fake_gate = make_gate_mock()
    config = _make_judge_config(
        judge_llm_provider='claude_cli',
        judge_llm_model='claude-sonnet-4-5',
    )
    judge = Judge(config=config, journal=mock_journal, usage_gate=fake_gate)

    rejected_result = AgentResult(
        success=False,
        output='Agent produced no output',
        subtype='error_cli_input_rejected',
        stderr=(
            'Error: Input must be provided either through stdin or as a '
            'prompt argument when using --print\n'
        ),
        session_id='jsess-2',
    )

    with patch(
        'fused_memory.reconciliation.judge.invoke_with_cap_retry',
        new_callable=AsyncMock,
        return_value=rejected_result,
    ), pytest.raises(JudgeInfraError) as excinfo:
        await judge._call_judge_cli('Evaluate this run.')

    assert 'Input must be provided' in str(excinfo.value), (
        f'the raised message must carry the CLI stderr cause, got {excinfo.value!r}'
    )


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

    # A usable structured payload: a success carrying none is now classified as
    # cli_output_empty infra, which would mask this test's kwargs assertion.
    fake_result = AgentResult(
        success=True,
        output='',
        structured_output={'severity': 'ok', 'findings': []},
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
# CLI failure path surfaces stderr + summary in JudgeInfraError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_judge_cli_failure_surfaces_stderr_and_summary_in_judge_infra_error(
    mock_journal,
):
    """_call_judge_cli embeds stderr and classify_agent_failure summary in the JudgeInfraError.

    Task 2947 (ask a) replaced the previous generic RuntimeError on the
    not-result.success path with a typed JudgeInfraError (transport/infra
    signal). The message is still build_failure_message('Claude CLI judge',
    result), so when result.output is '' (e.g. a JSON parse crash) the
    diagnostic signal still lives in result.stderr. This test asserts the
    JudgeInfraError message:
      - starts with 'Claude CLI judge failed:'
      - contains the stderr content ('Traceback: JSONDecodeError in line 42')
      - contains 'error_unexpected' (the subtype, present in diagnostic_detail)
    The subtype is NOT 'error_empty_output' (it classifies UNKNOWN, not
    EMPTY_OUTPUT), so the benign '' short-circuit does not apply — the
    infra-failure branch must fire.
    """
    from shared.cli_invoke import AgentResult

    from fused_memory.reconciliation.judge import JudgeInfraError

    fake_gate = make_gate_mock()
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

        with pytest.raises(JudgeInfraError) as excinfo:
            await judge._call_judge_cli('Evaluate this run.')

    msg = str(excinfo.value)
    assert msg.startswith('Claude CLI judge failed:'), f'unexpected prefix: {msg!r}'
    assert 'Traceback: JSONDecodeError in line 42' in msg, f'stderr missing from: {msg!r}'
    assert "subtype='error_unexpected'" in msg, f'subtype missing from: {msg!r}'


# ─────────────────────────────────────────────────────────────────────
# _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS forwarded by judge to
# invoke_with_cap_retry (task 1401)
# ─────────────────────────────────────────────────────────────────────


class TestJudgeCapWaitSanityBound:
    """_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS is forwarded to
    invoke_with_cap_retry by judge; prevents stalling the reconciliation
    queue under cap."""

    @pytest.mark.asyncio
    async def test_call_judge_cli_forwards_cap_wait_sanity_secs(self):
        """(b) _call_judge_cli forwards cap_wait_sanity_secs=_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS."""
        from unittest.mock import AsyncMock, MagicMock

        from shared.cli_invoke import AgentResult

        from fused_memory.reconciliation import _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS

        config = _make_judge_config(
            judge_llm_provider='claude_cli',
            judge_llm_model='claude-sonnet-4-5',
        )
        mock_jrnl = MagicMock()
        judge = Judge(config=config, journal=mock_jrnl, usage_gate=make_gate_mock())
        # A usable structured payload: a success carrying none is now classified
        # as cli_output_empty infra, which would mask this test's kwargs assertion.
        ok_result = AgentResult(
            success=True, output='', structured_output={'severity': 'ok', 'findings': []},
        )
        mock = AsyncMock(return_value=ok_result)
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=mock,
        ):
            await judge._call_judge_cli('p')
        assert mock.call_args.kwargs['cap_wait_sanity_secs'] == _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS


# ─────────────────────────────────────────────────────────────────────
# Judge._call_judge_cli infra-vs-content taxonomy (task 2947 ask a)
# ─────────────────────────────────────────────────────────────────────


class TestCallJudgeCliTaxonomy:
    """_call_judge_cli distinguishes transport/infra failures (JudgeInfraError)
    from a genuine benign empty verdict ('') and success (output.strip()).

    Infra failures must NOT reach _parse_verdict (where a cap-storm CLI failure
    was being fabricated into a phantom severity=serious halt); they must raise
    a typed JudgeInfraError so review_run can apply bounded backoff and a
    truthful judge-unreachable halt at threshold.
    """

    def _judge(self):
        config = _make_judge_config(
            judge_llm_provider='claude_cli',
            judge_llm_model='claude-sonnet-4-5',
        )
        return Judge(config=config, journal=MagicMock(), usage_gate=make_gate_mock())

    @pytest.mark.asyncio
    async def test_api_error_raises_judge_infra_error(self):
        """(a) A cap-storm api_error (api_error_status set, empty stdout) is
        classified API_ERROR → JudgeInfraError, NOT a benign '' or a plain
        RuntimeError, and never reaches _parse_verdict."""
        from shared.cli_invoke import AgentResult

        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._judge()
        result = AgentResult(success=False, output='', api_error_status=429)
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=result),
        ), pytest.raises(JudgeInfraError):
            await judge._call_judge_cli('prompt')

    @pytest.mark.asyncio
    async def test_timed_out_raises_judge_infra_error(self):
        """(b) A judge CLI timeout is classified TIMED_OUT → JudgeInfraError."""
        from shared.cli_invoke import AgentResult

        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._judge()
        result = AgentResult(success=False, output='', timed_out=True)
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=result),
        ), pytest.raises(JudgeInfraError):
            await judge._call_judge_cli('prompt')

    @pytest.mark.asyncio
    async def test_genuine_empty_output_stays_benign(self):
        """(c) A genuine exit-0/empty-stdout run (subtype='error_empty_output',
        no api_error, not timed out) stays the benign '' legacy contract."""
        from shared.cli_invoke import AgentResult

        judge = self._judge()
        result = AgentResult(
            success=False,
            output='',
            subtype='error_empty_output',
            api_error_status=None,
            timed_out=False,
        )
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=result),
        ):
            assert await judge._call_judge_cli('prompt') == ''

    @pytest.mark.asyncio
    async def test_success_returns_structured_payload(self):
        """(d) A successful run returns the validated structured_output payload.

        Replaces the retired ``result.output.strip()`` contract: the model's text
        channel is no longer the verdict.  A success carrying NO usable payload is
        a distinct case (infra, not content) — see
        TestCallJudgeCliStructuredOutputTaxonomy.
        """
        from shared.cli_invoke import AgentResult

        judge = self._judge()
        result = AgentResult(
            success=True,
            output='  prose the CLI happened to print  \n',
            structured_output={'severity': 'ok', 'findings': []},
        )
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=result),
        ):
            assert await judge._call_judge_cli('prompt') == {'severity': 'ok', 'findings': []}

    @pytest.mark.asyncio
    async def test_all_accounts_capped_raises_judge_infra_error(self):
        """(e) AllAccountsCappedException (cap-wait exhaustion) is re-raised as
        JudgeInfraError so cap-wait exhaustion is counted/backed-off like other
        transport failures instead of propagating opaquely to review_run's broad
        except."""
        from shared.cli_invoke import AllAccountsCappedException

        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._judge()
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(side_effect=AllAccountsCappedException(3, 1800.0, 'judge')),
        ), pytest.raises(JudgeInfraError):
            await judge._call_judge_cli('prompt')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('subtype', ['error_max_turns', 'error_empty_output'])
    async def test_schema_tool_denied_gets_its_own_code(self, subtype):
        """(f) A blocked ``StructuredOutput`` tool is a systemic config break with
        its own machine-readable code, not an anonymous UNKNOWN failure.

        ``--json-schema`` rides the synthetic ``StructuredOutput`` tool, which
        cli_invoke's wildcard expansion deliberately omits from
        ``_REAL_BUILTIN_TOOLS_DENYLIST``.  If a future CLI change starts denying
        it, EVERY judge run is starved of its verdict — the deny-list needs
        fixing, and the log has to say so.  cli_invoke can only ever set
        ``schema_tool_denied`` on a NON-success result (cli_invoke.py:1807 guards
        on ``not is_success``), so the detection has to live on this branch;
        checking it on the success branch is dead code.

        The ``error_empty_output`` case is the ordering pin: a denial whose stdout
        happens to be empty must NOT fall through to the benign ''
        legacy contract.  Nothing was reviewed.
        """
        from shared.cli_invoke import AgentResult

        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._judge()
        result = AgentResult(
            success=False,
            output='',
            subtype=subtype,
            api_error_status=None,
            timed_out=False,
            schema_tool_denied=True,
        )
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=result),
        ), pytest.raises(JudgeInfraError) as exc_info:
            await judge._call_judge_cli('prompt')
        assert exc_info.value.code == 'cli_schema_tool_denied'
        assert 'cli_schema_tool_denied' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ordinary_transport_failure_carries_no_code(self):
        """(g) The new code is SPECIFIC: an ordinary transport failure keeps the
        generic build_failure_message dump and a ``code`` of None, so
        _handle_infra_failure's log can tell a deny-list break apart from a
        run-of-the-mill api_error."""
        from shared.cli_invoke import AgentResult

        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._judge()
        result = AgentResult(success=False, output='', api_error_status=429)
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=result),
        ), pytest.raises(JudgeInfraError) as exc_info:
            await judge._call_judge_cli('prompt')
        assert exc_info.value.code is None


# ─────────────────────────────────────────────────────────────────────
# _call_judge_cli: success with NO usable verdict is INFRA, not content
# ─────────────────────────────────────────────────────────────────────


class TestCallJudgeCliStructuredOutputTaxonomy:
    """A CLI success that carries no usable verdict is an INFRA failure.

    This is the exact 2947 residual this task closes.  The 2026-07-25 halt came
    from a cap-retry SUCCESS — exit 0, is_error=false, 727 output tokens — whose
    text channel held conversational prose because the resume dropped the system
    prompt.  Nothing was reviewed, so there is no verdict to trust; classifying it
    as fail-closed malformed CONTENT fabricated a severity=serious halt that lied
    about a review that never happened.

    No verdict exists => JudgeInfraError => review_run's bounded backoff and a
    truthful 'judge-unreachable' halt only at threshold.  Deliberately kept
    DISTINCT from case (g): the benign exit-0/empty-stdout legacy contract.
    """

    def _judge(self):
        config = _make_judge_config(
            judge_llm_provider='claude_cli',
            judge_llm_model='claude-sonnet-4-5',
        )
        return Judge(config=config, journal=MagicMock(), usage_gate=make_gate_mock())

    async def _raises(self, result):
        """Invoke _call_judge_cli against *result*, returning the JudgeInfraError."""
        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._judge()
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=result),
        ), pytest.raises(JudgeInfraError) as exc_info:
            await judge._call_judge_cli('prompt')
        return exc_info.value

    @pytest.mark.asyncio
    async def test_missing_structured_output_is_cli_output_empty(self):
        """(a) success + structured_output=None, with the 07-25 prose in `output`:
        the prose is NOT a verdict — this is cli_output_empty infra."""
        from shared.cli_invoke import AgentResult

        err = await self._raises(AgentResult(
            success=True,
            output="I'll check the repo for context...\n\n*Searches project files*",
            output_tokens=727,
        ))
        assert err.code == 'cli_output_empty'
        assert 'cli_output_empty' in str(err)

    @pytest.mark.asyncio
    async def test_empty_dict_structured_output_is_cli_output_empty(self):
        """(b) success + structured_output={} carries no verdict either."""
        from shared.cli_invoke import AgentResult

        err = await self._raises(AgentResult(success=True, output='', structured_output={}))
        assert err.code == 'cli_output_empty'

    @pytest.mark.asyncio
    async def test_unparseable_json_string_is_cli_output_unparseable(self):
        """(c) A structured_output string that is not valid JSON must raise
        JudgeInfraError, NOT let json.loads' JSONDecodeError escape into
        review_run's broad except (which would silently return None)."""
        from shared.cli_invoke import AgentResult

        err = await self._raises(
            AgentResult(success=True, output='', structured_output='not json {{{')
        )
        assert err.code == 'cli_output_unparseable'

    @pytest.mark.asyncio
    async def test_non_dict_non_str_payload_is_cli_output_unparseable(self):
        """(d) A list (or any non-dict) payload is a wrong-typed verdict."""
        from shared.cli_invoke import AgentResult

        err = await self._raises(
            AgentResult(success=True, output='', structured_output=['a', 'list'])
        )
        assert err.code == 'cli_output_unparseable'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('payload', [
        {'findings': []},                              # severity missing
        {'severity': 'catastrophic', 'findings': []},  # severity outside the enum
    ])
    async def test_bad_severity_is_cli_output_unparseable(self, payload):
        """(e) A payload whose severity is missing or outside VerdictSeverity is
        not a verdict we can act on — do not normalise it silently."""
        from shared.cli_invoke import AgentResult

        err = await self._raises(
            AgentResult(success=True, output='', structured_output=payload)
        )
        assert err.code == 'cli_output_unparseable'

    @pytest.mark.asyncio
    async def test_non_list_findings_is_cli_output_unparseable(self):
        """(f) findings must be a list: JudgeVerdict construction would otherwise
        raise a pydantic ValidationError that review_run's broad except would
        swallow into a silent None — no verdict, no halt, no signal."""
        from shared.cli_invoke import AgentResult

        err = await self._raises(AgentResult(
            success=True, output='', structured_output={'severity': 'ok', 'findings': 'not-a-list'},
        ))
        assert err.code == 'cli_output_unparseable'

    @pytest.mark.asyncio
    async def test_non_dict_findings_items_is_cli_output_unparseable(self):
        """A findings LIST whose ITEMS are not dicts is still not a usable verdict.

        ``JudgeVerdict.findings`` is ``list[dict]``
        (models/reconciliation.py:241), so a list of bare strings passes the
        findings-is-a-list guard and then dies as a pydantic ValidationError
        inside _verdict_from_payload — uncaught on the dict branch of
        _parse_verdict, hence swallowed by review_run's broad except into a
        silent None with NO infra accounting (the streak is popped one line
        before _parse_verdict runs).

        severity='serious' is deliberate: this is the exact payload shape that
        would otherwise produce the phantom halt this whole task exists to
        prevent.
        """
        from shared.cli_invoke import AgentResult

        err = await self._raises(AgentResult(
            success=True,
            output='prose',
            structured_output={'severity': 'serious', 'findings': ['a bare string finding']},
            output_tokens=727,
        ))
        assert err.code == 'cli_output_unparseable'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('payload', [
        {'severity': 'ok', 'findings': [{'issue': 'x'}]},
        {'severity': 'ok', 'findings': []},
    ])
    async def test_findings_dict_items_are_accepted(self, payload):
        """The guard must not be over-broad: a well-formed findings list is
        RETURNED, not raised.  Without this pin, the fix above could be
        'satisfied' by rejecting every payload that has findings at all."""
        from shared.cli_invoke import AgentResult

        judge = self._judge()
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=AgentResult(
                success=True, output='', structured_output=payload,
            )),
        ):
            assert await judge._call_judge_cli('prompt') == payload

    @pytest.mark.asyncio
    @pytest.mark.parametrize('payload', [
        {'severity': 'ok', 'findings': [['nested', 'list']]},
        {'severity': 'ok', 'findings': [None]},
    ])
    async def test_every_judge_verdict_field_constraint_is_enforced(self, payload):
        """GENERALISATION GUARD — structurally plausible payloads that violate a
        JudgeVerdict field constraint no hand-written guard above checks.

        This is what forces validation to run CENTRALLY against the real model
        rather than accreting one more hand-mirrored isinstance() check that the
        next field added to JudgeVerdict would slip straight past.
        """
        from shared.cli_invoke import AgentResult

        err = await self._raises(
            AgentResult(success=True, output='prose', structured_output=payload)
        )
        assert err.code == 'cli_output_unparseable'

    @pytest.mark.asyncio
    async def test_genuine_empty_output_is_not_conflated_with_infra(self):
        """(g) REGRESSION GUARD — the benign exit-0/empty-stdout legacy contract
        ('empty stdout = valid empty verdict') still returns '' and must NOT be
        raised as infra.  This is a NON-success result whose failure kind is
        EMPTY_OUTPUT, categorically different from (a)-(f) above, which are all
        SUCCESSES carrying no usable payload."""
        from shared.cli_invoke import AgentResult

        judge = self._judge()
        result = AgentResult(
            success=False,
            output='',
            subtype='error_empty_output',
            api_error_status=None,
            timed_out=False,
        )
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=result),
        ):
            # Explicitly must NOT raise.
            assert await judge._call_judge_cli('prompt') == ''


class TestReviewRunNoStructuredVerdict:
    """review_run treats a no-verdict CLI success as bounded-backoff infra.

    THE headline user-observable signal of this task: a cap-storm resume no
    longer writes a fabricated severity=serious row into judge_verdicts nor a
    halt_state row whose reason starts 'Unparseable judge response'.
    """

    def _setup(self, mock_journal, project_id, **cfg):
        from fused_memory.models.reconciliation import (
            ReconciliationRun,
            RunStatus,
            RunType,
        )

        config = _make_judge_config(
            judge_llm_provider='claude_cli',
            judge_llm_model='claude-sonnet-4-5',
            **cfg,
        )
        judge = Judge(config=config, journal=mock_journal, usage_gate=make_gate_mock())
        run = ReconciliationRun(
            id='run-nostruct',
            project_id=project_id,
            run_type=RunType.full,
            trigger_reason='buffer_size:3',
            started_at=datetime.now(UTC),
            events_processed=3,
            status=RunStatus.completed,
            stage_reports={},
        )
        mock_journal.get_run = AsyncMock(return_value=run)
        mock_journal.get_run_actions_combined = AsyncMock(return_value=[])
        return judge

    @staticmethod
    def _prose_success():
        """The literal 2026-07-25 shape: exit 0, is_error=false, real output
        tokens, conversational prose, NO structured payload."""
        from shared.cli_invoke import AgentResult

        return AgentResult(
            success=True,
            output="I'll check the repo for context...\n\n*Searches project files*",
            session_id='jsess-cap',
            output_tokens=727,
        )

    @staticmethod
    def _non_dict_findings_success():
        """A success whose payload passes every hand-written guard but violates
        ``JudgeVerdict.findings: list[dict]`` — the reviewer-confirmed hole."""
        from shared.cli_invoke import AgentResult

        return AgentResult(
            success=True,
            output='prose',
            structured_output={'severity': 'serious', 'findings': ['a bare string finding']},
            session_id='jsess-cap',
            output_tokens=727,
        )

    @pytest.mark.asyncio
    async def test_non_dict_findings_success_is_counted_as_infra(self, mock_journal):
        """A model-invalid structured payload must be COUNTED, not vanish.

        Before the central validation probe, this payload produced "no verdict,
        no halt, AND no signal": the ValidationError escaped _parse_verdict's
        dict branch into review_run's broad except, and the infra streak had
        already been popped one line earlier — so the occurrence never counted
        toward judge_infra_max_consecutive_failures and no halt could EVER fire.
        """
        judge = self._setup(
            mock_journal, 'proj-nostruct-findings',
            halt_on_judge_serious=True, judge_infra_max_consecutive_failures=3,
        )
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=self._non_dict_findings_success()),
        ):
            first = await judge.review_run('run-nostruct')

            # First occurrence: still no fabricated severity=serious row.
            assert first is None
            mock_journal.add_verdict.assert_not_called()
            mock_journal.set_halt.assert_not_called()
            assert judge.is_halted('proj-nostruct-findings') is False

            for _ in range(2):
                await judge.review_run('run-nostruct')

        # The assertion the reviewer's finding turns on: the occurrences are
        # COUNTED, so the third consecutive one halts — truthfully.
        assert judge.is_halted('proj-nostruct-findings')
        reason = judge.halt_reason('proj-nostruct-findings')
        assert reason is not None
        assert reason.startswith('judge-unreachable halt')
        assert 'Serious verdict' not in reason
        assert 'Unparseable judge response' not in reason
        mock_journal.add_verdict.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_occurrence_stamps_no_verdict_and_does_not_halt(self, mock_journal):
        """(h) First no-verdict success: no verdict row, no halt_state row, not
        halted.  Before this task it stamped a fabricated severity=serious verdict
        and halted the project on the very first occurrence."""
        judge = self._setup(
            mock_journal, 'proj-nostruct-a',
            halt_on_judge_serious=True, judge_infra_max_consecutive_failures=3,
        )
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=self._prose_success()),
        ):
            verdict = await judge.review_run('run-nostruct')

        assert verdict is None
        mock_journal.add_verdict.assert_not_called()
        mock_journal.set_halt.assert_not_called()
        assert judge.is_halted('proj-nostruct-a') is False

    @pytest.mark.asyncio
    async def test_third_consecutive_occurrence_halts_truthfully(self, mock_journal):
        """(i) At threshold the halt fires, but with the TRUTHFUL
        'judge-unreachable halt' reason — never 'Serious verdict' (a review that
        never happened) and never 'Unparseable judge response' (reachable
        malformed content, which this is not) — and still stamps no verdict."""
        judge = self._setup(
            mock_journal, 'proj-nostruct-b',
            halt_on_judge_serious=True, judge_infra_max_consecutive_failures=3,
        )
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=self._prose_success()),
        ):
            for _ in range(3):
                await judge.review_run('run-nostruct')

        assert judge.is_halted('proj-nostruct-b')
        reason = judge.halt_reason('proj-nostruct-b')
        assert reason is not None
        assert reason.startswith('judge-unreachable halt')
        assert 'Serious verdict' not in reason
        assert 'Unparseable judge response' not in reason
        mock_journal.add_verdict.assert_not_called()

    @pytest.mark.asyncio
    async def test_genuine_serious_structured_verdict_still_halts(self, mock_journal):
        """PRESERVE task 2947's intent: a REAL severity=serious payload still halts.

        If this fails, the no-verdict classification over-reached and started
        treating genuine content verdicts as infra — do NOT 'fix' this test.
        """
        from shared.cli_invoke import AgentResult

        judge = self._setup(
            mock_journal, 'proj-genuine-struct', halt_on_judge_serious=True,
        )
        payload = {
            'severity': 'serious',
            'findings': [{'issue': 'mass incorrect deletions', 'severity': 'serious'}],
        }
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=AgentResult(
                success=True, output='', structured_output=payload,
            )),
        ):
            verdict = await judge.review_run('run-nostruct')

        assert verdict is not None
        assert verdict.severity == VerdictSeverity.serious
        assert verdict.action_taken == VerdictAction.halt
        mock_journal.add_verdict.assert_called_once()
        assert judge.is_halted('proj-genuine-struct')
        reason = judge.halt_reason('proj-genuine-struct')
        assert reason == 'Serious verdict in run run-nostruct'
        assert 'judge-unreachable' not in reason
        assert 'Unparseable judge response' not in reason

    @pytest.mark.asyncio
    async def test_moderate_structured_verdict_still_recommends_rollback(self, mock_journal):
        """PRESERVE: a REAL severity=moderate payload still recommends rollback,
        and does not halt."""
        from shared.cli_invoke import AgentResult

        judge = self._setup(
            mock_journal, 'proj-moderate-struct', halt_on_judge_serious=True,
        )
        payload = {'severity': 'moderate', 'findings': [{'issue': 'drifted entry'}]}
        with patch(
            'fused_memory.reconciliation.judge.invoke_with_cap_retry',
            new=AsyncMock(return_value=AgentResult(
                success=True, output='', structured_output=payload,
            )),
        ):
            verdict = await judge.review_run('run-nostruct')

        assert verdict is not None
        assert verdict.severity == VerdictSeverity.moderate
        assert verdict.action_taken == VerdictAction.rollback
        assert judge.is_halted('proj-moderate-struct') is False
        mock_journal.set_halt.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# review_run bounded-backoff handling of JudgeInfraError (task 2947 ask a)
# ─────────────────────────────────────────────────────────────────────


class TestReviewRunInfraFailureHandling:
    """review_run treats JudgeInfraError (transport/infra) as bounded backoff,
    NOT a phantom serious verdict (task 2947 ask a).

    Below judge_infra_max_consecutive_failures consecutive infra failures,
    review_run returns None WITHOUT stamping a verdict or halting. At the
    threshold it applies a TRUTHFUL 'judge-unreachable' halt (never the lying
    'Serious verdict' reason, and never a persisted verdict). Any successful
    transport resets the per-project streak. The infra halt is gated on the
    same halt_on_judge_serious master switch as the serious-verdict halt.
    """

    def _setup(self, mock_journal, project_id, **cfg):
        from fused_memory.models.reconciliation import (
            ReconciliationRun,
            RunStatus,
            RunType,
        )

        config = _make_judge_config(**cfg)
        judge = Judge(config=config, journal=mock_journal)
        now = datetime.now(UTC)
        run = ReconciliationRun(
            id='run-infra',
            project_id=project_id,
            run_type=RunType.full,
            trigger_reason='buffer_size:3',
            started_at=now,
            events_processed=3,
            status=RunStatus.completed,
            stage_reports={},
        )
        mock_journal.get_run = AsyncMock(return_value=run)
        mock_journal.get_run_actions_combined = AsyncMock(return_value=[])
        return judge

    @pytest.mark.asyncio
    async def test_below_threshold_infra_failures_do_not_halt_or_stamp(self, mock_journal):
        """(a) The first two (< 3) consecutive infra failures return None, stamp
        no verdict, and do not halt — bounded backoff, not a phantom verdict."""
        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._setup(
            mock_journal, 'proj-infra-a',
            halt_on_judge_serious=True, judge_infra_max_consecutive_failures=3,
        )
        with patch.object(
            judge, '_call_llm', AsyncMock(side_effect=JudgeInfraError('cap storm'))
        ):
            v1 = await judge.review_run('run-infra')
            v2 = await judge.review_run('run-infra')

        assert v1 is None
        assert v2 is None
        mock_journal.set_halt.assert_not_called()
        mock_journal.add_verdict.assert_not_called()
        assert not judge.is_halted('proj-infra-a')

    @pytest.mark.asyncio
    async def test_threshold_infra_failure_applies_truthful_judge_unreachable_halt(
        self, mock_journal
    ):
        """(b) The 3rd consecutive infra failure halts the project with a truthful
        'judge-unreachable' reason — NOT the phantom 'Serious verdict' reason —
        and persists no fabricated verdict."""
        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._setup(
            mock_journal, 'proj-infra-b',
            halt_on_judge_serious=True, judge_infra_max_consecutive_failures=3,
        )
        with patch.object(
            judge, '_call_llm', AsyncMock(side_effect=JudgeInfraError('cap storm'))
        ):
            await judge.review_run('run-infra')
            await judge.review_run('run-infra')
            await judge.review_run('run-infra')

        assert judge.is_halted('proj-infra-b')
        mock_journal.set_halt.assert_called_once()
        reason = mock_journal.set_halt.call_args.kwargs['reason']
        assert 'judge-unreachable' in reason
        assert 'Serious verdict' not in reason
        mock_journal.add_verdict.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_review_resets_infra_streak(self, mock_journal):
        """(c) A successful review between infra failures resets the per-project
        streak, so a later lone infra failure is well below the threshold and
        does not halt."""
        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._setup(
            mock_journal, 'proj-infra-c',
            halt_on_judge_serious=True, judge_infra_max_consecutive_failures=3,
        )
        # Two infra failures (streak → 2)
        with patch.object(
            judge, '_call_llm', AsyncMock(side_effect=JudgeInfraError('cap storm'))
        ):
            await judge.review_run('run-infra')
            await judge.review_run('run-infra')
        # A clean 'ok' verdict is a successful transport — resets the streak
        with patch.object(
            judge, '_call_llm',
            AsyncMock(return_value='{"severity": "ok", "findings": []}'),
        ):
            await judge.review_run('run-infra')
        # A further lone infra failure is streak #1, below threshold → no halt
        with patch.object(
            judge, '_call_llm', AsyncMock(side_effect=JudgeInfraError('cap storm'))
        ):
            v = await judge.review_run('run-infra')

        assert v is None
        assert not judge.is_halted('proj-infra-c')
        mock_journal.set_halt.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_infra_halt_when_halt_on_judge_serious_disabled(self, mock_journal):
        """With halt_on_judge_serious=False, even many consecutive infra failures
        past the threshold never halt (same master switch as the serious halt)."""
        from fused_memory.reconciliation.judge import JudgeInfraError

        judge = self._setup(
            mock_journal, 'proj-infra-d',
            halt_on_judge_serious=False, judge_infra_max_consecutive_failures=3,
        )
        with patch.object(
            judge, '_call_llm', AsyncMock(side_effect=JudgeInfraError('cap storm'))
        ):
            for _ in range(5):
                await judge.review_run('run-infra')

        assert not judge.is_halted('proj-infra-d')
        mock_journal.set_halt.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Truthful content-serious halt reasons (task 2947 ask b)
# ─────────────────────────────────────────────────────────────────────


def test_parse_verdict_unparseable_tags_code_marker(mock_journal):
    """_parse_verdict tags its single fabricated finding with a machine-readable
    code='unparseable_judge_response' marker (severity stays serious).

    The marker lets review_run pick a TRUTHFUL 'Unparseable judge response'
    halt reason instead of the lying 'Serious verdict' one, via a structured
    field rather than fragile substring matching. Keeping it on the SAME single
    finding preserves len(findings)==1 (the existing
    test_parse_verdict_unparseable_response_is_loud_not_fabricated contract).
    """
    judge = Judge(config=_make_judge_config(), journal=mock_journal)

    verdict = judge._parse_verdict('garbage {{{', 'run-code-marker')

    assert verdict.severity == VerdictSeverity.serious
    assert len(verdict.findings) == 1
    assert verdict.findings[0].get('code') == 'unparseable_judge_response'


@pytest.mark.asyncio
async def test_review_run_unparseable_content_halts_with_unparseable_reason(mock_journal):
    """Genuinely-malformed REACHABLE content (success, non-empty, non-JSON) still
    halts loudly (fail-closed preserved), but with the truthful 'Unparseable
    judge response' reason — NOT the phantom 'Serious verdict' reason."""
    from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType

    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-unparseable',
        project_id='proj-unparseable',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.completed,
        stage_reports={},
    )
    mock_journal.get_run = AsyncMock(return_value=run)
    mock_journal.get_run_actions_combined = AsyncMock(return_value=[])

    with patch.object(
        judge, '_call_llm', AsyncMock(return_value='not valid json at all {{{')
    ):
        verdict = await judge.review_run('run-unparseable')

    assert verdict is not None
    assert verdict.severity == VerdictSeverity.serious
    assert judge.is_halted('proj-unparseable')
    mock_journal.set_halt.assert_called_once()
    reason = mock_journal.set_halt.call_args.kwargs['reason']
    assert 'Unparseable judge response' in reason
    assert 'Serious verdict' not in reason


@pytest.mark.asyncio
async def test_review_run_genuine_serious_halts_with_serious_verdict_reason(mock_journal):
    """A genuine content severity=serious verdict keeps the already-truthful
    'Serious verdict in run X' reason (not the unparseable relabel)."""
    from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType

    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-genuine-serious',
        project_id='proj-genuine-serious',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.completed,
        stage_reports={},
    )
    mock_journal.get_run = AsyncMock(return_value=run)
    mock_journal.get_run_actions_combined = AsyncMock(return_value=[])

    serious_json = '{"severity": "serious", "findings": [{"issue": "real corruption"}]}'
    with patch.object(judge, '_call_llm', AsyncMock(return_value=serious_json)):
        verdict = await judge.review_run('run-genuine-serious')

    assert verdict is not None
    assert verdict.severity == VerdictSeverity.serious
    assert judge.is_halted('proj-genuine-serious')
    mock_journal.set_halt.assert_called_once()
    reason = mock_journal.set_halt.call_args.kwargs['reason']
    assert 'Serious verdict in run' in reason
    assert 'Unparseable' not in reason


# ─────────────────────────────────────────────────────────────────────
# halt_reason accessor exposes the real halt reason (task 2947 ask b)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_halt_reason_returns_truthful_reason_after_halt(mock_journal):
    """halt_reason(project_id) returns the exact reason passed to _apply_halt so
    the harness can thread the REAL reason into the on_judge_halt escalation;
    it returns None for a never-halted project."""
    from fused_memory.models.reconciliation import ReconciliationRun, RunStatus, RunType

    config = _make_judge_config(halt_on_judge_serious=True)
    judge = Judge(config=config, journal=mock_journal)

    # Never-halted project → None
    assert judge.halt_reason('proj-halt-reason') is None

    now = datetime.now(UTC)
    run = ReconciliationRun(
        id='run-halt-reason',
        project_id='proj-halt-reason',
        run_type=RunType.full,
        trigger_reason='buffer_size:3',
        started_at=now,
        events_processed=3,
        status=RunStatus.completed,
        stage_reports={},
    )
    mock_journal.get_run = AsyncMock(return_value=run)
    mock_journal.get_run_actions_combined = AsyncMock(return_value=[])

    serious_json = '{"severity": "serious", "findings": [{"issue": "real corruption"}]}'
    with patch.object(judge, '_call_llm', AsyncMock(return_value=serious_json)):
        await judge.review_run('run-halt-reason')

    assert judge.is_halted('proj-halt-reason')
    # Matches the reason _apply_halt was called with (verified via set_halt too).
    expected = mock_journal.set_halt.call_args.kwargs['reason']
    assert judge.halt_reason('proj-halt-reason') == expected
    assert expected == 'Serious verdict in run run-halt-reason'
    # A different, never-halted project still returns None.
    assert judge.halt_reason('some-other-project') is None


@pytest.mark.asyncio
async def test_halt_reason_cleared_on_unhalt(mock_journal):
    """unhalt(project_id) clears the stored halt reason (halt_reason → None)."""
    judge = Judge(config=_make_judge_config(halt_on_judge_serious=True), journal=mock_journal)

    await judge._apply_halt('proj-unhalt-reason', reason='judge-unreachable halt: test')
    assert judge.halt_reason('proj-unhalt-reason') == 'judge-unreachable halt: test'

    await judge.unhalt('proj-unhalt-reason')
    assert judge.halt_reason('proj-unhalt-reason') is None


# ─────────────────────────────────────────────────────────────────────
# Acceptance regression: the 2026-07-25 cap-retry resume (task 3067)
# ─────────────────────────────────────────────────────────────────────


class TestJudgeCapRetryResumeKeepsVerdictContract:
    """Replays the 2026-07-25 429-then-succeed sequence end to end.

    Nothing inside judge.py is patched — the REAL _call_judge_cli drives the REAL
    invoke_with_cap_retry, with only shared.cli_invoke.invoke_claude_agent (and the
    cap cooldown sleep) mocked. That is the point: the resume branch at
    cli_invoke.py:1270-1272 must be exercised for real.

    The load-bearing assertion is that the RESUMED invocation still carries
    output_schema. build_claude_argv drops --system-prompt-file on the resume path
    (cli_invoke.py:1501-1503) but emits --json-schema unconditionally (:1552-1553),
    so a prose-only verdict contract evaporates on resume while a schema-carried
    one survives. On 2026-07-25 the resumed judge answered in prose and
    _parse_verdict fabricated it into a severity=serious halt of a project nobody
    had reviewed.

    MUTATION CHECK (verified locally before committing): deleting
    ``output_schema=JUDGE_VERDICT_SCHEMA`` from _call_judge_cli must make
    assertion 1 fail. If it does not, this test is not pinning what it claims.
    """

    def _judge(self, mock_journal, project_id):
        from fused_memory.models.reconciliation import (
            ReconciliationRun,
            RunStatus,
            RunType,
        )

        config = _make_judge_config(
            judge_llm_provider='claude_cli',
            judge_llm_model='claude-sonnet-4-5',
            halt_on_judge_serious=True,
        )
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        judge = Judge(config=config, journal=mock_journal, usage_gate=gate)
        run = ReconciliationRun(
            id='run-cap-resume',
            project_id=project_id,
            run_type=RunType.full,
            trigger_reason='buffer_size:3',
            started_at=datetime.now(UTC),
            events_processed=3,
            status=RunStatus.completed,
            stage_reports={},
        )
        mock_journal.get_run = AsyncMock(return_value=run)
        mock_journal.get_run_actions_combined = AsyncMock(return_value=[])
        return judge

    @pytest.mark.asyncio
    async def test_429_then_succeed_yields_a_real_verdict_and_no_halt(self, mock_journal):
        from shared.cli_invoke import CAP_HIT_RESUME_PROMPT, AgentResult

        from fused_memory.reconciliation.judge import JUDGE_VERDICT_SCHEMA

        judge = self._judge(mock_journal, 'proj-cap-resume')

        # First attempt: capped (429) but carrying a session id, so the loop takes
        # the RESUME branch rather than restarting fresh.
        capped = AgentResult(
            success=False, output='', cost_usd=0.0,
            api_error_status=429, session_id='jsess-cap',
        )
        # The resumed attempt succeeds with a real payload — and, exactly as on
        # 2026-07-25, an empty text channel is no obstacle because the verdict no
        # longer lives there.
        ok = AgentResult(
            success=True, output='',
            structured_output={'severity': 'ok', 'findings': []},
            session_id='jsess-cap', output_tokens=727,
        )

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock, side_effect=[capped, ok],
            ) as mock_agent,
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            verdict = await judge.review_run('run-cap-resume')

        # 1. The resume happened, and the verdict contract survived it.
        assert mock_agent.call_count == 2
        second = mock_agent.call_args_list[1].kwargs
        assert second['resume_session_id'] == 'jsess-cap'
        assert second['prompt'] == CAP_HIT_RESUME_PROMPT
        assert second['output_schema'] is JUDGE_VERDICT_SCHEMA, (
            'the resumed invocation must still carry the verdict schema — the '
            'system prompt is dropped on resume, the schema is not'
        )

        # 2. A real, parseable verdict came back.
        assert verdict is not None
        assert verdict.severity == VerdictSeverity.ok
        assert verdict.action_taken == VerdictAction.none

        # 3. No halt_state row, project not halted.
        mock_journal.set_halt.assert_not_called()
        assert judge.is_halted('proj-cap-resume') is False

        # 4. The judge_verdicts row is a real verdict, not a fabricated one.
        mock_journal.add_verdict.assert_called_once()
        stamped = mock_journal.add_verdict.call_args.args[0]
        assert all(
            'could not be parsed' not in f.get('issue', '') for f in stamped.findings
        )
        assert all(
            f.get('code') != 'unparseable_judge_response' for f in stamped.findings
        )
