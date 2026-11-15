"""The per-invocation judge_invocations record written by every claude_cli judge call (task 6573).

The CLI result is faked at ``invoke_with_cap_retry``, the same seam
test_judge.py's claude_cli tests use. The transcript the rejected-call count
is read from is a real JSONL file under a per-test ``CLAUDE_CONFIG_DIR``.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _sdk_client_stubs import sdk_client_double
from shared.cli_invoke import AgentResult
from shared.testing import make_gate_mock

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    JudgeInvocation,
    JudgeVerdict,
    ReconciliationRun,
    RunStatus,
    RunType,
)
from fused_memory.reconciliation.judge import (
    Judge,
    JudgeInfraError,
    NoVerdict,
    NoVerdictReason,
)

_INVOKE = 'fused_memory.reconciliation.judge.invoke_with_cap_retry'
_JUDGE_LOGGER = 'fused_memory.reconciliation.judge'
_TURN_CAP_HIT = 'reconciliation.judge_turn_cap_hit'
_RECORD_FAILED = 'reconciliation.judge_invocation_record_failed'
_OK_PAYLOAD = {'severity': 'ok', 'findings': []}


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch) -> Path:
    path = tmp_path / 'cfg'
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(path))
    return path


@pytest.fixture
def journal() -> MagicMock:
    j = MagicMock()
    j.get_run = AsyncMock(return_value=None)
    j.get_recent_verdicts = AsyncMock(return_value=[])
    j.get_run_actions_combined = AsyncMock(return_value=[])
    j.add_verdict = AsyncMock(return_value=None)
    j.get_halt_states = AsyncMock(return_value=[])
    j.set_halt = AsyncMock(return_value=None)
    j.clear_halt = AsyncMock(return_value=None)
    j.decrement_unhalt_grace = AsyncMock(return_value=0)
    j.record_judge_invocation = AsyncMock(return_value=None)
    return j


def _judge(journal, provider: str = 'claude_cli') -> Judge:
    config = ReconciliationConfig(judge_llm_provider=provider, judge_llm_model='sonnet')
    return Judge(config=config, journal=journal, usage_gate=make_gate_mock())


def _assistant(*blocks: dict) -> dict:
    return {'type': 'assistant', 'message': {'role': 'assistant', 'content': list(blocks)}}


def _tool_use(name: str, tool_input: dict | None = None, *, tool_use_id: str) -> dict:
    return {'type': 'tool_use', 'id': tool_use_id, 'name': name, 'input': tool_input or {}}


def _write_transcript(config_dir: Path, session_id: str, records: list[dict]) -> None:
    path = config_dir / 'projects' / '-x' / f'{session_id}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))


def _two_rejected_greps_then_verdict(config_dir: Path, session_id: str = 's1') -> None:
    _write_transcript(config_dir, session_id, [
        _assistant(
            {'type': 'text', 'text': 'Let me check the repo first.'},
            _tool_use('Grep', {'pattern': 'x'}, tool_use_id='tu-1'),
            _tool_use('Grep', {'pattern': 'y'}, tool_use_id='tu-2'),
        ),
        {'type': 'user', 'message': {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'tu-1', 'is_error': True,
             'content': '<tool_use_error>Error: No such tool available: Grep</tool_use_error>'},
        ]}},
        _assistant(_tool_use('StructuredOutput', _OK_PAYLOAD, tool_use_id='tu-3')),
    ])


def _success(**overrides) -> AgentResult:
    fields = {
        'success': True, 'output': '', 'structured_output': _OK_PAYLOAD,
        'subtype': 'success', 'turns': 3, 'session_id': 's1',
    }
    return AgentResult(**(fields | overrides))


def _turn_cap(**overrides) -> AgentResult:
    fields = {
        'success': False, 'output': '', 'subtype': 'error_max_turns',
        'turns': 4, 'session_id': 's1', 'output_tokens': 500,
    }
    return AgentResult(**(fields | overrides))


def _salvaged_turn_cap() -> AgentResult:
    return _success(subtype='error_max_turns', schema_salvaged=True, turns=4)


async def _call(judge: Judge, result: AgentResult, run_id: str = 'run-1'):
    with patch(_INVOKE, new_callable=AsyncMock, return_value=result):
        return await judge._call_judge_cli('Evaluate this run.', run_id=run_id)


def _recorded(journal) -> JudgeInvocation:
    journal.record_judge_invocation.assert_awaited_once()
    (invocation,) = journal.record_judge_invocation.await_args.args
    return invocation


def _logged(caplog, event: str) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name == _JUDGE_LOGGER
        and r.levelno == logging.WARNING
        and r.getMessage().split(' ', 1)[0] == event
    ]


def _fields(record: logging.LogRecord) -> dict[str, str]:
    """The ``key=value`` tokens after the event name, as the service's LOG_FORMAT renders them."""
    return dict(token.split('=', 1) for token in record.getMessage().split(' ')[1:])


class TestRecordedOutcome:
    @pytest.mark.asyncio
    async def test_ordinary_success_records_turns_and_rejected_calls(self, journal, config_dir):
        _two_rejected_greps_then_verdict(config_dir)

        verdict = await _call(_judge(journal), _success(), run_id='run-7')

        assert verdict == _OK_PAYLOAD
        invocation = _recorded(journal)
        assert invocation.run_id == 'run-7'
        assert invocation.ended_error_max_turns is False
        assert invocation.schema_salvaged is False
        assert invocation.num_turns == 3
        assert invocation.rejected_tool_calls == 2
        assert invocation.completed_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_non_salvaged_turn_cap_is_recorded_before_the_infra_error(
        self, journal, config_dir,
    ):
        _two_rejected_greps_then_verdict(config_dir)

        with pytest.raises(JudgeInfraError):
            await _call(_judge(journal), _turn_cap())

        invocation = _recorded(journal)
        assert invocation.ended_error_max_turns is True
        assert invocation.schema_salvaged is False
        assert invocation.num_turns == 4
        assert invocation.rejected_tool_calls == 2

    @pytest.mark.asyncio
    async def test_schema_salvaged_turn_cap_counts_and_keeps_the_verdict(self, journal):
        verdict = await _call(_judge(journal), _salvaged_turn_cap())

        assert verdict == _OK_PAYLOAD
        invocation = _recorded(journal)
        assert invocation.ended_error_max_turns is True
        assert invocation.schema_salvaged is True

    @pytest.mark.asyncio
    async def test_rejected_calls_unknown_without_a_transcript(self, journal):
        await _call(_judge(journal), _success(session_id='no-such-session'))

        assert _recorded(journal).rejected_tool_calls is None

    @pytest.mark.asyncio
    async def test_rejected_calls_unknown_without_a_session_id(self, journal, config_dir):
        _two_rejected_greps_then_verdict(config_dir, session_id='')

        await _call(_judge(journal), _success(session_id=''))

        assert _recorded(journal).rejected_tool_calls is None

    @pytest.mark.asyncio
    async def test_unreadable_transcript_is_unknown_and_still_recorded(self, journal, caplog):
        with (
            patch(
                'fused_memory.reconciliation.judge.transcript_evidence_for_session',
                side_effect=OSError('disk gone'),
            ),
            caplog.at_level(logging.WARNING, logger=_JUDGE_LOGGER),
        ):
            await _call(_judge(journal), _success())

        assert _recorded(journal).rejected_tool_calls is None
        (failed,) = _logged(caplog, 'reconciliation.judge_rejected_tool_count_failed')
        assert _fields(failed)['session_id'] == 's1'


class TestTelemetryNeverChangesTheOutcome:
    @pytest.mark.asyncio
    async def test_write_failure_keeps_the_verdict(self, journal, caplog):
        journal.record_judge_invocation.side_effect = RuntimeError('db locked')

        with caplog.at_level(logging.WARNING, logger=_JUDGE_LOGGER):
            verdict = await _call(_judge(journal), _success())

        assert verdict == _OK_PAYLOAD
        (failed,) = _logged(caplog, _RECORD_FAILED)
        assert _fields(failed) == {'run_id': 'run-1', 'ended_error_max_turns': 'False'}

    @pytest.mark.asyncio
    async def test_write_failure_keeps_the_infra_error_and_the_hit_line(self, journal, caplog):
        journal.record_judge_invocation.side_effect = RuntimeError('db locked')

        with (
            caplog.at_level(logging.WARNING, logger=_JUDGE_LOGGER),
            pytest.raises(JudgeInfraError),
        ):
            await _call(_judge(journal), _turn_cap())

        (failed,) = _logged(caplog, _RECORD_FAILED)
        assert _fields(failed) == {'run_id': 'run-1', 'ended_error_max_turns': 'True'}
        assert len(_logged(caplog, _TURN_CAP_HIT)) == 1


class TestTurnCapHitLog:
    @pytest.mark.asyncio
    async def test_turn_cap_logs_the_hit(self, journal, config_dir, caplog):
        _two_rejected_greps_then_verdict(config_dir)

        with (
            caplog.at_level(logging.WARNING, logger=_JUDGE_LOGGER),
            pytest.raises(JudgeInfraError),
        ):
            await _call(_judge(journal), _turn_cap(), run_id='run-cap')

        (hit,) = _logged(caplog, _TURN_CAP_HIT)
        assert _fields(hit) == {
            'run_id': 'run-cap',
            'num_turns': '4',
            'max_turns': '3',
            'rejected_tool_calls': '2',
            'schema_salvaged': 'False',
        }

    @pytest.mark.asyncio
    async def test_turn_cap_without_a_transcript_logs_rejected_calls_unknown(
        self, journal, caplog,
    ):
        with (
            caplog.at_level(logging.WARNING, logger=_JUDGE_LOGGER),
            pytest.raises(JudgeInfraError),
        ):
            await _call(_judge(journal), _turn_cap(session_id='no-such-session'))

        (hit,) = _logged(caplog, _TURN_CAP_HIT)
        assert _fields(hit)['rejected_tool_calls'] == 'None'

    @pytest.mark.asyncio
    async def test_salvaged_turn_cap_hit_says_it_was_salvaged(self, journal, caplog):
        with caplog.at_level(logging.WARNING, logger=_JUDGE_LOGGER):
            await _call(_judge(journal), _salvaged_turn_cap())

        (hit,) = _logged(caplog, _TURN_CAP_HIT)
        assert _fields(hit)['schema_salvaged'] == 'True'

    @pytest.mark.asyncio
    async def test_ordinary_success_logs_no_hit(self, journal, caplog):
        with caplog.at_level(logging.WARNING, logger=_JUDGE_LOGGER):
            await _call(_judge(journal), _success())

        assert not _logged(caplog, _TURN_CAP_HIT)


class TestThroughReviewRun:
    @staticmethod
    def _judge_over_run(journal) -> Judge:
        journal.get_run = AsyncMock(return_value=ReconciliationRun(
            id='run-42',
            project_id='proj',
            run_type=RunType.full,
            trigger_reason='buffer_size:3',
            started_at=datetime.now(UTC),
            events_processed=3,
            status=RunStatus.completed,
            stage_reports={},
        ))
        return _judge(journal)

    @pytest.mark.asyncio
    async def test_success_records_the_reviewed_run(self, journal):
        judge = self._judge_over_run(journal)

        with patch(_INVOKE, new_callable=AsyncMock, return_value=_success()):
            outcome = await judge.review_run('run-42')

        assert isinstance(outcome, JudgeVerdict)
        assert _recorded(journal).run_id == 'run-42'

    @pytest.mark.asyncio
    async def test_turn_cap_is_a_judge_infra_failure_and_still_recorded(self, journal):
        judge = self._judge_over_run(journal)

        with patch(_INVOKE, new_callable=AsyncMock, return_value=_turn_cap()):
            outcome = await judge.review_run('run-42')

        assert isinstance(outcome, NoVerdict)
        assert outcome.reason is NoVerdictReason.JUDGE_INFRA_FAILURE
        invocation = _recorded(journal)
        assert invocation.run_id == 'run-42'
        assert invocation.ended_error_max_turns is True

    @pytest.mark.asyncio
    async def test_write_failure_does_not_feed_the_no_verdict_streak(self, journal):
        journal.record_judge_invocation.side_effect = RuntimeError('db locked')
        judge = self._judge_over_run(journal)

        with patch(_INVOKE, new_callable=AsyncMock, return_value=_success()):
            outcome = await judge.review_run('run-42')

        assert isinstance(outcome, JudgeVerdict)
        journal.add_verdict.assert_awaited_once()


class TestOtherProviders:
    @pytest.mark.asyncio
    async def test_anthropic_records_nothing(self, journal):
        client = sdk_client_double()
        client.messages.create = AsyncMock(return_value=MagicMock(content=[]))

        with patch('anthropic.AsyncAnthropic', return_value=client):
            await _judge(journal, provider='anthropic')._call_llm('Evaluate.', run_id='run-1')

        journal.record_judge_invocation.assert_not_awaited()
