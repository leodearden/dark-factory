"""A provider error on any judge arm surfaces as itself after ONE attempt (task 8521).

Driven over a real loopback socket (``_mock_openai_server``), so the SDK
client the production credential path builds is the one under test. Each
error is followed by a 200 the server would have served a retry: a client
that retried would consume it and return a verdict, which is the hidden
second time policy inside ``write_triage.judge_timeout_seconds`` this file
pins as absent.
"""

from __future__ import annotations

import logging
import types
from typing import Any, NamedTuple
from unittest.mock import AsyncMock

import anthropic
import openai
import pytest
from _mock_openai_server import (
    MockOpenAIServer,
    chat_completion_body,
    mock_openai_server,
    responses_body,
)

from fused_memory.server.write_triage import (
    OUTCOME_JUDGE,
    OUTCOME_STORED,
    BandDecision,
    TriageFailOpenCounter,
    triage_write,
)
from fused_memory.server.write_triage_judge import judge_write
from fused_memory.services.memory_service import SearchResults
from server.write_triage_harness import T_HIGH, T_LOW, search_candidate

_RESTATES = '{"verdict": "restates", "candidate_id": "m1"}'
_DISTINCT = '{"verdict": "distinct", "candidate_id": null}'
_ERROR_BODY = {'error': {'message': 'boom', 'type': 'server_error'}}
_IN_JUDGE_BAND = 0.80


def _decision() -> BandDecision:
    return BandDecision(OUTCOME_JUDGE, 'm1', _IN_JUDGE_BAND, T_HIGH, T_LOW)


def _anthropic_message_body(text: str) -> dict[str, Any]:
    return {
        'id': 'msg_mock',
        'type': 'message',
        'role': 'assistant',
        'model': 'mock-model',
        'content': [{'type': 'text', 'text': text}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 3, 'output_tokens': 4},
    }


class _Arm(NamedTuple):
    """One judge arm: how it is selected, where it posts, and what it raises."""

    provider: str
    client_class: str
    suffix: str
    ok_body: dict[str, Any]
    status_error: type[openai.APIStatusError] | type[anthropic.APIStatusError]
    sdk_appends_v1: bool

    def judge_api_url(self, server: MockOpenAIServer) -> str:
        if self.sdk_appends_v1:
            return f'http://{server.host}:{server.port}'
        return server.base_url


_ARMS = {
    'responses': _Arm(
        provider='openai', client_class='openai', suffix='/responses',
        ok_body=responses_body(_RESTATES), status_error=openai.APIStatusError,
        sdk_appends_v1=False,
    ),
    'chat': _Arm(
        provider='openai', client_class='openai_generic', suffix='/chat/completions',
        ok_body=chat_completion_body(_RESTATES), status_error=openai.APIStatusError,
        sdk_appends_v1=False,
    ),
    'anthropic': _Arm(
        provider='anthropic', client_class='openai', suffix='/messages',
        ok_body=_anthropic_message_body(_RESTATES), status_error=anthropic.APIStatusError,
        sdk_appends_v1=True,
    ),
}


def _service(arm: _Arm, server: MockOpenAIServer) -> types.SimpleNamespace:
    """A memory service whose judge reaches *server* through the production credential path.

    The explicit ``api_key`` matters: a dispatched shell has no ``OPENAI_API_KEY``.
    """
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            write_triage=types.SimpleNamespace(
                judge_provider=arm.provider,
                judge_model='test-model',
                judge_api_url=arm.judge_api_url(server),
                t_high=T_HIGH,
                t_low=T_LOW,
            ),
            llm=types.SimpleNamespace(
                provider=arm.provider,
                model='test-model',
                client_class=arm.client_class,
                providers=types.SimpleNamespace(**{
                    arm.provider: types.SimpleNamespace(api_key='sk-test', api_url=None),
                }),
            ),
        ),
    )


class TestEachJudgeArmMakesOneAttempt:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', [500, 429])
    @pytest.mark.parametrize('arm_id', list(_ARMS))
    async def test_a_retryable_provider_status_is_raised_after_one_request(
        self, arm_id: str, status: int,
    ) -> None:
        arm = _ARMS[arm_id]
        with mock_openai_server() as server:
            server.set_response_sequence(arm.suffix, [(status, _ERROR_BODY), (200, arm.ok_body)])

            with pytest.raises(arm.status_error) as excinfo:
                await judge_write(
                    memory_service=_service(arm, server),
                    content='c', project_id='p',
                    decision=_decision(), candidates=[search_candidate(score=_IN_JUDGE_BAND)],
                )

            assert excinfo.value.status_code == status
            assert len(server.requests_to(arm.suffix)) == 1


class TestTheFailOpenRecordNamesTheProviderError:
    @pytest.mark.asyncio
    async def test_a_provider_5xx_is_counted_once_and_logged_as_itself(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        arm = _ARMS['responses']
        counter = TriageFailOpenCounter()
        with mock_openai_server() as server:
            server.set_response_sequence(
                arm.suffix, [(500, _ERROR_BODY), (200, responses_body(_DISTINCT))],
            )
            service = _service(arm, server)
            service.search = AsyncMock(
                return_value=SearchResults([search_candidate(score=_IN_JUDGE_BAND)]),
            )

            with caplog.at_level(logging.WARNING, logger='fused_memory.server.write_triage'):
                decision = await triage_write(
                    service, content='c', project_id='p', counter=counter, judge=judge_write,
                )

            assert len(server.requests_to(arm.suffix)) == 1

        assert decision.outcome == OUTCOME_STORED
        assert counter.live_count() == 1
        [record] = [
            r for r in caplog.records
            if 'write_triage fail-open at stage=judge' in r.getMessage()
        ]
        assert 'exc=InternalServerError' in record.getMessage()
        assert record.exc_info is not None
        assert isinstance(record.exc_info[1], openai.InternalServerError)
