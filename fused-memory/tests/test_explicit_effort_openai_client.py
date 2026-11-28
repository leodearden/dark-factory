"""ExplicitEffortOpenAIClient on the wire (task 9943, promoted from the O6 replay script).

Every assertion reads what a real local server received; nothing is read from
graphiti_core's internals.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from _mock_openai_server import MockOpenAIServer, mock_openai_server, responses_body
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel

from fused_memory.backends.llm_clients import ExplicitEffortOpenAIClient
from fused_memory.backends.llm_token_usage import (
    AttributingTokenUsageTracker,
    LlmTokenUsage,
    measure_llm_tokens,
)

_MODEL = 'gpt-6.1-sol'
_MAX_TOKENS = 4096


class _Answer(BaseModel):
    name: str


_ANSWER = json.dumps({'name': 'Alice'})


@pytest.fixture
def server() -> Iterator[MockOpenAIServer]:
    with mock_openai_server() as srv:
        srv.set_response('/responses', responses_body(_ANSWER, input_tokens=11, output_tokens=5))
        yield srv


def _built(server: MockOpenAIServer) -> OpenAIClient:
    built = OpenAIClient(
        config=GraphitiLLMConfig(
            api_key='k', model=_MODEL, small_model=_MODEL, base_url=server.base_url, max_tokens=_MAX_TOKENS,
        ),
        max_tokens=_MAX_TOKENS,
    )
    built.token_tracker = AttributingTokenUsageTracker()
    return built


def _messages() -> list[Message]:
    return [Message(role='system', content='extract'), Message(role='user', content='the record')]


class TestTheWireCarriesExactlyTheChosenEffort:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('effort', ['low', 'medium'])
    async def test_a_structured_call_sends_the_effort_and_no_other_sampling_control(
        self, server: MockOpenAIServer, effort: ReasoningEffort,
    ) -> None:
        client = ExplicitEffortOpenAIClient(_built(server), effort)

        result = await client.generate_response(
            _messages(), response_model=_Answer, prompt_name='extract_edges.edge',
        )

        assert result == {'name': 'Alice'}
        [sent] = server.requests_to('/responses')
        body = sent['json_body']
        assert set(body) == {'model', 'input', 'max_output_tokens', 'text', 'reasoning'}
        assert body['model'] == _MODEL
        assert body['reasoning'] == {'effort': effort}
        assert set(body['text']) == {'format'}
        assert body['max_output_tokens'] == client.max_tokens == _MAX_TOKENS
        assert [message['role'] for message in body['input']] == ['system', 'user']

    @pytest.mark.parametrize('effort', ['low', 'medium'])
    def test_it_declares_the_effort_it_sends(self, server: MockOpenAIServer, effort: ReasoningEffort) -> None:
        assert ExplicitEffortOpenAIClient(_built(server), effort).reasoning_effort == effort

    @pytest.mark.asyncio
    async def test_a_json_mode_call_is_refused_before_anything_is_sent(
        self, server: MockOpenAIServer,
    ) -> None:
        client = ExplicitEffortOpenAIClient(_built(server), 'low')

        with pytest.raises(NotImplementedError, match='reasoning.effort'):
            await client.generate_response(_messages(), response_model=None)

        assert server.requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('message', [
        {'role': 'assistant', 'content': 'an earlier answer'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'a content part'}]},
    ])
    async def test_a_message_graphiti_never_emits_is_refused_before_anything_is_sent(
        self, server: MockOpenAIServer, message: Any,
    ) -> None:
        """graphiti's message conversion emits only system and user text, so through
        generate_response this is unreachable; the hook graphiti calls is where it is refused."""
        client = ExplicitEffortOpenAIClient(_built(server), 'low')
        messages: list[Any] = [{'role': 'system', 'content': 'extract'}, message]

        with pytest.raises(TypeError, match='is not Responses input'):
            await client._create_structured_completion(
                model=_MODEL, messages=messages, temperature=None,
                max_tokens=_MAX_TOKENS, response_model=_Answer,
            )

        assert server.requests == []

    def test_no_effort_is_refused(self, server: MockOpenAIServer) -> None:
        with pytest.raises(ValueError):
            ExplicitEffortOpenAIClient(_built(server), None)


class TestTheBuiltClientsTrackerIsReused:
    @pytest.mark.asyncio
    async def test_a_measured_window_credits_the_structured_call(self, server: MockOpenAIServer) -> None:
        built = _built(server)
        client = ExplicitEffortOpenAIClient(built, 'low')

        async with measure_llm_tokens(client) as measurement:
            await client.generate_response(
                _messages(), response_model=_Answer, prompt_name='extract_edges.edge',
            )

        assert client.token_tracker is built.token_tracker
        assert measurement.usage == LlmTokenUsage(11, 5, 1)
