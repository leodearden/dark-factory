"""ExplicitEffortOpenAIClient on the wire (task 9943, promoted from the O6 replay script).

Every assertion reads what a real local server received; nothing is read from
graphiti_core's internals.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from _mock_openai_server import MockOpenAIServer, mock_openai_server, responses_body
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.backends.llm_clients import ExplicitEffortOpenAIClient
from fused_memory.backends.llm_token_usage import (
    AttributingTokenUsageTracker,
    LlmTokenUsage,
    measure_llm_tokens,
)

_MODEL = 'gpt-6.1-sol'


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
            api_key='k', model=_MODEL, small_model=_MODEL, base_url=server.base_url, max_tokens=4096,
        ),
        max_tokens=4096,
    )
    built.token_tracker = AttributingTokenUsageTracker()
    return built


def _messages() -> list[Message]:
    return [Message(role='user', content='the record')]


class TestTheWireCarriesExactlyTheChosenEffort:
    @pytest.mark.asyncio
    async def test_a_structured_call_sends_effort_and_no_temperature_or_verbosity(
        self, server: MockOpenAIServer,
    ) -> None:
        client = ExplicitEffortOpenAIClient(_built(server), 'low')

        result = await client.generate_response(
            _messages(), response_model=_Answer, prompt_name='extract_edges.edge',
        )

        assert result == {'name': 'Alice'}
        sent = server.requests_to('/responses')
        assert len(sent) == 1
        body = sent[0]['json_body']
        assert body['model'] == _MODEL
        assert body['reasoning'] == {'effort': 'low'}
        assert 'temperature' not in body
        assert 'verbosity' not in body.get('text', {})

    @pytest.mark.asyncio
    async def test_a_json_mode_call_is_refused_before_anything_is_sent(
        self, server: MockOpenAIServer,
    ) -> None:
        client = ExplicitEffortOpenAIClient(_built(server), 'low')

        with pytest.raises(NotImplementedError):
            await client.generate_response(_messages(), response_model=None)

        assert server.requests_to('/responses') == []
        assert server.requests_to('/chat/completions') == []

    def test_no_effort_is_refused(self, server: MockOpenAIServer) -> None:
        with pytest.raises(ValueError):
            ExplicitEffortOpenAIClient(_built(server), None)


class TestTheBuiltClientsTrackerIsReused:
    @pytest.mark.asyncio
    async def test_a_measured_window_credits_the_structured_call(self, server: MockOpenAIServer) -> None:
        client = ExplicitEffortOpenAIClient(_built(server), 'low')

        async with measure_llm_tokens(client) as measurement:
            await client.generate_response(
                _messages(), response_model=_Answer, prompt_name='extract_edges.edge',
            )

        assert measurement.usage == LlmTokenUsage(11, 5, 1)
