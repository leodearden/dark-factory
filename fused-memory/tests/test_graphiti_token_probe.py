"""Per-write LLM token attribution (task 3716).

One graphiti LLM client, and so one token tracker, is shared by every
concurrent write. These tests pin that a measurement window counts exactly
the tokens spent by its own asyncio context and nothing else.

No test here builds a ``Graphiti``, a ``FalkorDriver`` or calls
``GraphitiBackend.initialize()``. LLM clients come only from the driver-free
``build_llm_client`` seam or direct construction with an injected fake
``client=``, and the only sockets are the loopback ``_mock_openai_server``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from _mock_openai_server import mock_openai_server
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.token_tracker import TokenUsageTracker
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, ValidationError

from fused_memory.backends.graphiti_client import GraphitiBackend, build_llm_client
from fused_memory.backends.llm_clients import ForceJsonObjectOpenAIGenericClient
from fused_memory.backends.llm_token_usage import (
    AttributingTokenUsageTracker,
    LlmTokenUsage,
    measure_llm_tokens,
)
from fused_memory.config.schema import (
    AnthropicProviderConfig,
    FusedMemoryConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
)


def _attributing_client() -> SimpleNamespace:
    return SimpleNamespace(token_tracker=AttributingTokenUsageTracker())


@pytest.mark.asyncio
async def test_window_reports_the_tokens_recorded_inside_it():
    client = _attributing_client()

    async with measure_llm_tokens(client) as m:
        client.token_tracker.record('extract_nodes', 30, 10)

    assert m.usage is not None
    assert m.usage == LlmTokenUsage(input_tokens=30, output_tokens=10, llm_calls=1)
    assert m.usage.total_tokens == 40
    assert m.usage.as_journal_dict() == {
        'input_tokens': 30,
        'output_tokens': 10,
        'total_tokens': 40,
        'llm_calls': 1,
    }


@pytest.mark.asyncio
async def test_records_outside_the_window_are_excluded_but_still_cumulative():
    client = _attributing_client()
    client.token_tracker.record('before', 1000, 100)

    async with measure_llm_tokens(client) as m:
        client.token_tracker.record('inside', 30, 10)

    client.token_tracker.record('after', 2000, 200)

    assert m.usage == LlmTokenUsage(30, 10, 1)
    total = client.token_tracker.get_total_usage()
    assert (total.input_tokens, total.output_tokens) == (3030, 310)


@pytest.mark.asyncio
async def test_concurrent_windows_over_one_tracker_each_count_only_their_own():
    client = _attributing_client()
    record = client.token_tracker.record
    a_open, b_open = asyncio.Event(), asyncio.Event()
    a_first, b_recorded, a_second = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def window_a():
        async with measure_llm_tokens(client) as m:
            a_open.set()
            await b_open.wait()
            record('extract_nodes', 100, 40)
            a_first.set()
            await b_recorded.wait()
            record('extract_edges', 5, 5)
            a_second.set()
        return m

    async def window_b():
        await a_open.wait()
        async with measure_llm_tokens(client) as m:
            b_open.set()
            await a_first.wait()
            record('dedupe_nodes', 7, 3)
            b_recorded.set()
            await a_second.wait()
        return m

    a, b = await asyncio.wait_for(
        asyncio.gather(asyncio.create_task(window_a()), asyncio.create_task(window_b())),
        10,
    )

    assert a.usage == LlmTokenUsage(105, 45, 2)
    assert b.usage == LlmTokenUsage(7, 3, 1)


@pytest.mark.asyncio
async def test_records_from_child_tasks_spawned_in_the_window_are_attributed():
    client = _attributing_client()

    async def _llm_call(prompt_name: str, input_tokens: int, output_tokens: int):
        await asyncio.sleep(0)
        client.token_tracker.record(prompt_name, input_tokens, output_tokens)

    async with measure_llm_tokens(client) as m:
        await asyncio.gather(
            _llm_call('extract_nodes', 10, 1),
            _llm_call('extract_edges', 20, 2),
        )

    assert m.usage == LlmTokenUsage(30, 3, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'llm_client',
    [None, SimpleNamespace(token_tracker=TokenUsageTracker())],
    ids=['no-client', 'non-attributing-tracker'],
)
async def test_unmeasurable_client_yields_no_usage_and_still_runs_the_body(llm_client):
    body_ran = False

    async with measure_llm_tokens(llm_client) as m:
        body_ran = True

    assert body_ran
    assert m.usage is None


@pytest.mark.asyncio
async def test_measured_window_without_llm_calls_is_a_real_zero():
    async with measure_llm_tokens(_attributing_client()) as m:
        pass

    assert m.usage == LlmTokenUsage(0, 0, 0)


@pytest.mark.asyncio
async def test_raising_body_propagates_and_still_freezes_the_tokens_it_burned():
    client = _attributing_client()

    with pytest.raises(RuntimeError, match='extraction failed'):
        async with measure_llm_tokens(client) as m:
            client.token_tracker.record('extract_nodes', 10, 2)
            raise RuntimeError('extraction failed')

    assert m.usage == LlmTokenUsage(10, 2, 1)


class TestGraphitiBackendTokenProbe:
    @pytest.mark.asyncio
    async def test_without_an_llm_client_the_probe_measures_nothing(self, mock_config):
        backend = GraphitiBackend(mock_config)

        async with backend.token_probe() as m:
            pass

        assert m.usage is None

    @pytest.mark.asyncio
    async def test_probe_counts_tokens_on_the_backend_llm_client(self, mock_config):
        backend = GraphitiBackend(mock_config)
        client = build_llm_client(mock_config)
        assert client is not None
        backend._llm_client = client

        async with backend.token_probe() as m:
            client.token_tracker.record('extract_nodes', 12, 3)

        assert m.usage == LlmTokenUsage(12, 3, 1)


# ── every arm of the production construction seam records attributable tokens ──

_UNROUTABLE_SENTINEL = 'http://127.0.0.1:1/v1'
_PROMPT_NAME = 'extract_nodes.extract_message'


@pytest.fixture
def no_ambient_openai_env(monkeypatch):
    for var in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_URL'):
        monkeypatch.setenv(var, _UNROUTABLE_SENTINEL)
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key-must-not-be-used')


def _openai_arm_config(
    base_url: str,
    client_class: Literal['openai', 'openai_generic'],
    mode: Literal['auto', 'json_object'],
) -> FusedMemoryConfig:
    return FusedMemoryConfig(
        llm=LLMConfig(
            provider='openai',
            client_class=client_class,
            structured_output_mode=mode,
            model='mock-model',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key', api_url=base_url),
            ),
        ),
    )


def _chat_body(content: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        'id': 'chatcmpl-usage',
        'object': 'chat.completion',
        'created': 1700000000,
        'model': 'mock-model',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': 'stop',
            },
        ],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    }


@pytest.mark.timeout(60)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('client_class', 'mode'),
    [('openai', 'auto'), ('openai_generic', 'auto'), ('openai_generic', 'json_object')],
)
async def test_wire_usage_is_recorded_and_attributed_on_every_openai_shaped_arm(
    no_ambient_openai_env, client_class, mode,
):
    with mock_openai_server() as server:
        server.set_response('/chat/completions', _chat_body('{"ok": true}', 120, 45))
        client = build_llm_client(_openai_arm_config(server.base_url, client_class, mode))
        assert client is not None

        async with measure_llm_tokens(client) as m:
            await client.generate_response(
                [Message(role='system', content='s'), Message(role='user', content='u')],
                prompt_name=_PROMPT_NAME,
            )

    assert m.usage == LlmTokenUsage(120, 45, 1)
    assert client.token_tracker.get_usage()[_PROMPT_NAME].call_count == 1


@pytest.mark.asyncio
async def test_anthropic_arm_gets_an_attributing_tracker_too(mock_config):
    mock_config.llm.provider = 'anthropic'
    mock_config.llm.providers.anthropic = AnthropicProviderConfig(api_key='anthropic-test-key')
    client = build_llm_client(mock_config)
    assert client is not None

    async with measure_llm_tokens(client) as m:
        client.token_tracker.record('extract_nodes', 9, 4)

    assert m.usage == LlmTokenUsage(9, 4, 1)


class _Inner(BaseModel):
    name: str


class _Extraction(BaseModel):
    items: list[_Inner]


def _completion(content: str, prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


_OFF_ENVELOPE = '{"ok": true}'
_VALID_EXTRACTION = json.dumps({'items': [{'name': 'Alice'}]})


def _generic_client_over(*responses: SimpleNamespace) -> ForceJsonObjectOpenAIGenericClient:
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(side_effect=list(responses))
    return ForceJsonObjectOpenAIGenericClient(
        config=GraphitiLLMConfig(api_key='k', model='m'), client=fake,
    )


@pytest.mark.asyncio
async def test_generic_client_records_only_the_successful_attempt_like_upstream():
    client = _generic_client_over(
        _completion(_OFF_ENVELOPE, 50, 5),
        _completion(_VALID_EXTRACTION, 60, 6),
    )

    await client.generate_response(
        [Message(role='system', content='s'), Message(role='user', content='u')],
        response_model=_Extraction,
        prompt_name=_PROMPT_NAME,
    )

    usage = client.token_tracker.get_usage()
    assert list(usage) == [_PROMPT_NAME]
    assert usage[_PROMPT_NAME].call_count == 1
    assert (usage[_PROMPT_NAME].total_input_tokens, usage[_PROMPT_NAME].total_output_tokens) == (
        60,
        6,
    )


@pytest.mark.asyncio
async def test_generic_client_records_nothing_when_every_attempt_fails_like_upstream():
    attempts = ForceJsonObjectOpenAIGenericClient.MAX_RETRIES + 1
    client = _generic_client_over(*[_completion(_OFF_ENVELOPE, 50, 5)] * attempts)

    with pytest.raises(ValidationError):
        await client.generate_response(
            [Message(role='system', content='s'), Message(role='user', content='u')],
            response_model=_Extraction,
            prompt_name=_PROMPT_NAME,
        )

    assert client.token_tracker.get_usage() == {}
