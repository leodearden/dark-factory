"""The shared OpenAI Responses helper, its usage reader, host rule and list prices.

The SDK is faked under ``patch('openai.AsyncOpenAI')`` with
``_sdk_client_stubs.py::sdk_client_double``. No network.
"""
from __future__ import annotations

import asyncio
import dataclasses
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _sdk_client_stubs import sdk_client_double

from fused_memory.llm.list_prices import list_price
from fused_memory.llm.openai_responses import (
    ResponsesReply,
    TokenUsage,
    call_openai_responses,
    endpoint_host,
    read_token_usage,
)

_FORMAT = {'type': 'json_object'}


def _response(
    text: str | None = '{"ok": true}',
    *,
    status: str = 'completed',
    reason: str | None = None,
    model: object = 'gpt-6.1-sol-2026-10-01',
    usage: object = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        output_text=text,
        status=status,
        incomplete_details=(
            types.SimpleNamespace(reason=reason) if reason is not None else None
        ),
        model=model,
        usage=usage,
    )


def _client(response: object) -> MagicMock:
    client = sdk_client_double()
    client.responses.create = AsyncMock(return_value=response)
    return client


async def _call(client: MagicMock, **overrides) -> tuple[ResponsesReply, MagicMock]:
    """Run the helper against *client*; the reply and the faked constructor."""
    kwargs = {
        'model': 'gpt-6.1-sol',
        'instructions': 'be terse',
        'user_content': 'the input',
        'text_format': _FORMAT,
        'max_output_tokens': 4000,
        'timeout': 5.0,
        'reasoning_effort': None,
    }
    kwargs.update(overrides)
    client_kwargs = kwargs.pop('client_kwargs', {'api_key': 'k', 'base_url': 'https://x/v1'})
    with patch('openai.AsyncOpenAI', return_value=client) as ctor:
        reply = await call_openai_responses(client_kwargs, **kwargs)
    return reply, ctor


class TestTheRequest:
    @pytest.mark.asyncio
    async def test_without_effort_the_wire_carries_temperature_zero(self) -> None:
        client = _client(_response())
        await _call(client)
        client.responses.create.assert_awaited_once()
        assert client.responses.create.call_args.kwargs == {
            'model': 'gpt-6.1-sol',
            'instructions': 'be terse',
            'input': 'the input',
            'max_output_tokens': 4000,
            'text': {'format': _FORMAT},
            'temperature': 0.0,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize('effort', ['none', 'low', 'medium', 'high'])
    async def test_a_set_effort_replaces_temperature(self, effort: str) -> None:
        client = _client(_response())
        await _call(client, reasoning_effort=effort)
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs['reasoning'] == {'effort': effort}
        assert 'temperature' not in kwargs

    @pytest.mark.asyncio
    async def test_the_client_is_built_from_the_kwargs_and_max_retries(self) -> None:
        client = _client(_response())
        _, ctor = await _call(
            client, client_kwargs={'api_key': 'sk', 'base_url': 'https://h/v1'}, max_retries=2,
        )
        ctor.assert_called_once_with(api_key='sk', base_url='https://h/v1', max_retries=2)

    @pytest.mark.asyncio
    async def test_max_retries_defaults_to_one_attempt(self) -> None:
        client = _client(_response())
        _, ctor = await _call(client, client_kwargs={})
        ctor.assert_called_once_with(max_retries=0)

    @pytest.mark.asyncio
    async def test_the_client_is_closed_after_the_call(self) -> None:
        client = _client(_response())
        await _call(client)
        client.__aexit__.assert_awaited_once()


class TestTheReply:
    @pytest.mark.asyncio
    async def test_a_completed_answer(self) -> None:
        usage = types.SimpleNamespace(
            input_tokens=900, output_tokens=60,
            output_tokens_details=types.SimpleNamespace(reasoning_tokens=40),
        )
        reply, _ = await _call(_client(_response('{"a": 1}', usage=usage)))
        assert reply == ResponsesReply(
            text='{"a": 1}',
            incomplete=False,
            incomplete_reason=None,
            resolved_model='gpt-6.1-sol-2026-10-01',
            usage=TokenUsage(900, 60, 40),
        )

    @pytest.mark.asyncio
    async def test_the_reply_is_frozen(self) -> None:
        reply, _ = await _call(_client(_response()))
        with pytest.raises(dataclasses.FrozenInstanceError):
            reply.text = 'other'  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_an_incomplete_answer_is_returned_as_data(self) -> None:
        reply, _ = await _call(_client(_response(
            '{"partial', status='incomplete', reason='max_output_tokens',
        )))
        assert reply.incomplete is True
        assert reply.incomplete_reason == 'max_output_tokens'
        assert reply.text == '{"partial'

    @pytest.mark.asyncio
    async def test_a_missing_output_text_is_none(self) -> None:
        reply, _ = await _call(_client(_response(None)))
        assert reply.text is None

    @pytest.mark.asyncio
    async def test_a_non_string_model_is_not_a_resolved_model(self) -> None:
        reply, _ = await _call(_client(_response(model=MagicMock())))
        assert reply.resolved_model is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'usage',
        [
            None,
            types.SimpleNamespace(input_tokens=True, output_tokens=5),
            types.SimpleNamespace(input_tokens=5),
            MagicMock(),
        ],
        ids=['absent', 'bool-count', 'missing-field', 'mock-attributes'],
    )
    async def test_unreadable_usage_is_none(self, usage: object) -> None:
        reply, _ = await _call(_client(_response(usage=usage)))
        assert reply.usage is None

    @pytest.mark.asyncio
    async def test_missing_reasoning_details_give_no_reasoning_count(self) -> None:
        usage = types.SimpleNamespace(input_tokens=10, output_tokens=2)
        reply, _ = await _call(_client(_response(usage=usage)))
        assert reply.usage == TokenUsage(10, 2, None)


class TestFailuresPropagate:
    @pytest.mark.asyncio
    async def test_an_sdk_exception_propagates_unwrapped(self) -> None:
        class Boom(Exception):
            pass

        client = sdk_client_double()
        client.responses.create = AsyncMock(side_effect=Boom('x'))
        with pytest.raises(Boom):
            await _call(client)

    @pytest.mark.asyncio
    async def test_a_call_past_its_timeout_raises_timeout_error(self) -> None:
        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(5)

        client = sdk_client_double()
        client.responses.create = AsyncMock(side_effect=_hang)
        with pytest.raises(TimeoutError):
            await _call(client, timeout=0.01)


class TestReadTokenUsage:
    def test_the_chat_completions_field_names(self) -> None:
        usage = types.SimpleNamespace(
            prompt_tokens=7, completion_tokens=3,
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=1),
        )
        assert read_token_usage(
            usage, input_field='prompt_tokens', output_field='completion_tokens',
            details_field='completion_tokens_details',
        ) == TokenUsage(7, 3, 1)

    def test_the_anthropic_field_names_have_no_details(self) -> None:
        usage = types.SimpleNamespace(input_tokens=7, output_tokens=3)
        assert read_token_usage(
            usage, input_field='input_tokens', output_field='output_tokens',
            details_field=None,
        ) == TokenUsage(7, 3, None)

    def test_a_bool_reasoning_count_is_not_a_count(self) -> None:
        usage = types.SimpleNamespace(
            input_tokens=7, output_tokens=3,
            output_tokens_details=types.SimpleNamespace(reasoning_tokens=False),
        )
        assert read_token_usage(
            usage, input_field='input_tokens', output_field='output_tokens',
            details_field='output_tokens_details',
        ) == TokenUsage(7, 3, None)


class TestEndpointHost:
    def test_credentials_are_dropped_and_the_port_kept(self) -> None:
        assert endpoint_host('https://user:k@api.example.com:8443/v1') == 'api.example.com:8443'

    def test_a_plain_url(self) -> None:
        assert endpoint_host('https://api.openai.com/v1') == 'api.openai.com'

    @pytest.mark.parametrize('url', [None, ''])
    def test_no_url_is_no_host(self, url: str | None) -> None:
        assert endpoint_host(url) is None


class TestListPrices:
    def test_a_priced_model(self) -> None:
        price = list_price('gpt-6.1-sol')
        assert price is not None
        assert price.usd(1_000_000, 100_000) == pytest.approx(3.0)

    def test_an_unpriced_model_is_none(self) -> None:
        assert list_price('no-such-model') is None
