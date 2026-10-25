"""The conformance audit: a hard, per-attempt response validator on the arm's real client."""

import json
import urllib.error
import urllib.request

import openai
import pytest
from _mock_openai_server import chat_completion_body, mock_openai_server, responses_body
from graphiti_core.llm_client import LLMClient, LLMConfig
from graphiti_core.llm_client.openai_base_client import BaseOpenAIClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, ValidationError

from arm_harness._fakes import UNREACHABLE_BASE_URL, incumbent_control_spec, llm_spec
from fused_memory.arm_harness.arm_config import llm_arm_config
from fused_memory.arm_harness.conformance import (
    ConformanceLedger,
    conformance_rate_metric,
    install_conformance_audit,
)
from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.backends.llm_clients import (
    ForceJsonObjectOpenAIGenericClient,
    TokenRecordingOpenAIGenericClient,
)
from fused_memory.backends.llm_token_usage import measure_llm_tokens


class _Entity(BaseModel):
    name: str


class _Entities(BaseModel):
    entities: list[_Entity]


VALID = json.dumps({'entities': [{'name': 'Alice'}]})
OFF_SCHEMA = json.dumps({'unexpected': 1})


def _messages() -> list[Message]:
    return [
        Message(role='system', content='Extract entities.'),
        Message(role='user', content='Alice knows Bob.'),
    ]


def _arm_client(
    mock_config, base_url: str, mode: str = 'json_schema'
) -> TokenRecordingOpenAIGenericClient:
    spec = llm_spec(base_url=base_url, structured_output_mode=mode)
    client = build_llm_client(llm_arm_config(spec, mock_config))
    assert isinstance(client, TokenRecordingOpenAIGenericClient)
    return client


def _audited_client(mock_config, base_url: str, mode: str = 'json_schema'):
    client = _arm_client(mock_config, base_url, mode)
    ledger = ConformanceLedger()
    install_conformance_audit(client, ledger)
    return client, ledger


def _post(url: str) -> int:
    request = urllib.request.Request(url, data=b'{}', method='POST')
    try:
        with urllib.request.urlopen(request) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_mock_server_serves_a_response_sequence_and_repeats_the_last():
    with mock_openai_server() as server:
        server.set_response_sequence('/chat/completions', [(500, {'e': 1}), (200, {'ok': 1})])
        url = f'{server.base_url}/chat/completions'

        statuses = [_post(url) for _ in range(3)]

    assert statuses == [500, 200, 200]


@pytest.mark.asyncio
async def test_schema_valid_response_is_counted_valid(mock_config):
    with mock_openai_server() as server:
        server.chat_content = VALID
        client, ledger = _audited_client(mock_config, server.base_url)

        result = await client.generate_response(_messages(), response_model=_Entities)

    counts = ledger.snapshot()
    assert result == {'entities': [{'name': 'Alice'}]}
    assert (counts.calls, counts.schema_valid, counts.schema_invalid) == (1, 1, 0)


@pytest.mark.asyncio
async def test_stock_json_schema_client_returns_off_schema_json_unchecked(mock_config):
    with mock_openai_server() as server:
        server.chat_content = OFF_SCHEMA
        client = _arm_client(mock_config, server.base_url)

        result = await client.generate_response(_messages(), response_model=_Entities)

    assert result == {'unexpected': 1}


@pytest.mark.asyncio
async def test_off_schema_json_is_detected_on_every_retried_attempt(mock_config):
    with mock_openai_server() as server:
        server.chat_content = OFF_SCHEMA
        client, ledger = _audited_client(mock_config, server.base_url)
        attempts = type(client).MAX_RETRIES + 1

        with pytest.raises(ValidationError):
            await client.generate_response(_messages(), response_model=_Entities)

        received = len(server.requests_to('/chat/completions'))

    counts = ledger.snapshot()
    assert received == attempts
    assert counts.calls == counts.schema_invalid == attempts
    assert counts.schema_valid == 0
    assert dict(counts.invalid_by_error_class) == {'ValidationError': attempts}


@pytest.mark.asyncio
async def test_json_object_arm_off_envelope_attempts_are_counted_invalid(mock_config):
    with mock_openai_server() as server:
        server.chat_content = OFF_SCHEMA
        client, ledger = _audited_client(mock_config, server.base_url, mode='json_object')
        assert isinstance(client, ForceJsonObjectOpenAIGenericClient)
        attempts = type(client).MAX_RETRIES + 1

        with pytest.raises(ValidationError):
            await client.generate_response(_messages(), response_model=_Entities)

    counts = ledger.snapshot()
    assert counts.calls == counts.schema_invalid == attempts


@pytest.mark.asyncio
async def test_undecodable_body_is_counted_invalid_by_its_error_class(mock_config):
    with mock_openai_server() as server:
        server.chat_content = 'not json at all'
        client, ledger = _audited_client(mock_config, server.base_url)
        attempts = type(client).MAX_RETRIES + 1

        with pytest.raises(json.JSONDecodeError):
            await client.generate_response(_messages(), response_model=_Entities)

    assert dict(ledger.snapshot().invalid_by_error_class) == {'JSONDecodeError': attempts}


@pytest.mark.asyncio
async def test_off_schema_then_valid_counts_both_attempts(mock_config):
    with mock_openai_server() as server:
        server.set_response_sequence(
            '/chat/completions',
            [(200, chat_completion_body(OFF_SCHEMA)), (200, chat_completion_body(VALID))],
        )
        client, ledger = _audited_client(mock_config, server.base_url)

        result = await client.generate_response(_messages(), response_model=_Entities)

    counts = ledger.snapshot()
    metric = conformance_rate_metric(counts)
    assert result == {'entities': [{'name': 'Alice'}]}
    assert (counts.calls, counts.schema_valid, counts.schema_invalid) == (2, 1, 1)
    assert metric is not None
    assert metric.metric_id == 'conformance-rate'
    assert metric.kind == 'proportion'
    assert metric.direction == 'lower_is_worse'
    assert (metric.value, metric.denominator, metric.n) == (0.5, 2, 2)


@pytest.mark.asyncio
async def test_http_500_is_a_transport_error_outside_the_denominator(mock_config):
    with mock_openai_server() as server:
        server.set_response_sequence('/chat/completions', [(500, {'error': {'message': 'boom'}})])
        client, ledger = _audited_client(mock_config, server.base_url)

        with pytest.raises(openai.InternalServerError):
            await client.generate_response(_messages(), response_model=_Entities)

    counts = ledger.snapshot()
    assert counts.transport_errors == 1
    assert counts.schema_valid + counts.schema_invalid == 0
    assert conformance_rate_metric(counts) is None


@pytest.mark.asyncio
async def test_connection_error_is_a_transport_error(mock_config):
    client, ledger = _audited_client(mock_config, UNREACHABLE_BASE_URL)

    with pytest.raises(openai.APIConnectionError):
        await client.generate_response(_messages(), response_model=_Entities)

    counts = ledger.snapshot()
    assert (counts.calls, counts.transport_errors) == (1, 1)


@pytest.mark.asyncio
async def test_json_object_without_response_model_counts_valid(mock_config):
    with mock_openai_server() as server:
        server.chat_content = '{"ok": true}'
        client, ledger = _audited_client(mock_config, server.base_url)

        await client.generate_response(_messages())

    assert ledger.snapshot().schema_valid == 1


@pytest.mark.asyncio
async def test_non_object_json_without_response_model_counts_invalid(mock_config):
    with mock_openai_server() as server:
        server.chat_content = '[1, 2]'
        client, ledger = _audited_client(mock_config, server.base_url)

        with pytest.raises(ValidationError):
            await client.generate_response(_messages())

    assert ledger.snapshot().schema_valid == 0
    assert ledger.snapshot().schema_invalid == type(client).MAX_RETRIES + 1


def test_conformance_rate_is_absent_when_no_response_was_received():
    assert conformance_rate_metric(ConformanceLedger().snapshot()) is None


def test_install_refuses_a_client_without_the_per_attempt_seam():
    class NoSeam:
        async def generate_response(self, *args, **kwargs):
            return {}

    with pytest.raises(TypeError, match='_generate_response'):
        install_conformance_audit(NoSeam(), ConformanceLedger())  # type: ignore[arg-type]


# --- the production-default client class: BaseOpenAIClient's tuple-returning attempt --------


def _incumbent_client(mock_config, base_url: str) -> BaseOpenAIClient:
    spec = incumbent_control_spec(serving={'stack': 'openai', 'base_url': base_url})
    client = build_llm_client(llm_arm_config(spec, mock_config))
    assert isinstance(client, BaseOpenAIClient)
    return client


def _audited_incumbent(mock_config, base_url: str):
    client = _incumbent_client(mock_config, base_url)
    ledger = ConformanceLedger()
    install_conformance_audit(client, ledger)
    return client, ledger


def test_the_incumbent_control_arm_runs_on_the_base_openai_client(mock_config):
    client = build_llm_client(llm_arm_config(incumbent_control_spec(), mock_config))

    assert isinstance(client, BaseOpenAIClient)


@pytest.mark.asyncio
async def test_base_openai_valid_response_is_counted_and_its_tokens_still_recorded(mock_config):
    with mock_openai_server() as server:
        server.set_response('/responses', responses_body(VALID, input_tokens=3, output_tokens=4))
        client, ledger = _audited_incumbent(mock_config, server.base_url)

        async with measure_llm_tokens(client) as measurement:
            result = await client.generate_response(_messages(), response_model=_Entities)

    counts = ledger.snapshot()
    usage = measurement.usage
    assert result == {'entities': [{'name': 'Alice'}]}
    assert (counts.calls, counts.schema_valid) == (1, 1)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (3, 4, 7)


@pytest.mark.asyncio
async def test_base_openai_off_schema_is_detected_on_every_retried_attempt(mock_config):
    with mock_openai_server() as server:
        server.set_response('/responses', responses_body(OFF_SCHEMA))
        client, ledger = _audited_incumbent(mock_config, server.base_url)
        attempts = type(client).MAX_RETRIES + 1

        with pytest.raises(ValidationError):
            await client.generate_response(_messages(), response_model=_Entities)

    counts = ledger.snapshot()
    assert counts.schema_invalid == counts.calls == attempts
    assert dict(counts.invalid_by_error_class) == {'ValidationError': attempts}


@pytest.mark.asyncio
async def test_base_openai_off_schema_then_valid_counts_both_attempts(mock_config):
    with mock_openai_server() as server:
        server.set_response_sequence(
            '/responses', [(200, responses_body(OFF_SCHEMA)), (200, responses_body(VALID))]
        )
        client, ledger = _audited_incumbent(mock_config, server.base_url)

        await client.generate_response(_messages(), response_model=_Entities)

    counts = ledger.snapshot()
    metric = conformance_rate_metric(counts)
    assert (counts.schema_valid, counts.schema_invalid) == (1, 1)
    assert metric is not None
    assert (metric.value, metric.denominator) == (0.5, 2)


@pytest.mark.asyncio
async def test_base_openai_json_object_response_without_response_model_counts_valid(mock_config):
    with mock_openai_server() as server:
        server.chat_content = '{"ok": true}'
        client, ledger = _audited_incumbent(mock_config, server.base_url)

        result = await client.generate_response(_messages())
        request = server.requests_to('/chat/completions')[0]['json_body']

    assert result == {'ok': True}
    assert request['response_format'] == {'type': 'json_object'}
    assert ledger.snapshot().schema_valid == 1


class _ForeignFamilyClient(LLMClient):
    async def _generate_response(self, messages, response_model=None, max_tokens=0, model_size=None):
        return {}


def test_install_refuses_an_attempt_seam_of_an_unknown_client_family():
    client = _ForeignFamilyClient(LLMConfig(api_key='x'))

    with pytest.raises(TypeError, match='_ForeignFamilyClient'):
        install_conformance_audit(client, ConformanceLedger())
