"""B5: the startup endpoint probe measures what the embedder endpoint really serves.

The probe calls ``/v1/models`` and ``/v1/embeddings`` directly, because
graphiti's OpenAIEmbedder truncates to ``embedding_dim`` and so would hide a
model wider than the config. It reports typed faults and never raises for a
network fault, coerces a value, or falls back to an ambient endpoint.
"""

import json
import socket
import time
from collections.abc import Iterator

import pytest
from _mock_openai_server import MockOpenAIServer, mock_openai_server

from fused_memory.config.schema import (
    EmbedderConfig,
    EmbedderProvidersConfig,
    OpenAIProviderConfig,
)
from fused_memory.embedder.endpoint_probe import (
    EmbedderEndpointProbe,
    EmbedderProbeResult,
    ProbeFaultKind,
    probe_embedder_endpoint,
)

MODEL = 'granite-x'
DIMS = 768
TIMEOUT = 2.0


@pytest.fixture
def endpoint() -> Iterator[MockOpenAIServer]:
    with mock_openai_server() as server:
        server.served_models = [MODEL]
        server.embedding_len = DIMS
        yield server


def _embedder(api_url: str, *, api_key: str = 'test-key') -> EmbedderConfig:
    return EmbedderConfig(
        model=MODEL,
        dimensions=DIMS,
        providers=EmbedderProvidersConfig(
            openai=OpenAIProviderConfig(api_key=api_key, api_url=api_url),
        ),
    )


def _dead_url() -> str:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    return f'http://127.0.0.1:{port}/v1'


def _probe(embedder: EmbedderConfig) -> EmbedderProbeResult:
    return probe_embedder_endpoint(embedder, timeout_seconds=TIMEOUT)


def _kinds(result: EmbedderProbeResult) -> list[ProbeFaultKind]:
    return [fault.kind for fault in result.faults]


class TestProbeResult:
    def test_a_healthy_endpoint_reports_its_native_width_and_models(self, endpoint):
        result = _probe(_embedder(endpoint.base_url))

        assert result.reachable is True
        assert result.faults == ()
        assert result.ok is True
        assert result.native_width == DIMS
        assert result.served_models == (MODEL,)

    @pytest.mark.parametrize('served_width', [1024, 1536])
    def test_a_width_other_than_the_config_is_a_mismatch(self, endpoint, served_width):
        endpoint.embedding_len = served_width

        result = _probe(_embedder(endpoint.base_url))

        [fault] = result.faults
        assert fault.kind is ProbeFaultKind.WIDTH_MISMATCH
        assert fault.expected == DIMS
        assert fault.observed == served_width
        assert result.native_width == served_width

    def test_a_model_the_endpoint_does_not_serve_is_reported(self, endpoint):
        endpoint.served_models = ['some-other-model']

        result = _probe(_embedder(endpoint.base_url))

        [fault] = result.faults
        assert fault.kind is ProbeFaultKind.MODEL_NOT_SERVED
        assert fault.expected == MODEL
        assert fault.observed == ('some-other-model',)

    def test_an_endpoint_nothing_listens_on_is_unreachable_and_returns_in_bound(self):
        started = time.monotonic()

        result = _probe(_embedder(_dead_url()))

        assert time.monotonic() - started < TIMEOUT + 1.0
        assert result.reachable is False
        assert _kinds(result) == [ProbeFaultKind.UNREACHABLE]

    def test_an_absent_provider_block_is_not_configured(self):
        embedder = EmbedderConfig(model=MODEL, dimensions=DIMS)

        result = _probe(embedder)

        assert _kinds(result) == [ProbeFaultKind.NOT_CONFIGURED]

    def test_an_empty_api_key_is_not_configured_and_sends_nothing(self, endpoint):
        result = _probe(_embedder(endpoint.base_url, api_key=''))

        assert _kinds(result) == [ProbeFaultKind.NOT_CONFIGURED]
        assert endpoint.requests == []

    def test_ambient_openai_endpoints_are_never_a_fallback(self, endpoint, monkeypatch):
        monkeypatch.setenv('OPENAI_BASE_URL', endpoint.base_url)
        monkeypatch.setenv('OPENAI_API_BASE', endpoint.base_url)

        result = _probe(_embedder(_dead_url()))

        assert _kinds(result) == [ProbeFaultKind.UNREACHABLE]
        assert endpoint.requests == []

    def test_the_probe_asks_for_the_configured_model_at_native_width(self, endpoint):
        _probe(_embedder(endpoint.base_url))

        [request] = endpoint.requests_to('/embeddings')
        assert 'dimensions' not in request['json_body']
        assert request['json_body']['model'] == MODEL

    def test_an_http_error_status_is_an_endpoint_error(self, endpoint):
        endpoint.set_response_sequence(
            '/embeddings', [(400, {'error': {'message': 'dimensions is not supported'}})],
        )

        result = _probe(_embedder(endpoint.base_url))

        assert result.reachable is True
        [fault] = result.faults
        assert fault.kind is ProbeFaultKind.ENDPOINT_ERROR
        assert fault.observed == 400

    def test_every_fault_renders_as_a_json_status_record(self, endpoint):
        endpoint.embedding_len = 1024
        endpoint.served_models = ['some-other-model']

        result = _probe(_embedder(endpoint.base_url))

        records = [fault.as_status() for fault in result.faults]
        assert len(records) == 2
        for record in records:
            assert set(record) == {'kind', 'expected', 'observed', 'detail'}
        json.dumps(records)
        json.dumps(result.as_status())


class TestEndpointProbeHolder:
    @pytest.mark.asyncio
    async def test_start_never_blocks_and_reports_pending(self, endpoint):
        probe = EmbedderEndpointProbe(_embedder(endpoint.base_url), timeout_seconds=TIMEOUT)

        probe.start()
        block = probe.status_block()

        assert block['status'] == 'pending'
        assert block['reachable'] is None
        assert block['model'] == MODEL
        assert block['endpoint'] == endpoint.base_url
        assert block['dimensions'] == DIMS
        await probe.wait()

    @pytest.mark.asyncio
    async def test_a_finished_probe_reports_ok(self, endpoint):
        probe = EmbedderEndpointProbe(_embedder(endpoint.base_url), timeout_seconds=TIMEOUT)

        probe.start()
        await probe.wait()
        block = probe.status_block()

        assert block['status'] == 'ok'
        assert block['reachable'] is True
        assert block['native_width'] == DIMS
        assert block['served_models'] == [MODEL]
        assert block['errors'] == []
        assert block['probed_at']
        assert {'model', 'endpoint', 'dimensions'} <= set(block)

    @pytest.mark.asyncio
    async def test_a_finished_probe_with_faults_reports_degraded(self, endpoint):
        endpoint.embedding_len = 1024
        probe = EmbedderEndpointProbe(_embedder(endpoint.base_url), timeout_seconds=TIMEOUT)

        probe.start()
        await probe.wait()
        block = probe.status_block()

        assert block['status'] == 'degraded'
        [error] = block['errors']
        assert error['kind'] == ProbeFaultKind.WIDTH_MISMATCH.value
        json.dumps(block)
