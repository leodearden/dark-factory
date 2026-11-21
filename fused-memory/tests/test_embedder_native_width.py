"""B4: no embedder client asks the endpoint for a ``dimensions`` parameter.

PRD C3 (plans/local-embedder-cutover-prd.md §6): a local vLLM endpoint
rejects ``dimensions`` for granite with a 400, so every client requests the
model's native width. Each client below is driven through a public seam
against the recording fake, configured at 768 so a width-dependent emission
rule cannot hide behind the 1536 default.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from _mem0_stub_backend import backend_against
from _mock_openai_server import MockOpenAIServer, mock_openai_server
from _stub_qdrant_server import stub_qdrant_server

from fused_memory.backends.graphiti_client import build_embedder
from fused_memory.config.schema import FusedMemoryConfig, OpenAIProviderConfig
from fused_memory.middleware.curator_types import CandidateTask
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.models.scope import Scope

DIMS = 768

CALIBRATE_SCRIPT = Path(__file__).parent.parent / 'scripts' / 'calibrate_write_triage.py'


@pytest.fixture
def endpoint() -> Iterator[MockOpenAIServer]:
    with mock_openai_server() as server:
        server.embedding_len = DIMS
        yield server


@pytest.fixture
def config(mock_config, endpoint, monkeypatch) -> FusedMemoryConfig:
    """``mock_config`` at 768 wide, embedding (and chatting) through the fake.

    mem0's telemetry is off, so no posthog call and no mem0migrations
    collection reaches the stub.
    """
    monkeypatch.setattr('mem0.memory.main.capture_event', lambda *a, **kw: None)
    monkeypatch.setattr('mem0.memory.main.MEM0_TELEMETRY', False)
    config = mock_config.model_copy(deep=True)
    config.embedder.dimensions = DIMS
    config.embedder.providers.openai = OpenAIProviderConfig(
        api_key='test-key', api_url=endpoint.base_url,
    )
    assert config.llm.providers.openai is not None
    config.llm.providers.openai.api_url = endpoint.base_url
    return config


def _assert_only_native_width_requested(endpoint: MockOpenAIServer) -> None:
    requests = endpoint.requests_to('/embeddings')
    assert requests, 'no embeddings request reached the configured endpoint'
    for request in requests:
        assert 'dimensions' not in request['json_body'], request['json_body']


@pytest.mark.asyncio
async def test_mem0_requests_the_native_width(config, endpoint):
    with stub_qdrant_server() as stub:
        async with backend_against(stub.url, config) as backend:
            await backend.add('A memory embedded at native width.', Scope(project_id='native'))

    _assert_only_native_width_requested(endpoint)


@pytest.mark.asyncio
async def test_graphiti_requests_the_native_width(config, endpoint):
    embedder = build_embedder(config)
    assert embedder is not None

    await embedder.create('x')

    _assert_only_native_width_requested(endpoint)


@pytest.mark.asyncio
async def test_the_task_curator_requests_the_native_width(config, endpoint):
    with stub_qdrant_server() as stub:
        config.mem0.qdrant_url = stub.url
        curator = TaskCurator(config)
        try:
            await curator.record_task('t1', CandidateTask(title='x'), 'native')
        finally:
            await curator.close()

    _assert_only_native_width_requested(endpoint)


def test_calibration_requests_the_native_width(config, endpoint):
    calibrate = load_script_module(CALIBRATE_SCRIPT, mod_name='calibrate_write_triage')

    calibrate.build_embed_fn(config)('m', 'x')

    _assert_only_native_width_requested(endpoint)
