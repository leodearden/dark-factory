"""arm_config: one arm's FusedMemoryConfig variant, built without mutating the base."""

import pytest
from _mock_openai_server import mock_openai_server
from arm_harness._fakes import embedding_spec, incumbent_control_spec, llm_spec
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.prompts.models import Message

from fused_memory.arm_harness.arm_config import (
    LOCAL_ARM_API_KEY,
    embedding_arm_config,
    llm_arm_config,
)
from fused_memory.backends.graphiti_client import build_llm_client
from fused_memory.backends.llm_clients import (
    ForceJsonObjectOpenAIGenericClient,
    TokenRecordingOpenAIGenericClient,
)
from fused_memory.config.schema import EmbedderConfig, LLMConfig


def _header(request: dict, name: str) -> str | None:
    lowered = {key.lower(): value for key, value in request['headers'].items()}
    return lowered.get(name.lower())


def test_llm_variant_is_a_new_object_and_base_is_unchanged(mock_config):
    before = mock_config.model_dump()

    cfg = llm_arm_config(llm_spec(), mock_config)

    assert cfg is not mock_config
    assert mock_config.model_dump() == before


def test_embedding_variant_is_a_new_object_and_base_is_unchanged(mock_config):
    before = mock_config.model_dump()

    cfg = embedding_arm_config(embedding_spec(), mock_config)

    assert cfg is not mock_config
    assert mock_config.model_dump() == before


def test_llm_block_carries_the_arm(mock_config):
    spec = llm_spec(model_id='qwen3-14b', params={'temperature': 0.3, 'max_tokens': 2048})

    llm = llm_arm_config(spec, mock_config).llm

    assert llm.model == 'qwen3-14b'
    assert llm.client_class == 'openai_generic'
    assert llm.temperature == 0.3
    assert llm.max_tokens == 2048
    assert llm.providers.openai is not None
    assert llm.providers.openai.api_url == spec.serving.base_url


@pytest.mark.parametrize(
    ('arm_mode', 'config_mode'),
    [('json_schema', 'auto'), ('json_object', 'json_object')],
)
def test_structured_output_mode_is_mapped_to_the_config_vocabulary(
    mock_config, arm_mode, config_mode
):
    spec = llm_spec(structured_output_mode=arm_mode)

    assert llm_arm_config(spec, mock_config).llm.structured_output_mode == config_mode


def test_llm_variant_leaves_the_embedder_block_identical(mock_config):
    cfg = llm_arm_config(llm_spec(), mock_config)

    assert cfg.embedder.model_dump() == mock_config.embedder.model_dump()


def test_embedding_variant_sets_only_the_embedder_block(mock_config):
    spec = embedding_spec(embedding_dim=768)

    cfg = embedding_arm_config(spec, mock_config)

    assert cfg.llm.model_dump() == mock_config.llm.model_dump()
    assert cfg.embedder.model == spec.model_id
    assert cfg.embedder.dimensions == 768
    assert cfg.embedder.providers.openai is not None
    assert cfg.embedder.providers.openai.api_url == spec.serving.base_url


def test_local_llm_arm_never_receives_the_incumbent_key(mock_config):
    llm = llm_arm_config(llm_spec(), mock_config).llm

    assert llm.providers.openai is not None
    assert llm.providers.openai.api_key == LOCAL_ARM_API_KEY
    assert LOCAL_ARM_API_KEY != 'test-key'


def test_local_embedding_arm_never_receives_the_incumbent_key(mock_config):
    embedder = embedding_arm_config(embedding_spec(), mock_config).embedder

    assert embedder.providers.openai is not None
    assert embedder.providers.openai.api_key == LOCAL_ARM_API_KEY


def test_metered_arm_keeps_the_base_key(mock_config):
    llm = llm_arm_config(incumbent_control_spec(), mock_config).llm

    assert llm.providers.openai is not None
    assert llm.providers.openai.api_key == 'test-key'


@pytest.mark.parametrize('mode', ['json_schema', 'json_object'])
def test_produced_blocks_pass_their_own_validators(mock_config, mode):
    llm = llm_arm_config(llm_spec(structured_output_mode=mode), mock_config).llm
    embedder = embedding_arm_config(embedding_spec(), mock_config).embedder

    assert LLMConfig.model_validate(llm.model_dump()) == llm
    assert EmbedderConfig.model_validate(embedder.model_dump()) == embedder


@pytest.mark.parametrize(
    ('spec', 'client_type'),
    [
        (llm_spec(structured_output_mode='json_object'), ForceJsonObjectOpenAIGenericClient),
        (llm_spec(structured_output_mode='json_schema'), TokenRecordingOpenAIGenericClient),
        (incumbent_control_spec(client_class='openai'), OpenAIClient),
    ],
)
def test_beta_seam_builds_the_arm_client_class(mock_config, spec, client_type):
    client = build_llm_client(llm_arm_config(spec, mock_config))

    assert type(client) is client_type


@pytest.mark.asyncio
async def test_generic_arm_traffic_reaches_the_arm_base_url(mock_config):
    with mock_openai_server() as server:
        spec = llm_spec(base_url=server.base_url)
        client = build_llm_client(llm_arm_config(spec, mock_config))
        assert client is not None

        await client.generate_response([Message(role='user', content='Alice knows Bob.')])

        received = server.requests_to('/chat/completions')

    assert len(received) == 1
    assert _header(received[0], 'Authorization') == f'Bearer {LOCAL_ARM_API_KEY}'


def test_llm_variant_refuses_an_embedding_spec(mock_config):
    with pytest.raises(TypeError, match='embedding'):
        llm_arm_config(embedding_spec(), mock_config)  # type: ignore[arg-type]


def test_embedding_variant_refuses_an_llm_spec(mock_config):
    with pytest.raises(TypeError, match='llm'):
        embedding_arm_config(llm_spec(), mock_config)  # type: ignore[arg-type]
