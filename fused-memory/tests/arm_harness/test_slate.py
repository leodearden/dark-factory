"""The candidate slate read from arms.yaml, and the ArmSpec each arm runs under (arm_harness/slate.py)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from arm_harness._fakes import (
    CODE_SHA,
    CORPUS_SHA,
    PREREG_SHA,
    QWEN_QUERY_PREFIX,
    embedding_slate_arm,
)
from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec, LlmParams, ServingSpec
from fused_memory.arm_harness.scratch_guard import (
    GuardCheckpoint,
    ScratchGuardError,
    require_scratch_name,
)
from fused_memory.arm_harness.slate import (
    SlateArm,
    arm_endpoint,
    candidate_spec,
    embedding_candidate_spec,
    embedding_control_spec,
    load_embedding_slate,
    load_llm_slate,
)

REPO_ROOT = Path(__file__).parents[3]
COMMITTED_ARMS = REPO_ROOT / 'scripts' / 'local-model-serving' / 'arms.yaml'
PARAMS = LlmParams(temperature=0.0, max_tokens=4096)
TAP_BASE_URL = 'http://127.0.0.1:8418/v1'

FIXTURE = """\
# a comment, as the real manifest carries many
port_block: [8410, 8417]

arms:
  - arm_id: qwen3.5-9b
    axis: llm
    stack: vllm
    image: vllm/vllm-openai:v0.26.0
    image_digest: sha256:ffff
    model_ref: QuantTrio/Qwen3.5-9B-AWQ
    quant: awq
    port: 8410
    served_model_name: qwen3.5-9b
    structured_output_mode: json_schema
    reasoning: 'on'
    reasoning_parser: qwen3
    est_vram_gib: 12.0
    max_model_len: 32768
    max_num_seqs: 8
    notes: >-
      free text the reader ignores

  - arm_id: bge-embed
    axis: embedding
    stack: vllm
    model_ref: BAAI/bge
    quant: none
    port: 8414
    served_model_name: bge-embed
    structured_output_mode: none
    est_vram_gib: 1.0
    max_model_len: 8192
    dims: 1024
    query_prefix: "Query: "

  - arm_id: moe-stretch
    axis: llm
    stack: llamacpp
    image: ghcr.io/ggml-org/llama.cpp:server-cuda-b10276
    model_ref: unsloth/gemma-4-26B-A4B-it-qat-GGUF
    gguf_file: gemma.gguf
    quant: q4_k_xl
    port: 8413
    served_model_name: moe-stretch
    structured_output_mode: json_object
    reasoning: 'off'
    est_vram_gib: 14.5
    max_model_len: 16384
    max_num_seqs: 4

  - arm_id: gte-embed
    axis: embedding
    stack: vllm
    fallback_stack: tei
    model_ref: Alibaba-NLP/gte
    quant: none
    port: 8417
    served_model_name: gte-embed
    structured_output_mode: none
    est_vram_gib: 1.0
    max_model_len: 8192
    dims: 768
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'arms.yaml'
    path.write_text(text)
    return path


def _slate_arm(**overrides) -> SlateArm:
    data = {
        'arm_id': 'qwen3.5-9b',
        'stack': 'vllm',
        'port': 8410,
        'served_model_name': 'qwen3.5-9b',
        'structured_output_mode': 'json_schema',
        'quant': 'awq',
        'reasoning': 'on',
        'max_model_len': 32768,
    }
    return SlateArm.model_validate(data | overrides)


def _spec_for(arm: SlateArm, **overrides) -> LlmArmSpec:
    kwargs = {
        'base_url': TAP_BASE_URL,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'params': PARAMS,
    }
    return candidate_spec(arm, **(kwargs | overrides))


def test_reads_only_llm_arms_in_manifest_order(tmp_path):
    slate = load_llm_slate(_write(tmp_path, FIXTURE))

    assert [arm.arm_id for arm in slate] == ['qwen3.5-9b', 'moe-stretch']
    qwen, moe = slate
    assert qwen == _slate_arm()
    assert moe == _slate_arm(
        arm_id='moe-stretch',
        stack='llamacpp',
        port=8413,
        served_model_name='moe-stretch',
        structured_output_mode='json_object',
        quant='q4_k_xl',
        reasoning='off',
        max_model_len=16384,
    )


def test_slate_arms_are_frozen(tmp_path):
    arm = load_llm_slate(_write(tmp_path, FIXTURE))[0]

    with pytest.raises(ValidationError):
        arm.port = 9999  # type: ignore[misc]


@pytest.mark.parametrize(
    ('field', 'line'),
    [('max_model_len', '    max_model_len: 32768\n'), ('reasoning', "    reasoning: 'on'\n")],
)
def test_refuses_an_llm_arm_missing_a_required_field(tmp_path, field, line):
    assert FIXTURE.count(line) == 1
    path = _write(tmp_path, FIXTURE.replace(line, ''))

    with pytest.raises(ValueError, match=rf'(?s)qwen3\.5-9b.*{field}'):
        load_llm_slate(path)


def test_refuses_an_unquoted_yaml_reasoning_on_rather_than_coercing_the_bool(tmp_path):
    path = _write(tmp_path, FIXTURE.replace("reasoning: 'on'", 'reasoning: on'))
    assert yaml.safe_load(path.read_text())['arms'][0]['reasoning'] is True

    with pytest.raises(ValueError, match=r'(?s)qwen3\.5-9b.*reasoning'):
        load_llm_slate(path)


@pytest.mark.parametrize(
    ('text', 'fragment'),
    [
        ('arms: [\n', 'not parseable YAML'),
        ('- arm_id: qwen3.5-9b\n', 'is a mapping'),
        ('port_block: [8410, 8417]\n', 'is a list'),
        ('arms:\n  arm_id: qwen3.5-9b\n', 'is a list'),
        ('arms:\n  - axis: embedding\n  - just-a-string\n', 'entries [1] are not mappings'),
    ],
    ids=['malformed-yaml', 'top-level-list', 'no-arms', 'arms-not-a-list', 'non-mapping-entry'],
)
@pytest.mark.parametrize('reader', [load_llm_slate, load_embedding_slate], ids=['llm', 'embedding'])
def test_refuses_a_manifest_of_the_wrong_shape_naming_it(tmp_path, text, fragment, reader):
    path = _write(tmp_path, text)

    with pytest.raises(ValueError) as raised:
        reader(path)

    assert str(path) in str(raised.value)
    assert fragment in str(raised.value)


def test_refuses_duplicate_arm_ids(tmp_path):
    path = _write(tmp_path, FIXTURE.replace('arm_id: moe-stretch', 'arm_id: qwen3.5-9b'))

    with pytest.raises(ValueError, match=r'(?s)duplicate.*qwen3\.5-9b'):
        load_llm_slate(path)


def test_the_committed_manifest_loads_as_a_unique_llm_slate():
    slate = load_llm_slate(COMMITTED_ARMS)

    raw_arms = yaml.safe_load(COMMITTED_ARMS.read_text())['arms']
    llm_ids = [arm['arm_id'] for arm in raw_arms if arm['axis'] == 'llm']
    ids = [arm.arm_id for arm in slate]
    assert ids == llm_ids
    assert len(set(ids)) == len(ids)
    assert ids


def test_arm_endpoint_is_the_loopback_port():
    assert arm_endpoint(_slate_arm(port=8412)) == 'http://127.0.0.1:8412'


def test_candidate_spec_carries_the_arm_and_the_given_pins():
    arm = _slate_arm()

    spec = _spec_for(arm)

    assert isinstance(spec, LlmArmSpec)
    assert spec.axis == 'llm'
    assert spec.arm_id == 'qwen3.5-9b'
    assert spec.model_id == 'qwen3.5-9b'
    assert spec.serving.stack == 'vllm'
    assert spec.serving.quant == 'awq'
    assert spec.serving.unit_name == 'lms-arm@qwen3.5-9b.service'
    assert spec.serving.base_url == TAP_BASE_URL
    assert spec.client_class == 'openai_generic'
    assert spec.structured_output_mode == 'json_schema'
    assert spec.params == PARAMS
    assert spec.pricing is None
    assert spec.arm_role == 'candidate'
    assert spec.code_sha == CODE_SHA
    assert spec.corpus_sha == CORPUS_SHA
    assert spec.preregistration_sha == PREREG_SHA


@pytest.mark.parametrize(
    ('arm_id', 'scratch'),
    [
        ('qwen3.5-9b', 'evalmem_lme_eta_qwen3_5_9b'),
        ('phi-4-14b', 'evalmem_lme_eta_phi_4_14b'),
        ('moe-stretch', 'evalmem_lme_eta_moe_stretch'),
    ],
)
def test_candidate_scratch_group_is_the_sanitised_arm_id(arm_id, scratch):
    spec = _spec_for(_slate_arm(arm_id=arm_id, served_model_name=arm_id))

    assert spec.scratch_group_id == scratch
    assert require_scratch_name(spec.scratch_group_id, checkpoint=GuardCheckpoint.ARM_SPEC)


def test_a_json_object_llamacpp_arm_yields_a_valid_generic_client_spec():
    arm = _slate_arm(
        arm_id='moe-stretch',
        stack='llamacpp',
        served_model_name='moe-stretch',
        structured_output_mode='json_object',
        quant='q4_k_xl',
        reasoning='off',
    )

    spec = _spec_for(arm)

    assert spec.structured_output_mode == 'json_object'
    assert spec.client_class == 'openai_generic'
    assert spec.serving.stack == 'llamacpp'


# --- embedding axis ----------------------------------------------------------------------


def _embedding_spec_for(arm) -> EmbeddingArmSpec:
    return embedding_candidate_spec(
        arm, code_sha=CODE_SHA, corpus_sha=CORPUS_SHA, preregistration_sha=PREREG_SHA
    )


def test_reads_only_embedding_arms_in_manifest_order(tmp_path):
    slate = load_embedding_slate(_write(tmp_path, FIXTURE))

    assert slate == (
        embedding_slate_arm(
            arm_id='bge-embed', served_model_name='bge-embed', query_prefix='Query: '
        ),
        embedding_slate_arm(
            arm_id='gte-embed', port=8417, served_model_name='gte-embed', dims=768,
            query_prefix=None,
        ),
    )
    assert slate[1].query_prefix is None


def test_embedding_slate_arms_are_frozen(tmp_path):
    arm = load_embedding_slate(_write(tmp_path, FIXTURE))[0]

    with pytest.raises(ValidationError):
        arm.dims = 768  # type: ignore[misc]


@pytest.mark.parametrize(
    ('field', 'line'),
    [('dims', '    dims: 1024\n'), ('port', '    port: 8414\n')],
)
def test_refuses_an_embedding_arm_missing_a_required_field(tmp_path, field, line):
    assert FIXTURE.count(line) == 1
    path = _write(tmp_path, FIXTURE.replace(line, ''))

    with pytest.raises(ValueError, match=rf'(?s)bge-embed.*{field}') as raised:
        load_embedding_slate(path)

    assert str(path) in str(raised.value)


def test_refuses_a_quoted_dims_rather_than_coercing_it(tmp_path):
    path = _write(tmp_path, FIXTURE.replace('dims: 1024', 'dims: "1024"'))

    with pytest.raises(ValueError, match=r'(?s)bge-embed.*dims'):
        load_embedding_slate(path)


def test_refuses_duplicate_embedding_arm_ids(tmp_path):
    path = _write(tmp_path, FIXTURE.replace('arm_id: gte-embed', 'arm_id: bge-embed'))

    with pytest.raises(ValueError, match=r'(?s)duplicate.*bge-embed') as raised:
        load_embedding_slate(path)

    assert str(path) in str(raised.value)


def test_the_committed_manifest_loads_as_the_four_iota_embedding_arms():
    slate = load_embedding_slate(COMMITTED_ARMS)

    assert [(arm.arm_id, arm.dims, arm.query_prefix is not None) for arm in slate] == [
        ('qwen3-embedding-0.6b', 1024, True),
        ('granite-embedding-english-r2', 768, False),
        ('qwen3-embedding-4b', 2560, True),
        ('gte-modernbert-base', 768, False),
    ]
    assert slate[0] == embedding_slate_arm()


def test_arm_endpoint_accepts_an_embedding_arm():
    assert arm_endpoint(embedding_slate_arm(port=8416)) == 'http://127.0.0.1:8416'


def test_embedding_candidate_spec_carries_the_arm_and_the_given_pins():
    spec = _embedding_spec_for(embedding_slate_arm())

    assert isinstance(spec, EmbeddingArmSpec)
    assert spec.axis == 'embedding'
    assert spec.arm_id == 'qwen3-embedding-0.6b'
    assert spec.model_id == 'qwen3-embedding-0.6b'
    assert spec.serving == ServingSpec(
        stack='vllm',
        base_url='http://127.0.0.1:8414/v1',
        quant='none',
        unit_name='lms-arm@qwen3-embedding-0.6b.service',
    )
    assert spec.embedding_dim == 1024
    assert spec.query_prefix == QWEN_QUERY_PREFIX
    assert '\n' in spec.query_prefix
    assert spec.arm_role == 'candidate'
    assert spec.code_sha == CODE_SHA
    assert spec.corpus_sha == CORPUS_SHA
    assert spec.preregistration_sha == PREREG_SHA


def test_an_unprefixed_arm_yields_a_spec_without_a_query_prefix():
    arm = embedding_slate_arm(
        arm_id='granite-embedding-english-r2',
        port=8415,
        served_model_name='granite-embedding-english-r2',
        dims=768,
        query_prefix=None,
    )

    spec = _embedding_spec_for(arm)

    assert spec.query_prefix is None
    assert spec.embedding_dim == 768


@pytest.mark.parametrize(
    ('arm_id', 'scratch'),
    [
        ('qwen3-embedding-0.6b', 'evalmem_lme_emb_qwen3_embedding_0_6b'),
        ('granite-embedding-english-r2', 'evalmem_lme_emb_granite_embedding_english_r2'),
        ('qwen3-embedding-4b', 'evalmem_lme_emb_qwen3_embedding_4b'),
        ('gte-modernbert-base', 'evalmem_lme_emb_gte_modernbert_base'),
    ],
)
def test_embedding_scratch_group_is_the_sanitised_arm_id(arm_id, scratch):
    spec = _embedding_spec_for(embedding_slate_arm(arm_id=arm_id, served_model_name=arm_id))

    assert spec.scratch_group_id == scratch
    assert require_scratch_name(spec.scratch_group_id, checkpoint=GuardCheckpoint.ARM_SPEC)


def _control(**overrides) -> EmbeddingArmSpec:
    kwargs = {
        'model_id': 'text-embedding-3-small',
        'embedding_dim': 1536,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'scratch_group_id': 'evalmem_lme_emb_incumbent_embed_a',
    }
    return embedding_control_spec('incumbent-embed-a', **(kwargs | overrides))


def test_embedding_control_spec_is_an_unregistered_openai_control():
    spec = _control()

    assert isinstance(spec, EmbeddingArmSpec)
    assert spec.arm_id == 'incumbent-embed-a'
    assert spec.model_id == 'text-embedding-3-small'
    assert spec.serving == ServingSpec(stack='openai', base_url='https://api.openai.com/v1')
    assert spec.embedding_dim == 1536
    assert spec.query_prefix is None
    assert spec.scratch_group_id == 'evalmem_lme_emb_incumbent_embed_a'
    assert spec.arm_role == 'control'
    assert spec.preregistration_sha is None
    assert spec.code_sha == CODE_SHA
    assert spec.corpus_sha == CORPUS_SHA


def test_embedding_control_spec_refuses_a_live_graph_as_its_scratch_group():
    with pytest.raises(ScratchGuardError):
        _control(scratch_group_id='dark_factory')
