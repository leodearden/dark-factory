"""ArmSpec: the validated description of one arm, rejected loudly at construction."""

import json

import pytest
from pydantic import ValidationError

from fused_memory.arm_harness.arm_spec import (
    EmbeddingArmSpec,
    LlmArmSpec,
    LlmParams,
    ServingSpec,
    TokenPricing,
    load_arm_spec,
    parse_arm_spec,
)
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError

CODE_SHA = 'a' * 40
CORPUS_SHA = 'b' * 64
PREREG_SHA = 'c' * 40


def control_llm_data(**overrides) -> dict:
    data = {
        'arm_id': 'incumbent-ctrl-a',
        'axis': 'llm',
        'model_id': 'gpt-4.1-mini',
        'serving': {'stack': 'openai', 'base_url': 'https://api.openai.com/v1'},
        'client_class': 'openai_generic',
        'structured_output_mode': 'json_schema',
        'params': {'temperature': 0.0, 'max_tokens': 4096},
        'pricing': {'usd_per_mtok_input': 0.4, 'usd_per_mtok_output': 1.6},
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': None,
        'scratch_group_id': 'evalmem_ctrl_a',
        'arm_role': 'control',
    }
    return data | overrides


def moe_candidate_data(**overrides) -> dict:
    data = control_llm_data(
        arm_id='qwen3-moe-q4',
        model_id='qwen3-30b-a3b',
        serving={
            'stack': 'llamacpp',
            'base_url': 'http://127.0.0.1:8081/v1',
            'quant': 'Q4_K_M',
            'unit_name': 'lms-arm@qwen3-moe',
        },
        structured_output_mode='json_object',
        pricing=None,
        arm_role='candidate',
        preregistration_sha=PREREG_SHA,
        scratch_group_id='evalmem_qwen3_moe',
    )
    return data | overrides


def embedding_data(**overrides) -> dict:
    data = {
        'arm_id': 'bge-m3',
        'axis': 'embedding',
        'model_id': 'BAAI/bge-m3',
        'serving': {'stack': 'tei', 'base_url': 'http://127.0.0.1:8090'},
        'embedding_dim': 1024,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'scratch_group_id': 'evalmem_bge_m3',
        'arm_role': 'candidate',
    }
    return data | overrides


QWEN_QUERY_PREFIX = 'Instruct: Given a search query, retrieve relevant memory records\nQuery: '


def prefixed_embedding_data(**overrides) -> dict:
    return embedding_data(query_prefix=QWEN_QUERY_PREFIX) | overrides


def test_incumbent_control_arm_is_valid():
    spec = parse_arm_spec(control_llm_data())

    assert isinstance(spec, LlmArmSpec)
    assert spec.serving == ServingSpec(stack='openai', base_url='https://api.openai.com/v1')
    assert spec.params == LlmParams(temperature=0.0, max_tokens=4096)
    assert spec.pricing == TokenPricing(usd_per_mtok_input=0.4, usd_per_mtok_output=1.6)
    assert spec.preregistration_sha is None


def test_llamacpp_moe_candidate_is_valid():
    spec = parse_arm_spec(moe_candidate_data())

    assert isinstance(spec, LlmArmSpec)
    assert spec.structured_output_mode == 'json_object'
    assert spec.serving.quant == 'Q4_K_M'
    assert spec.pricing is None


def test_embedding_candidate_is_valid():
    spec = parse_arm_spec(embedding_data())

    assert isinstance(spec, EmbeddingArmSpec)
    assert spec.embedding_dim == 1024


@pytest.mark.parametrize(
    ('model', 'factory'),
    [(LlmArmSpec, control_llm_data), (EmbeddingArmSpec, embedding_data)],
)
def test_protected_scratch_name_raises_the_typed_guard_error_at_construction(model, factory):
    data = factory(scratch_group_id='dark_factory')

    with pytest.raises(ScratchGuardError) as direct:
        model(**data)
    with pytest.raises(ScratchGuardError) as parsed:
        parse_arm_spec(data)

    assert direct.value.checkpoint is GuardCheckpoint.ARM_SPEC
    assert parsed.value.checkpoint is GuardCheckpoint.ARM_SPEC
    assert parsed.value.name == 'dark_factory'


def _rejection(data: dict) -> str:
    with pytest.raises(ValidationError) as caught:
        parse_arm_spec(data)
    return str(caught.value)


def test_candidate_without_preregistration_sha_is_rejected():
    message = _rejection(moe_candidate_data(preregistration_sha=None))
    assert 'preregistration_sha' in message
    assert 'candidate' in message
    assert 'qwen3-moe-q4' in message


def test_control_with_preregistration_sha_is_rejected():
    message = _rejection(control_llm_data(preregistration_sha=PREREG_SHA))
    assert 'preregistration_sha' in message
    assert 'control' in message
    assert PREREG_SHA in message


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('code_sha', 'A' * 40),
        ('code_sha', 'a' * 39),
        ('preregistration_sha', 'xyz'),
        ('corpus_sha', 'b' * 40),
        ('corpus_sha', 'B' * 64),
    ],
)
def test_malformed_shas_are_rejected(field, value):
    data = moe_candidate_data(**{field: value})
    message = _rejection(data)
    assert field in message
    assert value in message


@pytest.mark.parametrize('arm_id', ['Upper', '-leading', '', 'has space', 'a/b'])
def test_arm_id_must_be_a_safe_path_segment(arm_id):
    message = _rejection(control_llm_data(arm_id=arm_id))
    assert 'arm_id' in message


def test_json_object_mode_requires_the_generic_client():
    message = _rejection(moe_candidate_data(client_class='openai'))
    assert 'json_object' in message
    assert 'openai_generic' in message


def test_tei_stack_is_not_an_llm_stack():
    data = moe_candidate_data(serving={'stack': 'tei', 'base_url': 'http://127.0.0.1:8090'})
    message = _rejection(data)
    assert 'tei' in message


def test_llamacpp_stack_is_not_an_embedding_stack():
    data = embedding_data(serving={'stack': 'llamacpp', 'base_url': 'http://127.0.0.1:8081'})
    message = _rejection(data)
    assert 'llamacpp' in message


def test_pricing_on_a_local_stack_is_rejected():
    pricing = {'usd_per_mtok_input': 1.0, 'usd_per_mtok_output': 1.0}
    message = _rejection(moe_candidate_data(pricing=pricing))
    assert 'pricing' in message
    assert 'llamacpp' in message


def test_missing_pricing_on_the_paid_stack_is_rejected():
    message = _rejection(control_llm_data(pricing=None))
    assert 'pricing' in message
    assert 'openai' in message


@pytest.mark.parametrize(
    'data',
    [
        embedding_data(embedding_dim=0),
        control_llm_data(params={'temperature': 0.0, 'max_tokens': 0}),
        control_llm_data(params={'temperature': -0.1, 'max_tokens': 10}),
    ],
)
def test_non_positive_sizes_and_negative_temperature_are_rejected(data):
    _rejection(data)


def test_base_url_must_be_http():
    data = control_llm_data(serving={'stack': 'openai', 'base_url': 'ftp://example.com'})
    message = _rejection(data)
    assert 'base_url' in message


def test_unknown_extra_key_is_rejected():
    message = _rejection(control_llm_data(surprise=True))
    assert 'surprise' in message


def test_preregistration_sha_is_required_even_though_nullable():
    data = control_llm_data()
    del data['preregistration_sha']
    message = _rejection(data)
    assert 'preregistration_sha' in message


def test_specs_are_frozen():
    spec = parse_arm_spec(control_llm_data())
    with pytest.raises(ValidationError):
        spec.model_id = 'other'  # type: ignore[misc]


GPT_4O_MINI_PRICING = TokenPricing(usd_per_mtok_input=0.15, usd_per_mtok_output=0.60)


def test_usd_for_one_million_tokens_each_way_is_the_sum_of_the_two_rates():
    assert GPT_4O_MINI_PRICING.usd_for(1_000_000, 1_000_000) == 0.75


def test_usd_for_prices_input_and_output_at_their_per_million_rates():
    assert GPT_4O_MINI_PRICING.usd_for(1650, 95) == (1650 * 0.15 + 95 * 0.60) / 1_000_000


@pytest.mark.parametrize(
    ('input_tokens', 'output_tokens', 'offending'),
    [(-1, 10, 'input_tokens'), (10, -7, 'output_tokens')],
)
def test_usd_for_refuses_a_negative_token_count(input_tokens, output_tokens, offending):
    with pytest.raises(ValueError, match=offending) as caught:
        GPT_4O_MINI_PRICING.usd_for(input_tokens, output_tokens)

    assert str(min(input_tokens, output_tokens)) in str(caught.value)


@pytest.mark.parametrize(
    'factory', [control_llm_data, moe_candidate_data, embedding_data, prefixed_embedding_data]
)
def test_load_arm_spec_round_trips_a_dumped_spec(tmp_path, factory):
    spec = parse_arm_spec(factory())
    path = tmp_path / 'arm.json'
    path.write_text(json.dumps(spec.model_dump(mode='json')))

    assert load_arm_spec(path) == spec


def test_an_embedding_arm_without_a_query_prefix_carries_none():
    spec = parse_arm_spec(embedding_data())

    assert isinstance(spec, EmbeddingArmSpec)
    assert spec.query_prefix is None


def test_an_embedding_arm_carries_its_query_prefix_verbatim():
    spec = parse_arm_spec(prefixed_embedding_data())

    assert isinstance(spec, EmbeddingArmSpec)
    assert spec.query_prefix == QWEN_QUERY_PREFIX


def test_an_empty_query_prefix_is_rejected_naming_the_arm():
    with pytest.raises(ValueError) as caught:
        parse_arm_spec(embedding_data(query_prefix=''))

    assert 'query_prefix' in str(caught.value)
    assert 'bge-m3' in str(caught.value)


def test_an_llm_arm_has_no_query_prefix():
    message = _rejection(control_llm_data(query_prefix=QWEN_QUERY_PREFIX))

    assert 'query_prefix' in message
