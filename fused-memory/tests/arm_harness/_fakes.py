"""Shared arm-harness test doubles: valid spec builders and public-Protocol fakes."""

from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec

CODE_SHA = 'a' * 40
CORPUS_SHA = 'b' * 64
PREREG_SHA = 'c' * 40
UNREACHABLE_BASE_URL = 'http://127.0.0.1:9/v1'


def llm_spec(*, base_url: str = UNREACHABLE_BASE_URL, **overrides) -> LlmArmSpec:
    """A local-stack candidate LLM arm; override any field by keyword."""
    data = {
        'arm_id': 'qwen3-8b-vllm',
        'axis': 'llm',
        'model_id': 'qwen3-8b',
        'serving': {'stack': 'vllm', 'base_url': base_url},
        'client_class': 'openai_generic',
        'structured_output_mode': 'json_schema',
        'params': {'temperature': 0.0, 'max_tokens': 4096},
        'pricing': None,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'scratch_group_id': 'evalmem_qwen3_8b',
        'arm_role': 'candidate',
    }
    return LlmArmSpec.model_validate(data | overrides)


def incumbent_control_spec(**overrides) -> LlmArmSpec:
    """The metered incumbent control arm (stack 'openai', priced, no prereg sha)."""
    data = {
        'arm_id': 'incumbent-ctrl-a',
        'model_id': 'gpt-4.1-mini',
        'serving': {'stack': 'openai', 'base_url': 'https://api.openai.com/v1'},
        'pricing': {'usd_per_mtok_input': 0.4, 'usd_per_mtok_output': 1.6},
        'preregistration_sha': None,
        'scratch_group_id': 'evalmem_ctrl_a',
        'arm_role': 'control',
    }
    return llm_spec(**(data | overrides))


def embedding_spec(*, base_url: str = UNREACHABLE_BASE_URL, **overrides) -> EmbeddingArmSpec:
    """A local-stack candidate embedding arm; override any field by keyword."""
    data = {
        'arm_id': 'bge-m3',
        'axis': 'embedding',
        'model_id': 'BAAI/bge-m3',
        'serving': {'stack': 'tei', 'base_url': base_url},
        'embedding_dim': 1024,
        'code_sha': CODE_SHA,
        'corpus_sha': CORPUS_SHA,
        'preregistration_sha': PREREG_SHA,
        'scratch_group_id': 'evalmem_bge_m3',
        'arm_role': 'candidate',
    }
    return EmbeddingArmSpec.model_validate(data | overrides)
