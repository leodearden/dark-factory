"""One arm's FusedMemoryConfig: the base with only that arm's axis block replaced.

The variant is built by dumping the touched block, replacing the arm's fields and
re-validating with that block's own pydantic class. Mutating it in place would
skip ``LLMConfig``'s validator. A fresh ``FusedMemoryConfig(...)`` would re-read
env and YAML.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from fused_memory.arm_harness.arm_spec import (
    METERED_STACK,
    ArmAxis,
    EmbeddingArmSpec,
    LlmArmSpec,
    StructuredOutputMode,
)
from fused_memory.config.schema import EmbedderConfig, FusedMemoryConfig, LLMConfig

LOCAL_ARM_API_KEY = 'local-arm-no-key'
"""Sent to every non-metered endpoint, so the incumbent's OpenAI key never leaves for one."""

CONFIG_STRUCTURED_OUTPUT_MODE: Mapping[StructuredOutputMode, str] = MappingProxyType({
    'json_schema': 'auto',
    'json_object': 'json_object',
})
"""ArmSpec mode -> LLMConfig mode. Config 'auto' is graphiti's response_model-driven json_schema."""


def llm_arm_config(spec: LlmArmSpec, base: FusedMemoryConfig) -> FusedMemoryConfig:
    _require_spec_type(spec, LlmArmSpec, 'llm')
    llm = base.llm.model_dump()
    llm.update(
        provider='openai',
        model=spec.model_id,
        client_class=spec.client_class,
        structured_output_mode=CONFIG_STRUCTURED_OUTPUT_MODE[spec.structured_output_mode],
        temperature=spec.params.temperature,
        max_tokens=spec.params.max_tokens,
        providers=_with_arm_endpoint(llm['providers'], spec),
    )
    return base.model_copy(update={'llm': LLMConfig.model_validate(llm)}, deep=True)


def embedding_arm_config(spec: EmbeddingArmSpec, base: FusedMemoryConfig) -> FusedMemoryConfig:
    _require_spec_type(spec, EmbeddingArmSpec, 'embedding')
    embedder = base.embedder.model_dump()
    embedder.update(
        model=spec.model_id,
        dimensions=spec.embedding_dim,
        providers=_with_arm_endpoint(embedder['providers'], spec),
    )
    return base.model_copy(update={'embedder': EmbedderConfig.model_validate(embedder)}, deep=True)


def _with_arm_endpoint(
    providers: dict[str, Any], spec: LlmArmSpec | EmbeddingArmSpec
) -> dict[str, Any]:
    if spec.serving.stack == METERED_STACK:
        openai = {**(providers.get('openai') or {}), 'api_url': spec.serving.base_url}
    else:
        openai = {'api_key': LOCAL_ARM_API_KEY, 'api_url': spec.serving.base_url}
    return {**providers, 'openai': openai}


def _require_spec_type(spec: object, expected: type, axis: ArmAxis) -> None:
    if not isinstance(spec, expected):
        actual = getattr(spec, 'axis', type(spec).__name__)
        raise TypeError(f'{axis}_arm_config needs an {axis} arm spec, got axis {actual!r}')
