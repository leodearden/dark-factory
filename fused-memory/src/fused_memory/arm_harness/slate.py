"""The candidate slate scripts/local-model-serving/arms.yaml declares on each axis, and the ArmSpec each arm runs under."""

import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from fused_memory.arm_harness.arm_spec import (
    METERED_STACK,
    ArmAxis,
    ArmId,
    EmbeddingArmSpec,
    GitSha,
    LlmArmSpec,
    LlmParams,
    ServingSpec,
    StructuredOutputMode,
)
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name

LLM_AXIS: ArmAxis = 'llm'
EMBEDDING_AXIS: ArmAxis = 'embedding'
LOOPBACK_HOST = '127.0.0.1'
OPENAI_BASE_URL = 'https://api.openai.com/v1'
SCREENING_SCRATCH_PREFIX = 'evalmem_lme_eta_'
EMBEDDING_SCRATCH_PREFIX = 'evalmem_lme_emb_'
_NON_SCRATCH_CHAR = re.compile(r'[^a-z0-9]')

LocalLlmStack = Literal['vllm', 'llamacpp']
LocalEmbeddingStack = Literal['vllm', 'tei']
ReasoningMode = Literal['on', 'off']


class _SlateEntry(BaseModel):
    """The fields every arms.yaml arm shares that the harness reads; lms_manifest owns the rest."""

    model_config = ConfigDict(frozen=True, extra='ignore', strict=True)

    arm_id: ArmId
    port: int
    served_model_name: str
    quant: str


class SlateArm(_SlateEntry):
    """One arms.yaml LLM arm, as screening reads it."""

    stack: LocalLlmStack
    structured_output_mode: StructuredOutputMode
    reasoning: ReasoningMode
    max_model_len: int


class EmbeddingSlateArm(_SlateEntry):
    """One arms.yaml embedding arm, as the embedding runs read it."""

    stack: LocalEmbeddingStack
    dims: int
    query_prefix: str | None = None


_ArmT = TypeVar('_ArmT', SlateArm, EmbeddingSlateArm)


def load_llm_slate(path: Path | str) -> tuple[SlateArm, ...]:
    return _load_axis(Path(path), LLM_AXIS, SlateArm)


def load_embedding_slate(path: Path | str) -> tuple[EmbeddingSlateArm, ...]:
    return _load_axis(Path(path), EMBEDDING_AXIS, EmbeddingSlateArm)


def _load_axis(manifest_path: Path, axis: ArmAxis, model: type[_ArmT]) -> tuple[_ArmT, ...]:
    slate = tuple(
        _slate_arm(manifest_path, axis, model, entry)
        for entry in _arm_entries(manifest_path)
        if entry.get('axis') == axis
    )
    _require_unique_arm_ids(manifest_path, axis, slate)
    return slate


def _arm_entries(manifest_path: Path) -> list[dict[str, object]]:
    try:
        document = yaml.safe_load(manifest_path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f'{manifest_path}: not parseable YAML: {error}') from error
    entries = document.get('arms') if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f'{manifest_path}: an arms manifest is a mapping whose `arms` is a list')
    misshapen = [index for index, entry in enumerate(entries) if not isinstance(entry, dict)]
    if misshapen:
        raise ValueError(f'{manifest_path}: `arms` entries {misshapen} are not mappings')
    return entries


def _slate_arm(
    manifest_path: Path, axis: ArmAxis, model: type[_ArmT], entry: Mapping[str, object]
) -> _ArmT:
    try:
        return model.model_validate(dict(entry))
    except ValidationError as error:
        raise ValueError(
            f'{manifest_path}: {axis} arm {entry.get("arm_id")!r} is not a readable slate arm: '
            f'{error}'
        ) from error


def _require_unique_arm_ids(
    manifest_path: Path, axis: ArmAxis, slate: tuple[_SlateEntry, ...]
) -> None:
    duplicates = sorted(
        arm_id for arm_id, count in Counter(arm.arm_id for arm in slate).items() if count > 1
    )
    if duplicates:
        raise ValueError(f'{manifest_path}: duplicate {axis} arm ids {duplicates}')


def arm_endpoint(arm: SlateArm | EmbeddingSlateArm) -> str:
    return f'http://{LOOPBACK_HOST}:{arm.port}'


def _unit_name(arm: SlateArm | EmbeddingSlateArm) -> str:
    return f'lms-arm@{arm.arm_id}.service'


def _scratch_group(prefix: str, arm: SlateArm | EmbeddingSlateArm) -> str:
    return require_scratch_name(
        prefix + _NON_SCRATCH_CHAR.sub('_', arm.arm_id), checkpoint=GuardCheckpoint.ARM_SPEC
    )


def screening_scratch_group(arm: SlateArm) -> str:
    return _scratch_group(SCREENING_SCRATCH_PREFIX, arm)


def candidate_spec(
    arm: SlateArm,
    *,
    base_url: str,
    code_sha: GitSha,
    corpus_sha: str,
    preregistration_sha: GitSha,
    params: LlmParams,
) -> LlmArmSpec:
    return LlmArmSpec(
        arm_id=arm.arm_id,
        axis='llm',
        model_id=arm.served_model_name,
        serving=ServingSpec(
            stack=arm.stack,
            base_url=base_url,
            quant=arm.quant,
            unit_name=_unit_name(arm),
        ),
        client_class='openai_generic',
        structured_output_mode=arm.structured_output_mode,
        params=params,
        pricing=None,
        code_sha=code_sha,
        corpus_sha=corpus_sha,
        preregistration_sha=preregistration_sha,
        scratch_group_id=screening_scratch_group(arm),
        arm_role='candidate',
    )


def embedding_candidate_spec(
    arm: EmbeddingSlateArm,
    *,
    code_sha: GitSha,
    corpus_sha: str,
    preregistration_sha: GitSha,
) -> EmbeddingArmSpec:
    return EmbeddingArmSpec(
        arm_id=arm.arm_id,
        axis='embedding',
        model_id=arm.served_model_name,
        serving=ServingSpec(
            stack=arm.stack,
            base_url=f'{arm_endpoint(arm)}/v1',
            quant=arm.quant,
            unit_name=_unit_name(arm),
        ),
        embedding_dim=arm.dims,
        query_prefix=arm.query_prefix,
        code_sha=code_sha,
        corpus_sha=corpus_sha,
        preregistration_sha=preregistration_sha,
        scratch_group_id=_scratch_group(EMBEDDING_SCRATCH_PREFIX, arm),
        arm_role='candidate',
    )


def embedding_control_spec(
    arm_id: str,
    *,
    model_id: str,
    embedding_dim: int,
    code_sha: GitSha,
    corpus_sha: str,
    scratch_group_id: str,
) -> EmbeddingArmSpec:
    return EmbeddingArmSpec(
        arm_id=arm_id,
        axis='embedding',
        model_id=model_id,
        serving=ServingSpec(stack=METERED_STACK, base_url=OPENAI_BASE_URL),
        embedding_dim=embedding_dim,
        code_sha=code_sha,
        corpus_sha=corpus_sha,
        preregistration_sha=None,
        scratch_group_id=scratch_group_id,
        arm_role='control',
    )
