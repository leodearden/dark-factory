"""The LLM candidate slate as scripts/local-model-serving/arms.yaml declares it, and the candidate ArmSpec each arm is screened under."""

import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from fused_memory.arm_harness.arm_spec import (
    ArmId,
    GitSha,
    LlmArmSpec,
    LlmParams,
    ServingSpec,
    StructuredOutputMode,
)
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name

LLM_AXIS = 'llm'
LOOPBACK_HOST = '127.0.0.1'
SCREENING_SCRATCH_PREFIX = 'evalmem_lme_eta_'
_NON_SCRATCH_CHAR = re.compile(r'[^a-z0-9]')

LocalLlmStack = Literal['vllm', 'llamacpp']
ReasoningMode = Literal['on', 'off']


class SlateArm(BaseModel):
    """The fields of one arms.yaml LLM arm that screening reads; lms_manifest owns the rest."""

    model_config = ConfigDict(frozen=True, extra='ignore', strict=True)

    arm_id: ArmId
    stack: LocalLlmStack
    port: int
    served_model_name: str
    structured_output_mode: StructuredOutputMode
    quant: str
    reasoning: ReasoningMode
    max_model_len: int


def load_llm_slate(path: Path | str) -> tuple[SlateArm, ...]:
    manifest_path = Path(path)
    entries = _arm_entries(manifest_path)
    slate = tuple(
        _slate_arm(manifest_path, entry) for entry in entries if entry.get('axis') == LLM_AXIS
    )
    _require_unique_arm_ids(manifest_path, slate)
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


def _slate_arm(manifest_path: Path, entry: Mapping[str, object]) -> SlateArm:
    try:
        return SlateArm.model_validate(dict(entry))
    except ValidationError as error:
        raise ValueError(
            f'{manifest_path}: LLM arm {entry.get("arm_id")!r} is not a readable slate arm: {error}'
        ) from error


def _require_unique_arm_ids(manifest_path: Path, slate: tuple[SlateArm, ...]) -> None:
    duplicates = sorted(
        arm_id for arm_id, count in Counter(arm.arm_id for arm in slate).items() if count > 1
    )
    if duplicates:
        raise ValueError(f'{manifest_path}: duplicate LLM arm ids {duplicates}')


def arm_endpoint(arm: SlateArm) -> str:
    return f'http://{LOOPBACK_HOST}:{arm.port}'


def screening_scratch_group(arm: SlateArm) -> str:
    return require_scratch_name(
        SCREENING_SCRATCH_PREFIX + _NON_SCRATCH_CHAR.sub('_', arm.arm_id),
        checkpoint=GuardCheckpoint.ARM_SPEC,
    )


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
            unit_name=f'lms-arm@{arm.arm_id}.service',
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
