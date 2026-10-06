"""RunManifest: the record an arm run leaves behind as its ``run.json``.

It holds what the run measured and under which settings, so cross-arm checks can
compare runs from their manifests alone. The text form follows the null
convention stated in ``shared/src/shared/memory_eval_metrics.py``'s module
docstring, applied at every nested model, and renders through ``canonical_json_text``.
"""

from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator
from shared.memory_eval_metrics import canonical_json_text

from fused_memory.arm_harness.arm_spec import ArmSpec
from fused_memory.arm_harness.frozen_model import FrozenModel
from fused_memory.arm_harness.instrument_checks import CheckResult
from fused_memory.arm_harness.metrics_record import IndexConfiguration, UtcDatetime
from fused_memory.arm_harness.replay_types import ArmAbort

RUN_MANIFEST_SCHEMA_VERSION = 1


class SettingsSummary(FrozenModel):
    concurrency: int = Field(ge=1)
    index_configuration: IndexConfiguration
    episode_timeout_s: float = Field(gt=0)


class EffectiveEmbedder(FrozenModel):
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)


class RunManifest(FrozenModel):
    schema_version: Literal[1]
    spec: ArmSpec
    settings_summary: SettingsSummary
    effective_embedder: EffectiveEmbedder
    graphiti_max_coroutines: int = Field(ge=1)
    graphiti_semaphore_limit: int = Field(ge=1)
    episode_ids: tuple[str, ...]
    incomplete: bool
    abort: ArmAbort | None
    check_results: tuple[CheckResult, ...]
    started_at: UtcDatetime
    finished_at: UtcDatetime

    @model_validator(mode='after')
    def _manifest_rules(self) -> Self:
        if self.abort is not None and not self.incomplete:
            raise ValueError(f'arm {self.spec.arm_id!r} aborted, so the run must be incomplete')
        if self.finished_at < self.started_at:
            raise ValueError(
                f'finished_at {self.finished_at} precedes started_at {self.started_at}'
            )
        repeated = sorted(i for i, count in Counter(self.episode_ids).items() if count > 1)
        if repeated:
            raise ValueError(f'episode_ids repeat {repeated}')
        return self


def _null_convention_payload(model: BaseModel) -> dict[str, object]:
    """``model`` as JSON data, omitting None-valued optional fields at every nested model."""
    dumped = model.model_dump(mode='json')
    payload: dict[str, object] = {}
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        if value is None and not field.is_required():
            continue
        nested = isinstance(value, BaseModel)
        payload[name] = _null_convention_payload(value) if nested else dumped[name]
    return payload


def serialize_run_manifest(manifest: RunManifest) -> str:
    return canonical_json_text(_null_convention_payload(manifest))


def load_run_manifest(path: Path | str) -> RunManifest:
    return RunManifest.model_validate_json(Path(path).read_text())
