"""ArmSpec: the validated description of one eval arm (PRD §Contract).

A discriminated union on ``axis``. LLM arms carry the client and structured-output
choices; embedding arms carry the vector dimension. Everything else is shared.
Invalid specs are rejected at construction. A non-scratch ``scratch_group_id``
raises the typed ``ScratchGuardError`` itself, not a ``ValidationError``.
"""

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name

GIT_SHA_PATTERN = re.compile(r'[0-9a-f]{40}')
CONTENT_SHA_PATTERN = re.compile(r'[0-9a-f]{64}')
ARM_ID_PATTERN = re.compile(r'[a-z0-9][a-z0-9._-]*')

ArmAxis = Literal['llm', 'embedding']
ArmRole = Literal['control', 'candidate']
ServingStack = Literal['vllm', 'llamacpp', 'tei', 'openai']
ClientClass = Literal['openai', 'openai_generic']
StructuredOutputMode = Literal['json_schema', 'json_object']

LLM_STACKS: frozenset[ServingStack] = frozenset({'vllm', 'llamacpp', 'openai'})
EMBEDDING_STACKS: frozenset[ServingStack] = frozenset({'vllm', 'tei', 'openai'})
METERED_STACK: ServingStack = 'openai'


def _fullmatching(pattern: re.Pattern[str], what: str) -> Callable[[str], str]:
    def check(value: str) -> str:
        if pattern.fullmatch(value) is None:
            raise ValueError(f'{what} {value!r} does not match ^{pattern.pattern}$')
        return value

    return check


def _http_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise ValueError(f'base_url {value!r} must be an absolute http(s) URL')
    return value


GitSha = Annotated[str, AfterValidator(_fullmatching(GIT_SHA_PATTERN, 'git sha'))]
ContentSha = Annotated[str, AfterValidator(_fullmatching(CONTENT_SHA_PATTERN, 'content sha'))]
ArmId = Annotated[str, AfterValidator(_fullmatching(ARM_ID_PATTERN, 'arm_id'))]


def require_preregistration_matches_role(
    arm_id: str, arm_role: ArmRole, preregistration_sha: str | None
) -> None:
    """Candidates need a preregistration sha; controls predate the prereg doc, so carry none."""
    if arm_role == 'candidate' and preregistration_sha is None:
        raise ValueError(
            f'candidate arm {arm_id!r} has preregistration_sha None: candidate-arm '
            'artifacts missing a matching preregistration_sha are invalid by schema'
        )
    if arm_role == 'control' and preregistration_sha is not None:
        raise ValueError(
            f'control arm {arm_id!r} carries preregistration_sha {preregistration_sha!r}: '
            'controls predate the preregistration doc, so their sha must be None'
        )


class _Frozen(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class ServingSpec(_Frozen):
    stack: ServingStack
    base_url: Annotated[str, AfterValidator(_http_url)]
    quant: str | None = None
    unit_name: str | None = None


class LlmParams(_Frozen):
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)


class TokenPricing(_Frozen):
    usd_per_mtok_input: float = Field(ge=0)
    usd_per_mtok_output: float = Field(ge=0)


class _ArmSpecBase(_Frozen):
    arm_id: ArmId
    model_id: str = Field(min_length=1)
    serving: ServingSpec
    code_sha: GitSha
    corpus_sha: ContentSha
    preregistration_sha: GitSha | None
    scratch_group_id: str
    arm_role: ArmRole

    @field_validator('scratch_group_id', mode='before')
    @classmethod
    def _scratch_only(cls, value: object) -> str:
        return require_scratch_name(value, checkpoint=GuardCheckpoint.ARM_SPEC)

    @model_validator(mode='after')
    def _preregistration_matches_role(self) -> Self:
        require_preregistration_matches_role(
            self.arm_id, self.arm_role, self.preregistration_sha
        )
        return self

    def _require_stack_in(self, allowed: frozenset[ServingStack], axis: ArmAxis) -> None:
        if self.serving.stack not in allowed:
            raise ValueError(
                f'arm {self.arm_id!r}: stack {self.serving.stack!r} is not an {axis} stack '
                f'(allowed: {", ".join(sorted(allowed))})'
            )


class LlmArmSpec(_ArmSpecBase):
    axis: Literal['llm']
    client_class: ClientClass
    structured_output_mode: StructuredOutputMode
    params: LlmParams
    pricing: TokenPricing | None

    @model_validator(mode='after')
    def _llm_axis_rules(self) -> Self:
        self._require_stack_in(LLM_STACKS, 'llm')
        self._require_json_object_on_generic_client()
        self._require_pricing_iff_metered()
        return self

    def _require_json_object_on_generic_client(self) -> None:
        if self.structured_output_mode == 'json_object' and self.client_class != 'openai_generic':
            raise ValueError(
                f'arm {self.arm_id!r}: structured_output_mode json_object requires '
                f"client_class 'openai_generic', got {self.client_class!r}"
            )

    def _require_pricing_iff_metered(self) -> None:
        metered = self.serving.stack == METERED_STACK
        if metered and self.pricing is None:
            raise ValueError(
                f'arm {self.arm_id!r}: pricing is required on the metered '
                f'{METERED_STACK!r} stack'
            )
        if not metered and self.pricing is not None:
            raise ValueError(
                f'arm {self.arm_id!r}: pricing must be absent on local stack '
                f'{self.serving.stack!r}; local arms cost 0 by definition'
            )


class EmbeddingArmSpec(_ArmSpecBase):
    axis: Literal['embedding']
    embedding_dim: int = Field(gt=0)

    @model_validator(mode='after')
    def _embedding_axis_rules(self) -> Self:
        self._require_stack_in(EMBEDDING_STACKS, 'embedding')
        return self


ArmSpec = Annotated[LlmArmSpec | EmbeddingArmSpec, Field(discriminator='axis')]

_ARM_SPEC_ADAPTER: TypeAdapter[LlmArmSpec | EmbeddingArmSpec] = TypeAdapter(ArmSpec)


def parse_arm_spec(data: Mapping[str, object]) -> LlmArmSpec | EmbeddingArmSpec:
    return _ARM_SPEC_ADAPTER.validate_python(dict(data))


def load_arm_spec(path: Path | str) -> LlmArmSpec | EmbeddingArmSpec:
    return parse_arm_spec(json.loads(Path(path).read_text()))
