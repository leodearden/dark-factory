"""Load and validate `arms.yaml` — the serving-substrate contract surface.

PRD-MARKER:local-memory-models-eval serving

Task 3713 (LME-alpha) of `plans/local-memory-models-eval-prd.md`.

One manifest is the whole contract.  The unit template, the launch argv
builders, the weight fetcher, the health check, and every downstream consumer
(eta screening, theta full runs, iota embedding arms) read `base_url`,
`served_model_name`, `dims` and `structured_output_mode` from HERE and nowhere
else.  That makes it structurally impossible for a tool to serve one thing and
a probe to measure another.

Validation is loud and typed on purpose.  The failure mode this guards against
is not a crash — it is a manifest that half-loads, drops the arm it could not
parse, and lets the health report show every REMAINING arm green.  That reads
as "the slate is healthy" when the slate is actually narrower than the PRD
commissioned, and nothing downstream would ever notice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ARM_AXES = ('llm', 'embedding')
ARM_STACKS = ('vllm', 'llamacpp', 'tei')
STRUCTURED_OUTPUT_MODES = ('json_schema', 'json_object', 'none')

#: Fields whose value still awaits a live-measurement decision carry this
#: prefix (PRD Open Q1 embedding stack, Open Q3 MoE model+quant).  Tools refuse
#: to act on a placeholder arm rather than launching a container for a
#: literal "TBD-Q3" model id and reporting the resulting 404 as an arm failure.
PLACEHOLDER_PREFIX = 'TBD'

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / 'arms.yaml'


class ArmManifestError(Exception):
    """A manifest is missing, unreadable, or self-contradictory.

    Typed rather than a bare ValueError so callers can distinguish "your
    manifest is wrong" from "the GPU probe failed" from "the endpoint is down"
    without string-matching, and so no `except Exception` on a transport path
    can absorb a configuration error.
    """


class ArmEntry(BaseModel):
    """One servable endpoint.

    ``frozen`` because several tools hold the same entry concurrently while
    building argv and rendering a report; a mutation in one would silently
    change what the other reports it measured.
    """

    # `protected_namespaces=()` because `model_ref` legitimately starts with
    # pydantic's reserved `model_` prefix and this is a manifest OF models.
    # `extra='forbid'` because a typo'd key (`quantisation:` for `quant:`)
    # silently ignored reads exactly like "the default applied".
    model_config = ConfigDict(extra='forbid', frozen=True, protected_namespaces=())

    arm_id: str
    axis: Literal['llm', 'embedding']
    stack: Literal['vllm', 'llamacpp', 'tei']
    image: str
    model_ref: str
    port: int
    served_model_name: str
    structured_output_mode: Literal['json_schema', 'json_object', 'none']
    est_vram_gib: float
    quant: str = 'none'
    max_model_len: int = 8192
    max_num_seqs: int = 5

    #: Embedding arms only, and REQUIRED there: iota compares arms at their
    #: native/MRL dims (PRD D7), and the health check verifies the server's
    #: actual vector length against this number.
    dims: int | None = None
    #: The Qwen3-Embedding family requires a query-side instruct prefix
    #: (PRD line 134).  A silently-dropped prefix degrades every retrieval
    #: number iota reports, so it lives in the contract, not in a call site.
    query_prefix: str | None = None
    #: llamacpp arms: the ONE pinned quant file inside the GGUF repo.  Fetching
    #: without it pulls every quant in the repo (~200 GB instead of ~10 GB).
    gguf_file: str | None = None
    #: Resolved in step 19 so a re-run serves the same bits, not merely the
    #: same tag.
    image_digest: str | None = None
    #: PRD Open Q1: TEI is the per-arm fallback if the vLLM pooling runner will
    #: not load an embedding model.
    fallback_stack: str | None = None
    notes: str | None = None

    @model_validator(mode='after')
    def _check_arm_invariants(self) -> ArmEntry:
        if self.axis == 'embedding' and self.dims is None:
            raise ValueError(
                f'embedding arm {self.arm_id!r} declares no dims; iota compares '
                'arms at their native dims and the health check verifies the '
                "server's actual vector length against this field"
            )
        if self.stack == 'llamacpp' and self.structured_output_mode == 'json_schema':
            raise ValueError(
                f'arm {self.arm_id!r} is stack=llamacpp but claims '
                'structured_output_mode=json_schema. llama.cpp silently falls '
                'back to UNCONSTRAINED output on $ref/$defs schemas '
                '(ggml-org/llama.cpp#21228), so that claim is false by '
                'construction; use json_object plus a client-side validator'
            )
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f'arm {self.arm_id!r} declares an invalid port {self.port}')
        if self.est_vram_gib <= 0:
            raise ValueError(
                f'arm {self.arm_id!r} declares est_vram_gib={self.est_vram_gib}; a '
                'non-positive footprint would let the VRAM pre-flight pass any arm'
            )
        return self

    @property
    def base_url(self) -> str:
        """Always 127.0.0.1, never `localhost`.

        `localhost` can resolve to ::1 while the server listens on IPv4 only,
        which presents as a connection refusal that looks like a dead arm
        (documented at scripts/run_vllm_eval.py:505-512).
        """
        return f'http://127.0.0.1:{self.port}'

    @property
    def container_name(self) -> str:
        """Must match `lms-arm@.service`'s `ExecStop=docker stop lms-arm-%i`."""
        return f'lms-arm-{self.arm_id}'

    @property
    def is_placeholder(self) -> bool:
        """True while this arm's identity still awaits a live-measured decision."""
        return any(
            str(value).startswith(PLACEHOLDER_PREFIX)
            for value in (self.model_ref, self.image, self.quant)
        )


class ArmManifest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    port_block: tuple[int, int]
    arms: list[ArmEntry] = Field(min_length=1)

    @model_validator(mode='after')
    def _check_manifest_invariants(self) -> ArmManifest:
        seen_ids: set[str] = set()
        seen_ports: dict[int, str] = {}
        for arm in self.arms:
            if arm.arm_id in seen_ids:
                raise ValueError(f'duplicate arm_id {arm.arm_id!r}')
            seen_ids.add(arm.arm_id)
            if arm.port in seen_ports:
                # Two arms on one port is the precondition for the 2026-04-08
                # 404 bug (scripts/run_vllm_eval.py:541-553): a probe lands on
                # whichever unit holds the port and a DIFFERENT model answers,
                # mis-attributing an entire run.
                raise ValueError(
                    f'duplicate port {arm.port} shared by {seen_ports[arm.port]!r} '
                    f'and {arm.arm_id!r}'
                )
            seen_ports[arm.port] = arm.arm_id

        low, high = self.port_block
        if low > high:
            raise ValueError(f'port_block {self.port_block} is inverted')
        outside = [a.arm_id for a in self.arms if not low <= a.port <= high]
        if outside:
            raise ValueError(
                f'arms {outside} declare a port outside the reserved block '
                f'{low}-{high}; the block exists so an arm can never land on a '
                'port another service already owns'
            )
        return self

    def arm_ids(self) -> list[str]:
        return [arm.arm_id for arm in self.arms]

    def by_id(self, arm_id: str) -> ArmEntry:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise ArmManifestError(
            f'unknown arm_id {arm_id!r}; manifest declares {self.arm_ids()}'
        )

    def by_axis(self, axis: str) -> list[ArmEntry]:
        if axis not in ARM_AXES:
            raise ArmManifestError(
                f'unknown axis {axis!r}; known axes are {list(ARM_AXES)}'
            )
        return [arm for arm in self.arms if arm.axis == axis]


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        location = '.'.join(str(piece) for piece in err['loc']) or '<manifest>'
        given = repr(err.get('input'))
        if len(given) > 80:
            given = given[:77] + '...'
        parts.append(f'{location}: {err["msg"]} (given: {given})')
    return '; '.join(parts)


def load_arms(path: str | Path = DEFAULT_MANIFEST_PATH) -> ArmManifest:
    """Parse and validate a manifest, raising :class:`ArmManifestError` loudly.

    Every failure mode — absent file, unparseable YAML, wrong document shape,
    field-level violation, cross-arm collision — surfaces as one typed error
    carrying the offending location and value.  Nothing is defaulted, skipped,
    or repaired.
    """
    path = Path(path)
    try:
        raw_text = path.read_text()
    except OSError as exc:
        raise ArmManifestError(f'cannot read arms manifest at {path}: {exc}') from exc

    try:
        document: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ArmManifestError(f'{path} is not valid YAML: {exc}') from exc

    if not isinstance(document, dict):
        raise ArmManifestError(
            f'{path} must be a mapping with `port_block` and `arms` keys, '
            f'got {type(document).__name__}'
        )

    try:
        return ArmManifest.model_validate(document)
    except ValidationError as exc:
        raise ArmManifestError(
            f'{path} is not a valid arms manifest: {_format_validation_error(exc)}'
        ) from exc
