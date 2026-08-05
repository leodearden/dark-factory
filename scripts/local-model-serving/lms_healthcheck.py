"""Probe every serving arm and report whether it answers with VALID output.

PRD-MARKER:local-memory-models-eval serving

Task 3713 (LME-alpha) of `plans/local-memory-models-eval-prd.md`.

"The endpoint is up" is not the question.  The question is whether the arm can
do the thing the eval depends on: return a completion that conforms to a
NESTED JSON schema, of the shape graphiti actually emits.  So this module is
built around one deliberate asymmetry — it is far easier to make a health
check pass than to make it mean something, and every design choice here buys
meaning at the cost of a cheap PASS:

* The probe model is nested, so `model_json_schema()` emits `$defs`/`$ref`.
  That is exactly the shape llama.cpp silently mishandles, falling back to
  UNCONSTRAINED output while reporting success (ggml-org/llama.cpp#21228).  A
  flat stand-in schema would hand a PASS to an arm that cannot serve the eval.

* The SAME pydantic model that produces the schema sent to the server also
  validates the response client-side.  For the `json_object`-only MoE arm that
  client-side leg is the only thing separating an unconstrained fallback
  returning prose from a green row in the report.

* A completion counts only after `/v1/models` lists the arm's
  `served_model_name`.  On 2026-04-08 a `/health` 200 landed on a colliding
  port, a DIFFERENT model answered, and an entire eval run was attributed to
  the wrong model (scripts/run_vllm_eval.py:541-553).  This rig starts and
  stops units repeatedly on a fixed port block, so that is the expected
  failure mode, not an exotic one.

* Every failure is a typed, distinct reason code.  A report that says only
  "FAIL" costs an operator the whole diagnosis again; "schema_wrong_type at
  entities.0.attributes" costs them nothing.

No `jsonschema` dependency is added: pydantic is declared by shared /
orchestrator / fused-memory and is what `uv run --project shared` actually
provides, while `jsonschema` is declared by no workspace member.
"""
from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from lms_manifest import ArmEntry

#: Per-request ceiling for the readiness/identity GET.  A plain float, never an
#: `httpx.Timeout` object: the shared test fake (scripts/tests/conftest.py)
#: exposes only `get`/`post`, and reaching for anything else is a loud miss.
IDENTITY_TIMEOUT_S = 15.0
#: The completion itself gets a far longer ceiling: a 24B AWQ model on a 3090
#: that is also holding whisper-writer is not fast, and a probe that times out
#: on a healthy-but-slow arm would report a capability failure that isn't one.
COMPLETION_TIMEOUT_S = 180.0
#: Long enough for the probe payload, short enough that a runaway generation
#: from an unconstrained arm terminates instead of pinning the GPU.
PROBE_MAX_TOKENS = 512

PROBE_TEXT = (
    'Leo runs Dark Factory, a software factory whose memory layer is Graphiti, '
    'a temporal knowledge graph backed by FalkorDB.'
)

Verdict = Literal['PASS', 'FAIL']


class HealthcheckError(Exception):
    """The check was asked to do something incoherent.

    Distinct from a FAIL verdict on purpose.  A FAIL is a measurement — this
    arm is broken.  This exception is a caller error — probing an embedding
    arm through the LLM path, say — and must never be recorded as an arm
    failure, because that would blame the model for the harness.
    """


class Reason(StrEnum):
    """Machine-readable outcome codes, one per distinguishable failure mode.

    Consumed by the JSON report (and by eta/theta triage), so the strings are
    part of the contract: rename one and a downstream filter silently matches
    nothing.
    """

    OK = 'ok'
    #: The arm's identity is still an unresolved PRD Open Question (TBD-*).
    PLACEHOLDER_ARM = 'placeholder_arm'
    #: Connection refused, DNS, timeout — the exception type is kept in detail.
    TRANSPORT_ERROR = 'transport_error'
    #: The server answered, but not with 200.
    HTTP_STATUS = 'http_status'
    #: 200 with a body this code cannot read as an OpenAI response.
    MALFORMED_RESPONSE = 'malformed_response'
    #: /v1/models does not list this arm's served_model_name.
    IDENTITY_MISMATCH = 'identity_mismatch'
    EMPTY_COMPLETION = 'empty_completion'
    #: Prose, or JSON that is not an object.
    NOT_JSON = 'not_json'
    #: Valid JSON wrapped in a ``` fence — see `verify_llm_response`.
    MARKDOWN_FENCED_JSON = 'markdown_fenced_json'
    SCHEMA_MISSING_FIELD = 'schema_missing_field'
    SCHEMA_WRONG_TYPE = 'schema_wrong_type'


# ---------------------------------------------------------------------------
# The probe model.  NESTED on purpose: `model_json_schema()` must emit
# `$defs`/`$ref`, because that is the shape graphiti's real extraction schemas
# take and the shape #21228 breaks on.
# ---------------------------------------------------------------------------


class ProbeAttribute(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    value: str


class ProbeEntity(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    entity_type: str
    attributes: list[ProbeAttribute]


class ProbeExtraction(BaseModel):
    """A miniature graphiti-shaped extraction payload."""

    model_config = ConfigDict(extra='forbid')

    entities: list[ProbeEntity]
    summary: str


class ProbeResult(BaseModel):
    """One arm's verdict.  Rendered verbatim into the JSON report."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    reason: Reason
    detail: str = ''
    latency_ms: float = 0.0

    def with_latency(self, latency_ms: float) -> ProbeResult:
        return self.model_copy(update={'latency_ms': round(latency_ms, 1)})


def _ok(detail: str = '') -> ProbeResult:
    return ProbeResult(verdict='PASS', reason=Reason.OK, detail=detail)


def _fail(reason: Reason, detail: str) -> ProbeResult:
    return ProbeResult(verdict='FAIL', reason=reason, detail=detail)


def _exc_detail(exc: BaseException) -> str:
    """Keep the exception TYPE, not just its message.

    "it timed out" and "it refused the connection" are different operational
    problems with different fixes, and the message alone often says neither.
    """
    return f'{type(exc).__name__}: {exc}'


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def build_llm_probe_request(arm: ArmEntry) -> dict[str, Any]:
    """The chat-completions body for *arm*.

    The schema is described IN THE PROMPT for every arm, not only the
    unconstrained one.  Keeping the prompt identical across arms means the
    only difference under measurement is the enforcement mechanism, which is
    the thing the eval is actually comparing.
    """
    if arm.axis != 'llm':
        raise HealthcheckError(
            f'arm {arm.arm_id!r} is axis={arm.axis!r}; the LLM probe does not '
            'apply. Use build_embedding_probe_request for the embedding axis.'
        )
    if arm.structured_output_mode == 'none':
        raise HealthcheckError(
            f'arm {arm.arm_id!r} declares structured_output_mode=none; there is '
            'no structured-output capability to probe'
        )

    schema = ProbeExtraction.model_json_schema()
    body: dict[str, Any] = {
        'model': arm.served_model_name,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You extract structured records. Reply with one JSON object '
                    'and nothing else: no prose, no code fence.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'{PROBE_TEXT}\n\nExtract the entities. Reply with one JSON '
                    f'object conforming to this schema:\n{json.dumps(schema)}'
                ),
            },
        ],
        # Capability, not sampling luck.
        'temperature': 0,
        'max_tokens': PROBE_MAX_TOKENS,
    }

    if arm.structured_output_mode == 'json_schema':
        body['response_format'] = {
            'type': 'json_schema',
            'json_schema': {'name': 'probe_extraction', 'schema': schema},
        }
    else:
        # llama.cpp cannot honour a $ref/$defs schema (#21228); asking it to
        # would produce an unconstrained completion that LOOKS constrained.
        # The client-side validator in verify_llm_response is what actually
        # decides this arm's verdict.
        body['response_format'] = {'type': 'json_object'}

    return body


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def _extract_content(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    choices = body.get('choices')
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get('message')
    if not isinstance(message, dict):
        return None
    content = message.get('content')
    return content if isinstance(content, str) else None


def verify_llm_response(arm: ArmEntry, body: Any) -> ProbeResult:
    """Judge one completion against the probe model.

    Applied identically to every LLM arm regardless of `structured_output_mode`
    — that uniformity IS the measurement.  A `json_object`-only arm whose
    server-side constraint is weaker has to earn its PASS on the same terms.

    Markdown-fenced JSON is a FAIL with its own code, not a tolerant parse.
    That is a deliberate choice: a fence means the server did not honour the
    structured-output contract, which is precisely the signal this eval is
    trying to read.  Stripping the fence here would launder an unconstrained
    arm into a PASS and hide the capability gap.  The distinct code keeps it
    diagnosable, so nobody has to guess whether an arm emitted prose or merely
    dressed up valid JSON.
    """
    content = _extract_content(body)
    if content is None:
        preview = json.dumps(body)[:200] if body is not None else 'None'
        return _fail(
            Reason.MALFORMED_RESPONSE,
            f'no choices[0].message.content in the response body: {preview}',
        )

    stripped = content.strip()
    if not stripped:
        return _fail(Reason.EMPTY_COMPLETION, 'the completion was empty')

    if stripped.startswith('```'):
        return _fail(
            Reason.MARKDOWN_FENCED_JSON,
            'the completion is wrapped in a markdown code fence, so the '
            'structured-output contract was not honoured',
        )

    try:
        parsed = json.loads(stripped)
    except ValueError as exc:
        return _fail(Reason.NOT_JSON, f'{_exc_detail(exc)} | content={stripped[:200]!r}')

    if not isinstance(parsed, dict):
        return _fail(
            Reason.NOT_JSON,
            f'the completion parsed as {type(parsed).__name__}, not a JSON object',
        )

    try:
        ProbeExtraction.model_validate(parsed)
    except ValidationError as exc:
        errors = exc.errors()
        missing = [e for e in errors if e['type'] == 'missing']
        chosen = missing or errors
        located = '; '.join(
            f'{".".join(str(piece) for piece in e["loc"]) or "<root>"}: {e["msg"]}'
            for e in chosen
        )
        reason = Reason.SCHEMA_MISSING_FIELD if missing else Reason.SCHEMA_WRONG_TYPE
        return _fail(reason, located)

    return _ok(f'valid {ProbeExtraction.__name__} for {arm.served_model_name}')


def check_model_identity(arm: ArmEntry, models_body: Any) -> ProbeResult:
    """Verify the arm serving this port is the arm we think it is.

    Carried over from scripts/run_vllm_eval.py:541-553 — the 2026-04-08 404
    bug, where a stale unit holding a port made a DIFFERENT model answer a
    healthy-looking probe and mis-attributed a whole run's metrics.
    """
    data = models_body.get('data') if isinstance(models_body, dict) else None
    if not isinstance(data, list):
        return _fail(
            Reason.IDENTITY_MISMATCH,
            f'/v1/models returned no `data` list (got {json.dumps(models_body)[:200]}), '
            f'so {arm.served_model_name!r} could not be confirmed on port {arm.port}',
        )

    served = [entry.get('id') for entry in data if isinstance(entry, dict)]
    if arm.served_model_name in served:
        return _ok(f'{arm.served_model_name} confirmed on port {arm.port}')

    return _fail(
        Reason.IDENTITY_MISMATCH,
        f'port {arm.port} serves {served!r}, not {arm.served_model_name!r} — a '
        'stale unit is holding the port and would mis-attribute this arm',
    )


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def probe_llm_arm(arm: ArmEntry) -> ProbeResult:
    """Identity, then completion.  Never raises on a transport failure.

    A traceback out of here would abort the whole sweep on the first dead arm
    and lose the verdicts already measured for the others.
    """
    if arm.is_placeholder:
        return _fail(
            Reason.PLACEHOLDER_ARM,
            f'arm {arm.arm_id!r} still carries TBD placeholders '
            f'(model_ref={arm.model_ref!r}, image={arm.image!r}, quant={arm.quant!r}); '
            'the PRD Open Question it depends on is unresolved, so there is '
            'nothing to probe',
        )

    request = build_llm_probe_request(arm)

    import httpx  # lazy: keeps import cost off every consumer of this module

    started = time.monotonic()

    def elapsed_ms() -> float:
        return (time.monotonic() - started) * 1000.0

    try:
        models = httpx.get(f'{arm.base_url}/v1/models', timeout=IDENTITY_TIMEOUT_S)
    except Exception as exc:
        return _fail(
            Reason.TRANSPORT_ERROR, f'GET /v1/models: {_exc_detail(exc)}'
        ).with_latency(elapsed_ms())

    if models.status_code != 200:
        return _fail(
            Reason.HTTP_STATUS, f'GET /v1/models returned {models.status_code}'
        ).with_latency(elapsed_ms())

    try:
        models_body = models.json()
    except Exception as exc:
        return _fail(
            Reason.MALFORMED_RESPONSE, f'GET /v1/models: {_exc_detail(exc)}'
        ).with_latency(elapsed_ms())

    identity = check_model_identity(arm, models_body)
    if identity.verdict == 'FAIL':
        return identity.with_latency(elapsed_ms())

    try:
        completion = httpx.post(
            f'{arm.base_url}/v1/chat/completions',
            json=request,
            timeout=COMPLETION_TIMEOUT_S,
        )
    except Exception as exc:
        return _fail(
            Reason.TRANSPORT_ERROR, f'POST /v1/chat/completions: {_exc_detail(exc)}'
        ).with_latency(elapsed_ms())

    if completion.status_code != 200:
        return _fail(
            Reason.HTTP_STATUS,
            f'POST /v1/chat/completions returned {completion.status_code}',
        ).with_latency(elapsed_ms())

    try:
        completion_body = completion.json()
    except Exception as exc:
        return _fail(
            Reason.MALFORMED_RESPONSE, f'POST /v1/chat/completions: {_exc_detail(exc)}'
        ).with_latency(elapsed_ms())

    return verify_llm_response(arm, completion_body).with_latency(elapsed_ms())
