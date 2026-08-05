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
import math
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
#: Embedding arms are probed with a QUERY, not a passage: the Qwen3-Embedding
#: family's `query_prefix` is a query-side instruct prefix, so probing with a
#: document would exercise the wrong half of an asymmetric model.
EMBEDDING_PROBE_QUERY = 'Which graph database backs the memory layer?'
EMBEDDING_TIMEOUT_S = 60.0
#: Below this L2 norm a vector carries no direction, so cosine similarity
#: against it is undefined and every retrieval score computed from it is noise.
MIN_EMBEDDING_NORM = 1e-6

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

    # -- embedding axis --------------------------------------------------
    #: 200 with a `data` list that is empty.
    EMPTY_EMBEDDING_DATA = 'empty_embedding_data'
    #: The vector's length is not the arm's declared `dims`.
    DIMS_MISMATCH = 'dims_mismatch'
    #: A NaN, an infinity, or a value that is not a number at all.
    NON_FINITE_EMBEDDING = 'non_finite_embedding'
    #: A zero (or near-zero) vector — finite, right length, and useless.
    DEGENERATE_EMBEDDING = 'degenerate_embedding'


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
# Embedding axis
#
# An embedding arm fails QUIETLY in a way an LLM arm does not: it returns a
# vector of plausible-looking floats no matter what.  A wrong-length vector, a
# NaN, or an all-zero degenerate output all still LOOK like an embedding, and
# every one of them would corrupt iota's retrieval numbers rather than
# crashing anything.  These checks are the only place they can be caught.
# ---------------------------------------------------------------------------


def build_embedding_probe_request(arm: ArmEntry) -> dict[str, Any]:
    """The `/v1/embeddings` body for *arm*, with its declared prefix applied.

    The prefix lives in the manifest precisely so it cannot be forgotten at a
    call site: the Qwen3-Embedding family REQUIRES a query-side instruct
    prefix (PRD line 134), and omitting it produces a perfectly well-formed
    vector that is simply worse.  That degradation is invisible to every check
    except a side-by-side retrieval comparison — which is exactly what iota
    runs, and exactly what it would then mis-attribute to the model.
    """
    if arm.axis != 'embedding':
        raise HealthcheckError(
            f'arm {arm.arm_id!r} is axis={arm.axis!r}; the embedding probe does '
            'not apply. Use build_llm_probe_request for the LLM axis.'
        )

    text = f'{arm.query_prefix or ""}{EMBEDDING_PROBE_QUERY}'
    return {
        'model': arm.served_model_name,
        'input': [text],
        # Ask for plain floats: some servers default to base64, which would
        # make every vector fail the finiteness check for the wrong reason.
        'encoding_format': 'float',
    }


def verify_embedding_response(arm: ArmEntry, body: Any) -> ProbeResult:
    """Judge one embedding response against the arm's declared contract."""
    if arm.dims is None:
        raise HealthcheckError(
            f'arm {arm.arm_id!r} declares no dims; there is nothing to verify '
            'the returned vector against'
        )

    if not isinstance(body, dict):
        return _fail(
            Reason.MALFORMED_RESPONSE,
            f'the embeddings response is a {type(body).__name__}, not an object',
        )

    # Defence in depth behind the /v1/models gate: the OpenAI embeddings
    # response echoes the model that answered.  Absence is tolerated (not
    # every stack fills it); a MISMATCH never is.
    answered_by = body.get('model')
    if isinstance(answered_by, str) and answered_by != arm.served_model_name:
        return _fail(
            Reason.IDENTITY_MISMATCH,
            f'port {arm.port} was answered by {answered_by!r}, not '
            f'{arm.served_model_name!r}',
        )

    data = body.get('data')
    if not isinstance(data, list):
        return _fail(
            Reason.MALFORMED_RESPONSE,
            f'the embeddings response carries no `data` list: '
            f'{json.dumps(body, default=str)[:200]}',
        )
    if not data:
        return _fail(
            Reason.EMPTY_EMBEDDING_DATA,
            'the embeddings response carries an empty `data` list, so the arm '
            'returned no vector at all',
        )

    first = data[0]
    vector = first.get('embedding') if isinstance(first, dict) else None
    if not isinstance(vector, list):
        return _fail(
            Reason.MALFORMED_RESPONSE,
            f'data[0].embedding is a {type(vector).__name__}, not a list of floats',
        )

    if len(vector) != arm.dims:
        return _fail(
            Reason.DIMS_MISMATCH,
            f'arm declares dims={arm.dims} but the server returned a vector of '
            f'length {len(vector)}; iota compares arms at their declared dims, '
            'so this arm and the manifest disagree about what is being measured',
        )

    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _fail(
                Reason.NON_FINITE_EMBEDDING,
                f'data[0].embedding[{index}] is {value!r} '
                f'({type(value).__name__}), not a number',
            )
        if not math.isfinite(value):
            return _fail(
                Reason.NON_FINITE_EMBEDDING,
                f'data[0].embedding[{index}] is {value!r}; a NaN or infinity '
                'poisons every similarity computed from this vector',
            )

    norm = math.sqrt(math.fsum(float(v) * float(v) for v in vector))
    if norm < MIN_EMBEDDING_NORM:
        return _fail(
            Reason.DEGENERATE_EMBEDDING,
            f'the vector has L2 norm {norm!r} — finite and the right length, '
            'but with no direction, so cosine similarity against it is '
            'undefined and every retrieval score from it would be noise',
        )

    return _ok(f'{len(vector)}-dim vector, L2 norm {norm:.4f}')


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _identity_gate(arm: ArmEntry, http: Any) -> ProbeResult:
    """PASS, or the FAIL that must abort this probe before it measures anything.

    Carried over from scripts/run_vllm_eval.py:541-553: without this leg the
    worst outcome is silent — a stale unit holding the port answers, and a
    whole arm's metrics land under the wrong model's name.
    """
    try:
        models = http.get(f'{arm.base_url}/v1/models', timeout=IDENTITY_TIMEOUT_S)
    except Exception as exc:
        return _fail(Reason.TRANSPORT_ERROR, f'GET /v1/models: {_exc_detail(exc)}')

    if models.status_code != 200:
        return _fail(
            Reason.HTTP_STATUS, f'GET /v1/models returned {models.status_code}'
        )

    try:
        models_body = models.json()
    except Exception as exc:
        return _fail(Reason.MALFORMED_RESPONSE, f'GET /v1/models: {_exc_detail(exc)}')

    return check_model_identity(arm, models_body)


def _probe(
    arm: ArmEntry,
    *,
    path: str,
    request: dict[str, Any],
    timeout_s: float,
    verify: Any,
) -> ProbeResult:
    """Identity, then the measurement.  Never raises on a transport failure.

    A traceback out of here would abort the whole sweep on the first dead arm
    and lose the verdicts already measured for the others — the report would
    then be both incomplete AND silent about being incomplete.
    """
    import httpx  # lazy: keeps import cost off every consumer of this module

    started = time.monotonic()

    def elapsed_ms() -> float:
        return (time.monotonic() - started) * 1000.0

    identity = _identity_gate(arm, httpx)
    if identity.verdict == 'FAIL':
        return identity.with_latency(elapsed_ms())

    try:
        response = httpx.post(f'{arm.base_url}{path}', json=request, timeout=timeout_s)
    except Exception as exc:
        return _fail(
            Reason.TRANSPORT_ERROR, f'POST {path}: {_exc_detail(exc)}'
        ).with_latency(elapsed_ms())

    if response.status_code != 200:
        return _fail(
            Reason.HTTP_STATUS, f'POST {path} returned {response.status_code}'
        ).with_latency(elapsed_ms())

    try:
        body = response.json()
    except Exception as exc:
        return _fail(
            Reason.MALFORMED_RESPONSE, f'POST {path}: {_exc_detail(exc)}'
        ).with_latency(elapsed_ms())

    return verify(arm, body).with_latency(elapsed_ms())


def _placeholder_refusal(arm: ArmEntry) -> ProbeResult | None:
    """Refuse a TBD arm BEFORE any request.

    Probing it anyway would 404 on a literal `TBD-Q3` model id and record that
    as an arm failure, burying the real cause — an unresolved PRD Open
    Question — under a transport error that looks like every other one.
    """
    if not arm.is_placeholder:
        return None
    return _fail(
        Reason.PLACEHOLDER_ARM,
        f'arm {arm.arm_id!r} still carries TBD placeholders '
        f'(model_ref={arm.model_ref!r}, image={arm.image!r}, quant={arm.quant!r}); '
        'the PRD Open Question it depends on is unresolved, so there is '
        'nothing to probe',
    )


def probe_llm_arm(arm: ArmEntry) -> ProbeResult:
    """Probe one LLM arm end to end: identity, then a constrained completion."""
    refusal = _placeholder_refusal(arm)
    if refusal is not None:
        return refusal

    return _probe(
        arm,
        path='/v1/chat/completions',
        request=build_llm_probe_request(arm),
        timeout_s=COMPLETION_TIMEOUT_S,
        verify=verify_llm_response,
    )


def probe_embedding_arm(arm: ArmEntry) -> ProbeResult:
    """Probe one embedding arm end to end: identity, then a real vector."""
    refusal = _placeholder_refusal(arm)
    if refusal is not None:
        return refusal

    return _probe(
        arm,
        path='/v1/embeddings',
        request=build_embedding_probe_request(arm),
        timeout_s=EMBEDDING_TIMEOUT_S,
        verify=verify_embedding_response,
    )
