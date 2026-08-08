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

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import lms_ctl
import lms_vram
from lms_manifest import ArmEntry, ArmManifestError, load_arms
from pydantic import BaseModel, ConfigDict, ValidationError

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
#:
#: DERIVED, NOT CHOSEN.  This is `llm.max_tokens` from the product's own
#: `fused-memory/config/config.yaml:15` — the output budget the memory
#: subsystem actually gives its LLM.  Sourcing it there is what stops this
#: constant being a tuning knob: it cannot be nudged upward to turn a row green
#: without moving the production budget it mirrors, which is a decision with a
#: different owner and a visible blast radius.  (Leo's ruling, esc-3713-10.)
#:
#: The number it replaces — 512 — was a runaway watchdog, never a model of
#: production use, and it was silently deciding a question it had no business
#: deciding: gemma-4 needs 1807 tokens to answer this probe with thinking on
#: and 236-276 with thinking off, so a 512 cap FAILED the arm for being in a
#: mode nobody had chosen.  At 4096 the cap stops participating in the
#: reasoning decision (`arms.yaml: reasoning`) and that decision can be made on
#: its merits.  Runaway containment is unaffected — COMPLETION_TIMEOUT_S still
#: bounds the request at 180 s.
#:
#: Uniform across every LLM arm, because a per-arm cap would hand eta a
#: probe-shaped difference between arms.  Uniformity is cheap here: every arm
#: measured to date finishes well inside it (`finish_reason` `stop`).
PROBE_MAX_TOKENS = 4096

PROBE_TEXT = (
    'Leo runs Dark Factory, a software factory whose memory layer is Graphiti, '
    'a temporal knowledge graph backed by FalkorDB.'
)
#: The entities PROBE_TEXT unambiguously contains.  An extraction that finds
#: none of them is not an extraction, however well-formed its JSON is.
#:
#: This exists because schema validity turned out to be a much weaker signal
#: than this module claimed.  Measured 2026-08-06: qwen3.5-9b returned
#: `{"entities": [], "summary": "No entities extracted from the provided
#: text."}` — 18 tokens, schema-VALID, and recorded as a PASS in the committed
#: README.  `ProbeExtraction` accepts it because an empty list is a valid
#: `list[ProbeEntity]`, so the client-side validator this module calls its
#: safeguard could not tell a correct extraction from a refusal to extract.
#: The same model finds all four the moment it is allowed to reason.
#:
#: DELIBERATELY A FLOOR, NOT A SCORE.  Ranking extraction quality is eta's job
#: on delta's real-episode corpus; this only answers alpha's own question —
#: can this endpoint do the thing the eval depends on at all.  The slack of one
#: entity keeps a defensible naming difference from failing an arm that worked.
PROBE_KNOWN_ENTITIES = ('Leo', 'Dark Factory', 'Graphiti', 'FalkorDB')
PROBE_MIN_ENTITIES = 3
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
    #: The arm hit `max_tokens` (finish_reason == "length") before it produced
    #: readable output.  DISTINCT from EMPTY_COMPLETION and NOT_JSON on
    #: purpose: those two say "the arm answered badly", this one says "the arm
    #: was cut off, so we never learned whether it can answer".  Reporting a
    #: truncation as an emptiness blames the model for the harness's budget.
    COMPLETION_TRUNCATED = 'completion_truncated'
    #: Schema-valid output that extracted none (or almost none) of the entities
    #: PROBE_TEXT plainly contains.  A structurally perfect empty answer — the
    #: failure `ProbeExtraction` alone cannot see.
    EXTRACTION_MISSED_ENTITIES = 'extraction_missed_entities'
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
    #: How many known entities were promoted to TOP-LEVEL entities, as opposed
    #: to captured anywhere.  REPORTED, NON-GATING — the gap between this and
    #: the floor is a representation-quality signal eta reads and alpha declines
    #: to judge.  `None` where the question does not apply: an embedding arm, or
    #: an LLM response that never parsed.
    top_level_entities_named: int | None = None

    def with_latency(self, latency_ms: float) -> ProbeResult:
        return self.model_copy(update={'latency_ms': round(latency_ms, 1)})

    def with_top_level(self, count: int) -> ProbeResult:
        return self.model_copy(update={'top_level_entities_named': count})


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

    if arm.reasoning == 'off':
        # Suppression is the ONLY request-level lever, on both stacks: Qwen3.5's
        # template tests `enable_thinking is defined and enable_thinking is
        # false`, so passing `true` is a measured no-op, and gemma-4's template
        # defaults to false but llama.cpp overrides it to true.  Sending the
        # kwarg for `off` and nothing for `on` is therefore the only spelling
        # that means the same thing on both — and sending NOTHING at all, as
        # this probe used to, means whatever the server happens to default to.
        body['chat_template_kwargs'] = {'enable_thinking': False}

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


def _extract_reasoning(body: Any) -> str:
    """The thought channel, under either stack's spelling.

    llama.cpp emits `message.reasoning_content`; vLLM's reasoning parsers emit
    `message.reasoning`.  BOTH were observed on this rig on 2026-08-06, which
    is why neither name can be the only one checked — a helper that knew only
    one would report "no reasoning" for half the slate.
    """
    if not isinstance(body, dict):
        return ''
    choices = body.get('choices')
    if not isinstance(choices, list) or not choices:
        return ''
    first = choices[0]
    if not isinstance(first, dict):
        return ''
    message = first.get('message')
    if not isinstance(message, dict):
        return ''
    for key in ('reasoning_content', 'reasoning'):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ''


def _norm(text: str) -> str:
    return ''.join(ch for ch in text.lower() if ch.isalnum())


def _known_entities_found(
    extraction: ProbeExtraction, *, top_level_only: bool = False
) -> list[str]:
    """Which of PROBE_KNOWN_ENTITIES this extraction captured ANYWHERE.

    Attribute VALUES count, not just entity names.  Measured 2026-08-06:
    phi-4-14b returned `Dark Factory` and `Graphiti` as top-level entities and
    `FalkorDB` as an attribute value (`{"name": "Backend", "value":
    "FalkorDB"}`) — a nested, relational representation rather than a refusal.
    Scoring that 2/4 made the floor adjudicate representation QUALITY, which is
    theta's `graph-sameness` metric, not alpha's question.  The floor asks only
    whether the endpoint extracted at all.

    The SUMMARY is deliberately not scanned.  It is prose, and counting it would
    let an arm that returned zero entities pass on the strength of a
    well-written sentence — which is the refusal this floor exists to catch.

    Comparison is on alphanumerics, case-folded, and ONE-DIRECTIONAL: the
    extracted string must contain the known name, never the reverse.  'Dark
    Factory (software factory)' counts; a bare 'Factory' does not.  Loose in
    both directions, a one-character value would match every known entity at
    once and quietly restore the hole this closes.

    `top_level_only` restricts the scan to entity names.  That variant is
    REPORTED and NON-GATING — it is how eta sees whether an arm promoted the
    entities to nodes or buried them in attributes, a real quality difference
    that alpha declines to judge.
    """
    haystack = {_norm(entity.name) for entity in extraction.entities}
    if not top_level_only:
        haystack |= {
            _norm(attribute.value)
            for entity in extraction.entities
            for attribute in entity.attributes
        }
    return [
        known for known in PROBE_KNOWN_ENTITIES
        if any(candidate and _norm(known) in candidate for candidate in haystack)
    ]


def _was_truncated(body: Any) -> bool:
    """True when the server reports it stopped because it hit `max_tokens`.

    Read defensively: a body this cannot parse is reported as NOT truncated,
    so an unreadable response keeps whatever reason the content itself earns
    rather than being laundered into the softer truncation code.
    """
    if not isinstance(body, dict):
        return False
    choices = body.get('choices')
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    return first.get('finish_reason') == 'length'


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
    truncated = _was_truncated(body)
    reasoning = _extract_reasoning(body)

    if content is None:
        # A reasoning arm that spends its whole budget thinking returns a NULL
        # content on vLLM and an EMPTY STRING on llama.cpp.  Reading the null as
        # "unreadable body" — which this branch used to do unconditionally —
        # blamed the harness's own cap on a malformed response AND made
        # COMPLETION_TRUNCATED unreachable on the vLLM stack entirely.  Measured
        # 2026-08-06 on qwen3.5-9b with --reasoning-parser: 512 of 512 tokens
        # spent reasoning, finish_reason `length`, reported `malformed_response`.
        if truncated:
            return _fail(
                Reason.COMPLETION_TRUNCATED,
                f'the arm spent its whole {PROBE_MAX_TOKENS}-token budget '
                f'(finish_reason=length) and returned a null content; '
                f'{len(reasoning)} chars of it went to the thought channel',
            )
        if reasoning:
            return _fail(
                Reason.EMPTY_COMPLETION,
                f'the arm stopped normally having emitted {len(reasoning)} '
                'chars of reasoning and no content at all',
            )
        preview = json.dumps(body)[:200] if body is not None else 'None'
        return _fail(
            Reason.MALFORMED_RESPONSE,
            f'no choices[0].message.content in the response body: {preview}',
        )

    stripped = content.strip()
    if not stripped:
        if truncated:
            return _fail(
                Reason.COMPLETION_TRUNCATED,
                f'the arm spent its whole {PROBE_MAX_TOKENS}-token budget '
                '(finish_reason=length) without emitting any content — a '
                'reasoning arm whose thinking outran the cap looks exactly '
                'like this, and is NOT the same failure as an empty answer',
            )
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
        if truncated:
            return _fail(
                Reason.COMPLETION_TRUNCATED,
                f'the arm hit its {PROBE_MAX_TOKENS}-token budget mid-answer '
                f'(finish_reason=length), so the fragment does not parse: '
                f'{_exc_detail(exc)} | content={stripped[:200]!r}',
            )
        return _fail(Reason.NOT_JSON, f'{_exc_detail(exc)} | content={stripped[:200]!r}')

    if not isinstance(parsed, dict):
        return _fail(
            Reason.NOT_JSON,
            f'the completion parsed as {type(parsed).__name__}, not a JSON object',
        )

    try:
        extraction = ProbeExtraction.model_validate(parsed)
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

    # Schema validity is NOT capability.  Everything above this line passes an
    # arm that returned `{"entities": [], "summary": "..."}` — well-formed,
    # correctly typed, and empty.  That is not a hypothetical: it is what
    # qwen3.5-9b returned on every non-reasoning configuration measured on
    # 2026-08-06, and it was recorded as a PASS.
    found = _known_entities_found(extraction)
    top_level = len(_known_entities_found(extraction, top_level_only=True))
    if len(found) < PROBE_MIN_ENTITIES:
        missed = [k for k in PROBE_KNOWN_ENTITIES if k not in found]
        return _fail(
            Reason.EXTRACTION_MISSED_ENTITIES,
            f'the completion is schema-valid but captured only {len(found)} of '
            f'the {len(PROBE_KNOWN_ENTITIES)} entities PROBE_TEXT contains '
            f'(found {found or "none"}, missed {missed}); '
            f'{len(extraction.entities)} entities were returned in total',
        ).with_top_level(top_level)

    return _ok(
        f'valid {ProbeExtraction.__name__} for {arm.served_model_name}, '
        f'capturing {len(found)}/{len(PROBE_KNOWN_ENTITIES)} known entities '
        f'({top_level} as top-level entities)'
    ).with_top_level(top_level)


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


def probe_arm(arm: ArmEntry) -> ProbeResult:
    """Dispatch one arm to its axis's probe."""
    if arm.axis == 'embedding':
        return probe_embedding_arm(arm)
    return probe_llm_arm(arm)


# ---------------------------------------------------------------------------
# The report
#
# ONE structure, rendered two ways.  The JSON artifact is what step 21's
# verification test and the downstream eval tasks read; the table is what an
# operator reads.  Both come from the object below and nothing else, so they
# cannot disagree about what was measured.
# ---------------------------------------------------------------------------

#: Bump when the artifact's shape changes.  A consumer that finds an
#: unrecognised version must refuse the file rather than reinterpret it.
#: v2 (2026-08-06, esc-3713-6): the vram block gained the live per-arm baseline
#: and the arm footprint the verdict is now judged on; rows gained their own
#: measured_at and footprint so a merged slate report stays attributable.
#: v3 (2026-08-06, esc-3713-10): rows carry the REASONING MODE they were
#: measured in.  Without it a green slate is not interpretable — the two
#: thinking-capable arms were measured in opposite modes by accident, and a
#: report that omits the mode lets eta/theta inherit that choice invisibly.
#: v4 (2026-08-06, esc-3713-10): rows carry `top_level_entities_named`, the
#: reported-and-non-gating count behind the extraction floor.  Added when the
#: floor was widened to count attribute values: the widening is right for
#: alpha's question but discards a real signal, and eta needs to see whether an
#: arm promoted entities to nodes or buried them in attributes.
REPORT_SCHEMA_VERSION = 4

#: The delivered-check literal for task 3713.  Spelled once, here, and carried
#: into the JSON artifact as a field.
PRD_MARKER = 'PRD-MARKER:local-memory-models-eval serving'

EXIT_OK = 0
#: At least one arm answered wrongly, or not at all.
EXIT_ARM_FAILED = 1
#: The manifest, or the arm id asked for, is wrong. Matches lms_ctl's code.
EXIT_MANIFEST_ERROR = 2
#: Every arm passed but total VRAM is outside the budget.  Distinct from
#: EXIT_ARM_FAILED because the fix is different: stop something, don't debug a
#: model.
EXIT_VRAM_FAILED = 3
#: nvidia-smi is missing or unreadable, so nothing was measured.
EXIT_PROBE_ERROR = 4
#: `--active` found no running units.  Non-zero on purpose: a sweep that
#: measured NOTHING must never be mistaken for a sweep where everything passed.
EXIT_NO_ACTIVE_ARMS = 5
#: `--merge` refused to combine the inputs.  Distinct from an arm failure: the
#: arms may all be fine and the ASSEMBLY wrong, which is a different fix.
EXIT_MERGE_ERROR = 6


class GpuBlock(BaseModel):
    """Which hardware these verdicts belong to."""

    model_config = ConfigDict(frozen=True)

    name: str
    driver_version: str
    total_mib: int


class ArmRow(BaseModel):
    """One arm's line in the report."""

    model_config = ConfigDict(frozen=True)

    arm_id: str
    axis: str
    stack: str
    endpoint: str
    served_model_name: str
    verdict: Verdict
    reason: Reason
    detail: str
    latency_ms: float
    #: When THIS row was measured.  The merged artifact spans one run per arm,
    #: so a single top-level `measured_at` cannot say when any given arm was up.
    measured_at: str
    #: What the card held over baseline while this arm was serving.  Carried per
    #: row because the merged report keeps only ONE vram block, and without this
    #: the other seven arms' footprints would leave the artifact entirely.
    arm_footprint_mib: int
    #: The reasoning mode this row was measured in, from the manifest.  A
    #: verdict is only interpretable alongside it: the same arm passes in both
    #: modes at very different cost, and a different arm passes in one and
    #: returns an empty extraction in the other.
    reasoning: str
    #: Known entities promoted to TOP-LEVEL entities.  Reported, non-gating —
    #: see ProbeResult.  `None` for an embedding arm or an unparseable response.
    top_level_entities_named: int | None = None


class VramBlock(BaseModel):
    """The budget answer, carrying every number it was computed from.

    `verdict` judges `arm_footprint_mib` (`used_mib - baseline_mib`) against
    `budget_mib` -- the free VRAM measured immediately BEFORE this arm started.
    That subject was corrected on 2026-08-06 (esc-3713-6): judging TOTAL card
    usage against PRD D10's nominal ceiling charged every arm a second time for
    the ~7.3 GiB desktop+whisper baseline D10 had already subtracted, and failed
    a 9B AWQ that was serving correctly.

    `nominal_ceiling_gib` (PRD D10's 19.5) and `operating_budget_gib` (this
    host's 2026-08-05 measurement) are both REPORTED and NON-GATING.  They stay
    so the deviation this task measured remains legible in the artifact instead
    of living in a commit message.
    """

    model_config = ConfigDict(frozen=True)

    total_mib: int
    used_mib: int
    free_mib: int
    #: nvidia-smi `used` taken immediately before the arm started -- LIVE, never
    #: `lms_vram.MEASURED_BASELINE_GIB`.  A frozen baseline would misattribute
    #: desktop drift to the arm, in the direction that flatters it.
    baseline_mib: int
    #: nvidia-smi `free` at that same baseline reading; the ceiling `verdict`
    #: applies.
    budget_mib: int
    arm_footprint_mib: int
    used_gib: float
    free_gib: float
    baseline_gib: float
    budget_gib: float
    arm_footprint_gib: float
    nominal_ceiling_gib: float
    operating_budget_gib: float
    headroom_gib: float
    verdict: Verdict
    reason: str


class HealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    measured_at: str
    gpu: GpuBlock
    arms: list[ArmRow]
    vram: VramBlock
    overall: Verdict
    #: The delivered-check literal, in a real FIELD because JSON carries no
    #: comments.  `test_lms_marker_contract.py` enumerates committed files; this
    #: is how the artifact satisfies the same grep the source files do.
    prd_marker: str = PRD_MARKER


def _now_iso() -> str:
    """Timezone-AWARE UTC.  A naive stamp would make a stale artifact
    indistinguishable from a fresh one, and this artifact's entire job is to
    prove that a live run happened."""
    return datetime.now(UTC).isoformat()


def run_healthcheck(
    arms: Sequence[ArmEntry],
    gpu_probe: Callable[[], lms_vram.GpuSnapshot] | None = None,
    probe: Callable[[ArmEntry], ProbeResult] | None = None,
    baseline: lms_vram.GpuReading | None = None,
) -> HealthReport:
    """Probe every arm and assemble the report.

    The GPU is read FIRST and its failure is allowed to propagate.  A swallowed
    :class:`~lms_vram.VramProbeError` would produce a report whose vram block
    reads `used 0 MiB, headroom 19.5 GiB` -- a passing budget with maximal
    headroom off a dead probe, which is the most trustworthy-looking wrong
    answer this rig can emit.  No report at all is strictly better.

    Individual arm failures, by contrast, are recorded and the sweep continues:
    aborting on the first dead arm would drop verdicts already measured for the
    others and leave the report silently short of rows.
    """
    read_gpu = gpu_probe if gpu_probe is not None else lms_vram.probe_gpu_snapshot
    probe_one = probe if probe is not None else probe_arm

    snapshot = read_gpu()
    reading = snapshot.reading
    # The baseline is read BEFORE any probing, and its absence propagates for
    # the same reason a dead GPU probe does: without the reading taken before
    # these arms started, the report cannot say what the arms took, only what
    # the card holds -- and that was the miscalibrated subject esc-3713-6 fixed.
    base = baseline if baseline is not None else lms_vram.read_baselines(
        [arm.arm_id for arm in arms]
    )
    budget = lms_vram.evaluate_budget(
        reading.used_mib,
        reading.total_mib,
        baseline_mib=base.used_mib,
        baseline_free_mib=base.free_mib,
    )

    measured_at = _now_iso()
    rows: list[ArmRow] = []
    for arm in arms:
        result = probe_one(arm)
        rows.append(
            ArmRow(
                arm_id=arm.arm_id,
                axis=arm.axis,
                stack=arm.stack,
                endpoint=arm.base_url,
                served_model_name=arm.served_model_name,
                verdict=result.verdict,
                reason=result.reason,
                detail=result.detail,
                latency_ms=result.latency_ms,
                measured_at=measured_at,
                arm_footprint_mib=budget.arm_footprint_mib,
                reasoning=arm.reasoning,
                top_level_entities_named=result.top_level_entities_named,
            )
        )

    vram = VramBlock(
        total_mib=reading.total_mib,
        used_mib=reading.used_mib,
        free_mib=reading.free_mib,
        baseline_mib=budget.baseline_mib,
        budget_mib=budget.budget_mib,
        arm_footprint_mib=budget.arm_footprint_mib,
        used_gib=reading.used_gib,
        free_gib=reading.free_gib,
        baseline_gib=budget.baseline_gib,
        budget_gib=budget.budget_gib,
        arm_footprint_gib=budget.arm_footprint_gib,
        nominal_ceiling_gib=budget.nominal_ceiling_gib,
        operating_budget_gib=budget.operating_budget_gib,
        headroom_gib=budget.headroom_gib,
        verdict=budget.verdict,
        reason=budget.reason,
    )

    everything_passed = (
        all(row.verdict == 'PASS' for row in rows) and vram.verdict == 'PASS'
    )
    return HealthReport(
        schema_version=REPORT_SCHEMA_VERSION,
        measured_at=measured_at,
        gpu=GpuBlock(
            name=snapshot.identity.name,
            driver_version=snapshot.identity.driver_version,
            total_mib=reading.total_mib,
        ),
        arms=rows,
        vram=vram,
        overall='PASS' if everything_passed else 'FAIL',
    )


class ReportMergeError(Exception):
    """Two per-arm reports cannot be combined into one slate artifact."""


def merge_reports(
    reports: Sequence[HealthReport], expected_arm_ids: Sequence[str] | None = None
) -> HealthReport:
    """Combine per-arm runs into the one committed slate artifact.

    The PRD's funnel does not run all units simultaneously and this card cannot
    hold them, so the slate is measured one arm at a time.  Merging is therefore
    unavoidable -- and every way it could launder a failure is refused here:

    * Reports from different GPUs, or different schema versions, do not merge.
      A row measured on another card cannot be compared with these.
    * A duplicate `arm_id` does not merge.  Silently keeping one of two runs
      would let a failed arm be papered over by a re-run appended to the same
      file, which is the single cheapest way to fake this artifact.
    * A set that does not COVER the manifest does not merge.  An arm that never
      became ready produces no per-arm report at all, so it goes MISSING from
      the merge rather than red in it — the "slate is quietly narrower than the
      PRD commissioned" hazard this package names for `load_arms`, reappearing
      in the assembly path.  Measured 2026-08-06: a 7-row artifact was produced
      from an 8-arm manifest because mistral-small-3.2-24b never started, and
      nothing in this function noticed.  Absence is the one failure a report
      cannot describe, which is exactly why the check belongs here and not only
      in the commit-time gate.
    * The surviving vram block is the FIRST FAILING one if any input failed, and
      otherwise the one with the LARGEST arm footprint -- the binding
      measurement.  Keeping only a passing block while an input failed would put
      `overall` out of reach of its own evidence.

    Per-arm footprints survive on each row, so collapsing to one vram block
    loses no measurement.
    """
    if not reports:
        raise ReportMergeError(
            'nothing to merge; an artifact assembled from zero runs would '
            'describe a slate nobody measured'
        )

    versions = {r.schema_version for r in reports}
    if len(versions) > 1:
        raise ReportMergeError(
            f'reports carry mixed schema versions {sorted(versions)}; they '
            'describe different report shapes and cannot be combined'
        )

    gpus = {(r.gpu.name, r.gpu.driver_version, r.gpu.total_mib) for r in reports}
    if len(gpus) > 1:
        raise ReportMergeError(
            f'reports were measured on different GPUs {sorted(gpus)}; every '
            'verdict is relative to specific hardware'
        )

    rows: list[ArmRow] = []
    seen: dict[str, ArmRow] = {}
    for report in reports:
        for row in report.arms:
            if row.arm_id in seen:
                raise ReportMergeError(
                    f'arm {row.arm_id!r} appears in more than one report '
                    f'({seen[row.arm_id].verdict} at {seen[row.arm_id].measured_at}, '
                    f'{row.verdict} at {row.measured_at}). Merging would pick a '
                    'winner between two measurements of the same arm; re-run the '
                    'slate instead'
                )
            seen[row.arm_id] = row
            rows.append(row)

    if expected_arm_ids is not None:
        uncovered = sorted(set(expected_arm_ids) - set(seen))
        if uncovered:
            raise ReportMergeError(
                f'arms {uncovered} are declared in the manifest but carry no row '
                'in any input report. They were never measured, so this artifact '
                'would describe a NARROWER slate than the manifest commissions '
                'while reading as a complete one. Re-run the missing arms, or '
                'escalate with the reason they cannot be served'
            )

    failing = [r for r in reports if r.vram.verdict == 'FAIL']
    binding = (
        failing[0] if failing
        else max(reports, key=lambda r: r.vram.arm_footprint_mib)
    )

    everything_passed = (
        all(row.verdict == 'PASS' for row in rows)
        and binding.vram.verdict == 'PASS'
    )
    return HealthReport(
        schema_version=binding.schema_version,
        measured_at=max(r.measured_at for r in reports),
        gpu=binding.gpu,
        arms=rows,
        vram=binding.vram,
        overall='PASS' if everything_passed else 'FAIL',
    )


def exit_code_for(report: HealthReport) -> int:
    """An arm failure outranks a budget failure: it is the more actionable of
    the two, and the budget block is right there in the artifact either way."""
    if any(row.verdict == 'FAIL' for row in report.arms):
        return EXIT_ARM_FAILED
    if report.vram.verdict == 'FAIL':
        return EXIT_VRAM_FAILED
    return EXIT_OK


def render_table(report: HealthReport) -> str:
    """Render the report for a human.

    Takes the report and NOTHING else -- no arms, no manifest, no GPU -- so
    there is nothing left for it to recompute and the text can never contradict
    the JSON beside it.
    """
    headers = ('ARM', 'AXIS', 'STACK', 'ENDPOINT', 'VERDICT', 'REASON', 'MS')
    rows = [
        (
            row.arm_id,
            row.axis,
            row.stack,
            row.endpoint,
            row.verdict,
            str(row.reason),
            f'{row.latency_ms:.0f}',
        )
        for row in report.arms
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: tuple[str, ...]) -> str:
        return '  '.join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    out = [
        f'local model serving health — {report.measured_at}',
        f'GPU: {report.gpu.name} (driver {report.gpu.driver_version}, '
        f'{report.gpu.total_mib} MiB)',
        '',
        line(headers),
        '  '.join('-' * w for w in widths),
    ]
    out.extend(line(row) for row in rows)
    out.extend(
        [
            '',
            f'VRAM {report.vram.verdict}: the arm took '
            f'{report.vram.arm_footprint_mib} MiB '
            f'({report.vram.used_mib} used - {report.vram.baseline_mib} baseline) '
            f'of the {report.vram.budget_mib} MiB free before it started',
            f'  headroom {report.vram.headroom_gib} GiB; card total '
            f'{report.vram.total_mib} MiB, now {report.vram.free_mib} MiB free',
            f'  reported, NOT gating — PRD D10 nominal ceiling '
            f'{report.vram.nominal_ceiling_gib} GiB, measured operating budget '
            f'on this host {report.vram.operating_budget_gib} GiB',
            f'  {report.vram.reason}',
            '',
            f'OVERALL: {report.overall}',
        ]
    )
    for row in report.arms:
        if row.verdict == 'FAIL':
            out.append(f'  {row.arm_id}: {row.reason} — {row.detail}')
    return '\n'.join(out)


def _write_report(report: HealthReport, output: str | None) -> None:
    if not output:
        return
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `mode='json'` so the StrEnum reasons serialise as their string values:
    # a python-mode dump emits a repr no downstream consumer can match on.
    out_path.write_text(json.dumps(report.model_dump(mode='json'), indent=2) + '\n')
    print(f'\nwrote {out_path}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='lms_healthcheck',
        description='Probe local model serving arms for VALID output (LME-alpha).',
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--all', action='store_true', help='every arm in arms.yaml')
    selector.add_argument('--arm', help='one arm by id')
    selector.add_argument(
        '--active', action='store_true',
        help='only the arms whose lms-arm@ unit is currently running',
    )
    selector.add_argument(
        '--merge', nargs='+', metavar='REPORT',
        help='combine per-arm JSON reports into one slate artifact',
    )
    parser.add_argument('--output', help='write the JSON report to this path')
    args = parser.parse_args(argv)

    if args.merge:
        try:
            parts = [
                HealthReport.model_validate_json(Path(p).read_text())
                for p in args.merge
            ]
            # The manifest is the authority on what a complete slate is, so the
            # coverage check reads it here rather than trusting the caller to
            # pass every part — an operator who forgets one file would otherwise
            # produce a short artifact that looks whole.
            report = merge_reports(parts, expected_arm_ids=load_arms().arm_ids())
        except (ReportMergeError, ValidationError, OSError, ArmManifestError) as exc:
            # Nothing is written on refusal: a partially-merged artifact on disk
            # would later read as "the slate was checked".
            print(f'lms_healthcheck: cannot merge: {exc}', file=sys.stderr)
            return EXIT_MERGE_ERROR
        print(render_table(report))
        _write_report(report, args.output)
        return exit_code_for(report)

    try:
        manifest = load_arms()
        if args.arm:
            arms = [manifest.by_id(args.arm)]
        elif args.active:
            running = lms_ctl.active_arms()
            arms = [a for a in manifest.arms if a.arm_id in running]
        else:
            arms = list(manifest.arms)
    except ArmManifestError as exc:
        print(f'lms_healthcheck: {exc}', file=sys.stderr)
        return EXIT_MANIFEST_ERROR

    if args.active and not arms:
        # Deliberately not an empty PASS: nothing was measured, and saying so
        # is the only honest outcome.  No artifact is written either — an
        # empty report on disk would later read as "the slate was checked".
        print('lms_healthcheck: no active arms — nothing was probed')
        return EXIT_NO_ACTIVE_ARMS

    try:
        report = run_healthcheck(arms)
    except lms_vram.VramProbeError as exc:
        print(
            f'lms_healthcheck: GPU probe failed, so no report was produced: {exc}',
            file=sys.stderr,
        )
        return EXIT_PROBE_ERROR

    print(render_table(report))
    _write_report(report, args.output)
    return exit_code_for(report)


if __name__ == '__main__':  # pragma: no cover - process entry point
    raise SystemExit(main())
