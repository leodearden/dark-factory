"""The consolidation adjudicator: one LLM ruling on one proposal (plans/consolidation-adjudicator-prd.md K3).

A function of its typed request. The instructions and the input template are
module constants and the strict output schema is built afresh per use;
:func:`adjudicator_identity` hashes them with every setting that can change a
verdict or what code does with it.
K1's keep rule lives here too, because the executor and the calibration both
apply it to a kept list. Imports no executor code.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

from fused_memory.config.env_precedence import warn_if_ambient_base_url_is_overridden
from fused_memory.llm.list_prices import list_price
from fused_memory.llm.openai_responses import (
    ResponsesCall,
    ResponsesReply,
    call_openai_responses,
    endpoint_host,
)
from fused_memory.reconciliation.consolidation_auto import (
    REASON_CODE_CLASSES,
    AutoReasonCode,
    ReasonCodeClass,
)

if TYPE_CHECKING:
    import openai
    from openai.types.responses import ResponseFormatTextJSONSchemaConfigParam

    from fused_memory.config.schema import FusedMemoryConfig, ReasoningEffort


INSTRUCTIONS = """\
You review one proposal to consolidate notes in a knowledge base that software agents write and later search. The proposal names a set of member notes that share one category and a one-sentence claim meant to state the lesson they share.

What proceed does: a short index note carrying the claim is written and pinned first in search results for this topic, and each kept member is tagged with the topic. Nothing is deleted. A wrong proceed pins a misleading claim above notes it misrepresents and ties unrelated notes together; a wrong drop only leaves the notes as they were. When the topic's existing index note is shown, a proceed attaches the kept members to that note, which will not be amended.

Rules:
1. Notes belong together when they teach the same lesson about the same subject: a restatement, a rediscovery, the same lesson with different supporting detail, or a note and its later correction when the claim states the corrected fact. They do NOT belong together when they merely share an area, a tool, a file, a task or a mechanism but teach different things.
2. Exclude members that do not belong rather than dropping the whole proposal: if at least two members belong together, answer proceed with reason_code same_lesson and list exactly those members in keep_member_ids.
3. Facts computed from a member's record are listed inside that member's block. Keeping a member that carries a listed fact means you have read the fact and judge the fold right all the same; if the fact makes the member wrong for this fold, leave it out of keep_member_ids.
4. When the notes make incompatible claims and the texts do not show which is current, answer unresolved with reason_code conflicting_claims. Never guess which one is current. A note that explicitly corrects or supersedes an earlier one shows which is current; that is not a conflict.
5. If no two members teach the same lesson, answer drop with reason_code not_same_lesson. If the shared lesson is no longer true because the notes are stale or superseded, answer drop with reason_code stale_or_superseded. If the texts are too thin to judge, answer unresolved with reason_code insufficient_information.
6. Members excluded because they belong to another topic are shown with that topic and the first paragraph of its index note. If an existing topic already teaches this proposal's lesson, answer drop with reason_code same_referent_as_existing_topic.
7. The claim must be right for the kept members: one sentence, at most 200 characters, one assertion, stating the lesson they share and nothing they do not support. If the given claim is wrong for the kept members (says something they do not say, is broader or narrower than their shared lesson, or states a fact they correct), put a replacement in claim; otherwise claim is null. Give a claim only when the decision is proceed.

Answer with one JSON object holding exactly these fields: decision (proceed, drop or unresolved), keep_member_ids (the kept members' ids; empty unless proceed), claim (a replacement claim, or null), reason_code, and reason (one sentence).
"""


@dataclass(frozen=True)
class InputTemplate:
    """Every string :func:`render_request` fills; the renderer adds only layout."""

    topic_line: str
    claim_line: str
    category_line: str
    member_count_line: str
    member_header: str
    member_category: str
    facts_header: str
    hazard_line: str
    none_line: str
    cut_marker: str
    excluded_header: str
    excluded_member: str
    incumbent_header: str
    hazard_descriptions: tuple[tuple[AutoReasonCode, str], ...]

    def describe(self, code: AutoReasonCode) -> str:
        return dict(self.hazard_descriptions)[code]


INPUT_TEMPLATE = InputTemplate(
    topic_line='Topic: {topic}',
    claim_line='Proposed claim: {claim}',
    category_line='Category: {category}',
    member_count_line='Members: {count}',
    member_header='### Member {id}',
    member_category='category: {category}',
    facts_header='Facts computed from the record:',
    hazard_line='- {description}',
    none_line='- none',
    cut_marker=' [cut]',
    excluded_header='Members excluded because they belong to another topic:',
    excluded_member=(
        '### Excluded member {id}\n'
        'topic: {topic}\n'
        "first paragraph of that topic's index note: {first_paragraph}"
    ),
    incumbent_header=(
        "This topic's existing index note (a proceed attaches the kept members to it; "
        'it will not be amended):'
    ),
    hazard_descriptions=(
        (
            AutoReasonCode.member_carries_correction_banner,
            'its body carries a correction or supersession banner: it corrects, '
            'or was corrected by, another note',
        ),
    ),
)


class Decision(StrEnum):
    proceed = 'proceed'
    drop = 'drop'
    unresolved = 'unresolved'


class VerdictReasonCode(StrEnum):
    same_lesson = 'same_lesson'
    not_same_lesson = 'not_same_lesson'
    conflicting_claims = 'conflicting_claims'
    stale_or_superseded = 'stale_or_superseded'
    same_referent_as_existing_topic = 'same_referent_as_existing_topic'
    insufficient_information = 'insufficient_information'


def output_schema() -> dict[str, Any]:
    """Strict structured output: every property required, no others (OpenAI strict mode).

    Built afresh on each call, so no holder's mutation can reach the identity
    or a later request.
    """
    return {
        'type': 'object',
        'properties': {
            'decision': {'type': 'string', 'enum': [d.value for d in Decision]},
            'keep_member_ids': {'type': 'array', 'items': {'type': 'string'}},
            'claim': {'type': ['string', 'null']},
            'reason_code': {'type': 'string', 'enum': [c.value for c in VerdictReasonCode]},
            'reason': {'type': 'string'},
        },
        'required': ['decision', 'keep_member_ids', 'claim', 'reason_code', 'reason'],
        'additionalProperties': False,
    }


def text_format() -> ResponseFormatTextJSONSchemaConfigParam:
    """The ``text.format`` every adjudicator call sends, carrying :func:`output_schema`; fresh per call."""
    return {
        'type': 'json_schema',
        'name': 'consolidation_adjudication',
        'strict': True,
        'schema': output_schema(),
    }

MAX_OUTPUT_TOKENS = 4000

#: K1's keep rule as an integer ratio: at least two thirds of the proposed members.
KEEP_RULE_MIN_SHARE = (2, 3)

#: The claim arm of ``server/consolidation.py::proposal_shape_codes``, which the
#: executor adapts: ``(claim, topic)`` -> problems, empty when the claim passes.
ClaimCheck = Callable[[str, str | None], Sequence[str]]


def keep_rule_holds(kept: int, proposed: int, *, member_min: int) -> bool:
    """K1: ``kept >= member_min`` and ``kept / proposed >= 2/3``, in integers."""
    numerator, denominator = KEEP_RULE_MIN_SHARE
    return kept >= member_min and denominator * kept >= numerator * proposed


@dataclass(frozen=True)
class MemberInput:
    id: str
    category: str
    text: str


@dataclass(frozen=True)
class ExcludedMember:
    """A member the repair excluded because it carries another topic."""

    id: str
    topic: str
    canonical_first_paragraph: str


@dataclass(frozen=True)
class Hazard:
    code: AutoReasonCode
    subject_id: str


@dataclass(frozen=True)
class AdjudicationRequest:
    """One proposal as the adjudicator sees it. ``topic`` is None only where none exists."""

    topic: str | None
    claim: str
    category: str
    members: tuple[MemberInput, ...]
    excluded: tuple[ExcludedMember, ...] = ()
    incumbent_text: str | None = None
    hazards: tuple[Hazard, ...] = ()

    def __post_init__(self) -> None:
        problem = _request_problem(self)
        if problem is not None:
            raise ValueError(f'invalid adjudication request: {problem}')

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member.id for member in self.members)


def _request_problem(request: AdjudicationRequest) -> str | None:
    if not request.members:
        return 'it names no member'
    seen: set[str] = set()
    for member_id in request.member_ids:
        if member_id in seen:
            return f'member id {member_id!r} is repeated'
        seen.add(member_id)
    for hazard in request.hazards:
        decided_by = REASON_CODE_CLASSES.get(hazard.code)
        if decided_by is not ReasonCodeClass.ADJUDICATOR:
            return (
                f'hazard {hazard.code} is decided by {decided_by or "nobody"}, '
                'not by the adjudicator'
            )
        if hazard.subject_id not in seen:
            return f'hazard {hazard.code} names {hazard.subject_id!r}, which is not a shown member'
    return None


def _cut(text: str, field_chars: int) -> str:
    if len(text) <= field_chars:
        return text
    return text[:field_chars] + INPUT_TEMPLATE.cut_marker


def _render_member(member: MemberInput, hazards: Sequence[Hazard], field_chars: int) -> str:
    template = INPUT_TEMPLATE
    facts = [
        template.hazard_line.format(description=template.describe(hazard.code))
        for hazard in hazards
        if hazard.subject_id == member.id
    ]
    return '\n'.join([
        template.member_header.format(id=member.id),
        template.member_category.format(category=member.category),
        template.facts_header,
        *(facts or [template.none_line]),
        _cut(member.text, field_chars),
    ])


def render_request(request: AdjudicationRequest, *, field_chars: int) -> str:
    """The user content for *request*: :data:`INPUT_TEMPLATE` filled, members in request order.

    Every free text (member, excluded canonical paragraph, incumbent) is cut at
    *field_chars* and marked.
    """
    template = INPUT_TEMPLATE
    head = [] if request.topic is None else [template.topic_line.format(topic=request.topic)]
    head += [
        template.claim_line.format(claim=request.claim),
        template.category_line.format(category=request.category),
        template.member_count_line.format(count=len(request.members)),
    ]
    blocks = ['\n'.join(head)]
    blocks += [_render_member(m, request.hazards, field_chars) for m in request.members]
    if request.excluded:
        blocks.append('\n'.join([
            template.excluded_header,
            *(
                template.excluded_member.format(
                    id=excluded.id,
                    topic=excluded.topic,
                    first_paragraph=_cut(excluded.canonical_first_paragraph, field_chars),
                )
                for excluded in request.excluded
            ),
        ]))
    if request.incumbent_text is not None:
        blocks.append(
            template.incumbent_header + '\n' + _cut(request.incumbent_text, field_chars),
        )
    return '\n\n'.join(blocks) + '\n'


@dataclass(frozen=True)
class AdjudicatorSettings:
    """The configuration one adjudication runs under."""

    model: str
    reasoning_effort: ReasoningEffort | None
    #: ``host[:port]`` the call goes to, or None for the SDK's default endpoint.
    endpoint_host: str | None
    timeout_seconds: float
    field_chars: int
    member_max: int
    member_min: int

    @classmethod
    def from_config(cls, config: FusedMemoryConfig) -> AdjudicatorSettings:
        """The K7 leaves, ``member_min`` and the host of the endpoint actually called."""
        auto = config.consolidation_auto
        return cls(
            model=auto.adjudicator_model,
            reasoning_effort=auto.adjudicator_reasoning_effort,
            endpoint_host=endpoint_host(_openai_client_kwargs(config).get('base_url')),
            timeout_seconds=auto.adjudicator_timeout_seconds,
            field_chars=auto.adjudicator_field_chars,
            member_max=auto.adjudicator_member_max,
            member_min=auto.member_min,
        )


def _openai_client_kwargs(config: FusedMemoryConfig) -> dict[str, str]:
    """``api_key``/``base_url`` from ``llm.providers.openai``; a blank value is not forwarded."""
    section = config.llm.providers.openai
    if section is None:
        return {}
    kwargs: dict[str, str] = {}
    if section.api_key:
        kwargs['api_key'] = section.api_key
    if section.api_url:
        kwargs['base_url'] = section.api_url
    return kwargs


def adjudicator_identity(settings: AdjudicatorSettings) -> str:
    """K3's identity: sha256 over everything that can change a verdict or its use.

    The timeout is left out: it changes whether a reply arrives, not what it says.
    """
    material = {
        'model': settings.model,
        'reasoning_effort': settings.reasoning_effort,
        'endpoint_host': settings.endpoint_host,
        'instructions': INSTRUCTIONS,
        'input_template': dataclasses.asdict(INPUT_TEMPLATE),
        'output_schema': output_schema(),
        'max_output_tokens': MAX_OUTPUT_TOKENS,
        'field_chars': settings.field_chars,
        'member_max': settings.member_max,
        'member_min': settings.member_min,
        'keep_rule_min_share': list(KEEP_RULE_MIN_SHARE),
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Verdict:
    """A reply that parsed, kept only shown members and carried a passing claim."""

    decision: Decision
    keep_member_ids: tuple[str, ...]
    #: A replacement claim, or None to keep the proposal's.
    claim: str | None
    reason_code: VerdictReasonCode
    reason: str

    def __post_init__(self) -> None:
        if self.decision is not Decision.proceed and (self.keep_member_ids or self.claim):
            raise ValueError(
                f'a {self.decision} verdict keeps no member and replaces no claim',
            )


class VerdictInvalid(ValueError):
    """A reply that is not a verdict; the message says why."""


_VERDICT_KEYS = frozenset(output_schema()['properties'])


_E = TypeVar('_E', bound=StrEnum)


def _enum_value(enum: type[_E], key: str, value: object) -> _E:
    try:
        return enum(value)
    except ValueError:
        raise VerdictInvalid(
            f'{key} {value!r} is not one of {[member.value for member in enum]}',
        ) from None


def _kept_ids(value: object, shown: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(i, str) for i in value):
        raise VerdictInvalid(f'keep_member_ids must be a list of ids, got {value!r}')
    unshown = [i for i in value if i not in shown]
    if unshown:
        raise VerdictInvalid(f'keep_member_ids names members not shown: {unshown!r}')
    repeated = sorted({i for i in value if value.count(i) > 1})
    if repeated:
        raise VerdictInvalid(f'keep_member_ids repeats {", ".join(map(repr, repeated))}')
    return tuple(value)


def parse_verdict(text: str, request: AdjudicationRequest, claim_check: ClaimCheck) -> Verdict:
    """K3: *text* is a verdict only if it parses, keeps only shown ids and its claim passes.

    Raises :class:`VerdictInvalid` naming the first condition that fails.
    """
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VerdictInvalid(f'reply is not JSON: {exc}') from None
    if not isinstance(body, dict):
        raise VerdictInvalid(f'reply is not a JSON object: {type(body).__name__}')
    missing, extra = _VERDICT_KEYS - body.keys(), body.keys() - _VERDICT_KEYS
    if missing:
        raise VerdictInvalid(f'reply is missing {sorted(missing)}')
    if extra:
        raise VerdictInvalid(f'reply carries unknown keys {sorted(extra)}')
    decision = _enum_value(Decision, 'decision', body['decision'])
    reason_code = _enum_value(VerdictReasonCode, 'reason_code', body['reason_code'])
    kept = _kept_ids(body['keep_member_ids'], request.member_ids)
    claim, reason = body['claim'], body['reason']
    if not isinstance(reason, str):
        raise VerdictInvalid(f'reason must be a string, got {reason!r}')
    if claim is not None and not isinstance(claim, str):
        raise VerdictInvalid(f'claim must be a string or null, got {claim!r}')
    if decision is not Decision.proceed and kept:
        raise VerdictInvalid(f'a {decision} verdict must leave keep_member_ids empty')
    if decision is not Decision.proceed and claim is not None:
        raise VerdictInvalid(f'a {decision} verdict must leave claim null')
    if claim is not None:
        problems = list(claim_check(claim, request.topic))
        if problems:
            raise VerdictInvalid(f'replacement claim fails the claim rules: {"; ".join(problems)}')
    return Verdict(
        decision=decision, keep_member_ids=kept, claim=claim,
        reason_code=reason_code, reason=reason,
    )


def openai_model_call(config: FusedMemoryConfig) -> ResponsesCall:
    """The production model call: the endpoint :meth:`AdjudicatorSettings.from_config` names, one attempt."""
    client_kwargs = _openai_client_kwargs(config)
    warn_if_ambient_base_url_is_overridden(
        client_kwargs.get('base_url'), context='consolidation adjudicator',
    )
    return functools.partial(call_openai_responses, client_kwargs, max_retries=0)


class FailureKind(StrEnum):
    """K3's two kinds of failure: the reply did not arrive, or it is not a verdict."""

    TRANSPORT = 'transport'
    CONTENT = 'content'


@dataclass(frozen=True)
class CallMeta:
    """What one adjudicator call cost and which model answered it (PRD E13)."""

    identity: str
    resolved_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    #: At list price for the requested model; None when unpriced or without usage.
    cost_usd: float | None
    latency_s: float


@dataclass(frozen=True)
class Adjudication:
    verdict: Verdict
    meta: CallMeta


@dataclass(frozen=True)
class AdjudicationFailure:
    """Never a judgement. ``meta`` is None when no reply arrived."""

    kind: FailureKind
    detail: str
    meta: CallMeta | None


AdjudicationResult = Adjudication | AdjudicationFailure


def classify_failure(exc: TimeoutError | openai.OpenAIError) -> FailureKind:
    """Which side of K3's partition a failed call falls on.

    Transport: the provider or the network did not answer. That is K3's list (a
    timeout, a connection failure, a 429, a 5xx) plus the refusals that come
    before any model answers: a rejected key (401), a denied permission (403),
    and an SDK client that could not be built (a bare ``OpenAIError``, e.g. no
    key). Scoring those as content would let an account outage retire the
    identity, which K5 says a provider outage never does.

    Content: the request or the reply is at fault, e.g. a 400.
    """
    import openai

    if isinstance(
        exc,
        (
            TimeoutError, openai.APITimeoutError, openai.APIConnectionError,
            openai.RateLimitError, openai.AuthenticationError, openai.PermissionDeniedError,
        ),
    ):
        return FailureKind.TRANSPORT
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return FailureKind.TRANSPORT
    if isinstance(exc, openai.OpenAIError) and not isinstance(exc, openai.APIError):
        return FailureKind.TRANSPORT
    return FailureKind.CONTENT


def _call_meta(
    settings: AdjudicatorSettings, reply: ResponsesReply, latency_s: float,
) -> CallMeta:
    price = list_price(settings.model)
    usage = reply.usage
    return CallMeta(
        identity=adjudicator_identity(settings),
        resolved_model=reply.resolved_model,
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        cost_usd=(
            price.usd(usage.input_tokens, usage.output_tokens)
            if price is not None and usage is not None
            else None
        ),
        latency_s=latency_s,
    )


async def adjudicate(
    request: AdjudicationRequest,
    *,
    settings: AdjudicatorSettings,
    model_call: ResponsesCall,
    claim_check: ClaimCheck,
) -> AdjudicationResult:
    """One model call on *request*, its reply parsed into a verdict or a typed failure.

    Exactly one attempt: retrying is the caller's choice. Only a timeout or an
    SDK error becomes a failure; any other exception is a fault in the code,
    not in the model, and propagates, as a cancellation does.
    """
    import openai

    user_content = render_request(request, field_chars=settings.field_chars)
    started = time.monotonic()
    try:
        reply = await model_call(
            model=settings.model,
            instructions=INSTRUCTIONS,
            user_content=user_content,
            text_format=text_format(),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            timeout=settings.timeout_seconds,
            reasoning_effort=settings.reasoning_effort,
        )
    except (TimeoutError, openai.OpenAIError) as exc:
        return AdjudicationFailure(
            kind=classify_failure(exc), detail=f'{type(exc).__name__}: {exc}', meta=None,
        )
    meta = _call_meta(settings, reply, time.monotonic() - started)
    if reply.incomplete:
        return AdjudicationFailure(
            kind=FailureKind.CONTENT,
            detail=f'response incomplete ({reply.incomplete_reason})',
            meta=meta,
        )
    try:
        verdict = parse_verdict(reply.text or '', request, claim_check)
    except VerdictInvalid as exc:
        return AdjudicationFailure(kind=FailureKind.CONTENT, detail=str(exc), meta=meta)
    return Adjudication(verdict=verdict, meta=meta)
