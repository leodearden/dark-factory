"""Middle-band adjudication for ``add_memory`` write triage (task 3128, leaf gamma).

This module is leaf beta's ``write_triage._stub_judge`` replacement. Beta left
a slot — ``triage_write(..., judge=None)`` — and named that stub "LEAF GAMMA'S
REPLACEMENT POINT"; this is what fills it.

**It DETECTS, it does not adjudicate.** The judge classifies the RELATIONSHIP
between a submitted write and the candidates retrieval found — does this entry
restate one, add to one, contradict one, or none of them — and it never
decides which text is true. That boundary is decision D3 and it is empirical:
reify esc-5557/esc-5626 showed that adjudicating a apparent contradiction
needs code-reading and cross-checking that the synchronous ``add_memory``
write path cannot do and should not try. A ``contests`` verdict routes the
entry to the machinery that CAN adjudicate; it is a detection, not a ruling.
The band already named the target, so a verdict is only a word.

**It RAISES; it does not swallow.** ``triage_write`` already wraps the judge
call in an ``except`` arm that logs with ``exc_info``, records exactly one
fail-open on the storm counter, and returns ``stored``. That IS contract C1's
"judge error/timeout ⇒ stored + storm counter (INV-4)". A second fail-open
apparatus here would either double-count or, worse, hide the failure: every
write during a judge outage would look identical to "nothing matched", the
counter would never increment, and no storm escalation would ever fire. So
nothing in this module catches anything. The one deliberate exception is the
disabled/no-candidate early return in :func:`judge_write`, which answers
``stored`` WITHOUT raising — a decision is not an outage, and counting it
would guarantee a storm escalation describing a failure that is not happening
(the same boundary ``_stub_judge``'s own docstring draws).

IMPORT DIRECTION IS ONE-WAY AND DELIBERATE. This module imports from
``write_triage``; ``write_triage`` must NEVER import this one. The judge needs
beta's ``OUTCOME_*`` constants, so making the real judge ``triage_write``'s
default would be a circular import. ``server/tools.py`` is the single wiring
point that passes ``judge=judge_write`` into ``triage_write``. Do not "tidy"
that into a default here. Keeping ``_stub_judge`` as ``triage_write``'s own
default is also what keeps beta's judge-slot contract tests meaningful — they
inject their own fakes and must not accidentally exercise a live LLM.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from fused_memory.routing.json_extract import extract_json
from fused_memory.server.grouped_read import PARENT_ID_KEY

# The per-store-cosine reader, IMPORTED rather than re-implemented (INV-5) —
# the same import, for the same reason, that ``write_triage`` itself makes.
# A second copy does not raise when task 3658's
# ``relevance_score → metadata['store_score']`` move happens again: it scores
# every candidate as uncomparable, which empties the judge's slate and reads
# exactly like a genuinely novel corpus.
from fused_memory.server.near_duplicate_guard import _cosine_of
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)

if TYPE_CHECKING:
    from fused_memory.models.memory import MemoryResult


class JudgeOutputError(Exception):
    """The judge produced something that is not a verdict.

    Module-local so a caller can tell a malformed ANSWER apart from a
    transport failure. Both end at the same place — ``triage_write``'s
    fail-open arm — but they are different bugs with different fixes, and the
    logged exception type is what tells them apart in the record.
    """


#: The key the judge is instructed to put its answer under.
VERDICT_KEY = 'verdict'

#: The CLOSED 4-way judge vocabulary, mapped onto leaf beta's ack outcomes.
#:
#: Judge-facing words are VERBS about the submitted entry ("this entry
#: restates that one"); the ack words are PAST-TENSE facts about what triage
#: did with it ("it was attached as a restatement"). Keeping the two
#: vocabularies distinct is not decoration: it means a model echoing an ack
#: word it saw in a prompt is an out-of-vocabulary answer rather than an
#: accidental pass, and it keeps the wire contract renameable without
#: retraining the prompt.
#:
#: The VALUES are imported from ``write_triage``, never re-spelled (INV-5):
#: beta's ack contract has exactly one home, and the tests derive this
#: mapping's value set from ``TRIAGE_OUTCOMES`` so a fifth outcome added there
#: fails loudly here — at the one place that would need a fifth judge word and
#: a fifth attach kind — instead of arriving forever as a counted fail-open.
JUDGE_VERDICTS: dict[str, str] = {
    'distinct': OUTCOME_STORED,
    'restates': OUTCOME_RESTATED,
    'amends': OUTCOME_AMENDED,
    'contests': OUTCOME_CONTESTED,
}

#: How much of a rejected payload is quoted back in the raised message. The
#: message reaches a log line via ``triage_write``'s ``exc_info``, and a model
#: that answered with its entire context window would otherwise put all of it
#: there. Long enough to see the actual answer, short enough to stay a log
#: line.
_REJECTED_PAYLOAD_CHARS = 400


def _reject(reason: str, payload: object) -> JudgeOutputError:
    """Build the rejection carrying a bounded quote of what arrived."""
    quoted = repr(payload)
    if len(quoted) > _REJECTED_PAYLOAD_CHARS:
        quoted = quoted[:_REJECTED_PAYLOAD_CHARS] + '…(truncated)'
    return JudgeOutputError(f'judge output rejected ({reason}): {quoted}')


def parse_judge_verdict(raw: str) -> str:
    """Map one raw judge response onto a member of ``TRIAGE_OUTCOMES``.

    Accepts a bare JSON object, a fenced ```json block, and JSON embedded in
    surrounding prose — via :func:`fused_memory.routing.json_extract.extract_json`,
    the repo's existing fenced-code/brace-counting extractor, rather than a
    fourth private JSON scraper.

    RAISES :class:`JudgeOutputError` for everything else: an unrecognised
    verdict word, a missing or non-string verdict key, a non-object payload,
    empty or whitespace-only text, and prose containing no JSON at all.

    Raising rather than defaulting is the load-bearing part. ``triage_write``
    counts a fail-open for a judge that raises or answers out-of-vocabulary;
    a parser that quietly returned ``stored`` on a malformed answer would make
    a broken judge and a healthy one reporting a novel corpus produce
    byte-identical acks, byte-identical counters, and no alarm — the exact
    silent degradation INV-4 exists to prevent.

    Note that an ACK word — any member of ``TRIAGE_OUTCOMES`` — is
    deliberately NOT accepted here. The judge is instructed in the judge
    vocabulary, so a model answering in ack words is answering a question it
    was not asked, and treating that as a pass would hide a prompt that had
    drifted out of sync with this parser.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise _reject('empty or non-text response body', raw)

    json_str = extract_json(raw)
    if not json_str:
        raise _reject('no JSON object in response', raw)

    try:
        payload = json.loads(json_str)
    except ValueError as exc:
        raise _reject(f'malformed JSON ({exc})', json_str) from exc

    if not isinstance(payload, dict):
        raise _reject('JSON payload is not an object', payload)

    word = payload.get(VERDICT_KEY)
    if not isinstance(word, str):
        raise _reject(f'no string {VERDICT_KEY!r} key', payload)

    verdict = JUDGE_VERDICTS.get(word.strip().lower())
    if verdict is None:
        raise _reject(
            f'{word!r} is not one of {sorted(JUDGE_VERDICTS)}', payload,
        )
    return verdict


# --- candidate selection ----------------------------------------------------

#: How many candidates the judge is shown when nothing configures otherwise —
#: the top of PRD C1's "top 3–5".
_DEFAULT_JUDGE_CANDIDATE_COUNT = 5

#: Per-field character budget for the rendered prompt. The calibration fixture
#: holds a ~9k-character canonical, and C1 sizes the judge call at roughly
#: 2.5k tokens; five untrimmed candidates would blow through that by an order
#: of magnitude on exactly the consolidated topics where triage matters most.
_FIELD_CHARS = 1_200

#: Appended to any field this module cut. The marker is not decoration: a
#: silent truncation hands the model a severed sentence to read as the whole
#: record, and "this text continues" is information the verdict depends on.
_ELIDED_MARKER = '…[elided]'


def _elide(text: object) -> str:
    """*text* as a string, bounded by :data:`_FIELD_CHARS` and marked if cut."""
    body = text if isinstance(text, str) else str(text or '')
    if len(body) <= _FIELD_CHARS:
        return body
    return body[:_FIELD_CHARS] + _ELIDED_MARKER


def select_judge_candidates(
    results: Any,
    n: int,
    *,
    canonical_id: str | None,
) -> list[MemoryResult]:
    """The top *n* comparable candidates, with the band's winner guaranteed in.

    *results* is whatever ``triage_write`` retrieved — the un-transformed
    ``SearchResults`` object, iterated as given. Trimming to PRD C1's "top
    3–5" happens HERE rather than at the call site, so the object keeps its
    ``degraded``/``failed_stores`` attributes all the way to the one place
    that reads them.

    Ordered by DESCENDING per-store cosine, read through
    ``near_duplicate_guard._cosine_of`` — the SAME reader ``decide_band``
    uses, imported rather than re-implemented (INV-5). A second copy is
    precisely how task 3658's ``relevance_score → metadata['store_score']``
    move would go wrong again, and it would not raise: it would score every
    candidate as uncomparable, which reads as a novel corpus.

    A candidate with no numeric cosine is DROPPED. A topic-anchored pin
    (``services/topic_anchor.py``) deliberately carries no ``store_score``,
    and ``decide_band`` already drops it because it can never clear a
    threshold; dropping it here is the adjacent point — there is no measured
    similarity to show the model, so the slot is better spent on a record that
    can actually be compared.

    *canonical_id* — the band's winner, hoisted to a parent where the winner
    was itself a child — is guaranteed present in the returned set, evicting
    the weakest candidate if it would otherwise fall outside the top *n*. A
    judge shown a set that excludes the attach target is answering about a
    different memory than the one the attach will touch. When the hoisted
    parent is not itself in the result set, the CHILD that carried the
    evidence is kept instead, because dropping both would leave no view of
    the match at all.

    Returns ``[]`` for an empty or wholly uncomparable slate, and raises
    nothing: an empty candidate set is a decision (:func:`judge_write` answers
    ``stored`` without calling out), not a failure.
    """
    scored: list[tuple[float, MemoryResult]] = []
    for result in results or ():
        cosine = _cosine_of(result)
        if cosine is not None:
            scored.append((cosine, result))
    if not scored:
        return []

    scored.sort(key=lambda pair: pair[0], reverse=True)
    ordered = [result for _, result in scored]
    if n <= 0:
        n = _DEFAULT_JUDGE_CANDIDATE_COUNT
    selected = ordered[:n]

    if canonical_id is not None and all(r.id != canonical_id for r in selected):
        # The winner is outside the window. Either it is further down the
        # slate (take it, evicting the weakest), or it is a HOISTED parent id
        # that never appeared as a result of its own — in which case the child
        # whose `parent_id` points at it is the record carrying the evidence.
        winner = next(
            (r for r in ordered if r.id == canonical_id),
            None,
        ) or next(
            (
                r
                for r in ordered
                if (r.metadata or {}).get(PARENT_ID_KEY) == canonical_id
            ),
            None,
        )
        if winner is not None and winner not in selected:
            selected = [*selected[: max(n - 1, 0)], winner]
    return selected


# --- prompt -----------------------------------------------------------------

#: The judge's standing instructions. D3 lives HERE, in the model's own
#: prompt, not only in a docstring: a model told merely to "classify" will
#: happily decide which of two contradictory memories is true, and reify
#: esc-5557/esc-5626 showed that adjudicating an apparent contradiction needs
#: code-reading and cross-checking the synchronous ``add_memory`` write path
#: cannot do. So the instruction says what ``contests`` MEANS — a detection
#: that routes the entry onward — and says the judge is not deciding truth.
JUDGE_SYSTEM_PROMPT = f"""\
You classify the RELATIONSHIP between a new memory entry and a small set of \
existing entries retrieved as its closest matches. You do not decide which \
entry is correct, and you do not merge, rewrite or rank them.

Answer with exactly one of these four words:

- "distinct" — the new entry is about something the candidates do not cover. \
Overlapping vocabulary is not enough; the new entry has to be making \
substantially the same claim as a candidate to be anything else.
- "restates" — the new entry asserts what a candidate already asserts, adding \
nothing new. A paraphrase restates.
- "amends" — the new entry asserts what a candidate asserts AND adds \
something the candidate does not have: a detail, a scope, a later \
observation, a correction of degree.
- "contests" — the new entry asserts something that CANNOT be true at the \
same time as a candidate. Use this only for a genuine incompatibility, not \
for a difference in emphasis, scope, or point in time — two entries \
describing different situations, or the same situation at different times, \
are not in conflict. You are DETECTING a contradiction so a human or a \
downstream gate can adjudicate it; you are NOT deciding which side is true, \
and nothing you say here deletes or edits anything.

When more than one word fits, prefer the earlier one in that list: \
"distinct" over "restates", "restates" over "amends", "amends" over \
"contests".

Reply with a bare JSON object and nothing else:

{{"{VERDICT_KEY}": "<one of: {', '.join(JUDGE_VERDICTS)}>"}}\
"""


def build_judge_prompt(content: str, candidates: list[MemoryResult]) -> str:
    """Render the user-side prompt: the new entry, then the candidates.

    CONTENT ONLY. No metadata is interpolated — not the agent_id, not the
    project_id, not a task id, not a source path. That is PRD C1's "no repo or
    task context reaches the judge", and rendering no metadata at all is what
    makes it a structural property rather than an incidental one: there is no
    field list to keep in sync and no leak to notice later. Candidate ids ARE
    rendered, because the model must be able to say which candidate it means —
    they are opaque memory uuids, not context.

    Every field is bounded by :data:`_FIELD_CHARS` and marked with
    :data:`_ELIDED_MARKER` when cut, so the call stays near C1's ~2.5k-token
    budget regardless of a pathological canonical.

    Pure, synchronous and total: it renders for an empty candidate list too,
    though :func:`judge_write` never calls it with one.
    """
    lines = [
        'NEW ENTRY:',
        _elide(content),
        '',
        'EXISTING CANDIDATES:',
    ]
    for candidate in candidates:
        lines.append(f'- id: {candidate.id}')
        lines.append(f'  text: {_elide(candidate.content)}')
    if not candidates:
        lines.append('(none)')
    lines.append('')
    # The closed vocabulary and the output shape are RESTATED here, next to
    # the data, even though JUDGE_SYSTEM_PROMPT already carries both. Two
    # reasons, neither cosmetic. (1) The two provider arms deliver the system
    # prompt differently — openai as messages[0], anthropic via `system=` —
    # so a wiring mistake on either arm could drop it entirely; restating the
    # contract in the user turn means the worst case is a weaker prompt, not
    # a model answering in a vocabulary parse_judge_verdict rejects on every
    # single write. (2) An out-of-vocabulary answer is a counted fail-open
    # (write_triage.py:839), so vocabulary drift does not surface as a bad
    # verdict — it surfaces as a storm escalation describing an outage.
    lines.append(
        'Classify the relationship between NEW ENTRY and the candidates. '
        f'Answer with exactly one of: {", ".join(JUDGE_VERDICTS)}.',
    )
    lines.append(
        'Reply with a bare JSON object and nothing else: '
        f'{{"{VERDICT_KEY}": "<one of those four words>"}}',
    )
    return '\n'.join(lines)


# --- defensive config resolvers ---------------------------------------------
#
# Same shape as ``write_triage``'s own: ``getattr`` at every hop, so a missing
# service, config, section or leaf reads as the default rather than raising
# into a write path contract C1 forbids from failing. Nothing is captured at
# import or construction — every value is read LIVE per call, which is the
# precondition that makes the green-tier ``RELOADABLE_FIELDS`` registration
# real rather than restart-only in disguise (``apply_reload`` mutates the
# shared config object in place, and a captured value cannot observe that).

#: The judge's own kill switch defaults ON, asymmetrically with
#: ``write_triage.enabled``, which ships OFF. The judge is structurally INERT
#: while triage is disabled — no triage code executes at all — so this costs
#: nothing on today's shipped config. Default-OFF would be the footgun: at the
#: task-3169 flip the operator would turn ``enabled`` on, silently get stub
#: behaviour, and read the resulting all-``stored`` ack stream as evidence
#: that the corpus is novel. Its real purpose is stopping an in-flight JUDGE
#: incident (spend, latency, a bad model) while leaving the deterministic
#: bands running — a strictly finer lever than ``enabled``.
_DEFAULT_JUDGE_ENABLED = True

#: The providers with an implemented arm in :func:`_call_llm`.
_KNOWN_PROVIDERS = ('openai', 'anthropic')

#: Provider and model defaults, matching ``LLMConfig``'s own. "haiku-class" in
#: PRD C1 is a cost/size class, not a vendor pin: measured on this deployment
#: ``ANTHROPIC_API_KEY`` is unset (CLAUDE.md: agents use OAuth) while
#: ``OPENAI_API_KEY`` is set and demonstrably works — leaf alpha's committed
#: calibration report is the product of live OpenAI calls from this same
#: checkout. The anthropic arm is implemented and selectable by config for a
#: deployment that has the key.
_DEFAULT_JUDGE_PROVIDER = 'openai'
_DEFAULT_JUDGE_MODEL = 'gpt-4o-mini'

#: No LLM call anywhere in fused-memory sets a timeout today, and the openai
#: SDK default is 600 seconds. On the SYNCHRONOUS ``add_memory`` write path
#: that is a wedge, not a degradation: the caller waits ten minutes for a
#: write C1 promises never to block. Ten seconds is generous for a ~2.5k-token
#: single-turn classification and bounded enough that a hung provider costs
#: one slow write rather than a hung server.
_DEFAULT_JUDGE_TIMEOUT_SECONDS = 10.0


def _judge_attr(memory_service: Any, attr: str) -> Any:
    """Navigate ``memory_service.config.write_triage.<attr>`` defensively.

    A private twin of ``write_triage._write_triage_attr`` rather than an
    import of it: that name is module-private to leaf beta, and reaching
    across for it would couple two modules through a leading underscore. The
    navigation is four lines; the coupling would be forever.
    """
    config = getattr(memory_service, 'config', None)
    write_triage = getattr(config, 'write_triage', None)
    return getattr(write_triage, attr, None)


def _llm_attr(memory_service: Any, attr: str) -> Any:
    """Navigate ``memory_service.config.llm.<attr>`` defensively."""
    config = getattr(memory_service, 'config', None)
    llm = getattr(config, 'llm', None)
    return getattr(llm, attr, None)


def resolve_judge_enabled(memory_service: Any) -> bool:
    """The judge's kill switch. Defaults :data:`_DEFAULT_JUDGE_ENABLED` (True).

    ``isinstance(bool)`` only, deliberately, and in BOTH directions. A truthy
    ``1`` must not enable by accident — the usual reason — but neither may a
    falsy ``0`` DISABLE by accident, because a silently-stubbed judge produces
    exactly the all-``stored`` ack stream that default-True exists to prevent
    an operator from misreading.
    """
    value = _judge_attr(memory_service, 'judge_enabled')
    if isinstance(value, bool):
        return value
    return _DEFAULT_JUDGE_ENABLED


def resolve_judge_provider(memory_service: Any) -> str:
    """``write_triage.judge_provider``, else ``llm.provider``, else the default.

    Inheriting rather than defaulting means the judge follows whatever model
    the deployment already trusts for ``add_memory`` auto-classification,
    without a second knob to keep in sync.

    An unrecognised provider string FALLS BACK rather than raising: this
    resolver runs on the write path, where C1 forbids raising. The
    unresolvable-provider raise belongs in the fan-out
    (:func:`_call_llm`) — which sits INSIDE ``triage_write``'s fail-open arm,
    so it surfaces as a counted fail-open instead of an errored write.
    """
    pinned = _judge_attr(memory_service, 'judge_provider')
    if isinstance(pinned, str) and pinned in _KNOWN_PROVIDERS:
        return pinned
    inherited = _llm_attr(memory_service, 'provider')
    if isinstance(inherited, str) and inherited in _KNOWN_PROVIDERS:
        return inherited
    return _DEFAULT_JUDGE_PROVIDER


def resolve_judge_model(memory_service: Any) -> str:
    """``write_triage.judge_model``, else ``llm.model``, else the default.

    A non-string or blank model name falls back: an empty model reaches the
    SDK as a 404 on every single write, i.e. a total judge outage caused by a
    config typo.
    """
    for value in (
        _judge_attr(memory_service, 'judge_model'),
        _llm_attr(memory_service, 'model'),
    ):
        if isinstance(value, str) and value.strip():
            return value
    return _DEFAULT_JUDGE_MODEL


def resolve_judge_timeout(memory_service: Any) -> float:
    """The per-call budget, in seconds. Positive numbers only.

    A zero or negative timeout would fail EVERY call, turning a config typo
    into a permanent storm escalation whose stated cause — a broken judge — is
    not what is actually wrong. ``bool`` is excluded for the usual reason:
    ``True`` would resolve to a one-second budget with nothing to explain it.
    """
    value = _judge_attr(memory_service, 'judge_timeout_seconds')
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return _DEFAULT_JUDGE_TIMEOUT_SECONDS


def resolve_judge_candidate_count(memory_service: Any) -> int:
    """How many candidates reach the prompt. Positive ``int`` only.

    A zero count would empty the slate on every write, and
    :func:`judge_write`'s no-candidate early return would then answer
    ``stored`` every time — triage silently reduced to its below-``t_low``
    behaviour, with no error anywhere to say so.
    """
    value = _judge_attr(memory_service, 'judge_candidate_count')
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return _DEFAULT_JUDGE_CANDIDATE_COUNT


# --- the LLM call ------------------------------------------------------------

#: Output cap. The answer is a four-word closed vocabulary inside a one-key
#: JSON object, so this is generous by an order of magnitude — sized to leave
#: room for a model that adds a `reasoning` key (the parser ignores extra
#: keys) without leaving room for an essay billed per token on every
#: middle-band write.
_JUDGE_MAX_TOKENS = 64


def _provider_credentials(memory_service: Any, provider: str) -> dict[str, Any]:
    """``api_key``/``base_url`` for *provider*, defensively, possibly empty.

    An empty dict is a FIRST-CLASS result, not a failure: both SDKs fall back
    to their standard environment variables, which is how this deployment is
    actually configured (``OPENAI_API_KEY`` in the shell, nothing in
    config.yaml). Reading the config section is for a deployment that pins a
    key or points at an OpenAI-compatible local endpoint.
    """
    config = getattr(memory_service, 'config', None)
    llm = getattr(config, 'llm', None)
    providers = getattr(llm, 'providers', None)
    section = getattr(providers, provider, None)
    creds: dict[str, Any] = {}
    api_key = getattr(section, 'api_key', None)
    if isinstance(api_key, str) and api_key:
        creds['api_key'] = api_key
    api_url = getattr(section, 'api_url', None)
    if isinstance(api_url, str) and api_url:
        creds['base_url'] = api_url
    return creds


async def _call_llm(
    *,
    provider: str,
    model: str,
    prompt: str,
    memory_service: Any,
    timeout: float,
) -> str:
    """One single-turn call to *provider*, returning the raw response text.

    Mirrors ``reconciliation/judge.py::_call_llm``'s two-arm fan-out at
    write-path scale: ``temperature=0`` (this is a classification, not a
    generation), a small :data:`_JUDGE_MAX_TOKENS`, and — on the openai arm —
    ``response_format={'type': 'json_object'}`` so the happy path is the
    parser's happy path. The anthropic arm passes the system prompt via
    ``system=`` because Anthropic has no system ROLE; a system message would
    arrive as an ordinary user turn.

    The client is constructed PER CALL and deliberately not cached on a module
    global. ``add_memory`` is served by one long-lived server process, and a
    cached client keyed to a config that hot-reloads would pin a stale
    model/api_url past a reload that the operator was told had applied —
    silently converting a green-tier knob into a restart-only one. Client
    construction is cheap relative to the round-trip it precedes.

    NO ``try``/``except`` ANYWHERE. Every failure propagates to
    ``triage_write``'s ``except`` arm (write_triage.py:835), which logs with
    ``exc_info``, counts exactly one fail-open, and returns ``stored``.

    An unrecognised *provider* RAISES rather than falling back to a default.
    That is the opposite of :func:`resolve_judge_provider`'s behaviour, and
    deliberately so: the resolver runs on the write path where C1 forbids
    raising, whereas this raise lands INSIDE the fail-open arm. Silently
    picking an arm here would bill an account the operator never chose.
    """
    if provider == 'openai':
        import openai  # noqa: PLC0415 — per-call import, matching judge.py

        client = openai.AsyncOpenAI(**_provider_credentials(memory_service, provider))
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.0,
                max_tokens=_JUDGE_MAX_TOKENS,
                response_format={'type': 'json_object'},
            ),
            timeout=timeout,
        )
        return response.choices[0].message.content or ''

    if provider == 'anthropic':
        import anthropic  # noqa: PLC0415 — per-call import, matching judge.py

        client = anthropic.AsyncAnthropic(
            **_provider_credentials(memory_service, provider),
        )
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=_JUDGE_MAX_TOKENS,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
            ),
            timeout=timeout,
        )
        # First TEXT block, not first block: a leading thinking/tool_use block
        # must not be read as the answer.
        text_blocks = [b for b in response.content if b.type == 'text']
        return text_blocks[0].text if text_blocks else ''

    raise ValueError(
        f'unknown judge provider {provider!r}; implemented arms are '
        f'{list(_KNOWN_PROVIDERS)}',
    )


async def judge_write(
    *,
    memory_service: Any,
    content: str,
    project_id: str,
    decision: Any,
    candidates: Any = (),
) -> str:
    """Adjudicate one middle-band write. Returns a member of ``TRIAGE_OUTCOMES``.

    This is what ``tools.py`` passes as ``triage_write(..., judge=...)``,
    replacing leaf beta's ``_stub_judge``. The signature is beta's, plus the
    ``candidates`` keyword beta's slot did not carry: PRD C1 requires the
    judge to see the new entry AND its top 3–5 candidates, and a judge shown
    only a canonical ID cannot classify anything.

    Flow: resolve config LIVE → return ``stored`` early if disabled or if the
    slate selects to empty → build the prompt → call the provider under
    ``asyncio.wait_for`` → parse.

    RAISES on every failure — transport, timeout, unparseable output,
    out-of-vocabulary verdict — and catches nothing. ``triage_write`` owns the
    fail-open apparatus (INV-4): its ``except`` arm logs with ``exc_info``,
    counts exactly one fail-open, and returns ``stored``. A second apparatus
    here would double-count or hide the failure, and hiding it is the
    catastrophic direction: every write during a judge outage would look
    identical to "nothing matched", the counter would never increment, and no
    storm escalation would ever fire.

    The two early returns are the deliberate exceptions, and they are
    DECISIONS rather than failures. ``judge_enabled: false`` is an operator
    stopping the LLM arm on purpose; an empty candidate set is a write with
    nothing to be compared against. Routing either through the counter would
    guarantee a storm escalation describing an outage that is not happening,
    which trains an operator to ignore the alarm that exists to catch a real
    one — the same boundary ``_stub_judge``'s own docstring draws.

    PROVIDER. ``judge_provider``/``judge_model`` default to None and INHERIT
    ``llm.provider``/``llm.model``, which ship as ``openai``/``gpt-4o-mini``.
    That default is evidence-based, not preference: measured on this
    deployment ``ANTHROPIC_API_KEY`` is unset (CLAUDE.md — agents use OAuth)
    while ``OPENAI_API_KEY`` is set and demonstrably works, since leaf alpha's
    committed calibration report is the product of live OpenAI calls from this
    same checkout. The anthropic arm is implemented and selectable by config
    for a deployment that has the key; PRD C1's "haiku-class" is a cost/size
    class, not a vendor pin.
    """
    if not resolve_judge_enabled(memory_service):
        return OUTCOME_STORED

    selected = select_judge_candidates(
        candidates,
        resolve_judge_candidate_count(memory_service),
        canonical_id=getattr(decision, 'canonical_id', None),
    )
    if not selected:
        return OUTCOME_STORED

    raw = await _call_llm(
        provider=resolve_judge_provider(memory_service),
        model=resolve_judge_model(memory_service),
        prompt=build_judge_prompt(content, selected),
        memory_service=memory_service,
        timeout=resolve_judge_timeout(memory_service),
    )
    return parse_judge_verdict(raw)
