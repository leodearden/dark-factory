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
The verdict names the candidate it is about; the band only names the record
the slate is guaranteed to contain.

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
import logging
from typing import TYPE_CHECKING, Any, NamedTuple, TypeGuard, get_args

from fused_memory.config.schema import JudgeReasoningEffort
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
    JudgeUsage,
    TriageJudgeVerdict,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from fused_memory.models.memory import MemoryResult

logger = logging.getLogger(__name__)


class JudgeOutputError(Exception):
    """The judge produced something that is not a verdict.

    Module-local so a caller can tell a malformed ANSWER apart from a
    transport failure. Both end at the same place — ``triage_write``'s
    fail-open arm — but they are different bugs with different fixes, and the
    logged exception type is what tells them apart in the record.
    """


#: The key the judge is instructed to put its answer under.
VERDICT_KEY = 'verdict'

#: The key naming the candidate an attach verdict is about.
CANDIDATE_ID_KEY = 'candidate_id'

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


class JudgeExemplar(NamedTuple):
    """One worked example of the vocabulary: a pair, and the word for it.

    Three fields rather than one pre-formatted line, because the formatting
    belongs to the renderer: held as data, the set can be checked for
    vocabulary closure and verdict coverage instead of grepped for.
    """

    entry: str
    candidate: str
    verdict: str


#: A worked example per verdict, rendered into :data:`JUDGE_SYSTEM_PROMPT`.
#:
#: WHY THESE EXIST. The 2026-08-27 calibration run answered `stored` on 31 of
#: 75 duplicates with the correct canonical sitting in the slate, while the
#: distractor control scored 18/18 — so the judge was not attaching
#: indiscriminately, it was systematically over-answering "distinct". A
#: vocabulary word the model has never seen USED is the one it under-produces,
#: which is why coverage of all four is an asserted invariant and not a
#: stylistic goal.
#:
#: WHY THIS PARTICULAR SET. One candidate, four entries. Holding the candidate
#: fixed isolates the only variable that should decide the answer — the
#: RELATIONSHIP — and makes the two failure directions visible side by side:
#:
#: * `restates` and `amends` share almost no surface vocabulary with the
#:   candidate and still attach, which is the measured defect stated as an
#:   example (the judge was demanding lexical overlap before it would attach);
#: * `distinct` repeats the candidate's own words verbatim and still does NOT
#:   attach, so the lesson reads as "the claim decides" rather than as the
#:   cruder "attach more readily" — the latter would cost the distractor
#:   control, which is exactly what that control is there to report.
#:
#: The `restates`/`amends` pair differs only by a trailing novel fragment, so
#: the discrimination between them is shown on otherwise identical material.
#:
#: Declaration order mirrors the instruction the prompt states — find the
#: candidate that shares the entry's core claim first, answer `distinct` only
#: when none does — not the vocabulary's alphabet.
#:
#: SYNTHETIC AND OFF-CORPUS BY CONSTRUCTION. Nothing here is drawn from
#: ``tests/fixtures/write_triage_calibration.jsonl``; drawing from it would be
#: training on the test set, and the judge suite asserts the disjointness
#: rather than trusting this note. Nothing here interpolates an id, a
#: category, an agent or any repo context either: PRD C1 keeps all of that out
#: of the judge, and a prompt that renders no metadata AT ALL is what makes
#: that structural.
_EXEMPLAR_CANDIDATE = 'The greenhouse thermostat is calibrated in Fahrenheit.'

JUDGE_EXEMPLARS: tuple[JudgeExemplar, ...] = (
    JudgeExemplar(
        entry='Setpoints for the glasshouse heater are entered in degrees F.',
        candidate=_EXEMPLAR_CANDIDATE,
        verdict='restates',
    ),
    JudgeExemplar(
        entry=(
            'Glasshouse setpoints are entered in degrees F, and the display '
            'rounds to the nearest whole degree.'
        ),
        candidate=_EXEMPLAR_CANDIDATE,
        verdict='amends',
    ),
    JudgeExemplar(
        entry='The greenhouse thermostat was replaced in March after its relay failed.',
        candidate=_EXEMPLAR_CANDIDATE,
        verdict='distinct',
    ),
    JudgeExemplar(
        entry='The greenhouse thermostat reads only in Celsius; it has no Fahrenheit mode.',
        candidate=_EXEMPLAR_CANDIDATE,
        verdict='contests',
    ),
)

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


def parse_judge_verdict(raw: str, slate_ids: Collection[str]) -> TriageJudgeVerdict:
    """Map one raw judge response onto a :class:`TriageJudgeVerdict`.

    Accepts a bare JSON object, a fenced ```json block, and JSON embedded in
    surrounding prose — via :func:`fused_memory.routing.json_extract.extract_json`,
    the repo's existing fenced-code/brace-counting extractor, rather than a
    fourth private JSON scraper.

    An attach verdict must name its candidate under :data:`CANDIDATE_ID_KEY`,
    and that id must be one of *slate_ids* — the slate the model was actually
    SHOWN. A record retrieved but trimmed off the slate was never seen, so
    naming it is a hallucination, not a choice. ``distinct`` names none.

    RAISES :class:`JudgeOutputError` for everything else: an unrecognised
    verdict word, a missing or non-string verdict key, a non-object payload,
    empty or whitespace-only text, prose containing no JSON at all, an attach
    verdict naming no candidate or one off the slate, and ``distinct`` naming
    one.

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

    outcome = JUDGE_VERDICTS.get(word.strip().lower())
    if outcome is None:
        raise _reject(
            f'{word!r} is not one of {sorted(JUDGE_VERDICTS)}', payload,
        )
    return TriageJudgeVerdict(outcome, _named_candidate(payload, outcome, slate_ids))


def _named_candidate(
    payload: dict[str, Any], outcome: str, slate_ids: Collection[str],
) -> str | None:
    """The slate id *payload* names, or ``None`` for a verdict that attaches nothing."""
    candidate_id = payload.get(CANDIDATE_ID_KEY)
    if outcome == OUTCOME_STORED:
        if candidate_id is not None:
            raise _reject(
                f'a verdict that attaches nothing names {candidate_id!r}', payload,
            )
        return None
    if not isinstance(candidate_id, str) or not candidate_id:
        raise _reject(
            f'an attach verdict must name its candidate as a non-empty string '
            f'{CANDIDATE_ID_KEY!r}',
            payload,
        )
    if candidate_id not in slate_ids:
        raise _reject(
            f'{candidate_id!r} is not a candidate on the rendered slate '
            f'{sorted(slate_ids)}',
            payload,
        )
    return candidate_id


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

#: C1's ~2.5k-token call budget, expressed in the units this module can
#: actually count: 2_500 tokens at the conventional 4 chars/token.
#:
#: WHAT IT BOUNDS is the WHOLE call — :data:`JUDGE_SYSTEM_PROMPT` plus a
#: worst-case :func:`build_judge_prompt` render, meaning
#: :data:`_DEFAULT_JUDGE_CANDIDATE_COUNT` candidates and a new entry with
#: every field over :data:`_FIELD_CHARS`, each candidate carrying the 36-char
#: uuid a real record has. Not the system prompt alone: the two halves are
#: summed on every request, so budgeting either in isolation budgets nothing.
#: And not a construction :func:`judge_write` never makes, for the same reason.
#:
#: WHY IT IS A CONSTANT rather than a literal in the test that checks it. The
#: system prompt is the half that grows — a vocabulary word, a worked example,
#: a decision rule all land there — so the ceiling needs a home next to the
#: rationale for its value. Raising it is then an edit to the thing being
#: budgeted, made where C1 is cited, rather than a number quietly relaxed in a
#: test until it stops failing.
#:
#: THE TOLERANCE TERM IS NOT PADDING. Both inputs to the headline figure are
#: approximations: C1 writes "~2.5k", and 4 chars/token is a convention, NOT a
#: measurement — this package does not depend on a tokenizer, so nothing here
#: has counted the real tokens and no claim is made about them. Multiplying
#: two approximations and then treating the product as a hard wall is a false
#: precision, and it bites asymmetrically: a rendering 0.5% over would read as
#: a C1 violation when it is inside "~2.5k" on any reading.
#:
#: The tolerance is SIZED, not chosen for comfort — but what has to stay
#: small is the SLACK, budget minus the measured worst case, which is what a
#: future addition could spend without anyone having to come here. That slack
#: is 58 chars. The four worked examples presently rendered cost 157 to 192
#: chars apiece including the blank line between them, so even the cheapest
#: fifth one does not fit and its author has to either make room or make the
#: case here. A ceiling that admitted another example would have stopped
#: bounding anything.
#:
#: Measured at task 5794, PRODUCTION-SHAPED and with the elision marker
#: counted: the worst case is 10_342 chars — system 2_468, plus a 7_874-char
#: render of six fields each OVER `_FIELD_CHARS`. The ids are not slop: every
#: stored record's id is a 36-char uuid — all 104 in
#: ``tests/fixtures/write_triage_calibration.jsonl`` are — and
#: :func:`build_judge_prompt` renders ``- id: {candidate.id}`` UN-elided, so a
#: full slate costs 155 chars more than 5-char stand-in ids suggest.
#:
#: Note "over" `_FIELD_CHARS`, not "at": `_elide` returns a field of exactly
#: `_FIELD_CHARS` unchanged and cuts a longer one to `_FIELD_CHARS` plus
#: `_ELIDED_MARKER`, so the widest render is 9 chars per field — 54 across the
#: six — wider than a slate built at the cap.
_PROMPT_CHAR_BUDGET = 2_500 * 4 + 400


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
    the weakest candidate if it would otherwise fall outside the top *n*. It
    is the strongest evidence retrieval found, so the model is always SHOWN
    it, even though the verdict may name a different candidate. When the
    hoisted parent is not itself in the result set, the CHILD that carried
    the evidence is kept instead, because dropping both would leave no view
    of the match at all.

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

def _render_exemplars(exemplars: Sequence[JudgeExemplar]) -> str:
    """The EXAMPLES section of the system prompt: one block per exemplar.

    A FUNCTION OF THE TUPLE ALONE — pure, total, and walking the sequence in
    the order given. That is what makes the system prompt byte-identical on
    every import, which is in turn the precondition for an eval run being
    reproducible: this file has been bitten once by an iteration order moving
    between two processes.

    The formatting lives here rather than in :data:`JUDGE_EXEMPLARS` so the
    data stays checkable — vocabulary closure and verdict coverage are
    assertions about fields, not greps over a rendered blob — and so two
    exemplars cannot disagree about their own layout.
    """
    return '\n\n'.join(
        f'new entry: {exemplar.entry}\n'
        f'candidate: {exemplar.candidate}\n'
        f'answer: {exemplar.verdict}'
        for exemplar in exemplars
    )


#: The one spelling of the judge's output contract, rendered into BOTH halves
#: of the call so the system prompt and the user turn cannot disagree about it.
JUDGE_REPLY_SHAPE = json.dumps({
    VERDICT_KEY: f'<one of: {", ".join(JUDGE_VERDICTS)}>',
    CANDIDATE_ID_KEY: '<the id of the candidate your verdict is about; null for distinct>',
})

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

- "distinct" — no candidate makes the same core claim as the new entry. \
Shared wording alone neither makes a match nor rules one out.
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

Find the candidate whose core claim the new entry shares (or, for \
"contests", contradicts), answer about THAT candidate, and name it by its \
id: the verdict is filed against the candidate you name and no other. \
Answer "distinct", naming none, only when no candidate qualifies. Between \
"amends" and "contests", prefer "amends" — a genuine incompatibility is a \
last resort, not a default reading.

Worked examples:

{_render_exemplars(JUDGE_EXEMPLARS)}

Reply with a bare JSON object and nothing else:

{JUDGE_REPLY_SHAPE}\
"""


def build_judge_prompt(content: str, candidates: list[MemoryResult]) -> str:
    """Render the user-side prompt: the new entry, then the candidates.

    CONTENT ONLY. No metadata is interpolated — not the agent_id, not the
    project_id, not a task id, not a source path. That is PRD C1's "no repo or
    task context reaches the judge", and rendering no metadata at all is what
    makes it a structural property rather than an incidental one: there is no
    field list to keep in sync and no leak to notice later. Candidate ids ARE
    rendered, because the model NAMES the candidate its verdict is about by
    id — they are opaque memory uuids, not context.

    No candidate is marked or singled out. The verdict carries its own id, so
    neither slate position nor the band's winner steers the answer.

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
    # reasons, neither cosmetic. (1) Each provider arm delivers the system
    # prompt its own way — `instructions=` on the Responses API, messages[0]
    # on a chat endpoint, `system=` on anthropic — so a wiring mistake on any
    # arm could drop it entirely; restating the
    # contract in the user turn means the worst case is a weaker prompt, not
    # a model answering in a vocabulary parse_judge_verdict rejects on every
    # single write. (2) An out-of-vocabulary answer is a counted fail-open
    # (write_triage.py::triage_write), so vocabulary drift does not surface as
    # a bad verdict — it surfaces as a storm escalation describing an outage.
    lines.append(
        'Classify the relationship between NEW ENTRY and the candidates. '
        f'Answer with exactly one of: {", ".join(JUDGE_VERDICTS)}.',
    )
    lines.append(f'Reply with a bare JSON object and nothing else: {JUDGE_REPLY_SHAPE}')
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

#: The haiku-class default for each implemented arm, used when the judge's
#: provider is pinned AWAY from ``llm.provider``. Inheriting ``llm.model``
#: across a provider boundary is the exact 404-on-every-write outage
#: :func:`resolve_judge_model` exists to prevent: the documented configuration
#: ``judge_provider: anthropic`` + ``judge_model: null`` on an
#: ``llm.provider: openai`` deployment would otherwise post ``gpt-4o-mini`` to
#: ``anthropic.messages.create``, and the resulting total judge outage would be
#: counted as a fail-open storm whose stated cause is not what is wrong.
_DEFAULT_MODEL_BY_PROVIDER = {
    'openai': 'gpt-4o-mini',
    'anthropic': 'claude-3-5-haiku-latest',
}

#: No LLM call anywhere in fused-memory sets a timeout today, and the openai
#: SDK default is 600 seconds. On the SYNCHRONOUS ``add_memory`` write path
#: that is a wedge, not a degradation: the caller waits ten minutes for a
#: write C1 promises never to block. This bound keeps a hung provider to one
#: slow write rather than a hung server. It is 15 s rather than 10 s because
#: the frontier arms the judge-arm selection chooses between measured p95
#: 4.6–6.4 s against gpt-4o-mini's 1.7–1.9 s, and 10 s would cut into the tail
#: of the arm being selected (plans/write-triage-flip-readiness-prd.md §11.2
#: D15).
_DEFAULT_JUDGE_TIMEOUT_SECONDS = 15.0


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

    The ``llm.model`` leg is CONDITIONAL on the two providers agreeing. A model
    id is vendor-specific, so borrowing one across a provider boundary produces
    the same total-outage 404 as a blank name, only harder to read: the config
    is schema-valid, ``judge_provider``'s own schema description invites
    pinning it away from ``llm.provider``, and ``judge_model`` ships as
    ``null``. When ``llm.provider`` is a KNOWN provider that differs from the
    resolved judge provider, the inheritance is skipped in favour of that
    provider's own default. When ``llm.provider`` is absent or unrecognised
    there is nothing to disagree with — the vendor of ``llm.model`` is simply
    unknown — so the inheritance stands, as it did before.
    """
    provider = resolve_judge_provider(memory_service)
    llm_provider = _llm_attr(memory_service, 'provider')
    inheritable = not (
        isinstance(llm_provider, str)
        and llm_provider in _KNOWN_PROVIDERS
        and llm_provider != provider
    )

    candidates = [_judge_attr(memory_service, 'judge_model')]
    if inheritable:
        candidates.append(_llm_attr(memory_service, 'model'))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return _DEFAULT_MODEL_BY_PROVIDER.get(provider, _DEFAULT_JUDGE_MODEL)


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


def resolve_judge_reasoning_effort(memory_service: Any) -> str | None:
    """The reasoning effort to send, or ``None`` to omit the parameter.

    Only a member of ``config.schema.JudgeReasoningEffort`` is returned; any
    other value reads as ``None`` rather than raising, because this runs on the
    write path. ``None`` is a first-class answer, not a fallback: it is what a
    non-reasoning model needs. Whether the resolved arm can SEND a set effort is
    :func:`_call_llm`'s question, asked inside the fail-open arm.
    """
    value = _judge_attr(memory_service, 'judge_reasoning_effort')
    if isinstance(value, str) and value in get_args(JudgeReasoningEffort):
        return value
    return None


# --- the LLM call ------------------------------------------------------------

#: Output cap for the two answer-only arms (anthropic, and a chat-only compat
#: endpoint). The answer is a closed-vocabulary word plus a candidate id — a
#: 36-char opaque uuid, which tokenizes far worse than prose — inside a
#: two-key JSON object. Sized to leave room for a model that adds a short
#: `reasoning` key (the parser ignores extra keys) without leaving room for an
#: essay billed per token on every middle-band write. Too tight is not a worse
#: verdict: a truncated answer is unparseable, i.e. a counted fail-open.
_JUDGE_MAX_TOKENS = 128

#: The Responses API's ``max_output_tokens``, which bounds reasoning PLUS the
#: answer, so it is a different dimension from :data:`_JUDGE_MAX_TOKENS`.
#: Measured 2026-09-30 over 3,669 Responses calls, the largest output was 1,161
#: tokens (gpt-6-luna at medium effort); gpt-6.1-sol at low peaked at 168 and
#: every low-effort arm at 476. 2,048 is ~1.8x the largest measured and ~4x any
#: low-effort arm. Exhaustion is a counted fail-open naming its reason, never a
#: wrong verdict, and a pathological answer that reaches the cap costs about
#: USD 0.02 at gpt-6.1-sol list price. See plans/write-triage-flip-readiness-prd.md
#: §11.
_JUDGE_MAX_OUTPUT_TOKENS = 2_048


class _ProviderCredentials(NamedTuple):
    """How to reach *provider*'s endpoint, and what that endpoint speaks."""

    #: The SDK constructor kwargs (``api_key``/``base_url``), possibly empty.
    client_kwargs: dict[str, Any]
    #: Whether the endpoint serves the OpenAI Responses API.
    serves_responses_api: bool


def _provider_credentials(memory_service: Any, provider: str) -> _ProviderCredentials:
    """``api_key``/``base_url`` for *provider*, defensively, and its capability.

    Empty ``client_kwargs`` is a FIRST-CLASS result, not a failure: both SDKs
    fall back to their standard environment variables, which is how this
    deployment is actually configured (``OPENAI_API_KEY`` in the shell, nothing
    in config.yaml). Reading the config section is for a deployment that pins a
    key or points at an OpenAI-compatible local endpoint.

    ``serves_responses_api`` is the deployment's existing declaration of what
    the configured OpenAI endpoint speaks — ``llm.client_class``, the same
    declaration ``backends/graphiti_client.py`` builds its client from, where
    ``'openai_generic'`` names a compat endpoint serving chat.completions only.
    It is deliberately NOT inferred from the model name, nor from ``base_url``
    being present: the shipped config always sets a ``base_url``. A missing or
    unrecognised value reads as ``'openai'``, ``LLMConfig.client_class``'s
    default.
    """
    config = getattr(memory_service, 'config', None)
    llm = getattr(config, 'llm', None)
    providers = getattr(llm, 'providers', None)
    section = getattr(providers, provider, None)
    client_kwargs: dict[str, Any] = {}
    api_key = getattr(section, 'api_key', None)
    if isinstance(api_key, str) and api_key:
        client_kwargs['api_key'] = api_key
    api_url = getattr(section, 'api_url', None)
    if isinstance(api_url, str) and api_url:
        client_kwargs['base_url'] = api_url
    serves_responses_api = (
        provider == 'openai' and _llm_attr(memory_service, 'client_class') != 'openai_generic'
    )
    return _ProviderCredentials(client_kwargs, serves_responses_api)


class _JudgeReply(NamedTuple):
    """What one judge call returned: the raw answer text and the provider's usage."""

    text: str
    usage: JudgeUsage | None


def _is_token_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _usage_from(
    usage: object, *, input_field: str, output_field: str, details_field: str | None,
) -> JudgeUsage | None:
    """Read a provider's *usage* object as a :class:`JudgeUsage`, or ``None``.

    Counts are taken only when they are real ``int``s — never coerced, because
    ``int()`` of an unspecced Mock is 1 and a ``bool`` is an ``int``. Never
    raises: usage is metadata about a call that already succeeded, so an
    unreadable one is carried as ``None`` (which the eval reports as unpriced)
    rather than turning a good verdict into a fail-open.
    """
    input_tokens = getattr(usage, input_field, None)
    output_tokens = getattr(usage, output_field, None)
    if not (_is_token_count(input_tokens) and _is_token_count(output_tokens)):
        return None
    reasoning_tokens = None
    if details_field is not None:
        details = getattr(usage, details_field, None)
        reasoning_tokens = getattr(details, 'reasoning_tokens', None)
    return JudgeUsage(
        input_tokens,
        output_tokens,
        reasoning_tokens if _is_token_count(reasoning_tokens) else None,
    )


async def _call_llm(
    *,
    provider: str,
    model: str,
    prompt: str,
    memory_service: Any,
    timeout: float,
    reasoning_effort: str | None,
) -> _JudgeReply:
    """One single-turn call to *provider*: the raw response text and its usage.

    Three arms behind one dispatch. NATIVE OPENAI — an endpoint that serves the
    Responses API, per :class:`_ProviderCredentials` — posts
    ``responses.create``: the system prompt as ``instructions=``, the user turn
    as ``input=``, a JSON-object text format so that arm's happy path is the
    parser's, :data:`_JUDGE_MAX_OUTPUT_TOKENS`, and ``reasoning={'effort': …}``
    only when ``write_triage.judge_reasoning_effort`` is set. It sends NO
    ``temperature``: reasoning models reject the parameter. A consequence worth
    knowing when comparing runs: gpt-4o-mini on this arm samples at the
    provider default rather than at 0.0.

    An OPENAI-COMPATIBLE endpoint (``llm.client_class: openai_generic`` —
    llama.cpp, vLLM, LM Studio) serves chat.completions only, so it keeps the
    chat request unchanged: ``temperature=0.0``, :data:`_JUDGE_MAX_TOKENS` and
    ``response_format`` json_object (task 5277 C owns its compat 400s). Which
    OpenAI arm runs is decided by that capability flag, never by the model
    name. The ANTHROPIC arm is unchanged too: ``temperature=0.0``, because
    Anthropic's default is 1.0 and sampling a closed-vocabulary classifier
    buys nothing but parse failures, and the system prompt via ``system=``,
    because Anthropic has no system ROLE.

    A set *reasoning_effort* on an arm that cannot send it RAISES before any
    client is built. Silently dropping it would let the operator believe the
    selected configuration is running when it is not (INV-11). Like the
    unknown-provider raise, it lands in ``triage_write``'s fail-open arm, so it
    is counted and logged naming the leaf to fix.

    The client is constructed PER CALL and deliberately not cached on a module
    global. ``add_memory`` is served by one long-lived server process, and a
    cached client keyed to a config that hot-reloads would pin a stale
    ``model``/``api_url`` past a reload that the operator was told had
    applied — silently converting a green-tier knob into a restart-only one.
    Constructing here means a hot-reloaded ``api_key``/``api_url`` takes
    effect on the very next call.

    It is also CLOSED per call, via ``async with``. Neither SDK client
    defines ``__del__`` (measured: openai 2.31.0, anthropic 0.92.0), so an
    unclosed one abandons an ``httpx`` connection pool to the garbage
    collector on every triaged write. The ``asyncio.wait_for`` sits INSIDE
    the context deliberately: a timeout cancels the in-flight request, and a
    close written after the awaited call would never run — leaking precisely
    when the provider is slow and writes are piling up.

    The remaining cost is the lost connection reuse: ~100–300ms of TCP+TLS
    handshake per call, on the synchronous write path. That is a deliberate
    trade — correctness of the hot-reload contract over latency — and is
    recorded as a follow-up rather than resolved with a cache here.

    ``async with`` is NOT an exception handler and must not become one: it
    swallows nothing, so the no-``try``/``except`` property below still
    holds exactly.

    NO ``try``/``except`` ANYWHERE. Every failure propagates to
    ``triage_write``'s ``except`` arm (write_triage.py:835), which logs with
    ``exc_info``, counts exactly one fail-open, and returns ``stored``.

    An unrecognised *provider* RAISES rather than falling back to a default.
    That is the opposite of :func:`resolve_judge_provider`'s behaviour, and
    deliberately so: the resolver runs on the write path where C1 forbids
    raising, whereas this raise lands INSIDE the fail-open arm. Silently
    picking an arm here would bill an account the operator never chose.
    """
    if provider not in _KNOWN_PROVIDERS:
        raise ValueError(
            f'unknown judge provider {provider!r}; implemented arms are '
            f'{list(_KNOWN_PROVIDERS)}',
        )
    creds = _provider_credentials(memory_service, provider)
    if reasoning_effort is not None and not creds.serves_responses_api:
        why = (
            'the anthropic arm has no reasoning-effort parameter'
            if provider == 'anthropic'
            else "llm.client_class='openai_generic' names an endpoint serving "
            'chat.completions only'
        )
        raise ValueError(
            f'write_triage.judge_reasoning_effort={reasoning_effort!r} cannot be '
            f'sent: {why}. Set it to null for this arm, or judge on an endpoint '
            'that serves the Responses API.',
        )
    if provider == 'anthropic':
        return await _call_anthropic(creds, model=model, prompt=prompt, timeout=timeout)
    if creds.serves_responses_api:
        return await _call_openai_responses(
            creds, model=model, prompt=prompt, timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
    return await _call_openai_chat(creds, model=model, prompt=prompt, timeout=timeout)


async def _call_openai_responses(
    creds: _ProviderCredentials,
    *,
    model: str,
    prompt: str,
    timeout: float,
    reasoning_effort: str | None,
) -> _JudgeReply:
    """Native OpenAI, on the Responses API that frontier reasoning models require."""
    import openai  # noqa: PLC0415 — per-call import, matching judge.py

    # Omitted outright when unset, not sent as the SDK's `omit`, so the call's
    # kwargs say exactly what reaches the wire.
    reasoning: dict[str, Any] = (
        {} if reasoning_effort is None else {'reasoning': {'effort': reasoning_effort}}
    )
    async with openai.AsyncOpenAI(**creds.client_kwargs) as client:
        response = await asyncio.wait_for(
            client.responses.create(
                model=model,
                instructions=JUDGE_SYSTEM_PROMPT,
                input=prompt,
                max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
                text={'format': {'type': 'json_object'}},
                **reasoning,
            ),
            timeout=timeout,
        )
    # An incomplete answer is not a verdict even when its partial text parses,
    # and the logged reason must say so: an exhausted budget would otherwise
    # read as an empty body, indistinguishable from a model that answered
    # nothing (INV-2).
    if getattr(response, 'status', None) == 'incomplete':
        reason = getattr(getattr(response, 'incomplete_details', None), 'reason', None)
        raise _reject(f'response incomplete ({reason})', response.output_text)
    return _JudgeReply(
        response.output_text or '',
        _usage_from(
            getattr(response, 'usage', None),
            input_field='input_tokens',
            output_field='output_tokens',
            details_field='output_tokens_details',
        ),
    )


async def _call_openai_chat(
    creds: _ProviderCredentials, *, model: str, prompt: str, timeout: float,
) -> _JudgeReply:
    """An OpenAI-compatible endpoint that serves chat.completions only."""
    import openai  # noqa: PLC0415 — per-call import, matching judge.py

    async with openai.AsyncOpenAI(**creds.client_kwargs) as client:
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
    return _JudgeReply(
        response.choices[0].message.content or '',
        _usage_from(
            getattr(response, 'usage', None),
            input_field='prompt_tokens',
            output_field='completion_tokens',
            details_field='completion_tokens_details',
        ),
    )


async def _call_anthropic(
    creds: _ProviderCredentials, *, model: str, prompt: str, timeout: float,
) -> _JudgeReply:
    """The anthropic Messages API."""
    import anthropic  # noqa: PLC0415 — per-call import, matching judge.py

    async with anthropic.AsyncAnthropic(**creds.client_kwargs) as client:
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                temperature=0.0,
                max_tokens=_JUDGE_MAX_TOKENS,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
            ),
            timeout=timeout,
        )
    # First TEXT block, not first block: a leading thinking/tool_use block
    # must not be read as the answer.
    text_blocks = [b for b in response.content if b.type == 'text']
    return _JudgeReply(
        text_blocks[0].text if text_blocks else '',
        _usage_from(
            getattr(response, 'usage', None),
            input_field='input_tokens',
            output_field='output_tokens',
            details_field=None,
        ),
    )


async def judge_write(
    *,
    memory_service: Any,
    content: str,
    project_id: str,
    decision: Any,
    candidates: Any = (),
) -> TriageJudgeVerdict:
    """Adjudicate one middle-band write, naming the candidate the verdict is about.

    This is what ``tools.py`` passes as ``triage_write(..., judge=...)``,
    replacing leaf beta's ``_stub_judge``. The signature is beta's, plus the
    ``candidates`` keyword beta's slot did not carry: PRD C1 requires the
    judge to see the new entry AND its top 3–5 candidates, and a judge shown
    only a canonical ID cannot classify anything.

    Flow: resolve config LIVE → return ``stored`` early if disabled or if the
    slate selects to empty (the selector guarantees the band's winner is
    shown) → build the prompt → call the provider under ``asyncio.wait_for``
    → parse against the rendered slate, so the verdict can only name a
    candidate the model was shown.

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

    The disabled branch SAYS so, at INFO, once per write while the switch is
    engaged. Uncounted is not the same as unannounced: an unlogged kill
    switch is indistinguishable from a novel corpus in the ack stream, which
    is the very confusion the ``_DEFAULT_JUDGE_ENABLED = True`` comment
    argues against — and defaulting the knob to True does nothing for the
    operator who sets it to False. The empty-slate return stays unlogged
    deliberately: it is the ordinary per-write case and would drown the line
    that matters.

    PROVIDER. ``judge_provider``/``judge_model`` default to None and INHERIT
    ``llm.provider``/``llm.model``, which ship as ``openai``/``gpt-4o-mini``.
    That default is evidence-based, not preference: measured on this
    deployment ``ANTHROPIC_API_KEY`` is unset (CLAUDE.md — agents use OAuth)
    while ``OPENAI_API_KEY`` is set and demonstrably works, since leaf alpha's
    committed calibration report is the product of live OpenAI calls from this
    same checkout. The anthropic arm is implemented and selectable by config
    for a deployment that has the key; PRD C1's "haiku-class" is a cost/size
    class, not a vendor pin. On openai, an endpoint that serves the Responses
    API (the shipped ``llm.client_class: openai``) is called through it, with
    ``write_triage.judge_reasoning_effort`` sent when set; a chat-only
    compatible endpoint keeps chat.completions (see :func:`_call_llm`).

    USAGE. The verdict carries what the provider reported for the call
    (``TriageJudgeVerdict.usage``), so the eval can price a write without
    patching the SDK (task 5846 item 3). The early returns made no call and
    carry ``None``; so does a call whose usage could not be read.
    """
    if not resolve_judge_enabled(memory_service):
        # SAID OUT LOUD, unlike the empty-slate return below. An unlogged
        # kill switch is indistinguishable from a novel corpus in the ack
        # stream — the exact confusion `_DEFAULT_JUDGE_ENABLED = True` is
        # justified against, which defaulting the knob does nothing about for
        # the operator who sets it False. INFO and not a counter: this is a
        # decision, not a failure.
        logger.info(
            'write_triage judge disabled by config '
            '(write_triage.judge_enabled=false); middle-band write acked as '
            '%r without an LLM call',
            OUTCOME_STORED,
        )
        return TriageJudgeVerdict(OUTCOME_STORED)

    selected = select_judge_candidates(
        candidates,
        resolve_judge_candidate_count(memory_service),
        canonical_id=getattr(decision, 'canonical_id', None),
    )
    if not selected:
        return TriageJudgeVerdict(OUTCOME_STORED)

    reply = await _call_llm(
        provider=resolve_judge_provider(memory_service),
        model=resolve_judge_model(memory_service),
        prompt=build_judge_prompt(content, selected),
        memory_service=memory_service,
        timeout=resolve_judge_timeout(memory_service),
        reasoning_effort=resolve_judge_reasoning_effort(memory_service),
    )
    verdict = parse_judge_verdict(reply.text, [candidate.id for candidate in selected])
    return verdict._replace(usage=reply.usage)
