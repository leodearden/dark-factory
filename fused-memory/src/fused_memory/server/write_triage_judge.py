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
