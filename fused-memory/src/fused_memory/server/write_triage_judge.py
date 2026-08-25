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

from fused_memory.routing.json_extract import extract_json
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)


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
