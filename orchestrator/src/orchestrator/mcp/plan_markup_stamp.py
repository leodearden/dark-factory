"""The rejected-call counter a plan-tools markup refusal stamps onto the plan.

Task 4597 (esc-4528-1). ``plan_tools.create_server`` registers the boundary
guard under ``RepairPolicy.REJECT_WITH_REPAIR``, so a leaking call is REFUSED
before the tool body runs and NOTHING is written to ``plan.json``. That is the
right disposition — forwarding a repair would write a guessed-at document every
later reader inherits — but it leaves the loss invisible at the one artifact
every later reader actually opens. A refusal reaches three places today: a
``ToolError`` the caller may not survive to act on, one line in
``data/orchestrator/markup-guard/plan-tools.jsonl``, and (unrepairable only) a
residue escalation. The plan itself records nothing, so ``design_decisions: []``
is genuinely ambiguous between "the architect never called" and "the architect
called six times and was refused six times". In task 4528 a reviewer could tell
the two apart only because the architect happened to hand-file an info note.

So the fact channel gets a SECOND consumer: a guard-owned block under the
top-level :data:`PLAN_MARKUP_REJECTIONS_KEY`, underscore-prefixed to match the
document's existing machine-written envelope (``_schema_version``,
``_finalized_at``, ``_revalidated_at``). It carries a total ``count``, a
``by_tool`` aggregate, the ``first_at`` / ``last_at`` bounds of the window, up
to :data:`MARKUP_STAMP_MAX_EVENTS` individual events, and one static ``note``
naming what the block means — so a reader who has never seen the key does not
have to infer it.

WHAT AN EVENT HOLDS: ``ts``, ``tool``, ``param``, ``outcome``, and nothing
else. :data:`STAMP_EVENT_KEYS` is the DECLARED allowlist, and
:func:`build_event` copies FROM it rather than filtering the record — so a fact
that grows a field tomorrow cannot land in ``plan.json`` by default.

WHAT IT DELIBERATELY DOES NOT HOLD: the matched ``pattern``, the ``misclose``
tag, and the raw caller payload. Three independent measured reasons:

1. ``pattern`` and ``misclose`` ARE envelope markup. ``plan.json`` is embedded
   verbatim into four architect-facing prompts (``briefing.py``'s revalidation
   and completion passes, ``workflow._replan``, and the evals judge) and agents
   EDIT the document, so a literal stored there reproduces the exact
   over-consumption defect at the one artifact a reader is told to open. It is
   the same reason ``markup_journal._encode`` escapes its own bytes and the
   committed specimen corpus escapes every literal.
2. ``scripts/sweep_toolcall_markup.py`` walks dead-lane ``plan.json``
   RECURSIVELY (``_repair_dict`` over ``list(node.keys())``, ``_string_holes``
   over ``node.items()``), so a stored literal would be classified as fresh
   corruption and inflate the very census the sweep exists to report.
3. The raw payload already has exactly one owner — the residue escalation
   (``markup_sink.residue_detail``), which is by contract the only surviving
   copy of data the caller may never be able to resend. A second, weaker copy
   is the INV-5 duplication the containment PRD rules against.

``outcome`` is recorded VERBATIM from the fact channel rather than re-derived
into the caller-facing ``error_type`` spelling. The two are the same fact under
two names (``rejected`` renders as ``mcp_markup_detected``, ``unrepairable`` as
``mcp_markup_unrepairable``), but only the second spelling is exported as a
constant; the first is an inline literal inside the middleware's ``_reject``.
Respelling it here would be a second copy that must stay in lock step with a
private line in another package — precisely the duplication INV-5 exists to end.

Sink shape inherited from ``markup_journal.make_fact_journal`` and
``markup_sink.make_escalation_sink``: never raises, does its blocking work
under ``asyncio.to_thread``, returns a locator string or ``None``. The three
injected channels on this one boundary behave identically under every failure
mode, so a reader who has read one has read all three.

PRD: ``plans/toolcall-markup-containment-prd.md``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# NO ENVELOPE LITERAL APPEARS IN THIS FILE, and none is needed: the module
# names the fields it refuses to copy rather than exhibiting them. Should one
# ever be required here, spell its angle bracket from ``chr(60)`` as
# ``markup_journal`` does — an agent editing a file that holds a raw literal has
# to emit that literal inside its own tool-call argument, reproducing the very
# defect this module records.

#: The plan's top-level key this block lives under. Underscore-prefixed to join
#: the document's machine-written envelope rather than its authored fields, so
#: a reader can tell at a glance that no agent wrote it.
PLAN_MARKUP_REJECTIONS_KEY = '_markup_rejections'

#: How many individual events the block retains. ``plan.json`` is embedded
#: verbatim into four architect-facing prompts, so an unbounded list would let
#: one leaking session bloat every later prompt that renders the document. The
#: uncapped ``count`` and the tool-bounded ``by_tool`` aggregate are what
#: actually answer the question; the events are the texture.
MARKUP_STAMP_MAX_EVENTS = 20

#: Per-string cap on every copied field. ``tool`` and ``param`` come from the
#: invoked tool's own registration and schema, so they are short in practice —
#: the cap is what makes that a guarantee rather than an observation.
MARKUP_STAMP_MAX_FIELD_CHARS = 120

#: THE DECLARED ALLOWLIST of what a stamped event may carry, and the one place
#: it is stated. :func:`build_event` copies FROM this tuple rather than
#: filtering the fact record, so the default for a field nobody has reviewed is
#: "not in the plan". ``ts`` is this module's own (from the injected clock);
#: the other three are copied from the record.
STAMP_EVENT_KEYS: tuple[str, ...] = ('ts', 'tool', 'param', 'outcome')

#: The subset of :data:`STAMP_EVENT_KEYS` read off the middleware's fact record.
_FACT_COPIED_KEYS: tuple[str, ...] = ('tool', 'param', 'outcome')

#: One sentence saying what the block means, carried INSIDE it. A bare integer
#: under an underscore-prefixed key is not self-describing to a reader who has
#: never seen it, and the implementer, the reviewer and reconciliation all meet
#: this document cold. Deliberately STATIC: interpolating counts or tool names
#: would put caller-adjacent text into a field whose whole justification is that
#: it holds none, and would duplicate numbers that sit two keys away.
STAMP_NOTE = (
    'Machine-written by the plan-tools markup guard. Each event is one tool '
    'call this plan refused because its arguments carried leaked tool-call '
    'envelope markup; the call never ran, so a short or empty design_decisions '
    '/ reuse / steps list may be a LOSS rather than an omission. The refused '
    'payloads are preserved in the markup-guard journal and, on the '
    'unrepairable path, in a residue escalation.'
)


def _capped(value: object) -> object:
    """*value* bounded to :data:`MARKUP_STAMP_MAX_FIELD_CHARS`, as a PREFIX.

    A trimmed string is a prefix of what was sent, never a rewrite — the same
    discipline ``markup_journal._capped`` and the middleware's own repair keep,
    so a reader comparing this block against a journal line or a transcript
    sees the beginning of the real value rather than an elision of it.

    A non-string passes through untouched: the fact channel emits ``str`` here
    today, but coercing an unexpected type would destroy the evidence that it
    arrived.
    """
    if isinstance(value, str) and len(value) > MARKUP_STAMP_MAX_FIELD_CHARS:
        return value[:MARKUP_STAMP_MAX_FIELD_CHARS]
    return value


def build_event(
    record: dict[str, Any], *, now: Callable[[], float] = time.time
) -> dict[str, Any]:
    """One stamped event from a middleware fact *record*.

    AN ALLOWLIST COPY, never a filtered one. The distinction is the whole
    argument for what this block is allowed to hold: filtering inverts the
    default, so a middleware that grows a field tomorrow would land it in
    ``plan.json`` without anyone deciding to. Copying from
    :data:`STAMP_EVENT_KEYS` means a new field is absent until someone adds its
    name here and re-argues the three reasons in this module's docstring.

    In particular ``pattern`` and ``misclose`` — which ARE the leaked envelope
    markup — and ``recovered_params`` / ``agent_id`` / ``project`` are left
    behind. The first two would poison the document; the rest are either
    derived from the caller's payload or structurally ``None`` on this boundary.

    *now* is a clock callable (``time.time``'s shape), injected rather than
    called directly so a test can assert the stamped ``ts`` as a value.
    """
    event: dict[str, Any] = {
        'ts': datetime.fromtimestamp(now(), tz=UTC).isoformat(),
    }
    for key in _FACT_COPIED_KEYS:
        event[key] = _capped(record.get(key))
    return event


# ---------------------------------------------------------------------------
# The block algebra — pure functions over plain dicts, no I/O.
# ---------------------------------------------------------------------------
#
# Kept apart from the sink deliberately: the merge is where every subtlety
# lives (the cap, the bounds, the disclosure key, the degradation) and it is
# reviewable and testable without a filesystem. The sink below is then thin
# enough to read in one pass.


def _as_int(value: object, *, field: str) -> int:
    """*value* as a non-negative count, degrading to ``0``.

    A merge runs INSIDE a decided refusal, so raising here would turn a working
    guard into an outage of its own — and ``plan.json`` is agent-adjacent, so
    the block on disk may be anything. An unusable value therefore contributes
    its IDENTITY rather than propagating: the merged block degrades to a
    partially-recovered count, and the INCOMING event is never the thing that
    gets dropped. Logged at warning, never silently.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        if value is not None:
            logger.warning(
                'markup stamp: %s on the stored rejection block is %r, which '
                'is not a count; treating it as 0', field, value,
            )
        return 0
    return max(value, 0)


def _as_by_tool(value: object) -> dict[str, int]:
    """*value* as a ``{tool: count}`` tally, dropping only what is unusable."""
    if not isinstance(value, dict):
        if value is not None:
            logger.warning(
                'markup stamp: by_tool on the stored rejection block is %r, '
                'which is not a mapping; treating it as empty', value,
            )
        return {}
    tally: dict[str, int] = {}
    for tool, count in value.items():
        if not isinstance(tool, str):
            continue
        # PER ENTRY, not all-or-nothing: one junk tally must not cost the
        # other tools their counts.
        tally[tool] = _as_int(count, field=f'by_tool[{tool!r}]')
    return tally


def _as_events(value: object) -> list[dict[str, Any]]:
    """*value* as a list of event dicts, dropping only what is unusable."""
    if not isinstance(value, list):
        if value is not None:
            logger.warning(
                'markup stamp: events on the stored rejection block is %r, '
                'which is not a list; treating it as empty', value,
            )
        return []
    return [event for event in value if isinstance(event, dict)]


def _as_bound(value: object) -> str | None:
    """*value* as a window bound, or ``None`` when it is not one."""
    return value if isinstance(value, str) and value else None


def block_of(event: dict[str, Any]) -> dict[str, Any]:
    """The one-event block — the identity every merge starts from."""
    return {
        'count': 1,
        'by_tool': {event['tool']: 1} if isinstance(event.get('tool'), str) else {},
        'first_at': _as_bound(event.get('ts')),
        'last_at': _as_bound(event.get('ts')),
        'events': [event],
        'note': STAMP_NOTE,
    }


def merge_block(left: object, right: object) -> dict[str, Any]:
    """*left* and *right* folded into one block. Total, and never raises.

    ASSOCIATIVE over a sequence of events, which is load-bearing rather than
    tidy: the sink folds ONE incoming event into whatever is on disk, while
    ``_create_plan`` folds a whole BUFFERED block into a carried-forward one.
    Those are different association orders over the same events, and a merge
    that disagreed between them would let the same refusals produce two
    different documents.

    THE EVENT LIST KEEPS THE FIRST N, not the last. The consequence is that the
    list STOPS CHANGING once it is full, so a pathological leak churns two
    integers instead of rewriting the whole block on every refusal — and the
    events that survive are the ones from the beginning of the leak, which is
    where its shape is legible. The uncapped ``count`` and the tool-bounded
    ``by_tool`` are what answer the question anyway; the events are texture.

    ``events_truncated`` is written only once ``count`` EXCEEDS the retained
    list, following the convention that a disclosure key's PRESENCE means
    something was cut rather than merely that the emitter ran — a full list is
    not the same as a cut one. It carries no companion dropped-count on
    purpose: ``count - len(events)`` already says how many were dropped, and a
    stored second accounting is a number that can drift from the two beside it.

    ``note`` is REWRITTEN from :data:`STAMP_NOTE` rather than inherited, so the
    constant stays the single owner of the wording and a stale or hand-edited
    note on an agent-adjacent document is corrected instead of preserved.
    """
    sides = [side if isinstance(side, dict) else {} for side in (left, right)]
    for side, given in zip(sides, (left, right), strict=True):
        if not side and given is not None and not isinstance(given, dict):
            logger.warning(
                'markup stamp: the stored rejection block is %r, which is not '
                'a mapping; starting a fresh one', given,
            )

    by_tool: dict[str, int] = {}
    for side in sides:
        for tool, count in _as_by_tool(side.get('by_tool')).items():
            by_tool[tool] = by_tool.get(tool, 0) + count

    events: list[dict[str, Any]] = []
    for side in sides:
        events.extend(_as_events(side.get('events')))

    bounds = [
        [b for b in (_as_bound(side.get(field)) for side in sides) if b is not None]
        for field in ('first_at', 'last_at')
    ]

    count = sum(_as_int(side.get('count'), field='count') for side in sides)
    kept = events[:MARKUP_STAMP_MAX_EVENTS]

    merged: dict[str, Any] = {
        'count': count,
        'by_tool': by_tool,
        # A null bound never wins a min or a max: the bounds are taken over the
        # values that EXIST, so a side that does not know its window cannot
        # erase a bound the other side does know.
        'first_at': min(bounds[0]) if bounds[0] else None,
        'last_at': max(bounds[1]) if bounds[1] else None,
        'events': kept,
        'note': STAMP_NOTE,
    }
    if count > len(kept):
        merged['events_truncated'] = True
    return merged


def summary(plan: object) -> dict[str, Any] | None:
    """The compact ``{count, by_tool}`` view, or ``None`` when there is none.

    Deliberately NOT the whole block. This is what ``_confirm_plan`` folds into
    the architect's LAST tool result — a signal that something was lost, in the
    one place the loss reaches the durable agent transcript — not a second copy
    of a block that is already on disk two keys away.

    ``None`` on an absent OR unusable block, so the omit-when-absent response
    convention has an unambiguous thing to omit and a half-formed block can
    never reach a tool response.
    """
    if not isinstance(plan, dict):
        return None
    block = plan.get(PLAN_MARKUP_REJECTIONS_KEY)
    if not isinstance(block, dict):
        return None
    return {
        'count': _as_int(block.get('count'), field='count'),
        'by_tool': _as_by_tool(block.get('by_tool')),
    }
