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
