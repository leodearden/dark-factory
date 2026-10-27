"""Provenance predicates: is a carrier's text, or a record's provenance,
material other than this session's own dialogue?

Two answers live here. The text half recognises machine content re-ingested
into a transcript (a prior trickle-coder judgment, or a harness-injected
prompt, briefing or memory block) by its literal shape; the record half reads
Claude Code's structured ``origin`` stamp. Both are pure functions over plain
strings and record dicts, and this module imports nothing from the legibility
pipeline it serves.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

CODER_JUDGMENT_KEYS: frozenset[str] = frozenset({'matches', 'candidates'})
"""The two top-level keys of a trickle-coder judgment, whose canonical source
is the response schema in scripts/legibility/coder.py::build_prompt.
Restated rather than imported: this module feeds the digest coder.py
consumes, and importing the consumer would invert that layering for two key
names. A lockstep test over the reply build_prompt prescribes holds the two
in step, so renaming that schema fails a test instead of silently letting
coder answers back into the signal counts."""

_JSON_FENCE_RE = re.compile(r'\A```(?:json)?\s*\n(.*)\n```\Z', re.DOTALL)
"""One outer ```/```json fence enclosing an ENTIRE stripped carrier."""


def is_coder_judgment_payload(text: str) -> bool:
    """True when the WHOLE of *text* is a trickle-coder judgment: a JSON
    object carrying every :data:`CODER_JUDGMENT_KEYS` key, bare or inside at
    most ONE outer fence spanning the entire carrier -- the two spellings
    coder.py::parse_coder_output already reads as one payload.

    Deliberately narrower than that parser, which brace-slices an object out
    of surrounding prose because its job is rescuing a reply. Here that
    looseness would read an assistant turn that merely QUOTES the schema as
    machine content and suppress a genuine self-correction in it, and
    over-excluding genuine dialogue is the worse error -- the same trade-off
    :func:`is_harness_injected_turn` makes.

    Any carrier json.loads cannot handle answers False, RecursionError
    included (it is not a ValueError): this runs on every carrier of
    arbitrary transcripts and must answer, never abort a digest.
    """
    stripped = text.strip()
    fence = _JSON_FENCE_RE.match(stripped)
    if fence:
        stripped = fence.group(1).strip()
    if not stripped.startswith('{'):
        return False
    try:
        parsed = json.loads(stripped)
    except (ValueError, RecursionError):
        return False
    return isinstance(parsed, dict) and parsed.keys() >= CODER_JUDGMENT_KEYS


HARNESS_BRIEFING_HEADINGS: tuple[str, ...] = (
    '# context', '## agent identity', '# task',
)
"""ANCHOR heading literals for an injected orchestrator briefing
(orchestrator/src/orchestrator/agents/briefing.py):
``orchestrator/src/orchestrator/agents/memory_recall.py::render_context_block``
emits '# Context' on both its recalled-sections and its no-recalled-sections
paths, always with a SINGLE hash; ``_agent_identity`` emits
'## Agent Identity' (:272); the role
prompt templates emit '# Task' (:362/:435/:506/:589/:666/:1198). At least
one anchor must be present as a line-anchored heading for a turn to
classify as briefing-injected -- see :func:`is_harness_injected_turn` for
the corroboration rule that goes with it.

'## Context' (DOUBLE hash) is deliberately NOT an anchor, and must not be
re-added. briefing.py emits no such heading on any path. The literal does
occur elsewhere in the repo -- orchestrator/src/orchestrator/agents/roles.py
(:826 DEBUGGER, :1034 JUDGE, :1100 MERGER, :1378 STEWARD) and
orchestrator/src/orchestrator/evals/reviewer_trial/mining.py:359 -- but
those are agent system-prompt and eval-prompt body text, which never appear
as a user-turn record and so are never seen by this filter. Meanwhile
'## Context' followed by '## Conventions' is a common human-authored
markdown shape in this repo's own plans/ and docs/ prose, and since one
anchor plus one corroborator already meets the >=2 threshold, anchoring it
costs genuine gold turns for zero recall."""

HARNESS_BRIEFING_SUBHEADINGS: tuple[str, ...] = (
    '## project context', '## conventions', '## conventions & gotchas',
    '## recent decisions', '## task context',
)
"""CORROBORATING heading literals -- the structural sub-blocks a real
briefing carries alongside an anchor. Matched as WHOLE lowercased LINES
(see :func:`is_harness_injected_turn`), so a renamed heading needs its own
entry rather than matching by prefix.

Two generations, both live, because this filter reads transcripts written
long before it: today
``orchestrator/src/orchestrator/agents/memory_recall.py::MemoryRecall``
renders its headings from the section titles in
``shared/src/shared/briefing_queries.py`` -- '## Conventions & Gotchas' and
'## Task Context' -- while '## Project Context', '## Conventions' and
'## Recent Decisions' are the pre-task-3659 spellings, retired from the
query table but still present throughout the archived corpus. Do not prune
the retired three: dropping them silently un-classifies every historical
briefing turn.

Never sufficient alone: a human turn headed '## Conventions' carries no
anchor and stays gold.

Every corroborator is a '##'-level SUB-block, and that is a rule, not an
accident -- a '# '-level entry would pair with the '# task' anchor to
clear the >=2 threshold by itself. '# Action' is emitted by every role
prompt template (:367/:670/:827/:927/:968/:1007/:1095/:1120/:1212) and was
listed here until the task 3610 amendment pass, but '# Task' + '# Action'
is also an ordinary human spec-writing shape, and losing a genuine gold
turn is a SILENT error where an admitted briefing turn is a visible one.
It costs almost no recall: every template that carries a memory block
begins with ``{context}``, so such a briefing always carries the
'# Context' anchor and, whenever memory context is available, its '##'
sub-blocks too. The corner this declines is the memory-UNAVAILABLE variant
of ``BriefingAssembler.build_reviewer_prompt``, the one identity-less
template that still has a context slot, which then shows only '# Context' +
'# Action' -- a shape the pre-3610 all-of-three rule did not catch either,
so nothing regresses. ``BriefingAssembler.build_merger_prompt`` is out of
this rule's reach entirely: since task 3659 it carries no memory block at
all, so it emits no '# Context' anchor to be corroborated (D7 -- the merger
is mechanical and the generic block was never shown to help it). It is
covered instead by its own :data:`MERGER_HEADINGS` set below, because
losing it here would have been a REGRESSION, not a declined corner: before
3659 the merger prompt opened with '# Context' and was classified by this
very rule."""

RECON_RUN_REVIEW_HEADINGS: tuple[str, ...] = (
    '## reconciliation run review', '### run metadata', '### stage reports',
)
"""Injected reconciliation run-review PROMPT heading literals
(``fused_memory/reconciliation/judge.py::Judge._build_review_prompt``
emits '## Reconciliation Run Review', '### Run Metadata', '### Stage
Reports' and '### MCP Actions (N total)' as a single f-string). The three
parenthesis-free headings are used so the marker is an exact line match;
all must co-occur as line-anchored headings (all(), not any() -- the same
false-positive guard as HARNESS_BRIEFING_HEADINGS).

Harness-injected, NOT a human paste, verified by transcript forensics on
all five confusion-census-2026-07-31 §1.1 facet-(b) sightings (2b93092c,
5871ae29, 47c7fd98, ccf68b39, 47e690d3): the judge dispatches this prompt
through ``shared.cli_invoke`` -> ``claude --print`` (schema.py:871
``judge_llm_provider`` defaults to 'claude_cli'), which records it as the
session's sole user-role record. In each of the five, the run-review block
is the FIRST user record, preceded only by harness bookkeeping
(queue-operation enqueue/dequeue, an ``entrypoint: sdk-cli`` attachment),
with no ordinary human turn anywhere in the transcript. The census's
"pasted into a turn for review/discussion" reading is refuted: facets (a)
and (b) are one source class (harness injection) with two injectors
(orchestrator briefing vs. judge prompt), which is why these headings live
here rather than behind a separate 'pasted report' predicate."""

MERGER_HEADINGS: tuple[str, ...] = (
    '# task intent', '# merge conflicts', '# action',
)
"""Injected merge-resolution PROMPT heading literals
(``orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_merger_prompt``,
a single f-string emitting '# Task Intent', '# Merge Conflicts', '# Action').

The merger prompt carries no '# Context' anchor, so without this set
:data:`HARNESS_BRIEFING_HEADINGS` cannot recognise a merger dispatch, and
every merge-conflict dispatch would be mined as a genuine human turn and
rendered in the digest's gold 'User Correction' section (PRD Sec 5).

Strict all-of, like every non-briefing injector. All three headings are
'# '-level, which is exactly why the all-of match is required here and why
these literals must NOT be folded into
:data:`HARNESS_BRIEFING_SUBHEADINGS` -- a '# '-level corroborator would
pair with the '# task' anchor and clear the >=2 threshold on its own, the
false-positive the 3610 amendment pass removed '# action' to prevent. As a
co-occurring triple they are not an ordinary human shape: '# Merge
Conflicts' carries the discrimination."""

HARNESS_HEADING_SETS: tuple[tuple[str, ...], ...] = (
    HARNESS_BRIEFING_HEADINGS,
    RECON_RUN_REVIEW_HEADINGS,
    MERGER_HEADINGS,
)
"""Every known harness-injected heading set, matched independently: a turn
is harness-injected when ALL headings of ANY one set co-occur. Adding a
newly-sighted injector shape is a one-line addition here, the same way
HARNESS_PROMPT_MARKERS invites one-line additions for prose preambles.

HARNESS_BRIEFING_HEADINGS is retained as a member even though task 3610's
anchor+corroborator rule (see :func:`is_harness_injected_turn`) SUBSUMES
the all-of match for it -- three anchors clear that rule's >=2 threshold
by themselves -- so the two rules agree by construction and this stays the
one inventory of known injector heading sets. The relaxation is NOT
extended to the other members: it was justified by forensics on the
briefing shape specifically, and a newly-sighted injector earns one only
on its own evidence."""

HARNESS_PROMPT_MARKERS: tuple[str, ...] = (
    'you are the trickle coder for the dark-factory agent-confusion codebook',
    'your previous run was interrupted by a usage limit',
    'you were interrupted by an orchestrator restart',
)
"""Harness-authored PROSE preamble literals, matched as plain
case-insensitive substrings (unlike the line-anchored heading markers
above): the trickle coder's system prompt
(scripts/legibility/coder.py:174 ``build_prompt``), and the harness's two
resume prompts -- ``shared.cli_invoke.CAP_HIT_RESUME_PROMPT`` (cap-hit)
and ``shared.cli_invoke.CRASH_RECOVERY_RESUME_PROMPT`` (orchestrator
restart), the canonical sources of the two literals above. Both resume
prompts are one defect class -- harness-injected continuation boilerplate
typed into the transcript as an ordinary isMeta=False user turn -- even
though cli_invoke deliberately keeps the two strings separate because
their causes differ (a usage-cap interrupt vs. an orchestrator restart).
Each marker is deliberately its constant's stable FIRST SENTENCE, not the
whole string: a future rewording of a later sentence must not silently
un-cover the marker. Held in lockstep with the canonical constants by
``TestHarnessInjectedTurnFilter.test_resume_prompt_is_excluded_lockstep``
in scripts/tests/test_legibility_reingestion.py, which asserts
``is_harness_injected_turn(<constant>) is True`` for each rather than
restating the literal -- a harness rewording turns that test red instead
of silently regressing coverage. Extend with future harness prompt
literals as one-line additions."""

HARNESS_CONTEXT_BLOCK_MARKERS: tuple[str, ...] = (
    '_this context was recalled from the ',
    '_memory unavailable (',
    '_memory unavailable — proceed with codebase exploration',
    '_no memory context available',
)
"""Body literals the briefing's memory recall renders right after its
'# Context' heading (orchestrator/src/orchestrator/agents/memory_recall.py):
the standing provenance caveat's prefix
(``orchestrator.agents.memory_recall.MEMORY_CONTEXT_CAVEAT``, when a memory
section was actually recalled), and its two no-recalled-sections literal
families (memory-unavailable / no-memory-context-available). Every
marker deliberately stops BEFORE an interpolation point -- the caveat's
``{project_id}`` and MEMORY_OUTAGE_NOTICE's ``{reasons}`` -- because a
marker spanning one is not stable across call sites: the caveat's would
be project-specific and would fail for every non-dark_factory project the
census runs against (this module has no knowledge of which project a
transcript belongs to), and the outage notice's would pin one reason
class out of three.

The memory-unavailable family has TWO markers, both live, for the reason
the sibling ``HARNESS_BRIEFING_SUBHEADINGS`` carries two generations of
heading spellings: task 3659 reworded that notice to name WHY memory was
unavailable, so ``'_memory unavailable ('`` covers what is rendered today
and the longer literal covers the archived pre-3659 corpus. Do not prune
the retired one -- dropping it silently un-classifies every historical
outage turn.

These four markers are EXHAUSTIVE over
``orchestrator/src/orchestrator/agents/memory_recall.py::render_context_block``'s return
paths as of this commit: the no-recalled-sections paths (each of the two
literal families has a plain, a drop_note-bearing and a notices-bearing
variant, all three covered by the same family marker, since the family
line leads the body and the notices and drop_note are appended after it),
PLUS the recalled-sections path -- covered by the caveat marker ALONE, including
its own drop_note suffix (``'\n\n_In total, {drop_note}._'``) and the
trailing "_Memory unavailable for the remaining queries..._" note a
recall loop that broke later appends, since both are appended AFTER the caveat
prefix this marker matches on, never before it. That exhaustiveness
claim is what
``TestHarnessInjectedTurnFilter.test_no_recalled_sections_variant_is_excluded``
(every no-recalled-sections shape) and
``test_recalled_sections_with_trailing_unavailable_note_is_excluded``
(the composite recalled path), both in scripts/tests/test_legibility_digest.py,
together check, so a new return path added
to that function should arrive with its own marker here. Matched only in CONJUNCTION with
a line-anchored '# context' heading (see :func:`is_harness_injected_turn`),
never as a relaxation of the briefing anchor+corroborator guard -- that
guard is load-bearing and its two negative tests
(test_single_heading_alone_is_not_excluded,
test_context_heading_mentioned_mid_sentence_is_not_excluded) must keep
passing unchanged."""


def _heading_lines(text: str) -> frozenset[str]:
    """*text*'s lines, each stripped and lowercased.

    The single normalization used by every line-anchored heading match.
    Shared rather than repeated per rule so a future change to the
    anchoring rule (stripping trailing '#', handling '\\r\\n', ...) lands
    in one place instead of silently diverging between two copies.
    """
    return frozenset(line.strip() for line in text.lower().splitlines())


def is_harness_injected_turn(text: str) -> bool:
    """True when *text* is harness-injected rather than genuine human-typed
    input: the orchestrator's briefing preamble, any OTHER injected
    preamble whose whole heading set co-occurs each as its own stripped
    line (any :data:`HARNESS_HEADING_SETS` entry -- notably the
    reconciliation judge's run-review prompt), a harness prose preamble
    (any HARNESS_PROMPT_MARKERS substring), or a lone memory-context
    block (a line-anchored '# context' heading together with any
    HARNESS_CONTEXT_BLOCK_MARKERS substring).

    The briefing rule is ANCHOR + CORROBORATOR: at least one
    HARNESS_BRIEFING_HEADINGS anchor must appear as its own stripped line,
    AND at least two DISTINCT briefing headings (anchors union
    HARNESS_BRIEFING_SUBHEADINGS) must be present in total.

    This replaces the original all-of-three rule ('# Context' AND '## Agent
    Identity' AND '# Task'), which the 07-31 confusion census showed was too
    strict to catch the shape actually injected most often: cluster 1.1(b)
    (plans/confusion-census-2026-07-31.md:79/:85) sighted a context-ONLY
    injection -- '# Context' followed by '## Project Context' and the
    memory-search JSON dump, with neither '## Agent Identity' nor '# Task'
    present -- passing the filter BY CONSTRUCTION and being rendered as a
    fabricated gold "User Correction" in four dispatched-task digests.

    Both false-positive guards the original all() was written to protect
    still hold, for reasons the relaxation preserves rather than tolerates:
    a genuine human turn opening with a lone '# Task' has one heading and
    NO corroborator, so it falls below the >=2 threshold; and a '# Context'
    quoted mid-prose is not a stripped line at all, so it is never a
    heading -- matching stays line-anchored and never substring.

    The memory-context rule is the residual the briefing rule declines: a
    '# Context' block whose memory sections were recalled but which carries
    no second heading is caught by its standing provenance caveat
    (HARNESS_CONTEXT_BLOCK_MARKERS) instead of by a corroborator.

    Every OTHER known injector keeps the strict all-of-set rule, applied
    over :data:`HARNESS_HEADING_SETS` (task 3614): the relaxation above is
    justified by forensics on the BRIEFING shape specifically, and a
    newly-sighted injector earns a relaxation only on its own evidence.
    HARNESS_BRIEFING_HEADINGS stays listed in that inventory as the
    canonical example of a heading set, where it is subsumed -- three
    anchors clear the >=2 threshold by themselves -- never contradicted.

    All of it stays deliberately literal and line-anchored rather than a
    structural "looks machine-generated" heuristic: user corrections are
    gold (PRD Sec 5) and the highest-priority digest section, so
    over-excluding a genuine human turn -- e.g. a long, well-formatted
    human bug report with headings and a pasted log excerpt -- is strictly
    the worse error.
    """
    lowered = text.lower()
    lines = _heading_lines(text)
    anchors = [h for h in HARNESS_BRIEFING_HEADINGS if h in lines]
    if anchors:
        corroborators = [h for h in HARNESS_BRIEFING_SUBHEADINGS if h in lines]
        if len(anchors) + len(corroborators) >= 2:
            return True
    if any(lines.issuperset(headings) for headings in HARNESS_HEADING_SETS):
        return True
    if any(marker in lowered for marker in HARNESS_PROMPT_MARKERS):
        return True
    return '# context' in lines and any(m in lowered for m in HARNESS_CONTEXT_BLOCK_MARKERS)


def is_reingested_content(text: str) -> bool:
    """True when *text* is machine content re-ingested as session material
    rather than this session's own dialogue.

    Two shapes are known: a prior trickle-coder answer
    (:func:`is_coder_judgment_payload`) and an injected harness prompt or
    briefing (:func:`is_harness_injected_turn`). Both quote signal literals
    that belong to another session or to the harness. Session b203a05c
    record 22, a coder answer, fired self_correct; session 6a527d51 record 3,
    the trickle-coder prompt embedding the digest it codes, fired df_guard.

    Consulted by EVERY signal bucket, never by one: task 5685's ruling is
    that this contamination is fixed at the content-classification layer,
    not per bucket. A newly-sighted machine-payload shape is a one-line
    addition to this union.
    """
    return is_coder_judgment_payload(text) or is_harness_injected_turn(text)


HUMAN_ORIGIN_KIND: str = 'human'
"""The only Claude Code ``origin.kind`` that asserts a human typed the record."""


def has_non_human_origin(record: Mapping[str, Any]) -> bool:
    """True when *record*'s own structured provenance names a non-human
    producer: Claude Code stamps queued user prompts with ``origin`` (a
    background task-notification, an auto-continuation, a coordinator or
    peer message, ...), and any kind but :data:`HUMAN_ORIGIN_KIND` counts.

    Unknown provenance -- no ``origin``, a non-dict one, or one without a str
    ``kind`` -- answers False, so the record falls through to the text rules.
    Why there is no text fallback: plans/confusion-reduction-prd.md §7.2.2
    (generation 4).
    """
    origin = record.get('origin')
    if not isinstance(origin, dict):
        return False
    kind = origin.get('kind')
    return isinstance(kind, str) and kind != HUMAN_ORIGIN_KIND
