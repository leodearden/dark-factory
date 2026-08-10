"""Stale task-status snapshot Graphiti edge sweep — task 2613.

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that keeps VALID
(``invalid_at IS NULL``) task-status-snapshot Graphiti edges in sync with the
tasks they describe. A status-snapshot edge asserts that some task_id is
currently active/pending/in-progress; once that task reaches a terminal
status (done/cancelled), the edge is stale but nothing retires it. The prior
approach — a manual LLM `search` sweep — was unreliable (run b1408864 found
only 17 of an estimated 23+ stale edges). This module replaces it with a
deterministic direct-lookup sweep, mirroring the deterministic-lookup-over-
semantic-search precedent set by task 1680 (count_memories_by_metadata) and
task 2107 (degenerate_task_node_sweep).

Design decisions (captured in plan.json):

- extract_snapshot_edge_task_ids returns the empty set for pure count-only
  snapshots with no specific task-id reference (e.g. "There are 8 tasks in
  progress", "1505 done / 148 cancelled") — these are OUT OF SCOPE,
  stale-by-design audit trail per Snapshot Discipline, and must never be
  invalidated by this sweep.
- extract_snapshot_edge_task_ids anchors the status marker directly to its
  own task reference (INDIVIDUAL_SNAPSHOT_RE) rather than gating on
  whole-fact marker presence, so an incidental status word describing
  something else in the same fact (e.g. "Task 142 landed on the active
  branch" — the active BRANCH, not task 142) is never wrongly attributed to
  the reference. (amendment, reviewer_comprehensive precision finding, task
  2613)
- Invalidate-only-on-positively-terminal: an unknown/missing/still-active
  status for a referenced id never triggers invalidation (fail-safe,
  mirrors flag_dedup.filter_terminal_metadata_flags) — a transient census
  hiccup can only under-invalidate (self-heals next cycle), never wrongly
  retire a valid edge.
- An aggregate snapshot edge ("the active pending tasks are [A, B, C]") is
  invalidated as a whole the moment ANY one referenced id is now terminal —
  the snapshot as asserted no longer holds.
- Best-effort throughout (modelled on
  ``degenerate_task_node_sweep.sweep_degenerate_task_nodes``): every caught
  failure is logged and tallied into ``stats['errors']`` rather than
  raised. ``get_all_valid_edges``/``get_statuses`` are single bulk calls the
  rest of the cycle depends on, so a failure there ends this cycle's sweep
  early with the stats gathered so far (self-heals next cycle). A per-edge
  ``update_edge`` failure, by contrast, does NOT abort the loop — the
  remaining stale edges are still attempted, exactly like the per-item
  find/delete calls in ``sweep_degenerate_task_nodes``.
- 'blocked' is now a recognized non-terminal snapshot status marker. Its
  absence made the gate short-circuit
  (``if not SNAPSHOT_STATUS_RE.search(fact): return set()``) fire before
  the ``INACTIVE_TASK_STATUSES`` cross-reference could ever run, so
  blocked-worded edges were structurally invisible regardless of the
  referenced task's real status (task 2885 repro: 4 stale edges survived a
  blocked->done transition; scanned=5868/invalidated=0). (amendment, task
  3042)
- The single ``_STATUS_MARKER_ALT`` constant now feeds the gate, the
  anchored individual regex, and the phrase regex, so the gate can never
  again be narrower than the anchored matchers — that drift (a marker
  present in one but not the other) is the exact bug class task 3042
  fixes. (amendment, task 3042)
- 'blocked' is matched only in copula/article form by the anchored
  individual matcher, because unlike the adjective-only markers it is
  also a common transitive verb ('Task 5 blocked the merge queue' is a
  permanently-true historical fact, not a status snapshot). Same hazard
  and same remedy as task 2824's transitive-verb caveat on
  ``task_filter.PRESENT_TENSE_COMPLETION_RE``. (amendment,
  reviewer_comprehensive precision finding, task 3042)
- EVERY extraction path anchors its marker adjacently to whatever the
  marker describes; no path may rest on the whole-fact gate alone. The
  gate only decides whether a fact is worth considering at all — it
  matches a marker ANYWHERE in the fact, so it can never be the thing
  separating a status snapshot from a historical fact. The aggregate
  list path was left resting on the gate and had to be re-anchored once
  'blocked' widened it (LIST_INTRODUCER_RE). Any future marker added to
  ``_STATUS_MARKER_ALT`` must be checked against every path, not just
  the individual form. (amendment, reviewer_comprehensive precision
  finding, task 3042)
- Two anchoring paths with an explicit precision contract: closed-class
  connective only (copula / article — NOT the preposition 'in', which
  would let the marker bind to a noun the task is merely located in) for
  the general individual form (``INDIVIDUAL_SNAPSHOT_RE``); a lazy
  ``{0,3}``-word open-class gap and an 'in' preposition permitted ONLY
  when the marker is immediately followed by the literal noun 'status'
  (``SNAPSHOT_STATUS_PHRASE_RE``). That status-noun requirement is what
  keeps 'Task N is in the active branch' / 'is in the pending merge
  queue' out while still reaching 'Task N is in a blocked status ...'.
  (amendment, task 3042)
- The status-noun requirement narrows binding ambiguity but does not
  eliminate it, and the code must not claim otherwise. Two shapes were
  identified: a following head noun ('...in the pending status report' —
  a status REPORT), now excluded by requiring the span to end its noun
  phrase; and a gap-internal nominal subject ('Task 5 depends on work in
  blocked status' — the WORK is blocked), which still binds and is
  pinned by a documented-behaviour test rather than claimed away. See
  SNAPSHOT_STATUS_PHRASE_RE. (amendment, reviewer_comprehensive
  suggestion, task 3042)
- 'blocked' is deliberately NOT added to ``INACTIVE_TASK_STATUSES`` — that
  frozenset stays ``{done, cancelled}`` — which is what preserves
  invalidate-only-on-positively-terminal (a genuinely-still-blocked task is
  never selected). (amendment, task 3042)
- Known residual (task 3042): a blocked-status assertion that neither uses
  a closed-class connective nor the literal 'status' noun (e.g. "Task N
  remains parked awaiting adjudication") is still not extracted. This is
  the deliberate fail-safe direction — under-selection self-heals or is
  caught by Stage 2, whereas over-selection would wrongly retire true
  facts.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime

from fused_memory.reconciliation.task_filter import INACTIVE_TASK_STATUSES, TASK_REF_RE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# extract_snapshot_edge_task_ids — pure lexical extraction
# --------------------------------------------------------------------------- #

# Status markers this sweep treats as non-terminal snapshot assertions,
# split by part of speech. 'active'/'pending'/'in progress' are adjectives
# only; 'blocked' is ALSO a common transitive verb and so is matched only in
# copula/article form — see the transitive-verb caveat in the module
# docstring (task 3042), which mirrors task 2824's on
# task_filter.PRESENT_TENSE_COMPLETION_RE.
_ADJECTIVE_MARKER_ALT = r'(?:active|pending|in[-\s]?progress)'
_TRANSITIVE_MARKER_ALT = r'(?:blocked)'

# Copula/linking verbs admitted as a closed-class connective. Shared by both
# arms of INDIVIDUAL_SNAPSHOT_RE — optional for the adjective arm, mandatory
# for the transitive-capable arm.
_COPULA_ALT = r'(?:is|are|was|were|remains?)'

# Closed-class adverbs permitted between the copula and the marker, so
# 'Task 5 is currently blocked' / 'is still blocked' match as readily as
# 'Task 5 is blocked'. Mirrors the '(?:been|already|now|fully)\s+' slot in
# task_filter.PRESENT_TENSE_COMPLETION_RE. Deliberately an explicit
# closed-class list rather than a '\w+' gap: an open-class gap here would
# re-admit the very readings the two arms exist to exclude.
_ADVERB_ALT = r'(?:(?:currently|still|now|already)\s+)*'

# Tokens SNAPSHOT_STATUS_PHRASE_RE's open-class gap may not absorb, because
# each inverts the assertion instead of merely padding it: negation ('is not
# in blocked status') and past-exit qualifiers ('is no longer / was
# previously / was briefly in blocked status'). Written as a negative
# lookahead applied per gap word. The individual form needs no equivalent —
# its connective is closed-class, so these tokens simply are not in
# _ADVERB_ALT and never matched in the first place. See residual (2) on
# SNAPSHOT_STATUS_PHRASE_RE.
_GAP_EXCLUDED_ALT = (
    r'(?!(?:not|no|longer|never|previously|formerly|once|briefly|'
    r'until|after|before)\b)'
)

# Tokens SNAPSHOT_STATUS_PHRASE_RE's open-class gap may not absorb because
# they START ANOTHER TASK REFERENCE. Kept separate from _GAP_EXCLUDED_ALT
# above: that list is about words which INVERT the assertion, this one is
# about words which RE-SUBJECT it, and the two want independent rationale.
#
# Without this, the gap could span an intervening reference and bind the
# phrase to the wrong task — and because re.finditer is non-overlapping, the
# first (wrong) match consumed the inner reference so the CORRECT id was
# never extracted either: 'Task 5 blocks task 9 in blocked status' yielded
# {5} (over-selection on 5, silent under-selection on 9) where it must yield
# {9}. The token vocabulary mirrors TASK_REF_RE's own ('task'/'df'/digits);
# its third alternative, a literal '#', needs no exclusion because the gap's
# '\w+' cannot match '#' in the first place.
#
# The '\w*\b' tail makes this DELIBERATELY WIDER than TASK_REF_RE: 'tasks'
# is refused as a gap word even though the plural is not itself a reference
# (TASK_REF_RE anchors '\btask\b'). The cost is under-selection on shapes
# like 'Task 5 blocks tasks 9 in blocked status' (now set(), previously the
# wrong {5}) — the fail-safe direction this module prefers throughout, and
# pinned by a test rather than left implicit. (amendment,
# reviewer_comprehensive correctness-precision finding, task 3042)
_GAP_NO_TASK_REF = r'(?!(?:task|df|\d)\w*\b)'

# The union of both marker classes. Derived from the two constants above
# rather than hand-written, so the gate (SNAPSHOT_STATUS_RE) and the
# anchored matchers can never drift apart again — before task 3042 they were
# hand-copied duplicates, and a marker present in one but not the other
# silently suppressed a whole class of edges (the gate short-circuits below,
# before the INACTIVE_TASK_STATUSES cross-reference in
# select_stale_status_snapshot_edges ever runs). Used ONLY by the gate and
# by the status-noun-anchored SNAPSHOT_STATUS_PHRASE_RE — both of which are
# safe against the transitive-verb reading (the gate merely admits the fact
# for consideration; the phrase form requires the literal noun 'status'
# after the marker).
_STATUS_MARKER_ALT = r'(?:' + _ADJECTIVE_MARKER_ALT + r'|' + _TRANSITIVE_MARKER_ALT + r')'

# Gate: a status-snapshot edge must assert one of these non-terminal
# statuses somewhere in its fact text, else it is not a status-snapshot edge
# this sweep concerns itself with at all (e.g. 'Task 5 is done' — already
# terminal at write time, not what this sweep targets).
SNAPSHOT_STATUS_RE: re.Pattern[str] = re.compile(
    r'\b' + _STATUS_MARKER_ALT + r'\b',
    re.IGNORECASE,
)

# NOTE (amendment, reviewer_comprehensive correctness-recall finding, task
# 2613): individual-form extraction used to run against a COUNT_QUANTITY_RE-
# stripped copy of the WHOLE fact, so a verb-less snapshot like 'Task 5 in
# progress' or 'Task 5 pending' had its own digit stripped out first (it
# matches '\d+\s+in[-\s]?progress' / '\d+\s+pending') before the id
# extraction ever ran — a silent false negative. INDIVIDUAL_SNAPSHOT_RE
# below anchors the status marker directly to its own task reference
# instead (see its docstring), so COUNT_QUANTITY_RE is now only ever
# applied to an already list-scoped aggregate segment, never to the whole
# fact — see the aggregate-form loop in extract_snapshot_edge_task_ids.

# Strips 'N <count-noun>' spans (e.g. '8 tasks', '148 cancelled', '3 pending')
# from an aggregate list segment before bare-digit extraction, so an
# embedded count phrase (e.g. '...tasks: 142, 148, and 200 total remain')
# never contributes a spurious id.
COUNT_QUANTITY_RE: re.Pattern[str] = re.compile(
    r'\b\d+\s+(?:tasks?|active|pending|in[-\s]?progress|done|cancell?ed|'
    r'blocked|deferred|review|total|merge[-\s]?deferred)\b',
    re.IGNORECASE,
)

# Anchors the status marker directly to its own task reference, so an
# incidental status word elsewhere in the fact is never attributed to it
# ('Task 142 landed on the active branch' -> set()). Built from
# TASK_REF_RE.pattern rather than a hand-copied duplicate so the two stay in
# sync if the shared task-reference grammar changes.
#
# Two arms, split by the marker's part of speech: the adjective arm takes an
# OPTIONAL copula (so verb-less 'Task 5 pending' matches), the
# transitive-capable arm requires a MANDATORY copula/article (so 'Task 5 is
# blocked' matches but the bare verb 'Task 5 blocked the merge queue' does
# not). Only closed-class function words — copula, adverb, article — may sit
# between reference and marker; notably NOT the preposition 'in', which would
# bind the marker to a following noun. Full rationale for all of this,
# including the reverted 'in'/'the' widening, lives in the module docstring
# (tasks 2613 / 3042) — do not restate it here.
INDIVIDUAL_SNAPSHOT_RE: re.Pattern[str] = re.compile(
    TASK_REF_RE.pattern
    + r'(?:'
    # adjective arm — optional copula (task 2613 behaviour, unchanged)
    + r'\s*' + _COPULA_ALT + r'?\s*' + _ADVERB_ALT + r'(?:an?\s+)?'
    + _ADJECTIVE_MARKER_ALT
    + r'|'
    # transitive-capable arm — copula or article is MANDATORY
    + r'\s+(?:' + _COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:an?\s+)?|an?\s+)'
    + _TRANSITIVE_MARKER_ALT
    + r')\b',
    re.IGNORECASE,
)

# The permissive path: an open-class gap of up to 3 words between the task
# reference and the preposition 'in' (e.g. 'is deliberately parked in ...'),
# admitted ONLY when the marker is immediately followed by the literal noun
# 'status'. The gap is lazy ({0,3}?) and bounded, and cannot cross a clause
# boundary — '.', ',' and ';' are not \w and the gap's leading \s+ cannot
# absorb them — so 'Task 5 is done. Task 9 is in blocked status' anchors
# only to task 9.
#
# PRECISION RESIDUAL (amendment, reviewer_comprehensive suggestion, task
# 3042). An earlier version of this comment claimed the 'status' noun meant
# the open-class gap "costs no precision". That was an overstatement, and a
# future maintainer would have relied on it. The status-noun requirement
# removes most but not all binding ambiguity. Four shapes have been
# identified; (1), (2) and (4) are now EXCLUDED, and (3) alone still binds
# and is pinned by a documented-behaviour test rather than claimed away:
#
# (1) A following HEAD NOUN, where 'status' is a modifier rather than the
#     phrase head: 'Task 142 is tracked in the pending status report' — a
#     pending status REPORT, not a pending task. This one IS now excluded,
#     by requiring the marker+status span to end the noun phrase (the
#     trailing lookahead below: end of string, punctuation, or a
#     preposition/subordinator such as 'due'/'under'/'since'). 'status
#     report' is common phrasing in this repo's memory corpus, so this was
#     worth closing rather than merely documenting.
# (2) NEGATION / PAST-EXIT qualifiers absorbed by the gap, which invert
#     the assertion outright: 'Task N is no longer in blocked status',
#     'was previously in blocked status', 'is not in blocked status',
#     'was briefly in blocked status, then merged'. Each is a
#     permanently-true HISTORICAL fact — precisely the blocked->unblocked
#     transition that writes such an edge in the first place, i.e. the same
#     transition class the task-2885 repro is about, so not exotic
#     phrasing. This is now EXCLUDED via _GAP_EXCLUDED_ALT below.
#     Worth noting the asymmetry that made this phrase-form-only: the
#     individual form's closed-class connective already refuses the same
#     semantics for free ('Task N is no longer blocked' -> set()), because
#     'not'/'no'/'longer'/'never' were never in _ADVERB_ALT. An open-class
#     gap forfeits that protection, which is the standing cost of the
#     permissive path. (amendment, reviewer_comprehensive
#     correctness-precision finding, task 3042)
# (3) A gap-internal NOMINAL that is the real subject of the phrase:
#     'Task 5 depends on work in blocked status' — the WORK is blocked, not
#     task 5. This still yields {5} and is NOT fixed here: distinguishing it
#     needs to know that 'work' is the subject, which is beyond lexical
#     matching at this altitude. Documented as known behaviour (with a test
#     pinning it) rather than claimed away, so the trade-off stays visible.
#     Widening the gap beyond {0,3} words would make this strictly worse.
# (4) An intervening TASK REFERENCE absorbed by the gap, which RE-SUBJECTS
#     the assertion onto a different task: 'Task 5 blocks task 9 in blocked
#     status' and 'Task 5 supersedes task 9 in pending status' both yielded
#     {5}. Strictly worse than a plain over-selection, because re.finditer
#     is non-overlapping — the wrong match consumed the inner 'task 9', so
#     the CORRECT id was silently never extracted either. One match, both
#     error directions at once. Now EXCLUDED via _GAP_NO_TASK_REF above.
#     The bug needed BOTH a <=3-word gap and a word-initial reference token:
#     'df 9' was affected identically, '#9' never was (the gap's '\w+'
#     cannot match '#'), and a 4+-word gap ('Task 5 depends on task 9 in
#     active status') was already out of reach. (amendment,
#     reviewer_comprehensive correctness-precision finding, task 3042)
SNAPSHOT_STATUS_PHRASE_RE: re.Pattern[str] = re.compile(
    TASK_REF_RE.pattern
    # The gap is open-class BUT may absorb neither negation / past-exit
    # qualifiers (residual (2) — they INVERT the assertion) nor the opening
    # token of another task reference (residual (4) — it RE-SUBJECTS the
    # assertion). Both are per-gap-word negative lookaheads.
    + r'(?:\s+' + _GAP_EXCLUDED_ALT + _GAP_NO_TASK_REF + r'\w+){0,3}?'
    + r'\s+in\s+(?:an?\s+|the\s+)?'
    + _STATUS_MARKER_ALT + r'\s+status\b'
    # the span must END the noun phrase — else 'status' is a modifier of a
    # following head noun ('status report'), not the phrase head
    + r'(?=\s*[.,;:!?)\]]|\s*$|\s+(?:due|under|pending|since|because|'
    + r'awaiting|after|before|while|until|per|as|with|for|on|from|and|or)\b)',
    re.IGNORECASE,
)

# Detects the start of an aggregate list segment: '<marker> tasks [' or
# '<marker> tasks are [' or '<marker> tasks:' or '<marker> tasks are:'. The
# 'open' group records which delimiter opened the segment ('[' vs ':') so
# _list_segment can decide how to find the segment's end.
#
# NOTE (amendment, reviewer_comprehensive correctness-precision finding,
# task 3042): this used to be a bare r'\btasks?\b...' with no marker of its
# own, leaving the whole aggregate path anchored to nothing but the
# whole-fact gate (SNAPSHOT_STATUS_RE). Widening that gate with 'blocked'
# therefore routed straight around the transitive-verb hardening built into
# INDIVIDUAL_SNAPSHOT_RE: 'Task 5 blocked these tasks: 142, 148' and 'Task 5
# blocked the merge of tasks [142, 148]' both yielded {142, 148} (each
# returned set() before 'blocked' joined the gate, so the gate was the only
# thing holding them back). Those are permanently-true historical facts, so
# the sweep would retire them the moment 142 or 148 went terminal — the
# over-selection direction the module docstring forbids.
#
# The marker is now required IMMEDIATELY before the list noun, which is the
# same anchoring principle the individual and phrase forms already use:
# adjacency to the thing being described, not mere co-occurrence somewhere
# in the fact. 'The active pending tasks are [...]', 'active/in-progress
# tasks: ...' and 'Blocked tasks: ...' all still match; the two facts above
# no longer do, because the intervening open-class words ('these', 'the
# merge of') break adjacency.
#
# Residual (deliberate, both in the fail-safe direction):
# - Under-selection: a genuine aggregate whose marker is not adjacent to the
#   list noun (e.g. 'the tasks in the merge queue are [142, 148] and remain
#   blocked') is no longer extracted. Consistent with this module's stated
#   preference — under-selection self-heals next cycle or is caught by
#   Stage 2, over-selection wrongly retires true facts.
# - Ambiguity: the exact adjacency 'Task 5 blocked tasks: 142, 148' (no
#   intervening word) still matches, and is genuinely ambiguous between the
#   transitive-verb reading and the status-list reading 'Blocked tasks:
#   [...]' — a shape the sweep must keep. Requiring adjacency narrows the
#   hazard to that one collision rather than every '... blocked ... tasks:'
#   fact.
LIST_INTRODUCER_RE: re.Pattern[str] = re.compile(
    r'\b' + _STATUS_MARKER_ALT + r'\s+tasks?\b\s*(?:are|is|were)?\s*(?P<open>[:\[])',
    re.IGNORECASE,
)

# Clause-ish boundary used to close a colon-introduced list segment (no
# closing delimiter of its own) and as a fallback for an unterminated
# bracket segment. Mirrors task_filter._CLAUSE_SPLIT_RE.
_CLAUSE_BOUNDARY_RE: re.Pattern[str] = re.compile(r'[.;\n!?]')

# Bare digit token — used only within an already-detected list segment, so
# a bare '\d+' never contributes a candidate id outside that scoped context
# (dates/commit-hashes/ports in ordinary prose are never list segments).
_BARE_DIGIT_RE: re.Pattern[str] = re.compile(r'\b\d+\b')


def _list_segment(text: str, start: int, open_char: str) -> str:
    """Return the list-segment substring of *text* beginning at *start*.

    A bracket-opened segment (``open_char == '['``) runs to the matching
    ``']'`` when present, else falls back to the next clause boundary / end
    of string (mirrors the colon case) rather than running away to the end
    of a long text. A colon-opened segment has no closing delimiter of its
    own, so it always runs to the next clause boundary ([.;\\n!?]) or end of
    string.
    """
    if open_char == '[':
        close = text.find(']', start)
        if close != -1:
            return text[start:close]

    boundary = _CLAUSE_BOUNDARY_RE.search(text, start)
    end = boundary.start() if boundary else len(text)
    return text[start:end]


def extract_snapshot_edge_task_ids(fact: str) -> set[int]:
    """Return the task ids *fact* asserts as active/pending/blocked/in-progress.

    Returns the empty set (never a candidate for invalidation) when:
    - *fact* contains no active/pending/blocked/in-progress status marker at
      all (e.g. 'Task 5 is done', 'Task 7 landed as merge commit'); or
    - *fact* is a pure count-only snapshot with no specific task-id
      reference (e.g. 'There are 8 tasks in progress', '1505 done / 148
      cancelled') — out of scope per Snapshot Discipline; or
    - a task id is only incidentally near an unrelated status word (e.g.
      'Task 142 landed on the active branch' — the active BRANCH, not task
      142 — see INDIVIDUAL_SNAPSHOT_RE).

    Algorithm:
      1. Gate on SNAPSHOT_STATUS_RE against the raw fact text; short-circuit
         to the empty set when absent.
      2. Individual form: extract ids via INDIVIDUAL_SNAPSHOT_RE, which
         anchors 'task N' / '#N' / 'df N' (TASK_REF_RE's own grammar)
         directly to an adjacent status marker (only an optional copula
         and/or article may sit in between — deliberately NOT the
         preposition 'in') — so an incidental status word elsewhere in
         the fact is never wrongly attributed to the reference.
      3. Status-phrase form: extract ids via SNAPSHOT_STATUS_PHRASE_RE,
         which additionally admits a bounded open-class gap (e.g. 'is
         deliberately parked in ...') between the reference and the
         marker, but only when the marker is immediately followed by the
         literal noun 'status' — the token that makes the span an
         explicit status assertion rather than an incidental mention.
      4. Aggregate form: for each detected list segment ('<marker> tasks
         are [...]' / '<marker> tasks: ...'), strip COUNT_QUANTITY_RE
         spans from that segment only (so an embedded count phrase doesn't
         contribute a spurious id) and collect its bare digit tokens. The
         status marker must sit immediately before the list noun — the
         same adjacency anchoring the individual and phrase forms use, so
         this path is not left riding on the whole-fact gate alone (see
         LIST_INTRODUCER_RE).

    Pure: no I/O, no side effects.
    """
    fact = fact or ''
    if not SNAPSHOT_STATUS_RE.search(fact):
        return set()

    ids: set[int] = {int(m.group(1)) for m in INDIVIDUAL_SNAPSHOT_RE.finditer(fact)}
    ids |= {int(m.group(1)) for m in SNAPSHOT_STATUS_PHRASE_RE.finditer(fact)}

    for intro in LIST_INTRODUCER_RE.finditer(fact):
        segment = COUNT_QUANTITY_RE.sub(' ', _list_segment(fact, intro.end(), intro.group('open')))
        ids.update(int(tok) for tok in _BARE_DIGIT_RE.findall(segment))

    return ids


# --------------------------------------------------------------------------- #
# flatten_dedup_edges
# --------------------------------------------------------------------------- #


def flatten_dedup_edges(grouped: dict[str, list[dict]]) -> list[dict]:
    """Flatten get_all_valid_edges' dict[entity_uuid, list[EdgeDict]] shape.

    ``GraphitiBackend.get_all_valid_edges``'s undirected MATCH pattern
    double-attributes each directed edge under both its source and target
    entity UUID. This flattens the grouping and dedups by ``edge['uuid']``,
    keeping the first-seen occurrence, so each edge is considered exactly
    once by the rest of the sweep.

    Pure: no I/O, no side effects.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for edges in grouped.values():
        for edge in edges:
            uuid = edge['uuid']
            if uuid in seen:
                continue
            seen.add(uuid)
            result.append(edge)
    return result


# --------------------------------------------------------------------------- #
# select_stale_status_snapshot_edges — pure decision core
# --------------------------------------------------------------------------- #


def select_stale_status_snapshot_edges(
    edges: list[dict],
    statuses: dict[str, str],
    *,
    edge_ids: dict[str, set[int]] | None = None,
) -> list[dict]:
    """Return the subset of *edges* whose asserted status is now contradicted.

    An edge is selected (stale) iff ``extract_snapshot_edge_task_ids`` returns
    at least one id AND any one of those ids resolves (via *statuses*) to a
    positively-terminal status (``INACTIVE_TASK_STATUSES`` — done/cancelled).
    An aggregate edge referencing several ids is selected as a whole the
    moment ANY single referenced id is terminal — the snapshot as asserted no
    longer holds.

    Invalidate-only-on-positively-terminal: an id absent from *statuses*, or
    mapped to any non-terminal/unknown value, never contributes to
    selection — a transient census gap or genuinely-still-active task can
    only under-select (self-heals next cycle), never wrongly select a valid
    edge.

    Args:
        edges: Candidate edges (as returned by ``flatten_dedup_edges``).
        statuses: ``{task_id_str: status_str}`` census.
        edge_ids: Optional precomputed ``{edge['uuid']: extract_snapshot_edge_task_ids(fact)}``
            mapping. A caller that already extracted ids for its own purposes
            (e.g. ``sweep_stale_status_snapshot_edges``, which needs them to
            build its candidate-id set) passes this to avoid re-running the
            extraction regex pipeline a second time per edge. When omitted
            (the default), ids are computed directly from each edge's fact —
            unchanged standalone behavior. (amendment, reviewer_comprehensive
            efficiency finding, task 2613)

    Pure: no I/O, no side effects.
    """
    selected: list[dict] = []
    for edge in edges:
        if edge_ids is not None:
            ids = edge_ids.get(edge['uuid'], set())
        else:
            ids = extract_snapshot_edge_task_ids(edge.get('fact') or '')
        if not ids:
            continue
        if any(statuses.get(str(i)) in INACTIVE_TASK_STATUSES for i in ids):
            selected.append(edge)
    return selected


# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — async orchestrator
# --------------------------------------------------------------------------- #

# Written to update_edge's agent_id — identifies this sweep's writes in the
# write journal / audit trail, mirroring degenerate_task_node_sweep's
# convention of stamping a stable, identifiable actor.
_SWEEP_AGENT_ID = 'recon-stage-memory_consolidator'


async def sweep_stale_status_snapshot_edges(
    memory_service,
    taskmaster,
    project_id: str,
    project_root: str,
    *,
    run_id: str,
    now: datetime | None = None,
    log: logging.Logger = logger,
) -> dict:
    """Enumerate valid status-snapshot edges and invalidate the stale ones.

    Enumerates ALL currently-valid Graphiti edges for *project_id* via
    ``memory_service.graphiti.get_all_valid_edges`` (a deterministic bulk
    query — never the LLM's semantic search), extracts the specific task ids
    each edge asserts as active/pending/blocked/in-progress, cross-references those
    ids' CURRENT status via ``taskmaster.get_statuses`` (a direct status
    lookup, not semantic search), and invalidates
    (``memory_service.update_edge(..., invalid_at=...)``) every edge whose
    asserted status is now contradicted by a terminal (done/cancelled) task.

    Args:
        memory_service: Object exposing ``.graphiti.get_all_valid_edges`` and
            ``.update_edge``.
        taskmaster: Object exposing ``.get_statuses``. A falsy value (e.g.
            unavailable backend) short-circuits to all-zero stats.
        project_id: Graphiti group_id to enumerate and invalidate within.
        project_root: Taskmaster project root for the status lookup. A
            falsy value short-circuits to all-zero stats.
        run_id: Reconciliation run id, threaded through as update_edge's
            causation_id.
        now: Invalidation timestamp; defaults to ``datetime.now(UTC)``.
        log: Logger to use (default: this module's logger).

    Best-effort (mirrors
    ``degenerate_task_node_sweep.sweep_degenerate_task_nodes``): a transient
    backend error enumerating (``get_all_valid_edges``), cross-referencing
    (``get_statuses``), or invalidating (``update_edge``) is caught, logged,
    and tallied into ``stats['errors']``. An enumeration or cross-reference
    failure aborts the rest of this cycle's sweep (there is nothing left to
    act on without them) and returns the stats gathered so far; a per-edge
    ``update_edge`` failure does NOT abort the loop — the remaining stale
    edges are still attempted. ``asyncio.CancelledError``/``KeyboardInterrupt``/
    ``SystemExit`` are re-raised unchanged (never swallowed as best-effort).

    Returns:
        dict with int counts: ``scanned`` (edges enumerated after dedup),
        ``candidate_edges`` (edges selected as stale by
        ``select_stale_status_snapshot_edges``), ``invalidated`` (successful
        update_edge calls), ``errors`` (caught enumerate/cross-reference/
        invalidate failures).

    Empty *taskmaster*/*project_root* short-circuits to all-zero stats with
    no backend calls. When enumeration yields no candidate ids at all,
    ``taskmaster.get_statuses`` is never called.
    """
    stats = {'scanned': 0, 'candidate_edges': 0, 'invalidated': 0, 'errors': 0}

    if not taskmaster or not project_root:
        return stats

    try:
        grouped = await memory_service.graphiti.get_all_valid_edges(group_id=project_id)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'stale_status_snapshot_edge_sweep: get_all_valid_edges failed for group_id=%s',
            project_id,
        )
        stats['errors'] += 1
        return stats

    edges = flatten_dedup_edges(grouped)
    stats['scanned'] = len(edges)

    # Extract each edge's ids exactly once — reused below both to build
    # candidate_ids and (via select_stale_status_snapshot_edges's edge_ids
    # kwarg) for the final selection, instead of re-running the extraction
    # regex pipeline a second time per edge. (amendment,
    # reviewer_comprehensive efficiency finding, task 2613)
    edge_ids: dict[str, set[int]] = {
        edge['uuid']: extract_snapshot_edge_task_ids(edge.get('fact') or '') for edge in edges
    }

    candidate_ids: set[int] = set()
    for ids in edge_ids.values():
        candidate_ids |= ids

    if not candidate_ids:
        return stats

    try:
        statuses = await taskmaster.get_statuses(
            project_root, ids=[str(i) for i in sorted(candidate_ids)],
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'stale_status_snapshot_edge_sweep: get_statuses failed for project_root=%s',
            project_root,
        )
        stats['errors'] += 1
        return stats

    stale = select_stale_status_snapshot_edges(edges, statuses, edge_ids=edge_ids)
    stats['candidate_edges'] = len(stale)

    invalidate_at = now or datetime.now(UTC)
    for edge in stale:
        try:
            await memory_service.update_edge(
                edge['uuid'],
                invalid_at=invalidate_at,
                project_id=project_id,
                agent_id=_SWEEP_AGENT_ID,
                causation_id=run_id,
            )
            stats['invalidated'] += 1
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            log.exception(
                'stale_status_snapshot_edge_sweep: update_edge failed for uuid=%s',
                edge['uuid'],
            )
            stats['errors'] += 1

    return stats
