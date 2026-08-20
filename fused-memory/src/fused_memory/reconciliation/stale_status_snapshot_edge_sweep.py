"""Stale task-status snapshot Graphiti edge sweep — task 2613.

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that keeps VALID
(``invalid_at IS NULL``) task-status-snapshot Graphiti edges in sync with the
tasks they describe. A status-snapshot edge asserts that some task_id is
currently active/pending/blocked/stalled/in-progress; once that task reaches
a terminal status (done/cancelled), the edge is stale but nothing retires
it. The prior
approach — a manual LLM `search` sweep — was unreliable (run b1408864 found
only 17 of an estimated 23+ stale edges). This module replaces it with a
deterministic direct-lookup sweep, mirroring the deterministic-lookup-over-
semantic-search precedent set by task 1680 (count_memories_by_metadata) and
task 2107 (degenerate_task_node_sweep).

Extraction paths. Each anchors its status marker directly to what it
describes rather than resting on the whole-fact gate alone — see the
marker-addition rule below, and ``extract_snapshot_edge_task_ids`` for how
the paths compose. Full per-path rationale lives at each regex constant,
not here:

- ``INDIVIDUAL_SNAPSHOT_RE`` — 'Task N is/was <marker>' / 'Task N <marker>'.
- ``GENITIVE_STATUS_RE`` — "Task N's status is <marker>" (task 3079).
- ``SNAPSHOT_STATUS_PHRASE_RE`` — 'Task N ... in <marker> status'.
- ``PLURAL_ENUM_SNAPSHOT_RE`` — 'Tasks A, B and C are <marker>' (task 3079).
- ``LIST_INTRODUCER_RE`` (+ aggregate list segments) — '<marker> tasks:
  [...]' / '<marker> tasks are [...]'.

Standing invariants — these are cross-cutting and apply to every path
above; re-check all of them before changing any single path. (Per-path
detail, examples, and residuals are documented once, at the owning
constant — do not restate them here.)

- Fail-safe direction, throughout the module: under-selection self-heals
  next cycle (or is caught by Stage 2); over-selection is unrecoverable,
  since an invalidated Graphiti edge is never re-validated. Every
  extraction path is deliberately biased toward missing a snapshot over
  wrongly retiring one.
- Marker-addition rule: EVERY extraction path must anchor its marker
  adjacently to whatever it describes; the whole-fact gate
  (``SNAPSHOT_STATUS_RE``) only decides whether a fact is worth
  considering at all — it matches a marker ANYWHERE in the fact, so it can
  never be what separates a status snapshot from an unrelated fact. The
  aggregate list path was once left resting on the gate alone and had to
  be re-anchored (``LIST_INTRODUCER_RE``) once 'blocked' widened the gate
  — task 2885 repro: 4 stale edges survived a blocked->done transition
  (scanned=5868, invalidated=0) because the gate short-circuited before
  the ``INACTIVE_TASK_STATUSES`` cross-reference ever ran. Any marker
  added to ``_STATUS_MARKER_ALT`` must therefore be re-checked against
  every path individually, not just the individual form —
  ``COUNT_QUANTITY_RE`` is the one existing consumer that is NOT safe by
  default under such a widening (see its own comment). (amendment,
  reviewer_comprehensive precision finding, task 3042)
- 'blocked' and 'stalled' are also common transitive verbs ('Task 5
  blocked the merge queue' is a permanently-true historical fact, not a
  status snapshot), so both are matched only in copula/article form, never
  as a bare adjective — see ``_TRANSITIVE_MARKER_ALT``. Same hazard and
  remedy as task 2824's transitive-verb caveat on
  ``task_filter.PRESENT_TENSE_COMPLETION_RE``.
- ``INDIVIDUAL_SNAPSHOT_RE`` admits only a closed-class connective (copula
  / article — never the preposition 'in', which would let the marker bind
  to a noun the task is merely located in); ``SNAPSHOT_STATUS_PHRASE_RE``
  is the one path that DOES permit an open-class gap and the 'in'
  preposition, and only because it additionally requires the marker be
  immediately followed by the literal noun 'status'. That status-noun
  requirement is what keeps 'Task N is in the active branch' out while
  still reaching 'Task N is in a blocked status ...'. (amendment, task
  3042)
- Invalidate-only-on-positively-terminal, and an aggregate/plural edge is
  invalidated as a whole the moment any ONE referenced id is terminal —
  see ``select_stale_status_snapshot_edges``.
- The sweep is best-effort throughout (mirrors
  ``degenerate_task_node_sweep``): an edge-enumeration
  (``get_all_valid_edges``) or status cross-reference (``get_statuses``)
  failure ends this cycle's sweep early (self-heals next cycle); a
  per-edge invalidation failure does not — see
  ``sweep_stale_status_snapshot_edges``.
- 'blocked' is deliberately NOT added to ``INACTIVE_TASK_STATUSES`` (stays
  ``{done, cancelled}``), which is what preserves
  invalidate-only-on-positively-terminal for a genuinely-still-blocked
  task.

Known residuals (deliberate; all fail-safe/under-selection unless noted)
— shape-by-shape detail lives at the matching constant:

- task 3042: a blocked-status assertion using neither a closed-class
  connective nor the literal 'status' noun (e.g. "Task N remains parked
  awaiting adjudication") is not extracted.
- task 3042: ``SNAPSHOT_STATUS_PHRASE_RE``'s status-noun requirement
  narrows but does not eliminate binding ambiguity — see its PRECISION
  RESIDUAL comment for the one surviving shape.
- task 3079: the genitive path requires the literal noun 'status', so
  "Task N's state is pending" is not extracted.
- task 3079: ``PLURAL_ENUM_SNAPSHOT_RE``'s preposition guard has one
  residual pointed the WRONG way (an unlisted preposition over-selects)
  and one fail-safe residual (a genuine subject-position enumeration
  sharing a clause with a listed preposition is missed) — see
  ``_ENUM_PREP_WORDS``. The fail-safe one is MEASURED, not open: zero
  cost on the live corpus (task 3949).
- task 4149: ``_CLAUSE_BREAK_CHARS``'s ';' and '?' are unconditional
  breaks (only '.' gets the occurrence-level flanking test), so one
  residual is pointed the WRONG way: a '?' inside a URL query string or a
  ';' inside a path/branch name truncates the backward scan past the
  governing preposition and OVER-selects — see ``_CLAUSE_BREAK_CHARS``.

Two hypotheses were investigated and RULED OUT for the task-2613 miss
rate; do not re-open them:

1. Edge traversal/scope: ``GraphitiBackend.get_all_valid_edges`` runs an
   unfiltered ``MATCH`` (no entity-type, label, or episode filter),
   reproduced end-to-end scanning every missed edge and dropping it at
   extraction — every gap was lexical, not a query-scope gap.

   AMENDED, task 4340 (2026-08-17).  The lexical findings above STAND and
   are unaffected; what does not stand is the scope of the ruling-out.
   Those grounds covered query FILTERS only and were silent about
   server-side TRUNCATION.  Task 4340 measured a FalkorDB
   ``RESULTSET_SIZE`` cap silently truncating that very query to roughly
   HALF the valid-edge corpus on dark_factory (exact figures: the
   RESULT-SET CAP AUDIT block in ``backends/graphiti_client.py``, which is
   the one place they are recorded).  So at the time of the task-2613
   investigation this sweep saw about half the edges, and the miss rate was
   computed against a truncated denominator.  ``get_all_valid_edges`` is
   paginated as of task 4340 and the truncation is gone, but the RATE has
   not been recomputed: a re-measurement against the now-complete corpus is
   warranted, and the residuals list above is calibrated against the old
   figure.  Filed as ticket tkt_0RSJP92VQNATQB0FSR20YMXGW8 (a TICKET id,
   not a task id — the curator resolves it to a task asynchronously).  Do
   not re-open the LEXICAL hypothesis; do not treat the old rate as
   measured on a whole corpus.
2. A shared blind spot with ``task_count_verification``: that function is
   an aggregate census-vs-tree consistency check (``task_filter``), not a
   per-task edge sweep at all — its 'healthy' report was correct, not
   blind.

Why a regex in this module gets a performance test at all:
``sweep_stale_status_snapshot_edges`` calls
``extract_snapshot_edge_task_ids`` once per valid edge from an UNGUARDED
dict comprehension with no per-edge timeout, over the whole group's edge
set (tens of thousands of edges; current figures in the RESULT-SET CAP
AUDIT block in ``backends/graphiti_client.py``).  Extractor cost is
therefore a whole-cycle LIVENESS property — one pathological fact stalls
the entire reconciliation cycle — not a micro-optimisation. (amendment,
task 3079)

The figure was ~5868 in the task-3042 record; that number is consistent
with a truncated enumeration and has been corrected upward by task 4340's
live census (see the amendment to ruled-out hypothesis 1 above). The
LIVENESS argument gets STRONGER, not weaker: with the truncation removed
the per-edge extractor now runs over roughly twice as many edges per
cycle, so the per-edge cost this test guards matters more than the
original number implied, not less.  The read that feeds it was timed
2026-08-18 at ~3.3 s per full enumeration on dark_factory (~3.7 s on
reify) — bounded, and roughly +2.6 s per cycle over the old truncated
read; see the MEASURED COST section of that same audit block, so this
claim rests on a number rather than on an estimate. (amendment, task 4340)
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime

from fused_memory.reconciliation.task_filter import (
    INACTIVE_TASK_STATUSES,
    STRICT_CLAUSE_BOUNDARY_RE,
    TASK_REF_RE,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# extract_snapshot_edge_task_ids — pure lexical extraction
# --------------------------------------------------------------------------- #

# Status markers this sweep treats as non-terminal snapshot assertions,
# split by part of speech. 'active'/'pending'/'in progress' are adjectives
# only; 'blocked' and 'stalled' are ALSO common transitive verbs and so are
# matched only in copula/article form — see the transitive-verb caveat in
# the module docstring (tasks 3042, 3079), which mirrors task 2824's on
# task_filter.PRESENT_TENSE_COMPLETION_RE.
_ADJECTIVE_MARKER_ALT = r'(?:active|pending|in[-\s]?progress)'
_TRANSITIVE_MARKER_ALT = r'(?:blocked|stalled)'

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

# An optional hyphenated-compound modifier immediately before the marker, so
# 'pairwise-stalled' anchors to its copula exactly as bare 'stalled' does
# ('Tasks A, B and C are pairwise-stalled' — the finding's verbatim fact).
#
# Bounded to a single hyphenated token: '\w+' cannot match whitespace or
# punctuation, so this admits one compound modifier and nothing else. Were it
# allowed to span whitespace it would additionally re-admit the multi-word
# readings the mandatory-copula arms exist to exclude ('Tasks A and B are
# awaiting a blocked merge' — the MERGE is blocked).
#
# That bound is on the gap's WIDTH, NOT on its lexical class: 'un-', 'non-',
# 'not-', 'previously-' are each one '\w+-' token, so an unguarded prefix
# re-admits the negation / past-exit readings _ADVERB_ALT's closed-class
# discipline refuses for free on the bare marker:
#
#     'Task 5 is un-blocked.'                -> {5}
#     'Task 5 is previously-blocked.'        -> {5}
#     "Task 5's status is un-blocked."       -> {5}
#     'Tasks 1020 and 1030 are un-blocked.'  -> {1020, 1030}
#
# Each is a permanently-true HISTORICAL/negated assertion, so the sweep would
# call update_edge(invalid_at=...) and retire a still-true edge the moment any
# referenced id went terminal — the over-selection direction the module
# docstring forbids, and unrecoverable (an invalidated Graphiti edge is not
# re-validated next cycle).
#
# The remedy mirrors _GAP_EXCLUDED_ALT exactly: keep the prefix open-class
# (the observed modifier vocabulary — 'pairwise-', 'merge-', 'self-',
# 'auto-' — is not closed, and an unlisted-but-innocent modifier should not
# silently drop a genuine snapshot) but subtract the closed class of
# inverting prefixes with a negative lookahead. The lookahead sits BEFORE the
# optional group, so it is evaluated whether or not a prefix is present; no
# marker begins with one of these tokens, so the prefix-less shape ('is
# blocked') is unaffected.
#
# The privative 'in-' belongs to the same subtracted class but cannot be
# spelled as a bare 'in' entry in the alternation above: the lookahead is
# evaluated at the marker's own start position, and 'in[-\s]?progress' is
# itself a MARKER, so 'in' in the main list would refuse 'Task 5 is
# in-progress' along with 'Task 5 is in-active'. It gets a second, narrower
# lookahead that fires only when 'in-' is NOT the head of 'in-progress'.
# The spaced spelling ('is in progress') is untouched either way, since this
# lookahead requires a literal hyphen. (amendment, reviewer_comprehensive
# correctness-precision finding, task 3079)
#
# Used only in the adjacency-anchored arms (individual, genitive, plural
# enumeration); the phrase form needs no equivalent, since its marker is
# pinned on the far side by the literal noun 'status'. (task 3079)
_COMPOUND_PREFIX = (
    r'(?!(?:un|non|not|never|no|previously|formerly|once|briefly|de|dis)-)'
    r'(?!in-(?!progress\b))'
    r'(?:\w+-)?'
)

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
#
# This alternation must be kept in step with _STATUS_MARKER_ALT. It is the
# one _STATUS_MARKER_ALT consumer that is NOT safe by default under a marker
# widening, and so the concrete reason the module's standing marker-addition
# rule exists: the gate merely admits a fact for consideration, the phrase
# form requires the literal noun 'status' after the marker, and the list
# introducer requires adjacency to the list noun — but a marker missing HERE
# silently turns a count phrase into a task id ('Stalled tasks: 1020, 1030,
# and 3 stalled others' would yield a spurious 3). 'stalled' added for task
# 3079. Deliberately a separate literal list rather than an interpolation of
# _STATUS_MARKER_ALT: this one also carries non-marker count nouns
# ('tasks', 'done', 'total', 'deferred'), so the two sets overlap without
# either containing the other.
COUNT_QUANTITY_RE: re.Pattern[str] = re.compile(
    r'\b\d+\s+(?:tasks?|active|pending|in[-\s]?progress|done|cancell?ed|'
    r'blocked|stalled|deferred|review|total|merge[-\s]?deferred)\b',
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
    + _COMPOUND_PREFIX + _ADJECTIVE_MARKER_ALT
    + r'|'
    # transitive-capable arm — copula or article is MANDATORY
    + r'\s+(?:' + _COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:an?\s+)?|an?\s+)'
    + _COMPOUND_PREFIX + _TRANSITIVE_MARKER_ALT
    + r')\b',
    re.IGNORECASE,
)

# Genitive/possessive form: "Task N's status is <marker>" (task 3079). This
# is the canonical shape Graphiti writes for a per-task status snapshot, and
# before task 3079 EVERY path missed it while the whole-fact gate fired —
# the "matched the gate yet went undetected" symptom the finding reports:
# INDIVIDUAL_SNAPSHOT_RE's '\s*' cannot cross the intervening "'s status ";
# SNAPSHOT_STATUS_PHRASE_RE requires 'in <marker> status' (marker BEFORE the
# noun) whereas here the order is inverted ('status is <marker>'); and
# LIST_INTRODUCER_RE requires '<marker> tasks[:\[]'. So the miss rate was
# structural, not incidental.
#
# Both apostrophes are admitted: ASCII U+0027 and the typographic U+2019
# that most prose writers (and LLM writers) actually emit.
#
# Two properties worth stating because they are what keep this path cheap:
#
# - No trailing noun-phrase lookahead is needed, unlike
#   SNAPSHOT_STATUS_PHRASE_RE. Requiring the literal noun 'status' to be
#   IMMEDIATELY followed by the copula already excludes the modifier-head
#   hazard that lookahead exists for: in "Task 5's status report is pending
#   review", 'report' sits between 'status' and 'is', so nothing matches.
# - No _GAP_EXCLUDED_ALT equivalent is needed either. Reusing the
#   closed-class _ADVERB_ALT means negation and past-exit qualifiers ('is no
#   longer', 'is not', 'was previously') are refused for free, because those
#   tokens simply are not in the alternation — the same asymmetry the module
#   docstring already documents for the individual form. Only the open-class
#   gap of the phrase form has to buy that protection back explicitly.
#
# The copula is MANDATORY here, so _STATUS_MARKER_ALT (which includes the
# transitive-capable 'blocked') is safe to use whole: there is no bare-verb
# reading of "Task 5's status is blocked".
GENITIVE_STATUS_RE: re.Pattern[str] = re.compile(
    TASK_REF_RE.pattern
    + r"(?:'|’)s\s+status\s+"
    + _COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:an?\s+)?'
    + _COMPOUND_PREFIX + _STATUS_MARKER_ALT + r'\b',
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

# The enumeration tail of a plural multi-task snapshot: '1020, 1030, and
# 1031', '1020 and 1030', '1020, task 1030 and task 1031' (task 3079).
#
# Requires TWO OR MORE ids by construction (the repeated group is '+', not
# '*'), so a plural head over a single stray digit is never matched — after
# a plural noun a lone number is more likely a count than an id, and the
# well-formed singular shape already routes through INDIVIDUAL_SNAPSHOT_RE.
#
# The separator must absorb the OXFORD COMMA (', and'), not merely ',' or
# 'and' individually: a single-token separator alternation silently misses
# the finding's verbatim fact. Its alphabet is deliberately closed — digits,
# whitespace, ',', 'and'/'&'/'/', an optional repeated 'task'/'#' reference
# token — which is what lets the ids group be scanned for bare digits
# directly (see PLURAL_ENUM_SNAPSHOT_RE).
#
# The separator's LEADING element (a comma or whitespace) is mandatory and
# cannot match empty. That is what actually enforces the two-or-more-ids
# requirement: with every element optional, the repeated group matched
# between two adjacent digits of a SINGLE number, so '1020' parsed as
# '102' + '0' and 'Tasks 1020 are pending' wrongly satisfied a rule meant
# to need two ids.
#
# Every quantifier here is POSSESSIVE (task 3079, reviewer_comprehensive
# performance-redos finding). Written greedily, the separator's TRAILING
# '\s*' overlaps its own LEADING '\s*,\s*' / '\s+', so a whitespace run
# between two ids can be apportioned between those elements in
# (len(run) + 1) distinct ways. Under _ENUM_IDS_ALT's '+' repetition that
# is (w + 1) ** n parses, and ALL of them get explored whenever the overall
# match ultimately FAILS — the common case, since most enumerations in
# prose are not followed by a copula and a status marker. Measured pre-fix
# on a non-matching fact: 6.1s for 20 ids at ', ', and 16.6s for only NINE
# ids at ',\n    ' (the base scales with whitespace-run WIDTH, so a
# newline-indented list wedges the sweep at a much lower id count than a
# flat 2**n reading suggests). Post-fix all shapes parse in ~0.1ms.
#
# Possessive-ness is safe here — it changes only HOW a span is parsed,
# never WHETHER it matches — because every possessive element is followed
# by something that can never match whitespace: each '\s*+'/'\s++' is
# followed by ',', a letter, '#' or '\d++', and '\d++' is followed by
# whitespace or ','. So no quantifier can ever be holding a character a
# later element needs, and the accepted enumeration alphabet (including
# '1020, # 1030' with an internal space after the reference token) is
# unchanged. Possessive quantifiers require Python >= 3.11, which is
# exactly this package's floor (pyproject.toml).
_ENUM_SEP_ALT = r'(?:\s*+,\s*+|\s++)(?:(?:and|&|/)\s++)?(?:task\s*+)?#?\s*+'
_ENUM_IDS_ALT = r'\d++(?:' + _ENUM_SEP_ALT + r'\d++)+'

# Two independently-necessary constraints that together establish the
# enumeration is the copula's SUBJECT rather than a preposition's
# complement (amendment, reviewer_comprehensive correctness-precision
# finding, task 3079). Adjacency to the copula does NOT establish
# subjecthood — that was exactly the defect.
#
# (1) Plural agreement. If the enumeration really is the subject, the
# copula must agree with it, so 'is'/'was'/'remains' can only belong to
# some OTHER, singular subject ('The merge of tasks A and B is blocked' —
# the MERGE is blocked). 'Tasks A and B is pending' is malformed English,
# so nothing legitimate is lost. Note the shared _COPULA_ALT is
# deliberately left untouched and still serves the other paths; only this
# path narrows.
_PLURAL_COPULA_ALT = r'(?:are|were|remain)'

# (2) Preposition guard. Agreement alone is NOT sufficient:
# when the outer head is itself PLURAL the plural copula agrees with THAT
# head and the over-selection survives ('Dependencies for tasks A and B are
# blocked', 'The merges of tasks A and B are blocked', 'Reviews of tasks A,
# B, and C are pending' — all ordinary phrasings for this repo's memory
# corpus, all verified still over-selecting under a copula-only fix). The
# two remedies are complementary and ship together: agreement is a general
# grammatical constraint with no vocabulary to maintain, and the preposition
# check covers the plural-outer-head residue agreement cannot see.
#
# The check runs in PYTHON against the text preceding the match (see
# extract_snapshot_edge_task_ids), and it scans the whole CLAUSE rather than a
# fixed offset or a bounded slot: reject when a listed preposition appears
# anywhere between the last clause break and the enumeration. Both properties
# are load-bearing, because the gap between the preposition and the list noun
# is an OPEN class — determiners, quantifiers, bare adjectives, possessives,
# participles, and stacks of them — that neither a fixed offset nor any bound
# can cover. Each of these over-selects under a narrower spelling (a
# fixed-width lookbehind pinned before '\btasks\b'; a closed determiner slot;
# the review's suggested bound of two arbitrary words), and every outer head
# here is plural, so agreement does not save them either:
#
#     'Statuses of the tasks 1020 and 1030 are blocked.'         (1 word)
#     "Reviews of Leo's tasks 1020 and 1030 are pending."        (1 word)
#     'Notes about a few tasks 1020 and 1030 are pending.'       (2 words)
#     'Statuses of quite a few tasks 1020 and 1030 are pending.' (3 words)
#
# Clause scoping closes that class outright rather than moving its boundary,
# and it is the safe direction to err in: every case where it disagrees with a
# bounded form is one the clause form REJECTS. Its own cost is the converse
# residual below.
#
# '\b'-anchoring is retained, which is what keeps 'Migration tasks' /
# 'Verification tasks' matching — the 'on' inside 'Migration' is not
# '\b'-preceded. Matching runs over the fact's prefix from the last clause
# break onward, located by a reverse character scan rather than by splitting
# the prefix into clauses (see _enumeration_is_prepositional_complement), so
# cost is one bounded scan of text the regex engine has already walked.
#
# ONE residual now, and it points the WRONG way: the preposition vocabulary is
# still a closed list, so an UNLISTED preposition does not cause a miss — the
# guard simply fails to fire, the plural path matches, and the ids are
# selected. That is OVER-selection, the direction this module forbids, since
# the sweep would then invalidate a still-true edge the moment any named id
# goes terminal. (CORRECTION, amendment, task 3079: an earlier note called
# this "fail-safe under-selection", which is what made the gap read as
# acceptable.) So the list is a liability, not a safety margin, and it was
# widened once already — from/by/under/within/around/through/via/during/
# concerning — after the initial list was found to admit exactly the shapes it
# exists to refuse ('Dependencies from tasks 1020 and 1030 are blocked.').
#
# The converse residual IS fail-safe: a genuine subject-position enumeration
# sharing a clause with a listed preposition is missed. Its main real-world
# class is the sentence-initial ADVERBIAL PREAMBLE, since a comma does not end
# the clause:
#
#     'As of 2026-08-09, tasks 1020 and 1030 are pending.'      -> set()
#     'In the merge queue, tasks 1020 and 1030 are pending.'    -> set()
#
# Date-stamped and scope-setting preambles are common in this corpus — the
# finding's own motivating fact carries one ("... is pending as of
# 2026-07-14...") — so this is a real recall cost on the plural path, not a
# hypothetical one. It is pinned by
# test_adverbial_preamble_is_a_documented_under_selection so the cost stays
# visible and cannot widen unnoticed. Deliberately not tightened on
# speculation: every candidate tightening (requiring a plural/capitalized head
# before the preposition, or stopping the backward scan at a comma when no
# preposition follows it) trades back toward the unrecoverable over-selection
# direction, so it needed measurement against the real edge corpus first.
# (amendment, reviewer_comprehensive correctness-recall finding, task 3079)
#
# THAT MEASUREMENT IS NOW DONE (task 3949, measured 2026-08-17 against every
# populated FalkorDB graph the store reports — 40 graphs, discovered rather
# than hardcoded, because a hardcoded project list is a coverage claim nothing
# checks):
#
#     dark_factory              12548 valid edges | 172 'tasks <n>' near-misses
#     reify                     15871 valid edges | 168 near-misses
#     know_live                  1588 valid edges |  54 near-misses
#     solar_challenge_platform   1056 valid edges |   9 near-misses
#     autopilot_video             723 valid edges |  15 near-misses
#     + 35 further graphs         518 valid edges |   2 near-misses
#     (all)                     32304 valid edges | 420 near-misses
#                                                 | 2 PLURAL_ENUM_SNAPSHOT_RE
#                                                 |   matches, 2 rejections,
#                                                 |   BOTH correct
#
# The recall cost of this guard on the real corpus is ZERO edges. Two live
# facts reach it, and both are genuine prepositional complements it is right
# to reject — 'Task 3949 mentions that blockers for downstream, still-unmerged
# tasks 1020 and 1030 are pending' (the BLOCKERS are pending). Neither triages
# as an adverbial preamble, which is the only rejection class that costs
# recall. Note the observer effect, and discount accordingly: both edges were
# written at 07:24 on the measurement day by task 3949's own session, so what
# the corpus actually offers is still zero organic matches.
#
# What the plural path loses here it loses at the REGEX, not at the guard: the
# 420 near-misses carry 'tasks <digits>' but no status marker, no copula, or
# broken copula-marker adjacency ('Tasks 1752 and 1753 are related to ...',
# 'Both tasks 1920 and 1921 edit ...').
#
# Both candidates were re-validated against the full precision
# parametrization, as the finding required before shipping either:
#   (a) plural/capitalized head before the preposition — re-opens none of the
#       47 pinned precision+suppression shapes, but does NOT recover
#       'As of <date>, tasks ...', the finding's OWN motivating shape,
#       because 'of' there follows the capitalized sentence-initial 'As'.
#   (b) restart the backward scan after a preposition-free comma — recovers
#       all three preamble shapes, but RE-OPENS the pinned over-selection
#       'Blockers for down-stream, still-unmerged tasks 1020 and 1030 are
#       pending.', where the comma is intra-clause. It fails the required
#       re-validation outright — and now on LIVE data too: it re-opens one of
#       the two real corpus rejections above, which is the same shape.
#
# VERDICT: NOT TIGHTENED. Zero measured recall loss to buy back, against
# nonzero unrecoverable over-selection risk, on a module whose stated
# invariant is that under-selection self-heals and over-selection never does.
#
# Re-checkable as the corpus grows — the verdict is only as good as its
# re-runnability, and a rejection census is a point-in-time fact:
#     cd fused-memory && \
#       uv run python scripts/measure_plural_enum_guard_recall.py
# Artifacts: plans/plural-enum-guard-recall-report.{json,md}. The candidate
# re-validation is mechanical — it parametrizes over the shared pinned corpora
# in tests/reconciliation/plural_enum_shapes.py, which this suite parametrizes
# off too, so a shape appended there re-validates both candidates — in
# tests/test_measure_plural_enum_guard_recall.py.
#
# Kept as a single flat tuple rather than inlined into the pattern so the
# vocabulary has ONE source of truth: the regression test parametrizes
# directly over it, so an entry added here without a plural-head case fails
# immediately instead of shipping unexercised.
_ENUM_PREP_WORDS: tuple[str, ...] = (
    'of', 'in', 'on', 'to', 're', 'for', 'with', 'among', 'about', 'across',
    'against', 'between', 'regarding', 'from', 'by', 'under', 'within',
    'around', 'during', 'through', 'via', 'concerning',
)

# SENTENCE-FINAL punctuation ends the span a preposition can govern, so the
# guard looks no further back than the last one. This is what keeps 'Reviews
# for X. Tasks 1020 and 1030 are pending.' selected — the enumeration opens a
# NEW sentence and really is its copula's subject.
#
# A break requires TWO things together, not one: the character must be a
# member of this class (below), AND the occurrence must plausibly END a
# sentence — not be an INTRA-TOKEN '.' flanked by alphanumerics on both
# sides, e.g. a filename extension, version string, dotted module path or
# dotted section number (see _is_intra_token_dot / _last_clause_break;
# task 4149). Only '.' gets that second, occurrence-level test — because
# that is where over-selection was MEASURED and closed, not because ';',
# '!' and '?' are verified safe. They remain unconditional breaks
# whenever they appear, and at least one of them has a KNOWN, still-OPEN
# instance of the identical over-selection class: a '?' inside a URL
# query string or a ';' inside a path/branch name truncates the backward
# scan past the governing preposition exactly as an intra-token '.' did,
# e.g. (measured on this branch; amendment, reviewer_comprehensive
# correctness-precision finding, task 4149)
#
#     'Reviews for https://ci/build?ref=main tasks 1020 and 1030 are
#      pending.' -> {1020, 1030}
#     'Statuses of the branch;main tasks 1020 and 1030 are pending.'
#     -> {1020, 1030}
#
# Left open rather than fixed here: extending _is_intra_token_dot's
# flanking test to all four class members would be a strictly fail-safe
# generalization (narrowing an occurrence can only cost under-selection,
# per the asymmetry argument below) if the corpus supports it, but that
# needs its own measurement pass and is a follow-up, not this task.
#
# The CLASS is deliberately minimal — '.', ';', '!', '?' and nothing else,
# and membership is a verified precision requirement not reopened by this
# narrowing (task 3949 embargo) — because the two directions are not
# symmetric at either level, class or occurrence. Omitting a character from
# the class, or disqualifying an occurrence via the flanking test, can only
# LENGTHEN the scanned clause, so either can only ever cost under-selection
# (the fail-safe direction); admitting a character or occurrence that does
# not actually end a sentence truncates the scan and re-opens the
# over-selection this guard exists to close, which is unrecoverable. So a
# character earns a place in the class only by being able to end a
# sentence, and an occurrence of '.' is treated as a break only by
# plausibly ending one.
#
# Excluded from the CLASS on that rule, each verified over-selecting when it
# was treated as a break (amendment, reviewer_comprehensive
# correctness-precision finding, task 3079):
#
#     ':'          'Statuses of the following: tasks 1020 and 1030 are pending.'
#     '(' ')'      'Reviews of the following (still open): tasks 1020 and 1030
#                   are pending.'
#     '"' '“' '”'  'Statuses of the "next" tasks 1020 and 1030 are pending.'
#     ','          'Statuses of the following, tasks 1020 and 1030, are pending.'
#     newline      'Reviews for  the\n  tasks 1020 and 1030 are pending.'
#
# A colon after a governing preposition no more ends that preposition's
# government than a comma does — it typically introduces the very complement
# the preposition governs — and a parenthetical or quoted aside is an
# interpolation INSIDE the clause, not a new one. The comma and newline
# exclusions are load-bearing for the same reason and predate this narrowing.
# Colon-preambled genuine snapshots are unaffected ('Note: tasks 1020 and 1030
# are pending.' still extracts): they carry no listed preposition, so the
# longer clause gives the guard nothing to fire on.
#
# Excluded from OCCURRENCE by the flanking test — an intra-token '.' ends no
# sentence — each measured over-selecting before this narrowing (amendment,
# reviewer_comprehensive correctness-precision finding, task 4149):
#
#     filename extension     'Reviews for verify_cmd.py tasks 1020 and 1030
#                             are pending.' -> {1020, 1030}; the
#                             extension-free control 'Reviews for verify_cmd
#                             tasks 1020 and 1030 are pending.' -> set() is
#                             the measured minimal difference.
#     version string         'Statuses of the v1.2 tasks 1020 and 1030 are
#                             pending.'
#     dotted module path     'Statuses of tasks in df.core tasks 1020 and
#                             1030 are pending.'
#     dotted section number  'Reviews for section 4.2.1 tasks 1020 and 1030
#                             are pending.'
#
# A prefix-FINAL '.' (nothing to its right within prefix, e.g. prefix cut at
# '\\btasks\\b' immediately after the period) has no right flank and so is
# never intra-token — it stays a break regardless of this narrowing, which
# keeps the common 'X.Tasks 1020 and 1030 are pending.' shape selected.
_CLAUSE_BREAK_CHARS = '.;!?'


def _is_intra_token_dot(text: str, index: int) -> bool:
    """Is the '.' at ``text[index]`` flanked by alphanumerics on both sides?

    Unicode-aware by construction (``str.isalnum()``, the same predicate
    ``is_searchable_term`` uses in falkor_fulltext.py), so 'café.py' or a
    non-ASCII identifier is recognized as intra-token exactly like an ASCII
    one. A flanked '.' is a filename extension, version string, dotted
    module path or dotted section number — it ends no sentence, so it must
    not count as a clause break. (task 4149)

    Cost (fail-safe, under-selection): a genuine sentence period with no
    following space AND a following alphanumeric — e.g. 'Reviews for the
    branch are done.Then tasks 1020 and 1030 are pending.' — is flanked by
    alphanumerics on both sides too, so it reads as intra-token and the
    backward scan extends past the governing preposition, suppressing a
    real snapshot; the edge is simply not retired this cycle. Pinned by
    test_intra_token_dot_narrowing_costs_only_under_selection in
    test_stale_status_snapshot_edge_sweep.py.
    """
    return (
        index > 0
        and text[index - 1].isalnum()
        and index + 1 < len(text)
        and text[index + 1].isalnum()
    )


def _last_clause_break(prefix: str) -> int:
    """Index of the last sentence-plausible clause break in *prefix*, or -1.

    ';', '!' and '?' are treated as unconditional breaks; only '.' gets
    the occurrence-level flanking test below, because that is where
    over-selection was measured and closed — see the _CLAUSE_BREAK_CHARS
    comment block above for the known, still-open '?'/';' exception this
    leaves. '.' additionally requires that the occurrence not be an
    intra-token dot (see ``_is_intra_token_dot``); when it is, the walk
    retries at the next '.' to its left, stopping once it reaches or passes
    ``hard`` (the last unconditional break), since nothing further left
    could still change the answer. ``dot`` strictly decreases and each
    ``rfind`` resumes where the previous stopped, so the walk is
    O(len(prefix)) overall — no slicing, matching the cost property
    ``_enumeration_is_prepositional_complement`` documents. (task 4149)
    """
    hard = max((prefix.rfind(c) for c in _CLAUSE_BREAK_CHARS if c != '.'), default=-1)
    dot = prefix.rfind('.')
    while dot > hard and _is_intra_token_dot(prefix, dot):
        dot = prefix.rfind('.', 0, dot)
    return max(hard, dot)


_ENUM_PREP_WORD_RE: re.Pattern[str] = re.compile(
    r'\b(?:' + '|'.join(_ENUM_PREP_WORDS) + r')\b',
    re.IGNORECASE,
)


def _enumeration_is_prepositional_complement(prefix: str) -> bool:
    """Does a listed preposition govern the enumeration at ``prefix``'s end?

    ``prefix`` is the fact text preceding a ``PLURAL_ENUM_SNAPSHOT_RE`` match.
    True means the enumeration is a preposition's complement, not the copula's
    subject, and the match must be discarded. See ``_ENUM_PREP_WORDS`` for the
    full argument and the residual. (task 3079)
    """
    # Scan backwards for the last clause break and search from there, rather
    # than splitting the whole prefix into clauses and keeping only the last:
    # this runs once per plural match, and the extractor itself runs once per
    # valid edge over the whole group's edge set, so the module treats
    # extractor cost as a liveness property (see the module docstring).
    # ``rfind`` is a C-level reverse scan and ``search(..., pos)`` needs no
    # slice; '\b' at ``pos`` still sees the preceding character, and every
    # break char is non-word, so this is exactly equivalent to searching the
    # sliced clause. (amendment, reviewer_comprehensive efficiency finding,
    # task 3079)
    return _ENUM_PREP_WORD_RE.search(prefix, _last_clause_break(prefix) + 1) is not None

# Plural multi-task enumeration: 'Tasks A, B and C are <marker>' (task
# 3079). Before this, such an edge yielded NO ids at all — not merely the
# first, which is the precise question the finding asks. TASK_REF_RE anchors
# '\btask\b' and so does not match the plural head 'Tasks'; and even if it
# did, the enumeration tail carries no reference token of its own.
#
# Four anchoring properties, mirroring the adjacency discipline every other
# path in this module already follows:
#
# (a) The copula is MANDATORY — no optional-copula arm. This is the same
#     remedy INDIVIDUAL_SNAPSHOT_RE's transitive arm uses, and it is what
#     refuses the transitive-verb reading 'Tasks 1020, 1030, and 1031
#     blocked task 5' (a permanently-true historical fact). Because the
#     copula is mandatory, using _STATUS_MARKER_ALT whole — including the
#     transitive-capable markers — is safe here.
# (b) The enumeration must be the copula's SUBJECT, established jointly by
#     plural agreement (_PLURAL_COPULA_ALT) and the absence of a governing
#     preposition (_ENUM_PREP_WORDS, applied by
#     _enumeration_is_prepositional_complement inside
#     extract_snapshot_edge_task_ids). Adjacency does NOT establish this
#     reading — in 'The merge of tasks 1020 and 1030 is blocked' the marker
#     IS adjacent to the copula, but the copula's subject is the outer head
#     noun and the MERGE is what is blocked. Adjacency is the separate,
#     weaker property (c). Both remedies are needed: agreement alone still
#     admits a plural outer head ('Dependencies for tasks A and B are
#     blocked'), and the preposition check alone is a closed word list. See
#     _PLURAL_COPULA_ALT and _ENUM_PREP_WORDS for the full argument and the
#     residuals.
# (c) The marker must sit immediately after the enumeration and its copula,
#     with only closed-class function words (adverb, 'all', article) in
#     between, so 'Tasks 1020 and 1030 were merged into the active branch'
#     does not match — the BRANCH is active, not the tasks. This is a
#     necessary but NOT sufficient condition for the (b) reading.
# (d) Bare digits are collected from the 'ids' capture group ONLY, never
#     from the whole fact, preserving the module invariant that a '\d+'
#     contributes an id only from inside an already-detected,
#     marker-anchored span (see _BARE_DIGIT_RE).
#
# On (d): the aggregate path additionally strips COUNT_QUANTITY_RE spans
# from its segment before collecting digits, because a colon/bracket segment
# is free text that really can contain '...and 3 pending others'. This path
# needs no such strip and deliberately omits it: _ENUM_IDS_ALT's alphabet
# admits no count noun in the first place, so the strip could never fire
# usefully — while it WOULD misfire on the reference-token separator, where
# '1020 task 1030' matches COUNT_QUANTITY_RE's '\d+\s+tasks?' and stripping
# it would silently drop the legitimate leading id. A count phrase adjacent
# to a real enumeration ('tasks 1020 and 1030 plus 3 pending others are
# pending') instead breaks the copula adjacency and kills the match
# outright — under-selection, the fail-safe direction, and pinned by a test.
PLURAL_ENUM_SNAPSHOT_RE: re.Pattern[str] = re.compile(
    r'\btasks\b\s*#?\s*(?P<ids>' + _ENUM_IDS_ALT + r')\s*,?\s*'
    + _PLURAL_COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:all\s+)?(?:an?\s+)?'
    + _COMPOUND_PREFIX + _STATUS_MARKER_ALT + r'\b',
    re.IGNORECASE,
)


def _plural_enum_ids(
    fact: str,
    *,
    guard: Callable[[str], bool] = _enumeration_is_prepositional_complement,
) -> tuple[set[int], list[tuple[int, int]]]:
    """The plural-enumeration arm, in ONE place. (task 3079; task 3949)

    Returns ``(ids, rejected_spans)``: the ids the plural path yields, and the
    spans of every enumeration *guard* rejected. Both outputs are needed by
    the production extractor — a rejected span is, by construction, a region
    established to be a preposition's complement, so the drop it implies is
    applied to every other anchored path too (see
    ``extract_snapshot_edge_task_ids``).

    Ids come from the match's ``'ids'`` group only, via ``_BARE_DIGIT_RE``,
    preserving invariant (d): a bare ``\\d+`` contributes an id only from
    inside an already-detected, marker-anchored span.

    WHY THIS IS A FUNCTION AND NOT AN INLINE LOOP. ``guard`` is injectable
    solely so task 3949's recall probe
    (``fused-memory/scripts/measure_plural_enum_guard_recall.py``) can run the
    IDENTICAL arm under a candidate guard instead of re-implementing it. A
    hand-copied arm in the probe would keep measuring the old spelling after
    this one gained a step, and would report a reassuring zero while doing it
    — the same staleness argument that makes the probe import
    ``PLURAL_ENUM_SNAPSHOT_RE`` rather than re-spell it. Any future drift now
    surfaces as a signature change rather than as a silently stale copy.

    The default is the shipped guard, so production callers pass nothing and
    the injection point costs them nothing. It is a REPORTING seam, not an
    extension point: no production caller may pass a different guard.

    Pure: no I/O, no side effects.
    """
    ids: set[int] = set()
    rejected_spans: list[tuple[int, int]] = []
    for enum in PLURAL_ENUM_SNAPSHOT_RE.finditer(fact):
        # Subjecthood guard, second half: reject an enumeration that is a
        # PREPOSITION'S COMPLEMENT rather than the copula's subject
        # ('Reviews of the tasks A and B are pending' — the REVIEWS are
        # pending). Lives here rather than as an in-pattern lookbehind
        # because Python lookbehind is fixed-width, and a fixed offset is
        # defeated by a single intervening determiner — while the words that
        # may sit in that gap are an open class no bound can cover. See
        # _ENUM_PREP_WORDS.
        if guard(fact[: enum.start()]):
            rejected_spans.append(enum.span())
            continue
        ids.update(int(tok) for tok in _BARE_DIGIT_RE.findall(enum.group('ids')))
    return ids, rejected_spans


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

# Bare digit token — used only within an already-detected, marker-anchored
# span: an aggregate list segment, or a plural enumeration's 'ids' capture
# group (task 3079). A bare '\d+' therefore never contributes a candidate id
# outside that scoped context (dates/commit-hashes/ports in ordinary prose
# are neither).
_BARE_DIGIT_RE: re.Pattern[str] = re.compile(r'\b\d+\b')


# The clause-ish boundary that closes a colon-introduced list segment (which
# has no closing delimiter of its own) and backstops an unterminated bracket
# segment is task_filter.STRICT_CLAUSE_BOUNDARY_RE — the fail-safe-STRICT
# alphabet [.;\n!?], NOT task_filter._CLAUSE_SPLIT_RE, which narrowed its dot
# to '\.(?!\w)' at task 3403 so dotted technical tokens stop shattering a
# sentence. The two started out identical; the canonical rationale for keeping
# them apart lives next to STRICT_CLAUSE_BOUNDARY_RE. This module's stake in
# it: the segment closed here is one from which BARE DIGITS are harvested as
# task ids, and that path ends in memory_service.update_edge(invalid_at=...) —
# a real Graphiti edge retirement. A longer segment absorbs more incidental
# digits, i.e. over-selection, which this module's docstring explicitly
# forbids: under-selection self-heals next cycle or is caught by Stage 2,
# over-selection wrongly retires true facts. Copying the widening for symmetry
# would trade a bounded soft-block for a permanent wrong invalidation.
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

    boundary = STRICT_CLAUSE_BOUNDARY_RE.search(text, start)
    end = boundary.start() if boundary else len(text)
    return text[start:end]


def extract_snapshot_edge_task_ids(fact: str) -> set[int]:
    """Return the task ids *fact* asserts as active/pending/blocked/stalled/in-progress.

    Returns the empty set (never a candidate for invalidation) when:
    - *fact* contains no active/pending/blocked/stalled/in-progress status marker at
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
         anchors 'task N' / '#N' / 'df N' / 'task/N' (TASK_REF_RE's own
         grammar; the slash form was admitted by task 3403)
         directly to an adjacent status marker (only an optional copula
         and/or article may sit in between — deliberately NOT the
         preposition 'in') — so an incidental status word elsewhere in
         the fact is never wrongly attributed to the reference.
      3. Genitive form: extract ids via GENITIVE_STATUS_RE, which reaches
         the possessive shape "Task N's status is <marker>" (task 3079).
         The literal noun 'status' must be immediately followed by a
         copula, which is what keeps a following head noun ("Task N's
         status REPORT is pending review") out without needing the
         trailing lookahead SNAPSHOT_STATUS_PHRASE_RE carries.
      4. Status-phrase form: extract ids via SNAPSHOT_STATUS_PHRASE_RE,
         which additionally admits a bounded open-class gap (e.g. 'is
         deliberately parked in ...') between the reference and the
         marker, but only when the marker is immediately followed by the
         literal noun 'status' — the token that makes the span an
         explicit status assertion rather than an incidental mention.
      5. Plural enumeration form: extract ids via PLURAL_ENUM_SNAPSHOT_RE,
         which reaches 'Tasks A, B and C are <marker>' — a shape carrying
         no per-id reference token, so every id was previously
         unreachable (task 3079). The copula is mandatory and the marker
         must be adjacent to it; digits are collected from the match's
         'ids' group only.
      6. Aggregate form: for each detected list segment ('<marker> tasks
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

    # The plural arm lives in _plural_enum_ids so task 3949's recall probe can
    # measure THIS code rather than a copy of it (see that function).
    ids, rejected_spans = _plural_enum_ids(fact)

    # Suppressing only the PLURAL match left the rejected enumeration's tail
    # extractable by the other anchored paths, because an enumeration may
    # REPEAT the reference token — a shape this module newly supports as a
    # positive ('Tasks 1020, task 1030 and task 1031 are pending'). Its
    # prepositional-complement counterpart therefore reached the individual
    # arm on its own and over-selected the trailing id:
    #
    #     'Reviews for tasks 1020, task 1030 and task 1031 are pending.' -> {1031}
    #     'Statuses of the tasks 1020 and task 1030 are blocked.'        -> {1030}
    #
    # A rejected span is, by construction, a region established to be a
    # preposition's complement, so NO id inside it is the copula's subject
    # whichever pattern found it — hence the drop is applied to every
    # anchored path rather than to the individual arm alone. It is scoped to
    # the span, not the fact, so a genuine snapshot in a later clause
    # survives. (amendment, reviewer_comprehensive correctness-precision
    # finding, task 3079)
    for pattern in (INDIVIDUAL_SNAPSHOT_RE, GENITIVE_STATUS_RE, SNAPSHOT_STATUS_PHRASE_RE):
        ids.update(
            int(m.group(1))
            for m in pattern.finditer(fact)
            if not any(start <= m.start() < end for start, end in rejected_spans)
        )

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
    each edge asserts as active/pending/blocked/stalled/in-progress, cross-references those
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
