"""Status-snapshot extraction: the pure lexical layer that reads which task ids
a Graphiti edge fact asserts as non-terminal, consumed by
``stale_status_snapshot_edge_sweep.py``.

Every path here is biased toward missing a snapshot over wrongly retiring
one: under-selection self-heals next cycle, while over-selection is
unrecoverable, since an invalidated Graphiti edge is never re-validated.

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

All five are compiled by ONE builder (``_build_snapshot_patterns``) and
instantiated TWICE (task 3037): the UNION family, from which the five
constants above are bound, and a BLOCKED family narrowed to the literal
'blocked' marker. ``extract_blocked_assertion_task_ids`` runs the blocked
family to report WHICH ids a fact asserts as blocked — a distinction the
union extractor structurally cannot make, and one
``select_stale_status_snapshot_edges``'s blocked-assertion rule requires.
Adding a path, or hardening one, therefore has to be done once and reaches
both families; hand-copying a blocked-only path would reopen exactly the
drift surface task 3042 closed for the MARKER constants.

Standing invariants — these are cross-cutting and apply to every path
above; re-check all of them before changing any single path. (Per-path
detail, examples, and residuals are documented once, at the owning
constant — do not restate them here.)

- Marker-addition rule: EVERY extraction path must anchor its marker
  adjacently to whatever it describes; the whole-fact gate
  (``SNAPSHOT_STATUS_RE``) only decides whether a fact is worth
  considering at all — it matches a marker ANYWHERE in the fact, so it can
  never be what separates a status snapshot from an unrelated fact. The
  aggregate list path was once left resting on the gate alone and had to
  be re-anchored (``LIST_INTRODUCER_RE``) once 'blocked' widened the gate
  — task 2885 repro: 4 stale edges survived a blocked->done transition
  (invalidated=0) because the gate short-circuited before
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
- Every extraction path exists in TWO families — union and blocked-only —
  built from the same ``_build_snapshot_patterns`` call, and the blocked
  result is always a SUBSET of the union result. A future path or guard
  must be added inside the builder, never as a family-specific special
  case: a blocked id the union family cannot reach would let the sweep
  select an edge on an id it never put into its ``get_statuses`` census.
  (task 3037)  A guard that only ever NARROWS one family is the one
  permitted asymmetry — it cannot break the subset property — but it is
  still expressed as a builder PARAMETER carried on
  ``_SnapshotPatterns``, never as a branch on family identity inside
  ``_extract_ids``.  ``reject_list_after_task_ref`` is the only such
  parameter today; the two rules retire an edge on very different
  triggers, so a residual that is tolerable under one can be intolerable
  under the other.  (amendment, task 3037)
- WHAT ACTUALLY MAKES THAT SUBSET PROPERTY HOLD — and it is NOT pattern
  nesting, which is what this list claimed until it was measured false.
  Two mechanisms, both in ``_ids_within_union``: the union family's
  prepositional-complement ``rejected_spans`` are THREADED into the
  blocked call, and the returned ``blocked_ids`` are INTERSECTED with
  ``all_ids`` at the seam.  The general warning, which is the part worth
  carrying forward: ALGORITHM DATA computed inside ``_extract_ids`` is
  PER-FAMILY, and must be threaded explicitly whenever it can only NARROW
  a family's result — sharing the pattern builder does not carry it.  The
  bug this closes: ``rejected_spans`` is built from a family's OWN
  ``plural_enum`` matches, and the blocked family's ``plural_enum``
  requires the literal 'blocked' marker, so an enumeration the union
  family rejected on a 'pending' marker was never built by the blocked
  family, suppressed nothing there, and let its individual arm extract an
  id the union family had deliberately refused — over-selection, the
  unrecoverable direction.  Note the ONE piece of per-family data that is
  already safe and must be LEFT ALONE:
  ``patterns.reject_list_after_task_ref`` is True for the blocked family
  only, so it makes that family reject MORE (the transitive-verb reading
  'Task 5 blocked tasks: 142, 148').  It narrows blocked relative to
  union, which PRESERVES the subset property rather than breaking it.
  (amendment, reviewer_comprehensive correctness-over-selection finding,
  task 3037)

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
- task 3037: the blocked family drops any aggregate list whose introducer
  follows a task reference in the same clause ('Task 5 is waiting on
  blocked tasks: 142, 148'), so a genuine aggregate of that shape is not
  retired by the BLOCKED rule — the terminal rule still reaches it. The
  trade is deliberate: the same shape covers the transitive-verb reading
  'Task 5 blocked tasks: 142, 148', whose over-selection the blocked rule
  would trigger almost immediately rather than only at done/cancelled. See
  ``_list_is_governed_by_task_ref``.
- tasks 4149 / 4851: a genuine '.', ';', '!' or '?' sentence break
  immediately followed by an alphanumeric ('done.Then', 'done?Then') reads
  as token-internal, so a snapshot behind a listed preposition in the
  previous sentence is missed — see ``is_token_internal_break``.
- task 4851, measured: four lexical classes of genuine snapshot sit outside
  the closed-class copula/article anchoring and are not extracted — a
  relative clause ('task N, which is pending'), an attributive pre-modifier
  or predicate nominal ('Blocked task N ...'), a gerund ('due to being
  pending') and a secondary or coordinated predicate ('is filed and
  pending').  All four under-select; each would need its own precision
  pass before widening.  Triage and samples:
  ``plans/stale-status-snapshot-sweep-report.md``.

Why a regex in this module gets a performance test at all:
``sweep_stale_status_snapshot_edges`` calls
``extract_snapshot_edge_task_ids_by_marker_class`` once per valid edge from
an UNGUARDED dict comprehension with no per-edge timeout, over the whole
group's edge set (tens of thousands of edges; current figure in
``plans/stale-status-snapshot-sweep-report.md``).  Extractor cost is
therefore a whole-cycle LIVENESS property — one pathological fact stalls
the entire reconciliation cycle — not a micro-optimisation. (amendment,
task 3079)

The corpus this guards keeps growing, and the task-4340 pagination
removed a truncation that had hidden about half of it (see the amendment
to ruled-out hypothesis 1 in ``stale_status_snapshot_edge_sweep.py``'s
docstring), so the LIVENESS argument only gets
STRONGER over time.  The read that feeds it was timed
2026-08-18 at ~3.3 s per full enumeration on dark_factory (~3.7 s on
reify) — bounded, and roughly +2.6 s per cycle over the old truncated
read; see the MEASURED COST section of that same audit doc, so this
claim rests on a number rather than on an estimate. (amendment, task 4340)

Adding the second (blocked) pattern family did NOT add a second per-edge
pass (amendment, task 3037). The per-edge extraction remains exactly ONE
entry into the pipeline: ``extract_snapshot_edge_task_ids_by_marker_class``
evaluates the whole-fact gate once and returns both id sets, and the
sweep's comprehension is its only caller — pinned by
``test_extraction_runs_exactly_once_per_edge``, which counts pipeline
entries rather than wall-clock. What keeps the added family's cost off the
hot path is the cheap literal ``_BLOCKED_MARKER_RE`` pre-gate: the
overwhelming majority of edges carry no 'blocked' token, so they pay one
extra substring scan and never run the blocked family's five anchored
patterns at all. Only edges that really do mention 'blocked' pay for the
second family, and for those the work is irreducible — the two families
answer different questions.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

from fused_memory.reconciliation.task_filter import STRICT_CLAUSE_BOUNDARY_RE, TASK_REF_RE

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


# The blocked-only marker alternation (task 3037). Deliberately the literal
# 'blocked' ALONE, not the whole _TRANSITIVE_MARKER_ALT class: 'stalled' is
# not a member of the closed shared.task_statuses.TaskStatus vocabulary, so
# there is no census value a 'stalled' assertion could be compared against.
# The blocked selection rule keys on "positively known and != 'blocked'", a
# test that is only meaningful for a marker with a corresponding status
# value — see select_stale_status_snapshot_edges.
_BLOCKED_MARKER_ALT = r'(?:blocked)'

# Cheap literal pre-gate for the blocked family, mirroring SNAPSHOT_STATUS_RE's
# role for the union family: a fact with no 'blocked' token at all cannot be a
# blocked assertion, so extract_blocked_assertion_task_ids short-circuits to
# set() after ONE substring scan rather than running five anchored patterns.
#
# This is a LIVENESS measure, not a micro-optimisation. The extractor runs
# once per valid edge over the whole group's edge set (tens of thousands of
# edges; current figure in plans/stale-status-snapshot-sweep-report.md; ~3.3 s
# per full enumeration — see the module docstring), and the overwhelming majority of those edges carry no 'blocked' token, so
# the pre-gate is what keeps the second pattern family off the hot path.
_BLOCKED_MARKER_RE: re.Pattern[str] = re.compile(r'\bblocked\b', re.IGNORECASE)

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
# THAT MEASUREMENT IS NOW DONE (task 3949), against every populated FalkorDB
# graph the store reports — discovered rather than hardcoded, because a
# hardcoded project list is a coverage claim nothing checks. THE FIGURES LIVE
# IN THE ARTIFACT, DELIBERATELY NOT HERE:
#
#     plans/plural-enum-guard-recall-report.md    (per-graph table, totals)
#     plans/plural-enum-guard-recall-report.json  (same, machine-readable)
#
# An earlier draft of this block hand-copied that table inline, and every row
# of it went stale within days as the corpus grew — the per-graph edge counts,
# the graph count and the near-miss total all moved on the next re-run. Cite
# the artifact and re-run the script; do not paste the numbers back in here.
#
# What the measurement FOUND is the durable part, and is what belongs here:
# the recall cost of this guard on the real corpus is ZERO edges. Only two live
# facts reach the guard at all, and both are genuine prepositional complements
# it is right to reject — 'Task 3949 mentions that blockers for downstream,
# still-unmerged tasks 1020 and 1030 are pending' (the BLOCKERS are pending).
# Neither triages as an adverbial preamble, which is the only rejection class
# that costs recall. Note the observer effect, and discount accordingly: both
# edges were written at 07:24 on the FIRST measurement day by task 3949's own
# session, so what the corpus actually offers is still zero organic matches.
#
# What the plural path loses here it loses at the REGEX, not at the guard. The
# artifact's 'tasks <n>' column is a LEXICAL SUPERSET that already includes the
# two full matches above; the remainder carry 'tasks <digits>' but no status
# marker, no copula, or broken copula-marker adjacency ('Tasks 1752 and 1753
# are related to ...', 'Both tasks 1920 and 1921 edit ...').
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
# sentence — not be TOKEN-INTERNAL (see is_token_internal_break /
# _last_clause_break). Every member of the class gets that second,
# occurrence-level test (task 4149 for '.', task 4851 for ';', '!' and '?').
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
# sentence, and an occurrence is treated as a break only by plausibly
# ending one.
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
# Excluded from OCCURRENCE as token-internal — such an occurrence ends no
# sentence — each measured over-selecting when it was treated as a break
# (amendment, reviewer_comprehensive correctness-precision finding, task 4149):
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
# The same holds for a token-internal ';', '?' or '!' and for the ';' that
# ends an HTML entity (task 4851). Every excluded shape is pinned in
# tests/reconciliation/plural_enum_shapes.py::PRECISION_GUARD_SHAPES.
#
# A prefix-FINAL break (nothing to its right within prefix, e.g. prefix cut
# at '\\btasks\\b' immediately after the period) has no right flank and so is
# never alnum-flanked — it stays a break regardless of this narrowing, which
# keeps the common 'X.Tasks 1020 and 1030 are pending.' shape selected.
CLAUSE_BREAK_CHARS = '.;!?'


# The head of an HTML character entity reference ('&lt', '&#60', '&#x3c'),
# anchored to end exactly where the ';' that terminates it begins. Searched
# over a bounded window ending at that ';', so the check costs O(1) per
# occurrence and needs no slice.
_HTML_ENTITY_HEAD_RE: re.Pattern[str] = re.compile(
    r'&(?:[A-Za-z][A-Za-z0-9]{1,31}|#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6})\Z'
)
_HTML_ENTITY_HEAD_MAX_LEN = 40


def is_token_internal_break(text: str, index: int) -> bool:
    """Does the break character at ``text[index]`` end no sentence?

    True for either of two reasons (tasks 4149, 4851):

    (a) it is flanked by alphanumerics on both sides — a filename
        extension, version string, dotted module path, dotted section
        number, URL query '?', path/branch ';' or token-internal '!';
    (b) it is a ';' terminating an HTML character entity ('&lt;', '&#60;'),
        which is usually followed by a quote or paren rather than an
        alphanumeric, so (a) alone would leave it breaking the clause.

    Unicode-aware by construction (``str.isalnum()``, the same predicate
    ``is_searchable_term`` uses in falkor_fulltext.py), so 'café.py' or a
    non-ASCII identifier is recognized exactly like an ASCII one.

    Cost (fail-safe, under-selection): a genuine sentence break with no
    following space AND a following alphanumeric — 'Reviews for the branch
    are done.Then tasks 1020 and 1030 are pending.', or the same with '?' —
    reads as token-internal, so the backward scan extends past the governing
    preposition and suppresses a real snapshot; the edge is simply not
    retired this cycle. Pinned by
    test_token_internal_break_narrowing_costs_only_under_selection in
    test_stale_status_snapshot_edge_sweep.py.
    """
    if (
        index > 0
        and text[index - 1].isalnum()
        and index + 1 < len(text)
        and text[index + 1].isalnum()
    ):
        return True
    return text[index] == ';' and _HTML_ENTITY_HEAD_RE.search(
        text, max(0, index - _HTML_ENTITY_HEAD_MAX_LEN), index
    ) is not None


def _last_clause_break(prefix: str) -> int:
    """Index of the last sentence-plausible clause break in *prefix*, or -1.

    Every member of ``CLAUSE_BREAK_CHARS`` counts only where the occurrence
    is not token-internal (see ``is_token_internal_break``). The walk keeps
    one cursor per break character and repeatedly takes the rightmost; when
    that occurrence is token-internal, only its own cursor moves left, to the
    previous occurrence of the same character. Each cursor only moves left
    and each ``rfind`` resumes where that cursor's previous one stopped, so
    the walk is O(len(prefix)) overall — no slicing, matching the cost
    property ``_enumeration_is_prepositional_complement`` documents.
    (tasks 4149, 4851)
    """
    cursors = {c: prefix.rfind(c) for c in CLAUSE_BREAK_CHARS}
    while True:
        char, cursor = max(cursors.items(), key=lambda item: item[1])
        if cursor < 0 or not is_token_internal_break(prefix, cursor):
            return cursor
        cursors[char] = prefix.rfind(char, 0, cursor)


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


def _list_is_governed_by_task_ref(prefix: str) -> bool:
    """Does an explicit task reference precede this list introducer's clause?

    ``prefix`` is the fact text preceding a ``list_introducer`` match. True
    means the '<marker> tasks:' span reads as a TRANSITIVE VERB taking that
    reference as its subject ('Task 5 blocked tasks: 142, 148') rather than as
    a status list ('Blocked tasks: 142, 148'), and the segment must be
    discarded.

    Why this exists, and why only the BLOCKED family runs it (amendment,
    reviewer_comprehensive correctness-over-selection finding, task 3037).
    ``LIST_INTRODUCER_RE``'s adjacency requirement narrowed the transitive/
    status-list collision to exactly one shape — the marker sitting
    immediately before the list noun with no intervening word — and task 3042
    logged that survivor as a deliberate ambiguity residual. It was tolerable
    under the TERMINAL rule alone: a permanently-true historical fact was only
    wrongly retired once one of its listed ids reached done/cancelled, which
    is rare and late. The task-3037 BLOCKED rule retires on ANY positively
    known non-'blocked' status, so the same residual fires on essentially the
    next cycle — and additionally emits a superseding fact attributed to the
    wrong subject. That is a material amplification in the OVER-selection
    direction the module docstring calls unrecoverable, so the blocked family
    trades it for under-selection.

    The union family keeps the old behaviour deliberately: narrowing it too
    would change the task-2613 terminal rule, which this task did not touch
    and whose residual is unchanged in both likelihood and consequence.
    Neither choice can violate the blocked-subset-of-union invariant, because
    this guard only ever REMOVES ids from the blocked family.

    Clause-scoped for the same reason ``_enumeration_is_prepositional_
    complement`` is: a reference in an earlier sentence governs nothing here,
    so 'Task 5 is blocked. Blocked tasks: 142, 148' keeps its aggregate. Uses
    the same ``rfind`` + ``search(..., pos)`` shape rather than slicing, so it
    adds no allocation to a hot path the module treats as a liveness property.

    Cost (fail-safe, under-selection): a GENUINE aggregate that happens to
    follow a task reference in its own clause — 'Task 5 is waiting on blocked
    tasks: 142, 148' — is no longer reachable by the blocked family. Its edge
    is simply not retired by the blocked rule; the terminal rule still retires
    it when a listed id goes done/cancelled, exactly as before task 3037.
    """
    return TASK_REF_RE.search(prefix, _last_clause_break(prefix) + 1) is not None


# --------------------------------------------------------------------------- #
# Anchored pattern families — ONE builder, two instantiations (task 3037)
# --------------------------------------------------------------------------- #


class _SnapshotPatterns(NamedTuple):
    """The five anchored extraction patterns for one marker family.

    ``individual`` omits its adjective arm entirely when the family carries no
    adjective markers (see ``_build_snapshot_patterns``).

    ``reject_list_after_task_ref`` is not a pattern but a per-family SETTING of
    the shared algorithm, carried here so ``_extract_ids`` reads it off the
    family it was handed rather than branching on which family it is running —
    the same reason the marker alternations are builder parameters. See
    ``_build_snapshot_patterns``.
    """

    individual: re.Pattern[str]
    genitive: re.Pattern[str]
    status_phrase: re.Pattern[str]
    plural_enum: re.Pattern[str]
    list_introducer: re.Pattern[str]
    reject_list_after_task_ref: bool


def _build_snapshot_patterns(
    adjective_alt: str | None,
    transitive_alt: str | None,
    *,
    reject_list_after_task_ref: bool = False,
) -> _SnapshotPatterns:
    """Compile the five anchored extraction patterns over one marker family.

    Task 3037 needs a SECOND, blocked-scoped family alongside the union
    family, because the union extractor cannot say WHICH marker contributed
    an id ('Task 5 is pending' and 'Task 5 is currently blocked' are
    indistinguishable at ``{5}``) and the new blocked selection rule must not
    reach an active/pending snapshot. Hand-copying five blocked-only patterns
    would have created exactly the drift surface task 3042 closed for the
    MARKER constants — a hardening applied to one family and silently missed
    on the other. Deriving both families from this single builder makes that
    drift structurally impossible: every guard tasks 2613 / 3042 / 3079 /
    3403 / 4149 bought (transitive-verb, negation and past-exit,
    intervening-task-reference, prepositional-complement subjecthood,
    possessive quantifiers, token-internal-break narrowing) is written once and
    applies to both families by construction.

    Args:
        adjective_alt: Alternation of ADJECTIVE-only markers, or None to omit
            the individual form's adjective arm entirely. Adjective markers
            take an OPTIONAL copula ('Task 5 pending' matches).
        transitive_alt: Alternation of markers that are ALSO common transitive
            verbs, or None to omit that arm. These require a MANDATORY
            copula/article, which is what refuses the permanently-true
            historical reading 'Task 5 blocked the merge queue'.
        reject_list_after_task_ref: Drop an aggregate list segment whose
            introducer is preceded, IN THE SAME CLAUSE, by an explicit task
            reference — 'Task 5 blocked tasks: 142, 148'. Off for the union
            family (whose behaviour is unchanged), on for the blocked family.
            See ``_list_is_governed_by_task_ref`` for the full argument.

    At least one of the two must be supplied. The remaining four patterns run
    against the UNION of whichever were supplied, since each of them pins its
    marker with an independent anchor (a mandatory copula, the literal noun
    'status', or adjacency to the list noun) and so is safe against the
    transitive-verb reading whatever the family.

    Pure: no I/O, no side effects. Called exactly twice at import time.
    """
    if adjective_alt is None and transitive_alt is None:
        raise ValueError('_build_snapshot_patterns needs at least one marker alternation')

    supplied = [alt for alt in (adjective_alt, transitive_alt) if alt is not None]
    marker_alt = supplied[0] if len(supplied) == 1 else r'(?:' + r'|'.join(supplied) + r')'
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

    # The two arms are assembled conditionally so a single-class family (the
    # blocked family, which has no adjective markers) omits the arm it has no
    # markers for rather than compiling an empty alternative — an empty
    # alternative would match the empty string and turn every bare task
    # reference into a snapshot assertion.
    arms: list[str] = []
    if adjective_alt is not None:
        # adjective arm — optional copula (task 2613 behaviour, unchanged)
        arms.append(
            r'\s*' + _COPULA_ALT + r'?\s*' + _ADVERB_ALT + r'(?:an?\s+)?'
            + _COMPOUND_PREFIX + adjective_alt
        )
    if transitive_alt is not None:
        # transitive-capable arm — copula or article is MANDATORY
        arms.append(
            r'\s+(?:' + _COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:an?\s+)?|an?\s+)'
            + _COMPOUND_PREFIX + transitive_alt
        )
    individual = re.compile(
        TASK_REF_RE.pattern + r'(?:' + r'|'.join(arms) + r')\b',
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
    # The copula is MANDATORY here, so marker_alt (which includes the
    # transitive-capable 'blocked') is safe to use whole: there is no bare-verb
    # reading of "Task 5's status is blocked".
    genitive = re.compile(
        TASK_REF_RE.pattern
        + r"(?:'|’)s\s+status\s+"
        + _COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:an?\s+)?'
        + _COMPOUND_PREFIX + marker_alt + r'\b',
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
    status_phrase = re.compile(
        TASK_REF_RE.pattern
        # The gap is open-class BUT may absorb neither negation / past-exit
        # qualifiers (residual (2) — they INVERT the assertion) nor the opening
        # token of another task reference (residual (4) — it RE-SUBJECTS the
        # assertion). Both are per-gap-word negative lookaheads.
        + r'(?:\s+' + _GAP_EXCLUDED_ALT + _GAP_NO_TASK_REF + r'\w+){0,3}?'
        + r'\s+in\s+(?:an?\s+|the\s+)?'
        + marker_alt + r'\s+status\b'
        # the span must END the noun phrase — else 'status' is a modifier of a
        # following head noun ('status report'), not the phrase head
        + r'(?=\s*[.,;:!?)\]]|\s*$|\s+(?:due|under|pending|since|because|'
        + r'awaiting|after|before|while|until|per|as|with|for|on|from|and|or)\b)',
        re.IGNORECASE,
    )

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
    #     copula is mandatory, using marker_alt whole — including the
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
    plural_enum = re.compile(
        r'\btasks\b\s*#?\s*(?P<ids>' + _ENUM_IDS_ALT + r')\s*,?\s*'
        + _PLURAL_COPULA_ALT + r'\s+' + _ADVERB_ALT + r'(?:all\s+)?(?:an?\s+)?'
        + _COMPOUND_PREFIX + marker_alt + r'\b',
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
    #   AMENDED, task 3037: this residual survives in the UNION family only.
    #   The BLOCKED family additionally drops a segment whose introducer is
    #   preceded by a task reference in the same clause, because the blocked
    #   selection rule turns the same ambiguity from a rare late
    #   over-selection into a near-immediate one, and pairs it with a
    #   superseding fact attributed to the wrong subject — see
    #   ``_list_is_governed_by_task_ref`` for the argument and the
    #   under-selection it costs.
    list_introducer = re.compile(
        r'\b' + marker_alt + r'\s+tasks?\b\s*(?:are|is|were)?\s*(?P<open>[:\[])',
        re.IGNORECASE,
    )

    return _SnapshotPatterns(
        individual=individual,
        genitive=genitive,
        status_phrase=status_phrase,
        plural_enum=plural_enum,
        list_introducer=list_introducer,
        reject_list_after_task_ref=reject_list_after_task_ref,
    )


# The UNION family — every marker class. The five module-level constants below
# are BOUND from it rather than compiled separately, so their exported identity
# and behaviour are unchanged by the task-3037 refactor (the whole module test
# suite pins that).
_UNION_PATTERNS = _build_snapshot_patterns(_ADJECTIVE_MARKER_ALT, _TRANSITIVE_MARKER_ALT)

# The BLOCKED family — the same five patterns narrowed to the literal
# 'blocked' marker, so extract_blocked_assertion_task_ids can report WHICH ids
# a fact asserts as blocked rather than the union over every marker. 'blocked'
# is transitive-capable, so it is supplied as the transitive alternation and
# the individual form's adjective arm is omitted: the mandatory copula/article
# is exactly what keeps 'Task 5 blocked the merge queue' out.
_BLOCKED_PATTERNS = _build_snapshot_patterns(
    None, _BLOCKED_MARKER_ALT, reject_list_after_task_ref=True,
)

INDIVIDUAL_SNAPSHOT_RE: re.Pattern[str] = _UNION_PATTERNS.individual
GENITIVE_STATUS_RE: re.Pattern[str] = _UNION_PATTERNS.genitive
SNAPSHOT_STATUS_PHRASE_RE: re.Pattern[str] = _UNION_PATTERNS.status_phrase
PLURAL_ENUM_SNAPSHOT_RE: re.Pattern[str] = _UNION_PATTERNS.plural_enum
LIST_INTRODUCER_RE: re.Pattern[str] = _UNION_PATTERNS.list_introducer


# Bare digit token — used only within an already-detected, marker-anchored
# span: an aggregate list segment, or a plural enumeration's 'ids' capture
# group (task 3079). A bare '\d+' therefore never contributes a candidate id
# outside that scoped context (dates/commit-hashes/ports in ordinary prose
# are neither).
_BARE_DIGIT_RE: re.Pattern[str] = re.compile(r'\b\d+\b')


# The clause-ish boundary that closes a colon-introduced list segment (which
# has no closing delimiter of its own) and backstops an unterminated bracket
# segment is task_filter.STRICT_CLAUSE_BOUNDARY_RE — the fail-safe-STRICT
# alphabet [.;\n!?], NOT task_filter.WIDE_CLAUSE_BOUNDARY_RE, which narrowed
# its dot to '\.(?!\w)' at task 3403 so dotted technical tokens stop
# shattering a sentence. The two started out identical; the canonical
# rationale for keeping them apart lives next to STRICT_CLAUSE_BOUNDARY_RE.
# This module's stake in it: the segment closed here is one from which BARE
# DIGITS are harvested as task ids, and that path ends in
# memory_service.update_edge(invalid_at=...) — a real Graphiti edge
# retirement. A longer segment absorbs more incidental digits, i.e.
# over-selection, which this module's docstring explicitly forbids:
# under-selection self-heals next cycle or is caught by Stage 2,
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


def extract_snapshot_edge_task_ids(fact: str | None) -> set[int]:
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

    Algorithm (steps 2-6 are shared with
    ``extract_blocked_assertion_task_ids`` via ``_extract_ids``; only the
    step-1 gate and the pattern family differ between them):
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
    return extract_snapshot_edge_task_ids_by_marker_class(fact).all_ids


def extract_blocked_assertion_task_ids(fact: str | None) -> set[int]:
    """Return the subset of task ids *fact* asserts as BLOCKED specifically.

    The blocked-scoped counterpart of ``extract_snapshot_edge_task_ids``:
    same gate -> plural-with-prepositional-complement-rejection -> anchored
    paths -> aggregate-segment algorithm, run against the BLOCKED pattern
    family instead of the union family.

    Always a SUBSET of the union result — but NOT, as this docstring claimed
    until it was measured false, merely because both families come from
    ``_build_snapshot_patterns`` with nesting marker alternations. Shared
    patterns do not share the rejected-span suppression, which is algorithm
    data computed per family. The containment holds because
    ``_ids_within_union`` threads the union family's rejected spans into this
    family's call and intersects the result at the seam; see there.
    (amendment, reviewer_comprehensive
    correctness-over-selection finding, task 3037)

    ONE guard is blocked-family-only, and it too narrows: an aggregate list
    whose introducer follows a task reference in the same clause is dropped,
    so the transitive-verb reading 'Task 5 blocked tasks: 142, 148' does not
    reach the blocked rule. Narrowing blocked relative to union is the safe
    asymmetry — it preserves containment. See ``_list_is_governed_by_task_ref``.

    Why this exists (task 3037). ``extract_snapshot_edge_task_ids`` returns
    the UNION over every status marker, so it cannot say WHICH ids a fact
    asserts as blocked: 'Task 5 is pending' and 'Task 5 is currently blocked'
    both yield ``{5}``.  ``select_stale_status_snapshot_edges``'s
    blocked-assertion rule retires an edge whose asserted blocked status is
    contradicted by ANY positively-known non-'blocked' status; applied to
    union ids that rule would retire an active/pending snapshot the moment
    its task moved to 'review' — a large behavioural widening, and in the
    over-selection direction this module's docstring forbids.

    Scoped to the literal 'blocked' marker only, never 'stalled': see
    ``_BLOCKED_MARKER_ALT`` for why a marker with no corresponding
    ``shared.task_statuses.TaskStatus`` value cannot participate in the rule.

    Gated on the cheap literal ``_BLOCKED_MARKER_RE`` rather than the union
    ``SNAPSHOT_STATUS_RE``: 'blocked' is itself a union marker, so this gate
    is strictly stronger, and a fact with no 'blocked' token returns the
    empty set after ONE substring scan instead of running five anchored
    patterns. That is what keeps the second pattern family off the hot path
    — see ``_BLOCKED_MARKER_RE``.

    Pure: no I/O, no side effects.
    """
    return extract_snapshot_edge_task_ids_by_marker_class(fact).blocked_ids


class SnapshotEdgeIds(NamedTuple):
    """One edge fact's extracted task ids, split by marker class (task 3037).

    ``blocked_ids`` is always a SUBSET of ``all_ids``. NOT because the
    families' marker alternations nest — that was the original claim and it
    was measured FALSE — but because ``_ids_within_union`` threads the union
    family's rejected spans into the blocked call and then intersects the two
    sets at the seam. See that function for both mechanisms and for why
    pattern nesting is not sufficient. (amendment, reviewer_comprehensive
    correctness-over-selection finding, task 3037)
    """

    #: Every id the fact asserts as active/pending/blocked/stalled/in-progress.
    all_ids: set[int]
    #: The subset of those the fact asserts as BLOCKED specifically.
    blocked_ids: set[int]


def extract_snapshot_edge_task_ids_by_marker_class(fact: str | None) -> SnapshotEdgeIds:
    """Extract BOTH id sets from *fact* in a single pass (task 3037).

    THE single extraction implementation in this module.
    ``extract_snapshot_edge_task_ids`` and
    ``extract_blocked_assertion_task_ids`` are thin accessors over it, so the
    three entry points cannot drift and no caller has to choose between
    "correct" and "cheap".

    Why one entry point rather than two calls. The extractor runs once per
    valid edge over the whole group's edge set, with no per-edge timeout, so
    its cost is a whole-cycle LIVENESS property — see the module docstring's
    performance section. ``sweep_stale_status_snapshot_edges`` needs both id
    sets for every edge (the union set to build its ``get_statuses`` census,
    the blocked set for selection rule 2 and the superseding write), and
    reaching them through two separate entry points meant walking each fact
    through the pipeline twice. Pinned by
    ``test_extraction_runs_exactly_once_per_edge``.

    What one pass does and does not buy. The two pattern families are still
    run — that is irreducible, since they answer different questions — but
    the whole-fact gate is evaluated once instead of twice, and the
    cheap-literal ``_BLOCKED_MARKER_RE`` pre-gate keeps the second family off
    the hot path entirely for the overwhelming majority of edges, which carry
    no 'blocked' token at all.

    *fact* is typed ``str | None`` because THIS function is where a missing
    fact is normalised, and callers read it straight off an untyped edge dict
    (``edge.get('fact')``). Pushing an ``or ''`` back to each call site would
    type-check equally well but cost an extra truthiness test per edge per
    call site — which is also the probe
    ``test_extraction_runs_exactly_once_per_edge`` counts pipeline entries
    with, so a call-site guard reads there as a second extraction pass.

    Pure: no I/O, no side effects.
    """
    fact = fact or ''

    if not SNAPSHOT_STATUS_RE.search(fact):
        # No marker of ANY class — neither family can match, and the blocked
        # family is a subset of this gate, so one scan settles both.
        return SnapshotEdgeIds(all_ids=set(), blocked_ids=set())

    union = _extract_ids(fact, _UNION_PATTERNS)
    if not _BLOCKED_MARKER_RE.search(fact):
        return SnapshotEdgeIds(all_ids=union.ids, blocked_ids=set())
    return SnapshotEdgeIds(
        all_ids=union.ids,
        blocked_ids=_ids_within_union(fact, _BLOCKED_PATTERNS, union),
    )


def extract_marker_task_ids(
    fact: str | None, adjective_alt: str | None, transitive_alt: str | None,
) -> set[int]:
    """The ids *fact* asserts under ONE marker alternation. (task 4851)

    The family is built by the shipped builder from alternations shaped like
    ``_ADJECTIVE_MARKER_ALT`` / ``_TRANSITIVE_MARKER_ALT``, so it carries every
    shipped guard. Its ids pass the same union seam as
    ``SnapshotEdgeIds.blocked_ids``, which keeps them a subset of
    ``extract_snapshot_edge_task_ids(fact)``. A reporting seam for probes that
    ask which ids a fact asserts as, say, pending; selection itself reads
    ``extract_snapshot_edge_task_ids_by_marker_class``.

    Pure: no I/O, no side effects.
    """
    fact = fact or ''
    if not SNAPSHOT_STATUS_RE.search(fact):
        return set()
    union = _extract_ids(fact, _UNION_PATTERNS)
    return _ids_within_union(fact, _marker_family(adjective_alt, transitive_alt), union)


@functools.cache
def _marker_family(adjective_alt: str | None, transitive_alt: str | None) -> _SnapshotPatterns:
    return _build_snapshot_patterns(adjective_alt, transitive_alt)


def _ids_within_union(
    fact: str, family: _SnapshotPatterns, union: _ExtractionResult,
) -> set[int]:
    """*family*'s ids on an already-gated *fact*, inside the union family's result.

    THE seam every narrower family passes through — the blocked family and
    ``extract_marker_task_ids`` alike — so a narrower family can never assert
    an id the union family did not.
    """
    # THREAD the union family's prepositional-complement rejections into the
    # narrower family. A rejected span is, by construction, a region
    # established to be a PREPOSITION'S COMPLEMENT, so no id inside it is the
    # copula's subject — whichever family, and whichever pattern, found it.
    # That is verbatim the argument _extract_ids already gives for applying a
    # rejection across all three anchored paths; it extends across the two
    # FAMILIES for exactly the same reason, and that is where it was missed.
    #
    # Load-bearing, not tidiness. rejected_spans is computed per call from
    # THIS family's plural_enum matches, and the blocked family's plural_enum
    # needs the literal 'blocked' marker — so an enumeration the union family
    # rejected on a 'pending'/'active'/'in progress'/'stalled' marker was
    # never built by the blocked family, suppressed nothing, and left its
    # individual arm free to claim a repeated 'task N' reference inside a
    # region already ruled out:
    #
    #     'Dependencies for tasks 1020 and task 1030 are pending in blocked
    #      status'  ->  union set(), blocked {1030}   (MEASURED, pre-fix)
    #
    # i.e. an edge asserting the DEPENDENCIES are pending was retired, and a
    # status attributed to a subject the fact never made a claim about.
    # (amendment, reviewer_comprehensive correctness-over-selection finding,
    # task 3037)
    narrowed = _extract_ids(
        fact, family, inherited_rejected_spans=union.rejected_spans,
    )
    # INTERSECT AT THE SEAM. Be honest about what this is: with the
    # threading above in place it is MEASURED to be a no-op across the
    # whole 800-fact _SUBSET_PROPERTY_CORPUS, so it is NOT the fix — it is
    # what makes SnapshotEdgeIds' containment claim true BY CONSTRUCTION
    # rather than by argument, at the cost of one set intersection on the
    # small minority of edges that clear the _BLOCKED_MARKER_RE pre-gate.
    #
    # It exists because the ORIGINAL 'by construction' argument (the
    # marker alternations nest, therefore the results nest) is UNSOUND for
    # regex families in general, independently of the bug above:
    # finditer is non-overlapping, so narrowing an alternation can delete
    # an earlier match and thereby UNSHADOW a later one the wider family
    # never reported. The intersection bounds that entire class of future
    # surprise in the safe direction — under-selection self-heals on the
    # next cycle, over-selection retires a live edge forever.
    return narrowed.ids & union.ids


class _ExtractionResult(NamedTuple):
    """One family's extracted ids plus the rejected spans it established.

    The spans are returned rather than kept private because they are the one
    piece of ALGORITHM DATA that must cross the family boundary — see
    ``_extract_ids``'s ``inherited_rejected_spans`` parameter.
    """

    #: Ids this family attributed to the copula's subject.
    ids: set[int]
    #: Enumeration spans established to be a preposition's complement,
    #: inherited from the caller plus this family's own rejections.
    rejected_spans: tuple[tuple[int, int], ...]


def _plural_enum_ids(
    fact: str,
    pattern: re.Pattern[str] = PLURAL_ENUM_SNAPSHOT_RE,
    *,
    guard: Callable[[str], bool] = _enumeration_is_prepositional_complement,
) -> tuple[set[int], list[tuple[int, int]]]:
    """The plural-enumeration arm, in ONE place. (task 3079; task 3949; task 3037)

    Returns ``(ids, rejected_spans)``: the ids the plural path yields, and the
    spans of every enumeration *guard* rejected. Both outputs are needed by
    the production extractor — a rejected span is, by construction, a region
    established to be a preposition's complement, so the drop it implies is
    applied to every other anchored path too (see ``_extract_ids``).

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

    *pattern* defaults to ``PLURAL_ENUM_SNAPSHOT_RE`` — i.e. the UNION
    family's plural arm, since that constant is bound from
    ``_UNION_PATTERNS`` — so the probe's one-positional-argument call keeps
    measuring exactly the arm it always measured. ``_extract_ids`` passes its
    own family's ``patterns.plural_enum`` explicitly instead, because the
    BLOCKED family narrows this pattern and must not silently fall back to
    the union's. (task 3037 made the patterns per-family; this parameter is
    how the task-3949 seam survives that.)

    Pure: no I/O, no side effects.
    """
    ids: set[int] = set()
    rejected_spans: list[tuple[int, int]] = []
    for enum in pattern.finditer(fact):
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


def _extract_ids(
    fact: str,
    patterns: _SnapshotPatterns,
    *,
    inherited_rejected_spans: Sequence[tuple[int, int]] = (),
) -> _ExtractionResult:
    """Run one already-gated *fact* through one marker family's five paths.

    The single shared implementation of steps 2-6 of
    ``extract_snapshot_edge_task_ids``'s documented algorithm (the gate is
    step 1 and stays with each caller, since the two families gate
    differently). Factored out at task 3037 so the union and blocked
    extractors cannot drift in their ALGORITHM the way hand-copied patterns
    would have let them drift in their GRAMMAR.

    ``inherited_rejected_spans`` carries the prepositional-complement
    rejections ANOTHER family already established on the same fact, and the
    returned ``rejected_spans`` carries this family's own on top of them.
    That threading is load-bearing, not an optimisation — see
    ``_ids_within_union``, and the warning about
    per-family algorithm data in this module's docstring. Sharing the pattern
    BUILDER does not share this: ``rejected_spans`` is computed here, per
    call, from THIS family's own ``plural_enum`` matches, so a span the union
    family rejects is one the narrower blocked family may never build at all.
    (amendment, reviewer_comprehensive correctness-over-selection finding,
    task 3037)

    Pure: no I/O, no side effects.
    """
    # The plural arm lives in _plural_enum_ids so task 3949's recall probe can
    # measure THIS code rather than a copy of it (see that function). THIS
    # family's plural pattern is passed explicitly — the blocked family
    # narrows it, and must not fall back to the union default.
    #
    # This family's OWN rejections are concatenated AFTER the inherited ones,
    # which reproduces exactly the order the in-place loop built before the arm
    # was extracted (inherited seeded first, own appended in match order). The
    # loop only ever APPENDED to rejected_spans and never read it, so hoisting
    # it out is order- and value-preserving; the spans are consumed as an
    # unordered membership test below in any case.
    ids, own_rejected_spans = _plural_enum_ids(fact, patterns.plural_enum)
    rejected_spans: list[tuple[int, int]] = [
        *inherited_rejected_spans,
        *own_rejected_spans,
    ]

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
    for pattern in (patterns.individual, patterns.genitive, patterns.status_phrase):
        ids.update(
            int(m.group(1))
            for m in pattern.finditer(fact)
            if not any(start <= m.start() < end for start, end in rejected_spans)
        )

    for intro in patterns.list_introducer.finditer(fact):
        # Subject guard for the aggregate path: a family may refuse a list
        # whose introducer is governed by a preceding task reference in the
        # same clause — the transitive-verb reading of 'Task 5 blocked tasks:
        # 142, 148'. Read off the family rather than branched on family
        # identity; see _list_is_governed_by_task_ref.
        if patterns.reject_list_after_task_ref and _list_is_governed_by_task_ref(
            fact[: intro.start()]
        ):
            continue
        segment = COUNT_QUANTITY_RE.sub(' ', _list_segment(fact, intro.end(), intro.group('open')))
        ids.update(int(tok) for tok in _BARE_DIGIT_RE.findall(segment))

    return _ExtractionResult(ids=ids, rejected_spans=tuple(rejected_spans))
