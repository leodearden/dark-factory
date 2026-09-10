"""Pinned plural-enumeration shape corpora, shared by two suites. (task 3949)

These lists are PLAIN DATA, deliberately: they are the pinned corpora for
``PLURAL_ENUM_SNAPSHOT_RE`` and ``_enumeration_is_prepositional_complement``,
and they have two consumers that must never drift apart.

1. ``test_stale_status_snapshot_edge_sweep.py`` parametrizes its guard tests
   off them — the shapes' original home, and still the authority on what the
   SHIPPED guard must do.
2. ``tests/test_measure_plural_enum_guard_recall.py`` re-validates task 3949's
   two candidate tightenings against them. Task 3949 requires each candidate be
   re-validated 'against the full precision-guard parametrization' before
   shipping, so that suite needs the same corpus, not a copy of it.

WHY A DATA MODULE RATHER THAN A COPY, OR MARKER INTROSPECTION. A hardcoded
second copy satisfies the re-validation requirement on the day it is written
and silently stops covering the full set the moment someone adds a shape --
the same silent-loss failure mode the sweep suite's own
``test_every_guarded_preposition_is_exercised_against_a_plural_head`` exists to
prevent for ``_ENUM_PREP_WORDS``. Reading the shapes off the other suite's
``pytest.mark.parametrize`` markers by test-FUNCTION NAME (the mechanism this
module replaces) kept the gate mechanical but coupled it to test scaffolding:
renaming an upstream test, moving a parametrize onto a fixture, or wrapping
values in ``pytest.param`` broke the importing module at IMPORT time, taking
down its whole suite rather than one test. Plain exported lists keep the
mechanical property -- a shape added here re-validates both candidates
automatically -- with none of that coupling.

ADDING A SHAPE: append it to the right list here. Both suites pick it up on the
next run, and no test-function name or marker internal is involved.

Import as ``from reconciliation.plural_enum_shapes import ...``. ``tests/``
carries no ``__init__.py`` while ``tests/reconciliation/`` does, so pytest puts
``tests/`` on ``sys.path`` for modules in BOTH directories and the package
import resolves identically from each.
"""
from __future__ import annotations

# Shapes the plural path must NOT extract. Any candidate tightening that admits
# one of these has re-opened over-selection -- the unrecoverable direction --
# and is disqualified outright.
PRECISION_GUARD_SHAPES: list[str] = [
    # transitive-verb reading — the mandatory copula must refuse it,
    # exactly as INDIVIDUAL_SNAPSHOT_RE's transitive arm does. This
    # is a permanently-true historical fact, not a status snapshot.
    'Tasks 1020, 1030, and 1031 blocked task 5.',
    'Tasks 1020 and 1030 block the merge queue.',
    # terminal status — gate short-circuits
    'Tasks 1020, 1030, and 1031 are done.',
    # negation / past-exit, refused by the closed-class _ADVERB_ALT
    'Tasks 1020, 1030, and 1031 are no longer pending.',
    # marker present but NOT adjacent to the enumeration+copula: the
    # BRANCH is active, not the tasks
    'Tasks 1020 and 1030 were merged into the active branch.',
    # ---------------------------------------------------------- #
    # Prepositional-complement over-selection (amendment,
    # reviewer_comprehensive correctness-precision finding, task
    # 3079). In each of these the plural NP is the COMPLEMENT OF A
    # PREPOSITION, so the copula's real subject is an outer head
    # noun and the marker describes THAT, not the tasks. Every one
    # is a permanently-true historical/meta fact, so the sweep
    # would retire it the instant any referenced id went terminal —
    # the over-selection direction the module docstring forbids,
    # and the same class task 3042 closed for LIST_INTRODUCER_RE.
    # ---------------------------------------------------------- #
    # singular outer head — 'is' cannot agree with a plural subject,
    # so plural-agreement on the copula alone already refuses these
    'The merge of tasks 1020 and 1030 is blocked.',
    'Review of tasks 1020 and 1030 is pending.',
    'Documentation for tasks 1020 and 1030 is still pending.',
    'Verification of tasks 1020, 1030, and 1031 is pending.',
    # PLURAL outer head — the plural copula agrees with the OUTER
    # head here, so these survive a plural-agreement-only fix and
    # are what force the second remedy (preposition lookbehinds)
    'Dependencies for tasks 1020 and 1030 are blocked.',
    'The merges of tasks 1020 and 1030 are blocked.',
    'Reviews of tasks 1020, 1030, and 1031 are pending.',
    'Work on tasks 1020 and 1030 is blocked.',
    'The dependency between tasks 1020 and 1030 is blocked.',
    # DETERMINER between the preposition and the list noun. These
    # defeated the original fixed-width-lookbehind spelling of the
    # guard wholesale — one extremely common word re-opened every
    # entry in the preposition list at once, and the suite's own
    # positive case 'The tasks 1020 and 1030 are blocked.' is the
    # head of exactly this shape. Plural agreement does not save
    # them: 'Statuses'/'Reviews'/'Notes' are plural, which is the
    # residue the preposition check exists to cover. (amendment,
    # reviewer_comprehensive correctness-precision finding, task
    # 3079)
    'Statuses of the tasks 1020 and 1030 are blocked.',
    'Reviews for the tasks 1020 and 1030 are pending.',
    'Notes about the tasks 1020 and 1030 are pending.',
    'Reviews of these tasks 1020 and 1030 are pending.',
    'Notes regarding all tasks 1020 and 1030 are pending.',
    # multi-space / newline gap before the list noun — the other
    # defect the fixed-offset lookbehind could not see
    'Reviews for  the\n  tasks 1020 and 1030 are pending.',
    # NON-DETERMINER intervening words. The determiner cases above
    # were first fixed with a six-word slot
    # ('the|these|those|all|our|its'), which was a second closed
    # vocabulary and failed the same way one word over. The gap is an
    # OPEN class, so one case per class it admits — quantifier, bare
    # adjective, possessive, determiner stack, participle — plus the
    # THREE-word gaps that a merely-bounded slot (the review's
    # suggested '{0,2} arbitrary words') would still admit, which is
    # why the guard is clause-scoped and unbounded instead.
    # (amendment, reviewer_comprehensive correctness-precision
    # finding, task 3079)
    'Statuses of some tasks 1020 and 1030 are pending.',
    'Dependencies for both tasks 1020 and 1030 are blocked.',
    'Notes about a few tasks 1020 and 1030 are pending.',
    'Reviews of all the tasks 1020 and 1030 are pending.',
    'Notes on remaining tasks 1020 and 1030 are pending.',
    'Statuses of open tasks 1020 and 1030 are pending.',
    'Blockers for downstream tasks 1020 and 1030 are pending.',
    "Reviews of Leo's tasks 1020 and 1030 are pending.",
    'Statuses of quite a few tasks 1020 and 1030 are pending.',
    'Blockers for down-stream, still-unmerged tasks 1020 and 1030 are pending.',
    # HYPHENATED negation / past-exit reaching the marker through
    # _COMPOUND_PREFIX rather than the closed-class adverb slot
    'Tasks 1020 and 1030 are un-blocked.',
    'Tasks 1020 and 1030 are previously-blocked.',
    # NON-SENTENCE-FINAL punctuation between the preposition and the
    # list noun. Each of these over-selected while the clause-break
    # class still contained ':', the paren/bracket family and the
    # quote family: the break truncated the backward scan short of
    # the governing preposition, so the guard saw a clause with no
    # preposition in it and admitted the match. A colon typically
    # INTRODUCES the complement its preposition governs, and a
    # parenthetical or quoted aside is an interpolation inside the
    # clause rather than a new one — neither ends government, so
    # neither may be a break. (amendment, reviewer_comprehensive
    # correctness-precision finding, task 3079)
    'Statuses of the following: tasks 1020 and 1030 are pending.',
    'Reviews of the following (still open): tasks 1020 and 1030 are pending.',
    'Statuses of the "next" tasks 1020 and 1030 are pending.',
    'Blockers for the merge lane [df] tasks 1020 and 1030 are pending.',
    # INTRA-TOKEN '.' is not sentence-final punctuation, so it must
    # not count as a clause break either: a '.' flanked by
    # alphanumerics on both sides (filename extension, version
    # string, dotted module path, dotted section number) ends no
    # sentence, so treating it as a break truncates the backward
    # scan short of the governing preposition — the same
    # over-selection the punctuation-family exclusions above guard,
    # just for a '.' occurrence rather than a different character.
    # (amendment, reviewer_comprehensive correctness-precision
    # finding, task 4149)
    'Reviews for verify_cmd.py tasks 1020 and 1030 are pending.',
    # CONTROL: identical but for the extension; already passed before
    # this amendment and must keep passing.
    'Reviews for verify_cmd tasks 1020 and 1030 are pending.',
    'Blockers on scheduler.py tasks 1020 and 1030 are in progress.',
    'Statuses of the v1.2 tasks 1020 and 1030 are pending.',
    'Reviews for section 4.2.1 tasks 1020 and 1030 are pending.',
    'Statuses of tasks in df.core tasks 1020 and 1030 are pending.',
    # UNICODE flanking: _is_intra_token_dot's docstring claims
    # str.isalnum() makes the test unicode-aware by construction,
    # so a non-ASCII filename must be recognized as intra-token
    # exactly like an ASCII one — pin that claim by behaviour
    # rather than by prose. (amendment, reviewer_comprehensive
    # test-coverage finding, task 4149)
    'Reviews for café.py tasks 1020 and 1030 are pending.',
]


# Guard-rejected enumerations whose TAIL must not stay extractable via the
# individual path once the plural match is suppressed.
GUARD_REJECTED_SUPPRESSION_SHAPES: list[str] = [
    'Reviews for tasks 1020, task 1030 and task 1031 are pending.',
    'Statuses of the tasks 1020 and task 1030 are blocked.',
    'Notes about tasks 1020, task 1030 and task 1031 are pairwise-stalled.',
    "Reviews of Leo's tasks 1020 and task 1030 are pending.",
]


# Subject-position enumerations the shipped guard extracts, paired with the
# exact id set expected. No candidate tightening may disturb any of these --
# asserting the id SET (not merely 'unchanged') catches a candidate that yields
# a different set for the same fact.
SUBJECT_POSITIVE_SHAPES: list[tuple[str, set[int]]] = [
    # These two are the load-bearing pair: both end in the letters
    # 'on'/'ion' immediately before ' tasks', so they prove the
    # preposition guard is \b-anchored and does not fire on a word
    # that merely ENDS in a preposition.
    ('Migration tasks 1020 and 1030 are pending.', {1020, 1030}),
    ('Verification tasks 1020 and 1030 are pending.', {1020, 1030}),
    # a marker-qualified plural head is still a subject
    ('Blocked tasks 1020 and 1030 are pending.', {1020, 1030}),
    # a determiner before the plural head is not a preposition
    ('The tasks 1020 and 1030 are blocked.', {1020, 1030}),
    # 'remain' must survive the copula narrowing to plural agreement
    ('Tasks 1020 and 1030 remain pending.', {1020, 1030}),
    # CLAUSE SCOPE. The preposition guard is unbounded within a
    # clause, so these pin the other edge of it: strong punctuation
    # ends the span a preposition governs, and an enumeration opening
    # a new sentence/clause really is its own copula's subject even
    # though a listed preposition appears earlier in the fact. Without
    # a clause reset the guard would swallow the whole prefix and
    # disable the plural path for any multi-sentence fact. (amendment,
    # reviewer_comprehensive correctness-precision finding, task 3079)
    ('Reviews for the branch are done. Tasks 1020 and 1030 are pending.',
     {1020, 1030}),
    ('Blocked on review; tasks 1020 and 1030 are pending.', {1020, 1030}),
    # ...and the other edge of narrowing the break class to
    # sentence-final punctuation: a colon-preambled genuine snapshot
    # is unaffected, because the longer clause it now scans still
    # contains no listed preposition for the guard to fire on.
    # (amendment, reviewer_comprehensive correctness-precision
    # finding, task 3079)
    ('Note: tasks 1020 and 1030 are pending.', {1020, 1030}),
    # ...and the other edge of narrowing WHICH '.' occurrences count
    # as a break (task 4149): a real sentence-ending '.' must still
    # break the clause even when a dotted (intra-token) token
    # precedes it in the same fact — proving the narrowing did not
    # disable '.' as a break wholesale, only intra-token occurrences
    # of it. The second case also pins that '!' stayed unconditional.
    ('Reviews for verify_cmd.py are done. Tasks 1020 and 1030 are pending.',
     {1020, 1030}),
    ('Blockers on scheduler.py are resolved! Tasks 1020 and 1030 are pending.',
     {1020, 1030}),
    # ...and specifically the WALK'S RETRY behaviour: the loop must
    # continue past an intra-token '.' to an EARLIER genuine
    # sentence break, not give up at the first intra-token '.' it
    # meets. Every other intra-token case above either never enters
    # the loop (the break is already non-intra-token) or enters it
    # and exhausts to -1 (no real break precedes); only this one has
    # an earlier break to retry TO. Verified this pins the retry:
    # replacing the `while` with a single
    # `if dot > hard and _is_intra_token_dot(...): dot = -1` (give
    # up instead of retrying) still passes every other case here,
    # but turns this one from {1020, 1030} into set() — the scan
    # runs back over 'for' in the PREVIOUS sentence instead of
    # stopping at the '.' after 'branch'. (amendment,
    # reviewer_comprehensive test-coverage finding, task 4149)
    ('Notes v1.2 for the branch. Statuses v3.4 tasks 1020 and 1030 are pending.',
     {1020, 1030}),
]


# The documented under-selection: a sentence-initial adverbial preamble shares
# the clause with its own preposition and suppresses a genuine subject-position
# snapshot behind it. This is the fail-safe direction, so it is documented
# behaviour rather than a defect -- and it is what a tightening would be FOR.
ADVERBIAL_PREAMBLE_SHAPES: list[str] = [
    # 'of' inside a date stamp — the shape the finding's own
    # motivating fact carries ("... is pending as of 2026-07-14...")
    'As of 2026-08-09, tasks 1020 and 1030 are pending.',
    # 'in' inside a location-scoping preamble
    'In the merge queue, tasks 1020 and 1030 are pending.',
    # 'for' inside a cycle-scoping preamble
    'For this cycle, tasks 1020 and 1030 are pending.',
]
