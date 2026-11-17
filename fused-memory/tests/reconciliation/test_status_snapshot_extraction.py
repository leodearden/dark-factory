"""Tests for fused_memory.reconciliation.status_snapshot_extraction, the pure
lexical layer of the stale task-status snapshot edge sweep.

Covers:
- extract_snapshot_edge_task_ids: pure lexical extractor — returns the
  specific task ids a status-snapshot edge asserts as active/pending/
  blocked/in-progress; returns the empty set for text with no such status
  marker, and for pure count-only snapshots with no specific task-id (the
  stale-by-design audit trail the sweep must never touch). Includes the
  task-3042 'blocked' regression class: the closed-class-connective
  individual form ('Task N is in a blocked status ...'), the status-noun-
  anchored phrase form for open-class gaps ('Task N is deliberately parked
  in blocked status ...'), and the precision guards bounding both (no bind
  inside 'unblocked', no cross-clause/cross-comma gap).
- extract_blocked_assertion_task_ids: the blocked-scoped family, and its
  subset property against the union family.
- the plural-enumeration and clause-break walks' performance bounds.
- extract_marker_task_ids: the per-marker reporting seam.
- the layering: importing this module never loads the sweep module.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.reconciliation.status_snapshot_extraction import (
    _ENUM_PREP_WORDS,
    _last_clause_break,
    extract_blocked_assertion_task_ids,
    extract_marker_task_ids,
    extract_snapshot_edge_task_ids,
    extract_snapshot_edge_task_ids_by_marker_class,
)
from reconciliation.counting_str import CountingStr

# The pinned shape corpora live in a plain data module so task 3949's
# candidate-tightening suite can re-validate against the SAME lists rather than
# a copy that silently stops covering additions. Append new shapes there.
from reconciliation.plural_enum_shapes import (
    ADVERBIAL_PREAMBLE_SHAPES,
    GUARD_REJECTED_SUPPRESSION_SHAPES,
    PRECISION_GUARD_SHAPES,
    SUBJECT_POSITIVE_SHAPES,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


def test_extraction_module_imports_nothing_from_the_sweep():
    """The extraction layer sits below the sweep: importing it must never load
    stale_status_snapshot_edge_sweep.

    Probed in a FRESH interpreter: this test process has already imported the
    sweep, so an in-process ``sys.modules`` check would pass vacuously. The
    extraction round-trip at the end keeps the probe itself from silently
    no-opping if the import ever stops resolving. Prepending this tree's
    ``src`` to PYTHONPATH pins the probe to it, since PYTHONPATH outranks
    whichever checkout the interpreter's editable install points at; the
    env construction follows
    ``shared/tests/test_pure_stdlib_leaves.py::run_python_child``.
    """
    tree_under_test_src = Path(__file__).resolve().parents[2] / 'src'
    probe = (
        'import sys; '
        'import fused_memory.reconciliation.status_snapshot_extraction as x; '
        "assert 'fused_memory.reconciliation.stale_status_snapshot_edge_sweep' "
        'not in sys.modules; '
        "assert x.extract_snapshot_edge_task_ids('Task 5 is pending') == {5}"
    )
    result = subprocess.run(
        [sys.executable, '-c', probe],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            'PYTHONPATH': f'{tree_under_test_src}{os.pathsep}'
            + os.environ.get('PYTHONPATH', ''),
        },
    )
    assert result.returncode == 0, result.stderr


class TestExtractSnapshotEdgeTaskIds:
    """extract_snapshot_edge_task_ids(fact) returns the specific task ids a
    status-snapshot edge asserts as active/pending/in-progress.

    Returns the empty set both when no active/pending/in-progress status
    marker is present at all, AND for pure count-only snapshots with no
    specific task-id reference — the out-of-scope, stale-by-design audit
    trail (Snapshot Discipline) that this sweep must never touch.
    """

    def test_individual_form_active_pending(self):
        """'Task 142 is an active pending task' -> {142}."""
        assert extract_snapshot_edge_task_ids('Task 142 is an active pending task') == {142}

    def test_individual_form_in_progress(self):
        """'Task 148 is in-progress' -> {148}."""
        assert extract_snapshot_edge_task_ids('Task 148 is in-progress') == {148}

    def test_aggregate_list_form_brackets(self):
        """'The active pending tasks are [142, 148, 150]' -> {142, 148, 150}."""
        result = extract_snapshot_edge_task_ids(
            'The active pending tasks are [142, 148, 150]'
        )
        assert result == {142, 148, 150}

    def test_aggregate_list_form_colon(self):
        """'active/in-progress tasks: 142, 148' -> {142, 148}."""
        result = extract_snapshot_edge_task_ids('active/in-progress tasks: 142, 148')
        assert result == {142, 148}

    def test_count_only_no_specific_id_returns_empty(self):
        """'There are 8 tasks in progress' -> set() (out-of-scope count-only snapshot)."""
        assert extract_snapshot_edge_task_ids('There are 8 tasks in progress') == set()

    def test_count_pair_returns_empty(self):
        """'1505 done / 148 cancelled' -> set() ('148' is a count operand, not an id)."""
        assert extract_snapshot_edge_task_ids('1505 done / 148 cancelled') == set()

    def test_status_gate_done_returns_empty(self):
        """'Task 5 is done' -> set() (no active/pending/in-progress marker)."""
        assert extract_snapshot_edge_task_ids('Task 5 is done') == set()

    def test_status_gate_merge_commit_returns_empty(self):
        """'Task 7 landed as merge commit' -> set() (no active/pending/in-progress marker)."""
        assert extract_snapshot_edge_task_ids('Task 7 landed as merge commit') == set()

    def test_incidental_numbers_excluded(self):
        """Date/commit-hash digits near a valid reference must not be swept in as ids."""
        result = extract_snapshot_edge_task_ids(
            'Task 142 is pending as of 2026-07-14 (commit ab12cd)'
        )
        assert result == {142}

    def test_individual_form_no_copula_in_progress(self):
        """'Task 5 in progress' -> {5} (no linking verb).

        Regression test (reviewer_comprehensive correctness-recall finding,
        task 2613): the status-word-adjacent digit used to be stripped by
        COUNT_QUANTITY_RE (it matches '\\d+\\s+in[-\\s]?progress') before id
        extraction ever ran, silently dropping this id.
        """
        assert extract_snapshot_edge_task_ids('Task 5 in progress') == {5}

    def test_individual_form_no_copula_pending(self):
        """'Task 5 pending' -> {5} (no linking verb).

        Regression test (reviewer_comprehensive correctness-recall finding,
        task 2613): same COUNT_QUANTITY_RE false-negative class as the
        'in progress' case above, via '\\d+\\s+pending'.
        """
        assert extract_snapshot_edge_task_ids('Task 5 pending') == {5}

    def test_incidental_status_word_describing_something_else_excluded(self):
        """'Task 142 landed on the active branch' -> set().

        'active' describes the BRANCH, not task 142's status. Regression
        test (reviewer_comprehensive correctness-precision finding, task
        2613): the status marker must be anchored to its own task reference
        rather than merely co-occurring anywhere in the fact, else this
        genuinely-terminal fact ('landed') would be wrongly treated as an
        active/pending snapshot of task 142.
        """
        result = extract_snapshot_edge_task_ids('Task 142 landed on the active branch')
        assert result == set()

    def test_blocked_status_individual_form_task_2885_repro(self):
        """'Task 2885 is in a blocked status due to merge_retry_pending.' -> {2885}.

        Literal fact text from the task-2885 repro (run
        063f3e99-27f5-4bad-b6e6-d4a6804af09c). TWO separate gaps caused this
        fact to be invisible to the sweep before task 3042: 'blocked' was
        absent from the status-marker alternation, AND the preposition +
        article "in a" sitting between the copula and the marker was not a
        permitted connective (INDIVIDUAL_SNAPSHOT_RE previously allowed only
        an optional copula and/or a bare article). Both gaps had to close
        for this fact to match.
        """
        result = extract_snapshot_edge_task_ids(
            'Task 2885 is in a blocked status due to merge_retry_pending.'
        )
        assert result == {2885}

    def test_blocked_bare_copula_form(self):
        """'Task 2885 is blocked.' -> {2885}."""
        assert extract_snapshot_edge_task_ids('Task 2885 is blocked.') == {2885}

    def test_blocked_no_article_form(self):
        """'Task 2885 is in blocked status.' -> {2885}."""
        assert extract_snapshot_edge_task_ids('Task 2885 is in blocked status.') == {2885}

    def test_aggregate_blocked_list_form(self):
        """'Blocked tasks: 142, 148' -> {142, 148}.

        Before task 3042, the gate short-circuit (SNAPSHOT_STATUS_RE not
        matching 'blocked') suppressed the aggregate list-form path too, even
        though the aggregate extraction logic itself has nothing to do with
        the individual-form connective. Widening the gate to admit 'blocked'
        unlocks this aggregate form for free.
        """
        result = extract_snapshot_edge_task_ids('Blocked tasks: 142, 148')
        assert result == {142, 148}

    def test_count_only_blocked_snapshot_returns_empty(self):
        """'There are 3 blocked tasks' -> set().

        Count-only snapshots stay out of scope per Snapshot Discipline even
        with the widened gate — no list introducer, no specific task-id
        reference. Guard: passes both before and after the task-3042 change.
        """
        assert extract_snapshot_edge_task_ids('There are 3 blocked tasks') == set()

    def test_blocked_status_phrase_with_intervening_words_task_2885_repro(self):
        """'Task 2885 is deliberately parked in blocked status under escalation
        esc-2885-17 pending human adjudication.' -> {2885}.

        The SECOND literal edge fact from the task-2885 repro. Still set()
        after task 3042's first change (the widened INDIVIDUAL_SNAPSHOT_RE
        connective alone) because "deliberately parked" is an open-class gap
        between the copula and the preposition that the closed-class
        connective deliberately refuses. The phrase form is admitted only
        because the marker ('blocked') is immediately followed by the
        literal noun 'status' — that noun is what makes the span a status
        assertion rather than an incidental mention.
        """
        result = extract_snapshot_edge_task_ids(
            'Task 2885 is deliberately parked in blocked status under '
            'escalation esc-2885-17 pending human adjudication.'
        )
        assert result == {2885}

    def test_open_class_gap_without_status_noun_excluded(self):
        """'Task 142 landed on the blocked branch' -> set().

        An intervening verb with no 'status' noun following the marker must
        never bind — the task-2613 precision invariant, restated for the
        'blocked' marker.
        """
        result = extract_snapshot_edge_task_ids('Task 142 landed on the blocked branch')
        assert result == set()

    def test_unblocked_is_not_a_blocked_marker(self):
        """'Task 9 unblocked the merge queue' -> set().

        \\b prevents the marker from matching inside 'unblocked'.
        """
        assert extract_snapshot_edge_task_ids('Task 9 unblocked the merge queue') == set()

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 5 blocked the merge queue',
            'Task 142 blocked the release',
            'Task 2885 blocked task 3001 for three days',
            'Task 9 blocked the merge queue',
        ],
    )
    def test_blocked_as_bare_transitive_verb_excluded(self, fact):
        """'Task N blocked <object>' -> set().

        Regression guard (reviewer_comprehensive correctness-precision
        finding, task 3042). Unlike the adjective-only markers
        ('active'/'pending'/'in progress'), 'blocked' is also a common
        transitive past-tense verb. INDIVIDUAL_SNAPSHOT_RE's copula is
        optional for the adjective arm, so before the split alternation the
        bare-verb reading bound and every fact below wrongly yielded an id.

        These are permanently-true HISTORICAL facts ('task 5 blocked the
        merge queue' stays true forever), not status snapshots — the sweep
        would have retired them the moment task 5/142/2885 went done or
        cancelled, the over-selection direction the module docstring
        forbids. Same hazard, and the same copula-only remedy, as task
        2824's transitive-verb caveat on
        task_filter.PRESENT_TENSE_COMPLETION_RE.

        Note the last case is the false-positive twin of
        test_unblocked_is_not_a_blocked_marker: that test only exercises
        the \\b-protected 'unblocked' spelling, so it never covered the
        bare-verb 'blocked' reading of the same sentence shape.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 5 blocked these tasks: 142, 148',
            'Task 5 blocked the merge of tasks [142, 148]',
            'Task 5 blocked the release for tasks: 142, 148',
        ],
    )
    def test_aggregate_path_not_reachable_by_bare_transitive_verb(self, fact):
        """'Task 5 blocked <words> tasks: 142, 148' -> set().

        Regression guard (reviewer_comprehensive correctness-precision
        finding, task 3042). The aggregate list path used to be anchored to
        nothing but the whole-fact gate (SNAPSHOT_STATUS_RE), so widening
        that gate with 'blocked' routed straight around the transitive-verb
        hardening added to INDIVIDUAL_SNAPSHOT_RE — these facts yielded
        {142, 148} even though the individual form correctly refused the
        same bare-verb reading. Each returned set() before 'blocked' joined
        the gate, so the gate was the only thing holding them back.

        These are permanently-true historical facts; the sweep would have
        retired them the moment 142 or 148 reached a terminal status.
        LIST_INTRODUCER_RE now requires the marker immediately before the
        list noun, so the intervening open-class words ('these', 'the merge
        of', 'the release for') break adjacency.

        Distinct from test_blocked_as_bare_transitive_verb_excluded, which
        covers the individual form only — none of its cases carry a list
        introducer, so it never exercised this path.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            ('Task 2885 is blocked.', {2885}),
            ('Task 2885 was blocked', {2885}),
            ('Task 2885 remains blocked', {2885}),
            ('Task 2885 is a blocked task', {2885}),
            ('Task 2885 is currently blocked', {2885}),
            ('Task 2885 is still blocked', {2885}),
            ('Task 5 is currently active', {5}),
        ],
    )
    def test_blocked_in_copula_form_still_selected(self, fact, expected):
        """The copula/article forms of 'blocked' must keep matching.

        Pins the recall side of the transitive-verb split: restricting
        'blocked' to a mandatory copula/article connective must not cost
        any genuine blocked-status assertion, which is the whole point of
        task 3042.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    def test_status_phrase_does_not_cross_clause_boundary(self):
        """'Task 5 is done. Task 9 is in blocked status' -> {9}.

        The bounded gap must not reach across the sentence boundary and drag
        in task 5's already-terminal snapshot.
        """
        result = extract_snapshot_edge_task_ids(
            'Task 5 is done. Task 9 is in blocked status'
        )
        assert result == {9}

    def test_status_phrase_does_not_cross_comma(self):
        """'Task 5 landed, work is in blocked status' -> set().

        The comma must be the ONLY thing preventing the match, else this
        test cannot fail for the reason it names.

        (amendment, reviewer_comprehensive test-quality suggestion, task
        3042.) The previous fact here — 'Task 5 landed, then 9 shipped in
        active branch' — returned set() for three reasons unrelated to the
        comma: bare '9' is not a TASK_REF_RE reference, 'active branch' is
        not followed by the literal noun 'status' so the phrase form could
        not fire at all, and the gap from 'Task 5' would need 4+ words. It
        would still have passed had the gap been changed to absorb commas
        — dead coverage for the invariant it advertised.

        This fact discriminates: 'landed', 'work', 'is' is exactly 3 gap
        words followed by ' in blocked status', so it would yield {5} if
        the gap could absorb the comma, and set() only because it cannot.
        """
        result = extract_snapshot_edge_task_ids(
            'Task 5 landed, work is in blocked status'
        )
        assert result == set()

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 2885 is no longer in blocked status',
            'Task 2885 was previously in blocked status',
            'Task 2885 was briefly in blocked status, then merged',
            'Task 142 is not in blocked status',
            'Task 5 was never in blocked status',
            'Task 5 is formerly in pending status',
        ],
    )
    def test_status_phrase_gap_does_not_absorb_negation_or_past_exit(self, fact):
        """Negated / past-exit phrasings -> set().

        Regression guard (reviewer_comprehensive correctness-precision
        finding, task 3042). The phrase form's open-class gap used to
        absorb 'not'/'no longer'/'previously'/'briefly', which invert the
        assertion rather than padding it, so all of these yielded an id.
        Each is a permanently-true HISTORICAL fact — and a
        blocked->unblocked transition is exactly the event that writes such
        an edge, the same transition class the task-2885 repro is about, so
        this is common phrasing here rather than exotic.

        Note the asymmetry this closes: the individual form's closed-class
        connective already refused the same semantics for free ('Task 2885
        is no longer blocked' -> set()), because these tokens were never in
        _ADVERB_ALT. Only the permissive phrase path lost that protection.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    def test_status_phrase_still_matches_non_negated_qualifiers(self):
        """'Task 5 remains in blocked status' -> {5}.

        The negation guard excludes a specific closed list, not qualifiers
        generally — an ordinary gap word like 'remains' must still pass, or
        the guard would have cost recall rather than only precision.
        """
        assert extract_snapshot_edge_task_ids('Task 5 remains in blocked status') == {5}

    def test_status_phrase_excludes_following_head_noun(self):
        """'Task 142 is tracked in the pending status report' -> set().

        Here 'status' modifies the head noun 'report' — it is a pending
        status REPORT, not a pending task — so the span is not a status
        assertion about task 142. This is the same 'binds to a noun'
        failure that got 'in'/'the' reverted from INDIVIDUAL_SNAPSHOT_RE,
        reaching the phrase form by a different route.

        SNAPSHOT_STATUS_PHRASE_RE's trailing lookahead requires the
        marker+status span to END its noun phrase (punctuation, end of
        string, or a preposition/subordinator), which excludes it.
        'status report' is common phrasing in this repo's memory corpus,
        so this is not hypothetical. (amendment, reviewer_comprehensive
        suggestion, task 3042)
        """
        result = extract_snapshot_edge_task_ids(
            'Task 142 is tracked in the pending status report'
        )
        assert result == set()

    def test_status_phrase_gap_internal_subject_is_known_over_selection(self):
        """'Task 5 depends on work in blocked status' -> {5}.

        DOCUMENTED KNOWN BEHAVIOUR, not an endorsement: the WORK is
        blocked, not task 5, so this is an over-selection the phrase form
        does not catch. The gap ('depends on work') contains the real
        subject of the 'in blocked status' phrase, and distinguishing that
        requires knowing 'work' is a nominal subject — beyond lexical
        matching at this altitude.

        Pinned as a test so the residual stays visible rather than being
        claimed away by a comment (the previous comment asserted the
        status-noun requirement cost NO precision, which was an
        overstatement). If a future change fixes this, update the
        assertion to set() — a failure here is a signal to re-read
        SNAPSHOT_STATUS_PHRASE_RE's residual notes, not necessarily a
        regression. (amendment, reviewer_comprehensive suggestion, task
        3042)
        """
        result = extract_snapshot_edge_task_ids(
            'Task 5 depends on work in blocked status'
        )
        assert result == {5}

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            ('Task 5 blocks task 9 in blocked status', {9}),
            ('Task 5 supersedes task 9 in pending status', {9}),
            ('Task 5 unblocks task 9 in in-progress status', {9}),
            # the 'df N' spelling of a reference was affected identically
            ('Task 5 supersedes df 9 in pending status', {9}),
            # ...and the OUTER reference's spelling is irrelevant to the bug
            ('df 5 blocks task 9 in blocked status', {9}),
            ('#5 blocks task 9 in blocked status', {9}),
            # both error directions plus an unrelated individual-form id
            (
                'Task 5 blocks task 9 in blocked status and task 11 is pending',
                {9, 11},
            ),
        ],
    )
    def test_status_phrase_gap_does_not_absorb_intervening_task_reference(
        self, fact, expected
    ):
        """An intervening task reference must not be swallowed by the gap.

        Regression guard (reviewer_comprehensive correctness-precision
        finding, task 3042). SNAPSHOT_STATUS_PHRASE_RE's open-class {0,3}
        gap could span a whole second task reference, so the 'in <marker>
        status' phrase bound to the OUTER task rather than the one it
        actually describes.

        This is strictly worse than an ordinary over-selection, because
        re.finditer is non-overlapping: the wrong match CONSUMED the inner
        reference, so the correct id was never extracted either. One match
        produced both error directions at once — 'Task 5 blocks task 9 in
        blocked status' yielded {5} where it must yield {9}: over-selection
        on 5 (a permanently-true historical fact the sweep would retire the
        moment 5 went terminal) AND silent under-selection on 9 (the edge
        that genuinely IS a blocked-status snapshot, invisible to the
        sweep). Every case here yielded the outer id before _GAP_NO_TASK_REF
        was wired into the gap.

        The last case pins that the fix is local to the phrase path: 'task
        11 is pending' is still picked up by INDIVIDUAL_SNAPSHOT_RE, so the
        expected set is {9, 11} and not merely {9}.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            ('Task 5 depends on task 9 in active status', {9}),
            ('Task 5 blocks #9 in blocked status', {9}),
        ],
    )
    def test_intervening_reference_shapes_that_bounded_the_bug(
        self, fact, expected
    ):
        """The two shapes an intervening reference could NOT be absorbed in.

        Unlike the cases above, these two were already correct before
        _GAP_NO_TASK_REF existed — they are boundary pins, not regression
        evidence, and are labelled as such deliberately. They record what
        bounded the bug, so a future widening reintroduces a failure here
        rather than silently resurrecting it:

        - a 4+-word gap ('depends', 'on', 'task', '9') exceeds {0,3}, so
          widening the gap bound would re-open this class;
        - '#9' cannot be absorbed because the gap's '\\w+' does not match
          '#', so admitting punctuation into the gap token class would too.

        Both must keep yielding the INNER id.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    def test_task_ref_exclusion_is_word_initial_not_substring(self):
        """'Task 5 is multitasked in blocked status' -> {5}.

        _GAP_NO_TASK_REF is a per-gap-word lookahead anchored at the START
        of the gap word, not a substring test: an ordinary open-class word
        that merely CONTAINS 'task' must still pass, or the guard would have
        cost recall rather than only precision. Same shape of pin as
        test_status_phrase_still_matches_non_negated_qualifiers.
        """
        assert (
            extract_snapshot_edge_task_ids('Task 5 is multitasked in blocked status')
            == {5}
        )

    def test_plural_list_noun_in_gap_is_a_documented_under_selection(self):
        """'Task 5 blocks tasks 9 in blocked status' -> set().

        DOCUMENTED KNOWN BEHAVIOUR. _GAP_NO_TASK_REF's '\\w*\\b' tail makes
        its vocabulary deliberately WIDER than TASK_REF_RE's: 'tasks' is
        excluded from the gap even though the plural is not itself a
        reference (TASK_REF_RE's '\\btask\\b' does not match it). So this
        fact now yields nothing at all rather than the wrong {5} it yielded
        before.

        That is the fail-safe direction this module prefers throughout —
        under-selection self-heals next cycle or is caught by Stage 2,
        whereas over-selection wrongly retires a true fact. Pinned so the
        trade is visible; narrowing the tail to '\\b' would trade it back.
        """
        assert extract_snapshot_edge_task_ids(
            'Task 5 blocks tasks 9 in blocked status'
        ) == set()

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 5 is in the active branch',
            'Task 7 is in the pending merge queue',
            'Task 5 is in a blocked directory',
            'Task 5 is the pending review owner',
            'Task 142 is in the active sprint retrospective',
        ],
    )
    def test_prepositional_noun_phrase_is_not_a_status_assertion(self, fact):
        """'Task N is in the <marker> <noun>' -> set().

        Regression guard (reviewer_comprehensive correctness-precision
        finding, task 3042). An interim revision of this task widened
        INDIVIDUAL_SNAPSHOT_RE's connective to admit the preposition 'in'
        and the article 'the' so it could reach the repro fact 'Task 2885
        is in a blocked status ...'. That made the connective span a full
        prepositional noun phrase, so the marker bound to a NOUN the task
        is merely located in (the active BRANCH, the pending merge QUEUE,
        a blocked DIRECTORY, the pending review OWNER) rather than to the
        task's own status — every fact below wrongly yielded an id.

        This is the over-selection direction the module docstring forbids:
        'Task 142 is in the active sprint retrospective' stays true
        forever, but the sweep would retire it the moment 142 went done.
        The repro facts are reached instead by SNAPSHOT_STATUS_PHRASE_RE,
        whose 'in' path requires the literal noun 'status' after the
        marker — so these must all stay empty while the two
        task-2885-repro tests above keep passing.

        Distinct from test_open_class_gap_without_status_noun_excluded,
        which is blocked by the open-class verb 'landed' and never
        exercises the connective path at all.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    # ----------------------------------------------------------------- #
    # task 3079 — genitive/possessive status form
    # ----------------------------------------------------------------- #

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            # the finding's verbatim stale edge fact
            (
                "Task 2623's status is pending as of 2026-07-14T20:31:00Z, "
                'linked to memory f20fbf15-41b7-4781-97a6-334f4e1d066a.',
                {2623},
            ),
            # curly apostrophe (U+2019) — what most prose writers/LLMs emit
            ('Task 2623’s status is pending as of 2026-07-14.', {2623}),
            ("Task 5's status is blocked.", {5}),
            ("Task 5's status is in-progress.", {5}),
        ],
    )
    def test_genitive_status_form_extracts_id(self, fact, expected):
        """"Task N's status is <marker>" -> {N}. (task 3079)

        REPRO. This is the canonical shape Graphiti writes for a per-task
        status snapshot, and every pre-3079 path missed it while the gate
        (a bare '\\bpending\\b' search) fired — the exact "matched the gate
        yet went undetected" symptom the finding reports:

        - INDIVIDUAL_SNAPSHOT_RE is TASK_REF_RE + '\\s*' + copula, and the
          '\\s*' cannot cross the intervening "'s status ".
        - SNAPSHOT_STATUS_PHRASE_RE wants 'in <marker> status' — marker
          BEFORE the noun; here the order is inverted ('status is <marker>').
        - LIST_INTRODUCER_RE wants '<marker> tasks[:\\[]'.

        So the extractor returned set() for a fact that plainly asserts a
        non-terminal status, and the edge was never even a candidate. The
        control shape 'The status of task 2623 is pending' already worked,
        which is what makes this a lexical gap rather than a traversal one.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    @pytest.mark.parametrize(
        'fact',
        [
            # negation / past-exit — refused for free by the closed-class
            # _ADVERB_ALT, exactly as the individual form refuses it (the
            # asymmetry docs/stale-status-snapshot-sweep.md §"Genitive form"
            # documents: no _GAP_EXCLUDED_ALT equivalent is needed on a
            # closed-class-connective path).
            "Task 5's status is no longer pending.",
            "Task 5's status is not blocked.",
            "Task 5's status was previously pending.",
            # terminal status — the gate short-circuits before extraction
            "Task 5's status is done.",
            # head-noun hazard: a status REPORT that is pending review, not a
            # pending task. Requiring 'status' + copula ADJACENCY excludes it
            # outright, so this path needs no trailing noun-phrase lookahead
            # of the kind SNAPSHOT_STATUS_PHRASE_RE carries.
            "Task 5's status report is pending review.",
            "Task 5's status update is pending.",
            # HYPHENATED negation / past-exit, reaching the marker through
            # the compound-modifier prefix rather than through the copula
            # slot. The closed-class _ADVERB_ALT does not see these — the
            # inverting token is fused to the marker, not a separate word —
            # so _COMPOUND_PREFIX carries its own exclusion lookahead.
            # (amendment, reviewer_comprehensive correctness-precision
            # finding, task 3079)
            "Task 5's status is un-blocked.",
            "Task 5's status is non-blocked.",
            "Task 5's status is previously-blocked.",
        ],
    )
    def test_genitive_status_form_precision_guards(self, fact):
        """Shapes the genitive path must NOT extract. (task 3079)

        These pin the new path against the over-selection direction the
        module docstring forbids: each fact below is either permanently
        true (a past-exit/negated assertion), already terminal, or binds
        the marker to a following head noun rather than to the task.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    # ----------------------------------------------------------------- #
    # task 3079 — plural multi-task enumeration
    # ----------------------------------------------------------------- #

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            ('Tasks 1020, 1030 and 1031 are pending.', {1020, 1030, 1031}),
            # Oxford comma + closed-class adverb
            (
                'Tasks 1020, 1030, and 1031 are still in progress.',
                {1020, 1030, 1031},
            ),
            ('Tasks 1020 and 1030 are blocked.', {1020, 1030}),
            # repeated reference token in the enumeration tail
            ('Tasks 1020, task 1030 and task 1031 are pending.', {1020, 1030, 1031}),
        ],
    )
    def test_plural_enumeration_extracts_every_id(self, fact, expected):
        """"Tasks A, B and C are <marker>" -> {A, B, C}. (task 3079)

        REPRO, and the precise answer to the finding's second question: the
        pre-3079 extractor got NONE of these ids, not merely the first.
        TASK_REF_RE anchors '\\btask\\b', which does not match the plural
        'Tasks' at all; and even if it did, the enumeration tail (', 1030,
        and 1031') carries no reference token of its own, so the trailing
        ids were unreachable by every path.

        The Oxford comma case is load-bearing: a separator alternation
        that handles ',' or 'and' but not ', and' silently misses the
        finding's verbatim fact shape.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    @pytest.mark.parametrize('fact', PRECISION_GUARD_SHAPES)
    def test_plural_enumeration_precision_guards(self, fact):
        """Shapes the plural path must NOT extract. (task 3079)

        Same anchoring discipline the rest of the module enforces: the
        marker must sit immediately after the enumeration and its copula,
        the copula is mandatory so a plural subject followed by a transitive
        verb never reads as a status assertion, and the enumeration must be
        the copula's SUBJECT rather than a preposition's complement.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    @pytest.mark.parametrize('fact', GUARD_REJECTED_SUPPRESSION_SHAPES)
    def test_guard_rejected_enumeration_suppresses_ids_on_every_path(self, fact):
        """Rejecting the plural match must not leave its tail extractable.

        (amendment, reviewer_comprehensive correctness-precision finding,
        task 3079)

        The subjecthood guard used to ``continue`` past the plural match
        and nothing more, so an enumeration that REPEATS the reference
        token still handed its trailing ids to the unguarded individual
        path: 'Reviews for tasks 1020, task 1030 and task 1031 are
        pending' suppressed the plural match and then matched 'task 1031
        are pending' on its own, yielding {1031}.

        That leak is only reachable because this task newly blessed the
        repeated-reference-token enumeration as a supported POSITIVE shape
        (see the '{1020, 1030, 1031}' case below); its
        prepositional-complement counterpart is therefore a live shape the
        guard is documented to close. Every fact here is a permanently-true
        meta assertion about reviews/statuses/notes, so extracting any id
        retires a still-true edge as soon as that id goes terminal — the
        unrecoverable over-selection direction.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            ('Tasks 1020, task 1030 and task 1031 are pending.', {1020, 1030, 1031}),
            # The suppression is scoped to the REJECTED span, not to the
            # whole fact: a genuine snapshot in a later clause survives.
            ('Reviews for tasks 1020 and task 1030 are pending. Task 55 is blocked.', {55}),
        ],
    )
    def test_span_suppression_is_scoped_to_the_rejected_enumeration(self, fact, expected):
        """The suppression must not become a blanket whole-fact veto.

        Pins the other edge of the guard above: the repeated-reference-token
        enumeration is a supported positive when it IS the copula's subject,
        and a rejected enumeration only silences ids inside its own span.
        (task 3079)
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    #: A plural-head fact per preposition in ``_ENUM_PREP_WORDS``. Plural
    #: agreement is what refuses the far more natural singular-copula
    #: phrasing of most of these ('Work on tasks ... IS blocked'), so only a
    #: plural head puts the preposition guard itself on the hook. See the
    #: test below for why that distinction decides whether these cases test
    #: anything at all.
    #:
    #: This maps the SOURCE tuple's entries rather than restating the
    #: vocabulary: the test parametrizes over ``_ENUM_PREP_WORDS`` directly,
    #: so a preposition added to the guard without a case here fails as its
    #: own parametrized case instead of shipping unexercised. (Earlier this
    #: sync was enforced by scraping the then-lookbehind guard constant's
    #: ``.pattern`` with a second regex — an introspection meta-test on a
    #: private constant's spelling, deleted in favour of inverting the
    #: dependency. amendment, reviewer_comprehensive test-quality finding,
    #: task 3079)
    _PREPOSITION_GUARD_FACTS = {
        'of': 'Statuses of tasks 1020 and 1030 are blocked.',
        'in': 'Delays in tasks 1020 and 1030 are blocked.',
        'on': 'Notes on tasks 1020 and 1030 are pending.',
        'to': 'Updates to tasks 1020 and 1030 are pending.',
        're': 'Comments re tasks 1020 and 1030 are pending.',
        'for': 'Reviews for tasks 1020 and 1030 are pending.',
        'with': 'Problems with tasks 1020 and 1030 are blocked.',
        'among': 'Conflicts among tasks 1020 and 1030 are blocked.',
        'about': 'Notes about tasks 1020 and 1030 are pending.',
        'across': 'Regressions across tasks 1020 and 1030 are pending.',
        'against': 'Complaints against tasks 1020 and 1030 are pending.',
        'between': 'Dependencies between tasks 1020 and 1030 are blocked.',
        'regarding': 'Notes regarding tasks 1020 and 1030 are pending.',
        'from': 'Dependencies from tasks 1020 and 1030 are blocked.',
        'by': 'Reviews by tasks 1020 and 1030 are pending.',
        'under': 'Sub-issues under the tasks 1020 and 1030 are blocked.',
        'within': 'Failures within tasks 1020 and 1030 are blocked.',
        'around': 'Discussions around tasks 1020 and 1030 are pending.',
        'during': 'Blockers during tasks 1020 and 1030 are pending.',
        'through': 'Regressions through tasks 1020 and 1030 are pending.',
        'via': 'Escalations via tasks 1020 and 1030 are pending.',
        'concerning': 'Notes concerning tasks 1020 and 1030 are pending.',
    }

    @pytest.mark.parametrize('preposition', _ENUM_PREP_WORDS)
    def test_every_guarded_preposition_is_exercised_against_a_plural_head(
        self, preposition
    ):
        """Each entry in ``_ENUM_PREP_WORDS`` is load-bearing. (task 3079)

        The guard list could previously have SILENTLY LOST entries: most of
        its prepositions appeared only in singular-copula facts ('Work on
        tasks ... is blocked', 'The dependency between tasks ... is
        blocked'), and those are already refused by ``_PLURAL_COPULA_ALT``
        agreement — deleting 'on' or 'between' from the list would not have
        failed a single test. Only a PLURAL head noun ('Notes', 'Delays',
        'Dependencies') with a plural copula reaches the preposition check,
        so these are the cases that actually pin it.

        (amendment, reviewer_comprehensive test-coverage finding, task 3079)
        """
        fact = self._PREPOSITION_GUARD_FACTS.get(preposition)
        assert fact is not None, (
            f'preposition {preposition!r} was added to _ENUM_PREP_WORDS '
            f'without a plural-head case in _PREPOSITION_GUARD_FACTS'
        )
        assert extract_snapshot_edge_task_ids(fact) == set(), (
            f'preposition {preposition!r} is in _ENUM_PREP_WORDS but its '
            f'complement still over-selects'
        )

    @pytest.mark.parametrize(('fact', 'expected'), SUBJECT_POSITIVE_SHAPES)
    def test_plural_enumeration_subject_positives_survive_precision_guards(
        self, fact, expected
    ):
        """Subject-position enumerations still extract. (task 3079)

        These pass before the prepositional-complement fix and must keep
        passing after it — they are what stops the guard from over-reaching
        into a blanket disabling of the plural path. Together with the
        precision guards above they pin the fix from both directions.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    @pytest.mark.parametrize('fact', ADVERBIAL_PREAMBLE_SHAPES)
    def test_adverbial_preamble_is_a_documented_under_selection(self, fact):
        """Cost of the clause-scoped guard, pinned rather than left implicit.

        (amendment, reviewer_comprehensive correctness-recall finding, task
        3079)

        The guard rejects when a listed preposition appears ANYWHERE between
        the last sentence-final break and the enumeration, and a comma is
        deliberately not a break (it does not end prepositional government).
        Together those mean an ordinary sentence-initial ADVERBIAL PREAMBLE —
        a date stamp, a location or a cycle scope, all common in this repo's
        memory corpus — shares the clause with its own preposition and
        suppresses a genuine subject-position snapshot behind it.

        This is the fail-safe under-selection direction, so it is documented
        behaviour rather than a defect: the edge is simply not retired this
        cycle and the next sweep (or Stage 2) sees it again. It is pinned
        here so the recall cost is VISIBLE and intentional — if the guard
        widens further, these cases are where it shows up, and if the real
        edge corpus ever makes the cost matter, this is the test that has to
        be renegotiated first.

        The corpus has now been measured and says it does NOT (task 3949).
        Two live facts reach this guard, and BOTH are correctly-rejected
        prepositional complements; NEITHER triages as an adverbial preamble,
        which is the only rejection class that costs recall. So the measured
        recall cost of the guard is zero edges — which is a WEAKER claim
        than 'nothing matches the regex at all', and is the one the evidence
        actually supports. Both candidate tightenings were re-validated
        against the full precision parametrization: (b) re-opens a pinned
        over-selection, and (a) re-opens none but does not even recover the
        'As of <date>, ...' case above. VERDICT: not tightened — see the
        measurement block at ``_ENUM_PREP_WORDS``, and
        plans/plural-enum-guard-recall-report.md for the per-graph figures.
        Those figures are deliberately NOT restated here: an earlier draft of
        this docstring pinned an edge count and a graph count inline and both
        were stale within days. Re-run
        ``scripts/measure_plural_enum_guard_recall.py`` before renegotiating
        this test — every count in that report is a point-in-time fact about
        a growing corpus.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    def test_token_internal_break_narrowing_costs_only_under_selection(self):
        """Cost of the token-internal-break narrowing, pinned from both sides.

        (tasks 4149, 4851)

        Disqualifying a token-internal break introduces exactly one new
        residual: a genuine SENTENCE break ('.', ';', '!' or '?') with no
        following space AND a following alphanumeric, sitting behind a
        listed preposition. Such a break now reads as token-internal
        (flanked by alphanumerics on both sides), so the backward scan does
        not stop there and the clause extends back over the preposition,
        suppressing a genuine
        subject-position enumeration. This is the fail-safe under-selection
        direction — the edge is simply not retired this cycle and the next
        sweep sees it again — mirroring
        test_adverbial_preamble_is_a_documented_under_selection's framing
        for the (unrelated) adverbial-preamble residual.

        Pinned alongside the far more common no-space shape that is NOT
        affected, so the residual's true, narrow boundary stays legible:
        'prefix' is cut at '\\btasks\\b', so when the enumeration's own
        marker word immediately follows the period with no space, that
        period is the prefix's LAST character and has no right flank at
        all — it is still a break regardless of the narrowing.
        """
        # ACCEPTED residual: 'done.Then' — the period is flanked by 'e' and
        # 'T', both alnum, so it reads as token-internal and the scan continues
        # back over 'for', suppressing the match.
        assert extract_snapshot_edge_task_ids(
            'Reviews for the branch are done.Then tasks 1020 and 1030 are pending.'
        ) == set()
        # ACCEPTED residual, the same shape for '?' (task 4851).
        assert extract_snapshot_edge_task_ids(
            'Reviews for the branch are done?Then tasks 1020 and 1030 are pending.'
        ) == set()
        # UNAFFECTED: 'done.Tasks' — the period is the prefix's last
        # character (prefix ends right before '\\btasks\\b'), so it has no
        # right flank and stays a break regardless of the narrowing.
        assert extract_snapshot_edge_task_ids(
            'Reviews for the branch are done.Tasks 1020 and 1030 are pending.'
        ) == {1020, 1030}

    def test_plural_enumeration_needs_two_or_more_ids(self):
        """A plural head over a single id is not an enumeration. (task 3079)

        Guards the plural path from degenerating into a second, weaker
        individual form: 'Tasks 1020 are pending' is malformed English and
        a single stray digit after a plural head is more likely a count
        than an id. Requiring 2+ ids keeps the path's subject unambiguous.
        The well-formed singular shape still routes through
        INDIVIDUAL_SNAPSHOT_RE unchanged.

        This also pins the separator against a defect the first draft of
        _ENUM_IDS_ALT had: with every separator element optional, the
        repeated group could match between two adjacent digits of a SINGLE
        number, so '1020' parsed as '102' + '0' and this fact satisfied a
        rule meant to require two ids. The separator's leading comma-or-
        whitespace is now mandatory.
        """
        assert extract_snapshot_edge_task_ids('Tasks 1020 are pending.') == set()
        assert extract_snapshot_edge_task_ids('Task 1020 is pending.') == {1020}

    def test_plural_enumeration_collects_digits_only_from_its_own_span(self):
        """An embedded count phrase never contributes a spurious id. (task 3079)

        The module's standing invariant is that a bare '\\d+' may only
        contribute an id from inside an already-detected, marker-anchored
        segment — the property that keeps dates, commit hashes and ports in
        ordinary prose from becoming task ids. The plural path preserves it
        by collecting digits from its own 'ids' capture group ONLY, never
        from the whole fact.

        Here '3 tasks' sits in a second clause outside the enumeration, so
        3 must not appear even though the fact as a whole is packed with
        status markers.
        """
        ids = extract_snapshot_edge_task_ids(
            'Tasks 1020 and 1030 are pending; 3 tasks remain in progress.'
        )
        assert ids == {1020, 1030}
        assert 3 not in ids

    def test_plural_enumeration_does_not_reach_across_an_intervening_count(self):
        """A count phrase breaking the enumeration kills the match. (task 3079)

        'Blocked tasks 1020 and 1030 plus 3 pending others ...' — the
        enumeration cannot absorb 'plus 3', and the copula never follows
        the ids, so the path yields nothing at all rather than leaking a
        bare 3. Under-selection is the fail-safe direction this module
        prefers throughout.
        """
        assert extract_snapshot_edge_task_ids(
            'Blocked tasks 1020 and 1030 plus 3 pending others are pending.'
        ) == set()

    # ----------------------------------------------------------------- #
    # task 3079 — 'stalled' as a recognized non-terminal marker
    # ----------------------------------------------------------------- #

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            # the finding's verbatim stale edge fact — a hyphenated compound
            (
                'Tasks 1020, 1030, and 1031 are pairwise-stalled as of 2026-04-29.',
                {1020, 1030, 1031},
            ),
            (
                'Tasks 1020, 1030, and 1031 are pairwise-stalled and remain pending.',
                {1020, 1030, 1031},
            ),
            # every other path must reach the new marker too — the module's
            # standing rule is that a marker addition is checked against ALL
            # paths, not just the one that motivated it
            ('Task 5 is stalled.', {5}),
            ('Task 5 is pairwise-stalled.', {5}),
            ("Task 5's status is stalled.", {5}),
            ('Task 5 is in a stalled status.', {5}),
            ('Stalled tasks: 1020, 1030', {1020, 1030}),
        ],
    )
    def test_stalled_is_a_recognized_marker(self, fact, expected):
        """'stalled' asserts a non-terminal status. (task 3079)

        REPRO, and the reason the 1020/1030/1031 cluster was unreachable
        even in principle: SNAPSHOT_STATUS_RE — the whole-fact gate — did
        not know the word, so extraction short-circuited to set() before
        any reference parsing ran. A perfect plural-enumeration parser
        alone would still not have selected this edge.

        The cluster asserts a stalled (i.e. non-terminal, still-live)
        condition on tasks that are now done, which is squarely in this
        sweep's remit.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    @pytest.mark.parametrize(
        'fact',
        [
            # THE reason 'stalled' belongs in _TRANSITIVE_MARKER_ALT rather
            # than _ADJECTIVE_MARKER_ALT: like 'blocked' it is also a common
            # transitive verb, and these are permanently-true historical
            # facts. The optional-copula adjective arm would match them and
            # retire them the moment task 5 went done.
            'Task 5 stalled the merge queue.',
            'Task 5 stalled the release for a week.',
            'Tasks 1020 and 1030 stalled the merge queue.',
            # negation / past-exit
            'Task 5 is no longer stalled.',
            "Task 5's status is no longer stalled.",
        ],
    )
    def test_stalled_precision_guards(self, fact):
        """Shapes the new marker must NOT extract. (task 3079)

        Mirrors the task-3042 transitive-verb hardening exactly: 'stalled'
        is admitted only in copula/article form, so the bare verb reading
        stays out of every path.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    def test_stalled_count_quantity_is_stripped_from_aggregate_segment(self):
        """'3 stalled' must not leak a bare id 3. (task 3079)

        The non-obvious half of the module's standing marker-addition rule.
        Widening _STATUS_MARKER_ALT feeds four consumers; three are safe by
        construction (the gate merely admits a fact; the phrase form
        requires the literal noun 'status'; the list introducer requires
        adjacency to the list noun). COUNT_QUANTITY_RE is the one that is
        NOT — it strips 'N <count-noun>' spans from an aggregate segment
        before bare-digit collection, so a marker missing from ITS
        alternation turns a count phrase into a spurious task id.
        """
        ids = extract_snapshot_edge_task_ids('Stalled tasks: 1020, 1030, and 3 stalled others')
        assert ids == {1020, 1030}
        assert 3 not in ids

    def test_compound_prefix_is_a_single_hyphenated_token(self):
        """The compound prefix cannot cross whitespace. (task 3079)

        'pairwise-stalled' anchors to its copula via a bounded optional
        '\\w+-' prefix. Deliberately ONE hyphenated token: were it allowed
        to span whitespace it would additionally admit the multi-word
        readings the mandatory-copula arms exist to exclude — here, a
        merge that is blocked rather than the tasks.

        NOTE this pins the gap's WIDTH only. Its LEXICAL CLASS is a
        separate property with its own test below — the two were once
        conflated, which is how the un-/non-/previously- class slipped in.
        """
        assert extract_snapshot_edge_task_ids(
            'Tasks 1020 and 1030 are awaiting a blocked merge.'
        ) == set()
        assert extract_snapshot_edge_task_ids('Task 5 is awaiting stalled work.') == set()

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 5 is un-blocked.',
            'Task 5 is non-blocked.',
            'Task 5 is not-pending.',
            'Task 5 is never-blocked.',
            'Task 5 is previously-blocked.',
            'Task 5 is formerly-pending.',
            'Task 5 is once-blocked.',
            'Task 5 is briefly-stalled.',
            # The privative 'in-', reached through the same door. Guarded by
            # its own narrower lookahead because 'in-progress' is itself a
            # marker, so a bare 'in' entry in the main alternation would
            # refuse the positive along with the negation. (task 3079)
            'Task 5 is in-active.',
            "Task 5's status is in-active.",
            'Tasks 1020 and 1030 are in-active.',
        ],
    )
    def test_compound_prefix_refuses_inverting_modifiers(self, fact):
        """A hyphenated negation / past-exit prefix must not reach the marker.

        (amendment, reviewer_comprehensive correctness-precision finding,
        task 3079)

        The bare-marker forms of these ARE already refused, for free, by
        the closed-class ``_ADVERB_ALT`` — 'Task 5 is no longer blocked'
        and 'Task 5 is not blocked' never matched, because those tokens
        simply are not in the alternation. The compound prefix reopened
        the same class through a different door: bounding it to a single
        non-whitespace-crossing token bounds its WIDTH, not its lexical
        class, and 'un-'/'non-'/'previously-' are each one '\\w+-' token.

        Each fact here is a permanently-true HISTORICAL or negated
        assertion. Extracting an id from one means the sweep retires a
        still-true edge as soon as that id goes done/cancelled — the
        over-selection direction the module docstring forbids, and
        unrecoverable, since an invalidated Graphiti edge is not
        re-validated on the next cycle.
        """
        assert extract_snapshot_edge_task_ids(fact) == set()

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            ('Tasks 1020, 1030, and 1031 are pairwise-stalled.', {1020, 1030, 1031}),
            ('Task 5 is merge-blocked.', {5}),
            ('Task 5 is self-blocked.', {5}),
            ('Tasks 7 and 8 are auto-stalled.', {7, 8}),
            ("Task 5's status is merge-blocked.", {5}),
            # The 'in-' lookahead is narrowed to spare 'in-progress', which
            # is a MARKER rather than a prefixed one. Both the hyphenated
            # and the spaced spellings must survive it, on every anchored
            # path. (task 3079)
            ('Task 5 is in-progress.', {5}),
            ('Task 5 is in progress.', {5}),
            ("Task 5's status is in-progress.", {5}),
            ('Tasks 1020 and 1030 are in-progress.', {1020, 1030}),
        ],
    )
    def test_compound_prefix_still_admits_ordinary_modifiers(self, fact, expected):
        """Non-inverting compound modifiers must keep matching. (task 3079)

        The exclusion above is a closed SUBTRACTION from an open-class
        prefix, not a whitelist: the observed modifier vocabulary
        ('pairwise-', 'merge-', 'self-', 'auto-') is not closed, and an
        unlisted-but-innocent modifier must not silently drop a genuine
        snapshot. 'pairwise-stalled' is the finding's verbatim fact shape,
        so this is the case the prefix exists for in the first place.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected

    # ----------------------------------------------------------------- #
    # task 3403 — slash-form task references
    # ----------------------------------------------------------------- #

    def test_slash_form_reference_is_extracted(self):
        """'Task/5 is pending' -> {5}.

        This module hand-copies nothing: INDIVIDUAL_SNAPSHOT_RE and
        SNAPSHOT_STATUS_PHRASE_RE are both built by splicing
        TASK_REF_RE.pattern, precisely so the shared reference grammar
        stays in sync. Task 3403 widened that grammar to admit the
        orchestrator's own 'task/<id>' branch-name form, and this test
        exercises the propagation through the module's public entry point
        rather than re-testing the composites. MEASURED set() before the
        widening.

        Direction of risk, stated plainly: this path ends in
        memory_service.update_edge(invalid_at=...), so a slash-form
        reference now makes an edge retirable once its task goes terminal.
        That is correct RECALL, not over-selection — a slash-form reference
        IS a task reference, and a snapshot asserting 'task/5 is pending'
        goes just as stale as one asserting 'task 5 is pending'. The
        anchoring that bounds over-selection ('\\btask\\b', and the trailing
        '\\b' on the digits) is untouched.
        """
        assert extract_snapshot_edge_task_ids('Task/5 is pending') == {5}

    def test_slash_form_intervening_reference_still_re_subjects_the_gap(self):
        """'Task 5 blocks task/9 in blocked status' -> {9}, not {5}.

        The task-3042 fix (_GAP_NO_TASK_REF, wired into
        SNAPSHOT_STATUS_PHRASE_RE's open-class gap) must survive the
        widening: the gap is built from TASK_REF_RE too, so once the
        grammar admits the slash form the gap must refuse to span a
        slash-form reference exactly as it already refuses to span a spaced
        one. Otherwise the widening would reintroduce 3042's
        both-error-directions-at-once bug through the back door — the 'in
        blocked status' phrase binding to the outer task 5 (a
        permanently-true historical fact this sweep would then retire) AND
        consuming task/9, so the edge that genuinely IS a blocked-status
        snapshot stays invisible to the sweep.

        MEASURED set() before the widening — the slash reference was simply
        not a reference then, so neither id was reachable.
        """
        assert extract_snapshot_edge_task_ids(
            'Task 5 blocks task/9 in blocked status'
        ) == {9}

    @pytest.mark.parametrize(
        ('fact', 'expected'),
        [
            # The reachable over-selection: 'task/94' is a PATH SEGMENT naming a
            # worktree, and the sentence asserts something about the WORKTREE,
            # not about task 94's status. Nothing in the grammar can tell the
            # two apart — a path segment and a reference are the same six
            # characters — so the id is extracted. ACCEPTED.
            ('The worktree .worktrees/task/94 is pending cleanup', {94}),
            # One more path component and the extraction stops: 'plan.json'
            # sits between the reference and 'is pending', and the open-class
            # gap in SNAPSHOT_STATUS_PHRASE_RE does not span it.
            ('.worktrees/task/94/plan.json is pending review', set()),
            # The PLURAL segment was never a reference at all: the anchor is
            # '\btask\b', so 'tasks/94' does not match. (This is the same
            # property audit_duplicate_memories' untasked-snapshot accounting
            # depends on.)
            ('.worktrees/tasks/94/plan.json is pending review', set()),
            # A trailing word character defeats the digits' '\b': '94_notes'
            # is one token, not a reference followed by a suffix.
            ('src/task/94_notes.md is pending review', set()),
        ],
    )
    def test_path_embedded_slash_reference_is_documented_behaviour(self, fact, expected):
        """Slash refs that are PATH segments rather than sentence subjects:
        characterized, not silently inherited. (task 3403)

        The slash form was admitted to TASK_REF_RE so 'task/94' — the
        orchestrator's own branch-name convention — stops being invisible.
        The cost is that the SAME spelling occurs inside paths, and the
        sweep's path (stale_status_snapshot_edge_sweep.py) ends in
        memory_service.update_edge(invalid_at=...): the first case below makes
        an edge asserting something about a WORKTREE retirable as soon as task
        94 goes terminal. That is over-selection, the direction
        status_snapshot_extraction.py's docstring calls unrecoverable, and it
        is newly reachable — MEASURED set() before the widening.

        It is accepted rather than fixed. Excluding it would mean refusing a
        reference preceded by a path character, which would also refuse the
        genuine 'Merge task/3698 into main' shape that motivated the widening;
        the grammar has no way to distinguish them, and the anchoring that
        actually bounds over-selection ('\\btask\\b' and the trailing
        '\\b' on the digits) is what keeps the other three cases at set().
        Recording the boundary here means a later regression in any of the
        three safe shapes shows up as a test failure rather than as an
        unexplained widening of what this sweep retires.
        """
        assert extract_snapshot_edge_task_ids(fact) == expected


# --------------------------------------------------------------------------- #
# extract_blocked_assertion_task_ids — blocked-scoped extraction (task 3037)
# --------------------------------------------------------------------------- #

# The whole case table for the blocked family, shared by the per-shape tests
# and by the subset-parity test below so the two can never disagree about
# what the expected values are. Each expectation was measured against the
# UNION extractor at HEAD=c54b8cfcc5 before being written down, so none of
# them is a guess (task 3037).
_BLOCKED_EXTRACTION_CASES: tuple[tuple[str, set[int]], ...] = (
    # --- the five extraction paths, each asserting 'blocked' ---
    ('Task 2848 remains blocked as of 2026-07-22', {2848}),
    ('Task 5 is currently blocked', {5}),
    ('Task 5 is blocked on task 2862', {5}),
    ("Task 5's status is blocked", {5}),
    ('Task 5 is in a blocked status.', {5}),
    ('Tasks 1020 and 1030 are blocked', {1020, 1030}),
    ('Blocked tasks: 142, 148', {142, 148}),
    # --- the precision guards the union family already buys ---
    ('Task 5 blocked the merge queue', set()),
    ('Task 5 is no longer blocked', set()),
    # --- the blocked-family-only aggregate subject guard (task 3037) ---
    ('Task 5 blocked tasks: 142, 148', set()),
    ('Task 5 is waiting on blocked tasks: 142, 148', set()),
    ('Task 5 is blocked. Blocked tasks: 142, 148', {5, 142, 148}),
    # --- markers that are NOT blocked assertions ---
    ('Task 5 is pending', set()),
    ('The active pending tasks are [7, 9]', set()),
    ('Task 7 is stalled', set()),
    # --- rejections the blocked family must INHERIT from the union family ---
    # (amendment, reviewer_comprehensive correctness-over-selection finding,
    # task 3037).  Every one of these is a prepositional-complement
    # enumeration: the DEPENDENCIES / REVIEWS / MERGES / STATUSES are what the
    # copula predicates, so no enumerated id is its subject and no id is
    # asserted blocked.  The union family already refuses all of them
    # (measured -> set() at HEAD=fe8e74c12c); the expectations below are
    # set() because the blocked family must refuse them for the SAME reason,
    # not because 'blocked' fails to appear.
    #
    # Do NOT 'fix' these expectations to {1030} / {1031}.  That is what the
    # branch tip actually returned, and it was the bug: the blocked family's
    # own plural_enum needs the literal 'blocked' marker, so it never built
    # the enumeration span the union family rejected, and the trailing
    # 'in blocked status' let its individual arm claim the repeated 'task N'
    # reference inside a span already established to be a preposition's
    # complement.  See _extract_ids's rejected_spans threading.
    ('Dependencies for tasks 1020 and task 1030 are pending in blocked status', set()),
    ('Reviews of tasks 1020 and task 1030 are pending in blocked status', set()),
    ('The merges of tasks 1020 and task 1030 are pending in blocked status', set()),
    ('Statuses of tasks 1020 and task 1030 are pending in blocked status', set()),
    (
        'Dependencies for tasks 1020, task 1030 and task 1031 are pending in blocked status',
        set(),
    ),
)

# The subset-property corpus, deliberately DISJOINT from
# _BLOCKED_EXTRACTION_CASES above and GENERATED from axes rather than
# hand-picked instances (amendment, reviewer_comprehensive
# correctness-over-selection finding, task 3037).
#
# Why generated. The subset property USED to parametrize over
# _BLOCKED_EXTRACTION_CASES — the same hand-picked table the positive tests
# are drawn from — so it could only ever re-confirm cases someone had already
# thought of, and it passed for the whole life of the over-selection bug. A
# property that ranges over the same inputs as the examples is not a property
# test; it is the examples again.
#
# The four axes are chosen to be exactly where the defect lives:
#   heads   — one prepositional-complement subject per distinct
#             _ENUM_PREP_WORDS preposition class, since the union family's
#             rejection is what the blocked family must inherit;
#   enums   — with and without the REPEATED 'task N' reference token, which
#             is what let an anchored arm reach into a rejected span, and
#             with and without a leading determiner, which defeats a
#             fixed-width lookbehind;
#   markers — every union marker class including 'blocked' itself, so the
#             cross-family cases and the same-family control are both
#             covered;
#   tails   — the trailing fragments that give the blocked family a literal
#             'blocked' token the union enumeration span does not contain,
#             which is the precondition for the two families disagreeing.
#
# MEASURED: 800 facts, 64 subset violations at HEAD=fe8e74c12c (RED), 0 once
# _extract_ids threads the union family's rejected spans into the blocked
# family.
_SUBSET_PROPERTY_CORPUS: tuple[str, ...] = tuple(
    f'{head} {enum} are {marker} {tail}'
    for head in (
        'Dependencies for', 'Reviews of', 'The merges of', 'Statuses of',
        'Notes on', 'Comments about', 'Work by', 'Blockers between',
    )
    for enum in (
        'tasks 1020 and task 1030',
        'tasks 1020, task 1030 and task 1031',
        'the tasks 1020 and 1030',
        'tasks 1020 and 1030',
    )
    for marker in ('pending', 'active', 'in progress', 'blocked', 'stalled')
    for tail in (
        '', 'in blocked status', '; task 1040 is blocked', 'and blocked',
        'blocked tasks: 55, 66',
    )
)


class TestExtractBlockedAssertionTaskIds:
    """extract_blocked_assertion_task_ids(fact) returns the subset of ids the
    fact asserts as BLOCKED specifically — not the union over every status
    marker (task 3037).

    Why a blocked-SCOPED extractor is needed at all, rather than reusing
    extract_snapshot_edge_task_ids: the union extractor cannot say WHICH
    marker contributed an id, so 'Task 5 is pending' and 'Task 5 is
    currently blocked' are indistinguishable at {5}. The new selection rule
    retires an edge whose asserted status is contradicted by any
    positively-known non-'blocked' status; applied to union ids that rule
    would retire an 'active/pending' snapshot the moment its task moved to
    review — a behavioural widening far outside this task, and in the
    over-selection direction the module docstring forbids.

    The blocked family is generated from the same pattern builder as the
    union family (see _build_snapshot_patterns), so every hardening tasks
    2613/3042/3079/3403/4149 bought — transitive-verb guard, negation and
    past-exit guard, intervening-task-reference guard, prepositional-
    complement subjecthood guard, possessive quantifiers, token-internal-
    break narrowing — applies here for free and cannot drift.
    """

    # --- the five extraction paths ---

    def test_individual_form_remains_blocked(self):
        """The live repro fact: 'Task 2848 remains blocked as of 2026-07-22' -> {2848}."""
        assert extract_blocked_assertion_task_ids(
            'Task 2848 remains blocked as of 2026-07-22'
        ) == {2848}

    def test_individual_form_adverb_currently_blocked(self):
        """'Task 5 is currently blocked' -> {5} (closed-class adverb slot)."""
        assert extract_blocked_assertion_task_ids('Task 5 is currently blocked') == {5}

    def test_individual_form_never_attributes_the_blocker(self):
        """'Task 5 is blocked on task 2862' -> {5}, never the BLOCKING task 2862.

        Only task 5 is asserted blocked; 2862 is what it is blocked ON. A
        rule that retired this edge on 2862's status would be reading the
        fact backwards.
        """
        assert extract_blocked_assertion_task_ids('Task 5 is blocked on task 2862') == {5}

    def test_genitive_form(self):
        """"Task 5's status is blocked" -> {5} (GENITIVE_STATUS_RE path, task 3079)."""
        assert extract_blocked_assertion_task_ids("Task 5's status is blocked") == {5}

    def test_status_phrase_form(self):
        """'Task 5 is in a blocked status.' -> {5} (SNAPSHOT_STATUS_PHRASE_RE path)."""
        assert extract_blocked_assertion_task_ids('Task 5 is in a blocked status.') == {5}

    def test_plural_enumeration_form(self):
        """'Tasks 1020 and 1030 are blocked' -> {1020, 1030} (task 3079 path)."""
        assert extract_blocked_assertion_task_ids('Tasks 1020 and 1030 are blocked') == {
            1020,
            1030,
        }

    def test_aggregate_list_form(self):
        """'Blocked tasks: 142, 148' -> {142, 148} (LIST_INTRODUCER_RE path)."""
        assert extract_blocked_assertion_task_ids('Blocked tasks: 142, 148') == {142, 148}

    # --- precision guards inherited from the shared builder ---

    def test_transitive_verb_reading_excluded(self):
        """'Task 5 blocked the merge queue' -> set().

        A permanently-true HISTORICAL fact, not a status snapshot. The
        transitive-capable arm's mandatory copula/article refuses it — the
        same guard the union family carries (task 3042).
        """
        assert extract_blocked_assertion_task_ids('Task 5 blocked the merge queue') == set()

    def test_negated_blocked_assertion_excluded(self):
        """'Task 5 is no longer blocked' -> set().

        Negation INVERTS the assertion: this fact says the task is NOT
        blocked, so it is not a blocked assertion and must never be retired
        by the blocked rule. Refused for free by _ADVERB_ALT's closed class.
        """
        assert extract_blocked_assertion_task_ids('Task 5 is no longer blocked') == set()

    def test_transitive_verb_list_collision_excluded(self):
        """'Task 5 blocked tasks: 142, 148' -> set(), while the UNION family
        still gives {142, 148}.

        The one transitive/status-list collision LIST_INTRODUCER_RE's
        adjacency requirement could not close (task 3042 logged it as a
        deliberate ambiguity residual). Measured at HEAD before this
        amendment, the blocked family inherited it and returned {142, 148} —
        so this permanently-true historical fact would have been retired by
        the BLOCKED rule the moment 142 or 148 took ANY positively-known
        non-blocked status, and a superseding fact would have been written
        attributing the wrong subject. Under the terminal rule alone the same
        residual only fired at done/cancelled.

        The union assertion is half the test: the terminal rule's behaviour
        is deliberately UNCHANGED, so the narrowing is scoped to the rule
        whose trigger made the residual dangerous.
        (amendment, reviewer_comprehensive correctness-over-selection finding)
        """
        collision = 'Task 5 blocked tasks: 142, 148'

        assert extract_blocked_assertion_task_ids(collision) == set()
        assert extract_snapshot_edge_task_ids(collision) == {142, 148}

    def test_genuine_aggregate_after_a_task_reference_is_the_cost_of_that_guard(self):
        """'Task 5 is waiting on blocked tasks: 142, 148' -> set().

        The documented UNDER-selection this trade costs: 142 and 148 really
        are asserted blocked here, but the shape is indistinguishable from
        the transitive reading by the same clause-scoped test, so the blocked
        family drops it. Fail-safe direction — the edge is simply not retired
        by the blocked rule, and the terminal rule still reaches it (union
        assertion below), which is exactly where it stood before task 3037.
        """
        genuine = 'Task 5 is waiting on blocked tasks: 142, 148'

        assert extract_blocked_assertion_task_ids(genuine) == set()
        assert extract_snapshot_edge_task_ids(genuine) == {142, 148}

    def test_aggregate_in_a_later_clause_is_not_governed_by_the_earlier_ref(self):
        """'Task 5 is blocked. Blocked tasks: 142, 148' -> {5, 142, 148}.

        The guard is CLAUSE-scoped, not fact-scoped: a task reference in an
        earlier sentence governs nothing, so a genuine aggregate that merely
        follows one keeps its ids. Without this scoping the guard would
        suppress the aggregate path across most multi-sentence facts.
        """
        assert extract_blocked_assertion_task_ids(
            'Task 5 is blocked. Blocked tasks: 142, 148'
        ) == {5, 142, 148}

    # --- non-'blocked' markers are not blocked assertions ---

    def test_pending_marker_is_not_a_blocked_assertion(self):
        """'Task 5 is pending' -> set() even though the UNION extractor gives {5}.

        This is the whole point of the blocked-scoped set: the new selection
        rule must not reach an active/pending snapshot.
        """
        assert extract_blocked_assertion_task_ids('Task 5 is pending') == set()
        assert extract_snapshot_edge_task_ids('Task 5 is pending') == {5}

    def test_aggregate_pending_list_is_not_a_blocked_assertion(self):
        """'The active pending tasks are [7, 9]' -> set() (union gives {7, 9})."""
        assert extract_blocked_assertion_task_ids('The active pending tasks are [7, 9]') == set()
        assert extract_snapshot_edge_task_ids('The active pending tasks are [7, 9]') == {7, 9}

    def test_stalled_marker_is_deliberately_excluded(self):
        """'Task 7 is stalled' -> set() (union gives {7}).

        'stalled' is deliberately NOT in the blocked family. It is not a
        member of the closed shared.task_statuses.TaskStatus vocabulary, so
        there is no census value it could be compared against: the new rule
        keys on "positively known and != 'blocked'", and a marker with no
        corresponding status value can never be evaluated that way. Scoping
        the family to the literal 'blocked' marker keeps the rule and the
        census vocabulary in exact correspondence.
        """
        assert extract_blocked_assertion_task_ids('Task 7 is stalled') == set()
        assert extract_snapshot_edge_task_ids('Task 7 is stalled') == {7}

    # --- structural parity with the union family ---

    @pytest.mark.parametrize(('fact', 'expected'), _BLOCKED_EXTRACTION_CASES)
    def test_case_table(self, fact, expected):
        """Every case above, driven from the shared table."""
        assert extract_blocked_assertion_task_ids(fact) == expected

    @pytest.mark.parametrize('fact', _SUBSET_PROPERTY_CORPUS)
    def test_blocked_ids_are_always_a_subset_of_union_ids(self, fact):
        """``blocked_ids <= all_ids`` over a GENERATED corpus, not the example table.

        A blocked id outside the union set means the sweep can select an edge
        on an id it never put into its get_statuses census — i.e.
        cross-reference an assertion against a status it never looked up —
        and then permanently retire that edge. Over-selection is the
        direction status_snapshot_extraction.py's docstring calls
        unrecoverable, so this property is the one that must hold for inputs
        nobody enumerated.

        Driven from _SUBSET_PROPERTY_CORPUS rather than
        _BLOCKED_EXTRACTION_CASES on purpose (amendment,
        reviewer_comprehensive correctness-over-selection finding, task
        3037). The previous version parametrized over the same hand-picked
        table as the positive tests, which is exactly why it could not catch
        the prepositional-complement over-selection: the offending shapes
        were not in the table, so the 'property' only ever re-checked known
        examples. See that constant for the axes and the measured numbers.

        Both sets are read off ONE
        extract_snapshot_edge_task_ids_by_marker_class call rather than two
        public-accessor calls, so the property is checked against the single
        extraction implementation the sweep actually uses — two calls could
        agree while the one-pass path the sweep takes disagrees.
        """
        result = extract_snapshot_edge_task_ids_by_marker_class(fact)

        assert result.blocked_ids <= result.all_ids, (
            f'blocked_ids escaped all_ids for {fact!r}: '
            f'blocked_ids={result.blocked_ids!r} all_ids={result.all_ids!r}. '
            f'The extra ids {result.blocked_ids - result.all_ids!r} would be '
            f'cross-referenced against a get_statuses census that never '
            f'contained them, and the edge retired on the result.'
        )

# --------------------------------------------------------------------------- #
# extract_snapshot_edge_task_ids — catastrophic-backtracking regression
# --------------------------------------------------------------------------- #


class TestPluralEnumerationPerformance:
    """The plural enumeration separator must parse in linear, not exponential, time.

    (amendment, reviewer_comprehensive performance-redos finding, task 3079)

    ``_ENUM_SEP_ALT``'s trailing ``\\s*`` overlaps its own leading
    ``\\s*,\\s*`` / ``\\s+``, so a whitespace run between two ids can be
    apportioned between those elements in ``len(run) + 1`` distinct ways.
    Under ``_ENUM_IDS_ALT``'s ``+`` repetition that is ``(w + 1) ** n``
    parses, ALL of which get explored when the overall match ultimately
    FAILS — which is the common case, since most enumerations in real prose
    are not followed by a copula and a status marker.

    Why this is a liveness property and not a micro-optimisation:
    ``sweep_stale_status_snapshot_edges`` calls this extractor once per
    valid edge in an unguarded dict comprehension with no per-edge timeout,
    over the whole group's edge set (tens of thousands of edges; current
    figure in plans/stale-status-snapshot-sweep-report.md). One
    pathological fact stalls the entire reconciliation cycle.

    SIZING NOTE: the blowup base scales with the WIDTH of the whitespace
    run — each extra whitespace character adds another way to split it — so
    the three shapes are ~2**n, ~3**n and ~6**n and are NOT interchangeable.
    Each size below was measured pre-fix on this branch to land several
    times over the budget while staying BOUNDED, so a regression presents as
    a fast failure rather than a hang. A flat ~40-id input would instead
    cost ~2**40 and hang until the session died.
    """

    #: Wall-clock budget for a single extract call. Post-fix all three
    #: variants land at ~0.1ms, a >10,000x margin, so this is nowhere near
    #: the CI-flake boundary.
    BUDGET_SECONDS = 1.0

    @pytest.mark.parametrize(
        ('sep', 'count'),
        [
            # measured pre-fix: ~6.1s (base ~2**n)
            (', ', 20),
            # measured pre-fix: ~4.6s (base ~3**n)
            (',  ', 13),
            # measured pre-fix: ~16.6s (base ~6**n) — a newline-indented
            # list of only NINE ids already wedges the sweep, a far lower
            # real-world trigger threshold than a flat 2**n reading implies
            (',\n    ', 9),
        ],
        ids=['comma-space', 'comma-two-spaces', 'comma-newline-indent'],
    )
    def test_long_non_matching_enumeration_parses_in_linear_time(self, sep, count):
        """A long enumeration that does NOT match must not backtrack exponentially.

        The fact is real prose this module targets: a plural head, a genuine
        comma-separated id enumeration, and then a verb phrase that is NOT a
        copula + status marker — so ``PLURAL_ENUM_SNAPSHOT_RE`` fails, which
        is exactly what forces the engine to explore every apportionment of
        every whitespace run.

        The elapsed-time assertion below is the real regression signal. The
        suite-wide ``timeout = 60`` / ``timeout_method = "thread"`` backstop
        in pyproject.toml would eventually fire, but it ends in
        ``os._exit(1)`` and takes the whole xdist worker with it — a 60x
        slower, far less legible failure than the assertion. Pre-fix these
        cases measured 4.6-16.6s, i.e. inside that backstop and invisible to
        it, so BUDGET_SECONDS is what actually catches the regression.
        """
        ids = [str(1000 + i) for i in range(count)]
        fact = f'Pending tasks {sep.join(ids)} were reviewed by the merge worker.'

        start = time.perf_counter()
        extracted = extract_snapshot_edge_task_ids(fact)
        elapsed = time.perf_counter() - start

        # Correctness is pinned alongside cost, so the budget can never be
        # satisfied by making the pattern simply stop matching valid input:
        # 'were reviewed by' is not a copula + status marker, so nothing is a
        # candidate here regardless of how fast the parse is.
        assert extracted == set()
        assert elapsed < self.BUDGET_SECONDS, (
            f'extract_snapshot_edge_task_ids took {elapsed:.3f}s for {count} ids '
            f'separated by {sep!r} (budget {self.BUDGET_SECONDS}s) — the '
            f'enumeration separator is backtracking exponentially'
        )

    def test_intra_token_dot_walk_touches_linearly_many_characters(self):
        """The intra-token-dot backward walk must not degrade to quadratic. (task 4149)

        Supersedes an earlier wall-clock version of this test
        (reviewer_comprehensive test-quality finding): a wall-clock budget
        could not actually detect the regression it was meant to catch,
        because the regression's extra cost is cheap C-level slice-memcpy
        that stays under any reasonable budget until the input is enormous,
        while the regex work already inside
        ``extract_snapshot_edge_task_ids`` dominates the wall clock at any
        size this suite can afford to run. This test instead counts
        characters TOUCHED (``CountingStr``, above) and calls
        ``_last_clause_break`` directly, observing the walk's documented
        cost property itself rather than inferring it from a clock:

          - resuming walk (real code): a FLAT ~4.67x of prefix length at
            every size measured (200 / 1,000 / 2,000 / 20,000 dots) — size
            independence IS the linearity claim.
          - non-resuming walk (``dot = prefix[:dot].rfind('.')`` in place of
            ``prefix.rfind('.', 0, dot)`` — the one realistic non-resuming
            spelling, since re-calling ``prefix.rfind('.')`` with no
            end-bound would infinite-loop rather than run slowly): 103.5x /
            503.5x / 1,003.5x at 200 / 1,000 / 2,000 dots — the ratio
            GROWING with n is the quadratic signature, and it already blows
            the bound below at every size tested.

        The bound is derived, not guessed: 3 full backward scans (';', '!',
        '?' are each absent from this input, so each scans the whole prefix
        once) + 1 telescoped '.' walk covering the prefix
        once + 2 single-character reads per dot (~0.67 dots per 3-char
        'ab.' token) ~= 4.7x prefix length. 8x leaves headroom for an
        innocuous refactor while still failing any superlinear form
        decisively.
        """
        dotted_tokens = 'ab.' * 20_000 + 'ab'
        prefix = CountingStr(dotted_tokens)

        result = _last_clause_break(prefix)

        # Correctness alongside cost: every dot is intra-token and no
        # ';!?' precedes, so the budget can't be met by the walk answering
        # wrongly but cheaply.
        assert result == -1
        assert prefix.touched <= 8 * len(dotted_tokens), (
            f'_last_clause_break touched {prefix.touched} characters for a '
            f'{len(dotted_tokens)}-char dot-dense prefix (budget '
            f'{8 * len(dotted_tokens)} = 8x length) — the intra-token-dot '
            f'walk is no longer linear'
        )

    def test_mixed_token_internal_break_walk_touches_linearly_many_characters(self):
        """The merged four-character walk must not degrade to quadratic. (task 4851)

        Every member of ``CLAUSE_BREAK_CHARS`` now gets the occurrence-level
        test, so the walk must step past token-internal occurrences of all
        four characters, interleaved. A naive spelling that re-takes the max
        of four fresh ``rfind`` calls per step rescans the prefix once per
        occurrence and is quadratic on dense input like this one.

        The bound is derived, not guessed: four telescoped per-character
        walks, each covering the prefix about once (~4x length), plus 2
        single-character reads per occurrence (one occurrence every 2
        characters, ~1x length), is about 5x. 8x leaves headroom for an
        innocuous refactor while still failing any superlinear form.
        """
        mixed_tokens = 'a;b?c!d.' * 5_000 + 'e'
        prefix = CountingStr(mixed_tokens)

        result = _last_clause_break(prefix)

        # Correctness alongside cost: every break is alnum-flanked, so no
        # occurrence ends a sentence and the walk must exhaust to -1.
        assert result == -1
        assert prefix.touched <= 8 * len(mixed_tokens), (
            f'_last_clause_break touched {prefix.touched} characters for a '
            f'{len(mixed_tokens)}-char break-dense prefix (budget '
            f'{8 * len(mixed_tokens)} = 8x length) — the token-internal-break '
            f'walk is no longer linear'
        )

    def test_intra_token_dot_prefix_still_extracts_correctly_end_to_end(self):
        """A long dotted run still round-trips to set() through the real extractor. (task 4149)

        Companion to the cost test above, kept end-to-end (through
        ``extract_snapshot_edge_task_ids``, not the bare helper) so the
        integration path — the extractor's own prefix slicing,
        ``_enumeration_is_prepositional_complement``, the guard's regex
        search — stays covered. No wall-clock assertion: without a timing
        claim to justify a large adversarial input, this is sized small
        (~2,000 dots, ~6KB) and runs in well under a millisecond regardless;
        the suite-wide 60s xdist timeout still backstops a catastrophic
        regression at any size.
        """
        dotted_tokens = 'ab.' * 2_000 + 'ab'
        fact = f'Reviews for {dotted_tokens} tasks 1020 and 1030 are pending.'

        # The dotted prefix still carries the governing preposition 'for'
        # (none of the dots end a sentence), so the enumeration is still
        # correctly read as its complement and suppressed.
        assert extract_snapshot_edge_task_ids(fact) == set()


# --------------------------------------------------------------------------- #
# extract_marker_task_ids — the per-marker reporting seam (task 4851)
# --------------------------------------------------------------------------- #

_PENDING_ALT = r'(?:pending)'
_IN_PROGRESS_ALT = r'(?:in[-\s]?progress)'
_BLOCKED_ALT = r'(?:blocked)'


class TestExtractMarkerTaskIds:
    """Which ids a fact asserts under ONE marker, inside the union's result."""

    def test_each_marker_gets_only_its_own_ids(self):
        fact = 'Task 5 is pending and task 6 is in progress'

        assert extract_marker_task_ids(fact, _PENDING_ALT, None) == {5}
        assert extract_marker_task_ids(fact, _IN_PROGRESS_ALT, None) == {6}

    def test_a_fact_with_no_status_marker_yields_nothing(self):
        assert extract_marker_task_ids('Task 5 is done', _PENDING_ALT, None) == set()
        assert extract_marker_task_ids(None, _PENDING_ALT, None) == set()

    def test_the_union_rejections_are_threaded_into_the_marker_family(self):
        """The task-3037 over-selection: a narrower family must not claim an id
        inside an enumeration the union family ruled a prepositional complement."""
        fact = 'Dependencies for tasks 1020 and task 1030 are pending in blocked status'

        assert extract_marker_task_ids(fact, None, _BLOCKED_ALT) == set()

    @pytest.mark.parametrize(
        'fact',
        [
            *PRECISION_GUARD_SHAPES,
            *GUARD_REJECTED_SUPPRESSION_SHAPES,
            *(fact for fact, _ in SUBJECT_POSITIVE_SHAPES),
            *ADVERBIAL_PREAMBLE_SHAPES,
            'Task 5 is pending and task 6 is in progress',
            'Tasks 1020 and 1030 are blocked',
        ],
    )
    @pytest.mark.parametrize(
        ('adjective_alt', 'transitive_alt'),
        [(_PENDING_ALT, None), (_IN_PROGRESS_ALT, None), (None, _BLOCKED_ALT)],
        ids=['pending', 'in-progress', 'blocked'],
    )
    def test_marker_ids_are_always_a_subset_of_the_union(
        self, fact, adjective_alt, transitive_alt,
    ):
        assert extract_marker_task_ids(fact, adjective_alt, transitive_alt) <= (
            extract_snapshot_edge_task_ids(fact)
        )
