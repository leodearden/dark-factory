"""Tests for the stale task-status snapshot Graphiti edge sweep (task 2613).

Reconciliation Stage 1 (MemoryConsolidator) has no deterministic sweep that
invalidates VALID (invalid_at IS NULL) task-status-snapshot Graphiti edges
once the task(s) they reference reach a terminal status (done/cancelled).
This module adds a small deterministic post-processor: enumerate valid
edges, cross-reference each referenced task_id's CURRENT status via a direct
status lookup (taskmaster.get_statuses — NOT semantic search), and
invalidate any edge whose asserted non-terminal (active/pending/blocked/
in-progress) status now contradicts a terminal task.

Covers:
- extract_snapshot_edge_task_ids: pure lexical extractor — returns the
  specific task ids a status-snapshot edge asserts as active/pending/
  blocked/in-progress; returns the empty set for text with no such status
  marker, and for pure count-only snapshots with no specific task-id (the
  stale-by-design audit trail this sweep must never touch). Includes the
  task-3042 'blocked' regression class: the closed-class-connective
  individual form ('Task N is in a blocked status ...'), the status-noun-
  anchored phrase form for open-class gaps ('Task N is deliberately parked
  in blocked status ...'), and the precision guards bounding both (no bind
  inside 'unblocked', no cross-clause/cross-comma gap).
- flatten_dedup_edges: flattens get_all_valid_edges' dict[entity_uuid,
  list[EdgeDict]] shape and dedups by edge uuid (handles the backend's
  double-attribution of each directed edge under both endpoint entities).
- select_stale_status_snapshot_edges: pure decision core — selects edges
  whose asserted status contradicts a positively-terminal current status.
- sweep_stale_status_snapshot_edges: async orchestrator — enumerates,
  cross-references, and invalidates, best-effort throughout.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (
    extract_snapshot_edge_task_ids,
    flatten_dedup_edges,
    select_stale_status_snapshot_edges,
    sweep_stale_status_snapshot_edges,
)


def _make_memory_service() -> MagicMock:
    """MagicMock memory_service with an AsyncMock .graphiti and .update_edge
    (mirrors test_degenerate_task_node_sweep.py's _make_memory_service)."""
    memory_service = MagicMock()
    memory_service.graphiti = MagicMock()
    memory_service.graphiti.get_all_valid_edges = AsyncMock(return_value={})
    memory_service.update_edge = AsyncMock()
    return memory_service


def _make_taskmaster() -> MagicMock:
    """MagicMock taskmaster with an AsyncMock .get_statuses."""
    taskmaster = MagicMock()
    taskmaster.get_statuses = AsyncMock(return_value={})
    return taskmaster


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


# --------------------------------------------------------------------------- #
# flatten_dedup_edges
# --------------------------------------------------------------------------- #


class TestFlattenDedupEdges:
    """flatten_dedup_edges(grouped) flattens get_all_valid_edges' dict[entity_uuid,
    list[EdgeDict]] shape into a flat list, deduped by edge['uuid'] — the backend
    double-attributes each directed edge under both its source and target entity.
    """

    def test_double_attributed_edge_yields_once(self):
        """An edge uuid appearing under two entity uuids (double-attribution) is
        returned exactly once."""
        edge = {'uuid': 'edge-1', 'fact': 'Task 142 is an active pending task', 'name': ''}
        grouped = {
            'entity-a': [edge],
            'entity-b': [edge],
        }

        result = flatten_dedup_edges(grouped)

        assert result == [edge], f'Expected the double-attributed edge exactly once, got {result!r}'

    def test_multiple_distinct_edges_all_preserved(self):
        """Distinct edge uuids across entities are all preserved."""
        edge1 = {'uuid': 'edge-1', 'fact': 'Task 142 is an active pending task', 'name': ''}
        edge2 = {'uuid': 'edge-2', 'fact': 'Task 148 is in-progress', 'name': ''}
        grouped = {
            'entity-a': [edge1],
            'entity-b': [edge2],
        }

        result = flatten_dedup_edges(grouped)

        assert result == [edge1, edge2], f'Expected both distinct edges preserved, got {result!r}'

    def test_empty_dict_yields_empty_list(self):
        """An empty grouped dict yields []."""
        assert flatten_dedup_edges({}) == []

    def test_result_elements_retain_uuid_fact_name_keys(self):
        """Flattened elements retain their original uuid/fact/name keys unmodified."""
        edge = {'uuid': 'edge-1', 'fact': 'Task 142 is an active pending task', 'name': 'orchestrator'}
        grouped = {'entity-a': [edge]}

        result = flatten_dedup_edges(grouped)

        assert result[0]['uuid'] == 'edge-1'
        assert result[0]['fact'] == 'Task 142 is an active pending task'
        assert result[0]['name'] == 'orchestrator'


# --------------------------------------------------------------------------- #
# select_stale_status_snapshot_edges
# --------------------------------------------------------------------------- #


class TestSelectStaleStatusSnapshotEdges:
    """select_stale_status_snapshot_edges(edges, statuses) is the pure decision
    core: an edge is selected (stale) iff any task id it asserts as
    active/pending/in-progress now has a positively-terminal (done/cancelled)
    status. Unknown/missing/still-active statuses never select an edge
    (invalidate-only-on-positively-terminal fail-safe).
    """

    def test_individual_edge_selected_when_referenced_task_done(self):
        """'Task 142 ...' with statuses={'142': 'done'} -> selected."""
        edge = {'uuid': 'edge-1', 'fact': 'Task 142 is an active pending task', 'name': ''}

        result = select_stale_status_snapshot_edges([edge], {'142': 'done'})

        assert result == [edge]

    def test_individual_edge_not_selected_when_still_active(self):
        """'Task 142 ...' with statuses={'142': 'in-progress'} -> not selected."""
        edge = {'uuid': 'edge-1', 'fact': 'Task 142 is an active pending task', 'name': ''}

        result = select_stale_status_snapshot_edges([edge], {'142': 'in-progress'})

        assert result == []

    def test_individual_edge_selected_when_referenced_task_cancelled(self):
        """'Task 142 ...' with statuses={'142': 'cancelled'} -> selected."""
        edge = {'uuid': 'edge-1', 'fact': 'Task 142 is an active pending task', 'name': ''}

        result = select_stale_status_snapshot_edges([edge], {'142': 'cancelled'})

        assert result == [edge]

    def test_aggregate_edge_selected_when_one_member_terminal(self):
        """Aggregate edge referencing [142, 148]; 142 done, 148 still pending ->
        the whole edge is selected stale on the single terminal member."""
        edge = {
            'uuid': 'edge-agg',
            'fact': 'The active pending tasks are [142, 148]',
            'name': '',
        }

        result = select_stale_status_snapshot_edges(
            [edge], {'142': 'done', '148': 'pending'},
        )

        assert result == [edge]

    def test_count_only_edge_never_selected(self):
        """A count-only edge (no extractable ids) is never selected regardless
        of the statuses map contents."""
        edge = {'uuid': 'edge-count', 'fact': 'There are 8 tasks in progress', 'name': ''}

        result = select_stale_status_snapshot_edges(
            [edge], {'142': 'done', '999': 'cancelled'},
        )

        assert result == []

    def test_referenced_id_absent_from_statuses_not_selected(self):
        """The referenced id is missing from the statuses map entirely ->
        fail-safe: not selected (an unknown census gap never invalidates)."""
        edge = {'uuid': 'edge-1', 'fact': 'Task 142 is an active pending task', 'name': ''}

        result = select_stale_status_snapshot_edges([edge], {})

        assert result == []

    def test_referenced_id_unknown_status_not_selected(self):
        """The referenced id resolves to the 'unknown' status sentinel ->
        fail-safe: not selected."""
        edge = {'uuid': 'edge-1', 'fact': 'Task 142 is an active pending task', 'name': ''}

        result = select_stale_status_snapshot_edges([edge], {'142': 'unknown'})

        assert result == []

    def test_returned_entries_carry_edge_uuid(self):
        """Selected entries are the original edge dicts, carrying their uuid."""
        edge = {'uuid': 'edge-carries-uuid', 'fact': 'Task 142 is an active pending task', 'name': ''}

        result = select_stale_status_snapshot_edges([edge], {'142': 'done'})

        assert len(result) == 1
        assert result[0]['uuid'] == 'edge-carries-uuid'

    def test_incidental_status_word_not_selected_even_if_task_done(self):
        """'Task 142 landed on the active branch' with statuses={'142': 'done'}
        -> not selected.

        Regression test (reviewer_comprehensive correctness-precision
        finding, task 2613): 'active' describes the branch, not task 142, so
        no id is extracted at all and the edge must never be treated as a
        stale status-snapshot regardless of task 142's real status —
        otherwise a genuinely-true terminal fact would be wrongly retired.
        """
        edge = {
            'uuid': 'edge-incidental',
            'fact': 'Task 142 landed on the active branch',
            'name': '',
        }

        result = select_stale_status_snapshot_edges([edge], {'142': 'done'})

        assert result == []

    def test_blocked_snapshot_edge_selected_when_task_done(self):
        """'Task 2885 is in a blocked status due to merge_retry_pending.' with
        statuses={'2885': 'done'} -> selected."""
        edge = {
            'uuid': 'edge-1',
            'fact': 'Task 2885 is in a blocked status due to merge_retry_pending.',
            'name': '',
        }

        result = select_stale_status_snapshot_edges([edge], {'2885': 'done'})

        assert result == [edge]

    def test_blocked_snapshot_edge_not_selected_when_still_blocked(self):
        """Same fact with statuses={'2885': 'blocked'} -> not selected.

        Pins the invalidate-only-on-positively-terminal semantics: 'blocked'
        is deliberately NOT a member of INACTIVE_TASK_STATUSES, so a
        genuinely-still-blocked task can never be selected by this sweep.
        """
        edge = {
            'uuid': 'edge-1',
            'fact': 'Task 2885 is in a blocked status due to merge_retry_pending.',
            'name': '',
        }

        result = select_stale_status_snapshot_edges([edge], {'2885': 'blocked'})

        assert result == []

    def test_blocked_snapshot_edge_selected_when_task_cancelled(self):
        """Same fact with statuses={'2885': 'cancelled'} -> selected."""
        edge = {
            'uuid': 'edge-1',
            'fact': 'Task 2885 is in a blocked status due to merge_retry_pending.',
            'name': '',
        }

        result = select_stale_status_snapshot_edges([edge], {'2885': 'cancelled'})

        assert result == [edge]

    def test_blocked_status_phrase_edge_selected_when_task_done(self):
        """The second repro fact ('...deliberately parked in blocked status
        under escalation esc-2885-17...') with statuses={'2885': 'done'} ->
        selected."""
        edge = {
            'uuid': 'edge-1',
            'fact': (
                'Task 2885 is deliberately parked in blocked status under '
                'escalation esc-2885-17 pending human adjudication.'
            ),
            'name': '',
        }

        result = select_stale_status_snapshot_edges([edge], {'2885': 'done'})

        assert result == [edge]


# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — core behavior
# --------------------------------------------------------------------------- #


class TestSweepStaleStatusSnapshotEdgesCore:
    """sweep_stale_status_snapshot_edges enumerates valid edges via
    memory_service.graphiti.get_all_valid_edges, cross-references each
    candidate id's CURRENT status via taskmaster.get_statuses (never
    semantic search), and invalidates only the edges
    select_stale_status_snapshot_edges identifies as stale.
    """

    @pytest.mark.asyncio
    async def test_happy_path_invalidates_only_the_stale_edge(self):
        """One stale edge (references done task 142) + one healthy edge
        (references still-pending task 999). update_edge must be awaited
        exactly once, for the stale edge's uuid only, with invalid_at set
        and project_id threaded through; get_statuses must be called with
        only the two referenced candidate ids."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()

        stale_edge = {
            'uuid': 'edge-stale', 'fact': 'Task 142 is an active pending task', 'name': '',
        }
        healthy_edge = {
            'uuid': 'edge-healthy', 'fact': 'Task 999 is an active pending task', 'name': '',
        }
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [stale_edge, healthy_edge]},
        )
        taskmaster.get_statuses = AsyncMock(return_value={'142': 'done', '999': 'pending'})

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        memory_service.update_edge.assert_awaited_once()
        update_call = memory_service.update_edge.await_args
        assert update_call.args[0] == 'edge-stale', (
            f'Expected update_edge awaited for the stale edge uuid only, got {update_call!r}'
        )
        assert isinstance(update_call.kwargs.get('invalid_at'), datetime), (
            'Expected update_edge called with a datetime invalid_at'
        )
        assert update_call.kwargs.get('project_id') == 'test_project'
        assert update_call.kwargs.get('agent_id') == 'recon-stage-memory_consolidator', (
            "Expected update_edge stamped with the sweep's stable agent_id "
            f'(reviewer_comprehensive test-coverage finding, task 2613), got {update_call!r}'
        )
        assert update_call.kwargs.get('causation_id') == 'run-1', (
            'Expected update_edge threaded the reconciliation run_id as causation_id '
            f'(reviewer_comprehensive test-coverage finding, task 2613), got {update_call!r}'
        )

        taskmaster.get_statuses.assert_awaited_once()
        get_statuses_call = taskmaster.get_statuses.await_args
        assert get_statuses_call.args[0] == '/tmp/reify'
        assert set(get_statuses_call.kwargs.get('ids', [])) == {'142', '999'}, (
            'Expected get_statuses called with only the referenced candidate ids'
        )

        assert stats == {'scanned': 2, 'candidate_edges': 1, 'invalidated': 1, 'errors': 0}, (
            f'Expected exactly the stale edge counted+invalidated, got stats={stats!r}'
        )

    @pytest.mark.asyncio
    async def test_blocked_status_snapshot_edges_invalidated_end_to_end(self):
        """Deterministic stand-in for "re-run the sweep to confirm coverage"
        (task 3042): drives the full enumerate -> extract -> cross-reference
        -> invalidate path against the two literal task-2885 repro fact
        texts plus a still-blocked control edge.

        Both repro-fact edges reference task 2885, whose census status has
        moved to 'done' — both must be invalidated. The control edge
        references task 3001, whose census status is still 'blocked' (a
        non-terminal status) — it must be left alone, pinning
        invalidate-only-on-positively-terminal for the blocked marker
        end-to-end, not just at the pure-function layer.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()

        repro_edge_1 = {
            'uuid': 'edge-2885-a',
            'fact': 'Task 2885 is in a blocked status due to merge_retry_pending.',
            'name': '',
        }
        repro_edge_2 = {
            'uuid': 'edge-2885-b',
            'fact': (
                'Task 2885 is deliberately parked in blocked status under '
                'escalation esc-2885-17 pending human adjudication.'
            ),
            'name': '',
        }
        control_edge = {
            'uuid': 'edge-3001-control', 'fact': 'Task 3001 is blocked.', 'name': '',
        }
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [repro_edge_1, repro_edge_2, control_edge]},
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2885': 'done', '3001': 'blocked'})
        now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=UTC)

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            run_id='run-2885', now=now,
        )

        assert stats == {'scanned': 3, 'candidate_edges': 2, 'invalidated': 2, 'errors': 0}, (
            f'Expected both blocked-worded task-2885 edges invalidated and the '
            f'still-blocked control edge left alone, got stats={stats!r}'
        )

        assert memory_service.update_edge.await_count == 2, (
            'Expected update_edge awaited exactly twice, for the two repro-fact edges only'
        )
        invalidated_uuids = {call.args[0] for call in memory_service.update_edge.await_args_list}
        assert invalidated_uuids == {'edge-2885-a', 'edge-2885-b'}, (
            f'Expected only the two task-2885 repro edges invalidated, never the '
            f'still-blocked control edge, got {invalidated_uuids!r}'
        )
        for call in memory_service.update_edge.await_args_list:
            assert call.kwargs.get('invalid_at') == now, (
                f'Expected update_edge called with the explicit now= timestamp, got {call!r}'
            )
            assert call.kwargs.get('project_id') == 'test_project'
            assert call.kwargs.get('agent_id') == 'recon-stage-memory_consolidator'
            assert call.kwargs.get('causation_id') == 'run-2885'

        taskmaster.get_statuses.assert_awaited_once()
        get_statuses_call = taskmaster.get_statuses.await_args
        assert get_statuses_call.args[0] == '/tmp/reify'
        assert '2885' in set(get_statuses_call.kwargs.get('ids', [])), (
            'Expected get_statuses called with 2885 among the candidate ids — proving the '
            'blocked-worded ids now reach the cross-reference instead of being dropped by '
            'the gate'
        )


# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — guards
# --------------------------------------------------------------------------- #


class TestSweepStaleStatusSnapshotEdgesGuards:
    """Guard behavior: an unavailable taskmaster/project_root, or an
    enumeration with no candidate ids, must short-circuit without
    unnecessary backend calls."""

    @pytest.mark.asyncio
    async def test_taskmaster_none_yields_all_zero_stats_with_no_calls(self):
        """taskmaster=None -> all-zero stats; get_all_valid_edges never awaited."""
        memory_service = _make_memory_service()

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, None, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats == {'scanned': 0, 'candidate_edges': 0, 'invalidated': 0, 'errors': 0}
        memory_service.graphiti.get_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_candidate_ids_skips_get_statuses(self):
        """Enumeration returns only count-only/non-status edges (no candidate
        ids) -> get_statuses never awaited and invalidated == 0."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        count_only_edge = {
            'uuid': 'edge-count', 'fact': 'There are 8 tasks in progress', 'name': '',
        }
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [count_only_edge]},
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        taskmaster.get_statuses.assert_not_awaited()
        assert stats['invalidated'] == 0


# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — best-effort resilience
# --------------------------------------------------------------------------- #


class TestSweepStaleStatusSnapshotEdgesBestEffort:
    """A transient backend error enumerating, cross-referencing, or invalidating
    one edge must not abort the sweep for the remaining edges — it is tallied
    into stats['errors'] and the sweep continues (mirrors
    sweep_degenerate_task_nodes's best-effort posture). asyncio.CancelledError
    is never swallowed."""

    @pytest.mark.asyncio
    async def test_update_edge_failure_is_swallowed_and_second_update_still_attempted(self):
        """update_edge raises for the first stale edge; the second stale edge's
        update_edge is still attempted. Only the successful one counts as
        invalidated, and the failure counts as an error."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()

        stale_edge_1 = {
            'uuid': 'edge-stale-1', 'fact': 'Task 142 is an active pending task', 'name': '',
        }
        stale_edge_2 = {
            'uuid': 'edge-stale-2', 'fact': 'Task 144 is an active pending task', 'name': '',
        }
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [stale_edge_1, stale_edge_2]},
        )
        taskmaster.get_statuses = AsyncMock(return_value={'142': 'done', '144': 'cancelled'})
        memory_service.update_edge = AsyncMock(
            side_effect=[RuntimeError('lock contention'), None],
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert memory_service.update_edge.await_count == 2, (
            'The second stale edge must still be attempted after the first update fails'
        )
        assert stats == {'scanned': 2, 'candidate_edges': 2, 'invalidated': 1, 'errors': 1}, (
            f'Expected the failed update tallied as an error without blocking the second, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_get_all_valid_edges_failure_yields_all_zero_stats_without_raising(self):
        """get_all_valid_edges raises -> sweep returns all-zero stats (errors
        tallied) without raising and without calling get_statuses/update_edge."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            side_effect=RuntimeError('transient read timeout'),
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats == {'scanned': 0, 'candidate_edges': 0, 'invalidated': 0, 'errors': 1}, (
            f'Expected all-zero stats with the failure tallied as an error, got {stats!r}'
        )
        taskmaster.get_statuses.assert_not_awaited()
        memory_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_statuses_failure_is_best_effort_no_op(self):
        """get_statuses raises -> best-effort no-op: errors tallied, no
        update_edge attempted, no raise."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        candidate_edge = {
            'uuid': 'edge-candidate', 'fact': 'Task 142 is an active pending task', 'name': '',
        }
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [candidate_edge]},
        )
        taskmaster.get_statuses = AsyncMock(side_effect=RuntimeError('transient read timeout'))

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats == {'scanned': 1, 'candidate_edges': 0, 'invalidated': 0, 'errors': 1}, (
            f'Expected the get_statuses failure tallied as an error with no invalidation, got {stats!r}'
        )
        memory_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancelled_error_from_update_edge_propagates(self):
        """asyncio.CancelledError raised from update_edge must propagate — it
        is never swallowed as a best-effort error."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        stale_edge = {
            'uuid': 'edge-stale', 'fact': 'Task 142 is an active pending task', 'name': '',
        }
        memory_service.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [stale_edge]},
        )
        taskmaster.get_statuses = AsyncMock(return_value={'142': 'done'})
        memory_service.update_edge = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await sweep_stale_status_snapshot_edges(
                memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
            )
