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
- build_supersede_fact: the resulting-state-only fact written after a stale
  blocked edge is retired, worded so the sweep's own extractor never selects
  it again.
- flatten_dedup_edges: flattens get_all_valid_edges' dict[entity_uuid,
  list[EdgeDict]] shape and dedups by edge uuid (handles the backend's
  double-attribution of each directed edge under both endpoint entities).
- select_stale_status_snapshot_edges: pure decision core — selects edges
  whose asserted status contradicts a positively-terminal current status,
  and, through is_blocked_assertion_contradicted, blocked assertions whose
  task is positively known to be no longer blocked.
- sweep_stale_status_snapshot_edges: async orchestrator — enumerates,
  cross-references, and invalidates, best-effort throughout.
- sweep_stale_status_snapshot_edges_for_task: the same rules over a read
  narrowed to one task's edges, and overlapping sweeps of one project
  writing each superseding fact once.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import complete_paged_read, incomplete_paged_read
from shared.task_statuses import TaskStatus

from fused_memory.backends.graphiti_client import (
    INCOMPLETE_CENSUS_UNAVAILABLE,
    INCOMPLETE_PAGE_CAP,
    INCOMPLETE_SHORT_READ,
    INCOMPLETE_STRUCTURAL_KINDS,
    IncompleteEnumerationError,
)
from fused_memory.reconciliation import task_filter
from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (
    _MAX_SUPERSEDE_WRITES_PER_CYCLE,
    STATUS_SNAPSHOT_ENUMERATION_COMPLETE_STAT_KEY,
    STATUS_SNAPSHOT_ENUMERATION_INCOMPLETE_KIND_STAT_KEY,
    build_supersede_fact,
    flatten_dedup_edges,
    select_stale_status_snapshot_edges,
    sweep_stale_status_snapshot_edges,
    sweep_stale_status_snapshot_edges_for_task,
)
from fused_memory.reconciliation.status_snapshot_extraction import (
    extract_blocked_assertion_task_ids,
    extract_snapshot_edge_task_ids,
)
from reconciliation.counting_str import CountingStr


def _make_memory_service() -> MagicMock:
    """MagicMock memory_service with an AsyncMock .graphiti, .update_edge and
    .add_memory (mirrors test_degenerate_task_node_sweep.py's
    _make_memory_service).

    ``.add_memory`` must be an AsyncMock explicitly: a bare MagicMock
    attribute returns a MagicMock, and awaiting one raises TypeError — so
    without this every supersede-write path would fail for the wrong reason.
    (task 3037)
    """
    memory_service = MagicMock()
    memory_service.graphiti = MagicMock()
    memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
        return_value=({}, complete_paged_read()),
    )
    memory_service.update_edge = AsyncMock()
    memory_service.add_memory = AsyncMock()
    return memory_service


def _make_taskmaster() -> MagicMock:
    """MagicMock taskmaster with an AsyncMock .get_statuses."""
    taskmaster = MagicMock()
    taskmaster.get_statuses = AsyncMock(return_value={})
    return taskmaster


# --------------------------------------------------------------------------- #
# build_supersede_fact — the superseding temporal_fact's wording (task 3037)
# --------------------------------------------------------------------------- #


class TestBuildSupersedeFact:
    """build_supersede_fact(task_id, status, now) renders the resulting-state-
    only temporal_fact the sweep writes after retiring a stale blocked edge.

    The wording is a VERIFIED CONSTRAINT, not a style choice, and this class
    is where that verification lives. Two independent hazards bound it:

    1. SELF-CHURN. The sweep's own extractor must not find task ids in the
       fact it just wrote, or the superseding fact would select ITSELF for
       invalidation on the next cycle — an infinite write loop. See the
       contrast test below for the obvious wording that does exactly that.
    2. WRITE NORMS. prompts/stage1.py's resulting-state-only rule forbids
       PRIOR-state framing for temporal_facts, because Graphiti atomization
       can re-emit the stale fragment as a co-current edge. That norm is
       enforced at the MCP boundary (server/tools.py:2856/3061,
       ReconMixedFramingWriteRejected) for any recon-stage-* writer — but the
       sweep writes IN-PROCESS via memory_service.add_memory and so bypasses
       that gate entirely. The four predicates are therefore asserted here,
       directly, rather than assumed.
    """

    def test_template_shape(self):
        """'As of 2026-08-21, task 2848 has status pending.'"""
        out = build_supersede_fact(2848, 'pending', datetime(2026, 8, 21, 14, 3, tzinfo=UTC))

        assert out == 'As of 2026-08-21, task 2848 has status pending.'

    def test_date_comes_from_the_passed_in_timestamp(self):
        """The date is formatted from *now*, never from a module-level clock.

        The sweep threads one invalidation timestamp through the whole cycle;
        the superseding fact must carry that same instant, so a replayed or
        back-dated run is reproducible.
        """
        out = build_supersede_fact(7, 'done', datetime(2024, 1, 2, 23, 59, 59, tzinfo=UTC))

        assert out.startswith('As of 2024-01-02,'), out

    @pytest.mark.parametrize('status', sorted(s.value for s in TaskStatus))
    def test_generated_fact_does_not_select_itself(self, status):
        """Neither extractor finds an id in the generated fact, for any status.

        Exhaustive over the closed vocabulary rather than a sample: the sweep
        writes whatever status the census reports, so a single status whose
        rendering re-extracts would be an infinite write loop in production
        while every other status looked fine.
        """
        out = build_supersede_fact(2848, status, datetime(2026, 8, 21, tzinfo=UTC))

        assert extract_snapshot_edge_task_ids(out) == set(), (
            f'The superseding fact {out!r} re-extracts as a status snapshot, so the '
            f'sweep would select it for invalidation next cycle and write another one '
            f'— an infinite write loop.'
        )
        assert extract_blocked_assertion_task_ids(out) == set(), (
            f'The superseding fact {out!r} re-extracts as a BLOCKED assertion.'
        )

    @pytest.mark.parametrize('status', sorted(s.value for s in TaskStatus))
    def test_generated_fact_satisfies_the_recon_write_norms(self, status):
        """None of the four MCP-boundary write guards fires, for any status.

        These are the same predicates server/tools.py enforces on
        recon-stage-* temporal_facts writers. The sweep's agent_id IS
        recon-stage-memory_consolidator, but it writes in-process and never
        crosses that boundary, so this test is the only thing standing
        between a malformed superseding fact and the graph.
        """
        out = build_supersede_fact(2848, status, datetime(2026, 8, 21, tzinfo=UTC))

        assert not task_filter.is_mixed_temporal_framing(out), (
            f'{out!r} co-mentions a prior-state and a resulting-state marker'
        )
        assert task_filter.find_conflicting_task_status_ids(out) == set(), (
            f'{out!r} frames a task as both non-terminal and terminal'
        )
        assert not task_filter.frames_live_task_status_as_current_fact(out), (
            f'{out!r} frames a live task-table field as a standing fact rather than a '
            f'timestamped point-in-time check'
        )
        assert not task_filter.is_count_snapshot(out), (
            f'{out!r} reads as a count snapshot'
        )

    def test_contrast_the_obvious_wording_is_forbidden(self):
        """WHY the template is resulting-state-only, pinned as a live contrast.

        'Task 2848 was blocked and is now pending' is the wording a reader
        reaches for first. It is forbidden twice over:

        - this module's OWN extractor re-extracts it as {2848}, so writing it
          would make the superseding fact select itself for invalidation
          every cycle and write a fresh copy — an unbounded write loop
          against the graph;
        - it is mixed temporal framing, which prompts/stage1.py forbids for
          temporal_facts because Graphiti atomization can re-emit the stale
          'was blocked' fragment as a co-current edge — reintroducing exactly
          the stale assertion this sweep just retired.

        Kept as an executable contrast rather than a comment so the reason
        stays true rather than merely remembered.
        """
        forbidden = 'Task 2848 was blocked and is now pending'

        assert extract_snapshot_edge_task_ids(forbidden) == {2848}
        assert task_filter.is_mixed_temporal_framing(forbidden)

    def test_prior_state_negation_is_extractor_safe_but_still_not_the_template(self):
        """'Task 2848 is no longer blocked' passes every EXECUTABLE check and is
        still not the wording to use — the rule against it is prompt-level only.

        This test deliberately claims LESS than its sibling
        test_contrast_the_obvious_wording_is_forbidden, which can point at a
        real predicate (is_mixed_temporal_framing) that rejects
        'was blocked and is now pending'. Here there is no such predicate:
        both extractors return set() AND all four write-norm predicates the
        MCP boundary enforces pass it, as asserted below. So nothing in the
        code would stop an author who switched to this wording — the only
        thing forbidding it is the resulting-state-only norm written in
        prompts/stage1.py, which is prose.

        Named and scoped this way on purpose (amendment,
        reviewer_comprehensive test-quality finding): the earlier version
        asserted only the two extractor results while its docstring claimed
        the wording was "forbidden", which would have left a future reader
        believing an automated guard exists here. It does not, and the
        template's compliance rests on TestBuildSupersedeFact plus review.
        """
        prior_state = 'Task 2848 is no longer blocked'

        # Extractor-safe: it would not select itself for invalidation.
        assert extract_snapshot_edge_task_ids(prior_state) == set()
        assert extract_blocked_assertion_task_ids(prior_state) == set()

        # ...and NO write-norm predicate rejects it either. These four
        # assertions are the point of the test: they pin that the prior-state
        # prohibition has no executable backing, so a future author does not
        # go looking for the guard that supposedly caught it.
        assert not task_filter.is_mixed_temporal_framing(prior_state)
        assert task_filter.find_conflicting_task_status_ids(prior_state) == set()
        assert not task_filter.frames_live_task_status_as_current_fact(prior_state)
        assert not task_filter.is_count_snapshot(prior_state)

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


class TestSelectStaleStatusSnapshotEdgesBlockedRule:
    """The SECOND selection rule (task 3037): a BLOCKED assertion is stale as
    soon as the task's status is positively known and is not 'blocked' — not
    only when it reaches a terminal (done/cancelled) status.

    The measured defect this closes, live at HEAD=c54b8cfcc5 against the edge
    fact 'Task 2848 remains blocked as of 2026-07-22':

        extract_snapshot_edge_task_ids       -> {2848}   (extraction is fine)
        select(..., {'2848': 'pending'})     -> []       <- the defect
        select(..., {'2848': 'in-progress'}) -> []       <- the defect
        select(..., {'2848': 'review'})      -> []       <- the defect
        select(..., {'2848': 'blocked'})     -> []       (correct)
        select(..., {'2848': 'done'})        -> [edge]   (the only path that
                                                          retired it before)

    So a blocked->pending unblock left the 'Task N remains blocked' edge live
    until the task eventually reached done/cancelled. A blocked assertion is
    contradicted by ANY non-blocked status, and restricting the trigger to
    INACTIVE_TASK_STATUSES is precisely what made it survive.

    Every case here calls select_stale_status_snapshot_edges WITHOUT any
    precomputed-ids kwarg, so it pins the pure decision core rather than the
    sweep's precomputation plumbing.
    """

    #: The live repro fact, verbatim.
    BLOCKED_FACT = 'Task 2848 remains blocked as of 2026-07-22'

    def _edge(self, fact: str = BLOCKED_FACT) -> dict:
        return {'uuid': 'edge-2848', 'fact': fact, 'name': ''}

    @pytest.mark.parametrize(
        'status',
        sorted(s.value for s in TaskStatus if s is not TaskStatus.BLOCKED),
    )
    def test_blocked_assertion_selected_for_every_non_blocked_status(self, status):
        """A blocked assertion is stale under EVERY other member of the closed
        status vocabulary — the six non-terminal ones (pending, in-progress,
        deferred, review, merge-deferred, infra-hold) via the new rule, and
        done/cancelled via the pre-existing terminal rule, which must keep
        working.

        Parametrized over shared.task_statuses.TaskStatus rather than a
        hand-written list so the case table is exhaustive over the real
        vocabulary and cannot drift when a status is added.
        """
        edge = self._edge()

        result = select_stale_status_snapshot_edges([edge], {'2848': status})

        assert result == [edge], (
            f"Expected the blocked assertion {self.BLOCKED_FACT!r} selected as stale "
            f'when task 2848 status is {status!r} — a blocked assertion is contradicted '
            f'by ANY non-blocked status, not only a terminal one'
        )

    def test_blocked_assertion_not_selected_when_still_blocked(self):
        """statuses={'2848': 'blocked'} -> NOT selected (genuinely still blocked)."""
        edge = self._edge()

        assert select_stale_status_snapshot_edges([edge], {'2848': 'blocked'}) == []

    def test_blocked_assertion_not_selected_when_id_absent_from_census(self):
        """The id is missing from the census entirely -> NOT selected.

        Invalidate-only-on-positively-known: a transient census gap (a
        paginated get_statuses page that failed, a task filed since the
        enumeration) can only UNDER-select and self-heal next cycle. Over-
        selection is unrecoverable — an invalidated Graphiti edge is never
        re-validated.
        """
        edge = self._edge()

        assert select_stale_status_snapshot_edges([edge], {}) == []
        assert select_stale_status_snapshot_edges([edge], {'9999': 'pending'}) == []

    def test_blocked_assertion_not_selected_for_unrecognised_status_value(self):
        """A census value outside the closed TaskStatus vocabulary -> NOT selected.

        'unknown' is not 'blocked', so a naive ``!= 'blocked'`` test would
        retire the edge on it. "Positively known" is defined against the
        closed shared.task_statuses.TaskStatus vocabulary precisely so a
        malformed/renamed/sentinel census value reads as UNKNOWN rather than
        as a contradiction — the same fail-safe direction the terminal rule
        has.
        """
        edge = self._edge()

        assert select_stale_status_snapshot_edges([edge], {'2848': 'unknown'}) == []
        assert select_stale_status_snapshot_edges([edge], {'2848': ''}) == []

    @pytest.mark.parametrize('status', ['in-progress', 'review', 'deferred', 'blocked'])
    def test_non_blocked_assertion_is_out_of_the_new_rule_scope(self, status):
        """SCOPE GUARD — this is what forbids the new rule generalising.

        An edge asserting a NON-blocked marker ('Task 999 is an active
        pending task') must NOT be selected merely because 999 moved to
        another active status. Only the pre-existing terminal rule still
        retires it.

        Without this guard the new rule would silently widen to every
        active/pending snapshot edge in the corpus, since
        extract_snapshot_edge_task_ids returns the UNION over all markers
        and cannot tell a pending assertion from a blocked one. That is why
        the rule keys on extract_blocked_assertion_task_ids instead.
        """
        edge = {'uuid': 'edge-999', 'fact': 'Task 999 is an active pending task', 'name': ''}

        assert select_stale_status_snapshot_edges([edge], {'999': status}) == []

    @pytest.mark.parametrize('status', ['done', 'cancelled'])
    def test_non_blocked_assertion_still_retired_by_the_terminal_rule(self, status):
        """The other half of the scope guard: the terminal rule is untouched."""
        edge = {'uuid': 'edge-999', 'fact': 'Task 999 is an active pending task', 'name': ''}

        assert select_stale_status_snapshot_edges([edge], {'999': status}) == [edge]

    @pytest.mark.parametrize('fact', ['Task 7 is stalled.', 'Task 7 is an active task'])
    @pytest.mark.parametrize(
        ('status', 'selected'),
        [
            ('pending', False),
            ('in-progress', False),
            ('review', False),
            ('deferred', False),
            ('blocked', False),
            ('done', True),
            ('cancelled', True),
        ],
    )
    def test_stalled_and_active_assertions_are_terminal_rule_only(self, fact, status, selected):
        """Pins the task-4851 decision NOT to adopt a general contradiction rule.

        'active' maps to several TaskStatus values and 'stalled' to none, so
        neither marker has a live status it could be contradicted by. Only
        the terminal rule retires them; any non-terminal live status leaves
        them alone. Verdict: plans/stale-status-snapshot-sweep-report.md.
        """
        edge = {'uuid': 'edge-7', 'fact': fact, 'name': ''}

        expected = [edge] if selected else []
        assert select_stale_status_snapshot_edges([edge], {'7': status}) == expected

    def test_aggregate_blocked_edge_selected_on_a_single_unblocked_member(self):
        """'Blocked tasks: 142, 148' with 142 unblocked to 'pending' and 148
        still blocked -> the whole edge is selected.

        Same whole-edge semantics the terminal rule already has: the snapshot
        as asserted no longer holds the moment ANY one referenced id
        contradicts it.
        """
        edge = {'uuid': 'edge-agg', 'fact': 'Blocked tasks: 142, 148', 'name': ''}

        result = select_stale_status_snapshot_edges(
            [edge], {'142': 'pending', '148': 'blocked'},
        )

        assert result == [edge]

    def test_blocked_on_another_task_keys_on_the_blocked_task_not_the_blocker(self):
        """'Task 5 is blocked on task 2862': only 5 is asserted blocked.

        2862 is what task 5 is blocked ON. A rule keyed on 2862's status
        would be reading the fact backwards — so 2862 going to 'done' must
        NOT retire this edge through the BLOCKED rule.  (It is still retired
        by the pre-existing TERMINAL rule, which reads union ids and sees
        both — deliberately unchanged here; the guard is that the blocked
        rule alone does not fire on the blocker.)
        """
        edge = {'uuid': 'edge-5', 'fact': 'Task 5 is blocked on task 2862', 'name': ''}

        assert select_stale_status_snapshot_edges([edge], {'2862': 'pending'}) == []
        assert select_stale_status_snapshot_edges([edge], {'5': 'pending'}) == [edge]

    def test_transitive_verb_list_collision_is_not_retired_by_the_blocked_rule(self):
        """'Task 5 blocked tasks: 142, 148' is a permanently-true HISTORICAL
        fact, and the blocked rule must not retire it on a non-terminal status.

        The select-layer half of
        tests/reconciliation/test_status_snapshot_extraction.py::
        TestExtractBlockedAssertionTaskIds::test_transitive_verb_list_collision_excluded:
        the extractor guard is only worth anything if it actually keeps the
        edge out of the selection. Both directions are pinned — the blocked
        rule declines it at 'pending', and the pre-existing TERMINAL rule
        still retires it at 'done', which is the behaviour task 2613 shipped
        and task 3037 deliberately left alone.
        (amendment, reviewer_comprehensive correctness-over-selection finding)
        """
        edge = {'uuid': 'edge-hist', 'fact': 'Task 5 blocked tasks: 142, 148', 'name': ''}

        assert select_stale_status_snapshot_edges([edge], {'142': 'pending'}) == []
        assert select_stale_status_snapshot_edges([edge], {'142': 'in-progress'}) == []
        assert select_stale_status_snapshot_edges([edge], {'142': 'done'}) == [edge]

# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — core behavior
# --------------------------------------------------------------------------- #


class TestSweepStaleStatusSnapshotEdgesCore:
    """sweep_stale_status_snapshot_edges enumerates valid edges via
    memory_service.graphiti.enumerate_all_valid_edges, cross-references each
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
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [stale_edge, healthy_edge]},
                complete_paged_read(rows_seen=2),
            ),
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

        assert stats == {
            'scanned': 2, 'candidate_edges': 1,
            'invalidated': 1, 'errors': 0,
            'superseded': 0, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': True, 'enumeration_incomplete_kind': None,
        }, (
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
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [repro_edge_1, repro_edge_2, control_edge]},
                complete_paged_read(rows_seen=3),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2885': 'done', '3001': 'blocked'})
        now = datetime(2026, 7, 30, 12, 0, 0, tzinfo=UTC)

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            run_id='run-2885', now=now,
        )

        assert stats == {
            'scanned': 3, 'candidate_edges': 2,
            'invalidated': 2, 'errors': 0,
            'superseded': 1, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': True, 'enumeration_incomplete_kind': None,
        }, (
            f'Expected both blocked-worded task-2885 edges invalidated and the '
            f'still-blocked control edge left alone, got stats={stats!r}'
        )
        # superseded == 1, not 2: both edges assert task 2885 as blocked and
        # 'done' is positively known and is not 'blocked', so the task-3037
        # blocked rule fires here too and contributes ONE superseding fact for
        # the task — deduped across both edges. (The terminal rule alone would
        # have written none; see
        # TestSweepStaleStatusSnapshotEdgesSupersedeWrite for both halves of
        # that boundary.)

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

    @pytest.mark.asyncio
    async def test_genitive_and_plural_snapshot_edges_invalidated_end_to_end(self):
        """Deterministic stand-in for "re-run the sweep to confirm coverage"
        (task 3079), following the task-3042 precedent directly above.

        A LIVE re-run cannot prove anything here: Stage 1 already
        invalidated and superseded all four of the finding's real edges
        this cycle, so re-running the sweep would observe nothing and
        report zero by construction. Replaying both verbatim fact shapes
        (including the finding's real edge uuid) through the real
        orchestrator is the repeatable proof that survives long after the
        incident edges are gone.

        Drives the full enumerate -> extract -> cross-reference ->
        invalidate path. Both edges reference tasks whose census status is
        now 'done', so both must be invalidated; the control edge
        references task 3100, still 'pending', and must be left alone —
        pinning invalidate-only-on-positively-terminal end-to-end for the
        two new paths, not just at the pure-function layer.

        The get_statuses assertion is the load-bearing one for the
        finding's second question: ALL THREE ids of the pairwise fact must
        reach the cross-reference, not merely the first.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

        genitive_edge = {
            # the finding's real edge uuid and verbatim fact
            'uuid': '3e78fd74-83a9-48bb-84ba-b13d32050439',
            'fact': (
                "Task 2623's status is pending as of 2026-07-14T20:31:00Z, "
                'linked to memory f20fbf15-41b7-4781-97a6-334f4e1d066a.'
            ),
            'name': '',
        }
        plural_edge = {
            'uuid': 'edge-1020-1030-1031',
            'fact': 'Tasks 1020, 1030, and 1031 are pairwise-stalled as of 2026-04-29.',
            'name': '',
        }
        control_edge = {
            'uuid': 'edge-3100-control', 'fact': 'Task 3100 is pending.', 'name': '',
        }
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [genitive_edge, plural_edge, control_edge]},
                complete_paged_read(rows_seen=3),
            ),
        )
        taskmaster.get_statuses = AsyncMock(
            return_value={
                '1020': 'done', '1030': 'done', '1031': 'done',
                '2623': 'done', '3100': 'pending',
            },
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            run_id='run-3079', now=now,
        )

        assert stats == {
            'scanned': 3, 'candidate_edges': 2,
            'invalidated': 2, 'errors': 0,
            'superseded': 0, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': True, 'enumeration_incomplete_kind': None,
        }, (
            f'Expected both repro-shape edges invalidated and the still-pending '
            f'control edge left alone, got stats={stats!r}'
        )

        assert memory_service.update_edge.await_count == 2, (
            'Expected update_edge awaited exactly twice, for the two repro-shape edges only'
        )
        invalidated_uuids = {call.args[0] for call in memory_service.update_edge.await_args_list}
        assert invalidated_uuids == {
            '3e78fd74-83a9-48bb-84ba-b13d32050439', 'edge-1020-1030-1031',
        }, (
            f'Expected only the two repro-shape edges invalidated, never the '
            f'still-pending control edge, got {invalidated_uuids!r}'
        )
        for call in memory_service.update_edge.await_args_list:
            assert call.kwargs.get('invalid_at') == now, (
                f'Expected update_edge called with the explicit now= timestamp, got {call!r}'
            )
            assert call.kwargs.get('project_id') == 'test_project'
            assert call.kwargs.get('agent_id') == 'recon-stage-memory_consolidator'
            assert call.kwargs.get('causation_id') == 'run-3079'

        taskmaster.get_statuses.assert_awaited_once()
        get_statuses_call = taskmaster.get_statuses.await_args
        assert get_statuses_call.args[0] == '/tmp/reify'
        assert {'1020', '1030', '1031', '2623'} <= set(
            get_statuses_call.kwargs.get('ids', [])
        ), (
            'Expected every candidate id to reach the cross-reference — 2623 from the '
            'genitive shape, and ALL THREE of 1020/1030/1031 from the pairwise shape '
            'rather than only the first, which is the precise gap the finding reports'
        )

    @pytest.mark.asyncio
    async def test_prepositional_complement_enumeration_is_never_retired(self):
        """A prepositional-complement enumeration must survive the sweep whole —
        no invalidation, no superseding write. (amendment,
        reviewer_comprehensive correctness-over-selection finding, task 3037)

        'Dependencies for tasks 1020 and task 1030 are pending in blocked
        status' asserts that the DEPENDENCIES are pending, in blocked status.
        It asserts nothing whatever about the STATUS of task 1020 or task
        1030, so no rule may retire it. The terminal rule already declines
        (the union extractor refuses the whole fact -> set(), so the edge is
        never even a candidate); the blocked rule at the branch tip did not,
        because its own family never built the enumeration span the union
        family rejected and its individual arm then claimed the repeated
        'task 1030' reference inside it.

        This is the leg that makes the extractor finding a DATA-LOSS bug
        rather than a curiosity, which is why it is pinned at the sweep layer
        and not only at the extractor. MEASURED at HEAD=fe8e74c12c, this
        exact input produced stats {'scanned': 1, 'candidate_edges': 1,
        'invalidated': 1, 'superseded': 1}: update_edge(invalid_at=...)
        PERMANENTLY retired a TRUE edge, and add_memory additionally wrote
        'As of <date>, task 1030 has status pending.' — attributing a status
        to a subject the fact never made a claim about. Under-selection
        self-heals on the next cycle; this does not.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()

        true_edge = {
            'uuid': 'edge-prepositional-complement',
            'fact': 'Dependencies for tasks 1020 and task 1030 are pending in blocked status',
            'name': '',
        }
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [true_edge]},
                complete_paged_read(rows_seen=1),
            ),
        )
        # Both ids POSITIVELY KNOWN and != 'blocked', so the blocked rule
        # fires the moment the extractor hands it either of them: the census
        # is deliberately not the thing standing between this edge and
        # retirement.
        taskmaster.get_statuses = AsyncMock(
            return_value={'1020': 'pending', '1030': 'pending'},
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            run_id='run-3037-overselection', now=datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
        )

        assert stats == {
            'scanned': 1, 'candidate_edges': 0,
            'invalidated': 0, 'errors': 0,
            'superseded': 0, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': True, 'enumeration_incomplete_kind': None,
        }, (
            f'Expected the prepositional-complement enumeration scanned but never '
            f'selected — it asserts nothing about task 1020 or 1030 — got stats={stats!r}'
        )

        memory_service.update_edge.assert_not_awaited()
        memory_service.add_memory.assert_not_awaited()


class TestSweepStaleStatusSnapshotEdgesBlockedRuleAndCounters:
    """The blocked-assertion rule at the SWEEP layer, plus the stat-counter
    coverage task 3037 asks for over BOTH selection rules (task 3037).

    On the stat counter: the folded-in finding reported
    ``stale_status_snapshot_edges_invalidated`` reading 0 of 6312 scanned
    while an edge WAS invalidated that same run, and attributed it to "a
    code-level bug in the counter". Measured at HEAD=c54b8cfcc5 there is no
    arithmetic defect — the module has exactly ONE update_edge call site and
    ``stats['invalidated'] += 1`` sits immediately after the awaited call,
    inside the try and before the except. The 0 was CORRECT for the sweep:
    the terminal-only selection rule never selected the blocked edge, and the
    Stage-1 LLM agent invalidated it ad hoc via the MCP update_edge tool,
    which no sweep counter observes. Same root cause as this task's primary
    defect, not a second bug.

    The requirement is therefore met STRUCTURALLY rather than by chasing a
    nonexistent bug: the new rule is routed through the SAME single counted
    invalidation loop, both rules are covered by one parametrized counter
    test, and ``invalidated == candidate_edges - errors`` is pinned as an
    explicit invariant — which is what actually forbids a future second
    invalidation site from going uncounted.
    """

    @pytest.mark.asyncio
    async def test_blocked_assertion_edge_invalidated_end_to_end(self):
        """The live repro, driven through the full enumerate -> extract ->
        cross-reference -> invalidate path.

        'Task 2848 remains blocked as of 2026-07-22' with census
        {'2848': 'pending'} must be invalidated; the still-blocked control
        edge (task 3001, census 'blocked') must be left alone. Before task
        3037 the repro edge was not even SELECTED — done/cancelled was the
        only trigger — so it stayed live through the entire unblock.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()

        blocked_edge = {
            'uuid': 'edge-2848',
            'fact': 'Task 2848 remains blocked as of 2026-07-22',
            'name': '',
        }
        control_edge = {
            'uuid': 'edge-3001-control', 'fact': 'Task 3001 is blocked.', 'name': '',
        }
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [blocked_edge, control_edge]},
                complete_paged_read(rows_seen=2),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'pending', '3001': 'blocked'})
        now = datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC)

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            run_id='run-3037', now=now,
        )

        memory_service.update_edge.assert_awaited_once()
        call = memory_service.update_edge.await_args
        assert call.args[0] == 'edge-2848', (
            f'Expected only the unblocked task-2848 edge invalidated, never the '
            f'still-blocked control edge, got {call!r}'
        )
        assert call.kwargs.get('invalid_at') == now, (
            f'Expected update_edge called with the explicit now= timestamp, got {call!r}'
        )
        assert call.kwargs.get('project_id') == 'test_project'
        assert call.kwargs.get('agent_id') == 'recon-stage-memory_consolidator'
        assert call.kwargs.get('causation_id') == 'run-3037'

        assert stats['scanned'] == 2
        assert stats['candidate_edges'] == 1
        assert stats['errors'] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('rule_name', 'fact', 'census'),
        [
            ('blocked-assertion rule (task 3037)',
             'Task 2848 remains blocked as of 2026-07-22', {'2848': 'pending'}),
            ('terminal rule (task 2613)',
             'Task 142 is an active pending task', {'142': 'done'}),
        ],
    )
    async def test_invalidated_counter_covers_both_selection_rules(
        self, rule_name, fact, census,
    ):
        """stats['invalidated'] == 1 under EITHER rule.

        Written as ONE parametrized test over both rules rather than two
        independent tests, so a future change that counts one rule but not
        the other fails immediately instead of leaving a silent gap. The
        terminal leg is a REGRESSION PIN (already green on arrival); the
        blocked leg is the new coverage.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [{'uuid': 'edge-1', 'fact': fact, 'name': ''}]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value=census)

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats['invalidated'] == 1, (
            f'Expected the invalidation counted under the {rule_name}, got stats={stats!r}'
        )

    @pytest.mark.asyncio
    async def test_invalidated_equals_candidates_minus_errors_across_a_mixed_run(self):
        """COUNTER INVARIANT: invalidated == candidate_edges - errors.

        This is the structural remedy for the stat-counter requirement. It
        fails the moment ANY edge is invalidated outside the single counted
        loop, which is the failure mode the folded-in finding feared — far
        stronger than re-asserting the increment that already exists.

        Driven over a mixed run exercising both rules and both outcomes: a
        terminal-path stale edge, a blocked-path stale edge, a healthy edge,
        and one stale edge whose update_edge raises.

        The identity is EXACT only because stats['errors'] stays scoped to
        the enumerate / cross-reference / INVALIDATE paths. Supersede-write
        failures deliberately get their own 'supersede_errors' key rather
        than diluting 'errors' — see the sweep's Returns docstring.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()

        terminal_stale = {
            'uuid': 'edge-terminal', 'fact': 'Task 142 is an active pending task', 'name': '',
        }
        blocked_stale = {
            'uuid': 'edge-blocked',
            'fact': 'Task 2848 remains blocked as of 2026-07-22',
            'name': '',
        }
        healthy = {
            'uuid': 'edge-healthy', 'fact': 'Task 3001 is blocked.', 'name': '',
        }
        failing = {
            'uuid': 'edge-failing', 'fact': 'Task 144 is an active pending task', 'name': '',
        }
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [terminal_stale, blocked_stale, healthy, failing]},
                complete_paged_read(rows_seen=4),
            ),
        )
        taskmaster.get_statuses = AsyncMock(
            return_value={
                '142': 'done', '2848': 'pending', '3001': 'blocked', '144': 'cancelled',
            },
        )
        memory_service.update_edge = AsyncMock(
            side_effect=[None, None, RuntimeError('lock contention')],
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-mixed',
        )

        assert stats['scanned'] == 4
        assert stats['candidate_edges'] == 3, (
            f'Expected the terminal, blocked and failing edges all selected and the '
            f'still-blocked healthy edge left alone, got stats={stats!r}'
        )
        assert stats['invalidated'] == 2
        assert stats['errors'] == 1
        assert stats['invalidated'] == stats['candidate_edges'] - stats['errors'], (
            f"COUNTER INVARIANT VIOLATED: stats['invalidated'] must equal "
            f"stats['candidate_edges'] - stats['errors'], got stats={stats!r}. "
            f"Every selected edge is either invalidated or tallied as an error by the "
            f"ONE counted invalidation loop. If this broke because 'errors' was widened "
            f"to cover a NON-invalidation failure mode (e.g. the superseding "
            f"temporal_fact write), that widening is the bug: give the new failure mode "
            f"its own stats key instead, or this invariant stops being able to detect an "
            f"invalidation that went uncounted."
        )

    @pytest.mark.asyncio
    async def test_extraction_runs_exactly_once_per_edge(self):
        """LIVENESS: the sweep must enter the extraction pipeline exactly ONCE
        per edge, even though there are now TWO marker families.

        Extractor cost is a whole-cycle liveness property (see
        status_snapshot_extraction.py's module docstring and
        docs/stale-status-snapshot-sweep.md §"Extraction cost as a liveness
        property"): the extractor runs once per valid edge over the
        whole group's edge set — tens of thousands of edges, current figure
        in plans/stale-status-snapshot-sweep-report.md — with no per-edge
        timeout, so one
        pathological fact stalls the entire reconciliation cycle. A second
        full pass per edge doubles the cost the module's own performance tests
        exist to bound.

        Measured with CountingStr.reads, which tallies extraction-pipeline
        ENTRIES: every entry point opens with ``fact = fact or ''`` and that
        truthiness test is the only Python-level ``__len__`` the pipeline
        makes on the fact. The sibling ``touched`` character counter cannot
        express this property — running two marker families costs the same
        character traversals whether they are reached through one entry point
        or two — so the two counters measure genuinely different things.

        RED before the by-class extractor lands: the sweep precomputes only
        the UNION edge_ids map, so select_stale_status_snapshot_edges falls
        back to extracting blocked ids internally, per edge — a second entry
        the liveness discipline forbids.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()

        facts = [
            CountingStr('Task 2848 remains blocked as of 2026-07-22'),
            CountingStr('Task 142 is an active pending task'),
            CountingStr('Tasks 1020 and 1030 are blocked'),
        ]
        edges = [
            {'uuid': f'edge-{i}', 'fact': fact, 'name': ''}
            for i, fact in enumerate(facts)
        ]
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': edges},
                complete_paged_read(rows_seen=len(edges)),
            ),
        )
        taskmaster.get_statuses = AsyncMock(
            return_value={'2848': 'pending', '142': 'done', '1020': 'blocked',
                          '1030': 'blocked'},
        )

        await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        for fact in facts:
            assert fact.reads == 1, (
                f'Expected the sweep to enter the extraction pipeline exactly ONCE for '
                f'{str(fact)!r}, got {fact.reads}. More than one entry means the union '
                f'and blocked id sets are being extracted in separate passes; they must '
                f'come from a single by-class extraction per edge.'
            )

class TestSweepStaleStatusSnapshotEdgesSupersedeWrite:
    """After retiring a stale blocked assertion the sweep writes ONE
    superseding resulting-state-only temporal_fact per contradicted task per
    cycle (task 3037).

    The invalidation alone leaves the graph with a hole: the old assertion is
    gone and nothing records what replaced it, so the next reader has to
    re-derive the status from the task table. prompts/stage1.py already
    prescribes this invalidate-then-supersede pair for the Stage-1 agent; the
    sweep now performs it deterministically for blocked-status snapshot edges
    instead of leaving the agent to rediscover it per finding.
    """

    BLOCKED_FACT = 'Task 2848 remains blocked as of 2026-07-22'
    NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC)

    def _sweep_kwargs(self) -> dict:
        return {'run_id': 'run-3037', 'now': self.NOW}

    @pytest.mark.asyncio
    async def test_supersede_fact_written_after_successful_invalidation(self):
        """One add_memory per contradicted task, with the full write envelope."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-2848', 'fact': self.BLOCKED_FACT, 'name': ''},
                ]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'pending'})

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            **self._sweep_kwargs(),
        )

        memory_service.add_memory.assert_awaited_once()
        call = memory_service.add_memory.await_args
        content = call.kwargs.get('content', call.args[0] if call.args else None)
        assert content == build_supersede_fact(2848, 'pending', self.NOW), (
            f'Expected the resulting-state-only superseding fact, got {content!r}'
        )
        assert call.kwargs.get('category') == 'temporal_facts', (
            f'Expected the superseding fact routed to temporal_facts, got {call!r}'
        )
        assert call.kwargs.get('project_id') == 'test_project'
        assert call.kwargs.get('agent_id') == 'recon-stage-memory_consolidator', (
            "Expected the write stamped with the sweep's stable agent_id, so its "
            f'writes are attributable in the write journal, got {call!r}'
        )
        assert call.kwargs.get('causation_id') == 'run-3037'

        assert stats['superseded'] == 1
        assert stats['supersede_errors'] == 0

    @pytest.mark.asyncio
    async def test_supersede_write_is_deduped_to_one_per_task_per_cycle(self):
        """Two distinct edges naming the same task -> ONE superseding write.

        A contradicted task typically has several blocked-assertion edges
        (one per phrasing Graphiti extracted). Writing per EDGE would emit
        the identical fact several times in a single cycle and flood the
        graph with duplicates that the consolidation pass then has to clean
        up.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-a', 'fact': self.BLOCKED_FACT, 'name': ''},
                    {'uuid': 'edge-b', 'fact': 'Task 2848 is currently blocked', 'name': ''},
                ]},
                complete_paged_read(rows_seen=2),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'pending'})

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            **self._sweep_kwargs(),
        )

        assert memory_service.update_edge.await_count == 2, (
            'Both edges assert the stale blocked status, so both must be retired'
        )
        assert memory_service.add_memory.await_count == 1, (
            f'Expected exactly ONE superseding fact for task 2848 across the cycle, '
            f'got {memory_service.add_memory.await_count}'
        )
        assert stats['superseded'] == 1

    @pytest.mark.asyncio
    async def test_failed_supersede_write_is_retried_by_a_sibling_edge(self):
        """A transient add_memory failure does not consume the task's one write.

        The per-task dedupe marker is set on SUCCESS, not on attempt. This
        is the ONLY retry the fact can ever get: both edges are invalidated
        by the time the second attempt runs, so neither re-enumerates next
        cycle and there is no later chance. Marking on attempt would forfeit
        it — the sibling edge, already in hand, would be skipped.

        Both counters stay faithful: one write landed (superseded == 1) and
        one failed (supersede_errors == 1).
        (amendment, reviewer_comprehensive robustness finding, task 3037)
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-a', 'fact': self.BLOCKED_FACT, 'name': ''},
                    {'uuid': 'edge-b', 'fact': 'Task 2848 is currently blocked', 'name': ''},
                ]},
                complete_paged_read(rows_seen=2),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'pending'})
        memory_service.add_memory = AsyncMock(
            side_effect=[RuntimeError('transient graphiti write failure'), None],
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            **self._sweep_kwargs(),
        )

        assert memory_service.add_memory.await_count == 2, (
            "The second edge naming task 2848 must re-attempt the write the "
            f'first one failed, got {memory_service.add_memory.await_count} attempt(s)'
        )
        assert stats['superseded'] == 1, (
            f'The retry landed, so the fact is written exactly once, got {stats!r}'
        )
        assert stats['supersede_errors'] == 1, (
            f'The first attempt still counts as a failure, got {stats!r}'
        )
        assert stats['invalidated'] == 2, (
            f'Both invalidations are unaffected by either write, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_per_cycle_write_ceiling_truncates_and_reports_the_remainder(self):
        """More contradicted tasks than _MAX_SUPERSEDE_WRITES_PER_CYCLE ->
        the excess is skipped, counted, and logged — never silently dropped.

        The writes are awaited serially inside the invalidation loop, so an
        unbounded backlog (every blocked assertion in the corpus becomes
        eligible on the first cycle after the rule ships) would add minutes
        of network time to a loop whose enumeration cost this module already
        treats as a liveness property.

        The INVALIDATIONS are deliberately NOT capped: retiring the
        contradicted assertion is the correctness-bearing half of the step,
        and capping it would also break the
        invalidated == candidate_edges - errors identity.
        (amendment, reviewer_comprehensive performance finding, task 3037)
        """
        overflow = 3
        total = _MAX_SUPERSEDE_WRITES_PER_CYCLE + overflow
        task_ids = list(range(9000, 9000 + total))
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {
                        'uuid': f'edge-{task_id}',
                        'fact': f'Task {task_id} remains blocked as of 2026-07-22',
                        'name': '',
                    }
                    for task_id in task_ids
                ]},
                complete_paged_read(rows_seen=total),
            ),
        )
        taskmaster.get_statuses = AsyncMock(
            return_value={str(task_id): 'pending' for task_id in task_ids},
        )
        log = MagicMock()

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            log=log, **self._sweep_kwargs(),
        )

        assert memory_service.add_memory.await_count == _MAX_SUPERSEDE_WRITES_PER_CYCLE, (
            f'The ceiling bounds the number of awaited writes, got '
            f'{memory_service.add_memory.await_count}'
        )
        assert stats['superseded'] == _MAX_SUPERSEDE_WRITES_PER_CYCLE
        assert stats['supersede_skipped'] == overflow, (
            f'Expected the {overflow} tasks past the ceiling counted as skipped, '
            f'got {stats!r}'
        )
        assert stats['invalidated'] == total, (
            f'Every stale edge is still retired — only the superseding WRITE is '
            f'capped, got {stats!r}'
        )
        assert stats['supersede_errors'] == 0, (
            f'A ceiling truncation is not a write failure, got {stats!r}'
        )
        assert log.warning.call_count == 1, (
            'No silent caps: truncating must log exactly one WARNING naming how '
            f'much was dropped, got {log.warning.call_args_list!r}'
        )
        assert overflow in log.warning.call_args.args, (
            f'The WARNING must carry the skipped COUNT, got {log.warning.call_args!r}'
        )

    @pytest.mark.asyncio
    async def test_no_ceiling_warning_when_the_cycle_stays_under_it(self):
        """The common case logs nothing — the warning marks real truncation."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-2848', 'fact': self.BLOCKED_FACT, 'name': ''},
                ]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'pending'})
        log = MagicMock()

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            log=log, **self._sweep_kwargs(),
        )

        assert stats['supersede_skipped'] == 0
        log.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_supersede_write_when_the_invalidation_failed(self):
        """update_edge raised -> no superseding fact for that edge.

        The write is gated on a SUCCESSFUL invalidation. Writing anyway would
        leave the graph asserting both that task 2848 is blocked (the edge is
        still valid) and that it is pending — a self-contradiction, and one
        the sweep would repeat every cycle since the un-invalidated edge
        re-enumerates.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-2848', 'fact': self.BLOCKED_FACT, 'name': ''},
                ]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'pending'})
        memory_service.update_edge = AsyncMock(side_effect=RuntimeError('lock contention'))

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            **self._sweep_kwargs(),
        )

        memory_service.add_memory.assert_not_awaited()
        assert stats['superseded'] == 0
        assert stats['errors'] == 1

    @pytest.mark.asyncio
    async def test_no_supersede_write_for_a_terminal_rule_invalidation(self):
        """A done/cancelled invalidation writes NO superseding fact.

        The supersede write belongs to the BLOCKED rule only. Keeping it so
        is what makes 'superseded' a faithful per-step counter for the new
        deterministic step rather than a second, noisier copy of
        'invalidated' — and a terminal task's status is already recorded by
        the task table and the done-transition reconciliation; re-asserting
        it here would add nothing.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-142', 'fact': 'Task 142 is an active pending task', 'name': ''},
                ]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'142': 'done'})

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            **self._sweep_kwargs(),
        )

        assert stats['invalidated'] == 1
        memory_service.add_memory.assert_not_awaited()
        assert stats['superseded'] == 0

    @pytest.mark.asyncio
    async def test_blocked_task_that_went_terminal_supersedes_on_the_blocked_rule(self):
        """A blocked assertion whose task went DONE is contradicted by BOTH
        rules; it still yields exactly one superseding fact.

        Both rules select the edge, but only one invalidation happens and the
        supersede write keys on the blocked rule's own contradiction test —
        'done' is positively known and is not 'blocked' — so the counter
        stays exactly one per contradicted task.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-2848', 'fact': self.BLOCKED_FACT, 'name': ''},
                ]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'done'})

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            **self._sweep_kwargs(),
        )

        assert stats['invalidated'] == 1
        assert stats['superseded'] == 1
        content = memory_service.add_memory.await_args.kwargs.get('content')
        assert content == build_supersede_fact(2848, 'done', self.NOW)

    @pytest.mark.asyncio
    async def test_stats_gains_superseded_and_supersede_errors_keys(self):
        """Both new keys are present and zero on a run with nothing to do.

        'supersede_errors' is a SEPARATE key rather than a contribution to
        'errors' so the counter invariant
        ``invalidated == candidate_edges - errors`` stays exactly true — that
        invariant is the structural guard for the whole stat requirement, and
        diluting 'errors' with a non-invalidation failure mode would silently
        retire it.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify',
            **self._sweep_kwargs(),
        )

        assert stats['superseded'] == 0
        assert stats['supersede_errors'] == 0

# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — guards
# --------------------------------------------------------------------------- #


class TestSweepStaleStatusSnapshotEdgesGuards:
    """Guard behavior: an unavailable taskmaster/project_root, or an
    enumeration with no candidate ids, must short-circuit without
    unnecessary backend calls."""

    @pytest.mark.asyncio
    async def test_taskmaster_none_yields_all_zero_stats_with_no_calls(self):
        """taskmaster=None -> all-zero stats; enumerate_all_valid_edges never awaited."""
        memory_service = _make_memory_service()

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, None, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats == {
            'scanned': 0, 'candidate_edges': 0,
            'invalidated': 0, 'errors': 0,
            'superseded': 0, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': None, 'enumeration_incomplete_kind': None,
        }
        memory_service.graphiti.enumerate_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_candidate_ids_skips_get_statuses(self):
        """Enumeration returns only count-only/non-status edges (no candidate
        ids) -> get_statuses never awaited and invalidated == 0."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        count_only_edge = {
            'uuid': 'edge-count', 'fact': 'There are 8 tasks in progress', 'name': '',
        }
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [count_only_edge]},
                complete_paged_read(rows_seen=1),
            ),
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
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [stale_edge_1, stale_edge_2]},
                complete_paged_read(rows_seen=2),
            ),
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
        assert stats == {
            'scanned': 2, 'candidate_edges': 2,
            'invalidated': 1, 'errors': 1,
            'superseded': 0, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': True, 'enumeration_incomplete_kind': None,
        }, (
            f'Expected the failed update tallied as an error without blocking the second, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_enumerate_all_valid_edges_failure_yields_all_zero_stats_without_raising(
        self,
    ):
        """enumerate_all_valid_edges raises -> sweep returns all-zero stats (errors
        tallied) without raising and without calling get_statuses/update_edge."""
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            side_effect=RuntimeError('transient read timeout'),
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats == {
            'scanned': 0, 'candidate_edges': 0,
            'invalidated': 0, 'errors': 1,
            'superseded': 0, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': None, 'enumeration_incomplete_kind': None,
        }, (
            f'Expected all-zero stats with the failure tallied as an error, got {stats!r}'
        )
        taskmaster.get_statuses.assert_not_awaited()
        memory_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_structurally_incomplete_enumeration_is_a_best_effort_error(self):
        """A STRUCTURAL incompleteness aborts the cycle as a best-effort error.

        ``enumerate_all_valid_edges`` never raises — it reports incompleteness
        as a value — so the fail-closed guard the ``get_all_valid_edges`` shim
        used to provide has to be re-applied by the sweep itself.  Without it a
        page-capped read would be taken for the whole corpus and every edge the
        missing pages carry would be scanned as absent, which is exactly the
        direction that silently retires nothing and reports a clean cycle.

        The pin is that the truncated rows are NOT scanned (``scanned == 0``,
        no ``update_edge``) and that the abort is tallied (``errors == 1``)
        rather than propagated: the sweep is best-effort, and one truncated
        read must not take the surrounding reconciliation cycle down.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-142', 'fact': 'Task 142 is an active pending task',
                     'name': ''},
                ]},
                incomplete_paged_read(INCOMPLETE_PAGE_CAP, rows_seen=1, expected_rows=9999),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'142': 'done'})

        try:
            stats = await sweep_stale_status_snapshot_edges(
                memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
            )
        except IncompleteEnumerationError as exc:
            pytest.fail(
                'A structurally incomplete enumeration must be tallied as a best-effort '
                f'error, not propagated out of the sweep, got: {exc!r}'
            )

        assert stats == {
            'scanned': 0, 'candidate_edges': 0,
            'invalidated': 0, 'errors': 1,
            'superseded': 0, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': False,
            'enumeration_incomplete_kind': INCOMPLETE_PAGE_CAP,
        }, (
            f'Expected the structural incompleteness to abort the cycle with nothing '
            f'scanned and one error tallied, got {stats!r}'
        )
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
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [candidate_edge]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(side_effect=RuntimeError('transient read timeout'))

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats == {
            'scanned': 1, 'candidate_edges': 0,
            'invalidated': 0, 'errors': 1,
            'superseded': 0, 'supersede_errors': 0,
            'supersede_skipped': 0,
            'enumeration_complete': True, 'enumeration_incomplete_kind': None,
        }, (
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
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [stale_edge]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'142': 'done'})
        memory_service.update_edge = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await sweep_stale_status_snapshot_edges(
                memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
            )

    # ----------------------------------------------------------------- #
    # The superseding write is best-effort too (task 3037)
    #
    # It is the SECOND half of the deterministic step and strictly the
    # less important one: by the time it runs the stale edge is already
    # retired, which is the correctness-bearing effect. A transient
    # add_memory failure must therefore degrade observability only — never
    # abort the remaining edges' invalidations, and never be mistaken for a
    # failed invalidation in the counters.
    # ----------------------------------------------------------------- #

    BLOCKED_FACT_2848 = 'Task 2848 remains blocked as of 2026-07-22'
    BLOCKED_FACT_2849 = 'Task 2849 remains blocked as of 2026-07-22'

    @pytest.mark.asyncio
    async def test_add_memory_failure_is_swallowed_and_second_edge_still_invalidated(self):
        """add_memory raises for the first stale blocked edge; the second stale
        edge is still invalidated AND still counted.

        Mirrors this class's update_edge posture exactly: tally, log, carry
        on. The failure lands in stats['supersede_errors'] — NOT in
        stats['errors'], which stays scoped to the invalidation paths.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-2848', 'fact': self.BLOCKED_FACT_2848, 'name': ''},
                    {'uuid': 'edge-2849', 'fact': self.BLOCKED_FACT_2849, 'name': ''},
                ]},
                complete_paged_read(rows_seen=2),
            ),
        )
        taskmaster.get_statuses = AsyncMock(
            return_value={'2848': 'pending', '2849': 'in-progress'},
        )
        memory_service.add_memory = AsyncMock(
            side_effect=[RuntimeError('embedding backend timeout'), None],
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert memory_service.update_edge.await_count == 2, (
            'Both stale edges must still be invalidated after the first '
            'superseding write fails — the invalidation is the '
            'correctness-bearing half of the step'
        )
        assert memory_service.add_memory.await_count == 2, (
            "The second edge's superseding write must still be attempted "
            'after the first one raised'
        )
        assert stats == {
            'scanned': 2, 'candidate_edges': 2,
            'invalidated': 2, 'errors': 0,
            'superseded': 1, 'supersede_errors': 1,
            'supersede_skipped': 0,
            'enumeration_complete': True, 'enumeration_incomplete_kind': None,
        }, (
            'Expected the failed supersede tallied into supersede_errors only, '
            f'leaving both invalidations intact and errors at 0, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_add_memory_failure_does_not_undo_the_invalidation(self):
        """The already-succeeded update_edge is never rolled back or retried.

        There is no compensating write: the edge stays invalid_at-stamped and
        counted, and the sweep simply returns without its superseding fact.
        Retrying it next cycle is impossible anyway — the invalidated edge no
        longer enumerates — so the failure mode is a permanently missing
        superseding fact, not a corrupted edge.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-2848', 'fact': self.BLOCKED_FACT_2848, 'name': ''},
                ]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'pending'})
        memory_service.add_memory = AsyncMock(side_effect=RuntimeError('write rejected'))

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        memory_service.update_edge.assert_awaited_once()
        assert stats['invalidated'] == 1, (
            'The invalidation that already succeeded must stay counted, got '
            f'{stats!r}'
        )
        assert stats['superseded'] == 0
        assert stats['supersede_errors'] == 1

    @pytest.mark.asyncio
    async def test_counter_invariant_holds_when_only_the_supersede_write_fails(self):
        """`invalidated == candidate_edges - errors` still holds exactly.

        This is the structural guard for the whole stat-counter requirement
        (task 3037): every SELECTED edge is either invalidated or tallied in
        stats['errors'], by the one counted invalidation loop. Folding a
        supersede failure into 'errors' would silently retire that identity,
        so the invariant is re-asserted here on a run whose ONLY failure is a
        superseding write.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-2848', 'fact': self.BLOCKED_FACT_2848, 'name': ''},
                    {'uuid': 'edge-healthy', 'fact': 'Task 3001 is blocked.', 'name': ''},
                ]},
                complete_paged_read(rows_seen=2),
            ),
        )
        taskmaster.get_statuses = AsyncMock(
            return_value={'2848': 'pending', '3001': 'blocked'},
        )
        memory_service.add_memory = AsyncMock(side_effect=RuntimeError('write rejected'))

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats['errors'] == 0, (
            "A failed superseding write is not a failed invalidation — 'errors' "
            f'stays scoped to enumerate/cross-reference/invalidate, got {stats!r}'
        )
        assert stats['supersede_errors'] == 1
        assert stats['invalidated'] == stats['candidate_edges'] - stats['errors'], (
            "stats['errors'] is scoped to the enumerate / cross-reference / "
            'INVALIDATE paths so that every selected edge is either invalidated '
            'or tallied there. If this broke because a non-invalidation failure '
            "mode (e.g. the superseding write) was folded into 'errors', give "
            f'that mode its own key instead — got {stats!r}'
        )

    @pytest.mark.parametrize(
        'exc_type',
        [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
        ids=['cancelled', 'keyboard_interrupt', 'system_exit'],
    )
    @pytest.mark.asyncio
    async def test_control_flow_exceptions_from_add_memory_propagate(self, exc_type):
        """Control-flow exceptions are never swallowed as best-effort errors.

        Same posture the invalidation path already holds: shutdown and
        cancellation must tear the sweep down promptly rather than being
        logged and counted as a transient backend failure.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-2848', 'fact': self.BLOCKED_FACT_2848, 'name': ''},
                ]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'2848': 'pending'})
        memory_service.add_memory = AsyncMock(side_effect=exc_type())

        with pytest.raises(exc_type):
            await sweep_stale_status_snapshot_edges(
                memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
            )

    @pytest.mark.parametrize(
        'exc_type',
        [KeyboardInterrupt, SystemExit],
        ids=['keyboard_interrupt', 'system_exit'],
    )
    @pytest.mark.asyncio
    async def test_control_flow_exceptions_from_update_edge_propagate(self, exc_type):
        """The invalidation path's control-flow posture, beside the new one.

        The CancelledError leg is pinned above by
        test_cancelled_error_from_update_edge_propagates; these two complete
        the trio so both write paths are held to the identical contract.
        """
        memory_service = _make_memory_service()
        taskmaster = _make_taskmaster()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-stale', 'fact': 'Task 142 is an active pending task', 'name': ''},
                ]},
                complete_paged_read(rows_seen=1),
            ),
        )
        taskmaster.get_statuses = AsyncMock(return_value={'142': 'done'})
        memory_service.update_edge = AsyncMock(side_effect=exc_type())

        with pytest.raises(exc_type):
            await sweep_stale_status_snapshot_edges(
                memory_service, taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
            )


# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — enumeration completeness signal
# (task 4386)
# --------------------------------------------------------------------------- #


class TestEnumerationStatKeyValues:
    """Pin the SHIPPED spelling of the exported report.stats key names.

    The producer (memory_consolidator) and the wiring tests now reference
    these through the symbol, which is the point — a rename is one edit
    instead of a grep across five files. But that also means a rename of the
    STRING would slip past every one of those call sites with all of them
    still green, silently renaming the wire format the ledger, the journal
    and the judge read. So the literal is pinned in exactly one place: here.

    Mirrors test_task_count_snapshot_cadence.py's
    ``test_prune_enumeration_ok_stat_key_value`` for the sibling stat family.
    (amendment, reviewer_comprehensive pattern-consistency finding)
    """

    def test_status_snapshot_enumeration_complete_stat_key_value(self):
        assert STATUS_SNAPSHOT_ENUMERATION_COMPLETE_STAT_KEY == (
            'stale_status_snapshot_edges_enumeration_complete'
        )

    def test_status_snapshot_enumeration_incomplete_kind_stat_key_value(self):
        assert STATUS_SNAPSHOT_ENUMERATION_INCOMPLETE_KIND_STAT_KEY == (
            'stale_status_snapshot_edges_enumeration_incomplete_kind'
        )

    def test_the_two_keys_share_the_consolidator_prefix(self):
        """Both keys carry THIS sweep's prefix, so a reader scanning
        report.stats can attribute them to the read that produced them —
        and cannot confuse them with the priority sweep's independent pair."""
        for key in (
            STATUS_SNAPSHOT_ENUMERATION_COMPLETE_STAT_KEY,
            STATUS_SNAPSHOT_ENUMERATION_INCOMPLETE_KIND_STAT_KEY,
        ):
            assert key.startswith('stale_status_snapshot_edges_'), key


class TestSweepStaleStatusSnapshotEdgesEnumerationCompleteness:
    """The sweep reports whether the corpus it swept was the WHOLE corpus.

    Before this, a cycle that swept a truncated read was indistinguishable in
    its stats from one that swept everything: both reported the same counters,
    and the only trace of the truncation was a log line nobody correlates with
    the cycle. That is the ambiguity the two new keys close, and it is the same
    move the supersede counters already make against the same failure — a bare
    ``superseded=0`` cannot distinguish "nothing needed superseding" from
    "every superseding write failed".

    ``enumeration_complete`` is deliberately TRI-STATE, because three outcomes
    are genuinely distinct and collapsing any pair loses information a reader
    needs:

      ``True``  — the read was PROVEN complete. The only value a caller may
                  gate on; see ``test_only_is_true_gates_a_swept_corpus``.
      ``False`` — a corpus was observed and it was INCOMPLETE, with
                  ``enumeration_incomplete_kind`` naming which way.
      ``None``  — NO corpus was observed at all: either the sweep
                  short-circuited before reading, or the read itself failed.
                  Distinguished from each other by ``errors``.

    ``enumeration_incomplete_kind`` carries the backend's stable
    ``INCOMPLETE_*`` discriminator, never ``PagedRead.reason`` — that field is
    diagnostic prose the backend documents as an unstable interface, so
    asserting on it here would pin wording rather than behaviour.
    """

    STALE_FACT = 'Task 142 is an active pending task'

    def _memory_reading(self, paged) -> MagicMock:
        """A memory service whose enumeration returns one stale edge + *paged*."""
        memory_service = _make_memory_service()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            return_value=(
                {'entity-a': [
                    {'uuid': 'edge-142', 'fact': self.STALE_FACT, 'name': ''},
                ]},
                paged,
            ),
        )
        return memory_service

    @staticmethod
    def _taskmaster_saying_done() -> MagicMock:
        taskmaster = _make_taskmaster()
        taskmaster.get_statuses = AsyncMock(return_value={'142': 'done'})
        return taskmaster

    @pytest.mark.asyncio
    async def test_complete_read_reports_complete_true_and_no_kind(self):
        """A proven-complete read: the sweep behaves exactly as it always did."""
        memory_service = self._memory_reading(complete_paged_read(rows_seen=1))

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, self._taskmaster_saying_done(), 'test_project',
            '/tmp/reify', run_id='run-1',
        )

        assert stats['enumeration_complete'] is True, (
            f'Expected a proven-complete read reported as True, got {stats!r}'
        )
        assert stats['enumeration_incomplete_kind'] is None, (
            f'A complete read has no incompleteness kind, got {stats!r}'
        )
        # Unchanged behaviour on the path that was already correct.
        assert stats['scanned'] == 1
        assert stats['invalidated'] == 1
        assert stats['errors'] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'kind', [INCOMPLETE_CENSUS_UNAVAILABLE, INCOMPLETE_SHORT_READ],
        ids=['census_unavailable', 'short_read'],
    )
    async def test_empirical_incompleteness_still_sweeps_but_says_so(self, kind):
        """EMPIRICAL incompleteness -> sweep what was fetched, and report it.

        This is the whole point of the task. The empirical kinds are
        transient-capable on a graph under continuous write, so aborting would
        take the live rebuild down for something that self-heals next cycle —
        the sweep therefore proceeds normally and ``errors`` stays 0. What
        changes is that the cycle now SAYS the corpus was partial instead of
        reporting a clean sweep, so a consumer can decline to treat this
        cycle's absences as evidence.
        """
        memory_service = self._memory_reading(
            incomplete_paged_read(kind, rows_seen=1, expected_rows=40),
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, self._taskmaster_saying_done(), 'test_project',
            '/tmp/reify', run_id='run-1',
        )

        assert stats['enumeration_complete'] is False, (
            f'Expected the partial corpus reported as False, got {stats!r}'
        )
        assert stats['enumeration_incomplete_kind'] == kind, (
            f'Expected the stable {kind!r} discriminator surfaced, got {stats!r}'
        )
        # The sweep still did its job on what it did fetch.
        assert stats['scanned'] == 1
        assert stats['invalidated'] == 1
        assert stats['errors'] == 0, (
            f'An empirical incompleteness is NOT an error — raising on a census '
            f'disagreement would flap on a live graph, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_empirical_incompleteness_warns_through_the_injected_log(self):
        """The one warning about a truncated corpus rides the sweep's logger.

        ``apply_incompleteness_policy`` defaults to the BACKEND module logger,
        which would surface this message detached from the cycle that produced
        it. The sweep passes its own injected ``log`` so the truncation shows
        up beside the rest of that cycle's diagnostics.
        """
        memory_service = self._memory_reading(
            incomplete_paged_read(INCOMPLETE_SHORT_READ, rows_seen=1, expected_rows=40),
        )
        log = MagicMock()

        await sweep_stale_status_snapshot_edges(
            memory_service, self._taskmaster_saying_done(), 'test_project',
            '/tmp/reify', run_id='run-1', log=log,
        )

        assert log.warning.call_count == 1, (
            f'Expected exactly one truncation warning on the sweep\'s own logger, '
            f'got {log.warning.call_args_list!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('kind', sorted(INCOMPLETE_STRUCTURAL_KINDS))
    async def test_structural_incompleteness_reports_the_kind_beside_the_error(
        self, kind,
    ):
        """A STRUCTURAL abort is a STRUCTURED fact, not just a log line.

        The keys are assigned BEFORE the policy is applied, which is what lets
        an aborted cycle still say WHY it aborted. Without that ordering the
        operator gets ``errors=1`` and has to reconstruct the cause from logs —
        exactly the reconstruction this task exists to remove.

        Parametrised over the frozenset rather than the two named constants, so
        a future third structural path is covered by construction (the backend
        asks callers to branch on membership, not on a specific kind).
        """
        memory_service = self._memory_reading(
            incomplete_paged_read(kind, rows_seen=1, expected_rows=9999),
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, self._taskmaster_saying_done(), 'test_project',
            '/tmp/reify', run_id='run-1',
        )

        assert stats['enumeration_complete'] is False, (
            f'Expected the aborted cycle to still report the corpus as partial, '
            f'got {stats!r}'
        )
        assert stats['enumeration_incomplete_kind'] == kind, (
            f'Expected the abort reason carried as a structured fact, got {stats!r}'
        )
        assert stats['errors'] == 1, f'Expected the abort tallied, got {stats!r}'
        assert stats['scanned'] == 0, (
            f'A structurally truncated read must NOT be scanned as if whole, '
            f'got {stats!r}'
        )
        memory_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_short_circuit_observes_no_corpus(self):
        """No taskmaster -> no read happened, so completeness is UNKNOWN.

        ``None``, not ``True``: nothing was proven about the corpus because
        nothing was read. Reporting ``True`` here would let a caller treat a
        cycle that never looked as a cycle that looked and found everything.
        """
        memory_service = _make_memory_service()

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, None, 'test_project', '/tmp/reify', run_id='run-1',
        )

        assert stats['enumeration_complete'] is None, (
            f'Expected UNKNOWN on the short-circuit path, got {stats!r}'
        )
        assert stats['enumeration_incomplete_kind'] is None
        assert stats['errors'] == 0, (
            f'The short-circuit is not an error — that is what distinguishes it '
            f'from a failed read, got {stats!r}'
        )
        memory_service.graphiti.enumerate_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transport_failure_observes_no_corpus(self):
        """The read itself raised -> also UNKNOWN, distinguished by ``errors``.

        Both no-corpus paths report ``None``; ``errors`` is what tells them
        apart, which is why the tri-state does not need a fourth value.
        """
        memory_service = _make_memory_service()
        memory_service.graphiti.enumerate_all_valid_edges = AsyncMock(
            side_effect=RuntimeError('transient read timeout'),
        )

        stats = await sweep_stale_status_snapshot_edges(
            memory_service, _make_taskmaster(), 'test_project', '/tmp/reify',
            run_id='run-1',
        )

        assert stats['enumeration_complete'] is None, (
            f'Expected UNKNOWN when the read never returned, got {stats!r}'
        )
        assert stats['enumeration_incomplete_kind'] is None
        assert stats['errors'] == 1, (
            f'Expected the failed read tallied, which is what distinguishes it '
            f'from the short-circuit, got {stats!r}'
        )

    @pytest.mark.asyncio
    async def test_only_is_true_gates_a_swept_corpus(self):
        """``is True`` is the ONLY admissible gate predicate.

        A truthiness test (``if stats['enumeration_complete']``) happens to
        agree here, but an ``is not False`` or ``!= False`` test would wrongly
        admit the two UNKNOWN paths — so this pins identity, not truthiness,
        and pins that the three states are mutually exclusive.
        """
        taskmaster = self._taskmaster_saying_done()
        complete = await sweep_stale_status_snapshot_edges(
            self._memory_reading(complete_paged_read(rows_seen=1)), taskmaster,
            'test_project', '/tmp/reify', run_id='run-1',
        )
        empirical = await sweep_stale_status_snapshot_edges(
            self._memory_reading(
                incomplete_paged_read(INCOMPLETE_SHORT_READ, rows_seen=1, expected_rows=40),
            ),
            taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )
        structural = await sweep_stale_status_snapshot_edges(
            self._memory_reading(
                incomplete_paged_read(INCOMPLETE_PAGE_CAP, rows_seen=1, expected_rows=9999),
            ),
            taskmaster, 'test_project', '/tmp/reify', run_id='run-1',
        )
        unread = await sweep_stale_status_snapshot_edges(
            _make_memory_service(), None, 'test_project', '/tmp/reify', run_id='run-1',
        )

        gated = [
            name for name, stats in (
                ('complete', complete), ('empirical', empirical),
                ('structural', structural), ('unread', unread),
            )
            if stats['enumeration_complete'] is True
        ]
        assert gated == ['complete'], (
            f'Only a PROVEN-complete read may pass the gate, got {gated!r}'
        )
        assert empirical['enumeration_complete'] is False
        assert structural['enumeration_complete'] is False
        assert unread['enumeration_complete'] is None


# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges_for_task — the per-task sweep (task 4851)
# --------------------------------------------------------------------------- #


def _make_memory_service_mentioning(edges, paged=None) -> MagicMock:
    """A memory service whose NARROWED read returns *edges*.

    The full-corpus read keeps its default stub, so a test can assert it was
    never awaited.
    """
    memory_service = _make_memory_service()
    memory_service.graphiti.enumerate_valid_edges_mentioning = AsyncMock(
        return_value=(
            {'entity-a': edges},
            paged if paged is not None else complete_paged_read(rows_seen=len(edges)),
        ),
    )
    return memory_service


def _edge(uuid: str, fact: str) -> dict:
    return {'uuid': uuid, 'fact': fact, 'name': ''}


class TestSweepStaleStatusSnapshotEdgesForTask:
    """The same rules as the periodic sweep, over a narrowed read. (task 4851)"""

    NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    NOISE_EDGES = [
        _edge('e-1420', 'Task 1420 is pending'),
        _edge('e-noise', 'The 142 retries were pending review'),
    ]

    async def _sweep(self, memory_service, taskmaster, task_id):
        return await sweep_stale_status_snapshot_edges_for_task(
            memory_service, taskmaster, 'test_project', '/tmp/reify', task_id,
            run_id='run-4851', now=self.NOW,
        )

    @staticmethod
    def _taskmaster_saying(statuses: dict[str, str]) -> MagicMock:
        taskmaster = _make_taskmaster()
        taskmaster.get_statuses = AsyncMock(return_value=statuses)
        return taskmaster

    @pytest.mark.asyncio
    async def test_reads_only_the_narrowed_population(self):
        """HOT PATH: one narrowed read, never the whole-graph enumeration."""
        memory_service = _make_memory_service_mentioning(
            [_edge('e-142', 'Task 142 is an active pending task')],
        )

        await self._sweep(memory_service, self._taskmaster_saying({'142': 'done'}), 142)

        memory_service.graphiti.enumerate_valid_edges_mentioning.assert_awaited_once_with(
            '142', group_id='test_project',
        )
        memory_service.graphiti.enumerate_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_substring_superset_is_post_filtered_to_the_task(self):
        """'1420' and a bare '142' match the needle but do not name task 142."""
        memory_service = _make_memory_service_mentioning(
            [_edge('e-142', 'Task 142 is an active pending task'), *self.NOISE_EDGES],
        )
        taskmaster = self._taskmaster_saying({'142': 'done', '1420': 'done'})

        stats = await self._sweep(memory_service, taskmaster, 142)

        memory_service.update_edge.assert_awaited_once()
        assert memory_service.update_edge.await_args is not None
        assert memory_service.update_edge.await_args.args[0] == 'e-142'
        taskmaster.get_statuses.assert_awaited_once_with('/tmp/reify', ids=['142'])
        assert stats['scanned'] == 3
        assert stats['candidate_edges'] == 1
        assert stats['invalidated'] == 1

    @pytest.mark.asyncio
    async def test_an_aggregate_naming_the_task_is_judged_on_all_its_ids(self):
        """Exactly the periodic sweep's judgement: a sibling id may retire it."""
        memory_service = _make_memory_service_mentioning(
            [_edge('e-agg', 'Tasks 142 and 1030 are pending')],
        )
        taskmaster = self._taskmaster_saying({'142': 'pending', '1030': 'done'})

        stats = await self._sweep(memory_service, taskmaster, 142)

        taskmaster.get_statuses.assert_awaited_once_with('/tmp/reify', ids=['142', '1030'])
        assert stats['invalidated'] == 1

    @pytest.mark.asyncio
    async def test_the_blocked_rule_retires_and_supersedes_through_this_path(self):
        memory_service = _make_memory_service_mentioning(
            [_edge('e-2848', 'Task 2848 remains blocked as of 2026-07-22')],
        )

        stats = await self._sweep(
            memory_service, self._taskmaster_saying({'2848': 'deferred'}), 2848,
        )

        assert stats['invalidated'] == 1
        assert stats['superseded'] == 1
        memory_service.add_memory.assert_awaited_once()
        assert memory_service.add_memory.await_args is not None
        assert memory_service.add_memory.await_args.kwargs['content'] == (
            build_supersede_fact(2848, 'deferred', self.NOW)
        )

    @pytest.mark.asyncio
    async def test_an_uncontradicted_live_status_retires_nothing(self):
        memory_service = _make_memory_service_mentioning(
            [_edge('e-142', 'Task 142 is pending')],
        )

        stats = await self._sweep(
            memory_service, self._taskmaster_saying({'142': 'pending'}), 142,
        )

        memory_service.update_edge.assert_not_awaited()
        assert stats['candidate_edges'] == 0
        assert stats['invalidated'] == 0
        assert stats['errors'] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize('noise', [False, True], ids=['no-edges', 'only-noise'])
    async def test_no_edge_naming_the_task_skips_the_status_read(self, noise):
        edges = list(self.NOISE_EDGES) if noise else []
        memory_service = _make_memory_service_mentioning(edges)
        taskmaster = self._taskmaster_saying({'1420': 'done'})

        stats = await self._sweep(memory_service, taskmaster, 142)

        taskmaster.get_statuses.assert_not_awaited()
        memory_service.update_edge.assert_not_awaited()
        assert stats['scanned'] == len(edges)
        for key in ('candidate_edges', 'invalidated', 'errors', 'superseded'):
            assert stats[key] == 0, f'{key} should be 0, got {stats!r}'

    @pytest.mark.asyncio
    async def test_a_failing_narrowed_read_is_swallowed_and_counted(self):
        memory_service = _make_memory_service()
        memory_service.graphiti.enumerate_valid_edges_mentioning = AsyncMock(
            side_effect=RuntimeError('falkordb down'),
        )
        taskmaster = self._taskmaster_saying({'142': 'done'})

        stats = await self._sweep(memory_service, taskmaster, 142)

        assert stats['errors'] == 1
        assert stats['enumeration_complete'] is None
        taskmaster.get_statuses.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('kind', sorted(INCOMPLETE_STRUCTURAL_KINDS))
    async def test_a_structurally_incomplete_narrowed_read_aborts_like_the_periodic_one(
        self, kind,
    ):
        memory_service = _make_memory_service_mentioning(
            [_edge('e-142', 'Task 142 is an active pending task')],
            paged=incomplete_paged_read(kind, rows_seen=1, expected_rows=9999),
        )

        stats = await self._sweep(
            memory_service, self._taskmaster_saying({'142': 'done'}), 142,
        )

        assert stats['errors'] == 1
        assert stats['enumeration_complete'] is False
        assert stats['enumeration_incomplete_kind'] == kind
        memory_service.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_the_same_stats_keys_as_the_periodic_sweep(self):
        periodic = await sweep_stale_status_snapshot_edges(
            _make_memory_service(), _make_taskmaster(), 'test_project', '/tmp/reify',
            run_id='run-4851',
        )
        per_task = await self._sweep(
            _make_memory_service_mentioning([]), _make_taskmaster(), 142,
        )

        assert per_task.keys() == periodic.keys()

    @pytest.mark.asyncio
    async def test_extraction_runs_exactly_once_per_edge_read(self):
        """LIVENESS twin of test_extraction_runs_exactly_once_per_edge.

        The post-filter must reuse the single by-class extraction, so every
        edge READ enters the extraction pipeline exactly once, whether it is
        kept or filtered out.
        """
        facts = [
            CountingStr('Task 142 is an active pending task'),
            CountingStr('Tasks 142 and 1030 are blocked'),
            CountingStr('Task 1420 is pending'),
        ]
        memory_service = _make_memory_service_mentioning(
            [_edge(f'edge-{i}', fact) for i, fact in enumerate(facts)],
        )
        taskmaster = self._taskmaster_saying(
            {'142': 'done', '1030': 'pending', '1420': 'done'},
        )

        await self._sweep(memory_service, taskmaster, 142)

        for fact in facts:
            assert fact.reads == 1, (
                f'Expected exactly one extraction entry for {str(fact)!r}, '
                f'got {fact.reads}'
            )


# --------------------------------------------------------------------------- #
# Overlapping sweeps of one project (task 4851)
# --------------------------------------------------------------------------- #


class _LiveEdgeGraph:
    """Valid edges that stop enumerating once retired, yielding inside every call.

    The yields let two sweeps interleave the way concurrent asyncio tasks do in
    production: both reads can land before either retirement.
    """

    def __init__(self, edges: list[dict]):
        self.valid = {edge['uuid']: edge for edge in edges}

    async def _read(self, keep) -> tuple[dict, object]:
        await asyncio.sleep(0)
        edges = [edge for edge in self.valid.values() if keep(edge['fact'])]
        return {'entity-a': edges}, complete_paged_read(rows_seen=len(edges))

    async def enumerate_all_valid_edges(self, *, group_id):
        return await self._read(lambda fact: True)

    async def enumerate_valid_edges_mentioning(self, substring, *, group_id):
        return await self._read(lambda fact: substring in fact)

    async def retire(self, uuid, **_kwargs):
        await asyncio.sleep(0)
        self.valid.pop(uuid, None)


class TestOverlappingSweepsOfOneProject:
    """One superseding fact per contradicted task, however the sweeps overlap.

    A cancellation cascade starts one targeted sweep per task without awaiting
    any of them, and a full cycle's Stage-1 sweep can start beside them. Each
    overlap below would otherwise read the aggregate edge twice while it is
    still valid and write every task's superseding fact twice.
    """

    NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    AGGREGATE = _edge('e-agg', 'Tasks 4001 and 4002 are blocked')

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'sweep_task_ids',
        [(4001, 4002), (None, 4001), (4001, None)],
        ids=['two-per-task', 'periodic-then-per-task', 'per-task-then-periodic'],
    )
    async def test_each_task_gets_exactly_one_superseding_fact(self, sweep_task_ids):
        graph = _LiveEdgeGraph([self.AGGREGATE])
        memory_service = MagicMock()
        memory_service.graphiti = graph
        memory_service.update_edge = AsyncMock(side_effect=graph.retire)
        memory_service.add_memory = AsyncMock()
        taskmaster = _make_taskmaster()
        taskmaster.get_statuses = AsyncMock(
            return_value={'4001': 'cancelled', '4002': 'cancelled'},
        )

        async def sweep(task_id):
            if task_id is None:
                return await sweep_stale_status_snapshot_edges(
                    memory_service, taskmaster, 'test_project', '/tmp/reify',
                    run_id='run-periodic', now=self.NOW,
                )
            return await sweep_stale_status_snapshot_edges_for_task(
                memory_service, taskmaster, 'test_project', '/tmp/reify', task_id,
                run_id=f'run-{task_id}', now=self.NOW,
            )

        results = await asyncio.gather(*(sweep(task_id) for task_id in sweep_task_ids))

        written = sorted(
            call.kwargs['content'] for call in memory_service.add_memory.await_args_list
        )
        assert written == [
            build_supersede_fact(4001, 'cancelled', self.NOW),
            build_supersede_fact(4002, 'cancelled', self.NOW),
        ]
        assert sum(stats['invalidated'] for stats in results) == 1
        assert sum(stats['superseded'] for stats in results) == 2
