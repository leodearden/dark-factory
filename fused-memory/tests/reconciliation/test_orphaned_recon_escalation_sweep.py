"""Tests for the orphaned-recon-escalation reaper (task 3052).

The defect: ``stage1_stall_detector.maybe_escalate_stalled_gate_backlog``
files a ``reconciliation_stale_gate_backlog`` L1 only while its subject task
is ``status == 'blocked'``
(``stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids``).  Once
the subject goes terminal — ``done``/``cancelled`` — or vanishes from the
task store, the record is moot but NOTHING closes it: the recon harness never
calls ``queue.resolve()`` on its own queue (the A7b invariant stated above
``reconciliation/harness.py::_RECON_DEDUP_CONFIG``), and the orchestrator's
analogous revalidation sweep reads a DIFFERENT queue and gates on
``level == 2`` before its allowlist is consulted, while every recon record is
born at L1.

This module is the DETECTION half: it computes which pending records fall in
the branch the watcher playbook already sanctions ("**Resolve only** when the
underlying task will genuinely stop qualifying for re-selection"), and hands
that computation to the sole closer — the port-8103 watcher session, or an
operator running ``scripts/derive_orphaned_recon_escalations.py --apply``.

Covers here (steps 1/3/5 of the plan):
- REAPABLE_STALE_CATEGORIES / TERMINAL_TASK_STATUSES: the single-owner
  constants naming the reapable population and the terminal statuses.
- escalation_project_id: the single owner of the ``project_id:`` detail-line
  parse (a deliberate INV-2 exception — ``Escalation`` has no such field).
- select_reapable_escalations: pure category/status/level filter.
- classify_orphan: pure ``terminal`` | ``live`` | ``missing`` classifier
  against a ``{id: status}`` census.
"""

from __future__ import annotations

import pytest
from escalation.models import Escalation

from fused_memory.reconciliation.orphaned_recon_escalation_sweep import (
    REAPABLE_STALE_CATEGORIES,
    TERMINAL_TASK_STATUSES,
    classify_orphan,
    escalation_project_id,
    select_reapable_escalations,
)

GATE_BACKLOG = 'reconciliation_stale_gate_backlog'
HUMAN_OPERATOR = 'reconciliation_stale_human_operator'


def make_escalation(
    *,
    task_id: str = '650',
    esc_id: str | None = None,
    category: str = GATE_BACKLOG,
    status: str = 'pending',
    level: int = 1,
    detail: str | None = None,
    project_id: str = 'dark_factory',
) -> Escalation:
    """Build a real ``Escalation`` (never a mock) in the shape Stage 1 files.

    Real dataclass instances are used throughout this file so that a field
    rename or default change in ``escalation.models.Escalation`` breaks these
    tests instead of silently passing against a mock's ``__getattr__``.
    """
    if detail is None:
        detail = '\n'.join([
            f'project_id: {project_id}',
            'run_id: 62e9b073-a070-47dc-b179-03608db93bef',
            f'task_id: {task_id}',
            'gate_escalated_at: 2026-08-18T21:34:32.310663+00:00',
            'age_hours_at_filing: 48.6',
            'title: Human decision gate: something',
        ])
    return Escalation(
        id=esc_id or f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='reconciliation-stage1',
        severity='blocking',
        category=category,
        summary=f'Gate task {task_id} has awaited a human decision',
        detail=detail,
        status=status,
        level=level,
    )


class TestSingleOwnerConstants:
    """The two constants that define the reapable population.

    Both are single-owner module constants rather than inline literals: the
    sweep, the operator script and the watcher SKILL all read the same
    frozensets, so a divergence cannot make the in-cycle detection and the
    one-shot reap disagree about what qualifies.
    """

    def test_reapable_categories_are_the_two_recon_stale_families(self):
        """Both ``reconciliation_stale_*`` families are reapable, and only those.

        ``reconciliation_stale_human_operator`` is included deliberately (not
        by accident of a wildcard): it has the identical lifecycle — a pending
        L1 filed per-subject-task by Stage 1 whose premise a terminal subject
        moots — and the watcher SKILL already treats the two rows identically.
        """
        assert frozenset({GATE_BACKLOG, HUMAN_OPERATOR}) == REAPABLE_STALE_CATEGORIES, (
            'reapable category set must name exactly the two recon stale '
            f'families; got {sorted(REAPABLE_STALE_CATEGORIES)!r}'
        )

    def test_terminal_statuses_are_done_and_cancelled(self):
        """Only ``done``/``cancelled`` are terminal — ``deferred``/``blocked`` are not.

        ``deferred`` is deliberately excluded: a deferred task can return to
        ``blocked`` and re-qualify for selection, so reaping its record would
        re-arm the filing rule and reproduce the measured re-file churn.
        """
        assert frozenset({'done', 'cancelled'}) == TERMINAL_TASK_STATUSES, (
            'terminal statuses must be exactly done/cancelled; got '
            f'{sorted(TERMINAL_TASK_STATUSES)!r}'
        )


class TestEscalationProjectId:
    """``escalation_project_id`` owns the ``project_id:`` detail-line parse.

    ``Escalation`` carries NO ``project_id`` field, so there is no structured
    fact to read — the value must be recovered from the detail block both
    producers write.  This is the one place in the tree that does it.
    """

    def test_parses_the_current_gate_backlog_filing_format(self):
        """The shape ``maybe_escalate_stalled_gate_backlog`` writes today."""
        esc = make_escalation(project_id='dark_factory')

        assert escalation_project_id(esc) == 'dark_factory'

    def test_parses_the_older_live_vintage_using_age_hours(self):
        """The 2026-08-01 vintage (live record ``esc-5943-1``) uses ``age_hours:``.

        The older key spelling must not matter: the parse keys on the
        ``project_id:`` line alone, which both vintages write first.
        """
        esc = make_escalation(
            task_id='5943',
            esc_id='esc-5943-1',
            detail='\n'.join([
                'project_id: reify',
                'run_id: 62e9b073-a070-47dc-b179-03608db93bef',
                'task_id: 5943',
                'gate_escalated_at: 2026-08-01T18:18:50.294218+00:00',
                'age_hours: 87.8',
                "title: Human curator gate: correct stale clause in Mem0 memory",
            ]),
        )

        assert escalation_project_id(esc) == 'reify'

    def test_strips_whitespace_around_the_value(self):
        """Leading/trailing whitespace on the line and value is stripped."""
        esc = make_escalation(detail='   project_id:   solar_challenge   \nrun_id: x')

        assert escalation_project_id(esc) == 'solar_challenge'

    def test_finds_the_line_even_when_it_is_not_first(self):
        """Position is not load-bearing — only that the line exists."""
        esc = make_escalation(
            detail='run_id: abc\ntask_id: 7\nproject_id: know_live\ntitle: t',
        )

        assert escalation_project_id(esc) == 'know_live'

    def test_empty_detail_returns_none(self):
        """No detail is 'unresolvable', not a guessed project."""
        assert escalation_project_id(make_escalation(detail='')) is None

    def test_detail_without_a_project_id_line_returns_none(self):
        """A detail block that never names a project yields None."""
        esc = make_escalation(detail='run_id: abc\ntask_id: 7\ntitle: t')

        assert escalation_project_id(esc) is None

    def test_value_containing_a_colon_is_returned_whole(self):
        """Split on the FIRST colon only, so a colon in the value survives.

        No live project id contains a colon today, but splitting on every
        colon would silently truncate one that did — and a truncated id would
        miss ``known_projects`` and be counted ``unresolvable``, which is at
        least fail-safe but would be an avoidable recall loss.
        """
        esc = make_escalation(detail='project_id: weird:name\nrun_id: x')

        assert escalation_project_id(esc) == 'weird:name'

    def test_non_str_detail_returns_none_and_never_raises(self):
        """A ``None``/non-str detail is None, not a TypeError.

        ``detail`` is typed ``str`` but is deserialised from JSON on disk, so
        a malformed record must degrade to 'unresolvable' rather than aborting
        the sweep for every other record.
        """
        esc = make_escalation()
        esc.detail = None  # type: ignore[assignment]
        assert escalation_project_id(esc) is None

        esc.detail = 12345  # type: ignore[assignment]
        assert escalation_project_id(esc) is None

    def test_object_without_a_detail_attribute_returns_none(self):
        """A non-``Escalation`` element never raises AttributeError."""
        assert escalation_project_id(object()) is None
        assert escalation_project_id(None) is None


class TestSelectReapableEscalations:
    """``select_reapable_escalations`` is the category/status/level filter.

    All three conditions must hold.  Widening any of them would hand the sole
    closer records it has no sanction to close.
    """

    def test_keeps_pending_l1_records_of_both_reapable_categories(self):
        """Input order is preserved so downstream reporting is stable."""
        gate = make_escalation(task_id='1')
        hor = make_escalation(task_id='2', category=HUMAN_OPERATOR)

        assert select_reapable_escalations([gate, hor]) == [gate, hor]

    def test_rejects_a_non_reapable_category(self):
        """``recon_integrity_issue`` has a different lifecycle and is not reaped."""
        other = make_escalation(task_id='3', category='recon_integrity_issue')

        assert select_reapable_escalations([other]) == []

    def test_rejects_an_already_resolved_record(self):
        """A closed record is not pending and must never be re-closed."""
        resolved = make_escalation(task_id='4', status='resolved')

        assert select_reapable_escalations([resolved]) == []

    def test_rejects_a_level_2_record(self):
        """L2 records belong to the orchestrator's own revalidation sweep.

        Recon-filed gate escalations are born at L1; an L2 record with this
        category would have been promoted by a human-facing path, so this
        reaper stays out of it.
        """
        promoted = make_escalation(task_id='5', level=2)

        assert select_reapable_escalations([promoted]) == []

    def test_skips_a_non_escalation_element_without_raising(self):
        """A malformed element is skipped, not fatal to the whole selection."""
        good = make_escalation(task_id='6')

        assert select_reapable_escalations([object(), None, good, 'nope']) == [good]

    def test_empty_input_returns_empty_list(self):
        assert select_reapable_escalations([]) == []


class TestClassifyOrphan:
    """``classify_orphan(esc, statuses)`` -> ``'terminal' | 'live' | 'missing'``.

    The classifier is pure so the in-cycle sweep and the operator script share
    exactly one owner of the derivation rule; a second copy could drift and
    make the two disagree about which records are safe to close.
    """

    @pytest.mark.parametrize('status', ['done', 'cancelled'])
    def test_terminal_statuses_classify_terminal(self, status):
        """A ``done``/``cancelled`` subject can never re-qualify for selection."""
        esc = make_escalation(task_id='650')

        assert classify_orphan(esc, {'650': status}) == 'terminal'

    @pytest.mark.parametrize(
        'status', ['blocked', 'pending', 'in-progress', 'review', 'deferred'],
    )
    def test_non_terminal_statuses_classify_live(self, status):
        """Anything else is LIVE and must never be handed to the closer.

        ``blocked`` is the load-bearing case: it is exactly the state that
        re-qualifies the subject for re-selection, so resolving its record
        re-arms the filing rule (measured churn esc-650-1 -> esc-650-2 in ~4h).
        """
        esc = make_escalation(task_id='650')

        assert classify_orphan(esc, {'650': status}) == 'live'

    def test_absent_id_classifies_missing(self):
        """No row in the census (in any tag) is 'missing'."""
        esc = make_escalation(task_id='650')

        assert classify_orphan(esc, {'999': 'blocked'}) == 'missing'

    def test_empty_census_classifies_every_record_missing(self):
        """An empty map yields 'missing' — callers must never pass an errored census.

        The sweep guards this by refusing to classify at all when a census
        read failed; the classifier itself stays pure and total.
        """
        for tid in ('650', '5943', '101'):
            assert classify_orphan(make_escalation(task_id=tid), {}) == 'missing'

    def test_id_lookup_is_str_coerced_on_both_sides(self):
        """An int-typed id on either side must match its str spelling.

        Census maps come back ``{id_str: status_str}`` but task ids arrive as
        ints in some code paths; an un-coerced lookup would silently classify
        a ``done`` subject as ``missing``.  Both reap, but the EVIDENCE handed
        to the closer would be false.
        """
        esc = make_escalation(task_id='650')
        assert classify_orphan(esc, {'650': 'done'}) == 'terminal'

        int_keyed = make_escalation(task_id='650')
        int_keyed.task_id = 650  # type: ignore[assignment]
        assert classify_orphan(int_keyed, {'650': 'done'}) == 'terminal'
        assert classify_orphan(esc, {650: 'done'}) == 'terminal'  # type: ignore[dict-item]
