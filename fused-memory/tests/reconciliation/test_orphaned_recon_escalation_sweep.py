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

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.cli_stage_runner import FINDING_ITEM_SCHEMA
from fused_memory.reconciliation.orphaned_recon_escalation_sweep import (
    ORPHANED_ESCALATION_FLAG_CATEGORY,
    ORPHANED_ESCALATION_FLAG_TYPE,
    REAPABLE_STALE_CATEGORIES,
    TERMINAL_TASK_STATUSES,
    build_orphaned_escalation_flag,
    classify_orphan,
    escalation_project_id,
    select_reapable_escalations,
    sweep_orphaned_recon_escalations,
)
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator

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


def assert_conforms_to_finding_schema(flag: dict) -> None:
    """Assert *flag* satisfies ``FINDING_ITEM_SCHEMA`` (cli_stage_runner.py).

    Derived from the live schema object rather than restating its rules, so a
    future schema change (a new required key, a narrowed enum) breaks this
    helper instead of leaving it asserting a stale contract.  ``jsonschema``
    is only a transitive dependency of this package, so the check is written
    against the schema dict directly rather than importing a validator.
    """
    props = FINDING_ITEM_SCHEMA['properties']
    for required in FINDING_ITEM_SCHEMA['required']:
        assert required in flag, (
            f'{required!r} is required by FINDING_ITEM_SCHEMA but missing from the flag'
        )
    py_types = {'string': str, 'boolean': bool, 'array': list, 'object': dict}
    for key, value in flag.items():
        assert key in props, f'{key!r} is not a FINDING_ITEM_SCHEMA property'
        declared = props[key]['type']
        allowed = declared if isinstance(declared, list) else [declared]
        assert any(
            value is None if t == 'null' else isinstance(value, py_types[t])
            for t in allowed
        ), f'{key!r} = {value!r} does not match declared type {declared!r}'
        if 'enum' in props[key]:
            assert value in props[key]['enum'], (
                f'{key!r} = {value!r} is not in the schema enum {props[key]["enum"]!r}'
            )


class TestBuildOrphanedEscalationFlag:
    """``build_orphaned_escalation_flag`` emits the Stage-1 flag for one orphan.

    Stage 1 cannot close the record (the A7b invariant reserves that for the
    port-8103 watcher session), so the flag has to carry enough evidence for
    the closer to re-derive the finding independently, and has to route to the
    watcher playbook's existing "Resolve only when..." branch rather than its
    PARK default.
    """

    def test_conforms_to_the_finding_item_schema(self):
        """Every emitted key is a schema property of the declared type."""
        flag = build_orphaned_escalation_flag(
            make_escalation(task_id='650'),
            'terminal',
            subject_project_id='dark_factory',
            subject_status='done',
        )

        assert_conforms_to_finding_schema(flag)

    def test_pins_the_flag_type_and_category_dedup_key(self):
        """flag_type/category are the module constants, so the dedup key is stable.

        ``flag_dedup.compute_flag_signature`` keys on
        ``(task_id, flag_type, category)``; if either constant drifted between
        cycles the same orphan would re-emit as a brand-new finding every
        cycle, with no ``stage1_flag_marker`` recurrence row and no way for an
        operator to suppress it.
        """
        flag = build_orphaned_escalation_flag(
            make_escalation(task_id='650'),
            'terminal',
            subject_project_id='dark_factory',
            subject_status='cancelled',
        )

        assert flag['flag_type'] == ORPHANED_ESCALATION_FLAG_TYPE
        assert flag['category'] == ORPHANED_ESCALATION_FLAG_CATEGORY
        assert flag['category'] == 'cross_store_inconsistency', (
            'category names a store-vs-store disagreement; task_memory_mismatch '
            'would falsely imply a memory is involved'
        )

    def test_task_id_is_the_subject_task_not_the_escalation_id(self):
        """The recurrence row is keyed on the SUBJECT task, str-coerced.

        Keying on the escalation id instead would make each re-file of the
        same subject look like a distinct finding, defeating dedup entirely.
        """
        esc = make_escalation(task_id='650', esc_id='esc-650-1')
        flag = build_orphaned_escalation_flag(
            esc, 'terminal', subject_project_id='dark_factory', subject_status='done',
        )

        assert flag['task_id'] == '650' and isinstance(flag['task_id'], str)
        assert flag['task_id'] != esc.id

    def test_int_subject_task_id_is_coerced_to_str(self):
        """An int-typed task_id must not produce a non-str dedup key."""
        esc = make_escalation(task_id='650')
        esc.task_id = 650  # type: ignore[assignment]
        flag = build_orphaned_escalation_flag(
            esc, 'terminal', subject_project_id='dark_factory', subject_status='done',
        )

        assert flag['task_id'] == '650' and isinstance(flag['task_id'], str)

    def test_terminal_description_names_id_category_project_and_status(self):
        """A closer can re-derive the finding from the description alone."""
        esc = make_escalation(task_id='650', esc_id='esc-650-1')
        flag = build_orphaned_escalation_flag(
            esc, 'terminal', subject_project_id='dark_factory', subject_status='done',
        )

        assert 'esc-650-1' in flag['description']
        assert GATE_BACKLOG in flag['description']
        assert 'dark_factory' in flag['description']
        assert 'done' in flag['description']

    def test_missing_description_says_no_row_in_that_projects_task_store(self):
        """The 'missing' branch states the absence explicitly and names the project.

        This is the branch whose evidence is easiest to get wrong: 'absent'
        is only meaningful relative to a specific, cross-tag-complete census,
        so the description names the project whose store was read rather than
        implying the task does not exist anywhere.
        """
        esc = make_escalation(task_id='5943', esc_id='esc-5943-1', project_id='reify')
        flag = build_orphaned_escalation_flag(
            esc, 'missing', subject_project_id='reify', subject_status=None,
        )

        assert 'esc-5943-1' in flag['description']
        assert "no row in reify's task store" in flag['description']

    def test_description_never_claims_the_record_is_or_will_be_closed(self):
        """State only what was OBSERVED — the close is the watcher's action.

        The record is still pending when this flag is written, and this stage
        has no authority to close it, so an over-claiming description would
        put a false statement into the recurrence ledger.
        """
        for classification, status in (('terminal', 'done'), ('missing', None)):
            flag = build_orphaned_escalation_flag(
                make_escalation(task_id='650'),
                classification,
                subject_project_id='dark_factory',
                subject_status=status,
            )
            lowered = flag['description'].lower()
            for claim in (
                'has been closed', 'was closed', 'will be closed',
                'has been resolved', 'was resolved', 'auto-closed',
            ):
                assert claim not in lowered, (
                    f'description must not claim closure; found {claim!r} in '
                    f'{flag["description"]!r}'
                )

    def test_suggested_action_routes_to_the_watcher_resolve_branch(self):
        """Names the sole closer, the resolution_class, and the no-churn argument.

        Without the no-churn argument the watcher's PARK default is the
        correct read of its own playbook, so the flag would be ignored; the
        argument is what moves this record into the row's existing
        "Resolve only when the underlying task will genuinely stop qualifying
        for re-selection" branch.
        """
        flag = build_orphaned_escalation_flag(
            make_escalation(task_id='650'),
            'terminal',
            subject_project_id='dark_factory',
            subject_status='done',
        )
        action = flag['suggested_action']

        assert '8103' in action, 'must name the port-8103 watcher as the closer'
        assert "resolution_class='moot-terminal-subject'" in action
        assert "'blocked'" in action, (
            'must carry the no-churn argument: selection requires blocked'
        )
        assert 're-file' in action or 're-select' in action

    def test_suggested_action_offers_a_verify_or_dismiss_branch(self):
        """A closer must be told what to do if the subject is not actually terminal."""
        flag = build_orphaned_escalation_flag(
            make_escalation(task_id='650'),
            'missing',
            subject_project_id='dark_factory',
            subject_status=None,
        )

        assert 'dismiss' in flag['suggested_action'].lower()

    def test_human_operator_category_is_named_in_the_description(self):
        """The sibling category routes the watcher to its OWN playbook row."""
        esc = make_escalation(task_id='777', esc_id='esc-777-1', category=HUMAN_OPERATOR)
        flag = build_orphaned_escalation_flag(
            esc, 'terminal', subject_project_id='dark_factory', subject_status='cancelled',
        )

        assert HUMAN_OPERATOR in flag['description']
        assert GATE_BACKLOG not in flag['description']

    @pytest.mark.parametrize('classification', ['live', 'unresolvable', '', None, 'terminal '])
    def test_non_flaggable_classification_raises_value_error(self, classification):
        """Only 'terminal'/'missing' are flaggable — a wiring mistake must be loud.

        ``'live'`` is the case that matters: handing a still-``blocked``
        subject's record to the sole closer re-arms the filing rule and
        reproduces the measured re-file churn.  Raising here means such a bug
        can never reach the watcher silently.
        """
        with pytest.raises(ValueError):
            build_orphaned_escalation_flag(
                make_escalation(task_id='650'),
                classification,
                subject_project_id='dark_factory',
                subject_status='blocked',
            )


DARK_ROOT = '/srv/dark-factory'
REIFY_ROOT = '/srv/reify'
KNOWN_PROJECTS = {'dark_factory': DARK_ROOT, 'reify': REIFY_ROOT}


def make_queue(pending, *, get_pending_error: BaseException | None = None):
    """An ``EscalationQueue`` double whose ``get_pending`` is SYNC, as the real one is."""
    queue = MagicMock(spec=EscalationQueue)
    if get_pending_error is not None:
        queue.get_pending.side_effect = get_pending_error
    else:
        queue.get_pending.return_value = list(pending)
    return queue


def make_taskmaster(
    censuses_by_root,
    *,
    tags_by_root=None,
    list_tags_error_roots=None,
    statuses_error_roots=None,
):
    """A ``TaskBackendProtocol`` double with per-root, per-tag censuses.

    Args:
        censuses_by_root: ``{project_root: {tag: {task_id: status}}}``.
        tags_by_root: optional ``{project_root: [tag, ...]}`` overriding the
            tag list derived from *censuses_by_root* (used to pin the
            empty-tag-list fallback).
        list_tags_error_roots: roots whose ``list_tags`` raises.
        statuses_error_roots: roots whose ``get_statuses_fresh`` raises.
    """
    list_tags_error_roots = set(list_tags_error_roots or ())
    statuses_error_roots = set(statuses_error_roots or ())

    async def _list_tags(project_root):
        if project_root in list_tags_error_roots:
            raise RuntimeError(f'list_tags exploded for {project_root}')
        if tags_by_root is not None:
            return list(tags_by_root.get(project_root, []))
        return list(censuses_by_root.get(project_root, {}))

    async def _get_statuses_fresh(project_root, ids=None, tag=None):
        if project_root in statuses_error_roots:
            raise RuntimeError(f'get_statuses_fresh exploded for {project_root}')
        by_tag = censuses_by_root.get(project_root, {})
        if tag is None:
            merged: dict[str, str] = {}
            for per_tag in by_tag.values():
                merged.update(per_tag)
            return merged
        return dict(by_tag.get(tag, {}))

    taskmaster = MagicMock(spec=TaskBackendProtocol)
    taskmaster.list_tags = AsyncMock(side_effect=_list_tags)
    taskmaster.get_statuses_fresh = AsyncMock(side_effect=_get_statuses_fresh)
    return taskmaster


class TestSweepOrphanedReconEscalations:
    """The best-effort async orchestrator over the pending recon queue.

    Detection only: it never calls ``queue.resolve()`` — the A7b contract
    reserves closure for the port-8103 watcher session.
    """

    @pytest.mark.asyncio
    async def test_terminal_subjects_produce_one_flag_each(self):
        """A ``done`` and a ``cancelled`` subject each yield exactly one flag."""
        done_rec = make_escalation(task_id='650', esc_id='esc-650-1')
        cancelled_rec = make_escalation(task_id='651', esc_id='esc-651-1')
        queue = make_queue([done_rec, cancelled_rec])
        taskmaster = make_taskmaster(
            {DARK_ROOT: {'master': {'650': 'done', '651': 'cancelled'}}},
        )

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert len(stats['flags']) == 2
        assert stats['scanned'] == 2
        assert stats['terminal'] == 2
        assert stats['missing'] == 0
        assert stats['live'] == 0
        assert stats['unresolvable'] == 0
        assert stats['errors'] == 0
        assert {f['task_id'] for f in stats['flags']} == {'650', '651'}

    @pytest.mark.asyncio
    async def test_blocked_subject_is_never_flagged(self):
        """THE ANTI-CHURN INVARIANT — the single most important case in this file.

        A still-``blocked`` subject re-qualifies for re-selection, so closing
        its record re-arms the filing rule and produces the measured re-file
        churn (``esc-650-1`` -> ``esc-650-2`` in ~4h).  It must be counted
        ``live`` and never reach the closer.
        """
        queue = make_queue([make_escalation(task_id='650')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'blocked'}}})

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['flags'] == []
        assert stats['live'] == 1
        assert stats['terminal'] == 0
        assert stats['missing'] == 0
        assert stats['errors'] == 0

    @pytest.mark.asyncio
    async def test_subject_absent_from_every_tag_produces_a_missing_flag(self):
        """No row anywhere in its own project's store is a reapable orphan."""
        queue = make_queue([make_escalation(task_id='650', esc_id='esc-650-1')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'999': 'blocked'}}})

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['missing'] == 1
        assert len(stats['flags']) == 1
        assert "no row in dark_factory's task store" in stats['flags'][0]['description']

    @pytest.mark.asyncio
    async def test_census_is_cross_tag_complete(self):
        """A subject blocked in a NON-default tag classifies live, not missing.

        ``get_statuses_fresh`` defaults to a single tag
        (``backends/task_backend_protocol.py::list_tags``), so a single
        untagged read would report this subject as having "no row in the task
        store" and drive an irreversible reap of a live record.  This is the
        assertion that prevents that false positive.
        """
        blocked_elsewhere = make_escalation(task_id='777', esc_id='esc-777-1')
        done_in_master = make_escalation(task_id='650', esc_id='esc-650-1')
        queue = make_queue([blocked_elsewhere, done_in_master])
        taskmaster = make_taskmaster({
            DARK_ROOT: {
                'master': {'650': 'done'},
                'feature-x': {'777': 'blocked'},
            },
        })

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['live'] == 1, 'the non-default-tag subject must be live'
        assert stats['missing'] == 0, 'a live subject must never read as missing'
        assert stats['terminal'] == 1
        assert {f['task_id'] for f in stats['flags']} == {'650'}, (
            'only the genuinely terminal subject may be flagged'
        )

        taskmaster.list_tags.assert_awaited_once_with(DARK_ROOT)
        calls = taskmaster.get_statuses_fresh.await_args_list
        assert len(calls) == 2, 'one census read per tag'
        assert {c.kwargs['tag'] for c in calls} == {'master', 'feature-x'}
        for call in calls:
            assert call.args == (DARK_ROOT,)
            assert 'ids' not in call.kwargs, (
                "ids= must never be passed — 'absent from the map' has to be an "
                "unambiguous no-row signal, not a backend's missing-id convention"
            )

    @pytest.mark.asyncio
    async def test_empty_tag_list_falls_back_to_one_untagged_read(self):
        """A backend reporting no tags still gets exactly one census read.

        The fallback keeps the sweep working against a backend whose
        ``list_tags`` is a stub, without ever treating an empty list as "no
        tasks exist" (which would classify every subject ``missing``).
        """
        queue = make_queue([make_escalation(task_id='650')])
        taskmaster = make_taskmaster(
            {DARK_ROOT: {'master': {'650': 'done'}}}, tags_by_root={DARK_ROOT: []},
        )

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['terminal'] == 1
        taskmaster.get_statuses_fresh.assert_awaited_once_with(DARK_ROOT)

    @pytest.mark.asyncio
    async def test_each_record_is_classified_against_its_own_project(self):
        """Cross-project scoping: project B's blocked subject is not reaped.

        Classifying a foreign record against the querying project's census is
        exactly the conflation that turns a live record into a reap
        instruction; each record is checked against ITS OWN project's store.
        """
        dark_done = make_escalation(task_id='650', project_id='dark_factory')
        reify_blocked = make_escalation(task_id='5943', project_id='reify')
        reify_missing = make_escalation(task_id='5944', project_id='reify')
        queue = make_queue([dark_done, reify_blocked, reify_missing])
        taskmaster = make_taskmaster({
            DARK_ROOT: {'master': {'650': 'done'}},
            REIFY_ROOT: {'master': {'5943': 'blocked'}},
        })

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['terminal'] == 1
        assert stats['live'] == 1, "reify's blocked subject must stay live"
        assert stats['missing'] == 1
        assert {c.args[0] for c in taskmaster.list_tags.await_args_list} == {
            DARK_ROOT, REIFY_ROOT,
        }
        flagged = {f['task_id'] for f in stats['flags']}
        assert flagged == {'650', '5944'}
        assert '5943' not in flagged, (
            "task 650 being done in dark_factory says nothing about reify's 5943"
        )

    @pytest.mark.asyncio
    async def test_census_is_fetched_at_most_once_per_project(self):
        """Backend calls must not scale with record count.

        The live queue holds 124 records across seven projects; a per-record
        census read would be ~124 round trips per cycle instead of seven.
        """
        records = [make_escalation(task_id=str(i)) for i in range(20)]
        queue = make_queue(records)
        taskmaster = make_taskmaster(
            {DARK_ROOT: {'master': {str(i): 'blocked' for i in range(20)}}},
        )

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['scanned'] == 20
        assert taskmaster.list_tags.await_count == 1
        assert taskmaster.get_statuses_fresh.await_count == 1

    @pytest.mark.asyncio
    async def test_unparseable_project_id_is_unresolvable_not_missing(self):
        """A record with no ``project_id:`` line is never claimed as an orphan.

        Folding it into ``missing`` would silently reap records whose subject
        was never checked at all.
        """
        queue = make_queue([make_escalation(task_id='650', detail='run_id: abc')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {}}})

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['unresolvable'] == 1
        assert stats['missing'] == 0
        assert stats['flags'] == []
        taskmaster.list_tags.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_project_absent_from_known_projects_is_unresolvable(self):
        """A registry gap is surfaced as its own bucket, not silent recall loss."""
        queue = make_queue([make_escalation(task_id='650', project_id='pump_web_ui')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {}}})

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['unresolvable'] == 1
        assert stats['missing'] == 0
        assert stats['flags'] == []

    @pytest.mark.asyncio
    async def test_get_pending_failure_returns_all_zero_stats(self):
        """A queue read failure degrades to a no-op cycle, never a partial one."""
        queue = make_queue([], get_pending_error=OSError('queue dir vanished'))
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'done'}}})

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats == {
            'flags': [], 'scanned': 0, 'terminal': 0, 'missing': 0,
            'live': 0, 'unresolvable': 0, 'errors': 1,
        }

    @pytest.mark.asyncio
    async def test_census_failure_is_fail_safe_and_scoped_to_that_project(self):
        """An errored census NEVER yields terminal/missing, and never blocks siblings.

        The asymmetry is load-bearing: a false ``terminal`` tells the sole
        closer to resolve a live record, whereas a missed detection is
        re-checked next cycle.
        """
        broken = make_escalation(task_id='5943', project_id='reify')
        healthy = make_escalation(task_id='650', project_id='dark_factory')
        queue = make_queue([broken, healthy])
        taskmaster = make_taskmaster(
            {DARK_ROOT: {'master': {'650': 'done'}}, REIFY_ROOT: {'master': {}}},
            statuses_error_roots={REIFY_ROOT},
        )

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['errors'] == 1
        assert stats['missing'] == 0, 'an errored census is not evidence of absence'
        assert stats['terminal'] == 1, "the healthy project is still classified"
        assert {f['task_id'] for f in stats['flags']} == {'650'}

    @pytest.mark.asyncio
    async def test_list_tags_failure_is_fail_safe(self):
        """A ``list_tags`` failure is an ERROR, not a fall back to one tag.

        Falling back would read only the default tag, which is precisely the
        single-tag read whose false ``missing`` this design exists to prevent.
        """
        queue = make_queue([make_escalation(task_id='650')])
        taskmaster = make_taskmaster(
            {DARK_ROOT: {'master': {}}}, list_tags_error_roots={DARK_ROOT},
        )

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['errors'] == 1
        assert stats['missing'] == 0
        assert stats['terminal'] == 0
        assert stats['flags'] == []
        taskmaster.get_statuses_fresh.assert_not_awaited()

    @pytest.mark.parametrize('exc', [asyncio.CancelledError, KeyboardInterrupt])
    @pytest.mark.asyncio
    async def test_cancellation_propagates_from_get_pending(self, exc):
        """Cancellation is never swallowed as a best-effort error."""
        queue = make_queue([], get_pending_error=exc())
        taskmaster = make_taskmaster({})

        with pytest.raises(exc):
            await sweep_orphaned_recon_escalations(queue, taskmaster, KNOWN_PROJECTS)

    @pytest.mark.parametrize('exc', [asyncio.CancelledError, KeyboardInterrupt])
    @pytest.mark.asyncio
    async def test_cancellation_propagates_from_list_tags(self, exc):
        queue = make_queue([make_escalation(task_id='650')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {}}})
        taskmaster.list_tags = AsyncMock(side_effect=exc())

        with pytest.raises(exc):
            await sweep_orphaned_recon_escalations(queue, taskmaster, KNOWN_PROJECTS)

    @pytest.mark.parametrize('exc', [asyncio.CancelledError, KeyboardInterrupt])
    @pytest.mark.asyncio
    async def test_cancellation_propagates_from_get_statuses_fresh(self, exc):
        queue = make_queue([make_escalation(task_id='650')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {}}})
        taskmaster.get_statuses_fresh = AsyncMock(side_effect=exc())

        with pytest.raises(exc):
            await sweep_orphaned_recon_escalations(queue, taskmaster, KNOWN_PROJECTS)

    @pytest.mark.asyncio
    async def test_human_operator_record_is_swept_and_names_its_own_category(self):
        """The sibling category is covered, and its flag routes to its own row.

        Zero pending records carry this category today (live census
        2026-09-02), so this is future-proofing — which is exactly why the
        category has to be named in the flag rather than assumed.
        """
        queue = make_queue([
            make_escalation(task_id='650', esc_id='esc-650-1', category=HUMAN_OPERATOR),
        ])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'done'}}})

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats['terminal'] == 1
        assert HUMAN_OPERATOR in stats['flags'][0]['description']

    @pytest.mark.asyncio
    async def test_empty_queue_short_circuits_with_no_backend_calls(self):
        """Nothing reapable means no census reads at all."""
        queue = make_queue([make_escalation(task_id='9', category='recon_integrity_issue')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'9': 'done'}}})

        stats = await sweep_orphaned_recon_escalations(
            queue, taskmaster, KNOWN_PROJECTS,
        )

        assert stats == {
            'flags': [], 'scanned': 0, 'terminal': 0, 'missing': 0,
            'live': 0, 'unresolvable': 0, 'errors': 0,
        }
        taskmaster.list_tags.assert_not_awaited()
        taskmaster.get_statuses_fresh.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_sweep_never_closes_a_record(self):
        """Detection only — the A7b contract makes the watcher the sole closer."""
        queue = make_queue([make_escalation(task_id='650')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'done'}}})

        await sweep_orphaned_recon_escalations(queue, taskmaster, KNOWN_PROJECTS)

        queue.resolve.assert_not_called()


_STAT_KEYS = (
    'orphaned_recon_escalations_scanned',
    'orphaned_recon_escalations_terminal',
    'orphaned_recon_escalations_missing',
    'orphaned_recon_escalations_live',
    'orphaned_recon_escalations_unresolvable',
    'orphaned_recon_escalations_errors',
    'orphaned_recon_escalations_flags_emitted',
)


def _make_consolidator(*, escalation_queue, taskmaster):
    """Build a ``MemoryConsolidator`` wired for the sweep (mirrors test_stage1.py)."""
    memory_mock = AsyncMock()
    memory_mock.get_episodes = AsyncMock(return_value=[])
    memory_mock.mem0 = AsyncMock()
    memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_mock.get_status = AsyncMock(return_value={})

    stage = MemoryConsolidator(
        StageId.memory_consolidator,
        memory_mock,
        taskmaster,
        AsyncMock(),  # journal
        ReconciliationConfig(),
        scope=ProjectScope(ProjectId('dark_factory'), ProjectRoot(DARK_ROOT)),
        known_projects=dict(KNOWN_PROJECTS),
    )
    stage.episode_limit = 5
    stage.memory_limit = 10
    stage._escalation_queue = escalation_queue
    return stage


async def _run_stage(stage, *, base_flags=None, dedup_mock=None):
    """Drive ``stage.run()`` with ``BaseStage.run``/``dedup_flags`` patched out."""
    base_report = StageReport(
        stage=StageId.memory_consolidator,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        items_flagged=list(base_flags or []),
        stats={},
    )
    if dedup_mock is None:
        dedup_mock = AsyncMock(side_effect=lambda **kw: list(kw['flags']))
    with (
        patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
        patch(
            'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
            new=dedup_mock,
        ),
    ):
        report = await stage.run(
            events=[],
            watermark=Watermark(project_id='dark_factory'),
            prior_reports=[],
            run_id='run-3052',
        )
    return report, dedup_mock


class TestMemoryConsolidatorOrphanedEscalationWiring:
    """``MemoryConsolidator.run()`` must surface the sweep's flags and stats.

    The seven stats are always present so a reader never needs a
    ``.get(..., 0)`` fallback, and can tell a degraded cycle (``errors > 0``)
    apart from a clean cycle that found nothing.
    """

    @pytest.mark.asyncio
    async def test_terminal_subject_flag_and_stats_reach_the_report(self):
        """A full cycle appends the flag and publishes all seven counts."""
        queue = make_queue([make_escalation(task_id='650', esc_id='esc-650-1')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'done'}}})
        stage = _make_consolidator(escalation_queue=queue, taskmaster=taskmaster)

        report, _ = await _run_stage(stage)

        flags = [
            f for f in report.items_flagged
            if f.get('flag_type') == ORPHANED_ESCALATION_FLAG_TYPE
        ]
        assert len(flags) == 1
        assert flags[0]['task_id'] == '650'
        assert report.stats['orphaned_recon_escalations_scanned'] == 1
        assert report.stats['orphaned_recon_escalations_terminal'] == 1
        assert report.stats['orphaned_recon_escalations_missing'] == 0
        assert report.stats['orphaned_recon_escalations_live'] == 0
        assert report.stats['orphaned_recon_escalations_unresolvable'] == 0
        assert report.stats['orphaned_recon_escalations_errors'] == 0
        assert report.stats['orphaned_recon_escalations_flags_emitted'] == 1

    @pytest.mark.asyncio
    async def test_stats_are_present_and_zero_without_an_escalation_queue(self):
        """No queue means no sweep — but the stat keys still exist and read 0."""
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'done'}}})
        stage = _make_consolidator(escalation_queue=None, taskmaster=taskmaster)

        report, _ = await _run_stage(stage)

        for key in _STAT_KEYS:
            assert report.stats[key] == 0, f'{key} must be present and 0'
        taskmaster.list_tags.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stats_are_present_and_zero_without_a_taskmaster(self):
        """No status oracle means no classification — same always-present contract."""
        queue = make_queue([make_escalation(task_id='650')])
        stage = _make_consolidator(escalation_queue=queue, taskmaster=None)

        report, _ = await _run_stage(stage)

        for key in _STAT_KEYS:
            assert report.stats[key] == 0, f'{key} must be present and 0'

    @pytest.mark.asyncio
    async def test_a_raising_sweep_is_swallowed_and_leaves_stats_at_zero(self):
        """A whole-sweep failure must never abort the stage or half-mutate flags."""
        queue = make_queue([make_escalation(task_id='650')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'done'}}})
        stage = _make_consolidator(escalation_queue=queue, taskmaster=taskmaster)
        pre_existing = {
            'task_id': '100', 'flag_type': 'missing_deliverable',
            'description': 'unrelated', 'category': 'other', 'severity': 'minor',
        }

        with patch(
            'fused_memory.reconciliation.stages.memory_consolidator.'
            'sweep_orphaned_recon_escalations',
            new=AsyncMock(side_effect=RuntimeError('sweep exploded')),
        ):
            report, _ = await _run_stage(stage, base_flags=[pre_existing])

        assert report.items_flagged == [pre_existing], (
            'items_flagged must not be partially mutated by a failed sweep'
        )
        for key in _STAT_KEYS:
            assert report.stats[key] == 0, f'{key} must stay 0 on a failed sweep'

    @pytest.mark.asyncio
    async def test_sweep_does_not_run_on_a_remediation_pass(self):
        """Full cycles only — the same gate the curator-gate sweep uses.

        A remediation pass re-enters ``run()`` to act on findings already
        made; re-sweeping there would re-emit the same orphan flags mid-pass.
        """
        queue = make_queue([make_escalation(task_id='650')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'done'}}})
        stage = _make_consolidator(escalation_queue=queue, taskmaster=taskmaster)
        stage.remediation_findings = [{'description': 'fix something'}]

        report, _ = await _run_stage(stage)

        queue.get_pending.assert_not_called()
        for key in _STAT_KEYS:
            assert report.stats[key] == 0, (
                f'{key} must be present and 0 on a remediation pass'
            )

    @pytest.mark.asyncio
    async def test_flags_are_appended_above_dedup_flags(self):
        """The flag reaches ``dedup_flags``, so it earns a ``stage1_flag_marker`` row.

        Asserted behaviourally rather than by reading source order: appending
        BELOW dedup would bypass dedup entirely, so the orphan would re-emit
        unmarked every cycle with no recurrence history and no way for an
        operator to suppress it.
        """
        queue = make_queue([make_escalation(task_id='650', esc_id='esc-650-1')])
        taskmaster = make_taskmaster({DARK_ROOT: {'master': {'650': 'done'}}})
        stage = _make_consolidator(escalation_queue=queue, taskmaster=taskmaster)

        _, dedup_mock = await _run_stage(stage)

        dedup_mock.assert_awaited_once()
        seen = dedup_mock.await_args.kwargs['flags']
        assert any(
            f.get('flag_type') == ORPHANED_ESCALATION_FLAG_TYPE for f in seen
        ), 'the sweep flag must be present in the list handed to dedup_flags'
