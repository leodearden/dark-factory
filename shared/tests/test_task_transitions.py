"""Tests for shared.task_transitions — the task-status TRANSITION AUTHORITY.

Table A of the task-status-authority contract (PRD
``plans/task-status-authority-prd.md``, C1/D1/D5, findings 4.1 and the TABLE
half of 1.2). This module is the pure, shared companion to
``shared.task_statuses`` (Table A's vocabulary, task 2163, merged): it adds
``ActorClass``, ``derive_actor_class``, the ``TRANSITIONS`` union/per-actor
table, ``is_legal_transition``, and ``outcome_allows_status``.

``shared/tests/conftest.py`` only puts ``shared/src`` on ``sys.path`` — the
orchestrator and fused-memory packages are not importable from this test
environment, so foreign literals (the enumerated call-site anchors, the
WorkflowOutcome values) are inlined below with "verified verbatim against the
working tree on 2026-07-06" comments rather than imported.

TDD pair 1: ``ActorClass`` StrEnum + exact 5-member vocabulary
(GREEN on impl-actorclass).
TDD pair 2: ``derive_actor_class`` agent_id-prefix mapping, including the
``orchestrator-deterministic`` precedence-over-``orchestrator`` rule (D5)
(GREEN on impl-derive).
TDD pair 3: ``is_legal_transition`` core rules (same-status no-op,
terminal-exit-needs-reopen) + the enumerated ``TRANSITIONS`` union derivation
(GREEN on impl-transitions-core).
TDD pair 4: the single actor-specific restriction — RECONCILIATION may not
transition FROM in-progress (task-1655) — with a safe-open contrast for every
other actor (GREEN on impl-actor-restriction).
TDD pair 5: ``outcome_allows_status`` outcome->status map, keyed on the 8
mirrored ``WorkflowOutcome`` string values, with getattr(.,'value',.)
normalization and loud ValueError on unknown outcome/status
(GREEN on impl-outcome).
TDD pair 6: module contract — exact ``__all__``, ``TRANSITIONS`` completeness
over every ``ActorClass`` member, and a purity check that the module loads
from under ``shared/src`` (GREEN on impl-contract).
"""
from __future__ import annotations

import enum
import os
import types
from pathlib import Path

import pytest

from shared.task_statuses import TaskStatus
from shared.task_transitions import (
    _OUTCOME_ALLOWED,
    TRANSITIONS,
    ActorClass,
    derive_actor_class,
    is_legal_transition,
    outcome_allows_status,
)

# ---------------------------------------------------------------------------
# Pair 1 — ActorClass StrEnum (test-actorclass RED / impl-actorclass GREEN)
# ---------------------------------------------------------------------------


class TestActorClass:
    def test_is_str_enum(self):
        assert issubclass(ActorClass, enum.StrEnum)

    def test_exact_member_set(self):
        assert {m.name for m in ActorClass} == {
            'ORCHESTRATOR',
            'RECONCILIATION',
            'ESCALATION',
            'DETERMINISTIC',
            'HUMAN',
        }
        assert len(ActorClass) == 5

    def test_values(self):
        assert ActorClass.ORCHESTRATOR == 'orchestrator'
        assert ActorClass.RECONCILIATION == 'reconciliation'
        assert ActorClass.ESCALATION == 'escalation'
        assert ActorClass.DETERMINISTIC == 'deterministic'
        assert ActorClass.HUMAN == 'human'

    def test_members_are_genuine_str(self):
        assert ActorClass.HUMAN == 'human'
        assert isinstance(ActorClass.ORCHESTRATOR, str)


# ---------------------------------------------------------------------------
# Pair 2 — derive_actor_class (test-derive RED / impl-derive GREEN)
# ---------------------------------------------------------------------------

# Live agent_id prefixes verified verbatim against the working tree
# 2026-07-06: recon-stage-{stage_id} at
# fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py:2787;
# DETERMINISTIC_AGENT_ROLE = 'orchestrator-deterministic' at
# orchestrator/src/orchestrator/deterministic_runner.py:171.
_DERIVE_CASES = [
    # (agent_id, expected ActorClass) — order matters: prefix PRECEDENCE below.
    (None, ActorClass.HUMAN),
    ('', ActorClass.HUMAN),
    ('recon-stage-memory_consolidator', ActorClass.RECONCILIATION),
    ('recon-stage-task_knowledge_sync', ActorClass.RECONCILIATION),
    # Defensive doc-convention variant (not a live prefix, but matched anyway).
    ('reconciliation-stage-1', ActorClass.RECONCILIATION),
    # orchestrator-deterministic MUST win over the plain orchestrator* rule.
    ('orchestrator-deterministic', ActorClass.DETERMINISTIC),
    ('orchestrator-deterministic-7', ActorClass.DETERMINISTIC),
    ('orchestrator', ActorClass.ORCHESTRATOR),
    # Starts with 'orchestrator' but NOT 'orchestrator-deterministic'.
    ('orchestrator-escalation-watcher-auto', ActorClass.ORCHESTRATOR),
    ('harness-abc', ActorClass.ORCHESTRATOR),
    ('steward', ActorClass.ORCHESTRATOR),
    ('steward-9', ActorClass.ORCHESTRATOR),
    ('escalation-server', ActorClass.ESCALATION),
    ('claude-task-2168-architect', ActorClass.HUMAN),
]


class TestDeriveActorClass:
    @pytest.mark.parametrize('agent_id, expected', _DERIVE_CASES)
    def test_prefix_mapping(self, agent_id, expected):
        assert derive_actor_class(agent_id) is expected

    def test_returns_actor_class_instance(self):
        assert isinstance(derive_actor_class('claude-task-2168-architect'), ActorClass)
        assert isinstance(derive_actor_class(None), ActorClass)


# ---------------------------------------------------------------------------
# Pair 3 — is_legal_transition core rules + TRANSITIONS union derivation
# (test-transitions-core RED / impl-transitions-core GREEN)
# ---------------------------------------------------------------------------


class TestIsLegalTransitionCore:
    def test_same_status_always_legal(self):
        # Mirrors the interceptor's status==old_status early-return
        # (task_interceptor.py:645) — a no-op write is never illegal.
        assert is_legal_transition(TaskStatus.IN_PROGRESS, TaskStatus.IN_PROGRESS, ActorClass.HUMAN) is True
        assert is_legal_transition(TaskStatus.DONE, TaskStatus.DONE, ActorClass.HUMAN) is True

    def test_dispatch_and_completion_legal(self):
        assert is_legal_transition(TaskStatus.PENDING, TaskStatus.IN_PROGRESS, ActorClass.ORCHESTRATOR) is True
        assert is_legal_transition(TaskStatus.IN_PROGRESS, TaskStatus.DONE, ActorClass.ORCHESTRATOR) is True

    def test_non_terminal_source_to_terminal_needs_no_reopen(self):
        # merge-deferred -> done is a normal completion path (workflow.py:1015
        # /6424, harness.py:619/646) — reopen is only about LEAVING a
        # terminal status, not entering one.
        assert is_legal_transition(TaskStatus.MERGE_DEFERRED, TaskStatus.DONE, ActorClass.HUMAN) is True

    def test_terminal_exit_without_reopen_illegal(self):
        # frm in TERMINAL -> bool(reopen); default reopen=False rejects every
        # exit from done/cancelled, matching task_interceptor.py:702.
        assert is_legal_transition(TaskStatus.DONE, TaskStatus.PENDING, ActorClass.HUMAN) is False
        assert is_legal_transition(TaskStatus.DONE, TaskStatus.BLOCKED, ActorClass.HUMAN) is False
        assert is_legal_transition(TaskStatus.CANCELLED, TaskStatus.PENDING, ActorClass.HUMAN) is False

    def test_terminal_exit_with_reopen_legal(self):
        assert is_legal_transition(TaskStatus.DONE, TaskStatus.PENDING, ActorClass.HUMAN, reopen=True) is True
        assert (
            is_legal_transition(TaskStatus.DONE, TaskStatus.BLOCKED, ActorClass.ORCHESTRATOR, reopen=True) is True
        )
        assert (
            is_legal_transition(TaskStatus.CANCELLED, TaskStatus.IN_PROGRESS, ActorClass.HUMAN, reopen=True) is True
        )

    def test_non_enumerated_non_terminal_move_illegal(self):
        # Deferred must go via pending — it cannot jump straight back to
        # in-progress.
        assert is_legal_transition(TaskStatus.DEFERRED, TaskStatus.IN_PROGRESS, ActorClass.HUMAN) is False
        # pending -> merge-deferred is not an enumerated call site.
        assert is_legal_transition(TaskStatus.PENDING, TaskStatus.MERGE_DEFERRED, ActorClass.HUMAN) is False


# Core enumerated (from, to) pairs, grouped by kind, each with its call-site
# anchor — verified verbatim against the working tree on 2026-07-06. This
# documents the DERIVATION (not the full union: forward-compat infra-hold
# edges and soak-validated review/deferred edges live in the implementation
# but are not re-derived here) and pins that every one of these pairs is
# legal for an unrestricted actor (HUMAN gets the full union, D5 safe-open).
_CORE_UNION_PAIRS = [
    # dispatch
    (TaskStatus.PENDING, TaskStatus.IN_PROGRESS, 'dispatch: workflow.py:1510'),
    # completion
    (TaskStatus.IN_PROGRESS, TaskStatus.DONE, 'completion: merged workflow.py:1380, found_on_main workflow.py:3809/7396/harness.py:3623'),
    (TaskStatus.MERGE_DEFERRED, TaskStatus.DONE, 'completion: workflow.py:1015/6424, harness.py:619/646'),
    (TaskStatus.BLOCKED, TaskStatus.DONE, 'completion: train attribution workflow.py:6424'),
    # park
    (TaskStatus.IN_PROGRESS, TaskStatus.MERGE_DEFERRED, 'park: workflow.py:867/896'),
    # requeue
    (
        TaskStatus.IN_PROGRESS,
        TaskStatus.PENDING,
        'requeue: workflow.py:2464/3755, blast-radius scheduler.py:4484, stranded-revert harness.py:3487/3567',
    ),
    (
        TaskStatus.BLOCKED,
        TaskStatus.PENDING,
        'requeue: steward re-pend workflow.py:8007, escalation resume harness.py:8739',
    ),
    (TaskStatus.MERGE_DEFERRED, TaskStatus.PENDING, 'requeue: re-drive harness.py:682'),
    (TaskStatus.DEFERRED, TaskStatus.PENDING, 'requeue: deferred resumption, stage2 commit_planning'),
    # block
    (
        TaskStatus.IN_PROGRESS,
        TaskStatus.BLOCKED,
        '_mark_blocked workflow.py:7758, retry-cap scheduler.py:4659, substrate harness.py:4687, dep harness.py:3905',
    ),
    (TaskStatus.MERGE_DEFERRED, TaskStatus.BLOCKED, 'block: train failer workflow.py:6464'),
    (TaskStatus.DEFERRED, TaskStatus.BLOCKED, 'block: recon targeted.py:1041'),
    # cancel
    (TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED, 'cancel: abandon harness.py:8417'),
    (TaskStatus.BLOCKED, TaskStatus.CANCELLED, 'cancel: harness.py:8417'),
    (TaskStatus.DEFERRED, TaskStatus.CANCELLED, 'cancel: recon targeted.py:1001'),
]


class TestTransitionsUnionDerivation:
    def test_core_pairs_are_legal_for_human(self):
        for frm, to, anchor in _CORE_UNION_PAIRS:
            assert (frm, to) in TRANSITIONS[ActorClass.HUMAN], (frm, to, anchor)

    def test_transitions_human_is_frozenset_of_status_pairs(self):
        union = TRANSITIONS[ActorClass.HUMAN]
        assert isinstance(union, frozenset)
        assert all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and isinstance(pair[0], TaskStatus)
            and isinstance(pair[1], TaskStatus)
            for pair in union
        )


# ---------------------------------------------------------------------------
# Pair 4 — the single actor-specific restriction: RECONCILIATION may not
# transition FROM in-progress (task-1655, D5)
# (test-actor-restriction RED / impl-actor-restriction GREEN)
# ---------------------------------------------------------------------------


class TestActorRestriction:
    def test_reconciliation_cannot_transition_from_in_progress(self):
        # The live_workflow_status_write guard (task_knowledge_sync.py:521/
        # 2996) actively blocks recon writes on a live/in-progress task —
        # this is that restriction's pure-table mirror.
        assert is_legal_transition(TaskStatus.IN_PROGRESS, TaskStatus.PENDING, ActorClass.RECONCILIATION) is False
        assert is_legal_transition(TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, ActorClass.RECONCILIATION) is False
        assert is_legal_transition(TaskStatus.IN_PROGRESS, TaskStatus.DONE, ActorClass.RECONCILIATION) is False

    @pytest.mark.parametrize(
        'actor', [ActorClass.ORCHESTRATOR, ActorClass.HUMAN, ActorClass.ESCALATION, ActorClass.DETERMINISTIC]
    )
    def test_safe_open_contrast_for_other_actors(self, actor):
        # Same (from, to) pair the recon test above rejects — every other
        # actor gets the full union (safe-open, D5).
        assert is_legal_transition(TaskStatus.IN_PROGRESS, TaskStatus.PENDING, actor) is True

    def test_reconciliations_legitimate_writes_remain_legal(self):
        # Recon's real call sites (task_knowledge_sync.py:2787 stage2;
        # targeted.py:1001/1041) — none of these originate from in-progress,
        # so the restriction must not collaterally block them.
        assert is_legal_transition(TaskStatus.DEFERRED, TaskStatus.PENDING, ActorClass.RECONCILIATION) is True
        assert is_legal_transition(TaskStatus.PENDING, TaskStatus.CANCELLED, ActorClass.RECONCILIATION) is True
        assert is_legal_transition(TaskStatus.BLOCKED, TaskStatus.CANCELLED, ActorClass.RECONCILIATION) is True
        assert is_legal_transition(TaskStatus.DEFERRED, TaskStatus.BLOCKED, ActorClass.RECONCILIATION) is True

    def test_unknown_actor_is_safe_open(self):
        assert is_legal_transition(TaskStatus.IN_PROGRESS, TaskStatus.PENDING, 'mystery-actor') is True

    def test_reconciliation_is_strict_subset_of_human(self):
        assert TRANSITIONS[ActorClass.RECONCILIATION] < TRANSITIONS[ActorClass.HUMAN]


# ---------------------------------------------------------------------------
# Pair 5 — outcome_allows_status: WorkflowOutcome -> TaskStatus consistency
# map (test-outcome RED / impl-outcome GREEN)
# ---------------------------------------------------------------------------

# The 8 WorkflowOutcome string values, verified verbatim against
# orchestrator/src/orchestrator/workflow.py:337-345 on 2026-07-06. shared/
# must not import orchestrator (layering, program decision #4), so these are
# mirrored literals — test_recognized_outcome_keys_match_mirrored_workflow_outcomes
# below is the drift guard that catches a future WorkflowOutcome addition that
# isn't mirrored here.
_WORKFLOW_OUTCOME_VALUES = {
    'done',
    'planned',
    'blocked',
    'requeued',
    'escalated',
    'cancelled',
    'merge-deferred',
    'soft-cancelled',
}


class TestOutcomeAllowsStatus:
    def test_done_outcome(self):
        assert outcome_allows_status('done', TaskStatus.DONE) is True
        assert outcome_allows_status('done', 'done') is True
        assert outcome_allows_status('done', 'blocked') is False

    def test_blocked_outcome(self):
        assert outcome_allows_status('blocked', 'blocked') is True

    def test_escalated_outcome(self):
        # Dominant _mark_blocked+open-L1 path, and the merge-gating bail
        # that preserves in-progress.
        assert outcome_allows_status('escalated', 'blocked') is True
        assert outcome_allows_status('escalated', 'in-progress') is True

    def test_requeued_outcome(self):
        # Self-repend / deferred-to-stranded-sweep window / requeue-cap
        # exhausted -> blocked.
        assert outcome_allows_status('requeued', 'pending') is True
        assert outcome_allows_status('requeued', 'in-progress') is True
        assert outcome_allows_status('requeued', 'blocked') is True
        assert outcome_allows_status('requeued', 'done') is False

    def test_cancelled_outcome(self):
        assert outcome_allows_status('cancelled', 'cancelled') is True

    def test_merge_deferred_outcome(self):
        assert outcome_allows_status('merge-deferred', 'merge-deferred') is True

    def test_planned_outcome(self):
        # Internal-only sub-phase — the task stays claimed (in-progress).
        assert outcome_allows_status('planned', 'in-progress') is True
        assert outcome_allows_status('planned', 'done') is False

    def test_soft_cancelled_outcome(self):
        # preserved / release_workflow park->blocked / stranded-sweep->pending.
        assert outcome_allows_status('soft-cancelled', 'in-progress') is True
        assert outcome_allows_status('soft-cancelled', 'blocked') is True
        assert outcome_allows_status('soft-cancelled', 'pending') is True

    def test_normalizes_value_attribute(self):
        # W9 may pass a real WorkflowOutcome instance rather than a plain
        # string; proves the getattr(outcome, 'value', outcome)
        # normalization path without importing orchestrator.
        stub = types.SimpleNamespace(value='done')
        assert outcome_allows_status(stub, 'done') is True

    def test_recognized_outcome_keys_match_mirrored_workflow_outcomes(self):
        assert set(_OUTCOME_ALLOWED.keys()) == _WORKFLOW_OUTCOME_VALUES

    def test_unknown_outcome_raises(self):
        with pytest.raises(ValueError):
            outcome_allows_status('bogus-outcome', 'done')

    def test_unknown_status_raises(self):
        with pytest.raises(ValueError):
            outcome_allows_status('done', 'not-a-status')


# ---------------------------------------------------------------------------
# Pair 6 — module contract: exact __all__, TRANSITIONS completeness, purity
# (test-contract RED / impl-contract GREEN)
# ---------------------------------------------------------------------------

_EXPECTED_PUBLIC_API = {
    'ActorClass',
    'derive_actor_class',
    'TRANSITIONS',
    'is_legal_transition',
    'outcome_allows_status',
}


class TestModuleContract:
    def test_all_is_exactly_the_five_names(self):
        import shared.task_transitions as tt

        assert hasattr(tt, '__all__')
        assert set(tt.__all__) == _EXPECTED_PUBLIC_API

    def test_no_private_entries_in_all(self):
        import shared.task_transitions as tt

        assert not any(name.startswith('_') for name in tt.__all__)

    def test_every_all_entry_is_a_real_attribute(self):
        import shared.task_transitions as tt

        for name in tt.__all__:
            getattr(tt, name)  # raises AttributeError if missing

    def test_transitions_completeness_over_actor_class(self):
        for actor in ActorClass:
            assert actor in TRANSITIONS
            assert isinstance(TRANSITIONS[actor], frozenset)

    def test_transitions_human_is_the_union(self):
        assert TRANSITIONS[ActorClass.HUMAN] == TRANSITIONS[ActorClass.ORCHESTRATOR]
        assert TRANSITIONS[ActorClass.HUMAN] == TRANSITIONS[ActorClass.ESCALATION]
        assert TRANSITIONS[ActorClass.HUMAN] == TRANSITIONS[ActorClass.DETERMINISTIC]

    def test_transitions_reconciliation_is_a_proper_subset_of_human(self):
        assert TRANSITIONS[ActorClass.RECONCILIATION] < TRANSITIONS[ActorClass.HUMAN]

    def test_module_loads_from_under_shared_src(self):
        # shared/tests/conftest.py puts ONLY shared/src on sys.path, so the
        # mere fact that this import succeeds already proves the module
        # pulls in nothing from orchestrator/fused-memory. This test makes
        # that purity guarantee explicit and load-bearing rather than
        # implicit in every other test's successful collection.
        import shared.task_transitions as tt

        module_path = Path(tt.__file__).resolve()
        assert 'shared' + os.sep + 'src' in str(module_path)
