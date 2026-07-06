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

import pytest

from shared.task_transitions import ActorClass, derive_actor_class

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
