"""Tests for Table B: escalation/action_effects.py.

Table B is the escalation-resolution-intent authority — PRD
``plans/task-status-authority-prd.md`` contract C5 / decisions D1, D2
(finding 10.0). These tests pin the behaviour-preserving
``(action, level, category) -> TaskEffect`` mapping BEFORE
``resolve_issue`` is wired to consult it (that wiring is step-4 of this
task's plan).
"""

from __future__ import annotations

import pytest

# TEST-TIME only import: dark-factory-shared is already a declared workspace
# dependency of escalation/pyproject.toml, so this resolves without adding a
# new production dependency. action_effects.py itself does NOT import shared
# (see its module docstring / the plan's reuse notes) — it keeps plain
# lowercase status-string literals.
from shared.task_statuses import TaskStatus

from escalation.action_effects import (
    ACTION_EFFECTS,
    ANY,
    WORKFLOW_DISPOSITIONS,
    TaskEffect,
    effect_for,
)

# Behaviour-preserving mapping transcribed from
# orchestrator/src/orchestrator/harness.py:8259-8325 (_ACTION_TARGETS /
# _on_escalation_resolved) — the source of truth Table B mirrors verbatim.
_EXPECTED_TARGET_STATUS: dict[str, str | None] = {
    'resume': 'pending',
    'restart': 'pending',
    'park': 'blocked',
    'abandon': 'cancelled',
    'close_only': None,
}

_VALID_ACTIONS = tuple(_EXPECTED_TARGET_STATUS)

# Categories deliberately NOT in server.CATEGORIES (G6 archive check found
# these — and more — occurring live; server.CATEGORIES is inert/unvalidated).
_NOVEL_CATEGORIES = ('milestone_gate', 'ticket_failure', 'preexisting_main_break')
_KNOWN_CATEGORIES = ('scope_violation', 'design_concern')


class TestEffectForKnownActions:
    """(a) effect_for returns the correct TaskEffect per action."""

    @pytest.mark.parametrize('action', _VALID_ACTIONS)
    def test_effect_for_returns_expected_target_status(self, action: str) -> None:
        effect = effect_for(action, 0, 'scope_violation')
        assert effect is not None, f'{action!r} must be a legal action'
        assert effect.target_status == _EXPECTED_TARGET_STATUS[action]

    @pytest.mark.parametrize('action', _VALID_ACTIONS)
    def test_effect_for_workflow_disposition_is_known(self, action: str) -> None:
        effect = effect_for(action, 0, 'scope_violation')
        assert effect is not None
        assert effect.workflow_disposition in WORKFLOW_DISPOSITIONS

    def test_actions_have_distinct_dispositions(self) -> None:
        """Each of the five actions maps to its own, distinct workflow disposition."""
        dispositions = {}
        for action in _VALID_ACTIONS:
            effect = effect_for(action, 0, 'scope_violation')
            assert effect is not None
            dispositions[action] = effect.workflow_disposition
        assert len(set(dispositions.values())) == len(dispositions), dispositions


class TestTaskStatusParity:
    """(b) every non-None TaskEffect.target_status is a real TaskStatus value."""

    def test_all_target_statuses_are_valid_task_statuses(self) -> None:
        valid = {s.value for s in TaskStatus}
        checked = 0
        for key, effect in ACTION_EFFECTS.items():
            if effect.target_status is None:
                continue
            assert effect.target_status in valid, (
                f'{key}: target_status {effect.target_status!r} is not a TaskStatus value'
            )
            checked += 1
        assert checked > 0, 'expected at least one non-None target_status to check'


class TestEffectForUnknownAction:
    """(c) effect_for returns None for unknown / illegal actions."""

    @pytest.mark.parametrize('level', [0, 1, 2])
    @pytest.mark.parametrize('category', [*_KNOWN_CATEGORIES, *_NOVEL_CATEGORIES])
    def test_bogus_action_is_illegal(self, level: int, category: str) -> None:
        assert effect_for('bogus', level, category) is None


class TestBehaviourPreservationWildcard:
    """(d) legality is not narrowed by level or category for any valid action."""

    @pytest.mark.parametrize('action', _VALID_ACTIONS)
    @pytest.mark.parametrize('level', [0, 1, 2])
    @pytest.mark.parametrize('category', [*_KNOWN_CATEGORIES, *_NOVEL_CATEGORIES])
    def test_every_exotic_combo_is_legal_for_valid_actions(
        self, action: str, level: int, category: str
    ) -> None:
        assert effect_for(action, level, category) is not None


class TestActionEffectsTableShape:
    """ACTION_EFFECTS is keyed wildcarded (action, ANY, ANY) today — see module docstring."""

    def test_action_effects_keys_are_wildcarded(self) -> None:
        for action in _VALID_ACTIONS:
            assert (action, ANY, ANY) in ACTION_EFFECTS, (
                f'{action!r} must be keyed as (action, ANY, ANY) — narrowing by '
                'level/category is out of scope for this task'
            )

    def test_action_effects_has_exactly_five_entries(self) -> None:
        assert len(ACTION_EFFECTS) == 5

    def test_action_effects_values_are_task_effects(self) -> None:
        for effect in ACTION_EFFECTS.values():
            assert isinstance(effect, TaskEffect)
