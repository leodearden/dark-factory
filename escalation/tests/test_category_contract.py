"""The escalation-category contract, tested through its public names only.

``escalation.categories.EscalationCategory`` is the one registered vocabulary
every filed ``Escalation.category`` must belong to, and
``require_registered_category`` is the one check every enforcement point calls.
"""

from __future__ import annotations

import pytest

from escalation.categories import (
    EscalationCategory,
    UnregisteredEscalationCategory,
    require_registered_category,
)

_CATEGORIES_THIS_TASK_REGISTERS = (
    'cascade_status_rejection',
    'already_landed',
    'reconciliation_standing_decision_storm',
    'regressed_after_resolution',
    'milestone_check_stalled',
    'review_due',
)

_RETIRED_PROSE_LIST = (
    'scope_violation',
    'design_concern',
    'cleanup_needed',
    'dependency_discovered',
    'risk_identified',
    'infra_issue',
    'reconciliation_stale_human_operator',
    'reconciliation_stale_gate_backlog',
    'recon_missing_index',
)

_AGENT_INSTRUCTED = (
    'missing_premise',
    'planning_stalled',
    'out_of_scope',
    'merge_scope_mismatch',
    'workflow_exit_contract',
    'integration_skew',
)


class TestRegistry:
    @pytest.mark.parametrize('value', _CATEGORIES_THIS_TASK_REGISTERS)
    def test_outstanding_categories_are_registered(self, value: str):
        assert value in EscalationCategory

    @pytest.mark.parametrize('value', _RETIRED_PROSE_LIST)
    def test_retired_prose_list_stays_registered(self, value: str):
        assert value in EscalationCategory

    @pytest.mark.parametrize('value', _AGENT_INSTRUCTED)
    def test_agent_instructed_categories_are_registered(self, value: str):
        assert value in EscalationCategory

    def test_member_value_is_its_lowercased_name(self):
        mismatched = [m for m in EscalationCategory if m.value != m.name.lower()]
        assert mismatched == []

    def test_values_are_unique(self):
        assert len({m.value for m in EscalationCategory}) == len(EscalationCategory)


class TestRequireRegisteredCategory:
    def test_registered_plain_string_passes(self):
        assert require_registered_category('infra_issue') is None

    def test_member_passes(self):
        assert require_registered_category(EscalationCategory.INFRA_ISSUE) is None

    def test_typo_raises_informatively(self):
        with pytest.raises(UnregisteredEscalationCategory) as excinfo:
            require_registered_category('infra_isue')
        message = str(excinfo.value)
        assert repr('infra_isue') in message
        assert 'escalation.categories.EscalationCategory' in message
        assert 'infra_issue' in message

    def test_unregistered_is_a_value_error(self):
        assert issubclass(UnregisteredEscalationCategory, ValueError)

    @pytest.mark.parametrize('value', ['INFRA_ISSUE', ' infra_issue'])
    def test_membership_is_exact(self, value: str):
        with pytest.raises(UnregisteredEscalationCategory):
            require_registered_category(value)

    @pytest.mark.parametrize('value', [None, ''])
    def test_missing_category_raises(self, value: object):
        with pytest.raises(UnregisteredEscalationCategory):
            require_registered_category(value)
