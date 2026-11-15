"""Tests for shared.roles: the config-role vocabulary, the reviewer collapse and the settings row."""

from __future__ import annotations

import pytest

from shared.roles import (
    DISPATCHABLE_ROLES,
    FUSED_MEMORY_ROLES,
    ORCHESTRATOR_ROLES,
    REVIEWER_VARIANTS,
    RoleName,
    UnknownRole,
    config_role,
)
from shared.task_metadata import KNOWN_ROLE_NAMES


class TestVocabulary:
    @pytest.mark.parametrize('role', list(RoleName))
    def test_value_is_the_lower_cased_member_name(self, role: RoleName) -> None:
        assert role.value == role.name.lower()

    def test_every_role_belongs_to_exactly_one_service(self) -> None:
        assert ORCHESTRATOR_ROLES.isdisjoint(FUSED_MEMORY_ROLES)
        assert set(RoleName) == ORCHESTRATOR_ROLES | FUSED_MEMORY_ROLES

    def test_dispatchable_roles_exclude_the_non_dispatched_orchestrator_roles(self) -> None:
        assert ORCHESTRATOR_ROLES - {RoleName.UNBLOCK_AUTO, RoleName.WATCHER} == DISPATCHABLE_ROLES

    def test_dispatchable_roles_and_reviewer_variants_are_the_role_wire_contract(self) -> None:
        assert {r.value for r in DISPATCHABLE_ROLES} | REVIEWER_VARIANTS == KNOWN_ROLE_NAMES


class TestConfigRole:
    def test_reviewer_variant_collapses_to_reviewer(self) -> None:
        assert config_role('reviewer_comprehensive') is RoleName.REVIEWER

    def test_unknown_role_carries_value_and_names_the_vocabulary(self) -> None:
        with pytest.raises(UnknownRole) as caught:
            config_role('nope')
        exc = caught.value
        assert exc.value == 'nope'
        message = str(exc)
        assert 'nope' in message
        assert RoleName.ARCHITECT.value in message
        assert 'reviewer_<name>' in message

    @pytest.mark.parametrize('role', list(RoleName))
    def test_every_role_name_maps_to_itself(self, role: RoleName) -> None:
        assert config_role(role.value) is role

    @pytest.mark.parametrize('variant', sorted(REVIEWER_VARIANTS))
    def test_every_reviewer_variant_maps_to_reviewer(self, variant: str) -> None:
        assert config_role(variant) is RoleName.REVIEWER

    def test_reviewer_family_member_outside_the_variants_maps_to_reviewer(self) -> None:
        assert 'reviewer_security' not in REVIEWER_VARIANTS
        assert config_role('reviewer_security') is RoleName.REVIEWER

    @pytest.mark.parametrize(
        'dispatch_name',
        ['reviewer_', 'reviewerx', 'Architect', 'resettled_adjudicator', ''],
    )
    def test_names_outside_the_vocabulary_are_rejected(self, dispatch_name: str) -> None:
        with pytest.raises(UnknownRole) as caught:
            config_role(dispatch_name)
        assert caught.value.value == dispatch_name
        assert repr(dispatch_name) in str(caught.value)

    def test_unknown_role_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            config_role('nope')
