"""Tests for shared.roles: the config-role vocabulary, the reviewer collapse and the settings row."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.backends import Backend
from shared.roles import (
    DISPATCHABLE_ROLES,
    FUSED_MEMORY_ROLES,
    ORCHESTRATOR_ROLES,
    REVIEWER_VARIANTS,
    Effort,
    RoleName,
    RoleSettings,
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


EFFORT_SPELLINGS = ('off', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
POSITIVE_KNOBS = ('budget_usd', 'max_turns', 'timeout_secs')


def _error_locations(exc: ValidationError) -> list[tuple[int | str, ...]]:
    return [tuple(error['loc']) for error in exc.errors()]


class TestRoleSettings:
    def test_only_model_is_required(self) -> None:
        row = RoleSettings(model='opus')
        assert row.backend is Backend.CLAUDE
        assert row.effort is None
        assert row.budget_usd is None
        assert row.max_turns is None
        assert row.timeout_secs is None

    @pytest.mark.parametrize(
        'spelling, backend',
        [('claude_cli', Backend.CLAUDE), ('zcode', Backend.ZCODE)],
    )
    def test_backend_is_decoded_by_parse_backend(self, spelling: str, backend: Backend) -> None:
        row = RoleSettings.model_validate({'model': 'opus', 'backend': spelling})
        assert row.backend is backend

    def test_unknown_backend_fails_at_load_naming_value_and_allowed(self) -> None:
        with pytest.raises(ValidationError) as caught:
            RoleSettings.model_validate({'model': 'opus', 'backend': 'Claude'})
        message = str(caught.value)
        assert 'Claude' in message
        for backend in Backend:
            assert backend.value in message

    @pytest.mark.parametrize('spelling', EFFORT_SPELLINGS)
    def test_effort_accepts_each_level(self, spelling: str) -> None:
        row = RoleSettings.model_validate({'model': 'opus', 'effort': spelling})
        assert row.effort == spelling

    def test_effort_vocabulary(self) -> None:
        assert {e.value for e in Effort} == set(EFFORT_SPELLINGS)

    @pytest.mark.parametrize('spelling', ['hgih', 'High', '', 'ultra'])
    def test_effort_rejects_unknown_levels_at_load(self, spelling: str) -> None:
        with pytest.raises(ValidationError) as caught:
            RoleSettings.model_validate({'model': 'opus', 'effort': spelling})
        (error,) = caught.value.errors()
        assert tuple(error['loc']) == ('effort',)
        assert error['input'] == spelling

    def test_effort_renders_as_its_bare_value(self) -> None:
        row = RoleSettings(model='opus', effort=Effort.XHIGH)
        assert f'{row.effort}' == 'xhigh'

    @pytest.mark.parametrize('knob', POSITIVE_KNOBS)
    @pytest.mark.parametrize('value', [0, -1])
    def test_knob_must_be_positive(self, knob: str, value: int) -> None:
        with pytest.raises(ValidationError) as caught:
            RoleSettings.model_validate({'model': 'opus', knob: value})
        assert _error_locations(caught.value) == [(knob,)]

    @pytest.mark.parametrize('knob', POSITIVE_KNOBS)
    @pytest.mark.parametrize('value', [5, None])
    def test_knob_accepts_positive_or_none(self, knob: str, value: int | None) -> None:
        row = RoleSettings.model_validate({'model': 'opus', knob: value})
        assert getattr(row, knob) == value

    @pytest.mark.parametrize('raw', [{'model': ''}, {}])
    def test_model_must_be_a_non_empty_string(self, raw: dict[str, object]) -> None:
        with pytest.raises(ValidationError) as caught:
            RoleSettings.model_validate(raw)
        assert _error_locations(caught.value) == [('model',)]

    def test_unknown_knob_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as caught:
            RoleSettings.model_validate({'model': 'x', 'nope': 1})
        assert _error_locations(caught.value) == [('nope',)]

    def test_row_is_immutable(self) -> None:
        row = RoleSettings(model='x')
        with pytest.raises(ValidationError):
            row.model = 'y'
