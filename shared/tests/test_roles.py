"""Tests for shared.roles: the config-role vocabulary, the reviewer collapse and the settings row."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ValidationError, model_validator

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
    merge_role_table,
)
from shared.task_metadata import KNOWN_ROLE_NAMES


class TestVocabulary:
    @pytest.mark.parametrize('role', list(RoleName))
    def test_value_is_the_lower_cased_member_name(self, role: RoleName) -> None:
        assert role.value == role.name.lower()

    def test_every_role_belongs_to_exactly_one_service(self) -> None:
        assert ORCHESTRATOR_ROLES.isdisjoint(FUSED_MEMORY_ROLES)
        assert set(RoleName) == ORCHESTRATOR_ROLES | FUSED_MEMORY_ROLES

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
        assert exc.allowed == tuple(r.value for r in RoleName)
        assert exc.family_prefix == 'reviewer_'
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
    @pytest.mark.parametrize('value', [True, '5'])
    def test_knob_must_be_a_number(self, knob: str, value: object) -> None:
        with pytest.raises(ValidationError) as caught:
            RoleSettings.model_validate({'model': 'opus', knob: value})
        assert _error_locations(caught.value) == [(knob,)]

    @pytest.mark.parametrize('knob', ['budget_usd', 'timeout_secs'])
    @pytest.mark.parametrize('value', [float('inf'), float('nan')])
    def test_float_knob_must_be_finite(self, knob: str, value: float) -> None:
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


DEFAULTS: dict[RoleName, RoleSettings] = {
    RoleName.ARCHITECT: RoleSettings(
        model='opus', effort=Effort.XHIGH, budget_usd=10.0, max_turns=60, timeout_secs=2400.0
    ),
    RoleName.REVIEWER: RoleSettings(
        model='sonnet', effort=Effort.MEDIUM, budget_usd=2.0, max_turns=30, timeout_secs=600.0
    ),
}


class _ServiceConfig(BaseModel):
    roles: dict[RoleName, RoleSettings]

    @model_validator(mode='before')
    @classmethod
    def _merge_roles(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        return {**data, 'roles': merge_role_table(DEFAULTS, data.get('roles', {}))}


class TestMergeRoleTable:
    def test_empty_raw_table_is_the_defaults(self) -> None:
        merged = merge_role_table(DEFAULTS, {})
        assert merged == DEFAULTS
        assert list(merged) == list(DEFAULTS)

    def test_entry_overrides_only_the_knobs_it_names(self) -> None:
        merged = merge_role_table(DEFAULTS, {'architect': {'model': 'sonnet'}})
        architect = merged[RoleName.ARCHITECT]
        assert architect.model == 'sonnet'
        default = DEFAULTS[RoleName.ARCHITECT]
        assert architect.effort == default.effort
        assert architect.budget_usd == default.budget_usd
        assert architect.max_turns == default.max_turns
        assert architect.timeout_secs == default.timeout_secs
        assert architect.backend == default.backend
        assert merged[RoleName.REVIEWER] == DEFAULTS[RoleName.REVIEWER]

    def test_role_name_members_are_accepted_as_keys(self) -> None:
        merged = merge_role_table(DEFAULTS, {RoleName.ARCHITECT: {'max_turns': 90}})
        assert merged[RoleName.ARCHITECT].max_turns == 90

    def test_knob_values_are_decoded_like_a_fresh_row(self) -> None:
        merged = merge_role_table(DEFAULTS, {'architect': {'backend': 'claude_cli', 'effort': 'max'}})
        assert merged[RoleName.ARCHITECT].backend is Backend.CLAUDE
        assert merged[RoleName.ARCHITECT].effort is Effort.MAX

    def test_explicit_null_unsets_a_knob(self) -> None:
        merged = merge_role_table(DEFAULTS, {'architect': {'budget_usd': None}})
        assert merged[RoleName.ARCHITECT].budget_usd is None

    def test_row_instance_contributes_only_its_explicitly_set_knobs(self) -> None:
        merged = merge_role_table(DEFAULTS, {'architect': RoleSettings(model='haiku')})
        assert merged[RoleName.ARCHITECT].model == 'haiku'
        assert merged[RoleName.ARCHITECT].budget_usd == 10.0

    @pytest.mark.parametrize('key', ['nope', 'watcher', 'reviewer_comprehensive'])
    def test_keys_outside_the_table_are_rejected(self, key: str) -> None:
        with pytest.raises(UnknownRole) as caught:
            merge_role_table(DEFAULTS, {key: {}})
        assert caught.value.value == key
        assert caught.value.allowed == ('architect', 'reviewer')
        assert caught.value.family_prefix is None
        assert 'reviewer_<name>' not in str(caught.value)

    def test_unknown_knob_is_rejected_naming_role_and_knob(self) -> None:
        with pytest.raises(ValidationError) as caught:
            merge_role_table(DEFAULTS, {'architect': {'nope': 1}})
        message = str(caught.value)
        assert 'architect' in message
        assert 'nope' in message

    def test_bad_knob_value_is_rejected_naming_role_knob_and_input(self) -> None:
        with pytest.raises(ValidationError) as caught:
            merge_role_table(DEFAULTS, {'architect': {'budget_usd': -1}})
        message = str(caught.value)
        assert 'architect' in message
        assert 'budget_usd' in message
        assert '-1' in message

    def test_non_mapping_row_is_rejected_naming_role_and_value(self) -> None:
        with pytest.raises(ValueError) as caught:
            merge_role_table(DEFAULTS, {'architect': 'opus'})
        message = str(caught.value)
        assert 'architect' in message
        assert 'opus' in message

    @pytest.mark.parametrize('raw', [None, ['architect']])
    def test_non_mapping_table_is_rejected_naming_the_value(self, raw: object) -> None:
        with pytest.raises(ValueError) as caught:
            merge_role_table(DEFAULTS, raw)
        assert repr(raw) in str(caught.value)

    def test_result_follows_role_name_declaration_order(self) -> None:
        reversed_defaults = dict(reversed(list(DEFAULTS.items())))
        assert list(merge_role_table(reversed_defaults, {})) == [RoleName.ARCHITECT, RoleName.REVIEWER]

    def test_inputs_are_not_mutated(self) -> None:
        raw = {'architect': {'model': 'sonnet', 'budget_usd': None}}
        defaults_before = copy.deepcopy(DEFAULTS)
        raw_before = copy.deepcopy(raw)
        merge_role_table(DEFAULTS, raw)
        assert defaults_before == DEFAULTS
        assert raw_before == raw

    def test_consumer_keeps_default_knobs_at_load(self) -> None:
        config = _ServiceConfig.model_validate({'roles': {'architect': {'model': 'sonnet'}}})
        assert config.roles[RoleName.ARCHITECT].model == 'sonnet'
        assert config.roles[RoleName.ARCHITECT].budget_usd == 10.0

    def test_consumer_rejects_an_unknown_role_at_load(self) -> None:
        with pytest.raises(ValidationError) as caught:
            _ServiceConfig.model_validate({'roles': {'nope': {}}})
        assert 'nope' in str(caught.value)

    @pytest.mark.parametrize('raw', [None, ['architect']])
    def test_consumer_rejects_a_non_mapping_table_at_load(self, raw: object) -> None:
        with pytest.raises(ValidationError) as caught:
            _ServiceConfig.model_validate({'roles': raw})
        assert repr(raw) in str(caught.value)
