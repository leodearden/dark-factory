"""Tests for shared.config_reload: leaf iteration, diff and apply between two configs of one class."""

import json
from enum import StrEnum
from typing import Annotated, Self

import pytest
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.config_fields import DeprecatedSpelling, Reload, RestartOnlyRows, reloadable_paths
from shared.config_reload import ABSENT, Change, apply_reload, diff_config, iter_leaves


class Role(StrEnum):
    ARCHITECT = 'architect'
    REVIEWER = 'reviewer'
    UNBLOCK_AUTO = 'unblock_auto'


class Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model: str = 'sonnet'
    budget_usd: float = 1.0
    max_turns: int = 10


class Sub(BaseModel):
    alpha: int = 1
    beta: str = 'b'


class MixedSub(BaseModel):
    tuned: Annotated[int, Reload.GREEN] = 1
    fixed: int = 1


class FusedShaped(BaseModel):
    opt: Sub | None = None
    opt_green: Annotated[Sub | None, Reload.GREEN] = None
    opt_mixed: MixedSub | None = None
    populated: Sub | None = Field(default_factory=Sub)
    weights: Annotated[dict[str, int], Reload.GREEN] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class Timeouts(BaseModel):
    steward: int = 60
    verify: int = 600


class Inner(BaseModel):
    depth: int = 2


class Outer(BaseModel):
    inner: Inner = Field(default_factory=Inner)
    knob: int = 1


class Bounds(BaseModel):
    low: Annotated[int, Reload.GREEN] = 1
    high: int = 10


class FrozenLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    cap: int = 5


EXPLODING_STEWARD = 13


def _every_row() -> dict[Role, Row]:
    return {role: Row() for role in Role}


def _rows_with(role: Role, row: Row) -> dict[Role, Row]:
    return {**_every_row(), role: row}


class OrchestratorShaped(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='SHARED_RELOAD_FIXTURE_', extra='ignore', validate_assignment=True
    )

    timeouts: Annotated[Timeouts, Reload.GREEN] = Field(default_factory=Timeouts)
    nested: Annotated[Outer, Reload.GREEN] = Field(default_factory=Outer)
    bounds: Bounds = Field(default_factory=Bounds)
    limits: Annotated[FrozenLimits, Reload.GREEN] = Field(default_factory=FrozenLimits)
    max_slots: int = 4
    roles: Annotated[
        dict[Role, Row], Reload.GREEN, RestartOnlyRows(frozenset({'reviewer'}))
    ] = Field(default_factory=_every_row)

    @model_validator(mode='after')
    def _bounds_ordered(self) -> Self:
        if self.timeouts.steward == EXPLODING_STEWARD:
            raise RuntimeError('fixture validator exploded')
        if self.bounds.low > self.bounds.high:
            raise ValueError(
                f'bounds.low ({self.bounds.low}) must not exceed bounds.high ({self.bounds.high})'
            )
        return self


def _leaf_paths(model_cls, model) -> list[str]:
    return [path for path, _ in iter_leaves(model_cls, model)]


def _diff(live, fresh):
    return diff_config(live, fresh, reloadable_paths(type(live)))


def _assign_item(container, key, value) -> None:
    container[key] = value


def test_the_leaf_paths_depend_only_on_the_class():
    assert _leaf_paths(FusedShaped, FusedShaped()) == _leaf_paths(
        FusedShaped, FusedShaped(opt=Sub())
    )
    assert _leaf_paths(FusedShaped, None) == _leaf_paths(FusedShaped, FusedShaped())


def test_every_leaf_under_a_none_submodel_is_absent():
    leaves = dict(iter_leaves(FusedShaped, FusedShaped()))
    assert leaves['opt.alpha'] is ABSENT
    assert leaves['opt.beta'] is ABSENT
    assert leaves['populated.alpha'] == 1


def test_every_leaf_of_a_missing_table_row_is_absent():
    leaves = dict(iter_leaves(OrchestratorShaped, OrchestratorShaped(roles={Role.ARCHITECT: Row()})))
    assert leaves['roles.architect.model'] == 'sonnet'
    assert leaves['roles.reviewer.model'] is ABSENT
    assert leaves['roles.unblock_auto.max_turns'] is ABSENT


def test_atomic_dicts_and_lists_are_yielded_whole():
    leaves = dict(iter_leaves(FusedShaped, FusedShaped(weights={'a': 1}, tags=['x'])))
    assert leaves['weights'] == {'a': 1}
    assert leaves['tags'] == ['x']
    assert not [path for path in leaves if path.startswith(('weights.', 'tags.'))]


def test_none_to_populated_on_an_unmarked_optional_is_restart_required_leaf_by_leaf():
    diff = _diff(FusedShaped(), FusedShaped(opt=Sub(alpha=5)))
    assert dict(diff.applied_candidates) == {}
    assert dict(diff.restart_required) == {
        'opt.alpha': Change(ABSENT, 5),
        'opt.beta': Change(ABSENT, 'b'),
    }


def test_populated_to_none_on_an_unmarked_optional_is_restart_required():
    diff = _diff(FusedShaped(), FusedShaped(populated=None))
    assert dict(diff.restart_required) == {
        'populated.alpha': Change(1, ABSENT),
        'populated.beta': Change('b', ABSENT),
    }


def test_none_to_populated_on_a_green_optional_is_an_applied_candidate():
    diff = _diff(FusedShaped(), FusedShaped(opt_green=Sub(beta='z')))
    assert dict(diff.restart_required) == {}
    assert dict(diff.applied_candidates) == {
        'opt_green.alpha': Change(ABSENT, 1),
        'opt_green.beta': Change(ABSENT, 'z'),
    }


def test_a_toggle_over_mixed_tiers_is_restart_required_as_a_whole():
    diff = _diff(FusedShaped(), FusedShaped(opt_mixed=MixedSub(tuned=2)))
    assert dict(diff.applied_candidates) == {}
    assert set(diff.restart_required) == {'opt_mixed.tuned', 'opt_mixed.fixed'}


def test_a_green_leaf_two_levels_deep_is_an_applied_candidate():
    diff = _diff(OrchestratorShaped(), OrchestratorShaped(nested=Outer(inner=Inner(depth=3))))
    assert dict(diff.applied_candidates) == {'nested.inner.depth': Change(2, 3)}


def test_an_unmarked_root_scalar_is_restart_required():
    diff = _diff(OrchestratorShaped(), OrchestratorShaped(max_slots=8))
    assert dict(diff.restart_required) == {'max_slots': Change(4, 8)}


def test_a_knob_in_a_green_row_is_an_applied_candidate():
    fresh = OrchestratorShaped(roles=_rows_with(Role.ARCHITECT, Row(model='opus')))
    diff = _diff(OrchestratorShaped(), fresh)
    assert dict(diff.applied_candidates) == {'roles.architect.model': Change('sonnet', 'opus')}
    assert dict(diff.restart_required) == {}


def test_a_knob_in_a_restart_only_row_is_restart_required():
    fresh = OrchestratorShaped(roles=_rows_with(Role.REVIEWER, Row(model='opus')))
    diff = _diff(OrchestratorShaped(), fresh)
    assert dict(diff.applied_candidates) == {}
    assert dict(diff.restart_required) == {'roles.reviewer.model': Change('sonnet', 'opus')}


def test_unchanged_counts_the_equal_leaves():
    live = OrchestratorShaped()
    diff = _diff(live, OrchestratorShaped(max_slots=8, timeouts=Timeouts(steward=61)))
    assert diff.unchanged == len(_leaf_paths(OrchestratorShaped, live)) - 2


def test_the_diff_maps_are_read_only():
    diff = _diff(OrchestratorShaped(), OrchestratorShaped(max_slots=8))
    with pytest.raises(TypeError):
        _assign_item(diff.restart_required, 'max_slots', Change(4, 9))


def test_diffing_two_classes_is_a_type_error_naming_both():
    with pytest.raises(TypeError) as raised:
        diff_config(FusedShaped(), OrchestratorShaped(), frozenset())
    assert 'FusedShaped' in str(raised.value)
    assert 'OrchestratorShaped' in str(raised.value)


def test_a_green_leaf_is_applied_in_place_and_a_held_submodel_reads_it():
    live = OrchestratorShaped()
    held = live.timeouts
    report = apply_reload(live, OrchestratorShaped(timeouts=Timeouts(steward=90)))
    assert report['reloaded'] is True
    assert report['error'] is None
    assert report['applied'] == {'timeouts.steward': {'old': 60, 'new': 90}}
    assert live.timeouts is held
    assert held.steward == 90


def test_an_unmarked_root_change_is_reported_and_not_written():
    live = OrchestratorShaped()
    report = apply_reload(live, OrchestratorShaped(max_slots=8))
    assert report['applied'] == {}
    assert report['restart_required'] == {'max_slots': {'old': 4, 'new': 8}}
    assert live.max_slots == 4


def test_the_report_has_exactly_the_engine_keys():
    report = apply_reload(OrchestratorShaped(), OrchestratorShaped(max_slots=8))
    assert set(report) == {
        'reloaded',
        'applied',
        'restart_required',
        'unchanged',
        'error',
        'deprecated',
    }


def test_a_hybrid_that_breaks_a_cross_field_invariant_leaves_live_as_it_was():
    live = OrchestratorShaped()
    held = live.bounds
    before = live.model_dump()
    report = apply_reload(live, OrchestratorShaped(bounds=Bounds(low=15, high=20)))
    assert report['reloaded'] is False
    assert report['applied'] == {}
    assert report['restart_required'] == {'bounds.high': {'old': 10, 'new': 20}}
    error = report['error']
    assert error is not None
    assert error.startswith('hybrid-invariant:')
    assert 'bounds.low' in error
    assert live.model_dump() == before
    assert live.bounds is held


def test_a_green_optional_round_trips_none_to_populated_and_back():
    live = FusedShaped()
    start = live.model_dump()
    populated = FusedShaped(opt_green=Sub(alpha=7))
    first = apply_reload(live, populated)
    assert first['reloaded'] is True
    assert first['applied'] == {
        'opt_green.alpha': {'old': '<absent>', 'new': 7},
        'opt_green.beta': {'old': '<absent>', 'new': 'b'},
    }
    json.dumps(first)
    assert live.opt_green == populated.opt_green
    assert live.opt_green is not populated.opt_green
    second = apply_reload(live, FusedShaped())
    assert second['reloaded'] is True
    assert live.opt_green is None
    assert live.model_dump() == start


@pytest.mark.parametrize(
    ('live_opt', 'fresh_opt'),
    [(None, Sub()), (Sub(), None)],
    ids=['none-to-populated', 'populated-to-none'],
)
def test_a_toggle_on_an_unmarked_optional_writes_nothing(live_opt, fresh_opt):
    live = FusedShaped(opt=live_opt)
    before = live.model_dump()
    report = apply_reload(live, FusedShaped(opt=fresh_opt))
    assert report['applied'] == {}
    assert set(report['restart_required']) == {'opt.alpha', 'opt.beta'}
    assert live.model_dump() == before


def test_a_frozen_green_submodel_is_replaced_never_mutated():
    live = OrchestratorShaped()
    held = live.limits
    apply_reload(live, OrchestratorShaped(limits=FrozenLimits(cap=6)))
    assert live.limits.cap == 6
    assert held.cap == 5


def test_a_green_row_knob_swaps_the_frozen_row_inside_the_same_table():
    live = OrchestratorShaped()
    table = live.roles
    architect_before = table[Role.ARCHITECT]
    others = {role: table[role] for role in Role if role is not Role.ARCHITECT}
    report = apply_reload(
        live, OrchestratorShaped(roles=_rows_with(Role.ARCHITECT, Row(model='opus')))
    )
    assert report['applied'] == {'roles.architect.model': {'old': 'sonnet', 'new': 'opus'}}
    assert live.roles is table
    assert table[Role.ARCHITECT].model == 'opus'
    assert architect_before.model == 'sonnet'
    assert all(table[role] is row for role, row in others.items())


def test_a_restart_only_row_change_is_reported_and_not_written():
    live = OrchestratorShaped()
    report = apply_reload(
        live, OrchestratorShaped(roles=_rows_with(Role.REVIEWER, Row(model='opus')))
    )
    assert report['applied'] == {}
    assert report['restart_required'] == {
        'roles.reviewer.model': {'old': 'sonnet', 'new': 'opus'}
    }
    assert live.roles[Role.REVIEWER].model == 'sonnet'


def test_a_row_missing_on_live_is_inserted_by_enum_member_and_removed_again():
    live = OrchestratorShaped(roles={Role.ARCHITECT: Row()})
    table = live.roles
    apply_reload(
        live,
        OrchestratorShaped(roles={Role.ARCHITECT: Row(), Role.UNBLOCK_AUTO: Row(model='opus')}),
    )
    assert live.roles is table
    assert table[Role.UNBLOCK_AUTO].model == 'opus'
    assert [type(key) for key in table] == [Role, Role]
    apply_reload(live, OrchestratorShaped(roles={Role.ARCHITECT: Row()}))
    assert list(table) == [Role.ARCHITECT]


def test_an_explicit_reloadable_set_overrides_the_markers():
    live = OrchestratorShaped()
    report = apply_reload(
        live,
        OrchestratorShaped(max_slots=8, timeouts=Timeouts(steward=90)),
        reloadable=frozenset({'max_slots'}),
    )
    assert report['applied'] == {'max_slots': {'old': 4, 'new': 8}}
    assert report['restart_required'] == {'timeouts.steward': {'old': 60, 'new': 90}}
    assert (live.max_slots, live.timeouts.steward) == (8, 60)


def test_caller_supplied_deprecations_are_rendered_and_default_to_none():
    moved = DeprecatedSpelling('a', 'b.a')
    with_moves = apply_reload(OrchestratorShaped(), OrchestratorShaped(), deprecated=(moved,))
    assert with_moves['deprecated'] == [{'old_path': 'a', 'new_path': 'b.a'}]
    assert apply_reload(OrchestratorShaped(), OrchestratorShaped())['deprecated'] == []


def test_a_non_value_error_from_revalidation_is_reraised_after_rollback():
    live = OrchestratorShaped()
    held = live.timeouts
    before = live.model_dump()
    fresh = live.model_copy(update={'timeouts': Timeouts(steward=EXPLODING_STEWARD)}, deep=True)
    with pytest.raises(RuntimeError, match='exploded'):
        apply_reload(live, fresh)
    assert live.model_dump() == before
    assert live.timeouts is held
    assert held.steward == 60
