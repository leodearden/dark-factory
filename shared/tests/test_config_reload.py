"""Tests for shared.config_reload: leaf iteration, diff and apply between two configs of one class."""

from enum import StrEnum
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.config_fields import Reload, RestartOnlyRows, reloadable_paths
from shared.config_reload import ABSENT, Change, diff_config, iter_leaves


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
    max_slots: int = 4
    roles: Annotated[
        dict[Role, Row], Reload.GREEN, RestartOnlyRows(frozenset({'reviewer'}))
    ] = Field(default_factory=_every_row)


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
