"""Tests for shared.cgroup_slice's per-project slice name rule: the builder and its recogniser."""

from __future__ import annotations

import pytest

from shared.cgroup_slice import is_project_slice_name, project_slice_name


@pytest.mark.parametrize(
    'project_id, expected',
    [
        ('reify', 'df-reify.slice'),
        ('dark_factory', 'df-dark_factory.slice'),
    ],
)
def test_live_ids_name_their_slice_verbatim(project_id: str, expected: str) -> None:
    assert project_slice_name(project_id) == expected


@pytest.mark.parametrize('project_id', ['my-solar-challenge', 'My-Solar-Challenge'])
def test_dashed_or_mixed_case_id_is_flattened_to_one_slice_level(project_id: str) -> None:
    name = project_slice_name(project_id)

    assert name == 'df-my_solar_challenge.slice'
    assert name.count('-') == 1


@pytest.mark.parametrize('char', ['.', ':', '\\', ' ', 'é'])
def test_any_other_character_outside_the_alphabet_maps_to_underscore(char: str) -> None:
    assert project_slice_name(f'a{char}b') == 'df-a_b.slice'


def test_empty_id_raises_naming_the_invariant() -> None:
    with pytest.raises(ValueError, match="project id ''"):
        project_slice_name('')


@pytest.mark.parametrize(
    'project_id', ['reify', 'dark_factory', 'my-solar-challenge', 'My-Solar-Challenge', 'a.b', 'a:b', 'é', '-']
)
def test_every_built_slice_name_is_recognised(project_id: str) -> None:
    assert is_project_slice_name(project_slice_name(project_id))


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('df-reify-governed.slice', id='governed-sub-slice'),
        pytest.param('df-reify-governed-agents.slice', id='governed-agents-sub-slice'),
        pytest.param('df.slice', id='parent-slice'),
        pytest.param('df-.slice', id='empty-id'),
        pytest.param('df-X.slice', id='upper-case-outside-alphabet'),
        pytest.param('DF-x.slice', id='upper-case-prefix'),
        pytest.param('df-x.scope', id='scope-suffix'),
        pytest.param('df-verify-x-1234abcd-0123456789ab.scope', id='verify-scope'),
        pytest.param('reify.slice', id='reify-legacy-slice'),
        pytest.param('xdf-x.slice', id='unanchored-start'),
        pytest.param('df-x.slice.d', id='unanchored-end'),
    ],
)
def test_names_the_rule_never_builds_are_rejected(name: str) -> None:
    assert not is_project_slice_name(name)
