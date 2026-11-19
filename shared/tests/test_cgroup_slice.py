"""Tests for shared.cgroup_slice.project_slice_name, the per-project slice name rule."""

from __future__ import annotations

import pytest

from shared.cgroup_slice import project_slice_name


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
