"""Contract of shared/src/shared/testing_ceiling_anchors.py."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from shared.testing_ceiling_anchors import (
    MAGICMOCK_REPO_SCAN_TEST_TIMEOUT,
    CeilingAnchor,
    Population,
    PopulationKind,
    registered_anchors,
    required_ceiling_secs,
)
from shared.testing_timeout_headroom import UNDER_LOAD_HEADROOM_FACTOR
from shared.testing_timeout_markers import SiteKind, TimeoutSite
from shared.whole_tree_guard_family import (
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
    WHOLE_TREE_SCAN_TIMEOUT_NAME,
)

_SHARED_PYPROJECT = Path(__file__).resolve().parent.parent / 'pyproject.toml'


def _site(spelling: str, constant: str | None) -> TimeoutSite:
    return TimeoutSite('test_a', SiteKind.DECORATOR, None, 1, spelling, constant)


def _anchor(*, ceiling_secs: float = 600, measured_worst_secs: float = 50.0) -> CeilingAnchor:
    return CeilingAnchor(
        anchor_id='some_anchor',
        ceiling_secs=ceiling_secs,
        measured_worst_secs=measured_worst_secs,
        population=Population(PopulationKind.UNMARKED),
        provenance='pkg/tests/test_x.py::SOME_ANCHOR',
    )


@pytest.mark.parametrize(
    ('worst', 'ceiling'),
    [(148.65, 1200), (550.75, 4440), (61.52, 540), (45.48, 420)],
)
def test_required_ceiling_is_the_worst_times_the_factor_rounded_up_to_a_minute(
    worst: float, ceiling: int
) -> None:
    assert required_ceiling_secs(worst) == ceiling


def test_a_spelled_through_population_admits_only_a_closest_site_naming_its_constant() -> None:
    population = Population(PopulationKind.SPELLED_THROUGH, constant='X')

    assert population.admits(_site('X', 'X'))
    assert population.admits(_site('module.X', 'X'))
    assert not population.admits(_site('Y', 'Y'))
    assert not population.admits(_site('300', None))
    assert not population.admits(_site('', None))
    assert not population.admits(None)


def test_the_unmarked_population_admits_no_mark_and_an_argumentless_mark_only() -> None:
    population = Population(PopulationKind.UNMARKED)

    assert population.admits(None)
    assert population.admits(_site('', None))
    assert not population.admits(_site('300', None))
    assert not population.admits(_site('X', 'X'))


def test_a_spelled_through_population_without_a_constant_is_refused() -> None:
    with pytest.raises(ValueError, match=r'spelled_through.*names the constant.*None'):
        Population(PopulationKind.SPELLED_THROUGH)


def test_an_unmarked_population_naming_a_constant_is_refused() -> None:
    with pytest.raises(ValueError, match=r'unmarked.*names no constant.*X'):
        Population(PopulationKind.UNMARKED, constant='X')


@pytest.mark.parametrize('value', [0, -1.5])
def test_an_anchor_with_a_non_positive_ceiling_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match=rf'positive.*ceiling_secs={value}'):
        _anchor(ceiling_secs=value)


@pytest.mark.parametrize('value', [0, -1.5])
def test_an_anchor_with_a_non_positive_recorded_worst_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match=rf'positive.*measured_worst_secs={value}'):
        _anchor(measured_worst_secs=value)


def test_the_admissible_worst_is_the_ceiling_over_the_headroom_factor() -> None:
    assert _anchor(ceiling_secs=600).admissible_worst_secs == 600 / UNDER_LOAD_HEADROOM_FACTOR


def _by_id() -> dict[str, CeilingAnchor]:
    return {anchor.anchor_id: anchor for anchor in registered_anchors()}


def test_registered_anchor_ids_are_unique() -> None:
    ids = [anchor.anchor_id for anchor in registered_anchors()]

    assert len(ids) == len(set(ids)), ids


def test_the_whole_tree_family_anchor_reads_the_family_ceiling_and_name() -> None:
    anchor = _by_id()['whole_tree_family']

    assert anchor.ceiling_secs == WHOLE_TREE_SCAN_TEST_TIMEOUT
    assert anchor.population == Population(
        PopulationKind.SPELLED_THROUGH, constant=WHOLE_TREE_SCAN_TIMEOUT_NAME
    )


def test_the_bare_magicmock_anchor_is_spelled_through_its_own_ceiling() -> None:
    anchor = _by_id()['bare_magicmock_repo_scan']

    assert anchor.ceiling_secs == MAGICMOCK_REPO_SCAN_TEST_TIMEOUT
    assert anchor.population == Population(
        PopulationKind.SPELLED_THROUGH, constant='MAGICMOCK_REPO_SCAN_TEST_TIMEOUT'
    )


def test_the_ini_default_anchor_bounds_unmarked_tests_at_the_ini_timeout() -> None:
    ini_timeout = tomllib.loads(_SHARED_PYPROJECT.read_text(encoding='utf-8'))['tool']['pytest'][
        'ini_options'
    ]['timeout']
    anchor = _by_id()['ini_default']

    assert anchor.ceiling_secs == ini_timeout
    assert anchor.population == Population(PopulationKind.UNMARKED)


@pytest.mark.parametrize('anchor', registered_anchors(), ids=lambda anchor: anchor.anchor_id)
def test_every_recorded_anchor_still_justifies_its_ceiling(anchor: CeilingAnchor) -> None:
    assert required_ceiling_secs(anchor.measured_worst_secs) <= anchor.ceiling_secs, (
        f'{anchor.anchor_id}: the recorded worst {anchor.measured_worst_secs}s '
        f'({anchor.provenance}) x {UNDER_LOAD_HEADROOM_FACTOR} needs a ceiling of '
        f'{required_ceiling_secs(anchor.measured_worst_secs)}s, above the '
        f'{anchor.ceiling_secs}s in force.'
    )
