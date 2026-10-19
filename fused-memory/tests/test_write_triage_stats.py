"""Tests for write_triage_stats.py — the write-triage scorers' shared interval statistics.

The script is loaded by path (``scripts/`` is not a package), lazily through
``_mod()``. The Wilson and kappa expectations are literal oracle values,
computed independently with the textbook formulas (z = 1.96, four places).
"""
from __future__ import annotations

import functools
import types
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'write_triage_stats.py'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='write_triage_stats')


def _unit(stratum: str, numerator: float, denominator: float = 1.0) -> object:
    return _mod().StratumUnit(stratum=stratum, numerator=numerator, denominator=denominator)


class TestWilson95:
    @pytest.mark.parametrize(('hits', 'total', 'expected'), [
        (19, 38, [0.3485, 0.6515]),
        (0, 10, [0.0, 0.2775]),
        (52, 372, [0.1082, 0.1787]),
        (10, 10, [0.7225, 1.0]),
    ])
    def test_matches_the_textbook_interval(
        self, hits: int, total: int, expected: list[float],
    ) -> None:
        assert _mod().wilson95(hits, total) == expected

    def test_an_empty_sample_has_no_interval(self) -> None:
        assert _mod().wilson95(0, 0) is None


class TestNearestRank:
    @pytest.mark.parametrize(('percent', 'expected'), [(50, 2), (95, 4), (100, 4), (25, 1)])
    def test_is_the_smallest_value_with_percent_at_or_below_it(
        self, percent: float, expected: int,
    ) -> None:
        assert _mod().nearest_rank([1, 2, 3, 4], percent) == expected

    def test_percent_zero_is_the_smallest_value(self) -> None:
        assert _mod().nearest_rank([1, 2, 3, 4], 0) == 1


class TestCohenKappa:
    def test_matches_the_textbook_kappa(self) -> None:
        pairs = (
            [(True, True)] * 20 + [(True, False)] * 5
            + [(False, True)] * 10 + [(False, False)] * 15
        )
        assert _mod().cohen_kappa(pairs) == 0.4

    def test_perfect_agreement_on_mixed_labels_is_one(self) -> None:
        assert _mod().cohen_kappa([(True, True), (False, False), (True, True)]) == 1.0

    def test_no_pairs_has_no_kappa(self) -> None:
        assert _mod().cohen_kappa([]) is None

    def test_a_single_shared_label_has_no_kappa(self) -> None:
        assert _mod().cohen_kappa([(True, True)] * 7) is None


class TestStratifiedRatio:
    def test_weights_each_stratum_by_its_weight(self) -> None:
        units = [_unit('a', 1), _unit('a', 0), _unit('b', 1), _unit('b', 1)]
        # (3 * 1 + 1 * 2) / (3 * 2 + 1 * 2)
        assert _mod().stratified_ratio(units, {'a': 3.0, 'b': 1.0}) == 0.625

    def test_a_zero_weighted_denominator_has_no_ratio(self) -> None:
        assert _mod().stratified_ratio([_unit('a', 0, 0)], {'a': 1.0}) is None

    def test_no_units_has_no_ratio(self) -> None:
        assert _mod().stratified_ratio([], {'a': 1.0}) is None

    def test_a_stratum_without_a_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match='z'):
            _mod().stratified_ratio([_unit('z', 1)], {'a': 1.0})


class TestStratifiedBootstrap95:
    _UNITS = (
        [('a', 1), ('a', 1), ('a', 0), ('a', 1)]
        + [('b', 0), ('b', 0), ('b', 1), ('b', 0)]
    )

    def _units(self) -> list[object]:
        return [_unit(stratum, num) for stratum, num in self._UNITS]

    def test_the_same_seed_gives_the_same_interval(self) -> None:
        stats = _mod()
        first = stats.stratified_bootstrap95(
            self._units(), {'a': 1.0, 'b': 1.0}, resamples=500, seed=11,
        )
        second = stats.stratified_bootstrap95(
            self._units(), {'a': 1.0, 'b': 1.0}, resamples=500, seed=11,
        )
        assert first == second
        assert first is not None and first[0] <= first[1]

    def test_identical_units_give_a_point_interval(self) -> None:
        units = [_unit('a', 1, 2) for _ in range(5)] + [_unit('b', 2, 4) for _ in range(3)]
        assert _mod().stratified_bootstrap95(
            units, {'a': 1.0, 'b': 4.0}, resamples=200, seed=1,
        ) == [0.5, 0.5]

    def test_a_heavier_stratum_pulls_the_interval_towards_it(self) -> None:
        stats = _mod()
        even = stats.stratified_bootstrap95(
            self._units(), {'a': 1.0, 'b': 1.0}, resamples=2000, seed=3,
        )
        heavy_a = stats.stratified_bootstrap95(
            self._units(), {'a': 10.0, 'b': 1.0}, resamples=2000, seed=3,
        )
        assert even is not None and heavy_a is not None
        assert heavy_a[0] > even[0]
        assert heavy_a[1] > even[1]

    def test_resampling_stays_within_each_stratum(self) -> None:
        units = [_unit('a', 1) for _ in range(4)] + [_unit('b', 0) for _ in range(4)]
        assert _mod().stratified_bootstrap95(
            units, {'a': 1.0, 'b': 3.0}, resamples=300, seed=5,
        ) == [0.25, 0.25]

    def test_no_units_has_no_interval(self) -> None:
        assert _mod().stratified_bootstrap95([], {'a': 1.0}, resamples=100, seed=1) is None

    def test_a_zero_weighted_denominator_in_every_draw_has_no_interval(self) -> None:
        units = [_unit('a', 0, 0), _unit('a', 0, 0)]
        assert _mod().stratified_bootstrap95(
            units, {'a': 1.0}, resamples=100, seed=1,
        ) is None
