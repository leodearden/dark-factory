"""The embedding-axis vector invariant (arm_harness/normalization.py)."""

import math

import pytest
from pydantic import ValidationError

from fused_memory.arm_harness.normalization import (
    NormStats,
    VectorInvariantError,
    norm_stats,
    raw_norm,
    unit_vector,
)

GRANITE_SCALE_NORM = 30.36


def _l2(vector) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (_l2(a) * _l2(b))


def _granite_scale(dim: int = 768) -> list[float]:
    return [GRANITE_SCALE_NORM] + [0.0] * (dim - 1)


@pytest.mark.parametrize(
    ('raw', 'dim'),
    [([3.0, 4.0], 2), (_granite_scale(), 768), ([0.1, -0.2, 0.3, 1e-4], 4)],
    ids=['3-4', 'granite-scale', 'mixed-sign'],
)
def test_unit_vector_is_l2_unit_and_keeps_direction(raw, dim):
    unit = unit_vector(raw, dim)

    assert isinstance(unit, tuple)
    assert len(unit) == dim
    assert abs(_l2(unit) - 1.0) < 1e-9
    assert _cosine(unit, raw) == pytest.approx(1.0, abs=1e-12)


def test_unit_vector_of_three_four_is_three_fifths_four_fifths():
    assert unit_vector([3.0, 4.0], 2) == pytest.approx((0.6, 0.8))


def test_the_invariant_error_is_a_value_error():
    assert issubclass(VectorInvariantError, ValueError)


def test_a_zero_vector_is_refused_naming_the_invariant_and_the_norm():
    with pytest.raises(VectorInvariantError) as caught:
        unit_vector([0.0, 0.0, 0.0], 3)

    assert 'non-zero' in str(caught.value)
    assert '0.0' in str(caught.value)


@pytest.mark.parametrize(
    ('bad', 'shown'), [(math.nan, 'nan'), (math.inf, 'inf'), (-math.inf, '-inf')]
)
def test_a_non_finite_component_is_refused_naming_the_invariant_and_the_value(bad, shown):
    with pytest.raises(VectorInvariantError) as caught:
        unit_vector([0.5, bad, 0.5], 3)

    assert 'finite' in str(caught.value)
    assert shown in str(caught.value)
    assert 'component 1' in str(caught.value)


@pytest.mark.parametrize(('length', 'declared'), [(768, 1024), (1025, 1024), (0, 1024)])
def test_a_vector_of_another_dimension_is_refused_not_padded_or_truncated(length, declared):
    with pytest.raises(VectorInvariantError) as caught:
        unit_vector([0.25] * length, declared)

    message = str(caught.value)
    assert 'dimension' in message
    assert str(length) in message
    assert str(declared) in message


def test_raw_norm_is_the_pre_normalisation_l2_norm():
    assert raw_norm([3.0, 4.0]) == 5.0
    assert raw_norm(_granite_scale()) == pytest.approx(GRANITE_SCALE_NORM)


def test_norm_stats_summarises_the_raw_norms():
    stats = norm_stats([30.0, 37.0, 29.5, 31.0])

    assert stats == NormStats(count=4, min=29.5, median=30.5, max=37.0)


def test_norm_stats_of_one_norm_is_that_norm_throughout():
    assert norm_stats([1.0]) == NormStats(count=1, min=1.0, median=1.0, max=1.0)


def test_norm_stats_are_frozen():
    stats = norm_stats([1.0, 2.0])

    with pytest.raises(ValidationError):
        stats.max = 3.0  # type: ignore[misc]


def test_norm_stats_refuse_an_empty_input():
    with pytest.raises(ValueError, match='empty'):
        norm_stats([])
