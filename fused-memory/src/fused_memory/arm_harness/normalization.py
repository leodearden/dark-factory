"""Every embedding-axis vector is finite, non-zero, of its declared dimension and L2-unit; both stores are true-cosine, so normalising is a ranking-neutral guard (plans/local-memory-models-eval-embedding-report.md §Normalisation handling)."""

import math
import statistics
from collections.abc import Sequence

from fused_memory.arm_harness.frozen_model import FrozenModel


class VectorInvariantError(ValueError):
    """A vector broke the invariant, so it is refused rather than padded, truncated or repaired."""


class NormStats(FrozenModel):
    count: int
    min: float
    median: float
    max: float


def raw_norm(raw: Sequence[float]) -> float:
    return math.hypot(*raw)


def unit_vector(raw: Sequence[float], expected_dim: int) -> tuple[float, ...]:
    if len(raw) != expected_dim:
        raise VectorInvariantError(
            f'vector has dimension {len(raw)}, not the declared dimension {expected_dim}'
        )
    for index, component in enumerate(raw):
        if not math.isfinite(component):
            raise VectorInvariantError(f'vector is not finite: component {index} is {component!r}')
    norm = raw_norm(raw)
    if norm == 0.0:
        raise VectorInvariantError(f'vector is not non-zero: its L2 norm is {norm!r}')
    return tuple(component / norm for component in raw)


def norm_stats(norms: Sequence[float]) -> NormStats:
    if not norms:
        raise ValueError('norm_stats of an empty sequence: there is no norm to summarise')
    return NormStats(
        count=len(norms), min=min(norms), median=statistics.median(norms), max=max(norms)
    )
