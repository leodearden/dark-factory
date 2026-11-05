"""Small-sample summary statistics for eval reports: a mean with its two-sided
95% Student-t interval, stdlib only.

Neither scipy nor numpy is a declared dependency, so the interval uses stdlib
``statistics`` plus a df→t-critical lookup. Trials are few (decision 10: ≥3 per
cell), so the Student-t interval, not a normal z, is the right estimator, and
the ``sufficient`` flag marks a sample with n<3.
"""

from __future__ import annotations

import statistics
from math import sqrt
from typing import Any

# Two-sided 95% (0.975 one-tail) Student-t critical values, indexed by degrees
# of freedom (n-1). df>30 (or df<1) falls back to the normal approximation 1.96.
_T_CRITICAL_0975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_T_CRITICAL_NORMAL = 1.96


def _t_critical(df: int) -> float:
    """Two-sided 95% Student-t critical value for *df* degrees of freedom,
    falling back to the normal approximation (1.96) for df>30 or df<1.
    """
    if df < 1:
        return _T_CRITICAL_NORMAL
    return _T_CRITICAL_0975.get(df, _T_CRITICAL_NORMAL)


def mean_ci95(values: list[float]) -> dict[str, Any]:
    """Mean and two-sided 95% Student-t confidence interval of *values*.

    Returns ``{mean, lo, hi, n, sufficient}``. A CI is computed for n>=2 (via
    the df→t-critical table); for n<2 the interval collapses to the point
    estimate (``lo == hi == mean``). ``sufficient`` is ``n>=3`` (decision 10:
    ≥3 trials/cell), surfacing when the sample is too small to trust. An empty
    input yields ``mean==0.0, n==0``. Pure, stdlib-only (statistics.stdev is the
    ddof=1 sample stdev).
    """
    n = len(values)
    if n == 0:
        return {'mean': 0.0, 'lo': 0.0, 'hi': 0.0, 'n': 0, 'sufficient': False}
    mean = statistics.mean(values)
    if n < 2:
        return {'mean': mean, 'lo': mean, 'hi': mean, 'n': n, 'sufficient': False}
    stdev = statistics.stdev(values)  # ddof=1 (sample stdev)
    half_width = _t_critical(n - 1) * stdev / sqrt(n)
    return {
        'mean': mean,
        'lo': mean - half_width,
        'hi': mean + half_width,
        'n': n,
        'sufficient': n >= 3,
    }
