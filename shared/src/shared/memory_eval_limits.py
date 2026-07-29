"""The M2 limits evaluator — exact small-sample tests, budget-derived alpha.

``docs/prds/memory-eval-program.md`` §3 M2: *every alarm limit is a calibration
output with recorded provenance — no a-priori numeric threshold appears in code
or any leaf signal (G6). Pre-existing failures are grandfathered; alarms fire on
regressions.*

**Why exact tests rather than the canary's tolerances (D3).** The canary
(``orchestrator/src/orchestrator/evals/prompt_opt/canary.py``) compares windows
against a fixed relative tolerance plus an absolute floor. That shape is wrong
for these metrics: an E1 probe set is ~30 items, where a ratio is dominated by
sampling noise and any fixed tolerance is either deaf or deafening. Exact
binomial/Poisson tests cost microseconds at these n and answer the question
actually being asked — *how surprising is this run under the baseline?* What IS
reused from the canary is its shape, COPIED and never imported (importing
``orchestrator.evals`` from ``fused-memory`` would be a wrong-direction package
edge): frozen-dataclass verdicts, a ``min_samples`` short-circuit to
``insufficient_data``, and always computing the comparison numbers so they stay
inspectable even when the verdict is inconclusive.

**Both tests use the method of small p-values** — sum the probability of every
outcome at most as probable as the one observed — rather than doubling a
one-tail. Doubling is the natural wrong implementation for an asymmetric
distribution and gives materially different answers: for ``k=0, lam=5`` the
correct two-sided p is 0.012191, while ``exp(-5)`` is 0.006738 and
``2*exp(-5)`` is 0.013476. None agree, because the correct method also sweeps
the far right tail. The method also makes the two rule kinds structurally
identical (one summation over a pmf), so binomial and Poisson share a single
reviewed idea rather than two.

Stdlib only — ``math.comb``/``math.lgamma``, no scipy or numpy (neither is a
dependency of ``dark-factory-shared``, following the stdlib-only precedent of
``orchestrator/src/orchestrator/evals/report.py``).
"""

from __future__ import annotations

import math

__all__ = [
    'binomial_two_sided_p',
]

_PMF_TIE_SLACK = 1e-9
"""Relative slack when asking "is this outcome at most as probable as the observed one?".

A float-representation allowance, not a calibrated threshold. The mirrored
outcome of a symmetric case has the same pmf in exact arithmetic but can differ
in the last ulp once computed; without the slack it would be dropped from the
acceptance set and the p-value would silently lose a whole tail.
"""


def _binomial_pmf(i: int, n: int, p0: float) -> float:
    """P(X = i) for X ~ Binomial(n, p0), with the degenerate p0 handled first.

    ``0.0 ** 0`` is 1.0 in Python, so the general expression is already safe at
    ``p0 in {0.0, 1.0}``; it is spelled out anyway because a degenerate H0 is a
    real input here (a tripwire-adjacent proportion can legitimately have a
    baseline of all-pass or all-fail) and silently relying on that corner of the
    float spec would be a trap for the next reader.
    """
    if p0 == 0.0:
        return 1.0 if i == 0 else 0.0
    if p0 == 1.0:
        return 1.0 if i == n else 0.0
    return math.comb(n, i) * (p0**i) * ((1.0 - p0) ** (n - i))


def binomial_two_sided_p(k: int, n: int, p0: float) -> float:
    """Exact two-sided binomial p-value for *k* successes in *n* trials under *p0*.

    Rule (b)'s engine: how surprising is the current run's proportion, given the
    proportion pooled over the trailing baseline window?

    Computed by the method of small p-values — the total probability of every
    outcome no more probable than the observed one. For an observation AT the
    mode this is exactly 1.0 (every outcome qualifies), which is the identity
    the tests anchor on.

    A degenerate ``p0`` of 0 or 1 is answered rather than rejected: the outcome
    H0 makes certain has p == 1.0, and one H0 calls impossible has p == 0.0.
    Raises ``ValueError`` for a negative *n*, a *k* outside ``[0, n]``, or a
    *p0* outside ``[0, 1]`` — those are caller bugs, not data.
    """
    if n < 0:
        raise ValueError(f'binomial_two_sided_p: n={n} must be non-negative.')
    if not 0 <= k <= n:
        raise ValueError(f'binomial_two_sided_p: k={k} must lie in [0, {n}].')
    if not 0.0 <= p0 <= 1.0:
        raise ValueError(f'binomial_two_sided_p: p0={p0} must lie in [0, 1].')

    pmf = [_binomial_pmf(i, n, p0) for i in range(n + 1)]
    observed = pmf[k]
    total = math.fsum(p for p in pmf if p <= observed * (1.0 + _PMF_TIE_SLACK))
    return min(1.0, total)
