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

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from shared.memory_eval_metrics import Metric

__all__ = [
    'Alarm',
    'LimitsConfig',
    'MetricVerdict',
    'RuleKind',
    'VerdictStatus',
    'binomial_two_sided_p',
    'derive_alpha',
    'evaluate_tripwire',
    'grandfather_set_hash',
    'poisson_two_sided_p',
]

RuleKind = Literal['tripwire', 'proportion', 'count', 'scalar']
"""Which rule judged a metric. Deliberately the SAME vocabulary as ``MetricKind``.

A second set of names for the same four things ("rule (a)", "structural") would
be one more mapping for a reader of the artifact to hold, with no gain: the
metric's kind fully determines its rule.
"""

VerdictStatus = Literal['baseline_snapshot', 'ok', 'alarm', 'insufficient_data']
"""The four things a run can say about a metric.

Only ``alarm`` wakes anyone. ``insufficient_data`` in particular is a REPORT
STATUS, never an alarm — a thin sample is not evidence of a regression, and
saying so is the canary's ``min_samples`` precedent.
"""

_PMF_TIE_SLACK = 1e-9
"""Relative slack when asking "is this outcome at most as probable as the observed one?".

A float-representation allowance, not a calibrated threshold. Exact ties are
routine here and each one is a whole chunk of mass: the mirrored outcome of a
symmetric binomial, and — for an integer ``lam`` — the Poisson pmf plateau
where ``pmf(lam - 1) == pmf(lam)``. Both are equal in exact arithmetic but can
differ in the last ulp once computed in floating point, and without the slack
the tie would be dropped from the acceptance set and the p-value would silently
lose a whole tail.
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


def _poisson_pmf(i: int, lam: float) -> float:
    """P(X = i) for X ~ Poisson(lam), computed in log space.

    ``lam**i / i!`` overflows both ways well inside the range of counts this
    module sees, so the pmf is assembled as ``exp(i*log(lam) - lam -
    lgamma(i+1))`` instead. ``math.exp`` underflows silently to 0.0 for a
    sufficiently negative argument (it only raises on overflow), which is what
    makes the far right tail terminate structurally rather than by a horizon
    constant. Caller guarantees ``lam > 0``.
    """
    return math.exp(i * math.log(lam) - lam - math.lgamma(i + 1))


def poisson_two_sided_p(k: int, lam: float) -> float:
    """Exact two-sided Poisson p-value for *k* events under rate *lam*.

    Rule (c)'s engine: how surprising is the current run's count, given the
    mean rate over the trailing baseline window?

    Same method of small p-values as :func:`binomial_two_sided_p` — the total
    probability of every outcome no more probable than the observed one — so
    the two rule kinds share one reviewed idea. Note this is emphatically NOT a
    doubled one-tail: at ``k=0, lam=5`` the correct answer is 0.012191 because
    the sum also sweeps the far right tail, whereas ``exp(-5)`` is 0.006738 and
    twice it is 0.013476.

    Termination is the one thing the Poisson support makes interesting: it is
    unbounded on the right, so there is no support to enumerate. Rather than
    invent a horizon constant — which would be exactly the a-priori numeric
    threshold G6 forbids — the right tail is walked outward from the mode and
    stopped on two structural conditions: the pmf underflowing float64 to
    literal zero, or a term too small to change the running sum at double
    precision. Terms past that point cannot alter the result.

    A degenerate ``lam`` of 0 is answered rather than rejected (a baseline
    window of all-zero counts is a real input — a probe that has never fired):
    zero events is certain, any other count is impossible. Raises
    ``ValueError`` for a negative *k* or *lam* — those are caller bugs, not
    data.
    """
    if k < 0:
        raise ValueError(f'poisson_two_sided_p: k={k} must be non-negative.')
    if lam < 0:
        raise ValueError(f'poisson_two_sided_p: lam={lam} must be non-negative.')
    if lam == 0:
        return 1.0 if k == 0 else 0.0

    threshold = _poisson_pmf(k, lam) * (1.0 + _PMF_TIE_SLACK)
    mode = int(lam)
    terms: list[float] = []

    # Left shoulder. The pmf is non-decreasing on [0, mode], so the qualifying
    # outcomes form a contiguous prefix and the first failure ends it.
    for i in range(mode + 1):
        left = _poisson_pmf(i, lam)
        if left > threshold:
            break
        terms.append(left)

    # Right tail, walked outward from the mode. The pmf is non-increasing here,
    # so once an outcome qualifies every later one does too; the skip phase
    # ends and the accumulate phase begins at the same crossing. `running` is a
    # plain sum used ONLY to detect negligibility — the returned total is an
    # fsum over the collected terms, so the adaptive stop costs no accuracy.
    running = 0.0
    i = mode + 1
    while True:
        right = _poisson_pmf(i, lam)
        if right == 0.0:
            break  # the pmf has underflowed float64: nothing representable remains
        if right <= threshold:
            if running > 0.0 and running + right == running:
                break  # and neither can any later term, since they only shrink
            terms.append(right)
            running += right
        i += 1

    return min(1.0, math.fsum(terms))


@dataclass(frozen=True, kw_only=True)
class LimitsConfig:
    """The declared inputs an operator actually reasons about.

    Deliberately NOT a home for a significance threshold. ``false_alarm_budget``
    is a BUDGET DECLARATION in units a human can hold — *expected false alarms
    per quarter* — and alpha is derived from it by :func:`derive_alpha` (PRD M2,
    D3, G6). Nothing here is a p-value, and alpha is not stored: it depends on
    how many metrics are alarm-eligible in a given run, which this config
    cannot know.

    ``min_samples`` and ``baseline_window`` are SUFFICIENCY guards on the canary
    precedent (``canary.py`` ``min_samples``), not thresholds on the statistic:
    they decide whether there is enough data to say anything at all, and a run
    that fails them reports ``insufficient_data`` rather than any verdict.
    Keyword-only so a four-number call site can never silently transpose two of
    them.
    """

    false_alarm_budget: float = 1.0
    """Expected false alarms per quarter across the whole eval. PRD-sanctioned default."""

    runs_per_quarter: int
    """How often this eval runs — 90 for the D10 daily cadence."""

    min_samples: int
    """Fewest items in the current run before a verdict is meaningful."""

    baseline_window: int
    """How many trailing runs pool into the baseline the current run is judged against."""


def derive_alpha(
    false_alarm_budget: float, runs_per_quarter: int, alarmed_metric_count: int
) -> float:
    """Derive the per-metric significance level from the declared budget (G6).

    ``alpha = budget / (runs_per_quarter * alarmed_metric_count)`` — a
    Bonferroni split of one quarterly budget across every opportunity to spend
    it. Each run of each alarm-eligible metric is one such opportunity, so at
    this alpha the expected number of false alarms per quarter is exactly the
    declared budget.

    The consequence worth stating: alpha SHRINKS as metrics are added. That is
    intended, and it is why the evaluator recomputes alpha per run from the
    metrics actually present rather than resolving it once at authoring time —
    a new metric must not quietly raise the whole eval's false-alarm rate.

    Raises ``ValueError`` on a non-positive budget or a non-positive divisor.
    Zero is not "be infinitely strict": no finite alpha admits zero false
    alarms, and an alpha of 0.0 or infinity would silently turn every run into
    either never-an-alarm or always-one. Fail loudly instead.
    """
    if false_alarm_budget <= 0:
        raise ValueError(
            f'derive_alpha: false_alarm_budget={false_alarm_budget} must be positive '
            '(a budget of zero admits no finite alpha).'
        )
    if runs_per_quarter <= 0:
        raise ValueError(f'derive_alpha: runs_per_quarter={runs_per_quarter} must be positive.')
    if alarmed_metric_count <= 0:
        raise ValueError(
            f'derive_alpha: alarmed_metric_count={alarmed_metric_count} must be positive '
            '(nothing can alarm, so there is no alpha to derive).'
        )
    return false_alarm_budget / (runs_per_quarter * alarmed_metric_count)


@dataclass(frozen=True)
class Alarm:
    """One thing worth waking someone for, with the numbers that justify it.

    Never a bare boolean: whatever made this fire — the observed value, the
    baseline it was judged against, and for a statistical rule the p-value and
    the alpha it was compared to — travels attached to it, because the alarm
    lands in an artifact that a human reads hours later with no other context.
    """

    metric_id: str
    rule_kind: RuleKind
    detail: str
    item_key: str | None = None
    """The specific probe that regressed, for rule (a). ``None`` for whole-metric rules."""
    value: float | None = None
    baseline: float | None = None
    p_value: float | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class MetricVerdict:
    """What this run concluded about one metric.

    Following the canary (``canary.py`` ``MetricComparison``), the underlying
    numbers are ALWAYS computed and exposed — including when the status is
    ``ok`` or ``insufficient_data``. A verdict that only said "fine" would make
    the interesting question ("fine, but how close was it?") unanswerable
    without re-running the eval.
    """

    metric_id: str
    rule_kind: RuleKind
    status: VerdictStatus
    value: float
    n: int
    alarms: tuple[Alarm, ...] = ()
    baseline: float | None = None
    """The comparison point: pooled baseline rate/proportion, or grandfathered failure count."""
    p_value: float | None = None
    """``None`` for structural rules, which measure a fact rather than a surprise."""
    alpha: float | None = None
    """The derived bar this run's p-value was judged against. Provenance, not config."""
    baseline_run_stamps: tuple[str, ...] = ()
    """Exactly which runs produced ``baseline`` — so a verdict can be re-derived by hand."""
    detail: str = ''


def grandfather_set_hash(keys: Iterable[str]) -> str:
    """Stable digest of a grandfather set: sha256 over the sorted, newline-joined keys.

    Sorted and de-duplicated first, so the digest is a property of the SET and
    not of whatever order a caller happened to iterate in. That is what lets a
    reader diff two runs' artifacts and conclude "the known-bad list did not
    move" from one line instead of comparing two lists by eye.
    """
    joined = '\n'.join(sorted(set(keys)))
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()


def evaluate_tripwire(
    metric: Metric, grandfather: frozenset[str] | None
) -> tuple[MetricVerdict, frozenset[str]]:
    """Rule (a): grandfather pre-existing failures, alarm on regressions (D1).

    Returns the verdict and the NEXT grandfather set, which the caller persists.

    The rule is two lines, and both are load-bearing:

    * **Alarm** exactly when an item fails and is not grandfathered.
    * **Next set** is ``grandfather - {items that now pass}`` — items LEAVE by
      being fixed, and nothing ever joins after the first run.

    That asymmetry is the ratchet. Folding a newly-failing item into the set
    (the obvious-looking symmetry) would let every alarm silence itself on the
    following run by reclassifying the regression as known-bad — the exact
    failure mode grandfathering exists to prevent. Because the set only
    shrinks, "an item that was fixed and then regresses alarms again" holds by
    construction rather than by vigilance.

    A first run (``grandfather is None``) has nothing to regress against, so it
    SNAPSHOTS today's failures as the starting line and reports
    ``baseline_snapshot`` with no alarms. Those keys are the known-bad worklist,
    not news.
    """
    if metric.kind != 'tripwire':
        raise ValueError(
            f'evaluate_tripwire: metric {metric.metric_id!r} has kind {metric.kind!r}, '
            'not tripwire.'
        )
    items = metric.items or []
    failing = frozenset(item.item_key for item in items if not item.passed)
    passing = frozenset(item.item_key for item in items if item.passed)

    if grandfather is None:
        return (
            MetricVerdict(
                metric_id=metric.metric_id,
                rule_kind='tripwire',
                status='baseline_snapshot',
                value=metric.value,
                n=metric.n,
                baseline=float(len(failing)),
                detail=(
                    f'First run: snapshotted {len(failing)} failing item(s) as known-bad. '
                    'Regressions from here alarm.'
                ),
            ),
            failing,
        )

    next_grandfather = grandfather - passing
    alarms = tuple(
        Alarm(
            metric_id=metric.metric_id,
            rule_kind='tripwire',
            detail=f'Item {key!r} newly fails and is not grandfathered.',
            item_key=key,
            value=metric.value,
            baseline=float(len(grandfather)),
        )
        # Sorted so the artifact's alarm list is a function of the failures and
        # not of set iteration order — an unstable order would churn the diff.
        for key in sorted(failing - grandfather)
    )
    return (
        MetricVerdict(
            metric_id=metric.metric_id,
            rule_kind='tripwire',
            status='alarm' if alarms else 'ok',
            value=metric.value,
            n=metric.n,
            alarms=alarms,
            baseline=float(len(grandfather)),
            detail=(
                f'{len(failing)} failing, {len(grandfather)} grandfathered, '
                f'{len(alarms)} new; {len(grandfather - next_grandfather)} fixed and released.'
            ),
        ),
        next_grandfather,
    )
