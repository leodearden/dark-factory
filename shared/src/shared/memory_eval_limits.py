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

import contextlib
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shared.memory_eval_metrics import Metric, MetricSeries

__all__ = [
    'Alarm',
    'ARTIFACT_SCHEMA_VERSION',
    'EvaluationResult',
    'GENERATOR',
    'LimitsArtifact',
    'LimitsConfig',
    'LimitsSchemaError',
    'MetricVerdict',
    'RuleKind',
    'VerdictStatus',
    'binomial_two_sided_p',
    'derive_alpha',
    'evaluate_count',
    'evaluate_proportion',
    'evaluate_series',
    'evaluate_tripwire',
    'grandfather_set_hash',
    'limits_artifact_path',
    'load_limits_artifact',
    'poisson_two_sided_p',
    'write_limits_artifact',
]

ARTIFACT_SCHEMA_VERSION = 1
GENERATOR = 'shared.memory_eval_limits'
"""Who wrote a given artifact, recorded in it. Cheap now, invaluable at 3am later."""

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


def _window_contributions(
    baseline_window: Sequence[MetricSeries], metric_id: str, kind: RuleKind
) -> tuple[list[Metric], tuple[str, ...]]:
    """The baseline runs that actually measured this metric, plus their stamps.

    A metric added mid-programme simply has no history in the older runs, which
    is an ``insufficient_data`` case rather than an error — so absence is
    reported by returning an empty list, not by raising.
    """
    found: list[Metric] = []
    stamps: list[str] = []
    for series in baseline_window:
        for metric in series.metrics:
            if metric.metric_id == metric_id and metric.kind == kind:
                found.append(metric)
                stamps.append(series.run_stamp)
                break
    return found, tuple(stamps)


def _insufficient(
    metric: Metric,
    *,
    baseline: float | None,
    p_value: float | None,
    alpha: float,
    stamps: tuple[str, ...],
    detail: str,
) -> MetricVerdict:
    """A verdict that declines to judge — carrying whatever numbers it did compute.

    ``insufficient_data`` is never an alarm, but it is also never a black box:
    the point estimate and (where computable) the p-value ride along so a
    reader can watch a trend build before it is actionable.
    """
    return MetricVerdict(
        metric_id=metric.metric_id,
        rule_kind=metric.kind,
        status='insufficient_data',
        value=metric.value,
        n=metric.n,
        baseline=baseline,
        p_value=p_value,
        alpha=alpha,
        baseline_run_stamps=stamps,
        detail=detail,
    )


def evaluate_proportion(
    metric: Metric,
    baseline_window: Sequence[MetricSeries],
    alpha: float,
    min_samples: int,
) -> MetricVerdict:
    """Rule (b): is this run's proportion surprising under the pooled baseline?

    The window is POOLED — successes and trials summed across runs — rather
    than averaged as ratios. Pooling gives the exact binomial test one honest
    trial count, where averaging three ratios would discard how many trials
    each rested on and quietly overweight a thin run.

    Sufficiency is checked BEFORE significance, in both directions: too few
    items in this run, or no baseline history for this metric, and the answer
    is ``insufficient_data`` regardless of how alarming the point estimate
    looks. Absence of evidence is not evidence of a regression.
    """
    if metric.kind != 'proportion':
        raise ValueError(
            f'evaluate_proportion: metric {metric.metric_id!r} has kind {metric.kind!r}, '
            'not proportion.'
        )
    history, stamps = _window_contributions(baseline_window, metric.metric_id, 'proportion')
    trials = sum(m.denominator or 0 for m in history)
    if not history or trials <= 0:
        return _insufficient(
            metric,
            baseline=None,
            p_value=None,
            alpha=alpha,
            stamps=stamps,
            detail='No baseline history for this metric — nothing to be surprised by yet.',
        )

    successes = sum(round(m.value * (m.denominator or 0)) for m in history)
    p0 = successes / trials
    observed = round(metric.value * metric.n)
    p_value = binomial_two_sided_p(observed, metric.n, p0)

    if metric.n < min_samples:
        return _insufficient(
            metric,
            baseline=p0,
            p_value=p_value,
            alpha=alpha,
            stamps=stamps,
            detail=(
                f'n={metric.n} is below min_samples={min_samples}: too few items to '
                'distinguish a regression from noise.'
            ),
        )

    return _verdict_from_p(
        metric,
        baseline=p0,
        p_value=p_value,
        alpha=alpha,
        stamps=stamps,
        detail=(
            f'{observed}/{metric.n} against a pooled baseline of {successes}/{trials} '
            f'over {len(history)} run(s).'
        ),
    )


def evaluate_count(
    metric: Metric,
    baseline_window: Sequence[MetricSeries],
    alpha: float,
    min_samples: int,
) -> MetricVerdict:
    """Rule (c): is this run's event count surprising under the baseline rate?

    The window's MEAN count is the Poisson rate. Same sufficiency-before-
    significance ordering as rule (b).
    """
    if metric.kind != 'count':
        raise ValueError(
            f'evaluate_count: metric {metric.metric_id!r} has kind {metric.kind!r}, not count.'
        )
    history, stamps = _window_contributions(baseline_window, metric.metric_id, 'count')
    if not history:
        return _insufficient(
            metric,
            baseline=None,
            p_value=None,
            alpha=alpha,
            stamps=stamps,
            detail='No baseline history for this metric — nothing to be surprised by yet.',
        )

    lam = math.fsum(m.value for m in history) / len(history)
    p_value = poisson_two_sided_p(round(metric.value), lam)

    if metric.n < min_samples:
        return _insufficient(
            metric,
            baseline=lam,
            p_value=p_value,
            alpha=alpha,
            stamps=stamps,
            detail=(
                f'n={metric.n} is below min_samples={min_samples}: too few items to '
                'distinguish a regression from noise.'
            ),
        )

    return _verdict_from_p(
        metric,
        baseline=lam,
        p_value=p_value,
        alpha=alpha,
        stamps=stamps,
        detail=f'{metric.value:g} against a baseline rate of {lam:g} over {len(history)} run(s).',
    )


def _verdict_from_p(
    metric: Metric,
    *,
    baseline: float,
    p_value: float,
    alpha: float,
    stamps: tuple[str, ...],
    detail: str,
) -> MetricVerdict:
    """Turn a computed p-value into a verdict. The ONLY place the bar is applied."""
    alarmed = p_value < alpha
    alarms = (
        (
            Alarm(
                metric_id=metric.metric_id,
                rule_kind=metric.kind,
                detail=f'{detail} p={p_value:.3g} < alpha={alpha:.3g}.',
                value=metric.value,
                baseline=baseline,
                p_value=p_value,
                alpha=alpha,
            ),
        )
        if alarmed
        else ()
    )
    return MetricVerdict(
        metric_id=metric.metric_id,
        rule_kind=metric.kind,
        status='alarm' if alarmed else 'ok',
        value=metric.value,
        n=metric.n,
        alarms=alarms,
        baseline=baseline,
        p_value=p_value,
        alpha=alpha,
        baseline_run_stamps=stamps,
        detail=detail,
    )


@dataclass(frozen=True)
class EvaluationResult:
    """Everything one run concluded, plus how it got there.

    ``alpha`` and ``alarmed_metric_count`` are carried, not just used, because
    a verdict is only re-derivable by hand if the bar it was judged against is
    recorded alongside it (G6's "calibration output with recorded provenance").
    """

    eval_id: str
    run_stamp: str
    alpha: float
    alarmed_metric_count: int
    config: LimitsConfig
    verdicts: tuple[MetricVerdict, ...]
    alarms: tuple[Alarm, ...]
    grandfather: frozenset[str]
    grandfather_hash: str
    baseline_run_stamps: tuple[str, ...]


def evaluate_series(
    current: MetricSeries,
    baseline_window: Sequence[MetricSeries],
    config: LimitsConfig,
    grandfather: frozenset[str] | None,
) -> EvaluationResult:
    """Evaluate every metric in a run and return the verdicts, alarms and next state.

    Alpha is derived ONCE here, from the metrics actually present in THIS run
    (PRD M2: "recomputed as metrics are added"). Resolving it at authoring time
    instead would mean a newly added metric silently raised the whole eval's
    false-alarm rate — the budget is per quarter and per eval, so it has to be
    split across whatever is currently spending it.

    Scalar metrics are reported but never alarmed — no rule is attached to them
    yet — so they are excluded from the alpha split rather than being handed a
    share of a budget they cannot spend. Tripwires ARE counted, which is mildly
    conservative (a structural rule has no false-alarm probability under the
    null, so it strictly does not consume budget); the stricter bar is the safe
    direction and keeps the count legible as "metrics that can alarm".

    Only the trailing ``config.baseline_window`` runs are used. A baseline that
    grew without bound would keep drifting further from current behaviour and
    would eventually alarm on the system's own gradual, intended changes.
    """
    window = list(baseline_window)[-config.baseline_window :] if config.baseline_window > 0 else []
    alarm_eligible = [m for m in current.metrics if m.kind != 'scalar']
    alpha = derive_alpha(config.false_alarm_budget, config.runs_per_quarter, len(alarm_eligible))

    verdicts: list[MetricVerdict] = []
    next_grandfather = grandfather if grandfather is not None else frozenset()

    for metric in current.metrics:
        if metric.kind == 'tripwire':
            verdict, next_grandfather = evaluate_tripwire(metric, grandfather)
        elif metric.kind == 'proportion':
            verdict = evaluate_proportion(metric, window, alpha, config.min_samples)
        elif metric.kind == 'count':
            verdict = evaluate_count(metric, window, alpha, config.min_samples)
        else:
            verdict = MetricVerdict(
                metric_id=metric.metric_id,
                rule_kind='scalar',
                status='ok',
                value=metric.value,
                n=metric.n,
                detail='Scalar: reported for trend, no alarm rule attached (M2).',
            )
        verdicts.append(verdict)

    return EvaluationResult(
        eval_id=current.eval_id,
        run_stamp=current.run_stamp,
        alpha=alpha,
        alarmed_metric_count=len(alarm_eligible),
        config=config,
        verdicts=tuple(verdicts),
        alarms=tuple(alarm for verdict in verdicts for alarm in verdict.alarms),
        grandfather=next_grandfather,
        grandfather_hash=grandfather_set_hash(next_grandfather),
        baseline_run_stamps=tuple(series.run_stamp for series in window),
    )


class LimitsSchemaError(ValueError):
    """A ``limits-current.json`` that does not match the artifact schema.

    A ``ValueError`` subclass, and raised in place of ``pydantic.ValidationError``,
    so a runner or dashboard can catch one exception type without taking a
    pydantic import — the same contract ``MetricSchemaError`` offers for M1.
    """


class _AlarmRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    metric_id: str = Field(min_length=1)
    rule_kind: RuleKind
    detail: str
    item_key: str | None = None
    value: float | None = None
    baseline: float | None = None
    p_value: float | None = None
    alpha: float | None = None


class _VerdictRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    metric_id: str = Field(min_length=1)
    rule_kind: RuleKind
    status: VerdictStatus
    value: float
    n: int = Field(ge=0)
    alarms: list[_AlarmRecord] = Field(default_factory=list)
    baseline: float | None = None
    p_value: float | None = None
    alpha: float | None = None
    baseline_run_stamps: list[str] = Field(default_factory=list)
    detail: str = ''


class LimitsArtifact(BaseModel):
    """The persisted form of an :class:`EvaluationResult` — state AND alarm feed.

    ONE file, deliberately, rather than separate evaluator-state and published-
    alarms files. Two files would be two things that can disagree, and the
    disagreement would be invisible: the dashboard would publish alarms derived
    from limits the evaluator had already moved past. Keeping them the same
    bytes makes "the dashboard's alarms are the same limits" true by
    construction (INV-1/INV-5) rather than by a convention someone has to keep.

    Strict at emit (``extra='forbid'``, pinned ``schema_version``) so a typo in
    a producing runner is caught here; lenient at read for consumers, who need
    only ``json.load`` and dict access and will not break on a field they do not
    know about.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    schema_version: Literal[1]
    eval_id: str = Field(min_length=1)
    run_stamp: str = Field(min_length=1)
    generator: str = Field(min_length=1)

    alpha: float
    false_alarm_budget: float
    runs_per_quarter: int
    alarmed_metric_count: int
    min_samples: int
    baseline_window: int
    baseline_run_stamps: list[str] = Field(default_factory=list)

    grandfather_set: list[str] = Field(default_factory=list)
    grandfather_set_hash: str = Field(min_length=1)

    verdicts: list[_VerdictRecord] = Field(default_factory=list)
    alarms: list[_AlarmRecord] = Field(default_factory=list)


def limits_artifact_path(root: str | Path, eval_id: str) -> Path:
    """``<root>/<eval_id>/limits-current.json``.

    No run stamp in the name, unlike the metric series: this file is CURRENT
    state, and there is exactly one current. The per-run history lives in the
    stamped metrics artifacts beside it.
    """
    return Path(root) / eval_id / 'limits-current.json'


def _artifact_from_result(result: EvaluationResult) -> LimitsArtifact:
    return LimitsArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        eval_id=result.eval_id,
        run_stamp=result.run_stamp,
        generator=GENERATOR,
        alpha=result.alpha,
        false_alarm_budget=result.config.false_alarm_budget,
        runs_per_quarter=result.config.runs_per_quarter,
        alarmed_metric_count=result.alarmed_metric_count,
        min_samples=result.config.min_samples,
        baseline_window=result.config.baseline_window,
        baseline_run_stamps=list(result.baseline_run_stamps),
        grandfather_set=sorted(result.grandfather),
        grandfather_set_hash=result.grandfather_hash,
        verdicts=[_VerdictRecord(**asdict(verdict)) for verdict in result.verdicts],
        alarms=[_AlarmRecord(**asdict(alarm)) for alarm in result.alarms],
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via temp-in-dir + os.replace.

    Copied from ``shared.prompt_artifact._atomic_write_text`` rather than
    imported (it is module-private there, and ``shared.safe_io`` has only a
    lenient reader — there is no atomic writer in ``shared/`` to reuse). The
    temp comes from :func:`tempfile.mkstemp`, an OS-guaranteed fresh name,
    rather than a pid-derived one, because the memory-eval runners all write
    under a single artifact root and a shared ``<name>.<pid>.tmp`` would let
    concurrent writers clobber each other mid-write.

    This matters more here than for the metrics artifact: this file is
    RESUMABLE STATE, so a torn write would take the grandfather set with it and
    the next run would alarm on everything that was already known-bad. Either
    the old contents survive intact or the new ones land whole.
    """
    fd, tmp_name = tempfile.mkstemp(suffix='.tmp', prefix=f'{path.name}.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def write_limits_artifact(result: EvaluationResult, path: str | Path) -> Path:
    """Persist *result* to *path* atomically, returning the path written.

    ``sort_keys=True`` and a trailing newline so two runs that concluded the
    same thing produce the same bytes — a committed artifact should diff to
    nothing when nothing changed, which is what makes a real change legible.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact = _artifact_from_result(result)
    text = json.dumps(
        artifact.model_dump(mode='json'), indent=2, sort_keys=True, ensure_ascii=False
    )
    _atomic_write_text(target, text + '\n')
    return target


def load_limits_artifact(path: str | Path) -> LimitsArtifact:
    """Read and validate a ``limits-current.json``.

    Raises :class:`LimitsSchemaError` (a ``ValueError``) if the file does not
    match the schema, chaining the underlying validation error. Corrupt
    resumable state is not something to recover from silently: continuing from
    a half-understood grandfather set would either mute real alarms or fire
    phantom ones, and both are worse than stopping.
    """
    target = Path(path)
    raw = json.loads(target.read_text(encoding='utf-8'))
    try:
        return LimitsArtifact.model_validate(raw)
    except ValidationError as exc:
        raise LimitsSchemaError(f'{target}: not a valid limits artifact: {exc}') from exc
