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
reviewed idea rather than two — literally one function, :func:`_small_p_sum`,
handed a different pmf and a different support bound.

That shared summation walks OUTWARD FROM THE MODE in both directions, which is
what keeps the cost proportional to the sqrt-wide region that actually carries
mass rather than to the size of the support. Enumerating the support instead is
the obvious implementation and it does not scale: nothing in the M1 schema
bounds a proportion's ``n`` or a count metric's ``value``, so a corpus-scale
baseline rate would make the daily eval run take minutes-to-hours and drag the
grandfather-set update down with it. Both shoulders stop on the same two
STRUCTURAL conditions — the pmf underflowing float64 to literal zero, or a term
too small to change the running sum at double precision — so there is no
horizon constant anywhere, which matters because a tuned horizon would be
exactly the a-priori numeric threshold G6 forbids.

Both pmfs are assembled in log space via ``math.lgamma`` rather than from exact
``math.comb`` coefficients: a bignum binomial coefficient overflows float64
around ``n = 1030``, and nothing in the M1 schema bounds a proportion's ``n``.

Stdlib only — ``math.lgamma``/``math.exp``, no scipy or numpy (neither is a
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
from collections.abc import Callable, Iterable, Sequence
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
    'GRANDFATHER_KEY_SEPARATOR',
    'grandfather_set_hash',
    'grandfather_slice',
    'limits_artifact_path',
    'load_limits_artifact',
    'poisson_two_sided_p',
    'scoped_grandfather_key',
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


def _small_p_sum(pmf: Callable[[int], float], mode: int, k: int, upper: int | None) -> float:
    """Total probability of every outcome no more probable than *k*'s, walked from *mode*.

    The one summation both rule kinds share: hand it a binomial pmf and
    ``upper=n``, or a Poisson pmf and ``upper=None`` for the unbounded support.

    **Why walk rather than enumerate.** The qualifying outcomes are exactly the
    complement of a contiguous interval around the mode (the pmf is unimodal, so
    it is monotone on each side), which means each shoulder can start AT the peak
    and walk outward: skip while outcomes are still more probable than the
    observed one, then accumulate every remaining one. Both phases end on the
    same two structural conditions — the pmf underflowing float64 to literal
    zero, or a term too small to shift the running sum at double precision — so
    the walk visits O(sqrt(variance)) outcomes instead of the whole support, and
    it does so without a horizon constant (G6: a tuned horizon would be an
    a-priori numeric threshold). Terms past either stop cannot alter the result.

    **The peak short-circuit is an identity, not an optimisation.** If the
    observed outcome IS a mode then every outcome qualifies and the answer is
    exactly 1.0 by definition. Stating that structurally is what keeps it exact
    independently of how the pmf terms are computed — a log-space term is
    accurate to ~1e-15, not to the ulp, so summing the whole support instead
    returns 0.9999999999999996 and an equality test on it would be a lie. The
    comparison uses :data:`_PMF_TIE_SLACK` so the peak's TWIN also short-circuits:
    both ``pmf(lam-1) == pmf(lam)`` for an integer Poisson rate and
    ``pmf(m-1) == pmf(m)`` for an integer ``(n+1) p0`` are exact ties in real
    arithmetic that float can miss by an ulp.

    ``running`` is a plain sum used ONLY to detect negligibility; the returned
    total is an ``fsum`` over the collected terms, so the adaptive stop costs no
    accuracy. It is per-shoulder rather than shared so each side's stop depends
    only on the mass that side has actually collected.
    """
    peak = pmf(mode)
    observed = pmf(k)
    if observed >= peak * (1.0 - _PMF_TIE_SLACK):
        return 1.0
    threshold = observed * (1.0 + _PMF_TIE_SLACK)

    terms: list[float] = []
    for step in (-1, 1):
        running = 0.0
        i = mode
        while i >= 0 and (upper is None or i <= upper):
            term = pmf(i)
            if term == 0.0:
                break  # the pmf underflowed float64: nothing representable remains
            if term <= threshold:
                if running > 0.0 and running + term == running:
                    break  # and neither can any later term, since they only shrink
                terms.append(term)
                running += term
            i += step
    return min(1.0, math.fsum(terms))


def _binomial_mode(n: int, p0: float) -> int:
    """The most probable outcome of Binomial(n, p0): ``floor((n + 1) p0)``, clamped to n.

    Where the two-sided walk starts. Clamped because ``(n + 1) p0`` reaches
    ``n + 1`` at ``p0 == 1`` while the support ends at ``n``. When ``(n + 1) p0``
    is an integer ``m`` there are two modes, ``m`` and ``m - 1``, with equal pmf;
    the floor picks one and :func:`_small_p_sum`'s tie-aware peak comparison
    covers the other.
    """
    return min(n, int((n + 1) * p0))


def _binomial_pmf(i: int, n: int, p0: float) -> float:
    """P(X = i) for X ~ Binomial(n, p0), computed in log space.

    The degenerate ``p0`` cases are handled first, both because a degenerate H0
    is a real input here (a tripwire-adjacent proportion can legitimately have a
    baseline of all-pass or all-fail) and because they are the only inputs for
    which the log-space form would take ``log(0)``.

    The general case is assembled as ``exp(lgamma(n+1) - lgamma(i+1) -
    lgamma(n-i+1) + i*log(p0) + (n-i)*log1p(-p0))`` rather than the
    textbook-looking ``math.comb(n, i) * p0**i * (1-p0)**(n-i)``. ``math.comb``
    returns an exact arbitrary-precision int, and multiplying it by a float
    forces a conversion that raises ``OverflowError`` once the coefficient
    exceeds float64 — around ``n = 1030`` at ``p0 = 0.5``, which is well inside
    this programme's range (a corpus-scale proportion over ~1.2k entities is an
    ordinary M1 metric, and nothing in the M1 schema bounds ``n``). Building
    ``n+1`` exact bignum coefficients is also O(n^2) in digit work long before
    it crashes. ``_poisson_pmf`` next door already works in log space for
    exactly this reason; this is the same idea, applied to the same problem.

    ``math.exp`` underflows silently to 0.0 for a sufficiently negative
    argument (it only raises on overflow), so a far-tail term that is not
    representable becomes a structural zero rather than an error.
    """
    if p0 == 0.0:
        return 1.0 if i == 0 else 0.0
    if p0 == 1.0:
        return 1.0 if i == n else 0.0
    return math.exp(
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * math.log(p0)
        + (n - i) * math.log1p(-p0)
    )


def binomial_two_sided_p(k: int, n: int, p0: float) -> float:
    """Exact two-sided binomial p-value for *k* successes in *n* trials under *p0*.

    Rule (b)'s engine: how surprising is the current run's proportion, given the
    proportion pooled over the trailing baseline window?

    Computed by :func:`_small_p_sum` — the total probability of every outcome no
    more probable than the observed one, walked outward from the mode. For an
    observation AT the mode this is exactly 1.0 (every outcome qualifies), which
    is the identity the tests anchor on.

    The walk is what keeps this affordable at scale. Materialising the whole
    ``n + 1`` support is the obvious implementation and it is O(n) in both time
    and memory for an answer that only depends on the O(sqrt(n p (1-p)))-wide
    region carrying mass: at ``n = 1e7`` that is ~80 MB of pmf terms and seconds
    of work, on a metric the M1 schema permits (it bounds ``n`` not at all, and a
    corpus-scale proportion is an ordinary metric here).

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

    return _small_p_sum(lambda i: _binomial_pmf(i, n, p0), _binomial_mode(n, p0), k, n)


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
    unbounded on the right, so there is no support to enumerate at all — and the
    left shoulder, though finite, is ``lam`` wide, which is just as unaffordable
    once a baseline rate gets large (nothing bounds a count metric's ``value``,
    so nothing bounds the rate derived from a window of them). BOTH shoulders are
    therefore walked outward from the mode by :func:`_small_p_sum`, on structural
    stop conditions rather than a horizon constant — which would be exactly the
    a-priori numeric threshold G6 forbids.

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

    # int(lam) is floor(lam), which IS the Poisson mode; for an integer lam it
    # picks the upper of the two tied modes and _small_p_sum's tie-aware peak
    # comparison covers the lower one.
    return _small_p_sum(lambda i: _poisson_pmf(i, lam), int(lam), k, None)


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

    Every field is checked at construction because each nonsensical value fails
    in the SAME direction — toward permanent deafness, silently. A negative
    ``min_samples`` disables the sufficiency guard, so a one-item run is judged
    as if it were fully powered. A ``baseline_window`` below 1 leaves every
    statistical metric with an empty window, so the whole eval reports
    ``insufficient_data`` forever and never alarms again — with a healthy-looking
    artifact and no error to notice. A misconfiguration that makes an eval quiet
    must not be reachable by construction; see :func:`derive_alpha`, which
    refuses the same non-positive budget and cadence for the same reason.
    """

    false_alarm_budget: float = 1.0
    """Expected false alarms per quarter across the whole eval. PRD-sanctioned default."""

    runs_per_quarter: int
    """How often this eval runs — 90 for the D10 daily cadence."""

    min_samples: int
    """Fewest items in the current run before a verdict is meaningful."""

    baseline_window: int
    """How many trailing runs pool into the baseline the current run is judged against."""

    def __post_init__(self) -> None:
        if self.false_alarm_budget <= 0:
            raise ValueError(
                f'LimitsConfig: false_alarm_budget={self.false_alarm_budget} must be positive '
                '(a budget of zero admits no finite alpha).'
            )
        if self.runs_per_quarter <= 0:
            raise ValueError(
                f'LimitsConfig: runs_per_quarter={self.runs_per_quarter} must be positive — '
                'an eval that never runs has no budget to split.'
            )
        if self.min_samples < 0:
            raise ValueError(
                f'LimitsConfig: min_samples={self.min_samples} must be non-negative. '
                'A negative floor disables the sufficiency guard entirely, so a one-item '
                'run would be judged as if it were fully powered.'
            )
        if self.baseline_window < 1:
            raise ValueError(
                f'LimitsConfig: baseline_window={self.baseline_window} must be at least 1. '
                'A shorter window is an empty window, which makes every proportion and '
                'count metric report insufficient_data forever — a permanently deaf eval '
                'that still looks healthy.'
            )


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
    """The comparison point ``value`` was judged against, in ``value``'s own units.

    Per rule kind: the pooled baseline proportion (b), the baseline rate scaled to
    THIS run's exposure — i.e. the expected count, not the per-unit rate (c), or
    the grandfathered failure count (a).
    """
    p_value: float | None = None
    """``None`` for structural rules, which measure a fact rather than a surprise."""
    alpha: float | None = None
    """The derived bar this run's p-value was judged against. Provenance, not config."""
    baseline_run_stamps: tuple[str, ...] = ()
    """Exactly which runs produced ``baseline`` — so a verdict can be re-derived by hand."""
    detail: str = ''


GRANDFATHER_KEY_SEPARATOR = '::'
"""What separates a metric_id from an item_key in a persisted grandfather entry.

The grandfather set is stored — and published in ``limits-current.json`` — as a
flat list of ``"<metric_id>::<item_key>"`` strings rather than bare item_keys,
because item_keys are only unique WITHIN a metric. Two structural probes may
legitimately key on the same topic id, and a flat namespace would let one
metric's pass release the other metric's known-bad entry (and, worse, let an
unrelated metric's evaluation drop it entirely).

A flat scoped list rather than a nested mapping so the artifact field stays one
JSON array of strings for the dashboard, and so ``grandfather_set_hash``
remains a digest of one comparable set.
"""


def scoped_grandfather_key(metric_id: str, item_key: str) -> str:
    """Compose the persisted grandfather entry for *item_key* under *metric_id*.

    Rejects a ``metric_id`` containing the separator: prefix matching is how the
    slice is recovered, so a metric_id of ``'a::b'`` could shadow item ``'b/...'``
    of metric ``'a'``. Loudly refusing the ambiguous name costs nothing (no
    metric_id in this programme contains a colon) and beats a silent mis-scope.
    """
    if GRANDFATHER_KEY_SEPARATOR in metric_id:
        raise ValueError(
            f'scoped_grandfather_key: metric_id {metric_id!r} may not contain '
            f'{GRANDFATHER_KEY_SEPARATOR!r} — it is the grandfather-key scope separator.'
        )
    return f'{metric_id}{GRANDFATHER_KEY_SEPARATOR}{item_key}'


def grandfather_slice(grandfather: Iterable[str], metric_id: str) -> frozenset[str]:
    """The bare item_keys grandfathered under *metric_id*, from a persisted set.

    The inverse of :func:`scoped_grandfather_key`. Entries belonging to other
    metrics are simply not this metric's business and are dropped from the
    slice — the caller is responsible for carrying them forward untouched.
    """
    prefix = scoped_grandfather_key(metric_id, '')
    return frozenset(key[len(prefix) :] for key in grandfather if key.startswith(prefix))


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

    Operates on THIS METRIC'S SLICE of the grandfather state — bare item_keys,
    with no metric scoping — and returns the next such slice. Composing the
    slices back into the persisted, metric-scoped set is
    :func:`evaluate_series`'s job (see :data:`GRANDFATHER_KEY_SEPARATOR`); this
    function must never see another metric's entries, because item_keys are
    only unique within a metric.

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

    The baseline is a PER-UNIT-OF-EXPOSURE rate — events pooled over the window
    divided by the ``n`` those events were counted over — and the expectation the
    current run is judged against is that rate scaled back up by the current run's
    own ``n``. This is the same pooling rule (b) applies to successes and trials,
    for the same reason: ``Metric.n`` is the exposure a count was measured over,
    it is free to vary run to run (the schema asks only for ``n >= 0``, and the
    committed fixtures already carry the same metric_id at ``n=30`` and ``n=6``),
    and a plain mean of raw counts silently assumes it never does.

    Assuming it away produces false alarms in BOTH directions on unchanged
    behaviour. Three runs of 50 events over 30 probes, then a run where the probe
    set halves to 15 and returns 25: identical per-probe rate, nothing regressed,
    yet the raw comparison is ``poisson_two_sided_p(25, 50) = 1.6e-04`` — an alarm
    at any sane budget. Growing the probe set or the corpus does the same in
    reverse, and both are expected events here rather than edge cases (adding
    metrics and growing corpora is why alpha is recomputed per run at all). Worse,
    those alarms are not accounted for in the false-alarm budget, which assumes a
    stationary process; they would spend it on arithmetic.

    Sufficiency is checked before significance, as in rule (b), and now includes
    exposure on both sides: a window that measured nothing, and a current run that
    measured nothing, are both ``insufficient_data``. Judging a rate against zero
    exposure is not strictness — with ``lam == 0`` every non-zero count has
    ``p == 0.0``, so it would be a guaranteed alarm on no evidence at all.
    """
    if metric.kind != 'count':
        raise ValueError(
            f'evaluate_count: metric {metric.metric_id!r} has kind {metric.kind!r}, not count.'
        )
    history, stamps = _window_contributions(baseline_window, metric.metric_id, 'count')
    exposure = sum(m.n for m in history)
    if not history:
        return _insufficient(
            metric,
            baseline=None,
            p_value=None,
            alpha=alpha,
            stamps=stamps,
            detail='No baseline history for this metric — nothing to be surprised by yet.',
        )
    if exposure <= 0:
        return _insufficient(
            metric,
            baseline=None,
            p_value=None,
            alpha=alpha,
            stamps=stamps,
            detail=(
                f'The baseline window measured no exposure (n sums to {exposure} over '
                f'{len(history)} run(s)): there is no rate to derive.'
            ),
        )
    if metric.n <= 0:
        return _insufficient(
            metric,
            baseline=None,
            p_value=None,
            alpha=alpha,
            stamps=stamps,
            detail=(
                f'This run measured no exposure (n={metric.n}): a count cannot be judged '
                'against a rate over nothing.'
            ),
        )

    rate = math.fsum(m.value for m in history) / exposure
    lam = rate * metric.n
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
        detail=(
            f'{metric.value:g} event(s) over n={metric.n} against an expected {lam:g} '
            f'({rate:g}/unit pooled over {len(history)} run(s), exposure {exposure}).'
        ),
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

    ``alpha`` is ``None`` exactly when ``alarmed_metric_count == 0`` — a run
    carrying only scalar metrics (or no metrics at all) still reports, but
    nothing in it can spend the false-alarm budget, so there is no bar and
    recording a fabricated one would misrepresent the provenance.
    """

    eval_id: str
    run_stamp: str
    alpha: float | None
    alarmed_metric_count: int
    config: LimitsConfig
    verdicts: tuple[MetricVerdict, ...]
    alarms: tuple[Alarm, ...]
    grandfather: frozenset[str]
    grandfather_hash: str
    baseline_run_stamps: tuple[str, ...]
    snapshotted_metrics: frozenset[str] = frozenset()
    """Which tripwire metrics have ever been snapshotted — the other half of the state.

    Carried forward alongside ``grandfather`` into the next run (see
    :func:`evaluate_series`). An empty grandfather SLICE cannot distinguish "this
    metric is new" from "every one of its known-bad items was fixed", so
    first-run-ness is recorded rather than inferred.
    """


def evaluate_series(
    current: MetricSeries,
    baseline_window: Sequence[MetricSeries],
    config: LimitsConfig,
    grandfather: frozenset[str] | None,
    snapshotted_metrics: Iterable[str] | None = None,
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

    When NOTHING is alarm-eligible — a run of only scalar metrics, or one with
    no metrics yet — there is no alpha to derive and ``result.alpha`` is
    ``None``. Such a run still reports: the scalar verdicts are emitted and the
    artifact is written, it simply cannot alarm. :func:`derive_alpha` keeps
    rejecting a zero count, which is right for the pure function (no finite
    alpha splits a budget zero ways); the caller is the one that knows an empty
    split is a legitimate state rather than a mis-configuration, so the
    short-circuit lives here.

    Only the trailing ``config.baseline_window`` runs are used. A baseline that
    grew without bound would keep drifting further from current behaviour and
    would eventually alarm on the system's own gradual, intended changes.

    **A tripwire ADDED mid-programme snapshots rather than alarming.** Adding
    metrics is an expected event, not an edge case (it is why alpha is
    recomputed per run at all), and M2 says pre-existing failures are
    grandfathered — so a new structural probe's existing failures are the fix
    lineage's worklist, exactly as they were on day one, and not news. That
    cannot be inferred from the state: :func:`grandfather_slice` returns an
    empty set both for a metric that is new and for one whose every known-bad
    item has been fixed, and those two must behave in opposite ways. So
    ``snapshotted_metrics`` RECORDS which tripwires have already been
    snapshotted, and is carried from run to run beside ``grandfather``
    (``EvaluationResult.snapshotted_metrics``, persisted as the artifact's
    ``snapshotted_metric_ids``). Metrics not in the current run stay in the
    ledger — a probe that skipped a run has not become new again.

    The grandfather set handed in must be the PERSISTED, metric-scoped form.
    An unscoped entry is rejected outright rather than quietly ignored: it
    matches no metric's prefix, so every one of that metric's known-bad items
    would alarm and the artifact would then publish the unscoped junk as state.
    :func:`evaluate_tripwire`'s return value is the obvious wrong seed — it
    speaks bare item_keys for a single metric — so the guard names it.

    Passing ``snapshotted_metrics=None`` alongside a non-``None`` grandfather
    means the ledger was not carried, which is a caller bug rather than a
    supported mode (``load_limits_artifact`` always supplies it). It is read
    conservatively — every tripwire in the run is assumed already snapshotted —
    so half-carried state degrades toward alarming on a new metric's known-bad
    items, which is loud and self-correcting. The opposite default would
    re-snapshot a metric that already had state and silently swallow a real
    regression as known-bad.
    """
    # `config.baseline_window >= 1` is guaranteed by LimitsConfig.__post_init__,
    # so this slice is never the accidental whole-list `[-0:]`.
    window = list(baseline_window)[-config.baseline_window :]
    alarm_eligible = [m for m in current.metrics if m.kind != 'scalar']
    alpha = (
        derive_alpha(config.false_alarm_budget, config.runs_per_quarter, len(alarm_eligible))
        if alarm_eligible
        else None
    )

    if grandfather is not None:
        unscoped = sorted(k for k in grandfather if GRANDFATHER_KEY_SEPARATOR not in k)
        if unscoped:
            raise ValueError(
                f'evaluate_series: grandfather entries {unscoped} are not metric-scoped. '
                'This function takes the PERSISTED set, whose entries are '
                f'"<metric_id>{GRANDFATHER_KEY_SEPARATOR}<item_key>" — chain a previous '
                "EvaluationResult.grandfather (or a loaded artifact's grandfather_set), or "
                'compose keys with scoped_grandfather_key(). Note evaluate_tripwire returns '
                'BARE item_keys for one metric and its output is not a valid seed on its own.'
            )

    verdicts: list[MetricVerdict] = []
    # The persisted set is SCOPED BY METRIC (see GRANDFATHER_KEY_SEPARATOR).
    # Start from what was handed in and replace one metric's slice at a time, so
    # a second tripwire never clobbers the first one's known-bad list and a
    # metric absent from this run keeps its entries rather than losing them.
    # Neither a union nor an intersection across tripwires would do: a union
    # re-adds items another metric just released, an intersection breaks the
    # first-run snapshot.
    carried: set[str] = set(grandfather) if grandfather is not None else set()

    tripwire_ids = {m.metric_id for m in current.metrics if m.kind == 'tripwire'}
    if grandfather is None:
        # Whole-programme first run: nothing has been snapshotted yet, whatever
        # a caller may have passed.
        known = set()
    elif snapshotted_metrics is None:
        known = set(tripwire_ids)  # ledger not carried — see the docstring
    else:
        known = set(snapshotted_metrics)

    for metric in current.metrics:
        if metric.kind == 'tripwire':
            own = (
                grandfather_slice(grandfather, metric.metric_id)
                if grandfather is not None and metric.metric_id in known
                else None
            )
            verdict, next_own = evaluate_tripwire(metric, own)
            prefix = scoped_grandfather_key(metric.metric_id, '')
            carried = {key for key in carried if not key.startswith(prefix)}
            carried |= {scoped_grandfather_key(metric.metric_id, item) for item in next_own}
        elif metric.kind in ('proportion', 'count'):
            if alpha is None:
                # Unreachable: `alarm_eligible` above is every non-scalar metric,
                # so reaching a statistical rule guarantees a derived alpha. Loud
                # rather than silent if that filter ever drifts out of step with
                # this branch set — a fabricated stand-in alpha would be exactly
                # the a-priori threshold G6 forbids.
                raise AssertionError(
                    f'evaluate_series: no alpha was derived, yet metric '
                    f'{metric.metric_id!r} has alarm-eligible kind {metric.kind!r}.'
                )
            verdict = (
                evaluate_proportion(metric, window, alpha, config.min_samples)
                if metric.kind == 'proportion'
                else evaluate_count(metric, window, alpha, config.min_samples)
            )
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
        grandfather=frozenset(carried),
        grandfather_hash=grandfather_set_hash(carried),
        # Union, not replacement: a tripwire absent from this run has not
        # become new again, so it keeps its place in the ledger just as it
        # keeps its grandfather slice.
        snapshotted_metrics=frozenset(known | tripwire_ids),
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

    alpha: float | None
    """The derived per-metric bar, or ``null`` when ``alarmed_metric_count`` is 0.

    Required but nullable, not optional: a producer that forgets the field is a
    bug worth catching at emit, while a run of only scalar metrics genuinely has
    no bar. A consumer reading ``null`` should render "no alarm rules in this
    run", not substitute a default — there is no defensible default (G6).
    """

    false_alarm_budget: float
    runs_per_quarter: int
    alarmed_metric_count: int
    min_samples: int
    baseline_window: int
    baseline_run_stamps: list[str] = Field(default_factory=list)

    grandfather_set: list[str] = Field(default_factory=list)
    grandfather_set_hash: str = Field(min_length=1)
    snapshotted_metric_ids: list[str] = Field(default_factory=list)
    """Tripwires that have already been snapshotted — state, not a published signal.

    Present because a metric ADDED mid-programme must snapshot its pre-existing
    failures rather than alarm on them, and an empty grandfather slice cannot
    tell "new" from "all fixed". A dashboard has no use for this field; the
    evaluator's own resume path does.
    """

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
        snapshotted_metric_ids=sorted(result.snapshotted_metrics),
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
    """Read, validate and integrity-check a ``limits-current.json``.

    Raises :class:`LimitsSchemaError` (a ``ValueError``) if the file does not
    match the schema, chaining the underlying validation error. Corrupt
    resumable state is not something to recover from silently: continuing from
    a half-understood grandfather set would either mute real alarms or fire
    phantom ones, and both are worse than stopping.

    Shape validity is not enough for that promise, so ``grandfather_set_hash`` is
    RECOMPUTED here rather than merely carried. The artifact is committed and
    hand-readable, which makes a hand-edit, a partial rewrite or a badly resolved
    git conflict realistic paths to a set that parses perfectly and is wrong —
    and the two ways it can be wrong are exactly the two failures the docstring
    above refuses: entries dropped means a burst of phantom alarms on items
    nobody touched, entries added means real regressions silently muted. The
    digest the writer already emits is what makes both detectable, so it is
    checked at the one place a runner resumes from.
    """
    target = Path(path)
    raw = json.loads(target.read_text(encoding='utf-8'))
    try:
        artifact = LimitsArtifact.model_validate(raw)
    except ValidationError as exc:
        raise LimitsSchemaError(f'{target}: not a valid limits artifact: {exc}') from exc

    recomputed = grandfather_set_hash(artifact.grandfather_set)
    if recomputed != artifact.grandfather_set_hash:
        raise LimitsSchemaError(
            f'{target}: grandfather_set does not match grandfather_set_hash — '
            f'recorded {artifact.grandfather_set_hash}, recomputed {recomputed} over '
            f'{len(artifact.grandfather_set)} key(s). The known-bad list has been '
            'edited or partially rewritten; resuming from it would either fire phantom '
            'alarms on untouched items or mute real regressions. Regenerate the artifact '
            'with write_limits_artifact rather than repairing it by hand.'
        )
    return artifact
