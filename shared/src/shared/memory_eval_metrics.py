"""The M1 metric-series schema — one home for every memory-eval artifact shape.

``docs/prds/memory-eval-program.md`` §3 M1: *every eval run emits a
machine-readable metrics artifact conforming to one schema module; the artifact
series on disk is the sole read surface for both the limits evaluator and the
dashboard.* Per-run artifact layout is
``<root>/<eval_id>/metrics-<STAMP>.json``.

**Why this lives in ``shared/`` (D2).** Both ``fused-memory`` and
``orchestrator`` depend on ``dark-factory-shared``, but ``fused-memory`` does
NOT depend on ``orchestrator`` — a runner importing ``orchestrator.evals``
would be a wrong-direction package edge. The verdict/threshold *pattern* from
``orchestrator/src/orchestrator/evals/prompt_opt/canary.py`` is therefore
COPIED into this module family, never imported. Keep it that way.

The dashboard (separate PRD) consumes the emitted artifacts as plain JSON on
disk and never imports this module — which is exactly why the shape is pinned
here with a ``schema_version`` rather than left implicit.

**Strict at emit, lenient at read.** These are pydantic v2 models with
``extra='forbid'`` and ``frozen=True``, following
``shared.capability_manifest``'s convention for a strict authoring schema whose
deliverable *is* rejecting malformed/typo'd entries. A malformed metric must be
rejected in the producing runner (M1: "malformed metric rejected at emit time,
not read time"), because by the time the dashboard reads the artifact there is
nobody left to tell. Readers, by contrast, just ``json.load`` — so extra fields
a future schema version adds never break them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    'SCHEMA_VERSION',
    'Corpus',
    'Metric',
    'MetricKind',
    'MetricSchemaError',
    'MetricSeries',
    'TripwireItem',
    'parse_metric_series',
    'validate_metric_series',
]

SCHEMA_VERSION = 1

_WHOLE_NUMBER_TOL = 1e-9
"""Relative slack when asking "is this float a whole number?".

Not a calibrated threshold — a float-representation allowance. ``20 / 30 * 30``
is ``19.999999999999996``, so an exact ``== round(x)`` would reject a proportion
the writer itself produced.
"""


class MetricSchemaError(ValueError):
    """A metric series that must not be emitted (M1: rejected at emit time).

    A ``ValueError`` subclass so a runner can catch one exception type without
    importing pydantic — the schema library is this module's implementation
    detail, not part of the contract every producer must depend on. The
    underlying ``pydantic.ValidationError`` is always chained as ``__cause__``,
    so the field-level detail is never lost.
    """


def _is_whole(value: float) -> bool:
    return abs(value - round(value)) <= _WHOLE_NUMBER_TOL * max(1.0, abs(value))


MetricKind = Literal['tripwire', 'proportion', 'count', 'scalar']
"""The closed metric vocabulary, one per M2 rule kind plus ``scalar``.

``tripwire`` -> rule (a), the grandfathered per-item binary predicate;
``proportion`` -> rule (b), the exact binomial test; ``count`` -> rule (c), the
Poisson tail test; ``scalar`` is reported but never alarmed (there is no
sampling model for it, so no exact test applies).
"""


class TripwireItem(BaseModel):
    """One item of a structural tripwire's per-item binary predicate (M2 rule a).

    ``item_key`` must be stable across runs — it is what the grandfather set
    stores and what an alarm names. D5 calls for content-hash keys rather than
    UUIDs precisely so the key survives re-consolidation.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    item_key: str = Field(min_length=1)
    passed: bool


class Corpus(BaseModel):
    """What the run measured against — the denominator behind the denominators.

    ``counts`` is free-form (category -> size) rather than a fixed set of
    fields: the Mem0/Graphiti bucket vocabulary is owned by the
    memory-metadata-vocabulary PRD, not this one, so pinning it here would be a
    second home for someone else's list.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    project_id: str = Field(min_length=1)
    counts: dict[str, int] = Field(default_factory=dict)


class Metric(BaseModel):
    """One measurement in a run's series (M1 §3).

    ``value``/``n`` are always present; ``denominator`` (proportion trials),
    ``items`` (tripwire per-item results) and ``details_path`` (a companion
    human-readable artifact) are kind-conditional. The per-kind cross-field
    rules live in this model's validator so a malformed metric cannot be
    constructed at all, in or out of a series.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    metric_id: str = Field(min_length=1)
    kind: MetricKind
    value: float
    n: int = Field(ge=0)
    denominator: int | None = None
    items: list[TripwireItem] | None = None
    details_path: str | None = None

    @model_validator(mode='after')
    def _check_kind_fields(self) -> Metric:
        where = f'Metric {self.metric_id!r} (kind={self.kind!r})'
        if self.kind != 'tripwire' and self.items is not None:
            raise ValueError(f'{where}: items are only meaningful for a tripwire metric.')
        if self.kind != 'proportion' and self.denominator is not None:
            raise ValueError(f'{where}: denominator is only meaningful for a proportion metric.')

        if self.kind == 'tripwire':
            if not self.items:
                raise ValueError(f'{where}: a tripwire must carry a non-empty items list.')
            if self.n != len(self.items):
                raise ValueError(
                    f'{where}: n={self.n} disagrees with len(items)={len(self.items)}.'
                )
            failures = sum(1 for item in self.items if not item.passed)
            if self.value != failures:
                raise ValueError(
                    f'{where}: value={self.value} disagrees with the {failures} failing item(s). '
                    'A tripwire value IS its failure count.'
                )
        elif self.kind == 'proportion':
            if self.denominator is None:
                raise ValueError(f'{where}: a proportion must carry a denominator.')
            if self.denominator <= 0:
                raise ValueError(f'{where}: denominator={self.denominator} must be positive.')
            if self.n != self.denominator:
                raise ValueError(
                    f'{where}: n={self.n} disagrees with denominator={self.denominator}. '
                    'The sample size IS the trial count — the exact binomial test needs one n.'
                )
            if not 0.0 <= self.value <= 1.0:
                raise ValueError(f'{where}: value={self.value} is outside [0, 1].')
            if not _is_whole(self.value * self.denominator):
                raise ValueError(
                    f'{where}: value={self.value} x denominator={self.denominator} is '
                    f'{self.value * self.denominator}, not a whole number of successes.'
                )
        elif self.kind == 'count':
            if self.value < 0:
                raise ValueError(f'{where}: value={self.value} must be non-negative.')
            if not _is_whole(self.value):
                raise ValueError(f'{where}: value={self.value} must be a whole number of events.')
        return self


class MetricSeries(BaseModel):
    """One run's complete metrics artifact — the unit written to disk.

    ``schema_version`` is pinned as a ``Literal`` (the
    ``capability_manifest.py:219`` convention) so a version this loader does not
    understand is a loud validation failure rather than a silent branch that
    misreads a future shape.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    schema_version: Literal[1]
    eval_id: str = Field(min_length=1)
    run_stamp: str = Field(min_length=1)
    corpus: Corpus
    metrics: list[Metric]

    @model_validator(mode='after')
    def _check_unique_metric_ids(self) -> MetricSeries:
        # Uniqueness can only be checked with the whole list in view. It is
        # load-bearing downstream, not cosmetic: the limits evaluator joins the
        # current run to its baseline window BY metric_id, so a duplicate would
        # silently make one of the two invisible to the comparison.
        seen: set[str] = set()
        for metric in self.metrics:
            if metric.metric_id in seen:
                raise ValueError(
                    f'MetricSeries {self.eval_id!r}: duplicate metric_id {metric.metric_id!r}.'
                )
            seen.add(metric.metric_id)
        return self


def _describe(exc: ValidationError, data: object) -> str:
    """Render a ValidationError, naming the offending ``metric_id`` per error.

    pydantic reports a nested failure by positional location
    (``('metrics', 3, 'value')``), which tells a runner nothing about WHICH
    metric it got wrong. Resolving that index back to the raw payload's
    ``metric_id`` is what makes the message actionable — the
    structured-facts-at-failure invariant applied to a schema error.
    """
    metrics = data.get('metrics') if isinstance(data, dict) else None

    def offender_at(loc: tuple) -> str | None:
        if not (len(loc) >= 2 and loc[0] == 'metrics' and isinstance(loc[1], int)):
            return None
        if not isinstance(metrics, list) or not 0 <= loc[1] < len(metrics):
            return None
        entry = metrics[loc[1]]
        raw_id = entry.get('metric_id') if isinstance(entry, dict) else None
        return raw_id if isinstance(raw_id, str) else None

    parts: list[str] = []
    for err in exc.errors():
        loc = tuple(err.get('loc', ()))
        offender = offender_at(loc)
        prefix = f'metric {offender!r} at ' if offender is not None else 'at '
        parts.append(f'{prefix}{".".join(str(p) for p in loc) or "<root>"}: {err.get("msg", "")}')
    return '; '.join(parts)


def _coerce_series(series: MetricSeries | dict) -> MetricSeries:
    """Return *series* as a validated model, raising :class:`MetricSchemaError`.

    Accepts an already-parsed :class:`MetricSeries` (a no-op — it is valid by
    construction) so callers holding either shape use one entry point.
    """
    if isinstance(series, MetricSeries):
        return series
    try:
        return MetricSeries.model_validate(series)
    except ValidationError as exc:
        eval_id = series.get('eval_id') if isinstance(series, dict) else None
        raise MetricSchemaError(
            f'Malformed metric series (eval_id={eval_id!r}): {_describe(exc, series)}'
        ) from exc


def validate_metric_series(series: MetricSeries | dict) -> None:
    """Assert *series* is emittable, raising :class:`MetricSchemaError` if not.

    The emit-time gate of M1. Producers call this (or go straight to
    :func:`write_metric_series`, which calls it first) so a malformed metric
    fails in the runner that computed it, where there is still somebody to tell
    — never at read time in the dashboard, where there is not.
    """
    _coerce_series(series)


def parse_metric_series(data: dict) -> MetricSeries:
    """Validate a raw ``dict`` (already JSON-decoded) as a metric series.

    A thin ``model_validate`` delegator — the convenience entry point for a
    caller that already holds the decoded mapping, and what lets the whole
    rejection matrix be exercised in memory without touching disk (the
    ``parse_X``/``load_X`` split of ``shared.capability_manifest``).

    Raises ``pydantic.ValidationError`` on any malformed shape; never swallows.
    Producers that want a single non-pydantic exception type should go through
    :func:`validate_metric_series` instead.
    """
    return MetricSeries.model_validate(data)
