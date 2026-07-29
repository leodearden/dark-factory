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

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    'SCHEMA_VERSION',
    'Corpus',
    'Metric',
    'MetricKind',
    'MetricSeries',
    'TripwireItem',
    'parse_metric_series',
]

SCHEMA_VERSION = 1

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
