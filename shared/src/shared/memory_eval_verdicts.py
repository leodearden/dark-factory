"""The verdicts artifact — what one memory-eval run concluded, one entry per fingerprint.

``docs/prds/memory-eval-program.md`` §3 M2 (amendment) and M3: the scheduled
runner publishes, beside each eval's ``limits-current.json``, ONE root-level
verdicts artifact for the whole run. Its consumer is the dashboard, which reads
it as plain JSON and never imports this module — so the shape is pinned here,
strict at emit, exactly as ``shared.memory_eval_metrics`` pins M1.

**Entry granularity is the fingerprint.** A tripwire contributes one entry per
item, because each item can regress, be grandfathered or recover on its own;
a proportion, count or scalar metric contributes one entry for the whole
metric, labelled with its trailing baseline window. That label is derived from
the configured window length and never from a stamp, so a regression that
persists across runs keeps its fingerprint and is escalated once.

**Fingerprints are opaque.** Each is a sha256 hex digest over a unit-separated
identity tuple — eval, metric, and the item or window — and never over a value,
a stamp or prose. Consumers join on whole-string equality and must never parse
one; hashing makes that a property of the strings rather than a convention.

This module does not own filenames: the runner that produces the artifact owns
where it lands.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from shared.memory_eval_limits import (
    EvaluationResult,
    LimitsConfig,
    MetricVerdict,
    RuleKind,
    VerdictStatus,
)
from shared.memory_eval_metrics import Metric, MetricSeries, canonical_json_text, validate_stamp

__all__ = [
    'ARTIFACT_SCHEMA_VERSION',
    'GENERATOR',
    'RUNNER_FAILURE_FINGERPRINT',
    'RunnerFailureRecord',
    'RunnerPhase',
    'StormEscape',
    'Verdict',
    'VerdictEntry',
    'VerdictsArtifact',
    'VerdictsSchemaError',
    'build_verdicts_artifact',
    'load_verdicts_artifact',
    'metric_fingerprint',
    'serialize_verdicts_artifact',
    'storm_fingerprint',
    'verdict_entries',
]

ARTIFACT_SCHEMA_VERSION = 1
GENERATOR = 'shared.memory_eval_verdicts'

Verdict = Literal['alarm', 'no_alarm', 'insufficient_data', 'grandfathered']
RunnerPhase = Literal['produce', 'evaluate', 'runner']

_FINGERPRINT_NAMESPACE = 'memory-eval'
_UNIT_SEPARATOR = '\x1f'

_WHOLE_METRIC_VERDICT: dict[VerdictStatus, Verdict] = {
    'alarm': 'alarm',
    'ok': 'no_alarm',
    'improved': 'no_alarm',
    'insufficient_data': 'insufficient_data',
}


def _fingerprint(*parts: str) -> str:
    for part in parts:
        if _UNIT_SEPARATOR in part:
            raise ValueError(f'fingerprint part {part!r} contains the unit separator')
    joined = _UNIT_SEPARATOR.join((_FINGERPRINT_NAMESPACE, *parts))
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()


def metric_fingerprint(
    eval_id: str, metric_id: str, *, item: str | None = None, window: str | None = None,
) -> str:
    """The fingerprint of one judged identity: a tripwire item, or a metric's window."""
    if item is not None and window is None:
        return _fingerprint(eval_id, metric_id, 'item', item)
    if window is not None and item is None:
        return _fingerprint(eval_id, metric_id, 'window', window)
    raise ValueError('metric_fingerprint needs exactly one of item= or window=')


def storm_fingerprint(run_stamp: str) -> str:
    """The aggregate escalation's fingerprint: scoped to ONE run, so each storm files once."""
    return _fingerprint('storm', validate_stamp(run_stamp, what='run_stamp'))


RUNNER_FAILURE_FINGERPRINT = _fingerprint('runner-failure')
"""Program-scoped and stamp-free: a multi-day outage keeps ONE pending escalation."""


def _window_label(config: LimitsConfig) -> str:
    return f'trailing-{config.baseline_window}'


class VerdictEntry(BaseModel):
    """One judged identity in one run, carrying the fingerprint an escalation joins on."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    eval_id: str = Field(min_length=1)
    metric_id: str = Field(min_length=1)
    rule_kind: RuleKind
    item: str | None = None
    window: str | None = None
    fingerprint: str = Field(min_length=1)
    verdict: Verdict
    metric_status: VerdictStatus
    value: float
    limit_ref: str = Field(min_length=1)
    run_stamp: str
    detail: str

    @field_validator('run_stamp')
    @classmethod
    def _check_stamp_shape(cls, value: str) -> str:
        return validate_stamp(value, what='run_stamp')

    @model_validator(mode='after')
    def _check_identity(self) -> VerdictEntry:
        expected = metric_fingerprint(
            self.eval_id, self.metric_id, item=self.item, window=self.window,
        )
        if self.fingerprint != expected:
            raise ValueError(
                f'VerdictEntry {self.eval_id}/{self.metric_id}: fingerprint does not '
                'match its identity; regenerate the artifact rather than editing it.'
            )
        return self


def _entry(
    result: EvaluationResult,
    judged: MetricVerdict,
    *,
    verdict: Verdict,
    limit_ref: str,
    detail: str,
    item: str | None = None,
    window: str | None = None,
) -> VerdictEntry:
    return VerdictEntry(
        eval_id=result.eval_id,
        metric_id=judged.metric_id,
        rule_kind=judged.rule_kind,
        item=item,
        window=window,
        fingerprint=metric_fingerprint(result.eval_id, judged.metric_id, item=item, window=window),
        verdict=verdict,
        metric_status=judged.status,
        value=judged.value,
        limit_ref=limit_ref,
        run_stamp=result.run_stamp,
        detail=detail,
    )


def _tripwire_entries(
    metric: Metric, judged: MetricVerdict, result: EvaluationResult,
) -> Iterator[VerdictEntry]:
    alarm_details = {alarm.item_key: alarm.detail for alarm in judged.alarms}
    items = metric.items or []
    failing = {item.item_key for item in items if not item.passed}
    limit_ref = f'grandfather_set_hash={result.grandfather_hash}'
    for key in dict.fromkeys(item.item_key for item in items):
        if key in alarm_details:
            verdict: Verdict = 'alarm'
            detail = alarm_details[key]
        elif key in failing:
            verdict, detail = 'grandfathered', f'Item {key!r} fails and is grandfathered (known-bad).'
        else:
            verdict, detail = 'no_alarm', f'Item {key!r} passes.'
        yield _entry(result, judged, item=key, verdict=verdict, limit_ref=limit_ref, detail=detail)


def _whole_metric_limit_ref(judged: MetricVerdict, config: LimitsConfig) -> str:
    if judged.rule_kind == 'scalar':
        return 'no alarm rule'
    if judged.status == 'insufficient_data':
        return f'min_samples={config.min_samples}'
    return f'alpha={judged.alpha!r}'


def _whole_metric_entry(judged: MetricVerdict, result: EvaluationResult) -> VerdictEntry:
    verdict = _WHOLE_METRIC_VERDICT.get(judged.status)
    if verdict is None:
        raise ValueError(
            f'metric {judged.metric_id!r} ({judged.rule_kind}) has status '
            f'{judged.status!r}, which no whole-metric verdict maps to.'
        )
    return _entry(
        result, judged,
        window=_window_label(result.config),
        verdict=verdict,
        limit_ref=_whole_metric_limit_ref(judged, result.config),
        detail=judged.detail,
    )


def _require_same_run(series: MetricSeries, result: EvaluationResult) -> dict[str, MetricVerdict]:
    if (series.eval_id, series.run_stamp) != (result.eval_id, result.run_stamp):
        raise ValueError(
            f'series {series.eval_id}@{series.run_stamp} was not the run evaluated as '
            f'{result.eval_id}@{result.run_stamp}.'
        )
    judged = {verdict.metric_id: verdict for verdict in result.verdicts}
    if set(judged) != {metric.metric_id for metric in series.metrics}:
        raise ValueError(
            f'series {series.eval_id}@{series.run_stamp} and its evaluation judge '
            'different metrics.'
        )
    return judged


def verdict_entries(series: MetricSeries, result: EvaluationResult) -> tuple[VerdictEntry, ...]:
    """Classify one evaluated run into entries, in series metric order and item order."""
    judged = _require_same_run(series, result)
    entries: list[VerdictEntry] = []
    for metric in series.metrics:
        if metric.kind == 'tripwire':
            entries.extend(_tripwire_entries(metric, judged[metric.metric_id], result))
        else:
            entries.append(_whole_metric_entry(judged[metric.metric_id], result))
    return tuple(entries)


class VerdictsSchemaError(ValueError):
    """A verdicts artifact that does not match the schema; callers need no pydantic import."""


class StormEscape(BaseModel):
    """Whether this run's alarms exceed the storm limit K, collapsing them to one escalation."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    triggered: bool
    alarm_count: int = Field(ge=0)
    alarm_limit: int = Field(ge=1)
    aggregate_fingerprint: str | None = None

    @model_validator(mode='after')
    def _check_consistent(self) -> StormEscape:
        if self.triggered != (self.alarm_count > self.alarm_limit):
            raise ValueError(
                f'storm_escape: triggered={self.triggered} disagrees with '
                f'alarm_count={self.alarm_count} > alarm_limit={self.alarm_limit}.'
            )
        if self.triggered != (self.aggregate_fingerprint is not None):
            raise ValueError('storm_escape: aggregate_fingerprint is present iff triggered.')
        return self


class RunnerFailureRecord(BaseModel):
    """One thing the runner could not do this run; ``scope`` is an eval_id or ``'program'``."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    scope: str = Field(min_length=1)
    phase: RunnerPhase
    error_type: str = Field(min_length=1)
    message: str
    exit_code: int | None = None


class VerdictsArtifact(BaseModel):
    """The whole run: every eval's entries, the storm block, and what failed to run."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    schema_version: Literal[1]
    generator: str = Field(min_length=1)
    run_stamp: str
    entries: list[VerdictEntry] = Field(default_factory=list)
    storm_escape: StormEscape
    runner_failures: list[RunnerFailureRecord] = Field(default_factory=list)
    runner_failure_fingerprint: str | None = None

    @field_validator('run_stamp')
    @classmethod
    def _check_stamp_shape(cls, value: str) -> str:
        return validate_stamp(value, what='run_stamp')

    @model_validator(mode='after')
    def _check_entries(self) -> VerdictsArtifact:
        foreign = sorted({e.run_stamp for e in self.entries} - {self.run_stamp})
        if foreign:
            raise ValueError(f'verdicts {self.run_stamp}: entries from other runs {foreign}.')
        fingerprints = [entry.fingerprint for entry in self.entries]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError(f'verdicts {self.run_stamp}: more than one entry per fingerprint.')
        alarms = sum(1 for entry in self.entries if entry.verdict == 'alarm')
        if alarms != self.storm_escape.alarm_count:
            raise ValueError(
                f'verdicts {self.run_stamp}: storm_escape counts '
                f'{self.storm_escape.alarm_count} alarm(s), the entries carry {alarms}.'
            )
        return self

    @model_validator(mode='after')
    def _check_run_fingerprints(self) -> VerdictsArtifact:
        aggregate = self.storm_escape.aggregate_fingerprint
        if aggregate is not None and aggregate != storm_fingerprint(self.run_stamp):
            raise ValueError(f'verdicts {self.run_stamp}: aggregate_fingerprint is not this run\'s.')
        expected = RUNNER_FAILURE_FINGERPRINT if self.runner_failures else None
        if self.runner_failure_fingerprint != expected:
            raise ValueError(
                f'verdicts {self.run_stamp}: runner_failure_fingerprint must be '
                f'{expected!r} when {len(self.runner_failures)} failure(s) are recorded.'
            )
        return self


def build_verdicts_artifact(
    evaluations: Sequence[tuple[MetricSeries, EvaluationResult]],
    *,
    run_stamp: str,
    alarm_limit: int,
    runner_failures: Sequence[RunnerFailureRecord] = (),
) -> VerdictsArtifact:
    """Assemble one run's artifact; entries concatenate in *evaluations* order."""
    entries = [entry for series, result in evaluations for entry in verdict_entries(series, result)]
    alarm_count = sum(1 for entry in entries if entry.verdict == 'alarm')
    triggered = alarm_count > alarm_limit
    return VerdictsArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        generator=GENERATOR,
        run_stamp=run_stamp,
        entries=entries,
        storm_escape=StormEscape(
            triggered=triggered,
            alarm_count=alarm_count,
            alarm_limit=alarm_limit,
            aggregate_fingerprint=storm_fingerprint(run_stamp) if triggered else None,
        ),
        runner_failures=list(runner_failures),
        runner_failure_fingerprint=RUNNER_FAILURE_FINGERPRINT if runner_failures else None,
    )


def serialize_verdicts_artifact(artifact: VerdictsArtifact) -> str:
    """Canonical text under the null convention stated in ``shared.memory_eval_metrics``.

    Every nullable field here is optional, so ``exclude_none`` alone is the whole
    convention; the round-trip test is what would catch a required-nullable one.
    """
    return canonical_json_text(artifact.model_dump(mode='json', exclude_none=True))


def load_verdicts_artifact(path: str | Path) -> VerdictsArtifact:
    target = Path(path)
    raw = json.loads(target.read_text(encoding='utf-8'))
    try:
        return VerdictsArtifact.model_validate(raw)
    except ValidationError as exc:
        raise VerdictsSchemaError(f'{target}: not a valid verdicts artifact: {exc}') from exc
