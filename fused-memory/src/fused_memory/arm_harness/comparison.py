"""Cross-arm instrument checks over run manifests, and the client-class parity delta.

Arms are comparable only if they ran under symmetric settings at one code SHA.
``client_class_parity`` is boundary row 3's harness logic. It records each shared
metric's difference between two complete runs over one episode set as a delta
MetricsRecord.
"""

from collections.abc import Mapping, Sequence

from shared.memory_eval_metrics import Metric

from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.instrument_checks import (
    CheckResult,
    InstrumentCheckId,
    check_failed,
    check_passed,
)
from fused_memory.arm_harness.metrics_record import (
    DeltaOf,
    IndexConfiguration,
    MetricsRecord,
    record_for,
)
from fused_memory.arm_harness.run_manifest import RunManifest


class ParityPreconditionError(ValueError):
    """Two runs cannot be compared for client-class parity."""


def _symmetric_values(run: RunManifest) -> dict[str, object]:
    """The settings every arm in one comparison must share; model and client may differ."""
    params = run.spec.params if isinstance(run.spec, LlmArmSpec) else None
    return {
        'params.temperature': params.temperature if params else None,
        'params.max_tokens': params.max_tokens if params else None,
        'concurrency': run.settings_summary.concurrency,
        'index_configuration': run.settings_summary.index_configuration,
        'corpus_sha': run.spec.corpus_sha,
        'effective_embedder': run.effective_embedder,
        'graphiti_max_coroutines': run.graphiti_max_coroutines,
        'graphiti_semaphore_limit': run.graphiti_semaphore_limit,
    }


def _require_two_or_more(runs: Sequence[RunManifest]) -> None:
    if len(runs) < 2:
        raise ValueError(f'a cross-arm check needs at least two runs, got {len(runs)}')


def check_arm_config_symmetry(runs: Sequence[RunManifest]) -> CheckResult:
    _require_two_or_more(runs)
    check_id = InstrumentCheckId.ARM_CONFIG_SYMMETRY
    values = [(run.spec.arm_id, _symmetric_values(run)) for run in runs]
    first = values[0][1]
    asymmetric = [
        field for field in first if any(by_field[field] != first[field] for _, by_field in values)
    ]
    if asymmetric:
        lines = [
            f'{field}: ' + ', '.join(f'{arm_id}={by_field[field]!r}' for arm_id, by_field in values)
            for field in asymmetric
        ]
        detail = f'arms must run with symmetric settings; differing: {"; ".join(lines)}'
        return check_failed(check_id, detail, asymmetric)
    return check_passed(check_id, f'{len(runs)} arms ran with symmetric settings')


def check_single_code_sha(runs: Sequence[RunManifest]) -> CheckResult:
    _require_two_or_more(runs)
    check_id = InstrumentCheckId.SINGLE_CODE_SHA
    shas = sorted({run.spec.code_sha for run in runs})
    if len(shas) > 1:
        by_arm = ', '.join(f'{run.spec.arm_id}={run.spec.code_sha}' for run in runs)
        return check_failed(check_id, f'all arms must run at one code_sha; got {by_arm}', shas)
    return check_passed(check_id, f'{len(runs)} arms ran at code_sha {shas[0]}')


_RecordKey = tuple[str, IndexConfiguration | None]


def client_class_parity(
    run_a: RunManifest,
    run_b: RunManifest,
    records_a: Sequence[MetricsRecord],
    records_b: Sequence[MetricsRecord],
) -> tuple[MetricsRecord, ...]:
    """One scalar delta (a − b) per metric both arms report, carrying a's identity."""
    for run in (run_a, run_b):
        _require_complete(run)
    _require_same_episodes(run_a, run_b)
    keyed_a = _keyed(run_a, records_a)
    keyed_b = _keyed(run_b, records_b)
    delta_of = DeltaOf(minuend_arm_id=run_a.spec.arm_id, subtrahend_arm_id=run_b.spec.arm_id)
    return tuple(
        _delta(run_a, keyed_a[key], keyed_b[key], delta_of)
        for key in sorted(keyed_a.keys() & keyed_b.keys(), key=str)
    )


def _require_complete(run: RunManifest) -> None:
    if run.incomplete or run.abort is not None:
        raise ParityPreconditionError(
            f'arm {run.spec.arm_id!r} is incomplete (abort: {run.abort}); parity compares '
            'complete runs only'
        )


def _require_same_episodes(run_a: RunManifest, run_b: RunManifest) -> None:
    only_a = sorted(set(run_a.episode_ids) - set(run_b.episode_ids))
    only_b = sorted(set(run_b.episode_ids) - set(run_a.episode_ids))
    if only_a or only_b:
        raise ParityPreconditionError(
            f'parity needs one episode set: only in {run_a.spec.arm_id!r}: {only_a}; '
            f'only in {run_b.spec.arm_id!r}: {only_b}'
        )


def _keyed(
    run: RunManifest, records: Sequence[MetricsRecord]
) -> Mapping[_RecordKey, MetricsRecord]:
    keyed: dict[_RecordKey, MetricsRecord] = {}
    for record in records:
        key = (record.metric.metric_id, record.index_configuration)
        if record.arm_id != run.spec.arm_id:
            raise ParityPreconditionError(
                f'record {key} belongs to arm {record.arm_id!r}, not to run {run.spec.arm_id!r}'
            )
        if key in keyed:
            raise ParityPreconditionError(f'arm {run.spec.arm_id!r} reports {key} twice')
        keyed[key] = record
    return keyed


def _delta(
    run_a: RunManifest, record_a: MetricsRecord, record_b: MetricsRecord, delta_of: DeltaOf
) -> MetricsRecord:
    metric = Metric(
        metric_id=record_a.metric.metric_id,
        kind='scalar',
        value=record_a.metric.value - record_b.metric.value,
        n=min(record_a.metric.n, record_b.metric.n),
    )
    return record_for(
        run_a.spec,
        metric,
        measured_at=max(record_a.measured_at, record_b.measured_at),
        incomplete=False,
        index_configuration=record_a.index_configuration,
        delta_of=delta_of,
    )
