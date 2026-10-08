"""``harness.py incumbent-cost``: the incumbent's measured LLM spend, derived offline."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _fm_helpers import load_script_module
from shared.memory_eval_metrics import Metric

from arm_harness._fakes import (
    FINISHED_AT,
    embedding_spec,
    incumbent_control_spec,
    llm_spec,
    run_manifest_for,
    telemetry_row,
    untokened_telemetry_row,
    write_run,
)
from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.incumbent_cost import (
    INCUMBENT_COST_FILENAME,
    PRODUCTION_TELEMETRY_FILENAME,
    IncumbentCost,
    TelemetryWindow,
    derive_incumbent_cost,
    load_llm_writes,
    select_llm_writes,
    serialize_incumbent_cost,
)
from fused_memory.arm_harness.metrics_record import (
    LlmMetricId,
    MetricsRecord,
    load_metrics_records,
    record_for,
)

LME_DIR = Path(__file__).parents[2] / 'scripts' / 'local_memory_models_eval'
UNTIL = '2026-10-08T00:00:00+00:00'
FIRST_TOKENED = '2026-10-05T11:23:27.123456+00:00'
PRE_START_ROW = untokened_telemetry_row('2026-10-05T10:59:00+00:00')
TOKENED_ROWS = (
    telemetry_row('2026-10-07T10:00:00+00:00', tokens=(2000, 200, 12)),
    telemetry_row(
        '2026-10-06T08:00:00+00:00', project_id='reify', success=0, tokens=(500, 50, 3)
    ),
    telemetry_row(FIRST_TOKENED),
)
NOISE_ROWS = (
    untokened_telemetry_row('2026-10-07T09:00:00+00:00', backend='mem0'),
    untokened_telemetry_row(
        '2026-10-06T07:00:00+00:00',
        backend='sqlite_task_backend',
        operation='update_task',
        project_id='solar_challenge',
    ),
)


@pytest.fixture
def harness(monkeypatch) -> ModuleType:
    """The CLI module, loaded with its own directory importable, as a direct run resolves it."""
    monkeypatch.syspath_prepend(str(LME_DIR))
    return load_script_module(LME_DIR / 'harness.py', mod_name='lme_harness')


def _offline() -> Any:
    raise AssertionError('incumbent-cost is offline post-processing: it must never build live deps')


def _write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> Path:
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def _scalar(spec: LlmArmSpec, metric_id: LlmMetricId, value: float) -> MetricsRecord:
    metric = Metric(metric_id=metric_id, kind='scalar', value=value, n=3)
    return record_for(spec, metric, measured_at=FINISHED_AT, incomplete=False)


def _control_run(root: Path, arm_id: str, scratch: str, tokens: float, usd: float) -> Path:
    spec = incumbent_control_spec(arm_id=arm_id, scratch_group_id=scratch)
    records = [
        _scalar(spec, LlmMetricId.EPISODE_LATENCY_P95, 4000.0),
        _scalar(spec, LlmMetricId.TOKENS_PER_EPISODE, tokens),
        _scalar(spec, LlmMetricId.USD_PER_EPISODE, usd),
    ]
    return write_run(root / arm_id, run_manifest_for(spec), records)


def _spec_file(path: Path, spec: Any) -> Path:
    path.write_text(json.dumps(spec.model_dump(mode='json')))
    return path


@dataclass(frozen=True)
class CostInputs:
    telemetry: Path
    pricing_spec: LlmArmSpec
    pricing_spec_path: Path
    control_runs: tuple[Path, ...]
    out_dir: Path

    @property
    def telemetry_out(self) -> Path:
        return self.out_dir / PRODUCTION_TELEMETRY_FILENAME

    @property
    def cost_out(self) -> Path:
        return self.out_dir / INCUMBENT_COST_FILENAME

    def argv(self, until: str = UNTIL) -> list[str]:
        argv = [
            'incumbent-cost',
            '--telemetry', str(self.telemetry),
            '--until', until,
            '--pricing-spec', str(self.pricing_spec_path),
            '--out-dir', str(self.out_dir),
        ]
        for run in self.control_runs:
            argv += ['--control-run', str(run)]
        return argv


def _cost_inputs(
    tmp_path: Path,
    rows: tuple[dict[str, object], ...] = (*TOKENED_ROWS, *NOISE_ROWS, PRE_START_ROW),
    pricing_spec: LlmArmSpec | None = None,
) -> CostInputs:
    spec = pricing_spec or incumbent_control_spec(arm_id='incumbent-generic-a')
    return CostInputs(
        telemetry=_write_jsonl(tmp_path / 'telemetry-raw.jsonl', rows),
        pricing_spec=spec,
        pricing_spec_path=_spec_file(tmp_path / 'pricing-spec.json', spec),
        control_runs=(
            _control_run(tmp_path, 'incumbent-generic-a', 'evalmem_ctrl_a', 15688.47, 0.0027),
            _control_run(tmp_path, 'incumbent-generic-b', 'evalmem_ctrl_b', 15700.0, 0.0028),
        ),
        out_dir=tmp_path / 'out',
    )


def _expected(inputs: CostInputs) -> tuple[TelemetryWindow, IncumbentCost]:
    rows = [json.loads(line) for line in inputs.telemetry.read_text().splitlines()]
    window = select_llm_writes(rows, until=datetime.fromisoformat(UNTIL))
    cost = derive_incumbent_cost(
        window,
        pricing_spec=inputs.pricing_spec,
        control_records=[load_metrics_records(run) for run in inputs.control_runs],
    )
    return window, cost


def test_incumbent_cost_writes_the_window_and_the_cost_derived_from_its_inputs(
    harness, tmp_path, capsys
):
    inputs = _cost_inputs(tmp_path)

    code = harness.main(inputs.argv(), deps=_offline)

    assert code == harness.EXIT_OK
    window, cost = _expected(inputs)
    assert load_llm_writes(inputs.telemetry_out) == window.writes
    assert len(window.writes) == len(TOKENED_ROWS)
    assert inputs.cost_out.read_text() == serialize_incumbent_cost(cost)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert window.start.isoformat() in out
    assert UNTIL in out
    assert f'writes {cost.production.writes}' in out
    assert str(cost.production.usd_per_day) in out
    assert str(cost.production.projected_usd_per_30_days) in out
    for unit in cost.replay:
        assert any(unit.arm_id in line and str(unit.usd_per_episode) in line for line in lines)
    assert lines[-2:] == [f'wrote: {inputs.telemetry_out}', f'wrote: {inputs.cost_out}']


def _refused(harness: ModuleType, argv: list[str], capsys: Any) -> str:
    code = harness.main(argv, deps=_offline)

    assert code == harness.EXIT_REFUSED
    err = capsys.readouterr().err
    assert err.startswith('error: ')
    return err


@pytest.mark.parametrize('existing', [PRODUCTION_TELEMETRY_FILENAME, INCUMBENT_COST_FILENAME])
def test_incumbent_cost_never_overwrites_an_existing_output(harness, tmp_path, capsys, existing):
    inputs = _cost_inputs(tmp_path)
    inputs.out_dir.mkdir()
    kept = inputs.out_dir / existing
    kept.write_text('committed\n')

    err = _refused(harness, inputs.argv(), capsys)

    assert str(kept) in err
    assert kept.read_text() == 'committed\n'
    assert sorted(path.name for path in inputs.out_dir.iterdir()) == [existing]


def test_incumbent_cost_refuses_a_malformed_telemetry_line_naming_it(harness, tmp_path, capsys):
    inputs = _cost_inputs(tmp_path)
    lines = inputs.telemetry.read_text().splitlines()
    lines[2] = '{"created_at": '
    inputs.telemetry.write_text('\n'.join(lines) + '\n')

    err = _refused(harness, inputs.argv(), capsys)

    assert str(inputs.telemetry) in err
    assert 'line 3' in err
    assert not inputs.out_dir.exists()


def test_incumbent_cost_refuses_a_telemetry_line_that_is_not_an_object(harness, tmp_path, capsys):
    inputs = _cost_inputs(tmp_path)
    inputs.telemetry.write_text(inputs.telemetry.read_text() + '[1, 2]\n')

    err = _refused(harness, inputs.argv(), capsys)

    assert f'line {len(TOKENED_ROWS) + len(NOISE_ROWS) + 2}' in err
    assert not inputs.out_dir.exists()


def test_incumbent_cost_refuses_a_malformed_telemetry_row(harness, tmp_path, capsys):
    naive = telemetry_row('2026-10-06T12:00:00')
    inputs = _cost_inputs(tmp_path, rows=(*TOKENED_ROWS, naive, PRE_START_ROW))

    err = _refused(harness, inputs.argv(), capsys)

    assert str(inputs.telemetry) in err
    assert '2026-10-06T12:00:00' in err
    assert not inputs.out_dir.exists()


def test_incumbent_cost_refuses_an_unpriced_pricing_spec(harness, tmp_path, capsys):
    inputs = _cost_inputs(tmp_path, pricing_spec=llm_spec())

    err = _refused(harness, inputs.argv(), capsys)

    assert 'IncumbentCostError' in err
    assert 'pricing' in err
    assert not inputs.out_dir.exists()


@pytest.mark.parametrize(
    ('rows', 'error_type'),
    [
        (
            (*TOKENED_ROWS, untokened_telemetry_row('2026-10-06T09:00:00+00:00'), PRE_START_ROW),
            'TelemetryAccountingError',
        ),
        (TOKENED_ROWS, 'TelemetryWindowError'),
    ],
)
def test_incumbent_cost_refuses_telemetry_it_cannot_window_or_account(
    harness, tmp_path, capsys, rows, error_type
):
    inputs = _cost_inputs(tmp_path, rows=rows)

    err = _refused(harness, inputs.argv(), capsys)

    assert err.startswith(f'error: {error_type}: ')
    assert not inputs.out_dir.exists()


@pytest.mark.parametrize('until', ['2026-10-08T00:00:00', 'yesterday'])
def test_incumbent_cost_rejects_an_until_without_a_utc_offset(harness, tmp_path, capsys, until):
    inputs = _cost_inputs(tmp_path)

    with pytest.raises(SystemExit) as raised:
        harness.main(inputs.argv(until=until), deps=_offline)

    assert raised.value.code == 2
    assert 'argument --until' in capsys.readouterr().err
    assert not inputs.out_dir.exists()


def test_incumbent_cost_refuses_an_embedding_arm_control_run(harness, tmp_path, capsys):
    inputs = _cost_inputs(tmp_path)
    embedding_run = write_run(tmp_path / 'emb', run_manifest_for(embedding_spec()), [])
    argv = [*inputs.argv(), '--control-run', str(embedding_run)]

    err = _refused(harness, argv, capsys)

    assert 'embedding arm' in err
    assert not inputs.out_dir.exists()
