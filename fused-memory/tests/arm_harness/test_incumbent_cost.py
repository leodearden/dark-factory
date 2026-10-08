"""The incumbent's measured LLM spend: production write telemetry and the controls' unit cost."""

import json
import re
from datetime import UTC, date, datetime

import pytest
from shared.memory_eval_metrics import Metric, canonical_json_text

from arm_harness._fakes import incumbent_control_spec, llm_spec
from fused_memory.arm_harness.arm_spec import LlmArmSpec, TokenPricing
from fused_memory.arm_harness.incumbent_cost import (
    PROJECTION_DAYS,
    DailySpend,
    IncumbentCostError,
    LlmWriteOperation,
    LlmWriteTelemetry,
    ProductionCost,
    ProjectSpend,
    ReplayUnitCost,
    TelemetryAccountingError,
    TelemetryWindow,
    TelemetryWindowError,
    derive_incumbent_cost,
    load_incumbent_cost,
    load_llm_writes,
    production_cost,
    replay_unit_cost,
    select_llm_writes,
    serialize_incumbent_cost,
    serialize_llm_writes,
)
from fused_memory.arm_harness.metrics_record import (
    DeltaOf,
    LlmMetricId,
    MetricsRecord,
    record_for,
)

PRE_START = '2026-10-05T10:59:00.000001+00:00'
FIRST_TOKENED = '2026-10-05T11:23:27.123456+00:00'
LATER = '2026-10-06T08:00:00+00:00'
UNTIL_TEXT = '2026-10-08T00:00:00+00:00'
UNTIL = datetime.fromisoformat(UNTIL_TEXT)
_UNTOKENED = (None, None, None)


def telemetry_row(
    created_at: str,
    *,
    backend: str = 'graphiti',
    operation: str | None = 'add_memory',
    project_id: str | None = 'dark_factory',
    success: int = 1,
    tokens: tuple[int | None, int | None, int | None] = (1000, 100, 9),
) -> dict[str, object]:
    """One object shaped exactly like a ``telemetry_query.py`` output line."""
    input_tokens, output_tokens, llm_calls = tokens
    total_tokens = (
        None if input_tokens is None or output_tokens is None else input_tokens + output_tokens
    )
    return {
        'created_at': created_at,
        'operation': operation,
        'project_id': project_id,
        'backend': backend,
        'success': success,
        'duration_ms': 1234.5,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'llm_calls': llm_calls,
    }


def untokened_row(created_at: str, **overrides) -> dict[str, object]:
    return telemetry_row(created_at, tokens=_UNTOKENED, **overrides)


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def write_at(text: str, **overrides) -> LlmWriteTelemetry:
    return LlmWriteTelemetry.from_telemetry_row(telemetry_row(text, **overrides))


def test_only_graphiti_add_memory_and_add_episode_rows_are_llm_writes():
    rows = [
        untokened_row(PRE_START),
        telemetry_row(FIRST_TOKENED),
        telemetry_row(LATER, operation='add_episode'),
        untokened_row('2026-10-06T09:00:00+00:00', backend='mem0'),
        untokened_row('2026-10-06T10:00:00+00:00', operation='update_edge'),
        untokened_row(
            '2026-10-06T11:00:00+00:00',
            backend='sqlite_task_backend',
            operation='update_task',
            project_id='solar_challenge',
        ),
        untokened_row('2026-10-06T12:00:00+00:00', backend='graphiti', operation=None),
    ]

    window = select_llm_writes(rows, until=UNTIL)

    assert [(w.created_at, w.operation) for w in window.writes] == [
        (at(FIRST_TOKENED), LlmWriteOperation.ADD_MEMORY),
        (at(LATER), LlmWriteOperation.ADD_EPISODE),
    ]


def test_window_runs_from_the_first_token_bearing_write_up_to_but_excluding_until():
    rows = [
        untokened_row('2026-10-08T00:00:01+00:00'),
        telemetry_row(UNTIL_TEXT),
        telemetry_row(LATER, project_id='reify'),
        telemetry_row(FIRST_TOKENED),
        untokened_row(PRE_START),
        untokened_row('2026-10-01T00:00:00+00:00'),
    ]

    window = select_llm_writes(rows, until=UNTIL)

    assert window.start == at(FIRST_TOKENED)
    assert window.end == UNTIL
    assert [w.created_at for w in window.writes] == [at(FIRST_TOKENED), at(LATER)]
    assert [w.project_id for w in window.writes] == ['dark_factory', 'reify']


@pytest.mark.parametrize(
    'pre_start_rows',
    [
        [],
        [untokened_row(PRE_START, backend='mem0')],
        [untokened_row(PRE_START, operation='update_edge')],
    ],
)
def test_a_dump_that_does_not_reach_back_past_the_telemetry_start_is_refused(pre_start_rows):
    rows = [*pre_start_rows, telemetry_row(FIRST_TOKENED), telemetry_row(LATER)]

    with pytest.raises(TelemetryWindowError, match=re.escape(FIRST_TOKENED)):
        select_llm_writes(rows, until=UNTIL)


def test_an_untokened_llm_write_inside_the_window_is_unknown_not_zero():
    rows = [
        untokened_row(PRE_START),
        telemetry_row(FIRST_TOKENED),
        untokened_row('2026-10-06T01:00:00+00:00'),
        telemetry_row(LATER),
        untokened_row('2026-10-07T01:00:00+00:00', operation='add_episode'),
    ]

    with pytest.raises(TelemetryAccountingError) as caught:
        select_llm_writes(rows, until=UNTIL)

    message = str(caught.value)
    assert '2 ' in message
    assert '2026-10-06T01:00:00+00:00' in message
    assert 'unknown, not zero' in message


def test_a_partially_tokened_llm_write_inside_the_window_is_refused():
    rows = [
        untokened_row(PRE_START),
        telemetry_row(FIRST_TOKENED),
        telemetry_row(LATER, tokens=(1000, 100, None)),
    ]

    with pytest.raises(TelemetryAccountingError, match=re.escape(LATER)):
        select_llm_writes(rows, until=UNTIL)


def test_untokened_writes_after_until_are_outside_the_accounting():
    rows = [
        untokened_row(PRE_START),
        telemetry_row(FIRST_TOKENED),
        untokened_row('2026-10-08T03:00:00+00:00'),
    ]

    assert len(select_llm_writes(rows, until=UNTIL).writes) == 1


@pytest.mark.parametrize(
    'rows',
    [
        [untokened_row(PRE_START), untokened_row(LATER)],
        [untokened_row(PRE_START), telemetry_row(UNTIL_TEXT)],
        [],
    ],
)
def test_no_token_bearing_llm_write_before_until_is_refused(rows):
    with pytest.raises(TelemetryWindowError, match=re.escape(UNTIL_TEXT)):
        select_llm_writes(rows, until=UNTIL)


def test_a_naive_until_is_refused():
    rows = [untokened_row(PRE_START), telemetry_row(FIRST_TOKENED)]

    with pytest.raises(TelemetryWindowError, match='until'):
        select_llm_writes(rows, until=datetime(2026, 10, 8))


def test_a_naive_created_at_is_refused_at_the_boundary():
    with pytest.raises(ValueError, match=re.escape('2026-10-05T12:00:00')):
        LlmWriteTelemetry.from_telemetry_row(telemetry_row('2026-10-05T12:00:00'))


def test_a_naive_created_at_is_refused_by_selection():
    rows = [untokened_row('2026-10-05T10:00:00'), telemetry_row(FIRST_TOKENED)]

    with pytest.raises(ValueError, match=re.escape('2026-10-05T10:00:00')):
        select_llm_writes(rows, until=UNTIL)


def test_a_total_that_is_not_input_plus_output_is_refused_naming_the_row():
    row = telemetry_row(LATER) | {'total_tokens': 1}

    with pytest.raises(ValueError, match=re.escape(LATER)) as caught:
        LlmWriteTelemetry.from_telemetry_row(row)

    assert 'total_tokens' in str(caught.value)


@pytest.mark.parametrize(('raw', 'parsed'), [(1, True), (0, False)])
def test_success_parses_to_a_bool(raw, parsed):
    assert write_at(LATER, success=raw).success is parsed


def test_a_parsed_write_keeps_only_what_costing_needs():
    write = write_at(LATER, tokens=(1650, 95, 7))

    assert set(LlmWriteTelemetry.model_fields) == {
        'created_at',
        'operation',
        'project_id',
        'success',
        'input_tokens',
        'output_tokens',
        'llm_calls',
    }
    assert write.created_at == at(LATER)
    assert write.created_at.tzinfo is not None
    assert write.operation is LlmWriteOperation.ADD_MEMORY
    assert (write.input_tokens, write.output_tokens, write.llm_calls) == (1650, 95, 7)


def test_a_valid_window_constructs():
    writes = (write_at(FIRST_TOKENED), write_at(LATER))

    window = TelemetryWindow(start=at(FIRST_TOKENED), end=UNTIL, writes=writes)

    assert window.writes == writes


@pytest.mark.parametrize(
    ('start', 'end', 'stamps', 'invariant'),
    [
        (FIRST_TOKENED, UNTIL_TEXT, (LATER, FIRST_TOKENED), 'sorted'),
        (FIRST_TOKENED, LATER, (FIRST_TOKENED, LATER), 'end'),
        (FIRST_TOKENED, UNTIL_TEXT, (FIRST_TOKENED, '2026-10-09T00:00:00+00:00'), 'end'),
        (UNTIL_TEXT, FIRST_TOKENED, (), 'end'),
        (FIRST_TOKENED, UNTIL_TEXT, (), 'empty'),
        (PRE_START, UNTIL_TEXT, (FIRST_TOKENED, LATER), 'first write'),
    ],
)
def test_window_construction_enforces_its_invariant(start, end, stamps, invariant):
    with pytest.raises(ValueError, match=invariant):
        TelemetryWindow(
            start=at(start), end=at(end), writes=tuple(write_at(stamp) for stamp in stamps)
        )


PRICING = TokenPricing(usd_per_mtok_input=0.15, usd_per_mtok_output=0.60)
MEASURED_AT = datetime(2026, 10, 7, 1, 34, tzinfo=UTC)
SAME_DAY_FAILED = '2026-10-06T08:00:00+00:00'
SAME_DAY_OK = '2026-10-06T21:30:00+00:00'
LAST_DAY = '2026-10-07T12:00:00+00:00'


def costed_window() -> TelemetryWindow:
    writes = (
        write_at(FIRST_TOKENED, tokens=(1000, 100, 9)),
        write_at(SAME_DAY_FAILED, project_id='reify', success=0, tokens=(500, 50, 3)),
        write_at(SAME_DAY_OK, tokens=(2000, 200, 12)),
        write_at(LAST_DAY, tokens=(1500, 150, 8)),
    )
    return TelemetryWindow(start=at(FIRST_TOKENED), end=UNTIL, writes=writes)


def test_production_totals_count_every_write_failed_ones_included():
    cost = production_cost(costed_window(), PRICING)

    assert (cost.writes, cost.failed_writes, cost.llm_calls) == (4, 1, 32)
    assert (cost.input_tokens, cost.output_tokens) == (5000, 500)
    assert cost.usd == pytest.approx(
        PRICING.usd_for(1000, 100)
        + PRICING.usd_for(500, 50)
        + PRICING.usd_for(2000, 200)
        + PRICING.usd_for(1500, 150)
    )
    assert (cost.window_start, cost.window_end) == (at(FIRST_TOKENED), UNTIL)


def test_production_rates_derive_from_the_totals_and_the_window_length():
    cost = production_cost(costed_window(), PRICING)
    days = (UNTIL - at(FIRST_TOKENED)).total_seconds() / 86400

    assert cost.usd_per_ok_write == cost.usd / 3
    assert cost.tokens_per_write == 5500 / 4
    assert cost.days == days
    assert cost.usd_per_day == cost.usd / days
    assert PROJECTION_DAYS == 30
    assert cost.projected_usd_per_30_days == cost.usd_per_day * 30


def test_production_spend_is_broken_down_by_utc_day_and_by_project():
    cost = production_cost(costed_window(), PRICING)

    assert [(d.day, d.writes) for d in cost.by_day] == [
        (date(2026, 10, 5), 1),
        (date(2026, 10, 6), 2),
        (date(2026, 10, 7), 1),
    ]
    assert cost.by_day[1].usd == pytest.approx(
        PRICING.usd_for(500, 50) + PRICING.usd_for(2000, 200)
    )
    assert [(p.project_id, p.writes) for p in cost.by_project] == [
        ('dark_factory', 3),
        ('reify', 1),
    ]
    assert cost.by_project[1].usd == pytest.approx(PRICING.usd_for(500, 50))
    assert sum(d.usd for d in cost.by_day) == pytest.approx(cost.usd)
    assert sum(p.usd for p in cost.by_project) == pytest.approx(cost.usd)


def test_a_window_with_no_ok_write_has_nothing_to_cost():
    window = TelemetryWindow(
        start=at(FIRST_TOKENED), end=UNTIL, writes=(write_at(FIRST_TOKENED, success=0),)
    )

    with pytest.raises(IncumbentCostError, match='ok write'):
        production_cost(window, PRICING)


@pytest.mark.parametrize(
    ('field', 'replacement', 'named'),
    [
        ('by_day', (DailySpend(day=date(2026, 10, 5), writes=1, usd=0.1),), 'by_day'),
        ('by_project', (ProjectSpend(project_id='reify', writes=1, usd=0.1),), 'by_project'),
        ('usd_per_day', 1.0, 'usd_per_day'),
        ('days', 1.0, 'days'),
    ],
)
def test_production_cost_construction_checks_its_derivation(field, replacement, named):
    valid = production_cost(costed_window(), PRICING)
    data = dict(valid) | {field: replacement}

    with pytest.raises(ValueError, match=named) as caught:
        ProductionCost(**data)

    message = str(caught.value)
    assert str(replacement if not isinstance(replacement, tuple) else 1) in message


def control_record(
    metric_id: LlmMetricId,
    value: float,
    *,
    spec: LlmArmSpec | None = None,
    n: int = 200,
    incomplete: bool = False,
    delta_of: DeltaOf | None = None,
) -> MetricsRecord:
    metric = Metric(metric_id=metric_id, kind='scalar', value=value, n=n)
    return record_for(
        spec or incumbent_control_spec(arm_id='incumbent-generic-a'),
        metric,
        measured_at=MEASURED_AT,
        incomplete=incomplete,
        delta_of=delta_of,
    )


def control_run_records(arm_id: str = 'incumbent-generic-a', tokens: float = 15688.47):
    spec = incumbent_control_spec(arm_id=arm_id)
    return (
        control_record(LlmMetricId.EPISODE_LATENCY_P95, 42828.0, spec=spec),
        control_record(LlmMetricId.TOKENS_PER_EPISODE, tokens, spec=spec, n=199),
        control_record(LlmMetricId.USD_PER_EPISODE, 0.0027364, spec=spec),
    )


def test_replay_unit_cost_reads_one_control_runs_token_and_usd_records():
    assert replay_unit_cost(control_run_records()) == ReplayUnitCost(
        arm_id='incumbent-generic-a', n=200, tokens_per_episode=15688.47, usd_per_episode=0.0027364
    )


def _without(metric_id: LlmMetricId):
    return tuple(r for r in control_run_records() if r.metric.metric_id != metric_id)


@pytest.mark.parametrize(
    ('records', 'named'),
    [
        (_without(LlmMetricId.TOKENS_PER_EPISODE), 'tokens-per-episode'),
        (_without(LlmMetricId.USD_PER_EPISODE), 'usd-per-episode'),
        (
            (*control_run_records(), control_record(LlmMetricId.USD_PER_EPISODE, 0.1)),
            'usd-per-episode',
        ),
        (
            (*control_run_records(), *control_run_records(arm_id='incumbent-generic-b')[1:2]),
            'incumbent-generic-b',
        ),
        ((), 'arm'),
        (
            (
                control_record(LlmMetricId.TOKENS_PER_EPISODE, 1.0, incomplete=True),
                control_record(LlmMetricId.USD_PER_EPISODE, 0.1),
            ),
            'incomplete',
        ),
        (
            (
                control_record(LlmMetricId.TOKENS_PER_EPISODE, 1.0, spec=llm_spec()),
                control_record(LlmMetricId.USD_PER_EPISODE, 0.0, spec=llm_spec()),
            ),
            'control',
        ),
        (
            (
                control_record(LlmMetricId.TOKENS_PER_EPISODE, 1.0),
                control_record(
                    LlmMetricId.USD_PER_EPISODE,
                    -0.1,
                    delta_of=DeltaOf(
                        minuend_arm_id='incumbent-generic-a', subtrahend_arm_id='incumbent-b'
                    ),
                ),
            ),
            'delta',
        ),
    ],
)
def test_replay_unit_cost_refuses_records_that_are_not_one_complete_control_run(records, named):
    with pytest.raises(IncumbentCostError, match=named):
        replay_unit_cost(records)


def test_incumbent_cost_carries_the_price_production_and_each_control_in_order():
    spec = incumbent_control_spec(arm_id='incumbent-generic-a')
    run_a = control_run_records('incumbent-generic-a', tokens=15688.47)
    run_b = control_run_records('incumbent-generic-b', tokens=15700.0)

    cost = derive_incumbent_cost(costed_window(), pricing_spec=spec, control_records=[run_a, run_b])

    assert cost.schema_version == 1
    assert cost.pricing == spec.pricing
    assert cost.pricing_arm_id == 'incumbent-generic-a'
    assert spec.pricing is not None
    assert cost.production == production_cost(costed_window(), spec.pricing)
    assert cost.replay == (replay_unit_cost(run_a), replay_unit_cost(run_b))


@pytest.mark.parametrize(
    ('pricing_spec', 'control_records', 'named'),
    [
        (llm_spec(), [control_run_records()], 'pricing'),
        (incumbent_control_spec(), [], 'control'),
        (
            incumbent_control_spec(),
            [control_run_records(), control_run_records()],
            'incumbent-generic-a',
        ),
    ],
)
def test_derive_incumbent_cost_refuses_an_unpriced_spec_or_a_bad_control_set(
    pricing_spec, control_records, named
):
    with pytest.raises(IncumbentCostError, match=named):
        derive_incumbent_cost(
            costed_window(), pricing_spec=pricing_spec, control_records=control_records
        )


def test_incumbent_cost_round_trips_through_its_canonical_serialization(tmp_path):
    cost = derive_incumbent_cost(
        costed_window(),
        pricing_spec=incumbent_control_spec(),
        control_records=[control_run_records()],
    )
    path = tmp_path / 'incumbent-cost.json'
    text = serialize_incumbent_cost(cost)
    path.write_text(text)

    assert text == canonical_json_text(cost.model_dump(mode='json'))
    assert load_incumbent_cost(path) == cost


def test_llm_writes_round_trip_as_one_canonical_json_object_per_line(tmp_path):
    writes = costed_window().writes
    path = tmp_path / 'production-telemetry.jsonl'
    text = serialize_llm_writes(writes)
    path.write_text(text)

    lines = text.splitlines()
    assert len(lines) == len(writes)
    assert lines == [json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False) for line in lines]
    assert load_llm_writes(path) == writes
