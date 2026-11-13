"""The quality series one project serves: axis, per-metric points, bands, baselines and alarms."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import pytest
from _quality_rows import (
    NO_COST,
    NO_OUTCOMES,
    epochs_from,
    history,
    train,
    write_landing_events,
    write_quality_db,
)
from _runs_db_fixture import EventRow
from shared.project_id import project_id_for_root
from shared.quality_series.git_lines import ModuleLines
from shared.quality_series.landings import GitCrossCheck, LinesBase
from shared.quality_series.rows import (
    CharterClass,
    CharterClassSource,
    Cost,
    LandingRow,
    Outcomes,
)
from shared.quality_series.stats import (
    CostEffect,
    Effect,
    NotYet,
    RegimeEvent,
    before_after_cost,
    before_after_rate,
    block_rate_datum,
    bootstrap_interval,
    content_seed,
    cost_fit,
    ewma_limits,
    log_residuals,
    reference_cost,
    wilson_interval,
)

from dashboard.data.datum import DatumContractError, DatumState, validate_datum
from dashboard.data.db import DbPool
from dashboard.data.quality_series import (
    QUALITY_FRESHNESS_BOUND_SECONDS,
    Alarm,
    AlarmDirection,
    Provenance,
    QualitySeriesContractError,
    QualitySeriesInvariant,
    quality_series_view,
    read_quality_series,
    validate_quality_series,
)
from dashboard.data.quality_store import QualitySample

T = TypeVar('T')
T0 = datetime(2026, 9, 1, tzinfo=UTC)
AS_OF = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
FRESH = Provenance(as_of=AS_OF, state=DatumState.FRESH, reason=None, freshness_bound_seconds=1200,
                   served_at=AS_OF)

AXIS_KEYS = {
    'ordinal', 'merge_sha', 'task_id', 'landed_at', 'unit_id', 'git_first_parent', 'late_claim',
}
RATE_METRICS = (
    'rejected', 'reviewer_blocking', 'flakes_confirmed', 'flakes_real_red', 'review_rounds_ge2',
    'turns_capped',
)


def _rejections(xs: Sequence[int | None]) -> Callable[[int], Outcomes]:
    """Landing i rejected when xs[i] is 1, not when 0, unmeasured when None."""

    def outcome(i: int) -> Outcomes:
        x = xs[i]
        return replace(
            NO_OUTCOMES,
            rejected=None if x is None else bool(x),
            rejection_events=x,
        )

    return outcome


UNREVIEWED = -1


def _reviews(xs: Sequence[int | None]) -> Callable[[int], Outcomes]:
    """Landing i: None before the source, UNREVIEWED, or reviewed with blocking == xs[i]."""

    def outcome(i: int) -> Outcomes:
        x = xs[i]
        if x is None:
            return NO_OUTCOMES
        if x == UNREVIEWED:
            return replace(NO_OUTCOMES, reviewed=False)
        return replace(NO_OUTCOMES, reviewed=True, review_blocking=bool(x))

    return outcome


def _every(period: int, n: int) -> list[int]:
    """1 on every *period*-th landing from the first, else 0: a rate of 1/period."""
    return [1 if i % period == 0 else 0 for i in range(n)]


def _wire(
    rows: list[LandingRow], *, window: int | None = None, events=(),
    git_check: GitCrossCheck | None = None,
) -> dict:
    return quality_series_view(
        rows, window_landings=window, provenance=FRESH, events=events, git_check=git_check,
    ).to_wire()


def _metric(rows: list[LandingRow], metric: str, **kwargs) -> dict:
    return _wire(rows, **kwargs)['metrics'][metric]


def test_the_axis_serves_the_newest_window_newest_last() -> None:
    rows = history(150, start=T0)

    windowed = _wire(rows, window=100)['axis']
    everything = _wire(rows, window=None)['axis']

    assert [point['ordinal'] for point in windowed] == list(range(50, 150))
    assert [point['ordinal'] for point in everything] == list(range(150))
    assert set(windowed[-1]) == AXIS_KEYS
    newest = rows[-1].landing
    assert windowed[-1] == {
        'ordinal': 149, 'merge_sha': newest.merge_sha, 'task_id': newest.task_id,
        'landed_at': newest.landed_at.isoformat(), 'unit_id': newest.unit_id,
        'git_first_parent': True, 'late_claim': False,
    }


def test_off_first_parent_and_late_claim_landings_stay_on_the_axis() -> None:
    rows = history(3, start=T0)
    side = replace(rows[1].landing, git_first_parent=False)
    late = replace(
        rows[2].landing, late_claim=True, unit_id=f'{rows[2].landing.merge_sha}:late',
        lines_base=LinesBase.UNMEASURABLE, base_sha=None,
    )
    rows = [rows[0], replace(rows[1], landing=side), replace(rows[2], landing=late)]

    axis = _wire(rows)['axis']

    assert [(p['git_first_parent'], p['late_claim']) for p in axis] == [
        (True, False), (False, False), (True, True),
    ]


def test_every_metric_serves_one_point_per_axis_entry() -> None:
    rows = history(120, start=T0, outcome=_rejections(_every(4, 120)))

    wire = _wire(rows, window=100)

    for metric in RATE_METRICS:
        served = wire['metrics'][metric]
        assert len(served['points']) == len(wire['axis']) == 100, metric
        for band in ('center', 'lo', 'hi', 'n'):
            assert len(served['rolling'][band]) == 100, (metric, band)


def test_reviewer_blocking_is_null_before_its_source_and_for_unreviewed_landings() -> None:
    xs: list[int | None] = [None] * 10 + [UNREVIEWED, 0, 1] * 10
    epoch = T0 + timedelta(hours=2 * 10)
    rows = history(
        40, start=T0, outcome=_reviews(xs),
        epochs=replace(epochs_from(T0), review_completed=epoch),
    )

    points = _metric(rows, 'reviewer_blocking')['points']

    assert points == [None] * 10 + [None, 0, 1] * 10


def test_a_window_before_its_source_began_is_unknown_naming_the_date() -> None:
    began = datetime(2026, 10, 21, tzinfo=UTC)
    rows = history(
        40, start=T0, epochs=replace(epochs_from(T0), review_completed=began),
    )

    served = _metric(rows, 'reviewer_blocking')

    assert served['points'] == [None] * 40
    assert served['headline']['state'] == 'unknown'
    assert served['headline']['value'] is None
    assert '2026-10-21' in served['headline']['reason']
    assert served['rule']['measured_since'] == began.isoformat()


def test_a_full_block_headline_is_the_rate_over_the_last_hundred_measured() -> None:
    xs = _every(5, 130)
    rows = history(130, start=T0, outcome=_rejections(xs))
    expected = block_rate_datum(xs, [row.landing.task_id for row in rows])

    headline = _metric(rows, 'rejected')['headline']

    assert headline == {
        'value': {
            'form': 'rate', 'n': 100, 'k': expected.k, 'value': expected.rate,
            'lo': expected.lo, 'hi': expected.hi,
        },
        'as_of': AS_OF.isoformat(),
        'state': 'fresh',
        'reason': None,
        'freshness_bound_seconds': 1200,
    }


def test_forty_landings_are_k_of_n_with_a_band_and_no_chart_statistics() -> None:
    xs = _every(4, 40)
    rows = history(40, start=T0, outcome=_rejections(xs))

    served = _metric(rows, 'rejected')

    lo, hi = wilson_interval(10, 40)
    assert served['headline']['state'] == 'fresh'
    assert served['headline']['value'] == {
        'form': 'k_of_n', 'n': 40, 'k': 10, 'value': 0.25, 'lo': lo, 'hi': hi,
    }
    assert served['ewma'] is None
    assert served['alarms'] == []
    assert served['baseline']['p0'] is None
    assert served['baseline']['state'] is None


def test_under_thirty_measured_landings_are_unknown_naming_n() -> None:
    rows = history(20, start=T0, outcome=_rejections(_every(4, 20)))

    headline = _metric(rows, 'rejected')['headline']

    assert headline['state'] == 'unknown'
    assert headline['value'] is None
    assert 'n=20' in headline['reason']


def test_the_rule_is_structured_and_names_its_source() -> None:
    began = datetime(2026, 10, 21, tzinfo=UTC)
    rows = history(3, start=began, epochs=replace(epochs_from(began), review_completed=began))

    rule = _metric(rows, 'reviewer_blocking')['rule']

    assert set(rule) == {
        'counts', 'excludes', 'population', 'window_landings', 'measured_since', 'source',
    }
    assert (rule['window_landings'], rule['measured_since'], rule['source']) == (
        100, began.isoformat(), 'review_completed',
    )
    assert all(isinstance(rule[label], str) and rule[label]
               for label in ('counts', 'excludes', 'population'))


def test_a_source_with_no_row_in_runs_db_is_no_producer_not_zero() -> None:
    rows = history(
        60, start=T0, epochs=replace(epochs_from(T0), flake_occurrence=None),
        outcome=lambda i: replace(
            NO_OUTCOMES, flake_confirmed=False, flake_real_red=False, flake_rows=0,
        ),
    )

    served = _metric(rows, 'flakes_confirmed')

    assert served['rule']['measured_since'] is None
    assert served['points'] == [None] * 60
    assert served['headline']['state'] == 'unknown'
    assert served['headline']['reason'].startswith('no producer has written here')


def test_rolling_is_the_wilson_band_over_the_last_hundred_measured_landings() -> None:
    xs: list[int | None] = [None if i % 7 == 3 else (1 if i % 3 == 0 else 0) for i in range(180)]
    rows = history(180, start=T0, outcome=_rejections(xs))

    rolling = _metric(rows, 'rejected')['rolling']

    for t in range(180):
        block = [x for x in xs[:t + 1] if x is not None][-100:]
        assert rolling['n'][t] == len(block)
        if len(block) < 30:
            assert (rolling['center'][t], rolling['lo'][t], rolling['hi'][t]) == (None, None, None)
        else:
            lo, hi = wilson_interval(sum(block), len(block))
            assert (rolling['center'][t], rolling['lo'][t], rolling['hi'][t]) == (
                sum(block) / len(block), lo, hi,
            )


def test_a_baseline_freezes_at_two_hundred_measured_landings() -> None:
    xs = _every(5, 250)
    rows = history(250, start=T0, outcome=_rejections(xs))

    served = _metric(rows, 'rejected')

    assert served['baseline'] == {'p0': 0.2, 'state': 'frozen', 'since_ordinal': 0}
    assert set(served['ewma']) == {'line', 'ucl'}
    assert len(served['ewma']['line']) == len(served['ewma']['ucl']) == 250
    assert served['ewma']['ucl'][-1] == ewma_limits(0.2, 0.05, 3, 250).ucl


def test_a_baseline_is_provisional_from_one_hundred_measured_landings() -> None:
    rows = history(150, start=T0, outcome=_rejections(_every(5, 150)))

    served = _metric(rows, 'rejected')

    assert served['baseline']['state'] == 'provisional'
    assert served['ewma'] is not None


def test_a_degenerate_baseline_serves_no_ewma_and_no_alarms() -> None:
    for constant in (0, 1):
        rows = history(150, start=T0, outcome=_rejections([constant] * 150))

        served = _metric(rows, 'rejected')

        assert served['baseline']['p0'] == float(constant)
        assert served['ewma'] is None
        assert served['alarms'] == []


SHIFT = 300
"""The landing index where the shifted series moves from 1-in-10 to 1-in-2."""


def _shifted_rows() -> list[LandingRow]:
    xs = _every(10, SHIFT) + _every(2, 60)
    return history(len(xs), start=T0, outcome=_rejections(xs))


def _up_alarms(rows: list[LandingRow], events: Sequence[RegimeEvent] = ()) -> list[dict]:
    alarms = _metric(rows, 'rejected', events=events)['alarms']
    return [alarm for alarm in alarms if alarm['direction'] == 'up']


def test_an_upward_shift_raises_an_up_alarm_after_its_onset() -> None:
    [alarm, *_] = _up_alarms(_shifted_rows())

    assert alarm['onset_ordinal'] <= alarm['ordinal']
    assert alarm['ordinal'] >= SHIFT
    assert alarm['explained_by'] is None


def test_a_marker_near_the_onset_explains_the_alarm_and_a_far_one_does_not() -> None:
    rows = _shifted_rows()
    near = RegimeEvent('m1', at=rows[SHIFT - 10].landing.landed_at)
    far = RegimeEvent('m1', at=rows[SHIFT - 100].landing.landed_at)

    [explained, *_] = _up_alarms(rows, [near])
    [unexplained, *_] = _up_alarms(rows, [far])

    assert explained['explained_by'] == 'm1'
    assert unexplained['explained_by'] is None


def test_serving_twice_is_byte_equal() -> None:
    rows = _shifted_rows()
    events = [RegimeEvent('m1', at=rows[SHIFT - 10].landing.landed_at)]

    first = json.dumps(_wire(rows, window=100, events=events), sort_keys=True)
    second = json.dumps(_wire(rows, window=100, events=events), sort_keys=True)

    assert first == second


def test_nothing_is_served_as_zero_for_an_unmeasured_landing() -> None:
    rows = history(50, start=T0, cost=lambda i: NO_COST)

    wire = _wire(rows)

    for metric in RATE_METRICS:
        assert 0 not in wire['metrics'][metric]['points'], metric


# --- counts, costs, the cost model, modules, before/after and the no-producer states ---------

INVOCATION_METRICS = ('implementer_invocations', 'turns', 'all_role_usd')


def _known(x: T | None) -> T:
    assert x is not None
    return x


def _unit_outcome(rejections: int, reverifies: int) -> Outcomes:
    return replace(
        NO_OUTCOMES, rejected=rejections > 0, rejection_events=rejections,
        reverify_count=reverifies, verify_failed_rows=0,
    )


def _costed(
    usd: float, lines: int = 100, *, invocations: int = 1, turns: int | None = None,
    iterations: int | None = None, src: int | None = None,
) -> Cost:
    """A landing that cost *usd* over *invocations* implementer invocations and landed *lines*."""
    src_lines = lines if src is None else src
    return replace(
        NO_COST, all_role_usd=usd, by_role=(('implementer', usd),),
        implementer_invocations=invocations, turns=turns, execute_iterations=iterations,
        code_lines=lines, src_lines=src_lines, test_lines=lines - src_lines,
    )


def test_unit_counts_are_carried_once_at_the_unit_head() -> None:
    train_members = {1, 2, 3}
    rows = history(
        5, start=T0, units=[1, 3, 1],
        outcome=lambda i: _unit_outcome(2, 1) if i in train_members else _unit_outcome(0, 0),
    )

    metrics = _wire(rows)['metrics']

    assert metrics['rejections_per_landing']['points'] == [0, 2, 0, 0, 0]
    assert metrics['reverifies_per_landing']['points'] == [0, 1, 0, 0, 0]
    assert metrics['rejections_per_landing']['kind'] == 'count'


def test_a_unit_count_is_unmeasured_on_every_member_when_its_source_is() -> None:
    rows = history(4, start=T0, units=[1, 3])

    points = _metric(rows, 'reverifies_per_landing')['points']

    assert points == [None] * 4


def test_a_count_headline_is_the_mean_with_a_seeded_bootstrap_band() -> None:
    rows = history(130, start=T0, cost=lambda i: _costed(1.0, invocations=1 + i % 4))
    block = rows[-100:]
    values = [float(_known(row.cost.implementer_invocations)) for row in block]
    lo, hi = bootstrap_interval(
        values, statistics.fmean, content_seed(row.landing.task_id for row in block),
    )

    headline = _metric(rows, 'implementer_invocations')['headline']

    assert headline['state'] == 'fresh'
    assert headline['value'] == {
        'form': 'mean', 'n': 100, 'k': None, 'value': statistics.fmean(values),
        'lo': lo, 'hi': hi,
    }


def test_invocation_sourced_metrics_cover_orchestrated_landings_only() -> None:
    orchestrated = [i % 3 != 0 for i in range(60)]
    rows = history(
        60, start=T0,
        cost=lambda i: _costed(2.0, invocations=2, turns=40, iterations=3) if orchestrated[i]
        else NO_COST,
    )

    metrics = _wire(rows)['metrics']

    per_landing = {'implementer_invocations': 2, 'turns': 40, 'all_role_usd': 2.0,
                   'execute_iterations': 3}
    for metric, value in per_landing.items():
        assert metrics[metric]['points'] == [
            value if measured else None for measured in orchestrated
        ], metric
        assert metrics[metric]['headline']['value']['n'] == sum(orchestrated), metric
    for metric in INVOCATION_METRICS:
        assert 'orchestrated landings only' in metrics[metric]['rule']['excludes'], metric


def test_a_cost_headline_is_the_median_with_a_seeded_bootstrap_band() -> None:
    rows = history(130, start=T0, cost=lambda i: _costed(0.5 + (i * 7) % 11))
    block = rows[-100:]
    values = [_known(row.cost.all_role_usd) for row in block]
    lo, hi = bootstrap_interval(
        values, statistics.median, content_seed(row.landing.task_id for row in block),
    )

    served = _metric(rows, 'all_role_usd')

    assert served['kind'] == 'cost'
    assert served['headline']['value'] == {
        'form': 'median', 'n': 100, 'k': None, 'value': statistics.median(values),
        'lo': lo, 'hi': hi,
    }


def _unit_lines(u: int) -> int:
    return 10 + (u * 37) % 400


def _unit_usd(u: int, *, scale: float = 1.0) -> float:
    """A unit's cost: a power law in its lines with a repeating ±20% wobble."""
    wobble = 1 + 0.1 * ((u * 7) % 5 - 2)
    return scale * 0.05 * _unit_lines(u) ** 0.7 * wobble


TRAIN_EVERY = 10
"""Every TRAIN_EVERY-th unit is a train of two, splitting the unit's cost between members."""


def _costed_units(n_units: int, *, shift_at: int | None = None) -> list[LandingRow]:
    """*n_units* verify units, trains included; from unit *shift_at* on each costs three times more."""
    sizes = [2 if u % TRAIN_EVERY == 5 else 1 for u in range(n_units)]
    unit_of = [u for u, size in enumerate(sizes) for _ in range(size)]

    def cost(i: int) -> Cost:
        u = unit_of[i]
        scale = 3.0 if shift_at is not None and u >= shift_at else 1.0
        return _costed(_unit_usd(u, scale=scale) / sizes[u], _unit_lines(u))

    return history(len(unit_of), start=T0, units=sizes, cost=cost)


def _unit_points(rows: Sequence[LandingRow]) -> tuple[list[int], list[float]]:
    """Per verify unit: the head's lines and the members' summed cost."""
    lines: dict[str, int] = {}
    costs: dict[str, float] = {}
    for row in rows:
        unit = row.landing.unit_id
        lines.setdefault(unit, _known(row.cost.code_lines))
        costs[unit] = costs.get(unit, 0.0) + _known(row.cost.all_role_usd)
    return list(lines.values()), list(costs.values())


def test_the_cost_model_fits_per_unit_points_over_the_first_two_hundred_units() -> None:
    rows = _costed_units(250)
    lines, costs = _unit_points(rows)
    a, b = cost_fit(lines[:200], costs[:200])

    cost_model = _wire(rows)['cost_model']

    assert cost_model == {
        'a': a, 'b': b,
        'reference_cost_100': reference_cost(lines[:200], costs[:200]),
        'residual_median': statistics.median(
            r for r in log_residuals(lines[:200], costs[:200], a, b) if r is not None
        ),
        'state': 'frozen',
    }


def test_a_cost_model_is_provisional_from_one_hundred_units_and_absent_below() -> None:
    provisional = _wire(_costed_units(150))['cost_model']
    absent = _wire(_costed_units(80))['cost_model']

    assert provisional['state'] == 'provisional'
    assert provisional['b'] is not None
    assert absent == {
        'a': None, 'b': None, 'reference_cost_100': None, 'residual_median': None, 'state': None,
    }


def test_a_cost_shift_raises_a_sign_of_residual_alarm() -> None:
    rows = _costed_units(300, shift_at=250)
    first_shifted = next(
        row.landing.ordinal for row in rows if row.landing.unit_id == 'sha-00250'
    )

    alarms = _metric(rows, 'all_role_usd')['alarms']

    assert [alarm for alarm in alarms if alarm['ordinal'] < first_shifted] == []
    assert any(alarm['direction'] == 'up' for alarm in alarms)


def _early_rate_shift() -> list[LandingRow]:
    """1-in-10 rejected, then 1-in-2 for 60 landings from index 100, then 1-in-10 again: 360."""
    xs = _every(10, 100) + _every(2, 60) + _every(10, 200)
    return history(len(xs), start=T0, outcome=_rejections(xs))


@pytest.mark.parametrize(('metric', 'build'), [
    ('rejected', _early_rate_shift),
    ('all_role_usd', lambda: _costed_units(330, shift_at=100)),
])
def test_alarms_before_the_served_window_are_dropped_and_the_window_validates(
    metric: str, build: Callable[[], list[LandingRow]],
) -> None:
    rows = build()
    every = quality_series_view(rows, window_landings=None, provenance=FRESH)
    windowed = quality_series_view(rows, window_landings=100, provenance=FRESH)
    first_served = windowed.axis[0].ordinal
    [all_alarms] = [m.alarms for m in every.metrics if m.metric_id == metric]
    [served_alarms] = [m.alarms for m in windowed.metrics if m.metric_id == metric]

    assert any(alarm.ordinal < first_served for alarm in all_alarms)
    assert served_alarms == tuple(alarm for alarm in all_alarms if alarm.ordinal >= first_served)
    validate_quality_series(windowed, AS_OF)


def _module_cost(i: int) -> Cost:
    return {
        0: _costed(9.0, 900),
        1: _costed(4.0, 500, src=400),
        2: _costed(2.0, 400),
        3: _costed(1.0, 50, src=0),
        4: _costed(3.0, 600),
        5: _costed(3.0, 600),
        6: _costed(2.0, 200),
    }[i]


def _module_lines(i: int) -> tuple[ModuleLines, ...]:
    return {
        0: (ModuleLines('dashboard/data', 900),),
        1: (ModuleLines('dashboard/data', 300), ModuleLines('shared/quality_series', 100)),
        2: (ModuleLines('dashboard/data', 400),),
        3: (),
        4: (ModuleLines('dashboard/data', 600),),
        5: (ModuleLines('dashboard/data', 600),),
        6: (ModuleLines('shared/quality_series', 200),),
    }[i]


def test_modules_allocate_per_unit_lines_and_cost_over_the_last_four_weeks() -> None:
    rows = history(
        7, start=T0, step=timedelta(days=7), units=[1, 1, 1, 1, 2, 1],
        cost=_module_cost, modules=_module_lines,
    )

    modules = _wire(rows)['modules']

    assert modules == [
        {'module': 'dashboard/data', 'landings': 4, 'tasks': 4, 'allocated_lines': 1375.0,
         'allocated_usd': 11.0, 'usd_per_100_lines': 0.8, 'ranked': True},
        {'module': 'shared/quality_series', 'landings': 2, 'tasks': 2, 'allocated_lines': 325.0,
         'allocated_usd': 3.0, 'usd_per_100_lines': None, 'ranked': False},
        {'module': None, 'landings': 1, 'tasks': 1, 'allocated_lines': 50.0,
         'allocated_usd': 1.0, 'usd_per_100_lines': None, 'ranked': False},
    ]


def test_no_events_serve_no_before_after() -> None:
    rows = history(200, start=T0, outcome=_rejections(_every(5, 200)))

    assert _wire(rows)['before_after'] == []


def _rejected_around(n: int, boundary: int) -> tuple[list[LandingRow], list[int], RegimeEvent]:
    xs = _every(5, boundary) + _every(3, n - boundary)
    rows = history(n, start=T0, outcome=_rejections(xs))
    event = RegimeEvent('m1', at=rows[boundary].landing.landed_at, series=('rejected',))
    return rows, xs, event


LANDINGS_PER_DAY = 12.0
"""history()'s default step of two hours."""


def test_a_rate_comparison_is_not_yet_under_one_hundred_landings_after() -> None:
    rows, xs, event = _rejected_around(200, 150)
    expected = before_after_rate(xs[:150], xs[150:], landings_per_day=LANDINGS_PER_DAY)
    assert isinstance(expected, NotYet)

    [comparison] = _wire(rows, events=[event])['before_after']

    assert comparison == {
        'marker_id': 'm1', 'metric': 'rejected', 'n_before': 150, 'n_after': 50,
        'effect': None, 'lo': None, 'hi': None, 'state': 'not_yet',
        'eta_days': pytest.approx(expected.eta_days), 'min_detectable': expected.min_detectable,
    }


def test_a_rate_comparison_is_measured_from_one_hundred_landings_after() -> None:
    rows, xs, event = _rejected_around(300, 150)
    expected = before_after_rate(xs[:150], xs[150:])
    assert isinstance(expected, Effect)

    [comparison] = _wire(rows, events=[event])['before_after']

    assert comparison == {
        'marker_id': 'm1', 'metric': 'rejected', 'n_before': 150, 'n_after': 150,
        'effect': expected.d, 'lo': expected.lo, 'hi': expected.hi, 'state': 'measured',
        'eta_days': None, 'min_detectable': None,
    }


def test_a_cost_comparison_is_the_bootstrap_ratio_over_unit_points() -> None:
    rows = _costed_units(300, shift_at=150)
    boundary = next(i for i, row in enumerate(rows) if row.landing.unit_id == 'sha-00150')
    event = RegimeEvent('m2', at=rows[boundary].landing.landed_at, series=('all_role_usd',))
    lines, costs = _unit_points(rows)
    expected = before_after_cost(
        lines[:150], costs[:150], lines[150:], costs[150:],
        seed=content_seed(row.landing.task_id for row in rows),
    )
    assert isinstance(expected, CostEffect)

    [comparison] = _wire(rows, events=[event])['before_after']

    assert comparison == {
        'marker_id': 'm2', 'metric': 'all_role_usd', 'n_before': 150, 'n_after': 150,
        'effect': expected.ratio, 'lo': expected.lo, 'hi': expected.hi, 'state': 'measured',
        'eta_days': None, 'min_detectable': None,
    }


def _charter(cls: CharterClass | None, source: CharterClassSource | None) -> Cost:
    return replace(NO_COST, charter_class=cls, charter_class_source=source)


def test_charter_blocks_from_the_regex_backfill_alone_are_no_producer() -> None:
    rows = history(
        40, start=T0,
        cost=lambda i: _charter(CharterClass.UNDER_SPECIFIED, CharterClassSource.REGEX_BACKFILL)
        if i % 5 == 0 else NO_COST,
    )

    served = _metric(rows, 'charter_blocked')

    assert served['rule']['measured_since'] is None
    assert served['points'] == [None] * 40
    assert served['headline']['state'] == 'unknown'
    assert served['headline']['reason'].startswith('no producer has written here')
    assert 'task_results.charter_class' in served['headline']['reason']


def test_charter_blocks_are_measured_once_the_typed_class_is_written() -> None:
    def cost(i: int) -> Cost:
        if i == 35:
            return _charter(CharterClass.STALE_CHARTER, CharterClassSource.CHARTER_CLASS)
        if i % 5 == 0:
            return _charter(CharterClass.UNDER_SPECIFIED, CharterClassSource.REGEX_BACKFILL)
        return NO_COST

    rows = history(40, start=T0, cost=cost)

    served = _metric(rows, 'charter_blocked')

    assert served['points'] == [1 if i % 5 == 0 else 0 for i in range(40)]
    assert served['rule']['measured_since'] == T0.isoformat()
    assert served['headline']['value']['k'] == 8


def test_reverts_are_no_producer_until_late_events_has_a_writer() -> None:
    rows = history(60, start=T0, outcome=_rejections(_every(5, 60)))

    served = _metric(rows, 'reverts')

    assert served['points'] == [None] * 60
    assert served['headline']['state'] == 'unknown'
    assert served['headline']['reason'].startswith('no producer has written here')
    assert 'late_events' in served['headline']['reason']


@pytest.mark.parametrize(('metric', 'carried'), [
    ('review_rounds_ge2', 'a review round count'),
    ('execute_iterations', 'an execute iteration count'),
])
def test_a_value_no_stored_landing_carries_is_unknown_naming_the_value_not_its_source(
    metric: str, carried: str,
) -> None:
    served = _metric(history(60, start=T0), metric)

    assert served['rule']['measured_since'] is None
    assert served['headline']['state'] == 'unknown'
    assert served['headline']['reason'] == f'no stored landing of this project carries {carried}'


def test_the_git_cross_check_is_served_beside_the_series() -> None:
    rows = history(3, start=T0)

    served = _wire(rows, git_check=GitCrossCheck(2, 1, 0))['git_check']

    assert served == {
        'off_first_parent': 2, 'merges_without_landing': 1, 'late_claims': 0, 'disagreement': 3,
    }
    assert _wire(rows)['git_check'] is None


def test_counts_costs_and_comparisons_serve_byte_equal_twice() -> None:
    rows = _costed_units(300, shift_at=150)
    events = [RegimeEvent('m2', at=rows[160].landing.landed_at)]

    first = json.dumps(_wire(rows, window=100, events=events), sort_keys=True)
    second = json.dumps(_wire(rows, window=100, events=events), sort_keys=True)

    assert first == second


# --- validate_quality_series ---------------------------------------------------------------


def _series(rows: list[LandingRow]):
    return quality_series_view(rows, window_landings=100, provenance=FRESH)


def _with_metric(series, metric_id: str, **changes):
    metrics = tuple(
        replace(metric, **changes) if metric.metric_id == metric_id else metric
        for metric in series.metrics
    )
    return replace(series, metrics=metrics)


def test_a_built_series_passes_its_own_validation() -> None:
    series = _series(_costed_units(150, shift_at=120))

    validate_quality_series(series, AS_OF)


def test_a_mis_sized_points_array_is_refused_by_name() -> None:
    series = _series(history(40, start=T0, outcome=_rejections(_every(4, 40))))
    broken = _with_metric(series, 'rejected', points=(0,) * 39)

    with pytest.raises(QualitySeriesContractError) as refused:
        validate_quality_series(broken, AS_OF)

    assert refused.value.rule is QualitySeriesInvariant.ARRAY_LENGTH
    assert 'rejected.points' in str(refused.value)


def test_an_alarm_off_the_served_axis_is_refused() -> None:
    series = _series(history(40, start=T0))
    broken = _with_metric(series, 'rejected', alarms=(Alarm(999, 0, AlarmDirection.UP, None),))

    with pytest.raises(QualitySeriesContractError) as refused:
        validate_quality_series(broken, AS_OF)

    assert refused.value.rule is QualitySeriesInvariant.ALARM_ON_AXIS


def test_an_unordered_axis_is_refused() -> None:
    series = _series(history(3, start=T0))
    broken = replace(series, axis=tuple(reversed(series.axis)))

    with pytest.raises(QualitySeriesContractError) as refused:
        validate_quality_series(broken, AS_OF)

    assert refused.value.rule is QualitySeriesInvariant.AXIS_ORDER


def test_a_headline_breaking_the_envelope_is_refused() -> None:
    series = _series(history(40, start=T0, outcome=_rejections(_every(4, 40))))
    headline = next(m.headline for m in series.metrics if m.metric_id == 'rejected')
    broken = _with_metric(series, 'rejected', headline=replace(headline, as_of=None))

    with pytest.raises(DatumContractError):
        validate_quality_series(broken, AS_OF)


# --- read_quality_series: quality.db and runs.db under the project root ---------------------

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
NO_DISAGREEMENT = GitCrossCheck(0, 0, 0)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / 'dark_factory'
    project.mkdir()
    return project


def test_every_measured_headline_reads_stale_once_its_sample_is_past_the_bound() -> None:
    aged = replace(FRESH, as_of=AS_OF - timedelta(seconds=1201))

    series = quality_series_view(history(60, start=T0, outcome=_rejections(_every(5, 60))),
                                 window_landings=100, provenance=aged)

    measured = [m.headline for m in series.metrics if m.headline.value is not None]
    assert measured
    assert {(h.state, h.as_of) for h in measured} == {(DatumState.STALE, aged.as_of)}
    assert all('1200s freshness bound' in (h.reason or '') for h in measured)
    validate_quality_series(series, AS_OF)


def _quality_db(root: Path) -> Path:
    return root.parent / 'quality.db'


def _value(root: Path, at: datetime, rows: Sequence[LandingRow]) -> QualitySample:
    return QualitySample.value(
        project_id_for_root(root), at, cursor_sha=rows[-1].landing.merge_sha,
        landings_sampled=len(rows), landings_deferred=0, git_check=NO_DISAGREEMENT,
    )


def _gap(root: Path, at: datetime, reason: str) -> QualitySample:
    return QualitySample.gap(project_id_for_root(root), at, reason)


async def _read(root: Path, *, window: int | None = None, now: datetime = NOW):
    pool = DbPool()
    try:
        db = await pool.get(_quality_db(root))
        return await read_quality_series(db, root, window, now=now)
    finally:
        await pool.close_all()


def _recent(n: int) -> list[LandingRow]:
    """*n* landings ending a day before NOW, recorded in runs.db and stored alike."""
    return history(n, start=NOW - timedelta(days=1) - timedelta(hours=2 * (n - 1)))


async def _stored(root: Path, rows: Sequence[LandingRow], samples: Sequence[QualitySample]) -> None:
    await write_quality_db(_quality_db(root), rows, samples)


async def test_a_project_never_sampled_is_unknown(root: Path) -> None:
    await _stored(root, [], [])

    datum = await _read(root)

    assert (datum.state, datum.value, datum.as_of) == (DatumState.UNKNOWN, None, None)
    assert 'sampled' in (datum.reason or '')


async def test_a_project_with_only_gaps_is_unknown_with_the_newest_gap_verbatim(root: Path) -> None:
    await _stored(root, [], [
        _gap(root, NOW - timedelta(minutes=20), 'GitUnavailable: an older failure'),
        _gap(root, NOW - timedelta(minutes=10), 'GitUnavailable: no git binary to run'),
    ])

    datum = await _read(root)

    assert datum.state is DatumState.UNKNOWN
    assert datum.reason == 'GitUnavailable: no git binary to run'


async def test_an_absent_quality_db_is_unknown_naming_the_store(root: Path) -> None:
    datum = await read_quality_series(None, root, 100, now=NOW)

    assert datum.state is DatumState.UNKNOWN
    assert 'quality.db' in (datum.reason or '')


async def test_every_recorded_landing_stored_and_a_recent_value_is_fresh(root: Path) -> None:
    rows = _recent(30)
    write_landing_events(root, rows)
    sampled_at = NOW - timedelta(minutes=5)
    await _stored(root, rows, [_value(root, sampled_at, rows)])

    datum = await _read(root)

    assert datum.state is DatumState.FRESH
    assert datum.reason is None
    assert datum.as_of == sampled_at
    assert datum.freshness_bound_seconds == QUALITY_FRESHNESS_BOUND_SECONDS == 1200
    assert datum.value is not None
    assert len(datum.value.axis) == 30


def _train_after(rows: Sequence[LandingRow]) -> list[LandingRow]:
    newest = rows[-1].landing
    return train(['7001', '7002', '7003'], 'sha-train', newest.ordinal + 1,
                 newest.landed_at + timedelta(hours=1))


async def test_recorded_landings_the_store_lacks_make_it_stale_until_stored(root: Path) -> None:
    rows = _recent(30)
    pending = _train_after(rows)
    write_landing_events(root, [*rows, *pending])
    await _stored(root, rows, [_value(root, NOW - timedelta(minutes=5), rows)])

    behind = await _read(root)

    assert behind.state is DatumState.STALE
    assert behind.reason == '3 landings not yet sampled'
    assert behind.value is not None
    assert len(behind.value.axis) == 30

    caught_up = [*rows, *pending]
    await _stored(root, pending, [_value(root, NOW - timedelta(minutes=1), caught_up)])

    cleared = await _read(root)

    assert cleared.state is DatumState.FRESH
    assert cleared.value is not None
    assert len(cleared.value.axis) == 33


async def test_a_gap_after_the_newest_value_is_stale_with_its_reason(root: Path) -> None:
    rows = _recent(30)
    write_landing_events(root, rows)
    await _stored(root, rows, [
        _value(root, NOW - timedelta(minutes=15), rows),
        _gap(root, NOW - timedelta(minutes=5), 'OperationalError: database is locked'),
    ])

    datum = await _read(root)

    assert datum.state is DatumState.STALE
    assert datum.reason == 'OperationalError: database is locked'
    assert datum.as_of == NOW - timedelta(minutes=15)


async def test_a_value_older_than_the_bound_is_stale(root: Path) -> None:
    rows = _recent(30)
    write_landing_events(root, rows)
    await _stored(root, rows, [_value(root, NOW - timedelta(seconds=1201), rows)])

    datum = await _read(root)

    assert datum.state is DatumState.STALE
    assert '1200s' in (datum.reason or '')


async def test_a_missing_runs_db_cannot_confirm_freshness(root: Path) -> None:
    rows = _recent(30)
    await _stored(root, rows, [_value(root, NOW - timedelta(minutes=5), rows)])

    datum = await _read(root)

    assert datum.state is DatumState.STALE
    assert 'runs.db' in (datum.reason or '')


PENDING_TRAIN: EventRow = (NOW - timedelta(hours=3), '9001', 'train_merged', {'verdict': 'pending'})
"""Neither a coalesced nor an attributed train: the landing rule refuses it."""

NAIVE_LANDING: EventRow = ('2026-11-09T11:00:00', '9002', 'merge_finalized',
                           {'state': 'done', 'merge_sha': 'sha-naive'})
"""A done landing whose timestamp carries no offset: the landing rule refuses to guess one."""


async def test_a_train_event_the_landing_rule_refuses_cannot_confirm_freshness(root: Path) -> None:
    rows = _recent(40)
    write_landing_events(root, rows, extra_events=[PENDING_TRAIN])
    await _stored(root, rows, [_value(root, NOW - timedelta(minutes=5), rows)])

    datum = await _read(root)

    assert datum.state is DatumState.STALE
    reason = datum.reason or ''
    assert reason.startswith('freshness cannot be confirmed')
    assert 'runs.db' in reason
    assert 'UnrecognisedTrainEvent' in reason
    assert datum.value is not None
    assert len(datum.value.axis) == 40
    validate_datum(datum, NOW)


async def test_a_gap_after_the_value_still_names_itself_over_a_refused_train_event(
    root: Path,
) -> None:
    rows = _recent(40)
    write_landing_events(root, rows, extra_events=[PENDING_TRAIN])
    gap_reason = 'UnrecognisedTrainEvent: train_merged events with neither … (event ids=[41])'
    await _stored(root, rows, [
        _value(root, NOW - timedelta(minutes=15), rows),
        _gap(root, NOW - timedelta(minutes=5), gap_reason),
    ])

    datum = await _read(root)

    assert datum.state is DatumState.STALE
    assert datum.reason == gap_reason
    assert datum.value is not None
    assert len(datum.value.axis) == 40


async def test_a_landing_timestamp_without_an_offset_cannot_confirm_freshness(root: Path) -> None:
    rows = _recent(40)
    write_landing_events(root, rows, extra_events=[NAIVE_LANDING])
    await _stored(root, rows, [_value(root, NOW - timedelta(minutes=5), rows)])

    datum = await _read(root)

    assert datum.state is DatumState.STALE
    reason = datum.reason or ''
    assert reason.startswith('freshness cannot be confirmed')
    assert 'ValueError' in reason
    assert datum.value is not None
    assert len(datum.value.axis) == 40


async def test_a_window_serves_the_newest_landings_and_passes_validation(root: Path) -> None:
    rows = _recent(150)
    write_landing_events(root, rows)
    await _stored(root, rows, [_value(root, NOW - timedelta(minutes=5), rows)])

    datum = await _read(root, window=100)

    assert datum.value is not None
    assert [point.ordinal for point in datum.value.axis] == list(range(50, 150))
    validate_datum(datum, NOW)
    validate_quality_series(datum.value, NOW)
