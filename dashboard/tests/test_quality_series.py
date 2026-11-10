"""The quality series one project serves: axis, per-metric points, bands, baselines and alarms."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from _quality_rows import NO_COST, NO_OUTCOMES, epochs_from, history
from shared.quality_series.landings import LinesBase
from shared.quality_series.rows import LandingRow, Outcomes
from shared.quality_series.stats import (
    RegimeEvent,
    block_rate_datum,
    ewma_limits,
    wilson_interval,
)

from dashboard.data.datum import DatumState
from dashboard.data.quality_series import Provenance, quality_series_view

T0 = datetime(2026, 9, 1, tzinfo=UTC)
AS_OF = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
FRESH = Provenance(as_of=AS_OF, state=DatumState.FRESH, reason=None, freshness_bound_seconds=1200)

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


def _wire(rows: list[LandingRow], *, window: int | None = None, events=()) -> dict:
    return quality_series_view(
        rows, window_landings=window, provenance=FRESH, events=events,
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
