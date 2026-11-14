"""The two-way boundary suite: real served bodies through the real client (sketch #1-#13).

``plans/dashboard-one-datum-one-path-prd.md`` sketches each datum's server row
and the client decision that renders it; the suite also owns rows 20-21 of
``plans/dashboard-prd-lanes-prd.md``, the PRD lanes page rendered from the
served /prds and /tasks bodies (``prd lanes #N:`` tests). The ``test_quality_sketch_N_``
tests hold the server half of ``plans/quality-dashboard-prd.md`` rows 3, 4, 5, 9, 10 and 12;
``js/boundary_quality.test.mjs`` holds their client half (``quality #N:`` tests), the served
bodies read the way the Quality tab reads them. Component tests prove each half alone; this suite proves the pair agrees. ``_boundary_payloads`` drives the
real routes over fixture substrates, and ``js/boundary_*.test.mjs`` apply
those exact bodies through data.js's real refreshOne. This file is the node
half's one owner: ``test_graph_layout_js.py`` does not run it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import _boundary_payloads
import pytest
from _boundary_payloads import (
    CORPUS_TTL_GENERATIONS,
    ESCALATIONS_ARCHIVED,
    ESCALATIONS_FILED,
    HELD,
    LANES_ROOT,
    LOCKED_MODULE,
    MEMORY_OPS,
    MERGE_ATTEMPTS,
    OFFLINE_SCHEDULER,
    QUALITY_GIT_CHECK,
    QUALITY_LATE_CLAIM,
    QUALITY_MODULE_ORDINALS,
    QUALITY_OFF_LINE,
    QUALITY_PENDING_TRAIN,
    QUALITY_RECLAIMED_BACK,
    QUALITY_SMALL_N,
    QUALITY_SOURCE_EPOCH,
    QUALITY_STORED,
    QUALITY_TICK,
    QUALITY_WINDOW,
    RAGGED,
    RAGGED_CAP,
    RAGGED_DAYS,
    RAGGED_T1,
    RAGGED_T2,
    T0,
    ragged_a,
    ragged_b,
    ragged_day,
)
from _dashboard_helpers import node_path
from _lock_chip_matrix import (
    CELL_MATRIX,
    heatmap_cell_and_chip,
    heatmap_module,
    heatmap_row,
)
from shared.task_statuses import TaskStatus

from dashboard.data.burndown import aggregate_forecast_confidence
from dashboard.data.census import SERIES_KEYS
from dashboard.data.prd_view import LANE_DISPLAY_ORDER, Lane

_JS_DIR = Path(__file__).parent / 'js'

_SKETCH_ROWS = frozenset(range(1, 14))
_PRD_LANES_ROWS = frozenset({20, 21})
_QUALITY_ROWS = frozenset({3, 4, 5, 9, 10, 12})

# A top-level TAP result for a passing test named 'sketch #N: ...',
# 'prd lanes #N: ...' or 'quality #N: ...'; TAP escapes the '#' because a bare
# one opens a directive. A `not ok` line never matches, so a failing row counts
# as uncovered.
_PASSING_ROW_RE = re.compile(r'^ok \d+ - sketch \\#(\d+):', re.MULTILINE)
_PASSING_PRD_LANES_ROW_RE = re.compile(r'^ok \d+ - prd lanes \\#(\d+):', re.MULTILINE)
_PASSING_QUALITY_ROW_RE = re.compile(r'^ok \d+ - quality \\#(\d+):', re.MULTILINE)


@pytest.fixture(scope='module')
def substrates(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp('substrates')


@pytest.fixture(scope='module')
def served_bodies(substrates: Path) -> dict[str, Any]:
    return _boundary_payloads.build_all(substrates)


def test_the_boundary_suite_covers_every_sketch_row(served_bodies, tmp_path):
    for scenario, body in served_bodies.items():
        (tmp_path / f'{scenario}.json').write_text(json.dumps(body))
    files = sorted(_JS_DIR.glob('boundary_*.test.mjs'))
    assert files, f'no boundary_*.test.mjs under {_JS_DIR}, so no sketch row crosses the wire'

    result = subprocess.run(
        [node_path(), '--test', '--test-reporter=tap', *map(str, files)],
        capture_output=True,
        text=True,
        cwd=_JS_DIR,
        env={**os.environ, 'DF_BOUNDARY_PAYLOADS': str(tmp_path)},
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f'the boundary suite exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    for label, rows, pattern in (
        ('sketch', _SKETCH_ROWS, _PASSING_ROW_RE),
        ('prd lanes', _PRD_LANES_ROWS, _PASSING_PRD_LANES_ROW_RE),
        ('quality', _QUALITY_ROWS, _PASSING_QUALITY_ROW_RE),
    ):
        covered = {int(row) for row in pattern.findall(result.stdout)}
        missing = sorted(rows - covered)
        assert not missing, (
            f'{label} row(s) {missing} have no passing top-level "{label} #N:" test in '
            f'{[file.name for file in files]}\n--- stdout ---\n{result.stdout}'
        )


# ---------------------------------------------------------------------------
# The server half of each row, asserted on the very bodies the node half applies
# ---------------------------------------------------------------------------


def _census(body: dict[str, Any], project: str) -> dict[str, Any]:
    return body['TASKS_SNAPSHOT'][project]['census']


def _served_at(body: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(body['served_at'])


def test_sketch_1_the_served_census_is_the_measured_one(served_bodies):
    census = _census(served_bodies['census_fresh'], 'dark-factory')

    assert census['state'] == 'fresh', census
    assert census['value']['views'] == {'in_flight': 43, 'backlog': 1310, 'terminal': 4106}
    assert census['value']['sub_views'] == {'running': 25}
    assert census['value']['total'] == 5459


def test_sketch_2_an_unmeasured_census_is_unknown_and_says_why(served_bodies):
    body = served_bodies['census_unknown']
    census = _census(body, 'dark-factory')

    assert census['state'] == 'unknown', census
    assert census['value'] is None and census['as_of'] is None
    assert (census['reason'] or '').strip(), 'an unknown census must say why'
    assert body['TASKS_COUNT_UNKNOWN_PROJECTS'] == ['dark-factory']
    assert body['TASKS_COUNT_STALE_PROJECTS'] == [], 'no value, so not a stale one'
    assert _census(body, 'reify')['state'] == 'fresh', 'the healthy project stays measured'


def test_sketch_3_a_failed_refresh_serves_the_last_good_census_as_stale(served_bodies):
    body = served_bodies['census_stale_3h']
    census = _census(body, 'dark-factory')

    assert census['state'] == 'stale', census
    assert datetime.fromisoformat(census['as_of']) == T0
    assert _served_at(body) - datetime.fromisoformat(census['as_of']) == timedelta(hours=3)
    assert 'ReadTimeout' in census['reason'], census['reason']
    assert census['value']['total'] == 5459, 'the stale value is the last good one'
    assert body['TASKS_COUNT_STALE_PROJECTS'] == ['dark-factory']
    assert body['TASKS_COUNT_UNKNOWN_PROJECTS'] == [], (
        'a census serving its last good value is not an unknown count'
    )


def test_sketch_4_every_root_census_partitions_its_total(served_bodies):
    snapshot = served_bodies['census_fresh']['TASKS_SNAPSHOT']
    assert set(snapshot) == {'dark-factory', 'reify'}

    for project, entry in snapshot.items():
        value = entry['census']['value']
        assert sum(value['counts'].values()) == value['total'], project
        assert sum(value['views'].values()) == value['total'], project
        assert value['sub_views']['running'] <= value['views']['in_flight'], project


def test_sketch_5_each_of_the_nine_members_is_counted_once(served_bodies):
    value = _census(served_bodies['census_nine'], 'dark-factory')['value']

    assert value['counts'] == {status.value: 1 for status in TaskStatus}
    assert value['total'] == len(TaskStatus) == 9
    assert value['views'] == {'in_flight': 5, 'backlog': 2, 'terminal': 2}, (
        'in flight is in-progress, blocked, merge-deferred, review and infra-hold'
    )


def test_sketch_13_a_transient_failure_ages_the_census_then_recovers(served_bodies):
    before, during, after = (
        served_bodies[f'census_transient_{phase}'] for phase in ('before', 'during', 'after')
    )
    measured, aged, recovered = (_census(body, 'dark-factory') for body in (before, during, after))

    assert measured['state'] == 'fresh', measured
    assert aged['state'] == 'stale', aged
    assert 'ReadTimeout' in aged['reason'], aged['reason']
    assert aged['as_of'] == measured['as_of'], 'the stale census is the measured one, at its instant'
    assert recovered['state'] == 'fresh', recovered
    assert datetime.fromisoformat(recovered['as_of']) == _served_at(after)
    assert [
        (body['TASKS_COUNT_UNKNOWN_PROJECTS'], body['TASKS_COUNT_STALE_PROJECTS'])
        for body in (before, during, after)
    ] == [([], []), ([], ['dark-factory']), ([], [])]


def test_sketch_6_the_sampler_persists_the_census_the_route_serves(served_bodies):
    census = _census(served_bodies['census_at_t'], 'dark-factory')
    burndown = served_bodies['burndown_at_t']
    assert census['state'] == 'fresh', census

    for block in (burndown['BURNDOWN'], burndown['BURNDOWN_BY_PROJECT']['dark-factory']):
        assert datetime.fromisoformat(block['labels'][-1]) == datetime.fromisoformat(census['as_of'])
        for member, column in SERIES_KEYS.items():
            assert block[column][-1] == census['value']['counts'][member.value], (
                f'{column}: one datum, two homes, at one instant'
            )

    after = _census(served_bodies['census_after_t'], 'dark-factory')
    assert after['value']['counts']['in-progress'] != census['value']['counts']['in-progress'], (
        'the later census must move, or the client half cannot tell the tile from its spark'
    )


def _measured(tree, days: range) -> dict[str, list[Any]]:
    """The series one project's MEASURED samples make, built from the fixture alone."""
    return {
        'labels': [ragged_day(index).isoformat() for index in days],
        'done': [tree(index).get(TaskStatus.DONE, 0) for index in days],
        'pending': [tree(index).get(TaskStatus.PENDING, 0) for index in days],
    }


def test_sketch_7_a_gap_carries_the_last_measurement_and_is_judged_once(served_bodies, substrates):
    body = served_bodies[RAGGED]
    aggregate, b = body['BURNDOWN'], body['BURNDOWN_BY_PROJECT']['B']
    at_t2 = aggregate['labels'].index(RAGGED_T2.isoformat())
    a_at_t2, b_at_t1 = ragged_a(RAGGED_DAYS - 1), ragged_b(RAGGED_DAYS - 2)

    for member, column in SERIES_KEYS.items():
        assert aggregate[column][at_t2] == a_at_t2.get(member, 0) + b_at_t1.get(member, 0), column

    assert b['latest']['state'] == 'stale', b['latest']
    assert datetime.fromisoformat(b['latest']['as_of']) == RAGGED_T1
    assert _served_at(body) - datetime.fromisoformat(b['latest']['as_of']) == RAGGED_T2 - RAGGED_T1

    rows_at_t2 = {
        row['project']: row for row in _boundary_payloads.ragged_store_rows(substrates)
        if datetime.fromisoformat(row['ts']) == RAGGED_T2
    }
    assert rows_at_t2['A']['state'] == 'value', rows_at_t2['A']
    assert rows_at_t2['B']['state'] == 'gap', rows_at_t2['B']
    assert 'ReadTimeout' in rows_at_t2['B']['reason'], rows_at_t2['B']['reason']

    measured = [(ragged_a, range(RAGGED_DAYS)), (ragged_b, range(RAGGED_DAYS - 1))]
    breaches = sum(tree(index)[TaskStatus.IN_PROGRESS] > RAGGED_CAP for tree, days in measured for index in days)
    assert breaches == 1, 'the fixture breaches once, at B(t1), so a re-judged carry would show'
    assert aggregate['parity_breach_count'] == breaches
    assert aggregate['parity_projects'] == ['B']
    assert (aggregate['parity_peak'], aggregate['parity_cap']) == (30, RAGGED_CAP)

    forecast = aggregate_forecast_confidence([_measured(tree, days) for tree, days in measured])
    assert forecast['forecast_low'] is not None, 'eight days of history: the forecast is measured'
    assert aggregate['forecast']['value'] == forecast


def test_sketch_8_a_declined_window_is_echoed_as_the_one_served(served_bodies):
    body = served_bodies['costs_90d']

    assert body['WINDOW'] == {'requested': '90d', 'served': '30d', 'days': 30}
    assert 'COSTS' in body


@pytest.mark.parametrize('window, span', [('7d', timedelta(days=7)), ('24h', timedelta(hours=24))])
def test_sketch_9_a_window_counts_its_own_merges_and_each_attempt_once(served_bodies, window, span):
    body = served_bodies[f'merge_{window}']
    block = body['MERGE_QUEUE']['dark-factory']
    in_window = [attempt for attempt in MERGE_ATTEMPTS if attempt.age <= span]

    assert body['WINDOW']['served'] == window
    assert block['recent_total']['state'] == 'fresh', 'a readable runs.db window total is measured'
    assert block['speculative']['state'] == 'fresh', 'a readable runs.db speculative count is measured'
    assert block['recent_total']['value'] == len(block['recent']) == len(in_window)
    assert block['attempts']['state'] == 'fresh', 'a readable runs.db attempt window is measured'
    assert block['depth']['state'] == 'fresh', 'a readable runs.db depth series is measured'
    attempts = block['attempts']['value']
    latency = attempts['latency']
    assert sum(attempts['outcomes']['values']) == latency['with_duration'] + latency['without_duration']
    assert latency['with_duration'] == sum(1 for attempt in in_window if attempt.duration_ms)
    assert latency['without_duration'] == sum(1 for attempt in in_window if not attempt.duration_ms)


def test_sketch_9_the_two_windows_hold_different_merges(served_bodies):
    week, day = (served_bodies[f'merge_{window}']['MERGE_QUEUE']['dark-factory'] for window in ('7d', '24h'))

    assert week['recent_total']['value'] != day['recent_total']['value']
    assert len(week['recent']) != len(day['recent'])


@pytest.mark.parametrize('window, bucket_minutes', [('24h', 15), ('7d', 60)])
def test_the_served_depth_names_the_width_its_labels_are_spaced_at(served_bodies, window, bucket_minutes):
    depth = served_bodies[f'merge_{window}']['MERGE_QUEUE']['dark-factory']['depth']['value']
    starts = [datetime.fromisoformat(label) for label in depth['labels']]

    assert depth['bucket_minutes'] == bucket_minutes
    assert len(starts) >= 2, (
        "the fixture's merges at 20h and 70h before T0 leave at least two buckets after the leading-zero trim"
    )
    gaps = {later - earlier for earlier, later in zip(starts, starts[1:], strict=False)}
    assert gaps == {timedelta(minutes=bucket_minutes)}


@pytest.mark.parametrize('generation', sorted(CORPUS_TTL_GENERATIONS))
def test_sketch_10_the_pill_and_the_strip_count_one_walk_whatever_its_ttl(served_bodies, generation):
    escalations = served_bodies[f'escalations_{generation}']['ESCALATIONS']
    (project,) = served_bodies[f'analytics_{generation}']['ESCALATION_ANALYTICS']['per_project']
    queue_pending = escalations['views']['queue_pending']
    open_in_history = project['views']['open_in_history']

    assert queue_pending['value'] == ESCALATIONS_FILED - ESCALATIONS_ARCHIVED == 2
    assert open_in_history['value'] == ESCALATIONS_FILED == 5
    assert queue_pending['state'] == open_in_history['state'] == 'fresh'
    assert queue_pending['as_of'] == open_in_history['as_of'], 'two walks, where there must be one'


def test_sketch_10_each_ttl_generation_is_its_own_walk(served_bodies):
    as_of = {
        generation: served_bodies[f'escalations_{generation}']['ESCALATIONS']['views']['queue_pending']['as_of']
        for generation in CORPUS_TTL_GENERATIONS
    }
    assert len(set(as_of.values())) == len(as_of), f'a generation reused another\'s walk: {as_of}'


def test_sketch_11_every_journal_row_is_counted_once_in_the_totals(served_bodies):
    served = served_bodies['memory_ops']['MEMORY_OPS']['totals']
    assert served['state'] == 'fresh', served
    totals = served['value']
    kinds = [kind for _operation, kind in MEMORY_OPS]

    assert totals['reads'] + totals['writes'] + totals['other'] == totals['total'] == len(MEMORY_OPS)
    assert (totals['reads'], totals['writes']) == (kinds.count('read'), kinds.count('write'))
    assert totals['other'] == 1, 'the kind outside read and write is counted, as other'


def test_sketch_11_a_missing_journal_is_unknown_not_a_quiet_day(served_bodies):
    ops = served_bodies['memory_ops_missing']['MEMORY_OPS']

    for reading in ('totals', 'newest_hour_total'):
        assert ops[reading]['state'] == 'unknown', ops[reading]
        assert (ops[reading]['reason'] or '').strip(), f'an unknown {reading} must say why'
    for series in ('labels', 'reads', 'writes', 'other', 'total', 'by_operation'):
        assert ops[series] == [], f'a hole serves no {series} to draw as a flat zero line'


def test_sketch_12_an_offline_scheduler_is_named_while_its_tasks_are_still_served(served_bodies):
    scheduler = served_bodies['scheduler_offline']['SCHEDULER']
    rows = served_bodies['tasks_with_p']['TASKS_SNAPSHOT'][OFFLINE_SCHEDULER]['rows']

    assert scheduler['offline_projects'] == [OFFLINE_SCHEDULER]
    assert rows['state'] == 'fresh', rows
    assert sum(row['status'] == 'in-progress' for row in rows['value']) >= 2


def test_sketch_12_a_served_parked_and_held_lock_is_one_class_in_the_cell_and_the_chip(served_bodies):
    scheduler = served_bodies['scheduler_offline']['SCHEDULER']
    (module,) = (m for m in scheduler['modules'] if (m['project'], m['path']) == (HELD, LOCKED_MODULE))
    (row,) = (r for r in scheduler['rows'] if (r['project'], r['task_id']) == (HELD, '2'))
    assert (module['holder'], module['parked_by'], module['parked_owner_live']) == ('1', '2', True), module

    fixture_module = heatmap_module(**CELL_MATRIX['parked and held'])
    fixture_row = heatmap_row(fixture_module)
    assert set(fixture_module) <= set(module), 'the lock matrix must be built in the served module shape'
    assert set(fixture_row) <= set(row), 'the lock matrix must be built in the served row shape'

    cell, chip = heatmap_cell_and_chip(row, module)
    _fixture_cell, fixture_chip = heatmap_cell_and_chip(fixture_row, fixture_module)
    assert cell == chip, f'heatmap cell {cell!r} disagrees with the lock chip {chip!r}'
    assert chip['cls'] == fixture_chip['cls'] == 'lock-taken'


def test_prd_lanes_20_the_served_view_fills_every_lane_in_display_order(served_bodies):
    view = served_bodies['prd_lanes']['PRD_VIEW'][LANES_ROOT]
    assert view['state'] == 'fresh', view['reason']
    lanes = view['value']['lanes']

    assert [entry['lane'] for entry in lanes] == [lane.value for lane in LANE_DISPLAY_ORDER]
    assert all(entry['prd_keys'] for entry in lanes), 'an empty lane renders nothing to check'
    (tail,) = (entry for entry in lanes if entry['lane'] == Lane.TAIL.value)
    assert len(tail['prd_keys']) >= 2, 'one tail title cannot show the collapsed line lists them all'

    contention = served_bodies['prd_lanes_contended']['PRD_CONTENTION'][LANES_ROOT]
    assert contention['state'] == 'fresh', contention['reason']
    assert contention['value']['top'], 'an empty top renders every served entry vacuously'

    for scenario, key in (('prd_view_map_failed', 'PRD_VIEW'), ('prd_lanes_scheduler_offline', 'PRD_CONTENTION')):
        degraded = served_bodies[scenario][key][LANES_ROOT]
        assert degraded['state'] == 'unknown', degraded
        assert (degraded['reason'] or '').strip(), f'{scenario}: an unknown {key} must say why'


def test_prd_lanes_20_a_queueless_view_is_a_lower_bound_that_still_draws_rows(served_bodies):
    view = served_bodies['prd_lanes_queueless']['PRD_VIEW'][LANES_ROOT]
    assert view['state'] == 'lower_bound', view
    assert (view['reason'] or '').strip(), 'a lower-bound view must say why'
    lanes = {entry['lane']: entry['prd_keys'] for entry in view['value']['lanes']}

    assert all(lanes.get(lane.value) for lane in LANE_DISPLAY_ORDER), 'an empty lane renders nothing to check'
    rows = view['value']['prds']
    assert any(row['signal']['reason'] for row in rows), 'no row serves a signal reason to keep on hover'
    assert any(row['followup_open'] for row in rows), 'no row serves an open follow-up to split on hover'


def _quality(body: dict[str, Any]) -> dict[str, Any]:
    """The QUALITY Datum's value, once its envelope has been checked."""
    datum = body['QUALITY']
    assert set(datum) == {'value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'}, datum
    assert datum['value'] is not None, datum['reason']
    return datum['value']


def _ordinals(series: dict[str, Any]) -> list[int]:
    return [point['ordinal'] for point in series['axis']]


def test_quality_sketch_3_git_disagreement_is_served_not_dropped(served_bodies):
    body = served_bodies['quality_fresh']
    series = _quality(body)
    by_ordinal = {point['ordinal']: point for point in series['axis']}

    assert body['QUALITY']['state'] == 'fresh', body['QUALITY']['reason']
    assert body['WINDOW'] == {'requested': '100', 'served': '100', 'landings': QUALITY_WINDOW}
    assert _ordinals(series) == list(range(QUALITY_STORED - QUALITY_WINDOW, QUALITY_STORED)), (
        'the axis is the stored landings, newest last: none dropped, none invented'
    )
    off_line, late = by_ordinal[QUALITY_OFF_LINE], by_ordinal[QUALITY_LATE_CLAIM]
    assert (off_line['git_first_parent'], off_line['late_claim']) == (False, False)
    assert (late['git_first_parent'], late['late_claim']) == (True, True)
    assert late['merge_sha'] == by_ordinal[QUALITY_LATE_CLAIM - QUALITY_RECLAIMED_BACK]['merge_sha']
    assert series['git_check'] == {
        'off_first_parent': QUALITY_GIT_CHECK.off_first_parent,
        'merges_without_landing': QUALITY_GIT_CHECK.merges_without_landing,
        'late_claims': QUALITY_GIT_CHECK.late_claims,
        'disagreement': QUALITY_GIT_CHECK.disagreement,
    }
    for metric_id, metric in series['metrics'].items():
        assert len(metric['points']) == QUALITY_WINDOW, metric_id


def test_quality_sketch_4_a_source_that_has_not_started_is_unknown_not_zero(served_bodies):
    body = served_bodies['quality_before_epoch']
    metrics = _quality(body)['metrics']
    blocking = metrics['reviewer_blocking']

    assert blocking['points'] == [None] * QUALITY_WINDOW, 'every landing was reviewed: only the epoch nulls them'
    assert all(center is None for center in blocking['rolling']['center'])
    assert blocking['rule']['measured_since'] == QUALITY_SOURCE_EPOCH.isoformat()
    headline = blocking['headline']
    assert (headline['state'], headline['value']) == ('unknown', None), headline
    assert QUALITY_SOURCE_EPOCH.date().isoformat() in headline['reason']
    assert metrics['rejected']['headline']['state'] == 'fresh', (
        'a source that did start is measured in the same body'
    )


def test_quality_sketch_5_small_n_is_k_of_n_with_no_chart_statistics(served_bodies):
    body = served_bodies['quality_small_n']
    series = _quality(body)
    headline = series['metrics']['rejected']['headline']

    assert len(series['axis']) == QUALITY_SMALL_N
    assert headline['state'] == 'fresh', headline
    value = headline['value']
    assert (value['form'], value['n']) == ('k_of_n', QUALITY_SMALL_N)
    assert value['k'] == sum(1 for ordinal in range(QUALITY_SMALL_N) if ordinal % 5 == 0)
    assert value['lo'] < value['value'] < value['hi'], 'k of n carries its Wilson band'
    for metric_id, metric in series['metrics'].items():
        assert metric['ewma'] is None, metric_id
        assert metric['alarms'] == [], metric_id
        assert metric['baseline']['p0'] is None, metric_id
    assert series['before_after'] == []


def test_quality_sketch_10_the_same_window_served_twice_is_byte_equal(served_bodies):
    first, again = served_bodies['quality_fresh'], served_bodies['quality_fresh_again']
    metrics = _quality(first)['metrics']

    assert first['served_at'] != again['served_at'], 'two reads, not one body twice'
    assert json.dumps(first['QUALITY'], sort_keys=True) == json.dumps(again['QUALITY'], sort_keys=True)
    seeded = {metric_id: metrics[metric_id]['headline']['value']
              for metric_id in ('rejections_per_landing', 'all_role_usd')}
    assert [band['form'] for band in seeded.values()] == ['mean', 'median'], seeded
    assert all(band['lo'] < band['hi'] for band in seeded.values()), (
        'a degenerate band cannot show the bootstrap is seeded by content'
    )


def test_quality_sketch_12_main_ahead_of_the_cursor_is_stale_until_the_next_tick(served_bodies):
    behind, cleared = served_bodies['quality_main_ahead'], served_bodies['quality_main_ahead_cleared']

    assert behind['QUALITY']['state'] == 'stale', behind['QUALITY']
    assert behind['QUALITY']['reason'] == f'{len(QUALITY_PENDING_TRAIN)} landings not yet sampled'
    assert _ordinals(_quality(behind))[-1] == QUALITY_STORED - 1, 'the stored series is still served'

    assert cleared['QUALITY']['state'] == 'fresh', cleared['QUALITY']
    assert cleared['QUALITY']['reason'] is None
    assert datetime.fromisoformat(cleared['QUALITY']['as_of']) == (
        datetime.fromisoformat(behind['QUALITY']['as_of']) + QUALITY_TICK
    )
    newest = _quality(cleared)['axis'][-len(QUALITY_PENDING_TRAIN):]
    assert [point['task_id'] for point in newest] == list(QUALITY_PENDING_TRAIN)
    assert len({point['unit_id'] for point in newest}) == 1, 'a train is one verify unit'


def _alarm_and_marker(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The rejection rate's first alarm, and the one derived routing marker the store holds."""
    [alarm, *_] = _quality(body)['metrics']['rejected']['alarms']
    [marker] = [marker for marker in body['QUALITY_MARKERS']['value']['markers']
                if marker['source']['type'] == 'routing']
    return alarm, marker


def test_quality_sketch_9_an_alarm_is_explained_only_by_a_marker_near_its_onset(served_bodies):
    near, near_marker = _alarm_and_marker(served_bodies['quality_alarm_explained'])
    far, far_marker = _alarm_and_marker(served_bodies['quality_alarm_unexplained'])

    assert near['onset_ordinal'] == far['onset_ordinal'], 'one shift, two marker placements'
    assert (near['direction'], far['direction']) == ('up', 'up')
    assert near['onset_ordinal'] - near_marker['ordinal'] == 10
    assert far['onset_ordinal'] - far_marker['ordinal'] == 100
    assert near['explained_by'] == near_marker['id']
    assert far['explained_by'] is None


def test_quality_module_drill_down_serves_the_landings_touching_the_module(served_bodies):
    series = _quality(served_bodies['quality_module'])

    assert _ordinals(series) == list(QUALITY_MODULE_ORDINALS)
    for metric_id, metric in series['metrics'].items():
        assert len(metric['points']) == len(QUALITY_MODULE_ORDINALS), metric_id
