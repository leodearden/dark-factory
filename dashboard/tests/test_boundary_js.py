"""The two-way boundary suite: real served bodies through the real client (sketch #1-#13).

``plans/dashboard-one-datum-one-path-prd.md`` sketches each datum's server row
and the client decision that renders it. Component tests prove each half
alone; this suite proves the pair agrees. ``_boundary_payloads`` drives the
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
    MERGE_ATTEMPTS,
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
from _lock_chip_matrix import node_path
from shared.task_statuses import TaskStatus

from dashboard.data.burndown import aggregate_forecast_confidence
from dashboard.data.census import SERIES_KEYS

_JS_DIR = Path(__file__).parent / 'js'

_SKETCH_ROWS = frozenset(range(1, 14))

# A top-level TAP result for a passing test named 'sketch #N: ...'; TAP
# escapes the '#' because a bare one opens a directive. A `not ok` line never
# matches, so a failing row counts as uncovered.
_PASSING_ROW_RE = re.compile(r'^ok \d+ - sketch \\#(\d+):', re.MULTILINE)


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
    covered = {int(row) for row in _PASSING_ROW_RE.findall(result.stdout)}
    missing = sorted(_SKETCH_ROWS - covered)
    assert not missing, (
        f'sketch row(s) {missing} have no passing top-level "sketch #N:" test in '
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
    assert _census(body, 'reify')['state'] == 'fresh', 'the healthy project stays measured'


def test_sketch_3_a_failed_refresh_serves_the_last_good_census_as_stale(served_bodies):
    body = served_bodies['census_stale_3h']
    census = _census(body, 'dark-factory')

    assert census['state'] == 'stale', census
    assert datetime.fromisoformat(census['as_of']) == T0
    assert _served_at(body) - datetime.fromisoformat(census['as_of']) == timedelta(hours=3)
    assert 'ReadTimeout' in census['reason'], census['reason']
    assert census['value']['total'] == 5459, 'the stale value is the last good one'


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
    assert block['recent_total'] == len(block['recent']) == len(in_window)
    latency = block['latency']
    assert sum(block['outcomes']['values']) == latency['with_duration'] + latency['without_duration']
    assert latency['with_duration'] == sum(1 for attempt in in_window if attempt.duration_ms)
    assert latency['without_duration'] == sum(1 for attempt in in_window if not attempt.duration_ms)


def test_sketch_9_the_two_windows_hold_different_merges(served_bodies):
    week, day = (served_bodies[f'merge_{window}']['MERGE_QUEUE']['dark-factory'] for window in ('7d', '24h'))

    assert week['recent_total'] != day['recent_total']
    assert len(week['recent']) != len(day['recent'])
