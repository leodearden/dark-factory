"""The census-only projection of ``/api/v2/dashboard/tasks``, end to end.

The dashboard chrome (the left-rail badges and the topbar) reads only each
root's CENSUS, on every tab. ``?projection=census`` serves exactly that from
the SAME snapshot unit the full render reads, with each measured rows half
withheld as an UNKNOWN ``Datum`` rather than omitted.

Every test drives the real collectors over the shared substrate fake
(``tests/_canned_mcp.py``), patched at ``dashboard.data.tasks.mcp_tool_call``,
so the real ``fetch_tasks`` / ``fetch_statuses`` / unit cache run underneath.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from _canned_mcp import CannedMCP, _raw_row
from shared.task_statuses import ACTIVE

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
"""The one injected instant every collector call here stamps against."""

_WIRE_KEYS = {'census', 'rows', 'in_progress_live', 'in_progress_stranded', 'skew_seconds'}

_PAIRS = ((1, 'in-progress'), (2, 'pending'), (3, 'blocked'), (4, 'done'), (5, 'cancelled'))


def _canned(pairs=_PAIRS) -> CannedMCP:
    return CannedMCP(
        rows=[_raw_row(task_id, status) for task_id, status in pairs],
        status_map=dict(pairs),
        status_page_size=2000,
    )


@pytest.fixture(autouse=True)
def _isolate_caches():
    """No unit, last good or rotation offset may cross a test."""
    import dashboard.data.active_tasks as active_tasks_mod
    import dashboard.data.task_snapshot as snapshot_mod

    snapshot_mod._snapshot_cache_clear()
    active_tasks_mod._reset_root_rotation()
    yield
    snapshot_mod._snapshot_cache_clear()


@pytest.fixture()
def two_roots(tmp_path):
    from dashboard.config import DashboardConfig

    roots = [tmp_path / 'alpha', tmp_path / 'beta']
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    return DashboardConfig(project_root=roots[0], known_project_roots=[roots[1]])


@pytest.fixture()
def no_runtime(monkeypatch):
    """An empty runtime probe, so the full collector dials no orchestrator."""
    probe = AsyncMock(return_value={})
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', probe)
    return probe


async def _acquire(canned, client, config, root, *, now):
    from dashboard.data.task_snapshot import acquire_snapshot

    with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
        return await acquire_snapshot(client, config, root, now=now)


def _assert_withheld(withheld, original):
    """*withheld* is *original* with its rows replaced by the projection's UNKNOWN."""
    from dashboard.data.datum import DatumState, validate_datum

    rows = withheld.rows
    assert rows.state is DatumState.UNKNOWN, rows
    assert rows.value is None and rows.as_of is None, rows
    assert (rows.reason or '').strip(), 'an unknown datum must say why'
    assert 'projection=census' in rows.reason, (
        f'the withheld reason must name the census projection: {rows.reason!r}'
    )
    validate_datum(rows, NOW)
    assert withheld.census == original.census
    assert withheld.in_progress_live == original.in_progress_live
    assert withheld.in_progress_stranded == original.in_progress_stranded
    assert withheld.skew_seconds == original.skew_seconds
    assert withheld.failure is original.failure

    wire = withheld.to_wire()
    assert set(wire) == _WIRE_KEYS
    assert wire['rows']['value'] is None, 'no row dict may reach the wire'


# ---------------------------------------------------------------------------
# withhold_rows — the projection's one transformation of a unit
# ---------------------------------------------------------------------------


class TestWithholdRows:
    async def test_a_fresh_measured_rows_half_is_withheld_as_unknown(
        self, dashboard_config, dummy_client,
    ):
        from dashboard.data.datum import DatumState
        from dashboard.data.task_snapshot import withhold_rows

        unit = await _acquire(
            _canned(), dummy_client, dashboard_config, dashboard_config.project_root, now=NOW,
        )
        assert unit.rows.state is DatumState.FRESH and unit.rows.value, (
            'the input must carry measured rows, or this test is vacuous'
        )

        _assert_withheld(withhold_rows(unit), unit)

    async def test_a_stale_rows_half_served_from_last_good_is_withheld_too(
        self, dashboard_config, dummy_client, monkeypatch,
    ):
        import dashboard.data.task_snapshot as snapshot_mod
        from dashboard.data.datum import DatumState
        from dashboard.data.task_snapshot import withhold_rows

        canned = _canned()
        root = dashboard_config.project_root
        await _acquire(canned, dummy_client, dashboard_config, root, now=NOW)
        monkeypatch.setattr(snapshot_mod, 'SNAPSHOT_TTL_SECONDS', 0.0)
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        unit = await _acquire(
            canned, dummy_client, dashboard_config, root, now=NOW + timedelta(seconds=20),
        )
        assert unit.rows.state is DatumState.STALE and unit.rows.value, (
            'the input must carry last-good rows, or this test is vacuous'
        )

        _assert_withheld(withhold_rows(unit), unit)

    def test_an_unmeasured_rows_half_keeps_the_producers_reason(self, tmp_path):
        from dashboard.data.datum import DatumState
        from dashboard.data.task_snapshot import (
            SnapshotFailure,
            unmeasured_snapshot,
            withhold_rows,
        )

        unit = unmeasured_snapshot(
            tmp_path / 'proj', now=NOW, reason='exceeded its budget',
            failure=SnapshotFailure.BUDGET,
        )

        withheld = withhold_rows(unit)

        assert withheld.rows == unit.rows
        assert withheld.rows.state is DatumState.UNKNOWN
        assert withheld.rows.reason == 'exceeded its budget', (
            'a rows half that was never measured is not "withheld" — its '
            "reason must stay the producer's verbatim text"
        )
        assert withheld.failure is SnapshotFailure.BUDGET


# ---------------------------------------------------------------------------
# collect_census_snapshots — the same unit, under the same budgets
# ---------------------------------------------------------------------------


class TestCollectCensusSnapshots:
    async def test_every_root_has_a_fresh_census_and_withheld_rows_for_only_the_units_reads(
        self, two_roots, dummy_client, monkeypatch,
    ):
        from dashboard.data.active_tasks import collect_census_snapshots
        from dashboard.data.datum import DatumState

        probe = AsyncMock(return_value={})
        external = AsyncMock(return_value={})
        monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', probe)
        monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', external)
        canned = _canned()

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            snapshots = await collect_census_snapshots(dummy_client, two_roots, now=NOW)

        assert set(snapshots) == {'alpha', 'beta'}
        for label, unit in snapshots.items():
            assert unit.census.state is DatumState.FRESH, (label, unit.census)
            assert unit.census.as_of == NOW
            assert unit.rows.state is DatumState.UNKNOWN, (label, unit.rows)
            assert 'projection=census' in (unit.rows.reason or ''), unit.rows.reason
            rows_wire = unit.to_wire()['rows']
            assert isinstance(rows_wire, dict), rows_wire
            assert rows_wire['value'] is None

        assert {call['tool'] for call in canned.calls} == {'get_statuses', 'get_tasks'}, (
            [call['tool'] for call in canned.calls]
        )
        assert len(canned.calls_to('get_statuses')) == 2, 'one map page per root'
        row_reads = canned.calls_to('get_tasks')
        assert len(row_reads) == 2, 'exactly the unit\'s one row read per root'
        for call in row_reads:
            assert sorted(call['args'].get('statuses') or []) == sorted(ACTIVE), call
        probe.assert_not_awaited()
        external.assert_not_awaited()

    async def test_it_is_served_the_unit_the_full_render_just_acquired(
        self, two_roots, dummy_client, no_runtime,
    ):
        """Inside the TTL the projection reads the SAME unit — no second acquisition."""
        from dashboard.data.active_tasks import (
            collect_census_snapshots,
            collect_tasks_with_counts,
        )

        canned = _canned()
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            _rows, full = await collect_tasks_with_counts(dummy_client, two_roots, now=NOW)
            reads = len(canned.calls)
            census = await collect_census_snapshots(
                dummy_client, two_roots, now=NOW + timedelta(seconds=5),
            )

        assert reads > 0
        assert len(canned.calls) == reads, (
            f'the census projection issued {len(canned.calls) - reads} MCP call(s) '
            'inside the TTL — it must be served the cached unit'
        )
        assert set(census) == set(full)
        for label in full:
            assert census[label].census.as_of == full[label].census.as_of
            assert census[label].census.value == full[label].census.value

    async def test_a_root_past_its_budget_is_degraded_exactly_as_the_full_collector_says(
        self, two_roots, dummy_client, no_runtime, monkeypatch,
    ):
        """The same per-root budget routes the same root to the same banner.

        The budget is squeezed through the module constant, the idiom
        ``test_active_tasks.py`` uses to cut off a real acquisition without
        waiting out the shipped 14 s budget.
        """
        import dashboard.data.task_snapshot as snapshot_mod
        from dashboard.data.active_tasks import (
            collect_census_snapshots,
            collect_tasks_with_counts,
        )
        from dashboard.data.task_snapshot import SnapshotFailure, SnapshotHealth, classify

        async def _hangs(client, url, tool, args, **kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr('dashboard.data.active_tasks._TASKS_PER_PROJECT_BUDGET', 0.05)
        with patch('dashboard.data.tasks.mcp_tool_call', new=_hangs):
            census = await collect_census_snapshots(dummy_client, two_roots, now=NOW)
            snapshot_mod._snapshot_cache_clear()
            _rows, full = await collect_tasks_with_counts(dummy_client, two_roots, now=NOW)

        assert set(census) == set(full) == {'alpha', 'beta'}
        for label in full:
            assert census[label].failure is SnapshotFailure.BUDGET, census[label]
            assert classify(census[label]) is SnapshotHealth.DEGRADED
            assert classify(census[label]) is classify(full[label])
            assert census[label].failure is full[label].failure
            assert census[label].rows.state is full[label].rows.state
            assert census[label].census.state is full[label].census.state


# ---------------------------------------------------------------------------
# GET /api/v2/dashboard/tasks?projection=census — the endpoint projection
# ---------------------------------------------------------------------------

_TASKS = '/api/v2/dashboard/tasks'
_CENSUS = f'{_TASKS}?projection=census'

_BANNER_KEYS = (
    'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'TASKS_DEGRADED_PROJECTS',
    'TASKS_COUNT_UNKNOWN_PROJECTS', 'TASKS_PROJECT_COUNT',
)


@pytest.fixture()
def two_root_app(client, two_roots):
    """The app's own client, fanning out over the two-root config."""
    client.app.state.config = two_roots
    return client


def _get_all(client, canned, *urls):
    """GET each of *urls* in order over *canned*, sharing one unit cache."""
    with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
        return [client.get(url) for url in urls]


class TestCensusProjectionEndpoint:
    def test_the_census_render_serves_the_full_renders_census_with_rows_withheld(
        self, two_root_app,
    ):
        full, census = _get_all(two_root_app, _canned(), _TASKS, _CENSUS)

        assert full.status_code == 200, full.text
        assert census.status_code == 200, census.text
        full_body, census_body = full.json(), census.json()
        assert set(census_body['TASKS_SNAPSHOT']) == set(full_body['TASKS_SNAPSHOT']) == {
            'alpha', 'beta',
        }
        for label, entry in census_body['TASKS_SNAPSHOT'].items():
            assert set(entry) == _WIRE_KEYS
            assert entry['census']['state'] == 'fresh', entry['census']
            assert entry['census']['value'] == full_body['TASKS_SNAPSHOT'][label]['census']['value']
            rows = entry['rows']
            assert rows['state'] == 'unknown', rows
            assert rows['value'] is None and rows['as_of'] is None
            assert 'projection=census' in (rows['reason'] or ''), rows['reason']
        for key in _BANNER_KEYS:
            assert census_body[key] == full_body[key], key
        served_at = datetime.fromisoformat(census_body['served_at'])
        for label, entry in census_body['TASKS_SNAPSHOT'].items():
            as_of = datetime.fromisoformat(entry['census']['as_of'])
            assert as_of <= served_at, (label, entry['census']['as_of'], census_body['served_at'])

    def test_the_census_render_never_reaches_the_full_collector(self, two_root_app):
        refuse = AsyncMock(side_effect=AssertionError(
            'the census projection must not run the full collector',
        ))
        with patch('dashboard.api.tasks.collect_tasks_with_counts', new=refuse):
            (resp,) = _get_all(two_root_app, _canned(), _CENSUS)

        assert resp.status_code == 200, resp.text
        refuse.assert_not_awaited()

    def test_a_root_whose_row_read_failed_is_offline_under_both_projections(
        self, two_root_app, two_roots,
    ):
        dead = str(two_roots.known_project_roots[0])
        canned = _canned()
        canned.fail_when = lambda call: (
            call['tool'] == 'get_tasks' and call['args'].get('project_root') == dead
        )

        full, census = _get_all(two_root_app, canned, _TASKS, _CENSUS)

        assert full.status_code == 200 and census.status_code == 200
        assert full.json()['TASKS_OFFLINE_PROJECTS'] == ['beta']
        assert census.json()['TASKS_OFFLINE_PROJECTS'] == ['beta']
        rows = census.json()['TASKS_SNAPSHOT']['beta']['rows']
        assert rows['state'] == 'unknown', rows
        assert 'ReadTimeout' in (rows['reason'] or ''), (
            "an unmeasured rows half must carry the producer's verbatim failure, "
            f'not the withheld text: {rows["reason"]!r}'
        )
        assert 'projection=census' not in rows['reason']
        healthy = census.json()['TASKS_SNAPSHOT']['alpha']['rows']
        assert healthy['value'] is None and 'projection=census' in (healthy['reason'] or ''), (
            'the healthy root must be the projection, or this test is vacuous'
        )

    def test_an_unknown_projection_is_refused_never_served_as_a_full_render(
        self, two_root_app,
    ):
        (resp,) = _get_all(two_root_app, _canned(), f'{_TASKS}?projection=bogus')

        assert resp.status_code == 422, resp.text

    @pytest.mark.parametrize('url', [_TASKS, f'{_TASKS}?projection=full'])
    def test_no_projection_and_the_full_projection_keep_the_rows(self, two_root_app, url):
        (resp,) = _get_all(two_root_app, _canned(), url)

        assert resp.status_code == 200, resp.text
        for label, entry in resp.json()['TASKS_SNAPSHOT'].items():
            rows = entry['rows']
            assert rows['state'] == 'fresh', (label, rows)
            assert rows['value'] and all(isinstance(row, dict) for row in rows['value'])


# ---------------------------------------------------------------------------
# The two halves meet: the URL the real client polls, served by the real endpoint
# ---------------------------------------------------------------------------

# The REAL client, in index.html's order and with no `document`, so data.js
# loads inert (data_poll.test.mjs::loadDataJs records why). `url` prints the
# /tasks url pollSetFor polls on the Scheduler tab; `apply` runs a served body
# through the real refreshOne under endpointsFor's /tasks key specs and prints
# what the rail reads back: censusOver(DF_DATA, null).
_CENSUS_CLIENT = r"""
const fs = require('fs');
const path = require('path');
const [redux, mode, url] = process.argv.slice(1);
globalThis.window = { dispatchEvent() {} };
for (const name of ['endpoint_staleness.js', 'datum.js']) require(path.join(redux, name));
const api = require(path.join(redux, 'data.js'));
for (const name of ['task_vocab.js', 'task_snapshot.js']) require(path.join(redux, name));
const TASKS = '/api/v2/dashboard/tasks';
if (mode === 'url') {
  const polled = Object.keys(api.pollSetFor('scheduler', '24h')).filter(u => api.pollKey(u) === TASKS);
  process.stdout.write(JSON.stringify(polled));
} else {
  const body = JSON.parse(fs.readFileSync(0, 'utf8'));
  api.refreshOne(url, api.endpointsFor('24h')[TASKS], api.createPollState(), {
    fetchImpl: () => Promise.resolve({ ok: true, json: async () => body }),
    now: () => Date.now(),
    setTimeoutImpl: () => 0,
    clearTimeoutImpl: () => {},
  }).then(outcome => {
    const census = window.DF_TASK_SNAPSHOT.censusOver(window.DF_DATA, null);
    process.stdout.write(JSON.stringify({ outcome, census }));
  });
}
"""


def _census_client(mode: str, url: str = '', body: dict | None = None):
    import json
    import subprocess
    from pathlib import Path

    from _lock_chip_matrix import node_path

    redux = Path(__file__).parent.parent / 'src' / 'dashboard' / 'static' / 'redux'
    result = subprocess.run(
        [node_path(), '-e', _CENSUS_CLIENT, str(redux), mode, url],
        input=json.dumps(body or {}), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f'the client driver exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    return json.loads(result.stdout)


def test_the_census_the_client_polls_on_a_rowless_tab_is_the_census_the_rail_reads(two_root_app):
    (polled,) = _census_client('url')
    assert polled == _CENSUS, f'the Scheduler tab polls {polled}, not the census projection'

    full, census = _get_all(two_root_app, _canned(), _TASKS, polled)
    assert census.status_code == 200, census.text
    applied = _census_client('apply', polled, census.json())

    assert applied['outcome'] == 'applied', applied
    datum = applied['census']
    assert datum['state'] == 'fresh', datum
    served_in_flight = sum(
        entry['census']['value']['views']['in_flight']
        for entry in full.json()['TASKS_SNAPSHOT'].values()
    )
    assert served_in_flight > 0, 'the substrate must hold in-flight tasks, or this is vacuous'
    assert datum['value']['views']['in_flight'] == served_in_flight
