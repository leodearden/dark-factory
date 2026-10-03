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
    """No unit, last good, row cache or rotation offset may cross a test."""
    import dashboard.data.active_tasks as active_tasks_mod
    import dashboard.data.task_snapshot as snapshot_mod
    import dashboard.data.tasks as tasks_mod

    snapshot_mod._snapshot_cache_clear()
    tasks_mod._fetch_tasks_cache_clear()
    active_tasks_mod._reset_root_rotation()
    yield
    snapshot_mod._snapshot_cache_clear()
    tasks_mod._fetch_tasks_cache_clear()


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
            assert unit.to_wire()['rows']['value'] is None

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
