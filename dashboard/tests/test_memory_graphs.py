"""The /memory-graphs contract: one window's memory operations, served as Datums.

``redux_api.shape_memory_graphs`` is the server half and
``js/memory_readings.test.mjs`` the client half; the boundary suite
(``test_boundary_js.py``) proves the two agree. A journal that cannot be read
is an ``unknown`` reading carrying its reason, never a zero window that would
read as a quiet day.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from dashboard.data import redux_api
from dashboard.data.datum import Datum, DatumState, aged_at, unknown_datum
from dashboard.data.write_journal import MemoryOps

SERVED_AT = datetime(2026, 10, 3, 12, 0, 30, tzinfo=UTC)
BOUND = 60
WIRE_DATUM_KEYS = {'value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'}
MEMORY_OPS_KEYS = {
    'labels', 'reads', 'writes', 'other', 'total', 'by_operation',
    'totals', 'newest_hour_total',
}
SERIES_KEYS = ('labels', 'reads', 'writes', 'other', 'total', 'by_operation')
JOURNAL_REASON = 'the write journal query failed: OperationalError: database is locked'
MEMORY_GRAPHS = '/api/v2/dashboard/memory-graphs'


def _memory_ops() -> MemoryOps:
    return MemoryOps(
        labels=('11:00', '12:00'),
        reads=(3, 7),
        writes=(1, 2),
        other=(0, 2),
        by_operation=(('search', 10), ('add_memory', 3), ('compact', 2)),
    )


def _fresh(*, age_seconds: int = 2) -> Datum[MemoryOps]:
    return Datum(
        _memory_ops(), SERVED_AT - timedelta(seconds=age_seconds),
        DatumState.FRESH, None, BOUND,
    )


def _shaped(ops: Datum[MemoryOps]) -> dict:
    return redux_api.shape_memory_graphs(ops, served_at=SERVED_AT)


def _assert_reconciles(block: dict) -> None:
    """PRD sketch #11: the caption's three numbers sum to the donut's total."""
    totals = block['totals']['value']
    assert totals['total'] == totals['reads'] + totals['writes'] + totals['other']
    assert totals['reads'] == sum(block['reads'])
    assert totals['writes'] == sum(block['writes'])
    assert totals['other'] == sum(block['other'])
    assert totals['total'] == sum(row['value'] for row in block['by_operation'])
    assert totals['total'] == sum(block['total'])


def _assert_unknown_block(block: dict) -> None:
    """A hole: no series to draw a flat zero line, and both readings unknown."""
    for key in SERIES_KEYS:
        assert block[key] == [], (key, block[key])
    for reading in ('totals', 'newest_hour_total'):
        wired = block[reading]
        assert set(wired) == WIRE_DATUM_KEYS, (reading, wired)
        assert wired['state'] == 'unknown', (reading, wired)
        assert wired['value'] is None, (reading, wired)
        assert wired['as_of'] is None, (reading, wired)


# ---------------------------------------------------------------------------
# shape_memory_graphs
# ---------------------------------------------------------------------------


def test_the_body_is_one_memory_ops_block_and_its_served_at():
    body = _shaped(_fresh())

    assert set(body) == {'MEMORY_OPS', 'served_at'}, (
        'one query, one datum, one key: MEMORY_TIMESERIES and '
        'MEMORY_OPS_BREAKDOWN are retired'
    )
    assert body['served_at'] == SERVED_AT.isoformat()
    assert set(body['MEMORY_OPS']) == MEMORY_OPS_KEYS


def test_a_fresh_reading_serves_the_series():
    ops = _shaped(_fresh())['MEMORY_OPS']

    assert ops['labels'] == ['11:00', '12:00']
    assert ops['reads'] == [3, 7]
    assert ops['writes'] == [1, 2]
    assert ops['other'] == [0, 2]


def test_the_hourly_total_sums_the_three_series():
    ops = _shaped(_fresh())['MEMORY_OPS']

    assert ops['total'] == [
        r + w + o for r, w, o in zip(ops['reads'], ops['writes'], ops['other'], strict=True)
    ]
    assert ops['total'] == [4, 11]


def test_the_by_operation_view_keeps_memory_ops_order():
    ops = _shaped(_fresh())['MEMORY_OPS']

    assert ops['by_operation'] == [
        {'label': 'search', 'value': 10},
        {'label': 'add_memory', 'value': 3},
        {'label': 'compact', 'value': 2},
    ]


def test_fresh_totals_are_a_fresh_wire_datum_reconciling_with_by_operation():
    measured = _fresh()
    ops = _shaped(measured)['MEMORY_OPS']
    totals = ops['totals']

    assert set(totals) == WIRE_DATUM_KEYS
    assert totals['state'] == 'fresh'
    assert totals['reason'] is None
    assert totals['as_of'] == measured.as_of.isoformat()
    assert totals['freshness_bound_seconds'] == BOUND
    assert totals['value'] == {'reads': 10, 'writes': 3, 'other': 2, 'total': 15}
    _assert_reconciles(ops)


def test_the_newest_hour_total_is_the_last_hourly_total_measured_with_the_totals():
    ops = _shaped(_fresh())['MEMORY_OPS']
    newest = ops['newest_hour_total']

    assert set(newest) == WIRE_DATUM_KEYS
    assert newest['state'] == 'fresh'
    assert newest['value'] == ops['total'][-1] == 11
    assert newest['as_of'] == ops['totals']['as_of']


def test_an_unknown_reading_serves_no_series_and_keeps_the_producers_reason():
    ops = _shaped(unknown_datum(JOURNAL_REASON, BOUND))['MEMORY_OPS']

    _assert_unknown_block(ops)
    assert ops['totals']['reason'] == JOURNAL_REASON
    assert ops['newest_hour_total']['reason'] == JOURNAL_REASON


def test_a_reading_measured_past_its_bound_is_served_stale_with_its_series():
    measured = _fresh(age_seconds=BOUND + 30)
    ops = _shaped(measured)['MEMORY_OPS']
    aged_reason = aged_at(measured, SERVED_AT).reason

    for reading in ('totals', 'newest_hour_total'):
        assert ops[reading]['state'] == 'stale', (reading, ops[reading])
        assert ops[reading]['reason'] == aged_reason, (reading, ops[reading])
        assert ops[reading]['as_of'] == measured.as_of.isoformat()
    assert ops['reads'] == [3, 7]
    _assert_reconciles(ops)


# ---------------------------------------------------------------------------
# GET /api/v2/dashboard/memory-graphs
# ---------------------------------------------------------------------------


def test_a_missing_journal_is_unknown_not_a_quiet_day(client):
    assert not client.app.state.config.write_journal_db.exists(), (
        'this pin needs a dashboard with no write journal on disk'
    )

    resp = client.get(MEMORY_GRAPHS)

    assert resp.status_code == 200
    ops = resp.json()['MEMORY_OPS']
    _assert_unknown_block(ops)
    assert 'write journal' in ops['totals']['reason']


def test_a_failed_read_is_unknown_naming_the_error_not_a_500(client):
    with patch(
        'dashboard.app.get_memory_ops', new=AsyncMock(side_effect=RuntimeError('boom')),
    ):
        resp = client.get(MEMORY_GRAPHS)

    assert resp.status_code == 200
    ops = resp.json()['MEMORY_OPS']
    _assert_unknown_block(ops)
    assert 'RuntimeError' in ops['totals']['reason']
    assert 'boom' in ops['totals']['reason']


def test_a_fresh_read_serves_one_reconciling_memory_ops_block(client):
    measured = Datum(_memory_ops(), datetime.now(UTC), DatumState.FRESH, None, BOUND)
    with patch('dashboard.app.get_memory_ops', new=AsyncMock(return_value=measured)):
        resp = client.get(MEMORY_GRAPHS)

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {'MEMORY_OPS', 'served_at'}
    ops = body['MEMORY_OPS']
    assert set(ops) == MEMORY_OPS_KEYS
    assert ops['totals']['state'] == 'fresh'
    assert ops['newest_hour_total']['value'] == ops['total'][-1]
    _assert_reconciles(ops)
