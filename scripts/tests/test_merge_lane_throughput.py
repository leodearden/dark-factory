"""Tests for scripts/merge_lane_throughput.py — the merge-lane baseline reporter.

Every number asserted here is a KNOWN ANSWER computed by hand from a
programmatically-built fixture ``runs.db``. Deliberately NO test asserts a
number read from a live store: ``data/orchestrator/runs.db`` is mutated
continuously by the running orchestrator, so a live-number assertion would be
non-hermetic and would self-invalidate within hours (plan decision 3).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import merge_lane_throughput as mlt
import pytest

# ---------------------------------------------------------------------------
# parse_window — clock-free: ``now`` is always injected, never read inside.
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 5, 12, 30, tzinfo=UTC)


def test_parse_window_relative_14d_is_now_minus_14_days():
    lo, hi = mlt.parse_window('14d', NOW)
    assert hi == NOW
    assert lo == NOW - timedelta(days=14)
    assert lo.tzinfo is not None and hi.tzinfo is not None
    assert lo.utcoffset() == timedelta(0)
    assert hi.utcoffset() == timedelta(0)


def test_parse_window_relative_30d_is_now_minus_30_days():
    lo, hi = mlt.parse_window('30d', NOW)
    assert (lo, hi) == (NOW - timedelta(days=30), NOW)


def test_parse_window_dated_range_is_exactly_those_two_instants():
    # The dated form is how a caller reproduces the PRD's dated baseline;
    # `14d` resolves relative to NOW and therefore cannot (plan decision 3).
    lo, hi = mlt.parse_window(
        '2026-08-20T16:10:00+00:00..2026-09-03T16:10:00+00:00', NOW
    )
    assert lo == datetime(2026, 8, 20, 16, 10, tzinfo=UTC)
    assert hi == datetime(2026, 9, 3, 16, 10, tzinfo=UTC)


def test_parse_window_naive_iso_endpoint_is_interpreted_as_utc():
    lo, hi = mlt.parse_window('2026-08-20T16:10:00..2026-09-03T16:10:00', NOW)
    assert lo == datetime(2026, 8, 20, 16, 10, tzinfo=UTC)
    assert hi == datetime(2026, 9, 3, 16, 10, tzinfo=UTC)
    assert lo.utcoffset() == timedelta(0)


def test_parse_window_mixed_naive_and_aware_endpoints_both_utc():
    lo, hi = mlt.parse_window('2026-08-20T16:10:00..2026-09-03T16:10:00+00:00', NOW)
    assert lo == datetime(2026, 8, 20, 16, 10, tzinfo=UTC)
    assert hi == datetime(2026, 9, 3, 16, 10, tzinfo=UTC)


def test_parse_window_date_only_endpoints_are_midnight_utc():
    lo, hi = mlt.parse_window('2026-08-20..2026-09-03', NOW)
    assert lo == datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    assert hi == datetime(2026, 9, 3, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize('spec', ['', 'd', '14', '14x', 'a..b', '0d', '-3d'])
def test_parse_window_rejects_malformed_specs_loudly(spec):
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        mlt.parse_window(spec, NOW)
    # The offending text must be echoed back — a bare "bad window" would make
    # the operator guess which of several --window flags was wrong.
    assert repr(spec) in str(exc.value)


def test_parse_window_rejects_reversed_range():
    spec = '2026-09-03T16:10:00+00:00..2026-08-20T16:10:00+00:00'
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        mlt.parse_window(spec, NOW)
    assert repr(spec) in str(exc.value)


def test_parse_window_rejects_empty_range_lo_equals_hi():
    spec = '2026-09-03T16:10:00+00:00..2026-09-03T16:10:00+00:00'
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        mlt.parse_window(spec, NOW)
    assert repr(spec) in str(exc.value)


def test_parse_window_rejects_three_part_range():
    spec = '2026-08-20..2026-08-25..2026-09-03'
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        mlt.parse_window(spec, NOW)
    assert repr(spec) in str(exc.value)


def test_iso_formats_the_spelling_the_events_table_stores():
    # Window bounds are used directly as SQL string comparands against the
    # `timestamp` TEXT column, so the spelling must match byte-for-byte.
    assert mlt._iso(datetime(2026, 8, 20, 16, 10, tzinfo=UTC)) == (
        '2026-08-20T16:10:00+00:00'
    )


# ---------------------------------------------------------------------------
# Fixture builder — DDL copied verbatim from event_store.py::_SCHEMA.
#
# The house convention (scripts/tests/test_audit_wiped_metadata_files.py,
# test_analyze_speculation_depth.py) is to inline the DDL and build the DB in
# tmp_path rather than check a binary .db into git: a "known answer" has to be
# legible to a reviewer, and a blob makes the input to every asserted number
# unreviewable in a diff.
# ---------------------------------------------------------------------------

_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_events_run  ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(timestamp);
"""


def _make_runs_db(
    tmp_path: Path, events: list[dict[str, Any]], name: str = 'runs.db'
) -> Path:
    """Build a temp runs.db mirroring the live events schema and insert *events*.

    Each event dict may carry: ``event_type`` (required), ``task_id``,
    ``run_id``, ``timestamp``, ``phase``, ``role``, ``data``, ``cost_usd`` and
    ``duration_ms``. ``data`` is passed through VERBATIM when it is a str or
    None — so a test can inject malformed JSON or a NULL payload — and
    json-encoded otherwise. Rows are inserted in list order, so *events* order
    IS ascending ``id`` order.
    """
    db_path = tmp_path / name
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_EVENTS_SCHEMA)
        for i, event in enumerate(events):
            data = event.get('data')
            if data is not None and not isinstance(data, str):
                data = json.dumps(data)
            task_id = event.get('task_id')
            conn.execute(
                'INSERT INTO events (timestamp, run_id, task_id, event_type, '
                'phase, role, data, cost_usd, duration_ms) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    event.get('timestamp', f'2026-08-11T00:00:{i:02d}+00:00'),
                    event.get('run_id', 'run-1'),
                    None if task_id is None else str(task_id),
                    event['event_type'],
                    event.get('phase'),
                    event.get('role'),
                    data,
                    event.get('cost_usd'),
                    event.get('duration_ms'),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _make_project_root(
    tmp_path: Path, name: str, events: list[dict[str, Any]]
) -> Path:
    """Materialise a synthetic project root holding data/orchestrator/runs.db."""
    root = tmp_path / name
    (root / 'data' / 'orchestrator').mkdir(parents=True)
    _make_runs_db(root / 'data' / 'orchestrator', events)
    return root


# ---------------------------------------------------------------------------
# The shared KNOWN-ANSWER corpus, for two synthetic project roots.
#
# Window: CORPUS_LO .. CORPUS_HI = 2026-08-10T12:00Z .. 2026-08-15T00:00Z.
# `lo` is deliberately MID-DAY so Aug 10 is a partial leading bucket, and `hi`
# is midnight so there is no partial trailing bucket. Complete day buckets are
# therefore Aug 11, 12, 13, 14 (n_days=4).
#
# ROOT A known answers
#   landings/day (state == 'done', complete buckets only):
#       Aug 11: 3, Aug 12: 1, Aug 13: 0 (zero-filled), Aug 14: 4
#       -> median 2.0, max 4
#     Aug 10 also holds 5 `done` rows; keeping that partial bucket would give
#     median 3, max 5. Aug 12 also holds 2 'blocked' + 1 'superseded'; counting
#     all states would give median 3.5. Both wrong answers are pinned below.
#   speculation: 10 speculative_merge, 3 verdict_voided(chain_dead) -> 0.30
#
# ROOT B known answers
#   landings/day: Aug 11: 1, Aug 12: 2, Aug 13: 2, Aug 14: 1
#       -> median 1.5, max 2
#   speculation: 12 speculative_merge, 7 verdict_voided(chain_dead) -> 7/12
#
# The two void rates are deliberately far apart (0.30 vs ~0.58): that spread —
# reported PER PROJECT, never merged into one tally — is the shape the PRD's
# downstream void-rate decomposition needs.
# ---------------------------------------------------------------------------

CORPUS_LO = '2026-08-10T12:00:00+00:00'
CORPUS_HI = '2026-08-15T00:00:00+00:00'
CORPUS_WINDOW = f'{CORPUS_LO}..{CORPUS_HI}'
CORPUS_LO_DT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
CORPUS_HI_DT = datetime(2026, 8, 15, 0, 0, tzinfo=UTC)


def _ts(day: int, hour: int = 0, minute: int = 0) -> str:
    """An events-table-shaped ISO-8601 timestamp in August 2026, UTC."""
    return f'2026-08-{day:02d}T{hour:02d}:{minute:02d}:00+00:00'


def _fin(task_id, day, hour, state='done', **data) -> dict[str, Any]:
    return {
        'event_type': 'merge_finalized',
        'task_id': task_id,
        'timestamp': _ts(day, hour),
        'data': {'branch': f'task/{task_id}', 'state': state, **data},
    }


def corpus_a() -> list[dict[str, Any]]:
    """Event corpus for synthetic project root A (see known answers above)."""
    events: list[dict[str, Any]] = []

    # Out-of-window landing: must never be counted.
    events.append(_fin('a00', 9, 3))

    # Partial leading bucket (Aug 10, cut mid-day by CORPUS_LO): 5 landings.
    events += [_fin(f'a0{i}', 10, 13 + i) for i in range(1, 6)]
    # Complete buckets.
    events += [_fin('a06', 11, 1), _fin('a07', 11, 5), _fin('a08', 11, 9)]
    events.append(_fin('a09', 12, 3))
    events += [
        _fin('x01', 12, 4, state='blocked'),
        _fin('x02', 12, 5, state='blocked'),
        _fin('x03', 12, 6, state='superseded'),
    ]
    # Aug 13 deliberately empty -> must zero-fill.
    events += [
        _fin('a10', 14, 2, landed_via_chain=1),
        _fin('a11', 14, 6, landed_via_chain=1),
        _fin('a12', 14, 10),
        _fin('a13', 14, 14),
    ]

    # Lead-time: a06 re-queues (gate_retry churn) before landing at 01:00.
    events += [
        {'event_type': 'merge_queued', 'task_id': 'a06',
         'timestamp': _ts(11, 0, 0), 'data': {'branch': 'task/a06', 'queue_depth': 2}},
        {'event_type': 'merge_queued', 'task_id': 'a06',
         'timestamp': _ts(11, 0, 20), 'data': {'branch': 'task/a06', 'queue_depth': 1}},
        {'event_type': 'merge_dequeued', 'task_id': 'a06',
         'timestamp': _ts(11, 0, 30), 'data': {'branch': 'task/a06', 'queue_depth': 0}},
        {'event_type': 'merge_verify', 'task_id': 'a06', 'timestamp': _ts(11, 0, 45),
         'data': {'runner': 'local', 'passed': True, 'duration_ms': 600_000,
                  'attempt': 1, 'depth': 0, 'speculative': False, 'chain_items': 1},
         'duration_ms': None},
        {'event_type': 'merge_queued', 'task_id': 'a07',
         'timestamp': _ts(11, 4, 0), 'data': {'branch': 'task/a07', 'queue_depth': 1}},
        {'event_type': 'merge_dequeued', 'task_id': 'a07',
         'timestamp': _ts(11, 4, 5), 'data': {'branch': 'task/a07', 'queue_depth': 0}},
        {'event_type': 'merge_verify', 'task_id': 'a07', 'timestamp': _ts(11, 4, 10),
         'data': {'runner': 'laptop', 'passed': True, 'duration_ms': 300_000,
                  'attempt': 1, 'depth': 2, 'speculative': True, 'chain_items': 3},
         'duration_ms': None},
    ]

    # Heartbeats: depth series [3, 1, 0, 2]; two hosts.
    for day, hour, depth, local, laptop in (
        (11, 0, 3, 'busy', 'free'),
        (11, 6, 1, 'free', 'busy'),
        (12, 0, 0, 'free', 'free'),
        (14, 0, 2, 'busy', 'free'),
    ):
        events.append({
            'event_type': 'merge_heartbeat', 'task_id': None,
            'timestamp': _ts(day, hour),
            'data': {
                'depth': depth, 'verify_in_progress': depth > 0,
                'hosts': [
                    {'name': 'local', 'is_local': True, 'slot_state': local},
                    {'name': 'laptop', 'is_local': False, 'slot_state': laptop},
                ],
            },
        })

    # merge_attempt outcome mix, including the out-of-OutcomeKind 'superseded'.
    for i, outcome in enumerate(
        ['success'] * 4 + ['gate_retry', 'gate_retry', 'conflict', 'superseded']
    ):
        events.append({
            'event_type': 'merge_attempt', 'task_id': f'a{i:02d}',
            'timestamp': _ts(11, 2, i), 'data': {'outcome': outcome},
        })

    # Speculation: 10 speculative_merge, 3 chain_dead voids -> 0.30.
    for i in range(10):
        events.append({
            'event_type': 'speculative_merge', 'task_id': f'a{i:02d}',
            'timestamp': _ts(11, 3, i),
            'data': {'base_sha': f'sha{i}', 'depth': str(i % 3)},
        })
    for i, point in enumerate(['dispatch', 'dispatch', 'adoption']):
        events.append({
            'event_type': 'verdict_voided', 'task_id': f'a{i:02d}',
            'timestamp': _ts(11, 7, i),
            'data': {'dead_link': f'link{i}', 'reason': 'chain_dead', 'point': point},
        })
    return events


def corpus_b() -> list[dict[str, Any]]:
    """Event corpus for synthetic project root B (see known answers above)."""
    events: list[dict[str, Any]] = [
        _fin('b01', 11, 3),
        _fin('b02', 12, 3), _fin('b03', 12, 9),
        _fin('b04', 13, 3), _fin('b05', 13, 9),
        _fin('b06', 14, 3),
    ]
    events += [
        {'event_type': 'merge_queued', 'task_id': 'b01',
         'timestamp': _ts(11, 1, 0), 'data': {'branch': 'task/b01', 'queue_depth': 1}},
        {'event_type': 'merge_dequeued', 'task_id': 'b01',
         'timestamp': _ts(11, 1, 30), 'data': {'branch': 'task/b01', 'queue_depth': 0}},
        {'event_type': 'merge_verify', 'task_id': 'b01', 'timestamp': _ts(11, 2, 0),
         'data': {'runner': 'laptop', 'passed': True, 'duration_ms': 1_200_000,
                  'attempt': 1, 'depth': 1, 'speculative': True, 'chain_items': 2}},
        {'event_type': 'merge_heartbeat', 'task_id': None, 'timestamp': _ts(11, 0),
         'data': {'depth': 5, 'hosts': [
             {'name': 'laptop', 'is_local': False, 'slot_state': 'busy'}]}},
        {'event_type': 'merge_heartbeat', 'task_id': None, 'timestamp': _ts(13, 0),
         'data': {'depth': 1, 'hosts': [
             {'name': 'laptop', 'is_local': False, 'slot_state': 'free'}]}},
        {'event_type': 'merge_attempt', 'task_id': 'b01', 'timestamp': _ts(11, 2),
         'data': {'outcome': 'success'}},
        {'event_type': 'merge_attempt', 'task_id': 'b02', 'timestamp': _ts(12, 2),
         'data': {'outcome': 'cas_retry'}},
    ]
    # Speculation: 12 speculative_merge, 7 chain_dead voids -> 7/12 (~0.583).
    for i in range(12):
        events.append({
            'event_type': 'speculative_merge', 'task_id': f'b{i:02d}',
            'timestamp': _ts(12, 4, i),
            'data': {'base_sha': f'shb{i}', 'depth': str(i % 4)},
        })
    for i in range(7):
        events.append({
            'event_type': 'verdict_voided', 'task_id': f'b{i:02d}',
            'timestamp': _ts(12, 8, i),
            'data': {'dead_link': f'linkb{i}', 'reason': 'chain_dead',
                     'point': 'dispatch' if i % 2 else 'adoption'},
        })
    return events


@pytest.fixture
def corpus_roots(tmp_path):
    """Two synthetic project roots, A and B, each with its own runs.db."""
    return (
        _make_project_root(tmp_path, 'proj_a', corpus_a()),
        _make_project_root(tmp_path, 'proj_b', corpus_b()),
    )


# ---------------------------------------------------------------------------
# Fixture-builder self-checks — every later assertion rests on these producing
# live-shaped rows.
# ---------------------------------------------------------------------------


def test_make_runs_db_inserts_rows_in_list_order_with_ascending_ids(tmp_path):
    db = _make_runs_db(tmp_path, [
        {'event_type': 'merge_verify', 'task_id': 1, 'timestamp': _ts(11, 1)},
        {'event_type': 'merge_verify', 'task_id': 2, 'timestamp': _ts(11, 2)},
        {'event_type': 'merge_verify', 'task_id': 3, 'timestamp': _ts(11, 3)},
    ])
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute('SELECT id, task_id FROM events ORDER BY id').fetchall()
    finally:
        conn.close()
    assert [r[1] for r in rows] == ['1', '2', '3']
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)


def test_make_runs_db_passes_data_through_verbatim_for_str_and_none(tmp_path):
    db = _make_runs_db(tmp_path, [
        {'event_type': 'merge_verify', 'data': '{not json'},
        {'event_type': 'merge_verify', 'data': None},
        {'event_type': 'merge_verify', 'data': {'runner': 'local'}},
    ])
    conn = sqlite3.connect(db)
    try:
        rows = [r[0] for r in conn.execute('SELECT data FROM events ORDER BY id')]
    finally:
        conn.close()
    assert rows == ['{not json', None, '{"runner": "local"}']


# ---------------------------------------------------------------------------
# The I/O rim: _connect_ro + load_events.
# ---------------------------------------------------------------------------


def test_connect_ro_opens_readable_and_refuses_writes(tmp_path):
    db = _make_runs_db(tmp_path, [{'event_type': 'merge_verify'}])
    conn = mlt._connect_ro(db)
    try:
        assert conn.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 1
        # The mode=ro pin. If this ever starts passing silently, the script has
        # gained the ability to mutate a live orchestrator store.
        with pytest.raises(sqlite3.OperationalError, match='readonly database'):
            conn.execute(
                "INSERT INTO events (timestamp, run_id, event_type) "
                "VALUES ('x', 'y', 'z')"
            )
    finally:
        conn.close()


def test_connect_ro_refuses_a_missing_database(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        mlt._connect_ro(tmp_path / 'nope.db')


def test_load_events_returns_only_in_window_rows_of_the_requested_type(tmp_path):
    db = _make_runs_db(tmp_path, [
        # Before lo.
        {'event_type': 'merge_verify', 'task_id': 'early', 'timestamp': _ts(10, 11),
         'data': {'runner': 'local'}},
        # Exactly at lo -> inclusive.
        {'event_type': 'merge_verify', 'task_id': 'at_lo', 'timestamp': CORPUS_LO,
         'data': {'runner': 'local'}},
        {'event_type': 'merge_verify', 'task_id': 'mid', 'timestamp': _ts(12, 3),
         'data': {'runner': 'laptop'}},
        # Exactly at hi -> exclusive.
        {'event_type': 'merge_verify', 'task_id': 'at_hi', 'timestamp': CORPUS_HI,
         'data': {'runner': 'local'}},
        # Right type, out of window on the far side.
        {'event_type': 'merge_verify', 'task_id': 'late', 'timestamp': _ts(16, 3),
         'data': {'runner': 'local'}},
        # In window, wrong type.
        {'event_type': 'merge_finalized', 'task_id': 'other', 'timestamp': _ts(12, 4),
         'data': {'state': 'done'}},
    ])
    conn = mlt._connect_ro(db)
    try:
        rows = mlt.load_events(conn, 'merge_verify', CORPUS_LO_DT, CORPUS_HI_DT)
    finally:
        conn.close()
    assert [r['task_id'] for r in rows] == ['at_lo', 'mid']
    assert rows[0] == {'timestamp': CORPUS_LO, 'task_id': 'at_lo',
                       'data': {'runner': 'local'}}


def test_load_events_degrades_malformed_and_null_data_to_empty_dict(tmp_path):
    db = _make_runs_db(tmp_path, [
        {'event_type': 'merge_verify', 'task_id': 'bad', 'timestamp': _ts(12, 1),
         'data': '{not json'},
        {'event_type': 'merge_verify', 'task_id': 'null', 'timestamp': _ts(12, 2),
         'data': None},
        {'event_type': 'merge_verify', 'task_id': 'scalar', 'timestamp': _ts(12, 3),
         'data': '42'},
        {'event_type': 'merge_verify', 'task_id': 'ok', 'timestamp': _ts(12, 4),
         'data': {'runner': 'local'}},
    ])
    conn = mlt._connect_ro(db)
    try:
        rows = mlt.load_events(conn, 'merge_verify', CORPUS_LO_DT, CORPUS_HI_DT)
    finally:
        conn.close()
    # emit() is fire-and-forget, so a truncated row is possible and must not
    # abort the read of every other row in the window.
    assert [r['data'] for r in rows] == [{}, {}, {}, {'runner': 'local'}]


def test_load_events_returns_empty_list_when_nothing_matches(tmp_path):
    db = _make_runs_db(tmp_path, [
        {'event_type': 'merge_verify', 'timestamp': _ts(12, 1), 'data': {}},
    ])
    conn = mlt._connect_ro(db)
    try:
        assert mlt.load_events(conn, 'verdict_voided', CORPUS_LO_DT, CORPUS_HI_DT) == []
    finally:
        conn.close()


def test_corpus_roots_materialise_both_dbs_at_the_expected_path(corpus_roots):
    root_a, root_b = corpus_roots
    for root in (root_a, root_b):
        assert (root / 'data' / 'orchestrator' / 'runs.db').is_file()
    conn = mlt._connect_ro(root_a / 'data' / 'orchestrator' / 'runs.db')
    try:
        fins = mlt.load_events(conn, 'merge_finalized', CORPUS_LO_DT, CORPUS_HI_DT)
    finally:
        conn.close()
    # 16 in-window merge_finalized rows (13 done + 3 non-done); the Aug 9 row
    # is out of window.
    assert len(fins) == 16
    assert sum(1 for f in fins if f['data']['state'] == 'done') == 13


def test_percentile_returns_none_on_an_empty_series():
    # NOT 0.0 — "no laptop verify in this window" must never render as a p50 of
    # zero minutes.
    assert mlt._percentile([], 50) is None


def test_percentile_uses_linear_interpolation_between_order_statistics():
    assert mlt._percentile([10.0], 50) == 10.0
    assert mlt._percentile([0.0, 1.0, 3.0, 4.0], 50) == 2.0
    assert mlt._percentile([0.0, 1.0, 2.0, 3.0], 90) == pytest.approx(2.7)
    # Unsorted input must be sorted internally.
    assert mlt._percentile([4.0, 0.0, 3.0, 1.0], 50) == 2.0


# ---------------------------------------------------------------------------
# compute_landings_per_day
#
# The definition is load-bearing and was pinned empirically (plan decision 1):
# `merge_finalized` with data.state == 'done', bucketed by UTC calendar date,
# COMPLETE buckets only. The two plausible-but-wrong variants — counting every
# state, and keeping the partial leading bucket — are pinned as regressions
# below, because each yields a different median from the same rows.
# ---------------------------------------------------------------------------


def _in_window(events, event_type) -> list[dict[str, Any]]:
    """The corpus rows of *event_type* inside the corpus window, load_events-shaped."""
    return [
        {'timestamp': e['timestamp'], 'task_id': e['task_id'], 'data': e['data']}
        for e in events
        if e['event_type'] == event_type
        and CORPUS_LO <= e['timestamp'] < CORPUS_HI
    ]


def _landings_fixture() -> list[dict[str, Any]]:
    """Corpus A's in-window merge_finalized rows."""
    return _in_window(corpus_a(), 'merge_finalized')


def test_landings_counts_only_done_over_complete_buckets():
    result = mlt.compute_landings_per_day(
        _landings_fixture(), CORPUS_LO_DT, CORPUS_HI_DT
    )
    assert result['per_day'] == {
        '2026-08-11': 3,
        '2026-08-12': 1,
        '2026-08-13': 0,
        '2026-08-14': 4,
    }
    assert result['n_days'] == 4
    assert result['median'] == 2.0
    assert result['max'] == 4


def test_landings_excludes_the_partial_leading_bucket():
    result = mlt.compute_landings_per_day(
        _landings_fixture(), CORPUS_LO_DT, CORPUS_HI_DT
    )
    # Aug 10 holds 5 `done` rows, but CORPUS_LO cuts it mid-day, so counting it
    # would compare a half-day against four whole ones.
    assert '2026-08-10' not in result['per_day']
    # REGRESSION PIN: keeping it would give median 3, max 5 — a different table.
    assert (result['median'], result['max']) != (3, 5)


def test_landings_ignores_non_done_states():
    result = mlt.compute_landings_per_day(
        _landings_fixture(), CORPUS_LO_DT, CORPUS_HI_DT
    )
    # Aug 12 carries 1 done + 2 blocked + 1 superseded.
    assert result['per_day']['2026-08-12'] == 1
    # REGRESSION PIN: counting every state would give Aug 12: 4 and median 3.5.
    assert result['median'] != 3.5


def test_landings_zero_fills_an_empty_interior_day():
    result = mlt.compute_landings_per_day(
        _landings_fixture(), CORPUS_LO_DT, CORPUS_HI_DT
    )
    # Aug 13 saw no landing. Dropping the day instead of zero-filling would
    # inflate the median by shortening the series.
    assert result['per_day']['2026-08-13'] == 0
    assert result['n_days'] == 4


def test_landings_buckets_by_utc_calendar_date():
    events = [
        {'timestamp': '2026-08-11T23:59:59+00:00', 'task_id': 't1',
         'data': {'state': 'done'}},
        {'timestamp': '2026-08-12T00:00:01+00:00', 'task_id': 't2',
         'data': {'state': 'done'}},
    ]
    result = mlt.compute_landings_per_day(
        events,
        datetime(2026, 8, 11, 0, 0, tzinfo=UTC),
        datetime(2026, 8, 13, 0, 0, tzinfo=UTC),
    )
    assert result['per_day'] == {'2026-08-11': 1, '2026-08-12': 1}


def test_landings_keeps_but_labels_a_partial_trailing_bucket():
    # ASYMMETRY, deliberate: the leading partial bucket is dropped, the
    # trailing one is kept and named. Dropping both is more symmetric but is
    # NOT the rule the PRD's table was computed under — on the live dated
    # window it gives median 13.0 over 13 buckets against the PRD's 12.0 over
    # 14. Keeping it silent would be the real defect, so it is labelled.
    events = [
        {'timestamp': '2026-08-11T02:00:00+00:00', 'task_id': 't1',
         'data': {'state': 'done'}},
        {'timestamp': '2026-08-12T02:00:00+00:00', 'task_id': 't2',
         'data': {'state': 'done'}},
        {'timestamp': '2026-08-13T02:00:00+00:00', 'task_id': 't3',
         'data': {'state': 'done'}},
    ]
    result = mlt.compute_landings_per_day(
        events,
        datetime(2026, 8, 11, 0, 0, tzinfo=UTC),
        datetime(2026, 8, 13, 6, 0, tzinfo=UTC),   # mid-day hi
    )
    assert result['per_day'] == {
        '2026-08-11': 1, '2026-08-12': 1, '2026-08-13': 1,
    }
    assert result['n_days'] == 3
    assert result['partial_trailing_day'] == '2026-08-13'


def test_landings_reports_no_partial_trailing_day_when_hi_is_midnight():
    result = mlt.compute_landings_per_day(
        _landings_fixture(), CORPUS_LO_DT, CORPUS_HI_DT
    )
    assert result['partial_trailing_day'] is None


def test_landings_reports_none_when_no_complete_bucket_exists():
    events = [
        {'timestamp': '2026-08-11T13:00:00+00:00', 'task_id': 't1',
         'data': {'state': 'done'}},
    ]
    result = mlt.compute_landings_per_day(
        events,
        datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
        datetime(2026, 8, 11, 18, 0, tzinfo=UTC),
    )
    assert result['n_days'] == 0
    assert result['per_day'] == {}
    # None, not 0 — "the window is too short to hold a whole day" is not the
    # same finding as "a whole day passed with no landing".
    assert result['median'] is None
    assert result['max'] is None


def test_landings_tolerates_a_row_with_no_state_key():
    events = [
        {'timestamp': '2026-08-11T02:00:00+00:00', 'task_id': 't1', 'data': {}},
        {'timestamp': '2026-08-11T03:00:00+00:00', 'task_id': 't2',
         'data': {'state': 'done'}},
    ]
    result = mlt.compute_landings_per_day(
        events,
        datetime(2026, 8, 11, 0, 0, tzinfo=UTC),
        datetime(2026, 8, 12, 0, 0, tzinfo=UTC),
    )
    assert result['per_day'] == {'2026-08-11': 1}


def test_landings_corpus_b_known_answer():
    events = _in_window(corpus_b(), 'merge_finalized')
    result = mlt.compute_landings_per_day(events, CORPUS_LO_DT, CORPUS_HI_DT)
    assert result['per_day'] == {
        '2026-08-11': 1, '2026-08-12': 2, '2026-08-13': 2, '2026-08-14': 1,
    }
    assert result['median'] == 1.5
    assert result['max'] == 2


# ---------------------------------------------------------------------------
# compute_lead_time — the four-segment split
#   merge_queued -> merge_dequeued -> sum(merge_verify.data.duration_ms)
#                -> merge_finalized
#
# The join is on the LAST merge_queued STRICTLY BEFORE the landing, per task_id
# (plan decision 2). A task re-enters the queue on gate_retry, cas_retry and
# supersede, so a first-queued join measures "time since the task first tried",
# not lead time — live, that was off by 18x at p90.
#
# LEAD-TIME FIXTURE, hand-computed (minutes):
#   task  lead  wait  verify  residual   note
#   T1     60    10      15        35    queued THREE times before landing
#   T2     40    20       0        20    no merge_verify row in window
#   T3     20     5      10         5    re-queued AFTER landing (must not join)
#   T4     80    15      30        35
#   T5      -     -       -         -    landing with no preceding queue
#   T6      -     -       -         -    finalized 'blocked' -> not a landing
# lead     sorted [20,40,60,80] -> p50 50.0, p90 74.0
# wait     sorted [5,10,15,20]  -> p50 12.5, p90 18.5
# verify   sorted [10,15,30]    -> p50 15.0, p90 27.0  (n=3: T2 has no rows)
# residual sorted [5,20,35,35]  -> p50 27.5, p90 35.0
# ---------------------------------------------------------------------------


def _q(task_id, hour, minute) -> dict[str, Any]:
    return {'timestamp': _ts(11, hour, minute), 'task_id': task_id,
            'data': {'branch': f'task/{task_id}', 'queue_depth': 1}}


def _v(task_id, hour, minute, duration_ms, **extra) -> dict[str, Any]:
    return {'timestamp': _ts(11, hour, minute), 'task_id': task_id,
            'data': {'runner': 'local', 'passed': True,
                     'duration_ms': duration_ms, **extra}}


def _f(task_id, hour, minute, state='done') -> dict[str, Any]:
    return {'timestamp': _ts(11, hour, minute), 'task_id': task_id,
            'data': {'branch': f'task/{task_id}', 'state': state}}


def _lead_fixture():
    queued = [
        _q('T1', 0, 0), _q('T1', 0, 30), _q('T1', 1, 0),
        _q('T2', 3, 0),
        _q('T3', 5, 0),
        _q('T3', 6, 0),          # re-queue AFTER T3 landed at 05:20
        _q('T4', 7, 0),
        _q('T6', 10, 0),
    ]
    dequeued = [
        _q('T1', 1, 10), _q('T2', 3, 20), _q('T3', 5, 5), _q('T4', 7, 15),
        _q('T6', 10, 5),
    ]
    verify = [
        _v('T1', 1, 15, 300_000), _v('T1', 1, 25, 600_000),
        _v('T3', 5, 6, 600_000),
        _v('T4', 7, 20, 1_800_000),
    ]
    finalized = [
        _f('T1', 2, 0), _f('T2', 3, 40), _f('T3', 5, 20), _f('T4', 8, 20),
        _f('T5', 9, 0),                       # no preceding merge_queued
        _f('T6', 10, 30, state='blocked'),    # not a landing
    ]
    return queued, dequeued, verify, finalized


def _lead_result():
    q, d, v, f = _lead_fixture()
    return mlt.compute_lead_time(q, d, v, f, CORPUS_LO_DT, CORPUS_HI_DT)


def test_lead_time_joins_on_the_last_queued_strictly_before_the_landing():
    result = _lead_result()
    assert result['lead']['p50'] == 50.0
    assert result['lead']['p90'] == pytest.approx(74.0)
    assert result['lead']['n'] == 4
    # REGRESSION PIN: a FIRST-queued join would take T1's 00:00 row and give a
    # 120-minute lead, moving p50 to 60.0. Live, the same mistake moved p90
    # from 171 to 3114 minutes.
    assert result['lead']['p50'] != 60.0


def test_lead_time_ignores_a_requeue_that_happened_after_the_landing():
    result = _lead_result()
    # T3 was re-queued at 06:00, twenty minutes AFTER it landed at 05:20.
    # Taking the last queued row overall (rather than the last one BEFORE the
    # landing) would give a negative lead.
    assert result['lead']['min'] == 20.0


def test_lead_time_queue_wait_is_dequeued_minus_last_queued():
    result = _lead_result()
    assert result['wait']['p50'] == 12.5
    assert result['wait']['p90'] == pytest.approx(18.5)
    assert result['wait']['n'] == 4


def test_lead_time_verify_sums_the_payload_duration_not_the_column():
    q, d, v, f = _lead_fixture()
    # The events table's duration_ms COLUMN is NULL for merge_verify; the real
    # figure lives in data['duration_ms']. Poison the column to prove it is
    # never read.
    v = [dict(row, duration_ms=999_999_999) for row in v]
    result = mlt.compute_lead_time(q, d, v, f, CORPUS_LO_DT, CORPUS_HI_DT)
    assert result['verify']['p50'] == 15.0
    assert result['verify']['p90'] == pytest.approx(27.0)
    # n=3, not 4: T2 has no merge_verify row in the window, and a zero there
    # would drag the verify percentile down as if the verify were instant.
    assert result['verify']['n'] == 3


def test_lead_time_residual_is_lead_minus_wait_minus_verify():
    result = _lead_result()
    assert result['residual']['p50'] == 27.5
    assert result['residual']['p90'] == pytest.approx(35.0)
    assert result['residual']['n'] == 4


def test_lead_time_reports_a_residual_when_verify_rows_are_absent():
    q, d, _v_rows, f = _lead_fixture()
    result = mlt.compute_lead_time(q, d, [], f, CORPUS_LO_DT, CORPUS_HI_DT)
    # With no verify rows at all the split still resolves: residual absorbs the
    # unattributed time rather than the section vanishing.
    assert result['verify']['n'] == 0
    assert result['verify']['p50'] is None
    assert result['residual']['n'] == 4
    # residuals become lead - wait: [50, 20, 15, 65] -> sorted [15,20,50,65]
    assert result['residual']['p50'] == 35.0


def test_lead_time_excludes_a_landing_with_no_preceding_queue():
    result = _lead_result()
    # T5 landed without a merge_queued row in the supplied set. It is counted
    # and reported, never silently folded into the matched series.
    assert result['matched'] == 4
    assert result['unmatched'] == 1
    assert 'T5' in result['unmatched_task_ids']


def test_lead_time_ignores_non_done_finalize_rows():
    result = _lead_result()
    # T6 was queued, dequeued and finalized 'blocked'. It is not a landing, so
    # it appears in neither the matched nor the unmatched tally.
    assert 'T6' not in result['unmatched_task_ids']
    assert result['matched'] + result['unmatched'] == 5


def test_lead_time_is_empty_but_not_zero_on_an_empty_window():
    result = mlt.compute_lead_time([], [], [], [], CORPUS_LO_DT, CORPUS_HI_DT)
    for series in ('lead', 'wait', 'verify', 'residual'):
        assert result[series]['n'] == 0
        assert result[series]['p50'] is None
        assert result[series]['p90'] is None
    assert result['matched'] == 0
    assert result['unmatched'] == 0


def test_lead_time_wait_needs_a_dequeue_after_the_joined_queue():
    q = [_q('Z1', 1, 0)]
    d = [_q('Z1', 0, 30)]          # dequeue predates the joined queue row
    f = [_f('Z1', 2, 0)]
    result = mlt.compute_lead_time(q, d, [], f, CORPUS_LO_DT, CORPUS_HI_DT)
    assert result['lead']['n'] == 1
    assert result['wait']['n'] == 0
    # Residual needs a wait to mean "finalize + CAS", so it is not invented.
    assert result['residual']['n'] == 0


# ---------------------------------------------------------------------------
# compute_verify_by_runner
#
# RUNNER FIXTURE, hand-computed:
#   local:  durations 10, 20, 30, 40 min; passed T,T,T,F -> pass rate 0.75
#           p50 25.0, p90 37.0
#   laptop: durations 5, 15 min;          passed T,F     -> pass rate 0.5
#           p50 10.0, p90 14.0
#   (unknown): one row with runner=None
# ---------------------------------------------------------------------------


def _runner_fixture() -> list[dict[str, Any]]:
    rows = [
        _v('r1', 1, 0, 600_000, runner='local', passed=True),
        _v('r2', 1, 1, 1_200_000, runner='local', passed=True),
        _v('r3', 1, 2, 1_800_000, runner='local', passed=True),
        _v('r4', 1, 3, 2_400_000, runner='local', passed=False),
        _v('r5', 1, 4, 300_000, runner='laptop', passed=True),
        _v('r6', 1, 5, 900_000, runner='laptop', passed=False),
    ]
    # _v seeds runner='local'/passed=True; the kwargs above override.
    return [dict(r, data={**r['data']}) for r in rows]


def test_verify_by_runner_groups_and_summarises_each_runner():
    result = mlt.compute_verify_by_runner(_runner_fixture())
    assert set(result['runners']) == {'local', 'laptop'}
    local = result['runners']['local']
    assert local['n'] == 4
    assert local['p50'] == 25.0
    assert local['p90'] == pytest.approx(37.0)
    assert local['pass_rate'] == 0.75
    laptop = result['runners']['laptop']
    assert laptop['n'] == 2
    assert laptop['p50'] == 10.0
    assert laptop['p90'] == pytest.approx(14.0)
    assert laptop['pass_rate'] == 0.5


def test_verify_by_runner_reads_payload_duration_not_the_column():
    rows = [dict(r, duration_ms=999_999_999) for r in _runner_fixture()]
    result = mlt.compute_verify_by_runner(rows)
    assert result['runners']['local']['p50'] == 25.0


def test_verify_by_runner_omits_a_runner_with_no_rows():
    result = mlt.compute_verify_by_runner(
        [r for r in _runner_fixture() if r['data']['runner'] == 'local']
    )
    # Absent, not reported as a zero — "no laptop verify ran in this window" is
    # not the same finding as "the laptop verified instantly, and never passed".
    assert 'laptop' not in result['runners']


def test_verify_by_runner_buckets_a_missing_runner_under_an_explicit_label():
    rows = _runner_fixture()
    rows.append(_v('r7', 1, 6, 600_000, runner=None, passed=True))
    rows.append({'timestamp': _ts(11, 1, 7), 'task_id': 'r8',
                 'data': {'duration_ms': 600_000, 'passed': True}})
    result = mlt.compute_verify_by_runner(rows)
    unknown = result['runners'][mlt.UNKNOWN]
    assert unknown['n'] == 2


def test_verify_by_runner_skips_a_row_with_no_usable_duration():
    rows = [
        _v('r1', 1, 0, 600_000, runner='local', passed=True),
        _v('r2', 1, 1, None, runner='local', passed=True),
        {'timestamp': _ts(11, 1, 2), 'task_id': 'r3',
         'data': {'runner': 'local', 'passed': False}},
    ]
    result = mlt.compute_verify_by_runner(rows)
    local = result['runners']['local']
    assert local['n'] == 3            # every row counts toward the pass rate
    assert local['n_durations'] == 1  # only one carried a usable duration
    assert local['p50'] == 10.0


# FORWARD COMPAT: `fallback_reason` is a task-C key. Task C is downstream of
# this one, so on main NO merge_verify row carries it. "This window contains no
# row with the key" must stay distinguishable from "the key was present and the
# count was 0", or a pre-task-C window reads as evidence that busy-fallbacks
# never happen.


def test_verify_by_runner_reports_fallback_as_key_not_present_before_task_c():
    result = mlt.compute_verify_by_runner(_runner_fixture())
    assert result['fallback_key_present'] is False
    for runner in result['runners'].values():
        assert runner['fallback_reasons'] == {}


def test_verify_by_runner_counts_fallback_reason_per_runner_once_task_c_lands():
    rows = _runner_fixture()
    rows.append(_v('r7', 1, 6, 600_000, runner='local', passed=True,
                   fallback_reason='remote_busy'))
    rows.append(_v('r8', 1, 7, 600_000, runner='local', passed=True,
                   fallback_reason='remote_busy'))
    rows.append(_v('r9', 1, 8, 600_000, runner='laptop', passed=True,
                   fallback_reason='quarantined'))
    result = mlt.compute_verify_by_runner(rows)
    assert result['fallback_key_present'] is True
    assert result['runners']['local']['fallback_reasons'] == {'remote_busy': 2}
    assert result['runners']['laptop']['fallback_reasons'] == {'quarantined': 1}


def test_verify_by_runner_is_empty_on_no_rows():
    result = mlt.compute_verify_by_runner([])
    assert result['runners'] == {}
    assert result['fallback_key_present'] is False


# ---------------------------------------------------------------------------
# compute_occupancy — THREE estimators, side by side, never reconciled.
#
# OCCUPANCY FIXTURE. Window 2026-08-11 00:00Z .. 10:00Z (600 min). Heartbeats
# are at irregular intervals and hosts drop in and out of the `hosts` list, so
# each host's LOCF integral is over ITS OWN samples.
#
#   laptop  samples 00:00 busy, 01:00 busy, 03:00 free, 04:00 busy, 09:00 free
#           LOCF busy = 60 + 120 + 0 + 300 + 0 = 480 / 600 = 0.80
#           raw-sample = 3 busy / 5 samples          = 0.60
#           verify-duration = 60 min / 600           = 0.10
#   local   samples 00:00 free, 02:00 busy, 03:00 free, 05:00 busy
#           LOCF busy = 60 + 300 (last sample carried to hi) = 360 / 600 = 0.60
#           raw-sample = 2 / 4                        = 0.50
#           verify-duration = 120 min / 600           = 0.20
#   idle    samples 00:00 parked, 05:00 None -> never busy: 0.0, not None
#
# All three estimators differ for BOTH real hosts, so a test that conflated any
# two of them fails. That spread is the finding (plan decision 4): the live
# measurement on reify over the dated 14d window was LOCF 22.2% (the PRD row)
# against raw-sample 33.4% against verify-duration-sum 1.3%.
# ---------------------------------------------------------------------------

OCC_LO = datetime(2026, 8, 11, 0, 0, tzinfo=UTC)
OCC_HI = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)


def _hb(hour, hosts, depth=0) -> dict[str, Any]:
    return {'timestamp': _ts(11, hour), 'task_id': None,
            'data': {'depth': depth,
                     'hosts': [{'name': n, 'is_local': n == 'local',
                                'slot_state': s} for n, s in hosts]}}


def _occ_fixture():
    heartbeats = [
        _hb(0, [('laptop', 'busy'), ('local', 'free'), ('idle', 'parked')]),
        _hb(1, [('laptop', 'busy')]),
        _hb(2, [('local', 'busy')]),
        _hb(3, [('laptop', 'free'), ('local', 'free')]),
        _hb(4, [('laptop', 'busy')]),
        _hb(5, [('local', 'busy'), ('idle', None)]),
        _hb(9, [('laptop', 'free')]),
    ]
    verify = [
        _v('v1', 1, 0, 3_600_000, runner='laptop'),
        _v('v2', 2, 0, 5_400_000, runner='local'),
        _v('v3', 6, 0, 1_800_000, runner='local'),
    ]
    return heartbeats, verify


def _occ_result():
    hb, v = _occ_fixture()
    return mlt.compute_occupancy(hb, v, OCC_LO, OCC_HI)


def test_occupancy_reports_all_three_estimators_for_every_host():
    hosts = _occ_result()['hosts']
    assert set(hosts) == {'laptop', 'local', 'idle'}
    for entry in hosts.values():
        # None suppressed, none reconciled, no "preferred" figure.
        assert set(entry) >= {
            'locf_busy_fraction', 'raw_sample_fraction',
            'verify_duration_fraction', 'slot_states', 'n_samples',
        }


def test_occupancy_locf_integral_carries_each_sample_to_the_next():
    hosts = _occ_result()['hosts']
    assert hosts['laptop']['locf_busy_fraction'] == pytest.approx(0.80)
    assert hosts['local']['locf_busy_fraction'] == pytest.approx(0.60)


def test_occupancy_raw_sample_fraction_disagrees_with_the_locf_integral():
    hosts = _occ_result()['hosts']
    assert hosts['laptop']['raw_sample_fraction'] == pytest.approx(0.60)
    assert hosts['local']['raw_sample_fraction'] == pytest.approx(0.50)
    for name in ('laptop', 'local'):
        assert (hosts[name]['raw_sample_fraction']
                != hosts[name]['locf_busy_fraction'])


def test_occupancy_verify_duration_fraction_disagrees_with_both():
    hosts = _occ_result()['hosts']
    assert hosts['laptop']['verify_duration_fraction'] == pytest.approx(0.10)
    assert hosts['local']['verify_duration_fraction'] == pytest.approx(0.20)
    for name in ('laptop', 'local'):
        entry = hosts[name]
        assert len({entry['locf_busy_fraction'], entry['raw_sample_fraction'],
                    entry['verify_duration_fraction']}) == 3


def test_occupancy_counts_parked_and_none_as_not_busy_but_tallies_them():
    idle = _occ_result()['hosts']['idle']
    assert idle['locf_busy_fraction'] == 0.0
    assert idle['raw_sample_fraction'] == 0.0
    # The states themselves are reported — 'parked' is a quarantine signal, not
    # an unremarkable idle, and must not disappear into "not busy".
    assert idle['slot_states'] == {'parked': 1, mlt.UNKNOWN: 1}


def test_occupancy_last_sample_is_carried_forward_to_the_window_end():
    # local's final sample is 'busy' at 05:00; the 300 minutes to `hi` are the
    # difference between 0.60 and 0.10.
    hosts = _occ_result()['hosts']
    assert hosts['local']['locf_busy_fraction'] == pytest.approx(0.60)


def test_occupancy_denominator_starts_at_a_hosts_first_sample():
    hb = [_hb(5, [('late', 'busy')])]
    result = mlt.compute_occupancy(hb, [], OCC_LO, OCC_HI)
    late = result['hosts']['late']
    # The host was unobserved for the window's first five hours. Charging it
    # with those hours as "not busy" would report 0.5 for a host that was busy
    # for every minute it was actually seen.
    assert late['locf_busy_fraction'] == pytest.approx(1.0)
    assert late['observed_span_minutes'] == pytest.approx(300.0)


def test_occupancy_tolerates_a_heartbeat_with_an_empty_hosts_list():
    hb = [
        {'timestamp': _ts(11, 0), 'task_id': None,
         'data': {'depth': 0, 'hosts': []}},
        _hb(1, [('laptop', 'busy')]),
    ]
    result = mlt.compute_occupancy(hb, [], OCC_LO, OCC_HI)
    # `hosts` is [] before the allocator has ever dispatched; that contributes
    # no host rather than an unnamed one, and must not crash.
    assert set(result['hosts']) == {'laptop'}


def test_occupancy_tolerates_a_heartbeat_with_no_hosts_key_at_all():
    hb = [{'timestamp': _ts(11, 0), 'task_id': None, 'data': {'depth': 3}}]
    result = mlt.compute_occupancy(hb, [], OCC_LO, OCC_HI)
    assert result['hosts'] == {}


def test_occupancy_is_none_not_zero_on_a_window_with_no_heartbeats():
    result = mlt.compute_occupancy([], [], OCC_LO, OCC_HI)
    assert result['hosts'] == {}
    # A window with no heartbeat is unmeasured, not idle.
    assert result['n_heartbeats'] == 0


def test_occupancy_verify_fraction_is_none_when_a_host_ran_no_verify():
    hosts = _occ_result()['hosts']
    assert hosts['idle']['verify_duration_fraction'] is None


def test_occupancy_stamps_the_window_span_it_divided_by():
    result = _occ_result()
    assert result['window_span_minutes'] == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# compute_speculation
#
# The depth ASYMMETRY is the trap: `_emit_speculative` str-coerces every value,
# so speculative_merge.data.depth is a STR ("0"), while merge_verify.data.depth
# is a native int|None. The two distributions are kept SEPARATE so nothing ever
# aggregates across the types.
# ---------------------------------------------------------------------------


def _sm(task_id, minute, depth) -> dict[str, Any]:
    return {'timestamp': _ts(11, 3, minute), 'task_id': task_id,
            'data': {'base_sha': f'sha{minute}', 'depth': depth}}


def _vv(task_id, minute, point, reason='chain_dead') -> dict[str, Any]:
    return {'timestamp': _ts(11, 7, minute), 'task_id': task_id,
            'data': {'dead_link': f'l{minute}', 'reason': reason, 'point': point}}


def _spec_fixture():
    speculative = [
        _sm('s1', 0, '0'), _sm('s2', 1, '0'), _sm('s3', 2, '1'),
        _sm('s4', 3, '2'), _sm('s5', 4, None), _sm('s6', 5, 'not-a-depth'),
    ]
    voided = [
        _vv('s1', 0, 'dispatch'), _vv('s2', 1, 'dispatch'),
        _vv('s3', 2, 'adoption'), _vv('s4', 3, 'dispatch', reason='other'),
    ]
    verify = [
        _v('L1', 4, 0, 600_000, depth=0, speculative=False),
        _v('L2', 4, 1, 600_000, depth=1, speculative=True),
        _v('L3', 4, 2, 600_000, depth=1, speculative=False),
        _v('L4', 4, 3, 600_000, depth=None, speculative=False),
    ]
    finalized = [_f('L1', 5, 0), _f('L2', 5, 1), _f('L3', 5, 2)]
    return speculative, voided, verify, finalized


def _spec_result():
    return mlt.compute_speculation(*_spec_fixture())


def test_speculation_coerces_str_depth_and_skips_unusable_values():
    dist = _spec_result()['speculative_depth']
    # "0","0","1","2" coerce; None and 'not-a-depth' are skipped, not crashes
    # and not a zero bucket.
    assert dist['distribution'] == {0: 2, 1: 1, 2: 1}
    assert dist['n_coerced'] == 4
    assert dist['n_skipped'] == 2


def test_speculation_keeps_the_verify_depth_distribution_separate():
    result = _spec_result()
    assert result['verify_depth']['distribution'] == {0: 1, 1: 2}
    assert result['verify_depth']['n_skipped'] == 1
    # Two distributions, never summed: one is keyed off str depths, the other
    # off native ints, and merging them would silently double-count.
    assert result['verify_depth'] is not result['speculative_depth']


def test_speculation_void_rate_counts_only_chain_dead():
    result = _spec_result()
    # 3 chain_dead voids over 6 speculative merges. The fourth voided row
    # carries reason='other' and is excluded from the numerator.
    assert result['void_rate'] == pytest.approx(0.5)
    assert result['n_voided_chain_dead'] == 3
    assert result['n_speculative'] == 6


def test_speculation_tallies_the_void_point():
    assert _spec_result()['void_points'] == {'dispatch': 2, 'adoption': 1}


def test_speculation_ahead_landing_share_is_matched_over_total():
    result = _spec_result()
    # L1 had a speculative_merge? No — but L2 has a speculative merge_verify,
    # and L3 has neither. Only tasks with speculative evidence count.
    assert result['speculative_ahead']['total'] == 3
    assert result['speculative_ahead']['matched'] == 1
    assert result['speculative_ahead']['share'] == pytest.approx(1 / 3)


def test_speculation_ahead_counts_a_preceding_speculative_merge_too():
    speculative, voided, verify, finalized = _spec_fixture()
    speculative = [*speculative, _sm('L3', 6, '1')]
    result = mlt.compute_speculation(speculative, voided, verify, finalized)
    assert result['speculative_ahead']['matched'] == 2


def test_speculation_ignores_a_speculative_merge_after_the_landing():
    speculative, voided, verify, finalized = _spec_fixture()
    # _sm builds a 03:xx timestamp; this landing is at 02:00, so the
    # speculative merge happened afterwards and is not evidence for it.
    finalized = [{'timestamp': _ts(11, 2, 0), 'task_id': 'LATE',
                  'data': {'state': 'done'}}]
    speculative = [*speculative, _sm('LATE', 7, '1')]
    result = mlt.compute_speculation(speculative, voided, verify, finalized)
    assert result['speculative_ahead']['matched'] == 0


def test_speculation_returns_none_rates_on_a_zero_speculation_window():
    result = mlt.compute_speculation([], [], [], [])
    # None, not 0.0: "nothing speculated in this window" is not "speculation
    # was tried and never voided".
    assert result['void_rate'] is None
    assert result['speculative_ahead']['share'] is None
    assert result['n_speculative'] == 0
    assert result['speculative_depth']['distribution'] == {}


def test_speculation_void_rate_is_none_when_voids_exist_without_speculation():
    result = mlt.compute_speculation([], [_vv('x', 0, 'dispatch')], [], [])
    assert result['void_rate'] is None
    assert result['n_voided_chain_dead'] == 1
