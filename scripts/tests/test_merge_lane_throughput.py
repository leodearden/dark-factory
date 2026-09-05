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


def _make_runs_db(tmp_path: Path, events: list[dict], name: str = 'runs.db') -> Path:
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


def _make_project_root(tmp_path: Path, name: str, events: list[dict]) -> Path:
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


def _fin(task_id, day, hour, state='done', **data):
    return {
        'event_type': 'merge_finalized',
        'task_id': task_id,
        'timestamp': _ts(day, hour),
        'data': {'branch': f'task/{task_id}', 'state': state, **data},
    }


def corpus_a() -> list[dict]:
    """Event corpus for synthetic project root A (see known answers above)."""
    events: list[dict] = []

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


def corpus_b() -> list[dict]:
    """Event corpus for synthetic project root B (see known answers above)."""
    events: list[dict] = [
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


def _in_window(events, event_type):
    """The corpus rows of *event_type* inside the corpus window, load_events-shaped."""
    return [
        {'timestamp': e['timestamp'], 'task_id': e['task_id'], 'data': e['data']}
        for e in events
        if e['event_type'] == event_type
        and CORPUS_LO <= e['timestamp'] < CORPUS_HI
    ]


def _landings_fixture():
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


def test_landings_drops_a_partial_trailing_bucket_too():
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
    assert result['per_day'] == {'2026-08-11': 1, '2026-08-12': 1}
    assert result['n_days'] == 2


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
