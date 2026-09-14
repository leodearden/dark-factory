"""Tests for dashboard.data.load — /api/load data module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from dashboard.data.load import (
    KNOWN_METRICS,
    LOAD_SAMPLES_SCHEMA,
    PROCESS_METRICS,
    PSI_METRICS,
    get_load_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def populated_load_db(tmp_path: Path) -> Path:
    """Build a tmp_path samples DB with a mix of known and unknown metrics.

    Inserts:
    - 3 rows for occt_queue_depth (ts=100/110/120, value=1/2/3; latest has window_mean=2.0, window_max=3.0)
    - 2 rows for psi_cpu_some_avg10 (ts=100/120, value=5/7.5; window_mean=NULL, window_max=NULL)
    - 1 row for bogus_metric (unknown, should be filtered out)
    """
    db_path = tmp_path / 'load-samples.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(LOAD_SAMPLES_SCHEMA)

    # occt_queue_depth — process metric, last row carries window_mean/window_max
    conn.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        [
            (100, 'occt_queue_depth', 1.0, None, None),
            (110, 'occt_queue_depth', 2.0, None, None),
            (120, 'occt_queue_depth', 3.0, 2.0, 3.0),
        ],
    )

    # psi_cpu_some_avg10 — PSI metric, window cols always NULL
    conn.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        [
            (100, 'psi_cpu_some_avg10', 5.0, None, None),
            (120, 'psi_cpu_some_avg10', 7.5, None, None),
        ],
    )

    # Unknown metric — should not appear in result
    conn.execute(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        (100, 'bogus_metric', 42.0, None, None),
    )

    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests: core shape contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_all_known_keys_with_values_or_placeholders(populated_load_db: Path) -> None:
    """All 9 KNOWN_METRICS keys present; populated and placeholder metrics correct."""
    async with aiosqlite.connect(str(populated_load_db)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    # All 9 known metrics present
    assert set(result.keys()) == set(KNOWN_METRICS)

    # Populated process metric with window data on the latest row
    assert result['occt_queue_depth'] == {
        'current': 3.0,
        'sparkline': [1.0, 2.0, 3.0],
        'window_mean': 2.0,
        'window_max': 3.0,
    }

    # PSI metric: window cols always NULL
    assert result['psi_cpu_some_avg10'] == {
        'current': 7.5,
        'sparkline': [5.0, 7.5],
        'window_mean': None,
        'window_max': None,
    }

    # Unpopulated known metric → placeholder shape
    assert result['verify_concurrency'] == {
        'current': None,
        'sparkline': [],
        'window_mean': None,
        'window_max': None,
    }

    # Unknown metric must be absent
    assert 'bogus_metric' not in result


# ---------------------------------------------------------------------------
# Tests: empty and None DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_db_returns_all_placeholders(tmp_path: Path) -> None:
    """Empty DB (schema exists, no rows) returns placeholder for every known metric."""
    db_path = tmp_path / 'empty-load.db'
    conn_sync = sqlite3.connect(str(db_path))
    conn_sync.executescript(LOAD_SAMPLES_SCHEMA)
    conn_sync.commit()
    conn_sync.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    placeholder = {'current': None, 'sparkline': [], 'window_mean': None, 'window_max': None}

    assert len(result) == len(KNOWN_METRICS) == 9
    assert set(result.keys()) == set(KNOWN_METRICS)
    for metric in KNOWN_METRICS:
        assert result[metric] == placeholder, f'{metric} not placeholder: {result[metric]}'


@pytest.mark.asyncio
async def test_none_db_returns_all_placeholders() -> None:
    """None connection (DB absent/unavailable) returns placeholder for every known metric."""
    result = await get_load_metrics(None)

    placeholder = {'current': None, 'sparkline': [], 'window_mean': None, 'window_max': None}

    assert len(result) == len(KNOWN_METRICS) == 9
    assert set(result.keys()) == set(KNOWN_METRICS)
    for metric in KNOWN_METRICS:
        assert result[metric] == placeholder, f'{metric} not placeholder: {result[metric]}'


# ---------------------------------------------------------------------------
# Tests: sparkline cap and ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sparkline_clamped_to_60_and_chronological(tmp_path: Path) -> None:
    """75 rows for verify_concurrency → sparkline has exactly 60 entries (oldest 60..75th retained),
    in ascending ts order. current == 75.0."""
    db_path = tmp_path / 'capped-load.db'
    conn_sync = sqlite3.connect(str(db_path))
    conn_sync.executescript(LOAD_SAMPLES_SCHEMA)
    conn_sync.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        [(ts, 'verify_concurrency', float(ts), None, None) for ts in range(1, 76)],
    )
    conn_sync.commit()
    conn_sync.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    sparkline = result['verify_concurrency']['sparkline']
    assert len(sparkline) == 60
    # Oldest kept row (ts=16) is first; newest (ts=75) is last
    assert sparkline[0] == 16.0
    assert sparkline[-1] == 75.0
    assert result['verify_concurrency']['current'] == 75.0


# ---------------------------------------------------------------------------
# Tests: DashboardConfig.load_samples_db property
# ---------------------------------------------------------------------------


def test_load_samples_db_property_points_to_canonical_path(tmp_path: Path) -> None:
    """DashboardConfig.load_samples_db returns <project_root>/data/load-samples.db."""
    from dashboard.config import DashboardConfig

    config = DashboardConfig(project_root=tmp_path)
    # tmp_path is already resolved; __post_init__ resolves project_root too
    assert config.load_samples_db == config.project_root / 'data' / 'load-samples.db'


# ---------------------------------------------------------------------------
# Tests: schema and metric drift guard
# ---------------------------------------------------------------------------


def test_load_schema_and_metrics_match_sampler() -> None:
    """LOAD_SAMPLES_SCHEMA and KNOWN_METRICS stay aligned with the sampler package.

    Skips gracefully when the sampler package is not importable (dashboard-only
    CI environments that do not install sampler as a dependency).
    """
    sampler_store = pytest.importorskip('sampler.store')
    sampler_metrics_mod = pytest.importorskip('sampler.metrics')

    # --- Schema alignment ---
    assert LOAD_SAMPLES_SCHEMA == sampler_store._SCHEMA, (
        "LOAD_SAMPLES_SCHEMA in data/load.py has drifted from "
        "sampler.store._SCHEMA — update the constant to stay aligned."
    )

    # --- PSI metric names ---
    # collect_psi only needs a callable reader; no filesystem or kernel access.
    stub_psi_text = 'some avg10=0.0\nfull avg10=0.0'
    sampler_psi_keys = frozenset(
        sampler_metrics_mod.collect_psi(read=lambda _name: stub_psi_text).keys()
    )
    assert sampler_psi_keys == PSI_METRICS, (
        f'PSI metric mismatch — sampler emits {sorted(sampler_psi_keys)}, '
        f'dashboard PSI_METRICS has {sorted(PSI_METRICS)}'
    )

    # --- Process metric names ---
    # collect_process_metrics imports psutil at call time; skip if unavailable.
    pytest.importorskip('psutil', reason='psutil required for process-metric name check')
    process_keys = frozenset(
        sampler_metrics_mod.collect_process_metrics(
            proc_iter=lambda *_a, **_kw: [],
            fd9_exists=lambda _pid: False,
        ).keys()
    )
    assert process_keys == PROCESS_METRICS, (
        f'Process metric mismatch — sampler emits {sorted(process_keys)}, '
        f'dashboard PROCESS_METRICS has {sorted(PROCESS_METRICS)}'
    )


# ---------------------------------------------------------------------------
# Tests: recency bound (task 3592)
# ---------------------------------------------------------------------------


def test_recency_slack_is_sized_for_the_sparkline_span() -> None:
    """Pin the slack's SIZING, which no behavioural test can express.

    Once the bound exists, cost is linear in the SLACK rather than in
    retention: measured 1h = 28.7 ms, 24h = 347 ms, 7d = 2,168 ms against a
    4,665,600-row probe.  So the value has a floor (it must clear the
    sparkline's span or a full sparkline gets truncated) and a ceiling (or the
    bound stops paying for itself).  Both ends are real, and a value assertion
    is the only way to state them.

    The bound's BEHAVIOUR is deliberately not asserted here.  The two async
    tests below cover it non-vacuously, and an earlier version of this test
    grepped _QUERY_SQL for the substrings 'ts >=' and 'MAX(ts)' instead --
    which pinned the SQL's SPELLING, not its behaviour.  That form went red on
    a behaviour-preserving rewrite (`ts>=`, lowercase `max(ts)`, BETWEEN, a
    CTE) while staying green for a bound applied to the wrong side.
    """
    from dashboard.data.load import _RECENCY_SLACK_SECONDS

    # Sparkline spans 60 samples x 5s tick = 300s; slack must clear that...
    assert _RECENCY_SLACK_SECONDS >= 300
    # ...but stay modest, since cost is linear in the slack (7d measured 2.2s).
    assert _RECENCY_SLACK_SECONDS <= 86400


@pytest.mark.asyncio
async def test_bound_excludes_ancient_rows_but_keeps_the_live_window(tmp_path: Path) -> None:
    """Rows far older than the newest sample are excluded from the sparkline.

    Deliberately uses FEWER than 60 live samples: with a full 60 the ancient
    rows sort to rn 61+ and `rn <= 60` masks them regardless of the bound, so
    such a test would pass unbounded and guard nothing.  With 10 live samples
    the unbounded query yields a 12-entry sparkline starting at 1.0, and only
    the recency bound trims it back to the 10 live ones.
    """
    db_path = tmp_path / 'bounded-load.db'
    conn_sync = sqlite3.connect(str(db_path))
    conn_sync.executescript(LOAD_SAMPLES_SCHEMA)
    base = 10_000_000
    rows = [(base - 500_000, 'verify_concurrency', 1.0, None, None),
            (base - 400_000, 'verify_concurrency', 2.0, None, None)]
    # 10 live samples at the 5s tick, ending at `base`.
    rows += [(base - (9 - i) * 5, 'verify_concurrency', 100.0 + i, None, None)
             for i in range(10)]
    conn_sync.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        rows,
    )
    conn_sync.commit()
    conn_sync.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    sparkline = result['verify_concurrency']['sparkline']
    assert len(sparkline) == 10, 'ancient rows must not enter the sparkline'
    assert sparkline[0] == 100.0
    assert sparkline[-1] == 109.0
    assert result['verify_concurrency']['current'] == 109.0


@pytest.mark.asyncio
async def test_bound_is_anchored_to_newest_row_not_wall_clock(tmp_path: Path) -> None:
    """A stale DB (sampler down) still returns its last samples, not placeholders.

    A now()-relative bound would blank the card here; anchoring to MAX(ts)
    preserves the unbounded query's behaviour across a sampler outage.
    """
    db_path = tmp_path / 'stale-load.db'
    conn_sync = sqlite3.connect(str(db_path))
    conn_sync.executescript(LOAD_SAMPLES_SCHEMA)
    # ts values far in the past relative to any real wall clock.
    conn_sync.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max) VALUES (?, ?, ?, ?, ?)',
        [(100, 'occt_queue_depth', 1.0, None, None),
         (105, 'occt_queue_depth', 2.0, 1.5, 2.0)],
    )
    conn_sync.commit()
    conn_sync.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    assert result['occt_queue_depth']['current'] == 2.0
    assert result['occt_queue_depth']['sparkline'] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_a_group_that_stops_writing_blanks_while_its_siblings_keep_ticking(
    tmp_path: Path,
) -> None:
    """The PARTIAL degrade, which the whole-sampler-down test above does not cover.

    ``sampler/__main__.py`` degrades each collection group independently: the
    PSI group can hand run_tick ``{}`` every tick while the process and load
    groups keep writing.  The recency bound is anchored to a GLOBAL MAX(ts), so
    the still-writing groups advance the anchor and the stalled group's last
    rows fall outside the window.  Those cards then return the placeholder
    shape, where the unbounded query kept serving hour-old values.

    That is INTENDED, and it is why the anchor stays global.  /api/load is
    polled every 5 s and the frontend renders ``current`` as the live number,
    so a value last written over an hour ago is not a stale reading of the
    host's load — it is a reading of a collector that has stopped, and saying
    "no data" is the honest answer.  The whole-sampler-down case above is
    genuinely different: there the anchor moves with the data, so nothing is
    claimed to be fresher than anything else.
    """
    db_path = tmp_path / 'partial-degrade.db'
    conn_sync = sqlite3.connect(str(db_path))
    conn_sync.executescript(LOAD_SAMPLES_SCHEMA)
    base = 10_000_000
    rows = [
        # The load group kept ticking right up to `base`.
        (base - (9 - i) * 5, 'verify_concurrency', 100.0 + i, None, None)
        for i in range(10)
    ]
    # The PSI group stopped two hours ago — beyond the 1 h slack.
    rows += [(base - 7200 - (9 - i) * 5, 'psi_cpu_some_avg10', 5.0 + i, None, None)
             for i in range(10)]
    conn_sync.executemany(
        'INSERT INTO samples (ts, metric, value, window_mean, window_max)'
        ' VALUES (?, ?, ?, ?, ?)',
        rows,
    )
    conn_sync.commit()
    conn_sync.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        result = await get_load_metrics(conn)

    assert result['verify_concurrency']['current'] == 109.0
    assert result['psi_cpu_some_avg10'] == {
        'current': None, 'sparkline': [], 'window_mean': None, 'window_max': None,
    }, 'a collector stalled beyond the slack must read as no-data, not as live'
