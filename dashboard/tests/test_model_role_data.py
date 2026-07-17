"""Tests for dashboard.data.model_role — per-(model×role) outcome rollup (task 2534 δ).

Dashboard-side mirror of ``orchestrator.digest.model_role_rollup``. Seeds the
SAME fixture shape as orchestrator/tests/test_digest.py's
``TestModelRoleRollupCore`` / ``TestModelRoleRollupTurnCapSaturation`` (with
timestamps placed relative to a fixed ``NOW`` reference, since this module's
window is a rolling ``days``-lookback cutoff rather than digest's explicit
``[window_start, window_end]`` bounds) and asserts numerically-equal
results — a shared-contract cross-check that both sides compute the
identical GROUP-BY-(model, role) semantics from the same runs.db shape.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import aiosqlite
import pytest

from dashboard.data.model_role import aggregate_model_role_rollup, get_model_role_rollup

# ---------------------------------------------------------------------------
# Schema — the three tables model_role.py reads from runs.db (mirrors the
# local COSTS_SCHEMA convention in test_costs_data.py / test_merge_queue_data.py
# rather than importing production DDL from CostStore/EventStore directly).
# ---------------------------------------------------------------------------

MODEL_ROLE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS task_results (
    run_id       TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    project_id   TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY (run_id, task_id)
);

CREATE TABLE IF NOT EXISTS invocations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    task_id      TEXT,
    project_id   TEXT NOT NULL,
    account_name TEXT NOT NULL,
    model        TEXT NOT NULL,
    role         TEXT NOT NULL,
    cost_usd     REAL NOT NULL DEFAULT 0.0,
    capped       INTEGER NOT NULL DEFAULT 0,
    started_at   TEXT NOT NULL,
    completed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inv_completed ON invocations(completed_at);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,
    run_id     TEXT NOT NULL,
    task_id    TEXT,
    event_type TEXT NOT NULL,
    role       TEXT,
    data       TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""

# Fixed reference "now" + a 1-day lookback window so fixture timestamps can be
# placed deterministically inside/outside the window, mirroring the fixed
# window bounds used by test_digest.py's rollup fixtures.
NOW = datetime(2026, 5, 10, 20, 0, 0, tzinfo=UTC)
DAYS = 1


def _seed_core_and_saturation_fixture(conn: sqlite3.Connection, run_id: str = 'run-test') -> None:
    """Same shape as test_digest.py's TestModelRoleRollupCore /
    TestModelRoleRollupTurnCapSaturation fixtures — see that module for the
    per-row rationale. Timestamps are the same 2026-05-10 values; the
    out-of-window row (2026-05-09T00:00:00) falls before NOW - DAYS
    (2026-05-09T20:00:00), matching digest's out-of-window exclusion.
    """
    conn.executemany(
        'INSERT INTO task_results (run_id, task_id, project_id, outcome) VALUES (?, ?, ?, ?)',
        [
            (run_id, 't1', 'p', 'done'),
            (run_id, 't2', 'p', 'blocked'),
            (run_id, 't3', 'p', 'done'),
        ],
    )

    def inv(task_id: str | None, model: str, role: str, cost: float, capped: bool, completed_at: str) -> None:
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (run_id, task_id, 'p', 'a', model, role, cost, int(capped),
             completed_at, completed_at),
        )

    # (sonnet, implementer): 2 invocations across 2 tasks — t1 done, t2
    # blocked; the t2 invocation is capped.
    inv('t1', 'sonnet', 'implementer', 2.0, False, '2026-05-10T08:00:00+00:00')
    inv('t2', 'sonnet', 'implementer', 3.0, True, '2026-05-10T09:00:00+00:00')
    # (haiku, implementer): 1 done invocation.
    inv('t3', 'haiku', 'implementer', 1.0, False, '2026-05-10T10:00:00+00:00')
    # steward + triage — task t4 has no task_results row, so both LEFT JOIN
    # to a NULL outcome.
    inv('t4', 'opus', 'steward', 0.5, False, '2026-05-10T11:00:00+00:00')
    inv('t4', 'opus', 'triage', 0.2, False, '2026-05-10T11:05:00+00:00')
    # module_tagger row: task_id=NULL.
    inv(None, 'haiku', 'module_tagger', 0.1, False, '2026-05-10T12:00:00+00:00')
    # Out-of-window row — must be excluded from every aggregate.
    inv('t5', 'sonnet', 'implementer', 99.0, False, '2026-05-09T00:00:00+00:00')

    def evt(task_id: str | None, event_type: str, role: str, data: dict, timestamp: str) -> None:
        conn.execute(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (timestamp, run_id, task_id, event_type, role, json.dumps(data)),
        )

    # simple_task: max_turns=30, invocation_end turns=30 and turns=12 -> 1/2 = 0.5
    evt('t1', 'routing_decision', 'simple_task', {'max_turns': 30}, '2026-05-10T08:00:00+00:00')
    evt('t1', 'invocation_end', 'simple_task', {'turns': 30}, '2026-05-10T08:01:00+00:00')
    evt('t2', 'invocation_end', 'simple_task', {'turns': 12}, '2026-05-10T08:02:00+00:00')
    # architect: max_turns=60, invocation_end turns=20 -> 0/1 = 0.0
    evt('t3', 'routing_decision', 'architect', {'max_turns': 60}, '2026-05-10T09:00:00+00:00')
    evt('t3', 'invocation_end', 'architect', {'turns': 20}, '2026-05-10T09:01:00+00:00')
    # no_max_turns_role: invocation_end rows but no routing_decision -> None
    evt('t4', 'invocation_end', 'no_max_turns_role', {'turns': 5}, '2026-05-10T10:00:00+00:00')
    # Out-of-window invocation_end — must be excluded.
    evt('t5', 'invocation_end', 'simple_task', {'turns': 999}, '2026-05-09T00:00:00+00:00')


@pytest.fixture()
def model_role_db(tmp_path):
    """Populated runs.db — same shape as the digest-side rollup fixture."""
    db_path = tmp_path / 'runs.db'
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(MODEL_ROLE_SCHEMA)
        _seed_core_and_saturation_fixture(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture()
async def model_role_conn(model_role_db):
    async with aiosqlite.connect(str(model_role_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


class TestGetModelRoleRollup:
    """Tests for get_model_role_rollup(db, days, now) — core rollup + saturation (step-11)."""

    @pytest.mark.asyncio
    async def test_rates_and_cost_per_done_match_digest_fixture(self, model_role_conn) -> None:
        """Numerically equal to test_digest.py's TestModelRoleRollupCore
        assertions given the same fixture shape (shared-contract cross-check)."""
        result = await get_model_role_rollup(model_role_conn, days=DAYS, now=NOW)
        by_key = {(row['model'], row['role']): row for row in result['rows']}

        sonnet_impl = by_key[('sonnet', 'implementer')]
        assert sonnet_impl['invocation_count'] == 2
        assert sonnet_impl['done_count'] == 1
        assert sonnet_impl['blocked_count'] == 1
        assert sonnet_impl['done_rate'] == pytest.approx(0.5)
        assert sonnet_impl['blocked_rate'] == pytest.approx(0.5)
        assert sonnet_impl['capped_count'] == 1
        assert sonnet_impl['cap_hit_rate'] == pytest.approx(0.5)
        # Out-of-window 99.0 row excluded — total is 2.0 + 3.0, not + 99.0.
        assert sonnet_impl['total_cost_usd'] == pytest.approx(5.0)
        assert sonnet_impl['cost_per_done'] == pytest.approx(5.0)  # 5.0 / 1 distinct done task

        haiku_impl = by_key[('haiku', 'implementer')]
        assert haiku_impl['invocation_count'] == 1
        assert haiku_impl['done_count'] == 1
        assert haiku_impl['cost_per_done'] == pytest.approx(1.0)

        steward_row = by_key[('opus', 'steward')]
        assert steward_row['invocation_count'] == 1
        assert steward_row['done_count'] == 0
        assert steward_row['blocked_count'] == 0
        assert steward_row['cost_per_done'] is None

        triage_row = by_key[('opus', 'triage')]
        assert triage_row['invocation_count'] == 1
        assert triage_row['done_count'] == 0

        module_tagger_row = by_key[('haiku', 'module_tagger')]
        assert module_tagger_row['invocation_count'] == 1
        assert module_tagger_row['done_count'] == 0
        assert module_tagger_row['blocked_count'] == 0
        assert module_tagger_row['cost_per_done'] is None

    @pytest.mark.asyncio
    async def test_turn_cap_saturation_matches_digest_fixture(self, model_role_conn) -> None:
        """Numerically equal to test_digest.py's TestModelRoleRollupTurnCapSaturation."""
        result = await get_model_role_rollup(model_role_conn, days=DAYS, now=NOW)
        saturation = result['turn_cap_saturation']

        assert saturation['simple_task'] == pytest.approx(0.5)
        assert saturation['architect'] == pytest.approx(0.0)
        assert saturation['no_max_turns_role'] is None

    @pytest.mark.asyncio
    async def test_none_db_returns_empty_rollup(self) -> None:
        """Fail-open: db=None returns the empty rollup shape, never raises."""
        result = await get_model_role_rollup(None, days=DAYS, now=NOW)
        assert result == {'rows': [], 'turn_cap_saturation': {}}


# ---------------------------------------------------------------------------
# aggregate_model_role_rollup — merges raw counts across DBs and recomputes
# rates from the merged totals (never averages already-divided per-db rates).
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_model_role_dbs(tmp_path):
    """Two DBs whose per-db rates/saturation ratios differ from the correctly
    merged (sum-then-divide) values — proves aggregate_model_role_rollup
    recomputes from merged totals rather than averaging per-db ratios."""
    db_a = tmp_path / 'a.db'
    conn = sqlite3.connect(str(db_a))
    try:
        conn.executescript(MODEL_ROLE_SCHEMA)
        conn.executemany(
            'INSERT INTO task_results (run_id, task_id, project_id, outcome) VALUES (?, ?, ?, ?)',
            [('run-a', 'ta1', 'p', 'done'), ('run-a', 'ta2', 'p', 'done')],
        )
        conn.executemany(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                # ta1, ta2 done -> 2 done invocations.
                ('run-a', 'ta1', 'p', 'a', 'sonnet', 'implementer', 1.0, 0,
                 '2026-05-10T08:00:00+00:00', '2026-05-10T08:00:00+00:00'),
                ('run-a', 'ta2', 'p', 'a', 'sonnet', 'implementer', 1.0, 0,
                 '2026-05-10T08:10:00+00:00', '2026-05-10T08:10:00+00:00'),
                # ta3 has no task_results row -> neither done nor blocked; capped.
                ('run-a', 'ta3', 'p', 'a', 'sonnet', 'implementer', 1.0, 1,
                 '2026-05-10T08:20:00+00:00', '2026-05-10T08:20:00+00:00'),
            ],
        )
        conn.execute(
            "INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) "
            "VALUES (?, 'run-a', 'ta1', 'routing_decision', 'simple_task', ?)",
            ('2026-05-10T08:00:00+00:00', json.dumps({'max_turns': 10})),
        )
        conn.executemany(
            "INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) "
            "VALUES (?, 'run-a', 'ta1', 'invocation_end', 'simple_task', ?)",
            [
                ('2026-05-10T08:01:00+00:00', json.dumps({'turns': 10})),  # hit
                ('2026-05-10T08:02:00+00:00', json.dumps({'turns': 2})),   # miss
            ],
        )
        conn.commit()
    finally:
        conn.close()
    # DB A alone: done_rate = 2/3 = 0.667, cap_hit_rate = 1/3 = 0.333,
    # turn_cap_saturation['simple_task'] = 1/2 = 0.5.

    db_b = tmp_path / 'b.db'
    conn = sqlite3.connect(str(db_b))
    try:
        conn.executescript(MODEL_ROLE_SCHEMA)
        # tb1 has no task_results row -> neither done nor blocked.
        conn.execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            ('run-b', 'tb1', 'p', 'a', 'sonnet', 'implementer', 5.0, 0,
             '2026-05-10T09:00:00+00:00', '2026-05-10T09:00:00+00:00'),
        )
        conn.execute(
            "INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) "
            "VALUES (?, 'run-b', 'tb1', 'routing_decision', 'simple_task', ?)",
            ('2026-05-10T09:00:00+00:00', json.dumps({'max_turns': 10})),
        )
        conn.executemany(
            "INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) "
            "VALUES (?, 'run-b', 'tb1', 'invocation_end', 'simple_task', ?)",
            [
                ('2026-05-10T09:01:00+00:00', json.dumps({'turns': 10})),  # hit
                ('2026-05-10T09:02:00+00:00', json.dumps({'turns': 10})),  # hit
                ('2026-05-10T09:03:00+00:00', json.dumps({'turns': 10})),  # hit
                ('2026-05-10T09:04:00+00:00', json.dumps({'turns': 2})),   # miss
            ],
        )
        conn.commit()
    finally:
        conn.close()
    # DB B alone: done_rate = 0/1 = 0.0, cap_hit_rate = 0/1 = 0.0,
    # turn_cap_saturation['simple_task'] = 3/4 = 0.75.

    return db_a, db_b


@pytest.fixture()
async def two_model_role_conns(two_model_role_dbs):
    db_a, db_b = two_model_role_dbs
    async with aiosqlite.connect(str(db_a)) as a, aiosqlite.connect(str(db_b)) as b:
        a.row_factory = aiosqlite.Row
        b.row_factory = aiosqlite.Row
        yield [a, b]


class TestAggregateModelRoleRollup:
    """Tests for aggregate_model_role_rollup(dbs, days, now) (step-11)."""

    @pytest.mark.asyncio
    async def test_sums_counts_and_recomputes_rates_from_merged_totals(
        self, two_model_role_conns,
    ) -> None:
        result = await aggregate_model_role_rollup(two_model_role_conns, days=DAYS, now=NOW)
        by_key = {(row['model'], row['role']): row for row in result['rows']}
        cell = by_key[('sonnet', 'implementer')]

        assert cell['invocation_count'] == 4  # 3 (DB A) + 1 (DB B)
        assert cell['done_count'] == 2
        assert cell['blocked_count'] == 0
        # Correct merge: (2 done)/(4 total) = 0.5 — NOT the naive average of
        # per-db rates ((2/3 + 0/1) / 2 = 0.333).
        assert cell['done_rate'] == pytest.approx(0.5)
        assert cell['capped_count'] == 1
        # Correct merge: 1/4 = 0.25 — NOT naive average ((1/3 + 0/1) / 2 = 0.167).
        assert cell['cap_hit_rate'] == pytest.approx(0.25)
        assert cell['total_cost_usd'] == pytest.approx(8.0)  # 3.0 (A) + 5.0 (B)
        assert cell['cost_per_done'] == pytest.approx(4.0)  # 8.0 / 2 distinct done tasks

    @pytest.mark.asyncio
    async def test_turn_cap_saturation_sums_raw_hits_across_dbs(
        self, two_model_role_conns,
    ) -> None:
        result = await aggregate_model_role_rollup(two_model_role_conns, days=DAYS, now=NOW)
        # Correct merge: (1 + 3) hits / (2 + 4) total = 4/6 = 0.667 — NOT the
        # naive average of per-db ratios ((0.5 + 0.75) / 2 = 0.625).
        assert result['turn_cap_saturation']['simple_task'] == pytest.approx(4 / 6)

    @pytest.mark.asyncio
    async def test_turn_cap_saturation_excludes_db_lacking_max_turns_from_denominator(
        self, tmp_path,
    ) -> None:
        """Regression test (task 2534 amendment): a DB whose invocation_end
        rows for a role have NO routing_decision max_turns in-window must not
        dilute the merged denominator with zero-credit rows, even though
        another DB does have that role's cap in-window.

        DB A: simple_task max_turns=10 known, 2/2 invocation_end rows hit ->
        1.0 on its own. DB B: simple_task invocation_end rows present but NO
        routing_decision for simple_task at all -> has_max_turns=False, so
        its 3 rows carry no valid pass/fail signal. The buggy merge (summing
        every DB's raw total regardless of has_max_turns) would compute
        2 hits / (2 + 3) total = 0.4; the correct merge excludes DB B's
        orphan rows entirely, leaving 2/2 = 1.0.
        """
        db_a = tmp_path / 'sat_a.db'
        conn = sqlite3.connect(str(db_a))
        try:
            conn.executescript(MODEL_ROLE_SCHEMA)
            conn.execute(
                "INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) "
                "VALUES (?, 'run-a', 'ta1', 'routing_decision', 'simple_task', ?)",
                ('2026-05-10T08:00:00+00:00', json.dumps({'max_turns': 10})),
            )
            conn.executemany(
                "INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) "
                "VALUES (?, 'run-a', 'ta1', 'invocation_end', 'simple_task', ?)",
                [
                    ('2026-05-10T08:01:00+00:00', json.dumps({'turns': 10})),  # hit
                    ('2026-05-10T08:02:00+00:00', json.dumps({'turns': 10})),  # hit
                ],
            )
            conn.commit()
        finally:
            conn.close()

        db_b = tmp_path / 'sat_b.db'
        conn = sqlite3.connect(str(db_b))
        try:
            conn.executescript(MODEL_ROLE_SCHEMA)
            # No routing_decision event for simple_task in this DB at all —
            # these invocation_end rows are orphans with no known cap.
            conn.executemany(
                "INSERT INTO events (timestamp, run_id, task_id, event_type, role, data) "
                "VALUES (?, 'run-b', 'tb1', 'invocation_end', 'simple_task', ?)",
                [
                    ('2026-05-10T09:01:00+00:00', json.dumps({'turns': 1})),
                    ('2026-05-10T09:02:00+00:00', json.dumps({'turns': 1})),
                    ('2026-05-10T09:03:00+00:00', json.dumps({'turns': 1})),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        async with aiosqlite.connect(str(db_a)) as a, aiosqlite.connect(str(db_b)) as b:
            a.row_factory = aiosqlite.Row
            b.row_factory = aiosqlite.Row
            result = await aggregate_model_role_rollup([a, b], days=DAYS, now=NOW)

        assert result['turn_cap_saturation']['simple_task'] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_none_db_is_skipped(self, two_model_role_conns) -> None:
        with_none = await aggregate_model_role_rollup(
            [*two_model_role_conns, None], days=DAYS, now=NOW,
        )
        without_none = await aggregate_model_role_rollup(
            two_model_role_conns, days=DAYS, now=NOW,
        )
        assert with_none == without_none

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_rollup(self) -> None:
        result = await aggregate_model_role_rollup([], days=DAYS, now=NOW)
        assert result == {'rows': [], 'turn_cap_saturation': {}}
