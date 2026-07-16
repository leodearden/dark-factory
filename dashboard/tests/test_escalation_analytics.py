"""Tests for dashboard.data.escalation_analytics — escalation lifecycle analytics.

Backend data layer for plans/escalation-lifecycle-dashboard-prd.md Seam 2
(task gamma / 2658): archive aggregator (origin/lifespan/workflow blocks),
regime-markers loader, and the pure-sync `build_escalation_analytics` core.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dashboard.data.escalation_analytics import load_regime_markers

# ---------------------------------------------------------------------------
# Schema (mirrors RUNS_SCHEMA in test_performance.py — task_results is the
# only table _done_by_day reads).
# ---------------------------------------------------------------------------

RUNS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    prd_path       TEXT,
    started_at     TEXT NOT NULL,
    completed_at   TEXT,
    total_tasks    INTEGER DEFAULT 0,
    completed      INTEGER DEFAULT 0,
    blocked        INTEGER DEFAULT 0,
    escalated      INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    paused_for_cap INTEGER DEFAULT 0,
    cap_pause_secs REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS task_results (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    task_id              TEXT NOT NULL,
    project_id          TEXT NOT NULL,
    title               TEXT,
    outcome             TEXT NOT NULL,
    cost_usd            REAL DEFAULT 0.0,
    duration_ms         INTEGER DEFAULT 0,
    agent_invocations   INTEGER DEFAULT 0,
    execute_iterations  INTEGER DEFAULT 0,
    verify_attempts     INTEGER DEFAULT 0,
    review_cycles       INTEGER DEFAULT 0,
    steward_cost_usd    REAL DEFAULT 0.0,
    steward_invocations INTEGER DEFAULT 0,
    completed_at        TEXT,
    PRIMARY KEY (run_id, task_id)
);
"""


def _make_runs_db(tmp_path: Path, rows: list[tuple[str, str, str | None]]) -> Path:
    """Create a runs.db with task_results rows: (task_id, outcome, completed_at)."""
    db_path = tmp_path / 'runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(RUNS_SCHEMA)
    conn.execute(
        "INSERT INTO runs (run_id, project_id, started_at) VALUES ('run-001', 'dark_factory', ?)",
        (datetime.now(UTC).isoformat(),),
    )
    for task_id, outcome, completed_at in rows:
        conn.execute(
            'INSERT INTO task_results (run_id, task_id, project_id, outcome, completed_at) '
            "VALUES ('run-001', ?, 'dark_factory', ?, ?)",
            (task_id, outcome, completed_at),
        )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# step-1: load_regime_markers
# ---------------------------------------------------------------------------


class TestLoadRegimeMarkers:
    """load_regime_markers(path) -> (markers, parse_failures_delta). Never raises."""

    def test_default_path_parses_committed_seed_file(self):
        """The committed dashboard/regime-markers.yaml parses to exactly 3 markers."""
        markers, parse_failures_delta = load_regime_markers()

        assert parse_failures_delta == 0
        assert len(markers) == 3
        for m in markers:
            assert set(m) == {'date', 'label', 'tasks'}
            # Must be JSON-serializable: yaml.safe_load parses unquoted
            # YYYY-MM-DD as a datetime.date, which is NOT JSON-serializable —
            # the loader must normalize it to a str.
            assert isinstance(m['date'], str)
            assert isinstance(m['label'], str)
            assert isinstance(m['tasks'], list)

        all_tasks = sorted(t for m in markers for t in m['tasks'])
        assert all_tasks == [2593, 2630, 2631]

    def test_malformed_yaml_returns_empty_and_one_failure(self, tmp_path):
        """Unparseable YAML syntax -> ([], 1), never raises."""
        bad = tmp_path / 'bad.yaml'
        bad.write_text('date: [unclosed')

        markers, delta = load_regime_markers(bad)

        assert markers == []
        assert delta == 1

    def test_non_list_mapping_top_level_returns_empty_and_one_failure(self, tmp_path):
        """A top-level mapping (not a list) -> ([], 1)."""
        mapping_path = tmp_path / 'mapping.yaml'
        mapping_path.write_text('date: 2026-07-15\nlabel: not a list\n')

        markers, delta = load_regime_markers(mapping_path)

        assert markers == []
        assert delta == 1

    def test_non_list_scalar_top_level_returns_empty_and_one_failure(self, tmp_path):
        """A top-level scalar (not a list) -> ([], 1)."""
        scalar_path = tmp_path / 'scalar.yaml'
        scalar_path.write_text('just a string\n')

        markers, delta = load_regime_markers(scalar_path)

        assert markers == []
        assert delta == 1

    def test_missing_path_returns_empty_no_failure(self, tmp_path):
        """A missing file -> ([], 0) — not a parse failure, just absent."""
        missing = tmp_path / 'does-not-exist.yaml'

        markers, delta = load_regime_markers(missing)

        assert markers == []
        assert delta == 0


# ---------------------------------------------------------------------------
# step-3: _done_by_day
# ---------------------------------------------------------------------------


class TestDoneByDay:
    """_done_by_day(runs_db) -> {date: count of outcome='done' rows}."""

    def test_counts_only_done_bucketed_by_completed_at_date(self, tmp_path):
        from dashboard.data.escalation_analytics import _done_by_day

        db_path = _make_runs_db(tmp_path, [
            # Two 'done' rows on day 1
            ('t1', 'done', '2026-07-10T09:00:00+00:00'),
            ('t2', 'done', '2026-07-10T15:30:00+00:00'),
            # One 'done' row on day 2
            ('t3', 'done', '2026-07-11T08:00:00+00:00'),
            # Non-done outcomes on day 1 — must be excluded
            ('t4', 'blocked', '2026-07-10T10:00:00+00:00'),
            ('t5', 'cancelled', '2026-07-10T11:00:00+00:00'),
            # 'done' but NULL completed_at — must be excluded
            ('t6', 'done', None),
        ])

        result = _done_by_day(db_path)

        assert result == {'2026-07-10': 2, '2026-07-11': 1}

    def test_missing_db_returns_empty_dict(self, tmp_path):
        from dashboard.data.escalation_analytics import _done_by_day

        missing = tmp_path / 'does-not-exist' / 'runs.db'

        assert _done_by_day(missing) == {}
