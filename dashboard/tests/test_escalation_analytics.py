"""Tests for dashboard.data.escalation_analytics — escalation lifecycle analytics.

Backend data layer for plans/escalation-lifecycle-dashboard-prd.md Seam 2
(task gamma / 2658): archive aggregator (origin/lifespan/workflow blocks),
regime-markers loader, and the pure-sync `build_escalation_analytics` core.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Golden mini-archive fixture (PRD boundary row 6/7/10/11) — shared by the
# Origin (step-5), Lifespan (step-7), Workflow (step-9), 2555 forward-compat
# (step-11), and flow-cube consistency (step-13) tests. One corrupt file
# (parse_failures==1), a pending root record, two L1->L2 clusters, a churn
# pair on task 102, and a triaged_at record for row-10.
# ---------------------------------------------------------------------------


def _write_escalation(esc_dir: Path, esc: dict, *, archived: bool) -> None:
    """Write *esc* under esc_dir (root) or esc_dir/archive/<date> (archived).

    Mirrors test_performance.py's _write_archived_escalation for the archive
    case; pending (non-archived) records are written directly at the queue
    root, matching real EscalationQueue placement.
    """
    if archived:
        d = datetime.fromisoformat(esc['resolved_at']).date().isoformat()
        sub = esc_dir / 'archive' / d
    else:
        sub = esc_dir
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{esc['id']}.json").write_text(json.dumps(esc))


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def golden_now() -> datetime:
    """Fixed reference `now` for the golden archive (2026-07-16, today)."""
    return datetime(2026, 7, 16, 18, 0, 0, tzinfo=UTC)


def build_golden_archive(esc_dir: Path, now: datetime) -> dict:
    """Write the golden mini-archive under esc_dir; return bookkeeping for assertions.

    Returns a dict of the raw escalation dicts keyed by id, plus a few
    derived timestamps tests reference directly (so expected values are
    computed from the same deltas used to build the fixture, never
    hand-copied).
    """
    esc_101_1 = {
        'id': 'esc-101-1', 'task_id': '101', 'agent_role': 'implementer',
        'severity': 'blocking', 'category': 'cleanup_needed', 'summary': 'stale',
        'timestamp': _iso(now - timedelta(days=10)),
        'status': 'dismissed', 'level': 0,
        'resolved_at': _iso(now - timedelta(days=9)),
        'resolved_by': 'auto-dismissed',  # tier=reaper-sweep; unstamped -> benign inferred
    }
    esc_102_1 = {
        'id': 'esc-102-1', 'task_id': '102', 'agent_role': 'implementer',
        'severity': 'blocking', 'category': 'design_concern', 'summary': 'needs call',
        'timestamp': _iso(now - timedelta(days=8)),
        'status': 'resolved', 'level': 0,
        'resolved_at': _iso(now - timedelta(days=7)),
        'resolved_by': 'interactive',  # tier=human; unstamped -> actionable inferred
        'resolution_action': 'fix_forward',
        'triaged_at': _iso(now - timedelta(days=7, hours=-12)),  # row-10 field
        'triaged_by': 'escalation-watcher-auto',
    }
    # Churn pair: re-filings of task 102 relative to esc-102-1's resolved_at.
    esc_102_2 = {
        'id': 'esc-102-2', 'task_id': '102', 'agent_role': 'implementer',
        'severity': 'info', 'category': 'cleanup_needed', 'summary': 're-filed soon',
        'timestamp': _iso(now - timedelta(days=7) + timedelta(hours=12)),  # +12h -> churn
        'status': 'pending', 'level': 0,
    }
    esc_102_3 = {
        'id': 'esc-102-3', 'task_id': '102', 'agent_role': 'implementer',
        'severity': 'info', 'category': 'cleanup_needed', 'summary': 're-filed late',
        'timestamp': _iso(now - timedelta(days=7) + timedelta(hours=48)),  # +48h -> not churn
        'status': 'pending', 'level': 0,
    }
    esc_103_1 = {
        'id': 'esc-103-1', 'task_id': '103', 'agent_role': 'architect',
        'severity': 'blocking', 'category': 'risk_identified', 'summary': 'stamped benign',
        'timestamp': _iso(now - timedelta(days=6)),
        'status': 'resolved', 'level': 1,
        'resolved_at': _iso(now - timedelta(days=6) + timedelta(hours=2)),
        'resolved_by': 'escalation-watcher-auto',  # tier=auto-watcher
        'resolution_class': 'benign',  # STAMPED
        'triaged_at': _iso(now - timedelta(days=6) + timedelta(hours=1)),
        'triaged_by': 'escalation-watcher-auto',
    }
    # L1->L2 cluster #1: esc-104-0 (pending L1 member) under esc-104-1 (L2).
    esc_104_0 = {
        'id': 'esc-104-0', 'task_id': '104', 'agent_role': 'architect',
        'severity': 'blocking', 'category': 'design_concern', 'summary': 'member L1',
        'timestamp': _iso(now - timedelta(days=5, hours=2)),
        'status': 'pending', 'level': 1,
    }
    esc_104_1 = {
        'id': 'esc-104-1', 'task_id': '104', 'agent_role': 'architect',
        'severity': 'blocking', 'category': 'design_concern', 'summary': 'L2 cluster',
        'timestamp': _iso(now - timedelta(days=5)),
        'status': 'resolved', 'level': 2,
        'resolved_at': _iso(now - timedelta(days=5) + timedelta(hours=3)),
        'resolved_by': 'interactive',  # tier=human; unstamped -> actionable inferred
        'resolution_action': 'design_ruling',
        'members': ['esc-104-0'],
    }
    # L1->L2 cluster #2: esc-105-0 (pending L1 member) under esc-105-1 (L2).
    esc_105_0 = {
        'id': 'esc-105-0', 'task_id': '105', 'agent_role': 'implementer',
        'severity': 'blocking', 'category': 'design_concern', 'summary': 'member L1',
        'timestamp': _iso(now - timedelta(days=4, hours=1)),
        'status': 'pending', 'level': 1,
    }
    esc_105_1 = {
        'id': 'esc-105-1', 'task_id': '105', 'agent_role': 'implementer',
        'severity': 'blocking', 'category': 'design_concern', 'summary': 'L2 cluster 2',
        'timestamp': _iso(now - timedelta(days=4)),
        'status': 'resolved', 'level': 2,
        'resolved_at': _iso(now - timedelta(days=4) + timedelta(hours=1)),
        'resolved_by': 'interactive',
        'resolution_action': 'restart',
        'members': ['esc-105-0'],
    }
    # Pending root record — open_items / breach_6h (age = 7h > 6h).
    esc_106_1 = {
        'id': 'esc-106-1', 'task_id': '106', 'agent_role': 'implementer',
        'severity': 'info', 'category': 'cleanup_needed', 'summary': 'still open',
        'timestamp': _iso(now - timedelta(hours=7)),
        'status': 'pending', 'level': 0,
    }

    terminal = [esc_101_1, esc_102_1, esc_103_1, esc_104_1, esc_105_1]
    pending = [esc_102_2, esc_102_3, esc_104_0, esc_105_0, esc_106_1]
    for esc in terminal:
        _write_escalation(esc_dir, esc, archived=True)
    for esc in pending:
        _write_escalation(esc_dir, esc, archived=False)

    # One corrupt / non-JSON file at root -> parse_failures == 1.
    (esc_dir / 'esc-999-1.json').write_text('{not valid json')

    return {e['id']: e for e in terminal + pending}


# ---------------------------------------------------------------------------
# step-5: Origin block
# ---------------------------------------------------------------------------


class TestAggregateProjectOrigin:
    """_aggregate_project(...)['origin'] over the golden mini-archive."""

    def test_origin_block_and_parse_failures(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        build_golden_archive(esc_dir, now)

        entry, parse_failures = _aggregate_project(
            'dark_factory', esc_dir, tmp_path / 'runs.db', now=now,
        )

        assert parse_failures == 1
        origin = entry['origin']
        sources_by_name = {s['source']: s for s in origin['sources']}
        assert set(sources_by_name) == {'implementer', 'architect'}

        impl = sources_by_name['implementer']
        # filings: ALL implementer records (terminal + pending), regardless
        # of classification: 101-1, 102-1, 102-2, 102-3, 105-1, 106-1 = 6.
        assert impl['filings'] == 6
        # classified (terminal, valid times): 101-1 benign(inferred),
        # 102-1 actionable(inferred), 105-1 actionable(inferred).
        assert impl['benign'] == 1
        assert impl['actionable'] == 2
        assert impl['stamped_share'] == 0.0
        assert round(impl['benign_rate'], 4) == round(1 / 3, 4)
        # n=3 < 20 -> never predictably benign regardless of rate.
        assert impl['predictably_benign'] is False

        arch = sources_by_name['architect']
        # filings: 103-1, 104-1, 104-0 = 3.
        assert arch['filings'] == 3
        # classified: 103-1 benign(STAMPED), 104-1 actionable(inferred).
        assert arch['benign'] == 1
        assert arch['actionable'] == 1
        assert arch['stamped_share'] == 0.5
        assert arch['benign_rate'] == 0.5
        assert arch['predictably_benign'] is False

        # daily_by_source: date(timestamp) -> {source: n}. Spot-check the
        # date esc-101-1 was filed carries implementer=1.
        d101 = (now - timedelta(days=10)).date().isoformat()
        assert origin['daily_by_source'][d101]['implementer'] == 1

    def test_daily_spark_is_ascending_by_date(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        build_golden_archive(esc_dir, now)

        entry, _ = _aggregate_project('dark_factory', esc_dir, tmp_path / 'runs.db', now=now)
        origin = entry['origin']
        sources_by_name = {s['source']: s for s in origin['sources']}

        # daily_spark must be the source's daily_by_source counts, ordered
        # ascending by date — reconstruct the expected series independently
        # from daily_by_source (keyed by date -> {source: n}) and compare.
        for source, source_entry in sources_by_name.items():
            expected_dates = sorted(
                d for d, by_source in origin['daily_by_source'].items() if source in by_source
            )
            expected_spark = [origin['daily_by_source'][d][source] for d in expected_dates]
            assert source_entry['daily_spark'] == expected_spark
            # every filing must be reflected in the spark (no silent drops).
            assert sum(source_entry['daily_spark']) == source_entry['filings']


# ---------------------------------------------------------------------------
# step-5: predictably_benign boundary — trailing-28d (by resolved_at) window.
# Uses a dedicated fixture (not the shared golden archive) so the exact
# per-tier counts here aren't coupled to later steps' expectations.
# ---------------------------------------------------------------------------


class TestPredictablyBenign:
    """predictably_benign: benign_rate>0.9 AND n>=20 in a trailing-28d (by resolved_at) window."""

    def test_boundary_conditions(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'

        def _sweep(source: str, task_prefix: str, i: int) -> dict:
            """A dismissed, unstamped (-> benign inferred) reaper-sweep record."""
            return {
                'id': f'esc-{task_prefix}-{i}', 'task_id': f'{task_prefix}{i}',
                'agent_role': source,
                'severity': 'info', 'category': 'cleanup_needed', 'summary': 'stale sweep',
                'timestamp': _iso(now - timedelta(days=20, hours=i)),
                'status': 'dismissed', 'level': 0,
                'resolved_at': _iso(now - timedelta(days=10, hours=i)),
                'resolved_by': 'auto-dismissed',
            }

        def _human_actionable(source: str, task_prefix: str, i: int) -> dict:
            """A resolved, unstamped (-> actionable inferred) human-resolved record."""
            return {
                'id': f'esc-{task_prefix}-{i}', 'task_id': f'{task_prefix}{i}',
                'agent_role': source,
                'severity': 'blocking', 'category': 'design_concern', 'summary': 'needs call',
                'timestamp': _iso(now - timedelta(days=20, hours=i)),
                'status': 'resolved', 'level': 0,
                'resolved_at': _iso(now - timedelta(days=10, hours=i)),
                'resolved_by': 'interactive',
            }

        # (a) n=25, rate=1.0 -> True.
        for i in range(25):
            _write_escalation(esc_dir, _sweep('high-n-high-rate', 'a9', i), archived=True)
        # (b) n=19, rate=1.0 -> False (n below 20 despite a perfect rate).
        for i in range(19):
            _write_escalation(esc_dir, _sweep('low-n', 'b9', i), archived=True)
        # (c) n=25, rate=0.8 (20 benign + 5 actionable) -> False (rate <= 0.9 despite n>=20).
        for i in range(20):
            _write_escalation(esc_dir, _sweep('high-n-low-rate', 'c9', i), archived=True)
        for i in range(5):
            _write_escalation(esc_dir, _human_actionable('high-n-low-rate', 'c8', i), archived=True)

        entry, _ = _aggregate_project('dark_factory', esc_dir, tmp_path / 'runs.db', now=now)
        sources_by_name = {s['source']: s for s in entry['origin']['sources']}

        high = sources_by_name['high-n-high-rate']
        assert high['benign'] == 25
        assert high['predictably_benign'] is True

        low_n = sources_by_name['low-n']
        assert low_n['benign'] == 19
        assert low_n['predictably_benign'] is False

        low_rate = sources_by_name['high-n-low-rate']
        assert low_rate['benign'] == 20
        assert low_rate['actionable'] == 5
        assert low_rate['predictably_benign'] is False
