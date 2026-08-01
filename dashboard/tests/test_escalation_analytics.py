"""Tests for dashboard.data.escalation_analytics — escalation lifecycle analytics.

Backend data layer for plans/escalation-lifecycle-dashboard-prd.md Seam 2
(task gamma / 2658): archive aggregator (origin/lifespan/workflow blocks),
regime-markers loader, and the pure-sync `build_escalation_analytics` core.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
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


def _isoweek_key(dt: datetime) -> str:
    """'YYYY-Www' ISO-week key, matching the implementation's tier_weekly bucketing."""
    iso = dt.isocalendar()
    return f'{iso.year}-W{iso.week:02d}'


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
        # row-10 field: 18h after filing (timestamp=-8d), 6h before
        # resolution (resolved_at=-7d) -> filed_to_triaged=64800s,
        # triaged_to_resolved=21600s.
        'triaged_at': _iso(now - timedelta(days=7, hours=6)),
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
        # of classification: 101-1, 102-1, 102-2, 102-3, 105-0, 105-1, 106-1 = 7.
        assert impl['filings'] == 7
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


# ---------------------------------------------------------------------------
# step-7: Lifespan block
# ---------------------------------------------------------------------------


class TestAggregateProjectLifespan:
    """_aggregate_project(...)['lifespan'] over the golden mini-archive."""

    def test_percentiles_samples_and_promotion(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        build_golden_archive(esc_dir, now)

        entry, _ = _aggregate_project('dark_factory', esc_dir, tmp_path / 'runs.db', now=now)
        lifespan = entry['lifespan']

        # percentiles_by_level: level 0 -> [101-1, 102-1] both 1 day (86400s);
        # level 1 -> [103-1] alone at 2h (7200s); level 2 -> [104-1 @ 3h=10800s,
        # 105-1 @ 1h=3600s].
        by_level = lifespan['percentiles_by_level']
        assert by_level['0']['p50'] == 86400.0
        assert by_level['0']['p90'] == 86400.0
        assert by_level['1']['p50'] == 7200.0
        assert by_level['1']['p90'] == 7200.0
        assert by_level['2']['p50'] == 7200.0
        assert by_level['2']['p90'] == 10080.0

        # samples: one [date, tier, level, secs] row per terminal-with-valid
        # -times record, dated by resolved_at (matches flow_daily's key so
        # row-11's marginal-reconciliation can hold in a later step).
        samples = {tuple(row) for row in lifespan['samples']}
        assert samples == {
            ((now - timedelta(days=9)).date().isoformat(), 'reaper-sweep', 0, 86400.0),
            ((now - timedelta(days=7)).date().isoformat(), 'human', 0, 86400.0),
            (
                (now - timedelta(days=6) + timedelta(hours=2)).date().isoformat(),
                'auto-watcher', 1, 7200.0,
            ),
            (
                (now - timedelta(days=5) + timedelta(hours=3)).date().isoformat(),
                'human', 2, 10800.0,
            ),
            (
                (now - timedelta(days=4) + timedelta(hours=1)).date().isoformat(),
                'human', 2, 3600.0,
            ),
        }

        # open_items: the 5 pending records, all aged past the 6h breach
        # threshold in this fixture (breach_6h False is covered by a
        # dedicated fixture below — the golden archive has no young item).
        open_by_id = {item['id']: item for item in lifespan['open_items']}
        assert set(open_by_id) == {
            'esc-102-2', 'esc-102-3', 'esc-104-0', 'esc-105-0', 'esc-106-1',
        }
        assert all(item['breach_6h'] for item in open_by_id.values())
        root_item = open_by_id['esc-106-1']
        assert root_item['task_id'] == '106'
        assert root_item['level'] == 0
        assert root_item['age_secs'] == timedelta(hours=7).total_seconds()

        # l1_to_l2_promotion: two L2 clusters, each with one resolvable
        # member -> deltas [7200 (104), 3600 (105)] sorted [3600, 7200].
        promo = lifespan['l1_to_l2_promotion']
        assert promo['count'] == 2
        assert promo['p50_secs'] == 5400.0
        assert promo['p90_secs'] == 6840.0

        # Open question #7: no L0->L1 timing sub-metric (un-derivable; see
        # design_decisions).
        assert 'l0_to_l1' not in lifespan

    def test_promotion_zero_when_member_missing_and_young_item_not_breached(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'

        # L2 cluster whose only member id isn't present anywhere in the
        # archive -> l1_to_l2_promotion must stay count=0 (no crash).
        orphan_l2 = {
            'id': 'esc-200-1', 'task_id': '200', 'agent_role': 'architect',
            'severity': 'blocking', 'category': 'design_concern', 'summary': 'orphaned cluster',
            'timestamp': _iso(now - timedelta(days=1)),
            'status': 'resolved', 'level': 2,
            'resolved_at': _iso(now - timedelta(days=1) + timedelta(hours=1)),
            'resolved_by': 'interactive',
            'members': ['esc-does-not-exist'],
        }
        _write_escalation(esc_dir, orphan_l2, archived=True)

        # A fresh pending item aged 3h (<=6h) -> breach_6h False.
        young_item = {
            'id': 'esc-201-1', 'task_id': '201', 'agent_role': 'architect',
            'severity': 'info', 'category': 'cleanup_needed', 'summary': 'just filed',
            'timestamp': _iso(now - timedelta(hours=3)),
            'status': 'pending', 'level': 0,
        }
        _write_escalation(esc_dir, young_item, archived=False)

        entry, _ = _aggregate_project('dark_factory', esc_dir, tmp_path / 'runs.db', now=now)
        lifespan = entry['lifespan']

        promo = lifespan['l1_to_l2_promotion']
        assert promo == {'count': 0, 'p50_secs': None, 'p90_secs': None}

        open_by_id = {item['id']: item for item in lifespan['open_items']}
        assert open_by_id['esc-201-1']['breach_6h'] is False
        assert open_by_id['esc-201-1']['age_secs'] == timedelta(hours=3).total_seconds()


# ---------------------------------------------------------------------------
# step-9: Workflow block
# ---------------------------------------------------------------------------


class TestAggregateProjectWorkflow:
    """_aggregate_project(...)['workflow'] over the golden mini-archive."""

    def test_workflow_block(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        archive = build_golden_archive(esc_dir, now)

        # --- tier_weekly: independently regroup the 5 terminal records' known
        # (resolved_at, tier) pairs by ISO week, rather than hand-computing
        # week numbers.
        expected_tier_by_resolved = [
            (now - timedelta(days=9), 'reaper-sweep'),                        # 101-1
            (now - timedelta(days=7), 'human'),                               # 102-1
            (now - timedelta(days=6) + timedelta(hours=2), 'auto-watcher'),   # 103-1
            (now - timedelta(days=5) + timedelta(hours=3), 'human'),          # 104-1
            (now - timedelta(days=4) + timedelta(hours=1), 'human'),          # 105-1
        ]
        expected_tier_weekly: dict[str, dict[str, int]] = {}
        for dt, tier in expected_tier_by_resolved:
            wk = expected_tier_weekly.setdefault(_isoweek_key(dt), {})
            wk[tier] = wk.get(tier, 0) + 1

        # --- esc_per_done_daily fixture: pick the earliest/latest filing
        # dates from the archive itself (never hand-copied) plus a
        # done-only date the archive has no filings on.
        expected_filings_by_date: dict[str, int] = {}
        for esc in archive.values():
            d = datetime.fromisoformat(esc['timestamp']).date().isoformat()
            expected_filings_by_date[d] = expected_filings_by_date.get(d, 0) + 1
        dated_days = sorted(expected_filings_by_date)
        zero_done_day = dated_days[0]
        two_done_day = dated_days[-1]
        assert zero_done_day != two_done_day  # fixture must span >=2 distinct filing dates
        done_only_day = '2026-01-01'  # well outside the archive's ~10-day span

        runs_db = _make_runs_db(tmp_path, [
            ('t1', 'done', f'{two_done_day}T09:00:00+00:00'),
            ('t2', 'done', f'{two_done_day}T10:00:00+00:00'),
            ('t3', 'done', f'{done_only_day}T09:00:00+00:00'),
        ])

        entry, _ = _aggregate_project('dark_factory', esc_dir, runs_db, now=now)
        workflow = entry['workflow']

        assert workflow['tier_weekly'] == expected_tier_weekly

        # action_mix: terminal resolution_action, None -> 'unspecified'.
        # 101-1 & 103-1 carry no resolution_action key -> unspecified;
        # 102-1 -> fix_forward; 104-1 -> design_ruling; 105-1 -> restart.
        assert workflow['action_mix'] == {
            'unspecified': 2,
            'fix_forward': 1,
            'design_ruling': 1,
            'restart': 1,
        }

        # churn_daily: only esc-102-2 (task 102, +12h after 102-1's
        # resolved_at) counts; esc-102-3 (+48h) falls outside the 24h
        # lookback window.
        churn_date = (now - timedelta(days=7) + timedelta(hours=12)).date().isoformat()
        assert workflow['churn_daily'] == {churn_date: 1}

        # esc_per_done_daily: union of filing dates and done-by-day dates.
        by_date = {row['date']: row for row in workflow['esc_per_done_daily']}
        assert by_date[zero_done_day]['filings'] == expected_filings_by_date[zero_done_day]
        assert by_date[zero_done_day]['done'] == 0
        assert by_date[zero_done_day]['ratio'] is None

        assert by_date[two_done_day]['filings'] == expected_filings_by_date[two_done_day]
        assert by_date[two_done_day]['done'] == 2
        assert by_date[two_done_day]['ratio'] == expected_filings_by_date[two_done_day] / 2

        assert by_date[done_only_day]['filings'] == 0
        assert by_date[done_only_day]['done'] == 1
        assert by_date[done_only_day]['ratio'] == 0.0

        # flow_daily: sparse cube over the 5 terminal-with-valid-times
        # records, keyed by (date(resolved_at), source, level, tier, class).
        # In this fixture every record lands on a distinct date so each
        # cell's n == 1.
        flow_cells = {
            (row['date'], row['source'], row['level'], row['tier'], row['class'])
            for row in workflow['flow_daily']
        }
        assert flow_cells == {
            ((now - timedelta(days=9)).date().isoformat(), 'implementer', 0, 'reaper-sweep', 'benign'),
            ((now - timedelta(days=7)).date().isoformat(), 'implementer', 0, 'human', 'actionable'),
            (
                (now - timedelta(days=6) + timedelta(hours=2)).date().isoformat(),
                'architect', 1, 'auto-watcher', 'benign',
            ),
            (
                (now - timedelta(days=5) + timedelta(hours=3)).date().isoformat(),
                'architect', 2, 'human', 'actionable',
            ),
            (
                (now - timedelta(days=4) + timedelta(hours=1)).date().isoformat(),
                'implementer', 2, 'human', 'actionable',
            ),
        }
        assert all(row['n'] == 1 for row in workflow['flow_daily'])
        assert sum(row['n'] for row in workflow['flow_daily']) == len(entry['lifespan']['samples'])


# ---------------------------------------------------------------------------
# step-11: triage_segments — 2555 forward-compat (row 10)
# ---------------------------------------------------------------------------


def _percentile_2(vals: list[float], p: float) -> float:
    """p50/p90 of an exactly-2-element list, mirroring stats_utils.percentile."""
    lo, hi = sorted(vals)
    frac = p / 100.0
    return lo * (1 - frac) + hi * frac


class TestTriageSegments:
    """lifespan['triage_segments'] — render-when-present over triaged_at."""

    def test_present_and_correct_over_golden_archive(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        build_golden_archive(esc_dir, now)

        entry, _ = _aggregate_project('dark_factory', esc_dir, tmp_path / 'runs.db', now=now)
        lifespan = entry['lifespan']

        # Of the 5 terminal records, only esc-102-1 and esc-103-1 carry
        # triaged_at; esc-101-1/104-1/105-1 don't -> excluded.
        # esc-103-1: filed_to_triaged=1h, triaged_to_resolved=1h.
        # esc-102-1: filed_to_triaged=18h, triaged_to_resolved=6h.
        filed_to_triaged = [timedelta(hours=1).total_seconds(), timedelta(hours=18).total_seconds()]
        triaged_to_resolved = [timedelta(hours=1).total_seconds(), timedelta(hours=6).total_seconds()]

        assert 'triage_segments' in lifespan
        segments = lifespan['triage_segments']
        assert segments['count'] == 2
        assert segments['filed_to_triaged']['p50'] == _percentile_2(filed_to_triaged, 50)
        assert segments['filed_to_triaged']['p90'] == _percentile_2(filed_to_triaged, 90)
        assert segments['triaged_to_resolved']['p50'] == _percentile_2(triaged_to_resolved, 50)
        assert segments['triaged_to_resolved']['p90'] == _percentile_2(triaged_to_resolved, 90)

    def test_absent_when_no_record_has_triaged_at(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'

        esc = {
            'id': 'esc-300-1', 'task_id': '300', 'agent_role': 'implementer',
            'severity': 'blocking', 'category': 'cleanup_needed', 'summary': 'no triage stamp',
            'timestamp': _iso(now - timedelta(days=2)),
            'status': 'resolved', 'level': 0,
            'resolved_at': _iso(now - timedelta(days=1)),
            'resolved_by': 'interactive',
        }
        _write_escalation(esc_dir, esc, archived=True)

        entry, _ = _aggregate_project('dark_factory', esc_dir, tmp_path / 'runs.db', now=now)

        assert 'triage_segments' not in entry['lifespan']


# ---------------------------------------------------------------------------
# step-13: row 11 — flow-cube marginal-reconciliation regression pin
# ---------------------------------------------------------------------------


class TestFlowCubeConsistency:
    """flow_daily reconciles against lifespan.samples, tier_weekly, and sources[].

    PRD boundary row 11: sum(flow_daily.n) == len(lifespan.samples) ==
    resolved(terminal-with-valid-times)-record count; grouped by tier it
    matches tier_weekly's per-tier totals; grouped by (source, class) it
    matches sources[] benign/actionable counts. flow_daily and samples share
    an identical per-record date key (date(resolved_at)) over an identical
    population (both gate on parseable timestamp AND resolved_at), so their
    per-date marginals match exactly — which means summing over ANY date
    sub-window preserves the identity, not just the full aggregate. This is
    a regression pin over the step-6/8/10 implementation, not new behavior.
    """

    def test_flow_daily_reconciles_with_samples_tier_weekly_and_sources(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        archive = build_golden_archive(esc_dir, now)

        entry, _ = _aggregate_project('dark_factory', esc_dir, tmp_path / 'runs.db', now=now)
        samples = entry['lifespan']['samples']
        flow_daily = entry['workflow']['flow_daily']
        tier_weekly = entry['workflow']['tier_weekly']
        sources = entry['origin']['sources']

        expected_terminal_count = sum(
            1 for esc in archive.values() if esc['status'] in ('resolved', 'dismissed')
        )

        # sum(flow_daily.n) == len(samples) == resolved(terminal-with-valid
        # -times)-record count.
        total_flow_n = sum(row['n'] for row in flow_daily)
        assert total_flow_n == len(samples) == expected_terminal_count

        # Per-date marginals match exactly (both keyed by date(resolved_at)
        # over the identical population) -> summing over ANY date
        # sub-window preserves the identity, not just in aggregate.
        sample_date_counts = Counter(row[0] for row in samples)
        flow_date_counts = Counter()
        for row in flow_daily:
            flow_date_counts[row['date']] += row['n']
        assert sample_date_counts == flow_date_counts

        # sum(flow_daily.n) grouped by tier == tier_weekly totals per tier
        # (== samples grouped by tier too — same population/tier derivation).
        sample_tier_counts = Counter(row[1] for row in samples)
        flow_tier_counts: Counter = Counter()
        for row in flow_daily:
            flow_tier_counts[row['tier']] += row['n']
        tier_weekly_totals: Counter = Counter()
        for week_bucket in tier_weekly.values():
            for tier, n in week_bucket.items():
                tier_weekly_totals[tier] += n
        assert flow_tier_counts == tier_weekly_totals == sample_tier_counts

        # sum(flow_daily.n) grouped by (source, class) == sources[]
        # benign/actionable counts.
        flow_source_class_counts: Counter = Counter()
        for row in flow_daily:
            flow_source_class_counts[(row['source'], row['class'])] += row['n']
        for s in sources:
            source = s['source']
            assert flow_source_class_counts.get((source, 'benign'), 0) == s['benign']
            assert flow_source_class_counts.get((source, 'actionable'), 0) == s['actionable']


# ---------------------------------------------------------------------------
# step-14: deterministic stratified-by-tier downsample of lifespan.samples
# ---------------------------------------------------------------------------


class TestSamplesDownsampling:
    """lifespan.samples: deterministic stratified-by-tier downsample above threshold.

    ``_aggregate_project`` takes a ``downsample_threshold`` kwarg (mirroring
    ``build_escalation_analytics``'s top-level parameter). Above threshold,
    ``samples`` is capped with a per-tier-proportional, RNG-free selection;
    the payload carries a loud ``samples_downsampled`` marker plus
    ``samples_total`` (the pre-downsample count) — no silent truncation.
    Below/at threshold, ``samples`` is untouched and neither key appears.
    """

    # 3 tiers at a 3:2:1 ratio (60/40/20 of 120 total) so a threshold=40
    # downsample gives every tier a comfortably non-zero quota.
    _TIER_RESOLVERS = {
        'human': 'interactive',
        'auto-watcher': 'escalation-watcher-auto',
        'reaper-sweep': 'auto-dismissed',
    }
    _TIER_COUNTS = {'human': 60, 'auto-watcher': 40, 'reaper-sweep': 20}

    def _build_fixture(self, esc_dir: Path, now: datetime) -> int:
        """Write ``_TIER_COUNTS`` terminal records per tier; return the total count."""
        total = 0
        for tier, resolved_by in self._TIER_RESOLVERS.items():
            for i in range(self._TIER_COUNTS[tier]):
                esc = {
                    'id': f'esc-ds-{tier}-{i}', 'task_id': f'ds-{tier}-{i}',
                    'agent_role': 'implementer',
                    'severity': 'info', 'category': 'cleanup_needed',
                    'summary': 'downsample fixture',
                    'timestamp': _iso(now - timedelta(days=100) + timedelta(hours=i)),
                    'status': 'resolved', 'level': 0,
                    'resolved_at': _iso(
                        now - timedelta(days=100) + timedelta(hours=i, minutes=30)
                    ),
                    'resolved_by': resolved_by,
                }
                _write_escalation(esc_dir, esc, archived=True)
                total += 1
        return total

    def test_downsamples_above_threshold_roughly_proportional_and_deterministic(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        total = self._build_fixture(esc_dir, now)
        assert total == sum(self._TIER_COUNTS.values())

        threshold = 40
        entry1, _ = _aggregate_project(
            'dark_factory', esc_dir, tmp_path / 'runs.db', now=now,
            downsample_threshold=threshold,
        )
        lifespan1 = entry1['lifespan']
        samples1 = lifespan1['samples']

        assert len(samples1) <= threshold
        assert lifespan1['samples_downsampled'] is True
        assert lifespan1['samples_total'] == total

        # No tier zeroed out; representation roughly proportional to each
        # tier's original share (generous tolerance — the exact per-tier
        # quota algorithm is an implementation detail this test isn't
        # pinned to).
        downsampled_tier_counts = Counter(row[1] for row in samples1)
        assert set(downsampled_tier_counts) == set(self._TIER_COUNTS)
        for tier, orig_n in self._TIER_COUNTS.items():
            expected_share = orig_n / total
            actual_share = downsampled_tier_counts[tier] / len(samples1)
            assert abs(actual_share - expected_share) < 0.15

        # Deterministic: a second independent call over the same fixture
        # yields the exact same samples (no RNG involved).
        entry2, _ = _aggregate_project(
            'dark_factory', esc_dir, tmp_path / 'runs.db', now=now,
            downsample_threshold=threshold,
        )
        assert entry2['lifespan']['samples'] == samples1

    def test_no_downsample_when_threshold_at_or_above_sample_count(self, tmp_path):
        from dashboard.data.escalation_analytics import _aggregate_project

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        total = self._build_fixture(esc_dir, now)

        entry, _ = _aggregate_project(
            'dark_factory', esc_dir, tmp_path / 'runs.db', now=now,
            downsample_threshold=total + 1,
        )
        lifespan = entry['lifespan']

        assert len(lifespan['samples']) == total
        assert 'samples_downsampled' not in lifespan
        assert 'samples_total' not in lifespan


# ---------------------------------------------------------------------------
# archives_present — the payload's own "did this scan reach the archive?" signal
# ---------------------------------------------------------------------------

_ANALYTICS_KEYS = {'generated_at', 'parse_failures', 'regime_markers', 'per_project',
                   'archives_present', 'archives_reached'}


class TestArchivesPresent:
    """Did every configured project's escalation archive actually exist?

    The COMPLETENESS diagnostic: ``all()`` over the configured roots.  It
    answers "was this scan whole?", which is an operator's question — the
    cache's question ("did this scan reach anything at all?") is answered by
    ``archives_reached`` next door, and the two are deliberately not the same
    field.  Nothing else in the payload can answer either one:
    ``iter_all_escalation_paths`` returns silently on a missing dir and
    ``Path.glob`` swallows ``PermissionError``, so an absent archive, an
    unreadable one and a genuinely empty one are otherwise byte-identical —
    zero filings, zero parse failures, same shape.

    Deliberately NOT derived from ``parse_failures``.  That counts unparseable
    RECORDS and is a permanent property of a corrupt file, so a cache keyed on
    it would be defeated forever by one bad record sitting in front of the
    expensive walk.  The two fields answer different questions, and the last
    test here pins that they stay independent in both directions.
    """

    def test_a_present_archive_reports_true(self, tmp_path):
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir(parents=True)
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics([('dark_factory', esc_dir, runs_db)], now=now)

        assert result['archives_present'] is True

    def test_an_absent_archive_reports_false(self, tmp_path):
        """The dir was never created — the walk had nothing to reach.

        Indistinguishable from an empty archive in every other field, which is
        exactly why this key has to exist.
        """
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        never_created = tmp_path / 'nope' / 'data' / 'escalations'
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics([('dark_factory', never_created, runs_db)], now=now)

        assert result['archives_present'] is False
        # The rest of the payload is unchanged: still a well-formed, empty,
        # NON-erroring entry.  This signal reports on the scan, it does not
        # degrade the answer.
        assert result['parse_failures'] == 0
        assert len(result['per_project']) == 1
        assert result['per_project'][0]['project'] == 'dark_factory'

    def test_one_absent_project_of_two_reports_false(self, tmp_path):
        """It is ``all()``, not ``any()`` — this field reports COMPLETENESS.

        A multi-project payload where one root's archive is missing is a
        partial scan, and this field says so.  It does NOT follow that a
        partial scan is uncacheable: see ``TestArchivesReached``, which pins
        the cache predicate onto ``any()`` precisely because the ordinary
        production config is permanently partial.
        """
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        present = tmp_path / 'primary' / 'data' / 'escalations'
        present.mkdir(parents=True)
        absent = tmp_path / 'secondary' / 'data' / 'escalations'
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics(
            [('primary', present, runs_db), ('secondary', absent, runs_db)], now=now,
        )

        assert result['archives_present'] is False
        assert [p['project'] for p in result['per_project']] == ['primary', 'secondary']

    def test_the_existing_contract_keys_are_untouched(self, tmp_path):
        """Additive: the four Seam-2 keys keep their names, values and shapes."""
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        build_golden_archive(esc_dir, now)
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics([('dark_factory', esc_dir, runs_db)], now=now)

        assert set(result) == _ANALYTICS_KEYS
        assert result['generated_at'] == now.isoformat()
        assert isinstance(result['regime_markers'], list)
        entry = result['per_project'][0]
        assert {'project', 'origin', 'lifespan', 'workflow'} <= set(entry)

    def test_a_corrupt_record_does_not_flip_archives_present(self, tmp_path):
        """The two signals are independent, and this is the load-bearing case.

        ``build_golden_archive`` plants one non-JSON file, so ``parse_failures``
        is >= 1 permanently.  If ``archives_present`` tracked it, the route's
        cache would be permanently defeated by a single corrupt record — the
        exact failure the reviewer's suggestion was NOT asking for.  The
        archive dir exists and was walked; that a record inside it is garbage
        is a different fact, reported by a different field.
        """
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        build_golden_archive(esc_dir, now)
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics([('dark_factory', esc_dir, runs_db)], now=now)

        assert result['parse_failures'] >= 1
        assert result['archives_present'] is True


# ---------------------------------------------------------------------------
# archives_reached — the payload's "did this scan reach ANY archive?" signal
# ---------------------------------------------------------------------------


class TestArchivesReached:
    """Did the scan reach AT LEAST ONE configured archive?

    ``archives_present`` (``all``) and ``archives_reached`` (``any``) answer
    two different questions, and conflating them is what this class exists to
    prevent.  ``all`` is the completeness DIAGNOSTIC an operator reads; ``any``
    is the CACHE predicate, and only ``any`` is sound for that job.

    Measured on this machine, 2026-08-01, against the installed unit's own
    root list (``systemctl --user cat dark-factory-dashboard`` ->
    ``DASHBOARD_KNOWN_PROJECT_ROOTS``, 9 roots): 2 roots
    (``/home/leo/src/autotrade``, ``/home/leo/mission-control``) have no
    ``data/escalations`` dir at all, while the other 7 hold ~9.1k ``esc-*.json``
    records between them (dark_factory 3278, reify 5561, autopilot-video 69,
    +4 smaller).  So in the CURRENT production config ``all`` is permanently
    False and ``any`` is True.  Keying the cache on ``all`` means the 60s TTL
    never stores anything and every 3s poll (``POLL_INTERVAL_MS``,
    static/redux/data.js) re-runs the whole multi-second walk — i.e. ``all``
    reports "this scan was free to redo" at exactly the moment it was most
    expensive.  A root that has simply never escalated must not delete the
    cache in front of the other seven.

    The remaining question — is ``any`` too lax? — is answered by what the
    predicate is FOR: a build that walked nothing is genuinely free to redo
    (one ``is_dir`` stat per project, all of them negative), and pinning it
    would keep the tab reporting an empty archive for a full TTL window after
    the volume mounts.  A build that walked something paid for that walk, and
    the cache is what stops it being paid again three seconds later.
    """

    def test_a_present_archive_reports_true(self, tmp_path):
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir(parents=True)
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics([('dark_factory', esc_dir, runs_db)], now=now)

        assert result['archives_reached'] is True

    def test_an_absent_archive_reports_false(self, tmp_path):
        """Nothing was walked, so nothing is worth pinning — both signals agree."""
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        never_created = tmp_path / 'nope' / 'data' / 'escalations'
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics([('dark_factory', never_created, runs_db)], now=now)

        assert result['archives_reached'] is False
        assert result['archives_present'] is False

    def test_one_absent_project_of_two_is_reached_but_not_present(self, tmp_path):
        """THE case, and the whole reason this field exists.

        This is the shape of the installed 9-root config (2 archive-less roots,
        7 holding ~9.1k records — see the class docstring).  The scan reached
        an archive and paid the full walk for it, so the payload MUST be
        cacheable; it did not reach every archive, so the diagnostic must still
        say the picture is partial.  One field cannot be both.
        """
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        present = tmp_path / 'primary' / 'data' / 'escalations'
        present.mkdir(parents=True)
        absent = tmp_path / 'secondary' / 'data' / 'escalations'
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics(
            [('primary', present, runs_db), ('secondary', absent, runs_db)], now=now,
        )

        assert result['archives_reached'] is True
        assert result['archives_present'] is False
        assert [p['project'] for p in result['per_project']] == ['primary', 'secondary']

    def test_two_absent_projects_report_false_on_both_signals(self, tmp_path):
        """Reached nothing AND complete-of-nothing: the one case that agrees."""
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        runs_db = _make_runs_db(tmp_path, [])
        first = tmp_path / 'primary' / 'data' / 'escalations'
        second = tmp_path / 'secondary' / 'data' / 'escalations'

        result = build_escalation_analytics(
            [('primary', first, runs_db), ('secondary', second, runs_db)], now=now,
        )

        assert result['archives_reached'] is False
        assert result['archives_present'] is False

    def test_no_configured_projects_reports_not_reached(self, tmp_path):
        """The degenerate case, pinned deliberately rather than left to fall out.

        ``any([])`` is False and ``all([])`` is True, so an empty config is the
        one shape where the two signals INVERT: nothing was reached, yet
        everything configured was present.  Reached-nothing is the correct
        answer for the cache — there is no walk to protect — and writing it
        down here means a later refactor that flips either default trips a
        test instead of silently pinning an empty payload.
        """
        from dashboard.data.escalation_analytics import build_escalation_analytics

        result = build_escalation_analytics([], now=golden_now())

        assert result['archives_reached'] is False
        assert result['archives_present'] is True
        assert result['per_project'] == []

    def test_a_corrupt_record_does_not_flip_archives_reached(self, tmp_path):
        """Twin of the ``archives_present`` case: the three signals stay independent.

        ``build_golden_archive`` plants one non-JSON file, so ``parse_failures``
        is >= 1 permanently.  If the cache predicate tracked it, a single
        corrupt record would defeat the cache forever in front of the very
        walk it protects.  The archive was reached; that a record inside it is
        garbage is a different fact, reported by a different field.
        """
        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        build_golden_archive(esc_dir, now)
        runs_db = _make_runs_db(tmp_path, [])

        result = build_escalation_analytics([('dark_factory', esc_dir, runs_db)], now=now)

        assert result['parse_failures'] >= 1
        assert result['archives_reached'] is True


class TestArchiveScanSucceeded:
    """The exported cache predicate, unit-tested away from the route.

    Mirrors ``memory_evals.root_scan_succeeded``: a NAMED function rather than
    a lambda in the route, so the rule is testable in isolation and its
    rationale has somewhere to live that a route docstring is not.
    """

    def test_a_reached_archive_is_cacheable(self):
        from dashboard.data.escalation_analytics import archive_scan_succeeded

        assert archive_scan_succeeded({'archives_reached': True}) is True

    def test_a_partial_scan_is_cacheable(self):
        """The behaviour change, stated as a unit fact.

        Reached some, missed some — the walk was paid for, so it is cached.
        ``archives_present: False`` rides along as the diagnostic and has no
        say in the cacheability decision.
        """
        from dashboard.data.escalation_analytics import archive_scan_succeeded

        payload = {'archives_reached': True, 'archives_present': False}

        assert archive_scan_succeeded(payload) is True

    def test_an_unreached_archive_is_not_cacheable(self):
        from dashboard.data.escalation_analytics import archive_scan_succeeded

        assert archive_scan_succeeded({'archives_reached': False}) is False

    def test_a_payload_without_the_key_is_not_cacheable(self):
        """Reads through ``.get`` because it runs in the cache WRITE path.

        A partially-built or older-shaped payload must degrade to "don't
        cache" — a raise here surfaces as a 500 on a 3s dashboard poll.
        """
        from dashboard.data.escalation_analytics import archive_scan_succeeded

        assert archive_scan_succeeded({'parse_failures': 0}) is False
        assert archive_scan_succeeded({}) is False


# ---------------------------------------------------------------------------
# step-16: row 7 — perf: cold ~10k-record build_escalation_analytics() < 5s
# ---------------------------------------------------------------------------

_PERF_SOURCES = ('implementer', 'architect', 'orchestrator-merge-worker', 'harness-orphan-reaper')
_PERF_RESOLVERS = (
    'interactive', 'escalation-watcher-auto', 'auto-dismissed',
    'harness-orphan-reaper', 'l2-cascade:esc-perf-anchor', 'claude-task-9-steward',
)


def _write_perf_archive(esc_dir: Path, now: datetime, n: int) -> int:
    """Write *n* synthetic escalation records spread across dated subdirs.

    Deterministic (no RNG, for a reproducible perf run): mixed levels
    (0/1/2), mixed ``agent_role`` sources, mixed ``resolved_by`` tiers,
    ~90% terminal (resolved/dismissed, resolved_at spread across ~90
    distinct ``archive/YYYY-MM-DD`` dirs) / ~10% pending (root-tier).
    ``task_id`` repeats with a small period so the workflow block's
    per-task churn scan sees realistic (bounded, not O(n) group size)
    re-filing groups. Returns the terminal (resolved/dismissed) count.
    """
    terminal_count = 0
    for i in range(n):
        is_pending = i % 10 == 9
        timestamp = now - timedelta(days=90 - (i % 90), hours=i % 24, minutes=i % 60)
        esc = {
            'id': f'esc-perf-{i}', 'task_id': f'perf-{i % 4000}',
            'agent_role': _PERF_SOURCES[i % len(_PERF_SOURCES)],
            'severity': 'info', 'category': 'cleanup_needed', 'summary': 'perf fixture',
            'timestamp': _iso(timestamp),
            'status': 'pending' if is_pending else ('dismissed' if i % 2 == 0 else 'resolved'),
            'level': i % 3,
        }
        if not is_pending:
            esc['resolved_at'] = _iso(timestamp + timedelta(hours=1 + i % 48))
            esc['resolved_by'] = _PERF_RESOLVERS[i % len(_PERF_RESOLVERS)]
            terminal_count += 1
        _write_escalation(esc_dir, esc, archived=not is_pending)
    return terminal_count


class TestBuildEscalationAnalyticsPerf:
    """PRD boundary row 7: a cold ~10k-record archive walk stays within a
    CPU-time budget.

    This is an O(n^2) archive-walk regression guard, not a wall-clock SLA
    — see the in-body comment on the assertion for why the measure is
    CPU time rather than wall-clock, and how the budget was derived.
    """

    def test_cold_10k_archive_cpu_budget(self, tmp_path):
        import time

        from dashboard.data.escalation_analytics import build_escalation_analytics

        now = golden_now()
        esc_dir = tmp_path / 'escalations'
        terminal_count = _write_perf_archive(esc_dir, now, 10_000)

        runs_db = _make_runs_db(tmp_path, [
            ('perf-done-1', 'done', f'{(now - timedelta(days=1)).date().isoformat()}T09:00:00+00:00'),
        ])

        started = time.process_time()
        result = build_escalation_analytics([('dark_factory', esc_dir, runs_db)], now=now)
        cpu_elapsed = time.process_time() - started

        # The bound guards against an O(n^2) archive-walk regression, NOT a
        # wall-clock SLA. A wall-clock (time.monotonic) budget here proved
        # contention-sensitive twice: first to intra-process xdist worker
        # contention (task 2702/2722, observed 8.32s/5.05s against a then-5s
        # bound, widened to 15s and gated to single-worker-only), then to
        # cross-worktree contention from concurrent orchestrator verifies on
        # the same host (task 3344, observed 34.78s single-worker against
        # the widened 15s bound; nproc=32 with loadavg 78-138 at the time).
        # time.process_time() measures actual CPU seconds consumed by this
        # process — it excludes time spent waiting for a CPU slot or for
        # disk I/O while other processes (xdist workers, sibling worktree
        # verifies) are scheduled, so it stays stable regardless of host
        # contention while still catching a real O(n^2) blowup. No
        # single-worker gate is needed: the measure itself is
        # contention-insensitive. The correctness assertions below always
        # run, including under xdist.
        #
        # Budget derivation (task 3344 amendment): observed 0.81-1.09s CPU
        # across 4 repeated runs on this dev host, taken *while* it was
        # itself under heavy contention (32 cores, loadavg 150+) — if
        # anything an overestimate of a quiet-host baseline, since
        # process_time is expected to be stable regardless of load. Budget
        # is 5.0s, ~5x that observed baseline: enough slack for slower or
        # differently-provisioned CI hardware, tight enough that a real
        # O(n^2) blowup at 10k records — which would cost orders of
        # magnitude more CPU, not a small multiple — still trips it.
        assert cpu_elapsed < 5.0, (
            f'cold build_escalation_analytics used {cpu_elapsed:.2f}s of CPU time '
            '(budget 5.0s)'
        )

        # Well-formed at scale: parse_failures==0 (every fixture record is
        # valid), one per-project entry with non-empty samples.
        assert result['parse_failures'] == 0
        assert len(result['per_project']) == 1
        project = result['per_project'][0]
        assert project['project'] == 'dark_factory'
        samples = project['lifespan']['samples']
        assert len(samples) > 0

        # terminal_count (9000) is below the default downsample_threshold
        # (10_000) at this scale, so samples must carry every terminal
        # record 1:1 — no downsampling triggered.
        assert len(samples) == terminal_count
        assert 'samples_downsampled' not in project['lifespan']
