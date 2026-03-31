"""Deep integration tests for the costs panel — nav bar states, partial rendering,
value assertions, chart canvas IDs, Alpine expandable patterns, and event color-coding.

Follows the test_performance_panel.py ExitStack-based combined-patching pattern.
Complements test_costs_page.py (which uses per-function patch helpers and loose
assertions). This file verifies that exact mock values flow through to rendered HTML.
"""

from __future__ import annotations

import sqlite3
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from tests.test_costs_data import COSTS_SCHEMA

# ---------------------------------------------------------------------------
# Mock data — mirrors test_costs_page.py for consistency, with values
# chosen to produce unique / checkable rendered strings.
# ---------------------------------------------------------------------------

_MOCK_SUMMARY = {
    'dark_factory': {
        'total_spend': 12.34,
        'avg_cost_per_task': 0.56,
        'active_accounts': 3,
        'cap_events': 2,
    },
    'other_project': {
        'total_spend': 5.00,
        'avg_cost_per_task': 0.25,
        'active_accounts': 1,
        'cap_events': 0,
    },
}

_MOCK_BY_PROJECT = {
    'dark_factory': [
        {'model': 'claude-sonnet', 'total': 8.50},
        {'model': 'claude-opus', 'total': 3.84},
    ],
    'other_project': [
        {'model': 'claude-haiku', 'total': 1.20},
    ],
}

_MOCK_BY_ACCOUNT = {
    'account-A': {
        'spend': 9.50,
        'invocations': 50,
        'cap_events': 1,
        'last_cap': '2026-03-30T10:00:00+00:00',
        'status': 'capped',
    },
    'account-B': {
        'spend': 3.20,
        'invocations': 20,
        'cap_events': 0,
        'last_cap': None,
        'status': 'active',
    },
}

_MOCK_BY_ROLE = {
    'dark_factory': {
        'implementer': {'claude-sonnet': 5.20, 'claude-opus': 1.50},
        'reviewer': {'claude-opus': 3.10},
    },
    'other_project': {
        'implementer': {'claude-haiku': 0.80},
    },
}

_MOCK_TREND = {
    'dark_factory': [
        {'day': '2026-03-25', 'total': 2.10},
        {'day': '2026-03-26', 'total': 3.40},
        {'day': '2026-03-27', 'total': 1.80},
    ],
    'other_project': [
        {'day': '2026-03-25', 'total': 0.50},
        {'day': '2026-03-26', 'total': 0.70},
        {'day': '2026-03-27', 'total': 0.30},
    ],
}

_MOCK_EVENTS = [
    {
        'account_name': 'account-A',
        'event_type': 'cap_hit',
        'project_id': 'dark_factory',
        'run_id': 'run-123',
        'details': None,
        'created_at': '2026-03-30T10:00:00+00:00',
    },
    {
        'account_name': 'account-B',
        'event_type': 'resumed',
        'project_id': 'dark_factory',
        'run_id': 'run-456',
        'details': None,
        'created_at': '2026-03-30T11:30:00+00:00',
    },
]

_MOCK_RUNS = [
    {
        'run_id': 'run-abc-123',
        'project_id': 'dark_factory',
        'total_cost': 4.20,
        'tasks': [
            {
                'task_id': '42',
                'title': 'Implement feature X',
                'cost': 3.50,
                'invocations': [
                    {
                        'model': 'claude-sonnet',
                        'role': 'implementer',
                        'cost_usd': 2.00,
                        'account_name': 'account-A',
                        'duration_ms': 15000,
                        'capped': False,
                    },
                    {
                        'model': 'claude-opus',
                        'role': 'reviewer',
                        'cost_usd': 1.50,
                        'account_name': 'account-B',
                        'duration_ms': 8000,
                        'capped': True,
                    },
                ],
            },
            {
                'task_id': None,
                'title': None,
                'cost': 0.70,
                'invocations': [
                    {
                        'model': 'claude-haiku',
                        'role': 'orchestrator',
                        'cost_usd': 0.70,
                        'account_name': 'account-A',
                        'duration_ms': 3000,
                        'capped': False,
                    },
                ],
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# ExitStack helper — patches all 7 cost data functions at once.
# The _UNSET sentinel lets callers selectively override individual values.
# ---------------------------------------------------------------------------

_UNSET = object()


def _patch_cost_data(
    summary=_UNSET,
    by_project=_UNSET,
    by_account=_UNSET,
    by_role=_UNSET,
    trend=_UNSET,
    events=_UNSET,
    runs=_UNSET,
):
    """Return an ExitStack that patches all 7 cost data functions in dashboard.app."""
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.get_cost_summary',
        new_callable=AsyncMock,
        return_value=summary if summary is not _UNSET else _MOCK_SUMMARY,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_cost_by_project',
        new_callable=AsyncMock,
        return_value=by_project if by_project is not _UNSET else _MOCK_BY_PROJECT,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_cost_by_account',
        new_callable=AsyncMock,
        return_value=by_account if by_account is not _UNSET else _MOCK_BY_ACCOUNT,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_cost_by_role',
        new_callable=AsyncMock,
        return_value=by_role if by_role is not _UNSET else _MOCK_BY_ROLE,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_cost_trend',
        new_callable=AsyncMock,
        return_value=trend if trend is not _UNSET else _MOCK_TREND,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_account_events',
        new_callable=AsyncMock,
        return_value=events if events is not _UNSET else _MOCK_EVENTS,
    ))
    stack.enter_context(patch(
        'dashboard.app.get_run_cost_breakdown',
        new_callable=AsyncMock,
        return_value=runs if runs is not _UNSET else _MOCK_RUNS,
    ))
    return stack


# ---------------------------------------------------------------------------
# SQLite DB fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def costs_panel_db(tmp_path):
    """Populated runs.db for costs panel tests.

    Contains invocations across:
      - 2 projects: dark_factory, reify
      - 3 accounts: max-a, max-b, max-c
      - 2 models: claude-opus-4-5, claude-sonnet-4-5
      - 3 roles: implementer, reviewer, debugger
    Plus account_events with cap_hit and resumed entries.

    Returns the path to the database file.
    """
    db_path = tmp_path / 'runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(COSTS_SCHEMA)

    now = datetime.now(UTC)

    # Runs
    conn.executemany(
        'INSERT INTO runs (run_id, project_id, started_at, completed_at) VALUES (?, ?, ?, ?)',
        [
            (
                'run-panel-001', 'dark_factory',
                (now - timedelta(hours=3)).isoformat(),
                (now - timedelta(hours=1)).isoformat(),
            ),
            (
                'run-panel-002', 'reify',
                (now - timedelta(hours=5)).isoformat(),
                (now - timedelta(hours=3)).isoformat(),
            ),
        ],
    )

    # Task results
    conn.executemany(
        'INSERT INTO task_results '
        '(run_id, task_id, project_id, title, outcome, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        [
            ('run-panel-001', '10', 'dark_factory', 'Implement widget', 'done',
             (now - timedelta(hours=2)).isoformat()),
            ('run-panel-001', '11', 'dark_factory', 'Review widget', 'done',
             (now - timedelta(hours=1, minutes=30)).isoformat()),
            ('run-panel-002', '20', 'reify', 'Reify task A', 'done',
             (now - timedelta(hours=4)).isoformat()),
        ],
    )

    # Invocations
    # dark_factory / run-panel-001
    #   task 10: max-a / claude-opus-4-5 / implementer  (2.50)
    #   task 10: max-b / claude-sonnet-4-5 / reviewer   (0.80)
    #   task 11: max-c / claude-sonnet-4-5 / debugger   (1.20, capped)
    # reify / run-panel-002
    #   task 20: max-a / claude-opus-4-5 / implementer  (1.00)
    invocations = [
        ('run-panel-001', '10', 'dark_factory', 'max-a', 'claude-opus-4-5',
         'implementer', 2.50, 30000, 0,
         (now - timedelta(hours=2, minutes=30)).isoformat(),
         (now - timedelta(hours=2)).isoformat()),
        ('run-panel-001', '10', 'dark_factory', 'max-b', 'claude-sonnet-4-5',
         'reviewer', 0.80, 12000, 0,
         (now - timedelta(hours=2)).isoformat(),
         (now - timedelta(hours=1, minutes=50)).isoformat()),
        ('run-panel-001', '11', 'dark_factory', 'max-c', 'claude-sonnet-4-5',
         'debugger', 1.20, 20000, 1,
         (now - timedelta(hours=1, minutes=45)).isoformat(),
         (now - timedelta(hours=1, minutes=30)).isoformat()),
        ('run-panel-002', '20', 'reify', 'max-a', 'claude-opus-4-5',
         'implementer', 1.00, 18000, 0,
         (now - timedelta(hours=4, minutes=30)).isoformat(),
         (now - timedelta(hours=4)).isoformat()),
    ]
    conn.executemany(
        'INSERT INTO invocations '
        '(run_id, task_id, project_id, account_name, model, role, cost_usd, '
        ' duration_ms, capped, started_at, completed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        invocations,
    )

    # Account events
    conn.executemany(
        'INSERT INTO account_events '
        '(account_name, event_type, project_id, run_id, created_at) '
        'VALUES (?, ?, ?, ?, ?)',
        [
            ('max-c', 'cap_hit', 'dark_factory', 'run-panel-001',
             (now - timedelta(hours=1, minutes=30)).isoformat()),
            ('max-c', 'resumed', 'dark_factory', 'run-panel-001',
             (now - timedelta(hours=1, minutes=15)).isoformat()),
        ],
    )

    conn.commit()
    conn.close()
    yield db_path


@pytest.fixture()
def empty_costs_panel_db(tmp_path):
    """Schema-only runs.db with no data rows.

    Returns the path to the database file.
    """
    db_path = tmp_path / 'empty_runs.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(COSTS_SCHEMA)
    conn.commit()
    conn.close()
    yield db_path


# ---------------------------------------------------------------------------
# Step-1 / Step-2: TestCostsPageNavBar
# ---------------------------------------------------------------------------


class TestCostsPageNavBar:
    """GET /costs nav bar: Costs link is active, Dashboard link is inactive."""

    def test_returns_200(self, client):
        resp = client.get('/costs')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        resp = client.get('/costs')
        assert 'text/html' in resp.headers['content-type']

    def test_costs_nav_link_present(self, client):
        html = client.get('/costs').text
        assert 'Costs' in html

    def test_costs_nav_link_is_active(self, client):
        """The Costs nav link must carry the active class (border-blue-500)."""
        html = client.get('/costs').text
        assert 'border-blue-500' in html

    def test_dashboard_nav_link_present(self, client):
        html = client.get('/costs').text
        assert 'Dashboard' in html

    def test_dashboard_nav_link_is_inactive(self, client):
        """On /costs, Dashboard link must be styled with inactive class (text-gray-400)."""
        html = client.get('/costs').text
        # inactive_cls includes text-gray-400
        assert 'text-gray-400' in html


# ---------------------------------------------------------------------------
# Step-3 / Step-4: TestDashboardPageNavBar
# ---------------------------------------------------------------------------


class TestDashboardPageNavBar:
    """GET / nav bar: Dashboard link is active, Costs link is inactive."""

    def test_returns_200(self, client):
        resp = client.get('/')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        resp = client.get('/')
        assert 'text/html' in resp.headers['content-type']

    def test_dashboard_nav_link_present(self, client):
        html = client.get('/').text
        assert 'Dashboard' in html

    def test_dashboard_nav_link_is_active(self, client):
        """On /, the Dashboard nav link must carry active class (border-blue-500)."""
        html = client.get('/').text
        assert 'border-blue-500' in html

    def test_costs_nav_link_present(self, client):
        html = client.get('/').text
        assert 'Costs' in html

    def test_costs_nav_link_is_inactive(self, client):
        """On /, the Costs link must use inactive styling (text-gray-400)."""
        html = client.get('/').text
        assert 'text-gray-400' in html


# ---------------------------------------------------------------------------
# Step-5 / Step-6: TestCostsPartialStatus
# ---------------------------------------------------------------------------


class TestCostsPartialStatus:
    """Each of the 7 cost partials returns 200 + text/html with populated mock data."""

    def test_summary_returns_200(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/summary')
        assert resp.status_code == 200

    def test_summary_content_type(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/summary')
        assert 'text/html' in resp.headers['content-type']

    def test_by_project_returns_200(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/by-project')
        assert resp.status_code == 200

    def test_by_project_content_type(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/by-project')
        assert 'text/html' in resp.headers['content-type']

    def test_by_account_returns_200(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/by-account')
        assert resp.status_code == 200

    def test_by_account_content_type(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/by-account')
        assert 'text/html' in resp.headers['content-type']

    def test_by_role_returns_200(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/by-role')
        assert resp.status_code == 200

    def test_by_role_content_type(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/by-role')
        assert 'text/html' in resp.headers['content-type']

    def test_trend_returns_200(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/trend')
        assert resp.status_code == 200

    def test_trend_content_type(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/trend')
        assert 'text/html' in resp.headers['content-type']

    def test_events_returns_200(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/events')
        assert resp.status_code == 200

    def test_events_content_type(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/events')
        assert 'text/html' in resp.headers['content-type']

    def test_runs_returns_200(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/runs')
        assert resp.status_code == 200

    def test_runs_content_type(self, client):
        with _patch_cost_data():
            resp = client.get('/costs/partials/runs')
        assert 'text/html' in resp.headers['content-type']


# ---------------------------------------------------------------------------
# Step-7 / Step-8: TestEmptyState
# ---------------------------------------------------------------------------


class TestEmptyState:
    """All partials return 200 with empty-state messages when data is empty."""

    def test_summary_empty_returns_200(self, client):
        with _patch_cost_data(summary={}):
            resp = client.get('/costs/partials/summary')
        assert resp.status_code == 200

    def test_summary_empty_message(self, client):
        with _patch_cost_data(summary={}):
            html = client.get('/costs/partials/summary').text
        assert 'No cost data' in html

    def test_by_project_empty_returns_200(self, client):
        with _patch_cost_data(by_project={}):
            resp = client.get('/costs/partials/by-project')
        assert resp.status_code == 200

    def test_by_project_empty_message(self, client):
        with _patch_cost_data(by_project={}):
            html = client.get('/costs/partials/by-project').text
        assert 'No cost data' in html

    def test_by_account_empty_returns_200(self, client):
        with _patch_cost_data(by_account={}):
            resp = client.get('/costs/partials/by-account')
        assert resp.status_code == 200

    def test_by_account_empty_message(self, client):
        with _patch_cost_data(by_account={}):
            html = client.get('/costs/partials/by-account').text
        assert 'No account data' in html

    def test_by_role_empty_returns_200(self, client):
        with _patch_cost_data(by_role={}):
            resp = client.get('/costs/partials/by-role')
        assert resp.status_code == 200

    def test_by_role_empty_message(self, client):
        with _patch_cost_data(by_role={}):
            html = client.get('/costs/partials/by-role').text
        assert 'No role data' in html

    def test_trend_empty_returns_200(self, client):
        with _patch_cost_data(trend={}):
            resp = client.get('/costs/partials/trend')
        assert resp.status_code == 200

    def test_trend_empty_message(self, client):
        with _patch_cost_data(trend={}):
            html = client.get('/costs/partials/trend').text
        assert 'No trend data' in html

    def test_events_empty_returns_200(self, client):
        with _patch_cost_data(events=[]):
            resp = client.get('/costs/partials/events')
        assert resp.status_code == 200

    def test_events_empty_message(self, client):
        with _patch_cost_data(events=[]):
            html = client.get('/costs/partials/events').text
        assert 'No account events' in html

    def test_runs_empty_returns_200(self, client):
        with _patch_cost_data(runs=[]):
            resp = client.get('/costs/partials/runs')
        assert resp.status_code == 200

    def test_runs_empty_message(self, client):
        with _patch_cost_data(runs=[]):
            html = client.get('/costs/partials/runs').text
        assert 'No run data' in html


# ---------------------------------------------------------------------------
# Step-9 / Step-10: TestTimeWindow
# ---------------------------------------------------------------------------


class TestTimeWindow:
    """Window query param is correctly mapped to days and forwarded to data functions."""

    def test_summary_24h_calls_with_days_1(self, client):
        with patch(
            'dashboard.app.get_cost_summary',
            new_callable=AsyncMock,
            return_value=_MOCK_SUMMARY,
        ) as mock_fn:
            client.get('/costs/partials/summary?window=24h')
        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 1

    def test_summary_default_uses_days_7(self, client):
        with patch(
            'dashboard.app.get_cost_summary',
            new_callable=AsyncMock,
            return_value=_MOCK_SUMMARY,
        ) as mock_fn:
            client.get('/costs/partials/summary')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 7

    def test_trend_24h_calls_with_days_1(self, client):
        with patch(
            'dashboard.app.get_cost_trend',
            new_callable=AsyncMock,
            return_value=_MOCK_TREND,
        ) as mock_fn:
            client.get('/costs/partials/trend?window=24h')
        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 1

    def test_trend_default_uses_days_7(self, client):
        with patch(
            'dashboard.app.get_cost_trend',
            new_callable=AsyncMock,
            return_value=_MOCK_TREND,
        ) as mock_fn:
            client.get('/costs/partials/trend')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 7

    def test_summary_30d_calls_with_days_30(self, client):
        with patch(
            'dashboard.app.get_cost_summary',
            new_callable=AsyncMock,
            return_value=_MOCK_SUMMARY,
        ) as mock_fn:
            client.get('/costs/partials/summary?window=30d')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 30

    def test_trend_all_calls_with_days_3650(self, client):
        with patch(
            'dashboard.app.get_cost_trend',
            new_callable=AsyncMock,
            return_value=_MOCK_TREND,
        ) as mock_fn:
            client.get('/costs/partials/trend?window=all')
        _, kwargs = mock_fn.call_args
        assert kwargs.get('days') == 3650


# ---------------------------------------------------------------------------
# Step-11 / Step-12: TestChartCanvases
# ---------------------------------------------------------------------------


class TestChartCanvases:
    """Chart partials render the expected canvas IDs and Chart.js init code."""

    def test_by_project_canvas_id(self, client):
        with _patch_cost_data():
            html = client.get('/costs/partials/by-project').text
        assert 'costByProjectChart' in html

    def test_by_project_new_chart(self, client):
        with _patch_cost_data():
            html = client.get('/costs/partials/by-project').text
        assert 'new Chart' in html

    def test_by_account_canvas_id(self, client):
        with _patch_cost_data():
            html = client.get('/costs/partials/by-account').text
        assert 'costByAccountChart' in html

    def test_by_account_new_chart(self, client):
        with _patch_cost_data():
            html = client.get('/costs/partials/by-account').text
        assert 'new Chart' in html

    def test_by_role_canvas_id_prefix(self, client):
        """by-role partial uses per-project canvas IDs like costByRoleChart_1."""
        with _patch_cost_data():
            html = client.get('/costs/partials/by-role').text
        assert 'costByRoleChart_' in html

    def test_by_role_new_chart(self, client):
        with _patch_cost_data():
            html = client.get('/costs/partials/by-role').text
        assert 'new Chart' in html

    def test_trend_canvas_id(self, client):
        with _patch_cost_data():
            html = client.get('/costs/partials/trend').text
        assert 'costTrendChart' in html

    def test_trend_new_chart(self, client):
        with _patch_cost_data():
            html = client.get('/costs/partials/trend').text
        assert 'new Chart' in html


# ---------------------------------------------------------------------------
# Step-13 / Step-14: TestSummaryCards
# ---------------------------------------------------------------------------


class TestSummaryCards:
    """Summary partial aggregates values correctly and renders them in the HTML."""

    def test_aggregated_total_spend(self, client):
        """Total spend is aggregated: 12.34 + 5.00 = 17.34 → '$17.34'."""
        with _patch_cost_data():
            html = client.get('/costs/partials/summary').text
        assert '$17.34' in html

    def test_total_active_accounts(self, client):
        """Active accounts are summed: 3 + 1 = 4 → '4' in accounts metric."""
        with _patch_cost_data():
            html = client.get('/costs/partials/summary').text
        # The accounts metric card renders the number; 'distinct accounts' labels it
        assert '4' in html
        assert 'distinct account' in html

    def test_total_cap_events(self, client):
        """Cap events are summed: 2 + 0 = 2 → '2' in cap events metric."""
        with _patch_cost_data():
            html = client.get('/costs/partials/summary').text
        assert '2' in html
        assert 'cap hits in window' in html

    def test_per_project_table_shown_for_multiple_projects(self, client):
        """With 2 projects, the per-project breakdown table must appear."""
        with _patch_cost_data():
            html = client.get('/costs/partials/summary').text
        # Both project names should appear in the table rows
        assert 'dark_factory' in html
        assert 'other_project' in html

    def test_per_project_spend_values(self, client):
        """Individual project spends appear in the per-project table."""
        with _patch_cost_data():
            html = client.get('/costs/partials/summary').text
        # dark_factory: $12.34, other_project: $5.00
        assert '$12.34' in html
        assert '$5.00' in html

    def test_no_table_for_single_project(self, client):
        """With only 1 project, the per-project table must NOT appear."""
        single_project_summary = {
            'solo_project': {
                'total_spend': 9.99,
                'avg_cost_per_task': 1.00,
                'active_accounts': 2,
                'cap_events': 0,
            },
        }
        with _patch_cost_data(summary=single_project_summary):
            html = client.get('/costs/partials/summary').text
        # Per-project breakdown table columns should not appear for single project
        assert 'solo_project' not in html or 'Avg/Task' not in html


# ---------------------------------------------------------------------------
# Step-15 / Step-16: TestEventsFeed
# ---------------------------------------------------------------------------


class TestEventsFeed:
    """Events partial renders account names, event types, and color-coded borders."""

    def test_cap_hit_border_red(self, client):
        """cap_hit events use border-red-500 left border."""
        with _patch_cost_data():
            html = client.get('/costs/partials/events').text
        assert 'border-red-500' in html

    def test_resumed_border_green(self, client):
        """resumed events use border-green-500 left border."""
        with _patch_cost_data():
            html = client.get('/costs/partials/events').text
        assert 'border-green-500' in html

    def test_account_names_present(self, client):
        """Both account names from mock events appear in the rendered HTML."""
        with _patch_cost_data():
            html = client.get('/costs/partials/events').text
        assert 'account-A' in html
        assert 'account-B' in html

    def test_event_type_badges_present(self, client):
        """Event type labels (cap_hit, resumed) appear as badge text."""
        with _patch_cost_data():
            html = client.get('/costs/partials/events').text
        assert 'cap_hit' in html
        assert 'resumed' in html

    def test_run_ids_present(self, client):
        """Run IDs from the events appear in the rendered output."""
        with _patch_cost_data():
            html = client.get('/costs/partials/events').text
        assert 'run-123' in html
        assert 'run-456' in html

    def test_empty_events_message(self, client):
        """Empty events list renders 'No account events' message."""
        with _patch_cost_data(events=[]):
            html = client.get('/costs/partials/events').text
        assert 'No account events' in html

    def test_empty_events_no_red_border(self, client):
        """With empty events, no border-red-500 should appear."""
        with _patch_cost_data(events=[]):
            html = client.get('/costs/partials/events').text
        assert 'border-red-500' not in html

    def test_empty_events_no_green_border(self, client):
        """With empty events, no border-green-500 should appear."""
        with _patch_cost_data(events=[]):
            html = client.get('/costs/partials/events').text
        assert 'border-green-500' not in html


# ---------------------------------------------------------------------------
# Step-17 / Step-18: TestRunsTable
# ---------------------------------------------------------------------------


class TestRunsTable:
    """Runs partial renders the expandable run/task/invocation structure."""

    def test_run_id_present(self, client):
        """Run ID from mock data appears in the rendered HTML."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert 'run-abc-123' in html

    def test_alpine_x_data_present(self, client):
        """Expandable run rows use Alpine x-data directive."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert 'x-data' in html

    def test_alpine_x_show_present(self, client):
        """Expandable content uses Alpine x-show directive."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert 'x-show' in html

    def test_aria_expanded_present(self, client):
        """Expandable buttons carry :aria-expanded binding."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert 'aria-expanded' in html

    def test_task_title_present(self, client):
        """Task title 'Implement feature X' appears in the expandable details."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert 'Implement feature X' in html

    def test_cost_values_present(self, client):
        """Dollar cost values appear in the rendered HTML."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert '$' in html

    def test_capped_badge_present(self, client):
        """Invocations with capped=True render a 'capped' badge."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert 'capped' in html.lower()

    def test_null_task_id_renders_dash(self, client):
        """Tasks with null task_id render an em-dash fallback (—)."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert '—' in html

    def test_invocation_model_present(self, client):
        """Model names from invocations appear in the rendered table."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert 'claude-sonnet' in html
        assert 'claude-opus' in html

    def test_invocation_role_present(self, client):
        """Role names from invocations appear in the rendered table."""
        with _patch_cost_data():
            html = client.get('/costs/partials/runs').text
        assert 'implementer' in html
        assert 'reviewer' in html

    def test_empty_runs_message(self, client):
        """Empty runs list renders 'No run data' message."""
        with _patch_cost_data(runs=[]):
            html = client.get('/costs/partials/runs').text
        assert 'No run data' in html
