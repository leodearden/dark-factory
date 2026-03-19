"""Tests for the reconciliation panel route and supporting utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


class TestTimeagoFilter:
    """Tests for the timeago Jinja2 filter function."""

    def test_none_returns_never(self):
        from dashboard.app import timeago

        assert timeago(None) == 'never'

    def test_minutes_ago(self):
        from dashboard.app import timeago

        ts = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()
        assert timeago(ts) == '3m ago'

    def test_hours_ago(self):
        from dashboard.app import timeago

        ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert timeago(ts) == '2h ago'

    def test_days_ago(self):
        from dashboard.app import timeago

        ts = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        assert timeago(ts) == '2d ago'

    def test_invalid_string_returns_never(self):
        from dashboard.app import timeago

        assert timeago('not-a-date') == 'never'

    def test_empty_string_returns_never(self):
        from dashboard.app import timeago

        assert timeago('') == 'never'

    def test_filter_registered_on_jinja2_env(self):
        from dashboard.app import templates, timeago

        assert 'timeago' in templates.env.filters
        assert templates.env.filters['timeago'] is timeago
