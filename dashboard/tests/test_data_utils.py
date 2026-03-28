"""Tests for dashboard.data.utils shared utility functions."""

from __future__ import annotations


class TestParseUtc:
    """Tests for the parse_utc public helper in dashboard.data.utils."""

    def test_naive_iso_string_gets_utc(self):
        """Naive ISO string (no tzinfo) should be returned with UTC attached."""
        from datetime import UTC as _UTC

        from dashboard.data.utils import parse_utc

        result = parse_utc('2026-03-28T10:00:00')
        assert result.tzinfo is not None
        assert result.tzinfo == _UTC

    def test_aware_iso_string_preserved(self):
        """Aware ISO string (with tzinfo) should be returned unchanged."""
        from dashboard.data.utils import parse_utc

        ts = '2026-03-28T10:00:00+00:00'
        result = parse_utc(ts)
        assert result.tzinfo is not None
        # Value is preserved: tzinfo stays, offset is the same
        assert result.year == 2026
        assert result.hour == 10

    def test_invalid_string_raises_value_error(self):
        """Invalid ISO string should raise ValueError."""
        import pytest

        from dashboard.data.utils import parse_utc

        with pytest.raises(ValueError):
            parse_utc('not-a-timestamp')

    def test_none_raises_type_error(self):
        """None input should raise TypeError (ts: str does not accept None)."""
        import pytest

        from dashboard.data.utils import parse_utc

        with pytest.raises(TypeError):
            parse_utc(None)  # type: ignore[arg-type]


class TestTimeagoUsesParseUtc:
    """Tests verifying that app.py uses parse_utc from dashboard.data.utils (DRY)."""

    def test_naive_timestamp_timeago_works(self):
        """timeago with a naive ISO timestamp (no tzinfo) returns a relative string."""
        from dashboard.app import timeago

        # Naive timestamps must be handled correctly — this documents the expected
        # behavior after refactoring and serves as a regression guard.
        result = timeago('2026-03-28T10:00:00')
        # Result is a non-empty string (relative time, 'just now', or 'Xm/h/d ago')
        assert isinstance(result, str)
        assert result != 'never'

    def test_app_imports_parse_utc_from_utils(self):
        """dashboard.app must import parse_utc from dashboard.data.utils."""
        import dashboard.app as app_module

        # After step-6 refactoring, parse_utc should be in app.py's namespace
        assert hasattr(app_module, 'parse_utc'), (
            "dashboard.app does not expose parse_utc — "
            "timeago filter still uses inline UTC normalization instead of the shared helper"
        )
