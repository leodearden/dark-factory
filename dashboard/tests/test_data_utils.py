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
