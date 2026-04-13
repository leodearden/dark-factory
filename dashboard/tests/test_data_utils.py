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

    def test_aware_iso_string_with_non_utc_offset_preserved(self):
        """Aware ISO string with non-UTC offset should be returned unchanged (tzinfo preserved)."""
        from datetime import timedelta

        from dashboard.data.utils import parse_utc

        result = parse_utc('2026-03-28T10:00:00+05:30')
        assert result.utcoffset() == timedelta(hours=5, minutes=30)
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

    def test_none_raises_type_error_with_explicit_message(self):
        """TypeError raised for None should have the explicit message 'timestamp is None'."""
        import pytest

        from dashboard.data.utils import parse_utc

        with pytest.raises(TypeError, match='timestamp is None'):
            parse_utc(None)  # type: ignore[arg-type]

    def test_docstring_describes_none_raises_type_error_explicitly(self):
        """parse_utc docstring must document None raising TypeError explicitly."""
        from dashboard.data.utils import parse_utc

        doc = parse_utc.__doc__ or ''
        assert 'None raises TypeError explicitly' in doc, (
            f"Expected 'None raises TypeError explicitly' in docstring, got: {doc!r}"
        )
        assert 'propagates ValueError or TypeError from' not in doc, (
            f"Stale phrase 'propagates ValueError or TypeError from' still in docstring: {doc!r}"
        )

    def test_docstring_one_liner_says_timezone_aware(self):
        """parse_utc one-liner must contain 'timezone-aware' and not 'ensure it has UTC timezone'."""
        from dashboard.data.utils import parse_utc

        doc = parse_utc.__doc__ or ''
        one_liner = doc.strip().splitlines()[0]
        assert 'timezone-aware' in one_liner, (
            f"Expected one-liner to contain 'timezone-aware', got: {one_liner!r}"
        )
        assert 'ensure it has UTC timezone' not in one_liner, (
            f"Misleading phrase 'ensure it has UTC timezone' still in one-liner: {one_liner!r}"
        )


class TestTimeagoUsesParseUtc:
    """Tests verifying that app.py uses parse_utc from dashboard.data.utils (DRY)."""

    def test_naive_timestamp_timeago_works(self):
        """timeago with a naive ISO timestamp (no tzinfo) returns the correct relative string."""
        from datetime import UTC, datetime, timedelta

        from dashboard.app import timeago

        # Use a dynamic timestamp 5 minutes in the past to avoid hardcoded date rot.
        # timedelta(minutes=5) yields exactly 300s, so total_minutes == 5 exactly.
        ts = (datetime.now(UTC) - timedelta(minutes=5)).replace(tzinfo=None).isoformat()
        assert timeago(ts) == '5m ago'

    def test_non_utc_offset_timeago_uses_astimezone(self):
        """timeago with a non-UTC-offset timestamp returns the correct relative time string.

        The timestamp '5 minutes ago' expressed in +05:30 offset must still
        produce '5m ago'.  This verifies that .astimezone(UTC) is applied
        (consistent with the _ts_sort_key pattern) and that cross-tz arithmetic
        is correct.
        """
        from datetime import UTC, datetime, timedelta, timezone

        from dashboard.app import timeago

        # Build a timestamp 5 minutes ago expressed in IST (+05:30).
        ist = timezone(timedelta(hours=5, minutes=30))
        five_min_ago_utc = datetime.now(UTC) - timedelta(minutes=5)
        five_min_ago_ist = five_min_ago_utc.astimezone(ist)
        ts = five_min_ago_ist.isoformat()
        assert timeago(ts) == '5m ago'

