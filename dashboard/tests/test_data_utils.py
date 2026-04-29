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


class TestSafeGatherResult:
    """Tests for the safe_gather_result public helper in dashboard.data.utils."""

    def test_returns_value_when_not_exception(self):
        """A non-exception result is returned as-is (cast to _T)."""
        from dashboard.data.utils import safe_gather_result

        result = safe_gather_result({'key': 'value'}, {}, 'myop')
        assert result == {'key': 'value'}

    def test_returns_default_when_exception(self):
        """An Exception result returns the declared default without re-raising."""
        from dashboard.data.utils import safe_gather_result

        default = {'labels': [], 'values': []}
        result = safe_gather_result(RuntimeError('boom'), default, 'myop')
        assert result is default

    def test_logs_warning_with_label_on_exception(self, caplog):
        """An Exception result emits exactly one WARNING under dashboard.data.utils containing the label."""
        import logging

        from dashboard.data.utils import safe_gather_result

        with caplog.at_level(logging.WARNING, logger='dashboard.data.utils'):
            safe_gather_result(RuntimeError('something went wrong'), 'default', 'myop')

        records = [r for r in caplog.records if r.name == 'dashboard.data.utils']
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert 'myop' in records[0].getMessage()

    def test_reraises_cancelled_error(self):
        """asyncio.CancelledError (BaseException, not Exception) must propagate, not be swallowed."""
        import asyncio

        import pytest

        from dashboard.data.utils import safe_gather_result

        with pytest.raises(asyncio.CancelledError):
            safe_gather_result(asyncio.CancelledError(), 'default', 'myop')

    def test_reraises_keyboard_interrupt(self):
        """KeyboardInterrupt (BaseException, not Exception) must propagate, not be swallowed."""
        import pytest

        from dashboard.data.utils import safe_gather_result

        with pytest.raises(KeyboardInterrupt):
            safe_gather_result(KeyboardInterrupt(), 'default', 'myop')

    def test_reraises_system_exit(self):
        """SystemExit (BaseException, not Exception) must propagate, not be swallowed."""
        import pytest

        from dashboard.data.utils import safe_gather_result

        with pytest.raises(SystemExit):
            safe_gather_result(SystemExit(0), 'default', 'myop')


