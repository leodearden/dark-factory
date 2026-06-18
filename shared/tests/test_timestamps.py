"""Tests for shared.timestamps — parse_timestamp_or_warn primitive.

Covers:
  - TestSuccessPath: valid tz-aware and naive ISO strings parse correctly,
    no WARNING emitted.
  - TestFailureFallback: None / non-str / malformed strings all return the
    sortable UTC sentinel and emit exactly one WARNING carrying context.
  - TestCallerFallback: caller-supplied ``fallback`` is returned on failure
    instead of the sentinel.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from shared.timestamps import parse_timestamp_or_warn


class TestSuccessPath:
    def test_tz_aware_iso_string_parses_correctly(self, caplog):
        raw = '2026-06-18T12:00:00+00:00'
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn(raw)
        assert ok is True
        assert dt == datetime(2026, 6, 18, 12, 0, tzinfo=UTC)
        assert dt.tzinfo is UTC or str(dt.tzinfo) in ('+00:00', 'UTC')

    def test_naive_iso_string_normalized_to_utc(self, caplog):
        raw = '2026-06-18T12:00:00'
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn(raw)
        assert ok is True
        assert dt.tzinfo is not None
        assert dt.tzinfo == UTC

    def test_success_emits_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn('2026-06-18T12:00:00+00:00')
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records == []
