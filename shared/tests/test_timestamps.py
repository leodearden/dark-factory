"""Tests for shared.timestamps — parse_timestamp_or_warn primitive.

Covers:
  - TestSuccessPath: valid tz-aware and naive ISO strings parse correctly,
    no WARNING emitted.  Includes 'Z' (Zulu) suffix and non-UTC offset cases.
  - TestFailureFallback: None / non-str / malformed strings all return the
    sortable UTC sentinel and emit exactly one WARNING carrying context.
  - TestCallerFallback: caller-supplied ``fallback`` is returned on failure
    instead of the sentinel.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

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

    def test_zulu_suffix_parsed_as_utc(self, caplog):
        """'Z' (Zulu) suffix is handled natively on Python 3.11+ (no pre-processing needed)."""
        raw = '2026-06-18T12:00:00Z'
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn(raw)
        assert ok is True
        assert dt == datetime(2026, 6, 18, 12, 0, tzinfo=UTC)
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records == []

    def test_non_utc_offset_preserved_not_converted(self, caplog):
        """Tz-aware inputs with a non-UTC offset keep their original offset (NOT converted to UTC)."""
        raw = '2026-06-18T12:00:00+05:00'
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn(raw)
        assert ok is True
        assert dt.utcoffset() == timedelta(hours=5)
        # Must NOT have been silently converted to UTC
        assert dt.tzinfo != UTC
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records == []

    def test_success_emits_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn('2026-06-18T12:00:00+00:00')
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records == []


_DEFAULT_SENTINEL = datetime.min.replace(tzinfo=UTC)


class TestFailureFallback:
    """Every failure mode returns the sortable UTC sentinel and emits a WARNING."""

    def test_malformed_string_returns_sentinel(self, caplog):
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn('not-a-timestamp')
        assert ok is False
        assert dt == _DEFAULT_SENTINEL

    def test_none_returns_sentinel(self, caplog):
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn(None)
        assert ok is False
        assert dt == _DEFAULT_SENTINEL

    def test_non_str_int_returns_sentinel(self, caplog):
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn(12345)
        assert ok is False
        assert dt == _DEFAULT_SENTINEL

    def test_malformed_string_emits_exactly_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn('not-a-timestamp')
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1

    def test_none_emits_exactly_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn(None)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1

    def test_non_str_emits_exactly_one_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn(12345)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1

    def test_warning_message_contains_context(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn('bad', context='dedupe.find_parent')
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert 'dedupe.find_parent' in warnings[0].getMessage()

    def test_warning_message_contains_repr_of_raw(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn('bad-value', context='ctx')
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert repr('bad-value') in warnings[0].getMessage()

    def test_warning_message_contains_context_for_none(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn(None, context='dedupe.find_parent')
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert 'dedupe.find_parent' in warnings[0].getMessage()

    def test_sortable_sentinel_sorts_oldest(self):
        """Malformed sentinel sorts BEFORE any real timestamp — 'never silently dropped'."""
        valid_raw = '2026-06-18T12:00:00+00:00'
        dt_valid, _ = parse_timestamp_or_warn(valid_raw)
        dt_bad, _ = parse_timestamp_or_warn('garbage')
        mixed = [dt_valid, dt_bad]
        mixed.sort()
        # The sentinel (datetime.min) must sort first — before the real timestamp
        assert mixed[0] == _DEFAULT_SENTINEL
        assert mixed[1] == dt_valid


class TestCallerFallback:
    """Caller-supplied ``fallback`` is returned on failure instead of the default sentinel."""

    _CUSTOM_FB = datetime(2000, 1, 1, tzinfo=UTC)

    def test_custom_fallback_returned_for_malformed_string(self, caplog):
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn('bad', fallback=self._CUSTOM_FB)
        assert ok is False
        assert dt == self._CUSTOM_FB

    def test_custom_fallback_returned_for_none(self, caplog):
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn(None, fallback=self._CUSTOM_FB)
        assert ok is False
        assert dt == self._CUSTOM_FB

    def test_custom_fallback_still_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_timestamp_or_warn('bad', fallback=self._CUSTOM_FB)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1

    def test_none_fallback_uses_default_sentinel(self, caplog):
        """Explicit fallback=None → still use datetime.min sentinel (regression guard)."""
        with caplog.at_level(logging.WARNING):
            dt, ok = parse_timestamp_or_warn('bad', fallback=None)
        assert ok is False
        assert dt == _DEFAULT_SENTINEL
