"""Tests for CLI helpers."""

import pytest

from orchestrator.cli import _parse_duration


class TestParseDuration:
    def test_hours(self):
        assert _parse_duration("4h") == 14400

    def test_minutes(self):
        assert _parse_duration("30m") == 1800

    def test_seconds(self):
        assert _parse_duration("90s") == 90

    def test_bare_number(self):
        assert _parse_duration("3600") == 3600

    def test_uppercase(self):
        assert _parse_duration("2H") == 7200

    def test_whitespace(self):
        assert _parse_duration("  10m  ") == 600

    def test_invalid(self):
        with pytest.raises(ValueError):
            _parse_duration("abc")
