"""Tests for shared.safe_io.load_json_or_warn."""

from __future__ import annotations

import json
import logging

import pytest

SENTINEL = object()  # unique default marker for assertions


@pytest.fixture(autouse=True)
def clear_warned_paths():
    """Clear the per-process dedup set before and after each test.

    Mirrors fused-memory/tests/test_sqlite_task_backend.py lines 44-48 to
    ensure warn-once assertions are deterministic across test isolation.
    """
    import shared.safe_io as _safe_io

    _safe_io._warned_corrupt_paths.clear()
    yield
    _safe_io._warned_corrupt_paths.clear()


class TestBenignPaths:
    """(·, True) paths: valid JSON and missing file are both benign."""

    def test_valid_json_returns_parsed_and_true(self, tmp_path, caplog):
        """Valid JSON file returns (parsed_obj, True) with no WARNING."""
        from shared.safe_io import load_json_or_warn

        data = {'key': 'value', 'num': 42}
        p = tmp_path / 'state.json'
        p.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL)

        assert ok is True
        assert result == data
        assert result is not SENTINEL
        assert len(caplog.records) == 0, f'Expected no WARNING, got: {caplog.records}'

    def test_missing_file_returns_default_and_true_silently(self, tmp_path, caplog):
        """Nonexistent path returns (default, True) with no log records at all."""
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'nonexistent.json'
        assert not p.exists()

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL)

        assert ok is True
        assert result is SENTINEL
        assert len(caplog.records) == 0, f'Expected silent return, got: {caplog.records}'


class TestCorruptWarnMode:
    """Default on_corrupt='warn': return (default, False) + one WARNING."""

    def test_corrupt_file_returns_default_false_and_warns(self, tmp_path, caplog):
        """Corrupt JSON returns (SENTINEL, False) with exactly one WARNING naming the path."""
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'state.json'
        p.write_bytes(b'{not json')

        with caplog.at_level(logging.WARNING):
            result, ok = load_json_or_warn(p, default=SENTINEL)

        assert ok is False
        assert result is SENTINEL
        assert len(caplog.records) == 1, f'Expected exactly one WARNING, got: {caplog.records}'
        assert str(p) in caplog.records[0].message


class TestOnCorruptDispatch:
    """on_corrupt dispatch: fail_closed raises, unknown mode raises ValueError."""

    def test_fail_closed_raises_and_emits_no_warning(self, tmp_path, caplog):
        """on_corrupt='fail_closed' raises ValueError (JSONDecodeError subclass) with NO WARNING."""
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'state.json'
        p.write_bytes(b'{not json')

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError):
                load_json_or_warn(p, default=SENTINEL, on_corrupt='fail_closed')

        assert len(caplog.records) == 0, (
            f'fail_closed must emit NO WARNING (exception is the loud channel), '
            f'got: {caplog.records}'
        )

    def test_unknown_on_corrupt_raises_value_error(self, tmp_path, caplog):
        """Unknown on_corrupt value raises ValueError (unknown-mode guard)."""
        from shared.safe_io import load_json_or_warn

        p = tmp_path / 'state.json'
        p.write_bytes(b'{not json')

        with pytest.raises(ValueError, match='unknown on_corrupt'):
            load_json_or_warn(p, default=SENTINEL, on_corrupt='bogus')
