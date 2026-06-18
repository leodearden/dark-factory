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
