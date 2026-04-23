"""Tests for escalation.archive module."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from escalation import archive


class TestArchiveDirForDate:
    """archive_dir_for_date() computes the archive subdirectory for a resolved_at timestamp."""

    def test_tz_aware_iso_returns_correct_path(self, tmp_path: Path):
        """Timezone-aware ISO string yields archive/YYYY-MM-DD subpath without creating it."""
        result = archive.archive_dir_for_date(
            tmp_path, '2026-04-23T10:29:11.225972+00:00'
        )
        expected = tmp_path / 'archive' / '2026-04-23'
        assert result == expected
        # Must NOT create the directory
        assert not expected.exists()

    def test_naive_iso_returns_correct_path(self, tmp_path: Path):
        """Naive ISO string (no timezone) also yields the correct YYYY-MM-DD path."""
        result = archive.archive_dir_for_date(tmp_path, '2026-04-23T10:29:11')
        expected = tmp_path / 'archive' / '2026-04-23'
        assert result == expected
