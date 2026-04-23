"""Tests for escalation.archive module."""

from __future__ import annotations

import json
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


def _make_dummy_esc_file(directory: Path, name: str = 'esc-1-1.json') -> Path:
    """Create a placeholder esc-*.json in *directory* and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps({'id': name.removesuffix('.json'), 'task_id': '1'}))
    return path


class TestPruneArchive:
    """archive.prune_archive() removes dated subdirs older than retention_days."""

    def _now(self) -> datetime:
        return datetime(2026, 4, 23, tzinfo=UTC)

    def _make_archive_subdirs(self, tmp_path: Path, dates: list[str]) -> None:
        """Create archive/<date>/esc-1-1.json for each date string."""
        for date in dates:
            _make_dummy_esc_file(tmp_path / 'archive' / date)

    def test_prunes_subdirs_older_than_retention(self, tmp_path: Path):
        """Subdirs more than retention_days old are removed."""
        self._make_archive_subdirs(
            tmp_path, ['2026-03-01', '2026-03-15', '2026-04-20', '2026-04-23']
        )

        archive.prune_archive(tmp_path, retention_days=30, now=self._now())

        assert not (tmp_path / 'archive' / '2026-03-01').exists()
        assert not (tmp_path / 'archive' / '2026-03-15').exists()
        assert (tmp_path / 'archive' / '2026-04-20').exists()
        assert (tmp_path / 'archive' / '2026-04-23').exists()

    def test_keeps_recent_subdirs(self, tmp_path: Path):
        """Subdirs exactly at the retention boundary are kept (not-strict less-than)."""
        # 2026-04-23 - 30 days = 2026-03-24; boundary subdir should be kept
        self._make_archive_subdirs(tmp_path, ['2026-03-24', '2026-03-23'])

        archive.prune_archive(tmp_path, retention_days=30, now=self._now())

        # Exactly at cutoff (parsed date == cutoff): not strictly less-than → kept
        assert (tmp_path / 'archive' / '2026-03-24').exists()
        # One day before cutoff: strictly older → removed
        assert not (tmp_path / 'archive' / '2026-03-23').exists()

    def test_skips_non_date_entries(self, tmp_path: Path):
        """Non-YYYY-MM-DD entries (files, dirs) are left untouched."""
        archive_dir = tmp_path / 'archive'
        archive_dir.mkdir(parents=True)
        (archive_dir / 'README.md').write_text('notes')
        loose = archive_dir / 'esc-99-1.json'
        loose.write_text('{}')
        weird_dir = archive_dir / 'weird-dir'
        weird_dir.mkdir()

        archive.prune_archive(tmp_path, retention_days=30, now=self._now())

        assert (archive_dir / 'README.md').exists()
        assert loose.exists()
        assert weird_dir.exists()

    def test_returns_pruned_count(self, tmp_path: Path):
        """prune_archive() returns the number of dirs removed."""
        self._make_archive_subdirs(
            tmp_path, ['2026-03-01', '2026-03-15', '2026-04-20']
        )

        count = archive.prune_archive(tmp_path, retention_days=30, now=self._now())

        assert count == 2  # 2026-03-01 and 2026-03-15 removed
