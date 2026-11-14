"""Contract tests for ``shared.escalation_layout``."""

from __future__ import annotations

from pathlib import Path

from shared import escalation_layout


def test_layout_constants_are_the_on_disk_names():
    assert escalation_layout.ARCHIVE_SUBDIR == 'archive'
    assert escalation_layout.DATE_FORMAT == '%Y-%m-%d'


def test_terminal_status_names_are_resolved_and_dismissed():
    assert escalation_layout.RESOLVED == 'resolved'
    assert escalation_layout.DISMISSED == 'dismissed'
    assert {'resolved', 'dismissed'} == escalation_layout.TERMINAL_STATUS_NAMES


class TestArchiveDirForDate:
    def test_tz_aware_iso_returns_path_without_creating_it(self, tmp_path: Path):
        result = escalation_layout.archive_dir_for_date(
            tmp_path, '2026-04-23T10:29:11.225972+00:00'
        )
        assert result == tmp_path / 'archive' / '2026-04-23'
        assert not result.exists()

    def test_naive_iso_is_taken_as_utc(self, tmp_path: Path):
        result = escalation_layout.archive_dir_for_date(tmp_path, '2026-04-23T10:29:11')
        assert result == tmp_path / 'archive' / '2026-04-23'

    def test_non_utc_tz_bucketed_by_utc_date(self, tmp_path: Path):
        # 2026-04-24T01:00+05:00 is 2026-04-23T20:00 UTC.
        result = escalation_layout.archive_dir_for_date(
            tmp_path, '2026-04-24T01:00:00+05:00'
        )
        assert result == tmp_path / 'archive' / '2026-04-23'
