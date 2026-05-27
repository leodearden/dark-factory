"""Tests for escalation.sweep module."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from escalation import sweep
from escalation.models import Escalation


def _write_root_esc(
    queue_dir: Path,
    id: str,
    status: str,
    resolved_at: str | None = None,
    resolved_by: str | None = None,
    resolution_turns: int | None = None,
) -> Path:
    """Write a minimal esc-*.json directly to queue root."""
    esc = Escalation(
        id=id,
        task_id='1',
        agent_role='test',
        severity='info',
        category='cleanup_needed',
        summary='test escalation',
        status=status,
        resolved_at=resolved_at,
        resolved_by=resolved_by,
        resolution_turns=resolution_turns,
    )
    path = queue_dir / f'{id}.json'
    path.write_text(esc.to_json())
    return path


class TestSweepBasic:
    """Classification and basic move behavior on apply=True."""

    def test_resolved_in_root_moved_to_dated_archive_on_apply(self, tmp_path: Path):
        _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved', resolved_at='2026-05-20T10:00:00+00:00'
        )
        report = sweep.sweep(tmp_path, apply=True)
        assert (tmp_path / 'archive' / '2026-05-20' / 'esc-1-1.json').exists()
        assert not (tmp_path / 'esc-1-1.json').exists()
        assert report.archived == 1

    def test_dismissed_in_root_moved_to_dated_archive_on_apply(self, tmp_path: Path):
        _write_root_esc(
            tmp_path, 'esc-2-1', 'dismissed', resolved_at='2026-05-21T08:00:00+00:00'
        )
        report = sweep.sweep(tmp_path, apply=True)
        assert (tmp_path / 'archive' / '2026-05-21' / 'esc-2-1.json').exists()
        assert not (tmp_path / 'esc-2-1.json').exists()
        assert report.archived == 1

    def test_pending_in_root_untouched_on_apply(self, tmp_path: Path):
        _write_root_esc(tmp_path, 'esc-3-1', 'pending')
        report = sweep.sweep(tmp_path, apply=True)
        assert (tmp_path / 'esc-3-1.json').exists()
        assert not (tmp_path / 'archive').exists()
        assert report.untouched_pending == 1
        assert report.archived == 0

    def test_report_counts_archived_and_pending_on_apply(self, tmp_path: Path):
        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at='2026-05-20T10:00:00+00:00')
        _write_root_esc(tmp_path, 'esc-1-2', 'resolved', resolved_at='2026-05-20T11:00:00+00:00')
        _write_root_esc(tmp_path, 'esc-2-1', 'dismissed', resolved_at='2026-05-21T08:00:00+00:00')
        _write_root_esc(tmp_path, 'esc-3-1', 'pending')
        report = sweep.sweep(tmp_path, apply=True)
        assert report.archived == 3
        assert report.untouched_pending == 1
        assert report.root_before == 4
        assert report.root_after == 1
        assert report.reconciled_root_wins == 0
        assert report.reconciled_archive_wins == 0


class TestSweepDryRunDefault:
    """apply=False (default) leaves disk untouched but populates report with would-do counts."""

    def test_dry_run_default_does_not_move_files(self, tmp_path: Path):
        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at='2026-05-20T10:00:00+00:00')
        _write_root_esc(tmp_path, 'esc-1-2', 'resolved', resolved_at='2026-05-20T11:00:00+00:00')
        _write_root_esc(tmp_path, 'esc-2-1', 'dismissed', resolved_at='2026-05-21T08:00:00+00:00')
        report = sweep.sweep(tmp_path)  # default: apply=False
        # All 3 root files still present
        assert (tmp_path / 'esc-1-1.json').exists()
        assert (tmp_path / 'esc-1-2.json').exists()
        assert (tmp_path / 'esc-2-1.json').exists()
        # No archive dir created
        assert not (tmp_path / 'archive').exists()
        # But report says 3 would move
        assert report.archived == 3

    def test_dry_run_root_after_is_projected(self, tmp_path: Path):
        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at='2026-05-20T10:00:00+00:00')
        _write_root_esc(tmp_path, 'esc-1-2', 'resolved', resolved_at='2026-05-20T11:00:00+00:00')
        _write_root_esc(tmp_path, 'esc-2-1', 'dismissed', resolved_at='2026-05-21T08:00:00+00:00')
        report = sweep.sweep(tmp_path)
        assert report.root_before == 3
        assert report.root_after == 0  # projected: all 3 would move

    def test_apply_false_explicit_same_as_default(self, tmp_path: Path):
        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at='2026-05-20T10:00:00+00:00')
        report_default = sweep.sweep(tmp_path)
        # reset: file still there since dry-run
        report_explicit = sweep.sweep(tmp_path, apply=False)
        assert report_default.archived == report_explicit.archived
        assert report_default.root_before == report_explicit.root_before
        assert report_default.root_after == report_explicit.root_after
        # Both leave disk unchanged
        assert (tmp_path / 'esc-1-1.json').exists()


def _write_archive_esc(
    queue_dir: Path,
    id: str,
    resolved_at: str,
    status: str,
    resolved_by: str | None = None,
    resolution_turns: int | None = None,
) -> Path:
    """Write a minimal esc-*.json directly into queue_dir/archive/YYYY-MM-DD/<id>.json."""
    esc = Escalation(
        id=id,
        task_id='1',
        agent_role='test',
        severity='info',
        category='cleanup_needed',
        summary='test escalation',
        status=status,
        resolved_at=resolved_at,
        resolved_by=resolved_by,
        resolution_turns=resolution_turns,
    )
    from escalation import archive as _archive
    dated_dir = _archive.archive_dir_for_date(queue_dir, resolved_at)
    dated_dir.mkdir(parents=True, exist_ok=True)
    path = dated_dir / f'{id}.json'
    path.write_text(esc.to_json())
    return path


class TestSweepReconciliation:
    """When an archive copy already exists, pick the richer record."""

    RESOLVED_AT = '2026-05-20T10:00:00+00:00'

    def test_root_richer_wins_overwrites_archive_content(self, tmp_path: Path):
        """Root has resolved_by='steward', archive has resolved_by=None → root wins."""
        archive_path = _write_archive_esc(
            tmp_path, 'esc-1-1', self.RESOLVED_AT, 'resolved', resolved_by=None
        )
        root_path = _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved',
            resolved_at=self.RESOLVED_AT, resolved_by='steward'
        )
        report = sweep.sweep(tmp_path, apply=True)
        # Archive file still at its original path, now has root content
        assert archive_path.exists()
        assert not root_path.exists()
        winner = Escalation.from_json(archive_path.read_text())
        assert winner.resolved_by == 'steward'
        assert report.reconciled_root_wins == 1
        assert report.archived == 0
        assert report.reconciled_archive_wins == 0

    def test_archive_richer_wins_drops_root(self, tmp_path: Path):
        """Archive has resolved_by='steward', root has resolved_by=None → archive wins."""
        archive_path = _write_archive_esc(
            tmp_path, 'esc-1-1', self.RESOLVED_AT, 'resolved', resolved_by='steward'
        )
        root_path = _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved',
            resolved_at=self.RESOLVED_AT, resolved_by=None
        )
        report = sweep.sweep(tmp_path, apply=True)
        assert archive_path.exists()
        assert not root_path.exists()
        # Archive content is unchanged
        winner = Escalation.from_json(archive_path.read_text())
        assert winner.resolved_by == 'steward'
        assert report.reconciled_archive_wins == 1
        assert report.reconciled_root_wins == 0

    def test_tied_richness_archive_wins_drops_root(self, tmp_path: Path):
        """Both resolved_by=None, resolution_turns=None → tie → archive wins."""
        archive_path = _write_archive_esc(
            tmp_path, 'esc-1-1', self.RESOLVED_AT, 'resolved',
            resolved_by=None, resolution_turns=None
        )
        root_path = _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved',
            resolved_at=self.RESOLVED_AT, resolved_by=None, resolution_turns=None
        )
        report = sweep.sweep(tmp_path, apply=True)
        assert archive_path.exists()
        assert not root_path.exists()
        assert report.reconciled_archive_wins == 1

    def test_resolution_turns_tiebreaker_when_resolved_by_tied(self, tmp_path: Path):
        """Both have resolved_by='steward'; root has resolution_turns=5, archive=None → root wins."""
        archive_path = _write_archive_esc(
            tmp_path, 'esc-1-1', self.RESOLVED_AT, 'resolved',
            resolved_by='steward', resolution_turns=None
        )
        root_path = _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved',
            resolved_at=self.RESOLVED_AT, resolved_by='steward', resolution_turns=5
        )
        report = sweep.sweep(tmp_path, apply=True)
        assert archive_path.exists()
        assert not root_path.exists()
        winner = Escalation.from_json(archive_path.read_text())
        assert winner.resolution_turns == 5
        assert report.reconciled_root_wins == 1

    def test_dry_run_reconciliation_leaves_both_copies_intact(self, tmp_path: Path):
        """Dry-run for root-wins case: both files remain, report shows reconciled_root_wins=1."""
        archive_path = _write_archive_esc(
            tmp_path, 'esc-1-1', self.RESOLVED_AT, 'resolved', resolved_by=None
        )
        root_path = _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved',
            resolved_at=self.RESOLVED_AT, resolved_by='steward'
        )
        archive_orig = archive_path.read_text()
        root_orig = root_path.read_text()
        report = sweep.sweep(tmp_path, apply=False)
        # Both files untouched
        assert archive_path.read_text() == archive_orig
        assert root_path.read_text() == root_orig
        # But report says root would win
        assert report.reconciled_root_wins == 1
