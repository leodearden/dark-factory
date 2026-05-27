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
