"""Tests for escalation.sweep module."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
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

    def test_dry_run_archive_wins_leaves_both_copies_intact(self, tmp_path: Path):
        """Dry-run for archive-wins case: both files remain, report shows reconciled_archive_wins=1."""
        archive_path = _write_archive_esc(
            tmp_path, 'esc-1-1', self.RESOLVED_AT, 'resolved', resolved_by='steward'
        )
        root_path = _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved',
            resolved_at=self.RESOLVED_AT, resolved_by=None
        )
        archive_orig = archive_path.read_text()
        root_orig = root_path.read_text()
        report = sweep.sweep(tmp_path, apply=False)
        # Both files untouched
        assert archive_path.exists()
        assert root_path.exists()
        assert archive_path.read_text() == archive_orig
        assert root_path.read_text() == root_orig
        # But report says archive would win
        assert report.reconciled_archive_wins == 1
        assert report.reconciled_root_wins == 0


class TestSweepSafety:
    """Non-esc files, unparsable JSON, and missing resolved_at are handled safely."""

    RESOLVED_AT = '2026-05-20T10:00:00+00:00'

    def test_wal_and_shm_files_never_touched(self, tmp_path: Path):
        """Non-esc-*.json files in queue root are ignored; esc file is archived."""
        wal = tmp_path / 'escalations.db-wal'
        shm = tmp_path / 'escalations.db-shm'
        notes = tmp_path / 'notes.txt'
        wal.write_text('wal data')
        shm.write_text('shm data')
        notes.write_text('notes')
        mtime_wal = wal.stat().st_mtime
        mtime_shm = shm.stat().st_mtime
        mtime_notes = notes.stat().st_mtime

        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at=self.RESOLVED_AT)
        sweep.sweep(tmp_path, apply=True)

        assert wal.exists() and wal.stat().st_mtime == mtime_wal
        assert shm.exists() and shm.stat().st_mtime == mtime_shm
        assert notes.exists() and notes.stat().st_mtime == mtime_notes
        assert (tmp_path / 'archive' / '2026-05-20' / 'esc-1-1.json').exists()

    def test_unparsable_json_in_root_left_alone(self, tmp_path: Path):
        """Unparsable JSON file stays in root and increments skipped_unparsable."""
        bad = tmp_path / 'esc-99-1.json'
        bad.write_text('{not valid json')
        report = sweep.sweep(tmp_path, apply=True)
        assert bad.exists()
        assert report.skipped_unparsable == 1
        assert report.archived == 0

    def test_missing_resolved_at_on_resolved_file_skipped(self, tmp_path: Path):
        """resolved file with resolved_at=None stays in root, counted as skipped."""
        path = _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at=None)
        report = sweep.sweep(tmp_path, apply=True)
        assert path.exists()
        assert report.skipped_unparsable == 1
        assert report.archived == 0

    def test_unknown_status_value_skipped(self, tmp_path: Path):
        """File with unknown status does not get archived and is counted as skipped."""
        esc = Escalation(
            id='esc-1-1', task_id='1', agent_role='test',
            severity='info', category='cleanup_needed', summary='test',
            status='weird',
        )
        path = tmp_path / 'esc-1-1.json'
        path.write_text(esc.to_json())
        report = sweep.sweep(tmp_path, apply=True)
        assert path.exists()
        assert report.archived == 0
        assert report.skipped_unparsable == 1
        assert not (tmp_path / 'archive').exists()


class TestSweepIdempotency:
    """Running sweep(apply=True) twice leaves disk state unchanged on the second call."""

    def _disk_snapshot(self, queue_dir: Path) -> dict[str, bytes]:
        """Return {relative_path_str: content_bytes} for every file under queue_dir."""
        return {
            str(p.relative_to(queue_dir)): p.read_bytes()
            for p in sorted(queue_dir.rglob('*'))
            if p.is_file()
        }

    def test_double_apply_is_noop(self, tmp_path: Path):
        RESOLVED_AT_A = '2026-05-20T10:00:00+00:00'
        RESOLVED_AT_B = '2026-05-21T08:00:00+00:00'

        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at=RESOLVED_AT_A)
        _write_root_esc(tmp_path, 'esc-1-2', 'resolved', resolved_at=RESOLVED_AT_A)
        _write_root_esc(tmp_path, 'esc-2-1', 'dismissed', resolved_at=RESOLVED_AT_B)
        _write_root_esc(tmp_path, 'esc-3-1', 'pending')
        # Archive-only file with no root copy (should be untouched by both calls)
        _write_archive_esc(tmp_path, 'esc-4-1', RESOLVED_AT_A, 'resolved', resolved_by='steward')

        report1 = sweep.sweep(tmp_path, apply=True)
        assert report1.archived > 0 or report1.reconciled_root_wins > 0 or report1.reconciled_archive_wins > 0

        snapshot = self._disk_snapshot(tmp_path)

        report2 = sweep.sweep(tmp_path, apply=True)

        assert report2.archived == 0
        assert report2.reconciled_root_wins == 0
        assert report2.reconciled_archive_wins == 0
        assert report2.untouched_pending == 1
        assert self._disk_snapshot(tmp_path) == snapshot


class TestStartupSweepIdempotency:
    """run_startup_sweep is idempotent: first run clears the backlog, second is a no-op."""

    _NOW = datetime(2026, 6, 4, tzinfo=UTC)

    def _disk_snapshot(self, root: Path) -> dict[str, bytes]:
        """Return {relative_path_str: content_bytes} for every regular file under root."""
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob('*'))
            if p.is_file()
        }

    def test_first_run_clears_backlog_second_run_is_noop(self, tmp_path: Path):
        """Seed a mix: resolved + dismissed root escs, loose archive esc, pending esc."""
        # Resolved root esc
        _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved', resolved_at='2026-05-20T10:00:00+00:00'
        )
        # Dismissed root esc
        _write_root_esc(
            tmp_path, 'esc-2-1', 'dismissed', resolved_at='2026-05-21T08:00:00+00:00'
        )
        # Pending esc (must remain in root)
        _write_root_esc(tmp_path, 'esc-3-1', 'pending')
        # Loose archive esc
        archive_root = tmp_path / 'archive'
        archive_root.mkdir(parents=True)
        loose = archive_root / 'esc-4-1.json'
        loose_esc = Escalation(
            id='esc-4-1',
            task_id='1',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='loose',
            status='resolved',
            resolved_at='2026-05-22T06:00:00+00:00',
        )
        loose.write_text(loose_esc.to_json())

        # Run 1: should clean up the backlog
        report1 = sweep.run_startup_sweep(tmp_path, now=self._NOW)
        assert report1.sweep.archived >= 2, 'resolved + dismissed should be archived'
        assert report1.loose_reaped == 1, 'loose archive esc should be reaped'

        # After run 1: only the pending esc remains in queue root (+ lock sidecars)
        root_escs = list(tmp_path.glob('esc-*.json'))
        assert len(root_escs) == 1, f'Only pending esc should remain: {root_escs}'
        assert root_escs[0].name == 'esc-3-1.json'

        # Capture disk state after run 1
        snapshot_after_run1 = self._disk_snapshot(tmp_path)

        # Run 2: should be a complete no-op
        report2 = sweep.run_startup_sweep(tmp_path, now=self._NOW)
        assert report2.sweep.archived == 0, 'no new root escs to archive'
        assert report2.loose_reaped == 0, 'no loose archive escs to reap'
        assert report2.pruned_dirs == 0, 'no dirs to prune (all recent)'

        # Disk state byte-stable between the two post-run states
        assert self._disk_snapshot(tmp_path) == snapshot_after_run1, (
            'Disk state changed on second run_startup_sweep — not idempotent!'
        )


class TestD6GlobInvariant:
    """D6 HARD-INVARIANT regression: non-esc-* root files are NEVER touched by a sweep pass."""

    _NOW = datetime(2026, 6, 4, tzinfo=UTC)
    _RESOLVED_AT = '2026-05-20T10:00:00+00:00'

    def test_non_esc_files_untouched_by_run_startup_sweep(self, tmp_path: Path):
        """b3-state.json (PRD-2) and afk-digest.md in root survive a full startup sweep."""
        # Non-esc residents
        b3_state = tmp_path / 'b3-state.json'
        afk_digest = tmp_path / 'afk-digest.md'
        b3_state.write_bytes(b'{"state": "active"}')
        afk_digest.write_bytes(b'# AFK digest\n\nSome content here.')

        b3_bytes = b3_state.read_bytes()
        afk_bytes = afk_digest.read_bytes()
        b3_mtime = b3_state.stat().st_mtime
        afk_mtime = afk_digest.stat().st_mtime

        # One resolved esc that SHOULD be relocated
        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at=self._RESOLVED_AT)

        sweep.run_startup_sweep(tmp_path, now=self._NOW)

        # Non-esc files completely unchanged
        assert b3_state.exists(), 'b3-state.json was deleted by sweep — glob widened!'
        assert afk_digest.exists(), 'afk-digest.md was deleted by sweep — glob widened!'
        assert b3_state.read_bytes() == b3_bytes, 'b3-state.json content changed'
        assert afk_digest.read_bytes() == afk_bytes, 'afk-digest.md content changed'
        assert b3_state.stat().st_mtime == b3_mtime, 'b3-state.json mtime changed'
        assert afk_digest.stat().st_mtime == afk_mtime, 'afk-digest.md mtime changed'

        # The resolved esc WAS relocated
        assert (tmp_path / 'archive' / '2026-05-20' / 'esc-1-1.json').exists()
        assert not (tmp_path / 'esc-1-1.json').exists()


class TestRunStartupSweep:
    """Tests for sweep.run_startup_sweep and StartupSweepReport."""

    _NOW = datetime(2026, 6, 4, tzinfo=UTC)
    _RESOLVED_AT_RECENT = '2026-05-20T10:00:00+00:00'
    _RESOLVED_AT_LOOSE = '2026-05-21T12:00:00+00:00'
    _RESOLVED_AT_STALE = '2026-01-01T00:00:00+00:00'

    def test_run_startup_sweep_archives_and_reaps_and_prunes(self, tmp_path: Path, caplog):
        """Full integration: root esc archived, loose esc relocated, stale dir pruned."""
        # One resolved root esc
        _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved', resolved_at=self._RESOLVED_AT_RECENT
        )
        # One loose archive esc (sits at archive top-level)
        archive_root = tmp_path / 'archive'
        archive_root.mkdir(parents=True)
        loose_path = archive_root / 'esc-2-1.json'
        loose_esc = Escalation(
            id='esc-2-1',
            task_id='1',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='loose',
            status='resolved',
            resolved_at=self._RESOLVED_AT_LOOSE,
        )
        loose_path.write_text(loose_esc.to_json())
        # One stale dated dir (older than 30 days relative to _NOW)
        stale_dir = archive_root / '2026-01-01'
        stale_dir.mkdir(parents=True)
        stale_file = stale_dir / 'esc-9-1.json'
        stale_file.write_text(loose_esc.to_json())

        with caplog.at_level(logging.INFO, logger='escalation.sweep'):
            report = sweep.run_startup_sweep(tmp_path, now=self._NOW)

        # Root esc archived
        assert (archive_root / '2026-05-20' / 'esc-1-1.json').exists()
        assert not (tmp_path / 'esc-1-1.json').exists()
        # Loose esc relocated
        assert (archive_root / '2026-05-21' / 'esc-2-1.json').exists()
        assert not loose_path.exists()
        # Stale dir pruned (2026-01-01 is >30 days before 2026-06-04)
        assert not stale_dir.exists()

        # StartupSweepReport fields
        assert report.sweep.archived >= 1
        assert report.loose_reaped == 1
        assert report.pruned_dirs == 1

        # INFO log line emitted on escalation.sweep logger
        assert any(
            r.name == 'escalation.sweep' and r.levelno == logging.INFO
            for r in caplog.records
        ), f'Expected INFO report line; got: {[r.getMessage() for r in caplog.records]}'


class TestReapLooseArchiveFiles:
    """Tests for sweep.reap_loose_archive_files."""

    def test_loose_resolved_file_moved_to_dated_subdir_with_lock(self, tmp_path: Path):
        """(a) Loose resolved file at archive top-level is moved to dated subdir."""
        archive_root = tmp_path / 'archive'
        archive_root.mkdir(parents=True)
        loose_path = archive_root / 'esc-1-1.json'
        esc = Escalation(
            id='esc-1-1',
            task_id='1',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='loose test',
            status='resolved',
            resolved_at='2026-05-20T10:00:00+00:00',
        )
        loose_path.write_text(esc.to_json())

        count = sweep.reap_loose_archive_files(tmp_path, apply=True)

        # Moved to dated subdir
        assert (archive_root / '2026-05-20' / 'esc-1-1.json').exists()
        # Loose copy gone
        assert not loose_path.exists()
        # Returns 1
        assert count == 1
        # Per-id sidecar lock was created in queue root
        assert (tmp_path / 'esc-1-1.json.lock').exists()

    def test_loose_file_without_resolved_at_left_in_place(self, tmp_path: Path):
        """(b) Loose file with resolved_at=None is left in place and NOT counted."""
        archive_root = tmp_path / 'archive'
        archive_root.mkdir(parents=True)
        loose_path = archive_root / 'esc-2-1.json'
        esc = Escalation(
            id='esc-2-1',
            task_id='1',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='no resolved_at',
            status='resolved',
            resolved_at=None,
        )
        loose_path.write_text(esc.to_json())

        count = sweep.reap_loose_archive_files(tmp_path, apply=True)

        # Left untouched
        assert loose_path.exists()
        # Not counted
        assert count == 0

    def test_apply_false_reports_count_without_moving(self, tmp_path: Path):
        """(c) apply=False leaves loose file untouched but reports would-move count."""
        archive_root = tmp_path / 'archive'
        archive_root.mkdir(parents=True)
        loose_path = archive_root / 'esc-3-1.json'
        esc = Escalation(
            id='esc-3-1',
            task_id='1',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='dry run loose',
            status='resolved',
            resolved_at='2026-05-20T10:00:00+00:00',
        )
        loose_path.write_text(esc.to_json())

        count = sweep.reap_loose_archive_files(tmp_path, apply=False)

        # File still in place
        assert loose_path.exists()
        # But count is 1 (would-move)
        assert count == 1


class TestSweepRelocationLock:
    """sweep.sweep relocations must take the per-id sidecar lock."""

    def test_relocation_creates_sidecar_lock_file(self, tmp_path: Path):
        """After apply=True relocation, the stable sidecar .lock file exists in queue root."""
        _write_root_esc(
            tmp_path, 'esc-1-1', 'resolved', resolved_at='2026-05-20T10:00:00+00:00'
        )
        sweep.sweep(tmp_path, apply=True)
        # (a) File was moved to dated archive subdir
        assert (tmp_path / 'archive' / '2026-05-20' / 'esc-1-1.json').exists()
        assert not (tmp_path / 'esc-1-1.json').exists()
        # (b) Stable sidecar .lock exists in queue root — proves lock was taken
        assert (tmp_path / 'esc-1-1.json.lock').exists(), (
            'Expected sidecar lock file esc-1-1.json.lock in queue root — '
            'sweep.sweep must take escalation_id_lock around relocations'
        )


class TestSweepCli:
    """CLI entry point: main() and __main__ guard."""

    RESOLVED_AT = '2026-05-20T10:00:00+00:00'

    def test_cli_dry_run_default_returns_0_and_does_not_move(self, tmp_path: Path, caplog):
        """main(['--queue-dir', ...]) returns 0, leaves files in root, logs DRY-RUN."""
        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at=self.RESOLVED_AT)
        with caplog.at_level(logging.INFO, logger='escalation.sweep'):
            result = sweep.main(['--queue-dir', str(tmp_path)])
        assert result == 0
        assert (tmp_path / 'esc-1-1.json').exists()
        assert not (tmp_path / 'archive').exists()
        assert any(
            'DRY-RUN' in r.getMessage() and 'archived=1' in r.getMessage()
            for r in caplog.records
        ), f'Expected DRY-RUN summary; got: {[r.getMessage() for r in caplog.records]}'

    def test_cli_apply_returns_0_and_moves(self, tmp_path: Path, caplog):
        """main(['--queue-dir', ..., '--apply']) returns 0 and archives the file."""
        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at=self.RESOLVED_AT)
        with caplog.at_level(logging.INFO, logger='escalation.sweep'):
            result = sweep.main(['--queue-dir', str(tmp_path), '--apply'])
        assert result == 0
        assert not (tmp_path / 'esc-1-1.json').exists()
        assert (tmp_path / 'archive' / '2026-05-20' / 'esc-1-1.json').exists()
        assert any(
            'APPLIED' in r.getMessage() for r in caplog.records
        ), f'Expected APPLIED summary; got: {[r.getMessage() for r in caplog.records]}'

    def test_cli_missing_queue_dir_returns_2_and_logs_error(self, tmp_path: Path, caplog):
        """main(['--queue-dir', '/no-such-dir']) returns 2 and logs error."""
        missing = tmp_path / 'no-such-dir'
        with caplog.at_level(logging.ERROR, logger='escalation.sweep'):
            result = sweep.main(['--queue-dir', str(missing)])
        assert result == 2
        assert any(
            'queue-dir does not exist' in r.getMessage() for r in caplog.records
        ), f'Expected missing-dir error; got: {[r.getMessage() for r in caplog.records]}'

