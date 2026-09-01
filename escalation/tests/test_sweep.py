"""Tests for escalation.sweep module."""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from escalation import sweep
from escalation.models import Escalation
from escalation.queue import SEQ_COUNTER_SUFFIX


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


def _disk_snapshot(root: Path) -> dict[str, bytes]:
    """Return {relative_path_str: content_bytes} for every regular file under root.

    The byte-for-byte equality workhorse behind every "this pass mutated
    nothing" assertion — dry-run contracts, double-apply idempotency, and the
    reap pass's second-run no-op.
    """
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob('*'))
        if p.is_file()
    }


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

        snapshot = _disk_snapshot(tmp_path)

        report2 = sweep.sweep(tmp_path, apply=True)

        assert report2.archived == 0
        assert report2.reconciled_root_wins == 0
        assert report2.reconciled_archive_wins == 0
        assert report2.untouched_pending == 1
        assert _disk_snapshot(tmp_path) == snapshot


class TestStartupSweepIdempotency:
    """run_startup_sweep is idempotent: first run clears the backlog, second is a no-op."""

    _NOW = datetime(2026, 6, 4, tzinfo=UTC)

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
        snapshot_after_run1 = _disk_snapshot(tmp_path)

        # Run 2: should be a complete no-op
        report2 = sweep.run_startup_sweep(tmp_path, now=self._NOW)
        assert report2.sweep.archived == 0, 'no new root escs to archive'
        assert report2.loose_reaped == 0, 'no loose archive escs to reap'
        assert report2.pruned_dirs == 0, 'no dirs to prune (all recent)'

        # Disk state byte-stable between the two post-run states
        assert _disk_snapshot(tmp_path) == snapshot_after_run1, (
            'Disk state changed on second run_startup_sweep — not idempotent!'
        )


# The real non-esc residents of the production queue root (b3-state.* from
# PRD-2, the AFK digests), plus two hypothetical non-escalation SIDECARS that a
# bare ``*.json.lock`` glob would happily match.  ``b3-state.json.lock`` is
# shielded anyway by the root-record check (``b3-state.json`` is right there);
# ``sched-state.json.lock`` — a subsystem sidecar whose data file is absent — is
# the one that actually separates a ``*.json.lock`` glob from an
# ``esc-*.json.lock`` one, and so the one that fails first if D6 is weakened.
_NON_ESC_RESIDENTS: dict[str, bytes] = {
    'b3-state.json': b'{"state": "active"}',
    'b3-state.lock': b'',
    'afk-digest.md': b'# AFK digest\n\nSome content here.',
    'b3-state.json.lock': b'',
    'sched-state.json.lock': b'',
}


def _seed_non_esc_residents(queue_dir: Path) -> dict[str, tuple[bytes, float]]:
    """Write every non-esc root resident; return {name: (bytes, mtime)} to compare against."""
    queue_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _NON_ESC_RESIDENTS.items():
        (queue_dir / name).write_bytes(content)
    return {
        name: ((queue_dir / name).read_bytes(), (queue_dir / name).stat().st_mtime)
        for name in _NON_ESC_RESIDENTS
    }


def _assert_residents_unchanged(
    queue_dir: Path, before: dict[str, tuple[bytes, float]]
) -> None:
    """Assert every seeded non-esc resident survived byte- AND mtime-identical."""
    for name, (content, mtime) in before.items():
        path = queue_dir / name
        assert path.exists(), f'{name} was deleted by a sweep pass — glob widened!'
        assert path.read_bytes() == content, f'{name} content changed'
        assert path.stat().st_mtime == mtime, f'{name} mtime changed'


class TestD6GlobInvariant:
    """D6 HARD-INVARIANT regression: non-esc-* root files are NEVER touched by a sweep pass.

    Every pass that globs the queue root belongs here — the relocation pass and
    the sidecar-reap pass share one residents fixture so that extending the
    invariant (a new resident, a new foreign sidecar shape) covers both at once.
    """

    _NOW = datetime(2026, 6, 4, tzinfo=UTC)
    _RESOLVED_AT = '2026-05-20T10:00:00+00:00'

    def test_non_esc_files_untouched_by_run_startup_sweep(self, tmp_path: Path):
        """Every non-esc root resident survives a full four-pass startup sweep."""
        before = _seed_non_esc_residents(tmp_path)

        # One resolved esc that SHOULD be relocated
        _write_root_esc(tmp_path, 'esc-1-1', 'resolved', resolved_at=self._RESOLVED_AT)

        sweep.run_startup_sweep(tmp_path, now=self._NOW)

        _assert_residents_unchanged(tmp_path, before)

        # The resolved esc WAS relocated
        assert (tmp_path / 'archive' / '2026-05-20' / 'esc-1-1.json').exists()
        assert not (tmp_path / 'esc-1-1.json').exists()

    def test_non_esc_residents_untouched_by_the_reap_pass(self, tmp_path: Path):
        """Nothing outside the esc-* namespace is ever a reap_orphan_locks candidate.

        The sidecar-reap pass is the one that DELETES, so the foreign
        ``*.json.lock`` residents matter most here: ``sched-state.json.lock`` has
        no data file and would look exactly like an orphan to a widened glob.
        """
        before = _seed_non_esc_residents(tmp_path)

        # Positive control: one genuine escalation orphan that MUST be reaped.
        orphan = _write_lock(tmp_path, 'esc-1-1')

        count = sweep.reap_orphan_locks(tmp_path, apply=True)

        assert count == 1, 'only the esc-* orphan is a candidate'
        assert not orphan.exists(), 'the genuine orphan should have been reaped'
        _assert_residents_unchanged(tmp_path, before)


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

    def test_collision_target_already_exists_leaves_loose_untouched(self, tmp_path: Path):
        """Safety-critical: if target dated-subdir file already exists, loose file is NOT overwritten."""
        archive_root = tmp_path / 'archive'
        archive_root.mkdir(parents=True)
        loose_path = archive_root / 'esc-5-1.json'
        esc = Escalation(
            id='esc-5-1',
            task_id='1',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='collision test',
            status='resolved',
            resolved_at='2026-05-20T10:00:00+00:00',
        )
        loose_path.write_text(esc.to_json())

        # Pre-create the target file with sentinel bytes distinct from the loose copy
        target_dir = archive_root / '2026-05-20'
        target_dir.mkdir(parents=True)
        target_path = target_dir / 'esc-5-1.json'
        target_bytes = b'{"existing": "sentinel"}'
        target_path.write_bytes(target_bytes)

        count = sweep.reap_loose_archive_files(tmp_path, apply=True)

        # Loose file is left in place — never moved when target exists
        assert loose_path.exists()
        # Count is 0 — collision guard fired
        assert count == 0
        # Existing target bytes are unchanged — no data loss
        assert target_path.read_bytes() == target_bytes

    def test_non_terminal_status_left_in_place(self, tmp_path: Path):
        """Loose archive file with non-terminal status (pending) is skipped and not counted."""
        archive_root = tmp_path / 'archive'
        archive_root.mkdir(parents=True)
        loose_path = archive_root / 'esc-6-1.json'
        esc = Escalation(
            id='esc-6-1',
            task_id='1',
            agent_role='test',
            severity='info',
            category='cleanup_needed',
            summary='pending loose file',
            status='pending',
            resolved_at=None,
        )
        loose_path.write_text(esc.to_json())

        count = sweep.reap_loose_archive_files(tmp_path, apply=True)

        assert loose_path.exists()
        assert count == 0

    def test_unparsable_json_left_in_place(self, tmp_path: Path):
        """Loose archive file with unparsable JSON is skipped and not counted."""
        archive_root = tmp_path / 'archive'
        archive_root.mkdir(parents=True)
        loose_path = archive_root / 'esc-7-1.json'
        loose_path.write_bytes(b'{not valid json')

        count = sweep.reap_loose_archive_files(tmp_path, apply=True)

        assert loose_path.exists()
        assert count == 0


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



def _write_lock(queue_dir: Path, escalation_id: str) -> Path:
    """Create the bare ``{escalation_id}.json.lock`` sidecar in the queue root.

    A plain touch is exactly what ``escalation_id_lock``'s
    ``os.open(..., O_CREAT)`` leaves behind: an empty, never-renamed file whose
    only role is to be a stable flock() target.
    """
    queue_dir.mkdir(parents=True, exist_ok=True)
    path = queue_dir / f'{escalation_id}.json.lock'
    path.touch()
    return path


class TestReapOrphanLocks:
    """sweep.reap_orphan_locks unlinks sidecars whose record is in neither tier."""

    def test_orphan_lock_deleted_on_apply(self, tmp_path: Path):
        """A sidecar with no record in root and no archive at all is reaped."""
        lock_path = _write_lock(tmp_path, 'esc-1-1')
        assert not (tmp_path / 'archive').exists(), 'precondition: no archive tier'

        count = sweep.reap_orphan_locks(tmp_path, apply=True)

        assert count == 1
        assert not lock_path.exists(), 'orphaned sidecar should have been unlinked'

    def test_orphan_lock_counted_but_kept_on_dry_run(self, tmp_path: Path):
        """apply=False reports the would-reap count and mutates nothing on disk."""
        lock_path = _write_lock(tmp_path, 'esc-1-1')
        before = _disk_snapshot(tmp_path)

        count = sweep.reap_orphan_locks(tmp_path, apply=False)

        assert count == 1, 'dry-run still reports what it would reap'
        assert lock_path.exists(), 'dry-run must not unlink the sidecar'
        assert _disk_snapshot(tmp_path) == before, (
            'dry-run changed disk state — apply=False must be a pure count'
        )

    def test_lock_for_archived_record_is_kept(self, tmp_path: Path):
        """An archived record's sidecar is NOT an orphan.

        An archived record is still readable via ``get()``/``get_by_task`` and
        its resolve/dismiss paths still take the sidecar lock, so a lock counts
        as orphaned only when the record is absent from BOTH tiers.
        """
        _write_archive_esc(
            tmp_path, 'esc-1-1', '2026-05-20T10:00:00+00:00', 'resolved'
        )
        lock_path = _write_lock(tmp_path, 'esc-1-1')

        count = sweep.reap_orphan_locks(tmp_path, apply=True)

        assert count == 0, 'a record in the archive tier still owns its lock'
        assert lock_path.exists(), 'sidecar of an archived record was reaped'

    def test_lock_for_root_record_is_kept(self, tmp_path: Path):
        """A pending root record's sidecar is NOT an orphan."""
        _write_root_esc(tmp_path, 'esc-2-1', 'pending')
        lock_path = _write_lock(tmp_path, 'esc-2-1')

        count = sweep.reap_orphan_locks(tmp_path, apply=True)

        assert count == 0, 'a record in the queue root still owns its lock'
        assert lock_path.exists(), 'sidecar of a live root record was reaped'


class TestReapOrphanLocksSeqSafety:
    """SAFETY: the per-task_id .seq counter and its sidecar must survive the reap."""

    def test_seq_counter_and_its_lock_survive_reap(self, tmp_path: Path):
        """make_id's durable counter has no .json record — it must not look orphaned.

        The id is MINTED through the real ``make_id`` rather than hand-writing
        ``esc-4566.seq``: that couples this guard to the actual counter naming,
        so a future rename of the suffix fails HERE, loudly, instead of silently
        disarming the guard.
        """
        from escalation.queue import EscalationQueue

        q = EscalationQueue(tmp_path)
        esc_id = q.make_id('4566')
        assert esc_id == 'esc-4566-1'

        counter = tmp_path / 'esc-4566.seq'
        counter_lock = tmp_path / 'esc-4566.seq.json.lock'
        assert counter.exists(), 'precondition: make_id wrote the durable counter'
        assert counter_lock.exists(), 'precondition: make_id took the counter sidecar lock'
        counter_bytes = counter.read_bytes()

        count = sweep.reap_orphan_locks(tmp_path, apply=True)

        assert count == 0, 'the counter sidecar is not an orphan and must not be reaped'
        assert counter.exists(), 'the .seq counter itself was deleted'
        assert counter.read_bytes() == counter_bytes, '.seq counter contents changed'
        assert counter_lock.exists(), (
            'esc-4566.seq.json.lock was reaped — a counter has no .json record by '
            'construction, so it would look orphaned forever'
        )

        # The counter is still AUTHORITATIVE: a fresh queue mints the next id
        # from it rather than falling into _recover_seq_from_disk's repair path
        # (which, with no submitted records on disk, would rewind to esc-4566-1).
        assert EscalationQueue(tmp_path).make_id('4566') == 'esc-4566-2'


class TestStartupSweepOrphanLockPass:
    """run_startup_sweep's fourth pass: wiring, ordering, dry-run and idempotency."""

    _NOW = datetime(2026, 6, 4, tzinfo=UTC)
    _STALE_AT = '2026-01-01T00:00:00+00:00'
    _RECENT_AT = '2026-05-20T10:00:00+00:00'

    def test_lock_of_record_archived_by_this_run_is_kept(self, tmp_path: Path):
        """ORDERING: the reap runs AFTER sweep(), so a just-relocated record keeps its lock.

        The companion to the prune-ordering case below, and the more dangerous
        direction of the two.  Pass 1 moves esc-7-1 out of the root into
        ``archive/<date>/``; pass 4 is safe only because it builds its archive
        view AFTER that move.  Hoisting the index build — or the whole reap —
        above ``sweep()`` would unlink the sidecar of a perfectly LIVE record,
        data-affecting harm, where a bad prune-vs-reap order merely defers
        cleanup by one restart.  Nothing else in the suite fails under that
        reordering.
        """
        _write_root_esc(tmp_path, 'esc-7-1', 'resolved', resolved_at=self._RECENT_AT)
        lock_path = _write_lock(tmp_path, 'esc-7-1')

        report = sweep.run_startup_sweep(tmp_path, now=self._NOW)

        assert report.sweep.archived == 1, 'precondition: pass 1 relocated the record'
        assert (tmp_path / 'archive' / '2026-05-20' / 'esc-7-1.json').exists()
        assert report.orphan_locks_reaped == 0, (
            'the reap ran before (or independently of) the root->archive '
            'relocation and judged a live record dead'
        )
        assert lock_path.exists(), (
            'sidecar of a record archived by this very run was unlinked — the '
            'reap must take its archive snapshot after sweep()'
        )

    def test_lock_of_same_run_pruned_record_is_reaped(self, tmp_path: Path):
        """ORDERING: the reap runs AFTER prune, so this run's evictions are cleaned up.

        Reversing the two passes would leave the pruned record's sidecar behind
        for a whole restart cycle — the exact accumulation this task removes.
        """
        # Stale: prune_archive drops archive/2026-01-01 during THIS run.
        _write_archive_esc(tmp_path, 'esc-9-1', self._STALE_AT, 'resolved')
        stale_lock = _write_lock(tmp_path, 'esc-9-1')
        # Recent: inside retention, so record and sidecar both stay.
        _write_archive_esc(tmp_path, 'esc-8-1', self._RECENT_AT, 'resolved')
        recent_lock = _write_lock(tmp_path, 'esc-8-1')

        report = sweep.run_startup_sweep(tmp_path, now=self._NOW)

        assert report.pruned_dirs == 1, 'the stale dated dir should have been pruned'
        assert report.orphan_locks_reaped == 1
        assert not stale_lock.exists(), (
            'sidecar of a record pruned in this same run survived — the reap pass '
            'must run after prune_archive'
        )
        assert recent_lock.exists(), 'sidecar of a still-archived record was reaped'

    def test_startup_summary_line_reports_orphan_lock_count(self, tmp_path: Path, caplog):
        """The one INFO summary line carries orphan_locks= beside the other counts."""
        _write_lock(tmp_path, 'esc-1-1')

        with caplog.at_level(logging.INFO, logger='escalation.sweep'):
            sweep.run_startup_sweep(tmp_path, now=self._NOW)

        infos = [
            r for r in caplog.records
            if r.name == 'escalation.sweep' and r.levelno == logging.INFO
        ]
        assert len(infos) == 1, f'expected ONE summary line; got: {[r.getMessage() for r in infos]}'
        msg = infos[0].getMessage()
        assert 'orphan_locks=1' in msg, msg
        assert 'loose_reaped=' in msg and 'pruned_dirs=' in msg, msg

    def test_startup_dry_run_reports_orphans_without_deleting(self, tmp_path: Path, caplog):
        """apply=False counts orphans and leaves disk byte-identical."""
        lock_path = _write_lock(tmp_path, 'esc-1-1')
        before = _disk_snapshot(tmp_path)

        with caplog.at_level(logging.INFO, logger='escalation.sweep'):
            report = sweep.run_startup_sweep(tmp_path, apply=False, now=self._NOW)

        assert report.orphan_locks_reaped == 1, (
            'exactly one orphan is present; a >= assertion here would pass on a '
            'wrong count'
        )
        assert lock_path.exists(), 'dry-run unlinked a sidecar'
        assert _disk_snapshot(tmp_path) == before, 'dry-run changed disk state'
        assert any(
            'DRY-RUN' in r.getMessage() for r in caplog.records
        ), f'expected DRY-RUN summary; got: {[r.getMessage() for r in caplog.records]}'

    def test_dry_run_orphan_count_is_a_lower_bound_not_a_projection(self, tmp_path: Path):
        """DRY-RUN SEMANTICS: a sidecar an applying run WOULD reap is counted as a keep.

        ``archive.prune_archive`` has no dry-run mode for run_startup_sweep to
        forward ``apply`` to, so pass 3 is skipped outright and the stale record
        is still in the archive when pass 4 takes its snapshot — its sidecar
        looks live.  The applying run therefore reaps MORE than the dry-run
        promised.  That asymmetry is the contract documented on
        run_startup_sweep's ``apply`` arg: never an over-estimate, so an operator
        reading a dry-run is never surprised by fewer reaps than advertised.
        """
        _write_archive_esc(tmp_path, 'esc-9-1', self._STALE_AT, 'resolved')
        stale_lock = _write_lock(tmp_path, 'esc-9-1')
        plain_orphan = _write_lock(tmp_path, 'esc-1-1')
        before = _disk_snapshot(tmp_path)

        dry = sweep.run_startup_sweep(tmp_path, apply=False, now=self._NOW)

        assert dry.pruned_dirs == 0, 'prune_archive is skipped entirely in dry-run'
        assert dry.orphan_locks_reaped == 1, (
            "dry-run still sees the stale record archived, so only the plain "
            "orphan counts"
        )
        assert _disk_snapshot(tmp_path) == before, 'dry-run changed disk state'

        applied = sweep.run_startup_sweep(tmp_path, apply=True, now=self._NOW)

        assert applied.pruned_dirs == 1
        assert applied.orphan_locks_reaped == 2, (
            "applying reaps the pruned record's sidecar too — the dry-run count "
            "is a LOWER BOUND, never an over-estimate"
        )
        assert not stale_lock.exists() and not plain_orphan.exists()

    def test_second_startup_sweep_is_a_byte_stable_noop(self, tmp_path: Path):
        """After an applying run the pass neither re-reaps nor resurrects a sidecar."""
        _write_lock(tmp_path, 'esc-1-1')
        _write_root_esc(tmp_path, 'esc-3-1', 'pending')

        first = sweep.run_startup_sweep(tmp_path, now=self._NOW)
        assert first.orphan_locks_reaped == 1
        snapshot_after_run1 = _disk_snapshot(tmp_path)

        second = sweep.run_startup_sweep(tmp_path, now=self._NOW)

        assert second.orphan_locks_reaped == 0, 'nothing left to reap on the second run'
        assert _disk_snapshot(tmp_path) == snapshot_after_run1, (
            'second run_startup_sweep changed disk state — the reap pass either '
            're-created a sidecar via escalation_id_lock or re-reaped'
        )


class TestReapOrphanLocksConcurrency:
    """What happens when another process acts while we hold the per-id lock.

    Both tests model that process DETERMINISTICALLY — no threads, no sleeps — by
    wrapping ``sweep.escalation_id_lock`` in a context manager that delegates to
    the real one and performs its side effect INSIDE the critical section, which
    is exactly the interleaving a real concurrent writer produces.
    """

    def _patch_lock_with_side_effect(self, monkeypatch, side_effect):
        real_lock = sweep.escalation_id_lock

        @contextlib.contextmanager
        def _lock_with_side_effect(queue_dir: Path, escalation_id: str):
            with real_lock(queue_dir, escalation_id):
                side_effect(Path(queue_dir), escalation_id)
                yield

        monkeypatch.setattr(sweep, 'escalation_id_lock', _lock_with_side_effect)

    def test_record_created_while_lock_held_is_not_reaped(self, tmp_path: Path, monkeypatch):
        """A writer mid-submit that held the sidecar first must keep its lock."""
        def _submit_the_record(queue_dir: Path, escalation_id: str) -> None:
            _write_root_esc(queue_dir, escalation_id, 'pending')

        self._patch_lock_with_side_effect(monkeypatch, _submit_the_record)
        lock_path = _write_lock(tmp_path, 'esc-1-1')

        count = sweep.reap_orphan_locks(tmp_path, apply=True)

        assert count == 0, 'the record exists by the time we hold the lock — not an orphan'
        assert lock_path.exists(), (
            'reaped the sidecar of a record that appeared between the pre-check '
            'and the lock — the in-lock re-check is missing'
        )

    def test_lock_vanishing_under_us_is_tolerated(self, tmp_path: Path, monkeypatch):
        """Another process removing the sidecar first is the intended end state."""
        def _steal_the_sidecar(queue_dir: Path, escalation_id: str) -> None:
            os.unlink(queue_dir / f'{escalation_id}.json.lock')

        self._patch_lock_with_side_effect(monkeypatch, _steal_the_sidecar)
        lock_path = _write_lock(tmp_path, 'esc-1-1')

        count = sweep.reap_orphan_locks(tmp_path, apply=True)

        assert count == 1, 'the sidecar is gone, which is what we wanted — count it'
        assert not lock_path.exists()


class TestReapOrphanLocksResilience:
    """One unreapable sidecar must not cost the pass — or the whole startup report.

    The raise that actually happens in production comes out of
    ``escalation_id_lock``'s own ``os.open(O_CREAT|O_RDWR)`` — a root-owned
    sidecar, EIO, an exhausted fd table — NOT out of ``os.unlink``, which needs
    write permission on the DIRECTORY and so fails for every candidate or none.
    So the guard has to wrap the whole critical section, and these tests inject
    at the lock rather than at the unlink.

    Monkeypatching beats ``chmod 000`` here: under a root CI user chmod is not a
    barrier at all and the test would silently stop testing anything.
    """

    _NOW = datetime(2026, 6, 4, tzinfo=UTC)
    _RESOLVED_AT = '2026-05-20T10:00:00+00:00'

    def _fail_first_lock(self, monkeypatch, exc: OSError) -> dict:
        """Raise `exc` out of the FIRST lock acquisition; delegate for every later one.

        Failing whichever candidate readdir hands us first — rather than a named
        id — is what makes the assertion order-independent: ``Path.glob`` does
        not sort, so pinning a specific id would let an ABORTING implementation
        pass whenever that id happened to come last.
        """
        real_lock = sweep.escalation_id_lock
        state: dict = {'failed_id': None}

        @contextlib.contextmanager
        def _lock(queue_dir: Path, escalation_id: str):
            if state['failed_id'] is None:
                state['failed_id'] = escalation_id
                raise exc
            with real_lock(queue_dir, escalation_id):
                yield

        monkeypatch.setattr(sweep, 'escalation_id_lock', _lock)
        return state

    def _fail_lock_for(self, monkeypatch, bad_id: str, exc: OSError) -> None:
        """Raise `exc` only for `bad_id`, so other passes' lock use is unaffected."""
        real_lock = sweep.escalation_id_lock

        @contextlib.contextmanager
        def _lock(queue_dir: Path, escalation_id: str):
            if escalation_id == bad_id:
                raise exc
            with real_lock(queue_dir, escalation_id):
                yield

        monkeypatch.setattr(sweep, 'escalation_id_lock', _lock)

    def test_unopenable_sidecar_does_not_abort_the_remaining_candidates(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """A PermissionError out of the lock is logged and skipped, not propagated."""
        locks = {
            esc_id: _write_lock(tmp_path, esc_id)
            for esc_id in ('esc-1-1', 'esc-2-1', 'esc-3-1')
        }
        state = self._fail_first_lock(
            monkeypatch, PermissionError(13, 'Permission denied')
        )

        with caplog.at_level(logging.WARNING, logger='escalation.sweep'):
            count = sweep.reap_orphan_locks(tmp_path, apply=True)

        bad_id = state['failed_id']
        assert bad_id in locks, 'precondition: the injected failure hit a candidate'
        assert count == 2, (
            'the two healthy orphans must still be reaped — one unopenable '
            'sidecar cannot cost the pass'
        )
        assert locks[bad_id].exists(), 'the unreapable sidecar must be left in place'
        for esc_id, path in locks.items():
            if esc_id != bad_id:
                assert not path.exists(), f'{esc_id} was a healthy orphan and should be gone'

        warnings = [
            r.getMessage() for r in caplog.records
            if r.name == 'escalation.sweep' and r.levelno == logging.WARNING
        ]
        assert any(f'{bad_id}.json.lock' in m for m in warnings), warnings

    def test_startup_summary_still_emitted_when_a_sidecar_is_unreapable(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """Passes 1-3 counts reach the operator even though pass 4 hit an error.

        Without the guard the raise escapes run_startup_sweep before the report
        is built, server.py swallows it as a non-fatal WARNING, and the operator
        loses the archived/loose_reaped/pruned_dirs numbers from work that
        already SUCCEEDED.
        """
        # Pass 1 relocates this one, taking esc-5-1's lock on the way — hence the
        # named (not first-call) injection, so pass 1 is unaffected.
        _write_root_esc(tmp_path, 'esc-5-1', 'resolved', resolved_at=self._RESOLVED_AT)
        bad_lock = _write_lock(tmp_path, 'esc-1-1')
        good_lock = _write_lock(tmp_path, 'esc-2-1')
        self._fail_lock_for(monkeypatch, 'esc-1-1', OSError(5, 'Input/output error'))

        with caplog.at_level(logging.INFO, logger='escalation.sweep'):
            report = sweep.run_startup_sweep(tmp_path, now=self._NOW)

        assert report.sweep.archived == 1, 'pass 1 succeeded and must still be reported'
        assert report.orphan_locks_reaped == 1, 'only the healthy orphan was reaped'
        assert bad_lock.exists()
        assert not good_lock.exists()

        infos = [
            r.getMessage() for r in caplog.records
            if r.name == 'escalation.sweep' and r.levelno == logging.INFO
        ]
        assert len(infos) == 1, (
            'the single startup summary line was lost — the reap raised past '
            f'run_startup_sweep before the report was built; got: {infos}'
        )
        assert 'archived=1' in infos[0] and 'orphan_locks=1' in infos[0], infos[0]


class TestReapOrphanLocksEarlyExit:
    """With nothing to judge, the pass must not walk the archive tree at all."""

    def test_no_candidates_skips_the_archive_walk(self, tmp_path: Path, monkeypatch):
        """The steady state — a root of .seq sidecars and foreign files — costs no rglob.

        ``run_startup_sweep`` already pays one full archive rglob in ``sweep()``;
        a second one on every restart, for a root with zero reapable candidates,
        is pure waste on the largest tree in the queue dir.
        """
        _seed_non_esc_residents(tmp_path)
        _write_lock(tmp_path, f'esc-4566{SEQ_COUNTER_SUFFIX}')
        _write_archive_esc(tmp_path, 'esc-1-1', '2026-05-20T10:00:00+00:00', 'resolved')

        calls: list[Path] = []
        real_stems = sweep._archive_stems

        def _counting(archive_root: Path) -> set[str]:
            calls.append(archive_root)
            return real_stems(archive_root)

        monkeypatch.setattr(sweep, '_archive_stems', _counting)

        assert sweep.reap_orphan_locks(tmp_path, apply=True) == 0
        assert calls == [], (
            'reap_orphan_locks walked the archive tree with zero candidates — '
            'the early return ahead of the index build was removed'
        )

    def test_archive_walk_still_happens_when_a_candidate_exists(
        self, tmp_path: Path, monkeypatch
    ):
        """Positive control: the early return must not swallow a real candidate."""
        _seed_non_esc_residents(tmp_path)
        _write_lock(tmp_path, f'esc-4566{SEQ_COUNTER_SUFFIX}')
        orphan = _write_lock(tmp_path, 'esc-1-1')

        calls: list[Path] = []
        real_stems = sweep._archive_stems

        def _counting(archive_root: Path) -> set[str]:
            calls.append(archive_root)
            return real_stems(archive_root)

        monkeypatch.setattr(sweep, '_archive_stems', _counting)

        assert sweep.reap_orphan_locks(tmp_path, apply=True) == 1
        assert not orphan.exists()
        assert len(calls) == 1, 'the archive tier is consulted exactly once per pass'


class TestArchiveStemsHelper:
    """_archive_stems: presence-only, and quiet about duplicates _build_archive_index warns on."""

    def test_returns_stems_from_every_dated_subdir(self, tmp_path: Path):
        _write_archive_esc(tmp_path, 'esc-1-1', '2026-05-20T10:00:00+00:00', 'resolved')
        _write_archive_esc(tmp_path, 'esc-2-1', '2026-01-01T00:00:00+00:00', 'dismissed')

        stems = sweep._archive_stems(tmp_path / 'archive')

        assert stems == {'esc-1-1', 'esc-2-1'}

    def test_missing_archive_root_is_an_empty_set(self, tmp_path: Path):
        assert sweep._archive_stems(tmp_path / 'archive') == set()

    def test_duplicate_id_across_dated_dirs_logs_nothing(self, tmp_path: Path, caplog):
        """The duplicate WARNING belongs to the path-picking indexer, not to presence.

        sweep() has already emitted it once this startup; a second copy per
        multi-dated duplicate is noise for a caller that never looks at paths.
        """
        _write_archive_esc(tmp_path, 'esc-1-1', '2026-05-20T10:00:00+00:00', 'resolved')
        _write_archive_esc(tmp_path, 'esc-1-1', '2026-01-01T00:00:00+00:00', 'resolved')

        with caplog.at_level(logging.WARNING, logger='escalation.sweep'):
            stems = sweep._archive_stems(tmp_path / 'archive')

        assert stems == {'esc-1-1'}
        assert [r.getMessage() for r in caplog.records] == []
