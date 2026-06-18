"""Tests for EscalationQueue — focusing on dismiss_all_pending()."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue, iter_all_escalation_paths


def _make_escalation(esc_id: str, task_id: str = '1', status: str = 'pending', level: int = 0) -> Escalation:
    esc = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='task_failure',
        summary='Something failed',
        level=level,
    )
    esc.status = status
    return esc


def _submit_escalation(queue: EscalationQueue, esc: Escalation) -> None:
    """Write an escalation directly, bypassing the callback."""
    queue.submit(esc)


class TestDismissAllPending:
    """EscalationQueue.dismiss_all_pending() bulk-dismisses pending escalations."""

    def test_empty_queue_returns_zero(self, tmp_path: Path):
        """Empty queue: no-op, returns 0."""
        queue = EscalationQueue(tmp_path / 'queue')
        count = queue.dismiss_all_pending('Stale from prior run')
        assert count == 0

    def test_single_pending_dismissed(self, tmp_path: Path):
        """Single pending escalation is dismissed and count returned is 1."""
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')
        queue.submit(esc)

        count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 1
        updated = queue.get('esc-1-1')
        assert updated is not None
        assert updated.status == 'dismissed'
        assert updated.resolution == 'Stale from prior run'

    def test_multiple_pending_all_dismissed(self, tmp_path: Path):
        """Multiple pending escalations are all dismissed; count matches."""
        queue = EscalationQueue(tmp_path / 'queue')
        for i in range(3):
            queue.submit(_make_escalation(f'esc-{i}-1', task_id=str(i)))

        count = queue.dismiss_all_pending('Orchestrator restarted')

        assert count == 3
        for i in range(3):
            esc = queue.get(f'esc-{i}-1')
            assert esc is not None
            assert esc.status == 'dismissed'

    def test_resolved_escalation_not_touched(self, tmp_path: Path):
        """Already-resolved escalations are not re-dismissed."""
        queue = EscalationQueue(tmp_path / 'queue')

        pending = _make_escalation('esc-1-1')
        queue.submit(pending)

        already_resolved = _make_escalation('esc-2-1')
        queue.submit(already_resolved)
        queue.resolve('esc-2-1', 'Fixed manually', dismiss=False)

        count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 1  # only the pending one

        resolved_esc = queue.get('esc-2-1')
        assert resolved_esc is not None
        assert resolved_esc.status == 'resolved'
        assert resolved_esc.resolution == 'Fixed manually'

    def test_dismissed_escalation_not_touched(self, tmp_path: Path):
        """Already-dismissed escalations are not re-dismissed."""
        queue = EscalationQueue(tmp_path / 'queue')

        pending = _make_escalation('esc-1-1')
        queue.submit(pending)

        already_dismissed = _make_escalation('esc-2-1')
        queue.submit(already_dismissed)
        queue.resolve('esc-2-1', 'User dismissed earlier', dismiss=True)

        count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 1  # only the pending one

        dismissed_esc = queue.get('esc-2-1')
        assert dismissed_esc is not None
        assert dismissed_esc.status == 'dismissed'
        assert dismissed_esc.resolution == 'User dismissed earlier'  # unchanged

    def test_resolution_message_preserved(self, tmp_path: Path):
        """Resolution message is preserved on dismissed escalations."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))

        msg = 'Auto-dismissed: orchestrator restarted, stale from prior run'
        queue.dismiss_all_pending(msg)

        esc = queue.get('esc-1-1')
        assert esc is not None
        assert esc.resolution == msg

    def test_mixed_statuses_only_pending_dismissed(self, tmp_path: Path):
        """With a mix of pending/resolved/dismissed, only pending ones are dismissed."""
        queue = EscalationQueue(tmp_path / 'queue')

        queue.submit(_make_escalation('esc-1-1'))  # pending
        queue.submit(_make_escalation('esc-2-1'))  # pending

        queue.submit(_make_escalation('esc-3-1'))
        queue.resolve('esc-3-1', 'resolved already', dismiss=False)

        queue.submit(_make_escalation('esc-4-1'))
        queue.resolve('esc-4-1', 'dismissed already', dismiss=True)

        count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 2  # esc-1-1 and esc-2-1

        assert queue.get('esc-1-1').status == 'dismissed'  # type: ignore[union-attr]
        assert queue.get('esc-2-1').status == 'dismissed'  # type: ignore[union-attr]
        assert queue.get('esc-3-1').status == 'resolved'  # type: ignore[union-attr]
        assert queue.get('esc-4-1').status == 'dismissed'  # type: ignore[union-attr]
        assert queue.get('esc-4-1').resolution == 'dismissed already'  # type: ignore[union-attr]


class TestDismissAllPendingResilience:
    """EscalationQueue.dismiss_all_pending() is resilient to per-item resolve() failures."""

    def test_resolve_failure_does_not_abort_loop(self, tmp_path: Path):
        """If resolve() raises on one item, the remaining items are still dismissed."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1', task_id='1'))
        queue.submit(_make_escalation('esc-2-1', task_id='2'))
        queue.submit(_make_escalation('esc-3-1', task_id='3'))

        # Patch resolve() so it raises OSError for esc-2-1, succeeds for others
        original_resolve = queue.resolve

        def patched_resolve(esc_id: str, resolution: str, dismiss: bool = False, **kwargs):
            if esc_id == 'esc-2-1':
                raise OSError('disk full')
            return original_resolve(esc_id, resolution, dismiss=dismiss, **kwargs)

        with patch.object(queue, 'resolve', side_effect=patched_resolve):
            count = queue.dismiss_all_pending('Stale from prior run')

        # Only 2 of 3 succeeded (esc-2-1 failed)
        assert count == 2

    def test_resolve_failure_count_reflects_successes_only(self, tmp_path: Path):
        """Returned count only reflects successfully dismissed escalations."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1', task_id='1'))
        queue.submit(_make_escalation('esc-2-1', task_id='2'))

        original_resolve = queue.resolve

        def patched_resolve(esc_id: str, resolution: str, dismiss: bool = False, **kwargs):
            if esc_id == 'esc-1-1':
                raise OSError('permission denied')
            return original_resolve(esc_id, resolution, dismiss=dismiss, **kwargs)

        with patch.object(queue, 'resolve', side_effect=patched_resolve):
            count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 1  # only esc-2-1 succeeded

    def test_resolve_failure_does_not_propagate(self, tmp_path: Path):
        """An OSError from resolve() does not propagate to the caller."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1', task_id='1'))

        def always_fail(esc_id: str, resolution: str, dismiss: bool = False):
            raise OSError('disk full')

        with patch.object(queue, 'resolve', side_effect=always_fail):
            # Must not raise
            count = queue.dismiss_all_pending('Stale from prior run')

        assert count == 0


class TestDismissAllPendingPreservesL1:
    """EscalationQueue.dismiss_all_pending() must NOT dismiss level-1 escalations."""

    def test_dismiss_all_pending_skips_level_1_escalations(self, tmp_path: Path):
        """L0 escalation is dismissed; L1 escalation remains pending — count is 1."""
        queue = EscalationQueue(tmp_path / 'queue')
        l0 = _make_escalation('esc-l0-1', task_id='1', level=0)
        l1 = _make_escalation('esc-l1-1', task_id='2', level=1)
        queue.submit(l0)
        queue.submit(l1)

        count = queue.dismiss_all_pending('Auto-dismissed: orchestrator restarted')

        assert count == 1

        updated_l0 = queue.get('esc-l0-1')
        assert updated_l0 is not None
        assert updated_l0.status == 'dismissed'

        updated_l1 = queue.get('esc-l1-1')
        assert updated_l1 is not None
        assert updated_l1.status == 'pending'
        assert updated_l1.resolved_at is None
        assert updated_l1.resolved_by is None

    def test_dismiss_all_pending_only_l1_returns_zero(self, tmp_path: Path):
        """Queue with only an L1 escalation: dismiss_all_pending returns 0, L1 stays pending."""
        queue = EscalationQueue(tmp_path / 'queue')
        l1 = _make_escalation('esc-l1-1', task_id='1', level=1)
        queue.submit(l1)

        count = queue.dismiss_all_pending('Auto-dismissed: orchestrator restarted')

        assert count == 0

        updated_l1 = queue.get('esc-l1-1')
        assert updated_l1 is not None
        assert updated_l1.status == 'pending'

    def test_dismiss_all_pending_idempotent_with_l1(self, tmp_path: Path):
        """Calling dismiss_all_pending twice is safe: second call returns 0, L1 is unmodified."""
        queue = EscalationQueue(tmp_path / 'queue')
        l0 = _make_escalation('esc-l0-1', task_id='1', level=0)
        l1 = _make_escalation('esc-l1-1', task_id='2', level=1)
        queue.submit(l0)
        queue.submit(l1)

        first = queue.dismiss_all_pending('Auto-dismissed: orchestrator restarted')
        assert first == 1

        # Second call — no pending L0s remain; L1 is untouched.
        second = queue.dismiss_all_pending('Auto-dismissed: orchestrator restarted')
        assert second == 0

        # L1 status/resolved_at/resolved_by are unchanged across both calls.
        updated_l1 = queue.get('esc-l1-1')
        assert updated_l1 is not None
        assert updated_l1.status == 'pending'
        assert updated_l1.resolved_at is None
        assert updated_l1.resolved_by is None


class TestGetArchiveFallback:
    """EscalationQueue.get() falls back to the archive when the root path is missing."""

    def test_get_returns_archived_escalation_after_resolve(self, tmp_path: Path):
        """After resolve(), queue.get(id) returns the escalation from the archive."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))

        queue.resolve('esc-1-1', 'Archive fallback works')

        # File has been moved to archive — root path no longer exists
        assert not (queue.queue_dir / 'esc-1-1.json').exists()

        # get() must still return the escalation
        result = queue.get('esc-1-1')
        assert result is not None
        assert result.status == 'resolved'
        assert result.resolution == 'Archive fallback works'


class TestGetWithDuplicateArchiveCandidates:
    """get() must warn and pick the newest when multiple archive files share one id."""

    def test_get_with_duplicate_archive_files_logs_warning_and_returns_newest(
        self, tmp_path: Path, caplog,
    ):
        """When two archive files exist for the same id, get() warns (at logger \
`escalation.queue`, mentioning the id) and returns the newest.

        The exact contents of the warning message (paths, phrasing) are an
        implementation detail and intentionally NOT pinned by this test.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Manually create two archive files for the same id under different dated subdirs.
        # Neither file exists in queue_dir root (simulating a duplicate-archive state).
        older_dir = queue.queue_dir / 'archive' / '2025-01-01'
        newer_dir = queue.queue_dir / 'archive' / '2025-06-15'
        older_dir.mkdir(parents=True, exist_ok=True)
        newer_dir.mkdir(parents=True, exist_ok=True)

        older_esc = _make_escalation('esc-1-1', status='resolved')
        older_esc.resolution = 'older'
        (older_dir / 'esc-1-1.json').write_text(older_esc.to_json())

        newer_esc = _make_escalation('esc-1-1', status='resolved')
        newer_esc.resolution = 'newer'
        (newer_dir / 'esc-1-1.json').write_text(newer_esc.to_json())

        # Confirm setup: no root file, two archive files
        assert not (queue.queue_dir / 'esc-1-1.json').exists()
        archive_files = list((queue.queue_dir / 'archive').rglob('esc-1-1.json'))
        assert len(archive_files) == 2

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            result = queue.get('esc-1-1')

        # (a) Must return the newest (2025-06-15), not arbitrary candidates[0]
        assert result is not None
        assert result.resolution == 'newer', (
            f"Expected resolution='newer' (from 2025-06-15 dir), got {result.resolution!r}"
        )

        # (b) A WARNING must be emitted at logger 'escalation.queue' mentioning the id.
        # The exact message format (including whether full paths are embedded) is
        # not part of the test contract — it is an implementation detail of the
        # warn-and-pick-newest path.
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'escalation.queue'
        ]
        assert warning_records, (
            f"Expected a WARNING at logger 'escalation.queue'; got records: {caplog.records}"
        )
        assert any('esc-1-1' in r.message for r in warning_records), (
            f'Expected a WARNING mentioning esc-1-1; got: {[r.message for r in warning_records]}'
        )


class TestGetByTaskAcrossArchive:
    """get_by_task() two-tier scan: hot path skips archive; broad path includes it."""

    def _setup_mixed_queue(self, tmp_path: Path):
        """Return (queue, esc_a_resolved, esc_b_pending) for task '42'."""
        queue = EscalationQueue(tmp_path / 'queue')
        # Escalation A: submitted and resolved (will be archived)
        queue.submit(_make_escalation('esc-42-1', task_id='42'))
        queue.resolve('esc-42-1', 'Fixed it')
        # Escalation B: submitted, stays pending
        queue.submit(_make_escalation('esc-42-2', task_id='42'))
        return queue

    def test_get_by_task_no_filter_includes_archived(self, tmp_path: Path):
        """get_by_task(task_id) with no status filter returns pending AND archived."""
        queue = self._setup_mixed_queue(tmp_path)

        results = queue.get_by_task('42')

        ids = {e.id for e in results}
        assert 'esc-42-1' in ids  # archived / resolved
        assert 'esc-42-2' in ids  # still pending

    def test_get_by_task_status_pending_excludes_archive(self, tmp_path: Path):
        """get_by_task(task_id, status='pending') only returns files in queue root."""
        queue = self._setup_mixed_queue(tmp_path)

        results = queue.get_by_task('42', status='pending')

        ids = {e.id for e in results}
        assert 'esc-42-2' in ids   # pending — in root
        assert 'esc-42-1' not in ids  # resolved / archived

    def test_get_pending_excludes_archive(self, tmp_path: Path):
        """get_pending() does not scan the archive directory."""
        queue = self._setup_mixed_queue(tmp_path)

        results = queue.get_pending()

        ids = {e.id for e in results}
        assert 'esc-42-2' in ids   # pending — in root
        assert 'esc-42-1' not in ids  # resolved / archived


class TestResolveArchives:
    """EscalationQueue.resolve() moves the file into archive/YYYY-MM-DD/ after resolution.

    The implementation uses a two-step approach:
      1. Atomically write the resolved JSON to queue_dir/{id}.json (tmp+rename).
      2. Best-effort os.replace() into archive/YYYY-MM-DD/{id}.json.
    If step 2 fails, the resolved file stays in queue_dir — still readable by
    get() — so the resolution is never lost.  This is intentionally more robust
    than writing directly to the archive dir: a failed archive write would leave
    the file in queue_dir as *pending* rather than as *resolved*.
    """

    def test_resolve_moves_file_to_dated_archive_subdir(self, tmp_path: Path):
        """After resolve(), the esc-*.json is in archive/<date>/ not the queue root."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))

        queue.resolve('esc-1-1', 'All good')

        # (a) File no longer in queue root
        assert not (queue.queue_dir / 'esc-1-1.json').exists()

        # (b) File is in the archive under the correct date
        # Derive expected directory from the resolved_at that was written to disk
        archived_files = list((queue.queue_dir / 'archive').rglob('esc-1-1.json'))
        assert len(archived_files) == 1
        archived_path = archived_files[0]

        # (c) The archived JSON parses back correctly
        data = json.loads(archived_path.read_text())
        assert data['status'] == 'resolved'
        assert data['resolution'] == 'All good'

        # (d) The parent directory name matches the date in resolved_at
        resolved_at = data['resolved_at']
        expected_date = resolved_at[:10]  # YYYY-MM-DD prefix
        assert archived_path.parent.name == expected_date

    def test_resolve_archive_failure_leaves_file_in_queue_root(
        self, tmp_path: Path, caplog,
    ):
        """When the archive move (os.replace) fails, resolve() still succeeds.

        Contract (the except OSError arm at queue.py:~180):
        - resolve() returns the resolved Escalation.
        - The file stays in queue_dir (readable by get()).
        - A WARNING is emitted naming the escalation.
        """
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))

        with patch('os.replace', side_effect=OSError('cross-device link')), caplog.at_level(logging.WARNING, logger='escalation.queue'):
            result = queue.resolve('esc-1-1', 'Force archive failure')

        # resolve() must return the updated escalation despite the OSError
        assert result is not None
        assert result.status == 'resolved'
        assert result.resolution == 'Force archive failure'

        # File must remain readable from queue root (archive move failed → stayed)
        from_queue = queue.get('esc-1-1')
        assert from_queue is not None
        assert from_queue.status == 'resolved'

        # A warning naming the escalation ID must have been logged
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('esc-1-1' in msg for msg in warning_messages), (
            f'Expected a warning mentioning esc-1-1; got: {warning_messages}'
        )


class TestSubmitResolved:
    """EscalationQueue.submit_resolved() — atomic single-write, single-callback resolution.

    Writes the escalation directly in resolved state (no transient pending file),
    fires only _resolve_callback (not _notify_callback), and archives the file.

    All five tests FAIL until submit_resolved is added to EscalationQueue (step-4).
    """

    def test_submit_resolved_writes_file_with_resolved_status(self, tmp_path: Path):
        """(a) After submit_resolved, queue.get(id) returns status='resolved'."""
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')

        queue.submit_resolved(esc, 'auto: terminal', resolved_by='escalation-mcp-pre-submit-check')

        result = queue.get(esc.id)
        assert result is not None, f"Escalation {esc.id} not found after submit_resolved"
        assert result.status == 'resolved', f"Expected 'resolved', got: {result.status}"

    def test_submit_resolved_returns_escalation_with_resolved_fields(self, tmp_path: Path):
        """(b) submit_resolved returns the Escalation with all resolved fields populated."""
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')

        result = queue.submit_resolved(
            esc, 'auto: terminal', resolved_by='escalation-mcp-pre-submit-check'
        )

        assert result.status == 'resolved', f"Expected status='resolved', got: {result.status}"
        assert result.resolution == 'auto: terminal', (
            f"Expected resolution='auto: terminal', got: {result.resolution}"
        )
        assert result.resolved_by == 'escalation-mcp-pre-submit-check', (
            f"Expected resolved_by='escalation-mcp-pre-submit-check', got: {result.resolved_by}"
        )
        assert result.resolved_at is not None, "Expected resolved_at to be set (ISO 8601 string)"
        # Sanity-check ISO 8601 prefix: YYYY-MM-DD
        assert result.resolved_at[:4].isdigit() and result.resolved_at[4] == '-', (
            f"resolved_at does not look like ISO 8601: {result.resolved_at!r}"
        )

    def test_submit_resolved_fires_resolve_callback(self, tmp_path: Path):
        """(c) submit_resolved fires _resolve_callback exactly once with the escalation."""
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')

        fired: list[str] = []
        queue.set_resolve_callback(lambda e: fired.append(e.id))

        queue.submit_resolved(esc, 'auto: terminal', resolved_by='escalation-mcp-pre-submit-check')

        assert fired == [esc.id], (
            f"Expected resolve callback fired once with {esc.id!r}, got: {fired}"
        )

    def test_submit_resolved_does_not_fire_notify_callback(self, tmp_path: Path):
        """(d) submit_resolved must NOT fire _notify_callback (single-callback contract)."""
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')

        notified: list[str] = []
        queue.set_notify_callback(lambda e: notified.append(e.id))

        queue.submit_resolved(esc, 'auto: terminal', resolved_by='escalation-mcp-pre-submit-check')

        assert notified == [], (
            f"Expected notify callback NOT fired, but got: {notified}"
        )

    def test_submit_resolved_archives_file(self, tmp_path: Path):
        """(e) After submit_resolved, file is moved to archive/YYYY-MM-DD/, not in queue root."""
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')

        queue.submit_resolved(esc, 'auto: terminal', resolved_by='escalation-mcp-pre-submit-check')

        # File must not be in queue root
        assert not (queue.queue_dir / f'{esc.id}.json').exists(), (
            f"Expected {esc.id}.json to be moved to archive, but still in queue root"
        )

        # File must be in the archive under a dated subdir
        archived_files = list((queue.queue_dir / 'archive').rglob(f'{esc.id}.json'))
        assert len(archived_files) == 1, (
            f"Expected exactly 1 archive file for {esc.id}, got: {archived_files}"
        )
        # Parent dir name must match the date prefix of resolved_at
        archived_path = archived_files[0]
        assert esc.resolved_at is not None
        expected_date = esc.resolved_at[:10]  # YYYY-MM-DD
        assert archived_path.parent.name == expected_date, (
            f"Expected archive dir '{expected_date}', got '{archived_path.parent.name}'"
        )

    def test_submit_resolved_archive_failure_keeps_file_in_queue_dir(
        self, tmp_path: Path, caplog,
    ):
        """When the archive os.replace fails, submit_resolved still succeeds.

        Contract (mirrors TestResolveArchives.test_resolve_archive_failure_*):
        - submit_resolved() returns the resolved Escalation.
        - The file stays in queue_dir root (readable by get()).
        - A WARNING is emitted naming the escalation ID.
        """
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')

        with patch('os.replace', side_effect=OSError('cross-device link')), \
             caplog.at_level(logging.WARNING, logger='escalation.queue'):
            result = queue.submit_resolved(esc, 'auto: terminal', resolved_by='test')

        # submit_resolved must return the updated escalation despite the OSError
        assert result is not None
        assert result.status == 'resolved'
        assert result.resolution == 'auto: terminal'

        # File must remain readable from queue root (archive move failed → stayed)
        from_queue = queue.get('esc-1-1')
        assert from_queue is not None
        assert from_queue.status == 'resolved'

        # A warning naming the escalation ID must have been logged
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('esc-1-1' in msg for msg in warning_messages), (
            f'Expected a warning mentioning esc-1-1; got: {warning_messages}'
        )

    def test_submit_resolved_resolve_callback_exception_does_not_propagate(
        self, tmp_path: Path, caplog,
    ):
        """A raising _resolve_callback is swallowed; submit_resolved still returns normally.

        Contract (mirrors analogous coverage in resolve() tests): the except-warn
        guard at queue.py guarantees that a misbehaving callback cannot abort the
        resolution — the caller gets the resolved Escalation back regardless.
        """
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _make_escalation('esc-1-1')

        def _bad_callback(e: Escalation) -> None:
            raise RuntimeError('callback boom')

        queue.set_resolve_callback(_bad_callback)

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            result = queue.submit_resolved(esc, 'auto: terminal')

        # Must return normally despite the callback raising
        assert result is not None
        assert result.status == 'resolved'

        # A warning must have been logged
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('esc-1-1' in msg for msg in warning_messages), (
            f'Expected a warning about callback failure; got: {warning_messages}'
        )


class TestResolveIdempotent:
    """EscalationQueue.resolve() is idempotent: a second call returns the first resolution unchanged."""

    def test_resolve_twice_returns_same_escalation_without_orphan(self, tmp_path: Path):
        """Second resolve() call must be a no-op: same resolution, same archive file.

        Failure mode in current main: the second call re-reads from archive, writes a
        fresh queue_dir copy, and archives it again — creating an orphan archive file
        and overwriting resolution/resolved_at.
        """
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))

        # First resolve
        first_result = queue.resolve('esc-1-1', 'Fixed once')
        assert first_result is not None
        assert first_result.status == 'resolved'

        # Exactly one archive file after first resolve
        archive_files_after_first = list((queue.queue_dir / 'archive').rglob('esc-1-1.json'))
        assert len(archive_files_after_first) == 1, (
            f'Expected 1 archive file after first resolve, got {archive_files_after_first}'
        )
        first_archive_path = archive_files_after_first[0]
        first_resolved_at = first_result.resolved_at

        # Second resolve — must be a no-op
        second_result = queue.resolve('esc-1-1', 'Fixed again')

        # (a) Return value is non-None with status='resolved'
        assert second_result is not None
        assert second_result.status == 'resolved'

        # (b) Resolution and resolved_at are from the FIRST call, not overwritten
        assert second_result.resolution == 'Fixed once', (
            f"Expected resolution='Fixed once' (first call), got {second_result.resolution!r}"
        )
        assert second_result.resolved_at == first_resolved_at, (
            f'Expected resolved_at to be unchanged; '
            f'first={first_resolved_at!r}, second={second_result.resolved_at!r}'
        )

        # (c) Still exactly one archive file, same path as before (no orphan)
        archive_files_after_second = list((queue.queue_dir / 'archive').rglob('esc-1-1.json'))
        assert len(archive_files_after_second) == 1, (
            f'Expected 1 archive file after second resolve (no orphan), '
            f'got {[str(p) for p in archive_files_after_second]}'
        )
        assert archive_files_after_second[0] == first_archive_path, (
            f'Archive file path changed: first={first_archive_path}, '
            f'second={archive_files_after_second[0]}'
        )


class TestGetByTaskDedupAcrossArchive:
    """get_by_task() must deduplicate when the same escalation id appears in both
    the queue root and the archive (crash-mid-resolve / backup-restore scenario)."""

    def test_get_by_task_dedups_when_id_in_both_queue_and_archive(self, tmp_path: Path):
        """get_by_task() returns exactly one item when id exists in both locations.

        Failure mode in current main: two-tier scan concatenates without dedup,
        so the caller receives two Escalation instances for the same id.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Submit and resolve esc-42-1 — moves into archive
        queue.submit(_make_escalation('esc-42-1', task_id='42'))
        queue.resolve('esc-42-1', 'original_resolution')

        # Sanity: file should now be in archive, not in queue root
        assert not (queue.queue_dir / 'esc-42-1.json').exists()
        archive_files = list((queue.queue_dir / 'archive').rglob('esc-42-1.json'))
        assert len(archive_files) == 1

        # Simulate crash-mid-resolve / backup-restore: copy the archived file
        # back into the queue root with a modified resolution to test dedup precedence.
        archived_esc = _make_escalation('esc-42-1', task_id='42', status='resolved')
        archived_esc.resolution = 'from_queue_root'
        (queue.queue_dir / 'esc-42-1.json').write_text(archived_esc.to_json())

        # Confirm the "maybe-in-two-places" state
        assert (queue.queue_dir / 'esc-42-1.json').exists()
        assert len(list((queue.queue_dir / 'archive').rglob('esc-42-1.json'))) == 1

        # get_by_task must return exactly one item (deduped)
        results = queue.get_by_task('42')
        ids = [e.id for e in results]
        assert ids.count('esc-42-1') == 1, (
            f'Expected exactly one esc-42-1 in results, got {len(ids)} entries: {ids}'
        )

        # queue_dir copy must win (iteration order: root first, archive second)
        returned_esc = next(e for e in results if e.id == 'esc-42-1')
        assert returned_esc.resolution == 'from_queue_root', (
            f"Expected queue_dir copy to win (resolution='from_queue_root'), "
            f"got {returned_esc.resolution!r}"
        )

    def test_get_by_task_status_filter_returns_archive_copy_when_queue_dir_has_stale_pending(
        self, tmp_path: Path,
    ):
        """Bug repro: queue_dir has stale pending copy, archive has resolved copy.

        When get_by_task('42', status='resolved') is called, it must return the
        archive's resolved copy, not skip it because the pending copy was seen first.

        Failure mode in current main: dedup runs before filter — the pending
        queue_dir copy adds the id to 'seen', then the resolved archive copy is
        skipped as a duplicate, returning [].
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Simulated crash-mid-resolve: pending copy stuck in queue_dir
        pending_copy = _make_escalation('esc-42-1', task_id='42', status='pending')
        (queue.queue_dir / 'esc-42-1.json').write_text(pending_copy.to_json())

        # Resolved copy in archive (what the caller wants to see)
        archive_dir = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir.mkdir(parents=True, exist_ok=True)
        resolved_copy = _make_escalation('esc-42-1', task_id='42', status='resolved')
        resolved_copy.resolution = 'archive_copy'
        (archive_dir / 'esc-42-1.json').write_text(resolved_copy.to_json())

        results = queue.get_by_task('42', status='resolved')

        assert len(results) == 1, (
            f'Expected exactly 1 resolved result but got {len(results)}: '
            f'{[e.resolution for e in results]}'
        )
        assert results[0].status == 'resolved'
        assert results[0].resolution == 'archive_copy'

    def test_get_by_task_status_filter_prefers_queue_dir_when_both_copies_match(
        self, tmp_path: Path,
    ):
        """Invariant: queue_dir copy wins when both copies pass the status filter.

        Both queue_dir and archive hold a resolved esc-42-1. After the fix
        (filter-before-dedup), the first copy that passes the filter wins.
        Because queue_dir is iterated first, the queue_dir copy must win.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Both copies are resolved — queue_dir copy must win by iteration order
        queue_dir_copy = _make_escalation('esc-42-1', task_id='42', status='resolved')
        queue_dir_copy.resolution = 'queue_dir_wins'
        (queue.queue_dir / 'esc-42-1.json').write_text(queue_dir_copy.to_json())

        archive_dir = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_copy = _make_escalation('esc-42-1', task_id='42', status='resolved')
        archive_copy.resolution = 'archive_loses'
        (archive_dir / 'esc-42-1.json').write_text(archive_copy.to_json())

        results = queue.get_by_task('42', status='resolved')

        assert len(results) == 1, (
            f'Expected exactly 1 result (deduped) but got {len(results)}: '
            f'{[e.resolution for e in results]}'
        )
        assert results[0].resolution == 'queue_dir_wins', (
            f"Expected queue_dir copy to win, got {results[0].resolution!r}"
        )


    def test_get_by_task_warns_when_id_in_both_queue_and_archive_regardless_of_filter(
        self, tmp_path: Path, caplog,
    ):
        """Pre-scan WARNING is emitted even when a filter is applied that selects
        only the archive copy (status='resolved').

        The pre-scan fires before the filter loop, so the operator sees the
        cross-tier duplicate regardless of which copy (if any) passes the filter.

        Failure mode on current main: no pre-scan exists; the per-call dedup
        WARNING only fires when the queue_dir copy passes the filter.
        With status='resolved', the pending queue_dir copy is excluded, so
        the per-call warning never fires and the cross-tier duplicate is invisible.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Stale pending copy stuck in queue_dir root
        pending_copy = _make_escalation('esc-42-1', task_id='42', status='pending')
        (queue.queue_dir / 'esc-42-1.json').write_text(pending_copy.to_json())

        # Resolved copy in archive (what the caller wants)
        archive_dir = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir.mkdir(parents=True, exist_ok=True)
        resolved_copy = _make_escalation('esc-42-1', task_id='42', status='resolved')
        resolved_copy.resolution = 'archive_copy'
        (archive_dir / 'esc-42-1.json').write_text(resolved_copy.to_json())

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            results = queue.get_by_task('42', status='resolved')

        # (a) Exactly one result — the archive's resolved copy
        assert len(results) == 1, (
            f'Expected exactly 1 resolved result, got {len(results)}: '
            f'{[e.resolution for e in results]}'
        )
        assert results[0].status == 'resolved'
        assert results[0].resolution == 'archive_copy'

        # (b) A WARNING mentioning the id must have been logged (pre-scan)
        warning_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue' and r.levelno >= logging.WARNING
        ]
        assert any('esc-42-1' in r.message for r in warning_records), (
            f"Expected a WARNING mentioning 'esc-42-1'; "
            f"got: {[r.message for r in warning_records]}"
        )

    def test_get_by_task_warns_on_cross_tier_duplicate_even_when_filter_excludes_both(
        self, tmp_path: Path, caplog,
    ):
        """Pre-scan WARNING fires even when the status filter excludes BOTH copies.

        Core contract: the pre-scan is independent of the filter loop.
        With status='dismissed', neither the pending queue_dir copy nor the
        resolved archive copy passes the filter, so no per-call dedup warning
        can fire (no entry ever hits the ``seen`` check). The pre-scan must
        therefore be the source of any id-mentioning WARNING in this scenario.

        Failure mode on current main: no pre-scan exists — results == [] with
        no WARNING, making the cross-tier duplicate invisible to the operator.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Stale pending copy stuck in queue_dir root
        pending_copy = _make_escalation('esc-42-1', task_id='42', status='pending')
        (queue.queue_dir / 'esc-42-1.json').write_text(pending_copy.to_json())

        # Resolved copy in archive
        archive_dir = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir.mkdir(parents=True, exist_ok=True)
        resolved_copy = _make_escalation('esc-42-1', task_id='42', status='resolved')
        resolved_copy.resolution = 'archive_copy'
        (archive_dir / 'esc-42-1.json').write_text(resolved_copy.to_json())

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            results = queue.get_by_task('42', status='dismissed')

        # (a) No copy passes the filter
        assert results == [], f'Expected [] (filter excludes both copies), got {results}'

        # (b) A WARNING mentioning the id must have been logged (pre-scan, not per-call dedup)
        warning_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue' and r.levelno >= logging.WARNING
        ]
        assert any('esc-42-1' in r.message for r in warning_records), (
            f"Expected a WARNING mentioning 'esc-42-1' from pre-scan; "
            f"got: {[r.message for r in warning_records]}"
        )


    def test_get_by_task_does_not_warn_on_archive_only_multi_date_duplicates(
        self, tmp_path: Path, caplog,
    ):
        """Pre-scan must NOT warn when the same id appears in two different archive
        date subdirs but has NO queue_dir root copy.

        Scenario: `esc-42-1` exists under `archive/2025-06-15/` and
        `archive/2025-06-16/` — both archive-tier, neither in queue_dir.
        This mirrors `get()`'s own handling of multi-archive-date candidates
        (queue.py:102-117): it treats the situation as a legitimate (if unusual)
        archive artefact and selects the newest date, not as a cross-tier error.

        Expected fail mode on current main: the pre-scan's `len(paths) >= 2` check
        sees two paths for `esc-42-1` and emits a misleading cross-tier WARNING
        ("exists in both queue_dir and archive") even though both paths are
        archive-only — the invariant comment ("any id with len(paths) >= 2
        necessarily spans both tiers") is false.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Two archive-dated copies of the same id — no queue_dir copy
        archive_dir_old = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir_old.mkdir(parents=True, exist_ok=True)
        esc_old = _make_escalation('esc-42-1', task_id='42', status='resolved')
        esc_old.resolution = 'older_copy'
        (archive_dir_old / 'esc-42-1.json').write_text(esc_old.to_json())

        archive_dir_new = queue.queue_dir / 'archive' / '2025-06-16'
        archive_dir_new.mkdir(parents=True, exist_ok=True)
        esc_new = _make_escalation('esc-42-1', task_id='42', status='resolved')
        esc_new.resolution = 'newer_copy'
        (archive_dir_new / 'esc-42-1.json').write_text(esc_new.to_json())

        # Confirm no queue_dir root copy exists
        assert not (queue.queue_dir / 'esc-42-1.json').exists()

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            results = queue.get_by_task('42')

        # (a) Exactly one result — archive-only duplicates must still be deduped
        ids = [e.id for e in results]
        assert ids.count('esc-42-1') == 1, (
            f'Expected exactly one esc-42-1 in results, got {ids}'
        )

        # (b) No WARNING mentioning esc-42-1 — both paths are archive-tier,
        #     so this is NOT a cross-tier duplicate; no reconciliation warning
        #     is needed (the situation is handled by get()'s date-sorting logic).
        warning_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue' and r.levelno >= logging.WARNING
        ]
        assert not any('esc-42-1' in r.message for r in warning_records), (
            'Expected NO WARNING mentioning esc-42-1 for archive-only multi-date '
            f'duplicates, but got: {[r.message for r in warning_records]}'
        )


    def test_get_by_task_does_not_warn_without_cross_tier_duplicate(
        self, tmp_path: Path, caplog,
    ):
        """Pre-scan must stay silent when no id spans both queue_dir root and archive.

        Setup: queue_dir root has one pending `esc-42-1`; archive has an unrelated
        resolved `esc-99-1` (different id, different task). No id appears in both
        tiers, so the pre-scan must emit zero WARNINGs.

        This closes the "silently-passes-if-pre-scan-warns-unconditionally" gap:
        a future regression that makes the pre-scan warn on any id with len>=1,
        or that re-introduces the len>=2 check without tier discrimination, would
        be caught here because esc-42-1 appears in only one tier and esc-99-1
        appears in only one (different) tier — neither is cross-tier.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # One pending copy of esc-42-1 in queue_dir root only
        pending_copy = _make_escalation('esc-42-1', task_id='42', status='pending')
        (queue.queue_dir / 'esc-42-1.json').write_text(pending_copy.to_json())

        # Unrelated resolved esc-99-1 in archive only
        archive_dir = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir.mkdir(parents=True, exist_ok=True)
        unrelated = _make_escalation('esc-99-1', task_id='99', status='resolved')
        (archive_dir / 'esc-99-1.json').write_text(unrelated.to_json())

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            results = queue.get_by_task('42')

        # (a) Exactly one result — the pending esc-42-1 (archive is scanned because
        #     no status filter, but esc-99-1 belongs to task '99', not '42')
        ids = [e.id for e in results]
        assert ids == ['esc-42-1'], (
            f'Expected exactly [esc-42-1], got {ids}'
        )

        # (b) No WARNING emitted — no id spans both tiers
        warning_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue' and r.levelno >= logging.WARNING
        ]
        assert warning_records == [], (
            'Expected NO WARNINGs when no id is cross-tier, '
            f'but got: {[r.message for r in warning_records]}'
        )


    def test_get_by_task_per_call_dedup_logs_at_debug_not_warning(
        self, tmp_path: Path, caplog,
    ):
        """Per-call dedup message is logged at DEBUG, NOT WARNING.

        Locks in amend b89a8fae80's WARNING→DEBUG demotion of the per-call dedup
        log path.  A regression restoring it to WARNING would be caught here.

        Setup: both queue_dir root AND archive/2025-06-15/ hold a resolved
        esc-42-1 (cross-tier duplicate).  With status='resolved', both copies
        pass the status filter, so the second one (archive copy) hits the `seen`
        dedup branch and the per-call debug message fires.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Cross-tier: resolved copy in queue_dir root (wins by iteration order)
        queue_dir_copy = _make_escalation('esc-42-1', task_id='42', status='resolved')
        queue_dir_copy.resolution = 'queue_dir_copy'
        (queue.queue_dir / 'esc-42-1.json').write_text(queue_dir_copy.to_json())

        # Resolved copy in archive (will be deduped; triggers per-call debug log)
        archive_dir = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_copy = _make_escalation('esc-42-1', task_id='42', status='resolved')
        archive_copy.resolution = 'archive_copy'
        (archive_dir / 'esc-42-1.json').write_text(archive_copy.to_json())

        with caplog.at_level(logging.DEBUG, logger='escalation.queue'):
            results = queue.get_by_task('42', status='resolved')

        # (a) Exactly one result — dedup still works; queue_dir copy wins
        assert len(results) == 1, (
            f'Expected exactly 1 result (deduped), got {len(results)}'
        )
        assert results[0].resolution == 'queue_dir_copy', (
            f"Expected queue_dir copy to win, got {results[0].resolution!r}"
        )

        # (b) At least one DEBUG record mentions 'Duplicate escalation id' and esc-42-1
        debug_dedup_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue'
            and r.levelno == logging.DEBUG
            and 'Duplicate escalation id' in r.message
            and 'esc-42-1' in r.message
        ]
        assert debug_dedup_records, (
            "Expected at least one DEBUG record with 'Duplicate escalation id' "
            f"mentioning esc-42-1, but got none. All records: "
            f"{[(r.levelno, r.message) for r in caplog.records if r.name == 'escalation.queue']}"
        )

        # (c) No per-call dedup message at WARNING level
        #     ('Duplicate escalation id' is the unique substring for the per-call path,
        #     distinct from the pre-scan's "exists in both queue_dir and archive" wording)
        warning_dedup_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue'
            and r.levelno == logging.WARNING
            and 'Duplicate escalation id' in r.message
        ]
        assert warning_dedup_records == [], (
            "Per-call dedup must log at DEBUG, not WARNING. "
            f"Found WARNING records: {[r.message for r in warning_dedup_records]}"
        )


    def test_get_by_task_warns_on_three_path_cross_tier_duplicate(
        self, tmp_path: Path, caplog,
    ):
        """Pre-scan must warn when the same id has one queue_dir root copy AND two
        archive-dated copies (3-path cross-tier case).

        This is the intersection of the two regressions this PR addresses:
        - The cross-tier detection must still fire (real reconciliation signal).
        - The legitimate multi-date-dir aspect of the archive copies must NOT
          suppress the warning — only archive-only duplicates are benign.

        Exactly one WARNING is expected (the pre-scan loop body runs once per id
        regardless of how many archive copies exist), and the log message must
        mention all three on-disk paths so an operator can trace the situation.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # One copy in queue_dir root (root tier)
        root_copy = _make_escalation('esc-42-1', task_id='42', status='pending')
        (queue.queue_dir / 'esc-42-1.json').write_text(root_copy.to_json())

        # Two archive-dated copies of the same id (archive tier)
        archive_dir_old = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir_old.mkdir(parents=True, exist_ok=True)
        arc_old = _make_escalation('esc-42-1', task_id='42', status='resolved')
        arc_old.resolution = 'older_archive'
        (archive_dir_old / 'esc-42-1.json').write_text(arc_old.to_json())

        archive_dir_new = queue.queue_dir / 'archive' / '2025-06-16'
        archive_dir_new.mkdir(parents=True, exist_ok=True)
        arc_new = _make_escalation('esc-42-1', task_id='42', status='resolved')
        arc_new.resolution = 'newer_archive'
        (archive_dir_new / 'esc-42-1.json').write_text(arc_new.to_json())

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            queue.get_by_task('42')

        # Exactly one WARNING for esc-42-1 (the pre-scan loop body runs once per id)
        warning_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue'
            and r.levelno == logging.WARNING
            and 'esc-42-1' in r.message
        ]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING for esc-42-1 cross-tier duplicate, '
            f'got {len(warning_records)}: {[r.message for r in warning_records]}'
        )

        # The warning message must reference all three paths so an operator can
        # trace which files need reconciliation
        msg = warning_records[0].message
        root_path = str(queue.queue_dir / 'esc-42-1.json')
        arc_old_path = str(archive_dir_old / 'esc-42-1.json')
        arc_new_path = str(archive_dir_new / 'esc-42-1.json')
        for expected_path in (root_path, arc_old_path, arc_new_path):
            assert expected_path in msg, (
                f'Expected path {expected_path!r} in WARNING message, but got: {msg!r}'
            )


    def test_get_by_task_pending_silent_on_cross_tier_duplicate(
        self, tmp_path: Path, caplog,
    ):
        """Pre-scan stays silent when status='pending' and a cross-tier duplicate exists.

        Characterizes the documented limitation at queue.py:148-150: when status ==
        'pending' the archive is skipped entirely, so a cross-tier duplicate is
        invisible to the pre-scan in that mode.

        Complements the sibling tests in this class:
        - does_not_warn_on_archive_only_multi_date_duplicates: archive-only is benign
        - does_not_warn_without_cross_tier_duplicate: no cross-tier → silent
        - warns_on_three_path_cross_tier_duplicate: real cross-tier → WARNING

        Regression prevented: a future refactor that extends `paths` with archive
        entries regardless of `status == 'pending'` would start warning here and this
        test would catch it.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Pending copy in queue_dir root
        pending_copy = _make_escalation('esc-42-1', task_id='42', status='pending')
        (queue.queue_dir / 'esc-42-1.json').write_text(pending_copy.to_json())

        # Resolved copy in archive with a sentinel marker to prove it is never surfaced
        archive_dir = queue.queue_dir / 'archive' / '2025-06-15'
        archive_dir.mkdir(parents=True, exist_ok=True)
        resolved_copy = _make_escalation('esc-42-1', task_id='42', status='resolved')
        resolved_copy.resolution = 'archive_copy'
        (archive_dir / 'esc-42-1.json').write_text(resolved_copy.to_json())

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            results = queue.get_by_task('42', status='pending')

        # (a) Exactly one result — the pending queue_dir root copy
        assert len(results) == 1, (
            f'Expected exactly 1 result (pending queue_dir copy), got {len(results)}'
        )

        # (b) The returned copy is the queue_dir pending one, NOT the archive copy.
        # Positive identity check: the pending copy has resolution=None (the dataclass
        # default).  If the archive copy leaked instead, resolution would be
        # 'archive_copy', making this assertion fail distinctly.
        assert results[0].id == 'esc-42-1'
        assert results[0].status == 'pending'
        assert results[0].resolution is None, (
            'Pending copy must have resolution=None; '
            f'got {results[0].resolution!r} — archive copy may have leaked'
        )

        # (c) No WARNING from the pre-scan — archive was never scanned, so the
        #     cross-tier duplicate was invisible to the pre-scan
        warning_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue' and r.levelno >= logging.WARNING
        ]
        assert not any('esc-42-1' in r.message for r in warning_records), (
            'Pre-scan must stay silent when status="pending" skips the archive; '
            f'offending WARNING records: {[r.message for r in warning_records]}'
        )


class TestGetPendingParseFailure:
    """get_pending() must emit a WARNING when a queue file cannot be parsed."""

    def test_get_pending_logs_warning_for_corrupt_file(self, tmp_path: Path, caplog):
        """Corrupt JSON file in queue_dir: get_pending() returns [] and logs a WARNING.

        Failure mode on current main: the except branch is silent — the corrupt
        file is silently dropped without any operator-visible log message.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Write a syntactically broken JSON file directly into queue_dir
        (queue.queue_dir / 'esc-1-1.json').write_text('not valid json')

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            results = queue.get_pending()

        # (a) Parse failure drops the entry — return value is empty
        assert results == [], f'Expected [] for corrupt file, got {results}'

        # (b) A WARNING must be emitted at logger 'escalation.queue'
        warning_records = [
            r for r in caplog.records
            if r.name == 'escalation.queue' and r.levelno >= logging.WARNING
        ]
        assert warning_records, (
            f"Expected a WARNING at logger 'escalation.queue'; got records: {caplog.records}"
        )
        assert any('Failed to parse' in r.message for r in warning_records), (
            f"Expected a WARNING containing 'Failed to parse'; "
            f"got: {[r.message for r in warning_records]}"
        )


class TestMakeIdAcrossArchive:
    """make_id() must consider archived sequence numbers to avoid post-restart collisions."""

    def test_make_id_does_not_collide_with_archived_after_restart(self, tmp_path: Path):
        """After all escalations are archived and process restarts, make_id() skips used IDs.

        Scenario:
        - Submit esc-42-1 and esc-42-2 for task '42', resolve both (moves to archive).
        - Simulate restart: create a new EscalationQueue (in-memory _seq resets to 0).
        - make_id('42') must return 'esc-42-3', NOT 'esc-42-1' which is already in archive.
        - Submitting that new escalation and calling get_by_task('42') returns three distinct IDs.
        """
        queue_dir = tmp_path / 'queue'

        # First process: submit two escalations and resolve (archive) both.
        queue = EscalationQueue(queue_dir)
        queue.submit(_make_escalation('esc-42-1', task_id='42'))
        queue.submit(_make_escalation('esc-42-2', task_id='42'))
        queue.resolve('esc-42-1', 'fixed first')
        queue.resolve('esc-42-2', 'fixed second')

        # Queue root should now be empty for task 42.
        assert not (queue_dir / 'esc-42-1.json').exists()
        assert not (queue_dir / 'esc-42-2.json').exists()

        # Simulate process restart: fresh EscalationQueue, _seq resets to 0.
        queue2 = EscalationQueue(queue_dir)

        # make_id() MUST return 'esc-42-3', not 'esc-42-1'.
        new_id = queue2.make_id('42')
        assert new_id == 'esc-42-3', (
            f'Expected esc-42-3 (avoids archive collision) but got {new_id!r}'
        )

        # Submit the new escalation and verify all three are visible via get_by_task.
        new_esc = _make_escalation(new_id, task_id='42')
        queue2.submit(new_esc)

        all_escs = queue2.get_by_task('42')
        all_ids = {e.id for e in all_escs}
        assert all_ids == {'esc-42-1', 'esc-42-2', 'esc-42-3'}, (
            f'Expected three distinct IDs but got: {all_ids}'
        )

    @pytest.mark.parametrize(
        'archived_seq, pending_seq, expected_next',
        [
            (5, 2, 6),  # archive seq > pending seq (interleaved scenario)
            (3, 7, 8),  # pending seq > archive seq (symmetric scenario)
        ],
    )
    def test_make_id_takes_max_across_archive_and_pending(
        self,
        tmp_path: Path,
        archived_seq: int,
        pending_seq: int,
        expected_next: int,
    ):
        """make_id() takes max across BOTH archive and queue root in either ordering.

        The parametrization captures two orderings:
        - Row (5, 2, 6): archive seq (5) > pending seq (2) → next = 6
        - Row (3, 7, 8): pending seq (7) > archive seq (3) → next = 8

        In both cases make_id() must return max(archive_max, pending_max) + 1,
        proving the two-loop scan in queue.py:265-274 works regardless of which
        source holds the higher sequence number.

        Regression guard for the two-source max logic in make_id().
        """
        queue_dir = tmp_path / 'queue'
        archived_id = f'esc-42-{archived_seq}'
        pending_id = f'esc-42-{pending_seq}'

        # First process: submit archived_id and resolve it (moves to archive).
        queue = EscalationQueue(queue_dir)
        queue.submit(_make_escalation(archived_id, task_id='42'))
        queue.resolve(archived_id, f'fixed {archived_seq}')

        # Submit pending_id and leave it in queue root.
        queue.submit(_make_escalation(pending_id, task_id='42'))

        # Sanity: archived_id is in archive, pending_id is in queue root.
        assert not (queue_dir / f'{archived_id}.json').exists(), (
            f'{archived_id} should be archived'
        )
        assert (queue_dir / f'{pending_id}.json').exists(), (
            f'{pending_id} should be pending in queue root'
        )

        # Simulate process restart: fresh EscalationQueue, _seq resets to 0.
        queue2 = EscalationQueue(queue_dir)

        # make_id() MUST return expected_next: max(archive, pending) + 1.
        new_id = queue2.make_id('42')
        assert new_id == f'esc-42-{expected_next}', (
            f'Expected esc-42-{expected_next} '
            f'(max across archive={archived_seq} and queue_root={pending_seq}) '
            f'but got {new_id!r}'
        )


class TestIterAllEscalationPaths:
    """iter_all_escalation_paths(escalations_dir) yields all esc-*.json paths,
    scanning queue root first then archive subtree, deduplicating by stem with
    queue root taking precedence over archive on id collisions.
    """

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        """(a) Non-existent escalations_dir -> empty iterator, no raise."""
        result = list(iter_all_escalation_paths(tmp_path / 'does_not_exist'))
        assert result == []

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        """(b) Empty escalations_dir (exists, no files) -> empty iterator."""
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        result = list(iter_all_escalation_paths(esc_dir))
        assert result == []

    def test_root_only_yields_matching_files(self, tmp_path: Path):
        """(c) Root-only: two esc-*.json in root, non-matching file not yielded."""
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        esc1 = _make_escalation('esc-1-1', task_id='1')
        esc2 = _make_escalation('esc-2-1', task_id='2')
        (esc_dir / 'esc-1-1.json').write_text(esc1.to_json())
        (esc_dir / 'esc-2-1.json').write_text(esc2.to_json())
        (esc_dir / 'other.json').write_text('{"not": "an escalation"}')

        result = list(iter_all_escalation_paths(esc_dir))
        stems = {p.stem for p in result}
        assert stems == {'esc-1-1', 'esc-2-1'}, (
            f'Expected only esc-1-1 and esc-2-1, got: {stems}'
        )
        assert all(p.parent == esc_dir for p in result), (
            'Root-only files should have parent == esc_dir'
        )

    def test_archive_only_yields_archived_files(self, tmp_path: Path):
        """(d) archive-only: files under archive/YYYY-MM-DD/ are yielded."""
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        arc1_dir = esc_dir / 'archive' / '2026-04-20'
        arc2_dir = esc_dir / 'archive' / '2026-04-21'
        arc1_dir.mkdir(parents=True)
        arc2_dir.mkdir(parents=True)

        esc1 = _make_escalation('esc-1-1', task_id='1')
        esc2 = _make_escalation('esc-2-1', task_id='2')
        (arc1_dir / 'esc-1-1.json').write_text(esc1.to_json())
        (arc2_dir / 'esc-2-1.json').write_text(esc2.to_json())

        result = list(iter_all_escalation_paths(esc_dir))
        stems = {p.stem for p in result}
        assert stems == {'esc-1-1', 'esc-2-1'}, (
            f'Expected esc-1-1 and esc-2-1 from archive, got: {stems}'
        )

    def test_dedup_root_wins_over_archive(self, tmp_path: Path):
        """(e) Same esc id in root and archive: root path is yielded (root wins)."""
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        arc_dir = esc_dir / 'archive' / '2026-04-20'
        arc_dir.mkdir(parents=True)

        esc = _make_escalation('esc-42-1', task_id='42')
        (esc_dir / 'esc-42-1.json').write_text(esc.to_json())
        (arc_dir / 'esc-42-1.json').write_text(esc.to_json())

        result = list(iter_all_escalation_paths(esc_dir))
        stems = [p.stem for p in result]
        assert stems.count('esc-42-1') == 1, (
            f'Expected exactly one esc-42-1 in results, got {stems}'
        )
        # The yielded path must be the root copy (its parent is esc_dir)
        yielded = next(p for p in result if p.stem == 'esc-42-1')
        assert yielded.parent == esc_dir, (
            f'Root copy must win; expected parent={esc_dir}, got {yielded.parent}'
        )

    def test_archive_only_multidate_dedup(self, tmp_path: Path):
        """(f) Same stem in two archive dated subdirs (no root copy): yielded exactly once."""
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        arc1_dir = esc_dir / 'archive' / '2026-04-20'
        arc2_dir = esc_dir / 'archive' / '2026-04-21'
        arc1_dir.mkdir(parents=True)
        arc2_dir.mkdir(parents=True)

        esc = _make_escalation('esc-42-1', task_id='42')
        (arc1_dir / 'esc-42-1.json').write_text(esc.to_json())
        (arc2_dir / 'esc-42-1.json').write_text(esc.to_json())

        # Confirm no root copy
        assert not (esc_dir / 'esc-42-1.json').exists()

        result = list(iter_all_escalation_paths(esc_dir))
        stems = [p.stem for p in result]
        assert stems.count('esc-42-1') == 1, (
            f'Archive-only multi-date duplicate must be yielded exactly once, got: {stems}'
        )


class TestPatchResolutionMetadata:
    """EscalationQueue.patch_resolution_metadata() — archive-aware in-place metadata patch.

    Core contract: patching an archived escalation rewrites the archive copy in place;
    it NEVER creates a file in the queue root (no resurrection).
    """

    def test_patch_archived_no_resurrection_and_returns_updated_with_preserved_fields(
        self, tmp_path: Path,
    ):
        """Regression test: patching resolved_by on an archived escalation must rewrite
        the archive copy and MUST NOT create a file in the queue root.

        Failure mode (the bug): _rewrite() delegates to _atomic_write() which always
        writes to queue_dir/{id}.json (the ROOT), resurrecting the file out of the archive.
        """
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))

        # Resolve: file moves to archive/YYYY-MM-DD/esc-1-1.json
        queue.resolve('esc-1-1', 'Original resolution')

        # Confirm the escalation is in archive, root is empty
        assert not (queue.queue_dir / 'esc-1-1.json').exists(), (
            'Pre-condition: file must not be in queue root after resolve()'
        )
        archive_files_before = list((queue.queue_dir / 'archive').rglob('esc-1-1.json'))
        assert len(archive_files_before) == 1, (
            f'Pre-condition: expected exactly 1 archive file; got {archive_files_before}'
        )

        # Capture pre-patch fields for "unchanged" assertions
        pre_patch = queue.get('esc-1-1')
        assert pre_patch is not None
        pre_resolved_at = pre_patch.resolved_at
        pre_level = pre_patch.level
        pre_task_id = pre_patch.task_id
        pre_status = pre_patch.status

        # --- Action ---
        result = queue.patch_resolution_metadata('esc-1-1', resolved_by='steward', resolution_turns=5)

        # (a) NO FILE RESURRECTION — root stays clean
        assert not (queue.queue_dir / 'esc-1-1.json').exists(), (
            'RESURRECTION BUG: patch_resolution_metadata wrote to queue root; '
            'the archive copy should have been rewritten in place instead'
        )

        # (b) Exactly one archive file; patched fields present
        archive_files_after = list((queue.queue_dir / 'archive').rglob('esc-1-1.json'))
        assert len(archive_files_after) == 1, (
            f'Expected exactly 1 archive file after patch; got {archive_files_after}'
        )
        archived_data = json.loads(archive_files_after[0].read_text())
        assert archived_data['resolved_by'] == 'steward', (
            f"Expected resolved_by='steward' in archive; got {archived_data.get('resolved_by')!r}"
        )
        assert archived_data['resolution_turns'] == 5, (
            f"Expected resolution_turns=5 in archive; got {archived_data.get('resolution_turns')!r}"
        )

        # (c) Preserved fields unchanged
        assert archived_data['status'] == 'resolved', (
            f"Status must remain 'resolved'; got {archived_data['status']!r}"
        )
        assert archived_data['resolution'] == 'Original resolution', (
            f"Resolution must be preserved; got {archived_data.get('resolution')!r}"
        )
        assert archived_data['resolved_at'] == pre_resolved_at, (
            f'resolved_at must be unchanged; pre={pre_resolved_at!r}, '
            f'post={archived_data.get("resolved_at")!r}'
        )
        assert archived_data['level'] == pre_level
        assert archived_data['task_id'] == pre_task_id
        assert archived_data['status'] == pre_status

        # (d) Return value is the updated Escalation
        assert result is not None, 'patch_resolution_metadata must return the updated Escalation'
        assert result.resolved_by == 'steward', (
            f"result.resolved_by must be 'steward'; got {result.resolved_by!r}"
        )
        assert result.resolution_turns == 5, (
            f'result.resolution_turns must be 5; got {result.resolution_turns!r}'
        )
        assert result.status == 'resolved', (
            f"result.status must be 'resolved'; got {result.status!r}"
        )


    def test_patch_pending_returns_none_no_mutation(self, tmp_path: Path):
        """(a) Patching a pending escalation returns None and leaves the file unchanged."""
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))

        before = (queue.queue_dir / 'esc-1-1.json').read_text()

        result = queue.patch_resolution_metadata('esc-1-1', resolved_by='steward', resolution_turns=5)

        # Must return None for pending escalations
        assert result is None, (
            f'Expected None for pending escalation; got {result!r}'
        )

        # File must be byte-for-byte unchanged
        after = (queue.queue_dir / 'esc-1-1.json').read_text()
        assert after == before, (
            'File must be unchanged when patching a pending escalation'
        )

        # In-file resolved_by must still be None
        data = json.loads(after)
        assert data['resolved_by'] is None, (
            f"resolved_by should still be None; got {data['resolved_by']!r}"
        )

    def test_patch_nonexistent_returns_none_no_files_created(self, tmp_path: Path):
        """(b) Patching a non-existent id returns None and creates no files."""
        queue = EscalationQueue(tmp_path / 'queue')

        result = queue.patch_resolution_metadata('esc-does-not-exist', resolved_by='steward')

        assert result is None, (
            f'Expected None for non-existent escalation; got {result!r}'
        )
        assert list(queue.queue_dir.glob('*.json')) == [], (
            'No JSON files should be created for a non-existent id'
        )
        assert not (queue.queue_dir / 'archive').exists(), (
            'Archive dir must not be created for a non-existent id'
        )

    def test_patch_multi_date_archive_picks_newest_and_root_stays_clean(
        self, tmp_path: Path,
    ):
        """Multi-date archive: patch picks the newest copy and root stays clean.

        This drives extraction of the newest-by-date selection into a shared helper
        (_locate_path) used by both get() and patch_resolution_metadata().
        A naive candidates[0] pick would non-deterministically select either copy;
        this test asserts the newest copy is always patched.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Manually create two archive files for the same id under different dated subdirs
        older_dir = queue.queue_dir / 'archive' / '2025-01-01'
        newer_dir = queue.queue_dir / 'archive' / '2025-06-15'
        older_dir.mkdir(parents=True, exist_ok=True)
        newer_dir.mkdir(parents=True, exist_ok=True)

        older_esc = _make_escalation('esc-42-1', status='resolved')
        older_esc.resolution = 'older_copy'
        older_text_before = older_esc.to_json()
        (older_dir / 'esc-42-1.json').write_text(older_text_before)

        newer_esc = _make_escalation('esc-42-1', status='resolved')
        newer_esc.resolution = 'newer_copy'
        (newer_dir / 'esc-42-1.json').write_text(newer_esc.to_json())

        # Confirm no root file
        assert not (queue.queue_dir / 'esc-42-1.json').exists()

        # --- Action ---
        result = queue.patch_resolution_metadata('esc-42-1', resolved_by='steward', resolution_turns=3)

        # (a) Root stays clean
        assert not (queue.queue_dir / 'esc-42-1.json').exists(), (
            'Root must stay clean after patching an archive-resident escalation'
        )

        # (b) Newer copy is patched
        newer_data = json.loads((newer_dir / 'esc-42-1.json').read_text())
        assert newer_data['resolved_by'] == 'steward', (
            f"Expected newest copy to have resolved_by='steward'; got {newer_data.get('resolved_by')!r}"
        )
        assert newer_data['resolution_turns'] == 3, (
            f"Expected newest copy to have resolution_turns=3; got {newer_data.get('resolution_turns')!r}"
        )
        assert newer_data['resolution'] == 'newer_copy', (
            f"Payload must be preserved: expected resolution='newer_copy'; got {newer_data.get('resolution')!r}"
        )

        # (c) Older copy is byte-for-byte unchanged
        older_text_after = (older_dir / 'esc-42-1.json').read_text()
        assert older_text_after == older_text_before, (
            'Older archive copy must be byte-for-byte unchanged'
        )

        # (d) Return value reflects the patch
        assert result is not None
        assert result.resolved_by == 'steward', (
            f"result.resolved_by must be 'steward'; got {result.resolved_by!r}"
        )

    def test_patch_archive_atomicity_tmp_cleanup_on_rename_failure(
        self, tmp_path: Path,
    ):
        """When os.rename raises (e.g. disk full), the tmp file is cleaned up
        and the exception propagates.

        Key assertion: NO .tmp files remain in archive_dir OR queue root after failure.
        This catches the failure mode where the tmp file gets created in the queue root
        (wrong dir) and then orphaned when rename fails.
        """
        queue = EscalationQueue(tmp_path / 'queue')
        queue.submit(_make_escalation('esc-1-1'))
        queue.resolve('esc-1-1', 'archived')

        # Locate the archive dated subdir
        archive_files = list((queue.queue_dir / 'archive').rglob('esc-1-1.json'))
        assert len(archive_files) == 1, 'Pre-condition: escalation must be archived'
        archive_dir = archive_files[0].parent
        original_content = archive_files[0].read_text()

        # Patch os.rename to fail
        with patch('escalation.queue.os.rename', side_effect=OSError('disk full')), pytest.raises(OSError, match='disk full'):
            queue.patch_resolution_metadata('esc-1-1', resolved_by='steward')

        # (a) No .tmp files orphaned in archive_dir
        tmp_files_in_archive = list(archive_dir.glob('esc-1-1*.tmp'))
        assert tmp_files_in_archive == [], (
            f'Orphaned .tmp files in archive_dir: {tmp_files_in_archive}; '
            'tmp file must be created in path.parent and cleaned up on failure'
        )

        # (b) No .tmp files orphaned in queue root — key regression check
        # (if tmp was created in queue root instead of archive_dir, it would be orphaned here)
        tmp_files_in_root = list(queue.queue_dir.glob('esc-1-1*.tmp'))
        assert tmp_files_in_root == [], (
            f'Orphaned .tmp files in queue root: {tmp_files_in_root}; '
            'tmp file must NOT be created in queue root for archive-resident targets'
        )

        # (c) Original archive file content is unchanged (rename never happened)
        assert archive_files[0].read_text() == original_content, (
            'Original archive file must be unchanged after rename failure'
        )

    def test_patch_resolved_in_root_rewrites_root_in_place_no_archive_orphan(
        self, tmp_path: Path,
    ):
        """(c) When a resolved escalation is in queue root (archive move failed),
        patch rewrites the root file in place and creates NO archive file.
        """
        queue = EscalationQueue(tmp_path / 'queue')

        # Write a resolved escalation directly to root (simulating crash-mid-archive)
        resolved_esc = _make_escalation('esc-1-1', status='resolved')
        resolved_esc.resolution = 'stuck in root'
        (queue.queue_dir / 'esc-1-1.json').write_text(resolved_esc.to_json())

        result = queue.patch_resolution_metadata('esc-1-1', resolved_by='steward')

        # (i) File stays in root with updated field
        assert (queue.queue_dir / 'esc-1-1.json').exists(), (
            'Root file must still exist after patching'
        )
        data = json.loads((queue.queue_dir / 'esc-1-1.json').read_text())
        assert data['resolved_by'] == 'steward', (
            f"resolved_by must be 'steward' in root file; got {data.get('resolved_by')!r}"
        )

        # (ii) No archive file created (no orphan move)
        assert not (queue.queue_dir / 'archive').exists(), (
            'No archive dir should be created when root file is patched in place'
        )

        assert result is not None
        assert result.resolved_by == 'steward'


class TestAttachDedupeChild:
    """EscalationQueue.attach_dedupe_child() — mutate a pending parent in place."""

    def _make_infra_esc(self, esc_id: str, task_id: str = '1') -> Escalation:
        from escalation.models import Escalation
        return Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )

    def test_attach_first_child_returns_updated_escalation(self, tmp_path: Path):
        """(a) submit parent, attach one child: returns updated Escalation;
        parent on disk has dedupe_count==1 and dedupe_children==['esc-2-1']."""
        queue = EscalationQueue(tmp_path / 'esc')
        queue.submit(self._make_infra_esc('esc-1-1'))

        result = queue.attach_dedupe_child('esc-1-1', 'esc-2-1')

        assert result is not None
        assert result.dedupe_count == 1
        assert result.dedupe_children == ['esc-2-1']

        # Verify the file on disk reflects the mutation
        from_disk = queue.get('esc-1-1')
        assert from_disk is not None
        assert from_disk.dedupe_count == 1
        assert from_disk.dedupe_children == ['esc-2-1']

    def test_attach_second_child_accumulates(self, tmp_path: Path):
        """(b) Two attach calls accumulate: dedupe_count==2, dedupe_children has both."""
        queue = EscalationQueue(tmp_path / 'esc')
        queue.submit(self._make_infra_esc('esc-1-1'))

        queue.attach_dedupe_child('esc-1-1', 'esc-2-1')
        result = queue.attach_dedupe_child('esc-1-1', 'esc-3-1')

        assert result is not None
        assert result.dedupe_count == 2
        assert result.dedupe_children == ['esc-2-1', 'esc-3-1']

        from_disk = queue.get('esc-1-1')
        assert from_disk is not None
        assert from_disk.dedupe_count == 2
        assert from_disk.dedupe_children == ['esc-2-1', 'esc-3-1']

    def test_attach_to_nonexistent_parent_returns_none(self, tmp_path: Path):
        """(c) Attaching to a non-existent parent returns None and writes no new files."""
        queue = EscalationQueue(tmp_path / 'esc')

        result = queue.attach_dedupe_child('does-not-exist', 'esc-9-1')

        assert result is None
        # No new JSON files should exist in queue_dir
        files = list(queue.queue_dir.glob('*.json'))
        assert files == [], f'Expected no files, found: {files}'

    def test_attach_to_resolved_parent_returns_none(self, tmp_path: Path):
        """(d) Attaching to a resolved parent (in archive) returns None — no mutation."""
        queue = EscalationQueue(tmp_path / 'esc')
        queue.submit(self._make_infra_esc('esc-1-1'))
        queue.resolve('esc-1-1', 'fixed already')

        # Parent is now in archive, not in queue root
        assert not (queue.queue_dir / 'esc-1-1.json').exists()

        result = queue.attach_dedupe_child('esc-1-1', 'esc-2-1')

        assert result is None

    # --- Severity-promotion quadrant tests ---

    def test_attach_info_into_info_parent_keeps_info(self, tmp_path: Path):
        """(e) info child into info parent: parent.severity stays 'info'."""
        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        parent.severity = 'info'
        queue.submit(parent)

        result = queue.attach_dedupe_child('esc-1-1', 'esc-2-1', child_severity='info')

        assert result is not None
        assert result.severity == 'info'
        assert queue.get('esc-1-1').severity == 'info'  # type: ignore[union-attr]

    def test_attach_blocker_into_info_parent_promotes_to_blocker(self, tmp_path: Path):
        """(f) blocking child into info parent: parent.severity promoted to 'blocking'."""
        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        parent.severity = 'info'
        queue.submit(parent)

        result = queue.attach_dedupe_child('esc-1-1', 'esc-2-1', child_severity='blocking')

        assert result is not None
        assert result.severity == 'blocking'
        # Verify promotion is persisted to disk
        assert queue.get('esc-1-1').severity == 'blocking'  # type: ignore[union-attr]

    def test_attach_info_into_blocker_parent_keeps_blocker(self, tmp_path: Path):
        """(g) info child into blocking parent: parent.severity stays 'blocking'."""
        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        parent.severity = 'blocking'
        queue.submit(parent)

        result = queue.attach_dedupe_child('esc-1-1', 'esc-2-1', child_severity='info')

        assert result is not None
        assert result.severity == 'blocking'
        assert queue.get('esc-1-1').severity == 'blocking'  # type: ignore[union-attr]

    def test_attach_blocker_into_blocker_parent_keeps_blocker(self, tmp_path: Path):
        """(h) blocking child into blocking parent: parent.severity stays 'blocking'."""
        queue = EscalationQueue(tmp_path / 'esc')
        parent = self._make_infra_esc('esc-1-1')
        parent.severity = 'blocking'
        queue.submit(parent)

        result = queue.attach_dedupe_child('esc-1-1', 'esc-2-1', child_severity='blocking')

        assert result is not None
        assert result.severity == 'blocking'
        assert queue.get('esc-1-1').severity == 'blocking'  # type: ignore[union-attr]


class TestFindPendingL2ByRootCause:
    """EscalationQueue.find_pending_l2_by_root_cause() locates a pending L2 by root_cause string."""

    def _make_l2(self, queue: EscalationQueue, task_id: str, root_cause: str) -> Escalation:
        """Submit a pending L2 escalation with the given root_cause."""
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='L2 cluster test',
            level=2,
            root_cause=root_cause,
        )
        queue.submit(esc)
        return esc

    def test_returns_id_of_matching_pending_l2(self, tmp_path: Path):
        """(a) Returns the id of a pending L2 whose root_cause matches."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue, 'task-1', 'Bad merge strategy')

        result = queue.find_pending_l2_by_root_cause('Bad merge strategy')

        assert result == l2.id

    def test_returns_none_when_no_l2_has_root_cause(self, tmp_path: Path):
        """(b) Returns None when no L2 has the given root_cause."""
        queue = EscalationQueue(tmp_path / 'esc')
        self._make_l2(queue, 'task-1', 'Some other root cause')

        result = queue.find_pending_l2_by_root_cause('Bad merge strategy')

        assert result is None

    def test_returns_none_when_only_l0_l1_has_root_cause(self, tmp_path: Path):
        """(c) Returns None when only an L0 or L1 has the root_cause (level filter)."""
        queue = EscalationQueue(tmp_path / 'esc')
        # L0 with matching root_cause
        l0 = Escalation(
            id=queue.make_id('task-1'),
            task_id='task-1',
            agent_role='implementer',
            severity='blocking',
            category='design_concern',
            summary='L0 test',
            level=0,
            root_cause='Bad merge strategy',
        )
        queue.submit(l0)
        # L1 with matching root_cause
        l1 = Escalation(
            id=queue.make_id('task-2'),
            task_id='task-2',
            agent_role='steward',
            severity='blocking',
            category='design_concern',
            summary='L1 test',
            level=1,
            root_cause='Bad merge strategy',
        )
        queue.submit(l1)

        result = queue.find_pending_l2_by_root_cause('Bad merge strategy')

        assert result is None, f'Expected None (level filter), got {result!r}'

    def test_returns_none_when_matching_l2_is_already_resolved(self, tmp_path: Path):
        """(d) Returns None when the matching L2 is already resolved (status filter)."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue, 'task-1', 'Bad merge strategy')
        queue.resolve(l2.id, 'Fixed')

        result = queue.find_pending_l2_by_root_cause('Bad merge strategy')

        assert result is None

    def test_returns_oldest_when_multiple_pending_l2s_share_root_cause(self, tmp_path: Path):
        """(e) Returns the OLDEST (by timestamp) when multiple pending L2s share root_cause."""
        import time
        queue = EscalationQueue(tmp_path / 'esc')
        older = self._make_l2(queue, 'task-1', 'Bad merge strategy')
        time.sleep(0.01)  # ensure distinct timestamps
        newer = self._make_l2(queue, 'task-2', 'Bad merge strategy')

        result = queue.find_pending_l2_by_root_cause('Bad merge strategy')

        assert result == older.id, (
            f'Expected oldest id={older.id!r}, got {result!r} (newer={newer.id!r})'
        )

    def test_returns_none_for_empty_root_cause(self, tmp_path: Path):
        """(f) Returns None for an empty root_cause (never match)."""
        queue = EscalationQueue(tmp_path / 'esc')
        self._make_l2(queue, 'task-1', '')  # L2 with empty root_cause

        result = queue.find_pending_l2_by_root_cause('')

        assert result is None

    def test_returns_none_for_whitespace_only_root_cause(self, tmp_path: Path):
        """(f cont.) Returns None for whitespace-only root_cause (never match)."""
        queue = EscalationQueue(tmp_path / 'esc')
        self._make_l2(queue, 'task-1', 'Bad merge strategy')

        result = queue.find_pending_l2_by_root_cause('   ')

        assert result is None

    def test_match_is_whitespace_trim_tolerant(self, tmp_path: Path):
        """(g) Trailing/leading whitespace in the query is stripped — 'Bad merge' matches 'Bad merge '."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue, 'task-1', 'Bad merge strategy')

        result = queue.find_pending_l2_by_root_cause('Bad merge strategy  ')

        assert result == l2.id, (
            f'Expected match despite trailing whitespace; got {result!r}'
        )

    def test_l2_corrupt_timestamp_still_considered_with_warning(
        self, tmp_path: Path, caplog,
    ):
        """(h) Corrupt timestamp on a matching L2 → still returned AND a WARNING is emitted.

        Bug in current code: silent datetime.min substitution produces no log output.
        The entry is still considered (datetime.min sorts oldest, is included), but
        operators can't see the data-quality issue.

        Fix: parse_timestamp_or_warn emits a WARNING mentioning the context string.
        RED assertion: zero warnings → assertion (b) fails.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = Escalation(
            id=queue.make_id('task-99'),
            task_id='task-99',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='L2 cluster corrupt-ts test',
            level=2,
            root_cause='Bad merge strategy',
        )
        l2.timestamp = 'not-a-timestamp'
        queue.submit(l2)

        with caplog.at_level(logging.WARNING, logger='escalation.queue'):
            result = queue.find_pending_l2_by_root_cause('Bad merge strategy')

        # (a) Entry still CONSIDERED — datetime.min sorts oldest, included in results
        assert result == l2.id, (
            f'Expected corrupt-ts L2 to still be returned; got result={result!r}'
        )

        # (b) A WARNING must be emitted mentioning the context tag
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'queue.find_pending_l2_by_root_cause' in r.message
        ]
        assert warning_records, (
            f"Expected >=1 WARNING mentioning 'queue.find_pending_l2_by_root_cause'; "
            f"got caplog.records: {[(r.levelname, r.message) for r in caplog.records]}"
        )


class TestAddMembersToL2:
    """EscalationQueue.add_members_to_l2() appends member ids to a pending L2."""

    def _make_l2(self, queue: EscalationQueue, task_id: str = 'task-1') -> Escalation:
        """Submit a pending L2 escalation with an initial member."""
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='L2 cluster',
            level=2,
            root_cause='Bad merge strategy',
            members=['esc-l1-0'],
        )
        queue.submit(esc)
        return esc

    def test_returns_updated_escalation(self, tmp_path: Path):
        """(a) Returns the updated Escalation object."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue)

        result = queue.add_members_to_l2(l2.id, ['esc-l1-1'])

        assert result is not None, 'Expected updated Escalation, got None'
        assert isinstance(result, Escalation)

    def test_appends_new_members(self, tmp_path: Path):
        """(b) Appends new member ids, preserving order — no duplicates (set-union)."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue)  # already has 'esc-l1-0'

        result = queue.add_members_to_l2(l2.id, ['esc-l1-1', 'esc-l1-2'])

        assert result is not None
        assert 'esc-l1-0' in result.members  # original preserved
        assert 'esc-l1-1' in result.members  # new member added
        assert 'esc-l1-2' in result.members  # new member added
        assert len(result.members) == 3, f'Expected 3 members, got {result.members}'

    def test_set_union_semantics_no_duplicates(self, tmp_path: Path):
        """(b cont.) Adding an already-existing member id does NOT duplicate it."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue)  # already has 'esc-l1-0'

        result = queue.add_members_to_l2(l2.id, ['esc-l1-0', 'esc-l1-1'])

        assert result is not None
        assert result.members.count('esc-l1-0') == 1, 'Duplicate members must not be added'
        assert 'esc-l1-1' in result.members
        assert len(result.members) == 2

    def test_leaves_other_fields_untouched(self, tmp_path: Path):
        """(c) root_cause, options, summary, detail, timestamp are not modified."""
        queue = EscalationQueue(tmp_path / 'esc')
        esc = Escalation(
            id=queue.make_id('task-1'),
            task_id='task-1',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='Original summary',
            level=2,
            root_cause='Bad merge strategy',
            options=['A: fix', 'B: rollback'],
            detail='Original evidence',
            members=['esc-l1-0'],
        )
        queue.submit(esc)
        original_ts = esc.timestamp

        result = queue.add_members_to_l2(esc.id, ['esc-l1-1'])

        assert result is not None
        assert result.root_cause == 'Bad merge strategy'
        assert result.options == ['A: fix', 'B: rollback']
        assert result.summary == 'Original summary'
        assert result.detail == 'Original evidence'
        assert result.timestamp == original_ts

    def test_returns_none_when_l2_not_found_in_queue_root(self, tmp_path: Path):
        """(d) Returns None when the L2 is archived or unknown (refuses archived L2s)."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue)
        queue.resolve(l2.id, 'Fixed')  # moves to archive

        result = queue.add_members_to_l2(l2.id, ['esc-l1-1'])

        assert result is None, f'Expected None for archived L2, got {result!r}'

    def test_returns_none_for_unknown_id(self, tmp_path: Path):
        """(d cont.) Returns None for a completely unknown id."""
        queue = EscalationQueue(tmp_path / 'esc')

        result = queue.add_members_to_l2('esc-does-not-exist', ['esc-l1-1'])

        assert result is None, f'Expected None for unknown id, got {result!r}'

    def test_empty_new_member_ids_is_noop(self, tmp_path: Path):
        """(e) add_members_to_l2 with empty new_member_ids returns escalation unchanged."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue)  # has 'esc-l1-0'

        result = queue.add_members_to_l2(l2.id, [])

        assert result is not None
        assert result.members == ['esc-l1-0'], f'Expected unchanged members, got {result.members}'

    def test_write_is_durable_reload_shows_appended_members(self, tmp_path: Path):
        """(f) Reload via queue.get(l2_id) shows the appended members."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue)

        queue.add_members_to_l2(l2.id, ['esc-l1-1', 'esc-l1-2'])

        reloaded = queue.get(l2.id)
        assert reloaded is not None
        assert 'esc-l1-0' in reloaded.members
        assert 'esc-l1-1' in reloaded.members
        assert 'esc-l1-2' in reloaded.members
        assert len(reloaded.members) == 3

    def test_internal_duplicates_in_new_member_ids_are_deduplicated(self, tmp_path: Path):
        """Suggestion-3: new_member_ids=['a','a','b'] adds 'a' exactly once.

        Set-union is only against *existing* members; if new_member_ids itself
        contains duplicates they must be collapsed before the union so each id
        appears at most once in the resulting members list.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2(queue)  # already has 'esc-l1-0'

        result = queue.add_members_to_l2(l2.id, ['esc-l1-1', 'esc-l1-1', 'esc-l1-2'])

        assert result is not None
        assert result.members.count('esc-l1-1') == 1, (
            f"'esc-l1-1' must appear exactly once, got {result.members}"
        )
        assert 'esc-l1-2' in result.members
        assert result.members.count('esc-l1-0') == 1  # original untouched
        assert len(result.members) == 3, (
            f'Expected 3 unique members (esc-l1-0, esc-l1-1, esc-l1-2), got {result.members}'
        )


class TestResolveCascade:
    """EscalationQueue.resolve() cascades resolution to member L1s when L2 has members."""

    def _make_l1(self, queue: EscalationQueue, esc_id: str, task_id: str = 'task-1') -> Escalation:
        """Submit a pending L1 escalation."""
        esc = Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='steward',
            severity='blocking',
            category='design_concern',
            summary='L1 test escalation',
            level=1,
        )
        queue.submit(esc)
        return esc

    def _make_l2_with_members(
        self,
        queue: EscalationQueue,
        l2_id: str,
        member_ids: list[str],
    ) -> Escalation:
        """Submit a pending L2 escalation referencing the given member ids."""
        esc = Escalation(
            id=l2_id,
            task_id='task-cluster',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='L2 cluster',
            level=2,
            root_cause='Bad merge strategy',
            members=list(member_ids),
        )
        queue.submit(esc)
        return esc

    def test_resolve_l2_cascades_resolution_to_members(self, tmp_path: Path):
        """(a) resolve(l2_id) with members=[m1,m2] cascades — m1 and m2 are resolved."""
        queue = EscalationQueue(tmp_path / 'esc')
        self._make_l1(queue, 'esc-l1-1', 'task-1')
        self._make_l1(queue, 'esc-l1-2', 'task-2')
        l2 = self._make_l2_with_members(queue, 'esc-l2-1', ['esc-l1-1', 'esc-l1-2'])

        queue.resolve(l2.id, 'Resolved the root cause')

        r1 = queue.get('esc-l1-1')
        r2 = queue.get('esc-l1-2')
        assert r1 is not None
        assert r2 is not None
        assert r1.status == 'resolved', f'Expected m1 resolved, got {r1.status!r}'
        assert r2.status == 'resolved', f'Expected m2 resolved, got {r2.status!r}'
        assert r1.resolution == 'Resolved the root cause'
        assert r2.resolution == 'Resolved the root cause'

    def test_members_carry_cascade_resolved_by(self, tmp_path: Path):
        """(b) members carry resolved_by='l2-cascade:{l2_id}' for audit."""
        queue = EscalationQueue(tmp_path / 'esc')
        self._make_l1(queue, 'esc-l1-1')
        l2 = self._make_l2_with_members(queue, 'esc-l2-1', ['esc-l1-1'])

        queue.resolve(l2.id, 'Fixed upstream')

        m = queue.get('esc-l1-1')
        assert m is not None
        expected_by = f'l2-cascade:{l2.id}'
        assert m.resolved_by == expected_by, (
            f"Expected resolved_by={expected_by!r}, got {m.resolved_by!r}"
        )

    def test_dismiss_path_cascades_dismiss_to_members(self, tmp_path: Path):
        """(c) resolve(l2_id, dismiss=True) cascades dismiss to members (status='dismissed')."""
        queue = EscalationQueue(tmp_path / 'esc')
        self._make_l1(queue, 'esc-l1-1')
        self._make_l1(queue, 'esc-l1-2')
        l2 = self._make_l2_with_members(queue, 'esc-l2-1', ['esc-l1-1', 'esc-l1-2'])

        queue.resolve(l2.id, 'Dismissed: not actionable', dismiss=True)

        m1 = queue.get('esc-l1-1')
        m2 = queue.get('esc-l1-2')
        assert m1 is not None
        assert m2 is not None
        assert m1.status == 'dismissed', f'Expected m1 dismissed, got {m1.status!r}'
        assert m2.status == 'dismissed', f'Expected m2 dismissed, got {m2.status!r}'

    def test_cascade_is_idempotent_member_already_resolved(self, tmp_path: Path):
        """(d) Member already resolved keeps its prior resolved_by/resolution (idempotency)."""
        queue = EscalationQueue(tmp_path / 'esc')
        self._make_l1(queue, 'esc-l1-1')
        queue.resolve('esc-l1-1', 'Pre-resolved separately', resolved_by='steward')
        l2 = self._make_l2_with_members(queue, 'esc-l2-1', ['esc-l1-1'])

        # Cascade should hit the idempotency guard in resolve() for m1
        queue.resolve(l2.id, 'L2 resolution')

        m1_reloaded = queue.get('esc-l1-1')
        assert m1_reloaded is not None
        assert m1_reloaded.status == 'resolved'
        assert m1_reloaded.resolution == 'Pre-resolved separately', (
            'Prior resolution must be preserved (idempotency)'
        )
        assert m1_reloaded.resolved_by == 'steward', (
            'Prior resolved_by must be preserved (idempotency)'
        )

    def test_cascade_to_nonexistent_member_is_best_effort(self, tmp_path: Path):
        """(e) Cascade to a non-existent member id is best-effort — no raise."""
        queue = EscalationQueue(tmp_path / 'esc')
        # Only seed one of two referenced members
        self._make_l1(queue, 'esc-l1-1')
        l2 = self._make_l2_with_members(queue, 'esc-l2-1', ['esc-l1-1', 'esc-nonexistent'])

        # Must not raise
        result = queue.resolve(l2.id, 'Fixed')

        assert result is not None
        assert result.status == 'resolved'
        # The existing member should still be resolved
        m1 = queue.get('esc-l1-1')
        assert m1 is not None
        assert m1.status == 'resolved'

    def test_l2_with_empty_members_resolves_normally_no_cascade(self, tmp_path: Path):
        """(f) L2 with empty members resolves normally (no cascade, no error)."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = self._make_l2_with_members(queue, 'esc-l2-1', [])

        result = queue.resolve(l2.id, 'Fixed; no members')

        assert result is not None
        assert result.status == 'resolved'
        assert result.resolution == 'Fixed; no members'


# ---------------------------------------------------------------------------
# Step-1: Unit tests for the escalation_id_lock exported helper
# ---------------------------------------------------------------------------

class TestEscalationIdLock:
    """Unit tests for the escalation_id_lock(queue_dir, escalation_id) context manager."""

    def test_importable_from_escalation_queue(self):
        """(a) escalation_id_lock is importable from escalation.queue at module level."""
        from escalation.queue import escalation_id_lock  # noqa: F401

    def test_entering_context_creates_sidecar_file(self, tmp_path: Path):
        """(b) Entering the context creates the sidecar .lock file at the expected path."""
        from escalation.queue import escalation_id_lock

        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        lock_path = queue_dir / 'esc-1-1.json.lock'

        assert not lock_path.exists(), 'Sidecar must not exist before context entry'
        with escalation_id_lock(queue_dir, 'esc-1-1'):
            assert lock_path.exists(), 'Sidecar must exist while context is held'
        # File persists after release (stable inode — never deleted)
        assert lock_path.exists(), 'Sidecar must persist after context exit (stable inode)'

    def test_held_lock_blocks_second_acquire_nonblocking(self, tmp_path: Path):
        """(c) While the context is held, a non-blocking flock on the same path raises BlockingIOError."""
        from escalation.queue import escalation_id_lock

        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()
        lock_path = queue_dir / 'esc-1-1.json.lock'

        with escalation_id_lock(queue_dir, 'esc-1-1'):
            # Open a second fd on the same sidecar file
            fd2 = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd2)

        # After context exits the lock is released — non-blocking acquire succeeds
        fd3 = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            # Must NOT raise — lock has been released
            fcntl.flock(fd3, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd3, fcntl.LOCK_UN)
        finally:
            os.close(fd3)

    def test_lock_file_invisible_to_esc_json_glob(self, tmp_path: Path):
        """(d) The .lock sidecar does NOT appear in Path.glob('esc-*.json')."""
        from escalation.queue import escalation_id_lock

        queue_dir = tmp_path / 'queue'
        queue_dir.mkdir()

        with escalation_id_lock(queue_dir, 'esc-1-1'):
            pass

        matched = list(queue_dir.glob('esc-*.json'))
        lock_path = queue_dir / 'esc-1-1.json.lock'
        assert lock_path not in matched, (
            f'Lock sidecar must not match esc-*.json glob; matched={matched}'
        )


# ---------------------------------------------------------------------------
# Step-3: Two-OS-process data-loss test for add_members_to_l2 (RED)
# ---------------------------------------------------------------------------

_CHILD_SCRIPT = Path(__file__).parent / '_concurrent_queue_child.py'


class TestAddMembersToL2Concurrency:
    """Two-OS-process test: concurrent add_members_to_l2 must not lose any members."""

    @pytest.mark.timeout(30)
    def test_concurrent_appends_preserve_all_members(self, tmp_path: Path):
        """N=50 appends from two concurrent processes must produce exactly 100 distinct members.

        Without the sidecar lock, each process reads the same pre-mutation snapshot,
        appends one element, and rewrites — the second write clobbers the first's addition.
        With the lock, appends serialize per-id and the union is complete.
        """
        # Set up the queue and seed the L2 escalation
        queue_dir = tmp_path / 'queue'
        queue = EscalationQueue(queue_dir)
        l2 = Escalation(
            id=queue.make_id('task-1'),
            task_id='task-1',
            agent_role='escalation-watcher-auto',
            severity='info',
            category='design_concern',
            summary='Concurrent member test',
            level=2,
            root_cause='Test root cause',
            members=[],
        )
        queue.submit(l2)
        l2_id = l2.id

        # Build env with worktree src on PYTHONPATH so child imports in-tree escalation
        env = os.environ.copy()
        src_path = str(Path(__file__).parent.parent / 'src')
        existing_pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f'{src_path}:{existing_pythonpath}' if existing_pythonpath else src_path

        count = 50
        child_args = [sys.executable, str(_CHILD_SCRIPT), str(queue_dir)]

        # Start both child processes concurrently
        proc_a = subprocess.Popen(
            child_args + ['add_members', l2_id, 'a', str(count)],
            env=env,
        )
        proc_b = subprocess.Popen(
            child_args + ['add_members', l2_id, 'b', str(count)],
            env=env,
        )

        # Wait for both to complete; kill any orphaned child on timeout/exception
        rc_a = rc_b = None
        try:
            rc_a = proc_a.wait(timeout=25)
            rc_b = proc_b.wait(timeout=25)
        finally:
            for proc in (proc_a, proc_b):
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        assert rc_a == 0, f'Child process A exited with rc={rc_a}'
        assert rc_b == 0, f'Child process B exited with rc={rc_b}'

        # Read the final state and assert all 100 members are present
        result = queue.get(l2_id)
        assert result is not None, 'L2 escalation disappeared after concurrent appends'
        members_set = set(result.members)
        expected = {f'a-{i}' for i in range(count)} | {f'b-{i}' for i in range(count)}
        missing = expected - members_set
        assert not missing, (
            f'Lost {len(missing)} member(s) to concurrent RMW clobber: '
            f'{sorted(missing)[:10]}...'
        )
        assert len(result.members) == count * 2, (
            f'Expected {count * 2} total members, got {len(result.members)}'
        )


# ---------------------------------------------------------------------------
# Step-5: Two-OS-process data-loss test for attach_dedupe_child (RED)
# ---------------------------------------------------------------------------

class TestAttachDedupeChildConcurrency:
    """Two-OS-process test: concurrent attach_dedupe_child must not lose any children."""

    @pytest.mark.timeout(30)
    def test_concurrent_attaches_preserve_all_children(self, tmp_path: Path):
        """N=50 attaches from two concurrent processes must produce exactly 100 children and dedupe_count==100.

        Without the sidecar lock, each process reads the same pre-mutation snapshot,
        appends one child and increments count by 1, then rewrites — the second write
        clobbers the first, losing children and reverting the count.
        """
        # Seed a pending parent escalation
        queue_dir = tmp_path / 'queue'
        queue = EscalationQueue(queue_dir)
        parent = Escalation(
            id=queue.make_id('task-1'),
            task_id='task-1',
            agent_role='orchestrator',
            severity='info',
            category='design_concern',
            summary='Concurrent dedupe test',
            level=0,
        )
        queue.submit(parent)
        parent_id = parent.id

        # Build env with worktree src on PYTHONPATH
        env = os.environ.copy()
        src_path = str(Path(__file__).parent.parent / 'src')
        existing_pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f'{src_path}:{existing_pythonpath}' if existing_pythonpath else src_path

        count = 50
        child_args = [sys.executable, str(_CHILD_SCRIPT), str(queue_dir)]

        # Start both child processes concurrently
        proc_a = subprocess.Popen(
            child_args + ['attach', parent_id, 'a', str(count)],
            env=env,
        )
        proc_b = subprocess.Popen(
            child_args + ['attach', parent_id, 'b', str(count)],
            env=env,
        )

        # Wait for both to complete; kill any orphaned child on timeout/exception
        rc_a = rc_b = None
        try:
            rc_a = proc_a.wait(timeout=25)
            rc_b = proc_b.wait(timeout=25)
        finally:
            for proc in (proc_a, proc_b):
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        assert rc_a == 0, f'Child process A exited with rc={rc_a}'
        assert rc_b == 0, f'Child process B exited with rc={rc_b}'

        # Read the final state and assert all 100 children are present
        result = queue.get(parent_id)
        assert result is not None, 'Parent escalation disappeared after concurrent attaches'
        children_set = set(result.dedupe_children)
        expected = {f'a-{i}' for i in range(count)} | {f'b-{i}' for i in range(count)}
        missing = expected - children_set
        assert not missing, (
            f'Lost {len(missing)} child(ren) to concurrent RMW clobber: '
            f'{sorted(missing)[:10]}...'
        )
        assert result.dedupe_count == count * 2, (
            f'Expected dedupe_count={count * 2}, got {result.dedupe_count}'
        )


# ---------------------------------------------------------------------------
# Concurrent resolve: exactly-one-archive-copy invariant
# ---------------------------------------------------------------------------

class TestConcurrentResolveExactlyOneArchive:
    """Two-OS-process test: two concurrent resolve() calls for the same id produce exactly one archive copy."""

    @pytest.mark.timeout(30)
    def test_concurrent_resolves_produce_exactly_one_archive(self, tmp_path: Path):
        """Two concurrent resolve() calls for the same pending escalation must leave exactly one archive file.

        Without the sidecar lock the idempotency check (status != 'pending') can race:
        both processes read 'pending', mutate, write, and archive — producing two archive
        copies.  With the lock the second resolve() serializes after the first, sees
        status != 'pending', and returns early without archiving again.
        """
        queue_dir = tmp_path / 'queue'
        queue = EscalationQueue(queue_dir)
        esc = Escalation(
            id=queue.make_id('task-5'),
            task_id='task-5',
            agent_role='orchestrator',
            severity='blocking',
            category='task_failure',
            summary='Concurrent resolve test',
            level=0,
        )
        queue.submit(esc)
        esc_id = esc.id

        env = os.environ.copy()
        src_path = str(Path(__file__).parent.parent / 'src')
        existing_pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f'{src_path}:{existing_pythonpath}' if existing_pythonpath else src_path

        child_args = [sys.executable, str(_CHILD_SCRIPT), str(queue_dir)]

        proc_a = subprocess.Popen(child_args + ['resolve', esc_id, 'a', '1'], env=env)
        proc_b = subprocess.Popen(child_args + ['resolve', esc_id, 'b', '1'], env=env)

        rc_a = rc_b = None
        try:
            rc_a = proc_a.wait(timeout=25)
            rc_b = proc_b.wait(timeout=25)
        finally:
            for proc in (proc_a, proc_b):
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        assert rc_a == 0, f'Child process A exited with rc={rc_a}'
        assert rc_b == 0, f'Child process B exited with rc={rc_b}'

        # Queue root file must be absent — archived by the winning resolve()
        assert not (queue_dir / f'{esc_id}.json').exists(), (
            f'Queue root file {esc_id}.json was not archived after concurrent resolves'
        )

        # Exactly one archive copy must exist — the losing resolve() is a no-op
        archive_files = list((queue_dir / 'archive').rglob(f'{esc_id}.json'))
        assert len(archive_files) == 1, (
            f'Expected exactly 1 archive copy, found {len(archive_files)}: {archive_files}'
        )


# ---------------------------------------------------------------------------
# Step-7: Deterministic spy test proving submit/submit_resolved/resolve adopt lock (RED)
# ---------------------------------------------------------------------------

class TestSubmitResolveAdoptLock:
    """Spy test: submit, submit_resolved, and resolve must call escalation_id_lock with correct id."""

    def _make_pending_escalation(self, esc_id: str, task_id: str = '1') -> Escalation:
        return Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='task_failure',
            summary='Lock adoption test',
            level=0,
        )

    def test_submit_acquires_lock_for_escalation_id(self, tmp_path: Path):
        """(a) queue.submit(esc) records a lock acquisition for esc.id."""
        import escalation.queue as queue_mod
        from escalation.queue import escalation_id_lock as real_lock

        acquired: list[tuple[Path, str]] = []

        @contextlib.contextmanager
        def recording_lock(queue_dir: Path, escalation_id: str):
            acquired.append((queue_dir, escalation_id))
            with real_lock(queue_dir, escalation_id):
                yield

        queue = EscalationQueue(tmp_path / 'queue')
        esc = self._make_pending_escalation('esc-1-1')

        with patch.object(queue_mod, 'escalation_id_lock', recording_lock):
            queue.submit(esc)

        assert any(eid == 'esc-1-1' for _, eid in acquired), (
            f'Expected lock acquisition for esc-1-1; got {acquired}'
        )

    def test_submit_resolved_acquires_lock_for_escalation_id(self, tmp_path: Path):
        """(b) queue.submit_resolved(esc, ...) records a lock acquisition for esc.id."""
        import escalation.queue as queue_mod
        from escalation.queue import escalation_id_lock as real_lock

        acquired: list[tuple[Path, str]] = []

        @contextlib.contextmanager
        def recording_lock(queue_dir: Path, escalation_id: str):
            acquired.append((queue_dir, escalation_id))
            with real_lock(queue_dir, escalation_id):
                yield

        queue = EscalationQueue(tmp_path / 'queue')
        esc = self._make_pending_escalation('esc-2-1', task_id='2')

        with patch.object(queue_mod, 'escalation_id_lock', recording_lock):
            queue.submit_resolved(esc, resolution='Auto-resolved at submit')

        assert any(eid == 'esc-2-1' for _, eid in acquired), (
            f'Expected lock acquisition for esc-2-1; got {acquired}'
        )

    def test_resolve_acquires_lock_for_escalation_id(self, tmp_path: Path):
        """(c) queue.resolve('esc-3-1', ...) on a pending escalation records a lock for 'esc-3-1'."""
        import escalation.queue as queue_mod
        from escalation.queue import escalation_id_lock as real_lock

        queue = EscalationQueue(tmp_path / 'queue')
        esc = self._make_pending_escalation('esc-3-1', task_id='3')
        queue.submit(esc)  # seed without spy

        acquired: list[tuple[Path, str]] = []

        @contextlib.contextmanager
        def recording_lock(queue_dir: Path, escalation_id: str):
            acquired.append((queue_dir, escalation_id))
            with real_lock(queue_dir, escalation_id):
                yield

        with patch.object(queue_mod, 'escalation_id_lock', recording_lock):
            queue.resolve('esc-3-1', resolution='Fixed')

        assert any(eid == 'esc-3-1' for _, eid in acquired), (
            f'Expected lock acquisition for esc-3-1; got {acquired}'
        )
