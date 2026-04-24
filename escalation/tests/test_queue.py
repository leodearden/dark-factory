"""Tests for EscalationQueue — focusing on dismiss_all_pending()."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def _make_escalation(esc_id: str, task_id: str = '1', status: str = 'pending') -> Escalation:
    esc = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='task_failure',
        summary='Something failed',
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

