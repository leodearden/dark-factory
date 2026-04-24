"""Filesystem-based escalation queue with atomic writes."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from escalation import archive
from escalation.models import Escalation

logger = logging.getLogger(__name__)


class EscalationQueue:
    """Filesystem queue for escalations.

    Each escalation is a JSON file named {id}.json in the queue directory.
    Writes are atomic (tmp file + rename). Reads tolerate partial writes.
    """

    def __init__(self, queue_dir: Path):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._notify_callback: Callable[[Escalation], None] | None = None
        self._resolve_callback: Callable[[Escalation], None] | None = None

    def set_notify_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._notify_callback = callback

    def set_resolve_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._resolve_callback = callback

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _iter_archive_paths(self, pattern: str) -> Iterator[Path]:
        """Yield escalation JSON files from the archive subtree matching *pattern*.

        Returns an empty iterator when the archive root does not exist, avoiding
        a spurious ``rglob`` call on a missing directory.  Centralising the
        existence-check + rglob here means get(), get_by_task(), and make_id()
        all benefit from any future caching or indexing added in one place.
        """
        archive_root = self.queue_dir / archive.ARCHIVE_SUBDIR
        if archive_root.exists():
            yield from archive_root.rglob(pattern)

    def submit(self, escalation: Escalation) -> str:
        """Atomic write: {id}.tmp -> rename to {id}.json."""
        path = self.queue_dir / f'{escalation.id}.json'

        # Write to tmp file first
        fd, tmp_path = tempfile.mkstemp(
            suffix='.tmp', prefix=escalation.id, dir=str(self.queue_dir)
        )
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(escalation.to_json())
            os.rename(tmp_path, str(path))
        except Exception:
            # Clean up tmp on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        logger.info(f'Escalation submitted: {escalation.id} [{escalation.severity}]')

        if self._notify_callback:
            try:
                self._notify_callback(escalation)
            except Exception as e:
                logger.warning(f'Notify callback failed for {escalation.id}: {e}')

        return escalation.id

    def get(self, escalation_id: str) -> Escalation | None:
        """Read a single escalation by ID.

        Falls back to the archive directory when the file is not in the
        queue root (i.e. the escalation has been resolved and archived).

        Note: the archive fallback performs an O(|archive|) rglob on every miss.
        For a 30-day retention window this is bounded, but callers on hot paths
        should avoid repeated get() calls for ids known to be archived.
        TODO: memoise the archive listing per dated subdir to reduce repeated scans.
        """
        path = self.queue_dir / f'{escalation_id}.json'
        if not path.exists():
            # Fall back to archive: search all dated subdirs.
            candidates = list(self._iter_archive_paths(f'{escalation_id}.json'))
            if not candidates:
                return None
            if len(candidates) > 1:
                logger.warning(
                    f'Multiple archive files for {escalation_id}: '
                    f'{[str(p) for p in candidates]}; selecting newest by parent dir date'
                )
                # YYYY-MM-DD sorts lexicographically == chronologically.
                # Non-YYYY-MM-DD parent names fall back to '' (treated as oldest),
                # matching the comment and ensuring valid date dirs always win.
                path = max(
                    candidates,
                    key=lambda p: (
                        p.parent.name
                        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', p.parent.name)
                        else ''
                    ),
                )
            else:
                path = candidates[0]
        try:
            return Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
            return None

    def get_by_task(
        self, task_id: str, status: str | None = None, level: int | None = None,
    ) -> list[Escalation]:
        """Scan dir for escalations matching a task ID.

        Two-tier scan:
        - status == 'pending': scan queue root only (fast path, skips archive).
        - status is None or another value: scan queue root PLUS archive/**/ to
          include resolved/dismissed escalations that have been moved out.

        Deduplication: if the same escalation id appears in both the queue root
        and the archive (e.g. crash mid-resolve), only the first occurrence is
        returned and a WARNING is logged.  Iteration order is queue root first,
        archive second, so the queue_dir copy wins when both exist.
        """
        # Build the candidate path list.
        paths: list[Path] = list(self.queue_dir.glob('esc-*.json'))
        if status != 'pending':
            paths.extend(self._iter_archive_paths('esc-*.json'))

        seen: set[str] = set()
        results = []
        for path in paths:
            try:
                esc = Escalation.from_json(path.read_text())
                if esc.id in seen:
                    logger.warning(
                        f'Duplicate escalation id {esc.id!r} at {path}; '
                        'skipping (queue_dir copy takes precedence)'
                    )
                    continue
                seen.add(esc.id)
                if esc.task_id != task_id:
                    continue
                if status is not None and esc.status != status:
                    continue
                if level is not None and esc.level != level:
                    continue
                results.append(esc)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return results

    def has_open_l1(self, task_id: str) -> bool:
        """Return True when the task has at least one pending level-1 escalation.

        Level-1 is the handed-to-human tier: the presence of one signals that
        the workflow must not auto-requeue the task — a human will unblock.
        """
        return bool(self.get_by_task(task_id, status='pending', level=1))

    def get_pending(self) -> list[Escalation]:
        """Get all pending escalations."""
        results = []
        for path in self.queue_dir.glob('esc-*.json'):
            try:
                esc = Escalation.from_json(path.read_text())
                if esc.status == 'pending':
                    results.append(esc)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return results

    def resolve(
        self, escalation_id: str, resolution: str, dismiss: bool = False,
        *, resolved_by: str | None = None, resolution_turns: int | None = None,
    ) -> Escalation | None:
        """Update an escalation's status to resolved or dismissed.

        Idempotent: if the escalation is already resolved or dismissed, this
        method returns the existing escalation unchanged without re-archiving
        or re-firing the _resolve_callback.
        """
        esc = self.get(escalation_id)
        if esc is None:
            return None

        if esc.status != 'pending':
            logger.info(
                f'Escalation {escalation_id} already {esc.status}; resolve() is a no-op'
            )
            return esc

        esc.status = 'dismissed' if dismiss else 'resolved'
        esc.resolution = resolution
        esc.resolved_at = datetime.now(UTC).isoformat()
        if resolved_by is not None:
            esc.resolved_by = resolved_by
        if resolution_turns is not None:
            esc.resolution_turns = resolution_turns

        path = self.queue_dir / f'{escalation_id}.json'
        fd, tmp_path = tempfile.mkstemp(
            suffix='.tmp', prefix=escalation_id, dir=str(self.queue_dir)
        )
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(esc.to_json())
            os.rename(tmp_path, str(path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        # Best-effort: move resolved file into dated archive subdir.
        #
        # Two-step design (write-to-root then os.replace to archive) is
        # intentional: if the archive move fails, the *resolved* file remains in
        # queue_dir so get() can still return it — no data is lost.  Writing
        # directly to the archive dir would leave the file as *pending* in
        # queue_dir on failure, which is a worse fallback state.
        #
        # Failure logs a warning but does not abort the resolution.
        try:
            archive_dir = archive.archive_dir_for_date(self.queue_dir, esc.resolved_at)
            archive_dir.mkdir(parents=True, exist_ok=True)
            os.replace(str(path), str(archive_dir / f'{escalation_id}.json'))
        except OSError as exc:
            logger.warning(
                f'Failed to archive escalation {escalation_id}: {exc}; '
                'file remains in queue_dir'
            )

        logger.info(f'Escalation {escalation_id} {esc.status}: {resolution[:100]}')

        if self._resolve_callback:
            try:
                self._resolve_callback(esc)
            except Exception as e:
                logger.warning(f'Resolve callback failed for {escalation_id}: {e}')

        return esc

    def _rewrite(self, escalation_id: str, escalation: Escalation) -> None:
        """Atomically rewrite an escalation's JSON file."""
        path = self.queue_dir / f'{escalation_id}.json'
        fd, tmp_path = tempfile.mkstemp(
            suffix='.tmp', prefix=escalation_id, dir=str(self.queue_dir)
        )
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(escalation.to_json())
            os.rename(tmp_path, str(path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def dismiss_all_pending(self, resolution: str) -> int:
        """Dismiss all pending escalations with the given resolution message.

        Returns the number of escalations where resolve() returned non-None.
        In the common single-writer case this equals the number actually dismissed.
        In a concurrent-write race (another process resolves an escalation between
        get_pending() and resolve()), resolve() is a no-op but still returns the
        existing escalation, so the count may include those no-ops.  The counter
        is best read as "attempted dismissals, including no-ops for concurrent
        resolutions".  This function is single-writer in practice.
        """
        pending = self.get_pending()
        count = 0
        for esc in pending:
            try:
                if self.resolve(esc.id, resolution, dismiss=True, resolved_by='auto-dismissed') is not None:
                    count += 1
            except Exception as e:
                logger.warning(f'Failed to dismiss escalation {esc.id}: {e}')
        if count:
            logger.info(f'Dismissed {count} stale escalation(s): {resolution[:100]}')
        return count

    def make_id(self, task_id: str) -> str:
        """Generate a unique escalation ID.

        Scans existing files in both the queue root and the archive
        subdirectory to avoid reusing sequence numbers from escalations
        that have been archived after resolution (the in-memory counter
        resets on process restart, so we must re-derive max_seq from disk).

        Note: the archive scan is O(|archive files for task_id|) on every call.
        make_id() is a slow path (invoked only at submission), so this is
        acceptable within a 30-day retention window.
        TODO: cache max_seq per task_id to eliminate repeated archive scans.
        """
        prefix = f'esc-{task_id}-'
        max_seq = 0
        for path in self.queue_dir.glob(f'{prefix}*.json'):
            suffix = path.stem[len(prefix):]
            with contextlib.suppress(ValueError):
                max_seq = max(max_seq, int(suffix))
        # Also scan the archive so post-restart calls don't return IDs
        # already used by archived escalations for the same task.
        for path in self._iter_archive_paths(f'{prefix}*.json'):
            suffix = path.stem[len(prefix):]
            with contextlib.suppress(ValueError):
                max_seq = max(max_seq, int(suffix))
        seq = max(max_seq + 1, self._next_seq())
        return f'{prefix}{seq}'
