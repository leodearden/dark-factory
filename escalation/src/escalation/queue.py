"""Filesystem-based escalation queue with atomic writes."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

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
        """Read a single escalation by ID."""
        path = self.queue_dir / f'{escalation_id}.json'
        if not path.exists():
            return None
        try:
            return Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
            return None

    def get_by_task(
        self, task_id: str, status: str | None = None, level: int | None = None,
    ) -> list[Escalation]:
        """Scan dir for escalations matching a task ID."""
        results = []
        for path in self.queue_dir.glob('esc-*.json'):
            try:
                esc = Escalation.from_json(path.read_text())
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
        """Update an escalation's status to resolved or dismissed."""
        esc = self.get(escalation_id)
        if esc is None:
            return None

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

        Returns the number of escalations dismissed.
        Already-resolved or already-dismissed escalations are not modified.
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

        Scans existing files to avoid overwriting resolved escalations
        from prior process runs (the in-memory counter resets on restart).
        """
        prefix = f'esc-{task_id}-'
        max_seq = 0
        for path in self.queue_dir.glob(f'{prefix}*.json'):
            suffix = path.stem[len(prefix):]
            with contextlib.suppress(ValueError):
                max_seq = max(max_seq, int(suffix))
        seq = max(max_seq + 1, self._next_seq())
        return f'{prefix}{seq}'
