"""Filesystem-based escalation queue with atomic writes."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
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

    def set_notify_callback(self, callback: Callable[[Escalation], None]) -> None:
        self._notify_callback = callback

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
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f'Failed to parse escalation {escalation_id}: {e}')
            return None

    def get_by_task(self, task_id: str, status: str | None = None) -> list[Escalation]:
        """Scan dir for escalations matching a task ID."""
        results = []
        for path in self.queue_dir.glob('esc-*.json'):
            try:
                esc = Escalation.from_json(path.read_text())
                if esc.task_id == task_id and (status is None or esc.status == status):
                    results.append(esc)
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    def get_pending(self) -> list[Escalation]:
        """Get all pending escalations."""
        results = []
        for path in self.queue_dir.glob('esc-*.json'):
            try:
                esc = Escalation.from_json(path.read_text())
                if esc.status == 'pending':
                    results.append(esc)
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    def resolve(
        self, escalation_id: str, resolution: str, dismiss: bool = False
    ) -> Escalation | None:
        """Update an escalation's status to resolved or dismissed."""
        esc = self.get(escalation_id)
        if esc is None:
            return None

        esc.status = 'dismissed' if dismiss else 'resolved'
        esc.resolution = resolution

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
        return esc

    def dismiss_all_pending(self, resolution: str) -> int:
        """Dismiss all pending escalations with the given resolution message.

        Returns the number of escalations dismissed.
        Already-resolved or already-dismissed escalations are not modified.
        """
        pending = self.get_pending()
        count = 0
        for esc in pending:
            if self.resolve(esc.id, resolution, dismiss=True) is not None:
                count += 1
        if count:
            logger.info(f'Dismissed {count} stale escalation(s): {resolution[:100]}')
        return count

    def make_id(self, task_id: str) -> str:
        """Generate a unique escalation ID."""
        return f'esc-{task_id}-{self._next_seq()}'
