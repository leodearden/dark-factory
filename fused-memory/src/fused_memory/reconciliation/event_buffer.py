"""Event accumulation buffer with trigger logic for reconciliation cycles."""

import asyncio
import logging
from datetime import UTC, datetime

from fused_memory.models.reconciliation import ReconciliationEvent

logger = logging.getLogger(__name__)


class EventBuffer:
    """Buffers reconciliation events per project. Checks trigger conditions."""

    def __init__(self, buffer_size_threshold: int = 10, max_staleness_seconds: int = 1800):
        self.buffer_size_threshold = buffer_size_threshold
        self.max_staleness_seconds = max_staleness_seconds
        self._buffer: dict[str, list[ReconciliationEvent]] = {}
        self._lock = asyncio.Lock()
        self._active_runs: set[str] = set()

    async def push(self, event: ReconciliationEvent) -> None:
        """Add event to project buffer."""
        async with self._lock:
            if event.project_id not in self._buffer:
                self._buffer[event.project_id] = []
            self._buffer[event.project_id].append(event)
        logger.info(
            'reconciliation.event_buffered',
            extra={
                'project_id': event.project_id,
                'event_type': event.type.value,
                'buffer_size': len(self._buffer.get(event.project_id, [])),
            },
        )

    async def should_trigger(self, project_id: str) -> tuple[bool, str]:
        """Check if buffer crosses threshold or max staleness.

        Returns (should_trigger, reason).
        """
        async with self._lock:
            events = self._buffer.get(project_id, [])
            if not events:
                return False, ''
            if project_id in self._active_runs:
                return False, ''
            if len(events) >= self.buffer_size_threshold:
                return True, f'buffer_size:{len(events)}'
            oldest = min(e.timestamp for e in events)
            now = datetime.now(UTC)
            oldest_aware = oldest if oldest.tzinfo else oldest.replace(tzinfo=UTC)
            age = (now - oldest_aware).total_seconds()
            if age > self.max_staleness_seconds:
                return True, f'max_staleness:{oldest.isoformat()}'
            return False, ''

    async def drain(self, project_id: str) -> list[ReconciliationEvent]:
        """Atomically drain buffer for a project, returning events."""
        async with self._lock:
            events = self._buffer.pop(project_id, [])
            return events

    async def mark_run_active(self, project_id: str) -> bool:
        """Mark a run as active. Returns False if already active."""
        async with self._lock:
            if project_id in self._active_runs:
                return False
            self._active_runs.add(project_id)
            return True

    async def mark_run_complete(self, project_id: str) -> None:
        """Mark run complete, allowing new triggers."""
        async with self._lock:
            self._active_runs.discard(project_id)

    def get_buffer_stats(self, project_id: str) -> dict:
        """For dashboard: buffer size, oldest event age."""
        events = self._buffer.get(project_id, [])
        if not events:
            return {'size': 0, 'oldest_event_age_seconds': None}
        oldest = min(e.timestamp for e in events)
        now = datetime.now(UTC)
        oldest_aware = oldest if oldest.tzinfo else oldest.replace(tzinfo=UTC)
        age = (now - oldest_aware).total_seconds()
        return {'size': len(events), 'oldest_event_age_seconds': round(age, 1)}

    def get_active_projects(self) -> list[str]:
        """Return project IDs that have buffered events."""
        return list(self._buffer.keys())
