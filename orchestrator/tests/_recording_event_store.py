"""Shared minimal EventStore test double for orchestrator tests.

Lives in _recording_event_store.py (not conftest.py) to follow the
`_orch_helpers.py` pattern: importable from any test file without
triggering sys.modules['conftest'] collisions across subprojects.
"""
from __future__ import annotations


class _RecordingEventStore:
    """Minimal EventStore stand-in capturing emit() calls in-memory."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(
        self,
        event_type,
        *,
        task_id=None,
        phase=None,
        role=None,
        data=None,
        cost_usd=None,
        duration_ms=None,
    ) -> None:
        self.events.append((
            str(event_type),
            {
                'task_id': task_id,
                'data': dict(data or {}),
            },
        ))
