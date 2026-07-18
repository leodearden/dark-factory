"""Shared minimal EventStore test double for orchestrator tests.

Lives in _recording_event_store.py (not conftest.py) to follow the
`_orch_helpers.py` pattern: importable from any test file without
triggering sys.modules['conftest'] collisions across subprojects.
"""
from __future__ import annotations


class _RecordingEventStore:
    """Minimal EventStore stand-in capturing emit() calls in-memory."""

    def __init__(self, run_id: str = 'run-test') -> None:
        self.events: list[tuple[str, dict]] = []
        # Several dispatch sites read ``event_store.run_id`` unconditionally
        # when threading cost telemetry into ``invoke_with_cap_retry`` (e.g.
        # steward.py / dry_run_unblock.py); expose it so this store can stand
        # in for those sites, not just workflow._invoke (which gates the read
        # behind ``if self.cost_store``).
        self.run_id = run_id

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
