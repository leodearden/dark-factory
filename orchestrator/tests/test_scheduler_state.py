"""Tests for scheduler state snapshot, new event types, and related machinery.

Split from test_scheduler.py to keep that file's 4000+ lines from growing
further and to group all state-snapshot / reserve-now / park-tracking tests
together for future maintenance.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.scheduler import ModuleLockTable, Scheduler


# ---------------------------------------------------------------------------
# Reuse the _RecordingEventStore test double from test_scheduler.py
# (duplicated here to avoid cross-module import; small enough to justify).
# ---------------------------------------------------------------------------

class _RecordingEventStore:
    """Minimal EventStore stand-in capturing emit() calls in-memory."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type, *, task_id=None, phase=None, role=None,
             data=None, cost_usd=None, duration_ms=None):
        self.events.append((
            str(event_type),
            {
                'task_id': task_id,
                'data': dict(data or {}),
            },
        ))


def _pending_task(task_id: str, *, priority: str = 'medium',
                  files: list[str] | None = None,
                  deps: list[str] | None = None) -> dict:
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'dependencies': deps or [],
        'metadata': {'files': files or [f'{task_id}/src']},
        'priority': priority,
    }


# ===========================================================================
# Step-1: EventType new members
# ===========================================================================

class TestEventTypeAdditions:
    """Verify the two new EventType enum members added in step-2."""

    def test_reserve_now_armed_exists(self):
        assert EventType.reserve_now_armed == 'reserve_now_armed'

    def test_reserve_now_consumed_exists(self):
        assert EventType.reserve_now_consumed == 'reserve_now_consumed'
