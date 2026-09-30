"""Shared scaffolding for the scheduler fairness-park tests (task 5308).

Lives in ``_park_test_helpers.py`` (not ``conftest.py``) to follow the
``_recording_event_store.py`` pattern: importable from any test file without
triggering ``sys.modules['conftest']`` collisions across subprojects.

Events are matched by the last dotted segment of their recorded type, so a
name like ``'reservation_installed'`` matches however ``str(EventType.X)``
renders.  That rule lives only in :func:`event_matches`.
"""

from __future__ import annotations

from _recording_event_store import _RecordingEventStore


def make_task(tid: str, priority: str, files: list[str], *, status: str = 'pending') -> dict:
    return {
        'id': tid,
        'title': f'Task {tid}',
        'status': status,
        'priority': priority,
        'dependencies': [],
        'metadata': {'files': list(files)},
    }


def event_matches(event_type: str, name: str) -> bool:
    return event_type.split('.')[-1] == name


def event_payloads(store: _RecordingEventStore, name: str) -> list[dict]:
    """``{'task_id', 'data'}`` payloads of every recorded *name* event, in order."""
    return [payload for event_type, payload in store.events if event_matches(event_type, name)]


def event_data_for(store: _RecordingEventStore, name: str, task_id: str) -> list[dict]:
    """``data`` of every recorded *name* event about *task_id*, in order."""
    return [e['data'] for e in event_payloads(store, name) if e['task_id'] == task_id]


def event_index(store: _RecordingEventStore, name: str, task_id: str) -> int:
    """Position in the recording of the first *name* event about *task_id*."""
    for index, (event_type, payload) in enumerate(store.events):
        if event_matches(event_type, name) and payload['task_id'] == task_id:
            return index
    raise AssertionError(f'no {name} event for {task_id}')
