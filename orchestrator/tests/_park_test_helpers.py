"""Shared scaffolding for the scheduler fairness-park tests (task 5308).

Lives in ``_park_test_helpers.py`` (not ``conftest.py``) to follow the
``_recording_event_store.py`` pattern: importable from any test file without
triggering ``sys.modules['conftest']`` collisions across subprojects.

Events are matched by the last dotted segment of their recorded type, so a
name like ``'reservation_installed'`` matches however ``str(EventType.X)``
renders.  That rule lives only in :func:`event_matches`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from _recording_event_store import _RecordingEventStore

from orchestrator.config import OrchestratorConfig
from orchestrator.overrides import OverrideStore
from orchestrator.park_eviction_requests import ParkEvictionRequestStore
from orchestrator.scheduler import Scheduler


class ParkWorld(NamedTuple):
    scheduler: Scheduler
    store: _RecordingEventStore
    overrides: OverrideStore
    root: str


def park_world(
    tmp_path: Path,
    *,
    pinned: tuple[str, ...] = (),
    time_source: Callable[[], float] | None = None,
    park_eviction_store: ParkEvictionRequestStore | None = None,
    state_snapshot_path: Path | None = None,
    **config_overrides: Any,
) -> ParkWorld:
    """A started Scheduler whose top parks on its first skip.

    One holder per module, ``lock_depth=2`` and ``project_root=tmp_path``,
    with *config_overrides* applied on top.  Each id in *pinned* is pinned in
    the given order, so its pin_order is its 1-based position.  *time_source*
    is the Scheduler's monotonic clock (the real one when None);
    *park_eviction_store* and *state_snapshot_path* go to the Scheduler
    constructor unchanged.
    """
    config = OrchestratorConfig(**{
        'max_per_module': 1,
        'lock_depth': 2,
        'project_root': tmp_path,
        **config_overrides,
    })
    config.fairness.skip_threshold = 1
    root = str(config.project_root)
    overrides = OverrideStore(tmp_path / 'o.db')
    for tid in pinned:
        overrides.set_override(root, tid, pinned=True)
    store = _RecordingEventStore()
    scheduler = Scheduler(
        config,
        event_store=store,  # type: ignore[arg-type]
        override_store=overrides,
        time_source=time_source,
        park_eviction_store=park_eviction_store,
        state_snapshot_path=state_snapshot_path,
    )
    scheduler.finish_startup()
    return ParkWorld(scheduler, store, overrides, root)


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
