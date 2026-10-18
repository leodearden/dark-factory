"""B10 of ``plans/merge-status-durable-non-landed-prd.md``'s Boundary-test sketch: cross-run lookup cost.

The sibling of ``escalation/tests/test_merge_status_boundary.py`` (rows B1-B9).
The bound, p99 < 50 ms on a store of at least 300k events and 2,500
merge_finalized rows, is kept as written by Leo's ruling on esc-4834-2 and is
met by task 6927's partial indexes.  It is WALL time because INV-8
(``docs/legibility/design-invariants.md``) bounds event-loop wall time, and
merge_status's cross-run reads run synchronously inside its async resolver.
The timed reads are the ones the real merge_status issues for each poll,
recorded off the store and replayed.  Git is deliberately not touched: B10 is
scoped to the cross-run read.  Each window of replays is the ruled bound in
full; a window that misses it is re-measured after a pause, within a time
budget, because host scheduling stalls lift the tail of a ~2 ms read, and a
window whose median misses is not re-measured (task 7090).
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from _latency_windows import remeasure_until_p99_within, time_window
from escalation.queue import EscalationQueue
from escalation.server import create_server
from fastmcp.tools.function_tool import FunctionTool
from shared.merge_state import MergeState

from orchestrator.event_store import EventStore, EventType

pytestmark = pytest.mark.xdist_group('merge_status_boundary_b10')

_BOUND_MS = 50.0
_SAMPLES = 200
_REMEASURE_BUDGET_S = 90.0
_REMEASURE_PAUSE_S = 1.0
_ABSENT = 'absent-from-every-run'


_EVENTS = 300_000
_RUNS = 228
_TASKS = 1000
_MERGE_FINALIZED_EVERY = 115
_MERGE_FINALIZED_ROWS = len(range(0, _EVENTS, _MERGE_FINALIZED_EVERY))
_TRAIN_MEMBER_EVERY = 40
_ABSORBING_EVERY = 64
_NOISE_DATA = json.dumps({'phase': 'verify', 'detail': 'x' * 170})
_INSERT = (
    'INSERT INTO events '
    '(timestamp, run_id, task_id, event_type, phase, role, data, cost_usd, duration_ms) '
    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'
)


@dataclass(frozen=True)
class _SeededStore:
    """A restarted run's view of the seeded store, with the sizes B10's premise names."""

    store: EventStore
    events: int
    merge_finalized: int


def _task_id(n: int) -> str:
    return str(1000 + n % _TASKS)


def _request_id(n: int) -> str:
    return f'mr-{n:08x}'


def _loser_id(n: int) -> str:
    return f'mr-loser-{n:08x}'


def _train_id(n: int) -> str:
    return f'coalesce-{_task_id(n)}-{n:08x}'


def _is_train_member(n: int) -> bool:
    return n % _TRAIN_MEMBER_EVERY == _TRAIN_MEMBER_EVERY - 1


def _is_absorbing(n: int) -> bool:
    return n % _ABSORBING_EVERY == _ABSORBING_EVERY - 1


def _merge_finalized_data(n: int) -> dict[str, object]:
    """The n-th row's payload, in orchestrator/src/orchestrator/merge_lane/worker.py::enqueue_merge_request's key set."""
    member = _is_train_member(n)
    state = 'superseded' if member else ('done', 'blocked', 'conflict')[n % 3]
    return {
        'request_id': _request_id(n),
        'branch': _task_id(n),
        'state': state,
        'snapshot_tip': f'{n:040x}',
        'merge_sha': f'{n + 1:040x}' if state == 'done' else None,
        'superseded_by': _train_id(n) if member else None,
        'generation': 1,
        'reason': 'post-merge verify failed' if state == 'blocked' else None,
        'landed_via_chain': None,
        'absorbed_request_ids': [_loser_id(n)] if _is_absorbing(n) else [],
    }


def _rows(start: datetime) -> Iterator[tuple[object, ...]]:
    for i in range(_EVENTS):
        timestamp = (start + timedelta(seconds=i)).isoformat()
        run_id = f'run-{i * _RUNS // _EVENTS:04d}'
        if i % _MERGE_FINALIZED_EVERY:
            yield (
                timestamp, run_id, _task_id(i), EventType.phase_exit.value,
                'verify', 'implementer', _NOISE_DATA, None, None,
            )
            continue
        n = i // _MERGE_FINALIZED_EVERY
        yield (
            timestamp, run_id, _task_id(n), EventType.merge_finalized.value,
            'merge', None, json.dumps(_merge_finalized_data(n)), None, None,
        )


@pytest.fixture(scope='module')
def production_sized_store(tmp_path_factory: pytest.TempPathFactory) -> _SeededStore:
    db = tmp_path_factory.mktemp('merge_status_boundary_b10') / 'runs.db'
    EventStore(db, 'run-seed')
    conn = sqlite3.connect(db)
    try:
        with conn:
            conn.executemany(_INSERT, _rows(datetime(2026, 1, 1, tzinfo=UTC)))
        events, merge_finalized = conn.execute(
            'SELECT COUNT(*), SUM(event_type = ?) FROM events',
            (EventType.merge_finalized.value,),
        ).fetchone()
    finally:
        conn.close()
    assert events >= 300_000 and merge_finalized >= 2_500, (
        f'B10 needs >= 300,000 events and >= 2,500 merge_finalized rows; '
        f'seeded {events} and {merge_finalized}'
    )
    assert merge_finalized == _MERGE_FINALIZED_ROWS, (merge_finalized, _MERGE_FINALIZED_ROWS)
    return _SeededStore(
        store=EventStore(db, 'run-current'), events=events, merge_finalized=merge_finalized,
    )


@dataclass(frozen=True)
class _Read:
    """One EventStore call, replayable against the store it was recorded on."""

    method: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]

    def issue(self, store: EventStore) -> object:
        return getattr(store, self.method)(*self.args, **self.kwargs)


class _CrossRunReadRecorder:
    """Stands in for an EventStore: forwards every attribute, recording each cross_run=True call."""

    def __init__(self, store: EventStore) -> None:
        self._store = store
        self.reads: list[_Read] = []

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._store, name)
        if not callable(attribute):
            return attribute

        def forwarded(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get('cross_run') is True:
                self.reads.append(_Read(name, args, MappingProxyType(dict(kwargs))))
            return attribute(*args, **kwargs)

        return forwarded


async def _poll_recording_cross_run_reads(
    store: EventStore, escalations: Path, key: Mapping[str, str],
) -> tuple[dict[str, Any], tuple[_Read, ...]]:
    """The real merge_status's answer to *key* over *store*, and the cross-run reads it issued."""
    recorder = _CrossRunReadRecorder(store)
    server = create_server(
        EscalationQueue(escalations), event_store=recorder, startup_sweep=False,
    )
    tool = await server.get_tool('merge_status')
    assert isinstance(tool, FunctionTool), tool
    response = await tool.fn(**key)
    return response, tuple(recorder.reads)


def _newest_row_of_task(n: int) -> str:
    return _request_id(max(m for m in range(_MERGE_FINALIZED_ROWS) if _task_id(m) == _task_id(n)))


@dataclass(frozen=True)
class _Shape:
    """One poll, and the request id its last_outcome carries (None: the poll misses every run)."""

    key: str
    value: str
    last_outcome_request_id: str | None


_FIRST_ABSORBING = _ABSORBING_EVERY - 1
_FIRST_TRAIN_MEMBER = _TRAIN_MEMBER_EVERY - 1

_SHAPES: Mapping[str, _Shape] = MappingProxyType({
    'request_id_miss': _Shape('request_id', _ABSENT, None),
    'branch_miss': _Shape('branch', _ABSENT, None),
    'task_id_miss': _Shape('task_id', _ABSENT, None),
    'oldest_row_hit': _Shape('request_id', _request_id(0), _request_id(0)),
    'absorbed_loser_hit': _Shape(
        'request_id', _loser_id(_FIRST_ABSORBING), _request_id(_FIRST_ABSORBING),
    ),
    'train_id_hit': _Shape(
        'request_id', _train_id(_FIRST_TRAIN_MEMBER), _request_id(_FIRST_TRAIN_MEMBER),
    ),
    'branch_hit': _Shape('branch', _task_id(0), _newest_row_of_task(0)),
    'task_id_hit': _Shape('task_id', _task_id(1), _newest_row_of_task(1)),
})


def _replay(store: EventStore, reads: tuple[_Read, ...], shape: str, hit: bool) -> None:
    results = [read.issue(store) for read in reads]
    assert any(results) is hit, (shape, results)


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', sorted(_SHAPES))
async def test_b10_cross_run_lookup_p99_is_under_50_ms(
    production_sized_store: _SeededStore, shape: str, tmp_path: Path,
) -> None:
    spec = _SHAPES[shape]
    seeded = production_sized_store
    hit = spec.last_outcome_request_id is not None
    response, reads = await _poll_recording_cross_run_reads(
        seeded.store, tmp_path / 'escalations', {spec.key: spec.value},
    )
    assert reads, f'{shape}: merge_status issued no cross-run read: {response}'
    assert response['state'] == (MergeState.stale_record if hit else MergeState.no_record), (
        shape, response,
    )
    last = response['last_outcome']
    assert (None if last is None else last['request_id']) == spec.last_outcome_request_id, (
        shape, response,
    )

    remeasured = remeasure_until_p99_within(
        lambda: time_window(lambda: _replay(seeded.store, reads, shape, hit), _SAMPLES),
        bound_ms=_BOUND_MS, budget_s=_REMEASURE_BUDGET_S, pause_s=_REMEASURE_PAUSE_S,
    )
    assert remeasured.met, (
        f'{shape}: no window of {_SAMPLES} replays of {len(reads)} read(s) had p99 < {_BOUND_MS} ms '
        f'on a store of {seeded.events} events / {seeded.merge_finalized} merge_finalized rows; '
        f'windows (p50/p99/max ms): {remeasured.describe()}; load average {os.getloadavg()}'
    )
