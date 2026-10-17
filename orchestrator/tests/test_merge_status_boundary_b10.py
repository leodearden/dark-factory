"""B10 of ``plans/merge-status-durable-non-landed-prd.md``'s Boundary-test sketch: cross-run lookup cost.

The sibling of ``escalation/tests/test_merge_status_boundary.py`` (rows B1-B9).
The bound, p99 < 50 ms on a store of at least 300k events and 2,500
merge_finalized rows, is kept as written by Leo's ruling on esc-4834-2 and is
met by task 6927's partial indexes.  It is WALL time because INV-8
(``docs/legibility/design-invariants.md``) bounds event-loop wall time, and
merge_status's cross-run reads run synchronously inside its async resolver.
Git is deliberately not touched: B10 is scoped to the cross-run read.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from orchestrator.event_store import EventStore, EventType

pytestmark = pytest.mark.xdist_group('merge_status_boundary_b10')

_BOUND_MS = 50.0
_SAMPLES = 200
_ABSENT = 'absent-from-every-run'


_EVENTS = 300_000
_RUNS = 228
_MERGE_FINALIZED_EVERY = 115
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
    """A restarted run's view of the seeded store, and the ids its hit shapes look up."""

    store: EventStore
    events: int
    merge_finalized: int
    oldest_request_id: str
    absorbed_loser_id: str
    train_id: str


def _task_id(n: int) -> str:
    return str(1000 + n % 4000)


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
    return _SeededStore(
        store=EventStore(db, 'run-current'),
        events=events,
        merge_finalized=merge_finalized,
        oldest_request_id=_request_id(0),
        absorbed_loser_id=_loser_id(_ABSORBING_EVERY - 1),
        train_id=_train_id(_TRAIN_MEMBER_EVERY - 1),
    )


def _durable_history_reads(store: EventStore, key: str, value: str) -> object:
    """The cross-run reads escalation/src/escalation/server.py::merge_status's ``_durable_history`` issues for one key."""
    if key != 'request_id':
        return store.latest_merge_finalized(**{key: value}, cross_run=True)
    return (
        store.latest_merge_finalized(request_id=value, cross_run=True)
        or store.merge_finalized_absorbing(value, cross_run=True)
        or store.merge_finalized_superseded_by(value, cross_run=True)
    )


@dataclass(frozen=True)
class _Shape:
    key: str
    value: Callable[[_SeededStore], str]
    hit: bool


_SHAPES = {
    'request_id_miss': _Shape('request_id', lambda seeded: _ABSENT, hit=False),
    'branch_miss': _Shape('branch', lambda seeded: _ABSENT, hit=False),
    'task_id_miss': _Shape('task_id', lambda seeded: _ABSENT, hit=False),
    'oldest_row_hit': _Shape('request_id', lambda seeded: seeded.oldest_request_id, hit=True),
    'absorbed_loser_hit': _Shape('request_id', lambda seeded: seeded.absorbed_loser_id, hit=True),
    'train_id_hit': _Shape('request_id', lambda seeded: seeded.train_id, hit=True),
}


def _nearest_rank(ordered: list[float], percentile: float) -> float:
    return ordered[math.ceil(percentile * len(ordered)) - 1]


@pytest.mark.parametrize('shape', sorted(_SHAPES))
def test_b10_cross_run_lookup_p99_is_under_50_ms(
    production_sized_store: _SeededStore, shape: str,
) -> None:
    spec = _SHAPES[shape]
    seeded = production_sized_store
    value = spec.value(seeded)
    assert bool(_durable_history_reads(seeded.store, spec.key, value)) is spec.hit, shape

    samples: list[float] = []
    for _ in range(_SAMPLES):
        start = time.perf_counter()
        result = _durable_history_reads(seeded.store, spec.key, value)
        samples.append((time.perf_counter() - start) * 1000.0)
        assert bool(result) is spec.hit, (shape, result)

    ordered = sorted(samples)
    p99 = _nearest_rank(ordered, 0.99)
    assert p99 < _BOUND_MS, (
        f'{shape}: p50={_nearest_rank(ordered, 0.50):.2f} ms p99={p99:.2f} ms '
        f'max={ordered[-1]:.2f} ms over {len(ordered)} reads on a store of '
        f'{seeded.events} events / {seeded.merge_finalized} merge_finalized rows'
    )
