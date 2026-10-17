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

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from orchestrator.event_store import EventStore

pytestmark = pytest.mark.xdist_group('merge_status_boundary_b10')

_BOUND_MS = 50.0
_SAMPLES = 200
_ABSENT = 'absent-from-every-run'


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
