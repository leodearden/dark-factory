"""One owner-state predicate decides whether an owner may hold parks (task 6518).

esc-6040-7 ruled that a park, pin reservation or fairness, lives only while
its owner is PENDING and no dependency of it (local, cross-project, delivered
check) is known to be unsatisfied.  An unknown dependency keeps the park:
expiry needs affirmative evidence.

Every module is a single-segment ``.py`` file, so its lock key is its own name.
Holders take their locks through ``lock_table.try_acquire`` and appear in the
task list as ``in-progress``: they own no parks and are never candidates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import (
    event_data_for,
    event_payloads,
    make_task,
    make_withheld_task,
    park_world,
)


def _stack_owners(scheduler, module: str) -> list[str]:
    stacks = scheduler.lock_table.snapshot_park_stacks()
    return [entry['owner'] for entry in stacks.get(module, [])]


# ---------------------------------------------------------------------------
# D1 — only a pending owner may hold parks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['blocked', 'deferred', 'review', 'merge-deferred', 'in-progress'])
async def test_an_owner_that_leaves_pending_loses_its_parks_and_its_skip_count(tmp_path, status):
    w = park_world(tmp_path)
    assert w.scheduler.lock_table.try_acquire('H', ['m.py'])
    w.scheduler.lock_table.install_parks('L', ['m.py'], 'low')
    holder = make_task('H', 'low', ['m.py'], status='in-progress')
    shadowed = make_withheld_task('L', 'low', ['m.py'])
    w.scheduler.get_tasks = AsyncMock(
        return_value=[make_task('X', 'high', ['m.py']), shadowed, holder]
    )
    assert await w.scheduler.acquire_next() is None
    assert _stack_owners(w.scheduler, 'm.py') == ['L', 'X'], 'premise: X parks m above L'
    assert 'X' in w.scheduler.get_state_snapshot()['skip_counts'], 'premise: X was skipped'
    w.store.events.clear()

    w.scheduler.get_tasks = AsyncMock(
        return_value=[make_task('X', 'high', ['m.py'], status=status), shadowed, holder]
    )
    assert await w.scheduler.acquire_next() is None

    assert event_data_for(w.store, 'reservation_expired', 'X') == [
        {'reason': f'not_pending:{status}', 'source': 'fairness'},
    ]
    assert [e['data'] for e in event_payloads(w.store, 'reservation_restored')] == [
        {'restored_owner': 'L', 'modules': ['m.py'], 'source': 'fairness'},
    ]
    assert 'X' not in w.scheduler.lock_table.snapshot_parks()
    assert 'X' not in w.scheduler.get_state_snapshot()['skip_counts']
