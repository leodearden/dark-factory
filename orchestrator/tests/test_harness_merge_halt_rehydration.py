"""A restarted Harness re-asserts a preserved merge halt only while project_root still needs it (task 3876).

Each ``_restart`` is a fresh Harness holding a real merge worker over the same
durable escalation dir and the same real checkout, and keeping its real
startup step 1c0, so the predicate is measured as a redeployed orchestrator
measures it.

This file imports no ``orchestrator.merge_lane`` or ``orchestrator.git_ops``
module: it reads harness privates, which the merge-lane ratchet
(``scripts/merge_lane_metrics.py``) would count against any lane-importing file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _checkout_hygiene_fixtures import dirty_tracked_file, leave_unmerged
from _startup_harness import harness_with_real_merge_worker, merge_lane_harness_factory
from escalation.queue import EscalationQueue
from test_halt_owner import _make_wip_esc

from orchestrator.harness import Harness
from orchestrator.merge_halt_rehydration import HALT_NOT_REHYDRATED


def _factory(tmp_path: Path) -> Callable[[], Harness]:
    return merge_lane_harness_factory(tmp_path, real_steps=frozenset({'1c0'}))


async def _restart(make: Callable[[], Harness]) -> Harness:
    return await harness_with_real_merge_worker(make, MagicMock())


def _queue(h: Harness) -> EscalationQueue:
    assert h._escalation_queue is not None
    return h._escalation_queue


@pytest.mark.asyncio
async def test_restart_over_a_clean_project_root_leaves_the_queue_running(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    make = _factory(tmp_path)
    filer = await _restart(make)
    esc = _make_wip_esc(_queue(filer), '4843')
    h = await _restart(make)

    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_halt_rehydration'):
        owner = await h._rehydrate_merge_halt()

    assert owner is None
    assert h.get_merge_halt_status()['halted'] is False
    released = _queue(await _restart(make)).get(esc.id)
    assert released is not None
    assert released.halt_released_at is not None
    assert any(
        HALT_NOT_REHYDRATED in r.getMessage() and esc.id in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_restart_over_conflict_markers_re_asserts_the_halt(tmp_path: Path):
    make = _factory(tmp_path)
    filer = await _restart(make)
    esc = _make_wip_esc(_queue(filer), '4843')
    leave_unmerged(filer.git_ops.project_root, 'side')
    h = await _restart(make)

    await h._rehydrate_merge_halt()

    status = h.get_merge_halt_status()
    assert status['halted'] is True
    assert status['owner_esc_id'] == esc.id


@pytest.mark.asyncio
async def test_a_released_record_stays_released_across_a_later_restart(tmp_path: Path):
    make = _factory(tmp_path)
    filer = await _restart(make)
    esc = _make_wip_esc(_queue(filer), '4843')
    first = await _restart(make)
    await first._rehydrate_merge_halt()
    assert first.get_merge_halt_status()['halted'] is False

    dirty_tracked_file(filer.git_ops.project_root)
    second = await _restart(make)
    await second._rehydrate_merge_halt()

    status = second.get_merge_halt_status()
    assert status['halted'] is False
    assert status['owner_esc_id'] is None
    record = _queue(await _restart(make)).get(esc.id)
    assert record is not None
    assert record.halt_released_at is not None


@pytest.mark.asyncio
async def test_a_parked_halt_owner_over_a_dirty_project_root_still_halts(tmp_path: Path):
    make = _factory(tmp_path)
    filer = await _restart(make)
    queue = _queue(filer)
    esc = _make_wip_esc(queue, '3537', category='stash_failed')
    parked = queue.park(esc.id, 'promoted for human triage')
    assert parked is not None and parked.level == 2
    dirty_tracked_file(filer.git_ops.project_root)
    h = await _restart(make)

    assert await h._rehydrate_merge_halt() == parked.id

    status = h.get_merge_halt_status()
    assert status['halted'] is True
    assert status['owner_esc_id'] == parked.id
