"""A restarted Harness re-asserts a preserved merge halt only while project_root still needs it (task 3876).

Each ``_restart`` is a new Harness process view over the same durable
escalation queue and the same real checkout, so the predicate is measured as a
redeployed orchestrator measures it.

This file imports no ``orchestrator.merge_lane`` or ``orchestrator.git_ops``
module: it reads harness privates, which the merge-lane ratchet
(``scripts/merge_lane_metrics.py``) would count against any lane-importing file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from _git_fixtures import seed_repo
from escalation.queue import EscalationQueue
from test_halt_owner import _FakeMergeWorker, _make_wip_esc
from test_merge_halt_rehydration import _dirty, _leave_unmerged

from orchestrator.harness import Harness
from orchestrator.merge_halt_rehydration import HALT_NOT_REHYDRATED


def _restart(mock_orch_config: MagicMock, repo: Path, esc_dir: Path) -> Harness:
    mock_orch_config.project_root = repo
    mock_orch_config.orphan_l0_reaper_enabled = False
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)
    h._escalation_queue = EscalationQueue(esc_dir)
    h._escalation_queue.set_resolve_callback(h._on_escalation_resolved)
    h._merge_worker = _FakeMergeWorker()  # type: ignore[assignment]
    return h


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def esc_dir(tmp_path: Path) -> Path:
    return tmp_path / 'escalations'


@pytest.mark.asyncio
async def test_restart_over_a_clean_project_root_leaves_the_queue_running(
    mock_orch_config: MagicMock, repo: Path, esc_dir: Path,
    caplog: pytest.LogCaptureFixture,
):
    esc = _make_wip_esc(EscalationQueue(esc_dir), '4843')
    h = _restart(mock_orch_config, repo, esc_dir)

    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_halt_rehydration'):
        owner = await h._rehydrate_merge_halt()

    assert owner is None
    assert h.get_merge_halt_status()['halted'] is False
    released = EscalationQueue(esc_dir).get(esc.id)
    assert released is not None
    assert released.halt_released_at is not None
    assert any(
        HALT_NOT_REHYDRATED in r.getMessage() and esc.id in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_restart_over_conflict_markers_re_asserts_the_halt(
    mock_orch_config: MagicMock, repo: Path, esc_dir: Path,
):
    esc = _make_wip_esc(EscalationQueue(esc_dir), '4843')
    _leave_unmerged(repo, 'side')
    h = _restart(mock_orch_config, repo, esc_dir)

    await h._rehydrate_merge_halt()

    status = h.get_merge_halt_status()
    assert status['halted'] is True
    assert status['owner_esc_id'] == esc.id


@pytest.mark.asyncio
async def test_a_released_record_stays_released_across_a_later_restart(
    mock_orch_config: MagicMock, repo: Path, esc_dir: Path,
):
    esc = _make_wip_esc(EscalationQueue(esc_dir), '4843')
    first = _restart(mock_orch_config, repo, esc_dir)
    await first._rehydrate_merge_halt()
    assert first.get_merge_halt_status()['halted'] is False

    _dirty(repo)
    second = _restart(mock_orch_config, repo, esc_dir)
    await second._rehydrate_merge_halt()

    status = second.get_merge_halt_status()
    assert status['halted'] is False
    assert status['owner_esc_id'] is None
    record = EscalationQueue(esc_dir).get(esc.id)
    assert record is not None
    assert record.halt_released_at is not None


@pytest.mark.asyncio
async def test_a_parked_halt_owner_over_a_dirty_project_root_still_halts(
    mock_orch_config: MagicMock, repo: Path, esc_dir: Path,
):
    queue = EscalationQueue(esc_dir)
    esc = _make_wip_esc(queue, '3537', category='stash_failed')
    parked = queue.park(esc.id, 'promoted for human triage')
    assert parked is not None and parked.level == 2
    _dirty(repo)
    h = _restart(mock_orch_config, repo, esc_dir)

    assert await h._rehydrate_merge_halt() == parked.id

    status = h.get_merge_halt_status()
    assert status['halted'] is True
    assert status['owner_esc_id'] == parked.id
