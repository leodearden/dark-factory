"""The ordinary claim refuses a task started after its ready list was built.

plans/early-start-on-predecessor-branch-prd.md §5.4 "Never started twice",
invariant 11, test row 34 (ordinary half), task 10476. Pass A suspends in the
landed-outbox hook the harness installs, after its ready list already holds D;
pass B starts D meanwhile. The lock table admits a requester's own holds, so
only a claim-time check stops pass A from starting D a second time.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from _orch_helpers import HermeticMcpSession, wait_responsive
from _park_test_helpers import make_task

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.overrides import OverrideStore
from orchestrator.scheduler import Scheduler

pytestmark = [pytest.mark.usefixtures('code_default_config'), pytest.mark.asyncio]

D = make_task('D', 'high', ['a.py'])
E = make_task('E', 'medium', ['b.py'])


@dataclass(frozen=True)
class _Rig:
    scheduler: Scheduler
    store: EventStore
    entered: asyncio.Event
    release: asyncio.Event


def _rig(tmp_path: Path, *tasks: dict, pinned: str | None = None) -> _Rig:
    config = OrchestratorConfig()
    overrides = OverrideStore(tmp_path / 'scheduler_overrides.db')
    if pinned is not None:
        overrides.set_override(str(config.project_root), pinned, pinned=True)
    store = EventStore(tmp_path / 'runs.db', 'run-1')
    scheduler = Scheduler(
        config, store,
        mcp_session=HermeticMcpSession(tasks=list(tasks)),
        override_store=overrides,
        state_snapshot_path=tmp_path / 'scheduler_state.json',
        branch_work_reader=dict,
    )
    scheduler.finish_startup()
    entered, release = asyncio.Event(), asyncio.Event()
    first_call = True

    async def landed_outbox_hook(task_id: str) -> bool:
        nonlocal first_call
        if first_call:
            first_call = False
            entered.set()
            await release.wait()
        return False

    scheduler._landed_outbox_gate = landed_outbox_hook
    return _Rig(scheduler, store, entered, release)


async def _race(rig: _Rig) -> tuple[str | None, str | None]:
    """Pass A builds its ready list and suspends; pass B runs whole; then A resumes."""
    pass_a = asyncio.create_task(rig.scheduler.acquire_next())
    await wait_responsive(rig.entered.wait(), label='pass A reaches the landed-outbox hook')
    b = await rig.scheduler.acquire_next()
    rig.release.set()
    a = await wait_responsive(pass_a, label='pass A finishes after the release')
    return (a.task_id if a else None), (b.task_id if b else None)


def _acquired(store: EventStore, task_id: str) -> int:
    return sum(1 for event in store.fetch_events_by_type(EventType.lock_acquired) if event['task_id'] == task_id)


async def test_a_ranked_task_started_meanwhile_is_not_claimed_again(tmp_path):
    rig = _rig(tmp_path, D, E)

    a, b = await _race(rig)

    assert (b, a) == ('D', 'E')
    assert _acquired(rig.store, 'D') == 1
    assert _acquired(rig.store, 'E') == 1


async def test_a_pinned_task_started_meanwhile_is_not_claimed_again(tmp_path):
    rig = _rig(tmp_path, D, E, pinned='D')

    a, b = await _race(rig)

    assert (b, a) == ('D', 'E')
    assert _acquired(rig.store, 'D') == 1
    assert _acquired(rig.store, 'E') == 1


async def test_the_only_candidate_started_meanwhile_leaves_nothing_to_claim(tmp_path):
    rig = _rig(tmp_path, D)

    a, b = await _race(rig)

    assert (b, a) == ('D', None)
    assert _acquired(rig.store, 'D') == 1
