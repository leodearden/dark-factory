"""The scheduler's dispatch gate over a follow-up holder and a held task, as fused-memory writes them.

plans/follow-up-filing-shape-prd.md boundary row 18, held-absent half: the holder never dispatches,
and a held task is not ready while its holder is pending. Both rows are minted by fused-memory's
own holder code into a real SQLite store and read back, so the gate judges the shape fused-memory
actually writes rather than a hand-built copy of it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.followup_hold import FollowupHolders, hold_at_create

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

_FOLLOWUP = {'source': 'agent-followup', 'spawn_context': 'agent_followup', 'spawned_from': '101'}


@dataclass(frozen=True)
class _Minted:
    holder: dict[str, Any]
    held: dict[str, Any]

    def status_map(self, *, holder_status: str | None = None) -> dict[str, str]:
        return {
            str(self.holder['id']): holder_status or self.holder['status'],
            str(self.held['id']): self.held['status'],
        }


async def _unannounced(project_root: str, holder_id: int) -> None:
    return None


@pytest_asyncio.fixture
async def minted(tmp_path: Path) -> AsyncIterator[_Minted]:
    root = str(tmp_path / 'proj')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root), task_metadata_enforce=True)
    await backend.start()
    try:
        holder_id = await FollowupHolders(on_minted=_unannounced).ensure_holder(backend, root)
        kwargs, metadata = hold_at_create(
            {'title': 'Low follow-up', 'description': 'One observation.', 'priority': 'low'},
            _FOLLOWUP, holder_id, held_at='2026-11-19T00:00:00+00:00',
        )
        created = await backend.add_task(project_root=root, metadata=json.dumps(metadata), **kwargs)
        yield _Minted(
            holder=await backend.get_task(str(holder_id), root),
            held=await backend.get_task(str(created['id']), root),
        )
    finally:
        await backend.close()


@pytest.fixture
def scheduler(tmp_path: Path) -> Scheduler:
    return Scheduler(OrchestratorConfig(project_root=tmp_path))


def _eligible(scheduler: Scheduler, task: dict[str, Any], status_map: dict[str, str]) -> bool:
    eligible, _signal = scheduler._eligible_for_dispatch(task, str(task['id']), status_map)
    return eligible


def test_the_holder_never_dispatches(scheduler: Scheduler, minted: _Minted) -> None:
    assert minted.holder['status'] == 'pending'

    assert _eligible(scheduler, minted.holder, minted.status_map()) is False


def test_the_holder_is_held_back_by_its_deferred_watch_alone(scheduler: Scheduler, minted: _Minted) -> None:
    triggered = {**minted.holder, 'metadata': {**minted.holder['metadata'], 'trigger_met': True}}

    assert _eligible(scheduler, triggered, minted.status_map()) is True


def test_a_held_task_waits_while_its_holder_is_pending(scheduler: Scheduler, minted: _Minted) -> None:
    assert minted.held['status'] == 'pending'

    assert _eligible(scheduler, minted.held, minted.status_map()) is False


@pytest.mark.parametrize('holder_status', ['done', 'cancelled'])
def test_a_held_task_is_ready_once_its_holder_settles(
    scheduler: Scheduler, minted: _Minted, holder_status: str,
) -> None:
    assert _eligible(scheduler, minted.held, minted.status_map(holder_status=holder_status)) is True


def test_a_held_task_still_waits_on_a_deferred_holder(scheduler: Scheduler, minted: _Minted) -> None:
    assert _eligible(scheduler, minted.held, minted.status_map(holder_status='deferred')) is False
