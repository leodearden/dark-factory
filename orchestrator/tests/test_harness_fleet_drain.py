"""The harness wiring of the restart drain (task 5371).

The drain itself is ``orchestrator.fleet_drain.DrainParticipant``, pinned in
``test_fleet_drain.py``.  This one test proves the harness wires it: a
merge-heartbeat pass halts a real merge lane mid-verify and publishes the
acknowledgement, and stopping the merge worker reports the verify it kills.
The fleet dir and the unit's identity are set through the environment exactly
as systemd sets them.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import FakeClock, FakeVerifier, hangs_until, lane_scene_config, make_lane
from _orch_helpers import wait_responsive
from _recording_event_store import _RecordingEventStore

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness
from orchestrator.merge_lane import MergeRequest, QueuedBranch

pytestmark = pytest.mark.asyncio

UNIT = 'orchestrator-dark-factory.service'
INVOCATION = 'caaef715c93445dbb0c1cc608a9f4ef8'
_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)


@pytest.fixture
def fleet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fleet = tmp_path / 'fleet'
    fleet.mkdir()
    monkeypatch.setenv('ORCH_FLEET_DIR', str(fleet))
    monkeypatch.setenv('ORCH_UNIT', UNIT)
    monkeypatch.setenv('INVOCATION_ID', INVOCATION)
    return fleet


@pytest.fixture
def harness(mock_orch_config) -> Harness:
    mock_orch_config.orchestrator_restart_lease_max_age_secs = 14_400.0
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)
    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = _RecordingEventStore()  # type: ignore[assignment]
    return h


def _request_drain(fleet_dir: Path, *, invocation_id: str = INVOCATION) -> int:
    requested_ts = int(time.time())
    (fleet_dir / f'{UNIT}.drain.json').write_text(json.dumps({
        'unit': UNIT, 'invocation_id': invocation_id,
        'sweep_pid': os.getpid(), 'requested_ts': requested_ts,
    }))
    return requested_ts


def _heartbeat(fleet_dir: Path) -> dict[str, Any]:
    return json.loads((fleet_dir / f'{UNIT}.json').read_text())


def _drain_events(harness: Harness) -> list[dict[str, Any]]:
    store: _RecordingEventStore = harness.event_store  # type: ignore[assignment]
    return [record['data'] for name, record in store.events if name == 'fleet_drain']


async def test_the_heartbeat_halts_the_lane_and_the_stop_reports_what_it_kills(
    harness: Harness, fleet_dir: Path, tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path / 'repo', RepoSeed(files=(('README.md', '# Test\n'),), message='init'))
    git_ops = GitOps(_GIT, repo)
    worktree = (await git_ops.create_worktree('held')).path
    (worktree / 'held.txt').write_text('held\n')
    await git_ops.commit(worktree, 'held: own work')
    request = MergeRequest(
        task_id='held', branch=QueuedBranch.parse('held', _GIT.branch_prefix),
        worktree=worktree, pre_rebased=False, task_files=None, module_configs=[],
        config=lane_scene_config(repo, _GIT),
        result=asyncio.get_running_loop().create_future(),
    )
    release = asyncio.Event()
    verifier = FakeVerifier(scripts={'held': hangs_until(release)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        git_ops, queue, verifier=verifier,
        clock=FakeClock(content_mtime=1.0, content_tick=1.0),
    )
    harness._merge_worker = lane
    harness._merge_worker_task = asyncio.create_task(lane.run())
    try:
        await queue.put(request)
        await wait_responsive(verifier.await_entry(1), label='held verify under way')
        requested_ts = _request_drain(fleet_dir)
        await harness._write_merge_heartbeat()
        heartbeat = _heartbeat(fleet_dir)
        assert lane.is_admission_halted is True
        assert heartbeat['drain'] == {
            'requested_ts': requested_ts, 'admission_halted': True, 'refused': None,
        }
        assert [v['task_id'] for v in heartbeat['verifies_in_flight']] == ['held']
        assert _drain_events(harness) == []

        await harness._stop_merge_worker()
    finally:
        release.set()

    (event,) = _drain_events(harness)
    assert event['outcome'] == 'verifies_killed'
    assert event['requested_ts'] == requested_ts
    assert [v['task_id'] for v in event['merge_verifies_killed']] == ['held']
    assert [v['task_id'] for v in event['merge_verifies_awaited']] == ['held']
