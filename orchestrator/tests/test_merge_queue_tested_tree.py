"""Each in-flight entry shows the tree it verifies: its base, its root and its members.

plans/merge-pool-batches-prd.md §6.1: a live merger and verifier over a real
repo, observed only through ``snapshot()['entries'][i]['tree']``, the request
futures and the fake verify hosts' ``calls``. H verifies alone on main; S
chains F1 and F2 on H's merge commit, a batch built on an in-flight owner that
took no links; X stacks on S's chain tip.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeRemoteRunner,
    FakeVerifier,
    branch_merge_request,
    healthy_verify_clock,
    lane_entry,
    lane_scene_config,
    make_lane,
    running_lane,
)
from _orch_helpers import MERGE_GATE_BARRIER_TIMEOUT, wait_responsive

from orchestrator.config import DepthRow, GitConfig, MergeDeepConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_DEEP = MergeDeepConfig(
    chain_cap=3, chain_build_timeout_secs=300,
    chain_reservation_timeout_secs=1e9, depth_table=[DepthRow(0.0, 3)],
)
_NAMES = ('H', 'S', 'F1', 'F2', 'X')


async def _rev_parse(repo: Path, rev: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        'git', 'rev-parse', rev, cwd=repo,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    assert proc.returncode == 0, err.decode()
    return out.decode().strip()


class _Host:
    """A remote verify host that holds each verify until released."""

    def __init__(self, name: str) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.runner = FakeRemoteRunner(name, release=self.release, entered=self.entered)

    async def wait_entered(self, label: str) -> None:
        await wait_responsive(
            self.entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label=label,
        )


def _tree(lane: Any, request: Any) -> dict[str, Any]:
    entry = lane_entry(lane, request.request_id)
    assert entry is not None, request.task_id
    tree = entry['tree']
    assert tree is not None, request.task_id
    return tree


async def test_each_in_flight_entry_shows_the_tree_it_verifies(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path / 'repo')
    config = lane_scene_config(
        repo, _GIT, verify_host_policy='prefer_remote',
        verify_cross_check_remote_green=False, merge_deep=_DEEP,
    )
    assert config.merge_deep.model_dump() == _DEEP.model_dump(), (
        f'the operational yaml merged into the scene config: {config.merge_deep!r}'
    )
    git_ops = GitOps(_GIT, repo)
    reqs = {
        name: await branch_merge_request(git_ops, config, name, filename=f'{name}.txt')
        for name in _NAMES
    }
    head_host, s_host, spare = _Host('head-host'), _Host('s-host'), _Host('spare')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        git_ops, queue, verifier=FakeVerifier(), clock=healthy_verify_clock(),
        speculation_depth=3,
        remote_runners=[head_host.runner, s_host.runner, spare.runner],
        event_store=EventStore(db_path=tmp_path / 'events.db', run_id='tested-tree'),
    )
    main0 = await _rev_parse(repo, 'main')
    tips = {name: await _rev_parse(repo, f'task/{name}') for name in _NAMES}

    async with running_lane(lane):
        queue.put_nowait(reqs['H'])
        await head_host.wait_entered('H verify under way on head-host')
        for name in ('S', 'F1', 'F2'):
            queue.put_nowait(reqs[name])
        await s_host.wait_entered("S's tip verify under way on s-host")
        (row,) = lane.snapshot()['chain_reservations']
        assert (row['owner_task_id'], row['link_task_ids']) == ('S', ['F1', 'F2'])
        queue.put_nowait(reqs['X'])
        await spare.wait_entered("X's verify under way on spare")

        h_tree = _tree(lane, reqs['H'])
        s_tree = _tree(lane, reqs['S'])
        x_tree = _tree(lane, reqs['X'])
        link_ids = {reqs['F1'].request_id, reqs['F2'].request_id}
        link_entries = [e for e in lane.snapshot()['entries'] if e['request_id'] in link_ids]

        for host in (head_host, s_host, spare):
            host.release.set()
        outcomes = {
            name: await wait_responsive(reqs[name].result, label=f'{name} outcome')
            for name in _NAMES
        }

    h_commit = head_host.runner.calls[0]
    assert h_tree == {
        'base_commit': main0,
        'root_commit': main0,
        'members': [{
            'request_id': reqs['H'].request_id, 'task_id': 'H',
            'merge_commit': h_commit, 'tip': tips['H'], 'own': True,
        }],
    }

    s_members = s_tree['members']
    assert (s_tree['base_commit'], s_tree['root_commit']) == (h_commit, main0)
    assert [m['task_id'] for m in s_members] == ['H', 'S', 'F1', 'F2']
    assert [m['own'] for m in s_members] == [False, True, True, True]
    assert s_members[-1]['merge_commit'] == s_host.runner.calls[0]
    assert [m['tip'] for m in s_members] == [tips[m['task_id']] for m in s_members]
    for before, member in zip(s_members, s_members[1:], strict=False):
        assert await _rev_parse(repo, f"{member['merge_commit']}^1") == before['merge_commit']

    x_members = x_tree['members']
    assert (x_tree['base_commit'], x_tree['root_commit']) == (s_host.runner.calls[0], main0)
    assert [m['task_id'] for m in x_members] == ['H', 'S', 'F1', 'F2', 'X']
    assert [m['own'] for m in x_members] == [False, False, False, False, True]
    assert x_members[:4] == [{**m, 'own': False} for m in s_members]
    assert x_members[-1]['merge_commit'] == spare.runner.calls[0]

    assert len(link_entries) == 2
    assert [e['tree'] for e in link_entries] == [None, None]

    assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(_NAMES, 'done')
