"""The single-branch gate re-verify runs with the lane's own verify handles
(task 6278, folding 6344).

Every scene drives one request through the lane's own queue over a real tmp
git repo. Main moves from inside the request's first verify, so
``advance_main`` rebases the verified merge onto unverified drift and the
gate genuinely re-verifies the rebased tree.
"""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import (
    FakeVerifier,
    VerifyScript,
    fails,
    healthy_verify_clock,
    lane_scene_config,
    make_lane,
    merge_through_lane,
    passes,
)
from _orch_helpers import git_env_with_ceiling
from _recording_event_store import _RecordingEventStore
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeRequest, QueuedBranch

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')
_INDEX_LOCK_PATIENCE_SECS = 30.0


def _git_env(cwd: Path) -> dict[str, str]:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    return env


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ['git', *args], cwd=cwd, env=_git_env(cwd), check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _git_until_it_takes(repo: Path, *args: str) -> None:
    """Run one git command, retrying while the live lane holds the repo's index lock."""
    deadline = time.monotonic() + _INDEX_LOCK_PATIENCE_SECS
    while True:
        completed = subprocess.run(
            ['git', *args], cwd=repo, env=_git_env(repo), capture_output=True, text=True,
        )
        if completed.returncode == 0:
            return
        assert 'index.lock' in completed.stderr and time.monotonic() < deadline, completed.stderr
        time.sleep(0.05)


def _land_drift(repo: Path) -> None:
    """Move main the way a human direct commit does, disjoint from the branch.

    Stages only the drift file: ``add -A`` would also stage the lane's live
    ``.worktrees/_merge-*`` directories.
    """
    (repo / 'drift.txt').write_text('landed elsewhere\n')
    _git_until_it_takes(repo, 'add', '--', 'drift.txt')
    _git_until_it_takes(repo, 'commit', '-m', 'drift: unrelated direct commit')


@dataclasses.dataclass(frozen=True)
class _Scene:
    repo: Path
    git_ops: GitOps
    config: OrchestratorConfig

    def main_sha(self) -> str:
        return _git(self.repo, 'rev-parse', 'main')


def _scene(tmp_path: Path, **overrides: Any) -> _Scene:
    """A whole-tree gate that never takes the same-tree arm, so a moved main
    always re-verifies."""
    repo = seed_repo(tmp_path / 'repo', _SEED)
    config = lane_scene_config(
        repo, _GIT,
        merge_verify_breadth='full', merge_verify_min_free_disk_bytes=0,
        merge_gate_history_pure=False, **overrides,
    )
    return _Scene(repo=repo, git_ops=GitOps(_GIT, repo), config=config)


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    return _scene(tmp_path)


async def _single(scene: _Scene, name: str) -> MergeRequest:
    worktree = (await scene.git_ops.create_worktree(name)).path
    (worktree / f'{name}.txt').write_text(f'{name}\n')
    await scene.git_ops.commit(worktree, f'{name}: own work')
    return MergeRequest(
        task_id=name,
        branch=QueuedBranch.parse(name, _GIT.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
    )


class _MovesMainOnFirstCall(FakeVerifier):
    """A ``FakeVerifier`` that lands unrelated drift on main from inside its
    first call. ``heads`` keeps the HEAD each call was handed, sampled when
    the call starts."""

    def __init__(self, repo: Path, *, sequence: Iterable[VerifyScript] = ()) -> None:
        super().__init__(sequence=sequence)
        self._repo = repo
        self.heads: list[str] = []

    def next_script(self, task_id: str | None) -> VerifyScript:
        self.heads.append(_git(self.verify_calls[-1].worktree, 'rev-parse', 'HEAD'))
        if len(self.heads) == 1:
            _land_drift(self._repo)
        return super().next_script(task_id)


def _rows(store: _RecordingEventStore, kind: str, merge_sha: str) -> list[dict[str, Any]]:
    return [
        event['data'] for name, event in store.events
        if name == kind and event['data'].get('merge_sha') == merge_sha
    ]


# ─── The gate re-verifies on the lane's injected verifier ───────────────────


async def test_the_gate_reverifies_on_the_lanes_verifier(scene: _Scene) -> None:
    request = await _single(scene, 'gate')
    verifier = _MovesMainOnFirstCall(scene.repo)
    store = _RecordingEventStore()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=verifier, clock=healthy_verify_clock(), event_store=store,
    )

    outcome = await merge_through_lane(lane, queue, request)

    assert outcome.status == 'done', outcome
    landed = scene.main_sha()
    assert len(verifier.verify_calls) == 2, verifier.verify_calls
    assert verifier.heads[1] == landed != verifier.heads[0]
    assert len(_rows(store, 'merge_verify', landed)) == 1, store.events


async def test_a_red_gate_reverify_spawns_a_dry_run_investigation(scene: _Scene) -> None:
    request = await _single(scene, 'gate')
    verifier = _MovesMainOnFirstCall(
        scene.repo, sequence=[passes(), fails(category='test_failure', summary='red gate')],
    )
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue,
        verifier=verifier, clock=healthy_verify_clock(), event_store=_RecordingEventStore(),
        scheduler=MagicMock(), mcp=MagicMock(),
    )

    outcome = await merge_through_lane(lane, queue, request)

    assert outcome.status == 'blocked', outcome
    assert outcome.reason.startswith('Post-merge verification failed'), outcome.reason
    assert len(verifier.investigations) == 1, verifier.investigations
