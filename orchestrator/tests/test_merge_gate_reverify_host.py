"""The single-branch gate re-verify runs on the head's own leased host, with
the lane's own verify handles (task 6278, folding 6344).

Every scene drives one request through the lane's own queue over a real tmp
git repo. Main moves from inside the request's first verify, so
``advance_main`` rebases the verified merge onto unverified drift and the
gate genuinely re-verifies the rebased tree.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import functools
import subprocess
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import (
    FakeVerifier,
    VerifyScript,
    fails,
    hangs_until,
    healthy_verify_clock,
    lane_scene_config,
    make_lane,
    merge_through_lane,
    passes,
    running_lane,
)
from _orch_helpers import git_env_with_ceiling, wait_responsive
from _recording_event_store import _RecordingEventStore
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeRequest, QueuedBranch
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import RunnerUnavailable

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
    """A whole-tree gate that, unless *overrides* say otherwise, never takes
    the same-tree arm, so a moved main always re-verifies."""
    repo = seed_repo(tmp_path / 'repo', _SEED)
    config = lane_scene_config(repo, _GIT, **{
        'merge_verify_breadth': 'full', 'merge_verify_min_free_disk_bytes': 0,
        'merge_gate_history_pure': False, **overrides,
    })
    return _Scene(repo=repo, git_ops=GitOps(_GIT, repo), config=config)


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    return _scene(tmp_path)


def _remote_scene(tmp_path: Path, **overrides: Any) -> _Scene:
    """A scene whose lane prefers a free remote host."""
    return _scene(tmp_path, verify_host_policy='prefer_remote', **overrides)


@pytest.fixture
def remote_scene(tmp_path: Path) -> _Scene:
    """As DF's own config does, the lane adopts a remote green without a local cross-check."""
    return _remote_scene(tmp_path, verify_cross_check_remote_green=False)


def _request(scene: _Scene, name: str, worktree: Path) -> MergeRequest:
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


async def _single(scene: _Scene, name: str) -> MergeRequest:
    worktree = (await scene.git_ops.create_worktree(name)).path
    (worktree / f'{name}.txt').write_text(f'{name}\n')
    await scene.git_ops.commit(worktree, f'{name}: own work')
    return _request(scene, name, worktree)


class _SamplesHeads(FakeVerifier):
    """A ``FakeVerifier`` whose ``heads`` keeps the HEAD each call was handed,
    sampled when the call starts."""

    def __init__(self, *, sequence: Iterable[VerifyScript] = ()) -> None:
        super().__init__(sequence=sequence)
        self.heads: list[str] = []

    def next_script(self, task_id: str | None) -> VerifyScript:
        self.heads.append(_git(self.verify_calls[-1].worktree, 'rev-parse', 'HEAD'))
        return super().next_script(task_id)


class _MovesMainOnFirstCall(_SamplesHeads):
    """A ``_SamplesHeads`` that lands unrelated drift on main from inside its first call."""

    def __init__(self, repo: Path, *, sequence: Iterable[VerifyScript] = ()) -> None:
        super().__init__(sequence=sequence)
        self._repo = repo

    def next_script(self, task_id: str | None) -> VerifyScript:
        script = super().next_script(task_id)
        if len(self.heads) == 1:
            _land_drift(self._repo)
        return script


_HostScript = Callable[['_HeadHost'], Awaitable[VerifyResult]]


async def _host_passes(host: _HeadHost) -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='',
        summary=f'green on {host.name}',
    )


def _host_moves_main_then_passes(move_main: Callable[[], None]) -> _HostScript:
    async def script(host: _HeadHost) -> VerifyResult:
        move_main()
        return await _host_passes(host)

    return script


async def _host_fails(host: _HeadHost) -> VerifyResult:
    return VerifyResult(
        passed=False, test_output='', lint_output='', type_output='',
        summary=f'red on {host.name}', category='test_failure',
    )


async def _host_unreachable(host: _HeadHost) -> VerifyResult:
    host.healthy = False
    raise RunnerUnavailable(f'ssh {host.name} exited 255: connection reset')


def _host_unreachable_once(released: asyncio.Event) -> _HostScript:
    async def script(host: _HeadHost) -> VerifyResult:
        host.gate_entered.set()
        await released.wait()
        return await _host_unreachable(host)

    return script


async def _host_hangs(host: _HeadHost) -> VerifyResult:
    host.gate_entered.set()
    never: asyncio.Future[VerifyResult] = asyncio.get_running_loop().create_future()
    return await never


class _HeadHost:
    """The remote verify host a head leases, duck-typed to
    ``verify_runner.py::VerifyRunner`` as test_train_verify_host.py::_FakeRemote is.

    Not a ``RemoteRunner``, so ``VerifyRunnerPool.dispatch`` calls it with
    ``(merge_sha, spec)`` alone and skips the INV-2 code sync. Each
    ``run_merge_verify`` call keeps its merge SHA in ``calls``, plays the next
    of *scripts* (passing once they run out) and records how it ended in
    ``endings``. ``live`` is set while a call runs, as ``RemoteRunner``'s
    in-flight id is; ``cancel_verify`` answers *cancel_rc*, and ``cancels``
    records whether each remote cancel found a call still live to abort.
    ``clean_probes`` counts clean-probes. An unhealthy host never probes
    healthy or clean, so the reprobe sweep cannot re-admit it during a test.
    """

    is_local: ClassVar[bool] = False

    def __init__(self, *scripts: _HostScript, name: str = 'laptop', cancel_rc: int = 0) -> None:
        self.name = name
        self.healthy = True
        self.cancel_rc = cancel_rc
        self._scripts = collections.deque(scripts)
        self.calls: list[str] = []
        self.endings: list[str] = []
        self.cancels: list[bool] = []
        self.clean_probes = 0
        self.live = False
        self.gate_entered = asyncio.Event()

    async def health(self) -> bool:
        return self.healthy

    async def probe_clean(self) -> bool:
        self.clean_probes += 1
        return self.healthy

    async def cancel_verify(self) -> int:
        self.cancels.append(self.live)
        return self.cancel_rc

    async def run_merge_verify(self, merge_sha: str, spec: Any) -> VerifyResult:
        self.calls.append(merge_sha)
        script = self._scripts.popleft() if self._scripts else _host_passes
        self.live = True
        try:
            result = await script(self)
        except asyncio.CancelledError:
            self.endings.append('cancelled')
            raise
        except BaseException:
            self.endings.append('raised')
            raise
        finally:
            self.live = False
        self.endings.append('returned')
        return result


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


# ─── A remote-lease head gates on the host it already holds ─────────────────


def _moves_main_then_gates(scene: _Scene) -> _HeadHost:
    return _HeadHost(
        _host_moves_main_then_passes(functools.partial(_land_drift, scene.repo)), _host_passes,
    )


async def test_a_remote_lease_head_gates_on_the_host_it_holds(remote_scene: _Scene) -> None:
    request = await _single(remote_scene, 'gate')
    host = _moves_main_then_gates(remote_scene)
    verifier = FakeVerifier()
    store = _RecordingEventStore()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        remote_scene.git_ops, queue,
        verifier=verifier, clock=healthy_verify_clock(), event_store=store,
        remote_runners=[host],
    )

    outcome = await merge_through_lane(lane, queue, request)

    assert outcome.status == 'done', outcome
    landed = remote_scene.main_sha()
    assert len(host.calls) == 2, host.calls
    assert host.calls[1] == landed != host.calls[0]
    assert verifier.verify_calls == []
    assert [row['runner'] for row in _rows(store, 'merge_verify', landed)] == ['laptop']


async def test_a_red_gate_on_the_head_host_blocks_without_a_local_rerun(
    remote_scene: _Scene,
) -> None:
    request = await _single(remote_scene, 'gate')
    host = _HeadHost(
        _host_moves_main_then_passes(functools.partial(_land_drift, remote_scene.repo)),
        _host_fails,
    )
    verifier = FakeVerifier()
    store = _RecordingEventStore()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        remote_scene.git_ops, queue,
        verifier=verifier, clock=healthy_verify_clock(), event_store=store,
        remote_runners=[host],
    )

    outcome = await merge_through_lane(lane, queue, request)

    assert outcome.status == 'blocked', outcome
    assert outcome.reason.startswith('Post-merge verification failed'), outcome.reason
    first, rebased = host.calls
    assert rebased != first and rebased != remote_scene.main_sha()
    assert verifier.verify_calls == []
    assert [row['runner'] for row in _rows(store, 'merge_verify', rebased)] == ['laptop']


async def test_a_remote_gate_keeps_the_local_cross_check(tmp_path: Path) -> None:
    scene = _remote_scene(tmp_path, verify_cross_check_remote_green=True)
    request = await _single(scene, 'gate')
    host = _moves_main_then_gates(scene)
    store = _RecordingEventStore()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue,
        verifier=FakeVerifier(), clock=healthy_verify_clock(), event_store=store,
        remote_runners=[host],
    )

    outcome = await merge_through_lane(lane, queue, request)

    assert outcome.status == 'done', outcome
    assert len(host.calls) == 2, host.calls
    parity = [
        (event['data']['merge_sha'], event['data']['remote_runner'])
        for name, event in store.events if name == 'verdict_parity_ok'
    ]
    assert parity == [(host.calls[0], 'laptop'), (scene.main_sha(), 'laptop')]


# ─── A declared history leg stays local whatever the head holds ─────────────


def _commit_file(worktree: Path, name: str) -> str:
    (worktree / name).write_text(f'{name}\n')
    _git(worktree, 'add', '--', name)
    _git(worktree, 'commit', '-m', f'add {name}')
    return _git(worktree, 'rev-parse', 'HEAD')


async def test_a_same_tree_history_leg_stays_local_on_a_remote_lease_head(
    tmp_path: Path,
) -> None:
    """Main gains the head's patch under the follower's verify, so the rebased
    follower carries the tree it verified: the gate runs only the declared
    history leg, whose runner is local-only."""
    leg_head = tmp_path / 'leg-head'
    scene = _remote_scene(
        tmp_path, verify_cross_check_remote_green=False,
        merge_gate_history_pure=True,
        merge_gate_history_test_command=f'git rev-parse HEAD > {leg_head}',
    )
    head_wt = (await scene.git_ops.create_worktree('head')).path
    head_commit = _commit_file(head_wt, 'head.txt')
    follower_wt = (await scene.git_ops.create_worktree('follower')).path
    _git(follower_wt, 'merge', '--ff-only', head_commit)
    _commit_file(follower_wt, 'follower.txt')
    follower = _request(scene, 'follower', follower_wt)
    lands_the_heads_patch = functools.partial(
        _git_until_it_takes, scene.repo, 'cherry-pick', '-x', head_commit,
    )
    host = _HeadHost(_host_moves_main_then_passes(lands_the_heads_patch))
    store = _RecordingEventStore()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue,
        verifier=FakeVerifier(), clock=healthy_verify_clock(), event_store=store,
        remote_runners=[host],
    )

    outcome = await merge_through_lane(lane, queue, follower)

    assert outcome.status == 'done', outcome
    landed = scene.main_sha()
    [gate] = [event['data'] for name, event in store.events if name == 'merge_reverify_gate']
    assert gate['reason'] == 'same_tree_history_leg', gate
    assert host.calls == [gate['verified_merge_sha']]
    assert leg_head.read_text().strip() == landed
    assert [row['runner'] for row in _rows(store, 'merge_verify', landed)] == ['history-leg']


# ─── An unreachable head host moves the gate to local ───────────────────────


async def test_the_gate_falls_back_to_local_when_the_head_host_is_unreachable(
    remote_scene: _Scene,
) -> None:
    """A transport fault at gate time never strands or blocks the item: the
    host is benched for reprobe and the same gate re-verify re-runs on local,
    its restart-drain deadline measured from the local re-run's start."""
    request = await _single(remote_scene, 'gate')
    unreachable_now = asyncio.Event()
    host = _HeadHost(
        _host_moves_main_then_passes(functools.partial(_land_drift, remote_scene.repo)),
        _host_unreachable_once(unreachable_now),
    )
    release_local = asyncio.Event()
    verifier = _SamplesHeads(sequence=[hangs_until(release_local)])
    clock = healthy_verify_clock()
    store = _RecordingEventStore()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        remote_scene.git_ops, queue,
        verifier=verifier, clock=clock, event_store=store, remote_runners=[host],
    )

    async with running_lane(lane) as run:
        await queue.put(request)
        await wait_responsive(host.gate_entered.wait(), label='gate re-verify under way on the head host')
        (remote_gate,) = lane.snapshot()['restart_drain']['verifies_in_flight']
        clock.time += 5_000.0
        unreachable_now.set()
        await wait_responsive(verifier.await_entry(1), label='gate re-verify re-run on local')
        (local_gate,) = lane.snapshot()['restart_drain']['verifies_in_flight']
        release_local.set()
        outcome = await run.outcome(request)

    assert (remote_gate['kind'], local_gate['kind']) == ('gate_reverify', 'gate_reverify')
    assert local_gate['started_ts'] >= remote_gate['started_ts'] + 5_000.0
    assert outcome.status == 'done', outcome
    landed = remote_scene.main_sha()
    on_main = _git(remote_scene.repo, 'ls-tree', '--name-only', 'main').splitlines()
    assert {'drift.txt', 'gate.txt'} <= set(on_main), on_main
    assert verifier.heads == [landed]
    assert [row['runner'] for row in _rows(store, 'merge_verify', landed)] == ['local']
    (laptop,) = [row for row in lane.snapshot()['hosts'] if row['name'] == host.name]
    assert laptop['quarantined'] is True, laptop
    assert 'exited 255' in laptop['reason'], laptop


# ─── A lane cancelled mid remote gate aborts the remote verify ──────────────


@dataclasses.dataclass(frozen=True)
class _Unwound:
    """The head host's census row, and whether its verify was still live,
    read the moment a cancelled lane's ``run()`` finished."""

    host: dict[str, Any]
    verify_live: bool


async def _abort_a_gate_verifying_on(scene: _Scene, *, cancel_rc: int = 0) -> tuple[_HeadHost, _Unwound]:
    """Cancel the lane's ``run()`` while a head's gate re-verify runs on the
    remote it holds, as a lane crash or a hard shutdown does."""
    request = await _single(scene, 'gate')
    host = _HeadHost(
        _host_moves_main_then_passes(functools.partial(_land_drift, scene.repo)),
        _host_hangs,
        cancel_rc=cancel_rc,
    )
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue,
        verifier=FakeVerifier(), clock=healthy_verify_clock(), remote_runners=[host],
    )

    with pytest.raises(asyncio.CancelledError):
        async with running_lane(lane) as run:
            await queue.put(request)
            await wait_responsive(host.gate_entered.wait(), label='gate re-verify under way on the head host')
            run.task.cancel()
            await wait_responsive(asyncio.wait({run.task}), label='lane run() unwound')
            (row,) = [census for census in lane.snapshot()['hosts'] if census['name'] == host.name]
            unwound = _Unwound(host=row, verify_live=host.live)

    return host, unwound


async def test_cancelling_the_lane_mid_remote_gate_aborts_the_remote_verify_while_live(
    remote_scene: _Scene,
) -> None:
    """The merger awaits the gate inline, so cancelling the lane must abort the
    remote verify through the worker's teardown chokepoint while it is still
    live; otherwise the verify-merge is orphaned on the host (task 1757)."""
    host, unwound = await _abort_a_gate_verifying_on(remote_scene)

    assert host.cancels[:1] == [True], (
        'the abort must precede the local cancel, or the remote verify-merge '
        f'process is orphaned; cancels found live: {host.cancels}'
    )
    assert unwound.verify_live is False
    assert host.endings[1] == 'cancelled', host.endings
    assert (unwound.host['slot_state'], unwound.host['quarantined']) == ('free', False)


async def test_a_refused_cancel_parks_the_aborted_gates_remote_until_it_probes_clean(
    remote_scene: _Scene,
) -> None:
    """As for an aborted train: a remote that refuses the cancel may still be
    running the verify, so its slot is parked and freed only once the host
    probes clean."""
    host, unwound = await _abort_a_gate_verifying_on(remote_scene, cancel_rc=1)

    assert host.clean_probes == 1
    assert (unwound.host['slot_state'], unwound.host['quarantined']) == ('free', False)
