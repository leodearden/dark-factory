"""A main that moves under a train's verify: the train re-verifies its rebased tip and lands.

Task 5070.  Every scenario drives a real three-member train through the
production lane over a real git repo.  The members are stacked by
``GitOps.stack_train_branches`` -- the former a coalesced train is built
with -- and main is moved from inside the merge verify, which is the window
in which a direct commit or a queued landing really moves it.
"""
from __future__ import annotations

import asyncio
import dataclasses
import subprocess
import uuid
from collections.abc import Callable, Container
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, call, patch

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import lane_scene_config, make_lane, merge_through_lane
from _orch_helpers import git_env_with_ceiling
from _recording_event_store import _RecordingEventStore
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import (
    TRAIN_VERIFY_FAILED_REASON_PREFIX,
    GroupMergeRequest,
    MergeLane,
    MergeOutcome,
    QueuedBranch,
)
from orchestrator.merge_lane.gates import note_queue_verified_main_tip
from orchestrator.merge_lane.landed_outbox import LandedOutbox
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

_VERIFY_TARGET = 'orchestrator.merge_lane.worker.run_scoped_verification'
_CAS_DERAIL_REASON = 'Train merge advance failed: cas_failed'

_MEMBER_A, _MEMBER_B, _TIP = _MEMBERS = ('mm-a', 'mm-b', 'mm-c')
_MEMBER_FILES = {_MEMBER_A: 'a.txt', _MEMBER_B: 'b.txt', _TIP: 'c.txt'}
_MEMBER_A_LINE_2 = 'shared line 2, edited by member a'

_SEED = RepoSeed(
    files=(
        ('README.md', '# Test\n'),
        ('shared.txt', ''.join(f'shared line {n}\n' for n in range(1, 21))),
    ),
    message='Initial commit',
)

Mover = Callable[[Path], str]


def _git(cwd: Path, *args: str) -> str:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    completed = subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _commit(cwd: Path, message: str, *paths: str) -> str:
    _git(cwd, 'add', '--', *paths)
    _git(cwd, 'commit', '-m', message)
    return _git(cwd, 'rev-parse', 'HEAD')


def _rewrite_shared_line(root: Path, line_no: int, text: str) -> None:
    path = root / 'shared.txt'
    lines = path.read_text().splitlines(keepends=True)
    lines[line_no - 1] = f'{text}\n'
    path.write_text(''.join(lines))


def _main_sha(repo: Path) -> str:
    return _git(repo, 'rev-parse', 'main')


def _parents(repo: Path, rev: str) -> list[str]:
    return _git(repo, 'rev-list', '--parents', '-n', '1', rev).split()[1:]


def _files_at(repo: Path, rev: str) -> set[str]:
    return set(_git(repo, 'ls-tree', '-r', '--name-only', rev).splitlines())


def _shared_lines_at(repo: Path, rev: str) -> list[str]:
    return _git(repo, 'show', f'{rev}:shared.txt').splitlines()


# ─── Drift movers: each commits straight onto main in project_root ──────────


def _disjoint_drift(repo: Path) -> str:
    (repo / 'unrelated.txt').write_text(f'{uuid.uuid4().hex}\n')
    return _commit(repo, 'drift: unrelated direct commit', 'unrelated.txt')


def _shared_drift(repo: Path) -> str:
    """Edit shared.txt near its end, a region member a's line-2 edit rebases past."""
    _rewrite_shared_line(repo, 18, f'shared line 18, edited on main {uuid.uuid4().hex}')
    return _commit(repo, 'drift: shared.txt direct commit', 'shared.txt')


def _queue_landed(mover: Mover) -> Mover:
    """*mover*, with its new tip registered as one this queue landed green."""
    def landed(repo: Path) -> str:
        sha = mover(repo)
        note_queue_verified_main_tip(sha)
        return sha
    return landed


# ─── The train ───────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _Scene:
    repo: Path
    git_ops: GitOps
    config: OrchestratorConfig
    tip_worktree: Path


@dataclasses.dataclass(frozen=True)
class _Train:
    request: GroupMergeRequest
    mark_member_done: AsyncMock
    redrive_member: AsyncMock


async def _member_worktree(git_ops: GitOps, member: str) -> Path:
    worktree = (await git_ops.create_worktree(member)).path
    (worktree / _MEMBER_FILES[member]).write_text(f'{member}\n')
    paths = [_MEMBER_FILES[member]]
    if member == _MEMBER_A:
        _rewrite_shared_line(worktree, 2, _MEMBER_A_LINE_2)
        paths.append('shared.txt')
    _commit(worktree, f'{member}: own work', *paths)
    return worktree


async def _stacked_train_scene(tmp_path: Path, **config_overrides: Any) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    git = GitConfig(
        main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
        push_after_advance=False,
    )
    git_ops = GitOps(git, repo)
    worktrees = {member: await _member_worktree(git_ops, member) for member in _MEMBERS}
    stacked = await git_ops.stack_train_branches(list(_MEMBERS))
    assert stacked.survivors == list(_MEMBERS), stacked
    return _Scene(
        repo=repo, git_ops=git_ops,
        config=lane_scene_config(repo, git, **config_overrides),
        tip_worktree=worktrees[_TIP],
    )


def _train(scene: _Scene, *, train_id: str = 'train-5070') -> _Train:
    tip = QueuedBranch.parse(_TIP, scene.config.git.branch_prefix)
    mark_member_done = AsyncMock()
    redrive_member = AsyncMock()
    request = GroupMergeRequest(
        task_id=_TIP,
        branch=tip,
        worktree=scene.tip_worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        train_id=train_id,
        member_task_ids=list(_MEMBERS),
        tip_branch=tip,
        tip_task_id=_TIP,
        status_check=AsyncMock(return_value=dict.fromkeys(_MEMBERS, 'merge-deferred')),
        mark_member_done=mark_member_done,
        redrive_member=redrive_member,
    )
    return _Train(request, mark_member_done, redrive_member)


# ─── The merge verify ────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _VerifyCall:
    head: str
    """HEAD of the worktree the verify was handed: the tree under test."""
    main: str
    """main in project_root when the verify ran."""


class _VerifyDouble:
    """``run_scoped_verification`` stand-in that can move main while it runs.

    Call *n* (1-based) records what it saw, then commits a drift with
    *mover* when *n* is in *move_on*, then answers red when *n* is in
    *fail_on* and green otherwise.
    """

    def __init__(
        self,
        repo: Path,
        *,
        mover: Mover | None = None,
        move_on: Container[int] = (1,),
        fail_on: Container[int] = (),
    ) -> None:
        self._repo = repo
        self._mover = mover
        self._move_on = move_on
        self._fail_on = fail_on
        self.calls: list[_VerifyCall] = []
        self.drifts: list[str] = []

    async def run_scoped_verification(
        self, worktree: Path, *args: Any, **kwargs: Any,
    ) -> VerifyResult:
        self.calls.append(_VerifyCall(
            head=_git(worktree, 'rev-parse', 'HEAD'), main=_main_sha(self._repo),
        ))
        n = len(self.calls)
        if self._mover is not None and n in self._move_on:
            self.drifts.append(self._mover(self._repo))
        if n in self._fail_on:
            return VerifyResult(
                passed=False, test_output='', lint_output='', type_output='',
                summary='re-verify red', category='pytest',
            )
        return VerifyResult(
            passed=True, test_output='ok', lint_output='', type_output='', summary='ok',
        )


async def _run_train(
    scene: _Scene, train: _Train, verify: _VerifyDouble, store: _RecordingEventStore,
) -> MergeOutcome:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, event_store=store)
    with patch(_VERIFY_TARGET, side_effect=verify.run_scoped_verification):
        return await merge_through_lane(lane, queue, train.request)


def _events(store: _RecordingEventStore, event_type: str) -> list[dict[str, Any]]:
    return [record['data'] for name, record in store.events if name == event_type]


def _attempts(store: _RecordingEventStore, outcome: str, train_id: str) -> list[dict[str, Any]]:
    return [
        data for data in _events(store, 'merge_attempt')
        if data['outcome'] == outcome and data.get('train_id') == train_id
    ]


def _every_member_flipped(sha: str) -> list[Any]:
    return [call(member, sha) for member in _MEMBERS]


# ─── (A) main unmoved: the happy path is pinned ─────────────────────────────


async def test_unmoved_main_verifies_once_and_lands_the_merge_commit(tmp_path: Path) -> None:
    scene = await _stacked_train_scene(tmp_path)
    train = _train(scene)
    verify = _VerifyDouble(scene.repo)
    store = _RecordingEventStore()
    pre_train_main = _main_sha(scene.repo)

    with patch.object(
        scene.git_ops, 'advance_main', wraps=scene.git_ops.advance_main,
    ) as advance:
        outcome = await _run_train(scene, train, verify, store)

    assert outcome.status == 'done', outcome
    assert len(verify.calls) == 1
    assert advance.await_count == 1
    landed = _main_sha(scene.repo)
    parents = _parents(scene.repo, landed)
    assert len(parents) == 2 and parents[0] == pre_train_main, parents
    assert train.mark_member_done.await_args_list == _every_member_flipped(landed)
    assert not _attempts(store, 'gate_retry', train.request.train_id)
    assert not _attempts(store, 'cas_retry', train.request.train_id)
    assert not _events(store, 'train_derailed')


# ─── (B) a direct commit lands during the train verify ──────────────────────


async def test_direct_commit_during_verify_reverifies_the_rebased_tip_and_lands(
    tmp_path: Path,
) -> None:
    scene = await _stacked_train_scene(tmp_path)
    train = _train(scene)
    verify = _VerifyDouble(scene.repo, mover=_disjoint_drift)
    store = _RecordingEventStore()

    outcome = await _run_train(scene, train, verify, store)

    assert outcome.status == 'done', outcome
    assert not _events(store, 'train_derailed')
    assert len(verify.calls) == 2
    [drift] = verify.drifts
    landed = _main_sha(scene.repo)
    assert verify.calls[1] == _VerifyCall(head=landed, main=drift)
    assert {'unrelated.txt', 'a.txt', 'b.txt', 'c.txt'} <= _files_at(scene.repo, 'main')
    assert train.mark_member_done.await_args_list == _every_member_flipped(landed)
    assert len(_attempts(store, 'gate_retry', train.request.train_id)) == 1
    [merged] = _events(store, 'train_merged')
    assert merged['base_sha'] == drift
    outbox = LandedOutbox(scene.repo / 'data' / 'orchestrator' / 'landed_outbox.json')
    row = outbox.lookup(_TIP)
    assert row is not None and row.advanced_sha == landed


# ─── (F) main moves between the train reading it and building its merge ─────


async def test_main_moving_before_the_merge_commit_retries_the_cas_without_reverifying(
    tmp_path: Path,
) -> None:
    scene = await _stacked_train_scene(tmp_path)
    train = _train(scene)
    verify = _VerifyDouble(scene.repo)
    store = _RecordingEventStore()
    drifts: list[str] = []
    real_merge_to_main = scene.git_ops.merge_to_main

    async def merge_after_a_drift(*args: Any, **kwargs: Any) -> Any:
        drifts.append(_disjoint_drift(scene.repo))
        return await real_merge_to_main(*args, **kwargs)

    with patch.object(scene.git_ops, 'merge_to_main', new=merge_after_a_drift):
        outcome = await _run_train(scene, train, verify, store)

    assert outcome.status == 'done', outcome
    assert len(verify.calls) == 1
    [drift] = drifts
    landed = _main_sha(scene.repo)
    parents = _parents(scene.repo, landed)
    assert len(parents) == 2 and parents[0] == drift, parents
    assert {'unrelated.txt', 'a.txt', 'b.txt', 'c.txt'} <= _files_at(scene.repo, 'main')
    assert len(_attempts(store, 'cas_retry', train.request.train_id)) == 1
    assert not _attempts(store, 'gate_retry', train.request.train_id)
    assert not _events(store, 'train_derailed')
    assert train.mark_member_done.await_args_list == _every_member_flipped(landed)


# ─── (J) after a lost CAS, the gate's delta starts at the verified base ─────


async def test_rebase_after_a_lost_cas_is_gated_against_the_verified_base_not_the_reread_main(
    tmp_path: Path,
) -> None:
    """A queue-landed overlap that won the CAS sits below the re-read main, not above it.

    advance_main reports ``rebased_from`` as the main it was handed, which
    after a lost CAS is the re-read main itself, so a delta measured from it
    is empty and the overlap would land unverified.
    """
    scene = await _stacked_train_scene(tmp_path, merge_verify_breadth='scoped')
    train = _train(scene)
    verify = _VerifyDouble(scene.repo)
    store = _RecordingEventStore()
    drifts: list[str] = []
    advances: list[str] = []
    real_merge_to_main = scene.git_ops.merge_to_main
    real_advance_main = scene.git_ops.advance_main

    async def merge_after_a_drift(*args: Any, **kwargs: Any) -> Any:
        drifts.append(_disjoint_drift(scene.repo))
        return await real_merge_to_main(*args, **kwargs)

    async def overlap_lands_after_the_first_advance(*args: Any, **kwargs: Any) -> Any:
        outcome = await real_advance_main(*args, **kwargs)
        advances.append(outcome.result)
        if len(advances) == 1:
            drifts.append(_queue_landed(_shared_drift)(scene.repo))
        return outcome

    with (
        patch.object(scene.git_ops, 'merge_to_main', new=merge_after_a_drift),
        patch.object(scene.git_ops, 'advance_main', new=overlap_lands_after_the_first_advance),
    ):
        outcome = await _run_train(scene, train, verify, store)

    assert outcome.status == 'done', outcome
    assert advances == ['cas_failed', 'rebased_pending_reverify', 'advanced']
    assert len(verify.calls) == 2
    [_, overlap] = drifts
    landed = _main_sha(scene.repo)
    assert verify.calls[1] == _VerifyCall(head=landed, main=overlap)
    landed_shared = _shared_lines_at(scene.repo, 'main')
    assert landed_shared[1] == _MEMBER_A_LINE_2
    assert landed_shared[17] == _shared_lines_at(scene.repo, overlap)[17]
    assert len(_attempts(store, 'cas_retry', train.request.train_id)) == 1
    assert len(_attempts(store, 'gate_retry', train.request.train_id)) == 1
    assert train.mark_member_done.await_args_list == _every_member_flipped(landed)


# ─── (C/D/E) the _disjoint_skip_blockers matrix, for a train ────────────────


@pytest.mark.parametrize(
    ('mover', 'breadth', 'expected_verifies'),
    [
        pytest.param(_queue_landed(_disjoint_drift), 'scoped', 1, id='C-disjoint-queue-landed'),
        pytest.param(_queue_landed(_disjoint_drift), 'full', 2, id='D-disjoint-whole-tree-gate'),
        pytest.param(
            _queue_landed(_shared_drift), 'scoped', 2, id='E-drift-on-a-non-tip-members-file',
        ),
    ],
)
async def test_moved_main_costs_a_reverify_only_when_disjointness_cannot_be_trusted(
    tmp_path: Path, mover: Mover, breadth: str, expected_verifies: int,
) -> None:
    scene = await _stacked_train_scene(tmp_path, merge_verify_breadth=breadth)
    train = _train(scene)
    verify = _VerifyDouble(scene.repo, mover=mover)
    store = _RecordingEventStore()

    outcome = await _run_train(scene, train, verify, store)

    assert outcome.status == 'done', outcome
    assert not _events(store, 'train_derailed')
    assert len(verify.calls) == expected_verifies
    [drift] = verify.drifts
    assert {'a.txt', 'b.txt', 'c.txt'} <= _files_at(scene.repo, 'main')
    landed_shared = _shared_lines_at(scene.repo, 'main')
    assert landed_shared[1] == _MEMBER_A_LINE_2
    assert landed_shared[17] == _shared_lines_at(scene.repo, drift)[17]


# ─── (G) the re-verify of the rebased tip is red ────────────────────────────


async def test_red_reverify_derails_the_train_and_leaves_main_at_the_drift(
    tmp_path: Path,
) -> None:
    scene = await _stacked_train_scene(tmp_path)
    train = _train(scene)
    verify = _VerifyDouble(scene.repo, mover=_disjoint_drift, fail_on=(2,))
    store = _RecordingEventStore()

    outcome = await _run_train(scene, train, verify, store)

    assert outcome.status == 'blocked', outcome
    assert outcome.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), outcome.reason
    assert outcome.failure_category == 'pytest'
    assert _main_sha(scene.repo) == verify.drifts[0]
    train.mark_member_done.assert_not_awaited()
    assert len(_events(store, 'train_derailed')) == 1
    assert _attempts(store, 'verify_failed', train.request.train_id)


# ─── (I) main never stops moving ─────────────────────────────────────────────


async def test_perpetually_moving_main_exhausts_the_cas_budget_without_landing(
    tmp_path: Path,
) -> None:
    scene = await _stacked_train_scene(tmp_path)
    train = _train(scene)
    verify = _VerifyDouble(scene.repo, mover=_disjoint_drift, move_on=range(1, 1000))
    store = _RecordingEventStore()

    outcome = await _run_train(scene, train, verify, store)

    assert outcome.status == 'blocked', outcome
    assert len(verify.calls) == MergeLane.MAX_CAS_RETRIES + 1
    assert _attempts(store, 'cas_exhausted', train.request.train_id)
    [derailed] = _events(store, 'train_derailed')
    assert derailed['derail_reason'] == _CAS_DERAIL_REASON
    assert _main_sha(scene.repo) == verify.drifts[-1]
    assert not {'a.txt', 'b.txt', 'c.txt'} & _files_at(scene.repo, 'main')
    train.mark_member_done.assert_not_awaited()


# ─── (H) no coalesced member is stranded, on either arm ─────────────────────


async def test_coalesced_train_landing_after_a_moved_main_flips_every_member(
    tmp_path: Path,
) -> None:
    scene = await _stacked_train_scene(tmp_path)
    train = _train(scene, train_id='coalesce-5070-land')
    verify = _VerifyDouble(scene.repo, mover=_disjoint_drift)

    outcome = await _run_train(scene, train, verify, _RecordingEventStore())

    assert outcome.status == 'done', outcome
    landed = _main_sha(scene.repo)
    assert train.mark_member_done.await_args_list == _every_member_flipped(landed)
    train.redrive_member.assert_not_awaited()


async def test_coalesced_train_derailed_by_its_reverify_redrives_every_member_unstacked(
    tmp_path: Path,
) -> None:
    scene = await _stacked_train_scene(tmp_path)
    train = _train(scene, train_id='coalesce-5070-derail')
    verify = _VerifyDouble(scene.repo, mover=_disjoint_drift, fail_on=(2,))

    outcome = await _run_train(scene, train, verify, _RecordingEventStore())

    assert outcome.status == 'blocked', outcome
    assert outcome.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), outcome.reason
    train.mark_member_done.assert_not_awaited()
    assert train.redrive_member.await_args_list == [
        call(member, False, None) for member in _MEMBERS
    ]
    for member in _MEMBERS:
        subjects = _git(scene.repo, 'log', '--format=%s', f'main..task/{member}').splitlines()
        assert subjects == [f'{member}: own work'], (member, subjects)
