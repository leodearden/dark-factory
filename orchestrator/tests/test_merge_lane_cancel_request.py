"""``MergeLane.cancel_request``: an operator's cancel of a live request by id (task 5248).

An operator holding only a request id from ``get_merge_queue``/``merge_status``
must be able to cancel it whether it is queued, verifying or re-verifying a
rebased tree, with no caller blocked on its future; a request mid-CAS, a train,
or a single being stacked into one is refused.
Every scene runs a real lane over a tmp repo, and reads the outcome back off
the lane's public census, the request's own future, and main.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeVerifier,
    hangs_until,
    lane_scene_config,
    lane_state,
    make_lane,
    running_lane,
)
from _orch_helpers import wait_responsive
from escalation.queue import EscalationQueue
from escalation.server import create_server

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import AdvanceOutcome, GitOps
from orchestrator.merge_lane import (
    GroupMergeRequest,
    InFlightMergeRegistry,
    MergeCancelRefusal,
    MergeRequest,
    OperatorMergeCancel,
    QueuedBranch,
    WaiterRecord,
)
from orchestrator.merge_lane.queue_store import MergeQueueStore
from orchestrator.merge_lane.types import TrainCallbacks
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

_SCENE_TIMEOUT = 30.0


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _request(
    task_id: str, worktree: Path, config: OrchestratorConfig, **flags: bool,
) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        **flags,
    )


async def _branch(git_ops: GitOps, task_id: str, filename: str | None = None) -> Path:
    worktree = (await git_ops.create_worktree(task_id)).path
    (worktree / (filename or f'{task_id}.py')).write_text(f'{task_id} = 1\n')
    await git_ops.commit(worktree, f'Add {task_id}')
    return worktree


def _coalescing_lane(git_ops: GitOps, queue: asyncio.Queue[Any], **kwargs: Any) -> Any:
    return make_lane(git_ops, queue, train_callback_factory=lambda _train_id: (
        TrainCallbacks(
            status_check=AsyncMock(return_value={}), mark_member_done=AsyncMock(),
            redrive_member=AsyncMock(),
        )
    ), **kwargs)


async def _merge_cancel(server, request_id: str) -> dict[str, Any]:
    tool = await server.get_tool('merge_cancel')
    return await tool.fn(request_id=request_id)


def _main_moved_under_it(advances: list[Any]) -> Any:
    async def _advance(*args: Any, **kwargs: Any) -> AdvanceOutcome:
        advances.append(args)
        return AdvanceOutcome(
            'rebased_pending_reverify',
            advanced_sha='ab' * 20, rebased_from='ba' * 20, rebased_onto='cd' * 20,
        )

    return _advance


def _held_reverify(reverifying: asyncio.Event, release: asyncio.Event) -> Any:
    async def _reverify(*args: Any, **kwargs: Any) -> VerifyResult:
        reverifying.set()
        await release.wait()
        return VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='',
        )

    return _reverify


async def _until(predicate: Any, what: str) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(_poll(), _SCENE_TIMEOUT)
    except TimeoutError:
        raise AssertionError(f'timed out waiting for {what}') from None


def _operator_cancel_on(request: MergeRequest) -> OperatorMergeCancel | None:
    assert request.result.cancelled(), 'the request future must be cancelled'
    try:
        request.result.result()
    except asyncio.CancelledError as cancelled:
        return OperatorMergeCancel.carried_by(cancelled)
    raise AssertionError('a cancelled future must raise CancelledError')


class _CancelRecordingVerifier(FakeVerifier):
    """A ``FakeVerifier`` that also records each verify torn down mid-run."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.torn_down: list[str | None] = []

    async def run_scoped(self, worktree: Path, config: Any, module_configs: list[Any],
                         task_files: list[str] | None = None, **options: Any) -> VerifyResult:
        try:
            return await super().run_scoped(
                worktree, config, module_configs, task_files, **options,
            )
        except asyncio.CancelledError:
            self.torn_down.append(options.get('task_id'))
            raise


class TestCancellable:
    async def test_a_queued_request_is_cancelled_and_never_merged(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        verifier = FakeVerifier()
        lane = make_lane(git_ops, queue, verifier=verifier)
        cancelled = _request('cq', await _branch(git_ops, 'cq'), config)
        survivor = _request('sv', await _branch(git_ops, 'sv'), config)
        await queue.put(cancelled)

        verdict = lane.cancel_request(cancelled.request_id)

        assert verdict.cancelled
        assert (verdict.task_id, verdict.branch) == ('cq', 'cq')
        assert _operator_cancel_on(cancelled) == OperatorMergeCancel(cancelled.request_id)
        assert lane_state(lane, cancelled.request_id) is None
        async with running_lane(lane) as run:
            await queue.put(survivor)
            assert (await run.outcome(survivor)).status == 'done'
        assert verifier.verified == ['sv']
        assert not (Path(config.project_root) / 'cq.py').exists()

    async def test_a_verifying_request_is_cancelled_and_its_verify_torn_down(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        verifier = _CancelRecordingVerifier(default=hangs_until(asyncio.Event()))
        lane = make_lane(git_ops, queue, verifier=verifier)
        request = _request('cv', await _branch(git_ops, 'cv'), config)
        main_before = await git_ops.get_main_sha()

        async with running_lane(lane):
            await queue.put(request)
            await asyncio.wait_for(verifier.await_entry(), _SCENE_TIMEOUT)
            assert lane_state(lane, request.request_id) == 'verifying'

            verdict = lane.cancel_request(request.request_id)

            assert verdict.cancelled
            await _until(lambda: verifier.torn_down == ['cv'], 'the verify teardown')
            await _until(
                lambda: not lane.snapshot()['owned_merge_worktrees'],
                'the merge worktree reap',
            )
        assert _operator_cancel_on(request) == OperatorMergeCancel(request.request_id)
        assert lane_state(lane, request.request_id) is None
        assert await git_ops.get_main_sha() == main_before


    async def test_a_request_cancelled_while_its_rebased_tree_reverifies_is_dropped(
        self, git_ops: GitOps, config: OrchestratorConfig, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = make_lane(git_ops, queue)
        request = _request('gr', await _branch(git_ops, 'gr'), config)
        main_before = await git_ops.get_main_sha()
        advances: list[Any] = []
        reverifying, release = asyncio.Event(), asyncio.Event()
        monkeypatch.setattr(git_ops, 'advance_main', _main_moved_under_it(advances))
        monkeypatch.setattr(
            'orchestrator.merge_lane.worker.run_scoped_verification',
            _held_reverify(reverifying, release),
        )
        async with running_lane(lane):
            await queue.put(request)
            await wait_responsive(
                reverifying.wait(), timeout=_SCENE_TIMEOUT, label='the gate re-verify to start holding'
            )
            assert lane_state(lane, request.request_id) == 'gate_reverify'

            verdict = lane.cancel_request(request.request_id)
            release.set()

            assert verdict.cancelled
            await _until(
                lambda: lane_state(lane, request.request_id) is None, 'the drop',
            )
        assert _operator_cancel_on(request) == OperatorMergeCancel(request.request_id)
        assert len(advances) == 1, 'a cancelled request must not CAS again'
        assert await git_ops.get_main_sha() == main_before

    async def test_a_gate_reverify_cancel_leaves_the_merge_worktree_to_the_lane(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = make_lane(git_ops, queue)
        request = _request('gk', await _branch(git_ops, 'gk'), config)
        reverifying, release = asyncio.Event(), asyncio.Event()
        monkeypatch.setattr(git_ops, 'advance_main', _main_moved_under_it([]))
        monkeypatch.setattr(
            'orchestrator.merge_lane.worker.run_scoped_verification',
            _held_reverify(reverifying, release),
        )
        server = create_server(
            EscalationQueue(tmp_path / 'esc'),
            orch_config=config,
            harness=types.SimpleNamespace(
                _merge_worker=lane, _merge_store=MergeQueueStore(tmp_path / 'mq.json'),
                git_ops=git_ops,
            ),
        )
        async with running_lane(lane):
            await queue.put(request)
            await wait_responsive(
                reverifying.wait(), timeout=_SCENE_TIMEOUT, label='the gate re-verify to start holding'
            )
            [entry] = [
                e for e in lane.snapshot()['entries']
                if e['request_id'] == request.request_id
            ]
            tree = Path(entry['worktree'])

            response = await _merge_cancel(server, request.request_id)

            assert response['cancelled'] is True
            assert tree.exists(), 'the re-verify still runs in this tree'
            release.set()
            await _until(lambda: lane_state(lane, request.request_id) is None, 'the drop')
            await _until(lambda: not tree.exists(), 'the lane disposing of its tree')


class TestRefused:
    async def test_a_landing_request_is_refused_and_still_lands(
        self, git_ops: GitOps, config: OrchestratorConfig, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = make_lane(git_ops, queue)
        request = _request('fw', await _branch(git_ops, 'fw'), config)
        advancing, release = asyncio.Event(), asyncio.Event()
        advance_main = git_ops.advance_main

        async def _held_advance(*args: Any, **kwargs: Any) -> Any:
            advancing.set()
            await release.wait()
            return await advance_main(*args, **kwargs)

        monkeypatch.setattr(git_ops, 'advance_main', _held_advance)
        async with running_lane(lane) as run:
            await queue.put(request)
            await wait_responsive(
                advancing.wait(), timeout=_SCENE_TIMEOUT, label='the chain walk to reach its CAS advance'
            )
            assert lane_state(lane, request.request_id) == 'finalizing'

            verdict = lane.cancel_request(request.request_id)

            assert verdict.refusal is MergeCancelRefusal.FINALIZE_WINDOW
            assert not request.result.done()
            release.set()
            assert (await run.outcome(request)).status == 'done'

    async def test_a_train_is_refused(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = make_lane(git_ops, queue)
        tip = await _branch(git_ops, 'tip')
        train = GroupMergeRequest(
            task_id='tip',
            branch=QueuedBranch.parse('tip', config.git.branch_prefix),
            worktree=tip,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=asyncio.get_running_loop().create_future(),
            train_id='train-1',
            member_task_ids=['root', 'tip'],
            tip_branch=QueuedBranch.parse('tip', config.git.branch_prefix),
            tip_task_id='tip',
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )
        await queue.put(train)

        verdict = lane.cancel_request(train.request_id)

        assert verdict.refusal is MergeCancelRefusal.TRAIN_CANCEL_UNSUPPORTED
        assert not train.result.done()

    async def test_an_unknown_or_decided_request_is_not_live(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = make_lane(git_ops, queue)
        request = _request('dc', await _branch(git_ops, 'dc'), config)
        await queue.put(request)
        assert lane.cancel_request(request.request_id).cancelled

        again = lane.cancel_request(request.request_id)
        unknown = lane.cancel_request('mr-unknown')

        assert again.refusal is MergeCancelRefusal.NOT_LIVE
        assert unknown.refusal is MergeCancelRefusal.NOT_LIVE
        assert (unknown.task_id, unknown.branch) == (None, None)

    async def test_a_single_being_stacked_into_a_train_is_refused(
        self, git_ops: GitOps, git_repo: Path, git_config: GitConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = lane_scene_config(git_repo, git_config, merge_train_coalesce_enabled=True)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = make_lane(git_ops, queue, train_callback_factory=lambda _train_id: (
            TrainCallbacks(
                status_check=AsyncMock(return_value={}), mark_member_done=AsyncMock(),
                redrive_member=AsyncMock(),
            )
        ))
        first, second = [
            _request(name, await _branch(git_ops, name), config, awaited_by_workflow=True)
            for name in ('st-a', 'st-b')
        ]
        stacking, release = asyncio.Event(), asyncio.Event()
        line_ranges = git_ops.get_changed_line_ranges

        async def _held_line_ranges(task_id: str) -> Any:
            stacking.set()
            await release.wait()
            return await line_ranges(task_id)

        monkeypatch.setattr(git_ops, 'get_changed_line_ranges', _held_line_ranges)
        for request in (first, second):
            queue.put_nowait(request)
        async with running_lane(lane) as run:
            await wait_responsive(
                stacking.wait(), timeout=_SCENE_TIMEOUT, label='the coalesce stack to start'
            )

            verdict = lane.cancel_request(first.request_id)
            release.set()

            assert verdict.refusal is MergeCancelRefusal.TRAIN_CANCEL_UNSUPPORTED
            assert (await run.outcome(first)).status == 'superseded'


class TestAttachedWaiters:
    """An operator's cancel reaches every waiter attached to the request's slot."""

    def _slot_with_a_workflow_peer(self) -> tuple[Any, Any, InFlightMergeRegistry]:
        loop = asyncio.get_running_loop()
        primary, peer = loop.create_future(), loop.create_future()
        registry = InFlightMergeRegistry()
        registry.acquire('br', 't1', primary, request_id='mr-primary')
        registry.attach('br', WaiterRecord(request_id='mr-peer', future=peer, source='workflow'))
        return primary, peer, registry

    def _carried(self, future: Any) -> OperatorMergeCancel | None:
        with pytest.raises(asyncio.CancelledError) as cancelled:
            future.result()
        return OperatorMergeCancel.carried_by(cancelled.value)

    async def test_the_mirror_forwards_the_operator_cancel(self) -> None:
        primary, peer, _registry = self._slot_with_a_workflow_peer()

        primary.cancel(OperatorMergeCancel('mr-primary'))
        await asyncio.sleep(0)

        assert self._carried(peer) == OperatorMergeCancel('mr-primary')

    async def test_releasing_the_slot_forwards_the_operator_cancel(self) -> None:
        primary, peer, registry = self._slot_with_a_workflow_peer()

        primary.cancel(OperatorMergeCancel('mr-primary'))
        registry.release('br', detach_waiters=True)

        assert self._carried(peer) == OperatorMergeCancel('mr-primary')


class TestCoalescingReleasesItsSingles:
    """A single the coalescing pass did not absorb is cancellable again after it."""

    async def test_a_single_left_unselected_is_cancellable(
        self, git_ops: GitOps, git_repo: Path, git_config: GitConfig,
    ) -> None:
        config = lane_scene_config(git_repo, git_config, merge_train_coalesce_enabled=True)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        verifier = FakeVerifier(default=hangs_until(asyncio.Event()))
        lane = _coalescing_lane(git_ops, queue, verifier=verifier)
        first, second = [
            _request(name, await _branch(git_ops, name, 'same.py'), config,
                     awaited_by_workflow=True)
            for name in ('un-a', 'un-b')
        ]
        for request in (first, second):
            queue.put_nowait(request)
        async with running_lane(lane):
            await asyncio.wait_for(verifier.await_entry(), _SCENE_TIMEOUT)

            verdict = lane.cancel_request(second.request_id)

        assert verdict.cancelled, verdict

    async def test_a_single_is_cancellable_after_a_pass_that_raised(
        self, git_ops: GitOps, git_repo: Path, git_config: GitConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Driven by the pass itself: an exception out of it ends the merger, so a
        running lane cannot host this scene."""
        config = lane_scene_config(git_repo, git_config, merge_train_coalesce_enabled=True)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = _coalescing_lane(git_ops, queue)
        first, second = [
            _request(name, await _branch(git_ops, name), config, awaited_by_workflow=True)
            for name in ('rs-a', 'rs-b')
        ]
        for request in (first, second):
            queue.put_nowait(request)
        monkeypatch.setattr(
            git_ops, 'stack_train_branches', AsyncMock(side_effect=RuntimeError('stack')),
        )

        with pytest.raises(RuntimeError):
            await lane._maybe_coalesce_waiting_singles()

        assert lane.cancel_request(first.request_id).cancelled
