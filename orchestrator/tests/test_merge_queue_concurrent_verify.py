"""Concurrent-verify tests for γ (task 1735).

This module covers γ-specific tests that require multi-host verify infrastructure.
Tests are grouped per plan step:

  pre-1  SCAFFOLD — shared fixtures/helpers (git repo, gated runner, fake remote)
  step-1 RED      — _run_post_merge_verify(runner=<fake>) + runner=None byte-identical
  step-3 RED      — _inflight/_redispatch/_n_failed/_remerge_occurred attrs + InflightEntry
  step-5 RED      — _run_inflight_verify happy paths (LOCAL + REMOTE lease)
  step-7 RED      — _run_inflight_verify abort-poll (abandon + halt)
  step-9 RED      — _run_inflight_verify RUNNER_UNAVAILABLE sentinel
  step-11 RED     — _finalize_inflight PASS path (CAS advance, lease release)
  step-13 RED     — _finalize_inflight non-pass paths
  step-15 RED     — single-host serial byte-identical via restructured loop
  step-17 RED     — OVERLAP SIGNAL (two hosts, both verifies enter before either released)
  step-19 RED     — CHAIN-INVALIDATION UNDER OVERLAP
  step-21 RED     — operator-halt aborts all in-flight + RUNNER_UNAVAILABLE quarantine
  step-23 RED     — stop() drains inflight + snapshot() surfaces inflight entries

NOTE: Individual test classes are added in their respective RED steps.
This file starts with shared scaffolding only (pre-1).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest, MergeOutcome, SpeculativeMergeWorker
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import HostAllocator


# ---------------------------------------------------------------------------
# (a) Temp-repo GitOps + branch-with-file builders
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    """Single-host (no verify_runners) OrchestratorConfig."""
    return OrchestratorConfig(project_root=git_repo, git=git_config)


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    pre_rebased: bool = False,
    lane: str = 'normal',
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop."""
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        from _orch_helpers import make_placeholder_future
        future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane=lane,
    )


# ---------------------------------------------------------------------------
# (d) OrchestratorConfig builders: single-host vs two-host
# ---------------------------------------------------------------------------


def _make_config_no_runners(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    """Single-host OrchestratorConfig (no verify_runners)."""
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_config_with_runner(
    git_repo: Path,
    git_config: GitConfig,
    runner_name: str = 'laptop',
) -> OrchestratorConfig:
    """Two-host OrchestratorConfig with one enabled verify_runner."""
    runner_cfg = VerifyRunnerConfig(
        name=runner_name,
        ssh_host='h.local',
        git_remote='origin',
        enabled=True,
    )
    return OrchestratorConfig(
        project_root=git_repo,
        git=git_config,
        verify_runners=[runner_cfg],
    )


# ---------------------------------------------------------------------------
# (b) _gated_runner factory: slow fake verify gated on asyncio.Event
# ---------------------------------------------------------------------------


def _mock_verify_result(passed: bool) -> VerifyResult:
    """Return a VerifyResult with the given pass/fail status."""
    return VerifyResult(
        passed=passed,
        test_output='ok' if passed else 'FAILED',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category='' if passed else 'test_failure',
    )


def _mock_verify_pass() -> AsyncMock:
    """Return a mock that makes run_scoped_verification always pass."""
    return AsyncMock(return_value=MagicMock(passed=True, summary=''))


def _gated_runner(
    gate_release: asyncio.Event,
    gate_entered: asyncio.Event | None = None,
    *,
    passed: bool = True,
    name: str = 'gated',
) -> MagicMock:
    """Return a fake runner whose run_merge_verify blocks until gate_release is set.

    *gate_entered* (optional): set when the first call starts, so tests can
    await both enters before releasing.  Subsequent calls pass immediately
    (gate is already set after first call).

    This is a fake RemoteRunner shaped object: name, is_local=False,
    run_merge_verify/cancel_verify/probe_clean as AsyncMocks.
    """
    _first_blocked = [False]

    async def _side_effect(*args: Any, **kwargs: Any) -> VerifyResult:
        if not _first_blocked[0]:
            _first_blocked[0] = True
            if gate_entered is not None:
                gate_entered.set()
            await gate_release.wait()
        return _mock_verify_result(passed)

    runner = MagicMock()
    runner.name = name
    runner.is_local = False
    runner.run_merge_verify = AsyncMock(side_effect=_side_effect)
    runner.cancel_verify = AsyncMock(return_value=0)
    runner.probe_clean = AsyncMock(return_value=True)
    return runner


# ---------------------------------------------------------------------------
# (c) Fake RemoteRunner + two-host HostAllocator injection
# ---------------------------------------------------------------------------


def _make_fake_remote(name: str = 'laptop') -> MagicMock:
    """Build a fake RemoteRunner (MagicMock) with async cancel/probe/verify."""
    fake = MagicMock()
    fake.name = name
    fake.is_local = False
    fake.run_merge_verify = AsyncMock(return_value=_mock_verify_result(True))
    fake.cancel_verify = AsyncMock(return_value=0)  # 0 = clean cancel
    fake.probe_clean = AsyncMock(return_value=True)
    return fake


def _inject_two_host_allocator(
    worker: SpeculativeMergeWorker,
    fake_remote: Any,
) -> HostAllocator:
    """Inject a two-host HostAllocator (local + fake_remote) onto a worker.

    Returns the allocator so callers can introspect slot state or verify calls.
    """
    allocator = HostAllocator([fake_remote], quarantine=worker._runner_quarantine)
    worker._host_allocator = allocator
    return allocator


# ---------------------------------------------------------------------------
# step-1 RED: _run_post_merge_verify(runner=<fake>) wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunPostMergeVerifyRunnerParam:
    """_run_post_merge_verify(runner=<fake>) dispatches on the injected runner;
    runner=None (default) stays LOCAL-ONLY byte-identical.

    RED until step-2 GREEN adds the runner= keyword arg.
    """

    def _make_git_ops_mock(self) -> MagicMock:
        from unittest.mock import patch as _patch
        mock = MagicMock()
        mock.get_main_sha = AsyncMock(return_value='main-sha')
        mock.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)
        mock.cleanup_merge_worktree = AsyncMock()
        mock.create_throwaway_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
        return mock

    async def test_remote_runner_dispatched_when_runner_provided(self, tmp_path: Path) -> None:
        """runner=<fake remote> → pool is [runner], fake.run_merge_verify called."""
        from orchestrator.merge_queue import _run_post_merge_verify

        fake_remote = _make_fake_remote('laptop')
        config = OrchestratorConfig(git=GitConfig(main_branch='main'))
        req = _make_request('t1', 'task/t1', tmp_path, config)
        git_ops = self._make_git_ops_mock()

        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123',
            runner=fake_remote,  # RED: this param doesn't exist yet
        )

        assert outcome is None  # verify passed
        fake_remote.run_merge_verify.assert_called_once()

    async def test_runner_none_uses_local_runner_byte_identical(self, tmp_path: Path) -> None:
        """runner=None (default) → LocalRunner pool, byte-identical to today."""
        from orchestrator.merge_queue import _run_post_merge_verify
        from unittest.mock import patch

        config = OrchestratorConfig(git=GitConfig(main_branch='main'))
        req = _make_request('t2', 'task/t2', tmp_path, config)
        git_ops = self._make_git_ops_mock()

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            new=AsyncMock(return_value=_mock_verify_result(True)),
        ):
            outcome = await _run_post_merge_verify(
                git_ops, req, tmp_path,
                timeouts={}, enospc_retries={},
                max_timeouts=2, max_enospc=1,
                merge_sha='abc123',
                runner=None,  # RED: this param doesn't exist yet
            )

        assert outcome is None

    async def test_merge_verify_event_carries_runner_name_when_remote(self, tmp_path: Path) -> None:
        """merge_verify event runner field == injected runner's name, not 'local'."""
        from orchestrator.merge_queue import _run_post_merge_verify
        from orchestrator.event_store import EventStore

        fake_remote = _make_fake_remote('laptop')
        config = OrchestratorConfig(git=GitConfig(main_branch='main'))
        req = _make_request('t3', 'task/t3', tmp_path, config)
        git_ops = self._make_git_ops_mock()

        emitted: list[dict] = []

        class _FakeEventStore(EventStore):
            def __init__(self) -> None:
                object.__init__(self)

            def emit(self, event_type, *, task_id=None, phase=None, data=None, **kw):  # type: ignore[override]
                emitted.append({'event_type': event_type, 'data': data or {}})

        await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123',
            runner=fake_remote,   # RED: this param doesn't exist yet
            event_store=_FakeEventStore(),
        )

        merge_verify_events = [
            e for e in emitted
            if hasattr(e['event_type'], 'value') and e['event_type'].value == 'merge_verify'
        ]
        assert len(merge_verify_events) >= 1
        assert merge_verify_events[0]['data']['runner'] == 'laptop'


# ---------------------------------------------------------------------------
# step-3 RED: InflightEntry dataclass + new instance attrs
# ---------------------------------------------------------------------------


class TestInflightStateAttrs:
    """SpeculativeMergeWorker exposes _inflight/_redispatch/_n_failed/_remerge_occurred.
    InflightEntry dataclass is importable from merge_queue.

    RED until step-4 GREEN adds the dataclass and initialises the attrs.
    """

    def _make_worker(self, git_ops: GitOps) -> SpeculativeMergeWorker:
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        return SpeculativeMergeWorker(git_ops, q)

    def test_inflight_is_empty_deque(self, git_ops: GitOps) -> None:
        """_inflight starts as an empty collections.deque."""
        import collections
        worker = self._make_worker(git_ops)
        assert isinstance(worker._inflight, collections.deque)
        assert len(worker._inflight) == 0

    def test_redispatch_is_empty_deque(self, git_ops: GitOps) -> None:
        """_redispatch starts as an empty collections.deque."""
        import collections
        worker = self._make_worker(git_ops)
        assert isinstance(worker._redispatch, collections.deque)
        assert len(worker._redispatch) == 0

    def test_n_failed_is_false(self, git_ops: GitOps) -> None:
        """_n_failed starts False."""
        worker = self._make_worker(git_ops)
        assert worker._n_failed is False

    def test_remerge_occurred_is_false(self, git_ops: GitOps) -> None:
        """_remerge_occurred starts False."""
        worker = self._make_worker(git_ops)
        assert worker._remerge_occurred is False

    def test_inflight_entry_importable(self) -> None:
        """InflightEntry is importable from merge_queue."""
        from orchestrator.merge_queue import InflightEntry  # noqa: F401 (import check)

    def test_inflight_entry_has_required_fields(self) -> None:
        """InflightEntry has item/lease/verify_task/merge_wt/was_speculative/phase."""
        import dataclasses
        from orchestrator.merge_queue import InflightEntry
        field_names = {f.name for f in dataclasses.fields(InflightEntry)}
        assert 'item' in field_names
        assert 'lease' in field_names
        assert 'verify_task' in field_names
        assert 'merge_wt' in field_names
        assert 'was_speculative' in field_names
        assert 'phase' in field_names

    def test_inflight_entry_has_passthrough_outcome_field(self) -> None:
        """InflightEntry has passthrough_outcome for immediate-outcome passthrough entries."""
        import dataclasses
        from orchestrator.merge_queue import InflightEntry
        field_names = {f.name for f in dataclasses.fields(InflightEntry)}
        assert 'passthrough_outcome' in field_names


# ---------------------------------------------------------------------------
# step-5 RED: _run_inflight_verify happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunInflightVerifyHappyPath:
    """_run_inflight_verify(item, lease) happy paths: LOCAL lease and REMOTE lease.

    RED until step-6 GREEN adds the method.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        """Helper: create a branch, merge it to main, return (req, item)."""
        from orchestrator.merge_queue import SpeculativeItem
        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_event_loop()
        req = MergeRequest(
            task_id=branch,
            branch=branch,
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        base_sha = await git_ops.get_main_sha()
        item = SpeculativeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
            skip_verify=False,
        )
        return req, item

    async def test_local_lease_increments_attempt_count(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """LOCAL lease: _verify_attempt_count incremented by one."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(git_ops, config, 'inv-local-a', 'fa.py', 'a=1\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        fake_local = _make_fake_remote('local-fake')
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)

        count_before = worker._verify_attempt_count
        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            result = await worker._run_inflight_verify(item, lease)  # RED: method doesn't exist

        assert worker._verify_attempt_count == count_before + 1
        assert result.outcome is None  # pass

    async def test_local_lease_verify_phase_set_to_verifying(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """LOCAL lease: _verify_phase transitions to 'verifying' during the call."""
        from unittest.mock import patch
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(git_ops, config, 'inv-local-b', 'fb.py', 'b=2\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        fake_local = _make_fake_remote('local-fake2')
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)

        phases_seen: list[str] = []
        orig_run = worker._git_ops.get_main_sha  # noqa: F841

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            result = await worker._run_inflight_verify(item, lease)

        # After completion result.outcome is None (pass)
        assert result.outcome is None
        assert result.status is None or result.status == 'done'

    async def test_remote_lease_no_warm_swap(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REMOTE lease: _verify_attempt_count NOT incremented (no warm-swap),
        lease.runner.run_merge_verify called, outcome=None on pass.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(git_ops, config, 'inv-remote-a', 'fc.py', 'c=3\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        fake_remote = _make_fake_remote('laptop')
        lease = HostLease(name='laptop', runner=fake_remote, is_local=False)

        count_before = worker._verify_attempt_count
        result = await worker._run_inflight_verify(item, lease)  # RED: method doesn't exist

        # REMOTE path: warm-swap NOT run → attempt count unchanged
        assert worker._verify_attempt_count == count_before
        # runner.run_merge_verify called (remote dispatch path)
        fake_remote.run_merge_verify.assert_called_once()
        # outcome is None (pass)
        assert result.outcome is None

    async def test_run_inflight_verify_returns_merge_wt(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """_run_inflight_verify returns result with non-None merge_wt."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(git_ops, config, 'inv-wt', 'fd.py', 'd=4\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        fake_remote = _make_fake_remote('laptop2')
        lease = HostLease(name='laptop2', runner=fake_remote, is_local=False)

        result = await worker._run_inflight_verify(item, lease)

        assert result.merge_wt is not None


# ---------------------------------------------------------------------------
# step-7 RED: _run_inflight_verify abort-poll (abandon + operator halt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunInflightVerifyAbortPoll:
    """_run_inflight_verify abort-poll: abandon → DROPPED, halt → REQUEUED.

    RED until step-8 GREEN ports the abort-poll loop into _run_inflight_verify.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        """Create a merged SpeculativeItem on the given branch."""
        from orchestrator.merge_queue import SpeculativeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_event_loop()
        req = MergeRequest(
            task_id=branch,
            branch=branch,
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        base_sha = await git_ops.get_main_sha()
        item = SpeculativeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
            skip_verify=False,
        )
        return req, item

    async def test_abandon_mid_verify_returns_dropped(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Sole-waiter abandon mid-verify → inner task cancelled, merge_wt cleaned,
        returns status=DROPPED.

        RED (step-6): current code has no poll loop — verify runs to completion
        when gate is released, result.status is None.
        GREEN (step-8): poll loop detects abandon before gate released → DROPPED.
        """
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='slow-abandon')

        req, item = await self._make_merged_item(
            git_ops, config, 'abort-poll-a', 'pa.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02  # fast polling for tests
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='slow-abandon', runner=gated, is_local=False)

        # Start verify in background (gated runner blocks until gate_release)
        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))

        # Wait for the gated runner to enter
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

        # Trigger sole-waiter abandon
        req.result.cancel()

        # Give poll loop a chance to fire (step-6: will not detect — no loop)
        await asyncio.sleep(worker.VERIFY_ABANDON_POLL_SECS * 2)

        # Release gate so RED case doesn't hang
        gate_release.set()

        result = await verify_future

        # GREEN: poll loop detected abandon before gate → DROPPED
        # RED: verify ran to completion (gate released) → status=None
        assert result.status == 'DROPPED'  # RED: fails (None != 'DROPPED')

    async def test_abandon_mid_verify_cleans_merge_wt(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Abandon → merge_wt cleaned: result.merge_wt is None on DROPPED."""
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='slow-wt')

        req, item = await self._make_merged_item(
            git_ops, config, 'abort-poll-b', 'pb.py', 'y=2\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='slow-wt', runner=gated, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)
        req.result.cancel()
        await asyncio.sleep(worker.VERIFY_ABANDON_POLL_SECS * 2)
        gate_release.set()

        result = await verify_future

        # GREEN: wt cleaned by abort handler → result.merge_wt is None
        # RED: verify ran normally → result.merge_wt is not None
        assert result.merge_wt is None  # RED: fails (merge_wt is not None)

    async def test_operator_halt_mid_verify_returns_requeued(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Operator halt mid-verify → inner task cancelled, req re-queued, status=REQUEUED.

        RED (step-6): current code has no poll loop — verify completes normally.
        GREEN (step-8): poll loop detects halt, requeues req, returns REQUEUED.
        """
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='slow-halt')

        req, item = await self._make_merged_item(
            git_ops, config, 'abort-poll-c', 'pc.py', 'z=3\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='slow-halt', runner=gated, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

        # Trigger operator halt
        worker._operator_halt.set()

        await asyncio.sleep(worker.VERIFY_ABANDON_POLL_SECS * 2)
        # Release gate so RED case doesn't hang
        gate_release.set()

        result = await verify_future

        # GREEN: poll loop detected halt → REQUEUED
        # RED: verify ran to completion → status=None
        assert result.status == 'REQUEUED'  # RED: fails (None != 'REQUEUED')

    async def test_operator_halt_requeues_request_on_queue(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Operator halt → req is put back on worker._queue."""
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='slow-halt2')

        req, item = await self._make_merged_item(
            git_ops, config, 'abort-poll-d', 'pd.py', 'w=4\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='slow-halt2', runner=gated, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

        worker._operator_halt.set()
        await asyncio.sleep(worker.VERIFY_ABANDON_POLL_SECS * 2)
        gate_release.set()

        await verify_future

        # GREEN: req put_nowait onto _queue (which is worker._queue == q)
        # RED: _queue is empty
        assert not q.empty(), 'req should be back on _queue after halt'  # RED: fails


# ---------------------------------------------------------------------------
# step-9 RED: _run_inflight_verify with RUNNER_UNAVAILABLE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunInflightVerifyRunnerUnavailable:
    """_run_inflight_verify with a REMOTE lease whose run_merge_verify raises
    RunnerUnavailable → returns status=RUNNER_UNAVAILABLE, merge_wt NOT cleaned.

    RED until step-10 GREEN catches RunnerUnavailable separately.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        from orchestrator.merge_queue import SpeculativeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_event_loop()
        req = MergeRequest(
            task_id=branch,
            branch=branch,
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        base_sha = await git_ops.get_main_sha()
        item = SpeculativeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
            skip_verify=False,
        )
        return req, item

    async def test_runner_unavailable_returns_sentinel(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REMOTE run_merge_verify raises RunnerUnavailable → status=RUNNER_UNAVAILABLE,
        no exception escapes.

        RED (step-9): caught by generic except Exception → status=None, merge_wt=None.
        GREEN (step-10): caught by specific except RunnerUnavailable → status set.
        """
        from orchestrator.verify_runner import HostLease, RunnerUnavailable

        dead_remote = _make_fake_remote('dead-laptop')
        dead_remote.run_merge_verify = AsyncMock(
            side_effect=RunnerUnavailable('connection refused')
        )

        req, item = await self._make_merged_item(
            git_ops, config, 'runner-unavail-a', 'ra.py', 'a=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='dead-laptop', runner=dead_remote, is_local=False)

        # Should not raise — RunnerUnavailable is caught internally
        result = await worker._run_inflight_verify(item, lease)

        # GREEN: status sentinel set, merge_wt preserved for re-dispatch
        # RED: status=None (generic except catches RunnerUnavailable)
        assert result.status == 'RUNNER_UNAVAILABLE'  # RED: fails (None != ...)

    async def test_runner_unavailable_merge_wt_not_cleaned(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REMOTE RunnerUnavailable → merge_wt NOT cleaned (preserved for re-dispatch)."""
        from orchestrator.verify_runner import HostLease, RunnerUnavailable

        dead_remote = _make_fake_remote('dead-laptop2')
        dead_remote.run_merge_verify = AsyncMock(
            side_effect=RunnerUnavailable('timeout')
        )

        req, item = await self._make_merged_item(
            git_ops, config, 'runner-unavail-b', 'rb.py', 'b=2\n',
        )
        merge_wt_path = item.merge_wt
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='dead-laptop2', runner=dead_remote, is_local=False)

        result = await worker._run_inflight_verify(item, lease)

        # GREEN: merge_wt intact — item re-dispatched on another host with wt intact
        # RED: merge_wt=None (generic exception handler cleans it)
        assert result.merge_wt is not None  # RED: fails (merge_wt is None)
        assert result.merge_wt == merge_wt_path


# ---------------------------------------------------------------------------
# step-11 RED: _finalize_inflight PASS path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeInflightPass:
    """_finalize_inflight(entry) PASS path: CAS advance, lease release, _n_failed=False.

    RED until step-12 GREEN adds _finalize_inflight.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        from orchestrator.merge_queue import SpeculativeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_event_loop()
        req = MergeRequest(
            task_id=branch,
            branch=branch,
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        base_sha = await git_ops.get_main_sha()
        item = SpeculativeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
            skip_verify=False,
        )
        return req, item

    def _make_pass_entry(self, item, lease, was_speculative: bool = False):
        """Build an InflightEntry representing a completed, passing verify."""
        from orchestrator.merge_queue import InflightEntry
        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=None,
            merge_wt=item.merge_wt,
            was_speculative=was_speculative,
            phase='verifying',
            passthrough_outcome=None,
            verify_result=None,  # None = pass
            status=None,
        )

    def _make_mock_allocator(self):
        """Return a MagicMock with async release/cancel_and_release."""
        alloc = MagicMock()
        alloc.release = AsyncMock()
        alloc.cancel_and_release = AsyncMock()
        return alloc

    async def test_finalize_pass_returns_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry → _finalize_inflight returns True (main advanced).

        RED: method doesn't exist yet.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-a', 'fa.py', 'a=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            advanced = await worker._finalize_inflight(entry)  # RED: method missing

        assert advanced is True

    async def test_finalize_pass_resolves_req_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry → req.result resolved with outcome.status == 'done'."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-b', 'fb.py', 'b=2\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            await worker._finalize_inflight(entry)

        assert req.result.done()
        assert req.result.result().status == 'done'

    async def test_finalize_pass_releases_lease(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry → allocator.release called with the entry's lease."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-c', 'fc.py', 'c=3\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        mock_alloc = self._make_mock_allocator()
        worker._host_allocator = mock_alloc
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            await worker._finalize_inflight(entry)

        mock_alloc.release.assert_called_once_with(lease)

    async def test_finalize_pass_sets_n_failed_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry → _n_failed = False after finalize."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-d', 'fd.py', 'd=4\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._n_failed = True  # set to True to check it's reset to False

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            await worker._finalize_inflight(entry)

        assert worker._n_failed is False

    async def test_finalize_pass_releases_speculation_slot_iff_speculative(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """was_speculative=True → _speculation_slot.release() called."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-e', 'fe.py', 'e=5\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        # Acquire one slot to simulate that a speculative item is in-flight
        await worker._speculation_slot.acquire()
        slot_value_before = worker._speculation_slot._value

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease, was_speculative=True)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            await worker._finalize_inflight(entry)

        # Slot should be released back
        assert worker._speculation_slot._value == slot_value_before + 1

    async def test_finalize_pass_does_not_release_slot_if_not_speculative(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """was_speculative=False → _speculation_slot NOT released."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-f', 'ff.py', 'f=6\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        slot_value_before = worker._speculation_slot._value

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease, was_speculative=False)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            await worker._finalize_inflight(entry)

        # Slot value unchanged (no release)
        assert worker._speculation_slot._value == slot_value_before


# ---------------------------------------------------------------------------
# step-13 RED: _finalize_inflight non-pass paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeInflightNonPass:
    """_finalize_inflight non-pass paths: FAIL, PASSTHROUGH, DROPPED, REQUEUED.

    RED until step-14 GREEN extends _finalize_inflight.
    """

    # ------------------------------------------------------------------
    # Shared helpers (duplicated from TestFinalizeInflightPass for isolation)
    # ------------------------------------------------------------------

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        from orchestrator.merge_queue import SpeculativeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_event_loop()
        req = MergeRequest(
            task_id=branch,
            branch=branch,
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        base_sha = await git_ops.get_main_sha()
        item = SpeculativeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
            skip_verify=False,
        )
        return req, item

    def _make_passthrough_item(
        self,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
        outcome: MergeOutcome,
        already_delivered: bool = False,
    ):
        """Build a SpeculativeItem with immediate_outcome (no real merge)."""
        from orchestrator.merge_queue import SpeculativeItem

        loop = asyncio.get_event_loop()
        req = MergeRequest(
            task_id='pt-task',
            branch='pt-branch',
            worktree=git_repo / 'pt-wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        item = SpeculativeItem(
            request=req,
            merge_result=None,
            merge_wt=None,
            base_sha='deadbeef' * 5,
            speculative=False,
            skip_verify=False,
            immediate_outcome=outcome,
            already_delivered=already_delivered,
        )
        return req, item

    def _make_mock_allocator(self):
        """Return a MagicMock with async release/cancel_and_release."""
        alloc = MagicMock()
        alloc.release = AsyncMock()
        alloc.cancel_and_release = AsyncMock()
        return alloc

    def _make_fail_entry(self, item, lease, fail_outcome, merge_wt_val):
        """Build an InflightEntry with a verify_task that returns a FAIL result."""
        from orchestrator.merge_queue import InflightEntry, InflightVerifyResult

        async def _fake_fail_verify():
            return InflightVerifyResult(
                outcome=fail_outcome,
                merge_wt=merge_wt_val,
                status=None,
            )

        verify_task = asyncio.ensure_future(_fake_fail_verify())
        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=verify_task,
            merge_wt=merge_wt_val,
            was_speculative=False,
            phase='verifying',
        )

    def _make_sentinel_entry(self, item, lease, sentinel_status, merge_wt_val=None):
        """Build an InflightEntry with a verify_task returning a DROPPED/REQUEUED sentinel."""
        from orchestrator.merge_queue import InflightEntry, InflightVerifyResult

        async def _fake_sentinel_verify():
            return InflightVerifyResult(
                outcome=None,
                merge_wt=merge_wt_val,
                status=sentinel_status,
            )

        verify_task = asyncio.ensure_future(_fake_sentinel_verify())
        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=verify_task,
            merge_wt=merge_wt_val,
            was_speculative=False,
            phase='verifying',
        )

    # ------------------------------------------------------------------
    # (a) FAIL path
    # ------------------------------------------------------------------

    async def test_finalize_fail_returns_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """FAIL verify outcome → _finalize_inflight returns False.

        RED: current step-12 code ignores verify_task and runs CAS → returns True.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fail-a', 'fa.py', 'a=1\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        fail_outcome = MergeOutcome('blocked', reason='test verify fail')
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        result = await worker._finalize_inflight(entry)
        assert result is False  # RED: current code returns True

    async def test_finalize_fail_resolves_req_with_fail_outcome(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """FAIL outcome → req.result resolved with the fail MergeOutcome.

        RED: current code advances main and resolves with 'done'.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fail-b', 'fb.py', 'b=2\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        fail_outcome = MergeOutcome('blocked', reason='verify error')
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        await worker._finalize_inflight(entry)
        assert req.result.done()
        assert req.result.result().status == 'blocked'

    async def test_finalize_fail_releases_lease(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """FAIL outcome → allocator.release called with the lease.

        RED: current code may call release (in finally), but wrong behavior before that.
        This test verifies lease is released specifically for the fail path.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fail-c', 'fc.py', 'c=3\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        mock_alloc = self._make_mock_allocator()
        worker._host_allocator = mock_alloc
        worker._register_owned_merge_worktree(item.merge_wt)

        fail_outcome = MergeOutcome('blocked', reason='fail')
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        try:
            await worker._finalize_inflight(entry)
        except Exception:
            pass  # may raise in RED state; we care about release
        mock_alloc.release.assert_called_once_with(lease)

    async def test_finalize_fail_sets_n_failed_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """FAIL outcome → self._n_failed = True.

        RED: current code sets _n_failed = not advanced = False (wrong).
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fail-d', 'fd.py', 'd=4\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._n_failed = False  # start at False to detect the change
        worker._register_owned_merge_worktree(item.merge_wt)

        fail_outcome = MergeOutcome('blocked', reason='fail')
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        try:
            await worker._finalize_inflight(entry)
        except Exception:
            pass
        assert worker._n_failed is True  # RED: current code sets False

    # ------------------------------------------------------------------
    # (b) PASSTHROUGH path (immediate_outcome / conflict / already_merged)
    # ------------------------------------------------------------------

    async def test_finalize_passthrough_conflict_resolves_req(
        self, git_repo: Path, git_config: GitConfig, config: OrchestratorConfig,
    ) -> None:
        """PASSTHROUGH (conflict) → req.result resolved with the passthrough outcome.

        RED: current code asserts merge_wt is not None → AssertionError.
        """
        from orchestrator.merge_queue import InflightEntry

        conflict_outcome = MergeOutcome('conflict', reason='merge conflict')
        req, item = self._make_passthrough_item(
            git_repo, git_config, config, conflict_outcome, already_delivered=False,
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            GitOps(git_config, git_repo), q,
        )
        worker._host_allocator = self._make_mock_allocator()

        entry = InflightEntry(
            item=item,
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=False,
            phase='passthrough',
            passthrough_outcome=conflict_outcome,
        )
        await worker._finalize_inflight(entry)  # RED: raises AssertionError
        assert req.result.done()
        assert req.result.result().status == 'conflict'

    async def test_finalize_passthrough_n_failed_true_for_conflict(
        self, git_repo: Path, git_config: GitConfig, config: OrchestratorConfig,
    ) -> None:
        """PASSTHROUGH conflict → _n_failed = True (conflict is not 'done'/'already_merged').

        RED: current code raises AssertionError before reaching _n_failed.
        """
        from orchestrator.merge_queue import InflightEntry

        conflict_outcome = MergeOutcome('conflict', reason='conflict')
        req, item = self._make_passthrough_item(
            git_repo, git_config, config, conflict_outcome,
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(GitOps(git_config, git_repo), q)
        worker._host_allocator = self._make_mock_allocator()
        worker._n_failed = False

        entry = InflightEntry(
            item=item, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False, phase='passthrough',
            passthrough_outcome=conflict_outcome,
        )
        try:
            await worker._finalize_inflight(entry)
        except Exception:
            pass
        assert worker._n_failed is True

    async def test_finalize_passthrough_n_failed_false_for_already_merged(
        self, git_repo: Path, git_config: GitConfig, config: OrchestratorConfig,
    ) -> None:
        """PASSTHROUGH already_merged → _n_failed = False (chain not broken).

        RED: same AssertionError as above.
        """
        from orchestrator.merge_queue import InflightEntry

        am_outcome = MergeOutcome('already_merged')
        req, item = self._make_passthrough_item(
            git_repo, git_config, config, am_outcome,
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(GitOps(git_config, git_repo), q)
        worker._host_allocator = self._make_mock_allocator()
        worker._n_failed = True  # start True to detect reset

        entry = InflightEntry(
            item=item, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False, phase='passthrough',
            passthrough_outcome=am_outcome,
        )
        try:
            await worker._finalize_inflight(entry)
        except Exception:
            pass
        assert worker._n_failed is False  # already_merged → not n_failed

    async def test_finalize_passthrough_already_delivered_skips_resolve(
        self, git_repo: Path, git_config: GitConfig, config: OrchestratorConfig,
    ) -> None:
        """PASSTHROUGH with already_delivered=True → req.result NOT set by finalize.

        RED: current code raises AssertionError before reaching the delivery check.
        """
        from orchestrator.merge_queue import InflightEntry

        conflict_outcome = MergeOutcome('conflict', reason='conflict')
        req, item = self._make_passthrough_item(
            git_repo, git_config, config, conflict_outcome, already_delivered=True,
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(GitOps(git_config, git_repo), q)
        worker._host_allocator = self._make_mock_allocator()

        entry = InflightEntry(
            item=item, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False, phase='passthrough',
            passthrough_outcome=conflict_outcome,
        )
        # req.result must NOT be set because already_delivered=True
        try:
            await worker._finalize_inflight(entry)
        except Exception:
            pass
        # already_delivered means the merger already set the result OOB
        assert not req.result.done()

    # ------------------------------------------------------------------
    # (c) DROPPED / REQUEUED sentinel paths
    # ------------------------------------------------------------------

    async def test_finalize_dropped_calls_cancel_and_release(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """DROPPED sentinel → allocator.cancel_and_release(lease) called.

        RED: current code raises AssertionError (merge_wt=None).
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'drop-a', 'da.py', 'a=1\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        mock_alloc = self._make_mock_allocator()
        worker._host_allocator = mock_alloc

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'DROPPED', merge_wt_val=None)

        try:
            await worker._finalize_inflight(entry)
        except Exception:
            pass
        mock_alloc.cancel_and_release.assert_called_once_with(lease)

    async def test_finalize_dropped_sets_n_failed_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """DROPPED sentinel → _n_failed = True (chain stale: sole waiter abandoned).

        RED: current code raises before reaching _n_failed.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'drop-b', 'db.py', 'b=2\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._n_failed = False

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'DROPPED', merge_wt_val=None)

        try:
            await worker._finalize_inflight(entry)
        except Exception:
            pass
        assert worker._n_failed is True

    async def test_finalize_dropped_returns_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """DROPPED sentinel → _finalize_inflight returns False.

        RED: current code raises before returning.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'drop-c', 'dc.py', 'c=3\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'DROPPED', merge_wt_val=None)

        result = await worker._finalize_inflight(entry)
        assert result is False  # RED: current code raises AssertionError

    async def test_finalize_requeued_returns_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REQUEUED sentinel → _finalize_inflight returns False.

        RED: current code raises AssertionError (merge_wt=None).
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'requeue-a', 'ra.py', 'a=1\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'REQUEUED', merge_wt_val=None)

        result = await worker._finalize_inflight(entry)
        assert result is False  # RED: current raises AssertionError

    async def test_finalize_requeued_calls_cancel_and_release(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REQUEUED sentinel → allocator.cancel_and_release(lease) called.

        RED: current code raises before cancel_and_release.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'requeue-b', 'rb.py', 'b=2\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        mock_alloc = self._make_mock_allocator()
        worker._host_allocator = mock_alloc

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'REQUEUED', merge_wt_val=None)

        try:
            await worker._finalize_inflight(entry)
        except Exception:
            pass
        mock_alloc.cancel_and_release.assert_called_once_with(lease)


# ---------------------------------------------------------------------------
# step-15 RED: SINGLE-HOST serial byte-identical via the restructured loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSingleHostSerialByteIdentical:
    """SINGLE-HOST serial byte-identical via the restructured _verifier_loop.

    RED until step-16 GREEN restructures _verifier_loop, adds _dispatch_item,
    and makes _verify_and_advance a thin compat shim.

    The two structural tests (test_dispatch_item_method_exists and
    test_verifier_loop_sets_n_failed_via_instance_attr) are the RED markers.
    The end-to-end tests document the byte-identical oracle that must stay
    green before and after the restructuring.
    """

    # ------------------------------------------------------------------
    # RED marker 1: _dispatch_item must exist on the class
    # ------------------------------------------------------------------

    async def test_dispatch_item_method_exists(self, git_ops: GitOps) -> None:
        """_dispatch_item is a callable on SpeculativeMergeWorker.

        RED: method not yet added (step-16 adds it).
        GREEN: step-16 adds _dispatch_item.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        assert callable(getattr(worker, '_dispatch_item', None)), (
            '_dispatch_item not found on SpeculativeMergeWorker — '
            'restructured loop not yet added (step-16). '
            'RED: None (missing). GREEN: callable after step-16.'
        )

    # ------------------------------------------------------------------
    # RED marker 2: _verifier_loop must update self._n_failed on the instance
    # ------------------------------------------------------------------

    async def test_verifier_loop_sets_n_failed_via_instance_attr(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Restructured _verifier_loop sets self._n_failed (not just a loop-local).

        The OLD _verifier_loop keeps n_failed as a loop-local variable and
        NEVER writes to self._n_failed.  The NEW loop delegates to
        _finalize_inflight which always sets self._n_failed in its finally.

        Use an immediate_outcome CONFLICT token to trigger the non-done path:
          - conflict.status='conflict' → _n_failed_val=True in _finalize_inflight
          - OLD: self._n_failed stays False (never set by old loop)
          - NEW: self._n_failed becomes True after finalize

        RED: old loop never writes self._n_failed → stays False after conflict.
        GREEN: step-16 restructured loop → _finalize_inflight sets self._n_failed=True.
        """
        from orchestrator.merge_queue import SpeculativeItem

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req = _make_request('nf-conflict', 'nf-conflict', git_ops.project_root, config)
        conflict_outcome = MergeOutcome('conflict', reason='merge conflict')

        token = SpeculativeItem(
            request=req,
            merge_result=None,
            merge_wt=None,
            base_sha='deadbeef',
            speculative=False,
            skip_verify=False,
            immediate_outcome=conflict_outcome,
            already_delivered=False,
        )

        await worker._verifier_queue.put(token)
        await worker._verifier_queue.put(None)  # sentinel

        # Drive _verifier_loop directly (same pattern as existing 3514/3571/3605 tests)
        await worker._verifier_loop()

        # OLD: self._n_failed stays False (loop-local n_failed only)
        # NEW: _finalize_inflight sets self._n_failed=True for conflict status
        assert worker._n_failed is True, (
            '_n_failed should be True after a conflict token. '
            'OLD _verifier_loop never updates self._n_failed (loop-local only). '
            'NEW restructured loop uses _finalize_inflight which sets self._n_failed. '
            'RED: stays False with old loop. GREEN: True after step-16.'
        )

    # ------------------------------------------------------------------
    # Byte-identical oracle: end-to-end run with two real items (no verify_runners)
    # ------------------------------------------------------------------

    async def test_single_host_two_items_both_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Single-host config + two real items → both resolve done, main advanced in order.

        This test is the byte-identical oracle: it must stay GREEN both before
        (old serial loop) and after (new dispatch-fill + finalize-head) step-16.

        Submission order: N is submitted first and must land on main first.
        Both items use real git repos so advance_main genuinely runs.
        """
        wt_a = await _make_branch_with_file(git_ops, 'task/sh-a', 'sha.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/sh-b', 'shb.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker_task = asyncio.create_task(worker.run())

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='sh-a',
            branch='task/sh-a',
            worktree=wt_a,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        req_b = MergeRequest(
            task_id='sh-b',
            branch='task/sh-b',
            worktree=wt_b,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            await q.put(req_a)
            outcome_a = await asyncio.wait_for(req_a.result, timeout=30)
            await q.put(req_b)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=30)

        await worker.stop()
        await worker_task

        assert outcome_a.status == 'done', f'Item A: expected done, got {outcome_a}'
        assert outcome_b.status == 'done', f'Item B: expected done, got {outcome_b}'

        # Both files must be on main
        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'sha.py' in main_files, 'Item A file not on main'
        assert 'shb.py' in main_files, 'Item B file not on main'


# ---------------------------------------------------------------------------
# step-17 RED: OVERLAP SIGNAL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOverlapSignal:
    """Two-host overlap: both verifies enter before either is released.

    RED (step-17) until step-18 GREEN wires the fill loop to continue
    dispatching while a verify slot is free.

    RED markers:
    1. gate_b_entered times out — N+1 is NOT dispatched while N is verifying
       (serial blocking-get path finalizes N before fetching N+1).
    2. Peak _inflight length == 1 (never 2 simultaneously in serial mode).

    GREEN after step-18: dispatch-fill loop continues picking up N+1 while N
    is still running → both gates enter → peak == 2 → submission-order finalize
    confirmed even when N+1's verify completes first.
    """

    class _TrackingDeque(collections.deque):  # type: ignore[type-arg]
        """deque subclass that records peak length via append tracking."""

        def __init__(self) -> None:
            super().__init__()
            self.max_len: int = 0

        def append(self, item: Any) -> None:  # type: ignore[override]
            super().append(item)
            if len(self) > self.max_len:
                self.max_len = len(self)

        def appendleft(self, item: Any) -> None:  # type: ignore[override]
            super().appendleft(item)
            if len(self) > self.max_len:
                self.max_len = len(self)

    async def test_both_verifies_enter_before_either_released(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """N+1 starts verifying while N is still in-flight (concurrent overlap).

        RED: gate_b_entered times out (N+1 not dispatched until N finalizes).
        GREEN (step-18): both gates enter → overlap confirmed → submission-order
        finalize confirmed (N lands first even when N+1 verify completes first).
        """
        import collections as _col
        import contextlib

        # ── Gated verifies ─────────────────────────────────────────────────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # Remote runner (N+1's verify host) — gated on gate_b
        gated_remote = _gated_runner(gate_b_release, gate_b_entered, name='laptop')

        # Local verify (N's verify host) — gated via run_scoped_verification patch
        async def _gated_local_verify(*args: Any, **kwargs: Any) -> MagicMock:
            gate_a_entered.set()
            await gate_a_release.wait()
            return MagicMock(passed=True, summary='', test_output='', lint_output='',
                             type_output='', category='')

        # ── Branches ───────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(git_ops, 'task/ov-a', 'ov_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/ov-b', 'ov_b.py', 'b = 2\n')

        # ── Worker setup ───────────────────────────────────────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        # Replace _inflight with a tracking deque (captures peak length)
        tracked = self._TrackingDeque()
        worker._inflight = tracked  # type: ignore[assignment]

        # Inject two-host allocator: local + gated_remote
        _inject_two_host_allocator(worker, gated_remote)

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='ov-a', branch='task/ov-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='ov-b', branch='task/ov-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local_verify):
            worker_task = asyncio.create_task(worker.run())

            # Submit both requests; merger will process them while verifier runs
            await q.put(req_a)
            await q.put(req_b)

            # N (local) should enter its gated verify quickly
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)

            # While N is STILL verifying (gate_a NOT released), N+1 should ALSO
            # start verifying (remote slot is free → dispatch-fill overlap).
            # RED: times out — serial loop only dispatches N+1 after N finalizes.
            # GREEN (step-18): concurrent fill → gate_b_entered fires quickly.
            try:
                await asyncio.wait_for(gate_b_entered.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                # Cleanup: release gates so worker can drain cleanly
                gate_a_release.set()
                gate_b_release.set()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(req_a.result, timeout=10.0)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(req_b.result, timeout=10.0)
                await worker.stop()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(worker_task, timeout=5.0)
                pytest.fail(
                    'OVERLAP not observed: gate_b timed out while gate_a was set. '
                    'RED: serial blocking-get path finalizes N before fetching N+1 '
                    '— N+1 only dispatched after N fully completes. '
                    'GREEN (step-18): blocking-get path loops back to fill → '
                    'dispatch-fill picks up N+1 to remote slot while N is in-flight.'
                )

            # ── Both verifies entered — overlap confirmed ──────────────────
            # gate_a is still blocking, gate_b is still blocking → true overlap
            assert gate_a_entered.is_set(), 'N (local) gate not entered'
            assert gate_b_entered.is_set(), 'N+1 (remote) gate not entered'
            assert not gate_a_release.is_set(), 'gate_a already released (test bug)'
            assert not gate_b_release.is_set(), 'gate_b already released (test bug)'

            # Peak _inflight == 2: both entries in deque before head was popped
            # RED: max_len == 1 (N appended, popped, finalized; then N+1 appended)
            # GREEN: max_len == 2 (N appended, N+1 appended, then head popped)
            assert tracked.max_len == 2, (
                f'Expected peak _inflight length 2 (both entries before popleft), '
                f'got {tracked.max_len}. '
                'RED: serial dispatch never fills two slots simultaneously. '
                'GREEN (step-18): fill loop appends N then N+1 before any popleft.'
            )

            # ── Submission-order finalize: release N+1 FIRST ──────────────
            # Even though N+1's verify finishes before N's, main must advance N first.
            gate_b_release.set()   # N+1 (remote) verify completes first
            gate_a_release.set()   # N (local) verify completes second

            # Both should resolve 'done'
            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)

        await worker.stop()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        assert outcome_a.status == 'done', f'N: expected done, got {outcome_a}'
        assert outcome_b.status == 'done', f'N+1: expected done, got {outcome_b}'

        # Both files on main (submission order preserved)
        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'ov_a.py' in main_files, 'N (ov_a.py) not on main'
        assert 'ov_b.py' in main_files, 'N+1 (ov_b.py) not on main'


# ---------------------------------------------------------------------------
# step-19 RED: CHAIN-INVALIDATION UNDER OVERLAP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestChainInvalidationUnderOverlap:
    """N's verify fails while N+1 is in-flight: N+1 aborted, re-merged, re-verifies done.

    RED (step-19) until step-20 GREEN implements the head-failure cascade.

    RED markers:
    1. req_b times out — the fill-ahead blocking-get deadlocks: after N fails,
       the loop re-enters DISPATCH-FILL with _has_running_inflight=True and
       free_host_count=1, so it blocks on queue.get() waiting for new items.
       N+1's verify task finishes (gate_b released) but its result is never
       consumed because the loop is waiting for QUEUE items, not verify tasks.
    2. remote cancel_verify NOT called — N+1's verify runs to completion
       (or is left dangling) without the cancel signal.

    GREEN after step-20: head-failure cascade in _verifier_loop:
      · cancels N+1's in-flight verify task
      · cancel_and_release(N+1's lease) → remote cancel_verify called
      · cleans N+1's stale merge worktree
      · _remerge(N+1's req) → fresh SpeculativeItem on actual main
      · appends to _redispatch → re-dispatched on next fill iteration
    N+1 resolves 'done' and main has ci_b.py; N (ci_a.py) never landed.
    """

    async def test_n_fail_aborts_downstream_verify_reruns_remerge(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """N's local verify fails; N+1's remote verify is aborted and re-verified.

        RED: loop deadlocks on fill-ahead queue.get() → req_b times out.
        GREEN (step-20): head-failure cascade aborts N+1, re-merges onto
        actual main, re-dispatches → N+1 resolves 'done'.
        """
        # ── N's local verify: gated (fails when gate_a_release is set) ──────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()

        # Track local-verify call count: first call = N (gate + fail),
        # subsequent calls = N+1's re-dispatch (pass immediately).
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                # N's verify: gate and fail
                gate_a_entered.set()
                await gate_a_release.wait()
                return MagicMock(
                    passed=False,
                    summary='test_failure',
                    test_output='FAILED',
                    lint_output='',
                    type_output='',
                    category='test_failure',
                    timed_out=False,
                    verify_skipped=False,
                )
            # N+1's re-dispatched local verify (GREEN step-20 cascade path)
            return MagicMock(
                passed=True,
                summary='ok',
                test_output='ok',
                lint_output='',
                type_output='',
                category='',
                timed_out=False,
                verify_skipped=False,
            )

        # ── N+1's remote verify: gated (passes when gate_b_release is set) ──
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        # ── Branches ─────────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(
            git_ops, 'task/ci-a', 'ci_a.py', 'a = 1\n'
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/ci-b', 'ci_b.py', 'b = 2\n'
        )

        # ── Worker setup ──────────────────────────────────────────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _inject_two_host_allocator(worker, gated_remote)

        loop = asyncio.get_event_loop()
        req_a = MergeRequest(
            task_id='ci-a', branch='task/ci-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='ci-b', branch='task/ci-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        outcome_b: MergeOutcome | None = None

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await q.put(req_a)
            await q.put(req_b)

            # Wait for both verifies to enter (true concurrent overlap)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # N's verify fails
            gate_a_release.set()

            # N must resolve with a fail status
            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail, got status={outcome_a.status!r}.'
            )

            # Release N+1's gate so the test can complete in both paths:
            # RED: N+1's inner verify task unblocks, but the loop is still
            #      stuck on fill-ahead queue.get() → req_b never resolves.
            # GREEN: cascade already cancelled N+1's task; gate_b unblocks
            #        only the leaked inner task (result ignored).
            gate_b_release.set()

            # Wait for N+1 to resolve:
            # GREEN: cascade → re-merge → re-verify → 'done' (fast)
            # RED: deadlock → TimeoutError → outcome_b stays None
            try:
                outcome_b = await asyncio.wait_for(req_b.result, timeout=5.0)
            except asyncio.TimeoutError:
                pass  # RED: expected deadlock

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # ── RED: fails here (outcome_b is None due to timeout) ──────────────
        assert outcome_b is not None and outcome_b.status == 'done', (
            f'Expected N+1 to resolve "done" after re-merge/re-verify, '
            f'got {outcome_b!r}. '
            'RED: fill-ahead blocking-get deadlocks after N fails — '
            '_inflight still has N+1 but loop waits for queue items. '
            'GREEN (step-20): head-failure cascade clears _inflight, '
            're-dispatches N+1 via _redispatch → "done".'
        )

        # ── RED: fails here (cancel_verify not called) ───────────────────────
        gated_remote.cancel_verify.assert_called_once()

        # ── Main state: N+1 landed, N never did ─────────────────────────────
        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'ci_b.py' in main_files, (
            'N+1 (ci_b.py) not on main after re-merge/re-verify'
        )
        assert 'ci_a.py' not in main_files, (
            'N (ci_a.py) must NOT be on main (verify failed)'
        )
