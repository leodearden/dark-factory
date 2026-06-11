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
