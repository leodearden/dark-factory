"""Tests for the persistent warm merge-verify worktree feature (task 1692).

All tests in this file relate to PRD κ Phase 1 of reify warmer-builds-merge-verify.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeOutcome, MergeRequest, SpeculativeMergeWorker

# ---------------------------------------------------------------------------
# Real-git fixtures (mirroring test_merge_queue_restart_hook.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config_base() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        persistent_merge_worktree=False,
    )


@pytest.fixture
def git_ops(git_config_base: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config_base, git_repo)


def _make_persistent_config(git_repo: Path, *, persistent: bool) -> OrchestratorConfig:
    """Build OrchestratorConfig with the persistent_merge_worktree knob set."""
    git = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        persistent_merge_worktree=persistent,
    )
    return OrchestratorConfig(project_root=git_repo, git=git)


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


def _make_merge_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    *,
    persistent: bool = False,
    safety_valve: int = 0,
    push_after_advance: bool = False,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig with the given persistent-worktree knobs."""
    git = GitConfig(
        push_after_advance=push_after_advance,
        persistent_merge_worktree=persistent,
        persistent_merge_worktree_safety_valve_every_n=safety_valve,
    )
    return OrchestratorConfig(project_root=tmp_path, git=git)


# ---------------------------------------------------------------------------
# Step 11 — enforce_persistent_worktree_serial_lane startup guard
# ---------------------------------------------------------------------------


class TestEnforcePersistentWorktreeSerialLane:
    """enforce_persistent_worktree_serial_lane fail-closed startup guard.

    Step 11 (RED): function/exception absent today.
    """

    def test_knob_on_bound_gt1_raises(self, tmp_path: Path):
        """persistent_merge_worktree=True + merge_ahead_bound=2 → raises PersistentWorktreeConfigError."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            PersistentWorktreeConfigError,
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=True)
        with pytest.raises(PersistentWorktreeConfigError) as exc_info:
            enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=2)

        msg = str(exc_info.value)
        # Message must name the bound so the operator knows what to change
        assert '2' in msg, (
            f'PersistentWorktreeConfigError must mention the bad bound (2); got: {msg!r}'
        )

    def test_knob_on_bound_1_no_raise(self, tmp_path: Path):
        """persistent_merge_worktree=True + merge_ahead_bound=1 → no raise."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=True)
        result = enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=1)
        # Returns None (no return value needed)
        assert result is None

    def test_knob_off_bound_gt1_no_raise(self, tmp_path: Path):
        """persistent_merge_worktree=False + merge_ahead_bound=2 → guard inert."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=False)
        # Must not raise even with a large bound
        result = enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=2)
        assert result is None

    # ---- per-host reframe cases (task η, step-1 RED) ----

    def test_reframe_bound2_num_hosts2_no_raise(self, tmp_path: Path):
        """knob ON + bound=2 + num_hosts=2 → per_host=ceil(2/2)=1 → no raise (K=2 / 2-host)."""
        from orchestrator.merge_queue import (
            enforce_persistent_worktree_serial_lane,  # noqa: PLC0415
        )

        cfg = _make_config(tmp_path, persistent=True)
        result = enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=2, num_hosts=2)
        assert result is None

    def test_reframe_bound2_num_hosts1_raises(self, tmp_path: Path):
        """knob ON + bound=2 + num_hosts=1 → per_host=ceil(2/1)=2 → raises (single host, 2 in-flight)."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            PersistentWorktreeConfigError,
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=True)
        with pytest.raises(PersistentWorktreeConfigError) as exc_info:
            enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=2, num_hosts=1)
        msg = str(exc_info.value)
        # Message must mention bound and per-host count
        assert '2' in msg, f'Message must mention bound or per-host count; got: {msg!r}'

    def test_reframe_bound3_num_hosts2_raises(self, tmp_path: Path):
        """knob ON + bound=3 + num_hosts=2 → per_host=ceil(3/2)=2 → raises (uneven split)."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            PersistentWorktreeConfigError,
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=True)
        with pytest.raises(PersistentWorktreeConfigError):
            enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=3, num_hosts=2)

    def test_reframe_bound4_num_hosts4_no_raise(self, tmp_path: Path):
        """knob ON + bound=4 + num_hosts=4 → per_host=ceil(4/4)=1 → no raise."""
        from orchestrator.merge_queue import (
            enforce_persistent_worktree_serial_lane,  # noqa: PLC0415
        )

        cfg = _make_config(tmp_path, persistent=True)
        result = enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=4, num_hosts=4)
        assert result is None

    def test_reframe_bound4_num_hosts2_raises(self, tmp_path: Path):
        """knob ON + bound=4 + num_hosts=2 → per_host=ceil(4/2)=2 → raises."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            PersistentWorktreeConfigError,
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=True)
        with pytest.raises(PersistentWorktreeConfigError):
            enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=4, num_hosts=2)


# ---------------------------------------------------------------------------
# Step 13 — _acquire_warm_verify_worktree unit tests
# ---------------------------------------------------------------------------


def _make_stub_git_ops(warm_path: Path) -> MagicMock:
    """Build a stub GitOps with async reset/cleanup methods that record calls."""
    stub = MagicMock()
    stub.reset_persistent_merge_worktree = AsyncMock(return_value=warm_path)
    stub.cleanup_merge_worktree = AsyncMock(return_value=None)
    stub.persistent_merge_worktree_path = warm_path
    return stub


def _make_stub_req(tmp_path: Path, *, persistent: bool) -> MagicMock:
    """Build a stub MergeRequest with config.git.persistent_merge_worktree."""
    cfg = _make_config(tmp_path, persistent=persistent)
    req = MagicMock()
    req.config = cfg
    return req


class TestAcquireWarmVerifyWorktree:
    """Unit tests for _acquire_warm_verify_worktree with stub git_ops.

    Step 13 (RED): helper absent today — ImportError expected.
    """

    @pytest.mark.asyncio
    async def test_knob_off_returns_ephemeral_unchanged(self, tmp_path: Path):
        """Knob OFF: returns merge_wt unchanged, no reset/cleanup calls."""
        from orchestrator.merge_queue import _acquire_warm_verify_worktree  # noqa: PLC0415

        warm_path = tmp_path / '_merge-verify'
        ephemeral = tmp_path / '_merge-abc123'
        stub = _make_stub_git_ops(warm_path)
        req = _make_stub_req(tmp_path, persistent=False)

        result = await _acquire_warm_verify_worktree(
            stub, req, ephemeral, 'sha-abc', safety_valve_due=False
        )

        assert result == ephemeral, 'knob OFF: must return merge_wt unchanged'
        stub.reset_persistent_merge_worktree.assert_not_called()
        stub.cleanup_merge_worktree.assert_not_called()

    @pytest.mark.asyncio
    async def test_knob_on_not_due_swaps_to_warm(self, tmp_path: Path):
        """Knob ON, safety_valve_due=False → resets warm wt, cleans up ephemeral, returns warm path."""
        from orchestrator.merge_queue import _acquire_warm_verify_worktree  # noqa: PLC0415

        warm_path = tmp_path / '_merge-verify'
        ephemeral = tmp_path / '_merge-abc123'
        stub = _make_stub_git_ops(warm_path)
        req = _make_stub_req(tmp_path, persistent=True)

        result = await _acquire_warm_verify_worktree(
            stub, req, ephemeral, 'sha-abc', safety_valve_due=False
        )

        assert result == warm_path, 'knob ON+not due: must return warm path'
        stub.reset_persistent_merge_worktree.assert_awaited_once_with('sha-abc')
        stub.cleanup_merge_worktree.assert_awaited_once_with(ephemeral)

    @pytest.mark.asyncio
    async def test_knob_on_due_returns_ephemeral_unchanged(self, tmp_path: Path):
        """Knob ON, safety_valve_due=True → returns merge_wt unchanged (cold throwaway path)."""
        from orchestrator.merge_queue import _acquire_warm_verify_worktree  # noqa: PLC0415

        warm_path = tmp_path / '_merge-verify'
        ephemeral = tmp_path / '_merge-abc123'
        stub = _make_stub_git_ops(warm_path)
        req = _make_stub_req(tmp_path, persistent=True)

        result = await _acquire_warm_verify_worktree(
            stub, req, ephemeral, 'sha-abc', safety_valve_due=True
        )

        assert result == ephemeral, 'safety_valve_due=True: must return merge_wt unchanged (cold path)'
        stub.reset_persistent_merge_worktree.assert_not_called()
        stub.cleanup_merge_worktree.assert_not_called()

    @pytest.mark.asyncio
    async def test_knob_on_not_due_none_merge_wt_no_cleanup(self, tmp_path: Path):
        """Knob ON, merge_wt=None (edge case) → no cleanup_merge_worktree call, returns warm path."""
        from orchestrator.merge_queue import _acquire_warm_verify_worktree  # noqa: PLC0415

        warm_path = tmp_path / '_merge-verify'
        stub = _make_stub_git_ops(warm_path)
        req = _make_stub_req(tmp_path, persistent=True)

        result = await _acquire_warm_verify_worktree(
            stub, req, None, 'sha-abc', safety_valve_due=False
        )

        assert result == warm_path
        stub.reset_persistent_merge_worktree.assert_awaited_once_with('sha-abc')
        stub.cleanup_merge_worktree.assert_not_called()


# ---------------------------------------------------------------------------
# Step 15 — Integration: _verify_and_advance routes to warm worktree
# ---------------------------------------------------------------------------


class TestPersistentWorktreeVerifyRouting:
    """Integration tests driving SpeculativeMergeWorker with real-git fixtures.

    Step 15 (RED): no routing in _verify_and_advance today — warm worktree
    feature is wired in step 16.
    """

    @pytest.mark.asyncio
    async def test_verify_in_warm_worktree_when_knob_on(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """Knob ON: verify runs in warm _merge-verify; ephemeral gone; warm persists; main advanced."""
        cfg = _make_persistent_config(git_repo, persistent=True)
        wt = await _make_branch_with_file(git_ops, 'warm-test', 'warm.py', 'x = 1\n')
        req = _make_merge_request('warm-test', 'warm-test', wt, cfg)

        captured_merge_wt: list[Path] = []

        async def _fake_run_post_merge_verify(git_ops_arg, req_arg, merge_wt_arg, **kwargs):
            captured_merge_wt.append(merge_wt_arg)
            return None  # PASS

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            side_effect=_fake_run_post_merge_verify,
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)

        await worker.stop()
        await worker_task

        assert outcome.status == 'done', f'Expected done, got: {outcome}'

        # The warm worktree must have been used for verify
        assert len(captured_merge_wt) == 1, 'Expected exactly one verify call'
        warm_path = git_ops.persistent_merge_worktree_path
        assert captured_merge_wt[0].resolve() == warm_path.resolve(), (
            f'verify must run in warm worktree {warm_path}; '
            f'got: {captured_merge_wt[0]}'
        )

        # The warm worktree PERSISTS after a successful advance (cleanup is no-op on it)
        assert warm_path.exists(), 'Warm worktree must persist after successful advance'
        assert _is_registered_worktree(warm_path, git_repo), (
            'Warm worktree must remain registered after advance'
        )

        # Main has advanced to the merge commit
        assert outcome.merge_sha is not None

    @pytest.mark.asyncio
    async def test_verify_in_ephemeral_worktree_when_knob_off(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """Knob OFF: verify runs in an ephemeral _merge-<uuid>; no _merge-verify created."""
        cfg = _make_persistent_config(git_repo, persistent=False)
        wt = await _make_branch_with_file(git_ops, 'cold-test', 'cold.py', 'y = 2\n')
        req = _make_merge_request('cold-test', 'cold-test', wt, cfg)

        captured_merge_wt: list[Path] = []

        async def _fake_run_post_merge_verify(git_ops_arg, req_arg, merge_wt_arg, **kwargs):
            captured_merge_wt.append(merge_wt_arg)
            return None  # PASS

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            side_effect=_fake_run_post_merge_verify,
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)

        await worker.stop()
        await worker_task

        assert outcome.status == 'done', f'Expected done, got: {outcome}'

        # Verify ran in an ephemeral worktree (not _merge-verify)
        assert len(captured_merge_wt) == 1, 'Expected exactly one verify call'
        warm_path = git_ops.persistent_merge_worktree_path
        assert captured_merge_wt[0].resolve() != warm_path.resolve(), (
            'knob OFF: verify must NOT run in the warm worktree'
        )
        assert captured_merge_wt[0].name.startswith('_merge-'), (
            f'expected ephemeral _merge-<uuid>; got: {captured_merge_wt[0].name}'
        )

        # No _merge-verify should have been created
        assert not warm_path.exists(), 'knob OFF: _merge-verify must not exist'


def _is_registered_worktree(wt_path: Path, repo: Path) -> bool:
    """Return True if wt_path appears in `git worktree list --porcelain`."""
    import subprocess  # noqa: PLC0415
    result = subprocess.run(
        ['git', 'worktree', 'list', '--porcelain'],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return str(wt_path.resolve()) in result.stdout


# ---------------------------------------------------------------------------
# Step 17 — _safety_valve_due predicate + safety_valve_every_n integration
# ---------------------------------------------------------------------------


class TestSafetyValveDue:
    """Unit tests for the _safety_valve_due predicate.

    Step 17 (RED): predicate absent today — ImportError expected.
    """

    def test_every_n_zero_always_false(self):
        """every_n=0 (disabled) → always False regardless of attempt_count."""
        from orchestrator.merge_queue import _safety_valve_due  # noqa: PLC0415

        for attempt in range(0, 20):
            assert _safety_valve_due(attempt, 0) is False, (
                f'every_n=0 must be disabled; attempt={attempt}'
            )

    def test_every_n_negative_always_false(self):
        """every_n<0 → always False (guard against invalid config)."""
        from orchestrator.merge_queue import _safety_valve_due  # noqa: PLC0415

        for attempt in range(0, 10):
            assert _safety_valve_due(attempt, -1) is False

    def test_every_n_3_due_on_multiples(self):
        """every_n=3 → True exactly at attempt_count 3, 6, 9; False otherwise."""
        from orchestrator.merge_queue import _safety_valve_due  # noqa: PLC0415

        due_at = {3, 6, 9}
        for attempt in range(0, 12):
            expected = attempt in due_at
            assert _safety_valve_due(attempt, 3) is expected, (
                f'every_n=3, attempt={attempt}: expected {expected}'
            )

    def test_attempt_zero_never_due(self):
        """attempt_count=0 must never trigger the valve (first attempt is 1-based)."""
        from orchestrator.merge_queue import _safety_valve_due  # noqa: PLC0415

        for every_n in [1, 2, 3, 5]:
            assert _safety_valve_due(0, every_n) is False, (
                f'attempt=0 must not be due; every_n={every_n}'
            )

    def test_every_n_1_due_at_every_positive_attempt(self):
        """every_n=1 → True for every positive attempt (cold on every land)."""
        from orchestrator.merge_queue import _safety_valve_due  # noqa: PLC0415

        assert _safety_valve_due(0, 1) is False  # attempt 0 never due
        for attempt in range(1, 6):
            assert _safety_valve_due(attempt, 1) is True, (
                f'every_n=1, attempt={attempt}: must be due'
            )


class TestSafetyValveIntegration:
    """Integration: safety_valve_every_n=1 bypasses warm swap → cold ephemeral path.

    Step 17 (RED): counter and predicate not wired in _verify_and_advance yet.
    """

    @pytest.mark.asyncio
    async def test_safety_valve_every_n_1_uses_ephemeral(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """With safety_valve_every_n=1 and knob ON, every verify uses ephemeral (cold)."""
        cfg_git = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            persistent_merge_worktree=True,
            persistent_merge_worktree_safety_valve_every_n=1,
        )
        cfg = OrchestratorConfig(project_root=git_repo, git=cfg_git)

        wt = await _make_branch_with_file(git_ops, 'valve-test', 'valve.py', 'z = 3\n')
        req = _make_merge_request('valve-test', 'valve-test', wt, cfg)

        captured_merge_wt: list[Path] = []

        async def _fake_run_post_merge_verify(git_ops_arg, req_arg, merge_wt_arg, **kwargs):
            captured_merge_wt.append(merge_wt_arg)
            return None  # PASS

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            side_effect=_fake_run_post_merge_verify,
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)

        await worker.stop()
        await worker_task

        assert outcome.status == 'done', f'Expected done; got: {outcome}'

        # With safety_valve_every_n=1, the FIRST verifying attempt (count=1)
        # is due → must bypass the warm swap and use the ephemeral path.
        assert len(captured_merge_wt) == 1
        warm_path = git_ops.persistent_merge_worktree_path
        assert captured_merge_wt[0].resolve() != warm_path.resolve(), (
            f'safety_valve_every_n=1: first attempt must use ephemeral, not warm; '
            f'got: {captured_merge_wt[0]}'
        )
