"""Tests for the post-merge pyright equivalence check.

Covers:
- ``PostMergePyrightResult`` dataclass and ``.broken`` property
- ``_check_post_merge_pyright`` function — real-git behavioural tests
  (using hermetic stand-in type_check_command), classification / fail-open
  edge cases, and call-site integration with MergeWorker / SpeculativeMergeWorker.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    MergeWorker,
    PostMergePyrightResult,
    SpeculativeMergeWorker,
    _check_post_merge_pyright,
)

# ---------------------------------------------------------------------------
# Fixtures — shared real-git setup (mirrors TestCheckPostMergeEquivalence)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


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
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# Stand-in type_check_command that exits 0 on a clean tree and exits 1 with
# a deterministic 'synthetic type error' message written to stderr when the
# `.BROKEN_UNION` marker file is present.  Tests can assert that specific
# string appears in result.detail rather than relying on a non-empty
# failure_report() header.
_TYPE_CMD_CONDITIONAL = (
    'python3 -c "'
    "import sys, pathlib; "
    "(sys.stderr.write('synthetic type error\\n'), sys.exit(1)) "
    "if pathlib.Path('.BROKEN_UNION').exists() else sys.exit(0)"
    '"'
)


def _make_module_config(
    prefix: str = 'subpkg',
    type_check_command: str | None = _TYPE_CMD_CONDITIONAL,
) -> ModuleConfig:
    return ModuleConfig(
        prefix=prefix,
        test_command=None,
        lint_command=None,
        type_check_command=type_check_command,
    )


# ---------------------------------------------------------------------------
# PostMergePyrightResult unit tests
# ---------------------------------------------------------------------------


class TestPostMergePyrightResult:
    def test_empty_result_is_not_broken(self):
        result = PostMergePyrightResult()
        assert result.broken is False
        assert result.failing_subprojects == []
        assert result.detail == ''

    def test_non_empty_failing_subprojects_is_broken(self):
        result = PostMergePyrightResult(failing_subprojects=['subpkg'])
        assert result.broken is True

    def test_with_detail(self):
        result = PostMergePyrightResult(
            failing_subprojects=['subpkg'],
            detail='error: src/foo.py:10: Missing method bar',
        )
        assert result.broken is True
        assert 'bar' in result.detail

    def test_timed_out_subprojects_defaults_to_empty(self):
        """timed_out_subprojects defaults to [] and timed_out is False for an empty result."""
        result = PostMergePyrightResult()
        assert result.timed_out_subprojects == []
        assert result.timed_out is False

    def test_timed_out_true_when_timed_out_subprojects_non_empty(self):
        """timed_out is True iff timed_out_subprojects is non-empty."""
        result = PostMergePyrightResult(timed_out_subprojects=['pkg'])
        assert result.timed_out is True

    def test_broken_reflects_only_failing_subprojects_not_timed_out(self):
        """A result with only timed_out_subprojects (no failing_subprojects) is NOT broken."""
        result = PostMergePyrightResult(timed_out_subprojects=['pkg'])
        assert result.broken is False
        assert result.failing_subprojects == []

    def test_broken_and_timed_out_can_coexist(self):
        """When a module is both failing and timed-out, broken and timed_out are both True."""
        result = PostMergePyrightResult(
            failing_subprojects=['pkg'],
            timed_out_subprojects=['pkg'],
        )
        assert result.broken is True
        assert result.timed_out is True


# ---------------------------------------------------------------------------
# _check_post_merge_pyright — real-git behavioural tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckPostMergePyrightBehavioral:
    """Real-git tests using a hermetic stand-in type_check_command."""

    async def test_clean_tree_returns_not_broken(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Clean tree + command that exits 0 → result.broken is False."""
        # Create a branch with a simple file
        wt = (await git_ops.create_worktree('pyright-clean')).path
        (wt / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'Add mod.py')

        merge_result = await git_ops.merge_to_main(wt, 'pyright-clean')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='pyright-clean', max_attempts=1,
            )
            advanced_sha = (
                getattr(git_ops, '_last_advanced_sha', None)
                or merge_result.merge_commit
            )
            assert advanced_sha is not None

            mc = _make_module_config()
            result = await _check_post_merge_pyright(
                advanced_sha, git_ops, config, [mc], task_id='pyright-clean-test',
            )
            assert result.broken is False
            assert result.failing_subprojects == []
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_broken_tree_returns_broken_with_prefix_and_detail(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Broken tree + command that exits non-zero → broken=True, prefix listed, detail non-empty."""
        # Create a branch with a "broken union" marker file
        wt = (await git_ops.create_worktree('pyright-broken')).path
        (wt / 'mod.py').write_text('x = 1\n')
        # The marker file triggers the stand-in type_check_command to exit 1
        (wt / '.BROKEN_UNION').write_text('broken\n')
        await git_ops.commit(wt, 'Add mod.py + .BROKEN_UNION marker')

        merge_result = await git_ops.merge_to_main(wt, 'pyright-broken')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='pyright-broken', max_attempts=1,
            )
            advanced_sha = (
                getattr(git_ops, '_last_advanced_sha', None)
                or merge_result.merge_commit
            )
            assert advanced_sha is not None

            mc = _make_module_config(prefix='subpkg')
            result = await _check_post_merge_pyright(
                advanced_sha, git_ops, config, [mc], task_id='pyright-broken-test',
            )
            assert result.broken is True
            assert 'subpkg' in result.failing_subprojects
            # detail carries the deterministic sentinel written to stderr by
            # the stand-in command; asserts capture of real subprocess output.
            assert 'synthetic type error' in result.detail
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_empty_module_configs_returns_clean(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Empty module_configs → no-op, returns clean (not broken)."""
        # We don't even need a real merge for this; use any valid SHA
        advanced_sha = await git_ops.get_main_sha()

        result = await _check_post_merge_pyright(
            advanced_sha, git_ops, config, [], task_id='pyright-empty-mc',
        )
        assert result.broken is False
        assert result.failing_subprojects == []

    async def test_module_config_with_no_type_check_command_skipped(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """A ModuleConfig whose type_check_command=None → skipped, returns clean."""
        advanced_sha = await git_ops.get_main_sha()
        mc = _make_module_config(type_check_command=None)

        result = await _check_post_merge_pyright(
            advanced_sha, git_ops, config, [mc], task_id='pyright-no-cmd',
        )
        assert result.broken is False
        assert result.failing_subprojects == []


# ---------------------------------------------------------------------------
# _check_post_merge_pyright — classification / fail-open edge cases (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckPostMergePyrightClassification:
    """Classification / fail-open edge cases — mock run_verification and git_ops."""

    def _make_git_ops(self, *, raise_on_create: Exception | None = None):
        git_ops = MagicMock()
        if raise_on_create is not None:
            git_ops._create_merge_worktree = AsyncMock(side_effect=raise_on_create)
        else:
            git_ops._create_merge_worktree = AsyncMock(
                return_value=(Path('/tmp/fake-wt'), 'deadbeef')
            )
        git_ops.cleanup_merge_worktree = AsyncMock()
        return git_ops

    def _make_verify_result(
        self, *, passed: bool, timed_out: bool = False,
        type_output: str = 'type error output',
    ) -> MagicMock:
        v = MagicMock()
        v.passed = passed
        v.timed_out = timed_out
        v.type_output = type_output
        v.failure_report = MagicMock(return_value='failure report text' if not passed else '')
        return v

    @pytest.fixture
    def config(self, tmp_path: Path) -> OrchestratorConfig:
        """Minimal config for mocked tests."""
        from orchestrator.config import GitConfig
        return OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
                push_after_advance=False,
            ),
        )

    async def test_timeout_treated_as_fail_open(
        self, config: OrchestratorConfig, caplog,
    ):
        """verify.timed_out=True + passed=False → fail open (not broken), warning logged."""
        git_ops = self._make_git_ops()
        mc = _make_module_config(prefix='pkg', type_check_command='pyright src/')
        verify_result = self._make_verify_result(passed=False, timed_out=True)

        with patch(
            'orchestrator.merge_queue.run_verification',
            AsyncMock(return_value=verify_result),
        ), caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_post_merge_pyright(
                'abc123', git_ops, config, [mc], task_id='timeout-test',
            )

        assert result.broken is False
        assert result.failing_subprojects == []
        assert any('timed out' in r.message.lower() for r in caplog.records)

    async def test_genuine_failure_not_timed_out_is_broken(
        self, config: OrchestratorConfig,
    ):
        """verify.passed=False AND timed_out=False → genuine failure → broken=True."""
        git_ops = self._make_git_ops()
        mc = _make_module_config(prefix='pkg', type_check_command='pyright src/')
        verify_result = self._make_verify_result(
            passed=False, timed_out=False, type_output='error: foo.py:10',
        )

        with patch(
            'orchestrator.merge_queue.run_verification',
            AsyncMock(return_value=verify_result),
        ):
            result = await _check_post_merge_pyright(
                'abc123', git_ops, config, [mc], task_id='fail-test',
            )

        assert result.broken is True
        assert 'pkg' in result.failing_subprojects
        assert result.detail != ''

    async def test_create_merge_worktree_exception_fails_open(
        self, config: OrchestratorConfig, caplog,
    ):
        """_create_merge_worktree raising RuntimeError → fail open (not broken), WARNING logged."""
        git_ops = self._make_git_ops(raise_on_create=RuntimeError('git error'))
        mc = _make_module_config(prefix='pkg', type_check_command='pyright src/')

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_post_merge_pyright(
                'abc123', git_ops, config, [mc], task_id='infra-error-test',
            )

        assert result.broken is False
        assert result.failing_subprojects == []
        assert any('infra error' in r.message.lower() for r in caplog.records)
        # cleanup must NOT be called since worktree was never created
        git_ops.cleanup_merge_worktree.assert_not_awaited()

    async def test_cleanup_called_even_when_verify_fails(
        self, config: OrchestratorConfig,
    ):
        """Worktree cleanup is awaited in finally even when verify reports a failure."""
        fake_wt = Path('/tmp/fake-wt')
        git_ops = self._make_git_ops()
        git_ops._create_merge_worktree = AsyncMock(
            return_value=(fake_wt, 'deadbeef')
        )
        mc = _make_module_config(prefix='pkg', type_check_command='pyright src/')
        verify_result = self._make_verify_result(passed=False, timed_out=False)

        with patch(
            'orchestrator.merge_queue.run_verification',
            AsyncMock(return_value=verify_result),
        ):
            result = await _check_post_merge_pyright(
                'abc123', git_ops, config, [mc], task_id='cleanup-test',
            )

        assert result.broken is True
        git_ops.cleanup_merge_worktree.assert_awaited_once_with(fake_wt)


# ---------------------------------------------------------------------------
# _run_unscoped_typechecks — classification tests (step-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunUnscopedTypechecks:
    """Tests for the _run_unscoped_typechecks helper.

    Uses a caller-supplied fake worktree Path to confirm the helper does NOT
    create or clean up a worktree — it operates on what the caller provides.
    """

    def _make_git_ops(self) -> MagicMock:
        git_ops = MagicMock()
        git_ops._create_merge_worktree = AsyncMock()
        git_ops.cleanup_merge_worktree = AsyncMock()
        return git_ops

    def _make_verify_result(
        self, *, passed: bool, timed_out: bool = False,
        type_output: str = 'type error output',
    ) -> MagicMock:
        v = MagicMock()
        v.passed = passed
        v.timed_out = timed_out
        v.type_output = type_output
        v.failure_report = MagicMock(return_value='failure report text' if not passed else '')
        return v

    @pytest.fixture
    def config(self, tmp_path: Path) -> OrchestratorConfig:
        from orchestrator.config import GitConfig
        return OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
                push_after_advance=False,
            ),
        )

    async def test_genuine_failure_is_broken(self, config: OrchestratorConfig):
        """passed=False, timed_out=False → prefix in failing_subprojects, broken=True."""
        from orchestrator.merge_queue import _run_unscoped_typechecks
        fake_wt = Path('/tmp/fake-wt')
        git_ops = self._make_git_ops()
        mc = _make_module_config(prefix='pkg', type_check_command='pyright src/')
        verify_result = self._make_verify_result(passed=False, timed_out=False)

        with patch(
            'orchestrator.merge_queue.run_verification',
            AsyncMock(return_value=verify_result),
        ):
            result = await _run_unscoped_typechecks(
                fake_wt, config, [mc],
                block_on_timeout=False, task_id='fail-test',
            )

        assert result.broken is True
        assert 'pkg' in result.failing_subprojects
        # No worktree creation/cleanup by the helper itself
        git_ops._create_merge_worktree.assert_not_awaited()
        git_ops.cleanup_merge_worktree.assert_not_awaited()

    async def test_timeout_with_block_on_timeout_false_fails_open(
        self, config: OrchestratorConfig,
    ):
        """timed_out=True, block_on_timeout=False → fail-open: NOT in failing, IS in timed_out."""
        from orchestrator.merge_queue import _run_unscoped_typechecks
        fake_wt = Path('/tmp/fake-wt')
        mc = _make_module_config(prefix='pkg', type_check_command='pyright src/')
        verify_result = self._make_verify_result(passed=False, timed_out=True)

        with patch(
            'orchestrator.merge_queue.run_verification',
            AsyncMock(return_value=verify_result),
        ):
            result = await _run_unscoped_typechecks(
                fake_wt, config, [mc],
                block_on_timeout=False, task_id='timeout-open-test',
            )

        assert result.broken is False
        assert 'pkg' not in result.failing_subprojects
        assert 'pkg' in result.timed_out_subprojects

    async def test_timeout_with_block_on_timeout_true_is_broken(
        self, config: OrchestratorConfig,
    ):
        """timed_out=True, block_on_timeout=True → fail-closed: in BOTH failing AND timed_out."""
        from orchestrator.merge_queue import _run_unscoped_typechecks
        fake_wt = Path('/tmp/fake-wt')
        mc = _make_module_config(prefix='pkg', type_check_command='pyright src/')
        verify_result = self._make_verify_result(passed=False, timed_out=True)

        with patch(
            'orchestrator.merge_queue.run_verification',
            AsyncMock(return_value=verify_result),
        ):
            result = await _run_unscoped_typechecks(
                fake_wt, config, [mc],
                block_on_timeout=True, task_id='timeout-closed-test',
            )

        assert result.broken is True
        assert 'pkg' in result.failing_subprojects
        assert 'pkg' in result.timed_out_subprojects

    async def test_clean_result_is_empty(self, config: OrchestratorConfig):
        """passed=True → both lists empty, broken=False, timed_out=False."""
        from orchestrator.merge_queue import _run_unscoped_typechecks
        fake_wt = Path('/tmp/fake-wt')
        mc = _make_module_config(prefix='pkg', type_check_command='pyright src/')
        verify_result = self._make_verify_result(passed=True, timed_out=False)

        with patch(
            'orchestrator.merge_queue.run_verification',
            AsyncMock(return_value=verify_result),
        ):
            result = await _run_unscoped_typechecks(
                fake_wt, config, [mc],
                block_on_timeout=True, task_id='clean-test',
            )

        assert result.broken is False
        assert result.timed_out is False
        assert result.failing_subprojects == []
        assert result.timed_out_subprojects == []

    async def test_no_type_check_command_skips_module(self, config: OrchestratorConfig):
        """Modules with type_check_command=None are skipped; returns clean result."""
        from orchestrator.merge_queue import _run_unscoped_typechecks
        fake_wt = Path('/tmp/fake-wt')
        mc = _make_module_config(prefix='pkg', type_check_command=None)

        with patch(
            'orchestrator.merge_queue.run_verification',
            AsyncMock(),
        ) as mock_run:
            result = await _run_unscoped_typechecks(
                fake_wt, config, [mc],
                block_on_timeout=True, task_id='no-cmd-test',
            )

        mock_run.assert_not_awaited()
        assert result.broken is False


# ---------------------------------------------------------------------------
# MergeWorker._do_merge call-site integration tests  (step-5)
# ---------------------------------------------------------------------------


def _make_merge_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig] | None = None,
) -> MergeRequest:
    future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=module_configs or [],
        config=config,
        result=future,
    )


def _mock_scoped_verify_pass():
    """Return a mock that makes run_scoped_verification always pass."""
    return AsyncMock(return_value=MagicMock(passed=True, summary=''))


def _mock_pyright_clean():
    """Return a mock that makes _check_post_merge_pyright return clean (not broken)."""
    return AsyncMock(return_value=PostMergePyrightResult())


def _mock_pyright_broken(prefix: str = 'subpkg'):
    """Return a mock that makes _check_post_merge_pyright return broken."""
    return AsyncMock(return_value=PostMergePyrightResult(
        failing_subprojects=[prefix],
        detail='error: src/foo.py:10: Missing method bar',
    ))


@pytest.mark.asyncio
class TestMergeWorkerPyrightCallSite:
    """Integration: MergeWorker._do_merge calls _check_post_merge_pyright after equiv check."""

    async def test_broken_pyright_blocks_merge_without_push(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Broken pyright → blocked outcome, reason starts with prefix, push NOT called."""
        worktree = (await git_ops.create_worktree('mw-pyright-broken')).path
        (worktree / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add mod.py')

        mc = _make_module_config(prefix='subpkg', type_check_command='pyright src/')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='pushed')
        with (
            patch.object(git_ops, 'push_main', push_mock),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                _mock_scoped_verify_pass(),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                _mock_pyright_broken('subpkg'),
            ),
        ):
            req = _make_merge_request(
                'mw-pyright-broken', 'mw-pyright-broken', worktree, config,
                module_configs=[mc],
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX)
        assert 'subpkg' in outcome.reason
        # push_main must NOT be called on the broken path
        push_mock.assert_not_awaited()

    async def test_clean_pyright_allows_merge_to_succeed(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Clean pyright → 'done' outcome (same as no-pyright path)."""
        worktree = (await git_ops.create_worktree('mw-pyright-clean')).path
        (worktree / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add mod.py')

        mc = _make_module_config(prefix='subpkg', type_check_command='pyright src/')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with (
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                _mock_scoped_verify_pass(),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                _mock_pyright_clean(),
            ),
        ):
            req = _make_merge_request(
                'mw-pyright-clean', 'mw-pyright-clean', worktree, config,
                module_configs=[mc],
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert outcome.status == 'done'

    async def test_broken_pyright_emits_post_merge_pyright_broken_event(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """post_merge_pyright_broken merge_attempt event emitted on broken path."""
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        worktree = (await git_ops.create_worktree('mw-pyright-event')).path
        (worktree / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add mod.py')

        mc = _make_module_config(prefix='subpkg', type_check_command='pyright src/')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with (
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                _mock_scoped_verify_pass(),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                _mock_pyright_broken('subpkg'),
            ),
        ):
            req = _make_merge_request(
                'mw-pyright-event', 'mw-pyright-event', worktree, config,
                module_configs=[mc],
            )
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') FROM events "
            "WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        outcomes = [r[0] for r in rows]
        assert 'post_merge_pyright_broken' in outcomes, (
            f'Expected post_merge_pyright_broken event, got: {outcomes!r}'
        )


# ---------------------------------------------------------------------------
# SpeculativeMergeWorker._verifier_loop call-site integration tests  (step-7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculativeMergeWorkerPyrightCallSite:
    """Integration: SpeculativeMergeWorker._verifier_loop calls _check_post_merge_pyright."""

    async def test_broken_pyright_blocks_speculative_merge_without_push(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Broken pyright in speculative path → blocked outcome, reason starts with prefix, push NOT called."""
        worktree = (await git_ops.create_worktree('smw-pyright-broken')).path
        (worktree / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add mod.py')

        mc = _make_module_config(prefix='subpkg', type_check_command='pyright src/')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='pushed')
        with (
            patch.object(git_ops, 'push_main', push_mock),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                _mock_scoped_verify_pass(),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                _mock_pyright_broken('subpkg'),
            ),
        ):
            req = _make_merge_request(
                'smw-pyright-broken', 'smw-pyright-broken', worktree, config,
                module_configs=[mc],
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX)
        assert 'subpkg' in outcome.reason
        push_mock.assert_not_awaited()

    async def test_clean_pyright_allows_speculative_merge(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Clean pyright in speculative path → 'done' outcome."""
        worktree = (await git_ops.create_worktree('smw-pyright-clean')).path
        (worktree / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add mod.py')

        mc = _make_module_config(prefix='subpkg', type_check_command='pyright src/')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with (
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                _mock_scoped_verify_pass(),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                _mock_pyright_clean(),
            ),
        ):
            req = _make_merge_request(
                'smw-pyright-clean', 'smw-pyright-clean', worktree, config,
                module_configs=[mc],
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert outcome.status == 'done'

    async def test_broken_pyright_cleans_up_merge_worktree(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Broken pyright → merge_wt is cleaned up (cleanup_merge_worktree awaited)."""
        worktree = (await git_ops.create_worktree('smw-pyright-cleanup')).path
        (worktree / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add mod.py')

        mc = _make_module_config(prefix='subpkg', type_check_command='pyright src/')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        original_cleanup = git_ops.cleanup_merge_worktree
        cleanup_spy = AsyncMock(wraps=original_cleanup)
        with (
            patch.object(git_ops, 'cleanup_merge_worktree', cleanup_spy),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                _mock_scoped_verify_pass(),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                _mock_pyright_broken('subpkg'),
            ),
        ):
            req = _make_merge_request(
                'smw-pyright-cleanup', 'smw-pyright-cleanup', worktree, config,
                module_configs=[mc],
            )
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert cleanup_spy.await_count >= 1, (
            'cleanup_merge_worktree must be awaited on the broken pyright path'
        )
