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
from unittest.mock import AsyncMock, patch

import pytest
from _merge_lane_fakes import FakeVerifier, make_lane
from _serial_merge_worker import MergeWorker

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    PostMergePyrightResult,
    _check_post_merge_pyright,
    _run_unscoped_typechecks,
)
from orchestrator.merge_types import QueuedBranch

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
            outcome = await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='pyright-clean', max_attempts=1,
            )
            advanced_sha = outcome.advanced_sha or merge_result.merge_commit
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
            outcome = await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='pyright-broken', max_attempts=1,
            )
            advanced_sha = outcome.advanced_sha or merge_result.merge_commit
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
# Stand-in type_check_commands and config shims for the real-condition rows
# ---------------------------------------------------------------------------

# Outlives any test-sized verify budget; paired with _impatient() below to
# produce a genuine timeout rather than a mocked one.
_TYPE_CMD_HANGS = 'python3 -c "import time; time.sleep(60)"'


def _type_cmd_green_then_red(counter: Path) -> str:
    """A type_check_command green on its first run and red on every later one.

    Models what the POST-advance gate exists to catch: a type-check that
    passed in the merge worktree and fails once main carries the merge.  The
    pre-advance gate consumes the green run, the post-advance check sees the
    red one.  The counter lives outside the worktree so the fresh worktree
    created at the advanced SHA still sees it.
    """
    return (
        'python3 -c "'
        'import sys, pathlib; '
        f"c = pathlib.Path('{counter}'); "
        'first = not c.exists(); '
        "c.write_text('ran'); "
        'sys.exit(0) if first else '
        "(sys.stderr.write('synthetic type error\\n'), sys.exit(1))"
        '"'
    )


def _impatient(config: OrchestratorConfig, secs: float = 1.0) -> OrchestratorConfig:
    """*config* with a per-command verify budget short enough to trip in a test."""
    return config.model_copy(update={
        'verify_command_timeout_secs': secs,
        'verify_cold_command_timeout_secs': secs,
        'merge_verify_cold_command_timeout_secs': secs,
    })


def _merge_worktrees(git_ops: GitOps) -> list[Path]:
    """The merge worktrees on disk — the public trace of create/cleanup."""
    return sorted(git_ops.worktree_base.glob('_merge-*'))


async def _land(git_ops: GitOps, worktree: Path, branch: str) -> str:
    """Merge *branch* and advance main, returning the advanced SHA.

    Leaves no merge worktree behind, so a later ``_merge_worktrees`` reading
    reports only what the code under test created.
    """
    merge_result = await git_ops.merge_to_main(worktree, branch)
    assert merge_result.success
    assert merge_result.merge_commit is not None
    assert merge_result.merge_worktree is not None
    try:
        outcome = await git_ops.advance_main(
            merge_result.merge_commit, merge_result.merge_worktree,
            branch=branch, max_attempts=1,
        )
        return outcome.advanced_sha or merge_result.merge_commit
    finally:
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)


# ---------------------------------------------------------------------------
# _check_post_merge_pyright — fail-open edge cases, over real git
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckPostMergePyrightFailOpen:
    """The gate never blocks a landed merge on a hang or an infra error.

    Each row produces the condition for real — a type-check that outruns its
    budget, a SHA no worktree can be created at, a type-check that exits
    non-zero — and observes the worktree bookkeeping as the merge worktrees
    left on disk.
    """

    async def test_a_hanging_type_check_fails_open(
        self, git_ops: GitOps, config: OrchestratorConfig, caplog,
    ):
        """A type-check that outruns its budget → not broken, warning logged."""
        advanced_sha = await git_ops.get_main_sha()
        mc = _make_module_config(prefix='pkg', type_check_command=_TYPE_CMD_HANGS)

        with caplog.at_level(logging.WARNING):
            result = await _check_post_merge_pyright(
                advanced_sha, git_ops, _impatient(config), [mc], task_id='timeout-test',
            )

        assert result.broken is False
        assert result.failing_subprojects == []
        assert result.timed_out_subprojects == ['pkg']
        assert any('timed out' in r.message.lower() for r in caplog.records)
        assert _merge_worktrees(git_ops) == []

    async def test_a_sha_no_worktree_can_be_created_at_fails_open(
        self, git_ops: GitOps, config: OrchestratorConfig, caplog,
    ):
        """An unresolvable SHA makes worktree creation raise → not broken."""
        mc = _make_module_config(prefix='pkg')

        with caplog.at_level(logging.WARNING):
            result = await _check_post_merge_pyright(
                'f' * 40, git_ops, config, [mc], task_id='infra-error-test',
            )

        assert result.broken is False
        assert result.failing_subprojects == []
        assert any('infra error' in r.message.lower() for r in caplog.records)
        # Nothing was created, so nothing is left to clean up.
        assert _merge_worktrees(git_ops) == []

    async def test_the_merge_worktree_is_removed_even_when_the_check_fails(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Cleanup runs in ``finally``, so the failing path leaks no worktree."""
        wt = (await git_ops.create_worktree('pyright-cleanup')).path
        (wt / '.BROKEN_UNION').write_text('broken\n')
        await git_ops.commit(wt, 'Add .BROKEN_UNION marker')
        advanced_sha = await _land(git_ops, wt, 'pyright-cleanup')

        result = await _check_post_merge_pyright(
            advanced_sha, git_ops, config, [_make_module_config(prefix='pkg')],
            task_id='cleanup-test',
        )

        assert result.broken is True
        assert _merge_worktrees(git_ops) == []


# ---------------------------------------------------------------------------
# _run_unscoped_typechecks — classification, over real type-check commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunUnscopedTypechecks:
    """The per-module classification loop, over real commands.

    Every row runs the real ``type_check_command`` in a caller-supplied
    worktree.  That the helper creates and cleans up no worktree of its own
    is observed as the merge-worktree set staying empty across the call.
    """

    async def test_a_failing_type_check_is_broken(
        self, git_ops: GitOps, git_repo: Path, config: OrchestratorConfig,
    ):
        """Exits non-zero without timing out → genuine failure, detail captured."""
        (git_repo / '.BROKEN_UNION').write_text('broken\n')

        result = await _run_unscoped_typechecks(
            git_repo, config, [_make_module_config(prefix='pkg')],
            block_on_timeout=False, task_id='fail-test',
        )

        assert result.broken is True
        assert result.failing_subprojects == ['pkg']
        assert 'synthetic type error' in result.detail
        # The helper operates on the worktree it is given; it creates none.
        assert _merge_worktrees(git_ops) == []

    async def test_a_hanging_type_check_fails_open_when_timeouts_do_not_block(
        self, git_repo: Path, config: OrchestratorConfig,
    ):
        """block_on_timeout=False → timed out but NOT failing (post-advance shape)."""
        mc = _make_module_config(prefix='pkg', type_check_command=_TYPE_CMD_HANGS)

        result = await _run_unscoped_typechecks(
            git_repo, _impatient(config), [mc],
            block_on_timeout=False, task_id='timeout-open-test',
        )

        assert result.broken is False
        assert result.failing_subprojects == []
        assert result.timed_out_subprojects == ['pkg']

    async def test_a_hanging_type_check_is_broken_when_timeouts_block(
        self, git_repo: Path, config: OrchestratorConfig,
    ):
        """block_on_timeout=True → timed out AND failing (pre-advance shape)."""
        mc = _make_module_config(prefix='pkg', type_check_command=_TYPE_CMD_HANGS)

        result = await _run_unscoped_typechecks(
            git_repo, _impatient(config), [mc],
            block_on_timeout=True, task_id='timeout-closed-test',
        )

        assert result.broken is True
        assert result.failing_subprojects == ['pkg']
        assert result.timed_out_subprojects == ['pkg']

    async def test_a_passing_type_check_is_clean(
        self, git_repo: Path, config: OrchestratorConfig,
    ):
        """Exits zero → both lists empty, neither broken nor timed out."""
        result = await _run_unscoped_typechecks(
            git_repo, config, [_make_module_config(prefix='pkg')],
            block_on_timeout=True, task_id='clean-test',
        )

        assert result.broken is False
        assert result.timed_out is False
        assert result.failing_subprojects == []
        assert result.timed_out_subprojects == []

    async def test_a_module_without_a_type_check_command_is_skipped(
        self, git_repo: Path, config: OrchestratorConfig,
    ):
        """The command-less module is not run: on a red tree, only the other fails."""
        (git_repo / '.BROKEN_UNION').write_text('broken\n')

        result = await _run_unscoped_typechecks(
            git_repo, config,
            [
                _make_module_config(prefix='no-cmd', type_check_command=None),
                _make_module_config(prefix='pkg'),
            ],
            block_on_timeout=True, task_id='no-cmd-test',
        )

        assert result.failing_subprojects == ['pkg']


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
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=module_configs or [],
        config=config,
        result=future,
    )


async def _drain(worker, worker_task: asyncio.Task) -> None:
    """Stop *worker* and let its run loop finish."""
    await worker.stop()
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task


@pytest.mark.asyncio
class TestMergeWorkerPyrightCallSite:
    """MergeWorker._do_merge runs the post-advance type-check gate.

    The serial worker takes no ``VerifyPort``, so BOTH the pre-advance gate
    and the post-advance check run the real ``type_check_command``.  The
    broken rows therefore use the green-then-red stand-in: the pre-advance
    gate consumes the green run and the post-advance check sees the red one,
    which is exactly the condition this gate exists for.
    """

    async def test_broken_pyright_blocks_merge_without_push(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Broken pyright → blocked outcome, reason starts with prefix, push NOT called."""
        worktree = (await git_ops.create_worktree('mw-pyright-broken')).path
        (worktree / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add mod.py')

        mc = _make_module_config(
            prefix='subpkg',
            type_check_command=_type_cmd_green_then_red(tmp_path / 'mw-broken.count'),
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='pushed')
        with patch.object(git_ops, 'push_main', push_mock):
            req = _make_merge_request(
                'mw-pyright-broken', 'mw-pyright-broken', worktree, config,
                module_configs=[mc],
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        await _drain(worker, worker_task)

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

        mc = _make_module_config(prefix='subpkg')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_merge_request(
            'mw-pyright-clean', 'mw-pyright-clean', worktree, config,
            module_configs=[mc],
        )
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=30)

        await _drain(worker, worker_task)

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

        mc = _make_module_config(
            prefix='subpkg',
            type_check_command=_type_cmd_green_then_red(tmp_path / 'mw-event.count'),
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        req = _make_merge_request(
            'mw-pyright-event', 'mw-pyright-event', worktree, config,
            module_configs=[mc],
        )
        await queue.put(req)
        await asyncio.wait_for(req.result, timeout=30)

        await _drain(worker, worker_task)

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
# MergeLane._verifier_loop call-site integration tests  (step-7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeLanePyrightCallSite:
    """The lane's verifier loop runs the post-advance type-check gate.

    The lane threads its injected ``VerifyPort`` into post-merge verify, so
    ``FakeVerifier`` supplies a passing scoped verify and a clean PRE-advance
    unscoped gate.  The POST-advance check reaches back to the module
    function and is not a port call, so it runs the real command against the
    landed tree — which is what these rows exercise.
    """

    async def test_broken_pyright_blocks_lane_merge_without_push(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Broken pyright in the lane → blocked outcome, prefix in reason, no push."""
        worktree = (await git_ops.create_worktree('smw-pyright-broken')).path
        (worktree / '.BROKEN_UNION').write_text('broken\n')
        await git_ops.commit(worktree, 'Add .BROKEN_UNION marker')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = make_lane(git_ops, queue, verifier=FakeVerifier())
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='pushed')
        with patch.object(git_ops, 'push_main', push_mock):
            req = _make_merge_request(
                'smw-pyright-broken', 'smw-pyright-broken', worktree, config,
                module_configs=[_make_module_config(prefix='subpkg')],
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        await _drain(worker, worker_task)

        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX)
        assert 'subpkg' in outcome.reason
        push_mock.assert_not_awaited()

    async def test_clean_pyright_allows_lane_merge(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Clean pyright in the lane → 'done' outcome."""
        worktree = (await git_ops.create_worktree('smw-pyright-clean')).path
        (worktree / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add mod.py')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = make_lane(git_ops, queue, verifier=FakeVerifier())
        worker_task = asyncio.create_task(worker.run())

        req = _make_merge_request(
            'smw-pyright-clean', 'smw-pyright-clean', worktree, config,
            module_configs=[_make_module_config(prefix='subpkg')],
        )
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=30)

        await _drain(worker, worker_task)

        assert outcome.status == 'done'

    async def test_broken_pyright_leaves_no_merge_worktree_behind(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Broken pyright → every merge worktree the attempt created is gone."""
        worktree = (await git_ops.create_worktree('smw-pyright-cleanup')).path
        (worktree / '.BROKEN_UNION').write_text('broken\n')
        await git_ops.commit(worktree, 'Add .BROKEN_UNION marker')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = make_lane(git_ops, queue, verifier=FakeVerifier())
        worker_task = asyncio.create_task(worker.run())

        req = _make_merge_request(
            'smw-pyright-cleanup', 'smw-pyright-cleanup', worktree, config,
            module_configs=[_make_module_config(prefix='subpkg')],
        )
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=30)

        await _drain(worker, worker_task)

        assert outcome.status == 'blocked'
        assert _merge_worktrees(git_ops) == []
