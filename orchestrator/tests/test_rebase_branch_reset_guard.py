"""Regression tests for the requeue/inter-iteration rebase branch-reset guard
(RCA + fix, task 2403).

Incident: during merge-train churn around a frozen-prefix merge tip
(d48743f90a), the orchestrator's requeue/inter-iteration rebase path
silently RESET two unrelated task branches (task/2261, task/2223) to that
tip with ZERO commits ahead of main, wiping each task's committed work.
Reflog signature: ``branch: Created from main`` then ``rebase (finish):
... onto d48743f90a``. ``git rebase <ref>`` records the RESOLVED sha, so
"onto d48743f90a" only means main's tip WAS d48743 at rebase time — the
branch's own commits were dropped DURING the rebase, not that an unrelated
ref was deliberately chosen as --onto.

Fix: a mechanism-independent POST-CONDITION guard —
:meth:`GitOps.rebase_preserving_task_commits` — wraps the existing
``rebase_onto_main`` primitive with a pre/post "commits ahead of main"
check. If the wrapped rebase reports success but the branch's own commits
vanished (n_before > 0 and n_after == 0), the guard restores the
pre-rebase HEAD and raises :class:`BranchResetError` instead of silently
returning success.

This file starts RED (step-1): neither ``GitOps.rebase_preserving_task_commits``
nor ``BranchResetError`` exist yet on ``orchestrator.git_ops`` — the import
below fails collection until step-2 implements them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import BranchResetError, GitOps, WorktreeInfo, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Fixtures — copied from test_git_ops.py's git_repo/git_config/git_ops trio
# (test_git_ops.py:35/114/126) so this file builds real temp git repos
# without importing test-internal helpers from another test module.
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
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


async def _assert_no_rebase_in_progress(wt_path: Path) -> None:
    """Assert the worktree has no rebase in progress (copied from
    test_git_ops.py's helper of the same name)."""
    _, gitdir_str, _ = await _run(['git', 'rev-parse', '--git-dir'], cwd=wt_path)
    gitdir = Path(gitdir_str.strip())
    if not gitdir.is_absolute():
        gitdir = wt_path / gitdir
    rebase_merge = gitdir / 'rebase-merge'
    assert not rebase_merge.exists(), (
        f'rebase-merge directory {rebase_merge} should not exist — '
        'rebase was not properly aborted'
    )


async def _commits_over_main(worktree: Path) -> int:
    _, out, _ = await _run(
        ['git', 'rev-list', '--count', 'main..HEAD'], cwd=worktree,
    )
    return int(out.strip())


async def _head_sha(worktree: Path) -> str:
    _, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
    return out.strip()


async def _commit_unique_work(worktree: Path, filename: str = 'feature.txt') -> None:
    (worktree / filename).write_text('unique task work\n')
    await _run(['git', 'add', '-A'], cwd=worktree)
    await _run(['git', 'commit', '-m', 'task: add feature'], cwd=worktree)


async def _wipe_via_reset(worktree: Path, onto: str | None = None) -> bool:
    """Fake ``rebase_onto_main`` replacement: collapses the branch straight
    onto *onto* (default 'main') via ``git reset --hard`` and reports
    success — used to engineer a deterministic zero-commit wipe without
    depending on git-version-specific patch-id-drop heuristics during a real
    rebase. The guard's contract is mechanism-independent (see module
    docstring), so simulating the "rebase reported success but the branch's
    commits are gone" observable is sufficient to exercise it.
    """
    target = onto if onto is not None else 'main'
    await _run(['git', 'reset', '--hard', target], cwd=worktree)
    return True


# ---------------------------------------------------------------------------
# step-1: GitOps.rebase_preserving_task_commits + BranchResetError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRebasePreservingTaskCommitsGuard:
    async def test_guard_raises_and_restores_on_zero_commit_wipe(
        self, git_ops: GitOps, git_repo: Path, monkeypatch,
    ):
        """A rebase that collapses the branch to 0 commits over main raises
        BranchResetError and restores the pre-rebase HEAD (work preserved)."""
        wt_info = await git_ops.create_worktree('wipe-me')
        wt = wt_info.path
        await _commit_unique_work(wt)

        pre_head = await _head_sha(wt)
        assert await _commits_over_main(wt) == 1

        # Engineer the wipe: rebase_onto_main "succeeds" but actually
        # collapses HEAD onto main via reset (git_ops.py primitive is
        # patched, not `git rebase` behavior itself — see _wipe_via_reset).
        monkeypatch.setattr(git_ops, 'rebase_onto_main', _wipe_via_reset)

        with pytest.raises(BranchResetError):
            await git_ops.rebase_preserving_task_commits(wt)

        assert await _head_sha(wt) == pre_head, 'pre-rebase HEAD must be restored'
        assert await _commits_over_main(wt) == 1, 'task commit must survive'

    async def test_guard_happy_path_retains_commits(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """A clean rebase onto an advanced main retains the branch's own
        commit and returns True without raising."""
        wt_info = await git_ops.create_worktree('happy')
        wt = wt_info.path
        await _commit_unique_work(wt)

        # Advance main with an UNRELATED sibling commit.
        (git_repo / 'sibling.txt').write_text('sibling change\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'sibling change'], cwd=git_repo)

        result = await git_ops.rebase_preserving_task_commits(wt)

        assert result is True
        assert await _commits_over_main(wt) == 1

    async def test_guard_passthrough_on_real_conflict_returns_false(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """A genuine rebase conflict makes the guard return False (not
        raise); the branch is left clean on its old base."""
        wt_info = await git_ops.create_worktree('conflict')
        wt = wt_info.path
        # Branch and main both add fileX.txt with different content ->
        # add/add conflict on rebase (mirrors test_git_ops.py's
        # TestRebaseOntoArbitraryRef conflict pattern).
        (wt / 'fileX.txt').write_text('branch version\n')
        await _run(['git', 'add', '-A'], cwd=wt)
        await _run(['git', 'commit', '-m', 'task: add fileX'], cwd=wt)

        (git_repo / 'fileX.txt').write_text('main version\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'main: add fileX'], cwd=git_repo)

        pre_head = await _head_sha(wt)

        result = await git_ops.rebase_preserving_task_commits(wt)

        assert result is False
        assert await _head_sha(wt) == pre_head, 'branch must be left on its old base'
        await _assert_no_rebase_in_progress(wt)

    async def test_guard_noop_when_branch_had_zero_commits_over_main(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """A branch with no commits of its own (pure fast-forward) returns
        True and never raises — no false positive on an empty task."""
        wt_info = await git_ops.create_worktree('empty')
        wt = wt_info.path
        assert await _commits_over_main(wt) == 0

        (git_repo / 'sibling.txt').write_text('sibling change\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'sibling change'], cwd=git_repo)

        result = await git_ops.rebase_preserving_task_commits(wt)

        assert result is True
        assert await _commits_over_main(wt) == 0

    async def test_guard_detects_wipe_of_own_delta_measured_against_onto(
        self, git_ops: GitOps, git_repo: Path, monkeypatch,
    ):
        """Amendment review finding #1: for a stacked-train-style rebase
        (``onto=<predecessor tip>``, itself ahead of main), a wipe of just
        the branch's OWN delta must still be detected even though the
        predecessor's commits keep commits-over-MAIN > 0. Measuring only
        against main would hide this (n_after-over-main stays 1, from the
        surviving predecessor commit) — the guard must measure against
        ``onto`` instead."""
        # Predecessor: 1 commit ahead of main.
        pred_wt_info = await git_ops.create_worktree('pred')
        pred_wt = pred_wt_info.path
        await _commit_unique_work(pred_wt, filename='pred.txt')
        predecessor_sha = await _head_sha(pred_wt)
        assert await _commits_over_main(pred_wt) == 1

        # Successor built ON TOP of the predecessor's tip (mirrors a
        # stacked train member): main + predecessor's commit + its own.
        succ_wt = git_repo / '.worktrees' / 'succ'
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/succ', str(succ_wt), predecessor_sha],
            cwd=git_repo,
        )
        await _commit_unique_work(succ_wt, filename='succ.txt')
        pre_head = await _head_sha(succ_wt)
        assert await _commits_over_main(succ_wt) == 2, 'predecessor + own commit'

        # Engineer a wipe of JUST the successor's own delta: collapse HEAD
        # back to the predecessor's tip, as a rebase-onto-predecessor that
        # dropped the branch's own commit would look from the outside.
        async def _wipe_own_delta_only(worktree: Path, onto: str | None = None) -> bool:
            await _run(['git', 'reset', '--hard', predecessor_sha], cwd=worktree)
            return True

        monkeypatch.setattr(git_ops, 'rebase_onto_main', _wipe_own_delta_only)

        with pytest.raises(BranchResetError):
            await git_ops.rebase_preserving_task_commits(succ_wt, onto=predecessor_sha)

        assert await _head_sha(succ_wt) == pre_head, 'own commit must be restored'
        assert await _commits_over_main(succ_wt) == 2, 'both commits must survive'

    async def test_guard_returns_true_on_patch_id_dedup_not_wipe(
        self, git_ops: GitOps, git_repo: Path, monkeypatch,
    ):
        """Amendment review finding #2: a total wipe (n_after == 0) that is
        fully explained by the branch's pre-rebase commit(s) already being
        patch-id-equivalent to a commit landed at the baseline (e.g. a
        re-dispatched task whose work previously merged) is a legitimate
        dedup, not a destructive wipe — the guard must return True without
        raising or touching HEAD."""
        wt_info = await git_ops.create_worktree('dedup')
        wt = wt_info.path
        await _commit_unique_work(wt, filename='feature.txt')
        assert await _commits_over_main(wt) == 1

        # Simulate the task's work having already landed on main — same
        # patch content, different (later, independently-created) commit.
        # `git add feature.txt` specifically (not `-A`): the worktree
        # created above already lives under git_repo/.worktrees/dedup, and
        # a blanket `-A` here would also stage it as a gitlink, changing
        # this commit's diff/patch-id and defeating the scenario.
        (git_repo / 'feature.txt').write_text('unique task work\n')
        await _run(['git', 'add', 'feature.txt'], cwd=git_repo)
        await _run(
            ['git', 'commit', '-m', 'task: add feature (already landed)'],
            cwd=git_repo,
        )

        # Engineer the "rebase succeeded but collapsed to 0 commits over
        # main" observable exactly like the other wipe tests — this is
        # what a real patch-id-drop rebase would also produce, but without
        # depending on git-version-specific auto-drop behavior (see
        # _wipe_via_reset's docstring).
        monkeypatch.setattr(git_ops, 'rebase_onto_main', _wipe_via_reset)

        result = await git_ops.rebase_preserving_task_commits(wt)

        assert result is True, (
            'a total wipe that is fully patch-id-dedupable against main '
            'must not raise — it is a dedup, not a destructive wipe'
        )
        assert await _commits_over_main(wt) == 0, (
            'no restore should happen for a genuine dedup — HEAD stays as '
            'the rebase (here, simulated reset) left it'
        )

    async def test_guard_branch_reset_error_reports_failed_restore(
        self, git_ops: GitOps, git_repo: Path, monkeypatch,
    ):
        """Amendment review finding #3: if the recovery
        ``git reset --hard`` itself fails after a wipe is detected,
        ``BranchResetError.restore_ok`` is False and the message says so
        explicitly — it must never claim the work was restored when it
        wasn't."""
        wt_info = await git_ops.create_worktree('restore-fails')
        wt = wt_info.path
        await _commit_unique_work(wt)
        pre_head = await _head_sha(wt)
        assert await _commits_over_main(wt) == 1

        monkeypatch.setattr(git_ops, 'rebase_onto_main', _wipe_via_reset)

        import orchestrator.git_ops as git_ops_module

        real_run = git_ops_module._run

        async def _failing_restore_run(cmd: list[str], cwd: Path | None = None):
            if cmd[:3] == ['git', 'reset', '--hard'] and cmd[3:] == [pre_head]:
                return 1, '', 'simulated: recovery reset failed'
            return await real_run(cmd, cwd=cwd)

        monkeypatch.setattr(git_ops_module, '_run', _failing_restore_run)

        with pytest.raises(BranchResetError) as exc_info:
            await git_ops.rebase_preserving_task_commits(wt)

        assert exc_info.value.restore_ok is False
        assert 'FAILED TO RESTORE' in str(exc_info.value)
        # The (simulated-failed) restore never actually happened — HEAD is
        # left wherever the wipe left it, NOT back at pre_head.
        assert await _head_sha(wt) != pre_head


# ---------------------------------------------------------------------------
# step-3: the two git_ops requeue reuse sites route THROUGH the guard.
#
# Warm-lane helpers copied from test_git_ops.py (`_seed_default_warm_base`
# at :43, `_add_warm_lane_scripts` at :6636) so this file can drive
# `_reuse_warm_lane` without importing test-internal helpers from another
# test module.
# ---------------------------------------------------------------------------


def _seed_default_warm_base(repo: Path) -> None:
    """Pre-create the DEFAULT derived warm-lane base (task 2061) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK."""
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo."""
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)


@pytest.mark.asyncio
class TestRequeueReuseSitesRouteThroughGuard:
    """The cold-requeue (``create_worktree``) and live-requeue
    (``_reuse_warm_lane``) reuse paths must route through
    ``rebase_preserving_task_commits`` so a wipe raises BranchResetError
    and preserves the branch's commit instead of silently losing it.

    RED today (step-3): both sites still call the unguarded
    ``rebase_onto_main`` directly (rewired to the guard in step-4).
    """

    async def test_create_worktree_reuse_raises_on_wipe(
        self, git_ops: GitOps, git_repo: Path, monkeypatch,
    ):
        """A second create_worktree() call (the cold-requeue reuse block)
        that would zero the branch must raise BranchResetError and leave
        the branch's commit intact."""
        wt_info = await git_ops.create_worktree('requeue-wipe')
        wt = wt_info.path
        await _commit_unique_work(wt)
        assert await _commits_over_main(wt) == 1

        monkeypatch.setattr(git_ops, 'rebase_onto_main', _wipe_via_reset)

        with pytest.raises(BranchResetError):
            await git_ops.create_worktree('requeue-wipe')

        assert await _commits_over_main(wt) == 1, (
            'branch commit must survive the cold-requeue reuse rebase'
        )

    async def test_reuse_warm_lane_raises_on_wipe(
        self, git_repo: Path, monkeypatch,
    ):
        """_reuse_warm_lane (the live-requeue reuse path) that would zero
        the branch must raise BranchResetError and preserve the commit."""
        _seed_default_warm_base(git_repo)
        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        warm_config = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )
        warm_git_ops = GitOps(warm_config, git_repo, warm_lane_pool_size=1)

        info = await warm_git_ops.acquire_warm_lane('WR', start_ref)
        assert isinstance(info, WorktreeInfo), f'Acquire failed: {info!r}'
        lane = info.path

        await _commit_unique_work(lane, filename='wr_feature.txt')
        assert await _commits_over_main(lane) == 1

        monkeypatch.setattr(warm_git_ops, 'rebase_onto_main', _wipe_via_reset)

        with pytest.raises(BranchResetError):
            await warm_git_ops._reuse_warm_lane(lane, 'task/WR')

        assert await _commits_over_main(lane) == 1, (
            'branch commit must survive the live-requeue (warm-lane) reuse rebase'
        )


# ---------------------------------------------------------------------------
# step-5: workflow-level wiring — TaskWorkflow._inter_iteration_rebase must
# propagate BranchResetError (not swallow it to None), and TaskWorkflow.run()
# must route an escaping BranchResetError to a targeted human escalation
# (category='branch_reset', escalate_to_human=True) rather than the generic
# 'Workflow error:' steward-routed path.
#
# RED today (step-5): _inter_iteration_rebase still calls the unguarded
# ``rebase_onto_main`` directly (rewired to the guard in step-6), and
# run()'s shared exception handler has no isinstance(e, BranchResetError)
# branch yet (added in step-6).
# ---------------------------------------------------------------------------


# -- 5(a) harness — copied from test_verify_phase_rebase.py's real-git
# TaskWorkflow harness (_make_workflow @87, task_assignment fixture) so this
# file can drive the real _inter_iteration_rebase without importing
# test-internal helpers from another test module.


@pytest.fixture
def orchestrator_config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_verify_attempts=2,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='2403',
        task={
            'id': '2403', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


def _make_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    worktree: Path,
) -> tuple[TaskWorkflow, TaskArtifacts]:
    """Wire a minimal TaskWorkflow with all heavy collaborators mocked.

    Copied from test_verify_phase_rebase.py's ``_make_workflow`` (@87).
    """
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(
        worktree, TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
    )
    artifacts.init('2403', 'X', 'desc', base_commit='base-sha-old')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '2403', 'steps': []}
    workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug')  # type: ignore[attr-defined]
    from orchestrator.agents.invoke import AgentResult
    workflow._invoke = AsyncMock(  # type: ignore[method-assign]
        return_value=AgentResult(success=True, output=''),
    )
    workflow._get_head_commit = AsyncMock(return_value='head-sha')  # type: ignore[method-assign]
    return workflow, artifacts


@pytest.mark.asyncio
class TestInterIterationRebasePropagatesBranchReset:
    async def test_inter_iteration_rebase_propagates_branch_reset(
        self,
        orchestrator_config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        git_repo: Path,
        monkeypatch,
    ):
        """A wipe detected while rebasing during a real
        ``_inter_iteration_rebase`` call must RAISE BranchResetError
        (rather than being swallowed to ``None`` like an ordinary rebase
        conflict) and the branch's committed work must survive."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        await _commit_unique_work(wt)
        assert await _commits_over_main(wt) == 1

        workflow, artifacts = _make_workflow(
            orchestrator_config, git_ops, task_assignment, wt,
        )
        artifacts.update_base_commit(wt_info.base_commit)

        # Advance main so _inter_iteration_rebase actually triggers a rebase.
        (git_repo / 'sibling.txt').write_text('sibling change\n')
        await _run(['git', 'add', 'sibling.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'sibling change'], cwd=git_repo)

        # Engineer the wipe via the same monkeypatch used by the git_ops-level
        # guard tests above.
        monkeypatch.setattr(git_ops, 'rebase_onto_main', _wipe_via_reset)

        with pytest.raises(BranchResetError):
            await workflow._inter_iteration_rebase()

        assert await _commits_over_main(wt) == 1, (
            'task commit must survive the inter-iteration rebase wipe'
        )


# -- 5(b) harness — copied from test_workflow.py's mocked run()-exception-
# routing harness (_make_wip_conflict_workflow @1972, TestWorktreeConflict-
# ErrorRouting @2029) so this file can drive run()'s top-level exception
# routing without a full plan/execute setup or importing test-internal
# helpers from another test module.


def _make_branch_reset_workflow(*, task_id: str = '2403') -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'Test task', 'description': 'Test desc',
        'metadata': {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    # run()'s SM-2 exit check reads this back — _mark_blocked is stubbed
    # below (its real body, which would persist 'blocked', never runs), so
    # this must reflect the forced BLOCKED outcome directly.
    scheduler.get_status = AsyncMock(return_value='blocked')
    scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': {}})

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='SHA-A')

    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,
    )

    wf.worktree = Path('/tmp/fake-worktree-2403')
    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))
    wf.plan = {'task_id': task_id, 'steps': []}
    wf.initial_plan = {'task_id': task_id, 'steps': []}

    # Stub _mark_blocked — assert on how run() CALLS it, mirroring
    # TestWorktreeConflictErrorRouting rather than re-testing _mark_blocked's
    # own escalation-filing behavior (covered elsewhere).
    wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)

    return wf


@pytest.mark.asyncio
class TestBranchResetErrorRouting:
    """run(): a BranchResetError escaping the execute/verify path routes to
    _mark_blocked(category='branch_reset', escalate_to_human=True) — not
    the generic 'Workflow error:' handler (task 2403)."""

    async def test_run_routes_branch_reset_to_blocked_escalation(self):
        wf = _make_branch_reset_workflow()
        worktree = wf.worktree
        assert worktree is not None

        async def fake_setup(*args, **kwargs):
            pass

        with (
            patch.object(wf, '_setup_worktree_and_artifacts', side_effect=fake_setup),
            patch.object(wf, '_recover_if_already_merged',
                         new=AsyncMock(return_value=None)),
            patch.object(wf, '_check_branch_on_main',
                         new=AsyncMock(return_value=None)),
            patch.object(
                wf, '_execute_verify_review_loop',
                side_effect=BranchResetError(worktree, None, 'pre-rebase-sha', 1),
            ),
        ):
            outcome = (await wf.run()).outcome

        assert outcome == WorkflowOutcome.BLOCKED

        wf._mark_blocked.assert_awaited_once()  # type: ignore[attr-defined]
        call_kwargs = wf._mark_blocked.await_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs.get('category') == 'branch_reset', (
            f"Expected _mark_blocked(category='branch_reset') but got "
            f"category={call_kwargs.get('category')!r}. A BranchResetError "
            f"must not fall through to the generic 'Workflow error:' handler."
        )
        assert call_kwargs.get('escalate_to_human') is True, (
            'branch_reset block must set escalate_to_human=True (L1, skips steward)'
        )
