"""Tests for reconciling a done plan step's stale ``commit`` after an
inter-iteration / warm-lane rebase orphans it (task 2386).

``TaskWorkflow._detect_tip_wip_commits`` (task 2051) surfaces WIP
safety-commits at HEAD for still-PENDING steps, but explicitly dedups any
sha already recorded as a DONE step's ``commit`` — so a done step whose
recorded commit was rewritten/orphaned by ``_inter_iteration_rebase``
(workflow.py) or a warm-lane/requeue rebase (git_ops.py) is invisible to
that detector. This suite covers the new reconciliation machinery that
fixes that:

  - GitOps.get_commit_changed_files (git_ops.py)
  - TaskWorkflow._reconcile_done_step_commits (workflow.py)
  - _execute_iterations wiring the reconciler in before the WIP detector
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# step-1 RED: GitOps.get_commit_changed_files
# ---------------------------------------------------------------------------
#
# Real temp git repo pattern, mirroring test_harness_wip_step_detection.py's
# git_repo/config/git_ops fixtures.


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    # Mirror this repo's root .gitignore `.task/` entry (see
    # GitOps.commit's / GitOps.has_uncommitted_work's docstrings): without
    # this, `git add -A` in git_ops.commit() would stage .task/metadata.json
    # into the test's real commits, and the reconcile tests' `git reset
    # --hard <base>` (simulating a rebase orphaning a commit) would then
    # wipe .task/ along with it — silently breaking read_base_commit().
    (repo / '.gitignore').write_text('.task/\n')
    (repo / 'lib.py').write_text('x = 1\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.mark.asyncio
class TestGetCommitChangedFiles:
    async def test_normal_commit_returns_changed_files_vs_parent(self, git_repo, git_ops):
        """A normal commit's changed-file set is computed vs its sole parent."""
        (git_repo / 'a.txt').write_text('a\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'add a'], cwd=git_repo)
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        sha = sha.strip()

        result = await git_ops.get_commit_changed_files(sha)

        assert result == ['a.txt']

    async def test_root_commit_returns_its_own_files(self, git_repo, git_ops):
        """A ROOT commit (no parent) must still return the files it introduced.

        Plain `git diff-tree <sha>` (without --root) shows nothing for a
        root commit since it has no parent to diff against — the
        implementation must handle this so an orphaned original commit that
        happens to be the repo's first commit is not silently treated as
        unresolvable.
        """
        _, root_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        root_sha = root_sha.strip()

        result = await git_ops.get_commit_changed_files(root_sha)

        # The root commit ("Initial") introduces both the fixture's
        # .gitignore and lib.py — see _init_repo.
        assert sorted(result) == ['.gitignore', 'lib.py']

    async def test_nonexistent_sha_returns_empty_list(self, git_repo, git_ops):
        """A garbage/nonexistent SHA returns [] defensively, never raises."""
        result = await git_ops.get_commit_changed_files(
            'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
        )

        assert result == []

    async def test_multi_file_commit_returns_all_files(self, git_repo, git_ops):
        """A commit touching several files returns the full changed-file set."""
        (git_repo / 'b.txt').write_text('b\n')
        (git_repo / 'c.txt').write_text('c\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'add b and c'], cwd=git_repo)
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        sha = sha.strip()

        result = await git_ops.get_commit_changed_files(sha)

        assert sorted(result) == ['b.txt', 'c.txt']


# ---------------------------------------------------------------------------
# step-3 RED: TaskWorkflow._reconcile_done_step_commits (core / auto-reconcile)
# ---------------------------------------------------------------------------
#
# Real-git _make_workflow pattern, mirroring test_harness_wip_step_detection.py
# — real GitOps + a real worktree created via git_ops.create_worktree, heavy
# collaborators (scheduler/briefing/mcp) mocked.


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '',
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
    *,
    escalation_queue=None,
) -> tuple[TaskWorkflow, TaskArtifacts]:
    """Wire a minimal TaskWorkflow with heavy collaborators mocked.

    Mirrors the pattern in test_harness_wip_step_detection.py._make_workflow.
    """
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
        escalation_queue=escalation_queue,
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': [], 'prerequisites': []}
    return workflow, artifacts


def _write_done_step_plan(artifacts: TaskArtifacts, step_id: str, commit: str) -> dict:
    """Persist a single-step plan (status='done', the given commit) and
    return the re-read plan so ``workflow.plan`` mirrors what's on disk —
    required because ``update_step_status`` reads/writes plan.json directly,
    independent of any in-memory ``workflow.plan`` the caller also sets."""
    plan = {
        'task_id': '42',
        'title': 'X',
        'analysis': 'A',
        'prerequisites': [],
        'steps': [
            {'id': step_id, 'type': 'impl', 'status': 'done', 'commit': commit},
        ],
    }
    artifacts.write_plan(plan)
    return artifacts.read_plan()


def _write_steps_plan(artifacts: TaskArtifacts, steps: list[dict]) -> dict:
    """Like :func:`_write_done_step_plan`, but for multiple pre-built step
    dicts — used by the multi-orphaned-done-step coverage below."""
    plan = {
        'task_id': '42',
        'title': 'X',
        'analysis': 'A',
        'prerequisites': [],
        'steps': steps,
    }
    artifacts.write_plan(plan)
    return artifacts.read_plan()


@pytest.mark.asyncio
class TestReconcileDoneStepCommits:
    async def test_orphaned_done_step_commit_matching_wip_tip_is_reconciled(
        self, config, git_ops, task_assignment,
    ):
        """HAPPY PATH: a done step's commit was orphaned (rebase reset the
        branch back to base), but its exact file(s) reappear in a WIP
        safety-commit now sitting at HEAD -> auto-reconcile the step's
        commit to the WIP tip sha."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        # Step-1's original implementation commit.
        (wt / 'feature.py').write_text('original implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        # Simulate the rebase orphaning step_commit: reset hard to base, then
        # re-land the same file as a WIP safety-commit at HEAD (mirrors
        # _inter_iteration_rebase's squash-then-rebase sequence).
        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'feature.py').write_text('original implementation\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        await workflow._reconcile_done_step_commits()

        reconciled = artifacts.read_plan()
        assert reconciled['steps'][0]['commit'] == wip_sha, (
            f'Expected step-1 commit re-pointed to WIP tip {wip_sha}, '
            f"got {reconciled['steps'][0]['commit']}"
        )
        assert reconciled['steps'][0]['status'] == 'done'

    async def test_orphaned_commit_subset_of_larger_wip_run_is_reconciled(
        self, config, git_ops, task_assignment,
    ):
        """PARTIAL MATCH: the orphaned done-step commit's file set
        ({'feature.py'}) is a STRICT SUBSET of the tip WIP run's changed
        files ({'feature.py', 'other.py'}, i.e. the WIP run is a superset,
        not an exact match) -> still auto-reconciles. Exercises the
        subset-not-equality branch of the match predicate, which the
        exact-same-single-file happy path above doesn't distinguish from a
        stricter equality check."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'feature.py').write_text('original implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        # Orphan step_commit, then land a WIP commit that reintroduces
        # feature.py AND an unrelated additional file — the WIP run's files
        # are a strict superset of the orphaned commit's files.
        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'feature.py').write_text('original implementation\n')
        (wt / 'other.py').write_text('unrelated but also present\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        await workflow._reconcile_done_step_commits()

        reconciled = artifacts.read_plan()
        assert reconciled['steps'][0]['commit'] == wip_sha, (
            f'Expected step-1 commit re-pointed to WIP tip {wip_sha} even '
            f"though the WIP run is a strict superset, got "
            f"{reconciled['steps'][0]['commit']}"
        )
        assert reconciled['steps'][0]['status'] == 'done'

    async def test_reachable_done_step_commit_is_left_unchanged(
        self, config, git_ops, task_assignment,
    ):
        """A done step's recorded commit that IS reachable from HEAD (the
        ordinary, non-orphaned case) must not be touched."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'feature.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        await workflow._reconcile_done_step_commits()

        unchanged = artifacts.read_plan()
        assert unchanged['steps'][0]['commit'] == step_commit

    async def test_worktree_none_is_a_noop(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt_info.path)
        artifacts.update_base_commit(wt_info.base_commit)
        workflow.plan = _write_done_step_plan(artifacts, 'step-1', 'deadbeef')
        workflow.worktree = None

        await workflow._reconcile_done_step_commits()  # must not raise

        assert artifacts.read_plan()['steps'][0]['commit'] == 'deadbeef'

    async def test_git_ops_none_is_a_noop(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt_info.path)
        artifacts.update_base_commit(wt_info.base_commit)
        workflow.plan = _write_done_step_plan(artifacts, 'step-1', 'deadbeef')
        workflow.git_ops = None  # type: ignore[assignment]

        await workflow._reconcile_done_step_commits()  # must not raise

        assert artifacts.read_plan()['steps'][0]['commit'] == 'deadbeef'

    async def test_base_commit_unset_is_a_noop(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow = TaskWorkflow(
            assignment=task_assignment,
            config=config,
            git_ops=git_ops,
            scheduler=MagicMock(),  # type: ignore[arg-type]
            briefing=MagicMock(),  # type: ignore[arg-type]
            mcp=MagicMock(),  # type: ignore[arg-type]
        )
        workflow.worktree = wt
        artifacts = TaskArtifacts(wt)
        artifacts.init('42', 'X', 'desc')  # no base_commit
        workflow.artifacts = artifacts
        workflow.plan = _write_done_step_plan(artifacts, 'step-1', 'deadbeef')

        await workflow._reconcile_done_step_commits()  # must not raise

        assert artifacts.read_plan()['steps'][0]['commit'] == 'deadbeef'

    # -----------------------------------------------------------------
    # step-5 RED: flag-for-review (non-blocking info escalation) branch
    # -----------------------------------------------------------------
    #
    # None of these can be safely auto-reconciled, so the commit must be
    # left UNCHANGED and a non-blocking info Escalation filed instead,
    # mirroring how _escalate_corruption is exercised elsewhere in the
    # suite (escalation_queue as a MagicMock; assert .submit called once
    # with an Escalation, inspected via .submit.call_args.args[0]).

    async def test_content_mismatch_flags_for_review_and_leaves_commit_unchanged(
        self, config, git_ops, task_assignment,
    ):
        """CONTENT MISMATCH: the orphaned done-step commit's own file
        ('feature.py') is NOT a subset of the tip WIP run's files
        ('other.py') -> cannot be safely auto-reconciled. Flag for review
        via a non-blocking info escalation naming the step id, and leave
        the step's commit UNCHANGED (never re-point on an unverified
        content match)."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        escalation_queue = MagicMock()
        escalation_queue.make_id.return_value = 'esc-42-1'
        workflow, artifacts = _make_workflow(
            config, git_ops, task_assignment, wt, escalation_queue=escalation_queue,
        )
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'feature.py').write_text('original implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        # Orphan step_commit, then land a WIP commit touching a wholly
        # DIFFERENT file — {'feature.py'} is not a subset of {'other.py'}.
        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'other.py').write_text('unrelated\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        await workflow._reconcile_done_step_commits()

        unchanged = artifacts.read_plan()
        assert unchanged['steps'][0]['commit'] == step_commit, (
            'must not re-point the commit on an unverified content match'
        )
        escalation_queue.submit.assert_called_once()
        esc = escalation_queue.submit.call_args.args[0]
        assert esc.severity == 'info'
        assert esc.category == 'infra_issue'
        assert 'step-1' in esc.summary or 'step-1' in esc.detail

    async def test_unresolvable_original_commit_flags_for_review(
        self, config, git_ops, task_assignment,
    ):
        """UNRESOLVABLE ORIGINAL: a done step records a fabricated/GC'd sha
        (get_commit_changed_files returns []) — content cannot be verified
        even though a WIP run sits at HEAD -> flag for review, leave the
        (unresolvable) commit unchanged."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        escalation_queue = MagicMock()
        escalation_queue.make_id.return_value = 'esc-42-1'
        workflow, artifacts = _make_workflow(
            config, git_ops, task_assignment, wt, escalation_queue=escalation_queue,
        )
        artifacts.update_base_commit(wt_info.base_commit)

        fabricated_sha = 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef'
        workflow.plan = _write_done_step_plan(artifacts, 'step-1', fabricated_sha)

        (wt / 'other.py').write_text('unrelated\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        await workflow._reconcile_done_step_commits()

        unchanged = artifacts.read_plan()
        assert unchanged['steps'][0]['commit'] == fabricated_sha
        escalation_queue.submit.assert_called_once()
        esc = escalation_queue.submit.call_args.args[0]
        assert esc.severity == 'info'
        assert esc.category == 'infra_issue'
        assert 'step-1' in esc.summary or 'step-1' in esc.detail

    async def test_no_wip_run_at_head_flags_for_review(
        self, config, git_ops, task_assignment,
    ):
        """NO WIP AT HEAD: the done-step commit was orphaned, but HEAD
        carries a normal (non-WIP-safety-commit) commit rather than a WIP
        run -> nothing to reconcile against. Flag for review, commit
        unchanged."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        escalation_queue = MagicMock()
        escalation_queue.make_id.return_value = 'esc-42-1'
        workflow, artifacts = _make_workflow(
            config, git_ops, task_assignment, wt, escalation_queue=escalation_queue,
        )
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'feature.py').write_text('original implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'unrelated.py').write_text('normal, non-WIP work\n')
        normal_sha = await git_ops.commit(wt, 'feat: unrelated normal commit')
        assert normal_sha, 'Setup: expected a real (non-WIP) commit at HEAD'

        await workflow._reconcile_done_step_commits()

        unchanged = artifacts.read_plan()
        assert unchanged['steps'][0]['commit'] == step_commit
        escalation_queue.submit.assert_called_once()
        esc = escalation_queue.submit.call_args.args[0]
        assert esc.severity == 'info'
        assert esc.category == 'infra_issue'
        assert 'step-1' in esc.summary or 'step-1' in esc.detail

    async def test_escalation_queue_none_on_flagged_case_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        """When escalation_queue is None, a would-be-flagged (content
        mismatch) case must not raise — the escalation is best-effort /
        guarded, matching _escalate_corruption's own None-queue guard.
        The stale commit is simply left as today's baseline (unchanged)."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(
            config, git_ops, task_assignment, wt, escalation_queue=None,
        )
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'feature.py').write_text('original implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'other.py').write_text('unrelated\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        await workflow._reconcile_done_step_commits()  # must not raise

        assert artifacts.read_plan()['steps'][0]['commit'] == step_commit

    # -----------------------------------------------------------------
    # Amendment coverage: multiple orphaned done steps in one plan, and
    # the escalate-at-most-once-per-orphan dedup guard.
    # -----------------------------------------------------------------

    async def test_multiple_orphaned_done_steps_mixed_reconcile_and_escalate(
        self, config, git_ops, task_assignment,
    ):
        """Two done steps are orphaned in the SAME reconcile pass: step-1's
        file reappears in the tip WIP run (auto-reconciled) while step-2's
        file does not (left unchanged + flagged). Guards against the scan
        being accidentally single-item — e.g. stopping after the first
        reconcile/escalate, or a shared-state bug that lets one step's
        outcome leak into the other's."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        escalation_queue = MagicMock()
        escalation_queue.make_id.return_value = 'esc-42-1'
        workflow, artifacts = _make_workflow(
            config, git_ops, task_assignment, wt, escalation_queue=escalation_queue,
        )
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'feature.py').write_text('step-1 implementation\n')
        step1_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step1_commit, 'Setup: expected a real commit to be made'

        (wt / 'orphan.py').write_text('step-2 implementation\n')
        step2_commit = await git_ops.commit(wt, 'feat: GREEN — step-2 implementation')
        assert step2_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_steps_plan(artifacts, [
            {'id': 'step-1', 'type': 'impl', 'status': 'done', 'commit': step1_commit},
            {'id': 'step-2', 'type': 'impl', 'status': 'done', 'commit': step2_commit},
        ])

        # Orphan both, then land a WIP run that reintroduces ONLY step-1's
        # file — step-2's file is absent from the WIP run entirely.
        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'feature.py').write_text('step-1 implementation\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        await workflow._reconcile_done_step_commits()

        reconciled = artifacts.read_plan()
        step_1 = next(s for s in reconciled['steps'] if s['id'] == 'step-1')
        step_2 = next(s for s in reconciled['steps'] if s['id'] == 'step-2')
        assert step_1['commit'] == wip_sha, 'step-1 must auto-reconcile to the WIP tip'
        assert step_2['commit'] == step2_commit, (
            'step-2 has no matching WIP content and must be left unchanged'
        )
        escalation_queue.submit.assert_called_once()
        esc = escalation_queue.submit.call_args.args[0]
        assert 'step-2' in esc.summary or 'step-2' in esc.detail

    async def test_unreconcilable_orphan_escalates_only_once_across_repeated_calls(
        self, config, git_ops, task_assignment,
    ):
        """Calling _reconcile_done_step_commits twice for the SAME
        unreconcilable orphan (as happens across successive
        _execute_iterations loop iterations while the step remains
        unreconciled — the method deliberately leaves the commit unchanged
        on a mismatch) must only submit ONE info escalation, not one per
        call. Without a dedup guard, a single stuck orphan would otherwise
        flood the escalation queue with a duplicate escalation every
        iteration up to max_execute_iterations."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        escalation_queue = MagicMock()
        escalation_queue.make_id.return_value = 'esc-42-1'
        workflow, artifacts = _make_workflow(
            config, git_ops, task_assignment, wt, escalation_queue=escalation_queue,
        )
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'feature.py').write_text('original implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        # Orphan step_commit, then land a WIP commit touching a wholly
        # DIFFERENT file — never reconciles, so the condition recurs.
        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'other.py').write_text('unrelated\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        await workflow._reconcile_done_step_commits()
        workflow.plan = artifacts.read_plan()
        await workflow._reconcile_done_step_commits()  # same unresolved orphan, again

        unchanged = artifacts.read_plan()
        assert unchanged['steps'][0]['commit'] == step_commit
        escalation_queue.submit.assert_called_once()


# ---------------------------------------------------------------------------
# step-7 RED: _execute_iterations wires the reconciler in before the WIP
# detector, with a self.plan re-read in between
# ---------------------------------------------------------------------------
#
# Mirrors test_harness_wip_step_detection.py's
# TestExecuteIterationsForwardsWipNotice: real git worktree, heavy
# collaborators mocked, _invoke stubbed to mark the one pending step done so
# the loop exits after a single iteration.


@pytest.mark.asyncio
class TestExecuteIterationsReconcilesDoneSteps:
    async def test_done_step_commit_reconciled_before_prompt_is_built(
        self, config, git_ops, task_assignment,
    ):
        """A done step's orphaned commit must be reconciled to the tip WIP
        sha INSIDE _execute_iterations's loop, before build_implementer_prompt
        is called — and the reconciled value must be re-read into self.plan
        (proving _reconcile_done_step_commits actually ran and its result was
        picked up, not just left sitting unread on disk)."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        # step-1's original implementation commit, later orphaned.
        (wt / 'feature.py').write_text('original implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit, 'Setup: expected a real commit to be made'

        plan = {
            'task_id': '42',
            'title': 'X',
            'analysis': 'A',
            'prerequisites': [],
            'steps': [
                {'id': 'step-1', 'type': 'impl', 'status': 'done', 'commit': step_commit},
                {'id': 'step-2', 'type': 'impl', 'status': 'pending', 'commit': None},
            ],
        }
        artifacts.write_plan(plan)
        artifacts.stamp_plan_provenance(workflow.session_id)
        workflow.plan = artifacts.read_plan()

        # Orphan step_commit, then re-land its exact file as a WIP
        # safety-commit at HEAD (mirrors _inter_iteration_rebase's
        # squash-then-rebase sequence).
        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'feature.py').write_text('original implementation\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        def _mark_step2_done_side_effect(*args, **kwargs):
            artifacts.update_step_status('step-2', 'done', 'impl-commit-sha')
            return AgentResult(success=True, output='')

        workflow.briefing.build_implementer_prompt = AsyncMock(return_value='impl')
        workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
        workflow._get_head_commit = AsyncMock(return_value='head-sha')  # type: ignore[method-assign]
        workflow._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_mark_step2_done_side_effect,
        )

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        final_plan = artifacts.read_plan()
        step_1 = next(s for s in final_plan['steps'] if s['id'] == 'step-1')
        assert step_1['commit'] == wip_sha, (
            f'Expected step-1 commit reconciled to WIP tip {wip_sha} inside '
            f"_execute_iterations's loop, got {step_1['commit']}"
        )
        assert step_1['status'] == 'done'
