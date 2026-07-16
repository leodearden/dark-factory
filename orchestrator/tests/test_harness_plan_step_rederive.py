"""Tests for re-deriving plan.json step-completion status from branch state
after an inter-iteration rebase (task 2387).

Recurring failure mode: after a rebase, a plan step's implementation is
already committed on the branch (often folded into a WIP safety-commit) but
plan.json's status is stale "pending" — the EXECUTE loop then treats it as a
genuine remaining blocker and re-attempts already-done work. This suite
covers the new reconciliation machinery that fixes that:

  - TaskWorkflow._rederive_step_status_from_branch_state (workflow.py)
  - _inter_iteration_rebase wiring the re-derivation in right after
    update_base_commit

Complementary to (and non-overlapping with) task 2386's
``_reconcile_done_step_commits``, covered by test_reconcile_done_step_commits.py:
that method reconciles a *done* step's stale ``commit``; this suite covers
re-deriving a *pending* step's status back to ``done`` in the first place.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# ---------------------------------------------------------------------------
# Fixtures — mirrors test_harness_wip_step_detection.py's git_repo/config/
# git_ops/task_assignment/_make_workflow pattern (real temp git repo + real
# worktree via git_ops.create_worktree; heavy collaborators mocked).
# ---------------------------------------------------------------------------


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
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha-old')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': [], 'prerequisites': []}
    return workflow, artifacts


def _write_plan(artifacts: TaskArtifacts, workflow: TaskWorkflow, steps: list[dict]) -> dict:
    """Persist a plan with the given step dicts and stamp provenance,
    returning the re-read plan so ``workflow.plan`` mirrors what's on disk —
    required because ``update_step_status`` reads/writes plan.json directly,
    independent of any in-memory ``workflow.plan`` the caller also sets."""
    plan = {
        'task_id': '42',
        'title': 'X',
        'analysis': 'A',
        'prerequisites': [],
        'steps': steps,
    }
    artifacts.write_plan(plan)
    artifacts.stamp_plan_provenance(workflow.session_id)
    return artifacts.read_plan()


# ---------------------------------------------------------------------------
# step-1 RED: happy-path re-derivation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusHappyPath:
    async def test_pending_step_completed_in_log_is_rederived_to_done(
        self, config, git_ops, task_assignment,
    ):
        """A step recorded 'pending' in plan.json but marked complete in the
        durable iteration log (with a real branch commit beyond base) is
        re-derived to 'done', while an unrelated pending step is untouched."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
            {'id': 'step-2', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['step-1'],
            'commit': step_commit,
        })

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == ['step-1']
        plan = artifacts.read_plan()
        step_1 = next(s for s in plan['steps'] if s['id'] == 'step-1')
        step_2 = next(s for s in plan['steps'] if s['id'] == 'step-2')
        assert step_1['status'] == 'done'
        assert step_1['commit'], 'Expected a non-null commit recorded on the re-derived step'
        assert step_2['status'] == 'pending', (
            'step-2 was never in steps_completed and must stay pending'
        )


# ---------------------------------------------------------------------------
# step-3 RED: contamination / no-work guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusNoWorkGuard:
    async def test_orphan_log_on_unadvanced_branch_is_not_rederived(
        self, config, git_ops, task_assignment,
    ):
        """An inherited/contaminated iterations.jsonl entry claiming step-1
        completed must NOT be trusted when the branch HEAD has not actually
        diverged from base (no real commit was made) — the SHA-primary
        `_has_prior_implementation` guard must reject it."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['step-1'],
            'commit': 'orphan-sha',
        })

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == []
        plan = artifacts.read_plan()
        assert plan['steps'][0]['status'] == 'pending'


# ---------------------------------------------------------------------------
# step-5 RED: crash-safety / best-effort contract (mirrors
# TestDetectTipWipCommits/TestReconcileDoneStepCommits' defensive tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusDefensive:
    async def test_worktree_none_returns_empty_and_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })
        workflow.worktree = None

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == []
        assert artifacts.read_plan()['steps'][0]['status'] == 'pending'

    async def test_git_ops_none_returns_empty_and_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })
        workflow.git_ops = None  # type: ignore[assignment]

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == []
        assert artifacts.read_plan()['steps'][0]['status'] == 'pending'

    async def test_internal_failure_returns_empty_and_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })
        workflow._get_head_commit = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('boom'),
        )

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == []
        assert artifacts.read_plan()['steps'][0]['status'] == 'pending'


# ---------------------------------------------------------------------------
# step-7 RED: observability log entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusLogEntry:
    async def test_rederive_emits_plan_step_rederive_log_entry(
        self, config, git_ops, task_assignment,
    ):
        """Positive: a genuine re-derivation writes exactly one
        plan_step_rederive iteration-log entry naming the re-derived id(s)."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })

        result = await workflow._rederive_step_status_from_branch_state()
        assert result == ['step-1']

        entries, _ = artifacts.read_iteration_log()
        rederive_entries = [e for e in entries if e.get('event') == 'plan_step_rederive']
        assert len(rederive_entries) == 1, (
            f'Expected exactly one plan_step_rederive entry, got {rederive_entries}'
        )
        assert rederive_entries[0]['rederived_steps'] == ['step-1']

    async def test_no_rederive_emits_no_log_entry(
        self, config, git_ops, task_assignment,
    ):
        """Negative: reuses the no-work-guard setup (nothing re-derived) —
        no plan_step_rederive entry must be written."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['step-1'],
            'commit': 'orphan-sha',
        })

        result = await workflow._rederive_step_status_from_branch_state()
        assert result == []

        entries, _ = artifacts.read_iteration_log()
        assert not any(e.get('event') == 'plan_step_rederive' for e in entries), (
            f'No plan_step_rederive entry should be written when nothing was '
            f're-derived; entries: {entries}'
        )
