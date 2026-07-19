"""Truthful iteration-ledger commit stamping (task 2759).

All three orchestrator iteration-ledger writers (implementer in
``_execute_iterations``, debugger in ``_verify_debugfix_loop``, amender in
``_amend``) historically stamped ``commit: await self._get_head_commit()``
unconditionally — with no pre/post HEAD comparison. When an agent session
ended before committing (implementer died mid-background-wait, amendment left
uncommitted), HEAD was unchanged and the ledger recorded the round "at" an
unrelated pre-existing commit. Recovery guards then consumed that false
provenance (reify 5164 RCA).

The fix factors a single shared helper
``TaskWorkflow._iteration_commit_provenance(pre_head)`` that every writer
merges into its entry dict: HEAD advanced ⇒ ``{commit: <new sha>,
committed: True}``; HEAD unchanged ⇒ ``{commit: None, committed: False,
dirty: <porcelain-nonempty>}``. Two recovery guards
(``_iteration_entry_is_work`` and the ``_rederive_step_status_from_branch_state``
union) then treat ``committed is False`` as "no durable work this entry",
discriminating on the explicit ``False`` flag only so pre-2759 legacy entries
(no ``committed`` key) keep today's classification.

This file holds everything — the small real-git harness is replicated here
(mirroring test_harness_plan_step_rederive.py / test_workflow_verify_retry.py)
rather than editing any shared suite, keeping concurrency locks to workflow.py
plus this one new file.
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
from orchestrator.workflow import TaskWorkflow

# ---------------------------------------------------------------------------
# Real-git harness — mirrors test_harness_plan_step_rederive.py /
# test_workflow_verify_retry.py (real temp git repo + real worktree via
# git_ops.create_worktree; heavy collaborators mocked).
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
    # Mirror the real repo: .task/ execution metadata (written by
    # TaskArtifacts.init here, since these harnesses construct in-worktree
    # artifacts) is gitignored, so it never surfaces in `git status
    # --porcelain` — exactly the invariant GitOps.has_uncommitted_work
    # (and thus _iteration_commit_provenance's dirty read) relies on.
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

    Mirrors test_harness_plan_step_rederive.py._make_workflow.
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


def _write_plan(
    artifacts: TaskArtifacts,
    workflow: TaskWorkflow,
    steps: list[dict],
    prerequisites: list[dict] | None = None,
) -> dict:
    """Persist a plan with the given step dicts and stamp provenance so
    ``validate_plan_owner(workflow.session_id)`` passes, returning the re-read
    plan. Mirrors test_harness_plan_step_rederive.py._write_plan.
    """
    plan = {
        'task_id': '42',
        'title': 'X',
        'analysis': 'A',
        'prerequisites': prerequisites or [],
        'steps': steps,
    }
    artifacts.write_plan(plan)
    artifacts.stamp_plan_provenance(workflow.session_id)
    return artifacts.read_plan()


# ---------------------------------------------------------------------------
# step-1 RED: the _iteration_commit_provenance helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIterationCommitProvenanceHelper:
    """The shared helper compares pre/post HEAD and reports truthful
    provenance: advanced ⇒ {commit, committed:True} (no dirty read);
    unchanged ⇒ {commit:None, committed:False, dirty:<porcelain-nonempty>}.
    """

    async def test_head_advanced_records_new_commit_and_no_dirty(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        pre_head = await workflow._get_head_commit()
        (wt / 'impl.py').write_text('implementation\n')
        new_commit = await git_ops.commit(wt, 'feat: real commit')
        assert new_commit, 'Setup: expected a real commit to be made'

        result = await workflow._iteration_commit_provenance(pre_head)

        post_head = await workflow._get_head_commit()
        assert result['commit'] == post_head
        assert result['commit'] != pre_head
        assert result['committed'] is True
        assert 'dirty' not in result, (
            'the committed (happy) path must not perform the extra dirty read'
        )

    async def test_head_unchanged_clean_tree_records_null_commit(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        pre_head = await workflow._get_head_commit()  # == current HEAD, clean tree

        result = await workflow._iteration_commit_provenance(pre_head)

        assert result['commit'] is None
        assert result['committed'] is False
        assert result['dirty'] is False

    async def test_head_unchanged_dirty_tree_records_null_commit_dirty_true(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        pre_head = await workflow._get_head_commit()
        # Uncommitted change: HEAD stays put but the tree is dirty.
        (wt / 'uncommitted.py').write_text('scratch\n')

        result = await workflow._iteration_commit_provenance(pre_head)

        assert result['commit'] is None
        assert result['committed'] is False
        assert result['dirty'] is True


# ---------------------------------------------------------------------------
# step-3 RED: the amend writer (_amend) — the RCA origin
# ---------------------------------------------------------------------------


def _amend_workflow(config, git_ops, task_assignment, wt):
    """A real-git workflow wired to drive _amend once: a plan on disk with
    stamped provenance (so validate_plan_owner passes) and a mocked amender
    briefing. The caller configures _invoke."""
    workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
    _write_plan(artifacts, workflow, [
        {'id': 'step-1', 'type': 'impl', 'status': 'done', 'commit': 'x'},
    ])
    workflow.plan = artifacts.read_plan()
    workflow.briefing.build_amender_prompt = AsyncMock(  # type: ignore[attr-defined]
        return_value='amend-prompt',
    )
    return workflow, artifacts


@pytest.mark.asyncio
class TestAmendWriterTruthfulCommit:
    """_amend must stamp truthful provenance: commit:null / committed:false
    when the amender left HEAD unchanged; the real new sha / committed:true
    when it committed."""

    async def test_amend_no_commit_records_null_and_uncommitted(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _amend_workflow(config, git_ops, task_assignment, wt)
        # No-op amender: returns success but makes no commit ⇒ HEAD unchanged.
        workflow._invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=AgentResult(success=True, output=''),
        )

        in_scope = [{'file': 'lib.py', 'suggestion': 'tidy'}]
        ok = await workflow._amend(in_scope, amendment_round=1)
        assert ok is True

        entries, _ = artifacts.read_iteration_log()
        amendments = [e for e in entries if e.get('source') == 'amendment']
        assert len(amendments) == 1, f'expected one amendment entry, got {entries}'
        entry = amendments[0]
        assert entry['commit'] is None, (
            f'amender made no commit ⇒ commit must be null, got {entry.get("commit")!r}'
        )
        assert entry['committed'] is False

    async def test_amend_with_commit_records_new_head_and_committed(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _amend_workflow(config, git_ops, task_assignment, wt)

        async def _commit_effect(*_args, **_kwargs):
            (wt / 'amend.py').write_text('amended\n')
            await git_ops.commit(wt, 'feat: amendment commit')
            return AgentResult(success=True, output='')

        workflow._invoke = AsyncMock(side_effect=_commit_effect)  # type: ignore[method-assign]

        in_scope = [{'file': 'lib.py', 'suggestion': 'tidy'}]
        ok = await workflow._amend(in_scope, amendment_round=1)
        assert ok is True

        head = await workflow._get_head_commit()
        entries, _ = artifacts.read_iteration_log()
        amendments = [e for e in entries if e.get('source') == 'amendment']
        assert len(amendments) == 1, f'expected one amendment entry, got {entries}'
        entry = amendments[0]
        assert entry['commit'] == head, (
            f'amender committed ⇒ commit must be the new HEAD {head!r}, '
            f'got {entry.get("commit")!r}'
        )
        assert entry['committed'] is True
