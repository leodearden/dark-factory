"""The merge phase refuses a branch with nothing to land (task 7516).

In dark_factory 3493 / 3580 and reify 8363, a steward resolution left the task
branch with nothing honest to land, and the post-steward check read the
worktree HEAD, found it on main and reported a sha-less merge success. That
success then surfaced as the misleading "no merge_sha" L1 from
``workflow.py::TaskWorkflow._finalise_without_merge_sha``. These cases drive
``_merge_and_finalise`` over a real repository and worktree, with only the
merge-queue submission stubbed (its side effect plays the steward), and check
that each shape is refused by name and that a genuine landing still closes.

This file imports no merge-lane module, so ``scripts/merge_lane_metrics.py``
does not measure its private reads (see
``test_workflow_finalise_without_merge_sha.py``, which stays outside it the
same way).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling
from _workflow_helpers import _Fixture, _make, _make_git_ops
from escalation.queue import EscalationQueue

from orchestrator.workflow import WorkflowOutcome

TASK_ID = '50'
BRANCH = f'task/{TASK_ID}'
SEED = RepoSeed(
    files=(('README.md', '# Test\n'), ('.gitignore', '.task/\n')), message='init',
)


def _git(cwd: Path, *args: str) -> str:
    assert_isolated_git_repo(cwd)
    proc = subprocess.run(
        ['git', *args], cwd=str(cwd), capture_output=True,
        env=git_env_with_ceiling(cwd), text=True, check=False,
    )
    assert proc.returncode == 0, (
        f'git {" ".join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}'
    )
    return proc.stdout.strip()


class _Scene:
    """A real repo plus a ``task/50`` worktree one commit ahead, driven by a workflow."""

    def __init__(self, tmp_path: Path) -> None:
        self.repo = seed_repo(tmp_path / 'repo', SEED)
        self.wt = tmp_path / 'wt'
        _git(self.repo, 'worktree', 'add', str(self.wt), '-b', BRANCH, 'main')
        (self.wt / 'feature.py').write_text('x = 1\n')
        _git(self.wt, 'add', 'feature.py')
        _git(self.wt, 'commit', '-m', 'add feature')
        self.task_commit = _git(self.wt, 'rev-parse', 'HEAD')

        self.fixture: _Fixture = _make(
            worktree=self.wt, project_root=self.repo, task_id=TASK_ID,
        )
        wf = self.wf = self.fixture.wf
        git_ops = _make_git_ops(self.repo)
        git_ops.rebase_onto_main = AsyncMock(return_value=False)
        wf.git_ops = git_ops
        wf._merge_sha = None
        wf.escalation_queue = self.queue = EscalationQueue(tmp_path / 'escalations')
        wf.config.unblock_auto.enabled = False
        wf.config.max_merge_retries = 2
        wf.config.max_pre_merge_retries = 1
        wf.config.merge_train_former_enabled = False
        wf.scheduler.get_task = AsyncMock(return_value={'id': TASK_ID, 'metadata': {}})
        wf._ensure_steward_started = AsyncMock()
        wf._await_steward_completion = AsyncMock()
        self.fixture.artifacts.append_iteration_log({
            'agent': 'implementer', 'source': 'orchestrator',
            'steps_attempted': ['s1'], 'steps_completed': ['s1'],
            'commit': self.task_commit,
        })
        self.submit = AsyncMock(return_value=WorkflowOutcome.REQUEUED)
        wf._submit_to_merge_queue = self.submit

    def steward_does(self, effect: Callable[[], None]) -> None:
        """Make the queue submission run *effect*, then answer as a requeued resolution."""
        def _submit(*_args: object, **_kwargs: object) -> WorkflowOutcome:
            effect()
            return WorkflowOutcome.REQUEUED
        self.submit.side_effect = _submit

    def reset_branch_to_main_keeping_work_staged(self) -> None:
        _git(self.wt, 'reset', '--soft', 'main')

    def move_worktree_onto_a_squash_branch_at_main(self) -> None:
        _git(self.wt, 'checkout', '-b', f'{BRANCH}-squash', 'main')
        (self.wt / 'feature.py').write_text('x = 1\n')
        _git(self.wt, 'add', 'feature.py')

    def land_branch_on_main(self) -> None:
        _git(self.repo, 'merge', '--no-ff', '-m', f'Merge {BRANCH} into main', BRANCH)

    async def merge(self) -> WorkflowOutcome:
        return await self.wf._merge_and_finalise(BRANCH)

    def pending_root_causes(self) -> list[str]:
        return [
            esc.root_cause
            for esc in self.queue.get_by_task(TASK_ID, status='pending')
            if esc.level == 1
        ]


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    return _Scene(tmp_path)


@pytest.mark.asyncio
async def test_a_steward_reset_of_the_branch_to_main_is_refused(scene: _Scene) -> None:
    scene.steward_does(scene.reset_branch_to_main_keeping_work_staged)

    outcome = await scene.merge()

    assert outcome == WorkflowOutcome.BLOCKED
    assert scene.pending_root_causes() == ['merge-refused:branch_empty']
    scene.fixture.mark_done.assert_not_awaited()
    assert scene.submit.await_count == 1


@pytest.mark.asyncio
async def test_a_steward_move_of_the_worktree_onto_main_is_refused(scene: _Scene) -> None:
    scene.steward_does(scene.move_worktree_onto_a_squash_branch_at_main)

    outcome = await scene.merge()

    assert outcome == WorkflowOutcome.BLOCKED
    assert scene.pending_root_causes() == ['merge-refused:worktree_off_branch']
    assert _git(scene.repo, 'rev-parse', f'refs/heads/{BRANCH}') == scene.task_commit
    scene.fixture.mark_done.assert_not_awaited()
    assert scene.submit.await_count == 1


@pytest.mark.asyncio
async def test_an_empty_branch_at_merge_entry_is_refused_before_enqueue(
    scene: _Scene,
) -> None:
    scene.reset_branch_to_main_keeping_work_staged()

    outcome = await scene.merge()

    assert outcome == WorkflowOutcome.BLOCKED
    assert scene.pending_root_causes() == ['merge-refused:branch_empty']
    scene.submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_genuine_landing_after_a_steward_resolution_still_closes(
    scene: _Scene,
) -> None:
    scene.steward_does(scene.land_branch_on_main)

    outcome = await scene.merge()

    assert outcome == WorkflowOutcome.DONE
    scene.fixture.mark_done.assert_awaited_once()
    assert scene.fixture.mark_done.await_args.kwargs['kind'] == 'found_on_main'
    assert scene.pending_root_causes() == []
