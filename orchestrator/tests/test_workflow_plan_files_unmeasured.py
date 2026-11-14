"""The pre-merge plan-files gate never reports non-delivery for a range that measured nothing.

``TaskWorkflow._submit_to_merge_queue`` drives the real gate over a real
worktree. A stopped rebase (task 5030), uncommitted declared edits (task 5027)
and work left on a sibling ref (task 3560) each block with their remedy on
the normal ladder and skip the narrowing pass. A stale stored base from a
quarantined lane is still measured from the own-work base (task 7668).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merged_in_branch import git
from _unmeasured_delivery_scenes import (
    add_worktree,
    commit,
    seeded_repo,
    stopped_rebase,
    work_on_sibling_ref,
)
from _workflow_helpers import _make, _make_git_ops

from orchestrator.merge_lane.gates import PLAN_FILES_NOT_TOUCHED_REASON_PREFIX
from orchestrator.merge_lane.unmeasured_delivery import PLAN_DELIVERY_UNMEASURED_REASON_PREFIX
from orchestrator.workflow import WorkflowOutcome

pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class Submitted:
    mark_blocked: AsyncMock
    narrow: AsyncMock

    def reason(self) -> str:
        self.mark_blocked.assert_awaited_once()
        args, kwargs = self.mark_blocked.call_args
        return args[0] if args else kwargs['reason']

    def assert_blocked_unmeasured(self, suggested_action: str) -> str:
        reason = self.reason()
        kwargs = self.mark_blocked.call_args.kwargs
        assert reason.startswith(PLAN_DELIVERY_UNMEASURED_REASON_PREFIX), reason
        assert kwargs['category'] == 'plan_delivery_unmeasured'
        assert kwargs['suggested_action'] == suggested_action
        assert not kwargs.get('escalate_to_human')
        self.narrow.assert_not_awaited()
        return reason


async def _submit(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path, worktree: Path, creation_sha: str, plan_files: list[str],
) -> Submitted:
    f = _make(worktree=worktree, project_root=repo)
    wf = f.wf
    wf.git_ops = _make_git_ops(repo)
    f.artifacts.update_base_commit(creation_sha)
    wf._base_commit = creation_sha
    wf.plan = {'files': plan_files}
    wf.merge_queue = MagicMock()
    submitted = Submitted(
        mark_blocked=AsyncMock(return_value=WorkflowOutcome.BLOCKED),
        narrow=AsyncMock(return_value=False),
    )
    monkeypatch.setattr(wf, '_mark_blocked', submitted.mark_blocked)
    monkeypatch.setattr(wf, '_try_narrow_plan', submitted.narrow)

    await wf._submit_to_merge_queue('50')

    return submitted


async def test_a_stopped_rebase_blocks_with_finish_or_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = stopped_rebase(tmp_path, 'task/50')

    submitted = await _submit(
        monkeypatch, scene.repo, scene.worktree, scene.creation_sha, ['x.py', 'y.py'],
    )

    submitted.assert_blocked_unmeasured('finish_or_abort_rebase')


async def test_an_uncommitted_declared_file_blocks_with_commit_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n', 'b.py': 'b = 1\n'})
    creation_sha = git(repo, 'rev-parse', 'HEAD')
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    commit(worktree, {'a.py': 'a = 2\n'}, 'task work')
    (worktree / 'b.py').write_text('b = 2\n')

    submitted = await _submit(monkeypatch, repo, worktree, creation_sha, ['a.py', 'b.py'])

    assert 'b.py' in submitted.assert_blocked_unmeasured('commit_worktree_edits')


async def test_work_on_a_sibling_ref_blocks_with_recover_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = work_on_sibling_ref(tmp_path)

    submitted = await _submit(
        monkeypatch, scene.repo, scene.worktree, scene.creation_sha, ['a.py'],
    )

    reason = submitted.assert_blocked_unmeasured('recover_work_from_sibling_ref')
    assert scene.sibling_ref in reason


async def test_a_quarantined_lanes_stale_base_still_measures_the_own_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n'})
    stale_base = git(repo, 'rev-parse', 'HEAD')
    commit(repo, {'f.py': 'f = 1\n'}, 'main adds f')
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    commit(worktree, {'a.py': 'a = 2\n'}, 'task work')

    submitted = await _submit(monkeypatch, repo, worktree, stale_base, ['a.py', 'f.py'])

    reason = submitted.reason()
    assert reason.startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX), reason
    assert 'no commit on the branch touched them: f.py.' in reason
    assert 'a.py' not in reason
