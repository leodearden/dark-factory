"""The architect's already-committed list is the branch's own commits.

The worktree is a real git checkout in the task 5695 shape: the branch edits
``a.py``, main edits ``m1.py``, then the branch merges main. Listed from the
stored creation-time base, main's 'main advances' commit would read as branch
work; listed from the own-work base it does not (task 7668).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _merged_in_branch import branch_with_main_merged_in, git
from _workflow_helpers import _make, _make_git_ops

pytestmark = pytest.mark.asyncio


async def test_main_merged_in_lists_only_the_branch_s_own_commits(tmp_path: Path) -> None:
    merged = branch_with_main_merged_in(tmp_path, base_files={'.gitignore': '.task/\n'})
    f = _make(worktree=merged.worktree, project_root=merged.repo)
    f.wf.git_ops = _make_git_ops(merged.repo)
    f.artifacts.update_base_commit(merged.creation_sha)

    result = await f.wf._detect_committed_branch_work()

    merge_sha = git(merged.worktree, 'rev-parse', 'HEAD')
    task_sha = git(merged.worktree, 'rev-parse', 'HEAD^1')
    shas = [c['sha'] for c in result]
    assert shas == [merge_sha, task_sha]
    assert merged.main_tip not in shas
    assert 'main advances' not in [c['subject'] for c in result]
