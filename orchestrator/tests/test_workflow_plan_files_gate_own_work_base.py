"""The pre-merge plan-files gate measures the branch from its own-work base.

The worktree is a real git checkout in the task 5695 shape: the branch edits
``a.py``, main edits ``m1.py``, then the branch merges main. From the stored
creation-time base, main's ``m1.py`` commit reads as branch work; from the
own-work base it does not (task 7668).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merged_in_branch import branch_with_main_merged_in, git
from _workflow_helpers import _make, _make_git_ops

from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

pytestmark = pytest.mark.asyncio

_M1_ONLY_MISS = 'no commit on the branch touched them: m1.py.'


async def _submit_through_the_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_files: list[str],
    narrow: Callable[[TaskWorkflow], Callable[[list[str]], Awaitable[bool]]],
) -> AsyncMock:
    merged = branch_with_main_merged_in(
        tmp_path, base_files={'.gitignore': '.task/\n', 'x.py': 'x = 1\n'},
    )
    f = _make(worktree=merged.worktree, project_root=merged.repo)
    wf = f.wf
    wf.git_ops = _make_git_ops(merged.repo)
    f.artifacts.update_base_commit(merged.creation_sha)
    wf._base_commit = merged.creation_sha
    wf.plan = {'files': plan_files}
    wf.merge_queue = MagicMock()
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    monkeypatch.setattr(wf, '_mark_blocked', mark_blocked)
    monkeypatch.setattr(wf, '_try_narrow_plan', narrow(wf))

    await wf._submit_to_merge_queue('50')

    return mark_blocked


def _reason(mark_blocked: AsyncMock) -> str:
    mark_blocked.assert_awaited_once()
    args, kwargs = mark_blocked.call_args
    return args[0] if args else kwargs['reason']


def _declines(_wf: TaskWorkflow) -> AsyncMock:
    return AsyncMock(return_value=False)


def _drops_x_py(wf: TaskWorkflow) -> Callable[[list[str]], Awaitable[bool]]:
    async def narrow(_not_touched: list[str]) -> bool:
        wf.plan = {'files': ['a.py', 'm1.py']}
        return True
    return narrow


def _merges_main_again(wf: TaskWorkflow) -> Callable[[list[str]], Awaitable[bool]]:
    async def narrow(_not_touched: list[str]) -> bool:
        assert wf.worktree is not None
        repo = Path(wf.config.project_root)
        (repo / 'm2.py').write_text('m2 = 1\n')
        git(repo, 'add', 'm2.py')
        git(repo, 'commit', '-q', '-m', 'main advances again')
        git(wf.worktree, 'merge', '-q', '--no-edit', 'main')
        return True
    return narrow


async def test_a_file_only_main_touched_is_reported_not_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mark_blocked = await _submit_through_the_gate(
        tmp_path, monkeypatch, ['a.py', 'm1.py'], _declines,
    )

    assert _M1_ONLY_MISS in _reason(mark_blocked)


async def test_the_re_check_after_narrowing_still_measures_the_own_work_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mark_blocked = await _submit_through_the_gate(
        tmp_path, monkeypatch, ['a.py', 'm1.py', 'x.py'], _drops_x_py,
    )

    assert _M1_ONLY_MISS in _reason(mark_blocked)


async def test_the_re_check_re_derives_the_base_after_a_narrowing_pass_merges_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mark_blocked = await _submit_through_the_gate(
        tmp_path, monkeypatch, ['a.py', 'm2.py'], _merges_main_again,
    )

    assert 'no commit on the branch touched them: m2.py.' in _reason(mark_blocked)
