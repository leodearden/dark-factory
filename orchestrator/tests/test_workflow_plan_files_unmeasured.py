"""The pre-merge plan-files gate never reports non-delivery for a range that measured nothing.

``TaskWorkflow._submit_to_merge_queue`` drives the real gate over a real
worktree. A stopped rebase (task 5030), uncommitted declared edits (task 5027),
an unreadable branch history and work left on a sibling ref by a branch with
no commit of its own (task 3560) each block with their remedy on the normal
ladder and skip the narrowing pass; a narrowed plan whose re-check is still
unmeasured blocks the same way. A stale stored base from a quarantined lane
is still measured from the own-work base (task 7668), and a branch carrying
its own work is judged by it even when a sibling ref touched the rest.

Precedence: a sibling ref that an executed re-cut ruling parked is never
offered, even when the ruling's record lives only in the task store, and
work that already landed on main reaches the already-landed outcome before
any stale pre-rebase sibling is considered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merged_in_branch import git
from _unmeasured_delivery_scenes import (
    add_worktree,
    commit,
    landed_with_pre_rebase_sibling,
    seeded_repo,
    stopped_rebase,
    work_on_sibling_ref,
)
from _workflow_helpers import _make, _make_git_ops

from orchestrator.merge_lane.gates import (
    ALREADY_LANDED_REASON_PREFIX,
    PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
)
from orchestrator.merge_lane.unmeasured_delivery import PLAN_DELIVERY_UNMEASURED_REASON_PREFIX
from orchestrator.recut_ruling import ParkedIncarnation, RecutRuling, execution_record
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class Submitted:
    mark_blocked: AsyncMock
    narrow: AsyncMock

    def reason(self) -> str:
        self.mark_blocked.assert_awaited_once()
        args, kwargs = self.mark_blocked.call_args
        return args[0] if args else kwargs['reason']

    def assert_blocked_unmeasured(
        self, suggested_action: str, *, after_narrowing: bool = False,
    ) -> str:
        reason = self.reason()
        kwargs = self.mark_blocked.call_args.kwargs
        assert reason.startswith(PLAN_DELIVERY_UNMEASURED_REASON_PREFIX), reason
        assert kwargs['category'] == 'plan_delivery_unmeasured'
        assert kwargs['suggested_action'] == suggested_action
        assert not kwargs.get('escalate_to_human')
        assert self.narrow.await_count == (1 if after_narrowing else 0)
        return reason


def _declines_to_narrow(wf: TaskWorkflow) -> bool:
    return False


async def _submit(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path, worktree: Path, creation_sha: str, plan_files: list[str],
    *, backend_metadata: dict | None = None,
    narrow: Callable[[TaskWorkflow], bool] = _declines_to_narrow,
) -> Submitted:
    f = _make(worktree=worktree, project_root=repo)
    wf = f.wf
    wf.git_ops = _make_git_ops(repo)
    if backend_metadata is not None:
        wf.scheduler.get_task = AsyncMock(
            return_value={'id': '50', 'metadata': backend_metadata},
        )
    f.artifacts.update_base_commit(creation_sha)
    wf._base_commit = creation_sha
    wf.plan = {'files': plan_files}
    wf.merge_queue = MagicMock()
    submitted = Submitted(
        mark_blocked=AsyncMock(return_value=WorkflowOutcome.BLOCKED),
        narrow=AsyncMock(side_effect=lambda not_touched: narrow(wf)),
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

    submitted.assert_blocked_unmeasured('finish_or_abort_operation')


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


async def test_an_unreadable_history_blocks_with_retry_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n'})
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    vanished_base = 'deadbeef' * 5

    submitted = await _submit(monkeypatch, repo, worktree, vanished_base, ['a.py'])

    reason = submitted.assert_blocked_unmeasured('retry_merge')
    assert f'fatal: Invalid revision range {vanished_base}..' in reason


async def test_a_narrowed_plan_still_unmeasured_blocks_with_its_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = seeded_repo(tmp_path, {'a.py': 'a = 1\n', 'b.py': 'b = 1\n', 'c.py': 'c = 1\n'})
    creation_sha = git(repo, 'rev-parse', 'HEAD')
    worktree = add_worktree(repo, tmp_path / 'wt', 'task/50')
    commit(worktree, {'a.py': 'a = 2\n'}, 'task work')

    def drop_c_and_leave_b_uncommitted(wf: TaskWorkflow) -> bool:
        wf.plan = {'files': ['a.py', 'b.py']}
        (worktree / 'b.py').write_text('b = 2\n')
        return True

    submitted = await _submit(
        monkeypatch, repo, worktree, creation_sha, ['a.py', 'b.py', 'c.py'],
        narrow=drop_c_and_leave_b_uncommitted,
    )

    reason = submitted.assert_blocked_unmeasured('commit_worktree_edits', after_narrowing=True)
    assert 'b.py' in reason
    assert 'c.py' not in reason


async def test_a_sibling_ref_does_not_excuse_a_branch_with_its_own_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = work_on_sibling_ref(tmp_path)
    commit(scene.worktree, {'b.py': 'b = 2\n'}, 'own work')

    submitted = await _submit(
        monkeypatch, scene.repo, scene.worktree, scene.creation_sha, ['a.py', 'b.py'],
    )

    submitted.narrow.assert_awaited_once()
    reason = submitted.reason()
    assert reason.startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX), reason
    assert 'no commit on the branch touched them: a.py.' in reason
    assert scene.sibling_ref not in reason


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


async def test_a_sibling_parked_by_a_recut_is_never_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = work_on_sibling_ref(tmp_path)
    parked = ParkedIncarnation(
        parked_branch=scene.sibling_ref, parked_tip=scene.sibling_tip,
        quarantined_worktree=None, relocated_meta_roots=(),
    )
    record = execution_record(RecutRuling('r-1', datetime.now(UTC)), parked, datetime.now(UTC))

    submitted = await _submit(
        monkeypatch, scene.repo, scene.worktree, scene.creation_sha, ['a.py'],
        backend_metadata={'recut_executed': record},
    )

    submitted.narrow.assert_awaited()
    reason = submitted.reason()
    assert reason.startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX), reason
    assert scene.sibling_ref not in reason
    assert all(
        call.kwargs.get('category') != 'plan_delivery_unmeasured'
        for call in submitted.mark_blocked.call_args_list
    )


async def test_landed_work_reaches_already_landed_before_a_stale_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = landed_with_pre_rebase_sibling(tmp_path)

    submitted = await _submit(
        monkeypatch, scene.repo, scene.worktree, scene.creation_sha, ['a.py'],
    )

    reason = submitted.reason()
    kwargs = submitted.mark_blocked.call_args.kwargs
    assert kwargs['category'] == 'already_landed'
    assert kwargs['suggested_action'] == 'verify_landing_and_close'
    assert reason.startswith(ALREADY_LANDED_REASON_PREFIX), reason
    submitted.narrow.assert_not_awaited()
