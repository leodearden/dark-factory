"""orchestrator/src/orchestrator/dispatch_base.py against a real GitOps over a real repo (task 7736)."""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio
from _workflow_helpers import _init_repo

from orchestrator.config import GitConfig
from orchestrator.dispatch_base import (
    BaseConflict,
    IntegrationRefusal,
    MergerDisposition,
    Settlement,
    find_base_conflict,
    settle_base_conflict,
)
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.outcome_facts import ConflictEvidence

GREET_FILE = 'lib.py'


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _commit_file(cwd: Path, name: str, content: str, message: str) -> str:
    (cwd / name).write_text(content)
    _git(cwd, 'add', name)
    _git(cwd, 'commit', '-m', message)
    return _git(cwd, 'rev-parse', 'HEAD')


def _greeting(word: str) -> str:
    return f'def greet(name: str) -> str:\n    return f"{word}, {{name}}"\n'


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / 'repo'
    path.mkdir()
    _init_repo(path)
    return path


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    return GitOps(
        GitConfig(main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees'),
        repo,
    )


@pytest_asyncio.fixture
async def worktree(git_ops: GitOps) -> Path:
    return (await git_ops.create_worktree('42')).path


async def _conflicting_scene(repo: Path, worktree: Path) -> None:
    _commit_file(worktree, GREET_FILE, _greeting('Hi'), 'task: greet with Hi')
    _commit_file(repo, GREET_FILE, _greeting('Hey'), 'main: greet with Hey')


@pytest.mark.asyncio
class TestFindBaseConflict:
    async def test_same_line_edited_on_both_sides_is_a_conflict(self, repo, git_ops, worktree):
        await _conflicting_scene(repo, worktree)

        conflict = await find_base_conflict(git_ops, worktree)

        assert conflict is not None
        assert conflict.main_sha == _git(repo, 'rev-parse', 'main')
        assert conflict.head_sha == _git(worktree, 'rev-parse', 'HEAD')
        assert conflict.conflicted_paths == (GREET_FILE,)

    async def test_branch_behind_main_on_another_file_is_clean(self, repo, git_ops, worktree):
        _commit_file(worktree, 'notes.txt', 'task notes\n', 'task: notes')
        _commit_file(repo, GREET_FILE, _greeting('Hey'), 'main: greet with Hey')

        assert await find_base_conflict(git_ops, worktree) is None

    async def test_branch_without_own_commits_is_clean(self, repo, git_ops, worktree):
        _commit_file(repo, GREET_FILE, _greeting('Hey'), 'main: greet with Hey')

        assert await find_base_conflict(git_ops, worktree) is None

    async def test_event_payload_spells_evidence_as_merge_events_do(self, repo, git_ops, worktree):
        await _conflicting_scene(repo, worktree)

        conflict = await find_base_conflict(git_ops, worktree)

        assert conflict is not None
        assert conflict.to_event_payload() == {
            'main_sha': conflict.main_sha,
            'head_sha': conflict.head_sha,
            **ConflictEvidence(conflicted_paths=(GREET_FILE,), counterpart_task_ids=()).to_event_payload(),
        }
        assert conflict.to_event_payload()['counterpart_task_ids'] == []


@pytest.mark.parametrize('field_name', ['main_sha', 'head_sha', 'conflicted_paths'])
def test_base_conflict_is_frozen(field_name):
    conflict = BaseConflict(main_sha='a' * 40, head_sha='b' * 40, conflicted_paths=(GREET_FILE,))

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(conflict, field_name, 'other')


BRANCH = 'task/42'
RESOLVED_GREETING = _greeting('Hello there')


class RecordingMerger:
    """A MergerCall whose side effects are real git commands in the worktree."""

    def __init__(self, act: Callable[[], None], disposition: MergerDisposition) -> None:
        self._act = act
        self._disposition = disposition
        self.briefs: list[str] = []

    async def __call__(self, brief: str) -> MergerDisposition:
        self.briefs.append(brief)
        self._act()
        return self._disposition


def _resolve_and_conclude(worktree: Path) -> Callable[[], None]:
    def act() -> None:
        (worktree / GREET_FILE).write_text(RESOLVED_GREETING)
        _git(worktree, 'add', GREET_FILE)
        _git(worktree, 'commit', '--no-edit')
    return act


def _do_nothing() -> None:
    return None


def _squash_without_main(worktree: Path) -> Callable[[], None]:
    def act() -> None:
        _git(worktree, 'merge', '--abort')
        _commit_file(worktree, GREET_FILE, RESOLVED_GREETING, 'resolve by hand')
    return act


def _rewrite_to_main(worktree: Path) -> Callable[[], None]:
    def act() -> None:
        _git(worktree, 'merge', '--abort')
        _git(worktree, 'reset', '--hard', 'main')
    return act


def _raise_sandbox_unavailable() -> None:
    raise RuntimeError('sandbox unavailable')


def _remove_worktree_then_raise(worktree: Path) -> Callable[[], None]:
    def act() -> None:
        shutil.rmtree(worktree)
        _raise_sandbox_unavailable()
    return act


RESOLVED = MergerDisposition(resolved=True, reason='')
DECLINED = MergerDisposition(resolved=False, reason='needs judgement')


def _has_merge_head(worktree: Path) -> bool:
    probe = subprocess.run(
        ['git', 'rev-parse', '-q', '--verify', 'MERGE_HEAD'], cwd=worktree, capture_output=True,
    )
    return probe.returncode == 0


def _is_ancestor(cwd: Path, ancestor: str, descendant: str) -> bool:
    probe = subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, descendant], cwd=cwd)
    return probe.returncode == 0


def _head(worktree: Path) -> str:
    return _git(worktree, 'rev-parse', 'HEAD')


async def _conflict(repo: Path, git_ops: GitOps, worktree: Path) -> BaseConflict:
    await _conflicting_scene(repo, worktree)
    conflict = await find_base_conflict(git_ops, worktree)
    assert conflict is not None
    return conflict


async def _assert_task_side_restored(git_ops: GitOps, worktree: Path, conflict: BaseConflict) -> None:
    assert not _has_merge_head(worktree)
    assert await git_ops.unmerged_paths(worktree) == ()
    assert _head(worktree) == conflict.head_sha
    assert (worktree / GREET_FILE).read_text() == _greeting('Hi')


@pytest.mark.asyncio
class TestSettleBaseConflict:
    async def _settle(
        self, git_ops: GitOps, worktree: Path, conflict: BaseConflict, merger: RecordingMerger,
    ) -> Settlement:
        return await settle_base_conflict(git_ops, worktree, conflict, branch=BRANCH, merger=merger)

    async def test_resolved_merge_integrates_main_with_both_parents(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        merger = RecordingMerger(_resolve_and_conclude(worktree), RESOLVED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        assert settlement.settled is True
        assert settlement.refusals == ()
        head = _head(worktree)
        assert _is_ancestor(worktree, conflict.main_sha, head)
        assert _is_ancestor(worktree, conflict.head_sha, head)
        assert len(_git(worktree, 'log', '-1', '--format=%P').split()) == 2
        assert _git(worktree, 'log', '-1', '--format=%s') == f'Merge main into {BRANCH}'
        [brief] = merger.briefs
        assert GREET_FILE in brief
        assert conflict.main_sha in brief
        assert '<<<<<<<' in brief

    async def test_declined_merge_is_aborted(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        merger = RecordingMerger(_do_nothing, DECLINED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        assert settlement.refusals == (IntegrationRefusal.MERGER_DECLINED,)
        assert _head(worktree) == conflict.head_sha
        assert not _has_merge_head(worktree)
        assert 'needs judgement' in settlement.block_reason()

    async def test_unconcluded_merge_is_refused_and_aborted(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        merger = RecordingMerger(_do_nothing, RESOLVED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        assert IntegrationRefusal.MERGE_UNCONCLUDED in settlement.refusals
        assert not _has_merge_head(worktree)
        assert await git_ops.unmerged_paths(worktree) == ()
        assert _head(worktree) == conflict.head_sha

    async def test_single_parent_squash_is_refused_and_kept(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        merger = RecordingMerger(_squash_without_main(worktree), RESOLVED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        assert IntegrationRefusal.MAIN_NOT_INTEGRATED in settlement.refusals
        head = _head(worktree)
        assert _git(worktree, 'log', '-1', '--format=%s') == 'resolve by hand'
        assert _git(worktree, 'rev-parse', 'HEAD~1') == conflict.head_sha
        assert head != conflict.head_sha

    async def test_rewritten_history_names_the_lost_tip(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        merger = RecordingMerger(_rewrite_to_main(worktree), RESOLVED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        assert IntegrationRefusal.HISTORY_REWRITTEN in settlement.refusals
        assert conflict.head_sha in settlement.block_detail()

    async def test_detached_worktree_is_refused_untouched(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        _git(worktree, 'checkout', '--detach')
        merger = RecordingMerger(_resolve_and_conclude(worktree), RESOLVED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        assert settlement.refusals == (IntegrationRefusal.BRANCH_NOT_CHECKED_OUT,)
        assert merger.briefs == []
        assert _head(worktree) == conflict.head_sha

    async def test_merge_git_refuses_to_start_keeps_local_edit(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        local_edit = _greeting('Uncommitted')
        (worktree / GREET_FILE).write_text(local_edit)
        merger = RecordingMerger(_resolve_and_conclude(worktree), RESOLVED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        assert settlement.refusals == (IntegrationRefusal.MERGE_NOT_STARTED,)
        assert merger.briefs == []
        assert (worktree / GREET_FILE).read_text() == local_edit
        assert _head(worktree) == conflict.head_sha

    async def test_clean_merge_needs_no_merger(self, repo, git_ops, worktree):
        head_sha = _commit_file(worktree, 'notes.txt', 'task notes\n', 'task: notes')
        main_sha = _commit_file(repo, GREET_FILE, _greeting('Hey'), 'main: greet with Hey')
        conflict = BaseConflict(main_sha=main_sha, head_sha=head_sha, conflicted_paths=(GREET_FILE,))
        merger = RecordingMerger(_resolve_and_conclude(worktree), RESOLVED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        assert settlement.settled is True
        assert merger.briefs == []
        assert _is_ancestor(worktree, main_sha, _head(worktree))

    async def test_refusal_is_informative(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        merger = RecordingMerger(_do_nothing, DECLINED)

        settlement = await self._settle(git_ops, worktree, conflict, merger)

        reason = settlement.block_reason()
        assert IntegrationRefusal.MERGER_DECLINED.value in reason
        assert BRANCH in reason
        assert conflict.head_sha[:8] in reason
        assert conflict.main_sha[:8] in reason
        assert GREET_FILE in reason
        assert len(reason) < 200
        detail = settlement.block_detail()
        assert all(path in detail for path in conflict.conflicted_paths)
        assert conflict.head_sha in detail

    async def test_merger_exception_aborts_the_open_merge(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        merger = RecordingMerger(_raise_sandbox_unavailable, RESOLVED)

        with pytest.raises(RuntimeError, match='sandbox unavailable'):
            await self._settle(git_ops, worktree, conflict, merger)

        assert len(merger.briefs) == 1
        await _assert_task_side_restored(git_ops, worktree, conflict)

    async def test_cancelled_merger_aborts_the_open_merge(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        started = asyncio.Event()
        briefs: list[str] = []

        async def merger(brief: str) -> MergerDisposition:
            briefs.append(brief)
            started.set()
            await asyncio.Event().wait()
            return RESOLVED

        task = asyncio.create_task(
            settle_base_conflict(git_ops, worktree, conflict, branch=BRANCH, merger=merger),
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(briefs) == 1
        await _assert_task_side_restored(git_ops, worktree, conflict)

    async def test_failed_cleanup_does_not_replace_the_merger_exception(self, repo, git_ops, worktree):
        conflict = await _conflict(repo, git_ops, worktree)
        merger = RecordingMerger(_remove_worktree_then_raise(worktree), RESOLVED)

        with pytest.raises(RuntimeError, match='sandbox unavailable'):
            await self._settle(git_ops, worktree, conflict, merger)
