"""Real-git boundary suite for the merge content relation (task 10215).

PRD ``plans/one-merge-request-per-task-prd.md`` leaf A: the D10 containment
classifier, the range patch-id reader, and the D6 content relation.  Every case
runs against a genuine temporary repository, because the answers are defined by
what ``git cherry``, ``git patch-id`` and ``git merge-base`` do to real
histories; a mock would only restate this file's assumptions about them.

The file imports merge-lane modules, so the merge-lane ratchet measures it: it
reads no private attribute and patches no lane name.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.landing_evidence import (
    Containment,
    classify_containment,
)

SEED = RepoSeed(
    files=(('README.md', '# content relation fixtures\n'),),
    message='chore: initial commit',
)

#: A well-formed sha that names no object in any fixture repository.
MISSING_SHA = 'deadbeef' * 5


def numbered_lines(prefix: str, count: int, *, start: int = 0) -> str:
    """*count* unique lines, so no two fixture commits share a patch id by coincidence."""
    return ''.join(f'{prefix}_{i:04d} = {i * 7919}\n' for i in range(start, start + count))


class GuardedRepo:
    """A fixture repository whose every git call carries both isolation guards."""

    def __init__(self, root: Path) -> None:
        assert_isolated_git_repo(root)
        self.root = root
        self.env = git_env_with_ceiling(root)

    def git(self, *args: str, input_text: str | None = None) -> str:
        proc = subprocess.run(
            ['git', *args], cwd=str(self.root), capture_output=True,
            env=self.env, text=True, check=False, input=input_text,
        )
        assert proc.returncode == 0, (
            f'git {" ".join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}'
        )
        return proc.stdout.strip()

    def commit(self, message: str, files: dict[str, str]) -> str:
        """Write *files*, stage everything and commit; return the new sha."""
        for rel, content in files.items():
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.sha('HEAD')

    def sha(self, ref: str) -> str:
        return self.git('rev-parse', ref)


@pytest.fixture
def repo(tmp_path: Path) -> GuardedRepo:
    return GuardedRepo(seed_repo(tmp_path / 'repo', SEED))


@pytest.fixture
def git_ops(repo: GuardedRepo) -> GitOps:
    return GitOps(
        GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
        repo.root,
    )


def advance_main(repo: GuardedRepo, name: str) -> str:
    """Commit an unrelated file on main; return main's new tip."""
    repo.git('checkout', 'main')
    return repo.commit(f'chore: main advances ({name})', {f'main/{name}.txt': numbered_lines(name, 5)})


def branch_with_commit(repo: GuardedRepo, branch: str, *, start: str = 'main') -> str:
    """Create *branch* at *start* with one content-bearing commit; return its sha."""
    repo.git('checkout', '-b', branch, start)
    sha = repo.commit(f'feat: {branch} work', {f'work/{branch}.py': numbered_lines(branch, 12)})
    repo.git('checkout', 'main')
    return sha


@pytest.mark.asyncio
class TestClassifyContainment:
    async def test_tip_on_the_target_is_contained(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        tip = branch_with_commit(repo, 'task/merged')
        advance_main(repo, 'before-merge')
        repo.git('merge', '--no-ff', 'task/merged', '-m', 'Merge task/merged into main')

        assert await classify_containment(git_ops, tip=tip, target='main') is Containment.CONTAINED

    async def test_work_landed_by_rebase_is_contained(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        advance_main(repo, 'moved')
        tip = branch_with_commit(repo, 'task/rebased', start='main~1')
        repo.git('cherry-pick', tip)
        assert repo.git('cherry', 'main', tip).startswith('-')

        assert await classify_containment(git_ops, tip=tip, target='main') is Containment.CONTAINED

    async def test_merge_only_tip_beyond_the_target_is_not_contained(
        self, repo: GuardedRepo, git_ops: GitOps,
    ) -> None:
        advance_main(repo, 'merge-parent')
        tip = repo.git('commit-tree', 'main^{tree}', '-p', 'main~1', '-p', 'main', '-m', 'merge')
        assert repo.git('cherry', 'main', tip) == ''

        assert await classify_containment(git_ops, tip=tip, target='main') is Containment.NOT_CONTAINED

    async def test_unlanded_branch_commit_is_not_contained(
        self, repo: GuardedRepo, git_ops: GitOps,
    ) -> None:
        tip = branch_with_commit(repo, 'task/unlanded')
        advance_main(repo, 'unrelated')

        assert await classify_containment(git_ops, tip=tip, target='main') is Containment.NOT_CONTAINED

    async def test_unreadable_tip_is_unknown(self, git_ops: GitOps) -> None:
        assert await classify_containment(git_ops, tip=MISSING_SHA, target='main') is Containment.UNKNOWN
