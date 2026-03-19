"""Git worktree and merge operations."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from orchestrator.config import GitConfig

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    success: bool
    conflicts: bool = False
    details: str = ''
    merge_commit: str | None = None


async def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode().strip(), stderr.decode().strip()


class GitOps:
    """Git worktree and merge operations."""

    def __init__(self, config: GitConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.worktree_base = (project_root / config.worktree_dir).resolve()

    async def create_worktree(self, branch_name: str) -> tuple[Path, str]:
        """Create a git worktree for a task branch, based off main.

        Returns (worktree_path, base_commit_sha) so the base commit is
        captured at creation time and not affected by later main movement.

        If the worktree/branch already exist (e.g., from a requeued task),
        reuses them instead of failing.
        """
        worktree_path = self.worktree_base / branch_name
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        full_branch = f'{self.config.branch_prefix}{branch_name}'

        # Capture main's current SHA before creating worktree
        _, base_sha, _ = await _run(
            ['git', 'rev-parse', self.config.main_branch],
            cwd=self.project_root,
        )

        # If worktree already exists, reuse it (common after requeue)
        if worktree_path.exists():
            logger.info(f'Reusing existing worktree at {worktree_path} on branch {full_branch}')
            return worktree_path, base_sha

        # If branch exists but worktree doesn't (stale from a previous run), clean up
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', full_branch],
            cwd=self.project_root,
        )
        if rc == 0:
            logger.info(f'Cleaning up stale branch {full_branch} before creating worktree')
            await _run(['git', 'branch', '-D', full_branch], cwd=self.project_root)

        # Create worktree with new branch from main
        rc, out, err = await _run(
            ['git', 'worktree', 'add', '-b', full_branch, str(worktree_path), self.config.main_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(f'Failed to create worktree: {err}')

        logger.info(f'Created worktree at {worktree_path} on branch {full_branch} (base={base_sha[:8]})')
        return worktree_path, base_sha

    async def commit(self, worktree: Path, message: str) -> str | None:
        """Stage all changes and commit. Returns sha or None if nothing to commit."""
        # Stage all
        await _run(['git', 'add', '-A', '--', '.', ':!.task', ':!.claude'], cwd=worktree)

        # Check for changes
        rc, _, _ = await _run(['git', 'diff', '--cached', '--quiet'], cwd=worktree)
        if rc == 0:
            return None  # nothing staged

        rc, out, err = await _run(['git', 'commit', '-m', message], cwd=worktree)
        if rc != 0:
            raise RuntimeError(f'Commit failed: {err}')

        # Get sha
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        return sha

    async def get_diff_from_main(self, worktree: Path) -> str:
        """Get diff of worktree branch vs main (dynamic — may be empty if main moved)."""
        _, diff, _ = await _run(
            ['git', 'diff', f'{self.config.main_branch}...HEAD'],
            cwd=worktree,
        )
        return diff

    async def get_diff_from_base(self, worktree: Path, base_commit: str) -> str:
        """Get diff of worktree HEAD vs a fixed base commit.

        Use this instead of get_diff_from_main when main may have advanced
        since the worktree was created (e.g. during review stage).
        """
        _, diff, _ = await _run(
            ['git', 'diff', f'{base_commit}...HEAD'],
            cwd=worktree,
        )
        return diff

    async def get_current_branch(self, worktree: Path) -> str:
        """Get the current branch name in a worktree."""
        _, branch, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=worktree,
        )
        return branch

    async def merge_to_main(self, worktree: Path, branch: str) -> MergeResult:
        """Merge a task branch into main using --no-ff.

        Operates on the main repo (not the worktree) to avoid worktree merge complications.
        """
        full_branch = f'{self.config.branch_prefix}{branch}'

        # Fetch latest main
        await _run(
            ['git', 'fetch', self.config.remote, self.config.main_branch],
            cwd=self.project_root,
        )

        # Checkout main in the main repo
        rc, _, err = await _run(
            ['git', 'checkout', self.config.main_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            return MergeResult(success=False, details=f'Failed to checkout main: {err}')

        # Pull latest
        await _run(
            ['git', 'pull', '--ff-only', self.config.remote, self.config.main_branch],
            cwd=self.project_root,
        )

        # Remove .task/ artifacts from working dir — task branches may carry
        # .task/ as tracked files from their base commit, conflicting with
        # untracked/gitignored .task/ files on disk.
        task_dir = self.project_root / '.task'
        if task_dir.exists():
            import shutil
            shutil.rmtree(task_dir)

        # Merge with no-ff
        rc, out, err = await _run(
            ['git', 'merge', '--no-ff', full_branch, '-m', f'Merge {full_branch} into {self.config.main_branch}'],
            cwd=self.project_root,
        )

        if rc != 0:
            if 'CONFLICT' in out or 'CONFLICT' in err:
                conflict_details = await self.get_conflict_details(self.project_root)
                return MergeResult(success=False, conflicts=True, details=conflict_details)
            return MergeResult(success=False, details=f'{out}\n{err}')

        # Get merge commit
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=self.project_root)
        return MergeResult(success=True, merge_commit=sha)

    async def get_conflict_details(self, cwd: Path) -> str:
        """Parse conflict markers and return structured description."""
        _, status, _ = await _run(['git', 'diff', '--name-only', '--diff-filter=U'], cwd=cwd)
        if not status:
            return 'No conflicting files detected'

        details = [f'Conflicting files:\n{status}\n']
        for filepath in status.splitlines():
            filepath = filepath.strip()
            if filepath:
                _, diff, _ = await _run(['git', 'diff', '--', filepath], cwd=cwd)
                details.append(f'--- {filepath} ---\n{diff[:2000]}')

        return '\n'.join(details)

    async def abort_merge(self, cwd: Path) -> None:
        """Abort an in-progress merge."""
        await _run(['git', 'merge', '--abort'], cwd=cwd)
        logger.info('Merge aborted')

    async def revert_last_merge(self, cwd: Path) -> None:
        """Revert the last merge commit on main."""
        rc, _, err = await _run(
            ['git', 'revert', '--no-commit', '-m', '1', 'HEAD'],
            cwd=cwd,
        )
        if rc != 0:
            raise RuntimeError(f'Failed to revert merge: {err}')
        await _run(
            ['git', 'commit', '-m', 'Revert failed merge — post-merge verification failed'],
            cwd=cwd,
        )
        logger.warning('Reverted last merge commit due to post-merge verification failure')

    async def cleanup_worktree(self, worktree: Path, branch: str) -> None:
        """Remove worktree and delete branch."""
        full_branch = f'{self.config.branch_prefix}{branch}'

        # Remove worktree
        rc, _, err = await _run(
            ['git', 'worktree', 'remove', str(worktree), '--force'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(f'Failed to remove worktree {worktree}: {err}')

        # Delete branch
        rc, _, err = await _run(
            ['git', 'branch', '-D', full_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(f'Failed to delete branch {full_branch}: {err}')

        logger.info(f'Cleaned up worktree {worktree} and branch {full_branch}')
