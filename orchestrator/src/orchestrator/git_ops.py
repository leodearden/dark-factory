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
    pre_merge_sha: str | None = None
    merge_worktree: Path | None = None


async def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode if proc.returncode is not None else 1, stdout.decode().strip(), stderr.decode().strip()


class GitOps:
    """Git worktree and merge operations."""

    def __init__(self, config: GitConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.worktree_base = (project_root / config.worktree_dir).resolve()
        self._merge_lock = asyncio.Lock()

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
        await _run(['git', 'add', '-A', '--', '.', ':!.task', ':!.claude', ':!.taskmaster/tasks'], cwd=worktree)

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
        """Merge a task branch into main using a temporary merge worktree.

        Creates a disposable worktree, performs the merge there, and returns
        the result.  The caller is responsible for calling :meth:`advance_main`
        after verification and :meth:`cleanup_merge_worktree` when done.

        Never touches ``project_root``'s working tree or index.
        Caller must hold ``_merge_lock``.
        """
        full_branch = f'{self.config.branch_prefix}{branch}'
        merge_wt: Path | None = None

        try:
            merge_wt, pre_merge_sha = await self._create_merge_worktree()

            # Remove .task/ artifacts — task branches may carry .task/ as
            # tracked files from their base commit.
            task_dir = merge_wt / '.task'
            if task_dir.exists():
                import shutil
                shutil.rmtree(task_dir)

            # Merge with no-ff
            rc, out, err = await _run(
                ['git', 'merge', '--no-ff', full_branch,
                 '-m', f'Merge {full_branch} into {self.config.main_branch}'],
                cwd=merge_wt,
            )

            if rc != 0:
                if 'CONFLICT' in out or 'CONFLICT' in err:
                    conflict_details = await self.get_conflict_details(merge_wt)
                    return MergeResult(
                        success=False, conflicts=True,
                        details=conflict_details,
                        pre_merge_sha=pre_merge_sha,
                        merge_worktree=merge_wt,
                    )
                # Non-conflict failure — clean up immediately
                await self.cleanup_merge_worktree(merge_wt)
                return MergeResult(
                    success=False, details=f'{out}\n{err}',
                    pre_merge_sha=pre_merge_sha,
                )

            _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=merge_wt)
            return MergeResult(
                success=True, merge_commit=sha,
                pre_merge_sha=pre_merge_sha,
                merge_worktree=merge_wt,
            )

        except Exception:
            if merge_wt:
                await self.cleanup_merge_worktree(merge_wt)
            raise

    async def _create_merge_worktree(self) -> tuple[Path, str]:
        """Create a temporary detached worktree at main HEAD for merging."""
        import uuid
        merge_id = uuid.uuid4().hex[:8]
        merge_wt = self.worktree_base / f'_merge-{merge_id}'
        merge_wt.parent.mkdir(parents=True, exist_ok=True)

        # Fetch latest (best-effort — no remote in tests)
        await _run(
            ['git', 'fetch', self.config.remote, self.config.main_branch],
            cwd=self.project_root,
        )

        # Capture current main SHA
        _, pre_merge_sha, _ = await _run(
            ['git', 'rev-parse', self.config.main_branch],
            cwd=self.project_root,
        )

        # Detached worktree avoids "branch already checked out" error
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(merge_wt), self.config.main_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(f'Failed to create merge worktree: {err}')

        logger.info(f'Created merge worktree at {merge_wt} (HEAD={pre_merge_sha[:8]})')
        return merge_wt, pre_merge_sha

    async def cleanup_merge_worktree(self, merge_wt: Path) -> None:
        """Remove a temporary merge worktree."""
        rc, _, err = await _run(
            ['git', 'worktree', 'remove', str(merge_wt), '--force'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(f'Failed to remove merge worktree {merge_wt}: {err}')
        else:
            logger.info(f'Cleaned up merge worktree {merge_wt}')

    async def commit_task_statuses(self) -> str | None:
        """Commit task status changes in the project root. Returns sha or None.

        Only stages ``.taskmaster/tasks/tasks.json`` — no other files are
        touched.  Safe to call when nothing has changed (returns None).
        """
        tasks_file = '.taskmaster/tasks/tasks.json'
        rc, _, _ = await _run(
            ['git', 'diff', '--quiet', '--', tasks_file],
            cwd=self.project_root,
        )
        if rc == 0:
            return None  # no changes

        await _run(
            ['git', 'add', '--', tasks_file],
            cwd=self.project_root,
        )
        rc, _, err = await _run(
            ['git', 'commit', '-m',
             'chore: sync task statuses with orchestrator run'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(f'Failed to commit task statuses: {err}')
            return None

        _, sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=self.project_root,
        )
        return sha

    async def advance_main(self, merge_sha: str) -> bool:
        """Advance main branch ref to *merge_sha* atomically.

        Uses ``update-ref`` so the project_root working tree is never touched.
        Returns False if main has moved past the merge base (caller should
        mark the task as blocked).
        """
        rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor',
             self.config.main_branch, merge_sha],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                f'Cannot fast-forward: {merge_sha[:8]} is not a descendant '
                f'of {self.config.main_branch}'
            )
            return False

        rc, _, err = await _run(
            ['git', 'update-ref',
             f'refs/heads/{self.config.main_branch}', merge_sha],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.error(f'update-ref failed: {err}')
            return False

        logger.info(f'Advanced {self.config.main_branch} to {merge_sha[:8]}')
        return True

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
