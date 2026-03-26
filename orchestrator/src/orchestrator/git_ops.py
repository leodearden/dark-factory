"""Git worktree and merge operations.

IMPORTANT — .task/ contamination prevention
============================================
The .task/ directory is an ephemeral scratch space for orchestrator agents.
It must NEVER reach the main branch.  If it does, every future worktree
inherits it, agents treat it as state, and cross-task contamination follows.

This module contains multiple redundant safeguards ("belts and braces"):

1. _scrub_task_dir_from_tree() — removes .task/ from the git index in any
   worktree, amending the current commit.  Called after merges and during
   worktree creation.
2. _assert_no_task_dir() — hard assertion that a given commit SHA contains
   no .task/ entries.  Called before advance_main().
3. create_worktree() — scrubs inherited .task/ when main is contaminated.
4. commit() — post-staging safety net: unstages .task/ even if the pathspec
   exclusion (:!.task) was somehow bypassed.
5. merge_to_main() — scrubs .task/ after the merge commit is created.

If you are an AI agent reading this: DO NOT remove or weaken these guards.
They exist because .task/ contamination has happened repeatedly and caused
cascading bugs across all concurrent tasks.  The pre-commit hook, .gitignore,
and .task/.gitignore are NOT sufficient — agents bypass them routinely.
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from orchestrator.config import GitConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .task/ contamination helpers
# ---------------------------------------------------------------------------

async def _scrub_task_dir_from_tree(cwd: Path, context: str, *, amend: bool = True) -> bool:
    """Remove .task/ from the git index if present.

    This is the primary defense against .task/ reaching main.  It checks
    whether any .task/ entries exist in the current HEAD's tree and, if so,
    removes them from the index.

    Args:
        cwd: Working directory (a worktree or the project root).
        context: Human-readable label for log messages (e.g. "post-merge",
                 "worktree-creation").
        amend: If True (default), amend the current commit to exclude .task/.
               If False, create a NEW commit for the removal.  Use amend=False
               when the current HEAD is shared with another branch (e.g. right
               after create_worktree where HEAD == main's tip).

    Returns:
        True if .task/ was found and removed, False if the tree was clean.

    DO NOT REMOVE THIS FUNCTION.  It is the last reliable defense before
    .task/ reaches main via update-ref (which bypasses all git hooks).
    """
    rc, tracked, _ = await _run(
        ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', '.task/'],
        cwd=cwd,
    )
    if rc != 0 or not tracked.strip():
        return False

    files = [f for f in tracked.strip().splitlines() if f.strip()]
    if not files:
        return False

    logger.warning(
        '.task/ CONTAMINATION detected during %s — removing %d tracked file(s): %s',
        context, len(files), ', '.join(files[:10]),
    )

    # Remove from index (not filesystem — .task/ may still be needed as scratch)
    await _run(['git', 'rm', '-r', '--cached', '--', '.task/'], cwd=cwd)

    # Also remove from filesystem if present (cleanup inherited contamination)
    task_dir = cwd / '.task'
    if task_dir.exists():
        shutil.rmtree(task_dir)

    if amend:
        # Amend the current commit to exclude .task/ (used post-merge where
        # we own the merge commit and want a clean tree).
        rc, _, err = await _run(
            ['git', 'commit', '--amend', '--no-edit', '--allow-empty'],
            cwd=cwd,
        )
    else:
        # Create a new commit (used in create_worktree where HEAD is shared
        # with main — amending would rewrite main's history).
        rc, _, err = await _run(
            ['git', 'commit', '-m',
             'chore: remove .task/ contamination inherited from main\n\n'
             '.task/ is the orchestrator scratch directory and must never\n'
             'be on main.  This commit removes it from the branch tree.'],
            cwd=cwd,
        )

    if rc != 0:
        logger.error('.task/ scrub failed during %s: could not commit removal: %s', context, err)
        return False

    logger.info('.task/ scrub completed during %s — %d file(s) removed from tree', context, len(files))
    return True


async def _assert_no_task_dir(sha: str, cwd: Path, context: str) -> None:
    """Raise RuntimeError if the given commit SHA contains any .task/ entries.

    This is a hard gate — if this fires, something upstream failed to scrub
    .task/ and we must NOT advance main.

    DO NOT CATCH THIS EXCEPTION to "work around" it.  Fix the root cause:
    find where .task/ was committed and add a scrub there.
    """
    rc, tracked, _ = await _run(
        ['git', 'ls-tree', '-r', '--name-only', sha, '--', '.task/'],
        cwd=cwd,
    )
    if rc == 0 and tracked.strip():
        files = tracked.strip().splitlines()
        raise RuntimeError(
            f'.task/ CONTAMINATION GATE FAILED ({context}): commit {sha[:8]} '
            f'contains {len(files)} .task/ file(s): {", ".join(files[:5])}. '
            f'Refusing to advance main.  This is a bug — .task/ should have '
            f'been scrubbed before reaching this point.'
        )


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

        # ── .task/ contamination guard ────────────────────────────────
        # If main is contaminated (has .task/ tracked), this worktree
        # inherits it.  Scrub it NOW before any agent code runs, so the
        # task starts from a clean tree.  The scrub amends the initial
        # commit on the new branch — harmless since nothing else has
        # been committed yet.
        # amend=False: HEAD is shared with main — must NOT amend the shared commit.
        # Instead, create a new commit on the branch to remove .task/.
        scrubbed = await _scrub_task_dir_from_tree(worktree_path, 'worktree-creation', amend=False)
        if scrubbed:
            logger.warning(
                'MAIN IS CONTAMINATED — .task/ was inherited by new worktree %s. '
                'The contamination has been removed from this worktree, but main '
                'still carries .task/.  Run: git rm -r --cached .task/ on main.',
                worktree_path,
            )

        return worktree_path, base_sha

    async def commit(self, worktree: Path, message: str) -> str | None:
        """Stage all changes and commit. Returns sha or None if nothing to commit.

        The :!.task pathspec SHOULD prevent .task/ from being staged, but
        agents can (and have) staged .task/ files via direct git commands
        before this method runs.  The post-staging check catches that case.
        """
        # Stage all — :!.task excludes .task/ from staging
        await _run(['git', 'add', '-A', '--', '.', ':!.task', ':!.claude', ':!.taskmaster/tasks'], cwd=worktree)

        # ── Post-staging .task/ safety net ────────────────────────────
        # If .task/ files are staged (e.g. an agent ran "git add .task/"
        # before we got here), unstage them.  This is a belt-and-braces
        # check — the pathspec above should handle it, but agents bypass it.
        rc, staged_task, _ = await _run(
            ['git', 'diff', '--cached', '--name-only', '--', '.task/'],
            cwd=worktree,
        )
        if rc == 0 and staged_task.strip():
            logger.warning(
                '.task/ CONTAMINATION caught in commit() — %d file(s) were staged '
                'despite :!.task pathspec (an agent likely ran "git add .task/" directly). '
                'Unstaging now: %s',
                len(staged_task.strip().splitlines()),
                staged_task.strip()[:200],
            )
            await _run(['git', 'reset', 'HEAD', '--', '.task/'], cwd=worktree)

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

            # Pre-merge cleanup: remove .task/ from filesystem if inherited
            # from a contaminated main.  This is NOT sufficient on its own
            # because `git merge` will re-introduce .task/ from the branch.
            # The real fix is the post-merge scrub below.
            task_dir = merge_wt / '.task'
            if task_dir.exists():
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

            # ── Post-merge .task/ scrub (CRITICAL) ────────────────────
            # The merge commit now exists.  If the task branch had .task/
            # tracked (common — agents commit it despite safeguards), the
            # merge commit contains those files.  We MUST remove them
            # before this commit reaches main via advance_main().
            #
            # _scrub_task_dir_from_tree() checks git ls-tree, runs
            # git rm --cached, and amends the merge commit in-place.
            # This is the single most important .task/ defense.
            await _scrub_task_dir_from_tree(merge_wt, f'post-merge({full_branch})')

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

        IMPORTANT: This method is the LAST checkpoint before code reaches
        main.  update-ref bypasses ALL git hooks (including pre-commit),
        so the .task/ contamination gate here is the final defense.
        """
        # ── .task/ contamination gate (FINAL DEFENSE) ─────────────────
        # update-ref bypasses pre-commit hooks entirely.  This is the
        # absolute last chance to prevent .task/ from reaching main.
        # If this gate fires, something upstream failed to scrub — we
        # refuse to advance and log an error so the root cause can be
        # traced.  DO NOT remove this check or catch the exception.
        try:
            await _assert_no_task_dir(merge_sha, self.project_root, 'advance_main')
        except RuntimeError as e:
            logger.error(str(e))
            return False

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
