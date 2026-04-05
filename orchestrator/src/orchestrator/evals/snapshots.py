"""Git worktree management for eval isolation."""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


async def _run(cmd: list[str], cwd: Path) -> str:
    """Run a command and return stdout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f'Command {" ".join(cmd)} failed (rc={proc.returncode}): '
            f'{stderr.decode().strip()}'
        )
    return stdout.decode().strip()


async def create_eval_worktree(
    project_root: Path | str,
    task_id: str,
    pre_task_commit: str,
    setup_commands: list[str] | None = None,
) -> Path:
    """Create an isolated worktree at the pre-task commit for an eval run.

    After checkout, runs any setup_commands (e.g. 'uv sync') to create
    an isolated environment matching the worktree's source state.

    Returns ``(worktree_path, run_id)`` so callers can include the run ID in
    result filenames for multi-trial support.
    """
    project_root = Path(project_root)
    run_id = uuid4().hex[:8]
    worktree_path = project_root / '.eval-worktrees' / task_id / f'run-{run_id}'
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    await _run(
        ['git', 'worktree', 'add', '--detach', str(worktree_path), pre_task_commit],
        cwd=project_root,
    )

    logger.info(f'Created eval worktree: {worktree_path} at {pre_task_commit[:10]}')

    # Run setup commands to create isolated env
    if setup_commands:
        for cmd_str in setup_commands:
            logger.info(f'Eval worktree setup: {cmd_str}')
            proc = await asyncio.create_subprocess_shell(
                cmd_str,
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable='/bin/bash',
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    f'Setup command failed (rc={proc.returncode}): {cmd_str}\n'
                    f'{stderr.decode()[-500:]}'
                )
            else:
                logger.info(f'Setup command OK: {cmd_str}')

    return worktree_path, run_id


async def cleanup_eval_worktree(
    project_root: Path | str,
    worktree_path: Path,
) -> None:
    """Remove an eval worktree."""
    project_root = Path(project_root)
    try:
        await _run(
            ['git', 'worktree', 'remove', '--force', str(worktree_path)],
            cwd=project_root,
        )
        logger.info(f'Cleaned up eval worktree: {worktree_path}')
    except RuntimeError as e:
        logger.warning(f'Failed to cleanup worktree {worktree_path}: {e}')


async def get_diff(worktree_path: Path) -> str:
    """Get the full diff of changes in a worktree vs its base commit.

    Reads base_commit from .task/metadata.json (set at worktree creation).
    Falls back to uncommitted diff if metadata not available.
    """
    import json as _json

    metadata_file = worktree_path / '.task' / 'metadata.json'
    if metadata_file.exists():
        try:
            meta = _json.loads(metadata_file.read_text())
            base = meta.get('base_commit')
            if base:
                return await _run(
                    ['git', 'diff', f'{base}..HEAD'], cwd=worktree_path,
                )
        except Exception:
            pass
    return await _run(['git', 'diff', 'HEAD'], cwd=worktree_path)


async def get_diff_between_commits(
    project_root: Path, base_commit: str, target_commit: str,
) -> str:
    """Get diff between two commits directly (no worktree needed)."""
    return await _run(
        ['git', 'diff', f'{base_commit}..{target_commit}'],
        cwd=project_root,
    )
