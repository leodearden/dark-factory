"""Git worktree management for eval isolation."""

import asyncio
import logging
import shutil
from pathlib import Path
from uuid import uuid4

from orchestrator.artifacts import TaskArtifacts
from orchestrator.verify import _target_subprocess_env

logger = logging.getLogger(__name__)


def read_python_pin(root: Path) -> str | None:
    """Return the interpreter pin from ``<root>/.python-version``.

    Reads the file's stripped first line (the version ``uv`` resolves), or
    ``None`` when the file is missing, empty, or unreadable — the fail-safe
    signal to inject no pin (``uv`` falls back to its normal resolution)
    rather than guess. Derived from the target's own ``.python-version`` so
    the eval runner stays correct across targets (df fixtures pin 3.13; other
    targets pin whatever they use) instead of hardcoding a literal.

    Fail-safe on a garbled file: a missing/unreadable file (``OSError``) OR a
    non-UTF-8, undecodable one (``UnicodeDecodeError`` — a ``ValueError``, so
    it would otherwise escape a bare ``except OSError``) both return ``None``
    instead of propagating out of this fail-safe helper and aborting the eval.
    The read is pinned to UTF-8 so the decode outcome is deterministic across
    locales (``.python-version`` is canonical ASCII).
    """
    try:
        raw = (root / '.python-version').read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    lines = raw.splitlines()
    if not lines:
        return None
    return lines[0].strip() or None


def _eval_setup_env(worktree: Path) -> dict[str, str]:
    """Build the env for the eval worktree's ``setup_commands`` (``uv sync``).

    Scrubs the orchestrator's venv activation vars via
    ``verify._target_subprocess_env`` — so ``uv sync`` can't corrupt the live
    orchestrator ``.venv`` (the 2026-05-29 ghost-venv incident) — and pins the
    interpreter to the worktree's own ``.python-version`` (via ``UV_PYTHON``)
    when present, so setup and verify operate on the SAME worktree venv. When
    no pin is found, injects no ``UV_PYTHON`` (fail-safe), matching BUG 2a.
    """
    pin = read_python_pin(worktree)
    return _target_subprocess_env({'UV_PYTHON': pin} if pin else None)


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
) -> tuple[Path, str]:
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

    # Defensive: confirm the worktree HEAD actually matches the requested
    # baseline. Catches any drift in git's worktree handling and prevents
    # silently evaluating against the wrong commit.
    head = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree_path)
    if head != pre_task_commit:
        raise RuntimeError(
            f'create_eval_worktree: HEAD mismatch for {task_id}: '
            f'expected {pre_task_commit}, got {head}'
        )

    logger.info(f'Created eval worktree: {worktree_path} at {pre_task_commit[:10]}')

    # Run setup commands to create isolated env. The env is scrubbed of the
    # orchestrator venv and pinned to the worktree's own interpreter (BUG 2b)
    # so `uv sync` targets <worktree>/.venv, never the live orchestrator .venv.
    if setup_commands:
        setup_env = _eval_setup_env(worktree_path)
        for cmd_str in setup_commands:
            logger.info(f'Eval worktree setup: {cmd_str}')
            proc = await asyncio.create_subprocess_shell(
                cmd_str,
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable='/bin/bash',
                env=setup_env,
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

    # The architect eval writes plan.json to the RELOCATED .task-meta/<name>/
    # root — a SIBLING of the worktree, so `git worktree remove` above does NOT
    # delete it. Remove it here (best-effort, mirroring the worktree removal)
    # so run_architect_eval — the sole caller — leaves no residue behind.
    meta_root = TaskArtifacts.meta_root_for(worktree_path.parent, worktree_path.name)
    shutil.rmtree(meta_root, ignore_errors=True)


async def get_diff(worktree_path: Path, base_commit: str) -> str:
    """Get the full committed diff of an eval worktree vs its base commit.

    ``base_commit`` is the authoritative ``task['pre_task_commit']`` carried
    on the task record — the same base ``metrics._git_diff_stats`` uses. The
    diff is ``git diff {base_commit}..HEAD`` (two-dot range), i.e. the full
    change committed on the ``evals/<id>`` branch.

    This intentionally does NOT read ``<worktree>/.task/metadata.json`` and
    has no uncommitted (``git diff HEAD``) fallback: production TaskWorkflow
    moved its metadata to the sibling ``.task-meta/<name>/`` under W11 / task
    2258, so the old metadata read silently found nothing and the fallback
    graded empty committed diffs (D1). Threading the base is the fix.
    """
    return await _run(
        ['git', 'diff', f'{base_commit}..HEAD'], cwd=worktree_path,
    )


async def get_diff_between_commits(
    project_root: Path, base_commit: str, target_commit: str,
) -> str:
    """Get diff between two commits directly (no worktree needed)."""
    return await _run(
        ['git', 'diff', f'{base_commit}..{target_commit}'],
        cwd=project_root,
    )
