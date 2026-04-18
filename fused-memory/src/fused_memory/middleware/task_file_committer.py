"""Auto-commits .taskmaster/tasks/tasks.json after task write operations."""

import asyncio
import contextlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

TASKS_REL_PATH = ".taskmaster/tasks/tasks.json"


class TaskFileCommitter:
    """Git add+commit of tasks.json, serialized per project_root.

    Individual mutations schedule fire-and-forget commits; bulk operations
    (parse_prd, expand_task) should await ``commit()`` directly so the full
    batch is captured before the call returns.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, project_root: str) -> asyncio.Lock:
        if project_root not in self._locks:
            self._locks[project_root] = asyncio.Lock()
        return self._locks[project_root]

    async def commit(self, project_root: str, operation: str) -> None:
        """Git-add and git-commit the tasks file. Serialized, idempotent, never raises."""
        lock = self._get_lock(project_root)
        async with lock:
            try:
                await self._do_commit(project_root, operation)
            except Exception:
                logger.exception(
                    "Auto-commit of tasks.json failed (project_root=%s, op=%s)",
                    project_root,
                    operation,
                )

    async def _do_commit(self, project_root: str, operation: str) -> None:
        tasks_file = Path(project_root) / TASKS_REL_PATH
        if not tasks_file.exists():
            logger.debug("tasks.json does not exist at %s, skipping commit", tasks_file)
            return

        # Guard: refuse to commit if on-disk tasks.json has fewer tasks
        # than HEAD.  This catches stale working-tree snapshots after
        # advance_main() moves the ref without updating the working tree.
        try:
            head_rc, head_stdout, _ = await _run_subprocess(
                ["git", "show", f"HEAD:{TASKS_REL_PATH}"],
                cwd=project_root, timeout=10,
            )
            if head_rc == 0:
                head_data = json.loads(head_stdout)
                disk_data = json.loads(tasks_file.read_text())

                def _count(d: dict) -> int:
                    return len(d.get("master", d).get("tasks", []))

                head_count = _count(head_data)
                disk_count = _count(disk_data)
                if head_count > 0 and disk_count < head_count * 0.9:
                    logger.error(
                        "STALE SNAPSHOT GUARD: on-disk tasks.json has %d tasks "
                        "but HEAD has %d — refusing to commit stale snapshot "
                        "(op=%s). Working tree may be out of sync with HEAD "
                        "after advance_main.",
                        disk_count,
                        head_count,
                        operation,
                    )
                    return
        except Exception:
            logger.debug(
                "Stale snapshot guard check failed, proceeding with commit",
                exc_info=True,
            )

        # Stage the tasks file
        add_rc, _, add_stderr = await _run_subprocess(
            ["git", "add", "--", TASKS_REL_PATH],
            cwd=project_root, timeout=10,
        )
        if add_rc != 0:
            logger.warning("git add failed (rc=%d): %s", add_rc, add_stderr.decode())
            return

        # Check if there are staged changes for this file
        diff_rc, _, _ = await _run_subprocess(
            ["git", "diff", "--cached", "--quiet", "--", TASKS_REL_PATH],
            cwd=project_root, timeout=10,
        )
        if diff_rc == 0:
            # No staged changes — nothing to commit
            logger.debug("tasks.json unchanged, nothing to commit (op=%s)", operation)
            return

        # Commit only the tasks file
        msg = f"chore(tasks): auto-commit after {operation}"
        commit_rc, _, commit_stderr = await _run_subprocess(
            ["git", "commit", "--no-verify", "-m", msg, "--", TASKS_REL_PATH],
            cwd=project_root, timeout=10,
        )
        if commit_rc != 0:
            stderr_text = commit_stderr.decode()
            if "nothing to commit" in stderr_text:
                logger.debug("Nothing to commit for tasks.json (op=%s)", operation)
            else:
                logger.warning("git commit failed (rc=%d): %s", commit_rc, stderr_text)
        else:
            logger.info("Auto-committed tasks.json (op=%s)", operation)


async def _run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: str | None = None,
    timeout: float,
) -> tuple[int, bytes, bytes]:
    """Spawn ``cmd``, await its output, and always reap the child process.

    On TimeoutError or CancelledError, the subprocess is terminated (then killed
    if it doesn't exit within 2 s) and reaped via ``proc.wait()`` before the
    exception propagates. This prevents orphan processes from lingering inside
    the fused-memory cgroup — the WP-A root cause for the 2026-04-17 16-hour
    SQLite lock incident was a stuck ``git show`` child holding open fds for
    4d6h, pinned to the systemd unit's cgroup.

    Returns (returncode, stdout_bytes, stderr_bytes).
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (TimeoutError, asyncio.CancelledError):
            await _terminate_and_reap(proc)
            raise
        return proc.returncode or 0, stdout, stderr
    except BaseException:
        # Last-ditch reap for any other exception path.
        if proc.returncode is None:
            await _terminate_and_reap(proc)
        raise


async def _terminate_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Terminate, then kill if necessary, and always reap the child.

    Uses ``asyncio.shield`` so a second cancellation cannot abandon the
    waitpid call — that would leave the child as a zombie and the asyncio
    child-watcher thread alive (per project memory
    ``feedback_subprocess_cancel_pattern.md``).
    """
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.shield(asyncio.wait_for(proc.wait(), timeout=2.0))
        return
    except (TimeoutError, asyncio.CancelledError):
        pass
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(Exception):
        await asyncio.shield(proc.wait())
