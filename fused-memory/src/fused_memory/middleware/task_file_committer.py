"""Auto-commits .taskmaster/tasks/tasks.json after task write operations."""

import asyncio
import json
import logging
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
            head_proc = await asyncio.create_subprocess_exec(
                "git", "show", f"HEAD:{TASKS_REL_PATH}",
                cwd=project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            head_stdout, _ = await asyncio.wait_for(head_proc.communicate(), timeout=10)
            if head_proc.returncode == 0:
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
        add_proc = await asyncio.create_subprocess_exec(
            "git", "add", "--", TASKS_REL_PATH,
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, add_stderr = await asyncio.wait_for(add_proc.communicate(), timeout=10)
        if add_proc.returncode != 0:
            logger.warning("git add failed (rc=%d): %s", add_proc.returncode, add_stderr.decode())
            return

        # Check if there are staged changes for this file
        diff_proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--cached", "--quiet", "--", TASKS_REL_PATH,
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(diff_proc.communicate(), timeout=10)
        if diff_proc.returncode == 0:
            # No staged changes — nothing to commit
            logger.debug("tasks.json unchanged, nothing to commit (op=%s)", operation)
            return

        # Commit only the tasks file
        msg = f"chore(tasks): auto-commit after {operation}"
        commit_proc = await asyncio.create_subprocess_exec(
            "git", "commit", "--no-verify", "-m", msg, "--", TASKS_REL_PATH,
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(commit_proc.communicate(), timeout=10)
        if commit_proc.returncode != 0:
            stderr_text = stderr.decode()
            if "nothing to commit" in stderr_text:
                logger.debug("Nothing to commit for tasks.json (op=%s)", operation)
            else:
                logger.warning("git commit failed (rc=%d): %s", commit_proc.returncode, stderr_text)
        else:
            logger.info("Auto-committed tasks.json (op=%s)", operation)
