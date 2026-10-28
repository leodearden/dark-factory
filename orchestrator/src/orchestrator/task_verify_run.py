"""The executing host's half of a task-role verify: the body of ``orchestrator verify-task``.

PRD: plans/remote-task-leg-verify-prd.md §4.2, §5.3.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from orchestrator.config import OrchestratorConfig
from orchestrator.task_verify_spec import TaskVerifySpec, apply_task_verify_spec
from orchestrator.verify import VerifyResult, run_scoped_verification


def prepare_task_verify_worktree(worktree: Path) -> None:
    """Give *worktree* the fresh, empty ``.task/`` that makes its verify cold.

    verify.py::_is_verify_cold reads cold-vs-warm from ``.task/verify_warmed``.
    A worktree without ``.task/`` is classified WARM, so the cold pre-provision
    (``config.verify_cold_preprovision_command``) is skipped -- the task-6127
    false-red class -- and verify.py::_persist_attempt_logs no-ops.  A
    pre-existing ``.task/`` could carry a warm marker, so it is refused.
    """
    task_dir = worktree / '.task'
    try:
        task_dir.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f'{task_dir} already exists: a task verify worktree must start with a '
            'fresh .task/ so its verify runs cold'
        ) from None


async def run_task_verify_on_worktree(
    worktree: Path, config: OrchestratorConfig, spec: TaskVerifySpec,
) -> VerifyResult:
    """Run *spec*'s task-role verify on *worktree* under this host's *config*.

    Exactly the workstation's task-role verify: no lane flock, no unscoped
    typecheck, no flake suppression.  The returned log paths are this host's,
    meaningless to the dispatcher, so both lists are emptied.
    """
    prepare_task_verify_worktree(worktree)
    applied, modules = apply_task_verify_spec(config, spec)
    result = await run_scoped_verification(
        worktree,
        applied,
        modules,
        list(spec.task_files),
        attempt_id=spec.attempt_id,
        task_id=spec.task_id,
        archive_root=None,
        force_workspace=spec.force_workspace,
        role='task',
    )
    return dataclasses.replace(result, worktree_log_paths=[], archive_log_paths=[])
