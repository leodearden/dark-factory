"""An eval corpus cell's task gate: its fixture's own verify_commands, run verbatim.

These are the same commands the scoring verify runs. Production's module
registry and scoping describe today's tree, not the tree at the fixture's
``pre_task_commit``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from orchestrator import verify

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig
    from orchestrator.event_store import EventStore
    from orchestrator.verify import VerifyResult


async def run_fixture_gate(
    worktree: Path,
    config: OrchestratorConfig,
    *,
    attempt_id: int | None = None,
    task_id: str | None = None,
    archive_root: Path | None = None,
) -> VerifyResult:
    """Run *config*'s global commands, which an eval cell config takes from its fixture."""
    return await verify.run_verification(
        worktree, config,
        attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
    )


async def fixture_task_gate(
    worktree: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    task_files: list[str] | None = None,
    *,
    attempt_id: int,
    task_id: str,
    archive_root: Path | None,
    force_workspace: bool = False,
    role: Literal['task'] = 'task',
    event_store: EventStore | None = None,
) -> VerifyResult:
    """``run_fixture_gate`` as the workflow's ``TaskVerifyPort``.

    It deliberately ignores the *module_configs*, *task_files* and
    *force_workspace* the workflow resolved from today's registry.
    """
    return await run_fixture_gate(
        worktree, config,
        attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
    )
