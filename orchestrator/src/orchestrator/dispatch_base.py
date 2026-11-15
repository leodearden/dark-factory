"""Integrate current main into a resumed task branch that conflicts with it, before planning (task 7736).

The caller is ``orchestrator/src/orchestrator/workflow.py::TaskWorkflow._settle_conflicting_base``.
This module imports downward only, from ``orchestrator.git_ops``,
``orchestrator.merge_lane.outcome_facts`` and ``shared.git_async``; nothing it
imports imports it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.git_async import GitResult, run_git

from orchestrator.git_ops import GitOps, GitQueryFailed
from orchestrator.merge_lane.outcome_facts import ConflictEvidence

_GIT_TIMEOUT_SECS = 120.0


@dataclass(frozen=True)
class BaseConflict:
    """A resumed branch tip that does not merge cleanly onto main."""

    main_sha: str
    head_sha: str
    conflicted_paths: tuple[str, ...]

    def evidence(self) -> ConflictEvidence:
        """The conflict as 7302's evidence; ``()`` counterparts means against main itself."""
        return ConflictEvidence(self.conflicted_paths, counterpart_task_ids=())

    def to_event_payload(self) -> dict[str, Any]:
        return {
            'main_sha': self.main_sha,
            'head_sha': self.head_sha,
            **self.evidence().to_event_payload(),
        }


async def find_base_conflict(git_ops: GitOps, worktree: Path) -> BaseConflict | None:
    """The conflict *worktree*'s HEAD has with main, or None when it merges cleanly.

    The predicate is ``GitOps.merge_tree_conflicts``, the one the merge
    queue's suffix tracker bounces on. A git error raises; it is not a verdict.
    """
    main_sha = await git_ops.get_main_sha()
    head_sha = await worktree_head(worktree)
    probe = await git_ops.merge_tree_conflicts(main_sha, head_sha)
    if probe.clean:
        return None
    return BaseConflict(main_sha, head_sha, tuple(probe.conflicted_paths))


async def worktree_head(worktree: Path) -> str:
    """*worktree*'s HEAD as a sha, for GitOps methods that run at project_root."""
    result = await _git(worktree, 'rev-parse', 'HEAD')
    if not result.ok or not result.stdout:
        raise GitQueryFailed(f'cannot resolve HEAD in {worktree}: {result.stderr}')
    return result.stdout


async def _git(worktree: Path, *args: str) -> GitResult:
    return await run_git(['git', *args], cwd=worktree, timeout=_GIT_TIMEOUT_SECS)
