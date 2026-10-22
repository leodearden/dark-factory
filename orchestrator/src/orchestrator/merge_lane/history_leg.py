"""Run a project's declared merge-gate history leg against a merge worktree (task 7248).

A project whose gate verdict is a function of the tree alone, except for a
declared set of history-reading tests, names those tests' command in
``OrchestratorConfig.merge_gate_history_test_command``. On a same-tree rebase
``gates.py::_reverify_rebased_tree`` runs only that command, through
:func:`run_history_leg`, instead of the full re-verify: the tree already
passed, and only a test that reads git history can see the new parent.

The leg runs through ``verify.run_verification`` as a one-command module, so it
keeps every verify invariant: target-env stripping, admission, timeouts and
archive logging.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from orchestrator.config import ModuleConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest
from orchestrator.verify import run_verification

logger = logging.getLogger('orchestrator.merge_lane')

HISTORY_LEG_PREFIX = 'merge-gate-history'


async def run_history_leg(
    git_ops: GitOps,
    req: MergeRequest,
    merge_wt: Path,
    *,
    command: str,
    merge_sha: str,
) -> MergeOutcome | None:
    """Run *command* in *merge_wt*: ``None`` on a pass, else a blocking outcome.

    On a failure *merge_wt* is cleaned up before the outcome is returned, as
    ``worker.py::_run_post_merge_verify`` does for a red re-verify.
    """
    async with contextlib.AsyncExitStack() as stack:
        if req.config.git.persistent_merge_worktree and git_ops.is_persistent_merge_lane(
            merge_wt,
        ):
            await stack.enter_async_context(git_ops.merge_verify_lease())
        verify = await run_verification(
            merge_wt, req.config,
            ModuleConfig(prefix=HISTORY_LEG_PREFIX, test_command=command),
            max_retries=0, is_merge_verify=True, role='merge',
            task_id=req.task_id,
            archive_root=req.config.project_root / 'data' / 'verify-logs',
        )
    if verify.passed:
        return None
    reason = (
        f'Post-merge verification failed: history leg on same-tree rebase '
        f'{merge_sha[:8]}: {verify.summary}'
    )
    if verify.category:
        reason = f'{reason} [category: {verify.category}]'
    logger.warning('Task %s: %s', req.task_id, reason)
    await git_ops.cleanup_merge_worktree(merge_wt)
    return MergeOutcome(
        'blocked', reason=reason,
        failure_category=verify.category,
        failure_cause_hint=verify.cause_hint,
    )
