"""Support for tests that hand-drive a ``SpeculativeMergeWorker``."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Literal

from _orch_helpers import STOP_WORKER_JOIN_TIMEOUT, make_placeholder_future

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult
from orchestrator.merge_lane.types import QueuedBranch
from orchestrator.merge_lane.worker import (
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    SpeculativeMergeWorker,
)


def make_merge_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    pre_rebased: bool = False,
    lane: Literal['normal', 'high'] = 'normal',
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop."""
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane=lane,
    )


def make_real_merge_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    task_id: str,
    base_sha: str,
) -> RealMergeItem:
    """Build a dispatch-ready, non-speculative RealMergeItem over a fresh
    MergeRequest whose merge has notionally succeeded -- the single
    construction site for tests that hand a worker an item without running
    a real merge.
    """
    req = make_merge_request(task_id, f'task/{task_id}', git_ops.project_root, config)
    wt = git_ops.project_root / '.worktrees' / task_id
    return RealMergeItem(
        request=req,
        merge_result=MergeResult(success=True, merge_commit='deadbeef', merge_worktree=wt),
        merge_wt=wt,
        base_sha=base_sha,
        speculative=False,
    )


async def stop_merge_worker(
    worker: SpeculativeMergeWorker,
    worker_task: asyncio.Task[None],
    *,
    join_timeout: float = STOP_WORKER_JOIN_TIMEOUT,
) -> None:
    """Shut *worker* down and join its run task -- the ONE teardown shape
    every merge-worker test module shares, so no site can drift or be
    forgotten.

    ALWAYS call this from a ``finally:`` whose ``try:`` encloses every wait in
    the test body (task 3980 amendment, esc-3980-4). ``wait_responsive``
    gives up by raising ``_pytest.outcomes.Failed``, and an assertion
    mid-body raises too; on the old straight-line shape either one skipped
    ``stop()`` entirely and leaked a live merge worker plus its run task into
    pytest-asyncio teardown. That leak is why one red test used to cascade
    into unrelated failures elsewhere in the session.
    ``_wait_hygiene_scan.py::unguarded_worker_teardown_items`` enforces
    the ``finally:``.

    Safe on the give-up path even when a gate was never released: ``stop()``
    cancels every in-flight verify task rather than awaiting it
    (``orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.stop``),
    so it cannot itself block on an unreleased ``asyncio.Event``.  Whatever
    the lane still has to unwind on the way out keeps seeing the injected
    verifier, which is the lane's own collaborator for its whole lifetime
    rather than a binding swapped in for a block.

    The join stays best-effort (``suppress(Exception)``): it asserts nothing,
    and a slow join must not convert a real failure above into a confusing
    second one. It is also why this wait is exempt from the shared
    ``wall-clock-deadline`` rule
    (fused-memory/scripts/check_bare_magicmock_config.py): its target is a
    bare Name, not a ``.result`` future or a zero-argument ``.wait()`` barrier, so the
    exemption is structural rather than a listed name. It is still a real
    wait, so ``_wait_budget_audit.py::_call_wait_budget`` bills its
    *join_timeout*.
    """
    await worker.stop()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(worker_task, timeout=join_timeout)


# The shared per-test ceiling for every class whose waits outgrow the ambient
# budget. SHARED, not per-class, so the marks cannot drift apart (task 3477
# found two that had; task 3492 generalized CASCADE_TEST_TIMEOUT into this).
#
# No test's budget is written here or beside its mark:
# test_wait_budget_coverage_guard.py recomputes each one from source on every
# run and checks it against the closest timeout mark, or against the ambient
# budget for an unmarked test -- so leaving a class unmarked is a checked
# decision too.
#
# A literal, pinned by test_merge_worker_drive.py::TestHeavyBarrierTimeoutConstant
# against the measured worst case below x UNDER_LOAD_HEADROOM_FACTOR and against
# PYPROJECT_DEFAULT_TIMEOUT (never-narrow).  Derivation:
# plans/pytest-per-test-timeout-measurement-2026-09-17.md.
HEAVY_BARRIER_TEST_TIMEOUT = 540

# Worst per-test wall clock -- setup+call+teardown, what a pytest-timeout mark
# bounds -- of any HEAVY_BARRIER- or capstone-marked test: task 5572, 2026-10-08,
# `-n auto` at loadavg 152 -> 101,
# test_merge_speculation.py::TestSpecLaneAbortReleasesLane::test_operator_halt_releases_spec_lane.
# Corpus: plans/pytest-per-test-timeout-measurement-2026-09-17.md.
HEAVY_BARRIER_MEASURED_UNDER_LOAD_WORST_CASE = 14.59
