"""Tests for cross-dispatch review/amendment persistence (task 2749).

Two mechanisms drive ``_execute_verify_review_loop``:

1. A tree-hash-keyed verdict cache (``review_state.json`` via
   ``TaskArtifacts``): on entering REVIEW, if a non-blocking verdict is
   already recorded for HEAD's committed tree hash, the reviewer invocation
   is SKIPPED entirely and the loop takes the DONE path WITHOUT re-routing
   suggestions.
2. Task-lifetime counters (``amendment_rounds_total`` /
   ``review_cycles_total``): the loop seeds its per-dispatch locals from the
   persisted totals so ``max_amendment_rounds`` / ``max_review_cycles`` bound
   the whole task lifetime, not each dispatch.

These tests drive the loop with ``_execute_iterations`` /
``_verify_debugfix_loop`` / ``_review`` stubbed, mirroring the
``_make_workflow`` harness in ``test_workflow.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowOutcome,
)


def _make_workflow(
    *,
    tmp_path: Path,
    task_id: str = '2749',
    modules: list[str] | None = None,
    max_amendment_rounds: int = 1,
    max_review_cycles: int = 2,
) -> TaskWorkflow:
    """Minimal TaskWorkflow harness for driving _execute_verify_review_loop."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = modules if modules is not None else ['src/foo']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = max_review_cycles
    config.max_amendment_rounds = max_amendment_rounds
    config.lock_depth = 2
    config.project_root = tmp_path / 'proj'

    scheduler = MagicMock()
    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd')
    wf.artifacts = artifacts
    wf.worktree = worktree

    # Loop sub-steps default to success; the REVIEW branch is what each test
    # exercises, so stub EXECUTE/VERIFY to DONE.
    wf._execute_iterations = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)

    # Tree-hash source: a stable committed tree hash.
    wf.git_ops.get_head_tree_hash = AsyncMock(return_value='TREE1')

    # Spies on the REVIEW-branch side effects.
    wf._review = AsyncMock(
        return_value=ReviewAggregation(
            has_blocking_issues=False,
            blocking_issues=[],
            suggestions=[],
            reviews={},
        )
    )
    wf._route_review_suggestions_to_curator = AsyncMock()
    wf._write_suggestions_to_memory = AsyncMock()
    wf._amend = AsyncMock(return_value=True)
    return wf


@pytest.mark.asyncio
class TestVerdictCacheSkip:
    """A pre-recorded non-blocking verdict for the current tree SKIPs REVIEW."""

    async def test_cached_verdict_skips_review_and_routing(
        self, tmp_path: Path
    ):
        wf = _make_workflow(tmp_path=tmp_path)
        # Simulate a prior dispatch that already recorded a suggestions_only
        # verdict (and routed its suggestions) for this committed tree.
        wf.artifacts.record_review_verdict('TREE1', 'suggestions_only', True)

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        # The skip path re-mints nothing: no reviewer, no re-routing, no amend.
        wf._review.assert_not_called()
        wf._route_review_suggestions_to_curator.assert_not_called()
        wf._write_suggestions_to_memory.assert_not_called()
        wf._amend.assert_not_called()
