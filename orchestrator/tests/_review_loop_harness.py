"""A TaskWorkflow for driving ``TaskWorkflow._execute_verify_review_loop``.

EXECUTE and VERIFY succeed at once and the reviewer returns a clean aggregation. The architect
replan, the amendment and the suggestion routing are spies. A test sets ``wf._review`` (and
whatever else its branch reaches) and awaits the loop.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from _orch_helpers import pydantic_spec

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

__all__ = ['review_loop_workflow']


def review_loop_workflow(
    tmp_path: Path,
    *,
    task_id: str = '1',
    modules: list[str] | None = None,
    event_store: EventStore | None = None,
    max_review_cycles: int = 2,
    max_amendment_rounds: int = 1,
) -> TaskWorkflow:
    wf = TaskWorkflow(
        assignment=_assignment(task_id, ['src/foo'] if modules is None else modules),
        config=_config(tmp_path, max_review_cycles, max_amendment_rounds),
        git_ops=_clean_tree_git_ops(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=event_store,
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    wf.worktree = worktree
    wf.artifacts = _artifacts_with_a_done_plan(worktree, task_id, wf.session_id)

    wf._execute_iterations = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._get_head_commit = AsyncMock(return_value='tip')
    wf._review = AsyncMock(return_value=ReviewAggregation(
        has_blocking_issues=False, blocking_issues=[], suggestions=[], reviews={},
    ))
    wf._replan = AsyncMock()
    wf._amend = AsyncMock(return_value=True)
    wf._route_review_suggestions_to_curator = AsyncMock()
    wf._write_suggestions_to_memory = AsyncMock()
    return wf


def _assignment(task_id: str, modules: list[str]) -> MagicMock:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = modules
    return assignment


def _config(tmp_path: Path, max_review_cycles: int, max_amendment_rounds: int) -> MagicMock:
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = max_review_cycles
    config.max_amendment_rounds = max_amendment_rounds
    config.lock_depth = 2
    config.project_root = tmp_path / 'proj'
    config.suppress_resettled_review_suggestions = False
    # A real model string keeps TaskWorkflow._reviewer_config_fingerprint deterministic.
    config.models = MagicMock()
    config.models.reviewer = 'sonnet'
    return config


def _clean_tree_git_ops() -> MagicMock:
    git_ops = MagicMock()
    git_ops.get_head_tree_hash = AsyncMock(return_value='TREE1')
    git_ops.get_new_side_changed_line_ranges = AsyncMock(return_value={})
    git_ops.has_uncommitted_work = AsyncMock(return_value=False)
    return git_ops


def _artifacts_with_a_done_plan(worktree: Path, task_id: str, session_id: str) -> TaskArtifacts:
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd')
    artifacts.write_plan({
        'task_id': task_id,
        'title': 'T',
        'files': [],
        'analysis': 'a',
        'prerequisites': [],
        'steps': [{'id': 'step-1', 'type': 'impl', 'description': 'd', 'status': 'done',
                   'commit': 'abc123', 'prerequisites': [], 'files': []}],
    })
    artifacts.stamp_plan_provenance(session_id)
    return artifacts
