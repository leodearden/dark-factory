"""Plan-tools MCP server -- stdio tool server for safe plan.json operations.

Spawned per-agent invocation by the orchestrator.  The architect agent
builds plans via ``create_plan`` / ``add_plan_step`` / etc., and the
implementer marks steps done via ``mark_step_done``.  All writes go
through ``TaskArtifacts`` methods, preserving ``_session_id`` and
enforcing correct schema.

Usage (stdio transport, spawned by orchestrator):
    uv run --project <orchestrator-dir> python -m orchestrator.mcp.plan_tools \
        --worktree /path/to/worktree
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from orchestrator.artifacts import TaskArtifacts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone implementation functions (testable without MCP transport)
# ---------------------------------------------------------------------------


def _create_plan(
    artifacts: TaskArtifacts,
    task_id: str,
    title: str,
    analysis: str,
    modules: list[str],
    files: list[str] | None = None,
) -> dict[str, Any]:
    plan = {
        'task_id': task_id,
        'title': title,
        'analysis': analysis,
        'modules': modules,
        'files': files or [],
        'prerequisites': [],
        'steps': [],
        'design_decisions': [],
        'reuse': [],
    }
    artifacts.write_plan(plan)
    return {'status': 'ok', 'task_id': task_id}


def _add_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
    step_type: str,
    description: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    # Reject duplicate step IDs
    existing_ids = {
        s.get('id')
        for col in ('prerequisites', 'steps')
        for s in plan.get(col, [])
        if isinstance(s, dict)
    }
    if step_id in existing_ids:
        return {'status': 'error', 'message': f'Step {step_id!r} already exists in plan.'}

    plan.setdefault('steps', []).append({
        'id': step_id,
        'type': step_type,
        'description': description,
        'status': 'pending',
        'commit': None,
    })
    artifacts.write_plan(plan)
    return {'status': 'ok', 'step_id': step_id, 'total_steps': len(plan['steps'])}


def _add_prerequisite(
    artifacts: TaskArtifacts,
    prereq_id: str,
    description: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    existing_ids = {
        s.get('id')
        for col in ('prerequisites', 'steps')
        for s in plan.get(col, [])
        if isinstance(s, dict)
    }
    if prereq_id in existing_ids:
        return {'status': 'error', 'message': f'Prerequisite {prereq_id!r} already exists.'}

    plan.setdefault('prerequisites', []).append({
        'id': prereq_id,
        'description': description,
        'status': 'pending',
        'commit': None,
        'tests': [],
    })
    artifacts.write_plan(plan)
    return {'status': 'ok', 'prereq_id': prereq_id}


def _add_design_decision(
    artifacts: TaskArtifacts,
    decision: str,
    rationale: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    plan.setdefault('design_decisions', []).append({
        'decision': decision,
        'rationale': rationale,
    })
    artifacts.write_plan(plan)
    return {'status': 'ok', 'total_decisions': len(plan['design_decisions'])}


def _add_reuse_item(
    artifacts: TaskArtifacts,
    what: str,
    where: str,
    how: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    plan.setdefault('reuse', []).append({
        'what': what,
        'where': where,
        'how': how,
    })
    artifacts.write_plan(plan)
    return {'status': 'ok', 'total_reuse': len(plan['reuse'])}


def _mark_step_done(
    artifacts: TaskArtifacts,
    step_id: str,
    commit_sha: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    for collection in ('prerequisites', 'steps'):
        for item in plan.get(collection, []):
            if isinstance(item, dict) and item.get('id') == step_id:
                artifacts.update_step_status(step_id, 'done', commit=commit_sha)
                return {
                    'status': 'ok',
                    'step_id': step_id,
                    'new_status': 'done',
                    'commit': commit_sha,
                }
    return {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'}


# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------


def create_server(artifacts: TaskArtifacts) -> FastMCP:
    """Create the plan-tools MCP server with all tools registered."""
    mcp = FastMCP('plan-tools')

    @mcp.tool()
    def create_plan(
        task_id: str,
        title: str,
        analysis: str,
        modules: list[str],
        files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Initialize a new implementation plan with metadata.

        Call this first, before adding steps or prerequisites.
        Creates .task/plan.json with the provided metadata and empty
        step/prerequisite arrays.

        Args:
            task_id: The task identifier (e.g. "df_task_13").
            title: Human-readable task title.
            analysis: Your analysis of the task, existing code, and approach.
            modules: Code directories this task will touch.
            files: ALL files expected to be created or modified (drives concurrency locks).
        """
        return _create_plan(artifacts, task_id, title, analysis, modules, files)

    @mcp.tool()
    def add_plan_step(
        step_id: str,
        step_type: str,
        description: str,
    ) -> dict[str, Any]:
        """Add a TDD step to the plan. Steps execute in the order added.

        Args:
            step_id: Unique step ID (e.g. "step-1", "step-2").
            step_type: Either "test" or "impl".
            description: What this step does.
        """
        return _add_plan_step(artifacts, step_id, step_type, description)

    @mcp.tool()
    def add_prerequisite(
        prereq_id: str,
        description: str,
    ) -> dict[str, Any]:
        """Add a prerequisite to the plan. Prerequisites run before TDD steps.

        Use for setup work like config files, fixtures, etc.

        Args:
            prereq_id: Unique prerequisite ID (e.g. "pre-1").
            description: What this prerequisite sets up.
        """
        return _add_prerequisite(artifacts, prereq_id, description)

    @mcp.tool()
    def add_design_decision(
        decision: str,
        rationale: str,
    ) -> dict[str, Any]:
        """Record a design decision and its rationale in the plan.

        Args:
            decision: The decision that was made.
            rationale: Why this decision was made over alternatives.
        """
        return _add_design_decision(artifacts, decision, rationale)

    @mcp.tool()
    def add_reuse_item(
        what: str,
        where: str,
        how: str,
    ) -> dict[str, Any]:
        """Record an existing code pattern or utility being reused.

        Args:
            what: What is being reused (e.g. "MergeResult dataclass pattern").
            where: Where it exists (e.g. "orchestrator/src/orchestrator/git_ops.py:14-18").
            how: How it will be reused in this task.
        """
        return _add_reuse_item(artifacts, what, where, how)

    @mcp.tool()
    def mark_step_done(
        step_id: str,
        commit_sha: str,
    ) -> dict[str, Any]:
        """Mark a plan step or prerequisite as done with its commit SHA.

        Use this instead of directly editing .task/plan.json.
        Only the status and commit fields are updated; all other plan
        structure (including provenance metadata) is preserved.

        Args:
            step_id: The step or prerequisite ID (e.g. "step-1", "pre-1").
            commit_sha: The git commit SHA for this step's changes.
        """
        return _mark_step_done(artifacts, step_id, commit_sha)

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse --worktree and run the stdio MCP server."""
    parser = argparse.ArgumentParser(description='Plan-tools MCP server (stdio)')
    parser.add_argument(
        '--worktree', type=Path, required=True,
        help='Path to the git worktree containing .task/',
    )
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    if not (worktree / '.task').is_dir():
        print(f'Error: {worktree / ".task"} does not exist', file=sys.stderr)
        sys.exit(1)

    artifacts = TaskArtifacts(worktree)
    server = create_server(artifacts)
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
