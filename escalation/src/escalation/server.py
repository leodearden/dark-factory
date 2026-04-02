"""Escalation MCP server — FastMCP tools for agents and handlers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from escalation.models import Escalation
from escalation.queue import EscalationQueue

CATEGORIES = [
    'scope_violation',
    'design_concern',
    'cleanup_needed',
    'dependency_discovered',
    'risk_identified',
    'infra_issue',
    'task_failure',
    # Reconciliation categories
    'recon_failure',
    'recon_backlog_overflow',
    'recon_stale_run',
    'recon_integrity_issue',
    # Review triage
    'review_suggestions',
]


def create_server(
    queue: EscalationQueue,
    merge_queue: asyncio.Queue | None = None,
) -> FastMCP:
    """Create the escalation MCP server with all tools registered."""
    mcp = FastMCP('escalation')

    # --- Agent-side tools ---

    @mcp.tool()
    def escalate_info(
        task_id: str,
        agent_role: str,
        category: str,
        summary: str,
        detail: str = '',
        suggested_action: str = '',
        worktree: str | None = None,
        workflow_state: str | None = None,
    ) -> dict[str, Any]:
        """Report a non-blocking observation. The agent continues working after this call.

        Categories: scope_violation, design_concern, cleanup_needed,
        dependency_discovered, risk_identified, infra_issue.
        """
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity='info',
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            worktree=worktree,
            workflow_state=workflow_state,
        )
        esc_id = queue.submit(esc)
        return {'id': esc_id, 'status': 'queued'}

    @mcp.tool()
    def escalate_blocker(
        task_id: str,
        agent_role: str,
        category: str,
        summary: str,
        detail: str = '',
        suggested_action: str = '',
        worktree: str | None = None,
        workflow_state: str | None = None,
    ) -> dict[str, Any]:
        """Report a blocking problem. After calling this, commit any in-progress work,
        log your iteration, and STOP. Do NOT retry — the handler will resolve the issue
        and you will be re-invoked.

        Categories: scope_violation, design_concern, cleanup_needed,
        dependency_discovered, risk_identified, infra_issue.
        """
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity='blocking',
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            worktree=worktree,
            workflow_state=workflow_state,
        )
        esc_id = queue.submit(esc)
        return {'id': esc_id, 'status': 'queued', 'action': 'terminate_cleanly'}

    # --- Handler-side tools ---

    @mcp.tool()
    def resolve_issue(
        escalation_id: str,
        resolution: str,
        terminate: bool = False,
        resolved_by: str | None = None,
        resolution_turns: int | None = None,
    ) -> dict[str, Any]:
        """Resolve or dismiss an escalation. The resolution text is injected into the
        agent's briefing when the task resumes.

        Set terminate=true to abandon the task rather than resume it.
        Use resolved_by to attribute the resolver (e.g. "steward", "interactive").
        Use resolution_turns to record how many conversation turns resolution took.
        """
        esc = queue.resolve(
            escalation_id, resolution, dismiss=terminate,
            resolved_by=resolved_by, resolution_turns=resolution_turns,
        )
        if esc is None:
            return {'error': f'Escalation {escalation_id} not found'}
        return esc.to_dict()

    @mcp.tool()
    def get_pending_escalations(
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all pending escalations, optionally filtered by task ID."""
        if task_id:
            escalations = queue.get_by_task(task_id, status='pending')
        else:
            escalations = queue.get_pending()
        return [e.to_dict() for e in escalations]

    @mcp.tool()
    def get_escalation(
        escalation_id: str,
    ) -> dict[str, Any]:
        """Get a single escalation by ID."""
        esc = queue.get(escalation_id)
        if esc is None:
            return {'error': f'Escalation {escalation_id} not found'}
        return esc.to_dict()

    # --- Merge queue tools ---

    @mcp.tool()
    async def merge_request(
        task_id: str,
        branch: str,
        worktree: str,
        description: str = '',
    ) -> dict[str, Any]:
        """Submit a merge request to the orchestrator merge queue.

        Use this instead of directly merging into main.  The merge worker
        handles verification, conflict detection, and atomic ref advancement.
        Returns the merge outcome (done, conflict, blocked, already_merged).
        """
        if merge_queue is None:
            return {'error': 'Merge queue not available — orchestrator not running'}

        from orchestrator.config import OrchestratorConfig
        from orchestrator.merge_queue import MergeOutcome, MergeRequest

        future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        await merge_queue.put(MergeRequest(
            task_id=task_id,
            branch=branch,
            worktree=Path(worktree),
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=OrchestratorConfig(),
            result=future,
        ))

        outcome = await future
        return {
            'status': outcome.status,
            'reason': outcome.reason,
            'conflict_details': outcome.conflict_details,
        }

    return mcp
