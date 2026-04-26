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
    orch_config: Any = None,
    event_store: Any = None,
    harness: Any = None,
) -> FastMCP:
    """Create the escalation MCP server with all tools registered.

    *harness* is the running ``orchestrator.harness.Harness``.  When passed,
    it enables the ``release_workflow`` tool which lets external callers
    (humans via /unblock, automation) ask the orchestrator to soft-cancel a
    workflow whose task has been completed out-of-band.  When omitted (e.g.
    in tests with no orchestrator), the tool reports that no workflow is
    active.
    """
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
        if orch_config is None:
            return {'error': 'Merge queue available but no orchestrator config — cannot verify'}

        from orchestrator.merge_queue import MergeOutcome, MergeRequest, enqueue_merge_request

        module_configs = list(orch_config._module_configs.values())
        future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        merge_req = MergeRequest(
            task_id=task_id,
            branch=branch,
            worktree=Path(worktree),
            pre_rebased=False,
            task_files=None,
            module_configs=module_configs,
            config=orch_config,
            result=future,
        )
        await enqueue_merge_request(merge_queue, merge_req, event_store)

        outcome = await future
        return {
            'status': outcome.status,
            'reason': outcome.reason,
            'conflict_details': outcome.conflict_details,
            'push_status': outcome.push_status,
        }

    @mcp.tool()
    async def release_workflow(
        task_id: str,
        timeout_secs: int = 30,
    ) -> dict[str, Any]:
        """Soft-cancel an active workflow for ``task_id``.

        Use this when you have completed a task out-of-band (typical: marked
        it ``done`` via a manual merge in /unblock) and want the orchestrator
        to stop processing it.  The workflow re-reads task status and exits
        ``DONE`` if terminal, ``REQUEUED`` otherwise — never creates new
        escalations as a result of this call.

        Returns:
            ``{released, was_active, slot_cleared}``
            - ``was_active``: True if a workflow slot was registered when
              the call started.
            - ``released``: True if ``cancel_workflow`` accepted the request.
            - ``slot_cleared``: True if the workflow finished within
              ``timeout_secs``.
        """
        if harness is None:
            return {
                'released': False, 'was_active': False, 'slot_cleared': False,
                'error': 'No orchestrator harness wired in — running in standalone mode',
            }
        was_active = harness.is_workflow_active(task_id)
        released = harness.cancel_workflow(task_id)
        if not was_active:
            return {
                'released': False, 'was_active': False, 'slot_cleared': True,
            }
        # Wait up to timeout_secs for the slot to clear
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(0, int(timeout_secs))
        while harness.is_workflow_active(task_id):
            if loop.time() >= deadline:
                break
            await asyncio.sleep(0.5)
        slot_cleared = not harness.is_workflow_active(task_id)
        return {
            'released': released,
            'was_active': was_active,
            'slot_cleared': slot_cleared,
        }

    return mcp
