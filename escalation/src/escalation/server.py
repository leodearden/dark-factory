"""Escalation MCP server — FastMCP tools for agents and handlers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from escalation.dedupe import DedupeConfig
from escalation.dedupe import submit_or_dedupe as _dedupe_submit_or_dedupe
from escalation.models import BORN_AT_L2_SEVERITIES, KNOWN_SEVERITIES, Escalation
from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

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
    # Stranded-blocked recovery (PRD-3 D5 / C6)
    'stranded_blocked',
]

# Fields returned by get_pending_escalations(compact=True) — the triage-relevant
# subset a long-running L2 watcher needs to decide whether to pull a full record.
# The heavy fields (detail, members, options, root_cause, train_state,
# workflow_state, worktree, dedupe_*) are dropped to keep the watcher's context
# small as the pending pile grows during an AFK window.
_COMPACT_ESCALATION_FIELDS = (
    'id', 'task_id', 'category', 'severity', 'level', 'status',
    'summary', 'suggested_action', 'timestamp',
)


def create_server(
    queue: EscalationQueue,
    merge_queue: asyncio.Queue | None = None,
    orch_config: Any = None,
    event_store: Any = None,
    harness: Any = None,
    dedupe_config: DedupeConfig | None = None,
    task_status_lookup: Callable[[str], Awaitable[str | None]] | None = None,
    merge_inflight_registry: Any = None,
) -> FastMCP:
    """Create the escalation MCP server with all tools registered.

    *harness* is the running ``orchestrator.harness.Harness``.  When passed,
    it enables the ``release_workflow`` tool which lets external callers
    (humans via /unblock, automation) ask the orchestrator to soft-cancel a
    workflow whose task has been completed out-of-band.  When omitted (e.g.
    in tests with no orchestrator), the tool reports that no workflow is
    active.

    *dedupe_config* controls infra_issue deduplication.  When omitted,
    ``DedupeConfig()`` is used (enabled, 600 s window, infra_issue category).
    Pass ``DedupeConfig(infra_dedupe_enabled=False)`` to disable.

    *task_status_lookup* is an optional async callable ``(task_id) -> str|None``
    that returns the current status of a task.  When provided, ``escalate_blocker``
    and ``escalate_info`` will auto-resolve any escalation whose target task is
    already in a terminal state (``'done'`` or ``'cancelled'``).  When omitted
    (the default), the auto-resolve chokepoint is disabled and all escalations
    are submitted normally.

    *merge_inflight_registry* is an optional ``InFlightMergeRegistry`` injected
    for testing.  When *merge_queue* is not None and no registry is supplied, a
    fresh registry is created lazily inside this function so it is shared across
    all ``merge_request`` calls for the lifetime of the server.  When
    *merge_queue* is None (escalation standalone — orchestrator not wired) the
    registry is never imported, preserving the standalone import path.
    """
    mcp = FastMCP('escalation')
    cfg = dedupe_config if dedupe_config is not None else DedupeConfig()

    # --- Per-branch in-flight de-dup registry for merge_request ---
    # Lazily imported so escalation's standalone typecheck env (which does not
    # have orchestrator on its path) never triggers the import.  The import only
    # fires when merge_queue is wired, i.e. we are running inside the orchestrator
    # process where orchestrator is always importable.
    _registry = merge_inflight_registry
    if merge_queue is not None and _registry is None:
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            InFlightMergeRegistry,
        )
        _registry = InFlightMergeRegistry()

    # --- Shared submit/dedupe helper ---

    def _submit_or_dedupe(esc: Escalation) -> dict[str, Any]:
        """Submit *esc* to the queue, or fold it into an existing pending parent.

        Delegates to ``dedupe.submit_or_dedupe`` which centralises the gate +
        TOCTOU logic so recon (A7b) can reuse the same orchestration without
        duplication.

        Born-at-L2 escalations (``esc.severity in BORN_AT_L2_SEVERITIES``) are
        bypassed from deduplication: they need their own on-disk record stamped
        ``level=2``.  Folding them into an existing parent (which retains its
        original lower level) would silently drop the L2 routing signal.

        Response shapes (from dedupe.submit_or_dedupe):
        - Queued:        ``{'id': esc_id, 'status': 'queued'}``
        - Dedup-skipped: ``{'id': parent_id, 'status': 'dedup_skipped',
                            'parent_id': parent_id, 'child_id': esc.id}``
          (never returned for L2 escalations — they always produce 'queued')

        Cross-task resume contract: see DESIGN.md "Escalation cross-task dedupe"
        — the child re-runs on its next workflow invocation; no per-child wake
        signal is emitted.
        """
        # L2 escalations bypass deduplication: their level=2 stamp must be
        # preserved on an independent on-disk record (see docstring above).
        if esc.severity in BORN_AT_L2_SEVERITIES:
            queue.submit(esc)
            return {'id': esc.id, 'status': 'queued'}
        return _dedupe_submit_or_dedupe(queue, esc, cfg)

    # --- Terminal-task chokepoint helper ---

    async def _chokepoint_or_submit(
        esc: Escalation,
        terminal_state_is_the_bug: bool,
    ) -> dict[str, Any]:
        """Auto-resolve *esc* if the target task is already terminal, else submit normally.

        Gate order (first match wins, all others fall through to _submit_or_dedupe):
          1. terminal_state_is_the_bug=True  → bypass (submit normally)
          2. category == 'review_suggestions' → bypass (A4b owns this category)
          3. task_status_lookup is None       → bypass (chokepoint disabled)
          4. await task_status_lookup(task_id):
               done/cancelled → auto-resolve via submit_resolved (single write,
                              single callback), return minimal {id, status,
                              resolution, resolved_by} dict; blocker adds 'action'
               any other status or None → submit normally
          On any exception from the lookup: fail-open to _submit_or_dedupe (never drop).
        """
        # Severity gate: critical/urgent escalations are born at L2, bypassing
        # the auto-watcher and routing straight to a human (BORN_AT_L2_SEVERITIES).
        # This runs before all other gates so the on-disk record is stamped level=2
        # on every path: queued normally, auto-resolved via submit_resolved, or any
        # gate bypass.  L2 escalations also skip deduplication in _submit_or_dedupe
        # so they are never silently folded into a lower-level parent.
        if esc.severity in BORN_AT_L2_SEVERITIES:
            esc.level = 2

        # Gate 1: semantic bypass — this escalation is expected even for terminal tasks
        if terminal_state_is_the_bug:
            return _submit_or_dedupe(esc)

        # Gate 2: review_suggestions is owned by A4b
        if esc.category == 'review_suggestions':
            return _submit_or_dedupe(esc)

        # Gate 3: chokepoint disabled (no lookup injected)
        if task_status_lookup is None:
            return _submit_or_dedupe(esc)

        # Gate 4: query task status; fail-open on any error
        try:
            status = await task_status_lookup(esc.task_id)
        except Exception as exc:
            logger.warning(
                'task_status_lookup raised for task %s, failing open: %s',
                esc.task_id, exc,
            )
            return _submit_or_dedupe(esc)

        if status in {'done', 'cancelled'}:
            # Atomic submit-as-resolved: single file write, single resolve callback,
            # no transient pending intermediate.  Bypass _submit_or_dedupe to avoid
            # folding into a dedupe parent and resolving the wrong record.
            # Returns minimal shape: {id, status, resolution, resolved_by}.
            # The blocker wrapper adds 'action' separately.
            resolved = queue.submit_resolved(
                esc,
                f'auto-resolved: task already terminal (status={status})',
                resolved_by='escalation-mcp-pre-submit-check',
            )
            return {
                'id': resolved.id,
                'status': resolved.status,
                'resolution': resolved.resolution,
                'resolved_by': resolved.resolved_by,
            }

        # Non-terminal or unknown status → submit normally
        return _submit_or_dedupe(esc)

    # --- Agent-side tools ---

    @mcp.tool()
    async def escalate_info(
        task_id: str,
        agent_role: str,
        category: str,
        summary: str,
        severity: str = 'info',
        detail: str = '',
        suggested_action: str = '',
        worktree: str | None = None,
        workflow_state: str | None = None,
        terminal_state_is_the_bug: bool = False,
    ) -> dict[str, Any]:
        """Report a non-blocking observation. The agent continues working after this call.

        Categories: scope_violation, design_concern, cleanup_needed,
        dependency_discovered, risk_identified, infra_issue.

        *severity* defaults to ``'info'``.  Pass ``'critical'`` or ``'urgent'`` to
        create a born-at-L2 escalation (``models.BORN_AT_L2_SEVERITIES``) that
        bypasses the auto-watcher and routes directly to a human.

        *severity* must be one of ``models.KNOWN_SEVERITIES``
        (``'info'``, ``'blocking'``, ``'critical'``, ``'urgent'``).  Unknown values
        (including case variants such as ``'INFO'`` or ``'CRITICAL'``) are rejected
        with an ``{'error': ...}`` response so misconfigured callers get immediate
        feedback rather than silently-misrouted L0 escalations.

        *terminal_state_is_the_bug* — set True when the escalation is expected even
        if the target task is already terminal (bypasses the auto-resolve chokepoint).

        Response shape:
        - Queued (task alive):    ``{id, status}``  where status='queued'
        - Deduped (folded):       ``{id, status, parent_id, child_id}``
          (L2 escalations are never deduped — they always produce 'queued')
        - Auto-resolved (terminal task): ``{id, status, resolution, resolved_by}``
          Callers needing the full record can call get_escalation(id).
        """
        if severity not in KNOWN_SEVERITIES:
            return {
                'error': (
                    f'invalid severity {severity!r}; '
                    f'expected one of {sorted(KNOWN_SEVERITIES)}'
                ),
            }
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity=severity,
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            worktree=worktree,
            workflow_state=workflow_state,
        )
        # Returns {id, status} or resolved record.
        # No 'action' key — that is only on the blocker path.
        return await _chokepoint_or_submit(esc, terminal_state_is_the_bug)

    @mcp.tool()
    async def escalate_blocker(
        task_id: str,
        agent_role: str,
        category: str,
        summary: str,
        severity: str = 'blocking',
        detail: str = '',
        suggested_action: str = '',
        worktree: str | None = None,
        workflow_state: str | None = None,
        terminal_state_is_the_bug: bool = False,
    ) -> dict[str, Any]:
        """Report a blocking problem. After calling this, commit any in-progress work,
        log your iteration, and STOP. Do NOT retry — the handler will resolve the issue
        and you will be re-invoked.

        Categories: scope_violation, design_concern, cleanup_needed,
        dependency_discovered, risk_identified, infra_issue.

        *severity* defaults to ``'blocking'``.  Pass ``'critical'`` or ``'urgent'`` to
        create a born-at-L2 escalation (``models.BORN_AT_L2_SEVERITIES``) that
        bypasses the auto-watcher and routes directly to a human.

        *severity* must be one of ``models.KNOWN_SEVERITIES``
        (``'info'``, ``'blocking'``, ``'critical'``, ``'urgent'``).  Unknown values
        (including case variants such as ``'BLOCKING'`` or ``'CRITICAL'``) are
        rejected with an ``{'error': ...}`` response so misconfigured callers get
        immediate feedback rather than silently-misrouted escalations.

        *terminal_state_is_the_bug* — set True when the task being blocked is
        expected to be terminal (bypasses the auto-resolve chokepoint and submits
        normally).  action='terminate_cleanly' is still returned.

        Response shape always includes ``action='terminate_cleanly'`` plus:
        - Queued:        ``{id, status, action}``  where status='queued'
        - Deduped:       ``{id, status, parent_id, child_id, action}``
          (L2 escalations are never deduped — they always produce 'queued')
        - Auto-resolved: ``{id, status, resolution, resolved_by, action}``
          Callers needing the full record can call get_escalation(id).
        """
        if severity not in KNOWN_SEVERITIES:
            return {
                'error': (
                    f'invalid severity {severity!r}; '
                    f'expected one of {sorted(KNOWN_SEVERITIES)}'
                ),
            }
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity=severity,
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            worktree=worktree,
            workflow_state=workflow_state,
        )
        result = await _chokepoint_or_submit(esc, terminal_state_is_the_bug)
        return {**result, 'action': 'terminate_cleanly'}

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
        level: int | None = None,
        compact: bool = False,
    ) -> list[dict[str, Any]]:
        """List all pending escalations, optionally filtered by task ID and/or level.

        *level* — when set, returns only escalations at the given escalation level:
          0 = L0 (agent→steward), 1 = L1 (steward/workflow→auto-watcher),
          2 = L2 (auto-watcher→human).
        When omitted, all pending escalations are returned regardless of level.

        *task_id* — when set, restricts the search to escalations for that task.
        Both filters can be combined.

        *compact* — when True, each returned dict is projected to only the
        triage-relevant fields (``id``, ``task_id``, ``category``, ``severity``,
        ``level``, ``status``, ``summary``, ``suggested_action``, ``timestamp``);
        the heavy free-text/cluster fields (``detail``, ``members``, ``options``,
        ``root_cause``, ``train_state``, ``workflow_state``, ``worktree``,
        ``dedupe_*``) are omitted.  Use this from a long-running watcher to keep
        context small as the pending pile grows; fetch the full record for a
        specific id via ``get_escalation`` only when you are about to act on it.
        Default False preserves the full-dict shape for existing callers.
        """
        if task_id:
            escalations = queue.get_by_task(task_id, status='pending', level=level)
        else:
            escalations = queue.get_pending()
            if level is not None:
                escalations = [e for e in escalations if e.level == level]
        if compact:
            return [
                {k: d[k] for k in _COMPACT_ESCALATION_FIELDS}
                for d in (e.to_dict() for e in escalations)
            ]
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

    # --- L2 promotion tool ---

    @mcp.tool()
    async def promote_to_l2(
        task_id: str,
        agent_role: str,
        member_ids: list[str],
        root_cause: str,
        evidence: str,
        options: list[str],
        summary: str,
        category: str = 'design_concern',
        severity: str = 'blocking',
    ) -> dict[str, Any]:
        """Promote one or more L1 escalations to an L2 cluster (human-facing).

        This tool is for the auto-watcher to file a cluster of related L1
        escalations as a single L2 decision point.  The L2 records the member
        L1 ids, the root-cause dedup key, and the proposed options for human
        resolution.  When the human resolves (or dismisses) the L2, the
        resolution cascades automatically to all member L1s.

        **Root-cause dedup**: if a pending L2 with the same *root_cause* already
        exists, this call UPDATES that existing L2 (appends new members) rather
        than filing a duplicate.  The response ``status`` distinguishes the two
        outcomes: ``'created'`` for a new L2, ``'updated'`` for an append.

        **Members stay at L1**: the member L1 escalations are referenced but
        NOT promoted; they remain pending at L1 until the L2 is resolved.

        **Bypasses chokepoint**: this tool calls ``queue.submit()`` directly
        (create path) or ``queue.add_members_to_l2()`` (update path).  The
        terminal-task auto-resolve gate and severity→level=2 gate in
        ``_chokepoint_or_submit`` are intentionally bypassed — L2 is set
        explicitly by this tool.

        Parameters
        ----------
        task_id:
            Passed to ``queue.make_id()`` for id generation.  Typically the
            first member L1's task id.
        agent_role:
            Caller identity, e.g. ``'escalation-watcher-auto'``.
        member_ids:
            Non-empty list of L1 escalation ids forming this cluster.
            Passing an empty list returns ``{'error': ...}``.
        root_cause:
            Non-empty exact-string dedup key.  Whitespace-only input returns
            ``{'error': ...}`` (mirrors ``find_pending_l2_by_root_cause``).
        evidence:
            Supporting context — stored in the escalation's ``detail`` field.
        options:
            Proposed resolution paths, e.g. ``['A: rollback', 'B: fix forward',
            'C: something else']``.
        summary:
            One-line cluster hypothesis.
        category:
            Escalation category; defaults to ``'design_concern'``.
        severity:
            Severity tag; defaults to ``'blocking'``.  Decoupled from
            ``level=2`` — the tool sets ``level=2`` explicitly.  Must be one
            of ``models.KNOWN_SEVERITIES``; unknown values return
            ``{'error': ...}`` (mirrors ``escalate_blocker`` validation).

        Response shapes
        ---------------
        Create (new L2)::

            {'id': <new_id>, 'status': 'created', 'members': [<member_ids>]}

        Update (existing pending L2 with same root_cause)::

            {'id': <existing_id>, 'status': 'updated', 'members': [<all_members>]}

        Error::

            {'error': '<reason>'}
        """
        # Validate required non-empty fields
        if not member_ids:
            return {'error': 'member_ids must be a non-empty list'}
        if not root_cause.strip():
            return {'error': 'root_cause must be a non-empty string'}
        if severity not in KNOWN_SEVERITIES:
            return {
                'error': (
                    f'invalid severity {severity!r}; '
                    f'expected one of {sorted(KNOWN_SEVERITIES)}'
                ),
            }

        # Dedup check: look for an existing pending L2 with the same root_cause.
        existing_id = queue.find_pending_l2_by_root_cause(root_cause)
        if existing_id is not None:
            updated = queue.add_members_to_l2(existing_id, list(dict.fromkeys(member_ids)))
            if updated is not None:
                return {
                    'id': existing_id,
                    'status': 'updated',
                    'members': updated.members,
                }
            # Race: the pending L2 was resolved/archived between find and update.
            # Fall through to the create path so the caller gets a valid result
            # rather than a misleading {'status': 'updated', 'members': []}.
            logger.warning(
                'promote_to_l2: pending L2 %s disappeared during member-update (race); '
                'creating a new L2 for root_cause=%r',
                existing_id, root_cause,
            )

        # Create path: build a fresh L2 and submit it.
        # Deduplicate member_ids via dict.fromkeys so duplicate ids in the input
        # do not create duplicate entries in the on-disk record.
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role=agent_role,
            severity=severity,
            category=category,
            summary=summary,
            detail=evidence,
            level=2,
            members=list(dict.fromkeys(member_ids)),
            root_cause=root_cause.strip(),
            options=list(options),
        )
        queue.submit(esc)
        return {'id': esc.id, 'status': 'created', 'members': esc.members}

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

        Response shapes:
        - Normal outcome: ``{status, reason, conflict_details, push_status}``
          (plus optional ``failure_diagnostic`` on failure).
          ``status`` is one of: ``done``, ``conflict``, ``blocked``,
          ``already_merged``, ``unknown_branch``, ``failed``.
          ``unknown_branch`` means the requested branch has no ref in the
          target repo — usually a merge_request mis-routed to the wrong
          repo's escalation MCP; check that the branch belongs here.
        - Already in flight: ``{status='in_flight', branch, inflight_task_id,
          eta_seconds, reason, conflict_details=None, push_status=None}``.
          A merge for *branch* is already running; the caller should poll
          rather than re-queuing.  ``eta_seconds`` is a best-effort hint
          (``None`` once the estimate window is exceeded).
        """
        if merge_queue is None:
            return {'error': 'Merge queue not available — orchestrator not running'}
        if orch_config is None:
            return {'error': 'Merge queue available but no orchestrator config — cannot verify'}

        # Runtime-only reverse import: orchestrator depends on escalation, not
        # vice versa, so this lazy import deliberately avoids a static cycle. It
        # resolves at runtime because the escalation server is hosted inside the
        # orchestrator process; it is unresolvable in escalation's standalone
        # typecheck env (orchestrator is not on its path), hence the suppression.
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeOutcome,
            MergeRequest,
            coalesce_or_enqueue_merge_request,
        )

        # module_configs_or_empty normalises the post-1405 None sentinel (direct-
        # instantiation configs never call load_config, so _module_configs stays None).
        # See OrchestratorConfig.module_configs_or_empty (config.py) for details.
        module_configs = list(orch_config.module_configs_or_empty.values())
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
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

        # De-dup gate: consults the in-memory registry (and optionally the on-disk
        # _merge-* worktree scan via harness.git_ops) before enqueuing.  On coalesce
        # returns immediately with in_flight=True — no future await, no duplicate
        # enqueue.  On dispatch acquires the registry slot and awaits the future
        # exactly as the original enqueue_merge_request path.
        git_ops_for_scan = getattr(harness, 'git_ops', None)
        dispatch = await coalesce_or_enqueue_merge_request(
            merge_queue,
            merge_req,
            event_store,
            _registry,
            git_ops=git_ops_for_scan,
        )

        if dispatch.in_flight:
            # Branch already being merged — return in_flight immediately so the
            # caller can poll rather than block.  ETA is a best-effort heuristic
            # (None once the estimate window is exceeded).
            # conflict_details / push_status are included with None for shape
            # stability: callers that access result['conflict_details'] or
            # result['push_status'] must not KeyError on a coalesced response.
            return {
                'status': 'in_flight',
                'branch': branch,
                'inflight_task_id': dispatch.inflight_task_id,
                'eta_seconds': dispatch.eta_seconds,
                'reason': (
                    f'A merge for branch {branch!r} is already in flight '
                    f'(source={dispatch.source!r}). Poll for completion.'
                ),
                'conflict_details': None,
                'push_status': None,
            }

        outcome = await future
        result: dict[str, Any] = {
            'status': outcome.status,
            'reason': outcome.reason,
            'conflict_details': outcome.conflict_details,
            'push_status': outcome.push_status,
        }
        if outcome.failure_diagnostic is not None:
            result['failure_diagnostic'] = outcome.failure_diagnostic
        return result

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

        Once the workflow slot has cleared, if the task is still
        ``in-progress`` (the typical /unblock shape: an escalated task whose
        agent was paused) it is parked as ``blocked``.  ``blocked`` is the
        reaper-immune holding state — it stops the orchestrator from
        re-dispatching the task AND protects the worktree from the stranded-
        in-progress reconciliation sweep while the human finishes the work.
        From ``blocked`` the final ``set_task_status('done')`` after merge is
        the normal blocked→done transition.

        Returns:
            ``{released, was_active, slot_cleared, parked}``
            - ``was_active``: True if a workflow slot was registered when
              the call started.
            - ``released``: True if ``cancel_workflow`` accepted the request.
            - ``slot_cleared``: True if the workflow finished within
              ``timeout_secs``.
            - ``parked``: the status the task was parked into (``'blocked'``)
              once the slot cleared, or ``None`` if no park occurred (slot
              still active, or task already terminal/non-in-progress).
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
                'parked': None,
            }
        # Wait up to timeout_secs for the slot to clear
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(0, int(timeout_secs))
        while harness.is_workflow_active(task_id):
            if loop.time() >= deadline:
                break
            await asyncio.sleep(0.5)
        slot_cleared = not harness.is_workflow_active(task_id)

        # Park the task in a reaper-immune status once the slot has cleared.
        # Only when the slot is gone (a still-live slot is already reaper-safe
        # via dispatch-table membership; parking it would race the workflow).
        # An escalated /unblock task sits at 'in-progress' with an open L1
        # while the human works; without this, release_workflow clears the
        # only sweep protection (dispatch-table membership via scheduler.release)
        # without changing the persisted status, leaving the worktree exposed
        # to the stranded-in-progress reaper.  'blocked' is the existing
        # sweep-immune state (symmetric with the already-safe blocked-task path)
        # and won't trip park-stop (threshold is 15 blocked-transitions/hour).
        parked = None
        if not harness.is_workflow_active(task_id):
            cur = await harness.scheduler.get_status(task_id)
            if cur == 'in-progress':
                await harness.scheduler.set_task_status(task_id, 'blocked')
                parked = 'blocked'

        return {
            'released': released,
            'was_active': was_active,
            'slot_cleared': slot_cleared,
            'parked': parked,
        }

    @mcp.tool()
    async def unhalt_merge_queue(reason: str) -> dict[str, Any]:
        """Force-unhalt the orchestrator merge queue when a halt was orphaned.

        REFUSES to act if the halt has an active owning escalation — for those,
        use resolve_issue(escalation_id, resolution).  Use this tool only when
        get_merge_halt_status reports halted=True with owner_esc_id=None or a
        stale owner_esc_id whose escalation no longer exists / is already
        resolved.
        """
        if harness is None:
            return {
                'unhalted': False,
                'error': 'escalation server running standalone — no harness wired',
            }
        if not reason or not reason.strip():
            return {'unhalted': False, 'error': 'reason is required for audit'}
        return harness.force_unhalt_merge_queue(reason.strip())

    @mcp.tool()
    def get_merge_halt_status() -> dict[str, Any]:
        """Inspect the orchestrator merge queue's halt state."""
        if harness is None:
            return {'wired': False, 'error': 'escalation server running standalone'}
        return harness.get_merge_halt_status()

    @mcp.tool()
    def get_merge_queue() -> dict[str, Any]:
        """Return a live read-only snapshot of the merge worker's queue/pipeline state.

        Shows all in-flight and queued merge requests: entries that are queued
        (waiting for the merger), merging (merger is building a merge commit),
        awaiting_verify (waiting in the verifier queue), verifying (active
        verification), gate_reverify (re-verifying after a rebase), or
        finalizing (CAS-advancing main).

        This snapshot captures the blind spot missed by runs.db-events: a
        genuinely-queued entry with no _merge-* worktree and no merge events.
        """
        if merge_queue is None:
            return {'error': 'Merge queue not available — orchestrator not running'}
        worker = getattr(harness, '_merge_worker', None)
        if worker is None or not hasattr(worker, 'snapshot'):
            return {'error': 'Merge worker not available'}
        return worker.snapshot()

    @mcp.tool()
    async def resume_scheduler(reason: str) -> dict[str, Any]:
        """Resume a paused orchestrator scheduler in-process (no restart).

        The scheduler park-stops (or cost-ceiling / EWA-trips) by setting an
        in-memory pause that ALSO persists to runs.db, so the pause survives
        restarts.  This clears both: dispatch resumes on the next idle tick
        (~15s), no restart needed.

        Normally you resolve the scheduler-pause L1 instead — that now
        auto-resumes the scheduler.  Use this tool for the orphan case: the
        pause has no open escalation (it was dismissed, or filing failed), or
        you need to force a resume directly.  Idempotent; ``reason`` is
        required for audit.

        Returns ``{resumed, was_paused, prior_reason, reason}`` (or an
        ``error`` when the server is running standalone with no harness wired).
        """
        if harness is None:
            return {
                'resumed': False,
                'error': 'escalation server running standalone — no harness wired',
            }
        if not reason or not reason.strip():
            return {'resumed': False, 'error': 'reason is required for audit'}
        return await harness.force_resume_scheduler(reason.strip())

    return mcp
