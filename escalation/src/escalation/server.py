"""Escalation MCP server — FastMCP tools for agents and handlers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from escalation import sweep as _sweep
from escalation.dedupe import DedupeConfig
from escalation.dedupe import submit_or_dedupe as _dedupe_submit_or_dedupe
from escalation.models import BORN_AT_L2_SEVERITIES, KNOWN_SEVERITIES, Escalation
from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel role allowlist — resolves PRD Open question 2 (C4/D3)
# ---------------------------------------------------------------------------
# All harness-internal sentinels use one of these prefixes:
#   harness-*      → harness-stranded-blocked-reaper, harness-reconcile,
#                    harness-orphan-reaper (harness.py)
#   orchestrator-* → orchestrator-scheduler, orchestrator-watcher-supervisor
#                    (the watcher-outage L2 named in the user-observable signal)
# Verified against orchestrator/src/orchestrator/agents/roles.py: NO LLM agent
# role (architect, implementer, debugger, merger, steward, deep_reviewer,
# reviewer_comprehensive, judge, simple_task) uses either prefix.
# A prefix check is forward-compatible: new harness sentinels are auto-exempt.
_HARNESS_SENTINEL_ROLE_PREFIXES = ('harness-', 'orchestrator-')

# Maximum number of seconds the server will block waiting for a merge outcome.
# Callers may pass a larger wait_secs value; the server silently clamps it to
# this limit so MCP framework timeouts (≈120 s) are never breached.
# Tests monkeypatch this constant to a tiny value (e.g. 0.1) to exercise the
# clamp+timeout branch in milliseconds.
_MAX_WAIT_SECS: int | float = 100


def _is_harness_sentinel_role(agent_role: str) -> bool:
    """Return True if *agent_role* belongs to the harness sentinel namespace.

    Defensively coerces *agent_role* via ``(agent_role or '')`` so that
    ``None`` or empty strings (possible on legacy/deserialized records) fall
    through to the downgrade path instead of raising ``AttributeError``.
    """
    return any((agent_role or '').startswith(p) for p in _HARNESS_SENTINEL_ROLE_PREFIXES)


# C1 action enum for resolve_issue — five valid values, two disposition buckets.
RESOLVE_ACTIONS: tuple[str, ...] = ('resume', 'restart', 'park', 'abandon', 'close_only')
_DISMISS_ACTIONS: frozenset[str] = frozenset({'park', 'abandon', 'close_only'})

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
    startup_sweep: bool = True,
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

    *startup_sweep* (default True) — when True, runs
    ``sweep.run_startup_sweep(queue.queue_dir)`` at construction time (the
    pre-serving single-writer window) to relocate resolved/dismissed root
    orphans and loose archive files, and prune stale dated subdirs.  Non-fatal:
    any exception is logged at WARNING level and the server continues binding.
    Pass False in tests that pre-populate the queue and do not want the sweep
    to run.
    """
    mcp = FastMCP('escalation')
    cfg = dedupe_config if dedupe_config is not None else DedupeConfig()

    # --- Startup sweep (pre-serving single-writer window) ---
    if startup_sweep:
        try:
            _sweep.run_startup_sweep(queue.queue_dir)
        except Exception as _e:
            logger.warning('startup queue sweep failed (non-fatal): %s', _e)

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

    # Server-side durable-intent waiter store (β1 D2/I5).
    # Keyed by request_id; each entry is a WaiterRecord holding the shielded
    # future.  Cleaned up automatically via future.add_done_callback when the
    # entry resolves or is cancelled.  β2 (merge_cancel) looks up entries here
    # to cancel a specific in-flight request by id.
    _waiters: dict[str, Any] = {}

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
        # C4/D3: Agent-role severity downgrade — runs FIRST, before the born-at-L2
        # stamp, so the existing level=2 gate and the _submit_or_dedupe L2-bypass
        # both naturally observe 'blocking' and route the downgraded record through
        # the normal L0 + dedupe path.  Harness sentinel roles (harness-* /
        # orchestrator-*) are exempt and keep their born-at-L2 routing.
        if esc.severity in BORN_AT_L2_SEVERITIES and not _is_harness_sentinel_role(esc.agent_role):
            _original_severity = esc.severity
            esc.severity = 'blocking'
            # Marker appended (not prepended) so summary_dedupe_key's first-three-token
            # slice stays equal to the original summary's key.  Downgraded criticals
            # then fold into the equivalent normally-filed 'blocking' parent, and
            # unrelated issues with the same first two words don't false-merge on a
            # constant leading '[downgraded:...]' token (PRD C4 — marker on the
            # summary line, placed at the suffix to preserve the key).
            # Known edge case: summaries with fewer than 3 real tokens have the
            # marker leak into the dedupe key (e.g. 'lost link' → key becomes
            # ('lost','link','downgradedcritical') vs. ('lost','link')), so a
            # downgraded short-summary won't fold into its blocking parent.
            # Impact: missed dedupe for short one-line summaries only (no data
            # corruption).  A marker-aware fix in summary_dedupe_key (dedupe.py)
            # — stripping a trailing '[downgraded:...]' before the 3-token slice —
            # would close this gap; out of scope for α2.
            esc.summary = f'{esc.summary} [downgraded:{_original_severity}]'
            logger.warning(
                'Downgraded severity %r → blocking for agent_role=%r task_id=%r (C4/D3)',
                _original_severity, esc.agent_role, esc.task_id,
            )

        # Severity gate: critical/urgent escalations are born at L2, bypassing
        # the auto-watcher and routing straight to a human (BORN_AT_L2_SEVERITIES).
        # This runs before all other gates so the on-disk record is stamped level=2
        # on every path: queued normally, auto-resolved via submit_resolved, or any
        # gate bypass.  L2 escalations also skip deduplication in _submit_or_dedupe
        # so they are never silently folded into a lower-level parent.
        # After the downgrade above, only sentinel-filed criticals/urgents reach here.
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
        action: str = 'resume',
        resolved_by: str | None = None,
        resolution_turns: int | None = None,
        terminate: Any = None,
    ) -> dict[str, Any]:
        """Resolve or dismiss an escalation.

        ``action`` selects the resolution intent (default: ``'resume'``):

        +--------------+------------+----------------------------+------------------------+
        | action       | record     | live-workflow effect       | task-status effect     |
        +==============+============+============================+========================+
        | resume       | resolved   | resume from pause point;   | stays in-progress      |
        |              |            | resolution text injected   |                        |
        |              |            | into agent briefing (L0    |                        |
        |              |            | path only)                 |                        |
        +--------------+------------+----------------------------+------------------------+
        | restart      | resolved   | restart task from scratch  | reset to pending       |
        |              |            | (harness β handles)        | (harness β handles)    |
        +--------------+------------+----------------------------+------------------------+
        | park         | dismissed  | task left paused; no       | stays blocked/deferred |
        |              |            | immediate workflow action  |                        |
        +--------------+------------+----------------------------+------------------------+
        | abandon      | dismissed  | task cancelled outright    | cancelled              |
        |              |            | (harness β handles)        | (harness β handles)    |
        +--------------+------------+----------------------------+------------------------+
        | close_only   | dismissed  | escalation closed with no  | unchanged              |
        |              |            | workflow effect            |                        |
        +--------------+------------+----------------------------+------------------------+

        **Note — task-status effects are not yet wired (pending harness β).**
        Currently park, abandon, and close_only are indistinguishable at the
        harness layer: all three produce ``status='dismissed'`` on the escalation
        record.  The per-action task-status transitions described above will be
        realised when harness β lands.

        **Resolution text** reaches the agent only on the L0 live-resume path
        (``action='resume'``).  For all other actions the text is stored on the
        record for audit purposes but is not injected into any agent briefing.

        **Legacy mapping** (D10): callers that resolve without ``resolution_action``
        (i.e. records where ``resolution_action`` is ``None`` after the fact) are
        interpreted as follows — ``dismiss=False`` (old ``terminate=False``) maps
        to ``resume``; ``dismiss=True`` (old ``terminate=True``) maps to
        ``close_only``.

        ``terminate`` has been removed.  Passing any value raises a migration error
        naming the five replacement actions.

        ``resolved_by`` attributes the resolver (e.g. ``"steward"``, ``"interactive"``).
        ``resolution_turns`` records how many conversation turns resolution took.
        """
        if terminate is not None:
            return {
                'error': (
                    "'terminate' was removed; state your intent: "
                    "action='resume'|'restart'|'park'|'abandon'|'close_only' "
                    "— see resolve_issue docstring."
                )
            }
        if action not in RESOLVE_ACTIONS:
            return {'error': f'invalid action {action!r}; expected one of {list(RESOLVE_ACTIONS)}'}
        # Pre-stamp resolution_action on the pending record so resolve()'s
        # read-modify-write carries it into the archived JSON (C1 persistence).
        # Guard: only rewrite pending records — archived records must not be resurrected.
        pending = queue.get(escalation_id)
        if pending is None:
            return {'error': f'Escalation {escalation_id} not found'}
        if pending.status == 'pending':
            pending.resolution_action = action
            queue._rewrite(escalation_id, pending)
        dismiss = action in _DISMISS_ACTIONS
        esc = queue.resolve(
            escalation_id, resolution, dismiss=dismiss,
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
        wait_secs: int = 0,
    ) -> dict[str, Any]:
        """Submit a merge request to the orchestrator merge queue.

        Use this instead of directly merging into main.  The merge worker
        handles verification, conflict detection, and atomic ref advancement.

        Invariant I1: no merge_request code path awaits longer than
        ``_MAX_WAIT_SECS`` (100 s).  The unbounded blocking path is deleted.

        *wait_secs* controls how long the call blocks:
        - ``0`` (default): return immediately — dispatched branch returns
          ``status='queued'``; coalesced branch returns ``status='attached'``.
          Shape: ``{status, request_id, snapshot_tip, generation, position,
          queue_depth, eta_seconds}``.
        - ``>0``: server-clamped to ``≤_MAX_WAIT_SECS`` (100 s); bounded
          wait via ``asyncio.wait_for(asyncio.shield(future), clamp)``.
          Resolves within clamp → terminal outcome shape.
          Timeout → non-terminal ``status='queued'`` shape (shield ensures
          expiry never cancels the entry's future).

        Response shapes:
        - Normal outcome: ``{status, request_id, reason, conflict_details,
          push_status}`` (plus optional ``failure_diagnostic`` on failure).
          ``status`` is one of: ``done``, ``conflict``, ``blocked``,
          ``already_merged``, ``unknown_branch``, ``failed``.
          ``unknown_branch`` means the requested branch has no ref in the
          target repo — usually a merge_request mis-routed to the wrong
          repo's escalation MCP; check that the branch belongs here.
          ``request_id`` is the stable per-entry identity of this request
          (e.g. ``'mr-a1b2c3d4'``).
        - Queued: ``{status='queued', request_id, snapshot_tip, generation,
          position, queue_depth, eta_seconds}``.  Branch was freshly dispatched
          (or wait_secs timeout expired).
        - Attached: ``{status='attached', request_id, snapshot_tip, generation,
          position, queue_depth, eta_seconds, inflight_task_id}``.  Branch is
          already in-flight; request_id is the *existing* entry's id (D8), not
          the submitting call's id.  ``inflight_task_id`` is the authoritative
          poll handle (merge_status accepts task_id per D10).
        - Already merged: ``{status='already_merged', commit, reason='',
          conflict_details='', push_status=None}``.  Either the branch tip is
          already an ancestor of main (fast-path — no enqueue, no request_id)
          or the worker detected the branch was already merged via merge marker
          (worker-path — also carries request_id and a None commit from
          outcome.merge_sha).  All keys are present in both paths; callers
          can safely read reason/conflict_details/push_status without KeyError.
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
            WaiterRecord,
            coalesce_or_enqueue_merge_request,
        )

        # Single git_ops handle reused for both the already_merged fast-path
        # (below) and the coalesce-gate disk-scan (coalesce_or_enqueue call).
        # git_ops=None means no harness is wired (standalone / tests without
        # orchestrator) — both paths degrade gracefully to no-op.
        git_ops_for_scan = getattr(harness, 'git_ops', None)

        # PRD I4 — already_merged fast-path: if the branch tip is already an
        # ancestor of main, the submission is guaranteed-redundant.  Return
        # {status:'already_merged', commit} immediately — NO enqueue, NO
        # merge_queued event, NO asyncio.Future created (avoids an orphan future).
        # resolve_branch_sha uses the full ref 'task/<branch>' per the worker
        # convention (merge_queue.py:3796, git_ops.py:1461).  A missing branch
        # (tip=None) falls through to the normal enqueue so the worker still
        # emits its unknown_branch outcome, preserving existing semantics.
        # git_ops=None (standalone) skips the fast-path entirely.
        # The resolved tip is also stored as merge_req.snapshot_tip (β1 D8).
        resolved_tip: str | None = None
        if git_ops_for_scan is not None:
            full_branch = f'{orch_config.git.branch_prefix}{branch}'
            resolved_tip = await git_ops_for_scan.resolve_branch_sha(full_branch)
            if resolved_tip is not None and await git_ops_for_scan.is_ancestor(
                resolved_tip, orch_config.git.main_branch
            ):
                # Shape converged with worker-path already_merged (suggestion 1).
                # request_id is absent: the fast-path short-circuits before any
                # MergeRequest entry is constructed (no entry → no id).
                return {
                    'status': 'already_merged',
                    'commit': resolved_tip,
                    'reason': '',
                    'conflict_details': '',
                    'push_status': None,
                }

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
            snapshot_tip=resolved_tip,
        )

        # De-dup gate: consults the in-memory registry (and optionally the on-disk
        # _merge-* worktree scan via harness.git_ops) before enqueuing.  On coalesce
        # returns immediately with in_flight=True — no future await, no duplicate
        # enqueue.  On dispatch acquires the registry slot and awaits the future
        # exactly as the original enqueue_merge_request path.
        dispatch = await coalesce_or_enqueue_merge_request(
            merge_queue,
            merge_req,
            event_store,
            _registry,
            git_ops=git_ops_for_scan,
        )

        def _nonblocking_state_response(
            status: str,
            req: Any,
            req_id_override: str | None = None,
        ) -> dict[str, Any]:
            """Build the non-blocking state response shape (§7.1).

            Used by both wait_secs==0 and the wait_secs>0 TimeoutError branch.
            position/queue_depth come from the live worker snapshot matched by
            request_id; falls back to merge_queue.qsize() when no worker is
            reachable (standalone / unit tests that wire a bare asyncio.Queue).
            eta_seconds from the in-flight registry; generation is always 0 in β1.
            """
            request_id_val = req_id_override if req_id_override is not None else req.request_id
            worker = getattr(harness, '_merge_worker', None)
            position: int = 0
            queue_depth: int = merge_queue.qsize()  # type: ignore[union-attr]
            if worker is not None:
                try:
                    snap = worker.snapshot()
                    entries = snap.get('entries', [])
                    queue_depth = snap.get('depth', len(entries))
                    for i, e in enumerate(entries):
                        if e.get('request_id') == req.request_id:
                            position = i
                            break
                    else:
                        position = max(0, queue_depth - 1)
                except Exception:
                    pass
            else:
                # No live worker: queue_depth already holds merge_queue.qsize() from
                # the initialiser above; only position needs to be set.
                position = max(0, queue_depth - 1)
            eta = _registry.eta_seconds(branch) if _registry is not None else None
            return {
                'status': status,
                'request_id': request_id_val,
                'snapshot_tip': req.snapshot_tip,
                'generation': 0,
                'position': position,
                'queue_depth': queue_depth,
                'eta_seconds': eta,
            }

        if dispatch.in_flight:
            # Branch already being merged — return 'attached' with the existing
            # entry's request_id (D8) so the caller can correlate with the entry,
            # not the coalesced submission.  The 'in_flight' response is retired
            # (β8 Open Q5): no skill or surviving test reads status=='in_flight'.
            base = _nonblocking_state_response(
                'attached', merge_req,
                req_id_override=dispatch.inflight_request_id,
            )
            base['inflight_task_id'] = dispatch.inflight_task_id
            return base

        # Register durable-intent waiter record (β1 D2/I5).
        # shield(future) means cancelling the merge_request coroutine (MCP
        # disconnect) no longer cancels req.result — _request_abandoned fires
        # only for explicit merge_cancel (β2).
        _waiters[merge_req.request_id] = WaiterRecord(
            request_id=merge_req.request_id,
            future=future,
            source='mcp',
            submitted_tip=merge_req.snapshot_tip,
        )
        future.add_done_callback(
            lambda _f: _waiters.pop(merge_req.request_id, None)
        )

        # Non-positive / None → immediate non-blocking return — 'queued' shape.
        # Handles: wait_secs==0 (default), and the retired None sentinel.
        if not wait_secs:
            return _nonblocking_state_response('queued', merge_req)

        # wait_secs > 0: bounded wait — clamp to _MAX_WAIT_SECS (module constant,
        # tests monkeypatch to a small value like 0.1 s to exercise this path fast).
        # asyncio.shield(future) decouples req.result from the tool coroutine's
        # cancellation: a wait_for timeout cancels only the outer shield wrapper,
        # leaving req.result alive.  On timeout return the non-terminal 'queued'
        # shape so the caller can poll (PRD I1/Open Q4).
        clamp = min(wait_secs, _MAX_WAIT_SECS)
        try:
            outcome = await asyncio.wait_for(asyncio.shield(future), clamp)
        except TimeoutError:
            return _nonblocking_state_response('queued', merge_req)
        # Resolved within clamp → fall through to terminal outcome shape below.
        # 'commit' (outcome.merge_sha) is included for shape convergence with the
        # fast-path already_merged response.  It is None for most statuses and for
        # the worker-produced already_merged case (merge marker path returns no SHA);
        # it is non-None for 'done' and 'done_wip_recovery' where main was advanced.
        result: dict[str, Any] = {
            'status': outcome.status,
            'request_id': merge_req.request_id,
            'reason': outcome.reason,
            'conflict_details': outcome.conflict_details,
            'push_status': outcome.push_status,
            'commit': outcome.merge_sha,
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

    # ── merge_status — read-only lifecycle probe (PRD α3 / task 1630) ──────────

    _MERGE_STATUS_UNKNOWN_HINT = 'check git log main'

    def _epoch_to_iso8601(ts: float) -> str:
        """Convert an epoch-seconds float to an ISO-8601 UTC string (matches event-store format)."""
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()

    def _map_terminal_state(raw: str) -> str:
        """Map a raw terminal MergeOutcome.status / 'abandoned' / 'error' to coarse vocabulary."""
        if raw in ('done', 'done_wip_recovery', 'already_merged'):
            return 'done'
        if raw == 'conflict':
            return 'conflict'
        if raw == 'abandoned':
            return 'abandoned'
        # blocked / wip_halted / wip_recovery_no_advance / unmerged_state /
        # unknown_branch / error → blocked
        return 'blocked'

    def _map_live_state(raw: str) -> str:
        """Map a live snapshot state to the public merge_status vocabulary."""
        if raw == 'queued':
            return 'queued'
        if raw in ('merging', 'awaiting_verify', 'verifying'):
            return 'verifying'
        if raw == 'gate_reverify':
            return 'gate'
        if raw == 'finalizing':
            return 'finalizing'
        return raw  # pass through unknown states unchanged

    def _durable_terminal_state(
        request_id: str | None,
        branch: str | None = None,
        task_id: str | None = None,
    ) -> tuple[str, dict] | None:
        """Consult durable terminal tiers (retention ring → event store).

        Shared implementation of merge_status Tiers 2–3 and merge_cancel's
        waiter-absent resolution — keeps both tools' tier ordering and state
        vocabulary in sync.

        Returns ``(coarse_state, meta)`` when a durable record is found, else
        ``None``.  *meta* is a dict with keys:
            request_id  — the id from the resolved record (may differ from the
                          input request_id when resolved via branch/task_id).
            outcome     — raw state string from the record.
            finished_at — ISO-8601 string (ring records are normalised from their
                          epoch-float via _epoch_to_iso8601; event-store rows are
                          already strings).

        Tier 1 (live snapshot) is intentionally omitted — callers that need it
        handle it themselves.  merge_cancel intentionally skips Tier 1 because a
        request absent from _waiters cannot be a live waiter of this server (a
        live entry's future is still pending and thus still registered).
        """
        # Tier 2: retention ring (request_id-keyed lookup only).
        # finished_at is stored as epoch float; normalise to ISO-8601 so the
        # same logical merge returns the same type regardless of which tier serves it.
        ring = getattr(harness, '_terminal_retention', None) if harness is not None else None
        if ring is not None and request_id is not None:
            rec = ring.get(request_id)
            if rec is not None:
                return _map_terminal_state(rec.state), {
                    'request_id': rec.request_id,
                    'outcome': rec.state,
                    'finished_at': _epoch_to_iso8601(rec.finished_at),
                }

        # Tier 3: event store (supports all three lookup keys).
        if event_store is not None:
            row = event_store.latest_merge_finalized(
                request_id=request_id,
                branch=branch,
                task_id=task_id,
            )
            if row is not None:
                return _map_terminal_state(row['state']), {
                    'request_id': row['request_id'],
                    'outcome': row['state'],
                    'finished_at': row['finished_at'],
                }

        return None

    @mcp.tool()
    async def merge_status(
        request_id: str | None = None,
        branch: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the current merge state for a merge request.

        Accepts one of: request_id (authoritative), branch (most-recent), or
        task_id (most-recent).  Lookup order: live snapshot → retention ring →
        event store → {state:'unknown', hint}.

        Returns a dict with at minimum:
            state, request_id, generation (always 1 in α3).

        Live entries also carry: position, enqueued_at, eta_seconds.
        Terminal entries carry: outcome (raw state), finished_at.
        Unknown carries: hint.
        """
        # Validation — at least one key required
        if request_id is None and branch is None and task_id is None:
            return {'error': 'At least one of request_id, branch, or task_id is required'}

        # Tier 1: live snapshot — wrapped fire-safe so a transient worker-introspection
        # failure degrades to the durable tiers rather than erroring the read-only probe.
        worker = getattr(harness, '_merge_worker', None) if harness is not None else None
        if worker is not None and hasattr(worker, 'snapshot'):
            try:
                snap = worker.snapshot()
                entries = snap.get('entries', [])
                entry = None
                if request_id is not None:
                    entry = next(
                        (e for e in entries if e.get('request_id') == request_id), None
                    )
                else:
                    # Most-recent by enqueued_at for branch / task_id
                    candidates = [
                        e for e in entries
                        if (branch is not None and e.get('branch') == branch)
                        or (task_id is not None and e.get('task_id') == task_id)
                    ]
                    if candidates:
                        entry = max(candidates, key=lambda e: e.get('enqueued_at', 0))
                if entry is not None:
                    eta = None
                    if merge_inflight_registry is not None:
                        with contextlib.suppress(Exception):
                            eta = merge_inflight_registry.eta_seconds(entry['branch'])
                    return {
                        'state': _map_live_state(entry['state']),
                        'request_id': entry.get('request_id'),
                        'generation': 1,
                        'position': entry.get('position'),
                        'enqueued_at': entry.get('enqueued_at'),
                        'eta_seconds': eta,
                    }
            except Exception:
                logger.warning('merge_status: snapshot() failed, falling through to durable tiers',
                               exc_info=True)

        # Tiers 2–3: durable tiers (retention ring → event store).
        # _durable_terminal_state owns the tier logic shared with merge_cancel.
        durable = _durable_terminal_state(request_id, branch, task_id)
        if durable is not None:
            coarse_state, meta = durable
            return {
                'state': coarse_state,
                'request_id': meta['request_id'],
                'generation': 1,
                'outcome': meta['outcome'],
                'finished_at': meta['finished_at'],
            }

        # Tier 4: honest unknown
        return {
            'state': 'unknown',
            'request_id': request_id,
            'generation': 1,
            'hint': _MERGE_STATUS_UNKNOWN_HINT,
        }

    # ── merge_cancel — explicit cancellation via waiter-future cancel (PRD β2 / task 1632) ──

    @mcp.tool()
    async def merge_cancel(request_id: str) -> dict[str, Any]:
        """Cancel an in-flight merge request by its request_id.

        Accepts a single *request_id* parameter (authoritative merge-request identifier
        returned by merge_request).  Returns a dict with three fields:

            cancelled (bool)  — True only when a pending waiter was successfully cancelled.
            state     (str)   — Coarse terminal state in the same vocabulary as merge_status:
                                'abandoned' | 'done' | 'conflict' | 'blocked' | 'unknown'
            reason    (str|None) — None on success; non-None string on every other path.

        Branch order (all paths return — never raises):
          1. Waiter absent from _waiters (finalized+popped, never submitted, or server
             restarted): consult durable tiers via _durable_terminal_state (shared with
             merge_status Tiers 2–3: retention ring → event_store) to distinguish
             'already-terminal' from truly 'unknown'.  Note: coalesced submissions
             (in_flight / attached paths in merge_request) do not register a separate
             waiter — callers holding a coalesced id will resolve to 'unknown' here;
             use the in-flight request_id (inflight_task_id / merge_status) to cancel.
          2. Waiter present, future already cancelled (idempotent double-cancel):
             {cancelled: False, state: 'abandoned', reason: ...}.
          3. Waiter present, future resolved-but-not-cancelled (mid-finalize window, i.e.
             worker delivered an outcome but the _waiters.pop done-callback hasn't run yet):
             {cancelled: False, state: <coarse terminal via _map_terminal_state>, reason: ...}.
             Excepted futures (abnormal) map to state='blocked'.
          4. Waiter present, future pending: cancel the future, return
             {cancelled: True, state: 'abandoned', reason: None}.

        AUTOMATIC CONSEQUENCES of cancelling the future (NOT implemented here — covered by
        α1/β1 callbacks and tested in test_merge_queue.py:4520/:4601):
          - MergeWorker._request_abandoned: drops the request without halting the queue
          - InFlightMergeRegistry: releases the branch slot via the acquire-time done-cb
          - _on_finalized: records terminal state 'abandoned' to retention/event-store

        The tool is async so that future mutation runs on the event loop (not a FastMCP
        threadpool worker — PRD Open Q4 off-loop lesson).  The lookup → cancel sequence
        contains no await, preserving loop-synchronous race-freedom (I10).
        """
        rec = _waiters.get(request_id)

        if rec is None:
            # No live waiter — finalized+popped, never submitted, or server restarted.
            # Consult durable tiers to distinguish 'already-terminal' from truly 'unknown'.
            durable = _durable_terminal_state(request_id)
            if durable is not None:
                coarse_state, _ = durable
                return {
                    'cancelled': False,
                    'state': coarse_state,
                    'reason': 'Request already finalized; cannot cancel.',
                }
            return {
                'cancelled': False,
                'state': 'unknown',
                'reason': (
                    f'No in-flight waiter for request_id {request_id!r} '
                    '(already finalized, never submitted, server restarted, or this id '
                    'was coalesced onto an in-flight request — coalesced requests share '
                    "the in-flight entry's request_id and do not register a separate "
                    'waiter; use the in-flight request_id to cancel).'
                ),
            }

        if rec.future.cancelled():
            # Idempotent double-cancel: future is already cancelled.
            return {
                'cancelled': False,
                'state': 'abandoned',
                'reason': 'Request was already cancelled.',
            }

        if rec.future.done():
            # Mid-finalize window: the future resolved (not cancelled) but the
            # call_soon-scheduled _waiters.pop done-callback hasn't run yet.
            # Defensive: excepted futures are abnormal; treat as 'blocked'.
            if rec.future.exception() is not None:
                state: str = 'blocked'
            else:
                state = _map_terminal_state(rec.future.result().status)
            return {
                'cancelled': False,
                'state': state,
                'reason': 'Request already finalized; cannot cancel.',
            }

        # Pending waiter — cancel the future.
        rec.future.cancel()
        return {'cancelled': True, 'state': 'abandoned', 'reason': None}

    return mcp
