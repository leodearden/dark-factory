"""Per-task workflow state machine: PLAN → EXECUTE → VERIFY → REVIEW → MERGE → DONE."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, cast

# Runtime import of the BORN_AT_L2_SEVERITIES constant from escalation.models.
# escalation.models is listed under TYPE_CHECKING above (:77-78) for the Escalation
# type annotation; a separate runtime import is needed here because
# _is_gating_escalation evaluates the set at call time, not just for type hints.
# This is cycle-safe: escalation/src/escalation/models.py imports only stdlib and
# nothing in it imports orchestrator.
from escalation.dedupe import submit_or_dedupe
from escalation.models import BORN_AT_L2_SEVERITIES
from pydantic import ValidationError
from shared.cli_invoke import (
    AgentFailureKind,
    AllAccountsCappedException,
    classify_agent_failure,
    invoke_with_cap_retry,
    is_timed_out_with_progress,
    is_zero_output_timeout,
    read_transcript_records,
)
from shared.config_dir import TaskConfigDir
from shared.cost_store import CostStore
from shared.prompt_artifact import PromptArtifactStore, default_artifacts_root
from shared.task_claimant import compose_claimant_run_id
from shared.task_metadata import RetryLedger, RoutingDecisionMirror, RoutingState
from shared.task_statuses import TaskStatus
from shared.transcript_archive import archive_task_transcripts

from orchestrator import chronic_flake
from orchestrator.agents.invoke import AgentResult, invoke_agent
from orchestrator.agents.roles import (
    _ESCALATION_TOOLS,
    ALL_REVIEWERS,
    ARCHITECT,
    DEBUGGER,
    IMPLEMENTER,
    JUDGE,
    MERGER,
    REVIEWER_COMPREHENSIVE,
    ROLES,
    SIMPLE_TASK,
    AgentRole,
)
from orchestrator.agents.write_set import (
    WriteSet,
    compute_write_set,
    ensure_claude_fleet_dir,
)
from orchestrator.artifacts import (
    PLAN_SCHEMA_VERSION,
    ReviewAggregation,
    TaskArtifacts,
)
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.dry_run_unblock import run_dry_run_unblock
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import (
    BranchResetError,
    GitOps,
    TrainMembership,
    WarmLaneRequeue,
    WorktreeConflictError,
    _run,
    is_wip_safety_commit,
)
from orchestrator.landed_outbox import LandedRow, MergeProvenance
from orchestrator.mcp_lifecycle import plan_tools_mcp_server, verdict_tools_mcp_server
from orchestrator.module_charter import derive_modules, sanitize_files_for_persist
from orchestrator.routing import (
    PlanShape,
    RoleDefaults,
    RouteInputs,
    _config_key,
    resolve_route,
)
from orchestrator.scheduler import (
    SetTaskStatusRejected,
    TaskAssignment,
    TerminalExitRejection,
    files_to_modules,
    normalize_lock,
)
from orchestrator.session_registry import build_session_slug
from orchestrator.task_status import (
    ACTIVE_TASK_STATUSES,
    TERMINAL_STATUSES,
    WORKFLOW_PRESERVE_STATUSES,
)
from orchestrator.unblock_types import BlockClass, classify_block_reason
from orchestrator.usage_gate import SessionBudgetExhausted as _SessionBudgetExhausted
from orchestrator.verify import (
    VerifyInfraError,
    VerifyResult,
    _is_infra_oserror,
    run_scoped_verification,
    verify_failure_is_preexisting_on_main,
)
from orchestrator.verify_categories import (
    INFRA_TRANSIENT_CATEGORIES,
    PREEXISTING_BREAK_SKIP_CATEGORIES,
    FailureCategory,
)
from orchestrator.verify_checkpoint import green_checkpoint_at_tip
from orchestrator.workflow_types import (  # noqa: F401  re-export shim
    BlockDisposition,
    CancellationScope,
    IllegalTransition,
    OnTerminalEntry,
    RequeueKind,
    StewardBudgetExhausted,
    StewardInterrupted,
    StewardOutcome,
    StewardReescalatedL1,
    StewardResolved,
    StewardTerminalDecision,
    TerminalReport,
    WorkflowCancelled,
    WorkflowOutcome,
    WorkflowState,
    WorkflowStateMachine,
    classify_failure,
    outcome_allows_status,
)

# Orchestrator package directory — used to resolve ``uv run --project`` for
# the plan-tools stdio MCP server.
_ORCH_PROJECT_DIR = Path(__file__).resolve().parents[2]

# Role gates for per-invocation MCP/sandbox wiring in _invoke() are read
# directly off the AgentRole object (role.mcp_families / role.sandboxed) —
# see orchestrator/agents/roles.py's AgentRole.__post_init__ and per-role
# mcp_families/sandboxed declarations (W9-η, reify esc-4943-54: simple_task's
# allowed_tools referenced fused-memory/escalation/plan-tools tools, but the
# name-string gate tuples formerly here forgot to list it, so its sessions
# were told to call tools that did not exist and the Lever-C fast-path
# silently always fell through to the architect). The former
# _MCP_CONFIG_ROLES / _PLAN_TOOLS_ROLES name-string tuples are retired in
# favor of this role-object read — a role whose allowed_tools reference a
# tool family now MUST declare the matching mcp_families entry or
# AgentRole's import-time assertion raises, so the two can no longer drift.


def _meta_root_for_worktree(worktree: Path) -> Path:
    """Derive the `.task-meta` root for *worktree* — the single DRY seam used
    by both the orchestrator-side ``TaskArtifacts`` construction (``_setup``)
    and the agent-side plan-tools MCP injection (``_inject_plan_tools_mcp``),
    so both sides compute the IDENTICAL meta_root for a given worktree
    (worktree-lane-lifecycle PRD task ε1).

    Thin wrapper around ``TaskArtifacts.meta_root_for`` — the single owner of
    the `.task-meta` path shape — so callers here never hand-join ``.task``
    or ``.task-meta`` themselves.
    """
    return TaskArtifacts.meta_root_for(worktree.parent, worktree.name)


def _inject_plan_tools_mcp(mcp_config: dict | None, cwd: Path) -> dict:
    """Inject the plan-tools stdio MCP server entry into *mcp_config*.

    Creates a minimal ``{'mcpServers': {}}`` skeleton when *mcp_config* is
    ``None``; otherwise preserves all pre-existing server entries and adds
    (or replaces) the ``plan-tools`` key.

    The plan-tools server is launched via the direct-interpreter no-uv hot path
    (task 1776): ``sys.executable -m orchestrator.mcp.plan_tools --worktree <wt>``.
    This eliminates ``uv``-internal lock contention that caused 0-turn
    ``error_empty_output`` failures when many agents launch concurrently under
    load (reify esc-4415-240, esc-4437-123).  The orchestrator process runs
    inside the already-synced orchestrator venv, so ``sys.executable`` is a
    guaranteed-present interpreter with the ``orchestrator`` package importable.

    Defense-in-depth composition:
    - Task 1771: ``apply_mcp_startup_env`` injects ``MCP_TIMEOUT=30000`` so any
      residual slow start is bounded rather than hanging to the 1200s wall.
    - Task 1775: the ``uv --no-sync --frozen`` fallback is retained in
      ``plan_tools_mcp_server`` for callers that cannot supply an interpreter.
    - Task 1776 (this): ``python_executable=sys.executable`` eliminates ``uv``
      from the hot path entirely, collapsing the proc tree to ``claude → python3``.
    - Task 2258 (W11-ε1): ``meta_root=_meta_root_for_worktree(cwd)`` passes the
      sibling `.task-meta` root through as ``--meta-root``, so the agent-side
      plan-tools server writes plan.json to the same relocated artifacts root
      the orchestrator's own ``TaskArtifacts`` instance reads from.
    """
    if not mcp_config:
        mcp_config = {'mcpServers': {}}
    mcp_config.setdefault('mcpServers', {})['plan-tools'] = plan_tools_mcp_server(
        _ORCH_PROJECT_DIR, cwd, python_executable=sys.executable,
        meta_root=_meta_root_for_worktree(cwd),
    )
    return mcp_config


def _inject_verdict_tools_mcp(mcp_config: dict | None, cwd: Path, role: AgentRole) -> dict:
    """Inject the verdict-tools stdio MCP server entry into *mcp_config*.

    Creates a minimal ``{'mcpServers': {}}`` skeleton when *mcp_config* is
    ``None`` — the same None-skeleton path ``_inject_plan_tools_mcp`` uses.
    This is what lets the reviewer (which declares NO ``'orchestrator'``
    family, so its ``mcp_config`` would otherwise stay ``None``) still
    acquire the verdict-tools server: the family check that gates this
    injector (PRD task β's spawn-site gate) is independent of the
    ``'orchestrator'`` gate, so a role with only ``'verdict_tools'`` in
    ``mcp_families`` gets a config built from scratch here.

    ``role.name`` is passed as ``--verdict-role`` — the authoritative
    selector for both the single tool the verdict-tools server registers
    (judge/merger/reviewer-name, see ``verdict_tools.create_server``) and the
    ``verdicts/<role>.json`` filename it writes (α's I-AUTHORITATIVE-PATH
    invariant: never an agent-supplied field). This mirrors the existing
    review-key convention (``write_review(role.name)`` elsewhere in this
    file), so ``verdicts/reviewer_comprehensive.json`` mirrors
    ``reviews/reviewer_comprehensive.json``.

    Modeled on ``_inject_plan_tools_mcp``: same direct-interpreter no-uv hot
    path (``python_executable=sys.executable``) and the same
    ``meta_root=_meta_root_for_worktree(cwd)`` passthrough so the agent-side
    verdict-tools server targets the identical relocated `.task-meta` root
    the orchestrator's own ``TaskArtifacts`` instance reads from.

    Deliberately passes no ``--session-id`` to ``verdict_tools_mcp_server``,
    even though the active session id is available here at the ``_invoke``
    spawn site — see that function's docstring (task 2482 design_decisions
    #4) for why this is an intentional v1 scope boundary rather than an
    oversight.
    """
    if not mcp_config:
        mcp_config = {'mcpServers': {}}
    mcp_config.setdefault('mcpServers', {})['verdict-tools'] = verdict_tools_mcp_server(
        _ORCH_PROJECT_DIR, cwd, role.name, python_executable=sys.executable,
        meta_root=_meta_root_for_worktree(cwd),
    )
    return mcp_config


# Roles whose allowed_tools include at least one 'mcp__escalation__escalate*' tool.
# 'steward' and 'deep_reviewer' are excluded: they run in their own dispatchers
# (TaskSteward and ReviewCheckpoint respectively), not through TaskWorkflow._invoke.
# All other roles are included or excluded based on their actual allowed_tools entries in ROLES.
_ESCALATION_CAPABLE_ROLES: frozenset[str] = frozenset(
    name for name, role in ROLES.items()
    if any(t in _ESCALATION_TOOLS for t in (role.allowed_tools or []))
    and name not in {'steward', 'deep_reviewer'}
)

# Stable diagnostic tokens guaranteed to appear in the orphan-halt warning emitted
# by ``_warn_orphan_halt_no_queue``.  Referenced by tests so wording changes are
# caught at the constant definition rather than silently diverging across assertions.
_ORPHAN_HALT_NO_QUEUE_TOKENS: tuple[str, ...] = ('orphan halt', 'unhalt_merge_queue')


def _is_gating_escalation(e: Escalation) -> bool:
    """Return True if *e* should gate workflow progress (PRD C7 / decisions D4, D8).

    Gating policy — an escalation gates when ANY disjunct fires:
    1. Plain blocking L0: ``severity == 'blocking' and level == 0``
       (fresh agent-filed blocker awaiting steward triage).
    2. Born-at-L2 severity: ``severity in BORN_AT_L2_SEVERITIES``
       i.e. 'critical' or 'urgent' — these bypass the auto-watcher and route
       straight to a human regardless of level.
    3. Any level ≥ 2: ``level >= 2``
       (escalation-watcher promoted to L2; a human must decide before the
       workflow proceeds — applies to any severity, including 'info').

    Deliberately NOT gating:
    - Plain blocking L1 (``severity == 'blocking' and level == 1``): this is a
      steward hand-off from a *prior* run that is still awaiting human triage.
      Sinking the current run on a stale L1 caused the run-2 false-blocked
      outcome (esc-2911-22).  The B3 constraint preserves this property; note
      that a *prior-run* critical/urgent (``severity in BORN_AT_L2_SEVERITIES``)
      DOES gate the current run — that is the intended stop-the-line semantics
      for high-severity escalations (D4 accepted consequence).
    """
    return (
        (e.severity == 'blocking' and e.level == 0)
        or e.severity in BORN_AT_L2_SEVERITIES
        or e.level >= 2
    )


def _format_reescalation_detail(escalations: list[Escalation]) -> str:
    """Render the payload carried by ``_StewardReescalated`` into a rich
    ``_mark_blocked(detail=...)`` string (task 2553).

    Without this, ``run()``'s ``except _StewardReescalated`` clause discarded
    the exception's ``escalations`` payload and called ``_mark_blocked``
    without ``detail=``, so ``_mark_blocked``'s ``detail = detail or reason``
    fallback made ``TerminalReport.detail`` an exact copy of
    ``TerminalReport.reason`` ('Steward re-escalated to human') — block-time
    investigators started from zero information (harness ``block_detail``).

    Renders, per escalation: its id/severity/level/category header line,
    then — only when non-``None`` — the run phase (``workflow_state``) and
    resolution turn count (``resolution_turns``), then the full ``summary``
    and ``detail`` text. Multiple escalations (a burst may hold more than
    one) are joined with a blank line between them. Returns ``''`` for an
    empty list — defensive; the two raise sites in ``_wait_for_resolution``
    (born-at-L2 / level>=2, and steward-gave-up level-1) both guarantee a
    non-empty list, and ``''`` preserves ``_mark_blocked``'s
    ``detail or reason`` fallback rather than emitting an empty-but-present
    detail.
    """
    if not escalations:
        return ''

    blocks = []
    for e in escalations:
        lines = [
            f'Escalation {e.id} (severity={e.severity}, level={e.level}, '
            f'category={e.category})',
        ]
        if e.workflow_state is not None:
            lines.append(f'phase={e.workflow_state}')
        if e.resolution_turns is not None:
            lines.append(f'turns={e.resolution_turns}')
        lines.append(f'Summary: {e.summary}')
        lines.append(f'Detail: {e.detail}')
        blocks.append('\n'.join(lines))

    header = f'Steward re-escalated; {len(escalations)} pending escalation(s):'
    return header + '\n\n' + '\n\n'.join(blocks)


class _StewardReescalated(Exception):
    """Raised when the steward re-escalates to level-1 (consumed by the auto-watcher, which may promote to L2 for a human)."""

    def __init__(self, escalations):
        self.escalations = escalations

if TYPE_CHECKING:
    from escalation.models import Escalation, TrainState
    from escalation.queue import EscalationQueue

    from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome, SoloVerifyResult
    from orchestrator.scheduler import SchedulerFacade
    from orchestrator.usage_gate import UsageGate


# ---------------------------------------------------------------------------
# Structural protocols — allow test doubles without inheriting concrete classes
# ---------------------------------------------------------------------------


class _McpLike(Protocol):
    @property
    def url(self) -> str: ...
    def mcp_config_json(self, escalation_url: str | None = None) -> dict: ...


class _BriefingLike(Protocol):
    async def build_architect_prompt(
        self, task: dict, worktree: Path | None = ..., context: str | None = ...,
        *, include_prior_proposals: bool = ...,
    ) -> str: ...
    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree: Path | None = ...,
    ) -> str: ...
    async def build_implementer_prompt(
        self, plan: dict, iteration_log: list, context: str | None = ...,
        rebase_notice: dict | None = ..., task_id: str | None = ...,
        wip_notice: list[dict] | None = ...,
    ) -> str: ...
    async def build_amender_prompt(
        self, plan: dict, iteration_log: list[dict],
        suggestions: list[dict], locked_modules: list[str],
        context: str | None = ..., task_id: str | None = ...,
    ) -> str: ...
    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = ...,
        task_id: str | None = ...,
    ) -> str: ...
    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = ...,
        *, amendment_suggestions: list[dict] | None = ...,
    ) -> str: ...
    async def build_merger_prompt(
        self, conflicts: str, task_intent: str, context: str | None = ...
    ) -> str: ...
    async def build_revalidation_prompt(
        self, task: dict, existing_plan: dict,
        changed_files: list[str], worktree: Path | None = ...,
        context: str | None = ...,
    ) -> str: ...
    async def build_plan_completion_prompt(
        self, task: dict, partial_plan: dict,
        worktree: Path | None = ..., context: str | None = ...,
    ) -> str: ...
    async def build_plan_tightening_prompt(
        self, task: dict, plan: dict, not_touched: list[str],
        worktree: Path | None = ..., context: str | None = ...,
    ) -> str: ...
    async def build_simple_task_prompt(
        self, task: dict, worktree: Path | None = ...,
        context: str | None = ...,
    ) -> str: ...
    async def build_completion_judge_prompt(
        self, plan: dict, iteration_log: list[dict], diff: str,
        task_id: str | None = ..., context: str | None = ...,
    ) -> str: ...

logger = logging.getLogger(__name__)

# Reason string used when the fresh-invocation zero-output circuit breaker trips.
# Mirrors the _inherited_break_info pattern: set by _execute_iterations, consumed
# at the BLOCKED route in _execute_verify_review_loop (task 1739).
ZERO_OUTPUT_HANG_REASON = 'infra: zero-output CLI hang (consecutive fresh-invocation timeouts)'

# Reason string used when the progress-resume churn circuit breaker trips:
# config.max_progress_resume_iterations ceiling-kill+resume cycles of a
# productive (transcript_turns>0) session accumulated without the task ever
# converging.  Distinct from ZERO_OUTPUT_HANG_REASON (no progress at all) and
# from the generic 'Execution iterations exhausted' path (progress-resumes
# are excluded from that cap — see _execute_iterations).  Mirrors the
# _zero_output_hang_info stash idiom: set by _execute_iterations, consumed at
# the BLOCKED route in _execute_verify_review_loop (task 2360, reify-4827).
PROGRESS_RESUME_CHURN_REASON = (
    'infra: progress-resume churn (repeated ceiling-kill+resume of a '
    'productive session without convergence)'
)


# Matches the wrapper string ``_run_cmd`` injects when its own asyncio.wait_for
# fires.  When this is the only cause hint there is no actionable signal for
# the debugger; the verify-retry loop short-circuits to BLOCKED instead of
# burning ``max_verify_attempts × verify_command_timeout_secs``.  After the
# streamed-stdout fix in ``_run_cmd`` (Change 2) a real cause hint should
# surface on attempt 1 for any genuine in-test hang.
_OPAQUE_TIMEOUT_CAUSE_RE = re.compile(r'^Command timed out after \d+(\.\d+)?s:')

def _normalize_cause_hint(hint: str | None) -> str:
    """Thin delegator to :meth:`RetryLedger.normalize_cause_hint`.

    Kept as a module-level function (rather than inlining the call at each
    site) because ``orchestrator.verify`` and several tests
    (``test_workflow_signature_loop_guard.py``, ``test_merge_queue_auto_heal.py``)
    import this name directly from ``orchestrator.workflow``. The ledger is
    the single signature-keying authority (shared/src/shared/task_metadata.py);
    this wrapper preserves every existing importer's behaviour byte-for-byte.
    """
    return RetryLedger.normalize_cause_hint(hint)


def _compute_merge_outcome_signature(
    category: str | None,
    cause_hint: str | None,
    fallback_reason: str = '',
) -> str:
    """Thin delegator to :meth:`RetryLedger.compute_merge_outcome_signature`.

    Keys on (category, normalised cause_hint) when either field is set; falls
    back to sha256(normalised_reason) when both are empty — same logic as
    _merge_outcome_signature() but takes the values directly rather than reading
    the TaskWorkflow._last_merge_failure_* instance fields.

    Those instance fields are never set on the MAIN_HEALTH_RED_REASON_PREFIX
    fast-path in _submit_to_merge_queue (the method returns into
    _auto_heal_main_health before the generic block path that sets them), so
    calling self._merge_outcome_signature() there would always hash the empty
    basis — a single constant key for every main-health break.  This helper
    lets _auto_heal_main_health pass the outcome fields it already holds.

    _merge_outcome_signature() delegates here so the hash algorithm stays in
    one place; that method's behaviour and the #1688 thrash tests are unchanged.
    Kept as a module-level function (rather than inlining the call at each
    site) because ``test_merge_queue_auto_heal.py`` imports this name directly
    from ``orchestrator.workflow``.
    """
    return RetryLedger.compute_merge_outcome_signature(category, cause_hint, fallback_reason)


def compute_preexisting_main_break_fingerprint(
    category: str,
    cause_hint: str,
    probe_sha: str,
) -> str:
    """Shared fingerprint for preexisting-main-break escalations.

    Composes the same triple used by _verify_debugfix_loop inline so that
    task-verify escalations and merge-queue escalations for the SAME broken
    main fold into ONE parent via submit_or_dedupe.

    Returns '' on any exception (fail-safe — an empty fingerprint triggers
    the raw-submit path, not a crash).
    """
    try:
        from escalation.dedupe import compute_content_fingerprint
        return compute_content_fingerprint(
            'preexisting_main_break',
            category or '',
            [],
            description=_normalize_cause_hint(cause_hint) + '|' + probe_sha,
        )
    except Exception:
        return ''


def compute_failing_test_set_fingerprint(failing_test_ids: list[str]) -> str:
    """Dedup key for the offline-deep lane's confirmed-red fix-task spawn (β3).

    Keyed on the sorted CONFIRMED failing-test ID SET — NEVER ``main_sha``
    (DB3/C3). The model helper above (``compute_preexisting_main_break_fingerprint``)
    keys on a probe/main_sha, which advances every merge; reusing that shape
    here would spawn a fresh fix task + escalation on EVERY offline-lane
    advance while red — the exact flood PRD §7/§10/§11 forbids. Keying on the
    failing-test SET instead means one open fix task absorbs the same red
    across advances (via an appended suspect range), while a genuinely
    different failing set gets its own task.

    ``compute_content_fingerprint`` already sorts ``affected_ids`` and ignores
    ``description``, so the result is order-independent over the input list.

    Returns '' on any exception (fail-safe — an empty fingerprint degrades to
    the log-only path, never a crash).
    """
    try:
        from escalation.dedupe import compute_content_fingerprint
        return compute_content_fingerprint(
            'offline_lane_red', '', sorted(failing_test_ids),
        )
    except Exception:
        return ''


def build_offline_lane_fix_task_arguments(
    failing_test_ids: list[str],
    suspect_range: str,
    fingerprint: str,
    project_root: str | Path,
    head: str,
    priority: str = 'high',
) -> dict:
    """Build the ``submit_task`` argument block for an offline-lane fix task (β3).

    Modeled on :meth:`TaskWorkflow._spawn_main_health_fix_task`'s argument
    block (title/description/priority/project_root/metadata), but routed
    THROUGH the standard TDD→PR→merge gate rather than the high/red-main
    lane: ``status='pending'`` and ``metadata.merge_lane='normal'`` (never
    the B3 red-main fix-forward path — that class hard-aborts to a human;
    D1/§10 "the fix goes through the gate").

    ``metadata.failing_tests`` is stored sorted (matching the fingerprint's
    own sort) and ``metadata.suspect_ranges`` is seeded as a single-element
    list so the caller can append further ranges as the same fingerprint
    recurs across advances (C3) without restructuring the field.

    *priority* is the filed fix task's priority (task 2789, D3). Defaults to
    ``'high'`` so the legacy numeric/infra call sites in
    :meth:`OfflineLaneWorker._file_new_fix_task` stay byte-identical; a
    generic per-project command supplies its own ``LaneCommand.fix_task_priority``.
    """
    sorted_ids = sorted(failing_test_ids)
    test_list = ', '.join(sorted_ids)
    title = f'offline-lane: fix {len(sorted_ids)} failing test(s) ({test_list})'
    description = (
        f'The offline-deep lane confirmed a red run with failing tests: '
        f'{test_list}.\n\nSuspect commit range: {suspect_range}\n'
        f'Head at detection: {head}\n\n'
        f'This task was auto-filed by the offline-deep lane worker (β3). It '
        f'merges via the standard gate, never the red-main fix-forward path.'
    )
    return {
        'title': title,
        'description': description,
        'status': 'pending',
        'priority': priority,
        'project_root': str(project_root),
        'metadata': {
            'merge_lane': 'normal',
            'spawn_context': 'offline_lane_red',
            'offline_lane_fingerprint': fingerprint,
            'failing_tests': sorted_ids,
            'suspect_ranges': [suspect_range],
        },
    }


def _line_ranges_stackable(
    ranges_a: dict[str, list[tuple[int, int]]],
    ranges_b: dict[str, list[tuple[int, int]]],
) -> bool:
    """Return True iff tasks A and B are line-level stackable.

    Two tasks are stackable iff no file they both touch has overlapping changed
    line ranges relative to BASE (main).  Crate/file-disjointness is NOT
    required for stackability — same file, different lines is fine (PRD §A.2).

    Uses closed-interval intersection: range (s1,e1) intersects (s2,e2)
    iff s1 <= e2 and s2 <= e1.
    """
    shared_files = set(ranges_a) & set(ranges_b)
    for fname in shared_files:
        for s1, e1 in ranges_a[fname]:
            for s2, e2 in ranges_b[fname]:
                if s1 <= e2 and s2 <= e1:
                    return False
    return True


def _select_train_members(
    anchor_id: str,
    candidate_ids: list[str],
    ranges_by_id: dict[str, dict[str, list[tuple[int, int]]]],
    max_members: int,
) -> list[str]:
    """Select a mutually-stackable subset of candidates capped at max_members.

    Greedy selection: start with the anchor, then iterate candidates in
    deterministic (id-sorted) order, adding each that is _line_ranges_stackable
    against ALL already-selected members.  Stops when len(selected) == max_members.

    Returns [] when the result has fewer than 2 members — the sentinel for
    "no viable train" (a single-member train is meaningless).

    The anchor is always first (order-0) in the returned list.
    """
    selected: list[str] = [anchor_id]

    for candidate_id in sorted(candidate_ids):
        if len(selected) >= max_members:
            break
        candidate_ranges = ranges_by_id.get(candidate_id, {})
        # Must be stackable with every already-selected member (mutual stackability).
        stackable_with_all = all(
            _line_ranges_stackable(candidate_ranges, ranges_by_id.get(sel_id, {}))
            for sel_id in selected
        )
        if stackable_with_all:
            selected.append(candidate_id)

    # A single-member "train" is a no-op — return the empty sentinel.
    if len(selected) < 2:
        return []
    return selected


def classify_rebase_cohort(
    distance_commits: int,
    is_first_rebase: bool,
    threshold: int,
) -> str:
    """Label a rebase event into one of four cohorts.

    Args:
        distance_commits: Commit count in old_base..new_base (from
            ``get_rebase_distance``).  -1 means the distance could not be
            measured.
        is_first_rebase: True when this is the first rebase of the current
            per-dispatch WorkflowMetrics instance (i.e.
            ``metrics.inter_iteration_rebases == 0`` captured BEFORE the
            counter is incremented).
        threshold: ``config.rebase_reseed_distance_threshold`` — the labelling
            boundary between 'continuous' and large-jump cohorts.

    Returns:
        'unknown'      — distance < 0 (measurement failed; fail-safe)
        'continuous'   — 0 <= distance < threshold (normal orchestrator cadence)
        'post-unblock' — distance >= threshold AND is_first_rebase (first
                         rebase of a fresh instance; likely long-idle gap /
                         post-unblock resume)
        'big-jump'     — distance >= threshold AND NOT is_first_rebase
                         (large mid-run jump; hypothesis-refuting outlier)
    """
    if distance_commits < 0:
        return 'unknown'
    if distance_commits < threshold:
        return 'continuous'
    return 'post-unblock' if is_first_rebase else 'big-jump'


# Context tolerance (lines) applied on each side of an amendment's new-side
# changed ranges when partitioning post-amendment review suggestions.  Absorbs
# reviewer line-number drift so genuine delta-adjacent findings are not routed
# away, while fresh nits far from the amendment are still filtered (task 2750).
_AMENDMENT_DELTA_CONTEXT_LINES = 3


# Dedicated role for the task-2523 resettled-suggestion adjudication.
#
# It deliberately does NOT reuse REVIEWER_COMPREHENSIVE.  That role's contract
# (roles.py's _REVIEWER_CONTRACT_TEMPLATE) MANDATES a `submit_review_verdict`
# tool call and explicitly forbids JSON/prose output, and the role injects the
# `verdict_tools` MCP family — so an agent following it would emit a review
# verdict instead of the requested `{decisions:[...]}` StructuredOutput,
# leaving `result.structured_output` empty and the whole suppression pass
# silently inert (fail-safe → all-emit) while still burning a reviewer-tier
# invocation.  This adjudicator is instead a pure structured classifier: no
# verdict tooling (empty `mcp_families`, so `_invoke` wires NO MCP config) and
# no file/bash tools (empty `allowed_tools`, so `_invoke` passes
# `--allowed-tools` as None and the synthetic `StructuredOutput` schema tool is
# not gated out), driven by a system prompt that asks ONLY for the decisions
# schema.  It borrows REVIEWER_COMPREHENSIVE's model/budget/max-turns so the
# routing cost profile is unchanged.  Defined here rather than in
# agents/roles.py because this amendment is scoped to workflow.py; it is a
# local role object, never registered in the ROLES registry (an unknown role
# name resolves safely to its `role_default` layer in routing.resolve_route).
_RESETTLED_ADJUDICATOR_SYSTEM_PROMPT = """\
You are a precise CLASSIFICATION JUDGE, not a code reviewer. You are given two
lists of review suggestions — a set raised in a PRIOR amendment round of a task,
and the CURRENT round's suggestions — and you decide, per current suggestion,
whether it merely re-flags a concern already resolved earlier.

Return your answer ONLY as the structured `{"decisions": [...]}` payload defined
by the output schema — one entry per current suggestion index. Do NOT call
`submit_review_verdict` or any review/verdict tool, do NOT write prose findings,
and do NOT perform a fresh code review. Your entire job is the comparison and
the per-index decision; the caller reads nothing but the structured output.
"""

_RESETTLED_ADJUDICATOR = AgentRole(
    name='resettled_adjudicator',
    system_prompt=_RESETTLED_ADJUDICATOR_SYSTEM_PROMPT,
    allowed_tools=[],
    disallowed_tools=[],
    default_model=REVIEWER_COMPREHENSIVE.default_model,
    default_budget=REVIEWER_COMPREHENSIVE.default_budget,
    default_max_turns=REVIEWER_COMPREHENSIVE.default_max_turns,
    mcp_families=frozenset(),
)


@dataclass(frozen=True)
class AmendmentReviewContext:
    """Consume-once loop state scoping ONE post-amendment review to its delta.

    Captured in ``_execute_verify_review_loop`` immediately before an
    amendment ``_amend`` call and consumed by the single ``_review`` that
    immediately follows it, then reset to ``None`` so a later
    blocking-replan/re-execute cycle (which produces a materially different
    diff) is never scoped against this stale pre-amendment HEAD (task 2750).

    Fields:
        pre_amendment_head: the worktree HEAD SHA *before* the amendment, so
            ``{pre_amendment_head}..HEAD`` is exactly the amendment delta.
        amended_suggestions: the in-scope suggestions the amendment was asked
            to address — threaded into the advisory reviewer prompt.
    """

    pre_amendment_head: str
    amended_suggestions: list[dict]


@dataclass(frozen=True)
class _LedgerVerdict:
    """Result of a pure anti-thrash evaluator (task 2172 / W3-ε).

    Bundles the updated :class:`~shared.task_metadata.RetryLedger` with the
    escalate decision so the async guards can persist ``ledger`` and act on
    ``escalate``/``trigger`` without repeating the counter arithmetic.

    ``trigger`` is a short machine-readable label describing which condition
    caused ``escalate=True`` (e.g. ``'same-SHA counter'``, ``'total
    counter'``); it is the empty string when ``escalate`` is False.
    """

    ledger: RetryLedger
    escalate: bool
    trigger: str = ''


def _build_retry_ledger(metadata: dict) -> RetryLedger:
    """Safely reconstruct a :class:`RetryLedger` from ``metadata['retry_ledger']``.

    Tolerates any shape of corruption a hand-edited or legacy metadata blob
    might carry: a missing/None key, a non-dict value (a stray string, list,
    or scalar — ``RetryLedger(**raw)`` would otherwise raise ``TypeError``
    rather than the ``ValidationError`` callers expect), or a dict whose
    fields fail pydantic validation (e.g. a non-numeric counter). Any of
    these reset to a fresh all-zero ledger instead of crashing the calling
    guard — mirrors the old per-field ``int(...)`` parsing's tolerance for a
    mistyped value, now centralised for all three anti-thrash guards.
    """
    raw = metadata.get('retry_ledger')
    if not isinstance(raw, dict):
        return RetryLedger()
    try:
        return RetryLedger(**raw)
    except (ValidationError, TypeError):
        return RetryLedger()


def _evaluate_no_plan(ledger: RetryLedger, current_main_sha: str) -> _LedgerVerdict:
    """Pure decision core for the no-plan-failure anti-thrash guard.

    Mirrors the counter arithmetic formerly inlined in
    ``TaskWorkflow._handle_no_plan_failure``: the same main SHA increments
    ``consecutive_no_plan_failures``; a different (or empty) SHA resets it to
    1. ``total_no_plan_failures`` never resets — it backstops the SHA-keyed
    counter when main keeps moving and the per-SHA counter never reaches 2
    (the bug behind 16 successive Opus calls on task 917).

    Escalates when ``consecutive >= 2`` (same-SHA thrash) OR ``total >= 3``
    (persistent no-plan failures across changing SHAs); when both fire
    simultaneously the same-SHA trigger takes precedence in the label.
    """
    last_sha = ledger.last_no_plan_main_sha or ''
    counter = ledger.consecutive_no_plan_failures
    total = ledger.total_no_plan_failures

    if not current_main_sha or last_sha != current_main_sha:
        counter = 1
    else:
        counter += 1
    total += 1

    escalate = counter >= 2 or total >= 3
    trigger = ''
    if escalate:
        trigger = 'same-SHA counter' if counter >= 2 else 'total counter'

    new_ledger = ledger.model_copy(update={
        'last_no_plan_main_sha': current_main_sha,
        'consecutive_no_plan_failures': counter,
        'total_no_plan_failures': total,
    })
    return _LedgerVerdict(ledger=new_ledger, escalate=escalate, trigger=trigger)


def _evaluate_infra_resume(
    ledger: RetryLedger,
    current_iter_count: int,
    recent_category: str | None,
    threshold: int,
) -> _LedgerVerdict:
    """Pure decision core for the infra-resume-thrash anti-thrash guard.

    Mirrors the counter arithmetic formerly inlined in
    ``TaskWorkflow._check_infra_resume_thrash``. When the most recently
    resolved L0 was classified ``'infra_issue'``: no iteration-log growth
    since the previous resume increments the counter (thrash observed);
    growth resets it to 1 (a steward fix-commit is forward progress). Any
    other category — including ``None``, meaning no classifiable resolved
    L0 — resets the counter to 0, since the thrash signal does not apply.
    ``last_infra_resume_iteration_count`` is always refreshed to
    ``current_iter_count`` regardless of category.

    Escalates when the (possibly just-incremented) counter reaches
    ``threshold``.
    """
    counter = ledger.consecutive_infra_resume_failures

    if recent_category == 'infra_issue':
        last_iter_count = ledger.last_infra_resume_iteration_count
        if current_iter_count > last_iter_count:
            counter = 1
        else:
            counter += 1
    else:
        counter = 0

    escalate = counter >= threshold
    trigger = 'infra-resume thrash' if escalate else ''

    new_ledger = ledger.model_copy(update={
        'consecutive_infra_resume_failures': counter,
        'last_infra_resume_iteration_count': current_iter_count,
    })
    return _LedgerVerdict(ledger=new_ledger, escalate=escalate, trigger=trigger)


def _evaluate_merge_thrash(
    ledger: RetryLedger,
    prev_signature: str | None,
    current_signature: str,
    threshold: int,
) -> _LedgerVerdict:
    """Pure decision core for the merge-outcome-thrash anti-thrash guard.

    Mirrors the counter arithmetic formerly inlined in
    ``TaskWorkflow._check_merge_outcome_thrash``: a merge-outcome signature
    matching the previous attempt increments ``consecutive_merge_thrash``; a
    differing signature (or no previous signature) resets it to 1 — the
    steward made progress on something different, and we just observed one
    occurrence of it. ``last_merge_outcome_signature`` is always refreshed to
    ``current_signature``.

    Escalates when the (possibly just-incremented) counter reaches
    ``threshold``.
    """
    counter = ledger.consecutive_merge_thrash

    if prev_signature is not None and prev_signature == current_signature:
        counter += 1
    else:
        counter = 1

    escalate = counter >= threshold
    trigger = 'merge-outcome thrash' if escalate else ''

    new_ledger = ledger.model_copy(update={
        'consecutive_merge_thrash': counter,
        'last_merge_outcome_signature': current_signature,
    })
    return _LedgerVerdict(ledger=new_ledger, escalate=escalate, trigger=trigger)


@dataclass
class WorkflowMetrics:
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    agent_invocations: int = 0
    execute_iterations: int = 0
    # Cumulative count of progress-timeout+resume pairs across the WHOLE task
    # run (never reset mid-run — mirrors execute_iterations' own lifetime).
    # Lives on metrics rather than as a local in _execute_iterations so the
    # max_execute_iterations cap exclusion survives re-entry into
    # _execute_iterations after a review cycle (_replan) or amendment round
    # (_amend) — see _execute_verify_review_loop's `continue` statements
    # (task 2360, reify comprehensive review finding #1).
    progress_resume_total: int = 0
    verify_attempts: int = 0
    review_cycles: int = 0
    amendment_rounds: int = 0
    pre_merge_rebase_attempts: int = 0
    pre_merge_rebase_ok: int = 0
    advance_main_retries: int = 0
    inter_iteration_rebases: int = 0
    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_create_tokens: int = 0
    # Completion judge metrics (ζ). judge_cost_usd is a subset of total_cost_usd,
    # not disjoint — existing budget guards/cost reports using total_cost_usd
    # continue to work unchanged.
    judge_invocations: int = 0
    judge_cost_usd: float = 0.0
    judge_early_exits: int = 0


class _PriorImplStatus(NamedTuple):
    """Result of :meth:`TaskWorkflow._has_prior_implementation`.

    Bundles the three pieces of information that the merge-check call-site
    needs so it can avoid redundant artifact reads.
    """

    has_work: bool
    """True iff the worktree has implementation commits beyond the base."""

    entries: list[dict]
    """Full parsed iteration-log entries (may be empty)."""

    base_commit: str | None
    """SHA read from metadata.json, or None if the file is absent."""


def _iteration_entry_is_work(entry: dict) -> bool:
    """Classify a single iterations.jsonl entry as genuine prior-implementation
    work (task 2372, Layer A).

    Excludes ONLY the narrow zero-work signature: an ``'implementer'`` entry
    that is not an amendment pass and recorded no completed steps — the exact
    shape behind the task 2125/2315/2340 false-DONE recurrences. Everything
    else that represents real agent output still counts:

    - ``'debugger'`` entries always count, even though they hard-code
      ``steps_completed: []`` (workflow.py ~5222) — a debug pass is real work
      regardless of plan-step bookkeeping.
    - Amendment ``'implementer'`` entries (``source == 'amendment'``) always
      count — they omit ``steps_completed`` entirely (workflow.py ~5443).
    - ``'judge'`` ``early_exit`` entries count only when ``substantive_work``
      is True (workflow.py ~4559) — a judge can legitimately declare a task
      complete with zero plan-steps marked done.

    An entry that explicitly recorded no durable commit (``committed is False``,
    task 2759) is not prior-implementation work regardless of agent type — the
    round ended before HEAD advanced (implementer died mid-background-wait,
    amendment left uncommitted, …). Discriminates on the explicit ``False``
    flag only: legacy entries lacking the key (``entry.get('committed') is
    None``) fall through to today's classification unchanged.
    """
    if entry.get('committed') is False:
        return False
    agent = entry.get('agent')
    if agent == 'debugger':
        return True
    if agent == 'implementer':
        return entry.get('source') == 'amendment' or bool(entry.get('steps_completed'))
    if agent == 'judge':
        return entry.get('event') == 'early_exit' and bool(entry.get('substantive_work'))
    return False


def _routing_inputs_digest(
    role: AgentRole,
    task_id: str,
    plan: dict,
    modules: list[str],
    routing_tier: int,
) -> str:
    """Stable sha256[:16] digest of the salient routing-resolution inputs (PRD γ).

    Pure and deterministic: the same (role, task, plan shape, modules,
    routing_tier) always yields the same digest, so telemetry consumers can
    compare/dedupe resolutions without needing the full payload.  Digests the
    plan's *step count* (not its full contents) and the modules list — cheap,
    stable proxies for "what was being resolved" that avoid hashing large or
    volatile plan bodies.
    """
    payload = {
        'role': role.name,
        'task_id': str(task_id),
        'plan_step_count': len(plan.get('steps', [])),
        'module_count': len(modules),
        'modules': sorted(modules),
        'routing_tier': routing_tier,
    }
    basis = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(basis).hexdigest()[:16]


def _trailing_24h_window() -> tuple[str, str]:
    """Return ``(start_iso, end_iso)`` for the trailing-24h window ending now.

    Mirrors the ``cutoff_24h_iso, now_iso`` pattern used by
    ``harness.py``'s ``_enforce_cost_ceilings``/``digest.py``'s cost-stats
    helper — kept as its own function here (rather than inlined) so
    ``_invoke``'s ceiling-spend lookup reads as one call.
    """
    now = datetime.now(UTC)
    return (now - timedelta(hours=24)).isoformat(), now.isoformat()


class TaskWorkflow:
    """Per-task state machine."""

    def __init__(
        self,
        assignment: TaskAssignment,
        config: OrchestratorConfig,
        git_ops: GitOps,
        scheduler: SchedulerFacade,
        briefing: _BriefingLike,
        mcp: _McpLike | None,
        escalation_queue: EscalationQueue | None = None,
        escalation_event: asyncio.Event | None = None,
        usage_gate: UsageGate | None = None,
        initial_plan: dict | None = None,
        steward_factory=None,
        merge_queue: asyncio.Queue | None = None,
        merge_worker=None,
        merge_inflight_registry: InFlightMergeRegistry | None = None,
        event_store: EventStore | None = None,
        cost_store: CostStore | None = None,
        cancel_event: asyncio.Event | None = None,
        resume_session_id: dict | None = None,
        *,
        run_id: str | None = None,
        prompt_store: PromptArtifactStore | None = None,
    ):
        self.assignment = assignment
        self.config = config
        self.git_ops = git_ops
        self.scheduler = scheduler
        self.briefing = briefing
        self.mcp = mcp
        self.merge_queue = merge_queue
        # MergeWorker | SpeculativeMergeWorker | None — used by wip/unmerged
        # handlers to register halt ownership. The asyncio.Queue above carries
        # merge requests; this is the worker that owns the halt flag.
        self.merge_worker = merge_worker
        self.merge_inflight_registry = merge_inflight_registry
        self.event_store = event_store
        self.cost_store = cost_store
        # Prompt-artifact loader (shared/prompt_artifact.py, task 2492/2493):
        # None until injected (tests) or lazily built on first use in
        # _resolve_role_system_prompt (production — mirrors TaskCurator's
        # _prompt_store / _resolve_curator_prompt).
        self._prompt_store = prompt_store

        self.machine = WorkflowStateMachine(WorkflowState.PLAN)
        self._phase_cost_at_entry: float = 0.0
        self.task = assignment.task
        self.task_id = assignment.task_id
        self.modules = list(assignment.modules)
        self.worktree: Path | None = None
        self._worktree_external = False  # True when worktree was pre-created (eval mode)
        self._reify_debug_port: int | None = None
        self.artifacts: TaskArtifacts | None = None
        self.plan: dict = {}
        self.initial_plan = initial_plan
        self.metrics = WorkflowMetrics()

        # Per-module configs for scoped verification
        self._module_configs = self._resolve_module_configs()

        # Escalation support
        self.escalation_queue = escalation_queue
        self._escalation_event = escalation_event
        self._escalation_missing_warned: bool = False

        # Soft-cancel: settable from outside the workflow (e.g. by the
        # harness when reconciliation reports the task is now terminal,
        # or by the ``release_workflow`` MCP tool).  Long awaits in the
        # merge queue / steward grace period race against this event so
        # the workflow can exit promptly without waiting for a 900 s
        # timeout.  See Step 4 of the zombie-escalation fix.
        self._cancel_event: asyncio.Event = cancel_event or asyncio.Event()

        # Usage cap gate
        self.usage_gate = usage_gate

        # Unique session identifier for plan ownership (format: {task_id}-{uuid_hex[:8]})
        self.session_id = f'{self.task_id}-{uuid.uuid4().hex[:8]}'

        self._steward_factory = steward_factory
        self._steward: Any | None = None
        # In-process StewardOutcome channel (task 2248 / W9-delta): created
        # lazily in _ensure_steward_started and registered on the steward via
        # set_outcome_channel — replaces the escalation-queue forensic re-read
        # that _mark_blocked used to perform.
        self._steward_outcome_channel: asyncio.Queue | None = None
        self._config_dir: TaskConfigDir | None = None
        self._old_plan_base: str | None = None  # base commit from prior session (for revalidation diff)
        # Base commit for the current run's worktree (set in run() right after
        # create_worktree).  Captured here so _reconcile_metadata_files_for_done
        # can diff base..merge_sha and write the actually-changed paths instead
        # of the architect's plan.files (which the merge may have squashed).
        self._base_commit: str | None = None
        self._merge_sha: str | None = None  # merge commit SHA set by _submit_to_merge_queue on success
        # Set exclusively by _finalise_recovery_done — 'journal' (MergeProvenance
        # hit) or 'fallback' (_has_prior_implementation heuristic).  Stays None
        # unless an already-merged guard actually finalised a recovery-DONE
        # (PRD workflow-state-machine α, Contract §8 MP-2: no recovery-DONE
        # without a provenance basis).  Intentionally in-process-only: there is
        # no production reader today.  Its persisted counterpart is the `note`/
        # `done_provenance` scheduler.mark_done records in _finalise_recovery_done
        # (that's what a human or another process would inspect); this attribute
        # exists so tests can assert the MP-2 invariant directly against live
        # workflow state instead of re-deriving it from mock call args — see
        # TestNoPhantomDoneProperty in test_workflow_merge_provenance.py.
        self._merge_recovery_basis: str | None = None
        self._last_completed_role: str | None = None  # role of the last successfully-completed invocation
        self._last_verify_result: VerifyResult | None = None  # most recent failing VerifyResult from _verify_debugfix_loop
        # Durable verified-green checkpoint state (task 2752).
        # _verify_checkpoint_hit: set True for the current loop iteration when
        # the VERIFY-phase checkpoint fires (a durable prior-run workflow_verify
        # green exists at the current branch tip) and _verify_debugfix_loop is
        # SKIPPED.  Read in _enter_phase to SUPPRESS the VERIFY->REVIEW
        # workflow_verify re-emit — verify did not run this cycle, so
        # re-asserting a green would be dishonest (the honest signal is
        # phase_skipped(verify)).  Reset to False at the top of each loop pass.
        # _verify_green_tip_sha: the branch tip captured right after a PASSING
        # _verify_debugfix_loop and before _enter_phase(REVIEW); recorded in the
        # workflow_verify payload as the durable checkpoint key for the next run.
        self._verify_checkpoint_hit: bool = False
        self._verify_green_tip_sha: str | None = None
        # Set by _verify_debugfix_loop when a failure is classified as inherited
        # from main (preexisting break).  Read at the call site (run()) to route
        # _mark_blocked with dedupe_fingerprint instead of the generic reason.
        self._inherited_break_info: dict | None = None
        # Set by _verify_debugfix_loop when a VerifyInfraError exhausts the
        # bounded in-process retry window.  Read at the call site (run()) to
        # route _mark_blocked with category='infra_issue' + escalate_to_human.
        # Mirrors _inherited_break_info / _zero_output_hang_info stash idiom.
        self._infra_hold_info: dict | None = None
        # Set by the zero-output circuit breaker in _execute_iterations when
        # max_consecutive_zero_output_timeouts consecutive fresh-invocation
        # timeouts are detected.  Read by _execute_verify_review_loop to route
        # _mark_blocked with the distinct infra_issue reason instead of the
        # generic 'Execution iterations exhausted' (task 1739).
        self._zero_output_hang_info: dict | None = None
        # Set by the progress-resume churn circuit breaker in
        # _execute_iterations when config.max_progress_resume_iterations
        # ceiling-kill+resume cycles accumulate without convergence.  Read by
        # _execute_verify_review_loop to route _mark_blocked with the
        # distinct infra_issue reason (task 2360, reify-4827).
        self._progress_resume_churn_info: dict | None = None
        # When True, the finally-block config-dir cleanup is skipped so the
        # preserved dir can be used for forensic analysis.  Set alongside
        # _zero_output_hang_info when that circuit breaker trips (task 1739),
        # and alongside _progress_resume_churn_info when the churn breaker
        # trips (task 2360, reify-4827) — a repeatedly ceiling-killed
        # productive session is at least as worth preserving as a
        # zero-output wedge, since real work happened each time yet the
        # session never converged.
        self._preserve_config_dir: bool = False
        # Which breaker set _preserve_config_dir, for the forensic log line
        # in _cleanup_config_dir. Stays None only if _preserve_config_dir is
        # still at its False default.
        self._preserve_config_dir_reason: str | None = None
        # Per-run history of (category, normalised cause_hint) tuples for the
        # signature-repetition guard.  Ephemeral — intentionally not persisted
        # in task metadata because the verify loop is wholly within one
        # workflow run (unlike _check_infra_resume_thrash which crosses runs).
        self._failure_signature_history: list[tuple[str, str]] = []
        # TR-1: the atomic terminal-contract value returned by run().  Built
        # at the same choke points (_mark_blocked's _record helper, plus the
        # warm-lane-requeue and blast-radius-lock-conflict non-_mark_blocked
        # block paths); the harness reads it off workflow.run()'s return to
        # decide whether to increment the per-task requeue counter. A clean
        # DONE/CANCELLED exit (no block ever hit) has no stashed report, so
        # run() synthesizes one from machine.state instead.
        self._terminal_report: TerminalReport | None = None
        # Last blocked-from-merge-queue reason — captured by
        # _submit_to_merge_queue and consumed by the merge-phase thrash
        # check (Fix 3).  Cleared between merge attempts so a stale
        # reason from an earlier task slot can't poison the signature.
        self._last_merge_block_reason: str | None = None
        # Structured fingerprint fields from the post-merge VerifyResult —
        # task-1688.  Keyed by _merge_outcome_signature() for stable thrash
        # detection across retries where prose varies but root cause is the same.
        self._last_merge_failure_category: str = ''
        self._last_merge_failure_cause_hint: str = ''

        # Background asyncio.Tasks spawned by _spawn_dry_run_unblock.
        # Holding a strong reference prevents the event loop's weak-ref GC
        # from destroying long-running investigations before they complete.
        self._background_tasks: set = set()

        # In-task dedup cache for _route_review_suggestions_to_curator.
        # Stores the content_hash of the *most recently* routed suggestion
        # batch (scalar, not a set).  The escalation→resume cycle can re-enter
        # REVIEW→DONE with identical suggestions; this sentinel short-circuits
        # consecutive identical re-entries so the curator R4 gate never has to
        # absorb redundant N-HTTP batches.
        #
        # Boundary: sequence A→B→A will re-submit A on the third call because
        # B overwrites the cache entry.  This is intentional — the scalar
        # fast-path only eliminates *consecutive* duplicates.  The server-side
        # curator R4 idempotency gate (task_interceptor._check_escalation_idempotency)
        # is the durable source-of-truth dedup for non-consecutive repeats.
        self._last_routed_suggestion_hash: str | None = None

        # Dedup guard for _reconcile_done_step_commits' flag-for-review
        # branch (task 2386 amendment). That method runs on every
        # _execute_iterations loop iteration, and an unreconcilable orphaned
        # done-step commit (content mismatch / unresolvable original / no
        # WIP run at HEAD) is deliberately left unchanged rather than
        # guessed — so without this guard the identical condition would
        # re-fire _escalate_unreconciled_done_step and file a fresh info
        # Escalation every iteration up to max_execute_iterations, flooding
        # the queue with duplicates for one underlying anomaly. Keyed by
        # (step_id, stale_commit) rather than step_id alone so a genuinely
        # *new* orphaning event (the commit changes again) still escalates.
        # Never reset — including across _execute_iterations re-entry from
        # _replan/_amend — so the guard holds for this workflow instance's
        # whole lifetime.
        #
        # task 2764 — cross-restart durability: this in-memory set is
        # process-local, so an orchestrator restart constructs a fresh, empty
        # set and re-files every previously-filed pair. To close that gap, the
        # emitted keys are also persisted to the meta-root
        # (reconcile_state.json) and hydrated INTO this set on the workflow's
        # first _reconcile_done_step_commits pass (guarded by the
        # _loaded_persisted_step_escalations flag below), so a restarted
        # orchestrator files at most genuinely-new pairs.
        self._unreconciled_done_step_escalations: set[tuple[str, str]] = set()
        # False until the persisted emitted-escalation keys have been hydrated
        # into the set above (done lazily-once at the top of
        # _reconcile_done_step_commits, the set's sole consumer, where
        # self.artifacts is guaranteed constructed).
        self._loaded_persisted_step_escalations: bool = False

        # One-shot guard for the architect plan-tightening retry.  Set
        # True on first _try_narrow_plan call regardless of outcome so
        # the workflow never loops the narrowing pass on the same task.
        self._plan_tightened: bool = False

        # Crash-recovery resume: when the harness detected a surviving
        # agent-session sidecar at startup, it injects the parsed dict
        # here.  The next _invoke whose role matches will use --resume
        # against the original Claude session and a "continue" prompt
        # instead of spawning a fresh agent.  Consumed (set to None) on
        # first use so subsequent invocations are fresh.
        self._pending_resume_session_id: str | None = None
        self._pending_resume_role: str | None = None
        # Cumulative adopted-resume count carried across restarts in the v2
        # sidecar (task 2771).  β populates 'resume_count' in the recovered
        # sidecar dict; α reads it defensively (absent → 0) and applies the +1
        # increment at the adopted-resume point in _invoke.
        self._pending_resume_count: int = 0
        if resume_session_id:
            self._pending_resume_session_id = resume_session_id.get('session_id')
            self._pending_resume_role = resume_session_id.get('role')
            self._pending_resume_count = int(
                resume_session_id.get('resume_count', 0) or 0
            )
        # Session id stash for zero-output evidence enrichment.  Set in _invoke
        # right after session_id_val is determined; read by _capture_zero_output_evidence
        # so it can locate the transcript even when result.session_id is '' (hard SIGKILL).
        self._last_invoke_session_id: str | None = None

        # Architect's Claude session_id (the --session-id UUID), stashed in
        # _invoke when role.name == 'architect'.  Used by _build_spawn_env to
        # reconstruct the architect's SessionStart-hook registry slug as the
        # CLAUDE_SPAWN_PARENT_ID for post-architect roles (task 2512).  None
        # until an architect has run in this workflow instance.
        #
        # Deliberately in-process-only, NOT persisted to the crash-recovery
        # sidecar (self.artifacts.write_agent_session/clear_agent_session,
        # consumed via the resume_session_id ctor dict above) or rehydrated
        # on restart. If an orchestrator restart lands between the architect
        # finishing and a later role starting, the fresh Workflow instance's
        # copy is None again, so _build_spawn_env falls back to
        # self.session_id (workflow-root) as parent for that resumed role and
        # everything after it in this instance -- best-effort (parent is
        # never null, just less specific than a same-instance run would
        # produce), acknowledged by
        # test_implementer_falls_back_to_workflow_root_when_no_architect_ran.
        # A durable fix would need the architect's session id threaded
        # through harness.py's crash-recovery reconstruction, out of this
        # task's module scope.
        self._architect_spawn_session_id: str | None = None

        # PRD plans/task-status-authority-prd.md contract C4/D4 (task 2188,
        # omega1).  Process-level run_id (harness.py self._run_id), threaded
        # through so the dispatch-stamp claimant_run_id can embed it via
        # shared.task_claimant.compose_claimant_run_id.  None when the
        # workflow is constructed without a harness (e.g. some tests/evals);
        # compose_claimant_run_id callers treat that as an empty component.
        self._process_run_id: str | None = run_id
        # Background asyncio.Task running _claimant_heartbeat_loop, started
        # right after the dispatch stamp and cancelled in run()'s finally.
        self._claimant_heartbeat_task: asyncio.Task | None = None

    @property
    def state(self) -> WorkflowState:
        """Current workflow phase, delegated to :attr:`machine`.

        The setter bypasses transition validation (``machine.force_set``) —
        it exists because many tests stage a state directly (e.g.
        ``wf.state = WorkflowState.DONE``). Only :meth:`_enter_phase` drives
        a validated transition (``machine.transition``).
        """
        return self.machine.state

    @state.setter
    def state(self, value: WorkflowState) -> None:
        self.machine.force_set(value)

    @property
    def _task_files(self) -> list[str] | None:
        """Return the file list from the current plan, or None if unavailable/empty."""
        files = self.plan.get('files', [])
        return files if files else None

    @property
    def _train(self) -> TrainMembership | None:
        """Return the train membership dict if this task is a train member, else None.

        Reads ``task.metadata.train`` using the same dict-or-None shape that
        git_ops and scheduler already consume (β₂/β₁).  A non-dict value
        (e.g. a stale string) is treated as absent so callers get a clean
        ``is not None`` check.
        """
        train = (self.task.get('metadata') or {}).get('train')
        return cast(TrainMembership, train) if isinstance(train, dict) else None

    async def _enter_merge_deferred(self) -> WorkflowOutcome:
        """Park this train member in the merge-deferred holding state (γ₁, PRD §9.5).

        Called after the full execute→verify→review pipeline succeeds for a
        train member.  Instead of entering the merge phase, the task transitions
        to ``status='merge-deferred'`` and waits for the group-merge worker (δ₁)
        to drive the eventual done transition once all siblings are ready.

        After writing merge-deferred, invokes ``_maybe_enqueue_group_merge()``
        (δ₂).  When the trigger fires (all members deferred, self is tip), that
        method awaits the GroupMergeRequest and returns the final outcome
        directly.  When the train is incomplete (or self is not the tip), the
        trigger returns ``None`` and this method returns ``MERGE_DEFERRED`` to
        park the workflow.

        The worktree is NOT cleaned up here — the merge-queue worker's
        post-merge ``cleanup_worktree`` never runs for merge-deferred tasks,
        so the worktree is naturally preserved as γ₁'s observable signal.
        """
        self._enter_phase(WorkflowState.MERGE_DEFERRED)
        await self.scheduler.set_task_status(self.task_id, 'merge-deferred')
        # Clear any requeue counter accumulated from prior failed attempts: the
        # task is workspace-green now, so those attempts are no longer relevant.
        # The harness's _handle_outcome_post only special-cases REQUEUED/DONE;
        # MERGE_DEFERRED falls through with requeued=False, leaving a stranded
        # counter if we don't clear it here.
        self.scheduler.clear_requeue_count(self.task_id)
        logger.info(
            'Task %s: train member workspace-green — parking in merge-deferred '
            '(group-merge worker owns done transition)',
            self.task_id,
        )
        trigger_outcome = await self._maybe_enqueue_group_merge()
        if trigger_outcome is not None:
            return trigger_outcome
        return WorkflowOutcome.MERGE_DEFERRED

    async def _handle_superseded(self, result: MergeOutcome) -> WorkflowOutcome:
        """Park a single-task workflow whose merge was absorbed by a coalesced train or γ2 chain.

        Mirrors the PARK sequence of :meth:`_enter_merge_deferred` without the
        train-only ``_maybe_enqueue_group_merge()`` trigger — a superseded single
        task is not necessarily a train member, so that path must not be invoked.

        The absorbing request's ``mark_member_done`` callback (or the γ2 chain's
        terminal handler) is responsible for flipping status merge-deferred→done
        with merged provenance once the absorbing request lands.
        """
        self._enter_phase(WorkflowState.MERGE_DEFERRED)
        await self.scheduler.set_task_status(self.task_id, 'merge-deferred')
        # Clear any requeue counter accumulated from prior failed attempts: the
        # task's work is structurally complete — its branch was absorbed — so
        # old retry counts are no longer relevant.  Mirrors _enter_merge_deferred.
        self.scheduler.clear_requeue_count(self.task_id)
        if self.event_store:
            self.event_store.emit(
                EventType.merge_attempt,
                task_id=self.task_id,
                phase='merge',
                data={
                    'outcome': 'superseded',
                    'superseded_by': result.superseded_by,
                },
            )
        logger.info(
            'Task %s: merge absorbed into %s — parking in merge-deferred '
            '(absorbing request owns done transition)',
            self.task_id, result.superseded_by,
        )
        return WorkflowOutcome.MERGE_DEFERRED

    async def _maybe_enqueue_group_merge(self) -> WorkflowOutcome | None:
        """δ₂ trigger: enqueue a GroupMergeRequest when all train members are merge-deferred.

        Called immediately after ``_enter_merge_deferred`` writes the
        merge-deferred status.  Evaluates whether this task is the tip of a
        complete train and, if so, builds and enqueues a
        :class:`~orchestrator.merge_queue.GroupMergeRequest` from self's
        context (branch/worktree/task_files/module_configs/config).

        Gating (step-6 adds full guards; initial impl fires unconditionally
        when train metadata is present):

        * ``self._train is None`` → return ``None`` (non-train task; unreachable
          from ``_enter_merge_deferred`` but defensive).
        * ``members`` is empty → return ``None`` (no tasks found).
        * Any member is not merge-deferred (with self's status trusted as
          merge-deferred regardless of the get_tasks read lag) → return ``None``.
        * Self is not the tip (highest order) → return ``None`` (only the tip's
          branch contains all member commits).
        * ``self.worktree is None`` or ``self.merge_queue is None`` → log and
          return ``None``.

        Outcome mapping after awaiting the future:
        * soft-cancel (``_cancel_event`` wins) → ``_await_cancellable`` raises
          ``WorkflowCancelled('soft')`` (W9-θ), which propagates straight to
          ``run()``'s single catch — not handled in this method.
        * ``result.status == 'done'`` → ``WorkflowOutcome.DONE``
        * Any other status → ``_mark_blocked(..., escalate_to_human=True)``

        Returns the mapped ``WorkflowOutcome``, or ``None`` to signal the caller
        to park the workflow as ``MERGE_DEFERRED``.
        """
        from orchestrator.merge_queue import (
            TRAIN_INCOMPLETE_REASON_PREFIX,
            TRAIN_VERIFY_FAILED_REASON_PREFIX,
            GroupMergeRequest,
            MergeOutcome,  # noqa: F401 — kept for type completeness
            QueuedBranch,
            register_and_enqueue_merge_request,
        )

        train = self._train
        if train is None:
            return None

        train_id: str = train.get('id', '')  # type: ignore[assignment]
        if not train_id:
            return None

        members = await self.scheduler.tasks_by_train(train_id)
        if not members:
            return None

        # Guard: all members must be merge-deferred.
        # Trust self's just-written status (the get_tasks read may lag the write).
        def _effective_status(m: dict) -> str:
            return 'merge-deferred' if str(m.get('id')) == self.task_id else str(m.get('status', 'unknown'))

        if not all(_effective_status(m) == 'merge-deferred' for m in members):
            return None

        # Guard: self must be the tip (highest order → members[-1] after ascending sort).
        # Correctness depends on dispatch gating: β₁'s intra-train allowance lets
        # order=k start only once order=k-1 is merge-deferred, which guarantees the
        # tip (highest order) enters merge-deferred LAST.  The δ₁ group-merge worker
        # acts as a backstop: if this trigger does not fire (e.g. the tip re-enters
        # merge-deferred before a sibling propagates), the worker's own status pre-check
        # will reject the request and return TRAIN_INCOMPLETE, parking the train for retry.
        tip = members[-1]
        if str(tip.get('id')) != self.task_id:
            return None

        # Guard: infrastructure must be available.
        if self.worktree is None or self.merge_queue is None:
            logger.warning(
                'Task %s: train tip reached but worktree=%r merge_queue=%r — parking',
                self.task_id, self.worktree, self.merge_queue,
            )
            return None

        member_ids = [str(m.get('id')) for m in members]
        branch_name = self.task_id  # matches _submit_to_merge_queue convention

        async def _status_check(ids: list[str]) -> dict[str, str]:
            statuses, err = await self.scheduler.get_statuses(ids)
            if err is not None:
                # Log the transient MCP/read error so the silent discard is
                # visible.  We return the (possibly empty/partial) dict rather
                # than raising so the worker's existing train_incomplete pre-check
                # fires and routes back to the MERGE_DEFERRED park (see outcome
                # mapping below) instead of surfacing an unclassified 'Merge
                # worker error' that would escalate to a human.
                logger.warning(
                    'Task %s: train %r status_check got error from get_statuses '
                    '(will return partial %d-entry dict): %s',
                    self.task_id, train_id, len(statuses or {}), err,
                )
            return statuses or {}  # type: ignore[return-value]

        async def _mark_member_done(mid: str, sha: str) -> None:
            await self.scheduler.mark_done(
                mid, kind='merged', sha=sha, note=f'train {train_id}',
            )
            # task 2280 (PRD WA-3): consume the tip's write-ahead LandedRow inline
            # so a train-landed member no longer leaves a stale row surviving to the
            # next startup for RC-3 to prune. Idempotent (no-op for non-tip members),
            # fail-safe when unbound; runs only after a SUCCESSFUL mark_done (a raised
            # mark_done above skips it, leaving the row for the reconciler). Mirrors
            # the 2681 single-branch precedent at workflow.py:1912-1917.
            MergeProvenance.consume(mid)
            # Diff 5d (B3/T7): release warm lane for the done member.
            # Idempotent/never-raise via the shared primitive.
            await self.git_ops.release_lane_for_terminal_task(mid)

        if self.config.merge_verify_workspace:
            # workspace-wide verify ignores per-task scope; skip the member union loop
            union_task_files = self._task_files
            union_module_configs = self._module_configs
        else:
            union_task_files, union_module_configs = self._union_train_scope(members)

        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        # NOTE: merge_first_enqueued_at is intentionally NOT stamped here.
        # Train/group merges are out of α's scope (PRD §11.3/§11.4 open
        # questions on tie-break among coalesced members); ζ (task 1887) will
        # decide whether and how to stamp train members.  GroupMergeRequest
        # therefore inherits the None default from the base MergeRequest field,
        # and ζ's aging comparator falls back to enqueued_at for legacy/None.
        queued_branch = QueuedBranch.parse(branch_name, self.config.git.branch_prefix)
        req = GroupMergeRequest(
            task_id=self.task_id,
            branch=queued_branch,
            worktree=self.worktree,
            pre_rebased=False,
            task_files=union_task_files,
            module_configs=union_module_configs,
            config=self.config,
            result=future,
            train_id=train_id,
            member_task_ids=member_ids,
            tip_branch=queued_branch,
            tip_task_id=self.task_id,
            status_check=_status_check,
            mark_member_done=_mark_member_done,
        )

        logger.info(
            'Task %s: all %d train members merge-deferred — enqueuing GroupMergeRequest '
            '(train=%r, members=%r)',
            self.task_id, len(member_ids), train_id, member_ids,
        )
        # Only the tip branch (req.branch) is registered in the in-flight
        # registry.  Member branches are intentionally out of scope here:
        # GroupMergeRequest carries member_task_ids but not member branch names,
        # so registering them would require coupling knowledge of the branch-
        # naming convention into δ₂.  The on-disk _merge-* worktree scan in
        # coalesce_or_enqueue_merge_request provides a fallback for member-branch
        # coalescing after the train merge starts executing.
        await register_and_enqueue_merge_request(
            self.merge_queue, req, self.event_store, self.merge_inflight_registry,
        )

        # W9-θ: a cancel-win now raises WorkflowCancelled('soft') instead of
        # returning None — it propagates straight to run()'s single catch,
        # so there is no `result is None` branch to handle here any more.
        result = await self._await_cancellable(future)
        if result.status == 'done':
            if result.merge_sha:
                self._merge_sha = result.merge_sha
            return WorkflowOutcome.DONE
        # Distinguish transient from genuine failures.
        # train_incomplete: one or more members haven't propagated their
        # merge-deferred write yet (or a transient get_statuses error returned
        # a partial dict).  This is a timing artefact, not a real failure — park
        # the train and let δ₁ retry when the tip re-enters merge-deferred.
        if result.reason and result.reason.startswith(TRAIN_INCOMPLETE_REASON_PREFIX):
            logger.info(
                'Task %s: GroupMergeRequest returned train_incomplete for train %r '
                '— parking (MERGE_DEFERRED) for δ₁ retry. reason: %s',
                self.task_id, train_id, result.reason,
            )
            return None  # caller (_enter_merge_deferred) returns MERGE_DEFERRED
        # Attribution trigger: train union-verify red with a non-interaction-exempt
        # failure category → re-verify each member as a single (δ).  The prefix is
        # tagged in _do_train_merge ONLY on interaction-candidate verify-red outcomes
        # (failure_category set AND not main-health-red), so rebase-conflict,
        # main-health-red, transient-infra, unscoped-pyright, and wip-halt keep
        # their existing routing unchanged.
        if result.reason and result.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX):
            return await self._attribute_train_failure(result, train_id, members)
        # Orphan-halt probe: _map_advance_failure halted the merge queue (one of the
        # five halt-inducing statuses: wip_halted, done_wip_recovery,
        # wip_recovery_no_advance, unmerged_state, stash_failed) and halt_owner_esc_id is None,
        # so harness._on_escalation_resolved cannot unhalt the queue on L1 resolution.
        # Register the halt owner NOW so resolving the L1 auto-unhalts the queue
        # (parity with the single-task _handle_wip_conflict path — task 1599).
        # The probe is ground truth: it fires only when the queue IS halted with no
        # owner, is future-proof against new halt statuses, and never misfires on
        # genuine non-halt 'blocked' outcomes (rebase-conflict/cas_failed do NOT call
        # halt(), so is_wip_halted stays False for those paths).
        if (
            self.merge_worker is not None
            and self.merge_worker.is_wip_halted
            and self.merge_worker.halt_owner_esc_id is None
        ):
            return await self._escalate_train_halt(result, train_id)
        # Genuine failure (conflict/verify-red/rebase-conflict) → escalate to human
        # so a broken train is never silently parked forever (scenarios 5/8).
        return await self._mark_blocked(
            f'Group merge for train {train_id!r} failed: {result.status} — {result.reason}',
            escalate_to_human=True,
        )

    async def _train_candidates(self) -> list[dict]:
        """Discover other tasks that are merge-ready candidates for this train.

        Conservative proxy for "merge-ready": the task is in-progress, has no
        existing train metadata, and its branch resolves (i.e. the branch exists
        in the repo).  Self is always excluded.  Tighter merge-ready gating and
        trigger cadence/debounce are deferred to γ/ε; this proxy is safe because
        the former is off-by-default.

        The branch name for a task follows the established convention:
        task_id (bare, same as ``branch_name = self.task_id`` in run()).
        """
        # Server-side filter: only in-progress tasks are candidates; applying
        # the tightest filter here minimises the payload on this per-tick path.
        # The client-side status check below is kept as defence-in-depth.
        all_tasks: list[dict] = await self.scheduler.get_tasks(statuses=['in-progress'])
        # Quick filters that require no git I/O.
        pre_filtered: list[dict] = []
        for task in all_tasks:
            task_id: str = str(task.get('id', ''))
            # Exclude self.
            if task_id == self.task_id:
                continue
            # Exclude non-in-progress statuses (done, blocked, cancelled,
            # merge-deferred, deferred, …) — defence-in-depth against the
            # server-side filter missing any edge case.
            if task.get('status') != 'in-progress':
                continue
            # Exclude tasks already assigned to a train.
            metadata: dict = task.get('metadata') or {}
            if metadata.get('train'):
                continue
            pre_filtered.append(task)
        if not pre_filtered:
            return []
        # Branch-existence gate: fan out resolve_branch_sha concurrently
        # instead of one serial subprocess per candidate.
        _pre_ids = [str(t.get('id', '')) for t in pre_filtered]
        _shas = await asyncio.gather(*[
            self.git_ops.resolve_branch_sha(tid) for tid in _pre_ids
        ])
        return [t for t, sha in zip(pre_filtered, _shas, strict=True) if sha is not None]

    async def _maybe_form_train(self) -> bool:
        """β former: try to form a merge train for this task (PRD §7 β).

        Called at the merge decision point when the former is enabled and self
        is not already a train member.  Returns True iff a train was formed and
        self.task['metadata']['train'] was set (routing then sends self into
        merge-deferred).  Returns False in all other cases (self merges solo on
        the existing path).

        Guards (return False immediately):
          - merge_train_former_enabled is False — former is opt-in, off by default.
          - self._train is not None — self is already a train member (no double-forming).
          - merge_train_max_members < 2 — defensive; the ge=2 pydantic constraint
            should prevent this, but guard anyway to keep the invariant local.

        Formation (when guards pass and candidates exist):
          1. Fetch line ranges for self + each candidate.
          2. Select a mutually-stackable subset via _select_train_members (greedy, capped).
          3. Assign metadata.train={id, order, members} to every selected member via
             scheduler.update_task(append=True) — backend recursive-merge touches only
             the train key, preserving all other metadata keys without a race.
          4. Set self.task['metadata']['train'] in-memory so self._train flips truthy
             for the immediate merge-decision routing.
          5. Emit EventType.train_formed.
          6. Return True.
        """
        # Guard 1: former must be explicitly enabled.
        if not self.config.merge_train_former_enabled:
            return False
        # Guard 2: no double-forming — if self is already in a train, skip.
        if self._train is not None:
            return False
        # Guard 3: defensive cap sanity check.
        if self.config.merge_train_max_members < 2:
            return False

        # Discover merge-ready candidates.
        candidates = await self._train_candidates()
        if not candidates:
            return False

        # TOCTOU NOTE: Between _train_candidates() and the update_task writes
        # below, another anchor's former (running concurrently for a different
        # task) may assign trains to the same candidates.  The append=True
        # recursive merge is last-write-wins on the 'train' key, so a member
        # could end up carrying a train id whose members list disagrees with
        # its own.  Full closure via a compare-and-set or serialised-formation
        # lock is deferred to γ/ε (the former is off-by-default so this window
        # is safe to ship).

        # --- Selection ---
        # Coerce candidate ids to str so all member ids are string-typed,
        # consistent with self.task_id, regardless of the store's id type.
        candidate_ids: list[str] = [str(c['id']) for c in candidates]
        # Fan out line-range fetches concurrently (one git subprocess each).
        _all_range_ids = [self.task_id] + candidate_ids
        _range_results = await asyncio.gather(*[
            self.git_ops.get_changed_line_ranges(cid) for cid in _all_range_ids
        ])
        ranges_by_id: dict[str, dict[str, list[tuple[int, int]]]] = {
            cid: r for cid, r in zip(_all_range_ids, _range_results, strict=True)
        }

        selected = _select_train_members(
            self.task_id, candidate_ids, ranges_by_id,
            self.config.merge_train_max_members,
        )
        if len(selected) < 2:
            # No viable train (no stackable candidates or lone anchor).
            return False

        # --- γ: Materialize the linear branch stack ---
        # Rebase each successor's worktree onto the last surviving member's
        # branch.  Members that conflict during stacking are ejected; they
        # carry no train metadata and fall through to the solo merge path.
        stack_result = await self.git_ops.stack_train_branches(selected)
        survivors = stack_result.survivors
        ejected = stack_result.ejected

        # D4: abandon train formation when fewer than 2 members survive.
        if len(survivors) < 2:
            logger.info(
                'Task %s: train abandoned after stacking — only %d survivor(s) '
                '(ejected: %s); anchor merges solo.',
                self.task_id, len(survivors), ejected,
            )
            return False

        # --- Metadata assignment (survivors only) ---
        # NOTE: partial-metadata / half-formed-train hazard.
        # stack_train_branches has already physically rebased the survivor
        # branches before we reach this point.  If a scheduler.update_task
        # call below raises (or the process dies mid-loop), some survivors will
        # have train metadata and others will not — their branches are already
        # stacked but there is no metadata recording the relationship.
        # This is a tolerated TOCTOU window: the former is off-by-default
        # (merge_train_former_enabled=False), and the solo-merge fallback
        # recovers the survivors gracefully — a member without train metadata
        # falls through to the normal solo-merge path unchanged.  A future
        # hardening task should wrap this loop in a compensating transaction or
        # idempotent retry.
        train_id = f'train-{self.task_id}-{uuid.uuid4().hex[:8]}'
        all_members = survivors  # anchor first (order-0), then surviving members

        for order, member_id in enumerate(all_members):
            train_meta: dict = {
                'id': train_id,
                'order': order,
                'members': list(all_members),
            }
            await self.scheduler.update_task(
                member_id, {'train': train_meta}, append=True,
            )

        # Set self's metadata in-memory so self._train flips truthy immediately
        # for the subsequent merge-decision routing (avoids a round-trip read).
        (self.task.setdefault('metadata', {}))['train'] = {
            'id': train_id,
            'order': 0,
            'members': list(all_members),
        }

        # --- Event emission ---
        if self.event_store:
            self.event_store.emit(
                EventType.train_formed,
                task_id=self.task_id,
                data={
                    'train_id': train_id,
                    'members': list(all_members),
                    'size': len(all_members),
                    # Include ejected for δ failure-attribution telemetry.
                    'ejected': list(ejected),
                },
            )

        logger.info(
            'Task %s: train formed — id=%s members=%s ejected=%s',
            self.task_id, train_id, all_members, ejected,
        )
        return True

    async def _maybe_defer_as_train_member(self) -> WorkflowOutcome | None:
        """Merge-decision routing helper (PRD §7 β, design decision 4).

        Encapsulates the "form-or-defer-or-fall-through" logic so it can be
        unit-tested independently from the large run() state machine.

        Returns:
          - The MERGE_DEFERRED outcome (from _enter_merge_deferred) when self
            is — or becomes — a train member.
          - None when self is not a train member and the former either did not
            form a train or is disabled; the caller should fall through to the
            solo MERGE path.

        Non-train / former-disabled behavior is byte-identical to the previous
        inline ``if self._train is not None: return await self._enter_merge_deferred()``
        guard — the former returns False → None → caller falls through.
        """
        if self._train is None:
            await self._maybe_form_train()
        if self._train is not None:
            return await self._enter_merge_deferred()
        return None

    def _union_train_scope(
        self, members: list[dict],
    ) -> tuple[list[str] | None, list[ModuleConfig]]:
        """Compute the union verify scope over all train members.

        Starts from the tip's plan-derived scope (self._task_files / self.modules)
        so tip coverage never regresses, then folds in each member's metadata.files-
        derived files and modules (deduped by value for files, by mc.prefix for
        module_configs via _resolve_module_configs).
        """
        union_files: list[str] = list(self._task_files or [])
        seen_files: set[str] = set(union_files)
        union_modules: list[str] = list(self.modules)
        seen_modules: set[str] = set(union_modules)
        for member in members:
            member_files: list[str] = ((member.get('metadata') or {}).get('files')) or []
            for f in member_files:
                if f not in seen_files:
                    seen_files.add(f)
                    union_files.append(f)
            if member_files:
                for m in derive_modules(
                    member_files, self.config.lock_depth, task_id=self.task_id,
                ):
                    if m not in seen_modules:
                        seen_modules.add(m)
                        union_modules.append(m)
        return union_files or None, self._resolve_module_configs(union_modules or None)

    async def _finalise_merged_done(self) -> WorkflowOutcome:
        """Common DONE-finalisation for the happy-path merge.

        Refreshes ``metadata.files`` from the merge diff, then calls
        ``mark_done`` with ``kind='merged'``.  Extracted from ``run()`` so
        the state machine stays under pyright's complexity threshold.

        Returns ``WorkflowOutcome.DONE`` on success; ``BLOCKED`` (with L1)
        when the persistence layer refuses or ``_merge_sha`` is missing.
        """
        await self._reconcile_metadata_files_for_done()
        if not self._merge_sha:
            # Should never happen on the happy path — _submit_to_merge_queue
            # always populates _merge_sha on success.  Defensive bail-out
            # rather than passing done_provenance=None (which the server
            # gate now rejects with done_provenance_required).
            logger.error(
                'Task %s: SUCCESS path reached without _merge_sha — '
                'cannot construct done_provenance',
                self.task_id,
            )
            return await self._mark_blocked(
                'Internal: SUCCESS without merge_sha — provenance unconstructable',
                escalate_to_human=True,
            )
        try:
            await self.scheduler.mark_done(
                self.task_id, kind='merged', sha=self._merge_sha,
            )
        except SetTaskStatusRejected as exc:
            # Persistence layer refused the done write — the architect's
            # claim is contradicted by the row state.  Honest log + L1
            # instead of pretending the task is DONE.
            logger.error(
                'Task %s: set_task_status(done) rejected — %s: %s',
                self.task_id, exc.error_code, exc.raw,
            )
            return await self._mark_blocked(
                f'set_task_status(done) rejected: {exc.error_code} — {exc.raw}',
                escalate_to_human=True,
            )
        # task 2681/ζ: consume the write-ahead LandedRow on the happy path so it
        # no longer survives to the next startup (closing the RC-3 stale-row
        # window); only after a SUCCESSFUL mark_done — a rejected done-write
        # above returns early so the row survives for the reconciler to retry.
        # Fail-safe when the façade is unbound (bare-worker / eval / test).
        MergeProvenance.consume(self.task_id)
        logger.info(
            f'Task {self.task_id} DONE — '
            f'cost=${self.metrics.total_cost_usd:.2f} '
            f'invocations={self.metrics.agent_invocations}'
        )
        return WorkflowOutcome.DONE

    async def _reconcile_metadata_files_for_done(
        self, *, override_files: list[str] | None = None,
    ) -> None:
        """Set ``metadata.files`` to the merge-diff files before set_task_status('done').

        Truth source: ``git diff --name-only --no-renames <_merge_sha>^1..<_merge_sha>``
        (excluding ``.task/``) — the merge commit's OWN first parent, i.e. main's
        tip immediately before this task's merge — not the architect's plan.files
        (which the merge may have squashed, refactored, or rewritten).  Pre-fix,
        plan.files-derived metadata could include paths that no longer existed
        post-merge, tripping the phantom-done gate even though provenance was
        correct.

        ``self._base_commit`` is intentionally NOT a diff input here: it is the
        task's original branch point, captured once at worktree creation and
        never refreshed after a rebase.  Diffing it against ``_merge_sha`` would
        union in every sibling task that merged into main during the window
        between this task's branch point and its own merge — the cross-task
        ``metadata.files`` contamination this method used to produce.  The merge
        commit's first parent is captured atomically at merge time (``--no-ff``
        guarantees a two-parent merge commit — see ``advance_main``'s symmetric
        reliance on ``merge_sha^2`` for the verified branch tip) and yields
        exactly this task's own branch changes.

        Fall-back path: when ``_merge_sha`` is None (already-on-main shortcuts,
        eval mode without a merge), write an empty list — the fused-memory
        gate-skip-when-verified-provenance branch handles the missing-files
        case from there.

        ``override_files`` (task 2372, Layer C): when provided (not ``None``),
        it REPLACES the ``_merge_sha``-derived computation entirely — used by
        the pre-EXECUTE found_on_main recovery path, which has already computed
        the real ``base_commit..wt_head`` branch-content diff as the very
        evidence that authorized the recovery (see ``_recover_before_execute``).
        Stamping that diff here means a found_on_main DONE from that guard
        always carries real, non-empty files (ACTION #2) instead of the
        ``_merge_sha is None -> []`` blank that let task 2125's phantom-done
        slip the reconciliation gate undetected.  An empty list is a valid
        override (though the pre-EXECUTE guard never passes one — it returns
        None instead of recovering when the diff is empty).

        Sibling keys such as ``memory_hints`` and ``_causation_id`` (added by
        Stage-2 reconciliation after the workflow loaded ``self.task``) are
        preserved via the ``_merge_fresh_metadata`` read-modify-write; a
        pre-fix bare ``{'files': files}`` write clobbered them under the
        default ``append=False``.

        Directory-shaped entries from the merge diff (extension-less files like
        ``Dockerfile``, or non-allowlisted dotfiles like ``.gitignore``) are
        stripped via ``sanitize_files_for_persist`` before persisting, so
        ``metadata.files`` stays file-level and the done-reconcile
        ``update_task`` is not rejected by the lock-charter guard (changes #2/#3).
        The strip can only shrink the declared set (a file-level subset of what
        landed), which keeps the phantom-done gate honest.
        """
        if override_files is not None:
            files = override_files
        elif self._merge_sha:
            files, err = await self.git_ops.get_merge_commit_diff_files(
                self._merge_sha,
            )
            if err is not None:
                files = []
        else:
            files = []
        merged = await self._merge_fresh_metadata(
            self.task.get('metadata') or {},
            log_context='done metadata files reconcile',
        )
        merged['files'] = sanitize_files_for_persist(files)
        # Optimistically update in-memory so downstream reads in this session
        # see the expected state.  In-memory is intentionally optimistic; the
        # backend is the authority and will be re-read on the next reconcile
        # cycle if this write fails (consistent with _handle_no_plan_failure).
        self.task['metadata'] = merged
        await self.scheduler.update_task(self.task_id, merged)

    def _enter_phase(self, new_state: WorkflowState) -> None:
        """Transition to a new workflow phase, emitting events.

        The machine transition happens BEFORE event emission so an illegal
        move (e.g. a late phase change after a terminal state) raises
        ``IllegalTransition`` and short-circuits before any phase_exit/
        phase_enter event is emitted — no phantom events for a move that
        never actually happened.
        """
        prev = self.state
        self.machine.transition(new_state)
        if self.event_store:
            if prev not in (WorkflowState.DONE, WorkflowState.BLOCKED):
                cost_delta = self.metrics.total_cost_usd - self._phase_cost_at_entry
                self.event_store.emit(
                    EventType.phase_exit,
                    task_id=self.task_id, phase=prev.value,
                    cost_usd=cost_delta,
                )
            self.event_store.emit(
                EventType.phase_enter,
                task_id=self.task_id, phase=new_state.value,
            )
            # Task 2383 β: the branch's OWN pre-merge verify verdict, keyed
            # by task_id.  REVIEW is reachable from VERIFY only on a PASSING
            # verify (a failing verify routes to BLOCKED/ESCALATED before
            # _enter_phase(REVIEW) is ever called), so this edge is a
            # reliable "branch verified green pre-merge" signal.  Consumed
            # by the merge-skew attribution classifier's I5 branch-green
            # fact (merge_disposition._branch_pre_merge_verify_green), which
            # keys on task_id and reads only data['passed'] — base_sha/branch
            # below are informational only (kept for future telemetry/log
            # correlation; the classifier ignores them today).
            #
            # Note (reviewer_comprehensive, amendment round 2): I5 reads
            # "any-prior-green" (never most-recent-wins), so this row
            # survives untouched across a later review-bounce. If the branch
            # is re-executed after REVIEW and reaches this edge again having
            # introduced its OWN new bug, the stale green from before the
            # bounce can still cause that genuine BRANCH_BUG to be
            # misclassified as INTEGRATION_SKEW if an unrelated main landing
            # happens to overlap the same file. This is a documented I5
            # tradeoff (merge_disposition.py's any-prior-green keying is out
            # of this task's module scope to change); pinned by
            # test_merge_skew_end_to_end.py::TestReviewBounceStaleGreenTradeoff.
            #
            # Task 2752: 'tip_sha' (the branch tip captured right after the
            # passing verify) is the DURABLE cross-restart checkpoint key —
            # verify_checkpoint.green_checkpoint_at_tip matches on it to skip a
            # redundant re-verify at an unchanged tip in a later run.  The
            # `not self._verify_checkpoint_hit` gate SUPPRESSES this emit when
            # the checkpoint already fired this cycle: verify did NOT run, so
            # re-asserting workflow_verify(passed=True) would claim a verify
            # that never happened (the honest signal on a skip is
            # phase_skipped(verify), emitted in _execute_verify_review_loop).
            #
            # Consumer impact of this suppression (task 2752, step-10): on the
            # cross-restart fast-path the branch's only workflow_verify green
            # then lives under a PRIOR run_id, which WOULD have blinded the
            # run-scoped I5 reader and degraded a genuine INTEGRATION_SKEW to
            # INDETERMINATE.  So merge_disposition._branch_pre_merge_verify_green
            # was switched to the cross-run fetch_events_by_type_all_runs reader
            # (task 2752, step-10): it now sees the durable prior-run green, so
            # no INTEGRATION_SKEW classification is lost.  The remaining
            # run-scoped consumer, merge_completion.merge_completion_eligible,
            # still degrades to the documented task-2633 human-/unblock restart
            # gap and is intentionally left unchanged.  (Both read only
            # data['passed'], so the added tip_sha key itself affects neither.)
            if (
                prev is WorkflowState.VERIFY
                and new_state is WorkflowState.REVIEW
                and not self._verify_checkpoint_hit
            ):
                self.event_store.emit(
                    EventType.workflow_verify,
                    task_id=self.task_id,
                    data={
                        'passed': True,
                        'tip_sha': self._verify_green_tip_sha,
                        'base_sha': self._base_commit,
                        'branch': f'{self.config.git.branch_prefix}{self.task_id}',
                    },
                )
        self._phase_cost_at_entry = self.metrics.total_cost_usd

    async def _setup_worktree_and_artifacts(self, branch_name: str) -> None:
        """Set in-progress, create/inspect the worktree, init artifacts, scrub .task/.

        Extracted from ``run()`` so the state machine stays under pyright's
        complexity threshold.  Side-effects:

        - ``set_task_status('in-progress')``
        - populates ``self.worktree`` / ``self._base_commit`` /
          ``self._config_dir`` / ``self.artifacts`` / ``self._old_plan_base``
        - sets ``self._worktree_external = True`` when the caller pre-created
          the worktree (eval mode) so the cleanup path knows to leave it alone
        - syncs per-worktree venvs unless the worktree is external
        - removes any ``.task/`` paths inherited as tracked from main
        """
        # Pre-empt race: read live status BEFORE claiming 'in-progress'.
        # If the task was cancelled (or otherwise terminal) out-of-band in the
        # ~3 s window between acquire and dispatch, raise TerminalExitRejection
        # so run()'s existing 1489 handler converts it to WorkflowOutcome.CANCELLED
        # without an 'in-progress' write, no escalation, and no reopen.
        # A None / get_status failure is not in TERMINAL_STATUSES → fall through.
        live_status = await self.scheduler.get_status(self.task_id)
        if live_status in TERMINAL_STATUSES:
            raise TerminalExitRejection(
                task_id=self.task_id,
                old_status=live_status,
                target_status='in-progress',
                raw='preempt: task terminal at dispatch',
            )

        # PRD task-status-authority C4/D4 (task 2188, omega1): stamp the
        # claimant atomically with the dispatch status write, so there is no
        # window where the task is in-progress with no live claimant.
        await self.scheduler.set_task_status(
            self.task_id, 'in-progress',
            claimant_run_id=compose_claimant_run_id(
                self._process_run_id or '', self.session_id, os.getpid(),
            ),
            heartbeat_at=datetime.now(UTC).isoformat(),
        )
        self._claimant_heartbeat_task = asyncio.create_task(
            self._claimant_heartbeat_loop(),
        )

        # Create worktree (captures base commit for stable diffs).
        # If the caller pre-created the worktree (eval mode) skip creation
        # and rev-parse HEAD instead.
        if self.worktree is None:
            # Pass the live task title so create_worktree can quarantine a
            # reused worktree whose stored identity belongs to a different
            # (recycled-id) task — defense-in-depth behind Fix C's flag.
            train_meta = (self.task.get('metadata') or {}).get('train')
            worktree_info = await self.git_ops.create_worktree(
                branch_name,
                expected_title=(
                    self.task.get('title')
                    if self.config.worktree_identity_guard_enabled
                    else None
                ),
                train=cast(TrainMembership, train_meta) if isinstance(train_meta, dict) else None,
            )
            self.worktree = worktree_info.path
            self._reify_debug_port = worktree_info.reify_debug_port
            base_commit = worktree_info.base_commit
        else:
            self._worktree_external = True
            proc = await asyncio.create_subprocess_exec(
                'git', 'rev-parse', 'HEAD',
                cwd=str(self.worktree),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            # Soft-fail: log and fall through rather than raising, so a
            # misconfigured eval-mode worktree degrades gracefully to
            # citation-grep alone instead of crashing setup.
            # NOTE: unlike the update_task soft-fail below (which keeps a valid
            # SHA in memory because only the metadata write failed), this path
            # has no SHA to record — _base_commit is left None so consumers
            # gating on `is not None` (e.g. the merge-queue plan-files-touched
            # check at line 3062) skip cleanly instead of receiving an empty-SHA.
            if proc.returncode != 0:
                logger.warning(
                    'Task %s: git rev-parse HEAD failed in external worktree '
                    '(rc=%s); base_commit will be None. stderr=%s',
                    self.task_id, proc.returncode,
                    stderr.decode(errors='replace').strip()[:200] or '<empty>',
                )
                base_commit = None
            else:
                base_commit = stdout.decode().strip()
        self._base_commit = base_commit
        # Record branch_base_sha in task metadata immediately after branch
        # creation so _reconcile_one_stranded (harness.py) can detect
        # zero-commit branches that sit on a main ancestor (Guard 3 in
        # the is_ancestor fast-path) and stale merge-markers from prior
        # incarnations of the same task id (stale-marker check).
        # Soft-fail: a transient update_task failure degrades gracefully
        # to citation-grep alone (Guards 1+2) instead of crashing setup.
        if base_commit:
            try:
                await self.scheduler.update_task(
                    self.task_id,
                    {'branch_base_sha': base_commit},
                    append=True,
                )
            except Exception:
                logger.warning(
                    'Task %s: failed to record branch_base_sha=%s; '
                    'reconciler will fall back to citation-grep guard',
                    self.task_id, base_commit,
                )
        # Colocate the per-task Claude config dir inside the worktree so
        # the session JSONL travels with the worktree.  Crash recovery
        # needs both the sidecar and the session file together to resume.
        self._config_dir = TaskConfigDir(
            self.task_id, base_dir=self.worktree / '.task',
        )

        if not self._worktree_external:
            await self._sync_worktree_venvs()

        self.artifacts = TaskArtifacts(
            self.worktree, _meta_root_for_worktree(self.worktree),
        )
        # Capture old base_commit before init() overwrites metadata.json
        # — _plan() uses it for revalidation diff.
        self._old_plan_base = self.artifacts.read_base_commit()
        self.artifacts.init(
            self.task_id,
            self.task.get('title', ''),
            self.task.get('description', ''),
            base_commit=base_commit,
        )
        # Single-source plan.json: make the lane copy
        # <worktree>/.task/plan.json a symlink into the durable meta-root plan
        # (task 2763), so any residual reader of the lane path resolves to the
        # meta-root file — the esc-5205-9 stale-trap is then impossible.
        # Recreated each dispatch; no-op in legacy meta_root=None mode.
        self.artifacts.ensure_lane_plan_symlink()

        # ── Layer B: stale iteration-log hygiene on re-dispatch ─────
        # init() only overwrites metadata.json — iterations.jsonl otherwise
        # survives across re-dispatch.  When this is a definitive re-dispatch
        # onto a NEW fork point (old base_commit known and different from the
        # fresh one), a prior dispatch's entries could otherwise masquerade as
        # current-branch evidence for recovery guards that consult the
        # iteration log (task 2372, recurrence class 2125/2315/2340).
        # Same-base crash-resume (`_old_plan_base == base_commit`) and the
        # first-ever dispatch (`_old_plan_base is None`) are left untouched so
        # in-flight progress survives; external (eval-mode) worktrees are
        # never wiped since their lifecycle is caller-owned.
        if (
            not self._worktree_external
            and self._old_plan_base is not None
            and base_commit is not None
            and self._old_plan_base != base_commit
        ):
            logger.warning(
                'Task %s: re-dispatch onto new base (%s→%s) — '
                'clearing stale iterations.jsonl from prior dispatch',
                self.task_id, self._old_plan_base[:8], base_commit[:8],
            )
            self.artifacts.clear_iteration_log()

        # ── .task/ contamination guard ────────────────────────────
        # If .task/ slipped into git on main (inherited contamination),
        # untrack it here so agents don't accidentally commit it.
        # Defense-in-depth — create_worktree() should have scrubbed, but
        # this catches the eval-mode path and race conditions.
        rc, tracked, _ = await _run(
            ['git', 'ls-files', '--', '.task/'],
            cwd=self.worktree,
        )
        if rc == 0 and tracked.strip():
            logger.warning(
                'Task %s: .task/ is tracked in git (inherited contamination) '
                '— removing from index. Files: %s',
                self.task_id, tracked.strip()[:200],
            )
            await _run(
                ['git', 'rm', '-r', '--cached', '--', '.task/'],
                cwd=self.worktree,
            )

    async def _claimant_heartbeat_loop(self) -> None:
        """Background loop refreshing heartbeat_at on a bounded cadence.

        PRD task-status-authority C4/D4 (task 2188, omega1): guarantees the
        claimant's heartbeat stays fresh even across one long phase (e.g. a
        30-minute execute-agent call), which a phase-transition hook alone
        would miss. Deliberately uses ``scheduler.set_task_claimant`` — never
        ``set_task_status`` — so this refresh never re-triggers the
        status-FSM/reconciliation. ``claimant_run_id`` is intentionally NOT
        passed on each tick: this is a refresh of liveness, not a re-stamp of
        identity. Runs until cancelled by :meth:`_stop_claimant_heartbeat`;
        ``scheduler.set_task_claimant`` is itself best-effort, so a transient
        failure here just means the next tick tries again.
        """
        # Defensive interval guard (task 2780): read the cadence once and bail
        # loudly if it is not a positive real number. The only real-world
        # trigger is a fully-mocked OrchestratorConfig in tests — production
        # config is pydantic-schema-validated, so this branch is unreachable in
        # prod and there is zero prod behaviour change. Without it, a bare
        # MagicMock interval reaches ``asyncio.sleep(<MagicMock>)``, whose
        # ``if delay <= 0`` raises ``TypeError: '<=' not supported between
        # instances of MagicMock and int`` (MagicMock's comparison dunders
        # default to NotImplemented) onto the (often shared, xdist) event loop,
        # polluting a later innocent test. The ``isinstance`` check is FIRST so
        # a MagicMock short-circuits (via ``or``) before the ``<= 0`` comparison
        # that would itself raise that guarded TypeError. The heartbeat refresh
        # is already documented as best-effort, so not running it under an
        # invalid interval is safe; the WARNING keeps the exit loud
        # (no-silent-fail-soft). Note: the cadence is read ONCE here rather than
        # per iteration, so a mid-loop hot-reload of
        # ``claimant_heartbeat_interval_secs`` does not take effect until the
        # loop is next (re)started. This is intentional and harmless — the loop
        # is per-task and short-lived, the interval is not a documented
        # hot-reload knob, and reading once keeps the guard and the sleep using
        # a single consistent value.
        interval = self.config.claimant_heartbeat_interval_secs
        if not isinstance(interval, (int, float)) or interval <= 0:
            logger.warning(
                'Task %s: claimant_heartbeat_interval_secs is not a positive '
                'number (%r) — heartbeat loop exiting without running',
                self.task_id,
                interval,
            )
            return
        while True:
            await asyncio.sleep(interval)
            await self.scheduler.set_task_claimant(
                self.task_id, heartbeat_at=datetime.now(UTC).isoformat(),
            )

    async def _stop_claimant_heartbeat(self) -> None:
        """Cancel and await the heartbeat loop task, if one was started.

        Called first from :meth:`_on_terminal_cleanups` (before the harness
        clears the claimant at slot release) so the loop can never race a
        post-clear re-stamp. A no-op when the loop was never started (e.g.
        dispatch raised before reaching that point).

        W9-θ: this now runs as the first entry of the ordered ``on_terminal``
        list (moved out of ``_drive()``'s own ``finally``), so an unexpected
        failure inside the loop itself (vs. the expected ``CancelledError``
        from the ``.cancel()`` above) would otherwise propagate out of
        ``CancellationScope.supervise`` and abort every LATER cleanup entry —
        including lane release.  The loop's own refresh call is already
        documented as best-effort; catching-and-logging here extends that
        same guarantee to the loop's teardown, so one bad heartbeat tick can
        never take down the rest of terminal cleanup.
        """
        if self._claimant_heartbeat_task is None:
            return
        self._claimant_heartbeat_task.cancel()
        try:
            await self._claimant_heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                f'Task {self.task_id}: claimant heartbeat loop failed '
                f'(non-fatal — continuing terminal cleanup)'
            )
        self._claimant_heartbeat_task = None

    async def _drive(  # pyright: ignore[reportGeneralTypeIssues]
        self,
    ) -> WorkflowOutcome:
        # reportGeneralTypeIssues: pyright reports "Code is too complex to
        # analyze" on this method when run without the project pyproject.toml
        # config (typeCheckingMode = "basic").  The root cause is the number
        # of exception-handler branches added for infra-retry and bypass-done
        # discrimination; refactoring them out is a separate architectural
        # task.  The ignore is on the def line (the only site pyright accepts
        # it for this diagnostic) and is narrowed to reportGeneralTypeIssues
        # so genuine type regressions in other categories are still caught.
        """Execute the full state machine.

        Internal — the WorkflowOutcome propagation currency for the ~55
        ``_mark_blocked``-callers and their ``== WorkflowOutcome.X`` checks
        (W9-γ decided retyping all of them was disproportionate for one
        spine leaf). ``run()`` is the public boundary: a thin wrapper below
        that turns this outcome into the returned ``TerminalReport`` (TR-1).
        """
        branch_name = self.task_id
        try:
            await self._setup_worktree_and_artifacts(branch_name)
            # After setup, both attributes are guaranteed populated.
            # The asserts narrow the types so the rest of run() can
            # use self.worktree / self.artifacts without re-checking.
            assert self.worktree is not None
            assert self.artifacts is not None

            # ── Pre-PLAN ghost-loop recovery ──────────────────────────
            # If the worktree's branch is already merged to main AND there
            # is evidence of prior implementation work, mark the task DONE
            # immediately — a prior run merged it but died before writing
            # the DONE status.  Short-circuiting here prevents the architect
            # from being invoked and keeps the run idempotent.
            #
            # NOTE: a related guard runs just below (before EXECUTE) and one
            # more runs inside the merge phase.  All three already-merged
            # guards now share one journal-first check — MergeProvenance.lookup
            # — and finalise identically through _finalise_recovery_done
            # (PRD workflow-state-machine α).
            recovery = await self._recover_if_already_merged()
            if recovery == WorkflowOutcome.DONE:
                return recovery

            # ── Durable merge-retry resume (task 2795) ────────────────
            # If a merge-phase escalation was resolved via `resume`, _requeue
            # stamped metadata.merge_retry_pending and returned REQUEUED for an
            # in-RAM in-place retry that a restart would lose (Reify 5166). On
            # re-dispatch, if the post-rebase worktree HEAD still matches the
            # stamped branch_head, jump straight to the merge phase — skipping
            # plan/execute/verify/review. Ordered AFTER the already-merged DONE
            # guard (a landed branch wins) and BEFORE the SIMPLE_TASK/PLAN block.
            _mrp = await self._resume_merge_retry_if_pending(branch_name)
            if _mrp is not None:
                return _mrp

            # ── Lever C: SIMPLE_TASK optimistic path ──────────────────
            # Dispatch a single Sonnet agent that explores, plans (via
            # plan-tools MCP), and implements end-to-end when the task's
            # author declared metadata.complexity='simple' and no
            # hard-blocker token in the description contradicts that
            # declaration.  The priority guard and file/lock-footprint
            # caps are intentionally absent — a simple task may be
            # high-priority or span several files as long as the change is
            # mechanically simple.  Falls through to the architect path on
            # any failure (no plan written, unactionable artifact, no steps
            # marked done).  metadata.force_full_path is the hard escape
            # that always forces the full path.
            if self._should_run_simple_task():
                self._enter_phase(WorkflowState.PLAN)
                simple_outcome = await self._run_simple_task()
                if simple_outcome == WorkflowOutcome.PLANNED:
                    # Plan written + step(s) marked done by SIMPLE_TASK.
                    # _execute_iterations will see no pending steps and
                    # return cleanly, allowing VERIFY to take over.
                    pass  # fall through to the post-PLAN section below
                elif simple_outcome == WorkflowOutcome.DONE:
                    # SIMPLE_TASK reported task_already_done.
                    return simple_outcome
                elif simple_outcome == WorkflowOutcome.BLOCKED:
                    # SIMPLE_TASK reported unactionable_task — terminal.
                    return simple_outcome
                else:
                    # Fallthrough sentinel — drop the plan so the
                    # architect path runs cleanly.
                    if self.artifacts is not None:
                        plan_path = self.artifacts.root / 'plan.json'
                        plan_path.unlink(missing_ok=True)
                        (self.artifacts.root / 'plan.lock').unlink(
                            missing_ok=True,
                        )

            # PLAN (skip if initial_plan was provided — eval mode, or if
            # the SIMPLE_TASK path above already populated self.plan)
            if self.initial_plan:
                plan_tid = self.initial_plan.get('task_id')
                if plan_tid and plan_tid != self.task_id:
                    logger.error(
                        f'Task {self.task_id}: initial_plan has mismatched '
                        f'task_id {plan_tid} — discarding, will re-plan'
                    )
                    self.initial_plan = None
            if self.plan:
                # SIMPLE_TASK already populated self.plan — skip the
                # initial_plan / _plan() branch entirely.
                pass
            elif self.initial_plan:
                self.artifacts.write_plan(self.initial_plan)
                self.artifacts.stamp_plan_provenance(self.session_id)
                self.artifacts.lock_plan(self.session_id)
                self.plan = self.artifacts.read_plan()
                logger.info(
                    f'Task {self.task_id}: using provided plan '
                    f'({len(self.plan.get("steps", []))} steps)'
                )
            else:
                self._enter_phase(WorkflowState.PLAN)
                plan_outcome = await self._plan()
                if plan_outcome == WorkflowOutcome.REQUEUED:
                    return WorkflowOutcome.REQUEUED
                if plan_outcome == WorkflowOutcome.BLOCKED:
                    if self.event_store:
                        self.event_store.emit(
                            EventType.waste_detected,
                            task_id=self.task_id, phase='plan',
                            data={'waste_type': 'replan_after_failure'},
                        )
                    return plan_outcome  # _plan() already called _mark_blocked
                if plan_outcome == WorkflowOutcome.DONE:
                    # _handle_already_done_report set status=done with provenance
                    # and returned DONE.  Falling through to EXECUTE/VERIFY/MERGE
                    # wastes effort and (per the 2026-05-04 incident) can orphan a
                    # merge-queue halt if the workflow gets soft-cancelled
                    # mid-merge.
                    return plan_outcome
                if plan_outcome == WorkflowOutcome.ESCALATED:
                    # _mark_blocked returned ESCALATED: status='blocked' was already
                    # written, an L1 escalation is open, the steward handed off to
                    # a human.  Falling through to the post-PLAN ghost-loop guard
                    # would mistake any architect scratch in the worktree for
                    # "prior merge survived" and try to flip status to 'done'
                    # (the done_provenance gate currently blocks the DB write but
                    # the workflow logs a fake DONE — don't lie).  See task 2911
                    # incident 2026-05-06; mirrors 9760ba74bf for the DONE case.
                    return plan_outcome
                # WorkflowOutcome.PLANNED falls through to execute/verify/review.

            # ── Ghost-loop early exit (before EXECUTE) ───────────
            # Journal-first (PRD workflow-state-machine α, MP-1/MP-2): a
            # MergeProvenance hit or the _has_prior_implementation fallback
            # both route through _recover_before_execute() and the shared
            # _finalise_recovery_done chokepoint — the single source the
            # three already-merged guards now share.
            _r = await self._recover_before_execute()
            if _r is not None:
                return _r
            # Normal path: EXECUTE + VERIFY + REVIEW loop (with escalation retry)
            while True:
                outcome = await self._execute_verify_review_loop()
                if outcome == WorkflowOutcome.ESCALATED:
                    self._enter_phase(WorkflowState.ESCALATED)
                    await self._ensure_steward_started()
                    logger.info(f'Task {self.task_id}: waiting for escalation resolution')
                    try:
                        resolution = await self._wait_for_resolution()
                    except _StewardReescalated as reesc:
                        return await self._mark_blocked(
                            'Steward re-escalated to human',
                            detail=_format_reescalation_detail(reesc.escalations),
                            skip_escalation=True,
                        )
                    # If branch is already on main AND has genuine prior
                    # implementation work (e.g. steward merged during
                    # resolution), skip re-implementation — proceed to
                    # MERGE which will detect already_merged. A raw
                    # is_ancestor check alone is NOT enough: an empty
                    # branch's tip IS main's HEAD, so is_ancestor trivially
                    # passes and would wrongly skip resume, leaving the
                    # branch un-implemented (task 2504). wt_head is passed
                    # for the SHA-primary has-work signal — this path holds
                    # a reliable post-execution HEAD, so an empty branch
                    # correctly resolves has_work=False and resume proceeds.
                    _, wt_head_raw, _ = await _run(
                        ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
                    )
                    wt_head = wt_head_raw.strip()
                    esc_main_sha = await self.git_ops.get_main_sha()
                    if await self._branch_work_landed_on_main(
                        wt_head, esc_main_sha, wt_head=wt_head,
                    ):
                        logger.info(
                            'Task %s: branch already on main after '
                            'escalation resolution — skipping '
                            're-implementation', self.task_id,
                        )
                        break

                    # Honor steward terminal decisions BEFORE resuming the
                    # implementer.  The steward may have set the task to
                    # done / cancelled / deferred / blocked while resolving
                    # the L0 (e.g. queued a follow-up task that is now the
                    # durable fix and deferred this one onto it).  Without
                    # this guard the resume loop keeps invoking the
                    # implementer/debugger until verify-attempt budget
                    # exhausts — burning $7-8 per cycle on a task the
                    # steward already decided to park.  Mirrors the inline
                    # returns inside _mark_blocked (~L3125–3168) but
                    # bypasses it so we do NOT file an L1 for what is an
                    # intentional steward terminal decision.
                    current_status = await self.scheduler.get_status(self.task_id)
                    if current_status == 'done':
                        self._enter_phase(WorkflowState.DONE)
                        return WorkflowOutcome.DONE
                    if current_status in WORKFLOW_PRESERVE_STATUSES:
                        logger.info(
                            'Task %s: steward set status to %s during '
                            'escalation resolution — preserving, exiting '
                            'resume loop',
                            self.task_id, current_status,
                        )
                        self._enter_phase(WorkflowState.BLOCKED)
                        return WorkflowOutcome.BLOCKED

                    # Fix 2 — anti-thrash guard for repeated infra-issue
                    # resumes on the same root cause.  Status is confirmed
                    # non-terminal here (Fix 1 guard above), so it's safe
                    # to count this as a real resume attempt.  At
                    # threshold the helper short-circuits to BLOCKED + L1
                    # so a human can intervene rather than the orchestrator
                    # dispatching the implementer/debugger again.
                    thrash_outcome = await self._check_infra_resume_thrash()
                    if thrash_outcome is not None:
                        return thrash_outcome

                    # Fold any steward-granted scope expansion (task 2505)
                    # into plan.files/metadata.files/locks BEFORE resuming
                    # the implementer — otherwise the grant lives only as
                    # free-text resolution prose and the resumed
                    # implementer's briefing/_task_files never reflect the
                    # expanded scope. granted_files is read from resolved
                    # escalation records, not the joined `resolution` string.
                    granted = self._collect_granted_files()
                    if granted:
                        current_files = self.plan.get('files', [])
                        new_files = current_files + [
                            f for f in granted if f not in current_files
                        ]
                        # Fire whenever the grant actually WIDENS the file
                        # set — not only on a module-set change. A
                        # same-package sibling grant leaves the module set
                        # unchanged yet still must reach plan.files /
                        # metadata.files / _task_files, so route every real
                        # widen through _set_task_scope. (Task 2505 reviewer
                        # regression: the old `set(new_modules) !=
                        # set(self.modules)` gate silently dropped a
                        # same-module grant — plan.files was never widened.)
                        # The `and` short-circuits so _set_task_scope only
                        # runs on a real widen; on its False return (a genuine
                        # cross-module lock conflict) we requeue.
                        if (
                            new_files != current_files
                            and not await self._set_task_scope(new_files)
                        ):
                            # Lock conflict on a genuine cross-module
                            # expansion: the scheduler already requeued the
                            # task to pending and persisted
                            # metadata.files=new_files on its own (plan.json
                            # was widened to match by _set_task_scope before
                            # the conflict was detected). Do NOT resume the
                            # implementer under a lock a sibling task still
                            # holds — requeue and let this task redispatch
                            # once the lock frees.
                            # _requeue_on_lock_conflict re-derives the
                            # missing modules from new_files (α-strip via
                            # derive_modules, task 2373 amendment) on the
                            # same basis as the real conflict detection in
                            # _reconcile_scope_locks — a directory-shaped
                            # grant entry must not manufacture a phantom
                            # subtree module in the diagnostic.
                            return await self._requeue_on_lock_conflict(
                                new_files, 'scope_grant_lock_conflict',
                                'Scope grant',
                            )

                    # Resume with resolution context
                    logger.info(f'Task {self.task_id}: resuming after escalation resolution')
                    resume_prompt = await self.briefing.build_resume_prompt(
                        self.task, self.plan,
                        '\n'.join(e.summary for e in self._check_escalations()),
                        resolution, self.worktree,
                    )
                    await self._invoke(IMPLEMENTER, resume_prompt, self.worktree)
                    continue
                if outcome != WorkflowOutcome.DONE:
                    return outcome
                break

            # MERGE + SUCCESS/finalise tail — extracted into _merge_and_finalise
            # (task 2795) so the merge-retry resume guard can reuse it. Always
            # returns a WorkflowOutcome (a terminal merge outcome verbatim, or
            # the finalise-DONE outcome on merge success), so _drive never
            # mistakes a merge success for a fall-through to PLAN.
            return await self._merge_and_finalise(branch_name)

        except WorkflowCancelled:
            # W9-θ: a small handful of call sites inside this try block
            # (the merge-retry loop's explicit cancel re-check, and
            # _await_cancellable via _submit_to_merge_queue /
            # _maybe_enqueue_group_merge) now raise WorkflowCancelled
            # directly as ordinary control flow — not via CancellationScope's
            # own event-race, which never enters this method's body at all.
            # WorkflowCancelled IS an Exception subclass (unlike
            # asyncio.CancelledError), so without this clause it would be
            # swallowed by the generic `except Exception` ladder below and
            # misreported as a BLOCKED workflow error. Bare re-raise lets it
            # propagate to CancellationScope/run()'s single catch site,
            # preserving CX-1 ("caught at EXACTLY ONE place").
            raise

        except SetTaskStatusRejected as exc:
            # Fast-path: a terminal-status rejection arrived out-of-band before
            # setup completed (either from the pre-empt live-status read or from
            # set_task_status('in-progress') itself).  Route through the same
            # bypass-discrimination helpers 1489 introduced for mid-flight aborts:
            # - 'cancelled'  → CANCELLED, no reopen, no escalation (sub-case 0).
            # - 'done', provenance on main  → DONE, no escalation (legitimate done).
            # - 'done', provenance missing / off-main  → reopen + L1 bypass_done +
            #   BLOCKED (bypass detected).
            # Only non-TerminalExitRejection subclasses (rare persistence errors)
            # fall through to the 'unhandled rejection' L1 path below.
            if isinstance(exc, TerminalExitRejection):
                cancelled_outcome = self._handle_cancelled_terminal_exit(exc)
                if cancelled_outcome is not None:
                    return cancelled_outcome

                # Not 'cancelled' (must be 'done' given TERMINAL_STATUSES).
                # Reuse the bypass-discrimination helper: returns BLOCKED when a
                # bypass is detected (reopen + bypass_done L1 already filed), or
                # None when provenance is legitimate (commit reachable from main).
                bypass_outcome = await self._handle_terminal_exit_on_block(
                    exc,
                    reason=(
                        f'terminal at dispatch (old_status={exc.old_status!r})'
                    ),
                    detail='preempt: task terminal at dispatch',
                )
                if bypass_outcome is not None:
                    # Bypass done detected — helper already reopened the row and
                    # filed an L1 with category='bypass_done'.
                    return bypass_outcome

                # Legitimate done: provenance commit is reachable from main.
                # The row is already terminal-done out-of-band (e.g. a human
                # marked done during the ~3s acquire→dispatch window). Accept it.
                logger.info(
                    'Task %s: already legitimately done at dispatch (old_status=%r) '
                    '— accepting, no reopen, no escalation',
                    self.task_id, exc.old_status,
                )
                self._enter_phase(WorkflowState.DONE)
                return WorkflowOutcome.DONE

            # A persistence-layer rejection escaped one of the workflow's
            # set_task_status / mark_done call sites without an explicit
            # handler.  Route to L1: the row state contradicts what the
            # workflow tried to write, which a steward retry can't unstick.
            # Caught BEFORE the broad Exception so it gets dedicated framing.
            logger.error(
                'Task %s: unhandled set_task_status rejection — %s: %s',
                self.task_id, exc.error_code, exc.raw,
            )
            return await self._mark_blocked(
                f'Unhandled set_task_status rejection: {exc.error_code} — {exc.raw}',
                escalate_to_human=True,
            )

        except WarmLaneRequeue as e:
            # W9-ε: block_reason is now single-sourced from
            # classify_failure(e) -> BlockDisposition instead of this
            # clause's own inline isinstance(e, WarmLanePoolHardDown)/
            # ...Exhausted/... triage — MRO resolution means a real subclass
            # instance always matches its OWN _DISPOSITION_TABLE row first
            # (e.g. WarmLanePoolHardDown before the WarmLaneRequeue base
            # row), so this reproduces the old per-subclass strings exactly
            # — including for a bare WarmLaneRequeue (no subclass matches):
            # the base row is deliberately ALIASED to WarmLaneDiskPressure's
            # disposition, mirroring the pre-refactor `else: #
            # WarmLaneDiskPressure` fallback exactly (amendment,
            # reviewer_comprehensive behavior-parity — see the base row's
            # comment in workflow_types._disposition_table()).
            # FAULT (RuntimeError) is deliberately NOT caught here — it falls
            # through to the broad except below → _mark_blocked → BLOCKED + L1.
            #
            # counts_against_requeue_cap is DECLARED once per warm-lane
            # subclass in the table (EXHAUSTED=False as of task 2988;
            # DISK_PRESSURE/HARD_DOWN/SOFT_PRESSURE=False — transient infra /
            # capacity signals; RESEED_CONTAMINATED=True — per-task
            # data-integrity) — the single source of truth replacing the
            # buried NOTE formerly here (see workflow_types._disposition_
            # table()'s warm-lane rows for the full rationale). Task 2988
            # (PRD ε / W3) CONSUMES this flag: it is threaded onto the
            # TerminalReport below -> TaskReport -> Scheduler.record_requeue
            # (counts_against_cap=), so a non-counting warm-lane requeue no
            # longer burns the per-task requeue cap. The previously-deferred
            # "scheduler.py / harness.py" follow-up this comment named IS
            # task 2988. This clause still unconditionally returns REQUEUED.
            disp = classify_failure(e)
            block_reason = disp.reason_prefix
            logger.info(
                'Task %s: warm-lane requeue (%s, counts_against_requeue_cap=%s): %s',
                self.task_id, block_reason, disp.counts_against_requeue_cap, e,
            )
            # TerminalReport.phase is machine.state — this path never calls
            # _enter_phase, so it is the pre-existing working phase (PLAN,
            # since create_worktree runs before the first _enter_phase call
            # in run()).  blocked_from_phase mirrors it (no BLOCKED transition
            # here, so "pre-block" and "current" are the same phase) — keeps
            # the harness's retry-cap block_phase at 'plan' (REVIEW-CYCLE-1).
            self._terminal_report = TerminalReport(
                outcome=WorkflowOutcome.REQUEUED, reason=block_reason,
                phase=self.machine.state, detail=str(e), category=None,
                blocked_from_phase=self.machine.state,
                counts_against_requeue_cap=disp.counts_against_requeue_cap,
            )
            return WorkflowOutcome.REQUEUED

        except Exception as e:
            # W9-ε: every remaining block-kind failure (cap/budget/verify-
            # infra/OSError/worktree-conflict/generic) is now classified by
            # ONE classify_failure(e) -> BlockDisposition TABLE lookup
            # instead of a per-exception-type except clause.  Each branch
            # below still assembles its own reason/detail text — the table
            # doesn't carry instance-specific data (retries/elapsed/label,
            # phase/errno, cumulative_cost, ...) — but escalate_to_human and
            # the BlockDisposition threaded into _mark_blocked are now
            # single-sourced from disp.  Every disp.reason_prefix below is
            # the SAME leading text the pre-W9-ε ladder hard-coded per
            # exception, so the emitted reason strings stay byte-identical
            # (behavior-preserving).  Branch order mirrors the pre-W9-ε
            # ladder's except-clause order (most-specific first).
            disp = classify_failure(e)

            if isinstance(e, AllAccountsCappedException):
                logger.warning(
                    f'Task {self.task_id}: all accounts capped — '
                    f'{e.retries} retries in {e.elapsed_secs:.1f}s (label={e.label!r})'
                )
                return await self._mark_blocked(
                    f'{disp.reason_prefix}: {e.label} — {e.retries} retries in {e.elapsed_secs:.1f}s',
                    suggested_action='cap_wait_exceeded_sanity_bound',
                    escalate_to_human=disp.escalate_to_human,
                    disposition=disp,
                )

            if isinstance(e, _SessionBudgetExhausted):
                last_role = self._last_completed_role or 'n/a'
                budget_limit = self.config.usage_cap.session_budget_usd
                # Use the gate's own cumulative figure for the summary — it is
                # the value that actually exceeded the budget, whereas
                # self.metrics.total_cost_usd only advances on successful
                # returns and may lag the gate's running tally if a cap-retry
                # or partial invocation contributed cost without completing.
                reason = (
                    f'{disp.reason_prefix}: ${e.cumulative_cost:.2f} spent of '
                    f'${budget_limit:.2f} budget (last completed role: {last_role})'
                )
                detail = (
                    f'budget_limit=${budget_limit:.2f}\n'
                    f'total_cost_usd=${self.metrics.total_cost_usd:.2f}\n'
                    f'cumulative_cost (gate)=${e.cumulative_cost:.2f}\n'
                    f'agent_invocations={self.metrics.agent_invocations}\n'
                    f'total_turns={self.metrics.total_turns}\n'
                    f'last_completed_role={last_role}'
                )
                # _mark_blocked logs "Task %s BLOCKED: %s" — only log the
                # gate-specific cross-check figure that's unique to this call
                # site.
                logger.info(
                    'Task %s: session budget exhausted (gate cumulative $%.2f)',
                    self.task_id, e.cumulative_cost,
                )
                return await self._mark_blocked(
                    reason, detail=detail,
                    escalate_to_human=disp.escalate_to_human,
                    disposition=disp,
                )

            if isinstance(e, VerifyInfraError):
                # Infra-typed exception from the verify path that escaped the
                # in-process retry wrapper (e.g. raised outside
                # _run_scoped_verification_with_infra_retry).  Route to
                # infra_issue with escalate_to_human rather than the generic
                # 'Workflow error:' task_failure block below.
                logger.warning(
                    'Task %s: VerifyInfraError escaped run() (phase=%r errno=%r) '
                    '— routing to infra_issue',
                    self.task_id, e.phase, e.errno,
                )
                reason = f'{disp.reason_prefix} (phase={e.phase!r} errno={e.errno}): {e}'
                return await self._mark_blocked(
                    reason,
                    category='infra_issue',
                    escalate_to_human=disp.escalate_to_human,
                    disposition=disp,
                )

            if isinstance(e, OSError) and _is_infra_oserror(e):
                # Bare infra-class OSError (ENOSPC/EDQUOT/EROFS/EIO/EMFILE/
                # ENFILE) escaping the verify path outside the
                # VerifyInfraError wrapper (e.g. a log or marker write).
                # Route to infra_issue.  Non-infra OSErrors (EACCES, ENOENT,
                # etc.) fall through to the generic branch below — same as
                # every other unclassified exception.
                logger.warning(
                    'Task %s: bare infra OSError escaped run() (errno=%r) '
                    '— routing to infra_issue',
                    self.task_id, e.errno,
                )
                reason = f'{disp.reason_prefix} (errno={e.errno}): {e}'
                return await self._mark_blocked(
                    reason,
                    category='infra_issue',
                    escalate_to_human=disp.escalate_to_human,
                    disposition=disp,
                )

            if isinstance(e, WorktreeConflictError):
                # esc-2128-8: a WIP-save commit() (inter-iteration rebase, or the
                # requeue-rebase reuse path) hit a worktree with unresolved
                # (unmerged-index) conflicts and refused to stage/commit rather
                # than snapshotting conflict markers verbatim.  Route to a
                # targeted per-task BLOCKED + human L1 — the steward corrective
                # loop cannot resolve a conflicted worktree, so skip it entirely
                # (mirrors the VerifyInfraError/infra_issue handling above, not
                # _submit_halt_escalation_and_wait, which is merge-queue-halt
                # ownership and irrelevant to a task-worktree rebase).
                logger.warning(
                    'Task %s: WorktreeConflictError escaped run() — %s',
                    self.task_id, e,
                )
                return await self._mark_blocked(
                    f'{disp.reason_prefix} ({e})',
                    category='wip_conflict',
                    escalate_to_human=disp.escalate_to_human,
                    disposition=disp,
                )

            if isinstance(e, BranchResetError):
                # task 2403: GitOps.rebase_preserving_task_commits detected
                # that a requeue/inter-iteration rebase collapsed this
                # branch to zero commits ahead of its rebase baseline — the
                # guard already attempted to restore the pre-rebase HEAD
                # (best-effort; see BranchResetError.restore_ok — a failed
                # restore is called out explicitly in str(e) below rather
                # than silently assumed safe), but the wipe condition
                # itself needs a human to look at the merge-train state
                # that produced it. Route to a targeted per-task BLOCKED +
                # human L1, bypassing the steward corrective loop (mirrors
                # the WorktreeConflictError branch above) rather than
                # falling through to the generic 'Workflow error:' handler.
                # escalate_to_human is passed explicitly (True) to keep this
                # branch self-documenting; it now matches disp.escalate_to_human
                # since BranchResetError has a dedicated _DISPOSITION_TABLE row
                # (workflow_types.py) whose reason_prefix supplies the message.
                logger.warning(
                    'Task %s: BranchResetError escaped run() — %s',
                    self.task_id, e,
                )
                return await self._mark_blocked(
                    f'{disp.reason_prefix} ({e})',
                    category='branch_reset',
                    escalate_to_human=True,
                    disposition=disp,
                )

            # Generic fallback — every other exception, including a
            # non-infra OSError (Python's exception model has no re-raise-
            # into-a-sibling-clause, and classify_failure's OSError branch
            # already returns the same _DEFAULT_BLOCK disposition for it as
            # for a bare Exception, so one shared branch is correct here).
            logger.exception(f'Task {self.task_id} workflow error: {e}')
            return await self._mark_blocked(
                f'{disp.reason_prefix}: {e}',
                escalate_to_human=disp.escalate_to_human,
                disposition=disp,
            )

    def _on_terminal_cleanups(self) -> list[OnTerminalEntry]:
        """Ordered, kind-aware terminal-cleanup list run by ``CancellationScope``
        on every exit from ``run()`` (normal return, soft-cancel, or
        hard-cancel — ``kind`` is ``None`` for a normal exit).

        Replaces ``_drive()``'s old ``finally`` block (W9-θ): the same five
        cleanups, in the same order (``stop_claimant_heartbeat`` first —
        task 2188, PRD task-status-authority C4/D4 — so the heartbeat loop
        can never race the harness's post-release claimant clear with a
        stray refresh), relocated here so any long ``await`` anywhere inside
        the scope is cancellable by construction, without needing its own
        opt-in.

        The ``release_lane`` entry is the kind-aware 1:1 replacement for the
        deleted ``sys.exc_info()`` ``_hard_cancel`` sniff (which used to
        coordinate with harness.py's now-also-deleted B2 dual-guard via a
        "both must fire" comment contract): it releases the warm lane for
        every terminal exit EXCEPT a hard-cancel, which must leave the
        branch/lane assignment untouched so a forcibly-torn-down workflow's
        branch survives teardown — the harness R3 grace window + reconcile
        sweep own reclaim from there, not this teardown.
        ``kind == 'soft'`` releases even from a still-working state
        (SOFT_CANCELLED can exit mid-phase); ``kind is None`` only releases
        once the state has genuinely reached DONE/CANCELLED (mirrors the
        pre-θ non-hard-cancel behaviour). Uses the default
        ``allow_disk_backstop=False``: if ``_maybe_cleanup_done_worktree``
        already released the lane and dropped the in-memory assignment
        (T1/T2 sync-merge path), ``assignment_for`` returns ``None`` →
        ``release_lane_for_terminal_task`` is a true no-op (no disk scan, no
        redundant ``cleanup_worktree``/``git branch -D`` retry).
        """
        async def _stop_heartbeat(_kind: Literal['hard', 'soft'] | None) -> None:
            await self._stop_claimant_heartbeat()

        async def _stop_steward(_kind: Literal['hard', 'soft'] | None) -> None:
            if self._steward:
                await self._steward.stop()

        async def _cleanup_worktree(_kind: Literal['hard', 'soft'] | None) -> None:
            # Cleanup worktree (only if done AND branch is on main — preserve
            # otherwise so an agent's update_task(status='done') bypass
            # doesn't GC unmerged work). Skips externally-managed worktrees
            # (eval mode).
            await self._maybe_cleanup_done_worktree()

        async def _release_lane(kind: Literal['hard', 'soft'] | None) -> None:
            if (
                kind != 'hard'
                and not self._worktree_external
                and (
                    kind is not None
                    or self.state in (WorkflowState.DONE, WorkflowState.CANCELLED)
                )
            ):
                await self.git_ops.release_lane_for_terminal_task(self.task_id)

        async def _cleanup_config(_kind: Literal['hard', 'soft'] | None) -> None:
            # Preserve-aware — skips when circuit breaker tripped so the dir
            # is available for forensic analysis.
            self._cleanup_config_dir()

        return [
            ('stop_claimant_heartbeat', _stop_heartbeat),
            ('stop_steward', _stop_steward),
            ('cleanup_done_worktree', _cleanup_worktree),
            ('release_lane', _release_lane),
            ('cleanup_config_dir', _cleanup_config),
        ]

    async def _finalise_cancellation(self, kind: Literal['hard', 'soft']) -> WorkflowOutcome:
        """Resolve a caught :class:`WorkflowCancelled` into a ``WorkflowOutcome``.

        Drives the machine to ``CANCELLED`` (guarded by ``is_terminal()`` —
        a no-op if some other path already left it terminal; mirrors the
        existing ``_handle_cancelled_terminal_exit`` CANCELLED transition).
        ``kind == 'hard'`` maps straight to ``WorkflowOutcome.CANCELLED``.
        ``kind == 'soft'`` delegates to the existing ``_handle_soft_cancel``
        status-derived decision (DONE for a terminal-done scheduler row,
        else SOFT_CANCELLED) — folded in here rather than duplicated.
        """
        working_phase = self.machine.state.value
        if not self.machine.is_terminal():
            self._enter_phase(WorkflowState.CANCELLED)
        if kind == 'hard':
            return WorkflowOutcome.CANCELLED
        return await self._handle_soft_cancel(working_phase)

    async def run(self) -> TerminalReport:
        """Execute the full state machine and return the terminal contract.

        TR-1: the workflow↔harness terminal contract is this RETURN value,
        not the (now-deleted) ``_last_block_*`` side channel. ``_drive()``
        carries the actual control flow; this wrapper turns its
        ``WorkflowOutcome`` into the ``TerminalReport`` that ``_mark_blocked``
        (or a non-``_mark_blocked`` block path) already stashed at
        ``self._terminal_report`` for BLOCKED/REQUEUED/ESCALATED exits. A
        clean exit (DONE/CANCELLED, no block ever hit) has no stashed report,
        so one is synthesized here from ``machine.state`` with empty
        reason/detail and ``category=None``.

        SM-2: before returning, assert the report is internally consistent.
        ``report.phase`` must always equal the live state-machine state (it
        is built from ``machine.state`` at construction, so a mismatch here
        means a report was constructed from a stale/foreign source). When
        the authoritative last-persisted status is legible, ``report.outcome``
        must also be an allowed pairing with it (``outcome_allows_status``) —
        this is what catches the false-done class (DB row says 'done' while
        the actual outcome is BLOCKED). A ``None`` or out-of-vocabulary
        status row (transient ``get_status`` failure, or a status outside
        the closed vocabulary) is treated as fail-safe-wait rather than a
        mismatch, so a flaky read never crashes ``run()``.

        The outcome<->status half is skipped entirely in eval mode
        (``self._worktree_external``): the MERGE phase — and with it the
        only call that ever persists a terminal 'done' row — is
        unconditionally skipped for external worktrees (see the ``MERGE
        (skip for eval mode)`` guard above), so an eval-mode DONE exit
        legitimately leaves the last persisted status wherever the pre-empt
        claim left it. Every other terminal-bookkeeping step in this class
        (lane release, DONE-cleanup gate) already carries this same
        ``not self._worktree_external`` guard for the identical reason —
        eval mode's task row is not the authoritative record real dispatch
        relies on.

        CancellationScope (CX-1, W9-θ): ``_drive()`` runs under a
        :class:`CancellationScope` supervising both the harness's hard
        ``task.cancel()`` and the soft ``_cancel_event`` as ONE typed
        :class:`WorkflowCancelled`, caught at this EXACTLY ONE site.  The
        outcome<->status half of SM-2 is additionally skipped on a
        hard-cancel exit (``cancel_kind == 'hard'``): ``_enter_phase``
        never persists status, so a forcibly-torn-down workflow leaves the
        live row at whatever it was (typically 'in-progress') — it does not
        own its terminal row; the harness R3 grace window + reconcile sweep
        do. The phase-consistency half always still holds.
        """
        cancel_kind: Literal['hard', 'soft'] | None = None
        scope = CancellationScope(self._cancel_event, self._on_terminal_cleanups())
        try:
            outcome = await scope.supervise(self._drive())
        except WorkflowCancelled as wc:
            cancel_kind = wc.kind
            outcome = await self._finalise_cancellation(wc.kind)
        report = (
            self._terminal_report
            if (
                self._terminal_report is not None
                and self._terminal_report.outcome == outcome
            )
            else TerminalReport(
                outcome=outcome, reason='', phase=self.machine.state,
                detail='', category=None,
            )
        )
        assert report.phase == self.machine.state, (
            f'run()-exit SM-2: report.phase {report.phase!r} != '
            f'machine.state {self.machine.state!r} (task {self.task_id})'
        )
        if not self._worktree_external and cancel_kind != 'hard':
            try:
                last_status_row = await self.scheduler.get_status(self.task_id)
            except Exception:
                # get_status's own contract is str | None (it never raises —
                # see Scheduler.get_status), so this only fires against a
                # non-conforming test double.  Same fail-safe fallback as the
                # worktree-missing-fallback get_status call above.
                logger.exception(
                    'Task %s: get_status failed during run()-exit SM-2 check; '
                    'skipping the outcome<->status consistency check',
                    self.task_id,
                )
                last_status_row = None
            if last_status_row is not None:
                try:
                    status_consistent = outcome_allows_status(report.outcome, last_status_row)
                except ValueError:
                    # Out-of-vocabulary/unreadable status row — fail-safe:
                    # skip the check rather than crash run() on a transient
                    # or garbled read (mirrors the pre-empt check's None
                    # handling).
                    status_consistent = True
                if not status_consistent:
                    raise AssertionError(
                        f'run()-exit SM-2: outcome {report.outcome!r} inconsistent '
                        f'with status {last_status_row!r} (task {self.task_id})'
                    )
        return report

    async def _merge_and_finalise(self, branch_name: str) -> WorkflowOutcome:
        """Run the MERGE phase and, on success, the SUCCESS/finalise tail.

        The extracted ``_drive`` MERGE+SUCCESS tail (task 2795), shared by the
        normal pipeline tail and :meth:`_resume_merge_retry_if_pending`.
        :meth:`_run_merge_phase` returns ``None`` on merge SUCCESS (signalling
        the caller to finalise) or a terminal :class:`WorkflowOutcome` on a
        block/requeue. This helper maps a ``None`` result to the finalise-DONE
        path and returns a terminal outcome verbatim, so it ALWAYS returns a
        :class:`WorkflowOutcome` — letting the resume guard call one method that
        can never return ``None`` (a ``None`` would make ``_drive`` fall through
        to PLAN). Eval mode (``self._worktree_external``) skips the merge into
        main but still runs the finalise tail, exactly as the inline tail did.
        """
        # MERGE (skip for eval mode — no merge into main)
        if not self._worktree_external:
            _merge_result = await self._run_merge_phase(branch_name)
            if _merge_result is not None:
                return _merge_result

        # Merge SUCCEEDED — discharge any durable merge-retry obligation. On the
        # normal resolve→in-place-retry→success path _requeue's stamp is otherwise
        # never cleared (the resume guard's clear fires only on a restart
        # re-dispatch), so a DONE task would keep a stale merge_retry_pending.
        # Clearing here keeps the stamp/clear lifecycle symmetric; it is an
        # idempotent no-op when nothing was ever stamped (the common case).
        await self._clear_merge_retry_pending()
        # Task 2991: likewise discharge the durable merge-phase-liveness stamp.
        # The enqueue-boundary clear in _submit_to_merge_queue covers the normal
        # path, but ghost-loop / eval-mode successes reach DONE without passing
        # through it — so clear here too, keeping a DONE task from carrying a
        # fresh stamp that would briefly defer an unrelated stranded divergence
        # orphan in the reaper. Idempotent no-op when unstamped.
        await self._clear_merge_phase_entered()

        # SUCCESS — write completion knowledge (best-effort after merge)
        try:
            await self._write_completion_to_memory()
        except Exception as e:
            logger.warning(
                f'Task {self.task_id}: completion memory write failed '
                f'(non-fatal): {e}'
            )
        # Wait for steward to finish any pending work (suggestion triage, etc.)
        await self._ensure_steward_started()
        await self._await_steward_completion(skip_if_idle=True)
        self._enter_phase(WorkflowState.DONE)
        return await self._finalise_merged_done()

    async def _resume_merge_retry_if_pending(
        self, branch_name: str,
    ) -> WorkflowOutcome | None:
        """Fast-path back to the merge phase when a durable retry obligation matches.

        Reads ``metadata.merge_retry_pending`` (stamped by :meth:`_requeue`'s
        merge_phase path when a merge-phase escalation was resolved via
        ``resume``). If it is a dict AND the current post-rebase worktree HEAD
        equals the stamped ``branch_head``, the resolved-and-verified branch is
        unchanged since resolution — so clear the stamp and jump STRAIGHT to the
        merge phase (via :meth:`_merge_and_finalise`), skipping
        plan/execute/verify/review (zero reviewer_comprehensive; ``merge_queued``
        is the next pipeline event, matching the observable signal).

        A HEAD match alone is NOT sufficient (task 3024): it proves only that
        the branch has not MOVED, not that it still MERGES. Main advancing
        underneath an unchanged branch can introduce a rebase conflict, and the
        fast-path would then hand an empty ``plan.files`` workflow to the merge
        phase — tripping the merge-entry scope invariant on every dispatch, in a
        loop no escalation action could break. So the resume preconditions also
        require that the branch STILL cleanly merges onto current main, probed
        object-store-only via :meth:`GitOps.merge_tree_conflicts`.

        Fail-safe fall-through to the full pipeline (returns ``None``) on a
        missing/non-dict stamp, a rev-parse failure, a HEAD mismatch, a
        confirmed conflict, or a failed conflict probe. Whether it also CLEARS
        the stamp turns on whether the obligation is provably void:

        * A HEAD mismatch clears — main advanced and the branch was rebased to
          new SHAs, so it can never match again, and resubmitting
          possibly-divergent code is exactly what we must avoid.
        * A confirmed conflict clears — that obligation can never be satisfied
          as stamped, and the full pipeline gives an agent the chance to rebase
          and resolve (or fail as an ordinary task_failure).
        * A rev-parse or probe FAILURE does NOT clear — an error is a
          non-verdict, so the durable obligation is preserved for a later
          dispatch rather than silently discarded on a transient fault.

        Clear-on-consume is idempotent and loop-safe: if the resumed merge
        re-blocks and the steward resolves again, ``_requeue`` re-stamps a fresh
        obligation.
        """
        stamp = (self.task.get('metadata') or {}).get('merge_retry_pending')
        if not isinstance(stamp, dict):
            return None
        try:
            rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=self.worktree)
        except Exception as exc:  # noqa: BLE001 — fail-safe: fall through to full pipeline
            logger.warning(
                'Task %s: could not read worktree HEAD resolving '
                'merge_retry_pending (%s) — running full pipeline',
                self.task_id, exc,
            )
            return None
        current_head = out.strip()
        if rc != 0 or not current_head:
            logger.warning(
                'Task %s: git rev-parse HEAD failed (rc=%s) resolving '
                'merge_retry_pending — running full pipeline',
                self.task_id, rc,
            )
            return None
        stamped_head = stamp.get('branch_head')
        if current_head != stamped_head:
            logger.warning(
                'Task %s: merge_retry_pending branch_head %s != current HEAD %s '
                '(main advanced / branch rebased) — clearing stale stamp, '
                'running full pipeline', self.task_id, stamped_head, current_head,
            )
            await self._clear_merge_retry_pending()
            return None
        # The branch has not MOVED — but has it stopped MERGING?  Main may have
        # advanced underneath it since the steward resolved, introducing a
        # rebase conflict.  Resuming straight to merge with an empty plan.files
        # would then trip the merge-entry scope invariant on every dispatch
        # (task 3024), so probe the merge before honouring the fast-path.
        try:
            main_sha = await self.git_ops.get_main_sha()
            probe = await self.git_ops.merge_tree_conflicts(main_sha, current_head)
        except Exception as exc:  # noqa: BLE001 — fail-safe: fall through to full pipeline
            # A probe FAILURE is not a conflict VERDICT: it says nothing about
            # whether the branch still merges.  So fall back to the full
            # pipeline for this dispatch but leave the durable stamp intact for
            # a later one — the opposite of the confirmed-conflict arm below.
            logger.warning(
                'Task %s: could not verify branch still cleanly merges resolving '
                'merge_retry_pending (%s) — running full pipeline',
                self.task_id, exc,
            )
            return None
        if not probe.clean:
            logger.warning(
                'Task %s: merge_retry_pending branch %s no longer merges cleanly '
                'onto current main %s (conflicts: %s) — resume preconditions '
                'void, clearing stamp and running full pipeline',
                self.task_id, current_head, main_sha, probe.conflicted_paths,
            )
            await self._clear_merge_retry_pending()
            return None
        logger.info(
            'Task %s: merge_retry_pending HEAD match (%s) — resuming straight to '
            'merge phase, skipping plan/execute/verify/review',
            self.task_id, current_head,
        )
        await self._clear_merge_retry_pending()
        return await self._merge_and_finalise(branch_name)

    async def _run_merge_phase(self, branch_name: str) -> WorkflowOutcome | None:
        """Execute the MERGE phase. Returns None to fall through to SUCCESS."""
        assert self.worktree is not None  # guaranteed by _setup_worktree_and_artifacts
        # Train members hold at merge-deferred instead of merging;
        # the group-merge worker (δ₁) owns the eventual done
        # transition once all siblings are workspace-green (PRD §9.5,
        # γ₁).  The full execute→verify→review pipeline still ran
        # above (PRD acceptance criterion 5).  Non-train path is
        # byte-identical — this guard only fires when metadata.train
        # is a dict.
        _defer = await self._maybe_defer_as_train_member()
        if _defer is not None:
            return _defer

        self._enter_phase(WorkflowState.MERGE)

        # task 2991: write the durable, restart-survivable merge-phase liveness
        # stamp immediately on merge entry — BEFORE _check_scope_invariant files
        # its plan.files/metadata.files divergence L0 and BEFORE the gating bail
        # below. The orphan-L0 divergence reaper's _has_fresh_merge_phase gate
        # reads it to DEFER (not promote) a live merge-stage task, which the
        # pre-enqueue loop (no LLM calls) would otherwise fail the task-2931
        # routing.latest freshness gate. Placed here so it is refreshed on every
        # (re-)dispatch into merge phase, including passes that immediately bail
        # to ESCALATED — exactly the passes a live-but-wedged task cycles
        # through. (Distinct from the in-memory note_merge_phase_entered at the
        # retry-loop top, which is the self-redeploy coordinator's signal.)
        _entry_metadata = await self._stamp_merge_phase_entered()

        # Tripwire (task 2505): plan.files must equal metadata.files by
        # construction (the scope-reconciliation choke point keeps them in
        # lockstep) — a divergence here means some path bypassed it. Purely
        # observational: logs + escalates, never blocks the merge.
        # The stamp above already read this task's backend metadata blob;
        # thread it in rather than issuing a second identical get_task on the
        # merge hot path (review amendment). A None/unreadable prefetch falls
        # back to _check_scope_invariant's own read, so its fail-safe is
        # unchanged.
        await self._check_scope_invariant(backend_metadata=_entry_metadata)

        # Defense-in-depth: any blocking L0 escalation, or any
        # born-at-L2 (critical/urgent), or any level≥2 escalation
        # created during execute/verify/review (e.g. plan-overwrite
        # from _amend, or any future code path that escalates outside
        # the implementer/debugger callsites) must gate the merge.
        # Without this, an escalation queued mid-run would sit
        # invisible while a merge proceeded.
        # D4 accepted consequence: a pending critical/urgent from a
        # prior incarnation DOES gate a fresh run — stop-the-line
        # semantics.  See _is_gating_escalation for the full policy.
        gating = [e for e in self._check_escalations() if _is_gating_escalation(e)]
        if gating:
            logger.warning(
                'Task %s: %d gating escalation(s) at MERGE entry '
                '— bailing to ESCALATED',
                self.task_id, len(gating),
            )
            return WorkflowOutcome.ESCALATED

        # Ghost-loop early exit (PRD workflow-state-machine α, MP-1/MP-2): a
        # MergeProvenance hit or the _has_prior_implementation fallback both
        # route through _recover_before_merge() and the shared
        # _finalise_recovery_done chokepoint — the single source the three
        # already-merged guards now share.
        _, branch_head, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
        )
        main_sha = await self.git_ops.get_main_sha()
        _r = await self._recover_before_merge(branch_head.strip(), main_sha)
        if _r is not None:
            return _r

        for _merge_attempt in range(self.config.max_merge_retries):
            # task 2753: stamp the pre-enqueue MERGE-phase grace on the
            # Scheduler at the TOP of the retry loop — this is the precise
            # pre-enqueue window (Phase-1 rebase + scoped re-verify + Phase-2
            # submit, before register_and_enqueue_merge_request emits
            # merge_queued). Re-stamps on each REQUEUED retry (each re-runs a
            # vulnerable pre-merge verify). The orchestrator self-redeploy
            # coordinator reads this so a polite/force-fire restart does not
            # cancel the workflow before it reaches the durable merge journal.
            # Cleared at the durable-enqueue boundary in _submit_to_merge_queue
            # and defensively in the harness _run_slot finally.
            self.scheduler.note_merge_phase_entered(self.task_id)
            # task 2991 (review-fix R2-B): refresh the DURABLE liveness stamp
            # alongside the in-memory one above, for the same reason. Each
            # REQUEUED retry re-runs a full vulnerable pre-enqueue window
            # (Phase-1 rebase + scoped re-verify, minutes, zero LLM calls)
            # AFTER _submit_to_merge_queue discharged the stamp at the enqueue
            # boundary. Without this re-stamp that window has neither a durable
            # merge_phase_liveness nor a fresh routing.latest, so the task
            # 2931/2991 divergence false positive recurs on every merge retry
            # (a REQUEUED retry goes through a steward resolution first, so the
            # merge-entry L0 is already older than orphan_l0_timeout_secs when
            # attempt 2 begins). The merge-entry stamp above is KEPT — it must
            # precede _check_scope_invariant and the gating bail, both of which
            # sit above this loop — so attempt 1 writes twice: an accepted cost
            # (two cheap best-effort metadata writes, bounded by
            # max_merge_retries+1 per merge entry) for the simple invariant "a
            # fresh durable stamp exists at the start of every pre-enqueue
            # window", which an `if _merge_attempt > 0` would make dependent on
            # loop-index reasoning.
            await self._stamp_merge_phase_entered()
            # Phase 1: pre-merge rebase (no lock, no queue slot)
            # Rebase the task branch onto current main and re-verify
            # so the queued merge phase is fast/trivial.
            pre_rebased = False
            for _attempt in range(self.config.max_pre_merge_retries):
                main_before = await self.git_ops.get_main_sha()
                if not await self.git_ops.rebase_onto_main(self.worktree):
                    break  # true conflict — queue will detect it
                # Warm-lane consumer-hold (task 3027): same per-lane
                # <lane_dir>.lock hold as the implement-phase verify — this
                # post-rebase re-verify also builds/runs test binaries on the
                # task's warm lane, so it equally races reify's warm-lane-gc.sh
                # reclaim; the flock is the cross-process guard that script
                # honors. Fails open on contention.
                async with self.git_ops.task_verify_lease(self.worktree):
                    verify = await run_scoped_verification(
                        self.worktree, self.config, self._module_configs,
                        task_files=self._task_files,
                        # role='task' is explicit for γ's explicit-is-correct
                        # invariant (mirrors merge_queue.py role='merge').
                        # 'task' is already the default so this is documentary.
                        # This site's task_verify_lease wrap has its OWN
                        # held-flock call-site spy: test_task_verify_lease.py::
                        # TestMergePhaseReverifyHoldsLease drives _run_merge_phase
                        # to here and asserts the lane flock is HELD during the
                        # re-verify (the implement-phase site is covered
                        # separately by TestTaskLaneVerifyHoldsLease).
                        role='task',
                    )
                if not verify.passed:
                    if verify.timed_out:
                        logger.warning(
                            f'Task {self.task_id}: post-rebase verification '
                            f'timed out; merge queue will retry'
                        )
                    else:
                        logger.warning(
                            f'Task {self.task_id}: post-rebase verification '
                            f'failed: {verify.summary}'
                        )
                        if self.event_store:
                            self.event_store.emit(
                                EventType.waste_detected,
                                task_id=self.task_id, phase='merge',
                                data={
                                    'waste_type': 'post_rebase_verify_fail',
                                    'summary': verify.summary[:200],
                                },
                            )
                    break
                main_after = await self.git_ops.get_main_sha()
                if main_before == main_after:
                    pre_rebased = True
                    self.metrics.pre_merge_rebase_ok += 1
                    break
                self.metrics.pre_merge_rebase_attempts += 1
                logger.info(
                    f'Task {self.task_id}: main moved during pre-merge '
                    f'rebase, retrying'
                )

            # Phase 2: submit to merge queue (replaces _merge_lock)
            self._last_merge_block_reason = None
            self._last_merge_failure_category = ''
            self._last_merge_failure_cause_hint = ''
            merge_outcome = await self._submit_to_merge_queue(
                branch_name, pre_rebased=pre_rebased,
                merge_phase=True,
            )
            if merge_outcome == WorkflowOutcome.DONE:
                break
            if merge_outcome != WorkflowOutcome.REQUEUED:
                # BLOCKED / ESCALATED — exit slot. A soft-cancel inside
                # _submit_to_merge_queue no longer surfaces as a
                # merge_outcome value (W9-θ): _await_cancellable raises
                # WorkflowCancelled('soft') straight through this call,
                # so it never reaches this comparison.
                return merge_outcome

            # Defense-in-depth (root cause #2): _cancel_event is
            # never cleared during a run, so each retry iteration
            # would re-win the cancellable race instantly and burn
            # another pre-merge rebase+verify before exhausting
            # max_merge_retries. Checking here — immediately after
            # the REQUEUED guard and BEFORE the anti-thrash/retry
            # path — ensures a soft-cancel that arrived concurrently
            # with a legitimate steward-resolved REQUEUED exits on
            # first detection without any further rebase or log.
            # W9-θ: raise (not return _handle_soft_cancel(...) inline) so
            # it propagates to run()'s single WorkflowCancelled catch.
            if self._cancel_event.is_set():
                raise WorkflowCancelled('soft')

            # Fix 3 — anti-thrash guard for repeated
            # steward-resolved merge-phase loops on the same
            # outcome signature.  At threshold escalates to L1
            # rather than resubmitting the same merge.
            if self._last_merge_block_reason is not None:
                current_signature = self._merge_outcome_signature()
                prev_signature = (
                    (self.task.get('metadata') or {}).get('retry_ledger') or {}
                ).get('last_merge_outcome_signature')
                thrash_outcome = await self._check_merge_outcome_thrash(
                    prev_signature, current_signature,
                )
                if thrash_outcome is not None:
                    return thrash_outcome

            # Steward resolved — check if branch landed on main
            _, bh, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
            )
            main_sha = await self.git_ops.get_main_sha()
            if await self.git_ops.is_ancestor(bh.strip(), main_sha):
                logger.info(
                    'Task %s: branch on main after steward '
                    'resolution', self.task_id,
                )
                break
            # Retry merge
            logger.info(
                'Task %s: retrying merge (attempt %d/%d)',
                self.task_id, _merge_attempt + 1,
                self.config.max_merge_retries,
            )
        else:
            return await self._mark_blocked(
                'Merge retries exhausted after steward resolutions'
            )
        return None

    def _resolve_module_configs(self, modules: list[str] | None = None) -> list[ModuleConfig]:
        """Collect ModuleConfigs for this task's modules.

        Groups modules by subproject prefix and returns one ModuleConfig per
        subproject that has an ``orchestrator.yaml``.  Warns for subprojects
        without configs.  Returns an empty list when no modules are assigned
        (triggers global fallback in ``run_scoped_verification``).

        *modules* overrides ``self.modules`` when provided; pass the union module
        list from ``_maybe_enqueue_group_merge`` to build union ``module_configs``
        without duplicating the ``for_module`` / dedupe-by-prefix logic.
        """
        mods = modules if modules is not None else self.modules
        if not mods:
            return []
        seen: dict[str, ModuleConfig] = {}
        missing: set[str] = set()
        for m in mods:
            mc = self.config.for_module(m)
            if mc:
                seen[mc.prefix] = mc
            else:
                prefix = m.strip('/').split('/')[0]
                missing.add(prefix)
        if missing:
            logger.warning(
                'Task %s: subprojects without orchestrator.yaml: %s — '
                'these will use global verification config',
                self.task_id, missing,
            )
        return list(seen.values())

    def _maybe_warn_missing_escalation(self, role_name: str) -> None:
        """Emit a single WARNING when an escalation-capable role is invoked without a queue."""
        if self._escalation_missing_warned:
            return
        if self.escalation_queue is not None:
            return
        if role_name not in _ESCALATION_CAPABLE_ROLES:
            return
        logger.warning(
            'Task %s: escalation_queue is unavailable — agent role %r would normally'
            ' have escalation tools wired',
            self.task_id, role_name,
        )
        self._escalation_missing_warned = True

    async def _sync_worktree_venvs(self) -> None:
        """Run ``uv sync`` for task subprojects in the worktree.

        Creates per-worktree venvs so Python imports resolve from the
        worktree's source code rather than the main tree's editable installs.
        Local ``[tool.uv.sources]`` dependencies (e.g. ``shared``) are
        pulled in automatically via relative editable paths.
        """
        assert self.worktree is not None

        # Derive unique subproject prefixes from task modules
        prefixes: set[str] = set()
        for m in self.modules:
            prefix = m.strip('/').split('/')[0]
            if (self.worktree / prefix / 'pyproject.toml').exists():
                prefixes.add(prefix)

        if not prefixes:
            return

        worktree = self.worktree  # bind for closure (narrowed to Path)

        # Sync subprojects in parallel
        async def _sync(prefix: str) -> None:
            project_dir = str(worktree / prefix)
            proc = await asyncio.create_subprocess_exec(
                'uv', 'sync', '--project', project_dir,
                cwd=str(self.worktree),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    'Task %s: uv sync failed for %s: %s',
                    self.task_id, prefix, stdout.decode()[:500],
                )
            else:
                logger.info('Task %s: synced venv for %s', self.task_id, prefix)

        await asyncio.gather(*(_sync(p) for p in sorted(prefixes)))

    async def _reconcile_scope_locks(self, plan_files: list[str]) -> bool:
        """Shared blast-radius-expansion + ``self.modules``/``_module_configs``
        sync — the scope-reconciliation choke point (task 2505) used by every
        path that (re)establishes plan.files against the scheduler's file-lock
        set: ``_plan()``, ``_apply_revalidation_skip()``, ``_run_simple_task()``,
        and ``_set_task_scope()``.

        Derives the module set from *plan_files*; if it already matches
        ``self.modules`` this is a no-op (returns True without touching the
        scheduler). Otherwise asks the scheduler to expand/reconcile the lock
        via ``handle_blast_radius_expansion`` (``persist_files=plan_files`` —
        this call persists ``metadata.files=plan_files`` on EVERY successful
        refinement (widen, narrow, or shift) AND on the lock-conflict/requeue
        branch (task 2868); the ONLY non-persist path is its no-op early return
        when the derived module set is unchanged). On success, updates
        ``self.modules``/``self._module_configs`` and returns True. On a lock
        conflict, ``self.modules`` is left UNCHANGED (the scheduler has
        already persisted ``metadata.files=plan_files`` and requeued the task
        to pending on its own) and this returns False — callers decide what
        "not expanded" means for their own flow (REQUEUED report,
        decline-and-fall-through-to-architect, or silent no-op).
        """
        new_modules = derive_modules(
            plan_files, self.config.lock_depth, task_id=self.task_id,
        )
        if set(new_modules) == set(self.modules):
            return True
        expanded = await self.scheduler.handle_blast_radius_expansion(
            self.task_id, self.modules, new_modules,
            persist_files=plan_files,
        )
        if not expanded:
            return False
        # Persistence of metadata.files is centralized in
        # handle_blast_radius_expansion — on every successful refinement
        # (widen/narrow/shift) and the conflict/requeue branch (task 2868) —
        # not here.
        self.modules = new_modules
        self._module_configs = self._resolve_module_configs()
        return True

    async def _set_task_scope(self, new_files: list[str]) -> bool:
        """Orchestrator-side single choke point for widening a task's file
        scope OUTSIDE the architect/plan-boundary flow (task 2505) — used by
        the resume-path scope-grant consumer to fold a steward's
        ``granted_files`` into plan.files without re-invoking the architect.

        Writes plan.json first (``self.plan['files'] = new_files`` +
        ``artifacts.set_plan_files`` — preserves ownership/provenance), then
        reconciles the scheduler's locks/metadata.files via
        ``_reconcile_scope_locks``. On a lock conflict, plan.json is already
        widened (matching metadata.files, which ``handle_blast_radius_expansion``
        persists on every successful refinement and the conflict/requeue branch)
        but ``self.modules`` is left unchanged and
        this returns False so the caller does NOT resume under a foreign lock.

        Same-module widen (task 2505 reviewer regression): when the granted
        file maps to a module the task already locks, ``_reconcile_scope_locks``
        no-ops the ``handle_blast_radius_expansion`` call — which early-returns
        WITHOUT persisting ``metadata.files`` (the no-op early return is the
        only non-persist path). Without a direct persist here, plan.files would
        be widened while metadata.files stayed behind, tripping the MERGE-entry
        ``_check_scope_invariant`` divergence tripwire. So on a successful
        reconcile that left ``self.modules`` UNCHANGED, persist
        ``metadata.files=new_files`` directly (mirroring the done-metadata
        reconcile read-modify-write). On the module-CHANGE success path and the
        lock-CONFLICT path this is skipped — ``handle_blast_radius_expansion``
        already persisted ``metadata.files=new_files``. The module-CHANGE
        success path now ALWAYS persists (widen, narrow, or shift): a pure-widen
        module change previously did NOT persist, which made skipping the direct
        write unsound; task 2868 fixed that, so the skip is now sound. Only the
        SAME-module (no-op) widen still needs the direct persist here, because
        the blast call early-returns without persisting.
        """
        assert self.artifacts is not None
        self.plan['files'] = new_files
        self.artifacts.set_plan_files(new_files, self.session_id)
        modules_before = set(self.modules)
        reconciled = await self._reconcile_scope_locks(new_files)
        if reconciled and set(self.modules) == modules_before:
            merged = await self._merge_fresh_metadata(
                self.task.get('metadata') or {},
                log_context='scope-grant files persist',
            )
            merged['files'] = sanitize_files_for_persist(new_files)
            self.task['metadata'] = merged
            await self.scheduler.update_task(self.task_id, merged)
        return reconciled

    async def _requeue_on_lock_conflict(
        self, files: list[str], reason: str, prefix: str,
    ) -> WorkflowOutcome:
        """Build a REQUEUED ``TerminalReport`` for a genuine cross-module
        lock conflict, stash it on ``self._terminal_report``, and return
        ``WorkflowOutcome.REQUEUED`` (task 2874 amendment).

        Shared by the three sites that convert "the scheduler couldn't
        acquire an additional module lock for a widened file scope" into a
        requeue: ``_plan()``'s blast-radius path, the escalation-resume
        scope-grant path, and ``_execute_verify_review_loop()``'s
        post-replan reconcile. *files* is the full target file set the
        caller wanted scope to cover (not just the delta) — the missing
        locks are re-derived from it here so the diagnostic always reflects
        the actual unavailable modules on the same basis as the real
        conflict detection in ``_reconcile_scope_locks``. *reason* is the
        ``TerminalReport.reason`` string callers (and the retry-cap report)
        key off of; *prefix* only varies the human-readable ``detail``
        message's opening clause (e.g. ``'Plan expansion'``).
        """
        additional = sorted(
            set(derive_modules(
                files, self.config.lock_depth, task_id=self.task_id,
            ))
            - set(self.modules)
        )
        block_detail = (
            f'{prefix} blocked: additional locks {additional} unavailable '
            f'(held by other tasks). Held modules: {sorted(self.modules)}; '
            f'requested files: {sorted(files)}.'
        )
        self._terminal_report = TerminalReport(
            outcome=WorkflowOutcome.REQUEUED,
            reason=reason,
            phase=self.machine.state,
            detail=block_detail,
            category=None,
            # No BLOCKED transition on this path — blocked_from_phase
            # mirrors the current (working) phase, preserving the
            # harness's retry-cap block_phase semantics.
            blocked_from_phase=self.machine.state,
        )
        return WorkflowOutcome.REQUEUED

    async def _plan(self) -> WorkflowOutcome:
        """Invoke the architect to produce a plan."""
        assert self.worktree is not None and self.artifacts is not None

        # Defense-in-depth: if plan.lock already exists, check if it's stale.
        # If stale (self-lock from crashed session or age exceeds threshold),
        # clear it and proceed.  If held by an active session, requeue to
        # avoid duplicate execution.
        if self.artifacts.is_plan_locked() and self.artifacts.read_plan():
            cleared = self.artifacts.clear_stale_plan_lock(self.task_id)
            if not cleared:
                lock_data = self.artifacts.read_plan_lock()
                lock_owner = lock_data.get('session_id', 'unknown') if lock_data else 'unknown'
                logger.info(
                    f'Task {self.task_id}: plan.lock is held by session {lock_owner!r}, '
                    f'skipping architect — requeuing to avoid duplicate execution'
                )
                await self.scheduler.set_task_status(self.task_id, 'pending')
                return WorkflowOutcome.REQUEUED
            # Lock was stale and cleared.  Decide what to keep:
            #  - provenance-stamped or finalized → a complete plan from a prior
            #    session (e.g. blast-radius requeue) → keep for revalidation.
            #  - has steps but not finalized → a prior session was interrupted
            #    mid-build → keep the partial; the completion pass below finishes
            #    it instead of re-planning from scratch.
            #  - no steps, no provenance, not finalized → nothing recoverable →
            #    delete.
            existing = self.artifacts.read_plan()
            if (
                existing
                and not existing.get('_session_id')
                and not existing.get('_finalized_at')
                and not existing.get('steps')
            ):
                plan_path = self.artifacts.root / 'plan.json'
                if plan_path.exists():
                    plan_path.unlink()

        # ── Revalidation vs. fresh planning ──────────────────────────
        # If a provenance-stamped plan already exists (blast-radius requeue),
        # build a revalidation prompt with the diff of what changed on main
        # so the architect can confirm, update, or recreate the plan.
        revalidation = False
        revalidation_changed_files: list[str] = []
        existing_plan = self.artifacts.read_plan()
        if (
            existing_plan
            and existing_plan.get('steps')
            and not existing_plan.get('_finalized_at')
            and not existing_plan.get('_session_id')
        ):
            # ── Completion pass ───────────────────────────────────────
            # A prior session wrote steps but never finalized the plan
            # (it was cut off mid-build by a budget/turn cap or crash).
            # Hand the partial back to the architect to finish rather than
            # discard the work and re-plan from scratch.
            logger.info(
                'Task %s: resuming a partial plan (%d steps, not finalized) '
                'via the completion pass',
                self.task_id, len(existing_plan.get('steps', [])),
            )
            prompt = await self.briefing.build_plan_completion_prompt(
                self.task, existing_plan, worktree=self.worktree,
            )
        elif (
            existing_plan
            and existing_plan.get('steps')
            and existing_plan.get('_session_id')
            and self._old_plan_base
        ):
            current_main = await self.git_ops.get_main_sha()
            revalidation_changed_files = await self.git_ops.get_changed_files(
                self._old_plan_base, current_main,
            )
            plan_file_set = set(existing_plan.get('files', []))
            overlap = [f for f in revalidation_changed_files if f in plan_file_set]
            logger.info(
                'Task %s: revalidating existing plan '
                '(%d changed files, %d overlap with plan)',
                self.task_id, len(revalidation_changed_files), len(overlap),
            )

            # ── Lever B: revalidation skip ────────────────────────────
            # When main has gained no changes that touch the plan's files,
            # the architect's revalidation pass is a near-certain no-op.
            # Short-circuit by stamping _revalidated_at and bumping
            # base_commit ourselves, then return PLANNED. Falls through to
            # the existing architect path on any pre-flight failure.
            if (
                self.config.revalidation_skip_enabled
                and not overlap
                and await self._can_skip_revalidation(existing_plan)
            ):
                skipped = await self._apply_revalidation_skip(
                    existing_plan, current_main,
                )
                if skipped is not None:
                    return skipped
                # Pre-flight check failed inside _apply_revalidation_skip
                # (e.g. blast-radius lock denied) — fall through.

            prompt = await self.briefing.build_revalidation_prompt(
                self.task, existing_plan, revalidation_changed_files,
                worktree=self.worktree,
            )
            revalidation = True
        else:
            # include_prior_proposals=True only when an existing_plan fell
            # through both the completion-pass and revalidation branches
            # above (a genuine re-plan) — both of those branches require a
            # truthy existing_plan, so bool(existing_plan) is False ONLY for
            # a truly-fresh no-plan dispatch, keeping C-A1 anti-anchoring
            # intact for first dispatch.
            # bool(...), not `is not None`, is the correct discriminator:
            # existing_plan is self.artifacts.read_plan() (line ~3071), which
            # returns {} — never None — when plan.json is absent, so an
            # `is not None` check would always be True and defeat the flag
            # entirely. A present-but-empty {} plan carries no steps/session
            # data to re-plan from either, so it is semantically equivalent
            # to "no plan" and correctly falls to fresh-dispatch here too.
            prompt = await self.briefing.build_architect_prompt(
                self.task, worktree=self.worktree,
                include_prior_proposals=bool(existing_plan),
            )

        # Snapshot pre-architect open L0 ids so the post-loop check can
        # detect "architect filed an L0 in lieu of writing a plan."  This is
        # the deterministic catch for RC2 — if it fires, we route directly
        # to L1 without re-invoking the architect (the no-plan loop burned
        # 16 successive Opus calls on task 917 because the cycle counter
        # reset every time main moved).
        pre_l0_ids: set[str] = set()
        if self.escalation_queue:
            pre_l0_ids = {
                e.id for e in self.escalation_queue.get_by_task(
                    self.task_id, status='pending', level=0,
                )
            }

        result: AgentResult | None = None
        rebase_retry_used = False
        for _outer_attempt in range(2):  # at most one rebase-retry round-trip
            for attempt in range(2):
                result = await self._invoke(ARCHITECT, prompt, self.worktree)

                if not result.success:
                    cls = classify_agent_failure(result)
                    # Salvage: the architect may have finalized a complete plan
                    # on disk before the CLI run hit a budget/turn cap (or was
                    # otherwise cut off).  A plan carrying ``_finalized_at`` was
                    # explicitly declared complete, so it is safe to use despite
                    # the run-level failure — the same validation gates below
                    # still apply.  Without the marker the plan is an unfinished
                    # partial; block and let the next session's completion pass
                    # finish it.
                    salvaged = self.artifacts.read_plan()
                    if (
                        salvaged
                        and salvaged.get('_finalized_at')
                        and salvaged.get('steps')
                    ):
                        logger.warning(
                            'Task %s: architect run failed (%s) but a finalized '
                            'plan (%d steps) is on disk — salvaging instead of '
                            'discarding',
                            self.task_id, cls.kind.value,
                            len(salvaged.get('steps', [])),
                        )
                        if self.event_store:
                            self.event_store.emit(
                                EventType.plan_salvaged,
                                task_id=self.task_id,
                                phase='plan',
                                data={
                                    'failure_kind': cls.kind.value,
                                    'cost_usd': result.cost_usd,
                                    'turns': result.turns,
                                    'steps': len(salvaged.get('steps', [])),
                                },
                            )
                        self.plan = salvaged
                        break  # fall through to validation / provenance / lock
                    logger.error(
                        'Task %s: architect failed (%s): %s',
                        self.task_id, cls.kind.value, cls.summary,
                    )
                    return await self._mark_blocked(
                        f'Planning failed: {cls.summary}',
                        detail=cls.diagnostic_detail,
                    )

                # Detect anomalous premature exit: succeeded but suspiciously
                # few turns and low cost — likely a transient CLI issue.
                self.plan = self.artifacts.read_plan()
                if (
                    attempt == 0
                    and result.turns <= 2
                    and result.cost_usd < 0.20
                    and not self.plan
                ):
                    logger.warning(
                        f'Task {self.task_id}: architect completed anomalously '
                        f'(turns={result.turns}, cost=${result.cost_usd:.2f}, '
                        f'duration={result.duration_ms}ms, output_len={len(result.output)}) '
                        f'— retrying once'
                    )
                    continue

                break

            # Architect task-rejection artifacts.  All are terminal — handle
            # deterministically before the "no plan.json" failure path so
            # none interacts with the consecutive_no_plan_failures cycle
            # counter.  Order matters: unactionable_task and false_premise are
            # the most decisive (jump straight to L1, bypass steward);
            # already_done is a clean DONE; blocking_dependency may re-loop
            # the architect.
            if self.artifacts.read_unactionable_task() is not None:
                return await self._handle_unactionable_task_report()
            if self.artifacts.read_false_premise() is not None:
                return await self._handle_false_premise_report()
            if self.artifacts.read_already_done() is not None:
                return await self._handle_already_done_report()
            # Fix B: the architect may have written a blocking_dependency
            # report instead of a plan.
            if self.artifacts.read_blocking_dependency() is not None:
                dep_outcome = await self._handle_blocking_dep_report(
                    rebase_retry_used=rebase_retry_used,
                )
                if dep_outcome is not None:
                    return dep_outcome
                # Helper rebased + cleared the artifact; loop back to retry
                # the architect once.
                rebase_retry_used = True
                continue
            break

        assert result is not None  # range(2) always executes at least once

        # RC2 deterministic catch: the architect filed a fresh L0 escalation
        # in lieu of writing a plan.  Route straight to L1 (skip the steward,
        # skip the no-plan loop) — the architect already filed an L0, so
        # _mark_blocked must not create another one.
        if not self.plan and self.escalation_queue:
            post_l0 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            new_l0 = [e for e in post_l0 if e.id not in pre_l0_ids]
            if new_l0:
                summary = new_l0[0].summary
                logger.warning(
                    'Task %s: architect filed L0 without plan (%s) — '
                    'auto-promoting to L1 to break no-plan loop',
                    self.task_id, summary[:120],
                )
                return await self._mark_blocked(
                    f'Architect filed L0 without plan: {summary}',
                    detail=new_l0[0].detail,
                    escalate_to_human=True,
                    skip_escalation=True,  # the architect already filed one
                )

        if not self.plan:
            cls = classify_agent_failure(result)
            logger.error(
                'Task %s: architect produced no plan.json (%s): %s',
                self.task_id, cls.kind.value, cls.summary,
            )
            return await self._handle_no_plan_failure(
                f'Planning failed: no plan.json produced — {cls.summary}',
                detail=(
                    'Architect succeeded but did not write .task/plan.json\n'
                    f'{cls.diagnostic_detail}'
                ),
            )

        if not self.plan.get('steps'):
            # Normalization (in read_plan) didn't help — try a one-shot
            # repair prompt before blocking.
            logger.warning(
                'Task %s: plan has no "steps" after normalization — '
                'attempting repair prompt',
                self.task_id,
            )
            repaired = await self._repair_plan_schema()
            if repaired:
                self.plan = self.artifacts.read_plan()

        if not self.plan.get('steps'):
            plan_dump = json.dumps(self.plan, indent=2)
            logger.error(
                f'Task {self.task_id}: architect wrote plan.json but missing/empty '
                f'"steps" — full plan content: {plan_dump}'
            )
            return await self._handle_no_plan_failure(
                'Planning failed: plan missing "steps"',
                detail=f'Plan content:\n{plan_dump[:4000]}',
            )

        # Hard-require the completeness marker.  The architect must call
        # confirm_plan() as its final action; a plan with steps but no
        # _finalized_at is an unfinished partial — either the run was cut off
        # mid-build, or the architect skipped the final call.  Route it through
        # the no-plan failure handler (which has cycle detection).  The partial
        # is left on disk, so the next session's completion pass can resume it.
        #
        # This runs BEFORE stamp_plan_provenance so a missing confirm_plan is
        # caught here rather than masked by the marker that stamping backfills.
        if not self.plan.get('_finalized_at'):
            return await self._handle_no_plan_failure(
                'Planning failed: plan not finalized '
                '(architect did not call confirm_plan)',
                detail=(
                    'The plan on disk has steps but no _finalized_at marker. '
                    'The architect must call confirm_plan() as its final action '
                    'to mark the plan complete. Treating as an incomplete plan; '
                    'a later session will resume and finish it.'
                ),
            )

        if outcome := await self._validate_prerequisites_or_block('initial plan'):
            return outcome

        # Stamp provenance and acquire lock
        self.artifacts.stamp_plan_provenance(self.session_id)
        self.artifacts.lock_plan(self.session_id)
        self.plan = self.artifacts.read_plan()

        if revalidation and self.event_store:
            plan_file_set = set(self.plan.get('files', []))
            self.event_store.emit(
                EventType.plan_revalidated,
                task_id=self.task_id,
                data={
                    'changed_files_count': len(revalidation_changed_files),
                    'overlap_count': len(
                        [f for f in revalidation_changed_files
                         if f in plan_file_set]
                    ),
                },
            )

        plan_files = self.plan.get('files', [])
        if not plan_files:
            return await self._handle_no_plan_failure(
                'Planning failed: plan missing "files"',
                detail=(
                    'Architect wrote plan.json without a non-empty "files" array. '
                    'Files are required to derive module locks.'
                ),
            )
        plan_modules = derive_modules(
            plan_files, self.config.lock_depth, task_id=self.task_id,
        )
        logger.info(
            f'Task {self.task_id}: derived {len(plan_modules)} modules '
            f'from {len(plan_files)} files: {plan_modules}'
        )

        if (
            set(plan_modules) != set(self.modules)
            and not await self._reconcile_scope_locks(plan_files)
        ):
            # Annotate the requeue so the per-task retry-cap report can
            # name *why* — without this, three blast-radius requeues in a
            # row produce a cap-exhaust report with phase/reason='unknown'
            # (REVIEW-CYCLE-1; block_phase='plan').
            return await self._requeue_on_lock_conflict(
                plan_files, 'plan_blast_radius_lock_conflict', 'Plan expansion',
            )
        # self.modules/_module_configs already updated by
        # _reconcile_scope_locks on success (no-op if scope unchanged).

        # Write plan decisions to memory
        await self._write_decisions_to_memory()

        logger.info(
            f'Task {self.task_id}: plan created with '
            f'{len(self.plan.get("prerequisites", []))} prerequisites, '
            f'{len(self.plan.get("steps", []))} steps'
        )
        return WorkflowOutcome.PLANNED

    async def _can_skip_revalidation(self, plan: dict) -> bool:
        """Lever B pre-flight checks for the revalidation skip.

        All checks must pass for the optimisation to apply. Conservative —
        on any uncertainty, return False so the existing architect-driven
        revalidation runs.
        """
        assert self.worktree is not None
        # Schema version must match — bumped schemas mean the architect
        # may need to refresh fields the orchestrator can't synthesise.
        if plan.get('_schema_version') != PLAN_SCHEMA_VERSION:
            logger.info(
                'Task %s: revalidation skip declined — schema mismatch '
                '(plan=%r, current=%r)',
                self.task_id,
                plan.get('_schema_version'),
                PLAN_SCHEMA_VERSION,
            )
            return False
        # Every plan file must still exist in the worktree.
        for f in plan.get('files', []):
            if not (self.worktree / f).exists():
                logger.info(
                    'Task %s: revalidation skip declined — plan file %r '
                    'missing in worktree',
                    self.task_id, f,
                )
                return False
        # Plan provenance age bound.
        created_at = plan.get('_created_at') or plan.get('_revalidated_at')
        if created_at:
            try:
                stamped = datetime.fromisoformat(str(created_at))
                if stamped.tzinfo is None:
                    stamped = stamped.replace(tzinfo=UTC)
                age_hours = (
                    datetime.now(UTC) - stamped
                ).total_seconds() / 3600.0
                if age_hours > self.config.max_revalidation_age_hours:
                    logger.info(
                        'Task %s: revalidation skip declined — plan age '
                        '%.1fh exceeds bound %.1fh',
                        self.task_id, age_hours,
                        self.config.max_revalidation_age_hours,
                    )
                    return False
            except (TypeError, ValueError):
                logger.info(
                    'Task %s: revalidation skip declined — could not parse '
                    '_created_at=%r',
                    self.task_id, created_at,
                )
                return False
        # Conservative: if the prior run blocked at REVIEW, the architect
        # path may be needed to refresh strategy. Read the iteration log
        # for the most recent block annotation.
        metadata = self.task.get('metadata') or {}
        if (
            metadata.get('last_block_phase') == 'review'
            and metadata.get('last_block_outcome') == 'blocked'
        ):
            logger.info(
                'Task %s: revalidation skip declined — prior block at REVIEW',
                self.task_id,
            )
            return False
        return True

    async def _apply_revalidation_skip(
        self, plan: dict, current_main: str,
    ) -> WorkflowOutcome | None:
        """Lever B side-effect path. Returns PLANNED on success, or None to
        signal the caller should fall through to the architect path."""
        assert self.artifacts is not None and self.worktree is not None

        # Re-derive modules from plan files; if scope grew, ask scheduler
        # to expand the lock. Denial means a sibling task holds an
        # overlapping module — fall through to the architect path which
        # already handles the requeue case.
        plan_files = plan.get('files', [])
        plan_modules = derive_modules(
            plan_files, self.config.lock_depth, task_id=self.task_id,
        )
        if (
            set(plan_modules) != set(self.modules)
            and not await self._reconcile_scope_locks(plan_files)
        ):
            logger.info(
                'Task %s: revalidation skip declined — blast-radius '
                'expansion denied',
                self.task_id,
            )
            return None
        # self.modules/_module_configs already updated by
        # _reconcile_scope_locks on success (no-op if scope unchanged).

        # Bump revalidation stamp + base commit (mirrors confirm_plan).
        try:
            self.artifacts.bump_revalidation_stamp(
                self.session_id, base_commit=current_main,
            )
        except ValueError as exc:
            logger.warning(
                'Task %s: revalidation skip declined — bump_revalidation_stamp '
                'rejected plan: %s',
                self.task_id, exc,
            )
            return None

        # Acquire plan lock (the architect path also does this; the eval
        # path skips it entirely).  If another session holds the lock,
        # fall through to the architect path which has its own retry logic.
        if not self.artifacts.lock_plan(self.session_id):
            logger.info(
                'Task %s: revalidation skip declined — plan.lock contended',
                self.task_id,
            )
            return None

        self.plan = self.artifacts.read_plan()

        # Stamp optimistic-path metadata for the auto-eval hook.
        await self._stamp_optimistic_path('revalidation_skip')

        if self.event_store:
            self.event_store.emit(
                EventType.phase_skipped,
                task_id=self.task_id,
                phase='plan',
                data={
                    'reason': 'revalidation_skipped_no_overlap',
                    'plan_session_id': str(plan.get('_session_id') or ''),
                    'plan_files': plan_files,
                    'main_sha': current_main,
                },
            )

        logger.info(
            'Task %s: Lever B — revalidation skipped (overlap=0, '
            'main %s -> stamped on plan)',
            self.task_id, current_main[:12],
        )
        return WorkflowOutcome.PLANNED

    def _should_run_simple_task(self) -> bool:
        """Whether this dispatch should take the Lever C SIMPLE_TASK path.

        Extracted verbatim from the inline gate in ``_drive`` — a
        behaviour-preserving refactor that makes the gate unit-testable —
        plus ONE new AND term for task ν: ``not
        RoutingState.from_metadata(metadata).simple_saturated``.

        The pre-existing conditions are unchanged: no ``initial_plan`` (not
        eval mode), ``simple_task_enabled``, and the RUNTIME dispatch vetoes
        ``auto_eval_redo`` / ``force_full_path``, plus the author-declaration
        predicate ``is_declared_simple_task``. ``simple_saturated`` joins the
        runtime vetoes here (NOT inside ``is_declared_simple_task``, which its
        own docstring keeps a pure author-declaration predicate): once a
        SIMPLE_TASK dispatch has demonstrably exhausted its turn cap
        (``_stamp_simple_saturated``), the "simple" label is retired and every
        subsequent dispatch takes the full architect path.
        """
        from orchestrator.agents.triage import is_declared_simple_task
        metadata = self.task.get('metadata') or {}
        return (
            not self.initial_plan
            and self.config.simple_task_enabled
            and not metadata.get('auto_eval_redo')
            and not metadata.get('force_full_path')
            and is_declared_simple_task(self.task)
            and not RoutingState.from_metadata(metadata).simple_saturated
        )

    async def _run_simple_task(self) -> WorkflowOutcome:
        """Lever C — single-agent simple-task path.

        Dispatches the SIMPLE_TASK role (sonnet) to explore, register a
        plan via the plan-tools MCP server, edit the listed files, and
        mark the step(s) done in one session.

        Returns:
            ``PLANNED`` — plan.json was written and at least one step
                marked done; the workflow continues to VERIFY without
                invoking the implementer.
            ``DONE`` — SIMPLE_TASK reported the work is already on main.
            ``BLOCKED`` — SIMPLE_TASK reported the task is unactionable.
            ``REQUEUED`` (sentinel) — SIMPLE_TASK gave up partway; caller
                falls through to the architect path.
        """
        assert self.worktree is not None and self.artifacts is not None

        prompt = await self.briefing.build_simple_task_prompt(
            self.task, worktree=self.worktree,
        )

        try:
            result = await self._invoke(SIMPLE_TASK, prompt, self.worktree)
        except Exception as exc:  # noqa: BLE001 — fall through to architect
            logger.warning(
                'Task %s: SIMPLE_TASK invocation failed (%s) — '
                'falling through to architect path',
                self.task_id, exc,
            )
            return WorkflowOutcome.REQUEUED

        if not result.success:
            cls = classify_agent_failure(result)
            if cls.kind == AgentFailureKind.MAX_TURNS:
                # Task ν: the SIMPLE_TASK agent exhausted its turn cap without
                # completing — the "simple" label was demonstrated wrong, so
                # stamp metadata.routing.simple_saturated=True and let every
                # SUBSEQUENT dispatch take the full architect path. Detection
                # is on the error_max_turns subtype only (classify_agent_failure
                # orders MAX_TURNS below TIMED_OUT, so a wall-clock SIGKILL is
                # NOT treated as saturation).
                await self._stamp_simple_saturated()
            logger.info(
                'Task %s: SIMPLE_TASK did not succeed (%s) — '
                'falling through to architect path',
                self.task_id, cls.kind.value,
            )
            return WorkflowOutcome.REQUEUED

        # Architect-style escape hatches — same artifacts so the existing
        # handlers work unchanged.
        if self.artifacts.read_unactionable_task() is not None:
            return await self._handle_unactionable_task_report()
        if self.artifacts.read_false_premise() is not None:
            return await self._handle_false_premise_report()
        if self.artifacts.read_already_done() is not None:
            return await self._handle_already_done_report()
        if self.artifacts.read_blocking_dependency() is not None:
            # Falling through — the architect path's
            # _handle_blocking_dep_report logic re-acquires base_commit
            # context and registers the dependency cleanly.
            return WorkflowOutcome.REQUEUED

        plan = self.artifacts.read_plan()
        if not plan or not plan.get('steps'):
            logger.info(
                'Task %s: SIMPLE_TASK wrote no plan — falling through',
                self.task_id,
            )
            return WorkflowOutcome.REQUEUED

        # Verify at least one step is done — the SIMPLE_TASK contract is
        # plan + implement + mark-done. Anything less and we let the
        # architect path take over to avoid handing a half-built plan to
        # _execute_iterations.
        any_done = any(
            isinstance(s, dict) and s.get('status') == 'done'
            for col in ('prerequisites', 'steps')
            for s in plan.get(col, [])
        )
        if not any_done:
            logger.info(
                'Task %s: SIMPLE_TASK plan has no steps marked done — '
                'falling through to architect path',
                self.task_id,
            )
            return WorkflowOutcome.REQUEUED

        # Stamp provenance + acquire plan lock (same shape as _plan()).
        try:
            self.artifacts.stamp_plan_provenance(self.session_id)
        except ValueError as exc:
            logger.warning(
                'Task %s: SIMPLE_TASK provenance stamp failed (%s) — '
                'falling through',
                self.task_id, exc,
            )
            return WorkflowOutcome.REQUEUED
        self.artifacts.lock_plan(self.session_id)
        self.plan = self.artifacts.read_plan()

        # Refresh module assignments from the plan's files (handles the
        # case where the SIMPLE_TASK agent expanded scope by one file).
        plan_files = self.plan.get('files', [])
        if plan_files:
            plan_modules = derive_modules(
                plan_files, self.config.lock_depth, task_id=self.task_id,
            )
            if set(plan_modules) != set(self.modules):
                # Silent no-op on lock conflict (same as before this was
                # extracted into _reconcile_scope_locks): SIMPLE_TASK doesn't
                # fail out here, it just doesn't re-tighten the lock.
                await self._reconcile_scope_locks(plan_files)

        await self._stamp_optimistic_path('simple_task')

        if self.event_store:
            self.event_store.emit(
                EventType.phase_skipped,
                task_id=self.task_id,
                phase='plan',
                data={
                    'reason': 'architect_skipped_simple_task',
                    'classifier_signals': {
                        'title': str(self.task.get('title') or ''),
                        'files': list(
                            (self.task.get('metadata') or {}).get('files') or []
                        ),
                        'priority': str(self.task.get('priority') or ''),
                    },
                    'plan_files': plan_files,
                },
            )

        logger.info(
            'Task %s: Lever C — SIMPLE_TASK produced plan with %d step(s)',
            self.task_id, len(self.plan.get('steps', [])),
        )
        return WorkflowOutcome.PLANNED

    async def _stamp_optimistic_path(self, kind: str) -> None:
        """Stamp ``metadata.optimistic_path`` on the task so the harness's
        auto-eval hook can detect that this task took the optimistic path
        on its current attempt.

        Fire-and-forget — failure logs a warning and does not block.
        """
        try:
            metadata = dict(self.task.get('metadata') or {})
            metadata['optimistic_path'] = kind
            self.task['metadata'] = metadata
            await self.scheduler.update_task(
                self.task_id, metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                'Task %s: failed to stamp optimistic_path=%s: %s',
                self.task_id, kind, exc,
            )

    async def _stamp_simple_saturated(self) -> None:
        """Stamp ``metadata.routing.simple_saturated=True`` (task ν).

        Called when a SIMPLE_TASK invocation ends at its turn cap
        (``AgentFailureKind.MAX_TURNS``) without completing: the author's
        ``complexity='simple'`` declaration was demonstrated wrong, so the
        runtime retires the label — every SUBSEQUENT dispatch of this task
        fails the ``_should_run_simple_task`` gate and takes the full
        architect path ("the label stops being re-trusted once demonstrated
        wrong").

        A read-modify-write that flips ONLY ``simple_saturated`` and merges
        just the ``routing`` key, mirroring ``_record_routing_decision``:
        ``_invoke`` already wrote ``routing.latest``/``history``/
        ``routing_tier`` for this dispatch (in-memory and via merge), and a
        merge-mode write preserves that state rather than clobbering it.
        Idempotent (returns early if already stamped) and best-effort — a
        failed scheduler write logs a warning and never raises, honoring the
        "routing telemetry must never block or crash a caller" philosophy.
        The in-memory ``self.task['metadata']['routing']`` update always runs
        regardless of the scheduler write's outcome.
        """
        state = RoutingState.from_metadata(self.task.get('metadata'))
        if state.simple_saturated:
            return
        new_state = state.model_copy(update={'simple_saturated': True})
        if self.scheduler:
            try:
                await self.scheduler.update_task(
                    self.task_id,
                    {'routing': new_state.model_dump()},
                    metadata_mode='merge',  # type: ignore[reportCallIssue]
                )
            except Exception:
                logger.warning(
                    'Task %s: failed to stamp routing.simple_saturated',
                    self.task_id, exc_info=True,
                )
        self.task.setdefault('metadata', {})['routing'] = new_state.model_dump()

    async def _validate_prerequisites_or_block(
        self, context: str
    ) -> WorkflowOutcome | None:
        """Validate prerequisites format; block if invalid.

        Encapsulates the try/validate/except/mark-blocked pattern shared by the
        initial-plan checkpoint and the replan checkpoint, parameterised by a
        *context* string that appears in log and escalation messages.

        Args:
            context: Short description for log/error messages, e.g.
                     ``'initial plan'`` or ``'replan'``.

        Returns:
            A :class:`WorkflowOutcome` (BLOCKED) if validation fails,
            ``None`` if prerequisites are valid.
        """
        assert self.artifacts is not None
        try:
            self.artifacts.validate_plan_prerequisites()
        except ValueError as exc:
            plan_dump = json.dumps(self.plan, indent=2)
            logger.error(
                f'Task {self.task_id}: {context} produced plan.json with invalid '
                f'prerequisites — {exc}'
            )
            return await self._mark_blocked(
                f'Planning failed ({context}): invalid prerequisites format — {exc}',
                detail=f'Plan content:\n{plan_dump[:4000]}',
            )
        return None

    async def _repair_plan_schema(self) -> bool:
        """One-shot attempt to fix a plan that is missing ``steps``.

        Sends the broken plan back to the architect with a focused repair
        prompt.  Returns True if the repaired plan now has a ``steps`` array.
        """
        assert self.artifacts is not None  # caller guarantees
        assert self.worktree is not None

        broken_plan = self.artifacts.read_plan()
        if not broken_plan:
            return False

        plan_path = str(self.artifacts.root / 'plan.json')
        plan_dump = json.dumps(broken_plan, indent=2)[:6000]

        repair_prompt = (
            'The architect produced a plan that is structurally invalid — '
            'it is missing the required top-level "steps" array.\n\n'
            f'Here is the broken plan content:\n\n```json\n{plan_dump}\n```\n\n'
            'The required schema is:\n'
            '```json\n'
            '{\n'
            '  "task_id": "<task id>",\n'
            '  "title": "<task title>",\n'
            '  "files": ["path/to/file1.py"],\n'
            '  "analysis": "<analysis>",\n'
            '  "prerequisites": [\n'
            '    {"id": "pre-1", "description": "...", "status": "pending", "commit": null}\n'
            '  ],\n'
            '  "steps": [\n'
            '    {"id": "step-1", "type": "test", "description": "...", "status": "pending", "commit": null},\n'
            '    {"id": "step-2", "type": "impl", "description": "...", "status": "pending", "commit": null}\n'
            '  ],\n'
            '  "design_decisions": [\n'
            '    {"decision": "...", "rationale": "..."}\n'
            '  ]\n'
            '}\n'
            '```\n\n'
            'Your job: restructure the existing plan content into the required '
            'schema.  Do NOT explore the codebase or redesign the plan.  Simply '
            'reorganize the existing keys and values into the correct shape and '
            f'write the result to `{plan_path}` using the Write tool.'
        )

        try:
            await self._invoke(ARCHITECT, repair_prompt, self.worktree)
        except Exception as e:
            logger.warning(
                'Task %s: repair prompt invocation failed: %s',
                self.task_id, e,
            )
            return False

        repaired_plan = self.artifacts.read_plan()
        if repaired_plan.get('steps'):
            # The repair restructures an existing complete plan into valid
            # schema (the architect was told not to redesign), so the repaired
            # plan is complete by construction.  Stamp the completeness marker
            # — the repair prompt writes plan.json via the Write tool rather
            # than confirm_plan, so the _finalized_at gate would otherwise
            # reject this plan.
            repaired_plan.setdefault('_finalized_at', datetime.now(UTC).isoformat())
            self.artifacts.write_plan(repaired_plan)
            logger.info(
                'Task %s: repair prompt succeeded — plan now has %d steps',
                self.task_id, len(repaired_plan['steps']),
            )
            return True

        logger.warning(
            'Task %s: repair prompt did not produce a valid "steps" array',
            self.task_id,
        )
        return False

    async def _read_fresh_backend_metadata(
        self,
        *,
        log_context: str,
        require_fresh: bool = False,
    ) -> dict | None:
        """Read the backend's current task metadata blob, or ``None`` on failure.

        Extracted from :meth:`_merge_fresh_metadata` (task 2991) so a caller can
        distinguish "read OK" from "could not read".  That distinction matters
        only for the delete-by-omission clears
        (:meth:`_clear_merge_phase_entered`, :meth:`_clear_merge_retry_pending`):
        they persist with ``metadata_mode='replace'``, a whole-blob overwrite,
        so writing a payload built from in-memory-only metadata would DELETE
        every backend-only key (``memory_hints`` re-attached by Stage-2
        reconciliation, ``_causation_id``) — the #4271 sibling-clobber bug.

        Such a caller has two equivalent ways to refuse an unverified write, and
        both funnel through here (rebase reconciliation, task 2991 vs task 3024):

        * ``require_fresh=False`` (default) — a failed read returns ``None`` and
          the caller skips its durable write itself.  Used by
          :meth:`_clear_merge_phase_entered`, which needs the read-failed signal
          inline so it can still clear the in-memory copy.
        * ``require_fresh=True`` — a failed read RAISES instead of returning
          ``None``, so a caller already wrapping its read+write in one
          best-effort ``try`` (:meth:`_clear_merge_retry_pending`, via
          :meth:`_merge_fresh_metadata`) lands the refusal in the same handler
          as a persist failure.

        Args:
            log_context: Short descriptor of the write site, used verbatim in
                the warning (e.g. ``'merge_phase_liveness clear'``).
            require_fresh: When True, raise rather than report a failed read.
                Never returns ``None`` under this flag.

        Returns:
            ``{}`` — read OK, backend metadata is empty/absent.
            ``dict`` — the backend's metadata blob.
            ``None`` — could not read: ``get_task`` raised, returned a
            non-dict (e.g. ``None`` for a vanished/unreadable task), or
            returned a task whose persisted ``metadata`` is CORRUPT (a
            non-dict — a state ``Scheduler.update_task``'s own docstring
            acknowledges as the reason ``metadata_mode='replace'`` exists).
            Not reachable when ``require_fresh`` is True.

        Raises:
            Exception: Only when ``require_fresh`` is True — the underlying
                ``get_task`` error verbatim, or a :class:`RuntimeError` for
                every other unreadable shape (non-task result, corrupt
                non-dict ``metadata``).
        """
        try:
            fresh_task = await self.scheduler.get_task(self.task_id)
        except Exception as exc:  # noqa: BLE001 — best-effort unless require_fresh
            if require_fresh:
                logger.warning(
                    'Task %s: failed to refresh metadata before %s write; '
                    'refusing to build an unverified whole-blob payload '
                    '(caller requires a backend-verified read): %s',
                    self.task_id, log_context, exc,
                )
                raise
            logger.warning(
                'Task %s: failed to refresh metadata before %s write; '
                'falling back to in-memory metadata '
                '(memory_hints may be clobbered): %s',
                self.task_id, log_context, exc,
            )
            return None
        # Non-dict return (e.g. None for a vanished task) is the path production
        # actually takes — Scheduler.get_task catches every exception and
        # returns None.  Silent by design when require_fresh is False, which
        # preserves the pre-extraction behaviour (it warned only on the
        # exception branch).
        if not isinstance(fresh_task, dict):
            if require_fresh:
                msg = (
                    f'Task {self.task_id}: metadata refresh before {log_context} '
                    f'write returned {type(fresh_task).__name__}, not a task dict '
                    f'— refusing to build an unverified whole-blob payload'
                )
                logger.warning(msg)
                raise RuntimeError(msg)
            return None
        meta = fresh_task.get('metadata')
        if meta is None:
            return {}
        if not isinstance(meta, dict):
            # A corrupt persisted blob must be reported as UNREADABLE, not
            # returned raw: every caller immediately unpacks the result
            # (``{**in_memory, **(backend or {})}`` / ``{**metadata,
            # **backend}``) and a truthy non-dict raises `TypeError: 'X' object
            # is not a mapping` from an unguarded line — on the merge critical
            # path, since _clear_merge_phase_entered runs inside
            # _submit_to_merge_queue. Reporting None also gives the two clears
            # the right behaviour for free: skip the whole-blob 'replace' write
            # rather than overwrite a blob nobody understands.
            msg = (
                f'Task {self.task_id}: backend metadata is non-dict '
                f'({type(meta).__name__}) before {log_context} write; '
                f'treating as unreadable'
            )
            logger.warning(msg)
            if require_fresh:
                # A require_fresh caller reads the RETURN value as proof the
                # blob is backend-verified, so reporting None here would send it
                # down _merge_fresh_metadata's in-memory-only fallback and
                # straight into the 'replace' clobber this guard exists to
                # prevent. Refuse the same way the other unreadable shapes do.
                raise RuntimeError(msg)
            return None
        return meta

    async def _merge_fresh_metadata(
        self,
        in_memory_metadata: dict,
        *,
        log_context: str,
        require_fresh: bool = False,
    ) -> dict:
        """Read-modify-write helper: merge in-memory metadata with the backend's.

        Fetches the backend's current task metadata and overlays it onto
        ``in_memory_metadata`` so that keys added after ``self.task`` was loaded
        (e.g. ``memory_hints`` re-attached by Stage-2 reconciliation) survive the
        next write.  Backend keys win on collision, which is the correct policy: a
        backend-side addition represents an external write that must not be
        discarded.

        If ``get_task`` fails, logs a warning containing ``log_context`` and falls
        back to ``in_memory_metadata`` alone.  Mirrors the boundary-normalisation
        pattern used in ``_handle_terminal_exit_on_block``.  The backend read
        itself lives in :meth:`_read_fresh_backend_metadata`, which callers
        needing the read-failed signal (the ``'replace'``-mode clears) use
        directly.

        Args:
            in_memory_metadata: The in-memory metadata dict (from
                ``self.task.get('metadata') or {}``).
            log_context: Short descriptor of the write site, used verbatim in the
                fallback warning (e.g. ``'no-plan counter'``,
                ``'infra-resume thrash counter'``).
            require_fresh: When True, a failed (or non-dict) backend read RAISES
                instead of silently falling back to ``in_memory_metadata`` alone.
                Callers that persist with ``metadata_mode='replace'`` MUST pass
                True: under replace, every key omitted from the payload is
                DELETED, so the fallback payload would destroy backend-only keys
                (e.g. ``memory_hints`` re-attached by Stage-2 reconciliation)
                rather than merely failing to update them.  Such a caller is
                expected to catch and skip its write — no write is strictly safer
                than an unverified whole-blob one.  Defaults to False, which
                keeps the additive/merge-mode callers' best-effort behaviour
                (their fallback is bounded to the keys actually supplied).

        Returns:
            A new dict ready for counter-field assignment and persist.

        Raises:
            Exception: Only when ``require_fresh`` is True — the underlying
                ``get_task`` error verbatim, or a :class:`RuntimeError` if the
                read returned something other than a task dict.
        """
        # Merge: start from in-memory metadata so that locally-set keys not yet
        # persisted are preserved, then overlay fresh backend keys so that
        # backend-side additions (e.g. memory_hints from Stage-2 reconciliation)
        # win on collision.  When the read failed (None, only reachable with
        # require_fresh=False) the backend overlay is empty and we fall back to
        # the in-memory copy only.
        backend = await self._read_fresh_backend_metadata(
            log_context=log_context, require_fresh=require_fresh,
        )
        return {**in_memory_metadata, **(backend or {})}

    async def _handle_no_plan_failure(
        self, reason: str, *, detail: str,
    ) -> WorkflowOutcome:
        """Block on a no-plan / malformed-plan failure with cycle detection.

        Fix C — increments ``consecutive_no_plan_failures`` keyed by
        ``last_no_plan_main_sha`` inside the typed ``metadata.retry_ledger``
        blob (see :func:`_evaluate_no_plan`).  When the counter hits ≥ 2
        with the same main SHA, the no-plan loop has been observed and we
        escalate to a human directly (skip the steward) rather than
        letting the workflow re-pend.

        Persist failure escalates to a human immediately rather than
        logging and proceeding: a silently-lost counter increment would
        let this money-burning loop under-fire.

        Deploy note: pre-migration tasks carry this counter only at the
        legacy top-level ``consecutive_no_plan_failures``/``total_no_plan_failures``
        keys — there is no fallback that lifts them into ``retry_ledger``.
        The first guard invocation after deploy therefore sees a fresh
        all-zero ledger and costs at most one extra no-plan cycle before the
        counter re-accumulates. Self-healing and benign, same precedent as
        the signature-format migration note on ``_merge_outcome_signature``.
        """
        try:
            current_main_sha = await self.git_ops.get_main_sha()
        except Exception as exc:  # noqa: BLE001 — fall through to standard path
            logger.warning(
                'Task %s: could not read main SHA for no-plan cycle counter: %s',
                self.task_id, exc,
            )
            current_main_sha = ''

        metadata = self.task.get('metadata') or {}
        ledger = _build_retry_ledger(metadata)

        verdict = _evaluate_no_plan(ledger, current_main_sha)

        # Read-modify-write: see _merge_fresh_metadata for the merge policy.
        new_metadata = await self._merge_fresh_metadata(
            metadata, log_context='no-plan counter',
        )
        # In-memory ledger intentionally wins over any backend retry_ledger;
        # the backend overlay above is only for non-counter keys (e.g. memory_hints).
        new_metadata['retry_ledger'] = verdict.ledger.model_dump()
        self.task['metadata'] = new_metadata
        try:
            await self.scheduler.update_task(self.task_id, metadata=new_metadata)
        except Exception as exc:  # noqa: BLE001 — counter can't be trusted; escalate.
            logger.warning(
                'Task %s: failed to persist no-plan cycle counter; '
                'escalating to human (counter cannot be trusted): %s',
                self.task_id, exc,
            )
            return await self._mark_blocked(
                f'Failed to persist no-plan cycle counter: {exc}',
                detail=detail, escalate_to_human=True,
            )

        if verdict.escalate:
            counter = verdict.ledger.consecutive_no_plan_failures
            total = verdict.ledger.total_no_plan_failures
            logger.warning(
                'Task %s: no-plan loop confirmed (%s) — '
                'consecutive=%d on main SHA %s, total=%d; escalating to human',
                self.task_id, verdict.trigger, counter,
                current_main_sha[:12] or '<unknown>', total,
            )
            full_reason = (
                f'Repeated no-plan failure (counter={counter}, total={total}) '
                f'via {verdict.trigger}: {reason}'
            )
            return await self._mark_blocked(
                full_reason, detail=detail, escalate_to_human=True,
            )

        return await self._mark_blocked(reason, detail=detail)

    async def _check_infra_resume_thrash(self) -> WorkflowOutcome | None:
        """Fix 2 — detect & escalate repeated infra-issue resume thrash.

        Called from the ESCALATED branch of :meth:`run` after the steward
        has resolved the L0 and Fix 1 confirmed the task status is still
        non-terminal (pending / in-progress).  If the most recent resolved
        L0 was an ``infra_issue`` and the iteration log has not grown since
        the previous resume, increment ``consecutive_infra_resume_failures``
        inside the typed ``metadata.retry_ledger`` blob (see
        :func:`_evaluate_infra_resume`).  At ``max_consecutive_infra_resumes``,
        route to ``_mark_blocked(escalate_to_human=True)`` instead of
        dispatching the implementer again.

        Returns:
            ``WorkflowOutcome.BLOCKED`` when the threshold is hit; ``None``
            to fall through to the existing implementer-resume path.

        The iteration-log growth signal (rather than HEAD SHA) is canonical:
        steward fix-commits and ``--allow-empty`` commits both advance HEAD
        without representing real agent progress.  The iteration log is the
        signal already used by ``_has_prior_implementation``.

        Mirrors :meth:`_handle_no_plan_failure` style — same per-task
        concurrency assumption as the existing
        ``consecutive_no_plan_failures`` writer; no new hazard.  Persist
        failure escalates to a human immediately rather than logging and
        proceeding, for the same reason as the no-plan guard.

        Deploy note: pre-migration tasks carry
        ``consecutive_infra_resume_failures``/``last_infra_resume_iteration_count``
        only at the legacy top level — there is no fallback that lifts them
        into ``retry_ledger``. The first guard invocation after deploy
        therefore sees a fresh all-zero ledger, costing at most one extra
        infra-resume cycle before the counter re-accumulates. Self-healing
        and benign, same precedent as :meth:`_handle_no_plan_failure` and
        the signature-format migration note on ``_merge_outcome_signature``.
        """
        assert self.artifacts is not None

        metadata = self.task.get('metadata') or {}

        # PRD § 9.8 — train members bypass this counter entirely.
        # The loop-guard (γ₂/task 1523) owns train verify-phase thrash;
        # the train-merge worker owns the merge phase.  The counter adds
        # no value for train members and risks false-trips on legitimate
        # merge-deferred → merge-deferred re-stamps from sibling activity.
        if isinstance(metadata.get('train'), dict):
            logger.debug(
                'Task %s: train member — bypassing infra-resume thrash counter '
                '(loop-guard owns train thrash)',
                self.task_id,
            )
            return None

        # Determine the category of the most recent resolved L0 (the one
        # the steward just handled).  If no escalation queue is wired up
        # (e.g. eval mode), we cannot classify — fall through.
        recent_category: str | None = None
        if self.escalation_queue:
            resolved = [
                e
                for e in self.escalation_queue.get_by_task(self.task_id)
                if e.level == 0 and e.status == 'resolved'
            ]
            if resolved:
                resolved.sort(
                    key=lambda e: e.resolved_at or e.timestamp, reverse=True,
                )
                recent_category = resolved[0].category

        # Iteration-log entry count is the progress signal.
        iter_entries, _ = self.artifacts.read_iteration_log()
        current_iter_count = len(iter_entries)

        ledger = _build_retry_ledger(metadata)

        verdict = _evaluate_infra_resume(
            ledger, current_iter_count, recent_category,
            self.config.max_consecutive_infra_resumes,
        )

        # Read-modify-write: see _merge_fresh_metadata for the merge policy.
        new_metadata = await self._merge_fresh_metadata(
            metadata, log_context='infra-resume thrash counter',
        )
        # In-memory ledger intentionally wins over any backend retry_ledger;
        # the backend overlay above is only for non-counter keys (e.g. memory_hints).
        new_metadata['retry_ledger'] = verdict.ledger.model_dump()
        self.task['metadata'] = new_metadata
        try:
            await self.scheduler.update_task(
                self.task_id, metadata=new_metadata,
            )
        except Exception as exc:  # noqa: BLE001 — counter can't be trusted; escalate.
            logger.warning(
                'Task %s: failed to persist infra-resume thrash counter; '
                'escalating to human (counter cannot be trusted): %s',
                self.task_id, exc,
            )
            return await self._mark_blocked(
                f'Failed to persist infra-resume thrash counter: {exc}',
                detail=f'category={recent_category!r}',
                escalate_to_human=True,
            )

        if verdict.escalate:
            counter = verdict.ledger.consecutive_infra_resume_failures
            logger.warning(
                'Task %s: consecutive_infra_resume_failures=%d at threshold '
                '%d — infra-issue thrash confirmed; escalating to human',
                self.task_id, counter,
                self.config.max_consecutive_infra_resumes,
            )
            return await self._mark_blocked(
                f'Repeated infra-issue resume thrash (counter={counter})',
                detail=(
                    f'category={recent_category!r}, '
                    f'iteration_log_entries={current_iter_count}, '
                    f'last_iteration_log_entries='
                    f'{ledger.last_infra_resume_iteration_count}'
                ),
                escalate_to_human=True,
            )

        return None

    def _collect_granted_files(self) -> list[str]:
        """Union ``granted_files`` across every resolved escalation for this
        task (task 2505) — the steward's structured scope-expansion grant
        stamped via ``resolve_issue(..., granted_files=[...])`` when
        resolving a ``scope_violation`` L0 with ``action='resume'``.

        Order-preserving de-duplication across ALL resolved records (not
        just the most recently resolved one), so a grant from an earlier
        resolution in this task's history is never dropped on a later
        resume-loop re-entry. Returns ``[]`` when no ``escalation_queue`` is
        wired (eval mode) — the caller's union with ``plan.files`` is then a
        no-op.
        """
        if not self.escalation_queue:
            return []
        seen: set[str] = set()
        granted: list[str] = []
        for esc in self.escalation_queue.get_by_task(self.task_id):
            if esc.status != 'resolved':
                continue
            for f in esc.granted_files:
                if f not in seen:
                    seen.add(f)
                    granted.append(f)
        return granted

    async def _check_merge_outcome_thrash(
        self,
        prev_signature: str | None,
        current_signature: str,
    ) -> WorkflowOutcome | None:
        """Fix 3 — detect & escalate repeated steward-resolved merge-phase loops.

        Mirrors :meth:`_check_infra_resume_thrash` for the merge-retry loop.
        Called from the merge-phase loop after ``_submit_to_merge_queue``
        returns ``REQUEUED`` (steward resolved an L0 and the loop is about
        to resubmit).  If the merge-outcome signature matches the previous
        attempt, increment ``consecutive_merge_thrash`` inside the typed
        ``metadata.retry_ledger`` blob (see :func:`_evaluate_merge_thrash`).
        At ``max_consecutive_merge_thrash``, route to
        ``_mark_blocked(escalate_to_human=True)`` instead of resubmitting.

        ``current_signature`` is a sha256-short fingerprint of the blocked
        ``MergeOutcome.reason``; full text would bloat metadata when verify
        failure reports run multi-kilobyte.

        Returns ``WorkflowOutcome.BLOCKED`` at threshold; ``None`` to fall
        through to the resubmit.  Persist failure escalates to a human
        immediately rather than logging and proceeding, for the same reason
        as the no-plan and infra-resume guards.
        """
        metadata = self.task.get('metadata') or {}

        # PRD § 9.8 — train members bypass this counter entirely.
        # The train-merge worker owns the merge phase for train members;
        # the counter adds no value and risks false-trips on legitimate
        # merge-deferred → merge-deferred re-stamps.
        if isinstance(metadata.get('train'), dict):
            logger.debug(
                'Task %s: train member — bypassing merge-outcome thrash counter '
                '(train-merge worker owns merge phase)',
                self.task_id,
            )
            return None

        ledger = _build_retry_ledger(metadata)

        verdict = _evaluate_merge_thrash(
            ledger, prev_signature, current_signature,
            self.config.max_consecutive_merge_thrash,
        )

        # No backend overlay here (unlike no-plan/infra-resume) — preserves
        # the existing plain-dict merge policy for this guard.
        new_metadata = dict(metadata)
        new_metadata['retry_ledger'] = verdict.ledger.model_dump()
        self.task['metadata'] = new_metadata
        try:
            await self.scheduler.update_task(
                self.task_id, metadata=new_metadata,
            )
        except Exception as exc:  # noqa: BLE001 — counter can't be trusted; escalate.
            logger.warning(
                'Task %s: failed to persist merge-thrash counter; '
                'escalating to human (counter cannot be trusted): %s',
                self.task_id, exc,
            )
            return await self._mark_blocked(
                f'Failed to persist merge-thrash counter: {exc}',
                detail=f'merge_outcome_signature={current_signature}',
                escalate_to_human=True,
            )

        if verdict.escalate:
            counter = verdict.ledger.consecutive_merge_thrash
            logger.warning(
                'Task %s: consecutive_merge_thrash=%d at threshold %d — '
                'merge-phase thrash confirmed; escalating to human',
                self.task_id, counter,
                self.config.max_consecutive_merge_thrash,
            )
            root_cause = (
                f'merge-outcome-thrash:{self._last_merge_failure_category}:'
                f'{RetryLedger.normalize_cause_hint(self._last_merge_failure_cause_hint)}'
            )
            return await self._mark_blocked(
                f'Repeated merge-phase thrash (counter={counter})',
                detail=f'merge_outcome_signature={current_signature}',
                escalate_to_human=True,
                root_cause=root_cause,
            )
        return None

    async def _stamp_first_merge_enqueue(self) -> float:
        """Stamp the write-once first-submission wall-clock epoch into task metadata.

        Returns the (possibly pre-existing) epoch as a float, ensuring:

        * The value is written exactly once — subsequent calls on the same task
          (resubmit, post-restart re-dispatch) return the original value unchanged.
        * The value is persisted to the backend so it survives process restart.
        * Persistence failure is non-fatal (logged, does not block the merge).

        The epoch lives inside the typed ``metadata.retry_ledger`` blob (see
        :class:`shared.task_metadata.RetryLedger`) as ``str(epoch)`` — α typed
        ``RetryLedger.merge_first_enqueued_at`` as ``str | None`` even though the
        runtime value is a float epoch; ``str()``/``float()`` round-trip losslessly.

        Algorithm:
        1. **Fast-path (in-memory ledger)** — if ``metadata.retry_ledger`` already
           carries a parseable ``merge_first_enqueued_at``, return it as a float
           immediately (no I/O).
        2. **Fast-path (legacy top-level)** — no ledger value, but a numeric
           top-level ``merge_first_enqueued_at`` exists (a task stamped by
           pre-migration code): adopt it — restore it under ``retry_ledger`` in
           memory and return it as a float, no re-stamp and no persist (the
           value is already durable on the backend under the old key).
        3. **Slow-path (backend read)** — call :meth:`_merge_fresh_metadata` to
           overlay backend-persisted keys; if the backend ledger already carries
           the value, adopt it into the in-memory copy and return (no persist
           needed).
        4. **Stamp** — call ``time.time()``, write ``str(epoch)`` into the merged
           ledger (preserving its other fields), update the in-memory task, and
           best-effort persist via ``scheduler.update_task`` (wrapped in a logged
           try/except).

        ζ (task 1887) consumes this value at ``_pop_next_pickable`` for aging-
        priority ordering and owns the ``enqueued_at`` fallback for legacy
        ``None`` values.

        **Single-writer assumption** — per-task workflows are serialised by
        design, so only one coroutine calls this helper for a given task at
        any moment.  If that assumption ever changes (e.g. parallel submit
        paths for the same task), the slow-path check-then-stamp sequence
        would need an explicit per-task lock to remain strictly write-once.
        """
        metadata = self.task.get('metadata') or {}
        ledger = metadata.get('retry_ledger') or {}

        # Fast-path: in-memory ledger already carries a parseable value.
        existing = ledger.get('merge_first_enqueued_at')
        if existing is not None:
            try:
                return float(existing)
            except (TypeError, ValueError):
                pass  # corrupt — fall through to the legacy/backend/stamp path

        # Fast-path: legacy top-level value from pre-migration code. It is
        # already durable on the backend under the old key, so just restore
        # it into the ledger shape in memory — no persist needed.
        legacy_existing = metadata.get('merge_first_enqueued_at')
        if isinstance(legacy_existing, (int, float)):
            adopted_ledger = dict(ledger)
            adopted_ledger['merge_first_enqueued_at'] = str(legacy_existing)
            new_metadata = dict(metadata)
            new_metadata['retry_ledger'] = adopted_ledger
            self.task['metadata'] = new_metadata
            return float(legacy_existing)

        # Slow-path: consult backend to adopt a persisted value and protect
        # memory_hints on the subsequent write.
        fresh = await self._merge_fresh_metadata(
            metadata, log_context='merge_first_enqueued_at stamp',
        )
        fresh_ledger = fresh.get('retry_ledger') or {}
        backend_existing = fresh_ledger.get('merge_first_enqueued_at')
        if backend_existing is not None:
            try:
                adopted = float(backend_existing)
            except (TypeError, ValueError):
                adopted = None
            if adopted is not None:
                # Backend already has the value; adopt into in-memory copy, no persist.
                self.task['metadata'] = fresh
                return adopted

        # Neither in-memory nor backend carries the value — this is the first submit.
        stamped = time.time()
        new_ledger = dict(fresh_ledger)
        new_ledger['merge_first_enqueued_at'] = str(stamped)
        fresh['retry_ledger'] = new_ledger
        self.task['metadata'] = fresh
        try:
            await self.scheduler.update_task(
                self.task_id, metadata=fresh,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, log and proceed
            logger.warning(
                'Task %s: failed to persist merge_first_enqueued_at: %s',
                self.task_id, exc,
            )
        return stamped

    async def _stamp_merge_retry_pending(self) -> None:
        """Persist a durable merge-phase-resume obligation into task metadata.

        Best-effort record that "this task owes an in-place merge resubmission".
        ``_requeue``'s ``merge_phase=True`` path returns REQUEUED while leaving
        the task ``in-progress`` for an in-RAM in-place retry; without this stamp
        a restart mid-retry loses the obligation entirely (Reify 5166).
        Persisting ``{branch_head, base_sha, resolved_at}`` lets
        :meth:`_resume_merge_retry_if_pending` reconstruct it on re-dispatch and
        jump straight back to the merge phase when the post-rebase worktree HEAD
        still equals ``branch_head``.

        Mirrors :meth:`_stamp_first_merge_enqueue`'s durability contract: read
        fresh backend metadata (via :meth:`_merge_fresh_metadata`) so a
        concurrent write (``retry_ledger``, ``memory_hints``) is not clobbered,
        update the in-memory task, and persist via ``scheduler.update_task``
        inside a logged try/except. A persistence failure must never crash the
        block/resume path.
        """
        try:
            rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=self.worktree)
        except Exception as exc:  # noqa: BLE001 — best-effort, log and skip
            logger.warning(
                'Task %s: could not read worktree HEAD for merge_retry_pending '
                'stamp (skipping durable stamp): %s', self.task_id, exc,
            )
            return
        branch_head = out.strip()
        if rc != 0 or not branch_head:
            logger.warning(
                'Task %s: git rev-parse HEAD failed (rc=%s) for '
                'merge_retry_pending stamp — skipping durable stamp',
                self.task_id, rc,
            )
            return
        try:
            base_sha = await self.git_ops.get_main_sha()
        except Exception as exc:  # noqa: BLE001 — base_sha is advisory context only
            logger.warning(
                'Task %s: could not read main SHA for merge_retry_pending stamp: %s',
                self.task_id, exc,
            )
            base_sha = ''

        metadata = self.task.get('metadata') or {}
        fresh = await self._merge_fresh_metadata(
            metadata, log_context='merge_retry_pending stamp',
        )
        fresh['merge_retry_pending'] = {
            'branch_head': branch_head,
            'base_sha': base_sha,
            'resolved_at': datetime.now(UTC).isoformat(),
        }
        self.task['metadata'] = fresh
        try:
            await self.scheduler.update_task(self.task_id, metadata=fresh)
        except Exception as exc:  # noqa: BLE001 — durability best-effort, never fatal
            logger.warning(
                'Task %s: failed to persist merge_retry_pending stamp '
                '(retry obligation not durable this cycle): %s',
                self.task_id, exc,
            )

    async def _clear_merge_retry_pending(self) -> None:
        """Remove the durable merge-retry obligation from task metadata (best-effort).

        Called by :meth:`_resume_merge_retry_if_pending` on consume — both on a
        HEAD match (before delegating to the merge phase) and on a HEAD mismatch
        (the stamp can never match again once the branch moved) — and by
        :meth:`_merge_and_finalise` on merge SUCCESS, so the happy-path
        stamp/clear lifecycle is symmetric (a merged/DONE task carries no stale
        ``merge_retry_pending``).

        ``metadata_mode='replace'`` is REQUIRED, not incidental: the default
        'merge' mode preserves keys omitted from the payload
        (``{**existing, **incoming}``), so a mode-less write would log a removal
        it never performs.  Only a whole-blob replace can actually DELETE a key
        — which is why the payload is built as a full-dict
        :meth:`_merge_fresh_metadata` read-modify-write that carries every other
        key through.  (Same rule, same reason as the harness sibling
        ``_clear_merge_retry_pending_for_restart``.)

        That duplication is knowingly left in place: the delete-by-omission
        subtlety belongs in ONE place next to ``metadata_mode``, but the single
        home for it is ``Scheduler``/the fused-memory metadata contract, which is
        outside this task's lock scope.  Pending task 3151 (targeted
        ``delete_keys`` mode, filed with scheduler.py in scope) is the vehicle;
        when it lands, both clears should delegate to it and this whole-blob
        dance — plus the ``type: ignore`` below — goes away.

        Because replace DELETES every omitted key, the read is made with
        ``require_fresh=True``: a failed backend refresh skips the write entirely
        instead of replacing the blob with an in-memory-only payload that would
        destroy backend-only keys.  Neither a refresh nor a persist failure may
        crash the resume path, and neither loses anything permanently — the stamp
        simply survives, and a later dispatch retries the clear (a subsequent
        re-block also re-stamps a fresh obligation), so a lost clear is
        self-healing where clobbered metadata would not be.

        Two failure-visibility rules make that "the stamp simply survives" claim
        true of EVERY arm, not just the refresh one (amendment):

        * ``scheduler.update_task`` does not raise on an MCP-level failure — it
          logs and returns ``False``.  So the boolean is checked: an unlanded
          write logs the same warning as an exception instead of falling through
          to a success log an operator would read as proof the stamp is gone.
        * ``self.task['metadata']`` is only overwritten AFTER a confirmed
          persist.  Clearing it first would desynchronise memory from the backend
          on a failed write, and the merge-success clear in
          :meth:`_merge_and_finalise` would then early-return on the missing
          in-memory key — landing a DONE task that still carries the stamp
          server-side.  Keeping them in agreement means that same-dispatch clear
          is a real retry.

        ACCEPTED residual race (review amendment): ``'replace'`` is not
        unconditionally safe even when the read SUCCEEDED. Read and write
        straddle an ``await`` boundary, so a whole-blob overwrite silently
        drops any key another process wrote in between (``'merge'`` could not
        lose an unsupplied key that way). See
        :meth:`_clear_merge_phase_entered` for the full statement of the
        window, its mitigations, and why it cannot be eliminated without a
        targeted key-delete backend mode (the backend accepts only
        ``{'merge', 'additive', 'replace'}``) — task 3151 above is that
        vehicle for both clears.

        Idempotent no-op when no stamp is present: returns immediately without a
        fresh-metadata read or backend write, so it is cheap and safe to call
        unconditionally on every merge success (the common case never stamped).
        """
        metadata = self.task.get('metadata') or {}
        if 'merge_retry_pending' not in metadata:
            return
        try:
            # require_fresh: a 'replace' write deletes every omitted key, so an
            # unverified (in-memory-only) payload would clobber backend-only keys.
            # A refusal lands in the same best-effort handler as a persist failure,
            # skipping the write entirely rather than replacing on a guess.
            fresh = await self._merge_fresh_metadata(
                metadata, log_context='merge_retry_pending clear',
                require_fresh=True,
            )
            fresh.pop('merge_retry_pending', None)
            ok = await self.scheduler.update_task(
                self.task_id, metadata=fresh,
                # SchedulerFacade's Protocol (scheduler.py:681) predates
                # metadata_mode and only declares `append`; the concrete
                # Scheduler.update_task (scheduler.py:3757) and the harness
                # sibling clear already pass it. Same ignore + rationale as the
                # task-2533 sites above (workflow.py:4464, :10598); widening the
                # Protocol is outside task 3024's scope — see escalate_info.
                metadata_mode='replace',  # type: ignore[reportCallIssue]
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, never fatal
            logger.warning(
                'Task %s: could not clear merge_retry_pending (metadata refresh or '
                'persist failed) — leaving the stamp in place; a later dispatch '
                'retries the clear, and the resume conflict probe and the harness '
                'restart clear are independent remedies: %s',
                self.task_id, exc,
            )
            return
        if not ok:
            # update_task swallows MCP-level failures and returns False, so
            # without this branch a rejected write is indistinguishable from a
            # landed one and the stamp's survival goes unlogged.
            logger.warning(
                'Task %s: could not clear merge_retry_pending (scheduler.update_task '
                'reported failure) — leaving the stamp in place; a later dispatch '
                'retries the clear, and the resume conflict probe and the harness '
                'restart clear are independent remedies',
                self.task_id,
            )
            return
        # Backend confirmed: mirror the delete in memory. Doing this only after a
        # landed write keeps memory and backend in agreement, so a failed clear
        # leaves the in-memory stamp for the merge-success clear to retry.
        self.task['metadata'] = fresh

    async def _stamp_merge_phase_entered(self) -> dict | None:
        """Persist a durable merge-phase-liveness stamp into task metadata.

        Task 2991: the restart-survivable analog of ``routing.latest`` for the
        orphan-L0 divergence reaper. The pre-enqueue MERGE loop (rebase +
        scoped verify + queue submit) makes NO LLM calls, so it never refreshes
        ``metadata.routing.latest.decided_at`` — the reaper's task-2931
        ``_has_fresh_dispatch`` gate cannot see a legitimately-live merge-stage
        task and false-promotes its scope-invariant L0 to a human-facing L1
        (cluster esc-2789-22). This stamp records ``{'entered_at': <iso>}``;
        the reaper's ``_has_fresh_merge_phase`` gate defers a divergence L0
        whose task carries a fresh stamp. Being durable task metadata it
        survives an orchestrator restart (unlike the per-process
        ``Scheduler._merge_phase_at``), covering the exact redispatch window in
        which the false positive wedged 2789/2885.

        Written at TWO points in :meth:`_run_merge_phase`, so a fresh stamp
        exists at the start of every pre-enqueue window: (1) on merge entry,
        before ``_check_scope_invariant`` files the divergence L0 and before
        the gating bail — hence refreshed on every (re-)dispatch into merge
        phase, including passes that immediately bail to ESCALATED; and (2) at
        the top of each retry attempt, beside the in-memory
        ``note_merge_phase_entered``, because ``_submit_to_merge_queue``
        discharges the stamp at the enqueue boundary INSIDE that loop
        (review-fix R2-B).

        Mirrors :meth:`_stamp_merge_retry_pending`'s durability contract but
        carries only a timestamp (no worktree HEAD / base SHA): read fresh
        backend metadata (via :meth:`_read_fresh_backend_metadata`, the same
        read :meth:`_merge_fresh_metadata` performs) so a concurrent write
        (``retry_ledger``, ``memory_hints``) is not clobbered, update the
        in-memory task, and persist via ``scheduler.update_task`` inside a
        logged try/except. A persistence failure must never crash the merge
        path — a lost stamp is self-healing (re-written on the next merge entry,
        and at worst the reaper promotes rather than silently suppresses).

        Returns:
            The backend metadata blob this stamp just read (``{}`` when the
            backend blob is empty), or ``None`` when it could not be read.
            The merge-entry caller threads it straight into
            :meth:`_check_scope_invariant`, which needs the SAME blob — that
            saves a second ``get_task`` round-trip on the merge hot path and
            makes the stamp and the scope check evaluate one snapshot rather
            than two taken at different instants (review amendment).
        """
        metadata = self.task.get('metadata') or {}
        backend = await self._read_fresh_backend_metadata(
            log_context='merge_phase_liveness stamp',
        )
        # Same merge policy as _merge_fresh_metadata: in-memory first, backend
        # overlaid (a backend-side addition is an external write that must not
        # be discarded); an unreadable backend (None) falls back to in-memory.
        fresh = {**metadata, **(backend or {})}
        fresh['merge_phase_liveness'] = {
            'entered_at': datetime.now(UTC).isoformat(),
        }
        self.task['metadata'] = fresh
        try:
            await self.scheduler.update_task(self.task_id, metadata=fresh)
        except Exception as exc:  # noqa: BLE001 — durability best-effort, never fatal
            logger.warning(
                'Task %s: failed to persist merge_phase_liveness stamp '
                '(merge-phase liveness not durable this cycle): %s',
                self.task_id, exc,
            )
        return backend

    async def _clear_merge_phase_entered(self) -> None:
        """Remove the durable merge-phase-liveness stamp from task metadata.

        Task 2991 symmetric clear (mirrors :meth:`_clear_merge_retry_pending`):
        called at the durable-enqueue boundary in
        :meth:`_submit_to_merge_queue` (alongside ``scheduler.clear_merge_phase``)
        and on merge SUCCESS in :meth:`_merge_and_finalise`, so a just-enqueued
        / just-merged task does not carry a fresh stamp that would briefly defer
        an unrelated stranded divergence orphan.

        Deletion requires ``metadata_mode='replace'`` (review-fix R2-A). A plain
        ``scheduler.update_task(metadata=...)`` resolves to
        ``metadata_mode='merge'`` (scheduler.py), which the backend implements
        as a shallow last-write-wins ``{**old, **new}`` where omitted keys are
        PRESERVED — so popping a key out of the payload is a backend NO-OP.
        ``'replace'`` is the mode the scheduler docstring designates for
        delete-by-omission; it is a whole-blob overwrite, safe here ONLY because
        the payload is built from a backend blob that was just re-read into
        ``fresh``. Hence the guard: when
        :meth:`_read_fresh_backend_metadata` reports a failed read the durable
        write is SKIPPED (in-memory clear only), because a ``'replace'`` built
        from in-memory-only metadata would delete backend-only keys
        (``memory_hints``, ``_causation_id``) — the #4271 sibling-clobber bug.
        The bounded cost of skipping is that the reaper may keep deferring for
        up to ``orphan_l0_merge_phase_freshness_secs`` before the stale stamp
        ages out (deferral, never suppression).

        ACCEPTED residual race (review amendment) — ``'replace'`` is NOT
        unconditionally safe once the read succeeded. This is a
        read-modify-write across an ``await`` boundary (two MCP round-trips),
        and a whole-blob overwrite drops any key another process wrote in
        between; ``'merge'`` mode could not lose an unsupplied key that way.
        At risk in that window: ``memory_hints`` re-attached by Stage-2
        reconciliation, ``_causation_id``, and ``routing.latest`` — the input
        to the reaper's SIBLING ``_has_fresh_dispatch`` gate, so a clobber
        there would re-open the task-2931 false positive. Mitigations, in
        order: (1) read and write are back-to-back with no intervening awaits,
        which is the narrowest this can be without a new backend primitive;
        (2) this workflow issues no LLM call while the clear runs, so its own
        routing mirror is not a competing writer (an external writer such as
        reconciliation still can be); (3) the write is skipped entirely when
        the read failed or returned a corrupt blob. It cannot be ELIMINATED
        here: the backend accepts only ``{'merge', 'additive', 'replace'}``
        (``sqlite_task_backend._METADATA_MODES``) and offers no targeted
        key-delete. A ``delete_keys`` mode would make both clears atomic and
        is filed as follow-up work (fused-memory backend + scheduler, outside
        this task's module scope). Pinned by
        ``test_clear_replace_loses_concurrent_backend_write_in_read_window``.

        Deliberately NOT cleared defensively in the harness ``_run_slot``
        finally (unlike the in-memory ``clear_merge_phase`` grace stamp): an
        abnormal exit must LEAVE the durable stamp so the reaper keeps deferring
        across the crash/restart/redispatch window (fix-direction b). A
        persistence failure must never crash the merge path; a subsequent merge
        (re-)entry re-stamps, so a lost clear is self-healing.

        Idempotent no-op when no stamp is present: returns immediately without a
        fresh-metadata read or backend write, so it is cheap and safe to call
        unconditionally on every enqueue / merge success (the common case never
        stamped — the stamp is written only at merge entry).
        """
        metadata = self.task.get('metadata') or {}
        if 'merge_phase_liveness' not in metadata:
            return
        backend = await self._read_fresh_backend_metadata(
            log_context='merge_phase_liveness clear',
        )
        if backend is None:
            logger.warning(
                'Task %s: skipping durable merge_phase_liveness clear — could '
                'not re-read backend metadata, and a replace-mode write built '
                'from in-memory metadata alone would clobber backend-only keys '
                '(clearing in-memory only; the stale stamp ages out of the '
                'reaper grace on its own)',
                self.task_id,
            )
            self.task['metadata'] = {
                k: v for k, v in metadata.items() if k != 'merge_phase_liveness'
            }
            return
        fresh = {**metadata, **backend}
        fresh.pop('merge_phase_liveness', None)
        self.task['metadata'] = fresh
        try:
            await self.scheduler.update_task(
                self.task_id,
                metadata=fresh,
                metadata_mode='replace',  # type: ignore[reportCallIssue]
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, never fatal
            logger.warning(
                'Task %s: failed to persist merge_phase_liveness clear: %s',
                self.task_id, exc,
            )

    def _merge_outcome_signature(self) -> str:
        """Return a 16-hex-char signature for the current merge-block fingerprint.

        Delegates to the module-level _compute_merge_outcome_signature() helper
        using the instance's _last_merge_failure_* fields.

        Keys on (category, normalised cause_hint) when either structured field
        is populated — the same stable shape the in-branch contagion guard uses
        (#1645, workflow.py:3544).  Falls back to sha256(normalised_reason) when
        both structured fields are empty (e.g. git ff/merge errors with no
        VerifyResult), which is a strict improvement over the old raw-reason hash.

        Deploy note: this method replaced the old sha256(raw_reason) scheme. Any
        task in-flight at deploy time will have a persisted
        metadata.last_merge_outcome_signature in the old format; on the next merge
        failure prev_signature (old) != current_signature (new), so
        consecutive_merge_thrash resets to 1 — at most one extra thrash cycle
        before the counter re-accumulates. Self-healing and benign.

        NOTE: do NOT call this from _auto_heal_main_health — the
        MAIN_HEALTH_RED_REASON_PREFIX fast-path returns before the generic blocked
        path that sets _last_merge_failure_*; those fields are at their __init__
        defaults, so this method would return a constant hash.  Use
        _compute_merge_outcome_signature(category, cause_hint, reason) with the
        outcome's fields instead.
        """
        return _compute_merge_outcome_signature(
            self._last_merge_failure_category,
            self._last_merge_failure_cause_hint,
            self._last_merge_block_reason or '',
        )

    async def _handle_blocking_dep_report(
        self, *, rebase_retry_used: bool,
    ) -> WorkflowOutcome | None:
        """Process a ``.task/blocking_dependency.json`` report from the architect.

        Caller has already verified the artifact exists.

        - If the named dep is non-terminal: register the Taskmaster
          dependency, clear the artifact, and return REQUEUED.  Status
          stays ``pending`` — the scheduler's dep-check keeps the task
          from dispatching until the dep is ``done``/``cancelled``.
        - If the dep is already terminal: a race occurred (dep landed
          between architect-start and report).  Rebase onto current
          main, clear the artifact, and return ``None`` so the caller
          retries the architect once.  When ``rebase_retry_used`` is
          already ``True``, return BLOCKED to prevent unbounded retry.
        - If the artifact is malformed (missing ``depends_on_task_id``):
          clear it and return BLOCKED.
        """
        assert self.artifacts is not None and self.worktree is not None
        report = self.artifacts.read_blocking_dependency()
        assert report is not None  # caller must have verified

        dep_id = str(report.get('depends_on_task_id') or '').strip()
        reason = report.get('reason', '')
        if not dep_id:
            logger.error(
                'Task %s: blocking_dependency.json missing depends_on_task_id; '
                'treating as planning failure', self.task_id,
            )
            self.artifacts.clear_blocking_dependency()
            return await self._mark_blocked(
                'Architect wrote malformed blocking_dependency.json '
                '(missing depends_on_task_id)',
                detail=json.dumps(report, indent=2)[:2000],
            )

        dep_status = await self.scheduler.get_status(dep_id)

        if dep_status in TERMINAL_STATUSES:
            # Race: dep landed between architect-start and report.
            self.artifacts.clear_blocking_dependency()
            if rebase_retry_used:
                return await self._mark_blocked(
                    f'Architect repeatedly reported blocking dependency on '
                    f'task {dep_id} which is already {dep_status} '
                    f'(rebase-retry already used)',
                    detail=f'reason: {reason}\nfull_report: '
                           f'{json.dumps(report, indent=2)[:1500]}',
                )
            logger.info(
                'Task %s: architect reported dep on task %s but it is %s — '
                'rebasing onto main and retrying architect once',
                self.task_id, dep_id, dep_status,
            )
            rebased = await self.git_ops.rebase_onto_main(self.worktree)
            new_main_sha = await self.git_ops.get_main_sha()
            if rebased:
                self.artifacts.update_base_commit(new_main_sha)
            else:
                logger.warning(
                    'Task %s: rebase onto main failed during blocking-dep '
                    'recovery — retrying architect on stale base anyway',
                    self.task_id,
                )
            return None

        # Non-terminal dep — register the Taskmaster dependency.
        logger.info(
            'Task %s: architect reported blocking dependency on task %s '
            '(status=%s); registering dep and requeueing — reason: %s',
            self.task_id, dep_id, dep_status, reason[:200],
        )
        try:
            await self.scheduler.dispatch_tool(
                'add_dependency',
                {
                    'id': self.task_id,
                    'depends_on': dep_id,
                    'project_root': str(self.config.project_root),
                },
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, log and proceed
            logger.warning(
                'Task %s: add_dependency(%s -> %s) failed: %s — proceeding '
                'with requeue anyway',
                self.task_id, self.task_id, dep_id, exc,
            )

        self.artifacts.clear_blocking_dependency()
        await self.scheduler.set_task_status(self.task_id, 'pending')
        return WorkflowOutcome.REQUEUED

    async def _handle_already_done_report(self) -> WorkflowOutcome:
        """Process a ``.task/already_done.json`` report from the architect.

        Caller has already verified the artifact exists.

        Validation: ``commit`` must be non-empty and reachable from main.
        ``git merge-base --is-ancestor`` returns false for both unknown SHAs
        and SHAs not on main, so this single check covers both.

        On success: set task status to ``done`` with provenance pointing
        at the architect-named commit, return ``DONE``.
        On validation failure: clear the artifact, route to ``_mark_blocked``
        without escalating to a human — this is an architect mistake
        (wrong/missing commit), not an unworkable task.
        """
        assert self.artifacts is not None
        report = self.artifacts.read_already_done()
        assert report is not None  # caller must have verified

        commit = str(report.get('commit') or '').strip()
        evidence = str(report.get('evidence') or '')

        self.artifacts.clear_already_done()

        if not commit:
            return await self._mark_blocked(
                'Architect wrote malformed already_done.json '
                '(missing commit)',
                detail=json.dumps(report, indent=2)[:2000],
            )

        main_sha = await self.git_ops.get_main_sha()
        on_main = await self.git_ops.is_ancestor(commit, main_sha)
        if not on_main:
            return await self._mark_blocked(
                f'Architect reported task already done at {commit[:12]} '
                f'but commit is not reachable from main',
                detail=(
                    f'commit: {commit}\nmain_sha: {main_sha}\n'
                    f'evidence: {evidence}'
                )[:2000],
            )

        logger.info(
            'Task %s: architect reported task already done at %s — '
            'setting status done with provenance',
            self.task_id, commit[:12],
        )
        self._enter_phase(WorkflowState.DONE)
        await self._reconcile_metadata_files_for_done()
        try:
            await self.scheduler.mark_done(
                self.task_id,
                kind='found_on_main',
                sha=commit,
                note=(
                    f'architect-reported task already on main; '
                    f'evidence: {evidence[:400]}'
                ),
            )
        except SetTaskStatusRejected as exc:
            # Architect's claim is contradicted by the persistence layer —
            # L1-worthy: a steward retry can't reconcile a phantom-done or
            # provenance ancestor mismatch.
            logger.error(
                'Task %s: mark_done rejected after already_done report — %s: %s',
                self.task_id, exc.error_code, exc.raw,
            )
            return await self._mark_blocked(
                f'Architect already_done rejected: {exc.error_code} — {exc.raw}',
                escalate_to_human=True,
            )
        return WorkflowOutcome.DONE

    async def _handle_unactionable_task_report(self) -> WorkflowOutcome:
        """Process a ``.task/unactionable_task.json`` report from the architect.

        Caller has already verified the artifact exists.

        Stops the steward early to close the small async window where a
        stale L0 from a prior PLAN attempt could be processed concurrently
        with our L1 submission.  The ``finally`` block in ``run()`` also
        stops the steward, so this is defense-in-depth.

        Then short-circuits to ``_mark_blocked(escalate_to_human=True)``,
        which submits an L1 directly without invoking the steward — the
        steward consumes only L0 escalations and cannot fix a broken spec.
        """
        assert self.artifacts is not None
        report = self.artifacts.read_unactionable_task()
        assert report is not None  # caller must have verified

        reason = str(report.get('reason') or '').strip()
        evidence = str(report.get('evidence') or '')

        if self._steward:
            await self._steward.stop()
            self._steward = None

        self.artifacts.clear_unactionable_task()

        if not reason:
            return await self._mark_blocked(
                'Architect wrote malformed unactionable_task.json '
                '(missing reason)',
                detail=json.dumps(report, indent=2)[:2000],
                escalate_to_human=True,
            )

        return await self._mark_blocked(
            f'Architect reported task unactionable: {reason}',
            detail=f'reason: {reason}\nevidence: {evidence}'[:2000],
            escalate_to_human=True,
        )

    async def _handle_false_premise_report(self) -> WorkflowOutcome:
        """Process a ``.task/false_premise.json`` report from the architect.

        Caller has already verified the artifact exists.

        Stops the steward early to close the small async window where a
        stale L0 from a prior PLAN attempt could be processed concurrently
        with our L1 submission.  The ``finally`` block in ``run()`` also
        stops the steward, so this is defense-in-depth.

        Then short-circuits to ``_mark_blocked(escalate_to_human=True,
        category='design_concern')``, which submits a level-1 design_concern
        escalation directly without invoking the steward — only a human/curator
        can re-spec a test premise or relocate a signal to a different task.
        """
        assert self.artifacts is not None
        report = self.artifacts.read_false_premise()
        assert report is not None  # caller must have verified

        classification = str(report.get('classification') or '')
        premise = str(report.get('premise') or '').strip()
        evidence = str(report.get('evidence') or '')
        proposed_resolution = str(report.get('proposed_resolution') or '')

        if self._steward:
            await self._steward.stop()
            self._steward = None

        if not premise:
            result = await self._mark_blocked(
                'Architect wrote malformed false_premise.json '
                '(missing premise)',
                detail=json.dumps(report, indent=2)[:2000],
                escalate_to_human=True,
                category='design_concern',
            )
        else:
            result = await self._mark_blocked(
                f'Architect reported false RED-test premise: {premise}',
                detail=(
                    f'classification: {classification}\n'
                    f'evidence: {evidence}\n'
                    f'proposed_resolution: {proposed_resolution}'
                )[:2000],
                escalate_to_human=True,
                category='design_concern',
            )
        # Clear only after the L1 escalation has been successfully submitted —
        # if _mark_blocked raises, the artifact survives for the next retry.
        self.artifacts.clear_false_premise()
        return result

    def _reviewer_config_fingerprint(self) -> str | None:
        """Stable digest of the review-input identity for the verdict cache.

        Captures what determines the reviewers that WOULD run — the active
        reviewer roster (``ALL_REVIEWERS``) plus, per reviewer role, its
        configured model (``config.models`` at the ``_config_key`` collapsed
        key the resolver itself reads — reviewer* → ``reviewer``), its
        role-default model, and any per-task ``metadata.model_overrides`` pin
        (keyed by the FULL role name, matching the resolver's override layer)
        — so the tree-hash verdict cache (task 2749) can invalidate on a
        roster or model-config change even when the committed tree is
        byte-identical.

        Deliberately EXCLUDES volatile per-dispatch inputs (dispatch count,
        routing tier, plan shape, live model spend / ceilings): those are not
        a change in review-input *identity*, and folding them in would make
        the fingerprint churn every dispatch and defeat the cross-dispatch
        cache this task exists to provide.

        Returns a short sha256 hex digest, or ``None`` on any error — the
        reader treats a ``None`` fingerprint as "cannot confirm config
        identity" and forces a full review (fail-safe: a redundant review,
        never a stale skip).
        """
        try:
            models_cfg = getattr(self.config, 'models', None)
            metadata = self.task.get('metadata') or {}
            overrides = metadata.get('model_overrides') or {}
            if not isinstance(overrides, dict):
                overrides = {}
            roster = []
            for role in ALL_REVIEWERS:
                # config.models uses the reviewer* → 'reviewer' collapsed key
                # (the resolver's own _config_key); metadata.model_overrides
                # uses the FULL role name (the collapsed key is inert there).
                configured = getattr(models_cfg, _config_key(role.name), None)
                override = overrides.get(role.name)
                roster.append(
                    [
                        role.name,
                        None if configured is None else str(configured),
                        str(role.default_model),
                        None if override is None else str(override),
                    ]
                )
            payload = json.dumps(sorted(roster), sort_keys=True)
            return hashlib.sha256(payload.encode()).hexdigest()[:16]
        except Exception as exc:
            logger.warning(
                'Task %s: reviewer-config fingerprint failed (%s) — verdict '
                'cache will force a full review',
                self.task_id, exc,
            )
            return None

    async def _execute_verify_review_loop(self) -> WorkflowOutcome:
        """Execute → Verify → Review loop with retry limits."""
        # Clear stale merge-failure review from prior runs — prevents
        # the review phase from re-surfacing resolved merge issues.
        if self.artifacts:
            stale_merge = self.artifacts.root / 'reviews' / 'merge.json'
            if stale_merge.exists():
                logger.info('Task %s: removing stale merge.json review', self.task_id)
                stale_merge.unlink()

        # Seed the loop counters from the persisted task-lifetime totals
        # (task 2749) so max_amendment_rounds / max_review_cycles bound the
        # WHOLE task lifetime, not each dispatch — a re-dispatch (restart
        # churn, requeue, resume) no longer grants a fresh allowance.
        #
        # OPERATOR NOTE (task 2749 amendment): because the seed persists, a
        # task that has EXHAUSTED its allowance and escalated keeps the
        # exhausted counters across a human-driven re-pend. If an operator
        # resolves the review escalation and re-pends the task (the standard
        # manual re-pend recipe), the seeded review_cycle is already at the
        # cap, so the very next blocking review re-escalates WITHOUT a fresh
        # replan. resolve_issue / resume do NOT clear these counters. This is
        # deliberate: the loop cannot distinguish churn (must keep the
        # counters) from a legitimate re-pend (should reset them) without an
        # out-of-band signal, so a fresh allowance is an EXPLICIT operator
        # action — call TaskArtifacts.clear_review_counters() (the reset
        # hook) on the re-pend to grant one. Auto-wiring that hook into the
        # scheduler/escalation resume path is out of scope for this module.
        review_cycle = (
            self.artifacts.get_review_cycles_total() if self.artifacts else 0
        )
        amendment_round = (
            self.artifacts.get_amendment_rounds_total() if self.artifacts else 0
        )
        # Consume-once scope for the single REVIEW that immediately follows an
        # amendment (task 2750): set after a successful _amend, read+cleared at
        # the top of the next loop pass.  A later blocking-replan/re-execute
        # cycle produces a materially different diff, so the captured
        # pre-amendment HEAD must never leak into that unrelated review.
        amendment_ctx: AmendmentReviewContext | None = None

        while True:
            # EXECUTE
            self._enter_phase(WorkflowState.EXECUTE)
            exec_outcome = await self._execute_iterations()
            if exec_outcome == WorkflowOutcome.ESCALATED:
                return WorkflowOutcome.ESCALATED
            if exec_outcome == WorkflowOutcome.BLOCKED:
                # Zero-output hang: use the distinct infra_issue reason instead
                # of the generic 'Execution iterations exhausted' so the escalation
                # is classified correctly and ops can distinguish deterministic
                # CLI wedges from real task failures (task 1739).
                if self._zero_output_hang_info is not None:
                    info = self._zero_output_hang_info
                    return await self._mark_blocked(
                        info['reason'],
                        detail=info['detail'],
                        category='infra_issue',
                    )
                # Progress-resume churn: mutually exclusive with the
                # zero-output hang above (at most one info dict is set per
                # _execute_iterations call — see its reset-at-entry comment).
                # Distinct infra_issue reason so ops can tell "repeatedly
                # killed a productive, converging-too-slowly session" apart
                # from both the zero-output wedge and a genuine task-failure
                # exhaustion (task 2360, reify-4827).
                if self._progress_resume_churn_info is not None:
                    info = self._progress_resume_churn_info
                    return await self._mark_blocked(
                        info['reason'],
                        detail=info['detail'],
                        category='infra_issue',
                    )
                return await self._mark_blocked('Execution iterations exhausted')

            # VERIFY + DEBUGFIX loop
            #
            # Durable verified-green checkpoint (task 2752): reset fresh each
            # loop pass.  If a durable prior-run workflow_verify green exists at
            # the CURRENT branch tip, SKIP the whole (expensive) verify/debugfix
            # loop — the branch was already verified green at this exact tip, and
            # a rebase (the only base-mover) would rewrite commits → a new tip,
            # so an unchanged tip means an unchanged tree.  The honest signal on
            # a skip is phase_skipped(verify); the VERIFY→REVIEW workflow_verify
            # re-emit is suppressed in _enter_phase (gated on
            # _verify_checkpoint_hit) so we never assert a verify that did not
            # run this cycle.  Fail-closed on every axis: no event store, a
            # checkpoint miss, a RAISING _get_head_commit (caught below →
            # tip=None → miss), an empty tip, or a green_checkpoint_at_tip read
            # error all fall through to the normal verify path
            # (green_checkpoint_at_tip never raises, and the tip fetch is
            # wrapped in try/except so a spawn failure cannot crash the loop).
            # Task 2749's tree-hash verdict cache then skips REVIEW below,
            # composing into the fast-path to merge.
            self._verify_checkpoint_hit = False
            self._enter_phase(WorkflowState.VERIFY)
            if self.event_store is not None:
                try:
                    tip = await self._get_head_commit()
                except Exception:
                    # A raising _get_head_commit (git not on PATH →
                    # FileNotFoundError, a None worktree, a transient spawn
                    # failure) must NOT crash the loop: degrade to the normal
                    # verify path (tip=None → green_checkpoint_at_tip miss →
                    # _verify_debugfix_loop runs).  This is exactly the
                    # fail-safe the comment above promises for a
                    # _get_head_commit error.
                    logger.warning(
                        'Task %s: VERIFY checkpoint tip fetch failed; running '
                        'the normal verify (fail-safe) (task 2752)',
                        self.task_id, exc_info=True,
                    )
                    tip = None
                if green_checkpoint_at_tip(
                    self.event_store, EventType.workflow_verify, self.task_id, tip,
                ):
                    self._verify_checkpoint_hit = True
                    self.event_store.emit(
                        EventType.phase_skipped,
                        task_id=self.task_id,
                        phase='verify',
                        data={'reason': 'durable_verified_green', 'tip_sha': tip},
                    )
                    logger.info(
                        'Task %s: VERIFY skipped — durable verified-green '
                        'checkpoint at tip %s (task 2752)',
                        self.task_id, tip,
                    )
            if not self._verify_checkpoint_hit:
                verify_outcome = await self._verify_debugfix_loop()
                if verify_outcome == WorkflowOutcome.ESCALATED:
                    return WorkflowOutcome.ESCALATED
                if verify_outcome == WorkflowOutcome.BLOCKED:
                    # Infra hold takes priority: route to infra_issue with
                    # escalate_to_human so the open L1 keeps this branch OUT of
                    # pending/footprint-dispatch until the infra clears.
                    # Must be checked BEFORE _inherited_break_info to prevent the
                    # generic 'Verification attempts exhausted' task_failure block
                    # from clobbering the infra_issue category.
                    if self._infra_hold_info is not None:
                        info = self._infra_hold_info
                        return await self._mark_blocked(
                            info['reason'],
                            detail=info.get('detail', ''),
                            category='infra_issue',
                            escalate_to_human=True,
                            block_status='infra-hold',
                        )
                    if self._inherited_break_info is not None:
                        info = self._inherited_break_info
                        return await self._mark_blocked(
                            info['reason'],
                            detail=info['detail'],
                            category=info['category'],
                            dedupe_fingerprint=info['fingerprint'],
                            suggested_action='await_preexisting_main_hotfix',
                        )
                    detail = self._last_verify_result.failure_report() if self._last_verify_result else ''
                    return await self._mark_blocked('Verification attempts exhausted', detail=detail)
                # Passing fall-through (_verify_debugfix_loop returns DONE):
                # capture the verified branch tip so the subsequent
                # _enter_phase(REVIEW) records it in the workflow_verify payload
                # as the durable checkpoint key for the next run (task 2752).
                # Normalize an empty tip (a failed `git rev-parse` returns '' —
                # see _get_head_commit) to None so we never record '' as a
                # green key: a later run that also read '' must MISS the
                # checkpoint (green_checkpoint_at_tip fails closed on an empty
                # tip), not match '' == '' and falsely skip verify.
                self._verify_green_tip_sha = (await self._get_head_commit()) or None

            # REVIEW
            self._enter_phase(WorkflowState.REVIEW)
            # Tree-hash verdict cache (task 2749): if a non-blocking verdict
            # is already recorded for HEAD's committed tree hash — from a
            # prior dispatch that reviewed byte-identical content — skip the
            # reviewer and ALL suggestion routing.  Suggestions were routed
            # when the verdict was first recorded; re-routing on every
            # re-dispatch is exactly the nit-loop churn this eliminates.
            # Blocking verdicts are never cached, so any hit is safe→DONE.
            #
            # A cache hit is honoured ONLY when the reviewer-config
            # fingerprint (active reviewer roster + resolved models) also
            # matches (task 2749 amendment): the committed tree alone does
            # not capture review-input identity, so a byte-identical tree
            # reviewed under a CHANGED roster / re-pinned model (e.g. a new
            # reviewer, or metadata.model_overrides) must force a fresh
            # review rather than short-circuit to a stale verdict the new
            # config never produced.
            #
            # Fail-safe: a None tree hash (git error), a None fingerprint
            # (config unreadable), or a cache miss / stale fingerprint all
            # fall through to a normal review — the strictest safe direction
            # (a redundant review, never a masked blocking issue).
            # ``review_tree_hash`` / ``review_fingerprint`` stay in scope for
            # verdict recording on the DONE path below.
            review_tree_hash: str | None = None
            review_fingerprint: str | None = None
            if self.worktree and self.artifacts:
                try:
                    review_tree_hash = await self.git_ops.get_head_tree_hash(
                        self.worktree
                    )
                    review_fingerprint = self._reviewer_config_fingerprint()
                    if review_tree_hash:
                        cached = self.artifacts.get_cached_verdict(
                            review_tree_hash
                        )
                        # Positive match required: skip only when a verdict
                        # exists AND the current config fingerprint is known
                        # AND it equals the one the verdict was minted under.
                        if (
                            cached is not None
                            and review_fingerprint is not None
                            and cached.get('reviewer_fingerprint')
                            == review_fingerprint
                        ):
                            logger.info(
                                'Task %s: REVIEW skipped — cached %s verdict '
                                'for tree %s (reviewer config unchanged)',
                                self.task_id, cached.get('verdict'),
                                review_tree_hash,
                            )
                            return WorkflowOutcome.DONE
                except Exception as exc:
                    # Fail-safe (never load-bearing): a tree-hash or cache
                    # lookup error degrades to a normal review — never skips,
                    # never crashes the loop.  Logged loud, not silent.
                    logger.warning(
                        'Task %s: verdict-cache lookup failed, falling back '
                        'to a full review: %s',
                        self.task_id, exc,
                    )
                    review_tree_hash = None
                    review_fingerprint = None
            # Consume-once (task 2750): this review is scoped to the amendment
            # delta iff it immediately follows an amendment.  Read+clear the
            # armed context now — before _review — so it applies to exactly one
            # review and a later replan review runs unscoped.
            used_ctx = amendment_ctx
            amendment_ctx = None
            reviews = await self._review(amendment_ctx=used_ctx)
            if reviews.reviewer_errors:
                names = ', '.join(reviews.reviewer_errors)
                return await self._mark_blocked(
                    f'{len(reviews.reviewer_errors)} reviewer(s) failed with '
                    f'infrastructure errors after retries: {names}'
                )
            if not reviews.has_blocking_issues:
                # Scope a post-amendment review to the amendment delta (task
                # 2750).  Runs FIRST inside the non-blocking arm — after the
                # reviewer_errors early-return and outside the blocking-replan
                # path — so it never touches the blocking safety valve: only
                # `suggestions` is partitioned; out-of-delta suggestions are
                # routed to the curator inside _apply_amendment_delta_scope and
                # dropped from the verdict.  The re-arm check, the DONE-path
                # curator routing, and the task-2749 verdict cache below then
                # all see only in-delta suggestions.  No-op when this review
                # does not follow an amendment (used_ctx is None).
                if used_ctx is not None:
                    reviews = await self._apply_amendment_delta_scope(
                        reviews, used_ctx,
                    )
                # Temporal companion to the spatial delta scope above (task
                # 2523): drop suggestions already SETTLED in a PRIOR amendment
                # round so they neither re-arm the loop below nor churn the
                # DONE-path curator routing.  Composes SPATIAL(2750) →
                # TEMPORAL(2523); no-op on a first-pass review or when no prior
                # archive exists, and fails safe toward EMIT on any adjudication
                # error.  Blocking issues never reach here (short-circuited by
                # the enclosing non-blocking arm).
                reviews = await self._suppress_resettled_suggestions(
                    reviews, amendment_round,
                )
                # L2b: try an amendment pass before escalating suggestions.
                # In-scope suggestions (module-lock members) are applied by
                # the implementer directly — no architect, no new tasks.
                # Cap is config.max_amendment_rounds (default 1).
                in_scope = self._suggestions_in_scope(reviews.suggestions)
                if (
                    in_scope
                    and amendment_round < self.config.max_amendment_rounds
                ):
                    amendment_round += 1
                    # Persist the new task-lifetime total (task 2749).
                    if self.artifacts:
                        self.artifacts.set_review_counters(
                            amendment_rounds_total=amendment_round
                        )
                    logger.info(
                        'Task %s: amendment round %d, %d in-scope '
                        'suggestion(s) (of %d total)',
                        self.task_id, amendment_round,
                        len(in_scope), len(reviews.suggestions),
                    )
                    # Task 2640 (source point 2): file THIS round's out-of-scope
                    # complement to the curator at the source, BEFORE _amend, so a
                    # follow-up skipped as out-of-scope is captured deterministically
                    # even if the in-scope amendment below escalates.  The residual
                    # drop task 2750 does not deterministically close: 2750 only
                    # recovers these downstream IF the post-amendment re-review
                    # repeats them as out-of-delta.  Object-identity complement:
                    # _suggestions_in_scope returns the SAME dict objects it
                    # filtered, so an id()-difference is exact and order-preserving
                    # and cannot mis-drop a genuinely out-of-scope suggestion that is
                    # content-equal to an in-scope one (a value-based `not in`
                    # could).  Mirrors _apply_amendment_delta_scope's out-of-delta
                    # routing — parent linkage (spawned_from), priority=low, verbatim
                    # description, no-mcp fallback, and the in-task
                    # _last_routed_suggestion_hash dedup all come from the shared
                    # method (the dedup also absorbs the round-0-out-of-scope ==
                    # round-1-out-of-delta double-file case).
                    in_scope_ids = {id(x) for x in in_scope}
                    out_of_scope = [
                        s for s in reviews.suggestions
                        if id(s) not in in_scope_ids
                    ]
                    if out_of_scope:
                        await self._route_review_suggestions_to_curator(
                            ReviewAggregation(
                                has_blocking_issues=False,
                                blocking_issues=[],
                                suggestions=out_of_scope,
                                reviews={},
                                reviewer_errors=[],
                            )
                        )
                    # Archive pre-amendment reviews so post-mortem can compare
                    if self.artifacts:
                        import shutil
                        reviews_dir = self.artifacts.root / 'reviews'
                        archive_dir = (
                            self.artifacts.root
                            / f'reviews-amend-{amendment_round}'
                        )
                        if reviews_dir.exists() and not archive_dir.exists():
                            shutil.copytree(reviews_dir, archive_dir)
                            logger.info(
                                'Task %s: archived reviews to %s',
                                self.task_id, archive_dir.name,
                            )
                    # Capture HEAD immediately BEFORE the amendment so
                    # {pre_amendment_head}..HEAD is exactly the amendment delta
                    # the next REVIEW is scoped to (task 2750).  Co-located with
                    # the reviews-amend archive + set_review_counters above.
                    pre_amendment_head = await self._get_head_commit()
                    amend_ok = await self._amend(in_scope, amendment_round)
                    if not amend_ok:
                        return WorkflowOutcome.ESCALATED
                    self.metrics.amendment_rounds += 1
                    # Never re-enter VERIFY/REVIEW — which run on the dirty
                    # worktree — with UNCOMMITTED amendment work: _amend invokes
                    # the implementer but never commits, and the merge queue
                    # lands only committed branch state, so an uncommitted
                    # amendment could pass verify+review and reach DONE yet
                    # silently fail to land (task 2760).  Committing the WIP here
                    # also advances HEAD so 2750's pre_amendment_head..HEAD delta
                    # below is exactly the amendment change instead of empty.
                    # No-op on a clean tree.
                    await self._commit_amendment_wip(amendment_round)
                    # Arm the consume-once scope for the next REVIEW pass with
                    # the pre-amendment HEAD and the suggestions the amendment
                    # was asked to address (task 2750).
                    amendment_ctx = AmendmentReviewContext(
                        pre_amendment_head, in_scope,
                    )
                    continue  # re-loop: EXECUTE → VERIFY → REVIEW

                # Cap exhausted or nothing in-scope — existing DONE path.
                # Route suggestions directly to the curator intake (fire-and-
                # forget); fall back to memory write when there are none.
                # _escalate_suggestions is retained as the steward fallback
                # but is no longer called from this path.
                if reviews.suggestions:
                    await self._route_review_suggestions_to_curator(reviews)
                else:
                    await self._write_suggestions_to_memory(reviews)
                # Record the non-blocking verdict keyed by the committed tree
                # hash (task 2749) so a future re-dispatch on byte-identical
                # content skips the reviewer and its suggestion routing.  Only
                # PASS / suggestions_only are cached — blocking verdicts are
                # never recorded (a replan changes the tree anyway), so any
                # cache hit is unconditionally safe→DONE.  The reviewer-config
                # fingerprint is stamped alongside so a later dispatch under a
                # changed roster / model re-pin re-reviews instead of reusing
                # this verdict (task 2749 amendment).
                if review_tree_hash and self.artifacts:
                    self.artifacts.record_review_verdict(
                        review_tree_hash,
                        'suggestions_only' if reviews.suggestions else 'PASS',
                        suggestions_routed=bool(reviews.suggestions),
                        reviewer_fingerprint=review_fingerprint,
                    )
                return WorkflowOutcome.DONE

            review_cycle += 1
            # Persist the new task-lifetime total (task 2749).
            if self.artifacts:
                self.artifacts.set_review_counters(
                    review_cycles_total=review_cycle
                )

            # Archive reviews from this cycle before re-plan overwrites them
            if self.artifacts:
                reviews_dir = self.artifacts.root / 'reviews'
                archive_dir = self.artifacts.root / f'reviews-cycle-{review_cycle}'
                if reviews_dir.exists() and not archive_dir.exists():
                    import shutil
                    shutil.copytree(reviews_dir, archive_dir)
                    logger.info('Task %s: archived reviews to %s', self.task_id, archive_dir.name)

            if review_cycle >= self.config.max_review_cycles:
                self._escalate_review_issues(reviews)
                return WorkflowOutcome.ESCALATED

            # Re-plan based on review feedback
            logger.info(
                f'Task {self.task_id}: review cycle {review_cycle}, '
                f'{len(reviews.blocking_issues)} blocking issues'
            )
            await self._replan(reviews)
            # Re-stamp provenance — architect may have overwritten plan.json
            assert self.artifacts is not None
            self.plan = self.artifacts.read_plan()
            if not self.plan or not self.plan.get('steps'):
                plan_dump = json.dumps(self.plan, indent=2) if self.plan else 'None'
                logger.error(
                    f'Task {self.task_id}: replan produced plan.json with '
                    f'missing/empty "steps" — full plan content: {plan_dump}'
                )
                return await self._mark_blocked(
                    'Architect replan produced no valid steps',
                    detail=f'Replan content:\n{plan_dump[:4000]}',
                )
            if outcome := await self._validate_prerequisites_or_block('replan'):
                return outcome
            self.artifacts.stamp_plan_provenance(self.session_id)
            # Task 2874: the architect may have WIDENED plan.files while
            # addressing review feedback ("you may add new steps"), so route
            # the possibly-widened files through the same scope-
            # reconciliation choke point _plan()/_apply_revalidation_skip()/
            # _run_simple_task() use — keeping plan.files ⊆ metadata.files
            # (and module locks covering every plan.files module)
            # CONTINUOUSLY, not just at plan-entry/merge-entry. Without this,
            # metadata.files/the new module lock would lag plan.files for the
            # whole ensuing reviewer_comprehensive pass — the window the
            # orphan-reaper sweeps in and fires the plan.files ⊋
            # metadata.files divergence (esc-2865-11 cluster). A no-widen
            # replan (the common case) reconciles to an identical file set
            # and is a harmless no-op.
            replan_files = self.plan.get('files') or []
            if replan_files and not await self._set_task_scope(replan_files):
                # Genuine cross-module lock conflict (a sibling task holds an
                # additional lock the widened plan needs): _set_task_scope
                # already persisted metadata.files=replan_files and requeued
                # the task on the scheduler's own conflict branch, but
                # self.modules is left UNCHANGED — so the loop must NOT
                # proceed into another EXECUTE under a foreign lock. Mirrors
                # _plan()'s blast-radius conflict path via the shared
                # _requeue_on_lock_conflict helper (task 2874 amendment).
                return await self._requeue_on_lock_conflict(
                    replan_files, 'plan_blast_radius_lock_conflict',
                    'Replan expansion',
                )
            self.metrics.review_cycles += 1

    async def _execute_iterations(self) -> WorkflowOutcome:
        """Run implementer iterations until plan is complete."""
        assert self.worktree is not None and self.artifacts is not None
        # Zero-output hang circuit breaker: reset all circuit-breaker state each
        # time _execute_iterations is entered so that a re-entry within a single
        # run cannot inherit stale _zero_output_hang_info from a prior pass and
        # mis-route a later unrelated BLOCKED outcome to infra_issue.
        # Counts CONSECUTIVE zero-output CLI timeouts; any non-zero-output result
        # resets it.  Threshold: config.max_consecutive_zero_output_timeouts.
        consecutive_zero_output = 0
        self._zero_output_hang_info = None
        # Counts CONSECUTIVE progress-timeouts (is_timed_out_with_progress);
        # reset by any successful (non-timeout) result.  Purely diagnostic —
        # progress_resume_total (below) is what the cap check and the churn
        # circuit breaker actually consult.
        consecutive_progress_timeouts = 0
        # Cumulative (NOT consecutive, NOT reset here) count of progress-
        # timeouts across the WHOLE task run — self.metrics.progress_resume_total,
        # not a local.  Task 2360 fix #4 (reify-4827): a ceiling-kill+resume
        # pair of a productive session is excluded from max_execute_iterations
        # via the cap-check subtraction below, so it no longer erodes the
        # budget meant for genuine implementer attempts — but an endless
        # non-converging kill/resume stream must still terminate, so it is
        # bounded independently by config.max_progress_resume_iterations
        # (checked in the γ branch below). Supersedes the old PRD §9 Q4
        # design decision that max_execute_iterations was the only bound.
        # MUST live on self.metrics (not a local reset to 0 here): this
        # method is re-entered by _execute_verify_review_loop after a review
        # cycle (_replan) or amendment round (_amend), and a local would lose
        # the exclusion earned by earlier execute phases on re-entry.
        self._progress_resume_churn_info = None
        self._preserve_config_dir = False
        self._preserve_config_dir_reason = None
        while self.artifacts.get_pending_steps():
            if (
                self.metrics.execute_iterations - self.metrics.progress_resume_total
                >= self.config.max_execute_iterations
            ):
                # Iteration cap reached (progress-resumes excluded from the
                # count — see progress_resume_total above).  If the most-recent
                # batch of iterations were all zero-output timeouts
                # (consecutive_zero_output > 0), classify this as infra_issue
                # rather than the generic 'Execution iterations exhausted' —
                # this handles the edge case where
                # max_consecutive_zero_output_timeouts > max_execute_iterations
                # so the circuit-breaker threshold is never reached inside the loop
                # but the pattern is still clearly an infra hang.
                if consecutive_zero_output > 0:
                    self._zero_output_hang_info = {
                        'reason': ZERO_OUTPUT_HANG_REASON,
                        'detail': (
                            f'consecutive_zero_output={consecutive_zero_output} '
                            f'at_iteration_cap=True '
                            f'iteration={self.metrics.execute_iterations} '
                            f'evidence: .task/zero_output_evidence-iter*.json'
                        ),
                    }
                elif consecutive_progress_timeouts > 0:
                    # Defensive fallback, not expected to be reachable: a
                    # progress-timeout iteration leaves
                    # (execute_iterations - progress_resume_total) unchanged
                    # (both increment together — see progress_resume_total
                    # above), so it cannot itself push the adjusted count to
                    # this cap; a non-converging stream instead trips the
                    # independent max_progress_resume_iterations churn
                    # breaker in the γ branch first. Kept as a diagnostic in
                    # case that invariant is ever broken by a future change.
                    logger.warning(
                        'Task %s: exhausted max_execute_iterations with '
                        'consecutive_progress_timeouts=%s (progress_resume_total=%s) '
                        '— session may be slow or looping. '
                        'Bound: max_execute_iterations=%s, '
                        'max_progress_resume_iterations=%s.',
                        self.task_id, consecutive_progress_timeouts,
                        self.metrics.progress_resume_total, self.config.max_execute_iterations,
                        self.config.max_progress_resume_iterations,
                    )
                return WorkflowOutcome.BLOCKED

            # Inter-iteration rebase: keep the task branch close to main
            # so the eventual merge is less likely to conflict.  Skip on
            # the first iteration (nothing has changed yet).
            rebase_notice = None
            if (
                self.metrics.execute_iterations > 0
                and self.config.inter_iteration_rebase
            ):
                rebase_notice = await self._inter_iteration_rebase()

            # Validate plan ownership before each implementer invocation
            if not self.artifacts.validate_plan_owner(self.session_id):
                logger.error(
                    f'Task {self.task_id}: plan.json ownership mismatch — '
                    f'expected session {self.session_id}, plan has different _session_id'
                )
                self._escalate_plan_overwrite()
                return WorkflowOutcome.BLOCKED

            self.plan = self.artifacts.read_plan()
            iteration_log, corrupted = self.artifacts.read_iteration_log()
            if corrupted:
                self._escalate_corruption(corrupted)

            # Reconcile any done step's commit orphaned by a rebase (this
            # dispatch's inter-iteration rebase above, or a prior
            # dispatch's warm-lane/requeue rebase) before the WIP detector
            # runs — the re-read picks up any reconciled commit_sha so
            # _detect_tip_wip_commits' own done-step dedup correctly
            # excludes it instead of re-surfacing it as unattributed
            # pending work (task 2386).
            await self._reconcile_done_step_commits()
            self.plan = self.artifacts.read_plan()

            wip_notice = await self._detect_tip_wip_commits()

            # Snapshot completed steps before invocation
            completed_before = {
                s['id']
                for col in ('prerequisites', 'steps')
                for s in self.plan.get(col, [])
                if isinstance(s, dict) and s.get('status') == 'done'
            }

            prompt = await self.briefing.build_implementer_prompt(
                self.plan, iteration_log, rebase_notice=rebase_notice,
                task_id=self.task_id, wip_notice=wip_notice,
            )
            pre_head = await self._get_head_commit()
            result = await self._invoke(IMPLEMENTER, prompt, self.worktree)

            self.metrics.execute_iterations += 1

            # Check for escalations.  Plain blocking L1 (already-escalated-to-human)
            # issues from prior runs of this same task stay in the queue until a
            # human or escalation-watcher resolves them; we must not let them
            # sink the current run.  Fresh L0 blocking escalations gate progress
            # here, AND born-at-L2 (critical/urgent) escalations AND any level≥2
            # escalations also gate regardless of severity — D4 accepted consequence:
            # a pending critical/urgent from a prior incarnation DOES sink a fresh
            # run (intended stop-the-line semantics).  See _is_gating_escalation.
            blocking = [e for e in self._check_escalations() if _is_gating_escalation(e)]
            if blocking:
                return WorkflowOutcome.ESCALATED

            # Re-read plan to see progress
            self.plan = self.artifacts.read_plan()

            # Compute newly completed steps and write iteration log
            completed_after = {
                s['id']
                for col in ('prerequisites', 'steps')
                for s in self.plan.get(col, [])
                if isinstance(s, dict) and s.get('status') == 'done'
            }
            newly_completed = sorted(completed_after - completed_before)

            if newly_completed:
                step_descs = [
                    s.get('description', s['id'])
                    for col in ('prerequisites', 'steps')
                    for s in self.plan.get(col, [])
                    if isinstance(s, dict) and s.get('id') in newly_completed
                ]
                summary = '; '.join(step_descs)
            else:
                summary = 'No new steps completed'

            self.artifacts.append_iteration_log({
                'iteration': self.metrics.execute_iterations,
                'agent': 'implementer',
                'steps_attempted': newly_completed,
                'steps_completed': newly_completed,
                **await self._iteration_commit_provenance(pre_head),
                'summary': summary,
                'source': 'orchestrator',
            })

            # Defense-in-depth: re-stamp _session_id after each implementer
            # iteration.  The plan-tools MCP server preserves it, but if the
            # model also edited plan.json directly (dropping _session_id),
            # this recovers gracefully instead of blocking.
            if not self.artifacts.validate_plan_owner(self.session_id):
                logger.warning(
                    'Task %s: _session_id mismatch after implementer — re-stamping',
                    self.task_id,
                )
            self.artifacts.stamp_plan_provenance(self.session_id)

            if not result.success:
                logger.warning(
                    f'Task {self.task_id}: implementer iteration '
                    f'{self.metrics.execute_iterations} failed'
                )

            # --- Zero-output hang circuit breaker ---
            # Detect consecutive fresh-invocation CLI hangs (reify-4429 pattern):
            # the subprocess produced ZERO output for the full timeout, meaning
            # the CLI never started real work.  Retrying identically burns time
            # (~20 min/iteration × threshold) with no chance of progress.
            if is_zero_output_timeout(result):
                consecutive_zero_output += 1
                # Capture forensic evidence for every zero-output timeout (best-effort;
                # the helper suppresses all I/O errors internally).
                self._capture_zero_output_evidence(result, self.metrics.execute_iterations)
                if consecutive_zero_output >= self.config.max_consecutive_zero_output_timeouts:
                    self._zero_output_hang_info = {
                        'reason': ZERO_OUTPUT_HANG_REASON,
                        'detail': (
                            f'consecutive_zero_output={consecutive_zero_output} '
                            f'iteration={self.metrics.execute_iterations} '
                            f'duration_ms={result.duration_ms} '
                            f'subtype={result.subtype!r} '
                            f'evidence: .task/zero_output_evidence-iter*.json'
                        ),
                    }
                    self._preserve_config_dir = True
                    self._preserve_config_dir_reason = ZERO_OUTPUT_HANG_REASON
                    logger.error(
                        'Task %s: zero-output hang circuit breaker tripped after %d '
                        'consecutive fresh-invocation timeouts — blocking as infra_issue',
                        self.task_id, consecutive_zero_output,
                    )
                    return WorkflowOutcome.BLOCKED
                logger.warning(
                    'Task %s: zero-output CLI timeout (%d/%d consecutive) — '
                    'will retry; recycling config dir if enabled',
                    self.task_id, consecutive_zero_output,
                    self.config.max_consecutive_zero_output_timeouts,
                )
                # Optional mitigation: discard the wedged session state so the
                # next iteration starts with a clean CLAUDE_CONFIG_DIR.  Safe
                # because turns==0 — the destroyed session did no real work.
                # The tripping iteration's dir is preserved (see above).
                if self.config.recycle_config_dir_on_zero_output:
                    self._recycle_config_dir()
                continue  # skip judge on zero-output; increment next iteration
            elif is_timed_out_with_progress(result):
                # γ: a productive implementer iteration was killed at the wall
                # (transcript_turns > 0 — real work done).  Resume the SAME Claude
                # session next iteration so the ~20 min of context continues instead
                # of being discarded.  The existing _invoke resume lifecycle
                # (workflow.py:6240-6253) picks these up because
                # _pending_resume_role == role.name == 'implementer'.
                #
                # Progress is proof-of-life: reset the consecutive zero-output counter
                # so this iteration does not count toward the wedge circuit breaker.
                # (Preserves the reset the old `else` branch provided before γ.)
                consecutive_zero_output = 0
                consecutive_progress_timeouts += 1
                self.metrics.progress_resume_total += 1
                if self.metrics.progress_resume_total >= self.config.max_progress_resume_iterations:
                    # Independent churn circuit breaker (task 2360 fix #4,
                    # reify-4827): this kill+resume pair is excluded from
                    # max_execute_iterations (see the cap check above), so an
                    # endless non-converging progress-timeout stream would
                    # otherwise never terminate.  Bounded here instead — and,
                    # like the cap exclusion, cumulative across the WHOLE task
                    # run (not per execute-phase), consistent with the design
                    # decision's "cumulative progress-resume count" bounding
                    # what a normal task run is expected to need.
                    self._progress_resume_churn_info = {
                        'reason': PROGRESS_RESUME_CHURN_REASON,
                        'detail': (
                            f'progress_resume_total={self.metrics.progress_resume_total} '
                            f'consecutive_progress_timeouts={consecutive_progress_timeouts} '
                            f'iteration={self.metrics.execute_iterations} '
                            f'last_transcript_turns={result.transcript_turns}'
                        ),
                    }
                    # Preserve the config dir for forensic analysis, same as
                    # the zero-output hang breaker above: a session that made
                    # real work every time (transcript_turns > 0) yet never
                    # converged is at least as worth inspecting post-mortem
                    # as a zero-output wedge (task 2360 amendment, reify-4827).
                    self._preserve_config_dir = True
                    self._preserve_config_dir_reason = PROGRESS_RESUME_CHURN_REASON
                    logger.error(
                        'Task %s: progress-resume churn circuit breaker tripped '
                        'after %d ceiling-kill+resume cycles of a productive '
                        'session — blocking (session may be too slow to '
                        'converge, or looping)',
                        self.task_id, self.metrics.progress_resume_total,
                    )
                    return WorkflowOutcome.BLOCKED
                self._pending_resume_session_id = self._last_invoke_session_id
                self._pending_resume_role = IMPLEMENTER.name
                logger.info(
                    'Task %s: implementer timed out with progress '
                    '(transcript_turns=%s) — resuming session %s next iteration',
                    self.task_id, result.transcript_turns,
                    self._last_invoke_session_id,
                )
                continue  # re-dispatch with --resume; skip judge this iteration
            else:
                # Non-timeout result: reset both consecutive counters.
                consecutive_zero_output = 0
                consecutive_progress_timeouts = 0

            # --- Judge: decide whether to exit early (ζ) ---
            # Opt-in via config.judge_after_each_iteration (default False).
            # Eval mode flips it on per-task. Failures fall through silently
            # to the next iteration — current behavior is preserved as worst case.
            if self.config.judge_after_each_iteration:
                judge_verdict = await self._run_completion_judge(iteration_log)
                if judge_verdict is not None and judge_verdict.get('complete') is True:
                    # Safety: reject complete=True if substantive_work=False.
                    # An empty or trivial diff cannot be a completed task.
                    if not judge_verdict.get('substantive_work', False):
                        logger.warning(
                            f'Task {self.task_id}: judge returned complete=True '
                            f'with substantive_work=False — ignoring verdict'
                        )
                    else:
                        self.metrics.judge_early_exits += 1
                        logger.info(
                            f'Task {self.task_id}: judge signaled completion at '
                            f'iteration {self.metrics.execute_iterations} — '
                            f'reasoning: {judge_verdict.get("reasoning", "")[:200]}'
                        )
                        self.artifacts.append_iteration_log({
                            'iteration': self.metrics.execute_iterations,
                            'agent': 'judge',
                            'event': 'early_exit',
                            'complete': True,
                            'substantive_work': True,
                            'uncovered_plan_steps': judge_verdict.get('uncovered_plan_steps', []),
                            'summary': judge_verdict.get('reasoning', '')[:500],
                            'source': 'orchestrator',
                        })
                        return WorkflowOutcome.DONE

        return WorkflowOutcome.DONE

    def _recycle_config_dir(self) -> None:
        """Tear down the current TaskConfigDir and create a fresh one in place.

        Called between sub-threshold zero-output CLI timeouts when
        ``config.recycle_config_dir_on_zero_output`` is True.  Per-task
        session state (CLAUDE_CONFIG_DIR) is the prime deterministic-wedge
        suspect; because ``turns==0`` the destroyed session did no real work,
        so discarding it cannot lose progress — it aligns with crash-recovery
        semantics (resuming a wedged empty session would only re-hang).

        The tripping iteration's config dir is NOT recycled: that dir is
        preserved for forensic analysis via ``_preserve_config_dir=True``.

        Guards on ``self._config_dir`` and ``self.worktree`` so the method is
        a no-op when called before either is initialised (defensive, should
        not occur in practice).
        """
        if not self._config_dir or not self.worktree:
            return
        old_path = self._config_dir.path
        self._config_dir.cleanup()
        self._config_dir = TaskConfigDir(
            self.task_id,
            base_dir=self.worktree / '.task',
        )
        logger.warning(
            'Task %s: recycled TaskConfigDir for zero-output-hang mitigation '
            '(%s → %s)',
            self.task_id, old_path, self._config_dir.path,
        )

    def _cleanup_config_dir(self) -> None:
        """Preserve-aware wrapper around ``TaskConfigDir.cleanup()``.

        Called from ``run()``'s finally block instead of inlining the
        ``if self._config_dir: self._config_dir.cleanup()`` pattern.
        Extraction makes the preserve-skip behaviour unit-testable (task 1739).

        Behaviour:
        - ``self._config_dir is None`` → no-op (dir was never created).
        - ``self._preserve_config_dir is True`` → skip cleanup and log a
          warning naming the preserved path and the tripped breaker's reason
          (``self._preserve_config_dir_reason`` — zero-output hang or
          progress-resume churn, task 1739 / task 2360), so the on-call
          engineer knows the dir is intentional and why.
        - Otherwise → ``self._config_dir.cleanup()`` (normal path).
        """
        if not self._config_dir:
            return
        if self._preserve_config_dir:
            logger.warning(
                'Task %s: config dir preserved for forensic analysis '
                '(reason: %s) → %s',
                self.task_id,
                self._preserve_config_dir_reason or ZERO_OUTPUT_HANG_REASON,
                self._config_dir.path,
            )
            return
        self._config_dir.cleanup()

    def _capture_zero_output_evidence(self, result: AgentResult, iteration: int) -> None:
        """Persist forensic evidence for a zero-output CLI timeout to .task/.

        Called on EVERY zero-output timeout in ``_execute_iterations`` (before
        the threshold check) so both sub-threshold and tripping occurrences
        are captured.  Writes to ``artifacts.root`` which outlives the per-task
        ``TaskConfigDir`` cleanup and the BLOCKED worktree (not GC'd since the
        task is not DONE).

        Best-effort: any I/O failure is caught and logged rather than
        propagated — evidence capture must never crash the iteration loop.
        """
        if not self.artifacts:
            return
        try:
            # Build config_dir listing (relative paths as strings).
            config_dir_str: str | None = None
            config_dir_listing: list[str] = []
            if self._config_dir is not None:
                config_dir_str = str(self._config_dir.path)
                try:
                    config_dir_listing = [
                        str(p.relative_to(self._config_dir.path))
                        for p in self._config_dir.path.rglob('*')
                    ]
                except OSError:
                    config_dir_listing = ['<listing failed>']

            evidence = {
                'iteration': iteration,
                'duration_ms': result.duration_ms,
                'timed_out': result.timed_out,
                'turns': result.turns,
                'cost_usd': result.cost_usd,
                'subtype': result.subtype,
                'stderr_tail': (result.stderr or '')[-4000:],
                'proc_tree': result.proc_tree,
                'config_dir': config_dir_str,
                'config_dir_listing': config_dir_listing,
                'transcript_turns': result.transcript_turns,
            }

            # Enrich with transcript data when we have a session id stash and
            # a config_dir to search.  Best-effort: any failure leaves the keys absent.
            # NOTE: this calls read_transcript_records a second time on the same file;
            # _run_subprocess already called count_transcript_turns (which delegates to
            # read_transcript_records internally) to stamp transcript_turns.  The
            # double-read is intentional: caching the records would require expanding
            # the shared _SubprocessResult contract beyond the single transcript_turns
            # field, and this code path is rare (only reached on timeout/kill).
            if self._config_dir is not None and self._last_invoke_session_id:
                try:
                    records = read_transcript_records(
                        self._config_dir.path,
                        self._last_invoke_session_id,
                    )
                    if records is not None:
                        evidence['last_records'] = records[-5:]
                        # Find the last tool_use block name across all records.
                        # Real Claude CLI transcript schema: content blocks live at
                        # rec['message']['content'] (nested), NOT rec['content'].
                        # The top-level rec.get('content') branch is retained as a
                        # defensive fallback for synthetic/flattened record shapes.
                        last_tool: str | None = None
                        for rec in records:
                            msg = rec.get('message')
                            if isinstance(msg, dict):
                                content = msg.get('content')
                            else:
                                content = rec.get('content')
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                                        last_tool = block.get('name')
                        evidence['last_tool'] = last_tool
                except Exception as exc:
                    logger.warning(
                        'Task %s: failed to enrich zero-output evidence with transcript data: %s',
                        self.task_id, exc,
                    )

            evidence_path = self.artifacts.root / f'zero_output_evidence-iter{iteration}.json'
            evidence_path.write_text(json.dumps(evidence, indent=2))
            logger.info(
                'Task %s: zero-output evidence written → %s',
                self.task_id, evidence_path,
            )
        except Exception as exc:
            logger.warning(
                'Task %s: failed to write zero-output evidence (iteration=%d): %s',
                self.task_id, iteration, exc,
            )

    async def _run_completion_judge(
        self, iteration_log: list[dict]
    ) -> dict | None:
        """Invoke the completion judge. Returns parsed verdict dict or None on failure.

        Any failure mode (exception, success=False, malformed output) returns
        None so the caller continues the iteration loop — current behavior is
        preserved as the worst case.
        """
        assert self.worktree is not None and self.artifacts is not None

        base_commit = self.artifacts.read_base_commit()
        if base_commit:
            diff = await self.git_ops.get_diff_from_base(self.worktree, base_commit)
        else:
            diff = await self.git_ops.get_diff_from_main(self.worktree)

        if not diff or not diff.strip():
            logger.info(
                f'Task {self.task_id}: empty diff — skipping judge invocation'
            )
            return None

        prompt = await self.briefing.build_completion_judge_prompt(
            plan=self.plan,
            iteration_log=iteration_log,
            diff=diff,
            task_id=self.task_id,
        )

        # I-FRESH: never consume a stale verdict from a prior invocation on
        # this same worktree (mirrors the merger/reviewer pre-spawn clear,
        # workflow.py:_resolve_and_resubmit / _run_reviewer).
        self.artifacts.clear_verdict('judge')

        pre_cost = self.metrics.total_cost_usd
        try:
            result = await self._invoke(JUDGE, prompt, self.worktree)
        except Exception as exc:
            logger.warning(
                f'Task {self.task_id}: judge invocation raised '
                f'{type(exc).__name__}: {exc} — continuing iteration loop'
            )
            return None

        # judge_cost_usd is a subset of total_cost_usd (already incremented
        # inside _invoke), tracked separately for reporting.
        self.metrics.judge_invocations += 1
        self.metrics.judge_cost_usd += (self.metrics.total_cost_usd - pre_cost)

        if not result.success:
            logger.warning(
                f'Task {self.task_id}: judge invocation returned success=False — '
                f'continuing iteration loop'
            )
            return None

        # PRD task η / task 2487: the completion verdict is read from the
        # verdict-tools artifact (verdicts/judge.json) ONLY — the ζ
        # transition-window fallback to result.structured_output is gone,
        # so all four verdict roles (merger/reviewer/triage/judge) now read
        # artifact-only. The judge is completion-gating and this
        # orchestrator self-hosts the PRD's own tasks, so the fail-safe is
        # deliberate: an absent artifact, a non-dict envelope, or a non-dict
        # 'verdict' payload all yield None (keep iterating, never a false
        # completion — I-FAIL-SAFE). A dict 'verdict' payload that is merely
        # missing required keys is likewise judged by the required-keys
        # check below and returns None if incomplete — see design decision
        # "Absent-both / final-invalid ⇒ None" (plan.json).
        envelope = self.artifacts.read_verdict('judge')
        verdict = envelope.get('verdict') if isinstance(envelope, dict) else None
        if not isinstance(verdict, dict):
            logger.warning(
                f'Task {self.task_id}: judge returned no/invalid verdict artifact — '
                f'continuing iteration loop'
            )
            return None

        required = {'complete', 'reasoning', 'uncovered_plan_steps', 'substantive_work'}
        if not required <= verdict.keys():
            logger.warning(
                f'Task {self.task_id}: judge verdict missing keys '
                f'{required - verdict.keys()} — continuing iteration loop'
            )
            return None

        return verdict

    async def _reconcile_done_step_commits(self) -> None:
        """Re-point a done step's ``commit`` when a rebase orphaned it.

        ``_inter_iteration_rebase`` (below) squashes uncommitted work into a
        WIP safety-commit and then rewrites the branch onto a new base; the
        warm-lane-reclaim / requeue-rebase paths in ``git_ops.py`` do the
        same. Either can leave an already-``done`` plan step's recorded
        ``commit`` unreachable from the new HEAD even though its code now
        lives in the WIP squash commit — and because
        :meth:`_detect_tip_wip_commits` DEDUPS done-step commits out of its
        notice, that stale ``commit`` is otherwise invisible to the harness
        (task 2386).

        For every done step/prerequisite in ``self.plan`` whose recorded
        ``commit`` is no longer reachable from HEAD
        (``not is_ancestor(commit, head)``), this locates the contiguous run
        of WIP safety-commits sitting at HEAD (the same walk-and-stop over
        ``get_commit_subjects`` + :func:`is_wip_safety_commit` that
        :meth:`_detect_tip_wip_commits` performs) and matches by filename
        set the orphaned commit's own changed-file set
        (:meth:`GitOps.get_commit_changed_files`) against the WIP run's
        union changed-file set. On a match — the orphaned commit's files are
        non-empty and a subset of the WIP run's files — the step is
        re-pointed to the WIP tip sha via
        ``artifacts.update_step_status(id, 'done', wip_tip_sha)``. On a
        mismatch, an unresolvable original commit (empty file set — e.g.
        GC'd), or no WIP run at HEAD at all, the step is left unchanged and
        :meth:`_escalate_unreconciled_done_step` files a non-blocking info
        escalation flagging it for review instead — deduped per
        ``(step_id, stale_commit)`` via
        ``self._unreconciled_done_step_escalations`` so a still-stuck orphan
        doesn't re-escalate on every ``_execute_iterations`` loop iteration
        (see that attribute's docstring in ``__init__``).

        **Clean-replay remap tier (task 2762)**: the WIP-filename heuristic
        above only fires in Scenario A — uncommitted work squashed into a WIP
        safety-commit at HEAD. In Scenario B, a requeue/inter-iteration rebase
        that PRESERVES individual commits (replaying them onto a new base
        rather than squashing) leaves NO WIP run at HEAD, so ``wip_tip_sha`` is
        ``None`` and every orphaned done step would fall straight to the
        escalation below — the 5205/5055/5196 escalation storms, which RCA
        confirmed were pure bookkeeping noise (each orphaned step's commit is
        patch-id-identical to a commit already on the live branch). So before
        escalating, the else branch now asks
        :meth:`GitOps.find_equivalent_commit` for a commit in ``base..HEAD``
        that is patch-id-equivalent (or, on a diff-altering rebase that kept
        the message, uniquely subject-matching) to the orphaned commit, and
        re-points the step to it via the same
        ``artifacts.update_step_status(id, 'done', new_sha)`` call. Only when
        that returns ``None`` (no equivalent) does the step escalate. This tier
        keeps the same best-effort/fail-safe and zero-persisted-state posture:
        it is re-derived from live git every pass, and a false negative reverts
        to exactly the pre-2762 escalation baseline. The WIP heuristic stays
        FIRST because in Scenario A the step's diff is folded into a larger WIP
        diff with a different patch-id and no standalone equivalent, so the
        filename-subset signal is the only correct one there.

        **Cross-restart durability (task 2764)**: the per-``(step_id,
        stale_commit)`` dedup set is in-memory and process-local, so an
        orchestrator restart would otherwise construct a fresh, empty set and
        re-file every previously-filed pair (the 5205 04:33Z storm was 10 such
        duplicates after a restart). To close that gap, each escalated key is
        also persisted to the meta-root (``reconcile_state.json``) on the emit
        branch, and this method hydrates those persisted keys INTO the
        in-memory set exactly once per workflow instance (lazily at the top,
        guarded by ``self._loaded_persisted_step_escalations``). A restarted
        workflow therefore files at most genuinely-new pairs. This is
        defense-in-depth behind the same-instance flood guard and the 2762
        remap tier — the persisted store is an exact mirror of what was
        actually escalated (append happens only on the escalation path, after
        the ``if remapped: continue`` early-out), so hydration suppresses
        precisely the re-files it should and nothing more.

        **Heuristic, not a content diff**: the match above is a
        *filename-set* subset check only — it never compares file contents
        or blob shas. A WIP commit that happens to touch a superset of the
        orphaned step's filenames but with unrelated content would still
        count as a match and get silently re-pointed. This is accepted
        given the best-effort posture (the fallback on any doubt is an
        escalation, never silence) — :meth:`GitOps.branch_content_in_main`
        shows the byte-level ``git diff --quiet`` pattern this could
        upgrade to if the filename heuristic ever proves too loose in
        practice.

        Best-effort and defensive, identical posture to
        :meth:`_detect_tip_wip_commits`: no-ops on any missing collaborator
        (``worktree``/``git_ops``/``artifacts``), unset ``base_commit``, or
        git error — a false negative here just reverts to today's baseline
        (the documented manual `git show <wip-sha>` verification), never
        sinks the iteration loop.

        Callers should re-read ``self.plan = self.artifacts.read_plan()``
        immediately after this so a reconciled commit is picked up before
        ``_detect_tip_wip_commits`` runs (see ``_execute_iterations``) — that
        ordering lets the detector's own done-step dedup correctly exclude
        the now-reconciled sha instead of re-surfacing it as unattributed
        pending work.
        """
        if self.worktree is None or self.git_ops is None or self.artifacts is None:
            return
        # task 2764 — hydrate the persisted emitted-escalation keys into the
        # in-memory flood guard exactly once per workflow instance (disk read
        # guarded by the flag). This runs here — the set's sole consumer, past
        # the None-artifacts no-op guard so self.artifacts is guaranteed
        # constructed — so a restarted orchestrator's fresh, empty set is
        # seeded with what a prior process already filed, and the emit-branch
        # `if escalation_key not in self._unreconciled_done_step_escalations`
        # check below then skips already-filed (step_id, stale_commit) pairs.
        if not self._loaded_persisted_step_escalations:
            self._unreconciled_done_step_escalations |= (
                self.artifacts.read_emitted_step_escalations()
            )
            self._loaded_persisted_step_escalations = True
        base = self.artifacts.read_base_commit()
        if not base:
            return
        try:
            head = await self._get_head_commit()
            commits = await self.git_ops.get_commit_subjects(self.worktree, base)
            wip_run: list[tuple[str, str]] = []
            for sha, subject in commits:
                if not is_wip_safety_commit(subject):
                    break
                wip_run.append((sha, subject))
            wip_tip_sha = wip_run[0][0] if wip_run else None
            wip_files: set[str] = set()
            for wip_sha, _subject in wip_run:
                wip_files.update(await self.git_ops.get_commit_changed_files(wip_sha))

            done_items = [
                item
                for col in ('prerequisites', 'steps')
                for item in self.plan.get(col, [])
                if isinstance(item, dict) and item.get('status') == 'done' and item.get('commit')
            ]
            for item in done_items:
                commit = item['commit']
                if await self.git_ops.is_ancestor(commit, head):
                    continue  # still reachable — nothing to reconcile
                orphaned_files = await self.git_ops.get_commit_changed_files(commit)
                if orphaned_files and wip_tip_sha and set(orphaned_files) <= wip_files:
                    self.artifacts.update_step_status(item['id'], 'done', wip_tip_sha)
                else:
                    # No WIP-filename match. Before escalating, try to remap the
                    # orphaned commit to its rebase-replayed sha on the live
                    # branch. This is the clean-replay case (Scenario B): a
                    # requeue/inter-iteration rebase that preserves individual
                    # commits rather than squashing them into a WIP
                    # safety-commit leaves NO WIP run at HEAD, so wip_tip_sha is
                    # None and the filename heuristic above never fires.
                    # find_equivalent_commit locates a patch-id-equivalent (or,
                    # on a diff-altering rebase that kept the message, a
                    # uniquely subject-matching) commit in base..HEAD and
                    # returns its sha. It is fully fail-safe (None on any git
                    # error, an unresolvable/GC'd commit, ambiguity, or no
                    # equivalent), so a false negative simply falls through to
                    # the unchanged escalation path below — zero persisted
                    # state, re-derived from live git each pass.
                    remapped = await self.git_ops.find_equivalent_commit(
                        self.worktree, base, commit,
                    )
                    if remapped:
                        self.artifacts.update_step_status(item['id'], 'done', remapped)
                        continue
                    # Mismatch, unresolvable original (empty file set — e.g.
                    # GC'd), no WIP run at HEAD, and no live-branch equivalent:
                    # cannot safely auto-reconcile. Flag for review and leave
                    # the commit unchanged rather than guess. Deduped by
                    # _unreconciled_done_step_escalations so a still-stuck
                    # orphan doesn't re-file an info escalation on every
                    # _execute_iterations loop iteration.
                    escalation_key = (item['id'], commit)
                    if escalation_key not in self._unreconciled_done_step_escalations:
                        self._unreconciled_done_step_escalations.add(escalation_key)
                        self._escalate_unreconciled_done_step(item['id'], commit, wip_tip_sha)
                        # task 2764: durably persist the emitted key (in the
                        # meta-root reconcile_state.json sidecar) so a restarted
                        # orchestrator — which constructs a fresh, empty
                        # in-memory set — hydrates it and does not re-file this
                        # same (step_id, stale_commit) pair. Placed AFTER the
                        # `if remapped: continue` early-out above, so only a
                        # genuinely-escalated pair (not WIP-reconciled, not
                        # 2762 patch-id-remapped) gets a durable key.
                        # Escalate-FIRST-then-persist is deliberate: a rare
                        # persist failure at worst re-files the pair once on the
                        # next restart (a loud duplicate) rather than silently
                        # dropping the escalation.
                        self.artifacts.append_emitted_step_escalation(item['id'], commit)
        except Exception:
            logger.warning(
                'Done-step commit reconciliation failed; leaving plan commits unchanged',
                exc_info=True,
            )

    def _escalate_unreconciled_done_step(
        self, step_id: str, stale_commit: str, wip_tip_sha: str | None,
    ) -> None:
        """Submit a non-blocking info escalation for a done step whose
        recorded ``commit`` is orphaned but could not be safely
        filename-matched against the tip WIP safety-commit run (filename-set
        mismatch, an unresolvable/GC'd original commit, or no WIP run
        sitting at HEAD at all).

        Mirrors :meth:`_escalate_corruption`'s posture: informational only
        (never gates progress), and the step's ``commit`` is deliberately
        left unchanged so the documented manual `git show <wip-sha>`
        verification workaround remains available. Guards a missing
        ``escalation_queue`` with a log-and-continue, same as
        ``_escalate_corruption``.
        """
        if not self.escalation_queue:
            logger.warning(
                "Task %s: done step %s's commit %s is orphaned and could not "
                'be auto-reconciled against a WIP run (no escalation queue)',
                self.task_id, step_id, stale_commit,
            )
            return

        from escalation.models import Escalation

        tip_desc = wip_tip_sha or '<none>'
        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='info',
            category='infra_issue',
            summary=(
                f"Done step {step_id}'s commit {stale_commit[:10]} is orphaned "
                f'and could not be auto-reconciled against WIP tip {tip_desc}'
            ),
            detail=(
                f'Step {step_id} recorded commit {stale_commit}, which is no '
                'longer reachable from HEAD (likely rewritten/orphaned by an '
                "inter-iteration or warm-lane rebase). Its filenames could "
                f'not be matched against the tip WIP safety-commit run (tip '
                f'sha {tip_desc}) — either the changed-file sets did not '
                'match (this is a filename check only, not a byte-content '
                'diff), the original commit is unresolvable (possibly '
                'garbage-collected), or no WIP safety-commit run sits at '
                f'HEAD. Verify manually via `git show {stale_commit}` '
                "against the WIP commit(s) and re-point the step's commit "
                'if appropriate.'
            ),
            suggested_action='verify_wip_reconciliation',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)

    async def _detect_tip_wip_commits(self) -> list[dict]:
        """Detect a contiguous run of WIP safety-commits sitting at HEAD.

        Several harness paths auto-commit uncommitted work as a safety net
        before a rebase/requeue/reclaim (``_inter_iteration_rebase`` here,
        plus the requeue and warm-lane-reclaim paths in ``git_ops.py``). Any
        of these can land a still-"pending" plan step's complete
        implementation at branch HEAD *before* ``mark_step_done`` is called
        for that step. This surfaces those commits so
        ``build_implementer_prompt`` can point the implementer at them
        instead of it re-discovering (and potentially re-implementing) the
        already-committed work.

        Best-effort and defensive: returns ``[]`` on any missing
        collaborator (``worktree``/``git_ops``/``artifacts``), unset
        ``base_commit``, or git error — a false negative here just reverts
        to today's baseline behavior (no notice), never sinks the
        iteration loop.

        Returns HEAD-first ``[{'sha': ..., 'subject': ...}]`` for the
        contiguous run of WIP safety-commits at HEAD (stops at the first
        non-WIP commit), excluding any SHA already recorded as the
        ``commit`` of a done plan step (dedup against ``self.plan``).
        """
        if self.worktree is None or self.git_ops is None or self.artifacts is None:
            return []
        base = self.artifacts.read_base_commit()
        if not base:
            return []
        try:
            commits = await self.git_ops.get_commit_subjects(self.worktree, base)
            run: list[tuple[str, str]] = []
            for sha, subject in commits:
                if not is_wip_safety_commit(subject):
                    break
                run.append((sha, subject))
            if not run:
                return []
            recorded = {
                s['commit']
                for col in ('prerequisites', 'steps')
                for s in self.plan.get(col, [])
                if isinstance(s, dict) and s.get('status') == 'done' and s.get('commit')
            }
            # Prefix-match rather than exact-match: the prompt shows an
            # abbreviated 12-char SHA (see build_implementer_prompt's
            # wip_section), so an implementer may call mark_step_done with
            # that short form instead of the full SHA from git log. Exact
            # equality would then never dedup, re-surfacing the same
            # already-attributed commit on every subsequent iteration.
            return [
                {'sha': sha, 'subject': subject}
                for sha, subject in run
                if not any(sha.startswith(r) or r.startswith(sha) for r in recorded)
            ]
        except Exception:
            logger.warning(
                'WIP-tip detection failed; treating as no WIP commits',
                exc_info=True,
            )
            return []

    async def _rederive_step_status_from_branch_state(self) -> list[str]:
        """Re-derive stale-``pending`` plan steps to ``done`` from the
        durable iteration log's ``steps_completed`` records (task 2387).

        Unions every ``.task/iterations.jsonl`` entry's ``steps_completed``
        into the set of step IDs genuinely completed on this branch, then
        flips any plan.json step currently "pending" whose ID is in that set
        to "done" — recording the post-rebase HEAD (not the log entry's own
        ``commit``) as the step's commit. The log-reported commit is a
        *pre-rebase* SHA: this method runs immediately after
        ``rebase_preserving_task_commits`` has already rewritten the branch
        onto the new base, so that SHA is virtually guaranteed to be
        unreachable from the new HEAD. Recording it anyway would manufacture
        exactly the "done step with an orphaned commit" condition
        :meth:`_reconcile_done_step_commits` exists to repair — and that
        repair only succeeds when the step's code happens to have been
        folded into a WIP safety-commit sitting at HEAD; otherwise it falls
        through to a non-blocking info escalation on every such rebase.
        Recording ``head`` directly satisfies the reachable-commit invariant
        up front instead of depending on that downstream heuristic.

        GUARD: only reconciles when :meth:`_has_prior_implementation`
        (called with the current branch HEAD, i.e. SHA-primary mode) reports
        ``has_work=True`` — branch-SHA divergence from base AND at least one
        genuine implementer/debugger log entry. This is the same
        battle-tested false-DONE protection used elsewhere in this class; it
        prevents re-deriving a step to "done" from an inherited/contaminated
        ``iterations.jsonl`` on a branch that has no real commits beyond
        base. Reuses ``status.entries`` from the guard call rather than
        re-reading the iteration log.

        **Granularity caveat**: the guard above is branch-level, not
        per-step — it only asserts that *some* genuine work landed on this
        branch, then trusts every step id named anywhere in the union of
        ``steps_completed``. It does not verify that any individual step's
        own commit is reachable or that its claimed files actually changed.
        A log entry that over-claims (names a step in ``steps_completed``
        whose own change was later reverted or failed) is still flipped to
        "done" as long as the branch has real work elsewhere. This mirrors
        the project's existing trust-the-durable-log design (see this
        task's design_decisions) — per-step correctness depends entirely on
        implementer log fidelity, not on independent per-step verification.

        Entries recording no durable commit (``committed is False``, task
        2759) are excluded from the ``steps_completed`` union: a step marked
        completed in a round where HEAD never advanced did not actually land
        on the branch, so it must not be re-derived to done even when the
        branch has other genuine work. Legacy entries lacking the ``committed``
        key are unaffected.

        Emits a single ``event='plan_step_rederive'`` iteration-log entry
        (naming every re-derived step id) when at least one step is
        re-derived; emits nothing on a clean pass, so the common case does
        not pollute the log other tests/tools scan.

        Best-effort and defensive, identical posture to
        :meth:`_detect_tip_wip_commits` / :meth:`_reconcile_done_step_commits`:
        returns ``[]`` on any missing collaborator (``worktree``/``git_ops``/
        ``artifacts``), unset ``base_commit``, or internal error — a false
        negative here just reverts to today's (buggy) baseline rather than
        sinking the iteration loop.

        Returns the list of step IDs re-derived.
        """
        if self.worktree is None or self.git_ops is None or self.artifacts is None:
            return []
        base = self.artifacts.read_base_commit()
        if not base:
            return []
        try:
            head = await self._get_head_commit()
            status = self._has_prior_implementation(wt_head=head)
            if not status.has_work:
                return []

            completed_ids: set[str] = set()
            for entry in status.entries:
                # An entry that explicitly recorded no durable commit
                # (committed:False, task 2759) did not land its step on the
                # branch — exclude it from the union so a step "completed" with
                # no commit is not re-derived to done even when the branch has
                # other real work. Legacy entries lack the key and fall through.
                if entry.get('committed') is False:
                    continue
                completed_ids.update(entry.get('steps_completed') or [])

            plan = self.artifacts.read_plan()
            rederived: list[str] = []
            for collection in ('prerequisites', 'steps'):
                for item in plan.get(collection, []):
                    if not isinstance(item, dict):
                        continue
                    if item.get('status') == 'pending' and item.get('id') in completed_ids:
                        step_id = item['id']
                        # Always record post-rebase HEAD, never the log's
                        # pre-rebase commit — see the docstring above.
                        self.artifacts.update_step_status(step_id, 'done', commit=head)
                        rederived.append(step_id)

            if rederived:
                self.artifacts.append_iteration_log({
                    'iteration': self.metrics.execute_iterations,
                    'agent': 'orchestrator',
                    'event': 'plan_step_rederive',
                    'rederived_steps': rederived,
                    'source': 'orchestrator',
                    'summary': (
                        f'Re-derived {len(rederived)} plan step(s) from branch '
                        f'state after rebase: {rederived}'
                    ),
                })
                logger.info(
                    'Task %s: re-derived plan step status from branch state: %s',
                    self.task_id, rederived,
                )
            return rederived
        except Exception:
            logger.warning(
                'Plan step status re-derivation failed; leaving plan.json unchanged',
                exc_info=True,
            )
            return []

    async def _inter_iteration_rebase(
        self, *, event_label: str = 'rebase',
    ) -> dict | None:
        """Check if main advanced past our base; if so, rebase.

        Returns a dict ``{old_base, new_base, changed_files}`` when a
        rebase was performed, or ``None`` if no rebase was needed or the
        rebase failed (failure is non-blocking — the merge phase will
        handle conflicts).

        ``event_label`` populates the ``event`` field on the
        iteration_log entry so verify-phase calls (Fix 3) can be
        distinguished from execute-phase calls in post-mortem analysis.
        """
        assert self.worktree is not None and self.artifacts is not None

        old_base = self.artifacts.read_base_commit()
        if not old_base:
            return None

        current_main = await self.git_ops.get_main_sha()
        if current_main == old_base:
            return None

        if not await self.git_ops.is_ancestor(old_base, current_main):
            return None  # unexpected topology — skip

        changed_files = await self.git_ops.get_changed_files(
            old_base, current_main,
        )

        # Commit any uncommitted work before rebasing.  ``commit()`` no-ops
        # on a clean tree so verify-phase callers (which always run on a
        # clean tree post-execute) do not produce empty WIP commits.
        await self.git_ops.commit(
            self.worktree, 'chore: save WIP before inter-iteration rebase',
        )

        # rebase_preserving_task_commits wraps rebase_onto_main with a
        # post-condition guard (task 2403): a BranchResetError propagates
        # out of this method (and out of the execute/verify loops that
        # call it) rather than being swallowed here — a silent wipe of
        # committed work is not an ordinary, recoverable rebase conflict.
        if not await self.git_ops.rebase_preserving_task_commits(self.worktree):
            logger.warning(
                f'Task {self.task_id}: inter-iteration rebase failed, '
                f'continuing on old base'
            )
            return None

        self.artifacts.update_base_commit(current_main)

        # Re-derive any plan step whose status regressed to (or never
        # advanced past) "pending" even though it's genuinely complete on
        # this branch (task 2387) — before task 2386's
        # _reconcile_done_step_commits and the loop's self.plan = read_plan()
        # re-reads (see _execute_iterations), so a pending->done flip here is
        # picked up as part of that same refresh. Best-effort/self-contained
        # (never raises), so the return dict and event-label behavior below
        # are unaffected.
        await self._rederive_step_status_from_branch_state()

        # Capture is_first BEFORE incrementing the counter (0 == this is the
        # first rebase of a fresh per-dispatch WorkflowMetrics instance).
        is_first = self.metrics.inter_iteration_rebases == 0
        self.metrics.inter_iteration_rebases += 1

        distance = await self.git_ops.get_rebase_distance(old_base, current_main)
        cohort = classify_rebase_cohort(
            distance, is_first, self.config.rebase_reseed_distance_threshold,
        )

        self.artifacts.append_iteration_log({
            'iteration': self.metrics.execute_iterations,
            'agent': 'orchestrator',
            'event': event_label,
            'old_base': old_base,
            'new_base': current_main,
            'files_changed_on_main': changed_files[:50],
            'distance_commits': distance,
            'cohort': cohort,
            'source': 'orchestrator',
            'summary': (
                f'Rebased onto main ({old_base[:8]} -> {current_main[:8]}), '
                f'{len(changed_files)} files changed'
            ),
        })

        logger.info(
            f'Task {self.task_id}: rebased onto main '
            f'({old_base[:8]} -> {current_main[:8]}), '
            f'{len(changed_files)} files changed on main'
        )

        return {
            'old_base': old_base,
            'new_base': current_main,
            'changed_files': changed_files,
            'distance_commits': distance,
            'cohort': cohort,
            'is_first_rebase': is_first,
        }

    def _emit_rebase_verify_cost(
        self,
        rebase_notice: dict,
        verify_result: VerifyResult,
    ) -> None:
        """Emit one rebase_verify_cost event pairing a real rebase with its next verify.

        No-op when self.event_store is None (fire-and-forget; never raises).
        Called only when rebase_notice is not None (i.e. a real rebase happened).
        """
        if self.event_store is None:
            return
        self.event_store.emit(
            EventType.rebase_verify_cost,
            task_id=self.task_id,
            phase='verify',
            duration_ms=int(verify_result.duration_secs * 1000),
            data={
                'old_base': rebase_notice['old_base'],
                'new_base': rebase_notice['new_base'],
                'distance_commits': rebase_notice['distance_commits'],
                'files_changed_on_main': len(rebase_notice['changed_files']),
                'next_verify_wall_secs': verify_result.duration_secs,
                'verify_scope': {
                    'n_task_files': len(self._task_files or []),
                    'n_modules': len(self._module_configs or []),
                    'workspace': self._train is not None,
                },
                'cohort': rebase_notice['cohort'],
            },
        )

    async def _run_scoped_verification_with_infra_retry(
        self,
        *,
        verify_attempt: int,
    ) -> VerifyResult | None:
        """Run :func:`run_scoped_verification` with bounded infra-error retry.

        Catches :class:`VerifyInfraError` (transient infra OSErrors — ENOSPC,
        EDQUOT, EROFS, EIO, EMFILE, ENFILE) raised during verify, and also
        treats a RETURNED failing :class:`VerifyResult` whose ``category`` is
        infra-transient (``category in INFRA_TRANSIENT_CATEGORIES`` — task ν,
        verify-scope-inversion-prd.md) the same way — retries up to
        ``config.verify_infra_retry_max_attempts`` times with exponential
        back-off, keeping the task CLAIMED (in-progress) throughout.

        Returns
        -------
        VerifyResult
            On a successful run (no infra error, or infra error cleared within
            the retry window).
        None
            When the retry window is exhausted.  The caller
            (:meth:`_verify_debugfix_loop`) interprets ``None`` as the cue to
            propagate :attr:`_infra_hold_info` up to ``run()``, which stamps
            the task's first-class ``infra-hold`` status (via
            ``_mark_blocked(block_status='infra-hold')``, task 2200/ω4) and
            returns ``WorkflowOutcome.BLOCKED``.

        Modelled on the ``WarmLaneDiskPressure`` retry pattern
        (git_ops.py / workflow.py:1857-1889) but keeps the task in-progress
        rather than REQUEUED — the caller must not acquire a new implement
        footprint on resume (see Task-1883 A1 root-cause analysis).
        """
        assert self.worktree is not None

        max_attempts = self.config.verify_infra_retry_max_attempts
        backoff_base = self.config.verify_infra_retry_backoff_secs
        max_backoff = self.config.verify_infra_retry_max_backoff_secs

        last_infra_exc: VerifyInfraError | None = None
        last_infra_result: VerifyResult | None = None

        for infra_attempt in range(max_attempts):
            try:
                # Routing note (λ, task 2589, T1): role='task' here means
                # merge_verify_breadth never forks this call — the knob is
                # merge-role-gated only (see run_scoped_verification's
                # force_workspace branch).  A train member's own verify
                # always takes the legacy single global workspace command
                # when force_workspace=True, byte-identical regardless of
                # the knob's value; the task-role pytest floor (R3) still
                # applies independently, in the non-force_workspace
                # module-config branch, for any train member whose task
                # verify isn't force_workspace'd.
                # Warm-lane consumer-hold (task 3027): hold the task lane's
                # <lane_dir>.lock across this nextest run so a concurrent reify
                # warm-lane-gc.sh reclaim's per-lane `flock -n` refuses/queues
                # instead of reseeding the lane out from under this live
                # consumer and deleting its in-flight test binaries (esc-5236-7
                # / esc-5275-10). The flock is the cross-process guard reify's
                # warm-lane-gc.sh honors; task_verify_lease fails OPEN on
                # contention (per-attempt scope keeps the hold off the
                # infra-retry backoff sleeps below).
                async with self.git_ops.task_verify_lease(self.worktree):
                    result = await run_scoped_verification(
                        self.worktree, self.config, self._module_configs,
                        task_files=self._task_files,
                        attempt_id=verify_attempt + 1,
                        task_id=self.task_id,
                        archive_root=self.config.project_root / 'data' / 'verify-logs',
                        force_workspace=self._train is not None,
                        role='task',
                    )
            except VerifyInfraError as exc:
                last_infra_exc = exc
                last_infra_result = None
                delay = min(backoff_base * (2 ** infra_attempt), max_backoff)
                logger.warning(
                    'Task %s: VerifyInfraError during verify (phase=%r errno=%r), '
                    'attempt %d/%d — backing off %.1fs',
                    self.task_id, exc.phase, exc.errno,
                    infra_attempt + 1, max_attempts, delay,
                )
                # Skip the backoff sleep on the final attempt: the loop is
                # about to exhaust and block regardless, so sleeping here
                # would only add up to max_backoff latency to the block path
                # with no subsequent retry to wait for (task 2591 amendment,
                # reviewer_comprehensive/efficiency).
                if infra_attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                continue

            # Task ν (verify-scope-inversion-prd.md): a RETURNED failing
            # VerifyResult classified as infra-transient (category in
            # INFRA_TRANSIENT_CATEGORIES — disk_full, semaphore_timeout,
            # pytest_internalerror, env_transient) is retried within this
            # SAME bounded window exactly like a caught VerifyInfraError, so
            # it consumes no verify_attempt and never reaches the debugger.
            if not result.passed and (result.category or '') in INFRA_TRANSIENT_CATEGORIES:
                last_infra_exc = None
                last_infra_result = result
                delay = min(backoff_base * (2 ** infra_attempt), max_backoff)
                logger.warning(
                    'Task %s: verify returned infra-transient category %r, '
                    'attempt %d/%d — backing off %.1fs',
                    self.task_id, result.category,
                    infra_attempt + 1, max_attempts, delay,
                )
                # Same final-attempt sleep-skip as the exception branch above.
                if infra_attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                continue

            return result

        # Window exhausted — stash infra-hold info and signal caller to BLOCK.
        # Either last_infra_exc or last_infra_result is guaranteed non-None:
        # the loop only reaches here after at least one infra-classified
        # iteration (max_attempts >= 1) — a raised VerifyInfraError or a
        # returned classified-infra VerifyResult — and each branch above
        # resets the other to None, so whichever was seen LAST wins.
        if last_infra_exc is not None:
            reason = (
                f'Verify infra failure not resolved after {max_attempts} in-process '
                f'retries (phase={last_infra_exc.phase!r} errno={last_infra_exc.errno})'
            )
            detail = (
                f'VerifyInfraError: phase={last_infra_exc.phase!r} '
                f'errno={last_infra_exc.errno}'
            )
            hold_phase = last_infra_exc.phase
            hold_errno = last_infra_exc.errno
        else:
            assert last_infra_result is not None
            reason = (
                f'Verify infra-transient outcome not resolved after {max_attempts} '
                f'in-process retries (category={last_infra_result.category!r})'
            )
            detail = last_infra_result.failure_report()
            hold_phase = None
            hold_errno = None
        self._infra_hold_info = {
            'reason': reason,
            'detail': detail,
            'category': 'infra_issue',
            'escalate_to_human': True,
            'phase': hold_phase,
            'errno': hold_errno,
        }
        # No metadata write here (task 2200/ω4): the retired metadata.infra_hold
        # boolean is gone.  _infra_hold_info propagates to run() via the
        # WorkflowOutcome.BLOCKED return below, which stamps the task's
        # first-class 'infra-hold' status through
        # _mark_blocked(block_status='infra-hold') — see
        # orchestrator.task_status.is_infra_held, the single source of truth
        # the harness HOLD guard and RESUME cascade key on (PRD C7/D3).
        return None

    async def _maybe_file_chronic_flakes(self, verify_result: VerifyResult) -> None:
        """Best-effort chronic pool-infra flake auto-file after a verify
        completes (task 2358).

        Detects a chronic flake from *verify_result*'s ``test_output`` (a
        CHRONIC-FLAKY marker) and/or the on-disk flaky ledger, and
        auto-files a non-blocking, medium-priority De-flake fix task — see
        ``chronic_flake.py``'s module docstring for the full policy.

        Thin by design: ``chronic_flake.maybe_file_chronic_flake_tasks`` is
        already internally catch-all-defensive (chronic_flake.py
        step-13/14), and this call site adds a second try/except as
        belt-and-suspenders (mirrors ``_spawn_main_health_fix_task``'s
        guard-and-log pattern) — a filing failure must never fail the
        verify/merge path, so it can never alter the caller's DONE/BLOCKED
        outcome.
        """
        if not self.config.chronic_flake.enabled:
            return
        try:
            client = chronic_flake.SchedulerChronicFlakeTaskClient(
                self.scheduler, self.config.project_root,
            )
            await chronic_flake.maybe_file_chronic_flake_tasks(
                verify_result.test_output, self.config, client,
            )
        except Exception as exc:
            logger.warning(
                'Task %s: chronic-flake auto-file failed: %s', self.task_id, exc,
            )

    async def _verify_debugfix_loop(self) -> WorkflowOutcome:
        """Run verification, invoke debugger on failures."""
        assert self.worktree is not None and self.artifacts is not None
        verify_attempt = 0

        while True:
            # Fix 3: rebase onto main BEFORE each verify (including the first).
            # Closes the verify-only-retry rebase gap: when main advances
            # mid-task (e.g. a sibling task fixes the env collision the
            # verify is failing on), the existing _inter_iteration_rebase
            # only fires from the EXECUTE loop — it cannot pick up new main
            # commits while we're cycling verify ↔ debugger.  The helper
            # short-circuits cheaply when current_main == old_base, so
            # firing on every retry costs at most one ``git rev-parse``.
            rebase_notice: dict | None = None
            if self.config.rebase_before_verify:
                rebase_notice = await self._inter_iteration_rebase(
                    event_label='verify_phase_rebase',
                )

            result = await self._run_scoped_verification_with_infra_retry(
                verify_attempt=verify_attempt,
            )
            if result is None:
                # Infra retry window exhausted; _infra_hold_info already stamped.
                return WorkflowOutcome.BLOCKED
            # Task 2358: chronic pool-infra flake auto-file. Fires on every
            # completed verify (green AND red) now that `result` is
            # guaranteed non-None — non-blocking, never alters the outcome.
            await self._maybe_file_chronic_flakes(result)
            if rebase_notice is not None:
                self._emit_rebase_verify_cost(rebase_notice, result)
            if not result.passed:
                self._last_verify_result = result
            if result.passed:
                return WorkflowOutcome.DONE

            verify_attempt += 1

            # Broken-main contagion guard: detect whether this failure was
            # inherited from main (preexisting break) rather than introduced by
            # this task.  If so, block WITHOUT self-patching or advancing the
            # signature-repeat counter — the stashed _inherited_break_info is
            # read at the call site to route a single deduped escalation.
            # Skips flaky/env categories (infra_timeout, flock_error) that are
            # non-deterministic to re-check on main.
            self._inherited_break_info = None
            if (
                self.config.escalate_preexisting_main_break
                and not result.timed_out
                and (result.category or '') not in PREEXISTING_BREAK_SKIP_CATEGORIES
            ):
                # Helper returns (is_preexisting, probe_main_sha); we reuse probe_main_sha
                # for fingerprint composition so probe and fingerprint reference the SAME
                # main SHA — avoids the TOCTOU window of a second get_main_sha() call.
                _is_inherited, _probe_sha = await verify_failure_is_preexisting_on_main(
                    self.worktree, self.config, self._module_configs,
                    self._task_files, result, self.git_ops,
                )
            else:
                _is_inherited, _probe_sha = False, ''
            if _is_inherited:
                # Fold key: same (category, cause_hint, main_sha) -> identical fp
                # -> submit_or_dedupe collapses N sibling tasks to ONE parent.
                # Shared helper ensures merge-queue + task-verify paths compose
                # the same fingerprint (verified by TestCrossTaskInheritedBreakDedup).
                fp = compute_preexisting_main_break_fingerprint(
                    result.category or '', result.cause_hint, _probe_sha,
                )
                self._inherited_break_info = {
                    'reason': (
                        f'Verify failure is preexisting on main '
                        f'(category={result.category!r}): {result.cause_hint[:120]}'
                    ),
                    'detail': result.failure_report(),
                    'category': 'preexisting_main_break',
                    'fingerprint': fp,
                }
                logger.warning(
                    'Task %s: verify failure classified as inherited from main '
                    '(category=%r, fp=%s) — blocking without self-patch',
                    self.task_id, result.category, fp[:16] if fp else '',
                )
                return WorkflowOutcome.BLOCKED

            # Fast-fail: when the verifier's own injected ``Command timed out
            # after Ns: …`` wrapper string is the only signal, retrying gives
            # the debugger nothing actionable.  Escalate to L1 after
            # ``max_opaque_timeout_attempts`` instead of burning the full
            # ``max_verify_attempts × verify_command_timeout_secs`` budget.
            # The streamed-stdout fix in ``_run_cmd`` means a real cause hint
            # should appear on attempt 1 for any genuine in-test hang;
            # persistent opaque timeouts indicate infrastructure the debugger
            # can't fix.
            if (
                result.category == FailureCategory.INFRA_TIMEOUT
                and _OPAQUE_TIMEOUT_CAUSE_RE.match(result.cause_hint or '')
                and verify_attempt >= self.config.max_opaque_timeout_attempts
            ):
                logger.warning(
                    'Task %s: opaque infra_timeout cap hit at attempt %d/%d '
                    '(cause_hint=%r) — escalating to L1',
                    self.task_id, verify_attempt,
                    self.config.max_opaque_timeout_attempts,
                    (result.cause_hint or '')[:120],
                )
                return WorkflowOutcome.BLOCKED
            # Signature-repetition guard: escalate to L1 after N consecutive
            # verify failures with the same (category, normalised cause_hint)
            # tuple.  Placement is AFTER the opaque-timeout cap (the more
            # specific signal) and BEFORE the global max_verify_attempts cap.
            # Uses the last-N-equal window so an early "blip" of a different
            # category does not prevent the guard from firing once the loop
            # settles into a repeat pattern.
            # Empty-string signatures ('', '') count as identical triples and
            # trigger the guard intentionally: if the verifier repeatedly
            # returns opaque output with no category or cause_hint, escalating
            # to L1 is still the right response.
            _sig = (result.category or '', _normalize_cause_hint(result.cause_hint))
            self._failure_signature_history.append(_sig)
            _N = self.config.max_failure_signature_repeat
            if (
                len(self._failure_signature_history) >= _N
                and all(s == _sig for s in self._failure_signature_history[-_N:])
            ):
                logger.warning(
                    'Task %s: %d consecutive identical verify failures '
                    '(sig=%r) — escalating to L1',
                    self.task_id, _N, _sig,
                )
                return WorkflowOutcome.BLOCKED
            if verify_attempt >= self.config.max_verify_attempts:
                return WorkflowOutcome.BLOCKED

            self.metrics.verify_attempts += 1
            logger.info(
                f'Task {self.task_id}: verify attempt {verify_attempt} failed: {result.summary}'
            )

            # Invoke debugger
            self.plan = self.artifacts.read_plan()
            prompt = await self.briefing.build_debugger_prompt(
                result.failure_report(), self.plan, task_id=self.task_id,
            )
            pre_head = await self._get_head_commit()
            debug_result = await self._invoke(DEBUGGER, prompt, self.worktree)

            # Write debugger iteration log entry
            self.artifacts.append_iteration_log({
                'iteration': verify_attempt,
                'agent': 'debugger',
                'steps_attempted': [],
                'steps_completed': [],
                **await self._iteration_commit_provenance(pre_head),
                'summary': f'Debug fix for: {result.summary[:100]}',
                'source': 'orchestrator',
            })

            # Check for escalations from debugger.  Same filter as the post-implementer
            # check — a stale plain blocking L1 from a prior run must not sink a
            # successful debug pass.  Born-at-L2 (critical/urgent) AND any level≥2
            # escalations also gate here — D4 accepted consequence: a pending
            # critical/urgent from a prior incarnation DOES sink a fresh run
            # (intended stop-the-line semantics).  See _is_gating_escalation.
            blocking = [e for e in self._check_escalations() if _is_gating_escalation(e)]
            if blocking:
                return WorkflowOutcome.ESCALATED

            if not debug_result.success:
                logger.warning(f'Task {self.task_id}: debugger failed')

    async def _apply_amendment_delta_scope(
        self, reviews: ReviewAggregation, ctx: AmendmentReviewContext,
    ) -> ReviewAggregation:
        """Scope a post-amendment review verdict to the amendment delta (task 2750).

        Deterministically partitions ``reviews.suggestions`` by whether each
        finding's ``location`` (``file:line``) falls within the amendment's
        NEW-side changed line ranges (``{pre_amendment_head}..HEAD``).  In-delta
        suggestions stay in the returned verdict; out-of-delta suggestions are
        routed to the curator via the existing
        :meth:`_route_review_suggestions_to_curator` path so they neither
        re-arm the amendment loop nor bloat the DONE-path verdict.

        Blocking findings are NEVER filtered — ``blocking_issues`` /
        ``has_blocking_issues`` pass through unchanged (the safety valve).

        Fail-open: if the amendment delta is empty or uncomputable, all
        suggestions are kept in the verdict and a WARNING is logged — a git
        error or a pathological no-op amendment must never silently discard
        real reviewer findings (loud-over-silent).
        """
        assert self.worktree is not None
        from orchestrator.review_suggestions.amendment_scope import (
            partition_suggestions_by_delta,
        )

        delta_ranges = await self.git_ops.get_new_side_changed_line_ranges(
            self.worktree, ctx.pre_amendment_head,
        )
        if not delta_ranges:
            logger.warning(
                'Task %s: post-amendment delta empty/uncomputable '
                '(pre_amendment_head=%s); keeping all %d suggestion(s) in the '
                'verdict (fail-open)',
                self.task_id, ctx.pre_amendment_head, len(reviews.suggestions),
            )
            return reviews

        in_delta, out_of_delta = partition_suggestions_by_delta(
            reviews.suggestions,
            delta_ranges,
            context_lines=_AMENDMENT_DELTA_CONTEXT_LINES,
        )
        if out_of_delta:
            await self._route_review_suggestions_to_curator(
                ReviewAggregation(
                    has_blocking_issues=False,
                    blocking_issues=[],
                    suggestions=out_of_delta,
                    reviews={},
                    reviewer_errors=[],
                )
            )
        logger.info(
            'Task %s: post-amendment review scoped to amendment delta — '
            '%d in-delta suggestion(s) kept, %d out-of-delta routed to curator',
            self.task_id, len(in_delta), len(out_of_delta),
        )
        return replace(reviews, suggestions=in_delta)

    async def _adjudicate_resettled(
        self, current_list: list[dict], prior_settled: list[dict],
    ) -> list[str]:
        """Batched prior-round-resolution adjudication (task 2523), fail-safe to EMIT.

        Runs ONE ``_RESETTLED_ADJUDICATOR`` invocation comparing *current_list*
        (the live suggestions) against *prior_settled* (the prior-round
        suggestion set) and returns a per-index decision list aligned to
        *current_list*, each drawn from {``SETTLED``, ``NOT_SETTLED``,
        ``INCONCLUSIVE``}.  The dedicated adjudicator role is used rather than
        ``REVIEWER_COMPREHENSIVE`` on purpose: the reviewer contract mandates a
        ``submit_review_verdict`` tool call and injects ``verdict_tools``, which
        would starve the ``StructuredOutput`` decisions payload this method
        depends on and render suppression silently inert (see the role's
        definition comment above).

        Fails SAFE toward ``NOT_SETTLED`` (emit) on ANY failure: a raised
        exception, a ``None`` / non-success / timed-out result, missing or
        unparseable ``structured_output``, an unknown decision value, or an
        index the model omits.  Only an explicit ``settled`` maps to
        ``SETTLED`` — mirroring the fail-safe posture of the pure
        :func:`~orchestrator.review_suggestions.prior_round.partition_by_decisions`
        so neither layer can silently drop a suggestion on ambiguity or failure.
        """
        from orchestrator.review_suggestions.prior_round import (
            INCONCLUSIVE,
            NOT_SETTLED,
            SETTLED,
            build_resettled_adjudicator_prompt,
        )

        n = len(current_list)
        if n == 0:
            return []
        fail_safe = [NOT_SETTLED] * n
        prompt = build_resettled_adjudicator_prompt(current_list, prior_settled)
        schema = {
            'type': 'object',
            'properties': {
                'decisions': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'index': {'type': 'integer'},
                            'decision': {
                                'type': 'string',
                                'enum': [SETTLED, NOT_SETTLED, INCONCLUSIVE],
                            },
                        },
                        'required': ['index', 'decision'],
                    },
                },
            },
            'required': ['decisions'],
        }
        try:
            assert self.worktree is not None
            result = await self._invoke(
                _RESETTLED_ADJUDICATOR, prompt, self.worktree,
                output_schema=schema,
            )
        except Exception as exc:
            logger.warning(
                'Task %s: resettled adjudication invoke raised (%s); failing '
                'safe — emitting all %d suggestion(s)', self.task_id, exc, n,
            )
            return fail_safe
        if result is None or not result.success or result.timed_out:
            logger.warning(
                'Task %s: resettled adjudication non-success/timed-out '
                '(success=%s timed_out=%s); failing safe — emitting all %d '
                'suggestion(s)', self.task_id,
                getattr(result, 'success', None),
                getattr(result, 'timed_out', None), n,
            )
            return fail_safe
        payload = result.structured_output
        raw_decisions = (
            payload.get('decisions') if isinstance(payload, dict) else None
        )
        if not isinstance(raw_decisions, list):
            logger.warning(
                'Task %s: resettled adjudication returned no usable decisions; '
                'failing safe — emitting all %d suggestion(s)', self.task_id, n,
            )
            return fail_safe
        allowed = {SETTLED, NOT_SETTLED, INCONCLUSIVE}
        # Every slot defaults to NOT_SETTLED (emit); an omitted index stays so.
        decisions = list(fail_safe)
        for entry in raw_decisions:
            if not isinstance(entry, dict):
                continue
            idx = entry.get('index')
            # bool is an int subclass — reject it so decisions[True] can't alias.
            if isinstance(idx, bool) or not isinstance(idx, int):
                continue
            if not (0 <= idx < n):
                continue
            verdict = entry.get('decision')
            decisions[idx] = verdict if verdict in allowed else NOT_SETTLED
        settled_count = sum(1 for d in decisions if d == SETTLED)
        logger.info(
            'Task %s: resettled adjudication — %d/%d suggestion(s) settled',
            self.task_id, settled_count, n,
        )
        return decisions

    async def _suppress_resettled_suggestions(
        self, reviews: ReviewAggregation, amendment_round: int,
    ) -> ReviewAggregation:
        """Suppress suggestions already SETTLED in a PRIOR amendment round (task 2523).

        The TEMPORAL companion to :meth:`_apply_amendment_delta_scope` (spatial):
        once at least one amendment round has archived a prior verdict, a
        batched LLM adjudication (:meth:`_adjudicate_resettled`) decides which
        of the current suggestions merely re-flag a concern the team already
        settled, and those are dropped from the returned verdict so they neither
        re-arm the amendment loop nor churn the DONE-path curator routing.

        No-op (returns *reviews* unchanged) when the gate is off, this is a
        first-pass review (``amendment_round < 1``), there are no suggestions,
        the artifacts root is unavailable, or no prior-round archive exists —
        keeping the common single-round path a cheap no-op.  Only
        ``suggestions`` is ever filtered; ``blocking_issues`` /
        ``has_blocking_issues`` / ``reviews`` / ``reviewer_errors`` pass through
        untouched (blocking issues are the safety valve).  Fails SAFE: any
        adjudication error keeps every suggestion (loud-over-silent).
        """
        from orchestrator.review_suggestions.prior_round import (
            load_prior_round_suggestions,
            partition_by_decisions,
        )

        if not self.config.suppress_resettled_review_suggestions:
            return reviews
        if amendment_round < 1:
            return reviews
        if not reviews.suggestions:
            return reviews
        if self.artifacts is None:
            return reviews
        prior = load_prior_round_suggestions(self.artifacts.root)
        if not prior:
            return reviews

        try:
            decisions = await self._adjudicate_resettled(
                reviews.suggestions, prior,
            )
            kept, suppressed = partition_by_decisions(
                reviews.suggestions, decisions,
            )
        except Exception as exc:
            logger.warning(
                'Task %s: resettled suppression failed (%s); keeping all %d '
                'suggestion(s) (fail-safe)',
                self.task_id, exc, len(reviews.suggestions),
            )
            return reviews

        if not suppressed:
            return reviews
        logger.info(
            'Task %s: suppressed %d re-flagged suggestion(s) already settled '
            'in a prior amendment round; %d kept',
            self.task_id, len(suppressed), len(kept),
        )
        return replace(reviews, suggestions=kept)

    async def _review(self, amendment_ctx: AmendmentReviewContext | None = None):
        """Run all 5 reviewers with stagger, retry errors.

        When *amendment_ctx* is set (this review immediately follows an
        amendment round), the amendment's addressed suggestions are threaded
        into each reviewer's prompt as an advisory scope constraint (task
        2750). The deterministic partition filter applied by the caller is the
        enforceable guarantee; this only reduces wasted reviewer effort.
        """
        assert self.worktree is not None and self.artifacts is not None
        base_commit = self.artifacts.read_base_commit()
        if base_commit:
            diff = await self.git_ops.get_diff_from_base(self.worktree, base_commit)
        else:
            diff = await self.git_ops.get_diff_from_main(self.worktree)

        amendment_suggestions = (
            amendment_ctx.amended_suggestions if amendment_ctx is not None else None
        )
        stagger = self.config.reviewer_stagger_secs

        # Staggered launch — spread OAuth session creation
        async def _staggered(idx: int, role: AgentRole):
            if idx > 0:
                await asyncio.sleep(idx * stagger)
            return await self._run_reviewer(
                role, diff, amendment_suggestions=amendment_suggestions,
            )

        tasks = [_staggered(i, r) for i, r in enumerate(ALL_REVIEWERS)]
        results = list(await asyncio.gather(*tasks, return_exceptions=True))

        # Retry ERROR verdicts and exceptions
        for attempt in range(self.config.max_reviewer_retries):
            error_indices = [
                i for i, r in enumerate(results)
                if isinstance(r, Exception)
                or (isinstance(r, dict) and r.get('verdict') == 'ERROR')
            ]
            if not error_indices:
                break
            logger.info(
                f'Task {self.task_id}: retrying {len(error_indices)} failed '
                f'reviewer(s) (attempt {attempt + 1}/{self.config.max_reviewer_retries})'
            )
            for i in error_indices:
                await asyncio.sleep(stagger)
                try:
                    results[i] = await self._run_reviewer(
                        ALL_REVIEWERS[i], diff,
                        amendment_suggestions=amendment_suggestions,
                    )
                except Exception as exc:
                    results[i] = exc

        # Write results — synthesize ERROR for persistent exceptions
        for role, result in zip(ALL_REVIEWERS, results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    f'Reviewer {role.name} failed after retries: {result}',
                    exc_info=result,
                )
                result = {
                    'reviewer': role.name,
                    'verdict': 'ERROR',
                    'issues': [],
                    'summary': f'Reviewer exception: {result}',
                }
            if isinstance(result, dict):
                self.artifacts.write_review(role.name, result)

        return self.artifacts.aggregate_reviews()

    async def _run_reviewer(
        self, role: AgentRole, diff: str,
        amendment_suggestions: list[dict] | None = None,
    ) -> dict:
        """Run a single reviewer and read its verdict artifact.

        *amendment_suggestions*, when set, threads the post-amendment advisory
        scope section into the reviewer prompt (task 2750).
        """
        assert self.worktree is not None and self.artifacts is not None
        prompt = await self.briefing.build_reviewer_prompt(
            role.name, diff, amendment_suggestions=amendment_suggestions,
        )

        # I-FRESH: never consume a stale verdict from a prior invocation on
        # this same worktree (mirrors _resolve_and_resubmit's pre-spawn
        # clear, workflow.py:7073-7075).
        self.artifacts.clear_verdict(role.name)
        result = await self._invoke(role, prompt, self.worktree)

        # Read the reviewer's structured verdict instead of the
        # structured_output/json.loads cascade (task 2484 / PRD task δ).
        # Defensive extraction mirrors the merger's read_verdict handling
        # in _resolve_and_resubmit (workflow.py:7114): a dict envelope with
        # a dict 'verdict' payload carrying verdict∈{PASS,ISSUES_FOUND} is
        # trusted only when the invocation itself also reported success —
        # an invocation failure (crash / max_turns / budget exhaustion) is
        # untrusted even if it happened to write a verdict before failing;
        # anything else (absent, cleared, malformed, or unsuccessful)
        # degrades to the role's existing worst-case ERROR disposition
        # (I-FAIL-SAFE).
        envelope = self.artifacts.read_verdict(role.name)
        if envelope is None and result.success:
            # Observability (reviewer_comprehensive amendment, task 2484):
            # a missing meta-root would make the verdict-tools server
            # silently no-op its write, and read_verdict() then returns
            # None indistinguishably from a reviewer that simply never
            # called submit_review_verdict — mirrors the merger's
            # pre-existing diagnostic (workflow.py:_resolve_and_resubmit,
            # itself a reviewer_comprehensive amendment on task 2483).
            # This is purely diagnostic; the fail-safe ERROR outcome below
            # is unchanged either way.
            verdicts_dir = self.artifacts.root / 'verdicts'
            if not verdicts_dir.is_dir():
                logger.warning(
                    'Task %s: reviewer %s verdict absent AND %s does not '
                    'exist — likely a meta-root misconfiguration, not a '
                    'reviewer no-op',
                    self.task_id, role.name, verdicts_dir,
                )
            else:
                logger.warning(
                    'Task %s: reviewer %s verdict absent (verdicts dir %s '
                    'exists) — reviewer did not call submit_review_verdict',
                    self.task_id, role.name, verdicts_dir,
                )
        payload = envelope.get('verdict') if isinstance(envelope, dict) else None
        if (
            not result.success
            or not isinstance(payload, dict)
            or payload.get('verdict') not in {'PASS', 'ISSUES_FOUND'}
        ):
            return {
                'reviewer': role.name,
                'verdict': 'ERROR',
                'issues': [],
                'summary': f'Reviewer emitted no/invalid verdict: {result.output[:200]}',
            }
        return payload

    def _suggestions_in_scope(self, suggestions: list[dict]) -> list[dict]:
        """Filter suggestions to those whose location falls within a module
        this task already holds a lock for.

        Module-lock membership is the scheduler's own concurrency invariant
        (see ``scheduler.normalize_lock``). Filtering this way guarantees an
        amendment pass can't expand the task's lock footprint, and handles
        new files created inside a locked module by construction (a new path
        under a locked module normalizes to the same module key).
        """
        if not suggestions:
            return []
        locked = set(self.modules)
        if not locked:
            logger.warning(
                'Task %s: empty lock set at amendment filter time; '
                'returning zero in-scope suggestions',
                self.task_id,
            )
            return []
        depth = self.config.lock_depth
        in_scope: list[dict] = []
        for s in suggestions:
            location = (s.get('location') or '').strip()
            if not location:
                continue
            # Location format is 'src/foo.py:42' — strip the line number
            file_path = location.split(':', 1)[0].strip()
            if not file_path:
                continue
            module_key = normalize_lock(file_path, depth)
            if module_key and module_key in locked:
                in_scope.append(s)
        return in_scope

    async def _replan(self, reviews) -> None:
        """Feed review feedback back to architect for re-planning."""
        assert self.worktree is not None and self.artifacts is not None
        feedback = reviews.format_for_replan()
        self.plan = self.artifacts.read_plan()

        prompt = f"""\
The implementation was reviewed and blocking issues were found.

{feedback}

# Current Plan

```json
{json.dumps(self.plan, indent=2)}
```

# Action

Update the plan to address the blocking issues. You may add new steps to the `steps` array, but do NOT remove or reorder existing steps. Set new steps to status "pending". Write the updated plan to `.task/plan.json`.
"""
        await self._invoke(ARCHITECT, prompt, self.worktree)
        self.plan = self.artifacts.read_plan()

    async def _commit_amendment_wip(self, amendment_round: int) -> str | None:
        """Commit uncommitted amendment work as a WIP safety-commit (task 2760).

        ``_amend`` invokes the implementer but never commits — it only appends
        the iteration log and validates plan ownership.  If the implementer
        left file changes UNCOMMITTED, the subsequent VERIFY/REVIEW run on the
        dirty worktree and can pass, yet the merge queue lands only COMMITTED
        branch state — so a reviewed, verified amendment could silently fail to
        land.  This guard enforces the invariant "an amendment round's success
        ⟹ its work is committed on the branch" by committing the dirty tree as
        ``wip: amendment round N`` before the loop re-enters VERIFY/REVIEW.
        Committing also advances HEAD, so task 2750's
        ``{pre_amendment_head}..HEAD`` review delta becomes exactly the
        amendment change instead of empty.

        Reuses the WIP-safety-commit primitive already used by
        :meth:`_inter_iteration_rebase` (``git_ops.commit`` no-ops returning
        None on a clean tree and raises ``WorktreeConflictError`` on a
        conflicted one).  Event emission is fire-and-forget, guarded on
        ``self.event_store is not None`` (mirrors :meth:`_emit_rebase_verify_cost`).

        Returns the WIP commit sha (or None when the worktree was clean).
        """
        assert self.worktree is not None and self.git_ops is not None
        if not await self.git_ops.has_uncommitted_work(self.worktree):
            # Clean tree — a benign no-op amendment left nothing uncommitted to
            # lose.  No commit, no event (task 2760 gates purely on dirty-tree).
            return None
        sha = await self.git_ops.commit(
            self.worktree, f'wip: amendment round {amendment_round}',
        )
        logger.warning(
            'Task %s: amendment round %d left uncommitted work — auto-committed '
            'as WIP %s so VERIFY/REVIEW and the merge queue see committed '
            'branch state (task 2760)',
            self.task_id, amendment_round, (sha or '')[:8],
        )
        if self.event_store is not None:
            self.event_store.emit(
                EventType.amendment_uncommitted_recovered,
                task_id=self.task_id,
                phase='review',
                data={
                    'amendment_round': amendment_round,
                    'recovery': 'auto_commit',
                    'wip_sha': sha,
                },
            )
        return sha

    async def _amend(
        self, in_scope: list[dict], amendment_round: int,
    ) -> bool:
        """Invoke the implementer to apply in-scope review suggestions.

        Amendment passes skip the architect entirely — the plan is frozen,
        no new steps are added, the implementer patches the existing diff
        in place. Scope is enforced by the ``_suggestions_in_scope`` filter
        upstream (module-lock membership) and reinforced in the prompt.

        Returns:
            True on success. False if the amendment overwrote plan.json
            (ownership mismatch) — in that case ``_escalate_plan_overwrite``
            is called and the caller MUST halt the workflow rather than
            continue as if the pass succeeded.
        """
        assert self.worktree is not None and self.artifacts is not None

        self.plan = self.artifacts.read_plan()
        iteration_log, corrupted = self.artifacts.read_iteration_log()
        if corrupted:
            self._escalate_corruption(corrupted)

        prompt = await self.briefing.build_amender_prompt(
            plan=self.plan,
            iteration_log=iteration_log,
            suggestions=in_scope,
            locked_modules=list(self.modules),
            task_id=self.task_id,
        )
        pre_head = await self._get_head_commit()
        await self._invoke(IMPLEMENTER, prompt, self.worktree)

        self.artifacts.append_iteration_log({
            'iteration': self.metrics.execute_iterations,
            'agent': 'implementer',
            'source': 'amendment',
            'amendment_round': amendment_round,
            'suggestions_count': len(in_scope),
            **await self._iteration_commit_provenance(pre_head),
            'summary': (
                f'Amendment round {amendment_round} '
                f'({len(in_scope)} suggestions)'
            ),
        })

        # Validate plan ownership after the pass — amendment must NOT
        # overwrite plan.json. If it did, the session_id stamp will mismatch.
        if not self.artifacts.validate_plan_owner(self.session_id):
            logger.error(
                'Task %s: plan.json ownership mismatch after amendment pass '
                '(round %d) — implementer was instructed not to touch the plan',
                self.task_id, amendment_round,
            )
            self._escalate_plan_overwrite()
            return False

        return True

    async def _try_narrow_plan(self, not_touched: list[str]) -> bool:
        """Give the architect ONE chance to narrow the plan after the
        merge gate flagged declared-but-untouched files.

        Lenient semantics: the architect may keep some flagged entries
        (treating them as genuinely needed); the only hard constraint
        is no NEW files may be added.  The gate's re-check is the
        source of truth for pass/fail.

        Returns True when the narrowing pass should be re-checked
        against the gate, False otherwise (one-shot already fired,
        architect invocation failed, or post-pass subset check rejected
        the new plan because it introduced new files).
        """
        if self._plan_tightened:
            return False
        self._plan_tightened = True

        before = set(self.plan.get('files', []))
        prompt = await self.briefing.build_plan_tightening_prompt(
            self.task, self.plan, not_touched, worktree=self.worktree,
        )
        assert self.worktree is not None
        assert self.artifacts is not None
        result = await self._invoke(ARCHITECT, prompt, self.worktree)
        if not result.success:
            return False

        self.plan = self.artifacts.read_plan()
        after = set(self.plan.get('files', []))
        if not after.issubset(before):
            logger.warning(
                'Task %s: narrowing pass added new files %s — rejecting',
                self.task_id, sorted(after - before),
            )
            return False
        # Lenient: partial narrowing or confirm_plan() both fall through
        # to the gate's re-check below.
        return True

    async def _submit_to_merge_queue(
        self, branch_name: str, pre_rebased: bool = False,
        *, merge_phase: bool = False,
    ) -> WorkflowOutcome:
        """Submit a merge request to the queue and await the result.

        The merge worker handles merging, verification, and CAS
        advancement of main.  Conflicts are returned immediately —
        this method resolves them in the task worktree (outside the
        queue) and re-submits.

        When *merge_phase* is True, escalations created by failure
        paths suppress task-status transitions — the caller retries
        the merge in-place instead of requeueing via the scheduler.
        """
        from orchestrator.merge_disposition import MergeFailureDisposition
        from orchestrator.merge_queue import (
            PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
            AttachAction,
            MergeRequest,
            OutcomeKind,
            WaiterRecord,
            _check_plan_files_touched_in_branch,
            _emit_merge_attempt,
            _emit_merge_coalesced,
            register_and_enqueue_merge_request,
            resolve_attach_action,
        )

        assert self.worktree is not None
        assert self.merge_queue is not None

        # Decision-1 pre-merge subset check: refuse to merge a branch that
        # never touched the architect's declared plan files.  Before
        # escalating, give the architect ONE bounded chance to narrow the
        # plan against current branch state — only the architect (not the
        # steward) is allowed to mediate, and only to drop genuinely
        # unneeded entries.  Adding new files is rejected by the helper's
        # subset check.  The gate's re-check is the source of truth for
        # pass/fail.
        if (
            self._task_files
            and self._base_commit is not None
        ):
            # Cross-repo deliverable short-circuit (task 3004): when every
            # declared plan file belongs to another project, this task's branch
            # is legitimately EMPTY (the deliverable lands on the other
            # project's branch — the reify-task 5308 shape).  Routing it through
            # the not-touched gate below would false-flag the empty branch and
            # drag the architect through a dishonest narrowing pass, so recognise
            # it first and route to the honest ``plan_files_cross_repo`` terminal
            # outcome on the NORMAL ladder (no forced ``escalate_to_human``).
            # Import directly from merge_gates (NOT the merge_queue shim) to keep
            # hot merge_queue.py out of this task's lock scope.
            from orchestrator.merge_gates import (
                CROSS_REPO_DELIVERABLE_REASON_PREFIX,
                is_cross_repo_task,
            )
            if is_cross_repo_task(
                list(self._task_files),
                self.config.project_root,
                self.task.get('metadata'),
            ):
                _emit_merge_attempt(
                    self.event_store, self.task_id,
                    OutcomeKind.plan_files_cross_repo,
                )
                reason = (
                    f'{CROSS_REPO_DELIVERABLE_REASON_PREFIX}: every declared '
                    f'plan file belongs to another project, so this task\'s '
                    f'branch is legitimately empty — the deliverable lands on '
                    f'the other project\'s branch.  Verify the external landing '
                    f'rather than re-running this (empty) branch.'
                )
                external_deps = (self.task.get('metadata') or {}).get('external_deps')
                if external_deps:
                    reason += f' external_deps={external_deps}.'
                return await self._mark_blocked(
                    reason,
                    merge_phase=merge_phase,
                    category='cross_repo_deliverable',
                    suggested_action='verify_external_landing',
                )
            rc, branch_head, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
            )
            if rc == 0 and branch_head.strip():
                check = await _check_plan_files_touched_in_branch(
                    list(self._task_files),
                    self._base_commit,
                    branch_head.strip(),
                    self.git_ops,
                    task_id=self.task_id,
                )
                narrowing_succeeded = False
                if check.not_touched:
                    narrowed = await self._try_narrow_plan(check.not_touched)
                    if narrowed:
                        # Re-check the (possibly narrowed) plan against the
                        # gate.  ``_task_files`` reads from ``self.plan``,
                        # which ``_try_narrow_plan`` has refreshed from disk.
                        rc2, branch_head2, _ = await _run(
                            ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
                        )
                        if rc2 == 0 and branch_head2.strip():
                            check = await _check_plan_files_touched_in_branch(
                                list(self._task_files or []),
                                self._base_commit,
                                branch_head2.strip(),
                                self.git_ops,
                                task_id=self.task_id,
                            )
                            if not check.not_touched:
                                narrowing_succeeded = True
                if check.not_touched:
                    reason = (
                        f'{PLAN_FILES_NOT_TOUCHED_REASON_PREFIX}: '
                        f'{", ".join(check.not_touched)}. '
                        f'The architect declared these files but no commit '
                        f'on the branch touched them.  Implementation has '
                        f'not delivered against the plan.'
                    )
                    _emit_merge_attempt(
                        self.event_store, self.task_id,
                        OutcomeKind.plan_files_not_touched,
                    )
                    return await self._mark_blocked(
                        reason,
                        merge_phase=merge_phase,
                        escalate_to_human=True,
                    )
                if narrowing_succeeded:
                    # Distinguish honest over-declaration (handled here)
                    # from human-triage-required (emitted just above).
                    _emit_merge_attempt(
                        self.event_store, self.task_id,
                        OutcomeKind.plan_files_narrowed,
                    )

        # Belt-and-braces rebind (task-1923): re-assert refs/heads/task/<id>
        # == self.worktree HEAD immediately before every enqueue so the named
        # ref the merge worker resolves (merge_to_main → resolve_queued_branch_ref
        # by name) always matches the worktree HEAD.  This defends against the
        # live-requeue residual window where _reuse_warm_lane ran commit+rebase
        # on a DETACHED HEAD (leaving refs/heads/task/<id> stale) and the
        # rebind in _reuse_warm_lane was somehow not reached (e.g. a future code
        # path that bypasses it).  Best-effort: the helper never raises.
        await self.git_ops.rebind_branch_to_head(
            self.worktree,
            f'{self.git_ops.config.branch_prefix}{branch_name}',
        )

        # Stamp write-once first-submission epoch before construction so every
        # submit path (including coalesced/attached) records the lineage's age.
        # The stamp is a no-op (fast-path return) when metadata already carries
        # the value (resubmit / post-restart re-dispatch).
        first_enqueued_at = await self._stamp_first_merge_enqueue()

        future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        from orchestrator.merge_queue import QueuedBranch, lane_for_task_metadata
        merge_request = MergeRequest(
            task_id=self.task_id,
            branch=QueuedBranch.parse(branch_name, self.config.git.branch_prefix),
            worktree=self.worktree,
            pre_rebased=pre_rebased,
            task_files=self._task_files,
            module_configs=self._module_configs,
            config=self.config,
            result=future,
            lane=lane_for_task_metadata(self.task.get('metadata')),
            merge_first_enqueued_at=first_enqueued_at,
        )

        attached = False
        _registry = self.merge_inflight_registry
        if _registry is not None and _registry.entry(branch_name) is not None:
            # Synchronously capture entry state before any await (I10 await-gap).
            _entry = _registry.entry(branch_name)
            old_tip = _entry.snapshot_tip if _entry is not None else None
            verifying = _entry.verifying if _entry is not None else False

            rc_tip, tip_out, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
            )
            new_tip = tip_out.strip() if rc_tip == 0 and tip_out.strip() else None

            if not (old_tip and new_tip):
                # Cannot classify topology when either tip is unavailable (e.g.
                # rev-parse failed or entry has no snapshot_tip yet).  Fall
                # through to an independent enqueue rather than blindly attaching
                # — coalescing against an older tip could resolve this workflow
                # as DONE before its newer commits are included.
                logger.debug(
                    f'Task {self.task_id}: coalesce skipped — tips unavailable '
                    f'(old={old_tip!r}, new={new_tip!r}); enqueueing independently.'
                )
            else:
                # Tip-relation classification via the shared helper — same
                # classification decision as the MCP coalesce path.  Action
                # handling differs: on RESNAPSHOT the workflow coalesces the
                # new request as a peer waiter (see else-branch below); the MCP
                # path independent-enqueues instead (see coalesce_or_enqueue).
                action = await resolve_attach_action(
                    new_tip, old_tip, verifying=verifying, git_ops=self.git_ops,
                )
                if action is AttachAction.RESNAPSHOT:
                    _registry.re_snapshot(branch_name, new_tip)

                if action is AttachAction.ATTACH_AND_CHAIN:
                    # SUPERSET delta arrived while the primary is in its verify
                    # phase (verifying=True on the OLD snapshot).  Attach-as-peer
                    # would mirror the primary's terminal outcome without
                    # independently merging or verifying the SUPERSET delta —
                    # a silent drop.  Leave attached=False so control falls
                    # through to register_and_enqueue_merge_request, giving the
                    # new tip its own merge+verify on its own future.
                    # Full registry slot-handoff (gen-(n+1) coordination) remains
                    # γ3 / task-1641 scope; this removes the silent no-op.
                    logger.info(
                        f'Task {self.task_id}: ATTACH_AND_CHAIN — SUPERSET delta '
                        f're-enqueued independently for its own merge+verify pass.'
                    )
                else:
                    waiter = WaiterRecord(
                        request_id=merge_request.request_id,
                        future=merge_request.result,
                        source='workflow',
                        submitted_tip=new_tip,
                    )
                    # attach() may return False if the entry was released during the
                    # classify await (I10 await-gap).  Fall through to enqueue in that case.
                    attached = _registry.attach(branch_name, waiter)
                    if attached:
                        _emit_merge_coalesced(
                            self.event_store, merge_request, 'workflow',
                            _registry.eta_seconds(branch_name),
                        )

        _acquired = False
        if not attached:
            _acquired = await register_and_enqueue_merge_request(
                self.merge_queue, merge_request, self.event_store, self.merge_inflight_registry,
            )

        # task 2753: the request is now on the durable, crash-safe merge journal
        # (merge_queued emitted) — whether attached as a peer or acquired as the
        # sole waiter, a self-redeploy from here on is recovered on restart. Clear
        # the pre-enqueue MERGE-phase grace stamp immediately so it stops
        # withholding the redeploy; the grace protected ONLY the pre-enqueue
        # window. (A defensive clear in the harness _run_slot finally covers
        # abnormal exits before this point.)
        self.scheduler.clear_merge_phase(self.task_id)
        # Task 2991: discharge the DURABLE merge-phase-liveness stamp at the same
        # boundary — the pre-enqueue window the orphan reaper's
        # _has_fresh_merge_phase gate protects is over, so a lingering fresh
        # stamp would briefly defer an unrelated stranded divergence orphan.
        # Unlike the in-memory grace stamp above, this one is deliberately NOT
        # cleared defensively in the harness _run_slot finally: an abnormal exit
        # before this point must LEAVE the stamp so the reaper keeps deferring
        # across the crash/restart/redispatch window.
        #
        # KNOWN, ACCEPTED GAP (review amendment): discharging here leaves the
        # POST-enqueue window (queue wait + worker rebase/verify/merge) with no
        # durable liveness signal — the stamp is gone and routing.latest stays
        # stale, since the merge worker makes no LLM calls either. In-process
        # that window IS covered: the task is still in _dispatched, so
        # is_actively_held short-circuits the reaper. After a restart it is
        # NOT: Harness._recover_pending_merges rebuilds the request from the
        # merge journal and hands it to the WORKER, which never re-enters
        # _run_merge_phase (nothing re-stamps), and _reconcile_stranded_in_
        # progress LEAVEs — does not re-dispatch — a task that has an open
        # escalation, which the merge-entry divergence L0 is. So a restart
        # while the merge is queued/in-flight still promotes that L0.
        # Accepted here rather than fixed: closing it means moving the
        # discharge to the merge terminal outcome, which changes the
        # stamp/clear pairing this task deliberately co-located with task
        # 2753's in-memory clear_merge_phase (pinned by TestEnqueueClearWiring)
        # — filed as follow-up work. The gap is pinned meanwhile by
        # test_orphan_l0_reaper.py::...::
        # test_post_enqueue_restart_still_promotes_known_accepted_gap, so it is
        # visible in the suite instead of silently absent from it.
        await self._clear_merge_phase_entered()

        # Soft-cancel hook: detach the workflow waiter instead of cancelling
        # the future so the primary entry (and any remaining peers) stay alive.
        # detach() cancels primary_future only when the waiter count hits 0,
        # preserving the existing orphan-avoidance path (merge_queue.py).
        # Route through detach() only when this request is registered in the
        # registry (attached as peer OR acquired as sole waiter); otherwise fall
        # back to future.cancel() to avoid the orphan the old code prevented.
        _req_id = merge_request.request_id
        def _on_soft_cancel_detach() -> None:
            if _registry is not None and (attached or _acquired):
                _registry.detach(branch_name, _req_id)
            elif not future.done():
                future.cancel()

        # Race the future against the cancel event so a human marking the
        # task done out-of-band exits the workflow promptly instead of
        # waiting for the merge worker to finish. W9-θ: a cancel-win now
        # raises WorkflowCancelled('soft') (propagating to run()'s single
        # catch) instead of returning None — no `result is None` branch.
        result = await self._await_cancellable(
            future,
            on_soft_cancel=_on_soft_cancel_detach,
        )

        if result.status == 'wip_halted':
            return await self._handle_wip_conflict(result, branch_name)
        if result.status == 'done_wip_recovery':
            return await self._handle_wip_recovery(result)
        if result.status == 'wip_recovery_no_advance':
            return await self._handle_wip_recovery_no_advance(result)
        if result.status == 'unmerged_state':
            return await self._handle_unmerged_state(result, branch_name)
        if result.status == 'stash_failed':
            return await self._handle_stash_failed(result, branch_name)
        if result.status == 'done':
            if result.merge_sha is not None:
                self._merge_sha = result.merge_sha
            return WorkflowOutcome.DONE
        if result.status == 'already_merged':
            logger.info(f'Task {self.task_id}: already merged to main')
            return WorkflowOutcome.DONE
        if result.status == 'superseded':
            return await self._handle_superseded(result)
        if result.status == 'conflict':
            return await self._resolve_and_resubmit(
                branch_name, result.conflict_details,
                merge_phase=merge_phase,
            )
        # ``blocked`` — but first check for the worktree-missing race: if a
        # human marked the task ``done`` and removed the worktree while the
        # merge was in flight, the merge worker surfaces a known reason
        # prefix.  Re-read task status; if terminal, exit cleanly without
        # creating an escalation.
        from orchestrator.merge_queue import WORKTREE_MISSING_REASON_PREFIX
        if result.reason.startswith(WORKTREE_MISSING_REASON_PREFIX):
            try:
                status = await self.scheduler.get_status(self.task_id)
            except Exception:
                logger.exception(
                    f'Task {self.task_id}: get_status failed during '
                    f'worktree-missing fallback; falling through to blocked'
                )
                status = None
            if status in TERMINAL_STATUSES:
                logger.info(
                    f'Task {self.task_id}: worktree missing but task '
                    f'status={status!r} (terminal) — exiting DONE without '
                    f'escalation'
                )
                return WorkflowOutcome.DONE
        # Drop-guard short-circuit: a real merger-drop is the human-judgement
        # case the gate exists for.  Steward mediation (e.g. mutating plan.json
        # to silence the gate) would undermine the safeguard, so skip the L0
        # steward path entirely and submit an L1 immediately.
        from orchestrator.merge_queue import (
            DROPPED_PLAN_TARGETS_REASON_PREFIX,
            MAIN_HEALTH_RED_REASON_PREFIX,
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
            POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
            TRANSIENT_INFRA_REASON_PREFIX,
        )
        if result.reason.startswith(DROPPED_PLAN_TARGETS_REASON_PREFIX):
            self._write_merge_failure_review('dropped_plan_targets', result.reason)
            return await self._mark_blocked(
                result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
            )
        # Decision-2 short-circuit: post-merge equivalence is also a
        # human-judgement case (conflict-resolution drops / rebase
        # regressions); same L1-not-steward routing.
        if result.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX):
            self._write_merge_failure_review(
                'post_merge_equivalence_failed', result.reason,
            )
            return await self._mark_blocked(
                result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
            )
        # Post-merge unscoped type-check short-circuit: a union break caught
        # after the merge landed is also a human-judgement fix-forward case;
        # same L1-not-steward routing so the steward cannot mask the signal.
        # spawn_dry_run=True captures a sha-anchored dry-run proposal for the
        # B3 low-risk auto-unblock gate: advance_main has already moved
        # refs/heads/main to advanced_sha before this path runs, so
        # _capture_worktree_shas reflects post-merge reality at spawn time.
        if result.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX):
            self._write_merge_failure_review(
                'post_merge_pyright_broken', result.reason,
            )
            return await self._mark_blocked(
                result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
                spawn_dry_run=True,
            )
        # Transient-infra short-circuit: the merge worker already pruned stale
        # merge worktrees and retried the verify, and it still hit ENOSPC.  Go
        # straight to a human-facing L1 tagged ``infra_issue`` (not the
        # steward — it can't free disk).  The durable ref + infra_issue
        # category let the escalation-watcher auto-resolve if the disk has
        # recovered by read-time.
        if result.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX):
            self._write_merge_failure_review('transient_infra', result.reason)
            return await self._mark_blocked(
                result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
                category='infra_issue',
            )
        # Main-health-red short-circuit: the failure reproduces on bare main HEAD
        # (pre-existing break, not introduced by this task's merge).  Dispatch
        # to _auto_heal_main_health which implements the full GATED auto-heal:
        # halt the normal lane, submit a dedup'd escalation, spawn a HIGH-lane
        # fix task, and block.  Non-mechanical breaks and attempt-cap exceedances
        # fall back to escalate-to-human directly inside _auto_heal_main_health.
        if result.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX):
            self._write_merge_failure_review('main_health_red', result.reason)
            return await self._auto_heal_main_health(result, merge_phase=merge_phase)
        # Merge-skew short-circuit (task 2383 β, M2 of
        # plans/merge-skew-attribution-prd.md): the branch verified green
        # pre-merge, but a landing on main overlapping the failing test's
        # file(s) is implicated — this is a "port the landed change" case,
        # not a bug on the branch.  Route straight to a human-facing L1
        # tagged ``integration_skew`` (not the steward — porting a landed
        # diff requires human/architect judgement, not a retry) so the
        # implicated sha + overlap files carried in *result.reason* by
        # ``_render_skew_surfaces`` are never buried in a generic task-fault
        # escalation.  Checked AFTER the MAIN_HEALTH_RED short-circuit
        # (mutually exclusive: MAIN_RED is stamped only when
        # preexisting=True, INTEGRATION_SKEW only when preexisting=False)
        # and BEFORE the generic blocked path.
        if getattr(result, 'disposition', None) is MergeFailureDisposition.INTEGRATION_SKEW:
            self._write_merge_failure_review('integration_skew', result.reason)
            return await self._mark_blocked(
                result.reason,
                detail=result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
                category='integration_skew',
                suggested_action='port_landed_change',
            )
        # Fix 3 — capture the merge-queue blocked reason so the merge-phase
        # loop can fingerprint it for the thrash check before resubmitting.
        self._last_merge_block_reason = result.reason
        self._last_merge_failure_category = result.failure_category
        self._last_merge_failure_cause_hint = result.failure_cause_hint
        # blocked — infer review category from reason
        if 'verification failed' in result.reason.lower():
            category = 'post_merge_verify'
        elif 'ff' in result.reason.lower() or 'advanced' in result.reason.lower():
            category = 'merge_ff_failed'
        else:
            category = 'merge_error'
        self._write_merge_failure_review(category, result.reason)
        # Unconditional observability (task 2757; Reify 5120 RCA RC-2): emit the
        # durable merge_blocked event BEFORE _mark_blocked and ungated by
        # has_open_l1, so this generic fall-through can never again go
        # `merge_finalized state=blocked` -> total silence when an unrelated open
        # L1 suppresses the escalation at either has_open_l1 gate.  Threading the
        # real review-category into _mark_blocked (not the default 'task_failure')
        # also makes the signature-aware L1 dedup meaningful.
        if self.event_store:
            self.event_store.emit(
                EventType.merge_blocked,
                task_id=self.task_id, phase=self.state.value,
                data={
                    'reason': result.reason[:500],
                    'category': category,
                    'failure_category': result.failure_category,
                    'cause_hint': result.failure_cause_hint,
                },
            )
        return await self._mark_blocked(
            result.reason, merge_phase=merge_phase, category=category,
        )

    async def _spawn_main_health_fix_task(
        self,
        signature: str,
        escalation_id: str,
        category: str,
        cause_hint: str,
        detail: str,
    ) -> None:
        """Schedule a high-lane fix task for a confirmed main-health break.

        Builds a submit_task argument block with:
        - title/description from :func:`~orchestrator.merge_queue.compose_fix_main_brief`
        - ``merge_lane='high'`` so the fix merges via the HIGH lane
        - Correlation keys so the auto-watcher can link it to its escalation:
          ``spawn_context``, ``main_health_signature``, ``main_health_escalation_id``

        Delegates the actual POST to :meth:`_post_submit_tasks` via
        ``asyncio.create_task`` so the caller is not blocked.
        """
        from orchestrator.merge_queue import compose_fix_main_brief

        title, description = compose_fix_main_brief(category, cause_hint, detail)
        arguments = {
            'title': title,
            'description': description,
            'priority': 'high',
            'project_root': str(self.config.project_root),
            'metadata': {
                'merge_lane': 'high',
                'spawn_context': 'main_health_auto_heal',
                'main_health_signature': signature,
                'main_health_escalation_id': escalation_id,
            },
        }
        if self.mcp is not None:
            try:
                _task = asyncio.create_task(
                    self._post_submit_tasks([arguments]),
                    name=f'spawn_fix_main_{self.task_id}',
                )
                self._background_tasks.add(_task)
                _task.add_done_callback(self._background_tasks.discard)
            except Exception as exc:
                logger.warning(
                    'Task %s: failed to spawn main-health fix task: %s',
                    self.task_id, exc,
                )

    async def _auto_heal_main_health(
        self, result: Any, *, merge_phase: bool,
    ) -> WorkflowOutcome:
        """Orchestrate the main-health auto-heal response for a confirmed red-main verdict.

        Branches:
        (d) NOT mechanical (category not in AUTO_HEAL_MECHANICAL_CATEGORIES)
            → escalate to human, no halt, no spawn.
        (e) Attempt cap reached (sha-independent signature recurred after prior heal)
            → hard-escalate, no halt, no spawn.
        (idempotency) Normal lane already halted (auto-heal in flight)
            → fold escalation, no second halt/spawn.
        (a) Happy path: record attempt; halt 'normal' lane; submit one dedup'd
            L1 escalation; register it as halt-owner; spawn a HIGH-lane fix task;
            block via _mark_blocked(skip_escalation=True).
        """
        from escalation.dedupe import DedupeConfig, content_fingerprint_key
        from escalation.models import Escalation

        from orchestrator.merge_queue import (
            MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS,
            is_auto_heal_eligible,
        )

        fp = getattr(result, 'dedupe_fingerprint', None) or None
        category = getattr(result, 'failure_category', None) or ''
        cause_hint = getattr(result, 'failure_cause_hint', None) or ''
        reason = getattr(result, 'reason', str(result))

        # Branch (d): non-mechanical, no merge_worker (cannot halt lanes), or no
        # escalation_queue (cannot register a halt-owner → unhalt_lanes_owned_by can
        # never match → normal lane stays halted permanently → livelock).
        # → escalate-only, no halt/spawn.
        if (
            not is_auto_heal_eligible(category, cause_hint)
            or self.merge_worker is None
            or self.escalation_queue is None
        ):
            return await self._mark_blocked(
                reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
                category='preexisting_main_break',
                dedupe_fingerprint=fp,
                suggested_action='await_preexisting_main_hotfix',
            )

        # Sha-independent signature for attempt-cap / re-break-loop detection.
        # Derived from THIS outcome's (category, cause_hint) — distinct per break,
        # stable across the heal→re-break loop (same category+cause_hint at a new
        # main SHA).  Must NOT use self._merge_outcome_signature() here because the
        # MAIN_HEALTH_RED_REASON_PREFIX fast-path returns before the generic blocked
        # path that sets _last_merge_failure_*; those fields stay at __init__
        # defaults, collapsing every break to the same empty-basis constant key.
        sig = _compute_merge_outcome_signature(category, cause_hint, reason)

        registry = (
            self.merge_worker.auto_heal_registry
            if self.merge_worker is not None else None
        )

        # Branch (e): attempt cap reached — genuine re-break after a prior heal.
        # Only fires when the lane is NOT currently halted.  If an auto-heal is
        # in flight (lane already halted), concurrent same-signature tasks must
        # fold via the idempotency branch below, NOT be mis-classified as
        # re-break loops with a wrong root_cause.  The re-break cap is a
        # guard against a completed-heal → new-break cycle, not an in-flight
        # duplicate arriving while the same heal is still running.
        if (
            registry is not None
            # Runtime-redundant (registry is not None already implies
            # merge_worker is not None per the ternary above); kept so pyright
            # can narrow merge_worker for the .is_lane_halted() call below.
            and self.merge_worker is not None
            and registry.attempts(sig) >= MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS
            and not self.merge_worker.is_lane_halted('normal')
        ):
            return await self._mark_blocked(
                (
                    f'Main-health auto-heal attempt cap reached (signature recurred '
                    f'after auto-heal): {reason[:160]}'
                ),
                detail=reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
                category='preexisting_main_break',
                dedupe_fingerprint=fp,
                root_cause=f'main-health-auto-heal-rebreak:{sig}',
                suggested_action='await_preexisting_main_hotfix',
            )

        # Build and submit the dedup'd L1 escalation (shared across all three
        # remaining branches: idempotency + happy path both file/fold it).
        esc_id: str | None = None
        if self.escalation_queue is not None:
            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='preexisting_main_break',
                summary=reason[:200],
                detail=reason,
                suggested_action='main_health_auto_heal_in_flight',
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            if fp:
                esc.dedupe_fingerprint = fp
                submit_result = submit_or_dedupe(
                    self.escalation_queue,
                    esc,
                    DedupeConfig(
                        infra_dedupe_enabled=True,
                        infra_dedupe_window_secs=float('inf'),
                        infra_dedupe_categories=('preexisting_main_break',),
                        key_fn=content_fingerprint_key,
                    ),
                )
                # Derive esc_id from the SURVIVING (parent) escalation.
                # submit_or_dedupe returns {'id': parent_id, 'status': 'dedup_skipped',
                # 'parent_id': parent_id, 'child_id': esc.id} when the child folds
                # into a pending parent — only the parent resolves, so the halt owner
                # and the fix-task correlation key must reference the parent.
                # For a fresh (non-dedup) submit the dict has no 'parent_id', so we
                # fall back to 'id', then to esc.id as the final safety net.
                submit_result_dict: dict = submit_result if isinstance(submit_result, dict) else {}
                esc_id = (
                    submit_result_dict.get('parent_id')
                    or submit_result_dict.get('id')
                    or esc.id
                )
            else:
                self.escalation_queue.submit(esc)
                esc_id = esc.id

        # Branch (idempotency): lane already halted → adopt the in-flight auto-heal
        if self.merge_worker is not None and self.merge_worker.is_lane_halted('normal'):
            return await self._mark_blocked(
                reason,
                merge_phase=merge_phase,
                skip_escalation=True,
            )

        # Branch (a): happy path — record attempt, halt lane, register owner, spawn fix
        if registry is not None:
            registry.record_attempt(sig)

        if self.merge_worker is not None:
            self.merge_worker.halt_lane(
                'normal',
                f'main-health auto-heal in flight (task {self.task_id})',
            )
            if esc_id:
                self.merge_worker.set_lane_halt_owner('normal', esc_id)

        await self._spawn_main_health_fix_task(
            sig, esc_id or '', category, cause_hint, reason,
        )

        return await self._mark_blocked(
            reason,
            merge_phase=merge_phase,
            skip_escalation=True,
        )

    async def _resolve_and_resubmit(
        self, branch_name: str, conflict_details: str,
        *, merge_phase: bool = False,
    ) -> WorkflowOutcome:
        """Resolve merge conflicts in the task worktree, then re-submit.

        This runs OUTSIDE the merge queue — the worker is free to process
        other merges while this task resolves its conflicts.
        """
        assert self.worktree is not None and self.artifacts is not None
        logger.info(
            f'Task {self.task_id}: merge conflicts detected, '
            f'resolving outside queue'
        )
        task_intent = (
            f"Task: {self.task.get('title', '')}\n"
            f"{self.task.get('description', '')}"
        )
        prompt = await self.briefing.build_merger_prompt(
            conflict_details, task_intent,
        )

        # Rebase onto current main so MERGER works on up-to-date state
        await self.git_ops.rebase_onto_main(self.worktree)
        # I-FRESH: never consume a stale verdict from a prior invocation on
        # this same worktree (mirrors steward.py's pre-triage clear).
        self.artifacts.clear_verdict('merger')
        merger_result = await self._invoke(MERGER, prompt, self.worktree)

        # Read the merger's structured disposition instead of grepping its
        # free-text output for "BLOCKED" (task 2483 / PRD task γ). Defensive
        # extraction mirrors extract_triage_verdict's untrusted-shape
        # contract: only a dict envelope with a dict 'verdict' carrying a
        # bool 'blocked' is trusted; anything else is treated as absent.
        envelope = self.artifacts.read_verdict('merger')
        if envelope is None and merger_result.success:
            # Observability (reviewer_comprehensive amendment, task 2483):
            # a missing meta-root would make the verdict-tools server
            # silently no-op its write, and read_verdict() then returns
            # None indistinguishably from a merger that simply never
            # called submit_merge_disposition — mirrors steward.py's
            # pre-triage meta-root diagnostic (steward.py:701-719). This
            # is purely diagnostic; the fail-safe blocked outcome below is
            # unchanged either way.
            verdicts_dir = self.artifacts.root / 'verdicts'
            if not verdicts_dir.is_dir():
                logger.warning(
                    'Task %s: merger verdict absent AND %s does not '
                    'exist — likely a meta-root misconfiguration, not a '
                    'merger no-op',
                    self.task_id, verdicts_dir,
                )
            else:
                logger.warning(
                    'Task %s: merger verdict absent (verdicts dir %s '
                    'exists) — merger did not call '
                    'submit_merge_disposition',
                    self.task_id, verdicts_dir,
                )
        verdict = envelope.get('verdict') if isinstance(envelope, dict) else None
        if not isinstance(verdict, dict) or not isinstance(verdict.get('blocked'), bool):
            verdict = None

        if not merger_result.success or verdict is None:
            # Fail-safe (I-FAIL-SAFE): an invocation failure, an absent
            # verdict, or a malformed one all block the merge — a merger
            # that produces no trustworthy disposition must never proceed.
            reason = (
                f'Merger invocation failed: {merger_result.output[:200]}'
                if not merger_result.success
                else f'Merger emitted no/invalid disposition: {merger_result.output[:200]}'
            )
            self._write_merge_failure_review('merger_blocked', reason)
            return await self._mark_blocked(reason, merge_phase=merge_phase)

        if verdict['blocked']:
            reason = verdict.get('reason') or f'Merger blocked: {merger_result.output[:200]}'
            self._write_merge_failure_review('merger_blocked', reason)
            return await self._mark_blocked(reason, merge_phase=merge_phase)

        # Re-submit to queue (now resolved, needs fresh merge)
        return await self._submit_to_merge_queue(
            branch_name, pre_rebased=False, merge_phase=merge_phase,
        )

    def _submit_halt_owning_escalation(self, esc: Escalation) -> None:
        """Submit a halt-owning escalation and register halt ownership.  Non-waiting.

        This is the load-bearing submit → set_halt_owner ordering shared by:
          - ``_submit_halt_escalation_and_wait`` (single-task path; follows with an await)
          - ``_escalate_train_halt`` (train path; does not await — tip stays BLOCKED and
            re-dispatches when the L1 is resolved and the queue is unhalted)

        Order is significant: set_halt_owner MUST follow a successful submit.
        A registered owner with no pending escalation cannot be resolved by
        _on_escalation_resolved, so the halt would be permanent.  If submit
        raises, the exception propagates before set_halt_owner is reached and
        no orphan halt (OWNER-WITHOUT-ESCALATION) is registered.

        HALT-WITHOUT-OWNER guard (task 1765): when the merger has already engaged
        a per-lane WIP halt before we get here (``is_wip_halted=True,
        halt_owner_esc_id=None``), a submit failure leaves the lane halted with no
        owner registered.  ``_on_escalation_resolved`` has nothing to match, no
        escalation exists for a human, and the halt silently blocks ALL merges on
        that lane until ``force_unhalt_merge_queue``.  If submit raises and the
        halt is ownerless, we release it before re-raising — same rationale as the
        sibling guards in ``_submit_halt_escalation_and_wait`` (task 1448) and
        ``_map_advance_failure`` (task 1671).

        Guard condition: ``merge_worker.is_wip_halted and halt_owner_esc_id is None``
        — ensures we release ONLY genuine ownerless orphans, never a foreign owner's
        halt.  ``except BaseException`` (not ``Exception``) mirrors the sibling
        guards and also covers a CancelledError arriving between halt-engage and
        owner-registration.

        Callers must guard with ``if self.escalation_queue:`` before calling.
        """
        assert self.escalation_queue is not None, (
            '_submit_halt_owning_escalation requires escalation_queue; '
            'callers must guard with `if self.escalation_queue:`'
        )
        try:
            self.escalation_queue.submit(esc)  # propagates on failure; set_halt_owner NOT reached
        except BaseException:
            # HALT-WITHOUT-OWNER guard (task 1765): the merger may have already
            # engaged a per-lane WIP halt before we got here. If submit raises,
            # set_halt_owner is unreachable, so an engaged-but-ownerless halt would
            # silently block the whole lane until force_unhalt_merge_queue. Release
            # the orphan halt before propagating — only when it is genuinely
            # ownerless (never steal a foreign owner's halt). Mirrors the sibling
            # guards in _submit_halt_escalation_and_wait and _map_advance_failure.
            if (
                self.merge_worker is not None
                and self.merge_worker.is_wip_halted
                and self.merge_worker.halt_owner_esc_id is None
            ):
                try:
                    self.merge_worker.unhalt_wip(reason='halt_escalation_submit_failed')
                except Exception:
                    # Swallow unhalt_wip failures so the original submit exception
                    # always propagates to the caller — masking a disk-full or
                    # serialization error with a secondary unhalt failure would make
                    # diagnosis much harder.
                    logger.exception(
                        'Task %s: unhalt_wip failed during submit-failure cleanup '
                        '(original submit exception will still propagate)',
                        self.task_id,
                    )
            raise
        if self.merge_worker is not None:
            self.merge_worker.set_halt_owner(esc.id)

    async def _submit_halt_escalation_and_wait(self, esc: Escalation) -> None:
        """Submit a halt-owning escalation, register ownership, and wait for resolution.

        Delegates the load-bearing submit → set_halt_owner ordering to
        ``_submit_halt_owning_escalation``, then awaits ``_escalation_event``.

        On cancellation or any other BaseException raised after set_halt_owner
        (including event_store.emit failures or task cancellation during the
        await), releases the halt via unhalt_wip(reason='workflow_cancelled')
        iff this workflow still owns it, then re-raises so the cancellation
        propagates to the caller's Task.
        """
        assert self.escalation_queue is not None, (
            '_submit_halt_escalation_and_wait requires escalation_queue; '
            'callers must guard with `if self.escalation_queue:`'
        )
        self._submit_halt_owning_escalation(esc)  # submit + set_halt_owner (load-bearing order)
        # try/except starts here so emit, event setup, AND the await are all
        # protected — any BaseException after set_halt_owner triggers cleanup.
        try:
            if self.event_store:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=self.task_id, phase=self.state.value,
                    data={
                        'escalation_id': esc.id,
                        'category': esc.category,
                        'severity': esc.severity,
                        'summary': esc.summary[:200],
                    },
                )
            if self._escalation_event is None:
                self._escalation_event = asyncio.Event()
            self._escalation_event.clear()
            await self._escalation_event.wait()
        except BaseException:
            if (
                self.merge_worker is not None
                and self.merge_worker.is_halt_owner(esc.id)
            ):
                self.merge_worker.unhalt_wip(reason='workflow_cancelled')
            raise

    def _build_wip_halt_escalation_text(
        self,
        status: str,
        result,
        *,
        branch_name: str | None = None,
        train_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Build ``(category, summary, detail)`` for a WIP-halt level-1 escalation.

        Shared by both the single-task handlers (``branch_name=...``) and the
        train consumer ``_escalate_train_halt`` (``train_id=...``), so that
        human-readable recovery instructions live in one place and cannot drift.

        The task-context label (``'Merge for task X (branch Y)'`` vs
        ``'Train merge for task X (train Y)'``) and one train-specific
        split-brain note for ``done_wip_recovery`` are the only per-path
        variations.

        Parameters
        ----------
        status:
            The ``MergeOutcome.status`` string.  Handled: ``'unmerged_state'``,
            ``'stash_failed'``, ``'done_wip_recovery'``,
            ``'wip_recovery_no_advance'``, ``'wip_halted'`` (default branch).
        result:
            The ``MergeOutcome`` object (for ``overlap_files``,
            ``recovery_branch``, ``dirty_files``).
        branch_name:
            Single-task context — the local branch being merged.
            Mutually exclusive with ``train_id``.
        train_id:
            Train context — the train identifier string.
            Mutually exclusive with ``branch_name``.

        Returns
        -------
        ``(category, summary, detail)`` :
            Ready for ``Escalation(category=..., summary=..., detail=...)``.
        """
        is_train = train_id is not None
        task_id = self.task_id

        # Context label used in the detail intro sentence.
        if is_train:
            merge_ctx = f'Train merge for task {task_id} (train {train_id!r})'
        elif branch_name is not None:
            merge_ctx = f'Merge for task {task_id} (branch {branch_name})'
        else:
            merge_ctx = f'Merge for task {task_id}'

        if status == 'unmerged_state':
            category = 'unmerged_state'
            if is_train:
                summary = (
                    f'Train member {task_id} blocked: '
                    f'project_root has unresolved (UU/AA/DD) markers'
                )
            else:
                summary = (
                    'project_root has unresolved (UU/AA/DD) markers — '
                    'merge queue halted'
                )
            detail = (
                f'{merge_ctx} was blocked because project_root already has unresolved '
                f'merge conflicts (UU/AA/DD markers) from a prior, unrelated event — '
                f'the merge queue refuses to stash/advance over a partially resolved tree.\n\n'
                f'Action required: inspect ``git status`` in project_root, '
                f'resolve the existing merge state (``git mergetool`` / edit '
                f'the conflicted files / ``git reset`` to abandon the prior '
                f'merge), then resolve this escalation to un-halt the merge queue.\n\n'
                f'Manual intervention required — do NOT let automated tooling '
                f'resolve this escalation.'
            )

        elif status == 'done_wip_recovery':
            category = 'wip_conflict'
            recovery_branch = result.recovery_branch or '(unknown)'
            if is_train:
                summary = (
                    f'Train member {task_id}: stash pop conflict — '
                    f'WIP preserved on {recovery_branch}'
                )
                wip_clause = (
                    'uncommitted WIP produced conflicts '
                    '(split-brain: merge landed, members NOT flipped to done).'
                )
                # Task 1599 / Suggestion 4: the merge is already on main.
                # Instruct the human to flip all members done before resolving
                # so the re-dispatch does not re-attempt the already-landed merge.
                post_resolve = (
                    'Note: the merge commit is already on main.  Before resolving this '
                    'escalation, ensure all train members are marked done — the train '
                    're-dispatches on unhalt and re-entering the merge queue with an '
                    'already-merged tip may produce a confusing split result.\n\n'
                    'Resolve this escalation to un-halt the merge queue.'
                )
            else:
                summary = f'Stash pop conflict — WIP preserved on {recovery_branch}'
                wip_clause = 'your uncommitted WIP produced conflicts.'
                post_resolve = 'Resolve this escalation to un-halt the merge queue.'
            detail = (
                f'{merge_ctx} landed on main successfully, but the stash pop of '
                f'{wip_clause}\n\n'
                f'Your WIP has been preserved on branch: {recovery_branch}\n\n'
                f'To recover:\n'
                f'  git checkout {recovery_branch}\n'
                f'  # Review and cherry-pick or reapply your changes\n\n'
                f'{post_resolve}'
            )

        elif status == 'wip_recovery_no_advance':
            category = 'wip_conflict'
            recovery_branch = result.recovery_branch or '(unknown)'
            if is_train:
                summary = (
                    f'Train member {task_id}: stash pop conflict (no advance) — '
                    f'WIP on {recovery_branch}'
                )
                no_advance_intro = (
                    f'{merge_ctx} did NOT advance main (CAS failure path). '
                    f'A stash pop conflict occurred, leaving'
                )
            else:
                summary = (
                    f'Stash pop conflict (merge did not advance) — '
                    f'WIP on {recovery_branch}'
                )
                no_advance_intro = (
                    f'{merge_ctx} did NOT advance main. '
                    f'A stash pop conflict occurred after a CAS failure, leaving'
                )
            detail = (
                f'{no_advance_intro} the working tree in an unresolvable state.\n\n'
                f'Your WIP has been preserved on branch: {recovery_branch}\n\n'
                f'The merge queue has been halted. To recover:\n'
                f'  git checkout {recovery_branch}\n'
                f'  # Review and reapply your changes to a clean branch\n\n'
                f'Manual intervention required — do NOT let automated tooling '
                f'resolve this escalation. Resolve this escalation to un-halt '
                f'the merge queue.'
            )

        elif status == 'stash_failed':
            # SHARED main-checkout-hygiene fault (task 2758): advance_main could
            # not park project_root's dirty tracked WIP before advancing. Unlike
            # wip_halted (WIP overlapping the merge diff), these files need not
            # overlap the diff at all — project_root's tree is persistently
            # dirty, so EVERY task landing fails identically until a human
            # cleans it. The queue is halted so exactly ONE escalation is filed.
            category = 'stash_failed'
            dirty = result.dirty_files or []
            dirty_head = ', '.join(dirty[:5]) if dirty else '(unknown)'
            if is_train:
                summary = (
                    f'Train member {task_id} blocked: project_root park failed '
                    f'(dirty tracked tree) — {dirty_head}'
                )
            else:
                summary = (
                    f'project_root park failed (dirty tracked tree) — '
                    f'merge queue halted: {dirty_head}'
                )
            dirty_block = (
                '\n'.join(f'  - {f}' for f in dirty) if dirty
                else '  (dirty tracked file list unavailable)'
            )
            detail = (
                f'{merge_ctx} was blocked because advance_main could not park '
                f'the uncommitted WIP in project_root before advancing: '
                f'project_root has dirty TRACKED file(s) that could not be '
                f'stashed. This is a shared main-checkout infrastructure/hygiene '
                f'incident — it recurs identically for every task landing until '
                f'project_root is cleaned, so the merge queue has been halted '
                f'(one escalation, not one per stalled task).\n\n'
                f'Dirty tracked file(s):\n'
                f'{dirty_block}\n\n'
                f'Action required: inspect ``git status`` in project_root, then '
                f'commit or clean the persistently-dirty tracked file(s) '
                f'(``git commit`` / ``git checkout -- <path>`` / ``git stash``), '
                f'then resolve this escalation to un-halt the merge queue.\n\n'
                f'Manual intervention required — do NOT let automated tooling '
                f'resolve this escalation.'
            )

        else:
            # wip_halted (and any future halt-inducing status the probe catches)
            category = 'wip_conflict'
            overlap = result.overlap_files or []
            if is_train:
                summary = (
                    f'Train member {task_id}: WIP overlaps merge diff: '
                    + ', '.join(overlap[:5])
                )
                subject_ctx = f'train member {task_id} (train {train_id!r})'
            else:
                summary = f'WIP overlaps merge diff: {", ".join(overlap[:5])}'
                branch_str = f' (branch {branch_name})' if branch_name else ''
                subject_ctx = f'task {task_id}{branch_str}'
            detail = (
                f'Merge for {subject_ctx} was blocked because uncommitted '
                f'work in project_root overlaps the merge diff.\n\n'
                f'Overlapping files:\n'
                + '\n'.join(f'  - {f}' for f in overlap)
                + '\n\nAction required: commit or stash the WIP, then resolve this '
                'escalation to un-halt the merge queue and retry.'
            )

        return category, summary, detail

    def _warn_orphan_halt_no_queue(
        self,
        status: str,
        *,
        recovery_branch: str | None = None,
        train_id: str | None = None,
    ) -> None:
        """Log the orphan-halt diagnostic when ``escalation_queue`` is ``None``.

        Called by the five WIP-halt handlers when the merger has already engaged a
        per-lane halt but ``escalation_queue`` is ``None`` — the halt is left
        ownerless with no escalation filed and no human notified.

        Always contains the tokens in ``_ORPHAN_HALT_NO_QUEUE_TOKENS``
        (``'orphan halt'`` and ``'unhalt_merge_queue'``) so operators can grep
        for the actionable hint and tests can assert without coupling to free-text
        prose.  Use the module-level constant in assertions rather than the
        literal strings.

        Parameters
        ----------
        status:
            The ``MergeOutcome.status`` string (e.g. ``'wip_halted'``).
        recovery_branch:
            For WIP-recovery outcomes; appended to the context label.
        train_id:
            For the train path (``_escalate_train_halt``); replaces the
            status-based context label with ``'train <train_id>'``.
        """
        if train_id is not None:
            context = f'train {train_id!r}'
        elif recovery_branch is not None:
            context = f'{status}, recovery_branch={recovery_branch!r}'
        else:
            context = status
        logger.warning(
            'Task %s: merge queue is halted (%s) but escalation_queue is None — '
            'halt owner cannot be registered; manual unhalt_merge_queue required '
            'to clear the orphan halt',
            self.task_id, context,
        )

    async def _reverify_one_member(
        self,
        member: dict,
        predecessor_ref: str,
    ) -> SoloVerifyResult:
        """Un-stack a member's own delta onto main and verify it in isolation.

        1. Call ``git_ops.materialize_member_solo(member_id, predecessor_ref)``
           to create an isolated worktree carrying only this member's commits.
        2. If the un-stack fails (returns None — rebase conflict), return a
           failed result with reason='unstackable'.
        3. Otherwise call ``merge_queue.reverify_member_solo`` (verify-only;
           never advances main) and return its result.
        """
        from orchestrator.merge_queue import (
            SoloVerifyResult,
            reverify_member_solo,
        )

        member_id = str(member.get('id', ''))
        solo_wt_info = await self.git_ops.materialize_member_solo(
            member_id, predecessor_ref,
        )
        if solo_wt_info is None:
            return SoloVerifyResult(
                member_id=member_id,
                passed=False,
                merge_sha=None,
                reason='unstackable',
            )
        # WorktreeInfo carries path + base_commit (=rebased tip SHA).
        # The solo branch name mirrors materialize_member_solo's default prefix.
        solo_branch = f'_solo-{member_id}'
        return await reverify_member_solo(
            self.git_ops,
            member_id=member_id,
            solo_wt=solo_wt_info.path,
            solo_branch=solo_branch,
            tip_sha=solo_wt_info.base_commit,
            config=self.config,
            task_files=self._task_files,
            module_configs=self._module_configs,
            event_store=self.event_store,
        )

    async def _attribute_train_failure(
        self,
        result: object,
        train_id: str,
        members: list[dict],
    ) -> WorkflowOutcome:
        """Re-verify each train member as a solo to attribute a union-verify failure.

        Called from ``_maybe_enqueue_group_merge`` when the result carries a
        ``TRAIN_VERIFY_FAILED_REASON_PREFIX`` tag — meaning the train's post-merge
        verification failed with an interaction-candidate failure category.

        Design (δ):
        - For each member root→tip, un-stack its own delta onto current main via
          ``_reverify_one_member`` (wraps materialize_member_solo +
          reverify_member_solo; never advances).  Exactly N solo verifies.
        - Partition into passers / failers (un-stack conflict → failer).
        - All pass → genuine cross-member INTERACTION: emit train_derailed
          (verdict='interaction'), escalate the train, land nothing.
        - Some fail → land each passer, block each failer (steps 12/14).
        """
        # Sort members by their metadata train 'order' (root→tip) so that the
        # predecessor_ref computation is correct regardless of caller-side ordering.
        # tasks_by_train normally returns them sorted, but attributing the right
        # delta to each member depends on strict stacking order — a silently
        # mis-ordered list would rebase against the wrong predecessor, either
        # producing a spurious conflict (mis-classified 'unstackable') or
        # extracting the wrong delta.
        members = sorted(
            members,
            key=lambda m: (m.get('metadata') or {}).get('train', {}).get('order', 0),
        )

        member_ids = [str(m.get('id', '')) for m in members]
        branch_prefix = self.git_ops.config.branch_prefix
        main_branch = self.git_ops.config.main_branch

        logger.info(
            'Task %s: train %r union verify failed — re-verifying %d members as singles',
            self.task_id, train_id, len(members),
        )

        # Collect per-member solo verify results (root→tip; N verifies total).
        solo_results: list[SoloVerifyResult] = []
        for i, member in enumerate(members):
            predecessor_ref = (
                main_branch if i == 0
                else f'{branch_prefix}{members[i - 1].get("id", "")}'
            )
            solo = await self._reverify_one_member(member, predecessor_ref)
            solo_results.append(solo)
            logger.debug(
                'Task %s: member %s solo verify: passed=%s reason=%r',
                self.task_id, solo.member_id, solo.passed, solo.reason,
            )

        failers = [r for r in solo_results if not r.passed]

        if not failers:
            # ALL pass → genuine cross-member interaction; escalate train, land nothing.
            # Tear down every member's solo worktree+branch before returning
            # (reverify_member_solo left them alive for the land path; here we
            # land nothing, so we own the cleanup).
            for sr in solo_results:
                if sr.solo_wt is not None:
                    try:
                        await self.git_ops.cleanup_merge_worktree(sr.solo_wt)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            'Task %s: all-pass teardown: cleanup_merge_worktree '
                            'failed for member %s', self.task_id, sr.member_id,
                            exc_info=True,
                        )
                if sr.solo_branch is not None:
                    try:
                        await self.git_ops.delete_solo_branch(sr.solo_branch)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            'Task %s: all-pass teardown: delete_solo_branch '
                            'failed for member %s', self.task_id, sr.member_id,
                            exc_info=True,
                        )
            if self.event_store is not None:
                self.event_store.emit(
                    EventType.train_derailed,
                    task_id=self.task_id,
                    phase='merge',
                    data={
                        'train_id': train_id,
                        'member_task_ids': member_ids,
                        'verdict': 'interaction',
                        'members': member_ids,
                        'derail_reason': (
                            f'All {len(members)} members pass solo — '
                            'failure is a cross-member interaction'
                        ),
                    },
                )
            return await self._mark_blocked(
                f'Train {train_id!r} union verify failed but all '
                f'{len(members)} members pass solo — cross-member interaction',
                escalate_to_human=True,
            )

        # Some fail → land each passer, block each failer.
        passers = [r for r in solo_results if r.passed]
        offender_ids = [r.member_id for r in failers]
        passer_ids = [r.member_id for r in passers]

        # Land each passer: advance main with the solo-verified sha.
        # Re-read get_main_sha() immediately before each advance_main so that
        # sequential intra-loop landings CAS against the current tip (not a
        # stale pre-loop snapshot).  Pass branch=None so advance_main does NOT
        # try to resolve a non-existent task/_solo-<id> ref or fall back to
        # the merge_sha^2 re-merge path (a linear solo tip has no ^2).
        # Each passer's solo worktree and branch are cleaned up in a finally
        # block after the advance attempt.
        landed_ids: list[str] = []
        for r in passers:
            # Guard: passers always have a merge_sha (set by reverify_member_solo).
            if r.merge_sha is None:  # defensive — should not happen for a passer
                logger.warning(
                    'Task %s: train %r member %s passer has no merge_sha — skipping',
                    self.task_id, train_id, r.member_id,
                )
                continue
            try:
                expected_main = await self.git_ops.get_main_sha()
                outcome = await self.git_ops.advance_main(
                    r.merge_sha,
                    r.solo_wt,
                    branch=None,
                    expected_main=expected_main,
                    reverify_on_rebase=False,
                )
                adv = outcome.result
            except Exception as exc:  # noqa: BLE001
                # advance_main raised (unexpected); leave member parked for
                # re-dispatch and continue processing the remaining passers.
                logger.warning(
                    'Task %s: train %r member %s advance_main raised %r '
                    '— leaving parked for re-dispatch',
                    self.task_id, train_id, r.member_id, exc,
                )
                adv = 'exception'
                outcome = None
            finally:
                # Always clean up the solo worktree and branch after the
                # advance attempt — regardless of outcome, neither is needed
                # once advance_main has run.
                if r.solo_wt is not None:
                    try:
                        await self.git_ops.cleanup_merge_worktree(r.solo_wt)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            'Task %s: passer teardown: cleanup_merge_worktree '
                            'failed for member %s', self.task_id, r.member_id,
                            exc_info=True,
                        )
                if r.solo_branch is not None:
                    try:
                        await self.git_ops.delete_solo_branch(r.solo_branch)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            'Task %s: passer teardown: delete_solo_branch '
                            'failed for member %s', self.task_id, r.member_id,
                            exc_info=True,
                        )
            if adv == 'advanced':
                # outcome is guaranteed non-None here: 'exception' (the only
                # code path that sets outcome=None) never equals 'advanced'.
                landed_sha: str = (
                    (outcome.advanced_sha if outcome else None) or r.merge_sha
                )
                await self.scheduler.mark_done(
                    r.member_id,
                    kind='merged',
                    sha=landed_sha,
                    note=f'train {train_id} attribution: member passed solo',
                )
                # task 2280 (PRD WA-3): consume the tip's write-ahead LandedRow on
                # this attribution-passer done-write. Idempotent for non-tip passers
                # (they hold no row), fail-safe when unbound; runs only after a
                # SUCCESSFUL mark_done. Mirrors the 2681 single-branch precedent
                # (workflow.py:1912-1917) and the harness sites (steps 2/4).
                MergeProvenance.consume(r.member_id)
                landed_ids.append(r.member_id)
                if self.event_store is not None:
                    self.event_store.emit(
                        EventType.train_merged,
                        task_id=r.member_id,
                        phase='merge',
                        data={
                            'train_id': train_id,
                            'verdict': 'attributed',
                            'merge_sha': landed_sha,
                        },
                    )
                logger.info(
                    'Task %s: train %r member %s passed solo — advanced main to %s',
                    self.task_id, train_id, r.member_id, landed_sha,
                )
            else:
                # advance_main failure or exception: leave member parked for
                # re-dispatch — do NOT flip to done; log a warning.
                logger.warning(
                    'Task %s: train %r member %s advance_main returned %r '
                    '— leaving parked for re-dispatch',
                    self.task_id, train_id, r.member_id, adv,
                )

        # Build shared train_state once before the loop — it is identical for
        # every failer; only failing_member is overridden per iteration.
        # Hoisting avoids N redundant scheduler round-trips (get_statuses).
        esc_train_state_base: object = None
        if self.escalation_queue:
            esc_train_state_base = await self._build_train_state()

        # Block each failer: set status blocked and submit an L1.
        for r in failers:
            await self.scheduler.set_task_status(r.member_id, 'blocked')
            if self.escalation_queue:
                from escalation.models import Escalation
                # Copy the base train_state and override failing_member to the
                # actual offender (each failer gets its own attribution pointer).
                train_state = None
                if esc_train_state_base is not None:
                    train_state = dict(esc_train_state_base)  # type: ignore[arg-type]
                    train_state['failing_member'] = r.member_id  # type: ignore[index]
                esc = Escalation(
                    id=self.escalation_queue.make_id(r.member_id),
                    task_id=r.member_id,
                    agent_role='orchestrator',
                    severity='blocking',
                    category='task_failure',
                    summary=(
                        f'Train {train_id!r} attribution: member {r.member_id!r} '
                        f'failed solo verify — reason: {r.reason[:120]}'
                    ),
                    detail=(
                        f'Train {train_id!r} union verify failed; member '
                        f'{r.member_id!r} was re-verified in isolation and '
                        f'failed (reason: {r.reason}). '
                        f'Passers landed: {passer_ids}. '
                        f'Offenders blocked: {offender_ids}.'
                    ),
                    suggested_action='manual_intervention',
                    worktree=None,
                    workflow_state=self.state.value,
                    level=1,
                    train_state=train_state,  # type: ignore[arg-type]
                )
                self.escalation_queue.submit(esc)
                logger.warning(
                    'Task %s: train %r member %s blocked as attribution offender — L1 %s',
                    self.task_id, train_id, r.member_id, esc.id,
                )

        # Emit attributed train_derailed telemetry.
        if self.event_store is not None:
            self.event_store.emit(
                EventType.train_derailed,
                task_id=self.task_id,
                phase='merge',
                data={
                    'train_id': train_id,
                    'member_task_ids': member_ids,
                    'verdict': 'attributed',
                    'offenders': offender_ids,
                    'passers': passer_ids,
                    'derail_reason': (
                        f'Solo attribution: offenders={offender_ids} '
                        f'passers={passer_ids}'
                    ),
                },
            )

        # Tip return value: DONE if the tip landed, BLOCKED otherwise.
        if self.task_id in landed_ids:
            return WorkflowOutcome.DONE
        # Tip is blocked or advance failed — surface tip as blocked.
        # When the tip is itself an attribution offender, the failer loop above
        # already submitted a dedicated L1 for it (via escalation_queue.submit).
        # Calling _mark_blocked(escalate_to_human=True) here would create a
        # second L1 and a second blocked status transition for the same task.
        # Pass skip_escalation=True to suppress the redundant escalation while
        # still recording the BLOCKED state and updating internal fields.
        tip_is_offender = self.task_id in offender_ids
        return await self._mark_blocked(
            f'Train {train_id!r} union verify failed — tip {self.task_id!r} '
            f'{"is an offender" if tip_is_offender else "advance failed"}',
            escalate_to_human=not tip_is_offender,
            skip_escalation=tip_is_offender,
        )

    async def _escalate_train_halt(
        self, result, train_id: str,
    ) -> WorkflowOutcome:
        """Handle a train WIP-halt outcome: build per-status L1, own the halt, block tip.

        Called from ``_maybe_enqueue_group_merge`` when the orphan-halt probe fires::

            merge_worker is not None
            and merge_worker.is_wip_halted
            and merge_worker.halt_owner_esc_id is None

        Covers all five ``_map_advance_failure`` halt-inducing statuses (train
        coverage is automatic via the shared ``_build_wip_halt_escalation_text``):
          - ``wip_halted``             → category='wip_conflict'  (WIP overlaps diff)
          - ``done_wip_recovery``      → category='wip_conflict'  (merge landed; stash pop conflict)
          - ``wip_recovery_no_advance``→ category='wip_conflict'  (CAS failure; no advance)
          - ``unmerged_state``         → category='unmerged_state' (pre-existing UU/AA/DD)
          - ``stash_failed``           → category='stash_failed'  (project_root park failed; dirty tracked tree)

        Unlike the single-task ``_handle_wip_conflict`` etc., this helper does NOT
        await escalation resolution — the train tip stays BLOCKED and re-dispatches
        once the halt is cleared.  This avoids reintroducing the cancellation-orphan
        surface that task 1448 hardened for inline-waiting coroutines.

        When ``escalation_queue is None`` (config-absent deployment), logs a warning
        and falls back to plain BLOCKED with no owner registered.

        Auto-recovery parity (task 1599):
          Resolving the returned L1 triggers harness._on_escalation_resolved →
          unhalt_wip() because ``_submit_halt_owning_escalation`` registered this
          workflow as halt owner.  harness._rehydrate_merge_halt re-owns the L1
          across restarts (category in {wip_conflict, unmerged_state, stash_failed}).
        """
        status = result.status
        category, summary, detail = self._build_wip_halt_escalation_text(
            status, result, train_id=train_id,
        )

        reason = (
            f'Train halt: merge for task {self.task_id} (train {train_id!r}) '
            f'halted the merge queue ({status})'
        )
        logger.warning(
            'Task %s: train halt (%s) — %s',
            self.task_id, status,
            'creating level-1 escalation' if self.escalation_queue
            else 'escalation_queue=None, cannot register halt owner',
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            train_state = await self._build_train_state()

            # Defensive re-check: the consumer's orphan-halt probe validated
            # halt_owner_esc_id is None before calling us, but _build_train_state()
            # contains an await and another coroutine could (theoretically) set the
            # owner during that window.  In the current serial merge-worker design
            # this window is unreachable, but re-checking prevents a hard crash
            # from the 'owner already set' assertion inside set_halt_owner if the
            # worker ever becomes concurrent.  Mirror the escalation_queue=None
            # fallback: log a warning and fall through to plain BLOCKED.
            if (
                self.merge_worker is not None
                and self.merge_worker.halt_owner_esc_id is not None
            ):
                logger.warning(
                    'Task %s: halt owner set concurrently during _build_train_state '
                    '(owner: %r) — skipping duplicate set_halt_owner; plain BLOCKED',
                    self.task_id, self.merge_worker.halt_owner_esc_id,
                )
            else:
                esc = Escalation(
                    id=self.escalation_queue.make_id(self.task_id),
                    task_id=self.task_id,
                    agent_role='orchestrator',
                    severity='blocking',
                    category=category,
                    summary=summary,
                    detail=detail,
                    suggested_action='manual_intervention',
                    level=1,
                    worktree=str(self.worktree) if self.worktree else None,
                    workflow_state=self.state.value,
                    train_state=train_state,
                )
                self._submit_halt_owning_escalation(esc)
                logger.info(
                    'Task %s: train halt L1 %r submitted and halt ownership registered',
                    self.task_id, esc.id,
                )
        else:
            self._warn_orphan_halt_no_queue(result.status, train_id=train_id)

        return await self._mark_blocked(reason, detail=detail, skip_escalation=True)

    async def _handle_wip_conflict(
        self, result, branch_name: str,
    ) -> WorkflowOutcome:
        """Handle a wip_halted merge outcome: create level-1 escalation and wait.

        The merge did NOT land — WIP in project_root overlaps the merge diff.
        After the human resolves (commits/stashes WIP), the task retries the merge.
        """
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result, branch_name=branch_name,
        )
        logger.warning(f'Task {self.task_id}: WIP overlap — creating level-1 escalation')

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
            logger.info(f'Task {self.task_id}: WIP conflict resolved — retrying merge')
        else:
            self._warn_orphan_halt_no_queue(result.status)

        return WorkflowOutcome.REQUEUED

    async def _handle_wip_recovery(self, result) -> WorkflowOutcome:
        """Handle a done_wip_recovery merge outcome: merge landed but WIP conflicted.

        The merge IS on main, but the user's stashed WIP conflicted during pop.
        WIP has been preserved on a recovery branch. Create a level-1 escalation
        to inform the human, then return DONE (the task's merge succeeded).
        """
        # Capture the on-main SHA so the success-path set_task_status('done', ...)
        # in run() can build a valid done_provenance. Without this, _merge_sha
        # stays None and fused-memory rejects the transition with "kind required".
        if result.merge_sha is not None:
            self._merge_sha = result.merge_sha
        recovery_branch = result.recovery_branch or '(unknown)'
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result,
        )
        logger.warning(
            f'Task {self.task_id}: merge landed but stash pop conflicted — '
            f'WIP on {recovery_branch}'
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
            logger.info(f'Task {self.task_id}: WIP recovery escalation resolved')
        else:
            self._warn_orphan_halt_no_queue(result.status, recovery_branch=recovery_branch)

        return WorkflowOutcome.DONE

    async def _handle_wip_recovery_no_advance(self, result) -> WorkflowOutcome:
        """Handle a wip_recovery_no_advance merge outcome.

        The merge did NOT land on main (CAS failure path). A stash pop conflict
        occurred, and WIP has been preserved on a recovery branch. Create a
        level-1 escalation to inform a human, then return BLOCKED — the task
        cannot proceed until the tree is manually inspected.

        Unlike ``_handle_wip_recovery`` (which returns DONE because the merge
        landed), this returns BLOCKED because main was NOT advanced.
        """
        recovery_branch = result.recovery_branch or '(unknown)'
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result,
        )
        logger.warning(
            f'Task {self.task_id}: stash pop conflicted on CAS-failure path — '
            f'merge did not advance. WIP preserved on {recovery_branch}'
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
            logger.info(f'Task {self.task_id}: wip_recovery_no_advance escalation resolved')
        else:
            self._warn_orphan_halt_no_queue(result.status, recovery_branch=recovery_branch)

        return WorkflowOutcome.BLOCKED

    async def _handle_unmerged_state(
        self, result, branch_name: str,
    ) -> WorkflowOutcome:
        """Handle an unmerged_state merge outcome.

        ``project_root`` had pre-existing UU/AA/DD markers BEFORE this merge
        attempted to advance main. The merge did NOT land, and the tree is
        already in an inconsistent state. Halt stays in effect until a human
        inspects, cleans up project_root (``git mergetool`` / manual
        resolution / ``git reset``), and resolves the escalation.
        """
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result, branch_name=branch_name,
        )
        logger.warning(
            f'Task {self.task_id}: unmerged_state in project_root — '
            f'creating level-1 escalation'
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
            logger.info(
                f'Task {self.task_id}: unmerged_state escalation resolved'
            )
        else:
            self._warn_orphan_halt_no_queue(result.status)

        return WorkflowOutcome.BLOCKED

    async def _handle_stash_failed(
        self, result, branch_name: str,
    ) -> WorkflowOutcome:
        """Handle a stash_failed merge outcome.

        ``advance_main`` could not park project_root's dirty tracked WIP before
        advancing — a SHARED main-checkout-hygiene fault that recurs identically
        for every task landing (task 2758). The merge did NOT land and the queue
        is halted; the halt stays in effect until a human inspects project_root,
        commits or cleans the persistently-dirty tracked file(s), and resolves
        the escalation. Because the halt serializes the fleet, exactly ONE
        halt-owning level-1 escalation is filed (by the halt owner) instead of
        N per-task blocked finalizations. Mirrors ``_handle_unmerged_state``.
        """
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result, branch_name=branch_name,
        )
        dirty = result.dirty_files or []
        logger.warning(
            f'Task {self.task_id}: stash_failed — could not park project_root '
            f'WIP; dirty tracked file(s): {", ".join(dirty) or "(unknown)"} '
            f'— creating level-1 escalation'
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
            logger.info(
                f'Task {self.task_id}: stash_failed escalation resolved'
            )
        else:
            self._warn_orphan_halt_no_queue(result.status)

        return WorkflowOutcome.BLOCKED

    def _write_merge_failure_review(self, category: str, detail: str) -> None:
        """Write a review-format JSON describing a merge failure to .task/reviews/.

        Uses the same schema as reviewer agents so humans and retry agents
        can consume it uniformly.
        """
        if not self.artifacts:
            return
        review = {
            'reviewer': 'merge',
            'verdict': 'ISSUES_FOUND',
            'issues': [
                {
                    'severity': 'blocking',
                    'location': 'main',
                    'category': category,
                    'description': detail,
                },
            ],
            'summary': detail[:200],
        }
        self.artifacts.write_review('merge', review)

    def _build_agent_env(self, role: AgentRole) -> dict[str, str] | None:
        """Build the env_overrides dict for this agent invocation.

        Implementer and debugger inherit env_overrides and REIFY_DEBUG_PORT.
        Architect, implementer, and debugger also get:
        - CARGO_MAKEFLAGS (jobserver task-pool env) so their cargo build/test/metadata
          calls participate in the FIFO jobserver instead of running uncoordinated.
        - DF_AGENT_CPU_NICE (CPU nice de-prioritization env) so cli_invoke prepends
          ``nice -n N`` to the Claude CLI spawn, causing agents to yield CPU to
          reify's negatively-niced merge/task verifies.
        - DF_AGENT_CPU_GOVERN (cgroup placement env) so cli_invoke._cpu_govern_prefix
          prepends cpu-governed-exec.sh to the Claude CLI argv, placing the agent
          and its inherited cargo/rustc subtree into a cpu.weight-weighted cgroup
          scope (DF-1).
        - scripts/agent-bin prepended to PATH so the agent's ad-hoc ``cargo …``
          (Bash tool) hits the PSI shim instead of the system cargo (DF-2).
          PATH propagates to the agent and its cargo children (not popped).

        Every role additionally gets config.role_env_overrides.get(role.name, {})
        merged in LAST (task 2460): a per-role OPT-IN endpoint-env map (e.g.
        ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN) so an operator can point a
        SPECIFIC role at an alternate Claude-compatible endpoint with no
        harness change. role_env_overrides defaults to {}, so a role absent
        from it (e.g. merger, judge, reviewer by default) receives no endpoint
        env from this layer — forwarding is opt-in per role, not a hardcoded
        allow-list. Because this layer is merged last, an explicit per-role
        entry wins over the global config.env_overrides on a key collision.
        A collision with an INFRA-provided key (jobserver/cpu_priority/
        cpu_governance/REIFY_DEBUG_PORT) also logs a WARNING naming the role
        and key, since silently clobbering one of those can break the cargo
        jobserver, cpu cgroup placement, or a PATH-based tool shim for that
        role.
        """
        merged: dict[str, str] = {}
        if role.name in ('architect', 'implementer', 'debugger'):
            if role.name in ('implementer', 'debugger'):
                merged.update(self.config.env_overrides or {})
                if self._reify_debug_port is not None:
                    merged['REIFY_DEBUG_PORT'] = str(self._reify_debug_port)
            merged.update(self.config.jobserver.agent_env())
            merged.update(self.config.cpu_priority.agent_env())
            merged.update(
                self.config.cpu_governance.agent_env(self.worktree, os.environ.get('PATH', ''))
            )
        role_overrides = self.config.role_env_overrides.get(role.name, {})
        clobbered = sorted(set(role_overrides) & set(merged))
        if clobbered:
            logger.warning(
                'role_env_overrides[%r] overrides infra-provided env key(s) %s '
                '(jobserver/cpu-governance/REIFY_DEBUG_PORT/env_overrides) -- '
                'the per-role value wins (merged last); this may break the '
                'cargo jobserver, cpu cgroup placement, or PATH-based tool '
                'shims for this role',
                role.name,
                clobbered,
            )
        merged.update(role_overrides)
        return merged or None

    def _build_spawn_env(self, role: AgentRole) -> dict[str, str]:
        """Build the CLAUDE_SPAWN_* spawn-identity env for this agent invocation.

        Consumed by the SessionStart hook (session_hooks.run_session_start ->
        parse_spawn_identity) so every orchestrator-launched agent's registry
        record carries an accurate role/project/task and a non-null parent
        (task 2512). Built for EVERY role — unlike _build_agent_env, which
        returns None for merger/judge/reviewer.

        CLAUDE_SPAWN_PARENT_ID defaults to the workflow-root self.session_id
        (used for the architect itself, and as the fallback for any role
        dispatched before an architect ran in this instance). For a
        non-architect role once an architect HAS run, the parent is instead
        the architect's reconstructed registry slug — built via the exact
        same session_registry.build_session_slug the architect's own
        SessionStart hook uses, so it matches that real record precisely.
        """
        parent_id = self.session_id
        if role.name != 'architect' and self._architect_spawn_session_id is not None:
            # The architect's Claude session_id (str) deliberately fills the
            # launcher_pid slot as the uniqueness token here -- mirrors
            # session_hooks.hook_session_slug, which does the same for the
            # identical reason: build_session_slug only ever str()s this
            # argument, so the int annotation is not a real runtime
            # constraint.
            parent_id = build_session_slug(
                'architect',
                self.config.fused_memory.project_id,
                str(self.task_id),
                self._architect_spawn_session_id,  # type: ignore[arg-type]
            )
        return {
            'CLAUDE_SPAWN_ROLE': role.name,
            'CLAUDE_SPAWN_PROJECT': self.config.fused_memory.project_id,
            'CLAUDE_SPAWN_TASK_ID': str(self.task_id),
            'CLAUDE_SPAWN_PARENT_ID': parent_id,
        }

    async def _ceiling_spend_by_model(self) -> dict[str, float]:
        """Trailing-24h USD spend per model carrying a configured
        ``routing.per_model_daily_ceiling_usd`` entry (PRD adaptive-model-
        routing, task ε invariant 6).

        Queried ONLY for ceiling'd models, and only when a cost_store is
        wired — empty at stock config (no ceilings configured), so zero
        cost_store reads fire and byte-equivalence (invariant 3) holds.
        Best-effort per model: a query failure for one model is logged and
        that model is simply omitted from the result — ``resolve_route``
        then defaults its spend to 0.0 (fail-open; a dispatch is never
        blocked by a cost-query hiccup, mirroring harness.py's
        ``_enforce_cost_ceilings``).
        """
        ceilings = self.config.routing.per_model_daily_ceiling_usd
        if not ceilings or not self.cost_store:
            return {}
        start_iso, end_iso = _trailing_24h_window()
        spend: dict[str, float] = {}
        for model_name in ceilings:
            try:
                spend[model_name] = await self.cost_store.model_cost_in_window(
                    model_name, start_iso, end_iso,
                )
            except Exception:
                logger.warning(
                    'Task %s: failed to fetch trailing-24h spend for model %s',
                    self.task_id, model_name, exc_info=True,
                )
        return spend

    def _scope_capacity_snapshot(self) -> dict[str, bool] | None:
        """Resolve-time advisory per-scoped-model headroom snapshot for the
        routing resolver (task δ, invariants S7/S8).

        Returns ``None`` when no ``usage_gate`` is wired (the common case —
        out-of-band dispatch, tests), otherwise the gate's
        ``scope_capacity_snapshot()`` (task γ): True per scoped model iff >=1
        account has headroom, ``{}`` when the scoped-cap kill switch is off.
        Best-effort/fail-open — a read hiccup is logged and degrades to
        ``None`` (mirrors ``_ceiling_spend_by_model``), so a snapshot failure
        NEVER blocks dispatch (S7). The snapshot is advisory only: the gate's
        own invoke-time scope predicate stays authoritative, so a stale
        snapshot degrades to a scope-wait/failover, never a wrong-and-stuck
        decision (S8).
        """
        if self.usage_gate is None:
            return None
        try:
            return self.usage_gate.scope_capacity_snapshot()
        except Exception:
            logger.warning(
                'Task %s: failed to read scope-capacity snapshot from usage_gate',
                self.task_id, exc_info=True,
            )
            return None

    def _resolve_role_system_prompt(self, role: AgentRole, model: str) -> str:
        """Resolve *role*'s system prompt, applying a live artifact override when opted in.

        Roles with ``prompt_spec is None`` (every role but the reviewer(s),
        today) return ``role.system_prompt`` verbatim — the loader is never
        consulted. Roles that opt in (``prompt_spec`` + ``prompt_harness_version``
        set — see roles.py's ``build_reviewer_prompt_spec``) resolve through
        :class:`~shared.prompt_artifact.PromptArtifactStore`, keyed on *model*
        (the router-resolved ``executor_model`` — P-4 of PRD
        tier1-prompt-optimization) so a pinned artifact is per-model.

        Lazily builds ``self._prompt_store`` from :func:`default_artifacts_root`
        when no store was injected. :meth:`PromptArtifactStore.resolve` never
        raises — an absent or unverifiable pin falls back to
        ``role.prompt_spec.in_code_constant`` — so this call is always
        fail-safe (mirrors ``TaskCurator._resolve_curator_prompt``).
        """
        if role.prompt_spec is None or role.prompt_harness_version is None:
            return role.system_prompt
        if self._prompt_store is None:
            root = default_artifacts_root()
            logger.info(
                'Task %s [%s]: no prompt_store injected; lazily resolved artifacts '
                'root to %s (set DARK_FACTORY_PROMPT_ARTIFACTS to override)',
                self.task_id, role.name, root,
            )
            self._prompt_store = PromptArtifactStore(root)
        return self._prompt_store.resolve(
            role.prompt_spec,
            executor_model=model,
            harness_version=role.prompt_harness_version,
        ).text

    async def _invoke(
        self,
        role: AgentRole,
        prompt: str,
        cwd: Path,
        output_schema: dict | None = None,
    ) -> AgentResult:
        """Invoke an agent with role-specific configuration."""
        timeouts_cfg = self.config.timeouts
        backends_cfg = self.config.backends

        # Full-name lookup is deliberate (routing alpha, task 2531) — do NOT
        # re-derive a split('_')[0] prefix here. Every routing submodel now
        # carries a field per full role name (including underscore-named
        # roles like simple_task), so role.name itself is the addressable
        # key; a split-derivation would silently miss those fields again
        # (as it did for simple_task -> 'simple', which matched nothing).
        # This aligns _invoke with the two out-of-band dispatch sites that
        # already resolved config by full role name: module_tagger
        # (harness.py's self.config.models.module_tagger) and deep_reviewer
        # (review_checkpoint.py's getattr(self.config.models, 'deep_reviewer', ...)).
        role_key = role.name

        # Route resolution (PRD adaptive-model-routing, task ε): the single
        # layered authority for (model, effort, budget_usd, max_turns),
        # resolved BEFORE role.system_prompt is used below so the decision
        # is in scope at the prompt-build seam (invariant 9). Retires
        # _select_model_for_role — its Rust heuristic now ships as
        # defaults.yaml's rust-large-plan-implementer policy rule, applied
        # identically to both implementer and debugger via the rule's own
        # `role` match list (no more per-role rule_id string formatting).
        routing_state = RoutingState.from_metadata(self.task.get('metadata'))
        plan_shape = (
            PlanShape(len(self.plan.get('steps', [])), tuple(self.modules))
            if self.plan else None
        )
        task_metadata = self.task.get('metadata') or {}
        role_defaults = RoleDefaults(
            model=role.default_model,
            effort='high',
            budget_usd=role.default_budget,
            max_turns=role.default_max_turns,
        )
        route_inputs = RouteInputs(
            role_name=role.name,
            task_id=self.task_id,
            task_metadata=task_metadata,
            plan_shape=plan_shape,
            routing_tier=routing_state.routing_tier,
            dispatch_count=int(task_metadata.get('dispatch_count', 0)),
            role_defaults=role_defaults,
            spend_by_model=await self._ceiling_spend_by_model(),
            scope_capacity=self._scope_capacity_snapshot(),
        )
        decision = resolve_route(route_inputs, self.config)
        model = decision.model
        budget = decision.budget_usd
        max_turns_val = decision.max_turns
        effort_val = decision.effort
        source_layer = decision.source_layer
        rule_id = decision.rule_id

        timeout_val = getattr(timeouts_cfg, role_key, self.config.invocation_timeout)
        backend_val = getattr(backends_cfg, role_key, 'claude')

        # timeout_val/backend_val are resolver-external (route resolution
        # owns model/effort/budget_usd/max_turns only — see this task's plan
        # design_decisions) — the reviewer* collapse stays local to _invoke
        # for just these two fields.
        if role.name.startswith('reviewer'):
            timeout_val = timeouts_cfg.reviewer
            backend_val = backends_cfg.reviewer

        # Determine the sandbox write-set based on role (role.sandboxed is a
        # property of the role object — see roles.py's AgentRole/W9-η).
        # Whole-worktree granularity (PRD os-sandbox D1): sandbox_modules=[] is
        # the empty-list sandbox-on gate (no per-module writable dirs), and the
        # full contract write set (worktree root + carve-outs) rides on
        # sandbox_extras via compute_write_set — the INV-5 single source. Both
        # channels are set together so a broken half-state (extras without the
        # gate) never arises. compute_write_set is allowed to raise on a
        # malformed worktree (fail-loud; the fail-closed backend-unavailable
        # posture is a separate concern, not folded in here).
        sandbox_modules = None
        sandbox_extras = None
        if self.config.sandbox.enabled and role.sandboxed:
            write_set = compute_write_set(cwd)
            # Pre-create the load-bearing claude_fleet carve-out OUTSIDE the
            # sandbox at this single INV-5/D11 write-set consumption point, so it
            # exists before sandbox_extras (below) is built (task 2996) — see
            # ensure_claude_fleet_dir for the full backend rationale.
            self._ensure_sandbox_dirs(write_set)
            sandbox_modules = []
            sandbox_extras = [str(p) for p in write_set.writable_paths()]
            # Fail-CLOSED (task 2908, PRD D4 / INV-4): refuse to dispatch this
            # sandboxed role when no OS backend resolves rather than silently
            # running unconfined. backend=none is the explicit operator escape
            # hatch (runs UNSANDBOXED with a WARN, no refusal); a
            # required-yet-unavailable backend raises SandboxUnavailable, which
            # propagates out of _invoke to refuse the invocation. On the
            # non-refusing path _guard_sandbox returns the resolved backend,
            # which we record here via the per-invocation sandbox_applied event
            # (task 2909 β2, PRD §Goal 3 / INV-2): real backend AND the
            # none-escape hatch both emit sandbox_applied, while a refusal
            # instead emits sandbox_unavailable inside the guard — so every
            # sandboxed invocation emits exactly one of the two.
            resolved_backend = self._guard_sandbox(role.name)
            self._emit_sandbox_applied(role.name, resolved_backend, write_set)

        # Warn once per workflow instance when an escalation-capable role is
        # dispatched without an escalation queue wired up.
        self._maybe_warn_missing_escalation(role.name)

        # Build MCP config — fused-memory always, escalation when available.
        # Judge gets MCP so its jcodemunch tools (in allowed_tools) actually
        # work; it does not use escalation tools but mcp_config_json handles
        # escalation_url=None fine. Gated on role.mcp_families (W9-η) rather
        # than a role.name membership tuple.
        mcp_config = None
        if 'orchestrator' in role.mcp_families:
            escalation_url = None
            if self.escalation_queue:
                esc = self.config.escalation
                escalation_url = f'http://{esc.host}:{esc.port}/mcp'
            if self.mcp is not None:
                mcp_config = self.mcp.mcp_config_json(escalation_url=escalation_url)

        # Plan-tools stdio MCP server — architect builds plans, implementer/
        # debugger/simple_task mark steps done.  Per-invocation isolation: each
        # agent gets its own server bound to the worktree path.  Fast-start flags
        # (--no-sync, --frozen) reuse the already-synced orchestrator venv to
        # avoid cold-resolve stalls under load (reify esc-4415-240/esc-4437-123).
        if 'plan_tools' in role.mcp_families and cwd:
            mcp_config = _inject_plan_tools_mcp(mcp_config, cwd)

        # Verdict-tools stdio MCP server — reviewer/judge/merger submit a
        # constrained verdict artifact (PRD task β, task 2482). Gated on
        # role.mcp_families exactly like plan-tools above; the per-invocation
        # --verdict-role is role.name. _inject_verdict_tools_mcp skeletons a
        # config when None, so the reviewer (no 'orchestrator' family) still
        # acquires it.
        if 'verdict_tools' in role.mcp_families and cwd:
            mcp_config = _inject_verdict_tools_mcp(mcp_config, cwd, role)

        # Session-resume lifecycle: if the harness recovered a sidecar that
        # matches the role we're about to invoke, resume the prior session
        # by passing resume_session_id to invoke_with_cap_retry.  Otherwise,
        # allocate a fresh UUID up-front via --session-id so a future restart
        # can find and resume this session.  The sidecar is written before the
        # subprocess starts and cleared in the finally below — its presence
        # ⇔ "agent was in flight when the orchestrator exited".
        #
        # Prompt-substitution ownership: cli_invoke owns the resume-continuation
        # prompt swap (CRASH_RECOVERY_RESUME_PROMPT = 'continue').  Workflow
        # always passes the real task prompt so that cli_invoke's
        # original_prompt capture is correct for any fresh-fallback invocation.
        resume_session_id: str | None = None
        if (
            self._pending_resume_session_id
            and self._pending_resume_role == role.name
        ):
            session_id_val = self._pending_resume_session_id
            resume_session_id = session_id_val
            # Adopted resume: bump the cumulative count off the recovered base,
            # then reset it (consumed-on-first-use, mirroring the session-id/
            # role resets below).
            resume_count_to_write = self._pending_resume_count + 1
            self._pending_resume_session_id = None
            self._pending_resume_role = None
            self._pending_resume_count = 0
            logger.info(
                'Task %s [%s]: resuming prior session %s via --resume',
                self.task_id, role.name, session_id_val,
            )
        else:
            session_id_val = str(uuid.uuid4())
            resume_count_to_write = 0

        # Stash for _capture_zero_output_evidence — result.session_id is '' on
        # hard SIGKILL, so we capture the effective id here before the invocation.
        self._last_invoke_session_id = session_id_val

        # Stash the architect's own --session-id UUID so _build_spawn_env can
        # reconstruct its SessionStart-hook registry slug as the
        # CLAUDE_SPAWN_PARENT_ID for post-architect roles nesting under it
        # (task 2512).  In-process only -- see the field's __init__ comment
        # for why this deliberately does not survive an orchestrator restart.
        if role.name == 'architect':
            self._architect_spawn_session_id = session_id_val

        if self.artifacts is not None:
            self.artifacts.write_agent_session(
                session_id_val, role.name, datetime.now(UTC).isoformat(),
                task_id=str(self.task_id), resume_count=resume_count_to_write,
            )

        started_at = datetime.now(UTC).isoformat()
        # I1: the sidecar exists iff an invocation is in flight — cancellation
        # is not completion.  Track whether this invocation was cancelled
        # in-flight so the finally preserves (does not clear) its sidecar.
        session_preserved = False
        try:
            result = await invoke_with_cap_retry(
                usage_gate=self.usage_gate,
                label=f'Task {self.task_id} [{role.name}]',
                config_dir=self._config_dir,
                invoke_fn=invoke_agent,
                prompt=prompt,
                system_prompt=self._resolve_role_system_prompt(role, model),
                cwd=cwd,
                model=model,
                max_turns=max_turns_val,
                max_budget_usd=budget,
                allowed_tools=role.allowed_tools or None,
                disallowed_tools=role.disallowed_tools or None,
                mcp_config=mcp_config,
                output_schema=output_schema,
                sandbox_modules=sandbox_modules,
                sandbox_extras=sandbox_extras,
                effort=effort_val,
                backend=backend_val,
                # Rides the same **invoke_kwargs forwarding path as backend=
                # above (task 2457) straight to invoke_agent(prices=...) ->
                # the codex/gemini/pi cost estimators (claude ignores it —
                # reports native cost). See OrchestratorConfig.prices.
                prices=self.config.prices,
                timeout_seconds=timeout_val,
                startup_grace_secs=timeouts_cfg.startup_grace_secs,
                # Working-regime progress extension (task 2360 fix #1): once the
                # transcript proves liveness (≥1 turn), the watchdog no longer
                # kills at the flat per-role timeout_val ceiling — it extends to
                # max(working_idle_secs, timeout_val), bounded above by the
                # absolute cap.  _invoke is the SHARED chokepoint for every role,
                # so this engages uniformly; safe by construction since the
                # extended idle bound is always >= the old per-role ceiling.
                working_idle_secs=timeouts_cfg.working_idle_secs,
                absolute_cap_secs=self.config.invocation_timeout,
                session_id=session_id_val,
                resume_session_id=resume_session_id,
                # Judge is safe by OMISSION from config.role_env_overrides (the
                # per-role OPT-IN endpoint-env map, task 2460) — _build_agent_env
                # no longer gates roles by a hardcoded allow-list, so the judge
                # simply must not be named there.  Opting it in would propagate
                # ANTHROPIC_BASE_URL and route it through vLLM, where
                # max_model_len causes ServerDisconnectedError after 2 tool-use
                # rounds (3cd380a079) — do not add 'judge' to role_env_overrides
                # unless that endpoint is confirmed to handle its tool-use
                # pattern.  Cap hits on Claude API are handled by UsageGate
                # account failover (wired in runner.py for eval mode).
                env_overrides=self._build_agent_env(role),
                # Spawn-identity env for the SessionStart hook (task 2512) —
                # independent of env_overrides (which is None for
                # merger/judge/reviewer); spawn_env is built for every role.
                spawn_env=self._build_spawn_env(role),
            )
        except asyncio.CancelledError:
            # Cooperative cancellation of the agent invocation itself (a clean
            # SIGTERM shutdown surfaces as CancelledError from THIS await) is an
            # in-flight SUSPENSION, not a completion (I1).  Preserve the sidecar
            # so crash-recovery can --resume this session after restart, and emit
            # a structured fact at the decision point (INV-2 — fields in extra,
            # not scraped from prose) before re-raising to propagate cooperative
            # cancellation.  This except catches ONLY the main invoke await; a
            # cancellation arriving later during the finally's transcript-archival
            # await (agent already returned a result) is handled by that block's
            # own inner except and correctly falls through to clear.  A
            # non-CancelledError failure is an abnormal terminal end, not a
            # resumable suspension, so it too falls through to clear — unchanged
            # from prior behavior (the finally cleared on every exit before).
            session_preserved = True
            if self.artifacts is not None:
                logger.info(
                    'Task %s [%s]: agent_session_preserved — keeping sidecar '
                    'for crash-recovery resume of session %s',
                    self.task_id, role.name, session_id_val,
                    extra={
                        'event': 'agent_session_preserved',
                        'task_id': str(self.task_id),
                        'session_id': session_id_val,
                        'role': role.name,
                    },
                )
            raise
        finally:
            # Clear ONLY on a non-preserved exit (completion or abnormal error);
            # a cancelled-in-flight invocation keeps its sidecar for resume.
            if self.artifacts is not None and not session_preserved:
                self.artifacts.clear_agent_session()
            # Producer hook (task 2742, agent-transcript-archival-prd α): gzip
            # this just-finished session's transcripts to a durable archive root
            # OUTSIDE the worktree (project_root / config.root), so they survive
            # worktree teardown. _last_invoke_session_id is set before the try,
            # so it is present even when this finally runs during exception
            # propagation.
            ta = self.config.transcript_archive
            if ta.enabled and self._config_dir is not None and self._last_invoke_session_id:
                # Offload to a worker thread: archive_task_transcripts does
                # blocking, CPU-bound work (glob + stream-gzip each transcript).
                # This finally runs on the shared event loop for every role of
                # every concurrent task, so a multi-MB transcript archived inline
                # would stall all other in-flight tasks; to_thread keeps the loop
                # free.
                try:
                    await asyncio.to_thread(
                        archive_task_transcripts,
                        self._config_dir.path,
                        self.task_id,
                        self._last_invoke_session_id,
                        archive_root=self.config.project_root / ta.root,
                    )
                except asyncio.CancelledError:
                    # Cancellation (loop teardown / hard-kill) surfaces here from
                    # the await, NOT an archival error. Cooperative cancellation
                    # must propagate, so we re-raise — meaning a KILLED
                    # invocation's transcript is deliberately not archived by this
                    # producer hook. That is an accepted, documented gap: shielding
                    # the await to salvage it (asyncio.shield) risks a dangling
                    # background task during loop close, and the abandoned-in-flight
                    # tail is the explicit job of β/task 2729's idempotent
                    # teardown backstop (agent-transcript-archival-prd §3), so it
                    # is not lost overall.
                    raise
                except Exception:
                    # Defense-in-depth for a finally that awaits cross-module work.
                    # archive_task_transcripts is total by contract (per-file
                    # OSErrors are swallowed + counted), but its top-level glob /
                    # Path / archive_root construction is not individually guarded.
                    # Should any unexpected non-cancellation error ever escape it,
                    # swallow it here so the producer hook can never REPLACE the
                    # in-flight exception this finally is unwinding (the classic
                    # finally-masks-original antipattern) — independent of any
                    # future change to the helper's guarantees. Loud, not silent:
                    # the failure is logged as a structured fact.
                    logger.warning(
                        'Transcript archival hook failed for task %s (session %s)',
                        self.task_id,
                        self._last_invoke_session_id,
                        exc_info=True,
                        extra={
                            'task_id': self.task_id,
                            'session_id': self._last_invoke_session_id,
                        },
                    )
        completed_at = datetime.now(UTC).isoformat()

        # Record the last successfully-completed role (updated only on success,
        # mirrors the cost-accumulation path below — failed/raised invocations
        # do not advance either field).
        self._last_completed_role = role.name

        # Track metrics
        self.metrics.total_cost_usd += result.cost_usd
        self.metrics.total_duration_ms += result.duration_ms
        self.metrics.agent_invocations += 1
        self.metrics.total_turns += result.turns
        self.metrics.total_input_tokens += result.input_tokens or 0
        self.metrics.total_output_tokens += result.output_tokens or 0
        self.metrics.total_cache_read_tokens += result.cache_read_tokens or 0
        self.metrics.total_cache_create_tokens += result.cache_create_tokens or 0

        logger.info(
            'Task %s [%s]: success=%s cost=$%.2f turns=%d timeout=%.0fs',
            self.task_id, role.name, result.success, result.cost_usd,
            result.turns, timeout_val,
            extra={
                'task_id': self.task_id, 'role': role.name, 'model': model,
                'cost_usd': result.cost_usd, 'turns': result.turns,
                'input_tokens': result.input_tokens,
                'output_tokens': result.output_tokens,
            },
        )

        if self.event_store:
            self.event_store.emit(
                EventType.invocation_end,
                task_id=self.task_id,
                phase=self.state.value,
                role=role.name,
                cost_usd=result.cost_usd,
                duration_ms=result.duration_ms,
                data={
                    'turns': result.turns,
                    'success': result.success,
                    'subtype': result.subtype,
                    'model': model,
                    'account_name': result.account_name,
                    'input_tokens': result.input_tokens,
                    'output_tokens': result.output_tokens,
                    'cache_read_tokens': result.cache_read_tokens,
                    'cache_create_tokens': result.cache_create_tokens,
                    # Truthful ceiling-kill reporting (task 2360 fix #3): lets
                    # telemetry consumers distinguish a productive wall-clock
                    # kill (timed_out=True, transcript_turns>0) from a genuine
                    # zero-output wedge (transcript_turns==0/None).
                    'transcript_turns': result.transcript_turns,
                    'timed_out': result.timed_out,
                },
            )

        await self._record_routing_decision(
            role,
            model=model,
            effort=effort_val,
            budget_usd=budget,
            max_turns=max_turns_val,
            source_layer=source_layer,
            rule_id=rule_id,
            rejected=list(decision.rejected),
        )

        if self.cost_store:
            try:
                await self.cost_store.save_invocation(
                    run_id=self.event_store.run_id if self.event_store else '',
                    task_id=self.task_id,
                    project_id=self.config.fused_memory.project_id,
                    account_name=result.account_name,
                    model=model,
                    role=role.name,
                    cost_usd=result.cost_usd,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cache_read_tokens=result.cache_read_tokens,
                    cache_create_tokens=result.cache_create_tokens,
                    duration_ms=result.duration_ms,
                    capped=False,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            except Exception:
                logger.warning('Failed to save invocation cost', exc_info=True)

        return result

    async def _record_routing_decision(
        self,
        role: AgentRole,
        *,
        model: str,
        effort: str,
        budget_usd: float,
        max_turns: int,
        source_layer: str,
        rule_id: str | None,
        rejected: list[str] | None = None,
    ) -> None:
        """Persist the resolved routing decision for this invocation (PRD γ).

        Best-effort (never raises) but awaited synchronously inside
        ``_invoke``'s critical path — it is not fire-and-forget in the
        ``asyncio.create_task`` sense, so a slow/hanging scheduler write here
        does add latency to every invocation. What it guarantees is failure
        isolation, mirroring the "routing telemetry must never block or
        crash a caller" philosophy of ``RoutingState.from_metadata``:
        building the decision record itself is wrapped in try/except (a
        MagicMock-configured role in many existing ``_invoke`` unit tests
        fails ``RoutingDecisionMirror``'s strict str/float/int field
        validation — that must never break the invocation under test), the
        ``routing_decision`` event is guarded on ``self.event_store``, and
        the ``metadata.routing`` mirror via
        ``scheduler.update_task(metadata_mode='merge')`` is guarded on
        ``self.scheduler`` and wrapped in its own try/except.  The in-memory
        ``self.task['metadata']['routing']`` update always runs — regardless
        of the scheduler write's outcome — so successive ``_invoke`` calls
        within the same dispatch accumulate history without needing a
        round-trip read.
        """
        try:
            state = RoutingState.from_metadata(self.task.get('metadata'))
            digest = _routing_inputs_digest(
                role, self.task_id, self.plan, self.modules, state.routing_tier,
            )
            decision = RoutingDecisionMirror(
                role=role.name,
                model=model,
                effort=effort,
                budget_usd=budget_usd,
                max_turns=max_turns,
                source_layer=source_layer,
                rule_id=rule_id,
                rejected=rejected or [],
                routing_tier=state.routing_tier,
                decided_at=datetime.now(UTC).isoformat(),
            )
        except Exception:
            logger.warning(
                'Task %s: failed to build routing decision record (role=%s)',
                self.task_id, role.name, exc_info=True,
            )
            return

        if self.event_store:
            # Derived from `decision` (rather than hand-duplicating each
            # field) so the event payload and the metadata.routing mirror
            # below can never drift as fields are added in task ε — both
            # are two serializations of the one `decision` record, plus the
            # event-only `inputs_digest`.
            self.event_store.emit(
                EventType.routing_decision,
                task_id=self.task_id,
                role=role.name,
                data={**decision.model_dump(), 'inputs_digest': digest},
            )

        # Per-invocation write cost (task 2533 review): this adds one
        # merge-mode metadata upsert per invocation, alongside the
        # cost_store.save_invocation write _invoke already makes
        # unconditionally — not a new class of write, and the payload is
        # small and bounded (history capped at _ROUTING_HISTORY_MAX=5).
        # Coalescing to "only write on decision change" was considered and
        # rejected: the PRD goal is to persist WHICH decision was made per
        # invocation, including repeated identical decisions, and
        # test_successive_invocations_accumulate_history locks in that every
        # invocation's decision lands in history.
        new_state = state.with_decision(decision)
        if self.scheduler:
            try:
                await self.scheduler.update_task(
                    self.task_id,
                    {'routing': new_state.model_dump()},
                    # SchedulerFacade's Protocol (scheduler.py) predates
                    # metadata_mode and only declares `append`; the concrete
                    # Scheduler.update_task (and the harness reblock_guard
                    # call site) already support it. Out of task 2533's scope
                    # to widen the Protocol itself — see escalate_info.
                    metadata_mode='merge',  # type: ignore[reportCallIssue]
                )
            except Exception:
                logger.warning(
                    'Task %s: failed to mirror routing decision onto metadata',
                    self.task_id, exc_info=True,
                )
        self.task.setdefault('metadata', {})['routing'] = new_state.model_dump()

    def _check_escalations(self):
        """Check for pending escalations for this task."""
        if not self.escalation_queue:
            return []
        return self.escalation_queue.get_by_task(self.task_id, status='pending')

    async def _wait_for_resolution(self) -> str:
        """Wait for all level-0 pending escalations to be resolved.

        Born-at-L2 (critical/urgent severity) and any level≥2 pending
        escalations are stop-the-line events — they cannot be resolved by
        the auto-watcher/steward and require immediate human intervention.
        Rather than waiting (which would either spin or block forever with
        no resolver), they immediately raise ``_StewardReescalated`` so
        that ``run()`` calls ``_mark_blocked`` and halts.

        Raises ``_StewardReescalated`` if:
        - A pending escalation with ``severity in BORN_AT_L2_SEVERITIES``
          or ``level >= 2`` exists — immediate terminal, stop the line; OR
        - The steward re-escalated to level-1 (consumed by auto-watcher).

        When no escalation queue is available (e.g. eval mode), returns
        an empty string immediately — the caller treats this as "no
        resolution" and the workflow proceeds to ESCALATED/BLOCKED
        via its normal path.

        **Bounded (task 3170, fix C + review fixes D1/D2/D3).** The level-0
        wait is bounded by an IDLE window

            timeouts.steward + steward_completion_timeout

        refreshed for as long as the steward is observably working (its
        progress counter advancing — see :meth:`_steward_progress_counter`), so
        it trips ONLY on a genuinely SILENT producer.  Both terms are what they
        are for a reason, and the refresh is what makes the "future producer
        bug" framing below true rather than aspirational:

        - ``timeouts.steward`` is the ENFORCED per-SUBPROCESS ceiling — the
          longest a healthy steward can be silent WITHIN one agent subprocess,
          since past it ``invoke_agent`` itself SIGTERM/SIGKILLs the process
          group (cli_invoke.py:2184-2206).  Any window shorter than it is
          guaranteed to fire on healthy stewards.  It is emphatically NOT a
          bound on one ``_invoke_with_session`` call: that delegates to
          ``invoke_with_cap_retry``, which may run up to 16 subprocesses with
          cap cooldowns between them behind a single return, so the progress
          signal MUST tick per subprocess attempt rather than per completed
          invocation — which is exactly what ``metrics.subprocess_attempts``
          does (review fix D4).  Without that tick, an all-accounts-capped
          steward reads as silent for hours and this backstop kills a working
          producer.
        - ``steward_completion_timeout`` is kept in its DOCUMENTED role
          (config.py:211-222 — the post-invocation drain grace) as the slack
          for the steward to publish/dismiss after its invocation returns or
          is killed, rather than misused as the whole bound.
        - The refresh covers the retry tail: a healthy steward can occupy
          several full invocation ceilings on ONE escalation, because the
          timeout-kill path re-handles the still-pending record
          (steward.py:399-412).  Refreshing bounds each legitimate retry by
          its own window without hard-coding the ``steward_max_attempts +
          steward_max_timeouts_per_escalation`` multiplier, so the bound
          self-maintains if either knob changes.

        No new config knob and no new validator were added: the existing
        ``timeouts.steward >= steward_completion_timeout`` invariant
        (config.py:4071-4081) already forces this window above ``2x
        steward_completion_timeout`` for every config that constructs at all,
        and an operator who raises ``timeouts.steward`` widens the wait bound
        in lockstep for free.

        The wait relies on the same PRODUCER invariant its sibling waiter
        :meth:`_await_steward_completion` does:

            A steward give-up ALWAYS dismisses its own L0 before publishing
            an outcome.  No pending L0 survives a steward give-up.

        This path is woken ONLY by that dismissal (``EscalationQueue.resolve``
        → ``_resolve_callback`` → ``harness._on_escalation_resolved`` →
        ``_escalation_events[task_id].set()``); it never reads the steward
        outcome channel.  The idle window therefore exists to make the wait
        non-strandable if any FUTURE producer bug breaks that invariant — as
        task 2248's wip-gated give-up branches did, parking the workflow
        forever while the steward re-handled the capped escalation at loop
        speed.

        On expiry it FIRST stops the steward and clears the reference, and
        only then logs loudly, emits an ``escalation_resolved`` event with
        ``outcome='steward_wait_timeout'``, DISMISSES the orphan L0(s) — so
        the next ESCALATED entry cannot re-strand on the same records — and
        falls through to the unchanged tail below.  Stopping first is a safety
        requirement, not tidiness: the steward runs its agent in the SAME
        worktree (``cwd = self.worktree``, steward.py:542, 590) that the
        resumed implementer edits and commits in, so dismissing or resuming
        beside a live steward agent means two agents committing in one git
        worktree.  The tail then yields exactly two dispositions:
        ``_StewardReescalated`` (blocking, with the already-open L1) when one
        is open, or the collected resolutions (resuming) when none is.  It
        deliberately does NOT call :meth:`_mark_blocked`, which would file a
        fresh L0 and could add a SECOND full grace window for what is a pure
        wait-timeout.

        On BOTH exits it then calls :meth:`_drain_steward_outcomes`: whatever
        the steward published while this waiter held the wait is never consumed
        here, and would otherwise be replayed to a later ``_mark_blocked`` in
        this same workflow as if freshly published.
        """
        if self.escalation_queue is None:
            logger.warning(
                'Task %s: _wait_for_resolution called without escalation_queue '
                '(eval mode?) — returning immediately',
                self.task_id,
            )
            return ''

        if self._escalation_event is None:
            self._escalation_event = asyncio.Event()

        # Born-at-L2 (critical/urgent) and any level≥2 escalations are
        # stop-the-line: no auto-watcher can resolve them, so waiting for the
        # L0 loop to clear makes no sense and risks a busy resume-spin (gate
        # fires → _wait_for_resolution returns '' → resume → gate fires again).
        # Treat identically to _StewardReescalated so run() calls _mark_blocked
        # and halts for human intervention rather than looping.
        pending_high_sev = [
            e for e in self.escalation_queue.get_by_task(self.task_id, status='pending')
            if e.severity in BORN_AT_L2_SEVERITIES or e.level >= 2
        ]
        if pending_high_sev:
            raise _StewardReescalated(pending_high_sev)

        # Wait for level-0 pending escalations to clear, BOUNDED by a derived
        # window (task 3170, fix C + review fix D1).  See the method docstring
        # for the disposition.  Both terms are load-bearing, neither is
        # arbitrary:
        #
        #   timeouts.steward             the ENFORCED per-SUBPROCESS ceiling —
        #                                the longest a HEALTHY steward can be
        #                                silent within ONE agent subprocess,
        #                                since past it invoke_agent itself
        #                                SIGTERM/SIGKILLs the process group
        #                                (cli_invoke.py:2184-2206).  Any window
        #                                shorter than this is guaranteed to fire
        #                                on healthy stewards.  Cap-retry
        #                                cooldowns stack MORE subprocesses
        #                                behind one _invoke_with_session call;
        #                                the per-attempt progress tick (review
        #                                fix D4) is what keeps them visible.
        #   steward_completion_timeout   kept in its DOCUMENTED role
        #                                (config.py:211-222, the post-invocation
        #                                drain grace) as the slack for the
        #                                steward to publish/dismiss after its
        #                                invocation returns or is killed.
        #
        # Deliberately NO new config knob and NO new validator: the existing
        # `timeouts.steward >= steward_completion_timeout` invariant
        # (config.py:4071-4081) already forces this window to >= 2x
        # steward_completion_timeout for every config that constructs at all,
        # and an operator who raises timeouts.steward widens the wait bound in
        # lockstep for free.
        #
        # The window is an IDLE deadline, not an overall one (review fix D2):
        # it is refreshed whenever the steward is observably still working, so
        # only a GENUINELY SILENT producer trips the give-up path.  A healthy
        # steward can legitimately occupy several full invocation ceilings on
        # ONE escalation — the timeout-kill path re-handles the still-pending
        # record (steward.py:399-412) — and refreshing on the counter bounds
        # each legitimate retry by its own window without hard-coding the
        # steward_max_attempts + steward_max_timeouts_per_escalation multiplier.
        # The same argument covers the cap-retry tail (review fix D4): the
        # counter ticks per SUBPROCESS attempt, so an all-accounts-capped
        # steward waiting out cooldowns keeps refreshing instead of being
        # mistaken for a dead producer and killed mid-work.
        window = self.config.timeouts.steward + self.config.steward_completion_timeout
        deadline = asyncio.get_event_loop().time() + window
        last_progress = self._steward_progress_counter()
        while True:
            pending_l0 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            if not pending_l0:
                break
            self._escalation_event.clear()
            remaining = deadline - asyncio.get_event_loop().time()
            try:
                if remaining <= 0:
                    raise TimeoutError  # noqa: TRY301 — uniform expiry handling
                await asyncio.wait_for(self._escalation_event.wait(), remaining)
            except TimeoutError:  # asyncio.TimeoutError is an alias since 3.11
                # Before giving up: is the steward observably still working?
                # A counter that ADVANCED means one more invocation legitimately
                # completed inside this window, so extend rather than fire.  A
                # None counter (no steward, or a fake without metrics) means no
                # signal is available and degrades to a plain fixed deadline; a
                # counter that went backwards or non-monotonic is treated as no
                # progress, never as an extension.
                progress = self._steward_progress_counter()
                if (
                    progress is not None
                    and last_progress is not None
                    and progress > last_progress
                ):
                    logger.info(
                        'Task %s: steward still working (invocations %d → %d) '
                        '— extending the ESCALATED wait by another %.0fs idle '
                        'window rather than giving up',
                        self.task_id, last_progress, progress, window,
                    )
                    last_progress = progress
                    deadline = asyncio.get_event_loop().time() + window
                    continue
                # Give-up decided.  STOP THE STEWARD FIRST — before the event
                # emit, before the orphan dismissal, before the break (review
                # fix D3).  Ordering is load-bearing, not stylistic:
                # TaskSteward invokes its agent with cwd = self.worktree
                # (steward.py:542, 590), the SAME worktree the resumed
                # implementer edits and commits in, so stopping first closes
                # the two-agents-one-worktree window entirely rather than
                # narrowing it.  stop() cancels the loop task (steward.py:
                # 209-217) and on the stock `steward: "claude"` backend the
                # resulting CancelledError propagates into
                # cli_invoke.py:_run_subprocess, whose handler (:2240-2252)
                # terminates the agent's whole process group — so the in-flight
                # agent genuinely stops writing rather than merely being
                # detached from.  Clearing the reference (the existing
                # stop-then-clear idiom, cf. :5319-5320 / :5363-5364) makes a
                # later _mark_blocked build a FRESH steward through
                # _ensure_steward_started instead of awaiting a cancelled loop
                # that can never publish.  Suppressed because this is cleanup on
                # an already-degraded path: a failing stop() must not convert a
                # wait-timeout into a workflow crash.
                if self._steward is not None:
                    with contextlib.suppress(Exception):
                        await self._steward.stop()
                    self._steward = None
                orphan_ids = [e.id for e in pending_l0]
                logger.warning(
                    'Task %s: steward did not resolve %d level-0 escalation(s) '
                    'within the ESCALATED wait window (%.0fs = timeouts.steward '
                    '%.0fs + steward_completion_timeout %.0fs) — dismissing the '
                    'orphan(s) and unblocking the ESCALATED wait: %s',
                    self.task_id, len(orphan_ids), window,
                    self.config.timeouts.steward,
                    self.config.steward_completion_timeout,
                    ', '.join(orphan_ids),
                )
                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_resolved,
                        task_id=self.task_id, phase=self.state.value,
                        data={
                            'outcome': 'steward_wait_timeout',
                            'escalation_ids': orphan_ids,
                        },
                    )
                # Uphold the same "no pending L0 survives a give-up"
                # invariant fix A establishes on the producer side.  Leaving
                # them pending would re-strand the NEXT ESCALATED entry on
                # the same records for another full window.
                for esc in pending_l0:
                    self.escalation_queue.resolve(
                        esc.id,
                        'Auto-dismissed: steward did not resolve within the '
                        'ESCALATED wait window (timeouts.steward + '
                        'steward_completion_timeout) — unblocking the '
                        'ESCALATED wait',
                        dismiss=True,
                        resolved_by='auto-dismissed',
                    )
                break

        # Whatever the steward published while THIS waiter held the wait is
        # never consumed by it (see _drain_steward_outcomes) — drain it here,
        # covering both the normal-clear and the timeout exit above, so a
        # later _mark_blocked in this same workflow cannot pop a stale
        # outcome out of _await_steward_completion.
        self._drain_steward_outcomes()

        # Check for level-1 re-escalation (steward gave up)
        if self.escalation_queue.has_open_l1(self.task_id):
            pending_l1 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=1,
            )
            raise _StewardReescalated(pending_l1)

        # Collect resolutions (filter ensures resolution is non-None str for join)
        resolutions = [
            e.resolution
            for e in self.escalation_queue.get_by_task(self.task_id)
            if e.status == 'resolved' and e.resolution is not None
        ]
        return '\n'.join(resolutions)

    async def _get_head_commit(self) -> str:
        """Return the HEAD commit SHA for the current worktree."""
        proc = await asyncio.create_subprocess_exec(
            'git', 'rev-parse', 'HEAD',
            cwd=str(self.worktree),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()

    async def _iteration_commit_provenance(self, pre_head: str) -> dict:
        """Compare pre/post HEAD and report truthful commit provenance for an
        iteration-ledger entry (task 2759).

        Every iteration-ledger writer (implementer / debugger / amender)
        captures ``pre_head = await self._get_head_commit()`` immediately
        before invoking its agent, then merges this helper's result into the
        entry dict after the agent returns. This is the single source of truth
        for the pre/post comparison so the three writers cannot drift.

        - HEAD advanced (``post`` is truthy and ``!= pre_head``) ⇒ the agent
          committed: return ``{'commit': post, 'committed': True}``. No
          ``dirty`` read is performed on this happy path (avoids an extra
          ``git status`` on the common case).
        - HEAD unchanged ⇒ the agent's session ended before committing (died
          mid-background-wait, amendment left uncommitted, …). Recording the
          stale ``pre_head`` here is the reify-5164 false-provenance bug;
          instead return ``{'commit': None, 'committed': False,
          'dirty': <porcelain-nonempty>}`` so recovery guards can treat the
          round as no durable work.

        The ``dirty`` read reuses :meth:`GitOps.has_uncommitted_work` and is
        fail-soft (observability-only): guarded on ``git_ops``/``worktree``
        being present and wrapped in try/except, defaulting ``dirty=False`` on
        any error so it never sinks the iteration loop or the ledger append.
        """
        post = await self._get_head_commit()
        if post and post != pre_head:
            return {'commit': post, 'committed': True}
        dirty = False
        if self.git_ops is not None and self.worktree is not None:
            try:
                dirty = bool(
                    await self.git_ops.has_uncommitted_work(self.worktree),
                )
            except Exception:
                logger.warning(
                    'Task %s: dirty-tree probe failed while stamping iteration '
                    'provenance; recording dirty=False (observability-only)',
                    self.task_id, exc_info=True,
                )
                dirty = False
        return {'commit': None, 'committed': False, 'dirty': dirty}

    async def _check_branch_on_main(self) -> tuple[str, str] | None:
        """Probe whether the worktree HEAD is reachable from main.

        Returns ``(wt_head, main_sha)`` when ``git merge-base --is-ancestor
        wt_head main_sha`` succeeds (i.e. the branch has been merged to main
        or the HEAD is exactly main).  Returns ``None`` in three cases:

        1. ``self.worktree`` or ``self.git_ops`` is None — partially-wired
           workflow; callers that reach this state should treat the branch as
           not-on-main.
        2. HEAD is not an ancestor of main — branch has unmerged commits.
        3. Any of the above when combined with the caller's own guard logic.

        Does NOT catch subprocess or git exceptions — callers wrap as needed.
        ``_recover_if_already_merged`` wraps the call in ``try/except`` and
        logs ``'merge-check failed'`` before returning None; ``_recover_before_execute``
        lets exceptions propagate.  This exception-handling difference is the
        one thing that still varies per call site — otherwise all three
        already-merged guards (pre-PLAN, pre-EXECUTE, merge-phase) now share
        the same journal-first check (``MergeProvenance.lookup``) and
        finalise identically through ``_finalise_recovery_done`` (PRD
        workflow-state-machine α).

        See also: ``_recover_if_already_merged`` (pre-PLAN guard),
        ``_recover_before_execute`` (pre-EXECUTE guard), and
        ``_recover_before_merge`` (merge-phase guard).
        """
        if self.worktree is None or self.git_ops is None:
            return None
        wt_head = await self._get_head_commit()
        main_sha = await self.git_ops.get_main_sha()
        if await self.git_ops.is_ancestor(wt_head, main_sha):
            return (wt_head, main_sha)
        return None

    def _has_prior_implementation(self, wt_head: str | None = None) -> _PriorImplStatus:
        """Check whether a prior run did any implementation in this worktree.

        When *wt_head* is provided (the post-execution branch HEAD), the primary
        signal is a SHA comparison: the branch has advanced past its starting
        point iff ``wt_head.strip() != base_commit``.  This is invariant to
        iteration-log format changes and avoids the false-done regression where
        a stale iteration entry triggers the guard on a branch with no commits.

        When *wt_head* is not provided, falls back to scanning
        .task/iterations.jsonl for implementer/debugger entries.
        Planning-only runs don't write these, so absence means stale branch
        point rather than a legitimately merged prior run.  This fallback is
        also used by the ghost-loop guard and the merge-phase guard, where a
        post-rebase HEAD may coincide with base_commit even on a genuinely-
        implemented branch.

        When *wt_head* IS provided but base_commit is absent (metadata.json
        not yet stamped), returns ``has_work=False`` — this is the
        fail-closed safety net for SHA-primary callers.  Refusing to fall
        back to the iteration-log scan prevents false-DONE from inherited
        .task/iterations.jsonl contamination.  See
        test_returns_none_when_wt_head_provided_but_metadata_missing for the
        regression case.

        Returns a :class:`_PriorImplStatus` NamedTuple with ``has_work``,
        ``entries`` (full iteration-log list — callers can use ``len(entries)``
        in warning breadcrumbs without a second ``read_iteration_log()`` call),
        and ``base_commit`` (may be ``None`` if metadata.json is absent).

        Correctness invariants: the iteration-log fallback relies on
        .task/iterations.jsonl entries faithfully reflecting prior work on the
        *same branch*.  Two scenarios matter:

        *Intended* — ghost-loop re-run on the same branch: create_worktree
        may rebase a reused worktree onto main, so wt_head == base_commit even
        though the branch was genuinely implemented.  The fallback correctly
        returns True here because the earlier implementer run wrote its entries
        before the rebase.  This is the scenario exploited by the pre-EXECUTE
        guard and the pre-MERGE guard; it is safe for those callers.

        *Dangerous* — orphaned log: if iterations.jsonl were somehow copied
        from a different task's branch or inherited from main contamination,
        the fallback would return True for an empty branch → false-done.
        This is why callers that hold a reliable wt_head must pass it
        explicitly.  ``_recover_if_already_merged()`` passes wt_head to
        use the SHA-primary path, preventing false-DONE on inherited
        .task/iterations.jsonl contamination — see the comment there for
        the full trade-off analysis.

        Per-entry classification (task 2372, Layer A): both the SHA-primary
        and bare-fallback iteration-log scans use
        :func:`_iteration_entry_is_work` rather than a bare ``agent in
        ('implementer', 'debugger')`` check. That bare check false-positived
        on a *zero-work* implementer entry (``steps_completed: []``) left
        over from a prior dispatch onto the same worktree — the exact
        signature behind the task 2125/2315/2340 false-DONE recurrences,
        where a fresh/re-dispatched worktree (wt_head == base_commit) still
        resolved has_work=True from a stale poison entry. The classifier
        excludes that narrow signature while still counting debugger entries
        (hard-code ``steps_completed: []``), amendment-implementer entries
        (omit ``steps_completed`` entirely), and judge ``early_exit`` entries
        with ``substantive_work=True`` — see that function's docstring.
        """
        if self.artifacts is None:
            return _PriorImplStatus(has_work=False, entries=[], base_commit=None)
        base_commit = self.artifacts.read_base_commit()
        entries, _ = self.artifacts.read_iteration_log()
        if wt_head is not None and base_commit is not None:
            sha_diverges = wt_head.strip() != base_commit
            has_iter_log_work = any(_iteration_entry_is_work(e) for e in entries)
            # Defense in depth: SHA divergence alone is racy under
            # fused-memory's tasks.json auto-commit to main (the
            # pre-positioning rev-parse in create_worktree could lag
            # the actual worktree fork point).  Iteration-log evidence
            # alone is racy under inherited orphan logs.  Require BOTH
            # signals before declaring prior implementation work.  See
            # the audit notes in
            # ~/.claude/plans/do-2-3-misty-marshmallow.md and the
            # orphan-log scenario in this method's docstring.
            return _PriorImplStatus(
                has_work=sha_diverges and has_iter_log_work,
                entries=entries,
                base_commit=base_commit,
            )
        if wt_head is not None:
            # Fail-closed: caller signaled SHA-primary semantics by passing
            # wt_head, but base_commit is absent (metadata.json not stamped).
            # Refuse to fall back to the iteration-log scan — on a worktree
            # with inherited .task/iterations.jsonl contamination this would
            # still produce has_work=True and false-DONE.  See
            # test_returns_none_when_wt_head_provided_but_metadata_missing
            # for the regression case.
            return _PriorImplStatus(has_work=False, entries=entries, base_commit=None)
        # Fallback (no wt_head): iteration-log scan for pre-EXECUTE / merge-phase guards
        return _PriorImplStatus(
            has_work=any(_iteration_entry_is_work(e) for e in entries),
            entries=entries,
            base_commit=base_commit,
        )

    async def _branch_work_landed_on_main(
        self, branch_head: str, main_sha: str, *, wt_head: str | None,
    ) -> bool:
        """True iff branch_head is an ancestor of main AND there is prior
        implementation work — i.e. the branch's work genuinely landed, not
        an empty/stale branch whose base trivially satisfies the ancestor
        check.

        Extracted from :meth:`_recover_before_merge` (task 2504), which
        already implemented this ``is_ancestor AND has_work`` predicate
        correctly, to share it with the ESCALATED-resume guard in
        :meth:`_drive`, whose prior raw ``is_ancestor``-only check false-
        positived on an empty branch (``wt_head == base_commit`` is
        trivially an ancestor of main) and skipped implementer resume
        entirely.

        ``wt_head`` selects the :meth:`_has_prior_implementation` has-work
        mode and is a parameter (not computed internally) because the two
        call sites need different modes: the resume site passes its
        reliable post-execution HEAD for the SHA-primary signal (an empty
        branch correctly resolves ``has_work=False``); ``_recover_before_merge``
        passes ``wt_head=None`` for the iteration-log fallback, because the
        merge phase runs after a rebase where a post-rebase HEAD may
        coincide with base_commit even on a genuinely-implemented branch.
        See :meth:`_has_prior_implementation`'s docstring for the full
        trade-off analysis of both modes.
        """
        if not await self.git_ops.is_ancestor(branch_head, main_sha):
            return False
        return self._has_prior_implementation(wt_head=wt_head).has_work

    async def _finalise_recovery_done(
        self, *, basis: str, sha: str, kind: str, note: str,
        files: list[str] | None = None,
    ) -> WorkflowOutcome:
        """Shared DONE-finalisation for all already-merged guards (PRD α, MP-2).

        The sole writer of :attr:`_merge_recovery_basis`.  Every already-merged
        guard's only route to a recovery-DONE goes through this method, so a
        recovery-DONE always carries an explicit provenance ``basis`` — either
        ``'journal'`` (a caller's :meth:`MergeProvenance.lookup` call found a
        :class:`~orchestrator.landed_outbox.LandedRow`) or ``'fallback'`` (the
        legacy :meth:`_has_prior_implementation` heuristic). Extracted and
        generalized from the pre-PLAN guard's original DONE tail.

        ``files`` (task 2372, Layer C): optional pre-computed evidence
        (typically a real ``git diff`` of the branch's own content) threaded
        straight through to :meth:`_reconcile_metadata_files_for_done` as
        ``override_files`` — see that method for why this beats the
        ``_merge_sha``-derived default for a recovery guard that has already
        computed a more precise diff itself.  ``None`` (the default) leaves
        the existing ``_merge_sha``-derived behaviour unchanged.

        Returns ``WorkflowOutcome.DONE`` on success. If the persistence layer
        rejects the write, routes to ``_mark_blocked(escalate_to_human=True)``
        (returning its BLOCKED/ESCALATED outcome) instead of reporting a
        phantom DONE.

        Structural MP-2 guard: raises ``AssertionError`` if *basis* is not
        ``'journal'``/``'fallback'`` or *sha* is falsy — BEFORE any status
        mutation (no marker write, no phase transition, no scheduler call).
        This is the executable form of "no recovery-DONE without a
        provenance basis": a future guard that calls this chokepoint without
        a valid basis fails loudly instead of silently producing a
        phantom-done.
        """
        if basis not in ('journal', 'fallback') or not sha:
            raise AssertionError(
                f'_finalise_recovery_done requires a valid provenance basis '
                f"(basis='journal'|'fallback' and a truthy sha) — got "
                f'basis={basis!r}, sha={sha!r} (task {self.task_id})'
            )
        self._merge_recovery_basis = basis
        self._enter_phase(WorkflowState.DONE)
        await self._reconcile_metadata_files_for_done(override_files=files)
        try:
            await self.scheduler.mark_done(
                self.task_id, kind=kind, sha=sha, note=note,
            )
        except SetTaskStatusRejected as exc:
            logger.error(
                'Task %s: recovery-DONE mark_done rejected (basis=%s) — %s: %s',
                self.task_id, basis, exc.error_code, exc.raw,
            )
            return await self._mark_blocked(
                f'Recovery-DONE rejected: {exc.error_code} — {exc.raw}',
                escalate_to_human=True,
            )
        return WorkflowOutcome.DONE

    async def _recover_if_already_merged(self) -> WorkflowOutcome | None:
        """Check if the task's branch is already on main and transition to DONE.

        Called pre-PLAN to short-circuit ghost-loop re-runs: if a prior workflow
        run merged the branch but failed before writing DONE status, this guard
        detects the merged branch and immediately marks the task done.

        Journal-first (PRD workflow-state-machine α, MP-1): a
        :meth:`MergeProvenance.lookup` hit is authoritative and short-circuits
        before the git-layer probe or the legacy heuristic are ever consulted.
        A miss falls through to the existing git-layer/artifacts-layer checks.

        Returns WorkflowOutcome.DONE if the branch is already merged to main AND
        there is prior implementation work.  Returns None in all other cases
        (branch not merged, no prior work, missing worktree/git_ops, exceptions).
        """
        row: LandedRow | None = MergeProvenance.lookup(self.task_id)
        if row is not None:
            return await self._finalise_recovery_done(
                basis='journal', sha=row.advanced_sha, kind='merged',
                note='landed-outbox journal hit (pre-PLAN recovery)',
            )

        # Intentional double-check: _check_branch_on_main() has its own
        # None-guard and would return None silently, but this outer check lets
        # us emit the 'skipping merge-recovery' DEBUG log so the missing-wiring
        # condition is observable at the call-site level.
        if self.worktree is None or self.git_ops is None:
            logger.debug(
                'Task %s: skipping merge-recovery (no worktree or git_ops)',
                self.task_id,
            )
            return None

        # ── Git layer ─────────────────────────────────────
        # Delegates to _check_branch_on_main() which can fail for git/infra
        # reasons (e.g. corrupted index, network mount offline).
        # Returns (wt_head, main_sha) when the branch is on main, else None.
        try:
            _git_check = await self._check_branch_on_main()
        except Exception:
            logger.warning(
                'Task %s: merge-check failed, proceeding with normal workflow',
                self.task_id, exc_info=True,
            )
            return None

        if _git_check is None:
            return None

        wt_head, main_sha = _git_check

        # ── Artifacts layer ────────────────────────────────
        # Reads that can fail for filesystem/JSON reasons (e.g. corrupted
        # iterations.jsonl, missing metadata).  Wrapped in a SEPARATE
        # try/except from the git layer so operators can distinguish the
        # root cause from the log message.
        #
        # We pass wt_head to _has_prior_implementation() so the SHA-primary
        # path is taken: has_work = (wt_head != base_commit).  This prevents
        # false-DONE when a fresh worktree (wt_head == base_commit, no real
        # commits) has inherited an on-disk .task/iterations.jsonl written
        # directly to the worktree's .task/ directory (e.g. main-branch
        # contamination where the file exists on disk but is untracked by git).
        # Without wt_head the iteration-log fallback finds the implementer entry
        # and incorrectly returns DONE for an unimplemented task — catastrophic
        # silent failure.  See test_returns_none_for_inherited_iterations_log_on_fresh_worktree
        # for the regression case this guards.
        #
        # Additional safety net: if metadata.json was never stamped
        # (base_commit is None), _has_prior_implementation() now returns
        # has_work=False from its fail-closed branch rather than falling back
        # to the iteration-log scan.  This covers edge cases where
        # artifacts.init() was not called before this guard (e.g. eval-mode
        # paths or future refactors that re-order setup).  See
        # test_returns_none_when_wt_head_provided_but_metadata_missing.
        #
        # Trade-off: if create_worktree rebased a genuinely-implemented branch
        # onto a new main tip so that wt_head == new_base_commit, the SHA-primary
        # check returns has_work=False and this guard returns None (the workflow
        # proceeds to PLAN).  The pre-EXECUTE guard (_recover_before_execute)
        # still uses the iteration-log fallback and will catch the rebased
        # ghost-loop before EXECUTE, finalising through the shared
        # _finalise_recovery_done chokepoint (recovery-DONE).  Only one
        # architect invocation is wasted — bounded cost, far preferable to a
        # silent false-DONE that marks an unimplemented task complete with no
        # code written.  See _has_prior_implementation() for SHA-primary vs.
        # fallback semantics.
        try:
            status = self._has_prior_implementation(wt_head=wt_head)
            if not status.has_work:
                logger.warning(
                    'Task %s: branch HEAD %s is ancestor '
                    'of main %s but no implementation '
                    'entries (base=%s, entries=%d) — '
                    'proceeding with normal workflow',
                    self.task_id,
                    wt_head[:8],
                    main_sha[:8],
                    status.base_commit[:8] if status.base_commit else 'none',
                    len(status.entries),
                )
                return None
        except Exception:
            logger.warning(
                'Task %s: artifacts read failed during merge-check, '
                'proceeding with normal workflow',
                self.task_id, exc_info=True,
            )
            return None

        logger.info(
            'Task %s: branch already on main — completing instead of re-queueing',
            self.task_id,
        )
        return await self._finalise_recovery_done(
            basis='fallback', sha=main_sha, kind='found_on_main',
            note='branch already on main at workflow start (pre-PLAN recovery)',
        )

    async def _recover_before_execute(self) -> WorkflowOutcome | None:
        """Ghost-loop early exit before EXECUTE (Guard 2, PRD α, MP-1/MP-2).

        Runs after PLAN and before the execute/verify/review loop: if the
        worktree HEAD is already reachable from main, the task's code was
        merged in a prior run that never reached DONE status (e.g. post-merge
        memory write failed).  Short-circuiting here avoids the implementer
        making redundant commits that would defeat the merge-phase ancestor
        check.

        External (eval-mode) worktrees never ghost-recover — checked first,
        before the journal or any git probe, so a pre-created eval worktree
        always runs the execute loop regardless of journal/git state.

        Journal-first (MP-1): a :meth:`MergeProvenance.lookup` hit is
        authoritative and short-circuits before the git-layer probe or the
        branch-content-diff gate are ever consulted.  A miss falls back to
        :meth:`_check_branch_on_main` + a REAL branch-content diff (task
        2372, Layer C): ``git diff base_commit..wt_head`` (via
        :meth:`~orchestrator.git_ops.GitOps.get_merge_diff_files`) must be
        non-empty.  This REPLACES the previous iteration-log heuristic
        (:meth:`_has_prior_implementation` called WITHOUT ``wt_head``), which
        was spoofable by a stale ``iterations.jsonl`` entry surviving in
        ``.task-meta/<name>/`` across a re-dispatch: a fresh/re-dispatched
        worktree has ``wt_head == base_commit``, which is TRIVIALLY an
        ancestor of main, so ``_check_branch_on_main`` alone was never
        sufficient — only a non-empty content diff proves the branch
        actually produced committed, merged work.  The diff doubles as the
        ``metadata.files`` evidence stamped on the recovery-DONE (ACTION #2;
        see :meth:`_finalise_recovery_done`'s ``files`` parameter), so a
        found_on_main DONE from this guard can no longer carry
        ``metadata.files=[]`` — the exact shape that let task 2125's
        phantom-done slip the reconciliation gate undetected.

        Accepted trade-off: a genuinely-merged-then-rebased branch (where
        ``create_worktree`` rebased a reused worktree so ``wt_head``
        regressed to the new ``base_commit``) that ALSO misses the journal
        now re-executes instead of recovering, because
        ``base_commit..wt_head`` is empty even though the branch was once
        genuinely implemented.  This is deliberately preferred over the
        false-positive it replaces: genuine merges are journal-covered
        (:meth:`MergeProvenance.lookup` runs first, above), and a spurious
        re-execute is self-correcting (the merge-phase guard finalises it
        properly), whereas a phantom-DONE is not self-correcting and
        requires an out-of-band reconciliation sweep to detect — the
        asymmetry that makes this the right trade.

        Returns ``WorkflowOutcome.DONE`` (via the shared
        :meth:`_finalise_recovery_done` chokepoint, MP-2) when the branch is
        already merged AND ``base_commit..wt_head`` is a non-empty diff.
        Returns ``None`` in all other cases (external worktree, not on main,
        on main but a zero-diff branch — a fresh, re-dispatched, or
        otherwise stale/unadvanced branch point) so the caller proceeds with
        the normal execute/verify/review loop.
        """
        if self._worktree_external:
            return None

        row: LandedRow | None = MergeProvenance.lookup(self.task_id)
        if row is not None:
            return await self._finalise_recovery_done(
                basis='journal', sha=row.advanced_sha, kind='merged',
                note='landed-outbox journal hit (pre-EXECUTE recovery)',
            )

        _branch_check = await self._check_branch_on_main()
        if _branch_check is None:
            return None
        wt_head, main_sha = _branch_check

        base_commit = self.artifacts.read_base_commit() if self.artifacts else None
        branch_files: list[str] = []
        if base_commit:
            branch_files, _err = await self.git_ops.get_merge_diff_files(
                base_commit, wt_head,
            )
        if not branch_files:
            logger.info(
                f'Task {self.task_id}: worktree HEAD {wt_head[:8]} '
                f'is ancestor of main but base..wt_head diff is empty '
                f'— fresh/unadvanced branch point, proceeding normally'
            )
            return None

        logger.info(
            f'Task {self.task_id}: worktree HEAD {wt_head[:8]} '
            f'already on main with {len(branch_files)} changed file(s) '
            f'— skipping to DONE (prior merge survived)'
        )
        return await self._finalise_recovery_done(
            basis='fallback', sha=main_sha, kind='found_on_main',
            note='branch already on main at workflow start (pre-EXECUTE recovery)',
            files=branch_files,
        )

    async def _recover_before_merge(
        self, branch_head: str, main_sha: str,
    ) -> WorkflowOutcome | None:
        """Ghost-loop early exit inside the merge phase (Guard 3, PRD α, MP-1/MP-2).

        Runs at the top of :meth:`_run_merge_phase`, immediately after the
        caller computes ``branch_head``/``main_sha`` — prevents infinite
        merge retry when the code was already merged (e.g. by an external
        actor, or by a prior run of this same workflow).

        Journal-first (MP-1): a :meth:`MergeProvenance.lookup` hit is
        authoritative and short-circuits before ``is_ancestor`` or the legacy
        heuristic are ever consulted. A miss falls back to the same
        stale-branch-point guard as the pre-EXECUTE check: if *branch_head*
        is an ancestor of main but :meth:`_has_prior_implementation` (called
        WITHOUT ``wt_head`` — see that method's docstring) finds no
        implementation entries, this is a spurious merge signal (e.g. an
        empty branch whose base commit trivially satisfies the ancestor
        check) and the caller should proceed with the real merge rather than
        short-circuit.

        Returns ``WorkflowOutcome.DONE`` (via the shared
        :meth:`_finalise_recovery_done` chokepoint, MP-2) when the branch is
        already merged AND there is prior implementation work. Returns
        ``None`` in all other cases (not an ancestor, or a spurious merge
        signal) so the caller proceeds with the merge-retry loop.
        """
        row: LandedRow | None = MergeProvenance.lookup(self.task_id)
        if row is not None:
            return await self._finalise_recovery_done(
                basis='journal', sha=row.advanced_sha, kind='merged',
                note='landed-outbox journal hit (pre-MERGE recovery)',
            )

        # wt_head=None: the merge phase runs after a rebase, where a
        # post-rebase HEAD may coincide with base_commit even on a
        # genuinely-implemented branch, so this caller must keep the
        # iteration-log fallback rather than the SHA-primary signal (see
        # _has_prior_implementation's docstring and task 2504).
        if await self._branch_work_landed_on_main(branch_head, main_sha, wt_head=None):
            return await self._finalise_recovery_done(
                basis='fallback', sha=main_sha, kind='found_on_main',
                note='branch already on main at merge phase (pre-MERGE recovery)',
            )

        # Not landed. Re-check is_ancestor (cheap, redundant on this path
        # only) purely to scope the breadcrumb to the spurious-merge-signal
        # sub-case (ancestor but no work) — a normal divergent (non-ancestor)
        # merge must stay silent.
        if await self.git_ops.is_ancestor(branch_head, main_sha):
            logger.warning(
                f'Task {self.task_id}: branch appears merged at '
                f'merge phase but has no implementation entries '
                f'— proceeding with merge'
            )
        return None

    def _escalate_plan_overwrite(self) -> None:
        """Submit a blocking escalation when plan.json ownership doesn't match.

        Distinguishes two cases by reading the current _session_id:

        - **Empty/missing**: the architect failed before stamping provenance.
          This is usually a downstream effect of a planning-phase failure
          (e.g. 403/cap hit with no retry), not a duplicate workflow.
        - **Non-empty but different**: a genuine foreign session wrote plan.json.
        """
        summary = f'plan.json overwrite detected for task {self.task_id}'
        foreign_session = ''
        if self.artifacts is not None:
            try:
                plan_path = self.artifacts.root / 'plan.json'
                data = json.loads(plan_path.read_text())
                foreign_session = data.get('_session_id') or ''
            except Exception:
                pass

        if not foreign_session:
            detail = (
                f'plan.json is not stamped with the current session '
                f'(expected _session_id={self.session_id}). Most likely the '
                f'architect failed before stamping — a downstream effect of a '
                f'planning-phase failure, not a duplicate workflow.'
            )
        else:
            detail = (
                f'Expected _session_id={self.session_id} but plan.json contains '
                f'{foreign_session}. A duplicate workflow may have overwritten plan.json.'
            )
        logger.error(f'Task {self.task_id}: {summary}')

        if not self.escalation_queue:
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=summary,
            detail=detail,
            suggested_action='investigate_and_retry',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={'escalation_id': esc.id, 'category': esc.category,
                      'severity': esc.severity, 'summary': summary[:200]},
            )

    def _escalate_sandbox_unavailable(
        self, role_name: str, backend_state: str, detail: str,
    ) -> None:
        """Submit a blocking escalation when the sandboxed-dispatch guard refuses.

        Filed at the fail-CLOSED refusal point in :meth:`_guard_sandbox` (task
        2908, PRD plans/os-sandbox-worktree-containment-prd.md D4 / invariant
        INV-4): sandboxing is required (``config.sandbox.enabled`` +
        ``role.sandboxed``) but the configured backend resolved to no available
        OS sandbox (landlock/bwrap), so the agent invocation is REFUSED rather
        than run unconfined — mirroring reconciliation's
        ``RemediationSandboxUnavailable`` (task 1935). The escalation is DEDUPED
        upstream by the process-global dedup in ``sandbox_dispatch`` (exactly one
        filing across N refused invocations, re-armed on any backend-state
        change), so this method is only ever called for the one refusal that
        should escalate.

        The summary/detail are phrased as a HOST-LEVEL condition (an unavailable
        backend affects every sandboxed role on the host, not just the one task
        that happened to refuse first) — ``task_id`` still names the first
        refusal for traceability, but the text makes clear other sandboxed tasks
        are equally impacted (task 2908 review). The detail also spells out the
        re-arm caveat: because the dedup re-arms only on a backend-STATE change,
        resolving this escalation without actually installing/enabling a backend
        will NOT cause a fresh one to be filed — continued refusals stay deduped
        (still fail-closed) until the backend state changes.

        Filed by the orchestrator (not a harness sentinel): ``agent_role``
        'orchestrator', ``severity`` 'blocking' (L0->steward), ``category``
        'infra_issue' — an unavailable host sandbox backend is an infra
        condition. Submission shape mirrors the sibling
        :meth:`_escalate_plan_overwrite`.
        """
        summary = (
            f'sandbox unavailable — sandboxed dispatch refused for '
            f'backend={backend_state} (host-level: affects all sandboxed roles '
            f'on this host). First refusal: role {role_name}, task {self.task_id}'
        )
        detail_msg = (
            f'Sandboxing is required (sandbox.enabled + role.sandboxed) but the '
            f'configured backend={backend_state!r} resolved to no available OS '
            f'sandbox (landlock/bwrap) on this host, so sandboxed dispatch is '
            f'REFUSED fail-closed (PRD D4, mirrors recon '
            f'RemediationSandboxUnavailable / task 1935). This is a HOST-LEVEL '
            f'infra condition: EVERY sandboxed role on this host is affected, '
            f'not only the task named here. The first refusal observed was role '
            f'{role_name!r} for task {self.task_id}; other sandboxed tasks that '
            f'refuse against the same backend propagate SandboxUnavailable to '
            f'their own failure handlers but are NOT individually escalated. '
            f'This escalation is deduped process-wide: exactly one filing across '
            f'N refused invocations, re-armed only on a backend-STATE change '
            f'(INV-4). NOTE: resolving THIS escalation alone will NOT re-file a '
            f'new one — the dedup re-arms only when the backend state changes (a '
            f'config edit via sandbox.backend / reload) or a backend recovers '
            f'(landlock/bwrap becomes available). If you resolve this without '
            f'installing/enabling a backend, subsequent refusals stay deduped '
            f'and silent (they still fail-closed) until the backend state '
            f'actually changes. Remediation: install/enable landlock or bwrap on '
            f'this host, or set sandbox.backend=none to explicitly run '
            f'UNSANDBOXED. Underlying refusal: {detail}'
        )
        logger.error(f'Task {self.task_id}: {summary}')

        if not self.escalation_queue:
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=summary,
            detail=detail_msg,
            suggested_action='install_sandbox_backend_or_set_backend_none',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={'escalation_id': esc.id, 'category': esc.category,
                      'severity': esc.severity, 'summary': summary[:200]},
            )

    def _emit_sandbox_unavailable(self, role_name: str, backend_state: str) -> None:
        """Emit one ``sandbox_unavailable`` event on a fail-closed refusal (β2).

        Fires from :meth:`_guard_sandbox`'s ``except SandboxUnavailable`` branch
        on EVERY refusal — the per-invocation structured record γ1's soak
        predicate queries — BEFORE the (deduped) escalation, so the event and
        the escalation deliberately have different cardinalities (PRD
        plans/os-sandbox-worktree-containment-prd.md β2 §Goal 3 / INV-4).
        ``role``/``task_id`` are first-class emit columns; ``data`` carries the
        configured backend that failed to resolve. A silent no-op when no event
        store is wired up (fire-and-forget, mirroring the sibling
        escalation-emit guard).
        """
        if not self.event_store:
            return
        self.event_store.emit(
            EventType.sandbox_unavailable,
            task_id=self.task_id,
            phase=self.state.value,
            role=role_name,
            data={'backend': backend_state},
        )

    def _emit_sandbox_applied(
        self, role_name: str, backend: str, write_set: WriteSet,
    ) -> None:
        """Emit one ``sandbox_applied`` event per sandboxed invocation (β2).

        Fired from :meth:`_invoke` on the NON-refusing guard path — a real
        backend (landlock/bwrap) OR the explicit ``backend=none`` operator
        escape hatch (still emitted; ``data.backend`` lets γ1 distinguish
        deliberately-unsandboxed from real containment). ``role``/``task_id``
        are first-class emit columns; ``data`` carries the resolved ``backend``
        and ``write_set.digest()`` — the stable writable-set hash an operator
        diffs to see exactly what this invocation could touch (PRD
        plans/os-sandbox-worktree-containment-prd.md β2 §Goal 3 / INV-2). The
        digest is read from ``WriteSet.digest()`` (its single owner, INV-5), not
        recomputed here. A silent no-op when no event store is wired up
        (fire-and-forget, mirroring the sibling escalation-emit guard).
        """
        if not self.event_store:
            return
        self.event_store.emit(
            EventType.sandbox_applied,
            task_id=self.task_id,
            phase=self.state.value,
            role=role_name,
            data={'backend': backend, 'digest': write_set.digest()},
        )

    def _ensure_sandbox_dirs(self, write_set: WriteSet) -> None:
        """Pre-create the load-bearing ``claude_fleet`` carve-out OUTSIDE the
        sandbox before dispatch (task 2996) — see
        ``orchestrator.agents.write_set.ensure_claude_fleet_dir`` for the full
        backend rationale (why neither the pure ``compute_write_set`` nor either
        backend can materialize this carve-out from inside the sandbox).

        This wrapper owns the operator-facing failure posture: best-effort, so
        on failure it WARNs LOUDLY (loud-over-silent) with the task_id + path
        and CONTINUES — a missing fleet dir degrades only fleet session-registry
        recording, not the agent's task, so aborting the whole dispatch would be
        a disproportionate fail-closed.
        """
        if not ensure_claude_fleet_dir(write_set):
            logger.warning(
                'task %s: could not pre-create %s outside the sandbox — both '
                'backends will skip the claude_fleet carve-out and in-sandbox '
                'session-registry writes may fail',
                self.task_id,
                write_set.claude_fleet,
            )

    def _guard_sandbox(self, role_name: str) -> str:
        """Fail-CLOSED sandbox check for the sandboxed-dispatch call-site (task 2908).

        Called from :meth:`_invoke` inside the ``config.sandbox.enabled +
        role.sandboxed`` block. Delegates backend resolution to
        ``sandbox_dispatch.resolve_backend_or_refuse`` and RETURNS the resolved
        backend string (``'none'``/``'landlock'``/``'bwrap'``) on the
        non-refusing path — the single authoritative value :meth:`_invoke`
        carries into the ``sandbox_applied`` event (β2), avoiding a redundant
        second resolution and any TOCTOU skew. A healthy backend and the
        operator ``backend=none`` escape hatch both return without raising, but
        a required-yet-unavailable backend raises ``SandboxUnavailable`` — at
        which point we emit a per-refusal ``sandbox_unavailable`` event (β2, NOT
        deduped) and then file a DEDUPED escalation (only when
        ``exc.should_escalate``, the process-global dedup decision — INV-4) and
        re-raise to REFUSE the agent invocation rather than run it unconfined
        (PRD D4). The re-raised exception propagates out of ``_invoke`` to
        ``_drive``'s generic failure handler.
        """
        from orchestrator.agents.sandbox_dispatch import (
            SandboxUnavailable,
            resolve_backend_or_refuse,
        )

        try:
            return resolve_backend_or_refuse()
        except SandboxUnavailable as exc:
            self._emit_sandbox_unavailable(role_name, exc.backend_state)
            if exc.should_escalate:
                self._escalate_sandbox_unavailable(
                    role_name, exc.backend_state, str(exc),
                )
            raise

    async def _check_scope_invariant(
        self, *, backend_metadata: dict | None = None,
    ) -> None:
        """Tripwire (task 2505): warn + escalate if ``plan.files`` and
        ``metadata.files`` diverge at LOCK-MODULE granularity at MERGE entry.

        The scope-reconciliation choke point (``_reconcile_scope_locks`` /
        ``_set_task_scope``) keeps ``plan.files`` and ``metadata.files`` in
        lockstep on every path that changes either. This surfaces a genuine
        divergence loudly (the project's loud-over-silent-degradation norm)
        rather than letting scope drift ship silently into a merge.

        Compared at MODULE (lock) granularity, NOT file granularity. Locks —
        the only thing ``metadata.files`` functionally drives — are
        module-granular (``files_to_modules`` at ``lock_depth``), and
        ``metadata.files`` is only ever persisted by the scheduler's
        ``handle_blast_radius_expansion``, which no-ops (never persists)
        whenever the derived module set is unchanged. So a benign same-module
        file addition by the architect (author declares ``pkg/a.py``; the
        architect plans ``pkg/a.py`` + same-package ``pkg/b.py``) legitimately
        leaves the two file lists differing while the module sets — and thus
        the locks and the merge — agree. A file-granularity comparison would
        false-escalate that common architect widen with a blocking
        ``infra_issue`` that routes to a human even though nothing is wrong;
        only a MODULE-set divergence (a real lock/scope-reconciliation bug) is
        escalation-worthy. The stronger file-level equality is still maintained
        by construction on the resume/grant path (``_set_task_scope`` persists
        ``metadata.files`` directly), so nothing is lost there.

        Fail-safe: an unreadable task (``self.scheduler.get_task`` returns
        ``None`` — e.g. a transient backend hiccup) is treated as "cannot
        check" and skipped, not "divergent" — a read failure must not wedge
        an otherwise-valid merge or false-escalate.

        Args:
            backend_metadata: Optional pre-read backend metadata blob. The
                merge-entry caller passes the blob
                :meth:`_stamp_merge_phase_entered` just read (review
                amendment), which is the SAME data this check would otherwise
                fetch — two back-to-back ``get_task`` round-trips (15s timeout
                each) for one blob, on the merge hot path. Threading it also
                removes a small inconsistency: the two reads happened at
                different instants, so the stamp and this check could see
                different snapshots of the same task. Anything that is not a
                dict (including ``None`` — the stamp could not read) falls
                back to this method's own ``get_task``, preserving the
                fail-safe above unchanged.
        """
        if isinstance(backend_metadata, dict):
            metadata = backend_metadata
        else:
            fresh_task = await self.scheduler.get_task(self.task_id)
            if fresh_task is None:
                return
            metadata = fresh_task.get('metadata') or {}
        plan_files = sanitize_files_for_persist(self.plan.get('files', []))
        metadata_files = list(metadata.get('files') or [])
        plan_modules = set(files_to_modules(plan_files, self.config.lock_depth))
        metadata_modules = set(
            files_to_modules(metadata_files, self.config.lock_depth)
        )
        if plan_modules == metadata_modules:
            return
        logger.warning(
            'Task %s: plan.files/metadata.files LOCK-MODULE divergence detected '
            'at MERGE entry — plan modules=%s (files=%s), metadata modules=%s '
            '(files=%s)',
            self.task_id, sorted(plan_modules), sorted(plan_files),
            sorted(metadata_modules), sorted(metadata_files),
        )
        self._escalate_scope_invariant_violation(
            sorted(plan_files), sorted(metadata_files),
        )

    def _escalate_scope_invariant_violation(
        self, plan_files: list[str], metadata_files: list[str],
    ) -> None:
        """Submit an ``infra_issue`` escalation for a plan.files/metadata.files
        divergence caught by :meth:`_check_scope_invariant` (task 2505).
        Mirrors :meth:`_escalate_plan_overwrite`'s submission shape.
        """
        summary = (
            f'plan.files/metadata.files divergence detected for task {self.task_id}'
        )
        detail = (
            f'plan.files={plan_files} but metadata.files={metadata_files} — '
            f'these derive DIFFERENT lock-module sets at lock_depth='
            f'{self.config.lock_depth} (a benign same-module file delta does '
            f'NOT reach here). The scope-reconciliation choke point '
            f'(_reconcile_scope_locks/_set_task_scope) should keep the module '
            f'sets in lockstep on every path that changes either.'
        )
        logger.error(f'Task {self.task_id}: {summary}')

        if not self.escalation_queue:
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=summary,
            detail=detail,
            suggested_action='investigate_and_retry',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={'escalation_id': esc.id, 'category': esc.category,
                      'severity': esc.severity, 'summary': summary[:200]},
            )

    def _escalate_corruption(self, corrupted: list[str]) -> None:
        """Submit an info-severity escalation for corrupted iteration log lines."""
        if not self.escalation_queue:
            logger.warning(
                'Task %s: %d corrupted iteration log lines (no escalation queue)',
                self.task_id, len(corrupted),
            )
            return

        from escalation.models import Escalation

        detail = f'{len(corrupted)} corrupted line(s):\n' + '\n'.join(
            line[:200] for line in corrupted[:10]
        )
        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='info',
            category='infra_issue',
            summary=f'{len(corrupted)} corrupted iteration log line(s)',
            detail=detail,
            suggested_action='investigate_log_corruption',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)

    def _handle_cancelled_terminal_exit(
        self, exc: TerminalExitRejection,
    ) -> WorkflowOutcome | None:
        """Return ``WorkflowOutcome.CANCELLED`` when *exc* signals an authoritative
        user/manual cancellation (``exc.old_status == 'cancelled'``), else ``None``.

        Centralises the predicate and log message used at two sites — run()'s
        ``SetTaskStatusRejected`` guard and ``_handle_terminal_exit_on_block``'s
        sub-case 0 — so they cannot drift.
        """
        if exc.old_status == 'cancelled':
            logger.info(
                'Task %s: cancelled out-of-band; aborting gracefully '
                '(no reopen, no escalation)',
                self.task_id,
            )
            self._enter_phase(WorkflowState.CANCELLED)
            return WorkflowOutcome.CANCELLED
        return None

    async def _handle_terminal_exit_on_block(
        self,
        exc: TerminalExitRejection,
        reason: str,
        detail: str,
    ) -> WorkflowOutcome | None:
        """Detect ``update_task(status='done')`` bypass after a terminal-exit rejection.

        When ``set_task_status('blocked')`` raises ``TerminalExitRejection``
        the row is already terminal. Three sub-cases, branching on ``exc.old_status``:

        0. Cancelled (``old_status == 'cancelled'``) — a user/manual cancellation
           landed out-of-band. This is authoritative; the workflow must NOT reopen
           the row or file any escalation. Log an info line and return
           ``WorkflowOutcome.CANCELLED`` immediately (before any ``get_task``
           round-trip or provenance read).
        1. Legitimate done (``old_status == 'done'``, provenance commit reachable
           from main) — return ``None`` so the existing flow continues; the
           post-steward branch at ``current == 'done'`` returns DONE.
        2. Bypass (``old_status == 'done'``, provenance missing or commit not on
           main) — reopen the row via
           ``set_task_status('blocked', reopen_reason='bypass detected: …')``,
           file an L1 with ``category='bypass_done'``, and return
           ``WorkflowOutcome.BLOCKED``.
        """
        # Sub-case 0: authoritative user/manual cancellation — abort gracefully.
        outcome = self._handle_cancelled_terminal_exit(exc)
        if outcome is not None:
            return outcome

        task = await self.scheduler.get_task(self.task_id)
        # Scheduler.get_task normalises task['metadata'] to a dict at the
        # boundary (via _normalize_task_metadata) so we can read it directly.
        metadata: dict = (task.get('metadata') or {}) if isinstance(task, dict) else {}

        provenance = metadata.get('done_provenance') if metadata else None
        commit = ''
        if isinstance(provenance, dict):
            commit = str(provenance.get('commit') or '').strip()

        if commit:
            try:
                main_sha = await self.git_ops.get_main_sha()
                if await self.git_ops.is_ancestor(commit, main_sha):
                    logger.info(
                        'Task %s: terminal-exit on _mark_blocked but provenance '
                        'commit %s is reachable from main — legitimate done',
                        self.task_id, commit[:12],
                    )
                    return None
            except Exception:
                logger.warning(
                    'Task %s: ancestor check failed during bypass detection',
                    self.task_id, exc_info=True,
                )

        logger.warning(
            'Task %s: terminal-exit on _mark_blocked with missing or off-main '
            'done_provenance (commit=%s); reopening + filing L1 bypass_done',
            self.task_id, commit[:12] if commit else '<none>',
        )

        bypass_summary = f'bypass detected: {reason[:160]}'
        try:
            await self.scheduler.set_task_status(
                self.task_id, 'blocked', reopen_reason=bypass_summary,
            )
        except Exception:
            logger.exception(
                'Task %s: reopen with reopen_reason failed; continuing to L1',
                self.task_id,
            )

        if self.escalation_queue:
            from escalation.models import Escalation
            l1 = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='bypass_done',
                summary=(
                    f'Mark-done bypass detected for task {self.task_id}: '
                    f'row is {exc.old_status!r}, provenance commit not on main'
                ),
                detail=(
                    f'reason: {reason}\n'
                    f'detail: {detail}\n'
                    f'old_status: {exc.old_status}\n'
                    f'evidence_commit: {commit or "<none>"}\n'
                ),
                suggested_action='investigate_bypass_done',
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
                level=1,
            )
            self.escalation_queue.submit(l1)
            if self.event_store:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=self.task_id, phase=self.state.value,
                    data={
                        'escalation_id': l1.id, 'category': 'bypass_done',
                        'severity': 'blocking', 'level': 1,
                        'summary': l1.summary[:200],
                    },
                )

        return WorkflowOutcome.BLOCKED

    async def _maybe_cleanup_done_worktree(self) -> None:
        """Clean up worktree+branch only when the branch is reachable from main.

        Forensics 2026-05-08: when an agent bypassed the merge via
        ``update_task(status='done')``, the workflow returned DONE while the
        branch HEAD was still off-main. Cleanup deleted the unmerged branch;
        only the loose-object grace period kept the work recoverable.

        Mirrors the predicate at ``run()``'s ghost-loop early exit.
        """
        if (
            self.state != WorkflowState.DONE
            or self.worktree is None
            or self._worktree_external
        ):
            return
        branch_name = self.task_id
        try:
            rc, branch_head_raw, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
            )
        except Exception:
            logger.warning(
                'Task %s: rev-parse HEAD failed in cleanup gate; preserving worktree',
                self.task_id, exc_info=True,
            )
            return
        if rc != 0:
            logger.warning(
                'Task %s: rev-parse HEAD returned %d in cleanup gate; '
                'preserving worktree+branch for inspection',
                self.task_id, rc,
            )
            return
        branch_head = branch_head_raw.strip()
        try:
            main_sha = await self.git_ops.get_main_sha()
        except Exception:
            logger.warning(
                'Task %s: get_main_sha failed in cleanup gate; preserving worktree',
                self.task_id, exc_info=True,
            )
            return
        if await self.git_ops.is_ancestor(branch_head, main_sha):
            await self.git_ops.cleanup_worktree(self.worktree, branch_name)
            return
        # Fallback: branch tip is a pre-rebase duplicate that is NOT an ancestor
        # of main, but the recorded merge commit (_merge_sha) IS on main.  The
        # task's work was merged via a rebase-on-merge flow.  Reclaim the
        # regenerable build cache while preserving the worktree+branch for forensics.
        merge_sha = self._merge_sha
        if merge_sha and await self.git_ops.is_ancestor(merge_sha, main_sha):
            reclaimed = await self.git_ops.reclaim_worktree_build_artifacts(
                self.worktree,
            )
            logger.warning(
                'Task %s: branch HEAD %s not on main but merge commit %s is on '
                'main — reclaimed build artifacts %s; preserving worktree+branch',
                self.task_id, branch_head[:12], merge_sha[:12],
                [str(p) for p in reclaimed],
            )
            return
        logger.warning(
            'Task %s: state=DONE but branch HEAD %s not reachable from main %s '
            '— preserving worktree+branch for inspection',
            self.task_id, branch_head[:12], main_sha[:12],
        )

    async def _mark_blocked(
        self, reason: str, *, detail: str = '',
        block_status: str = 'blocked',
        skip_escalation: bool = False,
        merge_phase: bool = False,
        escalate_to_human: bool = False,
        suggested_action: str = 'investigate_and_retry',
        category: str = 'task_failure',
        dedupe_fingerprint: str | None = None,
        spawn_dry_run: bool = False,
        root_cause: str = '',
        disposition: BlockDisposition | None = None,
    ) -> WorkflowOutcome:
        """Mark task as blocked and optionally create an escalation entry.

        *reason* is used as the escalation summary (truncated to 200 chars).
        *detail* is the full diagnostic context persisted in the escalation
        file; defaults to *reason* when not provided.
        *block_status* (task 2200 / ω4) is the task-row status written when
        *merge_phase* is False — defaults to the generic ``'blocked'``.  Pass
        ``'infra-hold'`` to land the row on the first-class infra-hold status
        (PRD C7/D3) instead, e.g. for the verify-infra STAMP so the harness
        HOLD guard and RESUME cascade (both keyed on
        :func:`orchestrator.task_status.is_infra_held`) see it.  The internal
        ``WorkflowState.BLOCKED`` transition and escalation filing below are
        unaffected by this choice.
        *skip_escalation* suppresses escalation creation when a level-1
        escalation already exists (e.g. steward re-escalated to human).
        *merge_phase* suppresses task-status transitions (blocked/pending)
        when the caller will retry the merge in-place rather than requeueing
        through the scheduler.
        *escalate_to_human* (Fix C) skips the steward entirely and submits
        an L1 escalation immediately.  Use when the caller has determined
        a confirmed loop / unresolvable failure that the steward cannot
        meaningfully un-stick (e.g. ≥2 consecutive no-plan failures on
        the same main SHA).
        *dedupe_fingerprint* when provided stamps the escalation and routes
        submission through submit_or_dedupe so N tasks seeing the same
        inherited-from-main break collapse to a single parent escalation.
        All existing callers (no fingerprint) keep the raw-submit path.
        *spawn_dry_run* opts in to run_dry_run_unblock even when
        *merge_phase* is True.  Use ONLY for the post-merge red-main class
        (POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX): main is already advanced
        before this path runs, so _capture_worktree_shas naturally reflects
        post-merge reality and the B3 gate can act on the resulting proposal.
        All other merge_phase=True escalation paths (dropped_plan_targets,
        post_merge_equivalence, transient_infra, plan_files_not_touched) are
        human-judgement/infra cases and must NOT receive a proposal — leave
        their callers with spawn_dry_run=False (the default).
        """
        if self.machine.is_terminal():
            logger.warning(
                'Task %s: already %s, ignoring late blocked transition: %s',
                self.task_id, self.state.value, reason,
            )
            return WorkflowOutcome(self.state.value)

        # REVIEW-CYCLE-1 fix: snapshot the PRE-block WORKING phase (e.g.
        # VERIFY/REVIEW) BEFORE the `if not merge_phase` branch below calls
        # _enter_phase(BLOCKED) — mirrors the deleted `_last_block_phase =
        # self.state.value` pre-block stash.  Threaded into _record so every
        # return point (including the merge_phase=True paths, which never
        # transition) stamps TerminalReport.blocked_from_phase with the phase
        # this call was entered at, distinct from `phase` (machine.state at
        # _record time, kept == machine.state for SM-2).
        pre_block_state = self.machine.state

        def _record(outcome: WorkflowOutcome) -> WorkflowOutcome:
            """Build the atomic TerminalReport at a _mark_blocked return point.

            Captures ``self.machine.state`` AT CALL TIME — BLOCKED after
            ``_enter_phase(BLOCKED)`` for the block-return paths below, or
            DONE/etc. for the bypass/steward-resolved paths that never enter
            BLOCKED.  W9-ε: TerminalReport.category is sourced from
            *disposition*.category (a :class:`FailureCategory`) when the
            caller supplied a :class:`BlockDisposition`; back-compat callers
            with no disposition keep the pre-W9-ε hard ``None`` (see design
            decisions) — _mark_blocked's category= parameter remains a
            SEPARATE, caller-supplied ESCALATION taxonomy, not FailureCategory.
            """
            self._terminal_report = TerminalReport(
                outcome=outcome, reason=reason, phase=self.machine.state,
                detail=(detail or reason),
                category=(disposition.category if disposition is not None else None),
                blocked_from_phase=pre_block_state,
            )
            return outcome

        if not merge_phase:
            self._enter_phase(WorkflowState.BLOCKED)
            _status_set_ok = False
            try:
                await self.scheduler.set_task_status(self.task_id, block_status)
                _status_set_ok = True
            except TerminalExitRejection as exc:
                bypass_outcome = await self._handle_terminal_exit_on_block(
                    exc, reason, detail or reason,
                )
                if bypass_outcome is not None:
                    return _record(bypass_outcome)
                # Legitimate done — fall through; the existing post-steward
                # flow below handles current==done by returning DONE.
            if _status_set_ok:
                # Best-effort staleness-reference stamp (task 2557): records
                # the confirmed block transition so BriefingAssembler can
                # tell a stale persisted dry_run_proposals entry (from a
                # PRIOR block cycle, re-blocked without a fresh investigation)
                # apart from a fresh one. Awaited synchronously here, BEFORE
                # the fire-and-forget _spawn_dry_run_unblock below appends its
                # own later-timestamped proposal — so a fresh investigation's
                # proposal.timestamp always lands after last_blocked_at.
                # Default metadata_mode ('merge') preserves sibling keys
                # (incl. dry_run_proposals). Never raises — mirrors the
                # existing best-effort dry_run proposal-list trim in
                # dry_run_unblock.py.
                try:
                    await self.scheduler.update_task(
                        self.task_id, {'last_blocked_at': datetime.now(UTC).isoformat()},
                    )
                except Exception as exc:
                    logger.warning(
                        '_mark_blocked: last_blocked_at stamp failed for task %s '
                        '(best-effort, continuing): %s',
                        self.task_id, exc,
                    )
                self._spawn_dry_run_unblock(
                    reason, detail or reason,
                    block_class=(disposition.block_class if disposition is not None else None),
                )
        elif spawn_dry_run:
            # Post-merge red-main class: merge_phase suppressed the status
            # transition and the spawn above, but this IS the mechanically
            # auto-unblockable class.  _spawn_dry_run_unblock is idempotent
            # (enabled/worktree/in-flight-dedup guards) and captures SHAs
            # from the task worktree at spawn time — by then advance_main
            # has already moved refs/heads/main to advanced_sha, so the
            # proposal reflects post-merge reality for b3_gate.check_proposal.
            #
            # ORDERING INVARIANT: spawn_dry_run=True is only valid when
            # advance_main has already committed main.  The only reason prefix
            # in this category is POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX —
            # _check_post_merge_pyright is invoked exclusively from
            # _finalize_advanced_merge, which runs AFTER a successful
            # advance_main.  Asserting the prefix here enforces the contract
            # so a future caller on a pre-advance path fails loudly rather
            # than silently anchoring a dry-run proposal to a stale SHA that
            # would cause b3_gate.check_proposal to act on incorrect data.
            # If you need to add a second post-advance class, extend this
            # check rather than removing it.
            from orchestrator.merge_queue import (
                POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
            )
            if not reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX):
                raise AssertionError(
                    f'spawn_dry_run=True requires advance_main to have already '
                    f'moved refs/heads/main (post-merge red-main class only). '
                    f'Reason prefix {reason[:80]!r} does not match '
                    f'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX; this path may '
                    f'be pre-advance and would produce a stale-SHA proposal '
                    f'that causes b3_gate to act on incorrect data.  Update '
                    f'this guard if adding a new post-advance class.'
                )
            self._spawn_dry_run_unblock(
                reason, detail or reason,
                block_class=(disposition.block_class if disposition is not None else None),
            )
        logger.warning(f'Task {self.task_id} BLOCKED: {reason}')

        if self.escalation_queue and skip_escalation:
            # Defensive cleanup: L0 should already be cleared by
            # _wait_for_resolution, but dismiss any stragglers (race
            # between the L0-empty check and the L1 check).
            remaining_l0 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            if remaining_l0:
                logger.warning(
                    'Task %s: %d L0 escalation(s) still pending at '
                    'mark_blocked(skip_escalation=True) — dismissing',
                    self.task_id, len(remaining_l0),
                )
                for esc in remaining_l0:
                    self.escalation_queue.resolve(
                        esc.id,
                        'Auto-dismissed: task blocked after steward re-escalation',
                        dismiss=True,
                        resolved_by='auto-dismissed',
                    )

        if self.escalation_queue and not skip_escalation:
            # Fix C short-circuit: the caller determined this failure is not
            # resolvable by the steward (e.g. false premise, unactionable spec,
            # confirmed cycle).  Skip the L0 entirely — a stopped steward
            # cannot consume it — and submit only the human-facing L1.
            # This also prevents duplicate design_concern escalations in the
            # queue (an L0 the steward can never act on + an L1 for the same
            # report), keeping dashboards clean and preventing orphan-L0 reaper
            # noise.  The unactionable handler follows the same pattern.
            if escalate_to_human:
                await self._ensure_l1_escalation_for_blocked(
                    reason, detail or reason, category=category,
                    root_cause=root_cause,
                )
                return _record(WorkflowOutcome.BLOCKED)

            # Don't create a duplicate if level-1 already pending — but only a
            # SAME-signature L1 should suppress the steward-facing L0 (task 2757):
            # an unrelated open L1 (different category) must not silently drop a
            # new root cause's L0.
            if not self.escalation_queue.has_open_l1(self.task_id, category=category):
                from escalation.models import Escalation

                esc = Escalation(
                    id=self.escalation_queue.make_id(self.task_id),
                    task_id=self.task_id,
                    agent_role='orchestrator',
                    severity='blocking',
                    category=category,
                    summary=reason[:200],
                    detail=detail or reason,
                    suggested_action=suggested_action,
                    worktree=str(self.worktree) if self.worktree else None,
                    workflow_state=self.state.value,
                )
                if dedupe_fingerprint:
                    # Cross-task N->1 dedup: stamp the fingerprint and route
                    # through submit_or_dedupe so sibling tasks seeing the same
                    # inherited-from-main break collapse to a single parent.
                    # All callers without a fingerprint keep raw-submit behaviour.
                    from escalation.dedupe import (
                        DedupeConfig,
                        content_fingerprint_key,
                        submit_or_dedupe,
                    )
                    esc.dedupe_fingerprint = dedupe_fingerprint
                    result = submit_or_dedupe(
                        self.escalation_queue,
                        esc,
                        DedupeConfig(
                            infra_dedupe_enabled=True,
                            infra_dedupe_window_secs=float('inf'),
                            infra_dedupe_categories=(category,),
                            key_fn=content_fingerprint_key,
                        ),
                    )
                    # Emit escalation_created only when this task is the parent
                    # (status='queued' means a new parent was filed, not a child fold).
                    if result.get('status') == 'queued' and self.event_store:
                        self.event_store.emit(
                            EventType.escalation_created,
                            task_id=self.task_id, phase=self.state.value,
                            data={'escalation_id': esc.id, 'category': category,
                                  'severity': 'blocking', 'summary': reason[:200]},
                        )
                    # Per-task stewards for child folds: dedup collapses ESCALATION
                    # ENTRIES to one parent (N-1 siblings don't each add a queue file),
                    # but each sibling task still spins up its own steward below.  This
                    # is intentional — each blocked task needs its own steward to watch
                    # for resolution and unblock the task when the hotfix lands.  The
                    # single-hotfix goal is achieved by collapsing notification/queue
                    # entries, not by preventing individual tasks from monitoring their
                    # own blocked state.
                else:
                    self.escalation_queue.submit(esc)
                    if self.event_store:
                        self.event_store.emit(
                            EventType.escalation_created,
                            task_id=self.task_id, phase=self.state.value,
                            data={'escalation_id': esc.id, 'category': category,
                                  'severity': 'blocking', 'summary': reason[:200]},
                        )

            # Give the steward a chance to resolve the escalation
            await self._ensure_steward_started()
            if self._steward:
                outcome = await self._await_steward_completion()

                async def _requeue() -> WorkflowOutcome:
                    if self.event_store:
                        self.event_store.emit(
                            EventType.escalation_resolved,
                            task_id=self.task_id, phase=self.state.value,
                            data={'outcome': 'requeued'},
                        )
                    if not merge_phase:
                        await self.scheduler.set_task_status(self.task_id, 'pending')
                        logger.info(
                            f'Task {self.task_id}: steward resolved blocking '
                            f'escalation, reset to pending for re-scheduling'
                        )
                    else:
                        # merge_phase=True: the caller retries the merge
                        # in-place while the task stays in-progress (no
                        # scheduler re-dispatch), so nothing durable records
                        # the obligation. Stamp metadata.merge_retry_pending
                        # (best-effort) so a restart mid-retry can reconstruct
                        # it via _resume_merge_retry_if_pending (Reify 5166).
                        await self._stamp_merge_retry_pending()
                        logger.info(
                            f'Task {self.task_id}: steward resolved blocking '
                            f'escalation, caller will retry merge in-place'
                        )
                    return _record(WorkflowOutcome.REQUEUED)

                # Single isinstance dispatch on the typed outcome (task 2248 /
                # W9-delta, SO-1) — replaces the forensic re-read of scheduler
                # status + escalation-queue state (current-status / L0-empty /
                # has_open_l1 / deferred / timestamp-window dismissal probes)
                # that used to live here.
                if isinstance(outcome, StewardResolved):
                    return await _requeue()

                if isinstance(outcome, StewardTerminalDecision):
                    if outcome.new_status == TaskStatus.DONE:
                        self._enter_phase(WorkflowState.DONE)
                        return _record(WorkflowOutcome.DONE)
                    # 'cancelled'/'deferred' are steward-driven terminal or
                    # preserved decisions — no L1 needed, do not requeue.
                    logger.info(
                        'Task %s: steward-driven status is %s — preserving, '
                        'not re-queueing', self.task_id, outcome.new_status.value,
                    )
                    return _record(WorkflowOutcome.BLOCKED)

                if isinstance(outcome, StewardReescalatedL1):
                    # The steward's _auto_escalate_to_human already filed the
                    # L1 and dismissed its L0 before publishing this outcome —
                    # nothing left for _mark_blocked to do.
                    logger.info(
                        'Task %s: L1 escalation open — steward handed '
                        'off to human; leaving status as-is and exiting',
                        self.task_id,
                    )
                    return _record(WorkflowOutcome.ESCALATED)

                if isinstance(outcome, StewardInterrupted) and outcome.wip_commits_present:
                    # Task-2060 fix: a steward interruption (attempt-cap
                    # or timeout) with real work already committed must
                    # resume the plan, not be triaged as "steward
                    # failed".  Dismiss the still-pending L0 — its only
                    # consumer (the steward) is done — and re-pend.
                    if self.escalation_queue:
                        orphan_l0 = self.escalation_queue.get_by_task(
                            self.task_id, status='pending', level=0,
                        )
                        for esc in orphan_l0:
                            self.escalation_queue.resolve(
                                esc.id,
                                'Auto-dismissed: steward interrupted '
                                f'({outcome.reason}) with WIP present — '
                                'resuming plan, not escalating (task '
                                '2060 fix)',
                                dismiss=True, resolved_by='auto-dismissed',
                            )
                        if orphan_l0:
                            logger.info(
                                'Task %s: dismissed %d pending L0(s) — '
                                'steward interrupted (%s) with WIP '
                                'present, resuming plan',
                                self.task_id, len(orphan_l0), outcome.reason,
                            )
                    return await _requeue()
                # A StewardInterrupted with no WIP (checked above; falls
                # through when the combined condition is False) — and a
                # StewardBudgetExhausted (never matches any isinstance check
                # above) — both fall through to the shared BLOCKED
                # fall-through below (outside this `if self._steward:` block)
                # instead of returning here.  That shared block files the
                # same deduped L1 AND dismisses this still-pending L0 in one
                # place — fixing the orphan-L0 leak (task 2248 review fix /
                # incident esc-3576-234) where a workflow-synthesized
                # grace-timeout StewardInterrupted('timeout', wip=False) (the
                # steward was killed before its own _auto_escalate_to_human
                # could dismiss the L0) used to file an L1 and return without
                # dismissing the L0, stranding it for the orphan-L0 reaper to
                # promote into a duplicate L1.  For a steward-PUBLISHED
                # no-wip outcome the L0 is already dismissed by
                # _auto_escalate_to_human, so the shared block's dismissal
                # loop is a harmless no-op there.

        # Fall-through BLOCKED: either no escalation queue, the steward
        # never resolved the L0, or a StewardInterrupted(wip=False) /
        # StewardBudgetExhausted outcome fell through from above.  Either
        # way a human should know — submit an L1 (deduped) so the task
        # isn't silently parked.
        #
        # skip_escalation guard (task 2757): a skip_escalation=True caller has
        # ALREADY filed its own signature-specific L1 (train halt → wip_conflict/
        # unmerged_state; main-health auto-heal → preexisting_main_break; architect
        # no-plan → the promoted L0; steward re-escalation → the steward's L1).
        # Before the terminal gate became signature-aware, the bare
        # has_open_l1(task_id) dedup masked this fall-through for those callers
        # (any open L1 suppressed it).  Now that _ensure_l1_escalation_for_blocked
        # dedups on category, this default-category='task_failure' fall-through no
        # longer matches the caller's differently-categorised L1 and would
        # DOUBLE-file.  Respect skip_escalation here so the caller stays the sole
        # owner of the human-facing escalation.  The generic merge-blocked path
        # (merge_phase=True, skip_escalation unset) is unaffected — it still
        # surfaces unconditionally, which is task 2757's whole point.
        if not skip_escalation:
            await self._ensure_l1_escalation_for_blocked(
                reason, detail or reason, category=category,
            )

        # Fix #2 — dismiss any still-pending L0 now that we are exiting BLOCKED.
        # The steward has finished (or never ran) and this workflow slot is
        # about to exit, so the L0's only consumer is dead.  Leaving it pending
        # strands an orphan that the orphan-L0 reaper later promotes into a
        # DUPLICATE L1 (esc-3576-234 in the 2026-05-29 incident) — pure churn,
        # since _ensure_l1_escalation_for_blocked above already filed the human-
        # facing L1.  Mirrors the defensive dismissal block at the
        # skip_escalation path above.  Guarded on escalation_queue because the
        # fall-through is also reached when no queue is wired.
        if self.escalation_queue:
            orphan_l0 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            for esc in orphan_l0:
                self.escalation_queue.resolve(
                    esc.id,
                    'Auto-dismissed: workflow exiting BLOCKED, steward done '
                    '(L1 filed for human handoff)',
                    dismiss=True,
                    resolved_by='auto-dismissed',
                )
            if orphan_l0:
                logger.info(
                    'Task %s: dismissed %d orphan L0(s) on BLOCKED fall-through '
                    '(steward consumer dead; L1 already filed)',
                    self.task_id, len(orphan_l0),
                )
        return _record(WorkflowOutcome.BLOCKED)

    def _durable_ref_suffix(self) -> str:
        """Durable git identifiers to append to an L1 escalation's detail.

        The originating worktree is ephemeral — task and merge worktrees are
        reaped (e.g. by the merge queue or a disk-pressure prune) well before
        a human reads a human-facing L1.  So cite refs that survive: the task
        branch and the SHAs bracketing its work.  ``tip`` is the merge commit
        SHA when the merge already landed, else a label pointing at the
        branch's current HEAD.
        """
        branch = f'{self.config.git.branch_prefix}{self.task_id}'
        base = (self._base_commit or '')[:12] or 'unknown'
        tip = (self._merge_sha or '')[:12] or f'{branch}@HEAD'
        return f'\n\n[durable refs] branch={branch} base={base} tip={tip}'

    async def _build_train_state(self) -> TrainState | None:
        """Build per-train context for L1 escalation payloads (PRD § 9.8).

        Returns None when this task carries no valid metadata.train (non-train
        task or malformed metadata).  When valid, returns::

            {'id': str, 'order': int,
             'parked_members': list[str],  # siblings at merge-deferred, excl. self
             'failing_member': str}        # this task's id

        Fast path: if metadata.train.members is a list, call get_statuses on
        those ids and filter.  Fallback: scan get_tasks() for tasks whose
        metadata.train.id matches; status is read directly from the task dicts
        (avoiding a redundant get_statuses round-trip — get_tasks already embeds
        per-task status in each dict).

        Reuses the TrainMembership cast/isinstance convention from task 1522
        (git_ops.py:54-62); cast and TrainMembership are already imported.
        """
        metadata = self.task.get('metadata') or {}
        train_meta = metadata.get('train')
        if not isinstance(train_meta, dict):
            return None
        train = cast(TrainMembership, train_meta)

        train_id = train.get('id')
        train_order = train.get('order')
        if train_id is None or train_order is None:
            return None

        # Discover candidate sibling task ids and their statuses.
        members_cache = train.get('members')
        if isinstance(members_cache, list):
            # Fast path — use the cached members list; fetch statuses from the server.
            candidates: list[str] = [str(m) for m in members_cache]
            statuses, err = await self.scheduler.get_statuses(candidates)
            if err is not None:
                logger.warning(
                    'Task %s: train %r _build_train_state get_statuses error '
                    '(parked-member list may be incomplete): %s',
                    self.task_id, train_id, err,
                )
                # DEGRADE-intentional: an empty parked_members under-reports
                # siblings in the escalation payload but does not suppress the
                # escalation itself.  Prefer DEGRADE over ABORT — the L1
                # escalation must still fire even with partial context.
                # statuses falls through to the parked_members comprehension
                # below, which safely yields [] on an empty dict.
        else:
            # Fallback scan — discover siblings via get_tasks() filtered to the
            # active set (ACTIVE_TASK_STATUSES).  Done siblings are not parked
            # (merge-deferred ∈ active), so excluding terminal is correct and
            # shrinks the payload on this fallback path.
            # Status is already embedded in each task dict; avoid a second round-trip.
            tasks = await self.scheduler.get_tasks(statuses=ACTIVE_TASK_STATUSES)
            statuses: dict[str, str] = {str(t['id']): str(t.get('status', 'unknown')) for t in tasks}
            candidates = [
                str(t['id'])
                for t in tasks
                if isinstance((t.get('metadata') or {}).get('train'), dict)
                and cast(TrainMembership, (t.get('metadata') or {}).get('train')).get('id') == train_id
            ]

        parked_members = [
            c for c in candidates
            if c != self.task_id and statuses.get(c) == 'merge-deferred'
        ]
        return cast('TrainState', {
            'id': train_id,
            'order': train_order,
            'parked_members': parked_members,
            'failing_member': self.task_id,
        })

    async def _ensure_l1_escalation_for_blocked(
        self, reason: str, detail: str, *, category: str = 'task_failure',
        root_cause: str = '',
    ) -> None:
        """Submit a level-1 escalation if none is open for this task.

        Called from BLOCKED-return paths so a human is signaled when
        automated handlers cannot make progress.  Idempotent — deduped
        via ``has_open_l1``.

        Cites durable refs (branch + base/tip SHAs) in ``detail`` and leaves
        ``worktree=None``: the human reads this after the worktree may be
        gone, so the ephemeral path is worse than useless.  (The L0 builder
        keeps ``worktree=`` — the steward is live and acts *in* that tree.)
        """
        if not self.escalation_queue:
            return
        # Signature-aware dedup (task 2757): a NEW root cause (different category)
        # must not be silently suppressed by an UNRELATED open L1.  This is the
        # terminal silent-drop for the generic merge-blocked path — reached
        # unconditionally from _mark_blocked's fall-through call.
        if self.escalation_queue.has_open_l1(self.task_id, category=category):
            return
        from escalation.models import Escalation

        # PRD § 9.8 — inject per-train context for park-prefix derail escalations.
        # Non-train tasks get train_state=None (the Escalation field default).
        # This is the single L1 chokepoint (3 callers inside _mark_blocked at
        # ~4861/~4980/~5007), so it covers every park-prefix derail path.
        train_state = await self._build_train_state()

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category=category,
            summary=f'Workflow blocked, no automated resolution path: {reason[:160]}',
            detail=(detail or reason) + self._durable_ref_suffix(),
            suggested_action='manual_intervention',
            worktree=None,
            workflow_state=self.state.value,
            level=1,
            train_state=train_state,
            root_cause=root_cause,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={
                    'escalation_id': esc.id, 'category': category,
                    'severity': 'blocking', 'level': 1,
                    'summary': reason[:200],
                },
            )
        logger.warning(
            'Task %s: submitted L1 escalation %s for unresolved BLOCKED state',
            self.task_id, esc.id,
        )

    def _spawn_dry_run_unblock(
        self, reason: str, detail: str,
        block_class: BlockClass | None = None,
    ) -> None:
        """Fire-and-forget: spawn an autonomous dry-run investigation.

        Skips when unblock_auto is disabled. Wraps asyncio.create_task so
        _mark_blocked never awaits the investigation — it is a pure side-effect.
        Any exception inside run_dry_run_unblock is caught there and written
        as a fallback proposal entry, so unhandled task exceptions are closed.

        *block_class* (W9-ε) lets a disposition-aware caller supply the
        typed BlockClass explicitly (``_mark_blocked`` passes
        ``disposition.block_class`` when a :class:`BlockDisposition` was
        given); when omitted (``None``, the back-compat default for the
        ~55 disposition-less ``_mark_blocked`` callers), it is derived via
        the ``classify_block_reason(reason)`` prose sniff exactly as before.
        """
        if not getattr(self.config, 'unblock_auto', None) or not self.config.unblock_auto.enabled:
            return
        if self.worktree is None:
            logger.debug(
                'Task %s: skipping dry-run unblock hook — no worktree set',
                self.task_id,
            )
            return
        # Skip if an investigation for this task is already running (e.g. rapid
        # re-blocks during steward retries).  Duplicate proposals are unhelpful
        # and would multiply budget spend up to 600 s × N re-blocks.
        _task_name = f'unblock-auto-{self.task_id}'
        if any(t.get_name() == _task_name and not t.done() for t in self._background_tasks):
            logger.debug(
                'Task %s: dry-run investigation already in progress, skipping duplicate spawn',
                self.task_id,
            )
            return
        if block_class is None:
            block_class = classify_block_reason(reason)
        try:
            _task = asyncio.create_task(
                run_dry_run_unblock(
                    task_id=self.task_id,
                    worktree=str(self.worktree),
                    reason=reason,
                    detail=detail,
                    scheduler=self.scheduler,
                    mcp=self.mcp,
                    config=self.config,
                    event_store=getattr(self, 'event_store', None),
                    usage_gate=self.usage_gate,
                    cost_store=self.cost_store,
                    block_class=block_class,
                ),
                name=_task_name,
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
        except Exception as exc:
            logger.warning(
                'Task %s: failed to spawn dry-run unblock hook: %s',
                self.task_id, exc,
            )

    async def _write_completion_to_memory(self) -> None:
        """Write task completion summary so dependent tasks find it in briefings.

        The ``add_memory`` call is tagged with
        ``metadata={'task_id': str(self.task_id), 'source': 'orchestrator_completion'}``
        so this organic completion note is findable by task_id-keyed audits.
        Qdrant payload filters and the deterministic
        ``get_memories_by_metadata`` reconciliation audits match on
        ``metadata.task_id``; without the tag the note (which embeds the id only
        in ``agent_id``) is audit-invisible.  ``task_id`` is stamped in
        exact-string form to match the form those audits query on.

        Which reconciliation surface finds this note (design-coherence note):

        - **External / audit surfaces** — ``get_memories_by_metadata`` task_id
          sweeps and the Stage-2 done-task completion-memory audit's per-task
          *semantic search* step (Stage 2 searches related memories for each
          done task it evaluates and will surface this note there).  This is the
          intended reach of the stamp: the note is now visible to the widened
          done-task audit instead of being invisible.
        - **NOT the primary count-gate** — Stage 2's cheap first gate matches
          ``{'task_id': str, 'stage2_suppress': True}``.  This note deliberately
          does NOT carry ``stage2_suppress``, so it does not short-circuit that
          gate.  That is intentional, not an oversight: ``stage2_suppress`` is
          Stage 2's OWN "already audited & suppressed" marker; if the
          orchestrator pre-stamped it, every organically-completed task would be
          skipped by the count-gate before Stage 2 ever evaluated its coverage,
          re-introducing exactly the done-task audit gap the widened audit
          exists to close.  Leaving it off keeps each done task visible to the
          audit's evaluation path while the note still spares it a redundant
          re-write (the semantic search finds it).
        """
        parts = [f"Completed: {self.task.get('title', '')}"]

        desc = self.task.get('description', '')
        if desc:
            parts.append(f"Description: {desc}")

        analysis = self.plan.get('analysis', '')
        if analysis:
            parts.append(f"Analysis: {analysis}")

        decisions = self.plan.get('design_decisions', [])
        if decisions:
            decision_text = '; '.join(
                d.get('decision', '') for d in decisions[:3]
            )
            parts.append(f"Key decisions: {decision_text}")

        steps = self.plan.get('steps', [])
        done_count = sum(1 for s in steps if isinstance(s, dict) and s.get('status') == 'done')
        parts.append(f"Steps completed: {done_count}/{len(steps)}")

        if self.modules:
            parts.append(f"Modules: {', '.join(self.modules)}")

        content = '\n'.join(parts)

        if self.mcp is None:
            return
        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                await client.post(
                    f'{self.mcp.url}/mcp/',
                    json={
                        'jsonrpc': '2.0',
                        'id': 1,
                        'method': 'tools/call',
                        'params': {
                            'name': 'add_memory',
                            'arguments': {
                                'content': content,
                                'category': 'observations_and_summaries',
                                'project_id': self.config.fused_memory.project_id,
                                'agent_id': f'orchestrator-task-{self.task_id}',
                                # Stamp task_id (exact-string form) + source so the
                                # note is audit-visible via metadata.task_id —
                                # found by get_memories_by_metadata sweeps and the
                                # Stage-2 done-task audit's semantic-search step.
                                # Deliberately NO stage2_suppress: this must stay
                                # visible to the audit, not silently skip its
                                # count-gate (see method docstring).
                                'metadata': {
                                    'task_id': str(self.task_id),
                                    'source': 'orchestrator_completion',
                                },
                            },
                        },
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.warning(f'Failed to write completion to memory: {e}')

    async def _write_decisions_to_memory(self) -> None:
        """Write plan design decisions to fused-memory."""
        decisions = self.plan.get('design_decisions', [])
        if not decisions:
            return
        if self.mcp is None:
            return
        try:
            async with __import__('httpx').AsyncClient() as client:
                for decision in decisions:
                    await client.post(
                        f'{self.mcp.url}/mcp/',
                        json={
                            'jsonrpc': '2.0',
                            'id': 1,
                            'method': 'tools/call',
                            'params': {
                                'name': 'add_memory',
                                'arguments': {
                                    'content': f"Decision: {decision['decision']}\nRationale: {decision['rationale']}",
                                    'category': 'decisions_and_rationale',
                                    'project_id': self.config.fused_memory.project_id,
                                    'agent_id': f'orchestrator-task-{self.task_id}',
                                },
                            },
                        },
                        timeout=10,
                    )
        except Exception as e:
            logger.warning(f'Failed to write decisions to memory: {e}')

    async def _write_suggestions_to_memory(self, reviews) -> None:
        """Write review suggestions (non-blocking) to memory as conventions."""
        suggestions = reviews.suggestions
        if not suggestions:
            return
        if self.mcp is None:
            return
        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                for suggestion in suggestions[:5]:  # cap at 5 to avoid noise
                    await client.post(
                        f'{self.mcp.url}/mcp/',
                        json={
                            'jsonrpc': '2.0',
                            'id': 1,
                            'method': 'tools/call',
                            'params': {
                                'name': 'add_memory',
                                'arguments': {
                                    'content': f"[{suggestion.get('category', '')}] {suggestion.get('description', '')}",
                                    'category': 'preferences_and_norms',
                                    'project_id': self.config.fused_memory.project_id,
                                    'agent_id': f'orchestrator-task-{self.task_id}',
                                },
                            },
                        },
                        timeout=10,
                    )
        except Exception as e:
            logger.warning(f'Failed to write suggestions to memory: {e}')

    async def _post_submit_tasks(self, arguments_list: list[dict]) -> None:
        """Fire-and-forget: POST all submit_task calls to the fused-memory MCP.

        Uses a single shared ``httpx.AsyncClient`` for the entire batch so only
        one TCP connection pool is opened per routing call regardless of how many
        suggestions are being submitted.  Runs inside ``asyncio.create_task`` so
        the caller returns immediately.

        Per-POST exceptions are caught and logged as warnings; a failure on one
        suggestion does not abort the remaining submissions.
        """
        if self.mcp is None:
            return
        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                for arguments in arguments_list:
                    try:
                        await client.post(
                            f'{self.mcp.url}/mcp/',
                            json={
                                'jsonrpc': '2.0',
                                'id': 1,
                                'method': 'tools/call',
                                'params': {
                                    'name': 'submit_task',
                                    'arguments': arguments,
                                },
                            },
                            timeout=10,
                        )
                    except Exception as exc:
                        logger.warning(
                            'Task %s: failed to submit curator task (fire-and-forget): %s',
                            self.task_id, exc,
                        )
        except Exception as exc:
            logger.warning(
                'Task %s: failed to open HTTP client for curator submits: %s',
                self.task_id, exc,
            )

    async def _route_review_suggestions_to_curator(self, reviews) -> None:
        """Route review suggestions directly to the curator intake (fire-and-forget).

        Inserts suggestions as CandidateTask tickets via ``submit_task`` MCP calls.
        The curator's ``_curator_worker`` drains tickets asynchronously; this method
        returns immediately after scheduling regardless of curator speed.

        Cross-submission dedup is handled by the curator R4 gate
        ``_check_escalation_idempotency`` which matches
        ``(escalation_id, suggestion_hash)`` metadata against existing
        non-cancelled tasks.  Re-submitting identical suggestions produces the
        same content_hash → same metadata → ticket marked 'combined' → 0 new rows.

        Must NOT call curate_batch — that dispatches invoke_with_cap_retry and
        can stall up to 31 minutes.  Uses asyncio.create_task +
        ``self._background_tasks`` (mirrors ``_spawn_dry_run_unblock``).

        When ``self.mcp`` is not configured (e.g. CLI / dry-run / test contexts),
        falls back to ``_escalate_suggestions`` if an escalation queue is available,
        or to ``_write_suggestions_to_memory`` otherwise. Note that
        ``_write_suggestions_to_memory`` itself requires ``self.mcp`` to write, so the
        ``(no mcp, no queue)`` case is a documented no-op — a WARNING is logged so the
        drop is auditable, but the suggestions are not persisted to any sink.
        """
        from orchestrator.review_suggestions.dedup import review_suggestion_payload_hash

        suggestions = reviews.suggestions
        if not suggestions:
            return

        # In-task dedup: compute the hash first so the cache check sits above
        # ALL routing branches (curator submit, escalation-queue fallback, no-op).
        # The escalation→resume cycle can re-enter REVIEW→DONE with an identical
        # suggestion set; short-circuiting here avoids N HTTP round-trips that
        # the curator R4 gate (_check_escalation_idempotency) would otherwise
        # have to absorb on the server side.
        content_hash = review_suggestion_payload_hash(suggestions)
        if self._last_routed_suggestion_hash == content_hash:
            logger.info(
                'Task %s: skipping duplicate curator route (hash=%s already sent)',
                self.task_id, content_hash,
            )
            return

        # Guard: fall back if MCP transport is unavailable so suggestions are
        # never silently dropped.  _escalate_suggestions is the real caller of
        # the steward-escalation fallback path.
        if self.mcp is None:
            if self.escalation_queue:
                self._escalate_suggestions(reviews)
                # Record the hash after a successful fallback so identical
                # re-entries skip the escalation queue entirely.  The queue's
                # own dedup would also catch it, but the in-task fast-path
                # avoids touching the queue at all.
                self._last_routed_suggestion_hash = content_hash
            else:
                logger.warning(
                    'Task %s: dropping %d review suggestion(s) — no MCP transport and no '
                    'escalation queue configured (CLI/dry-run/test no-op)',
                    self.task_id, len(suggestions),
                )
                # NOTE: do NOT set _last_routed_suggestion_hash here.
                # Suggestions were dropped, not routed.  The WARNING must fire
                # on every drop call to preserve audit visibility of this
                # pathological branch.
            return
        task_id = self.task_id
        project_root = str(self.config.project_root)

        all_arguments = []
        for suggestion in suggestions:
            cat = suggestion.get('category', '')
            loc = suggestion.get('location', '')
            desc = suggestion.get('description', '')
            title = f'[{cat}] {loc}: {desc[:60]}'

            all_arguments.append({
                'title': title,
                'description': desc,
                'details': json.dumps(suggestion),
                'priority': 'low',
                'project_root': project_root,
                'metadata': {
                    'spawned_from': task_id,
                    'spawn_context': 'review_suggestions',
                    'escalation_id': f'review-suggestions-{task_id}',
                    'suggestion_hash': content_hash,
                },
            })

        # Schedule the entire batch as one background task with a shared HTTP
        # client — avoids opening N separate connection pools for N suggestions.
        try:
            _task = asyncio.create_task(
                self._post_submit_tasks(all_arguments),
                name=f'route-suggestions-{task_id}',
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
            # Record the hash only on successful scheduling so that a
            # create_task failure does not prevent a retry with the same
            # suggestion set.  The curator R4 gate is the durable backstop
            # for duplicates that slip through on a successful retry.
            self._last_routed_suggestion_hash = content_hash
        except Exception as exc:
            logger.warning(
                'Task %s: failed to schedule curator submits: %s',
                task_id, exc,
            )

        logger.info(
            'Task %s: scheduled %d suggestion(s) for direct curator intake '
            '(hash=%s)',
            task_id, len(suggestions), content_hash,
        )

    def _escalate_suggestions(self, reviews) -> None:
        """Submit review suggestions as an info escalation for steward triage.

        Retained as the steward-escalation fallback even though the primary
        call site now routes via ``_route_review_suggestions_to_curator``.
        The dedup helpers are shared with the curator path — both sites import
        from ``orchestrator.review_suggestions.dedup``.
        """
        from escalation.models import Escalation

        from orchestrator.review_suggestions.dedup import (
            find_prior_review_suggestion,
            hash_marker,
            review_suggestion_payload_hash,
        )

        suggestions = reviews.suggestions
        if not suggestions or not self.escalation_queue:
            return

        # Content fingerprint: skip if identical suggestions already escalated.
        content_hash = review_suggestion_payload_hash(suggestions)

        existing = self.escalation_queue.get_by_task(self.task_id)
        prior = find_prior_review_suggestion(existing, content_hash)
        if prior is not None:
            logger.info(
                'Task %s: skipping duplicate review_suggestions escalation '
                '(content hash %s matches %s)',
                self.task_id, content_hash, prior.id,
            )
            return

        detail = hash_marker(content_hash) + json.dumps(suggestions)

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='info',
            category='review_suggestions',
            summary=f'{len(suggestions)} review suggestion(s) for triage',
            detail=detail,
            suggested_action='triage_suggestions',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={'escalation_id': esc.id, 'category': 'review_suggestions',
                      'severity': 'info', 'count': len(suggestions)},
            )
        logger.info(
            f'Task {self.task_id}: submitted {len(suggestions)} suggestions '
            f'for steward triage ({esc.id})'
        )

    def _escalate_review_issues(self, reviews) -> None:
        """Submit remaining review issues as a blocking escalation for the steward."""
        if not self.escalation_queue:
            return

        from escalation.models import Escalation

        detail = reviews.format_for_replan()
        n_blocking = len(reviews.blocking_issues)
        n_suggestions = len(reviews.suggestions)

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='review_issues',
            summary=(
                f'Review cycles exhausted with {n_blocking} blocking issue(s) '
                f'and {n_suggestions} suggestion(s)'
            ),
            detail=detail,
            suggested_action='fix_review_issues',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={'escalation_id': esc.id, 'category': 'review_issues',
                      'severity': 'blocking', 'n_blocking': n_blocking,
                      'n_suggestions': n_suggestions},
            )
        logger.info(
            f'Task {self.task_id}: escalated {n_blocking} review issues '
            f'to steward ({esc.id})'
        )

    async def _worktree_has_wip_commits(self) -> bool:
        """Whether this task's worktree holds work worth resuming (task 2248).

        Wraps ``git_ops.worktree_has_unsaved_work`` (commits-beyond-main ∨
        dirty-tree, fail-safe ``True``) — the single wip-derivation primitive
        shared with the steward's own ``_wip_probe`` (injected below in
        :meth:`_ensure_steward_started`), so wip is derived exactly once
        rather than guessed independently by each side. Returns ``False``
        (not the primitive's fail-safe ``True``) when there is no
        worktree/git_ops to inspect at all — an absent worktree cannot hold
        WIP worth resuming.
        """
        if self.worktree is None or self.git_ops is None:
            return False
        return await self.git_ops.worktree_has_unsaved_work(self.worktree, self.task_id)

    async def _ensure_steward_started(self) -> None:
        """Start the steward lazily on first call, if factory was provided.

        The channel wired below is read by ONE of this workflow's two steward
        waiters.  :meth:`_await_steward_completion` (reached via
        ``_mark_blocked``) consumes it; :meth:`_wait_for_resolution` (reached
        via ``run()``'s ESCALATED branch) does not — it is escalation-queue-only
        and is woken solely by the steward dismissing its L0.  Both therefore
        depend on the SAME producer invariant (task 3170):

            A steward give-up ALWAYS dismisses its own L0 before publishing an
            outcome.  No pending L0 survives a steward give-up.

        A new steward early return that publishes without dismissing is
        invisible to the ESCALATED waiter and parks the workflow forever — see
        ``TaskSteward._handle_escalation``, which owns that obligation.
        """
        if self._steward is not None:
            return
        if not self._steward_factory or not self.worktree:
            return
        # Check if there are pending level-0 escalations worth starting for
        if self.escalation_queue:
            pending = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            if not pending:
                return
        steward = self._steward_factory(self.worktree, self._config_dir)
        self._steward = steward
        # Wire the in-process StewardOutcome channel + this workflow's own
        # wip probe onto the freshly-built steward (task 2248 / W9-delta,
        # SO-1): the queue is lazily created here (not eagerly in __init__)
        # so a task that never starts a steward never allocates one.
        if self._steward_outcome_channel is None:
            self._steward_outcome_channel = asyncio.Queue()
        steward.set_outcome_channel(self._steward_outcome_channel)
        steward.set_wip_probe(self._worktree_has_wip_commits)
        await steward.start()

    async def _await_cancellable(self, awaitable, *, on_soft_cancel=None):
        """Race ``awaitable`` against ``self._cancel_event``.

        Returns the awaitable's result, or raises ``WorkflowCancelled('soft')``
        (W9-θ) if the cancel event was set first — it propagates straight to
        ``run()``'s single ``WorkflowCancelled`` catch, which folds in the
        scheduler-status decision (:meth:`_handle_soft_cancel`) via
        :meth:`_finalise_cancellation`.

        If both the awaitable and the cancel event resolve in the same
        ``asyncio.wait`` window, the awaitable's result wins — the work
        already finished, no need to soft-cancel.

        *on_soft_cancel*: optional ``Callable[[], None]`` invoked when the
        cancel event wins and the future is not yet done.  When provided it
        takes responsibility for orphan-avoidance (e.g. calling
        ``registry.detach(branch, request_id)`` which cancels the primary
        future only when the waiter count hits 0).  When ``None`` (default),
        the existing ``fut.cancel()`` behaviour is preserved — this keeps the
        group-merge / train path's blanket cancel untouched (D9).
        """
        fut = asyncio.ensure_future(awaitable)
        cancel_task = asyncio.create_task(self._cancel_event.wait())
        cancel_won = False
        try:
            done, _pending = await asyncio.wait(
                {fut, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if fut in done:
                return fut.result()
            cancel_won = True
            raise WorkflowCancelled('soft')
        finally:
            if not cancel_task.done():
                cancel_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_task
            # If the cancel event won, propagate that cancellation to the
            # underlying awaitable.  Without this, an enqueued merge request
            # becomes "orphaned": the worker still processes it and may halt
            # the queue with no workflow left to create the owning escalation.
            # See merge_queue.py: workers check req.result.cancelled() at entry
            # and before each halt_for_wip site.
            if cancel_won and not fut.done():
                if on_soft_cancel is not None:
                    on_soft_cancel()
                else:
                    fut.cancel()

    async def _handle_soft_cancel(self, phase: str) -> WorkflowOutcome:
        """Decide an outcome after ``_cancel_event`` interrupted a long wait.

        Three-way decision based on scheduler status and cancel-event state:

        1. ``status in TERMINAL_STATUSES`` → ``DONE``
           A human marked the task done out-of-band; exit cleanly.

        2. ``self._cancel_event.is_set()`` (pending soft-cancel, non-terminal)
           → ``SOFT_CANCELLED``
           The slot exits immediately; the harness clears the slot just like
           ``CANCELLED`` (hard-cancel).  The ``release_workflow`` MCP tool then
           parks the task as ``blocked``.

        3. Otherwise (cancel event cleared or spurious wakeup) → ``REQUEUED``
           Defensive fallback: re-run the slot once the cancel condition clears.
           Preserves the original REQUEUED semantics for non-soft-cancel callers.

        **Watcher-race note** — ``_scan_for_terminal_active_tasks`` fires a soft-
        cancel when it observes a terminal status, but by the time this method
        re-reads status the task may have transitioned back to a live state
        (terminal → non-terminal race).  In that window the read above returns
        non-terminal, ``_cancel_event`` is set, and we return ``SOFT_CANCELLED``
        (slot exits with ``requeued=False``) rather than the prior ``REQUEUED``
        (which would have re-dispatched the now-live task).  Recovery for watcher-
        triggered soft-cancels therefore relies on the scheduler's stranded-in-
        progress sweep or normal re-dispatch of a ``pending`` task rather than
        immediate requeue.  The ``release_workflow`` MCP path (human-initiated
        takeover) is unaffected — ``release_workflow`` always follows a
        ``SOFT_CANCELLED`` exit with an explicit ``set_task_status`` park.
        """
        try:
            status = await self.scheduler.get_status(self.task_id)
        except Exception:
            logger.exception(
                f'Task {self.task_id}: get_status failed during soft-cancel'
            )
            status = None
        logger.info(
            f'Task {self.task_id}: soft-cancel during {phase} — '
            f'scheduler status={status!r}'
        )
        if status in TERMINAL_STATUSES:
            return WorkflowOutcome.DONE
        if self._cancel_event.is_set():
            return WorkflowOutcome.SOFT_CANCELLED
        return WorkflowOutcome.REQUEUED

    @staticmethod
    def _outcome_severity(outcome: StewardOutcome) -> int:
        """Rank a published ``StewardOutcome`` for the drain+reduce step in
        :meth:`_await_steward_completion` (task 2248 / W9-delta review fix).

        The steward handles escalations serially and can publish more than
        one outcome within a single grace window — e.g. resolve the
        workflow's own L0 (``StewardResolved``), then chain a follow-on L0
        of its own and give up on it (``StewardReescalatedL1``).  A single
        ``channel.get()`` would only ever see the leading (least severe)
        outcome.  Higher wins: an L1 hand-off outranks budget exhaustion,
        which outranks a no-wip interruption, which outranks a wip-present
        interruption (already the "softest" give-up — task-2060 resume-plan),
        which outranks a clean resolution.  ``StewardTerminalDecision`` is
        never published on the channel (it is synthesized from a scheduler
        status read, not ranked here) so it has no case.
        """
        if isinstance(outcome, StewardReescalatedL1):
            return 4
        if isinstance(outcome, StewardBudgetExhausted):
            return 3
        if isinstance(outcome, StewardInterrupted):
            return 2 if not outcome.wip_commits_present else 1
        return 0  # StewardResolved

    @staticmethod
    def _outcome_requeues(outcome: StewardOutcome) -> bool:
        """True for the ``StewardOutcome`` variants that lead ``_mark_blocked``
        to re-pend the task: a clean ``StewardResolved``, or a
        ``StewardInterrupted`` with WIP present (the task-2060 resume-plan
        branch).  These are exactly the outcomes the ``has_open_l1``
        source-of-truth override in :meth:`_await_steward_completion` must
        gate — a requeue must never race past an L1 that is already open.
        """
        if isinstance(outcome, StewardResolved):
            return True
        return isinstance(outcome, StewardInterrupted) and outcome.wip_commits_present

    def _steward_work_outstanding(self) -> bool:
        """Whether there is anything for :meth:`_await_steward_completion` to
        wait ON — its ``skip_if_idle`` gate, used by the post-merge success
        tail (task 3170).

        Two things count as outstanding: a pending level-0 escalation (the
        steward does not un-pend a record while it works on one, so "pending"
        covers both queued and in-flight), or an unread outcome already sitting
        on the channel (instantly consumable).

        With neither, the success tail would otherwise block for the FULL
        ``steward_completion_timeout`` — 900s on stock config — waiting for a
        publish that is never coming, at the finish line of every task that had
        an escalation.  That was previously masked: the outcome the steward
        published during the ESCALATED cycle stayed on the channel and made the
        call return instantly.  :meth:`_wait_for_resolution` now drains it (it
        must — replaying it into a later ``_mark_blocked`` is a spurious
        requeue), so the "nothing to wait for" case has to be stated
        explicitly rather than ridden on a leftover.

        The narrow window where the steward's AGENT has already resolved the
        L0 via MCP but the invocation has not yet returned to publish is NOT
        covered — and was not covered before either, since the stale outcome
        short-circuited this same call.  The success tail's contract is
        "give queued steward work a chance to finish", not "join the steward".
        """
        if self.escalation_queue is not None and self.escalation_queue.get_by_task(
            self.task_id, status='pending', level=0,
        ):
            return True
        channel = self._steward_outcome_channel
        return channel is not None and not channel.empty()

    def _steward_progress_counter(self) -> int | None:
        """Monotonic "the steward is still working" signal, or ``None`` when no
        signal is available (task 3170, review fix D2).

        Sums two public ``TaskSteward.metrics`` counters, because neither alone
        covers the whole of "the steward is alive":

        - ``invocations`` increments after EVERY invocation returns
          (steward.py:597, and on the pre-triage path) — one tick per completed
          invocation, including each timeout-kill retry.  That is the "one more
          invocation ceiling legitimately consumed" event
          :meth:`_wait_for_resolution` needs in order to extend its idle window
          without hard-coding the ``steward_max_attempts +
          steward_max_timeouts_per_escalation`` multiplier.
        - ``subprocess_attempts`` increments at the START of every agent
          subprocess attempt (``TaskSteward._invoke_agent_counted``, task 3170
          review fix D4).  This is load-bearing, not redundant:
          ``_invoke_with_session`` delegates to ``invoke_with_cap_retry``, which
          runs up to ``_MAX_CAP_RETRIES`` (16) subprocess attempts behind ONE
          return, sleeping a cap cooldown (<= 300s, cli_invoke.py:1295/1367)
          between them while it waits for a usage-cap reset.  ``invocations``
          cannot move during that window, so an all-accounts-capped steward — a
          routine, designed-for condition — would read as SILENT for as long as
          the retry loop is patient, and the waiter would kill a healthy steward
          mid-work.  Ticking per ATTEMPT caps the longest legitimate silence at
          one cooldown plus one enforced ``timeouts.steward`` ceiling, which
          fits inside the waiter's window.

        Both counters are monotonically non-decreasing, so their sum is too —
        which is all the caller's ``progress > last_progress`` comparison
        requires.  The absolute value is meaningless; only its movement is read.

        Read through a defensive ``getattr`` chain: no steward, no ``metrics``,
        or NEITHER counter present as an ``int`` all mean "no progress signal
        available" and return ``None``, which degrades the caller to a plain
        fixed deadline rather than breaking it.  A counter that is present but
        not a usable ``int`` contributes 0 rather than poisoning the other
        (older fakes and hand-rolled test doubles expose only ``invocations``,
        and must keep working).  ``bool`` is excluded explicitly — it is an
        ``int`` subclass, and a truthy flag masquerading as a counter would read
        as progress exactly once and never again.
        """
        steward = self._steward
        if steward is None:
            return None
        metrics = getattr(steward, 'metrics', None)
        if metrics is None:
            return None
        total: int | None = None
        for name in ('invocations', 'subprocess_attempts'):
            value = getattr(metrics, name, None)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            total = value if total is None else total + value
        return total

    def _drain_steward_outcomes(self) -> StewardOutcome | None:
        """Non-blockingly empty the steward outcome channel and LOG what was
        there, returning the most severe drained outcome (task 3170).

        This exists for :meth:`_wait_for_resolution` — the run()-ESCALATED
        waiter, which is escalation-queue-only and never consumes the channel.
        Anything the steward published while that waiter held the wait would
        otherwise sit on the channel indefinitely, and a LATER
        ``_mark_blocked`` in the same workflow would pop it out of
        :meth:`_await_steward_completion` as though it had just been
        published.  A stale ``StewardInterrupted(wip_commits_present=True)`` is
        exactly the task-2060 resume-plan outcome, so the consequence is a
        spurious ``_requeue()`` driven by an already-dispositioned escalation
        cycle.  (The hazard pre-dates task 3170 — ``StewardResolved`` is
        published on every steward success — but fix A makes it reachable far
        more often.)

        **Drain-and-log ONLY — deliberately does NOT route on the drained
        value.**  Every actionable variant is already fully determined by the
        escalation-queue state ``_wait_for_resolution``'s tail reads:
        ``StewardReescalatedL1``/``StewardBudgetExhausted`` ⇒ the L0 is
        dismissed and an L1 is open ⇒ ``has_open_l1``; ``StewardResolved`` ⇒
        the L0 is resolved; ``StewardInterrupted`` ⇒ the L0 is dismissed by the
        producer-side give-up contract with no L1.  Routing here would add a
        SECOND parallel outcome contract on a path that has none — precisely
        the asymmetry this task exists to remove.  The return value is for
        callers/tests that want to observe what was drained, not to branch on.

        Reduction reuses :meth:`_outcome_severity` rather than adding a
        parallel ranking.  A ``None`` channel (no steward was ever started for
        this workflow) is a no-op returning ``None``.
        """
        channel = self._steward_outcome_channel
        if channel is None:
            return None
        drained: list[StewardOutcome] = []
        while True:
            try:
                drained.append(channel.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not drained:
            return None
        most_severe = max(drained, key=self._outcome_severity)
        logger.info(
            'Task %s: drained %d unconsumed steward outcome(s) after the '
            'ESCALATED wait (most severe: %r) — the escalation-queue state '
            'already determined this cycle\'s disposition; draining only '
            'prevents a later _mark_blocked from replaying them',
            self.task_id, len(drained), most_severe,
        )
        return most_severe

    async def _await_steward_completion(
        self, *, skip_if_idle: bool = False,
    ) -> StewardOutcome:
        """Wait for the steward to publish an outcome, with a grace period
        (task 2248 / W9-delta, SO-1).

        *skip_if_idle* (task 3170) is passed by the post-merge success tail,
        whose contract is "give queued steward work a chance to finish" rather
        than "join the steward": with nothing outstanding
        (:meth:`_steward_work_outstanding`) it returns the same synthesized
        default as the no-channel case instead of burning the whole grace
        window on a publish that is never coming.  ``_mark_blocked`` never
        passes it — there the wait IS the point, and an L0 it just filed is
        outstanding by construction.

        Races ``self._steward_outcome_channel.get()`` against the soft-cancel
        event and the configured grace deadline — replaces the old
        escalation-queue file-polling loop entirely (no more re-reading
        ``_pending_l0()``; the channel is the sole synchronization signal).

        This is ONE of the workflow's two steward waiters, and both rest on the
        same producer invariant (task 3170): *a steward give-up ALWAYS dismisses
        its own L0 before publishing an outcome*.  Its sibling
        :meth:`_wait_for_resolution` (``run()``'s ESCALATED branch) never reads
        this channel — the dismissal is the only thing that wakes it — so a new
        steward early return that publishes without dismissing would satisfy
        this waiter while stranding that one.  The dismissal loops on this path
        (``_mark_blocked``'s, and the ``has_open_l1`` override's below) are
        idempotent backstops over an already-dismissed record, not the primary
        mechanism.

        The two waiters share that PRODUCER invariant but deliberately DIVERGE
        on their bound — do not "re-converge" them onto one knob, which is
        exactly the defect review fix D1 removed.  This one is a
        post-completion drain grace (``steward_completion_timeout`` alone,
        which is precisely what config.py:211-222 defines that knob as, and
        the right scale for "the task is finished, let the steward drain").
        :meth:`_wait_for_resolution` instead waits on a steward that is
        actively working an escalation, where the relevant scale is the
        per-invocation ceiling — so it uses an idle window of
        ``timeouts.steward + steward_completion_timeout``, refreshed on
        observable progress.  Sharing a knob gave that backstop a value a
        healthy steward routinely exceeds.

        A steward is only ever started (and the channel only ever created)
        by :meth:`_ensure_steward_started` when there is real pending work
        (see its own pending-L0 gate) — so ``self._steward_outcome_channel``
        being ``None`` here means no steward was started for this call and
        there is nothing to wait for; return a safe default without
        touching the scheduler (avoids an unnecessary status round-trip on
        every steward-free call site, e.g. the post-merge success path).

        Otherwise — whether the wait ends via a published outcome, the
        grace deadline, or a soft-cancel — a SINGLE fresh scheduler status
        read follows.  A terminal (``done``/``cancelled``) or ``deferred``
        status ALWAYS overrides, preserving the pre-W9-delta ordering where
        the terminal check preceded the L0-resolved check.  Absent an
        override, the published outcome (if any) is returned as-is;
        otherwise (nothing published) a synthesized
        ``StewardInterrupted('timeout', wip=...)`` is returned — wip/no-wip
        routing is resolved later, by ``_mark_blocked``'s single branch (the
        task-2060 fix), not here.  **Cancel-safety amendment**: when
        ``self._cancel_event`` is (or becomes) set — whether already set on
        entry, skipping the wait outright, or fired mid-wait — the
        synthesized outcome always carries ``wip_commits_present=False``,
        never the derived probe value.  A soft-cancel/preemption of this
        workflow slot must not be routed into the task-2060 resume-plan
        branch (requeue + L0 dismissal): that branch is reserved for a
        genuine steward give-up, not for this slot being told to stop.  A
        real (non-cancel) grace-timeout still derives wip normally.

        Two extra layers close the multi-outcome silent-requeue gap (review
        fix): (1) once an outcome is obtained, any further outcomes already
        sitting on the channel (the serial steward published a burst within
        this same grace window) are drained non-blockingly and reduced to
        the most severe via :meth:`_outcome_severity`.  (2) before returning
        a requeue-producing outcome (:meth:`_outcome_requeues`), the
        escalation queue's ``has_open_l1`` is re-checked as a source-of-truth
        backstop — catching the publish-timing race where a follow-on L1 was
        filed but its outcome has not yet landed on the channel — and
        ``StewardReescalatedL1`` is returned instead when one is open (and,
        when the overridden outcome was a wip-present ``StewardInterrupted``,
        the still-pending L0 it never dismissed is dismissed here too — that
        publisher deliberately skips ``_auto_escalate_to_human``, so the
        override's ``StewardReescalatedL1`` would otherwise strand it).
        """
        channel = self._steward_outcome_channel
        if channel is None:
            return StewardInterrupted('timeout', wip_commits_present=False)
        if skip_if_idle and not self._steward_work_outstanding():
            logger.info(
                'Task %s: no outstanding steward work — skipping the '
                'completion grace window', self.task_id,
            )
            return StewardInterrupted('timeout', wip_commits_present=False)

        timeout = self.config.steward_completion_timeout
        logger.info(
            f'Task {self.task_id}: waiting up to {timeout:.0f}s for steward completion'
        )

        outcome: StewardOutcome | None = None
        deadline = asyncio.get_event_loop().time() + timeout
        remaining = deadline - asyncio.get_event_loop().time()
        if not self._cancel_event.is_set() and remaining > 0:
            get_wait = asyncio.create_task(channel.get())
            cancel_wait = asyncio.create_task(self._cancel_event.wait())
            try:
                done, _pending = await asyncio.wait(
                    {get_wait, cancel_wait},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for t in (get_wait, cancel_wait):
                    if not t.done():
                        t.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await t
            if get_wait in done:
                most_severe: StewardOutcome = get_wait.result()
                # Drain any further outcomes the serial steward already
                # published in this same grace window and reduce to the
                # most severe (review fix — see class docstring above).
                while True:
                    try:
                        next_outcome: StewardOutcome = channel.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if self._outcome_severity(next_outcome) > self._outcome_severity(most_severe):
                        most_severe = next_outcome
                outcome = most_severe
            elif cancel_wait in done:
                logger.info(
                    f'Task {self.task_id}: cancel-event fired during steward grace — '
                    f'exiting completion wait'
                )
            else:
                logger.warning(
                    f'Task {self.task_id}: steward completion timed out after '
                    f'{timeout:.0f}s with no outcome published'
                )

        # Single fresh read of the store — a terminal/deferred status always
        # wins, regardless of what (if anything) the channel produced.
        status = await self.scheduler.get_status(self.task_id)
        if status in TERMINAL_STATUSES or status == 'deferred':
            return StewardTerminalDecision(new_status=TaskStatus(status))
        if outcome is not None:
            # has_open_l1 source-of-truth override (review fix): a
            # requeue-producing outcome must not race past an L1 that is
            # already open — the publish-timing race where the steward
            # filed a follow-on L1 whose outcome has not yet landed on the
            # channel.  _mark_blocked's existing StewardReescalatedL1
            # branch (unmodified) routes this to ESCALATED with no
            # duplicate L1.
            if (
                self._outcome_requeues(outcome)
                and self.escalation_queue is not None
                and self.escalation_queue.has_open_l1(self.task_id)
            ):
                open_l1 = self.escalation_queue.get_by_task(
                    self.task_id, status='pending', level=1,
                )
                if isinstance(outcome, StewardInterrupted) and outcome.wip_commits_present:
                    # Amendment (review fix — orphan_resource_leak): the
                    # wip-present StewardInterrupted publisher deliberately
                    # skips _auto_escalate_to_human (task-2060 fix — no L1
                    # for a resumable interruption), so it never dismisses
                    # the L0 it was handling.  _mark_blocked's
                    # StewardReescalatedL1 branch assumes
                    # _auto_escalate_to_human already dismissed the L0 —
                    # true only for a steward-PUBLISHED StewardReescalatedL1,
                    # not for this override-synthesized substitution.
                    # Dismiss it here, in the single choke point that made
                    # the substitution, mirroring the shared BLOCKED
                    # fall-through's dismissal loop — otherwise it strands
                    # for the orphan-L0 reaper to later promote into a
                    # duplicate L1.
                    orphan_l0 = self.escalation_queue.get_by_task(
                        self.task_id, status='pending', level=0,
                    )
                    for esc in orphan_l0:
                        self.escalation_queue.resolve(
                            esc.id,
                            'Auto-dismissed: steward interrupted with WIP '
                            'present but a concurrent L1 is already open — '
                            'deferring to that hand-off instead of resuming '
                            'the plan',
                            dismiss=True, resolved_by='auto-dismissed',
                        )
                    if orphan_l0:
                        logger.info(
                            'Task %s: dismissed %d pending L0(s) — '
                            'wip-present interruption overridden to '
                            'ESCALATED by an already-open L1',
                            self.task_id, len(orphan_l0),
                        )
                return StewardReescalatedL1(
                    esc_id=open_l1[0].id if open_l1 else '',
                )
            return outcome
        return StewardInterrupted(
            'timeout',
            wip_commits_present=(
                False if self._cancel_event.is_set()
                else await self._worktree_has_wip_commits()
            ),
        )


def build_workflow(
    *,
    assignment: TaskAssignment,
    config: OrchestratorConfig,
    git_ops: GitOps,
    scheduler: SchedulerFacade,
    briefing: _BriefingLike,
    mcp: _McpLike | None,
    escalation_queue: EscalationQueue | None = None,
    escalation_event: asyncio.Event | None = None,
    usage_gate: UsageGate | None = None,
    initial_plan: dict | None = None,
    steward_factory=None,
    merge_queue: asyncio.Queue | None = None,
    merge_worker=None,
    merge_inflight_registry: InFlightMergeRegistry | None = None,
    event_store: EventStore | None = None,
    cost_store: CostStore | None = None,
    cancel_event: asyncio.Event | None = None,
    resume_session_id: dict | None = None,
    run_id: str | None = None,
) -> TaskWorkflow:
    """Single construction point for :class:`TaskWorkflow` (PRD C2 / Invariant P2).

    Both dispatch sites — production ``harness.py`` and eval
    ``evals/runner.py`` — construct their ``TaskWorkflow`` through this
    factory rather than calling the constructor directly, so the two paths
    can never drift silently: a new MANDATORY ``TaskWorkflow.__init__``
    parameter is either threaded through here once (and thereby acquired by
    both call sites) or surfaces as a single required edit that breaks BOTH
    at once. The signature is a keyword-only, explicit pass-through mirroring
    ``TaskWorkflow.__init__`` exactly — deliberately NOT a ``**kwargs``
    forwarder, which would swallow that drift and defeat the tripwire guard
    in ``tests/test_workflow_factory.py``.
    """
    return TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=mcp,
        escalation_queue=escalation_queue,
        escalation_event=escalation_event,
        usage_gate=usage_gate,
        initial_plan=initial_plan,
        steward_factory=steward_factory,
        merge_queue=merge_queue,
        merge_worker=merge_worker,
        merge_inflight_registry=merge_inflight_registry,
        event_store=event_store,
        cost_store=cost_store,
        cancel_event=cancel_event,
        resume_session_id=resume_session_id,
        run_id=run_id,
    )
