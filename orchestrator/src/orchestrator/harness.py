"""Top-level orchestration loop."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import itertools
import json
import logging
import os
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, TypeGuard

from shared.cli_invoke import (
    AllAccountsCappedException,
    invoke_with_cap_retry,
    transcript_exists,
)
from shared.cost_store import CostStore
from shared.mcp_envelope import resolver_failed
from shared.task_metadata import RoutingState

from orchestrator import digest as digest_mod
from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.skill_prompt import load_skill_system_prompt
from orchestrator.artifacts import TaskArtifacts
from orchestrator.background_service import (
    DEFAULT_BACKOFF_SECS,
    BackgroundService,
    BackoffPolicy,
    LifecycleRegistry,
    ManagedService,
)
from orchestrator.config import (
    OrchestratorConfig,
    apply_reload,
    config_unknown_keys_signature,
    load_config,
)
from orchestrator.delivered_checks import (
    DeliveredChecksVerdict,
    verify_delivered_checks_on_main,
)
from orchestrator.deploy_state import DeployPhase, DeployState
from orchestrator.deterministic_runner import (
    DETERMINISTIC_AGENT_ROLE,
    DeterministicRunner,
    build_milestone_gate_escalation_fields,
)
from orchestrator.event_store import EventStore, EventType
from orchestrator.fleet_heartbeat import build_heartbeat_payload, resolve_fleet_dir, write_heartbeat
from orchestrator.git_ops import GitOps, classify_worktree_entry
from orchestrator.landed_outbox import MergeProvenance
from orchestrator.landing_evidence import (
    LandingEvidenceVerdict,
    file_unattributed_landing_escalation,
    validate_landing_evidence,
)
from orchestrator.lane_lifecycle import LaneRecord
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.mcp_lifecycle import McpLifecycle
from orchestrator.merge_queue import reconcile_landed_outbox, reconcile_landed_task
from orchestrator.merge_queue_store import MergeQueueStore, recover_pending_merges
from orchestrator.merge_skew_tripwire import emit_pipeline_landing_tripwire
from orchestrator.module_charter import sanitize_files_for_persist
from orchestrator.module_tagger_prompt import (
    TAGGER_SCHEMA,
    TAGGER_SYSTEM_PROMPT,
    build_tagger_prompt,
)
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.overrides import OverrideStore
from orchestrator.park_eviction_requests import ParkEvictionRequestStore
from orchestrator.proc_supervision import EscalationSpec
from orchestrator.provenance_conflict import ProvenanceConflictSink
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.routing import RoleDefaults
from orchestrator.routing_dispatch import resolve_and_record_route
from orchestrator.run_store import RunStore
from orchestrator.scheduler import (
    Scheduler,
    SchedulerCallbacks,
    SetTaskStatusRejected,
    StaleEvidenceRejection,
)
from orchestrator.service_restart import (
    FLEET_DEPLOY_CLOCK_RELPATH,
    StaleServiceRestartCoordinator,
    schedule_detached_systemd_restart,
)
from orchestrator.stranded_verified_green import detect_verified_green
from orchestrator.systemd_inspect import (
    _INSPECT_TIMEOUT_SECS,
    _deterministic_deploy_health_verdict,
    inspect_systemd_unit,
)
from orchestrator.task_ground_truth import (
    BranchStateKind,
    ClaimantSource,
    EscalationRef,
    RecoveryAction,
    TaskGroundTruth,
)
from orchestrator.task_runtime import TaskRuntimeState, build_task_runtime_snapshot
from orchestrator.task_status import (
    ACTIVE_TASK_STATUSES,
    TERMINAL_STATUSES,
    is_infra_held,
)
from orchestrator.usage_gate import UsageGate
from orchestrator.workflow import TerminalReport, WorkflowOutcome, build_workflow
from orchestrator.worktree_identity import identities_match, read_worktree_title
from orchestrator.zero_progress_requeue import (
    ZeroProgressRequeueTracker,
    emit_zero_progress_requeue_alert,
)

if TYPE_CHECKING:
    from escalation.models import Escalation

    from orchestrator.merge_queue import (
        BreakerTrip,
        SpeculativeMergeWorker,
        TrainCallbackFactory,
    )
    from orchestrator.warm_lane_pool import WarmLanePoolCensus

try:
    from escalation.queue import EscalationQueue
    from escalation.server import create_server
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

try:
    from orchestrator.steward import TaskSteward
    HAS_STEWARD = True
except ImportError:
    HAS_STEWARD = False

logger = logging.getLogger(__name__)

# Maximum number of same-signature blocked→pending re-pends allowed before the
# 4th flip is withheld and a born-at-L2 human escalation is filed.  A module
# constant so the value stays inside the declared module scope without
# touching config.py / defaults.yaml.
_REBLOCK_GUARD_THRESHOLD: int = 3


def _bumped_routing_dump(metadata: Any, by: int = 1) -> dict[str, Any]:
    """Return the ``metadata['routing']`` blob with ``routing_tier`` bumped by ``by``.

    The single pure authority for the retry-escalation tier bump (task μ,
    plans/adaptive-model-routing-prd.md Phase 4), reused by every bump site:
    the terminal-failure auto-bump (``_maybe_bump_routing_tier``) and the
    escalate_model by-id bump (``_bump_routing_tier_by_id``).

    Reads via :meth:`RoutingState.from_metadata` (tolerant-degrade: a missing/
    non-dict ``metadata`` or ``routing`` key yields a fresh ``RoutingState()``),
    increments only ``routing_tier`` via ``model_copy`` — ``latest``/``history``/
    ``simple_saturated`` and any ``extra`` fields ride through unchanged — and
    serializes with ``model_dump()``. Because it only ever *adds*, the
    harness-owned counter stays monotonic (invariant 8) even under a
    last-write-wins ``metadata_mode='merge'`` race.
    """
    state = RoutingState.from_metadata(metadata if isinstance(metadata, dict) else None)
    return state.model_copy(update={'routing_tier': state.routing_tier + by}).model_dump()

# Fixed sentinel task_id for the dirty-project-root-at-startup born-at-L2
# escalation (task 2380). Not a real task — _on_escalation only bumps a
# counter and wakes a workflow waiting on this exact task_id, and nothing
# waits on this sentinel, so dispatch is never blocked. A stable sentinel
# also gives get_by_task(..., level=2) a durable dedup key across restarts.
_DIRTY_TREE_ESCALATION_SENTINEL: str = 'dirty-project-root-startup'

# Sentinel task_id for the unknown-config-key born-at-L2 filer (task 2989),
# mirroring _DIRTY_TREE_ESCALATION_SENTINEL: a stable synthetic task_id that no
# workflow waits on, giving get_by_task(..., level=2) a durable self-heal handle
# and make_id a per-sentinel counter across restarts.
_CONFIG_UNKNOWN_KEYS_SENTINEL: str = 'config-unknown-keys-startup'

# Statuses swept by _reconcile_stranded_in_progress for stranded-task recovery.
# Intentionally EXCLUDES:
#   'done' / 'cancelled'   — terminal-by-decision; nothing to recover
#   'deferred'             — human-deferred; leave for manual resolution
#   'merge-deferred'       — train-parked (PRD § 9.8); worktree must survive
# The explicit merge-deferred early-return in _reconcile_one_stranded mirrors
# the open-L1 /unblock veto guard (harness.py _reconcile_one_stranded:~1598).
_RECONCILE_SWEEP_STATUSES: frozenset[str] = frozenset({'in-progress', 'blocked'})

# heartbeat_ttl the harness configures TaskGroundTruth (task 2243, W10-θ2)
# with — the staleness threshold TG-3's live_claimant folding applies to the
# W2 db claimant signal (shared.task_claimant.is_stranded) and the plan.lock
# freshness cross-check. No dedicated OrchestratorConfig field exists for
# this yet, so it is bound explicitly here rather than left to silently ride
# whatever default TaskGroundTruth ships with; the value mirrors
# TaskGroundTruth's own _DEFAULT_HEARTBEAT_TTL (task_ground_truth.py) and
# TaskArtifacts.clear_stale_plan_lock's hardcoded 600s default.
_RECONCILE_HEARTBEAT_TTL: timedelta = timedelta(minutes=10)

# Non-terminal parked statuses whose worktrees are inviolable — owned by a
# non-scheduler party, not by the task's own progress — and so must NEVER be
# selected as warm-lane reclaim victims (task 2018):
#   'merge-deferred'  — train-parked (PRD § 9.8); owned by the group-merge
#                        worker, which is promised an intact worktree.
#   'deferred'         — human-deferred; owned by manual resolution.
# Mirrors the merge-deferred early-return guard in _reconcile_one_stranded
# (harness.py:2437) and the _RECONCILE_SWEEP_STATUSES exclusions above.
# Deliberately EXCLUDES 'blocked': a stranded blocked task is a legitimate
# reclaim target (_reconcile_stranded_in_progress sweeps
# {'in-progress', 'blocked'} via _RECONCILE_SWEEP_STATUSES), so it must stay
# eligible here too.
_WARM_LANE_RECLAIM_PROTECTED_STATUSES: frozenset[str] = frozenset(
    {'merge-deferred', 'deferred'}
)

# Recency grace for the verified-green stranded-reaper's durable re-submit
# guard (PRD leaf α §7).  A ``metadata.stranded_merge_request`` marker whose
# tip_sha matches the current lane tip AND whose submitted_at is within this
# window short-circuits a re-submit — the merge is presumed still in-flight,
# so the periodic stranded sweep must not re-enqueue the same branch every
# tick.  Once the window elapses (or the lane tip advances past the marker),
# a fresh submit re-drives (self-healing).  Sized to comfortably cover an
# in-flight merge+verify while still re-driving a genuinely lost/hung one.
_STRANDED_MERGE_RESUBMIT_GRACE_S: float = 30 * 60.0  # 30 minutes

# Prior auto-eval redo siblings (task 2075) are only safe to silently
# supersede (cancel) when they are still idle and unclaimed by anyone.
# Deliberately just {'pending'} — EXCLUDES five of the six members of
# ACTIVE_TASK_STATUSES that _find_prior_auto_eval_redos uses to *discover*
# candidates:
#   'in-progress'    — an agent may be actively doing useful work on it; a
#                       cancel here can abort real progress and orphan its
#                       worktree/branch.
#   'blocked'        — it may be sitting on an open human-facing escalation;
#                       a cancel here would silently discard that escalation.
#   'deferred'       — WORKFLOW_PRESERVE_STATUSES (task_status.py) treats
#                       'deferred' as "leave this alone, human will sort
#                       it" — the same bucket as 'blocked' and
#                       'merge-deferred' below — because a steward may have
#                       explicitly deferred this exact redo. A redo that is
#                       'deferred' only because its own pending-flip failed
#                       is indistinguishable from that steward decision from
#                       here, so both are left alone; a later auto-eval
#                       trigger (if any) will re-discover and retry a
#                       genuinely-stuck one rather than silently discarding
#                       a deliberate hold.
#   'review'         — already past dispatch and under active verification;
#                       cancelling mid-review discards that work.
#   'merge-deferred' — train-parked (PRD § 9.8); its worktree is owned by
#                       the group-merge worker, not safe to reclaim here.
# This narrower set gates the actual cancel dispatch in _maybe_auto_eval via
# _filter_cancellable_redos, which re-checks each candidate's CURRENT status
# immediately before cancelling — closing the race window between the
# initial _find_prior_auto_eval_redos snapshot and the cancel loop (a
# candidate that completed/was cancelled on its own in the interim is simply
# excluded by that re-check and left alone).
_AUTO_EVAL_SUPERSEDE_SAFE_STATUSES: frozenset[str] = frozenset({'pending'})

# Grace period added to the watcher rotation timeout on top of
# watcher_rotation_hours*3600.  Gives the agent time to emit its digest and
# exit cleanly before the supervisor kills it with a SIGTERM timeout.
_WATCHER_TIMEOUT_GRACE_SECS: float = 300.0

# Maximum backoff between unclean watcher exits (seconds).
_WATCHER_MAX_BACKOFF_SECS: float = 3600.0

# Fixed quiescent gap after a failed background-loop pass (seconds). Shared by
# the main-tip sweep and no-landings breaker loops. Guarantees neither loop can
# tight-spin on a pass that fails immediately, even if the configured interval
# is non-numeric (task 1907). Derives from background_service.DEFAULT_BACKOFF_SECS
# (task 2241 amendment) rather than duplicating the literal, so the two
# constants cannot drift apart.
_BG_LOOP_FAILURE_BACKOFF_SECS: float = DEFAULT_BACKOFF_SECS

# LifecycleRegistry stop-bound constants (task 2241, W10-η — PRD §5.3 LR-2).
# No prior code bounded these stops at all: every existing _stop_* simply
# awaited cancellation unconditionally, which is exactly the unbounded-hang
# shape this task exists to make structurally impossible.  These are
# therefore deliberately generous new bounds, not a retuning of any existing
# knob (Open-Q Q5 parity concerns only intervals/backoff, which are carried
# verbatim into each BackgroundService registration below).
#
# _LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS bounds the seven BackgroundService
# sweeps, whose pass_fns are lightweight/cooperative and cancel almost
# instantly.  _LIFECYCLE_SERVICE_STOP_TIMEOUT_SECS bounds the four bespoke
# ManagedService adapters, whose teardown can involve more (aiohttp server
# shutdown, a worker's own internal drain, subprocess supervision) — merge
# worker's SpeculativeMergeWorker.stop() already self-bounds at 5s
# internally, so 15s leaves headroom above that.
_LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS: float = 10.0
_LIFECYCLE_SERVICE_STOP_TIMEOUT_SECS: float = 15.0

# Bounded-failure-log cap shared by every migrated BackgroundService sweep
# (task 1907's bounded-logging discipline is now a BackgroundService
# property — see background_service.py).  Three of the seven sweeps
# (orphan-l0-reaper, terminal-status-watcher, stranded-reconcile) previously
# logged via unbounded logger.exception on every failure; this is the new,
# uniform cap all seven share.
_BG_LOOP_MAX_FAILURE_LOGS: int = 5

# agent_role values eligible for Source-B re-validation (deterministic-recon-sweep,
# task 2074): the runner's own sentinel role for deploy infra_issue escalations,
# plus this sweep's own role so a previously-unconfirmed L1 it filed self-heals
# once the unit recovers.  Any other role (or category != 'infra_issue') is left
# untouched — in particular milestone_gate escalations are NEVER auto-resolved
# by Source B while the subject task remains live.  (Source C — task 2114,
# see _revalidate_open_l2 below — is a separate, later mechanism that DOES
# close a milestone_gate/design_concern/etc. L2 once its SUBJECT task goes
# terminal; that is a deliberate, category-agnostic broadening, not a
# contradiction of this guarantee.)
_DETERMINISTIC_ESCALATION_SENTINEL_ROLES: frozenset[str] = frozenset({
    'orchestrator-deterministic',
    'harness-deterministic-recon-sweep',
})

# resolved_by sentinel for the generalized escalation-revalidation sweep
# (task 2114): terminal-subject L2 closures (Source C, hosted inside
# _run_deterministic_recon_sweep) are attributed to this role in the audit
# trail, distinguishing them from a human/steward resolve.
_ESCALATION_REVALIDATION_SWEEP_ROLE: str = 'harness-escalation-revalidation-sweep'

# resolution_class stamped on a terminal-subject sweep close (task 2724). A
# DISTINCT, non-benign value (it must be a member of
# escalation.models.RESOLUTION_CLASSES) so swept records stay auditable rather
# than being mis-labelled 'benign' by the reaper-sweep resolver default. Passed
# explicitly to queue.resolve(), overriding that per-resolver default.
_ESCALATION_REVALIDATION_RESOLUTION_CLASS: str = 'moot-terminal-subject'

# resolved_by sentinel for the main-tip-sweep self-heal (task 2114): closes a
# pending orchestrator-main-sweep escalation once a later full-verify PASS
# supersedes its (now-fixed) swept SHA.
_MAIN_TIP_SWEEP_SELFHEAL_ROLE: str = 'harness-main-tip-sweep-selfheal'

# Bound on the thread-off load_config() call inside reload_config() (PRD
# plans/config-hot-reload-prd.md Open Q3). load_config() runs
# _discover_module_configs, a filesystem walk that can wedge; this keeps a
# hot-reload request from hanging indefinitely. The abandoned worker thread
# cannot be force-cancelled and is left to finish; its result is discarded.
_RELOAD_LOAD_TIMEOUT_SECS: float = 30.0

# Idle-while-paused tuning (task 1322 follow-up).  When the scheduler is paused
# and no tasks are active, the main run loop idles in-process instead of exiting
# 0 (which defeats Restart=on-failure and wedges the factory).
# _PAUSED_IDLE_POLL_SECS matches the existing 15s acquire_next wait so a resume
# is picked up within one poll; _PAUSED_IDLE_LOG_SECS throttles the
# "alive but paused" WARNING so a multi-hour pause doesn't flood the log.
_PAUSED_IDLE_POLL_SECS: float = 15.0
_PAUSED_IDLE_LOG_SECS: float = 180.0

# Run-forever idle (default; --until-idle opts out): when the queue drains and
# the scheduler is NOT paused, the loop polls for newly-scheduled tasks rather
# than exiting.  The poll cadence is configurable (config.idle_poll_secs);
# _IDLE_POLL_LOG_SECS throttles the "Idle — polling" INFO so a long dead-idle
# stretch doesn't flood journald (mirrors _PAUSED_IDLE_LOG_SECS).
_IDLE_POLL_LOG_SECS: float = 180.0

# Allowed and disallowed tools for the escalation-watcher-auto rotation.
# Scoped to what escalation triage actually needs: file reads, foreground
# escalation.watcher subprocess, safe git reads, and the MCP tools for
# autonomous dispatch (update_task, add_dependency, resolve_issue).
# Defence-in-depth mirrors the unblock-auto precedent (dry_run_unblock.py).
#
# NOTE — this allowlist is ADVISORY, not the durable enforcement boundary.
# The watcher subprocess runs under `--permission-mode bypassPermissions`
# (task 1326), so a hard boundary cannot live in a client-side tool
# allowlist. The durable enforcement of the watcher's escalation scope is
# now SERVER-SIDE: the escalation MCP server reads the connection-capability
# headers in _WATCHER_ESCALATION_HEADERS (X-Escalation-Levels/-Identity) and
# denies (level_forbidden) any resolve_issue/promote_to_l2 outside the
# granted level set, regardless of which tools the client attempted to call.
# See plans/escalation-connection-capability-guard-prd.md task alpha (2041,
# server-side enforcement) and task beta (2042, this header wiring).
_WATCHER_ALLOWED_TOOLS: list[str] = [
    'Read',
    'Glob',
    'Grep',
    # Foreground-blocking watcher subprocess (see SKILL.md wait-for-next-L1 step)
    'Bash(python -m escalation.watcher:*)',
    # Git reads for context
    'Bash(git log:*)',
    'Bash(git diff:*)',
    'Bash(git status:*)',
    'Bash(git show:*)',
    'Bash(git rev-parse:*)',
    'Bash(git branch:*)',
    'Bash(git ls-files:*)',
    # Subagent delegation for deep RCA on hard/investigation-class items
    # (task 2629): the sonnet top-level rotation spawns an opus subagent via
    # the Task tool for read-only deep-dive investigation feeding
    # promote_to_l2 (see SKILL.md "Delegating deep RCA to an opus subagent").
    # The spawned subagent inherits this same allowed/disallowed-tools
    # posture, so it cannot edit code or touch main regardless of its prompt.
    # Covered by TestWatcherAllowedTools.test_task_in_allowed_tools — keep in
    # lockstep with that test if this entry ever moves or is removed.
    'Task',
    # Escalation MCP: read + autonomous resolve + L1→L2 promotion
    # (promote_to_l2 is needed by the consumer-per-level contract so the
    # watcher can escalate out-of-scope L1s directly to a human L2 stream)
    'mcp__escalation__get_pending_escalations',
    'mcp__escalation__resolve_issue',
    'mcp__escalation__promote_to_l2',
    # Triage-ack annotation, ungated by level — lets the watcher stamp a
    # pending L1/L2 it assessed (task 2555) so future rotations can skip
    # re-deriving the same disposition every rotation.
    'mcp__escalation__stamp_triage',
    # Fused-memory MCP: read + autonomous dispatch (scope_violation/dependency/cleanup)
    'mcp__fused-memory__get_task',
    'mcp__fused-memory__get_tasks',
    'mcp__fused-memory__search',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__add_dependency',
]
# Connection-capability headers for the escalation-watcher-auto rotation's
# escalation MCP connection (plans/escalation-connection-capability-guard-prd.md
# task beta, consuming the server-side enforcement landed by task alpha/2041).
# X-Escalation-Levels='0,1' scopes resolve_issue/promote_to_l2 to L0/L1 only —
# the server denies (level_forbidden) any attempt to resolve an L2 directly.
# X-Escalation-Identity pins the server-stamped resolved_by so a watcher
# can't spoof attribution via a tool argument.
#
# CONTRACT: must stay in lockstep with escalation/src/escalation/server.py —
# header names match its _LEVELS_HEADER/_IDENTITY_HEADER (HTTP header names
# are case-insensitive on the wire, so title-case here is fine); the levels
# value must parse under its _parse_levels() (comma-separated non-negative
# ints, fail-closed on anything else). No test in this repo exercises this
# constant against the live server parser — escalation/tests/
# test_capability_guard_http.py covers the server side only. If either
# contract changes, update both sides together.
_WATCHER_ESCALATION_HEADERS: dict[str, str] = {
    'X-Escalation-Levels': '0,1',
    'X-Escalation-Identity': 'orchestrator-escalation-watcher-auto',
}
# Mutating tools blocked — no code edits, no destructive git, no infra commands.
_WATCHER_DISALLOWED_TOOLS: list[str] = [
    'Edit',
    'Write',
    'Bash(git commit:*)',
    'Bash(git push:*)',
    'Bash(git reset:*)',
    'Bash(git checkout:*)',
    'Bash(git restore:*)',
    'Bash(git clean:*)',
    'Bash(git merge:*)',
    'Bash(git rebase:*)',
]


def _pid_alive(pid: int) -> bool:
    """Return True if the process identified by *pid* is alive.

    Mirrors the semantics of
    fused-memory/src/fused_memory/services/orchestrator_detector.py:58-72
    without introducing a cross-package import edge.

    - Returns False for pid <= 0 (invalid).
    - Uses os.kill(pid, 0): success → alive; ProcessLookupError → dead;
      PermissionError → alive (we can see it but lack permission to signal it);
      other OSError → treat as dead.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _is_valid_sha_40(s: object) -> TypeGuard[str]:
    """Return True iff *s* is a well-formed 40-char lowercase hex SHA.

    Used to validate ``branch_base_sha`` values read from task metadata
    before comparing them against live git output.  Any non-conforming
    value is treated as missing so the reconciler falls through to the
    existing citation-grep guard rather than making a bogus comparison.
    """
    return (
        isinstance(s, str)
        and len(s) == 40
        and all(c in '0123456789abcdef' for c in s)
    )


# _deterministic_deploy_health_verdict is now defined in systemd_inspect.py
# (task 2119) and imported above verbatim; re-bound to this name so existing
# `from orchestrator.harness import _deterministic_deploy_health_verdict`
# call sites (incl. tests) keep working unchanged.


async def _recon_inspect_unit(unit: str) -> dict:
    """Query systemctl for unit state fields needed for the recon-sweep health verdict.

    Task 2119: thin delegate to the hoisted, hardened
    ``systemd_inspect.inspect_systemd_unit`` (previously a standalone
    duplicate of ``DeterministicRunner._default_inspect_unit``'s
    ``systemctl --user show`` pattern — task 2091's timeout/kill/reap
    hardening now lives there exactly once, so this sweep gets it too).
    Uses the module's default timeout (no DeterministicRunner instance is
    available here). Returns a dict with at least MainPID (int) and
    ActiveState (str); MainPID defaults to 0 on parse failure.
    """
    return await inspect_systemd_unit(unit, timeout_secs=_INSPECT_TIMEOUT_SECS)


def _deterministic_deploy_stranded(metadata: dict | None) -> bool:
    """Return True iff *metadata* represents a stranded deterministic deploy.

    ζ/task 2240 (DS-4): replaces the deleted 4-stamp combinatorial
    ``_is_stranded_deterministic_shape`` predicate — because ζ writes
    ``deploy_state.phase`` atomically with every stamp, the old
    stamp-combination archaeology collapses to a single enum compare.

    Metadata-only predicate (the empty-pending-escalation-queue check is I/O
    and is performed by the caller).  True iff ALL of:
      - ``task_kind == 'deterministic'``
      - ``before_done`` is a dict with a truthy ``target_unit``
      - AND either:
          - ``deploy_state`` is present and its ``phase`` is
            ``DeployPhase.RAN`` (the deploy ran but reached no terminal
            phase), or
          - ``deploy_state`` is ABSENT and ``before_done_ran_at`` is truthy
            AND none of ``before_done_verified_at`` / ``gate_escalated_at``
            / ``done_provenance`` is set — a bounded, documented migration
            shim for a deploy that began before ζ activated (no
            deploy_state was ever written), so it isn't silently
            un-stranded the moment ζ ships. The three exclusions mirror the
            deleted ``_is_stranded_deterministic_shape``'s terminal-outcome
            check (reviewer amendment, task 2240): without them, a legacy
            deploy that already reached a terminal state — e.g. an
            act-then-ask deploy blocked at its gate whose escalation was
            just resolved but not yet re-dispatched to done — would be
            misclassified as a RAN-strand.

    None/non-dict *metadata* and a non-dict ``before_done`` are treated as
    non-matching rather than raising.
    """
    if not isinstance(metadata, dict):
        return False
    if metadata.get('task_kind') != 'deterministic':
        return False
    before_done = metadata.get('before_done')
    if not isinstance(before_done, dict) or not before_done.get('target_unit'):
        return False
    if 'deploy_state' in metadata:
        state = DeployState.from_metadata(metadata)
        return state is not None and state.phase == DeployPhase.RAN
    return bool(metadata.get('before_done_ran_at')) and not (
        metadata.get('before_done_verified_at')
        or metadata.get('gate_escalated_at')
        or metadata.get('done_provenance')
    )


def _deterministic_gate_stranded(metadata: dict | None) -> bool:
    """Return True iff *metadata* represents a stranded deterministic GATE.

    Task 2954: the sibling of ``_deterministic_deploy_stranded`` for pure-gate
    / ``always_escalates=true`` strands — a ``task_kind=='deterministic'`` task
    that stamped ``gate_escalated_at`` (proof a born-at-L2 ``milestone_gate``
    was supposed to be filed) but whose escalation record never landed or was
    lost across a restart.  ``gate_escalated_at`` is written ONLY by the
    gate-filing paths (``DeterministicRunner._file_milestone_gate_and_block`` /
    ``_file_milestone_check_failed_and_block`` / the predicate leg), so it is
    the sole reliable "a gate was supposed to be filed" signal.

    Metadata-only predicate — the archive-inclusive empty-escalation-queue
    I/O check (the strand-vs-resolved discriminator) is performed by the
    caller (``_run_deterministic_recon_sweep`` Source A), mirroring
    ``_deterministic_deploy_stranded``'s "the I/O check is the caller's job"
    contract.

    DISJOINT from ``_deterministic_deploy_stranded`` by construction — this
    one REQUIRES ``gate_escalated_at`` whereas that predicate never matches a
    task once it is set, though for DIFFERENT reasons in its two branches:
      - ``deploy_state`` PRESENT: it matches ONLY ``deploy_state.phase == RAN``
        and never inspects ``gate_escalated_at`` directly.  Disjointness rests
        on the atomic stamp+phase-advance invariant, not an explicit exclusion:
        stamping ``gate_escalated_at`` on a deploy is done ATOMICALLY with
        advancing ``phase`` to ``ESCALATED`` (the runner's single
        ``_advance_deploy_phase`` merge in ``_file_milestone_gate_and_block``),
        so a deploy with ``gate_escalated_at`` set is ``phase == ESCALATED``,
        never ``RAN`` — the deploy predicate returns False.
      - ``deploy_state`` ABSENT (pre-ζ legacy shim): THAT branch is the one
        that EXPLICITLY excludes ``gate_escalated_at`` being set.
    So no task is ever matched by both, and a gate strand and a deploy strand
    are handled by separate recovery paths with no double-handling.  (Were the
    atomic stamp+phase-advance invariant ever to regress, the sweep's
    deploy-before-gate branch ordering is a belt-and-braces backstop.)

    None/non-dict *metadata* is treated as non-matching rather than raising.
    """
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get('task_kind') == 'deterministic'
        and bool(metadata.get('gate_escalated_at'))
    )


def _is_done_step_commit_orphan(esc: Escalation) -> bool:
    """Return True iff *esc* is the done-step-commit orphan class filed by
    ``TaskWorkflow._escalate_unreconciled_done_step`` (workflow.py:5488).

    Task 2725: this is the sole, stable, machine-readable discriminator for
    the one orphan-L0 class that is a false positive when its subject task
    was requeue-rebased — the step's recorded ``commit`` SHA is a
    pre-rebase intermediate no longer reachable from main, but the step's
    content landed on main under a new SHA via the merge.
    ``suggested_action='verify_wip_reconciliation'`` is set only by that
    one filing site (grep-confirmed sole occurrence repo-wide), so matching
    on it (plus ``agent_role``/``category``) is robust to summary-wording
    changes, unlike a fragile summary-substring match.
    """
    return (
        esc.agent_role == 'orchestrator'
        and esc.category == 'infra_issue'
        and esc.suggested_action == 'verify_wip_reconciliation'
    )


# The "merged family" of done_provenance.kind values that positively confirm
# a done task's content actually landed on main (task 2725). Deliberately
# EXCLUDES:
#   'found_on_main'                 — the class this reaper's skip must NOT
#                                      mask (a done task whose content did
#                                      not land); it already has its own
#                                      dedicated landing guards (PRD
#                                      5dd39a4c42, batch 2674-2683), so this
#                                      reaper is not that safety net.
#   'dispatch-gate-already-on-main' — not in the Leo-ratified list; left out
#                                      to avoid scope creep.
#   commitless kinds (e.g. 'operational-verified', 'deterministic-deploy',
#   'deterministic-deploy-scheduled', 'deterministic-milestone') — these
#   never had a step commit to orphan in the first place.
# Used only by _is_terminal_merged to gate the orphan-L0 reaper's
# done-step-commit dismiss branch.
_MERGED_DONE_PROVENANCE_KINDS: frozenset[str] = frozenset({
    'merged',
    'dispatch-gate-marker-found',
    'dispatch-gate-content-equivalent',
})


def _is_terminal_merged(task: dict | None) -> bool:
    """Return True iff *task* is a done task whose content is confirmed merged.

    Task 2725: used by the orphan-L0 reaper to recognise a
    rebase-superseded done-step-commit orphan as benign — the step's
    recorded commit SHA is a pre-rebase intermediate no longer reachable
    from main, but its content landed on main under a new SHA via the
    merge. True iff ``task['status'] == 'done'`` AND
    ``task['metadata']['done_provenance']`` is a dict whose ``kind`` is in
    :data:`_MERGED_DONE_PROVENANCE_KINDS`.

    Fail-safe: a ``None`` *task* (``get_task`` failure or absence), a
    missing/non-dict ``metadata``, and a missing/non-dict
    ``done_provenance`` are all treated as non-matching rather than
    raising — the caller promotes (surfaces) instead of silently
    dismissing whenever terminal+merged cannot be positively confirmed.
    """
    if task is None:
        return False
    if task.get('status') != 'done':
        return False
    metadata = task.get('metadata')
    if not isinstance(metadata, dict):
        return False
    provenance = metadata.get('done_provenance')
    if not isinstance(provenance, dict):
        return False
    return provenance.get('kind') in _MERGED_DONE_PROVENANCE_KINDS


def _is_scope_divergence_orphan(esc: Escalation) -> bool:
    """Return True iff *esc* is the plan.files/metadata.files divergence
    orphan class filed by ``TaskWorkflow._escalate_scope_invariant_violation``
    (workflow.py:11277).

    Task 2931: this class recurred as a false positive post-2878
    (esc-2865-19, esc-2869-10) because the lock-free
    ``reviewer_comprehensive`` / ``resettled_adjudicator`` stages hold no
    module locks and are absent from ``_dispatched``, so
    ``Scheduler.is_actively_held`` returns False for a task that is genuinely
    live mid-dispatch (its ``metadata.files`` legitimately lagging
    ``plan.files`` during in-flight scope reconciliation). The reaper gates
    this specific class on ``routing.latest`` freshness — see
    :func:`_has_fresh_dispatch`.

    Unlike the done-step-commit class, ``suggested_action`` is NOT a unique
    discriminator here — ``'investigate_and_retry'`` is shared by several
    other filing sites (scheduler.py, workflow.py). The distinctive summary
    substring ``'plan.files/metadata.files divergence detected'`` is
    grep-confirmed unique to that one filing site, so it is the robust
    discriminator (plus ``agent_role``/``category``).
    """
    return (
        esc.agent_role == 'orchestrator'
        and esc.category == 'infra_issue'
        and 'plan.files/metadata.files divergence detected'
        in (esc.summary or '')
    )


def _has_fresh_dispatch(
    task: dict | None, now: datetime, grace_secs: float,
) -> bool:
    """Return True iff *task* has a routing decision stamped within
    *grace_secs* of *now* — i.e. a live in-flight LLM dispatch.

    Task 2931: the lock-free reviewer/adjudicator stages leave no lock or
    ``_dispatched`` trace, but they DO stamp
    ``metadata.routing.latest.decided_at`` fresh per LLM invocation
    (``RoutingDecisionMirror``, shared/task_metadata.py). The orphan-L0
    reaper reads it as the missing liveness dimension ``is_actively_held``
    lacks, to defer (not drop) the divergence class while a dispatch is
    genuinely live.

    Fail-safe (mirrors :func:`_is_terminal_merged`): a ``None`` *task*, a
    missing/non-dict ``metadata``/``routing``/``latest``, an absent/non-str
    ``decided_at``, and an unparseable or tz-mismatched timestamp are all
    treated as "not fresh" (return False) rather than raising — the caller
    then promotes (surfaces) instead of silently suppressing whenever
    liveness cannot be positively confirmed, preserving task 2878's boundary
    guard and the loud-over-silent-degradation norm. A ``decided_at`` newer
    than *now* (a dispatch that landed after the sweep snapshot) yields a
    negative delta ``< grace_secs`` -> True (fresh), matching the "newer than
    the sweep snapshot" wording.
    """
    if task is None:
        return False
    metadata = task.get('metadata')
    if not isinstance(metadata, dict):
        return False
    routing = metadata.get('routing')
    if not isinstance(routing, dict):
        return False
    latest = routing.get('latest')
    if not isinstance(latest, dict):
        return False
    decided_at = latest.get('decided_at')
    if not isinstance(decided_at, str):
        return False
    try:
        decided_dt = datetime.fromisoformat(decided_at)
        delta_secs = (now - decided_dt).total_seconds()
    except (ValueError, TypeError):
        return False
    return delta_secs < grace_secs


def _acquire_project_lock(project_root: Path) -> IO:
    """Acquire an exclusive flock on a per-project lockfile.

    Returns the open file object — caller must keep a reference to it
    (closing or GC releases the lock).  Raises ``SystemExit(1)`` if
    another orchestrator instance already holds the lock.
    """
    lock_path = project_root / 'data' / 'orchestrator' / 'orchestrator.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = open(lock_path, 'w')  # noqa: SIM115
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Another instance holds the lock — read its diagnostic info
        try:
            with open(lock_path) as f:
                info = f.read().strip()
        except OSError:
            info = '(unknown)'
        logger.error(
            'Another orchestrator is already running for this project.\n'
            f'  Lock holder: {info}\n'
            f'  Lock file:   {lock_path}\n'
            'Kill the existing instance first, or wait for it to finish.'
        )
        lock_file.close()
        raise SystemExit(1) from None

    # Write diagnostic info for anyone who tries to acquire next
    lock_file.truncate(0)
    lock_file.seek(0)
    lock_file.write(f'PID {os.getpid()} started {datetime.now(UTC).isoformat()}')
    lock_file.flush()
    return lock_file


@dataclass
class TaskReport:
    task_id: str
    title: str
    outcome: WorkflowOutcome
    cost_usd: float = 0.0
    duration_ms: int = 0
    agent_invocations: int = 0
    execute_iterations: int = 0
    verify_attempts: int = 0
    review_cycles: int = 0
    steward_cost_usd: float = 0.0
    steward_invocations: int = 0
    completed_at: str = ''
    # Block-context surfacing for the per-task retry cap.  Populated by
    # _run_slot from the TerminalReport returned by workflow.run() (TR-1)
    # when the outcome is REQUEUED (harmless/empty on DONE paths).
    #
    # Task 3068 made block_reason/block_phase DURABLE: both are now emitted on
    # the EventType.task_completed payload AND persisted to runs.db's
    # task_results (via both save_task_result and save_run).  block_detail
    # remains purely in-memory — it carries unbounded raw agent/verify output,
    # so it stays out of the two operationally-queried, rotated stores.
    block_reason: str = ''
    block_detail: str = ''
    block_phase: str = ''
    # Task 2988 (PRD ε / W3): whether a REQUEUED outcome counts against the
    # per-task requeue cap.  Mapped from ``TerminalReport.counts_against_
    # requeue_cap`` in _run_slot and passed to
    # ``scheduler.record_requeue(counts_against_cap=...)`` in _apply_retry_cap.
    # Defaults True so DONE paths and any report built without it keep the
    # pre-2988 counting behaviour.
    counts_against_requeue_cap: bool = True


@dataclass
class HarnessReport:
    started_at: str = ''
    completed_at: str = ''
    total_tasks: int = 0
    completed: int = 0
    blocked: int = 0
    escalated: int = 0
    total_cost_usd: float = 0.0
    task_reports: list[TaskReport] = field(default_factory=list)
    paused_for_cap: bool = False
    cap_pause_duration_secs: float = 0.0
    review_checkpoints: int = 0
    review_findings: int = 0
    review_tasks_created: int = 0
    review_cost_usd: float = 0.0

    def summary(self) -> str:
        lines = [
            f'Orchestrator run complete: {self.completed}/{self.total_tasks} tasks done',
            f'  Blocked: {self.blocked}',
            f'  Escalated: {self.escalated}',
            f'  Total cost: ${self.total_cost_usd:.2f}',
            f'  Duration: {self.started_at} → {self.completed_at}',
        ]
        if self.paused_for_cap:
            lines.append(f'  Cap pause: {self.cap_pause_duration_secs:.0f}s total')
        if self.review_checkpoints > 0:
            lines.append(
                f'  Review checkpoints: {self.review_checkpoints} '
                f'({self.review_findings} findings, '
                f'{self.review_tasks_created} tasks, '
                f'${self.review_cost_usd:.2f})'
            )
        lines.extend([
            '',
            'Per-task results:',
        ])
        for r in self.task_reports:
            lines.append(
                f'  {r.task_id}: {r.outcome.value} '
                f'(${r.cost_usd:.2f}, {r.agent_invocations} invocations)'
            )
        return '\n'.join(lines)


def build_train_callback_factory(scheduler: Any, git_ops: Any = None) -> TrainCallbackFactory:
    """Build a per-train callback factory that captures the live scheduler.

    Returns a factory function ``factory(train_id) -> TrainCallbacks`` whose
    closures mirror ``workflow._status_check`` / ``workflow._mark_member_done``
    in ``workflow._maybe_enqueue_group_merge`` but are rooted at the harness
    scheduler and hardened to tolerate non-task members (e.g. MCP-submitted
    branches like 'cargo-run-prebuilt-fix') without raising.

    The factory is module-level (not a Harness method) so unit tests can drive
    it with a FakeScheduler without constructing a full Harness.

    NOTE (task γ follow-up): once γ wires GroupMergeRequest construction through
    this factory inside SpeculativeMergeWorker, the inline ``_status_check`` /
    ``_mark_member_done`` closures in ``workflow._maybe_enqueue_group_merge``
    should be retired so there is a single source of truth for train-callback
    semantics (synthesis logic, no-op guard, merged-provenance shape).
    """
    from orchestrator.merge_queue import TrainCallbacks

    def factory(train_id: str) -> TrainCallbacks:
        async def status_check(ids: list[str]) -> dict[str, str]:
            statuses, err = await scheduler.get_statuses(ids)
            if err is not None:
                logger.warning(
                    'train %s: status_check get_statuses error — returning partial (will park): %s',
                    train_id, err,
                )
                return statuses or {}
            # Synthesize _TRAIN_MEMBER_READY_STATUS for ids the scheduler does not
            # know about (non-task members such as MCP-submitted branches).
            # SpeculativeMergeWorker._do_train_merge status pre-check parks the train
            # if any member status != 'merge-deferred'; synthesising it here lets a
            # mixed task/non-task train pass the gate.  Synthesis only fires when
            # Scheduler.get_statuses succeeded (err is None) to prevent advancing a
            # train on a lie when the backend is down.
            #
            # TRADE-OFF: any id silently omitted by the scheduler — whether a genuine
            # non-task MCP branch or a real scheduler task that was dropped/lost — is
            # treated as a non-task member and synthesised ready.  A dropped real task
            # would advance (and mark_member_done would no-op), leaving it un-flipped
            # with a stale status.  A debug diagnostic is emitted for each synthesised
            # id so this path does not advance trains completely silently.
            synthesised = [mid for mid in ids if mid not in statuses]
            for mid in synthesised:
                logger.debug(
                    'train %s: member %s absent from scheduler — synthesising '
                    '%r as ready (treating as non-task member)',
                    train_id, mid, _TRAIN_MEMBER_READY_STATUS,
                )
            return {mid: statuses.get(mid, _TRAIN_MEMBER_READY_STATUS) for mid in ids}

        async def mark_member_done(mid: str, sha: str) -> None:
            # Existence check: issue a Scheduler.get_statuses probe before calling
            # mark_done so a non-task member (absent from the scheduler) can be
            # no-op'd without raising.  (err is None and mid not in result) is the
            # unambiguous "no scheduler task" signal: Scheduler.get_statuses silently
            # omits unknown ids, so this is NOT a transient error.  A raise would hit
            # SpeculativeMergeWorker._do_train_merge post-advance flip loop and
            # falsely trigger TRAIN_PARTIAL_FLIP.
            #
            # LIMITATION: the no-op guarantee holds only when the existence check
            # itself succeeds (err is None).  On a transient Scheduler.get_statuses
            # error (err is not None), control falls through to mark_done
            # unconditionally; for a genuine non-task member this could raise and
            # trigger TRAIN_PARTIAL_FLIP.  This is low-probability (transient error
            # must coincide with a non-task member's guard call) and accepted: for
            # real task members the fall-through is correct (best-effort flip; any
            # mark_done error propagates to the partial-flip handler as intended).
            #
            # PERFORMANCE NOTE: this issues an extra Scheduler.get_statuses round-trip
            # per member solely to detect task existence before mark_done.  Trains are
            # small and infrequent so the cost is acceptable.  Future task γ wiring
            # could reuse the statuses already fetched by status_check to avoid it.
            statuses, err = await scheduler.get_statuses([mid])
            if err is None and mid not in statuses:
                logger.info(
                    'train %s: member %s has no scheduler task — mark_member_done no-op',
                    train_id, mid,
                )
                return
            await scheduler.mark_done(mid, kind='merged', sha=sha, note=f'train {train_id}')
            # task 2280 (PRD WA-3): consume the tip's write-ahead LandedRow inline
            # so a train-landed member no longer leaves a stale row surviving to the
            # next startup for RC-3 to prune. Idempotent (no-op for non-tip members,
            # which hold no row) and fail-safe when the façade is unbound (tests /
            # bare worker). Placed AFTER a SUCCESSFUL mark_done — a raised mark_done
            # skips this so the row survives for the reconciler. Mirrors the 2681
            # single-branch precedent at workflow.py:1912-1917.
            MergeProvenance.consume(mid)
            # B3 (T7): release warm lane for the done member after the status flip.
            # Idempotent/never-raise via the shared primitive.
            if git_ops is not None:
                await git_ops.release_lane_for_terminal_task(mid)
            else:
                logger.debug(
                    'train %s: B3 lane release skipped for %s — '
                    'factory built without git_ops (missing wiring?)',
                    train_id, mid,
                )

        async def redrive_member(mid: str, found_on_main: bool, sha: str | None) -> None:
            # Existence check (mirrors mark_member_done): non-task members
            # (absent from the scheduler) are no-op'd without raising.
            # status_check synthesises 'merge-deferred' for them, so the worker
            # would otherwise attempt to re-drive a non-task member spuriously.
            statuses, err = await scheduler.get_statuses([mid])
            if err is None and mid not in statuses:
                logger.info(
                    'train %s: member %s has no scheduler task — redrive_member no-op',
                    train_id, mid,
                )
                return
            if found_on_main:
                # Double-landing guard: a partner's merge already brought this
                # branch into main, so we mark it done directly.
                await scheduler.mark_done(
                    mid,
                    kind='found_on_main',
                    sha=sha,
                    note=f'coalesce-derail re-drive: branch already on main (train {train_id})',
                )
                # task 2280 (PRD WA-3): consume the tip's write-ahead LandedRow on
                # this found_on_main done-write, mirroring mark_member_done above.
                # Idempotent for non-tip members, fail-safe when unbound; runs only
                # after a SUCCESSFUL mark_done. NOT added to the else (not-on-main →
                # pending re-drive) branch — that member is not done.
                MergeProvenance.consume(mid)
                # B3 (T6/T8): release warm lane after the done flip.
                # Idempotent/never-raise via the shared primitive.
                if git_ops is not None:
                    await git_ops.release_lane_for_terminal_task(mid)
                else:
                    logger.debug(
                        'train %s: B3 lane release skipped for %s — '
                        'factory built without git_ops (missing wiring?)',
                        train_id, mid,
                    )
            else:
                # Race-condition guard: re-check the current status from the
                # probe we just issued.  If the member has advanced to a live
                # status (e.g. 'in-progress') in the window between the
                # _redrive_coalesce_members snapshot and this flip, the live
                # workflow already owns the transition — skip the clobber.
                # When err is not None (transient backend error) we fall
                # through conservatively and attempt the flip, mirroring the
                # same fail-open policy as the non-task probe above.
                current = statuses.get(mid)
                if err is None and current is not None and current != 'merge-deferred':
                    logger.info(
                        'train %s: member %s is now %r (moved past merge-deferred '
                        'since re-drive snapshot) — live workflow owns transition; '
                        'skipping pending flip',
                        train_id, mid, current,
                    )
                    return
                # Flip to pending so the scheduler re-dispatches a fresh solo
                # merge workflow that will own the merge-deferred→done transition.
                await scheduler.set_task_status(mid, 'pending')
                scheduler.clear_requeue_count(mid)

        return TrainCallbacks(
            status_check=status_check,
            mark_member_done=mark_member_done,
            redrive_member=redrive_member,
        )

    return factory


# Status synthesised for non-task train members (MCP-submitted branches with no
# scheduler task).  Must match the value accepted by the status pre-check in
# SpeculativeMergeWorker._do_train_merge exactly; see merge_queue.py.
_TRAIN_MEMBER_READY_STATUS: str = 'merge-deferred'


class _OfflineLaneTaskClient:
    """Concrete ``offline_lane.OfflineLaneTaskClient`` adapter over the scheduler.

    Structurally satisfies the ``OfflineLaneTaskClient`` Protocol
    (``offline_lane.py``) so ``OfflineLaneWorker`` never talks to the
    fused-memory MCP directly (task 2016, closing Bug #1) — mirrors that
    module's ``SuiteRunner``/cross-project scope boundary. Delegates
    entirely to existing, tested ``Scheduler`` helpers; no new MCP plumbing.
    Module-level (not nested in ``Harness``) so it stays trivially
    constructible in isolation, e.g. ``_OfflineLaneTaskClient(scheduler)``.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    async def submit_fix_task(self, arguments: dict) -> str:
        """Submit a new fix task and return its id.

        Reuses ``Harness._extract_task_id`` to normalise the task_id out of
        the MCP ``tools/call`` response envelope, whatever shape it arrives
        in (``{'task_id': ...}``, ``structuredContent``, or a ``content``
        text block).
        """
        result = await self._scheduler.dispatch_tool('submit_task', arguments, timeout=30)
        return Harness._extract_task_id(result) or ''

    async def append_suspect_range(self, task_id: str, suspect_range: str) -> None:
        """Append a suspect commit range to an already-open fix task.

        Uses ``metadata_mode='additive'`` — a recursive list-union performed
        server-side — so the range is appended (and dedup'd) atomically with
        no read-modify-write round-trip.
        """
        await self._scheduler.update_task(
            task_id, {'suspect_ranges': [suspect_range]}, metadata_mode='additive',
        )

    async def get_status(self, task_id: str) -> str:
        """Return a previously filed fix task's current status.

        Logs (rather than silently swallows) a non-``None`` error half from
        ``get_statuses`` — e.g. a transient MCP outage — so it stays
        observable; the caller (``_maybe_promote_blocker``) still degrades
        to treating the empty status as "not done, not terminal" either way
        (task 2016 amendment).
        """
        statuses, err = await self._scheduler.get_statuses([task_id])
        if err is not None:
            logger.warning(
                'offline-lane: get_statuses failed for task %s: %s — '
                'treating status as unknown this check',
                task_id, err,
            )
        return statuses.get(task_id, '')


def _extract_tagger_entries(payload: Any) -> list:
    """Peel known StructuredOutput wrapper keys and return the entries list.

    The module tagger's output_schema is object-rooted with a top-level
    ``predictions`` key (renamed from ``tasks`` — task 2561 defect 3). When
    the schema's sole key collides with the StructuredOutput tool's own
    parameter name and the prompt's dominant domain noun, the model
    sometimes double-wraps its answer. This accepts, in order: the current
    flat ``{"predictions": [...]}`` shape, the legacy single-wrap
    ``{"tasks": [...]}``, and the legacy double-wrap
    ``{"tasks": {"tasks": [...]}}``. Bounded to a few iterations so
    pathological nesting fails safe to ``[]`` — no tagging, no crash —
    rather than looping, matching the existing bad-output early-return
    semantics. Pure function, no side effects.
    """
    for _ in range(3):
        if not isinstance(payload, dict):
            break
        if 'predictions' in payload:
            payload = payload['predictions']
        elif 'tasks' in payload:
            payload = payload['tasks']
        else:
            break
    return payload if isinstance(payload, list) else []


class Harness:
    """Top-level orchestration loop."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        from orchestrator.agents.sandbox_dispatch import set_backend
        set_backend(config.sandbox.backend)
        self.mcp = McpLifecycle(config)
        # Size the warm-lane pool from max_concurrent_tasks (D9: read once at startup,
        # single source of truth). Pass 0 when the knob is off so GitOps leaves
        # warm_lane_pool=None — non-dispatch call sites (cli.py verify-merge,
        # recover_main.py, evals/runner.py) default to size=0 and stay on cold path.
        self.git_ops = GitOps(
            config.git,
            config.project_root,
            warm_lane_pool_size=(
                (config.max_concurrent_tasks + config.git.spare_warm_lanes)
                if config.git.warm_lane_pool else 0
            ),
            # Spec-pool size: K (same as speculation_depth) when knob on, else 0.
            # self._speculation_k is safe here — self.config is already set above.
            merge_spec_warm_lane_pool_size=(
                self._speculation_k if config.git.merge_spec_warm_lane_pool else 0
            ),
            # Teardown-archival backstop config (task 2786, prd β): the live
            # submodel reference (green-tier hot-reloadable enabled/root) so the
            # cleanup_worktree backstop archives before removing a cold worktree.
            # The 3 non-dispatch construction sites (cli/recover/evals) omit this
            # kwarg and stay inert — byte-identical to before.
            transcript_archive=config.transcript_archive,
        )
        # Constructor-injected callback bundle (task 2235, W10-α): all nine
        # Harness↔Scheduler hooks are wired in ONE SchedulerCallbacks(...) at
        # construction time — no post-construction install window where the
        # Scheduler exists but a subset of its callbacks are still unset.
        #   - on_park_stop_trip: Scheduler trips → Harness.pause_scheduler, the
        #     full pause bundle (persistence + event + log).  Sibling tasks
        #     (cost-ceiling 1323, EWA digest 1327) call pause_scheduler() directly.
        #   - on_external_dep_block: an external dep is cancelled or persistently
        #     unresolvable → Harness._block_and_escalate_external_dep (needs the
        #     harness's EscalationQueue and set_task_status).
        #   - on_delivered_check_block (task 2583, epsilon): a delivered check has
        #     FAILED for delivered_checks.grace_cycles consecutive ticks →
        #     Harness._block_and_escalate_delivered_check, filing BORN-AT-L2
        #     (not on_external_dep_block's L1).
        #   - on_starvation_warn / on_starvation_resolve (task 1880): a
        #     deps-satisfied pending task starved past both thresholds files a
        #     non-blocking INFO escalation; resolved when the task dispatches
        #     (or is GC'd as terminal).
        #   - warm_base_health_probe / on_warm_base_warn / on_warm_base_promote_l2 /
        #     on_warm_base_resolve (task 2061): the scheduler probes base health
        #     once per tick via the injected async probe and reacts with the
        #     three injected callbacks.
        #   - suppress_blocked_write: self._is_action_teardown_task, a bound-
        #     method wrapper over ``_action_teardown_tasks`` (defined below). A
        #     bound method is late-bound — valid to reference here even though
        #     the Counter itself is created AFTER this Scheduler build, because
        #     it resolves the Counter at CALL time, not at reference time.
        self.scheduler = Scheduler(
            config,
            callbacks=SchedulerCallbacks(
                on_park_stop_trip=self.pause_scheduler,
                on_external_dep_block=self._block_and_escalate_external_dep,
                on_delivered_check_block=self._block_and_escalate_delivered_check,
                on_starvation_warn=self._file_starvation_info,
                on_starvation_resolve=self._resolve_starvation_info,
                warm_base_health_probe=self._probe_warm_base_health,
                on_warm_base_warn=self._file_warm_base_hard_down_notice,
                on_warm_base_promote_l2=self._promote_warm_base_hard_down_l2,
                on_warm_base_resolve=self._resolve_warm_base_hard_down,
                suppress_blocked_write=self._is_action_teardown_task,
            ),
            override_store=OverrideStore.from_config(config),
            park_eviction_store=ParkEvictionRequestStore.from_config(config),
        )
        # --- Action-teardown suppression (task 1620, β Pair F / C3.2) ---
        # Counter of task_ids currently undergoing action-teardown (park/restart/abandon).
        # Stamped (incremented) before the status write + kill; decremented in the
        # finally block once the kill window closes.  A Counter (vs plain set) ensures
        # overlapping teardown coros for the SAME task_id — low probability but possible
        # with concurrent escalation resolutions — do not prematurely clear each other's
        # suppression window.  Mirrors _workflow_cancel_at's grace lifecycle so a
        # re-dispatched (restart→pending) workflow can write 'blocked' legitimately in
        # its next incarnation rather than being permanently suppressed.  Consulted via
        # self._is_action_teardown_task, wired above as the suppress_blocked_write hook.
        self._action_teardown_tasks: Counter[str] = Counter()
        # Wire the landed-outbox consult-before-dispatch gate (task 2156, W1 δ
        # — SD-1/B5).  Declared in scheduler.py, installed here alongside the
        # SchedulerCallbacks bundle above — same declare-in-scheduler /
        # install-in-harness pattern, but OUT OF SCOPE for task 2235's
        # constructor-seam migration (see task 2156). γ's own
        # _reconcile_landed_outbox is the sibling read path.
        self.scheduler._landed_outbox_gate = self._landed_dispatch_gate
        # Wire the already-landed pre-dispatch gate (task 2313) — catches
        # out-of-band landings (ancestry / merge-marker / content-equivalence)
        # that never passed through this orchestrator's own merge queue.
        # Declared in scheduler.py, installed here alongside the other
        # callback installs — same declare-in-scheduler / install-in-harness
        # pattern as the adjacent _landed_outbox_gate, but OUT OF SCOPE for
        # task 2235's constructor-seam migration (see task 2313).
        self.scheduler._already_landed_gate = self._already_landed_dispatch_gate
        # Wire the reclaim-on-exhaustion safety valve callbacks (task 1933).
        # Declared on git_ops with default None (byte-identical when not wired);
        # installed here when the knob is on — mirrors the _on_park_stop_trip /
        # _on_external_dep_block declare-in-callee / install-in-harness pattern.
        # - warm_lane_reclaim_candidate_provider: async (list[str])->set[str] —
        #   returns the NON-TERMINAL subset of candidate branches (_reconcile_
        #   terminal_lanes INVERTED, with the same fail-safe-on-resolver-failure).
        # - warm_lane_dispatched_predicate: sync (str)->bool — re-checked
        #   atomically under the pool lock (TOCTOU guard) inside reclaim_victim.
        if config.git.warm_lane_reclaim_on_exhaustion:
            self.git_ops.warm_lane_reclaim_candidate_provider = (
                self._warm_lane_reclaim_candidates
            )
            self.git_ops.warm_lane_dispatched_predicate = self._is_branch_dispatched
        # Wire the pool-storage-absent callback (task 2099): git_ops guard
        # sites (prune_worktrees, _run_warm_lane_gc_reclaim, acquire
        # create-once for both pools) call this best-effort when
        # pool_storage_present() is False. Declared on git_ops with default
        # None (byte-identical when unwired) — same declare-in-callee /
        # install-in-harness pattern as warm_lane_reclaim_candidate_provider.
        self.git_ops._on_pool_storage_absent = self._file_pool_storage_absent_escalation
        # Wire the warm-lane record-drift callback (task 2986, W2b I3/I4): the
        # pool fires this opaque callback when drift_l2_threshold consecutive
        # durable .lane-state writes fail — the loud signal that the durable
        # ASSIGNED/RELEASED records have drifted from the in-memory assignment
        # map.  The pool NEVER raises on a mirror failure (fail-open, I3), so
        # this filer is the only path by which the drift becomes visible.  Same
        # declare-on-callee (default None) / install-in-harness pattern as
        # _on_pool_storage_absent above; installed only when a pool exists so
        # pool-less hosts stay byte-identical.
        if self.git_ops.warm_lane_pool is not None:
            self.git_ops.warm_lane_pool.set_on_lane_record_drift(
                self._file_lane_record_drift_l2
            )
            # Wire the structural-exhaustion callback (task 2988, PRD ε pole-2):
            # GitOps fires this once warm_lane_structural_exhaustion_l2_threshold
            # consecutive acquires return EXHAUSTED — the loud signal that the
            # pool is stuck emitting backpressure forever (silent-infinite-
            # requeue).  Same declare-on-callee (default None) / install-in-
            # harness pattern as _on_pool_storage_absent / _on_lane_record_drift;
            # installed only when a pool exists so pool-less hosts stay
            # byte-identical.
            self.git_ops._on_structural_exhaustion = (
                self._file_structural_exhaustion_l2
            )
        # In-memory hint gating the orphan-reaper's per-tick resolve scan
        # (task 2099 review-fix, efficiency). MUST default True ("maybe
        # pending") rather than False ("never filed") — a fresh process
        # start needs its first sweep to still scan the escalation queue so
        # a stale pool-storage-absent L1 left open by a pre-gate build is
        # auto-cleared across a restart (see _reap_orphan_worktrees). Set
        # True whenever _file_pool_storage_absent_escalation trips; cleared
        # to False only by _resolve_pool_storage_absent_escalation once it
        # confirms zero pending escalations, so the common healthy
        # steady-state stops paying the per-tick queue-scan cost.
        self._pool_storage_absent_maybe_pending: bool = True
        self.briefing = BriefingAssembler(config)
        self.report = HarnessReport()
        self._recovered_plans: dict[str, dict] = {}
        # Worktrees that survived crash recovery with a stamped pre-EXECUTE
        # plan (no completed steps) but were NOT pre-loaded into
        # _recovered_plans — we want _plan() to re-run revalidation against
        # the existing stamped plan rather than skipping straight to EXECUTE
        # via initial_plan.  Membership protects them from the
        # _reconcile_stranded_in_progress sweep, which otherwise wipes any
        # worktree not in _recovered_plans.
        self._preserved_worktrees: set[str] = set()
        # Worktrees whose agent_session.json sidecar survived a crash with
        # no plan.json — the next workflow for that task resumes the prior
        # Claude session via --resume rather than spawning fresh.  Keyed
        # by task_id; value is the parsed sidecar dict.
        self._recovered_sessions: dict[str, dict] = {}
        # Parallel to _recovered_sessions (task γ, session-resume guard):
        # the claude-config dir (``<lane>/.task/claude-config-<branch>``)
        # captured at adoption, so the _run_slot eligibility guard can
        # RE-glob the transcript at dispatch (not adoption) — catching the
        # B4 foreign-acquire reseed that wipes .task between boot and
        # re-dispatch (INV-3 corroborate-before-acting). Value is the config
        # dir path as a string. Kept separate from the adopted sidecar dict
        # (which flows into build_workflow) to keep the resume payload clean.
        self._recovered_session_config_dirs: dict[str, str] = {}
        # Consecutive-per-boot session_resume_fallback streak (task γ storm
        # escape, INV-4). Incremented on each reason-carrying fallback in
        # _run_slot; reset to 0 on any eligible resume. When it reaches
        # session_resume.fallback_storm_threshold, one deduped L1 is filed.
        # Capped/disabled degradations do NOT feed it (by design).
        self._session_resume_fallback_streak: int = 0

        # Usage cap gate
        self.usage_gate: UsageGate | None = (
            UsageGate(config.usage_cap) if config.usage_cap.enabled else None
        )

        # Review checkpoints
        self.review_checkpoint: ReviewCheckpoint | None = (
            ReviewCheckpoint(config, self.mcp, self.usage_gate)
            if config.review.enabled else None
        )
        self._review_running = False
        self._pending_review_task: asyncio.Task | None = None
        self._task_modules: dict[str, list[str]] = {}  # task_id -> modules

        # Escalation support
        self._escalation_queue: EscalationQueue | None = None
        self._escalation_events: dict[str, asyncio.Event] = {}
        self._escalation_task: asyncio.Task | None = None

        # Shared done_evidence_stale sink (task 2677, INV-5: one instance,
        # not N copies) — constructed here with escalation_queue=None since
        # self._escalation_queue (above) is still None at this point; late-
        # bound via the mutable public .escalation_queue attr once the real
        # queue is created below (~run()). Injected by reference into the
        # SpeculativeMergeWorker and threaded into the module-level
        # reconcile_landed_* calls so every done-writer site shares one
        # memo + one dedupe fingerprint namespace.
        self._provenance_conflict_sink = ProvenanceConflictSink()

        # Ground-truth resolver seam (task 2243, W10-θ2) — lazily built (and
        # memoized) by _get_ground_truth() the first time a reconcile sweep
        # needs it. MUST stay unbuilt here: TaskGroundTruth captures
        # escalation_queue at construction, and self._escalation_queue above
        # is still None at this point in __init__ (only populated later, in
        # run()) — building eagerly would freeze that None forever. See
        # _get_ground_truth's docstring.
        self._ground_truth: TaskGroundTruth | None = None

        # Unified lifecycle seam (task 2241, W10-η — PRD §5.3 LR-1/2/3):
        # the eleven background-loop/service lifecycles below register into
        # ONE ordered LifecycleRegistry instead of eleven hand-rolled
        # _start_*/_stop_*/_*_loop triplets.  None until
        # _build_lifecycle_registry() first runs (lazily, at the top of
        # run() — idempotent, so a test may pre-build + monkeypatch it
        # before driving run()).  A None _lifecycle is a safe no-op for the
        # finally-block's stop_all() guard if run() raises before startup
        # ever reaches it.
        self._lifecycle: LifecycleRegistry | None = None
        # Fire-and-forget async tasks (strong refs prevent GC mid-flight).
        # Mirrors the active-set + add_done_callback(discard) idiom at line ~856.
        self._background_tasks: set[asyncio.Task] = set()
        # The orchestrator's event loop, captured in run() once we're inside
        # asyncio.run().  Lets callbacks that may fire OFF the loop (the sync
        # escalation MCP tool resolve_issue runs on a FastMCP threadpool worker)
        # schedule coroutines back onto it via run_coroutine_threadsafe — see
        # _schedule_coro_threadsafe.  None until run() starts.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Idle-while-paused state (task 1322 follow-up).  Throttle timestamp for
        # the "alive but paused" WARNING emitted by the main run loop.
        self._last_paused_idle_log: float = 0.0
        # Run-forever idle state.  Set by run(); throttle timestamp for the
        # "Idle — polling" INFO emitted when a drained queue produced no work.
        self._until_idle: bool = False
        self._last_idle_poll_log: float = 0.0
        # Reason captured by _load_persisted_scheduler_pause() at run() startup,
        # filed as a (deduped) L1 escalation once the escalation queue exists.
        self._restored_pause_reason: str | None = None

        # Digest + EWA trip counters (task 1327 AFK hardening).
        # Delegated to _init_digest_state() so test helpers can call the same
        # canonical code without duplicating the counter names by value (task 1449).
        self._init_digest_state()

        # Soft-cancel registry — keyed by task_id, set externally to abort
        # long workflow waits (merge queue future, steward grace period).
        # Mirrors ``_escalation_events``: created in ``_run_slot`` for each
        # active task and cleared on slot exit.  Used by ``cancel_workflow``
        # and by the reconciliation subscription that watches for terminal
        # task status transitions.  See zombie-escalation fix Step 4.
        self._workflow_cancel_events: dict[str, asyncio.Event] = {}

        # Per-task asyncio.Task handle for the _run_slot coroutine — registered
        # at _run_slot start, popped in its finally.  Enables hard-cancel
        # fallback when a workflow ignores the soft cancel_event.
        # See task 1491, ITEM 2 (hard-cancel fallback).
        self._workflow_slot_tasks: dict[str, asyncio.Task] = {}

        # Per-task CONSECUTIVE streak of requeues that invoked no agent at all
        # (task 3068).  Fed from _apply_retry_cap — the single per-report
        # chokepoint that sees every outcome — and read by
        # _maybe_zero_progress_requeue_alert.  Pure in-memory; entries are
        # popped on any progress, so this stays proportional to the number of
        # tasks CURRENTLY looping rather than growing over a weeks-long run.
        self._zero_progress_tracker = ZeroProgressRequeueTracker()

        # Consecutive terminal-status poll counts per task.  Incremented each
        # poll a workflow is terminal but still active; reset when it is no
        # longer terminal or drops out of the active set.  Controls the
        # soft→hard cancel threshold (config.terminal_status_hard_cancel_polls).
        self._terminal_cancel_counts: dict[str, int] = {}

        # Escalation-watcher-auto subprocess supervisor (task 1326).
        # Keeps a fresh escalation-watcher-auto agent alive via invoke_with_cap_retry
        # across multi-day AFK windows with rotation, exponential backoff, and a
        # crashloop→pause_scheduler guard.
        self._watcher_supervisor_task: asyncio.Task | None = None
        # Monotonic timestamps of unclean watcher exits; used by the crashloop guard
        # to count failures within watcher_crashloop_window_secs.
        self._watcher_unclean_exits: deque[float] = deque()
        # Monotonic timestamps of degenerate-clean watcher exits (duration below
        # watcher_misconfigured_min_rotation_secs); used by the cost-runaway guard
        # (task 1388).  Separate from _watcher_unclean_exits so the trip reason
        # ('watcher_misconfigured' vs 'watcher_crashloop') is unambiguous.
        self._watcher_degenerate_clean_exits: deque[float] = deque()

        # Last main SHA successfully swept; used to skip the expensive full
        # verify when main has not advanced since the previous pass.
        self._last_swept_main_sha: str | None = None

        # Deterministic-strand reconciliation sweep — task 2074.
        # Periodic recovery sweep for deterministic gate/deploy tasks stranded
        # BLOCKED by a past occurrence (task 2059) and re-validation of
        # already-open deterministic-deploy escalations against live systemd
        # unit state.  See harness.py's deterministic-recon-sweep section.
        # Injectable unit-inspector seam (mirrors DeterministicRunner's
        # constructor-injected unit_inspector); None uses the module-level
        # default _recon_inspect_unit.  Injected in tests to avoid real systemd.
        self._recon_unit_inspector: Callable[[str], Awaitable[dict]] | None = None

        # No-landings circuit-breaker — θ=1893, PRD §5.5.
        # Pure detection logic lives in merge_queue.NoLandingsCircuitBreaker;
        # I/O wiring (disk-stat, halt/resume, escalation file/resolve) lives
        # here, in a periodic loop mirroring the main-tip-sweep lifecycle.
        from orchestrator.merge_queue import NoLandingsCircuitBreaker as _NLCB  # noqa: PLC0415
        self._no_landings_breaker: _NLCB = _NLCB(
            window_samples=config.no_landings_breaker_window_samples,
            disk_free_floor_bytes=config.no_landings_breaker_disk_free_floor_bytes,
        )
        # One-shot flag: log a WARNING if disk_free_floor_bytes exceeds the
        # volume's total capacity (making disk-recovery auto-resume unreachable).
        self._no_landings_floor_capacity_warned: bool = False

        # Wall-clock of the most recent _workflow_cancel_events.set() call,
        # keyed by task_id.  R3-race-guard window — the sweep skips a task
        # whose workflow was cancelled within the last grace period, since
        # the workflow's finally: block may still be writing state.  Task
        # 2235 (W10-alpha): this stamp, its grace constant, and the
        # membership predicate now live on the Scheduler (beside
        # ``_dispatched`` / ``lock_table``, their single writer) — see
        # ``scheduler.note_workflow_cancelled`` / ``.clear_workflow_cancel`` /
        # ``.workflow_cancel_recent``.  The ``_workflow_cancel_at`` property
        # and ``_workflow_cancel_recent`` method below are thin back-compat
        # shims that forward to the Scheduler.

        # Merge queue — single worker owns all main-branch advancement
        self._merge_queue: asyncio.Queue = asyncio.Queue()
        # Single shared registry injected into both the escalation server and
        # every TaskWorkflow so the MCP coalesce gate sees workflow-path merges.
        from orchestrator.merge_queue import InFlightMergeRegistry
        self._merge_inflight_registry: InFlightMergeRegistry = InFlightMergeRegistry()
        self._merge_worker: SpeculativeMergeWorker | None = None
        self._merge_worker_task: asyncio.Task | None = None
        # Durable journal for in-flight merge requests (task 1772).
        # Persisted at data/orchestrator/merge_queue.json; recovered on restart
        # by _recover_pending_merges (called from run() after _rehydrate_merge_halt).
        self._merge_store: MergeQueueStore = MergeQueueStore(
            config.project_root / 'data' / 'orchestrator' / 'merge_queue.json'
        )

        # Post-merge staleness hook — restart services when a landed diff
        # touches their watched paths.  Built in _start_merge_worker (where
        # git_ops/event_store/config are all live).  The list holds two entries:
        #   [0] fused-memory (require_idle=True — idle quiet window only)
        #   [1] dashboard    (require_idle=False — fires even during dispatch)
        self._service_restart_coordinators: list[StaleServiceRestartCoordinator] = []

        # Offline deep-test lane (task 1951, β2 — not yet built) — singleton
        # notifiee slot fanned out from _note_merge_all alongside the service
        # restart coordinators above.  None until β2 registers its own
        # on_post_merge callback here directly (mirrors the direct-attribute
        # registration convention used by _service_restart_coordinators).
        # Contract (see _note_offline_lane docstring): the notifiee is awaited
        # synchronously on the merge-landed hot path, so it must
        # enqueue-and-return promptly rather than block on the deep-test run.
        self._offline_lane_notifiee: Callable[[str, str, str], Awaitable[object]] | None = None
        # The OfflineLaneWorker instance + its background asyncio.Task (task
        # 1953, β2).  Both None until _start_offline_lane builds and launches
        # them (enable-gated: offline_lane_enabled AND
        # persistent_offline_deep_worktree); cleared by _stop_offline_lane.
        self._offline_lane_worker: OfflineLaneWorker | None = None
        self._offline_lane_task: asyncio.Task | None = None

        # Event store — created at run start with a generated run_id
        self.event_store: EventStore | None = None

        # Run store — incremental task result persistence (shares runs.db)
        self._run_store: RunStore | None = None
        self._run_id: str | None = None

        # Cost store — per-invocation cost tracking (shares runs.db)
        self.cost_store: CostStore | None = None

        # Auto-eval — process-local set of redo task ids dispatched during
        # this run. Used by the daily-budget query in ``_maybe_auto_eval``.
        self._auto_eval_redo_task_ids: set[str] = set()

        # Singleton lock — held for the duration of run()
        self._lock_file: IO | None = None

    def _is_action_teardown_task(self, tid: str) -> bool:
        """Bound-method wrapper wired as the ``suppress_blocked_write``
        SchedulerCallbacks hook (task 2235).

        ``_action_teardown_tasks`` (a ``Counter``) is created AFTER the
        Scheduler is constructed in ``__init__``, so it cannot be passed
        directly as ``self._action_teardown_tasks.__contains__`` at
        Scheduler-build time. A bound method is late-bound — it resolves
        the Counter at CALL time instead — so ``SchedulerCallbacks`` can be
        built with all nine hooks in one shot at construction, with no
        ``__init__`` reordering needed.
        """
        return tid in self._action_teardown_tasks

    @property
    def _speculation_k(self) -> int:
        """Single shared K source: 1 + len(enabled_verify_runners).

        Used by BOTH GitOps spec-pool sizing (harness.__init__) and
        _start_merge_worker (speculation_depth + enforce_persistent_worktree_serial_lane)
        so the spec pool size and the worker cap cannot drift as verify_runners grows.

        Mirrors the invariant comment at _start_merge_worker: all three Lever-C knobs
        derive from ONE expression.
        """
        return 1 + len(self.config.enabled_verify_runners)

    def _init_digest_state(self) -> None:
        """Initialise task-1327 AFK-hardening digest counters.

        Called from ``__init__`` and from the test helper
        ``_init_harness_state_for_test`` (orchestrator/tests/_orch_helpers.py)
        so that fixtures constructing ``Harness`` via ``Harness.__new__`` call
        canonical code rather than duplicating counter names by value.

        When new digest-related counters are added to ``_maybe_write_digest``
        in the future, add them HERE — ``__init__``, the test helper, and all
        seven ``Harness.__new__``-based fixtures pick them up automatically.

        Attributes
        ----------
        _escalation_event_count:
            Incremented on every escalation submit/resolve callback.
        _last_digest_event_count:
            Snapshot of the count at the last digest write.
        _ewa_value:
            Current EWA state (process-local; resets on restart).
        _last_digest_window_end_iso:
            ISO timestamp of the last digest window's end; set to start time
            on first run.  Note: done_count comes from EventStore
            (count_done_in_window) as the single source of truth — no
            scheduler-delta counter needed (task 1421).
        """
        self._escalation_event_count: int = 0
        self._last_digest_event_count: int = 0
        self._ewa_value: float = 0.0
        self._last_digest_window_end_iso: str = ''

    def _build_lifecycle_registry(self) -> None:
        """Build ``self._lifecycle`` in the canonical eleven-service order.

        Idempotent — a no-op once ``self._lifecycle`` is already set, so a
        test may pre-build (and monkeypatch) the registry before driving
        ``run()`` without ``run()``'s own call clobbering it (task 2241,
        W10-η — PRD §5.3 LR-1/2/3).

        Registration order (also the ``stop_all()`` REVERSE order — LR-2):
        escalation-server, merge-worker, offline-lane, orphan-l0-reaper,
        terminal-status-watcher, watcher-supervisor, stranded-reconcile,
        main-tip-sweep, no-landings-breaker, deterministic-recon-sweep,
        warm-lane-gc. escalation-server and merge-worker start first so the
        recovery block (which runs immediately after ``start_all()``
        returns) can depend on both being live. The seven sweeps are
        conditionally registered on their own ``config.X_enabled`` gate —
        ``BackgroundService`` itself has no ``enabled`` field (PRD §5.3
        shape); a disabled sweep is simply absent from the registry.
        """
        if self._lifecycle is not None:
            return
        registry = LifecycleRegistry()
        sweep_backoff = BackoffPolicy(_BG_LOOP_FAILURE_BACKOFF_SECS)

        registry.register(ManagedService(
            name='escalation-server',
            start_fn=self._start_escalation_server,
            stop_fn=self._stop_escalation_server,
            stop_timeout_secs=_LIFECYCLE_SERVICE_STOP_TIMEOUT_SECS,
        ))
        registry.register(ManagedService(
            name='merge-worker',
            start_fn=self._start_merge_worker,
            stop_fn=self._stop_merge_worker,
            stop_timeout_secs=_LIFECYCLE_SERVICE_STOP_TIMEOUT_SECS,
        ))
        registry.register(ManagedService(
            name='offline-lane',
            start_fn=self._start_offline_lane,
            stop_fn=self._stop_offline_lane,
            stop_timeout_secs=_LIFECYCLE_SERVICE_STOP_TIMEOUT_SECS,
        ))
        if self.config.orphan_l0_reaper_enabled:
            registry.register(BackgroundService(
                name='orphan-l0-reaper',
                pass_fn=self._run_orphan_l0_reaper_pass,
                interval_secs=self.config.orphan_l0_check_interval_secs,
                backoff=sweep_backoff,
                stop_timeout_secs=_LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS,
                max_failure_logs=_BG_LOOP_MAX_FAILURE_LOGS,
            ))
        if self.config.terminal_status_watcher_enabled:
            registry.register(BackgroundService(
                name='terminal-status-watcher',
                pass_fn=self._run_terminal_status_watcher_pass,
                interval_secs=self.config.terminal_status_poll_interval_secs,
                backoff=sweep_backoff,
                stop_timeout_secs=_LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS,
                max_failure_logs=_BG_LOOP_MAX_FAILURE_LOGS,
            ))
        registry.register(ManagedService(
            name='watcher-supervisor',
            start_fn=self._start_watcher_supervisor,
            stop_fn=self._stop_watcher_supervisor,
            stop_timeout_secs=_LIFECYCLE_SERVICE_STOP_TIMEOUT_SECS,
        ))
        if self.config.stranded_reconcile_enabled:
            registry.register(BackgroundService(
                name='stranded-reconcile',
                pass_fn=self._run_stranded_reconcile_pass,
                interval_secs=self.config.stranded_reconcile_interval_secs,
                backoff=sweep_backoff,
                stop_timeout_secs=_LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS,
                max_failure_logs=_BG_LOOP_MAX_FAILURE_LOGS,
            ))
        if self.config.main_tip_sweep_enabled:
            registry.register(BackgroundService(
                name='main-tip-sweep',
                pass_fn=self._run_main_tip_sweep,
                interval_secs=self.config.main_tip_sweep_interval_secs,
                backoff=sweep_backoff,
                stop_timeout_secs=_LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS,
                max_failure_logs=_BG_LOOP_MAX_FAILURE_LOGS,
            ))
        if self.config.no_landings_breaker_enabled:
            registry.register(BackgroundService(
                name='no-landings-breaker',
                pass_fn=self._run_no_landings_breaker_pass,
                interval_secs=self.config.no_landings_breaker_interval_secs,
                backoff=sweep_backoff,
                stop_timeout_secs=_LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS,
                max_failure_logs=_BG_LOOP_MAX_FAILURE_LOGS,
            ))
        if self.config.deterministic_recon_sweep_enabled:
            registry.register(BackgroundService(
                name='deterministic-recon-sweep',
                pass_fn=self._run_deterministic_recon_sweep,
                interval_secs=self.config.deterministic_recon_sweep_interval_secs,
                backoff=sweep_backoff,
                stop_timeout_secs=_LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS,
                max_failure_logs=_BG_LOOP_MAX_FAILURE_LOGS,
            ))
        if self.config.warm_lane_gc_enabled:
            registry.register(BackgroundService(
                name='warm-lane-gc',
                pass_fn=self._run_warm_lane_gc_pass,
                interval_secs=self.config.warm_lane_gc_interval_secs,
                backoff=sweep_backoff,
                stop_timeout_secs=_LIFECYCLE_SWEEP_STOP_TIMEOUT_SECS,
                max_failure_logs=_BG_LOOP_MAX_FAILURE_LOGS,
            ))
        self._lifecycle = registry

    async def _run_orphan_l0_reaper_pass(self) -> None:
        """Async ``pass_fn`` wrapper for the orphan-L0 pass.

        ``_reap_orphan_l0_escalations`` is itself async (task 2725 — it
        needs to ``await scheduler.get_task`` to check for a
        terminal+merged subject task on the done-step-commit orphan
        class); ``BackgroundService`` requires an ``Awaitable[None]``
        pass_fn (task 2241, W10-η).
        """
        await self._reap_orphan_l0_escalations()

    async def _run_terminal_status_watcher_pass(self) -> None:
        """Async ``pass_fn`` wrapper discarding the terminal-scan cancelled count.

        ``_scan_for_terminal_active_tasks`` returns ``int`` (the cancelled
        count — asserted directly by its own dedicated tests); ``BackgroundService``
        requires an ``Awaitable[None]`` pass_fn (task 2241, W10-η).
        """
        await self._scan_for_terminal_active_tasks()

    async def _run_stranded_reconcile_pass(self) -> None:
        """Async ``pass_fn`` wrapper folding the two-call stranded pass.

        Hoisted verbatim from the deleted ``_stranded_reconcile_loop`` body
        (task 2241, W10-η): the mid-run stranded-in-progress reconcile
        followed by the periodic terminal-lane sweep (Diff 7 layer-A
        backstop).
        """
        await self._reconcile_stranded_in_progress(mid_run=True)
        await self._reconcile_terminal_lanes()

    async def run(
        self,
        prd_path: Path | None = None,
        dry_run: bool = False,
        delay_secs: int = 0,
        force_dirty_start: bool = False,
        retag_modules: bool = False,
        until_idle: bool = False,
    ) -> HarnessReport:
        """Execute the full orchestration pipeline.

        If *prd_path* is ``None``, skip PRD parsing and run existing tasks.
        If *delay_secs* > 0, sleep that many seconds after startup (escalation
        server runs immediately) before executing tasks.
        If *until_idle* is True, exit when the task queue drains (legacy
        one-shot behavior); otherwise run forever, idling and polling for
        newly-scheduled tasks (the default, for long-lived systemd service use).
        """
        self._until_idle = until_idle
        self.report.started_at = datetime.now(UTC).isoformat()

        # Capture the live event loop so off-loop callbacks (e.g. the sync
        # escalation MCP tool resolve_issue, which FastMCP runs on a threadpool
        # worker) can schedule coroutines back onto it.  Must be inside
        # asyncio.run(); Harness is constructed in the sync body of cli.run()
        # where no loop exists yet.
        self._loop = asyncio.get_running_loop()

        # Install the usage-gate SIGHUP handler now that we're inside
        # asyncio.run(). The gate's __init__ runs before the loop exists
        # (Harness is constructed in the sync body of cli.run()), so its
        # auto-registration is a no-op there. The call is idempotent —
        # gates constructed inside an event loop (evals, fused-memory)
        # already installed the handler in __init__.
        if self.usage_gate is not None:
            self.usage_gate.register_signal_handlers()

        # 0. Singleton lock — prevent concurrent orchestrators on same project
        self._lock_file = _acquire_project_lock(self.config.project_root)

        # 0-gc. Disable auto-gc/maintenance on the managed shared repo.
        # PRD os-sandbox α5 (D2 corollary): set gc.auto=0 / maintenance.auto=false
        # at startup — before the first dispatch — so background auto-gc never
        # fires under the narrow shared-.git write-set (create_worktree reasserts
        # it per-dispatch too). Best-effort: the method never raises on a git rc,
        # and this try/except additionally guarantees a git_ops fault (or a bare
        # MagicMock git_ops in unrelated harness tests) can never block startup.
        try:
            await self.git_ops.disable_shared_repo_auto_maintenance()
        except Exception as e:
            logger.warning(
                'Failed to disable shared-repo auto-gc/maintenance at startup '
                '(non-fatal, auto-gc left enabled): %s', e,
            )

        # 0a. Create event store and run store for this run
        import uuid

        run_id = f'run-{uuid.uuid4().hex[:12]}'
        self._run_id = run_id
        db_path = self.config.project_root / 'data' / 'orchestrator' / 'runs.db'
        try:
            self.event_store = EventStore(db_path, run_id)
            self.scheduler.event_store = self.event_store
        except Exception:
            logger.warning('Failed to create event store', exc_info=True)

        # 0a. Create run store and register this run immediately so
        # task_results can be written incrementally as tasks complete.
        try:
            self._run_store = RunStore(db_path)
            self._run_store.start_run(
                run_id,
                self.config.fused_memory.project_id,
                self.report.started_at,
                str(prd_path) if prd_path else '',
            )
        except Exception:
            logger.warning('Failed to create run store', exc_info=True)

        # 0a-post. Restore scheduler pause state from prior run (if any).
        await self._load_persisted_scheduler_pause()

        # 0b. Create cost store (shares runs.db with EventStore/RunStore)
        try:
            self.cost_store = CostStore(db_path)
            await self.cost_store.open()
        except Exception:
            logger.warning('Failed to create cost store', exc_info=True)

        # Wire cost store into usage gate for cap/failover/resume events
        if self.usage_gate and self.cost_store:
            self.usage_gate._cost_store = self.cost_store
            self.usage_gate._project_id = self.config.fused_memory.project_id
            self.usage_gate._run_id = run_id

        # Wire cost store into review checkpoint for review invocation costs
        if self.review_checkpoint and self.cost_store:
            self.review_checkpoint.cost_store = self.cost_store
            self.review_checkpoint.run_id = run_id
            # Wire the event store too (task η) so the deep_reviewer route
            # resolution emits a routing_decision event, matching the
            # post-construction attribute-wiring pattern above.
            self.review_checkpoint.event_store = self.event_store

        # Hoisted out of the try block so the finally clause can cancel
        # in-flight workflow tasks even if an exception fires before the
        # main loop creates them.
        active: set[asyncio.Task] = set()
        task_reports: list[TaskReport] = []

        try:
            # 1. Start fused-memory HTTP server
            logger.info('Starting fused-memory HTTP server...')
            await self.mcp.start()

            # 1b. Start every background-loop/service lifecycle in one
            # ordered ladder (task 2241, W10-η — PRD §5.3 LR-1/2/3):
            # escalation-server, merge-worker, offline-lane, then the seven
            # sleep-first sweeps. escalation-server + merge-worker start
            # first (canonical order), so the recovery block immediately
            # below — which depends on both being live — is unaffected by
            # collapsing the eleven scattered _start_* calls into this one
            # seam. _build_lifecycle_registry() is idempotent: a test may
            # have already pre-built (and monkeypatched) self._lifecycle.
            self._build_lifecycle_registry()
            assert self._lifecycle is not None
            await self._lifecycle.start_all()

            # 1c. Dismiss stale escalations from prior runs (non-fatal)
            try:
                await self._dismiss_stale_escalations()
            except Exception as e:
                logger.warning(f'Failed to dismiss stale escalations: {e}')

            # 1c0-dirty. Dirty project_root no longer refuses to start (task
            # 2380); file (or refresh) a deferred born-at-L2 escalation
            # instead. Runs after _dismiss_stale_escalations, which only
            # touches L0, so a prior run's L2 survives for dedup here.
            # Own try/except (non-fatal), like every neighboring startup
            # step, so a fault in the escalation path itself (e.g. a
            # transient git subprocess/lock failure in
            # has_dirty_working_tree(), or an fsync/rename OSError in
            # _escalation_queue.submit()) never aborts startup and
            # recreates the RCA 2026-07-08 crash-loop this task exists to
            # eliminate.
            try:
                await self._file_dirty_tree_escalation(force_dirty_start)
            except Exception as e:
                logger.warning(f'Failed to file dirty-tree escalation: {e}')

            # 1c0-config-keys. Surface the unknown-config-key census (task 2989)
            # as a born-at-L2 escalation so a phantom key that pydantic's
            # extra='ignore' silently dropped (the 2026-07-22 spare_warm_lanes
            # incident) can never again vanish unnoticed.  Own try/except
            # (non-fatal), like every neighboring startup step and matching the
            # dirty-tree guard above, so a fault here never aborts startup.  The
            # method is itself fail-open; this is defense in depth.
            try:
                await self._file_config_unknown_keys_escalation()
            except Exception as e:
                logger.warning(f'Failed to file config-unknown-keys escalation: {e}')

            # 1c0. Rehydrate merge-halt state from preserved L1s (non-fatal).
            # Must run after _dismiss_stale_escalations so we scan the
            # settled post-dismissal queue (only real L1s remain).
            try:
                self._rehydrate_merge_halt()
            except Exception as e:
                logger.warning(f'Failed to rehydrate merge halt: {e}')

            # 1c0a. Recover in-flight merge requests from the durable journal
            # (task 1772). Runs after _rehydrate_merge_halt so a halted queue
            # holds re-enqueued items rather than racing to merge them.
            _recover_report: dict = {}
            try:
                _recover_report = await self._recover_pending_merges()
            except Exception as e:
                logger.warning(f'Failed to recover pending merges: {e}')

            # 1c0a2. Reap orphaned _merge-* worktrees left by a mid-run SIGTERM
            # or a restart that wiped the in-memory ledger (task 2060, I6 leak).
            # Runs after recovery so a worktree still backing a recovered
            # in-flight merge is re-adopted rather than reaped.  Own try/except
            # so a reaper fault never blocks startup.
            try:
                await self._reap_orphaned_merge_worktrees(
                    _recover_report.get('requests', []),
                )
            except Exception as e:
                logger.warning(f'Failed to reap orphaned merge worktrees: {e}')

            # 1c0a3. Reconcile the durable LandedOutbox (task 2155, W1 γ):
            # closes the crash window between a merge advancing main and the
            # task being marked done. Runs after the reap block so a worker
            # (and its bound LandedOutbox) is fully settled first. Own
            # try/except so a reconciler fault never blocks startup.
            try:
                await self._reconcile_landed_outbox()
            except Exception as e:
                logger.warning(f'Failed to reconcile landed outbox: {e}')

            # 1c0b. File the L1 escalation for a pause restored from a prior
            # run (deferred from _load_persisted_scheduler_pause, which ran
            # before the escalation queue existed).  Placed after the stale-L0
            # dismissal + merge-halt rehydration so the queue is settled;
            # has_open_l1 dedups against a prior run's surviving L1, so the
            # operator sees one persistent L1 across the restart loop.
            self._file_restored_pause_escalation()

            # 1c1g. Unconditional interactive-worktree (_iact-*) crash-safety
            # sweep (task δ/2012). Runs once at every boot regardless of
            # warm_lane_gc_enabled — crash recovery must not wait for (or
            # depend on) the periodic cadence kill-switch. The pass is itself
            # fail-soft, so a reaper fault can never break startup.
            await self._run_interactive_worktree_reaper_pass()

            # 1c1h. Unconditional leftover verify-scope (df-verify-{tag}-*.scope)
            # crash-safety sweep (task 2829; companion to reify enabling
            # verify_use_cgroup_scope). A crash/SIGKILL can strand a transient
            # verify scope whose cgroup subtree keeps running; this reaps ONLY
            # this project's tag-scoped leftovers. Placed beside the interactive
            # reaper so it inherits the same ordering guarantee — it runs before
            # scheduler.finish_startup() and the first acquire_next, i.e. before
            # first dispatch, on every boot. The pass is itself fail-soft.
            await self._run_leftover_verify_scope_reaper_pass()

            # 1c2. Delay before task execution (escalation server already running)
            if delay_secs > 0:
                hours, rem = divmod(delay_secs, 3600)
                mins, secs = divmod(rem, 60)
                parts = []
                if hours:
                    parts.append(f'{hours}h')
                if mins:
                    parts.append(f'{mins}m')
                if secs:
                    parts.append(f'{secs}s')
                human = ' '.join(parts)
                logger.info(
                    f'Delaying task execution by {human} — '
                    f'escalation server is live on port {self.config.escalation.port}'
                )
                await asyncio.sleep(delay_secs)
                logger.info('Delay complete — resuming task execution')

            # 1d. Usage cap startup check
            if self.usage_gate:
                logger.info('Checking usage cap status...')
                await self.usage_gate.check_at_startup()
                if self.usage_gate.is_paused:
                    logger.warning(f'Usage cap already hit: {self.usage_gate.paused_reason}')
                    if not self.config.usage_cap.wait_for_reset:
                        raise RuntimeError(
                            f'Usage cap hit at startup: {self.usage_gate.paused_reason}'
                        )
                    # wait_for_reset=True: probe loop is already running,
                    # workflows will block in before_invoke until gate reopens

            # 2. PRD decomposition retired with the parse_prd / Taskmaster
            #    cutover (see plans/do-1-on-a-happy-pony.md §Cycle 2). The
            #    task tree must be populated before invoking the orchestrator
            #    — typically via planning_mode + curator from an interactive
            #    session. The --prd flag is preserved for tagging existing
            #    tasks with their source PRD so review can trace provenance.
            if prd_path is not None:
                logger.info(
                    'PRD decomposition removed — assuming task tree is '
                    'already populated. Tagging tasks with source PRD: %s',
                    prd_path,
                )
                _pre_statuses, _ = await self.scheduler.get_statuses()
                pre_ids = set(_pre_statuses.keys())
                # 2a. Tag any tasks that don't yet carry a PRD with this one.
                await self._tag_prd_metadata(prd_path, pre_ids)

            existing_statuses, err = await self.scheduler.get_statuses()
            if not existing_statuses:
                # Distinguish transport failure from genuinely empty tree.
                if err is not None:
                    raise RuntimeError(
                        f'Failed to reach fused-memory: '
                        f'{type(err).__name__}: {err}'
                    ) from err
                # Genuinely empty but reachable — normal under run-until-stopped
                # lifecycle (orchestrator cold-starts before tasks are filed).
                logger.info(
                    'get_statuses returned an empty mapping — task tree is '
                    'empty but fused-memory is reachable.'
                )
            if 'pending' not in existing_statuses.values() and not self._until_idle:
                # No pending tasks under default lifecycle: log an idle banner
                # so operators know the service is running but idling.  Under
                # --until-idle the main loop's drain check exits immediately, so
                # emitting "entering run-until-stopped idle" would be misleading.
                logger.info(
                    'No pending tasks at startup — entering run-until-stopped '
                    'idle; will pick up tasks as they are filed.'
                )

            # 2b. Tag tasks with code modules for concurrency locking
            logger.info('Tagging tasks with code modules...')
            await self._tag_task_modules(force=retag_modules)

            # 2c. Recover crashed tasks from surviving worktrees
            await self._recover_crashed_tasks()

            # 2c2. Reconcile stale branch checkouts left behind by recovery's
            # skip branches (identity-guard defer, no-plan/corrupt-plan
            # release) — runs while no task is yet dispatched, so it only
            # repairs what _recover_crashed_tasks skipped and never races a
            # live dispatch.
            await self._reconcile_lane_checkouts()

            # 2d. Reconcile stranded in-progress tasks (live-claimant-aware)
            await self._reconcile_stranded_in_progress()

            # 2e. Reap/quarantine orphan worktrees (Fix B).  Runs here — after
            # recovery + stranded-reconcile and before the first acquire_next —
            # so no task is yet dispatched and no live workflow can race the
            # sweep.  Self-gates on worktree_orphan_reaper_enabled.
            await self._reap_orphan_worktrees()
            # Diff 7: startup terminal-lane sweep (layer A backstop).
            # Runs after the orphan reaper so no task is yet dispatched — no
            # live workflow can race the release.
            await self._reconcile_terminal_lanes()

            # 2f. Eager warm-lane pool prewarm (task 2879).  Gated on BOTH
            # git.warm_lane_pool AND the new default-off git.warm_lane_prewarm
            # knob, and only when the pool actually exists.  Runs HERE — after
            # every startup reconcile sweep (which have already restored the
            # existing on-disk lanes, so prewarm's resident-skip will not
            # double-create them) and BEFORE finish_startup()/the first
            # acquire_next — so it can never race a live acquire/release.
            # prewarm_pool is itself fail-open / never-raises / idempotent; the
            # only thing that can fail on THIS side is resolving start_ref, so
            # the whole block is best-effort: a start_ref lookup failure logs
            # and continues, never wedging startup.
            if (
                self.config.git.warm_lane_pool
                and self.config.git.warm_lane_prewarm
                and self.git_ops.warm_lane_pool is not None
            ):
                try:
                    # Resolve current-main SHA as the neutral detached start_ref
                    # for every materialized lane (same idiom as the substrate
                    # gate / dispatch paths; use the configured main_branch, not
                    # a literal 'main').  Fall back to the symbolic branch name
                    # if rev-parse returns None so prewarm still runs.
                    start_ref = await self.git_ops.resolve_branch_sha(
                        self.config.git.main_branch
                    )
                    if start_ref is None:
                        start_ref = self.config.git.main_branch
                        logger.warning(
                            'warm-lane prewarm: resolve_branch_sha returned '
                            'None for %r — falling back to the symbolic branch '
                            'ref', self.config.git.main_branch,
                        )
                    logger.info(
                        'warm-lane prewarm enabled — eagerly materializing the '
                        'warm-lane pool from %s', start_ref,
                    )
                    await self.git_ops.prewarm_pool(start_ref)
                except Exception:
                    logger.warning(
                        'warm-lane prewarm: unexpected error during startup '
                        'prewarm — continuing startup (prewarm is best-effort)',
                        exc_info=True,
                    )

            # Startup reconcile sweeps above have all run; mark the Scheduler
            # started so acquire_next() may now proceed (task 2235's runtime
            # enforcement of the "sweeps run before the first tick" invariant
            # the comments above describe).  Set even under dry_run, which
            # never reaches acquire_next but should still leave the Scheduler
            # in a consistent post-startup state.
            self.scheduler.finish_startup()

            statuses, _ = await self.scheduler.get_statuses()
            self.report.total_tasks = sum(1 for s in statuses.values() if s == 'pending')
            logger.info(f'Task tree populated: {self.report.total_tasks} pending tasks')

            if dry_run:
                logger.info('Dry run — stopping after task population')
                return self.report

            # 3. Run workflow slots
            sem = asyncio.Semaphore(self.config.max_concurrent_tasks)

            while True:
                # Pick up any pending review task spawned by _collect_done_reports
                if self._pending_review_task is not None:
                    active.add(self._pending_review_task)
                    self._pending_review_task = None

                # If a review checkpoint is running, don't acquire new tasks —
                # wait for in-flight tasks (and the review) to complete.
                if self._review_running:
                    if not active:
                        break  # shouldn't happen — review task is in active
                    done, active = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED
                    )
                    self._collect_done_reports(done, task_reports)
                    continue

                # Check daily cost ceilings before dispatching the next task.
                # On breach, pause_scheduler() is called and acquire_next()
                # returns None (task-1322 machinery), draining in-flight work
                # and exiting the run loop cleanly.  Task 1323.
                await self._enforce_cost_ceilings()

                assignment = await self.scheduler.acquire_next()

                if assignment is None:
                    if not active:
                        if self.scheduler.is_paused:
                            # Paused ≠ done.  acquire_next() returns None while
                            # paused (task-1322); treating that as completion
                            # exits 0 and defeats Restart=on-failure, stranding
                            # an AFK operator behind a wedged-but-"successful"
                            # service.  Idle in-process instead so an in-process
                            # resume_scheduler() (via the still-running
                            # escalation MCP) resumes dispatch without a restart.
                            # The six background loops keep running while we idle.
                            now = time.monotonic()
                            if (
                                now - self._last_paused_idle_log
                                >= _PAUSED_IDLE_LOG_SECS
                            ):
                                logger.warning(
                                    'Scheduler paused — idling (no active '
                                    'tasks). reason=%r  Resolve the L1 '
                                    'escalation / resume_scheduler() to '
                                    'continue.',
                                    self.scheduler.pause_reason,
                                )
                                self._last_paused_idle_log = now
                            await asyncio.sleep(_PAUSED_IDLE_POLL_SECS)
                            continue
                        # Before treating as "all done", sweep stranded
                        # in-progress.  Tasks stranded by transient backend
                        # failures may unblock pending dependents; a clean
                        # exit here would leave them for the next restart.
                        # See Fix 4 — stuck-blocked recovery.
                        changed = await self._reconcile_stranded_in_progress(
                            mid_run=True,
                        )
                        if changed > 0:
                            logger.info(
                                'Mid-run reconcile freed %d task(s) '
                                '— continuing main loop', changed,
                            )
                            continue

                        # Genuine empty queue (not paused, nothing freed).
                        if self._until_idle:
                            break  # --until-idle: exit on drain (one-shot run)

                        # Run forever (default): a drained queue is NOT "done".
                        # Under systemd this service is long-lived; stop is via
                        # SIGTERM (→ CancelledError → exit 130), not queue drain.
                        # Idle and re-poll the tree so tasks scheduled after
                        # startup get picked up.
                        #
                        # Invariant: this branch is only reachable with `active`
                        # empty and no focused review in flight — _review_running
                        # / _pending_review_task serialize at the loop top, so a
                        # full-review-on-idle here can't race a focused review.
                        now = time.monotonic()
                        self._compute_tallies(task_reports)
                        if self.report.completed > 0 or task_reports:
                            # The cycle did work: emit a per-cycle summary, run a
                            # rate-limited full review, then reset for next cycle.
                            # (Dead-idle polls skip this so an empty tree doesn't
                            # spam journald with 0/0 summaries + reconcile sweeps.)
                            logger.info(self.report.summary())
                            if (self.review_checkpoint
                                    and self.config.review.full_review_on_complete
                                    and self.report.completed > 0
                                    and self.review_checkpoint.should_run_full(now)):
                                await self._run_full_review_and_tag()
                            # Reset per-cycle aggregates.  Safe to drop the old
                            # task_reports list here (see invariant above: no
                            # report is still being collected).  Recompute
                            # total_tasks cheaply so completed/total stays honest.
                            task_reports = []
                            self.report = HarnessReport(
                                started_at=datetime.now(UTC).isoformat()
                            )
                            statuses, _ = await self.scheduler.get_statuses()
                            self.report.total_tasks = sum(
                                1 for s in statuses.values() if s == 'pending'
                            )
                        elif now - self._last_idle_poll_log >= _IDLE_POLL_LOG_SECS:
                            logger.info(
                                'Idle — polling for new tasks every %.0fs '
                                '(run forever; pass --until-idle to exit on drain)',
                                self.config.idle_poll_secs,
                            )
                            self._last_idle_poll_log = now
                        # Post-merge staleness hook: restart fused-memory.service
                        # if a code-touching merge has been debounced and the
                        # orchestrator is idle (no dispatched agents = quiet window).
                        await self._maybe_restart_stale_service(agents_idle=True)
                        # Fleet-common merge-idle heartbeat (task 2395, α): written
                        # in both rest branches so a saturated unit (steady-state
                        # busy-wait branch below) still heartbeats.  Does NOT cover
                        # a unit that is continuously dispatching with spare
                        # semaphore headroom (never idle, never busy-waiting) — see
                        # the docstring of _write_merge_heartbeat for why that gap
                        # is accepted rather than closed with an unconditional
                        # per-tick call.
                        await self._write_merge_heartbeat()
                        await asyncio.sleep(self.config.idle_poll_secs)
                        continue
                    # Wait for any active task to complete, then retry.
                    # Timeout ensures newly-added tasks are discovered
                    # within 15s even when no running task completes.
                    done, active = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED,
                        timeout=15,
                    )
                    self._collect_done_reports(done, task_reports)
                    # Leaf services (dashboard, require_idle=False) restart
                    # promptly after their diff lands even while agents are
                    # dispatching.  fused-memory (require_idle=True) no-ops here
                    # and is reserved for the idle branch above — the two
                    # branches are mutually exclusive per tick and maybe_restart
                    # clears pending on fire, so there is no double-fire.
                    await self._maybe_restart_stale_service(agents_idle=False)
                    # Fleet-common merge-idle heartbeat (task 2395, α): a
                    # saturated unit steady-states in this busy-wait branch, so
                    # it must heartbeat here too — see the idle branch above.
                    await self._write_merge_heartbeat()
                    continue

                await sem.acquire()
                self._task_modules[assignment.task_id] = list(assignment.modules)
                # Register escalation wake-event BEFORE create_task so Fix #1a
                # (_on_escalation_resolved orphan-flip gate) sees task_id in
                # _escalation_events immediately — closing the sub-second race
                # where a resolving escalation would mis-classify a just-
                # dispatched task as an orphan and double-flip it.
                self._register_escalation_event(assignment.task_id)
                task = asyncio.create_task(
                    self._run_slot(assignment, sem),
                    name=f'workflow-{assignment.task_id}',
                )
                active.add(task)
                task.add_done_callback(active.discard)
                # Guard against the narrow resource-leak window where the slot
                # task is cancelled before _run_slot's body begins executing
                # (e.g. hard_cancel_workflow or shutdown sweeping pending tasks
                # before the event loop schedules the coroutine's first step).
                # In that case _run_slot's finally block never runs and
                # _escalation_events[task_id] would leak — permanently
                # satisfying the active-workflow gate and suppressing both the
                # stranded_blocked re-file and Fix #1a orphan-flip for that
                # task forever.  The done_callback fires unconditionally when
                # the Task reaches any terminal state; pop() is idempotent (a
                # no-op if _run_slot's finally already removed the entry).
                _tid = assignment.task_id
                task.add_done_callback(
                    lambda _t, tid=_tid: self._escalation_events.pop(tid, None)
                )

            # Drain remaining
            if active:
                done, _ = await asyncio.wait(active)
                self._collect_done_reports(done, task_reports)

            self._compute_tallies(task_reports)

            # 3b. Optional full review after all tasks complete.  This is the
            # exit / --until-idle path: the gate is unconditional (no
            # should_run_full rate-limit — that ceiling applies only to the
            # run-forever idle path, which can fire repeatedly).
            if (self.review_checkpoint
                    and self.config.review.full_review_on_complete
                    and self.report.completed > 0):
                await self._run_full_review_and_tag()

        finally:
            # 4. Shutdown
            # 4a. Cancel any in-flight workflow tasks BEFORE shutting down
            # usage_gate — otherwise a cap-hit in a still-running agent can
            # spawn a fresh probe task via _handle_cap_detected AFTER
            # usage_gate.shutdown() has drained the existing ones, leaving
            # the event loop alive forever.
            if active:
                logger.info(f'Cancelling {len(active)} active workflow task(s)')
                for t in active:
                    t.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*active, return_exceptions=True),
                        timeout=15.0,
                    )
                except TimeoutError:
                    logger.error('Workflow tasks did not drain within 15s')
                active.clear()

            self.report.completed_at = datetime.now(UTC).isoformat()

            # Finalize run metrics in SQLite (task_results were written
            # incrementally in _collect_done_reports; this updates the
            # runs row with final aggregates).
            if self._run_store and self._run_id:
                try:
                    self._run_store.finish_run(self._run_id, self.report)
                except Exception as e:
                    logger.warning(f'Failed to finalize run metrics: {e}')

            # Checkpoint all SQLite WAL files while the event loop is still
            # healthy — truncates the WAL so a subsequent crash cannot leave
            # the stores in a half-written state.  Best-effort: failures are
            # logged as warnings but never block shutdown.
            self._checkpoint_stores()

            # Save HarnessReport alongside review checkpoint reports
            if self.report.review_checkpoints > 0:
                self._save_harness_report()

            if self.usage_gate:
                if self.usage_gate.total_pause_secs > 0:
                    self.report.paused_for_cap = True
                    self.report.cap_pause_duration_secs = self.usage_gate.total_pause_secs
                try:
                    await self.usage_gate.shutdown()
                except Exception as e:
                    logger.warning(f'usage_gate.shutdown() failed: {e}')
            if self.cost_store:
                try:
                    await self.cost_store.close()
                except Exception as e:
                    logger.warning(f'cost_store.close() failed: {e}')
            # Stop every registered background-loop/service lifecycle in one
            # bounded, reverse-registration-order ladder (task 2241, W10-η —
            # PRD §5.3 LR-1/2/3). LifecycleRegistry.stop_all() already
            # bounds each service's stop() in asyncio.wait_for and catches +
            # logs both timeouts and plain exceptions internally, so one
            # wedging or failing service can never abort the ladder or hang
            # shutdown — the structural elimination of the shutdown-hang
            # class this module exists to fix. self._lifecycle may be None
            # if run() raised before startup ever built it.
            if self._lifecycle is not None:
                await self._lifecycle.stop_all()
            try:
                await self.mcp.stop()
            except Exception as e:
                logger.warning(f'mcp.stop() failed: {e}')

            # 4b. Last-resort straggler sweep — catches any task the named
            # cleanup above missed (orphan probe tasks, cost-event
            # fire-and-forgets, sub-tasks spawned by merge/escalation stop).
            current = asyncio.current_task()
            stragglers = [
                t for t in asyncio.all_tasks()
                if t is not current and not t.done()
            ]
            if stragglers:
                names = [t.get_name() for t in stragglers]
                logger.warning(
                    f'Cancelling {len(stragglers)} straggler task(s): {names}'
                )
                for t in stragglers:
                    t.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*stragglers, return_exceptions=True),
                        timeout=5.0,
                    )
                except TimeoutError:
                    still = [t.get_name() for t in stragglers if not t.done()]
                    logger.error(f'Stragglers did not die within 5s: {still}')

            # Release singleton lock
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None

        logger.info(self.report.summary())
        return self.report

    async def _tag_prd_metadata(self, prd_path: Path, pre_parse_ids: set[str]) -> None:
        """Tag tasks with the PRD they were created from."""
        resolved_prd = str(prd_path.resolve())
        tasks = await self.scheduler.get_tasks(statuses=ACTIVE_TASK_STATUSES)
        new_ids = {str(t.get('id', '')) for t in tasks} - pre_parse_ids

        tagged = 0
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            metadata = t.get('metadata') or {}
            if metadata.get('prd'):
                continue  # already tagged
            if new_ids and tid not in new_ids:
                continue  # existed before parse_prd
            await self.scheduler.update_task(tid, {'prd': resolved_prd})
            tagged += 1

        if tagged:
            logger.info(f'Tagged {tagged} tasks with PRD: {resolved_prd}')

    async def _tag_task_modules(self, force: bool = False) -> None:
        """Invoke a Claude agent to tag each task with the files it touches.

        Uses structured output to get a JSON mapping of task_id → [files],
        then persists via scheduler.update_task() as ``metadata.files``.

        When *force* is ``True``, retag all non-done/cancelled tasks even if
        they already have file metadata.
        """
        tasks = await self.scheduler.get_tasks(statuses=ACTIVE_TASK_STATUSES)

        skip_statuses = {'done', 'cancelled'}
        untagged = []
        for t in tasks:
            if t.get('status') in skip_statuses:
                continue
            metadata = t.get('metadata') or {}
            if metadata.get('task_kind') == 'deterministic':
                # Deterministic tasks carry no worktree and no code (CLAUDE.md
                # "Deterministic task kind"), so a file-lock prediction is
                # meaningless for them — exclude them in every mode, including
                # a force-retag.
                continue
            if not force and (metadata.get('files') or metadata.get('files_tagged_at')):
                continue
            untagged.append(t)

        if not untagged:
            logger.info('No tasks to tag — skipping')
            return

        if force:
            logger.info(f'Force-retagging {len(untagged)} tasks with file metadata')

        # Get top-level directory listing for context
        try:
            entries = sorted(p.name for p in self.config.project_root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        except OSError:
            entries = []

        task_summaries = []
        for t in untagged:
            task_summaries.append({
                'id': str(t.get('id', '')),
                'title': t.get('title', ''),
                # Fall back to details when description is empty so the
                # tagger still has context to predict from.
                'description': t.get('description') or t.get('details') or '',
            })

        # Prompt/schema/system-prompt live in the shared module_tagger_prompt
        # module (task 2540) so the offline replay trial reuses byte-identical
        # production inputs. Behavior-preserving extraction.
        prompt = build_tagger_prompt(entries, task_summaries)

        # Route resolution (task η): resolve model/max_turns/budget through the
        # single layered resolver + emit a routing_decision event. This is a
        # BATCH tag over many untagged tasks (task_id intentionally omitted), so
        # no task_id/scheduler/in_memory_task is passed → event only, no
        # metadata.routing mirror. effort is deliberately NOT wired into the
        # invoke below: the site never passed it, and invoke_agent(effort=None)
        # emits no model_reasoning_effort flag, so wiring config.effort.
        # module_tagger would be a real behaviour change (plan design decision
        # 4). decision.effort is still carried in the routing_decision event.
        decision = await resolve_and_record_route(
            role_name='module_tagger',
            role_defaults=RoleDefaults(
                self.config.models.module_tagger,
                self.config.effort.module_tagger,
                self.config.budgets.module_tagger,
                self.config.max_turns.module_tagger,
            ),
            config=self.config,
            event_store=self.event_store,
            cost_store=self.cost_store,
        )
        try:
            result = await invoke_with_cap_retry(
                usage_gate=self.usage_gate,
                label='Module tagging',
                cost_store=self.cost_store,
                run_id=self._run_id or '',
                # task_id intentionally omitted: this invocation isn't
                # scoped to one task, and invoke_with_cap_retry's own
                # `task_id or None` normalization turns the default '' into
                # a NULL invocations.task_id row, matching the module_tagger
                # fixture in test_digest.py's rollup tests.
                project_id=self.config.fused_memory.project_id,
                role='module_tagger',
                invoke_fn=invoke_agent,
                prompt=prompt,
                system_prompt=TAGGER_SYSTEM_PROMPT,
                cwd=self.config.project_root,
                model=decision.model,
                max_turns=decision.max_turns,
                max_budget_usd=decision.budget_usd,
                output_schema=TAGGER_SCHEMA,
            )
        except AllAccountsCappedException as e:
            logger.warning(
                f'Module tagging skipped: all accounts capped '
                f'({e.retries} retries in {e.elapsed_secs:.1f}s)'
            )
            return

        if not result.success:
            logger.warning(f'Module tagger agent failed: {result.output[:200]}')
            return

        # Parse the structured output. _extract_tagger_entries peels known
        # StructuredOutput wrapper keys ('predictions', legacy 'tasks') up to
        # a bounded depth, so it accepts the current flat {"predictions": [...]}
        # shape, the legacy single-wrap {"tasks": [...]}, and the legacy
        # double-wrap {"tasks": {"tasks": [...]}} produced when the tool
        # param name collided with the schema's old 'tasks' key (defect 3).
        payload = result.structured_output
        if not payload:
            try:
                payload = json.loads(result.output)
            except (json.JSONDecodeError, TypeError):
                logger.warning('Module tagger produced no parseable output')
                return

        # Index predictions by task id. A missing/empty files list normalizes
        # to [] so a task the agent predicted no files for is still
        # indexable (defect 1) rather than absent from the mapping.
        pred_by_id = {
            str(e.get('id', '')): (e.get('files') or [])
            for e in _extract_tagger_entries(payload)
            if isinstance(e, dict) and e.get('id') is not None
        }

        # Stamp files_tagged_at for EVERY task in the untagged batch — not
        # just the ones present in the agent's response — so a task the
        # agent can't (or won't) predict files for still gets marked as
        # processed and never re-enters the tagging batch on the next cycle
        # (defect 1: an LLM-spend leak — 73 tagger sessions mined in one
        # month). The 'files' key is written ONLY when the prediction
        # sanitizes to a NON-EMPTY file-level list; otherwise the sentinel is
        # written alone, and scheduler.update_task's default merge mode
        # preserves any pre-existing real files rather than clobbering them.
        # This covers three no-clobber cases: the agent omits the task
        # (files == []), predicts an explicit empty list, OR predicts only
        # directory-shaped paths (sanitize strips them to []) — the last of
        # which is why the gate keys off the SANITIZED result, not the raw
        # (possibly all-directory but truthy) prediction.
        tagged_at = datetime.now(UTC).isoformat()

        tagged_count = 0
        for t in untagged:
            task_id = str(t.get('id', ''))
            if not task_id:
                continue
            files = pred_by_id.get(task_id, [])
            metadata_payload: dict[str, Any] = {'files_tagged_at': tagged_at}
            # Persist file-level paths only: strip directory-shaped entries
            # before writing so the lock-charter guard on update_task /
            # task_interceptor.update_task accepts the write. Without this,
            # any LLM-tagged directory entry makes update_task reject the
            # ENTIRE payload (LockCharterViolation), silently dropping the
            # valid file-level entries too. Consistent with the strip in
            # _persist_files_metadata / _reconcile_metadata_files_for_done.
            # An all-directory prediction sanitizes to [] and is treated
            # exactly like an empty/omitted one (sentinel alone, no clobber).
            sanitized = sanitize_files_for_persist(files) if files else []
            if sanitized:
                metadata_payload['files'] = sanitized
            await self.scheduler.update_task(task_id, json.dumps(metadata_payload))
            if sanitized:
                # Populate in-memory cache via the single cache-writing seam.
                # module_charter.derive_modules (called inside seed_modules)
                # applies the α strip so a directory-only charter cannot
                # poison the cache and bypass _get_modules' α strip on the
                # next tick. Seed with the RAW predicted files (the α strip
                # lives inside derive_modules); seeding is skipped whenever
                # the sanitized result is empty, so tagged_count reflects only
                # tasks that actually received file-level locks.
                self.scheduler.seed_modules(task_id, files)
                tagged_count += 1

        logger.info(f'Tagged {tagged_count}/{len(untagged)} tasks with file metadata')
        logger.info(f'Module cache has {len(self.scheduler._module_cache)} entries')
        if self.scheduler._module_cache:
            sample = dict(list(self.scheduler._module_cache.items())[:3])
            logger.info(f'Module cache sample: {sample}')

    def _resolve_recovery_artifact(self, entry: Path, name: str) -> Path:
        """Resolve a `.task` artifact path new-then-old (W11 delta relocation).

        Prefers ``TaskArtifacts.meta_root_for(worktree_base, entry.name) /
        name`` (the new ``.task-meta`` sibling location, W11 beta), falling
        back to the legacy ``entry / '.task' / name`` when the new path is
        absent.  Mirrors ``TaskArtifacts._read_path``'s new-then-old
        contract and ``GitOps._find_lane_by_plan_task_id``'s plan.json scan,
        for the ``_recover_crashed_tasks`` reads that don't go through a
        ``TaskArtifacts`` instance.
        """
        new_path = TaskArtifacts.meta_root_for(self.git_ops.worktree_base, entry.name) / name
        if new_path.exists():
            return new_path
        return entry / '.task' / name

    def _clear_recovery_artifact(self, entry: Path, name: str) -> None:
        """Remove a `.task` artifact from BOTH the new and legacy roots.

        The write-side mirror of :meth:`_resolve_recovery_artifact` — mirrors
        ``TaskArtifacts._clear_path``: clearing only the resolved path risks
        the OTHER root resurrecting it via the new-then-old read fallback on
        a later call.  Idempotent (``missing_ok=True``); never raises.
        """
        (TaskArtifacts.meta_root_for(self.git_ops.worktree_base, entry.name) / name).unlink(
            missing_ok=True
        )
        (entry / '.task' / name).unlink(missing_ok=True)

    def _adopt_recovered_session(self, entry: Path, task_id: str | None) -> str | None:
        """Best-effort adopt an `agent_session.json` sidecar into
        ``_recovered_sessions`` (task 2772, session-resume beta).

        Resolves the sidecar via :meth:`_resolve_recovery_artifact` (INV-5 —
        reuse the resolver rather than duplicating new-then-old path logic)
        and reads it as a RAW dict: ``_run_slot`` pops
        ``_recovered_sessions[assignment.task_id]`` and passes it straight
        through as ``resume_session_id`` (typed ``dict | None``), and
        ``TaskWorkflow.__init__`` consumes it via
        ``.get('session_id')/.get('role')/.get('resume_count')`` — there is
        no ``TaskArtifacts`` instance during recovery to read a typed
        ``AgentSession`` through.

        Keying: when *task_id* is given (the plan-present sites, keyed by
        the plan-derived recovery id) it is used directly — this covers both
        a v2 sidecar (whose own ``task_id`` should equal it) and a v1
        sidecar (which has no ``task_id`` of its own, B11 fallback). When
        *task_id* is ``None`` (the no-plan lane site, which has no
        plan-derived id) falls back to the sidecar's own v2 ``task_id``. If
        neither yields a usable key (a v1 sidecar on a no-plan lane) — or the
        sidecar is missing/unreadable — nothing is adopted and ``None`` is
        returned. Never raises.

        Returns the adopted key, or ``None`` if nothing was adopted.
        """
        sidecar_path = self._resolve_recovery_artifact(entry, 'agent_session.json')
        if not sidecar_path.exists():
            return None
        try:
            session_data = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                'Recovery: %s sidecar unreadable (%s) — not adopting session',
                entry.name, e,
            )
            return None
        key = task_id if task_id is not None else session_data.get('task_id')
        if not key:
            return None
        key = str(key)
        self._recovered_sessions[key] = session_data
        # Best-effort: stash the surviving worktree's claude-config dir so the
        # _run_slot guard (task γ) can RE-glob the transcript at dispatch. The
        # dir name embeds the branch (``claude-config-<branch>``), not derivable
        # from task_id at the pre-acquire dispatch point, so *entry* — the
        # surviving worktree, known only here — is the last place to capture it.
        # Never raises: a missing/globless .task simply leaves no stash, which
        # the guard treats as 'no_transcript' (fail-safe fresh dispatch, I3).
        try:
            config_dirs = sorted((entry / '.task').glob('claude-config-*'))
            if config_dirs:
                if len(config_dirs) > 1:
                    # >1 claude-config-<branch> dir in a single surviving
                    # worktree is abnormal: the lexically-first pick may not be
                    # the one holding THIS session's transcript, in which case
                    # the dispatch-time re-glob degrades to a 'no_transcript'
                    # fresh dispatch. Warn (loud-over-silent) so that otherwise
                    # silent missed resume is observable.
                    logger.warning(
                        'Recovery: %s has %d claude-config dirs %s — stashing the '
                        'lexically-first (%s) for session %s; if it lacks the '
                        'transcript the resume degrades to fresh dispatch',
                        entry.name, len(config_dirs),
                        [str(d) for d in config_dirs], config_dirs[0],
                        session_data.get('session_id'),
                    )
                self._recovered_session_config_dirs[key] = str(config_dirs[0])
        except OSError as e:
            logger.debug(
                'Recovery: %s config-dir glob failed (%s) — guard will treat '
                'the recovered session as uncorroborated', entry.name, e,
            )
        logger.info(
            'Recovery: adopting agent session for task %s (role=%s, '
            'session_id=%s) — will --resume on re-dispatch',
            key, session_data.get('role'), session_data.get('session_id'),
        )
        return key

    def _session_resume_eligible(
        self, session: dict, config_dir: str | None
    ) -> tuple[bool, str]:
        """Return ``(eligible, reason)`` for a recovered session (task γ).

        The PRD §7 eligibility predicate, evaluated in _run_slot BEFORE the
        β resume injection. Totally fail-safe (I3): every ambiguous or broken
        input degrades to an ineligible ``(False, <reason>)`` so the caller
        falls back to a fresh dispatch — this method NEVER raises.

        Reasons (checked in this order):
          - 'disabled'      — the session_resume kill switch is off (B6).
          - 'stale'         — (now - started_at) >= freshness_window_secs, OR
                              started_at is missing/unparseable (fail-safe).
          - 'capped'        — resume_count >= max_resumes_per_task (B7).
          - 'no_transcript' — no stashed config_dir, no session_id, or the
                              transcript is absent on disk (B4 reseed/wipe).
          - 'eligible'      — all corroboration passed; inject the session.
        """
        cfg = self.config.session_resume
        if not cfg.enabled:
            return (False, 'disabled')
        # Freshness — any parse failure or absent started_at is 'stale'.
        try:
            started_at = datetime.fromisoformat(session['started_at'])
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            age_secs = (datetime.now(UTC) - started_at).total_seconds()
            if age_secs >= cfg.freshness_window_secs:
                return (False, 'stale')
        except (KeyError, ValueError, TypeError):
            return (False, 'stale')
        # Per-task resume cap (throttling of a healthy long-running task).
        try:
            resume_count = int(session.get('resume_count', 0))
        except (ValueError, TypeError):
            resume_count = 0
        if resume_count >= cfg.max_resumes_per_task:
            return (False, 'capped')
        # Transcript corroboration — RE-glob at dispatch (INV-3), so a
        # reseed/wipe of .task between boot and re-dispatch is detected.
        # transcript_exists is itself total (any glob error → False), so no
        # outer try/except is needed here to uphold this method's I3 totality:
        # a non-empty config_dir str makes Path() safe, and a missing/absent
        # transcript degrades to the 'no_transcript' fallback below.
        session_id = session.get('session_id')
        if not config_dir or not session_id:
            return (False, 'no_transcript')
        if not transcript_exists(Path(config_dir), session_id):
            return (False, 'no_transcript')
        return (True, 'eligible')

    async def _recover_crashed_tasks(self) -> None:
        """Scan surviving worktrees and recover plans with completed work.

        For each worktree in the worktree base directory:
        - If it has a plan.json with completed steps, store the plan for
          injection into the resumed workflow.
        - Otherwise, clean up the worktree (no useful work to recover).

        Also resets any in-progress tasks to pending so acquire_next() picks
        them up.
        """
        worktree_base = self.git_ops.worktree_base
        if not worktree_base.exists():
            return

        # Pool-storage guard (task 2099): worktree_base can EXIST as an
        # unmounted mountpoint dir, making every mount-resident worktree
        # look planless/corrupt to the scan below — the Jul-3 incident this
        # guards against. Defer the ENTIRE recovery pass (no cleanup) rather
        # than destroy recoverable plans that merely appear empty because
        # the mount has not come up yet. Gated on pool_in_use() (step-16
        # review-fix): pool_storage_present() is permanently False on a
        # pool-less host (its only writer never runs without a configured
        # pool) — e.g. worktree_base.exists() is already True there from a
        # prior COLD worktree — so without this gate every pool-less host
        # would defer recovery and file a spurious escalation at every
        # startup.
        if self.git_ops.pool_in_use() and not self.git_ops.pool_storage_present():
            logger.warning(
                'Crash recovery: pool storage absent/unmounted at %s — '
                'deferring the ENTIRE recovery pass (no cleanup) to avoid '
                'destroying recoverable plans on mount-resident worktrees',
                worktree_base,
            )
            self._file_pool_storage_absent_escalation()
            return

        recovered = 0
        cleaned = 0

        # W11 delta: keep the durable LaneLifecycle record and the pool's
        # in-memory cache from ever drifting (PRD dec.3, I1) — every
        # restore_assignment/note_assignment call below (both the new
        # record-driven adopt path and the pre-existing heuristic path)
        # now mirrors onto the durable record.
        top_pool = self.git_ops.warm_lane_pool
        if top_pool is not None:
            top_pool.set_lane_lifecycle(self.git_ops._lane_lifecycle)

        for entry in worktree_base.iterdir():
            if not entry.is_dir():
                continue
            pool = self.git_ops.warm_lane_pool
            spec_pool = self.git_ops.spec_warm_lane_pool
            # '_spec-' merge-speculation lanes are a SEPARATE pool: they have
            # no '.task/plan.json' and their dir name is not a live task id,
            # so without this they would hit the mid-crash cold cleanup branch
            # and be removed.  Treat them as lanes too — they take the no-plan
            # branch (no recoverable task identity) and route through
            # cleanup_worktree, which now releases '_spec-' lanes back to the
            # spec pool instead of removing them.
            is_lane = (pool is not None and pool.is_lane(entry)) or (
                spec_pool is not None and spec_pool.is_lane(entry)
            )
            task_id = entry.name

            # ── C2 namespace invariant (task 2925, merge-worktree-lifecycle
            # -integrity PRD §4) ───────────────────────────────────────────
            # is_lane is checked FIRST above: adoptable warm/spec lanes are
            # `_`-prefixed, so they MUST bypass the classifier (which would
            # label them 'infra') and keep their adopt/release/quarantine
            # handling below.  For every NON-lane entry, the positive-match
            # classifier replaces the old NEGATIVE per-name exclusion lists
            # (D7): only task-id-shaped names reach the plan.json/cleanup
            # heuristic.  `_merge-*` and other `_`/`.`-prefixed infra bands
            # are SKIPPED with an EXPLICIT journal line (never the silent
            # 5326 "Cleaned up worktree _merge-verify" force-removal): the
            # merge reaper (`_reap_orphaned_merge_worktrees`) owns the
            # `_merge-*` disposition; every other infra band is left to its
            # owner.  This subsumes the former dedicated `.lane-state`/
            # `.task-meta` (LANE_STATE_DIRNAME/TASK_META_DIRNAME) skip —
            # both are `.`-prefixed => classified 'infra'.
            if not is_lane:
                worktree_class = classify_worktree_entry(entry.name)
                if worktree_class == 'merge':
                    logger.info(
                        'Recovery: %s is a merge worktree (infra) — reporting '
                        'to the merge reaper, never cleaned by the '
                        'crash-recovery sweep',
                        entry.name,
                    )
                    continue
                if worktree_class == 'infra':
                    logger.info(
                        'Recovery: %s is infra-owned (C2 namespace) — left to '
                        'its owner',
                        entry.name,
                    )
                    continue
                # 'task' falls through unchanged to the heuristic below.

            # ── Record-driven recovery (W11 delta, PRD mechanism 1) ─────
            # Consult the durable LaneLifecycle record FIRST, before any of
            # the plan.json heuristics below: a record in ASSIGNED/IN_USE is
            # the durable source of truth for "this lane belongs to this
            # task". A terminal task releases the lane outright (see below).
            # Otherwise adopt only on an EXACT git-reality match (still a
            # registered worktree AND — when resolvable — its checked-out
            # branch matches the record); ANY divergence (orphaned admin
            # entry OR a stale-branch collision) quarantines instead — never
            # adopt-on-doubt, never silently re-pin. ANY other rec.state
            # (including no record at all — the pre-W11 compat case) falls
            # through unchanged to the existing heuristic path below.
            if is_lane and pool is not None:
                rec = self.git_ops._lane_lifecycle.read(entry)
                if rec is not None and rec.state == DurableLaneState.QUARANTINED:
                    # Already quarantined (e.g. a previous quarantine move
                    # failed partway and left the dir behind) — skip
                    # entirely: no plan recovery, no cleanup, no re-adopt.
                    logger.info(
                        'Recovery: lane %s already QUARANTINED — skipping',
                        entry.name,
                    )
                    continue
                if (
                    rec is not None
                    and rec.state in (DurableLaneState.ASSIGNED, DurableLaneState.IN_USE)
                    and rec.task_id is not None
                ):
                    # Terminal-task release (T10 amplifier fix, mirrored onto
                    # the record-driven path): resolved BEFORE the
                    # git-reality adopt/quarantine decision below, so a
                    # terminal task's lane is released even when registration
                    # and branch both still check out fine — otherwise every
                    # restart would re-pin a dead task's lane forever,
                    # shrinking the pool. A transient/None status falls
                    # through to the git-reality check (safe default; layer A
                    # self-heals the lane on the next reconcile interval).
                    term_status = await self.scheduler.get_status(rec.task_id)
                    if term_status in ('done', 'cancelled'):
                        logger.info(
                            'Recovery: lane %s record ASSIGNED for task %s '
                            'but task is terminal (%s) — releasing, not '
                            'adopting',
                            entry.name, rec.task_id, term_status,
                        )
                        await self.git_ops.cleanup_worktree(entry, rec.task_id)
                        # Guarded, not unconditional (step-16 review-fix): for
                        # a warm lane, cleanup_worktree already routed through
                        # release_warm_lane -> pool.release ->
                        # _note_released_durable (task 2986, single writer),
                        # which records the ASSIGNED/IN_USE -> RELEASED edge
                        # itself (with the same guard). Re-issuing the
                        # transition unconditionally would then attempt an
                        # illegal RELEASED -> RELEASED edge and raise
                        # IllegalLaneTransition uncaught, aborting recovery
                        # for every remaining lane. Only finalize RELEASED
                        # here when cleanup did NOT already write it (e.g. a
                        # routing path other than release_warm_lane).
                        rec2 = self.git_ops._lane_lifecycle.read(entry)
                        if rec2 is not None and rec2.state in (
                            DurableLaneState.ASSIGNED, DurableLaneState.IN_USE,
                        ):
                            self.git_ops._lane_lifecycle.transition(
                                entry, DurableLaneState.RELEASED,
                            )
                        cleaned += 1
                        continue

                    try:
                        is_registered = await self.git_ops._is_registered_worktree(entry)
                    except OSError as e:
                        # Fail-safe: an unreachable git command must not be
                        # read as conclusively "orphaned" (mirrors the
                        # transient-safe default used elsewhere in this
                        # method).
                        logger.warning(
                            'Recovery: registration check raised for lane %s '
                            'task %s (%s) — restoring pin as safe default',
                            entry.name, rec.task_id, e,
                        )
                        is_registered = True
                    registered_branch = None
                    checkouts = await self.git_ops.lane_branch_checkouts()
                    if checkouts:
                        for bare_id, checkout_lane in checkouts.items():
                            if checkout_lane == entry:
                                registered_branch = (
                                    f'{self.git_ops.config.branch_prefix}{bare_id}'
                                )
                                break
                    # Fail-safe on an unresolvable read: lane_branch_checkouts()
                    # returns None when the pool is disabled or `git worktree
                    # list` errors (its documented contract — "never mass-mutate
                    # on an unreliable read"). A transient git hiccup must NOT be
                    # read as a branch divergence; otherwise every assigned lane
                    # carrying a branch record (the normal case, `task/<id>`)
                    # would be quarantined and its recovered plan dropped. Only
                    # treat a branch mismatch as divergence when the checkout map
                    # was actually obtained (checkouts is a dict — possibly empty,
                    # which IS a genuine "lane absent from checkouts" divergence).
                    # This mirrors the is_registered OSError fail-safe above.
                    branch_ok = (
                        rec.branch is None
                        or checkouts is None
                        or registered_branch == rec.branch
                    )
                    if is_registered and branch_ok:
                        # ── Semantic identity guard (Fix C, mirrored) ─────
                        # The checks above only prove this lane's admin entry
                        # and checked-out branch still belong to rec.task_id.
                        # For a RECYCLED id, a stale record could carry a
                        # task_id/branch pair that happens to match current
                        # git reality while the worktree's actual content
                        # belongs to a different (deleted) task — exactly the
                        # failure mode reify 3770 introduced this check to
                        # catch on the heuristic path below (~2322). Run the
                        # same check here, before pinning.
                        if self.config.worktree_identity_guard_enabled:
                            live = await self.scheduler.get_task(rec.task_id)
                            if live is None:
                                # get_task returns None for BOTH "deleted" and
                                # transient error — defer (no adopt, no
                                # destroy), same policy as the heuristic path.
                                logger.warning(
                                    'Recovery: lane %s record ASSIGNED for '
                                    'task %s but no live DB task — deferring '
                                    '(no adopt, no destroy)',
                                    entry.name, rec.task_id,
                                )
                                continue
                            stored_title = read_worktree_title(entry)
                            if not identities_match(stored_title, live.get('title')):
                                logger.warning(
                                    'Recovery: lane %s record ASSIGNED for '
                                    'task %s but identity MISMATCH — stored '
                                    'title %r != live %r; quarantining',
                                    entry.name, rec.task_id, stored_title,
                                    live.get('title'),
                                )
                                dest = await self.git_ops.quarantine_worktree(
                                    entry, rec.branch or rec.task_id,
                                    'recovery-identity-mismatch',
                                )
                                self.git_ops._lane_lifecycle.transition(
                                    entry, DurableLaneState.QUARANTINED,
                                )
                                if self.event_store:
                                    self.event_store.emit(
                                        EventType.worktree_quarantined,
                                        task_id=rec.task_id,
                                        data={
                                            'reason': 'recovery-identity-mismatch',
                                            'dest': str(dest) if dest else None,
                                        },
                                    )
                                cleaned += 1
                                continue

                        pool.restore_assignment(rec.task_id, entry)
                        rec_plan_path = self._resolve_recovery_artifact(
                            entry, 'plan.json',
                        )
                        if rec_plan_path.exists():
                            try:
                                rec_plan = json.loads(rec_plan_path.read_text())
                            except (json.JSONDecodeError, OSError) as e:
                                rec_plan = None
                                logger.warning(
                                    'Recovery: lane %s record-adopted for task '
                                    '%s but plan.json unreadable (%s)',
                                    entry.name, rec.task_id, e,
                                )
                            if rec_plan is not None:
                                # Mirror the heuristic path's completed-steps /
                                # stamped-preservation decision (~2394-2434
                                # below): only pre-load a plan that already
                                # has completed work into _recovered_plans —
                                # workflow.py treats a pre-loaded plan as
                                # initial_plan and SKIPS _plan() entirely, so
                                # a stale plan would never get revalidated
                                # against a possibly-advanced main. A stamped
                                # plan with zero completed steps (e.g. the
                                # blast-radius lock-conflict requeue) is
                                # preserved instead, so _plan() still takes
                                # its revalidation-against-current-main
                                # branch. An unstamped, zero-completed plan
                                # has nothing usable to recover — leave both
                                # dicts untouched, same as a missing plan.json.
                                rec_completed = [
                                    s for col in ('prerequisites', 'steps')
                                    for s in rec_plan.get(col, [])
                                    if isinstance(s, dict) and s.get('status') == 'done'
                                ]
                                if rec_completed:
                                    self._recovered_plans[rec.task_id] = rec_plan
                                    self._adopt_recovered_session(entry, rec.task_id)
                                elif rec_plan.get('_session_id'):
                                    logger.info(
                                        'Recovery: lane %s record-adopted for '
                                        'task %s has stamped plan with no '
                                        'completed steps — preserving for '
                                        'revalidation instead of pre-loading',
                                        entry.name, rec.task_id,
                                    )
                                    self._preserved_worktrees.add(rec.task_id)
                        rec_lock_path = self._resolve_recovery_artifact(
                            entry, 'plan.lock',
                        )
                        if rec_lock_path.exists():
                            rec_lock_path.unlink()
                            logger.info(
                                'Recovery: cleared stale plan.lock for '
                                'record-adopted task %s', rec.task_id,
                            )
                        logger.info(
                            'Recovery: lane %s record-adopted for task %s '
                            '(durable state=%s)',
                            entry.name, rec.task_id, rec.state.value,
                        )
                        recovered += 1
                        continue
                    else:
                        # Divergence: the durable record says this lane
                        # belongs to rec.task_id, but git reality disagrees
                        # (orphaned admin entry — 2097/2098 — or, once
                        # step-10 lands, a stale-branch collision — 2062).
                        # ANY divergence quarantines; never adopt-on-doubt,
                        # never silently re-pin (PRD dec.4/5, I3). Quarantine
                        # is two explicit steps rather than
                        # lane_lifecycle.quarantine(): git_ops captured
                        # quarantine_worktree at construction time, and tests
                        # (plus any post-construction rebind) replace the
                        # LIVE git_ops.quarantine_worktree — calling through
                        # lane_lifecycle would hit the stale original and
                        # bypass it.
                        reason = 'recovery-record-divergence'
                        logger.warning(
                            'Recovery: lane %s record ASSIGNED for task %s '
                            'diverges from git reality (registered=%s, '
                            'branch_ok=%s) — quarantining, not re-pinning',
                            entry.name, rec.task_id, is_registered, branch_ok,
                        )
                        dest = await self.git_ops.quarantine_worktree(
                            entry, rec.branch or rec.task_id, reason,
                        )
                        self.git_ops._lane_lifecycle.transition(
                            entry, DurableLaneState.QUARANTINED,
                        )
                        if self.event_store:
                            self.event_store.emit(
                                EventType.worktree_quarantined,
                                task_id=rec.task_id,
                                data={
                                    'reason': reason,
                                    'dest': str(dest) if dest else None,
                                },
                            )
                        cleaned += 1
                        continue

            plan_path = self._resolve_recovery_artifact(entry, 'plan.json')

            if not plan_path.exists():
                # Mid-invocation crash: an agent subprocess was in flight when
                # the prior orchestrator died (no plan was written yet).  The
                # sidecar pins the Claude session UUID so the next workflow
                # can --resume it with a "continue" prompt instead of spawning
                # a fresh agent.
                #
                # For lanes: a v1 sidecar carries session_id/role/owner_pid
                # but NO task_id, and the in-memory assignment map is empty
                # after a restart, so a v1-only lane still can't be mapped
                # back to its real task -- nothing is adopted for it.  A v2
                # sidecar (task 2772+) DOES carry its own task_id, though, so
                # it CAN be adopted below -- keyed by that real task_id,
                # never by the lane dir name ('_lane-k'), which sidesteps
                # the dead-state concern this comment used to warn about.
                # Either way the lane itself is always released back to the
                # pool (cleanup_worktree routes lanes to release_warm_lane)
                # -- see the adoption call just below for the current keying
                # behavior and its narrow orphan-reaper interaction.
                if is_lane:
                    # Best-effort: adopt only if the sidecar is v2 (carries
                    # its own task_id) — the only identity source available
                    # on a no-plan lane (B3). Keying by the real task_id
                    # (never the lane dir name) means the orphan reaper's
                    # `name in self._recovered_sessions` skip (harness.py
                    # ~3273) can only shield a worktree that happens to be
                    # named exactly like that task_id (e.g. a stale
                    # worktree literally named '73') -- narrow and
                    # plausibly desired (that task's session is queued for
                    # resume), not the lane itself. Disposition here is
                    # UNCHANGED either way: the lane is still released back
                    # to the pool below.
                    self._adopt_recovered_session(entry, None)
                    logger.info(
                        f'Recovery: lane {task_id} has no plan — '
                        f'releasing back to pool'
                    )
                    await self.git_ops.cleanup_worktree(entry, task_id)
                    cleaned += 1
                    continue

                if self._adopt_recovered_session(entry, task_id) is not None:
                    self._preserved_worktrees.add(task_id)
                    recovered += 1
                    continue
                logger.info(
                    f'Recovery: worktree {task_id} has no plan — cleaning up'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            try:
                plan = json.loads(plan_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f'Recovery: worktree {task_id} has corrupt plan — '
                    f'cleaning up ({e})'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            # For lanes, derive the real task id from plan.json.  A lane dir
            # name (e.g. '_lane-0') never equals the real task_id by design —
            # the lane dir is named after the pool slot, not the task.
            # Cold path: recovery_id == task_id (entry.name) — byte-identical.
            # Normalize to str: plan.json could store task_id as int; dispatch
            # and restore_assignment key off str (mirroring the str() coercion
            # in the orphan-reaper live_ids set).
            recovery_id = (
                str(plan.get('task_id'))
                if (is_lane and plan.get('task_id') is not None)
                else task_id
            )

            # For lanes: if plan.json has no task_id there is no recoverable
            # identity — release the lane back to the pool.
            if is_lane and not recovery_id:
                logger.warning(
                    'Recovery: lane %s plan.json has no task_id — '
                    'releasing back to pool',
                    task_id,
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            # Validate plan belongs to this task (cold path only).
            # Lanes skip this check: lane name != task_id is expected by
            # design, so the cold numeric-mismatch cleanup must not fire.
            if not is_lane:
                plan_task_id = plan.get('task_id')
                if plan_task_id and plan_task_id != task_id:
                    logger.warning(
                        f'Recovery: worktree {task_id} has plan for task '
                        f'{plan_task_id} — task_id mismatch, cleaning up'
                    )
                    await self.git_ops.cleanup_worktree(entry, task_id)
                    cleaned += 1
                    continue

            # ── Semantic identity guard (Fix C) ───────────────────────
            # The numeric guard above only proves plan.task_id == dirname.
            # For a RECYCLED id both equal the NEW task's id, so it passes
            # even when the worktree's content belongs to the deleted task
            # (reify task 3770 adopted a trajectory plan onto a cycle-breaker
            # task).  Compare the worktree's stored title to the LIVE DB
            # task's title and quarantine on a provable mismatch — this is the
            # exact line that would have caught 3770.
            # For lanes, recovery_id is the real task id read from plan.json.
            if self.config.worktree_identity_guard_enabled:
                live = await self.scheduler.get_task(recovery_id)
                if live is None:
                    # get_task returns None for BOTH "deleted" and transient
                    # error — both safely handled by deferring: do not adopt,
                    # do not destroy.  The orphan reaper (which fail-safes on
                    # an empty task list) handles a genuinely-deleted task.
                    logger.warning(
                        'Recovery: worktree %s has no live DB task — deferring '
                        '(no adopt, no destroy)', recovery_id,
                    )
                    continue
                stored_title = read_worktree_title(entry)
                if not identities_match(stored_title, live.get('title')):
                    logger.warning(
                        'Recovery: worktree %s identity MISMATCH — stored title '
                        '%r != live %r; quarantining',
                        recovery_id, stored_title, live.get('title'),
                    )
                    await self.git_ops.quarantine_worktree(
                        entry, recovery_id, 'recovery-identity-mismatch',
                    )
                    if self.event_store:
                        self.event_store.emit(
                            EventType.worktree_quarantined,
                            task_id=recovery_id,
                            data={'reason': 'recovery-identity-mismatch'},
                        )
                    cleaned += 1
                    continue

            # For lanes: restore the in-memory pool assignment so the lane is
            # reserved ASSIGNED before re-dispatch.  Both sets the map AND
            # flips the lane FREE→ASSIGNED, preventing a concurrent fresh
            # acquire from stealing the lane while the original task is queued.
            #
            # T10 amplifier fix: if the task is already terminal (done/cancelled),
            # release the lane instead of restoring it.  Without this, every
            # harness restart re-ASSIGNs a dead lane, shrinking the pool forever.
            # On a transient None status, fall through to restore (safe default;
            # layer A self-heals the lane on the next reconcile interval).
            if pool is not None and is_lane and recovery_id:
                term_status = await self.scheduler.get_status(recovery_id)
                if term_status in ('done', 'cancelled'):
                    logger.info(
                        'Recovery: lane %s task %s terminal (%s) — '
                        'releasing instead of restore',
                        entry.name, recovery_id, term_status,
                    )
                    await self.git_ops.cleanup_worktree(entry, recovery_id)
                    cleaned += 1
                    continue
                # W11 delta compat (PRD dec.5): by construction, any lane
                # reaching this point has already been proven NOT to carry a
                # live ASSIGNED/IN_USE durable record — the record-driven
                # block above exhaustively handles that case (adopt /
                # terminal-release / quarantine) and always `continue`s
                # before falling through here.  So this lane is either a
                # pre-W11 record-less lane or one whose record is
                # SEED/REGISTERED/RELEASED.  Never silently re-pin from a
                # stale plan.json heuristic — that was the old
                # restore-from-any-plan.json default that re-poisoned lanes
                # every restart (2097/2098).  Leave the lane FREE for the
                # create-once self-heal; the plan is still recovered below
                # via _recovered_plans, independent of the (skipped) pin.
                logger.info(
                    'Recovery: lane %s task %s has no live durable record — '
                    'plan recovered but lane left FREE (never silently '
                    're-pin a record-less lane)',
                    entry.name, recovery_id,
                )

            # Check if plan has any completed steps
            # Note: some plans have prerequisites as plain strings (not dicts)
            completed = [
                s for col in ('prerequisites', 'steps')
                for s in plan.get(col, [])
                if isinstance(s, dict) and s.get('status') == 'done'
            ]

            if not completed:
                # Provenance-stamped plans deserve preservation: the architect
                # ran successfully and wrote a plan that subsequently got the
                # session_id stamp.  A common path here is the blast-radius
                # lock-conflict requeue (workflow.py:1071-1088), which leaves
                # a stamped pre-EXECUTE plan behind.  Wiping the worktree
                # forces the next acquisition to call the architect again —
                # 17-20 wasted Opus calls per 14d.  Keep the worktree, unlink
                # plan.lock, and add to _preserved_worktrees so the next
                # acquisition reuses it; _plan() will then take the
                # revalidation branch via _old_plan_base.  We deliberately
                # do NOT pre-load into _recovered_plans because that bypasses
                # _plan() entirely.
                if plan.get('_session_id'):
                    logger.info(
                        f'Recovery: worktree {recovery_id} has stamped plan with '
                        f'no completed steps — preserving for revalidation'
                    )
                    lock_path = self._resolve_recovery_artifact(entry, 'plan.lock')
                    if lock_path.exists():
                        lock_path.unlink()
                        logger.info(
                            f'Recovery: cleared stale plan.lock for '
                            f'preserved task {recovery_id}'
                        )
                    # The architect already produced a stamped plan; any
                    # sidecar present is from a later (post-plan) invocation
                    # that crashed and isn't meaningful here — clear it to
                    # avoid confusing the next workflow on this task.
                    self._clear_recovery_artifact(entry, 'agent_session.json')
                    self._preserved_worktrees.add(recovery_id)
                    recovered += 1
                    continue
                logger.info(
                    f'Recovery: worktree {task_id} has unstamped plan with no '
                    f'completed steps — cleaning up'
                )
                # For lanes, pass recovery_id (the real task branch name) so
                # release_warm_lane deletes the actual task branch (e.g.
                # 'task/42') rather than the nonexistent 'task/_lane-0'.
                # Cold path: recovery_id == task_id, so behavior is unchanged.
                await self.git_ops.cleanup_worktree(
                    entry, recovery_id if is_lane else task_id
                )
                cleaned += 1
                continue

            total = sum(len(plan.get(col, [])) for col in ('prerequisites', 'steps'))
            logger.info(
                f'Recovery: worktree {recovery_id} has plan with '
                f'{len(completed)}/{total} steps done — storing for resumption'
            )
            self._recovered_plans[recovery_id] = plan
            self._adopt_recovered_session(entry, recovery_id)
            # Clear stale plan.lock so the new session doesn't immediately requeue
            lock_path = self._resolve_recovery_artifact(entry, 'plan.lock')
            if lock_path.exists():
                lock_path.unlink()
                logger.info(f'Recovery: cleared stale plan.lock for task {recovery_id}')
            recovered += 1

        if recovered or cleaned:
            logger.info(
                f'Crash recovery: {recovered} plans recovered, '
                f'{cleaned} worktrees cleaned'
            )

    async def _reconcile_lane_checkouts(self) -> None:
        """Reconcile stale `task/<id>` branch checkouts left behind by recovery.

        ``_recover_crashed_tasks`` re-pins ``id -> lane`` via
        ``pool.restore_assignment`` for every worktree it can positively
        identify, so re-dispatch takes the REUSE path and issues no fresh
        ``git worktree add``.  But several of its branches SKIP restore while
        leaving the git-admin checkout intact (the identity-guard defer on a
        transient ``scheduler.get_task() is None``; the no-plan/corrupt-plan/
        no-task-id release branches whose best-effort detach can fail under
        git-admin desync).  After a restart, ``_assignments`` is then empty
        for that id, a fresh dispatch grabs a DIFFERENT lane, and ``git
        worktree add`` collides with "already used by worktree" — a fault
        that persists until a human intervenes.

        This method closes that gap by reconciling from git's OWN
        authoritative worktree admin (:meth:`GitOps.lane_branch_checkouts`)
        rather than trusting the in-memory assignment map alone.  Called from
        :meth:`run` immediately after ``_recover_crashed_tasks`` — same
        startup envelope as the orphan reaper (no task is yet
        ``scheduler.is_dispatched()``, single asyncio loop), so ``restore_assignment`` /
        ``assignments_snapshot`` (both synchronous, no ``await``) cannot race
        a live dispatch.

        Fail-safe posture (never mass-destroy on an unreliable read):
        - ``lane_branch_checkouts()`` returns ``None`` (pool disabled, or a
          ``git worktree list`` error) -> no-op entirely.
        - A DEGRADED ``get_statuses`` read (:func:`resolver_failed`) -> a
          degraded read never detaches based on task STATUS; only the
          DB-independent dup-checkout/never-steal guards below may still
          act.  Every checkout that reaches the status branch is instead
          RE-PINNED — re-pin is DB-independent and self-heals (a
          genuinely-terminal re-pinned lane is reclaimed by the next
          :meth:`_reconcile_terminal_lanes` pass).

        Precedence (consulting ``pool.assignments_snapshot()``, computed once
        per pass): (1) the id is already pinned to THIS exact lane -> skip,
        idempotent; (2) this lane is pinned to a DIFFERENT id -> skip, never
        steal a lane recovery assigned to someone else; (3) the id is pinned
        to a DIFFERENT lane (a duplicate checkout) -> detach this un-pinned
        duplicate, keeping the original pin; (4) otherwise -> the
        degraded/absent/terminal/live status logic below.
        """
        pool = self.git_ops.warm_lane_pool
        if pool is None:
            return

        checkouts = await self.git_ops.lane_branch_checkouts()
        if checkouts is None:
            logger.warning(
                'Lane-checkout reconciler: lane_branch_checkouts() returned '
                'None (pool disabled or git error) — skipping reconcile',
            )
            return
        if not checkouts:
            return

        statuses, err = await self.scheduler.get_statuses()
        degraded = resolver_failed(statuses, err)
        live = {} if degraded else {str(k): v for k, v in statuses.items()}

        # Precedence guards consulting the CURRENT in-memory assignment map —
        # computed once before the loop so a lane already reconciled this
        # pass cannot be mistaken for a fresh dup/steal on a later iteration.
        assigned = pool.assignments_snapshot()
        lane_to_id: dict[Path, str] = {}
        for pinned_id, pinned_lane in assigned.items():
            canon = pool._match_lane(pinned_lane)
            if canon is not None:
                lane_to_id[canon] = pinned_id

        for bare_id, lane in checkouts.items():
            canon_pinned = (
                pool._match_lane(assigned[bare_id]) if bare_id in assigned else None
            )
            other_id = lane_to_id.get(lane)

            if canon_pinned == lane:
                # Idempotent skip: recovery already re-pinned this exact id
                # to this exact lane (e.g. _recover_crashed_tasks fired) —
                # nothing to do.
                continue
            if other_id is not None and other_id != bare_id:
                # NEVER-STEAL: this lane is pinned to a DIFFERENT id already —
                # never take a lane recovery assigned to someone else.
                logger.warning(
                    'Lane-checkout reconciler: lane %s is pinned to a '
                    'DIFFERENT id %s (never-steal) — skipping checkout for %s',
                    lane, other_id, bare_id,
                )
                continue
            if canon_pinned is not None:
                # DUP-CHECKOUT: bare_id is already pinned to a DIFFERENT lane
                # — detach this un-pinned duplicate, keep the original pin.
                logger.info(
                    'Lane-checkout reconciler: %s is already pinned to a '
                    'different lane (%s) — detaching duplicate checkout at %s',
                    bare_id, canon_pinned, lane,
                )
                detached = await self.git_ops.detach_lane_checkout(lane, bare_id)
                if not detached:
                    # Do NOT fall back to pool.restore_assignment(bare_id, lane)
                    # here: bare_id already has a TRUSTED pin at canon_pinned,
                    # and restore_assignment would silently steal it onto this
                    # unverified duplicate (and orphan canon_pinned as
                    # ASSIGNED-but-unmapped).  Warn loudly instead so the
                    # unrepaired duplicate is observable above git_ops' own
                    # lower-level ERROR log.
                    logger.warning(
                        'Lane-checkout reconciler: FAILED to detach duplicate '
                        'checkout of %s at %s — keeping the trusted pin at '
                        '%s, but %s remains checked out at %s and a future '
                        'reuse could still collide with "already used by '
                        'worktree" if that trusted pin is ever lost',
                        bare_id, lane, canon_pinned, bare_id, lane,
                    )
                continue

            if degraded:
                logger.warning(
                    'Lane-checkout reconciler: DEGRADED status read — '
                    're-pinning %s -> %s fail-safe (never detach on a bad read)',
                    bare_id, lane,
                )
                pool.restore_assignment(bare_id, lane)
            elif bare_id not in live:
                logger.info(
                    'Lane-checkout reconciler: %s absent from a healthy '
                    'status read (task deleted) — detaching stale checkout '
                    'at %s',
                    bare_id, lane,
                )
                detached = await self.git_ops.detach_lane_checkout(lane, bare_id)
                if not detached:
                    # No existing pin to protect here (canon_pinned was None) —
                    # fail-safe re-pin so the still-checked-out lane is not
                    # handed to a fresh dispatch, mirroring the degraded-read
                    # posture above.
                    logger.warning(
                        'Lane-checkout reconciler: FAILED to detach stale '
                        'checkout of deleted task %s at %s — re-pinning '
                        'fail-safe so the lane is not handed to a fresh '
                        'dispatch while still holding this checkout',
                        bare_id, lane,
                    )
                    pool.restore_assignment(bare_id, lane)
            elif live[bare_id] in TERMINAL_STATUSES:
                logger.info(
                    'Lane-checkout reconciler: %s is terminal (%s) — '
                    'detaching stale checkout at %s',
                    bare_id, live[bare_id], lane,
                )
                detached = await self.git_ops.detach_lane_checkout(lane, bare_id)
                if not detached:
                    logger.warning(
                        'Lane-checkout reconciler: FAILED to detach terminal '
                        'checkout of %s (%s) at %s — re-pinning fail-safe so '
                        'the lane is not handed to a fresh dispatch while '
                        'still holding this checkout',
                        bare_id, live[bare_id], lane,
                    )
                    pool.restore_assignment(bare_id, lane)
            else:
                logger.info(
                    'Lane-checkout reconciler: re-pinning live task %s -> %s '
                    '(recovery skipped this id)',
                    bare_id, lane,
                )
                pool.restore_assignment(bare_id, lane)

    async def _reap_orphan_worktrees(self) -> None:
        """Reap/quarantine worktrees whose id no longer maps to a live task.

        Worktrees are keyed purely on the numeric task id and are cleaned only
        on merge/done or crash-recovery heuristics — never when a task is
        deleted from the DB.  Such an orphan survives on disk and can later be
        adopted for an unrelated (recycled-id) task (Fix B closes that gap).

        Run ONCE at startup, AFTER ``_reconcile_stranded_in_progress`` and
        BEFORE the first ``acquire_next`` — at that point no task is yet
        ``scheduler.is_dispatched()`` so no live workflow can race the
        sweep.

        Policy: **quarantine** anything with commits OR uncommitted WIP
        (preserving it); **reap** only provably-empty/clean dirs.  Fail-safe:
        an empty task list (likely a transient DB failure) ABORTS the sweep —
        never mass-destroy.

        Self-gates on ``worktree_orphan_reaper_enabled`` so the call site can
        stay unconditional and the flag is honoured from a direct invocation.
        """
        if not self.config.worktree_orphan_reaper_enabled:
            return

        worktree_base = self.git_ops.worktree_base
        if not worktree_base.exists():
            return

        # Pool-storage guard (task 2099): worktree_base can EXIST as an
        # unmounted mountpoint dir, making every mount-resident worktree
        # APPEAR missing to the checks below — the Jul-3 incident this
        # guards against. Abort the ENTIRE sweep (no DB read, no
        # cleanup/quarantine, no prune tail) rather than risk treating a
        # live lane as an orphan. Gated on pool_in_use() (step-16/17
        # review-fix): pool_storage_present() is permanently False on a
        # pool-less host (its only writer never runs without a configured
        # pool), and cold worktrees at worktree_base/<branch> make
        # worktree_base.exists() True there too — without this gate, every
        # pool-less host that has ever run a task would abort its sweep and
        # file a spurious escalation at every startup. The resolve call
        # below is gated on _pool_storage_absent_maybe_pending rather than
        # skipped outright — that flag starts True on every process start,
        # so a pool-less host on a build that pre-dates this gate can still
        # auto-clear a stale open L1 on its first post-restart sweep; only
        # the subsequent, confirmed-clear steady-state ticks skip the scan.
        if self.git_ops.pool_in_use() and not self.git_ops.pool_storage_present():
            logger.warning(
                'Orphan reaper: pool storage absent/unmounted at %s — '
                'aborting the ENTIRE sweep (no cleanup/quarantine/prune) to '
                'avoid treating mount-resident worktrees as orphans',
                worktree_base,
            )
            self._file_pool_storage_absent_escalation()
            return
        if self._pool_storage_absent_maybe_pending:
            await self._resolve_pool_storage_absent_escalation()

        statuses, err = await self.scheduler.get_statuses()
        if resolver_failed(statuses, err):
            logger.warning(
                'Orphan reaper: get_statuses() returned %s — aborting sweep '
                '(fail-safe against transient DB failure or empty task tree)',
                'error' if err is not None else 'empty',
            )
            return
        live_ids = {str(k) for k in statuses}

        reaped = 0
        quarantined = 0
        for entry in worktree_base.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            # Skip warm pool lanes.  quarantine_worktree is NOT pool-aware
            # (it moves the dir), so moving a lane would leave the pool's
            # registered path dangling.  Crash-recovery already handles
            # lanes; the reaper must not undo a just-recovered lane.
            # The merge-speculation pool ('_spec-' lanes) is a SEPARATE
            # WarmLanePool instance — its lanes are not '_merge-*', not
            # members of warm_lane_pool, and their names are not live task
            # ids, so they would otherwise fall through to the orphan branch
            # and get moved/removed mid-verify.  Protect them identically.
            # Checked FIRST — before the C2 classifier below: adoptable
            # '_lane-'/'_spec-' lanes are '_'-prefixed, so the classifier
            # would mislabel them 'infra'; is_lane (actual pool registration)
            # must win.
            if (
                self.git_ops.warm_lane_pool is not None
                and self.git_ops.warm_lane_pool.is_lane(entry)
            ):
                continue
            if (
                self.git_ops.spec_warm_lane_pool is not None
                and self.git_ops.spec_warm_lane_pool.is_lane(entry)
            ):
                continue
            # Skip auto-eval '*-skip-attempt' worktrees: a SUFFIX namespace
            # orthogonal to C2's prefix rule (these names are NOT '_'/'.'-
            # prefixed, so the classifier would call them 'task').  Preserved
            # exactly as before.
            if name.endswith('-skip-attempt'):
                continue
            # ── C2 namespace invariant (task 2925, merge-worktree-lifecycle
            # -integrity PRD §4) ───────────────────────────────────────────
            # The positive-match classifier replaces the old '_merge-'
            # per-name skip.  '_merge-*' is REPORTED to the merge reaper
            # (_reap_orphaned_merge_worktrees owns its guarded readopt/
            # age-grace disposition — the sweep NEVER reaps/quarantines a
            # '_merge-*' directly); every OTHER '_'/'.'-prefixed infra band
            # (_mainprobe-*, _offline-deep, _iact-*, .reseed-trash, ...) is
            # left to its owner.  This closes the latent bug where those
            # bands fell through to the orphan quarantine/reap branch below.
            # Both dispositions are OBSERVED via an explicit journal line,
            # never silence.
            worktree_class = classify_worktree_entry(name)
            if worktree_class == 'merge':
                logger.info(
                    'Orphan reaper: %s is a merge worktree — reporting to the '
                    'merge reaper, never reaped here',
                    name,
                )
                continue
            if worktree_class == 'infra':
                logger.info(
                    'Orphan reaper: %s is infra-owned (C2) — left to its owner',
                    name,
                )
                continue
            # 'task' falls through to the live/recovered/preserved/session/
            # dispatched checks and the orphan quarantine/reap branch below.
            # Skip live, recovered, preserved, and in-flight worktrees.
            if (
                name in live_ids
                or name in self._recovered_plans
                or name in self._preserved_worktrees
                or name in self._recovered_sessions
                or self.scheduler.is_dispatched(name)
            ):
                continue

            # Orphan: numeric id no longer maps to a live task.
            if await self.git_ops.worktree_has_unsaved_work(entry, name):
                dest = await self.git_ops.quarantine_worktree(
                    entry, name, 'orphan-reaper',
                )
                if self.event_store:
                    self.event_store.emit(
                        EventType.worktree_quarantined,
                        task_id=name,
                        data={
                            'reason': 'orphan-reaper',
                            'dest': str(dest) if dest else None,
                        },
                    )
                quarantined += 1
                logger.info(
                    'Orphan reaper: quarantined %s (had unsaved work)', name,
                )
            else:
                await self.git_ops.cleanup_worktree(entry, name)
                if self.event_store:
                    self.event_store.emit(
                        EventType.worktree_reaped,
                        task_id=name,
                        data={'reason': 'orphan-reaper'},
                    )
                reaped += 1
                logger.info(
                    'Orphan reaper: reaped %s (clean, no commits)', name,
                )

        # Clear stale .git/worktrees admin entries (best-effort, never raises).
        await self.git_ops.prune_worktrees()

        if reaped or quarantined:
            logger.info(
                'Orphan reaper: %d reaped, %d quarantined', reaped, quarantined,
            )

    def _workflow_cancel_recent(self, tid: str) -> bool:
        """Return True if *tid* has a workflow-cancel stamp within the grace window.

        Task 2235 (W10-alpha): back-compat shim — the cancel-grace stamp,
        grace constant, and this predicate now live on the Scheduler (beside
        ``_dispatched`` / ``lock_table``, their single writer). Production
        call sites read ``self.scheduler.workflow_cancel_recent`` directly;
        this method forwards for the handful of tests still calling it on
        the Harness.
        """
        return self.scheduler.workflow_cancel_recent(tid)

    @property
    def _workflow_cancel_at(self) -> dict[str, float]:
        """Back-compat read/write shim (task 2235) over the Scheduler's own
        cancel-grace stamp dict — the Scheduler is the single owner (see
        ``Scheduler.note_workflow_cancelled`` / ``.clear_workflow_cancel``).
        Returns the live dict (not a copy): ``.clear()`` / item-assignment
        mutate the Scheduler's own state.
        """
        return self.scheduler._workflow_cancel_at

    @_workflow_cancel_at.setter
    def _workflow_cancel_at(self, value: dict[str, float]) -> None:
        self.scheduler._workflow_cancel_at = value

    def _register_escalation_event(self, task_id: str) -> asyncio.Event | None:
        """Register an escalation wake-event for *task_id* at dispatch time.

        Calling this BEFORE ``asyncio.create_task(_run_slot(...))`` closes the
        sub-second race in Fix #1a (``_on_escalation_resolved``):

          dispatch → create_task(_run_slot)
                          ↑
                     [gap: _run_slot hasn't run yet; task_id not in
                      _escalation_events → Fix #1a sees it as an orphan
                      and double-flips blocked→pending, racing the
                      workflow's own re-pend]

        With dispatch-time registration, ``task_id in _escalation_events`` is
        True the instant the slot is created, so Fix #1a's orphan-flip gate
        always sees an active workflow and skips the orphan path.

        Returns:
            The newly-created ``asyncio.Event`` stored at
            ``_escalation_events[task_id]``, or ``None`` when
            ``_escalation_queue`` is falsy (no escalation wiring — no-op).
        """
        if not self._escalation_queue:
            return None
        esc_event = asyncio.Event()
        self._escalation_events[task_id] = esc_event
        return esc_event

    async def _warm_lane_reclaim_candidates(
        self, candidates: list[str],
    ) -> set[str]:
        """Return the non-terminal subset of *candidates* for reclaim eligibility.

        Installed as :attr:`git_ops.warm_lane_reclaim_candidate_provider` when
        the knob is on (task 1933).  Mirrors ``_reconcile_terminal_lanes``
        INVERTED: keep branches whose status is known and NOT in
        :data:`TERMINAL_STATUSES`; abort to ``set()`` on ``resolver_failed``
        (same fail-safe as the reconciler — never act on a degraded/empty read).
        Both methods reference the same shared ``TERMINAL_STATUSES`` constant
        so the inversion cannot silently drift if the terminal set ever changes.

        Also excludes :data:`_WARM_LANE_RECLAIM_PROTECTED_STATUSES` — parked
        branches (``merge-deferred``, ``deferred``) whose worktrees are owned
        by a non-scheduler party and must survive intact (task 2018), mirroring
        the merge-deferred early-return guard in :meth:`_reconcile_one_stranded`
        (harness.py:~2437).  ``blocked`` is deliberately NOT excluded here — a
        stranded blocked task remains a legitimate reclaim target.

        **Single-orchestrator-ownership invariant:** The warm-lane pool is
        exclusively owned by this orchestrator process.  A non-terminal task
        for which ``scheduler.is_dispatched()`` is False (checked
        synchronously under the pool lock by :meth:`WarmLanePool.reclaim_victim`)
        implies it is stale or stranded — not actively executing — because any
        live task is guaranteed to be dispatched.  This invariant holds because
        :meth:`_reconcile_stranded_in_progress` reverts genuinely stranded
        ``in-progress`` tasks before the safety valve fires.  Do **not** reuse
        this logic in a shared-pool context where non-dispatched does not imply
        stale.

        Args:
            candidates: Branch names from the pool's assignment snapshot.

        Returns:
            Set of non-terminal, non-protected-parked branch names eligible
            for victim selection.  Returns ``set()`` on empty input
            (fast-path, no get_statuses call), or on ``resolver_failed``
            (fail-safe).
        """
        if not candidates:
            return set()
        statuses, err = await self.scheduler.get_statuses(list(candidates))
        if resolver_failed(statuses, err):
            logger.warning(
                'Reclaim-candidate provider: get_statuses returned %s — '
                'returning empty set (fail-safe; mirrors terminal-lane reconciler)',
                'error' if err is not None else 'empty',
            )
            return set()
        return {
            b for b in candidates
            if (status := statuses.get(b)) is not None
            and status not in TERMINAL_STATUSES
            and status not in _WARM_LANE_RECLAIM_PROTECTED_STATUSES
        }

    def _is_branch_dispatched(self, branch: str) -> bool:
        """Return True iff *branch* is currently in the scheduler's dispatched set.

        Installed as :attr:`git_ops.warm_lane_dispatched_predicate` when the
        knob is on (task 1933).  Re-checked synchronously under the pool lock
        inside :meth:`WarmLanePool.reclaim_victim` to close the TOCTOU window
        where a candidate is re-dispatched during the async ``get_statuses``
        await in :meth:`_warm_lane_reclaim_candidates`.
        """
        return self.scheduler.is_dispatched(branch)

    async def _reconcile_terminal_lanes(self) -> None:
        """Release warm lanes whose assigned tasks are terminal and not live.

        Layer A (invariant backstop): sweeps every ASSIGNED lane, checks the
        task status via ``scheduler.get_statuses``, and releases lanes whose
        task is done/cancelled and not ``scheduler.is_dispatched()`` (the
        live-acquire guard prevents racing a fresh ``acquire_for``).

        Fail-safe MIRRORS ``_reap_orphan_worktrees``: an empty or errored
        ``get_statuses`` result ABORTS the whole sweep — never mass-free.

        NOTE: there is an intentional conflation of two abort triggers:
        (a) transient DB failure — ``get_statuses`` returned an error, and
        (b) all assigned tasks were deleted — ``get_statuses`` returns ``{}``
            because the scheduler filtered out every unknown id.
        In case (b), the reconciler will not free those orphaned lanes; they
        are expected to be caught eventually by ``_reap_orphan_worktrees`` (if
        enabled) or by the next reconciler pass once the tasks are re-created.
        This is accepted because case (a) and (b) are indistinguishable at the
        ``resolver_failed`` check level and the never-mass-free invariant takes
        precedence.

        Wired into:
        - startup (after ``_reap_orphan_worktrees``, before first ``acquire_next``)
        - ``_run_stranded_reconcile_pass`` (after ``_reconcile_stranded_in_progress``)

        Reuses ``stranded_reconcile_interval_secs``; no new config knob.
        """
        pool = self.git_ops.warm_lane_pool
        if pool is None:
            return

        assignments = pool.assignments_snapshot()
        if not assignments:
            return

        statuses, err = await self.scheduler.get_statuses(list(assignments.keys()))
        if resolver_failed(statuses, err):
            logger.warning(
                'Terminal-lane reconciler: get_statuses returned %s — '
                'aborting sweep (fail-safe against transient DB failure or '
                'empty task tree)',
                'error' if err is not None else 'empty',
            )
            return

        released = 0
        for branch in list(assignments.keys()):
            status = statuses.get(branch)
            if status not in TERMINAL_STATUSES:
                continue
            if self.scheduler.is_dispatched(branch):
                # Live-acquire guard: a workflow may have just acquired this
                # branch; skip to avoid racing the fresh dispatch.
                continue
            if await self.git_ops.release_lane_for_terminal_task(branch):
                released += 1
                if self.event_store:
                    self.event_store.emit(
                        EventType.worktree_reaped,
                        task_id=branch,
                        data={
                            'reason': 'terminal-lane-reconciler',
                            'status': status,
                            # warm lane is FREE'd but NOT removed from disk —
                            # use this flag to avoid over-counting worktree
                            # removals in telemetry/downstream consumers.
                            'warm_lane_retained': True,
                        },
                    )

        if released:
            logger.info(
                'Terminal-lane reconciler: released %d lane(s)', released,
            )

    async def _assigned_durable_records_with_statuses(
        self,
    ) -> tuple[list[tuple[str, LaneRecord, str]], dict[str, str]] | None:
        """Shared prologue for the durable-record warm-lane passes (leaf γ, task 2891).

        Both :meth:`_reclaim_terminal_lane_records` and
        :meth:`_stale_lane_assignment_census` open identically: fetch the
        warm-lane pool (None-guard), enumerate the durable
        ``git_ops._lane_lifecycle.all_records()`` filtered to ASSIGNED/IN_USE
        records with a non-None ``task_id``, batch ``scheduler.get_statuses``
        for their distinct task ids, and ABORT the whole pass on a
        degraded/empty (``resolver_failed``) read — never mass-acting on a
        transient DB failure or an empty task tree.

        Returns ``(assigned, statuses)`` where ``assigned`` is the list of
        ``(lane_name, record, task_id)`` triples (``task_id`` narrowed non-None)
        and ``statuses`` the batched status map; returns ``None`` as the single
        abort sentinel when there is no pool, no assigned records, or the status
        read failed. Factoring this here keeps the None-guard / ASSIGNED-IN_USE
        filter / resolver_failed-abort semantics of the two passes identical by
        construction (they cannot drift apart).
        """
        pool = self.git_ops.warm_lane_pool
        if pool is None:
            return None

        records = self.git_ops._lane_lifecycle.all_records()
        assigned: list[tuple[str, LaneRecord, str]] = []
        for lane_name, rec in records.items():
            if (
                rec.state in (DurableLaneState.ASSIGNED, DurableLaneState.IN_USE)
                and rec.task_id is not None
            ):
                assigned.append((lane_name, rec, rec.task_id))
        if not assigned:
            return None

        task_ids = list({task_id for _, _, task_id in assigned})
        statuses, err = await self.scheduler.get_statuses(task_ids)
        if resolver_failed(statuses, err):
            logger.warning(
                'Durable warm-lane pass: get_statuses returned %s — aborting '
                '(fail-safe against a transient DB failure or empty task tree; '
                'never mass-acts on a degraded read)',
                'error' if err is not None else 'empty',
            )
            return None
        return assigned, statuses

    async def _reclaim_terminal_lane_records(self) -> int:
        """Release warm lanes whose DURABLE record is assigned to a terminal task.

        The durable-record complement to :meth:`_reconcile_terminal_lanes`
        (leaf γ, task 2891). Where the in-memory reconciler enumerates only
        ``pool.assignments_snapshot()``, this pass enumerates the durable
        ``.lane-state/*.json`` records via
        ``git_ops._lane_lifecycle.all_records()`` — the records that accumulate
        across restarts/churn and whose in-memory mapping is often lost, so the
        in-memory reconciler never sees them (the incident-07-21 pool-exhaustion
        census counted 41 such assigned durable records against zero free).

        For each durable ASSIGNED/IN_USE record whose task is TERMINAL
        (done/cancelled) and NOT ``scheduler.is_dispatched`` (the live-acquire
        guard), release the lane via the path-based ``git_ops.release_warm_lane``
        — the map-based ``release_lane_for_terminal_task`` no-ops for
        durable-only records whose in-memory assignment was lost, which is
        exactly what this pass targets. NON-terminal (pending/in-progress/
        blocked) lanes are NEVER released (the WIP-preserving invariant: a live
        task's lane may hold verified-green work — the incident 5260 lane did).

        Fail-safe MIRRORS :meth:`_reconcile_terminal_lanes`: a degraded/empty
        ``get_statuses`` read ABORTS the whole pass (never mass-free). Rides
        ``_run_warm_lane_gc_pass`` (no new timer/loop). Returns the number of
        lanes released.

        The branch-ref-resolve gate (assert the task branch still resolves
        before release) is added in step-8.
        """
        prologue = await self._assigned_durable_records_with_statuses()
        if prologue is None:
            return 0
        assigned, statuses = prologue

        released = 0
        for lane_name, _rec, task_id in assigned:
            status = statuses.get(task_id)
            if status not in TERMINAL_STATUSES:
                continue
            if self.scheduler.is_dispatched(task_id):
                # Live-acquire guard: a workflow may have just acquired this
                # task's lane; skip to avoid racing the fresh dispatch.
                continue
            # Assert-before-release (PRD): confirm the task branch ref still
            # resolves before freeing the lane. Branch survival ON release is
            # already guaranteed by inv.10 (release_warm_lane retains a branch
            # carrying commits beyond main); a terminal task whose branch has
            # ALREADY vanished is a rare anomaly we conservatively leave for the
            # in-memory reconciler / next acquire rather than silently releasing
            # — honoring the loud-over-silent, WIP-preserving posture.
            branch = f'{self.git_ops.config.branch_prefix}{task_id}'
            if await self.git_ops.resolve_branch_sha(branch) is None:
                logger.warning(
                    'Terminal-lane-record reclaim: branch %s for terminal task '
                    '%s (lane %s) does not resolve — skipping release, leaving '
                    'for the in-memory reconciler / next acquire',
                    branch, task_id, lane_name,
                )
                continue
            lane_dir = self.git_ops.worktree_base / lane_name
            await self.git_ops.release_warm_lane(lane_dir, task_id)
            released += 1
            if self.event_store:
                self.event_store.emit(
                    EventType.worktree_reaped,
                    task_id=task_id,
                    data={
                        'reason': 'terminal-lane-record-reclaim',
                        'status': status,
                        # warm lane is FREE'd but NOT removed from disk — mirror
                        # _reconcile_terminal_lanes so downstream telemetry does
                        # not over-count worktree removals.
                        'warm_lane_retained': True,
                    },
                )

        if released:
            logger.info(
                'Terminal-lane-record reclaim: released %d durable-record lane(s)',
                released,
            )
        return released

    async def _stale_lane_assignment_census(self) -> list[str]:
        """Census lines for NON-terminal warm-lane assignments idle past the threshold.

        The reporting complement to :meth:`_reclaim_terminal_lane_records`
        (leaf γ, task 2891): that pass reclaims TERMINAL-task lanes; this one
        surfaces the non-terminal (pending/in-progress/blocked) durable
        ASSIGNED/IN_USE records it deliberately LEAVES ALONE (WIP-preserving)
        but which have been idle longer than ``config.lane_stale_report_days``
        — the lanes an operator should look at (e.g. the incident-07-21 5260
        lane held verified-green work while its task sat pending).

        Rendered into the digest's ``## Stale lane assignments`` section via
        ``DigestInputs.stale_lane_census``. Fail-safe throughout: returns ``[]``
        on a missing pool or a degraded ``get_statuses`` read; a record with an
        empty/unparseable ``updated_at`` is skipped (never counted). QUARANTINED
        and terminal records are excluded.
        """
        prologue = await self._assigned_durable_records_with_statuses()
        if prologue is None:
            return []
        assigned, statuses = prologue

        now = datetime.now(UTC)
        threshold = timedelta(days=self.config.lane_stale_report_days)
        census: list[str] = []
        for lane_name, rec, task_id in assigned:
            status = statuses.get(task_id)
            if status is None or status in TERMINAL_STATUSES:
                continue
            if not rec.updated_at:
                continue
            try:
                updated_at = datetime.fromisoformat(rec.updated_at)
            except ValueError:
                continue
            # Records are written tz-aware (isoformat of a UTC datetime); guard
            # a legacy naive value so the subtraction never raises TypeError.
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            age = now - updated_at
            if age > threshold:
                census.append(
                    f'{lane_name} -> task {task_id} ({status}), '
                    f'stale {age.total_seconds() / 86400:.1f}d'
                )
        return census

    def _get_ground_truth(self) -> TaskGroundTruth:
        """Lazily build (and memoize) the ground-truth resolver (task 2243, W10-θ2).

        ``TaskGroundTruth`` captures its ``escalation_queue`` collaborator at
        construction time; ``self._escalation_queue`` is ``None`` at
        ``Harness.__init__`` and is only populated later, during ``run()``
        startup. Building the resolver once in ``__init__`` (or memoizing it
        on first use with no refresh) would freeze that startup-time
        ``None`` forever — every open-escalation-aware recovery row
        (``_RECOVERY`` rows f/g/h in task_ground_truth.py) would then never
        see a real queue for the rest of the process lifetime (the
        "frozen-None trap").

        So: build on first call, and REBUILD whenever ``self._escalation_queue``
        has since changed identity (the one-time ``None`` -> live-queue
        startup transition in production; a test re-pointing the queue).
        Once ``self._escalation_queue`` is stable, repeated calls return the
        SAME memoized instance rather than reconstructing on every reconcile
        pass.
        """
        # getattr (not self._ground_truth directly): several narrow-scope
        # test harnesses build a Harness via Harness.__new__(Harness),
        # bypassing __init__ (and its _ground_truth = None seed) entirely —
        # task 2243, W10-θ2 wiring made this the first call site to actually
        # reach _get_ground_truth from such a harness (test_harness_
        # deterministic_recon_sweep.py / test_deterministic_task.py).
        existing = getattr(self, '_ground_truth', None)
        if (
            existing is None
            or existing.escalation_queue is not self._escalation_queue
        ):
            existing = TaskGroundTruth(
                self.git_ops,
                self.scheduler,
                self._escalation_queue,
                self._resolve_task_worktree,
                heartbeat_ttl=_RECONCILE_HEARTBEAT_TTL,
            )
            self._ground_truth = existing
        return existing

    async def _reconcile_stranded_in_progress(self, *, mid_run: bool = False) -> int:
        """Sweep stranded in-progress tasks back to pending (or done).

        Examines every task that is currently in-progress and checks whether
        it has a live claimant via plan.lock / owner_pid.  Any task without a
        live claimant is reverted to pending so the scheduler can re-acquire it.

        At startup (``mid_run=False``) this is called AFTER
        ``_recover_crashed_tasks()`` (which may unlink plan.lock for recovered
        worktrees) and BEFORE the first ``scheduler.acquire_next()`` call, so
        no task is yet ``scheduler.is_dispatched()``.

        When ``mid_run=True`` the harness has dispatched tasks during this
        run; those are NOT stranded — they are actively held by the scheduler
        — and must be filtered before any liveness check.  Without the filter
        the sweep would race the workflow that legitimately holds the task.

        Returns the number of tasks reverted or marked done so the caller can
        decide whether to keep the main loop running (Fix 4: stuck-blocked
        recovery).

        **Ground-truth delegation** (task 2243, W10-θ2): per-candidate
        branch-state derivation and recovery classification are NOT performed
        here or in ``_reconcile_one_stranded`` — this driver only filters
        candidates (``_RECONCILE_SWEEP_STATUSES``) and dispatches each to
        ``_reconcile_one_stranded``, which calls
        ``TaskGroundTruth.recovery_for(tid)`` (``_get_ground_truth()``) to get
        a ``(TruthReport, RecoveryAction)`` pair and then applies the
        indicated action:

        - ``MARK_DONE_WITH_PROVENANCE`` → ``_mark_in_progress_done`` calls
          ``scheduler.mark_done(kind='found_on_main', sha=report.branch_state.sha, ...)``.
          ``report.branch_state.sha`` is populated whether the resolver found
          the landing via a ``MergeProvenance`` journal row (journal-first) or
          its git fallback (``is_ancestor`` / ``find_merge_marker``, resolved
          internally by ``derive_truth`` — see task_ground_truth.py).
        - ``REVERT_TO_PENDING`` → ``_revert_in_progress_if_no_live_claimant``
          flips the task back to pending.
        - ``RE_FILE_ESCALATION`` → a ``stranded_blocked`` L1 escalation is
          filed (blocked-status only).
        - ``LEAVE`` (default) → no change.

        ``done_provenance={'commit': <sha>, 'note': '…'}`` matches the
        workflow.py:656 convention so downstream consumers (fused-memory
        ``_validate_done_provenance``, ``invalidate_fabricated_shipping_edges.py``,
        Stage 2 reconciliation) can identify the SHA the task ended on. If the
        resolver's branch ref vanishes mid-derivation (rare TOCTOU race),
        ``report.branch_state.sha`` is ``None`` and ``_mark_in_progress_done``
        skips the flip with a WARNING log so operators can spot the race and
        the next sweep retries; reconciliation is best-effort and must not
        abort on this edge case.
        """
        statuses, err = await self.scheduler.get_statuses()
        reverted = 0
        marked_done = 0
        stale_conflicts = 0
        log_prefix = 'Reconcile (mid-run)' if mid_run else 'Reconcile'
        if resolver_failed(statuses, err):
            if err is not None:
                logger.warning(
                    '%s: get_statuses() returned error — aborting sweep (fail-safe)',
                    log_prefix,
                )
            else:
                # Empty-but-no-error is a normal idle state (no tasks yet);
                # DEBUG to avoid recurring noise during early/idle periods.
                logger.debug(
                    '%s: get_statuses() returned empty — aborting sweep (no tasks)',
                    log_prefix,
                )
            return 0

        # R4: sweep both 'in-progress' AND 'blocked' so out-of-band-merged
        # blocked tasks (manual `git merge` while task was blocked) get
        # marked done by the next sweep cycle.  See _RECONCILE_SWEEP_STATUSES
        # for the full list of intentionally excluded statuses.
        for tid, status in statuses.items():
            if status not in _RECONCILE_SWEEP_STATUSES:
                continue
            # task 2243, W10-θ2 step-12: the standalone `if mid_run and
            # is_actively_held(tid): continue` race guard is retired —
            # is_actively_held (dispatched / module-lock held / recent
            # workflow-cancel stamp) is folded into recovery_for's
            # report.live_claimant, and ANY live claimant collapses every
            # _RECOVERY row to the LEAVE default (see
            # _reconcile_one_stranded). is_actively_held is still consulted
            # here (cheap, in-memory, no I/O) purely to decide whether the
            # cancel-stamp prune below is safe right now — it no longer
            # gates whether _reconcile_one_stranded runs at all.
            if mid_run and not self.scheduler.is_actively_held(tid):
                # Grace window elapsed (or never set) — lazily prune so the
                # dict stays bounded for tasks that exit terminal and are never
                # re-dispatched (re-dispatch otherwise clears the stamp).
                # Gated on is_actively_held being FALSE right now (not merely
                # "reached this line") so a task still inside its cancel-grace
                # window isn't pruned out from under recovery_for's own
                # is_actively_held check a few lines down (which would read
                # the cleared stamp and wrongly conclude no live claimant).
                self.scheduler.clear_workflow_cancel(tid)

            try:
                outcome = await self._reconcile_one_stranded(
                    tid, status, mid_run=mid_run,
                )
            except SetTaskStatusRejected as exc:
                # Persistence layer refused our write — escalate directly
                # (task 2243, W10-θ2 step-14: the per-tid strike counter and
                # its MAX_RECONCILE_FAILURES threshold gate are retired; a
                # persistent rejection is now surfaced loudly on the sweep
                # it occurs, not tallied toward a threshold first).  Honest
                # log so future operators can see *why* the task is still
                # stranded instead of the old "marked done" misnomer.
                # Other (unexpected) exception types intentionally propagate
                # so bugs in the sweep surface rather than being silently
                # skipped.
                logger.error(
                    '%s: failed to mark task %s done — %s: %s',
                    log_prefix, tid, exc.error_code, exc.raw,
                )
                if self._escalation_queue:
                    self._escalate_reconcile_failure(tid, exc)
                continue

            if outcome == 'marked_done':
                marked_done += 1
            elif outcome == 'reverted':
                reverted += 1
            elif outcome == 'stale_conflict':
                # task 2677: an honest tally, not a false 'marked_done' —
                # the done-write was refused by the found_on_main
                # provenance-integrity gate (evidence predates reopen_at).
                # _reconcile_one_stranded/_mark_in_progress_done already
                # filed (or folded) the born-at-L2 provenance_conflict
                # escalation; this task stays in-progress awaiting
                # arbitration, so it is deliberately NOT added to the
                # reverted+marked_done "changed" total below.
                stale_conflicts += 1

        if reverted or marked_done or stale_conflicts:
            logger.info(
                '%s: %d stranded task(s) reverted to pending; '
                '%d marked done (branch already on main); '
                '%d held on provenance conflict (done_evidence_stale)',
                log_prefix, reverted, marked_done, stale_conflicts,
            )
        return reverted + marked_done

    def _resolve_task_worktree(self, tid: str) -> Path:
        """Return the on-disk worktree path for *tid*.

        When a WarmLanePool is active and has an assignment for *tid*, returns
        the assigned lane path (e.g. ``worktree_base/_lane-0``).  Otherwise
        falls back to the cold convention ``worktree_base/<tid>``.

        Pool-absent or unmapped tid → identical cold behaviour (byte-compatible
        with pre-pool code).
        """
        pool = self.git_ops.warm_lane_pool
        if pool is not None:
            assigned = pool.assignment_for(tid)
            if assigned is not None:
                return assigned
        return self.git_ops.worktree_base / tid

    @staticmethod
    def _stranded_merge_marker_is_fresh(marker: Any, tip_sha: str) -> bool:
        """Return True iff *marker* records a still-fresh submit for *tip_sha*.

        The verified-green stranded-reaper's race-guard (PRD leaf α §7): a
        ``metadata.stranded_merge_request`` marker suppresses a re-submit only
        when it is a well-formed dict whose ``tip_sha`` equals the current lane
        tip AND whose ``submitted_at`` is within ``_STRANDED_MERGE_RESUBMIT_
        GRACE_S`` of now.  Fail-safe: any malformed / non-dict / unparseable
        marker (or a mismatched tip / stale timestamp) returns False, so the
        caller falls through to a fresh submit rather than a wedged skip — a
        lost marker must never permanently strand the task.
        """
        if not isinstance(marker, dict):
            return False
        if marker.get('tip_sha') != tip_sha:
            return False
        submitted_at = marker.get('submitted_at')
        if not isinstance(submitted_at, str) or not submitted_at:
            return False
        try:
            ts = datetime.fromisoformat(submitted_at)
        except (ValueError, TypeError):
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        # A future timestamp (clock skew) yields a negative age < grace → treated
        # as fresh (skip), which is the safe side: never a duplicate submit.
        age_s = (datetime.now(UTC) - ts).total_seconds()
        return age_s < _STRANDED_MERGE_RESUBMIT_GRACE_S

    async def _maybe_submit_stranded_verified_green(
        self, tid: str, metadata: dict[str, Any],
    ) -> bool:
        """Detect the verified-green shape; on a match submit the lane branch
        DIRECTLY to the merge queue instead of re-pending (PRD leaf α §2.1).

        The incident (reify 5260): the stranded-blocked reaper re-pends a
        verified-green task into a *paused* scheduler that never re-dispatches,
        so the work sits stranded for hours.  When
        :func:`stranded_verified_green.detect_verified_green` matches, we
        instead submit a ``MergeRequest`` (tagged ``source='stranded-reaper'``)
        to the merge queue — which runs even under a scheduler pause — and
        leave the task ``blocked`` while the merge queue's own full verify runs
        as the sole gate (never bypasses, PRD §2.2).

        Returns ``True`` iff a MergeRequest was submitted (the caller then skips
        today's ``stranded_blocked`` re-file and returns None).  Returns
        ``False`` — leaving today's re-file/re-pend path byte-identical — when
        the kill-switch is off, the event-store / merge-queue is unavailable,
        or the shape does not match.  Naturally inert on non-pooled projects:
        no ASSIGNED lane record → ``detect_verified_green`` returns None →
        ``False``.

        This method owns the full ON-MATCH sequence (PRD leaf α §2.1): the
        durable-marker dedup (skip a re-submit while the merge is presumed
        in-flight), the ``MergeRequest`` build + durable-fail done-callback
        registration + ``enqueue_merge_request``, the durable
        ``metadata.stranded_merge_request`` marker stamp, and the
        auto-dismissed ``stranded_blocked`` record escalation.  Done-on-success
        is delegated to the EXISTING found_on_main MARK_DONE path (the marker
        lives in metadata, ignored there — non-interference).
        """
        if (
            not self.config.stranded_verified_green_merge_enabled
            or self.event_store is None
            or self._merge_queue is None
        ):
            return False

        match = await detect_verified_green(
            tid,
            git_ops=self.git_ops,
            event_store=self.event_store,
            worktree_resolver=self._resolve_task_worktree,
        )
        if match is None:
            return False

        # Idempotency / race-guard (PRD leaf α §7): a durable metadata marker
        # records the tip_sha + request_id + submitted_at of the last submit.
        # If a fresh marker for THIS lane tip is already present the merge is
        # presumed still in-flight — return True WITHOUT re-enqueuing so the
        # periodic sweep doesn't pile duplicate requests onto the same branch.
        # A stale marker or an advanced lane tip falls through to a fresh
        # submit (self-healing, restart-safe).
        if self._stranded_merge_marker_is_fresh(
            metadata.get('stranded_merge_request'), match.tip_sha,
        ):
            logger.info(
                'Reconcile: task %s stranded verified-green already submitted '
                '(marker tip=%s) — skipping re-submit (merge presumed '
                'in-flight; task stays blocked)',
                tid, match.tip_sha,
            )
            return True

        from orchestrator.merge_queue import enqueue_merge_request
        from orchestrator.merge_types import (
            MergeOutcome,
            MergeRequest,
            QueuedBranch,
        )

        future: asyncio.Future[MergeOutcome] = (
            asyncio.get_running_loop().create_future()
        )
        req = MergeRequest(
            task_id=tid,
            branch=QueuedBranch.parse(tid, self.config.git.branch_prefix),
            worktree=match.worktree,
            pre_rebased=False,
            task_files=None,
            module_configs=list(self.config.module_configs_or_empty.values()),
            config=self.config,
            result=future,
            snapshot_tip=match.tip_sha,
        )
        # Durable merge/verify FAILURE → born-at-L2 stranded_merge_failed; the
        # callback body is wired in step-14 (a strict no-op for success /
        # transient outcomes, so the branch + lane are preserved by omission).
        req.result.add_done_callback(
            lambda fut: self._on_stranded_merge_done(fut, tid=tid),
        )
        await enqueue_merge_request(
            self._merge_queue, req, self.event_store, source='stranded-reaper',
        )

        # Stamp the durable race-guard marker (PRD leaf α §7).  Best-effort:
        # a lost marker only means a benign re-submit next sweep, so a failed
        # write must never abort the remediation.  Merge-mode write (the
        # scheduler default) touches ONLY this key, preserving siblings; the
        # marker lives in metadata so the found_on_main MARK_DONE path ignores
        # it entirely (non-interference).
        try:
            await self.scheduler.update_task(
                tid,
                {
                    'stranded_merge_request': {
                        'tip_sha': match.tip_sha,
                        'request_id': req.request_id,
                        'submitted_at': datetime.now(UTC).isoformat(),
                    },
                },
            )
        except Exception:
            logger.warning(
                'Reconcile: task %s — failed to stamp stranded_merge_request '
                'marker (benign; only risks a re-submit next sweep)',
                tid, exc_info=True,
            )

        # Record the action (PRD §2.1): file a stranded_blocked escalation and
        # IMMEDIATELY auto-resolve it with a close_only/dismiss disposition.
        # dismiss=True → _resolve_escalation_action maps status='dismissed' to
        # 'close_only' → _on_escalation_resolved's WORKFLOW_NONE branch → NO
        # blocked→pending flip (unlike a resume-resolution, which would trigger
        # the exact Fix #1a re-pend we are replacing).  The dismissed record
        # still lands in the archive as an audit trail, and leaving NO pending
        # escalation keeps the later found_on_main MARK_DONE flip unblocked
        # (its open-escalation guard).  Since PRD leaf δ that guard is
        # _only_merge_remediable, so a pending stranded_blocked would no longer
        # block the flip either — the dismiss is now belt-and-braces for THIS
        # record rather than load-bearing, and is kept because the archive
        # audit trail plus a zero-pending-escalation task is the honest state:
        # nothing here awaits a human.
        if self._escalation_queue is not None:
            from escalation.models import Escalation  # noqa: PLC0415

            esc = Escalation(
                id=self._escalation_queue.make_id(tid),
                task_id=tid,
                agent_role='harness-stranded-blocked-reaper',
                severity='blocking',
                category='stranded_blocked',
                summary=(
                    f'Verified-green stranded remediation: task {tid} branch '
                    f'submitted directly to the merge queue (no re-pend).'
                )[:200],
                detail=(
                    f'Task {tid} was blocked with verified-green, all-steps-done '
                    f'lane work (branch tip {match.tip_sha}) but no open '
                    f'escalation and no live claimant — the stranded-verified-'
                    f'green shape (PRD leaf α §2.1).  Rather than re-pend it into '
                    f'a possibly-paused scheduler, the branch was submitted '
                    f'directly to the merge queue (request_id={req.request_id}, '
                    f'source=stranded-reaper).  The merge queue\'s own full '
                    f'verify is the sole gate; the task stays blocked and is '
                    f'marked done by the existing found_on_main path once the '
                    f'merge lands (or a stranded_merge_failed L2 is filed on a '
                    f'durable merge/verify failure).'
                ),
                suggested_action='manual_intervention',
                level=1,
            )
            self._escalation_queue.submit(esc)
            self._escalation_queue.resolve(
                esc.id,
                resolution=(
                    f'submitted branch to merge queue '
                    f'(request_id={req.request_id}, source=stranded-reaper); '
                    f'merge-queue verify is the gate — task stays blocked'
                ),
                dismiss=True,
                resolved_by='harness-stranded-blocked-reaper',
            )
            if self.event_store is not None:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=tid,
                    data={
                        'escalation_id': esc.id,
                        'category': 'stranded_blocked',
                        'severity': 'blocking',
                        'level': 1,
                        'reason': 'stranded-verified-green-submitted',
                    },
                )
            logger.warning(
                'Reconcile: task %s stranded verified-green — submitted branch '
                'to merge queue (request_id=%s, source=stranded-reaper); filed+'
                'dismissed record L1 %s (task stays blocked, no re-pend)',
                tid, req.request_id, esc.id,
            )
        return True

    # Every MergeOutcome.status is classified into exactly one of the two
    # frozensets below; the pair MUST exhaust MergeOutcome's status Literal.
    # That partition is enforced at CI time by
    # test_stranded_verified_green.test_status_sets_exhaust_merge_outcome_vocabulary
    # (adding a new status without classifying it fails that test), and — as a
    # runtime backstop — a status found in NEITHER set is treated LOUDLY as a
    # durable failure in _on_stranded_merge_done.  Together these guarantee a
    # new/unclassified outcome can never silently no-op into a stranded task
    # with no operator signal (the exact silent-strand class PRD leaf α fixes).
    #
    # Durable merge/verify failure outcomes for a stranded-reaper submission —
    # a stale-green branch failing the merge queue's own verify lands here and
    # warrants a born-at-L2 (PRD leaf α §2.2).
    _DURABLE_MERGE_FAILURE_STATUSES: frozenset[str] = frozenset({
        'conflict', 'blocked', 'error', 'unknown_branch',
        'unmerged_state', 'stash_failed', 'wip_recovery_no_advance',
    })
    # Success / transient outcomes — a strict no-op: the happy 'done' is
    # delivered by the existing found_on_main path, and a transient/superseded
    # outcome is re-driven by a later sweep.  The branch + lane are preserved
    # by omission (the callback never touches them).
    _SUCCESS_TRANSIENT_MERGE_STATUSES: frozenset[str] = frozenset({
        'done', 'already_merged', 'done_wip_recovery', 'superseded',
        'wip_halted',
    })

    # Escalation categories a MERGE can itself remediate (PRD leaf δ §2.2).
    #
    # The stranded-blocked reaper's own `stranded_blocked` L1 is filed to
    # REQUEST exactly the remediation the verified-green auto-merge performs —
    # so letting it veto that merge is an anti-synergy: the escalation asking
    # for the merge blocks the merge.  Membership here means "an open
    # escalation of this class does NOT veto the sweep-side self-heal", and
    # nothing more: the merge is still gated by detect_verified_green's 3-part
    # shape check and the merge queue's own re-verify (§2.2 "never bypasses").
    #
    # Deliberately MINIMAL — widen only with evidence:
    #   * `stranded_merge_failed` is EXCLUDED on purpose.  It is the DURABLE
    #     merge/verify-failure born-at-L2 (see _file_stranded_merge_failed): a
    #     re-merge cannot remediate a branch that already failed the queue's
    #     verify, so a task carrying only that escalation must keep vetoing or
    #     the reaper would re-submit into the same failure.
    #   * every human-concern class (design_concern / task_failure /
    #     review_issues / operator-action / infra_issue / ...) is excluded by
    #     omission — it names a problem a merge does not fix, and must keep
    #     holding the task for its handler.
    MERGE_REMEDIABLE_ESC_CATEGORIES: frozenset[str] = frozenset({
        'stranded_blocked',
    })

    @staticmethod
    def _only_merge_remediable(
        open_escalations: Sequence[EscalationRef],
    ) -> bool:
        """Are *open_escalations* ALL of a merge-remediable class?

        The single category authority for the relaxed verified-green veto
        (INV-5): called at both sweep-side upgrade clauses in
        :meth:`_reconcile_one_stranded` in place of the former
        ``not report.open_escalations``.

        Vacuously ``True`` for an empty list — so a task with no open
        escalation classifies exactly as it does today.  ``False`` as soon as
        ONE escalation falls outside :attr:`MERGE_REMEDIABLE_ESC_CATEGORIES`,
        preserving the safety invariant that a human-concern escalation still
        vetoes the self-heal.
        """
        return all(
            ref.category in Harness.MERGE_REMEDIABLE_ESC_CATEGORIES
            for ref in open_escalations
        )

    def _on_stranded_merge_done(
        self, fut: asyncio.Future, *, tid: str,
    ) -> None:
        """Done-callback for a stranded-reaper MergeRequest (durable-fail → L2).

        Sync (fires on the loop when the MergeRequest future resolves).  Derives
        the terminal outcome and, on a DURABLE merge/verify failure, schedules
        :meth:`_file_stranded_merge_failed` (a born-at-L2) via
        ``_schedule_coro_threadsafe``; success / transient outcomes (and a
        cancelled/abandoned future) are a strict no-op, so the branch + lane are
        preserved by omission.  Wrapped fail-safe: any error is logged and never
        propagated (mirrors ``enqueue_merge_request._on_finalized``).
        """
        try:
            from orchestrator.merge_types import MergeOutcome  # noqa: PLC0415

            if fut.cancelled():
                return  # abandoned/superseded — transient, no-op
            exc = fut.exception()
            if exc is not None:
                outcome = MergeOutcome(
                    status='error', reason=f'merge future raised: {exc}',
                )
            else:
                outcome = fut.result()
            status = outcome.status
            if status in self._SUCCESS_TRANSIENT_MERGE_STATUSES:
                return  # success/transient → strict no-op (branch+lane preserved)
            if status not in self._DURABLE_MERGE_FAILURE_STATUSES:
                # UNCLASSIFIED status — a new MergeOutcome.status reached the
                # callback without being sorted into either set (the
                # exhaustiveness test guards this at CI time; this is the
                # runtime backstop).  Treat an unknown outcome as a durable
                # failure and file the L2 rather than silently no-op'ing: a
                # stale-green branch that fails the merge-queue verify must
                # never strand with no operator signal (loud-over-silent — the
                # exact class PRD leaf α exists to fix).
                logger.error(
                    '_on_stranded_merge_done: UNCLASSIFIED merge status %r for '
                    'task %s — treating as a durable failure and filing a '
                    'born-at-L2 (classify it into _DURABLE_MERGE_FAILURE_STATUSES '
                    'or _SUCCESS_TRANSIENT_MERGE_STATUSES)', status, tid,
                )
            self._schedule_coro_threadsafe(
                self._file_stranded_merge_failed(tid, outcome),
                label=(
                    f'stranded-merge-failed task {tid} (status={status})'
                ),
            )
        except Exception:
            logger.warning(
                '_on_stranded_merge_done: failed to process merge outcome for '
                'task %s — no L2 filed (fail-safe)', tid, exc_info=True,
            )

    async def _file_stranded_merge_failed(
        self, tid: str, outcome: Any,
    ) -> None:
        """File a BORN-AT-L2 ``stranded_merge_failed`` for a durably-failed
        stranded-reaper merge (PRD leaf α §2.2; mirrors
        :meth:`_block_and_escalate_delivered_check`).

        ``agent_role='harness-stranded-blocked-reaper'`` (a harness-sentinel
        role) + ``severity='critical'`` + ``level=2`` make the record BORN AT
        L2 — it bypasses the auto-watcher and routes straight to a human.  A
        persistently-false verified-green claim is a "someone must look now"
        condition.

        Unlike ``_block_and_escalate_delivered_check`` this does NOT touch task
        status, the lane, or the branch: the task is already ``blocked`` and the
        branch + lane are preserved for inspection (preservation by omission).
        Deduped via a scoped ``get_by_task(tid, status='pending', level=2,
        agent_role=...)`` read filtered to this category.  No-ops when no
        escalation queue is attached (bare-Harness unit tests stay green).
        """
        if self._escalation_queue is None:
            return

        existing = self._escalation_queue.get_by_task(
            tid, status='pending', level=2,
            agent_role='harness-stranded-blocked-reaper',
        )
        if any(e.category == 'stranded_merge_failed' for e in existing):
            logger.warning(
                'stranded_merge_failed L2 already open for task %s — '
                'suppressing duplicate', tid,
            )
            return

        from escalation.models import Escalation  # noqa: PLC0415

        status = getattr(outcome, 'status', 'unknown')
        reason = getattr(outcome, 'reason', '') or ''
        esc = Escalation(
            id=self._escalation_queue.make_id(tid),
            task_id=tid,
            agent_role='harness-stranded-blocked-reaper',
            severity='critical',
            category='stranded_merge_failed',
            summary=(
                f'Verified-green stranded merge FAILED for task {tid} '
                f'(status={status}) — manual intervention.'
            )[:200],
            detail=(
                f'The verified-green stranded remediation submitted task {tid}\'s '
                f'branch directly to the merge queue, but the merge/verify failed '
                f'durably: status={status}, reason={reason!r}.  The task remains '
                f'blocked and the branch + lane are preserved (untouched) for '
                f'inspection.  A stale-green branch failing the merge queue\'s '
                f'own full verify lands here — the remediation never bypasses '
                f'that gate (PRD leaf α §2.2).  Investigate and re-drive or '
                f'triage manually.'
            ),
            suggested_action='manual_intervention',
            level=2,
        )
        self._escalation_queue.submit(esc)
        if self.event_store is not None:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=tid,
                data={
                    'escalation_id': esc.id,
                    'category': 'stranded_merge_failed',
                    'severity': 'critical',
                    'level': 2,
                    'reason': f'stranded-merge-{status}',
                },
            )
        logger.warning(
            'Filed born-at-L2 stranded_merge_failed %s for task %s '
            '(status=%s) — task stays blocked, branch+lane preserved',
            esc.id, tid, status,
        )

    async def _branch_is_degenerate(
        self, branch: str, metadata: dict[str, Any],
    ) -> bool:
        """Return True iff the branch is a provisioning-only degenerate branch.

        A branch is degenerate when its live tip SHA equals the recorded
        branch_base_sha (#1226), meaning zero commits were ever pushed beyond
        the creation point.  Called from the MARK_DONE_WITH_PROVENANCE
        degenerate-branch refinement in _reconcile_one_stranded (task 2243,
        W10-θ2) — downgrading a would-be mark-done to a revert when
        TaskGroundTruth's resolver classifies a zero-commit branch as
        ON_MAIN — and from _revert_in_progress_if_no_live_claimant's
        infra-held guard, which share this same degeneracy signal.

        Returns False when:
        - branch_base_sha is absent or not a valid 40-hex SHA (backward compat
          for pre-#1226 tasks or tasks whose metadata write failed transiently);
        - resolve_branch_sha returns None (branch ref vanished mid-sweep —
          treat as non-degenerate so the caller falls through to escalate); or
        - the live tip has advanced past the recorded base SHA.
        """
        branch_base_sha = metadata.get('branch_base_sha')
        if not _is_valid_sha_40(branch_base_sha):
            return False
        branch_tip_sha = await self.git_ops.resolve_branch_sha(branch)
        return branch_tip_sha is not None and branch_tip_sha == branch_base_sha

    async def _reconcile_one_stranded(
        self, tid: str, status: str, *, mid_run: bool,
    ) -> str | None:
        """Reconcile a single stranded task. Returns 'marked_done', 'reverted', or None.

        Raises ``SetTaskStatusRejected`` if the persistence layer refuses the
        recovery write — caller handles failure counting + escalation.

        Early-return guards (return None without touching the worktree):
          - open L1 escalation veto (~line 1598): human handoff in progress
          - merge-deferred guard (below): task is train-parked (PRD § 9.8);
            the worktree must survive intact for the train-merge worker.
            Mirrors the open-L1 veto pattern — explicit early-return as
            belt-and-suspenders on top of the _RECONCILE_SWEEP_STATUSES filter.
          - mid-run actively-held short-circuit (below): cheap pre-check for
            a task recovery_for would classify LEAVE anyway (review
            amendment, task 2243 W10-θ2 reviewer_comprehensive #1).
        """
        # PRD § 9.8 — train-parked tasks must never be reverted or have their
        # worktree cleaned up; the train-merge worker owns their lifecycle.
        # This guard fires BEFORE any branch compute / get_task / cleanup call.
        if status == 'merge-deferred':
            logger.info(
                'Reconcile: task %s is merge-deferred (train-parked); '
                'leaving worktree intact',
                tid,
            )
            return None

        # Cheap in-memory short-circuit (review amendment, task 2243 W10-θ2
        # reviewer_comprehensive #1): mid-run, an actively-held task (the
        # scheduler's own dispatched / module-lock-held / recent-cancel
        # bookkeeping — the exact signal recovery_for folds into
        # report.live_claimant) is GUARANTEED to classify LEAVE — no
        # _RECOVERY row (task_ground_truth.py) has live_claimant=True, so any
        # live claimant always falls through to the LEAVE default, and
        # neither sweep-side upgrade above can apply once one is established
        # (both require report.live_claimant is None). The deep
        # `is_actively_held` re-check below (kept as defense-in-depth for
        # mid_run=False and TOCTOU races) would reach the same conclusion —
        # but only AFTER paying for get_task + recovery_for's derive_truth
        # (a MergeProvenance lookup and, on a journal miss, up to two git
        # subprocess calls). This sweep runs periodically over every
        # in-progress/blocked task, so that cost would otherwise scale with
        # the number of concurrently-running (i.e. normal, not stranded)
        # tasks. Outcome-identical to letting recovery_for run — this only
        # changes WHERE the check happens, not what it decides. The driver
        # (_reconcile_stranded_in_progress) still calls this function
        # unconditionally for every candidate (task 2243 step-12 parity — no
        # driver-level continue reintroduced).
        if mid_run and self.scheduler.is_actively_held(tid):
            return None

        branch = f'{self.git_ops.config.branch_prefix}{tid}'

        # Fetch task metadata once for the downstream blocked/revert paths
        # below (unrelated to the mark-done decision, which now derives its
        # own task row internally via TaskGroundTruth.derive_truth — a second,
        # accepted fetch until a later step folds this one away too).
        # Scheduler.get_task normalises metadata at the boundary
        # (scheduler.py:_normalize_task_metadata), so task['metadata'] is always a
        # dict whenever task is not None.  `or {}` collapses any residual None value
        # (e.g. a manually-constructed task dict that bypasses normalisation); the
        # load-bearing guard against task itself being absent is `if task else {}`.
        task = await self.scheduler.get_task(tid)
        metadata = (task.get('metadata') or {}) if task else {}

        # Ground-truth mark-done decision (task 2243, W10-θ2): recovery_for
        # composes derive_truth (journal-first branch-state resolution, TG-1)
        # with the _RECOVERY classification table (TG-2), retiring the inline
        # is_ancestor/find_task_citation_commit/find_merge_marker archaeology
        # (including the citation-guard and stale-marker/prior-incarnation
        # checks) that used to live here.
        report, action = await self._get_ground_truth().recovery_for(tid)

        # θ1's _RECOVERY table only maps ON_MAIN/GONE_WITH_MERGE_MARKER to
        # MARK_DONE for TaskStatus.IN_PROGRESS — it carries no row for a
        # stranded 'blocked' task on the same evidence.  The pre-migration
        # sweep marks 'blocked' done here too (R4 — see the
        # _RECONCILE_SWEEP_STATUSES comment above: "out-of-band-merged
        # blocked tasks... get marked done"): a human `git merge`-ing a
        # blocked task's branch out-of-band must still self-heal to done.
        # This is a thin sweep-side upgrade — mirroring the
        # already-established degenerate-branch refinement pattern
        # (_branch_is_degenerate below) — rather than a change to θ1's
        # reviewed table (design decision, task 2243; esc-2243-4).
        #
        # The open-escalation clause is _only_merge_remediable, not the former
        # `not report.open_escalations` (PRD leaf δ): a task whose branch landed
        # while it was still blocked is often held by the reaper's OWN
        # stranded_blocked — the escalation that ASKED for this landing — and
        # letting it veto the self-heal pins the task blocked forever after its
        # work is already on main.  Any non-remediable (human-concern)
        # escalation still yields False and leaves the task alone, and an empty
        # list is still True, so every other task classifies exactly as before.
        if (
            action == RecoveryAction.LEAVE
            and status == 'blocked'
            and report.live_claimant is None
            and self._only_merge_remediable(report.open_escalations)
            and report.branch_state.kind in (
                BranchStateKind.ON_MAIN, BranchStateKind.GONE_WITH_MERGE_MARKER,
            )
        ):
            action = RecoveryAction.MARK_DONE_WITH_PROVENANCE

        # θ1's _RECOVERY table only maps GONE_NO_MARKER to RE_FILE_ESCALATION
        # for a stranded 'blocked' task — it carries no row for
        # EXISTS_OFF_MAIN (branch ref still present, just never merged).
        # Pre-migration, the sweep's blocked-escalation backstop fired on ANY
        # 'blocked' task reaching this point on NO on-main evidence,
        # regardless of which of the two no-evidence branch shapes it was
        # in. Second thin sweep-side upgrade — same pattern as the R4
        # MARK_DONE upgrade above — rather than a change to θ1's reviewed
        # table (design decision, task 2243; esc-2243-5).
        #
        # The open-escalation clause is _only_merge_remediable, not the former
        # `not report.open_escalations` (PRD leaf δ): this is the branch shape a
        # verified-green-but-never-merged task is in, and the escalation
        # holding it is usually the reaper's OWN stranded_blocked — filed to
        # REQUEST exactly the merge the verified-green gate below performs.
        # Letting that request veto its own remediation was the anti-synergy δ
        # closes.  Any non-remediable (human-concern) escalation still yields
        # False here and leaves the task alone, and an empty list is still
        # True, so every other task classifies exactly as before.
        if (
            action == RecoveryAction.LEAVE
            and status == 'blocked'
            and report.live_claimant is None
            and self._only_merge_remediable(report.open_escalations)
            and report.branch_state.kind == BranchStateKind.EXISTS_OFF_MAIN
        ):
            action = RecoveryAction.RE_FILE_ESCALATION

        if action == RecoveryAction.MARK_DONE_WITH_PROVENANCE:
            # Degenerate-branch refinement (task 2243, W10-θ2; RESOLVER GAP
            # design decision): θ1's _RECOVERY table has no degenerate-branch
            # guard — a zero-commit branch (tip == branch_base_sha) that is
            # an ancestor of main classifies ON_MAIN -> MARK_DONE the same as
            # a genuinely merged branch. Extending θ1's resolver would mean
            # threading task metadata into _resolve_branch_state, undoing its
            # reviewed branch-state/task-row concurrency split — so this
            # thin, sweep-side check downgrades the decision instead. Safe
            # for journal hits: a genuinely merged branch is never
            # degenerate, so this never fires for a real MergeProvenance
            # journal row.
            if (
                await self._branch_is_degenerate(branch, metadata)
                or await self.git_ops.warm_lane_ref_is_degenerate(tid)
            ):
                # A degenerate branch carries ZERO task work, so MARK_DONE
                # would phantom-complete a task that never actually landed
                # anything. An in-progress incarnation with no live claimant
                # (already established by recovery_for reaching this action)
                # is recovered by reverting to pending so the scheduler
                # re-dispatches it. 'blocked' keeps the leave-alone
                # discipline: only flip to done on positive evidence, which a
                # degenerate branch is not — matching pre-migration behavior.
                if status == 'in-progress':
                    return await self._revert_in_progress_if_no_live_claimant(
                        tid, mid_run=mid_run, metadata=metadata, status=status,
                    )
                return None

            # BranchState docstring invariant (task_ground_truth.py): sha is
            # always populated for ON_MAIN / GONE_WITH_MERGE_MARKER — the
            # only variants that reach MARK_DONE_WITH_PROVENANCE. Guard it
            # explicitly (review finding, task 2678 amendment): passing
            # candidate_sha=None below would silently switch
            # validate_landing_evidence from CANDIDATE to DISCOVERY mode
            # instead of failing loudly on the violated invariant — a
            # materially different code path (citation discovery + lineage
            # against `branch`) than the effect-only check this call site
            # intends. The one known way to reach sha=None is a TOCTOU in
            # the git-archaeology fallback (_resolve_branch_state: the
            # branch ref is deleted between the is_ancestor and
            # resolve_branch_sha calls) — practically near-unreachable, so
            # fail loud rather than silently degrade (project norm).
            assert report.branch_state.sha is not None, (
                f'BranchState invariant violated for task {tid}: kind='
                f'{report.branch_state.kind!r} carries no sha'
            )

            on_main = report.branch_state.kind == BranchStateKind.ON_MAIN

            # FIX 1' effect-present refinement (task 2500/2678): a cited
            # ON_MAIN commit or merge marker stays an ancestor of main
            # forever — ancestry is immutable history — even after a LATER
            # commit on main reverts exactly the paths it touched (the
            # found_on_main post-hoc-revert blind spot; reify
            # esc-5179-3/esc-5181-2). Sibling to the degenerate-branch
            # refinement above: same downgrade shape, flip only on positive
            # evidence. Delegates to the shared validate_landing_evidence
            # helper (task 2678, INV-5) in CANDIDATE mode, applied
            # UNIFORMLY to BOTH the on_main and marker sub-cases —
            # attribution is already established by recovery_for's
            # ground-truth report, so only the effect-present guard
            # remains. Previously this check was gated on ``on_main`` only,
            # so a branch-deleted merge marker whose effect had been
            # reverted at current main HEAD skipped it entirely and was
            # stamped done unconditionally (the task-1175 clobber,
            # reproduced inside this sweep). No escalation on reject here
            # (unlike the dispatch-gate marker/content-equivalence paths):
            # the evidence is already ground-truth-attributed by
            # recovery_for, and the existing revert-to-pending / leave-
            # blocked recovery self-heals without a human (design decision,
            # task 2678).
            verdict = await validate_landing_evidence(
                self.git_ops, tid, branch,
                branch_tip_sha=None,
                candidate_sha=report.branch_state.sha,
            )
            if not verdict.accepted:
                logger.warning(
                    'Reconcile: task %s %s evidence sha %s is an ancestor '
                    'of main but its effect is not present at current HEAD '
                    '(post-hoc revert) — not marking done',
                    tid, 'ON_MAIN' if on_main else 'merge-marker',
                    report.branch_state.sha,
                )
                if status == 'in-progress':
                    return await self._revert_in_progress_if_no_live_claimant(
                        tid, mid_run=mid_run, metadata=metadata, status=status,
                    )
                return None

            # Delivered-capability ground-truth guard (task 2794). Sibling to
            # the effect-present refinement above (same downgrade shape): git
            # attribution and the effect-present check prove a merge advanced
            # main and that the cited commit's paths survive at HEAD — but NOT
            # that THIS task's OWN declared capability
            # (metadata.delivered_checks) is actually complete on main. A
            # journal row / citation / merge marker attributes a landing; it
            # does not certify the deliverable. So — AFTER all
            # attribution/effect guards, applied UNIFORMLY to every evidence
            # source (ON_MAIN, GONE_WITH_MERGE_MARKER, MergeProvenance journal)
            # funneling through this one arm — verify the declared capability
            # is present on main before stamping found_on_main. Kept as an
            # early return on failed/errored (in-progress -> revert-to-pending
            # for re-dispatch; blocked -> leave alone, blocked discipline) so
            # _mark_in_progress_done below is structurally reachable ONLY on
            # the all_delivered fall-through — never behind a mutable boolean a
            # later refactor could drift out of sync. Gated on the enabled kill
            # switch AND truthy delivered_checks so check-less tasks keep their
            # exact pre-2794 attribution+effect-present bar (kill-switch parity
            # with the dependent-side dispatch gate's documented inertness).
            delivered_checks = metadata.get('delivered_checks')
            if delivered_checks and self.config.delivered_checks.enabled:
                try:
                    main_sha = await self.git_ops.get_main_sha()
                except Exception:
                    # get_main_sha() raises on git error — the caller owns the
                    # fail-safe: no mark, no revert, retry when main is
                    # readable again (sibling of the ERRORED outcome below).
                    logger.warning(
                        'Reconcile: task %s carries delivered_checks but the '
                        'main SHA could not be resolved — deferring mark-done '
                        '(fail-safe), will retry next sweep',
                        tid, exc_info=True,
                    )
                    return None
                if not main_sha:
                    logger.warning(
                        'Reconcile: task %s carries delivered_checks but '
                        'get_main_sha() returned empty — deferring mark-done '
                        '(fail-safe), will retry next sweep',
                        tid,
                    )
                    return None
                dc_verdict: DeliveredChecksVerdict = await verify_delivered_checks_on_main(
                    delivered_checks,
                    project_root=str(self.config.project_root),
                    main_sha=main_sha,
                    check_timeout_secs=self.config.delivered_checks.check_timeout_secs,
                )
                if dc_verdict.outcome == 'failed':
                    failed_check = dc_verdict.failed_check or {}
                    is_grep = failed_check.get('kind') == 'grep'
                    logger.warning(
                        'Reconcile: task %s delivered-check %r (%s=%r) is '
                        'absent from main@%s — declared capability not present, '
                        'not marking done',
                        tid, failed_check.get('name'),
                        'pattern' if is_grep else 'script',
                        failed_check.get('pattern') if is_grep else failed_check.get('script'),
                        main_sha,
                    )
                    if status == 'in-progress':
                        return await self._revert_in_progress_if_no_live_claimant(
                            tid, mid_run=mid_run, metadata=metadata, status=status,
                        )
                    return None
                if dc_verdict.outcome == 'errored':
                    # Some check could not be evaluated and none FAILED — make
                    # no claim either way; fail-safe wait (no mark, no revert).
                    logger.warning(
                        'Reconcile: task %s delivered-checks could not be '
                        'evaluated at main@%s (errored) — deferring mark-done '
                        '(fail-safe), will retry next sweep',
                        tid, main_sha,
                    )
                    return None
                # dc_verdict.outcome == 'all_delivered' -> the declared
                # capability is present; fall through to note-building /
                # _mark_in_progress_done below, unchanged.

            if status == 'blocked':
                note = (
                    'reconcile: branch on main while task was blocked (out-of-band merge)'
                    if on_main
                    else 'reconcile: merge marker found on main while task was blocked'
                )
            else:
                note = (
                    'reconcile: branch already on main when stranded in-progress'
                    if on_main
                    else 'reconcile: branch deleted but merge marker found on main'
                )
            reason = 'branch-already-on-main' if on_main else 'branch-deleted-marker-found'
            # task 2677: a prior sweep may have already had this exact
            # (task_id, reopen_at) rejected as done_evidence_stale — the
            # sink's in-memory memo makes that terminal-for-this-tick so we
            # don't re-attempt the same doomed write every sweep cycle.
            if self._provenance_conflict_sink.should_skip(
                tid, reopen_at=metadata.get('reopen_at'),
            ):
                return 'stale_conflict'
            marked = await self._mark_in_progress_done(
                tid, report.branch_state.sha, note, reason,
            )
            return 'marked_done' if marked else 'stale_conflict'

        # Ground-truth revert decision (task 2243, W10-θ2): the _RECOVERY
        # table maps a stranded in-progress task with no live claimant and no
        # on-main landing evidence (branch EXISTS_OFF_MAIN or GONE_NO_MARKER)
        # straight to REVERT_TO_PENDING — retiring the inline branch-state
        # derivation this dispatch used to require. Step-16 also retired the
        # applier's own plan.lock owner_pid liveness re-derivation — it now
        # trusts recovery_for's live_claimant and owns only the worktree
        # cleanup / lock unlink / retention bookkeeping side effects (see
        # _revert_in_progress_if_no_live_claimant).
        if action == RecoveryAction.REVERT_TO_PENDING:
            return await self._revert_in_progress_if_no_live_claimant(
                tid, mid_run=mid_run, metadata=metadata, status=status,
            )

        # No on-main evidence.  For 'blocked' tasks, leave the row alone —
        # blocked is a deliberate state and we only flip it to done on
        # observed evidence.
        if status == 'blocked':
            # Fix #1b — defense-in-depth backstop for the stranded-blocked gap.
            # A task left 'blocked' with NO open escalation AND no live
            # claimant is an orphaned recovery: its blocking escalation was
            # resolved directly with no live workflow to re-pend it (3576,
            # 2026-05-29).  blocked→pending recovery is owned exclusively by
            # the active-workflow resume path and the L2-cascade / direct-resolve
            # event flips (Fix #1a); a 'blocked' row with no escalation at all
            # has no event left to fire, so it would sit forever.
            #
            # We RE-FILE a single L1 — we NEVER change status.  Re-filing (not
            # re-pending) cannot yank a deliberate release_workflow blocked-park:
            # a parked /unblock task either still has an open L1 (→ recovery_for
            # classifies LEAVE, no re-file) or, if a human resolved its L1 and
            # is mid-merge, the re-filed L1 is harmless noise they dismiss.
            # Fix #1a then performs the actual re-pend when this L1 is resolved.
            #
            # Category: 'stranded_blocked' (PRD-3 task ε, 2026-06-04).
            # The escalation-watcher (task θ) auto-resolves stranded_blocked
            # L1s with action='resume', triggering this exact Fix #1a path:
            # _on_escalation_resolved → _cascade_unblock_member →
            # blocked→pending re-pend.
            #
            # task_kind == 'deterministic' tasks are excluded here (task 2074
            # amendment) and delegated exclusively to
            # _recover_stranded_deterministic_task via the deterministic-recon
            # sweep — that sweep's live-systemd-health check is the reason
            # deterministic-deploy recovery is routed through it instead of
            # this generic reaper.
            #
            # Dispatch + dedup (task 2243, W10-θ2): action ==
            # RE_FILE_ESCALATION (recovery_for's row g) already means no live
            # claimant (is_actively_held / db claimant / plan.lock) and no
            # escalation open at any level — both folded by the resolver in
            # place of the old in-sweep _escalation_events /
            # workflow_cancel_recent / get_by_task guards.  Once filed, the
            # next sweep sees the new pending escalation in
            # report.open_escalations and recovery_for classifies LEAVE —
            # self-dedup without a separate check here.
            if (
                action == RecoveryAction.RE_FILE_ESCALATION
                and self.config.stranded_blocked_escalate_enabled
                and self._escalation_queue is not None
                and metadata.get('task_kind') != 'deterministic'
            ):
                # Verified-green stranded remediation (PRD leaf α §2.1): BEFORE
                # re-filing, check whether this blocked task's warm lane holds
                # verified-green, all-steps-done work.  On a match we submit the
                # branch DIRECTLY to the merge queue (which runs even under a
                # scheduler pause — the incident's root cause) and record the
                # action via an auto-dismissed escalation, leaving the task
                # blocked; return None to SKIP today's re-file/re-pend.  On a
                # non-match (or kill-switch off / non-pooled project) this
                # returns False and we fall through to the unchanged re-file
                # path below — byte-identical for every non-matching task.
                if await self._maybe_submit_stranded_verified_green(tid, metadata):
                    return None

                # Dedup guard (PRD leaf δ): reaching here with an escalation
                # ALREADY open means the relaxed veto let us through on a
                # merge-remediable one (the two clauses above are the only way
                # in: θ1's resolver rows all require an empty list, and the
                # EXISTS_OFF_MAIN upgrade now requires _only_merge_remediable)
                # — and the verified-green submit just declined (non-match).
                # Re-filing would stack a SECOND stranded_blocked L1 on a task
                # that already has one pending, so leave the existing
                # escalation for its handler.  A plain truthiness check
                # suffices: merge-remediable-ness is already established
                # upstream, keeping _only_merge_remediable the sole category
                # authority (INV-5).  The empty case — every task that reached
                # here before δ — falls through to the unchanged re-file.
                if report.open_escalations:
                    return None

                from escalation.models import Escalation

                esc = Escalation(
                    id=self._escalation_queue.make_id(tid),
                    task_id=tid,
                    agent_role='harness-stranded-blocked-reaper',
                    severity='blocking',
                    category='stranded_blocked',
                    summary=(
                        f'Stranded blocked: task {tid} blocked with no open '
                        f'escalation and no active workflow — likely an orphaned '
                        f'recovery; needs re-pend or triage.'
                    )[:200],
                    detail=(
                        f'Task {tid} is in status=blocked with no commit on '
                        f'main (no out-of-band merge), no pending escalation, '
                        f'and no active workflow slot.  This is the stranded-'
                        f'blocked-on-direct-resolution shape: a blocking '
                        f'escalation was resolved directly while no workflow '
                        f'owned the re-pend, so blocked→pending never fired.\n\n'
                        f'Resolve this L1 to re-pend the task '
                        f'(Harness._on_escalation_resolved / Fix #1a flips it '
                        f'blocked→pending on resolution), or triage/cancel the '
                        f'task if it should stay parked.'
                    ),
                    suggested_action='manual_intervention',
                    level=1,
                )
                self._escalation_queue.submit(esc)
                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_created,
                        task_id=tid,
                        data={
                            'escalation_id': esc.id,
                            'category': 'stranded_blocked',
                            'severity': 'blocking',
                            'level': 1,
                            'reason': 'stranded-blocked-backstop',
                        },
                    )
                logger.warning(
                    'Reconcile: task %s stranded blocked with no open '
                    'escalation/active workflow — re-filed L1 %s (no status '
                    'change)', tid, esc.id,
                )
            return None

        # Deploy-phase-tracked in-progress tasks are never auto-recovered by
        # this generic reaper (review amendment, task 2243 W10-θ2
        # reviewer_comprehensive #2). θ1's _RECOVERY table (task_ground_truth.py)
        # requires deploy_phase is None for EVERY MARK_DONE_WITH_PROVENANCE /
        # REVERT_TO_PENDING row (a/b/c/d) — so an in-progress task carrying a
        # deploy_phase can only reach this point via the LEAVE default
        # (VERIFIED / FAILED / SCHEDULED / ESCALATED / DONE, or RAN paired
        # with a branch state other than GONE_NO_MARKER) or via
        # RE_FILE_ESCALATION (row h — GONE_NO_MARKER + RAN, the D1
        # crashed-mid-deploy shape). Both are deliberate: the table's
        # "Deliberately-unmapped deploy phases" comment explains FAILED /
        # ESCALATED already own a mandatory DS-2 recovery path, VERIFIED /
        # DONE are terminal-success, and reverting (or phantom-completing) a
        # stranded deploy could re-trigger one that already took effect. Row
        # h's RE_FILE_ESCALATION is likewise handled elsewhere (the runner's
        # own born-at-L2 escalation), not by this in-progress path. Without
        # this guard, either action fell through to the generic revert
        # below, silently overriding the table's LEAVE/defer-to-DS-2 intent
        # — most notably for a deterministic deploy task, which carries no
        # task_kind != 'deterministic' exclusion on this path (unlike the
        # 'blocked' branch above).
        if report.deploy_phase is not None:
            return None

        # task 2243, W10-θ2 step-12: is_actively_held is the scheduler's own
        # in-memory dispatch bookkeeping (dispatched / module-lock held /
        # recent workflow-cancel stamp) — unambiguous, and not something the
        # applier below independently re-derives (that was the now-deleted
        # driver-level guard's job). Trust it directly rather than falling
        # through.
        if self.scheduler.is_actively_held(tid):
            return None

        # An open escalation at ANY level (not just L1 — the resolver's row
        # (f) folds every level) is the deliberate human/automation-handoff
        # signal: don't reap it. Replaces the old has_open_l1(tid) veto
        # (L1-only), which missed an L2-only open escalation and fell
        # through to an incorrect revert.
        if report.open_escalations:
            return None

        # A live claimant is a deliberate leave-alone (task 2243, W10-θ2
        # step-16 — this blanket check replaces the applier's own owner_pid
        # re-derivation, now retired), EXCEPT the R3 mid-run exception: a
        # plan.lock's owner_pid is almost always the harness's OWN pid (it
        # stamps the lock on behalf of whichever workflow it runs), so an
        # alive owner_pid there proves nothing about whether that workflow
        # is still running — is_actively_held (already checked above, and
        # already False here) is the authoritative live signal mid-run, so a
        # PLAN_LOCK-sourced claimant must still fall through to recovery
        # mid-run (a workflow that exited without releasing the lock). A
        # startup sweep (mid_run=False) has no such live-tracking fallback,
        # so a fresh plan.lock is trusted at face value there. A DB- or
        # in-memory-sourced claimant has no such false-positive mode and
        # always leaves alone regardless of mid_run.
        live_claimant = report.live_claimant
        if live_claimant is not None:
            plan_lock_mid_run_exception = (
                mid_run and live_claimant.source == ClaimantSource.PLAN_LOCK
            )
            if not plan_lock_mid_run_exception:
                return None

        # 'in-progress', not on main, no live claimant (or the mid-run R3
        # plan.lock exception above): revert to pending.  Shared with the
        # is_ancestor==True degenerate-provisioning-branch guards above so a
        # never-advanced branch (the 2992 strand) is recovered, not left to
        # sit.
        return await self._revert_in_progress_if_no_live_claimant(
            tid, mid_run=mid_run, metadata=metadata, status=status,
        )

    async def _revert_in_progress_if_no_live_claimant(
        self, tid: str, *, mid_run: bool, metadata: dict | None = None,
        status: str | None = None,
    ) -> str | None:
        """Revert a stranded in-progress task to pending (REVERT_TO_PENDING applier).

        Returns ``'reverted'`` (status flipped to pending) or ``None`` (left
        intact — the sole remaining early-return is the infra-held
        non-degenerate branch guard below; task 2763 removed the
        unreadable-lock fail-closed early-return along with the worktree
        plan.lock read that produced it).

        Callers only reach this on the REVERT_TO_PENDING action:
        ``TaskGroundTruth.recovery_for``'s ``live_claimant`` resolution has
        already established there is no live claimant (task 2243, W10-θ2
        step-16 — the plan.lock owner_pid liveness re-derivation formerly
        performed directly in this function is retired as redundant). This
        function owns only the physical side effects: worktree cleanup, a
        defensive plan.lock unlink, and the
        ``_recovered_plans``/``_preserved_worktrees`` retention bookkeeping.
        Task 2763 retired the worktree ``plan.lock`` READ that formerly
        re-classified liveness here — moot once the durable lock moved to the
        meta-root (the worktree lock never exists in production) and the DB
        claimant became primary — so the applier no longer reads it and the
        ``mid_run`` parameter, kept for caller-ABI compat, no longer gates any
        branch.

        FORENSIC NOTE (task 2588 closeout, task 2623): the 2588 un-claim (an
        in-progress task reverted to pending with claimant_run_id/
        heartbeat_at nulled) was NOT caused by fused-memory ``update_task``
        performing a full-row UPDATE — that premise is refuted;
        ``SqliteTaskBackend.update_task`` never writes ``status`` /
        ``claimant_run_id`` / ``heartbeat_at``. The actual root cause was the
        PRE-2243 plan.lock/owner_pid-only sweep: it misjudged a live task
        whose DB claimant was fresh but whose plan.lock was stale or not
        visible cross-process, and reverted/un-claimed it out from under its
        own live owner. Task 2243 (W10-θ2) fixed this whole class by making
        ``recovery_for``'s DB-claimant resolution (``_resolve_live_claimant``,
        task_ground_truth.py:368-376) the primary cross-process liveness
        signal, consulted BEFORE plan.lock — see the paragraph above. Guarded
        by two regression pins so a future refactor cannot silently reorder
        DB-claimant-vs-plan.lock precedence without tripping them:
        ``test_task_ground_truth.py::TestDeriveTruthLiveClaimant::
        test_fresh_db_claimant_precedes_stale_plan_lock`` (resolver-level)
        and ``test_reconcile_stranded.py::TestReconcileStrandedInProgress::
        test_stale_plan_lock_but_live_db_claimant_left_alone`` (sweep-level,
        end-to-end through this function's caller).

        For warm tasks the real worktree is the assigned pool lane, not
        ``worktree_base/<tid>``.  ``_resolve_task_worktree`` handles both cases.
        """
        worktree_path = self._resolve_task_worktree(tid)
        lock_path = worktree_path / '.task' / 'plan.lock'

        # A1 guard (task 2200/ω4): a verify-complete task held by a transient
        # infra failure — first-class status == 'infra-hold', via
        # is_infra_held — must NOT be re-pended by the stranded recovery
        # sweep.  The open infra_issue L1 is the non-dispatch hold (dispatch
        # is pending-only; the open L1 suppresses stranded_blocked re-file).
        # Flipping to pending would force the task to re-win its full
        # implement footprint in the scheduler's footprint-locked dispatch —
        # the root cause of the 3465 starvation.
        # Guard conditions: is_infra_held(task) AND the branch is
        # non-degenerate (has commits beyond branch_base_sha).  Degenerate
        # branches (provisioned but never implemented) are not protected because
        # there is no real work to preserve.
        # PRIMARY protection is the sweep-exclusion: infra-hold is not in
        # _RECONCILE_SWEEP_STATUSES, so in production this function is only
        # ever reached with status=='in-progress' (the production
        # _reconcile_one_stranded call sites), making this guard structurally
        # dormant there.  It is kept as defense-in-depth for any caller that
        # hands this function an infra-held row directly.
        # Migration-window caveat (review amendment, task 2200): a row still
        # carrying the legacy metadata.infra_hold flag with status !=
        # 'infra-hold' at deploy time is NOT recognised here (is_infra_held
        # keys on status only) and reverts to pending like any other stranded
        # in-progress task — see the docstring on
        # orchestrator.task_status.is_infra_held for the accepted-risk
        # rationale and the operator follow-up.
        # Prefer the metadata/status the reconcile caller already hoisted (one
        # get_task per stranded task — see harness.py:2275). Re-fetching
        # unconditionally here was the task-1883 step-14 regression that
        # double-fetched on the neither-path; the row is immutable across the
        # read-only intervening is_ancestor/marker/citation/degenerate checks.
        # Standalone callers (direct unit tests) may omit metadata/status and
        # pay one fetch.
        if metadata is None:
            try:
                _infra_task = await self.scheduler.get_task(tid)
            except Exception:
                _infra_task = None
            metadata = (_infra_task or {}).get('metadata') or {}
            if status is None:
                status = (_infra_task or {}).get('status')
        _infra_meta = metadata
        if is_infra_held({'status': status}):
            _branch = f'{self.git_ops.config.branch_prefix}{tid}'
            if not await self._branch_is_degenerate(_branch, _infra_meta):
                logger.info(
                    'Reconcile: task %s is infra-held on non-degenerate branch '
                    '— skipping pending revert; held for infra resume',
                    tid,
                )
                return None

        # recovery_for's live_claimant has already ruled out a live claimant
        # before dispatching REVERT_TO_PENDING here (task 2243, W10-θ2
        # step-16). Task 2763 additionally retired the worktree plan.lock READ
        # that formerly re-classified liveness at this point: in production the
        # durable lock lives at the meta-root, so <worktree>/.task/plan.lock
        # never exists and the old stale-lock branch was unreachable, and the
        # DB claimant is the primary liveness signal (task 2243). So revert
        # UNCONDITIONALLY — this function owns only the physical side effects:
        # worktree cleanup, a defensive plan.lock unlink, and the
        # _recovered_plans/_preserved_worktrees retention bookkeeping.
        #
        # Guard on branch WIP before reaping: a worktree whose leftover branch
        # still carries commits beyond main is a re-attach-eligible shape, not
        # disposable orphan residue.  Reaping it here would destroy the
        # still-registered worktree dir — including the gitignored
        # .task/plan.json that git cannot restore — right before the next
        # dispatch could resume it via create_worktree's registered-worktree
        # REUSE path (git_ops.py:1557, `if worktree_path.exists()`), NOT the
        # cold-path γ reattach guard.  So RETAIN the dir instead.
        # `worktree_path.exists()` is checked first so the branch-WIP probe (a
        # subprocess call) never runs when there is no dir to preserve.
        _no_lock_branch = f'{self.git_ops.config.branch_prefix}{tid}'
        _branch_has_wip = worktree_path.exists() and await self.git_ops._orphan_has_commits(
            _no_lock_branch
        )
        if (
            worktree_path.exists()
            and not _branch_has_wip
            and tid not in self._recovered_plans
            and tid not in self._preserved_worktrees
        ):
            try:
                await self.git_ops.cleanup_worktree(worktree_path, tid)
            except Exception:
                logger.warning(
                    'Reconcile: cleanup_worktree failed for task %s'
                    ' (no-lock); continuing',
                    tid, exc_info=True,
                )
        elif _branch_has_wip:
            logger.info(
                'Reconcile: task %s no-lock but branch %s carries WIP'
                ' commits — retaining worktree for resume',
                tid, _no_lock_branch,
            )
        # Defensive: clear any vestigial worktree plan.lock (the documented
        # "plan.lock unlink" physical side-effect).  In production the durable
        # lock lives at the meta-root, so this worktree path usually does not
        # exist; the suppress + missing_ok keep it a harmless no-op then.
        with contextlib.suppress(OSError):
            lock_path.unlink(missing_ok=True)
        await self.scheduler.set_task_status(tid, 'pending')
        logger.info(
            'Reconcile: reverted task %s to pending', tid
        )
        return 'reverted'

    async def _mark_in_progress_done(
        self,
        tid: str,
        sha: str | None,
        note: str,
        reason: str,
    ) -> bool:
        """Mark a stranded task done via ``scheduler.mark_done``.

        Thin wrapper around ``Scheduler.mark_done`` that owns the side-effect
        sequence — pop ``_recovered_plans``, attempt ``cleanup_worktree``
        (swallow errors), call ``mark_done(kind='found_on_main', ...)``
        because the sweep is *finding* tasks already on main, not actively
        merging them.

        Args:
            tid: The task id (string form, as it appears in get_statuses).
            sha: The on-main commit anchoring the recovery.  When ``None``
                (rev-parse race for a vanishing branch), log + skip — the
                next sweep retries; ``mark_done`` requires a real SHA.
            note: Free-text provenance note distinguishing the recovery
                path (already-on-main vs deleted-but-marker-found).
            reason: Short slug used in cleanup-failure WARNING and the
                success INFO log; also the ``gate_source`` recorded on a
                ``provenance_conflict`` escalation (task 2677).

        Returns:
            ``True`` when the task was marked done (or the ``sha is None``
            race-skip no-op occurred — nothing to mark, not a conflict).
            ``False`` when ``scheduler.mark_done`` raised
            ``StaleEvidenceRejection`` (task 2677): the write was refused by
            the found_on_main provenance-integrity gate because the evidence
            commit predates the task's ``reopen_at``. The task is NOT done —
            callers must report an honest ``'stale_conflict'`` disposition
            (never ``'marked_done'``) and must not release the warm lane.

        Raises ``SetTaskStatusRejected`` (other than ``StaleEvidenceRejection``,
        which is caught and converted to a ``False`` return + a born-at-L2
        ``provenance_conflict`` escalation) on persistent persistence-layer
        rejection — caller decides whether to count strikes / escalate.
        """
        if sha is None:
            logger.warning(
                'Reconcile: rev-parse failed for task %s (%s) — '
                'skipping; next sweep will retry',
                tid, reason,
            )
            return True
        # For warm tasks the real worktree is the assigned pool lane.
        # _resolve_task_worktree falls back to the cold worktree_base/tid
        # convention when the pool is absent or has no assignment for tid.
        worktree_path = self._resolve_task_worktree(tid)
        self._recovered_plans.pop(tid, None)
        self._recovered_sessions.pop(tid, None)
        if worktree_path.exists():
            try:
                await self.git_ops.cleanup_worktree(worktree_path, tid)
            except Exception:
                logger.warning(
                    'Reconcile: cleanup_worktree failed for task %s'
                    ' (%s); continuing',
                    tid, reason, exc_info=True,
                )
        try:
            await self.scheduler.mark_done(
                tid, kind='found_on_main', sha=sha, note=note,
            )
        except StaleEvidenceRejection as exc:
            # The reopen-freshness gate refused: this evidence commit
            # predates a later reopen_at, so completing the task now would
            # phantom-complete a stale claim (task 2677). Route to the
            # shared sink (dedupe-guarded born-at-L2) instead of the
            # generic reconcile_persistent_rejection L1 escalation — this
            # is NOT an unexpected persistence-layer contradiction, it is
            # the provenance-integrity gate doing its job.
            self._provenance_conflict_sink.record_from_rejection(
                exc, gate_source=reason,
            )
            logger.warning(
                'Reconcile: task %s done_evidence_stale — evidence %s '
                '(%s) predates reopen_at %s; filed provenance_conflict '
                'escalation instead of marking done (reason=%s)',
                tid, exc.evidence_commit, exc.evidence_committed_at,
                exc.reopen_at, reason,
            )
            return False
        # Diff 5c (T9 hardening): release warm lane after the done flip.
        # cleanup_worktree (above) frees the lane only when the in-memory
        # assignment map still has tid; opt into the on-disk plan.json
        # backstop (allow_disk_backstop=True) to cover the lost-map /
        # post-restart case where the assignment map was cleared on restart.
        # The theft guard inside the primitive refuses if the disk-resolved
        # lane has since been re-acquired by a different live task.
        # Idempotent when cleanup_worktree already freed it.
        await self.git_ops.release_lane_for_terminal_task(tid, allow_disk_backstop=True)
        logger.info(
            'Reconcile: marked task %s done (reason=%s)', tid, reason,
        )
        return True

    def _escalate_reconcile_failure(
        self,
        tid: str,
        exc: SetTaskStatusRejected,
    ) -> None:
        """Submit an L1 escalation for a persistent reconcile failure.

        Fired directly on every ``mark_done`` rejection the reconcile sweep
        observes for *tid* (task 2243, W10-θ2 step-14 — the per-tid strike
        counter and its ``MAX_RECONCILE_FAILURES`` threshold gate are
        retired): the persistence layer contradicts a branch-on-main
        observation, which is L1-worthy on its own without waiting for a
        run of consecutive sweeps to confirm it.
        """
        if not self._escalation_queue:
            return
        from escalation.models import Escalation

        esc = Escalation(
            id=self._escalation_queue.make_id(tid),
            task_id=tid,
            agent_role='harness-reconcile',
            severity='blocking',
            category='reconcile_persistent_rejection',
            summary=f'Reconcile sweep failed to mark task {tid} done'[:200],
            detail=(
                f'set_task_status(done) rejected by fused-memory for task '
                f'{tid}.\n\n'
                f'error_code: {exc.error_code}\n'
                f'raw: {exc.raw}\n\n'
                f'The branch was observed on main (or a merge marker was '
                f'found) but the persistence layer refuses the transition. '
                f'Manual investigation required — the row may carry stale '
                f'metadata.files or the done_provenance ancestor check may '
                f'be failing.'
            ),
            suggested_action='investigate_persistence_layer_rejection',
        )
        self._escalation_queue.submit(esc)

    # Synthetic task_id for scheduler-pause escalations.  Filename-safe for
    # EscalationQueue.make_id (yields esc-__scheduler__-N); never a real task,
    # so the orphan-L0 reaper (L0-only), _on_escalation (no registered event),
    # and reconcile (no such task) all leave it alone.
    _SCHEDULER_PAUSE_SENTINEL: str = '__scheduler__'

    # Synthetic task_id for the no-landings circuit-breaker INFO escalation.
    # Deduped to one open at a time (via get_by_task + agent_role filter).
    # Same class-level immutability guarantees as _SCHEDULER_PAUSE_SENTINEL.
    _NO_LANDINGS_SENTINEL: str = '__no_landings_breaker__'

    # Synthetic task_id + agent_role for the warm-base hard-down watchdog
    # (task 2061).  HOST-SCOPED — one warm-lane CoW seed base, so exactly one
    # notice and (past the remediation window) one L2, deduped via
    # get_by_task + agent_role filter.  Same class-level immutability
    # guarantees as _NO_LANDINGS_SENTINEL.
    _WARM_BASE_HARD_DOWN_SENTINEL: str = '__warm_base_hard_down__'
    _WARM_BASE_HARD_DOWN_ROLE: str = 'orchestrator-warm-base-hard-down'

    # Synthetic task_id + agent_role for the pool-storage-absent guard (task
    # 2099).  HOST-SCOPED — one worktree_base mount, so exactly one open L1
    # at a time, deduped via get_by_task + agent_role filter.  Deliberately
    # DISTINCT from _WARM_BASE_HARD_DOWN_SENTINEL: that predicate answers "is
    # the CoW seed base buildable" (can be legitimately absent while the
    # mount is up); this one answers "is the pool storage mount itself
    # present" (the `.pool-root` sentinel) — the destructive-sweep guards
    # (prune_worktrees, _run_warm_lane_gc_reclaim, acquire create-once,
    # orphan reaper, crash recovery) need the latter, stable signal. Same
    # class-level immutability guarantees as _WARM_BASE_HARD_DOWN_SENTINEL.
    _POOL_STORAGE_ABSENT_SENTINEL: str = '__pool_storage_absent__'
    _POOL_STORAGE_ABSENT_ROLE: str = 'orchestrator-pool-storage-absent'

    # Synthetic task_id + agent_role for the session-resume fallback-storm L1
    # (task γ, INV-4).  PER-BOOT — one open L1 at a time, deduped via
    # has_open_l1.  Same class-level immutability guarantees as the sentinels
    # above.  Fires only on a RUN of consecutive genuine resume fallbacks
    # (suspected clock skew / wiped transcripts / mass reseed).
    _SESSION_RESUME_STORM_SENTINEL: str = '__session_resume_storm__'
    _SESSION_RESUME_STORM_ROLE: str = 'orchestrator-harness'

    def _file_pool_storage_absent_escalation(self) -> None:
        """File an L1 escalation when pool storage (worktree_base) is absent.

        Installed on ``self.git_ops._on_pool_storage_absent`` (declare-on-
        callee / install-in-harness — mirrors ``warm_lane_reclaim_candidate_
        provider``) and also called directly from the orphan-reaper and
        crash-recovery sweeps when ``pool_storage_present()`` is False.

        Covers the Jul-3 incident class: ``worktree_base`` exists (it is a
        live mountpoint dir) but is EMPTY because the warm-lanes mount has
        not come up yet — every destructive sweep site refuses to run and
        routes here instead of silently wiping ``.git/worktrees`` admin
        entries. Deduped by ``has_open_l1`` so repeated ticks / multiple
        tripped sites in the same tick do not stack duplicate L1s: the
        operator sees exactly one open pool-storage-absent escalation at a
        time.

        Best-effort: a missing queue (bare-Harness unit tests) or any submit
        failure is swallowed so escalation filing never breaks the guarded
        call site.

        Marks ``_pool_storage_absent_maybe_pending`` True unconditionally
        (even if the queue is missing or the dedup below finds one already
        open) so the reaper's resolve scan runs again once storage recovers.
        """
        self._pool_storage_absent_maybe_pending = True
        if not self._escalation_queue:        # bare-Harness unit tests stay green
            return
        try:
            if self._escalation_queue.has_open_l1(self._POOL_STORAGE_ABSENT_SENTINEL):
                return                         # dedup: one open L1 at a time
            from escalation.models import Escalation  # noqa: PLC0415
            esc = Escalation(
                id=self._escalation_queue.make_id(self._POOL_STORAGE_ABSENT_SENTINEL),
                task_id=self._POOL_STORAGE_ABSENT_SENTINEL,
                agent_role=self._POOL_STORAGE_ABSENT_ROLE,
                severity='blocking',
                category='infra_issue',
                summary=(
                    'Pool storage absent — worktree_base exists but the '
                    '.pool-root sentinel is missing (suspected unmounted '
                    'mount); destructive worktree sweeps are suppressed'
                )[:200],
                detail=(
                    'GitOps.pool_storage_present() returned False: '
                    'worktree_base exists on disk but its `.pool-root` '
                    'sentinel is absent, which is exactly what an unmounted '
                    'warm-lanes mountpoint looks like after a host reboot / '
                    'crash where this process started before the mount unit. '
                    'To avoid repeating the Jul-3 incident (an orphan-reaper '
                    'prune wiped every registered lane + _merge-verify admin '
                    'entry against an unmounted mount), the orphan reaper, '
                    'prune_worktrees, the warm-lane GC reclaim pass, crash '
                    'recovery, and warm/spec-lane acquire create-once all '
                    'refuse to run against this worktree_base until the '
                    'sentinel reappears.\n\n'
                    'Check whether the pool storage mount is up (e.g. '
                    '`mount | grep <worktree_base>`) and remount it if not. '
                    'If this is intentionally a fresh, plain-directory '
                    'worktree_base with no mount involved, restore service '
                    'by re-running a lane acquire/seed to bootstrap the '
                    '`.pool-root` sentinel, or create it manually.'
                ),
                suggested_action=(
                    'Verify the warm-lanes mount is present (check the '
                    'RequiresMountsFor= unit dependency and its sibling '
                    'mount unit); once remounted, resolve this escalation — '
                    'the next sweep auto-clears it via '
                    '_resolve_pool_storage_absent_escalation().'
                ),
                level=1,
            )
            self._escalation_queue.submit(esc)
            logger.warning('Filed L1 pool-storage-absent escalation %s', esc.id)
        except Exception:
            logger.warning('Failed to file pool-storage-absent escalation', exc_info=True)

    def _file_session_resume_storm_escalation(self) -> None:
        """File an L1 when session-resume fallbacks storm (task γ, INV-4).

        Called from the _run_slot guard once the consecutive-per-boot
        ``_session_resume_fallback_streak`` reaches
        ``session_resume.fallback_storm_threshold``. A single isolated
        fallback (a lone foreign-acquire) never trips this — only a RUN does,
        which is the signature of SYSTEMATIC corroboration breakage (clock
        skew, wiped transcripts, mass reseed). Deduped by ``has_open_l1`` so
        the operator sees exactly one open storm L1 at a time.

        Best-effort: a missing queue (bare-Harness unit tests) or any submit
        failure is swallowed so filing never breaks the guard path (I3).
        """
        if not self._escalation_queue:        # bare-Harness unit tests stay green
            return
        try:
            if self._escalation_queue.has_open_l1(self._SESSION_RESUME_STORM_SENTINEL):
                return                         # dedup: one open L1 at a time
            from escalation.models import Escalation  # noqa: PLC0415
            threshold = self.config.session_resume.fallback_storm_threshold
            esc = Escalation(
                id=self._escalation_queue.make_id(self._SESSION_RESUME_STORM_SENTINEL),
                task_id=self._SESSION_RESUME_STORM_SENTINEL,
                agent_role=self._SESSION_RESUME_STORM_ROLE,
                severity='blocking',
                category='infra_issue',
                summary=(
                    'Session-resume fallback storm — '
                    f'{threshold}+ consecutive resume corroboration failures '
                    'this boot; resume degraded to fresh dispatch for all'
                )[:200],
                detail=(
                    f'{threshold} or more consecutive session-resume '
                    'eligibility failures occurred this boot without an '
                    'intervening successful resume. Every recovered agent '
                    'session was rejected (stale sidecar or absent transcript) '
                    'and degraded to a fresh dispatch — safe, but a RUN this '
                    'long suggests a systematic cause rather than isolated '
                    'foreign-acquires: clock skew making every sidecar look '
                    'stale, a reseed/`git clean` wiping transcripts out from '
                    'under adopted sessions, or a mass lane reseed.\n\n'
                    'Fresh dispatch loses the in-flight agent context that '
                    'resume would have preserved, so throughput/cost is '
                    'degraded until the cause is fixed. Check host clock skew '
                    '(NTP) and whether warm-lane reseeds are wiping '
                    '.task/claude-config transcripts.'
                ),
                suggested_action=(
                    'Investigate clock skew (NTP) and transcript-wiping '
                    'reseeds; the streak resets on the next successful resume, '
                    'so resolve this L1 once the underlying cause is fixed.'
                ),
                level=1,
            )
            self._escalation_queue.submit(esc)
            logger.warning('Filed L1 session-resume fallback-storm escalation %s', esc.id)
        except Exception:
            logger.warning(
                'Failed to file session-resume storm escalation', exc_info=True
            )

    async def _resolve_pool_storage_absent_escalation(self) -> None:
        """Resolve any pending pool-storage-absent L1 (task 2099).

        Called from the orphan-reaper sweep when ``pool_storage_present()``
        is True, mirroring ``_resolve_warm_base_hard_down`` — resolves ALL
        pending escalations for the sentinel+role, idempotent
        (``EscalationQueue.resolve`` is a no-op if already resolved).
        Best-effort: no-op when ``_escalation_queue`` is None or resolve
        raises.

        On successful completion (no exception) clears
        ``_pool_storage_absent_maybe_pending`` to False — this scan has just
        proven there is nothing left to resolve, so the reaper's caller can
        skip the next scan(s) until ``_file_pool_storage_absent_escalation``
        sets the flag True again. Left True on exception (fail-safe: keep
        re-scanning rather than risk silently never resolving a real L1).
        """
        if not self._escalation_queue:
            return
        try:
            pending = [
                e
                for e in self._escalation_queue.get_by_task(
                    self._POOL_STORAGE_ABSENT_SENTINEL, status='pending',
                )
                if e.agent_role == self._POOL_STORAGE_ABSENT_ROLE
            ]
            for esc in pending:
                self._escalation_queue.resolve(
                    esc.id,
                    'auto-resolved: pool storage is present again',
                    resolved_by=self._POOL_STORAGE_ABSENT_ROLE,
                )
                logger.info(
                    'Pool-storage-absent: auto-resolved escalation %s', esc.id,
                )
            self._pool_storage_absent_maybe_pending = False
        except Exception:
            logger.warning(
                'Pool-storage-absent: failed to resolve escalation(s)',
                exc_info=True,
            )

    def _file_scheduler_pause_escalation(self, reason: str) -> None:
        """File an L1 escalation when scheduler dispatch is paused.

        Covers every pause path — park-stop, cost-ceiling (1323), EWA digest
        (1327), watcher-crashloop — because all route through
        ``pause_scheduler``.  Deduped by ``has_open_l1`` so a restored pause
        re-filing (or a second sibling trip) does not stack duplicate L1s: the
        operator sees exactly one open scheduler-pause escalation at a time.

        Best-effort: a missing queue (bare-Harness unit tests) or any submit
        failure is swallowed so escalation filing never breaks the pause path.
        """
        if not self._escalation_queue:        # bare-Harness unit tests stay green
            return
        try:
            if self._escalation_queue.has_open_l1(self._SCHEDULER_PAUSE_SENTINEL):
                return                         # dedup: one open L1 at a time
            from escalation.models import Escalation
            esc = Escalation(
                id=self._escalation_queue.make_id(self._SCHEDULER_PAUSE_SENTINEL),
                task_id=self._SCHEDULER_PAUSE_SENTINEL,
                agent_role='orchestrator-scheduler',
                severity='blocking',
                category='scheduler_paused',
                summary=(f'Scheduler paused — dispatch halted: {reason}')[:200],
                detail=(
                    'The orchestrator scheduler is paused and will dispatch no '
                    'new tasks until a human resolves this.\n\n'
                    f'Reason: {reason}\n\n'
                    'The process stays alive and idles; the escalation MCP server '
                    'remains reachable. Investigate why work was parked, then '
                    'resolve this escalation / call resume_scheduler() to resume '
                    'dispatch in-process (no restart needed).'
                ),
                suggested_action='investigate_and_resume_scheduler',
                level=1,
            )
            self._escalation_queue.submit(esc)
            logger.warning('Filed L1 scheduler-pause escalation %s', esc.id)
        except Exception:
            logger.warning('Failed to file scheduler-pause escalation', exc_info=True)

    async def _block_and_escalate_external_dep(
        self,
        task_id: str,
        *,
        summary: str,
        detail: str,
        category: str,
    ) -> None:
        """Block a task and file an L1 escalation for a cross-project dep failure.

        Wired as ``on_external_dep_block`` in the ``SchedulerCallbacks`` bundle
        passed to the Scheduler constructor, alongside ``on_park_stop_trip``.

        Sets the task to ``blocked`` via ``scheduler.set_task_status`` and
        submits a level-1 ``Escalation`` to the queue.  Deduped by
        ``has_open_l1`` so repeated ticks for the same task do not stack
        duplicate escalations.  No-ops gracefully when no escalation queue is
        attached (bare-Harness unit tests stay green).
        """
        # Set task to blocked regardless of queue state — the queue is only for
        # human notification; the gate stays closed via the external-dep cache.
        # Wrapped in try/except so a transient set_task_status failure (e.g.
        # fused-memory temporarily unavailable) does not prevent the L1 escalation
        # from being filed; the human must still be notified even if the status
        # write failed so they can intervene and set it manually.
        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
        except Exception:
            logger.warning(
                'External dep block for task %s — set_task_status raised; '
                'will still attempt to file escalation',
                task_id,
                exc_info=True,
            )

        if not self._escalation_queue:
            logger.warning(
                'External dep block for task %s — no escalation queue, skipping L1 file',
                task_id,
            )
            return

        if self._escalation_queue.has_open_l1(task_id):
            # Log at WARNING: the pre-existing L1 may be for an unrelated cause
            # (e.g. an infra escalation), so the external-dep block reason could
            # be masked.  The warning makes the suppression observable.
            logger.warning(
                'External dep block for task %s — open L1 already exists, suppressing '
                'duplicate; pre-existing L1 may be for an unrelated cause',
                task_id,
            )
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self._escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-scheduler',
            severity='blocking',
            category=category,
            summary=summary[:200],
            detail=detail,
            suggested_action='manual_intervention',
            level=1,
        )
        self._escalation_queue.submit(esc)
        logger.warning(
            'Filed L1 external-dep escalation %s for task %s',
            esc.id,
            task_id,
        )

    async def _block_and_escalate_delivered_check(
        self,
        task_id: str,
        *,
        summary: str,
        detail: str,
        category: str,
    ) -> None:
        """Block a task and file a BORN-AT-L2 escalation for a delivered
        check that has FAILED for ``delivered_checks.grace_cycles``
        consecutive scheduler ticks (task 2583, epsilon).

        Wired as ``on_delivered_check_block`` in the ``SchedulerCallbacks``
        bundle passed to the Scheduler constructor, alongside
        ``on_external_dep_block``. Structurally mirrors
        ``_block_and_escalate_external_dep`` but files with
        ``severity='critical'`` and ``level=2`` explicitly — combined with
        ``agent_role='orchestrator-scheduler'`` (a harness-sentinel role,
        ``escalation/server.py`` ``_is_harness_sentinel_role``), the record
        is BORN AT L2: it bypasses the auto-watcher and routes straight to
        a human (``escalation/models.py`` ``BORN_AT_L2_SEVERITIES``).

        Sets the task to ``blocked`` via ``scheduler.set_task_status`` and
        submits the L2 ``Escalation``. Deduped via a scoped
        ``get_by_task(task_id, status='pending', level=2,
        agent_role='orchestrator-scheduler')`` read — no new EscalationQueue
        method (escalation/ is out of this task's module scope); the
        role+level scoping keeps this dedupe from masking, or being masked
        by, an unrelated open L2 for the same task (e.g. the deterministic
        runner's ``orchestrator-deterministic`` L2). No-ops gracefully when
        no escalation queue is attached (bare-Harness unit tests stay
        green).
        """
        # Set task to blocked regardless of queue state — the queue is only for
        # human notification; the gate stays closed via the delivered-check cache.
        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
        except Exception:
            logger.warning(
                'Delivered-check block for task %s — set_task_status raised; '
                'will still attempt to file escalation',
                task_id,
                exc_info=True,
            )

        if not self._escalation_queue:
            logger.warning(
                'Delivered-check block for task %s — no escalation queue, skipping L2 file',
                task_id,
            )
            return

        if self._escalation_queue.get_by_task(
            task_id, status='pending', level=2, agent_role='orchestrator-scheduler'
        ):
            logger.warning(
                'Delivered-check block for task %s — open L2 already exists, '
                'suppressing duplicate',
                task_id,
            )
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self._escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-scheduler',
            severity='critical',
            category=category,
            summary=summary[:200],
            detail=detail,
            suggested_action='manual_intervention',
            level=2,
        )
        self._escalation_queue.submit(esc)
        logger.warning(
            'Filed L2 delivered-check escalation %s for task %s',
            esc.id,
            task_id,
        )

    async def _file_starvation_info(
        self,
        task_id: str,
        *,
        summary: str,
        detail: str,
    ) -> None:
        """File a non-blocking INFO escalation for a scheduler-starved task.

        Wired as ``on_starvation_warn`` in the ``SchedulerCallbacks`` bundle
        passed to the Scheduler constructor, alongside ``on_external_dep_block``.

        Deliberately does NOT call ``set_task_status`` — this is a pure
        observation signal (PROPERTY 1: must never gate/halt the scheduler or
        the task).  One open INFO escalation per task; subsequent calls are
        deduped while a pending watchdog escalation already exists.  No-ops
        gracefully when no escalation queue is attached.

        **Routing note:** filed at ``severity='info', level=0``.  Level-0 INFO
        escalations are surfaced via the steward's pending-escalation view and
        any external monitors polling the escalation queue (e.g. dashboards,
        alerting webhooks).  They do NOT auto-promote to L1/L2 and do NOT
        block or interrupt the scheduler — they are a passive signal intended
        for an operator reviewing the queue at their next opportunity.  If
        immediate paging is required for a specific deployment, configure an
        external monitor to watch for ``agent_role='orchestrator-starvation-watchdog'``
        pending entries and promote or alert as appropriate.
        """
        if not self._escalation_queue:
            return

        existing = [
            e
            for e in self._escalation_queue.get_by_task(task_id, status='pending')
            if e.agent_role == 'orchestrator-starvation-watchdog'
        ]
        if existing:
            logger.debug(
                'Starvation watchdog: open INFO escalation already exists for '
                'task %s — skipping duplicate file',
                task_id,
            )
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self._escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-starvation-watchdog',
            severity='info',
            category='risk_identified',
            summary=summary[:200],
            detail=detail,
            suggested_action='manual_investigation',
            level=0,
        )
        self._escalation_queue.submit(esc)
        logger.info(
            'Starvation watchdog: filed INFO escalation %s for task %s',
            esc.id,
            task_id,
        )

    async def _resolve_starvation_info(self, task_id: str) -> None:
        """Resolve an open starvation-watchdog INFO escalation for a task.

        Wired as ``on_starvation_resolve`` in the ``SchedulerCallbacks`` bundle.
        Called at both dispatch sites and from the GC backstop when the task
        is terminal.

        Uses ``EscalationQueue.resolve`` which is idempotent (no-op if already
        resolved), so double-resolve from the dispatch site and the GC backstop
        is safe.  No-ops gracefully when no escalation queue is attached.
        """
        if not self._escalation_queue:
            return

        pending = [
            e
            for e in self._escalation_queue.get_by_task(task_id, status='pending')
            if e.agent_role == 'orchestrator-starvation-watchdog'
        ]
        for esc in pending:
            self._escalation_queue.resolve(
                esc.id,
                'auto-resolved: task dispatched (starvation watchdog)',
                resolved_by='orchestrator-starvation-watchdog',
            )
            logger.info(
                'Starvation watchdog: auto-resolved INFO escalation %s for task %s',
                esc.id,
                task_id,
            )

    async def _probe_warm_base_health(self) -> str:
        """Bridge GitOps._warm_lane_base_resolvable to the scheduler's
        injected-callback string contract (task 2061).

        Wired as ``warm_base_health_probe`` in the ``SchedulerCallbacks``
        bundle.  Returns ``'ok'`` when the warm-lane pool is disabled
        (``git_ops.warm_lane_pool is None``) — a disabled pool is never
        "hard-down".  Otherwise
        delegates to the synchronous ``GitOps._warm_lane_base_resolvable()``
        (pure filesystem check, no await inside) and returns its
        ``WarmBaseHealth`` member's ``.value``.
        """
        if self.git_ops.warm_lane_pool is None:
            return 'ok'
        return self.git_ops._warm_lane_base_resolvable().value

    async def _file_warm_base_hard_down_notice(
        self, *, summary: str, detail: str,
    ) -> None:
        """File a non-blocking INFO escalation for the warm-base hard-down
        watchdog (task 2061).

        Wired as ``on_warm_base_warn`` in the ``SchedulerCallbacks`` bundle.
        Mirrors ``_file_no_landings_info_escalation`` (global sentinel task_id, one
        open INFO at a time, best-effort).

        Deliberately does NOT call ``set_task_status`` or halt anything
        itself — the scheduler's own ``_warm_base_hard_down`` latch already
        halts dispatch; this is a pure observation signal.  Deduped: if an
        open notice already exists (pending, level=0, this role), this is a
        no-op.  Best-effort: any submit failure is swallowed and logged.

        **Routing note:** ``severity='info', level=0``.  Level-0 INFO is
        surfaced via the steward's pending-escalation view and external
        monitors/dashboards/webhooks; it does NOT auto-promote and does NOT
        additionally halt the scheduler (the watchdog's own latch already did).
        """
        if not self._escalation_queue:
            return
        try:
            existing = [
                e
                for e in self._escalation_queue.get_by_task(
                    self._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
                )
                if e.agent_role == self._WARM_BASE_HARD_DOWN_ROLE and e.level == 0
            ]
            if existing:
                return  # dedup: one open notice at a time
            from escalation.models import Escalation  # noqa: PLC0415
            esc = Escalation(
                id=self._escalation_queue.make_id(self._WARM_BASE_HARD_DOWN_SENTINEL),
                task_id=self._WARM_BASE_HARD_DOWN_SENTINEL,
                agent_role=self._WARM_BASE_HARD_DOWN_ROLE,
                severity='info',
                category='infra_issue',
                summary=summary[:200],
                detail=detail,
                suggested_action='Run reify/scripts/ensure-warm-base.sh',
                level=0,
            )
            self._escalation_queue.submit(esc)
            logger.warning(
                'Warm-base hard-down: filed INFO escalation %s', esc.id,
            )
        except Exception:
            logger.warning(
                'Warm-base hard-down: failed to file INFO escalation',
                exc_info=True,
            )

    async def _promote_warm_base_hard_down_l2(
        self, *, summary: str, detail: str,
    ) -> None:
        """Promote the ONE born-at-L2 escalation for a warm base stuck ABSENT
        past the configured remediation window (task 2061).

        Wired as ``on_warm_base_promote_l2`` in the ``SchedulerCallbacks``
        bundle.  Filed ``severity='critical'`` (∈ ``BORN_AT_L2_SEVERITIES``) and ``level=2``
        so it routes straight to a human/L2-watcher, bypassing the auto-
        watcher — the reify reseed ladder is presumed stuck (a healthy ladder
        would have cleared the notice via ``_resolve_warm_base_hard_down``
        before the window elapsed).  Deduped: if an open L2 already exists
        (pending, level=2, this role), this is a no-op.  Best-effort: any
        submit failure is swallowed and logged.
        """
        if not self._escalation_queue:
            return
        try:
            existing = [
                e
                for e in self._escalation_queue.get_by_task(
                    self._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
                )
                if e.agent_role == self._WARM_BASE_HARD_DOWN_ROLE and e.level == 2
            ]
            if existing:
                return  # dedup: one open L2 at a time
            from escalation.models import Escalation  # noqa: PLC0415
            esc = Escalation(
                id=self._escalation_queue.make_id(self._WARM_BASE_HARD_DOWN_SENTINEL),
                task_id=self._WARM_BASE_HARD_DOWN_SENTINEL,
                agent_role=self._WARM_BASE_HARD_DOWN_ROLE,
                severity='critical',
                category='infra_issue',
                summary=summary[:200],
                detail=detail,
                suggested_action='Run reify/scripts/ensure-warm-base.sh',
                level=2,
            )
            self._escalation_queue.submit(esc)
            logger.warning(
                'Warm-base hard-down: promoted L2 escalation %s (stuck past window)',
                esc.id,
            )
        except Exception:
            logger.warning(
                'Warm-base hard-down: failed to promote L2 escalation',
                exc_info=True,
            )

    async def _resolve_warm_base_hard_down(self) -> None:
        """Resolve both the notice and any promoted L2 for the warm-base
        hard-down watchdog (task 2061).

        Wired as ``on_warm_base_resolve`` in the ``SchedulerCallbacks`` bundle,
        called when the probe reports the base healthy again.  Mirrors
        ``_resolve_no_landings_info_escalation`` — resolves ALL pending
        escalations for the sentinel+role (both the level-0 notice and any
        level-2 promotion), idempotent (``EscalationQueue.resolve`` is a
        no-op if already resolved).  Best-effort: no-op when
        ``_escalation_queue`` is None or resolve raises.
        """
        if not self._escalation_queue:
            return
        try:
            pending = [
                e
                for e in self._escalation_queue.get_by_task(
                    self._WARM_BASE_HARD_DOWN_SENTINEL, status='pending'
                )
                if e.agent_role == self._WARM_BASE_HARD_DOWN_ROLE
            ]
            for esc in pending:
                self._escalation_queue.resolve(
                    esc.id,
                    'auto-resolved: warm-lane base is healthy again',
                    resolved_by=self._WARM_BASE_HARD_DOWN_ROLE,
                )
                logger.info(
                    'Warm-base hard-down: auto-resolved escalation %s', esc.id,
                )
        except Exception:
            logger.warning(
                'Warm-base hard-down: failed to resolve escalation(s)',
                exc_info=True,
            )

    # ── No-landings circuit-breaker helpers (θ=1893, PRD §5.5) ──────────────

    def _file_no_landings_info_escalation(self, trip: BreakerTrip) -> None:
        """File a non-blocking INFO escalation for the no-landings circuit-breaker.

        Mirrors ``_file_starvation_info`` (pure operator-signal INFO, level=0)
        and uses the ``_NO_LANDINGS_SENTINEL`` sentinel task_id (mirroring
        ``_SCHEDULER_PAUSE_SENTINEL``) so the escalation is global / not tied
        to a specific task.

        Deduped: if an open breaker INFO escalation already exists (pending,
        agent_role=``orchestrator-no-landings-breaker``), this is a no-op.
        Best-effort: any submit failure is swallowed and logged.

        **Routing note:** ``severity='info', level=0``.  Level-0 INFO is
        surfaced via the steward's pending-escalation view and external monitors/
        dashboards/webhooks.  Does NOT auto-promote and does NOT additionally
        halt the scheduler (``force_halt_scheduler`` already handles the halt;
        ``file_escalation=False`` was passed there to avoid the auto-watcher
        resolving the scheduler-pause L1 and silently undoing the halt).
        """
        if not self._escalation_queue:
            return
        try:
            existing = [
                e
                for e in self._escalation_queue.get_by_task(
                    self._NO_LANDINGS_SENTINEL, status='pending'
                )
                if e.agent_role == 'orchestrator-no-landings-breaker'
            ]
            if existing:
                return  # dedup: one open INFO at a time
            from escalation.models import Escalation  # noqa: PLC0415
            esc = Escalation(
                id=self._escalation_queue.make_id(self._NO_LANDINGS_SENTINEL),
                task_id=self._NO_LANDINGS_SENTINEL,
                agent_role='orchestrator-no-landings-breaker',
                severity='info',
                category='risk_identified',
                summary=(
                    f'No-landings circuit-breaker tripped: '
                    f'{trip.window_samples} samples, 0 landings, '
                    f'disk {trip.free_start:,}→{trip.free_end:,} bytes'
                )[:200],
                detail=(
                    'The merge-queue no-landings circuit-breaker detected a '
                    'potential churn spiral and has halted scheduler dispatch.\n\n'
                    f'Window: {trip.window_samples} consecutive samples\n'
                    f'Landings in window: {trip.landings_in_window} (expected 0 to trip)\n'
                    f'Disk free-bytes: {trip.free_start:,} → {trip.free_end:,} '
                    f'(strictly falling)\n\n'
                    f'Trip reason: {trip.reason}\n\n'
                    'Auto-resume will occur on the next clean landing OR once '
                    'warm-lane disk free-bytes recover above the configured absolute '
                    f'floor ({self._no_landings_breaker.disk_free_floor_bytes:,} bytes '
                    f'= {self._no_landings_breaker.disk_free_floor_bytes // (1024**3)} GiB).  '
                    'Anti-flap: after auto-resume a fresh full window of flat+falling '
                    'samples is required to re-trip.\n\n'
                    'force_resume_scheduler() remains the manual override if neither '
                    'condition fires (e.g. disk is recovering slowly but has not yet '
                    'crossed the floor, and no new landings are occurring).'
                ),
                suggested_action='manual_investigation',
                level=0,
            )
            self._escalation_queue.submit(esc)
            logger.info(
                'No-landings breaker: filed INFO escalation %s '
                '(window=%d, disk %d→%d)',
                esc.id, trip.window_samples, trip.free_start, trip.free_end,
            )
        except Exception:
            logger.warning(
                'No-landings breaker: failed to file INFO escalation',
                exc_info=True,
            )

    def _resolve_no_landings_info_escalation(
        self, trip: BreakerTrip | None = None
    ) -> None:
        """Resolve an open no-landings breaker INFO escalation.

        Called on resume (with a ``BreakerTrip`` record) or on external-resume
        reconciliation (``trip=None``).  Idempotent — ``EscalationQueue.resolve``
        is a no-op if already resolved.  Best-effort: no-op when
        ``_escalation_queue`` is None or resolve raises.  Mirrors
        ``_resolve_starvation_info``.
        """
        if not self._escalation_queue:
            return
        resolution_reason = (
            f'auto-resolved: {trip.reason}'
            if trip is not None
            else 'auto-resolved: breaker re-armed (scheduler was externally resumed)'
        )
        try:
            pending = [
                e
                for e in self._escalation_queue.get_by_task(
                    self._NO_LANDINGS_SENTINEL, status='pending'
                )
                if e.agent_role == 'orchestrator-no-landings-breaker'
            ]
            for esc in pending:
                self._escalation_queue.resolve(
                    esc.id,
                    resolution_reason,
                    resolved_by='orchestrator-no-landings-breaker',
                )
                logger.info(
                    'No-landings breaker: auto-resolved INFO escalation %s', esc.id
                )
        except Exception:
            logger.warning(
                'No-landings breaker: failed to resolve INFO escalation',
                exc_info=True,
            )

    async def _run_no_landings_breaker_pass(self) -> None:
        """Single testable pass of the no-landings circuit-breaker.

        Reads ``landings_total`` from the merge-worker snapshot, samples disk
        free-bytes from ``git_ops.worktree_base`` (fail-open on OSError), feeds
        the breaker, and reacts to any ``BreakerTrip`` decision:

        * ``action='halt'``:  ``force_halt_scheduler(reason)`` +
          ``_file_no_landings_info_escalation(trip)``
        * ``action='resume'``: ``force_resume_scheduler(reason)`` +
          ``_resolve_no_landings_info_escalation(trip)``

        No-op when ``_merge_worker`` is None (bare harness / unit tests).
        Fail-open on ``OSError`` from ``shutil.disk_usage`` so a transient
        stat failure (or not-yet-created ``worktree_base``) never halts dispatch.
        Mirrors ``_run_main_tip_sweep`` as the single-pass testable entry point.

        **Reconciliation (suggestion 2):** if the breaker thinks it is tripped
        but the scheduler is no longer paused (an operator or another path
        resumed it externally), the breaker is reset so it re-arms trip
        detection rather than silently stopping guard.

        **Idle-queue guard (suggestion 3):** when the merge queue is confirmed
        empty (``snapshot['depth'] == 0``) and the breaker is not already
        tripped, trip evaluation is skipped — a quiescent queue cannot produce a
        churn spiral, so any disk fall in that state is from an unrelated cause.
        The key defaults to 1 (active) when absent, preserving backward
        compatibility with test mocks that omit ``'depth'``.
        """
        import shutil as _shutil  # noqa: PLC0415

        if self._merge_worker is None:
            return

        worker = self._merge_worker
        snapshot = worker.snapshot()
        metrics = snapshot.get('metrics')
        if metrics is None:
            return

        landings_total: int = metrics['landings_total']

        # Reconcile breaker state with actual scheduler state.  If something
        # outside the breaker (operator call, another auto-resume path) un-paused
        # the scheduler while the breaker still thinks it is tripped, the breaker
        # would silently stop evaluating trip conditions.  Detect and reset it so
        # guard re-arms for the next full window.
        if self._no_landings_breaker.is_tripped and not self.scheduler.is_paused:
            logger.info(
                'No-landings breaker: scheduler was externally resumed — re-arming'
            )
            self._no_landings_breaker.reset()
            self._resolve_no_landings_info_escalation()  # best-effort cleanup
            return

        # Gate trip evaluation on evidence of active dispatch.  An empty queue
        # (depth == 0) cannot be a churn spiral — any concurrent disk fall is
        # from an unrelated source.  Defaults to 1 (active) so test mocks that
        # omit 'depth' are treated as active (backward-compatible).
        #
        # Buffer clear: also reset() the breaker so stale flat+falling samples
        # accumulated before the idle gap are dropped.  Without this, an
        # oscillating queue (active → idle → active) can fill a trip window
        # across the idle gap and spuriously trip on a queue that was never
        # continuously active.  A fresh full window of flat+falling samples is
        # required to re-trip (anti-flap is preserved by the buffer-clear-on-
        # resume inside _check_resume; this reset() only clears pre-idle
        # samples).
        #
        # Sensitivity tradeoff: a bursty spiral whose queue drains to depth==0
        # between churn bursts (even briefly, within each 30-min window) will
        # never accumulate a full window of flat+falling samples and will never
        # trip.  This is intentional — brief queue drains are an anti-spiral
        # signal — but means the breaker is tuned for *continuous* spirals, not
        # bursty ones.  If bursty-drain spirals become a concern, gate the reset
        # on a sustained-idle counter (N consecutive depth==0 passes) rather
        # than a single observation.
        queue_depth: int = snapshot.get('depth', 1)
        if queue_depth == 0 and not self._no_landings_breaker.is_tripped:
            self._no_landings_breaker.reset()
            return

        try:
            _du = _shutil.disk_usage(self.git_ops.worktree_base)
            free_bytes: int = _du.free
            # One-shot sanity check: if the configured floor exceeds the volume's
            # total capacity, disk-recovery auto-resume (free_bytes >= floor) is
            # structurally unreachable and only a clean landing can clear the
            # breaker.  Guards with isinstance so test mocks that set only 'free'
            # (without 'total') do not trigger spurious log noise — in tests
            # MagicMock().total is a MagicMock, not an int.
            _total = _du.total
            if (
                not self._no_landings_floor_capacity_warned
                and isinstance(_total, int)
                and 0 < _total < self._no_landings_breaker.disk_free_floor_bytes
            ):
                logger.warning(
                    'No-landings breaker: configured disk_free_floor_bytes (%d GiB) '
                    'exceeds volume total capacity (%d GiB) — disk-recovery auto-resume '
                    'is unreachable on this host; only a clean landing can clear the '
                    'breaker.  Consider reducing no_landings_breaker_disk_free_floor_bytes '
                    'to a value below the volume total capacity.',
                    self._no_landings_breaker.disk_free_floor_bytes // (1024 ** 3),
                    _total // (1024 ** 3),
                )
                self._no_landings_floor_capacity_warned = True
        except OSError:
            # Fail-open: stat failure (e.g. worktree_base not yet created on a
            # fresh host) must never halt dispatch.  Mirror
            # _ensure_verify_disk_space (merge_queue.py:558-606).
            logger.debug(
                'No-landings breaker: disk_usage(%s) raised OSError — fail-open skip',
                self.git_ops.worktree_base,
                exc_info=True,
            )
            return

        trip = self._no_landings_breaker.observe(landings_total, free_bytes)
        if trip is None:
            return

        if trip.action == 'halt':
            await self.force_halt_scheduler(reason=trip.reason)
            self._file_no_landings_info_escalation(trip)
        elif trip.action == 'resume':
            await self.force_resume_scheduler(reason=trip.reason)
            self._resolve_no_landings_info_escalation(trip)

    # ------------------------------------------------------------------
    # Warm-lane auto-GC cadence loop — task 1926
    # ------------------------------------------------------------------

    async def _run_warm_lane_gc_pass(self) -> None:
        """Invoke the warm-lane GC reclaim helper once (single cadence tick).

        Delegates unconditionally to ``git_ops._run_warm_lane_gc_reclaim()``,
        which is already fail-soft: rc 127 when the script is absent (no-op),
        non-zero on script error (logged at WARNING inside the helper), and
        never raises.  This method therefore also never raises.

        Logging is tiered to avoid noise on hosts where gc.sh is absent
        (rc=127 is the expected no-op case on ~144 ticks/day with the 600s
        default):
          * rc==0   → INFO  (actual reclaim ran)
          * rc==127 → DEBUG (script absent — expected no-op)
          * other   → WARNING (unexpected non-zero)

        Also folds in the interactive-worktree (``_iact-*``) crash-safety
        reaper (task δ/2012) — no separate cadence loop or config knob is
        added for it; it rides this existing tick.  That delegate is itself
        fail-soft, so it cannot break this pass's never-raise contract.

        Registered as a ``BackgroundService`` pass_fn (task 2241, W10-η) —
        called on every interval tick.
        """
        rc = await self.git_ops._run_warm_lane_gc_reclaim()
        if rc == 0:
            logger.info('Warm-lane GC reclaim pass: reclaimed (rc=0)')
        elif rc == 127:
            logger.debug('Warm-lane GC reclaim pass: script absent, no-op (rc=127)')
        else:
            logger.warning('Warm-lane GC reclaim pass: non-zero rc=%d', rc)
        await self._run_interactive_worktree_reaper_pass()
        # Leaf γ (task 2891): the durable-record terminal-lane reclaim rides
        # this same fail-soft cadence tick (no new timer/loop). Belt-and-
        # suspenders try/except — mirroring the interactive-worktree reaper
        # delegate's bounded-log rationale (logger.error, NOT logger.exception)
        # — so a reclaim fault can never break the shared warm-lane GC cadence
        # loop or the never-raise contract of this pass.
        try:
            await self._reclaim_terminal_lane_records()
        except Exception as exc:
            logger.error(
                'Terminal-lane-record reclaim pass failed: %s: %s',
                type(exc).__name__,
                exc,
            )

    # ------------------------------------------------------------------
    # Interactive-worktree (_iact-*) crash-safety reaper — task δ/2012
    # ------------------------------------------------------------------

    async def _run_interactive_worktree_reaper_pass(self) -> None:
        """Sweep the ``_iact-*`` interactive-worktree band once (single tick).

        Delegates unconditionally to ``git_ops.reap_interactive_worktrees()``,
        which is itself fail-soft and never raises. This method wraps the
        call in a belt-and-suspenders try/except anyway (mirrors
        ``_run_warm_lane_gc_pass``'s contract) so a fault here can never kill
        the shared warm-lane GC cadence loop or the startup sequence that
        calls this directly.

        Logs one INFO line per reaped record naming slug/branch/reason — the
        per-worktree half of the I2 user-observable signal — plus a summary
        INFO line with the total count (mirroring
        ``prune_stale_merge_worktrees``'s "removed N" summary) when at least
        one worktree was reaped. When none were reaped, logs at DEBUG
        instead, to avoid noise on the ~144 ticks/day cadence. Failure
        logging is a bounded one-line ``logger.error`` summary — NOT
        ``logger.exception`` — matching ``BackgroundService``'s bounded-log
        rationale (unbounded traceback formatting can exceed per-test timeouts).

        Called by ``_run_warm_lane_gc_pass()`` on every cadence tick, and
        once unconditionally at ``run()`` startup for crash recovery.
        """
        try:
            reaped = await self.git_ops.reap_interactive_worktrees()
            for record in reaped:
                logger.info(
                    'Reaped interactive worktree %s (branch=%s, reason=%s)',
                    record.slug, record.branch, record.reason,
                )
            if reaped:
                logger.info(
                    'Interactive-worktree reaper: reaped %d worktree(s): %s',
                    len(reaped),
                    ', '.join(record.slug for record in reaped),
                )
            else:
                logger.debug(
                    'Interactive-worktree reaper: no worktrees reaped',
                )
        except Exception as exc:
            logger.error(
                'Interactive-worktree reaper pass failed: %s: %s',
                type(exc).__name__,
                exc,
            )

    # ------------------------------------------------------------------
    # Leftover verify-scope (df-verify-{tag}-*.scope) crash-safety reaper —
    # task 2829 (companion to reify enabling verify_use_cgroup_scope)
    # ------------------------------------------------------------------

    async def _run_leftover_verify_scope_reaper_pass(self) -> None:
        """Sweep THIS project's leftover ``df-verify-{tag}-*.scope`` verify
        scopes once at startup (crash recovery).

        Companion to reify flipping ``verify_use_cgroup_scope: true``: a
        crash/SIGKILL of a prior incarnation can strand a transient verify
        scope whose cgroup subtree (bash → cargo → rustc) keeps running.
        Delegates unconditionally to the fail-soft
        ``verify.reap_leftover_verify_scopes(project_root)``, which enumerates
        ONLY this project's TAG-SCOPED scopes (cross-project safety in the
        shared per-user ``systemctl --user`` session) and reaps each via
        ``_kill_cgroup_scope``.

        Logs one INFO line naming each reaped unit plus a summary INFO line
        with the count when at least one was reaped; logs at DEBUG when none,
        to avoid boot-log noise. Wrapped in a belt-and-suspenders try/except
        logging a bounded ``logger.error`` (NOT ``logger.exception``, matching
        the interactive reaper's bounded-log rationale) so a systemd/reaper
        fault can never break the startup sequence that calls this directly.

        Called once unconditionally at ``run()`` startup, adjacent to the
        interactive-worktree reaper, before the first dispatch.
        """
        from orchestrator import verify as verify_mod  # noqa: PLC0415

        try:
            reaped = await verify_mod.reap_leftover_verify_scopes(
                self.config.project_root,
            )
            for unit in reaped:
                logger.info('Reaped leftover verify scope: %s', unit)
            if reaped:
                logger.info(
                    'Verify-scope reaper: reaped %d leftover df-verify '
                    'scope(s): %s',
                    len(reaped),
                    ', '.join(reaped),
                )
            else:
                logger.debug('Verify-scope reaper: no leftover scopes reaped')
        except Exception as exc:
            logger.error(
                'Verify-scope reaper pass failed: %s: %s',
                type(exc).__name__,
                exc,
            )

    async def _block_and_escalate_substrate_flip(
        self,
        task_id: str,
        *,
        verdict,
    ) -> None:
        """Block a task and file an L1 escalation for a PASS→FAIL substrate flip.

        Mirrors ``_block_and_escalate_external_dep``:
        - Sets the task to ``blocked`` via ``scheduler.set_task_status`` (unconditional;
          wrapped in try/except so a transient write failure does not prevent the L1
          from being filed).
        - Submits a level-1 ``Escalation`` with category='design_concern' to the queue.
        - Deduped by ``has_open_l1`` so repeated dispatch attempts (after the requeue
          cooldown expires) do not stack duplicate escalations.
        - No-ops gracefully when ``_escalation_queue`` is None (bare-Harness unit tests).

        A PASS→FAIL flip means the task's probe-set premise no longer holds on
        current main; the design_concern class routes it to human/curator judgment
        (same class as report_false_premise).
        """
        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
        except Exception:
            logger.warning(
                'Substrate flip block for task %s — set_task_status raised; '
                'will still attempt to file escalation',
                task_id,
                exc_info=True,
            )

        if not self._escalation_queue:
            logger.warning(
                'Substrate flip block for task %s — no escalation queue, skipping L1 file',
                task_id,
            )
            return

        if self._escalation_queue.has_open_l1(task_id):
            logger.warning(
                'Substrate flip block for task %s — open L1 already exists, suppressing '
                'duplicate; pre-existing L1 may be for an unrelated cause',
                task_id,
            )
            return

        from escalation.models import Escalation  # noqa: PLC0415

        summary = f'SUBSTRATE_FLIP: probe set PASS→FAIL on current main (task {task_id})'
        detail = (
            f'Dispatch-time substrate re-check detected a PASS→FAIL flip.\n'
            f'Verdict: {verdict.verdict!r} | exit_code={verdict.exit_code} | '
            f'probe_set={verdict.probe_set!r}\n'
            f'Reason: {verdict.reason}\n'
            f'The task premise (probe set authored at investigation time) no longer holds '
            f'on current main — re-spec required before dispatch.'
        )
        esc = Escalation(
            id=self._escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-scheduler',
            severity='blocking',
            category='design_concern',
            summary=summary[:200],
            detail=detail,
            suggested_action='manual_intervention',
            level=1,
        )
        self._escalation_queue.submit(esc)
        logger.warning(
            'Filed L1 substrate-flip escalation %s for task %s',
            esc.id,
            task_id,
        )

    async def _run_substrate_gate(self, assignment) -> bool:
        """Run the dispatch-time substrate re-check gate (D4).

        Builds an ephemeral detached worktree at the current-main SHA (mirroring
        evals/snapshots.py), runs ``substrate_gate.run_substrate_recheck``, and
        either allows dispatch (True) or blocks + escalates (False).

        Called from ``_run_slot`` BEFORE ``TaskWorkflow`` construction so the
        agent is never spun up for a task whose probe-set premise has flipped.

        Returns:
            True   — PASS or SKIP (dispatch may proceed).
            False  — FLIP (task blocked + L1 filed; caller must arm requeue cooldown
                     and skip workflow construction).
        """
        import asyncio  # noqa: PLC0415

        from orchestrator import substrate_gate  # noqa: PLC0415

        task_id = assignment.task_id
        gate_path = self.git_ops.worktree_base / f'_substrate-gate-{task_id}'

        # Resolve current-main SHA for a deterministic, drift-isolated gate run.
        # Use self.config.git.main_branch (not the literal 'main') so any configured
        # non-default branch (e.g. 'master', 'trunk') works correctly.
        main_sha = await self.git_ops.resolve_branch_sha(self.config.git.main_branch)
        if main_sha is None:
            # Fallback: use the configured branch name as a symbolic ref.  If that
            # also fails the worktree add will error and map to FLIP below.
            main_sha = self.config.git.main_branch
            logger.warning(
                'substrate_gate: resolve_branch_sha returned None for task %s; '
                'falling back to %r symbolic ref',
                task_id, self.config.git.main_branch,
            )

        # Best-effort cleanup of any stale gate worktree left by a prior interrupted
        # run.  Without this, 'worktree add' fails with 'already exists' and maps to
        # a spurious FLIP + L1, requiring manual intervention.
        # The path-scoped `remove` is gated by the foreign-band guard (defense in
        # depth against this cleanup ever targeting a protected band it does not
        # own — gitops-chokepoints PRD, Mechanism 3); `prune` is registration-global,
        # not band-scoped, so it always runs — routed through the guarded
        # GitOps.prune_worktrees chokepoint (gitops-chokepoints PRD task β) rather
        # than a raw `git worktree prune` argv.
        if not self.git_ops.refuse_foreign_band(
            gate_path, frozenset({'_substrate-gate-'}), 'substrate-gate-cleanup',
        ):
            try:
                _cleanup_proc = await asyncio.create_subprocess_exec(
                    'git', 'worktree', 'remove', '--force', str(gate_path),
                    cwd=str(self.git_ops.project_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await _cleanup_proc.communicate()
            except Exception:
                pass  # best-effort; path may simply not exist
        # No local try/except needed: prune_worktrees never raises (it wraps its
        # own `git worktree prune` subprocess in try/except internally — see
        # GitOps._prune_registrations). This cleanup's best-effort contract
        # depends on that invariant; preserve it if that method is refactored.
        await self.git_ops.prune_worktrees(context='substrate-gate-cleanup')

        # Build ephemeral detached worktree — mirrors evals/snapshots.py pattern.
        proc = await asyncio.create_subprocess_exec(
            'git', 'worktree', 'add', '--detach', str(gate_path), main_sha,
            cwd=str(self.git_ops.project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _add_stdout, _add_stderr = await proc.communicate()
        if proc.returncode != 0:
            stderr_snippet = _add_stderr.decode('utf-8', errors='replace').strip()[:200]
            reason = (
                f'substrate unverifiable / git worktree add failed (rc={proc.returncode})'
            )
            if stderr_snippet:
                reason = f'{reason}: {stderr_snippet}'
            logger.warning(
                'substrate_gate: git worktree add failed for task %s (rc=%s) stderr=%r; '
                'treating as FLIP (substrate unverifiable)',
                task_id, proc.returncode, stderr_snippet,
            )
            await self._block_and_escalate_substrate_flip(
                task_id,
                verdict=substrate_gate.SubstrateVerdict(
                    verdict=substrate_gate.FLIP,
                    exit_code=proc.returncode,
                    checker_argv=None,
                    probe_set=None,
                    reason=reason,
                ),
            )
            return False

        try:
            verdict = await asyncio.to_thread(
                substrate_gate.run_substrate_recheck,
                task=assignment.task,
                worktree=gate_path,
            )
            logger.info(
                'substrate_gate: task %s verdict=%s reason=%r',
                task_id, verdict.verdict, verdict.reason,
            )
        except Exception as exc:
            logger.warning(
                'substrate_gate: run_substrate_recheck raised for task %s: %s',
                task_id, exc, exc_info=True,
            )
            verdict = substrate_gate.SubstrateVerdict(
                verdict=substrate_gate.FLIP,
                exit_code=None,
                checker_argv=None,
                probe_set=None,
                reason=f'substrate unverifiable / run_substrate_recheck raised: {exc}',
            )
        finally:
            # Always remove the gate worktree — leak is rare and reclaimable, but
            # we make a best effort to clean up immediately (mirrors snapshots.py).
            try:
                rm_proc = await asyncio.create_subprocess_exec(
                    'git', 'worktree', 'remove', '--force', str(gate_path),
                    cwd=str(self.git_ops.project_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await rm_proc.communicate()
            except Exception as rm_exc:
                logger.warning(
                    'substrate_gate: failed to remove gate worktree %s for task %s: %s',
                    gate_path, task_id, rm_exc,
                )

        if verdict.flipped:
            await self._block_and_escalate_substrate_flip(task_id, verdict=verdict)
            return False

        return True

    # Synthetic task_id sentinel and root_cause key for watcher-outage L2
    # escalations.  One canonical root_cause keys all watcher-outage L2s
    # regardless of trip reason (crashloop / misconfigured / cost-ceiling /
    # disabled) — the dedup contract from plans/escalation-l2-tiering.md.
    _WATCHER_OUTAGE_SENTINEL: str = '__watcher_supervisor__'
    _WATCHER_OUTAGE_ROOT_CAUSE: str = 'watcher_supervisor_down'

    def _file_watcher_outage_l2(self, reason: str) -> None:
        """File an L2 outage escalation when the watcher subsystem is not running.

        Mirrors _file_scheduler_pause_escalation (L1 helper) but targets the
        human L2 stream because a watcher outage means no-one is handling L1s.

        Dedup: find_pending_l2_by_root_cause(root_cause) returns early when an
        open L2 with the same root_cause already exists, so multiple trips
        (e.g. crashloop + cost-ceiling simultaneously) file exactly one L2.

        Best-effort: a missing queue (bare-Harness tests) or any filing failure
        is swallowed so this never breaks the supervisor / cost-ceiling /
        startup paths.
        """
        queue = getattr(self, '_escalation_queue', None)
        if not queue:        # bare-Harness unit tests / lifecycle tests stay green
            return
        try:
            if queue.find_pending_l2_by_root_cause(
                self._WATCHER_OUTAGE_ROOT_CAUSE
            ) is not None:
                return                         # dedup: one open L2 at a time
            from escalation.models import Escalation
            # NOTE: two get_pending() traversals occur here (one inside
            # find_pending_l2_by_root_cause above, one for n_l1 below).
            # This is acceptable at typical escalation-queue depths; if the
            # queue grows large, consider caching the pending list within a
            # single call or offloading to asyncio.to_thread.
            n_l1 = len([
                e for e in queue.get_pending() if e.level == 1
            ])
            esc = Escalation(
                id=queue.make_id(self._WATCHER_OUTAGE_SENTINEL),
                task_id=self._WATCHER_OUTAGE_SENTINEL,
                agent_role='orchestrator-watcher-supervisor',
                severity='urgent',
                category='infra_issue',
                level=2,
                root_cause=self._WATCHER_OUTAGE_ROOT_CAUSE,
                summary=(
                    f'escalation-watcher-auto down ({reason}); {n_l1} L1 pending'
                )[:200],
                detail=(
                    'The escalation-watcher-auto supervisor is not running or has '
                    'been paused.  No autonomous L1 triage is taking place.\n\n'
                    f'Reason: {reason}\n'
                    f'Pending L1 escalations: {n_l1}\n\n'
                    'Investigate the reason, fix the underlying issue, and restart '
                    'the orchestrator (or re-enable the watcher supervisor) to resume '
                    'autonomous triage.'
                ),
                suggested_action='investigate_watcher_supervisor',
            )
            queue.submit(esc)
            logger.warning(
                'Filed L2 watcher-outage escalation %s (reason=%s, n_l1=%d)',
                esc.id, reason, n_l1,
            )
        except Exception:
            logger.warning('Failed to file watcher-outage L2 escalation', exc_info=True)

    def _resolve_watcher_outage_l2(self) -> None:
        """Resolve the open watcher-outage L2 escalation on healthy recovery.

        Called from _watcher_supervisor_loop when a healthy-clean rotation
        completes (duration >= watcher_misconfigured_min_rotation_secs) to
        signal to the human that the watcher is running again.

        No-op when no pending L2 with root_cause='watcher_supervisor_down' exists.
        Best-effort: failures are swallowed so the supervisor loop never breaks.
        """
        queue = getattr(self, '_escalation_queue', None)
        if not queue:
            return
        try:
            existing_id = queue.find_pending_l2_by_root_cause(
                self._WATCHER_OUTAGE_ROOT_CAUSE
            )
            if existing_id is None:
                return
            queue.resolve(
                existing_id,
                resolution='watcher recovered',
                resolved_by='orchestrator-watcher-supervisor',
            )
            logger.info('Resolved watcher-outage L2 escalation %s (watcher recovered)', existing_id)
        except Exception:
            logger.warning('Failed to resolve watcher-outage L2 escalation', exc_info=True)

    def _file_restored_pause_escalation(self) -> None:
        """File the L1 for a pause restored from a prior run (deferred from run()).

        ``_load_persisted_scheduler_pause`` runs before the escalation queue
        exists, so it only captures ``_restored_pause_reason``; this helper is
        called once the queue is up.  ``has_open_l1`` makes it a no-op when the
        prior run's L1 survived the restart, so the operator sees a single
        persistent L1 across the Restart=on-failure loop.
        """
        if self._restored_pause_reason is None:
            return
        self._file_scheduler_pause_escalation(
            f'(restored from prior run) {self._restored_pause_reason}'
        )

    async def _file_dirty_tree_escalation(self, force_dirty_start: bool) -> None:
        """File (or refresh) the born-at-L2 escalation for a dirty project_root at startup.

        Deferred from the old top-of-run() guard (task 2380): the escalation
        queue only exists once ``_start_escalation_server`` has run, so this
        is called after startup has settled rather than at the original
        pre-server guard site.  RCA 2026-07-08: refusing to start on a dirty
        tree composed with systemd Restart=on-failure + the watchdog revive
        loop into a silent multi-hour crash-loop (459 aborted runs, nothing
        escalated because the process died before it could) — starting
        anyway and escalating loudly replaces that refusal.

        None-safe: a no-op in every branch when ``_escalation_queue`` is None
        (bare-Harness unit tests, or the escalation package missing).  A
        dirty tree with ``force_dirty_start`` set is a silent override (no
        L2 filed).  The orchestrator never auto-commits/stashes/cleans the
        dirty WIP — this only reads and escalates/resolves.

        Self-closing: a clean tree auto-resolves any L2 this sentinel filed
        on a prior dirty startup.  The operator's documented remediation is
        commit-or-stash-and-restart; without this, that remediation would
        leave the L2 pending forever unless the operator also separately
        called ``resolve_issue``.
        """
        dirty = await self.git_ops.has_dirty_working_tree()
        if not dirty:
            if self._escalation_queue is not None:
                for esc in self._escalation_queue.get_by_task(
                    _DIRTY_TREE_ESCALATION_SENTINEL, status='pending', level=2,
                ):
                    self._escalation_queue.resolve(
                        esc.id,
                        'tree now clean at startup',
                        resolved_by='orchestrator-dirty-tree-guard',
                    )
            return
        logger.warning(
            'project_root has uncommitted tracked changes at startup:\n%s',
            dirty,
        )
        if force_dirty_start:
            logger.warning(
                '--force-dirty-start set — starting dirty without filing a '
                'cleanup escalation (silent override)'
            )
            return
        if self._escalation_queue is None:
            return

        from escalation.models import Escalation

        detail = (
            f'project_root has uncommitted tracked changes at startup:\n{dirty}\n\n'
            'The orchestrator started anyway (task 2380: dirty trees no '
            'longer block startup) but this WIP should be committed or '
            'stashed — an uncommitted tree left indefinitely risks being '
            'clobbered by unrelated operations run against project_root.'
        )

        # Dedup across restarts: has_open_l1 is hardcoded to level=1
        # (escalation/queue.py), so get_by_task is used directly (mirrors
        # _alarm_verify_worktree_contention in merge_queue.py). A hit is
        # refreshed in place (same id, re-submitted) rather than filed as a
        # duplicate, so the watchdog's Restart=on-failure loop surfaces one
        # persistent L2 instead of a burst of criticals.
        existing = self._escalation_queue.get_by_task(
            _DIRTY_TREE_ESCALATION_SENTINEL, status='pending', level=2,
        )
        if existing:
            esc = existing[0]
            esc.detail = detail
            esc.timestamp = datetime.now(UTC).isoformat()
            esc.dedupe_count += 1
        else:
            esc = Escalation(
                id=self._escalation_queue.make_id(_DIRTY_TREE_ESCALATION_SENTINEL),
                task_id=_DIRTY_TREE_ESCALATION_SENTINEL,
                agent_role='orchestrator-dirty-tree-guard',
                severity='critical',
                level=2,
                category='cleanup_needed',
                summary='dirty project_root at startup - commit or stash WIP',
                detail=detail,
                suggested_action=(
                    'Commit or stash the dirty files listed above in project_root.'
                ),
            )
        self._escalation_queue.submit(esc)

    async def _file_config_unknown_keys_escalation(self) -> None:
        """File (or self-heal) a born-at-L2 escalation for unknown project-config keys.

        Surfaces the unknown-config-key census (config.py, stashed by load_config
        onto ``self.config.unknown_key_census``) so a key that pydantic's
        ``extra='ignore'`` silently dropped — the 2026-07-22 top-level
        ``spare_warm_lanes`` incident (the field lives on ``git.``) — can never
        again vanish unnoticed for weeks.

        Mirrors ``_file_dirty_tree_escalation``:
          - None-safe: a no-op when ``_escalation_queue`` is None (bare-Harness
            unit tests, or the escalation package missing).
          - Self-closing: an empty census resolves any pending L2 this filer left
            under ``_CONFIG_UNKNOWN_KEYS_SENTINEL`` (operator fixed the config and
            restarted), so the remediation clears the L2 without a manual resolve.
          - Fail-open: the whole body is wrapped in try/except so a fault in the
            escalation path never aborts startup and recreates the RCA 2026-07-08
            silent crash-loop (the startup call site also wraps this in its own
            try/except — defense in depth).

        Dedup: ``root_cause`` encodes the unknown-key-set signature, so an
        identical key-set files exactly one L2 (storm escape, INV-4) via
        ``find_pending_l2_by_root_cause``; a changed set re-files a distinct L2.
        """
        queue = getattr(self, '_escalation_queue', None)
        if queue is None:
            return
        try:
            census = self.config.unknown_key_census
            if not census:
                # Self-heal: the config is now clean — resolve any L2 we filed.
                for esc in queue.get_by_task(
                    _CONFIG_UNKNOWN_KEYS_SENTINEL, status='pending', level=2,
                ):
                    queue.resolve(
                        esc.id,
                        'config now has no unknown keys at startup',
                        resolved_by='orchestrator-config-key-guard',
                    )
                return

            project_id = self.config.fused_memory.project_id
            signature = config_unknown_keys_signature(census)
            root_cause = f'config_unknown_keys:{project_id}:{signature}'
            if queue.find_pending_l2_by_root_cause(root_cause) is not None:
                return  # dedup: one open L2 per unknown-key-set (same-set escape)

            from escalation.models import Escalation

            key_lines = '\n'.join(
                f'  {uk.path}'
                + (f'  → did you mean {uk.shadow_hint}?' if uk.shadow_hint else '')
                for uk in census
            )
            summary = (
                f'{len(census)} unknown config key(s) silently dropped by pydantic '
                'extra=ignore'
            )[:200]
            detail = (
                f'The project config for {project_id} has {len(census)} key(s) with '
                'no matching OrchestratorConfig field.  Pydantic discards unknown '
                'keys BEFORE validation (extra=ignore), so these are SILENTLY '
                'dropped with no error — the 2026-07-22 incident where a top-level '
                'spare_warm_lanes (the field actually lives on git.) was ignored '
                'for weeks.\n\n'
                f'Unknown keys (dotted path → placement hint):\n{key_lines}\n\n'
                'A placement hint is ADVISORY: it is a name match against the '
                'model tree and may be a coincidental collision, so confirm the '
                'key is really misplaced before moving it.\n\n'
                'Fix each key (move it to the hinted path, or remove it) and '
                'restart — a clean census auto-resolves this escalation.  Run '
                '`orchestrator check-config --config <path>` to verify.\n\n'
                'If a key is INTENTIONAL — deliberately present for tooling '
                'other than the orchestrator, e.g. read by this project\'s own '
                'scripts — do not delete it.  Excuse it instead, either by '
                'renaming it under the reserved `x_`/`x-` prefix (works at any '
                'depth, no config ceremony) or by adding its dotted path to '
                '`config_key_census.ignore` in the same YAML (fnmatch globs, so '
                '`some_namespace.*` excuses a whole namespace; note a '
                '`<name>.*` glob does NOT match the bare parent key `<name>`, '
                'which must be listed exactly).  Then restart — or hot-reload, '
                'since `config_key_census.*` is green-tier — and this escalation '
                'auto-resolves.  Excused keys stay listed by check-config at '
                'exit 0, so the opt-out remains auditable.'
            )
            esc = Escalation(
                id=queue.make_id(_CONFIG_UNKNOWN_KEYS_SENTINEL),
                task_id=_CONFIG_UNKNOWN_KEYS_SENTINEL,
                agent_role='orchestrator-config-key-guard',
                severity='critical',
                level=2,
                category='infra_issue',
                root_cause=root_cause,
                summary=summary,
                detail=detail,
                suggested_action='fix_unknown_config_keys',
            )
            queue.submit(esc)
            try:
                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_created,
                        task_id=_CONFIG_UNKNOWN_KEYS_SENTINEL,
                        data={
                            'escalation_id': esc.id,
                            'category': esc.category,
                            'severity': esc.severity,
                            'level': esc.level,
                            'reason': 'config-unknown-keys',
                        },
                    )
            except Exception:
                # Isolated: the L2 is already filed, so an emit failure is an
                # observability-only miss, never a "failed to file L2".
                logger.warning(
                    'config-unknown-keys: L2 %s filed but escalation_created '
                    'emit failed', esc.id, exc_info=True,
                )
            logger.warning(
                'config-unknown-keys: filed born-at-L2 %s for %d unknown key(s): %s',
                esc.id, len(census), ', '.join(uk.path for uk in census),
            )
        except Exception:
            # Fail-open (RCA 2026-07-08): a startup guard fault must never abort
            # startup.  The caller wraps this too; this inner guard keeps the
            # method safe when invoked directly (e.g. unit tests).
            logger.warning(
                'config-unknown-keys: failed to file/heal escalation (non-fatal)',
                exc_info=True,
            )

    async def _run_slot(
        self, assignment, sem: asyncio.Semaphore
    ) -> TaskReport | None:
        """Run a single workflow slot."""
        report = None
        # Set to True in the except Exception handler (any unhandled workflow-slot
        # exception) so the existing requeue cooldown is armed via scheduler.release,
        # preventing the rapid re-block loop where reconciliation immediately
        # re-dispatches the task before it can make progress.
        arm_requeue_cooldown = False
        try:
            logger.info(
                f'Starting workflow for task {assignment.task_id}: '
                f'{assignment.task.get("title", "")}'
            )

            # ── Deterministic gate route ─────────────────────────────────────
            # Check BEFORE substrate gate / steward / TaskWorkflow so that NO
            # worktree, branch, agent, or steward is created (I4/B2).  The
            # finally block still runs on the early return — releasing the
            # semaphore, popping _escalation_events, and calling
            # scheduler.release on the empty lock set held by a gate task.
            if self.scheduler.is_deterministic(assignment.task):
                return await self._run_deterministic_slot(assignment)
            # ────────────────────────────────────────────────────────────────

            # Retrieve the escalation wake-event registered at dispatch time.
            # _register_escalation_event was called at the dispatch point
            # (before create_task) so _escalation_events already has an entry.
            # The defensive create-if-missing branch handles direct _run_slot
            # invocations (e.g. unit tests) that bypass the dispatch path.
            esc_event = self._escalation_events.get(assignment.task_id)
            if esc_event is None and self._escalation_queue:
                esc_event = self._register_escalation_event(assignment.task_id)

            # Soft-cancel event — exposed so external code (reconciliation
            # subscriber, release_workflow MCP tool) can interrupt long
            # workflow waits when the task becomes terminal out-of-band.
            cancel_event = asyncio.Event()
            self._workflow_cancel_events[assignment.task_id] = cancel_event

            # Clear any stale soft-cancel grace stamp from a prior incarnation
            # of this task so a freshly re-dispatched run starts clean (the
            # stamp is no longer popped in the finally — see note there).
            self.scheduler.clear_workflow_cancel(assignment.task_id)

            # Register the asyncio.Task handle for this slot so hard_cancel_workflow
            # can request a hard cancel if the workflow ignores the soft event.
            # current_task() must be non-None here because _run_slot is always
            # scheduled as an asyncio.Task (via create_task in _acquire_and_run_slot).
            current = asyncio.current_task()
            assert current is not None, '_run_slot must be scheduled as a Task'
            self._workflow_slot_tasks[assignment.task_id] = current

            recovered_plan = self._recovered_plans.pop(assignment.task_id, None)
            recovered_session = self._recovered_sessions.pop(assignment.task_id, None)
            recovered_config_dir = self._recovered_session_config_dirs.pop(
                assignment.task_id, None
            )
            # Drop any preserved-worktree marker once the slot picks the task up.
            self._preserved_worktrees.discard(assignment.task_id)

            # ── γ session-resume eligibility guard (task 2774) ────────────────
            # A recovered session is injected as --resume below ONLY when it is
            # fresh, under its per-task resume cap, and its transcript is
            # corroborated on disk. Any ineligible session degrades to a fresh
            # dispatch WITH the recovered plan (I3 — never a stall, never a
            # scheduler-visible error), emitting a reason-carrying event. The
            # kill switch (enabled=False) degrades silently (B6). Streak
            # bookkeeping / storm-escape is layered in by task 2774 step-6.
            if recovered_session is not None:
                eligible, reason = self._session_resume_eligible(
                    recovered_session, recovered_config_dir
                )
                # Capture the session identity for the event BEFORE any nulling.
                resume_event_data = {
                    'session_id': recovered_session.get('session_id'),
                    'role': recovered_session.get('role'),
                }
                if eligible:
                    self._session_resume_fallback_streak = 0  # break any storm run
                    if self.event_store:
                        self.event_store.emit(
                            EventType.session_resume,
                            task_id=assignment.task_id,
                            data=resume_event_data,
                        )
                else:
                    recovered_session = None  # fresh dispatch, recovered plan kept
                    if reason == 'disabled':
                        pass  # kill switch — silent, no event, no streak (B6)
                    elif reason == 'capped':
                        # By-design throttling — its own event, does NOT feed
                        # the storm streak.
                        if self.event_store:
                            self.event_store.emit(
                                EventType.session_resume_capped,
                                task_id=assignment.task_id,
                                data=resume_event_data,
                            )
                    else:  # 'stale' / 'no_transcript' — genuine corroboration fail
                        if self.event_store:
                            self.event_store.emit(
                                EventType.session_resume_fallback,
                                task_id=assignment.task_id,
                                data={**resume_event_data, 'reason': reason},
                            )
                        self._session_resume_fallback_streak += 1
                        if (
                            self._session_resume_fallback_streak
                            >= self.config.session_resume.fallback_storm_threshold
                        ):
                            self._file_session_resume_storm_escalation()
            # ──────────────────────────────────────────────────────────────────

            # Build steward factory — steward starts when the workflow
            # creates its worktree (it needs the path).
            steward_factory = None
            if HAS_STEWARD and self._escalation_queue:
                esc_q = self._escalation_queue  # capture for closure (narrows type)

                def _make_steward(worktree: Path, config_dir=None, *, _assign=assignment) -> TaskSteward:  # type: ignore[name-defined]
                    return TaskSteward(  # type: ignore[reportPossiblyUnbound]
                        task_id=_assign.task_id,
                        task=_assign.task,
                        worktree=worktree,
                        config=self.config,
                        mcp=self.mcp,
                        escalation_queue=esc_q,
                        briefing=self.briefing,
                        usage_gate=self.usage_gate,
                        config_dir=config_dir,
                        event_store=self.event_store,
                        cost_store=self.cost_store,
                    )
                steward_factory = _make_steward

            # ── D4 substrate gate ────────────────────────────────────────────
            # Re-run the committed probe set against current main BEFORE
            # spinning up the agent.  A PASS→FAIL flip (e.g. a sibling deleted
            # Type::Real between author and dispatch) blocks dispatch +
            # escalates rather than wasting an agent spin-up and an L2.
            #
            # The gate runs here — after the slot's initialization bookkeeping
            # but BEFORE TaskWorkflow construction (which spins up the agent).
            # Non-probe tasks (no substrate_probe descriptor) skip the gate
            # entirely so existing dispatch performance is unaffected.
            #
            # NOTE: uses substrate_gate.carries_substrate_probe (key-presence)
            # rather than a Scheduler wrapper — this is the predicate
            # run_substrate_recheck itself uses to decide SKIP vs fail-closed
            # FLIP, so gating dispatch on any other definition (e.g. one that
            # requires a well-formed descriptor) would let a malformed
            # descriptor skip the gate entirely instead of failing closed.
            from orchestrator import substrate_gate  # noqa: PLC0415

            if substrate_gate.carries_substrate_probe(assignment.task) and not await self._run_substrate_gate(assignment):
                # FLIP detected: task is already blocked + escalated inside
                # the gate.  Return a BLOCKED report so the caller (and
                # reconciliation) can observe the outcome; arm the requeue
                # cooldown so the task is not immediately re-dispatched
                # before the blocked-status propagation window closes.
                arm_requeue_cooldown = True
                return TaskReport(
                    task_id=assignment.task_id,
                    title=assignment.task.get('title', ''),
                    outcome=WorkflowOutcome.BLOCKED,
                    block_reason='substrate_flip',
                )
            # ────────────────────────────────────────────────────────────────

            workflow = build_workflow(
                assignment=assignment,
                config=self.config,
                git_ops=self.git_ops,
                scheduler=self.scheduler,
                briefing=self.briefing,
                mcp=self.mcp,
                escalation_queue=self._escalation_queue,
                escalation_event=esc_event,
                usage_gate=self.usage_gate,
                initial_plan=recovered_plan,
                steward_factory=steward_factory,
                merge_queue=self._merge_queue,
                merge_worker=self._merge_worker,
                merge_inflight_registry=self._merge_inflight_registry,
                event_store=self.event_store,
                cost_store=self.cost_store,
                cancel_event=cancel_event,
                resume_session_id=recovered_session,
                run_id=self._run_id,
            )

            if self.event_store:
                self.event_store.emit(
                    EventType.task_started,
                    task_id=assignment.task_id,
                    data={'title': assignment.task.get('title', '')},
                )

            # TR-1: the workflow↔harness terminal contract is this RETURN
            # value, not a `_last_block_*` side channel (deleted — W9-γ).
            # Held in `terminal_report` (NOT `report`) so the `finally`
            # block's `report is not None` guard stays a reliable "did we
            # build a real TaskReport" check — `report` must stay None if an
            # exception fires in the narrow window below (steward-cost
            # computation) before the TaskReport is actually constructed.
            terminal_report: TerminalReport = await workflow.run()
            outcome = terminal_report.outcome

            steward_cost = 0.0
            steward_invocations = 0
            steward = workflow._steward
            if steward and hasattr(steward, 'metrics'):
                steward_cost = steward.metrics.total_cost_usd
                steward_invocations = steward.metrics.invocations

            report = TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=outcome,
                cost_usd=workflow.metrics.total_cost_usd,
                duration_ms=workflow.metrics.total_duration_ms,
                agent_invocations=workflow.metrics.agent_invocations,
                execute_iterations=workflow.metrics.execute_iterations,
                verify_attempts=workflow.metrics.verify_attempts,
                review_cycles=workflow.metrics.review_cycles,
                steward_cost_usd=steward_cost,
                steward_invocations=steward_invocations,
                completed_at=datetime.now(UTC).isoformat(),
                block_reason=terminal_report.reason,
                block_detail=terminal_report.detail,
                # REVIEW-CYCLE-1 fix: block_phase comes from blocked_from_phase
                # (the PRE-block WORKING phase), NOT terminal_report.phase (the
                # terminal machine.state — BLOCKED for a _mark_blocked exit).
                # _maybe_auto_eval gates its optimistic-path redo on block_phase
                # in config.auto_eval_phases = {plan,execute,verify,review};
                # 'blocked' is never a member of that set, so mapping from the
                # terminal phase would silently and permanently disable the
                # Lever B/C auto-eval recovery. blocked_from_phase defaults to
                # None on clean/non-block exits, mapped to '' here — matching
                # the pre-2247 _last_block_phase default.
                block_phase=(
                    terminal_report.blocked_from_phase.value
                    if terminal_report.blocked_from_phase is not None
                    else ''
                ),
                # Task 2988: carry the disposition's cap-accounting policy
                # through to _apply_retry_cap's record_requeue call.
                counts_against_requeue_cap=terminal_report.counts_against_requeue_cap,
            )

            if self.event_store:
                self.event_store.emit(
                    EventType.task_completed,
                    task_id=assignment.task_id,
                    cost_usd=report.cost_usd,
                    duration_ms=report.duration_ms,
                    data={
                        'outcome': outcome.value,
                        'agent_invocations': report.agent_invocations,
                        'execute_iterations': report.execute_iterations,
                        'verify_attempts': report.verify_attempts,
                        'review_cycles': report.review_cycles,
                        'steward_cost_usd': report.steward_cost_usd,
                        'steward_invocations': report.steward_invocations,
                        # Task 3068 (origin incident: reify esc-5556-1) — the
                        # WHY, not just the counters.  A 46h warm-lane requeue
                        # loop was forensically unqueryable because this payload
                        # recorded THAT ~349 dispatches requeued but never why;
                        # both fields were already in scope ~15 lines above and
                        # simply not passed through.
                        #
                        # These two keys are ALWAYS present — empty string on a
                        # clean/DONE exit, never omitted — so that
                        # `json_extract(data,'$.reason')` is uniform across every
                        # row: NULL means "event predates task 3068", '' means
                        # "clean exit, no block".  Conditional omission would make
                        # those two cases indistinguishable, which is exactly the
                        # ambiguity that made the origin incident unqueryable.
                        #
                        # `block_detail` is deliberately NOT emitted: it carries
                        # raw agent/verify output (full test logs, tracebacks) and
                        # is effectively unbounded, while events.db is queried
                        # operationally and rotated.  `block_reason` is the
                        # classified, low-cardinality, GROUP-BY-able field the
                        # requeue-cap path itself already uses.
                        'reason': report.block_reason,
                        'block_phase': report.block_phase,
                    },
                )

            return report
        except asyncio.CancelledError:
            # Hard-cancel: hard_cancel_workflow() called task.cancel() because the
            # workflow ignored the soft cancel_event for ≥ terminal_status_hard_cancel_polls
            # consecutive polls.  Catch here (before `except Exception`) so the
            # wrapper asyncio.Task completes normally (returns a result rather than
            # entering CANCELLED state).  A synthetic TaskReport(outcome=CANCELLED)
            # propagates through _collect_done_reports' normal append path and is
            # persisted by _run_store.save_task_result — symmetric with the BLOCKED
            # report from `except Exception`.  The `finally` block still runs
            # unconditionally (lock release, registry cleanup, scheduler.release).
            logger.warning(
                'Workflow slot for task %s hard-cancelled — '
                'returning synthetic CANCELLED report',
                assignment.task_id,
            )
            # This except only fires for a cancel landing OUTSIDE run()'s
            # CancellationScope (slot setup, post-run() report building, or a
            # mocked-run() unit test) — the workflow's own kind-aware
            # on_terminal cleanup (W9-θ) never ran, so it never released the
            # warm lane either.  The finally block below deliberately has no
            # lane-release call for this path (the retired B2 belt-and-
            # suspenders block used to gate that skip on a dedicated report
            # flag; now there is simply no release call to gate).  Process
            # teardown is NOT "work finished and discardable" — the lane
            # stays ASSIGNED and the periodic terminal-lane reconciler / next
            # acquire reclaims it later.
            report = TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=WorkflowOutcome.CANCELLED,
            )
            return report
        except Exception as e:
            logger.exception(f'Workflow slot error for task {assignment.task_id}: {e}')
            arm_requeue_cooldown = True  # any unhandled slot exception → arm cooldown in finally
            return TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=WorkflowOutcome.BLOCKED,
            )
        finally:
            self._escalation_events.pop(assignment.task_id, None)
            self._workflow_cancel_events.pop(assignment.task_id, None)
            # NB: deliberately do NOT pop _workflow_cancel_at here.  Popping it
            # in the finally defeated the R3 reconcile grace window — it only
            # ever overlapped the period the live slot already protected (via
            # dispatch-table membership), leaving the brief post-exit window
            # uncovered.  The stamp is instead cleared at (re)dispatch (top of
            # _run_slot) and lazily pruned by the sweep's grace-check reader.
            self._workflow_slot_tasks.pop(assignment.task_id, None)
            # Defensive merge-phase grace clear (task 2753): mirrors the
            # inline clear at the durable-enqueue boundary in
            # _submit_to_merge_queue, covering abnormal slot exits (e.g.
            # WorkflowCancelled raised before that inline clear). Unlike
            # _workflow_cancel_at above, clearing here is CORRECT: once the
            # slot exits there is no more pre-enqueue verify to protect — the
            # workflow either enqueued (and cleared inline) or died — so
            # releasing the grace immediately is exactly right (the monotonic
            # stamp would auto-expire regardless; this just does it promptly).
            self.scheduler.clear_merge_phase(assignment.task_id)
            self._terminal_cancel_counts.pop(assignment.task_id, None)
            # W9-θ: the former B2 belt-and-suspenders release (any DONE/CANCELLED
            # that missed B1, e.g. authoritative-cancel returning normally from
            # workflow.run()) is retired.  The workflow's own kind-aware
            # on_terminal cleanup (_on_terminal_cleanups' release_lane entry) now
            # SOLELY owns terminal lane release for every run()-scope exit —
            # hard-cancel, soft-cancel, and genuine DONE/CANCELLED alike — so a
            # second release here would be redundant at best.  A cancel landing
            # OUTSIDE run()'s CancellationScope (the `except asyncio.CancelledError`
            # above) never ran on_terminal either; that is intentional — see the
            # comment there — the periodic terminal-lane reconciler reclaims it.
            requeued = report is not None and report.outcome == WorkflowOutcome.REQUEUED
            if report is not None:
                requeued = await self._apply_retry_cap(
                    assignment.task_id, report, requeued,
                )
                # Auto-eval: when the optimistic path blocks at a phase we
                # care about, dispatch a sibling redo on the full architect
                # path. Best-effort — never blocks slot release.
                try:
                    await self._maybe_auto_eval(assignment, report)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        'Task %s: auto-eval hook failed (non-fatal): %s',
                        assignment.task_id, exc,
                    )
                # Retry-escalation rung (task μ, trigger 1a): on a
                # terminal-failed (BLOCKED) dispatch, bump the routing tier so
                # the NEXT dispatch routes one ladder rung stronger via the
                # retry-tier-up rule. Awaited (not fire-and-forget) so the
                # bump lands before slot release and any fast re-dispatch.
                # Self-guards on outcome; its own try/except keeps it
                # best-effort.
                await self._maybe_bump_routing_tier(assignment, report)
            # PRD task-status-authority C4/D4 (task 2188, omega1): clear the
            # claimant to NULL at slot release — unconditionally, for every
            # outcome (this runs even when report is None, e.g. the
            # deterministic-gate early return above, so a claimant is never
            # left stamped for a slot this process no longer owns). Placed
            # BEFORE scheduler.release so a concurrent re-dispatch racing in
            # right after release cannot have its fresh restamp clobbered by
            # this clear; the residual race is closed by
            # requeue_cooldown_secs (design doc). Best-effort + suppressed —
            # scheduler.set_task_claimant already swallows its own errors,
            # but this belt-and-suspenders suppress guarantees a missing
            # tool/param, absent column, or transient error here can never
            # block scheduler.release/sem.release below. plan.lock forensics
            # are left untouched.
            with contextlib.suppress(Exception):
                await self.scheduler.set_task_claimant(
                    assignment.task_id, claimant_run_id=None, heartbeat_at=None,
                )
            # arm_requeue_cooldown=True (any unhandled workflow-slot exception) arms
            # the existing requeue_cooldown_secs grace window so a freshly-unblocked
            # task is not immediately re-dispatched and re-blocked in a tight loop.
            self.scheduler.release(assignment.task_id, requeued=requeued or arm_requeue_cooldown)
            sem.release()

    async def _run_deterministic_slot(self, assignment) -> TaskReport:
        """Run a deterministic gate task via DeterministicRunner.

        Instantiated with only scheduler + escalation_queue (no git_ops) —
        structurally proving no worktree is created for a gate (I4/B2).
        Returns a TaskReport with block_reason='deterministic_gate' on BLOCKED,
        or a plain DONE report when the gate has been resolved (resume path).
        """
        from datetime import UTC, datetime
        if self._escalation_queue is None:
            raise RuntimeError('_escalation_queue must be initialised before dispatching deterministic tasks')
        # Task 2983 fix (b): dispatch-time double-dispatch guard.  The scheduler
        # can re-select a deterministic task off a STALE eligibility snapshot
        # (deterministic tasks hold no module locks and arm no requeue cooldown
        # on a clean DONE, so nothing but the fused-memory snapshot prevents
        # re-selection).  If the first dispatch already drove the task terminal,
        # a fresh single-task get_status (~30ms) collapses the seconds-wide
        # snapshot-age window: short-circuit WITHOUT constructing or invoking
        # the runner (no second workflow, no crash-window false-positive
        # escalation — the reported esc-2912-1 mode).  Mirrors the terminal-skip
        # idiom in _action_teardown_and_set_status.  Fail-open: get_status
        # returns None on a read failure/absence → not in TERMINAL_STATUSES →
        # dispatch proceeds normally (fix (a) + the runner's own idempotency
        # guards are the backstop, so a transient read failure never strands a
        # legitimately-pending task).
        current_status = await self.scheduler.get_status(assignment.task_id)
        if current_status in TERMINAL_STATUSES:
            logger.info(
                'Task %s: deterministic dispatch skipped — task is already %s '
                '(terminal) at dispatch time; no workflow started '
                '(double-dispatch guard, task 2983)',
                assignment.task_id, current_status,
            )
            return TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=WorkflowOutcome.DONE,
                completed_at=datetime.now(UTC).isoformat(),
                block_reason='',
            )
        runner = DeterministicRunner(
            scheduler=self.scheduler,
            escalation_queue=self._escalation_queue,
        )
        outcome = await runner.run(assignment)
        return TaskReport(
            task_id=assignment.task_id,
            title=assignment.task.get('title', ''),
            outcome=outcome,
            completed_at=datetime.now(UTC).isoformat(),
            block_reason='deterministic_gate' if outcome == WorkflowOutcome.BLOCKED else '',
        )

    async def _find_prior_auto_eval_redos(self, original_id: str) -> list[str]:
        """Find active auto-eval redo siblings spawned from ``original_id``.

        Used by ``_maybe_auto_eval`` to supersede stale redo attempts before
        dispatching a new one, so at most one live redo per ``spawned_from``
        survives at a time. Best-effort: fails open to ``[]`` on any error
        since this is housekeeping/dedupe, not a dispatch-blocking check.
        """
        try:
            tasks = await self.scheduler.get_tasks(statuses=ACTIVE_TASK_STATUSES)
            prior_ids: list[str] = []
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                meta = t.get('metadata') or {}
                if not isinstance(meta, dict):
                    continue
                if not meta.get('auto_eval_redo'):
                    continue
                if str(meta.get('spawned_from') or '') != str(original_id):
                    continue
                tid = str(t.get('id'))
                if tid == str(original_id):
                    continue
                prior_ids.append(tid)
            return prior_ids
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: auto-eval prior-redo lookup failed (%s) — '
                'skipping supersede',
                original_id, exc,
            )
            return []

    async def _filter_cancellable_redos(
        self, candidate_ids: list[str],
    ) -> list[str]:
        """Re-check prior-redo candidates immediately before cancelling them.

        ``candidate_ids`` is normally the snapshot ``_find_prior_auto_eval_redos``
        returned at the *start* of ``_maybe_auto_eval`` — several awaits
        (submit_task + a status flip) before the cancel loop actually runs.
        This re-check closes two hazards in that window (task 2075 review
        amendment):

        - A candidate may have reached a terminal state (done/cancelled) in
          the interim — force-cancelling it would wrongly revert a
          completed/already-cancelled task and emit a spurious reconciliation
          event. The lookup below reports a candidate's CURRENT status
          directly (not re-filtered through ``ACTIVE_TASK_STATUSES``), so a
          candidate that raced to 'done'/'cancelled' comes back with that
          status and is excluded because it is not a member of
          ``_AUTO_EVAL_SUPERSEDE_SAFE_STATUSES``.
        - A candidate may have moved to 'in-progress' (active agent work),
          'blocked' (an open human-facing escalation), 'deferred' (possibly
          a steward's deliberate hold), 'review', or 'merge-deferred' — none
          of those are safe to silently supersede; see
          ``_AUTO_EVAL_SUPERSEDE_SAFE_STATUSES``.

        Uses ``scheduler.get_statuses(ids=candidate_ids)`` — a status-only
        lookup scoped to just these ids (~95% less payload than fetching full
        task dicts via ``get_tasks``, per ``Scheduler.get_statuses``'s own
        docstring). ``_find_prior_auto_eval_redos`` already pays the cost of a
        full ``get_tasks`` fetch once to *discover* candidates by metadata;
        this second, narrower lookup only needs each candidate's current
        status, not its full task dict.

        Only candidates whose CURRENT status is a member of
        ``_AUTO_EVAL_SUPERSEDE_SAFE_STATUSES`` are returned. Best-effort:
        fails open to ``[]`` on any lookup error — a transient failure here
        skips superseding this round rather than cancelling against stale
        information; a later auto-eval trigger (if any) will re-discover the
        same stale siblings and retry.
        """
        if not candidate_ids:
            return []
        try:
            status_by_id, err = await self.scheduler.get_statuses(ids=candidate_ids)
            if err is not None:
                raise err
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Auto-eval supersede re-check failed (%s) — skipping cancel '
                'of %d candidate(s) this round: %s',
                exc, len(candidate_ids), candidate_ids,
            )
            return []
        return [
            cid for cid in candidate_ids
            if status_by_id.get(cid) in _AUTO_EVAL_SUPERSEDE_SAFE_STATUSES
        ]

    async def _maybe_bump_routing_tier(self, assignment, report: TaskReport) -> None:
        """Bump ``metadata.routing.routing_tier`` on a terminal-failed dispatch.

        Task μ (plans/adaptive-model-routing-prd.md Phase 4), trigger (1a).
        Called from ``_run_slot``'s finally next to ``_maybe_auto_eval``, and
        AWAITED before slot release so the bumped tier lands before any fast
        re-dispatch (e.g. a re-pend) reads it. On the NEXT dispatch the
        retry-tier-up rule (defaults.yaml) turns that ``tier>=1`` into one
        ladder rung stronger for the executor role.

        Fires ONLY on ``report.outcome == BLOCKED`` — the unambiguous
        terminal-failed dispatch (boundary test 5). DONE (success, boundary
        test 6) and REQUEUED (an in-process retry of the same work) do NOT
        bump: there is no clean per-requeue lost-work signal here, and bumping
        transient/no-progress requeues would burn top-tier quota against the
        PRD's own intent (design decision).

        The write goes through ``metadata_mode='merge'`` on the ``routing``
        key alone so it never clobbers the sibling metadata or races the
        blocked-status write; ``_bumped_routing_dump`` is read-current-then-+1,
        so the harness-owned counter stays monotonic (invariant 8). Best-effort
        + suppressed: a bump failure must never block slot release.
        """
        if report is None or report.outcome != WorkflowOutcome.BLOCKED:
            return
        try:
            await self.scheduler.update_task(
                assignment.task_id,
                {'routing': _bumped_routing_dump(assignment.task.get('metadata'))},
                metadata_mode='merge',
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: routing-tier bump failed (non-fatal): %s',
                assignment.task_id, exc,
            )

    async def _bump_routing_tier_by_id(self, task_id: str) -> None:
        """Bump ``metadata.routing.routing_tier`` for *task_id* by id.

        The async worker behind ``pre_increment_routing_tier`` (task μ,
        trigger 3 — the escalation ``escalate_model`` flag). Fetches the task's
        CURRENT metadata via ``scheduler.get_task`` (so the read-then-+1 sees
        the latest tier, keeping the counter monotonic — invariant 8) and
        writes the bumped ``routing`` blob through ``metadata_mode='merge'``,
        exactly like the terminal-failure bump. No-ops when the task is absent
        (a synthetic/sentinel task_id resolves to None). Best-effort +
        suppressed: an escalate_model hint must never fail the resolve.
        """
        try:
            task = await self.scheduler.get_task(task_id)
            if task is None:
                return
            await self.scheduler.update_task(
                task_id,
                {'routing': _bumped_routing_dump(task.get('metadata'))},
                metadata_mode='merge',
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: escalate_model routing-tier bump failed (non-fatal): %s',
                task_id, exc,
            )

    def pre_increment_routing_tier(self, task_id: str) -> None:
        """Pre-increment a task's routing tier for its next dispatch (SYNC shim).

        Called from the escalation server's ``resolve_issue`` when
        ``escalate_model=True`` on a resume/restart — a deliberately SYNC entry
        point, because ``resolve_issue`` runs on a FastMCP threadpool worker
        OFF the orchestrator loop and harness code relies on it staying sync
        (see this task's plan design decision). The escalation package must not
        write orchestrator task metadata directly either; the harness owns both
        the metadata write path and the loop. So this shim does NO write itself
        — it bridges the async ``_bump_routing_tier_by_id`` coroutine onto
        ``self._loop`` via ``_schedule_coro_threadsafe`` (mirroring the
        auto-resume-on-resolve pattern), keeping the write harness-owned and
        on-loop. Best-effort: the metadata write lands one dispatch late in the
        worst case (a benign ordering race, documented as a soft hint).
        """
        self._schedule_coro_threadsafe(
            self._bump_routing_tier_by_id(task_id),
            label=f'routing-tier-preincrement-{task_id}',
        )

    async def _maybe_auto_eval(
        self, assignment, report: TaskReport,
    ) -> None:
        """Dispatch an auto-eval redo when the optimistic path blocked.

        Triggers when:
        - ``report.outcome`` is BLOCKED.
        - The original task carries ``metadata.optimistic_path`` (set by
          Lever B or Lever C).
        - The original task is NOT itself an auto-eval redo.
        - ``report.block_phase`` is in ``config.auto_eval_phases``.
        - The 24h auto-eval USD budget has not been exhausted.

        On success, the original branch + worktree are renamed with a
        ``-skip-attempt`` suffix and a sibling task is submitted via
        ``submit_task(planning_mode=True)`` (curator dedupe bypass) with
        ``metadata.force_full_path=True`` to prevent the redo from taking
        an optimistic path itself.

        Once the replacement redo is confirmed created, any still-'pending'
        prior redo siblings sharing the same ``spawned_from`` are cancelled
        (task 2075) so at most one live redo survives per original task.
        'in-progress', 'blocked', 'deferred', 'review', and 'merge-deferred'
        siblings are deliberately left alone — see
        ``_AUTO_EVAL_SUPERSEDE_SAFE_STATUSES``.
        """
        if not getattr(self.config, 'auto_eval_enabled', False):
            return
        if report.outcome != WorkflowOutcome.BLOCKED:
            return
        if report.block_phase not in self.config.auto_eval_phases:
            return

        task_metadata = (assignment.task.get('metadata') or {})
        optimistic_path = str(task_metadata.get('optimistic_path') or '')
        if not optimistic_path:
            return
        if task_metadata.get('auto_eval_redo'):
            return

        # Daily budget check — skip the redo when exhausted.
        used = await self._auto_eval_budget_used_24h()
        if used >= self.config.auto_eval_redo_budget_usd:
            logger.info(
                'Task %s: auto-eval budget exhausted ($%.2f used of $%.2f) — '
                'skipping redo',
                assignment.task_id, used, self.config.auto_eval_redo_budget_usd,
            )
            return

        original_id = assignment.task_id

        # Find prior redo siblings up-front — BEFORE submitting the new
        # redo, so the not-yet-created replacement is never in this set.
        prior_redo_ids = await self._find_prior_auto_eval_redos(original_id)

        # Rename the original branch + worktree so the new task can use a
        # fresh task/<original_id>-redo branch (sibling task gets its own
        # branch from create_worktree on dispatch).
        old_branch = original_id
        new_branch = f'{original_id}-skip-attempt'
        old_path = self.git_ops.worktree_base / old_branch
        new_path = self.git_ops.worktree_base / new_branch
        renamed = False
        if old_path.exists():
            try:
                await self.git_ops.rename_worktree(
                    old_path=old_path,
                    new_path=new_path,
                    old_branch=old_branch,
                    new_branch=new_branch,
                )
                renamed = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'Task %s: auto-eval rename failed (%s) — continuing '
                    'without rename',
                    original_id, exc,
                )

        redo_metadata = {
            'auto_eval_redo': True,
            'auto_eval_pair': str(original_id),
            'spawned_from': str(original_id),
            'force_full_path': True,
            # Retry-escalation rung (task μ, trigger 1b): the redo sibling
            # starts one tier above its parent, so its full-path executor
            # routes a rung stronger via the retry-tier-up rule from its very
            # first dispatch. A FRESH state at parent+1 (only the counter
            # carries forward) — the parent's latest/history mirror entries
            # belong to the parent's dispatches, not this new sibling.
            # Harness-set at submit = harness-stamped, not author-supplied
            # (invariant 8).
            'routing': {
                'routing_tier': RoutingState.from_metadata(task_metadata).routing_tier + 1,
            },
            'modules': list(task_metadata.get('modules') or []),
            'files': list(task_metadata.get('files') or []),
        }

        title = (
            f'[auto-eval redo] {assignment.task.get("title", "")}'
        )[:200]
        description = (
            f'Automated full-architect redo of task {original_id} '
            f'(blocked at phase={report.block_phase} via optimistic path '
            f'{optimistic_path!r}). Original branch+worktree renamed to '
            f'{new_branch} for forensic comparison.'
        )

        try:
            submit_result = await self.scheduler.dispatch_tool(
                'submit_task',
                {
                    'project_root': str(self.config.project_root),
                    'title': title,
                    'description': description,
                    'priority': str(
                        assignment.task.get('priority') or 'medium'
                    ),
                    'metadata': redo_metadata,
                    'planning_mode': True,
                },
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: auto-eval submit_task raised (%s) — aborting redo',
                original_id, exc,
            )
            return

        new_task_id = self._extract_task_id(submit_result)
        if not new_task_id:
            logger.warning(
                'Task %s: auto-eval submit_task returned no task_id (%r)',
                original_id, submit_result,
            )
            return

        # Flip the new task from deferred → pending so the scheduler picks
        # it up on the next tick.
        try:
            await self.scheduler.dispatch_tool(
                'set_task_status',
                {
                    'project_root': str(self.config.project_root),
                    'id': new_task_id,
                    'status': 'pending',
                },
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: auto-eval status flip to pending failed (%s) — '
                'redo task %s left deferred',
                original_id, exc, new_task_id,
            )

        # Supersede prior redo siblings now that the replacement redo is
        # confirmed created — at most one live redo per spawned_from should
        # survive. Re-check each candidate's CURRENT status immediately
        # before cancelling (task 2075 review amendment): the initial
        # `prior_redo_ids` snapshot is several awaits stale by this point
        # (submit_task + the pending flip above), so a candidate may have
        # since completed/been cancelled on its own, or moved to
        # 'in-progress' (active agent work), 'blocked' (an open human
        # escalation), 'deferred' (possibly a steward's deliberate hold),
        # 'review', or 'merge-deferred' — none of those are safe to
        # silently cancel. Only still-'pending' siblings are superseded;
        # see _AUTO_EVAL_SUPERSEDE_SAFE_STATUSES and _filter_cancellable_redos.
        # Best-effort per-sibling: a single cancellation failure must not
        # abort the loop or affect the primary redo dispatch.
        cancellable_redo_ids = await self._filter_cancellable_redos(
            prior_redo_ids,
        )
        for stale_id in cancellable_redo_ids:
            try:
                await self.scheduler.dispatch_tool(
                    'set_task_status',
                    {
                        'project_root': str(self.config.project_root),
                        'id': stale_id,
                        'status': 'cancelled',
                    },
                    timeout=15,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'Task %s: auto-eval supersede cancel of prior redo '
                    '%s failed (%s) — continuing',
                    original_id, stale_id, exc,
                )

        # Cross-reference the new id back onto the original task.
        try:
            await self.scheduler.update_task(
                original_id,
                metadata={
                    **task_metadata,
                    'auto_eval_pair': str(new_task_id),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: auto-eval back-link update failed (%s)',
                original_id, exc,
            )

        # Track for budget accounting.
        self._auto_eval_redo_task_ids.add(str(new_task_id))

        if self.event_store:
            self.event_store.emit(
                EventType.auto_eval_dispatched,
                task_id=str(new_task_id),
                data={
                    'original_task_id': str(original_id),
                    'optimistic_path': optimistic_path,
                    'block_phase': report.block_phase,
                    'rename_succeeded': renamed,
                    'budget_used_24h': used,
                    'superseded_redo_ids': list(cancellable_redo_ids),
                },
            )

        logger.info(
            'Task %s: auto-eval dispatched redo task %s '
            '(rename=%s, budget_used=$%.2f, superseded=%d %s)',
            original_id, new_task_id, renamed, used,
            len(cancellable_redo_ids), cancellable_redo_ids,
        )

    async def _trailing_24h_fetch_one(
        self,
        sql: str,
        leading_params: tuple = (),
        *,
        label: str,
    ) -> tuple | None:
        """Execute an arbitrary trailing-24h aggregate query and return the first row.

        ``sql`` must be a full SELECT ending with ``... completed_at >= ?`` (the
        cutoff is appended internally as the LAST parameter).  ``leading_params``
        are passed in front of the cutoff in the same order.

        Fail-open: returns None when ``cost_store is None`` or when any
        exception is raised by ``_require_conn()`` / cursor execute / fetchone.
        Callers MUST treat None as "skip / assume zero" — dispatch must
        never be blocked by a transient cost-DB error.  The ``label`` is used
        only in the warning log.
        """
        if not sql.rstrip().endswith('completed_at >= ?'):
            raise ValueError(
                '_trailing_24h_fetch_one: sql must end with "completed_at >= ?" '
                f'(got {sql[:80]!r})'
            )
        if self.cost_store is None:
            return None
        cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        try:
            conn = self.cost_store._require_conn()  # type: ignore[attr-defined]
            cur = await conn.execute(sql, (*leading_params, cutoff))
            row = await cur.fetchone()
            await cur.close()
            return tuple(row) if row is not None else None
        except Exception as exc:  # noqa: BLE001 — never block dispatch on this
            logger.warning(
                '%s: trailing-24h cost query failed (%s) — fail-open',
                label, exc,
            )
            return None

    async def _auto_eval_budget_used_24h(self) -> float:
        """Sum cost_usd from invocations table for known auto-eval redo
        task_ids in the trailing 24h.

        Process-local: the redo set resets on harness restart. Acceptable
        because auto-eval is rollout instrumentation that will be sunset
        once metrics stabilise.
        """
        if not self._auto_eval_redo_task_ids:
            return 0.0
        placeholders = ','.join('?' for _ in self._auto_eval_redo_task_ids)
        row = await self._trailing_24h_fetch_one(
            f'SELECT COALESCE(SUM(cost_usd), 0.0) FROM invocations '
            f'WHERE task_id IN ({placeholders}) AND completed_at >= ?',
            tuple(self._auto_eval_redo_task_ids),
            label='auto_eval_budget_used_24h',
        )
        return float(row[0]) if row and row[0] is not None else 0.0

    async def _enforce_cost_ceilings(self) -> None:
        """Check daily cost ceilings and pause the scheduler on breach.

        Runs every dispatch-loop tick (Harness.run()) immediately before
        ``scheduler.acquire_next()``.  Two checks, evaluated in order:

        1. Watcher ceiling (early warning): trailing-24h cost for
           invocations with ``role LIKE '%watcher%'`` vs
           ``config.watcher_daily_cost_ceiling_usd``.
        2. Orch-wide ceiling (safety net): trailing-24h cost for ALL
           invocations vs ``config.orch_daily_cost_ceiling_usd``.

        The first ceiling that trips wins.  When the scheduler is already
        paused (from any source), returns immediately to avoid redundant
        RunStore writes and duplicate log spam.

        Delegates to ``CostStore.cost_totals_in_window`` so the
        role-LIKE-watcher aggregation pattern lives in exactly one place.
        The trailing-24h window is ``[cutoff_24h_iso, now_iso]`` (inclusive
        ``BETWEEN``); any invocations written after the ``now_iso`` snapshot
        are silently excluded — acceptable for a fail-open dispatch guard.
        On any DB error the method returns immediately without pausing
        (fail-open: a transient query failure must never stop dispatch).
        Task 1323.
        """
        if self.scheduler.is_paused:
            return
        if self.cost_store is None:
            return
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        cutoff_24h_iso = (now - timedelta(hours=24)).isoformat()
        try:
            total, watcher = await self.cost_store.cost_totals_in_window(cutoff_24h_iso, now_iso)
        except Exception as exc:  # noqa: BLE001 — never block dispatch on this
            logger.warning(
                '_enforce_cost_ceilings: trailing-24h cost query failed (%s) — fail-open',
                exc,
            )
            return

        if watcher >= self.config.watcher_daily_cost_ceiling_usd:
            logger.warning(
                '_enforce_cost_ceilings: watcher 24h cost $%.2f >= ceiling $%.2f '
                '— pausing scheduler',
                watcher, self.config.watcher_daily_cost_ceiling_usd,
            )
            await self.pause_scheduler('cost_ceiling_watcher_exceeded')
            # File an L2 outage escalation so the operator learns via their L2
            # stream that the watcher has been paused (dedup-safe: no-op if one
            # is already open).  Intentionally NOT done for the orch-wide ceiling
            # below — that's a different failure mode (the whole orchestrator is
            # over budget, not specifically the watcher).
            self._file_watcher_outage_l2('cost_ceiling_watcher_exceeded')
            return

        if total >= self.config.orch_daily_cost_ceiling_usd:
            logger.warning(
                '_enforce_cost_ceilings: orch-wide 24h cost $%.2f >= ceiling $%.2f '
                '— pausing scheduler',
                total, self.config.orch_daily_cost_ceiling_usd,
            )
            await self.pause_scheduler('cost_ceiling_orch_exceeded')

    @staticmethod
    def _extract_task_id(submit_result: Any) -> str | None:
        """Pull the task_id out of an MCP tools/call response wrapper.

        ``dispatch_tool`` returns the raw MCP result envelope. The shape may
        be ``{'task_id': ...}``, ``{'content': [{'text': '{...}'}], ...}``,
        or ``{'structuredContent': {...}}`` depending on transport. Normalise.
        """
        if not isinstance(submit_result, dict):
            return None
        if submit_result.get('task_id'):
            return str(submit_result['task_id'])
        for key in ('structuredContent', 'result'):
            inner = submit_result.get(key)
            if isinstance(inner, dict) and inner.get('task_id'):
                return str(inner['task_id'])
        content = submit_result.get('content')
        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get('type') == 'text':
                    try:
                        parsed = json.loads(str(chunk.get('text') or ''))
                    except (TypeError, ValueError):
                        continue
                    if isinstance(parsed, dict) and parsed.get('task_id'):
                        return str(parsed['task_id'])
        return None

    async def _apply_retry_cap(
        self, task_id: str, report: TaskReport, requeued: bool,
    ) -> bool:
        """Update the per-task REQUEUED counter and fire cap exhaustion.

        Called from ``_run_slot``'s finally block just before
        ``scheduler.release``.  Returns the effective *requeued* flag — False
        when cap exhaustion fires (task is blocked, cooldown is irrelevant),
        otherwise the caller's original value.
        """
        if self._run_id is None:
            return requeued

        # Task 3068 — zero-progress backstop.  Placed here, after the _run_id
        # guard (whose established role as a test no-op hook is preserved) and
        # BEFORE the outcome branch below, so EVERY outcome flows through the
        # tracker: requeues accumulate, DONE/BLOCKED reset.
        #
        # This is deliberately independent of report.counts_against_requeue_cap.
        # workflow_types._disposition_table() sets that False for the warm-lane
        # dispositions, which routes record_requeue below to its history-only
        # path — so NEITHER ceiling in this method can trip for them, and a hard
        # fault masquerading as transient backpressure would otherwise requeue
        # forever, invisibly.  If you are editing the cap logic below, this is
        # why a second, independent detector exists.
        self._maybe_zero_progress_requeue_alert(task_id, report)

        if report.outcome == WorkflowOutcome.REQUEUED:
            attempt_cost = report.cost_usd + report.steward_cost_usd
            count = self.scheduler.record_requeue(
                task_id,
                phase=report.block_phase or 'unknown',
                reason=report.block_reason or 'unknown',
                detail=report.block_detail or '',
                run_id=self._run_id,
                cost_usd=attempt_cost,
                # Task 2988 (PRD ε / W3): a non-counting requeue (e.g.
                # warm_lane_pool_exhausted) is history-only — it never trips
                # the retry-cap escalation; the pool-level structural-
                # exhaustion L2 is the loud signal instead.
                counts_against_cap=report.counts_against_requeue_cap,
            )
            genuine_exhausted = count >= self.config.requeue_cap
            transient_exhausted = (
                self.scheduler.transient_requeue_count(task_id)
                >= self.config.transient_requeue_cap
            )
            if genuine_exhausted or transient_exhausted:
                hit_cap = (
                    self.config.requeue_cap
                    if genuine_exhausted
                    else self.config.transient_requeue_cap
                )
                history = self.scheduler.requeue_history(task_id)
                cumulative_cost = sum(r.cost_usd for r in history)
                try:
                    await self.scheduler.trigger_retry_cap_exhausted(
                        task_id,
                        run_id=self._run_id,
                        cost_usd=cumulative_cost,
                        escalation_queue=self._escalation_queue,
                        cap=hit_cap,
                    )
                except Exception:
                    logger.exception(
                        'Retry-cap trigger failed for task %s', task_id,
                    )
                return False  # task is blocked, skip cooldown
        elif report.outcome == WorkflowOutcome.DONE:
            self.scheduler.clear_requeue_count(task_id)
        return requeued

    def _maybe_zero_progress_requeue_alert(
        self, task_id: str, report: TaskReport,
    ) -> None:
        """Fold one dispatch into the zero-progress streak; alarm at threshold.

        Thin config-reading adapter over ``orchestrator.zero_progress_requeue``
        (the ``_maybe_pipeline_landing_tripwire`` / ``merge_skew_tripwire``
        shape): the module stays pure and injectable, this method supplies
        ``self._escalation_queue`` / ``self.event_store`` / the config.

        Wholly wrapped in try/except by design.  This detector exists to
        backstop the requeue cap; a bug in it must never be able to disturb the
        cap accounting it sits beside.
        """
        try:
            if not self.config.zero_progress_requeue.enabled:
                return
            streak = self._zero_progress_tracker.record(
                task_id,
                outcome=report.outcome,
                agent_invocations=report.agent_invocations,
            )
            emit_zero_progress_requeue_alert(
                escalation_queue=self._escalation_queue,
                event_store=self.event_store,
                task_id=task_id,
                streak=streak,
                threshold=self.config.zero_progress_requeue.threshold,
                block_reason=report.block_reason,
                block_phase=report.block_phase,
            )
        except Exception as exc:  # noqa: BLE001 — never disturb cap accounting
            logger.warning(
                'Task %s: zero-progress requeue check failed (non-fatal): %s',
                task_id, exc,
            )

    def _collect_done_reports(
        self, done: set[asyncio.Task], task_reports: list[TaskReport]
    ) -> None:
        """Extract TaskReports from completed asyncio.Tasks and track merges for review."""
        for t in done:
            # Handle review checkpoint completion
            if t.get_name() == 'review-checkpoint':
                self._review_running = False
                try:
                    t.result()  # propagate exceptions
                except Exception as e:
                    logger.error(f'Review checkpoint error: {e}')
                continue

            try:
                report = t.result()
                if report:
                    task_reports.append(report)
                    # Persist task result immediately so it survives crashes
                    if self._run_store and self._run_id:
                        try:
                            self._run_store.save_task_result(
                                self._run_id, report,
                                self.config.fused_memory.project_id,
                            )
                        except Exception as e:
                            logger.warning(
                                f'Failed to persist task result '
                                f'{report.task_id}: {e}'
                            )
                    # Track module merges for review checkpoints
                    if (report.outcome == WorkflowOutcome.DONE
                            and self.review_checkpoint):
                        modules = self._task_modules.pop(report.task_id, [])
                        self.review_checkpoint.record_merge(modules)
                        # Check if review should trigger
                        if (self.review_checkpoint.should_trigger()
                                and not self._review_running):
                            self._trigger_review_checkpoint()
            except Exception as e:
                logger.error(f'Workflow slot error: {e}')

    def _compute_tallies(self, task_reports: list[TaskReport]) -> None:
        """Fill ``self.report`` aggregates from the collected task reports.

        Shared by the post-loop (exit / --until-idle) block and the
        run-forever idle branch, which recomputes per-cycle before logging the
        cycle summary.
        """
        self.report.task_reports = task_reports
        self.report.completed = sum(
            1 for r in task_reports if r.outcome == WorkflowOutcome.DONE
        )
        self.report.blocked = sum(
            1 for r in task_reports if r.outcome == WorkflowOutcome.BLOCKED
        )
        self.report.escalated = sum(
            1 for r in task_reports if r.outcome == WorkflowOutcome.ESCALATED
        )
        self.report.total_cost_usd = sum(r.cost_usd for r in task_reports)

    async def _run_full_review_and_tag(self) -> None:
        """Run a full review and tag any tasks it creates with code modules.

        The body of the post-completion review minus its gating ``if`` — the
        caller owns the gate (unconditional on the exit path; rate-limited via
        ``should_run_full`` on the run-forever idle path).  Mirrors
        ``_run_review_checkpoint`` but calls ``run_full`` instead of
        ``run_focused``.
        """
        assert self.review_checkpoint is not None
        logger.info('Running full post-completion review...')
        try:
            review_report = await self.review_checkpoint.run_full()
            self.report.review_checkpoints += 1
            self.report.review_findings += review_report.findings_count
            self.report.review_tasks_created += len(review_report.tasks_created)
            self.report.review_cost_usd += review_report.cost_usd
            if review_report.parse_failed:
                logger.warning(
                    'Full review %s: reviewer output was unparseable — review '
                    'inconclusive (findings_count=%d is not a clean pass)',
                    review_report.review_id,
                    review_report.findings_count,
                )
            logger.info(
                'Full review complete: %d findings, %d tasks created',
                review_report.findings_count,
                len(review_report.tasks_created),
            )
            if review_report.tasks_created:
                try:
                    await self._tag_task_modules()
                except Exception as tag_err:
                    logger.warning(f'Post-review module tagging failed: {tag_err}')
        except Exception as e:
            logger.error(f'Full review failed: {e}')

    def _trigger_review_checkpoint(self) -> None:
        """Spawn a review checkpoint as a concurrent task.

        The main loop will pause task acquisition while the review runs
        (``_review_running`` flag) but in-flight tasks continue in their
        worktrees.
        """
        self._review_running = True
        # We can't add to active here — the caller's done set is immutable.
        # Instead, the review task is spawned and tracked; the main loop checks
        # _review_running and waits on active (which includes this task after
        # the next iteration adds it).

        # Actually, we need the task in `active` for the main loop's asyncio.wait.
        # The caller iterates `done`, so we store the task on self for the loop
        # to pick up.
        self._pending_review_task = asyncio.create_task(
            self._run_review_checkpoint(), name='review-checkpoint',
        )

    async def _run_review_checkpoint(self) -> None:
        """Execute a focused review checkpoint."""
        assert self.review_checkpoint is not None
        logger.info('Starting review checkpoint...')
        try:
            review_report = await self.review_checkpoint.run_focused()
            self.report.review_checkpoints += 1
            self.report.review_findings += review_report.findings_count
            self.report.review_tasks_created += len(review_report.tasks_created)
            self.report.review_cost_usd += review_report.cost_usd
            if review_report.parse_failed:
                logger.warning(
                    'Review checkpoint %s: reviewer output was unparseable — '
                    'review inconclusive (findings_count=%d is not a clean pass)',
                    review_report.review_id,
                    review_report.findings_count,
                )
            logger.info(
                'Review checkpoint complete: %d findings, %d tasks created, '
                'cost=$%.2f',
                review_report.findings_count,
                len(review_report.tasks_created),
                review_report.cost_usd,
            )
            # Tag newly created tasks with module metadata (agents may have
            # included modules, but re-run the batch tagger as a fallback
            # for any tasks that lack them).
            if review_report.tasks_created:
                try:
                    await self._tag_task_modules()
                except Exception as tag_err:
                    logger.warning(f'Post-review module tagging failed: {tag_err}')
        except Exception as e:
            logger.error(f'Review checkpoint failed: {e}')

    def _checkpoint_stores(self) -> None:
        """Run WAL TRUNCATE checkpoint on each SQLite store — best-effort.

        Called from the shutdown ``finally:`` block after ``finish_run()`` so
        the WAL is truncated while the event loop is still healthy.  Each store
        is checkpointed independently; a failure in one does not prevent the
        others from running.  Stores that were never created (None) are skipped
        silently — mirrors the init-failure guard at lines 437-453.

        Follows the existing harness shutdown try/except pattern used by
        ``usage_gate.shutdown()``, ``cost_store.close()``, etc.
        """
        if self._run_store:
            try:
                result = self._run_store.checkpoint()
                logger.info(f'run_store checkpoint: {result}')
            except Exception as e:
                logger.warning(f'run_store.checkpoint() failed: {e}')

        if self.event_store:
            try:
                result = self.event_store.checkpoint()
                logger.info(f'event_store checkpoint: {result}')
            except Exception as e:
                logger.warning(f'event_store.checkpoint() failed: {e}')

        override_store = self.scheduler.override_store if self.scheduler else None
        if override_store is not None:
            try:
                result = override_store.checkpoint()
                logger.info(f'override_store checkpoint: {result}')
            except Exception as e:
                logger.warning(f'override_store.checkpoint() failed: {e}')

    def _save_harness_report(self) -> None:
        """Persist HarnessReport as JSON alongside review checkpoint reports."""
        reports_dir = self.config.project_root / self.config.review.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        ts = self.report.started_at.replace(':', '').replace('-', '')[:15]
        path = reports_dir / f'harness-{ts}.json'

        data = {
            'started_at': self.report.started_at,
            'completed_at': self.report.completed_at,
            'total_tasks': self.report.total_tasks,
            'completed': self.report.completed,
            'blocked': self.report.blocked,
            'escalated': self.report.escalated,
            'total_cost_usd': self.report.total_cost_usd,
            'paused_for_cap': self.report.paused_for_cap,
            'cap_pause_duration_secs': self.report.cap_pause_duration_secs,
            'review_checkpoints': self.report.review_checkpoints,
            'review_findings': self.report.review_findings,
            'review_tasks_created': self.report.review_tasks_created,
            'review_cost_usd': self.report.review_cost_usd,
            'task_reports': [
                {
                    'task_id': r.task_id,
                    'title': r.title,
                    'outcome': r.outcome.value,
                    'cost_usd': r.cost_usd,
                    'duration_ms': r.duration_ms,
                    'agent_invocations': r.agent_invocations,
                    'completed_at': r.completed_at,
                }
                for r in self.report.task_reports
            ],
        }

        try:
            path.write_text(json.dumps(data, indent=2))
            logger.info('HarnessReport saved: %s', path)
        except OSError as e:
            logger.warning('Failed to save HarnessReport: %s', e)

    async def _start_merge_worker(self) -> None:
        """Start the merge queue worker as a background asyncio task.

        Uses SpeculativeMergeWorker (two-coroutine pipeline) — the sole
        production merge worker (MQ-refactor task ν retired the legacy serial
        MergeWorker; its readable-reference role now lives as a test-local
        fixture in ``tests/_serial_merge_worker.py``).

        Also builds and stores the StaleServiceRestartCoordinator and wires
        its note_merge method as the merge worker's on_merge_landed callback.
        """
        from orchestrator.merge_queue import (
            MergeLivenessConfigError,
            SpeculativeMergeWorker,
            enforce_merge_liveness_margin,
            enforce_persistent_worktree_serial_lane,
        )

        # K = 1 (local trust-anchor) + number of enabled remote verify runners.
        # Sizes the liveness guard (merge_ahead_bound + num_hosts), the
        # serial-lane bound (num_hosts), and the worker's speculation cap
        # (speculation_depth).  All three knobs derive from ONE expression so
        # they cannot drift apart as verify_runners grows.  num_hosts=K makes
        # the per-host bound ceil(K/num_hosts)=ceil(K/K)=1, so each of the K
        # hosts runs at most one in-flight merge-verify at a time.
        #
        # NOTE: K was previously pinned to _MERGE_AHEAD_BOUND (=1).  Wired from
        # config.verify_runners by task 1716 (Lever C operator-enable path).
        # DO NOT flip reify's orchestrator.yaml verify_runners on until the
        # verdict-parity report is green (PRD D6 gate; see task 1716 analysis).
        # NOTE: use self._speculation_k — single shared source so GitOps spec-pool
        # size (set at __init__) and speculation_depth here cannot drift apart.
        _k: int = self._speculation_k

        # Fail-CLOSED on an over-budget liveness verdict: refuse to start the merge
        # worker and propagate MergeLivenessConfigError to the caller.
        #
        # Physical model (task 1729 / α owner-heartbeat, task 1728):
        # The α heartbeat touches every owned _merge-* worktree's mtime every
        # _HEARTBEAT_POLL_S seconds.  A live worker's worktrees therefore never
        # age past ~1 poll period regardless of K, cold timeout, prefer-remote
        # routing, or queue depth.  The reaper's worst-case frozen age is
        # _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE (event-loop-stall budget),
        # NOT bound × timeout.  K drops out of the formula because the *model*
        # changed, not because the guard weakened.
        #
        # Intentional asymmetry: enforce_persistent_worktree_serial_lane below
        # still receives num_hosts=_k (per-host serial-lane bound is genuinely
        # per-host; that guard's model did not change).  Only the liveness guard
        # drops _k: the two guards describe independent physical constraints.
        #
        # Any OTHER exception (e.g. config resolution failure in a mock or
        # misconfigured environment) is still fail-OPEN (non-fatal warning),
        # preserving the original crash-loop protection for unrelated errors.
        try:
            enforce_merge_liveness_margin(self.config)
        except MergeLivenessConfigError:
            raise  # fail-closed: over-budget verdict → refuse startup
        except Exception as e:
            logger.warning('enforce_merge_liveness_margin failed (non-fatal): %s', e)

        # Fail-CLOSED: persistent warm worktree is serial-lane-only (PRD §10
        # invariant 3).  If the knob is on and _k > 1, refuse startup rather
        # than risk concurrent cargo on a shared target/.
        # num_hosts=_k: each of the K hosts gets its own serial lane so
        # per-host lane bound = ceil(K/num_hosts) = 1.
        enforce_persistent_worktree_serial_lane(
            self.config, merge_ahead_bound=_k, num_hosts=_k,
        )

        self._service_restart_coordinators = [
            self._build_service_restart_coordinator(),
            self._build_dashboard_restart_coordinator(),
            self._build_orchestrator_restart_coordinator(),
        ]

        # Build the callback factory here (where self.scheduler is live) and
        # inject it opaquely into the worker so task γ can construct
        # GroupMergeRequests without the worker importing the scheduler
        # (pure-git-engine layering preserved; the worker never calls the factory).
        train_callback_factory = build_train_callback_factory(self.scheduler, self.git_ops)

        self._merge_worker = SpeculativeMergeWorker(
            self.git_ops,
            self._merge_queue,
            speculation_depth=_k,
            event_store=self.event_store,
            on_merge_landed=self._note_merge_all,
            escalation_queue=self._escalation_queue,
            train_callback_factory=train_callback_factory,
            merge_store=self._merge_store,
            scheduler=self.scheduler,
            mcp=self.mcp,
            usage_gate=getattr(self, 'usage_gate', None),
            cost_store=getattr(self, 'cost_store', None),
            provenance_conflict_sink=self._provenance_conflict_sink,
        )

        # task 2828 (limb 1) — STARTUP SURVIVOR BARRIER.  Reap any orphaned
        # verify subtree from a PREVIOUS run still alive under the warm
        # _merge-verify lane BEFORE scheduling the worker loop, hence before
        # its first reset_persistent_merge_worktree can `git reset --hard` /
        # `git clean -xfd` that tree out from under a live build.  On restart
        # both existing guards are blind to such a survivor: the merge_verify
        # flock died with the process and the holder-pgid lease reads stale
        # (dead pgid → fail-OPEN).  Fail-OPEN here (mirrors the
        # enforce_merge_liveness_margin generic-except precedent above): the
        # barrier is best-effort, so a failure is logged and swallowed rather
        # than aborting merge-worker startup.
        try:
            await self.git_ops.reap_merge_verify_survivors()
        except Exception as e:
            logger.warning(
                'reap_merge_verify_survivors failed (non-fatal): %s', e,
            )

        self._merge_worker_task = asyncio.create_task(
            self._merge_worker.run(), name='merge-worker',
        )
        logger.info('Speculative merge worker started')

    async def _stop_merge_worker(self) -> None:
        """Stop the merge worker gracefully."""
        if self._merge_worker_task is not None and self._merge_worker is not None:
            await self._merge_worker.stop()
            self._merge_worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._merge_worker_task
            self._merge_worker_task = None
            logger.info('Merge worker stopped')

    async def _note_merge_all(
        self, task_id: str, base_sha: str, head_sha: str
    ) -> None:
        """Fan out a merge-landed notification to every service-restart coordinator.

        Passed as the SpeculativeMergeWorker's ``on_merge_landed`` callback.
        The changed-file list is fetched once (shared across all coordinators) to
        avoid redundant git diff invocations — each coordinator applies its own
        prefix filter against the pre-fetched list.

        Each coordinator is notified independently inside its own try/except so
        that an error in one coordinator cannot prevent the remaining coordinators
        from being armed.  Errors are logged at WARNING level and execution
        continues (fail-open).

        A non-zero git diff exit code is surfaced via the error slot and skips
        all coordinators with a WARNING (fail-open); a legitimately empty diff
        (revert merges, ``.task/``-only merges) calls all coordinators with an
        empty file list — no restart is armed, but no coordinators are skipped.

        The offline deep-test lane notifiee (if registered) is notified BEFORE
        the diff fetch below, since it needs only the fact that ``main``
        advanced, not the changed-file list — it must fire on every landed
        advance even when the diff fetch errors (which skips the coordinators).
        """
        await self._note_offline_lane(task_id, base_sha, head_sha)

        prefetched_diff, err = await self.git_ops.get_merge_diff_files(base_sha, head_sha)
        if err is not None:
            logger.warning(
                '_note_merge_all: git diff fetch failed for %s..%s; skipping all coordinators',
                base_sha[:12],
                head_sha[:12],
                exc_info=True,
            )
            return

        try:
            await self._maybe_pipeline_landing_tripwire(
                task_id, base_sha, head_sha, prefetched_diff,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                '_note_merge_all: pipeline-landing tripwire failed for %s..%s: %s',
                base_sha[:12],
                head_sha[:12],
                exc,
                exc_info=True,
            )

        for coord in self._service_restart_coordinators:
            try:
                await coord.note_merge(task_id, base_sha, head_sha, prefetched_diff=prefetched_diff)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    '_note_merge_all: coordinator %s note_merge failed: %s',
                    getattr(coord, '_service_name', repr(coord)),
                    exc,
                    exc_info=True,
                )

    async def _maybe_pipeline_landing_tripwire(
        self, task_id: str, base_sha: str, head_sha: str, prefetched_diff: list[str],
    ) -> None:
        """Advisory pipeline-landing tripwire adapter (task 2382, merge-skew δ).

        Thin fail-open wiring over ``orchestrator.merge_skew_tripwire
        .emit_pipeline_landing_tripwire``: reads the per-project
        ``config.git.load_bearing_oracle_cmd`` knob (``None`` disables the
        tripwire — logged no-op), enumerates in-flight tasks (excluding the
        just-landed *task_id*) as ``(task_id, branch)`` pairs using the
        project's configured ``config.git.branch_prefix`` (NOT a hardcoded
        ``task/`` literal — a project that customizes ``branch_prefix`` must
        still resolve real branch refs here, or every in-flight task is
        silently skipped), and injects the real
        ``git_ops.get_branch_changed_files`` / ``scheduler.update_task``
        callables. The configured ``config.git.load_bearing_oracle_timeout_secs``
        bounds the oracle subprocess so a hung operator-supplied script
        cannot block this hot path indefinitely (I6).

        Called from ``_note_merge_all`` inside its own try/except, after the
        landing diff has already been fetched (``prefetched_diff`` reused,
        no redundant git call) — this method additionally wraps its own body
        in a try/except as a backstop so a bug here can never propagate up
        and skip the service-restart coordinator fan-out (I6: the tripwire
        must never block/delay/reorder the merge-landed hot path).
        """
        try:
            oracle_cmd = self.config.git.load_bearing_oracle_cmd
            if not oracle_cmd:
                logger.debug(
                    '_maybe_pipeline_landing_tripwire: no load_bearing_oracle_cmd '
                    'configured, no-op',
                )
                return

            tasks = await self.scheduler.get_tasks(statuses=ACTIVE_TASK_STATUSES)
            branch_prefix = self.config.git.branch_prefix
            inflight = [
                (str(t['id']), f"{branch_prefix}{t['id']}")
                for t in tasks
                if str(t.get('id')) != str(task_id)
            ]

            async def get_branch_diff(branch: str) -> list[str] | None:
                files, err = await self.git_ops.get_branch_changed_files(branch)
                if err is not None:
                    return None
                return files

            await emit_pipeline_landing_tripwire(
                project_root=self.config.project_root,
                oracle_cmd=oracle_cmd,
                escalation_queue=self._escalation_queue,
                landing_sha=head_sha,
                landing_task_id=task_id,
                landing_changed_files=prefetched_diff,
                inflight=inflight,
                get_branch_diff=get_branch_diff,
                update_task=self.scheduler.update_task,
                oracle_timeout_secs=self.config.git.load_bearing_oracle_timeout_secs,
            )
        except Exception:
            logger.warning(
                '_maybe_pipeline_landing_tripwire: unexpected error for landing %s',
                head_sha,
                exc_info=True,
            )

    async def _note_offline_lane(
        self, task_id: str, base_sha: str, head_sha: str
    ) -> None:
        """Notify the offline deep-test lane's on_post_merge notifiee, if registered.

        β2 (the offline lane worker, not yet built) will set
        ``self._offline_lane_notifiee`` to its own ``on_post_merge`` callback.
        Until then ``self._offline_lane_notifiee`` is None and this is a no-op.

        Contract: the notifiee is awaited synchronously and BLOCKS this call —
        it sits on the merge-landed hot path, ahead of the diff fetch and the
        service-restart coordinator fan-out in ``_note_merge_all``. Only
        exceptions are handled here (fail-open); slowness is not. β2's
        notifiee MUST enqueue-and-return promptly (e.g. flip a dirty flag and
        wake a waiter) rather than perform the deep-test run inline, or it
        will stall post-merge processing for the SpeculativeMergeWorker.
        """
        notifiee = self._offline_lane_notifiee
        if notifiee is None:
            return
        logger.info(
            'offline-lane: on_post_merge %s..%s',
            base_sha[:12],
            head_sha[:12],
        )
        try:
            await notifiee(task_id, base_sha, head_sha)
        except Exception:  # noqa: BLE001
            logger.warning(
                'offline-lane on_post_merge notifiee raised for task %s; ignoring (fail-open)',
                task_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Offline deep-test lane: lifecycle (task 1953, β2)
    # ------------------------------------------------------------------

    async def _start_offline_lane(self) -> None:
        """Start the singleton offline-deep lane worker, enable-gated.

        Both ``offline_lane_enabled`` and ``persistent_offline_deep_worktree``
        must be True — the worker cannot run without its dedicated
        ``_offline-deep`` worktree (δ, task 1952) — so a single knob left off
        is a clean no-op.  Default False/False → byte-identical to prior
        behaviour (trivially revertible).

        Mirrors ``_start_merge_worker``'s create_task launch, but is
        additionally gated on a lockfile acquire: a refused acquire (e.g. a
        second orchestrator instance racing to start the same lane) is
        logged and skipped (fail-open) rather than raised, leaving the
        notifiee slot at None so ``_note_offline_lane`` stays a clean no-op.
        """
        if not (
            self.config.git.offline_lane_enabled
            and self.config.git.persistent_offline_deep_worktree
        ):
            return
        if self._offline_lane_task is not None and not self._offline_lane_task.done():
            return

        lock_path = self.config.project_root / 'data' / 'orchestrator' / 'offline_lane.lock'
        worker = OfflineLaneWorker(
            self.git_ops,
            self.config,
            lock_path=lock_path,
            task_client=_OfflineLaneTaskClient(self.scheduler),
            escalation_queue=self._escalation_queue,
        )
        if not worker.acquire_lock():
            logger.warning(
                'Offline-deep lane: lock acquire refused (another instance holds '
                '%s); skipping start',
                lock_path,
            )
            return

        self._offline_lane_worker = worker
        self._offline_lane_notifiee = worker.on_post_merge
        self._offline_lane_task = asyncio.create_task(worker.run(), name='offline-lane')
        logger.info('Offline-deep lane worker started')

    async def _stop_offline_lane(self) -> None:
        """Stop the offline-deep lane worker gracefully and release its lock.

        Mirrors ``_stop_merge_worker``: cancel + suppress + None.  Also
        releases the worker's lockfile and clears the notifiee slot, so a
        stopped lane leaves ``_note_offline_lane`` a clean no-op again
        (mirroring the pre-start production default) rather than a dangling
        registration pointing at a worker with no running consumer loop.
        Safe to call when the lane was never started (clean no-op — logs
        nothing, matching ``_stop_merge_worker``'s guarded log line).
        """
        if self._offline_lane_task is None and self._offline_lane_worker is None:
            return
        if self._offline_lane_task is not None:
            self._offline_lane_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._offline_lane_task
            self._offline_lane_task = None
        if self._offline_lane_worker is not None:
            self._offline_lane_worker.release_lock()
            self._offline_lane_worker = None
        self._offline_lane_notifiee = None
        logger.info('Offline-deep lane worker stopped')

    def _merge_pipeline_idle(self) -> bool:
        """True when the merge queue and merge-worker pipeline are both quiescent.

        Used as the orchestrator restart coordinator's ``restart_precondition``
        (U2): even at the run-loop's idle quiet-window (``agents_idle=True``),
        a merge can still be queued or in-flight/verifying — restarting the
        orchestrator mid-merge would be disruptive. True only when there is
        nothing queued (``self._merge_queue.empty()``) AND nothing in-flight
        or verifying (``worker.snapshot()['depth'] == 0``) AND no pre-enqueue
        MERGE-phase workflow is inside its grace window
        (``not self._merge_phase_grace_active()`` — task 2753). The merge-phase
        term closes the pre-enqueue gap: a workflow doing its Phase-1 rebase +
        scoped re-verify has NOT yet reached the durable merge journal, so it
        is invisible to the queue-depth + snapshot terms; without this a polite
        redeploy would cancel it seconds before it becomes crash-recoverable.
        True when ``_merge_worker`` is None (bare / unit-test harness — no
        pipeline to drain). Fail-safe: any exception reading the snapshot
        returns False (never restart when drain state is unknown); the
        merge-phase term itself fails toward True-idle (see
        ``_merge_phase_grace_active``).
        """
        if self._merge_worker is None:
            return True
        try:
            worker = self._merge_worker
            depth = worker.snapshot().get('depth', 1)
        except Exception:
            logger.warning(
                '_merge_pipeline_idle: snapshot() failed; treating pipeline as'
                ' NOT idle (fail-safe)',
                exc_info=True,
            )
            return False
        return (
            self._merge_queue.empty()
            and depth == 0
            # NOTE: this polite-path grace term is intentionally UNBOUNDED. Each
            # fresh note_merge_phase_entered restamps a new monotonic anchor, so
            # under chronic merge saturation this arm can defer the polite
            # redeploy for an unbounded wall-clock span. That is by design, NOT a
            # missing bound: each individual stamp still auto-expires after
            # grace_secs, and the eventual-redeploy guarantee comes from the
            # force-fire ceiling (force_fire_after_secs + merge_phase_grace_secs,
            # see maybe_restart) plus the watchdog staleness pass — the polite
            # arm deliberately has no per-hold ceiling of its own.
            and not self._merge_phase_grace_active()
        )

    def _merge_phase_grace_active(self) -> bool:
        """True iff a pre-enqueue MERGE-phase workflow is inside its grace window.

        Reads ``config.orchestrator_restart_merge_phase_grace_secs`` and
        delegates to ``scheduler.merge_phase_grace_active`` (task 2753). Used
        BOTH as the polite path's ``_merge_pipeline_idle`` term and as the
        force-fire path's bounded ``merge_phase_hold`` (see
        ``_build_orchestrator_restart_coordinator``).

        Fails toward FALSE (no grace, proceed) — deliberately UNLIKE
        ``_merge_pipeline_idle``'s snapshot fail-safe which defers. This grace
        is a protective ADD-ON layered on the durable-queue crash-recovery: if
        the check itself errored and we deferred, we would introduce a NEW
        indefinite-veto/livelock failure mode — the very class of bug this task
        fixes. Degrading to "no grace considered" reproduces exact
        pre-task-2753 behaviour (already shipping, safe), and a WARNING keeps
        the failure loud. A ``grace_secs <= 0`` config disables the grace
        entirely (kill switch) without consulting the scheduler.
        """
        try:
            grace_secs = self.config.orchestrator_restart_merge_phase_grace_secs
            if grace_secs <= 0:
                return False
            return self.scheduler.merge_phase_grace_active(grace_secs)
        except Exception:
            logger.warning(
                '_merge_phase_grace_active: merge-phase grace check failed;'
                ' treating as NO grace (fail toward proceeding, NOT deferring)'
                ' — reproduces exact pre-task-2753 behaviour.',
                exc_info=True,
            )
            return False

    def _build_service_restart_coordinator(self) -> StaleServiceRestartCoordinator:
        """Construct a StaleServiceRestartCoordinator from the current config.

        Called once from _start_merge_worker (where git_ops, event_store, and
        config are all live).  The coordinator is stored on self so the
        run-forever idle branch can call maybe_restart(agents_idle=True).

        require_idle stays at its True default: a fused-memory ``--drain``
        restart disrupts in-flight reconciliation, so the polite idle
        quiet-window is preferred.  But ``force_fire_after_secs`` is wired
        from config as the anti-starvation backstop (task 2817) — under
        chronic fleet saturation the run-loop idle branch never runs, so a
        pending restart armed by ``note_merge`` would otherwise starve
        forever (the operator then has to restart fused-memory by hand, cf.
        esc-2814-1).  Once the pending restart is owed past the bound it
        force-fires on the busy-wait branch anyway (bypassing agents_idle +
        the debounce).  fused-memory keeps no min_interval rate cap
        (state_path=None), so nothing throttles that force-fire.  See
        service_restart.StaleServiceRestartCoordinator.maybe_restart and
        config.fused_memory_restart_force_fire_after_secs.
        """
        return StaleServiceRestartCoordinator(
            git_ops=self.git_ops,
            event_store=self.event_store,
            watch_prefixes=self.config.fused_memory_restart_watch_prefixes,
            debounce_secs=self.config.fused_memory_restart_debounce_secs,
            enabled=self.config.fused_memory_restart_on_merge_enabled,
            script_path=self.config.fused_memory_restart_script,
            project_root=self.config.project_root,
            force_fire_after_secs=self.config.fused_memory_restart_force_fire_after_secs,
        )

    def _build_dashboard_restart_coordinator(self) -> StaleServiceRestartCoordinator:
        """Construct a StaleServiceRestartCoordinator for the dashboard (leaf service).

        The dashboard is a leaf node — nothing depends on it — so its restart
        must NOT wait for the orchestrator to be idle.  require_idle=False means
        the coordinator fires as soon as the debounce window elapses, even while
        agents are dispatching.  script_args=[] omits --drain (no recon to drain).

        Called once from _start_merge_worker alongside
        _build_service_restart_coordinator.
        """
        return StaleServiceRestartCoordinator(
            git_ops=self.git_ops,
            event_store=self.event_store,
            watch_prefixes=self.config.dashboard_restart_watch_prefixes,
            debounce_secs=self.config.dashboard_restart_debounce_secs,
            enabled=self.config.dashboard_restart_on_merge_enabled,
            script_path=self.config.dashboard_restart_script,
            project_root=self.config.project_root,
            service_name='dashboard',
            require_idle=False,
            script_args=[],
        )

    def _build_orchestrator_restart_coordinator(self) -> StaleServiceRestartCoordinator:
        """Construct the StaleServiceRestartCoordinator for the orchestrator itself (U2).

        Unlike the fused-memory/dashboard builders, this coordinator restarts
        the SAME service the orchestrator process is running under
        (orchestrator-dark-factory.service), which requires two departures
        from the coordinator's generic default path:

        - ``restart_executor``: a cgroup-escaping ``systemd-run --user`` closure
          (built here, over ``schedule_detached_systemd_restart``) rather than
          the default ``create_subprocess_exec(start_new_session=True)`` spawn.
          ``start_new_session`` detaches the POSIX session/process-group but
          NOT the systemd cgroup, so under this service's
          ``KillMode=control-group`` a ``systemctl restart`` would SIGKILL a
          same-cgroup restart child before it could bring the service back.
          The closure holds its own ``itertools.count()`` so every fire gets a
          distinct transient unit name (``orch-selfrestart-on-merge-{n}.service``);
          collisions would require two fires within one process lifetime inside
          the ~30s external-verify window, which debounce + single-restart
          coalescing make effectively impossible on their own — the
          restart_precondition preference below narrows the polite-path window
          further, though force-fire relies on debounce + coalescing alone
          since it bypasses that preference.
        - ``restart_precondition``: ``self._merge_pipeline_idle`` — even at the
          run-loop's idle quiet-window (``agents_idle=True``), a merge can
          still be queued or in-flight/verifying; this gate additionally
          prefers to defer the fire until the merge pipeline is drained.
          Under chronic fleet saturation this preference alone would starve
          the coordinator indefinitely, so it is now a *polite-path
          preference* rather than a hard precondition: once a pending
          restart has been owed for ``orchestrator_restart_force_fire_after_secs``
          (fleet-redeploy PRD task delta), ``maybe_restart`` force-fires and
          bypasses ``agents_idle``, the debounce, AND this precondition —
          though it never bypasses ``min_interval_secs`` below. This
          in-process force-fire path is NOT covered by
          ``restart-all-orchestrators.sh``'s ``--drain`` merge-drain gate
          (task gamma, ``drain_check.py``) — that gate only guards the
          separate operator/deploy-triggered script path (e.g. a
          ``task_kind='deterministic'`` deploy task's ``before_done.script``),
          which this coordinator never invokes. The executor below always
          runs ``scripts/restart-orchestrator.sh`` directly via
          ``schedule_detached_systemd_restart``, and that script's own
          ``--drain`` is an accepted-and-ignored no-op (see its header):
          graceful shutdown is SIGTERM → cancel-main-task (``cli.py``'s
          ``_make_cancel_handler``, bounded by ``TimeoutStopSec=90s``), and a
          merge interrupted mid-verify by that shutdown is recovered from the
          merge queue's durable, crash-safe journal (task 1772/2153) on the
          next startup — not from a merge-drain gate at restart time.
        - ``merge_phase_hold`` / ``merge_phase_grace_secs`` (task 2753): the
          PRE-enqueue MERGE-phase window (Phase-1 rebase + scoped re-verify,
          before ``merge_queued``) is NOT yet on that durable journal, so it is
          invisible to ``_merge_pipeline_idle``'s queue-depth + snapshot terms.
          ``merge_phase_hold=self._merge_phase_grace_active`` additionally holds
          BOTH the polite path (via the precondition term) and the force-fire
          escape while such a workflow is racing to the queue — but the
          force-fire hold is bounded by an absolute owed-age ceiling of
          ``force_fire_after_secs + merge_phase_grace_secs`` (see
          ``StaleServiceRestartCoordinator.maybe_restart``), so the grace
          bounds rather than indefinitely vetoes. The check fails toward
          proceeding (no grace) on any internal error, so it can never itself
          introduce a new indefinite veto.

        ``require_idle=True`` and ``script_args=[]`` mirror the fused-memory
        coordinator (idle-only; restart-orchestrator.sh takes no positional
        args). Called once from _start_merge_worker alongside the other two
        builders.

        ``on_active_secs`` is clamped to a minimum of 5 (mirroring
        ``DeterministicRunner``'s ``max(int(...), 5)`` clamp on the
        analogous field) so an operator misconfiguring
        ``orchestrator_restart_on_active_secs`` to 0 or a negative value
        can't remove the settle window entirely or produce an invalid
        ``--on-active=`` argument at registration.
        """
        counter = itertools.count()

        async def _systemd_run_restart_executor() -> None:
            # transient_unit is computed once and reused below (both for the
            # escalation detail text and the schedule_ call) — next(counter)
            # must only advance once per fire, or the collision-avoidance unit
            # numbering (see class docstring) would skip a number every fire.
            transient_unit = f'orch-selfrestart-on-merge-{next(counter)}.service'
            on_failure_escalation = (
                EscalationSpec(
                    queue_dir=str(self._escalation_queue.queue_dir),
                    task_id='orchestrator-self-redeploy',
                    summary='Orchestrator self-restart fire-time failure',
                    detail=(
                        f'systemd-run transient unit {transient_unit} exited '
                        'non-zero at fire time (after registration already '
                        f'returned). See: journalctl --user -u {transient_unit}'
                    ),
                )
                if self._escalation_queue is not None
                else None
            )
            await schedule_detached_systemd_restart(
                script=self.config.orchestrator_restart_script,
                script_args=[],
                project_root=self.config.project_root,
                transient_unit=transient_unit,
                on_active_secs=max(self.config.orchestrator_restart_on_active_secs, 5),
                on_failure_escalation=on_failure_escalation,
            )

        # Restart-safe rate cap on the self-redeploy (task 2371). The last-fire
        # timestamp is persisted next to the merge-queue journal
        # (data/orchestrator/*.json); its parent dir is created lazily by the
        # coordinator's atomic writer (mirroring MergeQueueStore._save_raw), so
        # no eager mkdir is needed here. Only the orchestrator's OWN coordinator
        # gets a non-zero cap — the fused-memory/dashboard builders keep the
        # 0.0 default (no gating), so their behaviour is unchanged.
        redeploy_state_path = Path(self.config.project_root) / FLEET_DEPLOY_CLOCK_RELPATH

        return StaleServiceRestartCoordinator(
            git_ops=self.git_ops,
            event_store=self.event_store,
            watch_prefixes=self.config.orchestrator_restart_watch_prefixes,
            debounce_secs=self.config.orchestrator_restart_debounce_secs,
            enabled=self.config.orchestrator_restart_on_merge_enabled,
            script_path=self.config.orchestrator_restart_script,
            project_root=self.config.project_root,
            service_name='orchestrator',
            require_idle=True,
            script_args=[],
            restart_precondition=self._merge_pipeline_idle,
            restart_executor=_systemd_run_restart_executor,
            min_interval_secs=self.config.orchestrator_restart_min_interval_secs,
            force_fire_after_secs=self.config.orchestrator_restart_force_fire_after_secs,
            # Bounded pre-enqueue MERGE-phase hold (task 2753). merge_phase_hold
            # is the same gate as the polite-path restart_precondition term
            # (_merge_phase_grace_active); on the force-fire path maybe_restart
            # honors it only up to the absolute owed-age ceiling
            # force_fire_after_secs + merge_phase_grace_secs, so a rolling merge
            # stream can never push the redeploy past that ceiling. The
            # fused-memory/dashboard builders omit both (defaults None/0.0 → no
            # hold, byte-identical).
            merge_phase_hold=self._merge_phase_grace_active,
            merge_phase_grace_secs=self.config.orchestrator_restart_merge_phase_grace_secs,
            state_path=redeploy_state_path,
            # restart-all-orchestrators.sh is the SOLE on-disk clock writer,
            # stamping only on its verified-fresh exit-0 path — the watchdog
            # backstop reads the same file. Persisting here too (on mere fire/
            # registration of a detached deploy that may later fail) would
            # silently silence the backstop for a full min_interval_secs
            # window (task 2396, fleet-redeploy β; closes hole I2).
            # Caveat: this coordinator's OWN _last_fire_wall still re-seeds
            # from state_path on every process restart, so it can transiently
            # lag the script's post-restart stamp by one fire — see
            # StaleServiceRestartCoordinator's stamp_clock_on_fire docstring.
            stamp_clock_on_fire=False,
        )

    async def _maybe_restart_stale_service(self, *, agents_idle: bool) -> bool:
        """Delegate to all service-restart coordinators in the list.

        Called from the run-forever idle branch (agents_idle=True) and the
        busy-wait branch (agents_idle=False).  Iterates every coordinator in
        self._service_restart_coordinators and fires each whose gate conditions
        are satisfied.  Returns True if at least one coordinator fired a restart,
        False otherwise.  No-op (returns False) when the list is empty.
        """
        any_fired = False
        for coord in self._service_restart_coordinators:
            if await coord.maybe_restart(agents_idle=agents_idle):
                any_fired = True
        return any_fired

    async def _write_merge_heartbeat(self) -> None:
        """Write this unit's fleet-common merge-idle heartbeat (task 2395, α).

        Gathers ON the event loop: ``ORCH_UNIT`` (self-identification, same
        env convention as ``deterministic_runner._default_resolve_own_unit``),
        and ``merge_idle``/``queue_empty``/``depth`` from a SINGLE
        ``self._merge_worker.snapshot()`` read (when a worker exists) using
        the exact idle formula and fallback ``_merge_pipeline_idle`` uses
        (``queue_empty and depth == 0``, missing ``'depth'`` treated as ``1``
        i.e. active) — so idle-truth and the reported depth are always
        mutually consistent and the worker is polled at most once per tick,
        rather than once via ``_merge_pipeline_idle()`` and again here.
        Offloads only the serialize+atomic-write to a thread
        (``asyncio.to_thread``), mirroring
        ``Scheduler._write_snapshot_best_effort``'s loop/thread split — all
        asyncio/in-memory reads happen here, before the thread hop.

        Called from BOTH run-loop rest branches (idle + busy-wait) so a
        saturated unit — which steady-states in the busy-wait branch — still
        heartbeats.  Fail-open: any exception (state read or disk write) is
        swallowed and logged — a heartbeat write must never stop the run loop.

        Known gap: a unit that is continuously dispatching with spare
        semaphore headroom (an assignment ready every tick, ``sem`` never
        full) loops through neither rest branch and will not refresh its
        heartbeat while that lasts.  Closing this with an unconditional
        per-outer-loop-iteration call was evaluated and reverted: it makes
        every ``run()`` invocation write to the fleet-common path on its very
        first tick, and several existing tests drive ``run()`` for real
        without sandboxing ``ORCH_UNIT``/``ORCH_FLEET_DIR`` — verified to
        clobber this host's real ``data/fleet/orchestrator-dark-factory.
        service.json`` when this shell inherits ``ORCH_UNIT`` from the
        systemd unit.  Fixing that would require test-isolation changes
        outside this task's locked scope (harness.py only).  Readers (γ
        drain gate, ε ``--report``) must therefore treat a stale/absent
        heartbeat conservatively — i.e. NOT idle / restart-eligible only
        after a grace period — the same fail-safe stance
        ``_merge_pipeline_idle`` already takes on a read error.
        """
        try:
            unit = os.environ.get('ORCH_UNIT', '')
            queue_empty = self._merge_queue.empty()
            worker = self._merge_worker
            if worker is None:
                # No pipeline to drain (bare / unit-test harness) — mirrors
                # _merge_pipeline_idle's own worker-is-None short-circuit;
                # no snapshot() call is involved on this branch.
                merge_idle = self._merge_pipeline_idle()
                depth = 0
            else:
                depth = int(worker.snapshot().get('depth', 1))
                merge_idle = queue_empty and depth == 0
            ts_epoch = time.time()
            payload = build_heartbeat_payload(
                unit=unit,
                merge_idle=merge_idle,
                depth=depth,
                queue_empty=queue_empty,
                ts_epoch=ts_epoch,
            )
            await asyncio.to_thread(write_heartbeat, resolve_fleet_dir(), unit, payload)
        except Exception:
            logger.warning(
                'merge heartbeat write failed; continuing (fail-open)',
                exc_info=True,
            )

    def _build_task_status_lookup(self) -> Callable[[str], Awaitable[str | None]]:
        """Return an async callable (task_id) -> str|None backed by the scheduler.

        The returned closure is injected into the escalation MCP server as
        ``task_status_lookup`` so the chokepoint can query live task status
        without holding a direct reference to the Harness.
        """
        async def _lookup(task_id: str) -> str | None:
            return await self.scheduler.get_status(task_id)

        return _lookup

    async def _start_escalation_server(self) -> None:
        """Start the escalation MCP server as a background asyncio task."""
        if not HAS_ESCALATION:
            logger.info('Escalation package not installed — skipping escalation server')
            return
        assert HAS_ESCALATION  # narrows type: EscalationQueue and create_server are defined

        queue_dir = Path(self.config.escalation.queue_dir)
        if not queue_dir.is_absolute():
            queue_dir = self.config.project_root / queue_dir
        self._escalation_queue = EscalationQueue(queue_dir)  # type: ignore[possibly-unbound]
        self._escalation_queue.set_notify_callback(self._on_escalation)
        self._escalation_queue.set_resolve_callback(self._on_escalation_resolved)

        # Late-bind the shared provenance-conflict sink (task 2677) now that
        # the real queue exists — self._provenance_conflict_sink was
        # constructed with escalation_queue=None in __init__ (the queue
        # doesn't exist yet at that point). Injected by reference into the
        # SpeculativeMergeWorker, so the worker's copy of the attribute
        # updates automatically.
        self._provenance_conflict_sink.escalation_queue = self._escalation_queue

        # Wire escalation queue into review checkpoint so it can triage
        # escalations the deep reviewer emits against the synthetic review
        # task_id (which has no workflow/steward to handle them).
        if self.review_checkpoint is not None:
            self.review_checkpoint.escalation_queue = self._escalation_queue

        # Wire escalation queue into the scheduler (task 2408, mechanism 2):
        # the blocked-redispatch sweep (_phase_redispatch_stranded_blocked)
        # needs it to verify "no open escalation" before flipping a
        # genuinely-stranded blocked task back to pending. The Scheduler is
        # constructed long before self._escalation_queue exists, so
        # constructor injection is impossible — attribute injection here
        # mirrors the review_checkpoint line immediately above. Fails safe:
        # without this, scheduler.escalation_queue stays None and the sweep
        # never flips anything (see Scheduler._phase_redispatch_stranded_blocked).
        self.scheduler.escalation_queue = self._escalation_queue

        mcp_server = create_server(  # type: ignore[possibly-unbound]
            self._escalation_queue,
            merge_queue=self._merge_queue,
            orch_config=self.config,
            event_store=self.event_store,
            harness=self,
            task_status_lookup=self._build_task_status_lookup(),
            merge_inflight_registry=self._merge_inflight_registry,
        )
        host = self.config.escalation.host
        port = self.config.escalation.port

        async def _serve():
            import uvicorn
            app = mcp_server.http_app()
            uv_config = uvicorn.Config(
                app, host=host, port=port, log_level='warning',
            )
            server = uvicorn.Server(uv_config)
            await server.serve()

        self._escalation_task = asyncio.create_task(_serve(), name='escalation-server')
        logger.info(f'Escalation MCP server starting on {host}:{port}')
        # Give the server a moment to bind, then verify it didn't crash
        await asyncio.sleep(0.5)
        if self._escalation_task.done():
            exc = self._escalation_task.exception()
            if exc:
                raise RuntimeError(
                    f'Escalation server failed to start on {host}:{port}: {exc}'
                ) from exc

    async def _dismiss_stale_escalations(self) -> None:
        """Dismiss stale L0 escalations left over from prior orchestrator runs.

        Called right after _start_escalation_server() so that any L0
        (agent→steward) escalations persisted in the queue directory from a
        previous (crashed or completed) run are cleared before the new run
        begins.

        L1 (steward→auto-watcher) escalations are intentionally preserved across
        restart — they represent items pending auto-triage (which may then promote
        to L2 for a human) and must not be silently lost during long AFK periods.
        """
        if self._escalation_queue is None:
            return

        resolution = (
            'Auto-dismissed: orchestrator restarted — stale from prior run'
        )
        count = self._escalation_queue.dismiss_all_pending(resolution)
        if count:
            logger.info(
                f'Dismissed {count} stale L0 escalation(s) from prior run; '
                f'L1 escalations preserved across restart'
            )

    def _rehydrate_merge_halt(self) -> str | None:
        """Restore merge-halt state from preserved L1 escalations after restart.

        On restart, SpeculativeMergeWorker is constructed fresh (un-halted,
        no owner).  _dismiss_stale_escalations intentionally preserves pending
        level-1 wip_conflict/unmerged_state/stash_failed escalations — but
        NOTHING re-asserts the corresponding halt or re-registers the halt
        owner.  (stash_failed is the main-checkout-hygiene halt category from
        task 2758: a park of project_root's dirty tracked tree failed.)

        This method scans the settled post-dismissal queue for preserved L1s
        of the relevant categories and restores the (halted, owner-registered)
        state, so the existing _on_escalation_resolved -> unhalt_wip path
        cleanly releases the halt when the operator resolves the L1.

        Returns the escalation id that now owns the halt, or None if no action
        was taken.
        """
        if self._merge_worker is None or self._escalation_queue is None:
            return None

        candidates = [
            esc for esc in self._escalation_queue.get_pending()
            if esc.level == 1
            and esc.category in {'wip_conflict', 'unmerged_state', 'stash_failed'}
        ]
        if not candidates:
            return None

        if len(candidates) > 1:
            logger.warning(
                '_rehydrate_merge_halt: %d qualifying L1s found; registering '
                'only the most recent as halt owner.  The merge queue will '
                'resume as soon as that owner L1 is resolved — even though '
                '%d older L1(s) remain pending.  Resolve the most-recent L1 '
                'last to avoid premature queue resumption.',
                len(candidates),
                len(candidates) - 1,
            )
        esc = max(candidates, key=lambda e: datetime.fromisoformat(e.timestamp))
        reason = (
            f'Rehydrated merge halt from preserved L1 {esc.id} '
            f'(category={esc.category}) after restart'
        )
        self._merge_worker.halt_for_wip(reason)
        self._merge_worker.set_halt_owner(esc.id)
        logger.warning(reason)
        return esc.id

    async def _recover_pending_merges(self) -> dict:
        """Rehydrate in-flight merge requests from the durable journal (task 1772).

        Delegates to ``recover_pending_merges`` which:
        - Drops records whose branch is missing or already landed on main.
        - Collapses per-branch duplicate journal entries through the SHARED
          in-flight registry BEFORE enqueue (task 2926, C3 γ): the descendant-
          most snapshot tip enqueues once and the rest attach as peer waiters,
          so a branch double-rehydrated by the journal never dispatches two
          concurrent verifies of one work item (the 2026-07-22 task/5326
          double-enqueue).
        - Re-enqueues each surviving winner via ``enqueue_merge_request`` so a
          polling ``merge_request`` caller resolves once the merge finishes.

        Passes ``registry=self._merge_inflight_registry`` — the SAME shared
        :class:`InFlightMergeRegistry` the live submit path uses — so a
        concurrent live ``merge_request`` during startup coalesces against the
        recovered entries too.  ``retention`` is deliberately NOT threaded here:
        the merge worker (owner of ``TerminalOutcomeRetention``) may not be
        constructed yet at this startup step, so the recovery alias stays
        best-effort/None and the observable contract is the in-process attach
        future-mirror rather than a durable cross-restart poll-alias.

        Called once from ``run()`` immediately after ``_rehydrate_merge_halt``
        so a halted queue buffers the re-enqueued items rather than merging
        them.  Non-fatal: a corrupt or partial journal logs a warning and lets
        startup continue rather than blocking the orchestrator.

        Returns the ``recover_pending_merges`` report dict (``recovered`` /
        ``dropped`` / ``coalesced`` / ``requests`` / ``journal_corrupt``) so
        ``run()`` can thread the recovered requests' branches into
        :meth:`_reap_orphaned_merge_worktrees` (task 2060) — a recovered
        in-flight worktree is re-adopted rather than reaped.
        """
        report = await recover_pending_merges(
            self._merge_store,
            self._merge_queue,
            self.git_ops,
            self.config,
            event_store=self.event_store,
            main_branch=self.config.git.main_branch,
            branch_prefix=self.config.git.branch_prefix,
            registry=self._merge_inflight_registry,
        )
        logger.info(
            '_recover_pending_merges: recovered=%d dropped=%d coalesced=%d '
            'journal_corrupt=%s',
            report.get('recovered', 0),
            report.get('dropped', 0),
            report.get('coalesced', 0),
            report.get('journal_corrupt', False),
        )
        return report

    async def _reap_orphaned_merge_worktrees(self, recovered_requests) -> None:
        """Reap aged orphaned ``_merge-*`` worktrees at startup (task 2060, I6).

        Startup backstop for the merge-worktree-ledger leak: an on-disk
        ``_merge-<uuid>`` worktree can survive a mid-run SIGTERM (skipped
        ``finally`` cleanup) or an orchestrator restart (which wipes the
        in-memory ``_owned_merge_worktrees`` ledger while the worktree + its
        ``.git/worktrees`` admin entry persist).  Delegates to
        :meth:`SpeculativeMergeWorker.reap_orphaned_merge_worktrees`, which
        re-adopts any worktree still backing a recovered in-flight merge and
        reaps the rest once they age past the resource-audit grace window.

        *recovered_requests* is the ``report['requests']`` list from
        :meth:`_recover_pending_merges`; each request's ``branch`` is passed as
        ``recovered_branches`` so its worktree is re-adopted (registered) rather
        than reaped.  A no-op when the merge worker is absent (disabled / not
        yet constructed); the worker method is itself fail-open per record.
        """
        if self._merge_worker is None:
            return
        branches = [req.branch.bare_id for req in recovered_requests]
        report = await self._merge_worker.reap_orphaned_merge_worktrees(
            recovered_branches=branches,
        )
        logger.info(
            '_reap_orphaned_merge_worktrees: readopted=%d reaped=%d',
            len(report.get('readopted', [])),
            len(report.get('reaped', [])),
        )

    async def _reconcile_landed_outbox(self) -> None:
        """Reconcile the durable LandedOutbox at startup (task 2155, W1 γ).

        Closes the crash window between a merge advancing ``main`` and the
        task being marked done: delegates to the module-level
        :func:`reconcile_landed_outbox`, which resolves each unconsumed row
        to RC-1 (not actually landed → prune, no phantom done), RC-2 (landed
        but not yet marked done → drive the done-write, then prune), or RC-3
        (already done → prune only, no second done-write).  A no-op when the
        merge worker is absent (disabled / not yet constructed) or has no
        bound ``LandedOutbox``, mirroring
        :meth:`_reap_orphaned_merge_worktrees`'s None-guard.
        """
        if self._merge_worker is None or self._merge_worker._landed_outbox is None:
            return
        report = await reconcile_landed_outbox(
            self._merge_worker._landed_outbox, self.git_ops, self.scheduler,
            provenance_conflict_sink=self._provenance_conflict_sink,
        )
        logger.info(
            '_reconcile_landed_outbox: pruned_not_landed=%d marked_done=%d '
            'already_done_pruned=%d skipped=%d stale_conflict=%d errors=%d',
            report.get('pruned_not_landed', 0),
            report.get('marked_done', 0),
            report.get('already_done_pruned', 0),
            report.get('skipped', 0),
            report.get('stale_conflict', 0),
            report.get('errors', 0),
        )

    async def _landed_dispatch_gate(self, task_id: str) -> bool:
        """Consult-before-dispatch gate on the landed-outbox (task 2156, W1 δ
        — PRD merge-queue-reliability §8.2 SD-1, boundary B5).

        Installed as ``self.scheduler._landed_outbox_gate`` so
        ``Scheduler.acquire_next()`` consults it per candidate before either
        dispatch loop commits.  Delegates to the module-level
        :func:`reconcile_landed_task`, which shares the exact
        reconcile-to-done routine used by the startup reconciler
        (:func:`reconcile_landed_outbox`) — returning True means
        ``task_id``'s ``advanced_sha`` is an ancestor of ``main`` (its merge
        already landed) and it has just been driven to ``done`` inline, so
        the scheduler must not dispatch it this tick.

        None-guarded and resolved LAZILY at call time — mirrors
        :meth:`_reconcile_landed_outbox` — because the merge worker (and its
        bound ``LandedOutbox``) is constructed in ``_start_merge_worker``,
        AFTER this callback is wired onto the scheduler in ``__init__``.
        Fails open (returns False, i.e. does not gate dispatch) when the
        worker is absent (disabled / not yet started).

        Passes the shared ``self._provenance_conflict_sink`` through to
        :func:`reconcile_landed_task` (task 2677): a ``done_evidence_stale``
        rejection of the inline done-write now reports ``'stale_conflict'``
        rather than propagating, and ``reconcile_landed_task`` maps that
        disposition to ``True`` too — a contested task must not dispatch
        while under arbitration, same as an already-landed one.
        """
        if self._merge_worker is None or self._merge_worker._landed_outbox is None:
            return False
        return await reconcile_landed_task(
            task_id,
            git_ops=self.git_ops,
            scheduler=self.scheduler,
            outbox=self._merge_worker._landed_outbox,
            provenance_conflict_sink=self._provenance_conflict_sink,
        )

    async def _already_landed_dispatch_gate(self, task_id: str) -> bool:
        """Pre-dispatch gate for OUT-OF-BAND already-landed tasks (task 2313).

        Architecturally parallel to :meth:`_landed_dispatch_gate` but
        consults LIVE GIT STATE rather than the durable LandedOutbox, so it
        also catches landings that never went through this orchestrator's
        merge queue: a sibling direct-merge, a prior orchestrator run, or a
        squash/rebase/manual landing.  Installed as
        ``self.scheduler._already_landed_gate``.

        **Cheap pre-filter** (task 2313 review): ``resolve_branch_sha`` is
        resolved ONCE up front — a single ``git rev-parse --verify`` — so
        the common per-tick case (a branch that doesn't exist yet, or
        doesn't exist any more) skips ``is_ancestor`` and
        ``branch_content_in_main`` entirely instead of spawning subprocesses
        that would only fail through to False.  When the branch is absent,
        only the merge-marker search runs, since that's the one check that's
        meaningful without a live ref; ``find_merge_marker``'s own internal
        gate already returns None whenever the branch ref still exists, so
        calling it in the branch-exists case would always be a wasted
        no-op.  The task-citation lookup is likewise deferred into the two
        branches that actually consume it (ancestry and content-equivalence)
        rather than hoisted, so the common not-landed path never pays for
        it.

        Ancestry-path guards mirror ``_reconcile_one_stranded``'s own guard
        sequence so this gate can never flip a false positive: an open L1
        escalation is a deliberate human handoff (never second-guessed); a
        degenerate branch (tip == branch_base_sha) carries zero task work,
        so ``is_ancestor`` returning True is a trivial false "already on
        main" signal; and a missing citation rejects the zero-commit-branch
        shape where no commit on main actually cites this task.

        The branch-deleted merge-marker path (also mirroring
        ``_reconcile_one_stranded``) catches the case where the branch ref
        itself is gone but a commit citing this task landed on main: a
        marker that is an ancestor of ``branch_base_sha`` predates this
        incarnation (branch deleted + re-created under the same task id)
        and vetoes the flip.

        The content-equivalence fallback catches landings that are NOT
        ancestors of main (squashed/rebased/manually-applied) by comparing
        the branch's actual changed files against main
        (:meth:`~orchestrator.git_ops.GitOps.branch_content_in_main`).  It
        delegates to :func:`~orchestrator.landing_evidence.validate_landing_evidence`
        (DISCOVERY mode, task 2678) to discover and attribute the citation
        commit and anchors on it; a content-equivalent landing that carries
        no task-citing commit (or whose citation's effect is no longer
        present at main HEAD) is no longer silently anchored on main HEAD —
        it escalates instead via ``_file_unattributed_landing_escalation``.
        **Accepted risk**: this path can false-positive
        on a branch whose completed-so-far files coincidentally match
        main's independent content while the rest of the task's scope is
        still unfinished — the primitive only sees the branch's own diff
        footprint, not the task's intended full scope.  This is a
        deliberate tradeoff to catch genuine squash/rebase/manual landings
        (see :meth:`~orchestrator.git_ops.GitOps.branch_content_in_main`'s
        docstring for the accepted-risk rationale).

        **done_evidence_stale short-circuit** (task 2677): if any of the
        three flip attempts below hits ``_mark_in_progress_done`` and that
        call's ``scheduler.mark_done`` is refused by the found_on_main
        provenance-integrity gate (evidence predates the task's
        ``reopen_at``), ``_mark_in_progress_done`` catches the rejection,
        routes it to the shared ``ProvenanceConflictSink``, and returns
        ``False`` — but this gate still returns ``True`` immediately after
        (a contested task must never dispatch while under arbitration). The
        ``should_skip`` pre-check right after ``metadata`` is resolved below
        makes that terminal-for-this-tick outcome cheap on every SUBSEQUENT
        tick at the same ``reopen_at``: it short-circuits before the
        git-ancestry subprocess work and before re-attempting the
        already-rejected write.
        """
        if (
            self._escalation_queue is not None
            and self._escalation_queue.has_open_l1(task_id)
        ):
            return False

        branch = f'{self.git_ops.config.branch_prefix}{task_id}'
        task = await self.scheduler.get_task(task_id)
        metadata = (task.get('metadata') or {}) if task else {}

        # task 2677: a prior tick may have already had this exact
        # (task_id, reopen_at) rejected as done_evidence_stale by
        # _mark_in_progress_done below — the sink's in-memory memo makes
        # that terminal-for-this-tick. Short-circuit BEFORE the git-ancestry
        # work (is_ancestor / find_task_citation_commit /
        # commit_effect_present_in_main are each a subprocess) rather than
        # re-deriving the same contested evidence and re-attempting the
        # already-doomed write every dispatch tick.
        if self._provenance_conflict_sink.should_skip(
            task_id, reopen_at=metadata.get('reopen_at'),
        ):
            return True

        branch_tip_sha = await self.git_ops.resolve_branch_sha(branch)
        branch_exists = branch_tip_sha is not None

        if branch_tip_sha is not None and await self.git_ops.is_ancestor(
            branch, self.git_ops.config.main_branch,
        ):
            if await self._branch_is_degenerate(branch, metadata):
                return False
            # Delegates FIX 2 citation-lineage + FIX 1' effect-present to the
            # shared helper (task 2678, INV-5) — see
            # orchestrator.landing_evidence's module docstring for the full
            # DISCOVERY-mode contract (citation discovery, both-direction
            # lineage guard, branch-tip-or-citation effect anchor).
            verdict = await validate_landing_evidence(
                self.git_ops, task_id, branch,
                branch_tip_sha=branch_tip_sha,
                pattern_template=self.git_ops.config.commit_citation_pattern,
            )
            if not verdict.accepted:
                # No escalation here, deliberately — unlike the marker and
                # content-equivalence paths below (design decision, task
                # 2678). This branch is LIVE: a reject (no citation, a
                # lineage mismatch, or a task-1175-shape effect-absent
                # revert) self-heals by re-dispatch on the next tick with no
                # human involved, and this gate is re-run on the order of
                # every dispatch tick — escalating here would be noise, not
                # signal. This silent-False shape also predates task 2678
                # (it returned False here before the helper extraction too),
                # so this is not a newly-introduced silent failure.
                return False
            await self._mark_in_progress_done(
                task_id, verdict.evidence_sha,
                'reconcile: pre-dispatch check found branch already on main',
                'dispatch-gate-already-on-main',
            )
            return True

        if not branch_exists:
            marker = await self.git_ops.find_merge_marker(
                branch, gate_on_existing_ref=False,
            )
            if marker:
                branch_base_sha = metadata.get('branch_base_sha')
                if _is_valid_sha_40(
                    branch_base_sha,
                ) and await self.git_ops.is_ancestor(marker, branch_base_sha):
                    return False
                # CANDIDATE mode (task 2678): the marker's subject match
                # already attributes this landing to task_id — only the
                # FIX 1' effect-present guard remains, closing the task-1175
                # clobber (a reverted merge previously stamped done anyway).
                verdict = await validate_landing_evidence(
                    self.git_ops, task_id, branch,
                    branch_tip_sha=None,
                    candidate_sha=marker,
                )
                if not verdict.accepted:
                    self._file_unattributed_landing_escalation(task_id, branch, verdict)
                    return False
                await self._mark_in_progress_done(
                    task_id, marker,
                    'reconcile: pre-dispatch check found merge marker on main',
                    'dispatch-gate-marker-found',
                )
                return True
            return False

        if await self.git_ops.branch_content_in_main(branch):
            # DISCOVERY mode (task 2678): delegates citation discovery + FIX 2
            # lineage + FIX 1' effect-present to the shared helper, exactly
            # like the ancestry path above. The prior silent
            # ``citation or get_main_sha()`` fallback fabricated an anchor
            # from main HEAD whenever no citation was found; that is now a
            # rejected verdict ('no_citation') that escalates instead of
            # stamping done.
            verdict = await validate_landing_evidence(
                self.git_ops, task_id, branch,
                branch_tip_sha=branch_tip_sha,
                pattern_template=self.git_ops.config.commit_citation_pattern,
            )
            if not verdict.accepted:
                self._file_unattributed_landing_escalation(task_id, branch, verdict)
                return False
            await self._mark_in_progress_done(
                task_id, verdict.evidence_sha,
                'reconcile: pre-dispatch check found content-equivalent '
                'landing on main (squash/rebase/manual)',
                'dispatch-gate-content-equivalent',
            )
            return True

        return False

    def _file_unattributed_landing_escalation(
        self, task_id: str, branch: str, verdict: LandingEvidenceVerdict,
    ) -> None:
        """Best-effort, dedup-guarded L1 escalation for unattributable
        landing evidence (task 2678, INV-5).

        Filed by the already-landed re-derivation sites that found a
        positive landing signal (a merge marker, or branch content
        equivalent to main) but whose :func:`validate_landing_evidence`
        verdict came back rejected — an unattributed or effect-absent
        landing that must not be silently stamped done (the task-1175
        clobber this task closes). Escalate-instead-of-stamp is
        deliberately non-status-blocking: the task row is simply left
        pending and re-evaluated next tick, and the open-L1 veto at the top
        of :meth:`_already_landed_dispatch_gate` naturally suppresses
        reprocessing while this L1 stays open — no separate status
        transition is needed here.

        Best-effort (a no-op when ``_escalation_queue`` is None, e.g.
        bare-Harness unit tests) and deduped via ``has_open_l1`` so repeated
        ticks re-observing the same unattributable evidence don't stack
        duplicate L1s — one open escalation per task at a time.

        Delegates the actual filing (queue-None guard, dedup, ``Escalation``
        construction, submit/log/except shape) to the shared
        :func:`~orchestrator.landing_evidence.file_unattributed_landing_escalation`
        (task 2678 amendment, INV-5) — this method now only supplies
        ``agent_role``, the one thing that distinguishes it from
        ``SpeculativeMergeWorker``'s equivalent method.
        """
        file_unattributed_landing_escalation(
            self._escalation_queue, task_id, branch, verdict,
            agent_role='harness-reconcile',
        )

    async def _stop_escalation_server(self) -> None:
        """Stop the escalation server."""
        if self._escalation_task is not None:
            self._escalation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._escalation_task
            self._escalation_task = None
            logger.info('Escalation server stopped')

    async def _reap_orphan_l0_escalations(self) -> int:
        """Single pass: promote any overdue orphan L0 to L1.  Returns count.

        Extracted from the loop so tests can drive it deterministically.
        An escalation is an orphan when its ``task_id`` is not in
        ``_escalation_events`` (no running workflow) AND the scheduler does
        not show it actively held (task 2878 — see the live-recheck note
        below), and it is older than ``orphan_l0_timeout_secs``.

        Async (task 2725): the done-step-commit orphan class needs to
        ``await self.scheduler.get_task(...)`` to check whether its
        subject task is terminal+merged (rebase-superseded false
        positive) before promoting.
        """
        if self._escalation_queue is None:
            return 0

        from escalation.models import Escalation

        timeout = self.config.orphan_l0_timeout_secs
        now = datetime.now(UTC)
        promoted = 0

        for esc in self._escalation_queue.get_pending():
            if esc.level != 0:
                continue
            if esc.task_id in self._escalation_events:
                continue  # active workflow will handle it
            # Task 2878: _escalation_events is a stale sweep-start snapshot
            # — it's populated at dispatch and popped by the workflow
            # slot's done-callback, so a task's id can vanish from it (slot
            # rotation, a coordinated batch-redispatch tick) while a live
            # or re-dispatched workflow is still actively holding exactly
            # this task's metadata.files locks. That race produced false
            # promotions for the plan.files/metadata.files divergence class
            # (filed by TaskWorkflow._check_scope_invariant during genuine
            # in-flight scope-reconciliation lag), which watchers then
            # verified live and closed benign. Re-check live scheduler
            # state at flag time via the same liveness signal the watchers
            # use, and defer (not drop) promotion while it's live — the
            # next sweep re-checks and promotes once genuinely idle.
            if self.scheduler.is_actively_held(esc.task_id):
                continue
            try:
                age_secs = (now - datetime.fromisoformat(esc.timestamp)).total_seconds()
            except (ValueError, TypeError):
                continue
            if age_secs < timeout:
                continue

            # Defense-in-depth: never double-escalate a task a human is
            # already looking at.  B1 (commit 1a1eca9a67) stopped the main
            # orphan source by skipping L0 creation on escalate_to_human
            # paths, but other L0 sources remain (agent escalate_info,
            # deep-reviewer L0s, steward-chained L0s) and an /unblock can be
            # in flight.  If an L1 is already open for this task, dismiss the
            # orphan L0 rather than promoting it to a duplicate "echo" L1.
            if self._escalation_queue.has_open_l1(esc.task_id):
                self._escalation_queue.resolve(
                    esc.id,
                    (
                        'Dismissed by orphan reaper — open L1 already covers '
                        f'this task (task_id={esc.task_id})'
                    ),
                    dismiss=True,
                    resolved_by='harness-orphan-reaper',
                )
                continue

            # Live lock-free-stage race (task 2931): defer — don't promote —
            # a plan.files/metadata.files divergence orphan whose task is
            # live inside a lock-free reviewer_comprehensive /
            # resettled_adjudicator stage. Those stages hold no module locks
            # and are absent from _dispatched, so the is_actively_held gate
            # above cannot see them — but they stamp
            # metadata.routing.latest.decided_at fresh per LLM invocation. A
            # decided_at within orphan_l0_dispatch_freshness_secs of this
            # sweep's `now` means the task is live mid-dispatch and its
            # divergence (metadata.files legitimately lagging plan.files) is
            # self-healing, not stranded: defer, and the next sweep re-checks
            # and promotes once the decision ages out. A genuinely stranded
            # task has stale/absent routing.latest -> _has_fresh_dispatch
            # False -> still promoted (preserves task 2878's boundary guard).
            # Placed after the age check so only aged-out divergence orphans
            # pay the get_task cost; the divergence and done-step-commit
            # classes are mutually exclusive, so at most one get_task fires.
            if _is_scope_divergence_orphan(esc):
                task = await self.scheduler.get_task(esc.task_id)
                if _has_fresh_dispatch(
                    task, now, self.config.orphan_l0_dispatch_freshness_secs,
                ):
                    logger.info(
                        'Orphan L0 reaper: deferred divergence orphan for '
                        'task_id=%s — fresh routing.latest dispatch within '
                        '%.0fs grace (live lock-free reviewer/adjudicator '
                        'stage); next sweep re-checks',
                        esc.task_id,
                        self.config.orphan_l0_dispatch_freshness_secs,
                    )
                    continue

            # Rebase-superseded false positive (task 2725): a done-step-commit
            # orphan (_is_done_step_commit_orphan) whose subject task is done
            # is a false positive — the step's recorded commit is a
            # pre-rebase intermediate no longer reachable from main, but its
            # content landed on main under a new SHA via the merge. Dismiss
            # rather than promote a duplicate manual-triage L1.
            if _is_done_step_commit_orphan(esc):
                task = await self.scheduler.get_task(esc.task_id)
                if _is_terminal_merged(task):
                    self._escalation_queue.resolve(
                        esc.id,
                        (
                            'Dismissed by orphan reaper — done-step-commit '
                            f'orphan for task_id={esc.task_id} is '
                            'rebase-superseded: subject task is done, so '
                            'the step content landed on main under a new '
                            'SHA via the merge'
                        ),
                        dismiss=True,
                        resolved_by='harness-orphan-reaper',
                    )
                    logger.info(
                        'Orphan L0 reaper: dismissed rebase-superseded '
                        'done-step-commit orphan for task_id=%s (subject '
                        'task is done)',
                        esc.task_id,
                    )
                    continue

            # Cite a durable branch ref instead of the originating worktree:
            # the orphan's worktree is ephemeral and likely reaped before a
            # human reads the promoted L1 (see workflow._durable_ref_suffix).
            branch = f'{self.config.git.branch_prefix}{esc.task_id}'
            reesc = Escalation(
                id=self._escalation_queue.make_id(esc.task_id),
                task_id=esc.task_id,
                agent_role='harness-orphan-reaper',
                severity=esc.severity,
                category=esc.category,
                summary=(
                    f'Orphan L0 ({age_secs:.0f}s old, no active workflow): '
                    f'{esc.summary}'
                ),
                detail=(
                    (esc.detail or '')
                    + f'\n\n[note] originating worktree may be reaped; '
                    f'branch={branch}'
                ),
                suggested_action='manual_intervention',
                worktree=None,
                workflow_state=esc.workflow_state,
                level=1,
            )
            self._escalation_queue.submit(reesc)
            self._escalation_queue.resolve(
                esc.id,
                (
                    'Auto-promoted to level 1 — orphan L0 (no active '
                    f'workflow for task_id={esc.task_id})'
                ),
                dismiss=True,
                resolved_by='harness-orphan-reaper',
            )
            promoted += 1

        if promoted:
            logger.warning(
                'Orphan L0 reaper: promoted %d escalation(s) to L1', promoted,
            )
        return promoted

    def cancel_workflow(self, task_id: str) -> bool:
        """Soft-cancel an active workflow.

        Sets the per-task ``cancel_event`` so any long await inside the
        workflow (merge-queue future, steward grace period) wakes promptly.
        The workflow re-reads task status and exits ``DONE`` if terminal,
        ``REQUEUED`` otherwise.

        Returns ``True`` iff a workflow was active for ``task_id`` (i.e. the
        cancel event was registered).  Returns ``False`` when the task has
        no slot — the call is a no-op and the caller can avoid waiting.
        """
        event = self._workflow_cancel_events.get(task_id)
        if event is None:
            return False
        event.set()
        # Stamp wall-clock so the reconcile sweep can grace-skip this task
        # for the next _RECONCILE_CANCEL_GRACE_S seconds — the workflow's
        # finally: block may still be writing state (R3 race guard).
        self.scheduler.note_workflow_cancelled(task_id)
        return True

    def is_workflow_active(self, task_id: str) -> bool:
        """True iff a workflow slot is currently active for ``task_id``."""
        return task_id in self._workflow_cancel_events

    def hard_cancel_workflow(self, task_id: str, *, restamp: bool = True) -> bool:
        """Hard-cancel the asyncio.Task running the workflow slot for ``task_id``.

        This is the escalation path when a workflow ignores the soft
        ``cancel_event`` (set by ``cancel_workflow``) for too long.
        Requesting asyncio.Task.cancel() forces a ``CancelledError`` into the
        coroutine's next await point; the ``_run_slot`` finally block still
        runs (CancelledError is BaseException, so it bypasses the
        ``except Exception`` guard at harness.py:1833) ensuring lock release
        and registry cleanup.

        ``restamp`` controls whether ``_workflow_cancel_at`` is updated.  Pass
        ``restamp=True`` (the default) on the threshold-crossing call so the R3
        reconcile grace window is anchored to the hard-cancel moment; pass
        ``restamp=False`` on subsequent polls to avoid indefinitely extending
        the grace window past the original hard-cancel timestamp.

        Returns ``True`` iff a live (non-done) slot task was found and
        ``cancel()`` was requested.  Returns ``False`` when there is no
        registered slot task or it is already done — the call is a no-op and
        the caller should treat this as a no-op.
        """
        task = self._workflow_slot_tasks.get(task_id)
        if task is None or task.done():
            return False
        if restamp:
            # Stamp wall-clock so the reconcile sweep respects the R3 grace
            # window.  Only stamp at the threshold-crossing call; subsequent
            # polls pass restamp=False to keep the window anchored.
            self.scheduler.note_workflow_cancelled(task_id)
        task.cancel()
        return True

    # ------------------------------------------------------------------
    # Escalation-watcher-auto subprocess supervisor (task 1326)
    # ------------------------------------------------------------------

    def _start_watcher_supervisor(self) -> None:
        """Start the escalation-watcher-auto subprocess supervisor.

        No-op when config.watcher_supervisor_enabled is False.
        Idempotent: does nothing if the task is already alive.
        """
        if not self.config.watcher_supervisor_enabled:
            # Supervisor permanently disabled — file an outage L2 so the
            # human L2 stream is notified (idempotent: dedup guard no-ops if
            # one is already open, so repeated restarts produce exactly one L2).
            #
            # Intentional-disable note: when the supervisor is deliberately
            # disabled via config, this L2 will remain open indefinitely because
            # the only auto-resolver (_resolve_watcher_outage_l2) runs from the
            # healthy-clean branch of _watcher_supervisor_loop, which never
            # executes while the supervisor is disabled.  This is expected
            # behaviour — the operator should manually resolve/dismiss the L2
            # once they have verified the disable is intentional.  The 'urgent'
            # severity is intentional: no autonomous L1 triage is occurring.
            self._file_watcher_outage_l2('watcher_supervisor_enabled=false')
            return
        if (
            self._watcher_supervisor_task is not None
            and not self._watcher_supervisor_task.done()
        ):
            return
        self._watcher_supervisor_task = asyncio.create_task(
            self._watcher_supervisor_loop(),
            name='watcher-supervisor',
        )
        logger.info(
            'Escalation-watcher-auto supervisor started '
            '(rotation_escalations=%d rotation_hours=%.1f)',
            self.config.watcher_rotation_escalations,
            self.config.watcher_rotation_hours,
        )

    async def _stop_watcher_supervisor(self) -> None:
        """Cancel the watcher supervisor loop."""
        if self._watcher_supervisor_task is not None:
            self._watcher_supervisor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._watcher_supervisor_task
            self._watcher_supervisor_task = None
            logger.info('Escalation-watcher-auto supervisor stopped')

    async def _run_watcher_rotation(self):  # type: ignore[return]
        """Run one escalation-watcher-auto rotation via invoke_with_cap_retry.

        Extracted for deterministic unit testing (same rationale as
        _scan_for_terminal_active_tasks).  Returns the AgentResult from
        invoke_with_cap_retry.

        The agent is instructed to exit cleanly after ROTATION_ESCALATIONS or
        ROTATION_HOURS — whichever fires first — and to emit its digest as its
        final message.  The supervisor-side timeout is rotation_hours*3600 +
        _WATCHER_TIMEOUT_GRACE_SECS so that a wedged agent that ignores its
        own rotation instructions is force-killed (classified as unclean, feeds
        the crashloop guard).
        """

        cfg = self.config
        user_prompt = (
            f'You are running as an autonomous escalation watcher.\n'
            f'Rotation limits (injected by supervisor):\n'
            f'  ROTATION_ESCALATIONS={cfg.watcher_rotation_escalations}\n'
            f'  ROTATION_HOURS={cfg.watcher_rotation_hours}\n'
            f'\n'
            f'When you have handled ROTATION_ESCALATIONS escalations '
            f'or {cfg.watcher_rotation_hours} hours have elapsed since startup, '
            f'emit your digest as the final message and exit cleanly.\n'
            f'\n'
            f'Project root: {cfg.project_root}\n'
            f'Escalation queue: {cfg.project_root}/{cfg.escalation.queue_dir}\n'
        )
        system_prompt = load_skill_system_prompt('escalation-watcher-auto')
        escalation_url = f'http://{cfg.escalation.host}:{cfg.escalation.port}/mcp'
        mcp_config = self.mcp.mcp_config_json(
            escalation_url=escalation_url,
            escalation_headers=_WATCHER_ESCALATION_HEADERS,
        )
        timeout_secs = cfg.watcher_rotation_hours * 3600 + _WATCHER_TIMEOUT_GRACE_SECS
        bash_max_timeout_ms = str(int(timeout_secs * 1000))
        logger.info(
            'Escalation-watcher-auto rotation: injecting BASH_MAX_TIMEOUT_MS=%s (timeout_secs=%.0f)',
            bash_max_timeout_ms,
            timeout_secs,
        )

        return await invoke_with_cap_retry(
            self.usage_gate,
            'Escalation watcher (auto)',
            invoke_fn=invoke_agent,
            cost_store=self.cost_store,
            run_id=self._run_id or '',
            project_id=cfg.fused_memory.project_id,
            role='escalation-watcher-auto',
            task_id='',
            prompt=user_prompt,
            system_prompt=system_prompt,
            cwd=cfg.project_root,
            model=cfg.watcher_model,
            max_turns=cfg.watcher_max_turns,
            max_budget_usd=cfg.watcher_rotation_budget_usd,
            effort=cfg.watcher_effort,
            backend=cfg.watcher_backend,
            mcp_config=mcp_config,
            timeout_seconds=timeout_secs,
            env_overrides={'BASH_MAX_TIMEOUT_MS': bash_max_timeout_ms},
            allowed_tools=_WATCHER_ALLOWED_TOOLS,
            disallowed_tools=_WATCHER_DISALLOWED_TOOLS,
            # Isolate this rotation's capped `escalation` connection: its server
            # name + URL are IDENTICAL to the interactive header-less block, so
            # under the non-strict ambient .mcp.json merge the capped block can
            # bleed into a concurrent interactive session (task 2796, THREAD 2).
            # --strict-mcp-config scopes the invocation to only `mcp_config`'s
            # servers, which are complete for this read-only RCA/triage job.
            strict_mcp_config=True,
        )

    def _watcher_has_actionable_l1(self) -> bool:
        """Pre-boot precheck: does the L1 queue have any actionable work?

        "Actionable" means at least one pending level-1 escalation whose id
        is NOT already a member of a pending level-2 cluster.  Promoted
        member L1s remain ``status=='pending'`` at level 1 (SKILL.md), so a
        naive ``level == 1 and pending`` check would relaunch a rotation
        every poll interval for as long as any L2 cluster has unresolved
        members — a common steady state while a human is slow to resolve
        L2s — defeating the entire cost optimisation this precheck exists
        for.

        Scope: only L1 work counts.  A queue containing only L0s, or only
        pending L2s, is treated as non-actionable — L0->L1 promotion is
        owned by the separate ``_reap_orphan_l0_escalations`` loop, so
        skipping here does not starve that promotion path.

        FAIL-OPEN: returns True (launch the rotation) whenever the
        escalation queue is unset/None or ``get_pending()`` raises — a bug
        in this precheck must never silently stop L1 escalations from being
        handled (loud-over-silent-degradation norm).  Accessed via
        ``getattr`` so bare-Harness tests that never set
        ``_escalation_queue`` also fail open.
        """
        queue = getattr(self, '_escalation_queue', None)
        if queue is None:
            return True
        try:
            pending = queue.get_pending()
        except Exception:
            logger.warning(
                'Escalation-watcher-auto: _watcher_has_actionable_l1 precheck '
                'failed to read the escalation queue — failing open '
                '(launching rotation anyway)',
                exc_info=True,
            )
            return True

        l1_ids = [esc.id for esc in pending if esc.level == 1]
        if not l1_ids:
            return False

        promoted: set[str] = set()
        for esc in pending:
            if esc.level == 2:
                promoted.update(esc.members or [])

        return any(l1_id not in promoted for l1_id in l1_ids)

    async def _watcher_supervisor_loop(self) -> None:
        """Supervisor loop — restart watcher rotations until shutdown.

        Classification:
          clean exit (success=True and not timed_out) → immediate restart,
              reset consecutive-unclean counter.
          unclean exit (success=False OR timed_out=True) → record timestamp,
              apply exponential backoff, check crashloop guard (step-12).
          asyncio.CancelledError → re-raise (clean shutdown signal).
          generic Exception → treated as unclean (logged via logger.exception).

        Crashloop detection is added in step-12.
        """
        consecutive_unclean: int = 0
        consecutive_degenerate_clean: int = 0  # task 1430: exponential floor on fast-clean exits
        while True:
            # task 2629: pre-boot empty-queue precheck.  Skip the (expensive)
            # rotation launch entirely when the L1 queue has no actionable
            # work; fails open (see _watcher_has_actionable_l1) so a precheck
            # bug can never silently stop real L1 handling.  Deliberately
            # placed before `start = time.monotonic()` and does not touch the
            # clean/unclean/degenerate counters, the guards, or
            # _maybe_write_digest — this is a pure pre-boot bypass, not a
            # rotation outcome.
            if not self._watcher_has_actionable_l1():
                poll = self.config.watcher_empty_queue_poll_secs
                logger.debug(
                    'Escalation-watcher-auto: L1 queue has no actionable work; '
                    'skipping rotation launch (poll=%.1fs)', poll,
                )
                # task 2629 review fix: reaching this branch proves the
                # supervisor loop is alive and the queue was readable (the
                # precheck fails open on a None/erroring queue), which is
                # exactly the "watcher is up again" signal a stale
                # watcher-outage L2 is waiting on — resolve it here so it
                # never lingers as a false alarm just because the L1 queue
                # happens to be drained. Best-effort/no-op when none is open.
                self._resolve_watcher_outage_l2()
                try:
                    await asyncio.sleep(poll)
                except asyncio.CancelledError:
                    raise  # clean shutdown
                continue

            start = time.monotonic()
            try:
                result = await self._run_watcher_rotation()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    'Escalation-watcher-auto rotation raised unexpected exception'
                )
                # Treat as unclean: backoff below
                result = None  # sentinel for unclean path
            end = time.monotonic()  # captured after rotation; reused as 'now' below

            clean = (
                result is not None
                and bool(getattr(result, 'success', False))
                and not bool(getattr(result, 'timed_out', False))
            )

            # Best-effort digest check after each rotation (task 1327).
            # Single call site — applies to both clean and unclean paths.
            # Never allowed to break the supervisor: CancelledError re-raised,
            # AttributeError re-raised (task 1449: surfaces state-init drift to
            # tests rather than converting it to a silent warning log),
            # every other runtime exception logged and swallowed.
            try:
                await self._maybe_write_digest()
            except asyncio.CancelledError:
                raise
            except AttributeError:
                raise  # task 1449: surface fixture-drift / state-init bugs to tests
            except Exception:
                logger.warning(
                    '_maybe_write_digest raised unexpectedly in supervisor loop '
                    '(best-effort swallowed)',
                    exc_info=True,
                )

            if clean:
                # Healthy rotation completed — reset backoff.
                # Enforce a minimum floor between rotations even on clean exit:
                # prevents back-to-back opus invocations if the agent self-exits
                # near-instantly due to misconfiguration (empty queue, SKILL.md
                # drift, etc.).  Mirrors the terminal-status-watcher's always-sleep
                # pattern.  watcher_subprocess_restart_backoff_secs (default 30s)
                # doubles as the clean-restart floor.
                consecutive_unclean = 0
                # Cost-runaway guard: track degenerate-clean exits (task 1388).
                # A healthy rotation takes at least watcher_misconfigured_min_rotation_secs
                # seconds; one that exits faster is degenerate (empty queue, SKILL.md
                # drift, misconfigured env).
                duration = end - start
                # _check_watcher_guard: append+evict+trip, fully defensive.
                # Only called when duration is below the healthy-rotation floor;
                # reuses watcher_crashloop_window_secs as the burst-detection
                # window — semantically identical for both failure modes.
                if duration < self.config.watcher_misconfigured_min_rotation_secs:
                    # Increment before the guard call — mirrors the unclean path
                    # convention (consecutive_unclean += 1 before _check_watcher_guard).
                    consecutive_degenerate_clean += 1
                    if await self._check_watcher_guard(
                        self._watcher_degenerate_clean_exits,
                        'watcher_misconfigured',
                        self.config.watcher_max_misconfigured_clean_exits,
                        self.config.watcher_crashloop_window_secs,
                        end,
                    ):
                        return
                    # Degenerate-clean, no trip — apply exponential floor (task 1430).
                    # Fast rotation signals potential cost-runaway; grow the sleep
                    # exponentially to slow the burn rate before the trip arms.
                    # Mirrors the unclean-exit exponential backoff in the section below.
                    # Clamp exp to 60 (2**60 ≈ 1.15e18) to prevent OverflowError at very
                    # high consecutive counts; the outer min still bounds to _WATCHER_MAX_BACKOFF_SECS
                    # so observable behaviour is unchanged for any realistic consecutive value.
                    exp = min(consecutive_degenerate_clean - 1, 60)
                    floor = min(
                        self.config.watcher_subprocess_restart_backoff_secs * (2 ** exp),
                        _WATCHER_MAX_BACKOFF_SECS,
                    )
                    logger.warning(
                        'Escalation-watcher-auto rotation completed cleanly but fast '
                        '(consecutive_degenerate=%d floor=%.1fs)',
                        consecutive_degenerate_clean,
                        floor,
                    )
                    try:
                        await asyncio.sleep(floor)
                    except asyncio.CancelledError:
                        raise  # clean shutdown
                    except Exception:
                        logger.exception(
                            'Escalation-watcher-auto supervisor: unexpected error in '
                            'degenerate-clean-path floor sleep — sleeping base backoff '
                            'to avoid silent supervisor death'
                        )
                        try:
                            await asyncio.sleep(
                                self.config.watcher_subprocess_restart_backoff_secs
                            )
                        except asyncio.CancelledError:
                            raise
                else:
                    # Healthy clean (duration >= watcher_misconfigured_min_rotation_secs):
                    # positive evidence the queue had real work — reset the cost-runaway
                    # counter so the next degenerate burst starts fresh from base.
                    consecutive_degenerate_clean = 0
                    logger.info('Escalation-watcher-auto rotation completed cleanly; restarting')
                    # Resolve any open watcher-outage L2 — the watcher is running
                    # healthily again.  Degenerate-clean (fast exits) are NOT recovery
                    # signals; only healthy-clean (duration >= min) clears the outage.
                    self._resolve_watcher_outage_l2()
                    await asyncio.sleep(self.config.watcher_subprocess_restart_backoff_secs)
                continue

            # --- Unclean exit path ---
            # Symmetric with the clean path's ``consecutive_unclean = 0`` reset (which
            # fires on both healthy and degenerate-clean exits): an unclean exit breaks
            # any in-progress degenerate-clean burst, so reset the floor counter.
            # Without this, interleaved degen/unclean workloads grow
            # consecutive_degenerate_clean unboundedly across bursts (task 1443).
            consecutive_degenerate_clean = 0
            consecutive_unclean += 1
            # _check_watcher_guard handles append+evict+trip defensively.  Called
            # outside any broad try/except so pause_scheduler failure cannot be
            # swallowed by the backoff-degradation handler (S2 defect prevention).
            if await self._check_watcher_guard(
                self._watcher_unclean_exits,
                'watcher_crashloop',
                self.config.watcher_max_crashloop_restarts,
                self.config.watcher_crashloop_window_secs,
                end,
            ):
                return

            # No trip — apply exponential backoff.
            try:
                # Clamp exp to 60; see degenerate-clean path comment above re: overflow.
                exp = min(consecutive_unclean - 1, 60)
                backoff = min(
                    self.config.watcher_subprocess_restart_backoff_secs * (2 ** exp),
                    _WATCHER_MAX_BACKOFF_SECS,
                )
                logger.warning(
                    'Escalation-watcher-auto rotation exited uncleanly '
                    '(consecutive=%d backoff=%.1fs)',
                    consecutive_unclean,
                    backoff,
                )
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise  # clean shutdown
            except Exception:
                logger.exception(
                    'Escalation-watcher-auto supervisor: unexpected error in '
                    'unclean-path backoff — sleeping base backoff to avoid '
                    'silent supervisor death'
                )
                # Degrade gracefully: sleep base backoff then continue the loop.
                # Re-raise CancelledError from the fallback sleep if received.
                try:
                    await asyncio.sleep(
                        self.config.watcher_subprocess_restart_backoff_secs
                    )
                except asyncio.CancelledError:
                    raise

    async def _check_watcher_guard(
        self,
        exits_deque: deque[float],
        reason: str,
        max_count: int,
        window_secs: float,
        now: float,
    ) -> bool:
        """Append *now* to *exits_deque*, evict stale entries, trip if threshold met.

        Returns True  → guard tripped; caller should stop the supervisor.
        Returns False → threshold not reached or bookkeeping error; caller continues.

        Bookkeeping errors (deque ops) are logged and suppressed so a rare
        collection anomaly cannot silently kill the supervisor.

        pause_scheduler failure is also caught and logged, but the method
        still returns True so a broken pause_scheduler cannot defeat the stop.
        CancelledError is always re-raised (clean shutdown signal).
        """
        try:
            exits_deque.append(now)
            while exits_deque and exits_deque[0] < now - window_secs:
                exits_deque.popleft()
        except Exception:
            logger.exception(
                'watcher %s guard: bookkeeping error — skipping trip check',
                reason,
            )
            return False
        if len(exits_deque) < max_count:
            return False
        logger.error(
            'Escalation-watcher-auto %s guard tripped '
            '(%d exits in %ds window) — pausing scheduler',
            reason,
            len(exits_deque),
            window_secs,
        )
        try:
            await self.pause_scheduler(reason)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                'pause_scheduler raised on %s trip; stopping supervisor anyway',
                reason,
            )
        # File the watcher-outage L2 so the human L2 stream is notified even
        # when the watcher subsystem is not running.  No outer try/except needed
        # here: _file_watcher_outage_l2 is already best-effort (its body is
        # entirely wrapped in `except Exception`), so any filing failure is
        # swallowed internally.  CancelledError is not raised by the sync helper.
        self._file_watcher_outage_l2(reason)
        return True  # always stop, even if pause_scheduler raised

    async def _scan_for_terminal_active_tasks(self) -> int:
        """Single pass: cancel any active workflow whose task is terminal.

        Below ``config.terminal_status_hard_cancel_polls`` consecutive
        terminal polls the scan issues a soft cancel (``cancel_workflow``).
        At the threshold it escalates to a hard ``asyncio.Task.cancel()``
        (``hard_cancel_workflow``) and logs a WARNING exactly once at the
        crossing.  This stops the every-30s re-logging symptom: once hard-
        cancelled the workflow's ``finally`` block clears the registry entry
        so subsequent scans skip the task naturally.

        Counts are cleared when a task's status is no longer terminal so
        stale state cannot accumulate across transient status fluctuations.

        Returns the number of workflows on which a cancel action was taken.
        """
        active_ids = list(self._workflow_cancel_events.keys())
        if not active_ids:
            return 0
        statuses, error = await self.scheduler.get_statuses(active_ids)
        if error is not None:
            logger.warning(
                'Terminal-status watcher: get_statuses(%d active) failed: %s'
                ' — skipping scan this cycle',
                len(active_ids),
                error,
            )
            return 0

        threshold = self.config.terminal_status_hard_cancel_polls
        cancelled = 0
        for task_id, status in statuses.items():
            if status not in TERMINAL_STATUSES:
                # Status returned to non-terminal — clear stale count.
                self._terminal_cancel_counts.pop(task_id, None)
                continue

            # Status is terminal and the workflow slot is still active.
            count = self._terminal_cancel_counts.get(task_id, 0) + 1
            self._terminal_cancel_counts[task_id] = count

            if count < threshold:
                # Below threshold: soft cancel (set the asyncio.Event).
                if self.cancel_workflow(task_id):
                    logger.info(
                        'Terminal-status watcher: cancelling workflow for task '
                        '%s (status=%s, poll=%d/%d)',
                        task_id, status, count, threshold,
                    )
                    cancelled += 1
            else:
                # At or above threshold: escalate to hard asyncio.Task.cancel().
                # Only restamp _workflow_cancel_at at the threshold crossing so
                # the R3 grace window has a defined endpoint; subsequent polls
                # pass restamp=False to avoid extending it indefinitely.
                at_crossing = count == threshold
                result = self.hard_cancel_workflow(task_id, restamp=at_crossing)
                if at_crossing:
                    # Log the WARNING exactly once at the threshold crossing so
                    # a still-draining task is not re-warned every 30 s.
                    if result:
                        logger.warning(
                            'Terminal-status watcher: hard-cancelling workflow '
                            'for task %s — ignored soft cancel for %d polls '
                            '(status=%s)',
                            task_id, count, status,
                        )
                    else:
                        logger.warning(
                            'Terminal-status watcher: threshold reached but no '
                            'live slot task registered for task %s (status=%s)',
                            task_id, status,
                        )
                if result:
                    cancelled += 1

        return cancelled

    async def _run_main_tip_sweep(self) -> None:
        """Single testable pass of the main-tip integrity sweep.

        Resolves the current main SHA, runs a full unscoped verification against
        a throwaway detached worktree at that SHA, and files a level-1 infra_issue
        escalation when the sweep finds a failure on main.

        SHA dedup (``_last_swept_main_sha``) skips the expensive full verify
        when main has not advanced — N merges within one interval cost one sweep.
        When ``_escalation_queue`` is None the drift is logged but not submitted.
        Registered as a ``BackgroundService`` pass_fn (task 2241, W10-η) —
        see ``Harness._build_lifecycle_registry`` for the periodic invocation
        context.

        On a PASS (task 2114), also self-heals: calls
        ``_close_superseded_main_sweep_escalations`` with the just-verified
        clean SHA to auto-close any prior main-sweep escalation whose swept
        SHA this clean tip has superseded.  Never called on the failure path.

        Before filing, a drifting result is passed through
        ``verify.confirm_main_tip_failure_is_real`` (task 2370): a fresh
        isolated re-run of just the named failing tests at ``swept_sha`` that
        catches load-induced xdist flakes the same-worktree full-suite retry
        above cannot.  Confirmed-flake (``False``) suppresses the alarm
        without self-healing; confirmed-real (``True``) files it exactly as
        before.
        """
        from orchestrator import critical_gate  # noqa: PLC0415
        from orchestrator import verify as verify_mod  # noqa: PLC0415

        main_sha: str = await self.git_ops.get_main_sha()  # type: ignore[union-attr]
        if not main_sha:
            return

        # SHA dedup: skip the expensive full verify when main has not advanced.
        # This is the batching gate — N merges within one interval cost one sweep.
        if main_sha == self._last_swept_main_sha:
            return

        # Pass the already-resolved SHA so verify skips a second git rev-parse
        # and both the dedup gate and the worktree pin use the same value
        # (closes the TOCTOU window — suggestion 2 from the code review).
        outcome = await verify_mod.run_main_tip_sweep(
            self.config, self.git_ops, main_sha=main_sha
        )
        if outcome is None:
            # Infra failure in the sweep itself — retry next tick, don't mark swept.
            return

        swept_sha, vr = outcome
        self._last_swept_main_sha = swept_sha

        if vr.passed:
            await self._close_superseded_main_sweep_escalations(swept_sha)
            return

        # Drift detected: file one L1 escalation per distinct bad SHA.
        if not self._escalation_queue:
            logger.warning(
                'Main-tip integrity sweep: drift detected at %s (%s) but no escalation '
                'queue attached — skipping L1 file',
                swept_sha[:12], vr.category,
            )
            return

        from escalation.models import Escalation  # noqa: PLC0415

        task_id = f'main-sweep-{swept_sha[:12]}'
        # Dedup against a surviving L1 from a prior run: _last_swept_main_sha
        # resets to None on every restart, so without this guard a main that
        # stays broken would re-file a fresh duplicate each boot.  Matches the
        # convention in every other L1-filing path in this file
        # (_file_restored_pause_escalation, _mark_blocked, scheduler-pause, etc.).
        if self._escalation_queue.has_open_l1(task_id):
            return

        # Confirm-before-alarm gate (task 2370): the full-suite retry above ran
        # in the SAME contended worktree, so a load-induced xdist flake can
        # fail both passes and still reach here as "drift" — the false-alarm
        # source behind esc-main-sweep-ea2bd3c95e33-2 and the 2026-07-09
        # park_stop/symlink-loop incidents. Re-run just the named failing
        # tests, in isolation, in a FRESH probe worktree pinned at swept_sha.
        # A confirmed flake (False) is deliberately NOT self-healed: that
        # requires a genuine full-verify PASS (see vr.passed branch above),
        # and an isolated subset re-run is weaker evidence than that.
        subset_confirmed = await verify_mod.confirm_main_tip_failure_is_real(
            self.config, self.git_ops, vr, main_sha=swept_sha
        )

        # Current-tip re-confirmation arm (task 2558), COMPOSED with 2370's
        # subset re-run above — NOT a second full sweep.  The full verify +
        # subset re-run take minutes, during which main can advance past the
        # observed bad SHA; filing then would recommend a destructive
        # intervention against a SHA that is no longer the tip — the survey
        # §1.7 "last-green rewind named a commit that also failed / evidence
        # since mutated" precedent.  So re-resolve the current tip cheaply and
        # require it still equal swept_sha.  Filing requires BOTH the subset
        # confirm AND the tip being unchanged.  The default-on
        # main_tip_sweep_rerun_confirm_enabled kill-switch toggles ONLY this
        # tip arm; disabled forces tip_unchanged=True (byte-identical post-2370).
        if self.config.main_tip_sweep_rerun_confirm_enabled:
            current_sha = await self.git_ops.get_main_sha()  # type: ignore[union-attr]
            tip_unchanged = bool(current_sha) and current_sha == swept_sha
        else:
            tip_unchanged = True
        rerun_confirmed = subset_confirmed and tip_unchanged
        if not critical_gate.critical_filing_gate(rerun_confirmed=rerun_confirmed):
            # Not corroborated on the current tip: either the subset re-run
            # flaked (2370) OR main advanced past the observed bad SHA (2558).
            # Suppress — a stale/transient red must not fire a destructive alarm.
            logger.info(
                'Main-tip integrity sweep: failure at %s unconfirmed on current tip '
                '(subset_confirmed=%s tip_unchanged=%s) — not filing (stale/transient)',
                swept_sha[:12], subset_confirmed, tip_unchanged,
            )
            return

        summary = f'Main-tip integrity sweep failed at {swept_sha[:12]}: {vr.category}'[:200]
        detail = (
            f'Full verification of main at {swept_sha} failed.\n'
            f'Category: {vr.category}\n'
            f'Cause: {vr.cause_hint or "(no hint)"}\n'
            f'Summary: {vr.summary}'
        )
        esc = Escalation(
            id=self._escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-main-sweep',
            severity='blocking',
            category='infra_issue',
            summary=summary,
            detail=detail,
            suggested_action='manual_intervention',
            level=1,
        )
        self._escalation_queue.submit(esc)
        logger.warning(
            'Main-tip integrity sweep: filed L1 escalation %s for SHA %s (%s)',
            esc.id, swept_sha[:12], vr.category,
        )

    async def _close_superseded_main_sweep_escalations(self, clean_sha: str) -> None:
        """Self-heal: close any main-sweep escalation superseded by a clean PASS (task 2114).

        ``_run_main_tip_sweep`` never closed its own stale failing escalation
        once a LATER pass verified main clean — this is the pile-up gap.  The
        only fail-safe positive evidence that the cited defect is actually
        fixed is a FRESH full-verify PASS at a SHA that is a STRICT
        descendant of the failed swept SHA (main advanced and now verifies
        clean).  A bare ``is_ancestor(cited_sha, main)`` is unsafe: the
        failing SHA is itself an ancestor of an un-advanced (still-broken)
        main, and ``git merge-base --is-ancestor`` is reflexive (a commit is
        trivially an ancestor of itself) — so an escalation whose swept SHA
        equals *clean_sha* at 12-hex precision is explicitly guarded rather
        than auto-closed, even though ``is_ancestor`` would return True.

        Enumerates every pending escalation and closes (``close_only``, via
        ``resolve(dismiss=True)`` — never resumed/re-pended) each
        ``orchestrator-main-sweep`` one whose swept SHA — parsed from its
        ``main-sweep-<sha12>`` task_id — is a strict ancestor of *clean_sha*.
        Level-agnostic (acts on an L1 or an auto-watcher-promoted L2 alike),
        since ``_run_main_tip_sweep`` files at L1 but the auto-watcher may
        have since promoted it.

        Fail-soft at two layers: a method-level try/except around the whole
        enumeration (an unconfigured/erroring ``get_pending()`` degrades to a
        no-op) and a per-item try/except inside the loop (one bad escalation
        — e.g. a git_ops failure — never blocks the rest of the pass).  This
        is what keeps the existing main-tip-sweep suite's bare-MagicMock
        harness inert: called only from the ``vr.passed`` branch, after
        ``_last_swept_main_sha`` has already been updated.
        """
        if not self.config.escalation_revalidation_enabled:
            return
        if self._escalation_queue is None:
            return
        if not clean_sha:
            return

        try:
            for esc in self._escalation_queue.get_pending():
                try:
                    if esc.agent_role != 'orchestrator-main-sweep' or esc.status != 'pending':
                        continue
                    swept = esc.task_id.removeprefix('main-sweep-')
                    if not swept or swept == esc.task_id:
                        continue
                    if swept == clean_sha[:len(swept)]:
                        # Equal-sha guard: is_ancestor(X, X) is trivially True
                        # (reflexive) — a still-broken main re-verified at the
                        # exact same commit must not be treated as fixed.
                        continue
                    if not await self.git_ops.is_ancestor(swept, clean_sha):  # type: ignore[union-attr]
                        continue

                    resolution = (
                        f'main-tip-sweep self-heal: main advanced to {clean_sha[:12]} '
                        f'which verifies clean; swept SHA {swept} failure is '
                        f'superseded — auto-closed (close_only).'
                    )
                    self._escalation_queue.resolve(
                        esc.id, resolution, dismiss=True,
                        resolved_by=_MAIN_TIP_SWEEP_SELFHEAL_ROLE,
                        resolution_class='benign',
                    )
                    if self.event_store:
                        self.event_store.emit(
                            EventType.escalation_resolved,
                            task_id=esc.task_id,
                            data={
                                'escalation_id': esc.id,
                                'swept_sha': swept,
                                'clean_sha': clean_sha[:12],
                                'reason': 'main-tip-sweep-superseded',
                            },
                        )
                    logger.info(
                        'Main-tip-sweep self-heal: closed %s — swept SHA %s '
                        'superseded by clean tip %s',
                        esc.id, swept, clean_sha[:12],
                    )
                except Exception as exc:
                    logger.error(
                        'Main-tip-sweep self-heal: failed for escalation %s: %s: %s',
                        getattr(esc, 'id', '?'), type(exc).__name__, exc,
                    )
        except Exception as exc:
            logger.error(
                'Main-tip-sweep self-heal: enumeration failed: %s: %s',
                type(exc).__name__, exc,
            )

    # ------------------------------------------------------------------
    # Deterministic-strand reconciliation sweep (task 2074)
    # ------------------------------------------------------------------

    async def _revalidate_deterministic_deploy_health(self, metadata: dict) -> str:
        """Re-validate a deterministic deploy's target unit against live systemd state.

        Reads ``metadata['before_done']['target_unit']``; returns 'unconfirmed'
        WITHOUT invoking the inspector when before_done/target_unit is absent
        (nothing to inspect).  Otherwise awaits the injected
        ``self._recon_unit_inspector`` (falling back to the module-level
        ``_recon_inspect_unit`` default) and classifies the result via
        ``_deterministic_deploy_health_verdict`` — see that function's
        docstring for the important CAVEAT: without a persisted baseline,
        'healthy' is a liveness signal, not proof that *this* deploy is what
        brought the unit up, and for an always-on service unit it is
        near-constant 'healthy'.

        ζ/task 2240 (DS-3): also reads ``metadata['deploy_state'].verify_baseline``
        (via ``DeployState.from_metadata``, ``None`` when absent — e.g. a
        legacy pre-ζ deploy) and threads it through to the verdict fn,
        upgrading the check to real freshness whenever a baseline was
        persisted.
        """
        before_done = (metadata or {}).get('before_done')
        target_unit = before_done.get('target_unit') if isinstance(before_done, dict) else None
        if not target_unit:
            return 'unconfirmed'
        inspect_fn = self._recon_unit_inspector or _recon_inspect_unit
        result = await inspect_fn(target_unit)
        state = DeployState.from_metadata(metadata or {})
        verify_baseline = state.verify_baseline if state is not None else None
        return _deterministic_deploy_health_verdict(result, verify_baseline=verify_baseline)

    async def _recover_stranded_deterministic_task(
        self, tid: str, task: dict, metadata: dict,
    ) -> None:
        """Recover a task-2059-shaped stranded deterministic task (Source A).

        Dedup-guarded: skips (logging) when a pending escalation already
        exists for *tid* — self-dedupes across sweep passes once filed.
        Re-validates live systemd health for the deploy's target unit and
        RE-FILES a single L1 escalation — this method NEVER calls
        ``set_task_status`` (RE-FILE-NEVER-FLIP discipline, mirroring the
        stranded-blocked-reaper backstop):

          - ``healthy``     -> category='stranded_blocked', suggested_action='resume'.
            The existing auto-resume watcher resolves this, driving
            blocked->pending->re-dispatch->DeterministicRunner's resume path
            (case-(b): "escalation resolved -> resume, no re-run").
          - ``unconfirmed`` -> category='infra_issue', suggested_action='manual_intervention'.
            A human must inspect the unit before the task can be resumed.
        """
        if self._escalation_queue is None:
            return
        if self._escalation_queue.get_by_task(tid, status='pending'):
            logger.info(
                'Deterministic-recon-sweep: task %s already has a pending '
                'escalation — skipping strand recovery (dedup)',
                tid,
            )
            return

        verdict = await self._revalidate_deterministic_deploy_health(metadata)
        before_done = metadata.get('before_done') or {}
        target_unit = before_done.get('target_unit', 'unknown')

        if verdict == 'healthy':
            category = 'stranded_blocked'
            suggested_action = 'resume'
            guidance = (
                f'Live unit {target_unit} is healthy (MainPID>0, ActiveState=active) '
                f'— the deploy demonstrably succeeded and only the post-deploy '
                f'writeback failed.  Resolve this L1 (suggested_action=resume) to '
                f're-pend the task; the existing auto-resume watcher and '
                f'DeterministicRunner resume path drive it to done with NO re-run.'
            )
        else:
            category = 'infra_issue'
            suggested_action = 'manual_intervention'
            guidance = (
                f'Live unit {target_unit} state could not be confirmed healthy '
                f'(unit down/failed/unknown) — a human must inspect the unit '
                f'before this task can be safely resumed or re-triaged.'
            )

        from escalation.models import Escalation  # noqa: PLC0415

        description = (task or {}).get('description', '')
        summary = (
            f'Deterministic strand recovery: task {tid} blocked with '
            f'before_done_ran_at stamped and no open escalation — target unit '
            f'{target_unit} is {verdict}.'
        )[:200]
        detail = '\n\n'.join(filter(None, [
            description,
            f'Target unit: {target_unit}',
            f'Live health verdict: {verdict}',
            guidance,
        ]))

        esc = Escalation(
            id=self._escalation_queue.make_id(tid),
            task_id=tid,
            agent_role='harness-deterministic-recon-sweep',
            severity='blocking',
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            level=1,
        )
        self._escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=tid,
                data={
                    'escalation_id': esc.id,
                    'category': category,
                    'severity': 'blocking',
                    'level': 1,
                    'reason': 'deterministic-recon-sweep-strand-recovery',
                },
            )
        logger.warning(
            'Deterministic-recon-sweep: task %s stranded (verdict=%s) — filed '
            'L1 %s (category=%s, suggested_action=%s, no status change)',
            tid, verdict, esc.id, category, suggested_action,
        )

    async def _recover_stranded_deterministic_gate(
        self, tid: str, task: dict, metadata: dict,
    ) -> None:
        """Recover a stranded deterministic pure-gate / always_escalates GATE (Source A).

        Task 2954: the GATE-strand sibling of
        ``_recover_stranded_deterministic_task``.  A ``task_kind=='deterministic'``
        task stamped ``gate_escalated_at`` (proof a born-at-L2 ``milestone_gate``
        was supposed to be filed by
        ``DeterministicRunner._file_milestone_gate_and_block``) but its
        escalation record never landed — lost across a merge-triggered restart
        or a queue_dir storage/scoping divergence (the failure mode named in
        ``EscalationQueue.submit``'s docstring).  RE-FILES the born-at-L2 gate
        mirroring what the runner itself would have filed (same agent_role,
        level, severity, category, and summary/detail/options — the latter three
        built through the SHARED ``build_milestone_gate_escalation_fields`` seam
        the runner uses, so even the operational-LLM token prefix (task 2803 γ)
        on an operational-mode gate is reproduced verbatim rather than dropped
        and misrouted) so the runner's section-1 resume quiescence/resolve-to-
        done machinery — which scopes its scans on ``DETERMINISTIC_AGENT_ROLE``
        — integrates cleanly: a human resolving the re-filed gate drives the
        task to done exactly as designed.

        Unlike ``_recover_stranded_deterministic_task`` this is a HUMAN-decision
        gate: there is no target unit, no live systemd health check, and — like
        that method — it NEVER calls ``set_task_status`` (RE-FILE-NEVER-FLIP;
        the subject task is already blocked).  The archive-inclusive
        role-scoped emptiness check that discriminates a genuine strand from a
        filed+resolved gate is the caller's job
        (``_run_deterministic_recon_sweep`` Source A), which also self-dedupes
        across passes.
        """
        if self._escalation_queue is None:
            return

        from escalation.models import Escalation  # noqa: PLC0415

        # Task 2954 amendment: build summary/detail/options via the SAME
        # `build_milestone_gate_escalation_fields` seam the runner's own
        # `_file_milestone_gate_and_block` uses, so the re-filed gate is
        # byte-identical to what the runner would have filed — including the
        # operational-LLM token prefix (task 2803 γ) that a hand-rolled
        # re-build here would otherwise drop and misroute.
        summary, detail, options = build_milestone_gate_escalation_fields(
            task, metadata,
        )

        esc = Escalation(
            id=self._escalation_queue.make_id(tid),
            task_id=tid,
            agent_role=DETERMINISTIC_AGENT_ROLE,
            severity='critical',
            category='milestone_gate',
            summary=summary,
            detail=detail,
            options=options,
            level=2,
        )
        self._escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=tid,
                data={
                    'escalation_id': esc.id,
                    'category': 'milestone_gate',
                    'severity': 'critical',
                    'level': 2,
                    'reason': 'deterministic-recon-sweep-gate-strand-recovery',
                },
            )
        logger.warning(
            'Deterministic-recon-sweep: task %s pure-gate strand '
            '(gate_escalated_at stamped, no escalation record) — re-filed L2 '
            'milestone_gate %s (agent_role=%s, no status change)',
            tid, esc.id, DETERMINISTIC_AGENT_ROLE,
        )

    async def _revalidate_open_deterministic_escalation(
        self, esc, task: dict, metadata: dict,
    ) -> None:
        """Re-validate an OPEN deterministic-deploy ``infra_issue`` escalation (Source B).

        Only touches an escalation that is ALL of:
          - ``category == 'infra_issue'``
          - ``agent_role`` in ``_DETERMINISTIC_ESCALATION_SENTINEL_ROLES``
            (the runner's own sentinel role, or this sweep's own role so a
            previously-filed 'unconfirmed' L1 can self-heal)
          - filed against a task matching the stranded-deterministic metadata shape

        ``milestone_gate`` (human-decision) escalations and any non-matching
        role/category are NEVER touched — checked cheaply before the live
        unit-inspector call.  When live state now shows 'healthy' —
        contradicting the escalation's stated "deploy could not be verified"
        facts — resolves it.  Resolution fires the harness resolve-callback,
        driving the existing resume -> DeterministicRunner no-re-run path.
        Unhealthy verdicts are left open (untouched).
        """
        if self._escalation_queue is None:
            return
        if (
            esc.category != 'infra_issue'
            or esc.agent_role not in _DETERMINISTIC_ESCALATION_SENTINEL_ROLES
        ):
            return
        if not _deterministic_deploy_stranded(metadata):
            return

        verdict = await self._revalidate_deterministic_deploy_health(metadata)
        if verdict != 'healthy':
            return

        before_done = metadata.get('before_done') or {}
        target_unit = before_done.get('target_unit', 'unknown')
        resolution = (
            f'deterministic-recon-sweep: live unit {target_unit} healthy '
            f'(MainPID>0, ActiveState=active) — deploy verified post-hoc; '
            f'stated failure is stale.'
        )
        self._escalation_queue.resolve(
            esc.id, resolution, resolved_by='harness-deterministic-recon-sweep',
            resolution_class='benign',
        )
        logger.info(
            'Deterministic-recon-sweep: re-validated escalation %s for task %s '
            '— live unit %s healthy, auto-resolved (stale failure)',
            esc.id, esc.task_id, target_unit,
        )

    async def _revalidate_open_l2(self, esc, statuses: dict[str, str]) -> bool:
        """Close an open L2 escalation whose SUBJECT task has gone terminal (task 2114).

        Generalizes task 2074's revalidation machinery beyond the
        blocked-deterministic subset: ANY pending level-2 escalation whose
        subject task status (read from a batch ``scheduler.get_statuses``
        call by the caller) is ``done`` or ``cancelled`` is moot — the human
        decision it was waiting on no longer matters — and is auto-closed via
        ``close_only`` (dismiss=True), never resumed/re-pended.

        Deliberately conservative: only ``{'done', 'cancelled'}`` count as
        terminal here.  ``deferred`` is excluded (a deferred task can be
        un-deferred) and any live status (``blocked``, ``in-progress``,
        ``pending``, …) or an unknown/absent subject leaves the escalation
        untouched — preserving ambiguous cases (e.g. a design_concern on a
        still-live task) for a human to triage, per this task's conservatism
        requirement.

        Category-ALLOWLISTED (task 2724, superseding task 2114's category-
        agnostic close): a terminal subject only moots the escalation when the
        escalation was reliably ABOUT that task.  Task 2114 closed a terminal-
        subject L2 regardless of ``category``, but that status-only heuristic
        silently dropped still-required escalations whose real work lives
        OUTSIDE the task record.  This method now closes only escalations whose
        ``category`` is in ``config.escalation_revalidation_allowlist`` — by
        default ``{'task_failure', 'stranded_blocked'}``, the categories where
        a done/cancelled subject genuinely leaves nothing for a human to act
        on.  ``infra_issue`` is deliberately EXCLUDED (its remediation can
        outlive the task record; accepted tradeoff — a couple of historically-
        correct closes return to the human queue).  Any non-allowlisted
        category on a terminal subject is LEFT OPEN for a human to triage,
        exactly like the live-subject case.  The allowlist is a green-tier
        hot-reloadable knob, so operators can widen/narrow it live without a
        restart.

        Auditability (task 2724): the close stamps a DISTINCT
        ``resolution_class='moot-terminal-subject'`` (see
        ``_ESCALATION_REVALIDATION_RESOLUTION_CLASS``) — NOT ``'benign'`` — so
        swept records remain findable as their own dashboard dimension rather
        than being folded into the benign bucket the way task 2114's unstamped
        reaper-sweep default did.

        Returns True when the escalation was closed this call (the caller
        uses this to skip further per-item handling for the same escalation
        in the same pass), False otherwise.
        """
        if not self.config.escalation_revalidation_enabled:
            return False
        if self._escalation_queue is None:
            return False
        if getattr(esc, 'level', None) != 2:
            return False
        if esc.status != 'pending':
            return False

        subject_status = statuses.get(esc.task_id)
        if subject_status not in ('done', 'cancelled'):
            return False

        # Category allowlist gate (task 2724): only auto-close categories where a
        # terminal subject truly moots the escalation. A non-allowlisted category
        # stays pending for a human even though its subject went terminal. The
        # allowlist is a green-tier hot-reloadable knob (read live off config).
        if esc.category not in self.config.escalation_revalidation_allowlist:
            return False

        resolution = (
            f'escalation-revalidation-sweep: subject task {esc.task_id} is '
            f'{subject_status} (terminal) — escalation moot, auto-closed (close_only).'
        )
        self._escalation_queue.resolve(
            esc.id, resolution, dismiss=True, resolved_by=_ESCALATION_REVALIDATION_SWEEP_ROLE,
            resolution_class=_ESCALATION_REVALIDATION_RESOLUTION_CLASS,
        )
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_resolved,
                task_id=esc.task_id,
                data={
                    'escalation_id': esc.id,
                    'subject_status': subject_status,
                    'reason': 'terminal-subject',
                    'category': esc.category,
                    'resolution_class': _ESCALATION_REVALIDATION_RESOLUTION_CLASS,
                },
            )
        logger.info(
            'Escalation-revalidation-sweep: closed L2 %s — subject task %s is '
            '%s (terminal), close_only',
            esc.id, esc.task_id, subject_status,
        )
        return True

    async def _run_deterministic_recon_sweep(self) -> None:
        """Single testable pass of the deterministic-strand reconciliation sweep.

        Source A: enumerate blocked tasks and recover any absent-escalation
        strand via one of two DISJOINT detectors (task 2954): a deploy
        RAN-strand (task-2059 shape) via
        ``_recover_stranded_deterministic_task``, or a pure-gate /
        ``always_escalates`` GATE-strand (``gate_escalated_at`` stamped but no
        escalation record) via ``_recover_stranded_deterministic_gate``.  The
        gate branch's strand-vs-resolved discriminator is an archive-inclusive
        role-scoped emptiness check (a pending OR resolved record ⇒ not a
        strand), which also self-dedupes across passes.

        Source B: enumerate all pending escalations and re-validate any open
        deterministic-deploy ``infra_issue`` escalation via
        ``_revalidate_open_deterministic_escalation`` (a no-op for anything
        that doesn't match).  Reuses Source A's task map as a fast path,
        falling back to ``scheduler.get_task`` for an escalation whose task
        wasn't in the blocked set.

        Source C (task 2114): before Source B handles a pending escalation,
        first offer it to ``_revalidate_open_l2`` — the generalized
        terminal-subject closure (any L2 whose subject task has gone
        done/cancelled).  Evidence is a single batch ``scheduler.get_statuses``
        read over every pending L2's task-id, taken once per pass; the read
        is defensively wrapped so a scheduler that can't answer (unconfigured
        mock, transient error) degrades to an empty statuses dict rather than
        aborting the pass — Source C then closes nothing and Source B runs
        exactly as before.  When Source C closes an escalation, Source B is
        skipped for it (``continue``); this composes with the existing
        per-item fail-soft try/except so one bad escalation never blocks
        the rest.

        A and B are mutually exclusive per task ACROSS passes by construction
        (A requires an empty pending queue; B requires an open escalation) —
        once A files an escalation for a tid, the queue is non-empty so A
        will not re-fire for it on a later pass; B then owns re-validation
        and self-heals when the unit recovers.

        WITHIN a single pass, Source B's ``get_pending()`` re-globs the queue
        directory and so CAN observe an escalation Source A just filed a few
        lines above (task 2074 amendment — this was previously only true
        "by accident": an A-filed 'stranded_blocked' escalation isn't
        'infra_issue' so B's category filter skips it, and an A-filed
        'infra_issue' one re-verifies to the same 'unconfirmed' verdict
        within one pass, so re-observing it was a no-op in practice but not
        by any enforced invariant).  To make the exclusion literally true
        rather than an implementation-detail coincidence, this method tracks
        every tid Source A recovered this pass in ``recovered_this_pass`` and
        the Source-B loop skips any escalation for one of those tids
        outright — so no task is EVER handled by both sources in the same
        pass, regardless of category/role filtering or verdict stability.

        Both loops are per-item fail-soft: one bad task/escalation is logged
        and does not abort the rest of the pass.
        """
        if self._escalation_queue is None:
            return

        tasks = await self.scheduler.get_tasks(statuses=['blocked'])
        task_by_id: dict[str, dict] = {}
        recovered_this_pass: set[str] = set()
        for task in tasks:
            tid = str(task.get('id', ''))
            if not tid:
                continue
            task_by_id[tid] = task
            if task.get('status') != 'blocked':
                continue
            metadata = task.get('metadata') or {}
            # Two DISJOINT Source-A strand detectors (task 2954): a deploy
            # RAN-strand and a pure-gate / always_escalates GATE-strand. They
            # never both match one task — _deterministic_gate_stranded REQUIRES
            # gate_escalated_at, and _deterministic_deploy_stranded matches only
            # phase==RAN, but stamping gate_escalated_at on a deploy atomically
            # advances phase to ESCALATED (never RAN); its pre-ζ legacy shim
            # branch separately excludes gate_escalated_at outright. The
            # if/elif below orders deploy-before-gate as a belt-and-braces
            # backstop should that atomic invariant ever regress. Both share the
            # per-task fail-soft try/except and recovered_this_pass bookkeeping.
            _is_deploy = _deterministic_deploy_stranded(metadata)
            _is_gate = _deterministic_gate_stranded(metadata)
            if not (_is_deploy or _is_gate):
                continue
            try:
                if _is_deploy:
                    # Deploy strand: dedup on the pending queue, then re-file an
                    # L1 whose category depends on live systemd unit health.
                    if self._escalation_queue.get_by_task(tid, status='pending'):
                        continue
                    await self._recover_stranded_deterministic_task(tid, task, metadata)
                    recovered_this_pass.add(tid)
                elif _is_gate:
                    # Gate strand: the discriminator is an archive-INCLUSIVE,
                    # role-scoped emptiness check (status=None scans queue root +
                    # archive).  A PENDING record means the gate is still open
                    # (not a strand); a RESOLVED/archived record means a human
                    # already acted (genuinely-resolved, not a strand) — in
                    # either case leave it alone.  Only a TOTAL absence (never
                    # landed / lost across a restart) re-fires.  This also
                    # self-dedupes across passes: once re-filed, the next pass
                    # sees the pending record and skips.
                    if self._escalation_queue.get_by_task(
                        tid, agent_role=DETERMINISTIC_AGENT_ROLE,
                    ):
                        continue
                    await self._recover_stranded_deterministic_gate(tid, task, metadata)
                    recovered_this_pass.add(tid)
            except Exception as exc:
                logger.error(
                    'Deterministic-recon-sweep: Source-A recovery failed for '
                    'task %s: %s: %s',
                    tid, type(exc).__name__, exc,
                )

        pending = self._escalation_queue.get_pending()

        statuses: dict[str, str] = {}
        l2_ids = sorted({
            e.task_id for e in pending if getattr(e, 'level', None) == 2
        })
        if self.config.escalation_revalidation_enabled and l2_ids:
            try:
                statuses, statuses_err = await self.scheduler.get_statuses(l2_ids)
                if statuses_err is not None:
                    # get_statuses never raises on failure — it returns
                    # ({}, exception) — so this is the ONLY signal that
                    # distinguishes "no L2 subjects are terminal" (a genuine
                    # no-op pass) from "the status read itself failed"
                    # (statuses is already {} either way; log so an
                    # operator can tell the two apart instead of Source C
                    # silently closing nothing every pass).
                    logger.info(
                        'Deterministic-recon-sweep: Source-C get_statuses '
                        'returned a resolver error for %d L2 escalation(s) '
                        '(degrading to no terminal statuses this pass): '
                        '%s: %s',
                        len(l2_ids), type(statuses_err).__name__, statuses_err,
                    )
            except Exception as exc:
                logger.error(
                    'Deterministic-recon-sweep: Source-C get_statuses failed '
                    'for %d L2 escalation(s): %s: %s',
                    len(l2_ids), type(exc).__name__, exc,
                )
                statuses = {}

        for esc in pending:
            if esc.task_id in recovered_this_pass:
                continue
            try:
                if await self._revalidate_open_l2(esc, statuses):
                    continue
                task = task_by_id.get(esc.task_id)
                if task is None:
                    task = await self.scheduler.get_task(esc.task_id)
                if task is None:
                    continue
                metadata = task.get('metadata') or {}
                await self._revalidate_open_deterministic_escalation(esc, task, metadata)
            except Exception as exc:
                logger.error(
                    'Deterministic-recon-sweep: Source-B revalidation failed '
                    'for escalation %s: %s: %s',
                    esc.id, type(exc).__name__, exc,
                )

    def _on_escalation(self, escalation) -> None:
        """Callback when any escalation is submitted — wake the waiting workflow/steward."""
        # Increment before the event-set logic so the counter reflects every submission.
        # Best-effort observability counter — incremented here from arbitrary callbacks
        # without a lock.  _maybe_write_digest snapshots it once at entry so concurrent
        # callbacks cannot cause a double-skip between the threshold check and the advance.
        # May drift by a small constant under concurrency; not a correctness gate.
        self._escalation_event_count += 1  # task 1327 AFK hardening
        event = self._escalation_events.get(escalation.task_id)
        if event:
            event.set()

    def _schedule_coro_threadsafe(self, coro, *, label: str) -> None:
        """Schedule *coro* on the orchestrator event loop from any thread.

        ``_on_escalation_resolved`` can fire on EITHER thread:
          - in-loop, for callers like the watcher supervisor or unit tests
            that drive ``queue.resolve()`` from within the running loop; or
          - off-loop, when the sync escalation MCP tool ``resolve_issue`` runs
            on a FastMCP threadpool worker — there is no running loop in that
            thread, so a bare ``asyncio.create_task`` raises ``RuntimeError:
            no running event loop`` (the 2026-05-29 reify failure: the
            cascade-unblock flip was dropped AND, had it been wired, the
            scheduler auto-resume would have been too).

        On-loop: ``create_task`` + register in ``_background_tasks`` so callers
        awaiting that set still observe the work (preserves the cascade unit
        tests' ``await asyncio.gather(*_background_tasks)`` drain).

        Off-loop: hand the coroutine to ``self._loop`` via
        ``run_coroutine_threadsafe``, which is thread-safe and wakes the loop.

        Best-effort: if no loop is reachable (bare-Harness tests that never
        called ``run()`` and aren't inside a loop), the coroutine is closed and
        a WARNING logged, so the queue's callback wrapper never sees an
        exception.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            t = running.create_task(coro)
            self._background_tasks.add(t)
            t.add_done_callback(self._background_tasks.discard)
            return

        loop = self._loop
        if loop is None or loop.is_closed():
            logger.warning(
                '%s: no orchestrator event loop reachable; dropping', label,
            )
            coro.close()
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            logger.warning(
                '%s: failed to schedule on orchestrator loop', label,
                exc_info=True,
            )
            coro.close()
            return

        def _log_if_raised(f) -> None:
            try:
                exc = f.exception()
            except Exception:
                return
            if exc is not None:
                logger.warning('%s: scheduled coro raised: %r', label, exc)

        fut.add_done_callback(_log_if_raised)

    def _on_escalation_resolved(self, escalation) -> None:
        """Callback when an escalation is resolved — wake the waiting workflow."""
        # Increment for any status transition (resolved or dismissed) — both are
        # escalation events that the EWA digest needs to count.
        # Best-effort observability counter — same concurrency caveat as _on_escalation
        # above; _maybe_write_digest snapshots it at entry to avoid double-skip drift.
        self._escalation_event_count += 1  # task 1327 AFK hardening
        event = self._escalation_events.get(escalation.task_id)
        if event:
            event.set()

        # Un-halt the merge queue only when the escalation that OWNS the
        # halt resolves. Prior versions matched on category alone, which
        # let any wip_conflict resolve release the halt — leaving the real
        # blocker's escalation pending (phantom-L1 bug, esc-1888-57 on reify
        # 2026-04-16). The owner pointer is the single source of truth.
        if (
            self._merge_worker is not None
            and self._merge_worker.is_halt_owner(escalation.id)
        ):
            self._merge_worker.unhalt_wip()
            logger.info(
                'Merge queue un-halted: halt owner %s resolved', escalation.id,
            )

        # Auto-resume a paused scheduler when its pause L1 resolves.  This is
        # the documented resume path — the L1's own detail text says "resolve
        # this escalation / call resume_scheduler()" — but it was never wired,
        # so resolving the L1 cleared the marker while the scheduler stayed
        # parked and kept logging "Scheduler paused — idling" (2026-05-29 reify
        # incident).  Gated on is_paused so a resolution arriving after a manual
        # resume_scheduler() (or for a stale already-resolved sentinel) is a
        # no-op.  Scheduled threadsafe because this callback may run off-loop
        # (sync MCP resolve_issue on a FastMCP worker).  Resume only on
        # 'resolved' — 'dismissed' means the operator abandoned the pause.
        if (
            escalation.task_id == self._SCHEDULER_PAUSE_SENTINEL
            and escalation.status == 'resolved'
            and self.scheduler.is_paused
        ):
            logger.info(
                'scheduler-pause escalation %s resolved — auto-resuming '
                'scheduler dispatch', escalation.id,
            )
            self._schedule_coro_threadsafe(
                self.resume_scheduler(),
                label=f'auto-resume-scheduler (via {escalation.id})',
            )

        # Unified action dispatch — β (task 1620).
        # _resolve_escalation_action maps the escalation to a canonical action
        # string (resume/close_only/restart/park/abandon) using the explicit
        # resolution_action field first, then the parent L2's action for cascade
        # members, then the legacy dismiss→close_only / resolve→resume mapping
        # (D10) so all existing in-process tests stay green.
        #
        # The _SCHEDULER_PAUSE_SENTINEL is a synthetic task-id that has its own
        # dedicated auto-resume handling above and is not a real task; skip it.
        if escalation.task_id == self._SCHEDULER_PAUSE_SENTINEL:
            return

        action = self._resolve_escalation_action(escalation)

        # Table B (ω3, task 2196): target_status and workflow routing are
        # sourced from escalation.action_effects.effect_for — the SAME
        # authority escalation.server.resolve_issue already consumes
        # (server.py:671). This replaces the harness's former independent
        # _ACTION_TARGETS copy and action-string routing, closing the Table B
        # drift (PRD task-status-authority C6, finding 10.0).
        from escalation.action_effects import (  # noqa: PLC0415
            WORKFLOW_NONE,
            WORKFLOW_RESUME,
            effect_for,
        )

        effect = effect_for(action, escalation.level, escalation.category)
        if effect is None:
            # action outside the five actions ACTION_EFFECTS recognises.
            logger.warning(
                'escalation %s: unrecognised action %r for task %s — ignored',
                escalation.id, action, escalation.task_id,
            )
            return

        if effect.workflow_disposition == WORKFLOW_NONE:
            # close_only: operator closed the escalation without a re-pend —
            # leave the task in its current state (dismissed means "abandon;
            # stay blocked").
            logger.info(
                'escalation %s resolved with close_only — no status change '
                'for task %s', escalation.id, escalation.task_id,
            )
            return

        if effect.workflow_disposition == WORKFLOW_RESUME:
            # Re-pend the blocked task.  D7: level>=1 gate — covers L1 members
            # and born-at-L2 orphans alike.
            # Cascade member: resolved_by startswith 'l2-cascade:' (the cascade
            # fired _resolve_callback for the L2 first, then each member with
            # 'l2-cascade:<id>').  Direct/orphan: NOT l2-cascade AND task_id
            # NOT in _escalation_events (a live workflow owns its own re-pend).
            if escalation.level >= 1:
                is_l2_cascade = (
                    isinstance(escalation.resolved_by, str)
                    and escalation.resolved_by.startswith('l2-cascade:')
                )
                if is_l2_cascade:
                    # Scheduled via _schedule_coro_threadsafe so it works whether
                    # this callback fires on the orchestrator loop or off it (sync
                    # MCP resolve_issue on a FastMCP worker — where a bare
                    # asyncio.create_task raised "no running event loop").
                    self._schedule_coro_threadsafe(
                        self._cascade_unblock_member(escalation),
                        label=(
                            f'cascade-unblock task {escalation.task_id} '
                            f'(via {escalation.resolved_by})'
                        ),
                    )
                elif escalation.task_id not in self._escalation_events:
                    # Fix #1a — direct/orphan re-pend.  A live workflow owns its
                    # own re-pend (woken by event.set()); only flip when orphaned.
                    self._schedule_coro_threadsafe(
                        self._cascade_unblock_member(escalation),
                        label=(
                            f'orphan-unblock task {escalation.task_id} '
                            f'(via {escalation.resolved_by})'
                        ),
                    )
            return

        # restart / park / abandon → teardown + status write.
        # park → 'blocked' (version-a): quiescence rests on the open L2 escalation
        # suppressing Fix #1b stranded_blocked re-filing, not on the 'deferred' status.
        # effect.target_status is Table B's target for `action`; every entry
        # reaching this point (disposition not NONE/RESUME) carries a str target
        # today. Guarded explicitly rather than via a bare `assert` — `python -O`
        # strips assertions, which would otherwise let a future Table B edit that
        # mapped a teardown action to a None target flow straight into
        # scheduler.set_task_status.
        if effect.target_status is None:
            logger.warning(
                'escalation %s: action %r resolved to a teardown disposition '
                'but Table B has no target_status for task %s — ignored',
                escalation.id, action, escalation.task_id,
            )
            return
        self._schedule_coro_threadsafe(
            self._action_teardown_and_set_status(
                escalation.task_id, effect.target_status, action,
            ),
            label=(
                f'action-teardown task {escalation.task_id} '
                f'action={action} target={effect.target_status}'
            ),
        )

    async def _action_teardown_and_set_status(
        self,
        task_id: str,
        target_status: str,
        action: str,
    ) -> None:
        """Async helper: execute a non-resume resolution action on a task.

        Implements C3.1 (status-precedes-kill) ordering:
          1. Terminal recheck — skip if the task is already done/cancelled.
          2. restart only: clear ``metadata.merge_retry_pending`` (task 3024).
          3. Stamp ``_action_teardown_tasks`` (suppression, C3.2 / D9).
          4. Write ``target_status`` via scheduler.
          5. Kill live workflow if active (soft → grace → hard).
          6. In the finally block, once the kill window closes: clear the
             suppression stamp, then (restart only) re-run step 2's clear so a
             stamp resurrected by the dying workflow's own metadata write does
             not survive the restart.

        Preconditions per action:
          - restart: task must be non-terminal (checked at step 1); target='pending'.
            Any non-terminal current status is intentionally overwritten — restart is
            a forced re-pend.  SetTaskStatusRejected handles the terminal-exit gate
            if the task transitions terminally between the recheck and the write.
          - park:    task must be non-terminal; target='blocked'.  Same TOCTOU note.
            Quiescence from the open L2 escalation (Fix #1b skip gate), not from status.
          - abandon: task must be non-terminal; target='cancelled'.  SetTaskStatusRejected
            is redundant here (cancelled is terminal and the scheduler's terminal-exit
            gate would also catch it), but the recheck still avoids a no-op write.

        Suppression scope — why suppressing only 'blocked' writes is correct (C3.2):
          A workflow paused at an escalation-event wait has already written 'in-progress'
          at task-start (``_setup_worktree_and_artifacts``).  After ``event.set()`` wakes
          it, it continues mid-task execution and does NOT re-write 'in-progress' — that
          write happens only once, at task dispatch time.  On cancellation the workflow's
          cleanup path calls ``_mark_blocked``, which emits exactly one 'blocked' write.
          That 'blocked' write is the racing write suppression must intercept; suppressing
          'blocked' alone is therefore sufficient for this code path.

        The stamp in step 2 activates the scheduler's _suppress_blocked_write predicate,
        so any racing 'blocked' write emitted by the workflow before the kill lands is
        absorbed without reaching fused-memory.  The stamp is cleared in the finally
        block whether the status write succeeded, was rejected, or the kill completed —
        ensuring a re-dispatched (restart→pending) workflow can write 'blocked'
        legitimately in its next incarnation rather than being permanently suppressed.

        Suppression stamp is SKIPPED when target_status == 'blocked' (park, version-a):
          For park, the target itself is 'blocked'.  Stamping would cause
          _suppress_blocked_write to absorb park's own set_task_status('blocked') write
          → silent no-op.  There is no clobber risk: a racing workflow _mark_blocked
          writes the same value (idempotent), so no suppression is needed.

        Concurrency note: ``_action_teardown_tasks`` uses a Counter so that overlapping
        teardown coros for the same task_id (low probability — each L2 cascade member is
        a distinct task_id) do not prematurely clear each other's suppression windows.
        """
        current = await self.scheduler.get_status(task_id)
        if current in TERMINAL_STATUSES:
            logger.debug(
                'action-teardown %s: task %s is already %s (terminal) — skip',
                action, task_id, current,
            )
            return

        # restart means "run this task again from scratch", so void any durable
        # merge-retry obligation first (task 3024).  A surviving
        # metadata.merge_retry_pending stamp makes the re-dispatched workflow
        # fast-path straight to the merge phase (_resume_merge_retry_if_pending),
        # skipping plan/execute/verify/review entirely — so without this clear a
        # restart re-enters exactly the state it was invoked to escape, and an
        # operator has no reliable way to force a fresh plan.
        #
        # Ordered BEFORE the status write: once the task reads 'pending' the
        # scheduler may re-dispatch it, and a clear landing after that loses the
        # race to a workflow that has already read the stamp.  That ordering
        # leaves the opposite race open — the still-live workflow can resurrect
        # the stamp with a metadata write of its own before the kill lands — so
        # an idempotent second clear runs in the finally block once the kill
        # window has closed (see there).
        if action == 'restart':
            await self._clear_merge_retry_pending_for_restart(task_id)

        logger.info(
            'action-teardown %s: writing task %s → %s',
            action, task_id, target_status,
        )
        # Stamp suppression window BEFORE the status write (C3.2 / D9, task 1620 step-12).
        # Must be set prior to set_task_status so that a racing workflow 'blocked' write
        # (concurrent with our status write or arriving before the kill lands) is
        # absorbed by the scheduler guard and cannot clobber the action's target status.
        # Counter increment (not set.add) so overlapping teardowns for the same task_id
        # don't prematurely clear each other's suppression window (step-12 amend).
        #
        # SKIP the stamp when target_status == 'blocked' (park, version-a):
        # The scheduler's _suppress_blocked_write would absorb park's OWN write.
        # A racing _mark_blocked writes the same 'blocked' value (idempotent) — no
        # suppression is needed.  restart('pending') and abandon('cancelled') still stamp.
        #
        # Trade-off: because the stamp is the mechanism that absorbs the racing
        # _mark_blocked write, that workflow-emitted write now reaches the scheduler
        # unsuppressed.  A 'blocked' transition triggers reconciliation per the project
        # contract, so park can produce up to two 'blocked' writes (teardown + workflow
        # cleanup) and therefore up to two reconciliation passes for the same task.
        # Both writes are idempotent and the reconciliation passes converge immediately
        # (task is already blocked + open escalation → no stranded_blocked filed).
        # The duplicate cost is accepted because the alternative (stamping to suppress
        # the workflow write) would also suppress park's own write → silent no-op.
        _should_stamp = target_status != 'blocked'
        if _should_stamp:
            self._action_teardown_tasks[task_id] += 1
        try:
            try:
                await self.scheduler.set_task_status(task_id, target_status)
            except SetTaskStatusRejected as e:
                logger.warning(
                    'action-teardown %s: set_task_status(%s, %s) rejected: %s',
                    action, task_id, target_status, e,
                )
                return

            # Restart path: clear deterministic gate stamps so the re-dispatched
            # runner re-fires the gate from scratch.  resume preserves stamps so
            # the runner's idempotency drives the gate to done instead (I2/B4/B5).
            # park/abandon/close_only are left untouched — stamps preserved.
            if action == 'restart':
                task = await self.scheduler.get_task(task_id)
                if task is not None and Scheduler.is_deterministic(task):
                    await self.scheduler.update_task(
                        task_id,
                        {'gate_escalated_at': None, 'before_done_ran_at': None},
                        metadata_mode='merge',
                    )
                    logger.info(
                        'action-teardown restart: cleared deterministic gate stamps '
                        'for task %s so re-dispatch re-fires the gate',
                        task_id,
                    )

            # Kill sequence for a live workflow (C3.1, D9).
            # Status write is already done above — kill strictly follows.
            if self.is_workflow_active(task_id):
                self.cancel_workflow(task_id)
                logger.info(
                    'action-teardown %s: soft-cancelled workflow for task %s',
                    action, task_id,
                )
                # Poll for slot to clear.  The _action_teardown_tasks suppression
                # (still active in the stamp) ensures a racing _mark_blocked write is
                # absorbed while the kill is in flight.  Use the terminal-status
                # hard-cancel budget as the poll ceiling — consistent with the existing
                # cancel scan discipline (harness.py:_mark_terminal_status_cancel_scan).
                _POLL_SLEEP_S = 0.05
                max_polls = getattr(self.config, 'terminal_status_hard_cancel_polls', 10)
                polls = 0
                while self.is_workflow_active(task_id) and polls < max_polls:
                    await asyncio.sleep(_POLL_SLEEP_S)
                    polls += 1
                if self.is_workflow_active(task_id):
                    logger.warning(
                        'action-teardown %s: workflow for task %s did not clear '
                        'within %d polls — escalating to hard_cancel_workflow',
                        action, task_id, max_polls,
                    )
                    self.hard_cancel_workflow(task_id)
        finally:
            if _should_stamp:
                # Decrement the suppression refcount once the kill window closes (step-12).
                # Delete the key when it reaches zero so Counter.__contains__ returns False
                # and a re-dispatched (restart→pending) workflow can write 'blocked'
                # legitimately in its next incarnation.
                # (Sync and first, so an await below can never skip it.)
                self._action_teardown_tasks[task_id] -= 1
                if self._action_teardown_tasks[task_id] <= 0:
                    del self._action_teardown_tasks[task_id]
            # Second, idempotent stamp clear — closes the mirror race the
            # pre-status ordering opens (task 3024 amendment).  Between the first
            # clear and the kill, the still-live workflow can perform any metadata
            # read-modify-write; every one of those goes through
            # _merge_fresh_metadata's `{**in_memory, **backend}` union in merge
            # mode, so a stamp still held in that workflow's in-memory copy is
            # written straight back and the re-dispatch fast-paths to merge
            # anyway.  Re-running the clear after the kill window has closed
            # deletes any such resurrection.  Costs one get_task read and is a
            # zero-write no-op when (normally) no stamp is present.  It also runs
            # on the SetTaskStatusRejected early return (the task went terminal
            # mid-teardown), which is fine and mildly desirable: a done/cancelled
            # task should not carry a merge-retry obligation either.
            if action == 'restart':
                await self._clear_merge_retry_pending_for_restart(task_id)

    async def _clear_merge_retry_pending_for_restart(self, task_id: str) -> None:
        """Drop ``metadata.merge_retry_pending`` so a restart re-plans from scratch.

        The stamp is a durable obligation to resume straight into the merge
        phase (workflow ``_resume_merge_retry_if_pending``), which skips
        plan/execute/verify/review.  That is the opposite of what restart means,
        and leaving it in place is what makes restart unable to free a task
        wedged in that fast-path (task 3024).

        ``metadata_mode='replace'`` is required, not incidental: the default
        'merge' mode preserves keys omitted from the payload, so only a
        whole-blob replace can actually DELETE a key.  The read-modify-write
        therefore has to carry every other key through — hence reading current
        metadata first rather than writing a hand-built dict.

        This is the second implementation of that rule (the first is
        ``TaskWorkflow._clear_merge_retry_pending``), which is deliberate but not
        desirable: the subtlety belongs in one place next to ``metadata_mode``,
        and that place — ``Scheduler`` / the fused-memory metadata contract — is
        outside this task's lock scope.  Pending task 3151 (targeted
        ``delete_keys`` mode, scheduler.py in scope) is the vehicle for
        collapsing both call sites onto one helper.

        No-op (one ``get_task`` read, zero writes) unless the stamp is actually
        present — which is what makes it safe to call twice per restart teardown:
        once before the status write (to beat a re-dispatch) and once in the
        ``finally`` block after the kill window closes (to delete a stamp the
        dying workflow's own metadata write resurrected in between).

        Best-effort by design: the first call runs BEFORE the status write, so a raised
        metadata read/write error would abort the whole teardown and leave the
        task in its pre-restart status with the kill sequence never run —
        strictly worse than a surviving stamp.  So every failure is logged and
        swallowed; the restart proceeds either way.  (The workflow-side
        conflict probe in ``_resume_merge_retry_if_pending`` is the other,
        independent remedy for the same wedge, so a failure here is not the
        last line of defence.)
        """
        try:
            task = await self.scheduler.get_task(task_id)
            if task is None:
                # scheduler.get_task swallows every exception and returns None,
                # so this — not the except arm — is the read failure that
                # actually happens in production.  Returning quietly here would
                # make 'this task had no obligation' indistinguishable from 'we
                # could not tell' (no-silent-fail-soft).
                logger.warning(
                    'action-teardown restart: could not read task %s to clear '
                    'merge_retry_pending — proceeding with the restart; the '
                    're-dispatch may still fast-path to merge',
                    task_id,
                )
                return
            metadata = task.get('metadata')
            if not isinstance(metadata, dict) or 'merge_retry_pending' not in metadata:
                return
            cleaned = {k: v for k, v in metadata.items() if k != 'merge_retry_pending'}
            ok = await self.scheduler.update_task(
                task_id, metadata=cleaned, metadata_mode='replace',
            )
        except Exception as exc:  # noqa: BLE001 — best-effort: never block the restart
            logger.warning(
                'action-teardown restart: could not clear merge_retry_pending for '
                'task %s (%s) — proceeding with the restart; the re-dispatch may '
                'still fast-path to merge',
                task_id, exc,
            )
            return
        if not ok:
            # update_task reports MCP-level failures by returning False, not by
            # raising.  Emitting the success line below regardless would actively
            # mislead an operator debugging a restart that failed to break the
            # fast-path: they would read 'cleared' and rule out the real cause.
            logger.warning(
                'action-teardown restart: the merge_retry_pending clearing write for '
                'task %s was rejected (update_task returned False) — proceeding with '
                'the restart; the stamp survives, so the re-dispatch may still '
                'fast-path to merge',
                task_id,
            )
            return
        logger.info(
            'action-teardown restart: cleared merge_retry_pending stamp for task '
            '%s so the re-dispatch runs the full plan/execute/verify/review '
            'pipeline instead of fast-pathing to merge',
            task_id,
        )

    def _resolve_escalation_action(self, escalation) -> str:
        """Resolve the canonical action for a resolved/dismissed escalation.

        Priority:
          1. The explicit ``resolution_action`` on the record (set by the MCP
             server at resolve_issue time — α1 path).
          2. For cascade members (``resolved_by='l2-cascade:<parent_id>'``):
             read the parent L2's ``resolution_action`` (already archived by
             the time the member callback fires).  Falls through if absent.
          3. Legacy mapping (D10): ``status=='dismissed'`` → ``'close_only'``;
             ``status=='resolved'`` → ``'resume'``.

        The legacy path ensures all in-process flows and existing tests that
        carry no ``resolution_action`` continue to behave as before.
        """
        # 1) Explicit action on this record.
        if escalation.resolution_action is not None:
            return escalation.resolution_action

        # 2) Cascade member: inherit parent L2's action.
        if (
            isinstance(escalation.resolved_by, str)
            and escalation.resolved_by.startswith('l2-cascade:')
        ):
            parent_id = escalation.resolved_by.split(':', 1)[1]
            parent = (
                self._escalation_queue.get(parent_id)
                if self._escalation_queue is not None
                else None
            )
            if parent is not None and parent.resolution_action is not None:
                return parent.resolution_action
            # Fall through to legacy map (parent also lacks resolution_action).

        # 3) Legacy mapping.
        if escalation.status == 'dismissed':
            return 'close_only'
        return 'resume'  # status == 'resolved'

    async def _check_reblock_guard(self, escalation, task_id: str) -> bool:
        """Check the re-block guard counter; persist before proceeding.

        Returns True → proceed with the blocked→pending flip.
        Returns False → withhold the flip (threshold reached).

        Counter logic (C5):
          - Read fresh metadata via scheduler.get_task (fresh each incarnation).
          - Compute new signature via _reblock_signature.
          - Same-sig → prev_count + 1; different-sig → reset to 1.
          - Withhold iff same-sig AND prev_count >= _REBLOCK_GUARD_THRESHOLD
            (check BEFORE incrementing — so 3 same-sig flips proceed and the
            4th is withheld; this matches the "4th flip withheld" contract).
          - Otherwise: increment/reset, persist via update_task(append=True)
            BEFORE the flip (crash-safe: over-count never under-count — C5),
            then return True.
        """
        new_sig = self._reblock_signature(escalation)

        # Read fresh metadata so each incarnation sees the persisted count.
        task = await self.scheduler.get_task(task_id)
        metadata = (task or {}).get('metadata') or {}
        try:
            guard = metadata.get('reblock_guard')
            # Validate shape: a corrupt/non-dict truthy value (e.g. a string
            # from a bad write or manual edit) would raise AttributeError on
            # guard.get(...).  Degrade gracefully to an empty dict so the guard
            # resets to count 0 rather than aborting the flip entirely.
            if not isinstance(guard, dict):
                guard = {}
            prev_count = int(guard.get('count') or 0)
            prev_signature = guard.get('signature')
        except (TypeError, ValueError):
            prev_count = 0
            prev_signature = None

        same_sig = (prev_signature is not None and prev_signature == new_sig)

        # NOTE on cumulative counting (C5, intentional): the counter is NEVER
        # automatically cleared when the task makes progress (e.g. moves to
        # in-progress or done and later gets re-blocked).  Only two paths reset
        # it: (a) a human explicitly clears metadata.reblock_guard — an absent
        # or non-dict guard reads as count 0, starting fresh; or (b) the
        # escalation signature changes (different category or substantially
        # different summary), which resets count to 1 and proceeds.
        # This cumulative behaviour is intentional: repeated failures with an
        # identical signature are considered pathologically repetitive regardless
        # of intervening progress.  An operator who has genuinely fixed the root
        # cause should clear the guard to signal "new slate".

        # Threshold check (check-before-flip ordering — see design decision):
        # same-sig AND prev_count >= threshold → withhold + file L2.
        if same_sig and prev_count >= _REBLOCK_GUARD_THRESHOLD:
            self._file_reblock_guard_l2(task_id, prev_count, new_sig)
            return False

        # Same signature → increment; different signature → reset to 1.
        # Mirrors _check_*_thrash shape: same-sig +1, different-sig reset to 1
        # (reset to 1 = "we just saw one"; reset to 0 would lose the current flip).
        new_count = prev_count + 1 if same_sig else 1

        # Persist BEFORE the flip (crash-safe: over-count, never under-count — C5).
        # metadata_mode='merge': shallow last-write-wins — the whole reblock_guard
        # key is overwritten wholesale (new count/signature win), while sibling
        # metadata keys (files, memory_hints, …) are preserved.  'additive' would
        # use scalar OLD-wins under #1827's _merge_values, freezing the counter
        # across incarnations when the guard key already exists.
        await self.scheduler.update_task(
            task_id,
            {'reblock_guard': {'count': new_count, 'signature': new_sig}},
            metadata_mode='merge',
        )

        return True

    def _file_reblock_guard_l2(self, task_id: str, prev_count: int, sig: str) -> None:
        """File a born-at-L2 human escalation when the re-block guard threshold is hit.

        Mirrors _file_watcher_outage_l2:
          - No-op when _escalation_queue is None (bare-Harness unit tests).
          - Deduped via find_pending_l2_by_root_cause(root_cause) so repeated
            trips for the same stuck task file exactly one L2.
          - Best-effort: all exceptions are swallowed (same pattern as
            _file_watcher_outage_l2) so this never breaks the re-pend path.
          - agent_role='harness-reblock-guard' — a harness-sentinel that stays
            born-at-L2 under the C4 allowlist (severity='urgent').
        """
        queue = getattr(self, '_escalation_queue', None)
        if not queue:
            return
        try:
            root_cause = f'reblock-guard:{task_id}'
            if queue.find_pending_l2_by_root_cause(root_cause) is not None:
                return  # dedup: one open L2 per stuck task
            from escalation.models import Escalation
            summary = (
                f'persistent re-block: {prev_count} redispatches, signature {sig}'
            )[:200]
            detail = (
                f'Task {task_id} has been re-blocked {prev_count} time(s) with the '
                f'same failure signature, reaching the _REBLOCK_GUARD_THRESHOLD.\n\n'
                f'Signature: {sig}\n\n'
                f'The automatic blocked→pending flip is now suppressed for this '
                f'signature.  Investigate the root cause, fix the underlying issue, '
                f'and clear metadata.reblock_guard on the task to allow future '
                f're-pends to proceed.'
            )
            esc = Escalation(
                id=queue.make_id(task_id),
                task_id=task_id,
                agent_role='harness-reblock-guard',
                severity='urgent',
                category='task_failure',
                level=2,
                root_cause=root_cause,
                summary=summary,
                detail=detail,
                suggested_action='investigate_reblock_loop',
            )
            queue.submit(esc)
            try:
                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_created,
                        task_id=task_id,
                        data={
                            'escalation_id': esc.id,
                            'category': esc.category,
                            'severity': esc.severity,
                            'level': esc.level,
                            'reason': 'reblock-guard-threshold',
                        },
                    )
            except Exception:
                # Isolated from the outer handler: the L2 is already filed at
                # this point, so a failure here is an observability-only
                # miss, never a "failed to file L2" condition.
                logger.warning(
                    'reblock-guard: L2 %s filed for task %s but '
                    'escalation_created emit failed', esc.id, task_id,
                    exc_info=True,
                )
            logger.warning(
                'reblock-guard: task %s hit threshold (count=%d, sig=%r); '
                'flip withheld, L2 filed %s',
                task_id, prev_count, sig, esc.id,
            )
        except Exception:
            logger.warning(
                'reblock-guard: failed to file L2 for task %s', task_id,
                exc_info=True,
            )

    # Warm-lane record-drift born-at-L2 filer (task 2986, W2b I3/I4). Constant
    # (not per-task) dedup key — one open L2 at a time regardless of how many
    # lanes drift — mirroring _WATCHER_OUTAGE_ROOT_CAUSE's bare-literal style.
    # The literal 'lane_record_drift' matches the PRD/capability-manifest grep.
    _LANE_RECORD_DRIFT_SENTINEL: str = '__lane_record_drift__'
    _LANE_RECORD_DRIFT_ROOT_CAUSE: str = 'lane_record_drift'
    _LANE_RECORD_DRIFT_ROLE: str = 'orchestrator-lane-record-drift'

    def _file_lane_record_drift_l2(self, count: int) -> None:
        """File a born-at-L2 human escalation when warm-lane record drift persists.

        Installed on ``WarmLanePool._on_lane_record_drift`` (declare-on-callee
        default None, install-in-harness), fired by the pool once
        ``drift_l2_threshold`` consecutive durable ``.lane-state`` mirror writes
        fail.  The pool NEVER raises on a mirror failure (fail-open, I3) — the
        in-memory assignment map stays the source of truth and acquire/release
        keep succeeding — so this filer is the only path by which the drift
        between the map and the durable records becomes visible to a human.

        Mirrors _file_watcher_outage_l2 / _file_reblock_guard_l2:
          - No-op when _escalation_queue is None (bare-Harness unit tests).
          - Deduped via find_pending_l2_by_root_cause('lane_record_drift') — a
            bare fixed literal (not a per-task f-string) so repeated trips file
            exactly one pending L2.
          - Best-effort: every exception is swallowed so this never breaks the
            pool's acquire/release path (I3).
          - agent_role is an orchestrator sentinel + severity='urgent' → the L2
            is exempt from the agent-role downgrade gate and routes straight to
            a human.
        """
        queue = getattr(self, '_escalation_queue', None)
        if not queue:        # bare-Harness unit tests / lifecycle tests stay green
            return
        try:
            if queue.find_pending_l2_by_root_cause(
                self._LANE_RECORD_DRIFT_ROOT_CAUSE
            ) is not None:
                return                         # dedup: one open L2 at a time
            from escalation.models import Escalation
            summary = (
                f'warm-lane record drift: {count} consecutive durable-write '
                f'failures (in-memory assignment map diverged from .lane-state)'
            )[:200]
            detail = (
                f'The WarmLanePool durable-record mirror write has failed '
                f'{count} consecutive time(s), reaching '
                f'warm_lane_drift_l2_threshold.\n\n'
                'The in-memory FREE/ASSIGNED assignment map remains the single '
                'source of truth and acquire/release continue to succeed '
                '(fail-open), so task dispatch is NOT blocked.  But the durable '
                '.lane-state/<lane>.json records have drifted from the map (a '
                'write raised OSError — .lane-state unwritable — or '
                'IllegalLaneTransition), so a restart would rebuild the map '
                'from stale records.\n\n'
                'Investigate why the durable write is failing (disk full, '
                'permissions, corrupt .lane-state), fix the underlying issue, '
                'and restart the orchestrator so the pool re-seeds and the '
                'records reconcile with the map.'
            )
            esc = Escalation(
                id=queue.make_id(self._LANE_RECORD_DRIFT_SENTINEL),
                task_id=self._LANE_RECORD_DRIFT_SENTINEL,
                agent_role=self._LANE_RECORD_DRIFT_ROLE,
                severity='urgent',
                category='infra_issue',
                level=2,
                root_cause=self._LANE_RECORD_DRIFT_ROOT_CAUSE,
                summary=summary,
                detail=detail,
                suggested_action='investigate_lane_record_drift',
            )
            queue.submit(esc)
            try:
                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_created,
                        task_id=self._LANE_RECORD_DRIFT_SENTINEL,
                        data={
                            'escalation_id': esc.id,
                            'category': esc.category,
                            'severity': esc.severity,
                            'level': esc.level,
                            'reason': 'lane-record-drift-threshold',
                        },
                    )
            except Exception:
                # Isolated from the outer handler: the L2 is already filed, so a
                # failure here is an observability-only miss, never a "failed to
                # file L2" condition.
                logger.warning(
                    'lane-record-drift: L2 %s filed but escalation_created '
                    'emit failed', esc.id, exc_info=True,
                )
            logger.warning(
                'lane-record-drift: warm-lane durable record drift hit '
                'threshold (count=%d); flip fail-open, L2 filed %s',
                count, esc.id,
            )
        except Exception:
            logger.warning(
                'lane-record-drift: failed to file L2 escalation',
                exc_info=True,
            )

    _STRUCTURAL_EXHAUSTION_SENTINEL: str = '__warm_lane_structural_exhaustion__'
    _STRUCTURAL_EXHAUSTION_ROOT_CAUSE: str = 'warm_lane_pool_structurally_exhausted'
    _STRUCTURAL_EXHAUSTION_ROLE: str = 'orchestrator-warm-lane-structural-exhaustion'

    def _file_structural_exhaustion_l2(
        self, count: int, census: WarmLanePoolCensus,
    ) -> None:
        """File a born-at-L2 human escalation when the warm-lane pool is
        STRUCTURALLY exhausted — PRD ε pole-2 (the silent-infinite-requeue pole).

        Installed on ``GitOps._on_structural_exhaustion`` (declare-on-callee
        default None, install-in-harness), fired by GitOps once
        ``warm_lane_structural_exhaustion_l2_threshold`` consecutive
        ``acquire_warm_lane`` calls return EXHAUSTED with no fresh lane and no
        reclaimable capacity.  A warm-lane EXHAUSTED requeue no longer counts
        against the per-task requeue cap (task 2988 pole-1), so WITHOUT this
        filer a genuinely stuck pool (a lane leak, every lane pinned, or a pool
        sized too small) would requeue every task forever with NO loud signal.
        This born-at-L2 is that sole loud signal.

        Mirrors :meth:`_file_lane_record_drift_l2`:
          - No-op when _escalation_queue is None (bare-Harness unit tests).
          - Deduped via find_pending_l2_by_root_cause(
            'warm_lane_pool_structurally_exhausted') — a bare fixed literal (not
            a per-task f-string) so repeated threshold trips file exactly one
            pending L2.
          - Best-effort: every exception is swallowed so filing can never break
            the pool's acquire path (I3 fail-open).
          - agent_role is an orchestrator sentinel + severity='urgent' → the L2
            is exempt from the agent-role downgrade gate and routes straight to
            a human.
          - Carries the α census counts (size / n_free / n_assigned_dispatched /
            n_pinned_non_dispatched / n_unknown_dispatch / n_quarantined) as
            structured fields so an operator sees WHY the pool is full (INV-2).

        SIZING ASSUMPTION (review amendment — robustness): this L2 is a reliable
        ABNORMALITY signal only when the warm-lane pool is sized to
        ``max_concurrent_tasks`` — then a sustained run of EXHAUSTED genuinely
        means a leak or a mismatch, not honest saturation.  When the pool is
        deliberately sized SMALLER than max_concurrent_tasks, a legitimately
        saturated pool (every lane held by a live dispatched task, no leak) can
        accumulate consecutive EXHAUSTED from queued acquires and trip this
        human-routed 'urgent' L2.  That case is left to the operator to
        distinguish via the census carried below (n_assigned_dispatched == size
        with n_pinned_non_dispatched == 0 ⇒ pure saturation, not a leak), and the
        threshold (``warm_lane_structural_exhaustion_l2_threshold``) plus this
        severity are a CONSCIOUS, green-tier/hot-reloadable tuning choice — raise
        the threshold or run a size==max_concurrent_tasks pool to keep EXHAUSTED
        an abnormality signal.  Census-shape severity gating (saturation ⇒ lower
        tier) was considered and deliberately deferred as out of scope here.
        """
        queue = getattr(self, '_escalation_queue', None)
        if not queue:        # bare-Harness unit tests / lifecycle tests stay green
            return
        try:
            if queue.find_pending_l2_by_root_cause(
                self._STRUCTURAL_EXHAUSTION_ROOT_CAUSE
            ) is not None:
                return                         # dedup: one open L2 at a time
            from escalation.models import Escalation
            census_line = census.render()
            summary = (
                f'warm-lane pool structurally exhausted: {count} consecutive '
                f'EXHAUSTED acquires — {census_line}'
            )[:200]
            detail = (
                f'acquire_warm_lane has returned EXHAUSTED {count} consecutive '
                'time(s), reaching warm_lane_structural_exhaustion_l2_threshold — '
                'no FREE lane and no reclaimable capacity across that many '
                'attempts.\n\n'
                f'Pool census at exhaustion:\n  {census_line}\n\n'
                'A warm-lane EXHAUSTED requeue no longer burns the per-task '
                'requeue cap (task 2988 pole-1: a transient capacity crunch must '
                'not escalate the wrong task), so a pool that is STRUCTURALLY '
                'stuck would otherwise requeue every task forever with no loud '
                'signal.  This born-at-L2 is that signal.\n\n'
                'Investigate the census above: n_pinned_non_dispatched > 0 points '
                'to lanes held by stuck non-dispatched tasks (a leak); '
                'n_assigned_dispatched == size means genuine saturation (raise '
                'max_concurrent_tasks / pool size or shed load); '
                'n_quarantined > 0 means durably quarantined lanes need recovery.'
            )
            esc = Escalation(
                id=queue.make_id(self._STRUCTURAL_EXHAUSTION_SENTINEL),
                task_id=self._STRUCTURAL_EXHAUSTION_SENTINEL,
                agent_role=self._STRUCTURAL_EXHAUSTION_ROLE,
                severity='urgent',
                category='infra_issue',
                level=2,
                root_cause=self._STRUCTURAL_EXHAUSTION_ROOT_CAUSE,
                summary=summary,
                detail=detail,
                suggested_action='investigate_warm_lane_structural_exhaustion',
            )
            queue.submit(esc)
            try:
                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_created,
                        task_id=self._STRUCTURAL_EXHAUSTION_SENTINEL,
                        data={
                            'escalation_id': esc.id,
                            'category': esc.category,
                            'severity': esc.severity,
                            'level': esc.level,
                            'reason': 'warm-lane-structural-exhaustion-threshold',
                        },
                    )
            except Exception:
                # Isolated from the outer handler: the L2 is already filed, so a
                # failure here is an observability-only miss, never a "failed to
                # file L2" condition.
                logger.warning(
                    'warm-lane-structural-exhaustion: L2 %s filed but '
                    'escalation_created emit failed', esc.id, exc_info=True,
                )
            logger.warning(
                'warm-lane-structural-exhaustion: pool structurally exhausted '
                '(consecutive EXHAUSTED=%d); L2 filed %s — %s',
                count, esc.id, census_line,
            )
        except Exception:
            logger.warning(
                'warm-lane-structural-exhaustion: failed to file L2 escalation',
                exc_info=True,
            )

    @staticmethod
    def _reblock_signature(escalation) -> str:
        """Derive the re-block guard signature from an escalation.

        Signature = ``category + ':' + normalize(summary)[:120]``, where
        normalize = whitespace-collapse (``' '.join(s.split())``) + lowercase.

        This makes the signature robust to trivial formatting differences
        (extra spaces, newlines, capitalisation) across incarnations while
        keeping it specific enough to distinguish genuinely different failure
        modes (different category or substantially different summary).

        The 120-char truncation is applied to the *normalised* summary only;
        the category prefix is not counted toward the 120.
        """
        raw = escalation.summary or ''
        normalized = ' '.join(raw.split()).lower()
        return f'{escalation.category}:{normalized[:120]}'

    async def _cascade_unblock_member(self, escalation) -> None:
        """Async helper: flip a cascade-resolved L1 member task from blocked→pending.

        Only 'blocked' tasks are flipped. Every other status — including
        terminal statuses (done, cancelled), non-terminal live statuses
        (deferred, in-progress, pending, merge-deferred), and any future
        status — is DEBUG-skipped. The intent of this feature is purely to
        unblock 'blocked' tasks; terminal members completing normally while
        their L2 cluster is still pending is an expected, common outcome and
        should not produce operator-visible WARNINGs.

        EXCEPTION: an infra-held task (first-class status == 'infra-hold',
        via is_infra_held — task 2200/ω4) is checked FIRST, before the
        'blocked'-only gate below, because its status is 'infra-hold', never
        'blocked' — see the A1 guard.

        TOCTOU note: get_status and set_task_status are separate MCP
        round-trips with no atomic compare-and-set. If the task transitions
        away from 'blocked' between the read and the write (e.g. a workflow
        picks it up → 'in-progress'), set_task_status('pending') may succeed
        and clobber the newer status. This is accepted as a best-effort
        policy; the race window is narrow in practice.

        Efficiency note (review amendment, task 2200): get_task above (used
        only for the is_infra_held pre-gate) and get_status below both
        dispatch the same underlying fused-memory 'get_task' RPC, so the
        non-infra-held path pays for two round-trips over the same row. This
        is intentional, not an oversight: get_status is kept as an
        independent, as-late-as-possible read to narrow the TOCTOU window
        above rather than reuse the earlier (by-then slightly staler)
        get_task snapshot. Collapsing the two would also require reworking
        test_cascade_unblock.py (outside this task's locked module scope),
        whose mixed-status-cascade coverage
        (test_criterion_7_mixed_status_cascade) drives get_status with a
        per-task_id side_effect while get_task's mock stays fixed/shared —
        i.e. that suite already treats the two reads as independent by
        design.
        """
        task_id = escalation.task_id

        # A1 guard (task 2200/ω4): an infra-held task (first-class status ==
        # 'infra-hold', via is_infra_held) is a verify-complete branch held by
        # a transient infra failure.  Checked BEFORE the blocked-only gate
        # below — an infra-held task's status is 'infra-hold', never
        # 'blocked', so this must run first or the gate would skip it
        # entirely.  Flipping to 'pending' would force the task to re-compete
        # for its implement footprint in the scheduler's footprint-locked
        # dispatch — the 3465 starvation root cause.  Instead resume-at-verify:
        # set in-progress (the scheduler already skips re-implement for
        # branches with prior work via _has_prior_implementation).  There is
        # no metadata flag to clear anymore — the status IS the hold.
        # Migration-window caveat (review amendment, task 2200): this check
        # cannot see a legacy metadata.infra_hold-only row (status still
        # 'blocked') — it falls through to the ordinary Table B resume below
        # and re-competes for its footprint.  See
        # orchestrator.task_status.is_infra_held's docstring for the
        # accepted-risk rationale and the operator follow-up.
        _infra_task = await self.scheduler.get_task(task_id)
        if is_infra_held(_infra_task):
            try:
                await self.scheduler.set_task_status(task_id, 'in-progress')
                logger.info(
                    'cascade-unblock: task %s is infra-held — resuming at '
                    'verify (infra-hold→in-progress) via %s',
                    task_id, escalation.resolved_by,
                )
            except SetTaskStatusRejected as e:
                logger.warning(
                    'cascade-unblock: infra-hold resume refused for %s '
                    '(TOCTOU race or guard): %s',
                    task_id, e,
                )
            except Exception:
                logger.warning(
                    'cascade-unblock: infra-hold resume failed for %s',
                    task_id, exc_info=True,
                )
            return

        # Deliberately a fresh, independent round-trip — not reused from
        # _infra_task above.  See the "Efficiency note" in this method's
        # docstring.
        status = await self.scheduler.get_status(task_id)

        if status != 'blocked':
            logger.debug(
                'cascade-unblock: task %s is %s (not blocked; skipping flip via %s)',
                task_id, status, escalation.resolved_by,
            )
            return

        # Re-block guard (C5/D6): count same-signature re-pends cross-incarnation;
        # withhold the flip when the threshold is reached.
        if not await self._check_reblock_guard(escalation, task_id):
            return

        # Only 'blocked' reaches here — attempt the flip.
        # Table B (ω3, task 2196): source the target from the same authority
        # _on_escalation_resolved / escalation.server.resolve_issue use, so
        # resume→pending cannot drift into a third independent copy.
        from escalation.action_effects import effect_for  # noqa: PLC0415

        _resume_effect = effect_for('resume', escalation.level, escalation.category)
        if _resume_effect is not None and _resume_effect.target_status is not None:
            _resume_target = _resume_effect.target_status
        else:
            # Defensive only: 'resume' is always a key in ACTION_EFFECTS and its
            # target_status is never None, so this branch does not execute today.
            # An explicit guard (not a bare `assert`, which `python -O` strips) so
            # a future Table B drift degrades to the historical default instead
            # of forwarding None into scheduler.set_task_status.
            logger.warning(
                "cascade-unblock: ACTION_EFFECTS has no resume target for task "
                "%s — defaulting to 'pending'", task_id,
            )
            _resume_target = 'pending'
        try:
            await self.scheduler.set_task_status(task_id, _resume_target)
            logger.info(
                'cascade-unblock: task %s flipped blocked→%s (via %s)',
                task_id, _resume_target, escalation.resolved_by,
            )
        except SetTaskStatusRejected as e:
            # Defensive TOCTOU guard: task may have transitioned to a terminal
            # status between the read and the write.
            logger.warning(
                'cascade-unblock: refused to flip %s (TOCTOU race or guard): %s',
                task_id, e,
            )

    def get_merge_halt_status(self) -> dict[str, Any]:
        """Inspect the merge queue's halt state for operator tooling."""
        if self._merge_worker is None:
            return {'wired': False}
        return {
            'wired': True,
            'halted': self._merge_worker.is_wip_halted,
            'owner_esc_id': self._merge_worker.halt_owner_esc_id,
        }

    def task_runtime_snapshot(self) -> list[TaskRuntimeState]:
        """Per-task runtime state for every active task on this host (task
        2634, PRD plans/dashboard-task-runtime-endpoint-prd.md task alpha).

        Thin delegator to :func:`orchestrator.task_runtime.build_task_runtime_snapshot`
        — synchronous and side-effect-free, reading local disk via this
        harness's own git_ops/event_store, for a later MCP tool to project
        to the dashboard's wire schema.
        """
        return build_task_runtime_snapshot(git_ops=self.git_ops, event_store=self.event_store)

    def force_unhalt_merge_queue(self, reason: str) -> dict[str, Any]:
        """Operator escape hatch for orphan halts (no escalation owns the halt).

        Refuses to act if the halt has an active owning escalation — operators
        must use ``resolve_issue(owner_esc_id)`` on those, since the legitimate
        unhalt path (``Harness._on_escalation_resolved``) is intact.
        """
        if self._merge_worker is None:
            return {'unhalted': False, 'error': 'merge worker not initialised'}
        if not self._merge_worker.is_wip_halted:
            return {'unhalted': False, 'reason': 'queue not halted'}
        owner = self._merge_worker.halt_owner_esc_id
        if owner is not None and self._escalation_queue is not None:
            try:
                esc = self._escalation_queue.get(owner)
            except Exception:
                esc = None
            if esc is not None and esc.status not in ('resolved', 'dismissed'):
                return {
                    'unhalted': False,
                    'error': (
                        f'halt is owned by active escalation {owner!r} — '
                        f'use resolve_issue({owner!r}, ...) instead'
                    ),
                    'owner_esc_id': owner,
                }
        self._merge_worker.unhalt_wip()
        logger.warning(
            'Force-unhalted merge queue (prior_owner=%s, reason=%r)',
            owner, reason,
        )
        return {'unhalted': True, 'prior_owner': owner, 'reason': reason}

    def halt_merge_queue(self, reason: str) -> dict[str, Any]:
        """Operator-initiated merge-queue halt (backs the halt_merge_queue tool).

        Halts the merger (no new merge requests are taken) AND terminates the
        post-merge verify currently running, re-queuing the affected merge so it
        re-verifies after un-halt.  Sets the same ``_wip_halt`` as the automatic
        WIP halt with no owning escalation, so ``get_merge_halt_status`` reports
        ``halted=True`` with ``owner_esc_id=None`` and the existing
        ``force_unhalt_merge_queue`` / ``unhalt_merge_queue`` cleanly reverses it
        (the active-owner refusal does not trigger).

        In-memory and transient: unlike the scheduler pause (runs.db), the merge
        WIP halt is rehydrated only from preserved WIP L1 escalations.  An operator
        halt has no owning escalation, so a process restart clears it — acceptable
        and arguably desirable (a fresh process starts un-halted).
        """
        if self._merge_worker is None:
            return {'halted': False, 'error': 'merge worker not initialised'}
        if self._merge_worker.is_wip_halted:
            return {'halted': False, 'reason': 'queue already halted'}
        self._merge_worker.operator_halt(reason)
        logger.warning('Operator-halted merge queue (reason=%r)', reason)
        return {'halted': True, 'reason': reason}

    async def force_resume_scheduler(self, reason: str) -> dict[str, Any]:
        """Operator/watcher escape hatch: resume a paused scheduler in-process.

        Backs the ``resume_scheduler`` escalation MCP tool.  Mirrors
        ``force_unhalt_merge_queue``: a structured-return, audit-logged wrapper
        over the existing ``resume_scheduler()`` (which clears the in-memory
        pause, resets the EWA if the trip was EWA-driven, clears the persisted
        pause in runs.db, and emits ``scheduler_resumed``).

        Unlike resolving the scheduler-pause L1 (which now auto-resumes via
        ``_on_escalation_resolved``), this works even when the pause has no open
        escalation — the orphan case (escalation dismissed, or filing failed).

        Idempotent: reports ``resumed=False`` when nothing was paused, but
        still calls ``resume_scheduler()`` so a stale runs.db pause row can't
        resurrect the pause on the next restart.  ``reason`` is required for
        audit.
        """
        was_paused = self.scheduler.is_paused
        prior_reason = self.scheduler.pause_reason
        await self.resume_scheduler()
        logger.warning(
            'force_resume_scheduler: was_paused=%s prior_reason=%r by_reason=%r',
            was_paused, prior_reason, reason,
        )
        return {
            'resumed': was_paused,
            'was_paused': was_paused,
            'prior_reason': prior_reason,
            'reason': reason,
        }

    async def force_halt_scheduler(self, reason: str) -> dict[str, Any]:
        """Operator-initiated scheduler halt (backs the halt_scheduler tool).

        Pauses the scheduler so ``acquire_next()`` stops dispatching new tasks,
        persists the pause to runs.db (survives restart), and emits
        ``scheduler_paused`` — but does NOT file an auto-resumable scheduler-pause
        L1 (``file_escalation=False``).  A deliberate operator halt should not
        notify the operator of their own action, and an auto-watcher resolving
        that L1 would silently undo the halt.  Reversed by
        ``force_resume_scheduler`` / ``resume_scheduler``.

        ``reason`` is required for audit.  Returns
        ``{halted, was_paused, prior_reason, reason}``.
        """
        was_paused = self.scheduler.is_paused
        prior_reason = self.scheduler.pause_reason
        await self.pause_scheduler(reason, file_escalation=False)
        logger.warning(
            'force_halt_scheduler: was_paused=%s prior_reason=%r by_reason=%r',
            was_paused, prior_reason, reason,
        )
        return {
            'halted': True,
            'was_paused': was_paused,
            'prior_reason': prior_reason,
            'reason': reason,
        }

    # ------------------------------------------------------------------ #
    # Config hot-reload (plans/config-hot-reload-prd.md, task beta)       #
    # ------------------------------------------------------------------ #

    async def reload_config(self) -> dict[str, Any]:
        """Hot-apply the allowlisted subset of orchestrator config (task beta).

        Backs the ``reload_config`` escalation MCP tool (task gamma). Orchestrates
        task alpha's pure config.py engine:

        * I1 fail-closed thread-off load — ``load_config()`` is invoked with ZERO
          arguments (off the event loop, via ``asyncio.to_thread``) so a reload can
          only ever re-read this process's own ``ORCH_CONFIG_PATH`` — never retarget
          the orchestrator at a different project. Any exception raised while
          loading (bad YAML, a failing validator, a missing file) is treated as a
          failed reload: ``self.config`` is left completely untouched and the
          failure is reported rather than propagated.
        * I4 same-turn atomic apply — ``apply_reload(self.config, fresh)`` runs
          synchronously with no interleaved await, so coroutine readers of
          ``self.config`` never observe a torn multi-field state. Its internal I5
          hybrid re-validation + rollback is relied upon as-is (not re-wrapped here).
        * config_path injection — ``apply_reload``'s return dict intentionally omits
          ``config_path``; this method supplies it from ``ORCH_CONFIG_PATH``.
        * I7 audit — every call (success or failure) is recorded as a
          ``config_reload`` event carrying the full report.

        Returns ``{reloaded, config_path, applied, restart_required, unchanged,
        error}`` verbatim from ``apply_reload`` plus the injected ``config_path``.
        """
        config_path = os.environ.get('ORCH_CONFIG_PATH')

        def _load_failure_report(error: str) -> dict[str, Any]:
            """Build+audit+warn the shared I1 fail-closed shape for a load failure."""
            report = {
                'reloaded': False,
                'config_path': config_path,
                'applied': {},
                'restart_required': {},
                'unchanged': 0,
                'error': error,
                # Always present so callers can read report['unknown_config_keys']
                # / report['ignored_config_keys'] unconditionally; a failed load
                # has no fresh config to census.
                'unknown_config_keys': [],
                'ignored_config_keys': [],
            }
            if self.event_store:
                self.event_store.emit(EventType.config_reload, data=report)
            logger.warning('config reload: %s', report['error'])
            return report

        try:
            # asyncio.wait_for cancels the *wrapping* task on timeout, but the
            # thread-pool thread actually running load_config() cannot be
            # force-cancelled — it is abandoned to finish in the background and
            # its result (or exception) is discarded. Accepted per PRD Open Q3.
            fresh = await asyncio.wait_for(
                asyncio.to_thread(load_config), timeout=_RELOAD_LOAD_TIMEOUT_SECS
            )
        except TimeoutError:
            return _load_failure_report(f'load_config timed out after {_RELOAD_LOAD_TIMEOUT_SECS}s')
        except Exception as exc:
            return _load_failure_report(str(exc))

        report = apply_reload(self.config, fresh)
        report['config_path'] = config_path
        # Surface the freshly-loaded config's unknown-key census (task 2989) so
        # the reload MCP tool reports phantom keys pydantic silently dropped.
        # apply_reload itself is left unchanged (it only diffs model_fields).
        report['unknown_config_keys'] = [
            uk._asdict() for uk in fresh.unknown_key_census
        ]
        # Keys deliberately excused by an escape hatch (reserved x_/x- prefix, or
        # an operator config_key_census.ignore entry).  Reported separately and
        # never folded into unknown_config_keys, so an over-broad glob stays
        # visible to the operator without ever reading as a failure.
        report['ignored_config_keys'] = [
            ik._asdict() for ik in fresh.ignored_key_census
        ]
        # Treat reload symmetrically with startup (INV-5, ONE implementation).
        # apply_reload copies only model_fields, so the _unknown_key_census
        # PrivateAttr would otherwise keep its stale startup value on the live
        # config; copy the freshly-loaded census across, then re-run the
        # born-at-L2 filer so a hot-reload that INTRODUCES a phantom key files an
        # L2 and a hot-reload that FIXES the config self-heals the pending one —
        # exactly as startup does.  The filer is None-safe (no-op without an
        # escalation queue) and fail-open, so it can never break a reload.
        self.config._unknown_key_census = fresh.unknown_key_census
        self.config._ignored_key_census = fresh.ignored_key_census
        await self._file_config_unknown_keys_escalation()
        if self.event_store:
            self.event_store.emit(EventType.config_reload, data=report)
        if report['restart_required'] or not report['reloaded']:
            logger.warning(
                'config reload: restart_required=%s error=%s',
                sorted(report['restart_required']), report['error'],
            )
        return report

    # ------------------------------------------------------------------ #
    # Scheduler park-and-stop pause (task 1322)                           #
    # ------------------------------------------------------------------ #

    async def pause_scheduler(self, reason: str, *, file_escalation: bool = True) -> None:
        """Pause the scheduler so acquire_next() returns None until resumed.

        1. Delegates to ``scheduler.pause(reason)`` (idempotent in-memory state).
        2. Persists via ``RunStore.save_scheduler_pause`` (best-effort).
        3. Emits ``EventType.scheduler_paused`` (best-effort).
        4. Logs a WARNING so the operator sees it.
        5. Files an auto-resumable scheduler-pause L1 — unless ``file_escalation``
           is False.

        ``file_escalation`` defaults True for every automatic trip (park-stop,
        cost-ceiling, EWA, watcher) so an AFK operator is notified.  A *deliberate*
        operator halt (``force_halt_scheduler``) passes False: notifying the
        operator of their own action is noise, and worse — the auto-watcher /
        unblock-low-risk skill could resolve that L1 and silently undo the halt.

        Called directly by sibling tasks (cost-ceiling 1323, EWA digest 1327)
        and also wired as the callback for Scheduler's park-stop trip detector.

        Race note: when this is invoked as the park-stop trip callback, the
        scheduler's synchronous latch has already set ``is_paused=True`` with
        the trip reason, so a different external caller racing in between the
        latch and this callback executing would briefly see is_paused=True
        with the trip reason but no disk row yet.  After this callback runs,
        disk and memory both reflect the trip reason — first-wins on memory,
        last-write-wins on disk for the same reason value.  Sequential
        external callers after a trip see no-op idempotent in-memory plus a
        disk overwrite with the new reason; this divergence between memory
        and disk for the in-flight pause is accepted as the cost of keeping
        the trip's status write off the synchronous critical path.
        """
        self.scheduler.pause(reason)
        logger.warning('Scheduler paused: %s', reason)

        if self._run_store and self._run_id:
            try:
                self._run_store.save_scheduler_pause(
                    project_id=self.config.fused_memory.project_id,
                    reason=reason,
                    pause_at_iso=datetime.now(UTC).isoformat(),
                    set_by_run_id=self._run_id,
                )
            except Exception:
                logger.warning('pause_scheduler: failed to persist pause state', exc_info=True)

        if self.event_store:
            self.event_store.emit(
                EventType.scheduler_paused,
                data={'reason': reason},
            )

        # File an L1 so an AFK operator is actually notified (a WARNING log +
        # timeline event are invisible otherwise).  Covers park-stop and the
        # sibling cost-ceiling / EWA / watcher halts — all route through here.
        # Deduped via has_open_l1.  Note: the park-stop latch (scheduler.py)
        # sets is_paused=True *before* this callback runs, so a was-paused
        # transition check would never fire — dedup must be queue-state based.
        # Skipped for a deliberate operator halt (force_halt_scheduler) — see
        # the file_escalation docstring above.
        if file_escalation:
            self._file_scheduler_pause_escalation(reason)

    async def resume_scheduler(self) -> None:
        """Clear the scheduler pause so acquire_next() resumes dispatching.

        1. Delegates to ``scheduler.resume()`` (idempotent).
        2. Clears persistence via ``RunStore.clear_scheduler_pause`` (best-effort).
        3. Emits ``EventType.scheduler_resumed`` (best-effort).
        4. If the pause was caused by an EWA trip (reason starts with 'ewa_trip_'),
           resets ``_ewa_value`` to 0.0 so the next digest step does not immediately
           re-trigger a pause.  EWA decays naturally with alpha=0.3 (~N digests of
           low ratio to fall back below threshold), but an operator-driven resume
           signals that the situation is under control — a clean reset is less
           surprising than watching the EWA trip instantly again.
        5. Logs INFO.
        """
        # Snapshot pause reason BEFORE resume() clears it.
        prior_reason = self.scheduler.pause_reason or ''
        self.scheduler.resume()
        if prior_reason.startswith('ewa_trip_'):
            self._ewa_value = 0.0
            logger.info(
                'resume_scheduler: EWA value reset to 0.0 (prior pause was ewa_trip).'
            )
        logger.info('Scheduler resumed.')

        if self._run_store:
            try:
                self._run_store.clear_scheduler_pause(
                    self.config.fused_memory.project_id,
                )
            except Exception:
                logger.warning('resume_scheduler: failed to clear persisted pause', exc_info=True)

        if self.event_store:
            self.event_store.emit(EventType.scheduler_resumed)

    async def _load_persisted_scheduler_pause(self) -> None:
        """Restore scheduler pause state from runs.db on restart.

        Called once from ``Harness.run()`` after the RunStore is initialised.
        If a pause record exists, the scheduler is paused in-memory using
        ``scheduler.pause()`` directly — NOT ``pause_scheduler()`` — so that
        no duplicate persistence write or event emission occurs (the row is
        already on disk from the prior run that set it).

        Logs a WARNING with the persisted reason and pause_at so the operator
        is alerted on startup.  Any failure is caught and logged but never
        blocks startup.
        """
        if not self._run_store:
            return
        try:
            record = self._run_store.load_scheduler_pause(
                self.config.fused_memory.project_id,
            )
            if record:
                reason = record.get('reason', '<unknown reason>')
                pause_at = record.get('pause_at', '<unknown time>')
                restored_from_run_id = record.get('set_by_run_id', '<unknown>')
                logger.warning(
                    'Scheduler pause persisted from prior run — restoring. '
                    'reason=%r  pause_at=%r  set_by_run_id=%r  '
                    '(call Harness.resume_scheduler() to clear)',
                    reason,
                    pause_at,
                    restored_from_run_id,
                )
                self.scheduler.pause(reason)
                # Stash the reason so run() can file the L1 escalation once the
                # escalation queue exists (this runs at line ~492, before
                # _start_escalation_server creates _escalation_queue).  See
                # _file_restored_pause_escalation.
                self._restored_pause_reason = reason
                # Emit a distinct event so the timeline self-documents
                # cross-run continuity.  Operators querying the event log
                # for a run that starts with dispatch halted can see WHY
                # without having to cross-reference the previous run_id.
                if self.event_store:
                    self.event_store.emit(
                        EventType.scheduler_pause_restored,
                        data={
                            'reason': reason,
                            'pause_at': pause_at,
                            'restored_from_run_id': restored_from_run_id,
                        },
                    )
        except Exception:
            logger.warning(
                '_load_persisted_scheduler_pause: failed to read pause state',
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # Digest + EWA trip (task 1327)                                       #
    # ------------------------------------------------------------------ #

    async def _maybe_write_digest(self) -> None:
        """Check if it is time to write a digest; if so gather data, write, and update EWA.

        Called from _watcher_supervisor_loop after each rotation (best-effort).
        Any exception is swallowed and logged as a WARNING — a failed digest must
        never break the watcher supervisor loop.

        Algorithm:
        1. Early-return when digest_enabled=False.
        2. Snapshot _escalation_event_count (best-effort; see note below).
        3. Early-return when (snapshot - last_count) < N.
        4. Snapshot escalation delta.
        5. Compute window (last window_end → now).
        6. Aggregate escalation stats (fail-open).
        7. Count done tasks in EventStore (fail-open).
           done_count from EventStore is the single source of truth for both
           the rendered digest figure and the update_ewa input — ensuring the
           operator sees exactly the number that drove the EWA decision.
        8. Query cost stats from CostStore (fail-open).
        9. Read parked-task counts from Scheduler state.
        10. Update EWA using EventStore done_count.
        11. Compute trip flag and anomaly flags.
        12. Write digest file (fail-open via write_digest_entry).
        13. Advance counters/state using the snapshot from step 2 (not the
            live counter), so that concurrent callbacks firing inside this
            function do not silently skip counted events.
        14. If tripped and not already paused, call pause_scheduler (post-write).

        Note on _escalation_event_count: callbacks fire inline on the asyncio
        event loop thread, so there are no real concurrent writers — the
        snapshot at step 2 guards against the logical interleaving where a
        callback runs at an await point inside this function, not against torn
        integer writes.  Snapshotting once at step 2 makes the threshold check
        and the advance consistent — concurrent callbacks cannot cause a
        "double-skip" where the advance overshoots the events that triggered
        this digest.  The counter is best-effort observability; a small drift
        is acceptable.

        Task 1327 AFK hardening.
        """
        try:
            # (1) Early-return if disabled.
            if not self.config.digest_enabled:
                return

            # (2) Snapshot _escalation_event_count so the threshold check (3)
            # and the advance (13) are consistent even if a concurrent callback
            # increments the live counter between those two reads.
            event_count_snapshot = self._escalation_event_count

            # (3) Early-return if not enough new events.
            diff = event_count_snapshot - self._last_digest_event_count
            if diff < self.config.digest_every_n_escalations:
                return

            # (4) Snapshot escalation delta.
            escalations_in_step = diff

            # (5) Compute window timestamps.
            window_end = datetime.now(UTC).isoformat()
            window_start = self._last_digest_window_end_iso
            if not window_start:
                # First digest: use 24h ago as window start.
                window_start = (datetime.now(UTC) - timedelta(hours=24)).isoformat()

            # (6) Gather escalation stats (fail-open via aggregate_escalations).
            escalations_dir = (
                Path(self.config.project_root) / self.config.escalation.queue_dir
            )
            escalation_stats = digest_mod.aggregate_escalations(
                escalations_dir, window_start, window_end
            )

            # (7) Count done tasks in window (fail-open via count_done_in_window).
            events_db_path = self.event_store.db_path if self.event_store else None
            done_count = (
                digest_mod.count_done_in_window(
                    events_db_path, window_start, window_end
                )
                if events_db_path is not None
                else 0
            )

            # (8) Cost stats (fail-open via cost_in_window).
            cost_stats = await digest_mod.cost_in_window(
                self.cost_store, window_start, window_end
            )

            # (8b) Per-(model×role) outcome rollup (task 2534 δ, boundary test
            # 12). events_db_path (step 7) is the same runs.db file backing
            # CostStore/RunStore in production (harness.py:1373-1398), so no
            # separate DB handle is needed. Fail-open: empty rollup when the
            # events DB path is unavailable (model_role_rollup itself is
            # fail-open on a missing/unreadable DB).
            model_role_rollup = (
                digest_mod.model_role_rollup(
                    events_db_path, window_start, window_end
                )
                if events_db_path is not None
                else digest_mod.ModelRoleRollup()
            )

            # (9) Parked-task counts via public Scheduler properties (task 1327).
            parked_live = self.scheduler.parked_live_count
            parked_window_churn = self.scheduler.parked_window_churn_count

            # (10) Update EWA — done_count from EventStore (step 7) is the single
            # source of truth so the EWA input matches the rendered digest figure.
            new_ewa = digest_mod.update_ewa(
                prev_ewa=self._ewa_value,
                escalations_in_step=escalations_in_step,
                done_in_step=done_count,
                alpha=self.config.digest_ewa_alpha,
            )

            # (11) Trip flag and anomaly flags.
            tripped = new_ewa >= self.config.digest_ewa_threshold
            anomaly_flags = {
                'cost_spike': (
                    cost_stats.watcher_cost_in_window
                    > self.config.watcher_daily_cost_ceiling_usd * 0.5
                ),
                'park_spike': (
                    parked_window_churn >= self.config.park_stop_parked_threshold
                ),
                'ewa_above_threshold': tripped,
                'infra_dedupe_active': escalation_stats.dedupe_children_total > 0,
            }

            # (12) Resolve digest directory, assemble inputs, and write digest file
            # (fail-open via write_digest_entry — never raises).
            if self.config.digest_dir:
                digest_dir = Path(self.config.digest_dir)
            else:
                digest_dir = Path(self.config.project_root) / 'data' / 'digests'

            # (11b) Stale-lane census (leaf γ, task 2891) — best-effort; the
            # method is already fail-safe (returns [] on a missing pool or a
            # degraded status read) and runs inside this try/except so a census
            # hiccup can never break the digest write.
            stale_lane_census = await self._stale_lane_assignment_census()

            inputs = digest_mod.DigestInputs(
                window_start_iso=window_start,
                window_end_iso=window_end,
                escalation_stats=escalation_stats,
                done_count=done_count,
                cost_stats=cost_stats,
                parked_live=parked_live,
                parked_window_churn=parked_window_churn,
                ewa_value=new_ewa,
                ewa_threshold=self.config.digest_ewa_threshold,
                tripped=tripped,
                anomaly_flags=anomaly_flags,
                watcher_clusters=[],
                dry_run_proposals=[],
                model_role_rollup=model_role_rollup,
                stale_lane_census=stale_lane_census,
            )

            digest_mod.write_digest_entry(digest_dir, inputs)

            # (13) Advance EWA state and counters.
            # Use event_count_snapshot (not the live self._escalation_event_count)
            # so that concurrent callbacks that fired after the snapshot are not
            # silently skipped — they will be counted in the next digest step.
            self._ewa_value = new_ewa
            self._last_digest_event_count = event_count_snapshot
            self._last_digest_window_end_iso = window_end

            # (14) EWA trip: pause scheduler AFTER the digest is written so the
            # markdown captures the trip-causing state.
            if tripped and not self.scheduler.is_paused:
                await self.pause_scheduler(f'ewa_trip_{new_ewa:.4f}')

        except asyncio.CancelledError:
            raise
        except AttributeError:
            raise  # task 1449: surface fixture-drift / state-init bugs to tests
        except Exception:
            logger.warning('_maybe_write_digest: unexpected error (fail-open)', exc_info=True)
