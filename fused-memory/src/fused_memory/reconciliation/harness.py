"""Pipeline orchestrator — runs the three-stage reconciliation pipeline."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from shared.cli_invoke import AllAccountsCappedException, read_transcript_records
from shared.config_dir import TaskConfigDir
from shared.usage_gate import UsageGate

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.reconciliation import (
    AssembledPayload,
    ReconciliationEvent,
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
    StageReport,
)
from fused_memory.models.scope import (
    ProjectId,
    ProjectRoot,
    ProjectScope,
    build_known_projects_map,
)
from fused_memory.reconciliation.active_runs import ActiveRunRegistry
from fused_memory.reconciliation.backlog_policy import BacklogPolicy
from fused_memory.reconciliation.cli_stage_runner import (
    gc_run_config_dir,
    recon_config_base_dir,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.judge import Judge
from fused_memory.reconciliation.mem0_dedup import find_prior_memory
from fused_memory.reconciliation.policies import is_snapshot_write_blocked
from fused_memory.reconciliation.queue_health import summarize_graphiti_queue_health
from fused_memory.reconciliation.scope_freshness import (
    ScopeFreshnessResult,
    compute_scope_signature,
    precheck_scope_correction_freshness,
)
from fused_memory.reconciliation.stages.memory_consolidator import (
    MemoryConsolidator,
    write_stage1_cycle_summary,
)
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)
from fused_memory.reconciliation.stats_verifier import verify_and_rewrite_stats
from fused_memory.reconciliation.task_count_snapshot_cadence import (
    ESCALATION_CATEGORY as TASK_COUNT_SNAPSHOT_ESCALATION_CATEGORY,
)
from fused_memory.reconciliation.task_count_snapshot_cadence import (
    TASK_COUNT_SNAPSHOT_MISS_THRESHOLD,
    build_stale_snapshot_finding,
    evaluate_snapshot_cadence,
    extract_snapshot_written,
)
from fused_memory.reconciliation.task_filter import (
    FilteredTaskTree,
    cross_verify_task_counts,
    diff_status_correction,
    filter_task_tree,
)
from fused_memory.services.live_workflow_detector import is_workflow_live_for_task
from fused_memory.services.memory_service import MemoryService

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol

try:
    from escalation.dedupe import (  # type: ignore[import-untyped]
        DedupeConfig,
        compute_content_fingerprint,
        submit_or_dedupe,
    )
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import (  # type: ignore[import-untyped]
        EscalationQueue,
        iter_all_escalation_paths,
    )
    from escalation.server import (  # type: ignore[import-untyped]
        create_server as create_escalation_server,
    )
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

# Recon-wide dedup config: covers all four recon escalation categories.
# Wider than DedupeConfig.for_recon() (which only covers recon_integrity_issue)
# because A7b also folds non-finding categories so each DISTINCT recurring message
# files once — see design_decisions in plan.json for rationale.
# Set to None when escalation package is not installed; _escalate checks
# HAS_ESCALATION before using it.
#
# Escalation-closure contract (A7b / plans/afk-A7-recon-closure.md):
#   - The reconciliation harness NEVER calls queue.resolve() ON THE RECON
#     ESCALATION QUEUE (config.escalation_queue_dir).
#   - The watcher session (port 8103) is the sole closer of recon escalations.
#     Scope note (task 2998): the judge-halt record BacklogPolicy writes to
#     <project_root>/data/escalations/ lives in a DIFFERENT queue and is closed
#     by BacklogPolicy.on_judge_unhalt — write and close stay with the class
#     that owns that record, so this invariant is not contradicted.
#   - Dedup folds on the way IN only, via submit_or_dedupe + _RECON_DEDUP_CONFIG.
#   See ReconciliationHarness._escalate() docstring for per-call-site details.
_RECON_DEDUP_CONFIG = (
    dataclasses.replace(
        DedupeConfig.for_recon(),  # type: ignore[possibly-undefined]
        infra_dedupe_categories=(
            'recon_integrity_issue',
            'recon_failure',
            'recon_stale_run',
            'recon_backlog_overflow',
            # Task 1755 / PRD β: aggregate storm alarm for dead_owner_shielded
            # suppression bursts.  Adding it here ensures submit_or_dedupe folds
            # repeated storm alarms (stable _DEAD_OWNER_STORM_FINDING fingerprint)
            # into a single pending escalation for the 8103 watcher.
            'recon_watchdog_kill_storm',
            # Task 1970 amendment: aggregate storm alarm for referenceless
            # placeholder-finding drops (see _PLACEHOLDER_DROP_STORM_FINDING /
            # _record_placeholder_finding_drop).  Same fold rationale as
            # recon_watchdog_kill_storm above.
            'recon_remediation_placeholder_storm',
            # Task 2278: stable per-project finding identity (build_stale_snapshot_finding)
            # so a sustained task_count_snapshot cadence gap folds into a single pending
            # escalation per project instead of firing once per cycle.
            TASK_COUNT_SNAPSHOT_ESCALATION_CATEGORY,
        ),
    )
    if HAS_ESCALATION else None
)

logger = logging.getLogger(__name__)

# Task 2708 amendment (reviewer_comprehensive): bound the startup judge-recovery
# fan-out. Normal operation leaves at most a handful of judge_pending markers
# (the cycle-end→verdict-commit window is tiny), but a persistently-failing
# judge never clears its marker — _run_judge's `except Exception` path logs and
# returns WITHOUT reaching add_verdict's atomic clear — so markers accumulate
# and would otherwise ALL re-fire concurrently on every restart: a
# thundering-herd of LLM calls. Cap the per-restart spawn; the un-spawned
# remainder self-heals on later restarts (its markers persist until a verdict
# commits). Exceeding the cap is WARN-logged so a wedged judge is visible rather
# than silently re-run each restart.
_JUDGE_RECOVERY_MAX_SPAWN = 20

# Task 1755 / PRD β: stable finding identity for the dead_owner_shielded storm alarm.
# The fingerprint is keyed on finding['category'] / affected_ids / description
# (harness.py:_escalate), so keeping these fields constant — and putting all
# variable data (count, window, affected projects) only in summary/detail — ensures
# submit_or_dedupe folds repeated storm alarms into a single pending escalation.
_DEAD_OWNER_STORM_FINDING: dict[str, Any] = {
    'category': 'recon_watchdog_kill_storm',
    'affected_ids': ['dead_owner_shielded_suppression_storm'],
    'description': 'dead_owner_shielded recon_stale_run suppression storm',
}

# Task 1512: minimum number of completed runs that must contain a finding
# before it is escalated from _run_remediation_pass.  Below this count the
# finding is suppressed (emitting a structured log instead) on the grounds
# that remediation is likely to fix it on the next attempt.
#
# Counting note: each reconciliation cycle produces TWO completed runs — the
# parent full run (persisted at run_full_cycle ~line 1065) and the remediation
# run (persisted at _run_remediation_pass ~line 1405).  Both are counted by
# _finding_persistence_count because both are 'completed' journal rows.  A
# threshold of 4 therefore fires after 2 complete reconciliation cycles that
# both flagged the same finding — short enough that a genuinely broken finding
# escalates within a watchable window (≈ 10–180 minutes depending on cycle
# duration), long enough to filter transient findings.
_INTEGRITY_FINDING_RECURRENCE_THRESHOLD = 4

# Task 1669: suppress re-firing of a finding whose matching escalation was
# resolved within this window.  Beyond it, a recurrence re-escalates so a
# re-emerging problem is not hidden forever.  Value is a policy threshold —
# the exact constant is not load-bearing for tests (60s vs 8d test points sit
# robustly inside/outside any reasonable 24h window).
_RESOLVED_RECURRENCE_WINDOW_SECONDS = 86400  # 24h

# Task 1970 amendment (reviewer_comprehensive): coarse safety net for a
# runaway Stage 3 that stops citing anything.  Each individual referenceless-
# finding drop in _maybe_remediate is only logged
# (reconciliation.remediation_dropped_placeholder_finding) and, being noise
# rather than a human-actionable integrity issue, is deliberately never
# escalated on its own — see _maybe_remediate.  That means a systemic Stage 3
# regression (e.g. it stops calling cite_* entirely) would otherwise be
# invisible outside logs, and can no longer accumulate toward
# _INTEGRITY_FINDING_RECURRENCE_THRESHOLD above, since a dropped placeholder
# never enters a remediation run.  This rolling-window counter closes that
# gap by firing ONE coarse escalation once drops recur this often for the
# same project within the window.  Mirrors the dead_owner_shielded
# suppression-storm counter (_record_dead_owner_suppression /
# dead_owner_suppression_storm_threshold+window_seconds below) but as plain
# module constants rather than ReconciliationConfig fields, since this
# predicate and its guard are private to this module.
_PLACEHOLDER_DROP_STORM_THRESHOLD = 5
_PLACEHOLDER_DROP_STORM_WINDOW_SECONDS = 3600.0  # 1h

# Throttle window for the '_notify_judge_halt escalated to nothing' WARNING.
# Task 2998 deliberately stopped burning the per-process dedupe token on a
# failed write, so a halted project with no live orchestrator re-enters that
# path on EVERY halted tick (~5s) — unthrottled that is ~17k WARNING
# lines/day/project, drowning the signal it exists to raise. Matches the
# BacklogPolicy rejection-branch throttle and the default escalation
# rate-limit window; the first failure per halt always warns.
_HALT_ESCALATION_WARN_INTERVAL_SECONDS = 900.0  # 15m

# Stable finding identity for the placeholder-drop storm alarm — same
# fingerprint-stability rationale as _DEAD_OWNER_STORM_FINDING above.
_PLACEHOLDER_DROP_STORM_FINDING: dict[str, Any] = {
    'category': 'recon_remediation_placeholder_storm',
    'affected_ids': ['remediation_placeholder_drop_storm'],
    'description': (
        'Stage 3 is repeatedly filing actionable findings with no '
        'task/entity/edge/memory citation'
    ),
}


def _derive_affected_ids(finding: dict) -> list[str]:
    """Derive an affected-ids-equivalent list from a finding's typed citations.

    The recon_report cutover (task γ) retired the free-form ``affected_ids``
    field in favour of typed citation lists — ``cited_entities`` ({entity_uuid,
    canonical_name}), ``cited_edges`` ({edge_uuid, ...}), ``cited_tasks``
    ({project_id, task_id, ...}), ``cited_memories`` ({memory_id, ...}).  This
    helper flattens those typed citations back into a flat list of identity
    strings so the escalation dedup/recurrence fingerprint
    (``compute_content_fingerprint``) and the log/detail payloads keep a stable,
    structured identity component instead of degrading to description-only.

    A legacy ``affected_ids`` field takes precedence when present, so cross-run
    recurrence counting still works against pre-cutover journal rows that carry
    the old shape.
    """
    legacy = finding.get('affected_ids')
    if legacy:
        return [str(a) for a in legacy]
    ids: list[str] = []
    for c in finding.get('cited_tasks') or []:
        tid = c.get('task_id') if isinstance(c, dict) else None
        if tid:
            ids.append(str(tid))
    for c in finding.get('cited_entities') or []:
        if isinstance(c, dict):
            val = c.get('canonical_name') or c.get('entity_uuid')
            if val:
                ids.append(str(val))
    for c in finding.get('cited_edges') or []:
        eid = c.get('edge_uuid') if isinstance(c, dict) else None
        if eid:
            ids.append(str(eid))
    for c in finding.get('cited_memories') or []:
        mid = c.get('memory_id') if isinstance(c, dict) else None
        if mid:
            ids.append(str(mid))
    return ids


def _finding_has_reference(finding: dict) -> bool:
    """Return True iff *finding* cites at least one task/entity/edge/memory ID.

    A finding whose ``_derive_affected_ids`` is empty (no legacy ``affected_ids``
    and no typed citation carries a usable identity) references nothing
    concrete — it is a synthetic/placeholder finding (e.g. Stage 3 filed
    ``add_finding`` but never followed up with a ``cite_*`` call) that cannot
    be investigated or remediated.  Reusing ``_derive_affected_ids`` keeps
    reference-derivation single-sourced with the escalation dedup/fingerprint
    logic, so this predicate can never disagree about what counts as a
    reference.

    Task 1970: used by ``_maybe_remediate`` to drop referenceless actionable
    findings before they reach the production remediation batch.
    """
    return bool(_derive_affected_ids(finding))


# Module-local sleep binding — allows tests to patch sleep without touching
# the global asyncio namespace.
_sleep = asyncio.sleep


class UnknownProjectError(ValueError):
    """Raised when a project_id has no entry in the KNOWN_PROJECT_ROOTS registry.

    Introduced by task 1143 as the pre-flight cross-contamination guard.

    Design notes:
    (a) Signals a KNOWN_PROJECT_ROOTS misconfiguration — the project_id is not
        registered via ``taskmaster.project_root`` or ``DASHBOARD_KNOWN_PROJECT_ROOTS``.
    (b) Subclasses ``ValueError`` for backward-compat: any existing or future
        ``except ValueError`` callsite (test code, callers of
        ``_known_project_scope_for``) continues to match.
    (c) Exists as a distinct type so ``_project_loop`` can narrowly catch ONLY
        misconfiguration and let unrelated ``ValueError``s fall through to the
        existing ``except Exception`` retry path:
        - ``stages/base.py:108`` watermark↔stage project_id mismatch — transient
          during instance handover; naturally recoverable on the next cycle.
        - ``stages/memory_consolidator.py:96`` unset episode_limit/memory_limit —
          programming bug; should surface and retry, not abort the project loop.
    """


@dataclass
class TierConfig:
    """Model tier configuration for a reconciliation cycle."""

    model: str = 'sonnet'
    episode_limit: int = 125
    memory_limit: int = 250


def build_stale_run_diagnostics(
    run: ReconciliationRun,
    lock_holder: str | None,
    lock_age: float | None,
    cutoff: float,
) -> dict:
    """Return a diagnostic dict for a stale run's lock disposition.

    Classifies the disposition into one of five cases:

    - ``pre_migration``       — run.instance_id is None (legacy row)
    - ``no_lock``             — no lock row exists for the project
    - ``handed_off``          — lock held by a different instance
    - ``dead_owner_shielded`` — same iid as run, heartbeat older than cutoff
    - ``live``                — same iid as run, heartbeat fresh (within cutoff)

    Also includes: project_id, run_type (str), instance_id, age_seconds
    (seconds since run.started_at), lock_holder, lock_heartbeat_age.
    """
    now = datetime.now(UTC)
    # Normalize naive datetimes before subtracting — mirrors the guard in get_lock_status.
    started = run.started_at if run.started_at.tzinfo else run.started_at.replace(tzinfo=UTC)
    age_seconds = (now - started).total_seconds()

    if run.instance_id is None:
        disposition = 'pre_migration'
    elif lock_holder is None:
        disposition = 'no_lock'
    elif lock_holder != run.instance_id:
        disposition = 'handed_off'
    elif lock_age is not None and lock_age > cutoff:
        # Same iid, but heartbeat is older than the liveness threshold.
        disposition = 'dead_owner_shielded'
    else:
        # Same iid, heartbeat is fresh — owner is still alive.
        disposition = 'live'

    return {
        'disposition': disposition,
        'project_id': run.project_id,
        'run_type': str(run.run_type.value) if isinstance(run.run_type, RunType) else str(run.run_type),
        'instance_id': run.instance_id,
        'age_seconds': age_seconds,
        'lock_holder': lock_holder,
        'lock_heartbeat_age': lock_age,
    }


def _stage1_ledger_write_missing(report: object) -> bool:
    """Return True iff *report* is a Stage 1 report whose own in-stage
    ``cycle_summary`` ledger upsert failed (task 2734).

    Backs arm 2 of :meth:`ReconciliationHarness._ensure_stage1_cycle_summary`
    — the "Stage 1 completed but its own write failed" case, as opposed to
    task 2440's arm 1 ("Stage 1 raised before producing a report at all").

    Keys on the EXPLICIT failure value
    ``stats['stage1_cycle_summary_ledger_written'] == 0``, deliberately
    never ``!= 1``: a real full-cycle
    :meth:`~fused_memory.reconciliation.stages.memory_consolidator.MemoryConsolidator.run`
    always stamps this stat to 0 (in-stage write failed) or 1 (succeeded),
    so ``== 0`` captures exactly the defect. An ABSENT stat is produced only
    by test stubs / reports that never reached the write — treating an
    absent stat as no-fire (rather than ``!= 1``, which an absent stat would
    also satisfy) keeps every stubbed-Stage-1 test (``stats={}``) and the
    happy path (``stat == 1``) untouched.

    Also excludes arm 1's own harness-synthesized degraded report (stamped
    ``stats['stage1_cycle_summary_degraded_backstop'] = True``), so the two
    arms of ``_ensure_stage1_cycle_summary`` can never double-process the
    same run.

    Returns False for anything whose ``.stats`` isn't a dict — including a
    non-``StageReport`` object (e.g. a plain dict, the shape
    ``run.stage_reports['_error']`` entries use) — since
    ``run.stage_reports`` is typed ``dict[str, StageReport | dict]`` and
    this predicate must never raise when handed one of those.
    """
    stats = getattr(report, 'stats', None)
    if not isinstance(stats, dict):
        return False
    if stats.get('stage1_cycle_summary_degraded_backstop') is True:
        return False
    return stats.get('stage1_cycle_summary_ledger_written') == 0


class ReconciliationHarness:
    """Orchestrates the three-stage reconciliation pipeline."""

    # Declared, never assigned in __init__: production builds a fresh list per
    # cycle via _make_stages(scope) (task 2146 β, decision 5). Tests assign this
    # as a convenience instance attribute (harness.stages = harness._make_stages(scope))
    # to keep their harness.stages[N] access pattern — this annotation is what lets
    # pyright resolve that access.
    #
    # Known trade-off (task 2146 review follow-up): this couples dozens of test
    # call sites to the private _make_stages signature via monkeypatching. A
    # dedicated test factory/fixture (e.g. make_pinned_harness) that builds and
    # pins stages would centralize that contract instead — deferred, since
    # migrating those call sites spans files/ownership beyond a focused amendment.
    stages: list[Any]

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskBackendProtocol | None,
        journal: ReconciliationJournal,
        event_buffer: EventBuffer,
        config: FusedMemoryConfig,
        backlog_policy: BacklogPolicy | None = None,
        known_projects: dict[str, str] | None = None,
        recon_report_state=None,
        server_ready_event: asyncio.Event | None = None,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.buffer = event_buffer
        # PRD γ recon_report threading: capture port and state before config is
        # narrowed to config.reconciliation so stages can reach the full config values.
        self._recon_report_state = recon_report_state
        # Reviewer finding race_condition: run_loop() awaits this event before the
        # first stage subprocess fires, ensuring the recon-report MCP server is
        # accepting connections.  None when no recon_report server is started.
        self._server_ready_event = server_ready_event
        self._recon_report_port: int = config.server.recon_report_port
        _raw_root = config.taskmaster.project_root if config.taskmaster else ''
        if _raw_root:
            _raw_root = os.path.expanduser(_raw_root)
        self._project_root = str(Path(_raw_root).resolve()) if _raw_root else ''
        self.config = config.reconciliation
        self._backlog_policy = backlog_policy
        # Multi-project routing: build a {project_id → project_root} map from
        # the configured project + DASHBOARD_KNOWN_PROJECT_ROOTS env var so
        # Stage 2 can surface other projects to its LLM and authorise
        # cross-project task filing.  Reuses the dashboard's existing env var
        # to avoid two sources of truth — see plans/deep-squishing-lagoon.md
        # for the trade-off (rename to a neutral name is a tracked followup).
        # When a known_projects kwarg is supplied (server startup), the harness
        # uses that snapshot instead of building its own — single source of
        # truth across harness, ticket janitor, and path-scope guard (task 1164).
        self._known_projects: dict[str, str] = (
            dict(known_projects) if known_projects is not None
            else build_known_projects_map(self._project_root)
        )
        # Task 2998 GAP A: seed the policy's project_root cache NOW, at
        # construction.  Its only other writer is BacklogPolicy.check(), which
        # runs on a mutating MCP call — so a halt rehydrated by
        # Judge.initialize() fires _notify_judge_halt before any root has been
        # registered, _route_over_limit falls to the rejection branch, and
        # NOTHING is written (the 48h reify incident: 0 of 96 escalation files
        # carried ReconciliationJudgeHalted, and no log line of any kind).
        if self._backlog_policy is not None and self._known_projects:
            self._backlog_policy.register_known_project_roots(self._known_projects)
        # WP-D: track which halted projects we've already escalated so we
        # don't re-fire every harness tick.
        self._halt_escalated: set[str] = set()
        # Throttle clock (monotonic) for the 'escalated to nothing' WARNING,
        # per project. Cleared on a successful escalation and on unhalt, so the
        # first failure of the NEXT halt is always loud.
        self._halt_escalation_warn_ts: dict[str, float] = {}
        # Task 2998: escalation ids closed by unhalts not yet reported to the
        # MCP tool layer. Pop-once — see take_resolved_halt_escalations.
        self._resolved_halt_escalations: dict[str, list[str]] = {}

        # Task 1755 / PRD β, amended by task 2039: rolling-window counter of
        # DISTINCT dead-owner instance UUIDs among dead_owner_shielded
        # recon_stale_run suppressions.  Each entry is
        # (timestamp, project_id, instance_id); pruned on each call to
        # _record_dead_owner_suppression().  The count that matters is the
        # number of distinct non-None instance_id values in the window, NOT
        # the number of suppression events.
        #
        # ⚠ In-process lifetime limitation: these counters are reset on every
        # harness restart.  A single restart recovers one dead_owner_shielded
        # orphan PER PROJECT it owns, all left by the SAME prior (dead)
        # incarnation — every one of those orphans carries that one dead
        # owner's instance_id, so the distinct count stays 1 regardless of
        # how many projects the project registry holds, and a benign restart
        # never crosses the threshold of 6 (task 2039 regression:
        # esc-recon-50da2482-1 — a single restart across 6 registered
        # projects previously false-fired because the old counter counted
        # per-project suppressions, i.e. count=6, instead of distinct dead
        # owners, i.e. count=1).  The storm alarm instead fires when
        # >=threshold GENUINELY DISTINCT dead-owner instances — independent
        # watchdog kills — are recovered WITHIN a single harness lifetime,
        # e.g. the 2026-06-15 event that triggered this task (~one kill every
        # ~10 min, ~6 distinct incarnations per rolling 60-minute window).
        # Single-restart churn is instead observable via the per-event INFO
        # log emitted at harness.py:741.  If single-owner restart churn must
        # also alarm, count recent dead_owner_shielded _error records from
        # the journal over the window instead of the in-memory deque.
        self._dead_owner_suppressions: deque[tuple[datetime, str, str | None]] = deque()
        # Timestamp of the last storm escalation — None means never fired.
        self._last_suppression_storm_escalation_at: datetime | None = None

        # Task 1970 amendment: rolling-window counter for dropped
        # referenceless actionable findings (see
        # _record_placeholder_finding_drop / _maybe_remediate).  Same
        # (timestamp, project_id) shape, in-process-lifetime caveat, and
        # rate-limited single-fire semantics as _dead_owner_suppressions above.
        self._placeholder_finding_drops: deque[tuple[datetime, str]] = deque()
        self._last_placeholder_drop_storm_escalation_at: datetime | None = None

        # Task σ / 2717: rolling-window per-event counter of unresumable/failed
        # interrupted-run resume attempts (config-driven
        # resume_failure_storm_threshold / _window_seconds).  Same
        # (timestamp, project_id) shape, in-process-lifetime caveat, and
        # rate-limited single-fire semantics as _placeholder_finding_drops above.
        # Fed from _resume_interrupted_runs' failed+restore fallback arm so a
        # persistent resume failure (prompt/tool drift, stale transcripts) surfaces
        # ONE loud recon_resume_failure_storm escalation instead of silent churn.
        self._resume_failures: deque[tuple[datetime, str]] = deque()
        self._last_resume_failure_storm_escalation_at: datetime | None = None

        # Usage gate (multi-account cap failover)
        self.usage_gate: UsageGate | None = None
        if hasattr(self.config, 'usage_cap') and self.config.usage_cap.enabled:
            self.usage_gate = UsageGate(self.config.usage_cap)

        # Judge — receives a callback that clears _halt_escalated so a
        # subsequent halt in the same process re-fires the escalation.
        self.judge = (
            Judge(
                self.config, journal,
                usage_gate=self.usage_gate,
                on_unhalt_cb=self._on_judge_unhalt,
            )
            if self.config.judge_enabled else None
        )

        # Escalation support
        self._escalation_queue: EscalationQueue | None = None
        self._escalation_task: asyncio.Task | None = None
        self._escalation_url: str | None = None

        # Per-project concurrent loops
        self._project_tasks: dict[str, asyncio.Task] = {}

        # Fire-and-forget judge review tasks (task 2708). Held so run_loop's
        # shutdown finally can cancel + await them (see _drain_judge_tasks) and
        # so orphaned tasks don't trigger 'Task was destroyed but it is pending'.
        # DELIBERATELY separate from _active_runs / recon_busy_snapshot: the judge
        # stays excluded from the cycle-aware-restart deferral signal (task 2703 δ);
        # a durable judge_pending marker + startup re-run make that accepted
        # cancellation non-lossy instead.
        self._judge_tasks: set[asyncio.Task] = set()

        # Drain mode: stop starting new cycles, let current ones finish
        self._draining: bool = False
        # One-shot gate shared by drain() and run_loop()'s drain-status block.
        # Whichever site fires the 'Harness fully drained' marker first sets this
        # to True so subsequent drain() calls and main-loop iterations are silent.
        # Lifecycle is tied to process lifetime: there is no undrain path today,
        # but if one is ever added it MUST reset _drain_complete_logged together
        # with _draining, otherwise the next drain will silently fail to re-emit
        # the 'Harness fully drained' marker.
        self._drain_complete_logged: bool = False

        # In-flight full-cycle registry — source of the machine-readable
        # `recon_busy` signal on /health (task 2703 δ). Updated only inside
        # run_full_cycle via `with self._active_runs.track(...)`, so an entry
        # clears on every exit path (return / Exception / the CancelledError a
        # drain or timeout raises mid-stage), never leaking a phantom-busy run.
        self._active_runs = ActiveRunRegistry()

    async def _notify_judge_halt(self, project_id: str, reason: str) -> None:
        """WP-D: forward judge halts to the backlog policy exactly once.

        Routes to escalation when an orchestrator is live for this project.
        When it does not — the rejection branch, or a rate-limited call that
        wrote no file — the halt reaches NO caller: the ``ReconciliationJudgeHalted``
        verdict returned here is not surfaced anywhere (the next mutating MCP
        call runs ``BacklogPolicy.check()``, which re-evaluates the BACKLOG
        condition and yields ``ReconciliationBacklogExceeded`` instead). The
        only fallback is the retry below: the dedupe sentinel is claimed ONLY
        on a verdict that actually wrote an escalation file, so a failed
        attempt is retried on the next halted tick (~5s) rather than burning
        the single per-process token (task 2998 GAP B / 2b).

        Best-effort: a failure here must not break the harness loop.
        """
        if self._backlog_policy is None or project_id in self._halt_escalated:
            return
        try:
            verdict = await self._backlog_policy.on_judge_halt(project_id, reason)
        except Exception:
            logger.exception(
                'harness: backlog_policy.on_judge_halt raised for %s', project_id,
            )
            return

        if (
            verdict is not None
            and verdict.outcome == 'escalated'
            and verdict.escalation_path is not None
        ):
            self._halt_escalated.add(project_id)
            self._halt_escalation_warn_ts.pop(project_id, None)
            return

        # Throttled — see _HALT_ESCALATION_WARN_INTERVAL_SECONDS. The retry
        # itself is NOT throttled (that is the whole point of GAP B); only the
        # log line is, so a persistently-undeliverable halt stays visible
        # without emitting one WARNING every 5 seconds forever.
        outcome = getattr(verdict, 'outcome', None)
        now = time.monotonic()
        last_warned = self._halt_escalation_warn_ts.get(project_id)
        if (
            last_warned is None
            or (now - last_warned) >= _HALT_ESCALATION_WARN_INTERVAL_SECONDS
        ):
            self._halt_escalation_warn_ts[project_id] = now
            emit = logger.warning
        else:
            emit = logger.debug
        emit(
            'harness: judge halt for %s escalated to nothing (outcome=%s) — '
            'will retry on next halted tick',
            project_id, outcome,
        )

    async def _on_judge_unhalt(self, project_id: str) -> None:
        """Callback invoked by Judge.unhalt. Two responsibilities:

        1. Clear the escalation sentinel so a subsequent halt re-escalates.
           Without this, a manual unhalt followed by the halt re-firing (for
           whatever reason) would silently skip the escalation path because
           _notify_judge_halt dedupes per-process.
        2. Close the escalation the halt opened, via
           ``BacklogPolicy.on_judge_unhalt`` — otherwise the
           ``esc-reconciliation-halt-*.json`` stays pending forever and the
           dashboard keeps showing a halt that no longer exists (task 2998).

        Order matters: the sentinel is discarded FIRST, so a failure to close
        the record can never leave a stale sentinel behind — that would
        suppress the NEXT halt escalation entirely, which is strictly worse
        than one un-closed record.

        No Judge plumbing change is needed for the async signature:
        ``UnhaltCallback`` is typed ``Callable[[str], Awaitable[None] | None]``
        and ``Judge.unhalt`` already awaits an awaitable callback result.
        """
        self._halt_escalated.discard(project_id)
        self._halt_escalation_warn_ts.pop(project_id, None)
        if self._backlog_policy is None:
            return
        try:
            resolved = await self._backlog_policy.on_judge_unhalt(project_id)
        except Exception:
            logger.exception(
                'harness: backlog_policy.on_judge_unhalt raised for %s', project_id,
            )
            return
        if resolved:
            # ACCUMULATE rather than overwrite: an auto-unhalt-after-cooldown
            # stages ids that nothing pops, so a plain assignment would drop an
            # earlier unread close on the floor. Order-preserving dedupe keeps
            # a repeat close from being reported twice.
            staged = self._resolved_halt_escalations.setdefault(project_id, [])
            staged.extend(esc_id for esc_id in resolved if esc_id not in staged)

    def take_resolved_halt_escalations(self, project_id: str) -> list[str]:
        """Pop the escalation ids closed by the most recent unhalt.

        A pop-once handoff for the MCP tool layer (``unhalt_reconciliation``):
        reading it clears it, so the same close is never reported twice.
        """
        return self._resolved_halt_escalations.pop(project_id, [])

    @property
    def project_root(self) -> str:
        """Configured project root (from taskmaster config).

        Returns ``''`` when no taskmaster config is present.  Callers that need
        the canonical root for a *project_id* should use
        ``_known_project_scope_for(project_id)`` instead (task 1143; renamed +
        scope-returning per task 2146 β).
        """
        return self._project_root

    def _known_project_scope_for(self, project_id: str) -> ProjectScope:
        """Return the canonical ProjectScope for *project_id* from the registry.

        This is the pre-flight cross-contamination guard introduced by task 1143.
        It looks up ``self._known_projects`` (populated at init from the configured
        taskmaster root + ``DASHBOARD_KNOWN_PROJECT_ROOTS`` env var) and raises
        ``ValueError`` if no entry exists.  The error message includes both the
        unrecognised project_id *and* the sorted list of registered project_ids so
        the operator can immediately attribute and fix a misconfiguration.

        Raising here — before any journal or buffer side-effects — ensures that a
        missing registry entry never causes a partial cycle (events drained, journal
        row created, then a stage failure that leaves those events unrecoverable).

        Args:
            project_id: Canonical project identifier (e.g. ``'reify'``,
                ``'autopilot_video'``).  Must match a key in
                ``self._known_projects`` (derived from path basename,
                lowercase, dashes to underscores).

        Returns:
            The ProjectScope (project_id + absolute project_root) for *project_id*.

        Raises:
            UnknownProjectError: (a ``ValueError`` subclass) If *project_id* is not in ``self._known_projects``.
        """
        try:
            return ProjectScope(ProjectId(project_id), ProjectRoot(self._known_projects[project_id]))
        except KeyError:
            known_sorted = sorted(self._known_projects)
            raise UnknownProjectError(
                f'reconciliation: project_id {project_id!r} has no entry in '
                f'KNOWN_PROJECT_ROOTS (known: {known_sorted}). '
                f'Set DASHBOARD_KNOWN_PROJECT_ROOTS env var or add a '
                f'TaskmasterConfig.project_root that resolves to a known project.'
            ) from None

    def _resolve_known_root(self, project_id: str) -> str | None:
        """Return the registered project_root for *project_id*, or None if unknown.

        Thin lookup into ``self._known_projects``.  Unlike
        ``_known_project_scope_for`` (which raises ``UnknownProjectError`` for
        an unrecognised project_id), this returns ``None`` on a miss so it can
        be injected as the ``resolve_project_root`` callable for
        :func:`fused_memory.reconciliation.scope_freshness.precheck_scope_correction_freshness`,
        whose fail-open contract treats an unresolvable foreign project as
        "keep for re-investigation" rather than raising (task 2417).
        """
        return self._known_projects.get(project_id)

    def drain(self) -> None:
        """Signal the harness to stop starting new reconciliation cycles.

        Currently-running project loops complete their current cycle.
        The server continues serving reads/writes — only new cycles are suppressed.

        If no project loops are active at the moment of the call, emits
        ``'Harness fully drained — safe to restart'`` synchronously so that
        ``scripts/restart-fused-memory.sh --drain`` detects completion in <5 s
        rather than waiting up to 120 s for the main loop to observe idle state.
        When at least one loop is still in flight the marker is suppressed here;
        the main reconciliation loop emits it after all loops finish (existing
        behaviour, task 1053).
        """
        if self._draining:
            logger.info('Harness already draining')
            return
        self._draining = True
        logger.info('Harness drain requested — will finish current cycles, no new ones')
        # Task 1053: short-circuit when no project loops are active at drain time.
        # Delegates to _no_active_loops() so drain(), is_drained, and the main-loop
        # emission site (harness.py ~line 591) stay semantically coupled.
        # When at least one loop is still running, the main loop emits the marker
        # after all loops finish (existing behaviour, up to the 120-s timeout).
        if self._no_active_loops() and not self._drain_complete_logged:
            self._drain_complete_logged = True
            logger.info('Harness fully drained — safe to restart')

    def _no_active_loops(self) -> bool:
        """Return True when no project loops are currently running.

        A task is considered active when ``.done()`` returns False.
        Shared by :meth:`drain`, :attr:`is_drained`, and the main reconciliation
        loop so the three sites stay semantically coupled (task 1053).
        """
        return all(t.done() for t in self._project_tasks.values())

    @property
    def is_drained(self) -> bool:
        """True when draining and all project loops have completed."""
        return self._draining and self._no_active_loops()

    def _make_stages(self, scope: ProjectScope) -> list:
        """Create a fresh set of stage instances for one reconciliation cycle."""
        stage1 = MemoryConsolidator(
            StageId.memory_consolidator, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
            scope=scope, known_projects=self._known_projects,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
        )
        stage2 = TaskKnowledgeSync(
            StageId.task_knowledge_sync, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
            scope=scope, known_projects=self._known_projects,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
        )
        stage3 = IntegrityCheck(
            StageId.integrity_check, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
            scope=scope, known_projects=self._known_projects,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
        )
        stages = [stage1, stage2, stage3]
        self._propagate_escalation_queue(stages)
        return stages

    def _propagate_escalation_queue(self, stages: Iterable[Any]) -> None:
        """Apply harness's escalation URL and queue to each stage in *stages*.

        Called from ``_make_stages`` once per cycle (task 2146 β — there is no
        long-lived ``self.stages`` list to push into from
        ``_start_escalation_server`` anymore; escalation startup only stores
        ``self._escalation_url``/``self._escalation_queue`` for the next
        ``_make_stages`` call to pick up). Defensive: only assigns when the
        harness has a value, so calls before escalation startup leave stages
        untouched.

        Single-pass over *stages* so single-pass iterables (generators, ``iter(...)``)
        work correctly — the prior two-pass form would silently skip the queue
        assignment once the URL pass exhausted the iterator.
        """
        for s in stages:
            if self._escalation_url:
                s._escalation_url = self._escalation_url
            if self._escalation_queue is not None:
                s._escalation_queue = self._escalation_queue

    @staticmethod
    def _configure_consolidator(
        stage: MemoryConsolidator,
        tier: TierConfig,
        *,
        prior_s3_findings: list[dict] | None = None,
        cycle_fence_time: datetime | None = None,
        assembled_payload: AssembledPayload | None = None,
        remediation_findings: list[dict] | None = None,
        filtered_task_tree: FilteredTaskTree | None = None,
        task_count_verification: dict | None = None,
        graphiti_queue_health: dict | None = None,
        status_correction_reconciliation: dict | None = None,
    ) -> None:
        """Apply tier limits and mode-specific attributes to MemoryConsolidator.

        Shared between run_full_cycle and _run_remediation_pass to prevent
        attribute-configuration divergence.

        Note: filtered_task_tree here applies only to Stage 1 (MemoryConsolidator).
        Stage 2 (TaskKnowledgeSync) wiring is handled by the symmetric
        _configure_task_sync helper.

        task_count_verification: cross_verify_task_counts record (available only
            in full-cycle passes; None in remediation passes).
        graphiti_queue_health: summarize_graphiti_queue_health record (available only
            in full-cycle passes; None in remediation passes).
        status_correction_reconciliation: _reconcile_status_correction record
            (task 1938; available only in full-cycle passes; None in
            remediation passes).
        """
        stage.episode_limit = tier.episode_limit
        stage.memory_limit = tier.memory_limit
        stage.prior_s3_findings = prior_s3_findings
        stage.cycle_fence_time = cycle_fence_time
        stage.assembled_payload = assembled_payload
        stage.remediation_findings = remediation_findings
        stage.filtered_task_tree = filtered_task_tree
        stage.task_count_verification = task_count_verification
        stage.graphiti_queue_health = graphiti_queue_health
        stage.status_correction_reconciliation = status_correction_reconciliation

    @staticmethod
    def _configure_task_sync(
        stage: TaskKnowledgeSync,
        *,
        filtered_task_tree: FilteredTaskTree | None = None,
        remediation_mode: bool = False,
    ) -> None:
        """Apply Stage-2 attributes to TaskKnowledgeSync.

        Mirrors _configure_consolidator for Stage 1 — both full-cycle and
        remediation paths use this helper to keep attribute wiring symmetric
        and discoverable. Stage-2-specific attributes (filtered_task_tree,
        remediation_mode) are set here; Stage-1 wiring is handled by
        _configure_consolidator.
        """
        stage.filtered_task_tree = filtered_task_tree
        stage.remediation_mode = remediation_mode

    async def _fetch_filtered_task_tree(self, project_root: ProjectRoot) -> FilteredTaskTree:
        """Fetch the task tree once and return a filtered subset of active tasks.

        Degrades gracefully on failure — returns an empty FilteredTaskTree so
        stages can still do useful memory work without task data. (ref: task 455)

        Args:
            project_root: Absolute path to the project root for taskmaster.

        Returns:
            FilteredTaskTree with active tasks sorted by priority and aggregate
            counts. Returns empty FilteredTaskTree if taskmaster is unavailable,
            project_root is empty, non-absolute, or the fetch fails.
            Non-absolute paths are rejected before calling taskmaster to avoid
            silent failures from the backend's absolute-path validator.

        Log-level policy:
            Anomaly branches (taskmaster disabled, empty/non-absolute project_root)
            always log at INFO.  Successful fetches log at DEBUG unless
            ``raw_count > 0 and total_count == 0``, which indicates the filter
            silently dropped every task (e.g. all task entries are non-dict and
            skipped by the defensive isinstance guard in filter_task_tree —
            bare ints, malformed entries) and warrants an INFO-level alert.
            Healthy-but-empty projects (``raw_count == 0``) are classified as
            DEBUG (non-anomalous).
        """
        if not self.taskmaster:
            logger.info(
                'reconciliation.task_tree_taskmaster_disabled',
                extra={'project_root': project_root},
            )
            return FilteredTaskTree()
        if not project_root:
            logger.info(
                'reconciliation.task_tree_empty_project_root',
                extra={'project_root_repr': repr(project_root)},
            )
            return FilteredTaskTree()
        if not os.path.isabs(project_root):
            logger.warning(
                '_fetch_filtered_task_tree rejected non-absolute project_root %r'
                ' — cannot fetch tasks; Stage 2 will see empty tree',
                project_root,
            )
            return FilteredTaskTree()
        try:
            tasks_data = await self.taskmaster.get_tasks(project_root=project_root)
            raw_count = len(tasks_data.get('tasks', []))
            filtered = filter_task_tree(tasks_data)
            # Anomaly predicate is intentionally narrow (task-985 policy): only the
            # full-drop case (every top-level entry skipped by filter_task_tree's
            # defensive isinstance guard) is escalated to INFO.  Partial drops
            # (raw_count >> total_count when only some entries are skipped) remain at
            # DEBUG by design — the structured fields raw_count/total_count carry the
            # differentiating signal, and operators can grep DEBUG when investigating.
            # Broadening to a `raw_count > N * total_count` heuristic was considered and
            # rejected because total_count is post-flatten (includes subtasks) while
            # raw_count is the top-level pre-flatten count, making any threshold
            # semantically muddy without exposing a new skipped_count field from
            # filter_task_tree.
            is_anomaly = raw_count > 0 and filtered.total_count == 0
            logger.log(
                logging.INFO if is_anomaly else logging.DEBUG,
                'reconciliation.task_tree_fetched',
                extra={
                    'project_root': project_root,
                    'raw_count': raw_count,
                    'total_count': filtered.total_count,
                },
            )
            return filtered
        except Exception as exc:
            logger.warning(
                f'_fetch_filtered_task_tree failed for {project_root!r}: {exc}'
            )
            return FilteredTaskTree()

    async def _fetch_task_count_census(self, project_root: ProjectRoot) -> dict[str, str]:
        """Fetch the authoritative {id: status} map from taskmaster.get_statuses_fresh().

        Reads via get_statuses_fresh (task 2388), not get_statuses: get_statuses
        may be served from taskmaster's cached per-project connection, which can
        have a read transaction pinning it to a stale WAL snapshot. Because
        _fetch_filtered_task_tree's tree read shares that same cached connection,
        a pinned get_statuses would go stale *together* with the tree read, so
        they'd still agree with each other and cross_verify_task_counts
        (task_filter.py:598) would report a false consistent:true instead of
        surfacing the drift. get_statuses_fresh always reads a fresh snapshot on
        a dedicated connection, so it can't be pinned — a stale tree read now
        shows up as a genuine mismatch instead of being masked.

        Note on read-skew: reading the census fresh while the tree read stays
        on the cached connection makes the two reads slightly *more* likely to
        straddle a status transition than when both shared one pinned
        snapshot, so a transient single-cycle total_mismatch/done_mismatch
        that resolves itself next cycle is somewhat more likely post-fix.
        cross_verify_task_counts (task_filter.py:598) already documents
        single-cycle divergence as advisory and only escalates divergence
        that persists across consecutive cycles, so this is an accepted
        trade-off for surfacing real drift instead of masking it.

        Mirrors _fetch_filtered_task_tree's fail-open posture: guard against a
        falsy taskmaster, an empty project_root, and any exception raised by
        get_statuses_fresh — all degrade to an empty dict so the caller can
        distinguish 'not available' from a real status map.

        Args:
            project_root: Absolute path to the project root for taskmaster.

        Returns:
            {id: status} map, or {} when taskmaster is unavailable or the call fails.
        """
        if not self.taskmaster:
            logger.info(
                'reconciliation.task_count_census_taskmaster_disabled',
                extra={'project_root': project_root},
            )
            return {}
        if not project_root:
            logger.info(
                'reconciliation.task_count_census_empty_project_root',
                extra={'project_root_repr': repr(project_root)},
            )
            return {}
        try:
            statuses = await self.taskmaster.get_statuses_fresh(project_root=project_root)
            if isinstance(statuses, dict):
                return statuses
            logger.warning(
                '_fetch_task_count_census: get_statuses_fresh returned a non-dict'
                ' (type=%s, repr=%s) for %r; census cross-check will be skipped',
                type(statuses).__name__,
                repr(statuses)[:200],
                project_root,
            )
            return {}
        except Exception as exc:
            logger.warning(
                f'_fetch_task_count_census failed for {project_root!r}: {exc}'
            )
            return {}

    async def _check_graphiti_queue_health(self, project_id: ProjectId) -> dict | None:
        """Read the Graphiti async-queue dead-letter count for project_id.

        Uses DurableWriteQueue.get_stats(group_id=project_id) to classify the
        queue state via summarize_graphiti_queue_health.  When dead_count > 0,
        emits a WARNING-level log — this is the observable signal for the silent
        drop that add_memory's success=True at enqueue time hides.

        Args:
            project_id: Project identifier, used as the queue group_id.

        Returns:
            Health record dict, or None when durable_queue is unavailable or
            get_stats raises.
        """
        durable_queue = getattr(self.memory, 'durable_queue', None)
        if durable_queue is None:
            return None
        try:
            stats = await durable_queue.get_stats(group_id=project_id)
            health = summarize_graphiti_queue_health(stats)
            if not health['healthy']:
                logger.warning(
                    'reconciliation.graphiti_queue_unhealthy',
                    extra={
                        'project_id': project_id,
                        'dead_count': health['dead_count'],
                        'pending_count': health['pending_count'],
                        'oldest_pending_age_seconds': health['oldest_pending_age_seconds'],
                    },
                )
            return health
        except Exception as exc:
            logger.warning(
                f'_check_graphiti_queue_health failed for project_id={project_id!r}: {exc}'
            )
            return None

    async def _reconcile_status_correction(
        self, project_id: str, statuses: dict[str, str]
    ) -> dict | None:
        """Diff the cached Mem0 project_status_correction memory against the
        live get_statuses census and supersede it on divergence.

        Reuses the `statuses` census already fetched for task_count_verification
        in run_full_cycle — no extra get_statuses round-trip.  Fail-open at every
        branch (empty statuses / no cached memory / any exception), mirroring
        _check_graphiti_queue_health: a memory-store hiccup must never abort a
        reconciliation cycle.

        Args:
            project_id: Project identifier — scopes the metadata query and writes.
            statuses: Compact {id: status} map from get_statuses(), already
                fetched by run_full_cycle.  Empty/falsy means the live census is
                unavailable.

        Returns:
            None when statuses is empty (fail-open: no supersede attempted, and
            no memory-service calls are made).  Otherwise a record dict:
              - found=False when no cached project_status_correction memory exists.
              - diverged=False, superseded=False when the cached memory matches
                the live census (no-op).
              - diverged=True, superseded=True, memory_id, old, new when the
                cached memory was superseded.
              - an 'error' key when the query/diff/supersede body raised (nothing
                was written — add_memory itself failed, or the query/diff raised).
              - a 'delete_errors' key (list[str]) when add_memory succeeded (or
                the no-op branch ran) but one or more delete_memory calls raised —
                superseded still reflects whether the authoritative memory write
                landed, not whether every stale duplicate was cleaned up.
        """
        if not statuses:
            return None

        try:
            memories = await self.memory.get_memories_by_metadata(
                project_id=project_id,
                filters={'kind': 'project_status_correction'},
            )
            if not memories:
                return {
                    'available': True,
                    'found': False,
                    'diverged': False,
                    'superseded': False,
                }

            latest = max(memories, key=lambda m: m.get('created_at') or '')
            diff = diff_status_correction(latest.get('metadata') or {}, statuses)

            if not diff['diverged']:
                # Cap the pool to a single member even when already consistent —
                # otherwise duplicate project_status_correction memories returned
                # by the query (all matching the live census) would persist
                # indefinitely, since the no-op branch never used to touch them
                # (task 1938 amendment).
                duplicates = [m for m in memories if m['id'] != latest['id']]
                record = {
                    'available': True,
                    'found': True,
                    'diverged': False,
                    'superseded': False,
                    'memory_id': latest['id'],
                }
                delete_errors = await self._delete_status_correction_memories(
                    duplicates, project_id
                )
                if delete_errors:
                    record['delete_errors'] = delete_errors
                return record

            live = diff['live']
            corrected_metadata = {
                'kind': 'project_status_correction',
                # PRD D2 (task 3196): `supersedes` is a LIST of full UUIDs.  The
                # shape contract, the read tolerance for the legacy scalar, and
                # the writer/reader map all live in ONE place —
                # `memory_metadata.normalize_supersedes`'s docstring — rather
                # than being restated here.  Written list-shaped at the source
                # rather than leaning on the service-seam coercion in
                # validate_memory_metadata().  Exactly one predecessor is
                # recorded: `latest` is the single max()-by-created_at memory
                # being superseded.  Do NOT widen to every deleted duplicate —
                # the queried set is deleted for pool-capping (task 1938
                # amendment), a different relation from supersession.
                #
                # EXPECTED-DANGLING POINTER (deliberate; pre-dates the list
                # migration and is unchanged by it).  `latest` is a member of
                # the `memories` set deleted below, so this id does NOT resolve
                # via `get_memory_by_id` once this branch returns.  That is
                # intended: the pool cap requires the corrected predecessor to
                # go away, and `supersedes` is kept as an audit trace of WHICH
                # record was corrected, not as a live pointer.  Consequence for
                # the eval program's E4 dangling-pointer census
                # (docs/prds/memory-eval-program.md §γ, which resolves
                # `supersedes` targets via `get_memory_by_id`): 100% of this
                # writer's edges are dangling BY DESIGN, so E4 must allowlist
                # `kind=project_status_correction` rather than report a census
                # spike.  Making the target resolvable would mean keeping
                # `latest` alive, which reopens the unbounded-pool bug — i.e.
                # not a documentation-only change, which is why this leaf
                # records the invariant instead of "fixing" it.
                'supersedes': [latest['id']],
                'task_count_done': live['done'],
                'task_count_total': live['total'],
                'active_tasks': live['active_tasks'],
                'source': 'stage1_status_correction',
            }
            content = (
                f'Stage 1 status-correction reconciliation: the cached '
                f'project_status_correction memory diverged from the authoritative '
                f"get_statuses census: done={live['done']} total={live['total']} "
                f"active={len(live['active_tasks'])}."
            )
            # Add-then-delete: guarantees at least one correct memory always exists
            # even if a delete below fails — the fresh memory (supersedes=[<old_id>])
            # is still the most-recent, so next cycle's max()-by-created_at selection
            # ignores any stale leftover and self-heals.  Once add_memory lands, the
            # supersede has effectively happened regardless of delete outcomes below
            # — delete failures are reported via 'delete_errors', not by downgrading
            # 'superseded' (task 1938 amendment).
            await self.memory.add_memory(
                content=content,
                category='observations_and_summaries',
                project_id=project_id,
                metadata=corrected_metadata,
                _source='stage1_status_correction',
            )
            # Delete the whole queried set (not just `latest`) to bound the pool to
            # the single fresh memory just written — mirrors the stage2_cycle_summary
            # pool-cap fix (tasks 20e8c2f1/45489c2b/db2ea69e). Per-item failures are
            # swallowed inside the helper so one bad delete doesn't abort the rest.
            delete_errors = await self._delete_status_correction_memories(
                memories, project_id
            )

            logger.warning(
                'reconciliation.status_correction_superseded',
                extra={
                    'project_id': project_id,
                    'memory_id': latest['id'],
                    'old': diff['cached'],
                    'new': diff['live'],
                },
            )

            record = {
                'available': True,
                'found': True,
                'diverged': True,
                'superseded': True,
                'memory_id': latest['id'],
                'old': diff['cached'],
                'new': diff['live'],
            }
            if delete_errors:
                record['delete_errors'] = delete_errors
            return record
        except Exception as exc:
            logger.warning(
                '_reconcile_status_correction failed for project_id=%r: %s',
                project_id,
                exc,
            )
            return {
                'available': True,
                'superseded': False,
                'error': str(exc),
            }

    async def _delete_status_correction_memories(
        self, memories: list[dict], project_id: str
    ) -> list[str]:
        """Delete each memory in `memories`, one at a time, swallowing per-item
        exceptions so a single failed delete doesn't stop the rest (task 1938
        amendment — see _reconcile_status_correction).

        Args:
            memories: Memory records (each with an 'id' key) to delete. May be
                empty — a no-op in that case.
            project_id: Project identifier passed through to delete_memory.

        Returns:
            List of stringified errors, one per failed delete; empty when every
            delete succeeded (or `memories` was empty). Never raises.
        """
        errors: list[str] = []
        for m in memories:
            try:
                await self.memory.delete_memory(
                    memory_id=m['id'],
                    store='mem0',
                    project_id=project_id,
                )
            except Exception as exc:
                logger.warning(
                    'reconciliation.status_correction_delete_failed',
                    extra={
                        'project_id': project_id,
                        'memory_id': m.get('id'),
                        'error': str(exc),
                    },
                )
                errors.append(str(exc))
        return errors

    # ── Stale-run recovery ─────────────────────────────────────────────

    async def _recover_stale_runs(self) -> None:
        """Find runs stuck in 'running' state and mark them failed.

        Uses stale_run_recovery_seconds as the age cutoff (default 1800s),
        then double-checks that the *same* instance still holds the project's
        reconciliation lock before skipping — protecting legitimately long-running
        cycles owned by this process while still recovering orphans whose owning
        instance is dead even when a fresh instance has since acquired the lock.

        Pre-migration runs (instance_id IS NULL) are recovered unconditionally
        — the bug this guards against was exactly stale rows being shielded
        forever by a new instance's lock acquisition.
        """
        cutoff = self.config.stale_run_recovery_seconds
        stale_runs = await self.journal.get_stale_runs(cutoff)
        for run in stale_runs:
            # Skip only when the *same* instance that started the run still
            # holds the lock AND the lock is freshly heartbeated (within cutoff).
            # A dead owner's lock row satisfies the identity check but its
            # heartbeat_at will be > cutoff seconds old — treat that as an orphan.
            # A lock held by a different instance, or no lock at all, also means
            # the original owner is gone and the run is an orphan.
            lock_holder, lock_age = await self.buffer.get_lock_status(run.project_id)
            if (
                lock_holder is not None
                and run.instance_id is not None
                and lock_holder == run.instance_id
                and lock_age is not None
                and lock_age <= cutoff
            ):
                continue

            diag = await self._recover_one_run(
                run, lock_holder, lock_age,
                error_message=f'Run stale (>{cutoff}s, lock expired), recovered by harness',
            )
            logger.warning(
                f'Recovering stale run {run.id} for {run.project_id} '
                f'(started {run.started_at.isoformat()}, lock expired, '
                f'instance={run.instance_id}, disposition={diag["disposition"]})'
            )
            detail = (
                f"project={diag['project_id']} run_type={diag['run_type']} "
                f"instance={diag['instance_id']} age={diag['age_seconds']:.0f}s "
                f"disposition={diag['disposition']}"
            )
            if diag['disposition'] == 'dead_owner_shielded':
                # Operational-restart noise: same-owner orphan whose heartbeat
                # is older than the liveness cutoff — provably from a prior
                # process incarnation that was hard-killed (e.g. watchdog SIGABRT,
                # task 1731).  A fresh instance's reaper is recovering its orphan;
                # that is expected behaviour, not an integrity finding.  Suppress
                # the recon_stale_run escalation; recovery, journal _error, and
                # the warning log above remain so the event stays observable in
                # logs/journal without filing a noisy alert.
                logger.info(
                    'recon_stale_run suppressed: dead_owner_shielded orphan recovered '
                    '(operational kill/restart, task 1731)',
                    extra={
                        'run_id': run.id,
                        'project_id': run.project_id,
                        'age_seconds': diag['age_seconds'],
                    },
                )
                # Task 1755 / PRD β: aggregate storm alarm.  The per-event
                # suppression above is kept; the counter fires ONE loud
                # 'recon_watchdog_kill_storm' escalation when the burst
                # threshold is crossed within the rolling window.
                #
                # Observability note: storm['count'] is the rolling-window count
                # at the moment the threshold was first crossed in this window —
                # not the total orphans reaped in this single _recover_stale_runs
                # pass.  Within one pass, the first storm return sets the rate-
                # limit timestamp; subsequent orphans in the same pass are
                # rate-limited to None, so they do not add to the reported count.
                # Operators who need the per-tick total should check the INFO
                # log lines ("recon_stale_run suppressed: dead_owner_shielded
                # orphan recovered") emitted for every suppression regardless.
                storm = self._record_dead_owner_suppression(run.project_id, run.instance_id)
                if storm is not None:
                    window_min = storm['window_seconds'] / 60
                    proj_label = ', '.join(storm['projects']) or run.project_id
                    summary = (
                        f"dead_owner_shielded suppression storm: {storm['count']} in "
                        f"{window_min:.0f} min (distinct dead-owner instances; "
                        f"projects: {proj_label}) — "
                        f'watchdog SIGABRT churn — full recon runs not completing'
                    )
                    self._escalate(
                        'recon_watchdog_kill_storm',
                        run.id,
                        summary,
                        detail,
                        finding=_DEAD_OWNER_STORM_FINDING,
                    )
            else:
                self._escalate(
                    'recon_stale_run',
                    run.id,
                    f'Run stale (>{cutoff}s, lock expired), recovered',
                    detail,
                )

    async def _recover_one_run(
        self,
        run: ReconciliationRun,
        lock_holder: str | None,
        lock_age: float | None,
        *,
        disposition: str = 'failed',
        error_type: str = 'StaleRunRecovery',
        error_message: str | None = None,
    ) -> dict:
        """Recover a single stuck 'running' row to a terminal status.

        Shared mechanical body for both recovery passes — the age-based
        ``_recover_stale_runs`` reaper and the startup ``_recover_predecessor_runs``
        pass: error-stamp the run, complete it, restore the drained events it
        owned, release the lock when this run's own instance still owns it,
        and replay any deferred writes for the project. Returns the
        ``build_stale_run_diagnostics`` dict so the caller can drive its own
        logging/escalation policy on top of this shared body.

        The restore is run-scoped (task 2711 / E7): only events this run
        itself drained (plus any pre-task-2711, unattributed leftovers — see
        ``restore_drained``'s ``include_unattributed``) are restored, so a
        concurrent live run's in-flight drained events on the same project
        are never clobbered back to 'buffered'.

        ``disposition`` is the terminal journal status to complete the run
        with; it defaults to (and, as of task 2711, is always) 'failed'. This
        is the seam for task σ (session-resume): a future caller can pass a
        resumable disposition instead before falling into this same body.
        """
        cutoff = self.config.stale_run_recovery_seconds
        diag = build_stale_run_diagnostics(run, lock_holder, lock_age, cutoff)

        run.stage_reports['_error'] = {
            'error_type': error_type,
            'error_message': error_message or f'Run recovered by harness ({error_type})',
            'failed_stage': None,
            **diag,
        }
        await self.journal.update_run_stage_reports(run.id, run.stage_reports)
        await self.journal.complete_run(run.id, disposition)

        # Task 2744: sweep the config dir a dead predecessor process may have left
        # behind for this run. Defensive — never mask the recovery outcome.
        #
        # TODO(task σ / session-resume): this GC is UNCONDITIONAL because
        # ``disposition`` is always 'failed' today — a non-resumable terminal
        # status — so the per-run config dir (which holds the CLI transcript a
        # ``--resume`` would need) is always safe to delete here. When σ teaches
        # this seam to complete a run with a *resumable* disposition (see the
        # ``disposition`` docstring above), it MUST gate this GC on the
        # disposition and skip it for resumable ones, or the transcript
        # session-resume depends on is destroyed and resume has nothing to
        # resume from.
        try:
            gc_run_config_dir(self.journal.data_dir, run.id)
        except Exception as gc_err:  # noqa: BLE001
            logger.warning(
                'gc_run_config_dir failed for recovered run %s: %r', run.id, gc_err
            )

        # include_unattributed=True additionally sweeps drained_by_run_id IS
        # NULL rows in this project: events drained by a pre-task-2711
        # process (before drains stamped run attribution) would otherwise
        # never match any run_id and stay stuck 'drained' forever once a
        # recovery pass switched to run-scoped restore. Every current drain
        # path stamps a run_id, so a NULL row here is provably a leftover
        # from a process that no longer exists — safe to fold into this
        # run's recovery without resurrecting a *concurrent* run's events
        # (those always carry non-NULL attribution under current code).
        restored = await self.buffer.restore_drained(
            run.project_id, run_id=run.id, include_unattributed=True,
        )
        if restored:
            logger.info(f'Restored {restored} drained events for run {run.id}')

        # Ownership-scoped release: see plans/recon-stale-recovery-rca.md.
        # Releasing without an instance_id filter would strip a live cycle's
        # lock on the same project, causing the next reaper tick to
        # misclassify the live run as stale (cross-instance lock theft, the
        # 2026-05-28 false-positive cascade).  Only release when the
        # orphan's instance_id is known AND owns the current lock — a match
        # here means a defunct previous incarnation of this very instance (or,
        # for the predecessor pass, the dead predecessor itself) left the row
        # behind.  Genuinely-abandoned locks held by other dead instances are
        # cleaned up by the heartbeat-staleness sweep (stale_lock_seconds =
        # 7200s) inside mark_run_active / get_lock_holder_instance_id.
        if (
            run.instance_id is not None
            and lock_holder == run.instance_id
        ):
            await self.buffer.mark_run_complete(
                run.project_id, instance_id=run.instance_id,
            )
        await self._replay_deferred_writes(ProjectId(run.project_id))
        return diag

    async def _recover_predecessor_runs(self) -> None:
        """One-shot startup pass: recover 'running' rows owned by a dead
        predecessor instance, regardless of age (task 2711 / E6).

        fm is a single-instance deployment and each process constructs a
        fresh ``EventBuffer.instance_id`` — so at startup, any running row
        owned by a *different* instance_id is provably a dead predecessor,
        not a run that might still be live. This closes the gap left by the
        age-gated ``_recover_stale_runs`` reaper (default 1800s cutoff),
        which would otherwise leave such an orphan's lock wedging the
        project for up to half an hour.

        Ownership must be corroborated against the lock before acting
        (never reap on age alone when the owner might be live): a run is
        recovered only when the project's current lock holder equals the
        run's own instance_id, and that instance_id is not this process's
        own. A run whose owner cannot be corroborated this way — no lock,
        lock held by a third instance, or a NULL (pre-migration)
        instance_id — is left untouched for the existing age-based
        backstop to handle.

        Recovery always completes the run with disposition='failed' and
        never escalates: adopting your own dead predecessor's orphan at
        startup is expected operational-restart behaviour (mirrors the
        dead_owner_shielded suppression rationale in _recover_stale_runs),
        made observable via a structured log instead of an alert.
        """
        my_iid = self.buffer.instance_id
        for run in await self.journal.get_running_runs():
            if run.instance_id is None or run.instance_id == my_iid:
                continue
            lock_holder, lock_age = await self.buffer.get_lock_status(run.project_id)
            if lock_holder != run.instance_id:
                continue

            diag = await self._recover_one_run(
                run, lock_holder, lock_age,
                disposition='failed',
                error_type='PredecessorRunRecovery',
                error_message=(
                    'Run owned by dead predecessor instance, recovered at startup'
                ),
            )
            logger.info(
                'reconciliation.predecessor_run_recovered',
                extra={
                    'run_id': run.id,
                    'project_id': run.project_id,
                    'instance_id': run.instance_id,
                    'age_seconds': diag['age_seconds'],
                    'disposition': diag['disposition'],
                },
            )

    async def _resume_interrupted_runs(self) -> None:
        """One-shot startup pass: adopt and ``--resume`` runs a dead predecessor
        left ``interrupted`` mid-stage (task σ / 2717).

        Runs beside :meth:`_recover_predecessor_runs` and BEFORE the age-gated
        :meth:`_recover_stale_runs` reaper. For each ``interrupted`` run owned by
        an own-dead-predecessor (``instance_id`` not None and not this instance,
        with the project lock still corroborating that owner — the same
        ownership test the predecessor pass uses), the guard rails below decide
        between two paths:

        * **Resume** — when every guard rail passes, adopt the project lock,
          source the run's still-drained events read-only via
          :meth:`EventBuffer.get_drained_events` (restoring them would
          double-process on resume), and drive a resume-aware
          :meth:`run_full_cycle` that skips already-completed stages and
          ``--resume``s the ``stage_cursor`` stage. The adopted lock is released
          after the resumed cycle — this pass cannot heartbeat it, so holding it
          would wedge the normal project loop.
        * **Fallback** — on ANY doubt (see :meth:`_resume_guard_reason`:
          freshness expired, per-run attempt cap reached, this stage_cursor
          already resumed once, or the session transcript is missing) route to
          :meth:`_recover_one_run` with ``disposition='failed'``, which
          error-stamps the run, restores its drained events, and GCs the config
          dir — today's failed+restore path, verbatim.
        """
        my_iid = self.buffer.instance_id
        for run in await self.journal.get_interrupted_runs():
            if run.instance_id is None or run.instance_id == my_iid:
                continue
            lock_holder, lock_age = await self.buffer.get_lock_status(run.project_id)
            if lock_holder != run.instance_id:
                continue

            # ── Master switch (honoured on the RESUMING side too): when a deploy
            # opts out of resume via resume_after_restart=False — typically
            # because it changed recon prompts/tooling — a run the OLD
            # (resume-enabled) process already marked `interrupted` must NOT be
            # --resume'd into the NEW system prompt (a --resume finishes under the
            # OLD system prompt by construction). Route it to the same
            # failed+restore fallback the guard rails use so it is cleaned up
            # (drained events restored, config dir GC'd) rather than left orphaned
            # for the age-based reaper. This is a deliberate operator opt-out, NOT
            # a resume failure, so it deliberately does NOT feed the
            # _record_resume_failure storm counter.
            if not self.config.resume_after_restart:
                await self._recover_one_run(
                    run, lock_holder, lock_age,
                    disposition='failed',
                    error_type='InterruptedRunResumeDisabled',
                    error_message=(
                        'resume_after_restart disabled — interrupted run cleaned '
                        'up via failed+restore instead of --resume (deploy opted '
                        'out of resuming into a changed prompt/toolset)'
                    ),
                )
                logger.info(
                    'reconciliation.interrupted_run_resume_disabled',
                    extra={
                        'run_id': run.id,
                        'project_id': run.project_id,
                        'instance_id': run.instance_id,
                        'stage_cursor': run.stage_cursor,
                    },
                )
                continue

            # ── Guard rails: on ANY doubt, fall back to the failed+restore
            # recovery path rather than --resume a stale/unsafe run.  Checked
            # BEFORE adopting the lock so _recover_one_run's own ownership-scoped
            # release (lock_holder == run.instance_id) cleans up the dead
            # predecessor's lock exactly as the predecessor-recovery pass does.
            unresumable_reason = self._resume_guard_reason(run)
            if unresumable_reason is not None:
                await self._recover_one_run(
                    run, lock_holder, lock_age,
                    disposition='failed',
                    error_type='InterruptedRunUnresumable',
                    error_message=unresumable_reason,
                )
                logger.info(
                    'reconciliation.interrupted_run_unresumable',
                    extra={
                        'run_id': run.id,
                        'project_id': run.project_id,
                        'instance_id': run.instance_id,
                        'stage_cursor': run.stage_cursor,
                        'reason': unresumable_reason,
                    },
                )
                # Rolling-window storm counter: a single unresumable run is
                # expected operational noise (kept to the INFO log above), but
                # repeated resume failures across runs — prompt/tool drift or
                # systematically stale transcripts — fire ONE loud recon-scoped
                # escalation per window (rate-limited single-fire, mirroring
                # _record_dead_owner_suppression).
                storm = self._record_resume_failure(run.project_id)
                if storm is not None:
                    window_min = storm['window_seconds'] / 60
                    proj_label = ', '.join(storm['projects']) or run.project_id
                    summary = (
                        f"resume-failure storm: {storm['count']} unresumable "
                        f"interrupted runs in {window_min:.0f} min "
                        f"(projects: {proj_label}) — interrupted reconciliation "
                        f"runs are not resuming (check recon prompt/tool drift or "
                        f"stale transcripts)"
                    )
                    detail = (
                        f'run_id={run.id} project={run.project_id} '
                        f'reason={unresumable_reason}'
                    )
                    self._escalate(
                        'recon_resume_failure_storm', run.id, summary, detail,
                    )
                continue

            # Persist the incremented resume bookkeeping BEFORE adopting/resuming
            # so the per-run cap and one-resume-per-stage rails survive a
            # re-interrupt of this very resume.  Stored out-of-band in
            # stage_reports['_resume'] (mirroring the '_error' entry) — no new
            # runs-table column.
            resume_meta = run.stage_reports.get('_resume')
            prior_count = (
                resume_meta.get('count', 0) if isinstance(resume_meta, dict) else 0
            )
            run.stage_reports['_resume'] = {
                'count': prior_count + 1,
                'last_stage': run.stage_cursor,
            }
            await self.journal.update_run_stage_reports(run.id, run.stage_reports)

            # Adopt the project lock: release the dead predecessor's corroborated
            # lock, then re-acquire as this instance so the resumed cycle owns the
            # project and the reaper cannot clobber it mid-resume.
            await self.buffer.mark_run_complete(
                run.project_id, instance_id=run.instance_id,
            )
            acquired = await self.buffer.mark_run_active(run.project_id)
            if not acquired:
                # Lost the adopt race (not expected at single-instance startup);
                # leave the run for the age-based backstop rather than resume
                # without holding the lock.
                logger.warning(
                    'reconciliation.resume_adopt_lock_failed run_id=%s project=%s',
                    run.id, run.project_id,
                )
                continue

            try:
                events = await self.buffer.get_drained_events(run.project_id, run.id)
                await self.run_full_cycle(
                    run.project_id, 'resume_after_restart',
                    resume_run=run, events=events,
                )
                logger.info(
                    'reconciliation.interrupted_run_resumed',
                    extra={
                        'run_id': run.id,
                        'project_id': run.project_id,
                        'instance_id': run.instance_id,
                        'stage_cursor': run.stage_cursor,
                    },
                )
            finally:
                # Release the adopted lock so the normal project loop can
                # re-acquire — a one-shot startup pass cannot heartbeat it.
                await self.buffer.mark_run_complete(
                    run.project_id, instance_id=self.buffer.instance_id,
                )

    def _resume_guard_reason(self, run: ReconciliationRun) -> str | None:
        """Return a reason string if *run* is NOT safe to ``--resume``, else None.

        The task σ guard rails, evaluated cheapest-first (all config-driven). On
        ANY doubt this returns a non-None reason and the caller falls back to
        :meth:`_recover_one_run` (failed+restore). None means every rail passed
        and the run may be adopted and resumed.

        Rails:
        * **freshness** — ``now - run.completed_at`` (the interrupt instant
          stamped by ``complete_run``) must be within
          ``resume_freshness_window_seconds``.
        * **per-run cap** — ``stage_reports['_resume'].count`` must be below
          ``resume_max_attempts_per_run`` (bounds a resume→re-interrupt loop).
        * **one-resume-per-stage** — ``stage_reports['_resume'].last_stage``
          must differ from ``run.stage_cursor`` (never re-resume the same stage).
        * **transcript-exists** — only when a session was captured
          (``run.session_id`` non-null): its transcript must still be on disk in
          the per-run config dir.  ``record_run_session``'s snapshot is
          best-effort (can go stale after an internal cap retry), so a missing
          transcript means there is nothing to ``--resume``.  A NULL session_id
          is NOT a failure — the interrupt landed between stages, so the run
          resumes with every remaining stage fresh (no --resume), degrading
          cleanly.
        """
        # Freshness — measured from the interrupt instant (completed_at).
        if run.completed_at is None:
            return 'run has no completed_at (interrupt instant) — cannot verify freshness'
        age_secs = (datetime.now(UTC) - run.completed_at).total_seconds()
        if age_secs > self.config.resume_freshness_window_seconds:
            return (
                f'interrupt is {age_secs:.0f}s old (> '
                f'resume_freshness_window_seconds={self.config.resume_freshness_window_seconds})'
            )

        # Per-run attempt cap + one-resume-per-stage (out-of-band _resume meta).
        resume_meta = run.stage_reports.get('_resume')
        resume_meta = resume_meta if isinstance(resume_meta, dict) else {}
        count = resume_meta.get('count', 0)
        if count >= self.config.resume_max_attempts_per_run:
            return (
                f'resume attempt cap reached (count={count} >= '
                f'resume_max_attempts_per_run={self.config.resume_max_attempts_per_run})'
            )
        if resume_meta.get('last_stage') == run.stage_cursor:
            return (
                f'stage_cursor {run.stage_cursor!r} already resumed once '
                '(one-resume-per-stage)'
            )

        # Transcript existence — only meaningful when a session was captured.
        if run.session_id:
            config_dir = TaskConfigDir(
                task_id=run.id,
                base_dir=recon_config_base_dir(self.journal.data_dir),
            )
            transcript = read_transcript_records(config_dir.path, run.session_id)
            if not transcript:
                return (
                    f'session {run.session_id} has no transcript on disk '
                    f'({config_dir.path}) — unresumable'
                )

        return None

    # ── Dead-owner suppression storm counter ─────────────────────────

    def _record_dead_owner_suppression(
        self, project_id: str, instance_id: str | None = None, *, now: datetime | None = None
    ) -> dict | None:
        """Record one dead_owner_shielded suppression and check for a storm.

        Appends (effective_now, project_id, instance_id) to the rolling deque,
        prunes entries older than the configured window, then:
        - Returns None if the number of DISTINCT non-None instance_id values
          (dead-owner instances) in the window is below the threshold.
        - Returns None if the alarm already fired within this window
          (rate limit: <=1 per window).
        - Otherwise sets _last_suppression_storm_escalation_at = effective_now
          and returns a storm summary dict with 'count' (the distinct
          dead-owner-instance count), 'window_seconds', and 'projects'
          (sorted distinct project labels seen in the window).

        Task 2039: the count is keyed on DISTINCT dead-owner instance_id
        values, not on the number of suppression events. All orphans
        recovered by one restart share that ONE dead owner's instance_id, so
        a single multi-project restart contributes only 1 to the count no
        matter how many projects it touches; only genuinely-independent
        watchdog kills (distinct instance_id values) accumulate toward the
        threshold. instance_id=None entries (should not occur for
        dead_owner_shielded, which requires a matching non-None instance_id)
        are excluded from the distinct set defensively.

        The now= parameter follows the ``_finding_recently_resolved(..., now=None)``
        time-injection convention (harness.py:1592) for deterministic unit tests.
        Task 1755 / PRD β; distinct-instance counting added by task 2039.
        """
        effective_now = now if now is not None else datetime.now(UTC)

        # Append and prune the rolling window.
        self._dead_owner_suppressions.append((effective_now, project_id, instance_id))
        window = timedelta(seconds=self.config.dead_owner_suppression_storm_window_seconds)
        cutoff_ts = effective_now - window
        while self._dead_owner_suppressions and self._dead_owner_suppressions[0][0] < cutoff_ts:
            self._dead_owner_suppressions.popleft()

        count = len({iid for _, _, iid in self._dead_owner_suppressions if iid is not None})
        if count < self.config.dead_owner_suppression_storm_threshold:
            return None

        # Threshold crossed — apply the per-window rate limit.
        if (
            self._last_suppression_storm_escalation_at is not None
            and (effective_now - self._last_suppression_storm_escalation_at) < window
        ):
            return None

        # Fire: set rate-limit timestamp and build the storm summary dict.
        self._last_suppression_storm_escalation_at = effective_now
        projects = sorted({pid for _, pid, _ in self._dead_owner_suppressions})
        return {
            'count': count,
            'window_seconds': self.config.dead_owner_suppression_storm_window_seconds,
            'projects': projects,
        }

    # ── Placeholder-finding drop storm counter (task 1970 amendment) ───

    def _record_placeholder_finding_drop(
        self, project_id: str, *, now: datetime | None = None
    ) -> dict | None:
        """Record one dropped referenceless-finding event and check for a storm.

        Same rolling-window-counter + rate-limited-single-fire shape as
        _record_dead_owner_suppression above, applied to
        reconciliation.remediation_dropped_placeholder_finding events instead
        of dead_owner_shielded suppressions.  Thresholds are the plain module
        constants _PLACEHOLDER_DROP_STORM_THRESHOLD /
        _PLACEHOLDER_DROP_STORM_WINDOW_SECONDS rather than ReconciliationConfig
        fields, since this counter is private to this module.

        Appends (effective_now, project_id) to the rolling deque, prunes
        entries older than the configured window, then:
        - Returns None if the count is below the threshold.
        - Returns None if the alarm already fired within this window
          (rate limit: <=1 per window).
        - Otherwise sets _last_placeholder_drop_storm_escalation_at =
          effective_now and returns a storm summary dict with 'count',
          'window_seconds', and 'projects' (sorted distinct project labels
          seen in the window).

        The now= parameter follows the same time-injection convention as
        _record_dead_owner_suppression, for deterministic unit tests.
        """
        effective_now = now if now is not None else datetime.now(UTC)

        # Append and prune the rolling window.
        self._placeholder_finding_drops.append((effective_now, project_id))
        window = timedelta(seconds=_PLACEHOLDER_DROP_STORM_WINDOW_SECONDS)
        cutoff_ts = effective_now - window
        while (
            self._placeholder_finding_drops
            and self._placeholder_finding_drops[0][0] < cutoff_ts
        ):
            self._placeholder_finding_drops.popleft()

        count = len(self._placeholder_finding_drops)
        if count < _PLACEHOLDER_DROP_STORM_THRESHOLD:
            return None

        # Threshold crossed — apply the per-window rate limit.
        if (
            self._last_placeholder_drop_storm_escalation_at is not None
            and (effective_now - self._last_placeholder_drop_storm_escalation_at) < window
        ):
            return None

        # Fire: set rate-limit timestamp and build the storm summary dict.
        self._last_placeholder_drop_storm_escalation_at = effective_now
        projects = sorted({pid for _, pid in self._placeholder_finding_drops})
        return {
            'count': count,
            'window_seconds': _PLACEHOLDER_DROP_STORM_WINDOW_SECONDS,
            'projects': projects,
        }

    # ── Resume-failure storm counter (task σ / 2717) ───────────────────

    def _record_resume_failure(
        self, project_id: str, *, now: datetime | None = None
    ) -> dict | None:
        """Record one unresumable/failed interrupted-run resume and check for a storm.

        Same rolling-window per-event counter + rate-limited single-fire shape as
        :meth:`_record_placeholder_finding_drop`, applied to the failed+restore
        fallback arm of :meth:`_resume_interrupted_runs`.  Thresholds are the
        config fields ``resume_failure_storm_threshold`` /
        ``resume_failure_storm_window_seconds`` (mirroring
        :meth:`_record_dead_owner_suppression`, which likewise reads config)
        rather than module constants, so an operator can retune the alarm.

        Appends ``(effective_now, project_id)`` to the rolling deque, prunes
        entries older than the configured window, then:
        - Returns None if the count is below the threshold.
        - Returns None if the alarm already fired within this window (rate limit:
          <=1 per window).
        - Otherwise sets ``_last_resume_failure_storm_escalation_at =
          effective_now`` and returns a storm summary dict with ``count``,
          ``window_seconds``, and ``projects`` (sorted distinct project labels
          seen in the window).

        The now= parameter follows the same time-injection convention as the
        sibling storm counters, for deterministic unit tests.
        """
        effective_now = now if now is not None else datetime.now(UTC)

        # Append and prune the rolling window.
        self._resume_failures.append((effective_now, project_id))
        window = timedelta(seconds=self.config.resume_failure_storm_window_seconds)
        cutoff_ts = effective_now - window
        while self._resume_failures and self._resume_failures[0][0] < cutoff_ts:
            self._resume_failures.popleft()

        count = len(self._resume_failures)
        if count < self.config.resume_failure_storm_threshold:
            return None

        # Threshold crossed — apply the per-window rate limit.
        if (
            self._last_resume_failure_storm_escalation_at is not None
            and (effective_now - self._last_resume_failure_storm_escalation_at) < window
        ):
            return None

        # Fire: set rate-limit timestamp and build the storm summary dict.
        self._last_resume_failure_storm_escalation_at = effective_now
        projects = sorted({pid for _, pid in self._resume_failures})
        return {
            'count': count,
            'window_seconds': self.config.resume_failure_storm_window_seconds,
            'projects': projects,
        }

    # ── Deferred write replay ─────────────────────────────────────────

    async def _replay_deferred_writes(self, project_id: ProjectId) -> None:
        """Replay targeted-recon writes that were deferred during a full cycle.

        Uses a claim → replay-one → delete-on-success pattern so that
        cancellation or process crash mid-loop does not lose any writes:
        claimed-but-not-deleted rows remain in SQLite and are recovered
        on next startup by `release_stale_claims`.
        """
        deferred = await self.buffer.claim_deferred_writes(project_id)
        if not deferred:
            return
        logger.info(f'Replaying {len(deferred)} deferred writes for {project_id}')
        for write in deferred:
            meta = write['metadata'] or {}
            tid = meta.get('task_id')
            transition = meta.get('transition')

            # Dedup check: skip completion-summary writes that already exist in Mem0.
            # Only for transition='done' writes — other transitions are left as-is.
            # find_prior_memory degrades gracefully: search failures log a WARNING
            # under logger and return None so the write proceeds normally.
            if transition == 'done' and tid:
                prior = await find_prior_memory(
                    self.memory,
                    project_id=project_id,
                    task_id=tid,
                    kind={'transition': 'done'},
                    query=f'task {tid} targeted_reconciliation completion done',
                    # categories pinned to match the writer in targeted.py:_on_task_done
                    # (TargetedReconciliation writes completion summaries with
                    # transition='done' under 'observations_and_summaries'). Keep these
                    # in sync — a future writer using a different category would silently
                    # bypass this dedup.
                    categories=['observations_and_summaries'],
                    limit=20,
                    log=logger,
                )
                if prior is not None:
                    logger.info(
                        'Skipping deferred completion-summary for task %s — already written',
                        tid,
                    )
                    await self.buffer.delete_deferred_write(write['id'])
                    continue

            try:
                await self.memory.add_memory(
                    content=write['content'],
                    category=write['category'],
                    project_id=project_id,
                    metadata=write['metadata'],
                    _source='targeted_recon',
                )
            except Exception as e:
                # Leave the row claimed so it isn't retried in this process;
                # release_stale_claims at next startup will re-queue it.
                logger.warning(f'Failed to replay deferred write {write["id"]}: {e}')
                continue
            await self.buffer.delete_deferred_write(write['id'])

    # ── Escalation support ─────────────────────────────────────────────

    async def _start_escalation_server(self) -> None:
        """Start the escalation MCP server as a background asyncio task."""
        if not HAS_ESCALATION:
            logger.info('Escalation package not installed — skipping escalation server')
            return

        queue_dir = Path(self.config.escalation_queue_dir)
        if not queue_dir.is_absolute():
            queue_dir = Path(self.config.explore_codebase_root) / queue_dir
        self._escalation_queue = EscalationQueue(queue_dir)  # type: ignore[possibly-undefined]

        mcp_server = create_escalation_server(self._escalation_queue)  # type: ignore[possibly-undefined]
        host = self.config.escalation_host
        port = self.config.escalation_port

        async def _serve():
            try:
                await mcp_server.run_http_async(host=host, port=port)
            except Exception as e:
                logger.error(f'Escalation server error: {e}')

        self._escalation_task = asyncio.create_task(_serve(), name='recon-escalation-server')
        logger.info(f'Reconciliation escalation server starting on {host}:{port}')
        await _sleep(0.5)

        # Store escalation URL and queue; _make_stages propagates them to each
        # cycle's fresh stages via _propagate_escalation_queue (task 2146 β —
        # there is no long-lived self.stages list to push into here anymore).
        escalation_url = f'http://{host}:{port}/mcp'
        self._escalation_url = escalation_url

    async def _stop_escalation_server(self) -> None:
        """Stop the escalation server."""
        if self._escalation_task is not None:
            self._escalation_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._escalation_task
            self._escalation_task = None
            logger.info('Reconciliation escalation server stopped')

    def _escalate(
        self,
        category: str,
        run_id: str,
        summary: str,
        detail: str = '',
        *,
        finding: dict | None = None,
        resolved_fps: frozenset[str] | None = None,
    ) -> None:
        """Submit an escalation to the queue (fire-and-forget).

        Routing contract (A7b):
        - The harness NEVER calls queue.resolve() **on the recon escalation
          queue** (``config.escalation_queue_dir``) — the escalation-watcher
          session (port 8103) is the sole closer per plans/afk-A7-recon-closure.md.
          The judge-halt record in ``<project_root>/data/escalations/`` is a
          DIFFERENT queue, written and closed by BacklogPolicy (see task 2998).
        - Dedup folds only on the way IN, via submit_or_dedupe + _RECON_DEDUP_CONFIG.
        - When finding is not None (a finding dict with category / affected_ids /
          description), the fingerprint is keyed on finding identity so the same
          target across N cycles produces exactly one pending escalation.
        - When finding is None, the fingerprint falls back to a description-only
          hash of the summary, so identical recurring messages fold while distinct
          ones stay individually visible.
        """
        if not HAS_ESCALATION or self._escalation_queue is None:
            return
        try:
            queue = self._escalation_queue
            if finding is not None:
                fingerprint = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                    category,
                    finding.get('category') or '',
                    _derive_affected_ids(finding),
                    finding.get('description') or '',
                )
            else:
                # No finding in scope: use '' for finding_category (sentinel for
                # "summary-only, no finding identity") so the description-hash branch
                # of compute_content_fingerprint is used.  Using '' instead of
                # re-using escalation_category keeps the identity composition
                # semantically accurate — the second arg is the finding's own
                # category, which is absent here.
                fingerprint = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                    category,
                    '',
                    [],
                    summary,
                )
            if self._finding_recently_resolved(category, fingerprint, resolved_fps=resolved_fps):
                logger.info(
                    'reconciliation.escalation_suppressed_recently_resolved',
                    extra={
                        'category': category,
                        'run_id': run_id,
                        'fingerprint': fingerprint,
                        'summary': summary,
                    },
                )
                return
            esc = Escalation(  # type: ignore[possibly-undefined]
                id=queue.make_id(f'recon-{run_id[:8]}'),
                task_id=f'recon-{run_id[:8]}',
                agent_role='reconciliation-harness',
                severity='info' if category in (
                    'recon_stale_run', 'recon_integrity_issue', TASK_COUNT_SNAPSHOT_ESCALATION_CATEGORY,
                ) else 'blocking',
                category=category,
                summary=summary,
                detail=detail,
                dedupe_fingerprint=fingerprint,
            )
            submit_or_dedupe(queue, esc, _RECON_DEDUP_CONFIG)  # type: ignore[possibly-undefined]
        except Exception as e:
            logger.warning(f'Failed to submit escalation: {e}')

    # ── Tier selection ─────────────────────────────────────────────────

    async def _select_tier(self, project_id: ProjectId) -> TierConfig:
        """Choose model tier based on buffer size."""
        buffer_count = (await self.buffer.get_buffer_stats(project_id)).get('size', 0)
        use_opus = buffer_count > (self.config.buffer_size_threshold * self.config.opus_threshold_ratio)

        if use_opus:
            return TierConfig(
                model=self.config.opus_model,
                episode_limit=self.config.opus_episode_limit,
                memory_limit=self.config.opus_memory_limit,
            )
        return TierConfig(
            model=self.config.sonnet_model,
            episode_limit=self.config.sonnet_episode_limit,
            memory_limit=self.config.sonnet_memory_limit,
        )

    # ── Main loop ──────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Management loop — discover active projects, spawn per-project loops."""
        logger.info('Reconciliation harness background loop started')
        # Reviewer finding race_condition: wait for the recon-report server to be
        # accepting connections before the first stage subprocess is launched.  This
        # encodes the ordering invariant in code rather than relying on comment-only
        # guarantees.  server_ready_event is None when recon_report is not active.
        if self._server_ready_event is not None:
            logger.debug('Harness waiting for recon-report server readiness signal...')
            await self._server_ready_event.wait()
            logger.debug('Recon-report server ready — starting reconciliation loop')
        if self.usage_gate:
            await self.usage_gate.check_at_startup()

        # Rehydrate persistent halt state from the journal so a restart cannot
        # silently clear a halt that hasn't been explicitly cleared by an
        # operator. Called once at startup.
        if self.judge is not None:
            await self.judge.initialize()

        await self._start_escalation_server()

        # Re-queue any deferred writes left in-progress by a crashed prior process.
        # Cutoff is 0 (release every currently-claimed row) rather than a
        # time-based horizon.  A time-based horizon would miss rows whose
        # claimed_at falls within that horizon on a fast restart, silently
        # stalling those writes.
        # Safety: the per-project reconciliation lock (EventBuffer._is_run_locked /
        # mark_run_active) serialises replay — at startup no project loop has
        # spawned yet, so there is nothing to race with.
        try:
            released = await self.buffer.release_stale_claims(0)
            if released:
                logger.info(
                    f'Recovered {released} stale deferred write claim(s) on startup'
                )
        except Exception as e:
            logger.warning(f'release_stale_claims at startup failed: {e}')

        # One-shot: adopt any dead predecessor instance's orphaned 'running'
        # rows immediately, regardless of age (task 2711 / E6) — before the
        # age-gated _recover_stale_runs reaper gets its first tick. Guarded
        # like release_stale_claims above so a recovery hiccup can't crash
        # harness startup.
        try:
            await self._recover_predecessor_runs()
        except Exception as e:
            logger.warning(f'_recover_predecessor_runs at startup failed: {e}')

        # One-shot: adopt and --resume any interrupted runs a dead predecessor
        # left mid-stage (task σ / 2717), before the age-gated _recover_stale_runs
        # reaper gets its first tick. Guarded like the predecessor pass above so a
        # resume hiccup can't crash harness startup.
        try:
            await self._resume_interrupted_runs()
        except Exception as e:
            logger.warning(f'_resume_interrupted_runs at startup failed: {e}')

        # One-shot: re-fire the judge for any run whose verdict a restart dropped
        # in the cycle-end→verdict-commit window (task 2708). The durable
        # judge_pending marker (written before the fire-and-forget judge task) is
        # the recovery cursor. Judge is already .initialize()d above (2029-2030);
        # guarded (self.judge is not None) inside the method. Fail-safe wrapped like
        # the sibling one-shot passes so a recovery hiccup never crashes the loop.
        try:
            await self._recover_pending_judge_reviews()
        except Exception as e:
            logger.warning(f'_recover_pending_judge_reviews at startup failed: {e}')

        loop_count = 0
        try:
            while True:
                try:
                    await self._recover_stale_runs()

                    # Discover active projects, spawn loops for new ones
                    if not self._draining:
                        for project_id in await self.buffer.get_active_projects():
                            existing = self._project_tasks.get(project_id)
                            if existing is None or existing.done():
                                task = asyncio.create_task(
                                    self._project_loop(project_id),
                                    name=f'recon-{project_id}',
                                )
                                self._project_tasks[project_id] = task

                    # Reap completed tasks, log unexpected failures
                    for pid in list(self._project_tasks):
                        task = self._project_tasks[pid]
                        if task.done():
                            del self._project_tasks[pid]
                            if not task.cancelled() and task.exception():
                                logger.error(
                                    f'Project loop for {pid} crashed: {task.exception()}'
                                )

                    # Drain status logging
                    if self._draining:
                        if self._no_active_loops():
                            if not self._drain_complete_logged:
                                self._drain_complete_logged = True
                                logger.info('Harness fully drained — safe to restart')
                            # else: silent — drained marker already emitted
                        else:
                            active = sum(
                                1 for t in self._project_tasks.values() if not t.done()
                            )
                            logger.info(
                                f'Harness draining: {active} project loop(s) still running'
                            )

                    # Periodic cleanup of drained events (~every 50s / 10 iterations)
                    loop_count += 1
                    if loop_count % 10 == 0:
                        try:
                            deleted = await self.buffer.cleanup_drained()
                            if deleted:
                                logger.debug(f'Cleaned up {deleted} drained events')
                        except Exception as e:
                            logger.warning(f'Drained event cleanup failed: {e}')

                except Exception as e:
                    logger.error(f'Reconciliation loop error: {e}')
                await _sleep(5)
        finally:
            # Graceful shutdown: cancel all project loops
            for task in self._project_tasks.values():
                task.cancel()
            if self._project_tasks:
                await asyncio.gather(
                    *self._project_tasks.values(), return_exceptions=True,
                )
            self._project_tasks.clear()
            # Drain in-flight judge tasks too (task 2708). Cancelled judges leave
            # their judge_pending markers intact for startup re-run, so no verdict
            # is silently dropped by the shutdown.
            await self._drain_judge_tasks()
            await self._stop_escalation_server()

    def auto_resume_pending(self, project_id: str) -> bool:
        """True iff a currently-halted ``project_id`` will auto-resume on the
        next loop tick: this deployment enabled ``auto_unhalt_after_cooldown``
        AND the judge's halt cooldown has expired.

        The single source of truth for that rule. :meth:`_project_loop` calls
        it to decide auto-resume-after-cooldown (task 2920 deliverable c), and
        the ``trigger_reconciliation`` MCP tool calls the SAME method to decide
        whether forwarding a manual trigger is honest — a trigger is only ever
        consumed by a cycle that actually runs (task 3050).

        Keeping one predicate is the point: re-deriving the rule at the MCP
        boundary would let the tool keep answering ``trigger_requested: True``
        the moment this rule gained a condition (a usage gate, a drain flag, a
        grace check), silently reintroducing exactly the class of lie task 3050
        exists to remove.

        Returns False when no judge is wired, and for a project that is not
        halted (:meth:`Judge.cooldown_expired` is False for an unhalted
        project) — so this answers "pending auto-resume", never "healthy".
        """
        if self.judge is None:
            return False
        return bool(
            self.config.auto_unhalt_after_cooldown
            and self.judge.cooldown_expired(project_id)
        )

    async def _project_loop(self, project_id: str) -> None:
        """Independent reconciliation loop for a single project."""
        logger.info(f'Project reconciliation loop started for {project_id}')
        idle_ticks = 0
        while True:
            if self._draining:
                logger.info('reconciliation.drain_ack', extra={'project_id': project_id})
                return
            try:
                should, reason = await self.buffer.should_trigger(project_id)
                if not should:
                    idle_ticks += 1
                    if idle_ticks > 12:  # ~60s idle → exit, respawn on demand
                        logger.debug(f'Project loop idle exit for {project_id}')
                        return
                    await _sleep(5)
                    continue

                idle_ticks = 0
                acquired = await self.buffer.mark_run_active(project_id)
                if not acquired:
                    await _sleep(5)
                    continue

                # Halt check
                if self.judge and self.judge.is_halted(project_id):
                    # One predicate, shared with trigger_reconciliation — see
                    # auto_resume_pending's docstring.
                    if self.auto_resume_pending(project_id):
                        # Auto-resume-after-cooldown (task 2920 deliverable c):
                        # the halt cooldown has expired and this deployment opted
                        # in, so unhalt (which seeds the normal post-unhalt grace
                        # and clears the per-process _halt_escalated sentinel via
                        # the unhalt callback so a re-halt re-escalates) and FALL
                        # THROUGH to run the cycle. The judge re-halts on the very
                        # next serious verdict / infra-failure threshold if the
                        # pipeline is still sick — re-firing a distinct, loud halt
                        # escalation — rather than resuming silently forever.
                        halt_reason_text = (
                            self.judge.halt_reason(project_id)
                            or 'judge halted reconciliation'
                        )
                        logger.warning(
                            f'Auto-unhalting {project_id} after halt cooldown '
                            f'expiry — {halt_reason_text}; resuming with '
                            f'post-unhalt grace (judge will re-halt if still sick)'
                        )
                        await self.judge.unhalt(project_id)
                        # Record the resumed cycle's provenance as the auto-unhalt
                        # resume — NOT the stale halt reason — so run history
                        # reflects the true cause. `reason` (the buffer's
                        # should_trigger reason) is preserved for context and is
                        # passed to run_full_cycle below as trigger_reason.
                        # Deliberately NO return: fall through to the normal
                        # consume_grace_cycle + cycle path below.
                        reason = f'auto_unhalt_after_cooldown (buffer: {reason})'
                    else:
                        logger.warning(f'Skipping cycle for halted project {project_id}')
                        await self._notify_judge_halt(
                            project_id,
                            # Thread the REAL halt reason (infra 'judge-unreachable …',
                            # 'Unparseable judge response …', or 'Serious verdict …')
                            # into the escalation instead of a hardcoded generic
                            # string (task 2947 ask b). Falls back when unset.
                            reason=(
                                self.judge.halt_reason(project_id)
                                or 'judge halted reconciliation'
                            ),
                        )
                        try:
                            await self._replay_deferred_writes(ProjectId(project_id))
                        finally:
                            # Scope the release to this instance — see
                            # plans/recon-stale-recovery-rca.md.
                            await self.buffer.mark_run_complete(
                                project_id, instance_id=self.buffer.instance_id,
                            )
                        return  # Don't keep spinning on a halted project

                # Decrement post-unhalt grace counter. A just-unhalted project
                # runs `halt_grace_cycles` cycles with trend detection skipped,
                # so stale moderates in the DB can age out before the detector
                # re-engages.
                if self.judge is not None:
                    remaining = await self.judge.consume_grace_cycle(project_id)
                    if remaining > 0:
                        logger.info(
                            f'Running {project_id} within post-unhalt grace '
                            f'({remaining} cycles remaining)'
                        )

                tier = await self._select_tier(ProjectId(project_id))
                iterator = BacklogIterator(self.config, self.journal, self.buffer, self)
                heartbeat_task = asyncio.create_task(self._heartbeat_loop(project_id))
                use_iterator = False
                try:
                    use_iterator = await iterator.should_iterate(project_id)
                    if use_iterator:
                        # task 2040: symmetric outer bound with the else branch below.
                        # NOT redundant with the per-chunk wait_for(cycle_timeout_seconds)
                        # that already wraps run_full_cycle inside BacklogIterator.run —
                        # a hang inside a single chunk's run_full_cycle is already bounded
                        # there. What this outer wait_for actually closes: (a) the awaits
                        # between chunks that carry no timeout of their own — peek_buffered,
                        # assemble, drain_by_ids, record_chunk_boundary — and (b) an
                        # absolute cap on the TOTAL cumulative wall-clock of the whole
                        # run() invocation, rather than relying solely on the in-iterator
                        # between-chunks budget check.
                        await asyncio.wait_for(
                            iterator.run(project_id),
                            timeout=self.config.cycle_timeout_seconds,
                        )
                    else:
                        await asyncio.wait_for(
                            self.run_full_cycle(project_id, reason, tier=tier),
                            timeout=self.config.cycle_timeout_seconds,
                        )
                except TimeoutError:
                    # task 2040: name the event for the branch that actually timed
                    # out — conflating the two here previously meant an else-branch
                    # (full cycle) timeout was logged as 'iteration_timed_out' and an
                    # iterate-branch timeout was logged as "Full cycle timed out",
                    # which misleads anyone grepping logs/metrics for either failure
                    # mode.
                    branch = 'iterate' if use_iterator else 'full_cycle'
                    logger.error(
                        f'{"Backlog iteration" if use_iterator else "Full cycle"} '
                        f'timed out after '
                        f'{self.config.cycle_timeout_seconds}s for {project_id}'
                    )
                    logger.error(
                        'reconciliation.iteration_timed_out'
                        if use_iterator
                        else 'reconciliation.full_cycle_timed_out',
                        extra={
                            'project_id': project_id,
                            'timeout': self.config.cycle_timeout_seconds,
                            'branch': branch,
                        },
                    )
                    await self.buffer.restore_drained(project_id)
                finally:
                    try:
                        await self._replay_deferred_writes(ProjectId(project_id))
                    finally:
                        # Cancel heartbeat only after replay so the lock heartbeat
                        # keeps the per-project lock alive for the full replay
                        # duration, preventing a concurrent instance from claiming
                        # the lock as stale mid-replay.
                        heartbeat_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await heartbeat_task
                        # Scope the release to this instance — see
                        # plans/recon-stale-recovery-rca.md.
                        await self.buffer.mark_run_complete(
                            project_id, instance_id=self.buffer.instance_id,
                        )

            except asyncio.CancelledError:
                raise  # Propagate shutdown
            except UnknownProjectError as e:
                # task 1143: KNOWN_PROJECT_ROOTS misconfiguration — fail fast, no retry.
                # Catching the narrow UnknownProjectError (not bare ValueError) ensures
                # generic ValueErrors from stages fall through to the except Exception
                # retry path below:
                #   - stages/base.py: watermark↔stage project_id mismatch (transient
                #     during instance handover; naturally recoverable next cycle)
                #   - stages/memory_consolidator.py: unset episode_limit/memory_limit
                #     (programming bug; should surface and retry, not abort the loop)
                #
                # task 1549: quarantine buffered events so get_active_projects()
                # (WHERE status='buffered') stops returning this project_id, ending
                # the management-loop respawn storm (2026-05-28 'know-live' incident).
                # The dead_letter flip + WARNING below make the incident observable to
                # the recon-watcher; no escalation is emitted (deferred per design).
                # See task 1143 (read-side strictness) and task 1549 (this complement).
                quarantined = await self.buffer.mark_project_dead_letter(project_id)
                logger.warning(
                    f'Project loop quarantined {quarantined} buffered dead_letter event(s) '
                    f'for unknown project_id={project_id!r} and is aborting. '
                    f'Misconfiguration: {e}. '
                    f'Set DASHBOARD_KNOWN_PROJECT_ROOTS to register this project '
                    f'(task 1143 / task 1549).'
                )
                return
            except Exception as e:
                logger.error(f'Project loop error for {project_id}: {e}')

            await _sleep(5)  # Cooldown between cycles

    async def _heartbeat_loop(self, project_id: str) -> None:
        """Keep the reconciliation lock alive while a run is in progress."""
        while True:
            await _sleep(60)
            try:
                await self.buffer.heartbeat(project_id)
            except Exception as e:
                logger.warning(f'Heartbeat failed for {project_id}: {e}')

    async def _spawn_judge(self, run_id: str, project_id: str) -> None:
        """Durably mark a judge review pending, then fire a tracked judge task.

        The single seam for both fresh cycle-end launches (run_full_cycle,
        _run_remediation_pass) and startup recovery (task 2708). Awaiting the
        judge_pending marker BEFORE creating the task means a process killed the
        instant after cycle completion still has a recoverable marker; the
        marker is cleared atomically inside ``add_verdict``. The task is tracked
        in ``_judge_tasks`` (with a done-callback that discards it) so shutdown
        can drain it deterministically.
        """
        await self.journal.mark_judge_pending(run_id, project_id)
        task = asyncio.create_task(self._run_judge(run_id), name=f'judge-{run_id}')
        self._judge_tasks.add(task)
        task.add_done_callback(self._judge_tasks.discard)

    async def _drain_judge_tasks(self) -> None:
        """Cancel and await every in-flight judge task on shutdown (task 2708).

        A cancelled judge is cancelled mid-review, before add_verdict's atomic
        marker-clear, so its judge_pending marker survives for startup re-run —
        durability rests on the uncleared marker, not on catching here. Snapshot
        the set first because the per-task done-callback discards from it during
        iteration.
        """
        pending = list(self._judge_tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._judge_tasks.clear()

    async def _recover_pending_judge_reviews(self) -> None:
        """Re-fire the judge for every run left with a judge_pending marker (task 2708).

        One-shot startup pass: a restart landing in the cycle-end→verdict-commit
        window cancels the fire-and-forget judge before add_verdict's atomic
        marker-clear, so the marker survives. Re-running the judge is safe — the
        review is read-only over the run record, and the invariant 'marker present
        ⟹ no committed verdict' (add_verdict clears atomically) means the re-run's
        judge_verdicts INSERT never collides on the run_id PK. Routes through the
        same _spawn_judge seam as fresh launches (idempotent re-mark + tracked task).
        """
        if self.judge is None:
            return
        pending = await self.journal.get_pending_judge_runs()
        if not pending:
            return
        # Account-availability gate (task 2947 ask c): if a usage gate is wired
        # and every account is capped/auth-failed (active_account_name is None),
        # re-firing judges now would fail transport under the same fully-capped
        # pool — churning infra-failure counters and (pre-fix) minting phantom
        # halts — for no benefit. Defer the whole recovery pass: the judge_pending
        # markers are left intact, so recovery resumes naturally on a later restart
        # once capacity returns. When capacity exists (or no gate is configured),
        # recovery proceeds unchanged.
        if self.usage_gate is not None and self.usage_gate.active_account_name is None:
            logger.warning(
                f'Deferring recovery of {len(pending)} pending judge review(s) — '
                'all accounts capped; will retry on a later restart / when '
                'capacity returns'
            )
            return
        # Bound the fan-out (task 2708 amendment): get_pending_judge_runs is
        # ordered oldest-marked first, so the oldest markers are re-fired first
        # and any excess is deferred to a later restart. This caps the burst of
        # concurrent judge LLM calls a wedged judge (markers never clear) would
        # otherwise trigger on every restart.
        total = len(pending)
        to_spawn = pending[:_JUDGE_RECOVERY_MAX_SPAWN]
        if total > _JUDGE_RECOVERY_MAX_SPAWN:
            logger.warning(
                f'{total} pending judge review(s) at startup exceeds the '
                f'{_JUDGE_RECOVERY_MAX_SPAWN}-per-restart recovery cap — a judge may '
                'be persistently failing (a judge_pending marker clears only on a '
                f'committed verdict). Re-firing the {len(to_spawn)} oldest; the '
                f'remaining {total - len(to_spawn)} will be recovered on '
                'subsequent restarts.'
            )
        else:
            logger.info(
                f'Recovering {total} pending judge review(s) after restart'
            )
        for run_id, project_id in to_spawn:
            await self._spawn_judge(run_id, project_id)

    async def _run_judge(self, run_id: str) -> None:
        """Fire-and-forget judge wrapper with error logging."""
        try:
            assert self.judge is not None
            verdict = await self.judge.review_run(run_id)
            if verdict:
                logger.info(f'Judge verdict for {run_id}: severity={verdict.severity}')
            else:
                logger.warning(f'Judge returned no verdict for run {run_id}')
        except asyncio.CancelledError:
            # Shutdown drain (or a cycle-aware restart) cancelled the judge
            # mid-review, before add_verdict's atomic marker-clear ran. The
            # judge_pending marker is therefore intact and startup recovery
            # (_recover_pending_judge_reviews) will re-run this review — the
            # verdict is deferred, not lost (task 2708). CancelledError is a
            # BaseException, so the `except Exception` below cannot swallow it;
            # re-raise to preserve asyncio cancellation semantics for the drain.
            logger.warning(
                f'Judge task for run {run_id} cancelled during shutdown; '
                'judge_pending marker retained for startup re-run'
            )
            raise
        except Exception:
            logger.error(f'Judge task failed for run {run_id}', exc_info=True)

    def recon_busy_snapshot(self) -> list[dict]:
        """Return the reconciliation full cycles currently in flight.

        Synchronous, pure, never raises: a list of
        ``{project_id, run_id, stage, started_at}`` dicts (one per in-flight
        full cycle). Consumed by the /health endpoint's additive
        ``recon_busy`` field (task 2703 δ) so a cycle-aware restart can defer
        while a cycle is running — machine-readable, no journal scraping.

        Scope — the stage pipeline only (decision, task 2703 δ): an entry is
        live only for the duration of the ``with self._active_runs.track(...)``
        block in ``run_full_cycle``, i.e. the three-stage pipeline
        (memory_consolidator → task_knowledge_sync → integrity_check) plus the
        inline remediation pass. It is deliberately NOT held for the
        fire-and-forget judge task (``asyncio.create_task(self._run_judge(...))``),
        which outlives the cycle: stage reports are already durably persisted
        (``update_run_stage_reports``) BEFORE the judge fires, so the judge only
        appends an advisory verdict. A restart firing in the narrow window after
        ``run_full_cycle`` returns may therefore cancel an in-flight judge — an
        accepted trade-off, since the expensive, interruption-sensitive work a
        cycle-aware restart must protect is the stage pipeline, not the judge.
        The inline remediation pass runs inside the same ``track(...)`` block, so
        it reuses the parent cycle's entry rather than opening its own.
        """
        return self._active_runs.snapshot()

    async def run_full_cycle(
        self,
        project_id: str,
        trigger_reason: str,
        tier: TierConfig | None = None,
        events: list[ReconciliationEvent] | None = None,
        assembled_payload: AssembledPayload | None = None,
        resume_run: ReconciliationRun | None = None,
    ) -> ReconciliationRun:
        """Execute the three-stage pipeline for a project.

        Args:
            events: Optional pre-drained event list.  When provided, buffer.drain()
                    is skipped and these events are used directly.  This allows
                    BacklogIterator to pass already-drained chunk events without
                    a double-drain.
            assembled_payload: Optional token-budgeted payload from ContextAssembler.
                    When provided, Stage 1 uses this instead of generic
                    time-windowed episode/memory fetches.
            resume_run: Task σ adopt-and-resume seam.  When set, the cycle reuses
                    this interrupted run wholesale instead of starting a fresh one:
                    it keeps ``resume_run.id`` (no ``start_run``), does NOT drain
                    (the caller MUST supply the run's already-drained ``events`` so
                    a resumed cycle never double-processes them), skips any stage
                    whose real ``StageReport`` is already in
                    ``resume_run.stage_reports`` (its work landed before the
                    interrupt), and ``--resume``s ONLY the stage matching
                    ``resume_run.stage_cursor`` by threading
                    ``resume_run.session_id`` into it.  The remaining stages run
                    fresh, and the completion/judge/remediation tail is unchanged.
        """
        # task 1143: pre-flight guard — raises before any side effects (no journal row,
        # no buffer drain) if project_id has no KNOWN_PROJECT_ROOTS entry.
        scope = self._known_project_scope_for(project_id)
        project_root = scope.project_root

        tier = tier or TierConfig()
        watermark = await self.journal.get_watermark(project_id)

        if resume_run is not None:
            # Task σ adopt-and-resume: reuse the interrupted run wholesale — no
            # fresh run row, no re-drain. The startup pass supplies the run's
            # already-drained events via `events`; re-draining or re-attributing
            # them would double-process on resume, so both are skipped here.
            if events is None:
                raise ValueError(
                    'run_full_cycle(resume_run=...) requires caller-supplied '
                    "events (the interrupted run's already-drained events); "
                    'draining fresh would lose the in-flight batch'
                )
            run = resume_run
            run_id = run.id
            # The row is 'interrupted' on disk; mark it running in-memory so the
            # active-run tracker/logs reflect reality and the finally's GC gate
            # (skip iff status == interrupted) permits the transcript sweep once
            # this resumed cycle completes.
            run.status = RunStatus.running
        else:
            run_id = str(uuid4())
            if events is None:
                events = await self.buffer.drain(project_id, run_id=run_id)
            else:
                await self.buffer.mark_drained_run_id(
                    project_id, [e.id for e in events], run_id
                )

            run = ReconciliationRun(
                id=run_id,
                project_id=project_id,
                run_type=RunType.full,
                trigger_reason=trigger_reason,
                started_at=datetime.now(UTC),
                events_processed=len(events),
                status=RunStatus.running,
                instance_id=self.buffer.instance_id,
            )
            await self.journal.start_run(run)

        logger.info(
            'reconciliation.run_started',
            extra={
                'run_id': run_id,
                'project_id': project_id,
                'run_type': 'full',
                'trigger_reason': trigger_reason,
                'events_to_process': len(events),
                'model': tier.model,
                'resumed': resume_run is not None,
            },
        )

        # project_root is already hard-bound via _known_project_scope_for at the top
        # of this function (task 1143); no _resolve_project_root call needed here.

        # Load prior S3 findings from last completed run (backstop for normal pass)
        prior_s3_findings = await self._get_prior_s3_findings(project_id)

        # Fetch filtered task tree once for the whole cycle (ref: task 455)
        filtered_task_tree = await self._fetch_filtered_task_tree(project_root)

        # Fetch authoritative task-count census and cross-verify against tree (task 1785)
        statuses = await self._fetch_task_count_census(project_root)
        task_count_verification = cross_verify_task_counts(filtered_task_tree, statuses)

        # Diff the cached project_status_correction memory against the same live
        # census and supersede it on divergence (task 1938).
        status_correction_reconciliation = await self._reconcile_status_correction(
            project_id, statuses
        )

        # Read Graphiti async-queue dead-letter count — surfaces silent-drop tail (task 1785)
        graphiti_queue_health = await self._check_graphiti_queue_health(scope.project_id)

        current_stage_name: str | None = None
        cycle_start_time = datetime.now(UTC)
        stages = self._make_stages(scope)
        with self._active_runs.track(
            run_id, project_id, run.started_at.isoformat()
        ) as _active:
            try:
                reports = []
                for stage in stages:
                    stage_key = stage.stage_id.value

                    # Task σ resume: a stage whose real StageReport is already
                    # persisted on the resumed run completed BEFORE the interrupt
                    # — skip its subprocess entirely, but thread its persisted
                    # report into `reports` so later stages still receive it as
                    # prior context (exactly as a freshly-run stage would).
                    if resume_run is not None:
                        prior = run.stage_reports.get(stage_key)
                        if isinstance(prior, StageReport):
                            reports.append(prior)
                            continue

                    current_stage_name = stage_key
                    _active.stage(current_stage_name)

                    # Apply tier limits, prior S3 findings, cycle fence, and task tree to Stage 1
                    if isinstance(stage, MemoryConsolidator):
                        self._configure_consolidator(
                            stage, tier,
                            prior_s3_findings=prior_s3_findings,
                            cycle_fence_time=cycle_start_time,
                            assembled_payload=assembled_payload,
                            filtered_task_tree=filtered_task_tree,
                            task_count_verification=task_count_verification,
                            graphiti_queue_health=graphiti_queue_health,
                            status_correction_reconciliation=status_correction_reconciliation,
                        )

                    # Wire harness-fetched task tree into Stage 2 via symmetric helper (ref: task 455)
                    if isinstance(stage, TaskKnowledgeSync):
                        self._configure_task_sync(stage, filtered_task_tree=filtered_task_tree)

                    # Wire harness-fetched task tree into Stage 3 for task-dump spot-check (task 1661)
                    if isinstance(stage, IntegrityCheck):
                        stage.filtered_task_tree = filtered_task_tree

                    # Task σ resume: --resume ONLY the stage the interrupt caught
                    # mid-flight (stage_cursor), and only when a session was
                    # actually captured for it. Pass the kwarg solely on that
                    # path so every non-resume call site keeps today's signature.
                    resume_session_id = (
                        run.session_id
                        if (
                            resume_run is not None
                            and stage_key == run.stage_cursor
                            and run.session_id
                        )
                        else None
                    )
                    if resume_session_id is not None:
                        report = await stage.run(
                            events, watermark, reports, run_id,
                            model=tier.model, resume_session_id=resume_session_id,
                        )
                    else:
                        report = await stage.run(
                            events, watermark, reports, run_id, model=tier.model,
                        )
                    reports.append(report)
                    run.stage_reports[stage_key] = report

                # Update watermark
                watermark.last_full_run_id = run_id
                watermark.last_full_run_completed = datetime.now(UTC)
                watermark.last_episode_timestamp = datetime.now(UTC)
                watermark.last_memory_timestamp = datetime.now(UTC)
                watermark.last_task_change_timestamp = datetime.now(UTC)
                await self.journal.update_watermark(watermark)

                run.completed_at = datetime.now(UTC)
                run.status = RunStatus.completed
                await self.journal.complete_run(run_id, 'completed')

                # Cross-check self-reported stats against write-journal ops BEFORE the
                # judge reads them. Stage agents sometimes over-report successful
                # writes when Mem0 silently dedups; the verifier overwrites those
                # counts with observed truth and keeps the originals under _reported.
                await verify_and_rewrite_stats(
                    run_id, run.stage_reports, self.journal.write_journal,
                )

                # Persist stage reports before judge — the judge reads from the DB,
                # so reports must be committed before firing the async task.
                await self.journal.update_run_stage_reports(run_id, run.stage_reports)

                # Task 2278: structural cadence guard for the Stage-2 task_count_snapshot
                # write — escalates once a confirmed miss has recurred for
                # TASK_COUNT_SNAPSHOT_MISS_THRESHOLD consecutive full cycles.  Full-cycle
                # path only (not evaluated from _maybe_remediate); reads the just-persisted
                # stage report, so this runs after update_run_stage_reports above.
                await self._maybe_escalate_stale_task_count_snapshot(project_id, run_id, run)

                # Async judge review. Fire-and-forget: it outlives this cycle and
                # is intentionally NOT tracked by _active_runs / recon_busy (which
                # scopes to the stage pipeline only — see recon_busy_snapshot).
                # Stage reports are already persisted above, so a cycle-aware
                # restart cancelling an in-flight judge here is an accepted
                # trade-off (task 2703 δ) — made non-lossy by the durable
                # judge_pending marker _spawn_judge writes before firing (task 2708).
                if self.judge:
                    await self._spawn_judge(run_id, project_id)

                # Remediation pass: thread scope resolved above (task 1163) and pass
                # pre-fetched tree to avoid a redundant fetch (ref: task 478).
                await self._maybe_remediate(project_id, run_id, run, tier,
                                            scope=scope,
                                            filtered_task_tree=filtered_task_tree)

                logger.info(
                    'reconciliation.run_completed',
                    extra={
                        'run_id': run_id,
                        'project_id': project_id,
                        'status': 'completed',
                    },
                )
                return run

            except asyncio.CancelledError:
                # asyncio.wait_for cancels via CancelledError, which is NOT a subclass of
                # Exception in Python 3.8+.  Without this handler the journal run is left
                # stuck in 'running'.  Mark it failed, restore events, then re-raise so
                # asyncio cancellation semantics are preserved.
                #
                # Two defences against cleanup being interrupted:
                # 1. asyncio.shield() — runs the cleanup coroutine in its own Task so a
                #    second cancellation (e.g. server shutdown) cannot abort the DB write.
                # 2. Independent try/except BaseException per cleanup step — each step
                #    runs regardless of the other's outcome, and CancelledError is still
                #    re-raised to the caller.
                #
                # Task σ: when resume_after_restart is enabled, a restart cancelling an
                # in-flight stage is a RESUMABLE interrupt, not a failure. Mark the run
                # `interrupted` and DELIBERATELY skip restore_drained — the drained
                # events must stay drained so the startup adopt-and-resume pass can feed
                # a resumed cycle's fresh later stages without double-processing. The
                # (session_id, stage_cursor) snapshot survives on the run row (BaseStage
                # .run skips clear_run_session on cancel), and the finally below skips the
                # config-dir GC for interrupted runs so the transcript survives for
                # --resume. When the knob is off, keep today's failed + restore path
                # verbatim (a deploy that changed recon prompts/tooling opts out of resume
                # this way, since a --resume'd session finishes under the OLD system
                # prompt by construction).
                if self.config.resume_after_restart:
                    run.status = RunStatus.interrupted
                    try:
                        await asyncio.shield(
                            self.journal.complete_run(run_id, 'interrupted')
                        )
                    except BaseException as cleanup_err:
                        logger.error(
                            'complete_run(interrupted) failed after cancellation: '
                            f'{cleanup_err}'
                        )
                    logger.warning(
                        f'Reconciliation run {run_id} INTERRUPTED (resumable) for '
                        f'{project_id} (stage: {current_stage_name})'
                    )
                    raise
                run.status = RunStatus.failed
                run.stage_reports['_error'] = {
                    'error_type': 'CancelledError',
                    'error_message': 'Run cancelled (timeout or external cancellation)',
                    'failed_stage': current_stage_name,
                    'traceback': '',
                }
                try:
                    await asyncio.shield(self.journal.complete_run(run_id, 'failed'))
                except BaseException as cleanup_err:
                    logger.error(f'complete_run failed after cancellation: {cleanup_err}')
                try:
                    await asyncio.shield(self.buffer.restore_drained(project_id))
                except BaseException as cleanup_err:
                    logger.error(f'restore_drained failed after cancellation: {cleanup_err}')
                logger.error(
                    f'Reconciliation run {run_id} cancelled for {project_id} '
                    f'(stage: {current_stage_name})'
                )
                raise
            except AllAccountsCappedException as e:
                run.status = RunStatus.failed
                run.stage_reports['_error'] = {
                    'error_type': 'AllAccountsCappedException',
                    'error_message': str(e),
                    'failed_stage': current_stage_name,
                    'deferred': True,
                }
                await self.journal.complete_run(run_id, 'failed')
                await self.buffer.restore_drained(project_id)
                logger.warning(
                    f'Reconciliation deferred: all accounts capped during stage '
                    f'{current_stage_name} ({e.retries} retries in {e.elapsed_secs:.1f}s)'
                )
                return run
            except Exception as e:
                run.status = RunStatus.failed
                run.stage_reports['_error'] = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'failed_stage': current_stage_name,
                    'traceback': traceback.format_exc(),
                }
                await self.journal.complete_run(run_id, 'failed')
                await self.buffer.restore_drained(project_id)
                logger.error(f'Reconciliation failed: {e}')
                self._escalate(
                    'recon_failure', run_id,
                    f'Stage {current_stage_name} failed: {e}',
                )
                raise
            finally:
                await self._ensure_stage1_cycle_summary(
                    run, run_id, project_id, current_stage_name, cycle_start_time,
                )
                await self.journal.update_run_stage_reports(run_id, run.stage_reports)
                # Task 2744/σ: GC this run's per-run recon CLI config dir on every
                # exit path (success/failure) EXCEPT an interrupted (resumable) run —
                # its transcript must survive on disk for the startup --resume pass.
                # Defensive — a filesystem hiccup must never mask the run's real
                # terminal outcome.
                if run.status != RunStatus.interrupted:
                    try:
                        gc_run_config_dir(self.journal.data_dir, run_id)
                    except Exception as gc_err:  # noqa: BLE001
                        logger.warning(
                            'gc_run_config_dir failed for run %s: %r', run_id, gc_err
                        )

    async def _ensure_stage1_cycle_summary(
        self,
        run: ReconciliationRun,
        run_id: str,
        project_id: str,
        current_stage_name: str | None,
        cycle_start_time: datetime,
    ) -> None:
        """Guarantee a Stage 1 ``cycle_summary`` ledger row exists for *run_id*.

        Structural backstop with two independent arms, both firing from the
        harness's two S1→S2→S3 drivers' ``finally`` blocks
        (:meth:`run_full_cycle` and :meth:`_run_remediation_pass`) so this
        runs on every exit path of either:

        - **Arm 1** (task 2440) — Stage 1's own turn raised before ``run()``
          could return a report at all (its in-stage write,
          ``write_stage1_cycle_summary``, is the last statement of
          :meth:`~fused_memory.reconciliation.stages.memory_consolidator.MemoryConsolidator.run`,
          task 2229 W5-λ, and so never runs). Fires when
          ``current_stage_name`` is still Stage 1's and no report was
          recorded (the happy path and a Stage-2/3 failure both no-op — see
          task 2440's plan for the proof that this gate is exactly
          equivalent to "Stage 1 raised before its own write"; task 2626
          generalized the same gate to the remediation driver, which has its
          own run/run_id/current_stage_name locals).

          The synthesized report is honestly degraded, not fabricated:
          ``llm_calls``/``tokens_used`` are 0 and ``started_at`` is the
          whole-cycle anchor rather than Stage 1's real start — both are
          unrecoverable once ``run()`` raised without returning a report —
          so the implied duration is an upper bound, not a measurement. The
          anchor is ``cycle_start_time`` for a full cycle and the
          remediation run's own ``started_at`` for a remediation pass (it
          has no separate cycle_start_time local). The
          ``stage1_cycle_summary_degraded_backstop`` stat self-identifies
          the row as harness-synthesized.

        - **Arm 2** (task 2734) — Stage 1 completed and DID return a real
          report, but its own in-stage ledger upsert failed transiently
          (``write_cycle_summary``'s ``ledger.upsert`` caught the failure,
          logged a WARNING, and returned False — see
          ``MemoryConsolidator.run()``): the report carries the explicit
          failure signal (``stats['stage1_cycle_summary_ledger_written'] ==
          0``), but arm 1 never fires for this case because the report IS
          present (``current_stage_name`` has already moved past Stage 1 by
          the time any later failure — or the happy path itself — reaches
          ``finally``). Gated on three conditions (see
          :func:`_stage1_ledger_write_missing` for the stat/marker
          predicate itself):

          (a) ``run.run_type != RunType.remediation`` — a remediation
              pass's Stage 1 intentionally writes no cycle_summary (its
              ``run()`` early-returns before the write, leaving the ``= 0``
              default; task 2652 disambiguates a remediation run's
              expected-missing Stage 1); firing here would fabricate a
              spurious summary every remediation pass.
          (b) a ``ReconLedgerStore`` is actually wired
              (``self.memory.recon_ledger is not None``) — otherwise an
              intentionally ``recon_ledger_enabled=False`` deployment
              (whose stat is always 0) would re-fire, and WARNING, every
              cycle for no reason.
          (c) the report is not itself arm 1's harness-synthesized degraded
              backstop row (``_stage1_ledger_write_missing`` excludes any
              report stamped ``stage1_cycle_summary_degraded_backstop:
              True``), so the two arms can never double-process the same
              run — they are otherwise mutually exclusive by construction
              anyway, since arm 1 requires an ABSENT report and arm 2
              requires a real ``StageReport`` instance.

          When it fires, arm 2 RE-ATTEMPTS ``write_stage1_cycle_summary``
          with the REAL Stage 1 report (real llm_calls/tokens/stats), not a
          zeroed synth — the ledger upsert is idempotent (``ON CONFLICT`` on
          the 5-part identity), so a re-attempt after a transient failure is
          safe and cannot duplicate. It stamps a distinct
          ``stage1_cycle_summary_write_recovered_backstop`` marker on the
          report — optimistically ``True`` *before* the call, since
          ``write_stage1_cycle_summary`` serializes ``report.stats`` into
          the ledger row's ``payload_json`` synchronously at call time (see
          ``write_cycle_summary``'s docstring), so this is the only way a
          successful re-attempt's OWN ledger row ends up carrying the
          marker. It is corrected back to ``False`` immediately afterward
          if the re-attempt did NOT actually land a row
          (``ledger_written`` is falsy, or the call raised) — the
          correction cannot rewrite a ledger row (none exists in either
          failure case), but it DOES reach the run's own
          journal-persisted ``stage_reports`` copy
          (``update_run_stage_reports``, called right after this method
          returns), so that copy never falsely claims recovery succeeded
          (task 2734 amendment). ``stage1_cycle_summary_ledger_written``
          itself is left at its in-stage value of 0 either way — the
          ledger row's mere EXISTENCE is the authoritative presence signal
          that ``get_cycle_summary_presence`` reads, not either stat. Arm 2
          logs a WARNING distinct from arm 1's, with no ``_error``
          breadcrumb stamp (unlike arm 1, which repurposes a pre-existing
          ``_error`` record — the primary motivating scenario for arm 2 is
          a cycle that completed cleanly end-to-end and has no ``_error``
          record to stamp).

        Must never raise: awaited unshielded in the ``finally``, immediately
        before ``update_run_stage_reports``. An exception here would replace
        whatever exception is already propagating and skip that persistence
        call, so the body swallows ``BaseException`` and each write itself
        runs under ``asyncio.shield`` to survive a second cancellation
        arriving mid-write.
        """
        s1_key = StageId.memory_consolidator.value
        s1_report = run.stage_reports.get(s1_key)

        raised_before_write = current_stage_name == s1_key and s1_report is None
        completed_but_write_failed = (
            isinstance(s1_report, StageReport)
            and run.run_type != RunType.remediation
            and getattr(self.memory, 'recon_ledger', None) is not None
            and _stage1_ledger_write_missing(s1_report)
        )
        if not (raised_before_write or completed_but_write_failed):
            return

        try:
            if completed_but_write_failed and isinstance(s1_report, StageReport):
                # Arm 2 (task 2734): see docstring. Re-attempt with the REAL
                # report — reusing it (not a zeroed synth) records honest
                # llm_calls/tokens/stats; the upsert is idempotent so a
                # re-attempt after a transient failure is safe.
                #
                # Stamped True *before* the call: write_stage1_cycle_summary
                # serializes report.stats into the ledger row's payload_json
                # synchronously, at call time (see write_cycle_summary's
                # docstring) — a post-call mutation can never retroactively
                # reach an already-persisted row, so this is the only way a
                # successful re-attempt's OWN ledger row ends up carrying the
                # marker. Corrected back to False below if the re-attempt did
                # not actually land a row — that correction can't rewrite the
                # (nonexistent, in the failure case) ledger row, but it DOES
                # reach the run's own journal-persisted stage_reports copy
                # (update_run_stage_reports, called right after this method
                # returns), so that copy never falsely claims recovery
                # succeeded (task 2734 amendment).
                s1_report.stats['stage1_cycle_summary_write_recovered_backstop'] = True
                try:
                    ledger_written = await asyncio.shield(
                        write_stage1_cycle_summary(
                            self.memory, project_id, s1_report, run_id,
                        )
                    )
                except BaseException:
                    s1_report.stats['stage1_cycle_summary_write_recovered_backstop'] = False
                    raise
                if not ledger_written:
                    s1_report.stats['stage1_cycle_summary_write_recovered_backstop'] = False
                logger.warning(
                    'reconciliation.stage1_cycle_summary_write_recovered',
                    extra={
                        'run_id': run_id,
                        'project_id': project_id,
                        'ledger_written': ledger_written,
                    },
                )
            else:
                degraded_report = StageReport(
                    stage=StageId.memory_consolidator,
                    # Whole-cycle anchor, not Stage 1's real start (see docstring).
                    started_at=cycle_start_time,
                    completed_at=datetime.now(UTC),
                    items_flagged=[],
                    stats={'stage1_cycle_summary_degraded_backstop': True},
                    # Zeroed, not "no work happened" (see docstring). Dashboards summing
                    # llm_calls/tokens across cycles should filter out
                    # stats['stage1_cycle_summary_degraded_backstop'] rows first, or they
                    # will silently undercount.
                    llm_calls=0,
                    tokens_used=0,
                )
                # Shielded against a second cancellation arriving mid-write (see
                # docstring); the write keeps running to completion in its own Task.
                ledger_written = await asyncio.shield(
                    write_stage1_cycle_summary(
                        self.memory, project_id, degraded_report, run_id,
                    )
                )
                logger.warning(
                    'reconciliation.stage1_cycle_summary_backstop_fired',
                    extra={
                        'run_id': run_id,
                        'project_id': project_id,
                        'ledger_written': ledger_written,
                    },
                )
                # Breadcrumb on the existing _error record (when present) rather than
                # a new top-level stage_reports key — keeps this observable from the
                # same place operators already look for a failed cycle's diagnosis.
                error_record = run.stage_reports.get('_error')
                if isinstance(error_record, dict):
                    error_record['stage1_cycle_summary_backstop_written'] = ledger_written
        except BaseException:
            # BaseException (not Exception): also catches a second cancellation
            # and, deliberately, SystemExit/KeyboardInterrupt — this is a single
            # narrow, bounded best-effort upsert, and letting any of those
            # interrupt the finally risks skipping the update_run_stage_reports
            # call that follows (see docstring).
            logger.warning(
                'reconciliation.stage1_cycle_summary_backstop_failed',
                exc_info=True,
                extra={'run_id': run_id, 'project_id': project_id},
            )

    # ── Remediation support ───────────────────────────────────────────

    async def _get_prior_s3_findings(self, project_id: str) -> list[dict] | None:
        """Extract S3 findings from the last completed run's stage reports."""
        try:
            recent = await self.journal.get_recent_runs(project_id, limit=3)
            for r in recent:
                if r.status != 'completed':
                    continue
                s3_report = r.stage_reports.get('integrity_check')
                if s3_report is None:
                    continue
                if isinstance(s3_report, dict):
                    items = s3_report.get('items_flagged', [])
                else:
                    items = s3_report.items_flagged
                if items:
                    return items
        except Exception as e:
            logger.warning(f'Failed to load prior S3 findings: {e}')
        return None

    async def _finding_persistence_count(
        self,
        project_id: str,
        finding: dict,
        lookback: int = max(5, _INTEGRITY_FINDING_RECURRENCE_THRESHOLD),
    ) -> int:
        """Count how many of the last `lookback` completed runs contain a matching finding.

        Identity is determined by compute_content_fingerprint (same key as _escalate uses
        for dedup), so findings that differ only in non-identity fields do not inflate the
        count.  Counted per-run — duplicates within one run's items_flagged do not count
        twice.

        The default lookback is ``max(5, _INTEGRITY_FINDING_RECURRENCE_THRESHOLD)`` so
        that raising the threshold never silently caps the count below the gate value.

        Returns 0 if HAS_ESCALATION is False.  When the fingerprint call or the
        journal call raises, a WARNING is logged and a sentinel >=
        _INTEGRITY_FINDING_RECURRENCE_THRESHOLD is returned so a degraded count
        fails-toward-escalate rather than silently suppressing the escalation as a
        benign below-threshold count.

        Task 1512 / plans/afk-A7-recon-closure.md.
        """
        if not HAS_ESCALATION:
            return 0
        try:
            target_fp = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                'recon_integrity_issue',
                finding.get('category') or '',
                _derive_affected_ids(finding),
                finding.get('description') or '',
            )
        except Exception as exc:
            logger.warning(
                'reconciliation.persistence_fingerprint_failed',
                extra={
                    'project_id': project_id,
                    'finding_category': finding.get('category', ''),
                    'error': str(exc),
                },
            )
            return max(lookback, _INTEGRITY_FINDING_RECURRENCE_THRESHOLD)
        try:
            recent = await self.journal.get_recent_runs(project_id, limit=lookback)
            count = 0
            any_item_fp_failed = False
            for run in recent:
                if run.status != 'completed':
                    continue
                s3_report = run.stage_reports.get('integrity_check')
                if s3_report is None:
                    continue
                if isinstance(s3_report, dict):
                    items = s3_report.get('items_flagged', []) or []
                else:
                    items = s3_report.items_flagged or []
                # Count this run once if any item matches the target fingerprint
                for item in items:
                    try:
                        fp = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                            'recon_integrity_issue',
                            item.get('category') or '',
                            _derive_affected_ids(item),
                            item.get('description') or '',
                        )
                    except Exception:
                        any_item_fp_failed = True
                        continue
                    if fp == target_fp:
                        count += 1
                        break  # one match per run is enough
            if any_item_fp_failed:
                logger.warning(
                    'reconciliation.persistence_item_fingerprint_failed',
                    extra={
                        'project_id': project_id,
                        'finding_category': finding.get('category', ''),
                    },
                )
                return max(count, _INTEGRITY_FINDING_RECURRENCE_THRESHOLD)
            return count
        except Exception as _e:
            logger.warning(
                'reconciliation.persistence_count_failed',
                extra={
                    'project_id': project_id,
                    'finding_category': finding.get('category', ''),
                    'error': str(_e),
                },
            )
            return max(lookback, _INTEGRITY_FINDING_RECURRENCE_THRESHOLD)

    async def _maybe_escalate_stale_task_count_snapshot(
        self,
        project_id: str,
        run_id: str,
        run: ReconciliationRun,
    ) -> None:
        """Escalate a sustained task_count_snapshot write-cadence gap.

        Reads *run*'s Stage-2 freshness stat
        (``task_count_snapshot_mem0_written``, already computed by
        ``TaskKnowledgeSync.run()``'s post-flight check); only a CONFIRMED
        current miss (``False`` — not a fresh write and not an
        inconclusive/unknown check) is eligible to escalate.  Journal rows
        persisted before the task-3045 rename carry the old
        ``task_count_snapshot_written`` spelling and are still honored, via
        ``extract_snapshot_written``'s legacy-key fallback — without it the
        streak below would stop dead at the first pre-rename row.  The prior
        consecutive-miss streak is recomputed each call from
        ``journal.get_recent_runs`` — mirroring ``_finding_persistence_count``'s
        journal-recompute pattern — rather than a stored counter, so it
        naturally resets on any successful write and survives a harness
        restart with no new schema.

        Prior runs are filtered to full-cycle COMPLETED runs only (mirrors the
        completed filter in ``_get_prior_s3_findings``): a remediation/targeted
        run neither counts toward nor resets the streak (the snapshot is a
        full-cycle Stage-2 final action, so those runs' own stat reflects their
        own window, not a genuine gap), and a failed full run is skipped so the
        streak bridges across it to the next completed one.

        Escalates category 'recon_stale_task_count_snapshot' (info severity,
        deduped via ``_RECON_DEDUP_CONFIG.infra_dedupe_categories`` on the
        stable per-project ``build_stale_snapshot_finding`` identity) once the
        streak reaches ``TASK_COUNT_SNAPSHOT_MISS_THRESHOLD``.  Skipped
        entirely for projects where ``is_snapshot_write_blocked`` — a missing
        snapshot there is correct-by-design, not a gap. That check runs
        FIRST, before the current-stat read or the ``journal.get_recent_runs``
        call, so a blocked project incurs no per-cycle journal query or
        streak recompute (amendment round: reviewer finding, efficiency).

        Fails open — never raises — mirroring ``_get_prior_s3_findings`` /
        ``_check_graphiti_queue_health``, so a journal hiccup never aborts a
        reconciliation cycle.  Wired into ``run_full_cycle`` only, after the
        Stage-2 report is persisted — NOT evaluated during remediation passes.

        Task 2278.
        """
        if not HAS_ESCALATION:
            return
        try:
            if is_snapshot_write_blocked(project_id):
                return
            current = extract_snapshot_written(run.stage_reports.get('task_knowledge_sync'))
            if current is not False:
                return
            recent = await self.journal.get_recent_runs(
                project_id, limit=max(20, TASK_COUNT_SNAPSHOT_MISS_THRESHOLD * 4),
            )
            prior_runs = [
                r for r in recent
                if getattr(r, 'id', None) != run_id
                and str(getattr(r, 'run_type', '')) == RunType.full
                and str(getattr(r, 'status', '')) == 'completed'
            ]
            # Defensive re-sort: get_recent_runs already orders by started_at DESC,
            # but the filtered subset's ordering is re-asserted here rather than
            # assumed, since compute_snapshot_miss_streak depends on most-recent-first.
            prior_runs.sort(
                key=lambda r: getattr(r, 'started_at', None) or datetime.min.replace(tzinfo=UTC),
                reverse=True,
            )
            prior_flags = [
                extract_snapshot_written(r.stage_reports.get('task_knowledge_sync'))
                for r in prior_runs
            ]
            result = evaluate_snapshot_cadence(
                current, prior_flags, blocked=is_snapshot_write_blocked(project_id),
            )
            if result['escalate']:
                self._escalate(
                    TASK_COUNT_SNAPSHOT_ESCALATION_CATEGORY,
                    run_id,
                    summary=(
                        f'task_count_snapshot stale for {result["streak"]} consecutive '
                        f'full cycles (project {project_id})'
                    ),
                    detail=(
                        'Reconciliation Stage 2 has not confirmed a fresh '
                        'task_count_snapshot Mem0 write within the run window for '
                        f'{result["streak"]} consecutive completed full reconciliation '
                        f'cycles for project {project_id!r}.'
                    ),
                    finding=build_stale_snapshot_finding(project_id),
                )
        except Exception as e:
            logger.warning(
                'reconciliation.stale_task_count_snapshot_check_failed',
                extra={'project_id': project_id, 'run_id': run_id, 'error': str(e)},
            )

    def _log_non_actionable_finding(
        self,
        project_id: str,
        run_id: str,
        finding: dict,
    ) -> None:
        """Emit a structured INFO log for a non-actionable Stage-3 integrity finding.

        Non-actionable findings are suppressed from the escalation queue (task 1512 /
        plans/afk-A7-recon-closure.md): the only possible human action — "accept as
        known" — is achieved by *not* filing.  This helper keeps the finding observable
        in the recon log/journal without touching the queue.

        Called from both _maybe_remediate (parent pass) and _run_remediation_pass (after
        the second-pass actionable partition) so both sites stay in sync as fields evolve.
        """
        logger.info(
            'reconciliation.non_actionable_integrity_finding',
            extra={
                'project_id': project_id,
                'run_id': run_id,
                'finding_category': finding.get('category', ''),
                'affected_ids': _derive_affected_ids(finding),
                'description': finding.get('description', ''),
                'severity': finding.get('severity', ''),
            },
        )

    def _finding_has_open_escalation(
        self,
        finding: dict,
        pending_fps: set[str] | None = None,
    ) -> bool:
        """Return True iff an OPEN pending escalation already covers this finding.

        Uses the same compute_content_fingerprint key as _escalate and
        _finding_persistence_count (category='recon_integrity_issue', finding
        category, _derive_affected_ids, description) to guarantee suppression
        is consistent with what _escalate would fold via submit_or_dedupe.

        pending_fps: pre-fetched set of dedupe_fingerprints for
        'recon_integrity_issue' pending escalations.  When supplied (built once
        by _maybe_remediate), the check is an O(1) set membership test — no
        extra disk scan.  Pass None to fall back to a full get_pending() scan
        per-finding (direct / unit-test use).

        Fail-open: any exception (or HAS_ESCALATION False / queue None) returns
        False so that a transient queue-read glitch costs at most one extra
        remediation pass rather than silently suppressing a needed pass.

        Task 1570 / FIX A.
        """
        if not HAS_ESCALATION or self._escalation_queue is None:
            return False
        try:
            target_fp = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                'recon_integrity_issue',
                finding.get('category') or '',
                _derive_affected_ids(finding),
                finding.get('description') or '',
            )
            if pending_fps is not None:
                return target_fp in pending_fps
            # Fallback: per-finding scan (no pre-fetched set supplied).
            for e in self._escalation_queue.get_pending():
                if (
                    e.category == 'recon_integrity_issue'
                    and e.dedupe_fingerprint == target_fp
                ):
                    return True
            return False
        except Exception as _err:
            logger.warning(
                'reconciliation.open_escalation_check_failed',
                extra={
                    'finding_category': finding.get('category', ''),
                    'error': str(_err),
                },
            )
            return False  # fail-open: proceed with remediation

    def _finding_recently_resolved(
        self,
        category: str,
        fingerprint: str | None,
        *,
        now=None,
        resolved_fps: frozenset[str] | None = None,
    ) -> bool:
        """Return True iff a recently-resolved escalation matches category + fingerprint.

        Mirrors _finding_has_open_escalation but scans resolved/dismissed escalations
        in the full queue root + archive via iter_all_escalation_paths.  Used by
        _escalate() to suppress re-firing of a finding whose matching escalation was
        resolved within _RESOLVED_RECURRENCE_WINDOW_SECONDS (task 1669).

        resolved_fps: pre-fetched frozenset of dedupe_fingerprints for resolved/dismissed
        escalations within the recurrence window (filtered to recon categories by the
        caller).  When supplied (built once by _run_remediation_pass), the check is an
        O(1) set membership test — no archive scan.  Pass None to fall back to the full
        iter_all_escalation_paths scan per-call (direct / unit-test use).

        Gate: only covers categories in _RECON_DEDUP_CONFIG.infra_dedupe_categories,
        matching the category gate used by submit_or_dedupe.

        Fail-open: any exception (or HAS_ESCALATION False / queue None / no fingerprint)
        returns False so a transient read glitch costs at most one extra escalation
        rather than silently suppressing a needed one.  Same philosophy as
        _finding_has_open_escalation.
        """
        if not HAS_ESCALATION or self._escalation_queue is None or not fingerprint:
            return False
        if _RECON_DEDUP_CONFIG is None or category not in _RECON_DEDUP_CONFIG.infra_dedupe_categories:
            return False
        if resolved_fps is not None:
            # Pre-fetched set: O(1) membership test — no archive scan needed.
            return fingerprint in resolved_fps
        try:
            effective_now = now if now is not None else datetime.now(UTC)
            window = timedelta(seconds=_RESOLVED_RECURRENCE_WINDOW_SECONDS)
            for path in iter_all_escalation_paths(  # type: ignore[possibly-undefined]
                self._escalation_queue.queue_dir
            ):
                try:
                    esc = Escalation.from_json(path.read_text())  # type: ignore[possibly-undefined]
                except Exception:
                    continue
                if not (
                    esc.status in ('resolved', 'dismissed')
                    and esc.category == category
                    and esc.dedupe_fingerprint == fingerprint
                ):
                    continue
                if not esc.resolved_at:
                    continue
                try:
                    resolved = datetime.fromisoformat(esc.resolved_at)
                except (ValueError, TypeError):
                    continue
                if resolved.tzinfo is None:
                    resolved = resolved.replace(tzinfo=UTC)
                if effective_now - resolved <= window:
                    return True
            return False
        except Exception as e:
            logger.warning(
                'reconciliation.recently_resolved_check_failed',
                extra={'category': category, 'error': str(e)},
            )
            return False

    async def _maybe_remediate(
        self,
        project_id: str,
        parent_run_id: str,
        parent_run: ReconciliationRun,
        tier: TierConfig,
        *,
        scope: ProjectScope,
        filtered_task_tree: FilteredTaskTree | None = None,
    ) -> None:
        """Extract Stage 3 findings from the parent run and trigger remediation if needed."""
        try:
            s3_report = parent_run.stage_reports.get('integrity_check')
            if s3_report is None:
                return

            if isinstance(s3_report, dict):
                all_findings = s3_report.get('items_flagged', [])
            else:
                all_findings = s3_report.items_flagged

            if not all_findings:
                return

            # Partition into actionable vs escalation
            actionable = [f for f in all_findings if f.get('actionable', False)]
            non_actionable = [f for f in all_findings if not f.get('actionable', False)]

            # Task 1512 / plans/afk-A7-recon-closure.md:
            # Non-actionable findings are NOT escalated.  Per the Stage-3 contract
            # they are already (a) persisted in stage_reports.integrity_check.items_flagged
            # and (b) forward-fed into the next cycle's S1/S2 via _get_prior_s3_findings.
            # The only possible human action — "accept as known" — is achieved by *not*
            # filing, so escalating them is a category error.  Instead, emit a structured
            # log record so the finding stays observable in the recon log/journal.
            for finding in non_actionable:
                self._log_non_actionable_finding(project_id, parent_run_id, finding)

            if not actionable:
                return

            # Task 1970: drop any actionable finding that cites nothing (no
            # legacy affected_ids and no typed citation) — a synthetic/
            # placeholder finding that cannot be investigated or remediated
            # (e.g. Stage 3 filed add_finding but never followed up with a
            # cite_* call).  This is the last line of defense before findings
            # become a production remediation batch: _run_remediation_pass is
            # reached only through this method, so this guard fully closes
            # the leak.  Fail-open: _finding_has_reference only drops a
            # finding when _derive_affected_ids is clearly empty; anything
            # ambiguous (legacy affected_ids or any typed citation) passes.
            referenceable: list[dict] = []
            dropped_placeholders: list[dict] = []
            for finding in actionable:
                if _finding_has_reference(finding):
                    referenceable.append(finding)
                else:
                    dropped_placeholders.append(finding)

            for finding in dropped_placeholders:
                logger.warning(
                    'reconciliation.remediation_dropped_placeholder_finding',
                    extra={
                        'project_id': project_id,
                        'parent_run_id': parent_run_id,
                        'finding_category': finding.get('category', ''),
                        'description': finding.get('description', ''),
                    },
                )
                # Task 1970 amendment: coarse aggregate alarm for a runaway
                # Stage 3 that stops citing anything.  Each individual drop
                # stays logged-only (never escalated on its own — see the
                # module comment above _PLACEHOLDER_DROP_STORM_THRESHOLD);
                # this counter fires ONE 'recon_remediation_placeholder_storm'
                # escalation when drops recur often enough within the
                # rolling window.  Mirrors the dead_owner_shielded storm
                # wiring in _recover_stale_runs (~line 1122).
                storm = self._record_placeholder_finding_drop(project_id)
                if storm is not None:
                    window_min = storm['window_seconds'] / 60
                    proj_label = ', '.join(storm['projects']) or project_id
                    storm_summary = (
                        f"referenceless placeholder-finding drop storm: {storm['count']} in "
                        f"{window_min:.0f} min (projects: {proj_label}) — "
                        f'Stage 3 may have stopped citing findings (add_finding '
                        f'without a follow-up cite_* call)'
                    )
                    storm_detail = f'project={project_id} parent_run={parent_run_id}'
                    self._escalate(
                        'recon_remediation_placeholder_storm',
                        parent_run_id,
                        storm_summary,
                        storm_detail,
                        finding=_PLACEHOLDER_DROP_STORM_FINDING,
                    )

            if not referenceable:
                return

            actionable = referenceable

            # Task 1570 / FIX A: suppress remediation for any actionable finding
            # that already has an OPEN pending recon_integrity_issue escalation.
            # Performance: call get_pending() ONCE and build a fingerprint set
            # (single directory scan per cycle) rather than O(N findings) scans.
            # Fail-open: on any queue-read error emit a warning, treat pending_fps
            # as empty (no suppressions), and proceed with full remediation.
            pending_fps: set[str] = set()
            if HAS_ESCALATION and self._escalation_queue is not None:
                try:
                    pending_fps = {
                        e.dedupe_fingerprint
                        for e in self._escalation_queue.get_pending()
                        if e.category == 'recon_integrity_issue'
                        and e.dedupe_fingerprint is not None
                    }
                except Exception as _fps_err:
                    logger.warning(
                        'reconciliation.open_escalation_check_failed',
                        extra={
                            'finding_category': '',
                            'error': str(_fps_err),
                        },
                    )
                    # pending_fps stays empty → no suppressions → remediation
                    # proceeds normally (fail-open).

            suppressed: list[dict] = []
            to_remediate: list[dict] = []
            for finding in actionable:
                if self._finding_has_open_escalation(finding, pending_fps=pending_fps):
                    suppressed.append(finding)
                else:
                    to_remediate.append(finding)

            for finding in suppressed:
                logger.info(
                    'reconciliation.remediation_suppressed_open_escalation',
                    extra={
                        'project_id': project_id,
                        'parent_run_id': parent_run_id,
                        'finding_category': finding.get('category', ''),
                        'affected_ids': _derive_affected_ids(finding),
                        'description': finding.get('description', ''),
                    },
                )

            if not to_remediate:
                return

            logger.info(
                f'Remediation: {len(to_remediate)} actionable findings from run {parent_run_id}, '
                f'triggering second pass'
                + (
                    f' ({len(suppressed)} suppressed — open escalation exists)'
                    if suppressed else ''
                )
            )
            await self._run_remediation_pass(
                project_id, parent_run_id, to_remediate, tier,
                scope=scope,
                filtered_task_tree=filtered_task_tree,
            )
        except Exception as e:
            logger.error(f'Remediation check failed for run {parent_run_id}: {e}')
            self._escalate(
                'recon_integrity_issue',
                parent_run_id,
                f'Remediation orchestration failed: {e}',
            )

    async def _run_remediation_pass(
        self,
        project_id: str,
        parent_run_id: str,
        findings: list[dict],
        tier: TierConfig,
        *,
        scope: ProjectScope,
        filtered_task_tree: FilteredTaskTree | None = None,
    ) -> None:
        """Run a focused S1→S2→S3 pass to remediate actionable findings.

        scope is threaded from the parent caller: run_full_cycle resolves it
        once at entry via _known_project_scope_for, before any side-effects, and
        threads it through _maybe_remediate so remediation always uses the pre-cycle
        snapshot, immune to any mid-cycle registry mutations (task 1163).

        If filtered_task_tree is provided it is used directly; otherwise a fresh
        tree is fetched via _fetch_filtered_task_tree.  Callers that already hold
        a fetched tree (e.g. run_full_cycle) should pass it through to avoid a
        redundant taskmaster round-trip.
        """
        project_root = scope.project_root
        # Defense-in-depth assert deliberately omitted.  A registry-bound check such
        # as `assert project_root in self._known_projects.values()` would fail during
        # the mid-cycle mutation window that task 1163 was specifically designed to
        # tolerate: both test_remediation_pass_uses_threaded_project_root_over_registry
        # and test_remediation_uses_threaded_project_root_not_mutated_registry pass a
        # project_root that intentionally differs from the current registry value.

        run_id = str(uuid4())
        run = ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type=RunType.remediation,
            trigger_reason=f'integrity_findings:{len(findings)}',
            started_at=datetime.now(UTC),
            events_processed=0,
            status=RunStatus.running,
            triggered_by=parent_run_id,
            instance_id=self.buffer.instance_id,
        )
        await self.journal.start_run(run)

        logger.info(
            'reconciliation.remediation_started',
            extra={
                'run_id': run_id,
                'parent_run_id': parent_run_id,
                'project_id': project_id,
                'findings_count': len(findings),
            },
        )

        # Use caller-supplied tree if available; otherwise fetch (ref: task 455, task 478)
        remediation_tree = (
            filtered_task_tree
            if filtered_task_tree is not None
            else await self._fetch_filtered_task_tree(project_root)
        )

        # Task 2031/2067: {str(task_id): status} and {str(task_id): task_kind} maps
        # derived from remediation_tree in a single pass, used by the live-workflow
        # gate below so never-dispatched cited tasks (deferred/done/cancelled) drop
        # the project-wide orchestrator_live signal instead of being suppressed by
        # it, and so BLOCKED cited tasks that are deterministic (never acquire a
        # worktree/branch of their own — routed to DeterministicRunner) do too —
        # which status_by_id alone cannot express since 'blocked' is deliberately
        # not in ORCH_LIVE_INELIGIBLE_STATUSES (a normal blocked task may
        # legitimately auto-unblock mid-pipeline). remediation_tree is always a
        # valid FilteredTaskTree (degrades to empty on fetch failure), so this is
        # safe. Built together (rather than as two separate comprehensions) so the
        # two maps are guaranteed key-identical from one iteration of the source
        # lists.
        # Coverage caveat: active_tasks is uncapped (deferred/blocked — the cited
        # cases — always resolve), but done_tasks/cancelled_tasks are capped at
        # MAX_DONE_TASKS_RETAINED=30 / MAX_CANCELLED_TASKS_RETAINED=15
        # (task_filter.py). A cited done/cancelled task outside those caps, or one
        # with an untracked status, is simply absent here and status_by_id.get(tid)
        # / task_kind_by_id.get(tid) fall back to None below — the pre-2031
        # status-blind behavior for that one id, not a new failure mode.
        status_by_id: dict[str, str | None] = {}
        task_kind_by_id: dict[str, str | None] = {}
        for t in (
            list(remediation_tree.active_tasks)
            + list(remediation_tree.done_tasks)
            + list(remediation_tree.cancelled_tasks)
        ):
            if not isinstance(t, dict) or t.get('id') is None:
                continue
            tid = str(t.get('id'))
            status_by_id[tid] = t.get('status')
            _metadata = t.get('metadata')
            task_kind_by_id[tid] = _metadata.get('task_kind') if isinstance(_metadata, dict) else None

        current_stage_name: str | None = None

        # Task 2417: cheap, deterministic freshness pre-check — BEFORE any
        # stage is built or run. Filters cross-project scope-correction
        # findings whose subject task is unchanged since the last
        # consolidated snapshot out of `findings`, the single choke point
        # shared by both Stage 1 (remediation_findings, wired below via
        # _configure_consolidator) and Stage 2 (remediation_mode=True — no
        # findings list of its own) remediation re-derivation. Best-effort:
        # any failure here (unresolvable project, get_task/Mem0 errors, ...)
        # falls back to the original, unfiltered `findings` — see the
        # fused_memory.reconciliation.scope_freshness module docstring.
        # `freshness` is pre-initialized to None so the except branch below
        # (the pre-check raising before ever assigning it) leaves an
        # unambiguous "pre-check did not run" sentinel for the short-circuit
        # guard just below to key off of.
        # `max_consecutive_skips` is wired to the SAME
        # _INTEGRITY_FINDING_RECURRENCE_THRESHOLD used by the persistence-gated
        # escalation loop below (amendment: reviewer finding
        # robustness_silent_degradation) — a (task_ref, flag_key) pair can be
        # skipped by the pre-check at most threshold-1 cycles in a row before
        # it is forced back through a real Stage 1-3 pass. The cap ALONE is
        # not sufficient for the loud-failure guarantee, though: the
        # persistence-gated escalation loop only counts a run toward a
        # finding's recurrence if that run's stage_reports carries an
        # integrity_check entry, and a short-circuited run built none.  The
        # short-circuit block below also stamps `freshness.skipped` into a
        # synthetic integrity_check report before completing the run
        # (amendment — reviewer finding behavior_change), so BOTH the cap's
        # periodic forced re-investigation AND every intervening
        # short-circuited skip contribute to the window — only together do
        # they guarantee a genuinely stranded cross-project thread escalates
        # within a bounded number of cycles instead of being silently
        # suppressed forever.
        freshness: ScopeFreshnessResult | None = None
        try:
            if self.taskmaster is None:
                raise RuntimeError('taskmaster is not configured')
            freshness = await precheck_scope_correction_freshness(
                memory_service=self.memory,
                taskmaster=self.taskmaster,
                project_id=project_id,
                resolve_project_root=self._resolve_known_root,
                run_id=run_id,
                findings=findings,
                max_consecutive_skips=_INTEGRITY_FINDING_RECURRENCE_THRESHOLD,
            )
            findings = freshness.to_reinvestigate
            skipped_task_refs = []
            for _skipped in freshness.skipped:
                _sig = compute_scope_signature(_skipped, project_id)
                if _sig is not None:
                    skipped_task_refs.append(_sig[0])
            logger.info(
                'reconciliation.scope_freshness_precheck',
                extra={
                    'run_id': run_id,
                    'project_id': project_id,
                    'skipped_task_refs': skipped_task_refs,
                    **freshness.stats,
                },
            )
        except Exception as exc:
            logger.warning(
                'reconciliation.scope_freshness_precheck_wiring_failed',
                extra={'run_id': run_id, 'project_id': project_id, 'error': str(exc)},
            )

        # Short-circuit: every finding was confirmed fresh (unchanged) by the
        # pre-check above — skip building/running any stage entirely (no LLM
        # subprocess launches) and journal-complete the run as-is. Guarded on
        # `freshness is not None` so a pre-check that raised (and therefore
        # fell back to the original, unfiltered `findings` above) never
        # short-circuits — only a POSITIVE freshness confirmation may skip
        # remediation, never an error/uncertainty path.
        if freshness is not None and not findings and freshness.skipped:
            logger.info(
                'reconciliation.remediation_skipped_all_fresh',
                extra={
                    'run_id': run_id,
                    'project_id': project_id,
                    'parent_run_id': parent_run_id,
                    **freshness.stats,
                },
            )
            # Stamp the skipped findings into a synthetic integrity_check
            # stage report BEFORE completing the run, so this short-circuited
            # run still counts toward _finding_persistence_count's lookback
            # window exactly as a real Stage 3 re-flag would have (task 2417
            # amendment — reviewer finding behavior_change).  Without this,
            # a short-circuited run occupied a slot in the persistence
            # window while contributing 0, and the loud-failure guarantee
            # described above depended on BOTH the consecutive-skip cap's
            # periodic forced re-investigation AND this stamp — the cap
            # alone forces a real pass only every threshold-th cycle, which
            # is not by itself enough to saturate the persistence window.
            # No stage is built or run here — this uses the plain-dict
            # report shape the journal/tests already accept elsewhere
            # ({'integrity_check': {'items_flagged': [...]}}).
            run.stage_reports['integrity_check'] = {'items_flagged': list(freshness.skipped)}
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)
            run.completed_at = datetime.now(UTC)
            run.status = RunStatus.completed
            await self.journal.complete_run(run_id, 'completed')
            return

        stages = self._make_stages(scope)
        try:
            # Configure stages for remediation mode
            stage1 = stages[0]
            stage2 = stages[1]
            assert isinstance(stage1, MemoryConsolidator)
            assert isinstance(stage2, TaskKnowledgeSync)
            self._configure_consolidator(
                stage1, tier,
                remediation_findings=findings,
                filtered_task_tree=remediation_tree,
            )
            self._configure_task_sync(stage2, filtered_task_tree=remediation_tree, remediation_mode=True)

            watermark = await self.journal.get_watermark(project_id)
            reports = []
            for stage in stages:
                current_stage_name = stage.stage_id.value

                report = await stage.run(
                    [], watermark, reports, run_id, model=tier.model,
                )
                reports.append(report)
                run.stage_reports[stage.stage_id.value] = report

            # Do NOT update watermark — remediation processed no new episodes/events

            run.completed_at = datetime.now(UTC)
            run.status = RunStatus.completed
            await self.journal.complete_run(run_id, 'completed')

            # Cross-check self-reported stats against write-journal ops before
            # the judge reads them (same as run_full_cycle).
            await verify_and_rewrite_stats(
                run_id, run.stage_reports, self.journal.write_journal,
            )

            # Persist stage reports before judge (same fix as run_full_cycle)
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)

            # Judge review for remediation run. Same _spawn_judge seam as
            # run_full_cycle: durable judge_pending marker + tracked task (task 2708).
            if self.judge:
                await self._spawn_judge(run_id, project_id)

            # After second-pass S3: gate escalation on persistence (task 1512 /
            # plans/afk-A7-recon-closure.md).  Never a third pass.
            # A finding is only escalated when it has appeared in at least
            # _INTEGRITY_FINDING_RECURRENCE_THRESHOLD completed runs, which signals
            # that remediation cannot fix it on its own.  Below the threshold we
            # suppress the escalation and emit a structured log so the finding
            # stays observable without polluting the queue.
            # The except handlers further below (AllAccountsCappedException and
            # bare Exception) are untouched — they fire on stage *exceptions*, not
            # on Stage-3 findings, and are the genuine "needs human" signals.
            #
            # task 1512 review_feedback (design_inconsistency):
            # Partition remaining findings by actionability — identical to
            # _maybe_remediate's partition at lines 1260-1261.  Non-actionable
            # findings are logged exactly as in the parent pass and must NEVER reach
            # the persistence-gated escalation branch regardless of recurrence count.
            s3_report = run.stage_reports.get('integrity_check')
            if s3_report is not None:
                if isinstance(s3_report, dict):
                    all_remaining = s3_report.get('items_flagged', [])
                else:
                    all_remaining = s3_report.items_flagged
                actionable_remaining = [f for f in all_remaining if f.get('actionable', False)]
                non_actionable_remaining = [f for f in all_remaining if not f.get('actionable', False)]
                # Non-actionable findings are logged but never escalated — same
                # contract as the parent pass in _maybe_remediate.
                for finding in non_actionable_remaining:
                    self._log_non_actionable_finding(project_id, run_id, finding)

                # Task 1669: pre-build the in-window resolved fingerprints set ONCE
                # (single archive scan per remediation pass) so the per-finding
                # _escalate call is an O(1) membership test rather than an O(archive)
                # full scan.  Mirrors the pending_fps pattern in _maybe_remediate.
                # Fail-open: on any scan error, leave resolved_fps empty (no
                # suppressions) and proceed with full escalation — same philosophy
                # as the pending_fps build in _maybe_remediate.
                resolved_fps: frozenset[str] = frozenset()
                if (
                    HAS_ESCALATION
                    and self._escalation_queue is not None
                    and _RECON_DEDUP_CONFIG is not None
                ):
                    try:
                        _now = datetime.now(UTC)
                        _window = timedelta(seconds=_RESOLVED_RECURRENCE_WINDOW_SECONDS)
                        _rfps: set[str] = set()
                        for _path in iter_all_escalation_paths(  # type: ignore[possibly-undefined]
                            self._escalation_queue.queue_dir
                        ):
                            try:
                                _esc = Escalation.from_json(_path.read_text())  # type: ignore[possibly-undefined]
                            except Exception:
                                continue
                            if not (
                                _esc.status in ('resolved', 'dismissed')
                                and _esc.category in _RECON_DEDUP_CONFIG.infra_dedupe_categories
                                and _esc.dedupe_fingerprint
                                and _esc.resolved_at
                            ):
                                continue
                            try:
                                _resolved = datetime.fromisoformat(_esc.resolved_at)
                            except (ValueError, TypeError):
                                continue
                            if _resolved.tzinfo is None:
                                _resolved = _resolved.replace(tzinfo=UTC)
                            if _now - _resolved <= _window:
                                _rfps.add(_esc.dedupe_fingerprint)
                        resolved_fps = frozenset(_rfps)
                    except Exception as _rfps_err:
                        logger.warning(
                            'reconciliation.recently_resolved_check_failed',
                            extra={'category': '', 'error': str(_rfps_err)},
                        )
                        # resolved_fps stays empty → no suppressions → fail-open

                for finding in actionable_remaining:
                    persistence = await self._finding_persistence_count(project_id, finding)
                    if persistence >= _INTEGRITY_FINDING_RECURRENCE_THRESHOLD:
                        # Live-workflow gate (task 1655): if any cited task has an active
                        # workflow (registered worktree, recent branch commits, or live
                        # orchestrator), suppress this cycle's escalation with an INFO log.
                        # The finding keeps recurring and will escalate after the workflow
                        # finishes — so genuine stranded cases (all signals False) still
                        # escalate.  Guard detector errors as not-live (fail toward escalating
                        # rather than toward silencing a genuine stranded-work escalation).
                        # Task 2031: cited tasks in a never-dispatched status
                        # (deferred/done/cancelled, via status_by_id above) drop the
                        # project-wide orchestrator_live signal, so a deferred task stuck
                        # behind an unrelated live orchestrator still escalates.
                        # Task 2067: extends this to a BLOCKED cited task that is
                        # deterministic (via task_kind_by_id above) — it never
                        # acquires a worktree/branch of its own, so the bare
                        # orchestrator lock is not task-specific evidence for it
                        # either, and a stranded deterministic deploy still escalates.
                        # Task 2409: extends this further to a BLOCKED cited task
                        # that is normal (task_kind absent or 'normal') with no
                        # registered worktree and no recent commit — the bare
                        # project-wide orchestrator lock is dropped for it too, so
                        # a stranded normal task (the tasks 2335/2196 re-deferral
                        # loop) still escalates instead of being suppressed
                        # indefinitely. A blocked normal task WITH genuine
                        # per-task evidence is unaffected and still suppresses.
                        affected_ids = _derive_affected_ids(finding)
                        # For liveness, iterate only cited task ids.
                        # _derive_affected_ids mixes in entity canonical_names,
                        # edge_uuids, and memory_ids; passing those to
                        # is_workflow_live_for_task would build nonsensical
                        # branch names (e.g. 'task/<canonical_name>') and waste
                        # git subprocess calls that can never match.
                        cited_task_ids = [
                            str(c['task_id'])
                            for c in finding.get('cited_tasks') or []
                            if isinstance(c, dict) and 'task_id' in c
                        ]
                        any_live = False
                        for tid in cited_task_ids:
                            try:
                                if is_workflow_live_for_task(
                                    tid, project_root,
                                    status=status_by_id.get(tid),
                                    task_kind=task_kind_by_id.get(tid),
                                ):
                                    any_live = True
                                    break
                            except Exception as _det_exc:
                                logger.debug(
                                    'live_workflow_detector error for task %s; treating as not-live: %s',
                                    tid, _det_exc,
                                )
                        if any_live:
                            logger.info(
                                'reconciliation.integrity_escalation_suppressed_live_workflow',
                                extra={
                                    'project_id': project_id,
                                    'run_id': run_id,
                                    'affected_ids': affected_ids,
                                    'description': finding.get('description', ''),
                                    'finding_category': finding.get('category', ''),
                                },
                            )
                        else:
                            self._escalate(
                                'recon_integrity_issue',
                                run_id,
                                f'Persistently unresolved after remediation '
                                f'({persistence} cycles): {finding.get("description", "?")}',
                                detail=json.dumps(
                                    {**finding, 'persistence': persistence},
                                    default=str,
                                ),
                                finding=finding,
                                resolved_fps=resolved_fps,
                            )
                    else:
                        logger.info(
                            'reconciliation.unresolved_after_remediation_suppressed',
                            extra={
                                'project_id': project_id,
                                'run_id': run_id,
                                'parent_run_id': parent_run_id,
                                'finding_category': finding.get('category', ''),
                                'affected_ids': _derive_affected_ids(finding),
                                'description': finding.get('description', ''),
                                'persistence': persistence,
                                'threshold': _INTEGRITY_FINDING_RECURRENCE_THRESHOLD,
                            },
                        )

            logger.info(
                'reconciliation.remediation_completed',
                extra={'run_id': run_id, 'parent_run_id': parent_run_id},
            )

        except AllAccountsCappedException as e:
            run.status = RunStatus.failed
            run.stage_reports['_error'] = {
                'error_type': 'AllAccountsCappedException',
                'error_message': str(e),
                'failed_stage': current_stage_name,
                'deferred': True,
            }
            await self.journal.complete_run(run_id, 'failed')
            # Do NOT re-raise — parent run already completed
            # Do NOT restore events — there are none (remediation has no drained events)
            logger.warning(
                f'Remediation deferred: all accounts capped during stage '
                f'{current_stage_name} ({e.retries} retries in {e.elapsed_secs:.1f}s)'
            )
        except Exception as e:
            run.status = RunStatus.failed
            run.stage_reports['_error'] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'failed_stage': current_stage_name,
            }
            await self.journal.complete_run(run_id, 'failed')
            # Do NOT re-raise — parent run already completed
            # Do NOT restore events — there are none
            logger.error(f'Remediation pass failed: {e}')
            self._escalate(
                'recon_integrity_issue',
                run_id,
                f'Remediation pass failed at {current_stage_name}: {e}',
            )

        finally:
            await self._ensure_stage1_cycle_summary(
                run, run_id, project_id, current_stage_name, run.started_at,
            )
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)
            # Task 2744: GC this remediation run's per-run recon CLI config dir on
            # every exit path. Defensive — never mask the run's terminal outcome.
            try:
                gc_run_config_dir(self.journal.data_dir, run_id)
            except Exception as gc_err:  # noqa: BLE001
                logger.warning(
                    'gc_run_config_dir failed for remediation run %s: %r',
                    run_id, gc_err,
                )


# ── Backlog iteration ──────────────────────────────────────────────────


class BacklogIterator:
    """Processes large backlogs in token-budgeted chunks, oldest-first."""

    def __init__(
        self,
        config,
        journal: ReconciliationJournal,
        buffer: EventBuffer,
        harness: ReconciliationHarness,
        time_provider: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.journal = journal
        self.buffer = buffer
        self.harness = harness
        self.time_provider = time_provider

    async def should_iterate(self, project_id: str) -> bool:
        """Buffer count > 150% of trigger threshold."""
        stats = await self.buffer.get_buffer_stats(project_id)
        count = stats.get('size', 0)
        threshold = self.config.buffer_size_threshold * self.config.opus_threshold_ratio
        return count > threshold

    async def run(self, project_id: str) -> None:
        """Process backlog in token-budgeted chunks, oldest-first.

        Uses peek → assemble → drain: peeks at buffered events, builds a
        token-budgeted payload via ContextAssembler, then drains only the
        events that fit.  Stops when no events remain before the cutoff.
        """
        from fused_memory.reconciliation.context_assembler import ContextAssembler

        opus_tier = TierConfig(
            model=self.config.opus_model,
            episode_limit=self.config.opus_episode_limit,
            memory_limit=self.config.opus_memory_limit,
        )

        # Snapshot: only process events that existed when we started.
        cutoff = datetime.now(UTC)

        # task 1143: hard-bind project_root from registry — event payloads are informational only.
        scope = self.harness._known_project_scope_for(project_id)

        assembler = ContextAssembler(
            memory_service=self.harness.memory,
            taskmaster=self.harness.taskmaster,
            config=self.config,
            project_root=scope.project_root,
        )

        watermark = await self.journal.get_watermark(project_id)

        chunk_num = 0
        start = self.time_provider()
        yielded = False
        while True:
            # Cumulative between-chunks budget (task 2040): once at least one
            # chunk has completed and the wall-clock budget is exhausted,
            # stop launching new chunks and leave the remainder buffered.
            # The per-project lock releases via _project_loop's finally, so
            # the next cycle / trigger_reconciliation keeps draining instead
            # of one iteration holding the lock forever.
            if (
                chunk_num > 0
                and self.time_provider() - start >= self.config.backlog_iteration_budget_seconds
            ):
                stats = await self.buffer.get_buffer_stats(project_id)
                logger.info(
                    'reconciliation.backlog_iteration_yielded',
                    extra={
                        'project_id': project_id,
                        'chunks_processed': chunk_num,
                        'events_remaining': stats.get('size', 0),
                    },
                )
                yielded = True
                break

            # Peek at up to 1000 events (far more than a single budget can hold)
            peeked = await self.buffer.peek_buffered(
                project_id, limit=1000, before=cutoff,
            )
            if not peeked:
                break

            # Assemble token-budgeted payload
            assembled = await assembler.assemble(peeked, watermark, project_id)
            if not assembled.events:
                break

            # Drain exactly the events that fit the budget
            event_ids = [e.id for e in assembled.events]
            await self.buffer.drain_by_ids(project_id, event_ids)

            chunk_num += 1
            chunk_id = str(uuid4())
            await self.journal.record_chunk_boundary(
                project_id, chunk_id, len(assembled.events),
            )

            logger.info(
                f'Backlog chunk {chunk_num}: processing {len(assembled.events)} events '
                f'({assembled.total_tokens} tokens, {len(assembled.context_items)} context items, '
                f'{assembled.events_remaining} remaining) for {project_id}'
            )

            try:
                await asyncio.wait_for(
                    self.harness.run_full_cycle(
                        project_id, f'backlog_chunk:{chunk_num}:{len(assembled.events)}',
                        tier=opus_tier,
                        events=assembled.events,
                        assembled_payload=assembled,
                    ),
                    timeout=self.config.cycle_timeout_seconds,
                )
            except Exception as e:
                logger.error(f'Backlog chunk {chunk_num} failed: {e}')
                self.harness._escalate(
                    'recon_backlog_overflow', chunk_id,
                    f'Backlog chunk {chunk_num} failed, stopping iteration: {e}',
                )
                await self.buffer.restore_drained(project_id)
                return  # Stop iteration on failure

        # Final consolidation pass — skipped when the loop yielded on the
        # cumulative budget (task 2040): the remaining backlog stays
        # buffered for the next cycle instead of forcing an extra
        # consolidation cycle past the budget.
        if chunk_num > 0 and not yielded:
            logger.info(f'Backlog final consolidation for {project_id}')
            try:
                await asyncio.wait_for(
                    self.harness.run_full_cycle(
                        project_id, 'backlog_final_consolidation',
                        tier=opus_tier,
                    ),
                    timeout=self.config.cycle_timeout_seconds,
                )
            except Exception as e:
                logger.error(f'Backlog final consolidation failed: {e}')
                await self.buffer.restore_drained(project_id)
