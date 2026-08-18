"""Escalation MCP server — FastMCP tools for agents and handlers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from pydantic import ValidationError
from shared.branch_names import canonical_queued_branch_name
from shared.storm_counter import StormCounter
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot

from escalation import sweep as _sweep
from escalation.action_effects import effect_for
from escalation.authority import PROMOTE_ALLOWED, ROLE_LEVEL_ALLOWLIST, l2_auto_close_class
from escalation.dedupe import DedupeConfig
from escalation.dedupe import submit_or_dedupe as _dedupe_submit_or_dedupe
from escalation.models import (
    AGENT_FILABLE_LEVELS,
    BORN_AT_L2_SEVERITIES,
    KNOWN_SEVERITIES,
    RESOLUTION_CLASSES,
    Escalation,
    EvidenceEntry,
    max_severity,
)
from escalation.pins import classify_pins
from escalation.queue import EscalationQueue
from escalation.queue import observed_submit_response as _observed_submit_response

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


def _derive_l2_severity(queue: EscalationQueue, member_ids: list[str]) -> str | None:
    """Return max(member severities) for a promoted L2, or None if none is usable.

    This is what an OMITTED ``promote_to_l2(severity=...)`` argument resolves
    to (task 3976).  The old literal ``'blocking'`` default inflated every
    cluster of purely-informational L1s into a human-paging L2 — and did so
    non-deterministically, since the outcome hinged on whether the LLM caller
    happened to type the argument at all.

    Members are read through ``queue.get()`` rather than the queue root
    directly, so a member already resolved and archived between the watcher's
    drain and its promote still contributes its true severity (``get`` falls
    back to the archive), and repeated lookups of a genuinely nonexistent id
    are negative-cached rather than re-scanning the archive each time.

    **The fold ranges over ``KNOWN_SEVERITIES`` ONLY.**  A member is USABLE
    only if it resolves AND its ``severity`` is in the vocabulary.  Nothing
    validates a record's severity on write — ``Escalation`` is a plain
    dataclass and ``queue.submit``/``_rewrite`` are field-agnostic
    passthroughs — so a legacy, corrupt, or externally-written member can carry
    an out-of-vocabulary string (``''``, ``'warn'``).  ``max_severity`` ranks
    an unknown at 0, but its ``>=`` tie-break would let that string WIN over a
    genuine ``'info'`` sibling and be minted onto the L2 verbatim, reported
    back through the response ``severity`` key and missed by cockpit's
    ``severity_weights``.  Treating it as unusable keeps the tool's promise
    that a filed severity is always one of ``KNOWN_SEVERITIES``.

    **Returning None means "the members say nothing".**  The two call paths
    need different fail-safes, so this function reports the fact rather than
    picking one:

    - CREATE must choose a severity, so it fails safe UP to ``'blocking'`` —
      today's behaviour, unchanged.  Rejecting the promotion instead would
      silently drop a promotion the caller believed it had made, and quieting
      it to ``'info'`` would fail in the dangerous direction.  This matches
      ``escalation.pins`` link 2: an unknown severity fails safe to pinning,
      never to conversion.
    - UPDATE must NOT: the existing L2 already carries a severity derived from
      real members, so an underivable set has nothing to contribute.  Failing
      up there would inflate a correctly-inherited ``info`` L2 to ``blocking``
      (and bump ``updated_at``, re-triggering the watcher's re-assess) merely
      because an id was typo'd or momentarily unreadable.

    Either way the unusable ids are named at WARNING — loud, never silent.

    A PARTIALLY usable set derives from the usable subset only — discarding a
    known-info member because a sibling id was unreadable would reintroduce
    the very inflation this exists to remove.  The fold is therefore seeded
    from the first USABLE member rather than from ``'info'``: an empty-ish
    seed would be indistinguishable from a real info member, and the explicit
    nothing-usable branch is what carries the fail-safe.
    """
    resolved: list[str] = []
    unusable: list[str] = []
    for mid in member_ids:
        member = queue.get(mid)
        if member is None:
            unusable.append(mid)
        elif member.severity not in KNOWN_SEVERITIES:
            logger.warning(
                'promote_to_l2: member escalation %s carries out-of-vocabulary '
                'severity %r (expected one of %s); excluding it from the derived '
                'L2 severity rather than propagating it.',
                mid, member.severity, sorted(KNOWN_SEVERITIES),
            )
            unusable.append(mid)
        else:
            resolved.append(member.severity)

    if not resolved:
        logger.warning(
            'promote_to_l2: no member escalation yielded a usable severity for %s '
            '— cannot derive an L2 severity from members. Unusable ids: %s',
            member_ids, ', '.join(unusable) or '(none)',
        )
        return None

    if unusable:
        logger.warning(
            'promote_to_l2: %d of %d member escalation(s) yielded no usable '
            'severity; deriving from the usable subset only. Unusable ids: %s',
            len(unusable), len(member_ids), ', '.join(unusable),
        )

    derived = resolved[0]
    for sev in resolved[1:]:
        derived = max_severity(derived, sev)
    return derived


# The role the steward's own filings carry (orchestrator.steward
# _auto_escalate_to_human hard-codes ``agent_role='steward'``). Level-1 is the
# steward's documented recourse, so an L1 from this role — or from a harness
# sentinel — is the EXPECTED shape and is not logged.
_EXPECTED_L1_FILER_ROLE = 'steward'


def _warn_if_unexpected_l1_filer(agent_role: str, task_id: str, category: str) -> None:
    """Log a WARNING when a non-steward, non-sentinel role files at ``level=1``.

    Task 3236 amendment.  ``level=1`` is deliberately NOT role-gated, and that
    is a considered choice rather than an oversight:

    - ``agent_role`` is a free-form MCP tool argument, not an enforced
      property.  A hard gate on ``agent_role == 'steward'`` is defeated by any
      caller that simply passes that string, so it would buy the APPEARANCE of
      authority enforcement without the substance.
    - Worse, a hard REJECT is not fail-safe.  The steward briefing does not
      mandate any particular ``agent_role`` spelling, so a steward filing under
      a different string would have its re-escalation rejected and lost —
      re-introducing precisely the swallowed-re-escalation failure this task
      exists to fix.  The C4/D3 severity precedent is a DOWNGRADE (the record
      still lands), never a rejection.

    So the level axis is closed by OBSERVABILITY instead of by a gate: every
    unexpected L1 filing is loud and attributable in the server log, honouring
    the loud-over-silent-degradation norm.  ``_ESCALATION_INSTRUCTIONS`` in
    ``orchestrator.agents.roles`` states the same thing to agents.
    """
    role = agent_role or ''
    if role == _EXPECTED_L1_FILER_ROLE or _is_harness_sentinel_role(role):
        return
    logger.warning(
        'Level-1 escalation filed by agent_role=%r (task_id=%r, category=%r), '
        'which is neither %r nor a harness sentinel. Level 1 skips the steward, '
        'is read by escalation-watcher-auto (which may promote to L2), and pins '
        'the task via QUEUE_HANDOFF independently of the filer. This is allowed '
        'but recorded: agent_role is caller-supplied, so it is observed, not '
        'enforced.',
        role, task_id, category, _EXPECTED_L1_FILER_ROLE,
    )


# ---------------------------------------------------------------------------
# Connection-capability header names — resolve_issue enforcement
# (escalation-connection-capability-guard-prd.md, task alpha).
# ---------------------------------------------------------------------------
# No pre-existing house header convention was found in this codebase (a grep
# turned up only mcp-session-id / content-type in HTTP tests), so these adopt
# the PRD's proposed names. get_http_headers() lowercases every header key,
# so these constants are lowercase; HTTP header names are themselves
# case-insensitive on the wire, so a client may send any letter-casing.
_LEVELS_HEADER = 'x-escalation-levels'
_IDENTITY_HEADER = 'x-escalation-identity'


def _parse_levels(raw: str) -> set[int]:
    """Parse a comma-separated X-Escalation-Levels header value into a set of ints.

    Whitespace around each token is tolerated (e.g. ``"0, 1"``). Raises
    ``ValueError`` — fail-closed — when the header is empty/whitespace-only,
    any token is empty (e.g. ``"0,,1"``), or any token is not a non-negative
    integer (rejects negatives like ``"-1"`` and floats like ``"1.0"``), so a
    malformed restriction is never silently treated as "no restriction".
    """
    tokens = [t.strip() for t in raw.split(',')]
    if not tokens or any(not t or not t.isdigit() for t in tokens):
        raise ValueError(f'invalid X-Escalation-Levels value: {raw!r}')
    return {int(t) for t in tokens}


def _get_merge_worker(harness: Any | None) -> Any | None:
    """Return the live merge worker from *harness*, or None if not available.

    Centralises the ``getattr(harness, '_merge_worker', None)`` probe so that
    the private attribute name lives in exactly one place.  When *harness* is
    None (standalone mode / unit tests that wire no harness) returns None.
    Returns None — rather than raising — when the attribute is absent, so a
    future rename degrades gracefully rather than crashing; callers should warn
    if the worker is unexpectedly None.

    The ideal long-term fix is a public ``harness.merge_worker_snapshot()``
    accessor that removes the duck-typed coupling entirely; that requires a
    harness module change scoped to a separate task.
    """
    if harness is None:
        return None
    return getattr(harness, '_merge_worker', None)


def _get_terminal_retention(harness: Any | None) -> Any | None:
    """Return the harness-mounted TerminalOutcomeRetention ring, or None.

    Centralises the ``getattr(harness, '_terminal_retention', None)`` probe so
    the private attribute name lives in exactly one place, shared by the
    merge_request write side (records dispatch outcomes and coalesce
    aliases) and the merge_status / merge_cancel read sides
    (``_durable_terminal_state``'s Tier 2, ``retire_cancelled_merge_request``).
    Returns None — rather than raising — when *harness* is None (standalone
    mode / unit tests that wire no harness) or the attribute is absent, so a
    future rename degrades gracefully.

    No production code path currently constructs a TerminalOutcomeRetention
    and assigns it to ``harness._terminal_retention`` (task 3149 tracks
    deciding whether to wire it or delete the ring); until then this always
    resolves to None outside tests, and every call site keeps behaving
    exactly as it did before this accessor existed.
    """
    if harness is None:
        return None
    return getattr(harness, '_terminal_retention', None)


def _require_matching_project_root(harness: Any, project_root: str) -> str | None:
    """Return an error message if *project_root* doesn't match this server's project.

    The escalation server closes over exactly one project's ``harness``, so
    *project_root* can never be used to route a call — it is a defensive
    validation guard.  Resolved-path equality (not string equality) so
    trailing slashes / relative segments / symlink differences don't produce
    false-positive mismatches.  Returns None when they match.

    Shared by claim_warm_worktree and release_warm_worktree (PRD β, task
    2011) so a multi-project interactive caller (/do, /warm) that mis-targets
    an escalation MCP endpoint gets a clean testable failure instead of
    claiming/releasing a worktree against the wrong orchestrator.
    """
    gops_root = Path(harness.git_ops.project_root).resolve()
    given_root = Path(project_root).resolve()
    if given_root != gops_root:
        return (
            f'project_root mismatch: {project_root!r} does not match this '
            f"server's wired project ({harness.git_ops.project_root!s}, "
            f'resolved {gops_root!s}).'
        )
    return None


def _project_task_runtime_entry(s: Any) -> TaskRuntimeEntry:
    """Project one duck-typed ``TaskRuntimeState``-shaped object into the
    wire-checked ``TaskRuntimeEntry`` (used by ``get_task_runtime_state``).

    ``phase``/``lane_state`` are machine-checked ``Literal`` vocabularies on
    the wire (INV-1) but free-form ``str`` on the orchestrator source
    (``orchestrator.task_runtime.TaskRuntimeState``, produced by that
    module's ``_derive_phase``/``_LANE_STATE_MAP``). Both currently only
    ever emit values inside today's vocab, but that is a cross-package
    coupling, not an enforced invariant: if a future orchestrator change
    ever emits a value outside it, this degrades ONLY that one task to an
    honest per-task error entry (loud, but isolated) rather than letting a
    single bad task's ``ValidationError`` take down the entire snapshot —
    every other task must still render on the dashboard.
    ``TaskRuntimeEntry.model_construct`` (bypasses validation) is used for
    the fallback so a pathological double-fault (e.g. ``task_id`` itself
    also being malformed) still cannot raise.
    """
    try:
        return TaskRuntimeEntry(
            task_id=s.task_id,
            has_worktree=s.has_worktree,
            loops=s.loops,
            attempts=s.attempts,
            started=s.started,
            lane=s.lane,
            phase=s.phase,
            lane_state=s.lane_state,
            error=getattr(s, 'error', None),
        )
    except ValidationError as exc:
        logger.warning(
            'get_task_runtime_state: task %r failed wire-contract validation, '
            'degrading to an honest error entry: %s',
            getattr(s, 'task_id', '<unknown>'), exc,
        )
        return TaskRuntimeEntry.model_construct(
            task_id=s.task_id,
            has_worktree=s.has_worktree,
            loops=None,
            attempts=None,
            started=None,
            lane=None,
            phase=None,
            lane_state=None,
            error=f'wire-contract violation: {exc}'[:200],
        )


# C1 action enum for resolve_issue — five valid values, two disposition buckets.
RESOLVE_ACTIONS: tuple[str, ...] = ('resume', 'restart', 'park', 'abandon', 'close_only')
# park is no longer dismissed — it keeps the record open at L2 (version-a).
_DISMISS_ACTIONS: frozenset[str] = frozenset({'abandon', 'close_only'})

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
    # Verified-green stranded remediation: durable merge/verify failure of a
    # reaper-submitted branch (stranding-remediation-scheduler-ergonomics-prd.md
    # leaf α).  Inert/unvalidated (action_effects.py:23) — vocabulary parity only.
    'stranded_merge_failed',
]

# OUTPUT keys of a compact row — the triage-relevant subset a long-running L2
# watcher needs to decide whether to pull a full record.  NOTE these are output
# keys, not model field names: ``member_ids`` is a PROJECTION of the model's
# ``members`` list (renamed once, in _compact_escalation below), so this tuple is
# no longer a pure key subset of Escalation.to_dict().
# The heavy fields (detail, options, train_state, workflow_state, worktree,
# dedupe_*) are still dropped to keep the watcher's context small as the pending
# pile grows during an AFK window; `detail` in particular is the unbounded
# free-text field that motivated compact mode.  ``root_cause`` (a one-line dedup
# key) and ``member_ids`` (a short id list) are bounded by construction and are
# what let a rotating watcher rebuild `already_promoted` from the drain ALONE,
# with no session memory (task 3997, C1).  ``amendments`` is deliberately NOT
# projected, so preserved incoming framing never inflates a drain.
# The triage-ack fields (triaged_at, triaged_by, triage_note, updated_at) are
# included so a compact drain can decide stamp-then-skip without a per-record
# get_escalation round-trip.
_COMPACT_ESCALATION_FIELDS = (
    'id', 'task_id', 'category', 'severity', 'level', 'status',
    'summary', 'suggested_action', 'timestamp',
    'triaged_at', 'triaged_by', 'triage_note', 'updated_at',
    'root_cause', 'member_ids',
)

# get_pending_escalations(compact=True) additionally keeps its computed
# pins_recovery annotation: the dashboard reads compact records, so dropping it
# here would blank the whole PINNING surface.  Kept as a separate tuple so
# get_task_escalations — which never computes the annotation — is untouched.
_COMPACT_PENDING_FIELDS = (*_COMPACT_ESCALATION_FIELDS, 'pins_recovery')


def _compact_escalation(d: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Project one Escalation.to_dict() into a compact row over *fields*.

    THE single site that knows compact rows expose ``member_ids`` where the model
    says ``members`` (task 3997).  Both compact apply sites route through here so
    the rename exists exactly once; widening _COMPACT_ESCALATION_FIELDS is then
    enough to change both tools' wire shape.

    Keys absent from *d* are OMITTED rather than defaulted.  That is not
    defensive tidiness: ``pins_recovery`` is deliberately absent when it cannot
    be computed, and emitting a false ``[]`` there reads as "nothing pins this
    task" — the exact collapse (esc-3163) the omission contract exists to
    prevent.
    """
    row: dict[str, Any] = {}
    for k in fields:
        if k == 'member_ids':
            # Projection, not a model field: renamed from `members`.  Copied so a
            # caller mutating the row cannot reach back into the loaded record.
            row[k] = list(d.get('members', []))
        elif k in d:
            row[k] = d[k]
    return row


# INV-4 storm escape for L2 amendment truncation (task 3997).  Sizing: with
# ``queue._MAX_AMENDMENTS`` = 20, ONE L2 has to fold 21+ times inside the window
# to truncate even ONCE, so three truncations in an hour is not routine churn.
# It says either the cap is systematically wrong for the live fold rate, or
# root-cause matching is over-folding unrelated clusters into a single L2 — and
# task 3998 canonicalises that matching, which RAISES the fold rate BY DESIGN.
# Both readings are worth a human-adjacent signal rather than a WARNING nobody
# reads; the durable per-record ``amendments_truncated`` counter remains the
# primary structured fact (INV-8), this is the notification layered on it.
#
# Module constants rather than a config leaf, following the sanctioned
# precedent of ``reconciliation/harness.py``'s ``_PLACEHOLDER_DROP_STORM_*``,
# whose stated reason — the counter is private to this module — holds
# identically here.  They are still passed per ``record()`` call because that
# is ``StormCounter``'s API (see its RELOAD SAFETY note).
_AMENDMENT_TRUNCATION_STORM_THRESHOLD = 3
_AMENDMENT_TRUNCATION_STORM_WINDOW_SECONDS = 3600.0  # 1 h


# Task statuses from which a recovery/redispatch is still possible.  A record
# on a task outside this set pins nothing: there is no recovery to block.
_RECOVERABLE_STATUSES = frozenset({'in-progress', 'blocked'})


async def _annotate_pins_recovery(
    queue: EscalationQueue,
    harness: Any,
    escalations: Sequence[Any],
    dicts: list[dict[str, Any]],
    *,
    level: int | None,
) -> None:
    """Stamp ``pins_recovery`` on *dicts* in place, or leave the key ABSENT.

    See ``get_pending_escalations``' docstring for the contract.  The key is
    omitted rather than defaulted to ``[]`` on every path where the answer is
    unknown, because ``[]`` reads as "nothing pins this task" — the esc-3163
    collapse that routes a genuinely-pinned strand down the wrong branch.
    """
    if not dicts:
        return
    # No `if harness is not None` guard: getattr(None, 'scheduler', None) is
    # already None, so the test is redundant AND actively harmful — it narrows
    # `harness` to None for the whole function under pyright (the `else None`
    # arm survives the `scheduler is None` return, since narrowing does not
    # propagate backwards from `scheduler` to `harness`), which is what made
    # the liveness read below a reportOptionalMemberAccess error.
    scheduler = getattr(harness, 'scheduler', None)
    if scheduler is None:
        logger.debug(
            'pins_recovery omitted for %d pending record(s): no orchestrator '
            'harness/scheduler wired in, so task status is unreadable',
            len(dicts),
        )
        return

    task_ids = sorted({e.task_id for e in escalations})
    # Deliberately unguarded.  Scheduler.get_statuses is TOTAL: it wraps its
    # own body and returns ({}, exc) on any failure, logging the traceback
    # itself (scheduler.py:2560).  Its designed failure channel is the `err`
    # slot read just below.  A raise here would mean the duck-typed harness
    # violated that contract — a seam failure, caught once by the call site's
    # guard in get_pending_escalations, not swallowed at DEBUG per call.
    statuses, err = await scheduler.get_statuses(task_ids)
    if err is not None:
        # get_statuses' failure shape is ({}, exc); reading only the dict would
        # make a failed read indistinguishable from "no tasks" and report [].
        logger.debug('pins_recovery omitted: status read failed: %s', err)
        return

    # classify_pins judges a record against its task's WHOLE open set, so a
    # filtered view must not become a filtered classification.  Only a `level`
    # filter narrows that set; without one the selection already IS the full
    # open set for every task it mentions.
    if level is None:
        open_by_task: dict[str, list[Any]] = {}
        for esc in escalations:
            open_by_task.setdefault(esc.task_id, []).append(esc)
    else:
        open_by_task = {}
        for tid in task_ids:
            try:
                open_by_task[tid] = queue.get_by_task(tid, status='pending')
            except Exception as exc:  # noqa: BLE001 — real I/O, see below
                # The ONE genuinely-reachable failure in this function (a
                # filesystem scan plus JSON parse over esc-*.json), and the one
                # place per-task recovery is real rather than nominal: a single
                # unreadable file degrades exactly one task.  Leaving `tid` out
                # of open_by_task makes the loop below skip it, so its key stays
                # ABSENT (= UNKNOWN) instead of becoming a false [].  WARNING
                # because a queue directory this process cannot read is
                # operator-actionable and otherwise invisible on this surface.
                logger.warning(
                    'pins_recovery UNKNOWN for task %s: pending re-read failed: %s',
                    tid, exc,
                )

    reports: dict[str, Any] = {}
    live_by_task: dict[str, bool] = {}
    for tid in task_ids:
        if tid not in open_by_task:
            continue
        # Deliberately unguarded.  is_workflow_active is a dict membership test
        # (harness.py:11182), so it cannot fail for one task id and succeed for
        # the next — a per-task `continue` here would be granularity in name
        # only.  A harness that cannot answer at all is a seam failure, handled
        # once by the call site's guard.
        live = bool(harness.is_workflow_active(tid))
        live_by_task[tid] = live
        # live_claimant_id is unavailable here (the harness exposes no composed
        # claimant id), which classify_pins treats as UNKNOWN and fails safe to
        # pinning for a live-claimant L0.  Fail-safe is the correct direction:
        # this annotation must never under-report a pin.
        reports[tid] = classify_pins(tid, open_by_task[tid], live_claimant=live)

    for esc, d in zip(escalations, dicts, strict=True):
        tid = esc.task_id
        status = statuses.get(tid)
        report = reports.get(tid)
        if status is None or report is None:
            logger.debug(
                'pins_recovery omitted for %s: task %s status/classification unresolved',
                d.get('id'), tid,
            )
            continue
        pins = (
            status in _RECOVERABLE_STATUSES
            and not live_by_task.get(tid, True)
            and esc.id in report.queue_handoff
        )
        d['pins_recovery'] = [tid] if pins else []


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
    startup_sweep_now: datetime | None = None,
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

    *startup_sweep_now* (default None) — injectable reference datetime forwarded
    to ``run_startup_sweep(now=...)`` which in turn forwards it to
    ``archive.prune_archive(now=...)``.  When None, live UTC is used (production
    default).  Pass a fixed datetime in tests to make the prune cutoff
    deterministic and wall-clock-independent.
    """
    mcp = FastMCP('escalation')
    cfg = dedupe_config if dedupe_config is not None else DedupeConfig()

    # --- Startup sweep (pre-serving single-writer window) ---
    if startup_sweep:
        try:
            _sweep.run_startup_sweep(queue.queue_dir, now=startup_sweep_now)
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

        Response shapes (from dedupe.submit_or_dedupe).  ``level`` is present on
        every branch:
        - Queued:        ``{'id': esc_id, 'status': 'queued', 'level': <persisted>}``
        - Auto-resolved/dismissed (the record was NOT pending after the write,
          e.g. a concurrent sweep won the race): ``{'id', 'status',
          'resolution', 'resolved_by', 'level'}``.  Task 3236: both this
          function's L2 branch and dedupe.submit_or_dedupe report OBSERVED
          post-write state rather than write intent, and fail open to
          ``'queued'`` — carrying ``esc.level`` — when the re-read is
          unavailable.
        - Dedup-skipped: ``{'id': parent_id, 'status': 'dedup_skipped',
                            'parent_id': parent_id, 'child_id': esc.id,
                            'level': esc.level}``
          (never returned for L2 escalations — they always produce 'queued')

        Cross-task resume contract: see DESIGN.md "Escalation cross-task dedupe"
        — the child re-runs on its next workflow invocation; no per-child wake
        signal is emitted.
        """
        # L2 escalations bypass deduplication: their level=2 stamp must be
        # preserved on an independent on-disk record (see docstring above).
        if esc.severity in BORN_AT_L2_SEVERITIES:
            esc_id = queue.submit(esc)
            # Task 3236: this branch does NOT route through dedupe, so it needs
            # the observed-state response separately.  Fail-open to 'queued'
            # (still carrying esc.level, so the 'level' echo is never missing).
            return _observed_submit_response(queue, esc_id, fallback_level=esc.level)
        return _dedupe_submit_or_dedupe(queue, esc, cfg)

    # --- Amendment-truncation storm escape (INV-4, task 3997) ---

    # PROCESS-LOCAL and per-instance BY CONSTRUCTION.  StormCounter documents
    # its state as resetting on restart and not bleeding between servers (or
    # between tests), and server.py otherwise holds zero module-level mutable
    # state — every module-level name above is a frozen constant.  Keeping the
    # counter in this closure is what preserves that property.
    _amendment_truncation_storm = StormCounter()

    def _report_amendment_truncation_storm(l2_id: str, task_id: str) -> None:
        """File ONE info escalation when amendment truncation BURSTS.

        ``queue.add_members_to_l2`` already counts every dropped amendment on
        the record itself (``amendments_truncated``) and logs a WARNING.  The
        counter stays the PRIMARY structured fact — the contract is assertable
        from the record, never by log-scrape (INV-8) — but a WARNING has no
        audience.  This is the rate-thresholded NOTIFICATION layered on top,
        which is what INV-4 asks for: a hearer, at a threshold.

        Deliberately lives here and not in ``queue.py``.  That module is a pure
        storage leaf, and a self-file from inside ``add_members_to_l2`` would
        re-enter ``make_id``/``submit``/``_atomic_write`` while still holding
        ``escalation_id_lock``.  ``promote_to_l2`` already calls ``queue.submit``
        on its create path and runs outside that flock.

        PURELY ADDITIVE, NEVER FATAL, mirroring the house analogues
        ``emit_markup_storm_escalation`` and
        ``emit_residual_candidate_key_escalation``: nothing raised in here may
        fail the promote that triggered it.  A dropped report costs a
        notification; a raised one would cost the fold.
        """
        try:
            storm = _amendment_truncation_storm.record(
                threshold=_AMENDMENT_TRUNCATION_STORM_THRESHOLD,
                window_seconds=_AMENDMENT_TRUNCATION_STORM_WINDOW_SECONDS,
                # The label is load-bearing, not decoration: it is what lets the
                # report name WHICH L2s truncated instead of blaming whichever
                # call happened to cross the threshold.
                label=l2_id,
            )
            # None means below threshold, or a previous fire is still inside the
            # window (one report per window, so a runaway escalates once).
            if storm is None:
                return
            labels = ', '.join(storm['labels']) or l2_id
            _submit_or_dedupe(Escalation(
                id=queue.make_id(task_id),
                task_id=task_id,
                agent_role='escalation-server',
                # A report about lost framing is a notification, not a page:
                # 'info' keeps it off the born-at-L2 human-direct route.
                severity='info',
                category='infra_issue',
                summary=(
                    f"L2 amendment truncation storm: {storm['count']} truncations "
                    f"in {storm['window_seconds']}s "
                    f"(threshold {storm['threshold']}); L2s: {labels}"
                ),
                detail=(
                    f"OBSERVED: {storm['count']} amendment truncations within "
                    f"{storm['window_seconds']}s across L2 escalation(s): {labels}.\n"
                    f"Each truncation drops the OLDEST entry of that L2's "
                    f"`amendments` list at the queue._MAX_AMENDMENTS cap; the "
                    f"record's own `amendments_truncated` field holds the durable "
                    f"per-L2 total, and its own root_cause/detail/options/summary "
                    f"are never touched.\n"
                    f"Hypothesis: either the cap is too low for the live fold "
                    f"rate, or root-cause matching is over-folding unrelated L1 "
                    f"clusters into one L2."
                ),
                suggested_action=(
                    'Read the named L2s and compare each record\'s own framing '
                    'against its amendments to judge whether those folds belong '
                    'together; then either raise queue._MAX_AMENDMENTS or tighten '
                    'root-cause matching.'
                ),
            ))
        except Exception as e:  # pragma: no cover - defensive, never fatal
            logger.exception(
                'amendment-truncation storm report failed for L2 %s (non-fatal): %s',
                l2_id, e,
            )

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
        #
        # Task 3236 ordering dependency — do NOT move this above the Escalation
        # construction in escalate_blocker.  That tool now accepts an explicit
        # `level` (restricted to {0, 1}), stamped at construction time; because
        # this assignment runs AFTER construction, the born-at-L2 severity route
        # keeps precedence over any explicitly passed level, which is the
        # intended precedence and is pinned by
        # test_server.py::TestEscalateBlockerLevelParam.
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
            # resolution_class='benign' is explicit (not left to the tier-default
            # fallback): the underlying task was already terminal at filing time,
            # so no action was needed or taken beyond this auto-close — the Seam-1
            # definition of benign — and this resolver isn't in the reaper-sweep
            # tier, so an unstamped record here would otherwise be misread as
            # 'actionable' by the effective_benign() proxy.
            resolved = queue.submit_resolved(
                esc,
                f'auto-resolved: task already terminal (status={status})',
                resolved_by='escalation-mcp-pre-submit-check',
                resolution_class='benign',
            )
            return {
                'id': resolved.id,
                'status': resolved.status,
                'resolution': resolved.resolution,
                'resolved_by': resolved.resolved_by,
                # Task 3236: `resolved` is the persisted record, so echoing its
                # level costs no read and keeps the documented 'level' echo
                # present on this branch too.
                'level': resolved.level,
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
        evidence: list[dict[str, Any]] | None = None,
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

        *evidence* — optional list of structured raw-OBSERVATION entries, each a
        ``{observation, measured_at, ref}`` dict (e.g. the HEAD SHA at measurement,
        a ref listing, a rerun result, a raw exit code).  Stored and returned
        verbatim (no shape validation).  State OBSERVATIONS as fact — in
        ``summary``, ``detail`` and each ``evidence.observation`` record only what
        was measured, never an unverified cause.  Put any causal diagnosis on a
        clearly-marked hypothesis line (prefix ``Hypothesis:``), never asserted as
        fact.  A single observation is not sufficient to recommend a destructive
        intervention (a ref move / rewind) — re-run or re-measure first.

        Response shape (``level`` is present on every branch):
        - Queued (task alive):    ``{id, status, level}``  where status='queued'
        - Deduped (folded):       ``{id, status, parent_id, child_id, level}``
          (L2 escalations are never deduped — they always produce 'queued')
        - Auto-resolved (terminal task): ``{id, status, resolution, resolved_by, level}``
        - Already resolved/dismissed at filing time (a concurrent resolver or
          sweep won the race): ``{id, status, resolution, resolved_by, level}``
          with the record's REAL status — the response reports observed
          post-write state, never write intent (task 3236).
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
            evidence=cast(list[EvidenceEntry], evidence or []),
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
        evidence: list[dict[str, Any]] | None = None,
        terminal_state_is_the_bug: bool = False,
        level: int = 0,
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

        *level* — the escalation ladder rung this filing is born at.  Defaults to
        ``0`` (agent → steward).  Pass ``level=1`` to file a level-1
        re-escalation (steward → escalation-watcher-auto): this is the steward's
        documented recourse when it cannot resolve an L0 itself, and it is the
        ONLY way an agent-side filing reaches the auto-watcher, which filters on
        level.  A level-1 record is also outside the workflow's level=0-scoped
        dismissal sweeps by construction, and ``escalation.pins`` routes any
        ``level != 0`` record to QUEUE_HANDOFF, so it pins the task independently
        of the filer's liveness.

        ``level=1`` is not restricted to any role — ``agent_role`` is
        caller-supplied and so cannot be enforced, and a hard reject would lose
        a steward re-escalation filed under an unexpected role string.  It is
        instead OBSERVED: a level=1 filing from a role that is neither
        ``'steward'`` nor a harness sentinel is logged at WARNING naming the
        role and task_id.  The record is still filed.

        Only ``{0, 1}`` are accepted — anything else (including ``2``) is
        rejected with an ``{'error': ...}`` response and NOTHING is submitted.
        Agents must not self-mint L2: the legitimate routes there are a
        born-at-L2 *severity* filed by a harness sentinel role, or
        ``promote_to_l2``.  Note that born-at-L2 severity takes precedence over
        this parameter — a sentinel-filed ``severity='critical'`` still lands at
        level 2 even when ``level=1`` is passed.

        *evidence* — optional list of structured raw-OBSERVATION entries, each a
        ``{observation, measured_at, ref}`` dict (e.g. the HEAD SHA at measurement,
        a ref listing, a rerun result, a raw exit code).  Stored and returned
        verbatim (no shape validation).  State OBSERVATIONS as fact — in
        ``summary``, ``detail`` and each ``evidence.observation`` record only what
        was measured, never an unverified cause.  Put any causal diagnosis on a
        clearly-marked hypothesis line (prefix ``Hypothesis:``), never asserted as
        fact.  A single observation is not sufficient to recommend a destructive
        intervention (a ref move / rewind) — re-run or re-measure first.

        Response shape always includes ``action='terminate_cleanly'`` and
        ``level`` (on EVERY branch, including the fail-open one) plus:
        - Queued:        ``{id, status, level, action}``  where status='queued'
        - Deduped:       ``{id, status, parent_id, child_id, level, action}``
          (L2 escalations are never deduped — they always produce 'queued')
        - Auto-resolved: ``{id, status, resolution, resolved_by, level, action}``
        - Already resolved/dismissed at filing time (a concurrent resolver or
          sweep won the race): ``{id, status, resolution, resolved_by, level,
          action}`` with the record's REAL status.  Task 3236: the response
          reports observed post-write state, never write intent — a
          ``status='queued'`` reply now means the record really was pending
          after the write.  ``level`` echoes the level actually persisted
          (falling back to the level written when a post-write re-read is
          unavailable), so a caller that passed ``level=1`` can confirm it
          landed without risking a ``KeyError`` on a degraded path.
          Callers needing the full record can call get_escalation(id).
        """
        if severity not in KNOWN_SEVERITIES:
            return {
                'error': (
                    f'invalid severity {severity!r}; '
                    f'expected one of {sorted(KNOWN_SEVERITIES)}'
                ),
            }
        # Task 3236: validate `level` BEFORE constructing the Escalation, using
        # the same {'error': ...} early-return shape as the severity guard above,
        # so a misconfigured caller gets immediate feedback instead of a
        # silently-misrouted escalation.  bool is an int subclass — reject it
        # explicitly so escalate_blocker(level=True) is not read as level=1.
        if isinstance(level, bool) or not isinstance(level, int) or level not in AGENT_FILABLE_LEVELS:
            return {
                'error': (
                    f'invalid level {level!r}; expected one of '
                    f'{sorted(AGENT_FILABLE_LEVELS)} (0 = agent→steward, '
                    '1 = steward re-escalation → escalation-watcher-auto). '
                    'Agents cannot self-mint level 2: file with a born-at-L2 '
                    "severity ('critical'/'urgent') from a harness sentinel role, "
                    'or use promote_to_l2.'
                ),
            }
        # Level=1 is not role-gated (see _warn_if_unexpected_l1_filer for why a
        # gate on caller-supplied agent_role would be both defeatable and
        # fail-dangerous); an unexpected filer is made observable instead.
        if level == 1:
            _warn_if_unexpected_l1_filer(agent_role, task_id, category)
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
            evidence=cast(list[EvidenceEntry], evidence or []),
            level=level,
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
        resolution_class: str | None = None,
        granted_files: list[str] | None = None,
        escalate_model: bool = False,
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
        +--------------+------------+----------------------------+------------------------+
        | park         | kept OPEN  | workflow killed; task      | → blocked (held under  |
        |              | at L2      | blocked-on-human; no re-   | open L2 escalation)    |
        |              |            | dispatch while open        |                        |
        +--------------+------------+----------------------------+------------------------+
        | abandon      | dismissed  | task cancelled outright    | cancelled              |
        +--------------+------------+----------------------------+------------------------+
        | close_only   | dismissed  | escalation closed with no  | unchanged              |
        |              |            | workflow effect            |                        |
        +--------------+------------+----------------------------+------------------------+

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

        ``granted_files`` (task 2505): an optional structured scope-expansion
        grant — a list of file-level, project-relative paths — consumed only
        on the ``action='resume'`` orchestrator path (it widens
        ``plan.files``/``metadata.files``/locks there). Distinct from
        ``resolution``, which stays free-text human-readable rationale. Not
        forwarded to the ``park`` action; omit to leave the record's
        ``granted_files`` at its existing value (``[]`` for a fresh record).

        ``resolution_class`` optionally stamps an explicit
        ``'benign'``/``'actionable'`` classification
        (escalation-lifecycle-dashboard-prd.md Contract Seam 1). Validated
        against ``escalation.models.RESOLUTION_CLASSES`` before any gate runs
        or the record is touched; an unrecognised value returns
        ``{'error': ..., 'code': 'invalid_resolution_class'}`` with NO record
        change. Written only at resolution — never author-supplied
        otherwise. Not forwarded to the ``park`` action (the record stays
        open at L2, unclassified until eventually resolved).

        ``escalate_model`` (task μ, adaptive-routing trigger 3): when True and
        the action leads to a *next dispatch* (``resume`` / ``restart``), the
        resolver best-effort pre-increments the task's
        ``metadata.routing.routing_tier`` via
        ``harness.pre_increment_routing_tier(rec.task_id)``, so that the next
        dispatch routes its executor one ladder rung stronger (the
        ``retry-tier-up`` policy rule fires at ``routing_tier >= 1``). It is a
        soft telemetry hint: the write is delegated to the harness (which owns
        the metadata write path and the event loop) and wrapped in try/except
        so it can never fail the resolve. NOT applied on the ``park``
        early-return branch (park keeps the task blocked with no re-dispatch)
        nor for the dismiss actions (``abandon`` / ``close_only`` — no next
        dispatch). Degrades to a no-op when no harness is wired.

        **Table B** (``escalation.action_effects``) is the single authority for
        action legality, consulted BEFORE any record mutation:

        - ``resume``     -> target_status ``pending``,        disposition ``resume_from_pause``
        - ``restart``    -> target_status ``pending``,        disposition ``restart_from_scratch``
        - ``park``       -> target_status ``blocked``,        disposition ``park_kill_block_keep_l2_open``
        - ``abandon``    -> target_status ``cancelled``,      disposition ``abandon_kill_cancel``
        - ``close_only`` -> target_status ``None`` (no-op),   disposition ``no_effect``

        An ``(action, level, category)`` with no defined ``TaskEffect`` — today,
        any ``action`` outside the five above — is rejected loudly:
        ``{'error': ..., 'code': 'illegal_transition'}``, with NO record change.
        Level and category never narrow legality today (see the
        ``escalation.action_effects`` module docstring for the archive
        verification behind this).

        **Gate precedence**: the connection-capability gate (``bad_capability_header``
        / ``level_forbidden``, checked first) runs BEFORE this Table B legality
        gate. A caller that both asserts a forbidden/unparseable capability
        header AND passes an illegal ``action`` receives the capability error,
        not ``illegal_transition`` — no record is mutated by either gate, so
        this ordering is an error-reporting precedence only, not a correctness
        difference.

        NOTE: the ``target_status`` values above are not yet written by
        resolve_issue — this call changes only the escalation record; the
        task-status effects described in the table earlier in this docstring
        remain owned by the orchestrator harness. Wiring the harness to
        consume Table B directly is a separate, out-of-scope follow-up.
        """
        if terminate is not None:
            return {
                'error': (
                    "'terminate' was removed; state your intent: "
                    "action='resume'|'restart'|'park'|'abandon'|'close_only' "
                    "— see resolve_issue docstring."
                )
            }

        # resolution_class validation (escalation-lifecycle-dashboard-prd.md
        # Contract Seam 1) — checked before any gate, read, or mutation, so an
        # invalid value leaves the record fully untouched (INV-1 nothing-
        # persisted-on-rejection).
        if resolution_class is not None and resolution_class not in RESOLUTION_CLASSES:
            return {
                'error': (
                    f'invalid resolution_class {resolution_class!r}; expected '
                    f'one of {sorted(RESOLUTION_CLASSES)}'
                ),
                'code': 'invalid_resolution_class',
            }

        # Connection-capability gate (escalation-connection-capability-guard-prd.md,
        # task alpha; identity-derived ceiling extension per
        # plans/task-status-authority-prd.md contract C8 / decision D7).
        # get_http_headers() returns {} outside an ASGI request context
        # (in-process tool.fn() calls, stdio transport), so this gate is a no-op for
        # those callers — default-open is preserved byte-for-byte.
        headers = get_http_headers()
        levels_raw = headers.get(_LEVELS_HEADER)
        parsed: set[int] | None = None
        if levels_raw is not None:
            try:
                parsed = _parse_levels(levels_raw)
            except ValueError:
                return {
                    'error': (
                        f'unparseable X-Escalation-Levels header {levels_raw!r}; '
                        'expected comma-separated non-negative ints'
                    ),
                    'code': 'bad_capability_header',
                }
        identity = headers.get(_IDENTITY_HEADER)

        # Single record lookup reused by the capability gate below, the Table B
        # gate further down, and the pre-stamp write after it. queue.get() reads
        # from disk, so identity-mapped callers previously paid for two reads
        # per resolve (once for the ceiling check, again for Table B); one
        # fetch is now shared by both gates (efficiency note, PRD C8/D7 review).
        rec = queue.get(escalation_id)
        if rec is None:
            return {'error': f'Escalation {escalation_id} not found'}

        # D7 effective ceiling: an identity mapped in ROLE_LEVEL_ALLOWLIST is
        # AUTHORITATIVE — a present X-Escalation-Levels header may only
        # NARROW within that role ceiling (set intersection), never widen
        # past it, and dropping the header entirely still leaves the bare
        # role ceiling in force. An identity absent from
        # ROLE_LEVEL_ALLOWLIST (or no identity at all) falls back to the
        # pre-existing 2041 header-opt-in behaviour: `parsed` if a header
        # was sent, else None — unrestricted (the esc-2087-2 human-channel
        # guarantee; header-less callers are never default-denied).
        role_ceiling = ROLE_LEVEL_ALLOWLIST.get(identity) if identity is not None else None
        if role_ceiling is not None:
            effective = (role_ceiling & parsed) if parsed is not None else role_ceiling
        else:
            effective = parsed

        # Task 2630: narrow above-ceiling close_only carve-out for the
        # auto-watcher identity at L2 — only consulted here, where the role
        # ceiling would otherwise deny the call. l2_auto_close_class re-gates
        # on identity==watcher/level==2/action=='close_only' internally, so a
        # header-less connection (identity is None), any other identity, or
        # any level/action outside that narrow triple returns None
        # immediately, making the combined condition below equivalent to the
        # pre-2630 `effective is not None and rec.level not in effective`
        # check — byte-for-byte unaffected. A returned class name means an
        # allowlisted class matched AND its required structural evidence was
        # present in `resolution`; in that case the condition is False and we
        # fall through, letting the existing identity->resolved_by stamp,
        # Table B gate, and close path below handle the rest.
        if (
            effective is not None
            and rec.level not in effective
            and l2_auto_close_class(
                identity=identity,
                level=rec.level,
                action=action,
                category=rec.category,
                agent_role=rec.agent_role,
                resolution=resolution,
            )
            is None
        ):
            return {
                'error': (
                    f'connection not permitted to change level-{rec.level} '
                    'escalations'
                ),
                'code': 'level_forbidden',
            }
        if identity is not None:
            # Server-attributed identity overrides the tool arg for both the
            # park stamp (below) and the resolve call further down — a caller
            # cannot spoof resolved_by once the connection asserts an identity.
            resolved_by = identity

        # Table B legality gate (plans/task-status-authority-prd.md contract C5,
        # decisions D1/D2) — the SINGLE authority for resolve_issue action
        # legality, consulted BEFORE any record mutation. Replaces the old bare
        # `action not in RESOLVE_ACTIONS` check: an unrecognised action now
        # returns a typed error (mirroring the capability gate above) instead
        # of an untyped one; no record is changed either way.
        #
        # Precedence note: this gate runs AFTER the connection-capability gate
        # above, so a caller that fails BOTH gates (a forbidden/unparseable
        # capability header AND an illegal action) sees 'bad_capability_header'
        # / 'level_forbidden', not 'illegal_transition'. Neither gate mutates
        # the record, so this is an error-reporting precedence only — see the
        # "Gate precedence" note in the resolve_issue docstring. `rec` was
        # already fetched above (shared with the capability gate); nothing
        # mutates the record between that fetch and here, so it is reused
        # rather than re-read from disk.
        effect = effect_for(action, rec.level, rec.category)
        if effect is None:
            return {
                'error': (
                    f'illegal resolution: no TaskEffect for (action={action!r}, '
                    f'level={rec.level}, category={rec.category!r}); expected '
                    f'action in {list(RESOLVE_ACTIONS)}'
                ),
                'code': 'illegal_transition',
            }

        if action == 'park':
            # Version-a: park keeps the escalation open at L2.
            # queue.park() handles all stamping (level=2, resolution_action='park',
            # resolution text) and fires the resolve callback for teardown WITHOUT
            # archiving the record.  No pre-stamp needed — park() owns the full write.
            esc = queue.park(
                escalation_id, resolution,
                resolved_by=resolved_by, resolution_turns=resolution_turns,
            )
            if esc is None:
                return {'error': f'Escalation {escalation_id} not found'}
            return esc.to_dict()

        # Pre-stamp resolution_action on the pending record so resolve()'s
        # read-modify-write carries it into the archived JSON (C1 persistence).
        # Guard: only rewrite pending records — archived records must not be resurrected.
        # Reuses `rec` fetched at the top of the gate sequence above (shared by
        # the capability and Table B gates) instead of re-reading from disk:
        # queue.get() is a pure read and nothing mutates the record between
        # that fetch and here (the park branch above already returned), so
        # `rec` is still current and a further queue.get() would be redundant.
        if rec.status == 'pending':
            rec.resolution_action = action
            queue._rewrite(escalation_id, rec)
        dismiss = action in _DISMISS_ACTIONS
        esc = queue.resolve(
            escalation_id, resolution, dismiss=dismiss,
            resolved_by=resolved_by, resolution_turns=resolution_turns,
            resolution_class=resolution_class,
            granted_files=granted_files,
        )
        if esc is None:
            return {'error': f'Escalation {escalation_id} not found'}

        # escalate_model (task μ, adaptive-routing trigger 3): after a
        # SUCCESSFUL resolve of a next-dispatch action (resume/restart),
        # best-effort pre-increment the task's routing tier so its next
        # dispatch routes one ladder rung stronger (retry-tier-up rule).
        # Delegated to the harness — which owns the metadata write path and
        # the event loop — and guarded via getattr so a None/stub harness
        # degrades gracefully; wrapped in try/except so a telemetry hiccup can
        # never fail the resolve. Park returns early above (no bump); the
        # dismiss actions (abandon/close_only) fall through this guard (no
        # next dispatch to consume the bump).
        if escalate_model and action in ('resume', 'restart') and harness is not None:
            bump = getattr(harness, 'pre_increment_routing_tier', None)
            if callable(bump):
                try:
                    bump(rec.task_id)
                except Exception as _e:  # noqa: BLE001 — best-effort telemetry
                    logger.warning(
                        'escalate_model tier bump failed for task %s '
                        '(non-fatal): %s',
                        rec.task_id, _e,
                    )
        return esc.to_dict()

    @mcp.tool()
    async def get_pending_escalations(
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
        Both filters can be combined.  NOTE: this lookup is PENDING-ONLY by
        design (it never scans the resolved/dismissed archive).  To ask "did
        ANY escalation ever exist for this task", use ``get_task_escalations``
        instead — an empty result here is not evidence of absence.

        *compact* — when True, each returned dict is projected to the
        triage-relevant field subset, plus this tool's computed
        ``pins_recovery`` (below).  The authoritative list is
        :data:`_COMPACT_ESCALATION_FIELDS` — read it there rather than trusting
        a prose copy, which is how this paragraph drifted before.  What is
        DROPPED: the heavy free-text/cluster fields ``detail``, ``options``,
        ``train_state``, ``workflow_state``, ``worktree`` and ``dedupe_*``, plus
        ``amendments``.  ``detail`` is the unbounded free-text field compact mode
        exists to keep out of a long-running watcher's context.

        Two L2-cluster fields ARE returned, because they are bounded by
        construction and load-bearing for triage (task 3997): ``root_cause``,
        the one-line dedup key ``promote_to_l2`` folds on, and ``member_ids``,
        the PROJECTION of the record's ``members`` list under a contract name
        (the raw ``members`` key stays dropped).  Together they make a drain
        SELF-SUFFICIENT: a rotating watcher rebuilds ``already_promoted`` as
        {root_cause of the pending L2s} u {their member_ids} across the returned
        rows, with NO session memory — previously that set could only be carried
        forward in-session, so a rotation re-promoted clusters it had already
        promoted.

        Use this from a long-running watcher to keep context small as the
        pending pile grows; fetch the full record for a specific id via
        ``get_escalation`` only when you are about to act on it.  Default False
        preserves the full-dict shape for existing callers.

        *pins_recovery* — each returned dict carries a computed
        ``pins_recovery`` list: ``[task_id]`` when THIS record is what stops
        that task from being recovered/redispatched, else ``[]``.  It is the
        conjunction of four things (spec S8): the task is ``in-progress`` or
        ``blocked`` (there is something to recover), no live claimant holds it
        (it is stranded, not running), the record lands in
        :attr:`~escalation.pins.PinReport.queue_handoff` (so an info record and
        a dead L0 do NOT pin — derived from the classifier, never from
        ``bool(open_escalations)``), and the task's status could be read.

        The key is **OMITTED entirely** — never emitted as ``[]`` — when the
        annotation cannot be computed: no harness/scheduler wired in, the
        status read failed or raised, or that record's task is missing from
        the status map.  A false ``[]`` reads as "nothing pins this task",
        which is the exact collapse (esc-3163) that
        :attr:`~escalation.pins.PinReport.store_unavailable` exists to
        prevent, so callers must treat an absent key as UNKNOWN and render
        nothing rather than "not pinning".
        """
        if task_id:
            escalations = queue.get_by_task(task_id, status='pending', level=level)
        else:
            escalations = queue.get_pending()
            if level is not None:
                escalations = [e for e in escalations if e.level == level]

        dicts = [e.to_dict() for e in escalations]
        try:
            await _annotate_pins_recovery(queue, harness, escalations, dicts, level=level)
        except Exception:
            # THE seam guard.  `harness` is duck-typed Any because this package
            # deliberately does not import orchestrator, so the annotation can
            # never fully trust its contract.  Stating "an annotation must never
            # fail the tool" once, here, makes it true BY CONSTRUCTION for every
            # line inside _annotate_pins_recovery — including ones added later —
            # instead of by enumerating which calls someone guessed might throw.
            # Records already stamped keep their value; the rest keep the key
            # ABSENT, which is the contract's UNKNOWN, never a false [].
            # logger.exception (not .debug) so a real seam violation yields a
            # traceback rather than a one-line repr.
            logger.exception(
                'pins_recovery annotation failed for %d pending record(s); '
                'unstamped records report UNKNOWN (key absent)', len(dicts),
            )
        if compact:
            # _compact_escalation OMITS absent keys because pins_recovery is
            # deliberately absent when unknown — projecting it unconditionally
            # would KeyError on exactly the degraded path the omission contract
            # exists for.  The same helper also owns the members -> member_ids
            # rename, so both compact tools share one projection (task 3997).
            return [_compact_escalation(d, _COMPACT_PENDING_FIELDS) for d in dicts]
        return dicts

    @mcp.tool()
    def get_task_escalations(
        task_id: str,
        status: str | None = None,
        level: int | None = None,
        agent_role: str | None = None,
        compact: bool = False,
    ) -> list[dict[str, Any]]:
        """List EVERY escalation ever filed for a task — ARCHIVE-INCLUSIVE by default.

        This is the archive-inclusive counterpart to
        ``get_pending_escalations(task_id=...)``.  When an escalation is
        resolved or dismissed its file is MOVED out of the queue root into
        ``data/escalations/archive/<date>/``, where the pending-only lookup
        cannot see it — by design.  This tool scans the queue root PLUS that
        archive, so a human-resolved record still shows up here.

        **Evidence-of-absence contract (read this before asserting a gap).**
        An empty ``get_pending_escalations(task_id=...)`` result is NOT
        evidence that an escalation record never existed — it is the EXPECTED
        result once a human has resolved the record.  An empty
        ``get_task_escalations(task_id=...)`` result is evidence of absence
        **for the queue THIS server is backed by, and for nothing else.**
        ``create_server`` backs more than one queue (the orchestrator queue and
        the reconciliation queue are separate stores), so a caller connected to
        one of them learns nothing about records in the other: confirm which
        queue you are talking to before reading any [] as a gap.  Auditing a
        ``done`` ``task_kind='deterministic'`` gate task that has
        ``metadata.gate_escalated_at`` set?  Those records live in the
        ORCHESTRATOR queue — call this tool against THAT server before emitting
        any finding, flag, memory or remediation task claiming the escalation
        record was never written.  (Reconciliation stages are denied this tool
        outright — see DISALLOW_ESCALATION_READS in fused-memory's
        reconciliation/cli_stage_runner.py — precisely because their connection
        is to the other store.)

        *status* — ``None`` (default) returns records in every state,
        scanning root + archive.  ``'pending'`` short-circuits to the
        root-only fast path.  Any other value (``'resolved'``,
        ``'dismissed'``, …) filters across both tiers.

        *level* — 0 = L0 (agent→steward), 1 = L1 (steward/workflow→
        auto-watcher), 2 = L2 (auto-watcher→human).  ``None`` = no filter.

        *agent_role* — restricts to escalations filed by that exact role
        (e.g. ``'deterministic'`` for deterministic-gate records).
        ``None`` = no filter.

        *compact* — when True, each dict is projected to the same
        triage-relevant field subset ``get_pending_escalations(compact=True)``
        returns, so the two task-scoped lookups are shape-compatible.

        No connection-capability gate applies: this is a read-only lookup,
        mirroring ``get_escalation``.
        """
        escalations = queue.get_by_task(
            task_id, status=status, level=level, agent_role=agent_role,
        )
        if compact:
            return [
                _compact_escalation(d, _COMPACT_ESCALATION_FIELDS)
                for d in (e.to_dict() for e in escalations)
            ]
        return [e.to_dict() for e in escalations]

    @mcp.tool()
    def get_task_escalation_history(
        task_id: str,
        level: int | None = None,
    ) -> dict[str, Any]:
        """Every escalation ever filed for a task, as a self-describing envelope.

        **ARCHIVE-INCLUSIVE BY CONSTRUCTION.** Returns pending AND resolved/
        dismissed records that have been moved into ``data/escalations/
        archive/<date>/``. This is the tool that answers "was an escalation
        EVER filed for this task?".

        Contrast with ``get_pending_escalations(task_id=...)``: that lookup
        is pending-only by design and returns ``[]`` for a task whose only
        escalation was resolved. That ``[]`` means "none currently OPEN",
        never "none ever filed" — do not read it as evidence of absence.

        Contrast with the sibling ``get_task_escalations``: both scan the
        same underlying store (this tool delegates to it directly), but
        that tool is the general FILTERABLE form — it accepts ``status``,
        ``agent_role`` and ``compact``, and returns a bare list. Reach for
        THIS tool when the question is the unfiltered ever-filed one and
        you want a self-describing answer: it cannot be narrowed to
        pending (no ``status`` parameter exists — that is deliberate, not
        an oversight, so a caller can never silently recreate the
        false-absence trap this tool exists to remove), and the envelope
        echoes what was asked so a ``count: 0`` is attributable to the query
        that produced it rather than an anonymous ``[]`` — attributable, but
        only within one store: see the evidence-of-absence contract below.

        **Evidence-of-absence contract (read this before asserting a gap).**
        That attribution holds **for the queue THIS server is backed by, and
        for nothing else.** ``create_server`` backs more than one queue (the
        orchestrator queue and the reconciliation queue are separate stores),
        so a caller connected to one of them learns nothing about records in
        the other: confirm which queue you are talking to before treating
        ``count: 0`` as proof that no record was ever filed. Auditing a
        ``done`` ``task_kind='deterministic'`` gate task that has
        ``metadata.gate_escalated_at`` set? Those records live in the
        ORCHESTRATOR queue — query THAT server before emitting any finding,
        flag, memory or remediation task claiming the escalation record was
        never written. (Reconciliation stages are denied this tool outright
        — see DISALLOW_ESCALATION_READS in fused-memory's
        reconciliation/cli_stage_runner.py — precisely because their
        connection is to the other store.)

        *level* — 0 = L0 (agent→steward), 1 = L1 (steward/workflow→
        auto-watcher), 2 = L2 (auto-watcher→human). ``None`` = no filter.
        Mirrors ``get_pending_escalations``'s ``level`` argument.

        *level_filter* (in the response) echoes the ``level`` argument back
        so a caller can never misread a filtered ``count == 0`` as "no
        escalation was ever filed for this task" — that confusion is the
        exact failure mode this tool exists to remove.

        Inherits ``queue.get_by_task``'s documented dedup and cross-tier
        pre-scan WARNING behaviour (queue.py:396-430) via the sibling.

        A task with no matching records yields an empty envelope
        (``count == 0``), not an ``{'error': ...}`` dict — deliberately
        unlike the adjacent ``get_escalation``: this is a scan, not an id
        lookup, so zero matches is a valid answer rather than a failure.
        """
        escalations = get_task_escalations(task_id, level=level)
        return {
            'task_id': task_id,
            'count': len(escalations),
            'level_filter': level,
            'escalations': escalations,
        }

    @mcp.tool()
    def get_escalation(
        escalation_id: str,
    ) -> dict[str, Any]:
        """Get a single escalation by ID."""
        esc = queue.get(escalation_id)
        if esc is None:
            return {'error': f'Escalation {escalation_id} not found'}
        return esc.to_dict()

    @mcp.tool()
    def stamp_triage(
        escalation_id: str,
        triaged_by: str | None = None,
        triage_note: str = '',
    ) -> dict[str, Any]:
        """Stamp a triage-ack ANNOTATION on a pending escalation.

        This is deliberately NOT gated by the connection-capability level
        check (contrast ``resolve_issue``): triage is metadata, not a state
        transition, so a {0,1}-level-capped connection (e.g. the auto-watcher)
        can annotate a pending L2 it is still forbidden to resolve. Gating
        this tool would defeat its purpose — the auto-watcher must be able to
        record that it assessed a pending L2 so future rotations skip
        re-deriving the same disposition every rotation.

        Delegates to ``queue.stamp_triage()``, which stamps pending records
        only (root-only load; never resurrects an archived/resolved record).
        Does not change ``status``, ``level``, or ``updated_at``.

        *triaged_by* attribution: when the connection sends an
        X-Escalation-Identity header, that identity overrides the *triaged_by*
        arg — mirroring ``resolve_issue``'s non-spoofable ``resolved_by``
        override (server.py's identity gate). This is attribution ONLY, never
        a deny path, so the tool stays ungated: reading the identity header
        must never turn an annotation into a level denial.

        Returns the updated record as a full dict on success, or
        ``{'error': ...}`` when *escalation_id* is not found or not pending.
        """
        identity = get_http_headers().get(_IDENTITY_HEADER)
        if identity is not None:
            triaged_by = identity
        esc = queue.stamp_triage(escalation_id, triaged_by=triaged_by, triage_note=triage_note)
        if esc is None:
            return {'error': f'Escalation {escalation_id} not found or not pending'}
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
        severity: str | None = None,
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
        outcomes: ``'created'`` for a new L2, ``'updated'`` for an append.  An
        append RAISES the existing L2's severity when the incoming members (or
        an explicit *severity*) justify it, and never lowers it — an L2's
        severity is monotonically non-decreasing after mint, so a record cannot
        be quieted out from under a human already looking at it.  An append also
        APPENDS this call's *root_cause*/*evidence*/*options*/*summary* to the
        existing L2's ``amendments`` list rather than discarding them (task
        3997): the record's OWN framing stays immutable, but the framing a fold
        carried in is no longer lost.  That list is bounded — oldest-shed at
        ``queue._MAX_AMENDMENTS``, with every drop counted in the record's
        ``amendments_truncated``.

        **Members stay at L1**: the member L1 escalations are referenced but
        NOT promoted; they remain pending at L1 until the L2 is resolved.

        **Bypasses chokepoint**: this tool calls ``queue.submit()`` directly
        (create path) or ``queue.add_members_to_l2()`` (update path).  The
        terminal-task auto-resolve gate and severity→level=2 gate in
        ``_chokepoint_or_submit`` are intentionally bypassed — L2 is set
        explicitly by this tool.  Because that severity→level gate is bypassed
        by design, nothing else reconciles an L2's severity with the records it
        clusters; the inherited default below is what does it.

        **Identity gate** (PRD task-status-authority C8/D7): the create side
        is gated by ``escalation.authority.PROMOTE_ALLOWED`` — a connection
        asserting an ``X-Escalation-Identity`` not in that set is denied
        (``{'error': ..., 'code': 'level_forbidden'}``, no L2 minted); a
        header-less connection (no identity asserted) is always allowed.

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
            Severity tag, decoupled from ``level=2`` — the tool sets
            ``level=2`` explicitly.  **Omit it (or pass ``None``) and the L2
            INHERITS ``max(member severities)``** — this is the correct default
            in the overwhelming majority of cases, and is what stops a cluster
            of purely-informational L1s from being born ``'blocking'`` and
            paging a human (task 3976).

            An EXPLICIT value overrides the derivation in BOTH directions at
            mint time.  Upward in particular stays fully available and is not
            discouraged: a cluster of individually-informational findings CAN
            be collectively blocking, and a caller whose RCA concluded that
            should say so explicitly.  (Post-mint the update path applies a
            monotonic floor and will not accept a demotion — see
            **Root-cause dedup**.)

            The derivation ranges over ``models.KNOWN_SEVERITIES`` ONLY: a
            member that does not resolve, or that resolves carrying an
            out-of-vocabulary severity (nothing validates a record's severity
            on write), contributes nothing and is named at WARNING.  A
            partially usable set derives from the usable subset, so the filed
            severity is always a member of the vocabulary.

            When NO member yields a usable severity the two paths fail safe in
            DIFFERENT directions.  CREATE must pick something, so it fails safe
            UP to ``'blocking'``.  UPDATE deliberately does not: the existing
            L2 already carries a severity derived from real members, so it is
            left untouched (no floor, no ``updated_at`` bump) rather than being
            inflated on nothing more than a typo'd or momentarily unreadable
            member id.

            An explicit value must be one of ``models.KNOWN_SEVERITIES``;
            unknown values return ``{'error': ...}`` (mirrors
            ``escalate_blocker`` validation) and mint nothing.

            **An inherited ``'info'`` L2 is deliberately NON_PINNING.**  Before
            task 3976 no producer could mint an L2 below ``'blocking'``, so
            every L2 classified ``QUEUE_HANDOFF`` in ``escalation.pins``.  Link
            1 there short-circuits on ``severity == 'info'`` BEFORE the
            ``level != 0`` link — "an info record never pins, at any level" —
            so an inherited-info L2 no longer vetoes its subject task's
            ``done`` flip.  That is the INTENDED semantics and was considered
            here, not an oversight: the members it clusters were themselves
            non-pinning, and a record that does not merit a human's attention
            must not hold a task open waiting for one.  An L2 that genuinely
            should pin is one whose members are genuinely non-info, or one the
            caller filed with an explicit upward *severity*.

        Response shapes
        ---------------
        Create (new L2)::

            {'id': <new_id>, 'status': 'created', 'members': [<member_ids>],
             'severity': <severity_filed>}

        Update (existing pending L2 with same root_cause)::

            {'id': <existing_id>, 'status': 'updated', 'members': [<all_members>],
             'severity': <severity_after_floor>,
             'amendment_recorded': <bool>, 'amendments': <int>}

        ``amendment_recorded`` is True when THIS call's framing was appended.
        It is derived from what actually moved on the record — the amendment
        count growing, OR the truncation counter growing (an append that hits
        the cap sheds an entry, so the length stays put) — never from asserting
        that a write happened: a framing-free or framing-identical re-promote
        moves neither and correctly reports False.  ``amendments`` is the
        resulting list length, which saturates at ``queue._MAX_AMENDMENTS``.

        ``severity`` reports what was ACTUALLY filed, which for a caller that
        omitted the argument is how the inherited value becomes visible — and
        on the update path is the post-floor value, not the argument.

        Error::

            {'error': '<reason>'}
        """
        # Identity gate (PRD task-status-authority C8/D7 row C4) — checked
        # FIRST, before any validation or queue mutation, so a disallowed
        # caller mints nothing. Header-less (identity is None) stays
        # allowed, unchanged.
        identity = get_http_headers().get(_IDENTITY_HEADER)
        if identity is not None and identity not in PROMOTE_ALLOWED:
            return {
                'error': f'identity {identity!r} is not permitted to mint L2 escalations',
                'code': 'level_forbidden',
            }

        # Validate required non-empty fields
        if not member_ids:
            return {'error': 'member_ids must be a non-empty list'}
        if not root_cause.strip():
            return {'error': 'root_cause must be a non-empty string'}
        if severity is not None and severity not in KNOWN_SEVERITIES:
            return {
                'error': (
                    f'invalid severity {severity!r}; '
                    f'expected one of {sorted(KNOWN_SEVERITIES)}'
                ),
            }

        # Validate FIRST, derive second — an invalid explicit severity must mint
        # nothing and must never be reachable past the derive branch.  Derived
        # from the RAW member_ids: the fold is order-independent by
        # construction, and deduplicating the id list is a storage concern.
        #
        # `derived is None` means the members said nothing usable (no id
        # resolved, or every resolved member carried an out-of-vocabulary
        # severity).  The two paths below fail safe in DIFFERENT directions,
        # which is why the helper reports the fact instead of picking one.
        derived = (
            None if severity is not None else _derive_l2_severity(queue, member_ids)
        )

        # CREATE must land on some severity, so an underivable set fails safe
        # UP to 'blocking' — unchanged from before task 3976.
        effective_severity = (
            severity
            if severity is not None
            else (derived if derived is not None else 'blocking')
        )

        # UPDATE must NOT fail up: the existing L2 already carries a severity
        # derived from its real members, so an underivable set has nothing to
        # contribute and leaves the record (and its updated_at) alone.  Failing
        # up here would re-inflate a correctly-inherited info L2 to blocking on
        # nothing more than a typo'd or momentarily unreadable member id —
        # exactly the inflation this task removes.
        severity_floor = severity if severity is not None else derived

        # Dedup check: look for an existing pending L2 with the same root_cause.
        existing_id = queue.find_pending_l2_by_root_cause(root_cause)
        if existing_id is not None:
            # severity_floor is the caller's explicit value, or max(member
            # severities) over the ids in THIS call — exactly the floor the
            # incoming members justify — or None when they justify none, in
            # which case add_members_to_l2 leaves the severity untouched.
            # Upward-only inside add_members_to_l2, so an append can never
            # quiet an existing L2.
            # Pre-call amendment count, so `amendment_recorded` reports what
            # ACTUALLY grew rather than asserting a write happened.  A record
            # that could not be read (None) reads as 0, which can only
            # under-claim.
            before = queue.get(existing_id)
            amendments_before = len(before.amendments) if before is not None else 0
            # Same pre-call read serves the INV-4 storm escape below: a GROWN
            # truncation counter is the event worth counting, and reading it
            # from the record means no second source of truth.
            truncated_before = before.amendments_truncated if before is not None else 0
            updated = queue.add_members_to_l2(
                existing_id,
                list(dict.fromkeys(member_ids)),
                severity_floor=severity_floor,
                # The framing this promote carried in is APPENDED to the L2's
                # `amendments` rather than discarded (task 3997, C2).  The
                # record's OWN root_cause/detail/options/summary are untouched.
                root_cause=root_cause,
                evidence=evidence,
                options=list(options),
                summary=summary,
                agent_role=agent_role,
            )
            if updated is not None:
                # A TRUNCATING append leaves the list sitting AT the cap, so a
                # grown length alone under-reports: once `amendments` holds
                # _MAX_AMENDMENTS entries it can never grow again, and every
                # subsequent fold would falsely report its framing as dropped.
                # A grown truncation counter is the other observable half —
                # add_members_to_l2 only ever trims immediately after an append
                # — so the OR of the two is exact.  A framing-identical
                # re-promote appends nothing and moves neither, so it still
                # correctly reports False.
                truncated = updated.amendments_truncated > truncated_before
                amendment_recorded = (
                    len(updated.amendments) > amendments_before or truncated
                )
                # INV-4: repeated truncation gets a HEARER, not just a WARNING.
                # Purely additive — _report_amendment_truncation_storm never
                # raises, so a failed report can never fail this fold.
                if truncated:
                    _report_amendment_truncation_storm(existing_id, task_id)
                return {
                    'id': existing_id,
                    'status': 'updated',
                    'members': updated.members,
                    # Read off the returned Escalation, so this is the
                    # POST-floor value rather than the argument.
                    'severity': updated.severity,
                    # Report the preservation, so a caller LEARNS its framing
                    # landed instead of having to re-read the record to find out.
                    'amendment_recorded': amendment_recorded,
                    'amendments': len(updated.amendments),
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
            severity=effective_severity,
            category=category,
            summary=summary,
            detail=evidence,
            level=2,
            members=list(dict.fromkeys(member_ids)),
            root_cause=root_cause.strip(),
            options=list(options),
        )
        queue.submit(esc)
        return {
            'id': esc.id,
            'status': 'created',
            'members': esc.members,
            'severity': esc.severity,
        }

    # --- Merge queue tools ---

    @mcp.tool()
    async def merge_request(
        task_id: str,
        branch: str,
        worktree: str,
        description: str = '',
        wait_secs: int = 0,
        verified_green: bool = False,
        retry_failed_only: bool = False,
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

        *verified_green* — set True to vouch that the caller already ran a
        passing verification of this branch against its own base before
        submitting.  Only the caller can know this; the server cannot infer
        it.  When True (and an event store is wired), emits a
        ``workflow_verify`` event so the merge-skew classifier can attribute
        a later skewed merge failure to ``INTEGRATION_SKEW`` instead of
        degrading to ``INDETERMINATE``.  Default ``False`` — no attribution,
        emits nothing (mirrors the orchestrator's own emission at the
        VERIFY→REVIEW transition for its own submissions).

        *retry_failed_only* — set True to vouch that this request's post-merge
        verify retries should re-run only the previously-failed tests instead
        of a full re-verify (PRD docs/prds/verify-retry-failed-only.md task
        D1, following the same caller-vouched-bool shape as *verified_green*
        above).  Threaded onto the ``MergeRequest`` the worker dequeues so it
        is visible on ``req`` inside the worker's retry path
        (``_run_post_merge_verify``).  Default ``False`` is a strict no-op —
        the retry-set primitive that consumes this flag ships separately
        (reify, PRD task D2), so today every value leaves behavior unchanged.

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
          position, queue_depth, eta_seconds, inflight_task_id, source,
          inflight_request_id, poll_by, pollable}``.  Branch is
          already in-flight; request_id is the *existing* entry's id (D8), not
          the submitting call's id.  ``inflight_task_id`` is the authoritative
          poll handle (merge_status accepts task_id per D10).
          ``source`` names which coalesce arm attached (``'registry'`` /
          ``'worktree'``).  ``inflight_request_id`` is the in-flight entry's id
          when one is known, else None.  ``poll_by`` (task 3148) names WHICH
          handle to poll, so the caller never has to re-derive the remedy:

          * ``'request_id'`` — poll ``merge_status(request_id)``; the returned
            ``request_id`` IS the in-flight entry's id.
          * ``'task_id'`` — no in-flight request_id is known (e.g. a legacy
            registry entry predating the field), so the returned ``request_id``
            fell back to the *submitting* call's id and is NOT a handle; poll
            ``merge_status(task_id=inflight_task_id)`` instead (D10).
          * ``'branch'`` — NEITHER handle is known (the disk-scan/worktree
            arm: a foreign or pre-restart merger owns the tree, so there is no
            in-process entry, no retention alias, and no waiter).  The returned
            ``request_id`` was never enqueued and ``merge_status`` on it
            resolves ``'unknown'`` — poll by branch / ``get_merge_queue``, and
            do NOT read a first-tick ``'unknown'`` as a terminal failure.

          ``pollable`` is the boolean shorthand ``poll_by != 'branch'`` — i.e.
          "this response carries a handle naming the in-flight merge".  Both
          caller-side docs (skills/unblock/SKILL.md step 7;
          skills/merge-queue/SKILL.md "Poll for completion" + §5) consume
          this disclosure — picking the poll handle and gating
          ``merge_cancel`` off ``poll_by``/``pollable`` rather than assuming
          submit-then-poll-by-request_id and an unconditional
          ``merge_cancel`` on the attached request_id.
        - Duplicate-in-verify reject (C3/D3): ``{error, code='duplicate_in_verify',
          existing_mr, existing_sha, verify_age_secs, hint='merge_cancel then
          resubmit'}``.  Returned when a *newer* SHA for the branch is submitted
          while its earlier SHA is already IN VERIFY — the gate refuses to
          supersede a live verify.  ``existing_mr``/``existing_sha`` identify the
          in-flight entry's request_id/tip (D8); ``verify_age_secs`` is how long
          that verify has been running.  Cancel it (merge_cancel) then resubmit.
        - Already merged: ``{status='already_merged', commit, reason='',
          conflict_details='', push_status=None}``.  Either the branch tip is
          already an ancestor of main AND the branch is not degenerate — i.e.
          it advanced past its recorded ``branch_base_sha``; a zero-commit
          branch is parked at an OLD main commit, which satisfies ancestry
          while carrying none of the task's work (task 3103)
          (fast-path — no enqueue, no request_id)
          or the worker detected the branch was already merged via merge marker
          (worker-path — also carries request_id and a None commit from
          outcome.merge_sha).  The degeneracy guard REDIRECTS a degenerate
          branch from the fast path to the worker rather than eliminating it:
          the worker reaches ``already_merged`` by its own ancestry
          short-circuit, so this status is still reachable for that shape — but
          it arrives with ``commit=None`` and a request_id instead of the
          parked foreign SHA.  All keys are present in both paths; callers
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
        from orchestrator.landing_evidence import (  # type: ignore[reportMissingImports]
            branch_is_degenerate,
        )
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            MergeOutcome,
            MergeRequest,
            QueuedBranch,
            WaiterRecord,
            coalesce_or_enqueue_merge_request,
            patch_content_contained,
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
            full_branch = canonical_queued_branch_name(branch, orch_config.git.branch_prefix)
            resolved_tip = await git_ops_for_scan.resolve_branch_sha(full_branch)
            # Shape converged with worker-path already_merged (suggestion 1).
            # request_id is absent: the fast-path short-circuits before any
            # MergeRequest entry is constructed (no entry → no id).  Built once
            # so the is_ancestor arm and the patch-id backstop arm return a
            # byte-identical response.
            already_merged_response = {
                'status': 'already_merged',
                'commit': resolved_tip,
                'reason': '',
                'conflict_details': '',
                'push_status': None,
            }
            # Degeneracy guard (task 3103).  Gates BOTH fast-path arms,
            # deliberately.  patch_content_contained runs
            # `git cherry <main> <tip>` and returns True when no `+` lines
            # appear, which for a ZERO-COMMIT branch is true VACUOUSLY (git
            # emits nothing), so gating only the is_ancestor arm would leak the
            # degenerate branch into the backstop and still return
            # {status:'already_merged', commit:<parked foreign SHA>} — a phantom
            # done on a WRITE path (the runbooks treat already_merged the same
            # as done and stamp done_provenance from result['commit']).  The
            # 2945 backstop's own logic is untouched: a rebased landing has
            # commits beyond its base, so it is non-degenerate by construction
            # and this guard never fires on it.
            #
            # Evaluated LAST in each arm, and memoized across the two (review
            # #1).  The guard's only power is to SUPPRESS an already_merged
            # return, so it is pure cost on the overwhelmingly common
            # submission where neither arm hits.  Hoisting it above the block
            # made every merge_request pay a scheduler.get_task round-trip —
            # a Taskmaster MCP dispatch with an internal timeout=15
            # (scheduler.py:2485) — on the submit path.  Ordering it after the
            # arm test is logically identical (`not degenerate and (A or B)`
            # ≡ `(A and not degenerate) or (B and not degenerate)`) and pays
            # that cost only when an arm is about to return.
            #
            # Fail-soft with its OWN try/except (merge_request's fast-path has
            # no enclosing fire-safe wrapper): a probe fault must never break
            # submission.
            #
            # What declining the fast path actually does.  NOT "the worker
            # re-detects already-merged, a redundant no-op merge" — the worker
            # performs no merge at all here.  merge_queue.py:5616 finds
            # effective_tip (the parked base) an ancestor of main;
            # _already_merged_is_genuine (:5455) resolves candidate_tip to that
            # same parked base and returns True at its FIRST ancestry check
            # (:5515); :5650 returns a terminal MergeOutcome('already_merged')
            # with merge_sha=None, which the terminal-outcome block below maps
            # to {'status':'already_merged', 'commit': None, 'request_id': ...}.
            # A degenerate branch is therefore REDIRECTED to the worker, not
            # eliminated, and this guard buys exactly three things:
            #   (i)   the submit-time response can no longer carry the parked
            #         foreign SHA as 'commit', which skills/unblock/SKILL.md
            #         stamps verbatim as done_provenance={"commit": ...} — a
            #         fabricated provenance record pointing at an unrelated
            #         task's commit;
            #   (ii)  the submission becomes an auditable queue record (a
            #         request_id and a merge_queued event) instead of a silent
            #         submit-time short-circuit;
            #   (iii) on the worker path 'commit' is None, which routes that
            #         same runbook clause to its exact-subject marker search —
            #         empty for a genuinely unmerged branch — and thence to a
            #         {"note": ...} provenance rather than a SHA.
            # Residual, stated plainly: 'status' is still 'already_merged',
            # which the runbooks treat as terminal success, so the phantom-done
            # hole is narrowed, not closed.  Follow-up
            # tkt_0RSHM98C6F78MW4J0SK3S29YZG; single fix point
            # merge_queue.py:5515.
            #
            # Why this task does NOT instead gate _already_merged_is_genuine on
            # branch_is_degenerate: that makes the worker fall THROUGH and
            # merge.  _classify_branch_presence (merge_queue.py:3939) returns
            # None because the ref is present, and `git merge --no-ff
            # <ancestor>` is a no-op ("Already up to date.", rc 0, HEAD
            # unchanged), so merge_to_main's `git rev-parse HEAD`
            # (git_ops.py:9306) reads back that unchanged main HEAD and reports
            # MergeResult(success=True, merge_commit=<unrelated main commit>).
            # That converts today's already_merged/None into a 'done' carrying
            # a real-looking foreign SHA — strictly worse — and burns a
            # head-of-line verify slot on a guaranteed no-op.  Long form lives
            # in the ticket above.
            _degenerate_verdict: bool | None = None

            async def _declined() -> bool:
                """True iff the branch is degenerate → decline the fast path.

                Memoized so the two arms share one scheduler round-trip and
                one verdict.  ``branch_tip_sha=resolved_tip`` hands the probe
                the tip the arm above just tested, so the degeneracy verdict
                and the ancestry/patch-id evidence it gates are computed
                against the SAME observed SHA (review #2) — a warm-lane
                reseed between two independent ref reads cannot split them.
                """
                nonlocal _degenerate_verdict
                if _degenerate_verdict is not None:
                    return _degenerate_verdict
                try:
                    # Derive the id from the branch, exactly as merge_status
                    # does (server.py `tid = full_branch.removeprefix(prefix)`),
                    # NOT from the caller-supplied task_id parameter (review
                    # #6).  merge_request takes task_id and branch as two
                    # independent parameters; keying the metadata off one and
                    # the tip off the other would compare task X's recorded
                    # branch_base_sha against task Y's branch tip, silently
                    # disabling the guard on any mismatched submission.  The
                    # branch is the single source of truth here because it is
                    # what resolved_tip was read from.
                    _degenerate_verdict = await branch_is_degenerate(
                        git_ops_for_scan, full_branch,
                        await _git_authority_task_metadata(
                            full_branch.removeprefix(orch_config.git.branch_prefix),
                            site='merge_request',
                        ),
                        branch_tip_sha=resolved_tip,
                    )
                except Exception:
                    logger.warning(
                        'merge_request: degeneracy probe failed for %s — proceeding '
                        'as non-degenerate',
                        full_branch, exc_info=True,
                    )
                    _degenerate_verdict = False
                return _degenerate_verdict

            if (resolved_tip is not None and await git_ops_for_scan.is_ancestor(
                resolved_tip, orch_config.git.main_branch
            ) and not await _declined()):
                return already_merged_response
            # Rebased-landing backstop (task 2945): a branch whose content
            # landed on main as a rebased/cherry-picked commit is NOT a literal
            # ancestor of main (is_ancestor misses above), yet its work is
            # fully present by patch-id.  patch_content_contained (`git cherry`)
            # catches this dominant landing mode and kills the guaranteed-no-op
            # resubmission at the door — NO enqueue, NO merge_queued event, same
            # already_merged shape as the ancestor arm.  Fail-open: any git
            # error → False → falls through to the normal coalesce/enqueue path.
            if (resolved_tip is not None and await patch_content_contained(
                resolved_tip, orch_config.git.main_branch, git_ops_for_scan
            ) and not await _declined()):
                return already_merged_response

        # module_configs_or_empty normalises the post-1405 None sentinel (direct-
        # instantiation configs never call load_config, so _module_configs stays None).
        # See OrchestratorConfig.module_configs_or_empty (config.py) for details.
        module_configs = list(orch_config.module_configs_or_empty.values())
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        merge_req = MergeRequest(
            task_id=task_id,
            branch=QueuedBranch.parse(branch, orch_config.git.branch_prefix),
            worktree=Path(worktree),
            pre_rebased=False,
            task_files=None,
            module_configs=module_configs,
            config=orch_config,
            result=future,
            snapshot_tip=resolved_tip,
            retry_failed_only=retry_failed_only,
        )

        # Build a live_snapshot provider from the live worker handle so the
        # coalesce gate can reconcile its registry slot against the worker's
        # live snapshot.  This makes merge_request and get_merge_queue read
        # the same source of truth: if a slot's request_id is absent from
        # the snapshot (the request finalized abnormally without releasing
        # the slot), the gate reaps it and dispatches a fresh request instead
        # of silently attaching onto a dead id.
        # _nonblocking_state_response (below) resolves the same worker handle
        # for position/queue_depth via _get_merge_worker; keep both consistent.
        _live_merge_worker = _get_merge_worker(harness)
        live_snapshot = (
            _live_merge_worker.snapshot
            if _live_merge_worker is not None and hasattr(_live_merge_worker, 'snapshot')
            else None
        )

        # De-dup gate: consults the in-memory registry (and optionally the on-disk
        # _merge-* worktree scan via harness.git_ops) before enqueuing.  On coalesce
        # returns immediately with in_flight=True — no future await, no duplicate
        # enqueue.  On dispatch acquires the registry slot and awaits the future
        # exactly as the original enqueue_merge_request path.
        # Task 2411: mirror the orchestrator's own workflow_verify emission
        # (workflow.py:1724-1733) for non-orchestrator submission pathways
        # (/merge-queue, /unblock, /do) so the merge-skew classifier's I5
        # branch-green fact (merge_disposition._branch_pre_merge_verify_green,
        # keyed by task_id, reads only data['passed']) can source from these
        # too.  verified_green is a caller-supplied vouch — only the caller
        # (which just ran the verification) can know it happened; the server
        # cannot infer it.  base_sha is best-effort/informational only (the
        # classifier ignores it).  event_store is None for a standalone
        # server (no orchestrator wired) — guarded so this degrades to no
        # attribution instead of raising (fail-open, mirrors the git-error
        # None-degrade inside _resolve_dispatch_time_merge_base).
        if verified_green and event_store is not None:
            from orchestrator.event_store import (  # type: ignore[reportMissingImports]
                EventType,
            )
            from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
                _resolve_dispatch_time_merge_base,
            )
            base_sha = (
                await _resolve_dispatch_time_merge_base(
                    orch_config.project_root, orch_config.git.main_branch, resolved_tip,
                )
                if resolved_tip is not None
                else None
            )
            # SYNC CONTRACT (reviewer follow-up on step-4): this payload shape —
            # keys 'passed' / 'base_sha' / 'branch', task_id-keyed — MUST stay
            # byte-identical to the orchestrator's own emission at
            # workflow.py:1724-1733, since merge_disposition._branch_pre_merge_verify_green
            # reads both call sites as one logical event stream.  A shared
            # canonical constructor (e.g. an `emit_workflow_verify(...)` helper)
            # would remove this manual-sync risk, but its natural home is
            # orchestrator (workflow.py would need to call it too) — out of
            # this task's locked scope (escalation/server.py + tests +
            # skills/{merge-queue,unblock}/SKILL.md only).  Deferred as a
            # follow-up rather than expanding this task's file locks.
            event_store.emit(
                EventType.workflow_verify,
                task_id=task_id,
                data={
                    'passed': True,
                    'base_sha': base_sha,
                    'branch': canonical_queued_branch_name(branch, orch_config.git.branch_prefix),
                },
            )

        # classifier_git_ops: the same harness GitOps handle is reused here so
        # the recency check (resolve_attach_action) can classify the tip relation
        # between req.snapshot_tip and the in-flight entry's snapshot_tip.  When
        # git_ops_for_scan is None (standalone / tests without orchestrator) the
        # classifier is also None and the recency check is a no-op (back-compat).
        # retention: write-side counterpart of the Tier-2 reads in
        # _durable_terminal_state (merge_status) and the
        # retire_cancelled_merge_request call in merge_cancel — populates the
        # ring via enqueue_merge_request's _on_finalized callback (dispatch
        # arm) and registers a record_alias entry for coalesced ids.  Always
        # None today (no production path yet assigns
        # harness._terminal_retention — see task 3149), so this call keeps
        # behaving exactly as before until that ring is actually mounted.
        dispatch = await coalesce_or_enqueue_merge_request(
            merge_queue,
            merge_req,
            event_store,
            _registry,
            git_ops=git_ops_for_scan,
            live_snapshot=live_snapshot,
            classifier_git_ops=git_ops_for_scan,
            retention=_get_terminal_retention(harness),
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
            worker = _get_merge_worker(harness)
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

        if dispatch.rejected:
            # C3/D3: a newer SHA for this branch was submitted while its earlier
            # SHA is IN VERIFY.  The submit gate rejects it structurally
            # (duplicate_in_verify) rather than tearing down the live verify.
            # Envelope aligns with the server's existing {error, code}
            # convention; existing_mr/existing_sha carry the IN-FLIGHT entry's
            # request_id/snapshot_tip (D8) so the caller correlates with the
            # live verify, not the rejected submission.  The hint tells the
            # caller how to proceed: merge_cancel the in-flight entry, then
            # resubmit the newer tip.
            return {
                'error': (
                    f'a newer SHA for {branch} cannot be submitted while its '
                    'earlier SHA is in verify; merge_cancel then resubmit'
                ),
                'code': dispatch.reject_code,
                'existing_mr': dispatch.inflight_request_id,
                'existing_sha': dispatch.existing_sha,
                'verify_age_secs': dispatch.verify_age_secs,
                'hint': 'merge_cancel then resubmit',
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
            # task 3148: disclose WHICH handle this attach can be polled by,
            # rather than leaving the caller to re-derive it from prose.  The
            # disk-scan (source='worktree') arm registers no retention alias and
            # no waiter — the _waiters registration below is AFTER this early
            # return — so on that arm `request_id` is the submitting request's
            # own never-enqueued id and merge_status on it resolves 'unknown'.
            # Derived from the handles actually present rather than from
            # `source`, so this stays correct if a future arm gains or loses
            # alias registration, and so the legacy registry entry (no
            # request_id, but a perfectly good task_id) is routed to the handle
            # it does have instead of being written off as unpollable.
            base['source'] = dispatch.source
            base['inflight_request_id'] = dispatch.inflight_request_id
            if dispatch.inflight_request_id is not None:
                # req_id_override above already put this id in base['request_id'].
                base['poll_by'] = 'request_id'
            elif dispatch.inflight_task_id is not None:
                # merge_status accepts task_id (D10); base['request_id'] here is
                # the submitting call's own id and is NOT a handle.
                base['poll_by'] = 'task_id'
            else:
                base['poll_by'] = 'branch'
            # Shorthand: "some handle naming the in-flight merge is present".
            base['pollable'] = base['poll_by'] != 'branch'
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
            # ε: branch/task_id let merge_cancel drive per-branch retirement
            # (slot release + worktree reap) from a request_id alone.
            branch=merge_req.branch.bare_id,
            task_id=merge_req.task_id,
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
        clamp = min(max(wait_secs, 0), _MAX_WAIT_SECS)
        try:
            outcome = await asyncio.wait_for(asyncio.shield(future), clamp)
        except TimeoutError:
            return _nonblocking_state_response('queued', merge_req)
        # Resolved within clamp → fall through to terminal outcome shape below.
        # 'commit' (outcome.merge_sha) is included for shape convergence with the
        # fast-path already_merged response.  It is None for most statuses and for
        # the worker-produced already_merged case (neither the merge-marker path
        # nor the ancestry short-circuit that a degenerate branch takes returns a
        # SHA); it is non-None for 'done' and 'done_wip_recovery' where main was
        # advanced.
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
        if outcome.superseded_by is not None:
            result['superseded_by'] = outcome.superseded_by
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

        Once no workflow slot is live, if the task is still ``in-progress``
        (the typical /unblock shape: an escalated task whose agent was paused)
        it is parked as ``blocked``.  This applies **whether or not a slot was
        registered when the call started**: an orphaned ``in-progress`` task
        whose lane was already reaped — ``was_active`` False — is parked too,
        and is in fact the shape most in need of the hold, since nothing else
        is stopping the scheduler from re-dispatching it.  ``blocked`` is the
        reaper-immune holding state — it stops the orchestrator from
        re-dispatching the task AND protects the worktree from the stranded-
        in-progress reconciliation sweep while the human finishes the work.
        From ``blocked`` the final ``set_task_status('done')`` after merge is
        the normal blocked→done transition.

        Returns:
            ``{released, was_active, slot_cleared, parked}``
            - ``was_active``: True if a workflow slot was registered when
              the call started.  False does NOT mean "nothing happened" — the
              park below still applies.
            - ``released``: True if ``cancel_workflow`` accepted the request.
            - ``slot_cleared``: True if no slot is live by the end of the call
              (either it finished within ``timeout_secs``, or there was never
              one to begin with).  False also covers a slot the scheduler
              dispatched while this call was in flight.
            - ``parked``: the status the task was parked into (``'blocked'``),
              or ``None`` if no park occurred.  ``None`` means exactly one of:
              a slot was still active at the deadline, the task was not at
              ``in-progress`` (already terminal, parked elsewhere, or the
              status read failed), or the scheduler dispatched a fresh
              workflow while the status was being read (``slot_cleared`` comes
              back False in that case too).  Callers should CONFIRM the park by
              reading this field rather than assuming it.
        """
        if harness is None:
            return {
                'released': False, 'was_active': False, 'slot_cleared': False,
                # Every return path must satisfy the documented shape — callers
                # are told to read `parked` unconditionally, so a standalone
                # server must hand back an explicit None, not a missing key.
                'parked': None,
                'error': 'No orchestrator harness wired in — running in standalone mode',
            }
        was_active = harness.is_workflow_active(task_id)
        released = harness.cancel_workflow(task_id)
        # No early-return on `not was_active`: an orphaned task (lane already
        # reaped, no slot registered, row still 'in-progress') is exactly the
        # shape that most needs the park below.  Falling through costs nothing
        # — with no slot the wait loop's condition is false on its first
        # evaluation, so it never sleeps, `slot_cleared` computes True and
        # `released` stays False: the same values the old early-return
        # hardcoded, minus the skipped park.
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
                if harness.is_workflow_active(task_id):
                    # Re-check liveness AFTER the status read.  `get_status` is
                    # an MCP round-trip that yields the event loop, and the
                    # Harness runs on that same loop, so the scheduler is free
                    # to dispatch task_id behind the guard above: _run_slot
                    # registers a cancel event (and calls
                    # scheduler.clear_workflow_cancel, which would wipe the
                    # grace stamp below).  The no-slot arm makes this
                    # materially more likely than before — a task with no slot
                    # is precisely the one the scheduler may pick up on its
                    # next tick.  Parking here would write 'blocked' out from
                    # under a live agent while still reporting
                    # slot_cleared/parked as though the caller owned the task,
                    # which is exactly how both /unblock skills read this
                    # result.  Report the live slot instead and park nothing.
                    slot_cleared = False
                else:
                    if not released:
                        # No slot existed, so Harness.cancel_workflow returned False on its
                        # `event is None` arm without reaching note_workflow_cancelled — this
                        # park would land with ZERO grace and the scheduler's
                        # _phase_redispatch_stranded_blocked phase would flip it back to
                        # 'pending' within one 15 s idle tick.  Stamp it here so the orphan
                        # park gets the same _RECONCILE_CANCEL_GRACE_S window a slot-cancel
                        # park gets.  Sync call — note_workflow_cancelled is a plain `def`.
                        # Guarded on `not released`: on the slot-cancel arm cancel_workflow
                        # already stamped, and re-stamping would re-anchor that window.
                        # Must precede the write below — the scheduler tick can observe the
                        # 'blocked' row the instant it is persisted.
                        #
                        # Known asymmetry, deliberately left alone (out of scope): the
                        # slot-cancel arm's stamp is anchored at cancel_workflow time and
                        # never re-anchored, so a slot that takes ~25 s to exit parks with
                        # only ~5 s of _RECONCILE_CANCEL_GRACE_S left.  Extending it here
                        # would silently change the slot-cancel path's timing.  The durable
                        # protection for a long human /unblock session is the
                        # pending-escalation gate in
                        # Scheduler._phase_redispatch_stranded_blocked, NOT this 30 s
                        # stamp — a caller that resolves the escalation before finishing
                        # the merge loses the hold on BOTH arms.
                        harness.scheduler.note_workflow_cancelled(task_id)
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
    async def halt_merge_queue(reason: str) -> dict[str, Any]:
        """Operator-initiated halt of the orchestrator merge queue.

        Pauses the merger (no new merge requests are taken) AND terminates the
        post-merge verify currently running, re-queuing that merge so it
        re-verifies once the queue is un-halted.  Reversed by unhalt_merge_queue.

        Operator-only: this tool appears in no agent role's allow-list, so
        autonomous orchestrator agents cannot call it.  The halt is in-memory and
        transient — a process restart clears it (a fresh process starts
        un-halted).  ``reason`` is required for audit.
        """
        if harness is None:
            return {
                'halted': False,
                'error': 'escalation server running standalone — no harness wired',
            }
        if not reason or not reason.strip():
            return {'halted': False, 'error': 'reason is required for audit'}
        return harness.halt_merge_queue(reason.strip())

    @mcp.tool()
    def get_merge_halt_status() -> dict[str, Any]:
        """Inspect the orchestrator merge queue's halt state."""
        if harness is None:
            return {'wired': False, 'error': 'escalation server running standalone'}
        return harness.get_merge_halt_status()

    @mcp.tool()
    def get_task_runtime_state() -> dict[str, Any]:
        """Live per-task runtime snapshot, projected to the shared wire contract.

        Delegates to ``harness.task_runtime_snapshot()`` (task alpha, task
        2634) and projects each duck-typed entry into
        ``shared.task_runtime_state.TaskRuntimeEntry`` — read as attributes
        (no static ``orchestrator`` import, matching ``merge_request``'s
        reverse-dep discipline) so this works against both the real
        ``orchestrator.task_runtime.TaskRuntimeState`` and a test stub alike.
        A per-task artifact read failure is carried through unmodified
        (``loops``/``attempts``/``phase``/``started`` stay ``None`` plus a
        non-empty ``error`` — never coerced to a fabricated honest-looking
        value). A task whose ``phase``/``lane_state`` falls outside the wire
        model's ``Literal`` vocabulary (see ``_project_task_runtime_entry``)
        degrades the same way — that one task reports an honest error entry
        instead of a ``ValidationError`` failing the whole snapshot.
        Standalone (no harness wired) returns the model's legible empty
        envelope, never raises. This server always emits ``offline: False``;
        the dashboard synthesizes ``True`` client-side when this server
        itself is unreachable.
        """
        if harness is None:
            return TaskRuntimeSnapshot().model_dump(mode='json')
        states = harness.task_runtime_snapshot()
        entries = [_project_task_runtime_entry(s) for s in states]
        return TaskRuntimeSnapshot(offline=False, tasks=entries).model_dump(mode='json')

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
        worker = _get_merge_worker(harness)
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

    @mcp.tool()
    async def halt_scheduler(reason: str) -> dict[str, Any]:
        """Operator-initiated halt of the orchestrator scheduler.

        Stops the scheduler from dispatching new tasks (acquire_next() returns
        None) and persists the pause to runs.db, so the halt survives a restart.

        Unlike an automatic park-stop / cost-ceiling / EWA trip, this does NOT
        file an auto-resumable scheduler-pause L1: notifying the operator of their
        own deliberate halt is noise, and an auto-watcher resolving that L1 would
        silently undo the halt.  Reversed by resume_scheduler.

        Operator-only: this tool appears in no agent role's allow-list, so
        autonomous orchestrator agents cannot call it.  ``reason`` is required for
        audit.  Returns ``{halted, was_paused, prior_reason, reason}`` (or an
        ``error`` when the server is running standalone with no harness wired).
        """
        if harness is None:
            return {
                'halted': False,
                'error': 'escalation server running standalone — no harness wired',
            }
        if not reason or not reason.strip():
            return {'halted': False, 'error': 'reason is required for audit'}
        return await harness.force_halt_scheduler(reason.strip())

    @mcp.tool()
    async def reload_config() -> dict[str, Any]:
        """Operator-initiated hot-reload of the orchestrator's config file.

        Operator-only: this tool appears in no agent role's allow-list, so
        autonomous orchestrator agents cannot call it (I8).

        Takes NO arguments — always re-reads the orchestrator process's own
        ORCH_CONFIG_PATH, so a reload can never retarget the orchestrator at a
        different project (extends the cross-project-execution safeguard).

        What actually hot-applies is limited to the green-tier config allowlist
        (plans/config-hot-reload-prd.md §Allowlist); fields outside it are
        reported under ``restart_required`` — the edit was accepted but takes no
        effect until the orchestrator restarts.  A truthy ``reloaded`` does NOT
        mean every change took effect: always inspect the returned ``applied``
        and ``restart_required`` dispositions rather than just the top-level
        flag (I6).

        Returns the harness's report verbatim: ``{reloaded, config_path,
        applied, restart_required, unchanged, error}`` (or a standalone ``error``
        when the server is running with no harness wired).
        """
        if harness is None:
            return {
                'reloaded': False,
                'error': 'escalation server running standalone — no harness wired',
            }
        return await harness.reload_config()

    # --- Interactive warm-worktree tools (PRD β / task 2011) ---

    @mcp.tool()
    async def claim_warm_worktree(
        slug: str, project_root: str, *, start_ref: str | None = None,
    ) -> dict[str, Any]:
        """Claim an isolated interactive warm-worktree (PRD β, task 2011).

        Thin closure over :meth:`GitOps.create_interactive_worktree` (task α,
        2010) for out-of-process interactive callers (/do, /warm, the
        integration gate) that cannot reach ``harness.git_ops`` directly.
        Mints a fresh worktree in the ``_iact-*`` band on branch
        ``task/<slug>``, CoW-seeding its build cache when possible.

        *project_root* is validated against this server's wired
        ``harness.git_ops.project_root`` (resolved-path equality) — it is a
        defensive guard, NOT a routing key: this server always serves exactly
        one project, so a mismatch means the caller is talking to the wrong
        escalation MCP endpoint.

        Returns ``{path, branch, warm, base_ref}`` on success. ``path`` is an
        absolute string (not a ``Path``). ``warm=False`` is a fail-soft
        SUCCESS — the CoW seed faulted but the worktree is still usable; it
        is never surfaced as an error.

        On failure returns ``{error, reason}`` instead of raising — callers
        should surface ``error`` and, for ``reason='interactive_worktree_limit'``,
        fall back to a cold worktree/clone rather than retrying immediately.
        ``reason`` is one of: ``'interactive_worktree_limit'`` (the _iact-*
        cap is reached — free a slot and retry), ``'invalid_slug'`` (slug
        fails the safe-charset validation), ``'git_failure'`` (start_ref/main
        failed to resolve, or ``git worktree add`` failed).  A standalone
        server (no harness wired) or a ``project_root`` mismatch returns
        ``{error}`` with no ``reason`` key.
        """
        if harness is None or getattr(harness, 'git_ops', None) is None:
            return {'error': 'escalation server running standalone — no harness wired'}
        mismatch = _require_matching_project_root(harness, project_root)
        if mismatch is not None:
            return {'error': mismatch}

        # Runtime-only reverse import: orchestrator depends on escalation, not
        # vice versa (mirrors merge_request's lazy orchestrator.merge_queue
        # import above) — unresolvable in escalation's standalone typecheck
        # env, hence the suppression.
        from orchestrator.git_ops import (  # type: ignore[reportMissingImports]
            InteractiveWorktreeLimitError,
        )

        try:
            info = await harness.git_ops.create_interactive_worktree(
                slug, start_ref=start_ref,
            )
        except InteractiveWorktreeLimitError as e:
            return {'error': str(e), 'reason': 'interactive_worktree_limit'}
        except ValueError as e:
            return {'error': str(e), 'reason': 'invalid_slug'}
        except RuntimeError as e:
            return {'error': str(e), 'reason': 'git_failure'}

        return {
            'path': str(info.path),
            'branch': info.branch,
            'warm': info.warm,
            'base_ref': info.base_ref,
        }

    @mcp.tool()
    async def release_warm_worktree(path_or_branch: str, project_root: str) -> dict[str, Any]:
        """Release an interactive warm-worktree claimed via claim_warm_worktree.

        *path_or_branch* accepts either shape claim_warm_worktree returned: an
        absolute worktree ``path``, or the ``task/<slug>`` ``branch`` name
        (bare ``<slug>`` is also accepted). An existing directory, or any
        absolute path (even one already removed — the idempotent second-call
        case), is treated as path-mode; anything else is treated as
        branch-mode. In path-mode, the branch is read from the worktree's
        ``.task/interactive.json`` stamp when present and parseable, else
        derived from the path's basename — a truncated/corrupt stamp never
        raises, it just falls back to the basename derivation.

        *project_root* is validated exactly like claim_warm_worktree's guard.

        Runs ``git worktree remove --force`` + ``git worktree prune``
        directly (task α added no removal primitive, and
        ``GitOps.release_warm_lane`` is unusable here — it mutates
        WarmLanePool, violating isolation invariant I1: interactive
        worktrees never touch the dispatch/speculation pools).

        Returns ``{removed, path, branch, branch_pruned}``, plus an optional
        ``detail`` key holding the ``git worktree remove`` stderr whenever
        ``removed`` is False and git reported something — lets a caller
        distinguish an idempotent already-gone target from a genuine removal
        failure (locked worktree, permission, ...).  ``removed`` is False
        (not an error) when the target was already gone — idempotent, safe
        to call from a session-end hook after the δ reaper already cleaned
        up.  A standalone server or a ``project_root`` mismatch returns
        ``{error}`` instead, and removes nothing.

        After removal, ``branch`` is deleted (``git branch -D``) — and
        ``branch_pruned`` set True — only when its tip is an ancestor of
        (already merged into) ``main``; an unmerged branch is left in place
        so its commits are never lost.  ``branch_pruned`` is False whenever
        the branch is unmerged, already gone, or removal itself did nothing.
        """
        if harness is None or getattr(harness, 'git_ops', None) is None:
            return {'error': 'escalation server running standalone — no harness wired'}
        mismatch = _require_matching_project_root(harness, project_root)
        if mismatch is not None:
            return {'error': mismatch}

        gops = harness.git_ops
        candidate = Path(path_or_branch)
        # Path-mode covers both a still-live worktree directory AND an
        # already-removed one referenced by its (necessarily absolute)
        # claimed path — the idempotent second-release case. Keying off
        # shape (absolute / exists) rather than is_dir() alone keeps the
        # returned path/branch meaningful instead of silently falling
        # through to branch-mode and treating the whole path string as a
        # branch slug.
        if candidate.is_absolute() or candidate.exists():
            path = candidate.resolve()
            branch = None
            stamp_path = path / '.task' / 'interactive.json'
            if stamp_path.exists():
                try:
                    branch = json.loads(stamp_path.read_text())['branch']
                except (json.JSONDecodeError, KeyError, OSError):
                    # Truncated/corrupt stamp (e.g. a crash mid-write) — fall
                    # back to the basename derivation below rather than
                    # raising out of a documented never-raise verb.
                    branch = None
            if branch is None:
                branch = (
                    f'{gops.config.branch_prefix}'
                    f'{path.name.removeprefix(gops.config.iact_prefix)}'
                )
        else:
            branch = path_or_branch
            branch = canonical_queued_branch_name(branch, gops.config.branch_prefix)
            slug = branch.removeprefix(gops.config.branch_prefix)
            path = gops.worktree_base / f'{gops.config.iact_prefix}{slug}'

        # Runtime-only reverse import — mirrors claim_warm_worktree above.
        from orchestrator.git_ops import _run  # type: ignore[reportMissingImports]

        rc, _, remove_stderr = await _run(
            ['git', 'worktree', 'remove', '--force', str(path)], cwd=gops.project_root,
        )
        await _run(['git', 'worktree', 'prune'], cwd=gops.project_root)
        removed = rc == 0

        # Prune-if-merged: only delete branch once its tip is confirmed an
        # ancestor of main — an unmerged branch's commits must never be lost.
        branch_pruned = False
        sha = await gops.resolve_branch_sha(branch)
        if sha is not None and await gops.is_ancestor(sha, gops.config.main_branch):
            rc_del, _, _ = await _run(['git', 'branch', '-D', branch], cwd=gops.project_root)
            branch_pruned = rc_del == 0

        result: dict[str, Any] = {
            'removed': removed,
            'path': str(path),
            'branch': branch,
            'branch_pruned': branch_pruned,
        }
        if not removed and remove_stderr.strip():
            # Present only on removal failure — its text is what lets a
            # caller tell an idempotent already-gone target apart from a
            # genuine failure (locked worktree, permission, ...); the
            # removed=False contract itself is unchanged.
            result['detail'] = remove_stderr.strip()
        return result

    # ── merge_status — read-only lifecycle probe (PRD α3 / task 1630) ──────────

    _MERGE_STATUS_UNKNOWN_HINT = 'check git log main'

    # Optional fields that _durable_terminal_state threads into the meta dict
    # when present and non-None.  Add future terminal-metadata fields here so
    # every tier stays in sync automatically.
    _OPTIONAL_TERMINAL_META_FIELDS: tuple[str, ...] = ('superseded_by', 'reason')

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
        if raw == 'superseded':
            return 'superseded'
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
        # Tier 2: retention ring (request_id > branch > task_id precedence).
        # request_id lookup also resolves aliases registered via record_alias()
        # (e.g. coalesced ids that never get their own terminal record).
        # finished_at is stored as epoch float; normalise to ISO-8601 so the
        # same logical merge returns the same type regardless of which tier serves it.
        ring = _get_terminal_retention(harness)
        if ring is not None:
            if request_id is not None:
                rec = ring.get(request_id)
            elif branch is not None:
                rec = ring.get_by_branch(branch)
            elif task_id is not None:
                rec = ring.get_by_task(task_id)
            else:
                rec = None
            if rec is not None:
                meta: dict = {
                    'request_id': rec.request_id,
                    'outcome': rec.state,
                    'finished_at': _epoch_to_iso8601(rec.finished_at),
                }
                for _f in _OPTIONAL_TERMINAL_META_FIELDS:
                    _v = getattr(rec, _f, None)
                    if _v is not None:
                        meta[_f] = _v
                return _map_terminal_state(rec.state), meta

        # Tier 3: event store (supports all three lookup keys).
        if event_store is not None:
            row = event_store.latest_merge_finalized(
                request_id=request_id,
                branch=branch,
                task_id=task_id,
            )
            if row is not None:
                es_meta: dict = {
                    'request_id': row['request_id'],
                    'outcome': row['state'],
                    'finished_at': row['finished_at'],
                }
                for _f in _OPTIONAL_TERMINAL_META_FIELDS:
                    _v = row.get(_f)
                    if _v is not None:
                        es_meta[_f] = _v
                return _map_terminal_state(row['state']), es_meta

        return None

    def _found_on_main_response(request_id: str | None, merge_sha: str) -> dict[str, Any]:
        """Build the git-authority Tier-3.5 done/found_on_main response.

        ``merge_sha`` is a commit ON MAIN on both resolution paths, with one
        explicit exception stated below (task 3103):

        - **Live-branch path** (``is_ancestor`` hit): the citation commit
          discovered by ``validate_landing_evidence`` — a commit on main
          whose subject cites the task.
        - **Deleted-branch path** (``find_merge_marker`` hit): the
          merge-commit SHA found on main via ``git log``.

        Both are effect-present-checked against current main HEAD before
        being returned, so ``merge_sha`` is safe to record as provenance
        as-is.  (Before task 3103 the live-branch path returned the *branch
        tip*, which for a ``--no-ff`` merge is a distinct commit that is not
        on main's first-parent chain — callers were told to prefer the
        deleted-branch path's value.  That caveat no longer applies.)

        **The one exception — ``git.commit_citation_pattern == ''``.**  That
        is the documented per-project opt-out for projects with no citation
        convention (config.py; ``find_task_citation_commit`` honours it by
        returning None for everything, so running the gate would reject
        unconditionally and turn this tier into dead code).  On that setting
        the live-branch path skips the citation gate entirely and
        ``merge_sha`` is the raw BRANCH TIP, neither citation-discovered nor
        effect-present-checked — i.e. exactly the pre-3103 ``--no-ff`` wart,
        deliberately retained as the price of the opt-out (the degeneracy
        guard still applies).  Do not read the paragraphs above as
        unconditional: on such a project a caller stamping ``merge_sha`` as
        provenance is recording a branch tip, and a reverted landing is
        indistinguishable from a live one (review #4).  The opt-out is
        ``''`` only; ``None`` means "use the built-in default pattern" and
        keeps the full guarantee.  Both SKILL.md runbooks carry the same
        exception.
        """
        return {
            'state': 'done',
            'request_id': request_id,
            'generation': 1,
            'kind': 'found_on_main',
            'merge_sha': merge_sha,
            'outcome': 'found_on_main',
        }

    async def _git_authority_task_metadata(tid: str, *, site: str) -> dict[str, Any]:
        """Best-effort task metadata for the git-authority guards (task 3103).

        Returns ``{}`` on EVERY failure mode — no harness, no ``scheduler``
        attribute, ``get_task`` raising, or a None/metadata-less task — and
        never raises.  A scheduler fault must degrade a single guard, not
        swallow the whole probe.

        ``{}`` deliberately FAILS OPEN out of the degeneracy check.  On the
        ``merge_status`` path it then falls THROUGH to the citation gate,
        which is git-only and needs no task metadata; on the
        ``merge_request`` fast path there is no citation gate, so the block
        simply reverts to its pre-3103 ancestry/patch-id behaviour.  Either
        way this is exact parity with the harness, which treats an absent or
        non-40-hex ``branch_base_sha`` as "no degeneracy signal" rather than
        as grounds to reject: a metadata fault must never fabricate a
        confident answer, and must never hard-fail a genuinely merged branch.

        Args:
            tid: Bare task id (no ``task/`` prefix).  Both callers derive it
                from the branch ref they resolved the tip from, so the
                metadata and the tip always describe the same branch.
            site: The calling tool (``'merge_status'`` / ``'merge_request'``),
                interpolated into the degradation warning.  Without it a
                scheduler fault on the SUBMIT path was logged as a
                merge_status failure, so an operator grepping for a
                submit-path degradation would not find it (review #3).
        """
        if harness is None:
            return {}
        scheduler = getattr(harness, 'scheduler', None)
        if scheduler is None:
            return {}
        try:
            task = await scheduler.get_task(tid)
        except Exception:
            logger.warning(
                '%s: scheduler.get_task(%s) failed — proceeding without task '
                'metadata (degeneracy check skipped)',
                site, tid, exc_info=True,
            )
            return {}
        if not task:
            return {}
        return task.get('metadata') or {}

    @mcp.tool()
    async def merge_status(
        request_id: str | None = None,
        branch: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the current merge state for a merge request.

        Accepts one of: request_id (authoritative), branch (most-recent), or
        task_id (most-recent).  Lookup order:

            live snapshot → retention ring → event store
            → git-authority (is_ancestor / find_merge_marker against main)
            → {state:'unknown', hint}

        The git-authority tier (Tier-3.5) fires when the durable tiers miss.
        It derives the full branch ref from the passed ``branch`` or
        ``task_id`` via ``canonical_queued_branch_name`` (prepending
        ``orch_config.git.branch_prefix`` unless the value already starts
        with the prefix — the same shape-tolerant rule shared with
        ``recover_pending_merges``), then:
        - If the branch still exists: calls ``is_ancestor(tip, main)``, then
          applies THREE guards in order (task 3103 brought the last two to
          parity with the orchestrator harness's already-landed dispatch
          gate, which has had them since task 1226):
            1. ``tip != main_tip`` — a branch sitting at exactly main's HEAD
               satisfies ``is_ancestor`` trivially (a commit is its own
               ancestor) but nothing has been merged;
            2. NOT degenerate — a tip still equal to the recorded
               ``branch_base_sha`` proves zero commits were ever pushed, so
               the branch is merely parked at an OLD main commit (which IS an
               ancestor of main and IS distinct from main_tip, so guard 1
               does not catch it);
            3. ``validate_landing_evidence`` DISCOVERY mode — a commit on
               main must positively cite the task and its effect must still
               be present at main HEAD.
          On hit → state='done', kind='found_on_main',
          merge_sha=<the citation commit on main>.
          Guards 2 and 3 are independent and both required: a re-seeded
          branch is non-degenerate yet uncited, while a degenerate branch may
          still have a citing commit on main.  When
          ``git.commit_citation_pattern`` is ``''`` (the documented
          per-project opt-out) guard 3 is skipped and merge_sha is the branch
          tip — not a commit on main, and not effect-present-checked; guard 2
          still applies.  See ``_found_on_main_response`` for what that costs
          a caller stamping merge_sha as provenance.
        - If the branch ref is gone (tip is None): calls ``find_merge_marker``
          which searches git log for the merge commit subject.  On hit, two
          further guards (task 3103, mirroring the harness marker arm):
          the marker must NOT predate the recorded ``branch_base_sha`` (else
          the branch was deleted and recreated under the same id and the
          marker belongs to a previous incarnation), and
          ``validate_landing_evidence`` CANDIDATE mode must find the marker's
          effect still present at main HEAD (the marker's subject match
          already establishes attribution).
          On hit → state='done', kind='found_on_main',
          merge_sha=<merge-commit SHA on main>.
        Fire-safe: any git failure degrades to the honest Tier-4 unknown
        (``logger.warning(exc_info=True)``), never raises.  The tier is skipped
        when ``harness.git_ops`` or ``orch_config`` are absent.

        Returns a dict with at minimum:
            state, request_id, generation (always 1 in α3).

        Live entries also carry: position, enqueued_at, eta_seconds.
        Terminal entries carry: outcome (raw state), finished_at.
        git-authority terminal shape: state='done', kind='found_on_main',
            merge_sha=<a commit ON MAIN — the discovered citation or the
            merge marker; see ``_found_on_main_response``>,
            outcome='found_on_main'.
        Unknown carries: hint.
        """
        # Validation — at least one key required
        if request_id is None and branch is None and task_id is None:
            return {'error': 'At least one of request_id, branch, or task_id is required'}

        # Tier 1: live snapshot — wrapped fire-safe so a transient worker-introspection
        # failure degrades to the durable tiers rather than erroring the read-only probe.
        worker = _get_merge_worker(harness)
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
            resp: dict = {
                'state': coarse_state,
                'request_id': meta['request_id'],
                'generation': 1,
                'outcome': meta['outcome'],
                'finished_at': meta['finished_at'],
            }
            if meta.get('superseded_by') is not None:
                resp['superseded_by'] = meta['superseded_by']
            if meta.get('reason') is not None:
                resp['reason'] = meta['reason']
            return resp

        # Tier 3.5: git-authority — probe main directly when no durable record exists.
        # Obtain git_ops via the same accessor as the merge_request fast-path (server.py:766).
        # Guards: git_ops present AND orch_config present AND a resolvable key (branch or
        # task_id).  Any absent → straight to honest Tier-4 unknown (unchanged behaviour).
        # FIRE-SAFE: the entire probe is try/except Exception so a git failure degrades to
        # the honest unknown rather than raising.  Mirrors the Tier-1 snapshot fire-safe
        # wrapper at server.py:1278.
        git_ops = getattr(harness, 'git_ops', None) if harness is not None else None
        if git_ops is not None and orch_config is not None:
            key = branch if branch is not None else task_id
            if key is not None:
                try:
                    prefix = orch_config.git.branch_prefix
                    full_branch = canonical_queued_branch_name(key, prefix)
                    tip = await git_ops.resolve_branch_sha(full_branch)
                    main_tip = await git_ops.resolve_branch_sha(orch_config.git.main_branch)
                    tid = full_branch.removeprefix(prefix)
                    # Runtime-only reverse import: orchestrator depends on escalation,
                    # not vice versa, so this lazy import deliberately avoids a static
                    # cycle (same shape as server.py:1423 / :2049 / :2148).  It resolves
                    # at runtime because the escalation server is hosted inside the
                    # orchestrator process.  An ImportError is an Exception and therefore
                    # already degrades to the honest Tier-4 unknown via the wrapper below.
                    from orchestrator.landing_evidence import (  # type: ignore[reportMissingImports]
                        branch_is_degenerate,
                        is_valid_sha_40,
                        validate_landing_evidence,
                    )
                    if (tip is not None and tip != main_tip
                            and await git_ops.is_ancestor(tip, orch_config.git.main_branch)):
                        # Live branch is already an ancestor of main (normal merged case).
                        # tip != main_tip guards against the no-op case: a branch sitting at
                        # exactly main's HEAD satisfies is_ancestor trivially (a commit is
                        # its own ancestor) but nothing has been merged.
                        # Degeneracy guard (task 3103): a tip still equal to the
                        # recorded branch_base_sha proves ZERO commits were ever
                        # pushed beyond the creation point.  Such a branch is parked
                        # at an OLD main commit, which makes it an ancestor of main
                        # AND distinct from main_tip — both conjuncts above pass — so
                        # without this guard the tier stamps a confident `done`
                        # against a commit containing none of the task's work.  A
                        # degenerate branch falls through to the honest Tier-4
                        # unknown.  Runs FIRST and independently of the citation gate:
                        # a degenerate branch whose task DOES have a citing commit on
                        # main (reify 5493) is caught only by this ordering.
                        # branch_tip_sha=tip: the probe judges degeneracy
                        # against the SAME tip the is_ancestor check above
                        # just ran on, instead of re-reading the ref (review
                        # #2) — one subprocess fewer, and no window for a
                        # warm-lane reseed to split the two observations.
                        metadata = await _git_authority_task_metadata(
                            tid, site='merge_status',
                        )
                        if not await branch_is_degenerate(
                            git_ops, full_branch, metadata, branch_tip_sha=tip,
                        ):
                            # Citation gate.  Read the pattern off orch_config.git for
                            # consistency with the adjacent .main_branch / .branch_prefix
                            # reads (same object as git_ops.config in production).
                            pattern = orch_config.git.commit_citation_pattern
                            if pattern == '':
                                # Documented per-project opt-out (config.py
                                # commit_citation_pattern): '' disables the citation
                                # check entirely for projects without citation
                                # conventions, and find_task_citation_commit honours it
                                # by returning None for EVERYTHING.  Running the gate
                                # here would therefore reject unconditionally and turn
                                # Tier 3.5 into dead code rather than merely un-gated —
                                # a silent capability loss for an explicit opt-in.
                                # Note: None means "use the built-in
                                # DEFAULT_COMMIT_CITATION_PATTERN" and is NOT the
                                # opt-out.  The degeneracy guard above still applies in
                                # this mode.
                                # The returned merge_sha is therefore the raw BRANCH
                                # TIP — not a commit on main, and NOT effect-present
                                # checked.  That is the price of the opt-out, and it is
                                # called out explicitly in _found_on_main_response's
                                # docstring and in both SKILL.md runbooks so a caller
                                # on such a project does not stamp it as verified
                                # provenance (review #4).
                                return _found_on_main_response(request_id, tip)
                            # DISCOVERY mode: a commit on main must positively cite the
                            # task (FIX 2) AND its effect must still be present at main
                            # HEAD (FIX 1', the task-1175 reverted-landing guard).  The
                            # accepted evidence_sha is a commit ON MAIN, which also
                            # retires the old wart of answering with the branch tip.
                            # No escalation on reject — mirrors the harness ancestor
                            # arm's silent-False, and merge_status is a read-only probe.
                            verdict = await validate_landing_evidence(
                                git_ops, tid, full_branch,
                                branch_tip_sha=tip,
                                pattern_template=pattern,
                            )
                            # `accepted` implies a non-None evidence_sha (see
                            # LandingEvidenceVerdict), but assert it explicitly:
                            # _found_on_main_response's merge_sha is a hard `str`,
                            # and a contract violation must degrade to Tier-4
                            # unknown rather than emit a `done` with a null sha.
                            if verdict.accepted and verdict.evidence_sha is not None:
                                return _found_on_main_response(
                                    request_id, verdict.evidence_sha,
                                )
                    elif tip is None:
                        # Branch ref gone — the canonical 4352 deleted-branch shape.
                        # find_merge_marker internally gates on branch existence so it only
                        # fires when the ref is gone (consistent with the cheaper-common-path
                        # ordering: cheaper is_ancestor check first, find_merge_marker only
                        # when the branch has been deleted).
                        # merge_sha = merge-commit SHA on main (via git log scan).
                        marker = await git_ops.find_merge_marker(full_branch)
                        if marker is not None:
                            metadata = await _git_authority_task_metadata(
                                tid, site='merge_status',
                            )
                            branch_base_sha = metadata.get('branch_base_sha')
                            # Predates-this-incarnation veto (task 3103, mirroring
                            # the harness marker arm): the branch was deleted and
                            # recreated under the SAME task id, so a marker older
                            # than this incarnation's base attributes a previous
                            # run's merge to the current task.  is_valid_sha_40 sits
                            # on the LEFT of the `and` so a missing or malformed
                            # base never reaches is_ancestor with a bad argument.
                            if not (
                                is_valid_sha_40(branch_base_sha)
                                and await git_ops.is_ancestor(marker, branch_base_sha)
                            ):
                                # CANDIDATE mode: the marker's subject match already
                                # establishes attribution, so only the FIX 1'
                                # effect-present guard remains — closing the
                                # task-1175 clobber where a reverted merge still
                                # read as a genuine landing.  No escalation on
                                # reject (unlike the harness marker path):
                                # merge_status is a read-only probe with no write
                                # side, so a reject degrades to Tier-4 unknown.
                                verdict = await validate_landing_evidence(
                                    git_ops, tid, full_branch,
                                    branch_tip_sha=None,
                                    candidate_sha=marker,
                                )
                                # Same non-None assertion as the ancestor arm
                                # above: reject a null evidence sha into Tier-4
                                # unknown rather than into a `done` response.
                                if verdict.accepted and verdict.evidence_sha is not None:
                                    return _found_on_main_response(
                                        request_id, verdict.evidence_sha,
                                    )
                except Exception:
                    logger.warning(
                        'merge_status: git-authority probe failed, returning unknown',
                        exc_info=True,
                    )

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
          4. Waiter present, future pending: cancel the future, then FULLY RETIRE
             the entry (release the branch slot, reap the in-flight worktree via
             the C1 primitive, clear the sticky per-task result) BEFORE returning
             {cancelled: True, state: 'abandoned', reason: None}.

        RETIREMENT (task ε): on the pending path, after cancelling the future the
        entry is fully retired BEFORE returning via retire_cancelled_merge_request:
          - InFlightMergeRegistry: the branch slot is released synchronously-before-
            return (identity-guarded — only when it still belongs to this request_id,
            so a concurrent resubmit that reclaimed the slot is not clobbered).
          - the in-flight worktree is reaped via the C1 primitive
            remove_merge_worktree_guarded(reason='merge_cancel_retire').
          - the sticky per-task retention result is cleared (yield-then-forget: the
            cancel's _on_finalized records terminal 'abandoned' first, then forget
            removes it), so an immediate resubmit gets a FRESH entry and never
            coalesces onto / observes the cancelled corpse.
        Still async (unchanged): MergeWorker._request_abandoned drops the queued
        work item at its next checkpoint via the cancel→worker CancelledError seam.
        Because retirement yields the loop (the C1 removal is awaited), the
        merge_request _waiters.pop done-callback fires during the first cancel, so an
        immediate double-cancel finds rec=None and resolves via _durable_terminal_state.

        The tool is async so that future mutation runs on the event loop (not a FastMCP
        threadpool worker — PRD Open Q4 off-loop lesson).  The lookup → cancel sequence
        contains no await, preserving loop-synchronous race-freedom (I10) — the only
        awaits are AFTER cancel, inside retirement.
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

        # Pending waiter — cancel the future, then FULLY RETIRE the entry (task ε)
        # BEFORE returning: release the branch slot, reap the in-flight worktree
        # via the C1 primitive, and clear the sticky retention result, so an
        # immediate resubmit gets a FRESH entry.  lookup→cancel above stays
        # await-free (I10); the only awaits are here, after cancel.
        rec.future.cancel()
        from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
            retire_cancelled_merge_request,
        )
        await retire_cancelled_merge_request(
            request_id=request_id,
            branch=rec.branch,
            task_id=rec.task_id,
            registry=_registry,
            retention=_get_terminal_retention(harness),
            git_ops=getattr(harness, 'git_ops', None),
            event_store=event_store,
        )
        return {'cancelled': True, 'state': 'abandoned', 'reason': None}

    return mcp
