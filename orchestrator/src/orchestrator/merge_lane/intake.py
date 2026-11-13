"""The merge lane's request intake: the doors a merge request enters by.

:func:`enqueue_merge_request` is the one chokepoint that puts a request on the
merge queue. The workflow reaches it through
:func:`register_and_enqueue_merge_request`, the ``merge_request`` MCP tool
through :func:`coalesce_or_enqueue_merge_request`, which first coalesces a
request onto a live merge of the same branch, or rejects it.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.liveness import INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS
from orchestrator.merge_lane.tip_relation import (
    C3SubmitAction,
    TipRelation,
    classify_tip_relation,
    decide_c3_submit_action,
    resolve_divergent,
)
from orchestrator.merge_lane.types import (
    InFlightMergeRegistry,
    MergeDispatchResult,
    MergeOutcome,
    MergeRequest,
    TerminalOutcomeRecord,
    TerminalOutcomeRetention,
    _InFlightEntry,
)
from orchestrator.merge_pool_facts import queued_request_fields
from orchestrator.task_priority import (
    UNKNOWN_PRIORITY,
    TaskPriority,
    TaskPriorityLookup,
    read_task_priority,
)

logger = logging.getLogger('orchestrator.merge_lane')

# Structured reject code surfaced by the merge_request MCP tool when a NEWER
# SHA is submitted for a branch whose EARLIER SHA is already in verify
# (PRD merge-worktree-lifecycle-integrity §5 D3).  Consumed by the escalation
# server's merge_request handler to build the {error, code, …} envelope.
_C3_DUPLICATE_IN_VERIFY_CODE = 'duplicate_in_verify'

MERGE_LANES: tuple[Literal['high', 'normal'], ...] = ('high', 'normal')
"""Priority-ordered lane names (high first).  ``MergeIntake`` drains and picks
requests in this order: the first non-halted non-empty lane wins."""


def normalize_lane(lane: str) -> Literal['high', 'normal']:
    """Map an unrecognised lane value to 'normal' (defensive; prevents starvation)."""
    return lane if lane in MERGE_LANES else 'normal'


def _emit_merge_queued(
    event_store: EventStore | None,
    req: MergeRequest,
    reason: str | None = None,
    *,
    queue_depth: int | None = None,
    position: int | None = None,
    source: str | None = None,
    priority: TaskPriority = UNKNOWN_PRIORITY,
) -> None:
    """Emit a merge_queued event.  No-op when *event_store* is None.

    Centralises the emit payload so every caller produces the same record
    shape — currently only :func:`enqueue_merge_request`; the CAS-retry
    re-enqueue that passed *reason* and *position* went with the serial worker
    (see :class:`_TrainMergeHost`).  If
    *reason* is provided (e.g. ``'cas_retry'``) it is stored in ``data``.

    *queue_depth* (when provided) records how deep the main queue was at the
    moment of enqueue — O(1) qsize() from the call site.  *position* (when
    provided) records the front-of-line position for urgent re-inserts (0 ==
    head).  *source* (when provided) tags the submission origin — e.g.
    ``'stranded-reaper'`` for the verified-green stranded remediation
    (stranding-remediation-scheduler-ergonomics-prd.md leaf α §6α) —
    distinguishing reaper submissions from workflow/MCP ones.  Each key is
    omitted when None so the shape remains backward-compatible with existing
    consumers.  The keys of
    ``orchestrator/src/orchestrator/merge_pool_facts.py::queued_request_fields``
    are always present, null when unknown; *priority* supplies their tier and pin.
    """
    if event_store is None:
        return
    data: dict = {'branch': req.branch.bare_id, **queued_request_fields(req, priority)}
    if reason is not None:
        data['reason'] = reason
    if queue_depth is not None:
        data['queue_depth'] = queue_depth
    if position is not None:
        data['position'] = position
    if source is not None:
        data['source'] = source
    event_store.emit(
        EventType.merge_queued,
        task_id=req.task_id,
        phase='merge',
        data=data,
    )


async def enqueue_merge_request(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
    *,
    retention: TerminalOutcomeRetention | None = None,
    source: str | None = None,
    registry: InFlightMergeRegistry | None = None,
    task_priority: TaskPriorityLookup | None = None,
) -> None:
    """Enqueue a MergeRequest and emit a merge_queued event.

    *source* (when provided) tags the merge_queued event's ``data['source']``
    with the submission origin (e.g. ``'stranded-reaper'``); default None
    keeps every existing caller's event shape byte-identical — the key is
    omitted when unset.

    Puts the request on *queue* first so that a cancellation between put and
    emit (or any emit error) does not leave a dangling ``merge_queued`` row
    with no corresponding worker pickup.  Losing the event is less confusing
    than a stale "queued" row that persists until the TTL expires.

    If ``event_store`` is None the request is still enqueued; emission is
    silently skipped (mirrors the None-safe pattern used by
    ``_emit_merge_attempt``).

    Registers a single ``req.result.add_done_callback`` that, when the future
    reaches its terminal state (resolved, cancelled, or exception), emits a
    ``merge_finalized`` event and records the outcome into *retention* (when
    provided).  The callback is fire-and-forget: any exception is logged as a
    warning and never propagates.

    That row's ``absorbed_request_ids`` lists the requests that resolve to this
    one's outcome without a row of their own: the waiters and door-coalesced
    ids of the *registry* slot *req* owns at enqueue time, else ``[]``
    (PRD plans/merge-status-durable-non-landed-prd.md D7).

    Its ``queue_to_land_secs`` is measured on the monotonic clock from this
    call's enqueue, which is the request's last ``merge_queued``: every call
    emits exactly one, queue_store's restart re-enqueue included.
    """
    absorbing = (
        registry.primary_entry(req.branch.bare_id, req.result)
        if registry is not None
        else None
    )
    queued_at = time.monotonic()

    def _on_finalized(fut: asyncio.Future) -> None:  # noqa: ANN001
        # --- derive terminal state -------------------------------------------
        superseded_by: str | None = None
        reason: str | None = None
        landed_via_chain: int | None = None
        try:
            if fut.cancelled():
                state: str = 'abandoned'
                merge_sha: str | None = None
            elif fut.exception() is not None:
                state = 'error'
                merge_sha = None
            else:
                outcome: MergeOutcome = fut.result()
                state = outcome.status
                merge_sha = outcome.merge_sha
                superseded_by = outcome.superseded_by
                reason = outcome.reason or None
                landed_via_chain = outcome.landed_via_chain
        except Exception:  # noqa: BLE001
            logger.warning(
                'enqueue_merge_request: _on_finalized could not derive terminal '
                'state for request_id=%s task_id=%s',
                req.request_id, req.task_id, exc_info=True,
            )
            return
        # --- in-memory hot tier (recorded before durable tier so a DB failure
        #     does not also degrade the O(1) lookup ring) --------------------
        if retention is not None:
            try:
                retention.record(TerminalOutcomeRecord(
                    request_id=req.request_id,
                    task_id=req.task_id,
                    branch=req.branch.bare_id,
                    state=state,
                    snapshot_tip=req.snapshot_tip,
                    merge_sha=merge_sha,
                    superseded_by=superseded_by,
                    generation=req.generation,
                    reason=reason,
                    landed_via_chain=landed_via_chain,
                ))
            except Exception:  # noqa: BLE001
                logger.warning(
                    'enqueue_merge_request: _on_finalized retention.record failed '
                    'for request_id=%s task_id=%s',
                    req.request_id, req.task_id, exc_info=True,
                )
        # --- durable tier ----------------------------------------------------
        if event_store is not None:
            try:
                event_store.emit(
                    EventType.merge_finalized,
                    task_id=req.task_id,
                    phase='merge',
                    data={
                        'request_id': req.request_id,
                        'branch': req.branch.bare_id,
                        'state': state,
                        'snapshot_tip': req.snapshot_tip,
                        'merge_sha': merge_sha,
                        'superseded_by': superseded_by,
                        'generation': req.generation,
                        'reason': reason,
                        'landed_via_chain': landed_via_chain,
                        'absorbed_request_ids': (
                            list(absorbing.absorbed_request_ids())
                            if absorbing is not None
                            else []
                        ),
                        'queue_to_land_secs': round(time.monotonic() - queued_at, 3),
                    },
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    'enqueue_merge_request: _on_finalized event_store.emit failed '
                    'for request_id=%s task_id=%s',
                    req.request_id, req.task_id, exc_info=True,
                )

    req.result.add_done_callback(_on_finalized)
    await queue.put(req)
    _emit_merge_queued(
        event_store, req, queue_depth=queue.qsize(), source=source,
        priority=read_task_priority(task_priority, req.task_id),
    )


async def _enqueue_holding_slot(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
    registry: InFlightMergeRegistry | None,
    retention: TerminalOutcomeRetention | None,
    task_priority: TaskPriorityLookup | None = None,
) -> None:
    """Enqueue *req* through :func:`enqueue_merge_request`, guarding its slot.

    Slot-leak guard: if the enqueue raises (e.g. queue closed, cancellation)
    before the worker can ever resolve req.result, the done_callback will
    never fire.  When *req* holds its branch's slot in *registry*, release
    the slot explicitly so a future merge for this branch can proceed.
    """
    branch = req.branch.bare_id
    holds_slot = registry is not None and registry.primary_entry(branch, req.result) is not None
    try:
        await enqueue_merge_request(
            queue, req, event_store, retention=retention, registry=registry,
            task_priority=task_priority,
        )
    except BaseException:
        if holds_slot and registry is not None:
            registry.release(branch)
        raise

async def register_and_enqueue_merge_request(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
    registry: InFlightMergeRegistry | None,
    *,
    retention: TerminalOutcomeRetention | None = None,
    task_priority: TaskPriorityLookup | None = None,
) -> bool:
    """Workflow-path enqueue that registers the branch in the in-flight registry.

    Unlike the MCP path (``coalesce_or_enqueue_merge_request``), the workflow
    MUST always enqueue — it blocks on ``req.result`` for its outcome, so
    skipping the enqueue when the slot is already held would leave the future
    unresolved and deadlock the caller.

    When *registry* is not None and the slot is free, ``registry.acquire`` is
    called with ``req.result`` before enqueuing.  The existing
    ``Future.add_done_callback`` registered inside ``acquire`` releases the
    slot automatically on result, exception, or cancellation.  If the enqueue
    itself raises (e.g. queue closed or cancellation), the slot is released
    explicitly to avoid a leak — the same guard
    ``coalesce_or_enqueue_merge_request`` uses (:func:`_enqueue_holding_slot`).

    When *retention* is provided it is forwarded to the single
    :func:`enqueue_merge_request` chokepoint so the dominant workflow path
    populates the in-memory ring alongside the MCP path (see
    :func:`coalesce_or_enqueue_merge_request`).  Existing call sites in
    workflow.py pass only positional args so this keyword-only param is
    backwards-compatible.

    Returns True if the registry slot was newly acquired (caller's branch was
    free); False if the slot was already held by another task or *registry* is
    None.  The return value is informational — the request is always enqueued.
    """
    acquired = (
        registry.acquire(req.branch.bare_id, req.task_id, req.result, request_id=req.request_id)
        if registry is not None
        else False
    )
    await _enqueue_holding_slot(queue, req, event_store, registry, retention, task_priority)
    return acquired


def emit_merge_coalesced(
    event_store: EventStore | None,
    req: MergeRequest,
    source: str,
    eta: int | None,
) -> None:
    """Emit a merge_coalesced event.  No-op when *event_store* is None.

    Mirrors the ``_emit_merge_queued`` helper so the coalesced path emits
    an identical-shape record but with ``merge_coalesced`` as the event type.
    """
    if event_store is None:
        return
    data: dict = {
        'branch': req.branch.bare_id,
        'source': source,
    }
    if eta is not None:
        data['eta_seconds'] = eta
    event_store.emit(
        EventType.merge_coalesced,
        task_id=req.task_id,
        phase='merge',
        data=data,
    )


class _FindInflightWorktreeP(Protocol):
    """Narrow protocol for the disk-scan used by coalesce_or_enqueue_merge_request.

    Only :meth:`find_inflight_merge_worktree` is called on *git_ops* inside
    :func:`coalesce_or_enqueue_merge_request`.  Declaring the parameter with
    this Protocol (rather than the concrete :class:`~orchestrator.git_ops.GitOps`)
    lets test stubs that implement only this method satisfy the type-checker
    without inheriting from the full ``GitOps`` class.
    """

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None: ...
    async def cleanup_merge_worktree(self, merge_wt: Path) -> None: ...
    async def remove_merge_worktree_guarded(self, path: Path, *, reason: str) -> str: ...


def _inflight_entry_is_stale(
    entry: _InFlightEntry | None,
    branch: str,
    live_snapshot: Callable[[], dict] | None,
) -> bool:
    """Return True when *entry* appears stale relative to the live worker snapshot.

    Liveness is determined by matching ``entry.request_id`` against the
    ``request_id`` fields of snapshot entries.  When ``entry.request_id`` is
    None (legacy entries predating the request_id field) the match falls back
    to branch-name comparison.

    Returns **False** (not stale → keep coalescing) in four cases:

    1. *live_snapshot* is None — no provider wired (back-compat default; all
       existing callers without a worker).
    2. ``live_snapshot()`` raises — transient error (fail-safe to coalesce;
       avoids double-dispatch storms on transient worker hiccups).
    3. The snapshot is not a dict or lacks the ``'entries'`` key — malformed
       response (fail-safe to coalesce; mirrors case 2's philosophy: when the
       snapshot cannot be trusted, assume live rather than risk a double-dispatch).
    4. The entry's ``request_id`` IS present in the snapshot — slot is live.
    """
    if live_snapshot is None:
        return False
    try:
        snap = live_snapshot()
    except Exception:
        return False  # fail-safe: transient error → not stale → coalesce
    if not isinstance(snap, dict) or 'entries' not in snap:
        return False  # malformed snapshot → fail-safe: not stale → coalesce
    entries = snap['entries']
    rid = entry.request_id if entry is not None else None
    if rid is not None:
        return not any(e.get('request_id') == rid for e in entries)
    else:
        # Legacy entry without a request_id: fall back to branch matching.
        return not any(e.get('branch') == branch for e in entries)


def _inflight_worktree_is_stale(
    wt: Path,
    branch: str,
    live_snapshot: Callable[[], dict] | None,
) -> bool:
    """Return True when the on-disk ``_merge-*`` worktree *wt* is a CORPSE.

    The disk-scan companion to :func:`_inflight_entry_is_stale`, with the same
    polarity: **True = corpse → reap and dispatch fresh**, **False = keep
    coalescing**.

    Why a snapshot signal is needed at all.  The disk-scan coalesce arm decides
    liveness from *wt*'s ROOT inode mtime.  But
    :meth:`SpeculativeMergeWorker._touch_owned_merge_worktrees` ``os.utime``s
    the root inode of every path in ``_owned_merge_worktrees`` on every
    heartbeat tick (~30 s, i.e. ~360x inside the 10800 s liveness window).  So
    for an OWNED worktree root mtime measures *owner-process* liveness, not
    *merge* liveness: once such a worktree outlives its request's terminal
    finalization while still in the ledger, the heartbeat pins its mtime fresh
    forever, ``age <= liveness_secs`` is permanently true, and every resubmit
    for that branch coalesces onto a corpse until the process restarts.
    :func:`orchestrator.merge_lane.liveness.newest_content_mtime` states the same
    hazard for the in-flight verify no-progress budget ("the #1728 alpha
    owner-heartbeat ... can never mask a dead/hung verify subprocess"); this
    predicate is that treatment for the coalesce arm.

    Why OWNERSHIP gates the verdict.  The disk-scan arm exists for
    cross-process / post-restart crash-safety: an in-progress merger's
    ``_merge-*`` worktree persists on disk even when the in-memory registry was
    cleared by a restart, and it must still coalesce or two mergers race onto
    one branch.  A foreign or pre-restart merger's worktree is BY CONSTRUCTION
    absent from *this* worker's snapshot entries, so a bare branch-absence test
    would reap genuinely live foreign merges.  Ownership is the precise
    discriminator: for a worktree this worker is heartbeating, mtime is
    known-contaminated and cannot be evidence of life, so "no live entry for
    the branch" proves the owning request already finalized.  Everything else
    keeps the pre-existing behaviour.

    Returns **False** (not stale → keep coalescing) in six cases:

    1. *live_snapshot* is None — no provider wired (back-compat default; all
       existing callers without a worker).
    2. ``live_snapshot()`` raises — transient error (fail-safe to coalesce).
    3. The snapshot is not a dict or lacks the ``'entries'`` key — malformed
       response.  Same philosophy as the sibling predicate: when the snapshot
       cannot be trusted, assume live rather than risk a double-dispatch, which
       is strictly worse because a delayed coalesce is self-healing while a
       double-dispatch corrupts main.
    4. The snapshot lacks the ``'owned_merge_worktrees'`` key entirely — a
       producer predating that additive key (a test double, or a rolling deploy
       where the escalation server sees an older worker).  Distinguished from
       an *empty* ledger, which is a live worker's honest answer and still
       permits the decisive check below.
    5. The ownership comparison cannot be computed — *wt* is unresolvable
       (OSError), or ``owned_merge_worktrees`` is not an iterable of
       path-likes (TypeError/ValueError, e.g. an int, or a list holding a
       non-path element).  A malformed snapshot must degrade to "coalesce",
       never fail the caller's submission: this predicate is reached from the
       ``merge_request`` MCP tool, so an exception here would reject the
       submission outright instead of falling back to today's behaviour.
    6. *wt* is not in this worker's ledger — foreign / pre-restart merger; its
       mtime is real evidence, so the crash-safety property is preserved.

    The final branch scan is guarded the same way: a snapshot whose
    ``entries`` is not a list of dicts yields False rather than raising.

    Ownership is compared on ``Path.resolve()``, matching how
    :meth:`SpeculativeMergeWorker.worktree_ledger_violations` already
    normalises the same ledger, so the two ownership tests can never disagree.
    The real producer (:meth:`SpeculativeMergeWorker.snapshot`) already emits
    resolved strings, so re-normalising here is redundant for it — but it is
    what lets a hand-built ledger (a test double, or any caller assembling the
    dict itself) spell a path unnormalised and still be recognised as owned.
    """
    if live_snapshot is None:
        return False
    try:
        snap = live_snapshot()
    except Exception:
        return False  # fail-safe: transient error → not stale → coalesce
    if not isinstance(snap, dict) or 'entries' not in snap:
        return False  # malformed snapshot → fail-safe: not stale → coalesce
    owned = snap.get('owned_merge_worktrees')
    if owned is None:
        # Key absent (not merely empty) → producer predates the additive key.
        # Note the `is None` test: an EMPTY list from a live worker is a
        # trustworthy answer and must not short-circuit the decisive check.
        return False
    try:
        wt_key = str(Path(wt).resolve())
        owned_keys = {str(Path(p).resolve()) for p in owned}
    except (OSError, TypeError, ValueError):
        # Unresolvable path, or a ledger that is not an iterable of
        # path-likes.  Fail-safe: not stale → coalesce.  Broader than OSError
        # alone because the realistically-malformed input is a wrong-typed
        # ledger (TypeError), not an unresolvable path — `Path.resolve()` is
        # non-strict and practically never raises.
        return False
    if wt_key not in owned_keys:
        return False  # not ours → mtime is real evidence → keep coalescing
    # Owned by this worker: mtime is heartbeat-pinned, so the ONLY liveness
    # signal is whether the live pipeline still carries this branch.
    try:
        return not any(e.get('branch') == branch for e in snap['entries'])
    except (TypeError, AttributeError):
        return False  # malformed 'entries' → fail-safe: not stale → coalesce


def _snapshot_verify_state(
    live_snapshot: Callable[[], dict] | None,
    request_id: str | None,
) -> tuple[bool, int | None]:
    """Read the in-verify signal for *request_id* from the live worker snapshot.

    Returns ``(verifying, verify_age_secs)``:

    - *verifying* is True iff the snapshot entry matching *request_id* by
      ``request_id`` carries a non-None ``verify_started_at`` — i.e. the worker
      has begun verifying that slot (``SpeculativeMergeWorker.snapshot()`` sets
      ``verify_started_at`` only on ``_inflight`` entries).
    - *verify_age_secs* is the integer age of that verify (from the snapshot
      entry's ``verify_age_secs``), or None when unknown.

    This is the in-scope, in-production "in verify" signal consumed by the C3
    submit gate (the registry is not yet threaded into the worker to flip
    ``entry.verifying`` — γ3/task 1641); the gate ORs the returned *verifying*
    with ``entry.verifying`` so a unit test can drive the flag directly and a
    future worker-side wiring composes.

    Fail-safe to ``(False, None)`` — no provider wired, a raising or malformed
    snapshot, or an unmatched request_id all read as NOT verifying (mirrors
    :func:`_inflight_entry_is_stale`'s trust-the-registry-on-doubt philosophy;
    ``entry.verifying`` remains the fallback signal).  Synchronous (no await)
    so the caller can read it inside the atomic decision block.
    """
    if live_snapshot is None or request_id is None:
        return (False, None)
    try:
        snap = live_snapshot()
    except Exception:
        return (False, None)
    if not isinstance(snap, dict) or 'entries' not in snap:
        return (False, None)
    for e in snap['entries']:
        if e.get('request_id') == request_id:
            if e.get('verify_started_at') is None:
                return (False, None)
            age = e.get('verify_age_secs')
            return (True, int(age) if age is not None else None)
    return (False, None)


async def coalesce_or_enqueue_merge_request(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
    registry: InFlightMergeRegistry,
    git_ops: _FindInflightWorktreeP | None = None,
    *,
    liveness_secs: int = INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
    retention: TerminalOutcomeRetention | None = None,
    live_snapshot: Callable[[], dict] | None = None,
    classifier_git_ops: GitOps | None = None,
) -> MergeDispatchResult:
    """De-dup gate for the merge_request MCP chokepoint.

    Consults two sources of truth to detect an already-in-flight merge for
    *req.branch*:

    1. **In-memory registry fast-path** — O(1) dict lookup, covers within-
       process dispatch races (``merge_request`` /unblock spam).
    2. **On-disk worktree scan** (when *git_ops* is not None) — detects the
       workflow's in-flight merger whose ``_merge-*`` worktree exists on disk
       even after a process restart.  Alive worktrees (mtime ≤ liveness_secs)
       are coalesced; stale/abandoned worktrees are reaped via
       ``cleanup_merge_worktree`` and a fresh merge is dispatched (step-10).

    Returns a :class:`MergeDispatchResult` with ``dispatched=True`` if the
    request was enqueued, or ``in_flight=True`` if it was coalesced.

    On coalesce a ``merge_coalesced`` event is emitted so observers can track
    the de-dup rate without polling the queue depth.

    **NOT** used by workflow.py's single-task or train paths — those call
    :func:`enqueue_merge_request` directly.  This function is the
    ``merge_request`` MCP tool's entry point only.

    When *retention* is provided it is forwarded to the single
    :func:`enqueue_merge_request` chokepoint so the MCP path populates
    the ring alongside the workflow path (see
    :func:`register_and_enqueue_merge_request`).  Coalesced requests
    are NOT recorded — their terminal outcome is owned by the in-flight
    entry's callback.  However, when *retention* is supplied and an in-flight
    registry entry is available (and the slot is not reaped as stale; see
    *live_snapshot* below), the coalesced request's ``request_id`` is
    registered as an alias onto the in-flight entry's ``request_id`` via
    :meth:`TerminalOutcomeRetention.record_alias`, so callers polling the
    coalesced id will resolve to the primary outcome once it is recorded.

    *live_snapshot* (keyword-only, default None): a zero-argument callable
    that returns the live worker snapshot dict (same shape as
    ``SpeculativeMergeWorker.snapshot()``).  When provided, it **reconciles
    BOTH coalesce arms** against live worker state before coalescing.

    *Registry arm*: if the slot's ``request_id`` is absent from the
    snapshot the slot is considered stale (the request finalized but its
    slot was not auto-released), it is reaped via
    ``registry.release(branch, detach_waiters=True)``, and the call falls
    through to the acquire-and-enqueue block dispatching a fresh request.

    *Disk-scan arm*: the same reconciliation, but additionally gated on
    OWNERSHIP via the snapshot's ``owned_merge_worktrees`` ledger (see
    :func:`_inflight_worktree_is_stale`).  A worktree this worker is
    heartbeating has a root-inode mtime pinned fresh by
    ``_touch_owned_merge_worktrees``, so mtime cannot prove liveness; when the
    snapshot additionally holds no entry for the branch, the owning request has
    finalized and the worktree is a corpse — reaped with reason
    ``finalized_owner`` (distinct from the mtime path's ``stale_inflight``),
    falling through to a fresh dispatch.  Requiring ownership is what preserves
    the cross-process crash-safety property documented above: a foreign or
    pre-restart merger's worktree is by construction absent from this worker's
    snapshot entries, so branch-absence alone would reap live foreign merges.

    When absent (None) or when ``live_snapshot()`` raises, both arms behave
    exactly as before — trust the registry and the on-disk mtime.

    *classifier_git_ops* (keyword-only, default None): when provided AND both
    ``entry.snapshot_tip`` and ``req.snapshot_tip`` are set, the registry
    fast-path runs the SHA-sensitive **C3 submit-identity gate** (PRD
    merge-worktree-lifecycle-integrity §4 C3 / §5 D1-D3) before coalescing.
    The tip relation is classified via :func:`classify_tip_relation` (then
    :func:`resolve_divergent` for a DIVERGENT tip, folding patch-id-equivalent
    rebased twins to SUBSET) and mapped by :func:`decide_c3_submit_action`
    against the "in verify" signal (``entry.verifying`` OR the live-snapshot
    ``verify_started_at`` — see :func:`_snapshot_verify_state`):

      • SAME / SUBSET → COALESCE (fall through to the unconditional coalesce;
        never a second work item, incl. same-SHA-while-verifying — D1).
      • SUPERSET while QUEUED (verify not started) → REPLACE: dispatch the new
        tip (``dispatched=True``).  The atomic drop-and-reacquire + C1 scratch
        clean lands in step-6; the interim keeps the pre-C3 re_snapshot +
        independent-enqueue.
      • SUPERSET while IN VERIFY → REJECT: return a rejected
        :class:`MergeDispatchResult` (``rejected=True``,
        ``reject_code='duplicate_in_verify'``, ``existing_sha`` / ``verify_age_secs``
        from the in-flight entry) leaving the live entry UNDISTURBED (D3).

    The verify-started read and the replace/reject decision are ATOMIC with the
    registry mutation — the only await (ancestry classification) precedes them
    and performs no mutation.  When absent or either snapshot_tip is None, the
    gate is a no-op and the path preserves current behaviour (back-compat).
    The C3 submit-identity gate is scoped to the registry fast-path; the
    disk-scan branch has no in-process entry to classify a tip against.  That
    branch is reconciled instead by the ownership-gated staleness check
    described under *live_snapshot* above.
    """
    submission = _Submission(queue, req, event_store, registry, retention, live_snapshot)
    decided = await _registry_fast_path(submission, git_ops, classifier_git_ops)
    if decided is None and git_ops is not None:
        decided = await _disk_scan_path(submission, git_ops, liveness_secs)
    if decided is None:
        decided = await _acquire_and_enqueue(submission)
    return decided


@dataclasses.dataclass(frozen=True)
class _Submission:
    """One ``merge_request`` submission, as every step of the gate sees it."""

    queue: asyncio.Queue
    req: MergeRequest
    event_store: EventStore | None
    registry: InFlightMergeRegistry
    retention: TerminalOutcomeRetention | None
    live_snapshot: Callable[[], dict] | None

    @property
    def branch(self) -> str:
        """The bookkeeping key: registry slot, worktree scan and result."""
        return self.req.branch.bare_id


async def _registry_fast_path(
    submission: _Submission,
    git_ops: _FindInflightWorktreeP | None,
    classifier_git_ops: GitOps | None,
) -> MergeDispatchResult | None:
    """Step 1: decide against the in-memory registry slot, if the branch holds one.

    None means "no live slot decided it": the branch was free, its slot was
    stale and has been reaped, or the slot released during the C3
    classification await.
    """
    branch = submission.branch
    registry = submission.registry
    if not registry.is_inflight(branch):
        return None
    entry = registry.entry(branch)
    if _inflight_entry_is_stale(entry, branch, submission.live_snapshot):
        # Slot points at a dead request_id (finalize path left the slot
        # un-released).  Reap it so the caller gets a fresh dispatch.
        logger.warning(
            'coalesce_or_enqueue_merge_request: reaping STALE in-flight slot '
            'for branch %r (request_id=%r not present in live worker snapshot); '
            'dispatching fresh',
            branch, entry.request_id if entry else None,
        )
        registry.release(branch, detach_waiters=True)
        return None
    new_tip = submission.req.snapshot_tip
    if (
        classifier_git_ops is not None
        and entry is not None
        and entry.snapshot_tip is not None
        and new_tip is not None
    ):
        decided = await _c3_submit_gate(
            submission, git_ops, classifier_git_ops, new_tip, entry.snapshot_tip,
        )
        if decided is not None:
            return decided
        entry = registry.entry(branch)
    # Coalesce (SAME/SUBSET, back-compat no-classifier, or gate skipped).
    # Guarded on `entry is not None`: if the slot released during the
    # classification await above, fall through to the disk scan and a
    # fresh dispatch rather than coalescing onto a gone entry.
    if entry is None:
        return None
    return _coalesce_onto_entry(submission, entry)


async def _c3_submit_gate(
    submission: _Submission,
    git_ops: _FindInflightWorktreeP | None,
    classifier_git_ops: GitOps,
    new_tip: str,
    old_tip: str,
) -> MergeDispatchResult | None:
    """The C3 submit-identity gate (PRD §4 C3 / §5 D1-D3).

    Classifies the tip relation and routes the submission SHA-sensitively
    (retires the γ2 RESNAPSHOT/ATTACH_AND_CHAIN independent-enqueue
    disposition, which kept TWO work items per branch — the exact
    double-dispatch C3 fixes):

    • SAME / SUBSET   → COALESCE: returns None, so the caller coalesces
      (SUBSET folds stale retries AND patch-id-equivalent rebased twins,
      so genuine content is never lost — amendment).
    • SUPERSET queued → REPLACE (atomic drop-and-reacquire of the registry
      slot + C1 lease-guarded scratch clean, dispatch fresh; one live work
      item per branch — D2).
    • SUPERSET verify → REJECT (structured duplicate_in_verify, D3).

    Atomicity (PRD C3): the ancestry classification (the only await before
    the decision) runs first and is pure/no-mutation; the verifying read +
    decision then run in ONE synchronous block with no await between the
    read and any registry mutation, so the worker cannot transition the
    entry to verifying inside the decision window.
    """
    # (1) Ancestry classification — the ONLY await; pure, no mutation.
    relation = await classify_tip_relation(new_tip, old_tip, classifier_git_ops)
    original_was_divergent = relation is TipRelation.DIVERGENT
    if original_was_divergent:
        # Patch-id containment BEFORE genuine-divergence classification:
        # a rebased twin folds to SUBSET → COALESCE (amendment). A
        # tip that survives as SUPERSET here is a genuine force-push
        # (D2) — logged as a divergence WARNING on the REPLACE arm.
        relation = await resolve_divergent(new_tip, old_tip, classifier_git_ops)
    # (2) Atomic decision block — NO await between the verifying read
    #     and any registry mutation.  The slot may have released during
    #     the await above; re-fetch, and return None for the caller to
    #     coalesce or fall through to a fresh dispatch.
    entry = submission.registry.entry(submission.branch)
    # Re-narrow snapshot_tip: the classification await above may have
    # released+reacquired the slot with a tip-less entry, in which case
    # fall through to the safe coalesce/fresh-dispatch path rather than
    # making a C3 ancestry decision without a tip.
    if entry is None or entry.snapshot_tip is None:
        return None
    snap_verifying, verify_age = _snapshot_verify_state(
        submission.live_snapshot, entry.request_id,
    )
    verifying = snap_verifying or entry.verifying
    action = decide_c3_submit_action(relation, verifying=verifying)
    if action is C3SubmitAction.REJECT:
        return _reject_duplicate_in_verify(submission, entry, new_tip, verify_age)
    if action is C3SubmitAction.REPLACE:
        _replace_queued_entry(submission, entry, new_tip, original_was_divergent)
        # Slot now held by the new request → the following awaits are
        # safe (no free-slot window).
        await _reap_replaced_worktree(submission, git_ops)
        return await _dispatch(submission)
    return None


def _reject_duplicate_in_verify(
    submission: _Submission,
    entry: _InFlightEntry,
    new_tip: str,
    verify_age: int | None,
) -> MergeDispatchResult:
    """D3: a newer SHA cannot supersede an IN-VERIFY twin.

    Rejects structurally, leaving the live entry UNDISTURBED.
    """
    old_tip = entry.snapshot_tip or ''
    logger.warning(
        'coalesce_or_enqueue_merge_request: REJECT '
        'duplicate_in_verify — new tip %s is ahead of the '
        'IN-VERIFY in-flight %s (request_id=%r) for branch %r; '
        'merge_cancel then resubmit.',
        new_tip[:12], old_tip[:12],
        entry.request_id, submission.branch,
    )
    return MergeDispatchResult(
        dispatched=False,
        in_flight=False,
        branch=submission.branch,
        rejected=True,
        reject_code=_C3_DUPLICATE_IN_VERIFY_CODE,
        inflight_task_id=entry.task_id,
        inflight_request_id=entry.request_id,
        existing_sha=entry.snapshot_tip,
        verify_age_secs=verify_age,
    )


def _replace_queued_entry(
    submission: _Submission,
    entry: _InFlightEntry,
    new_tip: str,
    original_was_divergent: bool,
) -> None:
    """D2: SUPERSET while QUEUED (verify not started) — drop the old queued
    entry and claim the slot for the new tip, so the branch keeps exactly
    ONE live work item (retires the γ2 independent-enqueue that kept two).

    Synchronous, so the caller runs it with no await between its verifying
    read and this release+acquire: the worker cannot transition the old
    entry to verifying inside the window. release(detach_waiters=True)
    cancels the old primary_future — the worker drops the still-queued item
    at its next _request_abandoned checkpoint (detach() docstring, zero
    worker code change) — and acquire() reclaims the freed slot for the new
    request back-to-back so no concurrent merge_request double-dispatches
    across a free slot.
    """
    req = submission.req
    branch = submission.branch
    old_request_id = entry.request_id
    old_snapshot_tip = entry.snapshot_tip
    submission.registry.release(branch, detach_waiters=True)
    submission.registry.acquire(
        branch, req.task_id, req.result,
        request_id=req.request_id,
        source='mcp',
        submitted_tip=req.snapshot_tip,
        snapshot_tip=req.snapshot_tip,
    )
    if original_was_divergent:
        logger.warning(
            'coalesce_or_enqueue_merge_request: REPLACE across '
            'DIVERGENT force-push — new tip %s diverges from the '
            'dropped queued %s (request_id=%r) for branch %r; '
            'replacing (D2).',
            new_tip[:12],
            (old_snapshot_tip or '?')[:12], old_request_id, branch,
        )
    else:
        logger.info(
            'coalesce_or_enqueue_merge_request: REPLACE — new tip '
            '%s supersedes the dropped queued %s for branch %r; '
            'dispatching fresh (D2).',
            new_tip[:12],
            (old_snapshot_tip or '?')[:12], branch,
        )


async def _reap_replaced_worktree(
    submission: _Submission, git_ops: _FindInflightWorktreeP | None,
) -> None:
    """C1: clean the dropped queued entry's scratch worktree via the
    lease-guarded primitive (skips rather than yanks a live tree if a verify
    somehow holds the lease)."""
    if git_ops is None:
        return
    wt = await git_ops.find_inflight_merge_worktree(submission.branch)
    if wt is None:
        return
    outcome = await git_ops.remove_merge_worktree_guarded(
        wt, reason='c3_replace_drop_queued',
    )
    if submission.event_store is not None:
        submission.event_store.emit(
            EventType.worktree_reaped,
            task_id=submission.req.task_id,
            phase='merge',
            data={
                'branch': submission.branch,
                'path': str(wt),
                'reason': 'c3_replace_drop_queued',
                'outcome': outcome,
            },
        )


def _coalesce_onto_entry(
    submission: _Submission, entry: _InFlightEntry | None,
) -> MergeDispatchResult:
    """Coalesce the submission onto the branch's in-flight entry.

    The caller's request_id is registered as an alias onto the primary
    entry's request_id, so a poll on the coalesced id resolves to the primary
    terminal outcome (the coalesced request never gets its own terminal
    record).  A stale slot is never coalesced onto: it is reaped and
    dispatched fresh, and that request gets its own terminal record.
    """
    branch = submission.branch
    eta = submission.registry.eta_seconds(branch)
    retention = submission.retention
    if retention is not None and entry is not None and entry.request_id is not None:
        retention.record_alias(submission.req.request_id, entry.request_id)
    submission.registry.record_coalesced(branch, submission.req.request_id)
    emit_merge_coalesced(submission.event_store, submission.req, source='registry', eta=eta)
    return MergeDispatchResult(
        dispatched=False,
        in_flight=True,
        branch=branch,
        inflight_task_id=entry.task_id if entry else None,
        eta_seconds=eta,
        source='registry',
        inflight_request_id=entry.request_id if entry else None,
    )


async def _disk_scan_path(
    submission: _Submission, git_ops: _FindInflightWorktreeP, liveness_secs: int,
) -> MergeDispatchResult | None:
    """Step 2: the on-disk worktree scan (crash-safety / cross-actor).

    An alive ``_merge-*`` worktree for the branch coalesces; a corpse is
    reaped and None returned, so the caller dispatches fresh.
    """
    branch = submission.branch
    wt = await git_ops.find_inflight_merge_worktree(branch)
    if wt is None:
        return None
    reap_reason, reap_detail = _disk_scan_reap_reason(
        wt, branch, submission.live_snapshot, liveness_secs,
    )
    if reap_reason is None:
        # ALIVE: coalesce without enqueuing or reaping.
        # No alias is registered here: the primary request_id belongs to
        # a different process, so there is no in-process registry entry
        # to alias onto.  Callers polling the coalesced id fall through
        # to the event-store / git-authority tiers on a ring miss.
        emit_merge_coalesced(submission.event_store, submission.req, source='worktree', eta=None)
        return MergeDispatchResult(
            dispatched=False,
            in_flight=True,
            branch=branch,
            source='worktree',
        )
    await _reap_corpse_worktree(submission, git_ops, wt, reap_reason, reap_detail)
    return None


def _disk_scan_reap_reason(
    wt: Path,
    branch: str,
    live_snapshot: Callable[[], dict] | None,
    liveness_secs: int,
) -> tuple[str | None, str]:
    """Why *wt* is a corpse to reap, as ``(reason, detail)``; ``(None, '')`` when alive.

    Two independent corpse tests, evaluated as ALTERNATIVES (not nested
    cases):

    'finalized_owner' — owned by THIS worker (so its root mtime is
      heartbeat-pinned by _touch_owned_merge_worktrees and carries no
      information about merge liveness) while the live pipeline holds no
      entry for the branch: the owning request already finalized.  Without
      this test the mtime test below coalesces onto the corpse forever and
      the branch can never re-enter the merge queue until the process
      restarts.
    'stale_inflight' — the pre-existing age test: an owner that died long
      enough ago that even an unpinned mtime went cold.
    """
    if _inflight_worktree_is_stale(wt, branch, live_snapshot):
        return 'finalized_owner', (
            'owned by this worker but no live entry for the branch — its '
            'request already finalized; root mtime is heartbeat-pinned and '
            'cannot be trusted'
        )
    try:
        age = time.time() - wt.stat().st_mtime
    except OSError:
        age = 0.0  # stat failed — treat as alive to be safe
    if age > liveness_secs:
        return 'stale_inflight', f'age={age:.0f}s > liveness={liveness_secs}s'
    return None, ''


async def _reap_corpse_worktree(
    submission: _Submission,
    git_ops: _FindInflightWorktreeP,
    wt: Path,
    reap_reason: str,
    reap_detail: str,
) -> None:
    """Reap the corpse worktree *wt*, and record whether it actually went.

    `cleanup_merge_worktree` IS the lease-guarded path: it routes through
    `remove_merge_worktree_guarded` (git_ops.py:8875-8878), so a tree whose
    merge-verify flock is held by a LIVE holder is SKIPPED, not yanked — the
    same C1 protection the C3 REPLACE reap relies on.  It adds only a
    crash-safe filesystem fallback on the primitive's ``'failed'`` outcome,
    which by construction means the lease acquire already confirmed no live
    holder.  That matters most for 'finalized_owner', which (unlike the
    10800 s mtime arm) can fire milliseconds after the owner retired its
    item.  Foreign-process killing is deliberately out of scope — there is
    no cwd-based kill utility in the repo (terminate_process_group only
    handles procs the orchestrator spawned), and `git worktree remove
    --force` removes the tree so any orphaned build procs fail when their
    cwd vanishes.

    NB: this gate holds no worker handle, so it cannot drop the reaped path
    from `_owned_merge_worktrees` directly.  A 'finalized_owner' reap
    therefore leaves a phantom ledger entry for at most one heartbeat tick
    (~30 s), after which `_touch_owned_merge_worktrees` hits ENOENT and
    deregisters it.  Harmless in the interim: the exemption it grants
    (`keep_worktrees`) names a path that no longer exists, and the fresh
    merger gets a new `_merge-<uuid>` path, so the phantom can neither
    shadow it nor re-wedge the branch.
    """
    logger.warning(
        'coalesce_or_enqueue_merge_request: reaping %s _merge-* worktree '
        '%s for branch %r (%s). Dispatching fresh.',
        reap_reason, wt, submission.branch, reap_detail,
    )
    await git_ops.cleanup_merge_worktree(wt)
    # Observed post-condition, not the primitive's return value:
    # cleanup_merge_worktree returns None and may apply its own fallback
    # after the primitive returns, so "is the tree actually gone?" is the
    # only honest outcome to record.  False means the removal was skipped
    # (a live holder still owns the lane's flock) or failed — the reap is
    # then a no-op and this call dispatches alongside a surviving tree,
    # which the I6 audit/orphan reaper collect later.  Recorded rather
    # than silently dropped (no-silent-fail-soft).
    reap_removed = not wt.exists()
    if not reap_removed:
        logger.warning(
            'coalesce_or_enqueue_merge_request: %s reap did NOT remove %s '
            '(lease held by a live verify, or removal failed) — dispatching '
            'fresh alongside the surviving tree; the orphan reaper collects it',
            reap_reason, wt,
        )
    if submission.event_store is not None:
        submission.event_store.emit(
            EventType.worktree_reaped,
            task_id=submission.req.task_id,
            phase='merge',
            data={
                'branch': submission.branch,
                'path': str(wt),
                'reason': reap_reason,
                'removed': reap_removed,
            },
        )


async def _acquire_and_enqueue(submission: _Submission) -> MergeDispatchResult:
    """Step 3: atomically claim the branch's slot and dispatch.

    A concurrent dispatch that won the slot during the disk-scan await
    absorbs this submission instead.
    """
    req = submission.req
    if submission.registry.acquire(
        submission.branch, req.task_id, req.result, request_id=req.request_id,
    ):
        return await _dispatch(submission)
    return _coalesce_onto_entry(submission, submission.registry.entry(submission.branch))


async def _dispatch(submission: _Submission) -> MergeDispatchResult:
    """Enqueue the submission, which already holds its branch's slot."""
    await _enqueue_holding_slot(
        submission.queue, submission.req, submission.event_store,
        submission.registry, submission.retention,
    )
    return MergeDispatchResult(
        dispatched=True,
        in_flight=False,
        branch=submission.branch,
    )
