"""Startup liveness-margin guard, verify-host-unreachable alarms, and
persistent-worktree operational guards (MQ-refactor task γ).

Extracted verbatim from :mod:`orchestrator.merge_queue`: the startup
liveness-margin guard (heartbeat-floor vs. reaper-window safety check), the
verify-host-unreachable alarm/recovery helpers, and the persistent
warm-merge-verify-worktree serial-lane guards.  These three subsystems are
folded into one module as "operational guards" (PRD Open Q5 resolution —
see plan.json design_decisions) rather than split into their own modules,
since none is individually large and all three gate/monitor worker-level
operational health rather than verify-parity detection (the shadow/drift
family in :mod:`orchestrator.merge_shadow` / :mod:`orchestrator.merge_drift`).
``merge_queue`` re-exports every name here through a top-level shim so
existing importers (``from orchestrator.merge_queue import X``, etc.) keep
working unchanged.

A moved function that reads or calls a merge_queue-resident sibling —
whether a function (``check_merge_liveness_margin``) or a module-level
CONSTANT (``_HEARTBEAT_POLL_S``, ``TOUCH_MISS_TOLERANCE``,
``INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS``, ``_MERGE_AHEAD_BOUND``) that
stays in ``merge_queue.py`` and is monkeypatched by the existing test suite
via the string path ``orchestrator.merge_queue.<name>`` — resolves it
through a function-local (deferred) import from :mod:`orchestrator.merge_queue`
rather than a direct intra-module reference.  This mirrors the
``_main_health_fingerprint`` convention in ``merge_queue.py`` and keeps this
module free of any top-level import of ``merge_queue`` (which would deadlock
module load, since merge_queue's shim needs this module fully defined
first).

Engine-constant default-argument hazard: ``liveness_secs`` (on
``check_merge_liveness_margin`` / ``enforce_merge_liveness_margin``) and
``merge_ahead_bound`` (on ``enforce_persistent_worktree_serial_lane``)
default to a merge_queue-resident constant in the original code.  Default
values are evaluated at *def time* (module import), so defaulting to a bare
``INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`` / ``_MERGE_AHEAD_BOUND`` reference
here would require a top-level ``import orchestrator.merge_queue`` —
deadlocking module load, since merge_queue's shim needs this module fully
defined first.  Each default is therefore a ``None`` sentinel, resolved
in-body via the same deferred-import reach-back (identical effective
defaults: 10800 / 1).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.git_ops import GitOps
from orchestrator.merge_types import MergeRequest

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

logger = logging.getLogger('orchestrator.merge_queue')


# ---------------------------------------------------------------------------
# Merge-worktree CONTENT-mtime no-progress signal (task 2420 / DEFECT 1,
# split from 2357; extends #1728)
# ---------------------------------------------------------------------------


def newest_content_mtime(root: Path) -> float | None:
    """Return the newest mtime among files/dirs STRICTLY BELOW *root*.

    Pure, heartbeat-immune liveness signal backing the merge-worktree
    no-progress budget in
    :meth:`orchestrator.merge_queue.SpeculativeMergeWorker._run_inflight_verify`.
    ``root`` itself is never stat'd, and any ``.git`` subtree is pruned from
    the walk entirely — so the #1728 alpha owner-heartbeat
    (``_touch_owned_merge_worktrees``), which calls ``os.utime`` on every
    owned ``_merge-*`` worktree's ROOT inode every ``_HEARTBEAT_POLL_S``
    seconds, can never mask a dead/hung verify subprocess that has stopped
    writing under the worktree (e.g. to ``target/``).

    Best-effort: a per-entry ``OSError`` (vanished mid-scan, unreadable,
    broken symlink) is swallowed and that entry is simply skipped rather
    than raised. A missing *root*, an empty *root*, or a *root* whose scan
    fails entirely all return ``None`` — never raises.

    Cost (reviewer finding, task 2420 amend): this is a full recursive
    directory walk plus a ``stat()`` per entry — O(number of files/dirs
    below *root*), unavoidable for a "what is the true newest mtime"
    query without OS-level change notification (inotify) support. The
    caller (``SpeculativeMergeWorker._run_inflight_verify``) invokes this
    once per ``INFLIGHT_VERIFY_PROGRESS_PROBE_SECS`` for the duration of
    every LOCAL in-flight verify, so its probe cadence assumes this walk
    stays cheap relative to the verify — i.e. the worktree's content tree
    (build/test output, excluding ``.git``) has at most thousands, not
    millions, of entries. A repo with a much larger generated tree should
    widen the caller's probe interval accordingly.

    Args:
        root: The merge worktree root to scan.

    Returns:
        The newest ``st_mtime`` found strictly below *root*, or ``None``
        when there is no such content (missing/empty root, or every entry
        failed to stat).
    """
    newest: float | None = None
    try:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            # Prune .git in place so os.walk never descends into it (and it
            # is excluded from the current directory's own entries below).
            dirnames[:] = [d for d in dirnames if d != '.git']
            current_dir = Path(dirpath)
            for name in (*dirnames, *filenames):
                try:
                    mtime = (current_dir / name).stat().st_mtime
                except OSError:
                    # Vanished mid-scan / unreadable / broken symlink — skip.
                    continue
                if newest is None or mtime > newest:
                    newest = mtime
    except OSError:
        # Top-level scan failure (e.g. root exists but is unreadable).
        return None
    return newest


# ---------------------------------------------------------------------------
# Startup liveness-margin guard (task 1674)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MergeLivenessAssessment:
    """Return value from :func:`check_merge_liveness_margin`.

    All fields are informational; callers should treat :attr:`safe` as the
    primary decision bit and log/display the numeric fields for triage.

    **Heartbeat-floor model (task 1729 / β)**

    Since the α owner-heartbeat (task 1728) touches every owned ``_merge-*``
    worktree every :data:`_HEARTBEAT_POLL_S` seconds, the worst frozen age
    for a *live* worker's worktrees is bounded by the stall budget
    ``_HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE``, independent of K,
    cold timeout, or num_hosts.  The old ``timeout_secs / merge_ahead_bound /
    num_hosts / max_verify_timeouts`` fields are therefore removed.

    Attributes:
        worst_case_secs: Heartbeat floor — ``_HEARTBEAT_POLL_S ×
            TOUCH_MISS_TOLERANCE`` (seconds); invariant across config.
        threshold_secs: Safety threshold (``safety_factor * liveness_secs``);
            the guard fires when ``worst_case_secs >= threshold_secs``.
        liveness_secs: The reaper's liveness window passed to the guard.
        heartbeat_poll_secs: Value of :data:`_HEARTBEAT_POLL_S` at call time
            (for diagnostics and monkeypatch verification).
        touch_miss_tolerance: Value of :data:`TOUCH_MISS_TOLERANCE` at call
            time (for diagnostics and monkeypatch verification).
        safety_factor: Fraction of *liveness_secs* used as the threshold.
        safe: True iff ``worst_case_secs < threshold_secs``.
    """

    worst_case_secs: float
    threshold_secs: float
    liveness_secs: float
    heartbeat_poll_secs: float
    touch_miss_tolerance: int
    safety_factor: float
    safe: bool


def check_merge_liveness_margin(
    config: OrchestratorConfig,
    *,
    liveness_secs: float | None = None,
    safety_factor: float = 0.75,
    logger: logging.Logger = logger,
) -> MergeLivenessAssessment:
    """Evaluate whether the heartbeat floor fits safely within the reaper's
    liveness window and emit a WARNING when it does not.

    Called once at orchestrator startup (from ``Harness._start_merge_worker``)
    against the live per-project :class:`OrchestratorConfig`.

    **Physical model (task 1729 / β — heartbeat-floor)**

    The α owner-heartbeat (task 1728) touches every owned ``_merge-*``
    worktree's mtime every :data:`_HEARTBEAT_POLL_S` seconds.  Under normal
    operation a live worker's worktrees never age past ~1 poll period.  A
    sustained event-loop stall (GIL contention, heavy I/O, OS scheduling
    jitter) can delay the heartbeat, but the stall budget is:

    .. code-block:: text

        floor_secs = _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE

    K (verify pool size), cold timeout, and num_hosts drop out of the formula
    because the heartbeat model makes them irrelevant: a live worker touches
    its owned worktrees regardless of how many verifiers are in flight or how
    long each verify takes.

    **Formula**

    .. code-block:: text

        worst_case_secs = _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE  # read as module globals
        threshold_secs  = safety_factor × liveness_secs
        safe            = worst_case_secs < threshold_secs

    Both :data:`_HEARTBEAT_POLL_S` and :data:`TOUCH_MISS_TOLERANCE` are read
    in the function body (not as default-argument values) so that a
    harness-level test can monkeypatch either constant and have the change
    reflected at call time.

    **Default calibration**

    With ``_HEARTBEAT_POLL_S=30``, ``TOUCH_MISS_TOLERANCE=20``,
    ``safety_factor=0.75``, and ``liveness_secs=10800``:

    - floor = 600 s
    - threshold = 8100 s
    - margin = 13.5× ≥ 3× (PRD minimum)
    - All shipped configs (including ``merge_verify_cold_command_timeout_secs
      =9000``) are now safe because cold timeout no longer feeds the formula.

    Args:
        config: Live per-project orchestrator config.  Kept as the first
            positional parameter for call-signature stability; the heartbeat
            floor does not read config fields.
        liveness_secs: Override for :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`.
            The corrective operator lever: raise this to widen the safety
            threshold.
        safety_factor: Fraction of *liveness_secs* that constitutes the
            "comfortably below" threshold.  Default 0.75.
        logger: Logger to use for the WARNING (default: module logger, captured
            by ``pytest caplog`` as ``orchestrator.merge_queue``).

    Returns:
        :class:`MergeLivenessAssessment` with all computed values and the
        ``safe`` verdict.
    """
    # Reach-back (deferred import): _HEARTBEAT_POLL_S, TOUCH_MISS_TOLERANCE,
    # and INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS stay in merge_queue.py (engine
    # constants) and the existing test suite monkeypatches them by string path
    # at orchestrator.merge_queue.<name>.  Resolving them dynamically from
    # merge_queue's namespace at call time keeps those patches effective post-
    # extraction — see the module docstring's reach-back convention and its
    # note on the engine-constant default-argument hazard.
    from orchestrator.merge_queue import (
        _HEARTBEAT_POLL_S,
        INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
        TOUCH_MISS_TOLERANCE,
    )

    if liveness_secs is None:
        liveness_secs = INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS

    # Heartbeat-floor formula.  Read _HEARTBEAT_POLL_S and TOUCH_MISS_TOLERANCE
    # as module globals here (not as default-arg values) so that monkeypatching
    # either constant inside a test is reflected at call time.
    _poll = _HEARTBEAT_POLL_S
    _tolerance = TOUCH_MISS_TOLERANCE

    worst_case_secs = _poll * _tolerance
    threshold_secs = safety_factor * liveness_secs
    safe = worst_case_secs < threshold_secs

    assessment = MergeLivenessAssessment(
        worst_case_secs=worst_case_secs,
        threshold_secs=threshold_secs,
        liveness_secs=liveness_secs,
        heartbeat_poll_secs=_poll,
        touch_miss_tolerance=_tolerance,
        safety_factor=safety_factor,
        safe=safe,
    )

    if not safe:
        logger.warning(
            'check_merge_liveness_margin: heartbeat floor (%.0fs = '
            'heartbeat_poll=%.0fs × touch_miss_tolerance=%d) is not '
            'comfortably below the reaper liveness threshold '
            '(%.0fs, factor=%.2f, liveness=%.0fs). '
            'Raise INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS or lower '
            'TOUCH_MISS_TOLERANCE to reduce the floor.',
            worst_case_secs,
            _poll,
            _tolerance,
            threshold_secs,
            safety_factor,
            liveness_secs,
        )

    return assessment


class MergeLivenessConfigError(Exception):
    """Raised by :func:`enforce_merge_liveness_margin` when the heartbeat
    floor (``_HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE``) is not comfortably
    below the reaper liveness threshold, indicating that startup should be
    refused.

    The exception message names the heartbeat model and the corrective levers
    (raise :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` or lower
    :data:`TOUCH_MISS_TOLERANCE`) for operator triage.  Cold timeout and
    merge-ahead bound are NOT named because the heartbeat model decouples them.
    """


def enforce_merge_liveness_margin(
    config: OrchestratorConfig,
    *,
    liveness_secs: float | None = None,
    safety_factor: float = 0.75,
    logger: logging.Logger = logger,
) -> MergeLivenessAssessment:
    """Fail-closed wrapper around :func:`check_merge_liveness_margin`.

    Delegates entirely to :func:`check_merge_liveness_margin` (which remains
    WARNING-only / fail-open) and raises :exc:`MergeLivenessConfigError` when
    the assessment is ``not safe``, causing startup to be refused.

    Args:
        config: Live per-project orchestrator config.  Forwarded verbatim to
            :func:`check_merge_liveness_margin`.
        liveness_secs: Override for :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`.
            The corrective operator lever: raise this to widen the safety
            threshold.
        safety_factor: Fraction of *liveness_secs* that constitutes the
            threshold.  Default 0.75.
        logger: Logger forwarded to :func:`check_merge_liveness_margin`.

    Returns:
        :class:`MergeLivenessAssessment` when the config is safe.

    Raises:
        :exc:`MergeLivenessConfigError`: When ``worst_case_secs >= threshold_secs``
            (i.e. heartbeat floor ≥ threshold).
    """
    # Reach-back (deferred import): check_merge_liveness_margin is a co-moved
    # sibling that the existing test suite monkeypatches by string path at
    # orchestrator.merge_queue.check_merge_liveness_margin (this function
    # used to live in merge_queue.py, alongside its sibling).
    # INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS is the same engine-constant
    # default-argument hazard as in check_merge_liveness_margin above.
    # Resolving both dynamically from merge_queue's namespace at call time
    # keeps those patches effective post-extraction.
    from orchestrator.merge_queue import (
        INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
        check_merge_liveness_margin,
    )

    if liveness_secs is None:
        liveness_secs = INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS

    assessment = check_merge_liveness_margin(
        config,
        liveness_secs=liveness_secs,
        safety_factor=safety_factor,
        logger=logger,
    )
    if not assessment.safe:
        raise MergeLivenessConfigError(
            f'enforce_merge_liveness_margin: startup refused — heartbeat floor '
            f'({assessment.worst_case_secs:.0f}s = heartbeat_poll='
            f'{assessment.heartbeat_poll_secs:.0f}s × touch_miss_tolerance='
            f'{assessment.touch_miss_tolerance}) is not below the reaper '
            f'liveness threshold ({assessment.threshold_secs:.0f}s, '
            f'factor={safety_factor:.2f}, liveness={assessment.liveness_secs:.0f}s). '
            f'Raise INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS or lower '
            f'TOUCH_MISS_TOLERANCE to reduce the floor.'
        )
    return assessment


# ---------------------------------------------------------------------------
# Persistent warm merge-verify worktree — serial-lane startup guard
# ---------------------------------------------------------------------------


class PersistentWorktreeConfigError(Exception):
    """Raised by :func:`enforce_persistent_worktree_serial_lane` when the
    persistent warm merge-verify worktree is enabled but the per-host
    in-flight verify count would exceed 1, which would risk concurrent cargo
    invocations on a single shared ``target/`` directory.

    The exception message names the bound, host count, and per-host in-flight
    count so the operator knows what to change (lower merge_ahead_bound,
    increase num_hosts, or disable ``git.persistent_merge_worktree``).
    """


def _safety_valve_due(attempt_count: int, every_n: int) -> bool:
    """Return True when the periodic cold-verify safety valve should fire.

    The safety valve (PRD §10 invariant 6) bypasses the warm-worktree swap on
    every Nth verifying serial attempt so that a true from-scratch cold verify
    runs in a throwaway ephemeral worktree (target NOT retained).  A cold
    failure on the serial lane surfaces through the existing verify-failure
    escalation path.

    Args:
        attempt_count: The 1-based count of verifying attempts on this worker
            (incremented in ``_verify_and_advance`` before calling this).
        every_n: From ``config.git.persistent_merge_worktree_safety_valve_every_n``.
            0 or negative → disabled (always returns False).

    Returns:
        True when the valve is enabled (``every_n > 0``) and
        ``attempt_count`` is a positive multiple of ``every_n``.
    """
    return every_n > 0 and attempt_count > 0 and attempt_count % every_n == 0


# Prefix for per-host verify-unreachable escalation sentinels (task 1795).
# Each host gets its own sentinel: ``__verify_host_unreachable__<host>``.
# This keeps divergence-quarantine alarms (DriftDetector) and
# unreachability alarms (RunnerUnavailable) in separate dedup namespaces.
_VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX = '__verify_host_unreachable__'


# Distinct prefix for recovery info sentinels — kept separate from the
# unreachable-alarm prefix so a host literally named ``'recovered__X'``
# cannot alias the unreachable sentinel for host ``'X'``
# (``'__verify_host_unreachable__recovered__X'`` vs
#  ``'__verify_host_recovered__X'`` — no collision).
_VERIFY_HOST_RECOVERED_SENTINEL_PREFIX = '__verify_host_recovered__'


# Sentinel task_id for the merge-worker supervisor loop-death escalation.
# Per-loop deaths use this as a base key; terminal cap-exceeded escalations
# append ':terminal' so they form a distinct dedup namespace.  Not dedup'd
# via has_open_l1 — the restart cap bounds volume and "prefer loud escalation
# over silent degradation" is the guiding policy.
_MERGE_WORKER_LOOP_DIED_SENTINEL = '__merge_worker_loop_died__'


def _verify_host_unreachable_sentinel(host: str) -> str:
    """Return the per-host dedup sentinel task_id for unreachability alarms."""
    return f'{_VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX}{host}'


def _alarm_verify_host_unreachable(
    escalation_queue: Any,
    host: str,
    reason: str,
    *,
    streak: int,
    duration_s: float,
    event_store: Any = None,
) -> None:
    """Submit a dedup'd L1 escalation when a remote verify host is persistently unreachable.

    Fires at most ONCE per downtime episode per host (dedup'd via
    ``has_open_l1(sentinel)``).  Recovery clears the open L1 via
    :func:`_clear_verify_host_unreachable`.

    The escalation is:

    * ``level=1`` (L1 blocking) — loud (steward→auto-watcher→human ladder)
      but NON-halting: the merge-halt gate fires only for categories
      ``wip_conflict`` / ``unmerged_state``; ``verify_host_unreachable`` is
      intentionally excluded so the serial-local fallback keeps flowing.
    * ``category='verify_host_unreachable'``
    * ``task_id=_verify_host_unreachable_sentinel(host)``

    None-safe: returns immediately when *escalation_queue* is None.
    Dedup: returns immediately when an open L1 already exists for the sentinel.

    Args:
        escalation_queue: Live escalation queue or ``None``.
        host: Remote runner name (e.g. ``'leo-laptop'``).
        reason: ``str(exc)`` from the most-recent ``RunnerUnavailable`` exception.
        streak: Consecutive RunnerUnavailable failure count for this episode.
        duration_s: Seconds since the first failure in this episode.
        event_store: Optional event store; when provided a
            ``EventType.verify_host_unreachable`` event is emitted.
    """
    if escalation_queue is None:
        return

    sentinel = _verify_host_unreachable_sentinel(host)

    # Dedup: don't re-alarm while an open L1 already exists for this host.
    if escalation_queue.has_open_l1(sentinel):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    minutes = duration_s / 60.0
    duration_str = f'{minutes:.1f} min' if duration_s < 3600 else f'{duration_s / 3600:.1f} h'
    summary = (
        f'Remote verify host {host!r} unreachable for {duration_str} '
        f'({streak} consecutive RunnerUnavailable failures): {reason}'
    )
    detail = (
        f'Host: {host}\n'
        f'Reason: {reason}\n'
        f'Consecutive failures: {streak}\n'
        f'Duration since first failure: {duration_str}\n'
        '\n'
        f'The remote verify runner {host!r} has been quarantined after '
        f'{streak} consecutive RunnerUnavailable failures.  '
        'The orchestrator is falling back to serial-local verification; '
        'throughput is degraded but correctness is preserved.\n'
        '\n'
        'Auto-reprobe is active: when the host becomes reachable again '
        '(ssh health check passes) the quarantine is cleared automatically '
        'and this alarm is resolved without a restart.'
    )

    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-verify-host-monitor',
        severity='blocking',
        level=1,
        category='verify_host_unreachable',
        summary=summary,
        detail=detail,
        suggested_action=(
            f'Check SSH connectivity to {host!r}.  '
            'The auto-reprobe loop will clear the quarantine and resolve this '
            'alarm once the host responds to `ssh -o BatchMode=yes -o '
            'ConnectTimeout=10 <host> true`.  '
            'To manually re-engage: fix SSH access and the orchestrator will '
            'recover automatically on the next reprobe cycle.'
        ),
    )
    escalation_queue.submit(esc)

    if event_store is not None:
        from orchestrator.event_store import EventType
        event_store.emit(
            EventType.verify_host_unreachable,
            data={'host': host, 'reason': reason, 'streak': streak, 'duration_s': duration_s},
        )


def _clear_verify_host_unreachable(
    escalation_queue: Any,
    event_store: Any,
    host: str,
    *,
    downtime_s: float,
) -> None:
    """Resolve any open unreachability alarm and emit a recovery event.

    Called from :meth:`_reprobe_quarantined_hosts` when a previously unreachable
    remote verify host responds to an SSH health check again.  Performs three
    actions:

    1. **Resolve open L1**: looks up all pending escalations for the per-host
       sentinel (via ``get_by_task(sentinel, status='pending')``) and resolves
       each one with a recovery message.
    2. **Emit recovery event** *(conditional)*: emits
       ``EventType.verify_host_recovered`` when an event store is provided,
       **only if an alarm was actually open for this host**.
    3. **Submit info escalation** *(conditional)*: submits a ``severity='info'``,
       ``level=0`` informational escalation naming the host and the downtime
       duration, **only if an alarm was actually open for this host**.

    Steps 2 and 3 are gated on whether the escalation queue reports an open L1
    for the per-host sentinel (checked via ``has_open_l1`` before resolving).
    This suppresses spurious recovery noise for sub-threshold blips — a host
    that flapped briefly without ever crossing *escalate_after_n* or
    *escalate_after_secs* never filed an alarm, so no recovery record is needed.

    None-safe: returns immediately when *escalation_queue* is None.

    Args:
        escalation_queue: Live escalation queue or ``None``.
        event_store: Optional event store; when provided a
            ``EventType.verify_host_recovered`` event is emitted.
        host: Remote runner name (e.g. ``'leo-laptop'``).
        downtime_s: Seconds the host was unreachable (``now - first_unavailable_at``).
    """
    if escalation_queue is None:
        return

    sentinel = _verify_host_unreachable_sentinel(host)

    # Check whether an alarm was actually open before resolving — we read this
    # BEFORE calling resolve(), which clears the open-L1 flag in many impls.
    alarm_was_open = escalation_queue.has_open_l1(sentinel)

    # Resolve any pending L1 escalations for this host.
    pending = escalation_queue.get_by_task(sentinel, status='pending')
    for esc in pending:
        resolution = (
            f'Host {host!r} recovered automatically via SSH reprobe after '
            f'{downtime_s / 60.0:.1f} min of unavailability.'
        )
        escalation_queue.resolve(esc.id, resolution, resolved_by='orchestrator-verify-host-monitor')

    # Only emit recovery signal when an alarm was actually open for this host.
    # Sub-threshold blips (host flapped briefly without crossing either
    # escalate_after_n or escalate_after_secs) never filed an alarm, so no
    # recovery record is needed — emitting one would be spurious audit noise.
    if not alarm_was_open:
        return

    # Emit recovery event.
    if event_store is not None:
        from orchestrator.event_store import EventType
        event_store.emit(
            EventType.verify_host_recovered,
            data={'host': host, 'downtime_s': downtime_s},
        )

    # Submit an info-level recovery escalation for audit trail visibility.
    from escalation.models import Escalation  # local import — escalation optional dep

    minutes = downtime_s / 60.0
    duration_str = f'{minutes:.1f} min' if downtime_s < 3600 else f'{downtime_s / 3600:.1f} h'
    summary = (
        f'Remote verify host {host!r} recovered after {duration_str} of unavailability'
    )
    detail = (
        f'Host: {host}\n'
        f'Downtime: {duration_str}\n'
        '\n'
        f'The remote verify runner {host!r} responded to the SSH health check.  '
        'Quarantine cleared, host re-entered the live pool, and any open '
        'unreachability alarm resolved.  No restart required.'
    )

    recovery_sentinel = f'{_VERIFY_HOST_RECOVERED_SENTINEL_PREFIX}{host}'
    esc = Escalation(
        id=escalation_queue.make_id(recovery_sentinel),
        task_id=recovery_sentinel,
        agent_role='orchestrator-verify-host-monitor',
        severity='info',
        level=0,
        category='verify_host_unreachable',
        summary=summary,
        detail=detail,
    )
    escalation_queue.submit(esc)


async def _acquire_warm_verify_worktree(
    git_ops: GitOps,
    req: MergeRequest,
    merge_wt: Path | None,
    merge_commit: str,
    *,
    safety_valve_due: bool,
    speculative: bool = False,
) -> tuple[Path | None, bool]:
    """Swap the ephemeral merge worktree for the persistent warm worktree.

    Called at the top of ``_verify_and_advance`` (and ``MergeWorker._process``)
    when the persistent warm merge-verify worktree feature is enabled.  Resets
    the fixed ``_merge-verify`` worktree to *merge_commit* (retaining
    ``target/``), cleans up the now-redundant ephemeral *merge_wt*, and returns
    the warm path so that the verify, advance_main, and cleanup_merge_worktree
    calls all run on the stable fixed path.

    When the knob is OFF or the safety valve is due (PRD §10 invariant 6),
    *merge_wt* is returned unchanged so the caller proceeds with the original
    ephemeral worktree — byte-identical to today's behaviour.

    Args:
        git_ops: Live :class:`GitOps` instance.
        req: The :class:`MergeRequest` being processed (provides config).
        merge_wt: The ephemeral merge worktree path produced by
            ``merge_to_main``, or ``None`` if unavailable.
        merge_commit: The SHA of the merge commit to check out in the warm wt.
        safety_valve_due: ``True`` on the Nth verifying attempt (PRD §10
            invariant 6): bypass the swap and run a cold verify in the
            throwaway ephemeral worktree instead.
        speculative: ``True`` when the item being verified is a speculative
            merge (merged against a pending SHA, not the current main HEAD).
            When ``True`` and ``merge_spec_warm_lane_pool`` is enabled, routes
            to a ``_spec-`` warm lane instead of the serial ``_merge-verify``
            lane (reify §9.5 B9, task η/1789).

    Returns:
        A ``(path, warm)`` tuple:
        - *path*: the warm or ephemeral worktree to use for the verify;
          ``None`` only if *merge_wt* was ``None`` and no swap occurred.
        - *warm*: ``True`` when the path is a warm-seeded worktree (the
          shadow safety valve uses this to decide whether to run a cold
          shadow compare); ``False`` for cold/ephemeral paths.

        Callers route worktree release based on *warm*:
        ``warm=True`` → ``release_spec_lane`` (retains target/) for spec lanes
        or ``reset_persistent_merge_worktree`` already handled the swap;
        ``warm=False`` → ``cleanup_merge_worktree`` (ephemeral).
    """
    # ── Speculative branch: route LOCAL spec items to the _spec- warm pool ──
    # Preconditions (all must hold to use the spec pool):
    #   1. merge_spec_warm_lane_pool knob on
    #   2. item is a speculative merge (speculative=True)
    #   3. safety valve not due (inv.6: cold from-scratch on every Nth attempt)
    # On pool exhaustion or seed failure, acquire_spec_lane falls back to a
    # cold ephemeral worktree and returns warm=False (inv.6 preserved).
    if (
        req.config.git.merge_spec_warm_lane_pool
        and speculative
        and not safety_valve_due
    ):
        lane_path, warm = await git_ops.acquire_spec_lane(merge_commit)
        # Drop the ephemeral merge_wt now that the spec lane holds the commit.
        # If acquire_spec_lane returned the same path (should not happen in
        # practice) skip the cleanup to avoid removing the lane itself.
        if merge_wt is not None and merge_wt.resolve() != lane_path.resolve():
            await git_ops.cleanup_merge_worktree(merge_wt)
        return lane_path, warm

    # ── Existing serial-head path (persistent _merge-verify worktree) ──
    # Serial-head path and knob-off path remain byte-identical to pre-η.
    if not req.config.git.persistent_merge_worktree or safety_valve_due:
        return merge_wt, False

    warm_path = await git_ops.reset_persistent_merge_worktree(merge_commit)
    # The merge commit is already a reachable git object; the ephemeral worktree
    # is no longer needed — drop it immediately to free the worktree slot.
    if merge_wt is not None and merge_wt.resolve() != warm_path.resolve():
        await git_ops.cleanup_merge_worktree(merge_wt)
    return warm_path, True


def enforce_persistent_worktree_serial_lane(
    config: OrchestratorConfig,
    *,
    merge_ahead_bound: int | None = None,
    num_hosts: int = 1,
) -> None:
    """Fail-closed startup guard for the persistent warm merge-verify worktree.

    The warm worktree feature is SERIAL-LANE-ONLY per host (PRD §A invariant 4):
    a single shared ``target/`` directory is only safe when exactly one verify
    attempt runs at a time on that host.  The guard now computes the worst-case
    per-host in-flight count as ``ceil(merge_ahead_bound / num_hosts)`` and
    rejects only when that exceeds 1.

    At ``num_hosts=1`` (the default, matching the harness call site) the logic
    reduces exactly to the original ``bound != 1`` check so all pre-existing
    guard tests and the harness call site are unaffected.

    This guard is called immediately after :func:`enforce_merge_liveness_margin`
    in :meth:`Harness._start_merge_worker` so that any misconfiguration is
    caught at startup (fail-closed) rather than at the first concurrent verify.

    Args:
        config: Live per-project orchestrator config.
        merge_ahead_bound: The effective merge-ahead bound (defaults to the
            module-level :data:`_MERGE_AHEAD_BOUND`).
        num_hosts: Number of verify hosts sharing the workload (default 1).
            Set to the number of RemoteRunner hosts when operating in a
            multi-host configuration so that the per-host ceiling is computed
            correctly (e.g. K=2 across 2 hosts → per_host=1 → no raise).

    Returns:
        ``None`` when the configuration is safe (knob off OR per-host
        in-flight ≤ 1).

    Raises:
        :exc:`PersistentWorktreeConfigError`: When
            ``config.git.persistent_merge_worktree is True`` and the per-host
            in-flight count ``ceil(merge_ahead_bound / num_hosts) > 1``.
    """
    # Reach-back (deferred import): _MERGE_AHEAD_BOUND stays in merge_queue.py
    # (engine constant) and the existing test suite monkeypatches it by
    # string path at orchestrator.merge_queue._MERGE_AHEAD_BOUND — the same
    # engine-constant default-argument hazard as liveness_secs above (see the
    # module docstring's reach-back convention).
    from orchestrator.merge_queue import _MERGE_AHEAD_BOUND

    if merge_ahead_bound is None:
        merge_ahead_bound = _MERGE_AHEAD_BOUND

    if not config.git.persistent_merge_worktree:
        return None
    # max(1, ...) clamps degenerate inputs: merge_ahead_bound is always >= 1
    # in practice (harness pins _k = _MERGE_AHEAD_BOUND = 1; callers never pass
    # 0 or negative).  The clamp ensures we fail-safe (per_host_inflight stays
    # >= 1 → guard still raises for any positive bound with a single host) rather
    # than silently allowing division-by-zero or a spuriously permissive result.
    per_host_inflight = math.ceil(max(1, merge_ahead_bound) / max(1, num_hosts))
    if per_host_inflight > 1:
        raise PersistentWorktreeConfigError(
            f'enforce_persistent_worktree_serial_lane: startup refused — '
            f'git.persistent_merge_worktree is enabled but the per-host '
            f'in-flight verify count is {per_host_inflight} '
            f'(merge_ahead_bound={merge_ahead_bound}, num_hosts={num_hosts}). '
            f'The persistent warm _merge-verify worktree is serial-lane-only '
            f'per host (PRD §A invariant 4): a shared target/ is unsafe under '
            f'concurrent verify attempts on the same host. '
            f'Lower merge_ahead_bound, increase num_hosts, or disable '
            f'git.persistent_merge_worktree.'
        )
    return None
