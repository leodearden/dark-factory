"""Turn a red post-merge verify into the ``MergeOutcome`` that blocks its merge.

:func:`red_verify_outcome` answers one question for a verify that did not
pass: which blocked outcome, with which frozen reason prefix, does this
failure earn?  The arms run in a fixed order -- flock-worktree contention,
the unscoped type-check gate, persistent ENOSPC, persistent infra-transient,
a pre-existing main-HEAD break, and finally the task-fault outcome with its
failing-id, merge-skew, timeout loop-breaker, dry-run and deferred main-health
enrichments.  The caller,
``orchestrator/src/orchestrator/merge_lane/post_merge_verify.py::run_post_merge_verify``,
has already cleaned up the merge worktree.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any

from escalation.models import Escalation

from orchestrator.dry_run_unblock import investigation_in_flight, investigation_task_name
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.disposition import (
    MergeFailureDisposition,
    SkewEvidence,
    classify_merge_failure_disposition,
)
from orchestrator.merge_lane.main_health_probe import (
    MainHealthProbeHandles,
    classify_main_health_red,
    spawn_main_health_probe,
)
from orchestrator.merge_lane.outcome_facts import MergeBlockCategory
from orchestrator.merge_lane.ports import VerifyPort
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest
from orchestrator.unblock_types import BlockClass
from orchestrator.verify import (
    VerifyResult,
    cached_main_baseline_failing_ids,
    describe_new_failing_ids,
    diff_new_failures,
    red_module_prefixes_of,
)
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
from orchestrator.verify_classify import reports_disk_full
from orchestrator.verify_runner import (
    UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY,
    is_flock_contention_failure,
    is_unscoped_gate_failure,
    unscoped_gate_failing_subprojects,
)

logger = logging.getLogger('orchestrator.merge_lane')


TRANSIENT_INFRA_REASON_PREFIX = 'Transient infrastructure failure'
"""Cause-neutral prefix of every transient-infra hold's ``MergeOutcome.reason``:
the pre-verify disk guard, a persistent ENOSPC verify, and a persistent
``INFRA_TRANSIENT_CATEGORIES`` verify.  Every producer builds its reason with
``transient_infra_reason``, which appends the measured cause in parentheses.
The workflow routes this prefix to
``_mark_blocked(escalate_to_human=True, category='infra_issue')`` — the
infra_issue category plus the durable ref lets the escalation-watcher
auto-resolve if the host has recovered by read-time."""

DISK_PRESSURE_CAUSE = 'disk pressure'


def transient_infra_reason(cause: str, message: str) -> str:
    """The ``'<prefix> (<cause>): <message>'`` reason of a transient-infra hold."""
    return f'{TRANSIENT_INFRA_REASON_PREFIX} ({cause}): {message}'


MAIN_HEALTH_PROBE_PENDING_NOTE = (
    'provisional: an off-critical-path main-health probe is still checking '
    'whether this failure pre-exists on bare main (task 2564); if '
    'confirmed, this task is not at fault and main is being healed '
    'separately'
)
"""Appended, in brackets, to the provisional task-fault
``MergeOutcome.reason`` returned in DEFERRED mode
(``main_health_probe_handles`` is not ``None``) whenever
:func:`spawn_main_health_probe` reports a probe is actually in flight for
this failure.

Lets a steward/human reading the surviving reason string distinguish
"still being classified off-path" from a confirmed task fault — the
SYNCHRONOUS path (``main_health_probe_handles`` is ``None``) never appends
this note because :func:`classify_main_health_red` has, by construction,
already definitively ruled out a pre-existing break by the time the
task-fault reason is built.  Not appended when the deferred probe itself
would be skipped (feature disabled / timed-out verify / a
:data:`PREEXISTING_BREAK_SKIP_CATEGORIES` category) since those cases are
confirmed task faults in both modes — see
:func:`spawn_main_health_probe`'s return value."""


def verify_hit_enospc(verify: VerifyResult) -> bool:
    """True when a failed ``VerifyResult`` bears a disk-full signature.

    Scans the human-readable failure report and each captured output stream
    (test/lint/type) with ``verify_classify.py::reports_disk_full``.
    """
    streams = (
        verify.failure_report(),
        verify.test_output,
        verify.lint_output,
        verify.type_output,
    )
    return reports_disk_full(' '.join(filter(lambda s: isinstance(s, str), streams)))


# Sentinel task_id prefix for a laptop-side flock-worktree-contention alarm
# (task 2307 β, PRD plans/laptop-warm-verify-flock-orphan-prd.md §8.2).
# Mirrors _VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX's per-host sentinel shape.
_VERIFY_WORKTREE_CONTENTION_SENTINEL_PREFIX = '__verify_worktree_contention__'


def _verify_worktree_contention_sentinel(host: str) -> str:
    """Return the per-host sentinel task_id for a flock-worktree-contention alarm."""
    return f'{_VERIFY_WORKTREE_CONTENTION_SENTINEL_PREFIX}{host}'


def _alarm_verify_worktree_contention(
    escalation_queue: Any,
    *,
    host: str,
    holder_pgid: int | None,
    waiter_pgid: int | None,
) -> None:
    """Submit a born-at-L2 escalation for a laptop-side flock-worktree-contention outcome.

    Fired when a remote verify (task 2306 α's ``make_flock_contention_result``)
    reports that ``.merge_verify.lock`` could not be acquired within the
    bounded wait — another verify invocation already holds the persistent
    warm merge-verify worktree.  The laptop CLI holds no escalation client
    (the escalation MCP server binds 127.0.0.1:8100), so the workstation
    files this on the laptop's behalf.

    The escalation is:

    * ``level=2`` / ``severity='critical'`` — born-at-L2 (PRD §8.2): routes
      straight to a human, bypassing the auto-watcher.
    * ``agent_role='orchestrator-verify-host-monitor'`` — the ``orchestrator-``
      prefix marks this as a harness sentinel so the escalation server never
      downgrades the severity (mirrors ``_emit_loop_terminal_escalation``).
    * ``category='verify_worktree_contention'``
    * ``task_id=_verify_worktree_contention_sentinel(host)``

    None-safe: returns immediately when *escalation_queue* is None.  Dedup:
    returns immediately when an open L2 already exists for this host's
    sentinel task_id.  An orphaned lock-holder (the PRD scenario) persists
    until manually killed, so without this guard every subsequent distinct
    merge routed to the host would file another L2 — a burst of duplicate
    criticals at the human while the orphan lives.  The check goes through
    ``get_by_task(sentinel, status='pending', level=2)`` rather than
    ``_alarm_verify_host_unreachable``'s ``has_open_l1``: that helper is
    hardcoded to ``level=1`` (see ``escalation/queue.py``) and would never
    match this level=2 escalation.  No event emission — the escalation is
    the sole user-observable signal (PRD §8.2).

    Args:
        escalation_queue: Live escalation queue or ``None``.
        host: Laptop host name that reported the contention.
        holder_pgid: pgid of the process holding the lock, or ``None`` when
            the holder's pgid file is absent or corrupt (α's fail-safe).
        waiter_pgid: pgid of the process that lost the race and reported.
    """
    if escalation_queue is None:
        return

    sentinel = _verify_worktree_contention_sentinel(host)

    # Dedup: don't re-alarm while an open L2 already exists for this host.
    # has_open_l1 is hardcoded to level=1 (escalation/queue.py), so it would
    # never match this level=2 escalation; get_by_task is used directly.
    if escalation_queue.get_by_task(sentinel, status='pending', level=2):
        return

    # Rendered once and reused below so the detail and suggested_action
    # fields agree on how a fail-safe None pgid reads to the operator.
    holder_display = holder_pgid if holder_pgid is not None else '<unknown>'
    waiter_display = waiter_pgid if waiter_pgid is not None else '<unknown>'

    summary = (
        f'Flock contention on {host!r}: another verify holds the persistent '
        f'merge-verify worktree lock'
    )
    detail = (
        f'Host: {host}\n'
        f'Holder pgid: {holder_display}\n'
        f'Waiter pgid: {waiter_display}\n'
        '\n'
        f'The laptop-side verify on {host!r} could not acquire '
        '.merge_verify.lock within the bounded wait, so it reported '
        'contention instead of falling back to an ephemeral worktree. '
        'The merge is blocked pending investigation.'
    )

    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-verify-host-monitor',
        severity='critical',
        level=2,
        category='verify_worktree_contention',
        summary=summary,
        detail=detail,
        suggested_action=(
            f'Inspect the holder process (pgid {holder_display}) on {host!r} — '
            'confirm it is a legitimate in-progress verify and not an orphan. '
            'Kill an orphaned holder to release .merge_verify.lock, or wait '
            'for it to finish, then re-submit the blocked merge.'
        ),
    )
    escalation_queue.submit(esc)


@dataclasses.dataclass(frozen=True)
class DryRunInvestigationHandles:
    """Opaque bundle of harness-owned handles for the merge-verify dry-run spawn.

    SpeculativeMergeWorker holds none of scheduler/mcp/usage_gate/cost_store
    itself (the harness owns all of them); this bundles them into one
    optional, default-``None`` param so the post-merge verify stays a pure
    git engine (mirrors the ``train_callback_factory`` opaque-injection
    pattern — the worker never imports the scheduler).  Only
    ``VerifyDispatcher.post_merge_verify`` (the in-flight verify
    and the single-branch gate re-verify) passes a live instance; the
    solo-reverify and train module-level callers pass nothing,
    so their spawns automatically no-op.

    ``background_tasks`` is the SAME ``set`` instance the worker stores at
    ``self._background_tasks`` — a spawned investigation task's strong ref
    lives exactly as long as the worker, mirroring
    ``workflow.TaskWorkflow._background_tasks``.
    """

    scheduler: Any
    mcp: Any = None
    usage_gate: Any = None
    cost_store: Any = None
    background_tasks: set[asyncio.Task] = dataclasses.field(default_factory=set)


def spawn_merge_verify_dry_run(
    handles: DryRunInvestigationHandles | None,
    req: MergeRequest,
    reason: str,
    detail: str,
    *,
    event_store: EventStore | None = None,
    verifier: VerifyPort,
) -> None:
    """Fire-and-forget: spawn an autonomous dry-run investigation for a
    MERGE_VERIFY_RED post-merge-verify block.

    Mirrors ``workflow._spawn_dry_run_unblock``'s fire-and-forget shape
    (try/except-wrapped ``asyncio.create_task``, strong-ref registration into
    a ``background_tasks`` set with a discard done-callback) so the
    merge-verify and agent-block investigation spawns behave identically.
    ``block_class=BlockClass.MERGE_VERIFY_RED`` is passed explicitly — the
    generic post-merge reason is deliberately not mapped by
    ``classify_block_reason`` (see ``unblock_types`` module docstring).

    None-safe: no-ops when *handles* is ``None`` or ``handles.scheduler`` is
    ``None`` — the solo-reverify and train callers pass no handles.  Also
    no-ops when ``req.config.unblock_auto.enabled`` is falsy, when an
    investigation for this task is already running anywhere on the event loop
    (mirrors ``workflow._spawn_dry_run_unblock``'s enablement and in-flight-dedup
    guards), or when ``req.worktree`` is missing/nonexistent.

    The investigation always reads ``req.worktree`` — the task's OWN retained
    worktree — rather than the ephemeral merge worktree: by the time this
    fire-and-forget task actually runs, the post-merge verify has already
    handed the merge worktree to ``cleanup_merge_worktree`` (task 2141
    step-17/18 — the no-op cleanup mock in the test suite hid this).
    ``req.worktree`` survives while the task stays blocked, and is what
    ``b3_gate`` re-checks at gate time.

    *event_store* is forwarded to ``run_dry_run_unblock`` so the
    investigation emits the same ``invocation_end``/``'blocked'`` telemetry
    event the agent-block path emits (observability parity).
    """
    if handles is None or handles.scheduler is None:
        return
    ua = getattr(req.config, 'unblock_auto', None)
    if not ua or not ua.enabled:
        return
    if investigation_in_flight(req.task_id):
        logger.debug(
            'Task %s: merge-verify dry-run investigation already in '
            'progress, skipping duplicate spawn',
            req.task_id,
        )
        return
    if req.worktree is None or not req.worktree.exists():
        logger.debug(
            'Task %s: skipping merge-verify dry-run — task worktree missing',
            req.task_id,
        )
        return
    try:
        task = asyncio.create_task(
            verifier.dry_run_unblock(
                task_id=req.task_id,
                worktree=str(req.worktree),
                reason=reason,
                detail=detail,
                scheduler=handles.scheduler,
                mcp=handles.mcp,
                config=req.config,
                event_store=event_store,
                usage_gate=handles.usage_gate,
                cost_store=handles.cost_store,
                block_class=BlockClass.MERGE_VERIFY_RED,
            ),
            name=investigation_task_name(req.task_id),
        )
        handles.background_tasks.add(task)
        task.add_done_callback(handles.background_tasks.discard)
    except Exception as exc:
        logger.warning(
            'Task %s: failed to spawn merge-verify dry-run investigation: %s',
            req.task_id, exc,
        )


def _render_skew_surfaces(
    disposition: MergeFailureDisposition,
    evidence: SkewEvidence | None,
) -> tuple[str, dict[str, str] | None]:
    """Render the I4 reason_suffix + failure_diagnostic surfaces for a
    merge-skew attribution verdict (task 2383 β, M2 of
    plans/merge-skew-attribution-prd.md).

    Pure, no I/O.  Only ``INTEGRATION_SKEW`` with non-``None`` *evidence*
    yields a non-empty ``reason_suffix`` — appended to the task-fault reason
    so the debugger's dry-run context reads "port landed commit <sha>
    touching <files> — do not hunt your own diff" — and a
    ``failure_diagnostic`` ``dict[str, str]`` surfaced verbatim via
    ``merge_status`` (I4).  Every other ``(disposition, evidence)``
    combination — ``MAIN_RED`` / ``BRANCH_BUG`` / ``INDETERMINATE``, or
    ``evidence=None`` — returns ``('', None)``.
    """
    if disposition is not MergeFailureDisposition.INTEGRATION_SKEW or evidence is None:
        return '', None
    commits = ', '.join(evidence.implicated_commits)
    files = ', '.join(evidence.overlap_files)
    tests = ', '.join(evidence.failing_tests)
    reason_suffix = (
        f'integration_skew: port landed commit(s) {commits} touching '
        f'{files} — do not hunt your own diff'
    )
    failure_diagnostic = {
        'disposition': disposition.value,
        'implicated_commits': commits,
        'overlap_files': files,
        'failing_tests': tests,
    }
    return reason_suffix, failure_diagnostic


_SkewAttribution = tuple[
    MergeFailureDisposition, dict[str, str] | None, str, SkewEvidence | None,
]
_UNATTRIBUTED: _SkewAttribution = (MergeFailureDisposition.INDETERMINATE, None, '', None)


async def _classify_disposition_for_outcome(
    verify: VerifyResult,
    *,
    req: MergeRequest,
    merge_base_sha: str,
    main_sha: str,
    event_store: EventStore | None,
    real_main_head_sha: str | None = None,
) -> _SkewAttribution:
    """Classify a non-preexisting merge-verify failure and render its I4
    surfaces (task 2383 β).

    Thin async wrapper: calls ``classify_merge_failure_disposition`` (with
    ``preexisting=False`` — I1: this is only ever reached on the non-
    preexisting bucket) and threads the result through
    ``_render_skew_surfaces``.

    *real_main_head_sha* (task 2869, I6): the CURRENT real published main HEAD,
    forwarded into the classifier's orphan/ancestor filter so an orphaned
    speculative *main_sha* train tip never cites its dangling commits. ``None``
    (default; caller could not resolve real main) skips the filter — today's
    pre-2869 reference frame (fail-open, additive default).

    Belt-and-suspenders fail-open (I3): any exception — including one raised
    by the classifier itself, atop its own internal fail-open — is caught
    here, logged at WARNING, and degrades to ``(INDETERMINATE, None, '', None)``
    so the caller's outcome is never left half-attached.

    The 4th element (task 3178) is the classifier's GATHERED bundle
    (``ClassificationResult.observed_evidence``), NOT its adjudicated
    ``evidence``: it is the one the step-18 ``merge_attempt`` emit needs, because
    that emit now persists evidence on the ADJUDICATED-INDETERMINATE path too.
    ``_render_skew_surfaces`` keeps receiving the ADJUDICATED bundle, so only
    INTEGRATION_SKEW ever renders the "port the landed commit" directive — that
    contract is untouched.
    """
    try:
        result = await classify_merge_failure_disposition(
            verify_result=verify,
            branch=req.branch.bare_id,
            merge_base_sha=merge_base_sha,
            main_sha=main_sha,
            preexisting=False,
            real_main_head_sha=real_main_head_sha,
            task_id=req.task_id,
            repo_root=req.config.project_root,
            event_store=event_store,
        )
        # Read the two evidence slots BY NAME (ClassificationResult): the
        # ADJUDICATED bundle goes to the renderer (only INTEGRATION_SKEW gets a
        # directive), the OBSERVED/gathered one out to the caller (task 3178).
        reason_suffix, failure_diagnostic = _render_skew_surfaces(
            result.disposition, result.evidence,
        )
        return (
            result.disposition, failure_diagnostic, reason_suffix,
            result.observed_evidence,
        )
    except Exception:
        logger.warning(
            'Task %s: _classify_disposition_for_outcome failed; degrading to '
            'INDETERMINATE (fail-open, I3)',
            req.task_id,
            exc_info=True,
        )
        return _UNATTRIBUTED


@dataclasses.dataclass(frozen=True)
class RedVerifyContext:
    """What classifying one red post-merge verify needs from its call.

    *timeouts* is the caller's per-task timeout loop-breaker counter, bumped
    in place.  *dry_run_handles* and *main_health_probe_handles* are ``None``
    for every caller but the worker's lane verify, which makes the dry-run
    spawn a no-op and runs the main-health probe synchronously.
    *merge_base_sha* and *main_sha* are the dispatch-time base facts; both
    must be set for merge-skew attribution to run.  *origin_is_local* records
    (but never steers by) where the verify ran (task 2565).
    """

    git_ops: GitOps
    req: MergeRequest
    verifier: VerifyPort
    timeouts: dict[str, int]
    max_timeouts: int
    event_store: EventStore | None
    escalation_queue: Any
    dry_run_handles: DryRunInvestigationHandles | None
    main_health_probe_handles: MainHealthProbeHandles | None
    merge_base_sha: str | None
    main_sha: str | None
    origin_is_local: bool

    def bump_timeouts(self, exhausted_message: str) -> None:
        """Count one more consecutive timeout; warn *exhausted_message* once the budget is spent."""
        new_count = self.timeouts.get(self.req.task_id, 0) + 1
        self.timeouts[self.req.task_id] = new_count
        if new_count >= self.max_timeouts:
            logger.warning(exhausted_message, self.req.task_id, new_count)

    def spawn_dry_run(self, reason: str, detail: str) -> None:
        spawn_merge_verify_dry_run(
            self.dry_run_handles, self.req, reason, detail,
            event_store=self.event_store, verifier=self.verifier,
        )


async def red_verify_outcome(verify: VerifyResult, ctx: RedVerifyContext) -> MergeOutcome:
    """The blocked outcome a red post-merge *verify* earns, arm by arm in priority order."""
    # Flock-worktree-contention sentinel (task 2307 β): a laptop-side verify
    # reported that .merge_verify.lock was already held (task 2306 α).  Checked
    # FIRST — before the unscoped-gate sentinel and the main-health probe — since
    # contention is not a pre-existing main-HEAD break; probing would be wasteful
    # and misleading.  Files a born-at-L2 escalation and keeps the merge blocked.
    if is_flock_contention_failure(verify):
        payload = verify.contention or {}
        _alarm_verify_worktree_contention(
            ctx.escalation_queue,
            host=payload.get('host', '<unknown>'),
            holder_pgid=payload.get('holder_pgid'),
            waiter_pgid=payload.get('waiter_pgid'),
        )
        return MergeOutcome(
            'blocked',
            reason=f'Post-merge verification blocked: {verify.summary} [category: {verify.category}]',
            failure_category=verify.category,
        )

    # Unscoped-gate sentinel: check this BEFORE the ENOSPC guard because the
    # sentinel's type_output field carries type-check output (gate.detail) that
    # could contain disk-full markers and be misclassified as transient-infra.
    # Returns blocked directly without running the ENOSPC retry or main-health probe.
    if is_unscoped_gate_failure(verify):
        return _unscoped_gate_outcome(verify, ctx)

    # Persistent ENOSPC after the prune-and-retry → transient infra.
    if verify_hit_enospc(verify):
        return _transient_infra_outcome(
            verify,
            DISK_PRESSURE_CAUSE,
            f'post-merge verify '
            f'still reports no space left on device after pruning '
            f'stale merge worktrees and retrying. {verify.summary}',
        )

    # Persistent classified infra-transient outcome after the bounded retry
    # (task ν, verify-scope-inversion-prd.md).  Placed before the main-health
    # probe / μ new-ids attribution / skew disposition / timeout loop-breaker /
    # dry-run spawn / generic block so this outcome consumes no merge verify
    # attempt and routes through the same loud infra_issue L1 consumer as the
    # ENOSPC arm — no steward, no branch-blaming.
    if (verify.category or '') in INFRA_TRANSIENT_CATEGORIES:
        return _transient_infra_outcome(
            verify,
            f'category={verify.category}',
            f'post-merge verify classified infra-transient '
            f'and did not clear after retry. {verify.summary}',
        )

    # Main-health probe: classify whether this failure is pre-existing on bare
    # main HEAD rather than introduced by this merge.  The merge worktree is
    # already cleaned up; the probe builds its own _mainprobe- worktree and
    # always cleans it in a finally block.
    #
    # DEFERRED mode (main_health_probe_handles is not None — only the worker's
    # lane verify): SKIP the synchronous probe.  It can run for minutes and
    # would otherwise hold the caller's verify_task — hence the merge slot /
    # host lease — for its full duration (task 2564, the reify 5067 stall).
    # The task-fault outcome spawns it as a detached background task instead.
    # SYNCHRONOUS mode (handles is None): unchanged behaviour from before 2564.
    if ctx.main_health_probe_handles is None:
        main_health_outcome = await classify_main_health_red(
            ctx.git_ops, ctx.req, verify, ctx.event_store,
            escalation_queue=ctx.escalation_queue,
        )
        if main_health_outcome is not None:
            return main_health_outcome
    return await _task_fault_outcome(verify, ctx)


def _unscoped_gate_outcome(verify: VerifyResult, ctx: RedVerifyContext) -> MergeOutcome:
    failing = ', '.join(unscoped_gate_failing_subprojects(verify))
    if verify.category == UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY:
        ctx.bump_timeouts(
            'Task %s: post-merge unscoped type-check timed out %d times in a '
            'row — next submission will be abandoned',
        )
        return MergeOutcome(
            'blocked',
            reason=(
                f'Post-merge verification failed: unscoped type-check timed out '
                f'for {failing}.'
            ),
            block_category=MergeBlockCategory.POST_MERGE_VERIFY,
        )
    reason = (
        f'Post-merge verification failed: unscoped type-check failed '
        f'for {failing}.'
    )
    detail = verify.type_output or ''
    if detail:
        reason = f'{reason}\n\n{detail}'
    ctx.spawn_dry_run(reason, detail)
    return MergeOutcome(
        'blocked', reason=reason, block_category=MergeBlockCategory.POST_MERGE_VERIFY,
    )


def _transient_infra_outcome(verify: VerifyResult, cause: str, message: str) -> MergeOutcome:
    detail = verify.failure_report()
    reason = transient_infra_reason(cause, message)
    if detail:
        reason = f'{reason}\n\n{detail}'
    return MergeOutcome('blocked', reason=reason)


async def _task_fault_outcome(verify: VerifyResult, ctx: RedVerifyContext) -> MergeOutcome:
    detail = verify.failure_report()
    reason = await _task_fault_reason(verify, ctx.git_ops)
    if detail:
        reason = f'{reason}\n\n{detail}'
    # Merge-skew attribution (task 2383 β) runs BEFORE the dry-run spawn so the
    # enriched reason (with the 'port landed commit ... do not hunt your own
    # diff' suffix) flows into the debugger's dry-run context too.
    disposition, failure_diagnostic, reason_suffix, skew_evidence = (
        await _skew_attribution(verify, ctx)
    )
    if reason_suffix:
        reason = f'{reason}\n\n{reason_suffix}'
    # Loop-breaker bookkeeping: bump only when the failure was a pure
    # timeout.  Real test/lint/type failures already bubble up to the steward
    # and don't drive the re-queue oscillation the loop-breaker catches.
    if verify.timed_out:
        ctx.bump_timeouts(
            'Task %s: post-merge verify timed out %d times in a '
            'row — next submission will be abandoned',
        )
    else:
        ctx.spawn_dry_run(reason, detail)
    # DEFERRED mode: spawn the off-critical-path main-health classification now
    # that the provisional outcome is fully built (a no-op in SYNCHRONOUS
    # mode).  Only a probe actually in flight annotates the reason as pending
    # reclassification (task 2564 amendment), so a steward can tell "still
    # being classified off-path" apart from a confirmed fault.
    probe_pending = spawn_main_health_probe(
        ctx.main_health_probe_handles, ctx.git_ops, ctx.req, verify,
        escalation_queue=ctx.escalation_queue, event_store=ctx.event_store,
        origin_is_local=ctx.origin_is_local,
    )
    if probe_pending:
        reason = f'{reason}\n\n[{MAIN_HEALTH_PROBE_PENDING_NOTE}]'
    return MergeOutcome(
        'blocked', reason=reason,
        failure_category=verify.category,
        failure_cause_hint=verify.cause_hint,
        disposition=disposition,
        failure_diagnostic=failure_diagnostic,
        skew_evidence=skew_evidence,
        block_category=MergeBlockCategory.POST_MERGE_VERIFY,
    )


async def _task_fault_reason(verify: VerifyResult, git_ops: GitOps) -> str:
    """Cite only the NEW failing ids when main's baseline is cache-warm, else the summary.

    Task μ (verify-scope-inversion-prd.md, B1).  The peek is COMPLETE for this
    verify's red modules (the whole tree when its ids are unattributed) or
    None, so "not present on main" is never claimed from a partial baseline.
    CACHE-ONLY: never a probe on the critical path (G4, task 2564); a cold
    cache or failing_test_ids=None keeps the generic wording (B3).  The
    wholly-preexisting case (new_ids empty) is routed to MAIN_HEALTH_RED by
    the main-health probe, so this enrichment never contradicts it.
    """
    new_ids, main_sha = await _new_failing_ids(verify, git_ops)
    if new_ids:
        return 'Post-merge verification failed: ' + describe_new_failing_ids(new_ids, main_sha)
    reason = f'Post-merge verification failed: {verify.summary}'
    # The category makes the timeout-vs-real-failure distinction visible in the
    # surviving human-facing signal; appended after the summary so every
    # prefix/substring reason assertion holds.
    if verify.category:
        reason = f'{reason} [category: {verify.category}]'
    return reason


async def _new_failing_ids(
    verify: VerifyResult, git_ops: GitOps,
) -> tuple[frozenset[str] | None, str]:
    """*verify*'s failing ids absent from main's cached baseline, and the main sha read."""
    if verify.failing_test_ids is None:
        return None, ''
    try:
        _current_main_sha = await git_ops.get_main_sha()  # type: ignore[union-attr]
    except Exception:
        _current_main_sha = ''
    if not _current_main_sha:
        return None, ''
    _cached_baseline = cached_main_baseline_failing_ids(
        _current_main_sha, red_module_prefixes=red_module_prefixes_of(verify),
    )
    if _cached_baseline is None:
        return None, _current_main_sha
    return diff_new_failures(verify.failing_test_ids, _cached_baseline), _current_main_sha


async def _skew_attribution(verify: VerifyResult, ctx: RedVerifyContext) -> _SkewAttribution:
    """Merge-skew attribution (task 2383 β, M2 of plans/merge-skew-attribution-prd.md).

    Absent base facts (the train and solo-reverify callers pass none) skips
    classification: disposition stays INDETERMINATE and the skew evidence
    stays None — the seam the merge_attempt emit guard keys on (task 3178).
    """
    if ctx.merge_base_sha is None or ctx.main_sha is None:
        return _UNATTRIBUTED
    return await _classify_disposition_for_outcome(
        verify, req=ctx.req, merge_base_sha=ctx.merge_base_sha,
        main_sha=ctx.main_sha, event_store=ctx.event_store,
        real_main_head_sha=await _real_main_head_sha(ctx.git_ops, ctx.req),
    )


async def _real_main_head_sha(git_ops: GitOps, req: MergeRequest) -> str | None:
    """The CURRENT real published main HEAD for the merge-skew ancestor filter (task 2869, I6).

    The classifier filters cited landings to ancestors of real main, so an
    ORPHANED speculative-train tip (main_sha frozen at dispatch) never cites
    its dangling commits as a false INTEGRATION_SKEW (reify esc-5260-8).
    Fail-safe (loud-over-silent): any get_main_sha error degrades to None —
    the filter is skipped — WARN-logged, never crashing classification.
    """
    try:
        return (await git_ops.get_main_sha()) or None
    except Exception:
        logger.warning(
            'Task %s: get_main_sha() failed while resolving real main '
            'HEAD for the merge-skew ancestor filter; degrading to None '
            '(filter skipped, fail-safe)',
            req.task_id,
            exc_info=True,
        )
        return None
