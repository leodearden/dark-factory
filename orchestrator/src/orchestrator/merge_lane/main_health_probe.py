"""Does a failed post-merge verify fail the same way on bare main?

The merge lane's main-health probe (PRD ``plans/merge-lane-quality-prd.md``
task κ). A red post-merge verify is either the task's fault or a break
already on main HEAD; this module answers which, synchronously
(:func:`classify_main_health_red`) or off the critical path
(:func:`spawn_main_health_probe`), and files the dedup'd
``preexisting_main_break`` escalation (:func:`file_main_health_escalation`).

What the lane does about a confirmed break is not decided here: a deferred
probe hands it to the ``auto_heal`` callback on
:class:`MainHealthProbeHandles`, which the worker owns
(``orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._auto_heal_main_health_deferred``).
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from escalation.dedupe import DedupeConfig, content_fingerprint_key, submit_or_dedupe
from escalation.models import Escalation

from orchestrator import workflow
from orchestrator.event_store import EventStore
from orchestrator.flake_recorder import record_flake_observation
from orchestrator.git_ops import GitOps
from orchestrator.main_health_actions import MainHealthNextAction
from orchestrator.merge_lane.disposition import MergeFailureDisposition
from orchestrator.merge_lane.gates import _emit_merge_attempt
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest, OutcomeKind
from orchestrator.verify import (
    FlakeObservationSink,
    VerifyResult,
    verify_failure_is_preexisting_on_main,
)
from orchestrator.verify_categories import PREEXISTING_BREAK_SKIP_CATEGORIES

if TYPE_CHECKING:
    from orchestrator.flake_ledger import FlakeSuppression

logger = logging.getLogger('orchestrator.merge_lane')


MAIN_HEALTH_RED_REASON_PREFIX = 'Main-health red: merge verify failure reproduces on bare main'
"""Prefix of the ``MergeOutcome.reason`` string emitted when the post-merge
verify failure is classified as a pre-existing break already present on bare
main HEAD — i.e. not introduced by this task's merge.

The workflow ``_submit_to_merge_queue`` pattern-matches this prefix and routes
the outcome to a single dedup'd L1 escalation (``escalate_to_human=True``,
``category='preexisting_main_break'``), skipping the steward entirely.  The
steward operates in-branch and cannot fix a broken main; routing each
concurrent failing merge to the steward would cause an ~85 min-per-task
livelock for every task whose verify runs against a red main."""


AutoHeal = Callable[[MergeOutcome, MergeRequest], Awaitable[None]]
"""The lane's response to a confirmed, still-fresh pre-existing break."""


def _main_health_fingerprint(category: str, cause_hint: str, probe_sha: str) -> str:
    """Compose a dedupe fingerprint for a preexisting-main-break outcome.

    Fail-safe: returns '' when the composition raises.
    """
    try:
        return workflow.compute_preexisting_main_break_fingerprint(category, cause_hint, probe_sha)
    except Exception:
        logger.warning(
            'main-health fingerprint composition failed '
            '(category=%s cause_hint=%s probe_sha=%s) — '
            'returning empty fingerprint (dedupe disabled for this outcome)',
            category, cause_hint, probe_sha,
            exc_info=True,
        )
        return ''


def _build_main_health_outcome(verify: VerifyResult, probe_sha: str) -> MergeOutcome:
    """Build the main-health-red ``MergeOutcome`` for a confirmed pre-existing break.

    Pure function — no I/O, no side effects, no event emission.  Extracted
    (task 2564) from :func:`classify_main_health_red` so the reason /
    fingerprint / ``failure_category`` / ``failure_cause_hint`` composition is
    shared verbatim between the synchronous probe path
    (``classify_main_health_red``) and the deferred off-critical-path probe
    (``_run_deferred_main_health_probe``) — the two can never diverge.

    *probe_sha* is the bare-main SHA the probe actually tested against (fed
    into the dedupe fingerprint so concurrent failing merges against the same
    main HEAD fold to one escalation parent).
    """
    detail = verify.failure_report()
    suffix = (verify.cause_hint or verify.summary or '')[:160]
    reason = (
        f'{MAIN_HEALTH_RED_REASON_PREFIX} '
        f'(category={verify.category!r}): {suffix}'
    )
    if detail:
        reason = f'{reason}\n\n{detail}'
    return MergeOutcome(
        'blocked',
        reason=reason,
        failure_category=verify.category,
        failure_cause_hint=verify.cause_hint,
        dedupe_fingerprint=_main_health_fingerprint(
            verify.category or '', verify.cause_hint, probe_sha,
        ),
        disposition=MergeFailureDisposition.MAIN_RED,
    )


def _main_probe_flake_sink(
    req: MergeRequest,
    *,
    event_store: EventStore | None,
    escalation_queue: Any,
) -> FlakeObservationSink:
    """Where the main-health classifier's task-mu fork records a flake it
    suppressed: the flake ledger, attributed to *req*'s task and to the main
    SHA the isolated re-run ran on, with the INV-4 storm streak reached
    through the recorder."""
    async def sink(observation: FlakeSuppression, main_sha: str) -> None:
        await record_flake_observation(
            observation,
            project_root=req.config.project_root,
            project_id=req.config.fused_memory.project_id,
            merge_sha=main_sha,
            task_id=req.task_id,
            event_store=event_store,
            escalation_queue=escalation_queue,
        )
    return sink


def _classification_skipped(req: MergeRequest, verify: VerifyResult) -> bool:
    """The three cheap guards every classification path applies before probing.

    - ``req.config.escalate_preexisting_main_break`` is False
    - ``verify.timed_out`` is True (non-deterministic; re-probing is wasteful)
    - ``verify.category`` is in :data:`PREEXISTING_BREAK_SKIP_CATEGORIES`
      (infra_timeout / flock_error — inherently flaky)

    One predicate, so the deferred path never probes a case the synchronous
    path would have skipped (task 2564 step-6).
    """
    return (
        not req.config.escalate_preexisting_main_break
        or verify.timed_out
        or (verify.category or '') in PREEXISTING_BREAK_SKIP_CATEGORIES
    )


async def _bare_main_fails_alike(
    git_ops: GitOps,
    req: MergeRequest,
    verify: VerifyResult,
    *,
    event_store: EventStore | None,
    escalation_queue: Any,
) -> tuple[bool, str]:
    """``(is_preexisting, probe_sha)``: whether bare main fails *verify*'s way, at which sha."""
    return await verify_failure_is_preexisting_on_main(
        req.worktree, req.config, req.module_configs, req.task_files,
        verify, git_ops,
        flake_sink=_main_probe_flake_sink(
            req, event_store=event_store, escalation_queue=escalation_queue,
        ),
    )


async def classify_main_health_red(
    git_ops: GitOps,
    req: MergeRequest,
    verify: VerifyResult,
    event_store: EventStore | None = None,
    *,
    escalation_queue: Any = None,
) -> MergeOutcome | None:
    """Probe whether *verify* is a pre-existing break already on bare main HEAD.

    Returns a :class:`MergeOutcome` with reason starting with
    :data:`MAIN_HEALTH_RED_REASON_PREFIX` when the break is confirmed
    pre-existing.  Returns ``None`` to fall through to the normal task-fault
    outcome.

    Guards (short-circuit to None before calling the probe): see
    :func:`_classification_skipped`.

    Latency: the first failing merge per ``(main_sha, category, cause_hint)``
    tuple pays a full probe build/test cost, bounded by the same
    ``merge_verify_cold_command_timeout_secs`` / ``verify_cold_command_timeout_secs``
    timeout that ``merge_lane/post_merge_verify.py::run_post_merge_verify`` uses (inherited via
    ``run_scoped_verification``'s config lookup — the probe runs with
    ``max_retries=0`` so a hung probe cannot stall the worker beyond one
    timeout window).  Subsequent merges with the same signature hit the
    ``_PROBE_CACHE`` and pay only a ``get_main_sha()`` round-trip.
    """
    if _classification_skipped(req, verify):
        return None
    try:
        is_preexisting, probe_sha = await _bare_main_fails_alike(
            git_ops, req, verify, event_store=event_store, escalation_queue=escalation_queue,
        )
    except Exception:
        return None
    if not is_preexisting:
        return None
    outcome = _build_main_health_outcome(verify, probe_sha)
    _emit_merge_attempt(event_store, req.task_id, OutcomeKind.main_health_red)
    return outcome


@dataclasses.dataclass(frozen=True)
class MainHealthProbeHandles:
    """Opaque bundle carrying the worker's ``background_tasks`` set so the
    deferred main-health probe survives ``merge_lane/post_merge_verify.py::run_post_merge_verify`` returning
    (task 2564, reify 5067 merge-slot-stall fix).

    A DISTINCT bundle from the dry-run investigation's handles (not reused)
    — the main-health probe's ownership is independent of the dry-run
    investigation's, which keeps the shape minimal and future-friendly for
    the sibling host-affinity / warm-probe work.  ``background_tasks`` is the
    SAME ``set`` instance the worker stores at ``self._background_tasks`` —
    a spawned probe task's strong ref lives exactly as long as the worker
    and is drained by the worker's existing shutdown drain
    (``SpeculativeMergeWorker.stop``).

    Only ``VerifyDispatcher.post_merge_verify``, which serves the
    in-flight verify and the single-branch gate re-verify, passes a live
    instance (task 2564 step-16, task 6278).  The solo-reverify and train
    callers — and any bare test caller — pass ``None`` (the default), so
    ``merge_lane/post_merge_verify.py::run_post_merge_verify`` keeps running the main-health probe
    SYNCHRONOUSLY exactly as it did before task 2564.

    ``auto_heal`` (task 2564 step-22) is an optional callback threaded
    through to :func:`_run_deferred_main_health_probe`: when a confirmed,
    still-fresh pre-existing break is found, the probe routes to it
    (``await auto_heal(outcome, req)``) INSTEAD of filing the
    escalation-only fallback.  ``None`` (the default) preserves the
    escalation-only behaviour for every bare/test caller.  Only the
    production call site passes
    ``SpeculativeMergeWorker._auto_heal_main_health_deferred`` (task 2564
    step-24).
    """

    background_tasks: set[asyncio.Task] = dataclasses.field(default_factory=set)
    auto_heal: AutoHeal | None = None


async def _main_is_still_at(git_ops: GitOps, req: MergeRequest, probe_sha: str) -> bool:
    """Whether main HEAD is still *probe_sha*; fails safe (False) when it cannot tell."""
    try:
        current_main_sha = await git_ops.get_main_sha()
    except Exception:
        logger.warning(
            'Task %s: deferred main-health probe: get_main_sha() re-resolve '
            'failed; skipping escalation (stale-check fail safe)',
            req.task_id,
            exc_info=True,
        )
        return False
    if not current_main_sha or current_main_sha != probe_sha:
        logger.info(
            'Task %s: deferred main-health probe: main advanced since the '
            'probe ran (probe_sha=%s, current=%s); skipping escalation '
            '(stale-check fail safe)',
            req.task_id, probe_sha, current_main_sha,
        )
        return False
    return True


async def _run_deferred_main_health_probe(
    git_ops: GitOps,
    req: MergeRequest,
    verify: VerifyResult,
    *,
    escalation_queue: Any = None,
    event_store: EventStore | None = None,
    auto_heal: AutoHeal | None = None,
    origin_is_local: bool = True,
) -> None:
    """Off-critical-path main-health classification (task 2564).

    LEASE-SAFETY & HOST-AFFINITY (task 2565): this probe is a leaseless,
    local-only detective classifier.  It verifies via the local
    ``_mainprobe-`` ephemeral worktree and ``run_scoped_verification`` —
    never :class:`~orchestrator.verify_runner.HostAllocator`,
    :class:`~orchestrator.verify_runner.RemoteRunner`, or
    :class:`~orchestrator.verify_runner.HostLease` — so it never acquires
    or holds a merge host slot.  *origin_is_local* RECORDS, but does not
    STEER, where the triggering post-merge verify ran (derived by the
    caller from the ``runner`` ``merge_lane/post_merge_verify.py::run_post_merge_verify`` received:
    ``None``/local-lease is local, ``RemoteRunner.is_local=False`` is
    remote — defaults to ``True`` so any caller not yet threading this
    value is treated as local); the probe itself ALWAYS runs locally
    regardless of *origin_is_local*.  Remote-affinity (probing on the
    failing verify's origin host) is deliberately rejected: the probe's
    verdict drives a queue-halting born-at-L2 escalation and a "fix main"
    auto-heal, a correctness-critical global decision that must answer "is
    main broken on the trust anchor?" — reproducing a host-specific
    toolchain/env/flock/disk quirk from a remote host onto that same host's
    main would falsely conclude main is red, and would re-introduce the
    remote-lease / orphaned-remote-build hazards task 1757's
    ``_abort_remote_verify`` teardown exists to avoid.  See
    :func:`verify_failure_is_preexisting_on_main` for the full contract.

    Spawned by :func:`spawn_main_health_probe` as a DETACHED task with no
    awaiter, so every externally-visible effect must come from what this
    coroutine does internally (emitting the ``main_health_red`` signal,
    filing a dedup'd escalation) — its return value is always discarded.

    Re-applies the SAME three cheap guards :func:`spawn_main_health_probe`
    already applied before spawning (:func:`_classification_skipped`;
    defense in depth for any future direct caller).

    Calls :func:`verify_failure_is_preexisting_on_main` directly (rather than
    :func:`classify_main_health_red`) so it can capture ``probe_sha`` for
    the staleness check below.  A raising probe is caught and logged — this
    is a detached background task with no awaiter, so a swallowed exception
    here is the ONLY way it would ever be observed.

    Staleness re-check (:func:`_main_is_still_at`): the probe can run for
    minutes off the critical path, during which main may have advanced (e.g.
    a hotfix that already fixed the break).  ``probe_sha`` is the exact
    bare-main HEAD the probe actually tested against, so equality against a
    freshly-resolved ``git_ops.get_main_sha()`` is the precise freshness
    predicate.  Fails safe (skips the escalation, logging at ``info``) when
    the re-resolve raises, comes back empty, or disagrees with ``probe_sha``
    — a genuinely-still-broken main will simply be re-probed and
    re-surfaced by the next failing merge.

    This re-check is best-effort, not atomic (task 2564 amendment,
    reviewer_comprehensive finding 2): a narrow TOCTOU window remains
    between the ``get_main_sha()`` equality check above passing and the
    escalation actually being filed below — or, when *auto_heal* is wired,
    between that check and whatever further awaits the ``auto_heal``
    callback performs — during which main could advance again.  Left
    unguarded deliberately: the fingerprint-keyed ``submit_or_dedupe`` fold
    (see :func:`file_main_health_escalation`) makes a single
    just-superseded escalation harmless — it collapses into any existing
    parent and the parent resolves once the real fix lands — so closing
    this window is not worth the added complexity.

    On a confirmed, still-fresh pre-existing break: builds the outcome via
    :func:`_build_main_health_outcome` and emits the ``main_health_red``
    merge-attempt signal (task 2564 step-22: BEFORE the branch below, so the
    signal fires whichever branch is taken), recording *origin_is_local* as
    the ``origin_host``/``probe_host`` telemetry pair (task 2565 —
    ``probe_host`` is always ``'local'``; see the LEASE-SAFETY &
    HOST-AFFINITY note above) so the deliberate local-only placement
    decision is OBSERVABLE even when the triggering verify ran remote.
    Then, if *auto_heal* is supplied, routes the outcome to it
    (``await auto_heal(outcome, req)``)
    INSTEAD of filing an escalation directly — the callback owns filing
    (see ``SpeculativeMergeWorker._auto_heal_main_health_deferred``,
    which files the halt-owner escalation itself).  Otherwise (the default,
    every bare/test caller) falls back to the pre-existing escalation-only
    behaviour via :func:`file_main_health_escalation`.
    """
    if _classification_skipped(req, verify):
        return
    try:
        is_preexisting, probe_sha = await _bare_main_fails_alike(
            git_ops, req, verify, event_store=event_store, escalation_queue=escalation_queue,
        )
    except Exception:
        logger.warning(
            'Task %s: deferred main-health probe failed', req.task_id,
            exc_info=True,
        )
        return
    if not is_preexisting or not await _main_is_still_at(git_ops, req, probe_sha):
        return

    outcome = _build_main_health_outcome(verify, probe_sha)
    _emit_merge_attempt(
        event_store, req.task_id, OutcomeKind.main_health_red,
        origin_host='local' if origin_is_local else 'remote',
        probe_host='local',
    )
    if auto_heal is not None:
        await auto_heal(outcome, req)
    else:
        file_main_health_escalation(
            escalation_queue, req, outcome,
            suggested_action=MainHealthNextAction.NO_FIX_OWNER,
        )


def file_main_health_escalation(
    escalation_queue: Any,
    req: MergeRequest,
    outcome: MergeOutcome,
    *,
    suggested_action: MainHealthNextAction,
    detail: str | None = None,
) -> str | None:
    """File (or fold) a dedup'd ``preexisting_main_break`` escalation for a
    confirmed pre-existing main-red *outcome* (task 2564).  *detail* defaults
    to ``outcome.reason``.

    Returns the id of the SURVIVING (parent) escalation, or ``None`` when
    *escalation_queue* is None.  A folded submission returns the PARENT's id
    (the child is absorbed via ``attach_dedupe_child`` and never becomes
    independently addressable) — callers that need to register lane-halt
    ownership (``SpeculativeMergeWorker._auto_heal_main_health_deferred``)
    must reference the parent, not the folded child.  Existing callers that
    ignore the return value are unaffected (backward-compatible addition).

    Every caller names its bounded next action from
    :class:`~orchestrator.main_health_actions.MainHealthNextAction`.  The
    record is filed at L1 because its consumer is escalation-watcher-auto,
    never a steward: a steward cannot fix main, and L0s are dismissed by
    the blocking task's own cleanup and at restart.

    Folds via ``submit_or_dedupe`` using the SAME inf-window
    content-fingerprint :class:`~escalation.dedupe.DedupeConfig`
    ``workflow.py::TaskWorkflow._file_main_health_l1`` uses (categories=
    ``('preexisting_main_break',)``), so every worker- and workflow-filed
    merge-lane L1 for one ``(category, cause_hint, probe_sha)`` signature
    collapses into one parent, race-free; the task-verify L0 for that break stays separate
    (dedupe never folds across ladder levels). Falls back to a plain
    ``queue.submit`` when *outcome* carries no fingerprint (the fail-safe
    ``''`` path of :func:`_main_health_fingerprint`).
    """
    if escalation_queue is None:
        return None

    fp = outcome.dedupe_fingerprint or None
    esc = Escalation(
        id=escalation_queue.make_id(req.task_id),
        task_id=req.task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='preexisting_main_break',
        summary=outcome.reason[:200],
        detail=outcome.reason if detail is None else detail,
        suggested_action=suggested_action,
        dedupe_fingerprint=fp,
        level=1,
    )
    if fp:
        submit_result = submit_or_dedupe(
            escalation_queue,
            esc,
            DedupeConfig(
                infra_dedupe_enabled=True,
                infra_dedupe_window_secs=float('inf'),
                infra_dedupe_categories=('preexisting_main_break',),
                key_fn=content_fingerprint_key,
            ),
        )
        return submit_result.get('parent_id') or submit_result.get('id') or esc.id
    escalation_queue.submit(esc)
    return esc.id


def _probe_in_flight(handles: MainHealthProbeHandles, task_name: str) -> bool:
    return any(
        t.get_name() == task_name and not t.done()
        for t in handles.background_tasks
    )


def spawn_main_health_probe(
    handles: MainHealthProbeHandles | None,
    git_ops: GitOps,
    req: MergeRequest,
    verify: VerifyResult,
    *,
    escalation_queue: Any = None,
    event_store: EventStore | None = None,
    origin_is_local: bool = True,
) -> bool:
    """Fire-and-forget: spawn the deferred (off-critical-path) main-health
    classification for a post-merge-verify failure.

    *origin_is_local* (task 2565) is forwarded unchanged into the spawned
    :func:`_run_deferred_main_health_probe` coroutine — it only RECORDS
    where the triggering post-merge verify ran (for the ``origin_host``/
    ``probe_host`` telemetry pair); the probe itself always classifies
    locally regardless of this value.  Defaults to ``True`` (local) so
    every caller not yet threading the signal is treated as local.

    None-safe: no-ops when *handles* is ``None`` (mirrors
    ``_spawn_merge_verify_dry_run``'s ``handles is None`` guard) — every
    call site invokes this unconditionally and relies on this internal check,
    so the solo-reverify and train (its gate re-verify included)
    ``merge_lane/post_merge_verify.py::run_post_merge_verify`` callers (which pass no handles) are guaranteed
    no-ops.

    Applies the SAME three cheap guards :func:`classify_main_health_red`
    applies before probing (:func:`_classification_skipped`, task 2564
    step-6), so the deferred path never probes a case the synchronous path
    would have skipped.

    Also dedupes in-flight: skips the spawn when a not-done task named
    ``f'main-health-probe-{req.task_id}'`` is already registered in
    ``handles.background_tasks`` (mirrors
    ``_spawn_merge_verify_dry_run``'s in-flight dedup guard) — a rapid
    sequence of failing merges for the same task must not pile up duplicate
    probes.

    Registers the spawned task into ``handles.background_tasks`` with a
    discard done-callback, named ``f'main-health-probe-{req.task_id}'`` so
    ``SpeculativeMergeWorker``'s existing ``self._background_tasks``
    shutdown drain cancels it deterministically on stop — no new drain code
    is needed.

    Returns ``True`` when, as a result of this call, a main-health probe is
    (or remains) in flight for *req.task_id* — *handles* were provided, none
    of the three cheap guards above skipped classification, and either a new
    probe task was just spawned or one was already running (in-flight
    dedup).  Returns ``False`` when no probe is in flight: *handles* is
    ``None``, a guard skipped classification, or spawning itself raised.
    Task 2564 amendment (reviewer_comprehensive finding 1): the caller
    (``merge_lane/post_merge_verify.py::run_post_merge_verify``) uses this to decide whether the
    provisional task-fault outcome should carry
    ``MAIN_HEALTH_PROBE_PENDING_NOTE`` — a caller cannot otherwise tell
    "classification skipped, confirmed task fault" apart from "classification
    deferred, verdict pending" from the reason string alone.
    """
    if handles is None or _classification_skipped(req, verify):
        return False
    task_name = f'main-health-probe-{req.task_id}'
    if _probe_in_flight(handles, task_name):
        logger.debug(
            'Task %s: main-health probe already in progress, skipping '
            'duplicate spawn',
            req.task_id,
        )
        return True
    try:
        task = asyncio.create_task(
            _run_deferred_main_health_probe(
                git_ops, req, verify,
                escalation_queue=escalation_queue, event_store=event_store,
                auto_heal=handles.auto_heal,
                origin_is_local=origin_is_local,
            ),
            name=task_name,
        )
        handles.background_tasks.add(task)
        task.add_done_callback(handles.background_tasks.discard)
    except Exception as exc:
        logger.warning(
            'Task %s: failed to spawn main-health probe: %s',
            req.task_id, exc,
        )
        return False
    return True
