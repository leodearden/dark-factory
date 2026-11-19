"""The land-time cross-check of a remote green on the local trust anchor (task 2822, fix b).

A single remote host's green must not blindly decide a land: before it does,
:func:`cross_check_remote_green` re-runs the merge verify with the LOCAL
trust anchor on the intact merge worktree and returns the verdict the land
is decided on.

  AGREE (local passes too)  → emit verdict_parity_ok; the remote green stands.
  DIVERGE (local FAILs)     → fail-CLOSED: halt the queue on a genuine
                              divergence, quarantine the remote, file a dedup'd
                              verify_cross_check_mismatch escalation, and hand
                              back the local FAIL so the land is withheld.
  local RunnerUnavailable,  → fail-SAFE: emit verify_cross_check_inconclusive
  a contended lane lock, or   and keep the remote green (never block a land on
  a local leg with no         a local infra hiccup — DriftDetector Invariant 5).
  verdict

It executes a FULL LOCAL verify inside the same awaited verify task the
in-flight verify polls, which is why that poll's no-progress signal includes
the content-mtime arm for REMOTE leases too.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from escalation.models import Escalation

from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeVerifyLeaseContended
from orchestrator.merge_lane.types import MergeRequest
from orchestrator.verify import VerifyResult
from orchestrator.verify_categories import (
    INDETERMINATE_VERDICT_CATEGORIES,
    INFRA_TRANSIENT_CATEGORIES,
)
from orchestrator.verify_runner import MergeVerifySpec, RunnerUnavailable, VerifyRunner

logger = logging.getLogger('orchestrator.merge_lane')

# Dedup sentinel task_id for the per-land cross-check divergence escalation
# (task 2822) — mirrors verify_runner.py's _DRIFT_SENTINEL.  A single open L1
# per episode keeps a run of divergent lands from flooding the queue; the
# caller-owned quarantine sink does the per-host withholding.
_CROSS_CHECK_SENTINEL = '__verify_cross_check__'


@dataclasses.dataclass(frozen=True)
class CrossCheck:
    """One land's cross-check: the remote that passed and the local anchor that re-verifies it.

    *quarantine_divergent* is the lane's divergence-quarantine sink
    (``host_quarantine.py::HostQuarantine.quarantine_divergent``) and
    *halt_hook* its synchronous all-lane halt; either may be ``None`` for a
    caller without one, and the cross-check still fails closed.
    *record_observation* records the trust anchor's own flake observation.
    """

    git_ops: GitOps
    req: MergeRequest
    merge_wt: Path
    merge_sha: str
    spec: MergeVerifySpec
    remote: VerifyRunner
    trust_anchor: VerifyRunner
    event_store: EventStore | None
    escalation_queue: Any
    halt_hook: Callable[[str], None] | None
    quarantine_divergent: Callable[[str], None] | None
    record_observation: Callable[[VerifyResult], Awaitable[None]]

    def emit(self, event: EventType, data: dict[str, Any]) -> None:
        if self.event_store is not None:
            self.event_store.emit(event, task_id=self.req.task_id, data=data)


async def cross_check_remote_green(remote_green: VerifyResult, check: CrossCheck) -> VerifyResult:
    """The verdict to land on: *remote_green*, or the local trust anchor's FAIL on a divergence."""
    local_verify = await _trust_anchor_verdict(check)
    if local_verify is None:
        return remote_green
    if _produced_no_verdict(local_verify):
        _trust_verdictless_anchor(check, local_verify)
        return remote_green
    if local_verify.passed == remote_green.passed:
        check.emit(EventType.verdict_parity_ok, {
            'merge_sha': check.merge_sha,
            'local_runner': check.trust_anchor.name,
            'remote_runner': check.remote.name,
            'passed': remote_green.passed,
        })
        return remote_green
    _withhold_divergent_land(check, remote_green, local_verify)
    return local_verify


async def _trust_anchor_verdict(check: CrossCheck) -> VerifyResult | None:
    """The local trust anchor's verdict, or None when it could not be had (fail-safe)."""
    try:
        # task 2873: defense-in-depth lane-lock for the trust-anchor verify.
        # Lock the ACTUAL merge_wt lane (persistent OR the ephemeral
        # _merge-<hash> speculation worktree) so a concurrent reseed/reclaim of
        # THAT lane mutually excludes rather than clobbering the tree mid-verify
        # (reify 2026-07-20, merge_sha 83336a32: ENOENT storm → spurious DIVERGE
        # that withheld a good land).  Gated on persistent_merge_worktree: the
        # lane-lock's contenders only exist in the warm-lane regime.
        async with contextlib.AsyncExitStack() as cross_stack:
            if check.req.config.git.persistent_merge_worktree:
                await cross_stack.enter_async_context(
                    check.git_ops.merge_verify_lease(lane_dir=check.merge_wt)
                )
            local_verify = await check.trust_anchor.run_merge_verify(check.merge_sha, check.spec)
        # Task ε: the cross-check ran its OWN merge gate on the local trust
        # anchor, so it carries its own observation.  After the await, so a
        # raised verify records nothing — there is no verdict to record.
        await check.record_observation(local_verify)
    except RunnerUnavailable as exc:
        # A closed/flaky laptop trust-anchor must never block a remote green
        # (symmetric with DriftDetector's Invariant 5).
        check.emit(EventType.verify_cross_check_inconclusive, {
            'merge_sha': check.merge_sha,
            'remote_runner': check.remote.name,
            'reason': str(exc),
        })
        logger.warning(
            'Task %s: local cross-check trust-anchor unavailable for %s '
            '(remote=%s) — trusting the remote green (fail-safe): %s',
            check.req.task_id, check.merge_sha, check.remote.name, exc,
        )
        return None
    except MergeVerifyLeaseContended as exc:
        # A contended lane-lock means a reseed/reclaim is actively mutating
        # merge_wt; running the local verify now would reproduce the ENOENT
        # clobber the lease prevents.  The remote green is already a primary
        # verdict, so SKIP the second opinion — no quarantine, no escalation,
        # no adopted FAIL.  Deliberately DIFFERENT from a local-dispatch
        # contention, which propagates to the requeue handler (there is no
        # primary verdict there).
        check.emit(EventType.verify_cross_check_inconclusive, {
            'merge_sha': check.merge_sha,
            'remote_runner': check.remote.name,
            'reason': f'cross-check lane-lock contended: {exc}',
        })
        logger.warning(
            'Task %s: cross-check local trust-anchor could not acquire the '
            'lane lock for %s (remote=%s) — a reseed/reclaim is in flight; '
            'trusting the remote green (fail-safe, DriftDetector Invariant 5): %s',
            check.req.task_id, check.merge_sha, check.remote.name, exc,
        )
        return None
    return local_verify


def _produced_no_verdict(local_verify: VerifyResult) -> bool:
    """True when EVERY failing leg of the local run is verdict-less (task 3173).

    INVARIANT: only a COMPLETED failing verdict on either host may veto a
    completed PASS.  A leg killed by an external signal never produced a
    verdict, so discarding an already-verified merge commit on its say-so is
    pure waste (measured: merge_sha b1ac2c7f, a 1097s remote PASS thrown away
    for a local leg SIGKILLed at 0.31s).  The registry names only
    ``infra_kill``; every other category — infra_timeout, disk_full,
    env_transient, semaphore_timeout, an EMPTY category — keeps its veto.

    Gated on EVERY failing leg, not the aggregate category: severity dominance
    makes a completed branch-blaming test leg beside a SIGKILLed lint leg
    aggregate to 'infra_kill', and that run must still veto.  A None list
    (an older remote's wire payload) and an empty one both fail CLOSED.
    """
    local_failing_legs = local_verify.failing_leg_categories or []
    return not local_verify.passed and bool(local_failing_legs) and all(
        c in INDETERMINATE_VERDICT_CATEGORIES for c in local_failing_legs
    )


def _trust_verdictless_anchor(check: CrossCheck, local_verify: VerifyResult) -> None:
    local_failing_legs = local_verify.failing_leg_categories or []
    check.emit(EventType.verify_cross_check_inconclusive, {
        'merge_sha': check.merge_sha,
        'remote_runner': check.remote.name,
        'local_category': local_verify.category,
        'local_failing_leg_categories': local_failing_legs,
        'reason': (
            f'local trust-anchor produced no verdict on any '
            f'failing leg (legs={local_failing_legs!r}, '
            f'category={local_verify.category!r}): '
            f'{local_verify.summary}'
        ),
    })
    logger.warning(
        'Task %s: local cross-check trust-anchor produced NO verdict '
        'for %s (remote=%s, category=%s, failing_legs=%s) — trusting '
        'the remote green (fail-safe, DriftDetector Invariant 5): %s',
        check.req.task_id, check.merge_sha, check.remote.name,
        local_verify.category, local_failing_legs, local_verify.summary,
    )


def _withhold_divergent_land(
    check: CrossCheck, remote_green: VerifyResult, local_verify: VerifyResult,
) -> None:
    """DIVERGE — remote PASS / local trust-anchor FAIL: halt, quarantine, escalate, record.

    task 2886 fix 3 (PRD §3.4 / decision 4): a GENUINE concluded divergence
    means main is SUSPECTED-RED, so the queue is halted FIRST, synchronously —
    before quarantine, escalation and the adopted FAIL — so no further
    adoption can occur before a human looks (incident 83336a32 halted +3s too
    late via the async escalation gate).  A local FAIL whose category is
    infra-transient is a LOCAL hiccup, not evidence main is red (task 2886
    amendment, DriftDetector Invariant 5): the land is still withheld and the
    remote quarantined, but the queue is NOT halted and the escalation is an
    L1 blocking one rather than born-at-L2.
    """
    local_infra_fail = (
        (local_verify.category or '') in INFRA_TRANSIENT_CATEGORIES
    )
    remote = check.remote.name
    if check.halt_hook is not None and not local_infra_fail:
        check.halt_hook(
            f'cross-check divergence for {check.merge_sha}: remote '
            f'{remote} PASS / local trust-anchor FAIL — '
            f'suspected red main'
        )
    if check.quarantine_divergent is not None:
        check.quarantine_divergent(remote)
    _escalate_divergence(check, local_verify, local_infra_fail=local_infra_fail)
    check.emit(EventType.verify_cross_check_mismatch, {
        'merge_sha': check.merge_sha,
        'remote_runner': remote,
        'remote_passed': remote_green.passed,
        'local_passed': local_verify.passed,
    })
    logger.warning(
        'Task %s: cross-check DIVERGENCE for %s — remote %s PASS but '
        'local trust-anchor FAIL; withholding the land and quarantining %s',
        check.req.task_id, check.merge_sha, remote, remote,
    )


def _escalate_divergence(
    check: CrossCheck, local_verify: VerifyResult, *, local_infra_fail: bool,
) -> None:
    queue = check.escalation_queue
    if queue is None or queue.has_open_l1(_CROSS_CHECK_SENTINEL):
        return
    render = _local_hiccup_escalation if local_infra_fail else _genuine_divergence_escalation
    text = render(check.merge_sha, check.remote.name, local_verify)
    queue.submit(Escalation(
        id=queue.make_id(_CROSS_CHECK_SENTINEL),
        task_id=_CROSS_CHECK_SENTINEL,
        agent_role='orchestrator-cross-check',
        severity=text.severity,
        level=text.level,
        category='verify_cross_check_mismatch',
        summary=text.summary,
        detail=text.detail,
        suggested_action=text.suggested_action,
    ))


@dataclasses.dataclass(frozen=True)
class _EscalationText:
    severity: str
    level: int
    summary: str
    detail: str
    suggested_action: str


def _local_hiccup_escalation(
    merge_sha: str, remote: str, local_verify: VerifyResult,
) -> _EscalationText:
    """Pre-fix-3 L1 blocking shape: land withheld + remote quarantined, NO fleet halt."""
    summary = (
        f'Cross-check divergence for {merge_sha}: '
        f'remote={remote!r} PASS / local trust-anchor FAIL '
        f'(local category={local_verify.category!r}, '
        f'infra-transient) — land withheld, {remote!r} '
        f'quarantined (queue NOT halted)'
    )
    detail = (
        f'merge_sha={merge_sha!r} remote_runner={remote!r} '
        f'reported PASS but the local trust-anchor re-verify '
        f'FAILED with an INFRA-TRANSIENT category '
        f'(summary={local_verify.summary!r}, '
        f'category={local_verify.category!r}).  Treated as a '
        f'LOCAL infra hiccup rather than a genuine main-red '
        f'divergence (DriftDetector Invariant 5): the land is '
        f'withheld and {remote!r} is quarantined, but the '
        f'merge queue is NOT halted.  If this recurs the remote '
        f'host env may be genuinely diverging — investigate '
        f'before clearing the quarantine.'
    )
    action = (
        'Investigate the local infra failure (was it a real '
        'divergence or a transient local hiccup?).  Clear the '
        "remote quarantine once the host's env is re-proved; no "
        'queue un-halt is needed (the queue was not halted).'
    )
    return _EscalationText('blocking', 1, summary, detail, action)


def _genuine_divergence_escalation(
    merge_sha: str, remote: str, local_verify: VerifyResult,
) -> _EscalationText:
    """Born-at-L2 (task 2886 fix 3): critical, mirroring the warm/cold shadow alarm."""
    summary = (
        f'Cross-check divergence for {merge_sha}: '
        f'remote={remote!r} PASS / local trust-anchor FAIL '
        f'— merge queue HALTED, main suspected-red'
    )
    detail = (
        f'merge_sha={merge_sha!r} remote_runner={remote!r} '
        f'reported PASS but the local trust-anchor re-verify FAILED '
        f'(summary={local_verify.summary!r}, '
        f'category={local_verify.category!r}). A remote PASS / local '
        f'FAIL split can land unverified code on main; the land is '
        f'withheld and {remote!r} is quarantined.  The merge '
        f'queue has been HALTED synchronously and main is SUSPECTED '
        f'RED pending human review — no further land can proceed '
        f'until an operator confirms main is green and un-halts the '
        f'queue.'
    )
    action = (
        'Confirm whether main is actually red at this commit; if so, '
        'roll it back.  Re-prove the remote host env '
        '(run_verdict_parity) and clear its quarantine once parity is '
        'restored, then un-halt the merge queue.  Investigate why the '
        'remote green disagreed with the local trust-anchor.'
    )
    return _EscalationText('critical', 2, summary, detail, action)
