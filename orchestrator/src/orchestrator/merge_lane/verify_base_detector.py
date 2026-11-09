"""The §5.3 verify-base detector: what counts as a dispatch-time divergence,
and how one is escalated.

A real verify should start on the frozen-prefix tip (plans/two-layer-merge-queue-prd.md
§5.3).  The rule is a DETECTOR, not an invariant: it is judged once, when the
verify is dispatched, against that moment's main, and a hit is recorded and
escalated but never refused.  Refusing would be enforcement at dispatch, which
esc-3206-3 ruled out; the soundness property belongs to INV-3, which voids a
verdict built on a dead base at adoption.
"""

from __future__ import annotations

from typing import Any

from escalation.categories import EscalationCategory
from escalation.models import Escalation

from orchestrator.event_store import EventType
from orchestrator.merge_lane.types import (
    RealMergeItem,
    RemergeBase,
    SpeculativeItem,
    VerifyBaseDivergence,
    VerifyBaseDivergenceKind,
    fell_to_main,
)

VERIFY_BASE_DIVERGENCE_SENTINEL = '__merge_verify_base_divergence__'


def detect_verify_base_divergence(
    item: SpeculativeItem,
    expected_tip: str,
    *,
    verify_depth: int,
    observed_at: float,
) -> VerifyBaseDivergence | None:
    """The divergence *item* commits by verifying off *expected_tip*, or None.

    Only a real merge has a verify base.  A recovery re-merge that fell to
    main targets main by construction, so it is carved out (PRD §5.3; task
    8524 narrowed the carve-out to that case, plans/merge-lane-policy-prd.md D1).
    One re-anchored on a live tip is judged like any stacked dispatch, and a
    hit on it is its own kind.
    """
    if not isinstance(item, RealMergeItem) or not item.base_sha or fell_to_main(item.remerge_base):
        return None
    if item.base_sha.strip() == expected_tip.strip():
        return None
    return VerifyBaseDivergence(
        request_id=item.request.request_id,
        task_id=item.request.task_id,
        base_sha=item.base_sha,
        expected_tip=expected_tip,
        verify_depth=verify_depth,
        observed_at=observed_at,
        kind=_kind_of(item.remerge_base),
    )


def _kind_of(anchor: RemergeBase | None) -> VerifyBaseDivergenceKind:
    if anchor is not None and anchor.speculative:
        return VerifyBaseDivergenceKind.REANCHOR_ONTO_DEAD_TIP
    return VerifyBaseDivergenceKind.OFF_FROZEN_TIP


def alarm_verify_base_divergence(
    escalation_queue: Any,
    divergence: VerifyBaseDivergence,
    *,
    event_store: Any = None,
) -> None:
    """File *divergence* as an L1, unless one is already open for the sentinel.

    Observation only: the verify it describes runs on regardless.
    """
    if escalation_queue is None:
        return
    sentinel = VERIFY_BASE_DIVERGENCE_SENTINEL
    if escalation_queue.has_open_l1(sentinel):
        return
    wire = divergence.to_wire()
    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-verify-base-detector',
        severity='blocking',
        level=1,
        category=EscalationCategory.MERGE_VERIFY_BASE_DIVERGENCE,
        summary=(
            f'Merge verify dispatched off the frozen-prefix tip ({divergence.kind.value}): '
            f'{divergence.request_id} (task {divergence.task_id}) verifies on '
            f'{divergence.base_sha!r}, expected {divergence.expected_tip!r}'
        ),
        detail=(
            'A real verify started on a base that was not the frozen-prefix '
            "tip at dispatch (PRD §5.3).  The record, also in the lane's "
            "snapshot()['verify_base_divergences']:\n\n"
            + '\n'.join(f'- {key}: {value!r}' for key, value in wire.items())
            + '\n\nThe verify was not refused and the lane changed nothing '
            '(esc-3206-3).  If the verdict is built on a dead base, INV-3 '
            'voids it at adoption.'
        ),
        suggested_action=(
            'Triage which of three sources produced the hit.  (1) expected_tip '
            'is the merge commit of a head that was already requeued or '
            'dropped: the frozen-prefix liveness rule has regressed (PRD §5.3 '
            'class (a), task 5302).  (2) expected_tip is the merge commit of a '
            'head that was never vacated and is stuck in finalizing or '
            'gate_reverify, including a head whose own merge commit was '
            'recorded dead on a rebased landing (merge_lane/worker.py::'
            'SpeculativeMergeWorker::_land_rebased_or_park_gate, ticket '
            'tkt_0RWKN8Q02REA5YEFNQX4WDJRJT): a phantom tip the liveness rule '
            'cannot see, and the item itself is fine.  (3) Otherwise the item '
            'was stacked on a speculative-only base, a genuine class-(d) '
            'violation: find the path that built it.  A recovery re-merge '
            'that fell to main is exempt and the cached main is never read, '
            'so neither explains a hit; a reanchor_onto_dead_tip hit means a '
            're-anchored tip died or was vacated between re-merge and '
            'dispatch (expected 0; plans/merge-lane-policy-prd.md D12).'
        ),
    )
    escalation_queue.submit(esc)
    if event_store is not None:
        event_store.emit(
            EventType.escalation_created,
            task_id=divergence.task_id,
            data={
                **wire,
                'escalation_id': esc.id,
                'category': str(esc.category),
                'summary': esc.summary,
            },
        )
