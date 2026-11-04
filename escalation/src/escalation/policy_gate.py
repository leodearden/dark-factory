"""The escalation server's wire answers to the policy table's verdicts.

Task 8422, ``plans/decision-stream-prd.md`` §5.2. ``escalation.authority``
decides whether the executor may act on or claim a record, and whether the
coordinator must wait for an executor claim. This module reads the queue's
class switches (``escalation.policy_state``), asks those questions, and shapes
each answer as the dict ``resolve_issue`` or ``claim_policy_class`` returns.
:func:`perform_claim` is the one function here that writes: it stamps the
claim through ``EscalationQueue.claim_policy_class``. The rest read the state
file at most.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from escalation.authority import (
    COORDINATOR_IDENTITY,
    EXECUTOR_IDENTITY,
    PolicyRefusal,
    RefusalCode,
    claim_refusal,
    coordinator_denied,
    executor_refusal,
)
from escalation.models import Escalation
from escalation.policy_state import PolicyStateUnreadable, load_class_states
from escalation.queue import EscalationQueue
from escalation.store_identity import QueueKind, StoreIdentity


def queue_kind_of(store_identity: StoreIdentity | None) -> QueueKind:
    """The kind of queue a server serves; an identity-less server serves a project."""
    return 'project' if store_identity is None else store_identity.kind


def _state_unreadable(exc: PolicyStateUnreadable) -> dict[str, Any]:
    return {'error': str(exc), 'code': 'policy_state_unreadable'}


def _level_forbidden(detail: str) -> dict[str, Any]:
    return {'error': detail, 'code': 'level_forbidden'}


def identity_resolve_refusal(
    identity: str | None,
    rec: Escalation,
    *,
    action: str,
    resolution: str,
    resolution_class: str | None,
    queue_dir: Path,
    queue_kind: QueueKind,
    now: datetime,
) -> dict[str, Any] | None:
    """The error ``resolve_issue`` returns for *identity* on *rec*, or ``None``.

    Only the executor and the coordinator are gated here. The executor needs a
    ``resolution_class`` and a verdict of ``None`` from
    :func:`~escalation.authority.executor_refusal`. The coordinator is refused
    while *rec* carries a fresh executor claim.
    """
    if identity == EXECUTOR_IDENTITY:
        if resolution_class is None:
            return {
                'error': 'an executor resolve must state its resolution_class',
                'code': 'resolution_class_required',
            }
        try:
            state = load_class_states(queue_dir)
        except PolicyStateUnreadable as exc:
            return _state_unreadable(exc)
        refusal = executor_refusal(
            rec, action, resolution, state, now=now, queue_kind=queue_kind,
        )
        if refusal is None:
            return None
        return {
            **_level_forbidden(refusal.detail),
            'policy_refusal': str(refusal.code),
            'policy_class': rec.policy_class,
        }
    if identity == COORDINATOR_IDENTITY and coordinator_denied(rec, now):
        return {
            'error': (
                f'{rec.id} is claimed by the executor as {rec.policy_class!r} '
                f'at {rec.claimed_at}; wait for the claim to close or expire'
            ),
            'code': 'claimed_by_executor',
            'policy_class': rec.policy_class,
            'claimed_at': rec.claimed_at,
        }
    return None


def _claim_refused(refusal: PolicyRefusal, existing: Escalation) -> dict[str, Any]:
    if refusal.code == RefusalCode.DENIED_RECORD:
        return _level_forbidden(refusal.detail)
    payload: dict[str, Any] = {
        'error': refusal.detail,
        'code': 'claim_refused',
        'policy_refusal': str(refusal.code),
    }
    if refusal.code == RefusalCode.ALREADY_CLAIMED:
        payload['policy_class'] = existing.policy_class
        payload['claimed_at'] = existing.claimed_at
    return payload


def perform_claim(
    queue: EscalationQueue,
    *,
    identity: str | None,
    escalation_id: str,
    policy_class: str,
    clean: bool,
    queue_kind: QueueKind,
    now: datetime,
) -> dict[str, Any]:
    """Stamp the executor's claim on the record, then answer ``claim_policy_class``.

    The answer is the claimed record, or why the claim was refused and nothing
    was written.
    """
    if identity != EXECUTOR_IDENTITY:
        return _level_forbidden(f'only {EXECUTOR_IDENTITY} may claim a policy class')
    try:
        state = load_class_states(queue.queue_dir)
    except PolicyStateUnreadable as exc:
        return _state_unreadable(exc)

    def refuse(record: Escalation) -> tuple[PolicyRefusal, Escalation] | None:
        refusal = claim_refusal(
            record, policy_class, state, now=now, clean=clean, queue_kind=queue_kind,
        )
        return None if refusal is None else (refusal, record)

    result = queue.claim_policy_class(
        escalation_id, policy_class=policy_class, claimed_by=EXECUTOR_IDENTITY, refuse=refuse,
    )
    if result.refusal is not None:
        return _claim_refused(*result.refusal)
    if result.record is None:
        return {'error': f'Escalation {escalation_id} not found or not pending'}
    return result.record.to_dict()
