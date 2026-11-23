"""Adapt the offline lane's pass outcome to the verify-cache audit, and report the result as evidence notes and the kill-switch blocker."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from escalation.categories import EscalationCategory
from escalation.models import Escalation

from orchestrator.offline_lane_confirm import Confirmed, PassOutcome
from orchestrator.offline_lane_filing import EvidenceNote
from orchestrator.offline_lane_records import RunRecord
from orchestrator.verify_cache.audit import (
    AuditVerdict,
    LeakIdentity,
    LeakRecord,
    LeakResponse,
    apply,
    attribute,
)
from orchestrator.verify_cache.audit_run import AUDIT_COMMAND
from orchestrator.verify_cache.registry import PolicyState, Registry

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

NOTE_SOURCE = 'verify-cache-audit'

# The born-at-L2 sentinel form of flake_recorder.py::_bump_suppression_streak_and_maybe_escalate:
# the orchestrator- prefix marks a harness sentinel, so the server keeps its critical severity.
AUDIT_ESCALATION_ROLE = 'orchestrator-verify-cache-audit'
AUDIT_ESCALATION_SENTINEL = 'verify-cache-kill-switch'

ENABLE_COMMAND = 'orchestrator verify-cache enable --reason "<why the cache is safe again>"'

_SUMMARY_MAX_CHARS = 200


def _utc_now() -> datetime:
    return datetime.now(UTC)


class VerifyCacheAuditHook:
    """The lane's section 5.7 hook for the verify-cache-audit command (plans/verify-result-cache-prd.md § Audit).

    Only a confirmed red's reproduced nodes are evidence; registry errors propagate, so the lane drops only these notes.
    An escalation-queue error costs only the blocker sync, which the next attributable pass retries.
    """

    def __init__(
        self,
        registry: Registry,
        escalation_queue: EscalationQueue | None,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._registry = registry
        self._queue = escalation_queue
        self._now = now
        self._reported_unrecorded_consumption = False

    async def __call__(
        self, command: str, head: str, outcome: PassOutcome, run_records: Sequence[RunRecord],
    ) -> tuple[EvidenceNote, ...]:
        if command != AUDIT_COMMAND or not isinstance(outcome, Confirmed) or not outcome.reproduced:
            return ()
        verdict = await attribute(head, outcome.reproduced, self._registry)
        if not verdict.attributions:
            await self._note_unattributable(verdict)
            return ()
        now = self._now()
        response = await apply(verdict, self._registry, now)
        notes = _notes(response, now)
        try:
            await asyncio.to_thread(self._sync_blockers, response, now)
        except Exception:
            logger.exception(
                'verify-cache audit: syncing the kill-switch blocker with the escalation queue failed '
                '(policy disabled_until=%s); the next attributable pass retries it',
                response.policy.disabled_until,
            )
        return notes

    async def _note_unattributable(self, verdict: AuditVerdict) -> None:
        """Tell 'nothing records a consumption' apart from 'no leak', once per process (INV-13)."""
        if self._reported_unrecorded_consumption:
            return
        summary = await self._registry.summary()
        if summary.consumed:
            return
        self._reported_unrecorded_consumption = True
        logger.warning(
            'verify-cache audit: %d reproduced red(s) are unattributable and %s records no consumption at all, '
            'so the audit cannot tell a leak from a landing that replayed nothing: either no landing has '
            'replayed a green, or nothing records the greens it replays. Reported once per process.',
            len(verdict.unattributed), summary.path,
        )

    def _sync_blockers(self, response: LeakResponse, now: datetime) -> None:
        """Resolve the blockers of ended disables, then file the live disable's blocker if it has none."""
        if self._queue is None:
            if response.l2_required:
                logger.error(
                    'verify-cache audit: the cache is DISABLED until %s but there is no escalation queue '
                    'to file its L2 blocker on',
                    response.policy.disabled_until,
                )
            return
        resolve_ended_blockers(
            self._queue, response.policy, now,
            resolved_by=AUDIT_ESCALATION_ROLE, resolution_class='benign',
            resolution=f'The kill-switch disable this blocker describes has ended: {_policy_now(response.policy, now)}.',
        )
        if not response.l2_required:
            return
        episode = _disable_episode(response.policy)
        if any(blocker.root_cause == episode for blocker in _pending_blockers(self._queue)):
            return
        self._queue.submit(_blocker(self._queue.make_id(AUDIT_ESCALATION_SENTINEL), response))


def resolve_ended_blockers(
    queue: EscalationQueue, policy: PolicyState, now: datetime, *,
    resolved_by: str, resolution: str, resolution_class: str,
) -> tuple[str, ...]:
    """Resolve every pending kill-switch blocker except the one for *policy*'s live disable; the ids resolved."""
    live = _disable_episode(policy) if policy.is_disabled(now) else None
    ended = [blocker for blocker in _pending_blockers(queue) if blocker.root_cause != live]
    for blocker in ended:
        queue.resolve(blocker.id, resolution, resolved_by=resolved_by, resolution_class=resolution_class)
    return tuple(blocker.id for blocker in ended)


def _pending_blockers(queue: EscalationQueue) -> list[Escalation]:
    return queue.get_by_task(AUDIT_ESCALATION_SENTINEL, status='pending', level=2)


def _policy_now(policy: PolicyState, now: datetime) -> str:
    if policy.is_disabled(now):
        return f'the cache is now DISABLED until {_until(policy)} under a later disable'
    return f'the cache is enabled ({policy.reason})'


def _disable_episode(policy: PolicyState) -> str:
    """The blocker's dedup key: one blocker per disable, so a re-trip after an enable-only re-enable files again."""
    return f'verify-cache kill switch disabled until {_until(policy)}'


def _blocker(escalation_id: str, response: LeakResponse) -> Escalation:
    records = [LeakRecord.parse(text) for text in response.policy.leaks]
    units = ', '.join(dict.fromkeys(leak.unit_id for leak in response.streak_leaks)) or 'none counting toward the streak'
    summary = f'Verify cache DISABLED until {_until(response.policy)}: attributable leaks on {units}'
    return Escalation(
        id=escalation_id,
        task_id=AUDIT_ESCALATION_SENTINEL,
        agent_role=AUDIT_ESCALATION_ROLE,
        severity='critical',
        level=2,
        category=EscalationCategory.RISK_IDENTIFIED,
        summary=summary[:_SUMMARY_MAX_CHARS],
        root_cause=_disable_episode(response.policy),
        detail=_blocker_detail(records, response),
        suggested_action=(
            "Find the input the leaked units' cache keys missed, using the leak records in the detail. "
            f'Once it is understood, re-enable with `{ENABLE_COMMAND}`, which also resolves this escalation; '
            'the cache stays disabled until then, or until disabled_until passes.'
        ),
    )


def _blocker_detail(records: Sequence[LeakRecord], response: LeakResponse) -> str:
    streak = frozenset(leak.identity for leak in response.streak_leaks)
    lines = [
        'The verify-cache audit ran the merge-role command set uncached on the offline lane and reproduced '
        'reds on units whose greens the named landings had replayed from the cache.',
        f'Kill switch: disabled until {_until(response.policy)} - {response.policy.reason}',
        'Leak records:',
        *(f'- {_leak_line(record, streak)}' for record in records),
        'Evicted by this pass:',
        *(f'- key {key}: {count} entries' for key, count in response.evicted.items()),
    ]
    return '\n'.join(lines)


def _leak_line(record: LeakRecord, streak: frozenset[LeakIdentity]) -> str:
    counted = '' if record.identity in streak else ' (not counting toward the streak)'
    return (
        f'{record.unit_id} key {record.key} entry {record.entry_id} at landing {record.landing_sha}, '
        f'evidence event {record.evidence_event_id}, detected {record.detected_at.isoformat()}, '
        f'nodes: {", ".join(record.nodes)}{counted}'
    )


def _until(policy: PolicyState) -> str:
    return 'never' if policy.disabled_until is None else policy.disabled_until.isoformat()


def _notes(response: LeakResponse, now: datetime) -> tuple[EvidenceNote, ...]:
    return tuple(
        EvidenceNote(NOTE_SOURCE, node, _note_text(leak, response, now))
        for leak in response.leaks
        for node in leak.nodes
    )


def _note_text(leak: LeakRecord, response: LeakResponse, now: datetime) -> str:
    text = (
        f'verify-cache leak: {leak.unit_id} replayed green entry {leak.entry_id} (key {leak.key}, '
        f'evidence event {leak.evidence_event_id}) at landing {leak.landing_sha}, and the uncached audit '
        f'reproduced this failure. Evicted {response.evicted.get(leak.key, 0)} entries with that key; '
        f'leak streak {response.policy.streak}.'
    )
    if not response.policy.is_disabled(now):
        return text
    return f'{text} verify cache DISABLED until {_until(response.policy)}; re-enable with `{ENABLE_COMMAND}`.'
