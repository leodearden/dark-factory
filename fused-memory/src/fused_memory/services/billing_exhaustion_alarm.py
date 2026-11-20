"""One born-at-L2 page per episode of exhausted OpenAI API-key credits.

Exhausted credits clear only when a human pays, so the page goes straight to
L2 at severity ``critical``. Only a ``harness-``/``orchestrator-`` sentinel
role may file at a born-at-L2 severity
(``escalation/src/escalation/models.py::HARNESS_SENTINEL_ROLE_PREFIXES``), and
this background guard is one. One ``OPENAI_API_KEY`` serves every project, so
ONE record lands in the server's primary project root however many writes fail.

The latch is process-local and flips before the first await, so concurrent
failures page once. The first successful OpenAI-backed write resolves the
record, and :meth:`BillingExhaustionAlarm.restore` re-latches onto a record a
previous process left open, so a restart neither re-pages nor strands it.

:class:`BillingParkPolicy` is the durable queue's side of it: the queue parks
what the policy names and reports parking and clearing back to the alarm.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from fused_memory.middleware._folded_escalation import (
    file_folded_escalation,
    pending_folded_escalations,
    resolve_folded_escalations,
)
from fused_memory.services.durable_queue import ParkedWriteEvent
from fused_memory.services.provider_billing import (
    BillingExhaustion,
    BillingSource,
    classify_billing_exhaustion,
    exhaustion_of,
)

logger = logging.getLogger(__name__)

ANCHOR_TASK_ID = 'provider-billing-exhausted'
_AGENT_ROLE = 'harness-fused-memory-billing-guard'
_CATEGORY = 'provider_billing_exhausted'
_LOG_LABEL = 'provider_billing_exhausted'
#: The anchor is per billing source, so a re-latched record names its source.
_SOURCE = BillingSource.API_KEY_CREDITS

_SUGGESTED_ACTION = (
    "Top up the OpenAI organisation's credits behind OPENAI_API_KEY. Do NOT "
    'replay dead letters while this record is open: they would fail again into '
    'the same empty balance. Parked writes and this record clear themselves '
    'once a write succeeds.'
)


@dataclass(frozen=True)
class BillingAlarmState:
    """The open episode: what ran out, since when, and the page filed for it."""

    exhaustion: BillingExhaustion
    first_seen: datetime
    first_site: str
    escalation_id: str | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BillingExhaustionAlarm:
    """Latch exhausted credits into one page, and resolve it on recovery.

    No method raises.
    """

    def __init__(
        self,
        resolve_project_root: Callable[[], str | None],
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._resolve_project_root = resolve_project_root
        self._clock = clock
        self._state: BillingAlarmState | None = None
        # Orders the queue I/O, so a recovery's resolve cannot read the queue
        # before the page it must close has been written.
        self._queue_io = asyncio.Lock()

    def state(self) -> BillingAlarmState | None:
        return self._state

    async def record_exhausted(self, exhaustion: BillingExhaustion, *, site: str) -> None:
        if self._state is not None:
            logger.debug(
                '%s: already latched since %s; %s folds into it',
                _LOG_LABEL, self._state.first_seen.isoformat(), site,
            )
            return
        latched = BillingAlarmState(
            exhaustion=exhaustion, first_seen=self._clock(), first_site=site,
            escalation_id=None,
        )
        self._state = latched
        logger.warning(
            '%s: %s credits exhausted (%s) at %s; parking writes until an '
            'OpenAI-backed write succeeds',
            _LOG_LABEL, exhaustion.provider, exhaustion.source.value, site,
        )
        try:
            escalation_id = await self._page(latched)
        except Exception:
            logger.exception('%s: failed to page for exhausted credits', _LOG_LABEL)
            return
        if escalation_id is not None and self._state is latched:
            self._state = replace(latched, escalation_id=escalation_id)

    async def record_recovered(self, *, site: str) -> None:
        latched = self._state
        if latched is None:
            return
        self._state = None
        recovered_at = self._clock()
        logger.warning(
            '%s: %s succeeded at %s; credits exhausted since %s are back',
            _LOG_LABEL, site, recovered_at.isoformat(), latched.first_seen.isoformat(),
        )
        resolution = (
            f'{site} succeeded at {recovered_at.isoformat()}, so credits are back. '
            f'Exhausted since {latched.first_seen.isoformat()}, first seen at '
            f'{latched.first_site}.'
        )
        try:
            root = self._resolve_project_root()
            if root is None:
                return
            async with self._queue_io:
                await asyncio.to_thread(
                    resolve_folded_escalations, root,
                    anchor_task_id=ANCHOR_TASK_ID, agent_role=_AGENT_ROLE,
                    resolution=resolution, logger=logger, log_label=_LOG_LABEL,
                )
        except Exception:
            logger.exception('%s: failed to resolve the exhausted-credits page', _LOG_LABEL)

    async def restore(self) -> None:
        """Re-latch onto a page a previous process left open."""
        try:
            root = self._resolve_project_root()
            if root is None:
                return
            async with self._queue_io:
                pending = await asyncio.to_thread(
                    pending_folded_escalations, root,
                    anchor_task_id=ANCHOR_TASK_ID, agent_role=_AGENT_ROLE,
                    logger=logger, log_label=_LOG_LABEL,
                )
            if not pending or self._state is not None:
                return
            record = pending[0]
            self._state = BillingAlarmState(
                exhaustion=exhaustion_of(
                    _SOURCE, error_type='unknown',
                    provider_message=f'the error predates this process; see {record.id}',
                ),
                first_seen=datetime.fromisoformat(record.timestamp),
                first_site='before the last restart',
                escalation_id=record.id,
            )
        except Exception:
            logger.exception('%s: failed to restore an open page', _LOG_LABEL)
            return
        logger.warning(
            '%s: re-latched onto %s, open since %s',
            _LOG_LABEL, record.id, record.timestamp,
        )

    async def _page(self, latched: BillingAlarmState) -> str | None:
        root = self._resolve_project_root()
        if root is None:
            logger.warning(
                '%s: no primary project root is configured, so credits exhausted '
                'since %s cannot be paged', _LOG_LABEL, latched.first_seen.isoformat(),
            )
            return None
        async with self._queue_io:
            return await asyncio.to_thread(
                file_folded_escalation, root,
                anchor_task_id=ANCHOR_TASK_ID,
                agent_role=_AGENT_ROLE,
                category=_CATEGORY,
                severity='critical',
                level=2,
                dedupe=True,
                summary=_summary(latched),
                detail=_detail(latched),
                suggested_action=_SUGGESTED_ACTION,
                logger=logger,
                log_label=_LOG_LABEL,
            )


def _one_line(text: str) -> str:
    return ' '.join(text.split())


def _summary(latched: BillingAlarmState) -> str:
    return (
        f'OpenAI credits exhausted: {latched.exhaustion.key_scope} has no credits '
        f'remaining (insufficient_quota) since {latched.first_seen.isoformat()}; '
        'fused-memory writes are PARKED, not dead-lettered'
    )


def _detail(latched: BillingAlarmState) -> str:
    exhaustion = latched.exhaustion
    return '\n'.join([
        f'provider={exhaustion.provider}',
        f'billing_source={exhaustion.source.value}',
        f'key_scope={exhaustion.key_scope}',
        f'first_seen={latched.first_seen.isoformat()}',
        f'first_site={latched.first_site}',
        f'error_type={exhaustion.error_type}',
        f'provider_message={_one_line(exhaustion.provider_message)}',
        '',
        'Durable writes that hit this are parked, not dead-lettered: they drain '
        'by themselves, and this record resolves itself, on the first '
        'successful OpenAI-backed write. Mem0 writes keep retrying and land the '
        'same way. Dead letters filed before parking existed (task 10922) stay '
        'replayable with replay_dead_letters once credits are restored.',
    ])


class BillingParkPolicy:
    """Parks the durable writes a provider refuses for exhausted billing.

    Satisfies ``durable_queue.py::ParkPolicy``; a parked write that lands is
    the recovery evidence.
    """

    def __init__(self, alarm: BillingExhaustionAlarm) -> None:
        self._alarm = alarm

    def park_reason(self, exc: BaseException) -> str | None:
        exhaustion = classify_billing_exhaustion(exc)
        if exhaustion is None:
            return None
        return f'{exhaustion.provider} {exhaustion.source.value} exhausted (insufficient_quota)'

    async def on_parked(self, event: ParkedWriteEvent) -> None:
        exhaustion = classify_billing_exhaustion(event.exc)
        if exhaustion is not None:
            await self._alarm.record_exhausted(exhaustion, site=f'durable_queue:{event.operation}')

    async def on_cleared(self) -> None:
        await self._alarm.record_recovered(site='durable_queue')
