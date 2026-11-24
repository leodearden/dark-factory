"""One deduped level-2 record per ChatGPT-plan cap (task 9944).

Only Leo can upgrade the plan, so the first cap files ONE level-2 record at
severity ``blocking`` into the server's primary project root. Each probe that
re-confirms the cap refreshes that record in place, counting the repeat, and
files afresh only if the record was closed meanwhile. Recovery is logged and
leaves the record open for Leo to close. After a restart the first cap folds
into the record still pending, so nothing needs restoring.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from fused_memory.middleware._folded_escalation import (
    file_folded_escalation,
    refresh_folded_escalation,
)
from fused_memory.services.plan_route import PlanRouteSnapshot
from fused_memory.services.provider_billing import (
    BillingExhaustion,
    BillingSource,
    alarm_category_of,
    billing_evidence_lines,
    exhaustion_of,
)

logger = logging.getLogger(__name__)

ANCHOR_TASK_ID = 'plan-route-capped'
#: Never varies per call or per site: the dedupe fold is scoped to the role,
#: so a restart adopts the open record only while every filing uses this one.
_AGENT_ROLE = 'fused-memory/plan-route'
_SOURCE = BillingSource.PLAN_CAP
_CATEGORY = alarm_category_of(_SOURCE)
_LOG_LABEL = 'plan_route_capped'
_LEVEL = 2
_SEVERITY = 'blocking'
_UNKNOWN = 'unknown'


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PlanCapAlarm:
    """The plan route's listener: pages a cap once and keeps that page current.

    No method raises.
    """

    def __init__(
        self,
        resolve_project_root: Callable[[], str | None],
        count_parked: Callable[[], Awaitable[int]],
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._resolve_project_root = resolve_project_root
        self._count_parked = count_parked
        self._clock = clock
        self._escalation_id: str | None = None
        # Orders the queue I/O, so a re-confirmation cannot read the queue
        # before the page it must refresh has been written.
        self._queue_io = asyncio.Lock()

    def escalation_id(self) -> str | None:
        return self._escalation_id

    async def on_capped(self, snapshot: PlanRouteSnapshot) -> None:
        parked = await self._parked()
        async with self._queue_io:
            self._escalation_id = await self._file(snapshot, parked)

    async def on_cap_confirmed(self, snapshot: PlanRouteSnapshot) -> None:
        parked = await self._parked()
        async with self._queue_io:
            held = self._escalation_id
            if held is not None and await self._refresh(held, snapshot, parked):
                return
            self._escalation_id = await self._file(snapshot, parked)

    async def on_recovered(self, snapshot: PlanRouteSnapshot) -> None:
        logger.warning(
            '%s: the ChatGPT plan recovered at %s; %s stays open for Leo to close',
            _LOG_LABEL, self._clock().isoformat(), self._escalation_id or 'no filed record',
        )

    async def _parked(self) -> int | None:
        try:
            return await self._count_parked()
        except Exception:
            logger.warning('%s: could not count parked writes', _LOG_LABEL, exc_info=True)
            return None

    def _root(self) -> str | None:
        try:
            return self._resolve_project_root()
        except Exception:
            logger.warning('%s: could not resolve the project root', _LOG_LABEL, exc_info=True)
            return None

    async def _file(self, snapshot: PlanRouteSnapshot, parked: int | None) -> str | None:
        return await asyncio.to_thread(
            file_folded_escalation, self._root(),
            anchor_task_id=ANCHOR_TASK_ID,
            agent_role=_AGENT_ROLE,
            category=_CATEGORY,
            severity=_SEVERITY,
            level=_LEVEL,
            dedupe=True,
            summary=_summary(snapshot),
            detail=_detail(snapshot, parked),
            suggested_action=_suggested_action(snapshot),
            logger=logger,
            log_label=_LOG_LABEL,
        )

    async def _refresh(
        self, escalation_id: str, snapshot: PlanRouteSnapshot, parked: int | None,
    ) -> bool:
        return await asyncio.to_thread(
            refresh_folded_escalation, self._root(),
            escalation_id=escalation_id,
            level=_LEVEL,
            summary=_summary(snapshot),
            detail=_detail(snapshot, parked),
            logger=logger,
            log_label=_LOG_LABEL,
        )


def _iso(moment: datetime | None) -> str:
    return moment.isoformat() if moment is not None else _UNKNOWN


def _exhaustion(snapshot: PlanRouteSnapshot) -> BillingExhaustion:
    if snapshot.exhaustion is not None:
        return snapshot.exhaustion
    return exhaustion_of(_SOURCE, code=_UNKNOWN, error_type=_UNKNOWN, provider_message='')


def _summary(snapshot: PlanRouteSnapshot) -> str:
    return (
        f'ChatGPT plan capped since {_iso(snapshot.capped_since)} (resets '
        f'{_iso(snapshot.resets_at)}): plan-routed fused-memory work is PARKED and '
        'never falls back onto OPENAI_API_KEY'
    )


def _suggested_action(snapshot: PlanRouteSnapshot) -> str:
    return (
        f'upgrade the ChatGPT plan, or wait for the reset at {_iso(snapshot.resets_at)}; '
        'parked writes drain automatically when the cap clears'
    )


def _sites_affected(snapshot: PlanRouteSnapshot) -> str:
    counts = sorted(snapshot.affected_calls_by_site.items())
    return ','.join(f'{site.value}:{count}' for site, count in counts) or 'none'


def _detail(snapshot: PlanRouteSnapshot, parked: int | None) -> str:
    return '\n'.join([
        *billing_evidence_lines(_exhaustion(snapshot)),
        f'capped_since={_iso(snapshot.capped_since)}',
        f'resets_at={_iso(snapshot.resets_at)}',
        f'next_probe_at={_iso(snapshot.next_probe_at)}',
        f'proxy_host={snapshot.proxy_host or _UNKNOWN}',
        f'sites_affected={_sites_affected(snapshot)}',
        f'parked_items={_UNKNOWN if parked is None else parked}',
        f'judge_fail_opens={snapshot.judge_fail_opens}',
        '',
        'Plan-routed calls are refused locally until the next probe, which goes '
        'out at the earlier of the reset and one probe interval '
        '(plan_route.probe_interval_seconds). Durable writes that hit the cap '
        'are parked, never dead-lettered, and drain by themselves once a probe '
        'lands. The write-triage judge fails open (stores) meanwhile. This '
        'record is refreshed on each probe that re-confirms the cap and is not '
        'closed automatically.',
    ])
