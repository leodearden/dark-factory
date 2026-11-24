"""One billing park policy, dispatching per billing source (task 9944).

Each BillingSource has its own guard: the API-key credits alarm, and the plan
route's latch. A plan cap must never reach the API-key page, and a parked
plan-cap row re-probes when the latch next lets a probe through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import openai
import pytest
from _fm_helpers import (
    make_insufficient_quota_error,
    make_transient_rate_limit_error,
    make_usage_limit_reached_error,
    wrap_like_graphiti,
)

from fused_memory.config.schema import PlanRouteConfig
from fused_memory.services.billing_park_policy import BillingParkPolicy
from fused_memory.services.durable_queue import ParkedWriteEvent
from fused_memory.services.plan_route import (
    PlanRoute,
    PlanRouteParkGuard,
    PlanRouteSnapshot,
    PlanRouteStateName,
)
from fused_memory.services.provider_billing import (
    BillingExhaustion,
    BillingSource,
    classify_billing_exhaustion,
)

_PROBE_AT = 1_800_000_000.0
_NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


@dataclass
class _RecordingGuard:
    until: float | None = None
    parked: list[tuple[BillingExhaustion, str]] = field(default_factory=list)
    cleared: list[str] = field(default_factory=list)

    def park_until(self) -> float | None:
        return self.until

    async def on_parked(self, exhaustion: BillingExhaustion, *, site: str) -> None:
        self.parked.append((exhaustion, site))

    async def on_cleared(self, *, site: str) -> None:
        self.cleared.append(site)


@dataclass
class _Guards:
    api_key: _RecordingGuard = field(default_factory=_RecordingGuard)
    plan: _RecordingGuard = field(default_factory=lambda: _RecordingGuard(until=_PROBE_AT))

    def policy(self) -> BillingParkPolicy:
        return BillingParkPolicy({
            BillingSource.API_KEY_CREDITS: self.api_key, BillingSource.PLAN_CAP: self.plan,
        })


def _event(exc: BaseException, operation: str = 'add_episode') -> ParkedWriteEvent:
    return ParkedWriteEvent(
        item_id=1, group_id='proj1', operation=operation, attempts=0, reason='r', exc=exc,
    )


def _server_error() -> openai.InternalServerError:
    request = httpx.Request('POST', 'http://127.0.0.1:8317/v1/responses')
    response = httpx.Response(503, request=request)
    return openai.InternalServerError('proxy unavailable', response=response, body=None)


class TestCoverage:
    def test_a_mapping_missing_a_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match='plan_cap'):
            BillingParkPolicy({BillingSource.API_KEY_CREDITS: _RecordingGuard()})


class TestDispatchBySource:
    @pytest.mark.asyncio
    async def test_a_plan_cap_parks_until_the_plan_guards_probe(self) -> None:
        guards = _Guards()
        exc = wrap_like_graphiti(make_usage_limit_reached_error())

        decision = guards.policy().park_decision(exc)
        await guards.policy().on_parked(_event(exc))

        assert decision is not None
        assert decision.retry_at == _PROBE_AT
        assert 'plan_cap' in decision.reason
        [(exhaustion, site)] = guards.plan.parked
        assert exhaustion.source is BillingSource.PLAN_CAP
        assert site == 'durable_queue:add_episode'
        assert guards.api_key.parked == []

    @pytest.mark.asyncio
    async def test_exhausted_credits_park_on_the_queue_horizon(self) -> None:
        guards = _Guards()
        exc = make_insufficient_quota_error()

        decision = guards.policy().park_decision(exc)
        await guards.policy().on_parked(_event(exc))

        assert decision is not None
        assert decision.retry_at is None
        [(exhaustion, _site)] = guards.api_key.parked
        assert exhaustion.source is BillingSource.API_KEY_CREDITS
        assert guards.plan.parked == []

    @pytest.mark.parametrize(
        'exc', [make_transient_rate_limit_error(), _server_error()], ids=['transient-429', '5xx'],
    )
    def test_anything_else_does_not_park(self, exc: BaseException) -> None:
        assert _Guards().policy().park_decision(exc) is None

    @pytest.mark.asyncio
    async def test_a_landed_parked_write_clears_every_guard(self) -> None:
        guards = _Guards()

        await guards.policy().on_cleared()

        assert guards.api_key.cleared == ['durable_queue']
        assert guards.plan.cleared == ['durable_queue']


class _CapListener:
    def __init__(self) -> None:
        self.capped: list[PlanRouteSnapshot] = []

    async def on_capped(self, snapshot: PlanRouteSnapshot) -> None:
        self.capped.append(snapshot)

    async def on_cap_confirmed(self, snapshot: PlanRouteSnapshot) -> None:
        raise AssertionError('no probe ran')

    async def on_recovered(self, snapshot: PlanRouteSnapshot) -> None:
        raise AssertionError('no probe ran')


class TestThePlanRoutesGuard:
    def _route(self, listener: _CapListener | None = None) -> PlanRoute:
        config = PlanRouteConfig(proxy_url='http://127.0.0.1:8317/v1', probe_interval_seconds=60)
        return PlanRoute(config, listener=listener, clock=lambda: _NOW)

    def test_an_ok_route_names_no_probe_time(self) -> None:
        assert PlanRouteParkGuard(self._route()).park_until() is None

    @pytest.mark.asyncio
    async def test_a_cap_by_an_ungated_path_latches_the_route_once(self) -> None:
        listener = _CapListener()
        route = self._route(listener)
        guard = PlanRouteParkGuard(route)
        plan_cap = classify_billing_exhaustion(make_usage_limit_reached_error())
        assert plan_cap is not None

        await guard.on_parked(plan_cap, site='durable_queue:add_episode')
        await guard.on_parked(plan_cap, site='durable_queue:add_episode')

        assert route.state().state is PlanRouteStateName.CAPPED
        assert len(listener.capped) == 1
        assert guard.park_until() == route.next_probe_epoch()
        assert guard.park_until() is not None

    @pytest.mark.asyncio
    async def test_clearing_leaves_the_route_to_learn_recovery_itself(self) -> None:
        route = self._route()
        guard = PlanRouteParkGuard(route)
        plan_cap = classify_billing_exhaustion(make_usage_limit_reached_error())
        assert plan_cap is not None
        await guard.on_parked(plan_cap, site='durable_queue:add_episode')

        await guard.on_cleared(site='durable_queue')

        assert route.state().state is PlanRouteStateName.CAPPED
