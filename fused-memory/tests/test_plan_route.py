"""The plan route: config, the cap latch and its HTTP gate (task 9944).

Every plan-routed call goes through the ChatGPT-plan proxy. A cap parks the
work and pages L2; it never falls back onto OPENAI_API_KEY.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import openai
import pydantic
import pytest
from _fm_helpers import make_usage_limit_reached_error
from _mock_openai_server import MockOpenAIServer, chat_completion_body, mock_openai_server

from fused_memory.config.schema import FusedMemoryConfig, PlanRouteConfig
from fused_memory.services.plan_route import (
    PlanRoute,
    PlanRouteParkGuard,
    PlanRouteSite,
    PlanRouteSnapshot,
    PlanRouteStateName,
)
from fused_memory.services.provider_billing import (
    BillingExhaustion,
    BillingSource,
    classify_billing_exhaustion,
)

_PROXY_URL = 'http://127.0.0.1:8317/v1'


def _load_plan_route_yaml(leaf: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(f'plan_route:\n  {leaf}\n')
    monkeypatch.setenv('CONFIG_PATH', str(config_path))


class TestThePlanRoutesConfiguration:
    """Each test sets or deletes OPENAI_PLAN_PROXY_URL itself: server/main.py's
    import-time load_dotenv() can inject the main checkout's .env into this process.
    """

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('OPENAI_PLAN_PROXY_URL', raising=False)

        plan_route = FusedMemoryConfig().plan_route

        assert plan_route.proxy_url is None
        assert plan_route.probe_interval_seconds == 900.0

    @pytest.mark.parametrize('interval', ['0', '-5.0'])
    def test_a_non_positive_probe_interval_is_rejected_at_load(
        self, interval: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        _load_plan_route_yaml(f'probe_interval_seconds: {interval}', monkeypatch, tmp_path)

        with pytest.raises(pydantic.ValidationError) as caught:
            FusedMemoryConfig()

        message = str(caught.value)
        assert 'plan_route.probe_interval_seconds' in message
        assert f'input_value={interval}' in message

    def test_a_blank_proxy_url_reads_as_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        _load_plan_route_yaml("proxy_url: '  '", monkeypatch, tmp_path)

        assert FusedMemoryConfig().plan_route.proxy_url is None

    def test_the_shipped_config_routes_the_proxy_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('OPENAI_PLAN_PROXY_URL', _PROXY_URL)

        assert FusedMemoryConfig().plan_route.proxy_url == _PROXY_URL

    def test_the_shipped_config_leaves_the_proxy_unset_without_the_variable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv('OPENAI_PLAN_PROXY_URL', raising=False)

        assert FusedMemoryConfig().plan_route.proxy_url is None


_INTERVAL = 60.0
_START = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
_CHAT = '/chat/completions'


def _usage_limit_body(**reset: Any) -> dict[str, Any]:
    return {'error': {
        'type': 'usage_limit_reached', 'code': 'usage_limit_reached',
        'message': 'The usage limit has been reached', **reset,
    }}


class _Clock:
    def __init__(self) -> None:
        self.now = _START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class _Listener:
    def __init__(self) -> None:
        self.calls: list[tuple[str, PlanRouteSnapshot]] = []

    async def on_capped(self, snapshot: PlanRouteSnapshot) -> None:
        self.calls.append(('capped', snapshot))

    async def on_cap_confirmed(self, snapshot: PlanRouteSnapshot) -> None:
        self.calls.append(('confirmed', snapshot))

    async def on_recovered(self, snapshot: PlanRouteSnapshot) -> None:
        self.calls.append(('recovered', snapshot))

    def named(self, name: str) -> list[PlanRouteSnapshot]:
        return [snapshot for called, snapshot in self.calls if called == name]


class _Harness:
    def __init__(self, proxy: MockOpenAIServer) -> None:
        self.proxy = proxy
        self.clock = _Clock()
        self.listener = _Listener()
        self.route = PlanRoute(
            PlanRouteConfig(proxy_url=proxy.base_url, probe_interval_seconds=_INTERVAL),
            listener=self.listener, clock=self.clock,
        )

    @asynccontextmanager
    async def client(
        self, *, site: PlanRouteSite = PlanRouteSite.WRITE_TRIAGE_JUDGE, max_retries: int = 2,
    ) -> AsyncIterator[openai.AsyncOpenAI]:
        http_client = self.route.http_client_for(self.proxy.base_url, site=site)
        assert http_client is not None
        async with openai.AsyncOpenAI(
            base_url=self.proxy.base_url, api_key='k', max_retries=max_retries,
            http_client=http_client,
        ) as client:
            yield client

    def answer(self, *responses: Any) -> None:
        self.proxy.set_response_sequence(_CHAT, list(responses))

    def sent(self) -> int:
        return len(self.proxy.requests_to(_CHAT))


async def _chat(client: openai.AsyncOpenAI) -> Any:
    return await client.chat.completions.create(
        model='gpt-plan', messages=[{'role': 'user', 'content': 'hi'}],
    )


async def _assert_capped_call(client: openai.AsyncOpenAI) -> openai.RateLimitError:
    with pytest.raises(openai.RateLimitError) as caught:
        await _chat(client)
    exhaustion = classify_billing_exhaustion(caught.value)
    assert exhaustion is not None
    assert exhaustion.source is BillingSource.PLAN_CAP
    return caught.value


@pytest.fixture
def harness() -> Iterator[_Harness]:
    with mock_openai_server() as proxy:
        yield _Harness(proxy)


class TestWhichCallsTheRouteCovers:
    def test_only_the_proxy_base_url_gets_a_gated_client(self, harness: _Harness) -> None:
        site = PlanRouteSite.GRAPHITI
        assert harness.route.http_client_for('https://api.openai.com/v1', site=site) is None

    @pytest.mark.asyncio
    async def test_a_trailing_slash_still_matches(self, harness: _Harness) -> None:
        http_client = harness.route.http_client_for(
            harness.proxy.base_url + '/', site=PlanRouteSite.GRAPHITI,
        )
        assert http_client is not None
        await http_client.aclose()

    def test_an_unconfigured_route_gates_nothing(self) -> None:
        route = PlanRoute(PlanRouteConfig())
        assert route.http_client_for('http://127.0.0.1:8317/v1', site=PlanRouteSite.GRAPHITI) is None


class TestTheFirstCap:
    @pytest.mark.asyncio
    async def test_a_usage_limit_429_latches_after_one_request(self, harness: _Harness) -> None:
        harness.answer((429, _usage_limit_body()))

        async with harness.client(max_retries=2) as client:
            await _assert_capped_call(client)

        assert harness.sent() == 1
        state = harness.route.state()
        assert state.state is PlanRouteStateName.CAPPED
        assert state.capped_since == _START
        assert state.next_probe_at == _START + timedelta(seconds=_INTERVAL)
        assert len(harness.listener.named('capped')) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('response', 'resets_in', 'probe_in'),
        [
            ((429, _usage_limit_body(resets_in_seconds=30)), 30, 30),
            ((429, _usage_limit_body(resets_at=int(_START.timestamp()) + 3600)), 3600, _INTERVAL),
            ((429, _usage_limit_body(resets_at=(_START + timedelta(seconds=120)).isoformat())),
             120, _INTERVAL),
            ((429, _usage_limit_body(), {'Retry-After': '45'}), 45, 45),
        ],
        ids=['resets_in_seconds', 'epoch-resets_at', 'iso-resets_at', 'retry-after'],
    )
    async def test_the_reset_time_sets_the_next_probe(
        self, harness: _Harness, response: Any, resets_in: float, probe_in: float,
    ) -> None:
        harness.answer(response)

        async with harness.client() as client:
            await _assert_capped_call(client)

        state = harness.route.state()
        assert state.resets_at == _START + timedelta(seconds=resets_in)
        assert state.next_probe_at == _START + timedelta(seconds=probe_in)

    @pytest.mark.asyncio
    async def test_a_bare_429_is_a_cap(self, harness: _Harness) -> None:
        harness.answer((429, {}))

        async with harness.client() as client:
            await _assert_capped_call(client)

        assert harness.route.state().state is PlanRouteStateName.CAPPED
        assert harness.route.state().resets_at is None

    @pytest.mark.asyncio
    async def test_concurrent_first_caps_notify_once(self, harness: _Harness) -> None:
        harness.answer((429, _usage_limit_body()))

        async with harness.client() as client:
            results = await asyncio.gather(*(_chat(client) for _ in range(3)), return_exceptions=True)

        assert all(isinstance(result, openai.RateLimitError) for result in results)
        assert len(harness.listener.named('capped')) == 1


class TestWhileCapped:
    @pytest.mark.asyncio
    async def test_a_call_before_the_probe_is_refused_locally(self, harness: _Harness) -> None:
        harness.answer((429, _usage_limit_body()))
        async with harness.client() as client:
            await _assert_capped_call(client)
            harness.clock.advance(_INTERVAL / 2)

            await _assert_capped_call(client)

        assert harness.sent() == 1
        affected = harness.route.state().affected_calls_by_site
        assert affected[PlanRouteSite.WRITE_TRIAGE_JUDGE] == 2

    @pytest.mark.asyncio
    async def test_a_due_probe_admits_one_caller_and_a_429_confirms(self, harness: _Harness) -> None:
        harness.answer((429, _usage_limit_body()))
        async with harness.client() as client:
            await _assert_capped_call(client)
            harness.clock.advance(_INTERVAL + 1)

            results = await asyncio.gather(_chat(client), _chat(client), return_exceptions=True)

        assert all(isinstance(result, openai.RateLimitError) for result in results)
        assert harness.sent() == 2
        state = harness.route.state()
        assert state.state is PlanRouteStateName.CAPPED
        assert state.capped_since == _START
        assert state.next_probe_at == harness.clock.now + timedelta(seconds=_INTERVAL)
        assert len(harness.listener.named('confirmed')) == 1
        assert len(harness.listener.named('capped')) == 1

    @pytest.mark.asyncio
    async def test_a_probe_answered_2xx_recovers(self, harness: _Harness) -> None:
        harness.answer((429, _usage_limit_body()))
        async with harness.client() as client:
            await _assert_capped_call(client)
            harness.clock.advance(_INTERVAL + 1)
            harness.answer((200, chat_completion_body('{"ok": true}')))

            await _chat(client)

            assert harness.route.state().state is PlanRouteStateName.OK
            assert len(harness.listener.named('recovered')) == 1
            await _chat(client)

        assert harness.sent() == 3


class TestWhatIsNotACap:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', [500, 503])
    async def test_a_server_error_passes_through(self, harness: _Harness, status: int) -> None:
        harness.answer((status, {'error': {'message': 'proxy trouble'}}))

        async with harness.client(max_retries=0) as client:
            with pytest.raises(openai.InternalServerError):
                await _chat(client)

        assert harness.route.state().state is PlanRouteStateName.OK
        assert harness.listener.calls == []


class TestTheStatusField:
    @pytest.mark.asyncio
    async def test_a_capped_route_reports_its_latch(self, harness: _Harness) -> None:
        harness.answer((429, _usage_limit_body(resets_in_seconds=30)))
        async with harness.client(site=PlanRouteSite.WRITE_TRIAGE_JUDGE) as judge:
            await _assert_capped_call(judge)
            await _assert_capped_call(judge)
        async with harness.client(site=PlanRouteSite.GRAPHITI) as graphiti:
            await _assert_capped_call(graphiti)

        status = harness.route.state().to_status(parked_items=3, escalation_id='esc-x')

        assert status == {
            'state': 'capped',
            'proxy_host': f'{harness.proxy.host}:{harness.proxy.port}',
            'capped_since': _START.isoformat(),
            'resets_at': (_START + timedelta(seconds=30)).isoformat(),
            'next_probe_at': (_START + timedelta(seconds=30)).isoformat(),
            'parked_items': 3,
            'judge_fail_opens_since_latch': 2,
            'escalation_id': 'esc-x',
        }

    def test_an_ok_route_reports_no_latch(self, harness: _Harness) -> None:
        status = harness.route.state().to_status(parked_items=0, escalation_id=None)

        assert status['state'] == 'ok'
        assert status['capped_since'] is None
        assert status['judge_fail_opens_since_latch'] == 0

    def test_an_unconfigured_route_says_so(self) -> None:
        status = PlanRoute(PlanRouteConfig()).state().to_status(parked_items=0, escalation_id=None)

        assert status['state'] == 'unconfigured'
        assert status['proxy_host'] is None


_PARKED_SITE = 'durable_queue:add_episode'
_CLEARED_SITE = 'durable_queue'


def _plan_cap() -> BillingExhaustion:
    exhaustion = classify_billing_exhaustion(make_usage_limit_reached_error())
    assert exhaustion is not None
    return exhaustion


def _epoch_after(moment: datetime, seconds: float) -> float:
    return (moment + timedelta(seconds=seconds)).timestamp()


class TestACapAnUngatedPathReports:
    """The plan route's park guard: a plan cap that reached the durable queue
    by a path the route does not gate counts as a probe made outside the gate."""

    def test_an_ok_route_parks_a_reported_cap_one_probe_period_out(
        self, harness: _Harness,
    ) -> None:
        guard = PlanRouteParkGuard(harness.route)

        assert guard.park_until() == _epoch_after(_START, _INTERVAL)

    @pytest.mark.asyncio
    async def test_the_first_report_latches_the_route_once(self, harness: _Harness) -> None:
        guard = PlanRouteParkGuard(harness.route)

        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        state = harness.route.state()
        assert state.state is PlanRouteStateName.CAPPED
        assert state.next_probe_at == _START + timedelta(seconds=_INTERVAL)
        assert len(harness.listener.named('capped')) == 1
        assert guard.park_until() == _epoch_after(_START, _INTERVAL)

        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        assert len(harness.listener.calls) == 1
        assert guard.park_until() == _epoch_after(_START, _INTERVAL)

    @pytest.mark.asyncio
    async def test_a_report_once_the_probe_is_due_confirms_once_per_period(
        self, harness: _Harness,
    ) -> None:
        guard = PlanRouteParkGuard(harness.route)
        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)
        harness.clock.advance(_INTERVAL)

        assert guard.park_until() == _epoch_after(harness.clock.now, _INTERVAL)

        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        [confirmed] = harness.listener.named('confirmed')
        next_probe = harness.clock.now + timedelta(seconds=_INTERVAL)
        assert confirmed.next_probe_at == next_probe
        assert harness.route.state().next_probe_at == next_probe
        assert guard.park_until() == next_probe.timestamp()

        harness.clock.advance(_INTERVAL / 2)
        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        assert len(harness.listener.named('confirmed')) == 1

        harness.clock.advance(3.5 * _INTERVAL)
        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        assert len(harness.listener.named('confirmed')) == 2
        assert guard.park_until() >= _epoch_after(harness.clock.now, _INTERVAL)

    @pytest.mark.asyncio
    async def test_a_latch_only_ungated_evidence_set_recovers_when_a_parked_write_lands(
        self, harness: _Harness,
    ) -> None:
        guard = PlanRouteParkGuard(harness.route)
        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        await guard.on_cleared(site=_CLEARED_SITE)

        assert harness.route.state().state is PlanRouteStateName.OK
        assert len(harness.listener.named('recovered')) == 1

        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        assert harness.route.state().state is PlanRouteStateName.CAPPED
        assert len(harness.listener.named('capped')) == 2

    @pytest.mark.asyncio
    async def test_a_latch_the_gate_set_survives_clearing(self, harness: _Harness) -> None:
        guard = PlanRouteParkGuard(harness.route)
        harness.answer((429, _usage_limit_body()))
        async with harness.client() as client:
            await _assert_capped_call(client)

        await guard.on_cleared(site=_CLEARED_SITE)

        assert harness.route.state().state is PlanRouteStateName.CAPPED
        assert harness.listener.named('recovered') == []

    @pytest.mark.asyncio
    async def test_a_latch_the_gate_has_since_probed_survives_clearing(
        self, harness: _Harness,
    ) -> None:
        guard = PlanRouteParkGuard(harness.route)
        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)
        harness.clock.advance(_INTERVAL)
        harness.answer((429, _usage_limit_body()))
        async with harness.client() as client:
            await _assert_capped_call(client)
        assert harness.sent() == 1

        await guard.on_cleared(site=_CLEARED_SITE)

        assert harness.route.state().state is PlanRouteStateName.CAPPED
        assert harness.listener.named('recovered') == []

    @pytest.mark.asyncio
    async def test_an_unconfigured_route_latches_confirms_and_clears_the_same_way(self) -> None:
        config = PlanRouteConfig()
        clock = _Clock()
        listener = _Listener()
        route = PlanRoute(config, listener=listener, clock=clock)
        guard = PlanRouteParkGuard(route)

        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        state = route.state()
        assert state.state is PlanRouteStateName.CAPPED
        assert state.proxy_host is None
        assert len(listener.named('capped')) == 1

        clock.advance(config.probe_interval_seconds)

        assert guard.park_until() >= _epoch_after(clock.now, config.probe_interval_seconds)

        await guard.on_parked(_plan_cap(), site=_PARKED_SITE)

        assert len(listener.named('confirmed')) == 1

        await guard.on_cleared(site=_CLEARED_SITE)

        assert route.state().state is PlanRouteStateName.UNCONFIGURED
        assert len(listener.named('recovered')) == 1
