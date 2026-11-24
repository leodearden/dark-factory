"""Gate plan-routed HTTP on the ChatGPT plan's cap latch (task 9944).

A call whose base_url is the plan proxy (config ``plan_route.proxy_url``) rides
an httpx transport this module owns. A proxy 429 latches the route capped.
Until the next probe every call is answered locally, and at probe time exactly
one caller goes through; a 2xx answer to it recovers the route.

Every cap and every refusal reaches the caller as one 429 marked
``usage_limit_reached`` with ``x-should-retry: false``. The openai SDK then
raises one RateLimitError without retrying, and
``provider_billing.classify_billing_exhaustion`` reads it as PLAN_CAP. A capped
plan never falls back onto OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
import openai

from fused_memory.config.schema import PlanRouteConfig
from fused_memory.services.provider_billing import BillingExhaustion, plan_cap_of

logger = logging.getLogger(__name__)


class PlanRouteSite(StrEnum):
    """A call site that rides the plan route."""

    WRITE_TRIAGE_JUDGE = 'write_triage_judge'
    GRAPHITI = 'graphiti'


class PlanRouteStateName(StrEnum):
    OK = 'ok'
    CAPPED = 'capped'
    UNCONFIGURED = 'unconfigured'


def _no_affected_calls() -> Mapping[PlanRouteSite, int]:
    return MappingProxyType({})


@dataclass(frozen=True)
class PlanRouteSnapshot:
    """The route's latch at one moment.

    ``affected_calls_by_site`` counts the calls each site lost to the cap since
    it latched.
    """

    state: PlanRouteStateName
    proxy_host: str | None
    capped_since: datetime | None = None
    resets_at: datetime | None = None
    next_probe_at: datetime | None = None
    affected_calls_by_site: Mapping[PlanRouteSite, int] = field(default_factory=_no_affected_calls)
    exhaustion: BillingExhaustion | None = None

    def to_status(self, *, parked_items: int, escalation_id: str | None) -> dict[str, Any]:
        judge_calls = self.affected_calls_by_site.get(PlanRouteSite.WRITE_TRIAGE_JUDGE, 0)
        return {
            'state': self.state.value,
            'proxy_host': self.proxy_host,
            'capped_since': _iso(self.capped_since),
            'resets_at': _iso(self.resets_at),
            'next_probe_at': _iso(self.next_probe_at),
            'parked_items': parked_items,
            'judge_fail_opens_since_latch': judge_calls,
            'escalation_id': escalation_id,
        }


class PlanCapListener(Protocol):
    """Told of each latch transition, awaited inline on the call that caused it."""

    async def on_capped(self, snapshot: PlanRouteSnapshot) -> None: ...

    async def on_cap_confirmed(self, snapshot: PlanRouteSnapshot) -> None: ...

    async def on_recovered(self, snapshot: PlanRouteSnapshot) -> None: ...


class _Admission(Enum):
    OPEN = 'open'
    PROBE = 'probe'
    REFUSED = 'refused'


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PlanRoute:
    """The cap latch (ok -> capped -> ok) and the gated clients that obey it.

    Every latch mutation runs with no await in between, so under asyncio a
    probe claim is atomic and concurrent callers see one transition.
    """

    def __init__(
        self,
        config: PlanRouteConfig,
        *,
        listener: PlanCapListener | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._proxy_url = _normalised(config.proxy_url) if config.proxy_url else None
        self._interval = timedelta(seconds=config.probe_interval_seconds)
        self._listener = listener
        self._clock = clock
        self._capped_since: datetime | None = None
        self._resets_at: datetime | None = None
        self._next_probe_at: datetime | None = None
        self._exhaustion: BillingExhaustion | None = None
        self._affected: dict[PlanRouteSite, int] = {}

    def http_client_for(self, base_url: str, *, site: PlanRouteSite) -> httpx.AsyncClient | None:
        """A gated client for *base_url*, or None when it is not the plan proxy."""
        if self._proxy_url is None or _normalised(base_url) != self._proxy_url:
            return None
        inner = httpx.AsyncHTTPTransport(limits=openai.DEFAULT_CONNECTION_LIMITS)
        return openai.DefaultAsyncHttpxClient(transport=_PlanRouteTransport(self, site, inner))

    def state(self) -> PlanRouteSnapshot:
        if self._capped_since is None:
            name = PlanRouteStateName.UNCONFIGURED if self._proxy_url is None else PlanRouteStateName.OK
            return PlanRouteSnapshot(state=name, proxy_host=self._proxy_host())
        return PlanRouteSnapshot(
            state=PlanRouteStateName.CAPPED,
            proxy_host=self._proxy_host(),
            capped_since=self._capped_since,
            resets_at=self._resets_at,
            next_probe_at=self._next_probe_at,
            affected_calls_by_site=MappingProxyType(dict(self._affected)),
            exhaustion=self._exhaustion,
        )

    def next_probe_epoch(self) -> float | None:
        """When a capped route next lets a probe through, or None while it is not capped."""
        if self._capped_since is None or self._next_probe_at is None:
            return None
        return self._next_probe_at.timestamp()

    async def observe_cap(self, exhaustion: BillingExhaustion, *, site: str) -> None:
        """Latch a cap that reached a caller by a path this route does not gate."""
        if self._capped_since is not None:
            return
        self._latch(exhaustion, resets_at=None, now=self._clock(), site=site)
        snapshot = self.state()
        await self._notify(lambda listener: listener.on_capped(snapshot))

    async def _send(
        self, request: httpx.Request, *, site: PlanRouteSite, inner: httpx.AsyncBaseTransport,
    ) -> httpx.Response:
        admission = self._admit()
        if admission is _Admission.REFUSED:
            self._count(site)
            return self._cap_response(request)
        response = await inner.handle_async_request(request)
        if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
            if response.is_success and admission is _Admission.PROBE:
                await self._recover()
            return response
        body = await response.aread()
        await response.aclose()
        exhaustion = plan_cap_of(response.status_code, body)
        if exhaustion is None:
            return _replayed(response, body, request)
        await self._on_cap(
            exhaustion, resets_at=_resets_at(body, response.headers, now=self._clock()),
            site=site, probe=admission is _Admission.PROBE,
        )
        return self._cap_response(request)

    def _admit(self) -> _Admission:
        if self._capped_since is None:
            return _Admission.OPEN
        now = self._clock()
        if self._next_probe_at is not None and now < self._next_probe_at:
            return _Admission.REFUSED
        self._next_probe_at = now + self._interval
        return _Admission.PROBE

    async def _on_cap(
        self, exhaustion: BillingExhaustion, *, resets_at: datetime | None,
        site: PlanRouteSite, probe: bool,
    ) -> None:
        now = self._clock()
        if self._capped_since is None:
            self._latch(exhaustion, resets_at=resets_at, now=now, site=site)
            self._count(site)
            snapshot = self.state()
            await self._notify(lambda listener: listener.on_capped(snapshot))
            return
        self._count(site)
        if not probe:
            return
        self._exhaustion = exhaustion
        self._resets_at = resets_at or self._resets_at
        self._next_probe_at = self._next_probe(now)
        snapshot = self.state()
        await self._notify(lambda listener: listener.on_cap_confirmed(snapshot))

    def _latch(
        self, exhaustion: BillingExhaustion, *, resets_at: datetime | None, now: datetime, site: str,
    ) -> None:
        self._capped_since = now
        self._exhaustion = exhaustion
        self._resets_at = resets_at
        self._next_probe_at = self._next_probe(now)
        self._affected = {}
        logger.warning(
            'plan route: CAPPED since %s (%s, seen by %s); plan-routed calls are refused '
            'locally until the next probe at %s, and never fall back onto OPENAI_API_KEY',
            _iso(now), exhaustion.code, site, _iso(self._next_probe_at),
        )

    async def _recover(self) -> None:
        if self._capped_since is None:
            return
        logger.warning(
            'plan route: recovered at %s after a cap since %s',
            _iso(self._clock()), _iso(self._capped_since),
        )
        self._capped_since = None
        self._resets_at = None
        self._next_probe_at = None
        self._exhaustion = None
        self._affected = {}
        snapshot = self.state()
        await self._notify(lambda listener: listener.on_recovered(snapshot))

    def _next_probe(self, now: datetime) -> datetime:
        ceiling = now + self._interval
        if self._resets_at is not None and now < self._resets_at < ceiling:
            return self._resets_at
        return ceiling

    def _count(self, site: PlanRouteSite) -> None:
        self._affected[site] = self._affected.get(site, 0) + 1

    def _cap_response(self, request: httpx.Request) -> httpx.Response:
        provider_body = self._exhaustion.provider_message if self._exhaustion else ''
        message = (
            f'ChatGPT plan capped since {_iso(self._capped_since)}; '
            f'next probe at {_iso(self._next_probe_at)}'
        )
        return httpx.Response(
            httpx.codes.TOO_MANY_REQUESTS,
            headers={'x-should-retry': 'false'},
            json={'error': {
                'code': 'usage_limit_reached', 'type': 'usage_limit_reached',
                'message': message, 'provider_body': provider_body,
            }},
            request=request,
        )

    async def _notify(self, hook: Callable[[PlanCapListener], Awaitable[None]]) -> None:
        if self._listener is None:
            return
        try:
            await hook(self._listener)
        except Exception:
            logger.exception('plan route: the cap listener failed; the call it rode on carries on')

    def _proxy_host(self) -> str | None:
        return urlsplit(self._proxy_url).netloc if self._proxy_url else None


class _PlanRouteTransport(httpx.AsyncBaseTransport):
    def __init__(
        self, route: PlanRoute, site: PlanRouteSite, inner: httpx.AsyncBaseTransport,
    ) -> None:
        self._route = route
        self._site = site
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._route._send(request, site=self._site, inner=self._inner)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _replayed(response: httpx.Response, body: bytes, request: httpx.Request) -> httpx.Response:
    """*response* rebuilt around its already-decoded *body*."""
    headers = [
        (name, value) for name, value in response.headers.items()
        if name.lower() not in {'content-encoding', 'content-length', 'transfer-encoding'}
    ]
    return httpx.Response(response.status_code, headers=headers, content=body, request=request)


def _normalised(url: str) -> str:
    return url.strip().rstrip('/')


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def _resets_at(body: bytes, headers: Mapping[str, str], *, now: datetime) -> datetime | None:
    """When the cap a proxy 429 reports resets: the body's resets_at or
    resets_in_seconds (top level or under 'error'), else Retry-After seconds."""
    for fields in _reset_fields(body):
        instant = _instant(fields.get('resets_at'))
        if instant is not None:
            return instant
        seconds = _seconds(fields.get('resets_in_seconds'))
        if seconds is not None:
            return now + timedelta(seconds=seconds)
    seconds = _seconds(headers.get('retry-after'))
    return now + timedelta(seconds=seconds) if seconds is not None else None


def _reset_fields(body: bytes) -> list[Mapping[str, object]]:
    try:
        decoded = json.loads(body)
    except ValueError:
        return []
    if not isinstance(decoded, dict):
        return []
    nested = decoded.get('error')
    return [nested, decoded] if isinstance(nested, dict) else [decoded]


def _instant(value: object) -> datetime | None:
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, int | float):
            return datetime.fromtimestamp(value, UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except (ValueError, OverflowError, OSError):
        return None
    return None


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        seconds = float(value) if isinstance(value, int | float | str) else None
    except ValueError:
        return None
    return seconds if seconds is not None and math.isfinite(seconds) and seconds >= 0 else None
