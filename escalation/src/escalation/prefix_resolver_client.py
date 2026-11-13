"""The uuid-prefix guard's resolver port, answered by fused-memory over HTTP.

:class:`PrefixResolverClient` implements ``shared.uuid_prefix_guard``'s
``PrefixResolver`` port over fused-memory's ``POST /resolve-prefix``. It is a
pure function of ``(project_id, prefix)``: every call opens its own connection,
and the only thing shared between calls is the TLS configuration, which holds
no per-request state.

It fails OPEN, never quietly (INV-11). ANY failure to obtain a ``Resolution``
(a transport error, a timeout, an unusable *base_url*, a non-200 status, a
malformed 200) raises ``ResolverUnavailable``, the one failure the C2 port
declares, so the guard forwards the call unchanged and says so on ``meta``; it
would re-raise anything else as a resolver bug. Only cancellation propagates,
and no failure is ever read as a ``none``. A 503 names the
store that failed, because "mem0 is down" is actionable where "the resolver is
down" is not. Each failure also names WHY, as an ``UnavailableReason``:
``timed_out`` past a deadline, ``unreachable`` for no connection, an unusable
*base_url* or a 503, and ``bad_response`` for any other status or a malformed
200.

The bound (INV-8): one request's TOTAL deadline is
:data:`REQUEST_DEADLINE_SECONDS`, so this guard occupies a call for at most
ceil(MAX_DISTINCT_PREFIXES_PER_CALL / MAX_CONCURRENT_RESOLUTIONS) x 5 s = 10 s
of awaited I/O. Why 5 s against measured route latency:
``plans/uuid-prefix-resolution-prd.md`` §4-C5, erratum 2026-11-05. That
deadline cannot interrupt synchronous work, which blocks the shared event loop.
Each resolution's synchronous work is building a client and decoding a body,
a fraction of a millisecond. The TLS context costs ~10 ms
to build, so :func:`_tls_context` builds it once per process.

The route path and field names are restated here because escalation may not
import fused_memory. ``fused-memory/tests/test_prefix_resolver_client_contract.py``
round-trips this client against the real route, which keeps the two copies in
lockstep. Contract: ``plans/uuid-prefix-resolution-prd.md`` §4-C5.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import ssl
from dataclasses import dataclass
from typing import TypeGuard

import httpx
from shared.uuid_prefix_guard import (
    NAMESPACES,
    RESOLUTION_OUTCOMES,
    Candidate,
    Namespace,
    Resolution,
    ResolutionOutcome,
    ResolverUnavailable,
    UnavailableReason,
)

__all__ = [
    'REQUEST_DEADLINE_SECONDS',
    'RESOLVER_SERVICE_STORE',
    'RESOLVE_PREFIX_PATH',
    'PrefixResolverClient',
]

logger = logging.getLogger(__name__)

RESOLVE_PREFIX_PATH = '/resolve-prefix'
REQUEST_DEADLINE_SECONDS = 5.0
RESOLVER_SERVICE_STORE = 'fused-memory'


class _NoResolution(Exception):
    """A response arrived but is no Resolution: what it was, why, and which store to blame."""

    def __init__(
        self,
        detail: str,
        reason: UnavailableReason = UnavailableReason.BAD_RESPONSE,
        store: str = RESOLVER_SERVICE_STORE,
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.store = store


@functools.cache
def _tls_context() -> ssl.SSLContext:
    """httpx's default verifying context, whose CA-bundle load blocks the loop."""
    return httpx.create_ssl_context()


def _describe(exc: BaseException) -> str:
    """Name the leaf errors: an anyio ExceptionGroup's own message hides its cause."""
    if isinstance(exc, BaseExceptionGroup):
        return '; '.join(_describe(leaf) for leaf in exc.exceptions)
    return f'{type(exc).__name__}: {exc}'


def _reason_for(exc: BaseException) -> UnavailableReason:
    """A deadline, ours or httpx's, is a timeout — in an anyio group only if every leaf is."""
    if isinstance(exc, BaseExceptionGroup):
        timed_out = all(_reason_for(leaf) is UnavailableReason.TIMED_OUT for leaf in exc.exceptions)
        return UnavailableReason.TIMED_OUT if timed_out else UnavailableReason.UNREACHABLE
    if isinstance(exc, TimeoutError | httpx.TimeoutException):
        return UnavailableReason.TIMED_OUT
    return UnavailableReason.UNREACHABLE


def _is_outcome(value: object) -> TypeGuard[ResolutionOutcome]:
    return value in RESOLUTION_OUTCOMES


def _is_namespace(value: object) -> TypeGuard[Namespace]:
    return value in NAMESPACES


def _candidate(raw: object) -> Candidate:
    if not isinstance(raw, dict):
        raise _NoResolution(f'HTTP 200 carried a candidate that is not an object: {raw!r}')
    namespace, candidate_id, preview = raw.get('namespace'), raw.get('id'), raw.get('preview')
    if not _is_namespace(namespace):
        raise _NoResolution(f'HTTP 200 carried a candidate namespace {namespace!r} outside {NAMESPACES}')
    if not isinstance(candidate_id, str) or not isinstance(preview, str):
        raise _NoResolution(f'HTTP 200 carried a candidate without a string id and preview: {raw!r}')
    return Candidate(namespace, candidate_id, preview)


def _resolution(response: httpx.Response) -> Resolution:
    try:
        body = response.json()
    except ValueError as exc:
        raise _NoResolution(f'HTTP 200 body is not JSON: {exc}') from exc
    if not isinstance(body, dict):
        raise _NoResolution(f'HTTP 200 body is not a JSON object: {body!r}')
    outcome, candidates = body.get('outcome'), body.get('candidates')
    if not _is_outcome(outcome):
        raise _NoResolution(f'HTTP 200 carried outcome {outcome!r} outside {RESOLUTION_OUTCOMES}')
    if not isinstance(candidates, list):
        raise _NoResolution(f'HTTP 200 carried candidates that are not a list: {candidates!r}')
    return Resolution(outcome, tuple(_candidate(raw) for raw in candidates))


def _no_resolution_from_status(response: httpx.Response) -> _NoResolution:
    """A 503 is the service naming a store it could not reach; any other status is a bad answer."""
    detail = f'HTTP {response.status_code}'
    if response.status_code == 503:
        return _NoResolution(detail, UnavailableReason.UNREACHABLE, _unavailable_store(response))
    return _NoResolution(detail)


def _unavailable_store(response: httpx.Response) -> str:
    """A 503 body's ``store``, or the service as a whole when the body names none."""
    try:
        body = response.json()
    except ValueError:
        return RESOLVER_SERVICE_STORE
    store = body.get('store') if isinstance(body, dict) else None
    return store if isinstance(store, str) else RESOLVER_SERVICE_STORE


@dataclass(frozen=True)
class PrefixResolverClient:
    """``resolve_prefix(project_id, prefix)`` answered by fused-memory at *base_url*.

    *transport* is httpx's own seam, for a caller that must reach the route
    without a socket.
    """

    base_url: str
    timeout_seconds: float = REQUEST_DEADLINE_SECONDS
    transport: httpx.AsyncBaseTransport | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(
                f'timeout_seconds must be positive, got {self.timeout_seconds!r}: a non-positive '
                'deadline abandons every resolution before it is sent'
            )

    @property
    def url(self) -> str:
        return self.base_url.rstrip('/') + RESOLVE_PREFIX_PATH

    async def resolve_prefix(self, project_id: str, prefix: str) -> Resolution:
        try:
            return await self._request(project_id, prefix)
        except _NoResolution as exc:
            detail, reason, store, cause = str(exc), exc.reason, exc.store, exc
        except Exception as exc:
            detail, reason, store, cause = _describe(exc), _reason_for(exc), RESOLVER_SERVICE_STORE, exc
        logger.warning(
            'prefix resolver: POST %s for project=%s prefix=%s got no resolution (%s: %s); '
            'the %s store is treated as unavailable',
            self.url, project_id, prefix, reason, detail, store,
        )
        raise ResolverUnavailable(store, reason=reason) from cause

    async def _request(self, project_id: str, prefix: str) -> Resolution:
        async with (
            asyncio.timeout(self.timeout_seconds),
            httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport, verify=_tls_context(),
            ) as client,
        ):
            response = await client.post(self.url, json={'project_id': project_id, 'prefix': prefix})
        if response.status_code != 200:
            raise _no_resolution_from_status(response)
        return _resolution(response)
