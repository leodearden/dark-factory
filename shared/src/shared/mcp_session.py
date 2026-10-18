"""The MCP client the orchestrator and scripts use, async and sync.

The dashboard still carries its own twin; ``shared/mcp_post.py`` tracks it.

:class:`McpSession` serves long-lived async processes: a Streamable-HTTP
session with the ``initialize`` handshake, transport retries on a schedule the
caller injects, and one ``client_op_id`` per logical call to a mutating task
tool. :class:`SyncMcpSession` serves scripts: one handshake per ``with``
block, no retries. Both send the one handshake :func:`_initialize_params`
spells, with an injected ``clientInfo.name`` — the identity fused-memory
records for the caller — and both fail with one type, :class:`McpCallError`,
whose :class:`McpFailureKind` says what failed. Endpoint, headers and body
decoding come from :mod:`shared.mcp_post`.

Not re-exported from ``shared/__init__.py``; import ``shared.mcp_session``.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType, TracebackType
from typing import Any, TypeVar

import httpx

from shared.mcp_envelope import EnvelopeShape, parse_tool_result
from shared.mcp_idempotency import maybe_inject_client_op_id
from shared.mcp_post import MCP_POST_HEADERS, decode_mcp_response_body, mcp_endpoint_url

__all__ = [
    'MCP_PROTOCOL_VERSION',
    'McpCallError',
    'McpFailureKind',
    'McpSession',
    'SyncMcpSession',
    'is_timeout_failure',
]

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

MCP_PROTOCOL_VERSION = '2025-03-26'


class McpFailureKind(enum.StrEnum):
    TRANSPORT = 'TRANSPORT'
    """No usable HTTP answer: connection or timeout after retries, or a redirect loop."""

    HTTP_STATUS = 'HTTP_STATUS'
    """A non-2xx answer, ``status`` naming it: a 502/503/504 only once retries run out."""

    UNDECODABLE = 'UNDECODABLE'
    """The body is neither JSON nor SSE-framed JSON, an empty answer included."""

    JSON_RPC_ERROR = 'JSON_RPC_ERROR'
    """The JSON-RPC body carries an ``error`` member."""

    TOOL_ERROR = 'TOOL_ERROR'
    """``result.isError`` is set."""

    REPLY_SHAPE = 'REPLY_SHAPE'
    """The result is not a usable tool payload."""


class McpCallError(RuntimeError):
    """The one failure type of the MCP client; ``kind`` says what failed.

    The transport exception, when there is one, is chained as ``__cause__``.
    """

    def __init__(self, kind: McpFailureKind, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


def _initialize_params(client_name: str) -> dict[str, Any]:
    return {
        'protocolVersion': MCP_PROTOCOL_VERSION,
        'capabilities': {},
        'clientInfo': {'name': client_name, 'version': '0.1.0'},
    }


def _request_headers(session_id: str | None) -> dict[str, str]:
    headers = dict(MCP_POST_HEADERS)
    if session_id:
        headers['Mcp-Session-Id'] = session_id
    return headers


# A gateway answer means the server is restarting or overloaded, so the same
# request is worth sending again.
_RETRYABLE_STATUS = frozenset({502, 503, 504})

# Exceptions treated as transient for retry purposes.
#
# httpx.NetworkError is the parent of ConnectError, ReadError, WriteError and
# CloseError — cover them all so mid-transfer drops on a 6MB response also
# retry instead of blowing up the caller.  We also include bare OSError
# because httpx's map_httpcore_exceptions() only re-wraps exceptions it
# recognises from httpcore; low-level asyncio/anyio socket failures can
# occasionally surface as raw OSError (including FileNotFoundError, which
# stringifies to the unhelpful "[Errno 2] No such file or directory" with
# no filename).  Retrying these lets the client recover transparently and,
# if it keeps failing, the wrapping McpCallError preserves the type name
# so the top-level catch-all logs something diagnostic.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
    httpx.NetworkError,
    OSError,
)

_TIMEOUT_EXCEPTIONS: tuple[type[BaseException], ...] = (TimeoutError, httpx.TimeoutException)
_CAUSE_CHAIN_LIMIT = 20


def is_timeout_failure(exc: BaseException) -> bool:
    """Was *exc* caused, at any depth, by something timing out?

    Inverts the wrap performed below: ``McpSession`` retries
    ``httpx.TimeoutException`` along with the rest of
    ``_RETRYABLE_EXCEPTIONS`` and, on exhaustion, raises a plain
    ``McpCallError`` from the last failure's cause. That erases the timeout
    from the exception's TYPE while preserving it as ``__cause__`` — so a caller
    holding the raised error cannot tell a service that is alive and slow
    from one that is unreachable by an ``isinstance`` check, only by
    unwrapping. This is that unwrap, and it lives here, beside the wrap it
    inverts, so the pairing has one home (SPOT) and callers such as
    ``agents/briefing.py`` need know nothing of httpx's exception taxonomy.

    ``asyncio.TimeoutError`` is an alias of the builtin from 3.11 on, so
    naming ``TimeoutError`` covers both. The walk is bounded and
    cycle-guarded: ``__cause__`` is writable, so a chain can be circular,
    and a diagnostic predicate must never be the thing that hangs a caller.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_CAUSE_CHAIN_LIMIT):
        if current is None or id(current) in seen:
            return False
        if isinstance(current, _TIMEOUT_EXCEPTIONS):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False


def _detached(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A copy the session may annotate (``client_op_id``) without writing to the caller's dict."""
    if params is None:
        return None
    copied = dict(params)
    arguments = copied.get('arguments')
    if isinstance(arguments, Mapping):
        copied['arguments'] = dict(arguments)
    return copied


def _decoded(method: str, resp: httpx.Response) -> Any:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise McpCallError(
            McpFailureKind.HTTP_STATUS,
            f'MCP {method} answered HTTP {resp.status_code}: {resp.text[:200]}',
            status=resp.status_code,
        ) from exc
    try:
        return decode_mcp_response_body(resp)
    except ValueError as exc:
        raise McpCallError(
            McpFailureKind.UNDECODABLE, f'MCP {method} answer could not be decoded: {exc}',
        ) from exc


class _Transient(Exception):
    """One attempt's retryable failure; ``status`` is set when it was a gateway answer."""

    def __init__(self, cause: Exception, *, status: int | None = None) -> None:
        super().__init__(f'{type(cause).__name__}: {cause}')
        self.cause = cause
        self.status = status

    def exhausted(self, label: str, attempts: int) -> McpCallError:
        kind = McpFailureKind.TRANSPORT if self.status is None else McpFailureKind.HTTP_STATUS
        return McpCallError(
            kind, f'MCP {label} failed after {attempts} attempts: {self}', status=self.status,
        )


class McpSession:
    """An MCP Streamable-HTTP session for a long-lived async process.

    *retry_backoffs* is the sleep before each retry of a transient failure, so
    ``len(retry_backoffs) + 1`` attempts are made; the default ``()`` makes one.
    One schedule covers a whole logical call, its handshake included. A
    transient failure drops the session, since a restarted server has
    forgotten it, so the next attempt runs the handshake again before it
    re-posts the same request, ``client_op_id`` and all.

    ``call_tool`` returns the JSON-RPC body as the server sent it, a JSON-RPC
    error or ``result.isError`` reply included: reading the tool's verdict is
    ``shared.mcp_envelope.parse_tool_result``'s job.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client_name: str,
        retry_backoffs: Sequence[float] = (),
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.mcp_endpoint = mcp_endpoint_url(self.base_url)
        self._client_name = client_name
        self._retry_backoffs = tuple(retry_backoffs)
        self._session_id: str | None = None
        self._initialized = False
        self._request_id = 0

    async def initialize(self) -> None:
        """Run the ``initialize`` + ``notifications/initialized`` handshake, once."""
        await self._with_retries('initialize', self._ensure_handshake)

    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float = 30,
    ) -> dict:
        """Send one JSON-RPC request, running the handshake first if needed."""
        detached = _detached(params)
        # Once per logical call, before any attempt, so every retry posts the
        # same key.
        maybe_inject_client_op_id(method, detached)
        payload = self._message(method, detached)

        async def attempt() -> dict:
            await self._ensure_handshake()
            return await self._exchange(payload, timeout)

        return await self._with_retries(method, attempt)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        timeout: float = 30,
    ) -> dict:
        """Call the MCP tool *name* and return the JSON-RPC body."""
        return await self.call('tools/call', {'name': name, 'arguments': arguments}, timeout=timeout)

    def _message(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        self._request_id += 1
        message: dict[str, Any] = {'jsonrpc': '2.0', 'id': self._request_id, 'method': method}
        if params is not None:
            message['params'] = params
        return message

    async def _with_retries(self, label: str, attempt: Callable[[], Awaitable[_T]]) -> _T:
        attempts = len(self._retry_backoffs) + 1
        made = 0
        while True:
            made += 1
            try:
                return await attempt()
            except _Transient as failure:
                self._drop_session(label, made, attempts, failure)
                if made == attempts:
                    raise failure.exhausted(label, attempts) from failure.cause
            await asyncio.sleep(self._retry_backoffs[made - 1])

    async def _ensure_handshake(self) -> None:
        if self._initialized:
            return
        initialize = self._message('initialize', _initialize_params(self._client_name))
        result = await self._exchange(initialize, 30)
        logger.debug('MCP initialize response: %s', json.dumps(result)[:200])
        await self._notify('notifications/initialized')
        self._initialized = True
        logger.info('MCP session initialized (session_id=%s)', self._session_id)

    async def _exchange(self, message: dict[str, Any], timeout: float) -> Any:
        """POST *message* once and decode the answer, capturing a minted session id."""
        method = message['method']
        resp = await self._post(method, message, timeout, transient_statuses=_RETRYABLE_STATUS)
        result = _decoded(method, resp)
        session_id = resp.headers.get('mcp-session-id')
        if session_id:
            self._session_id = session_id
        return result

    async def _notify(self, method: str) -> None:
        label = f'notify {method}'
        message = {'jsonrpc': '2.0', 'method': method}
        resp = await self._post(label, message, 10, transient_statuses=frozenset())
        if resp.status_code not in (200, 202, 204):
            logger.warning(
                'Notification %s returned %d: %s', method, resp.status_code, resp.text[:200],
            )

    async def _post(
        self,
        label: str,
        message: dict[str, Any],
        timeout: float,
        *,
        transient_statuses: frozenset[int],
    ) -> httpx.Response:
        """POST *message* once; a retryable failure raises :class:`_Transient`."""
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.post(
                    self.mcp_endpoint,
                    json=message,
                    headers=_request_headers(self._session_id),
                    timeout=timeout,
                )
        except _RETRYABLE_EXCEPTIONS as exc:
            raise _Transient(exc) from exc
        except httpx.HTTPError as exc:
            raise McpCallError(
                McpFailureKind.TRANSPORT, f'MCP {label} failed: {type(exc).__name__}: {exc}',
            ) from exc
        if resp.status_code in transient_statuses:
            cause = httpx.HTTPStatusError(
                f'HTTP {resp.status_code}', request=resp.request, response=resp,
            )
            raise _Transient(cause, status=resp.status_code)
        return resp

    def _drop_session(self, label: str, made: int, attempts: int, failure: _Transient) -> None:
        logger.warning(
            'MCP %s transient failure (attempt %d/%d): %s', label, made, attempts, failure,
        )
        self._session_id = None
        self._initialized = False


def _rpc_result(method: str, resp: httpx.Response) -> Any:
    body = _decoded(method, resp)
    if isinstance(body, dict) and body.get('error') is not None:
        raise McpCallError(
            McpFailureKind.JSON_RPC_ERROR, f'MCP {method} returned a JSON-RPC error: {body["error"]}',
        )
    return body


# Every other shape parse_tool_result refuses is a REPLY_SHAPE failure.
_REFUSAL_KINDS: Mapping[EnvelopeShape, McpFailureKind] = MappingProxyType({
    EnvelopeShape.JSON_RPC_ERROR: McpFailureKind.JSON_RPC_ERROR,
    EnvelopeShape.IS_ERROR: McpFailureKind.TOOL_ERROR,
})


def _tool_reply(name: str, body: Any) -> dict:
    reply, err = parse_tool_result(body, None, dict)
    if err is not None:
        kind = _REFUSAL_KINDS.get(err.shape, McpFailureKind.REPLY_SHAPE)
        raise McpCallError(
            kind, f'MCP tool {name} reply refused ({err.shape}): {err.payload_prefix}',
        ) from err
    if not isinstance(reply, dict):
        raise McpCallError(
            McpFailureKind.REPLY_SHAPE, f'MCP tool {name} reply is not a dict: {reply!r:.200}',
        )
    return reply


class SyncMcpSession:
    """An MCP session for a script: entering handshakes, exiting releases.

    The handshake captures a server-minted ``mcp-session-id`` and echoes it on
    every later request; the client never mints one, so a stateless server
    sees none. Exit DELETEs a captured id, best-effort: a failed release must
    never turn a delivered call into a failure. No transport retry, and so no
    ``client_op_id``. ``call_tool`` returns the tool's reply dict and raises
    :class:`McpCallError` for every failure, ``result.isError`` included.
    """

    def __init__(self, base_url: str, *, client_name: str, timeout: float = 30.0) -> None:
        self._endpoint = mcp_endpoint_url(base_url)
        self._client_name = client_name
        self._timeout = timeout
        self._client: httpx.Client | None = None
        self._session_id: str | None = None
        self._request_id = 0

    def __enter__(self) -> SyncMcpSession:
        self._client = httpx.Client(follow_redirects=True, timeout=self._timeout)
        try:
            self._handshake()
        except BaseException:
            self._release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._release()

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict:
        """Call the MCP tool *name* and return its reply dict."""
        payload = self._request('tools/call', {'name': name, 'arguments': dict(arguments)})
        return _tool_reply(name, _decoded('tools/call', self._post('tools/call', payload)))

    def _handshake(self) -> None:
        payload = self._request('initialize', _initialize_params(self._client_name))
        resp = self._post('initialize', payload)
        self._session_id = resp.headers.get('mcp-session-id')
        _rpc_result('initialize', resp)
        notified = self._post(
            'notify notifications/initialized',
            {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        )
        if notified.status_code not in (200, 202, 204):
            raise McpCallError(
                McpFailureKind.HTTP_STATUS,
                f'MCP notify notifications/initialized answered HTTP {notified.status_code}',
                status=notified.status_code,
            )

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        return {'jsonrpc': '2.0', 'id': self._request_id, 'method': method, 'params': params}

    def _post(self, label: str, payload: dict[str, Any]) -> httpx.Response:
        if self._client is None:
            raise RuntimeError('SyncMcpSession is used only inside its with-block')
        try:
            return self._client.post(
                self._endpoint, json=payload, headers=_request_headers(self._session_id),
            )
        except httpx.HTTPError as exc:
            raise McpCallError(
                McpFailureKind.TRANSPORT, f'MCP {label} failed: {type(exc).__name__}: {exc}',
            ) from exc

    def _release(self) -> None:
        client, self._client = self._client, None
        session_id, self._session_id = self._session_id, None
        if client is None:
            return
        try:
            if session_id is not None:
                client.delete(self._endpoint, headers=_request_headers(session_id))
        except Exception as exc:  # noqa: BLE001 — a release must never mask the call's outcome
            logger.debug('MCP session %s release to %s failed: %s', session_id, self._endpoint, exc)
        finally:
            client.close()
