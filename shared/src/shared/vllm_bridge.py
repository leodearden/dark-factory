"""vLLM → Anthropic protocol bridge.

Provides:
- Pure translation functions (testable without a server):
  - ``_normalize_tool_use_block`` — normalises a single tool_use block
  - ``_translate_messages_response`` — normalises a full /v1/messages response body
- ``VllmBridge`` — async-context-manager aiohttp proxy that starts a local
  HTTP server, translates POST /v1/messages responses, and passes all other
  traffic straight through to the configured upstream URL.
"""

from __future__ import annotations

import json
import uuid

from aiohttp import web
from aiohttp.client import ClientSession

# ── pure translation helpers ─────────────────────────────────────────────────


def _normalize_tool_use_block(block: dict) -> dict:
    """Return a normalised copy of a tool_use content block.

    Handles the following malformations produced by vLLM:
    - ``input`` serialised as a JSON string instead of a dict → json.loads'd
    - ``id`` missing → generated as ``'toolu_' + uuid4().hex[:24]``
    - ``id`` present but not prefixed with ``'toolu_'`` → wrapped as
      ``'toolu_' + existing_id``

    The function is idempotent: a well-formed Anthropic block is returned
    unchanged (value-equal).  The input dict is never mutated; a new dict
    is always returned.
    """
    result = dict(block)

    # ── normalise `input` ────────────────────────────────────────────────────
    raw_input = result.get('input')
    if isinstance(raw_input, str):
        try:
            result['input'] = json.loads(raw_input)
        except json.JSONDecodeError:
            result['input'] = {'_raw': raw_input}

    # ── normalise `id` ──────────────────────────────────────────────────────
    existing_id = result.get('id')
    if not existing_id:
        result['id'] = 'toolu_' + uuid.uuid4().hex[:24]
    elif not str(existing_id).startswith('toolu_'):
        result['id'] = 'toolu_' + existing_id

    return result


def _translate_messages_response(body: dict) -> dict:
    """Return a normalised copy of a /v1/messages response body.

    Handles the following vLLM malformations:
    - OpenAI-style top-level ``tool_calls`` list → Anthropic content[] blocks
    - ``stop_reason='tool_calls'`` → ``'tool_use'`` when content has tool_use blocks
    - tool_use blocks with JSON-string ``input`` or non-Anthropic ``id``

    The function is idempotent: a well-formed Anthropic response body is
    returned unchanged.  Error bodies and non-assistant responses are passed
    through unchanged.  The input dict is never mutated; a new dict is returned.
    """
    # ── pass through non-assistant / error bodies ────────────────────────────
    if body.get('type') == 'error' or body.get('role') != 'assistant':
        return dict(body)

    result = dict(body)

    # ── convert OpenAI-style top-level tool_calls ────────────────────────────
    if 'tool_calls' in result:
        raw_content = result.get('content', '')
        content_list: list[dict] = []

        # Wrap any existing string content into a text block
        if isinstance(raw_content, str) and raw_content:
            content_list.append({'type': 'text', 'text': raw_content})
        elif isinstance(raw_content, list):
            content_list.extend(raw_content)

        # Convert each OpenAI tool_call to an Anthropic tool_use block
        for tc in result['tool_calls']:
            fn = tc.get('function', {})
            block = {
                'type': 'tool_use',
                'id': tc.get('id', ''),
                'name': fn.get('name', ''),
                'input': fn.get('arguments', '{}'),
            }
            content_list.append(_normalize_tool_use_block(block))

        result['content'] = content_list
        del result['tool_calls']

    # ── normalise any tool_use blocks already in content[] ───────────────────
    if isinstance(result.get('content'), list):
        normalised: list[dict] = []
        for block in result['content']:
            if block.get('type') == 'tool_use':
                normalised.append(_normalize_tool_use_block(block))
            else:
                normalised.append(block)
        result['content'] = normalised

    # ── fix stop_reason ──────────────────────────────────────────────────────
    has_tool_use = isinstance(result.get('content'), list) and any(
        b.get('type') == 'tool_use' for b in result['content']
    )
    if has_tool_use:
        result['stop_reason'] = 'tool_use'

    return result


# ── VllmBridge server ────────────────────────────────────────────────────────

# Hop-by-hop headers that must not be forwarded
_HOP_BY_HOP = frozenset({
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length',
})


class VllmBridge:
    """Local aiohttp HTTP proxy that translates vLLM /v1/messages responses.

    Usage::

        async with VllmBridge(upstream_url='http://vllm-host:8000') as bridge:
            # bridge.url is e.g. 'http://127.0.0.1:54321'
            env['ANTHROPIC_BASE_URL'] = bridge.url
            ...  # run subprocess

    The bridge starts an aiohttp server on a random local port.  POST
    /v1/messages requests are forwarded to the upstream, the response is
    translated via ``_translate_messages_response``, and the corrected body
    is returned to the caller.  All other requests are piped through verbatim.
    """

    def __init__(self, upstream_url: str) -> None:
        self.upstream_url = upstream_url.rstrip('/')
        self.url: str = ''
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start the bridge server and bind to a random local port."""
        app = web.Application()
        # Specific route must be registered before catch-all
        app.router.add_post('/v1/messages', self._handle_messages)
        app.router.add_route('*', '/{tail:.*}', self._proxy_catchall)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, '127.0.0.1', 0)
        await site.start()

        # Retrieve the OS-assigned port
        sockets = site._server.sockets  # type: ignore[union-attr]
        port = sockets[0].getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'

    async def stop(self) -> None:
        """Stop the bridge server.  Idempotent — safe to call multiple times."""
        if self._runner is not None:
            runner, self._runner = self._runner, None
            await runner.cleanup()

    async def __aenter__(self) -> VllmBridge:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    async def _handle_messages(self, request: web.Request) -> web.Response:
        """Forward POST /v1/messages, translate the response body."""
        body = await request.json()
        # Force non-streaming so we can buffer and translate the full response
        body['stream'] = False

        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        async with ClientSession() as session, session.post(
            self.upstream_url + '/v1/messages',
            json=body,
            headers=headers,
        ) as upstream:
            upstream_body = await upstream.json(content_type=None)
            status = upstream.status

        translated = _translate_messages_response(upstream_body)
        return web.json_response(translated, status=status)

    async def _proxy_catchall(self, request: web.Request) -> web.StreamResponse:
        """Pipe all other requests straight through to the upstream."""
        path = request.path
        query = request.query_string
        upstream_url = self.upstream_url + path
        if query:
            upstream_url += '?' + query

        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        data = await request.read()

        async with ClientSession() as session, session.request(
            method=request.method,
            url=upstream_url,
            headers=headers,
            data=data,
        ) as upstream:
            upstream_body = await upstream.read()
            return web.Response(
                status=upstream.status,
                body=upstream_body,
                content_type=upstream.content_type,
            )
