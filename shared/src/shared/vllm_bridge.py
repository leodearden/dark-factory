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
import logging
import uuid

from aiohttp import ClientTimeout, web
from aiohttp.client import ClientSession

logger = logging.getLogger(__name__)

# ── pure translation helpers ─────────────────────────────────────────────────


_PAD_TOKEN_RATIO_THRESHOLD = 0.5


def _is_pad_token_response(body: dict) -> bool:
    """Return True if the response content is predominantly NULL pad tokens.

    Inspects all text content in the response body.  If there is at least some
    text and the ratio of ``\\x00`` characters to total characters exceeds
    ``_PAD_TOKEN_RATIO_THRESHOLD`` (50%), the response is considered a pad-token
    response.

    Responses that contain *any* ``tool_use`` blocks are never flagged — a
    tool-call response with garbled trailing text is still actionable.
    """
    content = body.get('content')
    if content is None:
        return False

    # Collect all text from the response.
    texts: list[str] = []

    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        # If the response contains tool_use blocks, don't flag it.
        if any(b.get('type') == 'tool_use' for b in content):
            return False
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text', '')
                if isinstance(text, str):
                    texts.append(text)

    combined = ''.join(texts)
    if not combined:
        return False

    null_count = combined.count('\x00')
    return (null_count / len(combined)) > _PAD_TOKEN_RATIO_THRESHOLD


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

    # ── detect pad-token (NULL byte) responses ──────────────────────────────
    if _is_pad_token_response(result):
        logger.warning(
            'vLLM returned pad tokens (\\x00) instead of content — '
            'converting to error response'
        )
        return {
            'type': 'error',
            'error': {
                'type': 'invalid_response',
                'message': (
                    'vLLM returned pad tokens (\\u0000) instead of content '
                    '— likely a model/quantization issue'
                ),
            },
        }

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

    def __init__(
        self,
        upstream_url: str,
        *,
        max_output_tokens: int | None = None,
    ) -> None:
        self.upstream_url = upstream_url.rstrip('/')
        self.url: str = ''
        self._runner: web.AppRunner | None = None
        self._session: ClientSession | None = None
        self._max_output_tokens = max_output_tokens
        self._max_model_len: int | None = None

    async def start(self) -> None:
        """Start the bridge server and bind to a random local port."""
        # vLLM inference on large prompts (~39k tokens) can take 5-10+ min on
        # single-GPU configs.  aiohttp's default ClientTimeout(total=300) kills
        # requests at 5 min.  Use 30 min — the subprocess timeout is the real
        # outer bound.
        self._session = ClientSession(timeout=ClientTimeout(total=1800))

        # Discover max_model_len from the upstream /v1/models endpoint so we
        # can clamp max_tokens on incoming requests to avoid context-length
        # overflow 500s.  Claude CLI defaults to max_tokens=32000 which easily
        # exceeds smaller models (e.g. 63k context with a 31k prompt).
        await self._discover_max_model_len()

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
        if self._session is not None:
            session, self._session = self._session, None
            await session.close()

    async def _discover_max_model_len(self) -> None:
        """Query /v1/models to learn the model's max_model_len.

        Used to clamp ``max_tokens`` on incoming requests so that the total
        (prompt + output) doesn't exceed the model's context window.  Without
        this, Claude CLI's default ``max_tokens=32000`` causes 500s on models
        with smaller context windows (e.g. 63k with a 31k eval prompt).

        Failures are non-fatal — the bridge falls back to the explicit
        ``max_output_tokens`` constructor arg or no clamping at all.
        """
        assert self._session is not None
        try:
            async with self._session.get(
                self.upstream_url + '/v1/models',
                timeout=ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get('data', [])
                    if models:
                        mml = models[0].get('max_model_len')
                        if isinstance(mml, int) and mml > 0:
                            self._max_model_len = mml
                            logger.info(
                                'Bridge discovered max_model_len=%d from upstream',
                                mml,
                            )
        except Exception as e:
            logger.warning('Bridge failed to discover max_model_len: %s', e)

    # Default output token cap as a fraction of max_model_len.  Claude CLI
    # defaults to max_tokens=32000 which assumes a 200k+ context.  For
    # smaller models (63k context, 131k context), we need to leave most of
    # the context for the prompt.  Eval prompts are ~31-39k tokens, so
    # reserving 75% for input and 25% for output is a safe starting point.
    # 8k output tokens is still generous for tool calls + code generation.
    _OUTPUT_FRACTION = 0.25
    _MIN_OUTPUT_CAP = 4096    # Never clamp below 4k
    _DEFAULT_OUTPUT_CAP = 8000  # Fallback when max_model_len is unknown

    def _clamp_max_tokens(self, body: dict) -> None:
        """Clamp ``max_tokens`` to avoid context-length overflow.

        Strategy: if ``max_output_tokens`` was provided at init, use it
        directly. Otherwise, if we discovered ``max_model_len`` from the
        upstream, cap ``max_tokens`` to 25% of the context (reserving 75%
        for the prompt). This is conservative but avoids the 500 errors
        that occur when Claude CLI's default 32k + a 31k eval prompt
        exceeds the model's context window.
        """
        cap = self._max_output_tokens
        if cap is None and self._max_model_len is not None:
            cap = max(
                self._MIN_OUTPUT_CAP,
                int(self._max_model_len * self._OUTPUT_FRACTION),
            )

        if cap is not None:
            current = body.get('max_tokens')
            if isinstance(current, int) and current > cap:
                logger.info(
                    'Clamping max_tokens from %d to %d (max_model_len=%s)',
                    current, cap, self._max_model_len,
                )
                body['max_tokens'] = cap

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
        # Clamp max_tokens to avoid context-length overflow
        self._clamp_max_tokens(body)

        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        assert self._session is not None, 'bridge not started'
        async with self._session.post(
            self.upstream_url + '/v1/messages',
            json=body,
            headers=headers,
        ) as upstream:
            # Read raw bytes first so we can forward them verbatim on parse failure
            raw = await upstream.read()
            status = upstream.status
            upstream_content_type = upstream.content_type

        # Attempt JSON parsing; fall back to forwarding raw bytes if it fails.
        # This preserves the upstream status code and error body when the upstream
        # returns a non-JSON response (HTML error page, plain-text 503, truncated
        # body), rather than swallowing failures as a generic 500.
        try:
            upstream_body = json.loads(raw.decode('utf-8', errors='replace'))
        except (json.JSONDecodeError, ValueError):
            return web.Response(status=status, body=raw, content_type=upstream_content_type)

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

        assert self._session is not None, 'bridge not started'
        async with self._session.request(
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
