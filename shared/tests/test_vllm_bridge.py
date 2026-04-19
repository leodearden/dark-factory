"""Tests for vllm_bridge: protocol translation functions and VllmBridge server."""

from __future__ import annotations

import pytest

# Skip entire module when aiohttp is not installed (optional [vllm] extra)
pytest.importorskip('aiohttp')

import aiohttp  # noqa: E402
from aiohttp import web  # noqa: E402

from shared.vllm_bridge import (
    VllmBridge,
    _is_pad_token_response,
    _normalize_tool_use_block,
    _translate_messages_response,
)


def _site_port(site: web.TCPSite) -> int:
    """Return the bound port of a TCPSite whose `_server.sockets` holds a listener.

    aiohttp's ``TCPSite._server`` is typed ``AbstractServer | None`` (and
    ``AbstractServer`` does not formally expose ``sockets``), so we narrow with
    an assert and read the concrete ``asyncio.base_events.Server`` attribute.
    """
    server = site._server
    assert server is not None, 'site._server is None — site.start() not awaited?'
    sockets = getattr(server, 'sockets', None)
    assert sockets, 'site._server has no bound sockets'
    return sockets[0].getsockname()[1]


class TestTranslateMessagesResponseConvertsOpenAIToolCalls:

    def test_converts_openai_tool_calls_to_anthropic_content_list(self):
        """OpenAI-style top-level tool_calls are converted to Anthropic content blocks."""
        body = {
            'role': 'assistant',
            'content': 'I will look.',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'Read', 'arguments': '{"path": "/x"}'},
                }
            ],
            'stop_reason': 'tool_calls',
        }
        result = _translate_messages_response(body)
        # content is a list
        assert isinstance(result['content'], list)
        # contains a text block
        text_blocks = [b for b in result['content'] if b.get('type') == 'text']
        assert len(text_blocks) == 1
        assert text_blocks[0]['text'] == 'I will look.'
        # contains a tool_use block with normalised fields
        tool_blocks = [b for b in result['content'] if b.get('type') == 'tool_use']
        assert len(tool_blocks) == 1
        tb = tool_blocks[0]
        assert tb['name'] == 'Read'
        assert tb['id'].startswith('toolu_')
        assert isinstance(tb['input'], dict)
        assert tb['input'] == {'path': '/x'}
        # top-level tool_calls is gone
        assert 'tool_calls' not in result


class TestTranslateMessagesResponsePassthrough:

    def test_passthrough_for_error_body(self):
        """Error bodies (type='error') are returned unchanged."""
        body = {
            'type': 'error',
            'error': {'type': 'rate_limit_error', 'message': 'Rate limit exceeded'},
        }
        result = _translate_messages_response(body)
        assert result == body

    def test_passthrough_for_body_missing_role(self):
        """Bodies missing a 'role' key are returned unchanged."""
        body = {'some_key': 'some_value'}
        result = _translate_messages_response(body)
        assert result == body

    def test_no_key_error_on_passthrough(self):
        """No KeyError is raised for error or incomplete bodies."""
        bodies = [
            {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'bad'}},
            {},
            {'role': 'user'},
        ]
        for body in bodies:
            result = _translate_messages_response(body)
            assert isinstance(result, dict)


class TestTranslateMessagesResponseIdempotent:

    def test_idempotent_for_native_anthropic_response(self):
        """A well-formed Anthropic response body is returned value-equal (no-op)."""
        body = {
            'type': 'message',
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'Let me read that file.'},
                {'type': 'tool_use', 'id': 'toolu_x', 'name': 'Read', 'input': {'path': '/x'}},
            ],
            'stop_reason': 'tool_use',
            'model': 'claude-sonnet',
        }
        result = _translate_messages_response(body)
        assert result == body


class TestTranslateMessagesResponseStopReason:

    def test_rewrites_stop_reason_tool_calls_when_tool_use_in_content(self):
        """stop_reason='tool_calls' is rewritten to 'tool_use' when content has tool_use blocks."""
        body = {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'here'},
                {'type': 'tool_use', 'id': 'toolu_x', 'name': 'Read', 'input': {}},
            ],
            'stop_reason': 'tool_calls',
        }
        result = _translate_messages_response(body)
        assert result['stop_reason'] == 'tool_use'

    def test_rewrites_stop_reason_end_turn_when_tool_use_in_content(self):
        """stop_reason='end_turn' is rewritten to 'tool_use' when content has tool_use blocks."""
        body = {
            'role': 'assistant',
            'content': [
                {'type': 'tool_use', 'id': 'toolu_y', 'name': 'Write', 'input': {}},
            ],
            'stop_reason': 'end_turn',
        }
        result = _translate_messages_response(body)
        assert result['stop_reason'] == 'tool_use'

    def test_passes_through_stop_reason_when_no_tool_use_in_content(self):
        """stop_reason passes through unchanged when content has no tool_use blocks."""
        body = {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'done'}],
            'stop_reason': 'end_turn',
        }
        result = _translate_messages_response(body)
        assert result['stop_reason'] == 'end_turn'


class TestNormalizeToolUseBlockIdempotent:

    def test_idempotent_for_correct_block(self):
        """A well-formed Anthropic tool_use block is returned value-equal (no-op)."""
        block = {
            'type': 'tool_use',
            'id': 'toolu_abc',
            'name': 'Read',
            'input': {'path': '/tmp/x'},
        }
        result = _normalize_tool_use_block(block)
        assert result == block


class TestNormalizeToolUseBlockIdHandling:

    def test_generates_toolu_id_when_missing(self):
        """When block has no `id`, result gets an id starting with 'toolu_' of length > 7."""
        block = {'type': 'tool_use', 'name': 'get_weather', 'input': {}}
        result = _normalize_tool_use_block(block)
        assert result['id'].startswith('toolu_')
        assert len(result['id']) > 7

    def test_rewrites_openai_call_id(self):
        """When `id` is 'call_abc123', result id starts with 'toolu_'."""
        block = {'type': 'tool_use', 'id': 'call_abc123', 'name': 'Read', 'input': {}}
        result = _normalize_tool_use_block(block)
        assert result['id'].startswith('toolu_')


class TestNormalizeToolUseBlockJsonStringInput:

    def test_parses_jsonstring_input(self):
        """When `input` is a JSON string, it is parsed into a dict."""
        block = {
            'type': 'tool_use',
            'id': 'toolu_abc',
            'name': 'get_weather',
            'input': '{"city": "SF"}',
        }
        result = _normalize_tool_use_block(block)
        assert result['input'] == {'city': 'SF'}

    def test_type_preserved(self):
        """The `type` field stays 'tool_use' after normalization."""
        block = {
            'type': 'tool_use',
            'id': 'toolu_abc',
            'name': 'get_weather',
            'input': '{"city": "SF"}',
        }
        result = _normalize_tool_use_block(block)
        assert result['type'] == 'tool_use'

    def test_name_preserved(self):
        """The `name` field is preserved after normalization."""
        block = {
            'type': 'tool_use',
            'id': 'toolu_abc',
            'name': 'get_weather',
            'input': '{"city": "SF"}',
        }
        result = _normalize_tool_use_block(block)
        assert result['name'] == 'get_weather'


# ── Pad-token (NULL byte) detection ─────────────────────────────────────────


class TestIsPadTokenResponse:

    def test_normal_text_response_returns_false(self):
        """A response with normal text content is not a pad-token response."""
        body = {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'Hello, world!'}],
        }
        assert _is_pad_token_response(body) is False

    def test_all_null_text_response_returns_true(self):
        """A response with all-NULL text content is a pad-token response."""
        body = {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': '\x00' * 1000}],
        }
        assert _is_pad_token_response(body) is True

    def test_mixed_response_below_threshold_returns_false(self):
        """A response with mostly real text and some NULLs is not flagged."""
        # 10 NULLs out of 110 chars = ~9% — well below 50%
        body = {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'A' * 100 + '\x00' * 10}],
        }
        assert _is_pad_token_response(body) is False

    def test_mixed_response_above_threshold_returns_true(self):
        """A response with >50% NULLs is flagged."""
        # 80 NULLs out of 100 chars = 80%
        body = {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'A' * 20 + '\x00' * 80}],
        }
        assert _is_pad_token_response(body) is True

    def test_response_with_tool_use_blocks_returns_false(self):
        """A response with tool_use blocks is never flagged, even with NULL text."""
        body = {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': '\x00' * 1000},
                {'type': 'tool_use', 'id': 'toolu_x', 'name': 'Read', 'input': {}},
            ],
        }
        assert _is_pad_token_response(body) is False

    def test_empty_content_list_returns_false(self):
        """An empty content list is not a pad-token response."""
        body = {'role': 'assistant', 'content': []}
        assert _is_pad_token_response(body) is False

    def test_no_content_key_returns_false(self):
        """A body with no content key is not a pad-token response."""
        body = {'role': 'assistant'}
        assert _is_pad_token_response(body) is False

    def test_string_content_all_nulls_returns_true(self):
        """A response with string content (not list) that is all NULLs is flagged."""
        body = {
            'role': 'assistant',
            'content': '\x00' * 500,
        }
        assert _is_pad_token_response(body) is True

    def test_string_content_normal_returns_false(self):
        """A response with normal string content is not flagged."""
        body = {
            'role': 'assistant',
            'content': 'This is a normal response.',
        }
        assert _is_pad_token_response(body) is False

    def test_multiple_text_blocks_aggregated(self):
        """NULL ratio is computed across all text blocks combined."""
        # Block 1: 50 NULLs, Block 2: 50 real chars → 50% exactly, not > 50%
        body = {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': '\x00' * 50},
                {'type': 'text', 'text': 'A' * 50},
            ],
        }
        assert _is_pad_token_response(body) is False

    def test_multiple_text_blocks_over_threshold(self):
        """Multiple text blocks with >50% NULLs overall are flagged."""
        # Block 1: 80 NULLs, Block 2: 20 real chars → 80%
        body = {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': '\x00' * 80},
                {'type': 'text', 'text': 'A' * 20},
            ],
        }
        assert _is_pad_token_response(body) is True


class TestTranslateMessagesResponsePadTokenDetection:

    def test_pad_token_response_converted_to_error(self):
        """A pad-token response is transformed into an error response."""
        body = {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': '\x00' * 1000}],
            'stop_reason': 'end_turn',
        }
        result = _translate_messages_response(body)
        assert result['type'] == 'error'
        assert result['error']['type'] == 'invalid_response'
        assert 'pad tokens' in result['error']['message']

    def test_pad_token_response_with_string_content(self):
        """A pad-token response with string content is transformed to error."""
        body = {
            'role': 'assistant',
            'content': '\x00' * 500,
            'stop_reason': 'end_turn',
        }
        result = _translate_messages_response(body)
        assert result['type'] == 'error'
        assert result['error']['type'] == 'invalid_response'

    def test_normal_response_not_affected(self):
        """Normal text response passes through the pad-token check unharmed."""
        body = {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'Let me help with that.'}],
            'stop_reason': 'end_turn',
        }
        result = _translate_messages_response(body)
        assert result['role'] == 'assistant'
        assert result['content'] == [{'type': 'text', 'text': 'Let me help with that.'}]
        assert 'type' not in result or result.get('type') != 'error'

    def test_tool_use_with_null_text_not_converted_to_error(self):
        """A response with tool_use blocks is not converted to error even with NULL text."""
        body = {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': '\x00' * 100},
                {'type': 'tool_use', 'id': 'toolu_abc', 'name': 'Read', 'input': {'path': '/x'}},
            ],
            'stop_reason': 'tool_use',
        }
        result = _translate_messages_response(body)
        assert result.get('type') != 'error'
        assert result['role'] == 'assistant'
        assert result['stop_reason'] == 'tool_use'


# ── VllmBridge server lifecycle ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestVllmBridgeLifecycle:

    async def test_start_binds_local_port(self):
        """VllmBridge.start() binds to 127.0.0.1 on an OS-assigned port != the upstream port."""
        bridge = VllmBridge(upstream_url='http://127.0.0.1:1')
        await bridge.start()
        try:
            assert bridge.url.startswith('http://127.0.0.1:')
            port_str = bridge.url.split(':')[-1]
            port = int(port_str)
            assert port > 0
            assert port != 1
        finally:
            await bridge.stop()

    async def test_context_manager_lifecycle(self):
        """async with VllmBridge sets bridge.url inside block; stop() is idempotent outside."""
        async with VllmBridge(upstream_url='http://127.0.0.1:1') as bridge:
            assert bridge.url.startswith('http://127.0.0.1:')
        # stop() after context exit should be a no-op (no raise)
        await bridge.stop()


# ── Integration tests (mock upstream server) ─────────────────────────────────


@pytest.fixture
async def mock_upstream_models():
    """Start a mock upstream server with /v1/models returning a fixed response."""
    app = web.Application()

    async def models_handler(request):
        return web.json_response({'data': [{'id': 'test'}]})

    app.router.add_route('*', '/v1/models', models_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = _site_port(site)
    url = f'http://127.0.0.1:{port}'
    yield url
    await runner.cleanup()


@pytest.fixture
async def mock_upstream_messages():
    """Start a mock upstream server for /v1/messages with vLLM-style tool_calls response."""
    app = web.Application()
    received_bodies = []

    async def messages_handler(request):
        body = await request.json()
        received_bodies.append(body)
        response_body = {
            'role': 'assistant',
            'content': 'Using tool.',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'Read', 'arguments': '{"path": "/tmp/x"}'},
                }
            ],
            'stop_reason': 'tool_calls',
        }
        return web.json_response(response_body)

    app.router.add_post('/v1/messages', messages_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = _site_port(site)
    url = f'http://127.0.0.1:{port}'
    yield url, received_bodies
    await runner.cleanup()


@pytest.fixture
async def mock_upstream_stream_capture():
    """Start a mock upstream that captures the stream field in the request body."""
    app = web.Application()
    received_bodies = []

    async def messages_handler(request):
        body = await request.json()
        received_bodies.append(body)
        return web.json_response({'role': 'assistant', 'content': [], 'stop_reason': 'end_turn'})

    app.router.add_post('/v1/messages', messages_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = _site_port(site)
    url = f'http://127.0.0.1:{port}'
    yield url, received_bodies
    await runner.cleanup()


@pytest.fixture
async def mock_upstream_error():
    """Start a mock upstream that returns status 500 with an error body."""
    app = web.Application()

    async def messages_handler(request):
        error_body = {
            'type': 'error',
            'error': {'type': 'internal_error', 'message': 'boom'},
        }
        return web.json_response(error_body, status=500)

    app.router.add_post('/v1/messages', messages_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = _site_port(site)
    url = f'http://127.0.0.1:{port}'
    yield url
    await runner.cleanup()


@pytest.fixture
async def mock_upstream_text_error():
    """Start a mock upstream that returns a plain-text 503 (non-JSON) error."""
    app = web.Application()

    async def messages_handler(request):
        return web.Response(
            status=503,
            text='Service Unavailable: backend down',
            content_type='text/plain',
        )

    app.router.add_post('/v1/messages', messages_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = _site_port(site)
    url = f'http://127.0.0.1:{port}'
    yield url
    await runner.cleanup()


@pytest.fixture
async def mock_upstream_truncated_json():
    """Start a mock upstream that returns a truncated JSON body (unparseable)."""
    app = web.Application()

    async def messages_handler(request):
        return web.Response(
            status=200,
            body=b'{"role": "assistant", "content":',
            content_type='application/json',
        )

    app.router.add_post('/v1/messages', messages_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = _site_port(site)
    url = f'http://127.0.0.1:{port}'
    yield url
    await runner.cleanup()


@pytest.mark.asyncio
class TestVllmBridgeIntegration:

    async def test_passthrough_non_messages_path(self, mock_upstream_models):
        """Non-/v1/messages paths are proxied verbatim."""
        async with VllmBridge(upstream_url=mock_upstream_models) as bridge, aiohttp.ClientSession() as session, session.get(bridge.url + '/v1/models') as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data == {'data': [{'id': 'test'}]}

    async def test_messages_translates_tool_calls(self, mock_upstream_messages):
        """POST /v1/messages translates vLLM tool_calls to Anthropic content blocks."""
        upstream_url, _ = mock_upstream_messages
        async with (
            VllmBridge(upstream_url=upstream_url) as bridge,
            aiohttp.ClientSession() as session,
            session.post(
                bridge.url + '/v1/messages',
                json={'model': 'x', 'messages': []},
            ) as resp,
        ):
            assert resp.status == 200
            data = await resp.json()

        # tool_calls key is gone
        assert 'tool_calls' not in data
        # content is a list
        assert isinstance(data['content'], list)
        # has a tool_use block with toolu_-prefixed id and dict input
        tool_blocks = [b for b in data['content'] if b.get('type') == 'tool_use']
        assert len(tool_blocks) == 1
        tb = tool_blocks[0]
        assert tb['id'].startswith('toolu_')
        assert isinstance(tb['input'], dict)
        # stop_reason is 'tool_use'
        assert data['stop_reason'] == 'tool_use'

    async def test_forces_stream_false(self, mock_upstream_stream_capture):
        """Bridge sets stream=False on forwarded request body."""
        upstream_url, received = mock_upstream_stream_capture
        async with (
            VllmBridge(upstream_url=upstream_url) as bridge,
            aiohttp.ClientSession() as session,
            session.post(
                bridge.url + '/v1/messages',
                json={'model': 'x', 'stream': True, 'messages': []},
            ) as _resp,
        ):
            pass

        assert len(received) == 1
        assert received[0].get('stream') is False

    async def test_forwards_upstream_error_status(self, mock_upstream_error):
        """Bridge forwards upstream error status and passes error body through unchanged."""
        async with (
            VllmBridge(upstream_url=mock_upstream_error) as bridge,
            aiohttp.ClientSession() as session,
            session.post(
                bridge.url + '/v1/messages',
                json={'model': 'x', 'messages': []},
            ) as resp,
        ):
            assert resp.status == 500
            data = await resp.json()
            assert data['type'] == 'error'
            assert data['error']['message'] == 'boom'

    async def test_handles_non_json_upstream_response(self, mock_upstream_text_error):
        """Bridge forwards non-JSON upstream response verbatim, preserving status."""
        async with (
            VllmBridge(upstream_url=mock_upstream_text_error) as bridge,
            aiohttp.ClientSession() as session,
            session.post(
                bridge.url + '/v1/messages',
                json={'model': 'x', 'messages': []},
            ) as resp,
        ):
            assert resp.status == 503
            body = await resp.read()
            assert body.startswith(b'Service Unavailable')

    async def test_handles_truncated_json_upstream_response(self, mock_upstream_truncated_json):
        """Bridge forwards truncated/malformed JSON body verbatim, preserving status."""
        async with (
            VllmBridge(upstream_url=mock_upstream_truncated_json) as bridge,
            aiohttp.ClientSession() as session,
            session.post(
                bridge.url + '/v1/messages',
                json={'model': 'x', 'messages': []},
            ) as resp,
        ):
            assert resp.status == 200
            body = await resp.read()
            assert body == b'{"role": "assistant", "content":'
