"""Tests for vllm_bridge: protocol translation functions and VllmBridge server."""

from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web

from shared.vllm_bridge import VllmBridge, _normalize_tool_use_block, _translate_messages_response


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
    port = site._server.sockets[0].getsockname()[1]
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
    port = site._server.sockets[0].getsockname()[1]
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
    port = site._server.sockets[0].getsockname()[1]
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
    port = site._server.sockets[0].getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    yield url
    await runner.cleanup()


@pytest.mark.asyncio
class TestVllmBridgeIntegration:

    async def test_passthrough_non_messages_path(self, mock_upstream_models):
        """Non-/v1/messages paths are proxied verbatim."""
        async with VllmBridge(upstream_url=mock_upstream_models) as bridge:
            async with aiohttp.ClientSession() as session:
                async with session.get(bridge.url + '/v1/models') as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data == {'data': [{'id': 'test'}]}

    async def test_messages_translates_tool_calls(self, mock_upstream_messages):
        """POST /v1/messages translates vLLM tool_calls to Anthropic content blocks."""
        upstream_url, _ = mock_upstream_messages
        async with VllmBridge(upstream_url=upstream_url) as bridge:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    bridge.url + '/v1/messages',
                    json={'model': 'x', 'messages': []},
                ) as resp:
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
        async with VllmBridge(upstream_url=upstream_url) as bridge:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    bridge.url + '/v1/messages',
                    json={'model': 'x', 'stream': True, 'messages': []},
                ) as _resp:
                    pass

        assert len(received) == 1
        assert received[0].get('stream') is False

    async def test_forwards_upstream_error_status(self, mock_upstream_error):
        """Bridge forwards upstream error status and passes error body through unchanged."""
        async with VllmBridge(upstream_url=mock_upstream_error) as bridge:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    bridge.url + '/v1/messages',
                    json={'model': 'x', 'messages': []},
                ) as resp:
                    assert resp.status == 500
                    data = await resp.json()
                    assert data['type'] == 'error'
                    assert data['error']['message'] == 'boom'
