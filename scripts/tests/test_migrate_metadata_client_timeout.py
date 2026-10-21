"""FusedMemoryClient.call_tool: the timeout each request reaches the transport with."""

import asyncio
import json

import httpx
from migrate_metadata_modules_to_files import DEFAULT_CALL_TIMEOUT_S, FusedMemoryClient


def _recording_transport(seen: list) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get('method') == 'tools/call':
            seen.append((body['params']['name'], request.extensions['timeout']))
        if 'id' not in body:
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={'jsonrpc': '2.0', 'id': body['id'], 'result': {'structuredContent': {'success': True}}},
        )

    return httpx.MockTransport(handle)


_RESOLVE_TICKET_ARGUMENTS = {'ticket': 'tkt_test', 'project_root': '/p'}


def _call_in_order(*calls: tuple[str, dict, dict]) -> tuple[list[dict], list]:
    seen: list = []

    async def run() -> list[dict]:
        async with FusedMemoryClient('http://fused-memory.test', transport=_recording_transport(seen)) as client:
            return [await client.call_tool(name, arguments, **kwargs) for name, arguments, kwargs in calls]

    return asyncio.run(run()), seen


def _resolve_ticket_once(**call_kwargs) -> tuple[dict, list]:
    (reply,), seen = _call_in_order(('resolve_ticket', _RESOLVE_TICKET_ARGUMENTS, call_kwargs))
    return reply, seen


def _on_every_phase(seconds: float) -> dict:
    return {'connect': seconds, 'read': seconds, 'write': seconds, 'pool': seconds}


def test_a_call_without_a_timeout_reaches_the_transport_with_the_client_default():
    reply, seen = _resolve_ticket_once()

    assert seen == [('resolve_ticket', _on_every_phase(DEFAULT_CALL_TIMEOUT_S))]
    assert reply == {'success': True}


def test_a_per_call_timeout_reaches_the_transport_request():
    reply, seen = _resolve_ticket_once(timeout=180)

    assert seen == [('resolve_ticket', _on_every_phase(180))]
    assert reply == {'success': True}


def test_an_explicit_none_timeout_keeps_the_client_default_rather_than_disabling_it():
    _, seen = _resolve_ticket_once(timeout=None)

    assert seen == [('resolve_ticket', _on_every_phase(DEFAULT_CALL_TIMEOUT_S))]


def test_a_per_call_timeout_applies_to_its_own_call_only():
    _, seen = _call_in_order(
        ('resolve_ticket', _RESOLVE_TICKET_ARGUMENTS, {'timeout': 180}),
        ('get_task', {'id': '1', 'project_root': '/p'}, {}),
    )

    assert seen == [
        ('resolve_ticket', _on_every_phase(180)),
        ('get_task', _on_every_phase(DEFAULT_CALL_TIMEOUT_S)),
    ]
