"""Fidelity tests for shared.testing_mcp_stub against the live servers it imitates.

The stub stands in for the escalation server (stateful) and fused-memory
(stateless) in every B9 suite, so its own wire behaviour is pinned here with a
raw httpx client: a double that drifted from the servers would let a client
pass against the stub and fail in the fleet (INV-10).
"""
from __future__ import annotations

import dataclasses
import json

import httpx
import pytest

from shared.mcp_post import MCP_POST_HEADERS, decode_mcp_response_body
from shared.testing_mcp_stub import RecordedRequest, StubAnswer, StubMcpServer

PAYLOAD = {"success": True, "id": "7"}


def _request(method: str, params: dict | None = None, *, request_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


def _notification(method: str) -> dict:
    return {"jsonrpc": "2.0", "method": method}


INITIALIZE = _request(
    "initialize",
    {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "stub-fidelity", "version": "0"},
    },
)
TOOLS_CALL = _request("tools/call", {"name": "update_task", "arguments": {"id": "7"}}, request_id=2)


def _post(server: StubMcpServer, body: dict, *, session_id: str | None = None) -> httpx.Response:
    headers = dict(MCP_POST_HEADERS)
    if session_id is not None:
        headers["mcp-session-id"] = session_id
    return httpx.post(f"{server.url}/mcp", json=body, headers=headers, follow_redirects=False)


def _open_session(server: StubMcpServer) -> str | None:
    """Handshake the way a client must; the minted id for a stateful stub."""
    session_id = _post(server, INITIALIZE).headers.get("mcp-session-id")
    _post(server, _notification("notifications/initialized"), session_id=session_id)
    return session_id


class TestStateless:
    def test_initialize_answers_json_without_a_session_id(self):
        with StubMcpServer(stateful=False) as server:
            resp = _post(server, INITIALIZE)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert "mcp-session-id" not in resp.headers
        body = resp.json()
        assert body["id"] == 1
        assert body["result"]["protocolVersion"] == "2025-03-26"

    def test_initialized_notification_is_accepted_with_an_empty_body(self):
        with StubMcpServer(stateful=False) as server:
            resp = _post(server, _notification("notifications/initialized"))
        assert resp.status_code == 202
        assert resp.content == b""

    def test_tools_call_carries_structured_content_and_the_same_json_text(self):
        with StubMcpServer(stateful=False, payload=PAYLOAD) as server:
            resp = _post(server, TOOLS_CALL)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        result = resp.json()["result"]
        assert result["structuredContent"] == PAYLOAD
        assert json.loads(result["content"][0]["text"]) == PAYLOAD
        assert not result.get("isError")

    def test_mints_no_session(self):
        with StubMcpServer(stateful=False) as server:
            _open_session(server)
            _post(server, TOOLS_CALL)
            assert server.minted_session_ids == ()
            assert server.open_session_ids == frozenset()


class TestStateful:
    def test_tools_call_without_a_session_is_refused(self):
        with StubMcpServer(stateful=True) as server:
            resp = _post(server, TOOLS_CALL)
        assert resp.status_code == 400
        assert "Missing session ID" in resp.json()["error"]["message"]

    def test_initialize_mints_the_session_id(self):
        with StubMcpServer(stateful=True) as server:
            resp = _post(server, INITIALIZE)
            minted = resp.headers.get("mcp-session-id")
            assert resp.status_code == 200
            assert minted
            assert server.minted_session_ids == (minted,)
            assert server.open_session_ids == frozenset({minted})

    def test_unknown_session_id_is_not_found(self):
        with StubMcpServer(stateful=True) as server:
            resp = _post(server, TOOLS_CALL, session_id="never-minted")
        assert resp.status_code == 404

    def test_tools_call_in_session_is_sse_framed(self):
        with StubMcpServer(stateful=True, payload=PAYLOAD) as server:
            session_id = _open_session(server)
            resp = _post(server, TOOLS_CALL, session_id=session_id)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "event: message" in resp.text
        data_line = next(line for line in resp.text.splitlines() if line.startswith("data: "))
        assert json.loads(data_line[len("data: "):])["result"]["structuredContent"] == PAYLOAD

    def test_delete_releases_the_session(self):
        with StubMcpServer(stateful=True) as server:
            session_id = _open_session(server)
            assert session_id is not None
            resp = httpx.delete(
                f"{server.url}/mcp", headers={**MCP_POST_HEADERS, "mcp-session-id": session_id}
            )
            assert resp.status_code == 200
            assert server.open_session_ids == frozenset()
            assert _post(server, TOOLS_CALL, session_id=session_id).status_code == 404


@pytest.mark.parametrize("stateful", [False, True], ids=["stateless", "stateful"])
class TestEitherMode:
    def test_trailing_slash_redirects_to_mcp(self, stateful):
        with StubMcpServer(stateful=stateful) as server:
            resp = httpx.post(
                f"{server.url}/mcp/", json=INITIALIZE, headers=MCP_POST_HEADERS,
                follow_redirects=False,
            )
        assert resp.status_code == 307
        assert resp.headers["location"] == "/mcp"

    @pytest.mark.parametrize("accept", [None, "text/event-stream"], ids=["absent", "sse-only"])
    def test_post_without_the_needed_media_types_is_not_acceptable(self, stateful, accept):
        headers = {"Content-Type": "application/json"}
        if accept is not None:
            headers["Accept"] = accept
        with StubMcpServer(stateful=stateful) as server:
            resp = httpx.post(f"{server.url}/mcp", json=INITIALIZE, headers=headers)
        assert resp.status_code == 406
        assert resp.json()["error"]["message"].startswith("Not Acceptable")

    def test_requests_are_recorded_in_order_as_frozen_values(self, stateful):
        with StubMcpServer(stateful=stateful) as server:
            _post(server, INITIALIZE)
            httpx.delete(f"{server.url}/mcp")
            recorded = server.requests
        assert [(r.http_method, r.path) for r in recorded] == [("POST", "/mcp"), ("DELETE", "/mcp")]
        first = recorded[0]
        assert isinstance(first, RecordedRequest)
        assert first.body == INITIALIZE
        assert first.headers["accept"] == MCP_POST_HEADERS["Accept"]
        assert all(name == name.lower() for name in first.headers)
        assert recorded[1].body is None
        frozen_field = "path"
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(first, frozen_field, "/elsewhere")


    def test_unavailable_tool_calls_are_503_then_served(self, stateful):
        with StubMcpServer(stateful=stateful, unavailable_tool_calls=1) as server:
            session_id = _open_session(server)
            refused = _post(server, TOOLS_CALL, session_id=session_id)
            served = _post(server, TOOLS_CALL, session_id=session_id)
        assert refused.status_code == 503
        assert served.status_code == 200
        assert decode_mcp_response_body(served)["result"]["structuredContent"]


@pytest.mark.parametrize("stateful", [False, True], ids=["stateless", "stateful"])
class TestFailureAnswers:
    """A scripted answer replaces only the tools/call reply; the handshake still works."""

    def _call(self, stateful: bool, answer: StubAnswer) -> tuple[httpx.Response, str | None]:
        with StubMcpServer(stateful=stateful, answer=answer) as server:
            session_id = _open_session(server)
            return _post(server, TOOLS_CALL, session_id=session_id), session_id

    def test_handshake_is_unaffected(self, stateful):
        with StubMcpServer(stateful=stateful, answer=StubAnswer.NON_JSON_BODY) as server:
            init = _post(server, INITIALIZE)
            assert init.status_code == 200
            assert decode_mcp_response_body(init)["result"]["serverInfo"]

    def test_is_error(self, stateful):
        resp, _ = self._call(stateful, StubAnswer.IS_ERROR)
        assert resp.status_code == 200
        result = decode_mcp_response_body(resp)["result"]
        assert result["isError"] is True
        assert result["content"][0]["text"].startswith("Error calling tool")

    def test_non_json_body(self, stateful):
        resp, _ = self._call(stateful, StubAnswer.NON_JSON_BODY)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert resp.text == "Internal Server Error"

    def test_non_json_text(self, stateful):
        resp, _ = self._call(stateful, StubAnswer.NON_JSON_TEXT)
        assert resp.status_code == 200
        result = decode_mcp_response_body(resp)["result"]
        assert "structuredContent" not in result
        with pytest.raises(ValueError):
            json.loads(result["content"][0]["text"])

    def test_empty_202(self, stateful):
        resp, _ = self._call(stateful, StubAnswer.EMPTY_202)
        assert resp.status_code == 202
        assert resp.content == b""

    def test_redirect_307_points_back_at_mcp(self, stateful):
        resp, _ = self._call(stateful, StubAnswer.REDIRECT_307)
        assert resp.status_code == 307
        assert resp.headers["location"] == "/mcp"

    def test_json_rpc_error(self, stateful):
        resp, _ = self._call(stateful, StubAnswer.JSON_RPC_ERROR)
        assert resp.status_code == 200
        body = decode_mcp_response_body(resp)
        assert "result" not in body
        assert body["error"]["code"] == -32602
