#!/usr/bin/env python3
"""Resume another project's orchestrator scheduler through its escalation MCP.

The `before_done` action of a dark-factory deterministic deploy task whose job
is to lift an operator pause on a sibling project — the reify alpha-spine
capstone (`reify/docs/prds/alpha-spine.md` D3) is the first. A paused
scheduler dispatches nothing, deterministic tasks included, so the capstone
cannot live in the paused project's own store; it lives here and reaches the
paused project through the `resume_scheduler` tool its escalation server
exposes.

Stdlib only, so the runner can execute it with the host interpreter: a
streamable-HTTP MCP session (initialize, initialized, tools/call) with the
server-minted `mcp-session-id` echoed on every later post. Exit 0 only when
the server reports the scheduler resumed (or already running); any refusal,
transport failure or unexpected payload exits non-zero so the runner
escalates instead of stamping a resume that did not happen.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any

MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_TIMEOUT_SECONDS = 30


class ResumeRefused(Exception):
    """The escalation server answered, but did not resume the scheduler."""


class McpSession:
    def __init__(self, url: str) -> None:
        self.url = url
        self.session_id: str | None = None
        self.next_id = 0

    def open(self) -> None:
        self._post(
            self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "resume_project_scheduler", "version": "1"},
                },
            )
        )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        reply = self._post(self._request("tools/call", {"name": name, "arguments": arguments}))
        return tool_payload(reply, name)

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.next_id += 1
        return {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=MCP_TIMEOUT_SECONDS) as response:
            self.session_id = response.headers.get("mcp-session-id") or self.session_id
            return read_reply(response) if "id" in payload else None


def read_reply(response: Any) -> dict[str, Any] | None:
    if "text/event-stream" not in response.headers.get("Content-Type", ""):
        body = response.read()
        return json.loads(body) if body.strip() else None
    for line in response:
        if line.startswith(b"data:"):
            message = json.loads(line[len(b"data:"):])
            if isinstance(message, dict) and ("result" in message or "error" in message):
                return message
    raise ValueError("SSE reply ended with no JSON-RPC response")


def tool_payload(reply: dict[str, Any] | None, tool_name: str) -> dict[str, Any]:
    if reply is None:
        raise ResumeRefused("empty tools/call reply")
    if "error" in reply:
        raise ResumeRefused(f"JSON-RPC error: {reply['error']}")
    result = reply.get("result") or {}
    text = "".join(
        item.get("text", "") for item in result.get("content", []) if item.get("type") == "text"
    )
    if result.get("isError"):
        raise ResumeRefused(f"{tool_name} returned an error: {text}")
    payload = result.get("structuredContent")
    if not isinstance(payload, dict):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ResumeRefused(f"unparseable {tool_name} payload {text!r}: {error}") from error
    if not isinstance(payload, dict) or "error" in payload:
        raise ResumeRefused(f"{tool_name} refused: {payload}")
    return payload


def resume(url: str, reason: str) -> dict[str, Any]:
    session = McpSession(url)
    session.open()
    payload = session.call_tool("resume_scheduler", {"reason": reason})
    if payload.get("resumed") is False and not payload.get("was_paused", True):
        return payload
    if not payload.get("resumed", False):
        raise ResumeRefused(f"resume_scheduler did not report resumed: {payload}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--url", required=True, help="the project's escalation MCP endpoint, e.g. http://127.0.0.1:8100/mcp")
    parser.add_argument("--reason", required=True, help="audit reason recorded on the scheduler_resumed event")
    args = parser.parse_args(argv)
    try:
        payload = resume(args.url, args.reason)
    except (ResumeRefused, OSError, ValueError) as error:
        print(f"resume_project_scheduler: FAILED against {args.url}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"url": args.url, "reply": payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
