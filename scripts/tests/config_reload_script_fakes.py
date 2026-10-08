"""Test doubles for the environment of a shell script that commits an
orchestrator config and then hot-reloads it through
`legibility.census_trigger.post_mcp_tool_call`.

That environment has three parts, each faked here for real rather than by a
stand-in binary on PATH: the escalation MCP the reload talks to (a loopback
HTTP server speaking its measured STATEFUL protocol, plus the catalogue of
ways its transport can fail), the git repo holding the config, and the
`python3` a caller's shell offers the reload step.

Stdlib only, and free of pytest, so every consumer imports it by bare name
(scripts/tests/conftest.py appends scripts/tests to sys.path).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def sse_frame(payload):
    """One `event: message` SSE frame -- how the escalation MCP frames every
    `tools/call` reply."""
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# The stateful escalation MCP, for real, on an ephemeral port
# ---------------------------------------------------------------------------

class FakeEscalationMcp:
    """A real HTTP server speaking the STATEFUL streamable-HTTP protocol the
    escalation MCP really speaks, answering `reload_config` with *report*.

    Modelled on the `_FakeMcpServer` fixture that task 5247 later retired,
    and extended with the two behaviours that fixture has no
    reason to model, its target (fused-memory :8002) being STATELESS: the 400
    rejection of a session-less request, and `text/event-stream` framing of
    the `tools/call` reply. Every behaviour below was measured live against
    the dark-factory escalation MCP at 127.0.0.1:8102 on 2026-09-12:

      * any request but `initialize` without an `mcp-session-id` header -> 400
        carrying the server's own "Missing session ID" envelope;
      * `initialize` -> 200 plus an `mcp-session-id` RESPONSE header;
      * `notifications/initialized` -> 202 with no body at all;
      * `tools/call` -> 200 `text/event-stream`, SSE-framed;
      * `DELETE` (session termination) -> 200.

    `received` records every (verb, headers, payload) seen, so a test asserts
    the script really handshook — and really released the session — rather
    than merely exited 0.

    `initialize_reply` / `tool_call_reply` each override one step of that
    exchange with a literal (status, headers, body) triple, which is how the
    transport-fault cases below are built without restating the handshake.
    `tool_call_delay` holds the authenticated `tools/call` reply back that many
    seconds (released early at exit), modelling a tool that outlives the
    client's read timeout.
    """

    SESSION_ID = "sess-5379-fake"

    MISSING_SESSION_REPLY = (
        400,
        {"Content-Type": "application/json"},
        json.dumps({
            "jsonrpc": "2.0",
            "id": "server-error",
            "error": {"code": -32600, "message": "Bad Request: Missing session ID"},
        }),
    )

    def __init__(self, report=None, *, initialize_reply=None, tool_call_reply=None,
                 tool_call_delay=0.0):
        self.report = report
        self.initialize_reply = initialize_reply
        self.tool_call_reply = tool_call_reply
        self.tool_call_delay = tool_call_delay
        self._release = threading.Event()
        self.received = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    payload = {"_raw": raw.decode()}
                outer.received.append((self.command, dict(self.headers), payload))
                self._reply(*outer._respond(self.headers, payload))

            def do_DELETE(self):
                outer.received.append((self.command, dict(self.headers), {}))
                self._reply(200, {}, "")

            def _reply(self, status, headers, body):
                data = body.encode()
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_port

    def _respond(self, headers, payload):
        """(status, headers, body) for one request."""
        method = payload.get("method")
        if method != "initialize" and not headers.get("mcp-session-id"):
            return self.MISSING_SESSION_REPLY
        if method == "initialize":
            return self.initialize_reply or (
                200,
                {"Content-Type": "application/json",
                 "mcp-session-id": self.SESSION_ID},
                json.dumps({"jsonrpc": "2.0", "id": payload.get("id"),
                            "result": {"protocolVersion": "2024-11-05"}}),
            )
        if method == "notifications/initialized":
            return (202, {}, "")
        if self.tool_call_delay > 0:
            self._release.wait(self.tool_call_delay)
        return self.tool_call_reply or (
            200,
            {"Content-Type": "text/event-stream"},
            sse_frame({"jsonrpc": "2.0", "id": payload.get("id"),
                       "result": {"structuredContent": self.report}}),
        )

    def rpc_methods(self):
        """The JSON-RPC methods the server saw, in order (a DELETE carries none)."""
        return [payload["method"] for _, _, payload in self.received if payload.get("method")]

    def called_tools(self):
        """The tool names every `tools/call` the server saw asked for."""
        return [
            (payload.get("params") or {}).get("name")
            for _, _, payload in self.received
            if payload.get("method") == "tools/call"
        ]

    def delete_count(self):
        """How many session-terminating DELETEs the server saw.

        Read by the wire tests rather than left to census_trigger's own suite:
        the server a reload script talks to is the long-lived escalation
        process, and a session it never releases stays in
        `StreamableHTTPSessionManager`'s registry with a live anyio task behind
        it, one per run.
        """
        return sum(1 for verb, _, _ in self.received if verb == "DELETE")

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._release.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def reload_report(**overrides):
    """A reload report in orchestrator/src/orchestrator/harness.py::
    reload_config's return shape, every field defaulted so each test states
    only what it varies.

    `unchanged` defaults to an INT count because that is what is really on
    the wire (config.py::ConfigDiff declares `unchanged: int`) -- it carries
    no key names.
    """
    report = {
        "reloaded": True,
        "config_path": None,
        "applied": {},
        "restart_required": {},
        "unchanged": 37,
        "error": None,
    }
    report.update(overrides)
    return report


# ---------------------------------------------------------------------------
# Every way the transport can refuse to deliver reload_config
# ---------------------------------------------------------------------------

INITIALIZE_WITHOUT_A_SESSION = (
    200,
    {"Content-Type": "application/json"},
    json.dumps({"jsonrpc": "2.0", "id": 1,
                "result": {"protocolVersion": "2024-11-05"}}),
)

ENVELOPE_LEVEL_ERROR = (
    200,
    {"Content-Type": "text/event-stream"},
    sse_frame({"jsonrpc": "2.0", "id": 1,
               "error": {"code": -32602, "message": "Unknown tool: reload_config"}}),
)

TOOL_IS_ERROR = (
    200,
    {"Content-Type": "text/event-stream"},
    sse_frame({"jsonrpc": "2.0", "id": 1, "result": {
        "isError": True,
        "content": [{"type": "text", "text": "ToolError: reload_config failed"}],
    }}),
)

# id -> FakeEscalationMcp keyword arguments.
TRANSPORT_FAULTS = {
    # The shape a raw single-shot POST hits against the live server today --
    # the direct regression test for the defect.
    "always_400": {"initialize_reply": FakeEscalationMcp.MISSING_SESSION_REPLY},
    # census_trigger raises a RuntimeError naming this; the script must
    # surface it rather than swallow it.
    "initialize_assigns_no_session": {"initialize_reply": INITIALIZE_WITHOUT_A_SESSION},
    # The request reached the server but never a tool.
    "envelope_level_error": {"tool_call_reply": ENVELOPE_LEVEL_ERROR},
    # FastMCP's shape for a raised ToolError: the tool ran and failed.
    "tool_is_error": {"tool_call_reply": TOOL_IS_ERROR},
}


class ClosedPort:
    """A port with nothing listening on it, for the dead-socket case.

    Bound and released, so it is free -- and the script's connect is the only
    thing racing for it. Carries the same `port` attribute a script driver
    reads from a FakeEscalationMcp, so the no-server case is driven the same
    way.
    """

    def __init__(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]


# ---------------------------------------------------------------------------
# The `python3` a caller's shell offers
#
# Measured on 2026-09-12: the system interpreter (/usr/bin/python3, 3.12.3)
# lacks httpx and pydantic, both of which the reload transport needs, while
# every venv in this tree has them. Whether an interpreter lacks the transport
# is decided by running the gate's own import step under it, so the premise a
# test skips on is the very branch the gate takes.
# ---------------------------------------------------------------------------

SYSTEM_PYTHON = "/usr/bin/python3"

_LACKS_THE_TRANSPORT = 3
_TRANSPORT_PROBE = f"""
import sys
sys.path.insert(0, sys.argv[1])
from _config_reload_gate import load_transport
try:
    load_transport()
except ImportError:
    sys.exit({_LACKS_THE_TRANSPORT})
"""


def system_python_without_the_transport():
    """The system interpreter, or a skip reason if it cannot play the part.

    Returns (interpreter, "") or (None, reason). An interpreter that CAN
    import the transport would make an interpreter-resolution test pass while
    proving nothing, so the premise is checked rather than assumed. A probe
    that breaks any other way raises: misreading it as "lacks the transport"
    would run those tests on a false premise.
    """
    if not os.path.exists(SYSTEM_PYTHON):
        return None, f"{SYSTEM_PYTHON} is absent"
    scripts_dir = Path(__file__).resolve().parent.parent
    probe = subprocess.run(
        [SYSTEM_PYTHON, "-c", _TRANSPORT_PROBE, str(scripts_dir)],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        return None, f"{SYSTEM_PYTHON} can import the transport, so it proves nothing"
    if probe.returncode == _LACKS_THE_TRANSPORT:
        return SYSTEM_PYTHON, ""
    raise RuntimeError(
        f"the transport probe broke under {SYSTEM_PYTHON} (rc={probe.returncode}): {probe.stderr}"
    )


def venvless_checkout_copy(tmp_path, script):
    """*script* copied into a fresh checkout at <tmp_path>/checkout that holds
    everything its reload step imports -- the real scripts/legibility and
    scripts/_config_reload_gate.py, symlinked -- but NO .venv, so the script's
    bare-`python3` interpreter fallback is the leg taken. Returns the copy."""
    scripts_dir = tmp_path / "checkout" / "scripts"
    scripts_dir.mkdir(parents=True)
    copy = scripts_dir / script.name
    shutil.copy2(script, copy)
    for imported in ("legibility", "_config_reload_gate.py"):
        (scripts_dir / imported).symlink_to(script.parent / imported)
    return copy


def path_python3_shimmed_to(tmp_path, interpreter):
    """An env whose PATH `python3` is *interpreter*."""
    return _path_python3_running(tmp_path, f'exec {interpreter} "$@"')


def path_python3_that_cannot_run(tmp_path):
    """An env whose PATH `python3` exits 1 without running anything."""
    return _path_python3_running(tmp_path, "exit 1")


def _path_python3_running(tmp_path, shell_body):
    """An env whose PATH `python3` is a /bin/sh shim executing *shell_body*."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "python3"
    shim.write_text(f"#!/bin/sh\n{shell_body}\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return env


# ---------------------------------------------------------------------------
# Real temp git repo holding a temp dark-factory-orchestrator.yaml
# ---------------------------------------------------------------------------

def git(repo, *args):
    """Run a git command against *repo* and return its stdout, raising loudly
    on failure -- test setup must never silently produce a repo that doesn't
    match the spec."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout


def init_repo(tmp_path):
    """An empty real git repo at <tmp_path>/repo, ready to commit into.

    A real repo (rather than a faked `git`) keeps the script's checkout
    resolution and its commit step -- including the "nothing to commit"
    path -- faithful, with no fake to drift.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    return repo


def commit_config(repo, text):
    """Write *text* as *repo*'s dark-factory-orchestrator.yaml and commit it,
    so the tree the script sees is clean. Returns the yaml path."""
    config = repo / "dark-factory-orchestrator.yaml"
    config.write_text(text)
    git(repo, "add", "dark-factory-orchestrator.yaml")
    git(repo, "commit", "-q", "-m", "seed")
    return config


def head(repo, *, short=False):
    """*repo*'s HEAD sha -- abbreviated as the scripts print it if *short*."""
    return git(repo, "rev-parse", *(["--short"] if short else []), "HEAD").strip()
