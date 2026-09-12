"""Tests for merge-pytest-n-ab-switch.sh — drives the real script via
subprocess against a REAL loopback MCP server on an ephemeral port (the
test_consume_redispatch_requests.py::_FakeMcpServer idiom, extended to the
STATEFUL escalation server's measured behaviour) and a REAL temp git repo
holding a temp dark-factory-orchestrator.yaml (the
test_deploy_w11_lane_lifecycle.py idiom), so step 1's YAML editor, step 2's
commit idempotency and step 4's assertion are exercised in COMPOSITION —
which is where the crash-resume defect lives: file already at the value ->
nothing to commit -> the reload reports no verify_env change.

The reload travels over a real socket on purpose. A faked `curl` can only
ever return what the test author already believes the wire looks like, so it
cannot catch a wrong belief ABOUT the wire — and that is how a reload step
that never once reached the tool in production shipped green.

Nothing here touches a live orchestrator: the only server reachable is the
one the test itself owns, bound to an ephemeral port.
"""
from __future__ import annotations

import json
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "merge-pytest-n-ab-switch.sh"
KEY = "PYTEST_XDIST_AUTO_NUM_WORKERS"


def _sse_frame(payload):
    """One `event: message` SSE frame -- how the escalation MCP frames every
    `tools/call` reply."""
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# The stateful escalation MCP, for real, on an ephemeral port
# ---------------------------------------------------------------------------

class _FakeEscalationMcp:
    """A real HTTP server speaking the STATEFUL streamable-HTTP protocol the
    escalation MCP really speaks, answering `reload_config` with *report*.

    Modelled on scripts/tests/test_consume_redispatch_requests.py::
    _FakeMcpServer and extended with the two behaviours that fixture has no
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

    `received` records every (headers, payload) seen, so a test asserts the
    script really handshook rather than merely exited 0.

    `initialize_reply` / `tool_call_reply` each override one step of that
    exchange with a literal (status, headers, body) triple, which is how the
    transport-fault cases below are built without restating the handshake.
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

    def __init__(self, report=None, *, initialize_reply=None, tool_call_reply=None):
        self.report = report
        self.initialize_reply = initialize_reply
        self.tool_call_reply = tool_call_reply
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
                outer.received.append((dict(self.headers), payload))
                self._reply(*outer._respond(self.headers, payload))

            def do_DELETE(self):
                outer.received.append((dict(self.headers), {}))
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
        return self.tool_call_reply or (
            200,
            {"Content-Type": "text/event-stream"},
            _sse_frame({"jsonrpc": "2.0", "id": payload.get("id"),
                        "result": {"structuredContent": self.report}}),
        )

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _rpc_methods(server):
    """The JSON-RPC methods the server saw, in order (a DELETE carries none)."""
    return [payload["method"] for _, payload in server.received if payload.get("method")]


def _called_tools(server):
    """The tool names every `tools/call` the server saw asked for."""
    return [
        (payload.get("params") or {}).get("name")
        for _, payload in server.received
        if payload.get("method") == "tools/call"
    ]


def _report(**overrides):
    """A reload report in orchestrator/src/orchestrator/harness.py::
    reload_config's return shape, every field defaulted so each test states
    only what it varies.

    `unchanged` defaults to an INT count because that is what is really on
    the wire (config.py::ConfigDiff declares `unchanged: int`) -- it carries
    no key names, which is why absence from `applied` is the only converged
    signal there is.
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
# Real temp git repo holding a temp dark-factory-orchestrator.yaml
# ---------------------------------------------------------------------------

def _git(repo, *args):
    """Run a git command against *repo*, raising loudly on failure -- test
    setup must never silently produce a repo that doesn't match the spec."""
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path, verify_env_value, *, marker=False):
    """Build a real temp git repo at <tmp_path>/repo carrying a
    dark-factory-orchestrator.yaml whose top-level `verify_env:` block pins
    KEY to *verify_env_value*, and commit it so the tree is clean.

    A real repo (rather than a faked `git`) makes both halves of step 2
    faithful -- the `git rev-parse --show-toplevel` resolution and the
    "nothing to commit" idempotent path this bug lives on -- with no fake to
    drift. `marker=True` precedes the key with an `  # A/B arm: ...` line,
    the state a previously-switched config file is left in.

    Returns the yaml path.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    marker_line = (
        f'  # A/B arm: {KEY} set to "{verify_env_value}" at '
        "2026-09-10T12:42:14Z by scripts/merge-pytest-n-ab-switch.sh\n"
    )
    config = repo / "dark-factory-orchestrator.yaml"
    config.write_text(
        "project_root: /nowhere\n"
        "max_concurrent_tasks: 3\n"
        "\n"
        "verify_env:\n"
        + (marker_line if marker else "")
        + f'  {KEY}: "{verify_env_value}"\n'
        '  PYTHONWARNINGS: "ignore"\n'
        "\n"
        "review:\n"
        "  enabled: true\n"
    )
    _git(repo, "add", "dark-factory-orchestrator.yaml")
    _git(repo, "commit", "-q", "-m", "seed")
    return config


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run(server, config_path, value):
    """Run the real script for *value* against *config_path*, pointing its
    reload at *server*'s real ephemeral port."""
    return subprocess.run(
        ["bash", str(SCRIPT), value, str(config_path), str(server.port)],
        capture_output=True, text=True, timeout=60,
    )


def _head(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _verdict(proc):
    """Parse the script's last non-empty stdout line as the JSON verdict."""
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"no stdout at all; rc={proc.returncode} stderr={proc.stderr}"
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# The flip path: a real value change, hot-applied
# ---------------------------------------------------------------------------

def test_flip_reports_outcome_applied_and_commits(tmp_path):
    """A genuine 16 -> 8 switch edits the yaml, lands exactly one commit
    touching only that file, and reports `outcome: applied`."""
    config = _make_repo(tmp_path, "16")
    repo = config.parent
    before = _head(repo)

    with _FakeEscalationMcp(_report(
        reloaded=True,
        config_path=str(config),
        applied={"verify_env": {"old": {KEY: "16"}, "new": {KEY: "8"}}},
    )) as server:
        proc = _run(server, config, "8")
        methods = _rpc_methods(server)

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    verdict = _verdict(proc)
    assert verdict["outcome"] == "applied"
    assert verdict["switched_to"] == "8"
    assert verdict["commit"] != "already-at-8"
    assert "initialize" in methods, (
        f"the script never handshook; the server saw {methods}"
    )

    landed = subprocess.run(
        ["git", "-C", str(repo), "rev-list", f"{before}..HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    assert len(landed) == 1, f"expected exactly one new commit, got {landed}"
    touched = subprocess.run(
        ["git", "-C", str(repo), "show", "--name-only", "--format=", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    assert touched == ["dark-factory-orchestrator.yaml"]
    assert f'{KEY}: "8"' in config.read_text()


# ---------------------------------------------------------------------------
# The converged path: a crash-resume onto a config that already carries the
# value, against an orchestrator whose live config already matches it
# ---------------------------------------------------------------------------

def test_converged_resume_exits_zero_when_reload_reports_no_verify_env_change(tmp_path):
    """Re-running the switch for a value the file and the running config
    BOTH already carry is success, not failure.

    The reload genuinely happens and genuinely applies other hot leaves, but
    verify_env is equal on both sides so it never appears under `applied` --
    `unchanged` is a bare int count, so absence is the only converged signal
    on the wire. Nothing to commit is likewise already success (step 2's
    `already-at-<value>` sentinel); step 4 must reach the same verdict.
    """
    config = _make_repo(tmp_path, "8", marker=True)
    repo = config.parent
    before_head = _head(repo)
    before_bytes = config.read_bytes()

    with _FakeEscalationMcp(_report(
        reloaded=True,
        error=None,
        config_path=str(config),
        applied={"max_turns.architect": {"old": 40, "new": 60}},
    )) as server:
        proc = _run(server, config, "8")

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert "applied.verify_env does not carry" not in proc.stderr
    verdict = _verdict(proc)
    assert verdict["outcome"] == "already_converged"
    assert verdict["switched_to"] == "8"
    assert verdict["commit"] == "already-at-8"

    assert _head(repo) == before_head, "a converged re-run must commit nothing"
    assert config.read_bytes() == before_bytes, (
        "the idempotent rewrite must restore the prior A/B marker byte for "
        "byte, not stack a fresh one"
    )


def test_converged_resume_over_the_real_stateful_transport(tmp_path):
    """The converged verdict is reached THROUGH the session handshake.

    Its sibling above owns the git/yaml idempotence; this one owns the wire:
    the same converged report, asserted to have been fetched by an
    `initialize` followed by a `tools/call` naming `reload_config`. A
    session-less single-shot POST never gets that far -- it is rejected with
    a 400 before any tool runs -- so exiting 0 here cannot be luck.

    The sequence is asserted as "some tools/call AFTER the initialize", not
    as a fixed list: `post_mcp_envelope` is deliberately ADAPTIVE, sending
    the envelope session-less first and handshaking only on the 400, so the
    real wire also carries a rejected tools/call ahead of the handshake.
    """
    config = _make_repo(tmp_path, "8", marker=True)

    with _FakeEscalationMcp(_report(
        reloaded=True,
        config_path=str(config),
        applied={"max_turns.architect": {"old": 40, "new": 60}},
    )) as server:
        proc = _run(server, config, "8")
        methods = _rpc_methods(server)
        tools = _called_tools(server)

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert _verdict(proc)["outcome"] == "already_converged"
    assert "initialize" in methods, methods
    assert "tools/call" in methods[methods.index("initialize"):], methods
    assert set(tools) == {"reload_config"}, f"methods={methods}"


# ---------------------------------------------------------------------------
# Responses that must NOT be read as success
#
# Absence of verify_env from `applied` is strictly WEAKER than "converged":
# the very same absence is produced by a reload that rolled every leaf back,
# and by a reload of a different orchestrator entirely (the port is a
# caller-supplied argument, so a wrong one reaches another project's MCP).
# This is a DEPLOY gate -- blessing an undeployed arm is worse than the
# over-strict assertion the converged branch removes.
#
# Each case pins the SPECIFIC diagnostic it is about, not merely a non-zero
# exit: a transport rejection also exits non-zero, and a bare `returncode !=
# 0` would let one stand in for all four and pass for the wrong reason.
# ---------------------------------------------------------------------------

def _converged_verdict_lines(proc):
    """Every stdout line that parses as a verdict claiming convergence."""
    claims = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("outcome") == "already_converged":
            claims.append(line)
    return claims


# id -> (builder taking (this repo's config, another project's config) and
# returning a reload report with verify_env ABSENT from `applied`, the stderr
# marker naming why that absence is not convergence).
_UNCORROBORATED_ABSENCE = {
    # Control: the existing error branch already rejects this one, so a green
    # here proves the group's harness really drives the script.
    "reload_failed_loudly": (
        lambda config, other: _report(
            reloaded=False, error="load_config: while parsing a block mapping",
            config_path=str(config),
        ),
        "reload_config error",
    ),
    # apply_reload rolled every leaf back, so the live config is untouched.
    "reload_failed_silently": (
        lambda config, other: _report(
            reloaded=False, error=None, config_path=str(config),
        ),
        "did not commit it",
    ),
    # We reloaded something that is not the file we just edited, so its
    # verify_env says nothing about ours.
    "different_config_file": (
        lambda config, other: _report(reloaded=True, config_path=str(other)),
        "re-read a different file",
    ),
    "no_config_path": (
        lambda config, other: _report(reloaded=True, config_path=None),
        "re-read a different file",
    ),
}


@pytest.mark.parametrize(
    ("build_report", "diagnostic"),
    list(_UNCORROBORATED_ABSENCE.values()),
    ids=list(_UNCORROBORATED_ABSENCE),
)
def test_uncorroborated_absence_is_not_convergence(tmp_path, build_report, diagnostic):
    """An absent verify_env that is NOT corroborated by a committed reload of
    THIS config file must fail, naming which corroborator was missing."""
    config = _make_repo(tmp_path, "8", marker=True)
    other = tmp_path / "other-project" / "dark-factory-orchestrator.yaml"
    other.parent.mkdir()
    other.write_text(f'verify_env:\n  {KEY}: "2"\n')

    with _FakeEscalationMcp(build_report(config, other)) as server:
        proc = _run(server, config, "8")

    assert proc.returncode != 0, (
        f"an uncorroborated absence was read as success: stdout={proc.stdout}"
    )
    assert not _converged_verdict_lines(proc)
    assert diagnostic in proc.stderr, f"stderr={proc.stderr}"


def test_applied_verify_env_carrying_a_different_value_still_fails(tmp_path):
    """The strict branch stays strict: a PRESENT verify_env carrying some
    other value is a contradiction, never convergence."""
    config = _make_repo(tmp_path, "8", marker=True)

    with _FakeEscalationMcp(_report(
        reloaded=True,
        config_path=str(config),
        applied={"verify_env": {"old": {KEY: "8"}, "new": {KEY: "16"}}},
    )) as server:
        proc = _run(server, config, "8")

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert "applied.verify_env does not carry" in proc.stderr
    assert not _converged_verdict_lines(proc)


# ---------------------------------------------------------------------------
# The reload never reached the tool
#
# A transport rejection must never be readable as a rolled-back reload. The
# defect this file exists to close was not that the script passed when it
# should have failed -- it was that it failed while naming the WRONG cause:
# every session-less POST got a 400 before any tool ran, and the script
# reported `reloaded=None` and blamed an in-orchestrator rollback for it.
# ---------------------------------------------------------------------------

TRANSPORT_MARKER = "reload_config never reached the tool"

_INITIALIZE_WITHOUT_A_SESSION = (
    200,
    {"Content-Type": "application/json"},
    json.dumps({"jsonrpc": "2.0", "id": 1,
                "result": {"protocolVersion": "2024-11-05"}}),
)

_ENVELOPE_LEVEL_ERROR = (
    200,
    {"Content-Type": "text/event-stream"},
    _sse_frame({"jsonrpc": "2.0", "id": 1,
                "error": {"code": -32602, "message": "Unknown tool: reload_config"}}),
)

_TOOL_IS_ERROR = (
    200,
    {"Content-Type": "text/event-stream"},
    _sse_frame({"jsonrpc": "2.0", "id": 1, "result": {
        "isError": True,
        "content": [{"type": "text", "text": "ToolError: reload_config failed"}],
    }}),
)

_TRANSPORT_FAULTS = {
    # The shape a raw single-shot POST hits against the live server today --
    # the direct regression test for the defect.
    "always_400": {"initialize_reply": _FakeEscalationMcp.MISSING_SESSION_REPLY},
    # census_trigger raises a RuntimeError naming this; the script must
    # surface it rather than swallow it.
    "initialize_assigns_no_session": {"initialize_reply": _INITIALIZE_WITHOUT_A_SESSION},
    # The request reached the server but never a tool.
    "envelope_level_error": {"tool_call_reply": _ENVELOPE_LEVEL_ERROR},
    # FastMCP's shape for a raised ToolError: the tool ran and failed.
    "tool_is_error": {"tool_call_reply": _TOOL_IS_ERROR},
}


class _ClosedPort:
    """A port with nothing listening on it, for the dead-socket case.

    Bound and released, so it is free -- and the script's connect is the only
    thing racing for it. Carries the same `port` attribute `_run` reads from a
    real server, so the no-server case is driven by the same helper.
    """

    def __init__(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]


def _assert_failed_closed_on_the_transport(proc):
    """The one verdict every transport fault must reach: a loud failure that
    names the TRANSPORT, and never the in-orchestrator rollback it is not."""
    assert proc.returncode != 0, f"a transport fault was read as success: {proc.stdout}"
    assert not _converged_verdict_lines(proc)
    assert TRANSPORT_MARKER in proc.stderr, f"stderr={proc.stderr}"
    assert "rolls every leaf back" not in proc.stderr, (
        f"a request that never ran a tool was blamed on a rollback: {proc.stderr}"
    )
    assert "reloaded=None" not in proc.stderr, f"stderr={proc.stderr}"


@pytest.mark.parametrize(
    "fault", list(_TRANSPORT_FAULTS.values()), ids=list(_TRANSPORT_FAULTS)
)
def test_a_reload_that_never_reached_the_tool_fails_closed(tmp_path, fault):
    """Every way the transport can refuse to deliver reload_config."""
    config = _make_repo(tmp_path, "8", marker=True)
    repo = config.parent
    before_head = _head(repo)

    with _FakeEscalationMcp(_report(config_path=str(config)), **fault) as server:
        proc = _run(server, config, "8")

    _assert_failed_closed_on_the_transport(proc)
    assert _head(repo) == before_head, "a failed reload must not land a commit"


def test_a_dead_socket_fails_closed(tmp_path):
    """Nothing listening at all -- httpx's own transport exception, which is
    neither of the two census_trigger raises and must fail the same way."""
    config = _make_repo(tmp_path, "8", marker=True)
    repo = config.parent
    before_head = _head(repo)

    proc = _run(_ClosedPort(), config, "8")

    _assert_failed_closed_on_the_transport(proc)
    assert _head(repo) == before_head, "a failed reload must not land a commit"


def test_the_transport_diagnostic_keeps_the_committed_shas_remedy(tmp_path):
    """On the FLIP path the commit HAS landed, so the operator needs to know
    which sha carries the value and that a restart will pick it up.

    That is what the deleted `curl ... || die "... (committed as ${SHA}; the
    value lands at the next restart)"` carried; losing it when curl went away
    would leave an operator with a failed deploy and no next step.
    """
    config = _make_repo(tmp_path, "16")
    repo = config.parent

    with _FakeEscalationMcp(
        initialize_reply=_FakeEscalationMcp.MISSING_SESSION_REPLY,
    ) as server:
        proc = _run(server, config, "8")

    _assert_failed_closed_on_the_transport(proc)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert sha in proc.stderr, f"stderr={proc.stderr}"
    assert "lands at the next restart" in proc.stderr, f"stderr={proc.stderr}"
