"""The hot-reload step of a config-deploy script, shared by
scripts/merge-deep-set-cap.sh and scripts/merge-pytest-n-ab-switch.sh.

Each of those scripts commits an edit to an orchestrator's config, then runs a
`python3 -` heredoc that calls `reload_or_exit` and checks only its own knob's
disposition in the report it returns. Everything before that check lives here
once: importing the transport, calling reload_config through it, and the
operator-facing diagnostic for each way that can fail.

The calling script's side of the contract:

* Every input reaches the heredoc by ARGV, never a pipe: `python3 -` reads its
  PROGRAM from stdin and the heredoc IS stdin, so a pipe into it is silently
  discarded (task 5398).
* The heredoc puts the script's own directory on sys.path, so this module and
  `legibility` both resolve from the SCRIPT's checkout, never the config's.
* The heredoc runs under `<scripts>/../.venv/bin/python3` when that exists,
  else `python3`. The transport is not stdlib (httpx, pydantic), so this
  module imports only stdlib at load time and turns a missing transport into
  a remedy naming that venv, never a traceback.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn


def die(prog: str, message: str) -> NoReturn:
    """Fail the deploy gate: `<prog>: <message>` on stderr, exit 1."""
    print(f"{prog}: {message}", file=sys.stderr)
    sys.exit(1)


def reload_or_exit(port: str, *, prog: str, committed_as: str) -> dict:
    """reload_config's report from the orchestrator whose escalation MCP
    listens on 127.0.0.1:*port*, or exit through `die`.

    That MCP is STATEFUL: a single-shot `tools/call` POST is a transport-level
    400 before any tool runs. So the call goes through
    `legibility.census_trigger.post_mcp_tool_call`, the one transport every
    consumer of the server shares (task 3644): the session handshake, SSE
    decoding, raising on a JSON-RPC `error` and on `result.isError`, and the
    session-terminating DELETE.

    *committed_as* names what the caller already committed, so a reload that
    never ran still tells the operator the value lands at the next restart.
    """
    scripts_dir = Path(__file__).parent
    try:
        from legibility import census_trigger
    except ImportError as exc:
        die(prog, f"the MCP reload transport is not importable under {sys.executable}: {exc}\n"
                  f"  looked for legibility/ under: {scripts_dir}\n"
                  f"  remedy: sync this checkout so {scripts_dir.parent / '.venv'} exists, or re-run\n"
                  f"  this script under `uv run --project shared`")
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        report = census_trigger.post_mcp_tool_call(url, "reload_config", {})
    except Exception as exc:
        # Broad on purpose: census_trigger raises StatusFetchUnavailable for a
        # malformed or error envelope, RuntimeError for a failed handshake, and
        # httpx its own exceptions for a dead socket -- each leaves the live
        # config UNKNOWN, and a deploy gate must not pass on an unknown.
        die(prog, f"reload_config never reached the tool at {url}: {type(exc).__name__}: {exc} "
                  f"(committed as {committed_as}; the value lands at the next restart)")
    # reload_config's OWN error field -- a config that failed to parse --
    # rides a perfectly successful tools/call the transport has no reason to reject.
    if report.get("error"):
        die(prog, f"reload_config error: {report['error']}")
    return report
