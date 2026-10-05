"""The hot-reload step of a config-deploy script, shared by
scripts/merge-deep-set-cap.sh and scripts/merge-pytest-n-ab-switch.sh.

Each of those scripts commits an edit to an orchestrator's config, then runs a
`python3 -` heredoc that fetches reload_config's report through this module
and checks only its own knob's disposition in it. Everything before that check
lives here once: importing the transport, calling reload_config through it,
and the operator-facing diagnostic for each way that can fail.

Two entry points, one fetch:

* `fetch_reload_report` raises `ReloadNotConfirmed` on any failure. Its
  `.failure` is a `ReloadFailure` whose value is the wire tag a caller may
  print in its own verdict, and its `.detail` is the operator prose.
* `reload_or_exit` is that same fetch, exiting through `die` with the detail.

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
from enum import Enum
from pathlib import Path
from typing import NoReturn

RELOAD_REPLY_TIMEOUT_SECS = 30.0


class ReloadFailure(str, Enum):  # noqa: UP042 - see docstring
    """Why a reload left the live config unconfirmed.

    A str mixin rather than StrEnum: the venvless fallback may be an older
    system python3, and this module must import there to deliver its remedy.
    """

    TRANSPORT_NOT_IMPORTABLE = "transport_not_importable"
    NO_RELOAD_REPORT = "no_reload_report"
    REPLY_TIMED_OUT = "reload_reply_timed_out"
    RELOAD_ERROR = "reload_error"


class ReloadNotConfirmed(Exception):
    """reload_config did not confirm the live config: *failure* names why,
    *detail* tells the operator."""

    def __init__(self, failure: ReloadFailure, detail: str):
        super().__init__(detail)
        self.failure = failure
        self.detail = detail


def die(prog: str, message: str) -> NoReturn:
    """Fail the deploy gate: `<prog>: <message>` on stderr, exit 1."""
    print(f"{prog}: {message}", file=sys.stderr)
    sys.exit(1)


def fetch_reload_report(
    port: str, *, committed_as: str, timeout: float = RELOAD_REPLY_TIMEOUT_SECS,
) -> dict:
    """reload_config's report from the orchestrator whose escalation MCP
    listens on 127.0.0.1:*port*, or raise ReloadNotConfirmed.

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
        raise ReloadNotConfirmed(
            ReloadFailure.TRANSPORT_NOT_IMPORTABLE,
            f"the MCP reload transport is not importable under {sys.executable}: {exc}\n"
            f"  looked for legibility/ under: {scripts_dir}\n"
            f"  remedy: sync this checkout so {scripts_dir.parent / '.venv'} exists, or re-run\n"
            f"  this script under `uv run --project shared`",
        ) from exc
    import httpx

    url = f"http://127.0.0.1:{port}/mcp"
    try:
        report = census_trigger.post_mcp_tool_call(url, "reload_config", {}, timeout=timeout)
    except httpx.ReadTimeout as exc:
        raise ReloadNotConfirmed(
            ReloadFailure.REPLY_TIMED_OUT,
            f"no reply from reload_config at {url} within {timeout}s: the request was sent, so "
            f"the tool may have run and applied the change, and the live config is unknown "
            f"(committed as {committed_as}; re-run to converge, or the value lands at the next restart)",
        ) from exc
    except Exception as exc:
        # Broad on purpose: census_trigger raises StatusFetchUnavailable for a
        # malformed or error envelope, RuntimeError for a failed handshake, and
        # httpx its own exceptions for a dead socket -- each leaves the live
        # config UNKNOWN, and a deploy gate must not pass on an unknown.
        raise ReloadNotConfirmed(
            ReloadFailure.NO_RELOAD_REPORT,
            f"reload_config never reached the tool at {url}: {type(exc).__name__}: {exc} "
            f"(committed as {committed_as}; the value lands at the next restart)",
        ) from exc
    # reload_config's OWN error field -- a config that failed to parse --
    # rides a perfectly successful tools/call the transport has no reason to reject.
    if report.get("error"):
        raise ReloadNotConfirmed(
            ReloadFailure.RELOAD_ERROR, f"reload_config error: {report['error']}",
        )
    return report


def reload_or_exit(port: str, *, prog: str, committed_as: str) -> dict:
    """`fetch_reload_report`, exiting through `die` with the refusal's detail."""
    try:
        return fetch_reload_report(port, committed_as=committed_as)
    except ReloadNotConfirmed as exc:
        die(prog, exc.detail)
