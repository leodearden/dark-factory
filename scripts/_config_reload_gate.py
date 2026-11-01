"""The hot-reload step of a config-deploy script, shared by
scripts/merge-deep-set-cap.sh and scripts/merge-pytest-n-ab-switch.sh.

Each of those scripts runs a `python3 -` heredoc that first LANDS its edit to
an orchestrator's config through that project's own merge queue
(scripts/_merge_queue_landing.py::land_file), never as a direct commit to its
main, then fetches reload_config's report through this module and checks only
its own knob's disposition in it. Everything between the landing and that
check lives here once: importing the transport, calling reload_config through
it, confirming the reload committed a re-read of the caller's own config file,
and the operator-facing diagnostic for each way that can fail. The landing
module shares this module's transport loader, its remedy and `die`.

Two entry points, one fetch:

* `fetch_reload_report` raises `ReloadNotConfirmed` on any failure. Its
  `.failure` is a `ReloadFailure` whose value is the wire tag a caller may
  print in its own verdict, and its `.detail` is the operator prose.
* `reload_or_exit` is that same fetch without the re-read check, exiting
  through `die` with the detail.

The calling script's side of the contract:

* Every input reaches the heredoc by ARGV, never a pipe: `python3 -` reads its
  PROGRAM from stdin and the heredoc IS stdin, so a pipe into it is silently
  discarded (task 5398).
* The heredoc puts the script's own directory on sys.path, so this module
  resolves from the SCRIPT's checkout, never the config's, and binds `shared`
  from that same checkout.
* The heredoc runs under `<scripts>/../.venv/bin/python3` when that exists,
  else `python3`. The transport, `shared.mcp_session`, needs httpx, so this
  module imports only stdlib at load time and turns a missing transport into
  a remedy naming that venv, never a traceback.
"""
from __future__ import annotations

import os
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
    RELOAD_NOT_COMMITTED = "reload_not_committed"
    DIFFERENT_CONFIG_FILE = "different_config_file"


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


def load_transport():
    """The reload transport's two modules, `(shared.mcp_session, httpx)`, or
    ImportError when this interpreter cannot run it.

    httpx is imported first, so an interpreter without it is known before
    `shared` is bound to this script's own checkout.
    """
    import httpx
    if (shared_src := str(Path(__file__).resolve().parents[1] / "shared" / "src")) not in sys.path:
        sys.path.insert(0, shared_src)
    from shared import mcp_session
    return mcp_session, httpx


def transport_not_importable_remedy(exc: ImportError) -> str:
    """The operator's next step when this interpreter cannot import a module
    an MCP call needs (the transport, or the landing's merge-state
    vocabulary): which one, where it looked, and how to get a venv that has
    them all."""
    checkout = Path(__file__).resolve().parents[1]
    return (
        f"a module the MCP calls need is not importable under {sys.executable}: {exc}\n"
        f"  looked for shared/ under {checkout / 'shared' / 'src'}, and for httpx in\n"
        f"  that interpreter's environment\n"
        f"  remedy: run `uv sync --all-packages` in {checkout} so {checkout / '.venv'}\n"
        f"  provides httpx, or re-run this script under `uv run --project shared`"
    )


def fetch_reload_report(
    port: str, *, committed_as: str, config_path: str | None,
    timeout: float = RELOAD_REPLY_TIMEOUT_SECS,
) -> dict:
    """reload_config's report from the orchestrator whose escalation MCP
    listens on 127.0.0.1:*port*, or raise ReloadNotConfirmed.

    *committed_as* names what the caller already committed, so a reload that
    never ran still tells the operator the value lands at the next restart.

    *config_path* is the config file the caller committed. The report confirms
    nothing about that file unless its reload committed and re-read exactly
    that file: any disposition it carries -- a key absent from `applied`, or
    present with the caller's value -- is equally produced by a rolled-back
    reload or by a reload of another orchestrator. None skips that check, for
    a caller that asks only for the report.
    """
    report = _call_reload_config(port, committed_as=committed_as, timeout=timeout)
    # reload_config's OWN error field -- a config that failed to parse --
    # rides a perfectly successful tools/call the transport has no reason to reject.
    if report.get("error"):
        raise ReloadNotConfirmed(
            ReloadFailure.RELOAD_ERROR, f"reload_config error: {report['error']}",
        )
    if config_path is not None:
        _confirm_reread_of(config_path, report)
    return report


def _call_reload_config(port: str, *, committed_as: str, timeout: float) -> dict:
    """reload_config's unwrapped result, or ReloadNotConfirmed naming why the
    transport could not deliver it."""
    try:
        mcp_session, httpx = load_transport()
    except ImportError as exc:
        raise ReloadNotConfirmed(
            ReloadFailure.TRANSPORT_NOT_IMPORTABLE, transport_not_importable_remedy(exc),
        ) from exc

    base_url = f"http://127.0.0.1:{port}"
    try:
        with mcp_session.SyncMcpSession(
            base_url, client_name="config-reload-gate", timeout=timeout,
        ) as session:
            return session.call_tool("reload_config", {})
    except mcp_session.McpCallError as exc:
        # Only a READ timeout follows a completely sent request. A connect,
        # write or pool timeout leaves the server no complete request to run,
        # so those are a reload that never reached the tool. Either way the
        # live config is UNKNOWN, and a deploy gate must not pass on an unknown.
        if isinstance(exc.__cause__, httpx.ReadTimeout):
            raise ReloadNotConfirmed(
                ReloadFailure.REPLY_TIMED_OUT,
                f"no reply from {base_url} within {timeout}s to a request already sent; the transport "
                f"cannot tell whether that was the session handshake or reload_config itself, so "
                f"the tool may have run and applied the change, and the live config is unknown "
                f"(committed as {committed_as}; re-run to converge, or the value lands at the next restart)",
            ) from exc
        raise ReloadNotConfirmed(
            ReloadFailure.NO_RELOAD_REPORT,
            f"reload_config never reached the tool at {base_url}: {exc.kind}: {exc} "
            f"(committed as {committed_as}; the value lands at the next restart)",
        ) from exc


def _confirm_reread_of(config_path: str, report: dict) -> None:
    """Raise ReloadNotConfirmed unless *report*'s reload committed and re-read
    *config_path*."""
    reloaded = report.get("reloaded")
    if not reloaded:
        raise ReloadNotConfirmed(
            ReloadFailure.RELOAD_NOT_COMMITTED,
            f"reload did not commit: reloaded={reloaded!r} "
            f"(a failed reload rolls every leaf back, so the live config is untouched)",
        )
    expected = os.path.realpath(config_path)
    reported = report.get("config_path")
    # `reported` is the ORCHESTRATOR's own ORCH_CONFIG_PATH, so a RELATIVE one
    # is resolved by the SERVER's cwd. realpath here would resolve it against
    # OURS, and a unit whose ORCH_CONFIG_PATH is the bare filename would then
    # match us whenever the caller runs from its project root. Uncomparable is
    # not a match.
    if (not isinstance(reported, str) or not os.path.isabs(reported)
            or os.path.realpath(reported) != expected):
        raise ReloadNotConfirmed(
            ReloadFailure.DIFFERENT_CONFIG_FILE,
            f"reload re-read a different file: config_path={reported!r} expected={expected!r}",
        )


def reload_or_exit(port: str, *, prog: str, committed_as: str) -> dict:
    """`fetch_reload_report` without the re-read check, exiting through `die`
    with the refusal's detail."""
    try:
        return fetch_reload_report(port, committed_as=committed_as, config_path=None)
    except ReloadNotConfirmed as exc:
        die(prog, exc.detail)
