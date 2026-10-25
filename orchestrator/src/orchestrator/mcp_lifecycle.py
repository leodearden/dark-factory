"""Start/stop fused-memory HTTP server as a managed subprocess."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import httpx
from shared.jcodemunch_launch import jcodemunch_server_config
from shared.mcp_session import McpSession
from shared.proc_group import terminate_process_group

from orchestrator.config import OrchestratorConfig
from orchestrator.fm_retry import fm_retry_backoffs, is_fm_null_sentinel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP startup-timeout constant and helper (bounded drop-on-fail)
# ---------------------------------------------------------------------------

# Generous for the prebuilt pinned stdio launcher (starts in <1s) and
# already-running HTTP MCPs (connect near-instantly), yet ~40x faster than
# the 1200s SIGTERM wall for a hung/incompatible server.
# The Claude Code CLI drops any MCP server that doesn't become ready within
# MCP_TIMEOUT ms; the agent still reaches turn 1 with that server absent.
MCP_STARTUP_TIMEOUT_MS: str = '30000'


def apply_mcp_startup_env(
    env_overrides: dict[str, str] | None,
) -> dict[str, str]:
    """Return a copy of *env_overrides* with MCP_TIMEOUT set as a default.

    Injects the Claude Code CLI MCP-server startup-timeout env var so that a
    slow, hung, or incompatible stdio MCP is dropped within a bounded time and
    the agent still reaches turn 1 — instead of hanging to the 1200s SIGTERM
    wall (the 0-turn MCP-startup wedge, reify esc-4415-232).

    Uses ``setdefault`` so an explicit caller value wins (e.g. a test or
    per-invocation override that passes ``{'MCP_TIMEOUT': '5000'}``).
    """
    merged: dict[str, str] = dict(env_overrides or {})
    merged.setdefault('MCP_TIMEOUT', MCP_STARTUP_TIMEOUT_MS)
    return merged


# ---------------------------------------------------------------------------
# plan-tools stdio MCP launch contract (single source of truth)
# ---------------------------------------------------------------------------

# Fast-start flags for `uv run` that reuse the already-synced orchestrator
# venv instead of cold-resolving/building under load.  The orchestrator itself
# runs under `uv run --project orchestrator` (so _ORCH_PROJECT_DIR/.venv is
# guaranteed present and synced), and _sync_subprojects (workflow.py) pre-syncs
# task venvs.  Skipping the implicit pre-run sync (--no-sync) and lockfile
# re-resolution (--frozen) eliminates the resolve/build stall that caused the
# 0-turn MCP-startup wedge under load (reify esc-4415-240, esc-4437-123).
# This is the plan_tools analog of the jcodemunch "don't re-resolve/rebuild
# per launch" durable fix.
_PLAN_TOOLS_FAST_START_FLAGS: tuple[str, ...] = ('--no-sync', '--frozen')


# Quiet-env for the stdio MCP servers (plan-tools AND verdict-tools).  FastMCP
# 3.x defaults to printing a decorative startup banner, and rendering that
# banner runs a SYNCHRONOUS PyPI update-check (log_server_banner ->
# check_for_newer_version -> httpx.get('https://pypi.org/pypi/fastmcp/json',
# timeout=2.0)) on the server's startup path, BEFORE it begins serving stdio.
# On a fresh worktree / differing HOME / sandbox the version cache is cold or
# unwritable, so the network call re-fires every launch; under network or
# concurrent-launch pressure that latency can push the MCP `initialize`
# handshake past the CLI's MCP_TIMEOUT, and the CLI then silently DROPS the
# server (the intermittent plan-tools/verdict-tools absence, task 2942).
# Setting both env vars disables the banner and the update-check, so the
# server makes no network call before serving.  Declarative at the launch site
# and FastMCP-version-independent; complemented by show_banner=False in each
# server's main() as an in-code, env-independent guarantee.
_FASTMCP_STDIO_QUIET_ENV: dict[str, str] = {
    'FASTMCP_SHOW_SERVER_BANNER': 'false',
    'FASTMCP_CHECK_FOR_UPDATES': 'off',
}


def _stdio_mcp_server(
    module_path: str,
    orch_project_dir: Path,
    worktree: Path,
    python_executable: str | None = None,
    meta_root: Path | None = None,
    extra_args: tuple[str, ...] = (),
) -> dict:
    """Shared launch-dict factory for stdio MCP servers (task 2482 amendment).

    :func:`plan_tools_mcp_server` and :func:`verdict_tools_mcp_server` both
    delegate here — they differ only by *module_path* and an optional
    *extra_args* tail (verdict-tools' ``('--verdict-role', role)``), so the
    no-uv hot path vs. ``uv --no-sync --frozen`` fallback branch and the
    ``meta_root`` passthrough now live in exactly one place instead of two
    near-identical copies.

    **No-uv hot path (production, task 1776):** when *python_executable* is
    supplied, launches *module_path* directly via that interpreter with no
    ``uv`` involved at all, eliminating the ``uv``-internal futex contention
    that caused 0-turn ``error_empty_output`` failures under concurrent load
    (reify esc-4415-240, esc-4437-123).

    **uv fallback (task 1775):** when *python_executable* is ``None``
    (default), returns the ``uv run --project <orch> --no-sync --frozen
    python -m <module_path> ...`` form using
    :data:`_PLAN_TOOLS_FAST_START_FLAGS` — a stdio fast-start convention
    shared by every server launched this way, not plan-tools-specific.

    Args:
        module_path: Python module to launch via ``-m`` (e.g.
            ``'orchestrator.mcp.plan_tools'``).
        orch_project_dir: Path to the orchestrator package root (used for
            ``--project`` in the uv fallback path).
        worktree: Path to the agent's worktree (bound as ``--worktree``).
        python_executable: Full path to the Python interpreter to use for the
            direct-interpreter no-uv hot path.  When ``None``, falls back to
            the ``uv run --no-sync --frozen`` form.
        meta_root: Optional `.task-meta` root (see
            ``TaskArtifacts.meta_root_for``) passed through as
            ``--meta-root``.  When ``None`` (default), no ``--meta-root``
            flag is emitted and args are byte-identical to before this
            parameter existed.
        extra_args: Additional tokens appended after the ``--worktree``/
            ``--meta-root`` args (e.g. verdict-tools' ``--verdict-role``
            pair).  Empty by default — exactly plan-tools' shape.

    Returns:
        Claude Code MCP server config dict with ``command`` and ``args``.
    """
    meta_root_args = ['--meta-root', str(meta_root)] if meta_root is not None else []
    if python_executable:
        # Direct-interpreter no-uv hot path: eliminates the uv futex wedge.
        return {
            'command': python_executable,
            'args': [
                '-m', module_path, '--worktree', str(worktree),
                *meta_root_args,
                *extra_args,
            ],
            # Suppress FastMCP's banner + synchronous PyPI update-check so the
            # stdio server makes no network call before serving (task 2942).
            # A fresh dict() copy per call — never the shared module constant —
            # so a caller mutating the returned env can't corrupt the next
            # launch's env.
            'env': dict(_FASTMCP_STDIO_QUIET_ENV),
        }
    # uv fallback: backward-compatible form with fast-start flags (task 1775).
    return {
        'command': 'uv',
        'args': [
            'run', '--project', str(orch_project_dir),
            *_PLAN_TOOLS_FAST_START_FLAGS,
            'python', '-m', module_path,
            '--worktree', str(worktree),
            *meta_root_args,
            *extra_args,
        ],
        # Same FastMCP banner/update-check suppression as the hot path above;
        # fresh dict() copy per call (task 2942).
        'env': dict(_FASTMCP_STDIO_QUIET_ENV),
    }


def plan_tools_mcp_server(
    orch_project_dir: Path,
    worktree: Path,
    python_executable: str | None = None,
    meta_root: Path | None = None,
) -> dict:
    """Return a stdio MCP server config dict for the plan-tools server.

    **No-uv hot path (production, task 1776):** When *python_executable* is
    supplied, the config launches plan-tools directly via that interpreter
    without involving ``uv`` at all::

        <python_executable> -m orchestrator.mcp.plan_tools --worktree <wt>

    This eliminates the ``uv``-induced startup stalls that caused 0-turn
    ``error_empty_output`` failures under load: ``uv run`` acquires a
    per-invocation environment lock (futex) regardless of ``--no-sync
    --frozen``; when many agents launch concurrently they contend on that lock
    and the unlucky ones exceed the MCP-initialize handshake window (reify
    esc-4415-240, esc-4437-123, task 1776).  The orchestrator process itself
    runs inside the already-synced orchestrator venv (started via
    ``uv run --project orchestrator``), so ``sys.executable`` is a
    guaranteed-present, correct interpreter with the ``orchestrator`` package
    importable.  The proc tree collapses to ``claude → python3``.

    **uv fallback (backward-compat, tasks 1775/this):** When
    *python_executable* is ``None`` (default), the existing fast-start form is
    returned unchanged::

        uv run --project <orch> --no-sync --frozen python -m orchestrator.mcp.plan_tools --worktree <wt>

    The fallback preserves task 1775's ``TestPlanToolsLaunchFastStart``
    regression tests and serves any caller that cannot supply an interpreter.

    Task 1771's ``MCP_TIMEOUT`` backstop (``apply_mcp_startup_env``) is
    retained as defense-in-depth for both paths.

    Implementation note (task 2482 amendment): delegates to the shared
    ``_stdio_mcp_server`` helper — also used by ``verdict_tools_mcp_server``
    — so this docstring's behavioral contract is unchanged, but the launch
    -dict branch logic itself now lives in one place instead of two
    near-identical copies.

    Args:
        orch_project_dir: Path to the orchestrator package root (used for
            ``--project`` in the uv fallback path).
        worktree: Path to the agent's worktree (bound as ``--worktree``).
        python_executable: Full path to the Python interpreter to use for the
            direct-interpreter no-uv hot path.  When ``None``, falls back to
            the ``uv run --no-sync --frozen`` form.
        meta_root: Optional `.task-meta` root (see
            ``TaskArtifacts.meta_root_for``) to pass through to the
            plan-tools server as ``--meta-root``, so the agent-side plan.json
            writer targets the same relocated artifacts root as the
            orchestrator (worktree-lane-lifecycle PRD task ε1). When
            ``None`` (default), no ``--meta-root`` flag is emitted and args
            are byte-identical to before this parameter existed.

    Returns:
        Claude Code MCP server config dict with ``command`` and ``args``.
    """
    return _stdio_mcp_server(
        'orchestrator.mcp.plan_tools', orch_project_dir, worktree,
        python_executable=python_executable, meta_root=meta_root,
    )


def verdict_tools_mcp_server(
    orch_project_dir: Path,
    worktree: Path,
    role: str,
    python_executable: str | None = None,
    meta_root: Path | None = None,
) -> dict:
    """Return a stdio MCP server config dict for the verdict-tools server.

    Launch target is α's ``orchestrator.mcp.verdict_tools`` CLI (task 2481):
    ``<interpreter> -m orchestrator.mcp.verdict_tools --worktree <wt>
    [--meta-root <mr>] --verdict-role <role>``. *role* is the
    ``--verdict-role`` value — the authoritative selector for both which
    single tool the server registers (judge/merger/reviewer-name, see
    ``verdict_tools.create_server``) and the ``verdicts/<role>.json``
    filename it writes to (never an agent-supplied field).

    Delegates to the shared :func:`_stdio_mcp_server` helper (task 2482
    amendment) — same direct-interpreter no-uv hot path vs. ``uv --no-sync
    --frozen`` fallback branch, same ``meta_root`` passthrough as
    :func:`plan_tools_mcp_server`, differing only by module path and the
    added ``('--verdict-role', role)`` extra-args tail.

    The returned dict carries the shared ``_FASTMCP_STDIO_QUIET_ENV``
    (``FASTMCP_SHOW_SERVER_BANNER='false'`` / ``FASTMCP_CHECK_FOR_UPDATES=
    'off'``) via ``_stdio_mcp_server`` — the same env applied to
    ``plan_tools_mcp_server`` — suppressing FastMCP 3.x's startup banner and
    its synchronous PyPI update-check so the stdio server makes no network
    call before serving (task 2942).  ``MCP_TIMEOUT`` is still injected
    process-wide for every claude subprocess by ``apply_mcp_startup_env``
    (invoke.py) and inherited by all stdio MCP servers it spawns; the CLI
    merges (does not replace) this per-server ``env`` into that inherited
    environment, so both coexist.

    No ``--session-id`` is threaded through here (nor by
    ``_inject_verdict_tools_mcp`` in workflow.py): α made ``--session-id``
    optional (default ``''``), and task 2482's plan (design_decisions #4)
    deliberately defers passing it — PRD open-question 3 accepts an empty
    ``session_id`` provenance field on written verdicts for v1 (eval-revival
    can correlate a verdict by ``emitted_at`` + task id instead). This is an
    intentional scope boundary, not a wiring gap: no code reads
    ``verdicts/<role>.json`` yet, so the blank field has no consumer-visible
    effect today. A future consumer task can add a ``session_id`` parameter
    here and pass it through ``_inject_verdict_tools_mcp`` alongside its
    parse-site change.

    Args:
        orch_project_dir: Path to the orchestrator package root (used for
            ``--project`` in the uv fallback path).
        worktree: Path to the agent's worktree (bound as ``--worktree``).
        role: The ``--verdict-role`` value (e.g. ``'judge'``, ``'merger'``,
            or a reviewer-panel name like ``'reviewer_comprehensive'``).
        python_executable: Full path to the Python interpreter to use for the
            direct-interpreter no-uv hot path.  When ``None``, falls back to
            the ``uv run --no-sync --frozen`` form.
        meta_root: Optional `.task-meta` root (see
            ``TaskArtifacts.meta_root_for``) to pass through as
            ``--meta-root``, so the agent-side verdict-tools server targets
            the same relocated artifacts root as the orchestrator. When
            ``None`` (default), no ``--meta-root`` flag is emitted.

    Returns:
        Claude Code MCP server config dict with ``command`` and ``args``.
    """
    return _stdio_mcp_server(
        'orchestrator.mcp.verdict_tools', orch_project_dir, worktree,
        python_executable=python_executable, meta_root=meta_root,
        extra_args=('--verdict-role', role),
    )


# ---------------------------------------------------------------------------
# Concurrent-startup preflight: verify plan-tools can cold-start under load
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanToolsProbeResult:
    """Outcome of a single plan-tools startup probe.

    Attributes:
        ok: True iff the probe completed the MCP initialize handshake.
        server_name: Server name returned by ``initialize``, or None on failure.
        elapsed_s: Wall time for the probe (seconds), or None if not measured.
        error: ``repr(exc)`` for a failed probe, or None on success.
    """

    ok: bool
    server_name: str | None = None
    elapsed_s: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class PlanToolsStartupReport:
    """Aggregate report from ``verify_plan_tools_startup``.

    Attributes:
        probes: Per-probe results, one per concurrency slot.
        all_succeeded: True iff probes is non-empty and every probe has ok=True.
    """

    probes: list[PlanToolsProbeResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return bool(self.probes) and all(p.ok for p in self.probes)


async def verify_plan_tools_startup(
    orch_project_dir: Path,
    worktree: Path,
    *,
    python_executable: str | None = None,
    concurrency: int = 1,
    deadline_s: float = 30.0,
) -> PlanToolsStartupReport:
    """Launch *concurrency* plan-tools servers concurrently and probe each via MCP initialize.

    Drives K real plan-tools servers through an actual MCP ``initialize``
    handshake concurrently, under a bounded per-probe anyio deadline.  This is
    the executable form of the acceptance criterion: "plan_tools cannot hang the
    agent pre-turn-1 under realistic load."

    Each probe is independent — one timed-out or failed probe does not abort
    siblings.  The function always returns a complete :class:`PlanToolsStartupReport`.

    Args:
        orch_project_dir: Path to the orchestrator project root (passed to
            :func:`plan_tools_mcp_server`).
        worktree: Path to the agent worktree (must have a ``.task/`` subdirectory;
            passed to :func:`plan_tools_mcp_server`).
        python_executable: Interpreter for the no-uv hot path.  When ``None``,
            defaults to ``sys.executable`` — the production hot path.
        concurrency: Number of servers to launch concurrently.
        deadline_s: Per-probe anyio deadline in seconds (graceful timeout via
            :func:`anyio.fail_after`; tears down the subprocess on expiry).

    Returns:
        :class:`PlanToolsStartupReport` with one :class:`PlanToolsProbeResult`
        per concurrency slot and an ``all_succeeded`` summary.
    """
    # Lazy import — keeps module-load cost off the orchestrator hot path.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    interp = python_executable or sys.executable
    cfg = plan_tools_mcp_server(orch_project_dir, worktree, python_executable=interp)
    # Pass the full process environment and an explicit cwd so the preflight
    # mirrors the production launch faithfully (e.g. OPENAI_API_KEY visible,
    # PATH unchanged, cwd is the orchestrator project root).  Without env=,
    # mcp's StdioServerParameters defaults to get_default_environment() — a
    # reduced PATH with no project-specific vars — which could produce probe
    # results that diverge from production behaviour.
    params = StdioServerParameters(
        command=cfg['command'],
        args=cfg['args'],
        env=dict(os.environ),
        cwd=str(orch_project_dir),
    )

    async def _probe() -> PlanToolsProbeResult:
        t0 = time.monotonic()
        try:
            with anyio.fail_after(deadline_s):
                async with stdio_client(params) as (r, w):
                    async with ClientSession(r, w) as session:
                        result = await session.initialize()
                        elapsed = time.monotonic() - t0
                        server_name = (
                            result.serverInfo.name
                            if result.serverInfo is not None
                            else None
                        )
                        return PlanToolsProbeResult(
                            ok=True,
                            server_name=server_name,
                            elapsed_s=elapsed,
                            error=None,
                        )
        except Exception as exc:
            # Catch ALL normal failures: TimeoutError (deadline), FileNotFoundError
            # (bogus interpreter path), OSError (spawn failure), and any handshake
            # or protocol exception from stdio_client/ClientSession.
            # BaseException subclasses (CancelledError) are NOT caught so that anyio
            # task cancellation propagates correctly for graceful teardown.
            elapsed = time.monotonic() - t0
            error_msg = repr(exc) if not isinstance(exc, TimeoutError) else 'deadline exceeded'
            return PlanToolsProbeResult(
                ok=False,
                server_name=None,
                elapsed_s=elapsed,
                error=error_msg,
            )

    async def _run_all() -> list[PlanToolsProbeResult]:
        results: list[PlanToolsProbeResult] = []
        async with anyio.create_task_group() as tg:

            async def _slot(slot_results: list[PlanToolsProbeResult]) -> None:
                outcome = await _probe()
                slot_results.append(outcome)

            for _ in range(concurrency):
                tg.start_soon(_slot, results)

        return results

    probes = await _run_all()
    return PlanToolsStartupReport(probes=probes)


def tool_error_text(reply: Mapping[str, Any]) -> str | None:
    """The error prose of a tool-level failure, or None if the tool succeeded.

    An MCP tool reports its own failure in the response body, as a
    well-formed envelope carrying ``isError``, not by breaking the transport.
    Nothing about that envelope's shape says it failed, so a reader that
    checks only the shape would take the error prose for a result.
    """
    if not reply.get('isError'):
        return None
    texts = tool_text_blocks(reply)
    return texts[0] if texts else ''


def tool_text_blocks(reply: Mapping[str, Any]) -> tuple[str, ...]:
    """The text of each ``text`` content block of a tool result, in order.

    An envelope whose ``content`` is missing, null or not a list carries no text.
    """
    content = reply.get('content')
    if not isinstance(content, list):
        return ()
    return tuple(
        str(block.get('text', ''))
        for block in content
        if isinstance(block, dict) and block.get('type') == 'text'
    )


# The orchestrator's one self-identification spelling. clientInfo reaches only
# a stateful session; a stateless-HTTP server sees this identity only when a
# tool call's arguments carry it (Scheduler.set_task_status does).
ORCHESTRATOR_MCP_IDENTITY: str = 'orchestrator'

def fm_session(base_url: str) -> McpSession:
    """The orchestrator's fused-memory session: it identifies as ORCHESTRATOR_MCP_IDENTITY
    and rides fm's restart window, except at the D8 null sentinel (task 2880)."""
    return McpSession(
        base_url,
        client_name=ORCHESTRATOR_MCP_IDENTITY,
        retry_backoffs=() if is_fm_null_sentinel(base_url) else fm_retry_backoffs(),
    )


# Module-level session singleton (created by McpLifecycle after server starts)
_session: McpSession | None = None


def get_session() -> McpSession:
    """Get the current MCP session. Raises if not initialized."""
    if _session is None:
        raise RuntimeError('MCP session not initialized — call McpLifecycle.start() first')
    return _session


async def mcp_call(
    url: str,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 30,
    session_id: str | None = None,
) -> dict:
    """Send a JSON-RPC request to an MCP Streamable HTTP endpoint.

    Uses the module-level session if available, otherwise creates a one-shot session.
    """
    session = _session if _session is not None else fm_session(url.rstrip('/mcp').rstrip('/'))
    if method == 'tools/call' and params:
        return await session.call_tool(
            params['name'], params.get('arguments', {}), timeout=timeout
        )
    return await session.call(method, params, timeout=timeout)


# ---------------------------------------------------------------------------
# Managed-spawn runtime data dirs (task 2439)
# ---------------------------------------------------------------------------


def managed_runtime_data_dirs(project_id: str) -> tuple[Path, Path]:
    """Return (queue_dir, recon_dir) for a managed fused-memory spawn.

    Keeps the managed fused-memory server's runtime SQLite state — the
    write_queue.db (+ -shm/-wal) and reconciliation.db/tickets.db — out of the
    watched project's git-tracked tree. A prior CWD-relative default (the
    fused-memory config default is ``./data/queue`` /
    ``./data/reconciliation``) let this state land inside the project root
    that ``McpLifecycle.start()`` spawns fused-memory in, dirtying the tree
    and poisoning the startup dirty-tree-guard and warm-lane GC (task 2439).

    Rooted under ``${XDG_STATE_HOME:-~/.local/state}/dark-factory/<project_id>/``
    — outside any watched project's tree — and keyed by project_id so
    multiple managed instances never collide.

    Wider blast radius, intentionally: fused-memory derives its
    reconciliation *escalation* queue from the same var —
    ``escalation_queue_dir: "${RECONCILIATION_DATA_DIR:./data/reconciliation}/escalations"``
    (fused-memory/config/config.yaml) — so ``recon_dir`` here also relocates
    that escalation queue out of ``<project>/data/reconciliation/escalations``
    and into the XDG-rooted path. This is deliberate (task 2439's scope
    treats reconciliation.db/tickets.db/escalations as one pollution class,
    not just the queue DB); any tooling that globs the old project-relative
    escalations path needs to point at the new XDG-rooted one instead.

    Lockstep consumer: ``dashboard/src/dashboard/config.py``'s
    ``reconciliation_db`` / ``tickets_db`` / ``write_journal_db`` /
    ``write_queue_db`` / ``reconciliation_escalations_dir`` properties read
    this same runtime state and honor the same ``QUEUE_DATA_DIR`` /
    ``RECONCILIATION_DATA_DIR`` env vars (falling back to
    ``project_root/data/...`` when unset) — task 2439. If this function's
    rooting scheme ever changes, that dashboard consumer must move in
    lockstep, or its queue / reconciliation / tickets / journal /
    recon-escalation panels will silently go blank or stale reading a path
    the managed spawn no longer writes to.

    Degrades instead of crashing when ``$HOME`` can't be resolved: with
    ``XDG_STATE_HOME`` unset, ``Path.home()`` raises ``RuntimeError`` in a
    stripped daemon/CI environment with no ``HOME`` (and no ``pwd`` entry).
    Since this helper is called unconditionally on ``McpLifecycle.start()``'s
    managed-spawn path, an unguarded raise here would take down orchestrator
    startup; instead this falls back to the OS temp dir (task 2439
    amendment).
    """
    xdg_state_home = os.environ.get('XDG_STATE_HOME')
    if xdg_state_home:
        base = Path(xdg_state_home)
    else:
        try:
            base = Path.home() / '.local' / 'state'
        except RuntimeError:
            logger.warning(
                'managed_runtime_data_dirs: could not resolve a home '
                'directory (HOME unset?); falling back to the OS temp dir '
                'for managed fused-memory runtime state (task 2439)'
            )
            base = Path(tempfile.gettempdir())
    root = base / 'dark-factory' / project_id
    return root / 'queue', root / 'reconciliation'


class McpLifecycle:
    """Manages the fused-memory MCP HTTP server process."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config.fused_memory
        self.project_root = config.project_root
        self._process: asyncio.subprocess.Process | None = None
        # pgid captured at spawn (pgid == pid under start_new_session); kept
        # alongside _process so stop() never has to call os.getpgid, which
        # would be unsafe if the PID has been reused post-reap.
        self._pgid: int | None = None

    @property
    def url(self) -> str:
        return self.config.url

    async def start(self) -> None:
        """Start fused-memory with HTTP transport, wait for health."""
        global _session

        if self._process is not None:
            logger.warning('MCP server already running')
            return

        if not self.config.server_command:
            # External server mode — skip subprocess, just connect
            logger.info(f'Connecting to external fused-memory at {self.config.url}')
            if not await self._wait_for_health(timeout=30):
                raise RuntimeError(
                    f'External fused-memory at {self.config.url} not reachable within 30s'
                )
            logger.info(f'Fused-memory HTTP server ready at {self.config.url}')
        else:
            cmd = self.config.server_command
            config_path = (self.project_root / self.config.config_path).resolve()
            cmd = [*cmd, '--config', str(config_path)]

            # Redirect the managed fused-memory's runtime queue/reconciliation
            # SQLite state out of this (watched) project's tracked tree — task
            # 2439. env.setdefault() honors an operator-preset override.
            # RECONCILIATION_DATA_DIR also relocates the recon *escalation*
            # queue (see managed_runtime_data_dirs() docstring) — intentional.
            q_dir, r_dir = managed_runtime_data_dirs(self.config.project_id)
            env = dict(os.environ)
            env.setdefault('QUEUE_DATA_DIR', str(q_dir))
            env.setdefault('RECONCILIATION_DATA_DIR', str(r_dir))

            logger.info(f'Starting fused-memory HTTP server: {" ".join(cmd)}')
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self._pgid = self._process.pid

            # Wait for health
            if not await self._wait_for_health(timeout=30):
                await self.stop()
                raise RuntimeError('Fused-memory server failed to start within 30s')

            logger.info(f'Fused-memory HTTP server ready at {self.config.url}')

        # Initialize MCP session
        _session = fm_session(self.config.url)
        await _session.initialize()

    async def stop(self) -> None:
        """Graceful shutdown."""
        global _session

        if self._process is None:
            return

        logger.info('Stopping fused-memory HTTP server')
        _session = None
        try:
            # If start() crashed between subprocess spawn and self._pgid assignment,
            # fall back to self._process.pid — start_new_session=True guarantees
            # pgid == pid while the proc is unreaped, and terminate_process_group's
            # _unsafe_pgid_reason sanity-check refuses to signal if the invariant is
            # ever violated.
            pgid = self._pgid if self._pgid is not None else self._process.pid
            await terminate_process_group(self._process, pgid, grace_secs=10.0)
        except Exception as exc:
            logger.warning('Failed to terminate fused-memory server: %s', exc)
        finally:
            self._process = None
            self._pgid = None

    async def health_check(self) -> bool:
        """Check if the server is responding."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f'{self.config.url}/health', timeout=5)
                return resp.status_code == 200
        except Exception:
            return False

    async def _wait_for_health(self, timeout: float = 30) -> bool:
        """Poll health endpoint until ready or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                stderr = await self._process.stderr.read() if self._process.stderr else b''
                logger.error(f'Server exited prematurely: {stderr.decode()[-500:]}')
                return False
            if await self.health_check():
                return True
            await asyncio.sleep(0.5)
        return False

    def mcp_config_json(
        self,
        escalation_url: str | None = None,
        escalation_headers: dict | None = None,
    ) -> dict:
        """Return MCP server config dict suitable for --mcp-config.

        escalation_headers, when provided ALONGSIDE escalation_url, is copied
        into the escalation server block as 'headers' (e.g. the connection
        -capability X-Escalation-Levels / X-Escalation-Identity headers the
        supervised watcher rotation attaches — see
        plans/escalation-connection-capability-guard-prd.md task beta). It is
        a no-op when escalation_url is absent, and the default (None) leaves
        the escalation block — and this method's output overall — byte-for
        -byte unchanged, so every other caller (steward, review-checkpoint,
        workflow, dry-run-unblock) keeps full-authority, header-less
        connections.
        """
        config = {
            'mcpServers': {
                'fused-memory': {
                    'type': 'http',
                    'url': f'{self.config.url}/mcp',
                },
                'jcodemunch': jcodemunch_server_config(),
            },
        }
        if escalation_url:
            escalation_block: dict = {
                'type': 'http',
                'url': escalation_url,
            }
            if escalation_headers:
                escalation_block['headers'] = dict(escalation_headers)
            config['mcpServers']['escalation'] = escalation_block
        return config
