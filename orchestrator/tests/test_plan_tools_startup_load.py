"""Concurrent startup load guard for orchestrator.mcp.plan_tools.

Acceptance criterion (executable form): plan_tools cannot hang the agent
pre-turn-1 under realistic concurrent load.  This module drives K real
plan-tools servers through an actual MCP ``initialize`` handshake concurrently
under a bounded per-probe anyio deadline, proving:

- Clean, uncorrupted stdout (the JSON-RPC parse would fail otherwise)
- A live server reachable via the mcp stdio client
- The no-uv hot path (sys.executable) is what the production launch config
  uses — reintroducing uv would trip the per-probe deadline under contention

Task 1776's static tests (test_mcp_lifecycle.py) assert config-dict shape for
plan_tools_mcp_server.  This module asserts RUNTIME behaviour (live initialize
handshakes) and cross-checks the same no-uv property end-to-end.

Note: ``test_concurrent_startup_no_hang`` is heavyweight (6 real subprocesses).
It is marked ``@pytest.mark.slow`` so it can be deselected with ``-m "not slow"``
in fast unit runs.  The ``slow`` marker must be registered in pyproject.toml for
strict-marker mode; omit ``--strict-markers`` or add the marker to suppress the
PytestUnknownMarkWarning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import orchestrator.mcp_lifecycle as _lcmod
from orchestrator.mcp_lifecycle import plan_tools_mcp_server, verify_plan_tools_startup

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orch_project_dir() -> Path:
    """Orchestrator project root (directory containing pyproject.toml).

    Derived from mcp_lifecycle's __file__ path so it works regardless of the
    current working directory or how the test is invoked.

    mcp_lifecycle.__file__ = .../orchestrator/src/orchestrator/mcp_lifecycle.py
    parents[0] = .../orchestrator/src/orchestrator/
    parents[1] = .../orchestrator/src/
    parents[2] = .../orchestrator/  ← project root
    """
    return Path(_lcmod.__file__).resolve().parents[2]


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    """Minimal worktree with a bare .task/ directory.

    plan_tools.main() calls sys.exit(1) if (worktree / '.task').is_dir() is
    False.  TaskArtifacts.__init__ is pure path assignment — no plan.json is
    read at startup (tools read it lazily per-call) — so a bare .task/ is
    sufficient for the server to reach the stdio initialize loop.
    """
    wt = tmp_path / 'wt'
    (wt / '.task').mkdir(parents=True)
    return wt


# ---------------------------------------------------------------------------
# Happy-path: K concurrent servers each complete a real MCP initialize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)  # overrides global timeout=60; per-probe anyio deadline is primary
@pytest.mark.slow  # heavyweight: 6 real subprocesses; deselect with -m "not slow"
async def test_concurrent_startup_no_hang(orch_project_dir: Path, worktree: Path) -> None:
    """6 concurrent plan-tools servers each complete MCP initialize < 30s each.

    Proves the acceptance criterion: plan_tools cannot hang the agent pre-turn-1
    under realistic load with the production no-uv (sys.executable) hot path.

    Generous deadlines (30s per probe, 120s backstop) detect hangs/gross
    regressions, not micro-latency — a warm host completes in single-digit
    seconds (measured initialize ≈ 1.65s).
    """
    report = await verify_plan_tools_startup(
        orch_project_dir,
        worktree,
        python_executable=sys.executable,
        concurrency=6,
        deadline_s=30,
    )

    # All 6 probes must succeed — any failure means a hang, corruption, or crash.
    assert report.all_succeeded is True, (
        f'Expected all probes to succeed; report: {report.probes!r}'
    )

    # Each probe must have completed the initialize handshake.
    assert len(report.probes) == 6, (
        f'Expected 6 probe outcomes, got {len(report.probes)}'
    )
    for i, probe in enumerate(report.probes):
        assert probe.ok is True, (
            f'Probe {i} failed: ok={probe.ok!r} error={probe.error!r}'
        )
        assert probe.server_name is not None, (
            f'Probe {i}: server_name must be non-None after initialize; got {probe.server_name!r}'
        )
        assert probe.elapsed_s is not None and probe.elapsed_s >= 0, (
            f'Probe {i}: elapsed_s must be recorded; got {probe.elapsed_s!r}'
        )
        assert probe.error is None, (
            f'Probe {i}: no error expected on success; got {probe.error!r}'
        )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_production_launch_config_uses_no_uv(
    orch_project_dir: Path, worktree: Path
) -> None:
    """The canonical launch config produced by plan_tools_mcp_server uses sys.executable.

    Proves the no-uv property holds for the ACTUAL runtime config — not just in
    isolation.  If uv is reintroduced into the startup path, this assertion
    fails before the concurrent load test even runs.
    """
    cfg = plan_tools_mcp_server(
        orch_project_dir, worktree, python_executable=sys.executable
    )
    assert cfg['command'] == sys.executable, (
        f"Expected command == sys.executable ({sys.executable!r}); "
        f"got {cfg['command']!r} — uv or another wrapper was reintroduced"
    )


# ---------------------------------------------------------------------------
# Negative / robustness: bogus interpreter is reported as failure, no raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_nonexistent_interpreter_returns_failed_report(
    orch_project_dir: Path, worktree: Path
) -> None:
    """A bogus interpreter path yields all_succeeded=False without raising or hanging.

    Proves the preflight reports launch failures gracefully — returning a report
    with ok=False probes rather than propagating the exception or wedging.  A
    nonexistent interpreter raises FileNotFoundError (ENOENT) from stdio_client's
    subprocess spawn; the broad per-probe ``except Exception`` handler in
    ``verify_plan_tools_startup`` records it as a failed probe outcome
    (ok=False, error populated) without propagating or aborting sibling probes.
    """
    report = await verify_plan_tools_startup(
        orch_project_dir,
        worktree,
        python_executable='/nonexistent/python',
        concurrency=2,
        deadline_s=10,
    )

    assert report.all_succeeded is False, (
        f'Expected all_succeeded=False for bogus interpreter; got {report.all_succeeded!r}'
    )
    assert len(report.probes) == 2, (
        f'Expected 2 probe outcomes recorded (not raised); got {len(report.probes)}'
    )
    for i, probe in enumerate(report.probes):
        assert probe.ok is False, (
            f'Probe {i}: expected ok=False for failed launch; got {probe.ok!r}'
        )
        assert probe.error is not None, (
            f'Probe {i}: error must be populated for a failed probe; got {probe.error!r}'
        )
