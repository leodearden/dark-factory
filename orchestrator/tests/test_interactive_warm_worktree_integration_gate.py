"""ζ B+H integration gate: interactive warm-worktree boundary test (task 2015).

Ropes task α (``GitOps.create_interactive_worktree`` primitive, 2010), task β
(``claim_warm_worktree`` / ``release_warm_worktree`` escalation MCP verbs,
2011), and task δ (the harness-side interactive-worktree reaper cadence,
2012 — merged into main) into ONE end-to-end boundary test that proves the
interactive warm-worktree contract holds across the full B+H (escalation
verb + harness) surface — a composition none of the three tasks' own tests
exercises:

  - α's test_interactive_worktree.py::TestCreateInteractiveWorktreeIsolation
    already proves isolation invariant I1 at the raw git-primitive level
    (direct ``create_interactive_worktree`` + ``git worktree remove``).
  - β's escalation/tests/test_warm_worktree_verbs.py already covers
    claim/release verbs — but against a ``SimpleNamespace(git_ops=...)``
    fake harness with NO real ``WarmLanePool``.
  - δ's test_harness_interactive_reaper.py covers the reaper pass on a bare
    ``Harness`` with no pool involved.

This module drives claim/release THROUGH the β escalation verbs
(``create_server(...)`` + the FastMCP ``_call_tool`` unit-invocation
pattern) against a harness that ALSO carries a real ``WarmLanePool`` (K FREE
lanes) AND a seeded ``scheduler._dispatched`` set, proving I1 holds across
the entire verb surface (the "B+H integration gate"), then composes the δ
reaper (I2) into the same flow and re-asserts the pool is still untouched.

Like the sibling gates (test_warm_lane_integration_gate.py,
test_config_reload_integration_gate.py) this gate is EXPECTED GREEN on the
existing α/β/δ production code: the new artifact here is the test itself.
Impl steps touch production (git_ops.py / harness.py / escalation
server.py) ONLY if end-to-end composition surfaces a genuine defect.

G6 / PRD Open-Q#3 scope note: the "warm target/ → near-zero recompile"
observable is a reify-side (cargo ``target/``, filefrag/CoW extent-sharing)
guarantee that is NOT producible in dark-factory's Python CI (no cargo
build cache). That observable is EXPLICITLY OUT of scope in this module and
is deferred to the reify deploy capstone's out-of-band verification. This
module asserts only the CI-producible orchestration-observable boundary
invariants: I1 isolation across the verb surface, the claim/release
roundtrip, seed-invocation via ``target/seeded.bin`` existence, fail-soft
cold claims, reap I2 composed with I1, and the cold (no-harness) fallback.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.warm_lane_pool import LaneState

# ---------------------------------------------------------------------------
# Repo fixture helpers — copied/adapted per codebase convention (NOT
# cross-imported from sibling test files; see test_interactive_worktree.py,
# escalation/tests/test_warm_worktree_verbs.py,
# test_harness_interactive_reaper.py, test_warm_lane_integration_gate.py).
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    """git init -b main + one initial commit."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _add_seed_script(repo: Path, *, exit_code: int = 0) -> None:
    """Commit a seed-warm-lane.sh stub into repo/scripts/.

    On exit_code == 0: creates ``<lane>/target/seeded.bin`` (orchestration-
    observable seededness). On non-zero: exits with that code immediately
    (no target/ created) — models a faulting seed for the fail-soft case.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed = scripts_dir / 'seed-warm-lane.sh'
    if exit_code == 0:
        seed.write_text(
            '#!/usr/bin/env bash\n'
            'mkdir -p "$2/target"\n'
            'echo "seeded" > "$2/target/seeded.bin"\n'
        )
    else:
        seed.write_text(
            f'#!/usr/bin/env bash\necho "seed failure" >&2\nexit {exit_code}\n'
        )
    seed.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add seed-warm-lane.sh stub'], cwd=repo)


def _backdate_stamp(path: Path, created_at: datetime) -> None:
    """Rewrite the ``.task/interactive.json`` stamp's ``created_at`` field."""
    stamp_path = path / '.task' / 'interactive.json'
    stamp = json.loads(stamp_path.read_text())
    stamp['created_at'] = created_at.isoformat()
    stamp_path.write_text(json.dumps(stamp))


async def _registered_worktree_paths(repo: Path) -> set[str]:
    """Return the set of registered worktree paths (resolved) via `git worktree list`."""
    rc, out, _ = await _run(['git', 'worktree', 'list', '--porcelain'], cwd=repo)
    assert rc == 0, 'git worktree list --porcelain failed'
    paths = set()
    for line in out.splitlines():
        if line.startswith('worktree '):
            paths.add(str(Path(line[len('worktree '):].strip()).resolve()))
    return paths


# ---------------------------------------------------------------------------
# Harness + server builder — mirrors _build_harness/_make_orch_config from
# test_warm_lane_integration_gate.py / test_harness_warm_lane_wiring.py, plus
# the create_server(...)/_call_tool pattern from
# escalation/tests/test_warm_worktree_verbs.py.
# ---------------------------------------------------------------------------


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors _build_harness from test_harness_warm_lane_wiring.py /
    test_warm_lane_integration_gate.py. ``harness.git_ops`` is a REAL GitOps
    (not patched), so its ``warm_lane_pool`` is a real WarmLanePool and
    ``harness._run_interactive_worktree_reaper_pass`` is a real bound method.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


def _make_config(
    repo: Path, *, max_concurrent_tasks: int = 3,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig with a real warm-lane pool enabled.

    Mirrors _make_orch_config from test_warm_lane_integration_gate.py. Pool
    size == max_concurrent_tasks exactly (GitConfig.spare_warm_lanes
    defaults to 0), giving K FREE lanes for the I1 dispatch-capacity
    snapshot.
    """
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=GitConfig(warm_lane_pool=True),
    )


def _build_harness_and_server(
    config: OrchestratorConfig, tmp_path: Path,
) -> tuple[Harness, Any]:
    """Build a (harness, server) pair sharing one real GitOps/WarmLanePool.

    ``harness.scheduler`` is a MagicMock (Scheduler is patched out in
    ``_build_harness``); a real sentinel set is assigned to
    ``harness.scheduler._dispatched`` so the I1 dispatch-capacity assertion
    is a byte-for-byte set-equality snapshot rather than resting on a
    MagicMock auto-attribute. ``server`` is wired with ``harness=harness``
    so the β claim/release verbs operate on this exact GitOps/pool instance.
    """
    harness = _build_harness(config)
    harness.scheduler._dispatched = {'sentinel-a', 'sentinel-b'}
    server = create_server(EscalationQueue(tmp_path / 'esc'), harness=harness)
    return harness, server


async def _call_tool(server: Any, name: str, **kwargs: Any) -> Any:
    """Invoke an async MCP tool by name (mirrors test_server.py's _call_tool)."""
    tool = await server.get_tool(name)
    return await tool.fn(**kwargs)
