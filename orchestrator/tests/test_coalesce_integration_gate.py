"""Integration gate (ζ/1722): end-to-end retro-coalescing boundary scenarios.

Exercises the full α–ε stack (tasks 1717-1721) together through the REAL
SpeculativeMergeWorker + asyncio.Queue driven by the REAL
build_train_callback_factory(FakeScheduler).  Fakes are restricted to:
  • git edge  — real local git-repo fixture
  • scheduler edge — FakeScheduler (no HTTP calls)

Four boundary scenarios + one speculative-cap bookkeeping assertion:
  Scenario 1: 3 disjoint stackable singles coalesce, MCP member merge_status-observable
  Scenario 2: partial stackability — overlap keeps 3rd as solo
  Scenario 3: confidence-gate exclusion visible in train_coalesced event
  Scenario 4: in-flight + detached-waiter exclusion invariants
  Bookkeeping: depth-1/K cap accounting stays consistent across a coalesce
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from _workflow_helpers import FakeScheduler
from test_merge_queue_coalesce import (
    _events_of_type,
    _gated_verify,
    _make_branch_with_file,
    _make_req,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig
    from orchestrator.git_ops import GitOps


# ─── Fixtures ────────────────────────────────────────────────────────────────


async def _setup_repo(repo: Path) -> None:
    from orchestrator.git_ops import _run
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config():
    from orchestrator.config import GitConfig
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config, git_repo: Path):
    from orchestrator.git_ops import GitOps
    return GitOps(git_config, git_repo)


@pytest.fixture
def coalesce_config(git_repo: Path, git_config) -> OrchestratorConfig:
    """Config with merge_train_coalesce_enabled=True."""
    from orchestrator.config import OrchestratorConfig
    return OrchestratorConfig(
        project_root=git_repo,
        git=git_config,
        merge_train_coalesce_enabled=True,
    )


# ─── Scenario tests (bodies added in steps 1-10) ────────────────────────────
