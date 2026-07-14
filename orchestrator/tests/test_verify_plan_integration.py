"""Integration-gate suite ι: B+H boundary tests over the verify decision layer.

This file is the ι B+H integration-gate LEAF for the verify-plan PRD
(plans/verify-plan-prd.md; Boundary-test sketch rows 1-12; capability
manifest block ι).  It drives the REAL, already-landed verify decision layer
(α-θ) end-to-end across every seam, facing BOTH producer and runner sides of
each contract.

α-θ are all merged on main and are OUT OF SCOPE for this task — this module
contains NO production code changes:
  α verify_categories.py     — FailureCategory + CATEGORY_POLICY exhaustiveness
  β verify_cmd.py            — VerifyCmd / parse_config_command / render / mutators
  γ verify_plan.py           — derive_verify_plan (plan goldens)
  δ verify_classify.py       — classify_failure (tool-isolation)
  ε verify.py                — CheckRun / VerifyAttempt (timeout consistency)
  ζ unblock_types.py         — BlockRecord / BlockClass
  η merge_queue.py           — block-path spawn -> dry-run proposal
  θ git_ops.py               — ephemeral_worktree (no-prune probes)

REAL vs FAKED
-------------
REAL (composed as the genuine article, never hand-seeded): GitOps over a
real git repo; ``derive_verify_plan``, ``classify_failure``, ``check_proposal``,
``_run_post_merge_verify``, ``VerifyCmd`` + its mutators, ``VerifyAttempt``,
and ``ephemeral_worktree`` itself.

FAKED (boundary only — the ssh/build/agent edges, never the decision layer):
``run_scoped_verification``, ``run_full_verification``,
``orchestrator.dry_run_unblock.invoke_agent``, and ``git_ops._run`` where a
subprocess argv-spy is needed (scenario 12 only — everywhere else git
subprocesses run for real against the fixture repo).

Each scenario class below is RED until its paired wiring step imports the
exercised symbols and ports the needed test-local helpers (see each class's
own docstring); "GREEN" means the real-object driver correctly exercises the
already-landed code, not that new production logic was written anywhere. A
scenario that stays RED for a genuine composition reason is a design_concern
escalation, not a patch to α-θ (see plan.json design_decisions).

§ Scenario index (Boundary-test sketch rows 1-12, capability manifest §ι)
--------------------------------------------------------------------------
  1.  VerifyCmd render round-trip + producer<->runner scoped-pytest drive (P2).
  2.  OPAQUE never scoped (P1).
  3.  Plan golden — root conftest -> FULL_SUITE (D1, task-1077).
  4.  Plan golden — lone data module -> SKIPPED-with-reason (task-1852).
  5.  Plan golden — structural file -> unscoped pyright, module + fallback (D2).
  6.  Classifier tool-isolation (C1).
  7.  Category exhaustiveness (F1).
  8.  CheckRun/VerifyAttempt timeout consistency (the verify.py:2735-2744 drift).
  9.  Merge-verify block -> gateable proposal (the coverage gap + B4).
  10. POST_MERGE_RED_MAIN preserved (B2, task-1680).
  11. Legacy proposal bridge (B3) + BlockRecord round-trip (B1).
  12. ephemeral_worktree no-prune across both probes (E1/E2).
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator import (
    b3_gate,
    merge_queue,
    unblock_types,
    verify,
    verify_categories,
    verify_classify,
    verify_cmd,
    verify_plan,
)
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest

# ── Repo seeding (ported from test_merge_queue_two_layer_integration.py) ──────


async def _setup_repo(repo: Path) -> None:
    """Initialise a minimal git repo with a README committed on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


# ── Fixtures (ported from test_merge_queue_two_layer_integration.py) ─────────


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ── MergeRequest builder (ported from test_merge_queue_two_layer_integration.py) ──


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    *,
    module_configs: list[ModuleConfig] | None = None,
    task_files: list[str] | None = None,
    merge_first_enqueued_at: float | None = 1000.0,
    request_id: str | None = None,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    The optional *request_id* kwarg lets a test pin a stable identity; when
    omitted a fresh UUID is auto-generated (MergeRequest's own default).
    """
    kwargs: dict = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=task_files,
        module_configs=module_configs or [],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        merge_first_enqueued_at=merge_first_enqueued_at,
        **kwargs,
    )
