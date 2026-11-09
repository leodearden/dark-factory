"""Builders shared by the warm-lane durable-record tests.

A real git repo with stub warm-lane scripts, and a Harness over a REAL GitOps,
for tests that drive the writer (``GitOps.acquire_warm_lane``) into the reader
(``Harness._recover_crashed_tasks``); and a pure-FS GitOps for tests that never
shell out to git.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from _git_fixtures import seed_repo

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, WorktreeInfo, _run
from orchestrator.harness import Harness


def _git_config(**overrides) -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        **overrides,
    )


def pure_fs_git_ops(tmp_path: Path, *, pool_size: int = 0) -> GitOps:
    """A GitOps over a project root that is not a git repo, with a warm-lane
    pool of *pool_size* lanes when it is positive."""
    project_root = tmp_path / 'project'
    project_root.mkdir(exist_ok=True)
    return GitOps(
        _git_config(warm_lane_pool=pool_size > 0), project_root,
        warm_lane_pool_size=pool_size,
    )


def make_warm_lane_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit + a resolvable warm base.

    Pre-creates the default derived warm-lane base (task 2061 gate) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK,
    and marks `.worktrees/.pool-root` present (task 2099 create-once guard) —
    mirrors test_lane_lifecycle_gitops.py's git_repo / test_warm_lane_
    integration_gate.py's ig_git_repo fixtures.
    """
    repo = seed_repo(tmp_path / 'repo')
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


async def add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo.

    No real ssh/reify — the seed stub only needs to mkdir target/ and drop a
    marker file so acquire_warm_lane's seed step exits 0.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)


def make_orch_config(repo: Path, *, max_concurrent_tasks: int = 1) -> OrchestratorConfig:
    """Build a minimal real OrchestratorConfig with the warm-lane pool on."""
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=_git_config(warm_lane_pool=True),
    )


def build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    GitOps is deliberately left REAL (not patched), so callers drive the
    real writer/reader seam. Mirrors test_warm_lane_integration_gate.py /
    test_harness_warm_lane_wiring.py's _build_harness.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


async def get_head(repo: Path) -> str:
    """Return the HEAD commit SHA of the repo."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


def make_plan(done: int, total: int, task_id: str = 'test') -> dict:
    """Build a plan dict with the given step-completion counts."""
    steps = []
    for i in range(total):
        steps.append({
            'id': f'step-{i + 1}',
            'description': f'Step {i + 1}',
            'status': 'done' if i < done else 'pending',
            'commit': f'abc{i}' if i < done else None,
        })
    return {'task_id': task_id, 'title': 'Test Task', 'steps': steps}


async def acquire_lane(
    git_ops: GitOps,
    task_id: str,
    start_ref: str,
    *,
    expected_title: str | None = None,
) -> Path:
    """Acquire a warm lane for *task_id*, asserting success, and return its path."""
    wt = await git_ops.acquire_warm_lane(task_id, start_ref, expected_title=expected_title)
    assert isinstance(wt, WorktreeInfo), f'acquire_warm_lane failed: {wt!r}'
    return wt.path


def wire_recovery_scheduler(harness: Harness) -> None:
    """Stub the scheduler + event_store surface _recover_crashed_tasks reads.

    get_task returns a title-less (but non-None) dict so the identity guard
    (config.worktree_identity_guard_enabled defaults True) fails OPEN
    (identities_match treats an empty title as a match) instead of deferring.
    """
    harness.scheduler.get_status = AsyncMock(return_value=None)
    harness.scheduler.get_task = AsyncMock(return_value={})
    harness.scheduler.get_tasks = AsyncMock(return_value=[])
    harness.scheduler.is_deterministic = MagicMock(return_value=False)
    harness.event_store = MagicMock()
