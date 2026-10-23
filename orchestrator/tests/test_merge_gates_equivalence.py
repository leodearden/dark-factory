"""The post-merge equivalence gate's verdicts, driven over real git (task 6413).

Pins what the gate tells a human when a landing diverges from the branch: the
blocked reason, its rename-suppression proof and its γ2 auto-chain hook.  Every
test drives ``_finalize_advanced_merge`` over a real repository, a real
:class:`GitOps` and a real task worktree.  Faults are injected only through a
``git`` shim placed first on PATH.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import lane_scene_config
from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.gates import (
    POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
    _finalize_advanced_merge,
)
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest, QueuedBranch

TASK_ID = '6413'
TASK_BRANCH = f'task/{TASK_ID}'
GATES_LOGGER = 'orchestrator.merge_lane'


class Scene(NamedTuple):
    """A landing on main next to the task worktree it came from."""

    worktree: Path
    base: str
    landed: str
    branch_tip: str


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, repo: Path) -> GitOps:
    return GitOps(git_config, repo)


@pytest.fixture
def lane_config(repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return lane_scene_config(repo, git_config)


def _git(cwd: Path, *args: str) -> str:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    completed = subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _commit(cwd: Path, files: dict[str, str | bytes], message: str) -> str:
    for relpath, content in files.items():
        target = cwd / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)
    _git(cwd, 'add', '-A')
    _git(cwd, 'commit', '-m', message)
    return _git(cwd, 'rev-parse', 'HEAD')


def _task_worktree(repo: Path, tmp_path: Path) -> Path:
    worktree = tmp_path / 'wt'
    _git(repo, 'worktree', 'add', '-b', TASK_BRANCH, str(worktree), 'main')
    return worktree


def _request(config: OrchestratorConfig, worktree: Path) -> MergeRequest:
    return MergeRequest(
        task_id=TASK_ID,
        branch=QueuedBranch.parse(TASK_BRANCH, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


async def _finalize(
    git_ops: GitOps,
    req: MergeRequest,
    *,
    landed: str,
    base: str,
    event_store: EventStore | None = None,
    started_monotonic: float | None = None,
    **kwargs,
) -> MergeOutcome:
    return await _finalize_advanced_merge(
        git_ops, req, event_store,
        merge_commit_fallback=landed,
        advanced_sha=landed,
        base_sha=base,
        started_monotonic=started_monotonic,
        cas_retries={},
        timeouts={},
        enospc_retries={},
        **kwargs,
    )


def _fail_one_git_call(monkeypatch: pytest.MonkeyPatch, bin_dir: Path, argv: str) -> None:
    """Put a ``git`` first on PATH that fails exactly *argv* and runs every other call."""
    real_git = shutil.which('git')
    assert real_git is not None
    bin_dir.mkdir(parents=True)
    shim = bin_dir / 'git'
    shim.write_text(
        '#!/bin/sh\n'
        f'if [ "$*" = {shlex.quote(argv)} ]; then\n'
        "  echo 'fatal: injected failure (test fault injection)' >&2\n"
        '  exit 128\n'
        'fi\n'
        f'exec {shlex.quote(real_git)} "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')


def _blocked_reason(outcome: MergeOutcome) -> str:
    assert outcome.status == 'blocked', outcome
    assert outcome.reason is not None
    return outcome.reason


def _landing_that_drops_the_branch(repo: Path, tmp_path: Path, files: dict[str, str]) -> Scene:
    """Main records a ``--no-ff`` merge of the branch but keeps only its own tree.

    The merge's second parent is the branch tip, yet ``-s ours`` discards every
    branch file, so each of them diverges between the branch and main.
    """
    base = _git(repo, 'rev-parse', 'HEAD')
    worktree = _task_worktree(repo, tmp_path)
    branch_tip = _commit(worktree, files, f'impl({TASK_ID}): branch work')
    _git(repo, 'merge', '--no-ff', '-s', 'ours', '-m', f'Merge {TASK_BRANCH} into main', TASK_BRANCH)
    landed = _git(repo, 'rev-parse', 'HEAD')
    return Scene(worktree=worktree, base=base, landed=landed, branch_tip=branch_tip)


@pytest.mark.asyncio
async def test_a_failed_second_parent_probe_compares_the_task_worktree_head(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real merge whose ``^2`` probe failed falls back to the task worktree's HEAD.

    No tip is recoverable and the landing is not linear, so the gate compares
    the worktree's live HEAD and still sees the dropped file.
    """
    scene = _landing_that_drops_the_branch(repo, tmp_path, {'feature.txt': 'branch work\n'})
    _fail_one_git_call(monkeypatch, tmp_path / 'bin', f'rev-parse {scene.landed}^2')

    outcome = await _finalize(
        git_ops, _request(lane_config, scene.worktree), landed=scene.landed, base=scene.base,
    )

    reason = _blocked_reason(outcome)
    assert reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX)
    assert 'feature.txt' in reason
    assert outcome.merge_sha == scene.landed


@pytest.mark.asyncio
async def test_the_triage_hint_names_a_placeholder_when_no_tip_is_on_record(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a recorded tip the triage command names ``<branch-tip>`` for the human to fill in."""
    scene = _landing_that_drops_the_branch(repo, tmp_path, {'feature.txt': 'branch work\n'})
    _fail_one_git_call(monkeypatch, tmp_path / 'bin', f'rev-parse {scene.landed}^2')

    outcome = await _finalize(
        git_ops, _request(lane_config, scene.worktree), landed=scene.landed, base=scene.base,
    )

    assert f'Triage with `git diff <branch-tip> {scene.landed[:12]} -- <path>`' in _blocked_reason(outcome)


@pytest.mark.asyncio
async def test_every_diverging_path_is_listed_comma_separated(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
) -> None:
    """The reason lists each diverging path in git's order, separated by a comma and a space."""
    scene = _landing_that_drops_the_branch(
        repo, tmp_path, {'feature.txt': 'branch work\n', 'notes.txt': 'branch notes\n'},
    )

    outcome = await _finalize(
        git_ops, _request(lane_config, scene.worktree), landed=scene.landed, base=scene.base,
    )

    assert 'branch and main diverge in feature.txt, notes.txt. ' in _blocked_reason(outcome)


@pytest.mark.asyncio
async def test_the_triage_hint_names_the_merged_tip_and_the_landing(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
) -> None:
    """The triage command diffs the merged branch tip against the landing, both at 12 characters."""
    scene = _landing_that_drops_the_branch(repo, tmp_path, {'feature.txt': 'branch work\n'})

    outcome = await _finalize(
        git_ops, _request(lane_config, scene.worktree), landed=scene.landed, base=scene.base,
    )

    assert (
        f'Triage with `git diff {scene.branch_tip[:12]} {scene.landed[:12]} -- <path>`'
        in _blocked_reason(outcome)
    )
