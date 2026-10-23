"""The post-merge equivalence gate's verdicts, driven over real git (task 6413).

Pins what the gate tells a human when a landing diverges from the branch: the
blocked reason, its rename-suppression proof and its γ2 auto-chain hook.  Every
test drives ``_finalize_advanced_merge`` over a real repository, a real
:class:`GitOps` and a real task worktree.  Faults are injected only through a
``git`` shim placed first on PATH.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
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


BASE = ''.join(f'line{i}\n' for i in range(1, 41))
BRANCH_EDITED = BASE.replace('line2\n', 'line2\nBRANCH_EDIT\n')


class Relocation(NamedTuple):
    """A top-level ``a.py`` the branch moved to ``b.py``, landed on main.

    *fork* is where the branch left main, *main_tip* is main just before the
    landing (finalize's base), and *landed* is the landing itself.
    """

    worktree: Path
    fork: str
    main_tip: str
    branch_tip: str
    landed: str


def _relocate_on_branch(repo: Path, tmp_path: Path, *, edit: bool) -> tuple[Path, str, str]:
    """Main adds ``a.py``; the branch moves it to ``b.py``, optionally editing it near the top.

    The paths are top level on purpose: with no directory part, the relocated
    file's parent is the scratch directory the proof stages it in.
    """
    base = _commit(repo, {'a.py': BASE}, 'add a.py')
    worktree = _task_worktree(repo, tmp_path)
    _git(worktree, 'mv', 'a.py', 'b.py')
    branch_tip = _commit(
        worktree, {'b.py': BRANCH_EDITED} if edit else {}, f'impl({TASK_ID}): relocate a.py',
    )
    return worktree, base, branch_tip


def _main_edits_the_source(repo: Path, main_edit: bytes) -> str:
    """Main edits ``a.py`` near the bottom, far from the branch's hunk."""
    edited = BASE.encode().replace(b'line37\n', b'line37\n' + main_edit + b'\n')
    return _commit(repo, {'a.py': edited}, 'main edits a.py')


def _land(repo: Path) -> str:
    """A clean ``--no-ff`` landing: ``b.py`` carries both the branch's and main's edits."""
    _git(repo, 'merge', '--no-ff', '-m', f'Merge {TASK_BRANCH} into main', TASK_BRANCH)
    return _git(repo, 'rev-parse', 'HEAD')


def _land_dropping_the_relocation(repo: Path) -> str:
    """A ``--no-ff`` landing whose resolution deletes the relocated ``b.py`` outright."""
    _git(repo, 'merge', '--no-ff', '--no-commit', TASK_BRANCH)
    _git(repo, 'rm', '-q', '-f', 'b.py')
    _git(repo, 'commit', '-m', f'Merge {TASK_BRANCH} into main (resolution dropped b.py)')
    return _git(repo, 'rev-parse', 'HEAD')


def _relocation(
    repo: Path,
    tmp_path: Path,
    *,
    edit: bool = True,
    main_edit: bytes = b'MAIN_EDIT',
    land: Callable[[Path], str] = _land,
) -> Relocation:
    worktree, fork, branch_tip = _relocate_on_branch(repo, tmp_path, edit=edit)
    main_tip = _main_edits_the_source(repo, main_edit)
    landed = land(repo)
    return Relocation(
        worktree=worktree, fork=fork, main_tip=main_tip, branch_tip=branch_tip, landed=landed,
    )


async def _finalize_relocation(
    git_ops: GitOps, config: OrchestratorConfig, scene: Relocation,
) -> MergeOutcome:
    return await _finalize(
        git_ops, _request(config, scene.worktree), landed=scene.landed, base=scene.main_tip,
    )


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return _messages(caplog, logging.WARNING)


def _infos(caplog: pytest.LogCaptureFixture) -> list[str]:
    return _messages(caplog, logging.INFO)


@pytest.mark.asyncio
async def test_a_faithful_top_level_relocation_lands(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A relocation whose edit re-applies against the merged blob is not reported as diverging.

    Main edited the source too, so the proof must stage the merged blob and
    reverse-apply the branch's patch; at the top level that staging happens
    directly in the scratch directory.
    """
    scene = _relocation(repo, tmp_path)

    with caplog.at_level(logging.INFO, logger=GATES_LOGGER):
        outcome = await _finalize_relocation(git_ops, lane_config, scene)

    assert (outcome.status, outcome.merge_sha) == ('done', scene.landed)


@pytest.mark.asyncio
async def test_the_verified_relocation_is_logged_against_the_task(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The suppressed rename pair is logged once, naming the task it belongs to."""
    scene = _relocation(repo, tmp_path)

    with caplog.at_level(logging.INFO, logger=GATES_LOGGER):
        await _finalize_relocation(git_ops, lane_config, scene)

    verified = (
        "post-merge-equiv: content verified at the new name, not comparing "
        f"[('b.py', 'a.py')] (target, source). task_id={TASK_ID}"
    )
    assert _infos(caplog).count(verified) == 1, _infos(caplog)


@pytest.mark.asyncio
async def test_an_unreadable_branch_delta_keeps_the_relocation_flagged(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the branch's own delta cannot be read, the relocated path is compared and flagged."""
    scene = _relocation(repo, tmp_path)
    _fail_one_git_call(
        monkeypatch, tmp_path / 'bin', f'diff {scene.fork}:a.py {scene.branch_tip}:b.py',
    )

    with caplog.at_level(logging.INFO, logger=GATES_LOGGER):
        outcome = await _finalize_relocation(git_ops, lane_config, scene)

    assert 'b.py' in _blocked_reason(outcome)
    assert any(
        message.startswith(
            f'post-merge-equiv: branch-delta diff {scene.fork}:a.py..{scene.branch_tip}:b.py '
            'failed (rc=128'
        )
        and message.endswith(f'task_id={TASK_ID}')
        for message in _warnings(caplog)
    ), _warnings(caplog)


@pytest.mark.asyncio
async def test_an_undecodable_merged_blob_keeps_the_relocation_flagged(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A merged blob that is not UTF-8 cannot be patch-verified, so the relocation stays flagged.

    Main's latin-1 edit sits far from the branch's hunk, so the branch's patch
    itself stays ASCII and only the merged blob is undecodable.
    """
    scene = _relocation(repo, tmp_path, main_edit=b'MAIN_\xe9DIT')

    with caplog.at_level(logging.INFO, logger=GATES_LOGGER):
        outcome = await _finalize_relocation(git_ops, lane_config, scene)

    assert 'b.py' in _blocked_reason(outcome)
    assert (
        f'post-merge-equiv: merged blob {scene.landed}:b.py is not valid UTF-8 and cannot be '
        f'patch-verified; not suppressing. task_id={TASK_ID}'
    ) in _warnings(caplog)


@pytest.mark.asyncio
async def test_an_unreadable_merged_blob_keeps_the_relocation_flagged(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the merged blob cannot be read, the relocated path is compared and flagged."""
    scene = _relocation(repo, tmp_path)
    _fail_one_git_call(monkeypatch, tmp_path / 'bin', f'show {scene.landed}:b.py')

    with caplog.at_level(logging.INFO, logger=GATES_LOGGER):
        outcome = await _finalize_relocation(git_ops, lane_config, scene)

    assert 'b.py' in _blocked_reason(outcome)
    assert any(
        message.startswith(f'post-merge-equiv: merged blob {scene.landed}:b.py unreadable (rc=128')
        for message in _warnings(caplog)
    ), _warnings(caplog)


@pytest.mark.asyncio
async def test_a_working_tree_file_cannot_stand_in_for_a_dropped_relocation(
    repo: Path, tmp_path: Path, git_ops: GitOps, lane_config: OrchestratorConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pure relocation the resolution deleted is flagged, even beside a file named like its blob.

    The branch's delta is empty, so only the merged path's existence separates
    "relocated" from "dropped".  An untracked file literally named
    ``<landing>:b.py`` in the main checkout must not pass for that blob.
    """
    scene = _relocation(repo, tmp_path, edit=False, land=_land_dropping_the_relocation)
    (repo / f'{scene.landed}:b.py').write_text('not a blob\n')

    with caplog.at_level(logging.INFO, logger=GATES_LOGGER):
        outcome = await _finalize_relocation(git_ops, lane_config, scene)

    assert 'b.py' in _blocked_reason(outcome)
    assert any(
        message.startswith(
            f'post-merge-equiv: merged path {scene.landed}:b.py resolves to no blob (rc=128'
        )
        for message in _warnings(caplog)
    ), _warnings(caplog)
