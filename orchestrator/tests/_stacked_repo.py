"""Real-git scenes for a branch stacked on a predecessor's branch (task 10473).

Shared by test_git_ops_stacked_worktree.py and test_workflow_stacked_worktree.py,
because a test file never imports another test file.  Builders only: a git
command that fails raises, and nothing here asserts a test's expectation.

The PRD row 42 scene (:func:`build_restack_scene`), oldest first:
  main M0: base.txt
  task/P: p1 adds p.txt                                    -> P's old tip
  task/D: from P's old tip, own commit d1 adds d.txt;
          the stack-base ledger names P's old tip          -> D's tip
  main M1: m1.txt
  task/P: rebuilt on M1 as p1', which writes p.txt anew    -> P's new tip
  main M2: m2.txt, a file P's new tip does not carry

With ``conflicting=True``, d1 also rewrites p.txt's one line, which P's new
tip rewrites differently, so replaying d1 onto P's new tip conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from _git_fault_shim import fail_git
from _seed_script_stubs import seed_default_warm_base

from orchestrator.artifacts import TaskArtifacts
from orchestrator.branch_stack import StackBaseLedger
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

P_FILE = 'p.txt'
D_FILE = 'd.txt'
MAIN_ONLY_FILE = 'm2.txt'


class SceneGitError(RuntimeError):
    """A git command a scene builder ran failed."""


async def git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    if rc != 0:
        raise SceneGitError(f'git {" ".join(args)} in {cwd} failed: {err.strip()}')
    return out


async def rev_parse(cwd: Path, ref: str) -> str:
    return (await git(cwd, 'rev-parse', '--verify', ref)).strip()


async def commit_files(cwd: Path, writes: dict[str, str], msg: str) -> str:
    """Write and commit exactly *writes*; a bare ``git add -A`` in the main
    checkout would stage the nested ``.worktrees/*`` checkouts as gitlinks."""
    for name, content in writes.items():
        (cwd / name).write_text(content)
    await git(cwd, 'add', '--', *writes)
    await git(cwd, 'commit', '-m', msg)
    return await rev_parse(cwd, 'HEAD')


async def _set_identity(cwd: Path) -> None:
    await git(cwd, 'config', 'user.email', 'test@test.com')
    await git(cwd, 'config', 'user.name', 'Test')


def _git_config(*, warm_lane_pool: bool) -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=warm_lane_pool,
    )


async def _init_main(root: Path) -> None:
    root.mkdir(parents=True)
    await git(root, 'init', '-b', 'main')
    await _set_identity(root)
    await commit_files(root, {'base.txt': 'base\n'}, 'M0: initial')


async def make_repo(root: Path) -> GitOps:
    """A repo at *root* whose main carries base.txt, and a cold GitOps on it."""
    await _init_main(root)
    return GitOps(_git_config(warm_lane_pool=False), root)


async def make_warm_repo(root: Path) -> GitOps:
    """As :func:`make_repo`, with a one-lane warm pool: the seed-warm-lane.sh
    stub and a .gitignore that keeps the seeded target/ out of every commit."""
    await _init_main(root)
    seed = root / 'scripts' / 'seed-warm-lane.sh'
    seed.parent.mkdir(parents=True)
    seed.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho seeded > "$2/target/seeded.bin"\n'
    )
    seed.chmod(0o755)
    (root / '.gitignore').write_text('target/\n')
    await git(root, 'add', '--', 'scripts/seed-warm-lane.sh', '.gitignore')
    await git(root, 'commit', '-m', 'add warm-lane seed stub')
    seed_default_warm_base(root)
    return GitOps(_git_config(warm_lane_pool=True), root, warm_lane_pool_size=1)


def ledger_of(git_ops: GitOps) -> StackBaseLedger:
    return StackBaseLedger(git_ops.project_root, _run)


def pool_lane(git_ops: GitOps) -> Path:
    pool = git_ops.warm_lane_pool
    if pool is None:
        raise ValueError('this GitOps has no warm-lane pool')
    return pool.lane_paths()[0]


async def add_worktree(git_ops: GitOps, name: str, start_ref: str) -> Path:
    """A worktree at ``worktree_base/<name>`` on a new branch task/<name>."""
    wt = git_ops.worktree_base / name
    wt.parent.mkdir(parents=True, exist_ok=True)
    await git(git_ops.project_root, 'worktree', 'add', '-b', f'task/{name}', str(wt), start_ref)
    await _set_identity(wt)
    return wt


async def release_into_lane(git_ops: GitOps, name: str) -> Path:
    """Move task *name*'s worktree into the pool lane and release it there: the
    disk-backstop shape, a detached lane whose HEAD is the branch tip and whose
    metadata.json names the task."""
    lane = pool_lane(git_ops)
    await git(git_ops.project_root, 'worktree', 'move', str(git_ops.worktree_base / name), str(lane))
    TaskArtifacts.for_task(lane, name).init(name, name, '')
    await git_ops.release_warm_lane(lane, name)
    return lane


def fail_git_calls_starting(monkeypatch: pytest.MonkeyPatch, where: Path, prefix: str) -> Path:
    """Put a ``git`` first on PATH that exits 128 for every call whose
    arguments start with *prefix*.  Returns the file the shim appends to on
    each matched call, so a test can tell the shim was reached."""
    fail_git(monkeypatch, where / 'bin', f'{prefix}*')
    return where / 'bin' / 'matched'


@dataclass(frozen=True)
class RestackScene:
    git_ops: GitOps
    p_old: str
    p_new: str
    main_tip: str
    d_tip: str

    @property
    def root(self) -> Path:
        return self.git_ops.project_root

    @property
    def d_worktree(self) -> Path:
        return self.git_ops.worktree_base / 'D'

    @property
    def p_worktree(self) -> Path:
        return self.git_ops.worktree_base / 'P'

    @property
    def ledger(self) -> StackBaseLedger:
        return ledger_of(self.git_ops)


async def build_restack_scene(git_ops: GitOps, *, conflicting: bool = False) -> RestackScene:
    """Build the PRD row 42 scene (module docstring) on *git_ops*'s repo."""
    root = git_ops.project_root
    p_wt = await add_worktree(git_ops, 'P', 'main')
    p_old = await commit_files(p_wt, {P_FILE: 'p\n'}, 'p1: write p.txt')

    d_wt = await add_worktree(git_ops, 'D', p_old)
    d_writes = {D_FILE: 'd\n'}
    if conflicting:
        d_writes[P_FILE] = 'd edits p\n'
    d_tip = await commit_files(d_wt, d_writes, 'd1: own work')
    await ledger_of(git_ops).record('task/D', p_old)

    m1 = await commit_files(root, {'m1.txt': 'm1\n'}, 'M1: main advances')
    await git(p_wt, 'reset', '--hard', m1)
    p_new = await commit_files(p_wt, {P_FILE: 'p rebuilt\n'}, "p1': write p.txt anew")
    main_tip = await commit_files(root, {MAIN_ONLY_FILE: 'm2\n'}, 'M2: main advances again')
    return RestackScene(git_ops=git_ops, p_old=p_old, p_new=p_new, main_tip=main_tip, d_tip=d_tip)
