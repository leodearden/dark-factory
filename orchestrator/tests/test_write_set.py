"""Tests for orchestrator.agents.write_set — compute_write_set() / WriteSet.

OS-sandbox alpha2 (task 2904): compute_write_set() is the SINGLE source of
truth for the writable-path list both sandbox backends (bwrap, landlock)
consume via their existing writable_modules/writable_extras params. This
task is derivation-only — no backend or call-site is touched here.

Test coverage:
  step-1: happy-path core off a REAL linked worktree (git init + git
          worktree add), with an injected `home` — frozen dataclass, every
          named-field derivation, the git-admin-name==worktree.name
          coincidence, the FINAL-WRITABLE-LIST freeze (fleet-only; no
          hooks/state; no whole ~/.claude), writable_paths() aggregate
          shape, and purity (no makedirs side effects).
  step-3: `.task-meta` symlink-target resolution (a symlinked worktree
          base must resolve to the REAL meta-root path landlock enforces
          on, not the unresolved symlinked path).
  step-5: robust `.git` gitdir-parsing edge cases (missing/directory/
          malformed `.git`, and a RELATIVE gitdir value).
"""
from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from orchestrator.agents.write_set import WriteSet, compute_write_set
from orchestrator.artifacts import TaskArtifacts


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def main_repo(tmp_path: Path) -> Path:
    """A real git repo with one commit on `main`."""
    main = tmp_path / 'main-repo'
    main.mkdir()
    _git(['init', '-b', 'main'], main)
    _git(['config', 'user.email', 'test@test.com'], main)
    _git(['config', 'user.name', 'Test'], main)
    (main / 'README.md').write_text('# test\n')
    _git(['add', '-A'], main)
    _git(['commit', '-m', 'initial commit'], main)
    return main


@pytest.fixture
def worktree(tmp_path: Path, main_repo: Path) -> Path:
    """A REAL linked worktree off `main_repo`, at `<base>/<name>`."""
    base = tmp_path / 'worktrees'
    base.mkdir()
    wt = base / 'task-2904'
    _git(['worktree', 'add', '-b', 'task/2904', str(wt), 'main'], main_repo)
    return wt


@pytest.fixture
def tmp_home(tmp_path: Path) -> Path:
    home = tmp_path / 'home'
    home.mkdir()
    return home


# ---------------------------------------------------------------------------
# step-1: happy-path core (RED: orchestrator.agents.write_set does not exist)
# ---------------------------------------------------------------------------


class TestComputeWriteSetHappyPath:
    """Pins compute_write_set()'s per-field derivation off a REAL linked
    worktree + an injected `home`, and the FINAL-WRITABLE-LIST (fleet-only)
    freeze inherited from alpha1.
    """

    def test_writeset_is_frozen(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, home=tmp_home)
        assert isinstance(ws, WriteSet)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ws.worktree = Path('/tmp/other')  # type: ignore[misc]

    def test_worktree_field_is_resolved_worktree(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, home=tmp_home)
        assert ws.worktree == worktree.resolve()

    def test_git_carveouts_resolve_off_main_git(
        self, worktree: Path, main_repo: Path, tmp_home: Path,
    ):
        ws = compute_write_set(worktree, home=tmp_home)
        name = worktree.name
        assert ws.git_objects == (main_repo / '.git' / 'objects').resolve()
        assert ws.git_task_refs == (
            main_repo / '.git' / 'refs' / 'heads' / 'task'
        ).resolve()
        assert ws.git_task_reflogs == (
            main_repo / '.git' / 'logs' / 'refs' / 'heads' / 'task'
        ).resolve()
        assert ws.git_worktree_admin == (
            main_repo / '.git' / 'worktrees' / name
        ).resolve()

    def test_worktree_admin_name_matches_worktree_dir_basename(
        self, worktree: Path, tmp_home: Path,
    ):
        # git's own admin-dir naming coincidence: the internal worktrees/<name>
        # dir is named after the worktree's own basename — task_id never
        # enters path derivation.
        ws = compute_write_set(worktree, home=tmp_home)
        assert ws.git_worktree_admin.name == worktree.name

    def test_task_meta_matches_meta_root_for(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, home=tmp_home)
        expected = TaskArtifacts.meta_root_for(worktree.parent, worktree.name).resolve()
        assert ws.task_meta == expected

    def test_home_derived_static_paths(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, home=tmp_home)
        assert ws.uv_cache == (tmp_home / '.cache' / 'uv').resolve()
        assert ws.claude_fleet == (tmp_home / '.claude' / 'fleet').resolve()
        assert ws.tmp == Path('/tmp')
        assert ws.dev == Path('/dev')

    def test_final_writable_list_excludes_whole_claude_and_hooks_state(
        self, worktree: Path, tmp_home: Path,
    ):
        # alpha1 FINAL-WRITABLE-LIST: ~/.claude/fleet/ ONLY. Neither the
        # whole ~/.claude dir nor ~/.claude/hooks/state may appear anywhere
        # in the aggregate writable-path list.
        ws = compute_write_set(worktree, home=tmp_home)
        paths = ws.writable_paths()
        assert tmp_home.resolve() / '.claude' not in paths
        assert (tmp_home.resolve() / '.claude' / 'hooks' / 'state') not in paths

    def test_writable_paths_returns_exactly_the_ten_named_fields_in_order(
        self, worktree: Path, tmp_home: Path,
    ):
        ws = compute_write_set(worktree, home=tmp_home)
        expected = [
            ws.worktree,
            ws.task_meta,
            ws.git_objects,
            ws.git_task_refs,
            ws.git_task_reflogs,
            ws.git_worktree_admin,
            ws.uv_cache,
            ws.claude_fleet,
            ws.tmp,
            ws.dev,
        ]
        paths = ws.writable_paths()
        assert paths == expected
        assert len(paths) == 10

    def test_purity_no_makedirs_side_effect(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, home=tmp_home)
        assert not ws.uv_cache.exists()


# ---------------------------------------------------------------------------
# step-3: `.task-meta` symlink-target resolution
# ---------------------------------------------------------------------------


class TestTaskMetaSymlinkResolution:
    """step-3: `task_meta` must resolve to the REAL directory landlock
    enforces on, not the unresolved symlinked path — pins the task's
    explicit `.task-meta` symlink-resolution requirement.

    Builds a layout where the resolved `.task-meta/<name>` differs from the
    naive join: the real meta dir lives at `<tmp>/real-meta/<name>`, and
    `<base>/.task-meta` is a symlink to `<tmp>/real-meta`. Also creates
    `<worktree>/.task/plan.json` as an actual symlink into that real meta
    dir (via `TaskArtifacts.ensure_lane_plan_symlink`) to mirror production.
    """

    def test_task_meta_resolves_through_symlinked_base(
        self, tmp_path: Path, worktree: Path, tmp_home: Path,
    ):
        real_meta_root = tmp_path / 'real-meta'
        real_meta_root.mkdir()
        base_task_meta = worktree.parent / '.task-meta'
        base_task_meta.symlink_to(real_meta_root, target_is_directory=True)

        # Mirror production: TaskArtifacts creates the meta dir + a lane
        # symlink at <worktree>/.task/plan.json pointing into it.
        meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        ta = TaskArtifacts(worktree, meta_root)
        ta.init('task-2904', 'Test Task', 'Desc')
        ta.write_plan({'steps': []})
        ta.ensure_lane_plan_symlink()

        ws = compute_write_set(worktree, home=tmp_home)

        expected = (real_meta_root / worktree.name).resolve()
        assert ws.task_meta == expected
        # NOT the unresolved symlinked path.
        assert ws.task_meta != base_task_meta / worktree.name
