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

from orchestrator.agents.write_set import (
    WriteSet,
    compute_write_set,
    ensure_claude_fleet_dir,
)
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


# ---------------------------------------------------------------------------
# step-5: robust `.git` gitdir-parsing (hand-built dirs, no real git)
# ---------------------------------------------------------------------------


class TestGitdirParsingRobustness:
    """step-5: compute_write_set fails LOUD (ValueError, naming the
    worktree/`.git` path) on a missing/directory/malformed `.git`, and
    correctly supports a RELATIVE `gitdir:` value (resolved against the
    worktree) — real git itself writes relative gitdir files for
    relocatable worktrees.

    RED against step-2/4, which assumed an absolute, well-formed gitdir:
    a missing/directory `.git` raises FileNotFoundError/IsADirectoryError
    (not ValueError); malformed content silently yields a bogus cwd-relative
    admindir; a relative gitdir value resolves against the process cwd
    instead of the worktree.
    """

    def test_missing_git_file_raises_value_error(self, tmp_path: Path, tmp_home: Path):
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with pytest.raises(ValueError) as exc_info:
            compute_write_set(worktree, home=tmp_home)
        assert str(worktree / '.git') in str(exc_info.value)

    def test_git_is_directory_raises_value_error(self, tmp_path: Path, tmp_home: Path):
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        (worktree / '.git').mkdir()

        with pytest.raises(ValueError) as exc_info:
            compute_write_set(worktree, home=tmp_home)
        assert str(worktree / '.git') in str(exc_info.value)

    @pytest.mark.parametrize('content', ['', 'garbage\n'])
    def test_git_content_without_gitdir_prefix_raises_value_error(
        self, tmp_path: Path, tmp_home: Path, content: str,
    ):
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        (worktree / '.git').write_text(content)

        with pytest.raises(ValueError) as exc_info:
            compute_write_set(worktree, home=tmp_home)
        assert str(worktree / '.git') in str(exc_info.value)

    def test_relative_gitdir_resolves_relative_to_worktree(
        self, tmp_path: Path, tmp_home: Path,
    ):
        # Hand-built admin layout — no real `git` subprocess involved.
        main = tmp_path / 'main-repo'
        main_git = main / '.git'
        admin = main_git / 'worktrees' / 'my-wt'
        admin.mkdir(parents=True)
        (main_git / 'objects').mkdir(parents=True)

        worktree = tmp_path / 'my-wt'
        worktree.mkdir()
        (worktree / '.git').write_text('gitdir: ../main-repo/.git/worktrees/my-wt\n')

        ws = compute_write_set(worktree, home=tmp_home)

        assert ws.git_worktree_admin == admin.resolve()
        assert ws.git_objects == (main_git / 'objects').resolve()


# ---------------------------------------------------------------------------
# OS-sandbox β2 (task 2909): WriteSet.digest()
# ---------------------------------------------------------------------------


def _make_write_set(**overrides: Path) -> WriteSet:
    """Construct a WriteSet from dummy Paths (hermetic — digest() depends
    ONLY on writable_paths(), which needs no filesystem)."""
    fields: dict[str, Path] = dict(
        worktree=Path('/wt'),
        task_meta=Path('/base/.task-meta/wt'),
        git_objects=Path('/main/.git/objects'),
        git_task_refs=Path('/main/.git/refs/heads/task'),
        git_task_reflogs=Path('/main/.git/logs/refs/heads/task'),
        git_worktree_admin=Path('/main/.git/worktrees/wt'),
        uv_cache=Path('/home/.cache/uv'),
        claude_fleet=Path('/home/.claude/fleet'),
        tmp=Path('/tmp'),
        dev=Path('/dev'),
    )
    fields.update(overrides)
    return WriteSet(**fields)


class TestWriteSetDigest:
    """Pins WriteSet.digest() — the single owner (INV-5) of the stable
    writable-set hash the β2 ``sandbox_applied`` event carries so an operator
    can diff exactly what a given invocation could touch (INV-2). It is a
    sha256 over the contract-ordered ``writable_paths()`` strings.
    """

    def test_digest_is_deterministic_across_calls(self):
        # (a) Same WriteSet → identical hex string across two calls.
        ws = _make_write_set()
        assert ws.digest() == ws.digest()

    def test_equal_writable_paths_produce_equal_digest(self):
        # (b) Two independently-built WriteSets with identical writable_paths()
        # produce the same digest.
        a = _make_write_set()
        b = _make_write_set()
        assert a.writable_paths() == b.writable_paths()
        assert a.digest() == b.digest()

    def test_one_differing_path_changes_digest(self):
        # (c) A WriteSet differing in exactly one writable path yields a
        # different digest.
        base = _make_write_set()
        changed = _make_write_set(worktree=Path('/other-wt'))
        assert base.writable_paths() != changed.writable_paths()
        assert base.digest() != changed.digest()

    def test_digest_is_64_char_lowercase_sha256_hex(self):
        # (d) Shape: a 64-char lowercase sha256 hex string.
        digest = _make_write_set().digest()
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert digest == digest.lower()
        assert all(c in '0123456789abcdef' for c in digest)


# ---------------------------------------------------------------------------
# task 2996: ensure_claude_fleet_dir() — side-effecting pre-creation helper
# ---------------------------------------------------------------------------


class TestEnsureClaudeFleetDir:
    """Pins ``ensure_claude_fleet_dir(write_set) -> bool`` — the SEPARATE,
    explicitly side-effecting helper that materializes ~/.claude/fleet OUTSIDE
    the sandbox before dispatch (task 2996).

    ``compute_write_set`` stays PURE (``test_purity_no_makedirs_side_effect``);
    this helper is the one place the load-bearing ``claude_fleet`` carve-out is
    created so BOTH backends can grant it (neither can --bind / add a
    nonexistent path with parent ~/.claude read-only). Hermetic: ``claude_fleet``
    always points under ``tmp_path`` — never the real ~/.claude.
    """

    def test_creates_dir_and_missing_parents_when_absent(self, tmp_path: Path):
        # (a) parents=True: a claude_fleet whose parent does not yet exist is
        # created in full, and the call reports success.
        claude_fleet = tmp_path / 'nonexistent-home' / '.claude' / 'fleet'
        ws = _make_write_set(claude_fleet=claude_fleet)
        assert not claude_fleet.exists()
        assert not claude_fleet.parent.exists()  # missing parent chain

        assert ensure_claude_fleet_dir(ws) is True

        assert claude_fleet.is_dir()
        assert claude_fleet.parent.is_dir()  # parents=True built the chain

    def test_idempotent_when_dir_already_exists(self, tmp_path: Path):
        # (b) exist_ok=True: a second call on an already-present dir returns
        # True and does not raise.
        claude_fleet = tmp_path / '.claude' / 'fleet'
        ws = _make_write_set(claude_fleet=claude_fleet)
        assert ensure_claude_fleet_dir(ws) is True
        assert claude_fleet.is_dir()

        assert ensure_claude_fleet_dir(ws) is True
        assert claude_fleet.is_dir()

    def test_returns_false_when_mkdir_raises_oserror(self, tmp_path: Path):
        # (c) failure path: a regular FILE cannot be a parent dir, so
        # mkdir(parents=True) raises NotADirectoryError (an OSError subclass),
        # which the helper catches -> returns False, and nothing is created.
        blocker = tmp_path / 'blocker'
        blocker.write_text('i am a regular file, not a directory\n')
        claude_fleet = blocker / 'fleet'
        ws = _make_write_set(claude_fleet=claude_fleet)

        assert ensure_claude_fleet_dir(ws) is False
        assert not claude_fleet.exists()
