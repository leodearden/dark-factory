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
import os
import subprocess
from pathlib import Path

import pytest

from orchestrator.agents.write_set import (
    WriteSet,
    compute_write_set,
    ensure_sandbox_dirs,
    write_set_of,
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
    """A REAL linked worktree off `main_repo`, at `<base>/<task id>` as
    GitOps.create_worktree names it — never `task-<id>`, which is the
    task-root shape (`TaskArtifacts.task_root_for`)."""
    base = tmp_path / 'worktrees'
    base.mkdir()
    wt = base / '2904'
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
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert isinstance(ws, WriteSet)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ws.worktree = Path('/tmp/other')  # type: ignore[misc]

    def test_worktree_field_is_resolved_worktree(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert ws.worktree == worktree.resolve()

    def test_git_carveouts_resolve_off_main_git(
        self, worktree: Path, main_repo: Path, tmp_home: Path,
    ):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
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
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert ws.git_worktree_admin.name == worktree.name

    def test_task_meta_matches_meta_root_for(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        expected = TaskArtifacts.meta_root_for(worktree.parent, worktree.name).resolve()
        assert ws.task_meta == expected

    def test_task_root_matches_task_root_for(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert ws.task_root == TaskArtifacts.task_root_for(worktree.parent, '2904').resolve()

    def test_task_root_is_granted_right_after_task_meta(self, worktree: Path, tmp_home: Path):
        paths = compute_write_set(worktree, task_id='2904', home=tmp_home).writable_paths()
        assert paths[paths.index(TaskArtifacts.meta_root_for(
            worktree.parent, worktree.name).resolve()) + 1] == (
            TaskArtifacts.task_root_for(worktree.parent, '2904').resolve()
        )

    def test_a_single_root_dispatch_grants_no_task_root(self, worktree: Path, tmp_home: Path):
        lane_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        ws = compute_write_set(worktree, task_id='2904', task_root=lane_root, home=tmp_home)
        assert ws.task_root == lane_root.resolve()
        assert TaskArtifacts.task_root_for(worktree.parent, '2904').resolve() not in ws.writable_paths()

    def test_invalid_task_id_raises(self, worktree: Path, tmp_home: Path):
        with pytest.raises(ValueError):
            compute_write_set(worktree, task_id='a/b', home=tmp_home)

    def test_home_derived_static_paths(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert ws.uv_cache == (tmp_home / '.cache' / 'uv').resolve()
        assert ws.tree_sitter_cache == (tmp_home / '.cache' / 'tree-sitter').resolve()
        assert ws.claude_fleet == (tmp_home / '.claude' / 'fleet').resolve()
        assert ws.tmp == Path('/tmp')
        assert ws.dev == Path('/dev')

    def test_final_writable_list_excludes_whole_claude_and_hooks_state(
        self, worktree: Path, tmp_home: Path,
    ):
        # alpha1 FINAL-WRITABLE-LIST: ~/.claude/fleet/ ONLY. Neither the
        # whole ~/.claude dir nor ~/.claude/hooks/state may appear anywhere
        # in the aggregate writable-path list.
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        paths = ws.writable_paths()
        assert tmp_home.resolve() / '.claude' not in paths
        assert (tmp_home.resolve() / '.claude' / 'hooks' / 'state') not in paths

    def test_writable_paths_returns_exactly_the_twelve_named_fields_in_order(
        self, worktree: Path, tmp_home: Path,
    ):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        expected = [
            ws.worktree,
            ws.task_meta,
            ws.task_root,
            ws.git_objects,
            ws.git_task_refs,
            ws.git_task_reflogs,
            ws.git_worktree_admin,
            ws.uv_cache,
            ws.tree_sitter_cache,
            ws.claude_fleet,
            ws.tmp,
            ws.dev,
        ]
        paths = ws.writable_paths()
        assert paths == expected
        assert len(paths) == 12

    def test_purity_no_makedirs_side_effect(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert not ws.uv_cache.exists()
        assert not ws.tree_sitter_cache.exists()


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
    dir (via `TaskArtifacts.ensure_lane_symlinks`) to mirror production.
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
        ta.ensure_lane_symlinks()

        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)

        expected = (real_meta_root / worktree.name).resolve()
        assert ws.task_meta == expected
        # NOT the unresolved symlinked path.
        assert ws.task_meta != base_task_meta / worktree.name

    def test_task_root_resolves_through_symlinked_base(
        self, tmp_path: Path, worktree: Path, tmp_home: Path,
    ):
        real_meta_root = tmp_path / 'real-meta'
        real_meta_root.mkdir()
        base_task_meta = worktree.parent / '.task-meta'
        base_task_meta.symlink_to(real_meta_root, target_is_directory=True)
        TaskArtifacts.for_task(worktree, '2904').init('2904', 'Test Task', 'Desc')

        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)

        assert ws.task_root == (real_meta_root / 'task-2904').resolve()
        assert ws.task_root != base_task_meta / 'task-2904'


# ---------------------------------------------------------------------------
# task 4825: WriteSet.is_foreign_task_meta() and write_set_of()
# ---------------------------------------------------------------------------


class TestForeignTaskMeta:
    """A dispatch's only `.task-meta` grants are its lane dir (row 2) and its
    task root (row 2b); every other path inside the `.task-meta` base belongs
    to another lane or task, and no sandbox of this dispatch can write it."""

    def test_sibling_lane_and_task_root_are_foreign(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        base = worktree.parent / '.task-meta'
        assert ws.is_foreign_task_meta(base / '4762' / 'plan.json')
        assert ws.is_foreign_task_meta(base / 'task-4762' / 'plan.json')

    def test_own_lane_and_task_root_are_not_foreign(self, worktree: Path, tmp_home: Path):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        base = worktree.parent / '.task-meta'
        assert not ws.is_foreign_task_meta(base / '2904' / 'plan.json')
        assert not ws.is_foreign_task_meta(base / 'task-2904' / 'plan.json')

    def test_base_itself_worktree_and_outside_paths_are_not_foreign(
        self, worktree: Path, main_repo: Path, tmp_home: Path,
    ):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert not ws.is_foreign_task_meta(worktree.parent / '.task-meta')
        assert not ws.is_foreign_task_meta(worktree / 'x.py')
        assert not ws.is_foreign_task_meta(main_repo / 'README.md')

    def test_a_single_root_dispatch_does_not_own_the_task_root(
        self, worktree: Path, tmp_home: Path,
    ):
        lane_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        ws = compute_write_set(worktree, task_id='2904', task_root=lane_root, home=tmp_home)
        assert ws.is_foreign_task_meta(worktree.parent / '.task-meta' / 'task-2904' / 'plan.json')

    def test_a_path_spelled_through_a_symlinked_base_is_judged_on_its_real_path(
        self, tmp_path: Path, worktree: Path, tmp_home: Path,
    ):
        real_meta_root = tmp_path / 'real-meta'
        real_meta_root.mkdir()
        base_task_meta = worktree.parent / '.task-meta'
        base_task_meta.symlink_to(real_meta_root, target_is_directory=True)

        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)

        assert ws.is_foreign_task_meta(base_task_meta / '4762' / 'plan.json')
        assert not ws.is_foreign_task_meta(base_task_meta / '2904' / 'plan.json')

    def test_write_set_of_matches_compute_write_set(self, worktree: Path, tmp_home: Path):
        artifacts = TaskArtifacts.for_task(worktree, '2904')
        assert write_set_of(artifacts, home=tmp_home) == compute_write_set(
            worktree, task_id='2904', home=tmp_home,
        )

    def test_write_set_of_a_single_root_dispatch_grants_its_lane_as_task_root(
        self, worktree: Path, tmp_home: Path,
    ):
        lane_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        artifacts = TaskArtifacts(worktree, lane_root, lane_root)
        assert write_set_of(artifacts, home=tmp_home) == compute_write_set(
            worktree, task_id='2904', task_root=lane_root, home=tmp_home,
        )


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
            compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert str(worktree / '.git') in str(exc_info.value)

    def test_git_is_directory_raises_value_error(self, tmp_path: Path, tmp_home: Path):
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        (worktree / '.git').mkdir()

        with pytest.raises(ValueError) as exc_info:
            compute_write_set(worktree, task_id='2904', home=tmp_home)
        assert str(worktree / '.git') in str(exc_info.value)

    @pytest.mark.parametrize('content', ['', 'garbage\n'])
    def test_git_content_without_gitdir_prefix_raises_value_error(
        self, tmp_path: Path, tmp_home: Path, content: str,
    ):
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        (worktree / '.git').write_text(content)

        with pytest.raises(ValueError) as exc_info:
            compute_write_set(worktree, task_id='2904', home=tmp_home)
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

        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)

        assert ws.git_worktree_admin == admin.resolve()
        assert ws.git_objects == (main_git / 'objects').resolve()


# ---------------------------------------------------------------------------
# OS-sandbox β2 (task 2909): WriteSet.digest()
# ---------------------------------------------------------------------------


def _make_write_set(**overrides: Path) -> WriteSet:
    """Construct a WriteSet from dummy Paths (hermetic — digest() depends
    ONLY on writable_paths(), which needs no filesystem)."""
    base = WriteSet(
        worktree=Path('/wt'),
        task_meta=Path('/base/.task-meta/wt'),
        task_root=Path('/base/.task-meta/task-1'),
        git_objects=Path('/main/.git/objects'),
        git_task_refs=Path('/main/.git/refs/heads/task'),
        git_task_reflogs=Path('/main/.git/logs/refs/heads/task'),
        git_worktree_admin=Path('/main/.git/worktrees/wt'),
        uv_cache=Path('/home/.cache/uv'),
        tree_sitter_cache=Path('/home/.cache/tree-sitter'),
        claude_fleet=Path('/home/.claude/fleet'),
        tmp=Path('/tmp'),
        dev=Path('/dev'),
    )
    return dataclasses.replace(base, **overrides)


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
# tasks 2996 / 3382: ensure_sandbox_dirs() — side-effecting pre-creation helper
# ---------------------------------------------------------------------------


class TestEnsureSandboxDirs:
    """Pins ``ensure_sandbox_dirs(write_set) -> list[Path]`` — the SEPARATE,
    explicitly side-effecting helper that materializes every must-pre-exist
    carve-out OUTSIDE the sandbox before dispatch: ``claude_fleet`` (task 2996)
    and ``tree_sitter_cache`` (task 3382). It returns the paths it could NOT
    create; an empty list means every carve-out is present.

    ``compute_write_set`` stays PURE (``test_purity_no_makedirs_side_effect``);
    this helper is the one place those carve-outs are created so BOTH backends
    can grant them (neither can bind / add a nonexistent path, and the agent
    cannot create one from inside, where ~/.claude and ~/.cache are read-only).

    This class tests the MODULE-LEVEL helper. The same-named class in
    ``test_workflow_sandbox_events.py`` tests the ``TaskWorkflow._ensure_sandbox_dirs``
    wrapper that calls it and owns the operator-facing warning.

    Hermetic: every test overrides BOTH ``claude_fleet`` and
    ``tree_sitter_cache`` to paths under ``tmp_path``, because
    ``_make_write_set`` defaults them to real-looking absolute dummies.
    """

    def test_creates_tree_sitter_cache_when_absent(self, tmp_path: Path):
        claude_fleet = tmp_path / 'home' / '.claude' / 'fleet'
        claude_fleet.mkdir(parents=True)
        tree_sitter_cache = tmp_path / 'home' / '.cache' / 'tree-sitter'
        ws = _make_write_set(claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache)
        assert not tree_sitter_cache.exists()

        assert ensure_sandbox_dirs(ws) == []

        assert tree_sitter_cache.is_dir()

    def test_creates_both_carve_outs_in_one_call(self, tmp_path: Path):
        claude_fleet = tmp_path / 'home' / '.claude' / 'fleet'
        tree_sitter_cache = tmp_path / 'home' / '.cache' / 'tree-sitter'
        ws = _make_write_set(claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache)
        assert not claude_fleet.exists()
        assert not tree_sitter_cache.exists()

        assert ensure_sandbox_dirs(ws) == []

        assert claude_fleet.is_dir()
        assert tree_sitter_cache.is_dir()

    def test_creates_dirs_and_missing_parents_when_absent(self, tmp_path: Path):
        home = tmp_path / 'nonexistent-home'
        claude_fleet = home / '.claude' / 'fleet'
        tree_sitter_cache = home / '.cache' / 'tree-sitter'
        ws = _make_write_set(claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache)
        assert not home.exists()

        assert ensure_sandbox_dirs(ws) == []

        assert claude_fleet.is_dir()
        assert tree_sitter_cache.is_dir()

    def test_idempotent_when_dirs_already_exist(self, tmp_path: Path):
        claude_fleet = tmp_path / '.claude' / 'fleet'
        tree_sitter_cache = tmp_path / '.cache' / 'tree-sitter'
        ws = _make_write_set(claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache)
        assert ensure_sandbox_dirs(ws) == []

        assert ensure_sandbox_dirs(ws) == []

        assert claude_fleet.is_dir()
        assert tree_sitter_cache.is_dir()

    def test_returns_failing_path_and_still_creates_the_other(self, tmp_path: Path):
        # A regular FILE cannot be a parent dir, so mkdir(parents=True) raises
        # NotADirectoryError. One failing carve-out must not suppress the rest.
        blocker = tmp_path / 'blocker'
        blocker.write_text('i am a regular file, not a directory\n')
        claude_fleet = blocker / 'fleet'
        tree_sitter_cache = tmp_path / '.cache' / 'tree-sitter'
        ws = _make_write_set(claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache)

        assert ensure_sandbox_dirs(ws) == [claude_fleet]

        assert not claude_fleet.exists()
        assert tree_sitter_cache.is_dir()


# ---------------------------------------------------------------------------
# task 4627: the conditional docs-only project_root grant
# ---------------------------------------------------------------------------


_DOCS_ONLY = ['docs/existing.md', 'CLAUDE.md', 'docs/prds/new.md']


@pytest.fixture
def docs_root(main_repo: Path) -> Path:
    """`main_repo` seeded with the docs, source and `.taskmaster` layout the
    grant resolver distinguishes; returns the resolved project root."""
    (main_repo / 'docs' / 'prds').mkdir(parents=True)
    (main_repo / 'docs' / 'existing.md').write_text('existing\n')
    (main_repo / 'CLAUDE.md').write_text('claude\n')
    (main_repo / 'src').mkdir()
    (main_repo / 'src' / 'x.py').write_text('')
    (main_repo / '.taskmaster').mkdir()
    (main_repo / '.taskmaster' / 'tasks.db').write_text('')
    return main_repo.resolve()


class TestDocsMainGrant:
    """A task whose declared files are all docs also gets the main checkout's
    `.git` and the narrowest grant for each declared path; every other
    invocation's write set and digest are unchanged."""

    def _ws(self, worktree: Path, home: Path, declared: object = None) -> WriteSet:
        return compute_write_set(worktree, task_id='2904', home=home, declared_files=declared)

    def test_default_has_no_grant_and_is_unchanged(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        default = compute_write_set(worktree, task_id='2904', home=tmp_home)
        explicit_none = self._ws(worktree, tmp_home, None)
        assert default.docs_main_git is None
        assert default.docs_main_paths == ()
        assert explicit_none == default
        assert explicit_none.writable_paths() == default.writable_paths()
        assert explicit_none.digest() == default.digest()

    def test_non_docs_list_has_no_grant_and_is_unchanged(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        default = compute_write_set(worktree, task_id='2904', home=tmp_home)
        mixed = self._ws(worktree, tmp_home, ['docs/existing.md', 'src/x.py'])
        assert mixed.docs_main_git is None
        assert mixed.docs_main_paths == ()
        assert mixed.writable_paths() == default.writable_paths()
        assert mixed.digest() == default.digest()

    def test_docs_only_list_grants_git_and_narrowest_paths(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        default = compute_write_set(worktree, task_id='2904', home=tmp_home)
        ws = self._ws(worktree, tmp_home, _DOCS_ONLY)
        expected_paths = (
            docs_root / 'docs' / 'existing.md',
            docs_root / 'CLAUDE.md',
            docs_root / 'docs' / 'prds',
        )
        assert ws.docs_main_git == docs_root / '.git'
        assert ws.docs_main_paths == expected_paths
        assert ws.writable_paths()[-4:] == [docs_root / '.git', *expected_paths]
        assert ws.writable_paths()[:-4] == default.writable_paths()
        assert ws.digest() != default.digest()

    def test_declared_existing_directory_is_granted_itself(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        ws = self._ws(worktree, tmp_home, ['docs'])
        assert ws.docs_main_paths == (docs_root / 'docs',)

    def test_new_file_nested_in_docs_grants_its_nearest_existing_docs_dir(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        ws = self._ws(worktree, tmp_home, ['docs/a/b/c.md'])
        assert ws.docs_main_paths == (docs_root / 'docs',)

    @pytest.mark.parametrize(('declared', 'reason'), [
        ('BRAND-NEW.md', 'not strictly inside'),
        ('docs/out.md', 'not strictly inside'),
        ('.taskmaster/notes.md', '.taskmaster, which is never granted'),
        ('docs/link.md', '.taskmaster, which is never granted'),
        ('.git/notes.md', '.git, which is never granted'),
        ('src/pkg/NOTES.md', 'outside a top-level docs dir'),
        ('docs/existing.md/x.md', 'nearest existing ancestor'),
        ('docs/fifo.md', 'not a regular file or directory'),
    ])
    def test_refused_path_is_warned_with_its_reason_and_not_granted(
        self, declared: str, reason: str, docs_root: Path, worktree: Path, tmp_home: Path,
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        (docs_root / 'docs' / 'link.md').symlink_to(docs_root / '.taskmaster' / 'tasks.db')
        outside = tmp_path / 'outside.md'
        outside.write_text('outside\n')
        (docs_root / 'docs' / 'out.md').symlink_to(outside)
        (docs_root / '.git' / 'notes.md').write_text('notes\n')
        (docs_root / 'src' / 'pkg').mkdir()
        os.mkfifo(docs_root / 'docs' / 'fifo.md')

        with caplog.at_level('WARNING', logger='orchestrator.agents.write_set'):
            ws = self._ws(worktree, tmp_home, ['docs/existing.md', declared])

        assert ws.docs_main_paths == (docs_root / 'docs' / 'existing.md',)
        warnings = [r.getMessage() for r in caplog.records if r.levelname == 'WARNING']
        assert any(declared in m and reason in m for m in warnings), warnings

    def test_nothing_in_or_over_the_configured_lane_base_is_granted(
        self, docs_root: Path, tmp_home: Path, caplog: pytest.LogCaptureFixture,
    ):
        lanes = docs_root / 'lanes'
        lanes.mkdir()
        lane = lanes / '2905'
        _git(['worktree', 'add', '-b', 'task/2905', str(lane), 'main'], docs_root)
        meta = TaskArtifacts.task_root_for(lanes, '77')
        meta.mkdir(parents=True)
        (meta / 'plan.md').write_text('plan\n')
        declared = ['docs/existing.md', 'lanes/2905/README.md', 'lanes/.task-meta/task-77/plan.md']

        with caplog.at_level('WARNING', logger='orchestrator.agents.write_set'):
            ws = compute_write_set(lane, task_id='2905', home=tmp_home, declared_files=declared)

        assert ws.docs_main_paths == (docs_root / 'docs' / 'existing.md',)
        warnings = [r.getMessage() for r in caplog.records if r.levelname == 'WARNING']
        for rel in declared[1:]:
            assert any(rel in m and 'which is never granted' in m for m in warnings), warnings

    def test_lane_base_holding_the_project_root_beside_its_lanes_still_grants(
        self, docs_root: Path, tmp_home: Path,
    ):
        lane = docs_root.parent / '2906'
        _git(['worktree', 'add', '-b', 'task/2906', str(lane), 'main'], docs_root)

        ws = compute_write_set(
            lane, task_id='2906', home=tmp_home, declared_files=['docs/existing.md'],
        )

        assert ws.docs_main_paths == (docs_root / 'docs' / 'existing.md',)

    def test_common_dir_not_named_git_makes_no_grant(
        self, tmp_path: Path, tmp_home: Path, caplog: pytest.LogCaptureFixture,
    ):
        common = tmp_path / 'common.git'
        (common / 'worktrees' / 'lane').mkdir(parents=True)
        lane = tmp_path / 'lanes' / 'lane'
        lane.mkdir(parents=True)
        (lane / '.git').write_text(f'gitdir: {common}/worktrees/lane\n')

        with caplog.at_level('WARNING', logger='orchestrator.agents.write_set'):
            ws = compute_write_set(lane, task_id='2904', home=tmp_home, declared_files=['CLAUDE.md'])

        assert ws.docs_main_git is None
        assert ws.docs_main_paths == ()
        assert any(
            'is not a <project_root>/.git' in r.getMessage() for r in caplog.records
        ), [r.getMessage() for r in caplog.records]

    def test_every_path_refused_makes_no_grant_at_all(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        ws = self._ws(worktree, tmp_home, ['BRAND-NEW.md', '.taskmaster/notes.md'])
        assert ws.docs_main_git is None
        assert ws.docs_main_paths == ()

    def test_paths_without_git_violates_the_invariant(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        with pytest.raises(ValueError, match='docs_main_paths') as excinfo:
            dataclasses.replace(ws, docs_main_paths=(docs_root / 'CLAUDE.md',))
        assert 'docs_main_git' in str(excinfo.value)

    @pytest.mark.parametrize('rel', ['.taskmaster/tasks.db', '.git/config', '.'])
    def test_ungrantable_path_violates_the_invariant(
        self, rel: str, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        ws = self._ws(worktree, tmp_home, _DOCS_ONLY)
        with pytest.raises(ValueError, match='ungrantable'):
            dataclasses.replace(ws, docs_main_paths=((docs_root / rel).resolve(),))

    def test_git_without_paths_violates_the_invariant(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        ws = compute_write_set(worktree, task_id='2904', home=tmp_home)
        with pytest.raises(ValueError, match='docs_main_git') as excinfo:
            dataclasses.replace(ws, docs_main_git=docs_root / '.git')
        assert 'docs_main_paths' in str(excinfo.value)

    def test_docs_only_call_creates_nothing_on_disk(
        self, docs_root: Path, worktree: Path, tmp_home: Path,
    ):
        self._ws(worktree, tmp_home, _DOCS_ONLY)
        assert not (docs_root / 'docs' / 'prds' / 'new.md').exists()
        assert not (tmp_home / '.cache' / 'uv').exists()
