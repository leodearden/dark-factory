"""Tests for git operations — worktree lifecycle."""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME,
    GitOps,
    ScrubOutcome,
    ScrubResult,
    TrainStackResult,
    WorktreeInfo,
    WorktreeMissing,
    _merge_subject,
    _run,
    scrub_task_dir_from_tree,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _inject_uu_state(cwd: Path, path: str, tag: str = '') -> None:
    """Inject unmerged (stage 1/2/3) index entries for *path* via index surgery.

    Uses ``git hash-object -w --stdin`` to write three blob objects and
    ``git update-index --index-info`` to register them at stages 1, 2, 3.
    The resulting UU entry is detectable by ``_detect_unmerged_paths`` without
    creating an actual conflicting merge commit or setting MERGE_HEAD.

    *tag* is interpolated into the blob content so that multiple calls in the
    same repository produce distinct shas even for different paths.
    """
    def _run_sync(cmd, **kwargs):
        return subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, check=True, **kwargs,
        )

    h1 = _run_sync(
        ['git', 'hash-object', '-w', '--stdin'],
        input=f'version base{tag}\n'.encode(),
    ).stdout.decode().strip()
    h2 = _run_sync(
        ['git', 'hash-object', '-w', '--stdin'],
        input=f'version ours{tag}\n'.encode(),
    ).stdout.decode().strip()
    h3 = _run_sync(
        ['git', 'hash-object', '-w', '--stdin'],
        input=f'version theirs{tag}\n'.encode(),
    ).stdout.decode().strip()

    index_info = (
        f'100644 {h1} 1\t{path}\n'
        f'100644 {h2} 2\t{path}\n'
        f'100644 {h3} 3\t{path}\n'
    )
    _run_sync(['git', 'update-index', '--index-info'], input=index_info.encode())


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # Default push off in tests; TestPushMain enables it explicitly per-case.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


async def _setup_repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare origin repo and a local clone for remote-fetch tests."""
    origin = tmp_path / 'origin.git'
    origin.mkdir()
    await _run(['git', 'init', '--bare', '-b', 'main'], cwd=origin)

    # Seed origin via a temp non-bare repo
    seed = tmp_path / 'seed'
    seed.mkdir()
    await _run(['git', 'init', '-b', 'main'], cwd=seed)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=seed)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=seed)
    (seed / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=seed)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=seed)
    await _run(['git', 'remote', 'add', 'origin', str(origin)], cwd=seed)
    await _run(['git', 'push', 'origin', 'main'], cwd=seed)

    # Clone origin to local
    local = tmp_path / 'local'
    await _run(['git', 'clone', str(origin), str(local)])
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=local)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=local)

    return origin, local


@pytest.fixture
def git_repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Bare origin repo and a local clone with configured user (origin_path, local_path)."""
    return asyncio.run(_setup_repo_with_remote(tmp_path))


@pytest.fixture
def git_ops_with_remote(
    git_config: GitConfig,
    git_repo_with_remote: tuple[Path, Path],
) -> tuple[GitOps, Path]:
    """GitOps against a local clone that has a configured remote (origin)."""
    origin, local = git_repo_with_remote
    return GitOps(git_config, local), origin


async def _push_n_commits_to_origin(
    origin: Path,
    n: int,
    prefix: str = 'remote',
) -> None:
    """Push n new commits to the bare origin repo via a temporary clone."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td) / 'temp_push'
        await _run(['git', 'clone', str(origin), str(temp)])
        await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=temp)
        await _run(['git', 'config', 'user.name', 'Test'], cwd=temp)
        for i in range(n):
            (temp / f'{prefix}_{i}.txt').write_text(f'{prefix} content {i}\n')
            await _run(['git', 'add', '-A'], cwd=temp)
            await _run(['git', 'commit', '-m', f'{prefix} commit {i}'], cwd=temp)
        rc, _, err = await _run(['git', 'push', 'origin', 'main'], cwd=temp)
        assert rc == 0, f'push to bare origin failed: {err}'


def test_git_config_accepts_main_gate_commands():
    """GitConfig.main_gate_mark_command and main_gate_unmark_command default to None and round-trip."""
    # Defaults — feature off (unset => no-op, other projects unaffected)
    cfg_default = GitConfig()
    assert cfg_default.main_gate_mark_command is None
    assert cfg_default.main_gate_unmark_command is None

    # Both set — values round-trip correctly
    cfg = GitConfig(
        main_gate_mark_command='touch /tmp/sentinel',
        main_gate_unmark_command='rm -f /tmp/sentinel',
    )
    assert cfg.main_gate_mark_command == 'touch /tmp/sentinel'
    assert cfg.main_gate_unmark_command == 'rm -f /tmp/sentinel'

    # Only mark set — unmark stays None
    cfg_mark_only = GitConfig(main_gate_mark_command='echo mark')
    assert cfg_mark_only.main_gate_mark_command == 'echo mark'
    assert cfg_mark_only.main_gate_unmark_command is None


@pytest.mark.asyncio
class TestWorktreeLifecycle:
    async def test_worktree_info_stale_commits_field(self, git_ops: GitOps):
        """WorktreeInfo.stale_commits defaults to None and can be set explicitly."""
        info_default = WorktreeInfo(path=git_ops.project_root, base_commit='a' * 40)
        assert info_default.stale_commits is None

        info_explicit = WorktreeInfo(
            path=git_ops.project_root, base_commit='a' * 40, stale_commits=5,
        )
        assert info_explicit.stale_commits == 5

    async def test_worktree_info_reify_debug_port_field(self, git_ops: GitOps):
        """WorktreeInfo.reify_debug_port defaults to None and can be set explicitly."""
        info_default = WorktreeInfo(path=git_ops.project_root, base_commit='a' * 40)
        assert info_default.reify_debug_port is None

        info_explicit = WorktreeInfo(
            path=git_ops.project_root, base_commit='a' * 40, reify_debug_port=39411,
        )
        assert info_explicit.reify_debug_port == 39411

    async def test_create_worktree_provisions_reify_debug_port(self, git_ops: GitOps):
        """create_worktree runs setup-worktree-debug-port.sh and stamps the port."""
        # Commit a fake script into the repo main that just prints the port
        scripts_dir = git_ops.project_root / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        script = scripts_dir / 'setup-worktree-debug-port.sh'
        script.write_text('#!/usr/bin/env bash\necho 39411\n')
        script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'add fake debug-port script'], cwd=git_ops.project_root)

        worktree_info = await git_ops.create_worktree('rdp-1')
        assert worktree_info.reify_debug_port == 39411

    async def test_create_worktree_reify_debug_port_best_effort(
        self, git_ops: GitOps, tmp_path: Path,
    ):
        """_provision_reify_debug_port is fail-open: None for missing/failing/bad-output scripts."""
        scripts_dir = git_ops.project_root / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        script = scripts_dir / 'setup-worktree-debug-port.sh'

        # (a) no script present: default repo — reify_debug_port stays None
        info_no_script = await git_ops.create_worktree('rdp-err-a')
        assert info_no_script.reify_debug_port is None
        assert info_no_script.path.exists()

        # (b) script exits non-zero
        script.write_text('#!/usr/bin/env bash\necho boom >&2\nexit 1\n')
        script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'add failing script'], cwd=git_ops.project_root)
        info_fail = await git_ops.create_worktree('rdp-err-b')
        assert info_fail.reify_debug_port is None
        assert info_fail.path.exists()

        # (c) script prints a non-integer
        script.write_text('#!/usr/bin/env bash\necho not-a-port\n')
        script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'add non-int script'], cwd=git_ops.project_root)
        info_bad = await git_ops.create_worktree('rdp-err-c')
        assert info_bad.reify_debug_port is None
        assert info_bad.path.exists()

    async def test_create_worktree_reuse_provisions_reify_debug_port(
        self, git_ops: GitOps,
    ):
        """Reuse/requeue path also runs the script and stamps the port."""
        scripts_dir = git_ops.project_root / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        script = scripts_dir / 'setup-worktree-debug-port.sh'
        script.write_text('#!/usr/bin/env bash\necho 39411\n')
        script.chmod(0o755)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'add fake debug-port script'], cwd=git_ops.project_root)

        # First call: fresh-create path
        info1 = await git_ops.create_worktree('rdp-reuse')
        assert info1.reify_debug_port == 39411

        # Second call: reuse path
        info2 = await git_ops.create_worktree('rdp-reuse')
        assert info2.reify_debug_port == 39411

    async def test_create_worktree(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-1')
        assert worktree_info.path.exists()
        assert (worktree_info.path / 'README.md').exists()
        assert len(worktree_info.base_commit) == 40

    async def test_create_worktree_returns_worktree_info(self, git_ops: GitOps):
        """create_worktree returns WorktreeInfo with path and base_commit."""
        result = await git_ops.create_worktree('feature-wi')
        assert isinstance(result, WorktreeInfo)
        assert isinstance(result.path, Path)
        assert result.path.exists()
        assert (result.path / 'README.md').exists()
        assert len(result.base_commit) == 40
        # Assert base_commit matches main's HEAD at creation time
        _, main_sha, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root
        )
        assert result.base_commit == main_sha.strip()

    async def test_create_worktree_replaces_stale_directory(self, git_ops: GitOps):
        """A directory that exists but is NOT a registered git worktree must be
        replaced with a real worktree — not silently reused (regression test for
        esc-158-11: stale .task/ dirs mistaken for valid worktrees)."""
        worktree_path = git_ops.worktree_base / 'stale-1'
        worktree_path.mkdir(parents=True, exist_ok=True)
        # Simulate a stale .task/ directory left from a previous run
        task_dir = worktree_path / '.task'
        task_dir.mkdir()
        (task_dir / 'state.json').write_text('{}')

        # The directory exists but is NOT a registered git worktree
        assert worktree_path.exists()

        worktree_info = await git_ops.create_worktree('stale-1')
        # Should have created a real worktree with repo content
        assert worktree_info.path.exists()
        assert (worktree_info.path / 'README.md').exists()
        assert len(worktree_info.base_commit) == 40

    async def test_create_worktree_reuses_registered_worktree(self, git_ops: GitOps):
        """A directory that IS a registered git worktree should be reused."""
        worktree_info = await git_ops.create_worktree('reuse-1')
        assert worktree_info.path.exists()
        assert (worktree_info.path / 'README.md').exists()

        # Call again — should reuse (not fail or recreate)
        worktree_info2 = await git_ops.create_worktree('reuse-1')
        assert worktree_info2.path == worktree_info.path
        assert (worktree_info2.path / 'README.md').exists()

    # ── Fix #3: conservative leftover-branch cleanup ──────────────────────
    # The blind `git branch -D` at create_worktree could destroy a leftover
    # branch carrying commits beyond main, or silently fail when the branch is
    # checked out in a worktree (the 3576 trigger).  The cleanup must prove the
    # leftover is non-destructive to remove before deleting, and escalate (raise)
    # otherwise — never destroying WIP or orphan commits.

    async def test_create_worktree_removes_clean_leftover_branch(
        self, git_ops: GitOps,
    ):
        """A leftover 0-commit branch with no holding worktree (the 3576 shape,
        minus the WIP) is provably non-destructive → removed, worktree created."""
        full_branch = 'task/lo-clean'
        # Dangling ref at main HEAD: no commits beyond main, no worktree.
        rc, _, err = await _run(
            ['git', 'branch', full_branch, 'main'], cwd=git_ops.project_root,
        )
        assert rc == 0, err

        info = await git_ops.create_worktree('lo-clean')

        assert info.path.exists()
        assert (info.path / 'README.md').exists()

    async def test_create_worktree_refuses_leftover_branch_with_commits(
        self, git_ops: GitOps,
    ):
        """A leftover branch carrying a commit beyond main must NOT be deleted —
        raise instead, preserving the branch and its orphan commit."""
        full_branch = 'task/lo-commit'
        # Build the branch with a real commit beyond main via a throwaway
        # worktree, then remove the worktree so the branch is a dangling ref.
        tmp_wt = git_ops.project_root.parent / 'tmp-lo-commit'
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '-b', full_branch, str(tmp_wt), 'main'],
            cwd=git_ops.project_root,
        )
        assert rc == 0, err
        (tmp_wt / 'orphan_work.py').write_text('value = 42\n')
        await _run(['git', 'add', '-A'], cwd=tmp_wt)
        await _run(['git', 'commit', '-m', 'orphan WIP commit'], cwd=tmp_wt)
        _, commit_sha, _ = await _run(['git', 'rev-parse', full_branch], cwd=git_ops.project_root)
        commit_sha = commit_sha.strip()
        # Detach the worktree, leaving a dangling branch with one commit.
        await _run(['git', 'worktree', 'remove', '--force', str(tmp_wt)], cwd=git_ops.project_root)

        with pytest.raises(RuntimeError) as excinfo:
            await git_ops.create_worktree('lo-commit')

        # The branch and its commit must be preserved (NOT deleted).
        rc, sha_after, _ = await _run(['git', 'rev-parse', full_branch], cwd=git_ops.project_root)
        assert rc == 0, 'leftover branch must still exist'
        assert sha_after.strip() == commit_sha, 'orphan commit must be preserved'
        assert full_branch in str(excinfo.value)

    async def test_create_worktree_refuses_leftover_branch_in_dirty_worktree(
        self, git_ops: GitOps,
    ):
        """A leftover branch checked out in a DIRTY worktree must NOT be touched
        — raise with an actionable message; the worktree and its uncommitted
        edits survive intact (the precise 3576 trigger)."""
        full_branch = 'task/lo-dirty'
        holding = git_ops.project_root.parent / 'holding-lo-dirty'
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '-b', full_branch, str(holding), 'main'],
            cwd=git_ops.project_root,
        )
        assert rc == 0, err
        # Uncommitted WIP in the holding worktree.
        dirty_file = holding / 'uncommitted.py'
        dirty_file.write_text('work_in_progress = True\n')

        with pytest.raises(RuntimeError) as excinfo:
            await git_ops.create_worktree('lo-dirty')

        # Actionable error (distinct from the old opaque 'Failed to create
        # worktree: ... already exists') and nothing destroyed.
        assert 'refus' in str(excinfo.value).lower()
        assert dirty_file.exists(), 'uncommitted WIP must survive'
        assert dirty_file.read_text() == 'work_in_progress = True\n'
        rc, _, _ = await _run(['git', 'rev-parse', full_branch], cwd=git_ops.project_root)
        assert rc == 0, 'leftover branch must still exist'

    # ── Fix: worktree-wipe race — canonical-path match + liveness gate ────
    # esc-4146-268: reify's `.worktrees` became a symlink → a 17 TB mount on
    # 2026-05-28.  Worktrees whose admin entry was recorded under the symlink
    # path are listed by `git worktree list` in symlink form, but
    # `_is_registered_worktree` compared the RESOLVED path by exact string —
    # so a live worktree was judged unregistered and shutil.rmtree'd, losing
    # its gitignored .task/plan.json (which git cannot restore).  The fix
    # matches by canonical path on both sides, and the "not registered"
    # branch now refuses to delete anything that looks live.

    async def test_create_worktree_recognizes_symlink_form_registration(
        self, git_config: GitConfig, git_repo: Path, tmp_path: Path,
    ):
        """A worktree whose admin entry was recorded under a SYMLINK path
        (reify's `.worktrees`→mount migration shape) must be recognized as
        registered and REUSED — never wiped.  Core regression for
        esc-4146-268."""
        # Reproduce the migration: register the worktree while `.worktrees`
        # is a REAL dir (git records the plain path), then move that dir to
        # an out-of-tree "mount" and replace it with a symlink so the
        # recorded admin path now traverses a symlink (str != resolved).
        wt_real = git_repo / '.worktrees'
        wt_real.mkdir()
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '-b', 'task/3546',
             str(wt_real / '3546'), 'main'],
            cwd=git_repo,
        )
        assert rc == 0, err
        mount = tmp_path / 'wt_mount'
        wt_real.rename(mount)                       # move real dir (+3546)
        (git_repo / '.worktrees').symlink_to(mount)  # swap in the symlink

        # GitOps resolves worktree_base → the mount, so the path it checks is
        # the RESOLVED form; the old exact-string compare never matched it.
        git_ops = GitOps(git_config, git_repo)
        resolved_path = git_ops.worktree_base / '3546'

        # Guard: git must report the worktree under its SYMLINK form (distinct
        # from the resolved form) — else the test wouldn't reproduce the
        # path-form mismatch that the exact-string compare missed.
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=git_repo,
        )
        listed = [
            ln[len('worktree '):]
            for ln in out.splitlines() if ln.startswith('worktree ')
        ]
        assert any(
            Path(p).resolve() == resolved_path and p != str(resolved_path)
            for p in listed
        ), f'expected a symlink-form admin entry distinct from {resolved_path}:\n{out}'

        # The bug: the resolved path was judged NOT registered.
        assert await git_ops._is_registered_worktree(resolved_path) is True

        # A sentinel proves no rmtree: create_worktree must REUSE in place.
        sentinel = resolved_path / 'plan_sentinel.txt'
        sentinel.write_text('architect plan state\n')
        info = await git_ops.create_worktree('3546')
        assert info.path == resolved_path
        assert sentinel.exists(), \
            'symlink-form worktree must be reused, not rmtree-wiped'
        assert sentinel.read_text() == 'architect plan state\n'

    async def test_create_worktree_refuses_delinked_worktree_with_git_link(
        self, git_ops: GitOps,
    ):
        """A directory holding a `.git` link + source files but NOT registered
        (a worktree whose admin entry was lost — the 3546/3891 shape) must NOT
        be rmtree'd: raise instead, preserving the dir and its gitignored
        .task/ plan state."""
        wt = git_ops.worktree_base / '3546'
        wt.mkdir(parents=True)
        (wt / '.git').write_text('gitdir: /some/repo/.git/worktrees/3546\n')
        (wt / 'module.py').write_text('answer = 42\n')
        task_dir = wt / '.task'
        task_dir.mkdir()
        (task_dir / 'plan.json').write_text('{"plan": "precious"}')

        assert not await git_ops._is_registered_worktree(wt)
        with pytest.raises(RuntimeError) as excinfo:
            await git_ops.create_worktree('3546')

        assert 'refus' in str(excinfo.value).lower()
        # Nothing destroyed — the .git link, source, and plan state survive.
        assert (wt / '.git').exists()
        assert (wt / 'module.py').read_text() == 'answer = 42\n'
        assert (task_dir / 'plan.json').read_text() == '{"plan": "precious"}'

    async def test_create_worktree_refuses_delinked_dir_with_branch_commits(
        self, git_ops: GitOps,
    ):
        """The 4146 shape: the `.git` link is gone and the directory holds
        only .task/ residue, but the task branch still carries committed work.
        The branch-commits discriminator (git-based) must independently gate
        the delete → raise, preserving both the dir and the recoverable
        branch."""
        # Build task/4146 with a commit beyond main via a throwaway worktree,
        # then detach it so the branch is a dangling ref carrying that commit.
        tmp_wt = git_ops.project_root.parent / 'tmp-4146'
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '-b', 'task/4146', str(tmp_wt), 'main'],
            cwd=git_ops.project_root,
        )
        assert rc == 0, err
        (tmp_wt / 'recoverable.py').write_text('important = True\n')
        await _run(['git', 'add', '-A'], cwd=tmp_wt)
        await _run(['git', 'commit', '-m', 'committed work on task/4146'], cwd=tmp_wt)
        _, commit_sha, _ = await _run(
            ['git', 'rev-parse', 'task/4146'], cwd=git_ops.project_root,
        )
        commit_sha = commit_sha.strip()
        await _run(
            ['git', 'worktree', 'remove', '--force', str(tmp_wt)],
            cwd=git_ops.project_root,
        )

        # Leftover dir with NO .git link and only .task/ residue (so the
        # content/`.git` discriminators are both False — branch_has_work is
        # the sole trigger).
        wt = git_ops.worktree_base / '4146'
        wt.mkdir(parents=True)
        task_dir = wt / '.task'
        task_dir.mkdir()
        (task_dir / 'state.json').write_text('{}')

        with pytest.raises(RuntimeError) as excinfo:
            await git_ops.create_worktree('4146')

        assert 'refus' in str(excinfo.value).lower()
        # Dir survives, and the recoverable branch + its commit are untouched.
        assert (task_dir / 'state.json').exists()
        rc, sha_after, _ = await _run(
            ['git', 'rev-parse', 'task/4146'], cwd=git_ops.project_root,
        )
        assert rc == 0 and sha_after.strip() == commit_sha

    async def test_create_worktree_refuses_delinked_dir_with_source_content(
        self, git_ops: GitOps,
    ):
        """Git-independent fail-safe: a non-registered directory holding source
        files (beyond .task/) with no `.git` link and no task branch must still
        raise rather than rmtree.  The content discriminator never depends on a
        git command succeeding, so it holds even under ENOSPC / total git
        failure (the disk-pressure condition observed on reify's mount)."""
        wt = git_ops.worktree_base / 'orphan'
        wt.mkdir(parents=True)
        (wt / 'leftover_source.py').write_text('x = 1\n')

        with pytest.raises(RuntimeError) as excinfo:
            await git_ops.create_worktree('orphan')

        assert 'refus' in str(excinfo.value).lower()
        assert (wt / 'leftover_source.py').read_text() == 'x = 1\n'

    async def test_commit_in_worktree(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-2')
        (worktree_info.path / 'new_file.py').write_text('print("hello")\n')
        sha = await git_ops.commit(worktree_info.path, 'Add new file')
        assert sha is not None
        assert len(sha) == 40

    async def test_commit_nothing(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-3')
        sha = await git_ops.commit(worktree_info.path, 'Nothing')
        assert sha is None

    async def test_diff_from_main(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-4')
        (worktree_info.path / 'change.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add change')
        diff = await git_ops.get_diff_from_main(worktree_info.path)
        assert 'change.py' in diff
        assert 'x = 1' in diff

    async def test_diff_from_base(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-4b')
        (worktree_info.path / 'base_change.py').write_text('y = 2\n')
        await git_ops.commit(worktree_info.path, 'Add base change')
        diff = await git_ops.get_diff_from_base(worktree_info.path, worktree_info.base_commit)
        assert 'base_change.py' in diff
        assert 'y = 2' in diff

    async def test_diff_from_base_stable_when_main_advances(self, git_ops: GitOps):
        """get_diff_from_base returns branch changes even after main advances.

        This is the key test for the fix: when main advances during task execution,
        get_diff_from_base must still return the branch's changes by using the
        pinned base_commit instead of the moving main ref.
        """
        # Create worktree and capture base_commit
        worktree_info = await git_ops.create_worktree('feature-adv')
        base_commit = worktree_info.base_commit

        # Make a commit in the branch
        (worktree_info.path / 'branch_change.py').write_text('z = 3\n')
        await git_ops.commit(worktree_info.path, 'Add branch change')

        # Advance main with a separate commit (simulating another task merging)
        (git_ops.project_root / 'main_change.py').write_text('x = 1\n')
        await _run(['git', 'add', 'main_change.py'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Advance main'], cwd=git_ops.project_root)

        # get_diff_from_base should still return branch changes
        diff = await git_ops.get_diff_from_base(worktree_info.path, base_commit)
        assert 'branch_change.py' in diff
        assert 'z = 3' in diff

        # Contrast: get_diff_from_main might return empty/different (main absorbed branch)
        # This demonstrates that base_commit is needed for stable diffs

    async def test_cleanup_worktree(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-5')
        assert worktree_info.path.exists()
        await git_ops.cleanup_worktree(worktree_info.path, 'feature-5')
        assert not worktree_info.path.exists()

    async def test_rename_worktree_moves_path_and_branch(
        self, git_ops: GitOps,
    ):
        info = await git_ops.create_worktree('orig-task')
        old_path = info.path
        new_path = git_ops.worktree_base / 'orig-task-skip-attempt'

        await git_ops.rename_worktree(
            old_path=old_path,
            new_path=new_path,
            old_branch='orig-task',
            new_branch='orig-task-skip-attempt',
        )

        assert not old_path.exists()
        assert new_path.exists()

        # Branch was renamed too
        rc, branches, _ = await _run(
            ['git', 'branch', '--list'],
            cwd=git_ops.project_root,
        )
        assert rc == 0
        assert 'task/orig-task-skip-attempt' in branches
        assert 'task/orig-task ' not in branches  # exact-match exclusion

        # New worktree is registered with git
        rc, listing, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        assert str(new_path.resolve()) in listing

    async def test_rename_worktree_unregistered_path_raises(
        self, git_ops: GitOps,
    ):
        # An unregistered directory (not a real worktree) cannot be moved.
        bogus_old = git_ops.worktree_base / 'never-registered'
        bogus_old.mkdir(parents=True)
        new_path = git_ops.worktree_base / 'should-not-exist'

        with pytest.raises(RuntimeError, match='git worktree move'):
            await git_ops.rename_worktree(
                old_path=bogus_old,
                new_path=new_path,
                old_branch='never-registered',
                new_branch='renamed',
            )

    async def test_merge_to_main(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-6')
        (worktree_info.path / 'merged.py').write_text('merged = True\n')
        await git_ops.commit(worktree_info.path, 'Add merged file')

        result = await git_ops.merge_to_main(worktree_info.path, 'feature-6')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        assert result.merge_worktree != git_ops.project_root

        # Merge worktree has the file
        assert (result.merge_worktree / 'merged.py').exists()

        # Main ref not advanced yet — project_root working tree untouched
        assert not (git_ops.project_root / 'merged.py').exists()

        # Advance main and verify
        assert await git_ops.advance_main(result.merge_commit) == 'advanced'
        _, content, _ = await _run(
            ['git', 'show', 'main:merged.py'], cwd=git_ops.project_root,
        )
        assert 'merged = True' in content

        # File should also be in the working tree (working tree synced)
        assert (git_ops.project_root / 'merged.py').exists()

        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        assert not result.merge_worktree.exists()

    async def test_advance_main_rejects_non_ancestor(self, git_ops: GitOps):
        """advance_main rejects a SHA that isn't a descendant of main."""
        # Use a commit from a branch that hasn't been merged
        worktree_info = await git_ops.create_worktree('orphan')
        (worktree_info.path / 'orphan.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Orphan commit')

        # Advance main to a different commit first
        worktree_info2 = await git_ops.create_worktree('advance-first')
        (worktree_info2.path / 'first.py').write_text('y = 1\n')
        await git_ops.commit(worktree_info2.path, 'First commit')
        result = await git_ops.merge_to_main(worktree_info2.path, 'advance-first')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        await git_ops.advance_main(result.merge_commit)
        await git_ops.cleanup_merge_worktree(result.merge_worktree)

        # Now the orphan branch's commit is NOT a descendant of new main
        _, orphan_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree_info.path,
        )
        assert await git_ops.advance_main(orphan_sha) == 'not_descendant'

    async def test_get_current_branch(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-7')
        branch = await git_ops.get_current_branch(worktree_info.path)
        assert branch == 'task/feature-7'

    async def test_merge_to_main_cleans_worktree_on_cancellation(
        self, git_ops: GitOps,
    ):
        """merge_to_main must clean up the merge worktree on CancelledError.

        Covers review issue [resource_leak_on_cancellation] at git_ops.py:495.
        The cleanup guard uses ``except Exception:`` which does NOT catch
        ``asyncio.CancelledError`` (a BaseException subclass).  This test
        fails with the old guard and passes with ``except BaseException:``.
        """
        # Set up a feature branch with a committed file.
        worktree_info = await git_ops.create_worktree('feature-cancel')
        (worktree_info.path / 'cancel_test.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add cancel test file')

        # Patch scrub_task_dir_from_tree to raise CancelledError, simulating
        # task cancellation at the point where the merge commit already exists
        # but cleanup has not yet been called.
        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            side_effect=asyncio.CancelledError,
        ), pytest.raises(asyncio.CancelledError):
            await git_ops.merge_to_main(worktree_info.path, 'feature-cancel')

        # After CancelledError, no _merge-* worktrees should be registered.
        _, worktree_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        leak_lines = [
            line for line in worktree_list.splitlines()
            if '_merge-' in line
        ]
        assert not leak_lines, (
            f'Leaked merge worktrees still registered: {leak_lines}'
        )

        # Also confirm no _merge-* directories exist on disk.
        worktree_base = git_ops.worktree_base
        if worktree_base.exists():
            leak_dirs = list(worktree_base.glob('_merge-*'))
            assert not leak_dirs, (
                f'Leaked merge worktree directories on disk: {leak_dirs}'
            )

    async def test_create_worktree_base_commit_race_safe_new_path(
        self, git_ops: GitOps,
    ):
        """Race regression: main advances between rev-parse and `worktree add`.

        Simulates fused-memory's ``TaskFileCommitter._schedule_commit`` racing
        against ``create_worktree``: the pre-positioning ``git rev-parse main``
        captures SHA-A, then a fire-and-forget commit advances main to SHA-B
        before ``git worktree add`` runs.  ``git worktree add ... main``
        resolves the symbolic ref at call time, so the new worktree HEAD is at
        SHA-B.  ``WorktreeInfo.base_commit`` MUST reflect the post-positioning
        fork point (SHA-B), not the pre-race captured SHA (SHA-A).

        Without the fix, downstream callers like ``_recover_if_already_merged``
        observe ``wt_head (SHA-B) != base_commit (SHA-A)`` on a worktree that
        has done zero implementation work, and silently mark the task DONE.
        See ``~/.claude/plans/do-2-3-misty-marshmallow.md``.
        """
        from orchestrator import git_ops as git_ops_mod

        real_run = git_ops_mod._run
        advance_state = {'fired': False, 'pre_race_sha': None}

        async def racing_run(cmd, cwd=None):
            # Detect the pre-positioning rev-parse on main and inject a
            # racing commit between its return and the subsequent
            # `git worktree add ... main`.  Only fire once so the
            # subsequent merge-base lookup runs unmodified.
            is_pre_positioning_revparse = (
                cmd[:2] == ['git', 'rev-parse']
                and len(cmd) == 3
                and cmd[2] == git_ops.config.main_branch
                and cwd == git_ops.project_root
                and not advance_state['fired']
            )
            result = await real_run(cmd, cwd=cwd)
            if is_pre_positioning_revparse:
                advance_state['fired'] = True
                advance_state['pre_race_sha'] = result[1].strip()
                # Inject the race: advance main with a new commit before
                # `git worktree add` runs.  This mirrors any concurrent
                # commit landing during the rev-parse → worktree-add window.
                (git_ops.project_root / 'racing_commit.txt').write_text(
                    'committed during create_worktree race window\n'
                )
                await real_run(
                    ['git', 'add', 'racing_commit.txt'],
                    cwd=git_ops.project_root,
                )
                await real_run(
                    ['git', 'commit', '-m', 'race: advance main mid-create_worktree'],
                    cwd=git_ops.project_root,
                )
            return result

        with patch('orchestrator.git_ops._run', side_effect=racing_run):
            wt_info = await git_ops.create_worktree('race-new-path')

        # Sanity: the race actually fired
        assert advance_state['fired'], (
            'Race injection never fired — test setup is broken'
        )

        # Read post-race state
        _, post_race_main_sha, _ = await _run(
            ['git', 'rev-parse', git_ops.config.main_branch],
            cwd=git_ops.project_root,
        )
        post_race_main_sha = post_race_main_sha.strip()
        _, wt_head_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=wt_info.path,
        )
        wt_head_sha = wt_head_sha.strip()

        # The pre-race SHA differs from the post-race main SHA — race is real
        assert advance_state['pre_race_sha'] != post_race_main_sha, (
            'Pre-race SHA must differ from post-race main SHA — race injection '
            'did not actually advance main'
        )

        # Worktree HEAD lands on the new main (`git worktree add` resolves
        # `main` at call time)
        assert wt_head_sha == post_race_main_sha, (
            f'Precondition: wt_head ({wt_head_sha[:8]}) must equal post-race '
            f'main ({post_race_main_sha[:8]}) since `git worktree add` '
            f'resolves the ref at call time'
        )

        # Race-immunity guarantee: base_commit equals the worktree's actual
        # fork point (= new main = wt_head), NOT the pre-race captured SHA
        assert wt_info.base_commit == wt_head_sha, (
            f'base_commit ({wt_info.base_commit[:8]}) must equal post-positioning '
            f'merge-base/HEAD ({wt_head_sha[:8]}), not the pre-race captured SHA '
            f'({advance_state["pre_race_sha"][:8]}). The fix in create_worktree '
            f'must compute base_commit from `git merge-base main HEAD` inside '
            f'the worktree AFTER positioning, not from rev-parse before it.'
        )
        assert wt_info.base_commit != advance_state['pre_race_sha'], (
            'base_commit must NOT equal the pre-race SHA — that is the bug'
        )

    async def test_create_worktree_base_commit_race_safe_reused_path(
        self, git_ops: GitOps,
    ):
        """Race regression for the reused-worktree path: main advances during rebase.

        Setup mirrors a requeued task: the worktree already exists with a
        real branch commit, main has advanced once, and a SECOND advance
        races the reused-path rev-parse → rebase window.

        After ``rebase_onto_main`` completes, ``WorktreeInfo.base_commit``
        must equal the post-rebase fork point (the new main SHA), not the
        pre-race captured SHA from line 478.  The rebase brings the branch
        onto current main (resolved at rebase time), so the worktree's
        merge-base with main equals the post-rebase main SHA.

        Without the fix, ``actual_base = base_sha.strip()`` on the success
        branch returns the stale pre-race SHA — producing the same false-
        positive in ``_recover_if_already_merged`` as the new-path bug.
        """
        from orchestrator import git_ops as git_ops_mod

        # 1. Pre-create the worktree and make a real branch commit
        wt_info1 = await git_ops.create_worktree('race-reused')
        wt = wt_info1.path
        (wt / 'branch_work.py').write_text('z = 3\n')
        await git_ops.commit(wt, 'Branch work for race-reused test')

        # 2. Advance main once via a separate commit (simulates an unrelated merge)
        (git_ops.project_root / 'first_advance.py').write_text('a = 1\n')
        await _run(['git', 'add', 'first_advance.py'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'first main advance'],
            cwd=git_ops.project_root,
        )

        # 3. Set up race injection for the second advance
        real_run = git_ops_mod._run
        advance_state = {'fired': False, 'pre_race_sha': None}

        async def racing_run(cmd, cwd=None):
            is_pre_positioning_revparse = (
                cmd[:2] == ['git', 'rev-parse']
                and len(cmd) == 3
                and cmd[2] == git_ops.config.main_branch
                and cwd == git_ops.project_root
                and not advance_state['fired']
            )
            result = await real_run(cmd, cwd=cwd)
            if is_pre_positioning_revparse:
                advance_state['fired'] = True
                advance_state['pre_race_sha'] = result[1].strip()
                # Inject a SECOND advance to main — between the rev-parse
                # in create_worktree and the rebase_onto_main inside it.
                (git_ops.project_root / 'racing_advance.py').write_text(
                    'b = 2\n'
                )
                await real_run(
                    ['git', 'add', 'racing_advance.py'],
                    cwd=git_ops.project_root,
                )
                await real_run(
                    ['git', 'commit', '-m', 'race: advance main mid-reuse'],
                    cwd=git_ops.project_root,
                )
            return result

        # 4. Call create_worktree again → reused-worktree path → rebase →
        #    new merge-base capture.
        with patch('orchestrator.git_ops._run', side_effect=racing_run):
            wt_info2 = await git_ops.create_worktree('race-reused')

        assert advance_state['fired'], (
            'Race injection never fired — test setup is broken'
        )
        assert wt_info2.path == wt_info1.path, (
            'Reused-worktree path must return the same worktree path'
        )

        # 5. Compute expected post-rebase fork point and compare
        _, post_race_main_sha, _ = await _run(
            ['git', 'rev-parse', git_ops.config.main_branch],
            cwd=git_ops.project_root,
        )
        post_race_main_sha = post_race_main_sha.strip()

        _, mb_actual, _ = await _run(
            ['git', 'merge-base', git_ops.config.main_branch, 'HEAD'],
            cwd=wt,
        )
        mb_actual = mb_actual.strip()

        # The pre-race SHA differs from the post-race main SHA
        assert advance_state['pre_race_sha'] != post_race_main_sha, (
            'Pre-race SHA must differ from post-race main SHA — race injection '
            'did not actually advance main'
        )

        # After a successful rebase, the branch's fork point with main is
        # current main.
        assert mb_actual == post_race_main_sha, (
            f'Precondition: post-rebase merge-base ({mb_actual[:8]}) must '
            f'equal post-race main ({post_race_main_sha[:8]})'
        )

        # Race-immunity: base_commit equals the post-rebase fork point, NOT
        # the pre-race captured SHA.
        assert wt_info2.base_commit == mb_actual, (
            f'base_commit ({wt_info2.base_commit[:8]}) must equal '
            f'post-rebase merge-base ({mb_actual[:8]}), not the pre-race '
            f'captured SHA ({advance_state["pre_race_sha"][:8]}). The fix in '
            f'create_worktree must promote the merge-base capture to the '
            f'rebase-success branch as well as the rebase-failed branch.'
        )
        assert wt_info2.base_commit != advance_state['pre_race_sha'], (
            'base_commit must NOT equal the pre-race SHA — that is the bug'
        )

    async def test_merge_to_main_absent_ref_falls_back_to_worktree_head(
        self, git_ops: GitOps,
    ):
        """merge_to_main falls back to the worktree HEAD when task/<id> ref is absent.

        Scenario: the named ref 'task/orphan' is deleted after the worktree's
        HEAD carries a commit beyond main (ref absent but work is present).
        merge_to_main must still succeed, merging via the worktree HEAD SHA, and
        the merge-commit subject must be 'Merge task/orphan into main' so that
        find_merge_marker keeps already_merged idempotency on re-dispatch.

        Fails today: absent ref → git merge --no-ff task/orphan on a non-existent
        ref → MergeResult.success=False.
        """
        # Create a worktree and commit a file (this creates refs/heads/task/orphan)
        wt = (await git_ops.create_worktree('orphan')).path
        (wt / 'orphan_work.py').write_text('orphan = True\n')
        await git_ops.commit(wt, 'Add orphan work file')

        # Detach HEAD so we can safely delete the branch ref without
        # git complaining about the worktree's current branch being deleted.
        await _run(['git', 'checkout', '--detach'], cwd=wt)

        # Delete the named branch ref — the worktree HEAD still carries commits
        await _run(
            ['git', 'branch', '-D', 'task/orphan'], cwd=git_ops.project_root,
        )

        # Sanity: confirm the ref is truly absent
        assert await git_ops.resolve_branch_sha('task/orphan') is None

        # merge_to_main must succeed using the worktree HEAD as the merge source
        result = await git_ops.merge_to_main(wt, 'orphan')

        assert result.success is True, f'Expected success but got: {result}'
        assert result.merge_commit is not None

        # Advance main so the merge commit lands
        advanced = await git_ops.advance_main(result.merge_commit)
        assert advanced == 'advanced'

        # The orphan_work.py file must now be on main (content merged correctly)
        rc, content, err = await _run(
            ['git', 'show', 'main:orphan_work.py'], cwd=git_ops.project_root,
        )
        assert rc == 0, f'File not found on main: {err}'
        assert 'orphan = True' in content

        # The merge-commit subject must be canonical 'Merge task/orphan into main'
        # so that find_merge_marker works on re-dispatch (already_merged idempotency).
        _, commit_msg, _ = await _run(
            ['git', 'log', '-1', '--format=%s', result.merge_commit.strip()],
            cwd=git_ops.project_root,
        )
        assert commit_msg.strip() == 'Merge task/orphan into main', (
            f'Unexpected merge-commit subject: {commit_msg.strip()!r}'
        )

        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)


@pytest.mark.asyncio
class TestFreshenMain:
    async def test_freshen_main_no_remote(self, git_ops: GitOps):
        """Without a remote, _freshen_main returns (main_branch, None)."""
        ref, stale = await git_ops._freshen_main()
        assert ref == git_ops.config.main_branch
        assert stale is None

    async def test_freshen_main_remote_ahead(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When origin/main is 3 commits ahead, returns ('origin/main', 3)."""
        git_ops, origin = git_ops_with_remote
        await _push_n_commits_to_origin(origin, 3)
        ref, stale = await git_ops._freshen_main()
        assert ref == f'{git_ops.config.remote}/{git_ops.config.main_branch}'
        assert stale == 3

    async def test_freshen_main_already_current(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When local main == origin/main (no new commits), returns (main_branch, 0)."""
        git_ops, _origin = git_ops_with_remote
        ref, stale = await git_ops._freshen_main()
        assert ref == git_ops.config.main_branch
        assert stale == 0

    async def test_freshen_main_diverged(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When local and remote have diverged, returns (main_branch, N) with N behind count."""
        git_ops, origin = git_ops_with_remote
        local = git_ops.project_root
        # Add a local-only commit (not pushed to origin)
        (local / 'local_only.txt').write_text('local only\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'Local only commit'], cwd=local)
        # Add a different commit to origin (creates divergence)
        await _push_n_commits_to_origin(origin, 1, prefix='remote_div')
        ref, stale = await git_ops._freshen_main()
        # Diverged: use local ref to avoid losing advance_main commits
        assert ref == git_ops.config.main_branch
        assert stale == 1

    async def test_freshen_main_behind_rev_list_fails(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When behind rev-list exits non-zero, _freshen_main returns (main_branch, None)."""
        git_ops, _origin = git_ops_with_remote

        async def fake_run(cmd, cwd=None):
            if 'fetch' in cmd:
                return (0, '', '')          # fetch succeeds
            return (128, '', 'fatal: bad revision')   # rev-list fails

        with patch('orchestrator.git_ops._run', side_effect=fake_run):
            ref, stale = await git_ops._freshen_main()

        assert ref == git_ops.config.main_branch
        assert stale is None

    async def test_freshen_main_behind_count_value_error(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When behind rev-list returns non-numeric stdout, _freshen_main returns (main_branch, None)."""
        git_ops, _origin = git_ops_with_remote

        async def fake_run(cmd, cwd=None):
            if 'fetch' in cmd:
                return (0, '', '')           # fetch succeeds
            return (0, 'not-a-number', '')   # rev-list returns garbage

        with patch('orchestrator.git_ops._run', side_effect=fake_run):
            ref, stale = await git_ops._freshen_main()

        assert ref == git_ops.config.main_branch
        assert stale is None

    async def test_freshen_main_ahead_count_value_error(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When ahead rev-list returns non-numeric stdout, falls back to (main_branch, behind)."""
        git_ops, _origin = git_ops_with_remote

        remote_ref = f'{git_ops.config.remote}/'

        async def fake_run(cmd, cwd=None):
            if 'fetch' in cmd:
                return (0, '', '')           # fetch succeeds
            if 'rev-list' in cmd:
                # Distinguish behind vs ahead by which side of '..' the remote ref is on:
                #   behind range: <local>..<remote>  (e.g. main..origin/main)
                #   ahead  range: <remote>..<local>  (e.g. origin/main..main)
                range_arg = next((arg for arg in cmd if '..' in arg), '')
                if range_arg.startswith(remote_ref):
                    return (0, 'not-a-number', '')  # ahead rev-list: garbage
                return (0, '3', '')                 # behind rev-list: 3 behind
            return (0, '', '')

        with patch('orchestrator.git_ops._run', side_effect=fake_run):
            ref, stale = await git_ops._freshen_main()

        # Falls back to local main; reports behind count as-is
        assert ref == git_ops.config.main_branch
        assert stale == 3

    async def test_freshen_main_ahead_rev_list_fails(
        self, git_ops_with_remote: tuple[GitOps, Path], caplog,
    ):
        """When ahead rev-list exits non-zero, _freshen_main returns (main_branch, behind) and logs a warning."""
        git_ops, _origin = git_ops_with_remote

        call_count = 0

        async def fake_run(cmd, cwd=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (0, '', '')                      # fetch succeeds
            elif call_count == 2:
                return (0, '3', '')                     # behind rev-list: 3 commits behind
            return (128, '', 'fatal: bad revision')     # ahead rev-list fails

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'), \
             patch('orchestrator.git_ops._run', side_effect=fake_run):
            ref, stale = await git_ops._freshen_main()

        assert ref == git_ops.config.main_branch
        assert stale == 3
        assert any('rev-list (ahead) failed' in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestCreateWorktreeFreshening:
    async def test_create_worktree_freshens_from_remote(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """Worktree based on remote ref when origin is ahead — new file is present."""
        git_ops, origin = git_ops_with_remote
        await _push_n_commits_to_origin(origin, 1, prefix='fresh')
        worktree_info = await git_ops.create_worktree('freshen-test')
        assert (worktree_info.path / 'fresh_0.txt').exists()
        # Exactly 1 commit was pushed, so stale_commits must reflect that.
        assert worktree_info.stale_commits == 1
        # base_commit must match origin/main SHA captured after create_worktree
        # (which internally fetched, so origin/main is now up-to-date in the local repo).
        _, expected_sha, _ = await _run(
            ['git', 'rev-parse', 'origin/main'], cwd=git_ops.project_root,
        )
        assert worktree_info.base_commit == expected_sha

    async def test_create_worktree_stale_commits_populated(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """stale_commits == 2 when origin is 2 commits ahead at create_worktree time."""
        git_ops, origin = git_ops_with_remote
        await _push_n_commits_to_origin(origin, 2)
        worktree_info = await git_ops.create_worktree('stale-commits-test')
        assert worktree_info.stale_commits == 2

    async def test_create_worktree_stale_commits_none_without_remote(
        self, git_ops: GitOps,
    ):
        """stale_commits is None when no remote is configured (graceful degradation)."""
        worktree_info = await git_ops.create_worktree('no-remote-test')
        assert worktree_info.stale_commits is None

    async def test_create_worktree_revparse_fallback(self, git_ops: GitOps):
        """When rev-parse of start_ref fails, create_worktree falls back to local main.

        _freshen_main returns 'origin/nonexistent-ref' (a ref that doesn't exist
        in this no-remote repo). The rev-parse should fail, triggering a fallback
        to local main. The worktree should still be created successfully with a
        valid base_commit SHA.
        """
        _, local_main_sha, _ = await _run(
            ['git', 'rev-parse', git_ops.config.main_branch],
            cwd=git_ops.project_root,
        )
        local_main_sha = local_main_sha.strip()

        with patch.object(
            git_ops, '_freshen_main', return_value=('origin/nonexistent-ref', 3),
        ):
            worktree_info = await git_ops.create_worktree('revparse-fallback-test')

        assert (worktree_info.path / 'README.md').exists()
        assert len(worktree_info.base_commit) == 40
        assert worktree_info.base_commit == local_main_sha
        assert worktree_info.stale_commits == 3  # persists through fallback


@pytest.mark.asyncio
class TestMergeConflicts:
    async def test_conflict_detection(self, git_ops: GitOps):
        # Create BOTH branches before merging either (both fork from same main)
        wt_a_info = await git_ops.create_worktree('branch-a')
        wt_b_info = await git_ops.create_worktree('branch-b')

        # Both modify same file differently
        (wt_a_info.path / 'shared.py').write_text('value = "A"\n')
        await git_ops.commit(wt_a_info.path, 'Branch A change')

        (wt_b_info.path / 'shared.py').write_text('value = "B"\n')
        await git_ops.commit(wt_b_info.path, 'Branch B change')

        # Merge A first — should succeed
        result_a = await git_ops.merge_to_main(wt_a_info.path, 'branch-a')
        assert result_a.success
        assert result_a.merge_commit is not None
        assert result_a.merge_worktree is not None
        await git_ops.advance_main(result_a.merge_commit)
        await git_ops.cleanup_merge_worktree(result_a.merge_worktree)

        # Merge B — should conflict (main now has "A", branch has "B")
        result_b = await git_ops.merge_to_main(wt_b_info.path, 'branch-b')
        assert not result_b.success
        assert result_b.conflicts
        assert result_b.merge_worktree is not None


@pytest.mark.asyncio
class TestAdvanceMainCasRetrySha:
    """Regression: advance_main must expose the post-rebase SHA when CAS retry rebases."""

    async def test_last_advanced_sha_is_post_rebase_after_cas_retry(
        self, git_ops: GitOps,
    ):
        """When advance_main rebases the merge worktree onto a moved main,
        ``_last_advanced_sha`` must hold the post-rebase SHA (the one actually
        on main), not the original pre-rebase ``MergeResult.merge_commit``.

        The pre-rebase SHA is what merge_queue used to forward to
        ``set_task_status('done', done_provenance={'kind':'merged', 'commit':...})``;
        fused-memory's ancestor backstop rejects it because that SHA only
        exists in the now-discarded merge worktree, not on main. Result was
        56+ tasks stuck in-progress in reify on 2026-04-27.
        """
        # Pin the initial main SHA — B's merge worktree will be based here,
        # simulating a speculative-merge stack where N+1's merge was prepared
        # before N landed.
        _, original_main_sha, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        original_main_sha = original_main_sha.strip()

        # Two branches that touch different files (no conflict on rebase).
        wt_a = await git_ops.create_worktree('cas-a')
        (wt_a.path / 'a.py').write_text('a = 1\n')
        await git_ops.commit(wt_a.path, 'Add a')

        wt_b = await git_ops.create_worktree('cas-b')
        (wt_b.path / 'b.py').write_text('b = 1\n')
        await git_ops.commit(wt_b.path, 'Add b')

        # Land A first — main now has A's merge commit.
        merge_a = await git_ops.merge_to_main(wt_a.path, 'cas-a')
        assert merge_a.success and merge_a.merge_commit and merge_a.merge_worktree
        assert await git_ops.advance_main(merge_a.merge_commit) == 'advanced'
        await git_ops.cleanup_merge_worktree(merge_a.merge_worktree)

        # Merge B with merge worktree pinned to the ORIGINAL main — so B's
        # merge commit will not be a descendant of current main and
        # advance_main must rebase to land it.
        merge_b = await git_ops.merge_to_main(
            wt_b.path, 'cas-b', base_sha=original_main_sha,
        )
        assert merge_b.success and merge_b.merge_commit and merge_b.merge_worktree
        pre_rebase_sha = merge_b.merge_commit

        result = await git_ops.advance_main(
            pre_rebase_sha,
            merge_worktree=merge_b.merge_worktree,
            branch='cas-b',
        )
        assert result == 'advanced'

        # Side channel exposes a SHA that's actually on main.
        advanced_sha = git_ops._last_advanced_sha
        assert advanced_sha is not None

        rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor',
             advanced_sha, git_ops.config.main_branch],
            cwd=git_ops.project_root,
        )
        assert rc == 0, f'_last_advanced_sha {advanced_sha} must be on main'

        # And the pre-rebase SHA is NOT on main (the bug we're guarding
        # against — passing this to done_provenance fails fused-memory's
        # ancestor backstop).
        rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor',
             pre_rebase_sha, git_ops.config.main_branch],
            cwd=git_ops.project_root,
        )
        assert rc != 0, (
            f'pre-rebase SHA {pre_rebase_sha} should NOT be on main; '
            f'if it is, the test is no longer exercising the rebase path'
        )

        await git_ops.cleanup_merge_worktree(merge_b.merge_worktree)

    async def test_last_advanced_sha_equals_input_when_no_rebase_needed(
        self, git_ops: GitOps,
    ):
        """When no CAS retry is needed, ``_last_advanced_sha`` matches the input
        merge_sha — fast-forward case, both SHAs are on main.
        """
        wt = await git_ops.create_worktree('ff-only')
        (wt.path / 'x.py').write_text('x = 1\n')
        await git_ops.commit(wt.path, 'Add x')
        merge = await git_ops.merge_to_main(wt.path, 'ff-only')
        assert merge.success and merge.merge_commit
        assert await git_ops.advance_main(merge.merge_commit) == 'advanced'
        assert git_ops._last_advanced_sha == merge.merge_commit


# -- Shared helpers for re-merge-fallback tests --------------------------------


async def _build_remerge_scenario(
    git_ops: GitOps,
    branch_a: str,
    branch_b: str,
):
    """Set up the 'A lands, B not a descendant' re-merge fallback scenario.

    Creates two non-conflicting branches (a.py / b.py), builds B's merge
    worktree against the ORIGINAL main (so B is not a descendant of main
    after A lands), captures verified_tip = M^2, lands A, and returns
    (merge_b, verified_tip, wt_b).  The caller may add a post-verify commit
    to wt_b.path before calling advance_main to simulate a moved branch ref.
    """
    _, original_main_sha, _ = await _run(
        ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
    )
    original_main_sha = original_main_sha.strip()

    wt_a = await git_ops.create_worktree(branch_a)
    (wt_a.path / 'a.py').write_text('a = 1\n')
    await git_ops.commit(wt_a.path, 'Add a')

    wt_b = await git_ops.create_worktree(branch_b)
    (wt_b.path / 'b.py').write_text('b = 1\n')
    await git_ops.commit(wt_b.path, 'Add b')

    merge_b = await git_ops.merge_to_main(
        wt_b.path, branch_b, base_sha=original_main_sha,
    )
    assert merge_b.success and merge_b.merge_commit and merge_b.merge_worktree

    _, verified_tip_raw, _ = await _run(
        ['git', 'rev-parse', f'{merge_b.merge_commit}^2'],
        cwd=merge_b.merge_worktree,
    )
    verified_tip = verified_tip_raw.strip()
    assert verified_tip, 'M^2 must be resolvable from the merge worktree'

    merge_a = await git_ops.merge_to_main(wt_a.path, branch_a)
    assert merge_a.success and merge_a.merge_commit and merge_a.merge_worktree
    assert await git_ops.advance_main(merge_a.merge_commit) == 'advanced'
    await git_ops.cleanup_merge_worktree(merge_a.merge_worktree)

    return merge_b, verified_tip, wt_b


async def _failing_rebase_run(cmd, cwd=None):
    """Selective _run wrapper: forces 'git rebase main' to return rc=1.

    Used to deterministically route advance_main into the re-merge fallback.
    All other git commands are delegated to the real _run.
    """
    if cmd[:3] == ['git', 'rebase', 'main']:
        return (1, '', 'forced rebase failure for test')
    return await _run(cmd, cwd=cwd)


@pytest.mark.asyncio
class TestAdvanceMainRemergePin:
    """Regression: the re-merge fallback must pin to M^2 (verified branch tip),
    not the live branch ref, to prevent post-verify commits from landing on main.

    Reference: esc-1657-26 — advance_main's fallback re-merged a moving branch
    ref incorporating unverified commits committed during the verify/advance window.
    """

    async def test_advance_main_remerge_pins_to_verified_branch_tip(
        self, git_ops: GitOps,
    ):
        """When the rebase fallback runs, the landed tree must match M^2 (the
        verified branch tip), not the current live branch ref.

        Post-verify commit (stale.py) is pushed to B's branch, moving the live
        ref past M^2.  The rebase is forced to fail (deterministic fallback path).

        Assert: result == 'advanced' AND stale.py is absent from main
        (the advance must pin to M^2, not pick up the moved live ref).
        """
        merge_b, verified_tip, wt_b = await _build_remerge_scenario(
            git_ops, 'pin-a', 'pin-b',
        )
        assert merge_b.merge_commit is not None and merge_b.merge_worktree is not None

        # Post-verify commit: moves live ref (task/pin-b) past M^2.
        (wt_b.path / 'stale.py').write_text('stale = True\n')
        await git_ops.commit(wt_b.path, 'Post-verify stale commit')

        with patch('orchestrator.git_ops._run', side_effect=_failing_rebase_run):
            result = await git_ops.advance_main(
                merge_b.merge_commit,
                merge_worktree=merge_b.merge_worktree,
                branch='pin-b',
            )

        assert result == 'advanced', f'Expected advanced, got {result!r}'

        # KEY assertion: stale.py must NOT be in main — the advance must have
        # used M^2, not the moved live ref (which includes stale.py).
        _, tree_out, _ = await _run(
            ['git', 'ls-tree', '-r', 'main', '--name-only'],
            cwd=git_ops.project_root,
        )
        assert 'stale.py' not in tree_out, (
            f'stale.py must NOT land on main. '
            f'verified_tip={verified_tip[:8]}, '
            f'main tree: {tree_out!r}'
        )

        await git_ops.cleanup_merge_worktree(merge_b.merge_worktree)


@pytest.mark.asyncio
class TestAdvanceMainDivergenceWarning:
    """Regression canary: advance_main must emit a structured WARNING when the
    live branch ref has moved past the verified M^2 tip during the re-merge
    fallback (stale-tip divergence is self-evident in logs).

    Reference: esc-1657-26.
    """

    async def test_advance_main_warns_on_stale_branch_ref_divergence(
        self, git_ops: GitOps, caplog,
    ):
        """When the live branch ref has advanced past verified M^2, a WARNING
        is emitted recording both verified_branch_tip and the live ref tip.
        """
        merge_b, verified_tip, wt_b = await _build_remerge_scenario(
            git_ops, 'div-a', 'div-b',
        )
        assert merge_b.merge_commit is not None and merge_b.merge_worktree is not None

        # Post-verify commit advances live ref past M^2 — triggers divergence.
        (wt_b.path / 'stale.py').write_text('stale = True\n')
        await git_ops.commit(wt_b.path, 'Post-verify stale commit')

        # Resolve the live ref tip for assertion.
        _, live_tip_raw, _ = await _run(
            ['git', 'rev-parse', 'task/div-b'],
            cwd=git_ops.project_root,
        )
        live_tip = live_tip_raw.strip()
        assert live_tip != verified_tip, 'Live tip must have moved past M^2 for this test'

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'), \
             patch('orchestrator.git_ops._run', side_effect=_failing_rebase_run):
            result = await git_ops.advance_main(
                merge_b.merge_commit,
                merge_worktree=merge_b.merge_worktree,
                branch='div-b',
            )

        assert result == 'advanced'

        # A WARNING must record both the verified tip and the live ref tip.
        divergence_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and verified_tip[:8] in r.getMessage()
        ]
        assert divergence_warnings, (
            f'Expected a WARNING recording verified_tip={verified_tip[:8]} '
            f'when live ref ({live_tip[:8]}) diverged from M^2. '
            f'All warnings: {[r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]}'
        )
        # The warning must also mention the live ref tip.
        assert any(live_tip[:8] in r.getMessage() for r in divergence_warnings), (
            f'Divergence WARNING must include live ref tip {live_tip[:8]}. '
            f'Warning messages: {[r.getMessage() for r in divergence_warnings]}'
        )

        await git_ops.cleanup_merge_worktree(merge_b.merge_worktree)

    async def test_no_divergence_warning_when_branch_did_not_move(
        self, git_ops: GitOps, caplog,
    ):
        """When the live branch ref matches M^2 (no post-verify commits),
        no divergence WARNING is emitted.
        """
        merge_b, *_ = await _build_remerge_scenario(
            git_ops, 'nodiv-a', 'nodiv-b',
        )
        assert merge_b.merge_commit is not None and merge_b.merge_worktree is not None
        # Deliberately omit the post-verify commit — live ref == M^2.

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'), \
             patch('orchestrator.git_ops._run', side_effect=_failing_rebase_run):
            result = await git_ops.advance_main(
                merge_b.merge_commit,
                merge_worktree=merge_b.merge_worktree,
                branch='nodiv-b',
            )

        assert result == 'advanced'

        # No divergence WARNING: live ref == M^2, nothing moved.
        divergence_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and 'diverge' in r.getMessage().lower()
        ]
        assert not divergence_warnings, (
            f'No divergence WARNING expected (branch did not move). '
            f'Got: {[r.getMessage() for r in divergence_warnings]}'
        )

        await git_ops.cleanup_merge_worktree(merge_b.merge_worktree)


@pytest.mark.asyncio
class TestHasUncommittedWork:
    async def test_clean_worktree_returns_false(self, git_ops: GitOps):
        wt_info = await git_ops.create_worktree('clean-wt')
        assert not await git_ops.has_uncommitted_work(wt_info.path)

    async def test_untracked_file_returns_true(self, git_ops: GitOps):
        wt_info = await git_ops.create_worktree('untracked-wt')
        (wt_info.path / 'new_file.py').write_text('x = 1\n')
        assert await git_ops.has_uncommitted_work(wt_info.path)

    async def test_modified_tracked_file_returns_true(self, git_ops: GitOps):
        wt_info = await git_ops.create_worktree('modified-wt')
        (wt_info.path / 'README.md').write_text('# Changed\n')
        assert await git_ops.has_uncommitted_work(wt_info.path)

    async def test_file_only_in_task_dir_returns_false(self, git_ops: GitOps):
        wt_info = await git_ops.create_worktree('taskdir-wt')
        task_dir = wt_info.path / '.task'
        task_dir.mkdir(exist_ok=True)
        (task_dir / 'plan.json').write_text('{}')
        assert not await git_ops.has_uncommitted_work(wt_info.path)


@pytest.mark.asyncio
class TestWorkingTreeSync:
    """Tests for the stash/read-tree/pop working-tree protection in advance_main."""

    async def _merge_and_advance(self, git_ops: GitOps, branch: str, filename: str, content: str):
        """Helper: create a file on a branch, merge it, advance main."""
        worktree_info = await git_ops.create_worktree(branch)
        (worktree_info.path / filename).write_text(content)
        await git_ops.commit(worktree_info.path, f'Add {filename}')
        result = await git_ops.merge_to_main(worktree_info.path, branch)
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        advance = await git_ops.advance_main(result.merge_commit)
        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        return advance

    async def test_advance_syncs_working_tree(self, git_ops: GitOps):
        """Merged file appears in the working tree after advance_main."""
        result = await self._merge_and_advance(git_ops, 'sync-basic', 'synced.py', 'synced = True\n')
        assert result == 'advanced'
        assert (git_ops.project_root / 'synced.py').exists()
        assert 'synced = True' in (git_ops.project_root / 'synced.py').read_text()

    async def test_advance_stashes_and_restores_dirty_work(self, git_ops: GitOps):
        """Uncommitted tracked changes survive the merge advance."""
        # Modify a TRACKED file in project_root (uncommitted) — triggers stash/pop
        (git_ops.project_root / 'README.md').write_text('# work in progress\n')

        result = await self._merge_and_advance(git_ops, 'stash-restore', 'merged.py', 'merged = True\n')
        assert result == 'advanced'

        # Merged file should be in working tree
        assert (git_ops.project_root / 'merged.py').exists()
        # User's dirty tracked change should survive (stash/pop restored it)
        assert '# work in progress' in (git_ops.project_root / 'README.md').read_text()

    async def test_wip_overlap_blocks_advance(self, git_ops: GitOps):
        """Dirty file overlapping merge diff returns 'wip_overlap' without moving ref."""
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )

        # Create dirty change to README.md in project_root (don't commit)
        (git_ops.project_root / 'README.md').write_text('# Local WIP edit\n')

        # Merge a conflicting change to README.md
        worktree_info = await git_ops.create_worktree('overlap-readme')
        (worktree_info.path / 'README.md').write_text('# Merged from branch\n')
        await git_ops.commit(worktree_info.path, 'Change README on branch')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'overlap-readme')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'wip_overlap'

        # Main ref should NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

        # Working tree should be unchanged (no stash happened)
        assert (git_ops.project_root / 'README.md').read_text() == '# Local WIP edit\n'

        # Overlap files should be recorded
        assert hasattr(git_ops, '_last_overlap_files')
        assert 'README.md' in git_ops._last_overlap_files

    async def test_wip_overlap_with_staged_file(self, git_ops: GitOps):
        """Staged change overlapping merge diff returns 'wip_overlap'."""
        # Stage a change to README.md
        (git_ops.project_root / 'README.md').write_text('# Local WIP edit\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)

        # Merge a branch that also modifies README.md
        worktree_info = await git_ops.create_worktree('pop-recovery')
        (worktree_info.path / 'new_file.py').write_text('x = 1\n')
        (worktree_info.path / 'README.md').write_text('# Merged from branch\n')
        await git_ops.commit(worktree_info.path, 'Change files on branch')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'pop-recovery')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        # Overlap detected before stash — staged README.md overlaps merge diff
        assert result == 'wip_overlap'

    async def test_pop_conflict_recovery_via_mock(self, git_ops: GitOps):
        """When stash pop fails, advance_main creates recovery branch and returns 'pop_conflict'."""
        # Create a merge commit
        worktree_info = await git_ops.create_worktree('pop-mock')
        (worktree_info.path / 'pop_file.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add pop file')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'pop-mock')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Modify a tracked file (not overlapping merge diff) to trigger stash
        (git_ops.project_root / 'README.md').write_text('# WIP edit\n')

        # Mock _run: stash push succeeds, stash pop fails (simulating conflict)
        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict in README.md')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await git_ops.advance_main(merge_result.merge_commit)

        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'pop_conflict'

        # Recovery branch should be recorded
        assert hasattr(git_ops, '_last_recovery_branch')
        recovery = git_ops._last_recovery_branch
        assert recovery is not None
        assert recovery.startswith('wip/recovery-')

    async def test_consecutive_advance_after_pop_conflict(self, git_ops: GitOps):
        """After pop_conflict recovery, a subsequent advance_main succeeds normally."""
        # Merge first file
        wt1_info = await git_ops.create_worktree('consec-1')
        (wt1_info.path / 'first.py').write_text('first = True\n')
        await git_ops.commit(wt1_info.path, 'Add first')
        merge1 = await git_ops.merge_to_main(wt1_info.path, 'consec-1')
        assert merge1.success
        assert merge1.merge_commit is not None
        assert merge1.merge_worktree is not None

        # Modify a tracked file so stash is triggered, mock stash pop failure
        (git_ops.project_root / 'README.md').write_text('# WIP edit\n')
        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result1 = await git_ops.advance_main(merge1.merge_commit)
        await git_ops.cleanup_merge_worktree(merge1.merge_worktree)
        assert result1 == 'pop_conflict'

        # Working tree should be clean after recovery (read-tree reset)
        _, unstaged, _ = await _run(
            ['git', 'diff', '--name-only'], cwd=git_ops.project_root,
        )
        _, staged, _ = await _run(
            ['git', 'diff', '--name-only', '--cached'], cwd=git_ops.project_root,
        )
        assert not unstaged.strip(), f'Unstaged changes after recovery: {unstaged}'
        assert not staged.strip(), f'Staged changes after recovery: {staged}'

        # Second merge should succeed normally (no stash needed, tree is clean)
        wt2_info = await git_ops.create_worktree('consec-2')
        (wt2_info.path / 'second.py').write_text('second = True\n')
        await git_ops.commit(wt2_info.path, 'Add second')
        merge2 = await git_ops.merge_to_main(wt2_info.path, 'consec-2')
        assert merge2.success
        assert merge2.merge_commit is not None
        assert merge2.merge_worktree is not None

        result2 = await git_ops.advance_main(merge2.merge_commit)
        await git_ops.cleanup_merge_worktree(merge2.merge_worktree)
        assert result2 == 'advanced'

        # Both files should be on main
        _, content, _ = await _run(
            ['git', 'show', 'main:first.py'], cwd=git_ops.project_root,
        )
        assert 'first = True' in content
        _, content2, _ = await _run(
            ['git', 'show', 'main:second.py'], cwd=git_ops.project_root,
        )
        assert 'second = True' in content2

    async def test_wip_overlap_disjoint_files_proceeds(self, git_ops: GitOps):
        """Dirty file NOT overlapping merge diff proceeds normally."""
        # Create dirty file in a different path than what will be merged
        (git_ops.project_root / 'wip_unrelated.py').write_text('wip = True\n')

        # Merge a different file
        worktree_info = await git_ops.create_worktree('disjoint')
        (worktree_info.path / 'merged_file.py').write_text('merged = True\n')
        await git_ops.commit(worktree_info.path, 'Add merged file')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'disjoint')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'advanced'

        # Merged file appears on main
        _, content, _ = await _run(
            ['git', 'show', 'main:merged_file.py'], cwd=git_ops.project_root,
        )
        assert 'merged = True' in content

        # User's dirty file should survive
        assert (git_ops.project_root / 'wip_unrelated.py').exists()
        assert 'wip = True' in (git_ops.project_root / 'wip_unrelated.py').read_text()

    async def test_advance_no_stash_when_clean(self, git_ops: GitOps):
        """Clean working tree: no stash, but read-tree still syncs files."""
        result = await self._merge_and_advance(git_ops, 'clean-sync', 'clean.py', 'clean = True\n')
        assert result == 'advanced'
        assert (git_ops.project_root / 'clean.py').exists()

        # No stash entries should exist
        _, stash_list, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_list.strip() == ''

    async def test_advance_skips_sync_when_not_on_main(self, git_ops: GitOps):
        """When project_root is on another branch, working tree is untouched."""
        # Switch project_root to a different branch
        await _run(['git', 'checkout', '-b', 'other-branch'], cwd=git_ops.project_root)

        # Create a marker file to detect working tree changes
        marker_content = '# Should not change\n'
        (git_ops.project_root / 'README.md').write_text(marker_content)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Mark README'], cwd=git_ops.project_root)

        # Merge a file to main (via worktree from main)
        # Need to create worktree from main for the merge to work
        worktree_info = await git_ops.create_worktree('not-on-main')
        (worktree_info.path / 'should_not_appear.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'not-on-main')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'advanced'

        # File should NOT appear in working tree (we're on other-branch)
        assert not (git_ops.project_root / 'should_not_appear.py').exists()
        # README should be unchanged
        assert (git_ops.project_root / 'README.md').read_text() == marker_content

    async def test_stash_failure_returns_stash_failed(self, git_ops: GitOps):
        """If git stash push fails, advance_main returns 'stash_failed' without moving the ref."""
        # Get current main SHA before the attempt
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )

        # Create a merge commit that could be advanced
        worktree_info = await git_ops.create_worktree('stash-fail')
        (worktree_info.path / 'stash_fail.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'stash-fail')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Make working tree dirty with a TRACKED file (untracked files no longer
        # trigger stash — only tracked modifications do)
        (git_ops.project_root / 'README.md').write_text('# dirty tracked edit\n')

        # Sabotage git stash by making .git/refs/stash unwritable
        # Instead, use a simpler approach: lock the index
        from unittest.mock import patch

        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'push']:
                return (1, '', 'fatal: cannot stash changes')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await git_ops.advance_main(merge_result.merge_commit)

        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'stash_failed'

        # Main ref should NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

    async def test_stash_restored_on_cas_failure(self, git_ops: GitOps):
        """On CAS failure, stash is popped to restore the original working tree."""
        # Create and merge a file
        worktree_info = await git_ops.create_worktree('cas-stash')
        worktree = worktree_info.path
        (worktree / 'cas_file.py').write_text('cas = True\n')
        await git_ops.commit(worktree, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree, 'cas-stash')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Make working tree dirty with a TRACKED file modification
        (git_ops.project_root / 'README.md').write_text('# WIP edit\n')

        # Force CAS failure by passing a wrong expected_main
        result = await git_ops.advance_main(
            merge_result.merge_commit,
            merge_result.merge_worktree,
            branch='cas-stash',
            expected_main='0000000000000000000000000000000000000000',
        )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'cas_failed'

        # Dirty tracked file should be restored (stash popped)
        assert '# WIP edit' in (git_ops.project_root / 'README.md').read_text()

        # No leftover stash entries
        _, stash_list, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_list.strip() == ''

    async def test_sync_path_pop_conflict_still_uses_safe_helper(
        self, git_ops: GitOps,
    ):
        """After refactor, sync-path pop still returns 'pop_conflict' and leaves tree clean.

        This is a regression guard: the result code must remain 'pop_conflict'
        (merge DID advance), and _detect_unmerged_paths must return [] after
        recovery (proving the helper cleaned the tree).
        """
        # Create a merge commit
        wt = await git_ops.create_worktree('sync-safe-helper')
        (wt.path / 'sync_file.py').write_text('x = 1\n')
        await git_ops.commit(wt.path, 'Add sync_file')
        merge_result = await git_ops.merge_to_main(wt.path, 'sync-safe-helper')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Dirty tracked file so stash is created
        (git_ops.project_root / 'README.md').write_text('# WIP sync guard\n')

        # Mock: stash pop returns failure (simulates sync-path pop conflict)
        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict in README.md')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        # Sync path: merge DID advance, WIP conflicted — must still be 'pop_conflict'
        assert result == 'pop_conflict'

        # Tree must be fully clean (recovery helper ran read-tree reset)
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == [], f'Expected clean tree after recovery, got: {unmerged}'

    async def test_cas_failure_pop_conflict_returns_pop_conflict_no_advance(
        self, git_ops: GitOps,
    ):
        """When CAS fails AND stash pop conflicts, advance_main returns 'pop_conflict_no_advance'.

        Main ref must not move, _last_recovery_branch must be set, and no
        unmerged paths may remain in project_root after the call.
        """
        # Create a merge commit (adds cas_pop.py — no overlap with README.md)
        wt = await git_ops.create_worktree('cas-pop-conflict')
        (wt.path / 'cas_pop.py').write_text('x = 1\n')
        await git_ops.commit(wt.path, 'Add cas_pop')
        merge_result = await git_ops.merge_to_main(wt.path, 'cas-pop-conflict')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Dirty tracked file (no overlap with merge diff) so stash is created
        (git_ops.project_root / 'README.md').write_text('# WIP for CAS pop conflict\n')

        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )

        # Mock: stash pop returns failure (simulates pop conflict after CAS failure)
        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict in README.md')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await git_ops.advance_main(
                merge_result.merge_commit,
                merge_result.merge_worktree,
                branch='cas-pop-conflict',
                expected_main='0' * 40,  # force CAS failure
            )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result == 'pop_conflict_no_advance'

        # Recovery branch must have been created and recorded
        assert hasattr(git_ops, '_last_recovery_branch')
        recovery = git_ops._last_recovery_branch
        assert recovery is not None and recovery.startswith('wip/recovery-')

        # Main ref must NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

        # No leftover unmerged paths
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == []

    async def test_advance_main_halts_on_preexisting_unmerged_state(
        self, git_ops: GitOps,
    ):
        """advance_main returns 'unmerged_state' immediately when project_root has UU markers.

        No stash must be created and main ref must not advance.
        """
        # Step 1: prepare a valid merge commit via a clean worktree
        wt = await git_ops.create_worktree('uu-guard-advance')
        (wt.path / 'new_feature.py').write_text('feature = True\n')
        await git_ops.commit(wt.path, 'Add new_feature')
        merge_result = await git_ops.merge_to_main(wt.path, 'uu-guard-advance')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Record state before injecting UU markers
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        _, stash_before, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )

        # Step 2: inject unmerged (stage 1/2/3) entries into the index without
        # doing an actual merge commit or setting MERGE_HEAD.
        await _inject_uu_state(git_ops.project_root, 'uu_conflict_test.py')

        # Verify the UU state is detectable
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert len(unmerged) >= 1, f'Expected unmerged paths after index surgery, got: {unmerged}'

        # Step 3: advance_main must detect UU state and halt without touching main
        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'unmerged_state'

        # Main ref must NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

        # No stash was created during the halted advance attempt
        _, stash_after, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_before.strip() == stash_after.strip()

    async def test_advance_main_halts_on_preexisting_unmerged_state_speculative_shape(
        self, git_ops: GitOps,
    ):
        """advance_main with full speculative-worker call shape returns 'unmerged_state' without stash.

        Uses the full merge_queue.py call shape (branch, expected_main) and dirties
        README.md so the working-tree protection block is armed.  Without the
        unmerged-state guard, advance_main would reach the stash block and return
        'stash_failed' (git stash push fails on a UU index with 'you have unmerged
        paths') -- the guard must fire first so we see 'unmerged_state' instead,
        and no stash entry is ever attempted.

        Mirrors the real caller in merge_queue.py:265-270:
            result = await self._git_ops.advance_main(
                merge_result.merge_commit, merge_wt,
                branch=req.branch, max_attempts=..., expected_main=main_sha,
            )
        """
        # Step 1: prepare a valid merge commit via a clean worktree
        wt = await git_ops.create_worktree('uu-guard-spec')
        (wt.path / 'new_spec_feature.py').write_text('spec_feature = True\n')
        await git_ops.commit(wt.path, 'Add new_spec_feature')
        merge_result = await git_ops.merge_to_main(wt.path, 'uu-guard-spec')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Record state before injecting UU markers and dirtying the tree
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        _, stash_before, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )

        # Step 2: dirty a tracked file so the working-tree protection block
        # would attempt a stash if the unmerged guard were not present.
        # Uses a DIFFERENT path (README.md) from the UU injection below so
        # 'dirty file' and 'unmerged index entry' are independent states.
        (git_ops.project_root / 'README.md').write_text('# WIP speculative guard test\n')

        # Step 3: inject unmerged (stage 1/2/3) entries on 'uu_conflict_spec.py'
        # via index surgery -- no real conflicting merge needed.
        await _inject_uu_state(git_ops.project_root, 'uu_conflict_spec.py', tag=' spec')

        # Precondition: confirm injected UU state is detectable
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert len(unmerged) >= 1, f'Expected unmerged paths after index surgery, got: {unmerged}'

        # Step 4: call advance_main with the full speculative-worker call shape.
        # expected_main is the REAL current main SHA -- without the guard the CAS
        # would succeed and the ref would advance.  A passing test therefore certifies
        # the guard fires BEFORE the entire happy path, not just before CAS.
        #
        # Orthogonal probe: record every _run invocation during advance_main to
        # assert that git stash push was never attempted (decisive narrowing that
        # the unmerged-state guard short-circuited the working-tree protection block).
        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None, **kwargs):
            recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd, **kwargs)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await git_ops.advance_main(
                merge_result.merge_commit,
                merge_result.merge_worktree,
                branch='uu-guard-spec',
                expected_main=main_before.strip(),
            )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result == 'unmerged_state'

        # Decisive narrowing: the guard returned before the working-tree protection
        # block had any chance to invoke git stash push.
        assert not any(c[:3] == ['git', 'stash', 'push'] for c in recorded), (
            f'guard should fire before stash push; recorded stash cmds: '
            f'{[c for c in recorded if c[:2] == ["git", "stash"]]}'
        )

        # Positive-path: confirm the unmerged-state guard was actually entered.
        # _detect_unmerged_paths calls ['git', 'status', '--porcelain']; this
        # command does not appear in the pre-guard path (ls-tree / merge-base),
        # so its presence uniquely certifies that the guard ran rather than
        # short-circuiting for an unrelated reason.
        assert any(c[:2] == ['git', 'status'] and '--porcelain' in c for c in recorded), (
            f'expected _detect_unmerged_paths to invoke git status --porcelain '
            f'(guard path marker); recorded commands: {recorded}'
        )

        # Main ref must NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

        # Corroborating: stash list is unchanged (the guard returned before the stash
        # block, and even if the stash block had been entered it would have failed to
        # create an entry -- the decisive narrowing is the recording-_run probe above).
        _, stash_after, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_before.strip() == stash_after.strip()

    async def test_cas_failure_pop_conflict_does_not_cascade_to_stash_failed(
        self, git_ops: GitOps,
    ):
        """Full cascade regression guard: after pop_conflict_no_advance the tree is clean.

        Simulates the exact cascade the bug report describes:
        1. CAS failure → stash pop conflicts → pop_conflict_no_advance returned.
        2. Second advance_main call (no mocks, no dirty WIP) must NOT return
           'stash_failed' or 'unmerged_state' — it must succeed normally.
        This proves _safe_stash_pop_with_recovery fully cleans the tree.
        """
        # Setup: create a merge commit
        wt = await git_ops.create_worktree('cascade-regr')
        (wt.path / 'cascade.py').write_text('x = 1\n')
        await git_ops.commit(wt.path, 'Add cascade file')
        merge_result = await git_ops.merge_to_main(wt.path, 'cascade-regr')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Dirty a tracked file so advance_main creates a stash
        (git_ops.project_root / 'README.md').write_text('# WIP cascade regression\n')

        # First call: force CAS failure AND mock stash pop to conflict
        original_run = _run

        async def mock_run_conflict(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict in README.md')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run_conflict):
            result1 = await git_ops.advance_main(
                merge_result.merge_commit,
                merge_result.merge_worktree,
                branch='cascade-regr',
                expected_main='0' * 40,  # force CAS failure
            )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result1 == 'pop_conflict_no_advance', f'Expected pop_conflict_no_advance, got {result1}'

        # Tree must be fully clean now (recovery helper ran read-tree reset)
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == [], f'Tree not clean after first advance: {unmerged}'

        # Second call: create a fresh merge commit, no patching, no dirty WIP
        wt2 = await git_ops.create_worktree('cascade-regr-2')
        (wt2.path / 'cascade2.py').write_text('y = 2\n')
        await git_ops.commit(wt2.path, 'Add cascade2')
        merge_result2 = await git_ops.merge_to_main(wt2.path, 'cascade-regr-2')
        assert merge_result2.success

        assert merge_result2.merge_commit is not None
        result2 = await git_ops.advance_main(merge_result2.merge_commit)
        if merge_result2.merge_worktree:
            await git_ops.cleanup_merge_worktree(merge_result2.merge_worktree)

        # Must NOT cascade to stash_failed or unmerged_state
        assert result2 not in ('stash_failed', 'unmerged_state'), (
            f'Cascade failure: second advance returned {result2!r} '
            f'(tree was not cleaned by recovery helper)'
        )
        assert result2 == 'advanced', (
            f'Unexpected result: {result2!r} (fully controlled path: no CAS injection, '
            f'no mocks, no dirty WIP \u2014 cas_failed indicates an environmental regression)'
        )

    async def test_advance_main_mark_before_update_ref(self, git_repo: Path):
        """main_gate_mark_command fires immediately before the update-ref CAS call."""
        mark_cmd = 'echo mark-test'
        mark_ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command=mark_cmd,
            ),
            git_repo,
        )

        # Create a merge commit to advance
        wt = await mark_ops.create_worktree('mark-test-a')
        (wt.path / 'mark_a.py').write_text('a = 1\n')
        await mark_ops.commit(wt.path, 'Add mark_a')
        merge_result = await mark_ops.merge_to_main(wt.path, 'mark-test-a')
        assert merge_result.success
        assert merge_result.merge_commit is not None

        original_run = _run
        recorded: list[tuple[list[str], object]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append((list(cmd), cwd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await mark_ops.advance_main(merge_result.merge_commit)
        if merge_result.merge_worktree:
            await mark_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result == 'advanced'

        # Find the mark call and update-ref call in the recorded sequence
        mark_indices = [
            i for i, (cmd, _) in enumerate(recorded)
            if cmd == ['sh', '-c', mark_cmd]
        ]
        update_ref_indices = [
            i for i, (cmd, _) in enumerate(recorded)
            if cmd[:2] == ['git', 'update-ref'] and 'refs/heads/main' in cmd
        ]
        assert len(mark_indices) >= 1, f'No mark call recorded; calls: {[c for c, _ in recorded]}'
        assert len(update_ref_indices) >= 1, 'No update-ref call recorded'

        # The last mark before update-ref must be immediately adjacent (no other
        # advance-related command between them)
        mark_idx = mark_indices[-1]
        update_ref_idx = update_ref_indices[-1]
        assert mark_idx < update_ref_idx, (
            f'mark (idx={mark_idx}) must precede update-ref (idx={update_ref_idx})'
        )
        # Specifically: mark is at exactly update_ref_idx - 1
        assert mark_idx == update_ref_idx - 1, (
            f'mark must be IMMEDIATELY before update-ref; '
            f'mark_idx={mark_idx}, update_ref_idx={update_ref_idx}, '
            f'intervening: {[c for c, _ in recorded[mark_idx+1:update_ref_idx]]}'
        )
        # Mark runs with cwd=project_root
        assert recorded[mark_idx][1] == mark_ops.project_root

    async def test_advance_main_no_mark_when_unset(self, git_ops: GitOps):
        """With default git_config (main_gate_mark_command=None), no sh -c call is recorded."""
        wt = await git_ops.create_worktree('no-mark-test')
        (wt.path / 'no_mark.py').write_text('x = 0\n')
        await git_ops.commit(wt.path, 'Add no_mark')
        merge_result = await git_ops.merge_to_main(wt.path, 'no-mark-test')
        assert merge_result.success
        assert merge_result.merge_commit is not None

        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await git_ops.advance_main(merge_result.merge_commit)
        if merge_result.merge_worktree:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result == 'advanced'
        # No shell invocation via sh -c
        assert not any(c[:2] == ['sh', '-c'] for c in recorded), (
            f'Unexpected sh -c call with feature off; recorded: {recorded}'
        )

    async def test_advance_main_mark_runs_per_attempt(self, git_repo: Path):
        """mark command fires once per advance_main invocation (re-runs each attempt)."""
        mark_cmd = 'echo mark-per-attempt'
        mark_ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command=mark_cmd,
            ),
            git_repo,
        )

        # First merge + advance
        wt1 = await mark_ops.create_worktree('mark-attempt-1')
        (wt1.path / 'attempt1.py').write_text('a = 1\n')
        await mark_ops.commit(wt1.path, 'Add attempt1')
        merge1 = await mark_ops.merge_to_main(wt1.path, 'mark-attempt-1')
        assert merge1.success and merge1.merge_commit is not None

        original_run = _run
        all_recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None):
            all_recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result1 = await mark_ops.advance_main(merge1.merge_commit)
        if merge1.merge_worktree:
            await mark_ops.cleanup_merge_worktree(merge1.merge_worktree)
        assert result1 == 'advanced'

        # Count marks from first advance
        marks_first = sum(1 for c in all_recorded if c == ['sh', '-c', mark_cmd])

        all_recorded.clear()

        # Second merge + advance
        wt2 = await mark_ops.create_worktree('mark-attempt-2')
        (wt2.path / 'attempt2.py').write_text('b = 2\n')
        await mark_ops.commit(wt2.path, 'Add attempt2')
        merge2 = await mark_ops.merge_to_main(wt2.path, 'mark-attempt-2')
        assert merge2.success and merge2.merge_commit is not None

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result2 = await mark_ops.advance_main(merge2.merge_commit)
        if merge2.merge_worktree:
            await mark_ops.cleanup_merge_worktree(merge2.merge_worktree)
        assert result2 == 'advanced'

        marks_second = sum(1 for c in all_recorded if c == ['sh', '-c', mark_cmd])

        # Each invocation emits exactly one mark call
        assert marks_first == 1, f'Expected 1 mark in first advance, got {marks_first}'
        assert marks_second == 1, f'Expected 1 mark in second advance, got {marks_second}'

    async def test_advance_main_unmarks_on_cas_failure(self, git_repo: Path):
        """main_gate_unmark_command fires after a failed update-ref, clearing the sentinel.

        Setup:
        - GitOps with both mark and unmark commands set.
        - Patch _run: returns (1,'','CAS mismatch') for update-ref, delegates
          everything else to the real _run so merge setup runs normally.
        - After advance_main, assert:
          (1) result == 'cas_failed'
          (2) mark call occurred BEFORE the failed update-ref
          (3) unmark call occurred AFTER the failed update-ref (no lingering mark)
        """
        mark_cmd = 'echo mark-unmark'
        unmark_cmd = 'echo unmark-cleanup'
        mark_ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command=mark_cmd,
                main_gate_unmark_command=unmark_cmd,
            ),
            git_repo,
        )

        # Create a merge commit
        wt = await mark_ops.create_worktree('unmark-cas-fail')
        (wt.path / 'unmark_test.py').write_text('u = 1\n')
        await mark_ops.commit(wt.path, 'Add unmark_test')
        merge_result = await mark_ops.merge_to_main(wt.path, 'unmark-cas-fail')
        assert merge_result.success
        assert merge_result.merge_commit is not None

        original_run = _run
        recorded: list[tuple[list[str], object]] = []

        async def recording_run(cmd, cwd=None):
            # Fail on update-ref to simulate CAS mismatch
            if cmd[:2] == ['git', 'update-ref']:
                recorded.append((list(cmd), cwd))
                return (1, '', 'CAS mismatch: refs/heads/main has been updated')
            recorded.append((list(cmd), cwd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await mark_ops.advance_main(merge_result.merge_commit)
        if merge_result.merge_worktree:
            await mark_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result == 'cas_failed', f'Expected cas_failed, got {result!r}'

        commands = [cmd for cmd, _ in recorded]
        mark_indices = [i for i, c in enumerate(commands) if c == ['sh', '-c', mark_cmd]]
        unmark_indices = [i for i, c in enumerate(commands) if c == ['sh', '-c', unmark_cmd]]
        update_ref_indices = [
            i for i, c in enumerate(commands)
            if c[:2] == ['git', 'update-ref'] and 'refs/heads/main' in c
        ]

        assert len(mark_indices) >= 1, f'No mark call; commands: {commands}'
        assert len(unmark_indices) >= 1, f'No unmark call; commands: {commands}'
        assert len(update_ref_indices) >= 1, f'No update-ref call; commands: {commands}'

        mark_idx = mark_indices[-1]
        unmark_idx = unmark_indices[-1]
        update_ref_idx = update_ref_indices[-1]

        assert mark_idx < update_ref_idx, (
            f'mark (idx={mark_idx}) must precede failed update-ref (idx={update_ref_idx})'
        )
        assert unmark_idx > update_ref_idx, (
            f'unmark (idx={unmark_idx}) must come AFTER failed update-ref (idx={update_ref_idx}); '
            f'commands: {commands}'
        )

    async def test_advance_main_no_unmark_on_success(self, git_repo: Path):
        """unmark command is NOT invoked when update-ref succeeds.

        With both mark and unmark configured, a successful advance must fire
        mark (before update-ref) but must NOT fire unmark — that is failure
        cleanup only.  A regression that accidentally called unmark on success
        would clear a sentinel the hook should have consumed, or re-clear it
        after it was already consumed by the reference-transaction hook.
        """
        mark_cmd = 'echo mark-success-path'
        unmark_cmd = 'echo unmark-should-not-fire'
        mark_ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command=mark_cmd,
                main_gate_unmark_command=unmark_cmd,
            ),
            git_repo,
        )

        wt = await mark_ops.create_worktree('no-unmark-success')
        (wt.path / 'success_test.py').write_text('s = 1\n')
        await mark_ops.commit(wt.path, 'Add success_test')
        merge_result = await mark_ops.merge_to_main(wt.path, 'no-unmark-success')
        assert merge_result.success
        assert merge_result.merge_commit is not None

        original_run = _run
        recorded: list[tuple[list[str], object]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append((list(cmd), cwd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await mark_ops.advance_main(merge_result.merge_commit)
        if merge_result.merge_worktree:
            await mark_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result == 'advanced', f'Expected advanced, got {result!r}'

        commands = [cmd for cmd, _ in recorded]
        # mark must have fired on the success path
        assert any(c == ['sh', '-c', mark_cmd] for c in commands), (
            f'No mark call recorded on success path; commands: {commands}'
        )
        # unmark must NOT have fired — it is failure cleanup only
        assert not any(c == ['sh', '-c', unmark_cmd] for c in commands), (
            f'Unexpected unmark call on success path; commands: {commands}'
        )

    async def test_advance_main_mark_failure_still_advances(self, git_repo: Path):
        """A non-zero mark command is best-effort: advance_main continues to
        update-ref and returns 'advanced' even when the mark exits non-zero.

        Prevents regressions where a logged WARNING becomes an early abort —
        the exact failure mode this task was written to prevent (bricking the
        merge queue because the sentinel write failed).
        """
        mark_ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command='exit 1',  # always fails non-zero
            ),
            git_repo,
        )

        wt = await mark_ops.create_worktree('mark-fail-advance')
        (wt.path / 'mark_fail.py').write_text('f = 1\n')
        await mark_ops.commit(wt.path, 'Add mark_fail')
        merge_result = await mark_ops.merge_to_main(wt.path, 'mark-fail-advance')
        assert merge_result.success
        assert merge_result.merge_commit is not None

        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await mark_ops.advance_main(merge_result.merge_commit)
        if merge_result.merge_worktree:
            await mark_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        # A failed mark must not abort the advance
        assert result == 'advanced', (
            f'Mark failure must not abort advance_main; got {result!r}'
        )
        # update-ref was still called despite the failed mark
        assert any(
            c[:2] == ['git', 'update-ref'] and 'refs/heads/main' in c
            for c in recorded
        ), f'update-ref not called after failed mark; commands: {recorded}'


@pytest.mark.asyncio
class TestUnmergedDetection:
    """Tests for the _detect_unmerged_paths helper."""

    async def test_detect_unmerged_paths_empty_on_clean_tree(self, git_ops: GitOps):
        """On a freshly-initialized repo with no conflicts, helper returns []."""
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == []

    async def test_detect_unmerged_paths_returns_uu_files(self, git_ops: GitOps):
        """After a conflicting merge, helper returns paths containing the conflicted file."""
        # Create a divergent branch with a conflicting change to README.md
        await _run(
            ['git', 'checkout', '-b', 'conflict-b'],
            cwd=git_ops.project_root,
        )
        (git_ops.project_root / 'README.md').write_text('# From B\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'change README on B'],
            cwd=git_ops.project_root,
        )

        # Go back to main and make a divergent change
        await _run(['git', 'checkout', 'main'], cwd=git_ops.project_root)
        (git_ops.project_root / 'README.md').write_text('# From Main\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'change README on main'],
            cwd=git_ops.project_root,
        )

        # Trigger a conflicting merge — leaves UU markers in index/worktree
        rc, _, _ = await _run(
            ['git', 'merge', 'conflict-b'],
            cwd=git_ops.project_root,
        )
        assert rc != 0  # Must have conflicted

        # Now test the helper
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert 'README.md' in unmerged
        assert len(unmerged) >= 1

    async def test_inject_uu_state_helper_creates_unmerged_entries(
        self, git_ops: GitOps,
    ):
        """_inject_uu_state creates detectable UU index entries for the given path."""
        await _inject_uu_state(git_ops.project_root, 'helper_probe.py')
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert 'helper_probe.py' in unmerged, (
            f'Expected helper_probe.py in unmerged paths, got: {unmerged}'
        )

    async def test_inject_uu_state_raises_on_non_git_cwd(
        self, tmp_path: Path,
    ):
        """_inject_uu_state raises CalledProcessError when cwd is not a git repo.

        git hash-object exits with rc != 0 in a non-git directory; without
        check=True the helper silently builds an invalid payload.  With
        check=True it raises immediately, turning silent corruption into an
        actionable CalledProcessError that includes stderr.
        """
        with pytest.raises(subprocess.CalledProcessError):
            await _inject_uu_state(tmp_path, 'foo.py')


@pytest.mark.asyncio
class TestSafeStashPopWithRecovery:
    """Tests for the _safe_stash_pop_with_recovery helper."""

    async def test_safe_stash_pop_success_returns_ok(self, git_ops: GitOps):
        """_safe_stash_pop_with_recovery returns (True, None) on a clean pop.

        Dirty file content is restored, stash list is empty, no recovery
        branch is created.
        """
        # Stash a dirty tracked file
        (git_ops.project_root / 'README.md').write_text('# WIP content\n')
        await _run(
            ['git', 'stash', 'push', '-m', 'test stash'], cwd=git_ops.project_root,
        )

        # Verify stash was created
        _, stash_list, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_list.strip(), 'Stash should have an entry before pop'

        # Call the helper — no conflict exists, should succeed
        ok, recovery = await git_ops._safe_stash_pop_with_recovery('label-1')

        assert ok is True
        assert recovery is None

        # Dirty file content must be restored
        assert '# WIP content' in (git_ops.project_root / 'README.md').read_text()

        # No recovery branch was created
        _, branches, _ = await _run(['git', 'branch'], cwd=git_ops.project_root)
        assert 'wip/recovery' not in branches

        # Stash list is now empty
        _, stash_after, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_after.strip() == ''

    async def test_safe_stash_pop_conflict_creates_recovery_branch(
        self, git_ops: GitOps,
    ):
        """_safe_stash_pop_with_recovery returns (False, branch) and cleans up on conflict.

        The recovery branch points at the original stash tree, stash list is
        empty, and project_root has no leftover unmerged paths afterward.
        """
        # Write WIP content to README.md and stash it
        (git_ops.project_root / 'README.md').write_text('# WIP content for conflict\n')
        await _run(
            ['git', 'stash', 'push', '-m', 'wip-for-conflict'],
            cwd=git_ops.project_root,
        )

        # Capture stash tree before pop attempt (to verify recovery branch later)
        _, stash_tree, _ = await _run(
            ['git', 'rev-parse', 'stash@{0}^{tree}'], cwd=git_ops.project_root,
        )

        # Commit a DIFFERENT version of README.md on main so stash pop will conflict.
        # Three-way merge scenario:
        #   base (stash parent) : '# Test\n'
        #   ours (HEAD)         : '# Main version…\n'
        #   theirs (stash)      : '# WIP content for conflict\n'
        (git_ops.project_root / 'README.md').write_text('# Main version — conflicts with WIP\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Commit conflicting README'],
            cwd=git_ops.project_root,
        )

        # Call the helper — git stash pop will conflict (real git conflict)
        ok, recovery = await git_ops._safe_stash_pop_with_recovery('label-2')

        assert ok is False
        assert recovery is not None
        assert recovery.startswith('wip/recovery-label-2-'), (
            f'Recovery branch name should start with wip/recovery-label-2-, got {recovery!r}'
        )

        # Recovery branch tree must match the original stash tree
        _, branch_tree, _ = await _run(
            ['git', 'rev-parse', f'{recovery}^{{tree}}'], cwd=git_ops.project_root,
        )
        assert stash_tree.strip() == branch_tree.strip(), (
            'Recovery branch tree must equal original stash tree'
        )

        # Stash list must be empty (stash was dropped after branch creation)
        _, stash_after, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_after.strip() == ''

        # No unmerged paths remain in project_root
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == [], f'Expected no unmerged paths after recovery, got: {unmerged}'


@pytest.mark.asyncio
class TestScrubTaskDirFromTree:
    async def test_scrub_returns_clean_when_no_contamination(
        self, git_ops: GitOps, caplog,
    ):
        """scrub_task_dir_from_tree returns CLEAN when ls-tree shows no tracked .task/ files."""
        # Create a real worktree for a realistic working directory (no mock yet)
        worktree_info = await git_ops.create_worktree('scrub-clean')

        # Create a .task/ directory with sentinel content on disk — canary for rmtree
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('canary-no-rmtree\n')

        async def mock_run(cmd, cwd=None):
            # ls-tree returns empty stdout — no tracked .task/ files
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '', '')
            # Strict — no other git commands should be reached on the CLEAN path
            pytest.fail(f'unexpected _run call on CLEAN path: {cmd}')

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=mock_run),
        ):
            result = await scrub_task_dir_from_tree(worktree_info.path, 'test-clean')

        # Return value must be CLEAN — no tracked .task/ files in tree
        assert result.outcome == ScrubOutcome.CLEAN, (
            f'Expected ScrubOutcome.CLEAN when ls-tree is empty, got {result!r}'
        )

        # Filesystem .task/ must still exist — rmtree must NOT have run
        assert sentinel.exists(), (
            'sentinel.txt was deleted — rmtree must not run on CLEAN path'
        )

        # No WARNING or ERROR should have been logged on the CLEAN path
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, (
            f'Unexpected log entries on CLEAN path: {[r.getMessage() for r in warnings]}'
        )

    async def test_scrub_returns_failed_when_git_rm_fails(
        self, git_ops: GitOps, caplog,
    ):
        """scrub_task_dir_from_tree returns FAILED and skips rmtree/commit when git rm fails."""
        # Create a real worktree for a realistic working directory (no mock yet)
        worktree_info = await git_ops.create_worktree('scrub-rm-fail')

        # Create a .task/ directory with sentinel content on disk
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('keep-me\n')

        async def mock_run(cmd, cwd=None):
            # (a) Fake .task/ contamination detected via ls-tree
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            # (b) Fail git rm --cached to simulate index corruption / permission error
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (1, '', 'fatal: simulated git rm failure')
            # Strict — no other git commands should be reached on the failure path
            pytest.fail(f'unexpected _run call on git-rm failure path: {cmd}')

        with (
            caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=mock_run),
        ):
            result = await scrub_task_dir_from_tree(worktree_info.path, 'test-rm-fail')

        # Return value must be FAILED — git rm failed, scrub did not complete
        assert result.outcome == ScrubOutcome.FAILED, (
            f'Expected outcome=FAILED on git rm failure, got {result!r}'
        )

        # An ERROR must have been logged containing the context label and the stderr
        error_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            'test-rm-fail' in m and 'simulated git rm failure' in m
            for m in error_msgs
        ), f'Expected ERROR log with context and stderr, got: {error_msgs}'

        # Filesystem .task/ must still exist — rmtree must have been skipped
        assert sentinel.exists(), (
            'sentinel.txt was deleted — rmtree must be skipped on git rm failure'
        )

    async def test_scrub_returns_scrubbed_on_happy_path(
        self, git_ops: GitOps, caplog,
    ):
        """scrub_task_dir_from_tree returns SCRUBBED, runs rmtree, and commits on success."""
        worktree_info = await git_ops.create_worktree('scrub-happy')

        # Sentinel inside .task/ — must be removed by rmtree after a successful scrub
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('remove-me\n')

        commit_calls: list = []

        async def mock_run(cmd, cwd=None):
            # (a) Fake tracked .task/ file via ls-tree
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            # (b) git rm --cached succeeds
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (0, '', '')
            # (c) git commit (amend) succeeds — record and ack
            if len(cmd) >= 2 and cmd[1] == 'commit':
                commit_calls.append(list(cmd))
                return (0, '', '')
            # Strict — any other command is unexpected on the success path
            pytest.fail(f'unexpected _run call on scrub happy path: {cmd}')

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=mock_run),
        ):
            result = await scrub_task_dir_from_tree(worktree_info.path, 'test-happy')

        # Return value must be SCRUBBED
        assert result.outcome == ScrubOutcome.SCRUBBED, (
            f'Expected outcome=SCRUBBED on success, got {result!r}'
        )

        # No ERROR should have been logged
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, f'Unexpected ERROR log entries: {[r.getMessage() for r in errors]}'

        # Filesystem .task/ must have been cleaned up (rmtree ran)
        assert not sentinel.exists(), (
            'sentinel.txt still exists — rmtree must run on a successful scrub'
        )

        # git commit must have been called exactly once
        assert len(commit_calls) == 1, (
            f'Expected exactly one commit call, got: {commit_calls}'
        )

    async def test_scrub_returns_failed_when_git_commit_fails(
        self, git_ops: GitOps, caplog,
    ):
        """scrub_task_dir_from_tree returns FAILED and logs error when commit fails post-rm."""
        worktree_info = await git_ops.create_worktree('scrub-commit-fail')

        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('already-removed-by-rmtree\n')

        async def mock_run(cmd, cwd=None):
            # (a) Fake tracked .task/ file via ls-tree
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            # (b) git rm --cached succeeds (contamination removed from index)
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (0, '', '')
            # (c) git commit fails (e.g. locked index, hook failure)
            if len(cmd) >= 2 and cmd[1] == 'commit':
                return (1, '', 'fatal: simulated commit failure')
            # Strict — unexpected command
            pytest.fail(f'unexpected _run call on commit-failure path: {cmd}')

        with (
            caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=mock_run),
        ):
            result = await scrub_task_dir_from_tree(worktree_info.path, 'test-commit-fail')

        # Return value must be FAILED — commit did not succeed
        assert result.outcome == ScrubOutcome.FAILED, (
            f'Expected outcome=FAILED on commit failure, got {result!r}'
        )

        # An ERROR must have been logged with context and the commit stderr
        error_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            'test-commit-fail' in m and 'simulated commit failure' in m
            for m in error_msgs
        ), f'Expected ERROR log with context and stderr, got: {error_msgs}'

        # Filesystem .task/ must be GONE — rmtree runs before the commit step
        assert not sentinel.exists(), (
            'sentinel.txt still exists — rmtree must run before git commit attempt'
        )

    async def test_scrub_failed_result_carries_error(
        self, tmp_path: Path,
    ):
        """When git rm fails, the returned ScrubResult must carry the git stderr.

        After the ScrubResult → dataclass conversion, the failure path sets
        outcome=ScrubOutcome.FAILED and error=<stderr>.strip().  This test drives
        that conversion by asserting .outcome and .error on the return value.

        Uses tmp_path directly (no real worktree) since _run is fully mocked.
        """
        async def mock_run(cmd, cwd=None):
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (1, '', 'fatal: pathspec error from git rm')
            pytest.fail(f'unexpected _run call: {cmd}')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await scrub_task_dir_from_tree(tmp_path, 'carries-err')

        assert result.outcome == ScrubOutcome.FAILED, (
            f'Expected outcome=FAILED on git rm failure, got {result!r}'
        )
        assert result.error is not None, 'Expected error to be set on git rm failure'
        assert 'pathspec' in result.error, (
            f'Expected git rm stderr in .error, got: {result.error!r}'
        )

    async def test_scrub_failed_whitespace_stderr_collapses_to_none(
        self, tmp_path: Path,
    ):
        """Whitespace-only git rm stderr must collapse to error=None.

        The production code uses ``err.strip() or None`` (git_ops.py:126) so that
        whitespace-only stderr (e.g. '   \\n') is normalised to None rather than
        stored as a meaningless whitespace string.  This companion test to
        test_scrub_failed_result_carries_error drives that normalisation branch.

        Uses tmp_path directly (no real worktree) since _run is fully mocked.
        """
        async def mock_run(cmd, cwd=None):
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (1, '', '   \n')
            pytest.fail(f'unexpected _run call: {cmd}')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await scrub_task_dir_from_tree(tmp_path, 'whitespace-err')

        assert result.outcome == ScrubOutcome.FAILED, (
            f'Expected outcome=FAILED on git rm failure, got {result!r}'
        )
        assert result.error is None, (
            f'Expected error=None for whitespace-only stderr, got: {result.error!r}'
        )

    async def test_scrub_scrubbed_result_has_no_error(
        self, tmp_path: Path,
    ):
        """When scrub succeeds, ScrubResult must have outcome=SCRUBBED and error=None.

        Uses tmp_path directly (no real worktree) since _run is fully mocked.
        """
        async def mock_run(cmd, cwd=None):
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (0, '', '')
            if len(cmd) >= 2 and cmd[1] == 'commit':
                return (0, '', '')
            pytest.fail(f'unexpected _run call: {cmd}')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await scrub_task_dir_from_tree(tmp_path, 'no-err-ok')

        assert result.outcome == ScrubOutcome.SCRUBBED, (
            f'Expected outcome=SCRUBBED on success, got {result!r}'
        )
        assert result.error is None, f'Expected error=None on success, got {result.error!r}'

    async def test_scrub_clean_result_has_no_error(
        self, tmp_path: Path,
    ):
        """When no .task/ files are present, ScrubResult must have outcome=CLEAN and error=None.

        Uses tmp_path directly (no real worktree) since _run is fully mocked.
        """
        async def mock_run(cmd, cwd=None):
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '', '')  # empty — no .task/ tracked
            pytest.fail(f'unexpected _run call on clean path: {cmd}')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await scrub_task_dir_from_tree(tmp_path, 'clean-no-err')

        assert result.outcome == ScrubOutcome.CLEAN, (
            f'Expected outcome=CLEAN on empty tree, got {result!r}'
        )
        assert result.error is None, f'Expected error=None on clean, got {result.error!r}'

    async def test_scrub_amend_false_creates_new_commit(
        self, git_ops: GitOps,
    ):
        """scrub_task_dir_from_tree(amend=False) extends the commit chain.

        The amend=False path (used by create_worktree, line 342 of git_ops.py)
        must create a NEW child commit rather than rewriting the existing one.
        This integration test verifies against a real git repository:
        (a) outcome == SCRUBBED and error is None,
        (b) HEAD moved to a new SHA after the scrub,
        (c) the old HEAD is the first parent of the new HEAD (new commit, not amend),
        (d) .task/ is absent from the new HEAD commit tree,
        (e) the new commit message contains 'chore: remove .task/ contamination'.

        Uses git_ops fixture for a real git repo — no mocks.
        """
        # Create a worktree on a fresh branch with a regular commit.
        worktree_info = await git_ops.create_worktree('amend-false-branch')
        (worktree_info.path / 'work.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add work file')

        # Inject .task/ contamination, bypassing the .task/.gitignore defence.
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"contamination": true}\n')
        rc, _, _ = await _run(['git', 'add', '-f', '.task/plan.json'], cwd=worktree_info.path)
        assert rc == 0, 'setup: git add -f .task/plan.json failed'
        rc, _, _ = await _run(
            ['git', 'commit', '-m', 'Simulated .task/ contamination'],
            cwd=worktree_info.path,
        )
        assert rc == 0, 'setup: git commit of contamination failed'

        # Record HEAD before scrub.
        _, old_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree_info.path)
        old_head = old_head.strip()
        assert len(old_head) == 40, f'Pre-condition: expected 40-char SHA, got {old_head!r}'

        # Verify contamination is present before scrub.
        _, ls_before, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', '.task/'],
            cwd=worktree_info.path,
        )
        assert '.task/plan.json' in ls_before, (
            f'Pre-condition: expected .task/plan.json in tree, got: {ls_before!r}'
        )

        # Call scrub with amend=False — must create a new child commit.
        result = await scrub_task_dir_from_tree(
            worktree_info.path, 'amend-false-test', amend=False,
        )

        # (a) Outcome must be SCRUBBED with no error.
        assert result.outcome == ScrubOutcome.SCRUBBED, (
            f'Expected outcome=SCRUBBED, got {result!r}'
        )
        assert result.error is None, (
            f'Expected error=None on successful scrub, got: {result.error!r}'
        )

        # (b) HEAD must have moved to a new SHA.
        _, new_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree_info.path)
        new_head = new_head.strip()
        assert new_head != old_head, (
            f'Expected HEAD to move after amend=False scrub, but still {old_head!r}'
        )

        # (c) Old HEAD must be the first parent of new HEAD — proves a new child
        #     commit was created rather than an amendment of the contamination commit.
        _, parent, _ = await _run(
            ['git', 'rev-parse', 'HEAD^'],
            cwd=worktree_info.path,
        )
        assert parent.strip() == old_head, (
            f'Expected old HEAD ({old_head}) to be parent of new HEAD, '
            f'but HEAD^ is {parent.strip()!r}'
        )

        # (d) .task/ must be absent from the new HEAD commit tree.
        _, task_in_tree, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', '.task/'],
            cwd=worktree_info.path,
        )
        assert not task_in_tree.strip(), (
            f'.task/ must be absent from new commit tree, but found: {task_in_tree!r}'
        )

        # (e) New commit message must contain the expected scrub marker text.
        _, commit_msg, _ = await _run(
            ['git', 'log', '-1', '--format=%B'],
            cwd=worktree_info.path,
        )
        assert 'chore: remove .task/ contamination' in commit_msg, (
            f'Expected commit message to contain scrub marker, got: {commit_msg!r}'
        )


@pytest.mark.asyncio
class TestMergeToMainScrubFailure:
    """Tests for merge_to_main returning success=False when scrub fails."""

    async def test_merge_to_main_fails_when_scrub_fails(
        self, git_ops: GitOps,
    ):
        """merge_to_main must return MergeResult(success=False) when scrub fails.

        When scrub_task_dir_from_tree returns ScrubResult.FAILED, merge_to_main
        should fail fast: clean up the merge worktree and return
        MergeResult(success=False, conflicts=False, ...) rather than returning
        MergeResult(success=True) with a contaminated commit.
        """
        # Set up a feature branch with a committed file.
        worktree_info = await git_ops.create_worktree('scrub-fail-branch')
        (worktree_info.path / 'scrub_test.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add scrub test file')

        # Patch _scrub_task_dir_from_tree to return FAILED, simulating a scrub
        # failure after the merge commit has been created.
        async def fake_scrub(*args, **kwargs):
            return ScrubResult(outcome=ScrubOutcome.FAILED)

        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            new=fake_scrub,
        ):
            result = await git_ops.merge_to_main(worktree_info.path, 'scrub-fail-branch')

        # (1) success must be False — scrub failure is a hard stop
        assert result.success is False, (
            f'Expected success=False on scrub failure, got success={result.success!r}'
        )

        # (2) conflicts must be False — this is NOT a merge conflict
        assert result.conflicts is False, (
            f'Expected conflicts=False on scrub failure, got conflicts={result.conflicts!r}'
        )

        # (3) details must mention 'scrub' and the branch name
        assert 'scrub' in result.details.lower(), (
            f'Expected "scrub" in details, got: {result.details!r}'
        )
        assert 'task/scrub-fail-branch' in result.details, (
            f'Expected full prefixed branch name in details, got: {result.details!r}'
        )

        # (4) pre_merge_sha must be a valid 40-char SHA
        assert result.pre_merge_sha is not None, 'Expected pre_merge_sha to be set'
        assert len(result.pre_merge_sha.strip()) == 40, (
            f'Expected 40-char SHA, got: {result.pre_merge_sha!r}'
        )

        # (5) merge_commit must be None — no committed merge SHA on failure
        assert result.merge_commit is None, (
            f'Expected merge_commit=None on scrub failure, got: {result.merge_commit!r}'
        )

        # (6) merge_worktree must be None — mirrors the non-conflict failure path
        assert result.merge_worktree is None, (
            f'Expected merge_worktree=None on scrub failure, got: {result.merge_worktree!r}'
        )

        # (7) No _merge-* worktrees should remain registered.
        _, worktree_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        leak_lines = [
            line for line in worktree_list.splitlines()
            if '_merge-' in line
        ]
        assert not leak_lines, (
            f'Leaked merge worktrees still registered: {leak_lines}'
        )

        # (8) No _merge-* directories should remain on disk.
        worktree_base = git_ops.worktree_base
        if worktree_base.exists():
            leak_dirs = list(worktree_base.glob('_merge-*'))
            assert not leak_dirs, (
                f'Leaked merge worktree directories on disk: {leak_dirs}'
            )

    async def test_merge_to_main_succeeds_when_scrub_cleans_task_dir(
        self, git_ops: GitOps,
    ):
        """merge_to_main must return success=True when scrub_task_dir_from_tree
        returns ScrubResult.SCRUBBED (i.e. .task/ was found and removed cleanly).

        Guards against regressions where a future change accidentally treats
        SCRUBBED the same as FAILED.
        """
        worktree_info = await git_ops.create_worktree('scrub-ok-branch')
        (worktree_info.path / 'scrub_ok.py').write_text('y = 2\n')
        await git_ops.commit(worktree_info.path, 'Add scrub-ok file')

        async def fake_scrub_ok(*args, **kwargs):
            return ScrubResult(outcome=ScrubOutcome.SCRUBBED)

        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            new=fake_scrub_ok,
        ):
            result = await git_ops.merge_to_main(worktree_info.path, 'scrub-ok-branch')

        # SCRUBBED must not trigger the failure path — merge should succeed.
        assert result.success is True, (
            f'Expected success=True when scrub returns SCRUBBED, got {result.success!r}'
        )
        assert result.merge_commit is not None, (
            'Expected a valid merge_commit SHA when scrub returns SCRUBBED'
        )
        assert len(result.merge_commit.strip()) == 40, (
            f'Expected 40-char merge_commit SHA, got: {result.merge_commit!r}'
        )
        assert result.conflicts is False, (
            f'Expected conflicts=False on SCRUBBED result, got {result.conflicts!r}'
        )

        # Clean up the merge worktree to avoid polluting other tests.
        if result.merge_worktree is not None:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_merge_to_main_scrubs_real_task_dir(
        self, git_ops: GitOps,
    ):
        """merge_to_main strips .task/ from the merge commit via the real scrub.

        Unlike test_merge_to_main_succeeds_when_scrub_cleans_task_dir, this test
        uses NO mock — it commits a real .task/plan.json file on the feature branch
        and verifies that merge_to_main produces a clean merge commit with no .task/
        entries in the tree.  This exercises the real scrub_task_dir_from_tree with
        amend=True on an actual contaminated merge commit.
        """
        # Create a worktree and commit a regular file so the branch has content.
        worktree_info = await git_ops.create_worktree('scrub-real-branch')
        (worktree_info.path / 'feature.py').write_text('def feature(): pass\n')
        await git_ops.commit(worktree_info.path, 'Add feature file')

        # Inject .task/ contamination directly via git commands, bypassing the
        # safety guards in git_ops.commit (which would normally unstage .task/).
        # Use -f to force-add past the .task/.gitignore ('*') that create_worktree
        # places there as a defence-in-depth measure.
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"contamination": true}\n')
        await _run(['git', 'add', '-f', '.task/plan.json'], cwd=worktree_info.path)
        await _run(
            ['git', 'commit', '-m', 'Simulated .task/ contamination'],
            cwd=worktree_info.path,
        )

        # Verify contamination is present on the branch before merge.
        _, ls_before, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', '.task/'],
            cwd=worktree_info.path,
        )
        assert '.task/plan.json' in ls_before, (
            f'Pre-condition: expected .task/plan.json on branch, got: {ls_before!r}'
        )

        # Call merge_to_main with NO mock — uses real scrub_task_dir_from_tree.
        result = await git_ops.merge_to_main(worktree_info.path, 'scrub-real-branch')

        try:
            # (a) Merge must succeed.
            assert result.success is True, (
                f'Expected success=True when real scrub cleans .task/, got {result.success!r}'
            )

            # (b) A merge commit must have been created.
            assert result.merge_commit is not None, (
                'Expected a valid merge_commit SHA when scrub succeeds'
            )

            # (c) Verify .task/ is absent from the merge commit tree.
            _, task_in_tree, _ = await _run(
                ['git', 'ls-tree', '-r', '--name-only', result.merge_commit.strip(), '--', '.task/'],
                cwd=git_ops.project_root,
            )
            assert not task_in_tree.strip(), (
                f'.task/ must be absent from merge commit tree, but found: {task_in_tree!r}'
            )
        finally:
            # Ensure merge worktree is cleaned up even when assertions fail.
            if result.merge_worktree is not None:
                await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_merge_to_main_scrub_failure_details_include_root_cause(
        self, git_ops: GitOps,
    ):
        """merge_to_main must surface ScrubResult.error in MergeResult.details.

        When scrub_task_dir_from_tree returns a ScrubResult with error set,
        the failure reason (raw git stderr) must appear in MergeResult.details
        so MergeQueue propagates it to MergeOutcome.reason without log scraping.

        This test is the failing test for step-5.  It will pass once step-6
        wires scrub_result.error into the details f-string.
        """
        worktree_info = await git_ops.create_worktree('scrub-root-cause-branch')
        (worktree_info.path / 'rc_test.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add rc_test file')

        root_cause = 'fatal: cannot amend merge commit'

        async def fake_scrub_with_error(*args, **kwargs):
            return ScrubResult(outcome=ScrubOutcome.FAILED, error=root_cause)

        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            new=fake_scrub_with_error,
        ):
            result = await git_ops.merge_to_main(
                worktree_info.path, 'scrub-root-cause-branch',
            )

        assert result.success is False, (
            f'Expected success=False on scrub failure, got {result.success!r}'
        )
        assert 'cannot amend merge commit' in result.details, (
            f'Expected git stderr in details for operator visibility, got: {result.details!r}'
        )

    async def test_merge_to_main_scrub_failure_no_stderr_fallback(
        self, git_ops: GitOps,
    ):
        """merge_to_main must NOT include '(no stderr)' when scrub error is None.

        When scrub_task_dir_from_tree returns FAILED with error=None (e.g. stderr
        was empty/whitespace), MergeResult.details must not contain the old
        '(no stderr)' fallback string. The standardised format_error() helper
        returns '' in that case, so nothing extra is appended.
        """
        worktree_info = await git_ops.create_worktree('scrub-no-stderr-branch')
        (worktree_info.path / 'ns_test.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add ns_test file')

        async def fake_scrub_no_error(*args, **kwargs):
            return ScrubResult(outcome=ScrubOutcome.FAILED)  # error=None

        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            new=fake_scrub_no_error,
        ):
            result = await git_ops.merge_to_main(
                worktree_info.path, 'scrub-no-stderr-branch',
            )

        assert result.success is False, (
            f'Expected success=False on scrub failure, got {result.success!r}'
        )
        assert '(no stderr)' not in result.details, (
            f'Expected no "(no stderr)" fallback in details, got: {result.details!r}'
        )


class TestScrubResultInvariant:
    """Unit tests for ScrubResult.__post_init__ guard.

    The invariant: error may only be non-None when outcome is ScrubOutcome.FAILED.
    All other (outcome, error) combinations are semantically invalid and should
    raise ValueError at construction time.
    """

    def test_clean_with_error_raises(self):
        """ScrubResult(CLEAN, error=...) must raise ValueError."""
        with pytest.raises(ValueError):
            ScrubResult(outcome=ScrubOutcome.CLEAN, error='some error')

    def test_scrubbed_with_error_raises(self):
        """ScrubResult(SCRUBBED, error=...) must raise ValueError."""
        with pytest.raises(ValueError):
            ScrubResult(outcome=ScrubOutcome.SCRUBBED, error='some error')

    def test_failed_with_error_succeeds(self):
        """ScrubResult(FAILED, error=...) is valid and must not raise."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED, error='fatal: git error')
        assert result.outcome == ScrubOutcome.FAILED
        assert result.error == 'fatal: git error'

    def test_failed_without_error_succeeds(self):
        """ScrubResult(FAILED) with error=None is valid (no error captured)."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED)
        assert result.outcome == ScrubOutcome.FAILED
        assert result.error is None

    def test_clean_without_error_succeeds(self):
        """ScrubResult(CLEAN) with error=None is valid."""
        result = ScrubResult(outcome=ScrubOutcome.CLEAN)
        assert result.outcome == ScrubOutcome.CLEAN
        assert result.error is None

    def test_scrubbed_without_error_succeeds(self):
        """ScrubResult(SCRUBBED) with error=None is valid."""
        result = ScrubResult(outcome=ScrubOutcome.SCRUBBED)
        assert result.outcome == ScrubOutcome.SCRUBBED
        assert result.error is None


class TestScrubResultFormatError:
    """Unit tests for ScrubResult.format_error() helper method.

    format_error(prefix='') returns prefix+error when error is set,
    or empty string when error is None.
    """

    def test_failed_with_error_default_prefix(self):
        """FAILED with error and no prefix returns the bare error string."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED, error='fatal: git rm failed')
        assert result.format_error() == 'fatal: git rm failed', (
            f'Expected bare error string, got {result.format_error()!r}'
        )

    def test_failed_with_error_custom_prefix(self):
        """FAILED with error and custom prefix returns prefix+error."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED, error='fatal: git rm failed')
        assert result.format_error(prefix=' Error: ') == ' Error: fatal: git rm failed', (
            f'Expected prefixed error, got {result.format_error(prefix=" Error: ")!r}'
        )

    def test_failed_with_no_error_returns_empty(self):
        """FAILED with error=None returns empty string regardless of prefix."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED)
        assert result.format_error() == '', (
            f'Expected empty string when error is None, got {result.format_error()!r}'
        )
        assert result.format_error(prefix=' Error: ') == '', (
            'Expected empty string even with prefix when error is None'
        )

    def test_clean_with_no_error_returns_empty(self):
        """CLEAN with error=None returns empty string."""
        result = ScrubResult(outcome=ScrubOutcome.CLEAN)
        assert result.format_error() == '', (
            f'Expected empty string for CLEAN outcome, got {result.format_error()!r}'
        )

    def test_scrubbed_with_no_error_returns_empty(self):
        """SCRUBBED with error=None returns empty string."""
        result = ScrubResult(outcome=ScrubOutcome.SCRUBBED)
        assert result.format_error() == '', (
            f'Expected empty string for SCRUBBED outcome, got {result.format_error()!r}'
        )

    def test_failed_with_empty_string_error_raises_value_error(self):
        """FAILED with error='' is rejected at construction time.

        All production call-sites normalise empty/whitespace stderr to None via
        ``err.strip() or None`` before constructing ScrubResult.  Permitting an
        empty-string error would create an ambiguous state (``error is not None``
        but ``not error``).  The __post_init__ guard makes the invariant explicit:
        ``error`` is either None or a non-empty, non-whitespace-only string.
        """
        with pytest.raises(ValueError, match='empty or whitespace-only'):
            ScrubResult(outcome=ScrubOutcome.FAILED, error='')

    def test_failed_with_whitespace_only_error_raises_value_error(self):
        """FAILED with error='   ' (whitespace only) is also rejected."""
        with pytest.raises(ValueError, match='empty or whitespace-only'):
            ScrubResult(outcome=ScrubOutcome.FAILED, error='   ')


@pytest.mark.asyncio
class TestPushMain:
    """Best-effort push of local main to <remote>/<main_branch>.

    Lives next to advance_main: each successful CAS advance is mirrored to
    origin so an external clone (humans, CI, mirrors) sees the same history
    the merge worker just produced.
    """

    async def test_push_main_pushes_local_advance(
        self, git_repo_with_remote: tuple[Path, Path],
    ):
        """Happy path: a commit added locally lands on the bare origin."""
        origin, local = git_repo_with_remote
        git_ops = GitOps(GitConfig(push_after_advance=True), local)

        (local / 'local.txt').write_text('local\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'local commit'], cwd=local)
        _, local_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=local)

        result = await git_ops.push_main()

        assert result == 'pushed'
        _, origin_sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)
        assert origin_sha == local_sha

    async def test_push_main_noop_when_disabled(
        self, git_repo_with_remote: tuple[Path, Path],
    ):
        """push_after_advance=False short-circuits to 'noop' without contacting origin."""
        origin, local = git_repo_with_remote
        cfg = GitConfig(push_after_advance=False)
        git_ops = GitOps(cfg, local)

        (local / 'local.txt').write_text('local\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'local commit'], cwd=local)
        _, origin_sha_before, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)

        result = await git_ops.push_main()

        assert result == 'noop'
        _, origin_sha_after, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)
        assert origin_sha_after == origin_sha_before  # origin unchanged

    async def test_push_main_rejected_on_diverged_origin(
        self, git_repo_with_remote: tuple[Path, Path], caplog,
    ):
        """When origin has commits we lack, push must be rejected and NOT forced."""
        origin, local = git_repo_with_remote
        git_ops = GitOps(GitConfig(push_after_advance=True), local)

        # Origin gets a commit we don't have
        await _push_n_commits_to_origin(origin, 1, prefix='diverge')

        # Local diverges with its own commit (without fetching/merging)
        (local / 'local.txt').write_text('local\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'local divergent commit'], cwd=local)

        _, origin_sha_before, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)

        with caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'):
            result = await git_ops.push_main()

        assert result == 'rejected'
        # Origin must be unchanged — no force-push
        _, origin_sha_after, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)
        assert origin_sha_after == origin_sha_before
        assert any('rejected (non-fast-forward)' in r.message for r in caplog.records)

    async def test_push_main_error_on_unreachable_remote(
        self, tmp_path: Path, caplog,
    ):
        """Unreachable remote returns 'error' — best-effort, never raises."""
        # Local repo with origin pointing at a path that does not exist
        local = tmp_path / 'local'
        local.mkdir()
        await _run(['git', 'init', '-b', 'main'], cwd=local)
        await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=local)
        await _run(['git', 'config', 'user.name', 'Test'], cwd=local)
        (local / 'README.md').write_text('# Test\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'Initial commit'], cwd=local)
        await _run(
            ['git', 'remote', 'add', 'origin', str(tmp_path / 'does-not-exist')],
            cwd=local,
        )

        git_ops = GitOps(GitConfig(), local)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await git_ops.push_main()

        assert result == 'error'
        assert any('Push of main to origin failed' in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestResolveBranchSha:
    async def test_returns_sha_for_existing_ref(self, git_ops: GitOps):
        """resolve_branch_sha returns the 40-char SHA for a branch that exists.

        Uses create_worktree to materialise a task/resolve-1 branch, then
        asserts the returned SHA matches a direct rev-parse call.
        """
        wt_info = await git_ops.create_worktree('resolve-1')
        # Confirm the branch was created
        assert wt_info is not None

        resolved = await git_ops.resolve_branch_sha('task/resolve-1')

        # Get the expected SHA via _run
        _, expected_sha, _ = await _run(
            ['git', 'rev-parse', 'task/resolve-1'],
            cwd=git_ops.project_root,
        )
        expected_sha = expected_sha.strip()

        assert resolved is not None
        assert resolved == expected_sha
        assert len(resolved) == 40

    async def test_returns_none_for_missing_ref(self, git_ops: GitOps):
        """resolve_branch_sha returns None (not empty string, not an exception)
        when the branch ref does not exist.

        Regression lock: a future refactor must not silently switch to raising
        or returning '' — both would break the harness fallback path.
        """
        result = await git_ops.resolve_branch_sha('task/does-not-exist')
        assert result is None

    @pytest.mark.parametrize(
        'bad_ref',
        [
            'task/does-not-exist',   # simply absent branch
            'not a valid ref',       # contains spaces — syntactically malformed
            '..bad..',               # double-dot traversal form — rejected by git
        ],
    )
    async def test_returns_none_for_bad_refs(self, git_ops: GitOps, bad_ref: str):
        """resolve_branch_sha returns None for any ref git cannot resolve.

        Covers both 'missing' (rc=128 from rev-parse not finding the ref) and
        'malformed' (rc=128 from git rejecting the name) error modes, locking
        in the rc-based fallback contract for the harness fallback path.
        """
        result = await git_ops.resolve_branch_sha(bad_ref)
        assert result is None


@pytest.mark.asyncio
class TestFindMergeMarker:
    """Real-git tests for GitOps.find_merge_marker.

    Tests cover the four cases described in the plan:
    (a) branch deleted with a merge marker on main → returns SHA
    (b) branch still exists → returns None (resolve_branch_sha gate)
    (c) branch never existed, no marker → returns None
    (d) substring safety: task/1 query must not match 'Merge task/10 into main'
    """

    async def test_returns_merge_sha_when_branch_deleted_with_marker(
        self, git_ops: GitOps
    ):
        """find_merge_marker returns the merge commit SHA when the branch was
        merged to main and then deleted via cleanup_worktree.

        Real git fixture: create_worktree → commit → merge_to_main → advance_main
        → cleanup_merge_worktree → cleanup_worktree (branch deleted), then assert
        the returned SHA matches the merge commit SHA.
        """
        tid = 'mm-1'
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add {tid}')

        result = await git_ops.merge_to_main(wt_info.path, tid)
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        adv = await git_ops.advance_main(result.merge_commit)
        assert adv == 'advanced'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # Branch is now deleted — find_merge_marker should find the merge commit
        marker_sha = await git_ops.find_merge_marker(f'task/{tid}')

        assert marker_sha is not None
        assert marker_sha == result.merge_commit
        assert len(marker_sha) == 40

    async def test_returns_none_when_branch_still_exists(self, git_ops: GitOps):
        """find_merge_marker returns None when the branch ref still exists,
        even if there happens to be a merge commit matching the pattern.

        resolve_branch_sha gates the git-log search: if the branch is still
        present, is_ancestor is the authoritative check.
        """
        tid = 'still-here'
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        # Branch created but NOT merged — still exists

        result = await git_ops.find_merge_marker(f'task/{tid}')

        assert result is None

    async def test_returns_none_when_branch_never_existed_no_marker(
        self, git_ops: GitOps
    ):
        """find_merge_marker returns None when no such branch was ever created
        and no merge commit matching the pattern exists on main.
        """
        result = await git_ops.find_merge_marker('task/never-existed')
        assert result is None

    async def test_substring_safety_task_1_does_not_match_task_10(
        self, git_ops: GitOps
    ):
        """Substring safety: merging task/10 writes 'Merge task/10 into main'.
        find_merge_marker('task/1') must NOT match this commit.

        The trailing ' into ' literal in the --fixed-strings --grep pattern
        means 'Merge task/1 into ' is not a substring of 'Merge task/10 into main'.
        """
        # Merge task/10 and delete branch
        tid = '10'
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'task_{tid}.py').write_text(f'task_{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add task {tid}')

        result = await git_ops.merge_to_main(wt_info.path, tid)
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        adv = await git_ops.advance_main(result.merge_commit)
        assert adv == 'advanced'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # task/10 branch is deleted; merge marker 'Merge task/10 into main' exists
        # find_merge_marker('task/1') must NOT find it
        marker_sha = await git_ops.find_merge_marker('task/1')
        assert marker_sha is None

    async def test_returns_none_when_branch_deleted_without_merging(
        self, git_ops: GitOps
    ):
        """Branch was created and abandoned: deleted without ever being merged.

        This is subtly different from 'branch never existed' (case c): the
        branch ref existed at some point but was cleaned up without writing a
        merge commit on main.  find_merge_marker must return None because there
        is no matching marker subject to find.
        """
        tid = 'abandoned-1'
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add {tid}')

        # Delete the worktree and branch WITHOUT merging to main
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # Branch is gone but no merge marker was ever written on main
        result = await git_ops.find_merge_marker(f'task/{tid}')
        assert result is None

    async def test_returns_single_sha_when_branch_reopened_with_two_merges(
        self, git_ops: GitOps
    ):
        """find_merge_marker returns exactly one 40-char SHA in the re-opened-task
        scenario described in the function's own docstring: a task branch is merged,
        deleted, then re-created under the same name, merged again, and deleted again.
        Both merge commits share the same subject ('Merge task/reopened-1 into main'),
        so a git-log invocation with conflicting --max-count=1 and -n 5000 flags would
        return both SHAs newline-joined (last-wins: -n 5000 overrides --max-count=1),
        corrupting done_provenance={'commit': marker_sha} in harness reconcile.
        After dropping -n 5000, --max-count=1 alone ensures a single SHA is returned.
        """
        tid = 'reopened-1'

        # --- Iteration 1 ---
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'iter1_{tid}.py').write_text(f'iter1_{tid} = True\n')
        await git_ops.commit(wt_info.path, 'Add iter1')

        result1 = await git_ops.merge_to_main(wt_info.path, tid)
        assert result1.success
        assert result1.merge_commit is not None
        assert result1.merge_worktree is not None

        adv1 = await git_ops.advance_main(result1.merge_commit)
        assert adv1 == 'advanced'

        await git_ops.cleanup_merge_worktree(result1.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)
        first_sha = result1.merge_commit

        # --- Iteration 2 (same tid — branch was deleted, so this is a fresh branch) ---
        wt_info2 = await git_ops.create_worktree(tid)
        assert wt_info2 is not None
        (wt_info2.path / f'iter2_{tid}.py').write_text(f'iter2_{tid} = True\n')
        await git_ops.commit(wt_info2.path, 'Add iter2')

        result2 = await git_ops.merge_to_main(wt_info2.path, tid)
        assert result2.success
        assert result2.merge_commit is not None
        assert result2.merge_worktree is not None

        adv2 = await git_ops.advance_main(result2.merge_commit)
        assert adv2 == 'advanced'

        await git_ops.cleanup_merge_worktree(result2.merge_worktree)
        await git_ops.cleanup_worktree(wt_info2.path, tid)
        second_sha = result2.merge_commit

        # Both merge commits exist on main with the same subject.
        assert first_sha != second_sha  # sanity: two distinct commits

        # --- Assertion ---
        marker_sha = await git_ops.find_merge_marker(f'task/{tid}')

        assert marker_sha is not None
        assert '\n' not in marker_sha   # anti-multiline regression
        assert len(marker_sha) == 40    # single-SHA shape
        assert marker_sha == second_sha  # most-recent first (reverse chrono + --max-count=1)


@pytest.mark.asyncio
class TestMergeSubjectContract:
    """End-to-end contract: the merge subject written to main by merge_to_main
    equals _merge_subject output, and find_merge_marker locates that commit.

    If either the writer or reader drifts from _merge_subject (e.g. an inline
    f-string replaces the helper call with a different format), at least one of
    the two roundtrip assertions will fail.
    """

    async def test_merge_subject_roundtrip(
        self, git_ops: GitOps
    ) -> None:
        """Assert the on-main subject equals _merge_subject output and that
        find_merge_marker returns the same SHA.
        """
        tid = 'contract-1'
        full_branch = f'task/{tid}'

        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add {tid}')

        result = await git_ops.merge_to_main(wt_info.path, tid)
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        adv = await git_ops.advance_main(result.merge_commit)
        assert adv == 'advanced'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # End-to-end roundtrip: the subject on main must equal helper output
        _, subject, _ = await _run(
            ['git', 'log', '--format=%s', '-n', '1', result.merge_commit],
            cwd=git_ops.project_root,
        )
        assert subject == _merge_subject(full_branch, git_ops.config.main_branch)

        # find_merge_marker must locate the same commit
        marker_sha = await git_ops.find_merge_marker(full_branch)
        assert marker_sha == result.merge_commit


class TestMergeSubject:
    """Unit tests for the _merge_subject helper.

    Locks the canonical format: 'Merge {branch} into {main_branch}'.
    This helper is the single source of truth consumed by merge_to_main,
    advance_main (retry path), and find_merge_marker.
    """

    @pytest.mark.parametrize(
        'branch, main_branch, expected',
        [
            ('task/1', 'main', 'Merge task/1 into main'),
            ('task/123', 'main', 'Merge task/123 into main'),
            ('task/v1.0', 'develop', 'Merge task/v1.0 into develop'),
        ],
    )
    def test_canonical_format(
        self, branch: str, main_branch: str, expected: str
    ) -> None:
        """_merge_subject returns 'Merge {branch} into {main_branch}'."""
        assert _merge_subject(branch, main_branch) == expected


@pytest.mark.asyncio
class TestRunWorktreeMissing:
    """``_run`` raises typed :class:`WorktreeMissing` for deleted cwd.

    Distinguishes a deleted task worktree (recoverable race) from a
    PATH-missing binary (real bug).
    """

    async def test_deleted_cwd_raises_worktree_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / 'gone'
        # Path does not exist — pre-flight should classify it.
        with pytest.raises(WorktreeMissing) as exc:
            await _run(['git', 'status'], cwd=missing)
        assert exc.value.path == missing
        assert isinstance(exc.value, FileNotFoundError)

    async def test_missing_binary_raises_plain_filenotfound(
        self, tmp_path: Path
    ) -> None:
        # cwd exists, but the binary does not — must be plain FileNotFoundError,
        # not WorktreeMissing.
        with pytest.raises(FileNotFoundError) as exc:
            await _run(['definitely-not-a-real-binary-xyz'], cwd=tmp_path)
        assert not isinstance(exc.value, WorktreeMissing)

    async def test_no_cwd_does_not_classify(self) -> None:
        # cwd=None — pre-flight is skipped; missing binary surfaces as plain.
        with pytest.raises(FileNotFoundError) as exc:
            await _run(['definitely-not-a-real-binary-xyz'])
        assert not isinstance(exc.value, WorktreeMissing)


def _write_stored_title(worktree: Path, title: str) -> None:
    """Write a ``.task/metadata.json`` carrying ``title`` into *worktree*."""
    task_dir = worktree / '.task'
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / 'metadata.json').write_text(
        json.dumps({'title': title, 'task_id': worktree.name})
    )


@pytest.mark.asyncio
class TestWorktreeReuseIdentityGuard:
    """Fix C reuse path — create_worktree(expected_title=...) quarantines a
    recycled-id worktree instead of reusing it for the wrong task."""

    async def test_reuse_when_title_matches(self, git_ops: GitOps):
        info = await git_ops.create_worktree('reuse-match')
        _write_stored_title(info.path, 'Build the frobnicator')

        info2 = await git_ops.create_worktree(
            'reuse-match', expected_title='Build the frobnicator',
        )
        assert info2.path == info.path  # reused, not recreated
        assert not git_ops.quarantine_base.exists()  # nothing quarantined

    async def test_quarantine_and_fresh_create_on_mismatch(self, git_ops: GitOps):
        info = await git_ops.create_worktree('recycled')
        _write_stored_title(info.path, 'Trajectory beta: spline solver')
        # A tracked file that belongs to the WRONG (orphaned) task.
        (info.path / 'spline.rs').write_text('fn solve() {}\n')
        await git_ops.commit(info.path, 'trajectory WIP')

        info2 = await git_ops.create_worktree(
            'recycled', expected_title='Cycle-breaker beta: dedup edges',
        )

        # Fresh worktree at the original path, WITHOUT the orphan's file.
        assert info2.path == info.path
        assert info2.path.exists()
        assert (info2.path / 'README.md').exists()
        assert not (info2.path / 'spline.rs').exists()

        # The orphan was relocated to the sibling quarantine base, file intact.
        assert git_ops.quarantine_base.exists()
        quarantined = list(git_ops.quarantine_base.glob('recycled-*'))
        assert len(quarantined) == 1
        assert (quarantined[0] / 'spline.rs').exists()

    async def test_none_expected_title_skips_guard(self, git_ops: GitOps):
        info = await git_ops.create_worktree('reuse-noguard')
        _write_stored_title(info.path, 'Original task')

        # expected_title omitted → guard skipped → reused despite any mismatch.
        info2 = await git_ops.create_worktree('reuse-noguard')
        assert info2.path == info.path
        assert not git_ops.quarantine_base.exists()


@pytest.mark.asyncio
class TestOrphanWorktreeHelpers:
    """Fix B git_ops helpers: quarantine_worktree, worktree_has_unsaved_work."""

    async def test_quarantine_moves_and_preserves_committed_work(self, git_ops: GitOps):
        info = await git_ops.create_worktree('q-task')
        (info.path / 'work.py').write_text('x = 1\n')
        await git_ops.commit(info.path, 'committed work')

        dest = await git_ops.quarantine_worktree(info.path, 'q-task', 'unit-test')

        assert dest is not None
        assert dest.parent == git_ops.quarantine_base
        assert (dest / 'work.py').exists()  # committed work preserved
        assert not info.path.exists()  # moved out of the scanned base
        # Original branch was renamed away.
        _, branches, _ = await _run(
            ['git', 'branch', '--list', 'task/q-task'], cwd=git_ops.project_root,
        )
        assert branches.strip() == ''

    async def test_unsaved_work_false_when_clean_no_commits(self, git_ops: GitOps):
        info = await git_ops.create_worktree('clean-wt')
        assert await git_ops.worktree_has_unsaved_work(info.path, 'clean-wt') is False

    async def test_unsaved_work_true_on_commit(self, git_ops: GitOps):
        info = await git_ops.create_worktree('commit-wt')
        (info.path / 'f.py').write_text('a = 1\n')
        await git_ops.commit(info.path, 'a commit beyond main')
        assert await git_ops.worktree_has_unsaved_work(info.path, 'commit-wt') is True

    async def test_unsaved_work_true_on_dirty_tree(self, git_ops: GitOps):
        info = await git_ops.create_worktree('dirty-wt')
        (info.path / 'untracked.py').write_text('a = 1\n')  # uncommitted WIP
        assert await git_ops.worktree_has_unsaved_work(info.path, 'dirty-wt') is True

    async def test_unsaved_work_failsafe_on_missing_branch(self, git_ops: GitOps):
        info = await git_ops.create_worktree('exists-wt')
        # Query a branch with no task/ ref → rev-list fails → fail-safe True.
        assert await git_ops.worktree_has_unsaved_work(info.path, 'no-such-branch') is True


@pytest.mark.asyncio
class TestTrainPredecessor:
    async def test_returns_predecessor_for_order_gt_zero(self, git_ops: GitOps):
        from orchestrator.git_ops import TrainMembership, TrainPredecessor

        # order=1, members=['a', 'b'] → predecessor is members[0] = 'a'
        result = await git_ops._train_predecessor(
            TrainMembership(id='T1', order=1, members=['a', 'b'])
        )
        assert isinstance(result, TrainPredecessor)
        assert result.task_id == 'a'
        assert result.branch == 'task/a'

        # order=2, members=['a', 'b', 'c'] → predecessor is members[1] = 'b'
        result2 = await git_ops._train_predecessor(
            TrainMembership(id='T1', order=2, members=['a', 'b', 'c'])
        )
        assert result2.task_id == 'b'
        assert result2.branch == 'task/b'

    async def test_raises_when_invariants_violated(self, git_ops: GitOps):
        from orchestrator.git_ops import TrainMembership

        # (a) order=0: caller should not invoke for degenerate trains
        with pytest.raises(ValueError, match='order=0'):
            await git_ops._train_predecessor(TrainMembership(id='T1', order=0, members=['a']))

        # (b) members missing entirely
        with pytest.raises(ValueError, match='members'):
            await git_ops._train_predecessor(TrainMembership(id='T1', order=1))

        # (c) members=None
        with pytest.raises(ValueError, match='members'):
            await git_ops._train_predecessor(TrainMembership(id='T1', order=1, members=None))

        # (d) members too short for the requested order
        with pytest.raises(ValueError, match='members'):
            await git_ops._train_predecessor(
                TrainMembership(id='T1', order=2, members=['a'])
            )


@pytest.mark.asyncio
class TestCreateWorktreeTrain:
    async def test_train_none_regression(self, git_ops: GitOps):
        """train=None must produce byte-identical behaviour to the old default."""
        # Capture main SHA before creating the worktree
        _, main_sha, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root
        )
        main_sha = main_sha.strip()

        info = await git_ops.create_worktree('feature-notrain', train=None)

        assert info.base_commit == main_sha
        assert info.stale_commits is None  # no remote in git_repo fixture

    async def test_train_order_zero_degenerate(self, git_ops: GitOps):
        """order=0 is degenerate — must branch from main, same as train=None."""
        from orchestrator.git_ops import TrainMembership

        _, main_sha, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root
        )
        main_sha = main_sha.strip()

        info = await git_ops.create_worktree(
            'feature-order0',
            train=TrainMembership(id='T1', order=0, members=['feature-z']),
        )

        assert info.base_commit == main_sha

    async def test_train_order_gt_zero_branches_from_predecessor_tip(
        self, git_ops: GitOps
    ):
        """PRD § 10 scenario-2: β branches from α's tip, not from main."""
        from orchestrator.git_ops import TrainMembership

        # Create α worktree and commit one file in it
        alpha_info = await git_ops.create_worktree('alpha')
        (alpha_info.path / 'alpha.py').write_text('x = 1\n')
        alpha_tip = await git_ops.commit(alpha_info.path, 'alpha commit')
        assert alpha_tip is not None

        # Create β with train metadata pointing to α as predecessor
        beta_info = await git_ops.create_worktree(
            'beta',
            train=TrainMembership(id='T1', order=1, members=['alpha', 'beta']),
        )

        # (1) WorktreeInfo.base_commit == α's branch tip SHA
        assert beta_info.base_commit == alpha_tip

        # (2) stale_commits is None for train-based worktrees
        assert beta_info.stale_commits is None

        # (3) git merge-base task/beta task/alpha == α's tip (β forks from α)
        _, mb, _ = await _run(
            ['git', 'merge-base', 'task/beta', 'task/alpha'],
            cwd=git_ops.project_root,
        )
        assert mb.strip() == alpha_tip

        # (4) git log task/beta..task/alpha is empty (β already has all of α's commits)
        _, log_out, _ = await _run(
            ['git', 'log', 'task/beta..task/alpha', '--oneline'],
            cwd=git_ops.project_root,
        )
        assert log_out.strip() == ''

    async def test_missing_predecessor_branch_raises(self, git_ops: GitOps):
        """RuntimeError when the predecessor branch doesn't exist; no stale dir left."""
        from orchestrator.git_ops import TrainMembership

        with pytest.raises(RuntimeError) as exc_info:
            await git_ops.create_worktree(
                'beta',
                train=TrainMembership(
                    id='T1', order=1, members=['nonexistent-task', 'beta']
                ),
            )

        msg = str(exc_info.value)
        assert 'task/nonexistent-task' in msg
        assert 'T1' in msg
        assert 'beta' in msg

        # No stale worktree directory left on disk
        worktree_path = git_ops.worktree_base / 'beta'
        assert not worktree_path.exists()

    async def test_three_member_chain(self, git_ops: GitOps):
        """PRD § 4: transitive stacking — γ contains both α and β commits."""
        from orchestrator.git_ops import TrainMembership

        # α: create worktree, commit one file
        alpha_info = await git_ops.create_worktree('chain-alpha')
        (alpha_info.path / 'alpha.py').write_text('a = 1\n')
        alpha_tip = await git_ops.commit(alpha_info.path, 'alpha commit')
        assert alpha_tip is not None

        # β: stacks on α
        beta_info = await git_ops.create_worktree(
            'chain-beta',
            train=TrainMembership(
                id='T1', order=1, members=['chain-alpha', 'chain-beta', 'chain-gamma']
            ),
        )
        (beta_info.path / 'beta.py').write_text('b = 2\n')
        beta_tip = await git_ops.commit(beta_info.path, 'beta commit')
        assert beta_tip is not None

        # γ: stacks on β
        gamma_info = await git_ops.create_worktree(
            'chain-gamma',
            train=TrainMembership(
                id='T1', order=2, members=['chain-alpha', 'chain-beta', 'chain-gamma']
            ),
        )

        # (1) γ's base_commit == β's tip (NOT α's, NOT main)
        assert gamma_info.base_commit == beta_tip

        # (2) git log task/chain-gamma..task/chain-beta is empty
        _, log_bg, _ = await _run(
            ['git', 'log', 'task/chain-gamma..task/chain-beta', '--oneline'],
            cwd=git_ops.project_root,
        )
        assert log_bg.strip() == ''

        # (3) git log task/chain-gamma..task/chain-alpha is also empty (transitive)
        _, log_ag, _ = await _run(
            ['git', 'log', 'task/chain-gamma..task/chain-alpha', '--oneline'],
            cwd=git_ops.project_root,
        )
        assert log_ag.strip() == ''

    async def test_reuse_path_rebases_to_predecessor_tip(
        self, git_ops: GitOps
    ):
        """Requeued stacked train member must rebase onto predecessor tip (not main).

        Regression test for the reuse-existing-worktree path (git_ops.py):
          - beta is created stacked on alpha
          - beta's worktree already exists (simulating a requeue)
          - calling create_worktree again for beta must produce
            WorktreeInfo.base_commit == alpha_tip (predecessor tip, NOT main).
        """
        from orchestrator.git_ops import TrainMembership

        # α: create worktree, commit one file
        alpha_info = await git_ops.create_worktree('reuse-alpha')
        (alpha_info.path / 'alpha.py').write_text('a = 1\n')
        alpha_tip = await git_ops.commit(alpha_info.path, 'alpha commit')
        assert alpha_tip is not None

        # β: stacks on α (fresh create)
        train = TrainMembership(id='T-reuse', order=1, members=['reuse-alpha', 'reuse-beta'])
        beta_info = await git_ops.create_worktree('reuse-beta', train=train)
        (beta_info.path / 'beta.py').write_text('b = 2\n')
        await git_ops.commit(beta_info.path, 'beta commit')

        # Simulate requeue: call create_worktree again for the same branch.
        # The reuse path fires because worktree_path already exists.
        reused_info = await git_ops.create_worktree('reuse-beta', train=train)

        # (1) base_commit must be alpha_tip (predecessor tip), NOT main's SHA
        assert reused_info.base_commit == alpha_tip

        # (2) git log task/reuse-beta..task/reuse-alpha is empty
        # (β still contains all of α's commits after the rebase)
        _, log_out, _ = await _run(
            ['git', 'log', 'task/reuse-beta..task/reuse-alpha', '--oneline'],
            cwd=git_ops.project_root,
        )
        assert log_out.strip() == ''

    async def test_reuse_missing_predecessor_branch_raises(
        self, git_ops: GitOps
    ):
        """Reused worktree with a missing predecessor branch raises RuntimeError.

        If the predecessor branch no longer exists when a stacked train member
        is requeued, the reuse path must raise RuntimeError (mirroring the
        create-path guard) rather than silently rebasing onto main.
        """
        from orchestrator.git_ops import TrainMembership

        # Create a plain (non-train) worktree so the dir exists for reuse
        await git_ops.create_worktree('reuse-guard-beta')

        # Now re-invoke with train metadata pointing to a non-existent predecessor
        with pytest.raises(RuntimeError) as exc_info:
            await git_ops.create_worktree(
                'reuse-guard-beta',
                train=TrainMembership(
                    id='T-guard', order=1,
                    members=['ghost-predecessor', 'reuse-guard-beta'],
                ),
            )

        msg = str(exc_info.value)
        assert 'task/ghost-predecessor' in msg

    async def test_reuse_conflict_degrades_gracefully_for_stacked_member(
        self, git_ops: GitOps, caplog,
    ):
        """Rebase conflict on a reused stacked member logs ERROR and returns old base.

        Contract for the graceful-degradation path (conflict branch):
          - create_worktree returns normally (no exception raised)
          - WorktreeInfo.base_commit is the pre-rebase merge-base (old α tip)
          - the conflict is logged at ERROR level with train_id and order, not
            merely WARNING, so a broken stack is observable (Suggestion 2)

        Setup: β is stacked on α; α then advances with a conflicting change to
        the same file β already modified, so git rebase aborts.
        """
        from orchestrator.git_ops import TrainMembership

        # α: create worktree, write shared file
        alpha_info = await git_ops.create_worktree('reuse-conflict-alpha')
        (alpha_info.path / 'shared.py').write_text('x = 1\n')
        alpha_tip_v1 = await git_ops.commit(alpha_info.path, 'alpha v1: x=1')
        assert alpha_tip_v1 is not None

        # β: stacked on α (fresh create), modifies the same file
        train = TrainMembership(
            id='T-conflict', order=1,
            members=['reuse-conflict-alpha', 'reuse-conflict-beta'],
        )
        beta_info = await git_ops.create_worktree('reuse-conflict-beta', train=train)
        (beta_info.path / 'shared.py').write_text('x = 3\n')
        await git_ops.commit(beta_info.path, 'beta: x=3')

        # Advance α's tip with a conflicting change to the same file.
        # Both α (x→2) and β (x→3) diverge from the original x=1, so rebasing
        # β onto the new α tip will produce a merge conflict.
        (alpha_info.path / 'shared.py').write_text('x = 2\n')
        alpha_tip_v2 = await git_ops.commit(alpha_info.path, 'alpha v2: x=2')
        assert alpha_tip_v2 is not None
        assert alpha_tip_v2 != alpha_tip_v1

        # Simulate requeue of β: fire the reuse path.
        # The rebase onto the new α tip will conflict and be aborted.
        with caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'):
            reused_info = await git_ops.create_worktree(
                'reuse-conflict-beta', train=train
            )

        # Must not raise — graceful degradation
        # base_commit falls back to the old merge-base (α tip v1, the divergence
        # point between the two branches before α advanced)
        assert reused_info.base_commit == alpha_tip_v1, (
            f'Expected base_commit={alpha_tip_v1!r} (alpha tip v1) after '
            f'conflict degradation, got {reused_info.base_commit!r}'
        )

        # ERROR must be logged with train_id and order (not merely WARNING)
        error_records = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR
        ]
        assert error_records, (
            'Expected at least one ERROR log record for stacked-member rebase '
            f'conflict; got records: {[r.getMessage() for r in caplog.records]}'
        )
        error_msgs = [r.getMessage() for r in error_records]
        assert any('T-conflict' in m for m in error_msgs), (
            f'Expected train_id "T-conflict" in ERROR log; got: {error_msgs}'
        )
        assert any('order' in m for m in error_msgs), (
            f'Expected "order" in ERROR log message; got: {error_msgs}'
        )


# ---------------------------------------------------------------------------
# TestFindInflightMergeWorktree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFindInflightMergeWorktree:
    """Tests for GitOps.find_inflight_merge_worktree(branch).

    Uses real _merge-* worktrees created by merge_to_main, verified via HEAD
    subject matching against _merge_subject.
    """

    async def _make_branch_with_commit(
        self, git_ops: GitOps, branch: str, filename: str | None = None
    ) -> WorktreeInfo:
        """Create a branch worktree, add a file, and commit."""
        name = filename or f'{branch}.py'
        wt_info = await git_ops.create_worktree(branch)
        (wt_info.path / name).write_text(f'# {branch}\n')
        await git_ops.commit(wt_info.path, f'Add {name}')
        return wt_info

    async def test_returns_worktree_path_for_matching_branch(
        self, git_ops: GitOps,
    ):
        """find_inflight_merge_worktree returns the _merge-* path for a matching branch."""
        branch = 'findme'
        wt_info = await self._make_branch_with_commit(git_ops, branch)
        result = await git_ops.merge_to_main(wt_info.path, branch)
        assert result.success, f'merge_to_main failed: {result}'

        try:
            found = await git_ops.find_inflight_merge_worktree(branch)
            assert found is not None, 'Expected a merge worktree to be found'
            assert found == result.merge_worktree
        finally:
            if result.merge_worktree is not None:
                await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_returns_none_when_no_merge_worktree_exists(
        self, git_ops: GitOps,
    ):
        """Returns None when no _merge-* worktree exists for the branch."""
        found = await git_ops.find_inflight_merge_worktree('nonexistent-branch')
        assert found is None

    async def test_returns_none_for_subject_mismatch(
        self, git_ops: GitOps,
    ):
        """Returns None when a _merge-* worktree exists but for a DIFFERENT branch.

        Specifically 'X0' must NOT match a query for 'X' — guards against
        substring-safety issues.
        """
        branch_x0 = 'prefixbranch0'
        branch_x = 'prefixbranch'

        wt_info = await self._make_branch_with_commit(git_ops, branch_x0)
        result = await git_ops.merge_to_main(wt_info.path, branch_x0)
        assert result.success, f'merge_to_main failed: {result}'

        try:
            # Searching for 'prefixbranch' must NOT match the 'prefixbranch0' worktree
            found = await git_ops.find_inflight_merge_worktree(branch_x)
            assert found is None, (
                f'Expected None for branch {branch_x!r} but got {found} '
                f'(merge worktree is for {branch_x0!r})'
            )
        finally:
            if result.merge_worktree is not None:
                await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_ignores_ordinary_task_worktrees(
        self, git_ops: GitOps,
    ):
        """Ordinary task worktrees (not _merge-*) are never returned."""
        branch = 'ordinary-task'
        await self._make_branch_with_commit(git_ops, branch)
        # the created worktree is a task worktree, NOT a _merge-* worktree
        found = await git_ops.find_inflight_merge_worktree(branch)
        assert found is None, (
            f'find_inflight_merge_worktree must ignore task worktrees, got {found}'
        )


# ---------------------------------------------------------------------------
# TestReclaimWorktreeBuildArtifacts — unit tests for the new helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReclaimWorktreeBuildArtifacts:
    """GitOps.reclaim_worktree_build_artifacts removes regenerable build dirs.

    These tests cover the four cases described in the plan:
      (a) nominal reap — target/ exists, gets removed, path returned
      (b) idempotent no-op — calling again on the same (now-clean) worktree
      (c) dir_names override — only the named dir is reaped; 'target' survives
      (d) never-raises — non-existent worktree path returns [] without raising
    """

    async def test_nominal_reap(self, git_ops: GitOps):
        """(a) Creates a worktree, populates target/, reaps it via the helper."""
        wt_info = await git_ops.create_worktree('reap-nominal')
        wt = wt_info.path

        # Simulate a Rust/Cargo build cache under the worktree.
        target_dir = wt / 'target'
        target_dir.mkdir()
        (target_dir / 'cache.bin').write_bytes(b'\x00' * 1024)

        removed = await git_ops.reclaim_worktree_build_artifacts(wt)

        assert not target_dir.exists(), (
            'target/ must be removed by reclaim_worktree_build_artifacts'
        )
        assert removed == [target_dir], (
            f'expected [target_dir], got {removed}'
        )

    async def test_idempotent_noop(self, git_ops: GitOps):
        """(b) Calling a second time (target/ already gone) returns [] and does not raise."""
        wt_info = await git_ops.create_worktree('reap-idempotent')
        wt = wt_info.path

        # First call: create and reap target/
        target_dir = wt / 'target'
        target_dir.mkdir()
        (target_dir / 'file.txt').write_text('data\n')
        await git_ops.reclaim_worktree_build_artifacts(wt)
        assert not target_dir.exists()

        # Second call: target/ is absent — must return [] and not raise
        removed = await git_ops.reclaim_worktree_build_artifacts(wt)
        assert removed == [], (
            f'second call must return [] when target/ is absent, got {removed}'
        )

    async def test_explicit_dir_names_override(self, git_ops: GitOps):
        """(c) dir_names=['build'] reaps only 'build', leaves 'target' intact."""
        wt_info = await git_ops.create_worktree('reap-override')
        wt = wt_info.path

        build_dir = wt / 'build'
        build_dir.mkdir()
        (build_dir / 'artifact.o').write_bytes(b'\xff')

        target_dir = wt / 'target'
        target_dir.mkdir()
        (target_dir / 'cache.bin').write_bytes(b'\x00')

        removed = await git_ops.reclaim_worktree_build_artifacts(
            wt, dir_names=['build'],
        )

        assert not build_dir.exists(), 'build/ must be removed by override'
        assert target_dir.exists(), (
            'target/ must NOT be removed when dir_names=[\'build\']'
        )
        assert removed == [build_dir], (
            f'expected [build_dir], got {removed}'
        )

    async def test_never_raises_for_nonexistent_worktree(self, git_ops: GitOps):
        """(d) Non-existent worktree path returns [] without raising."""
        bogus_path = git_ops.worktree_base / 'does-not-exist-12345'
        assert not bogus_path.exists()

        removed = await git_ops.reclaim_worktree_build_artifacts(bogus_path)
        assert removed == [], (
            f'expected [] for non-existent path, got {removed}'
        )


# ---------------------------------------------------------------------------
# Task 1692 — persistent warm merge-verify worktree
# ---------------------------------------------------------------------------


async def _get_merge_commit(git_ops: GitOps, branch_name: str, filename: str) -> str:
    """Helper: create a feature branch, commit a file, merge to main, return merge_commit."""
    wt_info = await git_ops.create_worktree(branch_name)
    (wt_info.path / filename).write_text(f'{branch_name} = True\n')
    await git_ops.commit(wt_info.path, f'Add {filename}')
    result = await git_ops.merge_to_main(wt_info.path, branch_name)
    assert result.success and result.merge_commit
    return result.merge_commit


def test_git_config_persistent_merge_worktree_knobs():
    """GitConfig knobs for the persistent warm merge-verify worktree feature.

    Step 1 (RED): these fields do not yet exist — test must fail before impl.
    """
    # Defaults: feature off
    cfg_default = GitConfig()
    assert cfg_default.persistent_merge_worktree is False, (
        'persistent_merge_worktree must default to False (feature off)'
    )
    assert cfg_default.persistent_merge_worktree_safety_valve_every_n == 0, (
        'safety_valve_every_n must default to 0 (disabled)'
    )

    # Round-trip True
    cfg_on = GitConfig(persistent_merge_worktree=True)
    assert cfg_on.persistent_merge_worktree is True, (
        'persistent_merge_worktree=True must round-trip'
    )
    # safety_valve_every_n independent
    assert cfg_on.persistent_merge_worktree_safety_valve_every_n == 0

    # safety_valve_every_n set explicitly
    cfg_valve = GitConfig(
        persistent_merge_worktree=True,
        persistent_merge_worktree_safety_valve_every_n=5,
    )
    assert cfg_valve.persistent_merge_worktree is True
    assert cfg_valve.persistent_merge_worktree_safety_valve_every_n == 5

    # safety_valve_every_n >= 0 enforced (0 means disabled)
    import pydantic
    with pytest.raises((pydantic.ValidationError, ValueError)):
        GitConfig(persistent_merge_worktree_safety_valve_every_n=-1)


@pytest.mark.asyncio
class TestPersistentMergeWorktree:
    """Integration tests for reset_persistent_merge_worktree and its exemptions.

    Steps 3–10 of task 1692.
    """

    # ------------------------------------------------------------------
    # Step 3 — create-once path
    # ------------------------------------------------------------------

    async def test_persistent_merge_worktree_path_property(
        self, git_ops: GitOps,
    ):
        """persistent_merge_worktree_path == worktree_base / '_merge-verify'."""
        assert git_ops.persistent_merge_worktree_path == (
            git_ops.worktree_base / '_merge-verify'
        )

    async def test_reset_persistent_merge_worktree_create_once(
        self, git_ops: GitOps,
    ):
        """reset_persistent_merge_worktree creates worktree on first call.

        Step 3 (RED): method/property absent today — test must fail before impl.
        """
        merge_commit = await _get_merge_commit(
            git_ops, 'warm-create-1', 'warm_create.py',
        )

        warm_path = await git_ops.reset_persistent_merge_worktree(merge_commit)

        # Returns the fixed path
        assert warm_path == git_ops.persistent_merge_worktree_path
        assert warm_path.exists()

        # Path is a registered git worktree
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        assert rc == 0
        registered_paths = [
            line[len('worktree '):].strip()
            for line in out.splitlines()
            if line.startswith('worktree ')
        ]
        assert str(warm_path) in registered_paths, (
            f'_merge-verify not in registered worktrees: {registered_paths}'
        )

        # HEAD of warm worktree == merge_commit
        _, head_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=warm_path,
        )
        assert head_sha.strip() == merge_commit.strip(), (
            f'warm worktree HEAD {head_sha.strip()!r} != merge_commit {merge_commit.strip()!r}'
        )

    # ------------------------------------------------------------------
    # Step 5 — reset-in-place on an EXISTING warm worktree
    # ------------------------------------------------------------------

    async def test_reset_persistent_merge_worktree_reset_in_place(
        self, git_ops: GitOps,
    ):
        """reset_persistent_merge_worktree resets in-place on a second call.

        Verifies:
        - Still the same single registered path after reset.
        - HEAD updated to the new merge_commit B.
        - Tracked source reflects B (not A).
        - target/cache.bin STILL EXISTS (target/ retained → warm).
        - stray.txt was removed (git clean -xfd cleaned except target/).

        Step 5 (RED): reset-in-place branch not yet implemented.
        """
        # --- First reset: create at merge_commit A ---
        merge_commit_a = await _get_merge_commit(
            git_ops, 'warm-reset-a', 'warm_a.py',
        )
        warm_path = await git_ops.reset_persistent_merge_worktree(merge_commit_a)
        assert warm_path.exists()

        # Simulate a warm build: write a build artifact and a stray file
        target_dir = warm_path / 'target'
        target_dir.mkdir()
        cache_bin = target_dir / 'cache.bin'
        cache_bin.write_bytes(b'\xde\xad\xbe\xef')
        stray_txt = warm_path / 'stray.txt'
        stray_txt.write_text('stray untracked\n')

        # --- Second reset: create a different merge_commit B ---
        merge_commit_b = await _get_merge_commit(
            git_ops, 'warm-reset-b', 'warm_b.py',
        )
        warm_path_b = await git_ops.reset_persistent_merge_worktree(merge_commit_b)

        # Same single registered path
        assert warm_path_b == git_ops.persistent_merge_worktree_path
        assert warm_path_b == warm_path, 'path must not change on second call'

        # Only one _merge-verify worktree registered
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        assert rc == 0
        registered = [
            line[len('worktree '):].strip()
            for line in out.splitlines()
            if line.startswith('worktree ')
        ]
        merge_verify_paths = [p for p in registered if p.endswith('_merge-verify')]
        assert len(merge_verify_paths) == 1, (
            f'expected exactly 1 _merge-verify registration, got {merge_verify_paths}'
        )

        # HEAD == merge_commit_b (not A)
        _, head_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=warm_path_b,
        )
        assert head_sha.strip() == merge_commit_b.strip(), (
            f'HEAD should be B={merge_commit_b[:8]}, got {head_sha.strip()[:8]}'
        )

        # Source reflects B: warm_b.py present, warm_a.py absent
        assert (warm_path_b / 'warm_b.py').exists(), (
            'warm_b.py (B commit) should be present in warm worktree'
        )
        assert not (warm_path_b / 'warm_a.py').exists(), (
            'warm_a.py (A commit) should NOT be present after reset to B'
        )

        # target/cache.bin STILL EXISTS (target/ retained → warm)
        assert cache_bin.exists(), (
            'target/cache.bin must be retained (warm build artifact)'
        )

        # stray.txt removed (git clean -xfd removed it)
        assert not stray_txt.exists(), (
            'stray.txt must be cleaned by git clean -xfd'
        )

    # ------------------------------------------------------------------
    # Step 7 — cleanup_merge_worktree is a no-op on the fixed path
    # ------------------------------------------------------------------

    async def test_cleanup_merge_worktree_noop_on_persistent_path(
        self, git_ops: GitOps,
    ):
        """cleanup_merge_worktree is a no-op on _merge-verify (warm survives).

        Step 7 (RED): cleanup removes the fixed path today — must fail before
        the no-op guard is added in step-8.
        """
        merge_commit = await _get_merge_commit(
            git_ops, 'warm-cleanup-1', 'warm_cleanup.py',
        )
        warm_path = await git_ops.reset_persistent_merge_worktree(merge_commit)
        assert warm_path.exists()

        # Call cleanup on the fixed path — must be a no-op
        await git_ops.cleanup_merge_worktree(warm_path)

        # Warm worktree still registered and still on disk
        assert warm_path.exists(), 'warm worktree must survive cleanup_merge_worktree'
        assert await git_ops._is_registered_worktree(warm_path), (
            'warm worktree must still be registered after cleanup call'
        )

    async def test_cleanup_merge_worktree_removes_ephemeral(
        self, git_ops: GitOps,
    ):
        """cleanup_merge_worktree DOES remove an ephemeral _merge-<uuid> worktree.

        Step 7 (RED, control): the ephemeral path must still be removed.
        """
        # Use the internal helper to create a fresh ephemeral merge worktree
        merge_wt, _ = await git_ops._create_merge_worktree()
        assert merge_wt.exists()

        await git_ops.cleanup_merge_worktree(merge_wt)

        # Ephemeral worktree is gone
        assert not merge_wt.exists(), (
            'ephemeral _merge-<uuid> must be removed by cleanup_merge_worktree'
        )
        assert not await git_ops._is_registered_worktree(merge_wt), (
            'ephemeral _merge-<uuid> must be unregistered after cleanup'
        )

    # ------------------------------------------------------------------
    # Step 9 — _iter_merge_worktrees exempts _merge-verify
    # ------------------------------------------------------------------

    async def test_prune_skips_persistent_worktree(self, git_ops: GitOps):
        """prune_stale_merge_worktrees removes ephemeral but NOT _merge-verify.

        Step 9 (RED): prune force-removes _merge-verify today.
        """
        # Register the warm worktree
        merge_commit = await _get_merge_commit(
            git_ops, 'warm-prune-1', 'warm_prune.py',
        )
        warm_path = await git_ops.reset_persistent_merge_worktree(merge_commit)
        assert warm_path.exists()

        # Also create an ephemeral _merge-<uuid> worktree
        ephemeral_wt, _ = await git_ops._create_merge_worktree()
        assert ephemeral_wt.exists()

        # Prune: must remove ephemeral, NOT warm
        removed = await git_ops.prune_stale_merge_worktrees()

        assert any(
            '_merge-' in r and not r.endswith('_merge-verify')
            for r in removed
        ), f'ephemeral worktree must appear in removed list: {removed}'
        assert not any(
            r.endswith('_merge-verify') for r in removed
        ), f'_merge-verify must NOT appear in removed list: {removed}'

        # Warm worktree still on disk and registered
        assert warm_path.exists(), '_merge-verify must survive prune'
        assert await git_ops._is_registered_worktree(warm_path), (
            '_merge-verify must still be registered after prune'
        )

        # Ephemeral is gone
        assert not ephemeral_wt.exists(), (
            'ephemeral _merge-<uuid> must be gone after prune'
        )

    async def test_find_inflight_never_returns_persistent_path(
        self, git_ops: GitOps,
    ):
        """find_inflight_merge_worktree never returns _merge-verify.

        Step 9 (RED): _iter_merge_worktrees doesn't yet skip _merge-verify.
        """
        # Register the warm worktree at some merge commit
        merge_commit = await _get_merge_commit(
            git_ops, 'warm-inflight-1', 'warm_inflight.py',
        )
        warm_path = await git_ops.reset_persistent_merge_worktree(merge_commit)
        assert warm_path.exists()

        # find_inflight_merge_worktree(any branch) must never return warm_path
        result = await git_ops.find_inflight_merge_worktree('warm-inflight-1')
        assert result != warm_path, (
            'find_inflight_merge_worktree must never return _merge-verify'
        )

    # ------------------------------------------------------------------
    # Step 19 — multi-dir reap_build_artifact_dirs regression
    # ------------------------------------------------------------------

    async def test_reset_persistent_merge_worktree_multi_artifact_dirs(
        self, git_config: GitConfig, git_repo: Path,
    ):
        """Multi-dir reap_build_artifact_dirs: ALL configured dirs retained.

        Regression test for step-19: the buggy per-dir git-clean loop calls
        ``git clean -xfd -e build`` (which removes dist/) then
        ``git clean -xfd -e dist`` (which removes build/) — with >1 dir NONE
        survive, defeating the warm-cache purpose.

        With the fix (step-20: single invocation with all -e flags) both dirs
        must be retained after the reset-in-place.
        """
        cfg = git_config.model_copy(
            update={'reap_build_artifact_dirs': ['build', 'dist']}
        )
        multi_ops = GitOps(cfg, git_repo)

        # Create warm worktree at merge_commit_a
        merge_commit_a = await _get_merge_commit(
            multi_ops, 'multi-dir-a', 'multi_a.py',
        )
        warm_path = await multi_ops.reset_persistent_merge_worktree(merge_commit_a)
        assert warm_path.exists()

        # Simulate warm build artifacts in BOTH configured dirs + a stray file
        build_dir = warm_path / 'build'
        dist_dir = warm_path / 'dist'
        build_dir.mkdir()
        dist_dir.mkdir()
        build_cache = build_dir / 'cache.bin'
        dist_out = dist_dir / 'out.bin'
        build_cache.write_bytes(b'\xca\xfe\xba\xbe')
        dist_out.write_bytes(b'\xfe\xed\xfa\xce')
        stray_txt = warm_path / 'stray.txt'
        stray_txt.write_text('stray untracked\n')

        # Create a second merge_commit_b and reset in place
        merge_commit_b = await _get_merge_commit(
            multi_ops, 'multi-dir-b', 'multi_b.py',
        )
        await multi_ops.reset_persistent_merge_worktree(merge_commit_b)

        # BOTH configured build-artifact dirs must survive (warm retained)
        assert build_cache.exists(), (
            'build/cache.bin must be retained (build/ is a configured artifact dir)'
        )
        assert dist_out.exists(), (
            'dist/out.bin must be retained (dist/ is a configured artifact dir)'
        )

        # Stray untracked file must be cleaned
        assert not stray_txt.exists(), (
            'stray.txt must be cleaned by git clean -xfd'
        )


# ---------------------------------------------------------------------------
# Task 1952 — second persistent warm worktree `_offline-deep` (PRD δ / §5 C5)
# ---------------------------------------------------------------------------


def test_git_config_persistent_offline_deep_worktree_knob():
    """GitConfig knob for the second persistent offline-deep worktree.

    Mirrors test_git_config_persistent_merge_worktree_knobs.
    Step 3 (RED): the field does not yet exist — test must fail before impl.
    """
    # Default: feature off
    cfg_default = GitConfig()
    assert cfg_default.persistent_offline_deep_worktree is False, (
        'persistent_offline_deep_worktree must default to False (feature off)'
    )

    # Round-trip True
    cfg_on = GitConfig(persistent_offline_deep_worktree=True)
    assert cfg_on.persistent_offline_deep_worktree is True, (
        'persistent_offline_deep_worktree=True must round-trip'
    )


@pytest.mark.asyncio
class TestPersistentOfflineDeepWorktree:
    """Integration tests for reset_persistent_offline_deep_worktree and its exemptions.

    Steps 1–10 of task 1952 (PRD δ / §5 C5): a SECOND persistent warm
    worktree, dedicated to the offline-deep lane worker (β2), modeled on
    ``TestPersistentMergeWorktree`` above but with its OWN never-shared
    ``target/``.
    """

    # ------------------------------------------------------------------
    # Step 1 — module constant + path property
    # ------------------------------------------------------------------

    async def test_persistent_offline_deep_worktree_path_property(
        self, git_ops: GitOps,
    ):
        """persistent_offline_deep_worktree_path == worktree_base / '_offline-deep'.

        Step 1 (RED): constant and property are absent today
        (AttributeError/ImportError).
        """
        assert PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME == '_offline-deep'
        assert git_ops.persistent_offline_deep_worktree_path == (
            git_ops.worktree_base / '_offline-deep'
        )

    # ------------------------------------------------------------------
    # Step 5 — create-once path + prune-exemption
    # ------------------------------------------------------------------

    async def test_reset_persistent_offline_deep_worktree_create_once_and_prune_exempt(
        self, git_ops: GitOps,
    ):
        """reset_persistent_offline_deep_worktree creates on first call; prune-exempt.

        Mirrors test_reset_persistent_merge_worktree_create_once +
        test_prune_skips_persistent_worktree, combined for the offline-deep
        worktree.

        Step 5 (RED): method absent today — test must fail before impl.
        """
        merge_commit = await _get_merge_commit(
            git_ops, 'offline-deep-create-1', 'offline_deep_create.py',
        )

        warm_path = await git_ops.reset_persistent_offline_deep_worktree(merge_commit)

        # Returns the fixed path
        assert warm_path == git_ops.persistent_offline_deep_worktree_path
        assert warm_path.exists()

        # Path is a registered git worktree
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        assert rc == 0
        registered_paths = [
            line[len('worktree '):].strip()
            for line in out.splitlines()
            if line.startswith('worktree ')
        ]
        assert str(warm_path) in registered_paths, (
            f'_offline-deep not in registered worktrees: {registered_paths}'
        )

        # HEAD of warm worktree == merge_commit
        _, head_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=warm_path,
        )
        assert head_sha.strip() == merge_commit.strip(), (
            f'warm worktree HEAD {head_sha.strip()!r} != merge_commit {merge_commit.strip()!r}'
        )

        # --- Prune-exempt: create an ephemeral _merge-<uuid> worktree ---
        ephemeral_wt, _ = await git_ops._create_merge_worktree()
        assert ephemeral_wt.exists()

        removed = await git_ops.prune_stale_merge_worktrees()

        assert any(
            '_merge-' in r and not r.endswith('_offline-deep')
            for r in removed
        ), f'ephemeral worktree must appear in removed list: {removed}'
        assert not any(
            r.endswith('_offline-deep') for r in removed
        ), f'_offline-deep must NOT appear in removed list: {removed}'

        # _offline-deep worktree still on disk and registered
        assert warm_path.exists(), '_offline-deep must survive prune'
        assert await git_ops._is_registered_worktree(warm_path), (
            '_offline-deep must still be registered after prune'
        )

        # Ephemeral is gone
        assert not ephemeral_wt.exists(), (
            'ephemeral _merge-<uuid> must be gone after prune'
        )


# ---------------------------------------------------------------------------
# Task 1699 — per-host disk-persistent attempt counter (step-3 RED)
# ---------------------------------------------------------------------------


class TestHostVerifyAttemptCounter:
    """Disk-persistent per-host attempt counter for the verify-merge CLI.

    Step-3 (RED): _bump_host_verify_attempt_count absent today — AttributeError.
    """

    def test_counter_monotonically_increasing(self, git_ops: GitOps):
        """Successive calls return 1-based monotonically increasing counts."""
        assert git_ops._bump_host_verify_attempt_count() == 1
        assert git_ops._bump_host_verify_attempt_count() == 2
        assert git_ops._bump_host_verify_attempt_count() == 3

    def test_counter_persists_across_instances(self, git_config: GitConfig, git_repo: Path):
        """Counter survives across separate stateless GitOps instances (separate CLI invocations)."""
        ops1 = GitOps(git_config, git_repo)
        ops2 = GitOps(git_config, git_repo)
        ops3 = GitOps(git_config, git_repo)

        c1 = ops1._bump_host_verify_attempt_count()
        c2 = ops2._bump_host_verify_attempt_count()
        c3 = ops3._bump_host_verify_attempt_count()

        assert c1 == 1
        assert c2 == 2
        assert c3 == 3

    def test_counter_failsafe_on_corrupt_file(self, git_config: GitConfig, git_repo: Path):
        """A corrupt counter file is treated as 0; next call returns 1 (no exception)."""
        ops = GitOps(git_config, git_repo)
        # Manually write garbage into the counter file location
        ops.worktree_base.mkdir(parents=True, exist_ok=True)
        counter_file = ops.worktree_base / '.merge_verify_host_attempts'
        counter_file.write_text('not-an-integer\n')

        # Must not raise; must treat corrupt file as 0 and return 1
        result = ops._bump_host_verify_attempt_count()
        assert result == 1, f'Corrupt file must be treated as 0; got count={result}'

    def test_counter_failsafe_missing_file(self, git_config: GitConfig, git_repo: Path):
        """A missing counter file is treated as 0; first call returns 1 (no exception)."""
        ops = GitOps(git_config, git_repo)
        # Ensure no counter file exists
        counter_file = ops.worktree_base / '.merge_verify_host_attempts'
        if counter_file.exists():
            counter_file.unlink()

        result = ops._bump_host_verify_attempt_count()
        assert result == 1, f'Missing file must be treated as 0; got count={result}'


# ---------------------------------------------------------------------------
# Task 1699 — acquire_host_verify_worktree integration tests (step-5 RED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcquireHostVerifyWorktree:
    """Integration tests for acquire_host_verify_worktree.

    Step-5 (RED): method absent today — AttributeError.
    """

    async def test_knob_off_returns_ephemeral(
        self, git_config: GitConfig, git_repo: Path,
    ):
        """Knob OFF: returns an ephemeral _merge-<uuid> path, NOT the fixed warm path."""
        ops = GitOps(git_config, git_repo)
        merge_sha = await _get_merge_commit(ops, 'knob-off-a', 'knob_off_a.py')

        wt = await ops.acquire_host_verify_worktree(merge_sha)
        try:
            warm_path = ops.persistent_merge_worktree_path
            assert wt.name.startswith('_merge-'), (
                f'Expected ephemeral _merge-<uuid>; got: {wt}'
            )
            assert wt.resolve() != warm_path.resolve(), (
                f'Knob OFF must not use the fixed warm path; got: {wt}'
            )
            assert not warm_path.exists(), (
                'Warm path must NOT be created when knob is off'
            )
        finally:
            await ops.cleanup_merge_worktree(wt)

    async def test_knob_on_every_n_0_returns_warm_path(
        self, git_config: GitConfig, git_repo: Path,
    ):
        """Knob ON, every_n=0 (disabled): consecutive calls return the same fixed warm path."""
        cfg = git_config.model_copy(update={
            'persistent_merge_worktree': True,
            'persistent_merge_worktree_safety_valve_every_n': 0,
        })
        ops = GitOps(cfg, git_repo)
        warm_path = ops.persistent_merge_worktree_path

        sha1 = await _get_merge_commit(ops, 'warm-call1', 'warm_call1.py')
        wt1 = await ops.acquire_host_verify_worktree(sha1)
        assert wt1.resolve() == warm_path.resolve(), (
            f'First call with knob ON must return warm path; got: {wt1}'
        )

        # Plant a fake target/ cache to verify invariant 1 (retained across reset)
        target_dir = warm_path / 'target'
        target_dir.mkdir(exist_ok=True)
        cache_file = target_dir / 'cache.bin'
        cache_file.write_bytes(b'\xde\xad\xbe\xef')

        sha2 = await _get_merge_commit(ops, 'warm-call2', 'warm_call2.py')
        wt2 = await ops.acquire_host_verify_worktree(sha2)
        assert wt2.resolve() == warm_path.resolve(), (
            f'Second call must also return the same warm path; got: {wt2}'
        )

        # Exactly one _merge-verify registered (not multiple ephemeral worktrees)
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_repo,
        )
        worktree_paths = [
            line.split(' ', 1)[1]
            for line in out.splitlines()
            if line.startswith('worktree ')
        ]
        merge_verify_paths = [p for p in worktree_paths if '_merge-verify' in p]
        assert len(merge_verify_paths) == 1, (
            f'Exactly one _merge-verify worktree expected; got: {merge_verify_paths}'
        )

        # target/ retained (invariant 1)
        assert cache_file.exists(), (
            'target/cache.bin must be retained across reset-in-place (warm invariant 1)'
        )

    async def test_knob_on_every_n_1_always_uses_ephemeral(
        self, git_config: GitConfig, git_repo: Path,
    ):
        """Knob ON, every_n=1: every call is safety-valve-due → ephemeral, warm NOT created."""
        cfg = git_config.model_copy(update={
            'persistent_merge_worktree': True,
            'persistent_merge_worktree_safety_valve_every_n': 1,
        })
        ops = GitOps(cfg, git_repo)
        warm_path = ops.persistent_merge_worktree_path

        sha = await _get_merge_commit(ops, 'valve-n1', 'valve_n1.py')
        wt = await ops.acquire_host_verify_worktree(sha)
        try:
            assert wt.resolve() != warm_path.resolve(), (
                'every_n=1: first call must be valve-due → ephemeral path'
            )
            assert not warm_path.exists(), (
                'every_n=1: fixed warm worktree must NOT be created'
            )
        finally:
            await ops.cleanup_merge_worktree(wt)

    async def test_knob_on_every_n_3_call3_uses_ephemeral(
        self, git_config: GitConfig, git_repo: Path,
    ):
        """Knob ON, every_n=3: calls 1 and 2 → warm; call 3 → ephemeral (valve due)."""
        cfg = git_config.model_copy(update={
            'persistent_merge_worktree': True,
            'persistent_merge_worktree_safety_valve_every_n': 3,
        })
        ops = GitOps(cfg, git_repo)
        warm_path = ops.persistent_merge_worktree_path

        sha1 = await _get_merge_commit(ops, 'valve-n3-a', 'valve_n3_a.py')
        wt1 = await ops.acquire_host_verify_worktree(sha1)
        assert wt1.resolve() == warm_path.resolve(), 'Call 1 must use warm path'

        sha2 = await _get_merge_commit(ops, 'valve-n3-b', 'valve_n3_b.py')
        wt2 = await ops.acquire_host_verify_worktree(sha2)
        assert wt2.resolve() == warm_path.resolve(), 'Call 2 must use warm path'

        sha3 = await _get_merge_commit(ops, 'valve-n3-c', 'valve_n3_c.py')
        wt3 = await ops.acquire_host_verify_worktree(sha3)
        try:
            assert wt3.resolve() != warm_path.resolve(), (
                'Call 3 (every_n=3) must be valve-due → ephemeral path'
            )
        finally:
            await ops.cleanup_merge_worktree(wt3)


# ---------------------------------------------------------------------------
# parse_diff_line_ranges — pure function unit tests (step-3)
# ---------------------------------------------------------------------------

CANNED_DIFF_TWO_FILES = """\
diff --git a/src/foo.rs b/src/foo.rs
index aaaaaa..bbbbbb 100644
--- a/src/foo.rs
+++ b/src/foo.rs
@@ -10,3 +10,2 @@
 context
-deleted line 1
-deleted line 2
+added replacement
@@ -40,1 +41,1 @@
-old single line
+new single line
diff --git a/src/bar.rs b/src/bar.rs
index cccccc..dddddd 100644
--- a/src/bar.rs
+++ b/src/bar.rs
@@ -5,2 +5,3 @@
 context
-removed line
+inserted a
+inserted b
+inserted c
"""

CANNED_DIFF_INSERTION_ONLY = """\
diff --git a/src/baz.rs b/src/baz.rs
index 000000..111111 100644
--- a/src/baz.rs
+++ b/src/baz.rs
@@ -7,0 +8,3 @@
+new line 1
+new line 2
+new line 3
"""

CANNED_DIFF_DELETION_ONLY = """\
diff --git a/src/qux.rs b/src/qux.rs
index 000000..111111 100644
--- a/src/qux.rs
+++ b/src/qux.rs
@@ -20,4 +20,0 @@
-del 1
-del 2
-del 3
-del 4
"""

# File completely deleted — +++ /dev/null, no new path.
CANNED_DIFF_FILE_DELETED = """\
diff --git a/src/gone.rs b/src/gone.rs
deleted file mode 100644
index aaaaaa..000000
--- a/src/gone.rs
+++ /dev/null
@@ -1,5 +0,0 @@
-line 1
-line 2
-line 3
-line 4
-line 5
"""

# Pure rename with no content changes (R100) — no --- / +++ / hunk lines.
CANNED_DIFF_PURE_RENAME = """\
diff --git a/src/old_name.rs b/src/new_name.rs
similarity index 100%
rename from src/old_name.rs
rename to src/new_name.rs
"""

# Rename with content changes — old path vanishes, new path gets hunk ranges.
CANNED_DIFF_RENAME_WITH_CHANGES = """\
diff --git a/src/old_mod.rs b/src/new_mod.rs
similarity index 80%
rename from src/old_mod.rs
rename to src/new_mod.rs
index aaaaaa..bbbbbb 100644
--- a/src/old_mod.rs
+++ b/src/new_mod.rs
@@ -10,3 +10,4 @@
 context
-deleted line
+added line 1
+added line 2
"""

# New file — --- /dev/null, +++ b/src/new.rs.
CANNED_DIFF_NEW_FILE = """\
diff --git a/src/new.rs b/src/new.rs
new file mode 100644
index 000000..bbbbbb
--- /dev/null
+++ b/src/new.rs
@@ -0,0 +1,5 @@
+line 1
+line 2
+line 3
+line 4
+line 5
"""


class TestParseDiffLineRanges:
    """Unit tests for parse_diff_line_ranges (pure function, no git invocation)."""

    def _fn(self):
        from orchestrator.git_ops import parse_diff_line_ranges
        return parse_diff_line_ranges

    def test_two_files_multi_hunk(self):
        fn = self._fn()
        result = fn(CANNED_DIFF_TWO_FILES)
        # src/foo.rs: hunk @@ -10,3 → old_start=10, old_count=3 → (10, 12)
        #             hunk @@ -40,1 → old_start=40, old_count=1 → (40, 40)
        assert 'src/foo.rs' in result
        assert (10, 12) in result['src/foo.rs']
        assert (40, 40) in result['src/foo.rs']
        # src/bar.rs: hunk @@ -5,2 → old_start=5, old_count=2 → (5, 6)
        assert 'src/bar.rs' in result
        assert (5, 6) in result['src/bar.rs']

    def test_empty_diff_returns_empty_dict(self):
        fn = self._fn()
        assert fn('') == {}

    def test_insertion_only_hunk_uses_anchor_range(self):
        """@@ -7,0 +8,3 @@ (pure insertion) maps to a point range (7, 7)."""
        fn = self._fn()
        result = fn(CANNED_DIFF_INSERTION_ONLY)
        assert 'src/baz.rs' in result
        ranges = result['src/baz.rs']
        # old_start=7, old_count=0 → point range (7, 7)
        assert (7, 7) in ranges

    def test_deletion_only_hunk_counted_on_old_side(self):
        """@@ -20,4 +20,0 @@ maps to (20, 23)."""
        fn = self._fn()
        result = fn(CANNED_DIFF_DELETION_ONLY)
        assert 'src/qux.rs' in result
        ranges = result['src/qux.rs']
        # old_start=20, old_count=4 → (20, 23)
        assert (20, 23) in ranges

    def test_no_unexpected_keys(self):
        fn = self._fn()
        result = fn(CANNED_DIFF_TWO_FILES)
        assert set(result.keys()) == {'src/foo.rs', 'src/bar.rs'}

    # --- deleted / renamed / new-file edge cases (amendment, suggestion 1+2) ---

    def test_file_deletion_records_old_path_with_sentinel(self):
        """'+++ /dev/null' → old path represented with whole-file sentinel.

        A task that deletes a file must not appear stackable with any task that
        modifies the same file.  The sentinel ensures the shared-file intersection
        always returns non-stackable.
        """
        from orchestrator.git_ops import _WHOLE_FILE_SENTINEL
        fn = self._fn()
        result = fn(CANNED_DIFF_FILE_DELETED)
        assert 'src/gone.rs' in result, 'Deleted file path must be in result'
        assert _WHOLE_FILE_SENTINEL in result['src/gone.rs'], (
            'Deleted file must carry the whole-file sentinel'
        )
        # '/dev/null' must never appear as a key.
        assert all('/dev/null' not in k for k in result), (
            '/dev/null must not be a key in the result dict'
        )

    def test_pure_rename_records_old_path_with_sentinel(self):
        """Pure rename (R100, no hunks) → old path with sentinel; new path absent.

        The old file is gone; any task modifying the old name conflicts.
        """
        from orchestrator.git_ops import _WHOLE_FILE_SENTINEL
        fn = self._fn()
        result = fn(CANNED_DIFF_PURE_RENAME)
        assert 'src/old_name.rs' in result, 'Renamed-from path must be in result'
        assert _WHOLE_FILE_SENTINEL in result['src/old_name.rs'], (
            'Renamed-from path must carry the whole-file sentinel'
        )
        # New name has no hunks in a pure rename — absent from result.
        assert 'src/new_name.rs' not in result, (
            'Pure rename: new path has no hunk ranges, must not appear'
        )

    def test_rename_with_changes_records_old_sentinel_and_new_hunks(self):
        """Rename + content changes → old path sentinel + new path with hunk ranges."""
        from orchestrator.git_ops import _WHOLE_FILE_SENTINEL
        fn = self._fn()
        result = fn(CANNED_DIFF_RENAME_WITH_CHANGES)
        # Old path must carry the sentinel.
        assert 'src/old_mod.rs' in result, 'Old (renamed-from) path must be in result'
        assert _WHOLE_FILE_SENTINEL in result['src/old_mod.rs'], (
            'Old path must carry the whole-file sentinel'
        )
        # New path must carry hunk ranges from the content change.
        assert 'src/new_mod.rs' in result, 'New (renamed-to) path must be in result'
        # @@ -10,3 → old_start=10, old_count=3 → (10, 12)
        assert (10, 12) in result['src/new_mod.rs'], (
            'New path must record the content-change hunk range'
        )

    def test_new_file_records_new_path_normally(self):
        """'--- /dev/null' (new file) → new path recorded with hunk ranges; no /dev/null key."""
        fn = self._fn()
        result = fn(CANNED_DIFF_NEW_FILE)
        assert 'src/new.rs' in result, 'New file path must be in result'
        # @@ -0,0 +1,5 @@ → old_start=0, old_count=0 → point range (0, 0)
        assert (0, 0) in result['src/new.rs'], (
            'New-file insertion hunk must map to point range (0, 0)'
        )
        assert all('/dev/null' not in k for k in result), (
            '/dev/null must not be a key in the result dict'
        )


# ---------------------------------------------------------------------------
# get_changed_line_ranges — async method unit tests (step-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetChangedLineRanges:
    """Unit tests for GitOps.get_changed_line_ranges."""

    async def test_invokes_correct_git_command(self, git_config, git_repo):
        """Verify that git diff is called with main...ref --unified=0 --no-color."""
        ops = GitOps(git_config, git_repo)
        ref = 'task/123'

        captured_cmds: list[list[str]] = []

        async def mock_run(cmd, cwd=None):
            captured_cmds.append(cmd)
            return (0, CANNED_DIFF_TWO_FILES, '')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            await ops.get_changed_line_ranges(ref)

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert 'git' in cmd
        assert 'diff' in cmd
        assert f'{git_config.main_branch}...{ref}' in cmd
        assert '--unified=0' in cmd
        assert '--no-color' in cmd

    async def test_returns_parsed_ranges(self, git_config, git_repo):
        """Verify the returned dict matches parse_diff_line_ranges output."""
        from orchestrator.git_ops import parse_diff_line_ranges
        ops = GitOps(git_config, git_repo)
        ref = 'task/456'

        async def mock_run(cmd, cwd=None):
            return (0, CANNED_DIFF_TWO_FILES, '')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await ops.get_changed_line_ranges(ref)

        expected = parse_diff_line_ranges(CANNED_DIFF_TWO_FILES)
        assert result == expected

    async def test_empty_diff_returns_empty_dict(self, git_config, git_repo):
        ops = GitOps(git_config, git_repo)

        async def mock_run(cmd, cwd=None):
            return (0, '', '')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await ops.get_changed_line_ranges('task/789')

        assert result == {}


# ---------------------------------------------------------------------------
# Shared helpers: branch/worktree creation + clean-state assertion
# ---------------------------------------------------------------------------


async def _make_member(
    git_ops: GitOps,
    name: str,
    base_ref: str,
    filename: str,
    content: str,
) -> Path:
    """Create branch task/<name> off base_ref, write filename, commit.

    Returns the worktree path (git_ops.worktree_base/<name>).  Used by
    TestRebaseOntoArbitraryRef, TestStackTrainBranchesHappyPath, and
    TestStackTrainBranchesConflictEject — all need a branch with a single-file
    edit off an arbitrary base ref.  Mirrors the make_stacked_member helper in
    test_atomic_train_merge.py.
    """
    full_branch = f'{git_ops.config.branch_prefix}{name}'
    wt_path = git_ops.worktree_base / name
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    await _run(
        ['git', 'worktree', 'add', '-b', full_branch, str(wt_path), base_ref],
        cwd=git_ops.project_root,
    )
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=wt_path)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=wt_path)
    (wt_path / filename).write_text(content)
    await _run(['git', 'add', '-A'], cwd=wt_path)
    await _run(['git', 'commit', '-m', f'Add {filename}'], cwd=wt_path)
    return wt_path


async def _assert_no_rebase_in_progress(wt_path: Path) -> None:
    """Assert the worktree has no rebase in progress.

    Resolves the worktree's actual gitdir via ``git rev-parse --git-dir`` and
    checks that the ``rebase-merge`` directory is absent.  This is the correct
    clean-state check for both regular repos and ``git worktree add`` worktrees
    (where ``.git`` is a file pointer, not a directory).  A vacuous ``git
    status`` exit-code check does NOT distinguish mid-rebase from clean state.
    """
    _, gitdir_str, _ = await _run(['git', 'rev-parse', '--git-dir'], cwd=wt_path)
    gitdir = Path(gitdir_str.strip())
    if not gitdir.is_absolute():
        gitdir = wt_path / gitdir
    rebase_merge = gitdir / 'rebase-merge'
    assert not rebase_merge.exists(), (
        f'rebase-merge directory {rebase_merge} should not exist — '
        'rebase was not properly aborted'
    )


# ---------------------------------------------------------------------------
# step-1: rebase_onto_main generalized with optional onto= kwarg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRebaseOntoArbitraryRef:
    """rebase_onto_main(worktree, onto=<ref>) rebases onto an arbitrary ref.

    Currently RED: rebase_onto_main has no `onto` kwarg — calling it with
    onto=... raises TypeError.
    """

    async def test_onto_sibling_branch_clean_returns_true(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """Rebasing a feature (edits fileB) onto a base branch (edits fileA)
        returns True and the feature worktree now contains fileA.
        """
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        await _make_member(git_ops, 'base-rb', 'main', 'fileA.txt', 'content A\n')
        feature_wt = await _make_member(
            git_ops, 'feature-rb', 'main', 'fileB.txt', 'content B\n',
        )
        base_branch = f'{git_ops.config.branch_prefix}base-rb'

        result = await git_ops.rebase_onto_main(feature_wt, onto=base_branch)

        assert result is True
        # After rebasing onto the base branch, fileA should now be in the
        # feature worktree (it was introduced by the base branch).
        assert (feature_wt / 'fileA.txt').exists(), (
            'fileA.txt should be present after rebasing onto the base branch'
        )

    async def test_onto_sibling_branch_conflict_returns_false_and_clean(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """Rebasing a feature that conflicts (same lines as base) returns False
        and leaves the worktree clean (no .git/rebase-merge directory).
        """
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        # base edits shared.txt with one value
        await _make_member(
            git_ops, 'base-conf', 'main', 'shared.txt', 'version: alpha\n',
        )
        # feature also edits shared.txt with a different value → conflict
        feature_wt = await _make_member(
            git_ops, 'feature-conf', 'main', 'shared.txt', 'version: beta\n',
        )
        base_branch = f'{git_ops.config.branch_prefix}base-conf'

        result = await git_ops.rebase_onto_main(feature_wt, onto=base_branch)

        assert result is False
        # Worktree must be clean — rebase was aborted.
        # git status exits 0 even mid-rebase so we check the gitdir directly.
        await _assert_no_rebase_in_progress(feature_wt)


# ---------------------------------------------------------------------------
# step-3: stack_train_branches happy path (all members survive)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStackTrainBranchesHappyPath:
    """stack_train_branches happy path: anchor + two non-conflicting members.

    Currently RED: TrainStackResult and stack_train_branches do not exist.
    """

    async def test_all_members_survive_stackable_edits(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """Three branches off main each editing a different file.

        Expected: TrainStackResult(survivors=['A','B','C'], ejected=[])
        and the tip worktree (C) contains fileA, fileB, and fileC.
        """
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        main_ref = 'main'

        # Anchor: edits fileA
        await _make_member(git_ops, 'A', main_ref, 'fileA.txt', 'content A\n')
        # Member B: edits fileB (independent of A)
        await _make_member(git_ops, 'B', main_ref, 'fileB.txt', 'content B\n')
        # Member C: edits fileC (independent of A and B)
        await _make_member(git_ops, 'C', main_ref, 'fileC.txt', 'content C\n')

        result = await git_ops.stack_train_branches(['A', 'B', 'C'])

        assert isinstance(result, TrainStackResult)
        assert result.survivors == ['A', 'B', 'C']
        assert result.ejected == []

        # The tip worktree (C) must carry all three files.
        tip_wt = git_ops.worktree_base / 'C'
        assert (tip_wt / 'fileA.txt').exists(), 'tip must contain fileA (from anchor)'
        assert (tip_wt / 'fileB.txt').exists(), 'tip must contain fileB (from B)'
        assert (tip_wt / 'fileC.txt').exists(), 'tip must contain fileC (own commit)'


# ---------------------------------------------------------------------------
# step-5: conflict-eject + re-link + clean-abort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStackTrainBranchesConflictEject:
    """Conflict-eject: member with conflicting edits is ejected; branch left clean.

    Currently RED: step-4 always returns ejected=[] (no conflict detection).
    """

    async def test_tail_conflict_eject(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """TAIL conflict: C edits the same line as A → C ejected, A+B survive.

        Setup:
          - anchor A: writes foo.txt with "version: alpha\n"
          - member B: writes bar.txt (different file, clean)
          - member C: writes foo.txt with "version: gamma\n" (conflicts with A)

        Expected:
          - survivors == ['A', 'B']
          - ejected  == ['C']
          - task/C branch left clean (git status exits 0, no rebase in progress)
        """
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        main_ref = 'main'

        await _make_member(git_ops, 'ta', main_ref, 'foo.txt', 'version: alpha\n')
        await _make_member(git_ops, 'tb', main_ref, 'bar.txt', 'bar content\n')
        # C conflicts with A on foo.txt
        await _make_member(git_ops, 'tc', main_ref, 'foo.txt', 'version: gamma\n')

        result = await git_ops.stack_train_branches(['ta', 'tb', 'tc'])

        assert result.survivors == ['ta', 'tb'], f'got survivors={result.survivors}'
        assert result.ejected == ['tc'], f'got ejected={result.ejected}'

        # task/tc branch must be in a clean state (rebase properly aborted).
        # git status exits 0 even mid-rebase, so check the gitdir directly.
        tc_wt = git_ops.worktree_base / 'tc'
        await _assert_no_rebase_in_progress(tc_wt)

    async def test_middle_conflict_relink(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """MIDDLE conflict / re-link: B conflicts on A; C (clean) re-links onto A.

        Setup:
          - anchor A: writes foo.txt with "version: alpha\n"
          - member B: writes foo.txt with "version: beta\n" (conflicts with A)
          - member C: writes baz.txt (different file, clean)

        Expected:
          - survivors == ['A', 'C']   (C re-linked onto A, not the dropped B)
          - ejected  == ['B']
          - task/B branch left clean
          - tip worktree (C) contains foo.txt (from A) and baz.txt (own commit)
        """
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        main_ref = 'main'

        await _make_member(git_ops, 'ma', main_ref, 'foo.txt', 'version: alpha\n')
        # B conflicts with A on foo.txt
        await _make_member(git_ops, 'mb', main_ref, 'foo.txt', 'version: beta\n')
        await _make_member(git_ops, 'mc', main_ref, 'baz.txt', 'baz content\n')

        result = await git_ops.stack_train_branches(['ma', 'mb', 'mc'])

        assert result.survivors == ['ma', 'mc'], f'got survivors={result.survivors}'
        assert result.ejected == ['mb'], f'got ejected={result.ejected}'

        # task/mb branch must be in a clean state (rebase properly aborted).
        # git status exits 0 even mid-rebase, so check the gitdir directly.
        mb_wt = git_ops.worktree_base / 'mb'
        await _assert_no_rebase_in_progress(mb_wt)

        # tip worktree (C) must have foo.txt (from A, since C re-linked onto A)
        # and its own baz.txt
        mc_wt = git_ops.worktree_base / 'mc'
        assert (mc_wt / 'foo.txt').exists(), 'tip C must contain foo.txt from anchor A'
        assert (mc_wt / 'baz.txt').exists(), 'tip C must contain its own baz.txt'

    async def test_missing_worktree_eject_and_relink(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """Missing worktree directory is treated as an eject; re-link invariant holds.

        Setup:
          - anchor A: worktree exists, edits fileA.txt
          - member MISSING: worktree directory never created
          - member C: worktree exists, edits fileC.txt (different file, clean)

        Expected:
          - 'missing' is ejected (wt_path.is_dir() → False)
          - last_good_id stays 'xa' (not advanced past the missing member)
          - 'xc' re-links onto 'xa' and survives (clean rebase)
          - survivors == ['xa', 'xc'], ejected == ['xmissing']
          - No exception is raised
        """
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        main_ref = 'main'

        await _make_member(git_ops, 'xa', main_ref, 'fileA.txt', 'content A\n')
        # xmissing: deliberately omit worktree creation
        await _make_member(git_ops, 'xc', main_ref, 'fileC.txt', 'content C\n')

        result = await git_ops.stack_train_branches(['xa', 'xmissing', 'xc'])

        assert result.survivors == ['xa', 'xc'], (
            f'xc should re-link onto xa after xmissing is ejected; '
            f'got survivors={result.survivors}'
        )
        assert result.ejected == ['xmissing'], (
            f'xmissing should be ejected; got ejected={result.ejected}'
        )
        # xc's tip must carry fileA (from xa, since it re-linked onto xa)
        xc_wt = git_ops.worktree_base / 'xc'
        assert (xc_wt / 'fileA.txt').exists(), (
            'xc must contain fileA from anchor xa after re-linking'
        )


# ---------------------------------------------------------------------------
# Task 1715 — TestRecoverRedMain
# Tests for GitOps.recover_red_main: enforce-safe CAS recovery ref-move.
# Mirrors the advance_main mark-ordering tests (task 1678) for the recovery
# path: main_gate_mark_command fires immediately before update-ref so that
# reify's reference-transaction hook records the move as SANCTIONED.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRecoverRedMain:
    """Tests for GitOps.recover_red_main — enforce-safe CAS recovery ref-move."""

    async def _two_main_shas(self, repo: Path) -> tuple[str, str]:
        """Return (target_sha, expected_main).

        Makes one extra commit on main; the pre-commit HEAD is the 'good'
        target to restore to, the new HEAD simulates the 'bad merge' to undo.
        """
        _, old, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
        (repo / '_bad_merge.txt').write_text('simulated bad merge\n')
        await _run(['git', 'add', '_bad_merge.txt'], cwd=repo)
        await _run(['git', 'commit', '-m', 'Simulate bad merge on main'], cwd=repo)
        _, new, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
        return old.strip(), new.strip()

    async def test_recover_mark_before_update_ref(self, git_repo: Path):
        """main_gate_mark_command fires immediately before the CAS update-ref call."""
        mark_cmd = 'echo recover-mark-test'
        ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command=mark_cmd,
            ),
            git_repo,
        )
        target_sha, expected_main = await self._two_main_shas(git_repo)

        original_run = _run
        recorded: list[tuple[list[str], object]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append((list(cmd), cwd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await ops.recover_red_main(target_sha, expected_main)

        assert result == 'rewound', f'Expected rewound, got {result!r}'

        mark_indices = [
            i for i, (cmd, _) in enumerate(recorded)
            if cmd == ['sh', '-c', mark_cmd]
        ]
        update_ref_indices = [
            i for i, (cmd, _) in enumerate(recorded)
            if cmd[:2] == ['git', 'update-ref'] and 'refs/heads/main' in cmd
        ]
        assert len(mark_indices) >= 1, f'No mark call; commands: {[c for c, _ in recorded]}'
        assert len(update_ref_indices) >= 1, 'No update-ref call recorded'

        mark_idx = mark_indices[-1]
        update_ref_idx = update_ref_indices[-1]

        assert mark_idx == update_ref_idx - 1, (
            f'mark must be IMMEDIATELY before update-ref; '
            f'mark_idx={mark_idx}, update_ref_idx={update_ref_idx}, '
            f'intervening: {[c for c, _ in recorded[mark_idx + 1:update_ref_idx]]}'
        )
        assert recorded[mark_idx][1] == ops.project_root, (
            f'mark must run with cwd=project_root; got {recorded[mark_idx][1]}'
        )

        # CAS: update-ref must include expected_main as old-value
        update_ref_cmd = recorded[update_ref_idx][0]
        assert target_sha in update_ref_cmd, (
            f'target_sha not in update-ref args: {update_ref_cmd}'
        )
        assert expected_main in update_ref_cmd, (
            f'expected_main (CAS old-value) not in update-ref args: {update_ref_cmd}'
        )

    async def test_recover_no_mark_when_unset(self, git_repo: Path):
        """With default GitConfig (main_gate_mark_command=None), no sh -c call is recorded."""
        ops = GitOps(
            GitConfig(main_branch='main', branch_prefix='task/', push_after_advance=False),
            git_repo,
        )
        target_sha, expected_main = await self._two_main_shas(git_repo)

        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await ops.recover_red_main(target_sha, expected_main)

        assert result == 'rewound', f'Expected rewound, got {result!r}'
        assert not any(c[:2] == ['sh', '-c'] for c in recorded), (
            f'Unexpected sh -c call with feature off; recorded: {recorded}'
        )

    async def test_recover_unmarks_on_cas_failure(self, git_repo: Path):
        """main_gate_unmark_command fires after a failed update-ref, clearing the sentinel.

        Setup: GitOps with both mark and unmark set; patch _run to fail on
        update-ref only; assert result=='cas_failed', mark before update-ref,
        unmark after update-ref.  Mirrors advance_main's unmark-on-CAS-failure
        test (TestWorkingTreeSync.test_advance_main_unmarks_on_cas_failure).
        """
        mark_cmd = 'echo recover-mark-unmark'
        unmark_cmd = 'echo recover-unmark-cleanup'
        ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command=mark_cmd,
                main_gate_unmark_command=unmark_cmd,
            ),
            git_repo,
        )
        target_sha, expected_main = await self._two_main_shas(git_repo)

        original_run = _run
        recorded: list[tuple[list[str], object]] = []

        async def recording_run(cmd, cwd=None):
            if cmd[:2] == ['git', 'update-ref']:
                recorded.append((list(cmd), cwd))
                return (1, '', 'CAS mismatch: refs/heads/main has been updated')
            recorded.append((list(cmd), cwd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await ops.recover_red_main(target_sha, expected_main)

        assert result == 'cas_failed', f'Expected cas_failed, got {result!r}'

        commands = [cmd for cmd, _ in recorded]
        mark_indices = [i for i, c in enumerate(commands) if c == ['sh', '-c', mark_cmd]]
        unmark_indices = [i for i, c in enumerate(commands) if c == ['sh', '-c', unmark_cmd]]
        update_ref_indices = [
            i for i, c in enumerate(commands)
            if c[:2] == ['git', 'update-ref'] and 'refs/heads/main' in c
        ]

        assert len(mark_indices) >= 1, f'No mark call; commands: {commands}'
        assert len(unmark_indices) >= 1, f'No unmark call; commands: {commands}'
        assert len(update_ref_indices) >= 1, f'No update-ref call; commands: {commands}'

        mark_idx = mark_indices[-1]
        unmark_idx = unmark_indices[-1]
        update_ref_idx = update_ref_indices[-1]

        assert mark_idx < update_ref_idx, (
            f'mark (idx={mark_idx}) must precede failed update-ref (idx={update_ref_idx})'
        )
        assert unmark_idx > update_ref_idx, (
            f'unmark (idx={unmark_idx}) must come AFTER failed update-ref (idx={update_ref_idx}); '
            f'commands: {commands}'
        )

    async def test_recover_no_unmark_when_unmark_command_unset(self, git_repo: Path):
        """With main_gate_unmark_command=None, CAS failure returns 'cas_failed' without raising."""
        mark_cmd = 'echo recover-mark-only'
        ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command=mark_cmd,
                # main_gate_unmark_command intentionally unset
            ),
            git_repo,
        )
        target_sha, expected_main = await self._two_main_shas(git_repo)

        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None):
            if cmd[:2] == ['git', 'update-ref']:
                recorded.append(list(cmd))
                return (1, '', 'CAS mismatch')
            recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await ops.recover_red_main(target_sha, expected_main)

        assert result == 'cas_failed', f'Expected cas_failed, got {result!r}'
        # No sh -c after the update-ref failure (no unmark command set)
        update_ref_idx = next(
            i for i, c in enumerate(recorded)
            if c[:2] == ['git', 'update-ref'] and 'refs/heads/main' in c
        )
        post_cmds = recorded[update_ref_idx + 1:]
        assert not any(c[:2] == ['sh', '-c'] for c in post_cmds), (
            f'Unexpected sh -c after failed update-ref with unmark unset; post: {post_cmds}'
        )

    async def test_recover_syncs_working_tree_when_on_main(self, git_repo: Path):
        """After a successful move, read-tree fires when project_root is on main.

        Mirrors advance_main's post-advance sync (TestWorkingTreeSync).
        """
        ops = GitOps(
            GitConfig(main_branch='main', branch_prefix='task/', push_after_advance=False),
            git_repo,
        )
        target_sha, expected_main = await self._two_main_shas(git_repo)
        # git_repo starts on main (git init -b main in _setup_repo)

        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await ops.recover_red_main(target_sha, expected_main)

        assert result == 'rewound', f'Expected rewound, got {result!r}'
        assert any(
            c[:3] == ['git', 'read-tree', '-u'] for c in recorded
        ), f'read-tree not called when on main; commands: {recorded}'

    async def test_recover_skips_read_tree_when_not_on_main(self, git_repo: Path):
        """read-tree is NOT called when project_root is not on main."""
        ops = GitOps(
            GitConfig(main_branch='main', branch_prefix='task/', push_after_advance=False),
            git_repo,
        )
        target_sha, expected_main = await self._two_main_shas(git_repo)

        # Detach HEAD from main so symbolic-ref returns non-main
        await _run(['git', 'checkout', '--detach', expected_main], cwd=git_repo)
        # Now move main backward manually so the CAS can succeed
        # (recover_red_main's update-ref uses expected_main as old-value;
        # HEAD is detached so is_on_main will be False)

        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await ops.recover_red_main(target_sha, expected_main)

        assert result == 'rewound', f'Expected rewound, got {result!r}'
        assert not any(
            c[:3] == ['git', 'read-tree', '-u'] for c in recorded
        ), f'read-tree called when not on main; commands: {recorded}'

    async def test_recover_returns_error_for_invalid_sha(self, git_repo: Path):
        """recover_red_main returns 'error' (not 'cas_failed') when SHA pre-validation fails.

        A typo'd or non-existent target_sha must route the operator to 'fix the
        SHA', not into the 'retry' loop intended for genuine CAS races.
        No mark command should fire before we detect the bad input.
        """
        ops = GitOps(
            GitConfig(
                main_branch='main',
                branch_prefix='task/',
                push_after_advance=False,
                main_gate_mark_command='echo should-not-run',
            ),
            git_repo,
        )
        target_sha, expected_main = await self._two_main_shas(git_repo)

        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None):
            recorded.append(list(cmd))
            # Fail ALL rev-parse --verify calls (simulates invalid SHA)
            if cmd[:3] == ['git', 'rev-parse', '--verify']:
                return (128, '', f'fatal: Not a valid object name {cmd[-1]}')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await ops.recover_red_main(target_sha, expected_main)

        assert result == 'error', f'Expected error for invalid SHA; got {result!r}'
        # Mark command must NOT have been issued (fail-fast before mark)
        assert not any(c[:2] == ['sh', '-c'] for c in recorded), (
            f'Unexpected sh -c (mark) fired before pre-validation completed; recorded: {recorded}'
        )

    async def test_recover_warns_on_dirty_tree_when_on_main(
        self, git_repo: Path, caplog,
    ):
        """recover_red_main emits a WARNING when the working tree is dirty and on main.

        A dirty tree would cause read-tree to silently discard uncommitted WIP;
        the warning gives operators a last-resort advisory even if they skipped
        the runbook's 'ensure clean tree' prerequisite.  The function still
        returns 'rewound' — the warning is advisory only.
        """
        ops = GitOps(
            GitConfig(main_branch='main', branch_prefix='task/', push_after_advance=False),
            git_repo,
        )
        target_sha, expected_main = await self._two_main_shas(git_repo)
        # git_repo is on main; dirty a tracked file (committed in _two_main_shas)
        (git_repo / '_bad_merge.txt').write_text('modified after commit — uncommitted WIP\n')

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await ops.recover_red_main(target_sha, expected_main)

        assert result == 'rewound', (
            f'Expected rewound even with dirty tree (warning is advisory only); got {result!r}'
        )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'uncommitted' in r.getMessage() or 'WIP' in r.getMessage() or 'dirty' in r.getMessage()
            for r in warnings
        ), (
            f'No dirty-tree warning found in records; all warnings: '
            f'{[r.getMessage() for r in warnings]}'
        )


# ===========================================================================
# Step-13: RED — create_worktree routes through warm-lane pool (B8 + B10)
# ===========================================================================


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo."""
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


@pytest.mark.asyncio
class TestCreateWorktreeWarmLaneRouting:
    """create_worktree uses the pool when enabled; raises on exhaustion (no cold fallback)."""

    async def test_warm_path_returns_lane_not_branch_named_dir(
        self, git_repo: Path,
    ):
        """With pool enabled, create_worktree returns _lane-0 not <worktree_base>/A."""
        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        info = await git_ops.create_worktree('A')

        expected_lane = git_ops.worktree_base / '_lane-0'
        assert info.path == expected_lane, (
            f'Expected _lane-0 ({expected_lane}), got {info.path}'
        )

    async def test_warm_path_lane_registered_on_task_branch(
        self, git_repo: Path,
    ):
        """The returned lane is a registered worktree on branch task/A."""
        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False, warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        info = await git_ops.create_worktree('A')

        assert await git_ops._is_registered_worktree(info.path)
        _, branch, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=info.path,
        )
        assert branch.strip() == 'task/A'

    async def test_warm_path_landlock_confined_to_lane(
        self, git_repo: Path,
    ):
        """build_landlock_command with lane path produces --writable paths within the lane."""
        from orchestrator.agents.landlock import build_landlock_command

        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False, warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        info = await git_ops.create_worktree('A')

        # Simulate landlock wrapping with the returned path
        cmd = build_landlock_command(['cargo', 'test'], info.path, ['src'])

        # All --writable paths must be under the lane, not some other dir
        writable_paths = [
            cmd[i + 1] for i, arg in enumerate(cmd) if arg == '--writable'
        ]
        lane_resolved = str(info.path.resolve())
        for wp in writable_paths:
            assert wp.startswith(lane_resolved), (
                f'--writable {wp!r} is not under lane {lane_resolved!r}'
            )

    async def test_exhaustion_raises_no_cold_fallback(
        self, git_repo: Path,
    ):
        """Pool exhausted: create_worktree raises WarmLanePoolExhausted (no cold fallback).

        Design decision (task 1859 step-8): pool exhaustion is backpressure — the
        caller must requeue.  No cold worktree is created; the cold dir must not
        exist after the failed call.
        """
        from orchestrator.git_ops import WarmLanePoolExhausted
        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False, warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        # Exhaust the pool
        info_A = await git_ops.create_worktree('A')
        assert info_A.path == git_ops.worktree_base / '_lane-0'

        # Pool exhausted — must raise, NOT fall back to a cold worktree dir
        cold_path = git_ops.worktree_base / 'B'
        with pytest.raises(WarmLanePoolExhausted):
            await git_ops.create_worktree('B')
        assert not cold_path.exists(), (
            f'Cold dir {cold_path} must NOT be created on pool exhaustion'
        )

    async def test_knob_off_cold_path_unchanged(
        self, git_ops: GitOps,
    ):
        """With warm_lane_pool=False (default fixture), create_worktree is byte-identical to today."""
        # Default git_ops fixture has warm_lane_pool=False, no warm_lane_pool_size
        assert git_ops.warm_lane_pool is None
        info = await git_ops.create_worktree('C')
        expected = git_ops.worktree_base / 'C'
        assert info.path == expected
        assert info.path.exists()


# ===========================================================================
# Step-15: RED — cleanup_worktree is pool-aware (releases lane, not removes)
# ===========================================================================


@pytest.mark.asyncio
class TestCleanupWorktreePoolAware:
    """cleanup_worktree routes to release_warm_lane for lanes; cold path unchanged."""

    async def test_cleanup_lane_not_removed(
        self, git_repo: Path,
    ):
        """cleanup_worktree on a lane retains the registered worktree (not removed)."""
        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False, warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        info = await git_ops.create_worktree('A')

        # Cleanup the lane
        await git_ops.cleanup_worktree(info.path, 'A')

        # The lane dir must still exist as a registered worktree
        assert await git_ops._is_registered_worktree(info.path), (
            '_lane-0 must remain registered after cleanup (pool-aware)'
        )

    async def test_cleanup_lane_target_retained(
        self, git_repo: Path,
    ):
        """cleanup_worktree on a lane retains target/ warmth."""
        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False, warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        info = await git_ops.create_worktree('A')

        assert (info.path / 'target' / 'seeded.bin').exists(), 'prereq: seed ran'
        await git_ops.cleanup_worktree(info.path, 'A')

        assert (info.path / 'target').exists(), 'target/ must be retained after cleanup'

    async def test_cleanup_lane_pool_freed(
        self, git_repo: Path,
    ):
        """cleanup_worktree on a lane flips the pool state back to FREE."""
        from orchestrator.warm_lane_pool import LaneState
        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False, warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        info = await git_ops.create_worktree('A')
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.ASSIGNED

        await git_ops.cleanup_worktree(info.path, 'A')

        assert git_ops.warm_lane_pool.state(info.path) == LaneState.FREE

    async def test_cleanup_lane_branch_deleted(
        self, git_repo: Path,
    ):
        """cleanup_worktree on a lane deletes branch task/A."""
        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False, warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        info = await git_ops.create_worktree('A')

        await git_ops.cleanup_worktree(info.path, 'A')

        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/A'], cwd=git_repo,
        )
        assert rc != 0, 'task/A branch must be deleted after cleanup'

    async def test_cleanup_cold_worktree_removed(
        self, git_repo: Path,
    ):
        """cleanup_worktree on a cold (non-lane) path removes the worktree as before.

        Pool exhaustion now raises rather than falling back to cold, so we create
        the cold worktree directly via 'git worktree add' to set up the precondition.
        """
        await _add_warm_lane_scripts(git_repo)
        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False, warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        # Create a cold (non-lane) worktree directly — pool exhaustion raises,
        # so we bypass create_worktree and add the worktree via git directly.
        cold_path = git_ops.worktree_base / 'Z'
        cold_path.parent.mkdir(parents=True, exist_ok=True)
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/Z', str(cold_path), 'HEAD'],
            cwd=git_repo,
        )
        assert cold_path.exists()
        assert await git_ops._is_registered_worktree(cold_path)

        # Cleanup the cold worktree — must be removed
        await git_ops.cleanup_worktree(cold_path, 'Z')
        assert not cold_path.exists(), 'Cold worktree must be removed by cleanup'

    async def test_cleanup_knob_off_cold_path(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """With pool disabled (default fixture), cleanup_worktree removes the worktree."""
        info = await git_ops.create_worktree('D')
        assert info.path.exists()

        await git_ops.cleanup_worktree(info.path, 'D')

        assert not info.path.exists(), 'Cold worktree must be removed when pool disabled'

    async def test_cleanup_routes_spec_lane_to_release(
        self, git_repo: Path,
    ):
        """cleanup_worktree on a '_spec-' lane releases it back to the spec pool.

        Symmetric with the warm_lane_pool routing: a merge-speculation lane
        must be RELEASED (retain worktree + target/, flip FREE) rather than
        git-worktree-removed.  The crash-recovery sweep routes no-plan spec
        lanes through cleanup_worktree, so this routing is what prevents a
        spec lane from being destroyed (and its pool slot stranded) at
        recovery time.
        """
        from unittest.mock import AsyncMock, MagicMock

        config = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False,
            merge_spec_warm_lane_pool=True,
        )
        git_ops = GitOps(config, git_repo, merge_spec_warm_lane_pool_size=2)
        assert git_ops.spec_warm_lane_pool is not None
        assert git_ops.warm_lane_pool is None  # only the spec pool is active

        spec_lane = git_ops.worktree_base / '_spec-0'
        git_ops.spec_warm_lane_pool.is_lane = MagicMock(return_value=True)
        git_ops.release_spec_lane = AsyncMock()

        await git_ops.cleanup_worktree(spec_lane, '_spec-0')

        git_ops.release_spec_lane.assert_awaited_once_with(spec_lane, warm=True)


# ===========================================================================
# Step-25: RED — create_worktree requeue-of-a-warm-task end-to-end
#   + recycled-id identity guard
#
# Part (a): Requeue regression guard — same process requeue of a warm task
#   should reuse _lane-0, preserve .task/plan.json, commit WIP, retain target/.
#   This works today (steps 22-24 implemented live-requeue).
#
# Part (b): Recycled-id guard — after cleanup + re-dispatch with a DIFFERENT
#   expected_title, the disk backstop should NOT inherit the stale plan.json.
#   Today (step-24, no step-26): disk backstop detects task_id match → REUSE
#   → stale plan.json IS inherited → test FAILS → RED.
# ===========================================================================


@pytest.mark.asyncio
class TestCreateWorktreeRequeueAndRecycledId:
    """create_worktree requeue regression guard + recycled-id identity guard."""

    def _warm_config(self) -> GitConfig:
        return GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )

    async def test_requeue_same_lane_plan_preserved(self, git_repo: Path):
        """(a) Requeue: same task returns _lane-0 with .task/plan.json preserved."""
        import json as _json
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.create_worktree('A')
        lane = info1.path

        # Simulate agent work
        (lane / '.task').mkdir(exist_ok=True)
        plan_file = lane / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "A", "title": "Task A"}')
        (lane / 'README.md').write_text('WIP changes\n')

        # Requeue (no cleanup)
        info2 = await git_ops.create_worktree('A')

        assert info2.path == lane, f'Expected same _lane-0 ({lane}), got {info2.path}'
        assert plan_file.exists(), '.task/plan.json must be preserved on requeue'
        data = _json.loads(plan_file.read_text())
        assert data['task_id'] == 'A', f'plan.json was overwritten: {data}'

    async def test_requeue_wip_committed(self, git_repo: Path):
        """(a) Requeue: a WIP-save commit is created for uncommitted tracked changes."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.create_worktree('A')
        lane = info1.path
        (lane / '.task').mkdir(exist_ok=True)
        (lane / '.task' / 'plan.json').write_text('{"task_id": "A"}')
        (lane / 'README.md').write_text('WIP changes\n')

        info2 = await git_ops.create_worktree('A')
        assert info2.path == lane

        _, log_out, _ = await _run(['git', 'log', '--oneline'], cwd=lane)
        wip_commits = [
            line for line in log_out.splitlines()
            if 'save wip' in line.lower() or 'save WIP' in line
        ]
        assert wip_commits, f'No WIP-save commit found in log:\n{log_out}'

    async def test_requeue_target_retained(self, git_repo: Path):
        """(a) Requeue: target/cache.bin is retained (warmth preserved)."""
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        info1 = await git_ops.create_worktree('A')
        lane = info1.path
        (lane / '.task').mkdir(exist_ok=True)
        (lane / '.task' / 'plan.json').write_text('{"task_id": "A"}')
        (lane / 'README.md').write_text('WIP\n')
        cache_file = lane / 'target' / 'cache.bin'
        cache_file.write_bytes(b'\xca\xfe' * 32)

        info2 = await git_ops.create_worktree('A')
        assert info2.path == lane

        assert cache_file.exists(), 'target/cache.bin must be retained on requeue'

    async def test_recycled_id_stale_plan_not_inherited(self, git_repo: Path):
        """(b) Recycled-id guard: new task with different expected_title should NOT
        inherit the stale .task/plan.json from the prior deleted task.

        Today (step-24, no step-26): disk backstop sees task_id 'A' == 'A'
        → REUSE → plan.json IS inherited → this assertion FAILS → RED.
        After step-26: identity guard checks read_worktree_title vs
        expected_title → 'Old Task' != 'New Task' → MISMATCH → FRESH reset
        → plan.json cleared → this assertion PASSES.
        """
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        # First task 'A' runs on _lane-0 and writes plan.json with 'Old Task'
        info1 = await git_ops.create_worktree('A')
        lane = info1.path
        (lane / '.task').mkdir(exist_ok=True)
        plan_file = lane / '.task' / 'plan.json'
        plan_file.write_text('{"task_id": "A", "title": "Old Task"}')

        # Cleanup: lane returns to FREE; plan.json stays on disk
        await git_ops.cleanup_worktree(lane, 'A')
        assert plan_file.exists(), 'prereq: plan.json must still exist after cleanup'

        # NEW task with recycled id 'A' but DIFFERENT title — should NOT inherit plan
        info2 = await git_ops.create_worktree('A', expected_title='New Task')
        assert info2.path == lane  # same pool lane

        # TODAY: disk backstop reuses (task_id 'A' == 'A') → plan.json inherited → FAIL
        # AFTER step-26: identity mismatch ('Old Task' != 'New Task') → fresh reset → plan gone
        assert not plan_file.exists(), (
            'Recycled-id task MUST NOT inherit the prior task\'s plan.json '
            '(identity guard should route to fresh reset)'
        )


# ---------------------------------------------------------------------------
# Task 1809 step-13: get_merge_diff_files emits WARNING on rc!=0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetMergeDiffFilesWarning:
    """Task 1826: get_merge_diff_files returns (files, error) tuple contract.

    Error path: rc!=0 (non-existent SHA) → ([], Exception), WARNING emitted.
    Success path: valid base..head → (changed_files, None).
    """

    async def test_nonexistent_sha_returns_empty_tuple_and_warns(
        self, git_repo: Path, git_config: GitConfig, caplog,
    ) -> None:
        import logging

        ops = GitOps(git_config, git_repo)
        non_existent = 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef'

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            files, err = await ops.get_merge_diff_files(non_existent, 'HEAD')

        assert files == [], (
            f'Expected [] on git error; got {files!r}'
        )
        assert err is not None, (
            'Expected a non-None error on git diff rc!=0'
        )
        assert isinstance(err, Exception), (
            f'Expected err to be an Exception; got {type(err)!r}'
        )

        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, (
            'Expected a WARNING at orchestrator.git_ops when git diff fails; '
            'got no warnings'
        )
        assert any(
            'get_merge_diff_files' in t.lower() or 'diff' in t.lower()
            for t in warning_texts
        ), f'Expected WARNING to mention diff failure; got: {warning_texts}'

    async def test_success_returns_changed_files_and_no_error(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """Success path: two files committed → (paths, None)."""
        # Capture the base SHA (initial commit)
        rc, base_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        assert rc == 0
        base_sha = base_sha.strip()

        # Add two files and commit
        (git_repo / 'alpha.py').write_text('# alpha\n')
        (git_repo / 'beta.py').write_text('# beta\n')
        await _run(['git', 'add', 'alpha.py', 'beta.py'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'add alpha and beta'], cwd=git_repo)

        rc2, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        assert rc2 == 0
        head_sha = head_sha.strip()

        files, err = await git_ops.get_merge_diff_files(base_sha, head_sha)

        assert err is None, (
            f'Expected no error on successful diff; got {err!r}'
        )
        assert sorted(files) == ['alpha.py', 'beta.py'], (
            f'Expected [alpha.py, beta.py]; got {files!r}'
        )


# ---------------------------------------------------------------------------
# Task 1825 step-1: get_files_touched_in_branch emits WARNING on rc!=0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetFilesTouchedInBranchWarning:
    """Step-1 (RED): get_files_touched_in_branch must emit a WARNING when
    git log exits non-zero (e.g. non-existent base SHA), while still
    returning [].

    Before step-2 impl: rc!=0 path is silent → RED (no WARNING).
    After step-2 impl: WARNING captured at 'orchestrator.git_ops'.
    """

    async def test_nonexistent_sha_returns_empty_and_warns(
        self, git_repo: Path, git_config: GitConfig, caplog,
    ) -> None:
        import logging

        ops = GitOps(git_config, git_repo)
        non_existent = 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef'

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await ops.get_files_touched_in_branch(non_existent, 'HEAD')

        assert result == [], (
            f'Expected [] on git error; got {result!r}'
        )

        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, (
            'Expected a WARNING at orchestrator.git_ops when git log fails; '
            'got no warnings'
        )
        assert any(
            'get_files_touched_in_branch' in t.lower()
            for t in warning_texts
        ), f'Expected WARNING to mention get_files_touched_in_branch; got: {warning_texts}'


# ---------------------------------------------------------------------------
# TestMergeTreeConflicts — tests for merge_tree_conflicts() (PRD §5.2, task β)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMergeTreeConflicts:
    """Unit tests for GitOps.merge_tree_conflicts(base_tip, branch_head).

    The primitive answers "would branch_head merge cleanly onto base_tip?"
    using git merge-tree --write-tree, with ZERO worktree creation and
    no mutations to refs, index, or checkout.
    """

    async def test_clean_merge_returns_probe_with_no_conflicts(
        self, git_ops: GitOps,
    ) -> None:
        """CLEAN case: two branches touching DIFFERENT files merge cleanly.

        Checks ConflictProbe.clean, ConflictProbe.conflicted_paths,
        and that tuple-destructuring works (NamedTuple contract).
        """
        # Build branch mt-a (writes a.py) and mt-b (writes b.py) off the same main.
        wt_a = await git_ops.create_worktree('mt-a')
        (wt_a.path / 'a.py').write_text('a = 1\n')
        await git_ops.commit(wt_a.path, 'Add a.py on mt-a')

        wt_b = await git_ops.create_worktree('mt-b')
        (wt_b.path / 'b.py').write_text('b = 1\n')
        await git_ops.commit(wt_b.path, 'Add b.py on mt-b')

        probe = await git_ops.merge_tree_conflicts('task/mt-a', 'task/mt-b')

        # Named-field access
        assert probe.clean is True
        assert probe.conflicted_paths == []

        # Tuple-destructuring (NamedTuple contract — PRD's "(clean, conflicted_paths)")
        clean, paths = probe
        assert clean is True
        assert paths == []

    async def test_single_file_conflict(self, git_ops: GitOps) -> None:
        """CONFLICT case: both branches rewrite the same line of shared.py.

        Asserts probe.clean is False and probe.conflicted_paths == ['shared.py']
        (EXACT list — forces the parser to stop at the blank-line section
        boundary and exclude informational messages like "CONFLICT (content):…").
        """
        # shared.py must exist on main before branching (merge-tree needs a
        # common ancestor that already has the file).
        (git_ops.project_root / 'shared.py').write_text('value = 0\n')
        await _run(['git', 'add', 'shared.py'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Add shared.py on main'], cwd=git_ops.project_root)

        # Branch mt-c: change shared.py to "A"
        wt_c = await git_ops.create_worktree('mt-c')
        (wt_c.path / 'shared.py').write_text('value = "A"\n')
        await git_ops.commit(wt_c.path, 'Branch mt-c: value = A')

        # Branch mt-d: change shared.py to "B" (conflicts with A)
        wt_d = await git_ops.create_worktree('mt-d')
        (wt_d.path / 'shared.py').write_text('value = "B"\n')
        await git_ops.commit(wt_d.path, 'Branch mt-d: value = B')

        probe = await git_ops.merge_tree_conflicts('task/mt-c', 'task/mt-d')

        assert probe.clean is False
        assert probe.conflicted_paths == ['shared.py'], (
            f'Expected [\"shared.py\"], got {probe.conflicted_paths!r}'
        )

    async def test_multiple_file_conflicts(self, git_ops: GitOps) -> None:
        """CONFLICT case: two files both conflict.

        Asserts probe.clean is False and the set of conflicted_paths matches
        exactly {f1.txt, f2.txt}, with NO informational text leaked in
        (no 'CONFLICT' or 'Auto-merging' substrings in any path).
        """
        # Seed both files on main
        for fname, content in [('f1.txt', 'line1\n'), ('f2.txt', 'line2\n')]:
            (git_ops.project_root / fname).write_text(content)
        await _run(['git', 'add', 'f1.txt', 'f2.txt'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Add f1.txt and f2.txt on main'], cwd=git_ops.project_root)

        # Branch mt-e: change both files to "X"
        wt_e = await git_ops.create_worktree('mt-e')
        (wt_e.path / 'f1.txt').write_text('X\n')
        (wt_e.path / 'f2.txt').write_text('X\n')
        await git_ops.commit(wt_e.path, 'Branch mt-e: both X')

        # Branch mt-f: change both files to "Y" (conflicts with X on both)
        wt_f = await git_ops.create_worktree('mt-f')
        (wt_f.path / 'f1.txt').write_text('Y\n')
        (wt_f.path / 'f2.txt').write_text('Y\n')
        await git_ops.commit(wt_f.path, 'Branch mt-f: both Y')

        probe = await git_ops.merge_tree_conflicts('task/mt-e', 'task/mt-f')

        assert probe.clean is False
        assert set(probe.conflicted_paths) == {'f1.txt', 'f2.txt'}, (
            f'Expected {{f1.txt, f2.txt}}, got {probe.conflicted_paths!r}'
        )
        # No informational text should leak into the paths list
        assert all(
            'CONFLICT' not in p and 'Auto-merging' not in p
            for p in probe.conflicted_paths
        ), f'Informational text leaked into conflicted_paths: {probe.conflicted_paths!r}'

    async def test_zero_worktree_creation_and_side_effect_free(
        self, git_ops: GitOps,
    ) -> None:
        """Headline: merge_tree_conflicts MUST NOT create worktrees or mutate refs.

        Sets up a conflicting pair, captures before-state (worktree count,
        worktree_base children, and SHAs of base_tip/branch_head/HEAD),
        calls merge_tree_conflicts TWICE, then asserts:

        1. Both calls return equal ConflictProbe results (deterministic/idempotent).
        2. git worktree list count is UNCHANGED.
        3. No _merge-* or any new child directory appeared under worktree_base.
        4. base_tip, branch_head, and HEAD SHAs are all unchanged (side-effect-free).
        """
        # Seed shared.py on main so there is a common ancestor
        (git_ops.project_root / 'shared.py').write_text('base = 0\n')
        await _run(['git', 'add', 'shared.py'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Seed shared.py for side-effect test'], cwd=git_ops.project_root)

        # Build a conflicting pair
        wt_p = await git_ops.create_worktree('mt-p')
        (wt_p.path / 'shared.py').write_text('base = "P"\n')
        await git_ops.commit(wt_p.path, 'Branch mt-p: base = P')

        wt_q = await git_ops.create_worktree('mt-q')
        (wt_q.path / 'shared.py').write_text('base = "Q"\n')
        await git_ops.commit(wt_q.path, 'Branch mt-q: base = Q')

        # --- Capture before-state ---
        # 1) Count entries in git worktree list
        _, wt_list_before, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        wt_count_before = wt_list_before.count('\nworktree ')

        # 2) Children under worktree_base (may not exist yet)
        def _children(base: 'Path') -> 'set[str]':
            if not base.exists():
                return set()
            return {p.name for p in base.iterdir()}

        children_before = _children(git_ops.worktree_base)

        # 3) Resolve SHAs of the two refs and HEAD
        _, sha_p_before, _ = await _run(
            ['git', 'rev-parse', 'task/mt-p'], cwd=git_ops.project_root,
        )
        _, sha_q_before, _ = await _run(
            ['git', 'rev-parse', 'task/mt-q'], cwd=git_ops.project_root,
        )
        _, head_before, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=git_ops.project_root,
        )
        sha_p_before = sha_p_before.strip()
        sha_q_before = sha_q_before.strip()
        head_before = head_before.strip()

        # --- Call merge_tree_conflicts TWICE ---
        probe1 = await git_ops.merge_tree_conflicts('task/mt-p', 'task/mt-q')
        probe2 = await git_ops.merge_tree_conflicts('task/mt-p', 'task/mt-q')

        # --- Assert post-call state ---
        # 1) Idempotent: both calls return identical results
        assert probe1 == probe2, (
            f'merge_tree_conflicts is not idempotent: first={probe1!r}, second={probe2!r}'
        )
        assert probe1.clean is False  # they conflict

        # 2) worktree list count unchanged
        _, wt_list_after, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        wt_count_after = wt_list_after.count('\nworktree ')
        assert wt_count_after == wt_count_before, (
            f'git worktree list grew from {wt_count_before} to {wt_count_after} entries'
        )

        # 3) No _merge-* or new child under worktree_base
        children_after = _children(git_ops.worktree_base)
        new_children = children_after - children_before
        merge_children = {c for c in new_children if '_merge' in c}
        assert not merge_children, (
            f'_merge-* child(ren) appeared under worktree_base: {merge_children!r}'
        )
        assert not new_children, (
            f'New child dir(s) appeared under worktree_base: {new_children!r}'
        )

        # 4) Refs and HEAD unchanged
        _, sha_p_after, _ = await _run(
            ['git', 'rev-parse', 'task/mt-p'], cwd=git_ops.project_root,
        )
        _, sha_q_after, _ = await _run(
            ['git', 'rev-parse', 'task/mt-q'], cwd=git_ops.project_root,
        )
        _, head_after, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=git_ops.project_root,
        )
        assert sha_p_after.strip() == sha_p_before, 'task/mt-p ref was mutated'
        assert sha_q_after.strip() == sha_q_before, 'task/mt-q ref was mutated'
        assert head_after.strip() == head_before, 'HEAD was mutated'

    async def test_bad_revision_raises_runtime_error(
        self, git_ops: GitOps,
    ) -> None:
        """ERROR case: git merge-tree exits with an error rc for an unknown revision.

        merge_tree_conflicts must raise (not return a misleading ConflictProbe)
        when git exits with a non-{0,1} rc OR with rc==1 and empty stdout
        (git reports "not something we can merge" on stderr only).  A bad SHA
        is a caller bug; silently returning clean=True would admit a broken
        branch into verify, and returning clean=False would falsely bounce a
        mergeable branch.

        The error message must identify the offending rc (to distinguish the
        error path from the conflict path) and include the bogus ref (so the
        caller can pinpoint the bad input in logs).
        """
        bogus = 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef'

        with pytest.raises(RuntimeError) as exc_info:
            await git_ops.merge_tree_conflicts('main', bogus)

        err_text = str(exc_info.value)
        # Must mention the rc — distinguishes the error branch from the
        # rc==1/non-empty-stdout genuine-conflict branch.
        assert 'rc=' in err_text, f'Expected rc= in error message; got: {err_text!r}'
        # Must include the offending ref — makes the caller bug locatable in logs.
        assert bogus in err_text, (
            f'Expected bogus ref {bogus!r} in error message; got: {err_text!r}'
        )


# ===========================================================================
# Task-1912 step-1: RED — release_warm_lane branch-retention guard
# ===========================================================================


@pytest.mark.asyncio
class TestReleaseWarmLaneBranchRetention:
    """release_warm_lane retains task/<id> when it carries commits beyond main;
    deletes it (on-main only) when the branch is at the main tip."""

    def _warm_config(self) -> GitConfig:
        return GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )

    async def test_retains_branch_when_unmerged(self, git_repo: Path):
        """release_warm_lane RETAINS task/A when the branch has commits beyond main.

        RED today: the unguarded `git branch -D task/A` destroys the branch
        before the pool.release(); this assertion will flip to PASS after
        step-2 introduces the _branch_has_commits_beyond_main guard.
        """
        from orchestrator.warm_lane_pool import LaneState

        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        info = await git_ops.create_worktree('A')

        # Commit one file on the lane so task/A has 1 committed commit beyond main.
        (info.path / 'wip.txt').write_text('work in progress\n')
        await _run(['git', 'add', '-A'], cwd=info.path)
        await _run(['git', 'commit', '-m', 'wip'], cwd=info.path)

        await git_ops.release_warm_lane(info.path, 'A')

        # Branch must be RETAINED — it carries committed commits beyond main.
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/A'], cwd=git_repo,
        )
        assert rc == 0, (
            'task/A must be RETAINED after release_warm_lane when it carries '
            'commits beyond main'
        )

        # Cache lifecycle is independent — lane must be FREE regardless.
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.FREE, (
            'lane must be FREE after release even when branch is retained'
        )

    async def test_deletes_branch_when_merged(self, git_repo: Path):
        """release_warm_lane DELETES task/A when the branch is at the main tip
        (0 commits beyond main — i.e. a fresh lane that was never used).

        This assertion passes today and is the regression guard that prevents
        an always-retain implementation from leaking branch refs forever.
        It also confirms the fresh-lane base == main tip premise.
        """
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        # Fresh lane: task/A == main tip, 0 commits beyond main.
        info = await git_ops.create_worktree('A')

        await git_ops.release_warm_lane(info.path, 'A')

        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/A'], cwd=git_repo,
        )
        assert rc != 0, (
            'task/A must be DELETED after release_warm_lane when it is at '
            'the main tip (0 commits beyond main)'
        )

    async def test_retains_branch_on_rev_list_failure(self, git_repo: Path):
        """release_warm_lane RETAINS task/A when the rev-list probe fails.

        Validates the fail-safe: when _branch_has_commits_beyond_main returns
        True due to a git error (rc != 0) or unparseable output, the branch
        must be RETAINED rather than deleted.  This is the safety-critical
        path that protects WIP from destruction on uncertainty.
        """
        from orchestrator.warm_lane_pool import LaneState

        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        # Fresh lane (no commits beyond main) — normally this branch would be
        # deleted.  With the fail-safe patched to True it must be retained.
        info = await git_ops.create_worktree('A')

        async def _always_has_commits(*args, **kwargs) -> bool:
            return True  # simulates git error / unparseable output → fail-safe True

        with patch.object(
            git_ops, '_branch_has_commits_beyond_main', side_effect=_always_has_commits,
        ):
            await git_ops.release_warm_lane(info.path, 'A')

        # Branch must be RETAINED — fail-safe returned True.
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/A'], cwd=git_repo,
        )
        assert rc == 0, (
            'task/A must be RETAINED when _branch_has_commits_beyond_main '
            'returns True (fail-safe on git error)'
        )

        # Cache lifecycle is independent — lane must be FREE regardless.
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(info.path) == LaneState.FREE, (
            'lane must be FREE after release even when branch is retained via fail-safe'
        )


# ===========================================================================
# Task-1912 step-3: RED — cold cleanup_worktree branch-retention guard
# ===========================================================================


@pytest.mark.asyncio
class TestCleanupWorktreeColdBranchRetention:
    """cold cleanup_worktree retains task/<id> when it carries commits beyond main;
    deletes it (on-main only) when the branch is at the main tip."""

    async def test_retains_branch_when_unmerged(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """cleanup_worktree RETAINS task/Z when the branch has commits beyond main.

        Uses the default git_ops fixture (pool disabled) so create_worktree
        yields a cold (non-lane) worktree — mirrors test_cleanup_knob_off_cold_path.

        RED today: the unguarded `git branch -D task/Z` destroys the branch;
        this assertion will flip to PASS after step-4 adds the guard.
        """
        info = await git_ops.create_worktree('Z')

        # Commit one file on the worktree so task/Z has 1 commit beyond main.
        (info.path / 'wip.txt').write_text('cold path wip\n')
        await _run(['git', 'add', '-A'], cwd=info.path)
        await _run(['git', 'commit', '-m', 'wip'], cwd=info.path)

        await git_ops.cleanup_worktree(info.path, 'Z')

        # Worktree directory must be REMOVED (cold path unchanged).
        assert not info.path.exists(), (
            'Cold worktree path must be removed by cleanup_worktree'
        )

        # Branch must be RETAINED — it carries committed commits beyond main.
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/Z'], cwd=git_repo,
        )
        assert rc == 0, (
            'task/Z must be RETAINED after cold cleanup_worktree when it '
            'carries commits beyond main'
        )

    async def test_deletes_branch_when_merged(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """cleanup_worktree DELETES task/Z when the branch is at the main tip
        (0 commits beyond main — fresh worktree, never committed to).

        This assertion passes today and is the regression guard that prevents
        an always-retain implementation from leaking branch refs forever.
        """
        info = await git_ops.create_worktree('Z')

        # Fresh worktree: task/Z == main tip, 0 commits beyond main.
        await git_ops.cleanup_worktree(info.path, 'Z')

        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/Z'], cwd=git_repo,
        )
        assert rc != 0, (
            'task/Z must be DELETED after cold cleanup_worktree when it is '
            'at the main tip (0 commits beyond main)'
        )

    async def test_retains_branch_on_rev_list_failure(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """cleanup_worktree RETAINS task/Z when the rev-list probe fails.

        Validates the fail-safe: when _branch_has_commits_beyond_main returns
        True due to a git error (rc != 0) or unparseable output, the branch
        must be RETAINED rather than deleted.  The worktree is still removed
        (cold path's worktree removal is orthogonal to the branch guard).
        """
        # Fresh worktree (no commits beyond main) — normally this branch
        # would be deleted.  With the fail-safe patched to True it must be
        # retained.
        info = await git_ops.create_worktree('Z')

        async def _always_has_commits(*args, **kwargs) -> bool:
            return True  # simulates git error / unparseable output → fail-safe True

        with patch.object(
            git_ops, '_branch_has_commits_beyond_main', side_effect=_always_has_commits,
        ):
            await git_ops.cleanup_worktree(info.path, 'Z')

        # Worktree directory must be REMOVED (cold path unchanged).
        assert not info.path.exists(), (
            'Cold worktree path must be removed by cleanup_worktree even '
            'when branch is retained via fail-safe'
        )

        # Branch must be RETAINED — fail-safe returned True.
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/Z'], cwd=git_repo,
        )
        assert rc == 0, (
            'task/Z must be RETAINED when _branch_has_commits_beyond_main '
            'returns True (fail-safe on git error)'
        )


# ===========================================================================
# Task-1914 step-5: RED — create-once leftover protective guard + fail-safe
# ===========================================================================


@pytest.mark.asyncio
class TestAcquireWarmLaneCreateOnceLeftoverGuard:
    """create-once site NEVER destroys a leftover task/<id> branch that carries
    commits beyond main.  If the leftover cannot be re-attached (e.g. already
    checked out in another worktree), acquire returns WarmLaneUnavailable.FAULT
    while leaving the branch intact.  Fail-safe: _branch_has_commits_beyond_main
    True (including on git error) → reattach/retain direction taken on both paths."""

    def _warm_config(self) -> GitConfig:
        return GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )

    async def test_create_once_leftover_with_commits_not_destroyed_when_unattachable(
        self, git_repo: Path, tmp_path: Path,
    ):
        """Create-once leftover carrying commits is NEVER destroyed when it cannot
        be re-attached (already checked out in another worktree).

        Setup: create task/A with commits via a temp worktree and LEAVE the
        worktree in place (not removed).  When acquire_warm_lane tries
        `git worktree add <lane> task/A`, git refuses with 'branch already
        checked out' — and the branch must NOT be deleted as a fallback.

        After step-4: the reattach guard fires and `git worktree add` (no -b)
        fails → FAULT returned; task/A is never touched → both assertions PASS.
        After step-6: same external result; the raise-not-destroy contract is
        the same observable effect.
        """
        from orchestrator.git_ops import WarmLaneUnavailable

        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        # Create task/A with 2 commits AND leave the worktree checked out.
        # git_repo is at tmp_path/repo; tmp_path is the shared temp root.
        tmp_wt_a = tmp_path / 'kept_wt_A'
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/A', str(tmp_wt_a), start_ref],
            cwd=git_repo,
        )
        for i in range(2):
            (tmp_wt_a / f'wip_{i}.txt').write_text(f'work item {i}\n')
            await _run(['git', 'add', '-A'], cwd=tmp_wt_a)
            await _run(['git', 'commit', '-m', f'wip {i}'], cwd=tmp_wt_a)

        # Capture count BEFORE acquire (for regression comparison)
        _, count_before_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'], cwd=git_repo,
        )
        count_before = int(count_before_raw.strip())
        assert count_before >= 2, f'Expected >=2 commits for task/A, got {count_before}'

        # FRESH pool — _lane-0 never acquired
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        result = await git_ops.acquire_warm_lane('A', start_ref)

        # Must return WarmLaneUnavailable (FAULT), NOT WorktreeInfo
        assert isinstance(result, WarmLaneUnavailable), (
            f'Expected WarmLaneUnavailable for unattachable orphan, got {result!r}'
        )

        # task/A branch must be RETAINED — never destroyed regardless of failure
        rc_verify, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/A'], cwd=git_repo,
        )
        assert rc_verify == 0, (
            'task/A branch must be RETAINED after unattachable reattach attempt '
            '— the never-destroy contract must hold even on FAULT'
        )

        # Commit count must be UNCHANGED (no commits lost)
        _, count_after_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'], cwd=git_repo,
        )
        count_after = int(count_after_raw.strip())
        assert count_after == count_before, (
            f'task/A commit count must be unchanged: expected {count_before}, '
            f'got {count_after} — the branch was modified when it should not have been'
        )

    async def test_reattach_fail_safe_when_commits_probe_true(
        self, git_repo: Path, tmp_path: Path,
    ):
        """Reattach guard takes the retain/reattach direction when
        _branch_has_commits_beyond_main returns True (including on git error).

        Mirrors α's test_retains_branch_on_rev_list_failure: patches the probe
        to always-True (simulating a git error or unparseable output).  The
        guard must fire on the reset-in-place path and reattach rather than
        resetting to start_ref (which would destroy the commits).

        Validates the fail-safe direction: uncertainty → retain, not reset.
        """
        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        # Create orphan task/A with 2 commits via a temp worktree, then remove it.
        tmp_wt_a = tmp_path / 'orphan_wt_A'
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/A', str(tmp_wt_a), start_ref],
            cwd=git_repo,
        )
        for i in range(2):
            (tmp_wt_a / f'wip_{i}.txt').write_text(f'work item {i}\n')
            await _run(['git', 'add', '-A'], cwd=tmp_wt_a)
            await _run(['git', 'commit', '-m', f'wip {i}'], cwd=tmp_wt_a)
        _, count_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'], cwd=git_repo,
        )
        count_before = int(count_raw.strip())
        # Remove the temp worktree (task/A becomes orphan: exists but not checked out)
        await _run(['git', 'worktree', 'remove', '--force', str(tmp_wt_a)], cwd=git_repo)

        # Register+free _lane-0 to hit the reset-in-place path on next acquire.
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None
        info_seed = await git_ops.acquire_warm_lane('seed', start_ref)
        assert isinstance(info_seed, WorktreeInfo), (
            f'Seed acquire failed: {info_seed!r}'
        )
        await git_ops.warm_lane_pool.release(info_seed.path)

        # Patch _branch_has_commits_beyond_main to always-True (simulates git error)
        async def _always_has_commits(*args, **kwargs) -> bool:
            return True

        with patch.object(
            git_ops, '_branch_has_commits_beyond_main', side_effect=_always_has_commits,
        ):
            result = await git_ops.acquire_warm_lane('A', start_ref)

        # Reattach/retain direction: result must be WorktreeInfo (not FAULT)
        assert isinstance(result, WorktreeInfo), (
            f'Expected WorktreeInfo (reattach) when fail-safe is True, got {result!r}'
        )

        # task/A must be RETAINED and commit count PRESERVED (not reset to 0)
        rc_verify, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/A'], cwd=git_repo,
        )
        assert rc_verify == 0, (
            'task/A branch must be RETAINED when fail-safe fires (probe → True)'
        )
        _, count_after_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'], cwd=git_repo,
        )
        count_after = int(count_after_raw.strip())
        assert count_after >= count_before, (
            f'task/A commit count must be preserved (>=original): '
            f'expected >={count_before}, got {count_after}'
        )

    async def test_create_once_reattach_seed_failure_returns_fault_and_retains_branch(
        self, git_repo: Path, tmp_path: Path,
    ):
        """Seed failure on the create-once reattach path (worktree add succeeds
        but _seed_warm_lane returns non-zero) must:
          - Return WarmLaneUnavailable (FAULT)
          - Release the lane back to FREE
          - Leave task/<id> branch + commits intact (never-destroy contract)

        Exercises git_ops.py lines 1828-1849 — the worktree-remove + pool-
        release block that fires when _seed_warm_lane returns a non-zero rc
        during the create-once reattach path (after a successful worktree add).
        This path had no dedicated test; a bug here (e.g. forgetting to release
        the lane) would leak a lane or strand a registered worktree silently.
        """
        from orchestrator.git_ops import WarmLaneUnavailable
        from orchestrator.warm_lane_pool import LaneState

        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        # Create orphan task/A with 2 commits, then remove the temp worktree so
        # task/A is a dangling branch (not checked out anywhere — worktree add
        # during reattach will succeed, letting us exercise the seed-failure branch).
        tmp_wt_a = tmp_path / 'orphan_seed_fail_wt'
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/A', str(tmp_wt_a), start_ref],
            cwd=git_repo,
        )
        for i in range(2):
            (tmp_wt_a / f'seed_fail_{i}.txt').write_text(f'work {i}\n')
            await _run(['git', 'add', '-A'], cwd=tmp_wt_a)
            await _run(['git', 'commit', '-m', f'seed-fail wip {i}'], cwd=tmp_wt_a)
        _, count_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'], cwd=git_repo,
        )
        count_before = int(count_raw.strip())
        assert count_before >= 2, f'Expected >=2 commits for task/A, got {count_before}'
        await _run(['git', 'worktree', 'remove', '--force', str(tmp_wt_a)], cwd=git_repo)

        # FRESH pool — _lane-0 never acquired (create-once path).
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None

        # Capture lane path before the call so we can verify its state afterwards.
        lane_path = next(iter(git_ops.warm_lane_pool._lanes))

        # Patch _seed_warm_lane to return 1 (generic seed failure).
        async def _failing_seed(*args, **kwargs) -> int:
            return 1

        with patch.object(git_ops, '_seed_warm_lane', side_effect=_failing_seed):
            result = await git_ops.acquire_warm_lane('A', start_ref)

        # Must return WarmLaneUnavailable (FAULT), not WorktreeInfo
        assert isinstance(result, WarmLaneUnavailable), (
            f'Expected WarmLaneUnavailable for seed failure on create-once reattach '
            f'path, got {result!r}'
        )

        # Lane must be released back to FREE (pool.release was called)
        lane_state = git_ops.warm_lane_pool.state(lane_path)
        assert lane_state == LaneState.FREE, (
            f'Lane must be released to FREE after seed failure on reattach path, '
            f'got {lane_state!r} — indicates pool.release was not called'
        )

        # task/A branch + commits must be RETAINED (never-destroy contract)
        rc_verify, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/A'], cwd=git_repo,
        )
        assert rc_verify == 0, (
            'task/A branch must be RETAINED after seed failure on reattach path '
            '— the branch must not be deleted even when seeding fails'
        )
        _, count_after_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'], cwd=git_repo,
        )
        count_after = int(count_after_raw.strip())
        assert count_after == count_before, (
            f'task/A commit count must be unchanged after seed failure: '
            f'expected {count_before}, got {count_after}'
        )


# ===========================================================================
# Task-1914 step-7: RED — reset-in-place checkout failure must NOT silently
# proceed on the WRONG branch
# ===========================================================================


@pytest.mark.asyncio
class TestAcquireWarmLaneResetInPlaceCheckoutGuard:
    """reset-in-place reattach site must capture the rc of `git checkout -f`
    and raise (→ WarmLaneUnavailable.FAULT) when checkout fails, rather than
    silently calling _reuse_warm_lane against the wrong (previous-occupant)
    branch.

    Mirrors the create-once site (~1805-1826) which already checks the
    worktree-add rc and raises.  Makes both reattach sites symmetric."""

    def _warm_config(self) -> GitConfig:
        return GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )

    async def test_reset_in_place_checkout_failure_does_not_proceed_on_wrong_branch(
        self, git_repo: Path, tmp_path: Path,
    ):
        """When git checkout -f task/<id> fails in the reset-in-place reattach
        path (e.g. already checked out in another live worktree after a process
        restart), acquire must return WarmLaneUnavailable.FAULT instead of
        silently calling _reuse_warm_lane on the wrong branch.

        Setup:
        (a) Create task/A with 2 commits via tmp_wt_a; LEAVE it checked out.
        (b) Register+free _lane-0 as 'seed' so the lane is on task/seed and
            the next acquire for 'A' hits the registered (else) path.
        (c) Acquire for 'A' — reattach guard fires (task/A exists + commits)
            but git checkout -f task/A inside the lane FAILS because task/A
            is already checked out in tmp_wt_a.

        RED today: rc of checkout is discarded and _reuse_warm_lane(lane,
        task/A) runs against the lane still on task/seed, returning WorktreeInfo
        instead of WarmLaneUnavailable.

        GREEN after step-8: rc is captured; non-zero → RuntimeError →
        top-level except → WarmLaneUnavailable.FAULT, task/A intact.
        """
        from orchestrator.git_ops import WarmLaneUnavailable

        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        # (a) Create task/A with 2 commits and LEAVE tmp_wt_a checked out.
        tmp_wt_a = tmp_path / 'kept_wt_A_rip'
        await _run(
            ['git', 'worktree', 'add', '-b', 'task/A', str(tmp_wt_a), start_ref],
            cwd=git_repo,
        )
        for i in range(2):
            (tmp_wt_a / f'wip_{i}.txt').write_text(f'work item {i}\n')
            await _run(['git', 'add', '-A'], cwd=tmp_wt_a)
            await _run(['git', 'commit', '-m', f'wip {i}'], cwd=tmp_wt_a)

        # (c) Capture count before acquire (regression guard)
        _, count_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'], cwd=git_repo,
        )
        count_before = int(count_raw.strip())
        assert count_before >= 2, f'Expected >=2 commits for task/A, got {count_before}'

        # (b) Register+free _lane-0 as 'seed' so next acquire for 'A' hits
        #     the registered else-branch (reset-in-place path).
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)
        assert git_ops.warm_lane_pool is not None
        info_seed = await git_ops.acquire_warm_lane('seed', start_ref)
        assert isinstance(info_seed, WorktreeInfo), (
            f'Seed acquire failed (setup): {info_seed!r}'
        )
        lane_path = info_seed.path
        await git_ops.warm_lane_pool.release(lane_path)

        # Confirm the lane is currently on task/seed (previous occupant).
        _, seed_head_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane_path,
        )
        assert seed_head_raw.strip() == 'task/seed', (
            f'Lane should be on task/seed before reattach attempt, got {seed_head_raw.strip()!r}'
        )

        # (d) Acquire for 'A' — hits reset-in-place site → reattach guard fires
        #     → git checkout -f task/A FAILS (task/A checked out in tmp_wt_a)
        result = await git_ops.acquire_warm_lane('A', start_ref)

        # PRIMARY discriminator (RED today → GREEN after step-8):
        # must return WarmLaneUnavailable.FAULT, NOT WorktreeInfo
        assert isinstance(result, WarmLaneUnavailable), (
            f'Expected WarmLaneUnavailable.FAULT when checkout -f task/A fails '
            f'(branch already checked out in another worktree), got {result!r}. '
            f'The reset-in-place site must capture the checkout rc and raise, '
            f'not silently call _reuse_warm_lane on the wrong branch.'
        )

        # task/A must be RETAINED — never destroyed on the FAULT path
        rc_verify, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/A'], cwd=git_repo,
        )
        assert rc_verify == 0, (
            'task/A branch must be RETAINED after checkout failure '
            '— never-destroy contract must hold on FAULT path'
        )
        _, count_after_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/A'], cwd=git_repo,
        )
        count_after = int(count_after_raw.strip())
        assert count_after == count_before, (
            f'task/A commit count must be unchanged: expected {count_before}, '
            f'got {count_after} — branch was modified when it should not have been'
        )

        # Clarifying assert: the previous-occupant branch task/seed must have
        # gained no spurious WIP (lane was NOT silently committed onto it).
        _, seed_count_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/seed'], cwd=git_repo,
        )
        seed_count = int(seed_count_raw.strip())
        assert seed_count == 0, (
            f'task/seed must have 0 commits beyond main (lane was not silently '
            f'used), got {seed_count} — indicates _reuse_warm_lane ran on wrong branch'
        )


# ===========================================================================
# Task-1923 step-1: RED — rebind_branch_to_head contract
# ===========================================================================


@pytest.mark.asyncio
class TestRebindBranchToHead:
    """GitOps.rebind_branch_to_head(worktree, full_branch) contract.

    RED today: AttributeError — method does not exist.
    GREEN after step-2 implements `git checkout -B <full_branch>`.
    """

    def _warm_config(self) -> GitConfig:
        return GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )

    async def test_rebind_reattaches_detached_lane_and_updates_ref(
        self, git_repo: Path,
    ):
        """(a) On a DETACHED lane: rebind_branch_to_head returns True, rebinds
        refs/heads/task/X to the lane HEAD, and attaches the lane onto task/X.
        """
        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        # Acquire a lane on task/X
        info = await git_ops.acquire_warm_lane('X', start_ref)
        assert isinstance(info, WorktreeInfo), f'Acquire failed: {info!r}'
        lane = info.path

        # Commit a WIP file onto the lane so task/X has commits beyond main
        (lane / 'wip.txt').write_text('work in progress\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'wip commit'], cwd=lane)

        # Detach the lane (mirror what release_warm_lane does)
        rc, _, err = await _run(['git', 'checkout', '--detach'], cwd=lane)
        assert rc == 0, f'checkout --detach failed: {err}'

        # Confirm detached
        _, abbrev_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert abbrev_raw.strip() == 'HEAD', (
            f'Lane should be DETACHED, got {abbrev_raw.strip()!r}'
        )

        # Get the current HEAD SHA on the detached lane
        _, head_sha_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
        head_sha = head_sha_raw.strip()

        # === THE CALL UNDER TEST ===
        result = await git_ops.rebind_branch_to_head(lane, 'task/X')

        assert result is True, (
            f'rebind_branch_to_head should return True on success, got {result!r}'
        )

        # refs/heads/task/X must now point at the lane's HEAD
        _, ref_sha_raw, _ = await _run(
            ['git', 'rev-parse', 'refs/heads/task/X'], cwd=git_repo,
        )
        assert ref_sha_raw.strip() == head_sha, (
            f'refs/heads/task/X must equal lane HEAD ({head_sha[:8]}), '
            f'got {ref_sha_raw.strip()[:8]}'
        )

        # Lane must be attached to task/X (not detached)
        _, attached_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert attached_raw.strip() == 'task/X', (
            f'Lane must be ON task/X after rebind, got {attached_raw.strip()!r}'
        )

    async def test_rebind_idempotent_when_already_on_branch(
        self, git_repo: Path,
    ):
        """(b) When the lane is already on task/X, a second call is idempotent:
        still returns True, ref still == HEAD, no error.
        """
        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        info = await git_ops.acquire_warm_lane('X', start_ref)
        assert isinstance(info, WorktreeInfo)
        lane = info.path

        # Commit WIP so the branch has content
        (lane / 'wip.txt').write_text('work\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'wip'], cwd=lane)

        # Confirm lane is already on task/X
        _, abbrev_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert abbrev_raw.strip() == 'task/X', (
            f'Expected lane on task/X before idempotent test, got {abbrev_raw.strip()!r}'
        )

        _, head_sha_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
        head_sha = head_sha_raw.strip()

        # First call
        result1 = await git_ops.rebind_branch_to_head(lane, 'task/X')
        assert result1 is True, f'First call should return True, got {result1!r}'

        # Second call (idempotent)
        result2 = await git_ops.rebind_branch_to_head(lane, 'task/X')
        assert result2 is True, f'Second call should return True, got {result2!r}'

        # ref must still equal HEAD
        _, ref_sha_raw, _ = await _run(
            ['git', 'rev-parse', 'refs/heads/task/X'], cwd=git_repo,
        )
        assert ref_sha_raw.strip() == head_sha, (
            f'refs/heads/task/X must equal HEAD after idempotent rebind, '
            f'got {ref_sha_raw.strip()[:8]} vs {head_sha[:8]}'
        )

    async def test_rebind_returns_false_and_never_raises_on_git_error(
        self, git_repo: Path,
    ):
        """(c) best-effort / never-raise: when git checkout -B fails (e.g. invalid
        branch name), rebind_branch_to_head returns False and does NOT raise.

        NOTE on "branch checked out in another worktree": git checkout -B bypasses
        git's linked-worktree single-checkout guard (unlike bare `git checkout`).
        The "fail-safe" for that collision scenario is thus NOT enforced by git
        itself; it is an edge case that implies duplicate dispatch.  This test
        verifies the observable never-raise contract via an invalid branch name
        (which IS reliably rejected by git regardless of worktree state).
        Escalation esc-1923-18 records the false premise in the original design.
        """
        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        info = await git_ops.acquire_warm_lane('X', start_ref)
        assert isinstance(info, WorktreeInfo)
        lane = info.path

        # Commit WIP so the lane has content
        (lane / 'wip.txt').write_text('work\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'wip'], cwd=lane)

        # Detach the lane (mirrors release_warm_lane)
        await _run(['git', 'checkout', '--detach'], cwd=lane)

        # === THE CALL UNDER TEST ===
        # Use an invalid branch name — git rejects these reliably
        # (double-dots are not allowed in branch names per git-check-ref-format)
        result = await git_ops.rebind_branch_to_head(lane, 'task/in..valid..name')

        # Must return False (best-effort contract) — never raise
        assert result is False, (
            f'rebind_branch_to_head must return False when git checkout -B fails '
            f'(invalid branch name), got {result!r}'
        )

        # Lane HEAD must be intact — the rebased WIP is not destroyed by a failed rebind
        rc_head, _, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
        assert rc_head == 0, 'Lane HEAD must be resolvable after a failed rebind'

    async def test_rebind_succeeds_even_when_branch_checked_out_in_other_worktree(
        self, git_repo: Path,
    ):
        """(d) Documents the actual collision behavior on git 2.43.0.

        git checkout -B bypasses the linked-worktree single-checkout guard.
        When *full_branch* is concurrently checked out in a second live
        worktree, the call still returns True (rc=0) and force-resets the
        branch ref — leaving BOTH worktrees tracking the same branch.
        This is a duplicate-dispatch hazard, NOT the fail-safe the original
        docstring claimed (esc-1923-18).  The test pins this actual behavior
        so any future git version change that restores a non-zero rc for the
        collision case is immediately surfaced.
        """
        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        # Acquire a lane on task/C and commit WIP
        info = await git_ops.acquire_warm_lane('C', start_ref)
        assert isinstance(info, WorktreeInfo), f'Acquire failed: {info!r}'
        lane = info.path

        (lane / 'col_wip.txt').write_text('collision test\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'collision wip'], cwd=lane)

        _, lane_head_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
        lane_head = lane_head_raw.strip()

        # Create a SECOND worktree also on task/C (requires --force since task/C
        # is already checked out in the first lane)
        second_wt = git_repo / 'col_second_wt'
        rc_add, _, err_add = await _run(
            ['git', 'worktree', 'add', '--force', str(second_wt), 'task/C'],
            cwd=git_repo,
        )
        assert rc_add == 0, f'git worktree add --force failed: {err_add}'

        # Confirm second worktree is on task/C
        _, second_branch_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=second_wt,
        )
        assert second_branch_raw.strip() == 'task/C', (
            f'Second worktree should be on task/C, got {second_branch_raw.strip()!r}'
        )

        # Detach the FIRST lane (mirrors release_warm_lane)
        await _run(['git', 'checkout', '--detach'], cwd=lane)

        # === THE CALL UNDER TEST ===
        # task/C is checked out in second_wt — checkout -B does NOT refuse it
        result = await git_ops.rebind_branch_to_head(lane, 'task/C')

        # On git 2.43.0: rc=0 (bypasses single-checkout guard)
        assert result is True, (
            f'rebind_branch_to_head returned {result!r} when task/C is checked out '
            f'in a second worktree.  Expected True — checkout -B bypasses the '
            f'single-checkout guard and does NOT fail safe for this collision.  '
            f'If git changed this behavior, update the docstring + this assertion.'
        )

        # The ref was force-moved to the first lane's current HEAD
        _, ref_sha_raw, _ = await _run(
            ['git', 'rev-parse', 'refs/heads/task/C'], cwd=git_repo,
        )
        assert ref_sha_raw.strip() == lane_head, (
            f'refs/heads/task/C must be moved to first-lane HEAD ({lane_head[:8]}), '
            f'got {ref_sha_raw.strip()[:8]}'
        )

        # The first lane is now attached to task/C
        _, attached_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert attached_raw.strip() == 'task/C', (
            f'First lane must be ON task/C after rebind, got {attached_raw.strip()!r}'
        )

        # Clean up the second worktree
        await _run(['git', 'worktree', 'remove', '--force', str(second_wt)], cwd=git_repo)


# ===========================================================================
# Task-1923 step-5: RED — disk-backstop route end-to-end via acquire_warm_lane
# ===========================================================================


@pytest.mark.asyncio
class TestAcquireDiskBackstopReuseDetachedRebind:
    """End-to-end test of the PROVEN residual route: disk-backstop reuse (route 2)
    on an already-detached lane.

    RED today (before step-4 fix): after release_warm_lane detaches the lane,
    re-acquire via disk-backstop calls _reuse_warm_lane on a DETACHED HEAD,
    the ref stays stale (0-beyond-main), and the second release_warm_lane
    DELETES task/D (α's retention guard sees 0 commits and deletes it).

    GREEN after step-4 inserted rebind_branch_to_head in _reuse_warm_lane.
    """

    def _warm_config(self) -> GitConfig:
        return GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )

    async def test_disk_backstop_reuse_on_detached_lane_retains_on_second_release(
        self, git_repo: Path,
    ):
        """After release(detach) → re-acquire-via-disk-backstop → second release:
        task/D must be RETAINED by α (not deleted as 0-beyond-main).

        Sequence:
        1. Acquire lane on 'D', write .task/plan.json, commit WIP commit
        2. Advance main
        3. release_warm_lane('D') → detach + retain task/D (WIP > 0) + FREE
        4. acquire_warm_lane('D', main_HEAD) → disk-backstop route (route 2)
        5. Assert: WorktreeInfo; refs/heads/task/D == lane HEAD; lane ON task/D
        6. release_warm_lane('D') again → MUST RETAIN task/D (α protects rebased work)

        Acceptance criteria 1+2 from the plan.
        """
        import json as _json

        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        # Step 1: Acquire lane on 'D', write plan.json, commit WIP
        info = await git_ops.acquire_warm_lane('D', start_ref)
        assert isinstance(info, WorktreeInfo), f'Acquire failed: {info!r}'
        lane = info.path

        # Write .task/plan.json with task_id='D' (forces disk-backstop route on re-acquire)
        task_dir = lane / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text(_json.dumps({'task_id': 'D'}))

        # Commit WIP so task/D has commits beyond main
        (lane / 'wip_d.txt').write_text('work for task D\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'wip for D'], cwd=lane)

        # Step 2: Advance main with an unrelated commit
        (git_repo / 'main_advance.txt').write_text('main advance\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'advance main'], cwd=git_repo)

        _, main_head_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        main_head = main_head_raw.strip()

        # Step 3: release_warm_lane → detach + retain + FREE
        await git_ops.release_warm_lane(lane, 'D')

        # Verify: detached, task/D retained, plan.json intact
        _, abbrev_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert abbrev_raw.strip() == 'HEAD', (
            f'Lane should be DETACHED after release, got {abbrev_raw.strip()!r}'
        )
        rc_verify, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/D'], cwd=git_repo,
        )
        assert rc_verify == 0, 'task/D must be RETAINED after first release (α)'
        assert (task_dir / 'plan.json').exists(), (
            'plan.json must survive release_warm_lane (used by disk-backstop re-acquire)'
        )

        # Step 4: re-acquire via disk-backstop route
        # (pool sees lane as FREE+registered with plan.json task_id=='D' → route 2)
        info2 = await git_ops.acquire_warm_lane('D', main_head)

        # Step 5: assertions on re-acquire result
        assert isinstance(info2, WorktreeInfo), (
            f'Re-acquire must return WorktreeInfo via disk-backstop route, got {info2!r}'
        )
        assert info2.path == lane, (
            f'Re-acquire must reuse same lane ({lane}), got {info2.path}'
        )

        # refs/heads/task/D must equal lane HEAD (the rebind closed the drift)
        rc_verify2, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/D'], cwd=git_repo,
        )
        assert rc_verify2 == 0, (
            'task/D must resolve after disk-backstop re-acquire '
            '(RED today: rebind missing, ref stays stale and α deletes it)'
        )

        _, lane_head_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
        lane_head = lane_head_raw.strip()

        _, ref_raw, _ = await _run(['git', 'rev-parse', 'task/D'], cwd=git_repo)
        assert ref_raw.strip() == lane_head, (
            f'task/D ({ref_raw.strip()[:8]}) must equal lane HEAD ({lane_head[:8]}) '
            f'after disk-backstop re-acquire '
            f'(RED today: ref drifts from rebased detached HEAD)'
        )

        # Lane must be ON task/D (attached)
        _, abbrev2_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert abbrev2_raw.strip() == 'task/D', (
            f'Lane must be ON task/D after re-acquire, got {abbrev2_raw.strip()!r}'
        )

        # Step 6: second release_warm_lane — must RETAIN task/D (acceptance 1+2)
        await git_ops.release_warm_lane(lane, 'D')

        rc_verify3, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/D'], cwd=git_repo,
        )
        assert rc_verify3 == 0, (
            'task/D must be RETAINED after second release_warm_lane — α must '
            'protect the real rebased commits '
            '(RED today: α sees 0-beyond-main on stale ref and deletes task/D)'
        )

        _, count_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/D'], cwd=git_repo,
        )
        count = int(count_raw.strip())
        assert count > 0, (
            f'task/D must carry commits beyond main after second release (α retained it), '
            f'got count={count}'
        )


# ===========================================================================
# Task-1923 step-3: RED — _reuse_warm_lane on a DETACHED lane rebinds ref
# ===========================================================================


@pytest.mark.asyncio
class TestReuseWarmLaneBranchAware:
    """_reuse_warm_lane called on a DETACHED lane must rebind refs/heads/task/R
    to the lane HEAD after commit()+rebase_onto_main.

    RED today: _reuse_warm_lane runs commit+rebase on the detached HEAD but
    never rebinds refs/heads/task/R — the ref stays at the pre-detach tip
    (which diverges from the rebased detached HEAD).
    GREEN after step-4 inserts rebind_branch_to_head inside _reuse_warm_lane.
    """

    def _warm_config(self) -> GitConfig:
        return GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )

    async def test_reuse_on_detached_lane_rebinds_ref_and_attaches(
        self, git_repo: Path,
    ):
        """_reuse_warm_lane on a detached lane:
        (a) returns WorktreeInfo (not an error)
        (b) refs/heads/task/R resolves (rc==0) AND equals the lane HEAD
        (c) lane is ON task/R (--abbrev-ref HEAD == 'task/R')
        (d) task/R carries commits beyond main (rebase preserved the WIP)
        (e) the uncommitted WIP file (added before _reuse_warm_lane) is reachable
            from task/R (commit() committed it during _reuse_warm_lane)
        """
        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        # Acquire a lane on task/R with an initial committed WIP commit (1 beyond main)
        info = await git_ops.acquire_warm_lane('R', start_ref)
        assert isinstance(info, WorktreeInfo), f'Acquire failed: {info!r}'
        lane = info.path

        (lane / 'initial_wip.txt').write_text('initial work\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'initial wip commit'], cwd=lane)

        # Advance main with an unrelated commit (so rebase_onto_main has work to do)
        (git_repo / 'main_advance.txt').write_text('main advance\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'advance main'], cwd=git_repo)

        # Remember pre-detach tip of task/R (this is the STALE ref after detach)
        _, pre_detach_sha_raw, _ = await _run(
            ['git', 'rev-parse', 'refs/heads/task/R'], cwd=git_repo,
        )
        pre_detach_sha = pre_detach_sha_raw.strip()

        # Detach the lane (mirrors what release_warm_lane does)
        rc, _, err = await _run(['git', 'checkout', '--detach'], cwd=lane)
        assert rc == 0, f'checkout --detach failed: {err}'

        # Confirm detached AND that task/R ref still points at pre-detach tip
        _, abbrev_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert abbrev_raw.strip() == 'HEAD', (
            f'Lane should be DETACHED before _reuse_warm_lane, got {abbrev_raw.strip()!r}'
        )
        _, ref_before_raw, _ = await _run(
            ['git', 'rev-parse', 'refs/heads/task/R'], cwd=git_repo,
        )
        assert ref_before_raw.strip() == pre_detach_sha, (
            'refs/heads/task/R should still point at pre-detach tip before _reuse_warm_lane'
        )

        # Add uncommitted WIP on the detached lane (to be committed by _reuse_warm_lane)
        (lane / 'new_wip.txt').write_text('new work on detached lane\n')

        # === THE CALL UNDER TEST ===
        result = await git_ops._reuse_warm_lane(lane, 'task/R')

        # (a) Returns WorktreeInfo
        assert isinstance(result, WorktreeInfo), (
            f'_reuse_warm_lane must return WorktreeInfo on a detached lane, got {result!r}'
        )

        # (b) refs/heads/task/R resolves and equals lane HEAD
        rc_verify, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'refs/heads/task/R'], cwd=git_repo,
        )
        assert rc_verify == 0, (
            'refs/heads/task/R must resolve after _reuse_warm_lane on a detached lane '
            '(RED today: ref stays at stale pre-detach tip which was deleted by α)'
        )

        _, lane_head_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
        lane_head = lane_head_raw.strip()

        _, ref_after_raw, _ = await _run(
            ['git', 'rev-parse', 'refs/heads/task/R'], cwd=git_repo,
        )
        assert ref_after_raw.strip() == lane_head, (
            f'refs/heads/task/R ({ref_after_raw.strip()[:8]}) must equal lane HEAD '
            f'({lane_head[:8]}) after _reuse_warm_lane '
            f'(RED today: ref drifts from rebased detached HEAD)'
        )

        # (c) Lane is ON task/R (attached, not detached)
        _, attached_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert attached_raw.strip() == 'task/R', (
            f'Lane must be ON task/R after _reuse_warm_lane, got {attached_raw.strip()!r}'
        )

        # (d) task/R carries commits beyond main
        _, count_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/R'], cwd=git_repo,
        )
        count = int(count_raw.strip())
        assert count > 0, (
            f'task/R must carry commits beyond main after _reuse_warm_lane, got {count}'
        )

        # (e) New WIP file is reachable from task/R (was committed during _reuse)
        rc_show, _, _ = await _run(
            ['git', 'show', 'task/R:new_wip.txt'], cwd=git_repo,
        )
        assert rc_show == 0, (
            'new_wip.txt must be reachable from task/R after _reuse_warm_lane '
            '(commit() in _reuse_warm_lane committed the uncommitted WIP file)'
        )


# ===========================================================================
# Step-3 (1933): RED — git_ops reclaim-on-exhaustion integration tests
# ===========================================================================


@pytest.mark.asyncio
class TestAcquireWarmLaneReclaimOnExhaustion:
    """Integration tests for the reclaim-on-exhaustion safety valve (task 1933).

    warm_lane_pool_size=1 so one task exhausts the pool; then acquiring for a
    second task hits the reclaim path.

    All tests RED today: warm_lane_reclaim_candidate_provider and
    warm_lane_dispatched_predicate attributes do not exist; _try_reclaim_lane_for
    does not exist; acquire_warm_lane returns EXHAUSTED unconditionally when
    all lanes are ASSIGNED.
    """

    def _warm_config(self) -> GitConfig:
        return GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )

    async def _setup_exhausted_pool(
        self, git_repo: Path
    ) -> tuple[GitOps, Path, str]:
        """Set up a size-1 pool with V already occupying _lane-0.

        Returns (git_ops, lane_path, start_ref).
        """
        await _add_warm_lane_scripts(git_repo)
        git_ops = GitOps(self._warm_config(), git_repo, warm_lane_pool_size=1)

        # Get start_ref (main branch SHA)
        _, sha_raw, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_repo)
        start_ref = sha_raw.strip()

        # Acquire _lane-0 for victim 'V' → pool exhausted
        v_result = await git_ops.acquire_warm_lane('V', start_ref)
        assert isinstance(v_result, WorktreeInfo), (
            f'setup: expected WorktreeInfo for V, got {v_result!r}'
        )
        lane = v_result.path
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.assignment_for('V') == lane

        return git_ops, lane, start_ref

    async def test_reclaim_warm_returns_worktree_info(self, git_repo: Path, caplog):
        """(a) With callbacks wired: acquire_warm_lane returns WorktreeInfo, not EXHAUSTED."""
        git_ops, lane, start_ref = await self._setup_exhausted_pool(git_repo)

        # Wire reclaim callbacks: provider returns all candidates as non-terminal;
        # predicate never marks anything as dispatched.
        async def _provider(c):
            return set(c)
        git_ops.warm_lane_reclaim_candidate_provider = _provider
        git_ops.warm_lane_dispatched_predicate = lambda b: False

        pool = git_ops.warm_lane_pool
        assert pool is not None

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'With reclaim callbacks wired, acquire_warm_lane must return WorktreeInfo '
            f'(not EXHAUSTED); got {result!r}'
        )
        assert result.path == lane, (
            f'Reclaimed lane path must be _lane-0 ({lane}), got {result.path}'
        )

        # Lane HEAD must be on task/Z
        _, branch_raw, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        assert branch_raw.strip() == 'task/Z', (
            f'Lane HEAD must be on task/Z after reclaim, got {branch_raw.strip()!r}'
        )

        # Assignment map: V dropped, Z → _lane-0
        assert pool.assignment_for('V') is None, (
            'V assignment must be dropped after reclaim'
        )
        assert pool.assignment_for('Z') == lane, (
            'Z must be assigned to _lane-0 after reclaim'
        )

        # The steal must be logged at WARNING — this is the only ops signal
        # that the safety valve fired (= real pool pressure); a regression
        # that silently downgrades or drops it should fail this test.
        assert any(
            'reclaim-on-exhaustion — stole lane' in r.message
            for r in caplog.records
        ), (
            f'Expected a WARNING logging the reclaim steal; got: '
            f'{[r.getMessage() for r in caplog.records]}'
        )

    async def test_never_steal_dispatched(self, git_repo: Path):
        """(b) Dispatched victim is never stolen → EXHAUSTED."""
        from orchestrator.git_ops import WarmLaneUnavailable

        git_ops, lane, start_ref = await self._setup_exhausted_pool(git_repo)

        # predicate says V is dispatched → reclaim must be skipped
        async def _provider(c):
            return set(c)
        git_ops.warm_lane_reclaim_candidate_provider = _provider
        git_ops.warm_lane_dispatched_predicate = lambda b: b == 'V'

        pool = git_ops.warm_lane_pool
        assert pool is not None

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert result is WarmLaneUnavailable.EXHAUSTED, (
            f'Dispatched victim must never be stolen → EXHAUSTED, got {result!r}'
        )
        # V's lane must be untouched
        assert pool.assignment_for('V') == lane, (
            'V assignment must be untouched when victim is dispatched'
        )

    async def test_not_wired_byte_identical(self, git_repo: Path):
        """(c) With both callbacks None (default), acquire_warm_lane returns EXHAUSTED."""
        from orchestrator.git_ops import WarmLaneUnavailable

        git_ops, lane, start_ref = await self._setup_exhausted_pool(git_repo)

        # Both callbacks are None (the default, not wired)
        assert git_ops.warm_lane_reclaim_candidate_provider is None  # type: ignore[attr-defined]
        assert git_ops.warm_lane_dispatched_predicate is None  # type: ignore[attr-defined]

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert result is WarmLaneUnavailable.EXHAUSTED, (
            f'Unwired (both callbacks None) must return EXHAUSTED (byte-identical), got {result!r}'
        )

    async def test_wip_preserved_on_victim_branch(self, git_repo: Path):
        """(d) Uncommitted WIP in victim lane is committed before reset; task/V ref survives.

        What this actually exercises: ``git checkout -f -B task/Z`` on the
        stolen lane only repoints/creates the branch the lane checks out next
        — it never deletes the ``task/V`` ref, so task/V survives the steal
        regardless of any separate retention/GC policy. The property under
        test is that the pre-reset ``commit()`` call lands the victim's WIP
        onto that still-intact ref *before* the lane is repointed, so a
        resumed victim can recover it via the reattach path. This test does
        not drive a branch-deletion/GC path, so it is not itself a guard on
        1912 branch-retention behaviour.
        """
        git_ops, lane, start_ref = await self._setup_exhausted_pool(git_repo)

        # Leave uncommitted tracked WIP on V's lane
        wip_file = lane / 'victim_wip.txt'
        wip_file.write_text('uncommitted WIP from task V\n')
        await _run(['git', 'add', 'victim_wip.txt'], cwd=lane)
        # Deliberately do NOT commit — the reclaim should commit it

        # Wire reclaim callbacks
        async def _provider(c):
            return set(c)
        git_ops.warm_lane_reclaim_candidate_provider = _provider
        git_ops.warm_lane_dispatched_predicate = lambda b: False

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'With WIP-preservation reclaim, must return WorktreeInfo; got {result!r}'
        )

        # task/V branch ref must still exist: `checkout -f -B task/Z` only
        # repoints the lane's newly-checked-out branch — it never deletes
        # task/V, so the ref survives independent of any retention/GC policy.
        rc_verify, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'refs/heads/task/V'], cwd=git_repo,
        )
        assert rc_verify == 0, (
            'task/V branch ref must survive reclaim '
            '(checkout -f -B task/Z repoints the lane, it does not delete '
            'task/V; this test does not exercise branch GC/retention)'
        )

        # task/V must carry at least 1 commit beyond main — this is what
        # actually proves the pre-reset commit() call landed the WIP onto
        # task/V before the lane was repointed, not merely that the ref
        # happens to still exist.
        _, count_raw, _ = await _run(
            ['git', 'rev-list', '--count', 'main..task/V'], cwd=git_repo,
        )
        wip_commits = int(count_raw.strip())
        assert wip_commits >= 1, (
            f'task/V must carry at least 1 WIP commit beyond main, got {wip_commits} '
            '(commit-before-reset must have saved the uncommitted changes onto '
            'task/V prior to the lane being repointed to task/Z)'
        )

    async def test_empty_eligible_set_returns_exhausted(self, git_repo: Path):
        """(e) Provider returns set() → _try_reclaim_lane_for returns None → EXHAUSTED."""
        from orchestrator.git_ops import WarmLaneUnavailable

        git_ops, lane, start_ref = await self._setup_exhausted_pool(git_repo)

        # Provider returns empty set — no eligible victims even though pool is exhausted
        async def _empty_provider(c):
            return set()
        git_ops.warm_lane_reclaim_candidate_provider = _empty_provider
        git_ops.warm_lane_dispatched_predicate = lambda b: False

        pool = git_ops.warm_lane_pool
        assert pool is not None

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert result is WarmLaneUnavailable.EXHAUSTED, (
            f'Empty eligible set must yield EXHAUSTED (no victim to steal), got {result!r}'
        )
        # V's assignment must be untouched
        assert pool.assignment_for('V') == lane, (
            'V assignment must be untouched when provider returns empty eligible set'
        )

    async def test_commit_failure_does_not_block_reclaim(self, git_repo: Path, monkeypatch):
        """(f) If commit() raises during WIP-save, reclaim proceeds and returns WorktreeInfo."""
        git_ops, lane, start_ref = await self._setup_exhausted_pool(git_repo)

        # Force commit() to raise to exercise the except branch in _try_reclaim_lane_for
        async def _failing_commit(worktree: 'Path', message: str) -> None:
            raise RuntimeError('simulated commit failure')
        monkeypatch.setattr(git_ops, 'commit', _failing_commit)

        async def _provider(c):
            return set(c)
        git_ops.warm_lane_reclaim_candidate_provider = _provider
        git_ops.warm_lane_dispatched_predicate = lambda b: False

        result = await git_ops.acquire_warm_lane('Z', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'commit() failure must not block reclaim — expect WorktreeInfo, got {result!r}'
        )
        assert result.path == lane, (
            f'Reclaimed lane path must be _lane-0 ({lane}) even after commit failure'
        )
