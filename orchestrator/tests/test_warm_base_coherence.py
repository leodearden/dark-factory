"""Integration tests for warm-lane base coherence contract (D8/D10).

DF consumer side of the reify activation-seam R1:
- D8: _seed_warm_lane resolves a gen-dir symlink base to the concrete .gen.N dir
      and holds flock -s on .gen.N.lock during the cp.
- D10: refresh_warm_base passes --landed-commit <sha> to satisfy reify's inv.9 guard.

This file is self-contained (no shared conftest fixtures beyond tmp_path).
Scaffolding patterns mirror test_warm_lane_pool.py.
"""

from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Scaffolding helpers (module-local, mirrors test_warm_lane_pool.py patterns)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    """Initialise a bare git repo suitable for worktree tests."""
    await _run(['git', 'init', '-b', 'main', str(repo)])
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _make_lane_dir(repo: Path, lane_name: str = '_lane-0') -> Path:
    """Create a real worktree at <repo>/.worktrees/<lane_name>."""
    worktree_base = repo / '.worktrees'
    worktree_base.mkdir(parents=True, exist_ok=True)
    lane = worktree_base / lane_name
    await _run(
        ['git', 'worktree', 'add', '-b', f'task/{lane_name}', str(lane), 'HEAD'],
        cwd=repo,
    )
    return lane


def _make_gendir_base(base_dir: Path) -> tuple[Path, Path, Path]:
    """Create a reify-style gen-dir base layout.

    Creates:
        <base_dir>/.gen.0/          — the concrete gen dir
        <base_dir>/.gen.0/sentinel  — sentinel content file
        <base_dir>/target           — symlink -> .gen.0 (relative sibling)
        <base_dir>/.gen.0.lock      — reader-refcount lock file (reify GC protocol)

    Returns:
        (base_dir, gen_dir, target_symlink)
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = base_dir / '.gen.0'
    gen_dir.mkdir()
    (gen_dir / 'sentinel').write_text('gen-dir-content\n')
    target_symlink = base_dir / 'target'
    # Relative sibling, matches reify's: ln -sfn .gen.N target
    target_symlink.symlink_to('.gen.0')
    (base_dir / '.gen.0.lock').touch()
    return base_dir, gen_dir, target_symlink


def _wl_config_with_base(base_target_dir: str) -> GitConfig:
    """GitConfig with warm_lane_pool=True and warm_lane_base_target_dir set."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_base_target_dir=base_target_dir,
    )


def _install_recording_seed_stub(lane: Path, marker: Path) -> None:
    """Install a seed-warm-lane.sh that records its argv to *marker* and exits 0."""
    scripts_dir = lane / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'seed-warm-lane.sh'
    script.write_text(f'#!/usr/bin/env bash\necho "$@" >> {marker}\n')
    script.chmod(0o755)


# ===========================================================================
# Step 1 (RED): seed resolves gen-dir symlink / back-compat plain dir
# ===========================================================================


@pytest.mark.asyncio
class TestSeedSymlinkResolution:
    """D8 — _seed_warm_lane resolves gen-dir symlink base to concrete .gen.N dir."""

    async def test_seed_resolves_symlink_to_gen_dir(self, tmp_path: Path) -> None:
        """When warm_lane_base_target_path is a ``target``→``.gen.N`` symlink,
        _seed_warm_lane must pass the RESOLVED .gen.N directory path to the
        script, NOT the symlink path.

        RED against today's impl which passes str(warm_lane_base_target_path) raw.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        # Build a reify-style gen-dir base: .gen.0/ + sentinel + target->.gen.0 + .gen.0.lock
        base_dir = tmp_path / 'bases'
        _, gen_dir, target_symlink = _make_gendir_base(base_dir)

        # Point warm_lane_base_target_dir at the symlink path (as reify sets it)
        config = _wl_config_with_base(str(target_symlink))
        git_ops = GitOps(config, repo, warm_lane_pool_size=1)

        lane = await _make_lane_dir(repo, '_lane-0')
        marker = lane / 'seed_args.txt'
        _install_recording_seed_stub(lane, marker)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert result is True
        assert marker.exists(), 'seed-warm-lane.sh was not called'
        # First arg recorded is base_target
        recorded_base_target = marker.read_text().strip().split()[0]

        # Must be the RESOLVED .gen.0 path, NOT the target symlink
        assert recorded_base_target == str(gen_dir), (
            f'Expected resolved gen dir {gen_dir!r}, '
            f'got {recorded_base_target!r}'
        )
        assert recorded_base_target != str(target_symlink), (
            'base_target must NOT be the symlink path'
        )

    async def test_seed_plaindir_base_passes_raw_path(self, tmp_path: Path) -> None:
        """Back-compat: plain directory (not a symlink) is passed raw, unchanged.

        Existing default config (base == _merge-verify/target plain dir) must keep
        working unchanged. GREEN against today's impl and must stay green after step-2.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        # Plain directory — no symlink
        plain_dir = tmp_path / 'plain_base'
        plain_dir.mkdir()

        config = _wl_config_with_base(str(plain_dir))
        git_ops = GitOps(config, repo, warm_lane_pool_size=1)

        lane = await _make_lane_dir(repo, '_lane-0')
        marker = lane / 'seed_args.txt'
        _install_recording_seed_stub(lane, marker)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert result is True
        assert marker.exists()
        recorded_base_target = marker.read_text().strip().split()[0]

        # Must pass the RAW plain dir path (no resolution)
        assert recorded_base_target == str(plain_dir), (
            f'Expected raw path {plain_dir!r}, got {recorded_base_target!r}'
        )


# ===========================================================================
# Step 3 (RED): seed holds flock -s on the gen lock during the cp
# ===========================================================================


@pytest.mark.asyncio
class TestSeedFlockLock:
    """D8 — _seed_warm_lane holds flock -s on .gen.N.lock for the duration of the cp."""

    async def test_seed_holds_shared_flock_during_cp(self, tmp_path: Path) -> None:
        """While seed-warm-lane.sh is running, _seed_warm_lane must hold a shared lock
        on <base>/.gen.0.lock.  A concurrent non-blocking exclusive lock attempt from
        within the script must FAIL (i.e. the probe reports BLOCKED), proving the
        reader-refcount GC lock is held.  After the script exits the lane target must
        contain the gen dir's sentinel content (warm, not torn/cold).

        RED against today's impl: no flock is held, so the concurrent exclusive probe
        SUCCEEDS (reports FREE).
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        # Build reify-style gen-dir base
        base_dir = tmp_path / 'bases'
        _, gen_dir, target_symlink = _make_gendir_base(base_dir)
        lock_path = base_dir / '.gen.0.lock'

        config = _wl_config_with_base(str(target_symlink))
        git_ops = GitOps(config, repo, warm_lane_pool_size=1)

        lane = await _make_lane_dir(repo, '_lane-0')
        (lane / 'target').mkdir(exist_ok=True)

        # Install a stub seed-warm-lane.sh that:
        # (a) records its $1 (base_target) arg
        # (b) probes the lock with a non-blocking exclusive flock attempt from a subshell
        #     — records BLOCKED if it fails, FREE if it succeeds
        # (c) emulates a CoW seed: copies the gen-dir contents into lane/target/
        lock_probe_marker = lane / 'lock_probe.txt'
        seed_args_marker = lane / 'seed_args.txt'
        scripts_dir = lane / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / 'seed-warm-lane.sh'
        script.write_text(
            f'#!/usr/bin/env bash\n'
            f'echo "$1" >> {seed_args_marker}\n'
            # Probe: try a non-blocking exclusive lock; record BLOCKED on failure, FREE on success
            f'if flock -n -x {lock_path} true 2>/dev/null; then\n'
            f'  echo FREE >> {lock_probe_marker}\n'
            f'else\n'
            f'  echo BLOCKED >> {lock_probe_marker}\n'
            f'fi\n'
            # Emulate CoW seed: copy gen dir contents into lane/target/
            f'cp -a "$1"/. "$2/target/"\n'
        )
        script.chmod(0o755)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert result is True, 'seed script must exit 0'
        assert lock_probe_marker.exists(), 'lock probe script section did not execute'
        probe_result = lock_probe_marker.read_text().strip()

        # The concurrent exclusive probe must have been BLOCKED (shared lock held by caller)
        assert probe_result == 'BLOCKED', (
            f'Expected lock probe BLOCKED (flock -s held during cp), got {probe_result!r}. '
            'The reader-refcount GC lock was NOT held — concurrent base rewrite could '
            'ENOENT the clone mid-walk (torn read).'
        )

        # The lane target must be warm (gen dir content copied, not torn/cold)
        sentinel = lane / 'target' / 'sentinel'
        assert sentinel.exists(), (
            f'Lane target sentinel not found at {sentinel} — lane is cold/torn'
        )
        assert sentinel.read_text() == 'gen-dir-content\n', (
            f'Sentinel content mismatch: {sentinel.read_text()!r}'
        )


# ===========================================================================
# Step 5 (RED): refresh passes --landed-commit + inv.9 two-way seam
# ===========================================================================


def _setup_merge_verify_for_refresh(
    repo: Path,
    rolling_base: Path,
) -> tuple[Path, GitConfig]:
    """Create _merge-verify (inside a git repo dir) and a separate rolling base.

    Returns (merge_verify path, GitConfig pointing base at rolling_base).
    The merge_verify dir is inside *repo* so git rev-parse HEAD succeeds.
    """
    worktree_base = repo / '.worktrees'
    merge_verify = worktree_base / '_merge-verify'
    merge_verify.mkdir(parents=True, exist_ok=True)
    (merge_verify / 'target').mkdir(exist_ok=True)
    rolling_base.mkdir(parents=True, exist_ok=True)
    config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_base_target_dir=str(rolling_base),
    )
    return merge_verify, config


def _install_all_args_recording_stub(scripts_dir_parent: Path, recorder: Path) -> None:
    """Install a refresh-warm-base.sh that records all argv ($@) and exits 0."""
    scripts_dir = scripts_dir_parent / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'refresh-warm-base.sh'
    script.write_text(f'#!/usr/bin/env bash\necho "$@" >> {recorder}\n')
    script.chmod(0o755)


def _install_inv9_guard_stub(scripts_dir_parent: Path) -> None:
    """Install a refresh-warm-base.sh emulating reify's inv.9 --landed-commit guard.

    Exits 0 iff ``--landed-commit <non-empty-value>`` is present in argv;
    otherwise writes a diagnostic to stderr and exits 1.
    """
    scripts_dir = scripts_dir_parent / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'refresh-warm-base.sh'
    script.write_text(
        '#!/usr/bin/env bash\n'
        '# Emulate reify inv.9: require --landed-commit <sha>\n'
        'found=0\n'
        'while [[ $# -gt 0 ]]; do\n'
        '  if [[ "$1" == "--landed-commit" ]] && [[ -n "$2" ]]; then\n'
        '    found=1\n'
        '  fi\n'
        '  shift\n'
        'done\n'
        'if [[ $found -eq 0 ]]; then\n'
        '  echo "inv.9 REJECT: --landed-commit not provided or empty" >&2\n'
        '  exit 1\n'
        'fi\n'
        'exit 0\n'
    )
    script.chmod(0o755)


@pytest.mark.asyncio
class TestRefreshLandedCommit:
    """D10 — refresh_warm_base passes --landed-commit <sha> to satisfy reify's inv.9."""

    async def test_refresh_passes_landed_commit(self, tmp_path: Path) -> None:
        """refresh_warm_base() must invoke the script with ``--landed-commit <sha>``
        appended after ``<advancing> <base>``, where <sha> == ``git rev-parse HEAD``
        of the _merge-verify directory.

        RED against today's 2-arg impl (no --landed-commit flag emitted).
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        rolling_base = tmp_path / 'rolling'
        merge_verify, config = _setup_merge_verify_for_refresh(repo, rolling_base)
        git_ops = GitOps(config, repo, warm_lane_pool_size=1)

        recorder = tmp_path / 'refresh_args.txt'
        _install_all_args_recording_stub(merge_verify, recorder)

        result = await git_ops.refresh_warm_base()

        assert result is True, 'refresh_warm_base must return True on success'
        assert recorder.exists(), 'refresh-warm-base.sh was not called'
        recorded = recorder.read_text().strip()

        # Derive the expected sha from _merge-verify (same as impl will derive)
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=merge_verify)
        assert rc == 0, f'Could not derive HEAD from _merge-verify: {head_out!r}'
        head = head_out.strip()

        advancing = str(git_ops.persistent_merge_worktree_path / 'target')
        base = str(git_ops.warm_lane_base_target_path)
        expected = f'{advancing} {base} --landed-commit {head}'
        assert recorded == expected, (
            f'Wrong args: got {recorded!r}, expected {expected!r}'
        )

    async def test_refresh_satisfies_inv9_guard(self, tmp_path: Path) -> None:
        """With ``--landed-commit`` passed, the reify inv.9-emulating guard accepts
        (exits 0) → refresh_warm_base() returns True.

        RED against today's no-flag impl (guard exits 1 → returns False).
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        rolling_base = tmp_path / 'rolling'
        merge_verify, config = _setup_merge_verify_for_refresh(repo, rolling_base)
        git_ops = GitOps(config, repo, warm_lane_pool_size=1)

        _install_inv9_guard_stub(merge_verify)

        result = await git_ops.refresh_warm_base()

        assert result is True, (
            'refresh_warm_base must return True when reify inv.9 guard accepts '
            '(requires --landed-commit with non-empty value)'
        )

    async def test_refresh_rejected_without_landed_commit(self, tmp_path: Path) -> None:
        """When HEAD derivation fails (non-repo persistent_merge_worktree path),
        no ``--landed-commit`` flag is appended, and the inv.9 guard rejects
        (exit 1) → refresh_warm_base returns False.

        Documents the reify-facing 'exits non-zero without --landed-commit' direction.
        """
        # Use a non-git directory as the repo base — git rev-parse HEAD will fail there
        non_repo = tmp_path / 'non_repo'
        non_repo.mkdir()
        rolling_base = tmp_path / 'rolling'
        rolling_base.mkdir()

        config = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
            warm_lane_base_target_dir=str(rolling_base),
        )
        git_ops = GitOps(config, non_repo, warm_lane_pool_size=1)

        # Create _merge-verify inside the non-repo dir (git rev-parse HEAD will fail)
        merge_verify = non_repo / '.worktrees' / '_merge-verify'
        merge_verify.mkdir(parents=True, exist_ok=True)
        (merge_verify / 'target').mkdir()

        _install_inv9_guard_stub(merge_verify)

        result = await git_ops.refresh_warm_base()

        assert result is False, (
            'refresh_warm_base must return False when HEAD derivation fails and '
            'the inv.9 guard rejects (no --landed-commit passed)'
        )
