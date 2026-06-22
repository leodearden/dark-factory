"""Tests for ε warm-lane disk-guard admission control.

Each step group is labelled RED/GREEN to track TDD phase:

  Step 1 (RED): _warm_lane_disk_admission_blocked() — method and config knobs absent.
  Step 3 (RED): acquire_warm_lane() guard wiring — not yet wired.
  Step 5 (RED): CLI contract (threshold flags) and e2e create_worktree →
                WarmLaneDiskPressure — threshold args not yet forwarded.

Tests that are currently RED fail because the feature does not exist yet.
They turn GREEN in the corresponding implementation step.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    GitOps,
    WarmLaneDiskPressure,
    WarmLaneUnavailable,
    WorktreeInfo,
    _run,
)


# ---------------------------------------------------------------------------
# Repo fixture (mirrors test_git_ops.py pattern)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


# ---------------------------------------------------------------------------
# Disk-guard stub script helpers
# ---------------------------------------------------------------------------


async def _add_disk_guard_scripts(repo: Path) -> None:
    """Commit stub warm-lane-disk-guard.sh and warm-lane-gc.sh into repo/scripts/.

    guard stub (warm-lane-disk-guard.sh):
      - Appends "check <argv>" to <repo>/.test_disk_call_log.
      - Reads the first exit code from <repo>/.test_disk_check_exits (one per line),
        pops it, and exits with that code.  Exits 0 if the file is absent or empty.

    gc stub (warm-lane-gc.sh):
      - Appends "reclaim <argv>" to <repo>/.test_disk_call_log.
      - Exits 0.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)

    guard = scripts_dir / 'warm-lane-disk-guard.sh'
    guard.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'ROOT="$(dirname "$DIR")"\n'
        '# Append subcommand + full argv to call log\n'
        'echo "check $*" >> "$ROOT/.test_disk_call_log"\n'
        '# Pop first exit code from sequence file\n'
        'if [ -f "$ROOT/.test_disk_check_exits" ] && [ -s "$ROOT/.test_disk_check_exits" ]; then\n'
        '    rc=$(head -1 "$ROOT/.test_disk_check_exits")\n'
        '    tmpf="${ROOT}/.test_disk_check_exits.tmp"\n'
        '    tail -n +2 "$ROOT/.test_disk_check_exits" > "$tmpf"\n'
        '    mv "$tmpf" "$ROOT/.test_disk_check_exits"\n'
        'else\n'
        '    rc=0\n'
        'fi\n'
        'exit "${rc:-0}"\n'
    )
    guard.chmod(0o755)

    gc = scripts_dir / 'warm-lane-gc.sh'
    gc.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'ROOT="$(dirname "$DIR")"\n'
        '# Append subcommand + full argv to call log\n'
        'echo "reclaim $*" >> "$ROOT/.test_disk_call_log"\n'
        'exit 0\n'
    )
    gc.chmod(0o755)

    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add disk-guard stub scripts'], cwd=repo)


def _write_check_exits(repo: Path, exit_codes: list[int]) -> None:
    """Write a guard-script exit-code sequence to <repo>/.test_disk_check_exits."""
    (repo / '.test_disk_check_exits').write_text(
        '\n'.join(str(c) for c in exit_codes) + '\n'
    )


def _read_call_log(repo: Path) -> list[str]:
    """Return all non-empty lines from <repo>/.test_disk_call_log."""
    log = repo / '.test_disk_call_log'
    if not log.exists():
        return []
    return [line for line in log.read_text().splitlines() if line.strip()]


def _subcommands(repo: Path) -> list[str]:
    """Return just the first word of each call-log entry (subcommand names)."""
    return [line.split()[0] for line in _read_call_log(repo)]


def _make_disk_guard_config(**overrides: object) -> GitConfig:
    """Build a GitConfig with disk guard enabled and canonical warm-lane settings."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_disk_guard=True,
        warm_lane_min_free_gib=50,
        warm_lane_min_free_inodes=500_000,
        **overrides,
    )


# ===========================================================================
# Step 1 (RED): _warm_lane_disk_admission_blocked()
# ===========================================================================


@pytest.mark.asyncio
class TestWarmLaneDiskAdmissionBlocked:
    """Unit tests for GitOps._warm_lane_disk_admission_blocked().

    RED today — the method and config knobs (warm_lane_disk_guard,
    warm_lane_min_free_gib, warm_lane_min_free_inodes) do not yet exist.
    Turns GREEN in step 2.
    """

    async def test_still_pressured_returns_true(
        self, git_repo: Path,
    ):
        """check→75, reclaim, recheck→75 ⇒ True; call-log is check→reclaim→check."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_disk_admission_blocked()

        assert result is True, (
            'Still-pressured after reclaim should block admission (True)'
        )
        assert _subcommands(git_repo) == ['check', 'reclaim', 'check'], (
            f'Expected check→reclaim→check; got {_subcommands(git_repo)}'
        )

    async def test_reclaim_recovers_returns_false(
        self, git_repo: Path,
    ):
        """check→75, reclaim clears pressure, recheck→0 ⇒ False; reclaim WAS invoked."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [75, 0])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_disk_admission_blocked()

        assert result is False, (
            'Reclaim recovered disk pressure — should admit (False)'
        )
        assert _subcommands(git_repo) == ['check', 'reclaim', 'check'], (
            f'Expected check→reclaim→check even when reclaim succeeds; got {_subcommands(git_repo)}'
        )

    async def test_healthy_fast_path_no_reclaim(
        self, git_repo: Path,
    ):
        """check→0 ⇒ False; reclaim NOT invoked (call-log has exactly ['check'])."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [0])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_disk_admission_blocked()

        assert result is False, 'Healthy disk → False (admit)'
        assert _subcommands(git_repo) == ['check'], (
            f'Reclaim must NOT run when healthy; log={_subcommands(git_repo)}'
        )

    async def test_absent_guard_script_fail_open(
        self, git_repo: Path,
    ):
        """Guard script absent (rc 127) ⇒ False (fail-open); reclaim NOT invoked."""
        # No scripts committed to repo — guard is absent (rc 127)
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_disk_admission_blocked()

        assert result is False, (
            'Absent guard script must fail-open (False) — byte-identical to today'
        )
        assert _read_call_log(git_repo) == [], (
            'No call-log entries expected when guard script is absent'
        )


# ===========================================================================
# Step 3 (RED): acquire_warm_lane() consults the disk-guard before allocating
# ===========================================================================


async def _add_all_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed, debug-port, disk-guard, and gc scripts into repo/scripts/.

    Combines _add_warm_lane_scripts (seed + debug-port) with
    _add_disk_guard_scripts (warm-lane-disk-guard.sh + warm-lane-gc.sh) so that
    tests can exercise both the acquire path and the disk-guard admission gate.
    Commits all four scripts in a single commit.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Seed stub (mirrors test_git_ops._add_warm_lane_scripts)
    seed = scripts_dir / 'seed-warm-lane.sh'
    seed.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed.chmod(0o755)

    # Debug-port stub
    debug = scripts_dir / 'setup-worktree-debug-port.sh'
    debug.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug.chmod(0o755)

    # Disk-guard stub (same as _add_disk_guard_scripts)
    guard = scripts_dir / 'warm-lane-disk-guard.sh'
    guard.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'ROOT="$(dirname "$DIR")"\n'
        'echo "check $*" >> "$ROOT/.test_disk_call_log"\n'
        'if [ -f "$ROOT/.test_disk_check_exits" ] && [ -s "$ROOT/.test_disk_check_exits" ]; then\n'
        '    rc=$(head -1 "$ROOT/.test_disk_check_exits")\n'
        '    tmpf="${ROOT}/.test_disk_check_exits.tmp"\n'
        '    tail -n +2 "$ROOT/.test_disk_check_exits" > "$tmpf"\n'
        '    mv "$tmpf" "$ROOT/.test_disk_check_exits"\n'
        'else\n'
        '    rc=0\n'
        'fi\n'
        'exit "${rc:-0}"\n'
    )
    guard.chmod(0o755)

    # GC stub
    gc = scripts_dir / 'warm-lane-gc.sh'
    gc.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'ROOT="$(dirname "$DIR")"\n'
        'echo "reclaim $*" >> "$ROOT/.test_disk_call_log"\n'
        'exit 0\n'
    )
    gc.chmod(0o755)

    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add all warm-lane stub scripts'], cwd=repo)


async def _get_head(repo: Path) -> str:
    """Return the HEAD commit SHA of the repo."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


@pytest.mark.asyncio
class TestAcquireWarmLaneDiskGuard:
    """Integration: acquire_warm_lane() is gated by _warm_lane_disk_admission_blocked().

    RED today — the guard is not wired into acquire_warm_lane.
    Turns GREEN in step 4.
    """

    async def test_still_pressured_returns_disk_pressure(
        self, git_repo: Path,
    ):
        """check→75, reclaim, recheck→75 ⇒ DISK_PRESSURE; pool lane stays FREE."""
        from orchestrator.warm_lane_pool import LaneState

        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        result = await git_ops.acquire_warm_lane('A', start_ref)

        assert result is WarmLaneUnavailable.DISK_PRESSURE, (
            f'Expected DISK_PRESSURE; got {result!r}'
        )
        # Pool lane must remain FREE — guard runs before acquire_for()
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(
            git_ops.worktree_base / '_lane-0'
        ) == LaneState.FREE, 'Lane-0 must stay FREE when admission is blocked'

    async def test_still_pressured_lane_dir_not_created(
        self, git_repo: Path,
    ):
        """DISK_PRESSURE path: no worktree-add or seed ran (lane dir absent)."""
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        await git_ops.acquire_warm_lane('A', start_ref)

        assert not (git_ops.worktree_base / '_lane-0').exists(), (
            'Lane dir must not be created when disk admission is blocked'
        )

    async def test_still_pressured_call_order(
        self, git_repo: Path,
    ):
        """DISK_PRESSURE path: guard ran check→reclaim→check before any acquire."""
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        await git_ops.acquire_warm_lane('A', start_ref)

        assert _subcommands(git_repo) == ['check', 'reclaim', 'check'], (
            f'Expected check→reclaim→check; got {_subcommands(git_repo)}'
        )

    async def test_knob_off_guard_not_invoked(
        self, git_repo: Path,
    ):
        """warm_lane_disk_guard=False → guard scripts never run; acquire returns WorktreeInfo."""
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])  # Would block if guard ran
        config = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
            warm_lane_disk_guard=False,  # master knob OFF → byte-identical
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        result = await git_ops.acquire_warm_lane('A', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'Knob-off must return WorktreeInfo; got {result!r}'
        )
        assert result.path == git_ops.worktree_base / '_lane-0', (
            f'Expected _lane-0; got {result.path}'
        )
        assert _read_call_log(git_repo) == [], (
            'Guard scripts must NOT run when warm_lane_disk_guard=False'
        )


# ===========================================================================
# Step 5 (RED): e2e create_worktree → WarmLaneDiskPressure + CLI contract
# ===========================================================================


@pytest.mark.asyncio
class TestWarmLaneDiskGuardE2E:
    """E2E and CLI-contract tests for the ε warm-lane disk-guard admission path.

    The e2e tests (create_worktree raises WarmLaneDiskPressure, seed never
    invoked) turn GREEN in step 4 because the guard is wired.

    The CLI-contract threshold-flag assertions (check receives --min-free-gib
    and --min-free-inodes) are RED today because step-2 _run_warm_lane_disk_guard
    only passes --mount.  They turn GREEN in step 6.
    """

    async def test_create_worktree_raises_warm_lane_disk_pressure(
        self, git_repo: Path,
    ):
        """Still-pressured guard: create_worktree('A') raises WarmLaneDiskPressure."""
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('A')

    async def test_disk_pressure_seed_never_invoked(
        self, git_repo: Path,
    ):
        """WarmLaneDiskPressure path: seed script never ran (no seeded.bin, lane FREE)."""
        from orchestrator.warm_lane_pool import LaneState

        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('A')

        lane = git_ops.worktree_base / '_lane-0'
        # Seed never ran — no seeded.bin artifact
        assert not (lane / 'target' / 'seeded.bin').exists(), (
            'seeded.bin must not exist — seed must not run before disk-pressure blocks'
        )
        # Pool lane must stay FREE (never acquired)
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(lane) == LaneState.FREE, (
            'Lane must remain FREE — acquire_for must not run before disk-pressure blocks'
        )

    async def test_cli_contract_check_receives_mount_and_thresholds(
        self, git_repo: Path,
    ):
        """CLI contract: check received --mount, --min-free-gib, --min-free-inodes.

        RED today — step-2 _run_warm_lane_disk_guard passes only --mount.
        Turns GREEN in step 6 when threshold flags are appended.
        """
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        # _make_disk_guard_config defaults: min_free_gib=50, min_free_inodes=500_000
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('A')

        # First and third call-log entries are 'check' invocations
        check_lines = [
            line for line in _read_call_log(git_repo)
            if line.split()[0] == 'check'
        ]
        assert len(check_lines) >= 1, 'Expected at least one check invocation'
        first_check = check_lines[0]

        # Verify --mount flag is present with the worktree_base path
        assert '--mount' in first_check, (
            f'check must receive --mount; got: {first_check!r}'
        )
        mount_idx = first_check.split().index('--mount')
        assert first_check.split()[mount_idx + 1] == str(git_ops.worktree_base), (
            f'--mount value must be worktree_base={git_ops.worktree_base}; '
            f'got {first_check.split()[mount_idx + 1]!r}'
        )

        # Verify threshold flags — RED today (step-6 adds them)
        assert '--min-free-gib' in first_check, (
            f'check must receive --min-free-gib; got: {first_check!r}'
        )
        assert '--min-free-inodes' in first_check, (
            f'check must receive --min-free-inodes; got: {first_check!r}'
        )
        parts = first_check.split()
        gib_val = parts[parts.index('--min-free-gib') + 1]
        inodes_val = parts[parts.index('--min-free-inodes') + 1]
        assert gib_val == '50', f'--min-free-gib must be 50; got {gib_val!r}'
        assert inodes_val == '500000', f'--min-free-inodes must be 500000; got {inodes_val!r}'

    async def test_cli_contract_gc_receives_mount(
        self, git_repo: Path,
    ):
        """CLI contract: gc reclaim stub received 'reclaim --mount <worktree_base>'."""
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        config = _make_disk_guard_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneDiskPressure):
            await git_ops.create_worktree('A')

        reclaim_lines = [
            line for line in _read_call_log(git_repo)
            if line.split()[0] == 'reclaim'
        ]
        assert len(reclaim_lines) == 1, (
            f'Expected exactly one reclaim invocation; got {reclaim_lines}'
        )
        reclaim_line = reclaim_lines[0]
        assert '--mount' in reclaim_line, (
            f'reclaim must receive --mount; got: {reclaim_line!r}'
        )
        parts = reclaim_line.split()
        mount_idx = parts.index('--mount')
        assert parts[mount_idx + 1] == str(git_ops.worktree_base), (
            f'reclaim --mount must be worktree_base; got {parts[mount_idx + 1]!r}'
        )
