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
